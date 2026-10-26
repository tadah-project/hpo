#pragma once
/* ----- ANNEAL_PA  (population annealing) ------------------------------
 *
 * Refs: Hukushima & Iba, AIP Conf. Proc. 690 (2003); Machta, PRE 82
 *       (2010); Wang, Machta & Katzgraber, PRE 92 (2015) for the
 *       adaptive-ESS schedule; Corana et al., ACM TOMS 13:262 (1987)
 *       for the per-walker adaptive stepper.
 *
 * A population of R walkers anneals from T_HOT down to T_MIN. At each
 * temperature step the population is reweighted by the Boltzmann
 * factor exp(-dbeta * f_i) and resampled (systematic resampling --
 * lowest variance, ONE rng draw, deterministic), so low-loss walkers
 * multiply and FAILSCORE walkers die out naturally (the cliff handling
 * this landscape family needs). Each walker then runs SWEEPS rungs of
 * the shared annealing stepper (anneal_core.h) at the new temperature.
 *
 * Schedule: ESS_TARGET in (0,1) (default 0.5) picks dbeta adaptively so
 * the effective sample size stays at target * N_feasible -- zero extra
 * evaluations, capped so T never drops faster than RT per step;
 * degenerate populations (all-equal f, all-FAILSCORE) fall back to the
 * geometric T *= RT step. ESS_TARGET 0 = pure geometric schedule.
 *
 * Resampling copies walker STATE (x, f, V): children inherit the
 * parent's step-vector calibration (V is walker state here, unlike PT
 * where it belongs to the temperature slot). Each SLOT keeps its own
 * RNG stream, so copies of one parent placed in different slots
 * decorrelate deterministically and results are identical for any
 * THREADS.
 *
 * Execution: THREADS is the TOTAL width of the shared worker pool (the
 * per-rank CPU-core budget) running the pc_lammps fan-out + MLIP
 * train/predict; it no longer multiplies OMP_NUM_THREADS. min(THREADS,
 * n_local) driver threads step the walkers but block on the shared pool
 * during each eval, so THREADS is the true concurrency. Note the default
 * POPULATION scales with THREADS (4*THREADS) -- fix POPULATION to hold R
 * constant while varying the core budget. OMP_NUM_THREADS independently
 * drives the residual OpenMP kernels (NN lists, data-stats), not the pool.
 *
 * PA-specific PARAMs (shared stepper knobs as in ANNEAL; the full key
 * registry lives in blueprint/anneal_keys.toml):
 *   POPULATION   4*THREADS (>=16)  population size R (top-level knob)
 *   SWEEPS       1                 rungs per walker per T-step
 *   ESS_TARGET   0.5               adaptive-dbeta target; 0 = geometric
 *   T_HOT        auto              starting temperature
 *   T_MIN        auto (T0*1e-6)    final temperature
 *   RT           0.90              max cooling per step / geometric ratio
 *   INIT_MODE    JITTER            JITTER: walker0=seed; others Gaussian
 *                                  jitter (sigma=CLASSIC scale or Corana V)
 *                                  UNIFORM: walkers>0 uniform in the box
 */
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/warmstart_cache.h>   // current_chain() bridge
#include <tadah/hpo/run_end.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_parallel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tadah::hpo {

#ifdef TADAH_HPO_MPI
namespace pa_detail {

/* PA uses two MPI payload shapes: a dim+2 calibration broadcast and a
 * population-wide gather/scatter with stride 2*dim+1.  Validate both without
 * allocating either payload, and keep every multiplication in a checked
 * domain before it can be narrowed to the MPI C API's int counts. */
inline void validate_mpi_calibration_dimension(std::size_t dim)
{
  const std::size_t mpi_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  constexpr std::size_t calibration_fields = 2;
  if (dim > mpi_limit - calibration_fields)
    throw std::length_error(
        "TADAH:ANNEAL_PA: parameter dimension " + std::to_string(dim) +
        " exceeds the MPI calibration-payload limit "
        "(dimension + 2 must fit in int)");
}

inline void validate_mpi_population_state(std::size_t dim,
                                          std::size_t population)
{
  const std::size_t mpi_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  const std::size_t size_limit = std::numeric_limits<std::size_t>::max();
  if (dim > (size_limit - 1) / 2)
    throw std::length_error(
        "TADAH:ANNEAL_PA: state-vector stride overflows size_t");
  const std::size_t stride = 2 * dim + 1;
  if (population > mpi_limit || stride > mpi_limit ||
      population > mpi_limit / stride)
    throw std::length_error(
        "TADAH:ANNEAL_PA: population state exceeds MPI int "
        "count/displacement limits");
}

inline void validate_mpi_payload_dimensions(std::size_t dim,
                                            std::size_t population)
{
  validate_mpi_calibration_dimension(dim);
  validate_mpi_population_state(dim, population);
}

}  // namespace pa_detail
#endif

template<typename Vec, typename HostT>
  void run_anneal_pa_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
    /* The SEED the journal names, and the seed of every stream of the run
     * (phase_stream_seed: itself but in a PRESTAGE refinement). */
    const std::uint64_t seed_value = sp.seed.value_or(std::random_device{}());
    const std::uint64_t seed = phase_stream_seed(seed_value, sp.stream_phase);

    const Knobs kb = Knobs::parse(sp, AlgoKind::PA);
    anneal::announce_knobs(kb, AlgoKind::PA);

    const std::size_t threads = sp.num_threads.value_or(1);
    const std::size_t R = sp.pop ? static_cast<std::size_t>(*sp.pop)
                                 : std::max<std::size_t>(4 * threads, 16);

    /* --- MPI layout (no-op when built without TADAH_HPO_MPI or run on
     * one rank): contiguous walker-slot blocks per rank. Walker RNG
     * streams are keyed by GLOBAL slot index and resampling is decided
     * on rank 0 from the gathered population, so the evaluated set is
     * identical for any rank count (and any THREADS). */
    const int world = hpo_size();
    const int rank  = hpo_rank();
    const bool distributed_mpi = hpo_mpi_active() && world > 1;
#ifdef TADAH_HPO_MPI
    if (distributed_mpi)
      pa_detail::validate_mpi_payload_dimensions(dim, R);
#endif
    const std::vector<double> initial =
        anneal::make_initial_point(sp, AlgoKind::PA);
    if (static_cast<std::size_t>(world) > R)
      throw std::runtime_error(
          "TADAH:ANNEAL_PA: POPULATION (" + std::to_string(R) +
          ") must be >= the MPI world size (" + std::to_string(world) + ")");
    auto slot_count = [&](int r) -> std::size_t {
      return R / world + (static_cast<std::size_t>(r) < R % world ? 1 : 0);
    };
    auto slot_offset = [&](int r) -> std::size_t {
      std::size_t off = 0;
      for (int q = 0; q < r; ++q) off += slot_count(q);
      return off;
    };
    const std::size_t n_local   = slot_count(rank);
    const std::size_t off_local = slot_offset(rank);

    if (!sp.num_threads) {
      const std::string hint =
          "ANNEAL_PA: THREADS not set -- running " + std::to_string(R) +
          " walkers on 1 worker thread. Set THREADS <n> in the STRATEGY "
          "block to evaluate walkers concurrently.";
      std::cout << "[HPO] " << hint << "\n";
      HPOLog::instance().warning(hint);
    }

    /* Per-dimension proposal scale in transformed space: CLASSIC Gaussian
     * sigma, or Corana uniform half-width V (already floor/cap-clamped). */
    const std::vector<double> proposal_scale = anneal::make_initial_steps(sp, kb);
    /* Initial displacement, decoupled from the production step (1.5.0).
     * Identical to proposal_scale unless INIT_MODE JITTER carries a
     * magnitude, so the default trajectory is unchanged. */
    const std::vector<double> jitter_sigma =
        anneal::make_jitter_sigma(sp, kb, proposal_scale);

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval_global = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    /* Rank 0 alone owns seed/T0 calibration.  Cap it at the GLOBAL limit;
     * distributed initialization/sweeps use deterministic global slot
     * allowances and allreduce their rank-local counters. */
    anneal::SharedBudget budget(
        world > 1 && rank != 0 ? HUGE_UNSIGNED : max_eval_global);

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    const std::function<bool()> time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const auto stopval = sp.stop.stopval;

    std::atomic<bool> stop_flag{false};
    anneal::StopCtx ctx{budget, stop_flag, time_exceeded, stopval};

    anneal::CoranaPlateau plateau(sp.stop, kb.neps, sp.failscore);

    enum class StopReason {
      Done, MaxEval, MaxTime, StopVal, TMin, Plateau,
      StopValOnProbe, ExternalOnProbe, MaxTimeOnProbe, NoDims, NoBudget
    };
    StopReason stop_reason = StopReason::Done;

    double      T_hot = 0.0;
    std::size_t step_count = 0;
    std::size_t total_proposed = 0, total_accepted = 0, total_oob = 0;
    std::size_t total_culled = 0;
    /* g_best = GLOBAL best value (reduced, for logging/stops);
     * (l_best, l_best_x) = best THIS RANK evaluated. Ownership is
     * always resolved from l_best (a MINLOC over the already-reduced
     * value ties onto rank 0, which need not own the matching x). */
    double      g_best = std::numeric_limits<double>::infinity();
    double      l_best = std::numeric_limits<double>::infinity();
    Vec         g_best_x(dim);
    Vec         l_best_x(dim);
    bool        ran_sa = false;

    {
      std::ostringstream oss;
      oss << "ANNEAL_PA[" << (kb.corana ? "CORANA" : "CLASSIC") << "]: "
          << "seed=" << seed_value
          << "  POPULATION=" << R
          << "  THREADS=" << threads
          << "  SWEEPS=" << kb.sweeps
          << "  ESS_TARGET=" << kb.ess_target
          << "  RT=" << kb.rt;
      if (kb.t_min) oss << "  T_MIN=" << *kb.t_min;
      else          oss << "  T_MIN=auto(T0*1e-6)";
      if (kb.corana)
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT
            << "  CORANA_C=" << kb.corana_c
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
      else
        oss << "  CHAIN_LENGTH=" << kb.chain_length;
      oss << "  STEP_FRAC=" << kb.step_frac
          << (kb.corana ? " (Corana V half-width/width)"
                        : " (CLASSIC Gaussian sigma/width)");
      oss << "  BOUNDARY_MODE="
          << (kb.boundary_mode == anneal::BoundaryMode::Reflect
                  ? "REFLECT" : "REDRAW")
          << "  INIT_MODE="
          << (kb.init_mode == anneal_enums::INIT_MODE::JITTER ? "JITTER"
                                                              : "UNIFORM (box)");
      if (kb.init_mode == anneal_enums::INIT_MODE::JITTER) {
        switch (kb.init_jitter_mode) {
          case anneal::Knobs::JitterMode::Step:
            oss << (kb.corana ? " (Gaussian; sigma=initial V)"
                              : " (Gaussian; sigma=proposal sigma)");
            break;
          case anneal::Knobs::JitterMode::Fixed:
            oss << " (uniform bound; sigma=" << kb.init_jitter_frac
                << "*width)";
            break;
          case anneal::Knobs::JitterMode::Auto:
            /* make_jitter_sigma: the proposal scale of each dimension.
             * ANNEAL_PA runs no jitter calibration (ANNEAL_PT does), and
             * its JITTER AUTO takes no tolerance. */
            oss << " AUTO (uniform bound; sigma="
                << (kb.corana ? "initial V" : "proposal sigma")
                << " per dim, not calibrated)";
            break;
        }
      }
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Layout arithmetic -- exact, printed up front (mirrors ANNEAL_PT:
     * the first per-step line can be a long time away when the
     * per-walker product is large). */
    const std::uint64_t base_rung_cost =
        kb.corana ? anneal::saturating_mul(
                        anneal::saturating_mul(kb.NT, kb.NS), dim)
                  : static_cast<std::uint64_t>(kb.chain_length);
    const std::uint64_t rung_cost = anneal::saturating_mul(
        base_rung_cost, static_cast<std::uint64_t>(kb.sweeps));
    const std::uint64_t step_cost = anneal::saturating_mul(
        static_cast<std::uint64_t>(R), rung_cost);
    {
      std::ostringstream oss;
      oss << "ANNEAL_PA layout: walker T-step = ";
      if (kb.corana)
        oss << "NT*NS*dim*SWEEPS = " << kb.NT << "*" << kb.NS << "*" << dim
            << "*" << kb.sweeps << " = ";
      else
        oss << "CHAIN_LENGTH*SWEEPS = " << kb.chain_length << "*" << kb.sweeps << " = ";
      oss << rung_cost << " evals;  population T-step = POPULATION*rung = "
          << R << "*" << rung_cost << " = " << step_cost << " evals";
      HPOLog::instance().info(oss.str());
      /* Cadence registration is single-process only: the milestone
       * throughput window is rank-local and would mis-scale a global
       * budget under MPI. */
      if (world == 1 && step_cost > 0)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(step_cost), "T-step",
            max_eval_global == HUGE_UNSIGNED ? 0 : max_eval_global);
    }
    /* Pinned-dim audit (single-process, rank-local marks -- see PT). */
    if (world == 1)
      HPOLog::instance().arm_pin_audit(
          sp, kb.v_floor_frac, static_cast<int>(R), "T-step");

    /* A throw from the seed evaluation or the T0 calibration (rank 0) is
     * kept whole and rethrown after the handshake below: its type matters
     * -- core::ShutdownRequested must reach hpo_run() as a signal stop,
     * HPO_Host::StagnationStop the RESTART policy -- so it is never
     * reduced to its message. */
    bool calib_threw = false;
    std::exception_ptr calib_error;
    if (dim == 0) {
      stop_reason = StopReason::NoDims;
      if (rank == 0 && !sp.optim_init.empty()) {
        if (!budget.try_reserve()) {
          stop_reason = StopReason::NoBudget;
        } else {
          try {
            Vec xv(dim);
            for (std::size_t i = 0; i < dim; ++i) xv(i) = initial[i];
            double f = host(sp.out(xv));
            if (!std::isfinite(f))
              f = std::numeric_limits<double>::infinity();
            if (anneal::stopval_reached(f, stopval, sp.failscore))
              stop_reason = StopReason::StopVal;
            else if (time_exceeded()) stop_reason = StopReason::MaxTime;
          } catch (...) {
            calib_threw = true;
            calib_error = std::current_exception();
          }
        }
      }
    }

    bool proceed = (stop_reason == StopReason::Done);

    /* --- seed evaluation + T0 calibration --------------------------------
     * Rank 0 evaluates the seed and runs the probe calibration; the
     * outcome (T_hot, f0, x0, or an abort code) is broadcast so every
     * rank proceeds from identical state without duplicate probe evals
     * (same handshake as ANNEAL_PT: unconditional, deadlock-safe). */
    Vec x0(dim);
    double f0 = std::numeric_limits<double>::infinity();
    if (proceed && rank == 0) {
      for (std::size_t i = 0; i < dim; ++i) x0(i) = initial[i];
      if (!budget.try_reserve()) {
        stop_reason = StopReason::NoBudget;
        proceed = false;
      } else {
        try {
          f0 = host(sp.out(x0));
          if (!std::isfinite(f0))
            f0 = std::numeric_limits<double>::infinity();
          ran_sa = true;
          g_best = l_best = f0;
          g_best_x = l_best_x = x0;
          if (anneal::stopval_reached(f0, stopval, sp.failscore)) {
            stop_reason = StopReason::StopVal; proceed = false;
          } else if (time_exceeded()) {
            stop_reason = StopReason::MaxTime; proceed = false;
          }
        } catch (...) {
          calib_threw = true;
          calib_error = std::current_exception();
          proceed = false;
        }
      }
    }

    if (proceed && rank == 0) {
      std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);
      try {
        const auto t0out = anneal::calibrate_t0(
            sp, kb, proposal_scale, x0, f0, l_best_x, l_best,
            host, probe_rng, ctx, "ANNEAL_PA");
        g_best = l_best;
        T_hot = t0out.T0;
        if (t0out.aborted) {
          stop_reason = t0out.budget_exhausted
                            ? StopReason::MaxEval
                            : (t0out.stopval_hit
                                   ? StopReason::StopValOnProbe
                                   : (t0out.external_stop
                                          ? StopReason::ExternalOnProbe
                                          : StopReason::MaxTimeOnProbe));
          proceed = false;
        } else if (budget.left() == 0) {
          stop_reason = StopReason::MaxEval;
          proceed = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTimeOnProbe;
          proceed = false;
        }
      } catch (...) {
        calib_threw = true;
        calib_error = std::current_exception();
      }
    }
    if (world > 1) {
      /* A throw on rank 0 ends every rank the way it decides (run_stop.h):
       * rank 0 rethrows it, the others throw the error below or, for a
       * signal stop, PeerShutdown. */
      agree_on_rank_stop(calib_threw ? calib_error : nullptr, [] {
        return std::string(
            "TADAH:ANNEAL_PA: rank 0 aborted during seed/T0 calibration");
      });
      int code = 0;
      if (rank == 0)
        code = proceed ? 0 : 1 + static_cast<int>(stop_reason);
      hpo_bcast(code, 0);
      std::vector<double> pkt(dim + 2, 0.0);
      if (rank == 0) {
        pkt[0] = T_hot; pkt[1] = f0;
        for (std::size_t i = 0; i < dim; ++i) pkt[2 + i] = x0(i);
      }
      hpo_bcast(pkt, 0);
      if (rank != 0) {
        T_hot = pkt[0];
        f0    = pkt[1];
        for (std::size_t i = 0; i < dim; ++i) x0(i) = pkt[2 + i];
        if (code != 0) {
          stop_reason = static_cast<StopReason>(code - 1);
          proceed = false;
        } else {
          proceed  = true;
          ran_sa   = true;
          g_best   = f0;
          g_best_x = x0;
        }
      }
    } else if (calib_threw) {
      std::rethrow_exception(calib_error);
    }

    const double t_min = kb.t_min ? *kb.t_min
                                  : (T_hot > 0.0
                                         ? anneal::automatic_t_min(T_hot)
                                         : 1e-12);
    if (proceed && t_min > T_hot)
      throw std::runtime_error(
          "TADAH:ANNEAL_PA: resolved T_MIN must be <= resolved T_HOT");

    /* --- population (local slot block only) ------------------------------ */
    std::vector<anneal::ChainState<Vec>> walkers(n_local);
    anneal::ReplicaPool pool(std::min(threads, n_local ? n_local : 1),
                             stop_flag);
    /* After every epoch, on every rank: how the epoch ended, agreed over
     * the ranks (run_stop.h; as ANNEAL_PT). */
    auto rethrow_pool_error = [&](const char* phase) {
      agree_on_rank_stop(pool.take_error(), [phase] {
        return std::string(
                   "TADAH:ANNEAL_PA: worker failed on another MPI rank "
                   "during ") + phase;
      });
    };

    /* Per-rank element counts for the population-wide gathers.  Multi-rank
     * MPI limits were checked at the driver prologue, before allocation or
     * evaluation.  Preserve the prior late validation/error order for serial
     * and MPI-world1 runs; do not repeat it after the distributed prologue. */
    std::vector<int> f_counts(world), st_counts(world);
    if (dim > (std::numeric_limits<std::size_t>::max() - 1) / 2)
      throw std::runtime_error(
          "TADAH:ANNEAL_PA: state-vector stride overflows size_t");
    const std::size_t stride = 2 * dim + 1;   /* f | x | V per walker */
    const std::size_t mpi_int_max =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (!distributed_mpi &&
        (R > mpi_int_max || stride > mpi_int_max ||
         R > mpi_int_max / stride))
      throw std::runtime_error(
          "TADAH:ANNEAL_PA: population state exceeds MPI int "
          "count/displacement limits");
    const std::size_t population_state_size = R * stride;
    for (int r = 0; r < world; ++r) {
      const std::size_t rank_slots = slot_count(r);
#ifdef TADAH_HPO_MPI
      if (hpo_mpi_active()) {
        f_counts[r] = hpo_mpi_checked_count(
            rank_slots, "ANNEAL_PA MPI loss gather");
        st_counts[r] = hpo_mpi_checked_count(
            rank_slots * stride, "ANNEAL_PA MPI state gather/scatter");
      } else {
        f_counts[r] = static_cast<int>(rank_slots);
        st_counts[r] = static_cast<int>(rank_slots * stride);
      }
#else
      f_counts[r] = static_cast<int>(rank_slots);
      st_counts[r] = static_cast<int>(rank_slots * stride);
#endif
    }

    if (proceed) {
      /* Walker 0 starts at the seed exactly; higher walkers use the shared
       * kernel (under Corana the initial V[i] doubles as the jitter sigma).
       * The raw `kb.init_mode == 0` this used to compare against is now the
       * named enum inside seed_replica -- PA and PT can no longer drift. */
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        anneal::seed_replica(walkers[li], seed, k, x0, proposal_scale,
                             jitter_sigma, sp, kb);
      }
      if (rank == 0 && n_local > 0) {
        walkers[0].f      = f0;
        walkers[0].best_f = f0;
      }

      /* As in PT, admit a deterministic prefix of global non-seed slots so
       * partial initialization is independent of rank/thread layout. */
      const std::uint64_t spent_before_init = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t init_remaining = max_eval_global > spent_before_init
          ? max_eval_global - spent_before_init : 0;
      const std::uint64_t init_admitted = std::min<std::uint64_t>(
          R > 0 ? static_cast<std::uint64_t>(R - 1) : 0, init_remaining);
      pool.run_epoch(n_local, [&](std::size_t li) {
        const std::size_t gk = off_local + li;
        warmstart::current_chain() = gk;   /* global slot key */
        if (rank == 0 && li == 0) return;   /* global slot 0 = seed */
        if (gk == 0 || gk > init_admitted) return;
        auto& w = walkers[li];
        if (stop_flag.load(std::memory_order_relaxed)) return;
        if (!ctx.budget.try_reserve()) return;   /* f stays inf */
        w.f = host(sp.out(w.x));
        if (!std::isfinite(w.f))
          w.f = std::numeric_limits<double>::infinity();
        w.best_f = w.f;
        w.best_x = w.x;
        if (anneal::stopval_reached(w.f, stopval, sp.failscore))
          stop_flag.store(true, std::memory_order_relaxed);
      });
      rethrow_pool_error("population initialization");
      for (auto& w : walkers) {
        if (w.best_f < l_best) { l_best = w.best_f; l_best_x = w.best_x; }
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        int owner = 0;
        double gb = std::isfinite(l_best) ? l_best
                                          : std::numeric_limits<double>::max();
        hpo_allreduce_minloc(gb, owner);
        if (gb < std::numeric_limits<double>::max())
          g_best = std::min(g_best, gb);
        int code = (stopval && stop_flag.load(std::memory_order_relaxed) &&
                    anneal::stopval_reached(g_best, stopval, sp.failscore)) ? 1 : 0;
        code = hpo_allreduce_max(code);
        if (code) { stop_reason = StopReason::StopVal; proceed = false; }
      } else if (stopval && stop_flag.load(std::memory_order_relaxed) &&
                 anneal::stopval_reached(g_best, stopval, sp.failscore)) {
        stop_reason = StopReason::StopVal;
        proceed = false;
      }
    }

    /* One T-step of sweeps for the whole population (budget-sliced,
     * deterministic; slicing computed from the GLOBAL remaining budget
     * identically on every rank -- see the PT driver). */
    std::vector<anneal::RungStats> rstats(n_local);
    std::size_t last_proposed = 0, last_accepted = 0, last_oob = 0;
    /* Pinned-dim audit unit: one tick per run_sweeps call. Kept separate
     * from step_count (whose increment site differs between the first
     * and subsequent T-steps) -- only monotonicity matters here. */
    std::uint64_t pin_unit = 0;
    auto run_sweeps = [&](double T) -> bool {  /* false => stop fired */
      const std::uint64_t spent_global =
          world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t remaining =
          max_eval_global > spent_global ? max_eval_global - spent_global : 0;
      std::vector<std::uint64_t> allowance(R, rung_cost);
      if (remaining < step_cost) {
        const std::uint64_t base = remaining / R;
        const std::uint64_t rem  = remaining % R;
        for (std::size_t k = 0; k < R; ++k)
          allowance[k] = base + (k < rem ? 1 : 0);
      }
      pool.run_epoch(n_local, [&](std::size_t li) {
        const std::size_t k = off_local + li;
        warmstart::current_chain() = k;   /* global slot for warm-start key */
        anneal::RungStats acc;
        for (std::size_t s = 0; s < kb.sweeps; ++s) {
          const std::uint64_t left_here =
              allowance[k] > acc.proposed ? allowance[k] - acc.proposed : 0;
          if (left_here == 0) { acc.stop = RungStop::Budget; break; }
          const auto rs = anneal::run_rung(walkers[li], T, sp, kb, host, ctx,
                                           left_here);
          acc.proposed += rs.proposed;
          acc.accepted += rs.accepted;
          acc.oob      += rs.oob;
          acc.stop      = rs.stop;
          if (rs.stop != RungStop::Completed) break;
        }
        rstats[li] = acc;
      });
      rethrow_pool_error("walker sweeps");

      /* Pinned-dim audit refresh: post-sweeps, PRE-resampling (marks
       * follow what each walker's own sweeps did; resampling replaces
       * walkers and simply re-marks/unmarks on the next T-step). */
      if (world == 1) {
        ++pin_unit;
        for (std::size_t li = 0; li < n_local; ++li)
          HPOLog::instance().record_pin_state(
              static_cast<int>(off_local + li), pin_unit,
              anneal::pinned_dims(walkers[li], sp, kb));
      }

      std::size_t e_prop = 0, e_acc = 0, e_oob = 0;
      bool any_stopval = false, any_time = false;
      for (std::size_t li = 0; li < n_local; ++li) {
        e_prop += rstats[li].proposed;
        e_acc  += rstats[li].accepted;
        e_oob  += rstats[li].oob;
        if (rstats[li].stop == RungStop::StopVal) any_stopval = true;
        if (rstats[li].stop == RungStop::Time)    any_time    = true;
        if (walkers[li].best_f < l_best) {
          l_best   = walkers[li].best_f;
          l_best_x = walkers[li].best_x;
        }
      }
      g_best = std::min(g_best, l_best);
      any_time = any_time || time_exceeded();
      if (world > 1) {
        e_prop = hpo_allreduce_sum_sz(e_prop);
        e_acc  = hpo_allreduce_sum_sz(e_acc);
        e_oob  = hpo_allreduce_sum_sz(e_oob);
        int code = any_stopval ? 2 : (any_time ? 1 : 0);
        code = hpo_allreduce_max(code);
        any_stopval = (code == 2);
        any_time    = (code == 1);
      }
      total_proposed += e_prop;
      total_accepted += e_acc;
      total_oob      += e_oob;
      last_proposed = e_prop;
      last_accepted = e_acc;
      last_oob      = e_oob;
      if (any_stopval || (stopval &&
                          stop_flag.load(std::memory_order_relaxed) &&
                          anneal::stopval_reached(g_best, stopval, sp.failscore))) {
        stop_reason = StopReason::StopVal; return false;
      }
      if (any_time) { stop_reason = StopReason::MaxTime; return false; }
      if (e_prop == 0) {
        stop_reason = StopReason::MaxEval; return false;
      }
      if (max_eval_global != HUGE_UNSIGNED && e_prop >= remaining) {
        stop_reason = StopReason::MaxEval; return false;
      }
      return true;
    };

    /* --- anneal loop -------------------------------------------------------
     * Fixed per-step collective order (every rank, every step):
     * [gatherv f] -> [bcast ctrl] -> (resample: [gatherv states] ->
     * [scatterv states]) -> run_sweeps' collectives -> [minloc pop
     * best] -> [minloc g_best] -> [bcast rank-0 plateau code]. All
     * schedule decisions are made on rank 0 from the gathered
     * population, so any world size sees the identical schedule. */
    if (proceed) {
      double T = T_hot;

      /* Step 0: equilibrate the fresh (delta-ish) population at T_HOT
       * before the first reweighting. */
      bool ok = run_sweeps(T);
      /* T_HOT == T_MIN is a valid one-temperature run. Stop after its hot
       * equilibration instead of entering the resampler with dbeta == 0 and
       * performing a redundant second sweep. A stop from run_sweeps (most
       * importantly exact MAXEVAL) retains priority. */
      if (ok && T <= t_min) {
        stop_reason = StopReason::TMin;
        ok = false;
      }

      /* Planned total for the step=current/total display: the budget
       * bound (every step sweeps the whole population), tightened by
       * the geometric T_MIN bound when the schedule is pure geometric
       * (adaptive ESS cools no faster than RT, so its step count has
       * no useful upper T-bound). 0 = unbounded -> plain counter.
       * Computed once, symmetric across ranks (one extra allreduce). */
      std::size_t planned_steps = 0;
      {
        const std::uint64_t spent_now =
            world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t rem =
            max_eval_global > spent_now ? max_eval_global - spent_now : 0;
        if (max_eval_global != HUGE_UNSIGNED || kb.ess_target == 0.0) {
          planned_steps = anneal::planned_rungs(
              T, t_min, kb.ess_target == 0.0 ? kb.rt : 1.0,
              max_eval_global == HUGE_UNSIGNED
                  ? std::numeric_limits<std::uint64_t>::max()
                  : rem,
              step_cost);
          /* Plan line -- printed once, from the same arithmetic that
           * feeds the step=k/total display (an upper bound: STOPVAL /
           * plateau / MAXTIME end the run earlier). */
          if (rank == 0 && planned_steps > 0) {
            std::ostringstream oss;
            oss << "ANNEAL_PA plan: ~" << planned_steps
                << " temperature step(s) (";
            if (max_eval_global != HUGE_UNSIGNED)
              oss << "budget left " << rem << " evals";
            if (max_eval_global != HUGE_UNSIGNED && kb.ess_target == 0.0)
              oss << ", ";
            if (kb.ess_target == 0.0) oss << "geometric cooling to T_MIN";
            oss << "; upper bound)";
            HPOLog::instance().info(oss.str());
          }
        }
      }

      std::vector<double> f_local(n_local), f_all(R);
      std::vector<double> st_local(n_local * stride),
                          st_all(population_state_size);

      auto pack_states = [&]() {
        for (std::size_t li = 0; li < n_local; ++li) {
          double* p = st_local.data() + li * stride;
          p[0] = walkers[li].f;
          for (std::size_t d = 0; d < dim; ++d) p[1 + d]       = walkers[li].x(d);
          for (std::size_t d = 0; d < dim; ++d) p[1 + dim + d] = walkers[li].V[d];
        }
      };
      auto unpack_states = [&]() {
        for (std::size_t li = 0; li < n_local; ++li) {
          const double* p = st_local.data() + li * stride;
          walkers[li].f = p[0];
          for (std::size_t d = 0; d < dim; ++d) walkers[li].x(d) = p[1 + d];
          for (std::size_t d = 0; d < dim; ++d) walkers[li].V[d] = p[1 + dim + d];
        }
      };

      while (ok) {
        const auto step_wall_t0 = std::chrono::steady_clock::now();
        /* --- gather the population's current losses ------------------- */
        for (std::size_t li = 0; li < n_local; ++li) f_local[li] = walkers[li].f;
        hpo_gatherv(f_local, f_all, f_counts);

        /* --- rank 0 chooses the next temperature + resample plan ------ */
        double T_new = T, ess = 0.0;
        int    do_resample = 0;
        bool   adaptive_used = false;
        std::size_t culled = 0, unique_parents = R, n_finite_pre = 0;
        std::vector<std::size_t> parent;
        if (rank == 0) {
          std::vector<double> f_finite;
          f_finite.reserve(R);
          for (double f : f_all)
            if (anneal::loss_is_finite(f, sp.failscore)) f_finite.push_back(f);
          n_finite_pre = f_finite.size();
          if (static_cast<double>(n_finite_pre) < 0.10 * static_cast<double>(R)) {
            HPOLog::instance().warning(
                "ANNEAL_PA: population collapse -- only " +
                std::to_string(n_finite_pre) + "/" + std::to_string(R) +
                " walkers carry a feasible loss");
          }

          const double geometric_T =
              anneal::pa_geometric_temperature(T, kb.rt, t_min);
          const long double beta_ld = 1.0L / static_cast<long double>(T);
          const long double cap_ld =
              1.0L / static_cast<long double>(geometric_T) - beta_ld;
          const long double double_max_ld = static_cast<long double>(
              std::numeric_limits<double>::max());
          const bool cap_not_representable = cap_ld > double_max_ld ||
              (cap_ld > 0.0L && static_cast<double>(cap_ld) == 0.0);
          const double beta_rt_cap = cap_ld > double_max_ld
                                          ? std::numeric_limits<double>::max()
                                          : static_cast<double>(cap_ld);
          double requested_dbeta = beta_rt_cap;       /* geometric default */
          if (!cap_not_representable && kb.ess_target > 0.0 &&
              f_finite.size() >= 2) {
            const double lo = *std::min_element(f_finite.begin(), f_finite.end());
            const double hi = *std::max_element(f_finite.begin(), f_finite.end());
            if (hi > lo) {   /* non-degenerate -> adapt */
              requested_dbeta = anneal::choose_dbeta(
                  f_finite, kb.ess_target, beta_rt_cap);
              adaptive_used = true;
            }
          }
          /* choose_dbeta is public double arithmetic.  Materialize its
           * endpoint, force one representable decrease if it rounded back to
           * T, then reweight with the inverse-temperature increment actually
           * represented by that endpoint. */
          const double proposed_T = cap_not_representable
              ? geometric_T
              : static_cast<double>(
                    1.0L / (beta_ld +
                            static_cast<long double>(requested_dbeta)));
          const auto realized = anneal::realize_pa_temperature_step(
              T, geometric_T, proposed_T);
          T_new = realized.temperature;
          const long double realized_dbeta = realized.dbeta;

          /* Reweight + systematic resample plan over the gathered f. */
          std::vector<double> wgt(R, 0.0);
          long double wsum = 0.0L, wsum2 = 0.0L;
          double fmin = std::numeric_limits<double>::infinity();
          for (double f : f_all)
            if (anneal::loss_is_finite(f, sp.failscore) && f < fmin) fmin = f;
          for (std::size_t k = 0; k < R; ++k) {
            if (anneal::loss_is_finite(f_all[k], sp.failscore)) {
              const long double delta = static_cast<long double>(f_all[k]) -
                                        static_cast<long double>(fmin);
              const long double exponent = delta == 0.0L
                  ? 0.0L : -realized_dbeta * delta;
              const long double w = std::exp(exponent);
              wgt[k] = static_cast<double>(w);
              wsum += w;
              wsum2 += w * w;
            }
          }
          if (wsum > 0.0L && wsum2 > 0.0L) {
            ess = static_cast<double>((wsum * wsum) / wsum2);
            std::mt19937_64 rs_rng =
                anneal::make_stream(seed, anneal::SALT_PA_RESAMPLE, step_count);
            const double u =
                std::uniform_real_distribution<double>(0.0, 1.0)(rs_rng);
            parent = anneal::systematic_resample(wgt, u);
            std::unordered_set<std::size_t> uniq(parent.begin(), parent.end());
            unique_parents = uniq.size();
            culled = R - unique_parents;
            do_resample = 1;
          } else {
            HPOLog::instance().warning(
                "ANNEAL_PA: no feasible walker -- skipping resampling this step "
                "(geometric cooling continues)");
          }
        }

        if (world > 1) {
          std::vector<double> ctrl{
              T_new, static_cast<double>(do_resample), ess,
              static_cast<double>(culled),
              static_cast<double>(unique_parents),
              static_cast<double>(n_finite_pre),
              adaptive_used ? 1.0 : 0.0};
          hpo_bcast(ctrl, 0);
          T_new       = ctrl[0];
          do_resample = static_cast<int>(ctrl[1]);
          ess          = ctrl[2];
          culled       = static_cast<std::size_t>(ctrl[3]);
          unique_parents = static_cast<std::size_t>(ctrl[4]);
          n_finite_pre = static_cast<std::size_t>(ctrl[5]);
          adaptive_used = ctrl[6] != 0.0;
        }

        /* --- apply the resample plan ----------------------------------- */
        if (do_resample) {
          pack_states();
          hpo_gatherv(st_local, st_all, st_counts);
          if (rank == 0) {
            std::vector<double> permuted(population_state_size);
            for (std::size_t j = 0; j < R; ++j)
              std::copy(st_all.begin() + parent[j] * stride,
                        st_all.begin() + (parent[j] + 1) * stride,
                        permuted.begin() + j * stride);
            st_all.swap(permuted);
          }
          hpo_scatterv(st_all, st_local, st_counts);
          unpack_states();
        }
        total_culled += culled;

        /* --- sweeps at the new temperature ---------------------------- */
        T = T_new;
        ++step_count;
        ok = run_sweeps(T);
        const std::uint64_t spent_after_sweeps = world > 1
            ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const bool maxeval_hit = max_eval_global != HUGE_UNSIGNED &&
                                 spent_after_sweeps >= max_eval_global;

        anneal::StepSaturation step_vsat;
        if (kb.corana) {
          for (const auto& w : walkers) {
            const auto s = anneal::step_saturation(w.V, sp, kb);
            step_vsat.floor += s.floor;
            step_vsat.cap   += s.cap;
            step_vsat.total += s.total;
          }
          if (world > 1) {
            step_vsat.floor = hpo_allreduce_sum_sz(step_vsat.floor);
            step_vsat.cap   = hpo_allreduce_sum_sz(step_vsat.cap);
            step_vsat.total = hpo_allreduce_sum_sz(step_vsat.total);
          }
        }

        /* Population best after the sweeps (current f, not the
         * monotone best) + the global best for the plateau rule. */
        double f_pop_best = std::numeric_limits<double>::infinity();
        std::size_t n_fin_local = 0;
        for (const auto& w : walkers) {
          if (anneal::loss_is_finite(w.f, sp.failscore)) {
            ++n_fin_local;
            if (w.f < f_pop_best) f_pop_best = w.f;
          }
        }
        std::size_t n_fin = n_fin_local;
        if (world > 1) {
          int dummy = 0;
          if (!std::isfinite(f_pop_best))
            f_pop_best = std::numeric_limits<double>::max();
          hpo_allreduce_minloc(f_pop_best, dummy);
          double gb = std::isfinite(l_best) ? l_best
                                            : std::numeric_limits<double>::max();
          hpo_allreduce_minloc(gb, dummy);
          if (gb < std::numeric_limits<double>::max())
            g_best = std::min(g_best, gb);
          n_fin = hpo_allreduce_sum_sz(n_fin_local);
        }

        if (rank == 0) {
          const double accept_pct = last_proposed > 0
              ? 100.0 * static_cast<double>(last_accepted) /
                    static_cast<double>(last_proposed)
              : 0.0;
          const double oob_pct = last_proposed > 0
              ? 100.0 * static_cast<double>(last_oob) /
                    static_cast<double>(last_proposed)
              : 0.0;
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PA step=" << step_count;
          if (planned_steps > 0) oss << "/" << planned_steps;
          oss << "  T=" << T
              << "  ESS=" << std::fixed << std::setprecision(1) << ess
              << "  parents=" << unique_parents << "/" << R
              << "  culled=" << culled
              << "  finite=" << n_fin << "/" << R
              << "  proposed=" << last_proposed
              << "  accepted=" << std::fixed << std::setprecision(1)
              << accept_pct << "%"
              << "  oob=" << last_oob << " (" << oob_pct << "%)"
              << std::scientific << std::setprecision(3)
              << "  pop_best=" << f_pop_best
              << "  best=" << g_best;
          if (kb.corana)
            oss << "  V_floor=" << step_vsat.floor << "/" << step_vsat.total
                << "  V_cap=" << step_vsat.cap << "/" << step_vsat.total;
          oss
              << "  step_wall=" << HPOLog::format_duration(
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - step_wall_t0)
                         .count())
              << (adaptive_used ? "" : "  [geometric]");
          HPOLog::instance().info(oss.str());
        }

        int rank0_code = 0;   /* 0 none, 1 plateau */
        if (rank == 0 && ok &&
            plateau.update(f_pop_best, g_best)) rank0_code = 1;
        if (world > 1) hpo_bcast(rank0_code, 0);

        if (!ok) break;
        if (maxeval_hit) { stop_reason = StopReason::MaxEval; break; }
        if (rank0_code == 1) { stop_reason = StopReason::Plateau; break; }
        if (T <= t_min) { stop_reason = StopReason::TMin; break; }
      }
    }

    /* --- multi-rank finalisation: share the global winner without a
     * synthetic rank-0 re-evaluation (which would exceed MAXEVAL). */
    anneal::StepSaturation final_vsat;
    if (kb.corana) {
      for (const auto& w : walkers) {
        const auto s = anneal::step_saturation(w.V, sp, kb);
        final_vsat.floor += s.floor;
        final_vsat.cap   += s.cap;
        final_vsat.total += s.total;
      }
      if (world > 1) {
        final_vsat.floor = hpo_allreduce_sum_sz(final_vsat.floor);
        final_vsat.cap   = hpo_allreduce_sum_sz(final_vsat.cap);
        final_vsat.total = hpo_allreduce_sum_sz(final_vsat.total);
      }
    }
    std::uint64_t spent_report = budget.spent();
    if (world > 1 && dim > 0) {
      double gb = std::isfinite(l_best) ? l_best
                                        : std::numeric_limits<double>::max();
      int owner = 0;
      hpo_allreduce_minloc(gb, owner);
      if (gb < std::numeric_limits<double>::max()) {
        g_best = gb;
        std::vector<double> bx(dim, 0.0);
        if (rank == owner)
          for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
        hpo_bcast(bx, owner);
        for (std::size_t d = 0; d < dim; ++d) g_best_x(d) = bx[d];
      }
    }
    if (world > 1) spent_report = hpo_allreduce_sum(budget.spent());

    /* --- final summary ----------------------------------------------------- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) /
            static_cast<double>(total_proposed) : 0.0;
      const double oob_pct = total_proposed > 0
          ? 100.0 * static_cast<double>(total_oob) /
                static_cast<double>(total_proposed)
          : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL_PA summary: steps=" << step_count
          << "  population=" << R
          << "  proposed=" << total_proposed
          << "  accepted=" << std::fixed << std::setprecision(1)
          << (overall_rate * 100.0) << "%"
          << std::scientific << std::setprecision(3)
          << "  oob=" << total_oob << " (" << std::fixed
          << std::setprecision(1) << oob_pct << "%)"
          << std::scientific << std::setprecision(3)
          << "  culled_total=" << total_culled
          << "  T_hot=" << T_hot
          << "  best=" << g_best;
      if (kb.corana)
        oss << "  V_floor=" << final_vsat.floor << "/" << final_vsat.total
            << "  V_cap=" << final_vsat.cap << "/" << final_vsat.total;
      HPOLog::instance().info(oss.str());
    }

    /* The stop line, and the run's status (run_end.h) on every rank: the
     * stop decisions are collective, so every rank's journal gets the
     * same one. */
    {
      const char* reason = nullptr;
      EndReason end = EndReason::MaxEval;
      switch (stop_reason) {
        case StopReason::MaxEval:
          reason = "MAXEVAL exhausted"; end = EndReason::MaxEval; break;
        case StopReason::MaxTime:
          reason = "MAXTIME exceeded"; end = EndReason::MaxTime; break;
        case StopReason::StopVal:
          reason = "STOPVAL reached"; end = EndReason::StopVal; break;
        case StopReason::StopValOnProbe:
          reason = "STOPVAL reached during T0 probes";
          end = EndReason::StopVal; break;
        case StopReason::ExternalOnProbe:
          /* ctx.stop is raised only by a STOPVAL (anneal_core.h), and rank
           * 0 calibrates before the population is initialised:
           * unreachable today; were it reached, another chain's STOPVAL
           * ended the search. */
          reason = "external stop requested during T0 probes";
          end = EndReason::StopVal; break;
        case StopReason::MaxTimeOnProbe:
          reason = "MAXTIME exceeded during T0 probes";
          end = EndReason::MaxTime; break;
        case StopReason::TMin:
          reason = "T <= T_MIN"; end = EndReason::TMin; break;
        case StopReason::Plateau:
          reason = "FTOL plateau (Corana eps-NEPS rule, population best)";
          end = EndReason::FtolPlateau; break;
        case StopReason::NoDims:
          reason = "no optimisable dimensions"; end = EndReason::NoDims; break;
        case StopReason::NoBudget:
          reason = "MAXEVAL exhausted before SA proposal";
          end = EndReason::MaxEval; break;
        case StopReason::Done:
          throw std::logic_error(
              "internal error: TADAH:ANNEAL_PA ended without a stop reason");
      }
      std::ostringstream oss;
      oss << "ANNEAL_PA stopped: " << reason
          << " after " << spent_report << " host eval(s)"
          << " (" << step_count << " temperature step(s))";
      HPOLog::instance().info(oss.str());
      record_end("TADAH:ANNEAL_PA", end);
    }
  }

template<typename Vec, typename Runner>
  void run_anneal_pa(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pa_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo

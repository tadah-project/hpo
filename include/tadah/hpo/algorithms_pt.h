#pragma once
/* ----- ANNEAL_PT  (parallel tempering / replica exchange) ------------
 *
 * Refs: Swendsen & Wang, PRL 57 (1986); Geyer (1991); Earl & Deem,
 *       PCCP 7 (2005) [review]; Corana et al., ACM TOMS 13:262 (1987)
 *       for the per-replica adaptive stepper; the deterministic
 *       even/odd (DEO) pairing follows Okabe et al., CPL 335 (2001).
 *
 * K replicas run the shared annealing rung (anneal_core.h) on a geometric
 * ladder T_0 < T_1 < ... < T_{K-1} (slot 0 coldest). It is static by
 * default; RT < 1 cools the whole ladder each epoch. Every
 * SWAP_EVERY epochs, adjacent pairs attempt a state exchange with
 * Metropolis probability
 *
 *     p = min(1, exp((beta_i - beta_j) * (f_i - f_j)))     (i colder)
 *
 * A lower-f state found on the hot side moves cold-ward with probability
 * one; reverse exchanges remain possible to preserve detailed balance.
 * Swaps move STATES (x, f), not temperatures: the Corana step vector V is
 * calibrated to the acceptance rate AT its slot's temperature and
 * therefore stays with the slot.
 *
 * Execution: THREADS is the TOTAL width of the shared worker pool (the
 * per-rank CPU-core budget) that actually runs the pc_lammps fan-out and
 * MLIP train/predict; it no longer multiplies OMP_NUM_THREADS. A bounded
 * set of driver threads (min(THREADS, K)) steps the K replicas, but each
 * driver BLOCKS on the shared pool while its eval's scripts run, so the
 * real concurrency is THREADS and one replica's eval alone submits far
 * more tasks than THREADS -- enough to keep the pool saturated. One host,
 * many threads -- HPO_Host's operator() is thread-safe by contract.
 * With REPLICAS fixed, a deterministic objective, and no timing-dependent
 * stop, the same SEED gives the same evaluated set for any THREADS (per-replica
 * RNG streams, driver-side swap coins keyed by epoch, deterministic
 * per-epoch budget slicing). Documented exceptions: MAXTIME truncation and
 * runs cut short by STOPVAL/exceptions are timing-dependent.
 * (OMP_NUM_THREADS independently drives the residual OpenMP kernels --
 * neighbour-list build, data-stats -- not this pool.)
 *
 * Ladder construction: T_HOT from PARAM T_HOT or the shared T0
 * calibration (T0_PACCEPT is assigned to the calibration delta at the hot end);
 * T_COLD from PARAM T_COLD or T_HOT / LADDER_SPAN; geometric spacing
 * in between (adjacent ratio (T_HOT/T_COLD)^(1/(K-1)). The commonly cited
 * 20-40% per-pair acceptance range is a starting heuristic, not a
 * workload-independent target; rates are logged so the ladder can be
 * tuned). RT (default 1.0) < 1 additionally cools the WHOLE
 * ladder each epoch (PT -> SA hybrid); the run then ends when the hot
 * end crosses T_MIN.
 *
 * PT-specific PARAMs (shared stepper knobs as in ANNEAL):
 *   REPLICAS     max(THREADS,2)  ladder size K (default: one replica per
 *                                core; set REPLICAS to decouple K from the
 *                                THREADS core budget)
 *   T_HOT        auto            hottest slot
 *   T_COLD       T_HOT/LADDER_SPAN  coldest slot
 *   LADDER_SPAN  1e3             T_HOT/T_COLD when T_COLD unset
 *   SWAP_EVERY   1               exchange attempt every N epochs
 *   SWAP_WINDOW  20              windowed per-pair swap-acceptance table
 *                                every N epochs (0 = cumulative only)
 *   RT           1.0             ladder cooling per epoch ((0,1])
 *   INIT_MODE    0               0: chain0=seed; others use Gaussian
 *                                jitter (sigma=CLASSIC scale or Corana V)
 *                                1: chains>0 drawn uniformly in the box
 *   RESET_TO_BEST 0              when 1, snaps ONLY the coldest slot to
 *                                the global best after each epoch
 */
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/warmstart_cache.h>   // current_chain() bridge

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tadah::hpo {

namespace pt_detail {

#ifdef TADAH_HPO_MPI
/* PT broadcasts a calibration packet containing T_hot, f0, and the full
 * parameter vector (dim + 2 doubles), then exchanges dim doubles at ladder
 * block boundaries.  Reject an unrepresentable packet before allocating any
 * dimension-sized driver state or entering a collective. */
inline void validate_mpi_payload_dimension(std::size_t dim)
{
  const std::size_t mpi_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  constexpr std::size_t calibration_fields = 2;
  if (dim > mpi_limit - calibration_fields)
    throw std::length_error(
        "TADAH:ANNEAL_PT: parameter dimension " + std::to_string(dim) +
        " exceeds the MPI payload limit (dimension + 2 must fit in int)");
}
#endif

/* Replica-exchange decision for adjacent slots, `cold_temperature <=
 * hot_temperature`. Resolve deterministic limits before forming inverse
 * temperatures: cooling may legitimately underflow ladder endpoints to zero,
 * while equal temperatures/losses make the formal beta-gap product
 * indeterminate even though the exchange probability is one.
 *
 * For a genuine uphill exchange, compare in the log domain:
 *
 *   coin < exp(-m),
 *   m = (f_hot - f_cold) * (T_hot - T_cold) / (T_cold * T_hot).
 *
 * This is algebraically the usual replica-exchange rule for finite positive
 * temperatures, but it cannot create `inf - inf` or `inf * 0`. Normalized
 * unusable losses are represented by infinities; their ordering has the
 * corresponding zero-temperature limit. A coin is drawn only when the
 * decision is genuinely probabilistic, matching the annealing core RNG
 * discipline.
 */
template<class DrawCoin>
inline bool accept_exchange(double cold_temperature, double hot_temperature,
                            double cold_loss, double hot_loss,
                            DrawCoin&& draw_coin)
{
  if (!std::isfinite(cold_temperature) ||
      !std::isfinite(hot_temperature) || cold_temperature < 0.0 ||
      hot_temperature < cold_temperature)
    throw std::invalid_argument(
        "ANNEAL_PT exchange requires finite 0 <= T_cold <= T_hot");
  if (std::isnan(cold_loss) || std::isnan(hot_loss))
    throw std::invalid_argument(
        "ANNEAL_PT exchange requires losses normalized away from NaN");

  /* Equal beta or equal energy gives exponent zero, including 0/0 and
   * infinity/infinity limiting cases. */
  if (cold_temperature == hot_temperature || cold_loss == hot_loss)
    return true;

  /* Moving a strictly lower loss into the colder slot is always accepted.
   * This comparison also handles a normalized +/-infinity without doing
   * arithmetic on it. */
  if (hot_loss < cold_loss) return true;

  /* The only remaining ordering is f_cold < f_hot. At zero cold
   * temperature, or with an infinite endpoint loss, the reverse exchange has
   * probability zero. */
  if (cold_temperature == 0.0 || !std::isfinite(cold_loss) ||
      !std::isfinite(hot_loss))
    return false;

  const double coin = draw_coin();
  if (!std::isfinite(coin) || coin < 0.0 || !(coin < 1.0))
    throw std::invalid_argument(
        "ANNEAL_PT exchange coin must be finite and in [0,1)");

  const long double energy_gap = static_cast<long double>(hot_loss) -
                                 static_cast<long double>(cold_loss);
  const long double temperature_gap =
      static_cast<long double>(hot_temperature) -
      static_cast<long double>(cold_temperature);
  const long double log_m = std::log(energy_gap) +
                            std::log(temperature_gap) -
                            std::log(static_cast<long double>(cold_temperature)) -
                            std::log(static_cast<long double>(hot_temperature));

  /* coin==0 is below every positive, finite mathematical probability. An
   * infinite m is the exact zero-probability limit and remains rejected. */
  if (coin == 0.0) return !std::isinf(log_m);
  const long double log_threshold =
      std::log(-std::log(static_cast<long double>(coin)));
  return log_m < log_threshold;
}

}  // namespace pt_detail

template<typename Vec, typename HostT>
  void run_anneal_pt_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
#ifdef TADAH_HPO_MPI
    if (hpo_mpi_active()) pt_detail::validate_mpi_payload_dimension(dim);
#endif
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::PT);
    const std::vector<double> initial =
        anneal::make_initial_point(sp, AlgoKind::PT);

    const std::size_t threads = sp.num_threads.value_or(1);
    const std::size_t K = kb.replicas ? kb.replicas
                                      : std::max<std::size_t>(threads, 2);
    if (K < 2)
      throw std::runtime_error("TADAH:ANNEAL_PT: needs REPLICAS >= 2");

    /* --- MPI layout (no-op when built without TADAH_HPO_MPI or run on
     * one rank): contiguous ladder blocks per rank, slot 0 (coldest) on
     * rank 0. Chain RNG streams are keyed by GLOBAL slot index, budget
     * slices are computed from the GLOBAL remaining budget, and swap
     * coins come from per-pair streams -- so the evaluated set is
     * identical for any rank count (and any THREADS). */
    const int world = hpo_size();
    const int rank  = hpo_rank();
    if (static_cast<std::size_t>(world) > K)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: REPLICAS (" + std::to_string(K) +
          ") must be >= the MPI world size (" + std::to_string(world) + ")");
    auto slot_count = [&](int r) -> std::size_t {
      return K / world + (static_cast<std::size_t>(r) < K % world ? 1 : 0);
    };
    auto slot_offset = [&](int r) -> std::size_t {
      std::size_t off = 0;
      for (int q = 0; q < r; ++q) off += slot_count(q);
      return off;
    };
    const std::size_t n_local  = slot_count(rank);
    const std::size_t off_local = slot_offset(rank);

    if (!sp.num_threads) {
      const std::string hint =
          "ANNEAL_PT: THREADS not set -- running " + std::to_string(K) +
          " replicas on 1 worker thread. Set THREADS <n> in the STRATEGY "
          "block to evaluate replicas concurrently.";
      std::cout << "[HPO] " << hint << "\n";
      HPOLog::instance().warning(hint);
    }
#ifdef TADAH_ENABLE_OPENMP
    {
      // Under the shared-pool model THREADS is the actual concurrency: the
      // pc_lammps fan-out and the MLIP train/predict all run on the pool.
      // OMP_NUM_THREADS drives ONLY the residual NN-list / data-stats kernels,
      // which fire transiently during per-eval training and not at all for
      // pc_lammps-only configs -- so it is NOT multiplicative with THREADS.
      // Warn only when the pool itself oversubscribes the cores. Do NOT
      // re-introduce a "THREADS x OMP_NUM_THREADS > cores" warning: it is a
      // false positive for the common (pc_lammps) case, where OMP is inert.
      const unsigned hw = std::thread::hardware_concurrency();
      if (hw && threads > hw) {
        std::ostringstream oss;
        oss << "ANNEAL_PT: THREADS(" << threads
            << ") exceeds hardware_concurrency(" << hw
            << "); the shared worker pool is oversubscribed. Set THREADS <= cores.";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
#endif

    /* Per-dimension proposal scale in transformed space: CLASSIC Gaussian
     * sigma, or Corana uniform half-width V (already floor/cap-clamped). */
    const std::vector<double> proposal_scale = anneal::make_initial_steps(sp, kb);

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval_global = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    /* Rank 0 alone owns seed/T0 calibration.  Cap it at the GLOBAL limit;
     * subsequent distributed phases are admitted from deterministic global
     * slot allowances and allreduce their rank-local counters. */
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

    double      T_hot = 0.0, T_cold = 0.0;
    std::size_t epoch_count    = 0;
    std::size_t exchange_rounds = 0;
    std::size_t total_proposed = 0, total_accepted = 0, total_oob = 0;
    std::size_t reanneal_count = 0;  (void)reanneal_count;
    /* g_best is the GLOBAL best value (allreduced for logging/stops);
     * (l_best, l_best_x) is the best THIS RANK evaluated -- ownership
     * questions (who broadcasts the winning x) are always resolved
     * from l_best, never from the reduced value (after a reduction
     * every rank holds the same number, so a MINLOC over it would tie
     * onto rank 0 which need not own the matching x). */
    double      g_best = std::numeric_limits<double>::infinity();
    double      l_best = std::numeric_limits<double>::infinity();
    Vec         g_best_x(dim);
    Vec         l_best_x(dim);
    bool        ran_sa = false;

    {
      std::ostringstream oss;
      oss << "ANNEAL_PT[" << (kb.corana ? "CORANA" : "CLASSIC") << "]: "
          << "seed=" << seed
          << "  REPLICAS=" << K
          << "  THREADS=" << threads
          << "  SWAP_EVERY=" << kb.swap_every
          << "  RT=" << kb.rt;
      if (kb.t_cold)      oss << "  T_COLD=" << *kb.t_cold;
      else                oss << "  LADDER_SPAN=" << kb.ladder_span;
      if (kb.corana)
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT << "  C=" << kb.C
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
      else
        oss << "  L=" << kb.L;
      if (sp.step.has_value()) {
        if (kb.corana)
          oss << "  STEP=" << *sp.step
              << " (Corana U(-V,V) half-width; floor/cap applied per dim)";
        else
          oss << "  sigma=" << *sp.step
              << " (CLASSIC Gaussian; explicit STEP)";
      } else {
        oss << "  STEP_FRAC=" << kb.step_frac
            << (kb.corana ? " (Corana V half-width/width)"
                          : " (CLASSIC Gaussian sigma/width)");
      }
      oss << "  BOUNDARY_MODE=" << static_cast<int>(kb.boundary_mode)
          << (kb.boundary_mode == anneal::BoundaryMode::Reflect
                  ? "(REFLECT)" : "(REDRAW)")
          << "  INIT_MODE=" << kb.init_mode
          << (kb.init_mode == 0
                  ? (kb.corana ? "(Gaussian jitter; sigma=initial V)"
                               : "(Gaussian jitter; sigma=proposal sigma)")
                  : "(uniform box)");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Layout arithmetic -- exact, printed up front: with a large
     * NT*NS*dim product the first per-epoch line can be hours away, and
     * the epoch size is what sets both the swap and journal cadence. */
    const std::uint64_t rung_cost =
        kb.corana ? anneal::saturating_mul(
                        anneal::saturating_mul(kb.NT, kb.NS), dim)
                  : static_cast<std::uint64_t>(kb.L);
    const std::uint64_t epoch_cost =
        anneal::saturating_mul(static_cast<std::uint64_t>(K), rung_cost);
    {
      std::ostringstream oss;
      oss << "ANNEAL_PT layout: rung = ";
      if (kb.corana)
        oss << "NT*NS*dim = " << kb.NT << "*" << kb.NS << "*" << dim << " = ";
      else
        oss << "L = ";
      oss << rung_cost << " evals/replica;  epoch = REPLICAS*rung = "
          << K << "*" << rung_cost << " = " << epoch_cost
          << " evals;  exchange attempt every " << kb.swap_every
          << " epoch(s)";
      HPOLog::instance().info(oss.str());
      /* Cadence registration is single-process only: the milestone
       * throughput window is rank-local and would mis-scale a global
       * budget under MPI. */
      if (world == 1 && epoch_cost > 0)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(epoch_cost), "epoch",
            max_eval_global == HUGE_UNSIGNED ? 0 : max_eval_global);
    }
    /* Pinned-dim audit (single-process like the per-epoch report --
     * marks are rank-local, a rank-0 table would miss other ranks). */
    if (world == 1 && kb.corana)
      HPOLog::instance().arm_pin_audit(
          sp, kb.v_floor_frac, static_cast<int>(K), "epoch");

    bool calib_threw = false;
    std::string calib_what;
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
          } catch (const std::exception& e) {
            calib_threw = true;
            calib_what = e.what();
          } catch (...) {
            calib_threw = true;
            calib_what = "TADAH:ANNEAL_PT: non-standard exception during zero-dimensional evaluation";
          }
        }
      }
    }

    bool proceed = (stop_reason == StopReason::Done);

    /* --- seed evaluation + T0 calibration --------------------------------
     * Rank 0 evaluates the seed and runs the probe calibration; the
     * outcome (T_hot, f0, x0, or an abort code) is broadcast so every
     * rank proceeds from identical state without duplicate probe evals. */
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
        } catch (const std::exception& e) {
          calib_threw = true;
          calib_what = e.what();
          proceed = false;
        } catch (...) {
          calib_threw = true;
          calib_what =
              "TADAH:ANNEAL_PT: non-standard exception during seed evaluation";
          proceed = false;
        }
      }
    }

    if (proceed && rank == 0) {
      std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);
      try {
        const auto t0out = anneal::calibrate_t0(
            sp, kb, proposal_scale, x0, f0, l_best_x, l_best,
            host, probe_rng, ctx, "ANNEAL_PT");
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
      } catch (const std::exception& e) {
        calib_threw = true;
        calib_what  = e.what();
      } catch (...) {
        calib_threw = true;
        calib_what =
            "TADAH:ANNEAL_PT: non-standard exception during T0 calibration";
      }
    }
    if (world > 1) {
      /* Unconditional handshake: every rank participates, so a rank-0
       * abort can never leave the others hanging on the broadcast. */
      int code = 0;
      if (rank == 0)
        code = calib_threw ? -1
                           : (proceed ? 0 : 1 + static_cast<int>(stop_reason));
      hpo_bcast(code, 0);
      if (code == -1) {
        if (rank == 0) throw std::runtime_error(calib_what);
        throw std::runtime_error(
            "TADAH:ANNEAL_PT: rank 0 aborted during seed/T0 calibration");
      }
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
      throw std::runtime_error(calib_what);
    }
    T_cold = kb.t_cold ? *kb.t_cold : T_hot / kb.ladder_span;
    if (proceed && !(T_cold > 0.0 && T_cold < T_hot))
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved ladder is invalid (T_COLD=" +
          std::to_string(T_cold) + ", T_HOT=" + std::to_string(T_hot) +
          "); set PARAM T_COLD/T_HOT or LADDER_SPAN consistently");

    /* Hybrid cooling floor (only meaningful when RT < 1). */
    const double t_min = kb.t_min ? *kb.t_min
                                  : (T_hot > 0.0
                                         ? anneal::automatic_t_min(T_hot)
                                         : 1e-12);
    if (proceed && t_min > T_hot)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved T_MIN must be <= resolved T_HOT");

    /* --- ladder + chains ----------------------------------------------
     * Every rank builds the FULL temperature ladder (identical, cheap)
     * but only its own contiguous slot block of chains. Chain RNG
     * streams are keyed by the GLOBAL slot index. */
    std::vector<double> T(K, 0.0);
    long double log_t_cold = 0.0L, log_t_hot = 0.0L;
    if (proceed) {
      log_t_cold = std::log(static_cast<long double>(T_cold));
      log_t_hot  = std::log(static_cast<long double>(T_hot));
      for (std::size_t k = 0; k < K; ++k) {
        if (k == 0) T[k] = T_cold;
        else if (k + 1 == K) T[k] = T_hot;
        else {
          const long double u = static_cast<long double>(k) /
                                static_cast<long double>(K - 1);
          T[k] = static_cast<double>(
              std::exp(log_t_cold + (log_t_hot - log_t_cold) * u));
        }
      }
    }

    /* Resolved-ladder report: T0 calibration fixes only the HOT end; the
     * cold end and the per-slot geometry are decided here, so this is
     * the first point the full ladder is known (also fires when a
     * user-set T_HOT skipped the calibration entirely). */
    if (rank == 0 && proceed) {
      {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "ANNEAL_PT ladder: T_HOT=" << T_hot
            << "  T_COLD=" << T_cold
            << (kb.t_cold ? " (PARAM T_COLD)" : " (T_HOT/LADDER_SPAN)")
            << "  geometric over " << K << " slots, ratio="
            << std::fixed << std::setprecision(3)
            << std::exp((log_t_hot - log_t_cold) /
                        static_cast<long double>(K - 1))
            << " per rung";
        if (kb.rt < 1.0)
          oss << "  [initial; cools x" << kb.rt
              << " per epoch, floor T_MIN="
              << std::scientific << std::setprecision(3) << t_min << "]";
        else
          oss << "  [static ladder (RT=1)]";
        HPOLog::instance().info(oss.str());
      }
      /* Per-slot table: [lo-hi] range rows of 8 (HPOLog::event tags and
       * indents continuation lines). The per-epoch report and the
       * end-of-run new-best histogram mirror this grid so their values
       * map onto temperatures by eye. */
      {
        const int iw = static_cast<int>(std::to_string(K - 1).size());
        std::ostringstream oss;
        oss << "ANNEAL_PT ladder T per slot (0=coldest .. " << (K - 1)
            << "=hottest):";
        for (std::size_t lo = 0; lo < K; lo += 8) {
          const std::size_t hi = std::min(lo + 8, K) - 1;
          oss << "\n"
              << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
              << "]" << std::scientific << std::setprecision(3);
          for (std::size_t k = lo; k <= hi; ++k) oss << "  " << T[k];
        }
        HPOLog::instance().info(oss.str());
      }
    }

    std::vector<anneal::ChainState<Vec>> chains(n_local);
    anneal::ReplicaPool pool(std::min(threads, n_local ? n_local : 1),
                             stop_flag);
    auto rethrow_pool_error = [&](const char* phase) {
      const bool local_error = pool.has_error();
      const int any_error = world > 1
          ? hpo_allreduce_max(local_error ? 1 : 0)
          : (local_error ? 1 : 0);
      if (!any_error) return;
      if (local_error) pool.rethrow_if_error();
      throw std::runtime_error(
          std::string("TADAH:ANNEAL_PT: worker failed on another MPI rank during ") +
          phase);
    };

    if (proceed) {
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        auto& c = chains[li];
        c.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, k);
        c.V   = proposal_scale;
        c.x   = Vec(dim);
        if (k == 0 || kb.init_mode == 0) {
          /* Chain 0 starts at the seed exactly. Higher slots use Gaussian
           * seed jitter in both variants; under Corana the initial V[i]
           * value is reused as that initialization-only sigma. */
          std::normal_distribution<double> nd(0.0, 1.0);
          for (std::size_t i = 0; i < dim; ++i) {
            long double v = static_cast<long double>(x0(i));
            if (k > 0) {
              auto draw = [&]() {
                return static_cast<long double>(x0(i)) +
                       static_cast<long double>(proposal_scale[i]) * nd(c.rng);
              };
              v = draw();
              (void)anneal::constrain_proposal(
                  v, sp.low[i], sp.high[i], kb.boundary_mode, draw);
            }
            c.x(i) = static_cast<double>(v);
          }
        } else {
          for (std::size_t i = 0; i < dim; ++i)
            c.x(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(c.rng);
        }
        c.best_x = c.x;
      }
      if (rank == 0 && n_local > 0) {
        chains[0].f      = f0;
        chains[0].best_f = f0;
      }

      /* Evaluate a deterministic global prefix of non-seed starting points.
       * This makes partial initialization identical for every rank/thread
       * layout and prevents MPI ranks collectively overshooting MAXEVAL. */
      const std::uint64_t spent_before_init = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t init_remaining = max_eval_global > spent_before_init
          ? max_eval_global - spent_before_init : 0;
      const std::uint64_t init_admitted = std::min<std::uint64_t>(
          K > 0 ? static_cast<std::uint64_t>(K - 1) : 0, init_remaining);
      pool.run_epoch(n_local, [&](std::size_t li) {
        /* key this chain's warm-start by its GLOBAL slot (off_local+li) so
         * results are independent of which worker runs it. */
        const std::size_t gk = off_local + li;
        warmstart::current_chain() = gk;
        HPOLog::set_eval_slot(static_cast<int>(gk), K, T[gk]);
        if (rank == 0 && li == 0) return;   /* global slot 0 = seed */
        if (gk == 0 || gk > init_admitted) return;
        auto& c = chains[li];
        if (stop_flag.load(std::memory_order_relaxed)) return;
        if (!ctx.budget.try_reserve()) return;   /* f stays inf */
        c.f = host(sp.out(c.x));
        if (!std::isfinite(c.f))
          c.f = std::numeric_limits<double>::infinity();
        c.best_f = c.f;
        c.best_x = c.x;
        if (anneal::stopval_reached(c.f, stopval, sp.failscore))
          stop_flag.store(true, std::memory_order_relaxed);
      });
      rethrow_pool_error("population initialization");
      for (std::size_t li = 0; li < n_local; ++li) {
        if (chains[li].best_f < l_best) {
          l_best   = chains[li].best_f;
          l_best_x = chains[li].best_x;
        }
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        double gb = std::isfinite(l_best) ? l_best
                                          : std::numeric_limits<double>::max();
        int owner = 0;
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

    /* Per-pair swap-acceptance bookkeeping. */
    std::vector<std::size_t> swap_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept (K > 1 ? K - 1 : 0, 0);
    /* SWAP_WINDOW-epoch window of the same counters (reset after each
     * windowed table) -- shows acceptance DRIFT the cumulative table
     * averages away. */
    std::vector<std::size_t> swap_attempt_win(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept_win (K > 1 ? K - 1 : 0, 0);

    /* Ladder-flow bookkeeping (Katzgraber-style up/down tagging),
     * single-process runs only -- walker identities would have to ride
     * the MPI boundary-swap messages to stay exact across ranks.
     * walker_at[slot] tracks the STATE lineage (swaps move states, not
     * slots); each walker is tagged by the extreme it visited last.
     * A +1-tagged walker reaching the hot end = one genuine cold->hot
     * traversal (and vice versa), so the counts are exact -- no
     * initial-transient overcount. min(up, down) ~ completed round
     * trips, the literature's tempering-effectiveness currency. */
    std::vector<std::size_t> walker_at(K);
    std::vector<signed char> walker_tag(K, 0);  /* 0 none, +1 cold, -1 hot */
    std::size_t flow_up = 0, flow_down = 0;     /* cold->hot, hot->cold */
    for (std::size_t i = 0; i < K; ++i) walker_at[i] = i;
    auto flow_touch_extremes = [&]() {
      if (world != 1 || K < 2) return;
      {
        const std::size_t w = walker_at[0];          /* coldest slot */
        if (walker_tag[w] == -1) ++flow_down;
        walker_tag[w] = +1;
      }
      {
        const std::size_t w = walker_at[K - 1];      /* hottest slot */
        if (walker_tag[w] == +1) ++flow_up;
        walker_tag[w] = -1;
      }
    };
    flow_touch_extremes();   /* initial tags only: all tags are still 0 */

    /* --- epoch loop ------------------------------------------------------
     * Fixed per-epoch collective order (every rank, every epoch):
     * [allreduce spent] -> local rungs -> [allreduce stats+stopcode] ->
     * boundary sendrecvs -> [minloc best] -> [bcast rank-0 stop code]
     * (-> [bcast best-x] when RESET_TO_BEST). Ranks therefore always
     * agree on loop exits and no collective can be left unmatched. */
    if (proceed) {
      std::vector<anneal::RungStats> rstats(n_local);
      std::uniform_real_distribution<double> udist(0.0, 1.0);
      /* Planned total for the epoch=current/total display: min of the
       * budget bound (all replicas rung in lockstep) and, in hybrid
       * mode (RT < 1), the ladder's cooling bound on the hot end.
       * 0 = unbounded (pure PT with no MAXEVAL) -> plain counter. */
      std::size_t planned_epochs = 0;
      for (;;) {
        const auto epoch_wall_t0 = std::chrono::steady_clock::now();
        const std::uint64_t spent_global =
            world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t remaining =
            max_eval_global > spent_global ? max_eval_global - spent_global : 0;
        if (remaining == 0) { stop_reason = StopReason::MaxEval; break; }
        if (planned_epochs == 0 &&
            (max_eval_global != HUGE_UNSIGNED || kb.rt < 1.0)) {
          planned_epochs = anneal::planned_rungs(
              T[K - 1], t_min, kb.rt,
              max_eval_global == HUGE_UNSIGNED
                  ? std::numeric_limits<std::uint64_t>::max()
                  : remaining,
              epoch_cost);
          /* Plan line -- printed once, from the same arithmetic that
           * feeds the epoch=k/total display (an upper bound: STOPVAL /
           * plateau / MAXTIME end the run earlier). */
          if (rank == 0 && planned_epochs > 0) {
            const std::size_t rounds = planned_epochs / kb.swap_every +
                (planned_epochs % kb.swap_every != 0 ? 1 : 0);
            std::ostringstream oss;
            oss << "ANNEAL_PT plan: ~" << planned_epochs << " epoch(s) (";
            if (max_eval_global != HUGE_UNSIGNED)
              oss << "budget left " << remaining << " evals";
            if (max_eval_global != HUGE_UNSIGNED && kb.rt < 1.0) oss << ", ";
            if (kb.rt < 1.0) oss << "hot-end cooling to T_MIN";
            oss << "; upper bound), ~" << rounds << " exchange round(s)";
            HPOLog::instance().info(oss.str());
            /* Round-trip feasibility gates. The floor is STRUCTURAL: a
             * DEO exchange round moves a state at most one slot, so one
             * cold<->hot round trip needs >= 2*(K-1) rounds even at
             * 100% swap acceptance -- below that, a complete round trip is
             * structurally impossible (WARNING). Above it, the common 25%
             * planning approximation gives 2*(K-1)/0.25 rounds per nominal
             * trip; it is reported only as a heuristic HINT, while measured
             * ladder-flow counts remain authoritative. */
            if (K >= 2) {
              const std::size_t rt_floor = 2 * (K - 1);
              if (rounds < rt_floor) {
                std::ostringstream w;
                w << "ANNEAL_PT: planned exchange rounds " << rounds
                  << " < 2*(REPLICAS-1) = " << rt_floor
                  << " -- one cold<->hot ladder round trip is impossible "
                     "even at 100% swap acceptance (a DEO round moves a "
                     "state at most one slot). Exchanges can still couple "
                     "adjacent slots, but no lineage can complete a round trip. ";
                if (kb.corana)
                  w << "Reduce NS or NT (epoch = REPLICAS*NT*NS*dim evals) ";
                else
                  w << "Reduce L (epoch = REPLICAS*L evals) ";
                w << "or raise MAXEVAL.";
                HPOLog::instance().warning(w.str());
              } else if (rounds < 4 * rt_floor) {
                std::ostringstream h;
                h << "ANNEAL_PT: planned exchange rounds " << rounds
                  << " < 2*(REPLICAS-1)/0.25 = " << 4 * rt_floor
                  << " -- the 25%-acceptance planning heuristic yields fewer "
                     "than ~1 nominal ladder round trip (literature reference "
                     "values: Kone-Kofke 23%, Predescu 39% for optimization). "
                     "Measured ladder-flow counts are authoritative. ";
                if (kb.corana)
                  h << "Consider smaller NS/NT or larger MAXEVAL.";
                else
                  h << "Consider smaller L or larger MAXEVAL.";
                HPOLog::instance().hint(h.str());
              }
            }
          }
        }
        int pre_time = time_exceeded() ? 1 : 0;
        if (world > 1) pre_time = hpo_allreduce_max(pre_time);
        if (pre_time) {
          stop_reason = StopReason::MaxTime; break;
        }

        /* Deterministic per-epoch budget slicing: full rungs while the
         * budget allows, else floor(remaining/K) with the remainder
         * given to the coldest slots. Computed identically on every
         * rank from the GLOBAL remaining count. */
        std::vector<std::uint64_t> allowance(K, rung_cost);
        if (remaining < epoch_cost) {
          const std::uint64_t base = remaining / K;
          const std::uint64_t rem  = remaining % K;
          for (std::size_t k = 0; k < K; ++k)
            allowance[k] = base + (k < rem ? 1 : 0);
        }

        /* f at epoch start (post previous exchange), for the per-epoch
         * replica-state report's df column. */
        std::vector<double> f_start(n_local);
        for (std::size_t li = 0; li < n_local; ++li)
          f_start[li] = chains[li].f;

        pool.run_epoch(n_local, [&](std::size_t li) {
          const std::size_t k = off_local + li;
          warmstart::current_chain() = k;   /* global slot for warm-start key */
          HPOLog::set_eval_slot(static_cast<int>(k), K, T[k]);
          rstats[li] = anneal::run_rung(chains[li], T[k], sp, kb, host, ctx,
                                        allowance[k]);
        });
        rethrow_pool_error("replica rung");

        std::size_t e_prop = 0, e_acc = 0, e_oob = 0;
        anneal::StepSaturation e_vsat;
        bool any_stopval = false, any_time = false;
        for (std::size_t li = 0; li < n_local; ++li) {
          e_prop += rstats[li].proposed;
          e_acc  += rstats[li].accepted;
          e_oob  += rstats[li].oob;
          if (rstats[li].stop == RungStop::StopVal) any_stopval = true;
          if (rstats[li].stop == RungStop::Time)    any_time    = true;
          if (chains[li].best_f < l_best) {
            l_best   = chains[li].best_f;
            l_best_x = chains[li].best_x;
          }
          if (kb.corana) {
            const auto s = anneal::step_saturation(chains[li].V, sp, kb);
            e_vsat.floor += s.floor;
            e_vsat.cap   += s.cap;
            e_vsat.total += s.total;
          }
        }
        g_best = std::min(g_best, l_best);
        any_time = any_time || time_exceeded();
        if (world > 1) {
          e_prop = hpo_allreduce_sum_sz(e_prop);
          e_acc  = hpo_allreduce_sum_sz(e_acc);
          e_oob  = hpo_allreduce_sum_sz(e_oob);
          if (kb.corana) {
            e_vsat.floor = hpo_allreduce_sum_sz(e_vsat.floor);
            e_vsat.cap   = hpo_allreduce_sum_sz(e_vsat.cap);
            e_vsat.total = hpo_allreduce_sum_sz(e_vsat.total);
          }
          int code = any_stopval ? 2 : (any_time ? 1 : 0);
          code = hpo_allreduce_max(code);
          any_stopval = (code == 2);
          any_time    = (code == 1);
        }
        total_proposed += e_prop;
        total_accepted += e_acc;
        total_oob      += e_oob;
        ++epoch_count;
        const bool maxeval_hit = max_eval_global != HUGE_UNSIGNED &&
                                 e_prop >= remaining;
        if (any_stopval)       stop_reason = StopReason::StopVal;
        else if (any_time)     stop_reason = StopReason::MaxTime;
        else if (e_prop == 0 || maxeval_hit)
                               stop_reason = StopReason::MaxEval;
        const bool terminal_epoch = stop_reason != StopReason::Done;

        /* Pre-swap snapshot for the per-epoch replica-state report: the
         * exchange below permutes x/f, while the report describes what
         * each slot's rung actually did. (The walker row is deliberately
         * POST-swap -- walkers only move at exchanges, so the pre-swap
         * occupancy is last epoch's post-swap row.) Single-process only
         * (chains are rank-local), same as the report itself. */
        std::vector<double> f_pre, f_delta;
        if (world == 1) {
          f_pre.resize(n_local);
          f_delta.resize(n_local);
          for (std::size_t li = 0; li < n_local; ++li) {
            f_pre[li]   = chains[li].f;
            f_delta[li] = chains[li].f - f_start[li];
          }
          /* Pinned-dim audit refresh: PRE-swap (slot V + the x that
           * walked in the slot this epoch -- the exchange permutes x/f,
           * never V). A slot whose x left the bound is unmarked; the
           * table prints with the milestone SUMMARY. */
          if (kb.corana)
            for (std::size_t li = 0; li < n_local; ++li)
              HPOLog::instance().record_pin_state(
                  static_cast<int>(li), epoch_count,
                  anneal::pinned_dims(chains[li], sp, kb));
        }
        /* Per-pair outcome of THIS epoch's exchange round for the
         * events row: 'x' swapped, '.' rejected, ' ' not attempted
         * (DEO parity). */
        std::vector<char> pair_event(K > 1 ? K - 1 : 0, ' ');

        /* --- exchange phase (deterministic; per-pair coin streams so
         * the decision is identical no matter which rank(s) own the
         * pair or how many ranks there are) --------------------------- */
        double ep_swap_rate = -1.0;
        if (!terminal_epoch && epoch_count % kb.swap_every == 0) {
          const std::size_t parity =
              (epoch_count / kb.swap_every) % 2;  /* DEO alternation */
          std::size_t att = 0, acc = 0;
          for (std::size_t i = parity; i + 1 < K; i += 2) {
            const bool i_mine = (i >= off_local && i < off_local + n_local);
            const bool j_mine = (i + 1 >= off_local && i + 1 < off_local + n_local);
            if (!i_mine && !j_mine) continue;
            std::mt19937_64 pair_rng = anneal::make_stream(
                seed, anneal::SALT_PT_SWAP,
                static_cast<std::uint64_t>(epoch_count) * K + i);
            if (i_mine && j_mine) {
              auto& ci = chains[i - off_local];
              auto& cj = chains[i + 1 - off_local];
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], ci.f, cj.f,
                  [&]() { return udist(pair_rng); });
              ++swap_attempt[i];
              ++swap_attempt_win[i];
              ++att;
              pair_event[i] = '.';
              if (accepted) {
                std::swap(ci.x, cj.x);
                std::swap(ci.f, cj.f);
                if (world == 1) std::swap(walker_at[i], walker_at[i + 1]);
                ++swap_accept[i];
                ++swap_accept_win[i];
                ++acc;
                pair_event[i] = 'x';
              }
            } else {
              /* Boundary pair: I own exactly one side; my peer owns the
               * other. Both sides use the SAME pair coin stream when the
               * decision is probabilistic and compute the SAME result -- only f (and x
               * on accept) travel. */
              const std::size_t mine_g = i_mine ? i : i + 1;
              auto& c = chains[mine_g - off_local];
              const int peer = i_mine ? rank + 1 : rank - 1;
              std::vector<double> sendf{c.f}, recvf(1, 0.0);
              hpo_sendrecv(sendf, recvf, peer, static_cast<int>(i % 1024) * 2);
              const double f_i = i_mine ? c.f : recvf[0];
              const double f_j = i_mine ? recvf[0] : c.f;
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], f_i, f_j,
                  [&]() { return udist(pair_rng); });
              if (i_mine) {
                ++swap_attempt[i];
                ++swap_attempt_win[i];
                ++att;
                if (accepted) {
                  ++swap_accept[i];
                  ++swap_accept_win[i];
                  ++acc;
                }
              }
              if (accepted) {
                std::vector<double> sx(dim), rx(dim, 0.0);
                for (std::size_t d = 0; d < dim; ++d) sx[d] = c.x(d);
                hpo_sendrecv(sx, rx, peer, static_cast<int>(i % 1024) * 2 + 1);
                for (std::size_t d = 0; d < dim; ++d) c.x(d) = rx[d];
                c.f = i_mine ? f_j : f_i;
              }
            }
          }
          if (world > 1) {
            att = hpo_allreduce_sum_sz(att);
            acc = hpo_allreduce_sum_sz(acc);
          }
          ++exchange_rounds;
          ep_swap_rate = att > 0 ? static_cast<double>(acc) /
                                   static_cast<double>(att) : 0.0;
          flow_touch_extremes();   /* states moved -- update up/down flow */
        }

        /* --- global best + rank-0 stop decisions ----------------------
         * Ownership resolved from the rank-LOCAL best (see l_best). */
        int best_owner = 0;
        if (world > 1) {
          double gb = std::isfinite(l_best) ? l_best
                                            : std::numeric_limits<double>::max();
          hpo_allreduce_minloc(gb, best_owner);
          if (gb < std::numeric_limits<double>::max())
            g_best = std::min(g_best, gb);
        }

        int rank0_code = 0;   /* 0 none, 1 plateau */
        if (rank == 0 && !terminal_epoch) {
          if (plateau.update(chains[0].f, g_best)) rank0_code = 1;
        }
        if (world > 1) hpo_bcast(rank0_code, 0);
        const bool plateau_hit = rank0_code == 1;

        /* --- optional ladder cooling (PT -> SA hybrid) ---------------- */
        bool tmin_hit = false;
        if (!terminal_epoch && !plateau_hit && kb.rt < 1.0) {
          for (auto& t : T) t = anneal::cool_temperature(t, kb.rt);
          tmin_hit = T[K - 1] < t_min;
        }

        /* --- reset-to-best: coldest slot only ------------------------- */
        if (!terminal_epoch && !plateau_hit && kb.reset_to_best) {
          std::vector<double> bx(dim, 0.0);
          if (world > 1) {
            if (rank == best_owner)
              for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
            hpo_bcast(bx, best_owner);
          } else {
            for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
          }
          if (rank == 0 && n_local > 0 && chains[0].f > g_best) {
            for (std::size_t d = 0; d < dim; ++d) chains[0].x(d) = bx[d];
            chains[0].f = g_best;
            chains[0].best_x = chains[0].x;
            chains[0].best_f = g_best;
          }
        }

        /* --- one journal line per epoch (rank 0 carries the global
         * numbers; other ranks stay quiet) ----------------------------- */
        if (rank == 0) {
          const double rate = e_prop > 0
              ? static_cast<double>(e_acc) / static_cast<double>(e_prop) : 0.0;
          const double oob_pct = e_prop > 0
              ? 100.0 * static_cast<double>(e_oob) /
                    static_cast<double>(e_prop)
              : 0.0;
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PT epoch=" << epoch_count;
          if (planned_epochs > 0) oss << "/" << planned_epochs;
          oss << "  T=[" << T[0] << ", " << T[K - 1] << "]"
              << "  proposed=" << e_prop
              << "  accepted=" << std::fixed << std::setprecision(1)
              << (rate * 100.0) << "%"
              << "  oob=" << e_oob << " (" << oob_pct << "%)";
          if (ep_swap_rate >= 0.0)
            oss << "  swap_acc=" << (ep_swap_rate * 100.0) << "%";
          oss << std::scientific << std::setprecision(3)
              << "  f_cold=" << chains[0].f
              << "  best=" << g_best;
          if (kb.corana)
            oss << "  V_floor=" << e_vsat.floor << "/" << e_vsat.total
                << "  V_cap=" << e_vsat.cap << "/" << e_vsat.total;
          oss
              << "  epoch_wall=" << HPOLog::format_duration(
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - epoch_wall_t0)
                         .count());
          HPOLog::instance().info(oss.str());
          /* Per-pair swap-acceptance tables. DEO parity gives each pair
           * at most one attempt per 2*SWAP_EVERY epochs, so per-epoch
           * per-pair rates would be 0/100% noise -- the CUMULATIVE table
           * (every exchange epoch) is the live ladder diagnostic, and
           * the SWAP_WINDOW table (then reset) shows drift the
           * cumulative average hides. att/pair in the header states the
           * resolution; "-" = pair not attempted yet. Single-process
           * only: the counters are rank-local. */
          if (world == 1 && K >= 2 && ep_swap_rate >= 0.0) {
            const int iwp = static_cast<int>(std::to_string(K - 2).size());
            auto pair_table = [&](const std::string& label,
                                  const std::vector<std::size_t>& attv,
                                  const std::vector<std::size_t>& accv) {
              std::size_t amin = std::numeric_limits<std::size_t>::max();
              std::size_t amax = 0;
              for (std::size_t p = 0; p + 1 < K; ++p) {
                amin = std::min(amin, attv[p]);
                amax = std::max(amax, attv[p]);
              }
              std::ostringstream o;
              o << "ANNEAL_PT swap acc% per pair (" << label
                << ", att/pair " << amin;
              if (amax != amin) o << "-" << amax;
              o << "; pair p = slots p<->p+1):";
              for (std::size_t lo = 0; lo + 1 < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K - 1) - 1;
                o << "\n[" << std::setw(iwp) << lo << "-" << std::setw(iwp)
                  << hi << "]";
                for (std::size_t p = lo; p <= hi; ++p) {
                  if (attv[p] == 0)
                    o << "  " << std::setw(9) << "-";
                  else
                    o << "  " << std::setw(9)
                      << std::lround(100.0 *
                             static_cast<double>(accv[p]) /
                             static_cast<double>(attv[p]));
                }
              }
              HPOLog::instance().info(o.str());
            };
            pair_table("cumulative", swap_attempt, swap_accept);
            if (kb.swap_window > 0 && epoch_count % kb.swap_window == 0) {
              std::size_t att_tot = 0;
              for (auto a : swap_attempt_win) att_tot += a;
              if (att_tot > 0) {
                pair_table("last " + std::to_string(kb.swap_window) +
                           " epoch(s)", swap_attempt_win, swap_accept_win);
                std::fill(swap_attempt_win.begin(), swap_attempt_win.end(),
                          std::size_t{0});
                std::fill(swap_accept_win.begin(), swap_accept_win.end(),
                          std::size_t{0});
              }
            }
            /* THIS epoch's exchange outcomes -- the walker motion itself,
             * readable without diffing walker rows across epochs. */
            {
              std::ostringstream o;
              o << "ANNEAL_PT exchange this epoch "
                   "(x=swapped, .=rejected, blank=other parity):";
              for (std::size_t lo = 0; lo + 1 < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K - 1) - 1;
                o << "\n[" << std::setw(iwp) << lo << "-" << std::setw(iwp)
                  << hi << "]";
                for (std::size_t p = lo; p <= hi; ++p)
                  o << "  " << std::setw(9) << pair_event[p];
              }
              HPOLog::instance().info(o.str());
            }
          }
          /* Sparse per-slot new-best histogram for THIS epoch (chain-init
           * bests land in epoch 1's window), as its own tagged line.
           * Single-process only: the window counts are rank-local, so
           * under MPI a rank-0 line would silently miss the other ranks'
           * slots. */
          if (world == 1) {
            const auto win = HPOLog::instance().take_best_slot_window();
            if (!win.empty()) {
              std::ostringstream ossw;
              ossw << "ANNEAL_PT new bests per slot:";
              for (const auto& [s, n] : win) ossw << "  [" << s << "]:" << n;
              HPOLog::instance().info(ossw.str());
            }
          }
          /* --- per-epoch replica-state report (pre-swap snapshot) ------
           * One grid block per metric, on the same [lo-hi] rows and
           * column widths as the startup ladder table so slot columns
           * line up down the journal. f/df/imp/acc describe the rung
           * phase just run; best_f is slot-resident (never swapped);
           * walker is the state lineage that produced this slot's rung.
           * Single-process only, like the histogram above. */
          if (world == 1) {
            const int iw = static_cast<int>(std::to_string(K - 1).size());
            auto grid = [&](const std::string& head, auto&& cell) {
              std::ostringstream o;
              o << head;
              for (std::size_t lo = 0; lo < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K) - 1;
                o << "\n"
                  << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
                  << "]";
                for (std::size_t s = lo; s <= hi; ++s) cell(o, s);
              }
              HPOLog::instance().info(o.str());
            };
            grid("ANNEAL_PT f per slot (pre-swap):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::scientific << std::setprecision(3)
                     << std::setw(9) << f_pre[s];
                 });
            grid("ANNEAL_PT df per slot (epoch):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << " " << std::scientific << std::setprecision(3)
                     << std::showpos << std::setw(10) << f_delta[s]
                     << std::noshowpos;
                 });
            grid("ANNEAL_PT best_f per slot:",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::scientific << std::setprecision(3)
                     << std::setw(9) << chains[s].best_f;
                 });
            grid("ANNEAL_PT imp per slot (epoch, new personal bests):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::setw(9) << rstats[s].improved;
                 });
            grid("ANNEAL_PT acc% per slot (epoch):",
                 [&](std::ostringstream& o, std::size_t s) {
                   const std::size_t p = rstats[s].proposed;
                   const long pct = p ? std::lround(
                       100.0 * static_cast<double>(rstats[s].accepted) /
                       static_cast<double>(p)) : 0;
                   o << "  " << std::setw(9) << pct;
                 });
            /* POST-swap walker lineup (what the next epoch runs), each
             * tagged by the last ladder end it touched (Katzgraber
             * up/down labeling, from the flow bookkeeping): a 'v'
             * walker descending toward slot 0 is a hot-end discovery
             * being carried to the cold end. */
            grid("ANNEAL_PT walker per slot (post-swap = next epoch's "
                 "lineup; ^=up walker (last touched cold end), v=down "
                 "walker (last touched hot end), .=no extreme yet):",
                 [&](std::ostringstream& o, std::size_t s) {
                   const std::size_t w = walker_at[s];
                   const char t = walker_tag[w] > 0
                                      ? '^' : (walker_tag[w] < 0 ? 'v' : '.');
                   o << "  " << std::setw(8) << w << t;
                 });
            /* Running end-to-end communication scoreboard (the same
             * counters as the end-of-run ladder-flow line): 0 round
             * trips = the replicas are still effectively independent
             * annealers. */
            {
              std::ostringstream of;
              of << "ANNEAL_PT ladder flow: traversals cold->hot="
                 << flow_up << "  hot->cold=" << flow_down
                 << "  (~" << std::min(flow_up, flow_down)
                 << " round trip(s) so far)";
              HPOLog::instance().info(of.str());
            }
            /* Corana step-vector state per slot, normalised to the box:
             * r_i = V[i]/width[i], so every slot starts at STEP_FRAC and
             * lives in [V_FLOOR_FRAC, 1]. V is slot-resident (exchanges
             * swap x/f, never V), so each slot's step scale equilibrates
             * to its own temperature. CLASSIC never adapts V -- the
             * blocks would be a constant STEP_FRAC row, so skip them.
             * Emitted through the shared grid emitter -- HPOLog::event
             * stars the header and tags every row. */
            if (kb.corana) {
              std::vector<double> v_gm(K), v_mn(K), v_mx(K);
              for (std::size_t s = 0; s < K; ++s) {
                double slog = 0.0;
                double mn = std::numeric_limits<double>::infinity();
                double mx = 0.0;
                for (std::size_t i = 0; i < dim; ++i) {
                  const double r = chains[s].V[i] / (sp.high[i] - sp.low[i]);
                  slog += std::log(r);
                  mn = std::min(mn, r);
                  mx = std::max(mx, r);
                }
                v_gm[s] = std::exp(slog / static_cast<double>(dim));
                v_mn[s] = mn;
                v_mx[s] = mx;
              }
              auto vcell = [&](const std::vector<double>& v) {
                return [&](std::ostringstream& o, std::size_t s) {
                  o << "  " << std::scientific << std::setprecision(3)
                    << std::setw(9) << v[s];
                };
              };
              grid("ANNEAL_PT V geo-mean per slot "
                   "(geomean_i V[i]/width[i]; start=STEP_FRAC, "
                   "floor=V_FLOOR_FRAC, cap=1):", vcell(v_gm));
              grid("ANNEAL_PT V min per slot (min_i V[i]/width[i]):",
                   vcell(v_mn));
              grid("ANNEAL_PT V max per slot (max_i V[i]/width[i]):",
                   vcell(v_mx));
            }
          }
        }
        if (terminal_epoch) break;
        if (plateau_hit) { stop_reason = StopReason::Plateau; break; }
        if (tmin_hit)    { stop_reason = StopReason::TMin; break; }
      }
    }

    /* --- multi-rank finalisation ------------------------------------------
     * Broadcast the winning coordinates for the driver's global result.
     * Do not re-evaluate on rank 0: host-side best checkpoints are written
     * at discovery time, and a synthetic final evaluation would violate the
     * hard global MAXEVAL contract. */
    anneal::StepSaturation final_vsat;
    if (kb.corana) {
      for (const auto& c : chains) {
        const auto s = anneal::step_saturation(c.V, sp, kb);
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

    std::vector<std::size_t> swap_attempt_report = swap_attempt;
    std::vector<std::size_t> swap_accept_report  = swap_accept;
    if (world > 1) {
      for (std::size_t i = 0; i + 1 < K; ++i) {
        swap_attempt_report[i] = hpo_allreduce_sum_sz(swap_attempt[i]);
        swap_accept_report[i]  = hpo_allreduce_sum_sz(swap_accept[i]);
      }
    }

    /* --- final summary --------------------------------------------------- */
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
          << "ANNEAL_PT summary: epochs=" << epoch_count
          << "  replicas=" << K
          << "  proposed=" << total_proposed
          << "  accepted=" << std::fixed << std::setprecision(1)
          << (overall_rate * 100.0) << "%"
          << std::scientific << std::setprecision(3)
          << "  oob=" << total_oob << " (" << std::fixed
          << std::setprecision(1) << oob_pct << "%)"
          << std::scientific << std::setprecision(3)
          << "  T_hot=" << T_hot
          << "  T_cold=" << T_cold
          << "  best=" << g_best;
      if (kb.corana)
        oss << "  V_floor=" << final_vsat.floor << "/" << final_vsat.total
            << "  V_cap=" << final_vsat.cap << "/" << final_vsat.total;
      HPOLog::instance().info(oss.str());
      /* Per-pair swap-acceptance table -- THE diagnostic for tuning
       * LADDER_SPAN/REPLICAS (aim 20-40% per pair). */
      if (exchange_rounds > 0) {
        std::ostringstream ossp;
        ossp << "ANNEAL_PT swap acceptance per pair (cold->hot):";
        for (std::size_t i = 0; i + 1 < K; ++i) {
          const double r = swap_attempt_report[i] > 0
              ? 100.0 * static_cast<double>(swap_accept_report[i]) /
                        static_cast<double>(swap_attempt_report[i]) : 0.0;
          ossp << "  " << i << ":" << std::fixed << std::setprecision(0)
               << r << "%";
        }
        HPOLog::instance().info(ossp.str());
      }
      /* Per-slot new-best histogram over the whole run, on the same
       * [lo-hi] grid (and column widths) as the startup ladder table so
       * counts map onto temperatures by eye -- the direct answer to
       * "did the hot rungs ever contribute?". Single-process only: the
       * counts come from this rank's journal. */
      if (world == 1) {
        const auto tot = HPOLog::instance().best_slot_totals();
        if (!tot.empty()) {
          std::map<int, std::size_t> cnt(tot.begin(), tot.end());
          const int iw = static_cast<int>(std::to_string(K - 1).size());
          std::ostringstream ossb;
          ossb << "ANNEAL_PT new-best count per slot (0=coldest .. "
               << (K - 1) << "=hottest):";
          for (std::size_t lo = 0; lo < K; lo += 8) {
            const std::size_t hi = std::min(lo + 8, K) - 1;
            ossb << "\n"
                 << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
                 << "]";
            for (std::size_t k = lo; k <= hi; ++k) {
              const auto it = cnt.find(static_cast<int>(k));
              ossb << "  " << std::setw(9)
                   << (it != cnt.end() ? it->second : 0);
            }
          }
          HPOLog::instance().info(ossb.str());
        }
      }
      /* Measured ladder flow (single-process runs; see the bookkeeping
       * comment above). Exact counts of end-to-end traversals; 0 round
       * trips = the ladder never communicated end to end. */
      if (world == 1 && K >= 2 && exchange_rounds > 0) {
        const std::size_t trips = std::min(flow_up, flow_down);
        std::ostringstream ossf;
        ossf << "ANNEAL_PT ladder flow: traversals cold->hot=" << flow_up
             << "  hot->cold=" << flow_down;
        if (trips > 0)
          ossf << "  (~" << trips << " round trip(s) completed)";
        else
          ossf << "  (0 round trips -- the ladder never communicated "
                  "end to end)";
        HPOLog::instance().info(ossf.str());
      }
    }

    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::ExternalOnProbe: reason = "external stop requested during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T < T_MIN (hot end)"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule, coldest slot)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL_PT stopped: " << reason
          << " after " << spent_report << " host eval(s)"
          << " (" << epoch_count << " epoch(s), "
          << exchange_rounds << " exchange round(s))";
      HPOLog::instance().info(oss.str());
    }
  }

template<typename Vec, typename Runner>
  void run_anneal_pt(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pt_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo

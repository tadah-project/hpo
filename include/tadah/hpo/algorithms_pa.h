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
 * the effective sample size stays at target * N_finite -- zero extra
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
 * PA-specific PARAMs (shared stepper knobs as in ANNEAL):
 *   POPULATION   4*THREADS (>=16)  population size R (top-level knob)
 *   SWEEPS       1                 rungs per walker per T-step
 *   ESS_TARGET   0.5               adaptive-dbeta target; 0 = geometric
 *   T_HOT        auto              starting temperature
 *   T_MIN        auto (T0*1e-6)    final temperature
 *   RT           0.85              max cooling per step / geometric ratio
 *   INIT_MODE    0                 0: walker0=seed, rest sigma-jittered
 *                                  1: walkers>0 uniform in the box
 */
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/warmstart_cache.h>   // current_chain() bridge
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_parallel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tadah::hpo {

template<typename Vec, typename HostT>
  void run_anneal_pa_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::PA);

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

    /* per-dim sigma (transformed space) */
    std::vector<double> sigma(dim);
    if (sp.step.has_value()) {
      for (std::size_t i = 0; i < dim; ++i) sigma[i] = *sp.step;
    } else {
      for (std::size_t i = 0; i < dim; ++i)
        sigma[i] = kb.step_frac * (sp.high[i] - sp.low[i]);
    }

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval_global = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    /* Multi-rank runs enforce MAXEVAL through GLOBAL accounting (spent
     * counts allreduced per sweep round feed the deterministic
     * slicing), so the local reserve counter stays uncapped there. */
    anneal::SharedBudget budget(world > 1 ? HUGE_UNSIGNED : max_eval_global);

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    const std::function<bool()> time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const double stopval =
        sp.stop.stopval.value_or(-std::numeric_limits<double>::infinity());

    std::atomic<bool> stop_flag{false};
    anneal::StopCtx ctx{budget, stop_flag, time_exceeded, stopval};

    anneal::CoranaPlateau plateau(sp.stop, kb.neps);

    enum class StopReason {
      Done, MaxEval, MaxTime, StopVal, TMin, Plateau,
      StopValOnProbe, MaxTimeOnProbe, NoDims, NoBudget
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
          << "seed=" << seed
          << "  POPULATION=" << R
          << "  THREADS=" << threads
          << "  SWEEPS=" << kb.sweeps
          << "  ESS_TARGET=" << kb.ess_target
          << "  RT=" << kb.rt;
      if (kb.t_min) oss << "  T_MIN=" << *kb.t_min;
      else          oss << "  T_MIN=auto(T0*1e-6)";
      if (kb.corana)
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT << "  C=" << kb.C;
      else
        oss << "  L=" << kb.L;
      oss << "  INIT_MODE=" << kb.init_mode;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Layout arithmetic -- exact, printed up front (mirrors ANNEAL_PT:
     * the first per-step line can be a long time away when the
     * per-walker product is large). */
    const std::uint64_t rung_cost =
        (kb.corana ? static_cast<std::uint64_t>(kb.NT) * kb.NS * dim
                   : static_cast<std::uint64_t>(kb.L)) * kb.sweeps;
    {
      const std::uint64_t step_evals =
          static_cast<std::uint64_t>(R) * rung_cost;
      std::ostringstream oss;
      oss << "ANNEAL_PA layout: walker T-step = ";
      if (kb.corana)
        oss << "NT*NS*dim*SWEEPS = " << kb.NT << "*" << kb.NS << "*" << dim
            << "*" << kb.sweeps << " = ";
      else
        oss << "L*SWEEPS = " << kb.L << "*" << kb.sweeps << " = ";
      oss << rung_cost << " evals;  population T-step = POPULATION*rung = "
          << R << "*" << rung_cost << " = " << step_evals << " evals";
      HPOLog::instance().info(oss.str());
      /* Cadence registration is single-process only: the milestone
       * throughput window is rank-local and would mis-scale a global
       * budget under MPI. */
      if (world == 1 && step_evals > 0)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(step_evals), "T-step",
            max_eval_global == HUGE_UNSIGNED ? 0 : max_eval_global);
    }
    /* Pinned-dim audit (single-process, rank-local marks -- see PT). */
    if (world == 1 && kb.corana)
      HPOLog::instance().arm_pin_audit(
          sp, kb.v_floor_frac, static_cast<int>(R), "T-step");

    if (dim == 0) {
      if (!sp.optim_init.empty() && budget.left() > 0) {
        const auto v = sp.in(sp.optim_init.front());
        Vec xv(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) xv(i) = v[i];
        host(sp.out(xv));
      }
      stop_reason = StopReason::NoDims;
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
      if (!sp.optim_init.empty()) {
        const auto v = sp.in(sp.optim_init.front());
        for (std::size_t i = 0; i < v.size(); ++i) x0(i) = v[i];
      } else {
        for (std::size_t i = 0; i < dim; ++i)
          x0(i) = 0.5 * (sp.low[i] + sp.high[i]);
      }
      if (!budget.try_reserve() ||
          (world == 1 && max_eval_global == 0)) {
        stop_reason = StopReason::NoBudget;
        proceed = false;
      } else {
        f0 = host(sp.out(x0));
        ran_sa = true;
        g_best = l_best = f0;
        g_best_x = l_best_x = x0;
        if (f0 <= stopval) {
          stop_reason = StopReason::StopVal; proceed = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTime; proceed = false;
        }
      }
    }

    bool calib_threw = false;
    std::string calib_what;
    if (proceed && rank == 0) {
      std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);
      try {
        const auto t0out = anneal::calibrate_t0(
            sp, kb, sigma, x0, f0, l_best_x, l_best,
            host, probe_rng, ctx, "ANNEAL_PA");
        g_best = l_best;
        T_hot = t0out.T0;
        if (t0out.aborted) {
          stop_reason = t0out.stopval_hit ? StopReason::StopValOnProbe
                                          : StopReason::MaxTimeOnProbe;
          proceed = false;
        }
      } catch (const std::exception& e) {
        calib_threw = true;
        calib_what  = e.what();
      }
    }
    if (world > 1) {
      int code = 0;
      if (rank == 0)
        code = calib_threw ? -1
                           : (proceed ? 0 : 1 + static_cast<int>(stop_reason));
      hpo_bcast(code, 0);
      if (code == -1) {
        if (rank == 0) throw std::runtime_error(calib_what);
        throw std::runtime_error(
            "TADAH:ANNEAL_PA: rank 0 aborted during seed/T0 calibration");
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

    const double t_min = kb.t_min ? *kb.t_min
                                  : (T_hot > 0.0 ? T_hot * 1e-6 : 1e-12);

    /* --- population (local slot block only) ------------------------------ */
    std::vector<anneal::ChainState<Vec>> walkers(n_local);
    anneal::ReplicaPool pool(std::min(threads, n_local ? n_local : 1),
                             stop_flag);

    /* Per-rank element counts for the population-wide gathers. */
    std::vector<int> f_counts(world), st_counts(world);
    const std::size_t stride = 2 * dim + 1;   /* f | x | V per walker */
    for (int r = 0; r < world; ++r) {
      f_counts[r]  = static_cast<int>(slot_count(r));
      st_counts[r] = static_cast<int>(slot_count(r) * stride);
    }

    if (proceed) {
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        auto& w = walkers[li];
        w.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, k);
        w.V   = sigma;
        w.x   = Vec(dim);
        if (k == 0 || kb.init_mode == 0) {
          std::normal_distribution<double> nd(0.0, 1.0);
          for (std::size_t i = 0; i < dim; ++i) {
            double v = x0(i);
            if (k > 0) v += sigma[i] * nd(w.rng);
            anneal::reflect_into(v, sp.low[i], sp.high[i]);
            w.x(i) = v;
          }
        } else {
          for (std::size_t i = 0; i < dim; ++i)
            w.x(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(w.rng);
        }
        w.best_x = w.x;
      }
      if (rank == 0 && n_local > 0) {
        walkers[0].f      = f0;
        walkers[0].best_f = f0;
      }

      pool.run_epoch(n_local, [&](std::size_t li) {
        warmstart::current_chain() = off_local + li;   /* global slot key */
        if (rank == 0 && li == 0) return;   /* global slot 0 = seed */
        auto& w = walkers[li];
        if (stop_flag.load(std::memory_order_relaxed)) return;
        if (!ctx.budget.try_reserve()) return;   /* f stays inf */
        w.f = host(sp.out(w.x));
        w.best_f = w.f;
        w.best_x = w.x;
        if (w.f <= stopval) stop_flag.store(true, std::memory_order_relaxed);
      });
      pool.rethrow_if_error();
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
        int code = (stop_flag.load(std::memory_order_relaxed) &&
                    g_best <= stopval) ? 1 : 0;
        code = hpo_allreduce_max(code);
        if (code) { stop_reason = StopReason::StopVal; proceed = false; }
      } else if (stop_flag.load(std::memory_order_relaxed) &&
                 g_best <= stopval) {
        stop_reason = StopReason::StopVal;
        proceed = false;
      }
    }

    /* One T-step of sweeps for the whole population (budget-sliced,
     * deterministic; slicing computed from the GLOBAL remaining budget
     * identically on every rank -- see the PT driver). */
    std::vector<anneal::RungStats> rstats(n_local);
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
      if (remaining < static_cast<std::uint64_t>(R) * rung_cost) {
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
      pool.rethrow_if_error();

      /* Pinned-dim audit refresh: post-sweeps, PRE-resampling (marks
       * follow what each walker's own sweeps did; resampling replaces
       * walkers and simply re-marks/unmarks on the next T-step). */
      if (world == 1 && kb.corana) {
        ++pin_unit;
        for (std::size_t li = 0; li < n_local; ++li)
          HPOLog::instance().record_pin_state(
              static_cast<int>(off_local + li), pin_unit,
              anneal::pinned_dims(walkers[li], sp, kb));
      }

      std::size_t e_prop = 0;
      bool any_stopval = false, any_time = false;
      for (std::size_t li = 0; li < n_local; ++li) {
        e_prop += rstats[li].proposed;
        total_proposed += rstats[li].proposed;
        total_accepted += rstats[li].accepted;
        total_oob      += rstats[li].oob;
        if (rstats[li].stop == RungStop::StopVal) any_stopval = true;
        if (rstats[li].stop == RungStop::Time)    any_time    = true;
        if (walkers[li].best_f < l_best) {
          l_best   = walkers[li].best_f;
          l_best_x = walkers[li].best_x;
        }
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        e_prop = hpo_allreduce_sum_sz(e_prop);
        int code = any_stopval ? 2 : (any_time ? 1 : 0);
        code = hpo_allreduce_max(code);
        any_stopval = (code == 2);
        any_time    = (code == 1);
      }
      if (any_stopval || (stop_flag.load(std::memory_order_relaxed) &&
                          g_best <= stopval)) {
        stop_reason = StopReason::StopVal; return false;
      }
      if (any_time) { stop_reason = StopReason::MaxTime; return false; }
      if (e_prop == 0) {
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
              static_cast<std::uint64_t>(R) * rung_cost);
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
      std::vector<double> st_local(n_local * stride), st_all(R * stride);

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
                " walkers carry a finite loss");
          }

          const double beta = 1.0 / T;
          const double beta_rt_cap = 1.0 / (T * kb.rt) - beta;
          double dbeta = beta_rt_cap;                 /* geometric default */
          if (kb.ess_target > 0.0 && f_finite.size() >= 2) {
            const double lo = *std::min_element(f_finite.begin(), f_finite.end());
            const double hi = *std::max_element(f_finite.begin(), f_finite.end());
            if (hi > lo) {   /* non-degenerate -> adapt */
              dbeta = anneal::choose_dbeta(f_finite, kb.ess_target, beta_rt_cap);
              adaptive_used = true;
              if (dbeta <= 0.0) dbeta = beta_rt_cap * 1e-3;
            }
          }
          T_new = 1.0 / (beta + dbeta);
          if (T_new < t_min) T_new = t_min;

          /* Reweight + systematic resample plan over the gathered f. */
          std::vector<double> wgt(R, 0.0);
          double wsum = 0.0, fmin = std::numeric_limits<double>::infinity();
          for (double f : f_all)
            if (anneal::loss_is_finite(f, sp.failscore) && f < fmin) fmin = f;
          for (std::size_t k = 0; k < R; ++k) {
            if (anneal::loss_is_finite(f_all[k], sp.failscore)) {
              wgt[k] = std::exp(-dbeta * (f_all[k] - fmin));
              wsum += wgt[k];
            }
          }
          if (wsum > 0.0) {
            ess = anneal::reweight_ess(f_finite, dbeta);
            std::mt19937_64 rs_rng =
                anneal::make_stream(seed, anneal::SALT_PA_RESAMPLE, step_count);
            const double u =
                std::uniform_real_distribution<double>(0.0, 1.0)(rs_rng);
            parent = anneal::systematic_resample(wgt, u);
            std::unordered_set<std::size_t> uniq(parent.begin(), parent.end());
            unique_parents = uniq.size();
            for (std::size_t j = 0; j < R; ++j)
              if (parent[j] != j) ++culled;
            do_resample = 1;
          } else {
            HPOLog::instance().warning(
                "ANNEAL_PA: no finite walker -- skipping resampling this step "
                "(geometric cooling continues)");
          }
        }

        if (world > 1) {
          std::vector<double> ctrl{T_new, static_cast<double>(do_resample)};
          hpo_bcast(ctrl, 0);
          T_new       = ctrl[0];
          do_resample = static_cast<int>(ctrl[1]);
        }

        /* --- apply the resample plan ----------------------------------- */
        if (do_resample) {
          pack_states();
          hpo_gatherv(st_local, st_all, st_counts);
          if (rank == 0) {
            std::vector<double> permuted(R * stride);
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
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PA step=" << step_count;
          if (planned_steps > 0) oss << "/" << planned_steps;
          oss << "  T=" << T
              << "  ESS=" << std::fixed << std::setprecision(1) << ess
              << "  parents=" << unique_parents << "/" << R
              << "  culled=" << culled
              << "  finite=" << n_fin << "/" << R
              << std::scientific << std::setprecision(3)
              << "  pop_best=" << f_pop_best
              << "  best=" << g_best
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
        if (rank0_code == 1) { stop_reason = StopReason::Plateau; break; }
        if (T <= t_min) { stop_reason = StopReason::TMin; break; }
      }
    }

    /* --- multi-rank finalisation: rank 0's best_* outputs must carry
     * the GLOBAL winner (same scheme as ANNEAL_PT). */
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
        if (rank == 0 && owner != 0) {
          budget.try_reserve();   /* accounting only */
          (void)host(sp.out(g_best_x));
        }
      }
      spent_report = hpo_allreduce_sum(budget.spent());
    }

    /* --- final summary ----------------------------------------------------- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) /
            static_cast<double>(total_proposed) : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL_PA summary: steps=" << step_count
          << "  population=" << R
          << "  proposed=" << total_proposed
          << "  accepted=" << std::fixed << std::setprecision(1)
          << (overall_rate * 100.0) << "%"
          << std::scientific << std::setprecision(3)
          << "  oob=" << total_oob
          << "  culled_total=" << total_culled
          << "  T_hot=" << T_hot
          << "  best=" << g_best;
      HPOLog::instance().info(oss.str());
    }

    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T <= T_MIN"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule, population best)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL_PA stopped: " << reason
          << " after " << spent_report << " host eval(s)"
          << " (" << step_count << " temperature step(s))";
      HPOLog::instance().info(oss.str());
    }
  }

template<typename Vec, typename Runner>
  void run_anneal_pa(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pa_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo

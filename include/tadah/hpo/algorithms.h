#pragma once
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/anneal_core.h>
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

/* ----- RANDOM ---------------------------------------------------- */
template<typename Vec, typename Runner>
  void run_random(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    uint64_t seed = sp.seed.value_or(std::random_device{}());  // default seed
    {
      std::ostringstream oss;
      oss << "RANDOM search: seed=" << seed;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    std::mt19937_64 rng(seed);
    const auto& lo = sp.low, &hi = sp.high;

    constexpr uint64_t HUGE_UNSIGNED = std::numeric_limits<uint64_t>::max();
    uint64_t max = sp.stop.maxeval.value_or(HUGE_UNSIGNED);

    /* evaluate the user's config point first so the prior is always part
     * of the explored set. */
    if (!sp.optim_init.empty() && max > 0)
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      host(sp.out(x_init));
      if (max != HUGE_UNSIGNED) --max;
    }

    for(uint64_t c=0;c<max;++c)
    {
      Vec x(lo.size());
      for(size_t i=0;i<lo.size();++i)
        x(i)=std::uniform_real_distribution<>(lo[i],hi[i])(rng);
      host( sp.out(x) );           // evaluate in original scale
    }
  }

/* ----- GRID  (nave Cartesian product) --------------------------- */
template<typename Vec, typename Runner>
  void run_grid(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {

    unsigned steps = sp.step.value_or(20);  // default steps in each dimension

    const size_t dim = sp.low.size();
    std::vector<unsigned> idx(dim,0);
    {
      std::ostringstream oss;
      oss << "GRID search: " << steps << " steps in each dimension, "
          << dim << " dimensions.";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    auto total=[&](){
      double t=1; for(size_t d=0;d<dim;++d) t*=steps; return static_cast<uint64_t>(t); }();

    /* evaluate the user's config point first so the prior is always part
     * of the explored set (in addition to the regular grid sweep). */
    if (!sp.optim_init.empty())
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      host(sp.out(x_init));
    }

    uint64_t done=0;
    Vec x(dim);
    while (true)
    {
      for(size_t d=0;d<dim;++d)
      {
        double t= static_cast<double>(idx[d])/(steps-1);
        x(d) = sp.low[d]*(1-t) + sp.high[d]*t;
      }
      host( sp.out(x) );
      if (++done==total) break;

      /* increment mixed-radix counter */
      for(size_t d=0;d<dim;++d)
        if(++idx[d] < steps) break; else idx[d]=0;
    }
  }

/* ----- ANNEAL  (classic Kirkpatrick/Metropolis SA) ---------------- *
 *
 * Refs: Kirkpatrick et al., Science 220 (1983); Metropolis et al.,
 *       J. Chem. Phys. 21 (1953); Corana et al., ACM TOMS 13:262
 *       (1987); Brommer, Univ. Stuttgart (2003).
 *
 * Serial driver over the shared chain core in anneal_core.h (the
 * parallel drivers ANNEAL_PT / ANNEAL_PA reuse the same rungs, T0
 * calibration, plateau rule, and budget plumbing). Operates entirely
 * in transformed (log-scaled) parameter space: `sp.low/sp.high` are
 * post-`analyse_logscale`; `host()` is called via `sp.out(...)` so the
 * LAMMPS side sees real-space values.
 *
 * Stops honoured: MAXEVAL (hard cap on host calls, including T0
 * probes), MAXTIME (wall clock), STOPVAL (any host eval `<=` stopval),
 * T < T_MIN (with optional warm re-anneal, see REANNEAL), and
 * FTOL_REL/FTOL_ABS via the Corana sec.-4 eps-NEPS plateau rule (see
 * anneal::CoranaPlateau).
 *
 * T0 calibration (when PARAM T_HOT is not set): seed-anchored local
 * probing with FAILSCORE-robust statistics; see anneal::calibrate_t0.
 *
 * PARAMs (all optional, defaults shown):
 *   T_HOT          auto    starting/hottest temperature (skips T0
 *                          calibration; renamed from T_INIT)
 *   T_MIN          auto    cooling floor; auto = T0 * 1e-6
 *   RT             0.90    geometric cooling factor T <- T * RT
 *   L              50      Markov-chain length per T (CLASSIC only)
 *   STEP_FRAC      0.10    sigma_i = STEP_FRAC * (high[i] - low[i])
 *   T0_PROBES      20      local probes used to calibrate T0
 *   T0_PACCEPT     0.05    target acceptance probability for T0
 *   NEPS           4       plateau window (rungs); used only when
 *                          FTOL_REL and/or FTOL_ABS is set
 *   CORANA         0       enable the Corana adaptive variant
 *   NS             5       Corana step cycles per adjust cycle
 *   NT             5       Corana step-adjust cycles per T
 *   C              2.0     Corana step-adjust gain
 *   V_FLOOR_FRAC   1e-3    Corana V[i] floor as fraction of dim width
 *   RESET_TO_BEST  CORANA? Brommer reset-to-best after each cool
 *                          (default 1 under CORANA, 0 otherwise)
 *   REANNEAL       1       on T_MIN with budget left: snap to best and
 *                          re-anneal from REANNEAL_SCALE^n * T0
 *   REANNEAL_SCALE 0.25    per-re-anneal start-temperature factor
 *
 * If `STEP <s>` is set on the OptimizerSpec, sigma_i = s for all i
 * (overrides STEP_FRAC).
 *
 * Knobs that are dead for this algorithm are hard errors, not silent
 * no-ops: XTOL_REL/XTOL_ABS, VECTOR_ARR, GRAD_STRATEGY, and POPULATION
 * (use TADAH:ANNEAL_PA), unknown PARAM keys, and the renamed T_INIT.
 * THREADS IS accepted here under the shared-pool model (Option A): it
 * sizes the per-rank worker pool that runs the within-eval pc_lammps
 * fan-out -- serial ANNEAL still drives ONE chain, only the fan-out is
 * parallel. See anneal::Knobs::parse.
 *
 * `run_anneal_impl` is parameterised on a generic callable host
 * (`double operator()(const Vec&) const`) so unit tests can drive the
 * SA loop with a stub instead of a full HPO_Host. The public
 * `run_anneal<Vec, Runner>(...)` wrapper below delegates here.
 */
template<typename Vec, typename HostT>
  void run_anneal_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::Serial);

    /* Decorrelated streams: probing never shifts the proposal
     * sequence (see anneal_core.h). */
    std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);

    /* per-dim sigma (transformed-space) */
    std::vector<double> sigma(dim);
    if (sp.step.has_value()) {
      for (std::size_t i = 0; i < dim; ++i) sigma[i] = *sp.step;
    } else {
      for (std::size_t i = 0; i < dim; ++i)
        sigma[i] = kb.step_frac * (sp.high[i] - sp.low[i]);
    }

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    anneal::SharedBudget budget(sp.stop.maxeval.value_or(HUGE_UNSIGNED));

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

    /* Stop reason -- set at every exit path; emitted as a single
     * `ANNEAL stopped: ...` line at the end so the user can see why
     * the loop finished without grepping per-eval timestamps. */
    enum class StopReason {
      Done,            /* loop completed normally (rare) */
      MaxEval,
      MaxTime,
      StopVal,
      TMin,
      Plateau,         /* Corana sec.-4 eps-NEPS plateau */
      StopValOnProbe,  /* STOPVAL hit during T0 calibration probes */
      MaxTimeOnProbe,
      NoDims,
      NoBudget         /* MAXEVAL ran out before any SA proposal */
    };
    StopReason stop_reason = StopReason::Done;

    /* Aggregates referenced by the final summary; declared up front so
     * every early-exit path can fall through to one summary block. */
    double      T0              = 0.0;
    double      T_final         = 0.0;
    std::size_t total_proposed  = 0;
    std::size_t total_accepted  = 0;
    std::size_t total_oob       = 0;
    std::size_t rung_count      = 0;
    std::size_t reanneal_count  = 0;
    double      accept_rate_min = 1.0;
    double      accept_rate_max = 0.0;
    bool        any_rate        = false;
    bool        ran_sa          = false;   /* true once the seed evaluated */

    /* Entry banner. When Corana is enabled it shows NS/NT/C in place
     * of L (which Corana doesn't use). */
    {
      std::ostringstream oss;
      oss << "ANNEAL[" << (kb.corana ? "CORANA" : "CLASSIC") << "]: "
          << "seed=" << seed
          << "  RT=" << kb.rt;
      if (kb.t_min) oss << "  T_MIN=" << *kb.t_min;
      else          oss << "  T_MIN=auto(T0*1e-6)";
      if (kb.corana) {
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT << "  C=" << kb.C
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
        if (kb.reset_to_best) oss << "  RESET_TO_BEST=1";
      } else {
        oss << "  L=" << kb.L;
      }
      if (sp.step.has_value())
        oss << "  sigma=" << *sp.step << " (uniform; STEP override)";
      else
        oss << "  STEP_FRAC=" << kb.step_frac;
      oss << "  REANNEAL=" << (kb.reanneal ? 1 : 0);
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

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

    /* --- chain state + seed/anchor evaluation --------------------------
     * The seed is evaluated BEFORE T0 calibration: the estimator anchors
     * its local probes on it. */
    anneal::ChainState<Vec> chain;
    chain.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, 0);
    chain.V   = sigma;
    chain.x   = Vec(dim);
    if (proceed) {
      if (!sp.optim_init.empty()) {
        const auto v = sp.in(sp.optim_init.front());
        for (std::size_t i = 0; i < v.size(); ++i) chain.x(i) = v[i];
      } else {
        for (std::size_t i = 0; i < dim; ++i)
          chain.x(i) = 0.5 * (sp.low[i] + sp.high[i]);
      }

      if (!budget.try_reserve()) {
        stop_reason = StopReason::NoBudget;
        proceed     = false;
      } else {
        chain.f = host(sp.out(chain.x));
        ran_sa = true;
        chain.best_f = chain.f;
        chain.best_x = chain.x;

        if (chain.f <= stopval) {
          stop_reason = StopReason::StopVal;
          proceed     = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTime;
          proceed     = false;
        }
      }
    }
    if (chain.best_x.size() == 0) chain.best_x = chain.x;

    /* --- T0 calibration (skipped when PARAM T_HOT is set) --- */
    if (proceed) {
      const auto t0out = anneal::calibrate_t0(
          sp, kb, sigma, chain.x, chain.f, chain.best_x, chain.best_f,
          host, probe_rng, ctx);
      T0      = t0out.T0;
      T_final = T0;
      if (t0out.aborted) {
        stop_reason = t0out.stopval_hit ? StopReason::StopValOnProbe
                                        : StopReason::MaxTimeOnProbe;
        proceed = false;
      }
    }

    /* Auto cooling floor: reachable in ~85 rungs at RT=0.85 instead of
     * the historical fixed 1e-12 (~170+ rungs -- runs always died on
     * MAXEVAL instead). Explicit PARAM T_MIN always wins. */
    const double t_min = kb.t_min ? *kb.t_min
                                  : (T0 > 0.0 ? T0 * 1e-6 : 1e-12);

    /* --- one annealing schedule from T_begin ---------------------------
     * Shared by the first anneal and every REANNEAL cycle. */
    const std::uint64_t rung_cost =
        kb.corana ? static_cast<std::uint64_t>(kb.NT) * kb.NS * dim
                  : static_cast<std::uint64_t>(kb.L);
    auto run_schedule = [&](double T_begin) {
      double T = T_begin;
      /* Planned total for the rung=current/total display: whichever of
       * the T_MIN schedule or the eval budget will end this schedule
       * first (plateau/STOPVAL may still end it earlier). Recomputed
       * per REANNEAL cycle on top of the rungs already completed. */
      const std::size_t schedule_total =
          rung_count + anneal::planned_rungs(T_begin, t_min, kb.rt,
                                             budget.left(), rung_cost);
      while (T >= t_min && budget.left() > 0 && !time_exceeded()) {
        const auto rs = anneal::run_rung(chain, T, sp, kb, host, ctx);

        /* Per-rung summary line. Emitted whenever the rung consumed
         * any host evals, even if we broke out mid-rung. */
        if (rs.proposed > 0 || rs.oob > 0) {
          const double rate = rs.proposed > 0
              ? static_cast<double>(rs.accepted) / static_cast<double>(rs.proposed) : 0.0;
          const std::size_t r_total = rs.proposed + rs.oob;
          const double oob_pct = r_total > 0
              ? 100.0 * static_cast<double>(rs.oob) / static_cast<double>(r_total) : 0.0;
          if (rs.proposed > 0) {
            any_rate = true;
            if (rate < accept_rate_min) accept_rate_min = rate;
            if (rate > accept_rate_max) accept_rate_max = rate;
          }
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << (kb.corana ? "ANNEAL[CORANA] rung=" : "ANNEAL rung=")
              << (rung_count + 1) << "/" << schedule_total
              << "  T=" << T
              << "  proposed=" << rs.proposed
              << "  accepted=" << rs.accepted
              << " (" << std::fixed << std::setprecision(1) << (rate * 100.0) << "%)"
              << "  oob=" << rs.oob
              << " (" << oob_pct << "%)"
              << std::scientific << std::setprecision(3)
              << "  f_x=" << chain.f
              << "  best=" << chain.best_f;
          if (kb.corana) {
            const double V_min = *std::min_element(chain.V.begin(), chain.V.end());
            const double V_max = *std::max_element(chain.V.begin(), chain.V.end());
            oss << "  V=[" << V_min << ", " << V_max << "]";
          }
          HPOLog::instance().info(oss.str());
        }
        total_proposed += rs.proposed;
        total_accepted += rs.accepted;
        total_oob      += rs.oob;
        T_final         = T;
        ++rung_count;

        if (rs.stop != RungStop::Completed) {
          switch (rs.stop) {
            case RungStop::Budget:  stop_reason = StopReason::MaxEval; break;
            case RungStop::Time:    stop_reason = StopReason::MaxTime; break;
            case RungStop::StopVal: stop_reason = StopReason::StopVal; break;
            default: break;
          }
          break;
        }

        if (plateau.update(chain.f, chain.best_f)) {
          stop_reason = StopReason::Plateau; break;
        }

        T *= kb.rt;
        if (T < t_min) { stop_reason = StopReason::TMin; break; }
        if (budget.left() == 0) { stop_reason = StopReason::MaxEval; break; }
        if (time_exceeded()) { stop_reason = StopReason::MaxTime; break; }

        /* Brommer-style reset-to-best after each cool. Cheap (no host
         * eval consumed); just snaps the chain back to the best-so-far
         * point. The next rung then refines around best_x with the
         * freshly-cooled T and the V[] adapted so far. */
        if (kb.reset_to_best && chain.f > chain.best_f) {
          chain.x = chain.best_x;
          chain.f = chain.best_f;
        }
      }
    };

    /* --- main anneal + warm re-anneal driver ---------------------------
     * When a schedule cools out (T < T_MIN) with budget left, REANNEAL
     * snaps the chain to the best-so-far and restarts the schedule at
     * REANNEAL_SCALE^n * T0 (run15 lever: the schedule used to truncate
     * progress mid-improvement). Stops re-annealing when the next start
     * temperature could not complete ~2 rungs above T_MIN. */
    if (proceed) {
      double T_start = T0;
      for (;;) {
        run_schedule(T_start);
        if (stop_reason != StopReason::TMin) break;
        if (!kb.reanneal) break;
        if (budget.left() == 0 || time_exceeded()) break;
        const double next_T =
            T0 * std::pow(kb.reanneal_scale,
                          static_cast<double>(reanneal_count + 1));
        if (next_T < t_min / kb.rt) break;
        ++reanneal_count;
        chain.x = chain.best_x;
        chain.f = chain.best_f;
        T_start = next_T;
        stop_reason = StopReason::Done;
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "ANNEAL re-anneal #" << reanneal_count
            << ": T_start=" << T_start
            << "  from best=" << chain.best_f;
        HPOLog::instance().info(oss.str());
      }
    }

    /* --- final SA summary block (only when SA actually ran past the
     * seed evaluation). --- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) / static_cast<double>(total_proposed) : 0.0;
      const std::size_t total_evals = total_proposed + total_oob;
      const double oob_pct = total_evals > 0
          ? 100.0 * static_cast<double>(total_oob) / static_cast<double>(total_evals) : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL summary: rungs=" << rung_count
          << "  proposed=" << total_proposed
          << "  accepted=" << total_accepted
          << " (" << std::fixed << std::setprecision(1) << (overall_rate * 100.0) << "%)"
          << "  oob=" << total_oob
          << " (" << oob_pct << "%)"
          << std::scientific << std::setprecision(3)
          << "  T0=" << T0
          << "  T_final=" << T_final
          << "  reanneals=" << reanneal_count
          << "  best=" << chain.best_f;
      HPOLog::instance().info(oss.str());
      if (any_rate) {
        std::ostringstream oss2;
        oss2 << std::fixed << std::setprecision(1)
             << "ANNEAL accept-rate trajectory: min="
             << (accept_rate_min * 100.0) << "%  max="
             << (accept_rate_max * 100.0) << "%";
        HPOLog::instance().info(oss2.str());
      }
    }

    /* --- explicit stop reason (single line, every exit path) --- */
    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T < T_MIN"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL stopped: " << reason
          << " after " << budget.spent() << " host eval(s)"
          << " (" << rung_count << " rung(s) completed";
      if (reanneal_count > 0) oss << ", " << reanneal_count << " re-anneal(s)";
      oss << ")";
      HPOLog::instance().info(oss.str());
    }
  }

template<typename Vec, typename Runner>
  void run_anneal(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_impl<Vec>(sp, host);
  }
}   // namespace

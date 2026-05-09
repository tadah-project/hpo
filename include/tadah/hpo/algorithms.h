#pragma once
#include <tadah/hpo/hpo_host.h>
#include <algorithm>
#include <chrono>
#include <cmath>
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

/* ----- GRID  (naïve Cartesian product) --------------------------- */
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
 *       J. Chem. Phys. 21 (1953).
 *
 * Operates entirely in transformed (log-scaled) parameter space:
 * `sp.low/sp.high` are post-`analyse_logscale`; `host()` is called via
 * `sp.out(...)` so the LAMMPS side sees real-space values.
 *
 * Stops honoured: MAXEVAL (hard cap on host calls, including T0
 * probes), MAXTIME (wall clock), STOPVAL (any host eval `<=` stopval).
 * FTOL_REL/FTOL_ABS are not honoured by this driver (no NEPS history).
 *
 * PARAMs (all optional, defaults shown):
 *   T_INIT      auto    initial temperature (skips T0 calibration)
 *   T_MIN       1e-12   stop when T < T_MIN
 *   RT          0.85    geometric cooling factor T <- T * RT
 *   L           50      Markov-chain length per temperature
 *   STEP_FRAC   0.10    sigma_i = STEP_FRAC * (high[i] - low[i])
 *   T0_PROBES   20      uniform-bounds samples used to calibrate T0
 *   T0_PACCEPT  0.80    target acceptance probability for T0
 *
 * If `STEP <s>` is set on the OptimizerSpec, sigma_i = s for all i
 * (overrides STEP_FRAC).
 *
 * `run_anneal_impl` is parameterised on a generic callable host
 * (`double operator()(const Vec&) const`) so unit tests can drive the
 * SA loop with a stub instead of a full HPO_Host. The public
 * `run_anneal<Vec, Runner>(...)` wrapper below delegates here.
 */
template<typename Vec, typename HostT>
  void run_anneal_impl(const OptimizerSpec& sp, HostT& host)
  {
    const std::size_t dim = sp.low.size();
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());
    std::mt19937_64 rng(seed);

    auto pget = [&](const char* k, double dflt) {
      auto it = sp.param.find(k);
      return it == sp.param.end() ? dflt : it->second;
    };
    const bool        has_t_init    = sp.param.count("T_INIT") > 0;
    const double      t_init_user   = pget("T_INIT", 0.0);
    const double      t_min         = pget("T_MIN", 1e-12);
    const double      rt            = pget("RT", 0.85);
    const std::size_t L             = static_cast<std::size_t>(pget("L", 50.0));
    const double      step_frac     = pget("STEP_FRAC", 0.1);
    const std::size_t t0_probes_req = static_cast<std::size_t>(pget("T0_PROBES", 20.0));
    const double      t0_paccept    = pget("T0_PACCEPT", 0.8);

    if (!(rt > 0.0 && rt < 1.0))
      throw std::runtime_error("ANNEAL: PARAM RT must be in (0,1) (got "
        + std::to_string(rt) + ")");
    if (L == 0)
      throw std::runtime_error("ANNEAL: PARAM L must be >= 1");
    if (!(t0_paccept > 0.0 && t0_paccept < 1.0))
      throw std::runtime_error("ANNEAL: PARAM T0_PACCEPT must be in (0,1) (got "
        + std::to_string(t0_paccept) + ")");
    if (has_t_init && !(t_init_user > 0.0))
      throw std::runtime_error("ANNEAL: PARAM T_INIT must be > 0");

    /* per-dim sigma (transformed-space) */
    std::vector<double> sigma(dim);
    if (sp.step.has_value()) {
      for (std::size_t i = 0; i < dim; ++i) sigma[i] = *sp.step;
    } else {
      for (std::size_t i = 0; i < dim; ++i)
        sigma[i] = step_frac * (sp.high[i] - sp.low[i]);
    }

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    std::uint64_t n_used = 0;
    auto budget_left = [&]() -> std::uint64_t {
      return max_eval > n_used ? (max_eval - n_used) : 0;
    };

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    auto time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const double stopval =
        sp.stop.stopval.value_or(-std::numeric_limits<double>::infinity());

    /* Stop reason — set at every exit path; emitted as a single
     * `ANNEAL stopped: ...` line at the end so the user can see why
     * the loop finished without grepping per-eval timestamps. */
    enum class StopReason {
      Done,            /* loop completed normally (rare) */
      MaxEval,
      MaxTime,
      StopVal,
      TMin,
      StopValOnProbe,  /* STOPVAL hit during T0 calibration probes */
      MaxTimeOnProbe,
      NoDims,
      NoBudget         /* MAXEVAL ran out before any SA proposal */
    };
    StopReason stop_reason = StopReason::Done;

    /* Aggregates referenced by the final summary; declared up front so
     * every early-exit path can fall through to one summary block.
     * They start at zero, which is meaningful: "0 rungs, 0 proposals"
     * tells the user no SA progress happened. */
    double      T0              = 0.0;
    double      T_final         = 0.0;
    double      best_f          = std::numeric_limits<double>::infinity();
    std::size_t total_proposed  = 0;
    std::size_t total_accepted  = 0;
    std::size_t total_oob       = 0;
    std::size_t rung_count      = 0;
    double      accept_rate_min = 1.0;
    double      accept_rate_max = 0.0;
    bool        any_rate        = false;
    bool        ran_sa          = false;   /* true once we passed T0 + seed */

    /* Entry banner — uniform sigma override is reported as the actual
     * value used (was previously misreported as STEP_FRAC even when
     * STEP overrode it). */
    {
      std::ostringstream oss;
      oss << "ANNEAL: seed=" << seed
          << "  L=" << L
          << "  RT=" << rt
          << "  T_MIN=" << t_min;
      if (sp.step.has_value())
        oss << "  sigma=" << *sp.step << " (uniform; STEP override)";
      else
        oss << "  STEP_FRAC=" << step_frac;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    if (dim == 0) {
      if (!sp.optim_init.empty() && budget_left() > 0) {
        const auto v = sp.in(sp.optim_init.front());
        Vec xv(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) xv(i) = v[i];
        host(sp.out(xv));
      }
      stop_reason = StopReason::NoDims;
    }

    /* --- T0 calibration (skipped when PARAM T_INIT is set) ---
     * Whether calibrated or user-set, both branches set T0 below.
     * `proceed` tracks whether the SA loop should run after this. */
    bool proceed = (stop_reason == StopReason::Done);

    if (proceed) {
      if (has_t_init) {
        T0 = t_init_user;
        T_final = T0;
        HPOLog::instance().info(
          "ANNEAL: using user-specified T_INIT=" + std::to_string(T0));
      } else {
        const std::size_t n_probe = static_cast<std::size_t>(
            std::min<std::uint64_t>(t0_probes_req, budget_left()));
        std::vector<double> probes;
        probes.reserve(n_probe);
        for (std::size_t k = 0; k < n_probe; ++k) {
          Vec xp(dim);
          for (std::size_t i = 0; i < dim; ++i)
            xp(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(rng);
          const double f = host(sp.out(xp));
          ++n_used;
          probes.push_back(f);
          if (f < best_f) best_f = f;
          if (f <= stopval) {
            stop_reason = StopReason::StopValOnProbe;
            proceed     = false;
            break;
          }
          if (time_exceeded()) {
            stop_reason = StopReason::MaxTimeOnProbe;
            proceed     = false;
            break;
          }
        }

        /* Probe-loss spread stats (min/max/std) help the user see
         * whether probes landed in FAILSCORE territory or on a flat
         * plateau. Always emitted, even if we exited early. */
        double pmin = 0.0, pmax = 0.0, pmean = 0.0, pstd = 0.0;
        if (!probes.empty()) {
          pmin = pmax = probes.front();
          double sum = 0.0;
          for (double v : probes) {
            if (v < pmin) pmin = v;
            if (v > pmax) pmax = v;
            sum += v;
          }
          pmean = sum / static_cast<double>(probes.size());
          if (probes.size() >= 2) {
            double sq = 0.0;
            for (double v : probes) { const double d = v - pmean; sq += d*d; }
            pstd = std::sqrt(sq / static_cast<double>(probes.size() - 1));
          }
        }
        double sum_abs = 0.0; std::size_t pairs = 0;
        for (std::size_t i = 0; i < probes.size(); ++i)
          for (std::size_t j = i + 1; j < probes.size(); ++j) {
            sum_abs += std::abs(probes[i] - probes[j]);
            ++pairs;
          }
        const double mean_dE = pairs > 0 ? sum_abs / static_cast<double>(pairs) : 0.0;

        if (proceed) {
          if (mean_dE > 0.0) {
            T0 = -mean_dE / std::log(t0_paccept);
          } else {
            T0 = 1.0;
            HPOLog::instance().warning(
              "ANNEAL: T0 calibration produced mean|dE|=0 over "
              + std::to_string(probes.size())
              + " probe(s); falling back to T0=1.0. Consider PARAM T_INIT or wider bounds.");
          }
          T_final = T0;
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL: T0=" << T0
              << " from " << probes.size() << " probes,"
              << "  mean|dE|=" << mean_dE
              << "  loss range=[" << pmin << ", " << pmax << "]"
              << "  std=" << pstd
              << "  target p_accept=" << t0_paccept;
          std::cout << "[HPO] " << oss.str() << "\n";
          HPOLog::instance().info(oss.str());
        } else {
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL: T0 probes truncated after " << probes.size()
              << " sample(s)  loss range=[" << pmin << ", " << pmax << "]";
          HPOLog::instance().info(oss.str());
        }
      }
    }

    /* --- seed evaluation + main SA loop --- */
    if (proceed) {
      Vec x(dim);
      if (!sp.optim_init.empty()) {
        const auto v = sp.in(sp.optim_init.front());
        for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
      } else {
        for (std::size_t i = 0; i < dim; ++i)
          x(i) = 0.5 * (sp.low[i] + sp.high[i]);
      }

      if (budget_left() == 0) {
        stop_reason = StopReason::NoBudget;
        proceed     = false;
      }

      double f_x = std::numeric_limits<double>::infinity();
      if (proceed) {
        f_x = host(sp.out(x));
        ++n_used;
        ran_sa = true;
        if (f_x < best_f) best_f = f_x;

        if (f_x <= stopval) {
          stop_reason = StopReason::StopVal;
          proceed     = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTime;
          proceed     = false;
        }
      }

      /* --- annealing loop --- */
      if (proceed) {
        double T = T0;
        std::normal_distribution<double>          ndist(0.0, 1.0);
        std::uniform_real_distribution<double>    udist(0.0, 1.0);

        while (T >= t_min && budget_left() > 0 && !time_exceeded()) {
          std::size_t r_proposed = 0, r_accepted = 0, r_oob = 0;
          bool inner_break = false;

          for (std::size_t k = 0; k < L; ++k) {
            if (budget_left() == 0) {
              stop_reason = StopReason::MaxEval; inner_break = true; break;
            }
            if (time_exceeded()) {
              stop_reason = StopReason::MaxTime; inner_break = true; break;
            }

            Vec x_prop(dim);
            bool in_bounds = true;
            for (std::size_t i = 0; i < dim; ++i) {
              x_prop(i) = x(i) + sigma[i] * ndist(rng);
              if (x_prop(i) < sp.low[i] || x_prop(i) > sp.high[i]) {
                in_bounds = false; break;
              }
            }
            if (!in_bounds) { ++r_oob; continue; }

            const double f_prop = host(sp.out(x_prop));
            ++n_used; ++r_proposed;
            const double df = f_prop - f_x;
            const bool accept = (df <= 0.0) || (udist(rng) < std::exp(-df / T));
            if (accept) {
              x   = x_prop;
              f_x = f_prop;
              ++r_accepted;
              if (f_x < best_f) best_f = f_x;
            }
            if (f_x <= stopval) {
              stop_reason = StopReason::StopVal; inner_break = true; break;
            }
          }

          /* Per-rung summary line. Emitted whenever the rung consumed
           * any host evals, even if we broke out mid-rung. */
          if (r_proposed > 0 || r_oob > 0) {
            const double rate = r_proposed > 0
                ? static_cast<double>(r_accepted) / static_cast<double>(r_proposed) : 0.0;
            const std::size_t r_total = r_proposed + r_oob;
            const double oob_pct = r_total > 0
                ? 100.0 * static_cast<double>(r_oob) / static_cast<double>(r_total) : 0.0;
            if (r_proposed > 0) {
              any_rate = true;
              if (rate < accept_rate_min) accept_rate_min = rate;
              if (rate > accept_rate_max) accept_rate_max = rate;
            }
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(3)
                << "ANNEAL rung k=" << rung_count
                << "  T=" << T
                << "  proposed=" << r_proposed
                << "  accepted=" << r_accepted
                << " (" << std::fixed << std::setprecision(1) << (rate * 100.0) << "%)"
                << "  oob=" << r_oob
                << " (" << oob_pct << "%)"
                << std::scientific << std::setprecision(3)
                << "  f_x=" << f_x
                << "  best=" << best_f;
            HPOLog::instance().info(oss.str());
          }
          total_proposed += r_proposed;
          total_accepted += r_accepted;
          total_oob      += r_oob;
          T_final         = T;
          ++rung_count;

          if (inner_break) break;

          T *= rt;
          if (T < t_min) { stop_reason = StopReason::TMin; break; }
          if (budget_left() == 0) { stop_reason = StopReason::MaxEval; break; }
          if (time_exceeded()) { stop_reason = StopReason::MaxTime; break; }
        }
      }
    }

    /* --- final SA summary block (only when SA actually ran past
     * T0; for early T0 exits, the T0-line above already explained
     * what happened). --- */
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
          << "  best=" << best_f;
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
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL stopped: " << reason
          << " after " << n_used << " host eval(s)"
          << " (" << rung_count << " rung(s) completed)";
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

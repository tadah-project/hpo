#pragma once
/*
 * RunPolicy -- execution decorator around the per-optimiser dispatch.
 *
 * BasePolicy runs the optimiser exactly once via run_inner(). The other
 * policies (MultiStartPolicy, StagedPolicy, RestartPolicy) are wrappers
 * around the same run_inner().
 *
 * Splitting "what to run" (spec / algorithm dispatch) from "how to drive
 * it" (single shot, multi-start, restart, two-stage) keeps each concern
 * testable in isolation and lets the eventual init-strategy plumbing live
 * outside the optimiser adapters.
 */

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

/* RunPolicy<> + run_inner<> (the single-shot dispatch) live in
 * run_policy_base.h; this header builds the decorator stack on top. */
#include <tadah/core/checkpoint.h>        // core::ShutdownRequested
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/strategy_rules.h>

namespace tadah::hpo {

/* ----------------------------------------------------------------------
 * BasePolicy -- single-shot. Used as the default and as the innermost
 * layer of every decorator stack.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class BasePolicy : public RunPolicy<Vec,Runner> {
public:
  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    run_inner<Vec,Runner>(spec, host);
  }
};

/* ----------------------------------------------------------------------
 * MultiStartPolicy -- runs the inner optimiser K times from K starting
 * points, taking the K points from spec.optim_init (which the
 * InitStrategy chain has already populated). Local optimisers benefit
 * the most: each run can descend into a different local minimum; the
 * host's best-tracker collects the global best across the K runs.
 *
 * Compatibility (from the caps registry; strategy_rules.h):
 *   - backends the registry does not flag global (DLIB:BFGS, CERES:*,
 *     NLOPT:LD_*, NLOPT:LN_*, NLOPT:AUGLAG*)
 *                                         -> K independent runs (intended).
 *   - multi-seed globals (DLIB:GFS)       -> seeds the K points into
 *     find_min_global's `initial_function_evals`; the global engine then
 *     runs once (single call).
 *   - every other global (NLOPT:G*, TADAH:RANDOM/GRID/ANNEAL*, ASA)
 *                                         -> consumes a single starting
 *     point, so require_multistart_backend() rejects it for any N_STARTS;
 *     the user picks a local or multi-seed backend, or STRATEGY TYPE
 *     PLAIN. validate_hpo_setup() applies the rule at startup; validate()
 *     applies the same function again when the policy runs.
 *
 * Parallelism: the inner optimisers are not natively thread-safe in all
 * cases (NLopt's `nlopt::opt` is, dlib's `find_min_box_constrained`
 * uses local state). The starts run sequentially: PARALLEL is parsed and
 * range-checked, and a value > 1 only draws a warning. Each per-start run
 * still benefits from the shared worker pool (THREADS) that parallelises
 * the pc_lammps fan-out within a single evaluation.
 *
 * A start that fails is logged and the next one runs. Three exceptions
 * end the whole multistart instead, because they are about the run, not
 * the start: core::ShutdownRequested (a signal: hpo_run() ends the run),
 * HPO_Host::StagnationStop (RESTART re-initialises the whole multistart
 * from fresh points) and durable_best::Error (the incumbent can no longer
 * be persisted).
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class MultiStartPolicy : public RunPolicy<Vec,Runner> {
public:
  MultiStartPolicy(std::size_t k, std::size_t parallel,
                   std::optional<std::size_t> budget_per_start = std::nullopt)
      : k_(k), parallel_(parallel), budget_per_start_(budget_per_start) {}

  /* The startup rule (strategy_rules.h), applied again when the policy
   * runs: a safety net for specs built outside HPOTargets. */
  static void validate(const OptimizerSpec& spec) {
    require_multistart_backend(spec);
  }

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    validate(spec);

    const std::size_t n_init = spec.optim_init.size();
    if (n_init == 0) {
      run_inner<Vec,Runner>(spec, host);
      return;
    }

    /* Multi-seed-capable backends consume all K points as initial
     * function evaluations in a single inner run; keyed by
     * caps.accepts_multi_seed so future backends register the
     * capability without touching this branch. For everything else,
     * run the inner optimiser K times sequentially. */
    if (optimizer_caps(spec).accepts_multi_seed) {
      run_inner<Vec,Runner>(spec, host);
      return;
    }

    const std::size_t actual_k = std::min(k_, n_init);
    if (actual_k < k_) {
      HPOLog::instance().warning(
          "MULTISTART N_STARTS " + std::to_string(k_) + " requested but the "
          "resolved INIT chain produced only " + std::to_string(n_init) +
          " starting point(s); running " + std::to_string(actual_k) +
          " start(s). Add INIT points (e.g. METHOD LHS/RANDOM/SOBOL "
          "with K >= " + std::to_string(k_) + ") or lower N_STARTS.");
    }
    HPOLog::instance().info(
        "MULTISTART: running " + std::to_string(actual_k) +
        " starts (parallel=" + std::to_string(parallel_) + ")");

    /* Sequential per-start runs. Parallelism deferred until we have
     * thread-safety audits for each inner adapter; PARALLEL is parsed
     * and range-checked but not honoured > 1. */
    if (parallel_ > 1) {
      HPOLog::instance().warning(
          "MULTISTART PARALLEL > 1 not yet implemented; running starts "
          "sequentially. The shared THREADS pool still parallelises the "
          "pc_lammps fan-out within each evaluation.");
    }

    /* BUDGET_PER_START -- override per-start MAXEVAL.
     *
     * If the user set BUDGET_PER_START explicitly, use it
     * verbatim. Otherwise, when an outer MAXEVAL is set, split it
     * equally across starts so the total still respects MAXEVAL (round
     * up so K * per-start >= MAXEVAL when MAXEVAL is not divisible).
     * If neither is set, leave per-start MAXEVAL as the user's spec
     * (which may be unbounded -- explicit choice). */
    std::optional<uint64_t> per_start_maxeval;
    if (budget_per_start_) {
      per_start_maxeval = static_cast<uint64_t>(*budget_per_start_);
    } else if (spec.stop.maxeval) {
      const uint64_t total = *spec.stop.maxeval;
      const uint64_t k64 = static_cast<uint64_t>(actual_k);
      per_start_maxeval =
          (total + k64 - 1) / k64;   // ceil division
    }
    if (per_start_maxeval) {
      std::ostringstream oss;
      oss << "MULTISTART per-start MAXEVAL = " << *per_start_maxeval
          << (budget_per_start_ ? " (from BUDGET_PER_START)"
                                 : " (split from outer MAXEVAL)");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    for (std::size_t i = 0; i < actual_k; ++i) {
      OptimizerSpec sub = spec;                  // deep copy
      sub.optim_init = { spec.optim_init[i] };   // single start
      if (per_start_maxeval) sub.stop.maxeval = *per_start_maxeval;
      /* Decorrelate the starts: reusing one seed hands every start the
       * same proposal stream, so stochastic inner optimisers (ANNEAL)
       * produce near-duplicate chains that differ only by init point. */
      if (spec.seed) sub.seed = *spec.seed + static_cast<std::uint64_t>(i);

      const std::string start_msg =
          "MULTISTART start " + std::to_string(i+1) + "/" +
          std::to_string(actual_k) + " -- running inner optimiser" +
          (sub.seed ? " (seed=" + std::to_string(*sub.seed) + ")" : "");
      std::cout << "[HPO] " << start_msg << "\n";
      HPOLog::instance().info(start_msg);
      /* Snapshot global state at start so we can summarise this start's
       * contribution: how many evals it consumed and whether it produced
       * a new global best. Without this each start beyond the first
       * appears "silent" in the journal whenever its local fmin doesn't
       * beat the running global best. */
      const std::size_t evals_before = HPOLog::instance().n_eval();
      const double      best_before  = HPOLog::instance().best_loss();
      try {
        run_inner<Vec,Runner>(sub, host);
      } catch (const core::ShutdownRequested&) {
        throw;
      } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
        throw;
      } catch (const durable_best::Error&) {
        throw;
      } catch (const std::exception& e) {
        /* A failure in one start should not kill the whole batch. */
        HPOLog::instance().warning(
            "MULTISTART start " + std::to_string(i+1) + "/" +
            std::to_string(actual_k) + " failed: " + e.what());
      }
      const std::size_t evals_after = HPOLog::instance().n_eval();
      const double      best_after  = HPOLog::instance().best_loss();
      std::ostringstream done;
      done << "MULTISTART start " << (i+1) << "/" << actual_k
           << " done: " << (evals_after - evals_before) << " eval(s) consumed";
      if (best_after < best_before) {
        std::ostringstream tmp;
        tmp << std::scientific << std::setprecision(4)
            << ", improved global best " << best_before
            << " -> " << best_after;
        done << tmp.str();
      } else {
        std::ostringstream tmp;
        tmp << std::scientific << std::setprecision(4)
            << ", no global improvement (best stays at " << best_after << ")";
        done << tmp.str();
      }
      std::cout << "[HPO] " << done.str() << "\n";
      HPOLog::instance().info(done.str());
    }
  }

private:
  std::size_t k_;
  std::size_t parallel_;
  std::optional<std::size_t> budget_per_start_;
};

/* ----------------------------------------------------------------------
 * StagedPolicy -- runs the PRESTAGE block as a global exploration phase,
 * collects the top-K best (loss, params) pairs, then refines each via
 * the outer optimiser.
 *
 * Selected by STRATEGY TYPE PRESTAGE; the exploration phase is the
 * OptimizerSpec::prestage sub-spec parsed from the EXPLORE ... ENDEXPLORE
 * block, the refinement is the outer spec (STRATEGY-level keys or an
 * optional REFINE ... ENDREFINE block). With a configured TOPK > 1 the
 * refinement is a multistart over the candidates, so its optimiser must be
 * one MULTISTART accepts: require_prestage_refinement() (strategy_rules.h)
 * says so at startup, whatever number of candidates the exploration will
 * harvest.
 *
 * The two phases:
 *   1. PRESTAGE (broad search):    spec.prestage          (e.g. DLIB:GFS)
 *   2. REFINEMENT (local sweep):   spec     (the outer)   (e.g. DLIB:BFGS)
 *
 * Under RESTART, a stagnating exploration hands its top-K over to the
 * refinement early (the exploration catch below) instead of restarting,
 * and the refinement starts with a fresh stagnation window. Stagnation in
 * the refinement propagates, so RESTART re-runs the whole PRESTAGE --
 * exploration and refinement -- from a fresh init.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class StagedPolicy : public RunPolicy<Vec,Runner> {
public:
  explicit StagedPolicy(std::size_t top_k) : top_k_(top_k) {
    if (top_k_ == 0)
      throw std::runtime_error("Staged-global mode requires TOPK >= 1");
  }

  /* The startup rules (strategy_rules.h), applied again when the policy
   * runs: a safety net for specs built outside HPOTargets. */
  static void validate(const OptimizerSpec& spec) {
    require_prestage_explore(spec);
    require_prestage_refinement(spec);
  }

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    validate(spec);

    /* Step 1: enable top-K tracking, run the PRESTAGE. Real-space box
     * widths normalise the parameter half of the near-clone dedup (topk.h). */
    std::vector<double> widths(spec.low.size());
    for (std::size_t d = 0; d < widths.size(); ++d) {
      if (spec.space) {
        widths[d] = spec.space->real_width(d);
      } else {
        const bool log_d = d < spec.log_scaled.size() && spec.log_scaled[d];
        widths[d] = log_d ? std::exp(spec.high[d]) - std::exp(spec.low[d])
                          : spec.high[d] - spec.low[d];
      }
    }
    HPO_Host<Vec,Runner>::set_track_topk(top_k_, std::move(widths));
    {
      std::ostringstream oss;
      oss << "PRESTAGE: exploration phase "
          << spec.prestage->lib << ":" << spec.prestage->algo
          << " (MAXEVAL=" << spec.prestage->stop.maxeval.value_or(0) << ")";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    try {
      run_inner<Vec,Runner>(*spec.prestage, host);
    } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
      /* Stagnation during the PRESTAGE is a valid stop signal --
       * proceed to refinement with whatever we collected. */
    }
    /* Stage boundary: the refinement gets a stagnation window of its own.
     * Without this it inherited the exploration's count, so after a
     * stagnating exploration its first evaluation already crossed the
     * threshold. No-op without RESTART (threshold 0). */
    HPO_Host<Vec,Runner>::reset_stagnation();

    /* Step 2: snapshot top-K, disable tracking. */
    auto topk = HPO_Host<Vec,Runner>::get_topk();
    HPO_Host<Vec,Runner>::set_track_topk(0);

    if (topk.empty()) {
      HPOLog::instance().warning(
          "PRESTAGE: produced no candidates -- running outer optimiser as-is");
      run_inner<Vec,Runner>(spec, host);
      return;
    }

    /* Surface the snapshot so the user can sanity-check the PRESTAGE
     * actually found low-loss seeds before paying for the refinement. */
    {
      auto sorted = topk;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
      std::ostringstream oss;
      oss << "PRESTAGE: top-" << sorted.size() << " losses:";
      const std::size_t show = std::min<std::size_t>(sorted.size(), 5);
      for (std::size_t i = 0; i < show; ++i) oss << ' ' << sorted[i].first;
      if (sorted.size() > show) oss << " ...";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Step 3: prepare a refinement spec. Drop the PRESTAGE so we do not
     * recurse, replace optim_init with the top-K params, and run via
     * MULTISTART (or a single-shot if K=1), which runs candidate i with
     * SEED + i (OptimizerSpec::seeded_starts() counts TOPK of them). */
    OptimizerSpec refine = spec;
    refine.prestage.reset();
    refine.optim_init.clear();
    for (const auto& kv : topk) {
      const Vec& params = kv.second;
      std::vector<double> v(params.size());
      for (long i = 0; i < params.size(); ++i) v[i] = params(i);
      refine.optim_init.push_back(std::move(v));
    }

    {
      std::ostringstream oss;
      oss << "REFINEMENT: refining top-" << topk.size()
          << " seed(s) via " << spec.lib << ":" << spec.algo;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    if (topk.size() > 1) {
      MultiStartPolicy<Vec,Runner> ms(topk.size(), 1);
      ms.run(refine, host);
    } else {
      run_inner<Vec,Runner>(refine, host);
    }
  }

private:
  std::size_t top_k_;
};

/* ----------------------------------------------------------------------
 * RestartPolicy -- wraps an inner policy with stagnation-driven restart.
 *
 * On every operator() call HPO_Host checks whether
 *   (g_iter_ - iter_at_last_best_) >= window
 * and throws StagnationStop if so. RestartPolicy catches that, draws a
 * fresh init via the configured strategy, and re-enters the inner
 * policy with the new starting point. The cycle terminates when:
 *   - the inner policy returns naturally, or
 *   - max_restarts attempts have been used.
 *
 * The refresh strategy and the host's HPOTargets are captured at
 * build_run_policy() time.
 *
 * A restart needs StagnationStop to propagate out of the inner policy.
 * Every optimiser adapter lets it through (NLopt, DLIB:GFS and ASA stash
 * the exception and rethrow it after the library returns; it leaves
 * ceres::Solve and the other drivers by itself) and MultiStartPolicy
 * rethrows it, so a restart re-initialises the whole
 * inner policy: every MULTISTART start (as many as the fresh points
 * allow), or the whole PRESTAGE -- its exploration starts from the fresh
 * points unless it has an INIT chain of its own. The one place that
 * absorbs it is StagedPolicy's exploration phase, which hands its top-K
 * over to the refinement instead (see StagedPolicy).
 *
 * Restart r also runs every phase that has a SEED with a seed derived
 * from it (OptimizerSpec::derive_restart_seeds: the outer spec
 * SEED + r*seeded_starts() -- N_STARTS under MULTISTART, TOPK under
 * PRESTAGE, 1 under PLAIN -- so no two starts of the run share a seed; the
 * EXPLORE spec SEED + r), so a seeded stochastic optimiser draws new
 * random numbers instead of replaying round 0's; round 0 keeps every SEED.
 * An INNER block has no SEED: its search draws from the enclosing run's
 * random stream. The journal's "RESTART r/MAX" line lists the derived
 * seeds.
 *
 * When the MAX restarts are used up, the next stagnation ends the run
 * normally, with the trailer "STOPPED -- stagnation, max restarts (MAX)
 * used". The watchdog is armed only while this policy runs: however
 * run() ends, the threshold is reset to 0, so a later run in the same
 * process is not stopped by a stale window.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class RestartPolicy : public RunPolicy<Vec,Runner> {
public:
  RestartPolicy(std::unique_ptr<RunPolicy<Vec,Runner>> inner,
                std::size_t window,
                std::size_t max_restarts,
                std::unique_ptr<InitStrategy> refresh,
                const HPOTargets* trg,
                std::size_t refresh_k = 0)
      : inner_(std::move(inner)),
        window_(window),
        max_restarts_(max_restarts),
        refresh_(std::move(refresh)),
        trg_(trg),
        refresh_k_(refresh_k) {}

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    OptimizerSpec cur = spec;
    std::size_t restarts = 0;

    struct Disarm {
      ~Disarm() { HPO_Host<Vec,Runner>::set_stagnation_threshold(0); }
    } disarm;
    HPO_Host<Vec,Runner>::set_stagnation_threshold(window_);
    HPO_Host<Vec,Runner>::reset_stagnation();

    while (true) {
      try {
        inner_->run(cur, host);
        break;
      } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
        if (restarts >= max_restarts_) {
          const std::string used =
              "max restarts (" + std::to_string(max_restarts_) + ") used";
          HPOLog::instance().info(
              "RESTART: stagnation reached, " + used + "; stopping");
          /* The run ends here, on stagnation: that is its trailer, not a
           * status the stopped inner run may have recorded. */
          HPOLog::instance().set_status("STOPPED -- stagnation, " + used);
          break;
        }
        ++restarts;

        /* Refresh init points. Honour the RESTART block's K_TOTAL when
         * set; otherwise fall back to each strategy's default_k. */
        if (refresh_ && trg_) {
          std::random_device rd;
          InitContext ic{cur, *trg_, static_cast<std::uint64_t>(rd()) + restarts};
          const std::size_t k_use = (refresh_k_ > 0) ? refresh_k_
                                                     : refresh_->default_k();
          auto pts = refresh_->generate(ic, k_use);
          if (!pts.empty()) {
            cur.optim_init = std::move(pts);
            /* A PRESTAGE exploration that starts from the OUTER INIT
             * points restarts from the fresh ones too. */
            cur.share_init_with_nested_phases();
          }
        }
        /* Seeded phases draw new random numbers this round. */
        cur.derive_restart_seeds(spec, restarts);
        HPO_Host<Vec,Runner>::reset_stagnation();
        {
          std::ostringstream oss;
          oss << "RESTART " << restarts << "/" << max_restarts_
              << " -- fresh init from "
              << (refresh_ ? refresh_->name() : std::string("(none)"));
          const std::string seeds = describe_seeds(cur);
          if (!seeds.empty()) oss << "; " << seeds;
          std::cout << "[HPO] " << oss.str() << "\n";
          HPOLog::instance().info(oss.str());
        }
      }
    }
  }

private:
  /* "SEED 6, EXPLORE SEED 1718": the seeds of the phases that have one
   * (an INNER block has none); empty when none has. */
  static std::string describe_seeds(const OptimizerSpec& s)
  {
    std::ostringstream oss;
    const char* sep = "";
    const auto add = [&](const char* phase,
                         const std::optional<std::uint64_t>& seed) {
      if (!seed) return;
      oss << sep << phase << "SEED " << *seed;
      sep = ", ";
    };
    add("", s.seed);
    if (s.prestage) add("EXPLORE ", s.prestage->seed);
    return oss.str();
  }

  std::unique_ptr<RunPolicy<Vec,Runner>> inner_;
  std::size_t window_, max_restarts_;
  std::unique_ptr<InitStrategy> refresh_;
  const HPOTargets* trg_;
  std::size_t refresh_k_ = 0;   // 0 => use refresh_->default_k()
};

/* ----------------------------------------------------------------------
 * build_run_policy -- assembles the RunPolicy decorator stack from the
 * parsed OptimizerSpec.
 *
 * Layering (innermost -> outermost):
 *    BasePolicy / MultiStartPolicy / StagedPolicy   <- chosen by STRATEGY TYPE
 *      +-- wrapped by RestartPolicy if RESTART sets both AFTER_STAGNATION
 *          and MAX > 0
 *
 * Selection rules (STRATEGY TYPE):
 *   - PRESTAGE   -> StagedPolicy (EXPLORE block, then top-K refinement).
 *   - MULTISTART -> MultiStartPolicy (N_STARTS starts of LIB/ALGO).
 *   - PLAIN      -> BasePolicy (single shot).
 *
 * targets is captured for RestartPolicy's refresh_strategy.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
std::unique_ptr<RunPolicy<Vec,Runner>>
build_run_policy(const OptimizerSpec& spec, const HPOTargets* targets = nullptr)
{
  /* Strategy block dispatch -- TYPE chooses the inner policy. */
  std::unique_ptr<RunPolicy<Vec,Runner>> inner;
  switch (spec.strategy_type) {
    case StrategyType::PRESTAGE:
      inner = std::make_unique<StagedPolicy<Vec,Runner>>(spec.stage_topk);
      break;

    case StrategyType::MULTISTART:
      inner = std::make_unique<MultiStartPolicy<Vec,Runner>>(
          spec.multistart.n_starts, spec.multistart.parallel,
          spec.multistart.budget_per_start);
      break;

    case StrategyType::PLAIN:
    default:
      inner = std::make_unique<BasePolicy<Vec,Runner>>();
      break;
  }

  /* RESTART decorator: wraps every STRATEGY TYPE; a restart re-runs the
   * whole inner policy (see RestartPolicy). */
  const auto& rs = spec.restart;
  if (rs.active()) {
    auto refresh = build_init_strategy(rs.init_strategy);
    const std::size_t refresh_k = rs.init_strategy.k_total.value_or(0);
    return std::make_unique<RestartPolicy<Vec,Runner>>(
        std::move(inner),
        rs.after_stagnation,
        rs.max,
        std::move(refresh),
        targets,
        refresh_k);
  }
  return inner;
}

}  // namespace tadah::hpo

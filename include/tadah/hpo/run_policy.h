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

#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

/* RunPolicy<> + run_inner<> live in run_policy_base.h so basins_policy.h
 * can include them without pulling in the full decorator stack -- see
 * comments at the top of run_policy_base.h for the full rationale. */
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>

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
 * the most: each run explores a different basin; the host's
 * best-tracker collects the global best across the K runs.
 *
 * Compatibility:
 *   - DLIB:BFGS, NLOPT:LD_*, NLOPT:LN_*  -> K independent runs (intended).
 *   - DLIB:GFS                            -> seeds the K points into
 *     find_min_global's `initial_function_evals`; the global engine then
 *     runs once (single call). The dlib adapter already accepts K
 *     points (Increment 6 generalisation point).
 *   - NLOPT:GN_*                          -> no analogous API in NLopt;
 *     validate() throws so the user picks a different inner ALGO or
 *     drops MULTISTART.
 *   - TADAH:RANDOM, TADAH:GRID            -> already evaluate every
 *     init point as eval #1..#K via the existing prepend logic.
 *
 * Parallelism: the inner optimisers are not natively thread-safe in all
 * cases (NLopt's `nlopt::opt` is, dlib's `find_min_box_constrained`
 * uses local state). For correctness, we run sequentially today and
 * leave a parallel knob (multistart_parallel) wired through to a
 * dlib::thread_pool in a follow-up. Each per-start run still
 * benefits from the shared worker pool (THREADS) that parallelises
 * the pc_lammps fan-out within a single evaluation.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class MultiStartPolicy : public RunPolicy<Vec,Runner> {
public:
  MultiStartPolicy(std::size_t k, std::size_t parallel,
                   std::optional<std::size_t> budget_per_start = std::nullopt)
      : k_(k), parallel_(parallel), budget_per_start_(budget_per_start) {}

  static void validate(const OptimizerSpec& spec) {
    /* Globals that consume exactly one starting point per run can't
     * be multistarted in the conventional "K independent restarts"
     * sense: each restart re-seeds a stochastic global from a fresh
     * box-uniform draw, defeating the user-resolved INIT chain. The
     * caps registry tells us which backends accept multiple seeds in
     * a single run; if a global doesn't, refuse MULTISTART > 1.
     * Compatible alternatives are enumerated from the live registry
     * so the message stays accurate as backends are added/removed. */
    const auto& caps = optimizer_caps(spec);
    if (caps.is_global && !caps.accepts_multi_seed) {
      auto compat = filter_optimizer_caps(
          [](const OptimizerCaps& c) {
            return !c.is_global || c.accepts_multi_seed;
          });
      throw std::runtime_error(
          "MULTISTART > 1 is incompatible with " + spec.lib + ":" +
          spec.algo + " (single-seed global). Compatible alternatives: " +
          format_caps_list(compat) + ". Or set N_STARTS 1.");
    }
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
          "MULTISTART " + std::to_string(k_) + " requested but the resolved "
          "INIT_STRATEGY chain produced only " + std::to_string(n_init) +
          " starting point(s); running " + std::to_string(actual_k) +
          " start(s). Add more INIT_STRATEGY entries (e.g. LHS/RANDOM/SOBOL "
          "with K >= " + std::to_string(k_) + ") or lower MULTISTART.");
    }
    HPOLog::instance().info(
        "MULTISTART: running " + std::to_string(actual_k) +
        " starts (parallel=" + std::to_string(parallel_) + ")");

    /* Sequential per-start runs. Parallelism deferred until we have
     * thread-safety audits for each inner adapter; multistart_parallel
     * is parsed and validated but currently not honoured >1. */
    if (parallel_ > 1) {
      HPOLog::instance().warning(
          "MULTISTART_PARALLEL > 1 not yet implemented; running starts "
          "sequentially. The shared THREADS pool still parallelises the "
          "pc_lammps fan-out within each evaluation.");
    }

    /* MULTISTART_BUDGET_PER_START -- override per-start MAXEVAL.
     *
     * If the user set MULTISTART_BUDGET_PER_START explicitly, use it
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
          << (budget_per_start_ ? " (from MULTISTART_BUDGET_PER_START)"
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
      if (spec.seed) sub.seed = *spec.seed + static_cast<unsigned>(i);

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
 * Triggered by the presence of an OptimizerSpec::prestage sub-spec
 * (parsed from a PRESTAGE ... ENDPRESTAGE block). No separate
 * STAGE_GLOBAL flag -- the block's presence is the signal.
 *
 * The two phases:
 *   1. PRESTAGE (broad search):    spec.prestage          (e.g. DLIB:GFS)
 *   2. REFINEMENT (local sweep):   spec     (the outer)   (e.g. DLIB:BFGS)
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class StagedPolicy : public RunPolicy<Vec,Runner> {
public:
  explicit StagedPolicy(std::size_t top_k) : top_k_(top_k) {
    if (top_k_ == 0)
      throw std::runtime_error("Staged-global mode requires TOPK >= 1");
  }

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    if (!spec.prestage) {
      throw std::runtime_error(
          "Staged-global mode requires a PRESTAGE ... ENDPRESTAGE block "
          "inside OPTIMIZER (it defines the global exploration phase; the "
          "outer OPTIMIZER block defines the local refinement).");
    }

    /* Step 1: enable top-K tracking, run the PRESTAGE. Real-space box
     * widths normalise the parameter half of the basin dedup (topk.h). */
    std::vector<double> widths(spec.low.size());
    for (std::size_t d = 0; d < widths.size(); ++d) {
      const bool log_d = d < spec.log_scaled.size() && spec.log_scaled[d];
      widths[d] = log_d ? std::exp(spec.high[d]) - std::exp(spec.low[d])
                        : spec.high[d] - spec.low[d];
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
     * MULTISTART (or a single-shot if K=1). */
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

    HPO_Host<Vec,Runner>::set_stagnation_threshold(window_);
    HPO_Host<Vec,Runner>::reset_stagnation();

    while (true) {
      try {
        inner_->run(cur, host);
        break;
      } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
        if (restarts >= max_restarts_) {
          HPOLog::instance().info(
              "RESTART: stagnation reached, max restarts ("
              + std::to_string(max_restarts_) + ") used; stopping");
          break;
        }
        ++restarts;

        /* Refresh init points. Honor user-supplied INIT_K from
         * RESTART_INIT_STRATEGY block when set; otherwise fall back to
         * each strategy's default_k. */
        if (refresh_ && trg_) {
          std::random_device rd;
          InitContext ic{cur, *trg_, static_cast<std::uint64_t>(rd()) + restarts};
          const std::size_t k_use = (refresh_k_ > 0) ? refresh_k_
                                                     : refresh_->default_k();
          auto pts = refresh_->generate(ic, k_use);
          if (!pts.empty()) cur.optim_init = std::move(pts);
        }
        HPO_Host<Vec,Runner>::reset_stagnation();
        {
          std::ostringstream oss;
          oss << "RESTART " << restarts << "/" << max_restarts_
              << " -- fresh init from "
              << (refresh_ ? refresh_->name() : std::string("(none)"));
          std::cout << "[HPO] " << oss.str() << "\n";
          HPOLog::instance().info(oss.str());
        }
      }
    }
    HPO_Host<Vec,Runner>::set_stagnation_threshold(0);
  }

private:
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
 *    BasePolicy / MultiStartPolicy / StagedPolicy / BasinsPolicy   <- chosen by knobs
 *      +-- wrapped by RestartPolicy if RESTART_AFTER_STAGNATION > 0
 *
 * Selection rules:
 *   - BasinsPolicy if LIB BASINS -- top-level "run and forget" strategy
 *     with its own checkpoint/restart loop. Bypasses the rest of the
 *     stack: BASINS already drives global+local internally and isn't
 *     compatible with PRESTAGE / MULTISTART / RESTART decorators.
 *   - StagedPolicy if a PRESTAGE block is present (spec.prestage != nullptr).
 *   - MultiStartPolicy if MULTISTART > 1.
 *   - BasePolicy otherwise.
 *
 * targets is captured for RestartPolicy's refresh_strategy.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
std::unique_ptr<RunPolicy<Vec,Runner>>
build_run_policy(const OptimizerSpec& spec, const HPOTargets* targets = nullptr);

}  // namespace tadah::hpo

/* BasinsPolicy / BasinsV2Policy are forward-included templates; deferring the
 * build_run_policy implementation past their includes keeps the include
 * graph acyclic. */
#include <tadah/hpo/basins_policy.h>
#include <tadah/hpo/basins_v2_policy.h>

namespace tadah::hpo {

template<class Vec, class Runner>
std::unique_ptr<RunPolicy<Vec,Runner>>
build_run_policy(const OptimizerSpec& spec, const HPOTargets* targets)
{
  /* Strategy block dispatch -- TYPE chooses the inner policy. */
  std::unique_ptr<RunPolicy<Vec,Runner>> inner;
  switch (spec.strategy_type) {
    case StrategyType::BASINS:
      /* BASINS supersedes the decorator stack -- RESTART is not applied. */
      return std::make_unique<BasinsPolicy<Vec,Runner>>(spec, targets);

    case StrategyType::BASINSV2:
      /* BASINSV2 also supersedes the decorator stack -- its lifecycle
       * subsumes RESTART semantics. */
      return std::make_unique<BasinsV2Policy<Vec,Runner>>(spec, targets);

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

  /* RESTART decorator (composes with PLAIN/MULTISTART/PRESTAGE). */
  const auto& rs = spec.restart;
  if (rs.after_stagnation > 0 && rs.max > 0) {
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

#pragma once
/*
 *  BasinsPolicy implementation -- included from basins_policy.h.
 *
 *  Templated; requires the run_inner<>() dispatch from run_policy.h
 *  (already included by basins_policy.h via <tadah/hpo/run_policy.h>).
 */

#include <tadah/core/checkpoint_manager.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/basins_policy.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {

template<class Vec, class Runner>
BasinsPolicy<Vec, Runner>::BasinsPolicy(const OptimizerSpec& outer,
                                        const HPOTargets*    targets)
    : outer_(outer),
      targets_(targets),
      spec_(outer.basins),
      pool_(spec_.basin_dedup_dist, spec_.basin_pool_max)
{
    if (outer_.space)
        pool_.set_space(outer_.space);
    else if (!outer_.low.empty() && outer_.low.size() == outer_.high.size())
        pool_.set_normalisation(outer_.low, outer_.high);
    ensure_ladder();

    /* Register this policy and its basin pool with the global manager.
     * hpo_run() is responsible for installing the SignalHandler and
     * configuring the manager (path / period); calling try_load() AFTER
     * BasinsPolicy is constructed means the load() callbacks below will
     * receive any saved state and resumed_ will become true. */
    auto& mgr = core::CheckpointManager::instance();
    mgr.register_(this);
    mgr.register_(&pool_);
}

template<class Vec, class Runner>
BasinsPolicy<Vec, Runner>::~BasinsPolicy()
{
    auto& mgr = core::CheckpointManager::instance();
    mgr.unregister(&pool_);
    mgr.unregister(this);
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::ensure_ladder()
{
    if (!spec_.explore_ftol_ladder.empty() || !spec_.refine_ftol_ladder.empty()) {
        ladder_ = ToleranceLadder::from_explicit(
                spec_.explore_ftol_ladder, spec_.refine_ftol_ladder);
        /* If only one of the two was provided, fill the other from
         * the default. */
        if (ladder_.glo_ladder().empty() || ladder_.loc_ladder().empty()) {
            auto def = ToleranceLadder::build_default(std::nullopt);
            if (ladder_.glo_ladder().empty())
                ladder_ = ToleranceLadder::from_explicit(
                        def.glo_ladder(), ladder_.loc_ladder());
            if (ladder_.loc_ladder().empty())
                ladder_ = ToleranceLadder::from_explicit(
                        ladder_.glo_ladder(), def.loc_ladder());
        }
    } else {
        ladder_ = ToleranceLadder::build_default(std::nullopt);
    }

    /* Opt-in noise-floor clamp. Default OFF because NOISE_CALIBRATION
     * at a poor seed (BAD_POT-clamped objective) reports a misleading
     * rel_spread. When user enables CLAMP_TO_NOISE_FLOOR true and
     * a noise_floor has been plumbed by hpo_run(), drop ladder rungs
     * strictly below the floor. */
    if (spec_.clamp_to_noise_floor && spec_.noise_floor) {
        const double floor = *spec_.noise_floor;
        const std::size_t before_glo = ladder_.glo_ladder().size();
        const std::size_t before_loc = ladder_.loc_ladder().size();
        ladder_.trim_below(floor);
        const std::size_t after_glo = ladder_.glo_ladder().size();
        const std::size_t after_loc = ladder_.loc_ladder().size();
        if (before_glo != after_glo || before_loc != after_loc) {
            std::ostringstream os;
            os << "BASINS: ladder clamped to noise_floor=" << std::scientific
               << std::setprecision(3) << floor
               << "  glo " << before_glo << "->" << after_glo
               << " rungs, loc " << before_loc << "->" << after_loc
               << " rungs";
            HPOLog::instance().info(os.str());
            /* Sanity warn if we just halved a user-supplied ladder -- the
             * calibration may be unreliable on a poor seed. */
            if ((before_glo > 0 && after_glo * 2 < before_glo) ||
                (before_loc > 0 && after_loc * 2 < before_loc)) {
                HPOLog::instance().warning(
                    "BASINS: noise-floor clamp dropped >50% of ladder rungs -- "
                    "NOISE.CALIBRATION may have run at a poor seed (BAD_POT "
                    "clamps inflate rel_spread). Verify on a converged seed, "
                    "or set CLAMP_TO_NOISE_FLOOR false.");
            }
        }
    }
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::set_noise_floor(double f)
{
    if (!(f > 0.0)) return;   // ignore zero/neg/NaN
    spec_.noise_floor = f;
    ensure_ladder();
}

template<class Vec, class Runner>
OptimizerSpec
BasinsPolicy<Vec, Runner>::build_stage_spec(
        bool global, double tol,
        std::vector<std::vector<double>> seeds) const
{
    /* Start from the user's EXPLORE/REFINE block (LIB/ALGO/PARAMs/etc.).
     * outer_ contains bounds and other shared state we still need. */
    const OptimizerSpec& src = global ? *spec_.explore : *spec_.refine;
    OptimizerSpec s = outer_;        // bounds, log_threshold, optim_init, ...

    /* Override driver + algorithm-specific knobs from the stage block. */
    s.lib  = src.lib;
    s.algo = src.algo;
    s.pop  = src.pop;
    s.seed = src.seed;
    s.step = src.step;
    s.step_overrides = src.step_overrides;
    s.vector_storage = src.vector_storage;
    s.num_threads    = src.num_threads;
    s.param          = src.param;

    /* Per-stage stop criteria. Start from the stage's own stop, then
     * override the per-stage ftol from the BASINS ladder and maxeval
     * from the per-stage budget cap. Other criteria (xtol_*, stopval,
     * maxtime) flow through from the stage block (or from outer_ if
     * the stage didn't set them).
     *
     * The ladder rung is written to whichever ftol field the backend
     * actually honours -- ftol_rel for NLopt, ftol_abs for DLIB:BFGS
     * (which has no relative-objective stop and previously rejected
     * the ladder rung as an unsupported FTOL_REL). The choice lives
     * in OptimizerCaps so backends are swappable without policy edits. */
    if (src.stop.stopval)  s.stop.stopval  = src.stop.stopval;
    if (src.stop.ftol_rel) s.stop.ftol_rel = src.stop.ftol_rel;
    if (src.stop.ftol_abs) s.stop.ftol_abs = src.stop.ftol_abs;
    if (src.stop.xtol_rel) s.stop.xtol_rel = src.stop.xtol_rel;
    if (src.stop.xtol_abs) s.stop.xtol_abs = src.stop.xtol_abs;
    if (src.stop.maxtime)  s.stop.maxtime  = src.stop.maxtime;
    /* Clear any ladder-managed ftol the user previously set on src so
     * that backends with FtolKind::None (e.g. TADAH:ANNEAL) don't
     * inherit a stale rung from EXPLORE/REFINE block-level settings. */
    apply_ladder_ftol(s, tol);
    s.stop.maxeval  = static_cast<uint64_t>(
        global ? spec_.explore_budget_max_evals
               : spec_.refine_budget_max_evals);

    /* Preserve the stage's INNER block (MLSL/AUGLAG inner sub-optimizer).
     * NLopt's set_local_optimizer is consumed for those algorithms; for
     * leaf algorithms NLopt silently ignores it, so always carrying the
     * stage's inner is safe. Resetting it would break BASINS+G_MLSL_LDS. */
    s.inner = src.inner ? std::make_unique<OptimizerSpec>(*src.inner) : nullptr;

    /* Drop strategy nesting -- BASINS sits at the top of the policy stack. */
    s.strategy_type = StrategyType::PLAIN;
    s.basins        = BasinsSpec{};
    s.multistart    = MultiStartSpec{};
    s.restart       = RestartSpec{};
    s.stage_topk    = 1;
    s.prestage.reset();

    /* Seeds for this stage. */
    if (!seeds.empty()) s.optim_init = std::move(seeds);
    return s;
}

template<class Vec, class Runner>
bool BasinsPolicy<Vec, Runner>::overall_termination_requested(
        double t_start_secs) const
{
    if (core::SignalHandler::stop_requested()) return true;

    if (outer_.stop.maxeval) {
        if (HPOLog::instance().n_eval() >= *outer_.stop.maxeval)
            return true;
    }
    if (outer_.stop.maxtime) {
        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch())
            .count() - t_start_secs;
        if (elapsed >= *outer_.stop.maxtime) return true;
    }
    if (spec_.stop_no_improve_rounds > 0 &&
        rounds_no_improve_ >=
            static_cast<int>(spec_.stop_no_improve_rounds)) {
        HPOLog::instance().info(
            "BASINS STOP_NO_IMPROVE: " + std::to_string(rounds_no_improve_) +
            " consecutive rounds without global improvement; stopping");
        return true;
    }
    return false;
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::commit_checkpoint()
{
    try {
        core::CheckpointManager::instance().save_now();
        HPOLog::instance().info(
            "BASINS checkpoint saved: " +
            core::CheckpointManager::instance().path());
    } catch (const std::exception& e) {
        HPOLog::instance().warning(
            std::string("BASINS checkpoint save failed: ") + e.what());
    }
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::run_global_stage(HPO_Host<Vec, Runner>& host)
{
    const double tol = ladder_.glo(round_);

    /* Pick seed(s) for the global stage. With C2 (GLOBAL_SEED_TOP_N > 1)
     * AND a non-empty pool from a prior round, pre-evaluate the top-N
     * basins via host() -- this routes them through the existing top-K
     * capture (so the BasinPool merge picks up any improvement vs. the
     * stored basin loss) and lets us pick the lowest-loss one as the
     * optimiser start. For DLIB:GFS the spec carries all N and the
     * adapter uses them as initial function evaluations; for NLopt the
     * single-best is used as start (set_population only controls the
     * count of internal uniform-random points, not user seeds). */
    std::vector<std::vector<double>> seeds;
    const std::size_t top_n =
        std::max<std::size_t>(spec_.seed_top_n, 1u);
    if (!pool_.empty()) {
        const auto heads = pool_.heads();
        if (top_n == 1u || heads.size() == 1u) {
            seeds.push_back(heads.front().params);
        } else {
            const std::size_t k = std::min(top_n, heads.size());
            std::vector<std::pair<std::vector<double>, double>> evaluated;
            evaluated.reserve(k);
            /* Pre-evaluate. host() updates the top-K tracker and global
             * best-so-far comparator. We collect (params, loss) so we
             * can pick the lowest-loss one as the optimiser start. */
            for (std::size_t i = 0; i < k; ++i) {
                Vec v;
                v.set_size(static_cast<long>(heads[i].params.size()));
                for (long j = 0; j < v.size(); ++j)
                    v(j) = heads[i].params[static_cast<std::size_t>(j)];
                /* Basin params are REAL-space (BasinPool's contract), as
                 * host() expects. */
                const double ls = host(v);
                evaluated.emplace_back(heads[i].params, ls);
            }
            std::sort(evaluated.begin(), evaluated.end(),
                      [](const auto& a, const auto& b){ return a.second < b.second; });
            seeds.push_back(std::move(evaluated.front().first));
            {
                std::ostringstream os;
                os << "BASINS round=" << round_ << " GLOBAL pre-eval top-"
                   << k << ": best of pre-eval loss="
                   << std::scientific << std::setprecision(3)
                   << evaluated.front().second
                   << "  (used as optimiser start)";
                HPOLog::instance().info(os.str());
            }
        }
    } else if (!outer_.optim_init.empty()) {
        /* Use the first user-resolved init point. */
        seeds.push_back(outer_.optim_init.front());
    } else {
        seeds.clear();   // fall back to whatever the optimiser does
    }

    /* Top-K capture during the global. We collect 2 * basin_pool_max
     * candidates to give the dedup logic room to work. */
    const std::size_t topk = std::max<std::size_t>(spec_.basin_pool_max, 8) * 2;
    HPO_Host<Vec, Runner>::clear_topk();
    HPO_Host<Vec, Runner>::set_track_topk(topk);

    /* Per-stage stagnation cap: short-circuit the optimiser if no new
     * best is found within max(BUDGET_MIN_EVALS, BUDGET_MAX_EVALS *
     * BUDGET_RATE_FACTOR) evals. The user controls all three. */
    HPO_Host<Vec, Runner>::reset_stagnation();
    HPO_Host<Vec, Runner>::set_stagnation_threshold(
            std::max<std::size_t>(
                spec_.explore_budget_min_evals,
                static_cast<std::size_t>(spec_.explore_budget_max_evals
                                         * spec_.explore_budget_rate_factor)));

    auto sub = build_stage_spec(/*global=*/true, tol, std::move(seeds));

    {
        /* "ftol" not "ftol_rel": apply_ladder_ftol writes the rung to
         * whichever ftol field the backend honours (ftol_rel for NLopt,
         * ftol_abs for DLIB:BFGS), so the literal field name depends
         * on the backend. */
        std::ostringstream os;
        os << "BASINS round=" << round_ << " EXPLORE begin: "
           << sub.lib << ":" << sub.algo
           << "  ftol=" << std::scientific << std::setprecision(3) << tol
           << "  maxeval=" << spec_.explore_budget_max_evals;
        HPOLog::instance().info(os.str());
    }

    try {
        run_inner<Vec, Runner>(sub, host);
    } catch (const typename HPO_Host<Vec, Runner>::StagnationStop&) {
        /* Expected -- stage is done. */
        if (HPOLog::instance().unsaved_best()) {
            /* A new best could not be saved meanwhile (it may have
             * lost to this stop in a parallel stage): fail now. */
            HPO_Host<Vec, Runner>::set_track_topk(0);
            HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
            detail::rethrow_if_best_unsaved();
        }
    } catch (const core::ShutdownRequested&) {
        /* Honour shutdown -- flush top-K into the pool, commit, rethrow. */
        auto found = HPO_Host<Vec, Runner>::get_topk();
        for (auto& kv : found) {
            std::vector<double> p(kv.second.size());
            for (long i = 0; i < kv.second.size(); ++i) p[i] = kv.second(i);
            pool_.add_or_merge(std::move(p), kv.first, round_);
        }
        HPO_Host<Vec, Runner>::set_track_topk(0);
        HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
        throw;
    }

    /* Harvest top-K -> BasinPool. */
    auto found = HPO_Host<Vec, Runner>::get_topk();
    int n_added = 0;
    for (const auto& kv : found) {
        std::vector<double> p(kv.second.size());
        for (long i = 0; i < kv.second.size(); ++i) p[i] = kv.second(i);
        const int idx = pool_.add_or_merge(std::move(p), kv.first, round_);
        if (idx >= 0) ++n_added;
    }
    HPO_Host<Vec, Runner>::set_track_topk(0);
    HPO_Host<Vec, Runner>::clear_topk();
    HPO_Host<Vec, Runner>::set_stagnation_threshold(0);

    {
        std::ostringstream os;
        os << "BASINS round=" << round_ << " GLOBAL end: harvested "
           << found.size() << " candidate(s); pool now " << pool_.size()
           << " basin(s)";
        HPOLog::instance().info(os.str());
    }
    (void)n_added;
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::run_local_stage(HPO_Host<Vec, Runner>& host,
                                                int loc_step)
{
    const double tol = ladder_.loc(round_, loc_step);
    auto due = pool_.not_exhausted_at(tol);

    {
        std::ostringstream os;
        os << "BASINS round=" << round_ << " LOCAL step=" << loc_step
           << " ftol=" << std::scientific << std::setprecision(3) << tol
           << "  basins=" << due.size() << "/" << pool_.size();
        HPOLog::instance().info(os.str());
    }

    if (due.empty()) return;

    /* Per-basin LOCAL refinements. We capture each refinement's per-stage
     * minimum (top-K=1) and feed it back into the pool via add_or_merge.
     * Without this, every basin's pool entry is frozen at its round-1
     * GLOBAL discovery (params, loss) for the entire run -- even when its
     * LOCAL refinement substantially improves the loss. Downstream effect:
     * round-k>1 GLOBAL re-seeds from stale heads(), and the pool ordering
     * by loss becomes meaningless once any basin actually refines below
     * its initial value. Updates are batched until end-of-stage because
     * add_or_merge can dedup-merge or evict, shifting basin indices --
     * applying mid-loop would invalidate the remaining `due` indices. */
    std::vector<std::pair<std::vector<double>, double>> pending_updates;

    auto flush_pool_updates = [&]() {
        HPO_Host<Vec, Runner>::set_track_topk(0);
        HPO_Host<Vec, Runner>::clear_topk();
        for (auto& upd : pending_updates)
            pool_.add_or_merge(std::move(upd.first), upd.second, round_);
        pending_updates.clear();
    };

    try {
        for (std::size_t bi : due) {
            if (core::SignalHandler::stop_requested()) {
                throw core::ShutdownRequested{};
            }

            const auto basin_snapshot = pool_.heads_view()[bi];
            std::vector<std::vector<double>> seeds = { basin_snapshot.params };

            /* Per-basin stagnation cap (REFINE): max(BUDGET_MIN_EVALS,
             * BUDGET_MAX_EVALS * BUDGET_RATE_FACTOR), same shape as
             * EXPLORE. */
            HPO_Host<Vec, Runner>::reset_stagnation();
            HPO_Host<Vec, Runner>::set_stagnation_threshold(
                    std::max<std::size_t>(
                        spec_.refine_budget_min_evals,
                        static_cast<std::size_t>(spec_.refine_budget_max_evals
                                                 * spec_.refine_budget_rate_factor)));

            HPO_Host<Vec, Runner>::clear_topk();
            HPO_Host<Vec, Runner>::set_track_topk(1);

            auto sub = build_stage_spec(/*global=*/false, tol, std::move(seeds));

            const double best_before = HPOLog::instance().best_loss();
            bool optimizer_converged_naturally = false;
            try {
                run_inner<Vec, Runner>(sub, host);
                optimizer_converged_naturally = true;
            } catch (const typename HPO_Host<Vec, Runner>::StagnationStop&) {
                /* Stage stagnated; treat the basin as exhausted-at-tol. */
                if (HPOLog::instance().unsaved_best()) {
                    /* A new best could not be saved meanwhile (it may have
                     * lost to this stop in a parallel stage): fail now. */
                    HPO_Host<Vec, Runner>::set_track_topk(0);
                    HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
                    detail::rethrow_if_best_unsaved();
                }
            }

            HPO_Host<Vec, Runner>::set_stagnation_threshold(0);

            const double best_after = HPOLog::instance().best_loss();
            pool_.mark_attempted(bi, tol);
            if (optimizer_converged_naturally || best_after >= best_before) {
                pool_.mark_exhausted(bi, tol);
            }

            auto found = HPO_Host<Vec, Runner>::get_topk();
            for (const auto& kv : found) {
                std::vector<double> p(kv.second.size());
                for (long i = 0; i < kv.second.size(); ++i)
                    p[i] = kv.second(i);
                pending_updates.emplace_back(std::move(p), kv.first);
            }
            HPO_Host<Vec, Runner>::clear_topk();

            {
                std::ostringstream os;
                os << "BASINS basin[" << bi << "] tol=" << std::scientific
                   << std::setprecision(3) << tol
                   << "  loss=" << best_after
                   << (best_after < best_before ? " (improved)"
                                                : " (no improvement)");
                HPOLog::instance().info(os.str());
            }
        }
    } catch (const core::ShutdownRequested&) {
        /* Flush whatever we harvested so the checkpoint reflects partial
         * refinement progress, then rethrow for caller's commit. */
        flush_pool_updates();
        throw;
    }

    flush_pool_updates();
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::run(const OptimizerSpec& /*spec*/,
                                    HPO_Host<Vec, Runner>& host)
{
    using clock = std::chrono::steady_clock;
    const double t_start_secs =
        std::chrono::duration<double>(clock::now().time_since_epoch()).count();

    if (resumed_) {
        HPOLog::instance().info(
            "BASINS resumed at round=" + std::to_string(round_) +
            " stage=" + (stage_ == Stage::GLOBAL ? "GLOBAL" : "LOCAL") +
            " loc_step=" + std::to_string(current_loc_step_) +
            " pool=" + std::to_string(pool_.size()) + " basin(s)");
    } else {
        round_              = 1;
        stage_              = Stage::GLOBAL;
        current_loc_step_   = 0;
        rounds_no_improve_  = 0;
        best_at_round_start_ = HPOLog::instance().best_loss();
    }

    while (true) {
        if (overall_termination_requested(t_start_secs)) break;

        if (stage_ == Stage::GLOBAL) {
            best_at_round_start_ = HPOLog::instance().best_loss();
            try {
                run_global_stage(host);
            } catch (const core::ShutdownRequested&) {
                commit_checkpoint();
                throw;
            }
            stage_            = Stage::LOCAL;
            current_loc_step_ = 1;
            commit_checkpoint();
        }

        if (overall_termination_requested(t_start_secs)) break;

        if (stage_ == Stage::LOCAL) {
            const int max_step = ladder_.loc_steps_for(round_);
            for (; current_loc_step_ <= max_step; ++current_loc_step_) {
                if (overall_termination_requested(t_start_secs)) break;
                try {
                    run_local_stage(host, current_loc_step_);
                } catch (const core::ShutdownRequested&) {
                    commit_checkpoint();
                    throw;
                }
                commit_checkpoint();
            }

            /* End of round bookkeeping. */
            const double best_after = HPOLog::instance().best_loss();
            if (best_after < best_at_round_start_)
                rounds_no_improve_ = 0;
            else
                ++rounds_no_improve_;

            {
                std::ostringstream os;
                os << "BASINS round=" << round_ << " complete: best="
                   << std::scientific << std::setprecision(6) << best_after
                   << " rounds_no_improve=" << rounds_no_improve_
                   << "/" << spec_.stop_no_improve_rounds
                   << " pool=" << pool_.size();
                HPOLog::instance().info(os.str());
            }

            ++round_;
            stage_            = Stage::GLOBAL;
            current_loc_step_ = 0;
            commit_checkpoint();
        }
    }

    HPOLog::instance().info(
        "BASINS exiting cleanly at round=" + std::to_string(round_) +
        " stage=" + (stage_ == Stage::GLOBAL ? "GLOBAL" : "LOCAL") +
        " pool=" + std::to_string(pool_.size()) + " basin(s)");
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::save(core::Checkpoint& cp) const
{
    cp.section(tag());
    cp.put_int ("round",              round_);
    cp.put_int ("stage",
                stage_ == Stage::GLOBAL ? 0 : 1);
    cp.put_int ("current_loc_step",   current_loc_step_);
    cp.put_int ("rounds_no_improve",  rounds_no_improve_);
    cp.put_dbl ("best_at_round_start", best_at_round_start_);
}

template<class Vec, class Runner>
void BasinsPolicy<Vec, Runner>::load(const core::Checkpoint& cp)
{
    if (auto v = cp.get_int(tag(), "round"))
        round_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "stage"))
        stage_ = (*v == 0) ? Stage::GLOBAL : Stage::LOCAL;
    if (auto v = cp.get_int(tag(), "current_loc_step"))
        current_loc_step_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "rounds_no_improve"))
        rounds_no_improve_ = static_cast<int>(*v);
    if (auto v = cp.get_dbl(tag(), "best_at_round_start"))
        best_at_round_start_ = *v;
    resumed_ = true;
}

}   // namespace hpo
}   // namespace tadah

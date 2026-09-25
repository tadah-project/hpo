#pragma once
/*
 *  BasinsV2Policy implementation -- included from basins_v2_policy.h.
 *
 *  Templated; uses run_inner<>() from run_policy_base.h. The dispatch
 *  shape mirrors v1's BasinsPolicy at the optimiser-invocation layer
 *  (so optimiser caps, INNER block plumbing, and stagnation-stop
 *  semantics all work unchanged), but the surrounding control flow is
 *  the v2 priority scheduler.
 */

#include <tadah/core/checkpoint_manager.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/basins_v2_explore.h>
#include <tadah/hpo/basins_v2_path.h>
#include <tadah/hpo/basins_v2_policy.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {

template<class Vec, class Runner>
BasinsV2Policy<Vec, Runner>::BasinsV2Policy(const OptimizerSpec& outer,
                                            const HPOTargets*    targets)
    : outer_(outer),
      targets_(targets),
      spec_(outer.basins_v2),
      pool_(spec_.basin_dedup_dist, spec_.basin_pool_max, spec_.archive_max)
{
    /* Mirror priority weights from spec into the local weights_. The
     * spec is the single source of truth for user overrides; the
     * weights_ struct exists for clean parameter passing. */
    weights_.improve  = spec_.w_improve;
    weights_.neglect  = spec_.w_neglect;
    weights_.distance = spec_.w_distance;
    weights_.tol_gap  = spec_.w_tol_gap;
    weights_.state    = spec_.w_state;
    weights_.age      = spec_.w_age;

    if (outer_.space)
        pool_.set_space(outer_.space);
    else if (!outer_.low.empty() && outer_.low.size() == outer_.high.size())
        pool_.set_normalisation(outer_.low, outer_.high);
    ensure_ladder();

    auto& mgr = core::CheckpointManager::instance();
    mgr.register_(this);
    mgr.register_(&pool_);
}

template<class Vec, class Runner>
BasinsV2Policy<Vec, Runner>::~BasinsV2Policy()
{
    auto& mgr = core::CheckpointManager::instance();
    mgr.unregister(&pool_);
    mgr.unregister(this);
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::ensure_ladder()
{
    if (!spec_.explore_ftol_ladder.empty() || !spec_.refine_ftol_ladder.empty()) {
        ladder_ = ToleranceLadder::from_explicit(
                spec_.explore_ftol_ladder, spec_.refine_ftol_ladder);
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

    if (spec_.clamp_to_noise_floor && spec_.noise_floor) {
        const double floor = *spec_.noise_floor;
        const std::size_t before_glo = ladder_.glo_ladder().size();
        const std::size_t before_loc = ladder_.loc_ladder().size();
        ladder_.trim_below(floor);
        const std::size_t after_glo = ladder_.glo_ladder().size();
        const std::size_t after_loc = ladder_.loc_ladder().size();
        if (before_glo != after_glo || before_loc != after_loc) {
            std::ostringstream os;
            os << "BASINSv2: ladder clamped to noise_floor=" << std::scientific
               << std::setprecision(3) << floor
               << "  glo " << before_glo << "->" << after_glo
               << "  loc " << before_loc << "->" << after_loc;
            HPOLog::instance().info(os.str());
        }
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::set_noise_floor(double f)
{
    if (!(f > 0.0)) return;
    spec_.noise_floor = f;
    ensure_ladder();
}

template<class Vec, class Runner>
double BasinsV2Policy<Vec, Runner>::pick_initial_tol_for_new_basin() const
{
    /* Newly discovered basins enter the queue at the loosest local rung
     * -- they need exploring before deeper refinement is justified. */
    const auto& loc = ladder_.loc_ladder();
    if (loc.empty()) return 1.0e-3;
    /* The ladder is stored sorted-descending (loosest first) by
     * ToleranceLadder convention; loc[0] is the loosest rung. */
    return loc.front();
}

template<class Vec, class Runner>
std::size_t
BasinsV2Policy<Vec, Runner>::adaptive_refine_threshold(double tol) const
{
    /* Pure formula lives in basins_v2_priority.cpp so unit tests can
     * exercise it without a full policy instantiation. */
    const std::size_t dim = static_cast<std::size_t>(outer_.low.size());
    return tadah::hpo::adaptive_refine_threshold(
        tol, dim,
        spec_.refine_budget_min_evals,
        spec_.refine_budget_max_evals,
        spec_.plateau_eps_factor);
}

template<class Vec, class Runner>
std::vector<double>
BasinsV2Policy<Vec, Runner>::compute_min_dist_to_other_active() const
{
    const auto& basins = pool_.view();
    std::vector<double> out(basins.size(),
                            std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < basins.size(); ++i) {
        if (is_archive_state(basins[i].state)) continue;
        for (std::size_t j = 0; j < basins.size(); ++j) {
            if (i == j) continue;
            if (is_archive_state(basins[j].state)) continue;
            const double d = pool_.relative_l2(basins[i].params, basins[j].params);
            if (d < out[i]) out[i] = d;
        }
    }
    return out;
}

template<class Vec, class Runner>
OptimizerSpec
BasinsV2Policy<Vec, Runner>::build_stage_spec(
        bool global, double tol,
        std::vector<std::vector<double>> seeds) const
{
    /* Same shape as v1: copy outer_, override driver/algo from the
     * stage block, write tol into whichever ftol field the backend
     * honours, set per-stage maxeval. */
    const OptimizerSpec& src = global ? *spec_.explore : *spec_.refine;
    OptimizerSpec s = outer_;

    s.lib  = src.lib;
    s.algo = src.algo;
    s.pop  = src.pop;
    s.seed = src.seed;
    s.step = src.step;
    s.step_overrides = src.step_overrides;
    s.vector_storage = src.vector_storage;
    s.num_threads    = src.num_threads;
    s.param          = src.param;

    if (src.stop.stopval)  s.stop.stopval  = src.stop.stopval;
    if (src.stop.ftol_rel) s.stop.ftol_rel = src.stop.ftol_rel;
    if (src.stop.ftol_abs) s.stop.ftol_abs = src.stop.ftol_abs;
    if (src.stop.xtol_rel) s.stop.xtol_rel = src.stop.xtol_rel;
    if (src.stop.xtol_abs) s.stop.xtol_abs = src.stop.xtol_abs;
    if (src.stop.maxtime)  s.stop.maxtime  = src.stop.maxtime;
    apply_ladder_ftol(s, tol);
    s.stop.maxeval = static_cast<uint64_t>(
        global ? spec_.explore_budget_max_evals
               : spec_.refine_budget_max_evals);

    s.inner = src.inner ? std::make_unique<OptimizerSpec>(*src.inner) : nullptr;

    /* Drop nesting -- BASINSV2 sits at the top of the policy stack. */
    s.strategy_type = StrategyType::PLAIN;
    s.basins        = BasinsSpec{};
    s.basins_v2     = BasinsV2Spec{};
    s.multistart    = MultiStartSpec{};
    s.restart       = RestartSpec{};
    s.stage_topk    = 1;
    s.prestage.reset();

    if (!seeds.empty()) s.optim_init = std::move(seeds);
    return s;
}

template<class Vec, class Runner>
bool BasinsV2Policy<Vec, Runner>::overall_termination_requested(
        double t_start_secs) const
{
    if (core::SignalHandler::stop_requested()) return true;

    if (outer_.stop.maxeval) {
        if (HPOLog::instance().n_eval() >= *outer_.stop.maxeval)
            return true;
    }
    if (outer_.stop.maxtime) {
        const double elapsed_now =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch())
            .count() - t_start_secs;
        if (elapsed_now + wall_seconds_accumulated_
            >= *outer_.stop.maxtime) return true;
    }
    /* Iteration-based stop_no_improve: how many iterations since the
     * last *global* best improvement. */
    if (spec_.stop_no_improve_iterations > 0 &&
        iteration_ - iter_at_last_global_best_ >=
            static_cast<int>(spec_.stop_no_improve_iterations)) {
        HPOLog::instance().info(
            "BASINSv2 STOP_NO_IMPROVE: " +
            std::to_string(iteration_ - iter_at_last_global_best_) +
            " iterations without global best improvement; stopping");
        return true;
    }
    return false;
}

template<class Vec, class Runner>
typename BasinsV2Policy<Vec, Runner>::WorkItem
BasinsV2Policy<Vec, Runner>::select_work()
{
    WorkItem w;

    /* (1) bootstrap: empty pool -> EXPLORE. */
    if (pool_.empty() || pool_.active_count() == 0) {
        w.kind = WorkKind::EXPLORE;
        return w;
    }

    /* (2) merge cadence (Phase 2 implements the work; Phase 1's
     *     do_merge_pass is a no-op stub). Cheap, no LAMMPS evals. */
    if (spec_.merge_cadence > 0 &&
        iteration_ - last_merge_iteration_ >= static_cast<int>(spec_.merge_cadence) &&
        pool_.active_count() > 1) {
        w.kind = WorkKind::MERGE_PASS;
        return w;
    }

    /* (3) verify-kill: any INACTIVE basin past verify_cadence? */
    if (spec_.verify_cadence > 0) {
        const auto& basins = pool_.view();
        for (std::size_t i = 0; i < basins.size(); ++i) {
            if (basins[i].state != LifecycleState::INACTIVE) continue;
            if (iteration_ - basins[i].last_iteration >=
                static_cast<int>(spec_.verify_cadence)) {
                if (basins[i].verify_attempts >=
                    static_cast<int>(spec_.verify_max_attempts)) {
                    /* Out of attempts -- caller transitions to KILLED in
                     * update_lifecycle(); skip here. */
                    continue;
                }
                w.kind     = WorkKind::VERIFY_KILL;
                w.basin_id = basins[i].id;
                return w;
            }
        }
    }

    /* (4) starvation backstop: any active basin neglected past cap? */
    if (spec_.starvation_cap > 0) {
        const auto& basins = pool_.view();
        int worst_id = -1;
        int worst_neglect = 0;
        for (const auto& b : basins) {
            if (is_archive_state(b.state)) continue;
            if (b.state == LifecycleState::INACTIVE) continue;
            const int neg = iteration_ - b.last_iteration;
            if (neg > worst_neglect) {
                worst_neglect = neg;
                worst_id = b.id;
            }
        }
        if (worst_id >= 0 &&
            worst_neglect > static_cast<int>(spec_.starvation_cap)) {
            const BasinV2* b = pool_.by_id(worst_id);
            w.kind     = WorkKind::REFINE;
            w.basin_id = worst_id;
            w.tol      = b ? b->current_tol_cursor
                           : pick_initial_tol_for_new_basin();
            return w;
        }
    }

    /* (5) explore-kick: every explore_kick_cadence iterations force
     *     fresh-region EXPLORE regardless of pool state. */
    if (spec_.explore_kick_cadence > 0 &&
        iteration_ - last_explore_iteration_ >=
            static_cast<int>(spec_.explore_kick_cadence)) {
        w.kind = WorkKind::EXPLORE;
        return w;
    }

    /* (6) priority REFINE: argmax over active pool. */
    auto min_dist = compute_min_dist_to_other_active();
    PriorityContext ctx;
    ctx.current_iteration = iteration_;
    ctx.target_tol        = ladder_.loc_ladder().empty()
        ? 1.0e-9 : ladder_.loc_ladder().back();
    auto scores = compute_priorities(pool_.view(), min_dist, ctx, weights_);
    const int best_idx = argmax_priority(pool_.view(), scores);
    if (best_idx < 0) {
        /* No active basin -- fall back to EXPLORE. */
        w.kind = WorkKind::EXPLORE;
        return w;
    }
    const BasinV2& b = pool_.view()[static_cast<std::size_t>(best_idx)];
    w.kind     = WorkKind::REFINE;
    w.basin_id = b.id;
    w.tol      = b.current_tol_cursor;
    return w;
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::do_explore(HPO_Host<Vec, Runner>& host)
{
    last_explore_iteration_ = iteration_;

    /* Pick global-stage tol. Used as the local-refinement ftol for the
     * brief polish step on Sobol survivors. */
    const auto& glo = ladder_.glo_ladder();
    const std::size_t step_idx =
        glo.empty() ? 0
                    : std::min<std::size_t>(
                          static_cast<std::size_t>(iteration_),
                          glo.size() - 1u);
    const double tol = glo.empty() ? 1.0e-3 : glo[step_idx];

    const std::size_t dim = static_cast<std::size_t>(outer_.low.size());
    if (dim == 0) {
        HPOLog::instance().warning(
            "BASINSv2 EXPLORE: optimiser bounds are empty; skipping iteration");
        return;
    }

    {
        std::ostringstream os;
        os << "BASINSv2 iter=" << iteration_
           << " select=EXPLORE tol=" << std::scientific << std::setprecision(3) << tol;
        HPOLog::instance().info(os.str());
    }

    /* (1) Sobol candidate generation. K = max(8, 2*dim). seed_skip
     *     stamped from iteration so successive EXPLOREs sample
     *     different sub-sequences. */
    const std::size_t K =
        std::max<std::size_t>(8u, 2u * dim);
    const auto unit_pts = sobol_unit_points(
        K, dim, /*seed_skip=*/static_cast<std::uint64_t>(iteration_ + 1));

    /* Map [0,1]^dim -> INTERNAL box -> REAL space (outer_.out). The pool
     * and every seed handed to a stage optimiser are REAL-space. */
    std::vector<std::vector<double>> real_pts;
    real_pts.reserve(unit_pts.size());
    for (const auto& u : unit_pts) {
        std::vector<double> tx(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            tx[d] = outer_.low[d] + u[d] * (outer_.high[d] - outer_.low[d]);
        }
        real_pts.push_back(outer_.out(tx));
    }

    /* (2) Evaluate each Sobol point. host() expects a Vec in real
     *     space and returns the loss. */
    std::vector<double> losses;
    losses.reserve(real_pts.size());
    try {
        for (const auto& rp : real_pts) {
            if (core::SignalHandler::stop_requested())
                throw core::ShutdownRequested{};
            Vec v;
            v.set_size(static_cast<long>(rp.size()));
            for (long i = 0; i < v.size(); ++i)
                v(i) = rp[static_cast<std::size_t>(i)];
            losses.push_back(host(v));
        }
    } catch (const core::ShutdownRequested&) {
        /* Push whatever Sobol points we managed to evaluate into the
         * pool before propagating -- better than losing them. */
        for (std::size_t i = 0; i < losses.size(); ++i) {
            const int new_id = pool_.add_or_merge(
                real_pts[i], losses[i], iteration_);
            if (new_id >= 0) {
                if (BasinV2* b = pool_.by_id(new_id))
                    if (b->birth_iteration == iteration_)
                        b->current_tol_cursor = pick_initial_tol_for_new_basin();
            }
        }
        throw;
    }

    /* (3) Filter survivors that are far from any known basin /
     *     coarse path. */
    auto survivor_idx = filter_far_from_pool(
        real_pts, pool_, spec_.explore_dist_floor);

    const std::size_t n_filtered_out = real_pts.size() - survivor_idx.size();

    /* (4) Top-M survivors by loss. */
    std::vector<double> survivor_losses;
    survivor_losses.reserve(survivor_idx.size());
    for (auto i : survivor_idx) survivor_losses.push_back(losses[i]);
    const std::size_t M = std::max<std::size_t>(
        2u, std::max<std::size_t>(K / 4u, 2u));
    auto topm_local = top_m_by_loss(survivor_losses, M);
    std::vector<std::size_t> topm_global;
    topm_global.reserve(topm_local.size());
    for (auto k : topm_local) topm_global.push_back(survivor_idx[k]);

    {
        std::ostringstream os;
        os << "BASINSv2 EXPLORE Sobol K=" << K
           << " evaluated, M=" << topm_global.size()
           << " survived (filter dropped " << n_filtered_out
           << " in known regions)";
        HPOLog::instance().info(os.str());
    }

    /* (5) Brief local refinement of survivors using the EXPLORE block's
     *     LIB/ALGO (default LN_BOBYQA). Cap budget at 30 * dim evals
     *     per survivor -- the goal is to find each Sobol survivor's
     *     basin endpoint, not refine it. */
    const std::size_t cap_per_survivor =
        std::max<std::size_t>(30u * dim, spec_.explore_budget_min_evals);

    int n_added = 0, n_merged = 0;
    for (auto idx : topm_global) {
        if (core::SignalHandler::stop_requested())
            throw core::ShutdownRequested{};

        /* Insert the raw Sobol point first so it survives even if the
         * local refinement fails. */
        const int seeded_id = pool_.add_or_merge(
            real_pts[idx], losses[idx], iteration_);
        if (seeded_id < 0) continue;

        BasinV2* b_seed = pool_.by_id(seeded_id);
        if (b_seed && b_seed->birth_iteration == iteration_) {
            b_seed->current_tol_cursor = pick_initial_tol_for_new_basin();
            ++n_added;
        } else {
            ++n_merged;
        }

        /* Local refinement from this Sobol point. */
        std::vector<std::vector<double>> seeds = { real_pts[idx] };
        HPO_Host<Vec, Runner>::reset_stagnation();
        HPO_Host<Vec, Runner>::set_stagnation_threshold(cap_per_survivor);
        HPO_Host<Vec, Runner>::clear_topk();
        HPO_Host<Vec, Runner>::set_track_topk(1);

        auto sub = build_stage_spec(/*global=*/true, tol, std::move(seeds));
        sub.stop.maxeval = static_cast<uint64_t>(cap_per_survivor);

        const std::size_t evals_before = HPOLog::instance().n_eval();
        try {
            run_inner<Vec, Runner>(sub, host);
        } catch (const typename HPO_Host<Vec, Runner>::StagnationStop&) {
            /* expected */
            if (HPOLog::instance().unsaved_best()) {
                /* A new best could not be saved meanwhile (it may have
                 * lost to this stop in a parallel stage): fail now. */
                HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
                HPO_Host<Vec, Runner>::clear_topk();
                detail::rethrow_if_best_unsaved();
            }
        } catch (const core::ShutdownRequested&) {
            HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
            HPO_Host<Vec, Runner>::clear_topk();
            throw;
        }
        HPO_Host<Vec, Runner>::set_stagnation_threshold(0);

        const std::size_t evals_used =
            HPOLog::instance().n_eval() - evals_before;

        auto found = HPO_Host<Vec, Runner>::get_topk();
        if (!found.empty()) {
            const auto& kv = found.front();
            std::vector<double> p(kv.second.size());
            for (long i = 0; i < kv.second.size(); ++i) p[i] = kv.second(i);
            /* Update the seeded basin (or merge into a closer one). */
            pool_.on_refinement_complete(
                seeded_id, std::move(p), kv.first,
                evals_used, iteration_, tol, /*converged=*/false);
        }
        HPO_Host<Vec, Runner>::clear_topk();
    }

    {
        std::ostringstream os;
        os << "BASINSv2 EXPLORE end: added=" << n_added
           << " merged=" << n_merged
           << " filtered_out=" << n_filtered_out
           << " active=" << pool_.active_count();
        HPOLog::instance().info(os.str());
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::do_refine(HPO_Host<Vec, Runner>& host,
                                            int basin_id, double tol)
{
    const BasinV2* b = pool_.by_id(basin_id);
    if (!b) return;
    if (is_archive_state(b->state)) return;

    std::vector<std::vector<double>> seeds = { b->params };

    HPO_Host<Vec, Runner>::reset_stagnation();
    const std::size_t threshold = adaptive_refine_threshold(tol);
    HPO_Host<Vec, Runner>::set_stagnation_threshold(threshold);

    HPO_Host<Vec, Runner>::clear_topk();
    HPO_Host<Vec, Runner>::set_track_topk(1);

    auto sub = build_stage_spec(/*global=*/false, tol, std::move(seeds));

    const double best_before = HPOLog::instance().best_loss();
    const std::size_t evals_before = HPOLog::instance().n_eval();

    {
        std::ostringstream os;
        os << "BASINSv2 iter=" << iteration_ << " select=REFINE basin[id="
           << basin_id << "] tol=" << std::scientific << std::setprecision(3) << tol
           << " ema=" << b->improvement_ema
           << " stagn_thr=" << threshold;
        HPOLog::instance().info(os.str());
    }

    bool optimizer_converged_naturally = false;
    try {
        run_inner<Vec, Runner>(sub, host);
        optimizer_converged_naturally = true;
    } catch (const typename HPO_Host<Vec, Runner>::StagnationStop&) {
        /* plateau detected */
        if (HPOLog::instance().unsaved_best()) {
            /* A new best could not be saved meanwhile (it may have
             * lost to this stop in a parallel stage): fail now. */
            HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
            HPO_Host<Vec, Runner>::clear_topk();
            detail::rethrow_if_best_unsaved();
        }
    } catch (const core::ShutdownRequested&) {
        HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
        throw;
    }
    HPO_Host<Vec, Runner>::set_stagnation_threshold(0);

    const std::size_t evals_used = HPOLog::instance().n_eval() - evals_before;
    const double best_after = HPOLog::instance().best_loss();

    auto found = HPO_Host<Vec, Runner>::get_topk();
    /* Phase 4: capture a path snapshot of the start point BEFORE the
     * pool update, so the snapshot reflects the basin's state at the
     * start of this refinement. The end-point snapshot follows the
     * pool update. */
    if (spec_.path_max_points > 0) {
        const BasinV2* b_pre = pool_.by_id(basin_id);
        if (b_pre) {
            auto dist_fn = [this](const std::vector<double>& a,
                                  const std::vector<double>& c) {
                return pool_.relative_l2(a, c);
            };
            if (should_snapshot(b_pre->coarse_path, b_pre->params, b_pre->loss,
                                spec_.path_loss_eps, spec_.path_dist_eps,
                                dist_fn)) {
                PathPoint pp;
                pp.params = b_pre->params;
                pp.loss = b_pre->loss;
                pp.eval_index = evals_before;
                pool_.append_path_point(basin_id, std::move(pp), spec_.path_max_points);
            }
        }
    }

    if (!found.empty()) {
        const auto& kv = found.front();
        std::vector<double> p(kv.second.size());
        for (long i = 0; i < kv.second.size(); ++i) p[i] = kv.second(i);
        pool_.on_refinement_complete(basin_id, std::move(p), kv.first,
                                     evals_used, iteration_, tol,
                                     optimizer_converged_naturally);
    } else {
        /* No top-K captured -- still record the attempt for tol bookkeeping. */
        const BasinV2* now = pool_.by_id(basin_id);
        std::vector<double> snap = now ? now->params : std::vector<double>{};
        const double snap_loss = now ? now->loss : best_after;
        pool_.on_refinement_complete(basin_id, std::move(snap), snap_loss,
                                     evals_used, iteration_, tol,
                                     optimizer_converged_naturally);
    }
    HPO_Host<Vec, Runner>::clear_topk();

    /* Phase 4: post-refinement snapshot. */
    if (spec_.path_max_points > 0) {
        const BasinV2* b_post = pool_.by_id(basin_id);
        if (b_post) {
            auto dist_fn = [this](const std::vector<double>& a,
                                  const std::vector<double>& c) {
                return pool_.relative_l2(a, c);
            };
            if (should_snapshot(b_post->coarse_path, b_post->params, b_post->loss,
                                spec_.path_loss_eps, spec_.path_dist_eps,
                                dist_fn)) {
                PathPoint pp;
                pp.params = b_post->params;
                pp.loss = b_post->loss;
                pp.eval_index = HPOLog::instance().n_eval();
                pool_.append_path_point(basin_id, std::move(pp), spec_.path_max_points);
            }
        }
    }

    /* Advance basin's tol cursor when the optimiser converged or made
     * no progress at this rung. The next-deeper rung is picked on the
     * next REFINE selection. */
    BasinV2* bb = pool_.by_id(basin_id);
    if (bb) {
        const auto& loc = ladder_.loc_ladder();
        if (!loc.empty()) {
            /* Find current rung index, advance by 1 if optimiser
             * converged or made no improvement at this tol. */
            std::size_t cur_idx = 0;
            for (std::size_t i = 0; i < loc.size(); ++i) {
                if (std::abs(loc[i] - bb->current_tol_cursor)
                    < 1e-15 * std::max(1.0, std::abs(loc[i]))) {
                    cur_idx = i;
                    break;
                }
            }
            const bool no_progress = (best_after >= best_before);
            if (optimizer_converged_naturally || no_progress) {
                if (cur_idx + 1u < loc.size())
                    bb->current_tol_cursor = loc[cur_idx + 1u];
                pool_.mark_exhausted(basin_id, tol);
            }
        }
    }

    {
        std::ostringstream os;
        os << "BASINSv2 basin[id=" << basin_id << "] tol="
           << std::scientific << std::setprecision(3) << tol
           << " loss=" << best_after
           << (best_after < best_before ? " (improved)" : " (no improvement)")
           << " evals_used=" << evals_used;
        HPOLog::instance().info(os.str());
    }

    if (best_after < best_before)
        iter_at_last_global_best_ = iteration_;
    if (best_after < global_best_observed_)
        global_best_observed_ = best_after;
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::do_verify_kill(HPO_Host<Vec, Runner>& host,
                                                 int basin_id)
{
    const BasinV2* b = pool_.by_id(basin_id);
    if (!b) return;
    if (b->state != LifecycleState::INACTIVE) return;

    /* Verify budget. 0 in spec means "derive from dim". */
    const std::size_t dim = static_cast<std::size_t>(outer_.low.size());
    const std::size_t cap = spec_.verify_evals_cap > 0
        ? spec_.verify_evals_cap
        : (10u * std::max<std::size_t>(dim, 1u));

    const double tol = b->last_attempted_tol.value_or(b->current_tol_cursor);
    std::vector<std::vector<double>> seeds = { b->params };

    HPO_Host<Vec, Runner>::reset_stagnation();
    /* Use cap evals as the safety net; rely on optimiser convergence
     * to terminate earlier on truly-converged basins. */
    HPO_Host<Vec, Runner>::set_stagnation_threshold(cap);

    HPO_Host<Vec, Runner>::clear_topk();
    HPO_Host<Vec, Runner>::set_track_topk(1);

    auto sub = build_stage_spec(/*global=*/false, tol, std::move(seeds));
    /* Lower the per-stage maxeval to the verify cap. */
    sub.stop.maxeval = static_cast<uint64_t>(cap);

    const double best_before = HPOLog::instance().best_loss();
    const std::size_t evals_before = HPOLog::instance().n_eval();

    try {
        run_inner<Vec, Runner>(sub, host);
    } catch (const typename HPO_Host<Vec, Runner>::StagnationStop&) {
        /* expected */
        if (HPOLog::instance().unsaved_best()) {
            /* A new best could not be saved meanwhile (it may have
             * lost to this stop in a parallel stage): fail now. */
            HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
            HPO_Host<Vec, Runner>::clear_topk();
            detail::rethrow_if_best_unsaved();
        }
    } catch (const core::ShutdownRequested&) {
        HPO_Host<Vec, Runner>::set_stagnation_threshold(0);
        throw;
    }
    HPO_Host<Vec, Runner>::set_stagnation_threshold(0);

    const std::size_t evals_used = HPOLog::instance().n_eval() - evals_before;
    const double best_after = HPOLog::instance().best_loss();
    const bool improved = (best_after < best_before * (1.0 - 1.0e-3));

    auto found = HPO_Host<Vec, Runner>::get_topk();
    if (!found.empty() && improved) {
        const auto& kv = found.front();
        std::vector<double> p(kv.second.size());
        for (long i = 0; i < kv.second.size(); ++i) p[i] = kv.second(i);
        pool_.on_refinement_complete(basin_id, std::move(p), kv.first,
                                     evals_used, iteration_, tol, false);
    }
    HPO_Host<Vec, Runner>::clear_topk();

    pool_.on_verify_attempt(basin_id, improved, iteration_);

    {
        std::ostringstream os;
        os << "BASINSv2 verify basin[id=" << basin_id << "] tickle: best "
           << std::scientific << std::setprecision(4) << best_before
           << " -> " << best_after
           << (improved ? " (improved -- revived to ACTIVE)"
                        : " (no progress)")
           << " attempt=" << (b->verify_attempts + 1)
           << "/" << spec_.verify_max_attempts;
        HPOLog::instance().info(os.str());
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::do_merge_pass()
{
    last_merge_iteration_ = iteration_;
    if (pool_.active_count() < 2) return;

    /* Phase 2: find candidate pairs via the cheap O(N^2) sweep, flag
     * each as SUSPECT_DUP. The scheduler's next REFINE selection will
     * pick the lower-loss member of each suspect pair (priority bias
     * via state_bonus = -2). After that refinement the policy
     * post-checks whether the pair is still close -- confirm or
     * revert. No LAMMPS evals are spent inside this pass itself. */
    auto candidates = pool_.find_merge_candidates(
        spec_.basin_merge_dist, spec_.basin_merge_loss_threshold);

    if (candidates.empty()) {
        std::ostringstream os;
        os << "BASINSv2 merge-pass: " << pool_.active_count()
           << " active, no candidate-pairs";
        HPOLog::instance().info(os.str());
        return;
    }

    std::ostringstream os;
    os << "BASINSv2 merge-pass: " << pool_.active_count()
       << " active, found " << candidates.size() << " candidate-pair(s)";
    HPOLog::instance().info(os.str());

    /* For each pair: if both are still ACTIVE / PROMISING / INACTIVE,
     * flag SUSPECT_DUP. If one is already MERGED/KILLED, skip. */
    for (auto [i, j] : candidates) {
        if (i >= pool_.view().size() || j >= pool_.view().size()) continue;
        const BasinV2& bi = pool_.view()[i];
        const BasinV2& bj = pool_.view()[j];

        /* If a pair is already SUSPECT_DUP from a previous merge-pass
         * AND has remained close after a refinement, confirm the merge
         * here (no extra REFINE needed -- the previous one already gave
         * them a chance). */
        if (bi.state == LifecycleState::SUSPECT_DUP &&
            bj.state == LifecycleState::SUSPECT_DUP) {
            const int winner_id = (bi.loss <= bj.loss) ? bi.id : bj.id;
            const int loser_id  = (bi.loss <= bj.loss) ? bj.id : bi.id;
            pool_.confirm_merge(loser_id, winner_id, iteration_);
            /* Revive the winner to ACTIVE for re-prioritisation. */
            pool_.set_state(winner_id, LifecycleState::ACTIVE, iteration_);
            std::ostringstream o2;
            o2 << "BASINSv2 merge basin[id=" << loser_id
               << "] -> basin[id=" << winner_id
               << "] (root) dist=" << std::fixed << std::setprecision(3)
               << pool_.relative_l2(bi.params, bj.params)
               << " dloss="
               << std::fixed << std::setprecision(2)
               << (100.0 * std::abs(bi.loss - bj.loss)
                   / std::max(std::abs(bi.loss), std::max(std::abs(bj.loss), 1e-30)))
               << "%";
            HPOLog::instance().info(o2.str());
        } else {
            pool_.flag_suspect_pair(bi.id, bj.id, iteration_);
        }
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::update_lifecycle()
{
    /* Compute EMA quartiles over ACTIVE-family basins for promotion /
     * demotion to PROMISING. Inactive transitions: any basin whose
     * evals_since_improvement exceeds inactive_after_evals goes
     * INACTIVE. KILLED transitions: INACTIVE basins past
     * verify_max_attempts. */

    const std::size_t inactive_threshold = spec_.inactive_after_evals > 0
        ? spec_.inactive_after_evals
        : static_cast<std::size_t>(0.75 * spec_.refine_budget_max_evals);

    auto& basins = const_cast<std::vector<BasinV2>&>(pool_.view());
    std::vector<double> emas;
    emas.reserve(basins.size());
    for (const auto& b : basins) {
        if (is_archive_state(b.state)) continue;
        if (b.state == LifecycleState::INACTIVE) continue;
        emas.push_back(b.improvement_ema);
    }
    double p25 = 0.0, p75 = 0.0;
    if (!emas.empty()) {
        auto sorted = emas;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t n = sorted.size();
        p25 = sorted[std::min(n - 1, n / 4u)];
        p75 = sorted[std::min(n - 1, (3u * n) / 4u)];
    }

    for (auto& b : basins) {
        if (is_archive_state(b.state)) continue;

        /* INACTIVE -> KILLED if out of verify attempts. */
        if (b.state == LifecycleState::INACTIVE &&
            b.verify_attempts >= static_cast<int>(spec_.verify_max_attempts)) {
            const auto prev = b.state;
            b.state = LifecycleState::KILLED;
            b.last_iteration = iteration_;
            std::ostringstream os;
            os << "BASINSv2 basin[id=" << b.id << "] state="
               << lifecycle_name(prev) << "->" << lifecycle_name(b.state)
               << " verify_attempts=" << b.verify_attempts;
            HPOLog::instance().info(os.str());
            continue;
        }

        /* ACTIVE -> INACTIVE if stagnant. */
        if (b.state == LifecycleState::ACTIVE &&
            b.evals_since_improvement >= inactive_threshold &&
            b.evals_consumed_total > 0) {
            const auto prev = b.state;
            b.state = LifecycleState::INACTIVE;
            b.last_iteration = iteration_;
            std::ostringstream os;
            os << "BASINSv2 basin[id=" << b.id << "] state="
               << lifecycle_name(prev) << "->" << lifecycle_name(b.state)
               << " evals_since_improve=" << b.evals_since_improvement
               << " (cap=" << inactive_threshold << ")";
            HPOLog::instance().info(os.str());
            continue;
        }

        /* ACTIVE -> PROMISING (top-quartile EMA). */
        if (b.state == LifecycleState::ACTIVE && !emas.empty() &&
            b.improvement_ema > p75 && p75 > 0.0) {
            const auto prev = b.state;
            b.state = LifecycleState::PROMISING;
            b.last_iteration = iteration_;
            std::ostringstream os;
            os << "BASINSv2 basin[id=" << b.id << "] state="
               << lifecycle_name(prev) << "->" << lifecycle_name(b.state)
               << " ema=" << std::scientific << std::setprecision(3) << b.improvement_ema
               << " (p75=" << p75 << ")";
            HPOLog::instance().info(os.str());
            continue;
        }

        /* PROMISING -> ACTIVE (bottom-quartile EMA). */
        if (b.state == LifecycleState::PROMISING && !emas.empty() &&
            b.improvement_ema < p25) {
            const auto prev = b.state;
            b.state = LifecycleState::ACTIVE;
            b.last_iteration = iteration_;
            std::ostringstream os;
            os << "BASINSv2 basin[id=" << b.id << "] state="
               << lifecycle_name(prev) << "->" << lifecycle_name(b.state)
               << " ema=" << std::scientific << std::setprecision(3) << b.improvement_ema
               << " (p25=" << p25 << ")";
            HPOLog::instance().info(os.str());
            continue;
        }
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::maybe_emit_summary()
{
    if (spec_.summary_cadence == 0) return;
    if (iteration_ - last_summary_iteration_ <
        static_cast<int>(spec_.summary_cadence)) return;
    last_summary_iteration_ = iteration_;

    std::size_t n_active = 0, n_promising = 0, n_inactive = 0,
                n_suspect = 0, n_merged = 0, n_killed = 0;
    for (const auto& b : pool_.view()) {
        switch (b.state) {
            case LifecycleState::ACTIVE:      ++n_active;    break;
            case LifecycleState::PROMISING:   ++n_promising; break;
            case LifecycleState::INACTIVE:    ++n_inactive;  break;
            case LifecycleState::SUSPECT_DUP: ++n_suspect;   break;
            case LifecycleState::MERGED:      ++n_merged;    break;
            case LifecycleState::KILLED:      ++n_killed;    break;
        }
    }
    const double best = HPOLog::instance().best_loss();

    std::ostringstream os;
    os << "BASINSv2 round-summary iter=" << iteration_
       << " active=" << n_active
       << " promising=" << n_promising
       << " inactive=" << n_inactive
       << " suspect=" << n_suspect
       << " merged=" << n_merged
       << " killed=" << n_killed
       << " best=" << std::scientific << std::setprecision(4) << best;
    HPOLog::instance().info(os.str());
    best_at_last_summary_ = best;

    /* Verbose: full priority-sorted basin table, every 50 iterations.
     * Useful for debugging starvation / weight-sensitivity. */
    if (spec_.verbose_diagnostics && iteration_ % 50 == 0) {
        const auto min_dist = compute_min_dist_to_other_active();
        PriorityContext ctx;
        ctx.current_iteration = iteration_;
        ctx.target_tol = ladder_.loc_ladder().empty()
            ? 1.0e-9 : ladder_.loc_ladder().back();
        const auto scores = compute_priorities(
            pool_.view(), min_dist, ctx, weights_);
        HPOLog::instance().info("BASINSv2 PRIORITY-DUMP begin (iter=" +
                                std::to_string(iteration_) + ")");
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < pool_.view().size(); ++i)
            if (!is_archive_state(pool_.view()[i].state)) idx.push_back(i);
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b){ return scores[a] > scores[b]; });
        for (auto i : idx) {
            const auto& b = pool_.view()[i];
            std::ostringstream l;
            l << "  basin[id=" << b.id
              << "] state=" << lifecycle_name(b.state)
              << " loss=" << std::scientific << std::setprecision(4) << b.loss
              << " ema=" << b.improvement_ema
              << " tol=" << b.current_tol_cursor
              << " neglect=" << (iteration_ - b.last_iteration)
              << " priority=" << scores[i];
            HPOLog::instance().info(l.str());
        }
        HPOLog::instance().info("BASINSv2 PRIORITY-DUMP end");
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::commit_checkpoint()
{
    try {
        core::CheckpointManager::instance().save_now();
    } catch (const std::exception& e) {
        HPOLog::instance().warning(
            std::string("BASINSv2 checkpoint save failed: ") + e.what());
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::maybe_commit_checkpoint()
{
    /* CheckpointManager handles cadence; we just call save_now() at
     * "natural" boundaries (end of iteration). */
    commit_checkpoint();
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::initialize_or_resume(
        HPO_Host<Vec, Runner>& host)
{
    (void)host;
    if (resumed_) {
        std::ostringstream os;
        os << "BASINSv2 resumed: iter=" << iteration_
           << " pool=" << pool_.size() << " (active=" << pool_.active_count()
           << " archive=" << pool_.archive_count() << ")";
        HPOLog::instance().info(os.str());
    } else {
        iteration_ = 0;
        last_explore_iteration_ = -static_cast<int>(spec_.explore_kick_cadence);
        last_merge_iteration_   = -static_cast<int>(spec_.merge_cadence);
        last_summary_iteration_ = -1;
        iter_at_last_global_best_ = 0;
        global_best_observed_   = HPOLog::instance().best_loss();
    }
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::run(const OptimizerSpec& /*spec*/,
                                      HPO_Host<Vec, Runner>& host)
{
    using clock = std::chrono::steady_clock;
    const double t_start_secs =
        std::chrono::duration<double>(clock::now().time_since_epoch()).count();

    initialize_or_resume(host);

    while (true) {
        if (overall_termination_requested(t_start_secs)) break;

        WorkItem w;
        try {
            w = select_work();
        } catch (const std::exception& e) {
            HPOLog::instance().warning(
                std::string("BASINSv2 select_work failed: ") + e.what()
                + " -- falling back to EXPLORE");
            w.kind = WorkKind::EXPLORE;
        }

        try {
            switch (w.kind) {
                case WorkKind::EXPLORE:     do_explore   (host); break;
                case WorkKind::REFINE:      do_refine    (host, w.basin_id, w.tol); break;
                case WorkKind::VERIFY_KILL: do_verify_kill(host, w.basin_id); break;
                case WorkKind::MERGE_PASS:  do_merge_pass(); break;
            }
        } catch (const core::ShutdownRequested&) {
            commit_checkpoint();
            throw;
        }

        update_lifecycle();
        maybe_emit_summary();
        ++iteration_;
        maybe_commit_checkpoint();
    }

    /* Accumulate wall time for cross-resume budget accounting. */
    const double elapsed_now =
        std::chrono::duration<double>(clock::now().time_since_epoch()).count()
        - t_start_secs;
    wall_seconds_accumulated_ += elapsed_now;

    {
        std::ostringstream os;
        os << "BASINSv2 exiting: iter=" << iteration_
           << " active=" << pool_.active_count()
           << " archive=" << pool_.archive_count()
           << " best=" << std::scientific << std::setprecision(4)
           << HPOLog::instance().best_loss();
        HPOLog::instance().info(os.str());
    }
}

/* -------------------- checkpoint ------------------------------------ */

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::save(core::Checkpoint& cp) const
{
    cp.section(tag());
    cp.put_int  ("iteration",                   iteration_);
    cp.put_int  ("last_explore_iteration",      last_explore_iteration_);
    cp.put_int  ("last_merge_iteration",        last_merge_iteration_);
    cp.put_int  ("last_summary_iteration",      last_summary_iteration_);
    cp.put_int  ("iter_at_last_global_best",    iter_at_last_global_best_);
    cp.put_dbl  ("best_at_last_summary",        best_at_last_summary_);
    cp.put_dbl  ("global_best_observed",        global_best_observed_);
    cp.put_dbl  ("wall_seconds_accumulated",    wall_seconds_accumulated_);
    /* Priority weights -- for reproducibility audits. */
    std::vector<double> wvec = {
        weights_.improve, weights_.neglect, weights_.distance,
        weights_.tol_gap, weights_.state, weights_.age
    };
    cp.put_vec  ("priority_weights", wvec);
}

template<class Vec, class Runner>
void BasinsV2Policy<Vec, Runner>::load(const core::Checkpoint& cp)
{
    /* If a v1 BASINS_POLICY section is the only thing present, the
     * parent strategy mismatch should have already failed at config
     * parse time (different StrategyType). We just refuse to silently
     * load v1 fields here. */
    if (auto v = cp.get_int(tag(), "iteration"))
        iteration_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "last_explore_iteration"))
        last_explore_iteration_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "last_merge_iteration"))
        last_merge_iteration_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "last_summary_iteration"))
        last_summary_iteration_ = static_cast<int>(*v);
    if (auto v = cp.get_int(tag(), "iter_at_last_global_best"))
        iter_at_last_global_best_ = static_cast<int>(*v);
    if (auto v = cp.get_dbl(tag(), "best_at_last_summary"))
        best_at_last_summary_ = *v;
    if (auto v = cp.get_dbl(tag(), "global_best_observed"))
        global_best_observed_ = *v;
    if (auto v = cp.get_dbl(tag(), "wall_seconds_accumulated"))
        wall_seconds_accumulated_ = *v;
    if (auto v = cp.get_vec(tag(), "priority_weights")) {
        const auto& w = *v;
        if (w.size() == 6) {
            weights_.improve  = w[0]; weights_.neglect  = w[1];
            weights_.distance = w[2]; weights_.tol_gap  = w[3];
            weights_.state    = w[4]; weights_.age      = w[5];
        }
    }
    resumed_ = true;
}

}   // namespace hpo
}   // namespace tadah

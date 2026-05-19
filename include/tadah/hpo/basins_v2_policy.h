#ifndef TADAH_HPO_BASINS_V2_POLICY_H
#define TADAH_HPO_BASINS_V2_POLICY_H

/*
 *  BasinsV2Policy -- priority-driven, lifecycle-aware "run and forget"
 *  HPO strategy, additive to v1's BasinsPolicy.
 *
 *  In one paragraph:
 *  Every scheduler iteration picks ONE work item -- REFINE the highest-
 *  priority basin, EXPLORE for diversity, VERIFY-KILL an inactive basin,
 *  or run a deep MERGE pass. Each basin has its own lifecycle state and
 *  its own ftol cursor; basins are not forced through the ladder in
 *  lockstep. Adaptive REFINE termination uses the existing
 *  `set_stagnation_threshold` mechanism with a tol- and dim-derived
 *  threshold (`max(refine_budget_min_evals, plateau_eps_factor *
 *  base_threshold)`), avoiding magic per-stage budgets.
 *
 *  Phases delivered incrementally: Phase 1 ships scheduler + lifecycle
 *  + adaptive REFINE + checkpoint + verify-before-kill (this header).
 *  Phase 2 fills in WORK_MERGE_PASS. Phase 3 swaps EXPLORE for the
 *  Sobol+filter pipeline. Phase 4 turns on path recording + smarter
 *  archive eviction. Phase 5 polishes diagnostics + docs.
 *
 *  Like v1, BASINSv2 sits at the top of the policy stack -- the RESTART
 *  decorator is not applied (lifecycle subsumes RESTART semantics).
 */

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/basins_v2_priority.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/tolerance_ladder.h>

#include <chrono>
#include <memory>

namespace tadah {
namespace hpo {

template<class Vec, class Runner>
class BasinsV2Policy : public RunPolicy<Vec, Runner>,
                      public core::ICheckpointable
{
public:
    BasinsV2Policy(const OptimizerSpec& outer, const HPOTargets* targets);
    ~BasinsV2Policy() override;

    /* RunPolicy contract */
    void run(const OptimizerSpec& spec, HPO_Host<Vec, Runner>& host) override;

    /* ICheckpointable contract */
    std::string tag() const override { return "BASINS_POLICY_V2"; }
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;

    /* Test hooks (read-only). */
    int  current_iteration() const { return iteration_; }
    bool resumed()           const { return resumed_;   }
    const BasinPoolV2&  pool()   const { return pool_;   }
    const ToleranceLadder& ladder() const { return ladder_; }

    /* Exposed for test_basins_v2_*.cpp -- same role as v1's
     * build_stage_spec_for_test. */
    OptimizerSpec build_stage_spec_for_test(bool global, double tol) const {
        return build_stage_spec(global, tol, {});
    }

    /* Plumbed from hpo_run() after NOISE_CALIBRATION. Same shape as v1. */
    void set_noise_floor(double f);

private:
    enum class WorkKind { EXPLORE, REFINE, VERIFY_KILL, MERGE_PASS };
    struct WorkItem {
        WorkKind kind = WorkKind::EXPLORE;
        int      basin_id = -1;       // for REFINE / VERIFY_KILL
        double   tol      = 0.0;      // for REFINE
    };

    /* ---- top of run() loop ---------------------------------------- */
    void initialize_or_resume(HPO_Host<Vec, Runner>& host);
    bool overall_termination_requested(double t_start_secs) const;

    /* ---- scheduler ------------------------------------------------- */
    WorkItem select_work();

    /* ---- dispatchers (one per WorkKind) --------------------------- */
    void do_explore   (HPO_Host<Vec, Runner>& host);
    void do_refine    (HPO_Host<Vec, Runner>& host, int basin_id, double tol);
    void do_verify_kill(HPO_Host<Vec, Runner>& host, int basin_id);
    void do_merge_pass();   /* Phase 2 -- Phase 1 stub */

    /* ---- per-iteration bookkeeping -------------------------------- */
    void update_lifecycle();          // ACTIVE<->PROMISING / ACTIVE->INACTIVE
    void maybe_emit_summary();
    void maybe_commit_checkpoint();

    /* ---- helpers --------------------------------------------------- */
    OptimizerSpec build_stage_spec(bool global, double tol,
                                   std::vector<std::vector<double>> seeds) const;
    void   ensure_ladder();
    void   commit_checkpoint();
    double pick_initial_tol_for_new_basin() const;
    std::size_t adaptive_refine_threshold(double tol) const;
    std::vector<double> compute_min_dist_to_other_active() const;

    /* ---- configuration & state ------------------------------------ */
    OptimizerSpec       outer_;
    const HPOTargets*   targets_ = nullptr;
    BasinsV2Spec        spec_;
    ToleranceLadder     ladder_;
    BasinPoolV2         pool_;
    PriorityWeights     weights_;

    /* Scheduler state. */
    int                 iteration_                  = 0;
    int                 last_explore_iteration_     = -1;
    int                 last_merge_iteration_       = -1;
    int                 last_summary_iteration_     = -1;
    int                 iter_at_last_global_best_   = 0;
    double              best_at_last_summary_       = std::numeric_limits<double>::infinity();
    double              global_best_observed_       = std::numeric_limits<double>::infinity();
    bool                resumed_                    = false;

    /* Wall-time accumulation across resume boundaries. */
    double              wall_seconds_accumulated_   = 0.0;
};

}   // namespace hpo
}   // namespace tadah

#include "basins_v2_policy_impl.hpp"

#endif

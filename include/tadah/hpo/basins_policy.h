#ifndef TADAH_HPO_BASINS_POLICY_H
#define TADAH_HPO_BASINS_POLICY_H

/*
 *  BasinsPolicy — sophisticated "run and forget" HPO strategy.
 *
 *  See /home/marcin/.claude/plans/desing-sophisticated-run-and-rippling-steele.md
 *  for the full design rationale (anytime / basin-hopping / progressive
 *  tolerance ladder).
 *
 *  In one paragraph:
 *  Round k runs a GLOBAL stage at glo_ladder[k] to harvest candidate
 *  basins, then a cumulative LOCAL stage at loc_ladder[1..k] that
 *  refines every basin in the pool that hasn't already been exhausted
 *  at that tolerance. After each stage the policy commits a checkpoint
 *  so any interrupt (ctrl+C, SLURM SIGTERM/SIGUSR2) yields a usable
 *  state. Restart re-enters at the saved (round, stage_index) and runs
 *  from there — optimiser internal state is *not* required, the basin
 *  pool's per-basin best params are enough to "move the needle in the
 *  right direction".
 *
 *  Parameter `Vec` is a column vector type (dlib::matrix<double,0,1>
 *  in production); `Runner` is the LAMMPS runner template parameter
 *  passed through to HPO_Host.
 */

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/basin_pool.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/progress_tracker.h>
/* Use the small base header to avoid pulling run_policy.h's
 * BasinsPolicy back-reference (which would form a cycle).  */
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/tolerance_ladder.h>

#include <chrono>
#include <memory>

namespace tadah {
namespace hpo {

template<class Vec, class Runner>
class BasinsPolicy : public RunPolicy<Vec, Runner>,
                    public core::ICheckpointable
{
public:
    BasinsPolicy(const OptimizerSpec& outer, const HPOTargets* targets);
    ~BasinsPolicy() override;

    /* RunPolicy contract */
    void run(const OptimizerSpec& spec, HPO_Host<Vec, Runner>& host) override;

    /* ICheckpointable contract */
    std::string tag() const override { return "BASINS_POLICY"; }
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;

    /* Test hooks (read-only). */
    int  current_round() const { return round_; }
    bool resumed()       const { return resumed_; }

    /* Test hook: expose the OptimizerSpec that would be handed to the
     * inner driver for a stage. Used by test_basins_mlsl_inner_local.cpp
     * to verify that the LOCAL/ENDLOCAL sub-block is preserved through
     * build_stage_spec — required for NLopt MLSL/AUGLAG-family globals
     * which delegate the inner search via set_local_optimizer. */
    OptimizerSpec build_stage_spec_for_test(bool global, double tol) const {
        return build_stage_spec(global, tol, {});
    }

    /* Plumbed from hpo_run() after NOISE_CALIBRATION (or after resume,
     * if the persisted suggested_ftol is set). Stores the value into
     * spec_.noise_floor and re-runs ensure_ladder() so the clamp (if
     * spec_.clamp_ladder_to_noise_floor) takes effect. Idempotent —
     * ensure_ladder rebuilds the ladder from spec_.tol_*_ladder
     * (originals), then trim_below(noise_floor). */
    void set_noise_floor(double f);

private:
    enum class Stage { GLOBAL, LOCAL };

    /* The two stages — each builds a transient OptimizerSpec and
     * dispatches via run_inner<>(). */
    void run_global_stage(HPO_Host<Vec, Runner>& host);
    void run_local_stage (HPO_Host<Vec, Runner>& host, int loc_step);

    /* Construct a per-stage OptimizerSpec from the outer one. */
    OptimizerSpec build_stage_spec(bool global,
                                   double tol,
                                   std::vector<std::vector<double>> seeds) const;

    /* Helpers. */
    void ensure_ladder();
    void commit_checkpoint();
    bool overall_termination_requested(double t_start_secs) const;

    /* Configuration & state ------------------------------------------ */
    OptimizerSpec       outer_;             // user-supplied (copied)
    const HPOTargets*   targets_ = nullptr; // borrowed
    BasinsSpec          spec_;              // outer_.basins (cached for clarity)
    ToleranceLadder     ladder_;
    BasinPool           pool_;
    ProgressTracker     tracker_;

    int                 round_              = 1;
    Stage               stage_              = Stage::GLOBAL;
    int                 current_loc_step_   = 0;     // 0 = not started
    int                 rounds_no_improve_  = 0;
    double              best_at_round_start_ = std::numeric_limits<double>::infinity();
    bool                resumed_            = false;
};

}   // namespace hpo
}   // namespace tadah

#include "basins_policy_impl.hpp"

#endif

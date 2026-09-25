#ifndef TADAH_HPO_STARTUP_VALIDATION_H
#define TADAH_HPO_STARTUP_VALIDATION_H

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>

#include <cstddef>

namespace tadah {
namespace hpo {

/** Derive the total model weight count K from the training configuration
 *  alone (no data): replicate the Trainer's descriptor init on a copy of
 *  the context and query the basis-function column count. Returns 0 when
 *  K cannot be derived (a WARNING is logged) — callers must then skip the
 *  rules that need K. */
std::size_t derive_weight_count(const core::Context &train_ctx);

/** Fail-fast validation of the HPO setup, run once after HPOTargets
 *  parsing and before HPO_Host construction. Throws std::runtime_error
 *  with an actionable message on the first violated rule:
 *
 *   - OPTIM WEIGHTS is rejected (silent no-op: LS overwrites weights);
 *   - OPTIM FIXWEIGHT requires FIXINDEX in the training configuration;
 *   - NLOPT, DLIB and CERES need at least one OPTIM line;
 *   - the STRATEGY TYPE must suit the optimiser (validate_run_policy,
 *     strategy_rules.h): MULTISTART rejects a global optimiser that
 *     consumes a single starting point, for any N_STARTS; PRESTAGE needs
 *     an EXPLORE block and, with a configured TOPK > 1, a refinement
 *     optimiser MULTISTART accepts;
 *   - each phase's backend accepts its settings (validate_phase_backends,
 *     backend_rules.h): the backend's own attribute and PARAM checks on
 *     the spec the phase runs with, and MAXEVAL for a backend that has
 *     no stopping rule of its own (TADAH:RANDOM, DLIB:GFS);
 *   - FIXINDEX/FIXWEIGHT pairing: lone key, size mismatch, out-of-bounds
 *     index (via mlip::classify_weight_source);
 *   - no training DBFILE: FIXINDEX must cover every model weight;
 *   - at least one loss source must be declared (validation metrics,
 *     PC_OBSERVABLE, or PC_LAMMPS);
 *   - validation metrics (PC_ERMSE, ...) require a --validation file;
 *   - NORM true is incompatible with FIXINDEX/FIXWEIGHT (fixed values
 *     would be interpreted in the normalized feature space);
 *   - GRAD_STRATEGY ANALYTICAL (written out, or resolved for
 *     PC_OBSERVABLE) in a block that runs an optimiser and obtains
 *     gradients (the STRATEGY-level block, EXPLORE; an INNER search uses
 *     its enclosing block's gradients) needs a setup the analytical
 *     gradient covers and loss terms with an analytical derivative
 *     (analytical_gradient_blocker, analytical_loss_term_blocker,
 *     analytical_gate.h);
 *   - FAILSCORE_GRADING ON needs a finite FAILSCORE that leaves room for
 *     the four grading bands.
 */
void validate_hpo_setup(core::Context &train_ctx, const HPOTargets &trg,
                        bool has_train_db, bool has_validation);

} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_STARTUP_VALIDATION_H

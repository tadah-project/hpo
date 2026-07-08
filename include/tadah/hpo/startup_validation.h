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
 *   - FIXINDEX/FIXWEIGHT pairing: lone key, size mismatch, out-of-bounds
 *     index (via mlip::classify_weight_source);
 *   - no training DBFILE: FIXINDEX must cover every model weight;
 *   - at least one loss source must be declared (validation metrics,
 *     PC_OBSERVABLE, or PC_LAMMPS);
 *   - validation metrics (PC_ERMSE, ...) require a --validation file;
 *   - NORM true is incompatible with FIXINDEX/FIXWEIGHT (fixed values
 *     would be interpreted in the normalized feature space).
 */
void validate_hpo_setup(core::Context &train_ctx, const HPOTargets &trg,
                        bool has_train_db, bool has_validation);

} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_STARTUP_VALIDATION_H

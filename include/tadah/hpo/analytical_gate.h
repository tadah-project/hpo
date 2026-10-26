#pragma once
/*
 * analytical_gate -- whether a setup can form the analytical (IFT) gradient
 * of the HPO loss. The single source of truth for:
 *   - validate_hpo_setup(), which refuses GRAD_STRATEGY ANALYTICAL -- written
 *     out, or resolved from an unset GRAD_STRATEGY for PC_OBSERVABLE -- on a
 *     setup that cannot form it (it used to fail at the first gradient, with
 *     "no registered analytical derivative"): the fit settings
 *     (analytical_gradient_blocker) and the loss terms
 *     (analytical_loss_term_blocker);
 *   - the run-time gate of HPO_Host::analytical_gradient() and
 *     compute_analytical_gradient_from_trainer(), which hand every parameter
 *     to finite differences on a setup whose fit settings the gradient does
 *     not cover (what HYBRID then uses). A loss term without a derivative
 *     does that per evaluation (compute_observables_grad_dWchain).
 */

#include <tadah/core/context.h>

#include <string>

namespace tadah::hpo {

class HPOTargets;

/* Why the analytical gradient of the loss cannot be formed with the
 * training configuration `ctx` and the targets `trg`, or "" when it can.
 * The gradient differentiates the least-squares fit through its Cholesky
 * factor, so it needs training data (`has_train_db`) with every weight
 * fitted (no FIXINDEX), the normal equations (OALGO 4) with a fixed ridge
 * LAMBDA >= 0, raw features (NORM false), a linear basis (MODEL ...
 * BF_Linear or Kern_Linear) of a model that fits the training rows (M_BLR,
 * M_KRR implementation 1; M_KRR implementation 2 solves over its basis
 * vectors and reads no OALGO) and no stress rows (STRESS false, no
 * PC_SRMSE_GPA). PC_FRMSE is covered whatever FORCE is: the validation
 * descriptors carry their force parts whenever it is set
 * (fit_switch_guard.h). Settings only: a parameter without a
 * registered derivative still has none (ANALYTICAL stops there, HYBRID
 * uses finite differences for it). */
std::string analytical_gradient_blocker(const core::Context& ctx,
                                        const HPOTargets& trg,
                                        bool has_train_db);

/* Why a loss term of `trg` can have no analytical gradient on a setup
 * analytical_gradient_blocker() accepts, naming the term (cohesive_energy[k]
 * for the k-th PC_OBSERVABLE line) and the option, or "" when every term
 * has one: a PC_OBSERVABLE --loss without a derivative with respect to the
 * value (MAPE, RMSLE: at every evaluation), or an observable whose settings
 * report values without one (Observable::analytical_gradient_gap, e.g.
 * --unconverged force-scaled: at every relaxation that does not converge).
 * ANALYTICAL stops the run at such a gradient, so validate_hpo_setup refuses
 * it; HYBRID uses finite differences there and keeps the analytical gradient
 * elsewhere, so the run-time gate does not use this. */
std::string analytical_loss_term_blocker(const HPOTargets& trg);

}  // namespace tadah::hpo

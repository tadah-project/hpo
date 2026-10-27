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
 *     does that per evaluation (compute_observables_grad_dWchain);
 *   - the search parameters the descriptors do not differentiate on the
 *     data of the run (parameters_without_analytical_derivative), which
 *     require_analytical_parameters() (startup_validation.h) refuses under
 *     GRAD_STRATEGY ANALYTICAL once the data are read, by the rule
 *     compute_analytical_gradient_from_trainer() routes a parameter by
 *     (claims_differentiate, dloss_dtheta.h).
 */

#include <tadah/core/context.h>

#include <cstddef>
#include <string>
#include <vector>

namespace tadah::mlip { class StructureDB; }

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
 * (fit_switch_guard.h). Settings only: a search parameter without a
 * derivative is parameters_without_analytical_derivative()'s (below). */
std::string analytical_gradient_blocker(const core::Context& ctx,
                                        const HPOTargets& trg,
                                        bool has_train_db);

/* Why a loss term of `trg` can have no analytical gradient on a setup
 * analytical_gradient_blocker() accepts, naming the term (cohesive_energy[k]
 * for the k-th PC_OBSERVABLE line) and the option, or "" when every term
 * has one: a PC_OBSERVABLE loss function (its --loss, or the LOSS line
 * above it) without a derivative with respect to the value
 * (loss_has_derivative_wrt_observed; every loss function of LossFunction
 * has one), or an observable whose settings
 * report values without one (Observable::analytical_gradient_gap, e.g.
 * --unconverged force-scaled: at every relaxation that does not converge).
 * ANALYTICAL stops the run at such a gradient, so validate_hpo_setup refuses
 * it; HYBRID uses finite differences there and keeps the analytical gradient
 * elsewhere, so the run-time gate does not use this. */
std::string analytical_loss_term_blocker(const HPOTargets& trg);

/* A search parameter of an OPTIM line that the analytical gradient cannot
 * differentiate on the data of the run. */
struct ParameterWithoutDerivative {
    std::string key;            // the OPTIM key, "LAMBDA", "SGRIDMB"
    std::size_t index = 0;      // the 1-based index of its value
    std::size_t line = 0;       // its OPTIM line of the target file (0: none)
    /* The descriptors differentiate it in the energy rows, not in the
     * force rows the run has: those of the fit (FORCE true) when
     * `force_rows_of_fit`, else those the validation measures (PC_FRMSE). */
    bool energy_rows_only = false;
    bool force_rows_of_fit = false;
};

/* The search parameters of `trg` (every value an OPTIM line searches) that
 * the descriptors of the training configuration `ctx` do not differentiate
 * on the training structures `train` and the validation structures `val`
 * (each with its neighbour list, as HPO_Host holds them), in the order of
 * their lines: the parameters compute_analytical_gradient_from_trainer
 * hands to finite differences at the point of the configuration, by the
 * same walker and rule (theta_claims, claims_differentiate,
 * dloss_dtheta.h), with the descriptors an evaluation builds (an
 * mlip::Trainer of `ctx`). A key no descriptor has a derivative with
 * respect to (LAMBDA, a key of the fit) gets no claim; a descriptor whose
 * derivatives cover the energy rows only (DM_Blip, DM_EAD) gets the
 * energy claim, which is not enough with FORCE true or PC_FRMSE. For a
 * setup analytical_gradient_blocker() accepts (training data, no
 * FIXINDEX). */
std::vector<ParameterWithoutDerivative>
parameters_without_analytical_derivative(const core::Context& ctx,
                                         const HPOTargets& trg,
                                         const mlip::StructureDB& train,
                                         const mlip::StructureDB& val);

}  // namespace tadah::hpo

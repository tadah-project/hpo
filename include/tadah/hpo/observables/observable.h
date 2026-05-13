#ifndef TADAH_HPO_OBSERVABLES_OBSERVABLE_H
#define TADAH_HPO_OBSERVABLES_OBSERVABLE_H

#include <tadah/core/core_types.h>
#include <tadah/core/context.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/mlip/descriptors_calc_base.h>
#include <tadah/mlip/models/m_tadah_base.h>

#include <optional>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

/** \brief Result returned by every Observable::evaluate call.
 *
 *  Layout supports both the trivial closed-form path (`needs_adjoint=false`,
 *  cohesive energy / surface energy / phase-transition energy at the
 *  energy minimum) and the IFT-adjoint path (`needs_adjoint=true`,
 *  lattice constant / elastic constants / γ-surface where the observable
 *  depends explicitly on atomic positions or strain).
 *
 *  For the closed-form path the host uses dO_dW directly:
 *      dL/dtheta = (dL/dO) * (dW_star/dtheta)^T * dO_dW.
 *  For the adjoint path the host first solves H*lambda = dg/dx, calls
 *  Model::vjp_force(lambda) to fold dF/dW in, and adds the box-strain
 *  piece via dg_de when box_dofs > 0. Cohesive energy uses the former.
 */
struct ObservableEvalResult {
    /** Scalar value of the observable (e.g. cohesive energy in eV/atom). */
    double value = 0.0;

    /** ∂O/∂W (or the full direct contribution after adjoint folding).
     *  Size = number of weights in the model. */
    tadah::core::t_type dO_dW;

    /** True iff the Observable requires the host to run the adjoint
     *  solve (Hessian-vector). When false, dg_dx / dg_de are ignored. */
    bool needs_adjoint = false;

    /** ∂g/∂x at the relaxed state, one Vec3d per atom.
     *  Populated only when `needs_adjoint=true`. The host passes this as
     *  the rhs to DenseAdjointSolver. */
    std::vector<tadah::core::Vec3d> dg_dx;

    /** Box DOF count (0 / 1 / 6). Mirrors NativeRelaxer's box_mode of
     *  the simulation that produced this observable. */
    int box_dofs = 0;

    /** ∂g/∂ε at the relaxed strain (size = box_dofs).
     *  Populated only when `box_dofs > 0` and `needs_adjoint=true`. */
    tadah::core::t_type dg_de;

    /** Updated hot-var value to write back to the WarmStart cache.
     *  Set only for observables that maintain warm-start state. */
    std::optional<double> hotvar_updated_value;
};

/** \brief Abstract base for typed PC_OBSERVABLE entries.
 *
 *  Each concrete subclass encodes ONE physical-property computation:
 *  cell construction, relaxation strategy, and the closed-form
 *  mapping from {relaxed-state energies / forces / virials} to a scalar
 *  observable value with its W-derivative.
 *
 *  Concrete subclasses landing after the cohesive-energy MVP:
 *      - LatticeParamObservable        (∂g/∂x ≠ 0; first adjoint user)
 *      - VacancyFormationObservable    (multi-cell relax, ∂g/∂x = 0)
 *      - ElasticConstantObservable     (Voigt-6 strain + adjoint)
 *      - SurfaceEnergyObservable
 *      - GammaSurfaceObservable        (constrained relax)
 *      - PhaseTransitionEnergyObservable
 *      - GenericRelaxEnergyObservable  (file-loaded reference Structure)
 *
 *  The Observable does NOT own the Model/DC; the host passes them in by
 *  reference at evaluate() time. This keeps the Observable cheap to
 *  construct and to copy between HPO evaluations.
 */
class Observable {
public:
    virtual ~Observable() = default;

    /** Short kind tag, e.g. "cohesive_energy". Used in logs and the
     *  WarmStart cache key. */
    virtual std::string kind() const = 0;

    /** Evaluate observable value + W-derivative + (optional) adjoint
     *  rhs at the current model weights. */
    virtual ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx) = 0;

    /** Target value the optimiser drives toward. */
    virtual double target() const = 0;

    /** Loss weight on this observable's residual. */
    virtual double weight() const = 0;

    /** Loss function (L1/L2/HUBER/...). */
    virtual const tadah::LossFunction& lossfn() const = 0;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OBSERVABLES_OBSERVABLE_H

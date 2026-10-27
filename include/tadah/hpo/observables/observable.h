#ifndef TADAH_HPO_OBSERVABLES_OBSERVABLE_H
#define TADAH_HPO_OBSERVABLES_OBSERVABLE_H

#include <tadah/core/core_types.h>
#include <tadah/core/context.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/mlip/descriptors_calc_base.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/structure.h>

#include <cstddef>
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
 *  lattice constant / elastic constants / gamma-surface where the observable
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

    /** dO/dW (or the full direct contribution after adjoint folding).
     *  Size = number of weights in the model. */
    tadah::core::t_type dO_dW;

    /** True iff the Observable requires the host to run the adjoint
     *  solve (Hessian-vector). When false, dg_dx / dg_de are ignored. */
    bool needs_adjoint = false;

    /** dg/dx at the relaxed state, one Vec3d per atom.
     *  Populated only when `needs_adjoint=true`. The host passes this as
     *  the rhs to DenseAdjointSolver. */
    std::vector<tadah::core::Vec3d> dg_dx;

    /** Box DOF count (0 / 1 / 6). Mirrors NativeRelaxer's box_mode of
     *  the simulation that produced this observable. */
    int box_dofs = 0;

    /** dg/deps at the relaxed strain (size = box_dofs).
     *  Populated only when `box_dofs > 0` and `needs_adjoint=true`. */
    tadah::core::t_type dg_de;

    /** The value the next evaluation of the chain starts from (the warm
     *  start, Observable::warm_start_name): the host keeps it per chain in
     *  the warm-start cache (warmstart_cache.h). Set only by observables
     *  with a warm start. */
    std::optional<double> hotvar_updated_value;

    /** One model energy the value is linear in (closed-form path). */
    struct EnergyTerm {
        /** The structure at the geometry the value was taken at (the
         *  relaxed crystal, the isolated atom), neighbour list built. */
        tadah::mlip::Structure structure;
        double                 coeff = 0.0;
    };

    /** The value as value = sum_k coeff_k * E(structure_k) + const, the
     *  energies E predicted by the model. dO_dW is sum_k coeff_k *
     *  dE_k/dW; the host also differentiates each E_k with respect to
     *  the descriptor parameters at fixed weights -- the part of
     *  dO/dtheta the weights do not carry. A relaxed geometry's own
     *  dependence on theta adds nothing at a stationary point. Empty when
     *  the value is not such a combination (a constant, or a penalty such
     *  as the force-scaled unconverged value): the host then has no
     *  analytical dO/dtheta. */
    std::vector<EnergyTerm> energy_terms;

    /** The value is a constant (the penalty of an unconverged
     *  relaxation): it depends on neither the weights nor theta, and its
     *  gradient is zero. */
    bool value_is_constant = false;

    /** Why the value is not a finite number, when it is not (a relaxed
     *  state the observable rejects, an unconverged relaxation's
     *  --unconverged value that is not finite); empty otherwise. The host
     *  journals it with the failed term (HPO_Host::obs_loss). */
    std::string failure;
};

/** \brief Abstract base for typed PC_OBSERVABLE entries.
 *
 *  Each concrete subclass encodes ONE physical-property computation:
 *  cell construction, relaxation strategy, and the closed-form
 *  mapping from {relaxed-state energies / forces / virials} to a scalar
 *  observable value with its W-derivative.
 *
 *  Concrete subclasses landing after the cohesive-energy MVP:
 *      - LatticeParamObservable        (dg/dx != 0; first adjoint user)
 *      - VacancyFormationObservable    (multi-cell relax, dg/dx = 0)
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

    /** Short kind tag, e.g. "cohesive_energy". Used in the journal and
     *  as the WarmStart cache key (keep it stable). */
    virtual std::string kind() const = 0;

    /** Column tag in loss.tadah / outvar.tadah, e.g. "Ecoh": short
     *  enough that L(<tag>)[k] fits the narrowest column (11 characters)
     *  for every k up to 99. */
    virtual std::string column_tag() const = 0;

    /** The unit of the value ("eV/atom"), which the best_pot.tadah header
     *  writes with the term; "" for a value without one. */
    virtual std::string unit() const { return {}; }

    /** Why a value this observable reports under its settings can have no
     *  analytical derivative (ObservableEvalResult::dO_dW empty, and not a
     *  constant), naming the option; "" when every value has one or is a
     *  constant. The startup rule of GRAD_STRATEGY ANALYTICAL refuses such
     *  an observable (analytical_loss_term_blocker, analytical_gate.h):
     *  ANALYTICAL would stop the run at the first such value, where HYBRID
     *  uses finite differences. */
    virtual std::string analytical_gradient_gap() const = 0;

    /** What evaluate() carries from one evaluation to the next, so that two
     *  evaluations of the same model can give different values (a
     *  relaxation started where the previous one ended), or "" when it
     *  carries nothing. A search then evaluates a point it asks for again
     *  instead of reusing the loss it gave (repeat_points.h). An observable
     *  that does not say carries something. */
    virtual std::string carried_state() const
    {
        return "it does not say that it keeps no state between evaluations";
    }

    /** The elements the observable's structures are made of (its
     *  --species), which the potential must have: the startup checks
     *  refuse one it has not (validate_hpo_setup). */
    virtual std::vector<std::string> elements() const { return {}; }

    /** The name of the value an evaluation hands on to the next evaluation
     *  of its chain (ObservableEvalResult::hotvar_updated_value), "" for an
     *  observable that hands on none. The host keeps it per chain
     *  (HPO_Host::obs_loss): every chain of a parallel annealer starts
     *  from its own previous evaluation, whatever another chain did. */
    virtual std::string warm_start_name() const { return {}; }

    /** Evaluate observable value + W-derivative + (optional) adjoint
     *  rhs at the current model weights, from the observable's declared
     *  start. */
    virtual ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx) = 0;

    /** evaluate() from @p start, the value the previous evaluation of the
     *  chain handed on (nullopt: the declared start). An observable with a
     *  warm start overrides it; one without has none to start from. */
    virtual ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx,
        std::optional<double> start)
    {
        (void)start;
        return evaluate(model, dc, ctx);
    }

    /** Target value the optimiser drives toward. */
    virtual double target() const = 0;

    /** Loss weight on this observable's residual. */
    virtual double weight() const = 0;

    /** Loss function (L1/L2/HUBER/...). */
    virtual const tadah::LossFunction& lossfn() const = 0;

    /** Whether lossfn() is the default the parser hands the observable,
     *  the loss function of the LOSS line above its line, rather than one
     *  the line names (--loss): HPOTargets refuses a LOSS line that no
     *  term after it reads. An observable that names none reads it. */
    virtual bool reads_default_loss() const { return true; }
};

/* Names of the k-th PC_OBSERVABLE line, k its 1-based ordinal among the
 * PC_OBSERVABLE lines in configuration order: the loss.tadah column
 * L(<tag>)[k], the outvar.tadah column <tag>[k] and the journal name
 * <kind>[k] (the loss section, the new-best "top:" breakdown). */
inline std::string observable_loss_label(const Observable& o, std::size_t k)
{
    return "L(" + o.column_tag() + ")[" + std::to_string(k) + "]";
}

inline std::string observable_value_label(const Observable& o, std::size_t k)
{
    return o.column_tag() + "[" + std::to_string(k) + "]";
}

inline std::string observable_journal_label(const Observable& o, std::size_t k)
{
    return o.kind() + "[" + std::to_string(k) + "]";
}

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OBSERVABLES_OBSERVABLE_H

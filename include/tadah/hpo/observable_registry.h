#ifndef TADAH_HPO_OBSERVABLE_REGISTRY_H
#define TADAH_HPO_OBSERVABLE_REGISTRY_H

#include <Eigen/Dense>

#include <functional>
#include <string>

namespace tadah::core { class aeds_type2; }

namespace tadah::hpo {

/* Snapshot of the relaxed-state data an observable handle needs to
 * assemble its gradients. The host (Phase 3.5) populates this from a
 * LAMMPS+pair_tadah pair at the relaxed configuration x* — keeps the
 * registry LAMMPS-agnostic and unit-testable without an in-process
 * LAMMPS instance.
 *
 * `aeds` is the (n_desc × nlocal) descriptor matrix captured by
 * pair_tadah when capture_adjoint_state=true; cohesive-energy and pe
 * observables need it. Box-only observables (lx, vol) leave `aeds`
 * null. */
struct ObservableContext {
    int nlocal   = 0;
    int natoms   = 0;
    int n_desc   = 0;
    int box_dofs = 0;   /* 0, 1 (iso), 6 (Voigt aniso) */

    const tadah::core::aeds_type2* aeds = nullptr;

    /* Restricted-triangular box prds at x*. lx/ly/lz are the diagonal
     * extents; vol = lx·ly·lz for orthogonal boxes. */
    double lx = 0.0, ly = 0.0, lz = 0.0;
};

/* Pair of gradient functions for one scalar observable g(x, W).
 *
 *   dg_dx(ctx, out) → out has size 3*ctx.nlocal + ctx.box_dofs
 *   dg_dW(ctx, out) → out has size ctx.n_desc
 *
 * `dg_dx_is_zero` is the optimisation hint Phase 3.4 uses to skip the
 * Hessian solve entirely (∂g/∂x = 0 ⇒ λ_k = 0). `kind` is human
 * readable, for logging only. */
struct ObservableHandle {
    using GxFn = std::function<void(const ObservableContext&, Eigen::VectorXd& out)>;
    using GwFn = std::function<void(const ObservableContext&, Eigen::VectorXd& out)>;

    std::string kind;
    bool        dg_dx_is_zero = false;
    GxFn        dg_dx;
    GwFn        dg_dW;
};

/* Maps a LAMMPS variable expression string to closed-form handles.
 *
 * Recognised expressions (case-sensitive, whitespace tolerated;
 * leading `$(...)` immediate-eval wrappers stripped):
 *   - "c_eatoms/atoms"          cohesive E per atom
 *   - "pe"                      total potential energy
 *   - "vol"                     box volume (orthogonal)
 *   - "lx", "ly", "lz"          box edge length
 *   - "lx/<num>" / "ly/..."     box edge / numeric divisor
 *   - "atoms"                   atom count — constant, both gradients zero
 *   - "<a>+<b>" / "<a>-<b>"     linear combination, recurses on each side
 *   - "v_<name>"                resolved via `resolver(name)` if provided
 *
 * Unrecognised expressions (anything with a compute we don't know,
 * any division except `lx/<const>`, any product, etc.) return false
 * so the host can route the θ index to needs_fd. */
using VariableResolver = std::function<std::string(const std::string& varname)>;

bool try_register(const std::string& expr,
                  const VariableResolver& resolver,
                  ObservableHandle& out);

/* Overload without a resolver — same as passing a resolver that always
 * returns "". `v_<name>` references then fail to register. */
bool try_register(const std::string& expr, ObservableHandle& out);

}  // namespace tadah::hpo

#endif  // TADAH_HPO_OBSERVABLE_REGISTRY_H

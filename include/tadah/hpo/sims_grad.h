#ifndef TADAH_HPO_SIMS_GRAD_H
#define TADAH_HPO_SIMS_GRAD_H

#include <tadah/hpo/adjoint_solver.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observable_registry.h>

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

namespace tadah::hpo {

/* Phase 3.5 chain-rule core.
 *
 * Closes the gap between (a) the Phase 3.2 Hessian + Phase 3.3
 * observable registry + Phase 3.4 LDLT solver and (b) the orchestrator
 * gradient `g_out` consumed at `dloss_dtheta.h:800`.
 *
 * The math is, per LAMMPS observable `k` at the relaxed state x*(W):
 *
 *   solve   H · λ_k = (∂g_k/∂x)ᵀ           (Phase 3.4)
 *   apply   dO_k/dW = ∂g_k/∂W + λ_kᵀ · dF/dW          (this header)
 *   fold    dL/dW  += (∂L_k/∂O_k) · dO_k/dW           (loss gradient)
 *   chain   g_out[i] += Σ_c dL/dW[c] · dWdtheta[i, c] (Phase 2 dWdθ)
 *
 * Sign conventions:
 * - `AdjointSnapshot::dF_dW[(3*atom+axis) * n_desc + c]` is ∂F/∂W
 *   (BF_Linear: F = -fd·W ⇒ ∂F/∂W = -fd_at_xstar). So
 *   ∂²E/∂x∂W = -∂F/∂W and the chain reduces to
 *   `dO/dW = dg/dW − λ · (-∂F/∂W) = dg/dW + λ · ∂F/∂W = dg/dW + λ · dF_dW`.
 *   The kernel reads the snapshot directly without re-negating.
 * - `LossFunction::loss_prime(v, t, w)` returns ∂L/∂t (the second arg).
 *   `dL/dv = -loss_prime(v, t, w)` for residual-based losses (L1/L2/
 *   LOG_COSH/HUBER/TUKEY); MAPE/RMSLE are rejected (host falls back
 *   to FD for those θ).
 *
 * Box DOFs in `λ` (entries past 3*nlocal) are ignored by the IFT
 * kernel: the AdjointSnapshot only carries atom-axis rows of dF/dW.
 * For lattice-constant / vol observables (which carry rhs on box
 * DOFs), the box-tail of `λ` is still meaningful but Phase 3.7+ must
 * extend the snapshot with ∂(virial)/∂W to fold it in.
 */

/* ---------------------------------------------------------------- *
 *  Kernel 1 — per-observable IFT chain rule.
 *
 *  Computes dO/dW for one observable. Atom block: reads lambda
 *  entries [0, 3*nlocal) against `dF_dW`. Box block (Phase 3.7.5):
 *  when `dvirial_dW != nullptr && box_dofs > 0`, reads lambda entries
 *  [3*nlocal, 3*nlocal + box_dofs) against `dvirial_dW` (which stores
 *  `-d^2E/d eps_alpha dW_c`, i.e. the same "force-side" sign as
 *  dF_dW, so the chain contributes additively).
 *
 *  `dvirial_dW = nullptr` (or `box_dofs == 0`) preserves the pre-3.7.5
 *  atom-only behaviour byte-for-byte — the box-tail of lambda is
 *  silently ignored.
 * ---------------------------------------------------------------- */
void apply_ift_chain_kernel(const Eigen::VectorXd& dg_dW,
                            const Eigen::VectorXd& lambda,
                            const double*          dF_dW,
                            int                    nlocal,
                            int                    n_desc,
                            Eigen::VectorXd&       dO_dW_out,
                            const double*          dvirial_dW = nullptr,
                            int                    box_dofs   = 0);

/* ---------------------------------------------------------------- *
 *  Kernel 2 — loss-gradient fold.
 *
 *  Adds `(∂L/∂O) · dO_dW` into dL_dW. ∂L/∂O is computed from the
 *  LossFunction via the dispatcher described above. Returns false
 *  on unsupported loss kinds (MAPE/RMSLE in 3.5); caller routes that
 *  observable's θ to needs_fd.
 * ---------------------------------------------------------------- */
bool apply_loss_fold(const LossFunction&    loss_fn,
                     double                 v_observed,
                     double                 t_target,
                     double                 w_weight,
                     const Eigen::VectorXd& dO_dW,
                     Eigen::VectorXd&       dL_dW_inout);

/* ---------------------------------------------------------------- *
 *  Kernel 3 — external θ chain (gemv against dWdtheta).
 *
 *  dWdtheta is row-major over (n_theta_rows, n_cols); row `r` of
 *  dWdtheta is mapped into `g_out` at position `theta_row_to_g_idx[r]`.
 *  Additive: existing `g_out` entries are accumulated, not overwritten.
 * ---------------------------------------------------------------- */
void apply_dWdtheta_chain(const Eigen::VectorXd&     dL_dW,
                          const std::vector<double>& dWdtheta,
                          std::size_t                n_cols,
                          const std::vector<int>&    theta_row_to_g_idx,
                          std::vector<double>&       g_out_inout);

/* ---------------------------------------------------------------- *
 *  Single-script aggregator.
 *
 *  Loops over observables, invokes the registry, runs Kernel 1 +
 *  Kernel 2, and accumulates `dL_dW`. Returns true iff every
 *  observable was handled analytically. On the first unrecognised
 *  expression or unsupported loss kind, the function clears dL_dW
 *  back to its pre-call state and returns false — the caller treats
 *  the whole script as needs_fd. (3.7+ can lift this to a per-
 *  observable mask.)
 * ---------------------------------------------------------------- */
struct SimsGradScriptInput {
    ObservableContext         ctx;            /* populated from x* */
    Eigen::MatrixXd           H;              /* Phase 3.2 */

    /* Adjoint snapshot at x* — row-major (3*nlocal × n_desc). The
     * pointer's lifetime must outlive the call. */
    const double*             dF_dW = nullptr;

    /* Phase 3.7.5: box-DOF block of -d²E/d eps_α dW (row-major,
     * box_dofs × n_desc). nullptr or box_dofs==0 means "atom-only" —
     * the chain kernel then ignores the box-tail of λ as in pre-3.7.5
     * behaviour. Iso strain (box_dofs == 1) is the only supported mode
     * in 3.7.5; aniso Voigt-6 is deferred. */
    const double*             dvirial_dW = nullptr;
    int                       box_dofs   = 0;

    /* Per-observable arrays (size = k_count = i_varloss.size()). */
    std::vector<std::string>  expressions;
    std::vector<double>       v_observed;
    std::vector<double>       t_targets;
    std::vector<double>       w_weights;
    std::vector<LossFunction> loss_fns;

    /* Optional v_<name> resolver scoped to this script. */
    VariableResolver          resolver;

    /* Phase A masking: when the provider can't produce a valid input
     * for this script (LAMMPS run_script divergence, populate failure
     * on the random-init pot, …), it sets `failed=true` and leaves the
     * remaining fields empty/default. The chain helper skips failed
     * entries (zero contribution to the gradient) instead of forcing
     * the orchestrator to fall back to all-FD. Mathematically the
     * failing script's loss is FAIL_SCORE (a constant) so
     * ∂FAIL_SCORE/∂θ = 0 — skipping is correct in the continuous
     * region. */
    bool                      failed = false;
};

bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  Eigen::VectorXd&           dL_dW_out);

/* ---------------------------------------------------------------- *
 *  Top-level: loop over scripts and chain into `g_out` via dWdtheta.
 *
 *  Phase A masking: scripts with `failed=true` (LAMMPS divergence /
 *  populate failure under random-init pots) are SKIPPED — their loss
 *  is FAIL_SCORE, a constant, so ∂FAIL_SCORE/∂θ = 0 and the missing
 *  chain contribution is exact in the continuous region.
 *
 *  Returns true iff every NON-FAILED script was handled analytically.
 *  Returns false (and pushes all caller-supplied θ indices into
 *  `needs_fd`) only on STRUCTURAL failures: unsupported loss kind
 *  (MAPE/RMSLE), or observable-registry parser failure on the script's
 *  varloss expressions. The orchestrator then falls back to all-FD.
 * ---------------------------------------------------------------- */
bool compute_sims_grad_dWchain(
    const std::vector<SimsGradScriptInput>& scripts,
    const std::vector<double>&              dWdtheta,
    std::size_t                             n_cols,
    const std::vector<int>&                 theta_row_to_g_idx,
    const std::vector<int>&                 all_theta_indices,
    std::vector<double>&                    g_out_inout,
    std::vector<int>&                       needs_fd_out);

/* ---------------------------------------------------------------- *
 *  Convenience: ∂L/∂v for the supported loss kinds.
 *  Public so tests and 3.6's integration glue can re-use the same
 *  dispatcher without duplicating the sign-flip rule.
 *  Returns false (and leaves out untouched) on unsupported losses.
 * ---------------------------------------------------------------- */
bool loss_grad_wrt_observed(const LossFunction& fn,
                            double v_observed,
                            double t_target,
                            double w_weight,
                            double& out);

/* ---------------------------------------------------------------- *
 *  Typed-observable (PC_OBSERVABLE) chain rule.
 *
 *  The Tier-1/2 analogue of SimsGradScriptInput. Where SimsGrad
 *  carries an IFT-adjoint payload (Hessian, dF/dW, λ-rhs) because
 *  the LAMMPS-script observable's ∂g/∂x is generally non-zero, an
 *  Observable that sets `needs_adjoint=false` (cohesive energy and
 *  every observable evaluated at the energy minimum) reduces to a
 *  flat (value, target, weight, lossfn, dO/dW) tuple — no Hessian
 *  solve required.
 *
 *  MVP scope: only `needs_adjoint=false` observables. Anything that
 *  flips the flag (lattice_param, elastic_constants, ...) routes the
 *  whole call to FD until Phase 3 adjoint plumbing extends into the
 *  observable seam.
 *
 *  The helper chains `dO_dW` through the loss prime, then through the
 *  Phase 2 dW/dθ matrix (`apply_dWdtheta_chain`), and adds the
 *  contribution into `g_out_inout`.
 *
 *  Phase A masking: observables with `failed=true` (provider-detected
 *  relaxer divergence / NaN value) are SKIPPED — their loss is
 *  FAIL_SCORE, a constant, so ∂FAIL_SCORE/∂θ = 0 and the missing chain
 *  contribution is exact in the continuous region.
 *
 *  Returns true iff every NON-FAILED observable was handled
 *  analytically. Returns false (and pushes all caller-supplied θ
 *  indices into `needs_fd_out`) only on STRUCTURAL failures:
 *  unsupported loss kind (MAPE/RMSLE) or needs_adjoint=true on the
 *  MVP path. Matches compute_sims_grad_dWchain's contract.
 * ---------------------------------------------------------------- */
struct ObservableGradInput {
    /** Observed value at current W (e.g. Ecoh in eV/atom). */
    double          value   = 0.0;
    /** User-supplied target the optimiser drives toward. */
    double          target  = 0.0;
    /** Loss weight for this observable. */
    double          weight  = 1.0;
    /** Loss-function kind (L1/L2/HUBER/LOG_COSH/TUKEY supported;
     *  MAPE/RMSLE rejected, observable routes to FD). */
    LossFunction    loss_fn{};

    /** ∂O/∂W at current W (size = n_W). For cohesive_energy this is
     *  populated directly by the closed-form path in
     *  CohesiveEnergyObservable::evaluate. */
    Eigen::VectorXd dO_dW;

    /** Set true when the observable requires the IFT adjoint solve
     *  (∂g/∂x ≠ 0 at the relaxed state). MVP rejects — caller routes
     *  the call to FD. First user that will flip this is
     *  LatticeParamObservable (deferred). */
    bool            needs_adjoint = false;

    /** Phase A masking: provider sets `failed=true` (relaxer
     *  divergence, non-finite value, NN-finder fault, …) and leaves
     *  dO_dW empty. The chain helper skips failed entries (zero
     *  contribution) rather than poisoning the whole gradient. Same
     *  ∂FAIL_SCORE/∂θ = 0 justification as SimsGradScriptInput. */
    bool            failed = false;
};

bool compute_observables_grad_dWchain(
    const std::vector<ObservableGradInput>& observables,
    const std::vector<double>&              dWdtheta,
    std::size_t                             n_cols,
    const std::vector<int>&                 theta_row_to_g_idx,
    const std::vector<int>&                 all_theta_indices,
    std::vector<double>&                    g_out_inout,
    std::vector<int>&                       needs_fd_out);

}  // namespace tadah::hpo

#endif  // TADAH_HPO_SIMS_GRAD_H

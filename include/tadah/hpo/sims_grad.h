#ifndef TADAH_HPO_SIMS_GRAD_H
#define TADAH_HPO_SIMS_GRAD_H

#include <tadah/hpo/adjoint_solver.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observable_registry.h>
#include <tadah/hpo/observables/observable.h>

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

namespace tadah::hpo {

/* IFT-adjoint chain-rule core.
 *
 * Closes the gap between (a) the LAMMPS Hessian + the
 * observable registry + the LDLT solver and (b) the orchestrator
 * gradient `g_out` consumed at `dloss_dtheta.h:800`.
 *
 * The math is, per LAMMPS observable `k` at the relaxed state x*(W):
 *
 *   solve   H . lambda_k = (dg_k/dx)^T           (adjoint_solver.h)
 *   apply   dO_k/dW = dg_k/dW + lambda_k^T . dF/dW          (this header)
 *   fold    dL/dW  += (dL_k/dO_k) . dO_k/dW           (loss gradient)
 *   chain   g_out[i] += Sigma_c dL/dW[c] . dWdtheta[i, c] (dW/dtheta)
 *   direct  g_out[i] += Sigma_k (dL_k/dO_k) . (dg_k/dtheta_i
 *                       + lambda_k^T . dF/dtheta_i)       (at fixed x*, W)
 *
 * The last line is theta's own effect at fixed weights: a theta that is a
 * descriptor parameter changes the energy and the forces of the same W. By
 * the implicit function theorem dO_k/dtheta is the sum of the chain through
 * W and that direct term. SimsGradScriptInput carries its partial
 * derivatives (dg_dtheta, dF_dtheta, with direct_theta true) when the
 * provider can compute them; HPO_Host's provider does not (the adjoint
 * snapshot holds no derivative with respect to theta), so a script with
 * observables sends every theta to finite differences.
 *
 * Sign conventions:
 * - `AdjointSnapshot::dF_dW[(3*atom+axis) * n_desc + c]` is dF/dW
 *   (BF_Linear: F = -fd.W => dF/dW = -fd_at_xstar). So
 *   d^2E/dxdW = -dF/dW and the chain reduces to
 *   `dO/dW = dg/dW  lambda . (-dF/dW) = dg/dW + lambda . dF/dW = dg/dW + lambda . dF_dW`.
 *   The kernel reads the snapshot directly without re-negating.
 * - `LossFunction::loss_prime(v, t, w)` returns dL/dt (the second arg).
 *   `dL/dv = -loss_prime(v, t, w)` for residual-based losses (L1/L2/
 *   LOG_COSH/HUBER/TUKEY); MAPE/RMSLE are rejected (host falls back
 *   to FD for those theta).
 *
 * Box DOFs in `lambda` (entries past 3*nlocal) are ignored by the IFT
 * kernel: the AdjointSnapshot only carries atom-axis rows of dF/dW.
 * For lattice-constant / vol observables (which carry rhs on box
 * DOFs), the box-tail of `lambda` is still meaningful: Kernel 1 folds it in
 * through d(virial)/dW (its box block).
 */

/* ---------------------------------------------------------------- *
 *  Kernel 1 -- per-observable IFT chain rule.
 *
 *  Computes dO/dW for one observable. Atom block: reads lambda
 *  entries [0, 3*nlocal) against `dF_dW`. Box block:
 *  when `dvirial_dW != nullptr && box_dofs > 0`, reads lambda entries
 *  [3*nlocal, 3*nlocal + box_dofs) against `dvirial_dW` (which stores
 *  `-d^2E/d eps_alpha dW_c`, i.e. the same "force-side" sign as
 *  dF_dW, so the chain contributes additively).
 *
 *  `dvirial_dW = nullptr` (or `box_dofs == 0`) preserves the
 *  atom-only behaviour byte-for-byte -- the box-tail of lambda is
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
 *  Kernel 2 -- loss-gradient fold.
 *
 *  Adds `(dL/dO) . dO_dW` into dL_dW. dL/dO is computed from the
 *  LossFunction via the dispatcher described above. Returns false
 *  on unsupported loss kinds (MAPE/RMSLE); caller routes that
 *  observable's theta to needs_fd.
 * ---------------------------------------------------------------- */
bool apply_loss_fold(const LossFunction&    loss_fn,
                     double                 v_observed,
                     double                 t_target,
                     double                 w_weight,
                     const Eigen::VectorXd& dO_dW,
                     Eigen::VectorXd&       dL_dW_inout);

/* ---------------------------------------------------------------- *
 *  Kernel 3 -- external theta chain (gemv against dWdtheta).
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
 *  back to its pre-call state and returns false -- the caller treats
 *  the whole script as needs_fd. (A per-observable mask could lift
 *  this.)
 * ---------------------------------------------------------------- */
struct SimsGradScriptInput {
    ObservableContext         ctx;            /* populated from x* */
    Eigen::MatrixXd           H;              /* LammpsHessianAssembler */

    /* Adjoint snapshot at x* -- row-major (3*nlocal x n_desc). The
     * pointer's lifetime must outlive the call. */
    const double*             dF_dW = nullptr;

    /* Box-DOF block of -d^2E/d eps_alpha dW (row-major,
     * box_dofs x n_desc). nullptr or box_dofs==0 means "atom-only" --
     * the chain kernel then ignores the box-tail of lambda. pair_tadah
     * accumulates iso strain (box_dofs == 1) and the Voigt-6 block
     * (box_dofs == 6). */
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

    /* The direct theta term: what each theta row r of dWdtheta (n_theta
     * rows, the theta_row_to_g_idx of compute_sims_grad_dWchain) does at
     * fixed x* and W.
     *   dg_dtheta  -- k_count x n_theta, row-major: dg_k/dtheta_r (an
     *                 energy observable: the derivative of its energy; a
     *                 box observable such as lx: 0);
     *   dF_dtheta  -- n_theta x (3*ctx.nlocal + ctx.box_dofs), row-major:
     *                 the derivative of the generalised force, the atom
     *                 forces then the box rows in dvirial_dW's sign
     *                 (-d^2E/d eps_alpha dtheta_r), lambda's layout.
     * `direct_theta` says that the provider filled them. A script with
     * observables without it has an unknown direct term, and the chain
     * sends every theta to finite differences
     * (compute_sims_grad_dWchain); one whose theta enter only through W
     * gives zeros. A script without observables needs none. */
    bool                      direct_theta = false;
    std::vector<double>       dg_dtheta;
    std::vector<double>       dF_dtheta;

    /* Masking: when the provider can't produce a valid input
     * for this script (LAMMPS run_script divergence, populate failure
     * on the random-init pot, ...), it sets `failed=true` and leaves the
     * remaining fields empty/default. The chain helper skips failed
     * entries (zero contribution to the gradient) instead of forcing
     * the orchestrator to fall back to all-FD. Mathematically the
     * failing script's loss is FAIL_SCORE (a constant) so
     * dFAIL_SCORE/dtheta = 0 -- skipping is correct in the continuous
     * region. */
    bool                      failed = false;
};

bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  Eigen::VectorXd&           dL_dW_out);

/* The same, and also the direct theta term of the loss,
 *   dL_dtheta_direct_out[r] = Sigma_k (dL_k/dO_k) . (dg_k/dtheta_r
 *                             + lambda_k^T . dF/dtheta_r),
 * for the `n_theta` theta rows of the input's dg_dtheta and dF_dtheta,
 * with the lambda_k of dL_dW_out (input.direct_theta must be set when the
 * script has observables; a size that does not match throws). Returns
 * false where the two-argument form does, leaving both outputs. */
bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  std::size_t                n_theta,
                                  Eigen::VectorXd&           dL_dW_out,
                                  std::vector<double>&       dL_dtheta_direct_out);

/* ---------------------------------------------------------------- *
 *  Top-level: loop over scripts and chain into `g_out` via dWdtheta.
 *
 *  Masking: scripts with `failed=true` (LAMMPS divergence /
 *  populate failure under random-init pots) are SKIPPED -- their loss
 *  is FAIL_SCORE, a constant, so dFAIL_SCORE/dtheta = 0 and the missing
 *  chain contribution is exact in the continuous region.
 *
 *  Each script adds its chain through W and its direct theta term
 *  (SimsGradScriptInput::dg_dtheta, dF_dtheta) to the g_out slot of each
 *  theta row (theta_row_to_g_idx; -1: none).
 *
 *  Returns true iff every NON-FAILED script was handled analytically.
 *  Returns false (and pushes all caller-supplied theta indices into
 *  `needs_fd`) only on STRUCTURAL failures: unsupported loss kind
 *  (MAPE/RMSLE), observable-registry parser failure on the script's
 *  varloss expressions, or a script with observables whose input does not
 *  carry the direct theta term (direct_theta false: the term is unknown).
 *  The orchestrator then falls back to all-FD.
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
 *  Whether loss_grad_wrt_observed differentiates `fn`: the residual
 *  losses (L1, L2, LOG_COSH, HUBER, TUKEY) yes, MAPE and RMSLE no. The
 *  startup rule of GRAD_STRATEGY ANALYTICAL (analytical_loss_term_blocker,
 *  analytical_gate.h) asks it of each PC_OBSERVABLE --loss.
 * ---------------------------------------------------------------- */
bool loss_has_derivative_wrt_observed(const LossFunction& fn);

/* ---------------------------------------------------------------- *
 *  Convenience: dL/dv for the supported loss kinds.
 *  Public so tests and the LAMMPS-side adapter can re-use the same
 *  dispatcher without duplicating the sign-flip rule.
 *  Returns false (and leaves out untouched) on unsupported losses
 *  (loss_has_derivative_wrt_observed).
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
 *  carries an IFT-adjoint payload (Hessian, dF/dW, lambda-rhs) because
 *  the LAMMPS-script observable's dg/dx is generally non-zero, an
 *  Observable that sets `needs_adjoint=false` (cohesive energy and
 *  every observable evaluated at the energy minimum) reduces to a
 *  flat (value, target, weight, lossfn, dO/dW) tuple -- no Hessian
 *  solve required.
 *
 *  MVP scope: only `needs_adjoint=false` observables. Anything that
 *  flips the flag (lattice_param, elastic_constants, ...) routes the
 *  whole call to FD until the IFT-adjoint plumbing extends into the
 *  observable seam.
 *
 *  The helper chains `dO_dW` through the loss prime, then through the
 *  dW/dtheta matrix (`apply_dWdtheta_chain`), adds the
 *  observable's own dependence on theta at fixed weights
 *  (`dO_dtheta_explicit`, through the same loss prime), and adds the
 *  contribution into `g_out_inout`: dL/dtheta = dL/dO * (dO/dW . dW/dtheta
 *  + dO/dtheta|_W).
 *
 *  Masking: observables with `failed=true` (provider-detected
 *  relaxer divergence / NaN value) are SKIPPED -- their loss is
 *  FAIL_SCORE, a constant, so dFAIL_SCORE/dtheta = 0 and the missing chain
 *  contribution is exact in the continuous region. `constant=true`
 *  entries (the fixed penalty of an unconverged relaxation) are skipped
 *  for the same reason.
 *
 *  Returns true iff every NON-FAILED, non-constant observable was handled
 *  analytically. Returns false (and pushes all caller-supplied theta
 *  indices into `needs_fd_out`) only on STRUCTURAL failures:
 *  unsupported loss kind (MAPE/RMSLE), needs_adjoint=true on the
 *  MVP path, or no dO/dW (a value with no analytical derivative, such as
 *  the force-scaled penalty of an unconverged relaxation). Matches
 *  compute_sims_grad_dWchain's contract.
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

    /** dO/dW at current W (size = n_W). For cohesive_energy this is
     *  populated directly by the closed-form path in
     *  CohesiveEnergyObservable::evaluate. */
    Eigen::VectorXd dO_dW;

    /** Set true when the observable requires the IFT adjoint solve
     *  (dg/dx != 0 at the relaxed state). MVP rejects -- caller routes
     *  the call to FD. First user that will flip this is
     *  LatticeParamObservable (deferred). */
    bool            needs_adjoint = false;

    /** Masking: provider sets `failed=true` (relaxer
     *  divergence, non-finite value, NN-finder fault, ...) and leaves
     *  dO_dW empty. The chain helper skips failed entries (zero
     *  contribution) rather than poisoning the whole gradient. Same
     *  dFAIL_SCORE/dtheta = 0 justification as SimsGradScriptInput. */
    bool            failed = false;

    /** The value is a constant (ObservableEvalResult::value_is_constant,
     *  e.g. the penalty of an unconverged relaxation): zero gradient,
     *  skipped like a failed entry. */
    bool            constant = false;

    /** The model energies the value is linear in
     *  (ObservableEvalResult::energy_terms), for the orchestrator to
     *  differentiate into dO_dtheta_explicit. */
    std::vector<ObservableEvalResult::EnergyTerm> energy_terms;

    /** dO/dtheta at fixed weights, one entry per theta row of dWdtheta
     *  (theta_row_to_g_idx order); empty for none. */
    std::vector<double> dO_dtheta_explicit;
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

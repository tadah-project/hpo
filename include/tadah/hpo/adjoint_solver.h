#ifndef TADAH_HPO_ADJOINT_SOLVER_H
#define TADAH_HPO_ADJOINT_SOLVER_H

#include <Eigen/Dense>

namespace tadah::hpo {

/* Dense LDLT solver for the Phase 3 IFT-adjoint system  H . lambda = rhs.
 *
 * H is assembled by `LammpsHessianAssembler` (Phase 3.2) at the
 * relaxed atomic state x*. It is symmetric by construction, with a
 * null-space of (at most) 3 acoustic translation modes on the
 * atom block; box DOFs sit at the tail of the vector and are not
 * part of the projection.
 *
 * - **LDLT, not LLT.** A relaxed-but-not-converged minimiser can
 *   leave small negative pivots on near-zero modes; LDLT handles
 *   them without throwing.
 * - **Acoustic projection.** For each cartesian axis, subtract the
 *   per-axis mean of the atom block from `rhs` before solving;
 *   the translation modes are then orthogonal to the projected rhs.
 *   Skipped when `n_atom_dofs` is 0 (or the constructor's default
 *   sentinel, in which case the whole vector is treated as box DOFs).
 * - **Tikhonov fallback.** If the initial LDLT's smallest absolute
 *   pivot drops below `kPivotThreshold` (or the factor fails outright),
 *   `kTikhonovBump` is added to the diagonal once and the system
 *   re-factored; `regularised()` reflects this.
 * - **Zero-rhs fast path.** Phase 3.3's `dg_dx_is_zero` flag promises
 *   the rhs is identically zero (e.g. cohesive E per atom at x*).
 *   The solver short-circuits to `lambda = 0` without touching the LDLT.
 */
class DenseAdjointSolver {
public:
    /// Pivot magnitude below which the constructor applies a Tikhonov bump.
    static constexpr double kPivotThreshold = 1.0e-8;
    /// Amount added to the diagonal on the Tikhonov retry.
    static constexpr double kTikhonovBump   = 1.0e-8;

    /* Factorise H. `n_atom_dofs` defaults to -1, in which case the
     * full H is treated as box-only and acoustic projection is
     * skipped; pass the atom-DOF count (3 x n_atoms) explicitly to
     * enable the projection. H may be modified internally on the
     * Tikhonov retry. */
    explicit DenseAdjointSolver(Eigen::MatrixXd H, int n_atom_dofs = -1);

    /// lambda = H^-1 . (rhs projected). Resizes `lambda` to H.rows() doubles.
    void solve(const Eigen::VectorXd& rhs, Eigen::VectorXd& lambda);

    /// True if the ctor applied a Tikhonov diagonal bump to make H factorable.
    bool regularised() const { return regularised_; }

    /// Number of atom DOFs (0..H.rows()) carrying acoustic projection.
    int  n_atom_dofs() const { return n_atom_dofs_; }
    /// Total DOF count, == H.rows() at construction time.
    int  n_total() const     { return n_total_; }

private:
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    int  n_atom_dofs_ = 0;
    int  n_total_    = 0;
    bool regularised_ = false;
};

}  // namespace tadah::hpo

#endif  // TADAH_HPO_ADJOINT_SOLVER_H

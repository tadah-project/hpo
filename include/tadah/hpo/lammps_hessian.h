#ifndef TADAH_HPO_LAMMPS_HESSIAN_H
#define TADAH_HPO_LAMMPS_HESSIAN_H

#include <Eigen/Dense>

namespace LAMMPS_NS { class LAMMPS; }

namespace tadah::hpo {

/* Central-difference Hessian of E(x, eps) over pair_tadah::compute.
 *
 * Used by the IFT-adjoint to solve H . lambda_k = (dg_k/dx)^T at the
 * relaxed state x*. Owns only configuration; produces a fresh dense
 * Eigen::MatrixXd of size (3*nlocal + n_box_dofs) on every assemble().
 *
 * Box DOFs (Voigt order eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy):
 *   box_dofs == 0  no box block (atom-only Hessian).
 *   box_dofs == 1  iso strain only (single lambda DOF; affine atom scaling).
 *   box_dofs == 6  full Voigt aniso strain (deformation gradient F = I+eps).
 *
 * Atom positions and box matrix are restored to their pre-call state.
 * After the central-diff sweep destroys the IFT snapshot, the assembler
 * does one extra pair->compute(0, 0) with capture_adjoint_state=true to
 * refresh lt->adjoint_snapshot at x*; pass refresh_snapshot=false to
 * skip this step (e.g. when the caller will do its own capture pass).
 *
 * Symmetrisation and Tikhonov regularisation are applied unconditionally
 * before return; finite-precision drift can otherwise leave H slightly
 * asymmetric or carry near-zero pivots into the LDLT (adjoint_solver.h).
 */
class LammpsHessianAssembler {
public:
    LammpsHessianAssembler() = default;

    /// Tikhonov diagonal added to H before return (default 1e-8).
    void set_tikhonov(double t)       { tikhonov_ = t; }
    /// Symmetrise H = 0.5 (H + H^T) before regularising (default on).
    void set_symmetrize(bool on)      { symmetrize_ = on; }
    /// Refresh capture_adjoint_state snapshot post-assembly (default on).
    void set_refresh_snapshot(bool on){ refresh_snapshot_ = on; }

    /// Assemble H of size (3*atom->nlocal + box_dofs)^2.
    /// h_atom: atomic step in Ang. h_strain: strain step (dimensionless).
    Eigen::MatrixXd assemble(LAMMPS_NS::LAMMPS* lmp,
                             int    box_dofs = 0,
                             double h_atom   = 1.0e-4,
                             double h_strain = 1.0e-5);

private:
    double tikhonov_         = 1.0e-8;
    bool   symmetrize_       = true;
    bool   refresh_snapshot_ = true;
};

}  // namespace tadah::hpo

#endif  // TADAH_HPO_LAMMPS_HESSIAN_H

#include <tadah/hpo/adjoint_solver.h>

#include <stdexcept>

namespace tadah::hpo {

DenseAdjointSolver::DenseAdjointSolver(Eigen::MatrixXd H, int n_atom_dofs)
    : n_atom_dofs_(n_atom_dofs < 0 ? 0 : n_atom_dofs),
      n_total_(static_cast<int>(H.rows()))
{
    if (H.rows() != H.cols()) {
        throw std::runtime_error("DenseAdjointSolver: H must be square");
    }
    if (n_atom_dofs_ > n_total_) {
        throw std::runtime_error(
            "DenseAdjointSolver: n_atom_dofs exceeds H.rows()");
    }

    ldlt_.compute(H);

    bool needs_bump = false;
    if (ldlt_.info() != Eigen::Success) {
        needs_bump = true;
    } else {
        const auto D = ldlt_.vectorD();
        if (D.size() == 0 || D.cwiseAbs().minCoeff() < kPivotThreshold) {
            needs_bump = true;
        }
    }

    if (needs_bump) {
        for (int i = 0; i < n_total_; ++i) H(i, i) += kTikhonovBump;
        ldlt_.compute(H);
        regularised_ = true;
        if (ldlt_.info() != Eigen::Success) {
            throw std::runtime_error(
                "DenseAdjointSolver: LDLT failed even after Tikhonov bump");
        }
    }
}

void DenseAdjointSolver::solve(const Eigen::VectorXd& rhs,
                               Eigen::VectorXd& lambda)
{
    if (rhs.size() != n_total_) {
        throw std::runtime_error(
            "DenseAdjointSolver: rhs size does not match H.rows()");
    }

    /* Fast-path: identically-zero rhs => lambda = 0. Pairs with
     * `ObservableHandle::dg_dx_is_zero` so the caller can
     * skip the call entirely; this catch is the defensive belt. */
    if (rhs.cwiseAbs().maxCoeff() == 0.0) {
        lambda = Eigen::VectorXd::Zero(n_total_);
        return;
    }

    Eigen::VectorXd r = rhs;

    /* Acoustic translation-mode projection. For each cartesian axis,
     * subtract the per-axis mean of the atom block. The mode vectors
     * are (1,0,0,1,0,0,...), (0,1,0,...), (0,0,1,...) on the atom block and
     * zero on the box block; orthogonalising rhs against them lets the
     * LDLT solve absorb the null-space without polluting box DOFs. */
    if (n_atom_dofs_ > 0 && (n_atom_dofs_ % 3 == 0)) {
        const int N = n_atom_dofs_ / 3;
        for (int axis = 0; axis < 3; ++axis) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i) sum += r(3 * i + axis);
            const double mean = sum / static_cast<double>(N);
            for (int i = 0; i < N; ++i) r(3 * i + axis) -= mean;
        }
    }

    lambda = ldlt_.solve(r);
}

}  // namespace tadah::hpo

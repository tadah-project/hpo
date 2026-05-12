#include <tadah/hpo/sims_grad.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace tadah::hpo {

bool loss_grad_wrt_observed(const tadah::LossFunction& fn,
                            double v_observed,
                            double t_target,
                            double w_weight,
                            double& out)
{
    using Type = tadah::LossFunction::Type;
    switch (fn.type()) {
        /* Residual-based: L = f(v - t), so ∂L/∂v = -∂L/∂t = -loss_prime. */
        case Type::L1:
        case Type::L2:
        case Type::LOG_COSH:
        case Type::HUBER:
        case Type::TUKEY:
            out = -fn.loss_prime(v_observed, t_target, w_weight);
            return true;
        /* Non-residual: MAPE depends on |y|, RMSLE on log(y); the
         * "negate loss_prime" trick doesn't hold. Phase 3.7+ can
         * hand-code these; for 3.5 we route the observable to FD. */
        case Type::MAPE:
        case Type::RMSLE:
            return false;
    }
    return false;
}

void apply_ift_chain_kernel(const Eigen::VectorXd& dg_dW,
                            const Eigen::VectorXd& lambda,
                            const double*          dF_dW,
                            int                    nlocal,
                            int                    n_desc,
                            Eigen::VectorXd&       dO_dW_out,
                            const double*          dvirial_dW,
                            int                    box_dofs)
{
    if (dg_dW.size() != n_desc) {
        throw std::runtime_error(
            "apply_ift_chain_kernel: dg_dW size != n_desc");
    }
    if (lambda.size() < 3 * nlocal) {
        throw std::runtime_error(
            "apply_ift_chain_kernel: lambda smaller than atom block");
    }
    if (box_dofs < 0) {
        throw std::runtime_error(
            "apply_ift_chain_kernel: box_dofs must be >= 0");
    }
    if (box_dofs > 0 && lambda.size() < 3 * nlocal + box_dofs) {
        throw std::runtime_error(
            "apply_ift_chain_kernel: lambda smaller than atom+box block");
    }

    dO_dW_out = dg_dW;
    if (dF_dW && nlocal > 0) {
        /* dO_dW[c] += Σ_(atom,axis) λ[3*atom+axis] · dF_dW[(3*atom+axis), c]. */
        for (int atom = 0; atom < nlocal; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const double lam = lambda(3 * atom + axis);
                if (lam == 0.0) continue;
                const double* row = dF_dW + static_cast<std::ptrdiff_t>(
                                        (3 * atom + axis)) * n_desc;
                for (int c = 0; c < n_desc; ++c) {
                    dO_dW_out(c) += lam * row[c];
                }
            }
        }
    }

    /* Box block: dO_dW[c] += Σ_α λ[3*nlocal+α] · dvirial_dW[α, c]. */
    if (dvirial_dW && box_dofs > 0) {
        for (int alpha = 0; alpha < box_dofs; ++alpha) {
            const double lam = lambda(3 * nlocal + alpha);
            if (lam == 0.0) continue;
            const double* row = dvirial_dW + static_cast<std::ptrdiff_t>(
                                    alpha) * n_desc;
            for (int c = 0; c < n_desc; ++c) {
                dO_dW_out(c) += lam * row[c];
            }
        }
    }
}

bool apply_loss_fold(const tadah::LossFunction& loss_fn,
                     double                 v_observed,
                     double                 t_target,
                     double                 w_weight,
                     const Eigen::VectorXd& dO_dW,
                     Eigen::VectorXd&       dL_dW_inout)
{
    double g = 0.0;
    if (!loss_grad_wrt_observed(loss_fn, v_observed, t_target, w_weight, g)) {
        return false;
    }
    if (dL_dW_inout.size() == 0) {
        dL_dW_inout = Eigen::VectorXd::Zero(dO_dW.size());
    } else if (dL_dW_inout.size() != dO_dW.size()) {
        throw std::runtime_error(
            "apply_loss_fold: dL_dW size mismatches dO_dW");
    }
    dL_dW_inout += g * dO_dW;
    return true;
}

void apply_dWdtheta_chain(const Eigen::VectorXd&     dL_dW,
                          const std::vector<double>& dWdtheta,
                          std::size_t                n_cols,
                          const std::vector<int>&    theta_row_to_g_idx,
                          std::vector<double>&       g_out_inout)
{
    if (n_cols == 0) return;
    if (dL_dW.size() != static_cast<Eigen::Index>(n_cols)) {
        throw std::runtime_error(
            "apply_dWdtheta_chain: dL_dW size != n_cols");
    }
    const std::size_t n_rows = theta_row_to_g_idx.size();
    if (dWdtheta.size() < n_rows * n_cols) {
        throw std::runtime_error(
            "apply_dWdtheta_chain: dWdtheta buffer too small");
    }
    for (std::size_t r = 0; r < n_rows; ++r) {
        const int g_idx = theta_row_to_g_idx[r];
        if (g_idx < 0) continue;          /* skip unmapped rows */
        if (g_idx >= static_cast<int>(g_out_inout.size())) {
            throw std::runtime_error(
                "apply_dWdtheta_chain: theta_row_to_g_idx maps outside g_out");
        }
        const double* drow = dWdtheta.data() + r * n_cols;
        double acc = 0.0;
        for (std::size_t c = 0; c < n_cols; ++c) {
            acc += dL_dW(static_cast<Eigen::Index>(c)) * drow[c];
        }
        g_out_inout[static_cast<std::size_t>(g_idx)] += acc;
    }
}

bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  Eigen::VectorXd&           dL_dW_out)
{
    const std::size_t k_count = input.expressions.size();
    if (k_count == 0) {
        /* No observables on this script. Trivially analytic — dL_dW stays
         * at its incoming size; clear to zero of n_desc length for the
         * caller's convenience. */
        dL_dW_out = Eigen::VectorXd::Zero(input.ctx.n_desc);
        return true;
    }
    if (input.v_observed.size() != k_count ||
        input.t_targets.size()  != k_count ||
        input.w_weights.size()  != k_count ||
        input.loss_fns.size()   != k_count)
    {
        throw std::runtime_error(
            "compute_sims_grad_for_script: per-observable array size mismatch");
    }

    /* Defer the LDLT until at least one observable carries dg_dx ≠ 0;
     * cohesive-energy-per-atom-only scripts (pilot) bypass it. */
    DenseAdjointSolver* solver_ptr = nullptr;
    std::unique_ptr<DenseAdjointSolver> solver;

    Eigen::VectorXd staged = Eigen::VectorXd::Zero(input.ctx.n_desc);

    for (std::size_t k = 0; k < k_count; ++k) {
        ObservableHandle h;
        if (!try_register(input.expressions[k], input.resolver, h)) {
            return false;
        }

        Eigen::VectorXd dg_dW;
        h.dg_dW(input.ctx, dg_dW);

        Eigen::VectorXd lambda =
            Eigen::VectorXd::Zero(3 * input.ctx.nlocal + input.ctx.box_dofs);
        if (!h.dg_dx_is_zero) {
            Eigen::VectorXd dg_dx;
            h.dg_dx(input.ctx, dg_dx);
            /* Belt-and-braces: handle says non-zero but the actual vector
             * came out zero (e.g. lattice_constant with box_dofs=0 in the
             * context). Skip the solve. */
            if (dg_dx.cwiseAbs().maxCoeff() == 0.0) {
                lambda.setZero();
            } else {
                if (!solver) {
                    solver = std::make_unique<DenseAdjointSolver>(
                        Eigen::MatrixXd(input.H),
                        /*n_atom_dofs=*/3 * input.ctx.nlocal);
                    solver_ptr = solver.get();
                }
                solver_ptr->solve(dg_dx, lambda);
            }
        }

        Eigen::VectorXd dO_dW;
        apply_ift_chain_kernel(dg_dW, lambda, input.dF_dW,
                               input.ctx.nlocal, input.ctx.n_desc, dO_dW,
                               input.dvirial_dW, input.box_dofs);

        if (!apply_loss_fold(input.loss_fns[k],
                             input.v_observed[k],
                             input.t_targets[k],
                             input.w_weights[k],
                             dO_dW, staged))
        {
            return false;
        }
    }

    dL_dW_out = std::move(staged);
    return true;
}

bool compute_sims_grad_dWchain(
    const std::vector<SimsGradScriptInput>& scripts,
    const std::vector<double>&              dWdtheta,
    std::size_t                             n_cols,
    const std::vector<int>&                 theta_row_to_g_idx,
    const std::vector<int>&                 all_theta_indices,
    std::vector<double>&                    g_out_inout,
    std::vector<int>&                       needs_fd_out)
{
    /* Stage contributions to a local buffer so a mid-loop failure
     * doesn't leave a half-applied g_out behind. */
    std::vector<double> stage(g_out_inout.size(), 0.0);

    for (const auto& script : scripts) {
        Eigen::VectorXd dL_dW;
        if (!compute_sims_grad_for_script(script, dL_dW)) {
            /* All-or-nothing in 3.5: route every θ to needs_fd, discard
             * the staged contributions, leave g_out untouched. */
            for (int idx : all_theta_indices) needs_fd_out.push_back(idx);
            return false;
        }
        apply_dWdtheta_chain(dL_dW, dWdtheta, n_cols,
                             theta_row_to_g_idx, stage);
    }

    /* Commit. */
    for (std::size_t i = 0; i < g_out_inout.size(); ++i) {
        g_out_inout[i] += stage[i];
    }
    return true;
}

}  // namespace tadah::hpo

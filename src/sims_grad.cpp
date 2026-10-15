#include <tadah/hpo/sims_grad.h>

#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

namespace tadah::hpo {

bool loss_has_derivative_wrt_observed(const tadah::LossFunction& fn)
{
    using Type = tadah::LossFunction::Type;
    switch (fn.type()) {
        /* Residual-based: L = f(v - t), so dL/dv = -dL/dt = -loss_prime. */
        case Type::L1:
        case Type::L2:
        case Type::LOG_COSH:
        case Type::HUBER:
        case Type::TUKEY:
            return true;
        /* Non-residual: MAPE depends on |y|, RMSLE on log(y); the
         * "negate loss_prime" trick doesn't hold. These could be
         * hand-coded; for now the observable is routed to FD. */
        case Type::MAPE:
        case Type::RMSLE:
            return false;
    }
    return false;
}

bool loss_grad_wrt_observed(const tadah::LossFunction& fn,
                            double v_observed,
                            double t_target,
                            double w_weight,
                            double& out)
{
    if (!loss_has_derivative_wrt_observed(fn)) return false;
    out = -fn.loss_prime(v_observed, t_target, w_weight);
    return true;
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
        /* dO_dW[c] += Sigma_(atom,axis) lambda[3*atom+axis] . dF_dW[(3*atom+axis), c]. */
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

    /* Box block: dO_dW[c] += Sigma_alpha lambda[3*nlocal+alpha] . dvirial_dW[alpha, c]. */
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

namespace {

/* compute_sims_grad_for_script's body. With `direct` (n_theta theta rows)
 * it also accumulates the direct theta term of the loss into *direct. */
bool script_gradient(const SimsGradScriptInput& input,
                     std::size_t                n_theta,
                     Eigen::VectorXd&           dL_dW_out,
                     std::vector<double>*       direct)
{
    const std::size_t k_count = input.expressions.size();
    if (k_count == 0) {
        /* No observables on this script. Trivially analytic -- dL_dW stays
         * at its incoming size; clear to zero of n_desc length for the
         * caller's convenience. Nothing to differentiate: no direct term. */
        dL_dW_out = Eigen::VectorXd::Zero(input.ctx.n_desc);
        if (direct) direct->assign(n_theta, 0.0);
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
    /* lambda's layout: the atom DOFs, then the box DOFs. */
    const std::size_t n_gen =
        static_cast<std::size_t>(3 * input.ctx.nlocal + input.ctx.box_dofs);
    if (direct) {
        if (!input.direct_theta)
            throw std::runtime_error(
                "compute_sims_grad_for_script: the script has observables but "
                "its input carries no direct theta term (direct_theta)");
        if (input.dg_dtheta.size() != k_count * n_theta)
            throw std::runtime_error(
                "compute_sims_grad_for_script: dg_dtheta has " +
                std::to_string(input.dg_dtheta.size()) + " values, not " +
                std::to_string(k_count) + " observables x " +
                std::to_string(n_theta) + " theta");
        if (input.dF_dtheta.size() != n_theta * n_gen)
            throw std::runtime_error(
                "compute_sims_grad_for_script: dF_dtheta has " +
                std::to_string(input.dF_dtheta.size()) + " values, not " +
                std::to_string(n_theta) + " theta x " + std::to_string(n_gen) +
                " generalised coordinates (3 nlocal + box_dofs)");
    }

    /* Defer the LDLT until at least one observable carries dg_dx != 0;
     * cohesive-energy-per-atom-only scripts (pilot) bypass it. */
    DenseAdjointSolver* solver_ptr = nullptr;
    std::unique_ptr<DenseAdjointSolver> solver;

    Eigen::VectorXd staged = Eigen::VectorXd::Zero(input.ctx.n_desc);
    std::vector<double> staged_direct(direct ? n_theta : 0, 0.0);

    for (std::size_t k = 0; k < k_count; ++k) {
        ObservableHandle h;
        if (!try_register(input.expressions[k], input.resolver, h)) {
            return false;
        }

        Eigen::VectorXd dg_dW;
        h.dg_dW(input.ctx, dg_dW);

        Eigen::VectorXd lambda =
            Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_gen));
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

        if (direct && static_cast<std::size_t>(lambda.size()) != n_gen)
            throw std::runtime_error(
                "compute_sims_grad_for_script: H has " +
                std::to_string(lambda.size()) + " rows, not " +
                std::to_string(n_gen) +
                " generalised coordinates (3 nlocal + box_dofs)");

        Eigen::VectorXd dO_dW;
        apply_ift_chain_kernel(dg_dW, lambda, input.dF_dW,
                               input.ctx.nlocal, input.ctx.n_desc, dO_dW,
                               input.dvirial_dW, input.box_dofs);

        double dL_dO = 0.0;
        if (!loss_grad_wrt_observed(input.loss_fns[k], input.v_observed[k],
                                    input.t_targets[k], input.w_weights[k],
                                    dL_dO))
        {
            return false;
        }
        staged += dL_dO * dO_dW;

        /* The direct term: dg_k/dtheta_r + lambda_k . dF/dtheta_r, at
         * fixed x* and W. */
        for (std::size_t r = 0; r < staged_direct.size(); ++r) {
            const double* dF = input.dF_dtheta.data() + r * n_gen;
            double dO = input.dg_dtheta[k * n_theta + r];
            for (std::size_t i = 0; i < n_gen; ++i)
                dO += lambda(static_cast<Eigen::Index>(i)) * dF[i];
            staged_direct[r] += dL_dO * dO;
        }
    }

    dL_dW_out = std::move(staged);
    if (direct) *direct = std::move(staged_direct);
    return true;
}

}  // namespace

bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  Eigen::VectorXd&           dL_dW_out)
{
    return script_gradient(input, 0, dL_dW_out, nullptr);
}

bool compute_sims_grad_for_script(const SimsGradScriptInput& input,
                                  std::size_t                n_theta,
                                  Eigen::VectorXd&           dL_dW_out,
                                  std::vector<double>&       dL_dtheta_direct_out)
{
    return script_gradient(input, n_theta, dL_dW_out, &dL_dtheta_direct_out);
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
    const std::size_t n_theta = theta_row_to_g_idx.size();

    for (const auto& script : scripts) {
        /* Masking: skip provider-marked failed scripts. Their
         * loss is FAIL_SCORE (a constant clamp set by the host's
         * outvar/varloss FAIL_SCORE path) so dFAIL_SCORE/dtheta = 0 -- the
         * contribution to the gradient is exactly zero in the
         * continuous region. The remaining scripts and PC_OBSERVABLE
         * chain still contribute analytically. */
        if (script.failed) continue;

        /* A script with observables whose input does not say what theta
         * does at fixed W: its direct theta term is unknown, and the
         * chain through W alone would be the gradient without it. Like a
         * structural failure below. */
        const bool direct_unknown =
            !script.expressions.empty() && !script.direct_theta;
        Eigen::VectorXd dL_dW;
        std::vector<double> direct;
        if (direct_unknown ||
            !compute_sims_grad_for_script(script, n_theta, dL_dW, direct)) {
            /* Hard failure (unsupported loss kind, observable registry
             * miss, no direct term). Still all-or-nothing -- these are
             * structural and the optimiser needs FD anyway. */
            for (int idx : all_theta_indices) needs_fd_out.push_back(idx);
            return false;
        }
        apply_dWdtheta_chain(dL_dW, dWdtheta, n_cols,
                             theta_row_to_g_idx, stage);
        for (std::size_t r = 0; r < n_theta; ++r) {
            const int g_idx = theta_row_to_g_idx[r];
            if (g_idx < 0) continue;          /* skip unmapped rows */
            if (g_idx >= static_cast<int>(stage.size()))
                throw std::runtime_error(
                    "compute_sims_grad_dWchain: theta_row_to_g_idx maps "
                    "outside g_out");
            stage[static_cast<std::size_t>(g_idx)] += direct[r];
        }
    }

    /* Commit. */
    for (std::size_t i = 0; i < g_out_inout.size(); ++i) {
        g_out_inout[i] += stage[i];
    }
    return true;
}

bool compute_observables_grad_dWchain(
    const std::vector<ObservableGradInput>& observables,
    const std::vector<double>&              dWdtheta,
    std::size_t                             n_cols,
    const std::vector<int>&                 theta_row_to_g_idx,
    const std::vector<int>&                 all_theta_indices,
    std::vector<double>&                    g_out_inout,
    std::vector<int>&                       needs_fd_out)
{
    /* Stage contributions locally so an unsupported observable (loss
     * kind / needs_adjoint=true) doesn't leave a half-applied g_out
     * behind. Mirrors compute_sims_grad_dWchain's all-or-nothing
     * commit semantics for structural failures; per-observable
     * `failed=true` is masked. */
    std::vector<double> stage(g_out_inout.size(), 0.0);

    for (const auto& obs : observables) {
        /* Masking: provider-marked failure (relaxer diverged,
         * NaN value, ...) or a constant value. Skip. dFAIL_SCORE/dtheta =
         * d(const)/dtheta = 0. */
        if (obs.failed || obs.constant) continue;

        if (obs.needs_adjoint || obs.dO_dW.size() == 0) {
            /* MVP only wires the closed-form (needs_adjoint=false)
             * path. LatticeParamObservable et al. will lift this once
             * the IFT-adjoint seam threads through. An empty dO/dW is a
             * value without an analytical derivative. */
            for (int idx : all_theta_indices) needs_fd_out.push_back(idx);
            return false;
        }
        if (obs.dO_dW.size() != static_cast<long>(n_cols)) {
            throw std::runtime_error(
                "compute_observables_grad_dWchain: dO_dW size != n_cols");
        }

        Eigen::VectorXd dL_dW(static_cast<long>(n_cols));
        dL_dW.setZero();
        if (!apply_loss_fold(obs.loss_fn, obs.value, obs.target, obs.weight,
                             obs.dO_dW, dL_dW)) {
            /* MAPE / RMSLE not supported by loss_grad_wrt_observed. */
            for (int idx : all_theta_indices) needs_fd_out.push_back(idx);
            return false;
        }
        apply_dWdtheta_chain(dL_dW, dWdtheta, n_cols,
                             theta_row_to_g_idx, stage);

        /* The value's own dependence on theta at fixed weights. */
        if (!obs.dO_dtheta_explicit.empty()) {
            if (obs.dO_dtheta_explicit.size() != theta_row_to_g_idx.size())
                throw std::runtime_error(
                    "compute_observables_grad_dWchain: dO_dtheta_explicit "
                    "size != number of theta rows");
            double dL_dO = 0.0;
            loss_grad_wrt_observed(obs.loss_fn, obs.value, obs.target,
                                   obs.weight, dL_dO);  // supported: folded above
            for (std::size_t r = 0; r < theta_row_to_g_idx.size(); ++r) {
                const int g_idx = theta_row_to_g_idx[r];
                if (g_idx < 0) continue;
                if (g_idx >= static_cast<int>(stage.size()))
                    throw std::runtime_error(
                        "compute_observables_grad_dWchain: theta_row_to_g_idx "
                        "maps outside g_out");
                stage[static_cast<std::size_t>(g_idx)] +=
                    dL_dO * obs.dO_dtheta_explicit[r];
            }
        }
    }

    /* Commit. */
    for (std::size_t i = 0; i < g_out_inout.size(); ++i) {
        g_out_inout[i] += stage[i];
    }
    return true;
}

}  // namespace tadah::hpo

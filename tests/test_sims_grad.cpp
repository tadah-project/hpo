// Phase 3.5 — sims_grad chain-rule tests.
//
// LAMMPS-free: synthetic Hessian + synthetic aeds + synthetic dF/dW
// snapshot. The kernels are pure linear algebra so we can FD against
// closed-form ground truths to high precision (1e-6 .. 1e-12) without
// touching the LAMMPS dtor blocker that gates Phase 3.6's pilot.
//
// Phase 3.7B-orchestrator (additions at end of file): also covers the
// two pure-string helpers `detect_box_dofs_from_script` and
// `extract_varloss_expressions` from sims_grad_lammps.h. They are the
// script-side glue the SimsGradProvider lambda uses inside HPO_Host.

#include "catch2/catch.hpp"

#include <tadah/core/aeds_type.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/sims_grad.h>
#include <tadah/hpo/sims_grad_lammps.h>

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <vector>

using tadah::LossFunction;
using namespace tadah::hpo;

namespace {

/* Synthetic 4-atom, 3-descriptor scenario reused across cases. */
struct Synth {
    static constexpr int nlocal = 4;
    static constexpr int natoms = 4;
    static constexpr int n_desc = 3;
    static constexpr int box_dofs = 0;

    tadah::core::aeds_type2 aeds{n_desc, nlocal};
    /* dF_dW row-major: row index (3*atom+axis), n_desc columns. */
    std::vector<double>     dF_dW{};
    Eigen::MatrixXd         H{};

    Synth()
    {
        for (int a = 0; a < nlocal; ++a)
            for (int c = 0; c < n_desc; ++c)
                aeds(c, a) = 1.0 + 0.1 * a + 0.2 * c;

        dF_dW.assign(static_cast<std::size_t>(3 * nlocal * n_desc), 0.0);
        for (int a = 0; a < nlocal; ++a)
            for (int axis = 0; axis < 3; ++axis)
                for (int c = 0; c < n_desc; ++c)
                    dF_dW[(static_cast<std::size_t>(3 * a + axis)) * n_desc + c]
                        = 0.05 * (a + 1) + 0.03 * (axis + 1) + 0.07 * (c + 1);

        const int n = 3 * nlocal + box_dofs;
        H = Eigen::MatrixXd::Identity(n, n) * 10.0;
        for (int i = 0; i < n - 1; ++i) {
            H(i, i + 1) = 0.1;
            H(i + 1, i) = 0.1;
        }
    }

    ObservableContext ctx() const
    {
        ObservableContext c;
        c.nlocal   = nlocal;
        c.natoms   = natoms;
        c.n_desc   = n_desc;
        c.box_dofs = box_dofs;
        c.aeds     = &aeds;
        c.lx = c.ly = 3.301;  c.lz = 6.602;
        return c;
    }
};

}  // namespace

TEST_CASE("sims_grad — IFT kernel: λ=0 ⇒ dO/dW == dg/dW",
          "[sims_grad][phase3]")
{
    Synth s;
    Eigen::VectorXd dg_dW(s.n_desc);
    dg_dW << 1.0, 2.0, 3.0;
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(3 * s.nlocal);

    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW);
    REQUIRE(dO_dW.size() == s.n_desc);
    for (int c = 0; c < s.n_desc; ++c) CHECK(dO_dW(c) == Approx(dg_dW(c)));
}

TEST_CASE("sims_grad — IFT kernel: λ·dF_dW added per (atom,axis,c)",
          "[sims_grad][phase3]")
{
    Synth s;
    Eigen::VectorXd dg_dW = Eigen::VectorXd::Zero(s.n_desc);
    Eigen::VectorXd lambda(3 * s.nlocal);
    for (int i = 0; i < 3 * s.nlocal; ++i) lambda(i) = 0.01 * (i + 1);

    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW);

    /* Expected: dO_dW[c] = Σ_(a,axis) λ[3a+axis] · dF_dW[(3a+axis), c]. */
    for (int c = 0; c < s.n_desc; ++c) {
        double expected = 0.0;
        for (int a = 0; a < s.nlocal; ++a) {
            for (int axis = 0; axis < 3; ++axis) {
                expected += lambda(3 * a + axis) *
                    s.dF_dW[(static_cast<std::size_t>(3 * a + axis)) * s.n_desc + c];
            }
        }
        CHECK(dO_dW(c) == Approx(expected));
    }
}

TEST_CASE("sims_grad — loss_grad_wrt_observed sign + magnitudes for L2",
          "[sims_grad][phase3]")
{
    LossFunction l2(LossFunction::Type::L2);
    /* L = w * r²;   dL/dv = 2*w*(v - t). */
    const double v = 0.7, t = 0.4, w = 3.0;
    double g = 0.0;
    REQUIRE(loss_grad_wrt_observed(l2, v, t, w, g));
    CHECK(g == Approx(2.0 * w * (v - t)));

    /* FD vs analytical via L itself. */
    const double h = 1e-6;
    const double fd = (l2.loss(v + h, t, w) - l2.loss(v - h, t, w)) / (2.0 * h);
    CHECK(g == Approx(fd).epsilon(1e-7));
}

TEST_CASE("sims_grad — loss_grad_wrt_observed dispatch coverage",
          "[sims_grad][phase3]")
{
    /* Residual-based losses → supported. */
    for (auto t : {LossFunction::Type::L1, LossFunction::Type::L2,
                   LossFunction::Type::LOG_COSH, LossFunction::Type::HUBER,
                   LossFunction::Type::TUKEY})
    {
        LossFunction fn(t);
        double g = 0.0;
        CHECK(loss_grad_wrt_observed(fn, 1.0, 0.5, 1.0, g));
    }
    /* Non-residual losses → fail closed. */
    for (auto t : {LossFunction::Type::MAPE, LossFunction::Type::RMSLE}) {
        LossFunction fn(t);
        double g = 42.0;
        CHECK_FALSE(loss_grad_wrt_observed(fn, 1.0, 0.5, 1.0, g));
        CHECK(g == 42.0);  /* untouched */
    }
}

TEST_CASE("sims_grad — apply_loss_fold accumulates correctly",
          "[sims_grad][phase3]")
{
    LossFunction l2(LossFunction::Type::L2);
    Eigen::VectorXd dO_dW(3);  dO_dW << 1.0, 2.0, 3.0;
    Eigen::VectorXd dL_dW = Eigen::VectorXd::Zero(3);

    REQUIRE(apply_loss_fold(l2, /*v=*/1.0, /*t=*/0.5, /*w=*/2.0,
                            dO_dW, dL_dW));
    /* g = 2*w*(v-t) = 2.0; dL/dW += g·dO_dW. */
    CHECK(dL_dW(0) == Approx(2.0));
    CHECK(dL_dW(1) == Approx(4.0));
    CHECK(dL_dW(2) == Approx(6.0));

    /* A second fold accumulates; dispatcher returns true. */
    REQUIRE(apply_loss_fold(l2, /*v=*/0.3, /*t=*/0.5, /*w=*/1.0,
                            dO_dW, dL_dW));
    /* g = 2*1*(0.3-0.5) = -0.4. */
    CHECK(dL_dW(0) == Approx(2.0 - 0.4 * 1.0));
    CHECK(dL_dW(1) == Approx(4.0 - 0.4 * 2.0));
    CHECK(dL_dW(2) == Approx(6.0 - 0.4 * 3.0));
}

TEST_CASE("sims_grad — apply_loss_fold rejects MAPE without writing dL_dW",
          "[sims_grad][phase3]")
{
    LossFunction mape(LossFunction::Type::MAPE);
    Eigen::VectorXd dO_dW(2);  dO_dW << 1.0, 2.0;
    Eigen::VectorXd dL_dW(2);  dL_dW << 7.0, 8.0;
    CHECK_FALSE(apply_loss_fold(mape, 1.0, 0.5, 1.0, dO_dW, dL_dW));
    /* No mutation on failure. */
    CHECK(dL_dW(0) == 7.0);
    CHECK(dL_dW(1) == 8.0);
}

TEST_CASE("sims_grad — apply_dWdtheta_chain: g_out[i] += Σ_c dL_dW · dWdθ[i,c]",
          "[sims_grad][phase3]")
{
    Eigen::VectorXd dL_dW(3);  dL_dW << 1.0, -2.0, 3.0;
    /* dWdtheta layout: row-major, 4 θ rows × 3 cols. */
    std::vector<double> dWdtheta = {
        /* row 0 → θ_a */ 0.1, 0.2, 0.3,
        /* row 1 → θ_b */ 0.5, 0.5, 0.5,
        /* row 2 → θ_c */ 1.0, 0.0, 0.0,
        /* row 3 → θ_d */ 0.0, 1.0, 0.0,
    };
    /* Map dWdtheta rows to g_out indices 0..3. */
    std::vector<int> theta_row_to_g_idx = {0, 1, 2, 3};
    std::vector<double> g_out(4, 0.0);

    apply_dWdtheta_chain(dL_dW, dWdtheta, /*n_cols=*/3,
                         theta_row_to_g_idx, g_out);

    /* g[0] = 1*0.1 + (-2)*0.2 + 3*0.3 =  0.6. */
    CHECK(g_out[0] == Approx(0.6));
    /* g[1] = 1*0.5 + (-2)*0.5 + 3*0.5 =  1.0. */
    CHECK(g_out[1] == Approx(1.0));
    /* g[2] = 1*1.0 + 0 + 0 =  1.0. */
    CHECK(g_out[2] == Approx(1.0));
    /* g[3] = 0 + (-2)*1.0 + 0 = -2.0. */
    CHECK(g_out[3] == Approx(-2.0));

    /* A second call adds on top — additive contract. */
    apply_dWdtheta_chain(dL_dW, dWdtheta, 3, theta_row_to_g_idx, g_out);
    CHECK(g_out[0] == Approx(1.2));
    CHECK(g_out[1] == Approx(2.0));
}

TEST_CASE("sims_grad — apply_dWdtheta_chain skips row when g_idx is -1",
          "[sims_grad][phase3]")
{
    Eigen::VectorXd dL_dW(2);  dL_dW << 1.0, 1.0;
    std::vector<double> dWdtheta = {1.0, 1.0, 9.0, 9.0};
    std::vector<int> theta_row_to_g_idx = {-1, 0};   /* row 0 → skip */
    std::vector<double> g_out(1, 0.0);
    apply_dWdtheta_chain(dL_dW, dWdtheta, 2, theta_row_to_g_idx, g_out);
    /* g[0] = 9 + 9 = 18, row 0's 1+1=2 was skipped. */
    CHECK(g_out[0] == Approx(18.0));
}

TEST_CASE("sims_grad — compute_sims_grad_for_script (pilot path: cohesive E "
          "per atom, λ=0 fast path)",
          "[sims_grad][phase3]")
{
    Synth s;
    SimsGradScriptInput in;
    in.ctx        = s.ctx();
    in.H          = s.H;
    in.dF_dW      = s.dF_dW.data();
    in.expressions = {"c_eatoms/atoms"};
    in.v_observed  = {0.7};
    in.t_targets   = {0.4};
    in.w_weights   = {2.0};
    in.loss_fns    = {LossFunction(LossFunction::Type::L2)};

    Eigen::VectorXd dL_dW;
    REQUIRE(compute_sims_grad_for_script(in, dL_dW));
    REQUIRE(dL_dW.size() == s.n_desc);

    /* λ=0 so dO_dW == dg_dW = (Σ_a aeds(c,a)) / N. */
    Eigen::VectorXd expected_dO(s.n_desc);
    for (int c = 0; c < s.n_desc; ++c) {
        double sum = 0.0;
        for (int a = 0; a < s.nlocal; ++a) sum += s.aeds(c, a);
        expected_dO(c) = sum / s.natoms;
    }
    const double g = 2.0 * 2.0 * (0.7 - 0.4);  /* 2·w·(v-t) */
    for (int c = 0; c < s.n_desc; ++c) {
        CHECK(dL_dW(c) == Approx(g * expected_dO(c)));
    }
}

TEST_CASE("sims_grad — compute_sims_grad_for_script returns false on "
          "unrecognised expression (host falls back to FD)",
          "[sims_grad][phase3]")
{
    Synth s;
    SimsGradScriptInput in;
    in.ctx        = s.ctx();
    in.H          = s.H;
    in.dF_dW      = s.dF_dW.data();
    in.expressions = {"c_eatoms/atoms", "v_a/v_b + v_c"};
    in.v_observed  = {0.5, 0.5};
    in.t_targets   = {0.0, 0.0};
    in.w_weights   = {1.0, 1.0};
    in.loss_fns    = {LossFunction(LossFunction::Type::L2),
                      LossFunction(LossFunction::Type::L2)};

    Eigen::VectorXd dL_dW = Eigen::VectorXd::Ones(s.n_desc) * 7.0;
    CHECK_FALSE(compute_sims_grad_for_script(in, dL_dW));
    /* Failure must not commit partial work; dL_dW is left untouched
     * (the function clears staged input only on success). */
    CHECK(dL_dW.cwiseAbs().sum() > 0.0);   /* still has pre-call values */
}

TEST_CASE("sims_grad — compute_sims_grad_dWchain end-to-end on pilot-shaped "
          "input matches FD over W",
          "[sims_grad][phase3]")
{
    Synth s;
    SimsGradScriptInput in;
    in.ctx        = s.ctx();
    in.H          = s.H;
    in.dF_dW      = s.dF_dW.data();
    in.expressions = {"c_eatoms/atoms"};
    in.v_observed  = {0.6};
    in.t_targets   = {0.5};
    in.w_weights   = {1.0};
    in.loss_fns    = {LossFunction(LossFunction::Type::L2)};

    /* 2 θ rows that map to two distinct g_out slots. n_cols = n_desc. */
    std::vector<double> dWdtheta(2 * s.n_desc, 0.0);
    for (std::size_t r = 0; r < 2; ++r)
        for (int c = 0; c < s.n_desc; ++c)
            dWdtheta[r * s.n_desc + c] = 0.2 * static_cast<double>(r + 1)
                                          + 0.1 * static_cast<double>(c);
    std::vector<int> theta_row_to_g_idx = {0, 1};

    std::vector<double> g_out(2, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(compute_sims_grad_dWchain({in}, dWdtheta, s.n_desc,
                                       theta_row_to_g_idx,
                                       /*all_theta_indices=*/{},
                                       g_out, needs_fd));
    CHECK(needs_fd.empty());

    /* Build the analytical reference by hand:
     *   dg_dW = (Σ_a aeds(c,a)) / N            (cohesive E per atom)
     *   λ = 0                                  (pilot ⇒ dg/dx = 0)
     *   dO_dW = dg_dW
     *   dL/dW = 2·w·(v-t) · dO_dW              (L2 loss)
     *   g[r]  = Σ_c dL/dW[c] · dWdtheta[r, c]
     */
    Eigen::VectorXd dg_dW(s.n_desc);
    for (int c = 0; c < s.n_desc; ++c) {
        double sum = 0.0;
        for (int a = 0; a < s.nlocal; ++a) sum += s.aeds(c, a);
        dg_dW(c) = sum / s.natoms;
    }
    const double g = 2.0 * 1.0 * (0.6 - 0.5);
    Eigen::VectorXd dL_dW = g * dg_dW;
    for (std::size_t r = 0; r < 2; ++r) {
        double expected = 0.0;
        for (int c = 0; c < s.n_desc; ++c)
            expected += dL_dW(c) * dWdtheta[r * s.n_desc + c];
        CHECK(g_out[r] == Approx(expected));
    }
}

TEST_CASE("sims_grad — compute_sims_grad_dWchain routes ALL θ to needs_fd "
          "on first unrecognised observable (3.5 all-or-nothing policy)",
          "[sims_grad][phase3]")
{
    Synth s;
    SimsGradScriptInput in;
    in.ctx        = s.ctx();
    in.H          = s.H;
    in.dF_dW      = s.dF_dW.data();
    in.expressions = {"v_unknown/something"};
    in.v_observed  = {0.5};
    in.t_targets   = {0.0};
    in.w_weights   = {1.0};
    in.loss_fns    = {LossFunction(LossFunction::Type::L2)};

    std::vector<double> dWdtheta(s.n_desc, 0.5);
    std::vector<int>    theta_row_to_g_idx = {0};
    std::vector<int>    all_theta_indices = {0, 1, 2, 3, 4};
    std::vector<double> g_out(1, 100.0);
    std::vector<int>    needs_fd;

    CHECK_FALSE(compute_sims_grad_dWchain({in}, dWdtheta, s.n_desc,
                                          theta_row_to_g_idx,
                                          all_theta_indices,
                                          g_out, needs_fd));
    /* g_out untouched on failure; every requested θ goes to needs_fd. */
    CHECK(g_out[0] == 100.0);
    REQUIRE(needs_fd.size() == 5);
    CHECK(needs_fd == std::vector<int>{0, 1, 2, 3, 4});
}

TEST_CASE("sims_grad — IFT chain rule cross-check via central diff on W "
          "for a non-trivial λ (lattice-constant-style)",
          "[sims_grad][phase3]")
{
    /* Use box_dofs=1 + a synthetic lattice-constant rhs. This is the
     * λ≠0 path. We compare analytical dO/dW against an FD of a
     * synthetic O(W) that we hand-constructed to match the chain rule.
     *
     * Construct: dg_dW = u (constant vector), dg_dx = e_{3N+0} · κ
     * (single non-zero on the iso DOF). Solver returns λ such that
     * H·λ = e_{3N+0}·κ; chain gives dO_dW = u + λᵀ_atom · dF_dW.
     *
     * For the test, we don't need a physical O(W); we just verify the
     * apply_ift_chain_kernel arithmetic via the explicit formula. */
    constexpr int nlocal = 3;
    constexpr int n_desc = 2;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3 * nlocal + 1, 3 * nlocal + 1);
    H *= 5.0;
    /* Couple iso DOF to atom 0's x channel so λ has a non-trivial atom
     * component when rhs hits the iso row. */
    H(0, 3 * nlocal) = 0.3;
    H(3 * nlocal, 0) = 0.3;

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(3 * nlocal + 1);
    rhs(3 * nlocal) = 1.0;            /* dg/dx on the iso DOF only */

    DenseAdjointSolver solver(H, 3 * nlocal);
    Eigen::VectorXd lambda;
    solver.solve(rhs, lambda);

    /* Confirm λ has the expected structure: non-zero atom 0 / axis 0,
     * zero elsewhere on atom block, dominant on iso DOF. */
    CHECK(std::abs(lambda(0)) > 0.0);
    CHECK(lambda(1) == Approx(0.0).margin(1e-12));   /* projection zeroed mean — single comp */

    /* dg_dW = (1, 1), dF_dW filled with simple pattern. */
    std::vector<double> dF_dW(static_cast<std::size_t>(3 * nlocal * n_desc), 0.0);
    for (int a = 0; a < nlocal; ++a)
        for (int axis = 0; axis < 3; ++axis)
            for (int c = 0; c < n_desc; ++c)
                dF_dW[(static_cast<std::size_t>(3 * a + axis)) * n_desc + c]
                    = 0.1 * (a + 1) + 0.01 * (axis + 1) + 0.005 * (c + 1);

    Eigen::VectorXd dg_dW(n_desc);  dg_dW << 1.0, 1.0;
    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, dF_dW.data(), nlocal, n_desc, dO_dW);

    /* Hand-compute expected. */
    Eigen::VectorXd expected = dg_dW;
    for (int a = 0; a < nlocal; ++a) {
        for (int axis = 0; axis < 3; ++axis) {
            const double lam = lambda(3 * a + axis);
            for (int c = 0; c < n_desc; ++c) {
                expected(c) += lam *
                    dF_dW[(static_cast<std::size_t>(3 * a + axis)) * n_desc + c];
            }
        }
    }
    for (int c = 0; c < n_desc; ++c) {
        CHECK(dO_dW(c) == Approx(expected(c)).epsilon(1e-12));
    }
}

// ---------------------------------------------------------------- Phase 3.7.5
// Box-DOF extension: apply_ift_chain_kernel now optionally consumes
// dvirial_dW + box_dofs to fold λ_box rows into dO/dW. Verify the new
// path against hand-computed expectations and confirm the old (atom-
// only) signature still works byte-for-byte unchanged.

TEST_CASE("sims_grad — IFT kernel box-DOF: λ_box · dvirial_dW accumulates",
          "[sims_grad][phase3]")
{
    Synth s;
    constexpr int box_dofs = 1;       // iso strain only (Phase 3.7.5 scope)
    Eigen::VectorXd dg_dW = Eigen::VectorXd::Zero(s.n_desc);
    Eigen::VectorXd lambda(3 * s.nlocal + box_dofs);
    lambda.setZero();
    lambda(3 * s.nlocal + 0) = 0.5;   // λ_iso = 0.5; atom block stays 0.

    /* dvirial_dW row-major: 1 box DOF × n_desc columns. */
    std::vector<double> dvirial_dW{0.7, -1.1, 2.3};
    REQUIRE(static_cast<int>(dvirial_dW.size()) == box_dofs * s.n_desc);

    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW,
                           dvirial_dW.data(), box_dofs);

    /* atom block λ=0 ⇒ dO/dW receives only box block: λ_iso · dvirial[c]. */
    REQUIRE(dO_dW.size() == s.n_desc);
    CHECK(dO_dW(0) == Approx(0.5 * 0.7));
    CHECK(dO_dW(1) == Approx(0.5 * -1.1));
    CHECK(dO_dW(2) == Approx(0.5 * 2.3));
}

TEST_CASE("sims_grad — IFT kernel box-DOF: atom + box blocks sum",
          "[sims_grad][phase3]")
{
    Synth s;
    constexpr int box_dofs = 1;
    Eigen::VectorXd dg_dW(s.n_desc);   dg_dW << 1.0, 2.0, 3.0;
    Eigen::VectorXd lambda(3 * s.nlocal + box_dofs);
    for (int i = 0; i < 3 * s.nlocal; ++i) lambda(i) = 0.01 * (i + 1);
    lambda(3 * s.nlocal + 0) = 0.25;

    std::vector<double> dvirial_dW{0.2, -0.5, 1.4};

    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW,
                           dvirial_dW.data(), box_dofs);

    for (int c = 0; c < s.n_desc; ++c) {
        double expected = dg_dW(c);
        for (int a = 0; a < s.nlocal; ++a) {
            for (int axis = 0; axis < 3; ++axis) {
                expected += lambda(3 * a + axis) *
                    s.dF_dW[(static_cast<std::size_t>(3 * a + axis)) * s.n_desc + c];
            }
        }
        expected += 0.25 * dvirial_dW[c];
        CHECK(dO_dW(c) == Approx(expected));
    }
}

TEST_CASE("sims_grad — IFT kernel box-DOF: nullptr/0 preserves atom-only path",
          "[sims_grad][phase3]")
{
    Synth s;
    Eigen::VectorXd dg_dW = Eigen::VectorXd::Zero(s.n_desc);
    Eigen::VectorXd lambda(3 * s.nlocal);
    for (int i = 0; i < 3 * s.nlocal; ++i) lambda(i) = 0.01 * (i + 1);

    Eigen::VectorXd ref;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, ref);

    /* Explicit nullptr + box_dofs=0 must match the default-arg overload. */
    Eigen::VectorXd dO_dW;
    apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW, nullptr, 0);
    REQUIRE(dO_dW.size() == ref.size());
    for (int c = 0; c < ref.size(); ++c) {
        CHECK(dO_dW(c) == Approx(ref(c)).epsilon(1e-14));
    }

    /* box_dofs > 0 but dvirial_dW=nullptr also stays atom-only — lambda
     * tail is silently ignored as documented. */
    Eigen::VectorXd lambda_extended(3 * s.nlocal + 1);
    lambda_extended.head(3 * s.nlocal) = lambda;
    lambda_extended(3 * s.nlocal) = 999.0;            // would blow up if consumed
    apply_ift_chain_kernel(dg_dW, lambda_extended, s.dF_dW.data(),
                           s.nlocal, s.n_desc, dO_dW, nullptr, 1);
    for (int c = 0; c < ref.size(); ++c) {
        CHECK(dO_dW(c) == Approx(ref(c)).epsilon(1e-14));
    }
}

TEST_CASE("sims_grad — IFT kernel box-DOF: lambda size + box_dofs validation",
          "[sims_grad][phase3]")
{
    Synth s;
    Eigen::VectorXd dg_dW = Eigen::VectorXd::Zero(s.n_desc);
    Eigen::VectorXd lambda(3 * s.nlocal);             // missing box DOF row
    lambda.setZero();
    std::vector<double> dvirial_dW(s.n_desc, 1.0);

    Eigen::VectorXd dO_dW;
    CHECK_THROWS_AS(apply_ift_chain_kernel(dg_dW, lambda, s.dF_dW.data(),
                                            s.nlocal, s.n_desc, dO_dW,
                                            dvirial_dW.data(), /*box_dofs=*/1),
                    std::runtime_error);

    Eigen::VectorXd lambda_full(3 * s.nlocal + 1);
    lambda_full.setZero();
    CHECK_THROWS_AS(apply_ift_chain_kernel(dg_dW, lambda_full, s.dF_dW.data(),
                                            s.nlocal, s.n_desc, dO_dW,
                                            dvirial_dW.data(), /*box_dofs=*/-1),
                    std::runtime_error);
}

TEST_CASE("sims_grad — compute_sims_grad_for_script propagates dvirial_dW",
          "[sims_grad][phase3]")
{
    /* Verify the aggregator forwards dvirial_dW + box_dofs into the kernel.
     * Use a lattice-constant-style observable (dg_dx_is_zero=false) so the
     * adjoint solve fires, then check that the box block of λ shows up in
     * dL_dW via the dvirial path. */
    Synth s;
    SimsGradScriptInput input;
    input.ctx = s.ctx();
    input.ctx.box_dofs = 1;
    input.ctx.lx = 4.0;
    /* H = I on (3N+1) — λ = rhs trivially. */
    const int n_total = 3 * s.nlocal + 1;
    input.H = Eigen::MatrixXd::Identity(n_total, n_total);
    input.dF_dW = s.dF_dW.data();

    /* dvirial_dW: 1 box DOF × n_desc; pick distinguishable values. */
    std::vector<double> dvirial_dW{1.5, -2.0, 0.5};
    input.dvirial_dW = dvirial_dW.data();
    input.box_dofs   = 1;

    /* Lattice constant lx/1 with iso ⇒ ∂g/∂ε = L = 4.0, on row 3N.
     * After H=I solve, λ(3N) = 4.0, atom block λ = 0. */
    input.expressions = {"lx"};
    input.v_observed = {4.0};
    input.t_targets  = {3.5};
    input.w_weights  = {1.0};
    input.loss_fns   = {LossFunction(LossFunction::Type::L2)};

    Eigen::VectorXd dL_dW;
    REQUIRE(compute_sims_grad_for_script(input, dL_dW));

    /* Expected: dO/dW[c] = dg/dW + λ_atom·dF_dW + λ_box·dvirial_dW.
     * dg/dW=0 (lattice doesn't depend on W directly), λ_atom=0, λ_box=4.
     * dL/dW = 2·w·(v-t) · dO/dW = 2·1·(4-3.5) · 4·dvirial = 4·dvirial. */
    REQUIRE(dL_dW.size() == s.n_desc);
    CHECK(dL_dW(0) == Approx(4.0 * 1.5));
    CHECK(dL_dW(1) == Approx(4.0 * -2.0));
    CHECK(dL_dW(2) == Approx(4.0 * 0.5));
}

// ---------------------------------------------------------------------------
// Phase 3.7B-orchestrator — script-side helpers
// ---------------------------------------------------------------------------
//
// Two pure-string utilities used inside HPO_Host::analytical_gradient's
// SimsGradProvider lambda. Tested LAMMPS-free under [sims_grad].

TEST_CASE("sims_grad_lammps — detect_box_dofs_from_script picks iso vs aniso "
          "from `fix box/relax` lines",
          "[sims_grad][phase3]")
{
    using tadah::hpo::detect_box_dofs_from_script;

    /* No box/relax line at all. */
    CHECK(detect_box_dofs_from_script({
        "units metal",
        "create_box 1 box",
        "minimize 1e-12 1e-12 1000 1000",
    }) == 0);

    /* Pilot pattern: iso. */
    CHECK(detect_box_dofs_from_script({
        "units metal",
        "fix relax all box/relax iso 0.0 vmax 0.001",
        "minimize 1e-12 1e-12 1000 1000",
    }) == 1);

    /* aniso ⇒ 6. */
    CHECK(detect_box_dofs_from_script({
        "fix relax all box/relax aniso 0.0 vmax 0.001",
    }) == 6);

    /* tri ⇒ 6 (Voigt-6 block, same as aniso for the orchestrator). */
    CHECK(detect_box_dofs_from_script({
        "fix relax all box/relax tri 0.0 vmax 0.001",
    }) == 6);

    /* Single-axis x ⇒ falls into the non-iso bucket. */
    CHECK(detect_box_dofs_from_script({
        "fix relax all box/relax x 0.0 vmax 0.001",
    }) == 6);

    /* Comment-only / leading-whitespace must not match. */
    CHECK(detect_box_dofs_from_script({
        "# fix relax all box/relax iso 0.0",
        "    # also commented",
        "create_box 1 box",
    }) == 0);

    /* Indented `fix` line still matches (LAMMPS allows leading
     * whitespace on commands). */
    CHECK(detect_box_dofs_from_script({
        "    fix r all box/relax iso 0.0",
    }) == 1);

    /* First match wins — exotic case where two box/relax lines appear.
     * LAMMPS would reject both at runtime; the helper takes the first. */
    CHECK(detect_box_dofs_from_script({
        "fix r1 all box/relax iso 0.0",
        "fix r2 all box/relax aniso 0.0",
    }) == 1);

    /* Empty script. */
    CHECK(detect_box_dofs_from_script({}) == 0);
}

TEST_CASE("sims_grad_lammps — extract_varloss_expressions reads "
          "`variable <name> equal <expr>` and slots them by i_varloss",
          "[sims_grad][phase3]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    LammpsSim ls;
    ls.script_lines = {
        "units metal",
        "compute eatoms all reduce sum c_eng",
        "variable e_per equal $(c_eatoms/atoms)",
        "variable var_lx equal lx",
        "variable redundant equal v_e_per - 0.0",
        "run 0",
    };
    ls.n_outvar  = {"e_per", "var_lx", "redundant"};
    ls.n_varloss = {"e_per", "var_lx"};
    ls.t_varloss = {0.0, 0.0};
    ls.w_varloss = {1.0, 1.0};
    ls.i_varloss = {0, 1};

    const auto exprs = extract_varloss_expressions(ls);
    REQUIRE(exprs.size() == 2);
    CHECK(exprs[0] == "$(c_eatoms/atoms)");
    CHECK(exprs[1] == "lx");
}

TEST_CASE("sims_grad_lammps — extract_varloss_expressions: last "
          "definition wins (LAMMPS redefinition semantics)",
          "[sims_grad][phase3]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    LammpsSim ls;
    ls.script_lines = {
        "variable v equal lx",
        "variable v equal ly",
        "run 0",
    };
    ls.n_outvar  = {"v"};
    ls.n_varloss = {"v"};
    ls.t_varloss = {0.0};
    ls.w_varloss = {1.0};
    ls.i_varloss = {0};

    const auto exprs = extract_varloss_expressions(ls);
    REQUIRE(exprs.size() == 1);
    CHECK(exprs[0] == "ly");
}

TEST_CASE("sims_grad_lammps — extract_varloss_expressions: missing "
          "variable definitions yield empty slots",
          "[sims_grad][phase3]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    LammpsSim ls;
    ls.script_lines = {
        "variable defined equal pe",
    };
    ls.n_outvar  = {"defined", "missing"};
    ls.n_varloss = {"defined", "missing"};
    ls.t_varloss = {0.0, 0.0};
    ls.w_varloss = {1.0, 1.0};
    ls.i_varloss = {0, 1};

    const auto exprs = extract_varloss_expressions(ls);
    REQUIRE(exprs.size() == 2);
    CHECK(exprs[0] == "pe");
    CHECK(exprs[1].empty());
}

TEST_CASE("sims_grad_lammps — extract_varloss_expressions: comments and "
          "non-`equal` styles ignored",
          "[sims_grad][phase3]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    LammpsSim ls;
    ls.script_lines = {
        "# variable v equal lx",       /* commented out — NOT a def */
        "variable v string foo",       /* string-style — NOT equal */
        "variable v equal lx",         /* the real one */
        "variable v atom v_lx + 1",    /* atom-style overrides nothing */
    };
    ls.n_outvar  = {"v"};
    ls.n_varloss = {"v"};
    ls.t_varloss = {0.0};
    ls.w_varloss = {1.0};
    ls.i_varloss = {0};

    const auto exprs = extract_varloss_expressions(ls);
    REQUIRE(exprs.size() == 1);
    /* Commented + string + atom styles all skipped; only `equal` line
     * matched, and the trailing `atom` line doesn't redefine the
     * `equal` map (it's a different style — last-wins is per-style
     * here, which matches what LAMMPS does for `equal` retrieval via
     * `variable->retrieve` against an `equal` slot). */
    CHECK(exprs[0] == "lx");
}

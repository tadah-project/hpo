// sims_grad chain-rule tests.
//
// LAMMPS-free: synthetic Hessian + synthetic aeds + synthetic dF/dW
// snapshot. The kernels are pure linear algebra so we can FD against
// closed-form ground truths to high precision (1e-6 .. 1e-12) without
// touching the LAMMPS dtor blocker that gates the end-to-end pilot.
//
// Additions at the end of the file also cover the
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

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
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

/* The direct theta term of a script whose n_theta theta enter only through
 * the weights: zero (dg/dtheta and dF/dtheta at fixed W). */
void theta_only_through_w(SimsGradScriptInput& in, std::size_t n_theta)
{
    in.direct_theta = true;
    in.dg_dtheta.assign(in.expressions.size() * n_theta, 0.0);
    in.dF_dtheta.assign(
        n_theta * static_cast<std::size_t>(3 * in.ctx.nlocal + in.ctx.box_dofs),
        0.0);
}

}  // namespace

TEST_CASE("sims_grad — IFT kernel: λ=0 ⇒ dO/dW == dg/dW",
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
    /* These θ enter only through W ("matches FD over W"): no direct term.
     * (The chain routes every θ to FD for a script that does not say so.) */
    theta_only_through_w(in, 2);

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
          "on first unrecognised observable (all-or-nothing policy)",
          "[sims_grad][adjoint]")
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
    /* A direct term given, so that the expression is what fails. */
    theta_only_through_w(in, 1);

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

/* ---------------------------------------------------------------- *
 * Masking: SimsGradScriptInput::failed = true is silently
 * skipped (zero contribution to the chain). Surviving entries still
 * fold their analytical contributions. This is the load-bearing
 * behaviour for HPO configs that mix PC_LAMMPS scripts with random-
 * init pots — one diverging script must not poison Tier-1
 * PC_OBSERVABLE chain gradients.
 * ---------------------------------------------------------------- */
TEST_CASE("sims_grad — compute_sims_grad_dWchain skips failed=true entries "
          "and still folds the surviving scripts (failed-entry masking)",
          "[sims_grad][failed_masking][adjoint]")
{
    Synth s;

    /* Good script — same as the pilot-shaped end-to-end case. */
    SimsGradScriptInput good;
    good.ctx        = s.ctx();
    good.H          = s.H;
    good.dF_dW      = s.dF_dW.data();
    good.expressions = {"c_eatoms/atoms"};
    good.v_observed  = {0.6};
    good.t_targets   = {0.5};
    good.w_weights   = {1.0};
    good.loss_fns    = {LossFunction(LossFunction::Type::L2)};
    theta_only_through_w(good, 2);   /* θ only through W */

    /* Failed script: would be rejected outright by the chain helper
     * (`v_unknown/...`) but `failed=true` masks the whole entry, so
     * compute_sims_grad_for_script is never called. The chain helper
     * must NOT report a structural failure on this input. */
    SimsGradScriptInput bad;
    bad.failed = true;
    /* Leave the rest default-constructed: provider's failed path
     * doesn't populate expressions / loss_fns. */

    std::vector<double> dWdtheta(2 * s.n_desc, 0.0);
    for (std::size_t r = 0; r < 2; ++r)
        for (int c = 0; c < s.n_desc; ++c)
            dWdtheta[r * s.n_desc + c] = 0.2 * static_cast<double>(r + 1)
                                          + 0.1 * static_cast<double>(c);
    std::vector<int> theta_row_to_g_idx = {0, 1};

    /* Reference: re-run with only the good script. */
    std::vector<double> g_ref(2, 0.0);
    {
        std::vector<int> nf;
        REQUIRE(compute_sims_grad_dWchain({good}, dWdtheta, s.n_desc,
                                           theta_row_to_g_idx,
                                           /*all_theta_indices=*/{},
                                           g_ref, nf));
        CHECK(nf.empty());
    }

    /* Bad before good, after good, and sandwiched — order
     * must not matter, the bad entry contributes zero in all cases. */
    for (auto order : std::vector<std::vector<SimsGradScriptInput>>{
            {bad, good}, {good, bad}, {bad, good, bad}}) {
        std::vector<double> g_out(2, 0.0);
        std::vector<int>    needs_fd;
        REQUIRE(compute_sims_grad_dWchain(order, dWdtheta, s.n_desc,
                                           theta_row_to_g_idx,
                                           /*all_theta_indices=*/{},
                                           g_out, needs_fd));
        CHECK(needs_fd.empty());
        CHECK(g_out[0] == Approx(g_ref[0]));
        CHECK(g_out[1] == Approx(g_ref[1]));
    }
}

TEST_CASE("sims_grad — compute_sims_grad_dWchain on all-failed list returns "
          "true with empty needs_fd (no-op gradient contribution)",
          "[sims_grad][failed_masking][adjoint]")
{
    SimsGradScriptInput failed_a; failed_a.failed = true;
    SimsGradScriptInput failed_b; failed_b.failed = true;

    std::vector<double> dWdtheta(3, 1.0);
    std::vector<int>    theta_row_to_g_idx = {0};
    std::vector<int>    all_theta_indices = {0, 1, 2};
    std::vector<double> g_out(1, 42.0);
    std::vector<int>    needs_fd;

    REQUIRE(compute_sims_grad_dWchain({failed_a, failed_b}, dWdtheta,
                                       /*n_cols=*/3,
                                       theta_row_to_g_idx,
                                       all_theta_indices,
                                       g_out, needs_fd));
    /* No failed entry contributes — g_out stays at its incoming value,
     * needs_fd stays empty, and the helper does NOT signal failure. */
    CHECK(g_out[0] == 42.0);
    CHECK(needs_fd.empty());
}

TEST_CASE("sims_grad — compute_observables_grad_dWchain skips failed=true "
          "entries (failed-entry masking, mirrors sims branch)",
          "[sims_grad][failed_masking][adjoint]")
{
    /* Good observable: cohesive-energy-shaped, L2 loss, fold (v-t)·dO/dW
     * into the chain over a 2-θ dW/dθ matrix. */
    const std::size_t n_cols = 3;
    ObservableGradInput good;
    good.value    = 0.6;
    good.target   = 0.5;
    good.weight   = 1.0;
    good.loss_fn  = LossFunction(LossFunction::Type::L2);
    good.dO_dW    = Eigen::VectorXd(static_cast<long>(n_cols));
    good.dO_dW << 0.1, 0.2, 0.3;

    ObservableGradInput bad;
    bad.failed = true;

    std::vector<double> dWdtheta(2 * n_cols, 0.0);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < n_cols; ++c)
            dWdtheta[r * n_cols + c] = 0.5 * static_cast<double>(r + 1)
                                        + 0.1 * static_cast<double>(c);
    std::vector<int> theta_row_to_g_idx = {0, 1};

    /* Reference: good alone. */
    std::vector<double> g_ref(2, 0.0);
    {
        std::vector<int> nf;
        REQUIRE(compute_observables_grad_dWchain({good}, dWdtheta, n_cols,
                                                  theta_row_to_g_idx,
                                                  /*all_theta_indices=*/{},
                                                  g_ref, nf));
        CHECK(nf.empty());
    }

    /* With failed mixed in (any order), the masked result matches. */
    for (auto order : std::vector<std::vector<ObservableGradInput>>{
            {bad, good}, {good, bad}, {bad, good, bad}}) {
        std::vector<double> g_out(2, 0.0);
        std::vector<int>    needs_fd;
        REQUIRE(compute_observables_grad_dWchain(order, dWdtheta, n_cols,
                                                  theta_row_to_g_idx,
                                                  /*all_theta_indices=*/{},
                                                  g_out, needs_fd));
        CHECK(needs_fd.empty());
        CHECK(g_out[0] == Approx(g_ref[0]));
        CHECK(g_out[1] == Approx(g_ref[1]));
    }
}

/* The loss kinds loss_grad_wrt_observed differentiates are the ones
 * loss_has_derivative_wrt_observed reports: the startup rule for GRAD_STRATEGY
 * ANALYTICAL (analytical_loss_term_blocker) refuses the others. */
TEST_CASE("sims_grad — loss_has_derivative_wrt_observed agrees with "
          "loss_grad_wrt_observed",
          "[sims_grad][pc_observable][analytical_gradient]")
{
    for (auto t : {LossFunction::Type::L1, LossFunction::Type::L2,
                   LossFunction::Type::LOG_COSH, LossFunction::Type::HUBER,
                   LossFunction::Type::TUKEY, LossFunction::Type::MAPE,
                   LossFunction::Type::RMSLE})
    {
        LossFunction fn(t);
        INFO(LossFunction::to_string(t));
        double g = 0.0;
        CHECK(loss_has_derivative_wrt_observed(fn) ==
              loss_grad_wrt_observed(fn, 1.0, 0.5, 1.0, g));
    }
    CHECK_FALSE(loss_has_derivative_wrt_observed(
        LossFunction(LossFunction::Type::MAPE)));
    CHECK_FALSE(loss_has_derivative_wrt_observed(
        LossFunction(LossFunction::Type::RMSLE)));
    CHECK(loss_has_derivative_wrt_observed(
        LossFunction(LossFunction::Type::L2)));
}

/* compute_observables_grad_dWchain folds each PC_OBSERVABLE term into the
 * gradient as dL/dO * (dO/dW . dW/dtheta + dO/dtheta|_W):
 *   - a constant value (ObservableGradInput::constant, the fixed penalty of
 *     an unconverged relaxation) contributes nothing and needs no finite
 *     differences;
 *   - a value without dO/dW (the force-scaled penalty of an unconverged
 *     relaxation) has no analytical derivative: every theta goes to
 *     finite differences, and g is left as it was;
 *   - dO_dtheta_explicit, the value's own dependence on theta at fixed
 *     weights, adds dL/dO * dO/dtheta on the theta rows;
 *   - an explicit vector that does not match the theta rows is a wiring
 *     error. */
namespace {

struct ObsChainFixture {
    static constexpr std::size_t n_cols = 3;
    std::vector<double> dWdtheta = std::vector<double>(2 * n_cols, 0.0);
    std::vector<int>    theta_row_to_g_idx = {0, 1};
    std::vector<int>    all_theta = {0, 1};

    ObsChainFixture()
    {
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t c = 0; c < n_cols; ++c)
                dWdtheta[r * n_cols + c] = 0.5 * static_cast<double>(r + 1)
                                            + 0.1 * static_cast<double>(c);
    }

    static ObservableGradInput converged()
    {
        ObservableGradInput in;
        in.value   = 0.6;
        in.target  = 0.5;
        in.weight  = 2.0;
        in.loss_fn = LossFunction(LossFunction::Type::L2);
        in.dO_dW   = Eigen::VectorXd(static_cast<long>(n_cols));
        in.dO_dW << 0.1, 0.2, 0.3;
        return in;
    }

    /* dL/dO * dO/dW . dW/dtheta_r for `in`, by hand (L2: dL/dO = 2w(v-t)). */
    double chained(const ObservableGradInput& in, std::size_t r) const
    {
        const double dL_dO = 2.0 * in.weight * (in.value - in.target);
        double s = 0.0;
        for (std::size_t c = 0; c < n_cols; ++c)
            s += in.dO_dW(static_cast<long>(c)) * dWdtheta[r * n_cols + c];
        return dL_dO * s;
    }
};

}  // namespace

TEST_CASE("sims_grad — compute_observables_grad_dWchain: a constant value "
          "contributes no gradient and needs no finite differences",
          "[sims_grad][pc_observable][analytical_gradient]")
{
    ObsChainFixture f;
    ObservableGradInput penalty;       /* --unconverged const: no dO/dW */
    penalty.value    = 100.0;
    penalty.target   = 8.1;
    penalty.weight   = 800.0;
    penalty.loss_fn  = LossFunction(LossFunction::Type::L2);
    penalty.constant = true;

    std::vector<double> g = {1.5, -2.5};
    std::vector<int>    needs_fd;
    REQUIRE(compute_observables_grad_dWchain({penalty}, f.dWdtheta, f.n_cols,
                                              f.theta_row_to_g_idx, f.all_theta,
                                              g, needs_fd));
    CHECK(needs_fd.empty());
    CHECK(g[0] == 1.5);
    CHECK(g[1] == -2.5);

    /* Next to a converged term the result is that term's alone. */
    const ObservableGradInput good = ObsChainFixture::converged();
    std::vector<double> g2(2, 0.0);
    REQUIRE(compute_observables_grad_dWchain({penalty, good}, f.dWdtheta,
                                              f.n_cols, f.theta_row_to_g_idx,
                                              f.all_theta, g2, needs_fd));
    CHECK(needs_fd.empty());
    CHECK(g2[0] == Approx(f.chained(good, 0)));
    CHECK(g2[1] == Approx(f.chained(good, 1)));
}

TEST_CASE("sims_grad — compute_observables_grad_dWchain: a value without "
          "dO/dW sends every theta to finite differences",
          "[sims_grad][pc_observable][analytical_gradient]")
{
    ObsChainFixture f;
    ObservableGradInput force_scaled;  /* target + slope * fmax */
    force_scaled.value   = 8.1 + 10.0 * 0.4;
    force_scaled.target  = 8.1;
    force_scaled.weight  = 800.0;
    force_scaled.loss_fn = LossFunction(LossFunction::Type::L2);

    for (auto order : std::vector<std::vector<ObservableGradInput>>{
            {force_scaled}, {ObsChainFixture::converged(), force_scaled}}) {
        std::vector<double> g = {1.5, -2.5};
        std::vector<int>    needs_fd;
        CHECK_FALSE(compute_observables_grad_dWchain(
            order, f.dWdtheta, f.n_cols, f.theta_row_to_g_idx, f.all_theta, g,
            needs_fd));
        CHECK(needs_fd == f.all_theta);
        /* All or nothing: the converged term is not half-applied. */
        CHECK(g[0] == 1.5);
        CHECK(g[1] == -2.5);
    }
}

TEST_CASE("sims_grad — compute_observables_grad_dWchain adds the value's "
          "own dependence on theta at fixed weights",
          "[sims_grad][pc_observable][analytical_gradient]")
{
    ObsChainFixture f;
    ObservableGradInput in = ObsChainFixture::converged();
    in.dO_dtheta_explicit = {0.7, -0.3};
    const double dL_dO = 2.0 * in.weight * (in.value - in.target);

    std::vector<double> g(2, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(compute_observables_grad_dWchain({in}, f.dWdtheta, f.n_cols,
                                              f.theta_row_to_g_idx, f.all_theta,
                                              g, needs_fd));
    CHECK(needs_fd.empty());
    CHECK(g[0] == Approx(f.chained(in, 0) + dL_dO * 0.7));
    CHECK(g[1] == Approx(f.chained(in, 1) + dL_dO * -0.3));

    /* A theta row mapped to no gradient slot (-1) gets neither term. */
    std::vector<double> g2(2, 0.0);
    REQUIRE(compute_observables_grad_dWchain({in}, f.dWdtheta, f.n_cols,
                                              {0, -1}, f.all_theta, g2,
                                              needs_fd));
    CHECK(g2[0] == Approx(f.chained(in, 0) + dL_dO * 0.7));
    CHECK(g2[1] == 0.0);
}

TEST_CASE("sims_grad — compute_observables_grad_dWchain rejects an explicit "
          "dO/dtheta that does not match the theta rows",
          "[sims_grad][pc_observable][analytical_gradient]")
{
    ObsChainFixture f;
    ObservableGradInput in = ObsChainFixture::converged();
    in.dO_dtheta_explicit = {0.7, -0.3, 0.1};
    std::vector<double> g(2, 0.0);
    std::vector<int>    needs_fd;
    CHECK_THROWS_WITH(
        compute_observables_grad_dWchain({in}, f.dWdtheta, f.n_cols,
                                         f.theta_row_to_g_idx, f.all_theta, g,
                                         needs_fd),
        Catch::Contains("dO_dtheta_explicit size != number of theta rows"));
}

TEST_CASE("sims_grad — IFT chain rule cross-check via central diff on W "
          "for a non-trivial λ (lattice-constant-style)",
          "[sims_grad][adjoint]")
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

// ----------------------------------------------------------------
// Box-DOF extension: apply_ift_chain_kernel now optionally consumes
// dvirial_dW + box_dofs to fold λ_box rows into dO/dW. Verify the new
// path against hand-computed expectations and confirm the old (atom-
// only) signature still works byte-for-byte unchanged.

TEST_CASE("sims_grad — IFT kernel box-DOF: λ_box · dvirial_dW accumulates",
          "[sims_grad][adjoint]")
{
    Synth s;
    constexpr int box_dofs = 1;       // iso strain only
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
// The direct theta term of the PC_LAMMPS chain
// ---------------------------------------------------------------------------
//
// A theta that is a descriptor parameter changes the energy and the forces
// at the same weights. The chain's dO_k/dtheta_r is
//   (dg_k/dW + lambda_k . dF/dW) . dW/dtheta_r
//     + dg_k/dtheta_r + lambda_k . dF/dtheta_r      (H lambda_k = dg_k/dx),
// the last two at fixed x* and W: the direct term, which the provider gives
// as SimsGradScriptInput::dg_dtheta and dF_dtheta. The stub below is a
// relaxed state where everything is analytic, so the chain can be checked
// against central differences of the loss over theta, re-relaxing the stub.

namespace {

/* Two atoms and an iso box DOF: generalised coordinates q in R^7 (six atom
 * coordinates, then the strain of the box: lx = lx0 (1 + q_6)). Three
 * descriptors
 *   D_c(q; theta) = d_c(theta) + a_c(theta) . q + q . B_c q / 2,
 * d_c and a_c linear in two theta, B_c fixed and positive definite, and the
 * weights W(theta) quadratic. The energy E = sum_c W_c D_c relaxes to
 *   q*(theta) = -H^-1 sum_c W_c a_c(theta),   H = sum_c W_c B_c,
 * which is 0 at theta0 (a_2 balances a_0 and a_1 there), the state the chain
 * input describes. The generalised force is F = -dE/dq: its atom rows are
 * dF_dW's, its box row dvirial_dW's sign. */
struct DirectThetaStub {
    static constexpr int nlocal = 2;
    static constexpr int natoms = 2;
    static constexpr int n_desc = 3;
    static constexpr int n_theta = 2;
    static constexpr int n = 3 * nlocal + 1;
    const double lx0 = 3.3;
    Eigen::Vector2d theta0{0.4, -0.7};

    Eigen::VectorXd W0{n_desc}, d0{n_desc};
    Eigen::MatrixXd W1{n_desc, n_theta};   /* dW/dtheta at theta0 */
    Eigen::MatrixXd W2{n_desc, n_theta};   /* d2W/dtheta_r^2 */
    Eigen::MatrixXd Ed{n_desc, n_theta};   /* dd_c/dtheta_r */
    std::vector<Eigen::VectorXd> a0;       /* a_c(theta0) */
    std::vector<std::vector<Eigen::VectorXd>> U;   /* da_c/dtheta_r */
    std::vector<Eigen::MatrixXd> B;

    DirectThetaStub()
    {
        W0 << 1.3, -0.6, 0.9;
        d0 << -2.1, 0.8, 1.7;
        W1 << 0.25, -0.40, 0.10, 0.35, -0.20, 0.15;
        W2 << 0.05, -0.03, 0.02, 0.04, -0.01, 0.06;
        Ed << 0.7, -0.2, -0.5, 0.9, 0.3, 0.4;
        a0.assign(n_desc, Eigen::VectorXd::Zero(n));
        U.assign(n_desc, std::vector<Eigen::VectorXd>(n_theta,
                                                      Eigen::VectorXd::Zero(n)));
        B.assign(n_desc, Eigen::MatrixXd::Zero(n, n));
        for (int i = 0; i < n; ++i) {
            a0[0](i) = 0.30 + 0.05 * i;
            a0[1](i) = -0.20 + 0.07 * i * i - 0.1 * (i % 3);
            for (int c = 0; c < n_desc; ++c)
                for (int r = 0; r < n_theta; ++r)
                    U[c][r](i) = 0.1 * (c + 1) * (r == 0 ? 1.0 : -0.6) +
                                 0.03 * i * (r + 1) - 0.02 * c * i;
        }
        a0[2] = -(W0(0) * a0[0] + W0(1) * a0[1]) / W0(2);
        for (int c = 0; c < n_desc; ++c) {
            B[c] = Eigen::MatrixXd::Identity(n, n) * (2.0 + 0.5 * c);
            for (int i = 0; i + 1 < n; ++i)
                B[c](i, i + 1) = B[c](i + 1, i) = 0.1 * (c + 1);
        }
        /* W0(1) < 0, yet H = sum_c W_c B_c is positive definite (its
         * diagonal 3.8, its off-diagonal 0.28 at theta0). */
    }

    Eigen::VectorXd W(const Eigen::Vector2d& th) const
    {
        const Eigen::Vector2d dt = th - theta0;
        return W0 + W1 * dt + 0.5 * (W2 * dt.cwiseProduct(dt));
    }
    Eigen::VectorXd a(int c, const Eigen::Vector2d& th) const
    {
        const Eigen::Vector2d dt = th - theta0;
        return a0[c] + U[c][0] * dt(0) + U[c][1] * dt(1);
    }
    double d(int c, const Eigen::Vector2d& th) const
    {
        return d0(c) + Ed.row(c).dot(th - theta0);
    }

    /* The relaxed state at theta: q* and the energy there. */
    std::pair<Eigen::VectorXd, double> relax(const Eigen::Vector2d& th) const
    {
        const Eigen::VectorXd w = W(th);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        for (int c = 0; c < n_desc; ++c) {
            H += w(c) * B[c];
            rhs -= w(c) * a(c, th);
        }
        const Eigen::VectorXd q = H.ldlt().solve(rhs);
        double E = 0.0;
        for (int c = 0; c < n_desc; ++c)
            E += w(c) * (d(c, th) + a(c, th).dot(q) + 0.5 * q.dot(B[c] * q));
        return {q, E};
    }
    double cohesive(const Eigen::Vector2d& th) const
    {
        return relax(th).second / natoms;
    }
    double lx(const Eigen::Vector2d& th) const
    {
        return lx0 * (1.0 + relax(th).first(n - 1));
    }

    /* The snapshot at theta0, as pair_tadah and the Hessian assembler would
     * give it, held for the input's pointers. */
    tadah::core::aeds_type2 aeds{n_desc, nlocal};
    std::vector<double> dF_dW, dvirial_dW;
    Eigen::MatrixXd H0;
    void snapshot()
    {
        for (int c = 0; c < n_desc; ++c)
            for (int at = 0; at < nlocal; ++at)
                aeds(c, at) = d0(c) / nlocal;   /* sum: D_c(q* = 0) */
        dF_dW.assign(static_cast<std::size_t>(3 * nlocal * n_desc), 0.0);
        dvirial_dW.assign(n_desc, 0.0);
        H0 = Eigen::MatrixXd::Zero(n, n);
        for (int c = 0; c < n_desc; ++c) {
            for (int i = 0; i < 3 * nlocal; ++i)
                dF_dW[static_cast<std::size_t>(i * n_desc + c)] = -a0[c](i);
            dvirial_dW[c] = -a0[c](n - 1);
            H0 += W0(c) * B[c];
        }
    }

    struct Term { std::string expr; double target, weight; };
    double value(const Term& t, const Eigen::Vector2d& th) const
    {
        return t.expr == "lx" ? lx(th) : cohesive(th);
    }

    /* The chain input at theta0 for `terms` (L2 losses), with the direct
     * term when `direct`. */
    SimsGradScriptInput input(const std::vector<Term>& terms, bool direct)
    {
        snapshot();
        SimsGradScriptInput in;
        in.ctx.nlocal = nlocal;
        in.ctx.natoms = natoms;
        in.ctx.n_desc = n_desc;
        in.ctx.box_dofs = 1;
        in.ctx.aeds = &aeds;
        in.ctx.lx = in.ctx.ly = in.ctx.lz = lx0;
        in.H = H0;
        in.dF_dW = dF_dW.data();
        in.dvirial_dW = dvirial_dW.data();
        in.box_dofs = 1;
        for (const auto& t : terms) {
            in.expressions.push_back(t.expr);
            in.v_observed.push_back(value(t, theta0));
            in.t_targets.push_back(t.target);
            in.w_weights.push_back(t.weight);
            in.loss_fns.push_back(LossFunction(LossFunction::Type::L2));
        }
        if (direct) {
            in.direct_theta = true;
            in.dg_dtheta.assign(terms.size() * n_theta, 0.0);
            for (std::size_t k = 0; k < terms.size(); ++k)
                for (int r = 0; r < n_theta; ++r)
                    in.dg_dtheta[k * n_theta + r] =
                        terms[k].expr == "lx" ? 0.0
                                              : W0.dot(Ed.col(r)) / natoms;
            in.dF_dtheta.assign(static_cast<std::size_t>(n_theta * n), 0.0);
            for (int r = 0; r < n_theta; ++r)
                for (int c = 0; c < n_desc; ++c)
                    for (int i = 0; i < n; ++i)
                        in.dF_dtheta[static_cast<std::size_t>(r * n + i)] -=
                            W0(c) * U[c][r](i);
        }
        return in;
    }
    /* dW/dtheta, row-major n_theta x n_desc. */
    std::vector<double> dWdtheta() const
    {
        std::vector<double> m(n_theta * n_desc);
        for (int r = 0; r < n_theta; ++r)
            for (int c = 0; c < n_desc; ++c)
                m[r * n_desc + c] = W1(c, r);
        return m;
    }
    /* dL/dtheta_r by central differences of the loss, re-relaxing. */
    double fd(const std::vector<Term>& terms, int r, double h) const
    {
        const auto loss = [&](const Eigen::Vector2d& th) {
            double L = 0.0;
            for (const auto& t : terms)
                L += LossFunction(LossFunction::Type::L2)
                         .loss(value(t, th), t.target, t.weight);
            return L;
        };
        Eigen::Vector2d p = theta0, m = theta0;
        p(r) += h;
        m(r) -= h;
        return (loss(p) - loss(m)) / (2.0 * h);
    }
};

}  // namespace

TEST_CASE("sims_grad — compute_sims_grad_dWchain adds the direct theta term "
          "and matches central differences of the loss where theta enters the "
          "energy and the forces at fixed weights",
          "[sims_grad][adjoint][direct_theta]")
{
    /* Pins: for the cohesive energy alone (lambda = 0: its direct term is
     * dE/dtheta / N at fixed W), the lattice constant alone (lambda != 0:
     * lambda . dF/dtheta) and both in one script, g_out of each theta row --
     * mapped to the swapped slots {1, 0} -- equals the central difference
     * of the loss (h = 1e-5; the chain is exact at theta0, the difference
     * has an O(h^2) error: measured below 1e-9 relative, the tolerance is
     * 1e-7), and the direct term is a sizeable part of it (the chain
     * through W alone is off by more than 10 %).
     * Before the fix: SimsGradScriptInput has no direct term (dg_dtheta,
     * dF_dtheta, direct_theta): the case does not compile; the chain gave
     * the W path only, which misses the central differences by the
     * direct term. */
    using Term = DirectThetaStub::Term;
    const std::vector<std::pair<std::string, std::vector<Term>>> sets = {
        {"cohesive energy", {{"c_eatoms/atoms", -1.5, 1.0}}},
        {"lattice constant", {{"lx", 3.2, 2.0}}},
        {"both", {{"c_eatoms/atoms", -1.5, 1.0}, {"lx", 3.2, 2.0}}}};
    for (const auto& set : sets) {
        DYNAMIC_SECTION(set.first) {
            DirectThetaStub stub;
            const std::vector<double> dWdtheta = stub.dWdtheta();
            const std::vector<int> theta_row_to_g_idx = {1, 0};

            std::vector<double> g(2, 0.0);
            std::vector<int>    needs_fd;
            REQUIRE(compute_sims_grad_dWchain(
                {stub.input(set.second, /*direct=*/true)}, dWdtheta,
                DirectThetaStub::n_desc, theta_row_to_g_idx, {0, 1}, g,
                needs_fd));
            CHECK(needs_fd.empty());

            /* The W path alone: the direct term given as zero. */
            SimsGradScriptInput w_only = stub.input(set.second, true);
            std::fill(w_only.dg_dtheta.begin(), w_only.dg_dtheta.end(), 0.0);
            std::fill(w_only.dF_dtheta.begin(), w_only.dF_dtheta.end(), 0.0);
            std::vector<double> g_w(2, 0.0);
            REQUIRE(compute_sims_grad_dWchain({w_only}, dWdtheta,
                                              DirectThetaStub::n_desc,
                                              theta_row_to_g_idx, {0, 1}, g_w,
                                              needs_fd));

            for (int r = 0; r < DirectThetaStub::n_theta; ++r) {
                const double fd = stub.fd(set.second, r, 1e-5);
                const double chain = g[theta_row_to_g_idx[r]];
                const double w_path = g_w[theta_row_to_g_idx[r]];
                INFO("theta row " << r << ": chain " << chain << ", FD " << fd
                     << ", W path " << w_path);
                CHECK(std::abs(chain - fd) <= 1e-7 * std::abs(fd));
                CHECK(std::abs(w_path - fd) > 0.1 * std::abs(fd));
            }
        }
    }
}

TEST_CASE("sims_grad — compute_sims_grad_dWchain routes every theta to "
          "finite differences for a script without the direct theta term",
          "[sims_grad][adjoint][direct_theta]")
{
    /* The provider must say what theta does at fixed W (direct_theta); a
     * script with observables that does not is a structural failure, like
     * an unknown expression: every theta to needs_fd, false, g_out
     * untouched. A script without observables has nothing to
     * differentiate and needs none.
     * Pins: false, needs_fd == {0, 1, 2}, g_out {5, -3} unchanged; with the
     * observables cleared, true and g_out unchanged.
     * Before the fix: true, needs_fd empty, and g_out took the W path alone
     * (the chain had no direct term to ask for). */
    DirectThetaStub stub;
    SimsGradScriptInput in =
        stub.input({{"c_eatoms/atoms", -1.5, 1.0}, {"lx", 3.2, 2.0}},
                   /*direct=*/false);
    std::vector<double> g = {5.0, -3.0};
    std::vector<int>    needs_fd;
    CHECK_FALSE(compute_sims_grad_dWchain({in}, stub.dWdtheta(),
                                          DirectThetaStub::n_desc, {0, 1},
                                          {0, 1, 2}, g, needs_fd));
    CHECK(needs_fd == std::vector<int>{0, 1, 2});
    CHECK(g == std::vector<double>{5.0, -3.0});

    in.expressions.clear();
    in.v_observed.clear();
    in.t_targets.clear();
    in.w_weights.clear();
    in.loss_fns.clear();
    needs_fd.clear();
    CHECK(compute_sims_grad_dWchain({in}, stub.dWdtheta(),
                                    DirectThetaStub::n_desc, {0, 1}, {0, 1, 2},
                                    g, needs_fd));
    CHECK(needs_fd.empty());
    CHECK(g == std::vector<double>{5.0, -3.0});
}

TEST_CASE("sims_grad — compute_sims_grad_dWchain rejects a direct theta term "
          "that does not match the observables, the theta rows or the "
          "generalised coordinates",
          "[sims_grad][adjoint][direct_theta]")
{
    /* Pins: a dg_dtheta of other than k x n_theta values or a dF_dtheta of
     * other than n_theta x (3 nlocal + box_dofs) throws, naming the field.
     * Before the fix: the fields do not exist (the case does not
     * compile). */
    DirectThetaStub stub;
    const std::vector<DirectThetaStub::Term> terms = {
        {"c_eatoms/atoms", -1.5, 1.0}, {"lx", 3.2, 2.0}};
    std::vector<double> g(2, 0.0);
    std::vector<int>    needs_fd;
    SimsGradScriptInput in = stub.input(terms, true);
    in.dg_dtheta.pop_back();
    CHECK_THROWS_WITH(compute_sims_grad_dWchain({in}, stub.dWdtheta(),
                                                DirectThetaStub::n_desc, {0, 1},
                                                {0, 1}, g, needs_fd),
                      Catch::Contains("dg_dtheta"));
    in = stub.input(terms, true);
    in.dF_dtheta.push_back(0.0);
    CHECK_THROWS_WITH(compute_sims_grad_dWchain({in}, stub.dWdtheta(),
                                                DirectThetaStub::n_desc, {0, 1},
                                                {0, 1}, g, needs_fd),
                      Catch::Contains("dF_dtheta"));
}

// ---------------------------------------------------------------------------
// Script-side helpers
// ---------------------------------------------------------------------------
//
// Two pure-string utilities used inside HPO_Host::analytical_gradient's
// SimsGradProvider lambda. Tested LAMMPS-free under [sims_grad].

TEST_CASE("sims_grad_lammps — detect_box_dofs_from_script picks iso vs aniso "
          "from `fix box/relax` lines",
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "[sims_grad][adjoint]")
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
          "unsupported styles ignored",
          "[sims_grad][adjoint]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    LammpsSim ls;
    ls.script_lines = {
        "# variable v equal lx",       /* commented out — NOT a def */
        "variable v string foo",       /* string but literal — skipped */
        "variable v equal lx",         /* the real one */
        "variable v atom v_lx + 1",    /* atom-style — skipped */
    };
    ls.n_outvar  = {"v"};
    ls.n_varloss = {"v"};
    ls.t_varloss = {0.0};
    ls.w_varloss = {1.0};
    ls.i_varloss = {0};

    const auto exprs = extract_varloss_expressions(ls);
    REQUIRE(exprs.size() == 1);
    /* Commented + literal-string + atom styles all skipped; only the
     * `equal` line populates the map. */
    CHECK(exprs[0] == "lx");
}

/* `variable v string $(<expr>)` is LAMMPS' immediate-eval form: <expr>
 * is computed at definition time and the result stored as a literal
 * string. When the `variable` line appears AFTER the relaxation (the
 * canonical pattern: `... minimize ... variable e_per string $(...)
 * ... run 0`), the value reflects the post-minimize state — identical
 * to what `variable v equal $(<expr>)` would return on access. We
 * accept this form so users who prefer `string` (no re-evaluation cost)
 * stay on the analytical path. */
TEST_CASE("sims_grad_lammps — extract_varloss_expressions: "
          "`string $(...)` immediate-eval honoured",
          "[sims_grad][adjoint]")
{
    using tadah::hpo::extract_varloss_expressions;
    using tadah::hpo::LammpsSim;

    /* Three scripts in one case to keep the surface compact:
     *   (a) pure `string $(...)`
     *   (b) `string $(...)` redefined to `equal <expr>` (last-wins
     *       across styles)
     *   (c) literal `string foo` (no `$(...)`) ignored even when no
     *       other definition exists. */
    SECTION("string $(...) parsed identically to equal $(...)") {
        LammpsSim ls;
        ls.script_lines = {
            "variable e_per string $(c_eatoms/atoms)",
            "variable my_lx string $(lx/4.0)",
            "run 0",
        };
        ls.n_outvar  = {"e_per", "my_lx"};
        ls.n_varloss = {"e_per", "my_lx"};
        ls.t_varloss = {0.0, 0.0};
        ls.w_varloss = {1.0, 1.0};
        ls.i_varloss = {0, 1};

        const auto exprs = extract_varloss_expressions(ls);
        REQUIRE(exprs.size() == 2);
        CHECK(exprs[0] == "$(c_eatoms/atoms)");
        CHECK(exprs[1] == "$(lx/4.0)");
    }

    SECTION("last definition wins across string/equal styles") {
        LammpsSim ls;
        ls.script_lines = {
            "variable v string $(lx)",
            "variable v equal ly",          /* last-wins regardless of style */
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

    SECTION("literal string (no `$(...)`) skipped — gradient zero "
            "isn't trustworthy on a free-form constant") {
        LammpsSim ls;
        ls.script_lines = {
            "variable v string some_literal_value",
            "variable other equal pe",
        };
        ls.n_outvar  = {"v", "other"};
        ls.n_varloss = {"v", "other"};
        ls.t_varloss = {0.0, 0.0};
        ls.w_varloss = {1.0, 1.0};
        ls.i_varloss = {0, 1};

        const auto exprs = extract_varloss_expressions(ls);
        REQUIRE(exprs.size() == 2);
        CHECK(exprs[0].empty());         /* literal v skipped */
        CHECK(exprs[1] == "pe");
    }
}

// ObservableHandle registry tests.
//
// LAMMPS-free by design: the registry consumes a pre-populated
// ObservableContext, so we synthesise the context with known aeds and
// box dimensions and verify the closed-form gradients match the
// underlying math. The LAMMPS-driven parity test lives in
// test_hpo_adjoint_pilot.cpp, alongside the Hessian/adjoint pieces.

#include "catch2/catch.hpp"

#include <tadah/core/aeds_type.h>
#include <tadah/hpo/observable_registry.h>

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace tadah::hpo;

ObservableContext make_ctx_with_aeds(tadah::core::aeds_type2* aeds,
                                     int nlocal, int natoms,
                                     int box_dofs = 0,
                                     double lx = 3.301, double ly = 3.301,
                                     double lz = 6.602)
{
    ObservableContext ctx;
    ctx.aeds     = aeds;
    ctx.nlocal   = nlocal;
    ctx.natoms   = natoms;
    ctx.n_desc   = static_cast<int>(aeds->rows());
    ctx.box_dofs = box_dofs;
    ctx.lx = lx;  ctx.ly = ly;  ctx.lz = lz;
    return ctx;
}

}  // namespace

TEST_CASE("registry — cohesive E per atom: dg_dx is zero, dg_dW is "
          "column-sum / natoms",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(/*dim=*/4, /*natoms=*/3);
    /* Fill aeds(c, a) = 100*a + c — gives unique values to detect
     * stride/transpose errors. */
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t c = 0; c < 4; ++c)
            aeds(c, a) = 100.0 * a + c;

    ObservableContext ctx = make_ctx_with_aeds(&aeds, /*nlocal=*/3, /*natoms=*/3,
                                               /*box_dofs=*/1);

    ObservableHandle h;
    REQUIRE(try_register("c_eatoms/atoms", h));
    CHECK(h.kind == "cohesive_e_per_atom");
    CHECK(h.dg_dx_is_zero);

    Eigen::VectorXd dx;
    h.dg_dx(ctx, dx);
    REQUIRE(dx.size() == 3 * 3 + 1);
    CHECK(dx.cwiseAbs().maxCoeff() == 0.0);

    Eigen::VectorXd dW;
    h.dg_dW(ctx, dW);
    REQUIRE(dW.size() == 4);
    /* Expected: dW[c] = (sum_a aeds(c, a)) / natoms
     *               = ((0+100+200) + (1+101+201) + ...) / 3
     *               = (300 + c*natoms) / natoms ... actually:
     * sum_a aeds(c, a) = 100*(0+1+2) + 3*c = 300 + 3*c
     * dW[c] = (300 + 3*c) / 3 = 100 + c                                  */
    for (int c = 0; c < 4; ++c) {
        CHECK(dW(c) == Approx(100.0 + c));
    }
}

TEST_CASE("registry — pe (total potential energy) is column-sum, no division",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(/*dim=*/3, /*natoms=*/2);
    aeds(0, 0) = 1.0;  aeds(1, 0) = 2.0;  aeds(2, 0) = 3.0;
    aeds(0, 1) = 4.0;  aeds(1, 1) = 5.0;  aeds(2, 1) = 6.0;

    ObservableContext ctx = make_ctx_with_aeds(&aeds, 2, 2);

    ObservableHandle h;
    REQUIRE(try_register("pe", h));
    CHECK(h.kind == "pe");
    CHECK(h.dg_dx_is_zero);

    Eigen::VectorXd dW;
    h.dg_dW(ctx, dW);
    REQUIRE(dW.size() == 3);
    CHECK(dW(0) == Approx(5.0));   /* 1 + 4 */
    CHECK(dW(1) == Approx(7.0));   /* 2 + 5 */
    CHECK(dW(2) == Approx(9.0));   /* 3 + 6 */
}

TEST_CASE("registry — lx, ly, lz primitive forms place ∂g/∂ε on the "
          "correct box DOF",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(/*dim=*/2, /*natoms=*/1);  /* unused */

    SECTION("iso box DOF — single entry at row 3*nlocal") {
        ObservableContext ctx = make_ctx_with_aeds(&aeds, /*nlocal=*/1, 1,
                                                    /*box_dofs=*/1,
                                                    /*lx=*/3.5, 4.0, 5.0);

        for (const auto& [expr, axis_L] :
             std::vector<std::pair<std::string, double>>{
                 {"lx", 3.5}, {"ly", 4.0}, {"lz", 5.0}})
        {
            ObservableHandle h;
            REQUIRE(try_register(expr, h));
            Eigen::VectorXd dx;
            h.dg_dx(ctx, dx);
            REQUIRE(dx.size() == 3 + 1);
            /* atom block zero, iso DOF carries L. */
            for (int i = 0; i < 3; ++i) CHECK(dx(i) == 0.0);
            CHECK(dx(3) == Approx(axis_L));
        }
    }

    SECTION("aniso (6 Voigt) — entry on axis index (0/1/2)") {
        ObservableContext ctx = make_ctx_with_aeds(&aeds, 1, 1, /*box_dofs=*/6,
                                                    3.5, 4.0, 5.0);
        for (const auto& [expr, axis, L] :
             std::vector<std::tuple<std::string, int, double>>{
                 {"lx", 0, 3.5}, {"ly", 1, 4.0}, {"lz", 2, 5.0}})
        {
            ObservableHandle h;
            REQUIRE(try_register(expr, h));
            Eigen::VectorXd dx;
            h.dg_dx(ctx, dx);
            REQUIRE(dx.size() == 3 + 6);
            for (int i = 0; i < 9; ++i) {
                const double expected = (i == 3 + axis) ? L : 0.0;
                CHECK(dx(i) == Approx(expected));
            }
        }
    }

    SECTION("dg_dW is the zero vector of size n_desc") {
        ObservableContext ctx = make_ctx_with_aeds(&aeds, 1, 1, 1);
        ObservableHandle h;
        REQUIRE(try_register("lx", h));
        Eigen::VectorXd dW;
        h.dg_dW(ctx, dW);
        REQUIRE(dW.size() == 2);
        CHECK(dW.cwiseAbs().maxCoeff() == 0.0);
    }
}

TEST_CASE("registry — lx/<const> scales the box-DOF entry by 1/divisor",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(2, 1);
    ObservableContext ctx = make_ctx_with_aeds(&aeds, 1, 1, /*box_dofs=*/1,
                                                3.5, 4.0, 5.0);

    ObservableHandle h;
    REQUIRE(try_register("lx/4.0", h));
    Eigen::VectorXd dx;
    h.dg_dx(ctx, dx);
    REQUIRE(dx.size() == 3 + 1);
    CHECK(dx(3) == Approx(3.5 / 4.0));

    /* `$(lx/4.0)` immediate-eval wrapper is stripped before matching. */
    ObservableHandle h2;
    REQUIRE(try_register("$(lx/4.0)", h2));
    Eigen::VectorXd dx2;
    h2.dg_dx(ctx, dx2);
    CHECK(dx2(3) == Approx(3.5 / 4.0));
}

TEST_CASE("registry — vol on iso box DOF returns 3·V",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(2, 1);
    ObservableContext ctx = make_ctx_with_aeds(&aeds, /*nlocal=*/1, 1,
                                                /*box_dofs=*/1,
                                                3.0, 4.0, 5.0);
    ObservableHandle h;
    REQUIRE(try_register("vol", h));
    Eigen::VectorXd dx;
    h.dg_dx(ctx, dx);
    REQUIRE(dx.size() == 3 + 1);
    const double V = 3.0 * 4.0 * 5.0;
    CHECK(dx(3) == Approx(3.0 * V));
}

TEST_CASE("registry — vol on aniso (6 Voigt) lights up Voigt 0/1/2 each "
          "with magnitude V; shears 3/4/5 stay zero",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(2, 1);
    ObservableContext ctx = make_ctx_with_aeds(&aeds, 1, 1, /*box_dofs=*/6,
                                                3.0, 4.0, 5.0);
    ObservableHandle h;
    REQUIRE(try_register("vol", h));
    Eigen::VectorXd dx;
    h.dg_dx(ctx, dx);
    REQUIRE(dx.size() == 3 + 6);
    const double V = 60.0;
    for (int i = 0; i < 3; ++i) CHECK(dx(3 + i) == Approx(V));
    for (int i = 3; i < 6; ++i) CHECK(dx(3 + i) == Approx(0.0));
}

TEST_CASE("registry — atoms primitive: both gradients zero",
          "[observable_registry][adjoint]")
{
    tadah::core::aeds_type2 aeds(3, 2);
    ObservableContext ctx = make_ctx_with_aeds(&aeds, 2, 2, 1);
    ObservableHandle h;
    REQUIRE(try_register("atoms", h));
    CHECK(h.dg_dx_is_zero);
    Eigen::VectorXd dx, dW;
    h.dg_dx(ctx, dx);
    h.dg_dW(ctx, dW);
    CHECK(dx.cwiseAbs().maxCoeff() == 0.0);
    CHECK(dW.cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("registry — linear combination recurses and additively combines "
          "sub-gradients (sign-aware)",
          "[observable_registry][adjoint]")
{
    /* Build a known aeds; let the combo be (pe - 2·cohesive_per_atom) by
     * registering `pe - c_eatoms/atoms - c_eatoms/atoms`. */
    tadah::core::aeds_type2 aeds(/*dim=*/2, /*natoms=*/2);
    aeds(0, 0) = 10.0;  aeds(1, 0) = 20.0;
    aeds(0, 1) = 30.0;  aeds(1, 1) = 40.0;

    ObservableContext ctx = make_ctx_with_aeds(&aeds, 2, 2, /*box_dofs=*/0);

    ObservableHandle h;
    REQUIRE(try_register("pe - c_eatoms/atoms", h));
    Eigen::VectorXd dW;
    h.dg_dW(ctx, dW);
    REQUIRE(dW.size() == 2);
    /* pe.dg_dW    = (40, 60).  cohesive.dg_dW = (20, 30).  diff = (20, 30). */
    CHECK(dW(0) == Approx(20.0));
    CHECK(dW(1) == Approx(30.0));

    /* `a + b` and `a - b` should produce opposite-sign rhs contributions. */
    ObservableHandle hp;
    REQUIRE(try_register("pe + c_eatoms/atoms", hp));
    Eigen::VectorXd dWp;
    hp.dg_dW(ctx, dWp);
    CHECK(dWp(0) == Approx(60.0));
    CHECK(dWp(1) == Approx(90.0));
}

TEST_CASE("registry — variable resolver routes v_<name> back through "
          "try_register on the resolved expression",
          "[observable_registry][adjoint]")
{
    std::unordered_map<std::string, std::string> table = {
        {"Ebcc_per", "c_eatoms/atoms"},
        {"a0",       "lx/4.0"},
    };
    auto resolver = [&](const std::string& name) -> std::string {
        auto it = table.find(name);
        return (it == table.end()) ? std::string{} : it->second;
    };

    tadah::core::aeds_type2 aeds(2, 2);
    aeds(0, 0) = 1.0;  aeds(1, 0) = 2.0;
    aeds(0, 1) = 3.0;  aeds(1, 1) = 4.0;

    ObservableContext ctx = make_ctx_with_aeds(&aeds, 2, 2, /*box_dofs=*/1,
                                                3.5, 4.0, 5.0);

    ObservableHandle h;
    REQUIRE(try_register("v_Ebcc_per", resolver, h));
    CHECK(h.kind == "cohesive_e_per_atom");

    Eigen::VectorXd dW;
    h.dg_dW(ctx, dW);
    /* (1+3)/2 = 2;  (2+4)/2 = 3. */
    CHECK(dW(0) == Approx(2.0));
    CHECK(dW(1) == Approx(3.0));

    /* Without a resolver, v_<name> fails to register. */
    ObservableHandle h2;
    CHECK_FALSE(try_register("v_Ebcc_per", h2));
}

TEST_CASE("registry — rejects expressions we don't claim to handle "
          "(routes the θ index to needs_fd)",
          "[observable_registry][adjoint]")
{
    ObservableHandle h;
    /* Generic division between two variables — not lattice/<const>. */
    CHECK_FALSE(try_register("v_a/v_b + v_c", h));
    /* Triple-arg multiplication — no product registry. */
    CHECK_FALSE(try_register("lx*ly*lz", h));
    /* Unknown compute name. */
    CHECK_FALSE(try_register("c_unknown/atoms", h));
    /* lz/0 — zero divisor rejected by lattice parser. */
    CHECK_FALSE(try_register("lx/0", h));
    /* Garbage trailing characters after numeric divisor. */
    CHECK_FALSE(try_register("lx/4.0junk", h));
    /* Empty string. */
    CHECK_FALSE(try_register("", h));
}

TEST_CASE("registry — v_<name> resolver missing entries fail closed",
          "[observable_registry][adjoint]")
{
    auto resolver = [](const std::string&) { return std::string{}; };
    ObservableHandle h;
    CHECK_FALSE(try_register("v_doesnt_exist", resolver, h));
}

TEST_CASE("registry — pe-style FD cross-check on a synthetic aeds: "
          "central-diff in W matches analytical dg_dW",
          "[observable_registry][adjoint]")
{
    /* Synthesise a tiny linear model and aeds. The "scalar observable"
     * O(W) = pe(W) = sum_a (W · D_a). Pick W away from zero and verify
     * the central diff of O(W ± h·e_c) equals dg_dW[c] = sum_a D_a[c]. */
    tadah::core::aeds_type2 aeds(/*dim=*/3, /*natoms=*/4);
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t c = 0; c < 3; ++c)
            aeds(c, a) = 0.5 + 0.1 * static_cast<double>(a)
                              - 0.2 * static_cast<double>(c);

    ObservableContext ctx = make_ctx_with_aeds(&aeds, 4, 4);

    ObservableHandle h;
    REQUIRE(try_register("pe", h));
    Eigen::VectorXd dW;
    h.dg_dW(ctx, dW);

    /* FD reference: pe = Σ_a Σ_c W_c · D_ac.  Pick W = (1.5, -0.7, 2.3). */
    Eigen::VectorXd W(3);  W << 1.5, -0.7, 2.3;
    auto pe = [&](const Eigen::VectorXd& w) {
        double e = 0.0;
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t c = 0; c < 3; ++c)
                e += w(static_cast<Eigen::Index>(c)) * aeds(c, a);
        return e;
    };
    const double h_fd = 1e-6;
    for (int c = 0; c < 3; ++c) {
        Eigen::VectorXd Wp = W, Wm = W;
        Wp(c) += h_fd;  Wm(c) -= h_fd;
        const double fd = (pe(Wp) - pe(Wm)) / (2.0 * h_fd);
        CHECK(dW(c) == Approx(fd).epsilon(1e-7));
    }
}

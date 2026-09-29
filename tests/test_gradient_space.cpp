/* GradientProvider chain rule: the host returns dL/dtheta in REAL space,
 * the optimiser iterates in INTERNAL coordinates -- ANALYTICAL/HYBRID
 * gradients must be pulled back through the search space (a LOG dim
 * needs g_u = theta * dL/dtheta; a normalised dim needs the width). */
#include "catch2/catch.hpp"
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

using namespace tadah::hpo;

namespace {

using DlibVec = dlib::matrix<double, 0, 1>;

/* f(theta) = a0*theta0 + a1*theta1 in REAL space. */
struct LinearHost {
    double operator()(const DlibVec& theta) const { return 2.0 * theta(0) + 3.0 * theta(1); }
};

OptimizerSpec spec_with_space(bool normalise, bool log_second)
{
    DimSpec a; a.key = a.label = "A"; a.lo = -5.0; a.hi = 5.0;
    DimSpec b; b.key = b.label = "B"; b.lo = 0.1;  b.hi = 10.0;
    if (log_second) b.tf.kind = ScaleKind::Log;
    ScalePolicy pol; pol.normalise = normalise;
    OptimizerSpec sp;
    sp.lib = "CERES"; sp.algo = "LBFGS";
    sp.bind_space(std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol)));
    return sp;
}

/* The host's analytical gradient: dL/dtheta = (2, 3), REAL space. */
bool analytical_real(const DlibVec&, DlibVec& g, std::vector<int>& needs_fd)
{
    g.set_size(2); g(0) = 2.0; g(1) = 3.0;
    needs_fd.clear();
    return true;
}

bool analytical_partial(const DlibVec&, DlibVec& g, std::vector<int>& needs_fd)
{
    g.set_size(2); g(0) = 2.0; g(1) = 0.0;
    needs_fd = {1};              /* dim 1 falls back to FD */
    return true;
}

}  // namespace

TEST_CASE("GradientProvider ANALYTICAL applies the search-space chain rule",
          "[HPO][grad_provider][search_space]")
{
    for (bool normalise : {false, true}) {
        auto sp = spec_with_space(normalise, /*log_second=*/true);
        sp.grad_strategy = GradStrategy::Analytical;
        LinearHost host;
        std::atomic<unsigned long> counter{0};
        auto gp = make_gradient_provider<DlibVec, void>(sp, host, &counter, 0,
                                                        &analytical_real);
        /* internal point for theta = (1.0, 2.0) */
        DlibVec theta; theta.set_size(2); theta(0) = 1.0; theta(1) = 2.0;
        const DlibVec x = sp.in(theta);
        DlibVec g;
        REQUIRE(gp.gradient(x, g));
        REQUIRE(g.size() == 2);
        CHECK(g(0) == Approx(2.0 * sp.space->dreal_dinternal(0, x(0))));
        CHECK(g(1) == Approx(3.0 * sp.space->dreal_dinternal(1, x(1))));
        /* the LOG dim's factor is theta itself (times the ln-width when normalised) */
        const double expect1 = 3.0 * 2.0 * (normalise ? std::log(10.0 / 0.1) : 1.0);
        CHECK(g(1) == Approx(expect1));
    }
}

TEST_CASE("GradientProvider HYBRID: analytical dims pulled back, FD dims untouched",
          "[HPO][grad_provider][search_space]")
{
    auto sp = spec_with_space(/*normalise=*/true, /*log_second=*/true);
    sp.grad_strategy = GradStrategy::Hybrid;
    sp.param["fd_step"] = 1e-6;
    LinearHost host;
    std::atomic<unsigned long> counter{0};
    auto gp = make_gradient_provider<DlibVec, void>(sp, host, &counter, 0,
                                                    &analytical_partial);
    DlibVec theta; theta.set_size(2); theta(0) = 1.0; theta(1) = 2.0;
    const DlibVec x = sp.in(theta);
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    /* dim 0: analytical 2.0 * dx/du (= width 10) */
    CHECK(g(0) == Approx(2.0 * sp.space->dreal_dinternal(0, x(0))));
    /* dim 1: central FD in internal coordinates of f = 3*theta1 with
     * theta1 = exp(t), t = ln(0.1) + u*ln(100): df/du = 3*theta1*ln(100) */
    CHECK(g(1) == Approx(3.0 * 2.0 * std::log(100.0)).epsilon(1e-4));
}

TEST_CASE("GradientProvider legacy ln mask (no space) still gets the chain rule",
          "[HPO][grad_provider][search_space]")
{
    OptimizerSpec sp;
    sp.lib = "CERES"; sp.algo = "LBFGS";
    sp.low = {-5.0, std::log(0.1)}; sp.high = {5.0, std::log(10.0)};
    sp.is_int = {false, false};
    sp.log_scaled = {false, true};
    sp.grad_strategy = GradStrategy::Analytical;
    LinearHost host;
    std::atomic<unsigned long> counter{0};
    auto gp = make_gradient_provider<DlibVec, void>(sp, host, &counter, 0,
                                                    &analytical_real);
    DlibVec x; x.set_size(2); x(0) = 1.0; x(1) = std::log(2.0);
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    CHECK(g(0) == Approx(2.0));
    CHECK(g(1) == Approx(3.0 * 2.0));
}

namespace {

/* f = 2*theta0 + 3*theta1, recording every point it is asked for. */
struct RecordingHost {
    mutable std::vector<std::vector<double>> points;
    double operator()(const DlibVec& theta) const {
        points.push_back({theta(0), theta(1)});
        return 2.0 * theta(0) + 3.0 * theta(1);
    }
    /* The values of dimension 1 at the points that differ from x there. */
    std::vector<double> stencil_of_dim1(const DlibVec& x) const {
        std::vector<double> v;
        for (const auto& p : points)
            if (p[1] != x(1)) v.push_back(p[1]);
        return v;
    }
};

}  // namespace

TEST_CASE("GradientProvider NUMERICAL: PARAM fd_kernel selects the kernel "
          "under either OPTIM_SCALE",
          "[HPO][grad_provider][search_space][fd_kernel]")
{
    /* The in-house rule steps h = fd_step * max(|u|, 1), Ceres' rule
     * fd_step * |u| (evaluated in Tadah): at u = 0.5 with fd_step 0.2 the
     * stencil of dimension 1 is 0.7 and 0.3 in-house, 0.6 and 0.4 with
     * Ceres. `PARAM fd_kernel inhouse` under OPTIM_SCALE NONE was accepted
     * and ignored: the Ceres kernel, NONE's default, ran. */
    auto sp = spec_with_space(/*normalise=*/false, /*log_second=*/false);
    sp.param["fd_step"] = 0.2;
    DlibVec x; x.set_size(2); x(0) = 1.0; x(1) = 0.5;
    DlibVec g;

    sp.param_raw["fd_kernel"] = "inhouse";
    {
        RecordingHost host;
        std::atomic<unsigned long> counter{0};
        auto gp = make_gradient_provider<DlibVec, void>(sp, host, &counter, 0);
        REQUIRE(gp.gradient(x, g));
        const auto s = host.stencil_of_dim1(x);
        REQUIRE(s.size() == 2u);
        CHECK(s[0] == Approx(0.7));
        CHECK(s[1] == Approx(0.3));
        CHECK(g(1) == Approx(3.0));
    }
    /* And the other way: ceres in the unit box, where in-house is the
     * default. */
    auto unit = spec_with_space(/*normalise=*/true, /*log_second=*/false);
    unit.param["fd_step"] = 0.2;
    unit.param_raw["fd_kernel"] = "ceres";
    {
        RecordingHost host;
        std::atomic<unsigned long> counter{0};
        auto gp = make_gradient_provider<DlibVec, void>(unit, host, &counter,
                                                        0);
        REQUIRE(gp.gradient(x, g));
        /* u = 0.5 of dimension 1 is 0.1 + 0.5 * 9.9 = 5.05; Ceres' step
         * 0.1 of the unit width is 0.99. */
        const auto s = host.stencil_of_dim1(unit.out(x));
        REQUIRE(s.size() == 2u);
        CHECK(s[0] == Approx(5.05 + 0.99));
        CHECK(s[1] == Approx(5.05 - 0.99));
    }
}

/* GradientProvider: why a gradient() call returned false (gradient_provider.h).
 *
 * The adapters used to read every false return as "MAXEVAL spent": a
 * finite-difference gradient that is not finite because an evaluation of
 * its stencil failed was reported as MAXEVAL or success (DLIB:BFGS, NLopt
 * LD_*, CERES). GradientProvider::failure now says which of MAXEVAL
 * (Budget), a failed stencil evaluation (Stencil) or another non-finite
 * gradient (NotFinite) it was, with the point and the stencil point for
 * the journal; the backends' endings are covered through hpo_run() in
 * test_hpo_termination.cpp ("Gradient not finite: ..."). */

#include "catch2/catch.hpp"

#include <tadah/hpo/gradient_provider.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::GradientFailure;

constexpr double kMax = std::numeric_limits<double>::max();

/* f(x, y) = x + y on x >= 0; x < 0 is a failed evaluation (the default
 * FAILSCORE, the largest double). */
struct FailBelowZero {
    mutable std::atomic<unsigned long> calls{0};
    double operator()(const Vec& x) const {
        calls.fetch_add(1);
        return x(0) < 0.0 ? kMax : x(0) + x(1);
    }
};

/* NaN below zero, x + y elsewhere. */
struct NanBelowZero {
    double operator()(const Vec& x) const {
        return x(0) < 0.0 ? std::numeric_limits<double>::quiet_NaN()
                          : x(0) + x(1);
    }
};

/* Usable losses whose difference overflows: +1e308 for x >= 0, -1e308
 * below. Both are below the failure band of the largest FAILSCORE. */
struct HugeUsable {
    double operator()(const Vec& x) const {
        return x(0) < 0.0 ? -1e308 : 1e308;
    }
};

/* A two-dimensional spec with the in-house central differences, PARAM
 * fd_step 0.2 (h = 0.2 at |u| <= 1), the identity map between internal and
 * real coordinates and the default FAILSCORE. */
tadah::hpo::OptimizerSpec inhouse_spec()
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib  = "DLIB";
    sp.algo = "BFGS";
    sp.low  = {-5.0, -5.0};
    sp.high = { 5.0,  5.0};
    sp.is_int = {false, false};
    sp.log_scaled = {false, false};
    sp.parameter_labels = {"CGRID2B[2]", "SGRID2B[2]"};
    sp.param["fd_step"] = 0.2;
    sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = kMax;
    return sp;
}

Vec point(double a, double b)
{
    Vec x(2);
    x(0) = a;
    x(1) = b;
    return x;
}

}  // namespace

TEST_CASE("GradientProvider: a failed stencil evaluation is a Stencil "
          "failure that names the point and the stencil point",
          "[HPO][grad_provider][gradient_failure]")
{
    const auto sp = inhouse_spec();
    FailBelowZero host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    /* The stencil of 0.1 along the first dimension is 0.3 and -0.1. */
    CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->cause ==
          "a failed evaluation in the finite-difference stencil");
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 "
          "failed: loss 1.797693135e+308");
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- gradient not finite: a failed evaluation in the "
          "finite-difference stencil");

    /* The next call starts afresh. */
    REQUIRE(gp.gradient(point(1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx(1.0));
    CHECK(g(1) == Approx(1.0));
}

TEST_CASE("GradientProvider: a failed stencil evaluation whose difference "
          "stays finite gives a finite gradient",
          "[HPO][grad_provider][gradient_failure]")
{
    /* A finite FAILSCORE (1e9): the quotient does not overflow, so the
     * gradient is finite (and steep); the optimisers use it. */
    auto sp = inhouse_spec();
    sp.failscore = 1e9;
    struct FailAt1e9 {
        double operator()(const Vec& x) const {
            return x(0) < 0.0 ? 1e9 : x(0) + x(1);
        }
    } host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx((0.8 - 1e9) / 0.4));
}

TEST_CASE("GradientProvider: a non-finite stencil or centre loss is a "
          "Stencil failure", "[HPO][grad_provider][gradient_failure]")
{
    const auto sp = inhouse_spec();
    NanBelowZero host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 "
          "failed: loss nan");

    CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->detail ==
          "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is nan");
}

TEST_CASE("GradientProvider: an overflow between usable losses is "
          "NotFinite, not a failed stencil evaluation",
          "[HPO][grad_provider][gradient_failure]")
{
    const auto sp = inhouse_spec();
    HugeUsable host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::NotFinite);
    CHECK(gp.failure->cause == "the finite-difference quotient overflowed");
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- gradient not finite: the finite-difference quotient "
          "overflowed");
}

TEST_CASE("GradientProvider: a spent MAXEVAL is a Budget failure",
          "[HPO][grad_provider][gradient_failure]")
{
    const auto sp = inhouse_spec();
    FailBelowZero host;
    std::atomic<unsigned long> counter{0};
    /* The centre and one stencil point fit; the second does not. */
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 2);
    Vec g;
    CHECK_FALSE(gp.gradient(point(1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Budget);
    CHECK(host.calls.load() == 2u);
}

TEST_CASE("GradientProvider: HYBRID's masked pass and the analytical pass "
          "say why they failed", "[HPO][grad_provider][gradient_failure]"
          "[hybrid]")
{
    auto sp = inhouse_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    FailBelowZero host;
    std::atomic<unsigned long> counter{0};

    SECTION("a failed evaluation in the masked finite-difference pass") {
        auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(2);
            g(0) = 0.0;
            g(1) = 1.0;
            nfd.push_back(0);
            return true;
        };
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        Vec g;
        CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 "
              "failed: loss 1.797693135e+308");
    }
    SECTION("an analytical pass that cannot be formed") {
        auto agf = [](const Vec&, Vec&, std::vector<int>&) { return false; };
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        Vec g;
        CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::NotFinite);
        CHECK(gp.failure->cause ==
              "the analytical gradient could not be formed at this point");
    }
    SECTION("ANALYTICAL: a non-finite analytical gradient") {
        sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
        auto agf = [](const Vec&, Vec& g, std::vector<int>&) {
            g.set_size(2);
            g(0) = std::numeric_limits<double>::infinity();
            g(1) = 0.0;
            return true;
        };
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        Vec g;
        CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::NotFinite);
        CHECK(gp.failure->cause ==
              "the analytical gradient is not finite at this point");
    }
}

/* Ceres' step rule is evaluated in Tadah (fd_stencil.h): no Ceres build
 * needed. */
TEST_CASE("GradientProvider: the Ceres kernel reports a failed stencil "
          "evaluation, a non-finite loss and MAXEVAL",
          "[HPO][grad_provider][gradient_failure][ceres]")
{
    auto sp = inhouse_spec();
    sp.param_raw["fd_kernel"] = "ceres";
    std::atomic<unsigned long> counter{0};
    Vec g;

    SECTION("a failed stencil evaluation") {
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0);
        /* Ceres' relative step, 0.2 * |u| (at least sqrt(epsilon)): 0.01
         * at u = 0.05, whose stencil stays above zero ... */
        REQUIRE(gp.gradient(point(0.05, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::None);
        /* ... and 1.49e-8 at u = 0, whose stencil reaches below it. */
        CHECK_FALSE(gp.gradient(point(0.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK_THAT(gp.failure->detail,
                   Catch::StartsWith("at CGRID2B[2] = 0, the stencil point "
                                     "CGRID2B[2] = -") &&
                   Catch::EndsWith(" failed: loss 1.797693135e+308"));
    }
    SECTION("a non-finite loss") {
        NanBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is nan");
    }
    SECTION("MAXEVAL") {
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 1);
        CHECK_FALSE(gp.gradient(point(1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Budget);
    }
}

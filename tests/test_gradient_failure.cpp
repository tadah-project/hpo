/* GradientProvider: why a gradient() call returned false (gradient_provider.h).
 *
 * The adapters used to read every false return as "MAXEVAL spent": a
 * finite-difference gradient that is not finite because an evaluation of
 * its stencil failed was reported as MAXEVAL or success (DLIB:BFGS, NLopt
 * LD_*, CERES). GradientProvider::failure now says which of MAXEVAL
 * (Budget), a failed stencil evaluation (Stencil) or another non-finite
 * gradient (NotFinite) it was, with the point and the stencil point for
 * the journal; the backends' endings are covered through hpo_run() in
 * test_hpo_termination.cpp ("Gradient not finite: ...", "No usable
 * gradient: ...").
 *
 * A failed evaluation is a loss that is not usable: not finite, or at
 * least 0.999 FAILSCORE. Only a non-finite loss, or a difference that
 * overflowed, used to fail a gradient; HPO_Host clamps every failure to a
 * finite FAILSCORE, so between two failures the difference was an exact 0
 * (a start inside the failure region "converged") and under a finite
 * FAILSCORE a failed stencil point gave a steep, finite slope. The loss at
 * the point decides now: usable, the first failed stencil evaluation fails
 * the gradient; failed, the gradient fails at once -- unless it is a
 * graded failure under FAILSCORE_GRADING ON, whose differences are its
 * direction out of the failure region, taken as they are unless every one
 * is 0. */

#include "catch2/catch.hpp"

#include <tadah/hpo/gradient_provider.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <functional>
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

/* A loss that counts its calls: f(x) of the point. */
struct Counted {
    std::function<double(const Vec&)> f;
    mutable std::atomic<unsigned long> calls{0};
    double operator()(const Vec& x) const {
        calls.fetch_add(1);
        return f(x);
    }
};

/* A graded failure below x = 0 under FAILSCORE 1e9: band D graded by
 * v = -x, as HPO_Host grades a skip cap or a gate by its excess
 * (pc_fail_grading.h). */
double graded_below_zero(double x)
{
    return tadah::hpo::graded_failscore(1e9, tadah::hpo::FailBand::D, -x,
                                        1.0);
}

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
    /* Pins: the status names no usable gradient, whatever the cause.
     * Before the fix: "STOPPED -- gradient not finite: ...", which a failed
     * stencil evaluation under a finite FAILSCORE is not. */
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- no usable gradient: a failed evaluation in the "
          "finite-difference stencil");

    /* The next call starts afresh. */
    REQUIRE(gp.gradient(point(1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx(1.0));
    CHECK(g(1) == Approx(1.0));
}

TEST_CASE("GradientProvider: a failed stencil evaluation fails the gradient "
          "under a finite FAILSCORE too",
          "[HPO][grad_provider][gradient_failure]")
{
    /* FAILSCORE 1e9: the stencil of 0.1 is 0.3 and -0.1, and -0.1 fails.
     * Pins: no gradient, a Stencil failure naming -0.1 and its loss 1e9.
     * Before the fix: the difference (0.8 - 1e9) / 0.4 did not overflow and
     * was returned as the gradient, as steep as the jump to FAILSCORE, and
     * the optimisers followed it (DLIB:BFGS evaluated the bound 28 times in
     * a row). This case pinned that gradient. */
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
    CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->cause ==
          "a failed evaluation in the finite-difference stencil");
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 "
          "failed: loss 1000000000");
}

TEST_CASE("GradientProvider: a start inside the failure region: the failed "
          "loss at the point fails the gradient before its stencil is "
          "evaluated", "[HPO][grad_provider][gradient_failure]")
{
    /* FAILSCORE (the default, the largest double) for x < 0, at (-1, 0.5):
     * the point and every stencil point fail.
     * Pins: no gradient, a Stencil failure naming the loss at the point,
     * and one evaluation (the point; none with the caller's loss), with
     * the in-house kernel CENTRAL and FORWARD (diff_method 1) and Ceres'
     * step rule.
     * Before the fix: the gradient was (0, 0) -- each difference
     * (FS - FS) / 2h of two equal failures -- after five evaluations (three
     * FORWARD; with the caller's loss four, two), and a gradient search
     * took the start for converged. */
    struct Case { const char* name; const char* kernel; int method; unsigned long calls; };
    for (const Case& c : {Case{"inhouse CENTRAL", "inhouse", 0, 1},
                          Case{"inhouse FORWARD", "inhouse", 1, 1},
                          Case{"ceres CENTRAL", "ceres", 0, 1}}) {
        DYNAMIC_SECTION(c.name) {
            auto sp = inhouse_spec();
            sp.param_raw["fd_kernel"] = c.kernel;
            sp.param["diff_method"] = c.method;
            FailBelowZero host;
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0);
            Vec g;
            CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
            CHECK(gp.failure->kind == GradientFailure::Stencil);
            CHECK(gp.failure->cause ==
                  "a failed evaluation in the finite-difference stencil");
            CHECK(gp.failure->detail ==
                  "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is "
                  "1.797693135e+308");
            CHECK(host.calls.load() == c.calls);

            /* the caller's loss at the point: nothing evaluated */
            CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), kMax, g));
            CHECK(gp.failure->kind == GradientFailure::Stencil);
            CHECK(gp.failure->detail ==
                  "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is "
                  "1.797693135e+308");
            CHECK(host.calls.load() == c.calls);
        }
    }
}

TEST_CASE("GradientProvider: a usable point whose two stencil points fail "
          "has no gradient along that dimension",
          "[HPO][grad_provider][gradient_failure]")
{
    /* A feasible island narrower than the stencil: usable on |x| <= 0.1,
     * the default FAILSCORE elsewhere; at (0, 0.5) the stencil is 0.2 and
     * -0.2.
     * Pins: no gradient, a Stencil failure naming 0.2, after two
     * evaluations (the point, 0.2; -0.2 is not evaluated).
     * Before the fix: g(0) = (FS - FS) / 0.4 = 0 after five evaluations, a
     * gradient that said the island's edge was flat. */
    auto sp = inhouse_spec();
    Counted host{[](const Vec& x) {
        return std::abs(x(0)) <= 0.1 ? x(0) + x(1) : kMax;
    }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    CHECK_FALSE(gp.gradient(point(0.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0, the stencil point CGRID2B[2] = 0.2 failed: "
          "loss 1.797693135e+308");
    CHECK(host.calls.load() == 2u);
}

TEST_CASE("GradientProvider: the first failed stencil point ends the "
          "gradient", "[HPO][grad_provider][gradient_failure]")
{
    /* The default FAILSCORE above x = 0.2; at (0.1, 0.5) the stencil is 0.3
     * (evaluated first) and -0.1.
     * Pins: a Stencil failure naming 0.3, after two evaluations (the point,
     * 0.3; -0.1 is not evaluated).
     * Before the fix: three evaluations: -0.1 too, before the difference
     * overflowed. */
    auto sp = inhouse_spec();
    Counted host{[](const Vec& x) { return x(0) > 0.2 ? kMax : x(0) + x(1); }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = 0.3 failed: "
          "loss 1.797693135e+308");
    CHECK(host.calls.load() == 2u);
}

TEST_CASE("GradientProvider: a graded failure next to a usable point is a "
          "failed evaluation, with FAILSCORE_GRADING OFF and ON",
          "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9; x < 0 scores 0.9995e9, inside the failure band
     * [0.999e9, 1e9). At the usable point (0.1, 0.5) the stencil point -0.1
     * is such a failure: the difference is the jump to it, not a slope,
     * graded or not.
     * Pins: no gradient, a Stencil failure naming -0.1 and its loss, under
     * FAILSCORE_GRADING OFF and ON; a loss of 0.998e9 there, below the
     * band, is usable and gives the gradient.
     * Before the fix: the gradient (0.8 - 0.9995e9) / 0.4, finite and
     * steep, in both. */
    for (const bool grading : {false, true}) {
        DYNAMIC_SECTION("FAILSCORE_GRADING " << (grading ? "ON" : "OFF")) {
            auto sp = inhouse_spec();
            sp.failscore = 1e9;
            sp.failscore_grading = grading;
            Counted host{[](const Vec& x) {
                return x(0) < 0.0 ? 0.9995e9 : x(0) + x(1);
            }};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0);
            Vec g;
            CHECK_FALSE(gp.gradient(point(0.1, 0.5), g));
            CHECK(gp.failure->kind == GradientFailure::Stencil);
            CHECK(gp.failure->detail ==
                  "at CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 "
                  "failed: loss 999500000");

            Counted below{[](const Vec& x) {
                return x(0) < 0.0 ? 0.998e9 : x(0) + x(1);
            }};
            auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, below, &counter, 0);
            REQUIRE(gp2.gradient(point(0.1, 0.5), g));
            CHECK(g(0) == Approx((0.8 - 0.998e9) / 0.4));
        }
    }
}

TEST_CASE("GradientProvider: under FAILSCORE_GRADING a graded failure at the "
          "point has the graded slope as its gradient",
          "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9, FAILSCORE_GRADING ON: below x = 0 the graded band-D
     * loss of v = -x, which falls towards the usable side; x + y above. A
     * graded failure is no usable loss, but its differences point out of
     * the failure region, which is what the grading is for.
     * Pins (passes before the fix as well: a pin of the graded slope, which
     * a rule that failed every failed evaluation would break): at (-1,
     * 0.5) the gradient is the central difference of the graded losses at
     * -0.8 and -1.2, and 0 along y, after five evaluations; at (-0.1, 0.5),
     * next to the usable side, the difference of the usable 0.6 at 0.1 and
     * the graded loss at -0.3, a direction out (negative). */
    auto sp = inhouse_spec();
    sp.failscore = 1e9;
    sp.failscore_grading = true;
    Counted host{[](const Vec& x) {
        return x(0) < 0.0 ? graded_below_zero(x(0)) : x(0) + x(1);
    }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(-1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx((graded_below_zero(-0.8) - graded_below_zero(-1.2)) /
                         0.4).epsilon(1e-12));
    CHECK(g(0) < 0.0);
    CHECK(g(1) == 0.0);
    CHECK(host.calls.load() == 5u);

    REQUIRE(gp.gradient(point(-0.1, 0.5), g));
    CHECK(g(0) == Approx((0.6 - graded_below_zero(-0.3)) / 0.4));
    CHECK(g(0) < 0.0);
}

TEST_CASE("GradientProvider: under FAILSCORE_GRADING a flat graded failure "
          "region, or FAILSCORE itself at the point, fails the gradient",
          "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9, FAILSCORE_GRADING ON.
     * Pins: (a) a constant graded failure 0.99975e9 below x = 0 (a
     * saturated measure, a band without its measure): at (-1, 0.5) every
     * difference is 0, a Stencil failure that says so, after five
     * evaluations; (b) FAILSCORE itself below x = 0 (an ungraded failure:
     * an infrastructure failure, a loss the host clamped): the gradient
     * fails at the point, after one evaluation.
     * Before the fix: the gradient (0, 0) in both, a converged start. */
    auto sp = inhouse_spec();
    sp.failscore = 1e9;
    sp.failscore_grading = true;
    std::atomic<unsigned long> counter{0};
    Vec g;
    SECTION("a flat graded failure region") {
        Counted host{[](const Vec& x) {
            return x(0) < 0.0 ? 0.99975e9 : x(0) + x(1);
        }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->cause ==
              "a failed evaluation in the finite-difference stencil");
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is 999750000, "
              "and every difference at it is 0: a flat failure region");
        CHECK(host.calls.load() == 5u);
    }
    SECTION("FAILSCORE at the point") {
        Counted host{[](const Vec& x) { return x(0) < 0.0 ? 1e9 : x(0) + x(1); }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is 1000000000");
        CHECK(host.calls.load() == 1u);
    }
}

TEST_CASE("GradientProvider: HYBRID's finite differences follow the same "
          "rule", "[HPO][grad_provider][gradient_failure][hybrid]")
{
    /* The analytical pass differentiates SGRID2B[2] and leaves CGRID2B[2]
     * to the finite differences (needs_fd {0}); the counter counts the
     * analytical pass as one evaluation. */
    auto sp = inhouse_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    std::atomic<unsigned long> counter{0};
    Vec g;
    const auto agf = [](const Vec&, Vec& g2, std::vector<int>& nfd) {
        g2.set_size(2);
        g2(0) = 0.0;
        g2(1) = 1.0;
        nfd.push_back(0);
        return true;
    };
    SECTION("gradient() inside the failure region") {
        /* Pins: without the loss at the point, the first central stencil
         * point -0.8 fails the gradient at once under FAILSCORE_GRADING
         * OFF: two counted evaluations (the analytical pass, -0.8).
         * Before the fix: g(0) = 0 (two equal failures), three. */
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at CGRID2B[2] = -1, the stencil point CGRID2B[2] = -0.8 "
              "failed: loss 1.797693135e+308");
        CHECK(counter.load() == 2u);
    }
    SECTION("gradient_at() with a failed loss at the point") {
        /* Pins: the caller's loss fails the gradient before any stencil
         * point: one counted evaluation (the analytical pass).
         * Before the fix: the central stencil 1.2 and 0.8, usable, gave
         * g(0) = 1 at a point whose loss had failed. */
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient_at(point(1.0, 0.5), kMax, g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = 1, SGRID2B[2] = 0.5 is "
              "1.797693135e+308");
        CHECK(counter.load() == 1u);
        CHECK(host.calls.load() == 0u);
    }
    SECTION("nothing differenced: the analytical gradient as before") {
        /* Pins (passes before the fix as well: a pin of the scope; the
         * analytical gradient at a failed point is a follow-up): with
         * needs_fd empty, gradient_at at a failed loss returns the
         * analytical gradient and evaluates nothing. */
        FailBelowZero host;
        auto analytical_only = [](const Vec&, Vec& g2, std::vector<int>&) {
            g2.set_size(2);
            g2(0) = 2.0;
            g2(1) = 1.0;
            return true;
        };
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, analytical_only);
        REQUIRE(gp.gradient_at(point(-1.0, 0.5), kMax, g));
        CHECK(g(0) == 2.0);
        CHECK(host.calls.load() == 0u);
    }
    SECTION("FAILSCORE_GRADING ON: a graded failure at the point") {
        /* FAILSCORE 1e9, the graded band-D loss below x = 0.
         * Pins: gradient_at with the graded loss at (-1, 0.5) gives the
         * graded slope after two stencil evaluations (a pin: passes before
         * the fix as well); gradient() without it evaluates the point once
         * its first central stencil point has failed, to tell a graded
         * failure there from a usable loss, and then takes the graded
         * slope: four counted evaluations (the analytical pass, -0.8, the
         * point, -1.2).
         * Before the fix: gradient() never evaluated the point: three. */
        sp.failscore = 1e9;
        sp.failscore_grading = true;
        Counted host{[](const Vec& x) {
            return x(0) < 0.0 ? graded_below_zero(x(0)) : x(0) + x(1);
        }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        const double slope =
            (graded_below_zero(-0.8) - graded_below_zero(-1.2)) / 0.4;
        REQUIRE(gp.gradient_at(point(-1.0, 0.5), graded_below_zero(-1.0), g));
        CHECK(g(0) == Approx(slope).epsilon(1e-12));
        CHECK(counter.load() == 3u);
        counter = 0;
        REQUIRE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(g(0) == Approx(slope).epsilon(1e-12));
        CHECK(counter.load() == 4u);
    }
    SECTION("FAILSCORE_GRADING ON: a usable point whose central stencil "
            "points fail") {
        /* FAILSCORE 1e9 (ungraded) outside |x| <= 0.1, at (0, 0.5).
         * Pins: gradient() evaluates the point once the stencil point 0.2
         * has failed, finds it usable, and fails the gradient there: three
         * counted evaluations (the analytical pass, 0.2, the point).
         * Before the fix: g(0) = 0 (two equal failures), three (the
         * analytical pass, 0.2, -0.2). */
        sp.failscore = 1e9;
        sp.failscore_grading = true;
        Counted host{[](const Vec& x) {
            return std::abs(x(0)) <= 0.1 ? x(0) + x(1) : 1e9;
        }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient(point(0.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at CGRID2B[2] = 0, the stencil point CGRID2B[2] = 0.2 failed: "
              "loss 1000000000");
        CHECK(counter.load() == 3u);
    }
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
    /* Pins: the status of an overflow names no usable gradient too.
     * Before the fix: "STOPPED -- gradient not finite: ...". */
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- no usable gradient: the finite-difference quotient "
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

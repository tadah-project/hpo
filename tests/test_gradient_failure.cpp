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
 * the point decides now: failed, the gradient fails at once -- unless it is a
 * graded failure under FAILSCORE_GRADING ON, whose differences are its
 * direction out of the failure region, unless every one is 0. And a
 * difference never mixes the kinds: each
 * component takes the side of its stencil whose points are of the kind of
 * the loss at the point -- usable at a usable point, graded failures at a
 * graded one --, one-sided into that side as a bound there would make it,
 * and fails the gradient when no side is. ANALYTICAL and HYBRID read the
 * loss the caller gives (gradient_at) before their analytical pass: a failed
 * one, but for a graded failure under FAILSCORE_GRADING ON, fails the
 * gradient at once. */

#include "catch2/catch.hpp"

#include <tadah/hpo/gradient_provider.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
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

TEST_CASE("GradientProvider: a failed stencil point on one side: the "
          "difference is taken on the other side",
          "[HPO][grad_provider][gradient_failure]")
{
    /* f = x + y, the default FAILSCORE below x = 0. The central stencil of
     * 0.1 along the first dimension is 0.3 and -0.1, and -0.1 fails.
     * Pins: the gradient (1, 1): the first component the one-sided 3-point
     * difference into the usable side, 0.3 and 0.5 with the loss at the
     * point, as a bound at -0.1 would make it; six evaluations (the point,
     * 0.3, -0.1, 0.5, and 0.7 and 0.3 along y).
     * Before the fix: no gradient, a Stencil failure naming -0.1 ("at
     * CGRID2B[2] = 0.1, the stencil point CGRID2B[2] = -0.1 failed: loss
     * 1.797693135e+308"), after three evaluations. */
    const auto sp = inhouse_spec();
    FailBelowZero host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));
    CHECK(g(1) == Approx(1.0).epsilon(1e-12));
    CHECK(host.calls.load() == 6u);

    /* The next call starts afresh: the central differences. */
    REQUIRE(gp.gradient(point(1.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx(1.0));
    CHECK(g(1) == Approx(1.0));
}
TEST_CASE("GradientProvider: under a finite FAILSCORE a failed stencil point "
          "is not differenced either",
          "[HPO][grad_provider][gradient_failure]")
{
    /* FAILSCORE 1e9: the stencil of 0.1 is 0.3 and -0.1, and -0.1 fails.
     * Pins: the gradient along x is the one-sided difference of the usable
     * side (0.3 and 0.5): 1.
     * Before the fix: no gradient, a Stencil failure naming -0.1 and its
     * loss 1e9; before that, the difference (0.8 - 1e9) / 0.4, as steep as
     * the jump to FAILSCORE, which the optimisers followed (DLIB:BFGS
     * evaluated the bound 28 times in a row). */
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
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));
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
     * -0.2, and the island has no side to take.
     * Pins: no gradient, a Stencil failure naming both stencil points, after
     * three evaluations (the point, 0.2, -0.2); the status names no usable
     * gradient.
     * Before the fix: the failure named 0.2 alone ("the stencil point
     * CGRID2B[2] = 0.2 failed: loss 1.797693135e+308"), after two
     * evaluations (-0.2 was not evaluated). */
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
    CHECK(gp.failure->cause ==
          "a failed evaluation in the finite-difference stencil");
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0, the stencil points CGRID2B[2] = 0.2 and "
          "CGRID2B[2] = -0.2 failed: losses 1.797693135e+308 and "
          "1.797693135e+308");
    CHECK(host.calls.load() == 3u);
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- no usable gradient: a failed evaluation in the "
          "finite-difference stencil");
}
TEST_CASE("GradientProvider: a failed first stencil point: the other side is "
          "taken, one evaluation more for its far point",
          "[HPO][grad_provider][gradient_failure]")
{
    /* The default FAILSCORE above x = 0.2; at (0.1, 0.5) the stencil is 0.3
     * (evaluated first) and -0.1.
     * Pins: 0.3 fails, so the first component is the one-sided difference
     * into the lower side, -0.1 and -0.3 with the loss at the point: 1; six
     * evaluations (the point, 0.3, -0.1, -0.3, 0.7, 0.3).
     * Before the fix: a Stencil failure naming 0.3, after two evaluations
     * (the point, 0.3; -0.1 was not evaluated). */
    auto sp = inhouse_spec();
    Counted host{[](const Vec& x) { return x(0) > 0.2 ? kMax : x(0) + x(1); }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(0.1, 0.5), g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));
    CHECK(g(1) == Approx(1.0).epsilon(1e-12));
    CHECK(host.calls.load() == 6u);
}
TEST_CASE("GradientProvider: FORWARD differences: a forward point of another "
          "kind steps backward", "[HPO][grad_provider][gradient_failure]")
{
    /* diff_method 1 (FORWARD), the default FAILSCORE above x = 0.2, f =
     * x + 2y elsewhere; at (0.1, 0.5) the forward point along x is 0.3,
     * which fails.
     * Pins: the backward point -0.1 is evaluated and the first component
     * is the backward 2-point difference (f(-0.1) - f(0.1)) / -0.2 = 1, the
     * second the forward difference with 0.7, 2; the points asked for, in
     * order: the point, (0.3, 0.5), (-0.1, 0.5), (0.1, 0.7); with the
     * caller's loss the same gradient without the first. With the backward
     * point failing too (the default FAILSCORE outside [0.05, 0.2]) the
     * gradient fails, naming both points: "at CGRID2B[2] = 0.1, the stencil
     * points CGRID2B[2] = 0.3 and CGRID2B[2] = -0.1 failed: losses ...".
     * Before the fix: passes as well (a pin of fd_component's FORWARD rule,
     * which no test reached: the one FORWARD failure case fails at the
     * point; a mutant whose forward step of another kind fails the gradient
     * at once, as before the same-kind rule, fails it). Before the
     * same-kind rule the failed forward point failed the gradient after two
     * evaluations, naming 0.3. */
    auto sp = inhouse_spec();
    sp.param["diff_method"] = 1;
    std::vector<std::pair<double, double>> asked;
    Counted host{[&asked](const Vec& x) {
        asked.emplace_back(x(0), x(1));
        return x(0) > 0.2 ? kMax : x(0) + 2.0 * x(1);
    }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(0.1, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::None);
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));
    CHECK(g(1) == Approx(2.0).epsilon(1e-12));
    REQUIRE(asked.size() == 4u);
    CHECK(asked[0] == std::make_pair(0.1, 0.5));
    CHECK(asked[1].first == Approx(0.3));
    CHECK(asked[1].second == 0.5);
    CHECK(asked[2].first == Approx(-0.1));
    CHECK(asked[2].second == 0.5);
    CHECK(asked[3].first == 0.1);
    CHECK(asked[3].second == Approx(0.7));

    asked.clear();
    REQUIRE(gp.gradient_at(point(0.1, 0.5), 1.1, g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));
    CHECK(g(1) == Approx(2.0).epsilon(1e-12));
    CHECK(asked.size() == 3u);

    Counted island{[](const Vec& x) {
        return x(0) >= 0.05 && x(0) <= 0.2 ? x(0) + 2.0 * x(1) : kMax;
    }};
    auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(sp, island,
                                                             &counter, 0);
    CHECK_FALSE(gp2.gradient(point(0.1, 0.5), g));
    CHECK(gp2.failure->kind == GradientFailure::Stencil);
    CHECK(gp2.failure->cause ==
          "a failed evaluation in the finite-difference stencil");
    CHECK(gp2.failure->detail ==
          "at CGRID2B[2] = 0.1, the stencil points CGRID2B[2] = 0.3 and "
          "CGRID2B[2] = -0.1 failed: losses 1.797693135e+308 and "
          "1.797693135e+308");
    CHECK(island.calls.load() == 3u);
}
TEST_CASE("GradientProvider: under FAILSCORE_GRADING a graded failure without "
          "a side of graded failures has no usable gradient",
          "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9, FAILSCORE_GRADING ON: a graded failure 999750000 on
     * |x| <= 0.1, usable x + y above, FAILSCORE itself (an ungraded
     * failure) below. At (0, 0.5) the central stencil is 0.2 (usable, 0.7)
     * and -0.2 (1e9): neither side is of the kind of the loss at the point.
     * Pins: no gradient, a Stencil failure with the cause "a graded failure
     * without a side of graded failures in its finite-difference stencil"
     * and the detail "at CGRID2B[2] = 0 (a graded failure, loss 999750000),
     * the stencil points CGRID2B[2] = 0.2 and CGRID2B[2] = -0.2 are not
     * graded failures: losses 0.7 and 1000000000"; three evaluations (the
     * point, 0.2, -0.2); the status "STOPPED -- no usable gradient: a
     * graded failure without a side of graded failures in its
     * finite-difference stencil". With one side alone not graded (-0.2 a
     * graded failure too) the gradient is the one-sided graded difference
     * into that side.
     * Before the fix: passes as well (a pin of StencilReport::no_side at a
     * graded failure, which no test reached; a mutant giving it the usable
     * point's cause fails it).
     * Before the same-kind rule the gradient was the central difference
     * (0.7 - 1e9) / 0.4, the jump between the two sides, not a slope. */
    auto sp = inhouse_spec();
    sp.failscore = 1e9;
    sp.failscore_grading = true;
    Counted host{[](const Vec& x) {
        if (x(0) > 0.1) return x(0) + x(1);
        if (x(0) >= -0.1) return 0.99975e9;
        return 1e9;
    }};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    CHECK_FALSE(gp.gradient(point(0.0, 0.5), g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->cause ==
          "a graded failure without a side of graded failures in its "
          "finite-difference stencil");
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = 0 (a graded failure, loss 999750000), the stencil "
          "points CGRID2B[2] = 0.2 and CGRID2B[2] = -0.2 are not graded "
          "failures: losses 0.7 and 1000000000");
    CHECK(host.calls.load() == 3u);
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- no usable gradient: a graded failure without a side of "
          "graded failures in its finite-difference stencil");

    Counted graded_below{[](const Vec& x) {
        if (x(0) > 0.1) return x(0) + x(1);
        return graded_below_zero(x(0) - 0.1);
    }};
    auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(
        sp, graded_below, &counter, 0);
    REQUIRE(gp2.gradient(point(0.0, 0.5), g));
    const double y0 = graded_below_zero(-0.1);
    const double one_sided = (4.0 * (graded_below_zero(-0.3) - y0) -
                              (graded_below_zero(-0.5) - y0)) / (2.0 * -0.2);
    CHECK(g(0) == Approx(one_sided).epsilon(1e-9));
    CHECK(g(0) < 0.0);
}
TEST_CASE("GradientProvider: a graded failure next to a usable point is a "
          "failed evaluation, with FAILSCORE_GRADING OFF and ON",
          "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9; x < 0 scores 0.9995e9, inside the failure band
     * [0.999e9, 1e9). At the usable point (0.1, 0.5) the stencil point -0.1
     * is such a failure: the difference would be the jump to it, not a
     * slope, graded or not.
     * Pins: the first component is the one-sided difference of the usable
     * side, 1, under FAILSCORE_GRADING OFF and ON; a loss of 0.998e9 there,
     * below the band, is usable and the central difference takes it.
     * Before the fix: no gradient, a Stencil failure naming -0.1 and its
     * loss, in both; before that the gradient (0.8 - 0.9995e9) / 0.4. */
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
            REQUIRE(gp.gradient(point(0.1, 0.5), g));
            CHECK(g(0) == Approx(1.0).epsilon(1e-12));

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
          "point has the graded slope as its gradient, next to the usable "
          "side too", "[HPO][grad_provider][gradient_failure][failscore_grading]")
{
    /* FAILSCORE 1e9, FAILSCORE_GRADING ON: below x = 0 the graded band-D
     * loss of v = -x, which falls towards the usable side; x + y above. A
     * graded failure is no usable loss, but its differences point out of
     * the failure region, which is what the grading is for.
     * Pins: at (-1, 0.5) the gradient is the central difference of the
     * graded losses at -0.8 and -1.2, and 0 along y, after five evaluations
     * (unchanged by the fix: a pin of that part); at (-0.1, 0.5), next to the
     * usable side, the one-sided difference of the graded side, -0.3 and
     * -0.5 with the loss at the point: the band's own slope, negative (a
     * direction out of the failure region), of the order of 1e5, not the
     * jump to the usable side.
     * Before the fix: at (-0.1, 0.5) the difference of the usable 0.6 at 0.1
     * and the graded loss at -0.3, (0.6 - f(-0.3)) / 0.4, below -1e9: as
     * steep as the jump from the failure band, a documented limit
     * (DLIB:BFGS evaluated a bound again and again, LD_SLSQP stopped at
     * once, Ceres left the box). */
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
    const double y0 = graded_below_zero(-0.1);
    const double one_sided =
        (4.0 * (graded_below_zero(-0.1 - 0.2) - y0) -
         (graded_below_zero(-0.1 + 2.0 * -0.2) - y0)) / (2.0 * -0.2);
    CHECK(g(0) == Approx(one_sided).epsilon(1e-9));
    CHECK(g(0) < 0.0);
    CHECK(g(0) > -1e6);
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
        /* Pins: gradient() evaluates the loss at the point first (its kind
         * decides the gradient), finds it failed and fails the gradient
         * there, before the analytical pass: one counted evaluation, the
         * failure naming the loss at the point ("a failed evaluation at the
         * point").
         * Before the fix: the analytical pass ran first and the first
         * central stencil point -0.8 failed the gradient ("at CGRID2B[2] =
         * -1, the stencil point CGRID2B[2] = -0.8 failed: ..."), two counted
         * evaluations. */
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is "
              "1.797693135e+308");
        CHECK(counter.load() == 1u);
    }
    SECTION("gradient_at() with a failed loss at the point") {
        /* Pins: the caller's loss fails the gradient before the analytical
         * pass and any stencil point: nothing evaluated or counted, the
         * cause "a failed evaluation at the point", the detail naming the
         * loss. It was one counted evaluation (the analytical pass) with the
         * stencil's cause until ANALYTICAL and HYBRID read the caller's loss
         * before their analytical pass.
         * Before the fix: the central stencil 1.2 and 0.8, usable, gave
         * g(0) = 1 at a point whose loss had failed. */
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient_at(point(1.0, 0.5), kMax, g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->cause == "a failed evaluation at the point");
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = 1, SGRID2B[2] = 0.5 is "
              "1.797693135e+308");
        CHECK(counter.load() == 0u);
        CHECK(host.calls.load() == 0u);
    }
    SECTION("nothing differenced: a failed loss at the point fails the "
            "gradient") {
        /* With needs_fd empty (the analytical pass covers every parameter,
         * as under PC_OBSERVABLE on OALGO 4) the analytical gradient at a
         * failed point is that of the loss before the failure (a skip
         * cap's uncapped loss, the terms that did not fail), small but not
         * 0, while the optimiser sees FAILSCORE or the graded loss.
         * Pins: gradient_at at the failed loss fails before the analytical
         * pass (a Stencil failure, "a failed evaluation at the point",
         * nothing evaluated or counted); at a usable loss it gives the
         * analytical gradient, one counted evaluation; at a graded failure
         * under FAILSCORE_GRADING ON (FAILSCORE 1e9, the graded band-D loss
         * below x = 0) it runs no analytical pass and differences every
         * parameter: g(0) the graded slope, g(1) = 0 (the grading does not
         * depend on y), four counted evaluations, all of them the host's.
         * Before the fix: it gave the analytical gradient (2, 1) at the
         * failed loss too (this section pinned that, as the scope of the
         * finite-difference rule), and a start inside the failure region
         * converged on FAILSCORE (DLIB:BFGS, Ceres CONVERGENCE); at the
         * graded failure it gave the analytical gradient (2, 1), one
         * counted evaluation and no host call (pinned here until
         * 2026-10-12), and a search that started where a PC_OBSERVABLE term
         * failed converged on the graded loss
         * (test_hpo_partial_failure.cpp). */
        FailBelowZero host;
        auto analytical_only = [](const Vec&, Vec& g2, std::vector<int>&) {
            g2.set_size(2);
            g2(0) = 2.0;
            g2(1) = 1.0;
            return true;
        };
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, analytical_only);
        CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), kMax, g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->cause == "a failed evaluation at the point");
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is "
              "1.797693135e+308");
        CHECK(counter.load() == 0u);
        CHECK(host.calls.load() == 0u);

        REQUIRE(gp.gradient_at(point(1.0, 0.5), 1.5, g));
        CHECK(gp.failure->kind == GradientFailure::None);
        CHECK(g(0) == 2.0);
        CHECK(g(1) == 1.0);
        CHECK(counter.load() == 1u);

        sp.failscore = 1e9;
        sp.failscore_grading = true;
        Counted graded_host{[](const Vec& x) {
            return x(0) < 0.0 ? graded_below_zero(x(0)) : x(0) + x(1);
        }};
        auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, graded_host, &counter, 0, analytical_only);
        counter = 0;
        REQUIRE(gp2.gradient_at(point(-1.0, 0.5), graded_below_zero(-1.0), g));
        CHECK(g(0) == Approx((graded_below_zero(-0.8) -
                              graded_below_zero(-1.2)) / 0.4)
                          .epsilon(1e-12));
        CHECK(g(1) == 0.0);
        CHECK(counter.load() == 4u);
        CHECK(graded_host.calls.load() == 4u);
        CHECK(host.calls.load() == 0u);
    }
    SECTION("FAILSCORE_GRADING ON: a graded failure at the point") {
        /* FAILSCORE 1e9, the graded band-D loss below x = 0.
         * Pins: gradient_at with the graded loss at (-1, 0.5) runs no
         * analytical pass and differences both parameters: the graded
         * slope, g(1) = 0, after four stencil evaluations; gradient()
         * without it evaluates the point first, finds the graded failure and
         * does the same: five counted evaluations (the point, then the four
         * stencil points), g(1) = 0.
         * Before the fix: gradient() ran the analytical pass first and kept
         * its g(1) = 1 at the graded failure (four counted evaluations: the
         * analytical pass, -0.8, the point, -1.2). */
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
        CHECK(g(1) == 0.0);
        CHECK(counter.load() == 4u);
        CHECK(host.calls.load() == 4u);
        counter = 0;
        REQUIRE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(g(0) == Approx(slope).epsilon(1e-12));
        CHECK(g(1) == 0.0);
        CHECK(counter.load() == 5u);
    }
    SECTION("FAILSCORE_GRADING ON: a usable point whose central stencil "
            "points fail") {
        /* FAILSCORE 1e9 (ungraded) outside |x| <= 0.1, at (0, 0.5).
         * Pins: gradient() evaluates the point first (usable), runs the
         * analytical pass, and the differenced component has no usable
         * side: 0.2 and -0.2 fail; four counted evaluations (the point, the
         * analytical pass, 0.2, -0.2), the failure naming both.
         * Before the fix: three counted evaluations (the analytical pass,
         * 0.2, the point evaluated after 0.2 had failed), the failure
         * naming 0.2 alone. */
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
              "at CGRID2B[2] = 0, the stencil points CGRID2B[2] = 0.2 and "
              "CGRID2B[2] = -0.2 failed: losses 1000000000 and 1000000000");
        CHECK(counter.load() == 4u);
    }
    SECTION("FAILSCORE_GRADING ON: a flat graded failure region") {
        /* FAILSCORE 1e9; a constant graded failure 0.99975e9 below x = 0
         * (a saturated measure), x + y above.
         * Pins: at (-1, 0.5) the differenced component is 0, and HYBRID
         * fails the gradient as NUMERICAL does: a Stencil failure "... is
         * 999750000, and every difference at it is 0: a flat failure
         * region", with the graded loss given (gradient_at: four counted
         * evaluations, no analytical pass and both parameters differenced;
         * three, the analytical pass, -0.8 and -1.2, until 2026-10-12) and
         * without it (gradient(): five, the point first, then as
         * gradient_at).
         * Before the fix: gradient() made four counted evaluations (the
         * analytical pass, -0.8, the point evaluated once -0.8 had failed,
         * -1.2). */
        sp.failscore = 1e9;
        sp.failscore_grading = true;
        Counted host{[](const Vec& x) {
            return x(0) < 0.0 ? 0.99975e9 : x(0) + x(1);
        }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        const std::string flat =
            "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is 999750000, and "
            "every difference at it is 0: a flat failure region";
        CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), 0.99975e9, g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail == flat);
        CHECK(counter.load() == 4u);
        counter = 0;
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail == flat);
        CHECK(counter.load() == 5u);
    }
    SECTION("FAILSCORE_GRADING ON: FAILSCORE at the point that gradient() "
            "evaluates") {
        /* FAILSCORE 1e9 (ungraded: an infrastructure failure, a loss the
         * host clamped) below x = 0.
         * Pins: gradient() at (-1, 0.5) evaluates the point first, finds
         * FAILSCORE there, and fails the gradient at it before the
         * analytical pass: a Stencil failure naming the loss at the point,
         * one counted evaluation.
         * Before the fix: three counted evaluations (the analytical pass,
         * -0.8, the point evaluated once -0.8 had failed). */
        sp.failscore = 1e9;
        sp.failscore_grading = true;
        Counted host{[](const Vec& x) { return x(0) < 0.0 ? 1e9 : x(0) + x(1); }};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is 1000000000");
        CHECK(counter.load() == 1u);
    }
}

TEST_CASE("GradientProvider: ANALYTICAL fails at a failed loss the caller "
          "gives, before the analytical pass",
          "[HPO][grad_provider][gradient_failure][analytical]")
{
    /* The analytical gradient (2, 1) everywhere, its passes counted; the
     * default FAILSCORE below x = 0. The analytical gradient at a failed
     * point is that of the loss before the failure (a skip cap's uncapped
     * loss), small but not 0 in a run, while the optimiser sees FAILSCORE.
     * Pins: gradient_at with the failed loss at (-1, 0.5) fails -- a
     * Stencil failure, the cause "a failed evaluation at the point", the
     * status "STOPPED -- no usable gradient: a failed evaluation at the
     * point", the detail naming the loss -- and runs no analytical pass:
     * nothing is evaluated or counted. So do FAILSCORE 1e9 at the point
     * (FAILSCORE_GRADING OFF and ON), a loss that is not finite, and a
     * graded failure under OFF. With a usable loss, and with a graded
     * failure under FAILSCORE_GRADING ON, gradient_at gives the analytical
     * gradient. gradient() has no loss to read and gives the analytical
     * gradient (a pin: passes before the fix as well).
     * Before the fix: gradient_at gave (2, 1) whatever the loss, so a search
     * started inside the failure region converged on FAILSCORE (DLIB:BFGS,
     * Ceres CONVERGENCE, NLopt "algorithm converged"). */
    auto sp = inhouse_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
    std::atomic<unsigned long> counter{0};
    unsigned long passes = 0;
    const auto agf = [&passes](const Vec&, Vec& g2, std::vector<int>&) {
        ++passes;
        g2.set_size(2);
        g2(0) = 2.0;
        g2(1) = 1.0;
        return true;
    };
    FailBelowZero host;
    Vec g;
    const std::string at = "the loss at CGRID2B[2] = -1, SGRID2B[2] = 0.5 is ";
    SECTION("the default FAILSCORE") {
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), kMax, g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->cause == "a failed evaluation at the point");
        CHECK(gp.failure->detail == at + "1.797693135e+308");
        CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
              "STOPPED -- no usable gradient: a failed evaluation at the "
              "point");
        CHECK(passes == 0u);
        CHECK(counter.load() == 0u);

        REQUIRE(gp.gradient_at(point(1.0, 0.5), 1.5, g));
        CHECK(gp.failure->kind == GradientFailure::None);
        CHECK(g(0) == 2.0);
        CHECK(g(1) == 1.0);
        CHECK(passes == 1u);
        CHECK(counter.load() == 1u);

        REQUIRE(gp.gradient(point(-1.0, 0.5), g));
        CHECK(g(0) == 2.0);
        CHECK(passes == 2u);
        CHECK(host.calls.load() == 0u);
    }
    SECTION("FAILSCORE 1e9, FAILSCORE_GRADING OFF and ON") {
        sp.failscore = 1e9;
        for (const bool grading : {false, true}) {
            INFO("FAILSCORE_GRADING " << (grading ? "ON" : "OFF"));
            sp.failscore_grading = grading;
            passes = 0;
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0, agf);
            CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), 1e9, g));
            CHECK(gp.failure->kind == GradientFailure::Stencil);
            CHECK(gp.failure->detail == at + "1000000000");
            CHECK_FALSE(gp.gradient_at(
                point(-1.0, 0.5), std::numeric_limits<double>::quiet_NaN(), g));
            CHECK(gp.failure->detail == at + "nan");
            CHECK(passes == 0u);
            /* Pins: a loss in [0.999 FS, FS) fails the gradient before the
             * analytical pass under either setting: under OFF a failure,
             * under ON a graded failure, whose grading the analytical
             * gradient does not see (the same cause, the detail saying so).
             * Before the fix: under ON the analytical gradient (2, 1) was
             * returned there, one pass. */
            CHECK_FALSE(gp.gradient_at(point(-1.0, 0.5), 0.9995e9, g));
            CHECK(gp.failure->kind == GradientFailure::Stencil);
            CHECK(gp.failure->cause == "a failed evaluation at the point");
            CHECK(gp.failure->detail ==
                  at + "999500000" +
                      (grading ? ": a graded failure, whose grading the "
                                 "analytical gradient does not see; "
                                 "GRAD_STRATEGY HYBRID takes finite "
                                 "differences there"
                               : ""));
            CHECK(passes == 0u);
        }
        CHECK(host.calls.load() == 0u);
    }
}

/* A host whose candidate below x = 0 is rejected before training (the
 * bump-support or a density gate of HPO_Host, a graded band-D failure
 * under FAILSCORE 1e9): the thread's trainer slot holds no fit of such a
 * point, and analytical_gradient_gap() says why, as HPO_Host's does ("" at
 * a point it trained). */
struct RejectedBelowZero {
    std::string why =
        "the candidate was rejected before training (bump-support gate "
        "reject (c + w/eta past the cutoff, violation=0.5, graded band D)): "
        "it has no fit to differentiate";
    mutable bool last_rejected = false;
    double operator()(const Vec& x) const {
        last_rejected = x(0) < 0.0;
        return last_rejected ? graded_below_zero(x(0)) : x(0) + x(1);
    }
    std::string analytical_gradient_gap() const {
        return last_rejected ? why : std::string();
    }
};

TEST_CASE("GradientProvider: ANALYTICAL at a candidate rejected before "
          "training is a point without a usable gradient, not an error",
          "[HPO][grad_provider][gradient_failure][analytical]")
{
    /* Under FAILSCORE_GRADING ON a candidate the bump gate rejects is a
     * graded failure, and HPO_Host's analytical gradient hands every
     * parameter to finite differences there: its trainer slot holds no fit
     * of the point. ANALYTICAL took that for a parameter without an
     * analytical derivative and threw, which ended the run FAILED (NLopt
     * LD_*, CERES:LBFGS/WOLFE/NLCG, test_hpo_termination.cpp).
     * Pins: gradient_at at the rejected point (-0.5, 0.5) with its graded
     * loss fails before the analytical pass, as at any graded failure (the
     * analytical gradient does not see the grading): a Stencil failure "a
     * failed evaluation at the point", the detail naming the loss and the
     * graded failure, no analytical pass and nothing evaluated. gradient()
     * without the loss runs the analytical pass, whose host routes both
     * parameters to finite differences and says why: a NotFinite failure
     * (the kind of a point without a usable gradient that is not a failed
     * evaluation, which NLopt and Ceres reject and DLIB:BFGS stops at),
     * the cause "the analytical gradient could not be formed at this
     * point", the detail naming the point, the host's reason and HYBRID.
     * Where the host gives no reason -- a parameter without an analytical
     * derivative at a point it trained -- ANALYTICAL still throws, naming
     * the parameters (a pin: as before the fix).
     * Before the fix: gradient_at threw "GRAD_STRATEGY=ANALYTICAL: no
     * analytical gradient for 2 parameters at this point (indices: [0,
     * 1]) ..." at the rejected point; until 2026-10-12 gradient_at ran the
     * analytical pass there (one counted evaluation) and gave the host's
     * reason. */
    auto sp = inhouse_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
    sp.failscore = 1e9;
    sp.failscore_grading = true;
    std::atomic<unsigned long> counter{0};
    unsigned passes = 0;
    const auto all_fd = [&passes](const Vec&, Vec& g2, std::vector<int>& nfd) {
        ++passes;
        g2.set_size(2);
        g2(0) = 0.0;
        g2(1) = 0.0;
        nfd = {0, 1};
        return true;
    };
    RejectedBelowZero host;
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
        sp, host, &counter, 0, all_fd);
    Vec g;
    const Vec rejected = point(-0.5, 0.5);
    const double fx = host(rejected);
    REQUIRE(fx >= 0.999e9);
    REQUIRE(fx < 1e9);
    CHECK_FALSE(gp.gradient_at(rejected, fx, g));
    CHECK(gp.failure->kind == GradientFailure::Stencil);
    CHECK(gp.failure->cause == "a failed evaluation at the point");
    CHECK_THAT(gp.failure->detail,
               Catch::StartsWith("the loss at CGRID2B[2] = -0.5, SGRID2B[2] "
                                 "= 0.5 is 999") &&
                   Catch::EndsWith(": a graded failure, whose grading the "
                                   "analytical gradient does not see; "
                                   "GRAD_STRATEGY HYBRID takes finite "
                                   "differences there"));
    CHECK(passes == 0u);
    CHECK(counter.load() == 0u);

    CHECK_FALSE(gp.gradient(rejected, g));
    CHECK(gp.failure->kind == GradientFailure::NotFinite);
    CHECK(gp.failure->cause ==
          "the analytical gradient could not be formed at this point");
    CHECK(gp.failure->detail ==
          "at CGRID2B[2] = -0.5, SGRID2B[2] = 0.5: " + host.why +
          "; GRAD_STRATEGY HYBRID takes finite differences there");
    CHECK(tadah::hpo::gradient_stop_status(*gp.failure) ==
          "STOPPED -- no usable gradient: the analytical gradient could not "
          "be formed at this point");
    CHECK(passes == 1u);

    const Vec trained = point(1.0, 0.5);
    const double fy = host(trained);
    CHECK_THROWS_WITH(gp.gradient_at(trained, fy, g),
                      Catch::StartsWith("GRAD_STRATEGY=ANALYTICAL: no "
                                        "analytical gradient for 2 "
                                        "parameters at this point (indices: "
                                        "[0, 1])"));
}

TEST_CASE("GradientProvider: a non-finite stencil loss is of another kind, "
          "a non-finite loss at the point a Stencil failure",
          "[HPO][grad_provider][gradient_failure]")
{
    /* NaN below x = 0.
     * Pins: at (0.1, 0.5) the stencil point -0.1 is NaN, a failure, so the
     * first component takes the usable side (0.3, 0.5): 1; a NaN loss at
     * the point fails the gradient at once.
     * Before the fix: at (0.1, 0.5) a Stencil failure "the stencil point
     * CGRID2B[2] = -0.1 failed: loss nan". */
    const auto sp = inhouse_spec();
    NanBelowZero host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                            &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(point(0.1, 0.5), g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-12));

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

    SECTION("failed evaluations on both sides in the masked "
            "finite-difference pass") {
        /* Pins: a usable island |x| <= 0.1 at (0, 0.5): the differenced
         * component has no usable side, a Stencil failure naming both
         * stencil points; at (0.1, 0.5) with failures below x = 0 the
         * usable side is taken (0.3, 0.5): g(0) = 1, g(1) the analytical 1.
         * Before the fix: at (0.1, 0.5) a Stencil failure naming -0.1 alone
         * ("the stencil point CGRID2B[2] = -0.1 failed: ..."). */
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
        REQUIRE(gp.gradient(point(0.1, 0.5), g));
        CHECK(g(0) == Approx(1.0).epsilon(1e-12));
        CHECK(g(1) == 1.0);
        Counted island{[](const Vec& x) {
            return std::abs(x(0)) <= 0.1 ? x(0) + x(1) : kMax;
        }};
        auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, island, &counter, 0, agf);
        CHECK_FALSE(gp2.gradient(point(0.0, 0.5), g));
        CHECK(gp2.failure->kind == GradientFailure::Stencil);
        CHECK(gp2.failure->detail ==
              "at CGRID2B[2] = 0, the stencil points CGRID2B[2] = 0.2 and "
              "CGRID2B[2] = -0.2 failed: losses 1.797693135e+308 and "
              "1.797693135e+308");
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
        /* Pins: Ceres' relative step, 0.2 * |u| (at least sqrt(epsilon)):
         * 0.01 at u = 0.05, whose stencil stays above zero; 1.49e-8 at
         * u = 0, whose stencil reaches below it: the usable side is taken
         * (2^-26 and 2^-25), g(0) = 1; on an island |x| <= 1e-8 neither
         * side is usable, a Stencil failure naming both points.
         * Before the fix: at u = 0 a Stencil failure naming -2^-26. */
        FailBelowZero host;
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0);
        REQUIRE(gp.gradient(point(0.05, 0.5), g));
        CHECK(gp.failure->kind == GradientFailure::None);
        REQUIRE(gp.gradient(point(0.0, 0.5), g));
        CHECK(g(0) == Approx(1.0).epsilon(1e-6));
        Counted island{[](const Vec& x) {
            return std::abs(x(0)) <= 1e-8 ? x(0) + x(1) : kMax;
        }};
        auto gp2 = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, island, &counter, 0);
        CHECK_FALSE(gp2.gradient(point(0.0, 0.5), g));
        CHECK(gp2.failure->kind == GradientFailure::Stencil);
        CHECK_THAT(gp2.failure->detail,
                   Catch::StartsWith("at CGRID2B[2] = 0, the stencil points "
                                     "CGRID2B[2] = 1.4") &&
                   Catch::EndsWith(" failed: losses 1.797693135e+308 and "
                                   "1.797693135e+308"));
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

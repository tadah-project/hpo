/* The in-house finite-difference stencil at the bounds (fd_stencil.h,
 * gradient_provider.h; owner decision 2026-09-29, C4).
 *
 * The in-house kernel (GRAD_STRATEGY NUMERICAL with fd_kernel inhouse, the
 * default under OPTIM_SCALE AUTO) and HYBRID's masked pass took the
 * central difference (f(u+h) - f(u-h)) / 2h at every point. At a
 * coordinate within h of its bound one stencil point lay outside the
 * OPTIM box: evaluated there, or -- for a RHOFORM NONNEG/MONOTONE
 * amplitude, which HPO_Host projects onto AMPGRIDMB >= 0 -- at the bound,
 * so that at a coordinate on its bound the "derivative" was
 * (f(h) - f(0)) / 2h, half the one-sided slope. Within h of a bound the
 * stencil is now one-sided into the box, (-3 f(u) + 4 f(u+h) - f(u+2h)) /
 * 2h and its mirror, second order like the central one; away from the
 * bounds the central stencil is unchanged, bit for bit. The stub hosts
 * here project every point onto the box, as HPO_Host projects an
 * amplitude, and record what they were asked for. */

#include "catch2/catch.hpp"

#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::DimSpec;
using tadah::hpo::GradientFailure;
using tadah::hpo::GradStrategy;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::ScalePolicy;
using tadah::hpo::SearchSpace;

constexpr double kMax = std::numeric_limits<double>::max();

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

/* A spec over `boxes` (REAL bounds; labels X[1], X[2], ...), OPTIM_SCALE
 * AUTO (the unit box: the in-house kernel is the default) or NONE (with
 * PARAM fd_kernel inhouse), with PARAM fd_step and the default
 * FAILSCORE. */
OptimizerSpec spec_over(const std::vector<std::pair<double, double>>& boxes,
                        double fd_step, bool normalise = true,
                        const std::string& key = "X")
{
    std::vector<DimSpec> dims;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        DimSpec d;
        d.key = key;
        d.user_index = i + 1;
        d.label = key + "[" + std::to_string(i + 1) + "]";
        d.lo = boxes[i].first;
        d.hi = boxes[i].second;
        dims.push_back(d);
    }
    ScalePolicy pol;
    pol.normalise = normalise;
    OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.bind_space(std::make_shared<const SearchSpace>(
        SearchSpace::build(std::move(dims), pol)));
    sp.param["fd_step"] = fd_step;
    if (!normalise) sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = kMax;
    return sp;
}

Vec vec(std::initializer_list<double> v)
{
    Vec x(static_cast<long>(v.size()));
    long i = 0;
    for (double d : v) x(i++) = d;
    return x;
}

/* f of the REAL point projected onto the box, as HPO_Host evaluates an
 * AMPGRIDMB amplitude below 0 at 0; every point asked for is recorded as
 * given. */
struct ProjectingHost {
    std::function<double(const Vec&)> f;
    std::vector<std::pair<double, double>> box;
    mutable std::vector<Vec> asked;
    double operator()(const Vec& x) const {
        asked.push_back(x);
        Vec p = x;
        for (long i = 0; i < p.size(); ++i)
            p(i) = std::clamp(p(i), box[static_cast<std::size_t>(i)].first,
                              box[static_cast<std::size_t>(i)].second);
        return f(p);
    }
    /* Whether every point asked for lies in the box. */
    bool all_inside() const {
        for (const auto& x : asked)
            for (long i = 0; i < x.size(); ++i)
                if (!(x(i) >= box[static_cast<std::size_t>(i)].first &&
                      x(i) <= box[static_cast<std::size_t>(i)].second))
                    return false;
        return true;
    }
};

/* The NUMERICAL gradient of `host` at the internal point u. */
Vec gradient(const OptimizerSpec& sp, ProjectingHost& host, const Vec& u,
             unsigned long* evals = nullptr)
{
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host, &counter,
                                                            0);
    Vec g;
    REQUIRE(gp.gradient(u, g));
    CHECK(gp.failure->kind == GradientFailure::None);
    if (evals) *evals = counter.load();
    return g;
}

double one_d(const OptimizerSpec& sp, ProjectingHost& host, double u)
{
    return gradient(sp, host, vec({u}))(0);
}

}  // namespace

TEST_CASE("fd stencil: a coordinate on its bound gets the one-sided "
          "derivative, not half of it",
          "[HPO][grad_provider][fd_stencil]")
{
    /* The audit's case: RHOFORM NONNEG, OPTIM AMPGRIDMB (3) 0 0.5, a seed
     * amplitude 0, DLIB:BFGS with the default kernel. f = exp(10 a) in the
     * REAL amplitude a: df/du = 5 exp(10 a) (a = 0.5 u); at the bound
     * u = 0 that is 5. The central stencil evaluated u = -h (projected: 0)
     * and gave (f(h) - f(0)) / 2h, about 2.5. */
    const double fd_step = 1e-3;
    auto sp = spec_over({{0.0, 0.5}}, fd_step, true, "AMPGRIDMB");
    ProjectingHost host{[](const Vec& a) { return std::exp(10.0 * a(0)); },
                        {{0.0, 0.5}}, {}};
    const double g = one_d(sp, host, 0.0);
    /* second order: the error is -h^2 f'''/3 = -h^2 * 125/3 = -4.2e-5 */
    CHECK(g == Approx(5.0).margin(125.0 / 3.0 * fd_step * fd_step * 1.1));
    CHECK(host.all_inside());
    /* the upper bound, u = 1 (a = 0.5): df/du = 5 e^5 */
    host.asked.clear();
    const double gu = one_d(sp, host, 1.0);
    CHECK(gu == Approx(5.0 * std::exp(5.0))
                    .margin(125.0 * std::exp(5.0) / 3.0 * fd_step * fd_step * 1.1));
    CHECK(host.all_inside());
}

TEST_CASE("fd stencil: the one-sided stencil is second order at the lower "
          "and the upper bound",
          "[HPO][grad_provider][fd_stencil]")
{
    /* f = exp(u) on the unit box: f' = f''' = e^u. The one-sided error is
     * -h^2 f'''/3 + O(h^3); halving h quarters it. (A first-order
     * one-sided difference would halve it; the old projected central
     * difference is off by about f'/2 whatever h.) */
    ProjectingHost host{[](const Vec& u) { return std::exp(u(0)); },
                        {{0.0, 1.0}}, {}};
    for (const double u : {0.0, 1.0}) {
        CAPTURE(u);
        const double fp = std::exp(u);
        double err[2];
        for (int k = 0; k < 2; ++k) {
            const double h = k == 0 ? 1e-2 : 5e-3;
            const auto sp = spec_over({{0.0, 1.0}}, h);
            err[k] = one_d(sp, host, u) - fp;
            CAPTURE(h);
            CHECK(err[k] == Approx(-h * h * fp / 3.0).epsilon(0.05));
        }
        CHECK(err[0] / err[1] == Approx(4.0).epsilon(0.05));
        /* the formula: (-3 f(u) + 4 f(u+h) - f(u+2h)) / 2h, h pointing
         * into the box */
        const double h = u == 0.0 ? 1e-2 : -1e-2;
        const double f0 = std::exp(u), f1 = std::exp(u + h),
                     f2 = std::exp(u + 2.0 * h);
        CHECK(one_d(spec_over({{0.0, 1.0}}, 1e-2), host, u) ==
              Approx((-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * h)).epsilon(1e-12));
    }
    CHECK(host.all_inside());
}

TEST_CASE("fd stencil: no evaluated point lies outside the OPTIM box",
          "[HPO][grad_provider][fd_stencil]")
{
    const double h = 1e-2;
    const std::vector<std::pair<double, double>> box = {{0.0, 1.0}, {0.0, 1.0}};
    const auto f = [](const Vec& u) {
        return std::sin(3.0 * u(0)) + u(0) * u(1) + u(1) * u(1);
    };
    /* on, just inside and one step from each bound, and interior */
    const std::vector<double> at = {0.0, 0.25 * h, 0.999 * h, h, 0.5,
                                    1.0 - h, 1.0 - 0.999 * h, 1.0 - 0.25 * h,
                                    1.0};
    SECTION("NUMERICAL, the in-house kernel, OPTIM_SCALE AUTO") {
        const auto sp = spec_over(box, h);
        for (double a : at)
            for (double b : at) {
                ProjectingHost host{f, box, {}};
                (void)gradient(sp, host, vec({a, b}));
                CAPTURE(a, b);
                CHECK(host.all_inside());
            }
    }
    SECTION("NUMERICAL, fd_kernel inhouse, OPTIM_SCALE NONE") {
        /* a relative step above |u| = 1: h = fd_step * |u| */
        const std::vector<std::pair<double, double>> wide = {{2.0, 5.0}};
        const auto sp = spec_over(wide, 0.05, /*normalise=*/false);
        for (double a : {2.0, 2.05, 2.1, 3.0, 4.8, 4.9, 5.0}) {
            ProjectingHost host{[](const Vec& u) { return u(0) * u(0); },
                                wide, {}};
            const double g = gradient(sp, host, vec({a}))(0);
            CAPTURE(a);
            CHECK(host.all_inside());
            CHECK(g == Approx(2.0 * a));   /* exact for a quadratic */
        }
    }
    SECTION("HYBRID's finite-difference pass") {
        auto sp = spec_over(box, h);
        sp.grad_strategy = GradStrategy::Hybrid;
        auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(2);
            g(0) = 0.0;
            g(1) = 0.0;
            nfd = {0};   /* dimension 1 is analytical */
            return true;
        };
        for (double a : at) {
            ProjectingHost host{f, box, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0, agf);
            Vec g;
            REQUIRE(gp.gradient(vec({a, 0.5}), g));
            CAPTURE(a);
            CHECK(host.all_inside());
        }
    }
}

TEST_CASE("fd stencil: away from the bounds the central difference is "
          "unchanged, bit for bit",
          "[HPO][grad_provider][fd_stencil]")
{
    /* The points and the quotient of the pre-1.6.0 kernel: the centre,
     * then u + h and u - h per dimension, (f+ - f-) / (2.0 * h) with
     * h = fd_step * max(|u|, 1). u = h itself is interior (u - h = 0 is
     * in the box). */
    const auto f = [](const Vec& u) {
        return std::sin(3.0 * u(0)) + std::exp(u(0) * u(1)) + u(1) * u(1);
    };
    const auto check = [&](const OptimizerSpec& sp,
                           const std::vector<std::pair<double, double>>& box,
                           const Vec& u) {
        ProjectingHost host{f, box, {}};
        unsigned long evals = 0;
        const Vec g = gradient(sp, host, u, &evals);
        CHECK(evals == 5u);
        REQUIRE(host.asked.size() == 5u);
        CHECK(host.asked[0] == sp.out(u));
        for (long i = 0; i < 2; ++i) {
            const double h = sp.param.at("fd_step") * std::max(std::abs(u(i)), 1.0);
            Vec up = u, um = u;
            up(i) = u(i) + h;
            um(i) = u(i) - h;
            CHECK(host.asked[static_cast<std::size_t>(1 + 2 * i)] == sp.out(up));
            CHECK(host.asked[static_cast<std::size_t>(2 + 2 * i)] == sp.out(um));
            const double expect = (f(sp.out(up)) - f(sp.out(um))) / (2.0 * h);
            CAPTURE(i, g(i), expect);
            CHECK(same_bits(g(i), expect));
        }
    };
    SECTION("the unit box") {
        const std::vector<std::pair<double, double>> box = {{0.0, 1.0},
                                                            {0.0, 1.0}};
        const auto sp = spec_over(box, 1e-2);
        for (const auto& u : {vec({0.5, 0.25}), vec({1e-2, 0.99}),
                              vec({0.3, 1e-2}), vec({0.7, 0.123456789})})
            check(sp, box, u);
    }
    SECTION("OPTIM_SCALE NONE, a step relative to |u| > 1") {
        const std::vector<std::pair<double, double>> box = {{-5.0, 5.0},
                                                            {-5.0, 5.0}};
        const auto sp = spec_over(box, 1e-3, /*normalise=*/false);
        for (const auto& u : {vec({3.0, -0.5}), vec({-4.99, 4.0}),
                              vec({0.0, 1.5})})
            check(sp, box, u);
    }
}

TEST_CASE("fd stencil: the evaluation count and MAXEVAL",
          "[HPO][grad_provider][fd_stencil]")
{
    const std::vector<std::pair<double, double>> box = {{0.0, 1.0}, {0.0, 1.0}};
    const auto f = [](const Vec& u) { return u(0) + 2.0 * u(1); };

    SECTION("NUMERICAL: the one-sided stencil reuses the centre, 1 + 2 n "
            "evaluations as before") {
        const auto sp = spec_over(box, 1e-2);
        ProjectingHost host{f, box, {}};
        unsigned long evals = 0;
        const Vec g = gradient(sp, host, vec({0.0, 1.0}), &evals);
        CHECK(evals == 5u);
        CHECK(host.asked.size() == 5u);
        CHECK(g(0) == Approx(1.0));
        CHECK(g(1) == Approx(2.0));

        /* MAXEVAL: the centre and the first point of the first stencil */
        ProjectingHost capped{f, box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, capped,
                                                                &counter, 2);
        Vec gc;
        CHECK_FALSE(gp.gradient(vec({0.0, 1.0}), gc));
        CHECK(gp.failure->kind == GradientFailure::Budget);
        CHECK(capped.asked.size() == 2u);
    }
    SECTION("HYBRID: the centre once per gradient, for its one-sided "
            "stencils") {
        auto sp = spec_over(box, 1e-2);
        sp.grad_strategy = GradStrategy::Hybrid;
        auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(2);
            g(0) = 0.0;
            g(1) = 0.0;
            nfd = {0};   /* dimension 2 is analytical: g = 0 */
            return true;
        };
        auto agf2 = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(3);
            g(0) = 0.0;
            g(1) = 0.0;
            g(2) = 7.0;
            nfd = {0, 1};   /* dimension 3 is analytical */
            return true;
        };
        {   /* interior: the analytical pass and the central pair */
            ProjectingHost host{f, box, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0, agf);
            Vec g;
            REQUIRE(gp.gradient(vec({0.5, 0.5}), g));
            CHECK(counter.load() == 3u);
            CHECK(g(0) == Approx(1.0));
        }
        {   /* at a bound: the analytical pass, the centre, two points */
            ProjectingHost host{f, box, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 0, agf);
            Vec g;
            REQUIRE(gp.gradient(vec({0.0, 0.5}), g));
            CHECK(counter.load() == 4u);
            CHECK(g(0) == Approx(1.0));
            CHECK(host.all_inside());
        }
        {   /* two one-sided dimensions share the one centre evaluation */
            const std::vector<std::pair<double, double>> box3 = {
                {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
            auto sp3 = spec_over(box3, 1e-2);
            sp3.grad_strategy = GradStrategy::Hybrid;
            ProjectingHost host{[](const Vec& u) { return u(0) + 2.0 * u(1); },
                                box3, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp3, host, &counter, 0, agf2);
            Vec g;
            REQUIRE(gp.gradient(vec({0.0, 1.0, 0.5}), g));
            CHECK(counter.load() == 1u + 1u + 4u);
            CHECK(g(0) == Approx(1.0));
            CHECK(g(1) == Approx(2.0));
            CHECK(g(2) == 7.0);
            CHECK(host.all_inside());
        }
        {   /* MAXEVAL 2: the analytical pass and the centre fit, the
             * stencil does not */
            ProjectingHost host{f, box, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
                sp, host, &counter, 2, agf);
            Vec g;
            CHECK_FALSE(gp.gradient(vec({0.0, 0.5}), g));
            CHECK(gp.failure->kind == GradientFailure::Budget);
            CHECK(host.asked.size() == 1u);
        }
    }
}

TEST_CASE("fd stencil: failed evaluations of a one-sided stencil",
          "[HPO][grad_provider][fd_stencil][gradient_failure]")
{
    const std::vector<std::pair<double, double>> box = {{0.0, 1.0}};
    const auto sp = spec_over(box, 1e-2);

    SECTION("a failed stencil point is a Stencil failure naming it") {
        /* the default FAILSCORE above u = 0.015: at u = 0 the stencil is
         * 0.01 and 0.02 */
        ProjectingHost host{[](const Vec& u) {
                                return u(0) > 0.015 ? kMax : u(0);
                            },
                            box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({0.0}), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at X[1] = 0, the stencil point X[1] = 0.02 failed: loss "
              "1.797693135e+308");
    }
    SECTION("the loss at the point itself is part of the stencil") {
        ProjectingHost host{[](const Vec& u) {
                                return u(0) == 1.0 ? kMax : u(0);
                            },
                            box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({1.0}), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at X[1] = 1, the stencil point X[1] = 1 failed: loss "
              "1.797693135e+308");
    }
    SECTION("a finite FAILSCORE gives a finite, steep gradient") {
        auto sp9 = sp;
        sp9.failscore = 1e9;
        ProjectingHost host{[](const Vec& u) {
                                return u(0) > 0.015 ? 1e9 : u(0);
                            },
                            box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp9, host,
                                                                &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({0.0}), g));
        CHECK(g(0) == Approx((4.0 * 0.01 - 1e9) / 0.02));
    }
    SECTION("a non-finite loss at a stencil point") {
        ProjectingHost host{[](const Vec& u) {
                                return u(0) > 0.015
                                           ? std::numeric_limits<double>::quiet_NaN()
                                           : u(0);
                            },
                            box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({0.0}), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at X[1] = 0, the stencil point X[1] = 0.02 failed: loss nan");
    }
    SECTION("HYBRID: a centre that is not finite") {
        const std::vector<std::pair<double, double>> box2 = {{0.0, 1.0},
                                                             {0.0, 1.0}};
        auto sp2 = spec_over(box2, 1e-2);
        sp2.grad_strategy = GradStrategy::Hybrid;
        ProjectingHost host{[](const Vec& u) {
                                return u(0) == 0.0
                                           ? std::numeric_limits<double>::quiet_NaN()
                                           : u(0);
                            },
                            box2, {}};
        auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(2);
            g(0) = 0.0;
            g(1) = 0.0;
            nfd = {0};
            return true;
        };
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp2, host, &counter, 0, agf);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({0.0, 0.5}), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "the loss at X[1] = 0, X[2] = 0.5 is nan");
    }
}

TEST_CASE("fd stencil: a box narrower than the stencil is refused",
          "[HPO][grad_provider][fd_stencil]")
{
    const auto f = [](const Vec& u) { return u(0); };
    const auto refused = [&](const OptimizerSpec& sp,
                             const std::vector<std::pair<double, double>>& box,
                             const Vec& u) {
        ProjectingHost host{f, box, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host,
                                                                &counter, 0);
        Vec g;
        std::string what;
        try {
            (void)gp.gradient(u, g);
        } catch (const std::runtime_error& e) {
            what = e.what();
        }
        CHECK(host.asked.empty());   /* before any evaluation */
        return what;
    };
    SECTION("the unit box: fd_step above a third of the width") {
        const std::vector<std::pair<double, double>> box = {{0.0, 1.0},
                                                            {0.0, 1.0}};
        /* the old kernel evaluated u = 0.5 +- 0.4 inside the box and
         * u = 0.1 - 0.4 outside it */
        for (double u : {0.5, 0.1}) {
            CAPTURE(u);
            const std::string what = refused(spec_over(box, 0.4), box,
                                             vec({u, 0.5}));
            /* The message names the step's source, the rule, a point
             * without a stencil and the largest fd_step that fits (owner
             * decision D1(a); review item 8). */
            CHECK_THAT(what,
                       Catch::StartsWith("PARAM fd_step 0.4 does not fit these "
                                         "OPTIM boxes: no finite-difference "
                                         "point is evaluated outside the box, "
                                         "and next to a bound the CENTRAL "
                                         "stencil turns one-sided and needs "
                                         "two steps of h = 0.4 of each box "
                                         "width (in-house rule) inside it; the "
                                         "step is not shrunk:") &&
                           Catch::Contains("\n  - X[1]: OPTIM box [0, 1]; no "
                                           "stencil fits at X[1] = 0.3; fd_step "
                                           "up to 0.333333 fits it") &&
                           Catch::Contains("\n  - X[2]: OPTIM box [0, 1]") &&
                           Catch::EndsWith("\n  Set PARAM fd_step to at most "
                                           "0.333333."));
        }
        /* a third of the width fits */
        ProjectingHost host{f, box, {}};
        (void)gradient(spec_over(box, 0.3), host, vec({0.0, 1.0}));
        CHECK(host.all_inside());
    }
    SECTION("OPTIM_SCALE NONE: a step relative to |u| on a narrow box") {
        const std::vector<std::pair<double, double>> box = {{1000.0, 1000.001}};
        const std::string what = refused(
            spec_over(box, 1e-6, /*normalise=*/false), box, vec({1000.0005}));
        CHECK_THAT(what, Catch::Contains("\n  - X[1]: OPTIM box [1000, "
                                         "1000.001]; no stencil fits at X[1] = "
                                         "1000.0005; fd_step up to 3.33333e-07 "
                                         "fits it\n  Set PARAM fd_step to at "
                                         "most 3.33333e-07; or use OPTIM_SCALE "
                                         "AUTO (fd_step is then a fraction of "
                                         "each box width)."));
    }
    SECTION("HYBRID") {
        const std::vector<std::pair<double, double>> box = {{0.0, 1.0}};
        auto sp = spec_over(box, 0.4);
        sp.grad_strategy = GradStrategy::Hybrid;
        ProjectingHost host{f, box, {}};
        auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
            g.set_size(1);
            g(0) = 0.0;
            nfd = {0};
            return true;
        };
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, void>(
            sp, host, &counter, 0, agf);
        Vec g;
        CHECK_THROWS_WITH(gp.gradient(vec({0.5}), g),
                          Catch::StartsWith("PARAM fd_step 0.4 does not fit "
                                            "these OPTIM boxes"));
        CHECK(host.asked.empty());
    }
}

TEST_CASE("fd stencil: a point outside its box is differenced towards it",
          "[HPO][grad_provider][fd_stencil]")
{
    /* Only an optimiser that ignores the bounds (CERES, NLopt LN_NEWUOA)
     * evaluates such a point. It kept the central stencil, which reached
     * further outside (-0.51); owner decision D4 (2026-09-29): one-sided
     * towards the box. */
    const std::vector<std::pair<double, double>> box = {{0.0, 1.0}};
    const auto sp = spec_over(box, 1e-2);
    std::vector<Vec> asked;
    struct Free {
        std::vector<Vec>* asked;
        double operator()(const Vec& x) const {
            asked->push_back(x);
            return x(0) * x(0);
        }
    } host{&asked};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, void>(sp, host, &counter,
                                                            0);
    Vec g;
    REQUIRE(gp.gradient(vec({-0.5}), g));
    REQUIRE(asked.size() == 3u);
    CHECK(asked[1](0) == -0.5 + 1e-2);
    CHECK(asked[2](0) == -0.5 + 2.0 * 1e-2);
    CHECK(g(0) == Approx(-1.0));
}

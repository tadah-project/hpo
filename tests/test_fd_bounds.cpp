/* Finite-difference gradients near the edge of the OPTIM box, with both
 * step rules (fd_stencil.h, gradient_provider.h; owner decisions D1-D6,
 * 2026-09-29).
 *
 * The in-house stencil kept to the box (one-sided within a step of a
 * bound), but the other step rule -- Ceres' numeric differentiation, the
 * default under OPTIM_SCALE NONE and PARAM fd_kernel ceres -- stepped
 * outside it, and so did FORWARD differences (diff_method 1) at the upper
 * bound; a RHOFORM NONNEG amplitude on its zero bound then got half its
 * slope. Ceres' rule is now evaluated by Tadah, inside the box and bit
 * for bit Ceres' in the interior; diff_method 1 is honoured (it was
 * ignored by the in-house kernel and by HYBRID, as fd_kernel was by
 * HYBRID); a point outside the box -- an optimiser that ignores the
 * bounds -- is differenced towards it; the journal counts what a bound
 * turned; HYBRID takes the caller's loss at the point when it has it.
 *
 * The stub hosts record every REAL point asked for. */

#include "catch2/catch.hpp"

#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#ifdef TADAH_USE_CERES
#include <tadah/hpo/ceres_adapter.h>
#pragma push_macro("CHECK")
#include <ceres/ceres.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/numeric_diff_options.h>
#pragma pop_macro("CHECK")
#endif

#include <dlib/matrix.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::DimSpec;
using tadah::hpo::GradientFailure;
using tadah::hpo::GradStrategy;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::ScaleKind;
using tadah::hpo::ScalePolicy;
using tadah::hpo::SearchSpace;

struct NoRunner {};

constexpr double kMax = std::numeric_limits<double>::max();

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

Vec vec(std::initializer_list<double> v)
{
    Vec x(static_cast<long>(v.size()));
    long i = 0;
    for (double d : v) x(i++) = d;
    return x;
}

DimSpec dim(const std::string& label, double lo, double hi,
            ScaleKind kind = ScaleKind::Linear)
{
    DimSpec d;
    d.key = "X";
    d.label = label;
    d.lo = lo;
    d.hi = hi;
    d.tf.kind = kind;
    return d;
}

/* A DLIB:BFGS spec over `dims` with OPTIM_SCALE AUTO (`normalise`) or
 * NONE, PARAM fd_step, and PARAM fd_kernel `kernel` / diff_method
 * `method` when given. */
OptimizerSpec spec_on(std::vector<DimSpec> dims, bool normalise, double fd_step,
                      const std::string& kernel = "", int method = -1)
{
    ScalePolicy pol;
    pol.normalise = normalise;
    OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.bind_space(std::make_shared<const SearchSpace>(
        SearchSpace::build(std::move(dims), pol)));
    sp.param["fd_step"] = fd_step;
    if (!kernel.empty()) sp.param_raw["fd_kernel"] = kernel;
    if (method >= 0) sp.param["diff_method"] = method;
    sp.failscore = kMax;
    return sp;
}

/* f of the REAL point; every point asked for is recorded. */
struct Recorder {
    std::function<double(const Vec&)> f;
    mutable std::vector<Vec> asked;
    double operator()(const Vec& x) const {
        asked.push_back(x);
        return f(x);
    }
};

/* f of the REAL point projected onto the box, as HPO_Host evaluates an
 * AMPGRIDMB amplitude below 0 at 0 (RHOFORM NONNEG / MONOTONE). */
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
};

/* The step of each rule at u, written out independently of fd_stencil.h:
 * fd_kernel inhouse fd_step * max(|u|, 1); ceres max(2^-26, |u| fd_step). */
double step_of(bool ceres, double fd_step, double u)
{
    return ceres ? std::max(0x1p-26, std::abs(u) * fd_step)
                 : fd_step * std::max(std::abs(u), 1.0);
}

/* A journal for the case; `text()` finishes it and returns it. */
struct Journal {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("tadah_fd_bounds_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    Journal()
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        REQUIRE(log.is_open());
    }
    std::string text()
    {
        HPOLog::instance().finish("test-end");
        std::ifstream f(path);
        std::ostringstream s;
        s << f.rdbuf();
        return s.str();
    }
    ~Journal()
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("test-end");
        log.reset_for_test();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
}

}  // namespace

TEST_CASE("fd bounds: no finite-difference point lies outside the OPTIM box, "
          "with either step rule, CENTRAL or FORWARD, under OPTIM_SCALE AUTO "
          "and NONE, on a LIN and a LOG box",
          "[HPO][grad_provider][fd_bounds]")
{
    /* f = x^3 of the REAL x. At each bound and half a step inside it, every
     * point the gradient evaluates lies in the REAL box, and a CENTRAL
     * gradient is x^3's slope to 1e-6 relative. Ceres' rule (the default
     * under NONE) evaluated u -+ h there, outside the box, and FORWARD u + h
     * above it; a LOG box's upper end came back one ulp above it (D6). */
    const double fd_step = 5e-5;
    struct Box { const char* name; double lo, hi; ScaleKind kind; };
    for (const Box& box : {Box{"LIN [4, 6]", 4.0, 6.0, ScaleKind::Linear},
                           Box{"LOG [0.01, 10]", 0.01, 10.0, ScaleKind::Log}}) {
        for (const bool normalise : {true, false}) {
            for (const bool ceres : {false, true}) {
                for (const int method : {0, 1}) {
                    const auto sp = spec_on({dim("X[1]", box.lo, box.hi, box.kind)},
                                            normalise, fd_step,
                                            ceres ? "ceres" : "inhouse", method);
                    const double a = sp.low[0], b = sp.high[0];
                    const double ha = step_of(ceres, fd_step, a);
                    const double hb = step_of(ceres, fd_step, b);
                    for (const double u : {a, a + ha / 2, b - hb / 2, b}) {
                        CAPTURE(box.name, normalise, ceres, method, u);
                        Recorder host{[](const Vec& x) {
                                          return x(0) * x(0) * x(0);
                                      }, {}};
                        std::atomic<unsigned long> counter{0};
                        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                            sp, host, &counter, 0);
                        Vec g;
                        REQUIRE(gp.gradient(vec({u}), g));
                        /* the same cost as in the interior */
                        CHECK(counter.load() == (method == 0 ? 3u : 2u));
                        for (const Vec& x : host.asked) {
                            CAPTURE(x(0));
                            CHECK(x(0) >= box.lo);
                            CHECK(x(0) <= box.hi);
                        }
                        const double x = sp.out(vec({u}))(0);
                        const double exact =
                            3.0 * x * x * sp.space->dreal_dinternal(0, u);
                        if (method == 0)
                            CHECK(g(0) == Approx(exact).epsilon(1e-6));
                        else   /* first order */
                            CHECK(g(0) == Approx(exact).epsilon(2e-3));
                    }
                }
            }
        }
    }
}

TEST_CASE("fd bounds: a NONNEG amplitude on its zero bound gets its full "
          "slope, with either step rule",
          "[HPO][grad_provider][fd_bounds][rhoform]")
{
    /* f = 5 a of the REAL amplitude a in [0, 0.5], which the host projects
     * onto a >= 0 (HPO_Host, RHOFORM NONNEG). At a = 0 the central
     * difference of Ceres' rule evaluated -2^-26, projected to 0, and gave
     * (f(h) - f(0)) / 2h, half the slope. */
    for (const bool normalise : {true, false}) {
        for (const std::string kernel : {"", "inhouse", "ceres"}) {
            CAPTURE(normalise, kernel);
            const auto sp = spec_on({dim("AMPGRIDMB[3]", 0.0, 0.5)}, normalise,
                                    1e-6, kernel);
            ProjectingHost host{[](const Vec& p) { return 5.0 * p(0); },
                                {{0.0, 0.5}}, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                sp, host, &counter, 0);
            Vec g;
            REQUIRE(gp.gradient(vec({sp.low[0]}), g));
            const double slope = 5.0 * sp.space->dreal_dinternal(0, sp.low[0]);
            CHECK(g(0) == Approx(slope).epsilon(1e-9));
            for (const Vec& x : host.asked) CHECK(x(0) >= 0.0);
        }
    }
}

#ifdef TADAH_USE_CERES
namespace {

/* Ceres' functor: records the points it is asked for, in order. */
struct CeresRecorder {
    const std::function<double(const Vec&)>* f;
    std::vector<Vec>*                         asked;
    int                                       dim;
    bool operator()(double const* const* params, double* residuals) const {
        Vec x(dim);
        for (int i = 0; i < dim; ++i) x(i) = params[0][i];
        asked->push_back(x);
        residuals[0] = (*f)(x);
        return true;
    }
};

template<ceres::NumericDiffMethodType kMethod>
Vec ceres_gradient(const std::function<double(const Vec&)>& f, const Vec& x,
                   double fd_step, std::vector<Vec>& asked)
{
    const int dim = static_cast<int>(x.size());
    ceres::NumericDiffOptions opts;
    opts.relative_step_size = fd_step;
    auto* cf = new ceres::DynamicNumericDiffCostFunction<CeresRecorder, kMethod>(
        new CeresRecorder{&f, &asked, dim}, ceres::TAKE_OWNERSHIP, opts);
    std::unique_ptr<ceres::CostFunction> own(cf);
    cf->AddParameterBlock(dim);
    cf->SetNumResiduals(1);
    std::vector<double> px(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) px[static_cast<std::size_t>(i)] = x(i);
    const double* params[1] = {px.data()};
    std::vector<double> jac(static_cast<std::size_t>(dim), 0.0);
    double* jacs[1] = {jac.data()};
    double residual = 0.0;
    REQUIRE(cf->Evaluate(params, &residual, jacs));
    Vec g(dim);
    for (int i = 0; i < dim; ++i) g(i) = jac[static_cast<std::size_t>(i)];
    return g;
}

}  // namespace

TEST_CASE("fd bounds: Ceres' rule is Ceres' DynamicNumericDiffCostFunction "
          "bit for bit in the interior, CENTRAL and FORWARD",
          "[HPO][grad_provider][fd_bounds][ceres]")
{
    /* The same points in the same order -- the centre, then per dimension
     * +h and -h (FORWARD +h) -- and the same gradient bits, at 200 random
     * interior points (and at 0 and a tiny u, where the step is 2^-26). */
    const std::function<double(const Vec&)> f = [](const Vec& x) {
        return std::sin(3.0 * x(0)) + std::exp(0.5 * x(0) * x(1)) +
               x(1) * x(1) * x(2) + std::cos(x(2));
    };
    std::mt19937_64 rng(20260929);
    std::uniform_real_distribution<double> coord(-4.0, 4.0);
    const double steps[] = {1e-6, 1e-4, 1e-2};
    for (int k = 0; k < 200; ++k) {
        Vec x = vec({coord(rng), coord(rng), coord(rng)});
        if (k == 0) x = vec({0.0, 1e-12, -2.5});
        const double fd_step = steps[k % 3];
        for (const int method : {0, 1}) {
            OptimizerSpec sp;
            sp.lib = "DLIB";
            sp.algo = "BFGS";
            sp.low = {-5.0, -5.0, -5.0};
            sp.high = {5.0, 5.0, 5.0};
            sp.is_int.assign(3, false);
            sp.log_scaled.assign(3, false);   /* sp.out is the identity */
            sp.param["fd_step"] = fd_step;
            sp.param["diff_method"] = method;
            sp.param_raw["fd_kernel"] = "ceres";
            Recorder host{f, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                sp, host, &counter, 0);
            Vec g;
            REQUIRE(gp.gradient(x, g));
            std::vector<Vec> asked;
            const Vec want = method == 0
                ? ceres_gradient<ceres::CENTRAL>(f, x, fd_step, asked)
                : ceres_gradient<ceres::FORWARD>(f, x, fd_step, asked);
            CAPTURE(k, method, fd_step, x(0), x(1), x(2));
            REQUIRE(host.asked.size() == asked.size());
            for (std::size_t p = 0; p < asked.size(); ++p)
                for (long i = 0; i < 3; ++i)
                    CHECK(same_bits(host.asked[p](i), asked[p](i)));
            for (long i = 0; i < 3; ++i) {
                CAPTURE(i, g(i), want(i));
                CHECK(same_bits(g(i), want(i)));
            }
        }
    }
}
#endif

TEST_CASE("fd bounds: the in-house rule's interior gradients are those of "
          "d5457d0, bit for bit", "[HPO][grad_provider][fd_bounds]")
{
    /* Hex-float goldens printed by the GradientProvider of HPO d5457d0
     * (fd_kernel inhouse, the identity map, the box [-5, 5]^3), with the
     * evaluations each gradient made. */
    struct Golden { double fd_step; double x[3]; double g[3]; unsigned long evals; };
    const Golden goldens[] = {
        {0x1.0c6f7a0b5ed8dp-20, {0x1p-1, 0x1p-2, -0x1p+0}, {0x1.618f5a781f3p-2, -0x1.defa9513be4p-3, 0x1.ced548f1d5b8p-1}, 7},
        {0x1.0c6f7a0b5ed8dp-20, {0x1.8p+0, -0x1p+1, 0x1.8p-1}, {-0x1.b60665bd2e6aap-1, -0x1.6a945a6a05f4p+1, 0x1.a8c00fa51558p+1}, 7},
        {0x1.0c6f7a0b5ed8dp-20, {-0x1.ap+1, 0x1p-3, 0x1.4p+1}, {-0x1.65575b2dd0cp+1, -0x1.670f01034eb8p-1, -0x1.2a6af0dcfc68p-1}, 7},
        {0x1.0c6f7a0b5ed8dp-20, {0x0p+0, 0x1p+0, -0x1.3333333333333p-2}, {0x1.bfffffffdbc7p+1, -0x1.333333335924p-1, 0x1.4ba7365677eep+0}, 7},
        {0x1.0c6f7a0b5ed8dp-20, {0x1p+2, -0x1.2p+2, 0x1.f333333333333p+1}, {0x1.44011eed87dp+1, -0x1.18cc4b652e01cp+5, 0x1.4f01171617f9ep+4}, 7},
        {0x1.0624dd2f1a9fcp-10, {0x1p-1, 0x1p-2, -0x1p+0}, {0x1.618f45227f68p-2, -0x1.defa94b11a4cp-3, 0x1.ced5443bde2ep-1}, 7},
        {0x1.0624dd2f1a9fcp-10, {0x1.8p+0, -0x1p+1, 0x1.8p-1}, {-0x1.b60620ee8634p-1, -0x1.6a9459e383332p+1, 0x1.a8c01098f99f8p+1}, 7},
        {0x1.0624dd2f1a9fcp-10, {-0x1.ap+1, 0x1p-3, 0x1.4p+1}, {-0x1.6555e15b7baecp+1, -0x1.670f1499c4dep-1, -0x1.2a6adbf1f358p-1}, 7},
        {0x1.0624dd2f1a9fcp-10, {0x0p+0, 0x1p+0, -0x1.3333333333333p-2}, {0x1.bfffda6d1122p+1, -0x1.33333333330ep-1, 0x1.4ba735836061p+0}, 7},
        {0x1.0624dd2f1a9fcp-10, {0x1p+2, -0x1.2p+2, 0x1.f333333333333p+1}, {0x1.43ff213a029p+1, -0x1.18cc4b64c9343p+5, 0x1.4f0115428c181p+4}, 7},
        {0x1.999999999999ap-5, {0x1p-1, 0x1p-2, -0x1p+0}, {0x1.60bf33dbba74p-2, -0x1.def6f2e31bfap-3, 0x1.cea755cb0b4a8p-1}, 7},
        {0x1.999999999999ap-5, {0x1.8p+0, -0x1p+1, 0x1.8p-1}, {-0x1.b36838bd918b4p-1, -0x1.6a8f35f83faddp+1, 0x1.a8c95dd8cf31p+1}, 7},
        {0x1.999999999999ap-5, {-0x1.ap+1, 0x1p-3, 0x1.4p+1}, {-0x1.57194674ea93bp+1, -0x1.67ce55d510768p-1, -0x1.299ed268e9d7p-1}, 7},
        {0x1.999999999999ap-5, {0x0p+0, 0x1p+0, -0x1.3333333333333p-2}, {0x1.be917b3118a17p+1, -0x1.3333333333324p-1, 0x1.4b9f24c302c8cp+0}, 7},
        {0x1.999999999999ap-5, {0x1p+2, -0x1.2p+2, 0x1.f333333333333p+1}, {0x1.30e8624650e7p+1, -0x1.18cc46fbcd727p+5, 0x1.4eef45529d2fbp+4}, 7},
    };
    const auto f = [](const Vec& x) {
        return std::sin(3.0 * x(0)) + std::exp(0.5 * x(0) * x(1)) +
               x(1) * x(1) * x(2) + std::cos(x(2));
    };
    for (const Golden& gd : goldens) {
        OptimizerSpec sp;
        sp.lib = "DLIB";
        sp.algo = "BFGS";
        sp.low = {-5.0, -5.0, -5.0};
        sp.high = {5.0, 5.0, 5.0};
        sp.is_int.assign(3, false);
        sp.log_scaled.assign(3, false);
        sp.param["fd_step"] = gd.fd_step;
        sp.param_raw["fd_kernel"] = "inhouse";
        Recorder host{f, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({gd.x[0], gd.x[1], gd.x[2]}), g));
        CHECK(counter.load() == gd.evals);
        for (long i = 0; i < 3; ++i) {
            CAPTURE(gd.fd_step, i, g(i), gd.g[i]);
            CHECK(same_bits(g(i), gd.g[i]));
        }
    }
}

TEST_CASE("fd bounds: Ceres' rule next to a bound -- the cost, MAXEVAL and "
          "a failed stencil point", "[HPO][grad_provider][fd_bounds]"
          "[gradient_failure]")
{
    const std::vector<DimSpec> unit2 = {dim("X[1]", 0.0, 1.0),
                                        dim("X[2]", 0.0, 1.0)};
    SECTION("at a bound: 1 + 2n evaluations, as inside the box") {
        const auto sp = spec_on(unit2, true, 1e-2, "ceres");
        Recorder host{[](const Vec& x) { return x(0) + 2.0 * x(1); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({0.0, 1.0}), g));
        CHECK(counter.load() == 5u);
        REQUIRE(host.asked.size() == 5u);
        /* the centre; X[1] at 0 steps up by 2^-26; X[2] at 1 down by 1e-2 */
        CHECK(host.asked[1](0) == 0x1p-26);
        CHECK(host.asked[2](0) == 0x1p-25);
        CHECK(host.asked[3](1) == Approx(0.99).epsilon(1e-15));
        CHECK(host.asked[4](1) == Approx(0.98).epsilon(1e-15));
        CHECK(g(0) == Approx(1.0).epsilon(1e-7));
        CHECK(g(1) == Approx(2.0).epsilon(1e-12));
    }
    SECTION("MAXEVAL spent between u + h and u + 2h: Budget, never above it") {
        const auto sp = spec_on(unit2, true, 1e-2, "ceres");
        Recorder host{[](const Vec& x) { return x(0) + x(1); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 2);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({0.0, 0.5}), g));
        CHECK(gp.failure->kind == GradientFailure::Budget);
        CHECK(counter.load() == 2u);
        CHECK(host.asked.size() == 2u);
    }
    SECTION("a failed u - 2h is a Stencil failure naming that point") {
        const auto sp = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2, "ceres");
        Recorder host{[](const Vec& x) { return x(0) < 0.985 ? kMax : x(0); },
                      {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        CHECK_FALSE(gp.gradient(vec({1.0}), g));
        CHECK(gp.failure->kind == GradientFailure::Stencil);
        CHECK(gp.failure->detail ==
              "at X[1] = 1, the stencil point X[1] = 0.98 failed: loss "
              "1.797693135e+308");
    }
}

TEST_CASE("fd bounds: HYBRID honours PARAM fd_kernel and diff_method, and "
          "keeps to the box", "[HPO][grad_provider][fd_bounds][hybrid]")
{
    /* HYBRID's finite differences are those of NUMERICAL (D5); they used to
     * be the in-house central stencil whatever fd_kernel and diff_method
     * said. X[2] is analytical, X[1] differenced. */
    const auto agf = [](const Vec&, Vec& g, std::vector<int>& nfd) {
        g.set_size(2);
        g(0) = 0.0;
        g(1) = 0.0;
        nfd = {0};
        return true;
    };
    const auto run = [&](const OptimizerSpec& sp, const Vec& u,
                         std::vector<Vec>& asked) {
        Recorder host{[](const Vec& x) { return 3.0 * x(0) + x(1); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0, agf);
        Vec g;
        REQUIRE(gp.gradient(u, g));
        asked = host.asked;
        CHECK(g(0) == Approx(3.0).epsilon(1e-9));
        return counter.load();
    };
    const std::vector<DimSpec> unit2 = {dim("X[1]", 0.0, 1.0),
                                        dim("X[2]", 0.0, 1.0)};
    std::vector<Vec> asked;
    SECTION("fd_kernel ceres: Ceres' rule") {
        auto sp = spec_on(unit2, true, 1e-2, "ceres");
        sp.grad_strategy = GradStrategy::Hybrid;
        /* inside: 1 + 2m, the points u +- |u| fd_step */
        CHECK(run(sp, vec({0.5, 0.5}), asked) == 3u);
        REQUIRE(asked.size() == 2u);
        CHECK(asked[0](0) == 0.5 + 5e-3);
        CHECK(asked[1](0) == 0.5 - 5e-3);
        /* at the bound: 2 + 2m, the centre and one-sided 2^-26, 2^-25 */
        CHECK(run(sp, vec({0.0, 0.5}), asked) == 4u);
        REQUIRE(asked.size() == 3u);
        CHECK(asked[0](0) == 0.0);
        CHECK(asked[1](0) == 0x1p-26);
        CHECK(asked[2](0) == 0x1p-25);
    }
    SECTION("diff_method 1: FORWARD, backward at the upper bound") {
        auto sp = spec_on(unit2, true, 1e-2, "", 1);
        sp.grad_strategy = GradStrategy::Hybrid;
        CHECK(run(sp, vec({0.5, 0.5}), asked) == 3u);
        REQUIRE(asked.size() == 2u);
        CHECK(asked[0](0) == 0.5);
        CHECK(asked[1](0) == 0.5 + 1e-2);
        CHECK(run(sp, vec({1.0, 0.5}), asked) == 3u);
        REQUIRE(asked.size() == 2u);
        CHECK(asked[0](0) == 1.0);
        CHECK(asked[1](0) == 1.0 - 1e-2);
    }
}

TEST_CASE("fd bounds: diff_method 1 takes forward differences, 1 + n "
          "evaluations, with either step rule", "[HPO][grad_provider]"
          "[fd_bounds]")
{
    /* The in-house kernel ignored diff_method: CENTRAL, 1 + 2n. */
    for (const std::string kernel : {"inhouse", "ceres"}) {
        CAPTURE(kernel);
        const auto sp = spec_on({dim("X[1]", 0.0, 1.0), dim("X[2]", 0.0, 1.0)},
                                true, 1e-3, kernel, 1);
        Recorder host{[](const Vec& x) { return x(0) * x(0) + 3.0 * x(1); },
                      {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({0.5, 1.0}), g));
        CHECK(counter.load() == 3u);
        REQUIRE(host.asked.size() == 3u);
        const double h0 = step_of(kernel == "ceres", 1e-3, 0.5);
        const double h1 = step_of(kernel == "ceres", 1e-3, 1.0);
        CHECK(host.asked[1](0) == 0.5 + h0);
        CHECK(host.asked[2](1) == 1.0 - h1);   /* backward at the upper bound */
        CHECK(g(0) == Approx(1.0).epsilon(2e-3));
        CHECK(g(1) == Approx(3.0).epsilon(1e-9));
    }
}

TEST_CASE("fd bounds: a point outside the box is differenced towards it, and "
          "counted", "[HPO][grad_provider][fd_bounds]")
{
    /* Only an optimiser that ignores the bounds (the CERES drivers, NLopt
     * LN_NEWUOA) asks for it. The central stencil of 1.02 evaluated 1.03,
     * further outside (owner decision D4). */
    Journal j;
    const auto sp = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2);
    Recorder host{[](const Vec& x) { return x(0) * x(0); }, {}};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
        sp, host, &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(vec({1.02}), g));
    REQUIRE(host.asked.size() == 3u);
    const double h = 1e-2 * 1.02;   /* fd_step * max(|u|, 1) */
    CHECK(host.asked[0](0) == 1.02);
    CHECK(host.asked[1](0) == 1.02 - h);
    CHECK(host.asked[2](0) == 1.02 - 2.0 * h);
    CHECK(g(0) == Approx(2.04).epsilon(1e-9));
    /* below the box, FORWARD: the one step up */
    auto fwd = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2, "", 1);
    auto gp2 = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
        fwd, host, &counter, 0);
    host.asked.clear();
    REQUIRE(gp2.gradient(vec({-0.5}), g));
    REQUIRE(host.asked.size() == 2u);
    CHECK(host.asked[1](0) == -0.5 + 1e-2);
    const std::string t = j.text();
    CHECK_THAT(t, Catch::Contains("Out-of-box gradients    : 2 (at points outside the OPTIM "
                                  "box, each finite\n    difference stepped "
                                  "towards the box)\n") &&
                  Catch::Contains("finite differences: X[1] = 1.02 lies outside "
                                  "its OPTIM box [0, 1]"));
}

TEST_CASE("fd bounds: the journal names the first one-sided difference of "
          "each dimension and DIAGNOSTICS counts them; a run away from the "
          "bounds adds nothing", "[HPO][grad_provider][fd_bounds][hpolog]")
{
    const auto sp = spec_on({dim("RCUT2B", 4.0, 6.0), dim("CGRID2B[2]", 1.0, 2.0)},
                            true, 1e-2);
    Recorder host{[](const Vec& x) { return x(0) * x(1); }, {}};
    SECTION("next to the bounds") {
        Journal j;
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({1.0, 0.5}), g));      /* RCUT2B at 6 */
        REQUIRE(gp.gradient(vec({0.995, 0.0}), g));    /* both */
        REQUIRE(gp.gradient(vec({0.0, 0.5}), g));      /* RCUT2B at 4 */
        const std::string t = j.text();
        CHECK(count_of(t, "finite differences: RCUT2B = ") == 1u);
        CHECK(count_of(t, "finite differences: CGRID2B[2] = ") == 1u);
        CHECK_THAT(t, Catch::Contains(
                          "finite differences: RCUT2B = 6 lies within a step "
                          "of its upper bound (OPTIM box [4, 6]): one-sided "
                          "3-point difference, at RCUT2B = 5.98 and RCUT2B = "
                          "5.96 with the loss at the point itself; DIAGNOSTICS "
                          "counts such differences per dimension") &&
                      Catch::Contains("One-sided differences   : 4 (a coordinate "
                                      "within a step of its bound, or outside\n"
                                      "    the box, differenced into it): "
                                      "RCUT2B 3, CGRID2B[2] 1\n"));
        CHECK(t.find("Out-of-box gradients") == std::string::npos);
    }
    SECTION("away from the bounds: DIAGNOSTICS as before") {
        Journal j;
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({0.5, 0.5}), g));
        const std::string t = j.text();
        CHECK(t.find("finite differences:") == std::string::npos);
        CHECK(t.find("One-sided differences") == std::string::npos);
        CHECK(t.find("outside the OPTIM box") == std::string::npos);
        CHECK(t.find("HYBRID centre losses") == std::string::npos);
    }
}

TEST_CASE("fd bounds: the OPTIMISER section says how finite differences are "
          "taken", "[HPO][fd_bounds][hpolog]")
{
    struct Case { std::string kernel; int method; bool normalise; std::string line; };
    const std::vector<Case> cases = {
        {"", -1, true,
         "Finite differences    : CENTRAL, h = 0.001 of each box width "
         "(PARAM fd_step; in-house rule); next to a bound: one-sided 3-point "
         "into the box, same h, same cost\n"},
        {"", 1, true,
         "Finite differences    : FORWARD, h = 0.001 of each box width "
         "(PARAM fd_step; in-house rule); next to the upper bound: backward, "
         "same h, same cost\n"},
        {"", -1, false,
         "Finite differences    : CENTRAL, h = max(2^-26, 0.001 x |u|) in "
         "internal units (PARAM fd_step; Ceres' rule); next to a bound: "
         "one-sided 3-point into the box, same h, same cost\n"},
        {"inhouse", -1, false,
         "Finite differences    : CENTRAL, h = 0.001 x max(|u|, 1) in "
         "internal units (PARAM fd_step; in-house rule); next to a bound: "
         "one-sided 3-point into the box, same h, same cost\n"}};
    for (const Case& c : cases) {
        CAPTURE(c.kernel, c.method, c.normalise);
        Journal j;
        auto sp = spec_on({dim("RCUT2B", 4.0, 6.0)}, c.normalise, 1e-3, c.kernel,
                          c.method);
        sp.stop.maxeval = 10;
        HPOLog::instance().write_optimizer_section(sp, sp.parameter_labels);
        CHECK_THAT(j.text(), Catch::Contains(c.line));
    }
}

namespace {

/* A bowl in [0, 1]^2 with its minimum at (0.3, 0.6), and the analytical
 * derivative of its second coordinate only (HYBRID differences the
 * first). Records every point asked for. */
struct HybridBowl {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return 1.0 + (x(0) - 0.3) * (x(0) - 0.3) + (x(1) - 0.6) * (x(1) - 0.6);
    }
    bool analytical_gradient(const Vec& x, Vec& g, std::vector<int>& nfd) {
        g.set_size(2);
        g(0) = 0.0;
        g(1) = 2.0 * (x(1) - 0.6);   /* [0, 1]: internal = real */
        nfd = {0};
        return true;
    }
    /* Whether the start was asked for twice in a row: the optimiser's
     * loss there, then HYBRID's centre of the one-sided difference of X[1],
     * which lies on its bound. */
    bool start_repeated() const {
        return asked.size() >= 2 && same_bits(asked[1](0), asked[0](0)) &&
               same_bits(asked[1](1), asked[0](1));
    }
};

OptimizerSpec hybrid_bowl_spec(const std::string& lib, const std::string& algo)
{
    auto sp = spec_on({dim("X[1]", 0.0, 1.0), dim("X[2]", 0.0, 1.0)}, true,
                      1e-3);
    sp.lib = lib;
    sp.algo = algo;
    sp.grad_strategy = GradStrategy::Hybrid;
    sp.optim_init = {{0.0, 0.2}};   /* X[1] on its lower bound */
    sp.stop.maxeval = 30;
    return sp;
}

}  // namespace

TEST_CASE("fd bounds: HYBRID takes the optimiser's own loss at the point for "
          "its one-sided difference", "[HPO][grad_provider][fd_bounds]"
          "[hybrid]")
{
    /* The NLopt objective and Ceres' FirstOrderFunction evaluate the loss
     * and then ask for the gradient of the same point: HYBRID evaluated
     * that point again for a one-sided difference (review item 2). */
    SECTION("NLopt LD_LBFGS") {
        Journal j;
        HybridBowl host;
        tadah::hpo::run_nlopt_impl<Vec, NoRunner>(
            hybrid_bowl_spec("NLOPT", "LD_LBFGS"), host);
        REQUIRE(host.asked.size() >= 3u);
        CHECK_FALSE(host.start_repeated());
        CHECK(j.text().find("HYBRID centre losses") == std::string::npos);
    }
#ifdef TADAH_USE_CERES
    SECTION("CERES:LBFGS") {
        Journal j;
        HybridBowl host;
        tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
            hybrid_bowl_spec("CERES", "LBFGS"), host, ceres::LBFGS,
            ceres::WOLFE, "LBFGS");
        REQUIRE(host.asked.size() >= 3u);
        CHECK_FALSE(host.start_repeated());
        CHECK(j.text().find("HYBRID centre losses") == std::string::npos);
    }
#endif
    SECTION("a gradient without the caller's loss evaluates it, counted") {
        Journal j;
        HybridBowl host;
        const auto sp = hybrid_bowl_spec("DLIB", "BFGS");
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0,
            tadah::hpo::host_analytical_grad_fn<Vec>(sp, host));
        Vec g;
        REQUIRE(gp.gradient(vec({0.0, 0.2}), g));
        CHECK(counter.load() == 4u);   /* the analytical pass, the centre, 2 */
        CHECK(g(0) == Approx(-0.6).epsilon(1e-9));
        CHECK_THAT(j.text(), Catch::Contains("HYBRID centre losses    : 1 (evaluations of a "
                                             "gradient's point"));
    }
}

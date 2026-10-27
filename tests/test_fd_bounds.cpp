/* Finite-difference gradients near the edge of the OPTIM box, with both
 * step rules (fd_stencil.h, gradient_provider.h).
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
 * The last cases drive the NLopt and Ceres adapters at failed evaluations
 * ("No usable gradient: ...").
 *
 * The stub hosts record every REAL point asked for. */

#include "catch2/catch.hpp"

#include <tadah/hpo/box_penalty.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pc_fail_grading.h>
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
#include <optional>
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
     * above it; a LOG box's upper end came back one ulp above it. */
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
    /* f = 5 a of the REAL amplitude a in [0, 0.5] (RHOFORM NONNEG). At
     * a = 0 the central difference of Ceres' rule evaluated -2^-26, which
     * the host then evaluated at 0 (an amplitude below 0 is outside the
     * box, and not evaluated at all since 2026-10-15: box_penalty.h), and
     * gave (f(h) - f(0)) / 2h, half the slope. Every point asked for lies in
     * the box.
     * Before the fix: passes as well (a pin; the stub host evaluated an
     * amplitude below 0 at 0, as HPO_Host's RHOFORM projection did, which no
     * stencil point asks for). */
    for (const bool normalise : {true, false}) {
        for (const std::string kernel : {"", "inhouse", "ceres"}) {
            CAPTURE(normalise, kernel);
            const auto sp = spec_on({dim("AMPGRIDMB[3]", 0.0, 0.5)}, normalise,
                                    1e-6, kernel);
            Recorder host{[](const Vec& p) { return 5.0 * p(0); }, {}};
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
    SECTION("a failed u - 2h: the 2-point difference with u - h") {
        /* Pins: at the upper bound 1 the one-sided stencil is 0.99 and
         * 0.98; 0.98 fails, so the backward difference of the point and
         * 0.99 is the gradient, 1.
         * Before the fix: a Stencil failure naming 0.98 ("at X[1] = 1, the
         * stencil point X[1] = 0.98 failed: loss 1.797693135e+308"). */
        const auto sp = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2, "ceres");
        Recorder host{[](const Vec& x) { return x(0) < 0.985 ? kMax : x(0); },
                      {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({1.0}), g));
        CHECK(g(0) == Approx(1.0).epsilon(1e-9));
        CHECK(host.asked.size() == 3u);
    }
}

TEST_CASE("fd bounds: HYBRID honours PARAM fd_kernel and diff_method, and "
          "keeps to the box", "[HPO][grad_provider][fd_bounds][hybrid]")
{
    /* HYBRID's finite differences are those of NUMERICAL; they used to
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
        /* inside: 2 + 2m, the centre (gradient() evaluates the loss at the
         * point first: its kind decides every difference) and the points
         * u +- |u| fd_step. Before the fix: 1 + 2m, without the centre. */
        CHECK(run(sp, vec({0.5, 0.5}), asked) == 4u);
        REQUIRE(asked.size() == 3u);
        CHECK(asked[0](0) == 0.5);
        CHECK(asked[1](0) == 0.5 + 5e-3);
        CHECK(asked[2](0) == 0.5 - 5e-3);
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

TEST_CASE("box penalty: which points lie outside the OPTIM box, their "
          "distance, score and slope", "[HPO][fd_bounds][out_of_box]")
{
    /* box_penalty.h. Pins:
     *  - a point inside, on a bound, within kSnapUlps (8) units in the last
     *    place of one (evaluated on it) or NaN is not outside;
     *  - beyond that, the signed excess per dimension in the search's
     *    coordinates (OPTIM_SCALE AUTO: a fraction of the box's width;
     *    NONE on a LOG dimension: ln x), the first dimension outside, how
     *    many are, and the Euclidean distance; x <= 0 on a LOG dimension
     *    is at an infinite distance, where the slope is 0;
     *  - the score lies in [0.99975, 1) x FAILSCORE (band D's range) for
     *    the smallest normal FAILSCORE, 1, 1e9 and the largest double (the
     *    parser refuses a subnormal one), grows with the distance, and is
     *    never a usable loss (loss_is_feasible false);
     *  - the slope, 0.25e-3 FS / (1 + d)^2 e_k / d, points away from the
     *    box and is the derivative of the score.
     * Before the fix: does not compile (box_penalty.h is new). */
    ScalePolicy autos;
    autos.normalise = true;
    ScalePolicy none;
    none.normalise = false;
    const SearchSpace lin = SearchSpace::build(
        {dim("X[1]", 0.0, 2.0), dim("X[2]", 1.0, 5.0)}, autos);
    const SearchSpace logd = SearchSpace::build(
        {dim("X[1]", 0.1, 10.0, ScaleKind::Log)}, none);
    const auto at = [](std::initializer_list<double> v) {
        return std::vector<double>(v);
    };

    SECTION("inside, on a bound, within the snap, NaN: not outside") {
        CHECK_FALSE(tadah::hpo::outside_box(lin, at({1.0, 3.0})));
        CHECK_FALSE(tadah::hpo::outside_box(lin, at({0.0, 5.0})));
        CHECK_FALSE(tadah::hpo::outside_box(
            lin, at({std::nextafter(2.0, 3.0), 3.0})));
        CHECK_FALSE(tadah::hpo::outside_box(
            lin, at({std::numeric_limits<double>::quiet_NaN(), 3.0})));
    }
    SECTION("beyond: the excess in fractions of each width") {
        const auto o = tadah::hpo::outside_box(lin, at({2.5, 0.0}));
        REQUIRE(o);
        CHECK(o->count == 2u);
        CHECK(o->first == 0u);
        REQUIRE(o->excess.size() == 2u);
        CHECK(o->excess[0] == Approx(0.25).epsilon(1e-12));
        CHECK(o->excess[1] == Approx(-0.25).epsilon(1e-12));
        CHECK(o->distance == Approx(std::sqrt(0.125)).epsilon(1e-12));
        const auto one = tadah::hpo::outside_box(lin, at({1.0, 6.0}));
        REQUIRE(one);
        CHECK(one->count == 1u);
        CHECK(one->first == 1u);
        CHECK(one->distance == Approx(0.25).epsilon(1e-12));
    }
    SECTION("a LOG dimension under NONE: ln x, and x <= 0 infinitely far") {
        const auto o = tadah::hpo::outside_box(logd, at({20.0}));
        REQUIRE(o);
        CHECK(o->distance == Approx(std::log(2.0)).epsilon(1e-12));
        const auto z = tadah::hpo::outside_box(logd, at({-1.0}));
        REQUIRE(z);
        CHECK(z->distance == std::numeric_limits<double>::infinity());
        std::vector<double> g;
        tadah::hpo::out_of_box_slope(*z, 1e9, g);
        REQUIRE(g.size() == 1u);
        CHECK(g[0] == 0.0);
        CHECK(tadah::hpo::out_of_box_score(1e9, z->distance) < 1e9);
    }
    SECTION("the score: band D's range, growing, never usable") {
        for (const double fs : {std::numeric_limits<double>::min(), 1.0, 1e9,
                                std::numeric_limits<double>::max()}) {
            CAPTURE(fs);
            double previous = 0.0;
            for (const double d : {1e-12, 1e-3, 0.5, 1.0, 10.0, 1e6}) {
                CAPTURE(d);
                const double l = tadah::hpo::out_of_box_score(fs, d);
                CHECK(l >= 0.99975 * fs);
                CHECK(l < fs);
                CHECK(l > previous);
                CHECK_FALSE(tadah::hpo::loss_is_feasible(l, fs));
                previous = l;
            }
        }
    }
    SECTION("the slope points away from the box: the score's derivative") {
        const double fs = 1e9;
        const auto o = tadah::hpo::outside_box(lin, at({2.5, 0.0}));
        REQUIRE(o);
        std::vector<double> g;
        tadah::hpo::out_of_box_slope(*o, fs, g);
        REQUIRE(g.size() == 2u);
        CHECK(g[0] > 0.0);   /* above the box: the score grows upwards */
        CHECK(g[1] < 0.0);   /* below it: the score grows downwards */
        const double d = o->distance, h = 1e-6;
        const double dl = (tadah::hpo::out_of_box_score(fs, d + h) -
                           tadah::hpo::out_of_box_score(fs, d - h)) / (2 * h);
        CHECK(g[0] == Approx(dl * o->excess[0] / d).epsilon(1e-5));
        CHECK(g[1] == Approx(dl * o->excess[1] / d).epsilon(1e-5));
    }
}

TEST_CASE("fd bounds: a point outside the box is not differenced: every "
          "strategy gives it the slope of its out-of-box score",
          "[HPO][grad_provider][fd_bounds][out_of_box]")
{
    /* A point outside the OPTIM box is not evaluated (box_penalty.h):
     * HPO_Host scores it
     * graded_failscore(FS, D, d, 1) of its distance d to the box in the
     * search's coordinates, and the gradient there is that score's slope.
     * Pins: NUMERICAL (gradient and gradient_at), HYBRID and ANALYTICAL give
     * dL/du_k = 0.25e-3 FS sigma'(d) e_k / d (sigma'(d) = 1 / (1 + d)^2,
     * e_k the signed excess), with no host call, nothing counted and no
     * analytical pass: under OPTIM_SCALE AUTO at u = 1.02 over [0, 1] (d =
     * 0.02), at (-0.5, 1.5) in two dimensions (d = sqrt(0.5)); under NONE
     * on a LOG dimension [0.1, 10] at ln 20 (d = ln 2); DIAGNOSTICS has no
     * "Out-of-box gradients" line, and a point within rounding of a bound
     * is differenced on the bound.
     * Before the fix: the centre was differenced one-sided towards the box
     * (u = 1.02: the host asked for 1.02, 1.0098 and 0.9996, g = 2.04 for
     * x^2), counted ("Out-of-box gradients    : 2 ..."), and ANALYTICAL and
     * HYBRID ran their analytical pass. */
    const double fs = 1e9;
    const auto slope = [fs](double e, double d) {
        return 0.25e-3 * fs / ((1.0 + d) * (1.0 + d)) * e / d;
    };
    unsigned long passes = 0;
    const auto agf = [&passes](const Vec& x, Vec& g, std::vector<int>& nfd) {
        ++passes;
        g.set_size(x.size());
        for (long i = 0; i < x.size(); ++i) g(i) = 1.0;
        nfd.clear();
        return true;
    };
    SECTION("OPTIM_SCALE AUTO, one dimension, every strategy") {
        Journal j;
        for (const auto strat :
             {GradStrategy::Numerical, GradStrategy::Hybrid,
              GradStrategy::Analytical}) {
            CAPTURE(static_cast<int>(strat));
            auto sp = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2);
            sp.failscore = fs;
            sp.grad_strategy = strat;
            Recorder host{[](const Vec& x) { return x(0) * x(0); }, {}};
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                sp, host, &counter, 0, agf);
            Vec g;
            passes = 0;
            REQUIRE(gp.gradient(vec({1.02}), g));
            CHECK(g(0) == Approx(slope(0.02, 0.02)).epsilon(1e-9));
            REQUIRE(gp.gradient_at(vec({1.02}), 0.99975e9, g));
            CHECK(g(0) == Approx(slope(0.02, 0.02)).epsilon(1e-9));
            CHECK(host.asked.empty());
            CHECK(counter.load() == 0u);
            CHECK(passes == 0u);
        }
        const std::string t = j.text();
        CHECK(t.find("Out-of-box gradients") == std::string::npos);
        CHECK(t.find("finite differences: X[1] = 1.02") == std::string::npos);
    }
    SECTION("two dimensions: the slope along the excess") {
        auto sp = spec_on({dim("X[1]", 0.0, 1.0), dim("X[2]", 0.0, 1.0)}, true,
                          1e-2);
        sp.failscore = fs;
        Recorder host{[](const Vec& x) { return x(0) + x(1); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({-0.5, 1.5}), g));
        const double d = std::sqrt(0.5);
        CHECK(g(0) == Approx(slope(-0.5, d)).epsilon(1e-9));
        CHECK(g(1) == Approx(slope(0.5, d)).epsilon(1e-9));
        CHECK(host.asked.empty());
    }
    SECTION("OPTIM_SCALE NONE, a LOG dimension: the distance in ln x") {
        auto sp = spec_on({dim("X[1]", 0.1, 10.0, ScaleKind::Log)}, false,
                          1e-2);
        sp.failscore = fs;
        Recorder host{[](const Vec& x) { return x(0); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({std::log(20.0)}), g));
        const double e = std::log(20.0) - std::log(10.0);
        CHECK(g(0) == Approx(slope(e, e)).epsilon(1e-9));
        CHECK(host.asked.empty());
    }
    SECTION("within rounding of a bound: differenced on the bound") {
        auto sp = spec_on({dim("X[1]", 0.0, 1.0)}, true, 1e-2);
        sp.failscore = fs;
        Recorder host{[](const Vec& x) { return x(0) * x(0); }, {}};
        std::atomic<unsigned long> counter{0};
        auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
            sp, host, &counter, 0);
        Vec g;
        REQUIRE(gp.gradient(vec({std::nextafter(1.0, 2.0)}), g));
        REQUIRE(host.asked.size() == 3u);
        CHECK(host.asked[1](0) == 1.0 - 1e-2);
        CHECK(host.asked[2](0) == 1.0 + 2.0 * -1e-2);
        CHECK(g(0) == Approx(2.0).epsilon(1e-9));
    }
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
                      /* it said "(a coordinate within a step of its
                       * bound, or outside\n    the box, differenced into
                       * it)" before the fix: a point outside the box is not
                       * differenced now */
                      Catch::Contains("One-sided differences   : 4 (a coordinate "
                                      "within a step of its bound, differenced\n"
                                      "    into the box): "
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

TEST_CASE("fd bounds: a difference taken without a point of another kind is "
          "journalled once per dimension and counted in DIAGNOSTICS",
          "[HPO][grad_provider][fd_bounds][hpolog][gradient_failure]")
{
    /* The default FAILSCORE below X[1] = 0.3. At X[1] = 0.35 the central
     * stencil is 0.45 and 0.25, and 0.25 fails: the one-sided difference
     * into the usable side, 0.45 and 0.55, is taken (a difference never
     * mixes the kinds).
     * Pins: two such gradients (at 0.35 and 0.36) give the slope 1 along
     * X[1]; one INFO line names the first ("finite differences: at X[1] =
     * 0.35 (a usable loss), the stencil point X[1] = 0.25 (loss
     * 1.797693135e+308) is of another kind, so the difference is taken
     * without it: one-sided 3-point difference, at X[1] = 0.45 and X[1] =
     * 0.55 with the loss at the point itself; ..."), and DIAGNOSTICS counts
     * both, "Mixed-kind stencils     : 2 (...): X[1] 2".
     * Before the fix: both gradients failed (a Stencil failure naming 0.25),
     * and neither line was written. */
    Journal j;
    const auto sp = spec_on({dim("X[1]", 0.0, 1.0), dim("X[2]", 0.0, 1.0)},
                            true, 0.1);
    Recorder host{[](const Vec& x) { return x(0) < 0.3 ? kMax : x(0) + x(1); },
                  {}};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(sp, host,
                                                                &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(vec({0.35, 0.5}), g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-9));
    REQUIRE(gp.gradient(vec({0.36, 0.5}), g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-9));
    const std::string t = j.text();
    CHECK(count_of(t, "finite differences: at X[1] = ") == 1u);
    CHECK_THAT(t, Catch::Contains(
                      "finite differences: at X[1] = 0.35 (a usable loss), the "
                      "stencil point X[1] = 0.25 (loss 1.797693135e+308) is of "
                      "another kind, so the difference is taken without it: "
                      "one-sided 3-point difference, at X[1] = 0.45 and X[1] = "
                      "0.55 with the loss at the point itself; a difference "
                      "never mixes a usable loss with a failure; DIAGNOSTICS "
                      "counts such differences per dimension") &&
                  Catch::Contains("Mixed-kind stencils     : 2 (a stencil "
                                  "point held a loss of another kind than "
                                  "the\n    point's own -- a failure next to a "
                                  "usable loss, a usable loss or an\n    "
                                  "ungraded failure next to a graded one -- "
                                  "and the difference was taken\n    without "
                                  "it): X[1] 2\n"));
}

TEST_CASE("fd bounds: a difference taken without two points of another kind "
          "names both in its journal line",
          "[HPO][grad_provider][fd_bounds][hpolog][gradient_failure]")
{
    /* A usable island [0.3, 0.5] in the default FAILSCORE. At X[1] = 0.35
     * the central stencil is 0.45 and 0.25, and 0.25 fails; the one-sided
     * difference into the usable side would read 0.55, which fails too.
     * Pins: the gradient along X[1] is the forward 2-point difference with
     * 0.45, 1; the INFO line names both points ("finite differences: at
     * X[1] = 0.35 (a usable loss), the stencil points X[1] = 0.25 (loss
     * 1.797693135e+308) and X[1] = 0.55 (loss 1.797693135e+308) are of
     * another kind, so the difference is taken without them: forward
     * difference, at X[1] = 0.45; ..."), and DIAGNOSTICS counts the
     * component, "Mixed-kind stencils     : 1 (...): X[1] 1".
     * Before the fix: passes as well (a pin of the plural line of
     * StencilReport::mixed and of the 2-point fallback when the far point
     * of the other side fails, which no test reached; a mutant that took the
     * one-sided difference through the failed far point fails it). Before
     * the same-kind rule the gradient failed at 0.25. */
    Journal j;
    const auto sp = spec_on({dim("X[1]", 0.0, 1.0), dim("X[2]", 0.0, 1.0)},
                            true, 0.1);
    Recorder host{[](const Vec& x) {
                      return x(0) >= 0.3 && x(0) <= 0.5 ? x(0) + x(1) : kMax;
                  },
                  {}};
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(sp, host,
                                                                &counter, 0);
    Vec g;
    REQUIRE(gp.gradient(vec({0.35, 0.5}), g));
    CHECK(g(0) == Approx(1.0).epsilon(1e-9));
    CHECK(g(1) == Approx(1.0).epsilon(1e-9));
    const std::string t = j.text();
    CHECK(count_of(t, "finite differences: at X[1] = ") == 1u);
    CHECK_THAT(t, Catch::Contains(
                      "finite differences: at X[1] = 0.35 (a usable loss), the "
                      "stencil points X[1] = 0.25 (loss 1.797693135e+308) and "
                      "X[1] = 0.55 (loss 1.797693135e+308) are of another "
                      "kind, so the difference is taken without them: forward "
                      "difference, at X[1] = 0.45; a difference never mixes a "
                      "usable loss with a failure; DIAGNOSTICS counts such "
                      "differences per dimension") &&
                  Catch::Contains("Mixed-kind stencils     : 1 (") &&
                  Catch::Contains("without it): X[1] 1\n"));
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
     * that point again for a one-sided difference. */
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
    SECTION("CERES:STEEPEST_DESCENT") {
        /* Before the fix: passes as well (a pin; it ran CERES:LBFGS,
         * refused since). */
        Journal j;
        HybridBowl host;
        tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
            hybrid_bowl_spec("CERES", "STEEPEST_DESCENT"), host,
            ceres::STEEPEST_DESCENT, ceres::ARMIJO, "STEEPEST_DESCENT");
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

/* ======================================================================
 * A failed evaluation fails the gradient, whatever FAILSCORE is, and the
 * backends' endings at such a point (gradient_provider.h): one dimension,
 * X[1] in [1, 2] under OPTIM_SCALE AUTO, PARAM fd_step 0.2 (h = 0.2 of the
 * width), MAXEVAL 60, the stub hosts below. The end-to-end cases are in
 * test_hpo_termination.cpp ("No usable gradient: ..."); these drive the
 * NLopt and Ceres adapters directly.
 * ==================================================================== */

namespace {

const std::string kNoUsableStencil =
    "no usable gradient: a failed evaluation in the finite-difference "
    "stencil";

/* A spec over X[1] in [1, 2] for `lib`:`algo` from `start`. */
OptimizerSpec region_spec(const std::string& lib, const std::string& algo,
                          double start, double failscore = kMax)
{
    auto sp = spec_on({dim("X[1]", 1.0, 2.0)}, true, 0.2);
    sp.lib = lib;
    sp.algo = algo;
    sp.failscore = failscore;
    sp.stop.maxeval = 60;
    sp.optim_init = {{start}};
    return sp;
}

/* f(x, k) of the REAL x at the k-th call (from 0); every point and loss
 * recorded. */
struct RegionHost {
    std::function<double(double, std::size_t)> f;
    std::vector<double> asked;
    std::vector<double> losses;
    double operator()(const Vec& x) {
        const double y = f(x(0), asked.size());
        asked.push_back(x(0));
        losses.push_back(y);
        return y;
    }
    /* Whether a usable loss (below 1e300, every failure here is above)
     * was evaluated. */
    bool reached_usable(double below) const {
        for (double y : losses)
            if (y < below) return true;
        return false;
    }
};

/* The straddle of test_hpo_termination.cpp as a stub: 95 + 10 (2 - x)^2
 * from 1.47 up, the default FAILSCORE below. */
double straddle(double x, std::size_t)
{
    return x < 1.47 ? kMax : 95.0 + 10.0 * (2.0 - x) * (2.0 - x);
}

/* An outage: 1 + 10 (x - 1.2)^2 until the call `from` (counted from 0),
 * the default FAILSCORE at every call from it on, wherever the point. */
std::function<double(double, std::size_t)> outage(std::size_t from)
{
    return [from](double x, std::size_t k) {
        return k >= from ? kMax : 1.0 + 10.0 * (x - 1.2) * (x - 1.2);
    };
}

/* Runs the NLopt or the Ceres adapter on `host`; returns the status. */
std::string run_region(const OptimizerSpec& sp, RegionHost& host)
{
    if (sp.lib == "NLOPT") {
        tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host);
    } else {
#ifdef TADAH_USE_CERES
        const std::string& a = sp.algo;
        tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
            sp, host,
            a == "LBFGS" ? ceres::LBFGS
            : a == "WOLFE" ? ceres::BFGS
            : a == "STEEPEST_DESCENT" ? ceres::STEEPEST_DESCENT
                                      : ceres::NONLINEAR_CONJUGATE_GRADIENT,
            a == "STEEPEST_DESCENT" ? ceres::ARMIJO : ceres::WOLFE, a);
#endif
    }
    return HPOLog::instance().status();
}

}  // namespace

TEST_CASE("No usable gradient: every local NLopt LD_* search from a start "
          "inside the failure region stops there",
          "[HPO][grad_provider][fd_bounds][gradient_failure][nlopt]")
{
    /* The straddle stub from 1.25: the start and its stencil 1.45 and 1.05
     * fail.
     * Pins: one evaluation, the start, whose failed loss fails the
     * gradient; the search is stopped at its starting point with the stop
     * line "(the loss at X[1] = 1.25 is 1.797693135e+308); a local gradient
     * search has no step to take from a starting point without a usable
     * gradient" and the status "STOPPED -- no usable gradient: ...".
     * Before the fix: the gradient at the start was the zero difference of
     * two failures: LD_LBFGS, LD_TNEWTON and LD_VAR2 made 3 evaluations
     * (the start and its stencil) and ended "SUCCESS -- generic success",
     * LD_SLSQP 3 (roundoff-limited), LD_MMA and LD_CCSAQ 60 (the same three
     * points until MAXEVAL_REACHED); no stop line. */
    for (const char* algo : {"LD_LBFGS", "LD_MMA", "LD_CCSAQ", "LD_SLSQP",
                             "LD_TNEWTON", "LD_VAR2"}) {
        DYNAMIC_SECTION(algo) {
            Journal j;
            RegionHost host{straddle, {}, {}};
            const std::string status =
                run_region(region_spec("NLOPT", algo, 1.25), host);
            CHECK(host.asked.size() == 1u);
            CHECK(status == "STOPPED -- " + kNoUsableStencil);
            CHECK(count_of(j.text(),
                           std::string("NLOPT:") + algo + " stopped: " +
                               kNoUsableStencil +
                               " (the loss at X[1] = 1.25 is "
                               "1.797693135e+308); a local gradient search "
                               "has no step to take from a starting point "
                               "without a usable gradient") == 1u);
        }
    }
}

TEST_CASE("No usable gradient: MLSL and the global GD_* searches from a start "
          "inside the failure region go on to usable points",
          "[HPO][grad_provider][fd_bounds][gradient_failure][nlopt]")
{
    /* A global search is not one local search: a rejected start leaves it
     * the rest. The straddle stub from 1.25.
     * Pins (passes before the fix as well: a pin that the rule does not stop
     * the global searches): G_MLSL_LDS (POPULATION 3, INNER LD_LBFGS),
     * GD_MLSL (its default local search) and GD_STOGO reach a usable loss,
     * spend MAXEVAL 60 (MAXEVAL_REACHED) and write no stop line. */
    struct Case { std::string algo; bool inner; };
    for (const Case& c : {Case{"G_MLSL_LDS", true}, Case{"GD_MLSL", false},
                          Case{"GD_STOGO", false}}) {
        DYNAMIC_SECTION(c.algo) {
            Journal j;
            auto sp = region_spec("NLOPT", c.algo, 1.25);
            if (c.inner) {
                sp.pop = 3;
                sp.inner = std::make_unique<OptimizerSpec>(
                    region_spec("NLOPT", "LD_LBFGS", 1.25));
                sp.inner->stop.maxeval.reset();
            }
            RegionHost host{straddle, {}, {}};
            const std::string status = run_region(sp, host);
            CHECK(host.reached_usable(1e300));
            CHECK(host.asked.size() == 60u);
            CHECK(status == "MAXEVAL_REACHED -- maximum evaluations exhausted");
            CHECK(j.text().find("stopped: no usable gradient") ==
                  std::string::npos);
        }
    }
}

namespace {

/* The LD_* algorithms an INNER block of MLSL or AUGLAG can name (NLopt
 * 2.10; LD_LBFGS_NOCEDAL is refused). */
const std::vector<std::string> kLocalGradientAlgorithms = {
    "LD_LBFGS",   "LD_MMA",    "LD_CCSAQ",
    "LD_SLSQP",   "LD_TNEWTON", "LD_TNEWTON_RESTART",
    "LD_TNEWTON_PRECOND", "LD_TNEWTON_PRECOND_RESTART",
    "LD_VAR1",    "LD_VAR2",   "LD_AUGLAG", "LD_AUGLAG_EQ"};

/* A wall: the default FAILSCORE below 1.53, 1 + (x - 1.53) from it up, so
 * that the best usable points lie next to the failure region, where their
 * stencils fail. 1.53 is no stencil point of the points these cases ask
 * for (with the boundary on one, 1.5 for the start 1.9, a point an ulp
 * away fails where the point does not). */
double wall(double x, std::size_t)
{
    return x < 1.53 ? kMax : 1.0 + (x - 1.53);
}

/* The wall with a pit, failed too, on (1.85, 1.89): at 1.6875 the central
 * stencil 1.8875 and 1.4875 both fail, so the point has no usable side (a
 * difference never mixes a usable loss with a failure), where the wall
 * alone failed only 1.4875 and the gradient is one-sided into 1.8875 and
 * 2.0875. */
double wall_pit(double x, std::size_t k)
{
    return x > 1.85 && x < 1.89 ? kMax : wall(x, k);
}

/* A wall at 1.75: from the start 1.9 the one-sided stencil (1.7, 1.5) at
 * the upper bound fails at its near point, and the other side lies outside
 * the box: the start has no usable gradient. */
double high_wall(double x, std::size_t)
{
    return x < 1.75 ? kMax : 1.0 + (x - 1.75);
}

/* The gradient requests at `x` from the call `from` on: consecutive
 * (x, x + h, x - h) calls, the point and its central stencil. */
std::size_t requests_at(const std::vector<double>& asked, std::size_t from,
                        double x, double h)
{
    std::size_t n = 0;
    for (std::size_t i = from;
         i + 2 < asked.size() && asked[i] == Approx(x).margin(1e-12) &&
         asked[i + 1] == Approx(x + h).margin(1e-12) &&
         asked[i + 2] == Approx(x - h).margin(1e-12);
         i += 3)
        ++n;
    return n;
}

}  // namespace

TEST_CASE("No usable gradient: MLSL moves on from a rejected local start "
          "with every LD_* local search, and AUGLAG is stopped there",
          "[HPO][grad_provider][fd_bounds][gradient_failure][nlopt][mlsl]"
          "[auglag]")
{
    /* MLSL and the global GD_* searches are not stopped at a rejected
     * local start (the case above); this probes whether the local search
     * MLSL runs, any LD_* algorithm, moves on from one or asks for it again
     * and again (a stall that would spend the budget there). The wall stub
     * with a pit (wall_pit) from 1.9; MAXEVAL 60 (the enclosing block's,
     * which NLopt hands on).
     *
     * G_MLSL_LDS, POPULATION 3, with an INNER block of each LD_* algorithm,
     * and GD_MLSL_LDS without one (NLopt's default local search, LD_MMA):
     * MLSL samples 1.9 (the start), 1.6875, 1.9375 and 1.4375 (failed),
     * and starts its first local search at the best of them, 1.6875, whose
     * stencil points 1.8875 and 1.4875 both fail: the local start is
     * rejected.
     * Pins: the requests at 1.6875 (1.6875, 1.8875, 1.4875): 1 for LD_LBFGS,
     * LD_SLSQP, LD_TNEWTON*, LD_VAR1 and LD_VAR2, 2 for LD_MMA, LD_CCSAQ and
     * the default local search, 2 for LD_AUGLAG and LD_AUGLAG_EQ and then
     * AUGLAG's evaluation of the point its local search returned; no later
     * evaluation of 1.6875; a usable loss after them; MAXEVAL_REACHED after
     * exactly 60 evaluations; no stop line.
     *
     * AUGLAG and LD_AUGLAG with an INNER block of each LD_* algorithm run
     * one local search, from 1.9, whose one-sided stencil's near point 1.7
     * fails (high_wall), with no other side in the box.
     * Pins: they are stopped there: 2 evaluations (1.9, 1.7), the stop line
     * once and the status "STOPPED -- no usable gradient: ...".
     *
     * A difference never mixes a usable loss with a failure: next to the
     * plain wall, a point whose one
     * stencil side fails takes the other, and is no rejected start; these
     * stubs fail both sides, or the only side, to keep one.
     * Before the fix: MLSL's requests at 1.6875 were (1.6875, 1.8875) only,
     * the first failed stencil point ending the gradient (requests_at
     * counted none of the (x, x + h, x - h) triples), and the plain wall
     * gave the rejected start these cases pinned, AUGLAG's three
     * evaluations (1.9, 1.7, 1.5) included; with the plain wall the start's
     * gradient is now one-sided and AUGLAG ran to MAXEVAL. The 60
     * evaluations end before a local search reaches the edge of the failure
     * region; how MLSL ends there is the next case's. */
    struct MlslCase { std::string inner; std::size_t requests, reevaluations; };
    std::vector<MlslCase> mlsl;
    for (const auto& a : kLocalGradientAlgorithms) {
        const bool trust_region = a == "LD_MMA" || a == "LD_CCSAQ";
        const bool auglag = a.rfind("LD_AUGLAG", 0) == 0;
        mlsl.push_back({a, trust_region || auglag ? 2u : 1u, auglag ? 1u : 0u});
    }
    mlsl.push_back({"", 2u, 0u});   /* GD_MLSL_LDS, NLopt's default LD_MMA */
    for (const auto& c : mlsl) {
        const std::string name =
            c.inner.empty() ? "GD_MLSL_LDS" : "G_MLSL_LDS, INNER " + c.inner;
        DYNAMIC_SECTION(name) {
            Journal j;
            auto sp = region_spec("NLOPT",
                                  c.inner.empty() ? "GD_MLSL_LDS" : "G_MLSL_LDS",
                                  1.9);
            sp.pop = 3;
            if (!c.inner.empty()) {
                sp.inner = std::make_unique<OptimizerSpec>(
                    region_spec("NLOPT", c.inner, 1.9));
                sp.inner->stop.maxeval.reset();
            }
            RegionHost host{wall_pit, {}, {}};
            const std::string status = run_region(sp, host);
            const auto& x = host.asked;
            REQUIRE(x.size() > 4u);
            CHECK(x[0] == Approx(1.9));
            CHECK(x[1] == Approx(1.6875));
            CHECK(x[2] == Approx(1.9375));
            CHECK(x[3] == Approx(1.4375));
            CHECK(host.losses[3] == kMax);
            CHECK(requests_at(x, 4, 1.6875, 0.2) == c.requests);
            const std::size_t moved_on = 4 + 3 * c.requests + c.reevaluations;
            REQUIRE(x.size() > moved_on);
            CHECK(std::count_if(x.begin(), x.end(), [](double v) {
                      return v == Approx(1.6875).margin(1e-12);
                  }) == static_cast<long>(1 + c.requests + c.reevaluations));
            bool usable_after = false;
            for (std::size_t i = moved_on; i < x.size(); ++i)
                usable_after = usable_after || host.losses[i] < 1e300;
            CHECK(usable_after);
            CHECK(x.size() == 60u);
            CHECK(status == "MAXEVAL_REACHED -- maximum evaluations exhausted");
            CHECK(j.text().find("stopped: no usable gradient") ==
                  std::string::npos);
        }
    }
    for (const std::string outer : {"AUGLAG", "LD_AUGLAG"}) {
        for (const auto& inner : kLocalGradientAlgorithms) {
            DYNAMIC_SECTION(outer << ", INNER " << inner) {
                Journal j;
                auto sp = region_spec("NLOPT", outer, 1.9);
                sp.inner = std::make_unique<OptimizerSpec>(
                    region_spec("NLOPT", inner, 1.9));
                sp.inner->stop.maxeval.reset();
                RegionHost host{high_wall, {}, {}};
                const std::string status = run_region(sp, host);
                REQUIRE(host.asked.size() == 2u);
                CHECK(host.asked[0] == Approx(1.9));
                CHECK(host.asked[1] == Approx(1.7));
                CHECK(status == "STOPPED -- " + kNoUsableStencil);
                CHECK(count_of(j.text(),
                               "NLOPT:" + outer + " stopped: " +
                                   kNoUsableStencil +
                                   " (at X[1] = 1.9, the stencil point X[1] = "
                                   "1.7 failed: loss 1.797693135e+308); a local "
                                   "gradient search has no step to take from a "
                                   "starting point without a usable gradient") ==
                      1u);
            }
        }
    }
}

TEST_CASE("No usable gradient: MLSL ends when an LD_LBFGS, LD_TNEWTON* or "
          "LD_VAR* local search fails at the edge of the failure region, and "
          "goes on with the other LD_* local searches",
          "[HPO][grad_provider][fd_bounds][gradient_failure][nlopt][mlsl]")
{
    /* The wall stub from 1.9, MAXEVAL 400: G_MLSL_LDS (POPULATION 3) with
     * an INNER block of each LD_* algorithm, and GD_MLSL_LDS with the same
     * INNER blocks (the same runs). Within a step of the wall, 1.53, the
     * gradient is the one-sided difference into the usable side (a
     * difference never mixes the kinds), so
     * MLSL's local searches converge on the wall itself; the line search of
     * LD_LBFGS, LD_TNEWTON* and LD_VAR1/LD_VAR2 (NLopt's Luksan codes)
     * steps across it, to 1.529962214, whose loss fails, and NLopt returns
     * NLOPT_FAILURE; NLopt's MLSL ends its whole search when a local search
     * returns an error (mlsl.c: `if (lret < 0) { ret = lret; goto done; }`).
     * HPO cannot end one local search alone: a forced stop is an error to
     * MLSL as well. LD_MMA, LD_CCSAQ, LD_SLSQP and the LD_AUGLAG local
     * searches end theirs without an error, and MLSL goes on to MAXEVAL.
     * Pins: LD_LBFGS 87 evaluations, LD_TNEWTON and LD_TNEWTON_PRECOND 132,
     * LD_TNEWTON_RESTART and LD_TNEWTON_PRECOND_RESTART 125, LD_VAR1 and
     * LD_VAR2 98, the last of them 1.529962214, failed; the status
     * "STOPPED -- no usable gradient: ..." and the stop line once, "(the
     * loss at X[1] = 1.529962214 is 1.797693135e+308); NLopt ended right
     * after it (nlopt failure)". LD_MMA, LD_CCSAQ, LD_SLSQP, LD_AUGLAG and
     * LD_AUGLAG_EQ: 400 evaluations, MAXEVAL_REACHED, no stop line.
     * Before the fix: below 1.73 (the wall plus h) the stencil point x - h
     * failed and every request there was rejected, so the local searches
     * converged on 1.73 and ended after LD_LBFGS 141, LD_TNEWTON* 185 and
     * 179, LD_VAR* 170 evaluations, their last 8 requests the line search's
     * (x, x + h, x - h) rising towards 1.73, the stop line "(at X[1] =
     * 1.729564343, the stencil point X[1] = 1.529564343 failed: ...)". */
    struct Ending { std::size_t evals; bool early; };
    const auto ending = [](const std::string& a) {
        if (a == "LD_LBFGS") return Ending{87u, true};
        if (a == "LD_TNEWTON" || a == "LD_TNEWTON_PRECOND")
            return Ending{132u, true};
        if (a == "LD_TNEWTON_RESTART" || a == "LD_TNEWTON_PRECOND_RESTART")
            return Ending{125u, true};
        if (a == "LD_VAR1" || a == "LD_VAR2") return Ending{98u, true};
        return Ending{400u, false};
    };
    for (const std::string outer : {"G_MLSL_LDS", "GD_MLSL_LDS"}) {
        for (const auto& inner : kLocalGradientAlgorithms) {
            DYNAMIC_SECTION(outer << ", INNER " << inner) {
                Journal j;
                auto sp = region_spec("NLOPT", outer, 1.9);
                sp.stop.maxeval = 400;
                sp.pop = 3;
                sp.inner = std::make_unique<OptimizerSpec>(
                    region_spec("NLOPT", inner, 1.9));
                sp.inner->stop.maxeval.reset();
                RegionHost host{wall, {}, {}};
                const std::string status = run_region(sp, host);
                const std::string t = j.text();
                const auto& x = host.asked;
                const Ending e = ending(inner);
                CHECK(x.size() == e.evals);
                if (!e.early) {
                    CHECK(status ==
                          "MAXEVAL_REACHED -- maximum evaluations exhausted");
                    CHECK(t.find("stopped: no usable gradient") ==
                          std::string::npos);
                } else {
                    CHECK(status == "STOPPED -- " + kNoUsableStencil);
                    CHECK(count_of(t, "NLOPT:" + outer + " stopped: " +
                                          kNoUsableStencil +
                                          " (the loss at X[1] = 1.529962214 "
                                          "is 1.797693135e+308); NLopt ended "
                                          "right after it (nlopt failure)") ==
                          1u);
                    REQUIRE_FALSE(x.empty());
                    CHECK(x.back() == Approx(1.529962214).margin(1e-9));
                    CHECK(host.losses.back() == kMax);
                }
            }
        }
    }
}

TEST_CASE("No usable gradient: under FAILSCORE_GRADING a search from a graded "
          "failure follows the graded slope to usable points",
          "[HPO][grad_provider][fd_bounds][gradient_failure]"
          "[failscore_grading]")
{
    /* FAILSCORE 1e9: below 1.47 the graded band-D loss of v = 10 (1.47 -
     * x) (HPO_Host's grading of a skip cap or a gate by its excess), which
     * falls towards the usable side; 1 + (x - 1.8)^2 from 1.47 up.
     * Pins (the graded searches pass before the fix as well: a pin of the
     * graded slope, which a rule that failed every failed evaluation would
     * break): from 1.25 with FAILSCORE_GRADING ON, LD_LBFGS, LD_MMA and
     * CERES LBFGS reach a usable loss below 2 and end without the stencil
     * status or a stop line; with it OFF, LD_LBFGS stops at the start after
     * one evaluation.
     * Before the fix: the OFF section failed: the differences of the
     * failures were a slope all the same, and LD_LBFGS made 18 evaluations
     * and ended "SUCCESS -- generic success". */
    const auto graded = [](double x, std::size_t) {
        if (x < 1.47)
            return tadah::hpo::graded_failscore(
                1e9, tadah::hpo::FailBand::D, 10.0 * (1.47 - x), 1.0);
        return 1.0 + (x - 1.8) * (x - 1.8);
    };
    std::vector<std::pair<std::string, std::string>> backends = {
        {"NLOPT", "LD_LBFGS"}, {"NLOPT", "LD_MMA"}};
#ifdef TADAH_USE_CERES
    /* CERES:STEEPEST_DESCENT, the Ceres solver HPO runs (this case ran
     * CERES:LBFGS, refused since: a pin of the same rule). */
    backends.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    for (const auto& b : backends) {
        DYNAMIC_SECTION(b.first << ":" << b.second) {
            Journal j;
            auto sp = region_spec(b.first, b.second, 1.25, 1e9);
            sp.failscore_grading = true;
            RegionHost host{graded, {}, {}};
            const std::string status = run_region(sp, host);
            CHECK(host.losses.front() >= 0.999e9);
            CHECK(host.reached_usable(2.0));
            CHECK(status != "STOPPED -- " + kNoUsableStencil);
            CHECK(j.text().find("stopped: no usable gradient") ==
                  std::string::npos);
        }
    }
    SECTION("FAILSCORE_GRADING OFF") {
        Journal j;
        RegionHost host{graded, {}, {}};
        const std::string status =
            run_region(region_spec("NLOPT", "LD_LBFGS", 1.25, 1e9), host);
        CHECK(host.asked.size() == 1u);
        CHECK(status == "STOPPED -- " + kNoUsableStencil);
    }
}

TEST_CASE("No usable gradient: under FAILSCORE_GRADING NLopt is given the "
          "gradient of the stand-in of a graded failure",
          "[HPO][grad_provider][fd_bounds][failscore_grading][failscore]")
{
    /* NLopt is given a finite stand-in for a failed evaluation
     * (failed_loss_stand_in.h), and for a graded failure with a usable
     * gradient that gradient times the derivative of the stand-in in the
     * loss (FailedLossStandIn::slope): K / FAILSCORE before any usable loss
     * (K = 1e100 for a FAILSCORE above it), the margin over the width of
     * the failure band after usable losses. Below 1.47 the graded band-D
     * loss of v = 10 (1.47 - x), which falls towards the usable side.
     * Pins: (a) FAILSCORE 1e200, usable 1 + (x - 1.8)^2 from 1.47 up,
     * LD_LBFGS from 1.25, a graded failure: NLopt is given about 1e100 for
     * the start and a gradient of that scale, follows it to the usable side
     * and reaches the minimum, 1.8 (loss 1), ending "SUCCESS -- generic
     * success" after 16 evaluations. (b) FAILSCORE 1e9, usable 1 + (x -
     * 1.3)^2 from 1.47 up, LD_LBFGS from 1.9: its line search tries 1.0,
     * a graded failure after usable losses, whose gradient (the stencil
     * 1.2 and 1.4, graded too) it is given scaled with the stand-in, and
     * steps back to eval 7 at 1.84951119517.
     * Before the fix (each gradient given as the finite differences of the
     * loss, unscaled): (a) the gradient, about 1e197, was not the
     * derivative of the value NLopt was given, and LD_LBFGS ended at once,
     * "STOPPED -- nlopt failure (...)" after the start and its stencil, 3
     * evaluations; (b) eval 7 was at 1.891. */
    SECTION("before any usable loss, FAILSCORE 1e200") {
        const auto graded = [](double x, std::size_t) {
            if (x < 1.47)
                return tadah::hpo::graded_failscore(
                    1e200, tadah::hpo::FailBand::D, 10.0 * (1.47 - x), 1.0);
            return 1.0 + (x - 1.8) * (x - 1.8);
        };
        Journal j;
        auto sp = region_spec("NLOPT", "LD_LBFGS", 1.25, 1e200);
        sp.failscore_grading = true;
        RegionHost host{graded, {}, {}};
        const std::string status = run_region(sp, host);
        CHECK(host.losses.front() >= 0.999e200);
        CHECK(status == "SUCCESS -- generic success");
        CHECK(host.asked.size() == 16u);
        CHECK(std::any_of(host.asked.begin(), host.asked.end(),
                          [](double x) { return x == Approx(1.8); }));
        CHECK(host.reached_usable(1.0 + 1e-9));
    }
    SECTION("after usable losses, FAILSCORE 1e9") {
        const auto graded = [](double x, std::size_t) {
            if (x < 1.47)
                return tadah::hpo::graded_failscore(
                    1e9, tadah::hpo::FailBand::D, 10.0 * (1.47 - x), 1.0);
            return 1.0 + (x - 1.3) * (x - 1.3);
        };
        Journal j;
        auto sp = region_spec("NLOPT", "LD_LBFGS", 1.9, 1e9);
        sp.failscore_grading = true;
        RegionHost host{graded, {}, {}};
        run_region(sp, host);
        REQUIRE(host.asked.size() >= 7u);
        CHECK(host.losses[0] < 2.0);
        CHECK(host.asked[3] == Approx(1.0));
        CHECK(host.losses[3] >= 0.999e9);
        CHECK(host.losses[3] < 1e9);
        CHECK(host.asked[6] == Approx(1.84951119517).epsilon(1e-10));
    }
}

#ifdef TADAH_USE_CERES
TEST_CASE("No usable gradient: a CERES search from a start inside the "
          "failure region stops there",
          "[HPO][grad_provider][fd_bounds][gradient_failure][ceres]")
{
    /* The straddle stub from 1.25.
     * Pins: one evaluation; Ceres' first evaluation fails ("Initial cost
     * and jacobian evaluation failed"), which the stop line reports, and
     * the status is "STOPPED -- no usable gradient: ...".
     * Before the fix: 3 evaluations (the start and its stencil, a zero
     * difference of two failures), Ceres' CONVERGENCE at iteration 0, no
     * status and no stop line. */
    /* CERES:STEEPEST_DESCENT, the Ceres solver HPO runs (the case also
     * ran CERES:LBFGS, WOLFE and NLCG, refused since). */
    for (const char* algo : {"STEEPEST_DESCENT"}) {
        DYNAMIC_SECTION(algo) {
            Journal j;
            RegionHost host{straddle, {}, {}};
            const std::string status =
                run_region(region_spec("CERES", algo, 1.25), host);
            CHECK(host.asked.size() == 1u);
            CHECK(status == "STOPPED -- " + kNoUsableStencil);
            CHECK(count_of(j.text(),
                           std::string("CERES:") + algo + " stopped: " +
                               kNoUsableStencil +
                               " (the loss at X[1] = 1.25 is "
                               "1.797693135e+308); Ceres ended right after it "
                               "(FAILURE: Initial cost and jacobian "
                               "evaluation failed.)") == 1u);
        }
    }
}
#endif

TEST_CASE("No usable gradient: an NLopt search that ends right after it "
          "rejected a point says so, on an outage",
          "[HPO][grad_provider][fd_bounds][gradient_failure][nlopt][auglag]")
{
    /* An outage: 1 + 10 (x - 1.2)^2 until the 16th call, the default
     * FAILSCORE from the 17th on. From 1.6 LD_SLSQP converges on 1.2 and
     * asks for it again and again; the outage fails it there, and each
     * request is rejected (its loss failed, so has its gradient) until the
     * search ends right after a rejection. These are the endings of the
     * straddle cases that a zero gradient between two failures used to
     * make (test_hpo_termination.cpp, "No usable gradient: the straddle
     * searches that ended right after a rejection go on to MAXEVAL"): the
     * search ends on the rejection with neither MAXEVAL, MAXTIME nor
     * STOPVAL met, and a tolerance reached there is the rejection's ending.
     * Pins: LD_SLSQP 23 evaluations, the last 7 failed, ending with the stop
     * line "(the loss at X[1] = 1.2 is 1.797693135e+308); NLopt ended right
     * after it (nlopt roundoff-limited)"; with XTOL_REL 1e-3 and FTOL_REL
     * 1e-6 "(XTOL_REACHED -- parameter tolerance converged)"; as the INNER
     * search of AUGLAG 24 evaluations (AUGLAG's evaluation of the point its
     * search returned the 24th) and "(FTOL_REACHED -- function tolerance
     * converged)"; an INNER MAXEVAL 15 (NLopt calls) ends the search on
     * that criterion at its last rejection, FTOL_REACHED and no stop line,
     * and an INNER MAXEVAL 16 is not reached: the rejection ends it, as
     * without one; both 24 evaluations.
     * Before the fix: the outage's failures came with a zero gradient (the
     * point and its stencil all fail) and were not rejected: LD_SLSQP made
     * 25 evaluations and ended roundoff-limited (with the tolerances
     * XTOL_REACHED), AUGLAG 26 and FTOL_REACHED, with either INNER MAXEVAL
     * as well; no stop line. */
    const auto tolerances = [](OptimizerSpec& s) {
        s.stop.xtol_rel = 1e-3;
        s.stop.ftol_rel = 1e-6;
    };
    const auto auglag = [&](std::optional<long> inner_maxeval) {
        auto sp = region_spec("NLOPT", "AUGLAG", 1.6);
        sp.inner = std::make_unique<OptimizerSpec>(
            region_spec("NLOPT", "LD_SLSQP", 1.6));
        sp.inner->stop.maxeval.reset();
        if (inner_maxeval) sp.inner->stop.maxeval = *inner_maxeval;
        tolerances(*sp.inner);
        return sp;
    };
    const std::string stop = "stopped: " + kNoUsableStencil +
                             " (the loss at X[1] = 1.2 is 1.797693135e+308); "
                             "NLopt ended right after it (";
    struct Case {
        std::string label;
        std::function<OptimizerSpec()> spec;
        std::size_t evals;
        std::string ended;   /* "": no stop line */
        std::string status;
    };
    const std::string fs = "STOPPED -- " + kNoUsableStencil;
    const std::vector<Case> cases = {
        {"LD_SLSQP", [] { return region_spec("NLOPT", "LD_SLSQP", 1.6); }, 23u,
         "nlopt roundoff-limited", fs},
        {"LD_SLSQP, XTOL_REL and FTOL_REL",
         [&] {
             auto sp = region_spec("NLOPT", "LD_SLSQP", 1.6);
             tolerances(sp);
             return sp;
         },
         23u, "XTOL_REACHED -- parameter tolerance converged", fs},
        {"AUGLAG, INNER LD_SLSQP", [&] { return auglag(std::nullopt); }, 24u,
         "FTOL_REACHED -- function tolerance converged", fs},
        {"AUGLAG, INNER MAXEVAL 15", [&] { return auglag(15); }, 24u, "",
         "FTOL_REACHED -- function tolerance converged"},
        {"AUGLAG, INNER MAXEVAL 16", [&] { return auglag(16); }, 24u,
         "FTOL_REACHED -- function tolerance converged", fs}};
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.label) {
            Journal j;
            const OptimizerSpec sp = c.spec();
            RegionHost host{outage(16), {}, {}};
            const std::string status = run_region(sp, host);
            CHECK(host.asked.size() == c.evals);
            CHECK(status == c.status);
            const std::string t = j.text();
            if (c.ended.empty()) {
                CHECK(t.find("stopped: no usable gradient") == std::string::npos);
            } else {
                CHECK(count_of(t, "NLOPT:" + sp.algo + " " + stop + c.ended +
                                      ")") == 1u);
            }
        }
    }
}

#ifdef TADAH_USE_CERES
TEST_CASE("No usable gradient: a Ceres search that ends with FAILURE right "
          "after it rejected a point says so, on an outage",
          "[HPO][grad_provider][fd_bounds][gradient_failure][ceres]")
{
    /* The outage of the NLopt case from the 7th call on. Ceres' line
     * search steps back from a point without a usable gradient, and ends
     * with FAILURE when it finds no valid step: from 1.6
     * CERES:STEEPEST_DESCENT, the Ceres solver HPO runs, ends "Numerical
     * failure in line search" after the outage. These are the endings of
     * the straddle cases that a zero gradient between two failures used to
     * make (test_hpo_termination.cpp, "No usable gradient: the straddle
     * searches that ended right after a rejection go on to MAXEVAL").
     * Pins: 27 evaluations, the stop line "(the loss at X[1] = 1.000000005
     * is 1.797693135e+308); Ceres ended right after it (FAILURE: Numerical
     * failure in line search", the status "STOPPED -- no usable gradient:
     * ...".
     * Before Ceres was given a failed evaluation's loss scaled to at most
     * 1e100: the stop line named X[1] = 1.000001376.
     * Before the fix: the case ran CERES:LBFGS, WOLFE and NLCG, refused
     * since (their Wolfe line search stalls at a point outside the OPTIM
     * box): from 1.6 they ended at 1.2 after 7 evaluations, and from 1.95
     * they left the box and ended "Cost and jacobian evaluation failed."
     * after 27. */
    for (const char* algo : {"STEEPEST_DESCENT"}) {
        DYNAMIC_SECTION(algo << " from 1.6") {
            Journal j;
            RegionHost host{outage(6), {}, {}};
            const std::string status =
                run_region(region_spec("CERES", algo, 1.6), host);
            CHECK(host.asked.size() == 27u);
            CHECK(status == "STOPPED -- " + kNoUsableStencil);
            CHECK(count_of(j.text(),
                           std::string("CERES:") + algo + " stopped: " +
                               kNoUsableStencil +
                               " (the loss at X[1] = 1.000000005 is "
                               "1.797693135e+308); Ceres ended right after it "
                               "(FAILURE: Numerical failure in line "
                               "search") == 1u);
        }
    }
}
#endif

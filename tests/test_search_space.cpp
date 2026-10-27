/* SearchSpace -- the owner of the optimiser coordinate system. */
#include "catch2/catch.hpp"
#include <tadah/hpo/search_space.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace tadah::hpo;

namespace {

DimSpec dim(const char* label, double lo, double hi, ScaleKind k = ScaleKind::Linear,
            double c = 1.0)
{
    DimSpec d;
    d.key = label; d.label = label; d.lo = lo; d.hi = hi;
    d.tf.kind = k; d.tf.c = c;
    return d;
}

std::vector<DimSpec> three()
{
    return { dim("LIN", -2.0, 6.0),
             dim("LOG", 0.1, 10.0, ScaleKind::Log),
             dim("ASINH", -100.0, 100.0, ScaleKind::Asinh, 0.01) };
}

}  // namespace

TEST_CASE("SearchSpace: round trips and bounds in both policies",
          "[HPO][search_space]")
{
    for (bool normalise : {false, true}) {
        ScalePolicy pol; pol.normalise = normalise;
        const auto ss = SearchSpace::build(three(), pol);
        REQUIRE(ss.dim() == 3);
        CHECK(ss.normalised() == normalise);
        for (std::size_t i = 0; i < 3; ++i) {
            const double lo = ss.at(i).lo, hi = ss.at(i).hi;
            for (double x : {lo, 0.5 * (lo + hi), hi, lo + 0.1 * (hi - lo)}) {
                const double u = ss.to_internal(i, x);
                CHECK(ss.to_real(i, u) == Approx(x).epsilon(1e-12).margin(1e-12));
                CHECK(u >= ss.internal_lo(i) - 1e-12);
                CHECK(u <= ss.internal_hi(i) + 1e-12);
            }
            if (normalise) {
                CHECK(ss.internal_lo(i) == 0.0);
                CHECK(ss.internal_hi(i) == 1.0);
                CHECK(ss.to_internal(i, lo) == Approx(0.0).margin(1e-15));
                CHECK(ss.to_internal(i, hi) == Approx(1.0).margin(1e-15));
            }
        }
        if (!normalise) {
            /* NONE reproduces the legacy internal coordinates exactly */
            CHECK(ss.internal_lo(0) == -2.0);
            CHECK(ss.internal_lo(1) == Approx(std::log(0.1)));
            CHECK(ss.internal_hi(1) == Approx(std::log(10.0)));
            CHECK(ss.to_internal(1, 1.0) == Approx(0.0).margin(1e-15));
        }
    }
}

TEST_CASE("SearchSpace: jacobian matches a central finite difference",
          "[HPO][search_space]")
{
    for (bool normalise : {false, true}) {
        ScalePolicy pol; pol.normalise = normalise;
        const auto ss = SearchSpace::build(three(), pol);
        for (std::size_t i = 0; i < 3; ++i) {
            for (double f : {0.05, 0.5, 0.95}) {
                const double u = ss.internal_lo(i) + f * ss.internal_width(i);
                const double h = 1e-6 * ss.internal_width(i);
                const double fd = (ss.to_real(i, u + h) - ss.to_real(i, u - h)) / (2 * h);
                CHECK(ss.dreal_dinternal(i, u) == Approx(fd).epsilon(1e-6));
            }
        }
    }
}

TEST_CASE("SearchSpace: ASINH is monotone with unit slope near zero",
          "[HPO][search_space]")
{
    DimTransform t; t.kind = ScaleKind::Asinh; t.c = 0.01;
    CHECK(t.fwd(0.0) == 0.0);
    CHECK(t.dinv(0.0) == Approx(0.01));           /* dx/dt = c at 0 */
    double prev = t.fwd(-100.0);
    for (double x = -99.0; x <= 100.0; x += 1.0) {
        const double v = t.fwd(x);
        CHECK(v > prev);
        prev = v;
    }
    CHECK(t.inv(t.fwd(37.5)) == Approx(37.5));
}

TEST_CASE("SearchSpace: LOG_HP rule and explicit tags",
          "[HPO][search_space]")
{
    ScalePolicy pol; pol.log_ratio = 100.0;
    auto dims = std::vector<DimSpec>{
        dim("A", 0.01, 10.0),                    /* ratio 1000 -> auto LOG */
        dim("B", 1.0, 50.0),                     /* ratio 50 -> LIN */
        dim("C", -1.0, 1000.0),                  /* lo <= 0 -> LIN */
        dim("D", 0.01, 10.0, ScaleKind::Asinh, 1.0) /* explicit tag wins */
    };
    const auto ss = SearchSpace::build(dims, pol);
    CHECK(ss.at(0).tf.kind == ScaleKind::Log);
    CHECK(ss.at(0).auto_log);
    CHECK(ss.describe_dim(0) == "LOG (auto)");
    CHECK(ss.at(1).tf.kind == ScaleKind::Linear);
    CHECK(ss.describe_dim(1) == "LIN");
    CHECK(ss.at(2).tf.kind == ScaleKind::Linear);
    CHECK(ss.at(3).tf.kind == ScaleKind::Asinh);
    CHECK(ss.describe_dim(3).rfind("ASINH c=", 0) == 0);
}

TEST_CASE("SearchSpace: build errors name the dimension",
          "[HPO][search_space]")
{
    ScalePolicy pol;
    REQUIRE_THROWS_WITH(SearchSpace::build({dim("X", 1.0, 1.0)}, pol),
                        Catch::Contains("OPTIM X") && Catch::Contains("high > low"));
    REQUIRE_THROWS_WITH(SearchSpace::build({dim("Y", -1.0, 1.0, ScaleKind::Log)}, pol),
                        Catch::Contains("OPTIM Y") && Catch::Contains("LOG scale requires low > 0"));
    REQUIRE_THROWS_WITH(SearchSpace::build({dim("Z", -1.0, 1.0, ScaleKind::Asinh, 0.0)}, pol),
                        Catch::Contains("OPTIM Z") && Catch::Contains("positive linear scale"));
    auto d = dim("I", 0.0, 3.0); d.is_int = true;
    ScalePolicy norm; norm.normalise = true;
    REQUIRE_THROWS_WITH(SearchSpace::build({d}, norm), Catch::Contains("integer"));
    REQUIRE_NOTHROW(SearchSpace::build({d}, pol));
}

TEST_CASE("SearchSpace: vector maps on std::vector and dlib::matrix",
          "[HPO][search_space]")
{
    ScalePolicy pol; pol.normalise = true;
    const auto ss = SearchSpace::build(three(), pol);
    const std::vector<double> x = {1.0, 1.0, 5.0};
    const auto u = ss.to_internal(x);
    REQUIRE(u.size() == 3);
    const auto back = ss.to_real(u);
    for (std::size_t i = 0; i < 3; ++i) CHECK(back[i] == Approx(x[i]));

    dlib::matrix<double, 0, 1> m(3);
    m(0) = 1.0; m(1) = 1.0; m(2) = 5.0;
    const auto um = ss.to_internal(m);
    for (std::size_t i = 0; i < 3; ++i) CHECK(um(static_cast<long>(i)) == Approx(u[i]));
    const auto bm = ss.to_real(um);
    for (std::size_t i = 0; i < 3; ++i) CHECK(bm(static_cast<long>(i)) == Approx(x[i]));

    /* chain rule: g_u = g_x * dx/du */
    std::vector<double> g_real = {2.0, 3.0, 4.0};
    const auto g_u = ss.pull_gradient(u, g_real);
    for (std::size_t i = 0; i < 3; ++i)
        CHECK(g_u[i] == Approx(g_real[i] * ss.dreal_dinternal(i, u[i])));
}

TEST_CASE("SearchSpace: require_inside_real, with_boxes",
          "[HPO][search_space]")
{
    ScalePolicy pol; pol.normalise = true;
    const auto ss = SearchSpace::build(three(), pol);
    REQUIRE_NOTHROW(ss.require_inside_real({6.0, 0.1, 0.0}, "seed"));
    REQUIRE_THROWS_WITH(ss.require_inside_real({6.1, 0.1, 0.0}, "seed"),
                        Catch::Contains("LIN") && Catch::Contains("outside its OPTIM box"));
    REQUIRE_THROWS_WITH(ss.require_inside_real({6.0, 0.1}, "seed"),
                        Catch::Contains("expected 3"));

    const auto grown = ss.with_boxes({{-2.0, 8.0}, {0.1, 10.0}, {-100.0, 100.0}});
    CHECK(grown.version() == 2);
    CHECK(grown.at(0).hi == 8.0);
    CHECK(grown.at(0).origin == BoxOrigin::Adapted);
    CHECK(grown.at(1).origin == BoxOrigin::Explicit);
    CHECK(grown.at(1).tf.kind == ScaleKind::Log);
}

/* ---------------------------------------------------------------------
 * Seed anchors: a REAL seed evaluates at exactly its loaded double.
 * ------------------------------------------------------------------- */
namespace {

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

/* The coordinate system of a space: dimensions, policy, version and, per
 * dimension, its scale, real box and internal box. Seed anchors are not
 * part of it. */
bool same_coordinates(const tadah::hpo::SearchSpace& a,
                      const tadah::hpo::SearchSpace& b)
{
    if (a.dim() != b.dim() || a.normalised() != b.normalised() ||
        a.version() != b.version())
        return false;
    for (std::size_t i = 0; i < a.dim(); ++i)
        if (a.describe_dim(i) != b.describe_dim(i) ||
            !same_bits(a.at(i).lo, b.at(i).lo) ||
            !same_bits(a.at(i).hi, b.at(i).hi) ||
            !same_bits(a.internal_lo(i), b.internal_lo(i)) ||
            !same_bits(a.internal_hi(i), b.internal_hi(i)))
            return false;
    return true;
}

struct AnchorCase { DimSpec d; double x0; };

/* One seed per box kind whose plain x -> u -> x round trip under AUTO is
 * NOT bit-exact (asserted below, so the anchors are never vacuous): an
 * explicit wide box on small weights, an AUTO FRAC 0
 * FLOOR 0.5 box on a grid centre, LOG boxes and an ASINH box. */
std::vector<AnchorCase> anchor_cases()
{
    const double c0 = 0.100000000000000006;
    return {
        {dim("FIXWEIGHT[13]", -300.0, 300.0), -0.000287486722631648},
        {dim("FIXWEIGHT[3]",  -300.0, 300.0), 108.326725555511771},
        {dim("FIXWEIGHT[11]", -300.0, 300.0), -5.55896250563770522},
        {dim("FIXWEIGHT[2]",  -300.0, 300.0), 1.2345678901234568e-14},
        {dim("FIXWEIGHT[4]",  -300.0, 300.0), 1e-20},
        {dim("FIXWEIGHT[5]",  -300.0, 300.0), -0.0},
        {dim("CGRID2B[1]", c0 - 0.5, c0 + 0.5), c0},
        {dim("SGRID2B[3]", 0.5, 2.0, ScaleKind::Log), 1.2},
        {dim("RCUT2B", 4.0, 6.3, ScaleKind::Log), 4.01370204427021893},
        {dim("AMP[1]", -100.0, 100.0, ScaleKind::Asinh, 0.01), 37.5},
        {dim("AMP[2]", -100.0, 100.0, ScaleKind::Asinh, 0.01), 1e-5},
    };
}

}  // namespace

TEST_CASE("SearchSpace: seed anchors return every seed bitwise",
          "[HPO][search_space][seed_anchor]")
{
    for (bool normalise : {true, false}) {
        CAPTURE(normalise);
        ScalePolicy pol; pol.normalise = normalise;
        std::size_t plain_inexact = 0;
        for (const auto& c : anchor_cases()) {
            CAPTURE(c.d.label, c.x0);
            const auto plain = SearchSpace::build({c.d}, pol);
            const auto anch  = plain.with_seed_anchors({{c.x0}});
            const double u0 = plain.to_internal(0, c.x0);
            CHECK(same_bits(anch.to_internal(0, c.x0), u0));
            const bool exact = same_bits(plain.to_real(0, u0), c.x0);
            if (!exact) ++plain_inexact;
            /* the evaluated value IS the seed; every in-box seed coordinate
             * is anchored (an exact one too: see the window test below) */
            CHECK(same_bits(anch.to_real(0, u0), c.x0));
            CHECK(anch.anchor_count() == 1u);
            /* the coordinate system itself is unchanged */
            CHECK(same_coordinates(anch, plain));
            CHECK(anch.version() == plain.version());
            CHECK(anch.dreal_dinternal(0, u0) == plain.dreal_dinternal(0, u0));
        }
        if (normalise) CHECK(plain_inexact == anchor_cases().size());
        else           CHECK(plain_inexact >= 1u);   // LOG/ASINH: exp/log rounding
    }
}

TEST_CASE("SearchSpace: points other than an anchored seed keep the plain map",
          "[HPO][search_space][seed_anchor]")
{
    ScalePolicy pol; pol.normalise = true;
    for (const auto& c : anchor_cases()) {
        CAPTURE(c.d.label, c.x0);
        const auto plain = SearchSpace::build({c.d}, pol);
        const auto anch  = plain.with_seed_anchors({{c.x0}});
        REQUIRE(anch.anchor_count() == 1u);
        const double u0 = plain.to_internal(0, c.x0);
        const double tol = anch.seed_anchor_tolerance(0, u0);
        for (double u : {u0 - 5 * tol, u0 + 5 * tol,
                         0.0, 0.25, 0.5, 0.75, 1.0, 0.1 * u0, u0 + 1e-9}) {
            if (std::fabs(u - u0) <= tol) continue;     // the anchor window
            CAPTURE(u);
            CHECK(same_bits(anch.to_real(0, u), plain.to_real(0, u)));
        }
        /* real_step_at measures from the seed itself at u0 */
        CHECK(anch.real_step_at(0, u0, 0.01) == plain.to_real(0, u0 + 0.01) - c.x0);
    }
}

/* NLopt BOBYQA/COBYLA (u -> (u/s)*s) and PRAXIS (atanh/tanh start map)
 * evaluate a seed a few ulp away from u0. Within kSeedAnchorUlps ulp of the
 * internal box's magnitude the seed is still evaluated exactly; from
 * kSeedAnchorUlps + 1 ulp on, the plain map applies. */
TEST_CASE("SearchSpace: a u within kSeedAnchorUlps ulp of the box maps to the "
          "seed", "[HPO][search_space][seed_anchor]")
{
    const double inf = std::numeric_limits<double>::infinity();
    REQUIRE(kSeedAnchorUlps == 4);
    std::vector<AnchorCase> cases = anchor_cases();
    /* a seed near the lower bound (u0 ~ 1e-10): PRAXIS's start rounding
     * there is ~1e6 ulp(u0), but < 1 ulp of the unit box */
    cases.push_back({dim("FIXWEIGHT[1]", -300.0, 300.0), -299.99999994});
    /* a seed whose plain round trip is exact still needs the window */
    cases.push_back({dim("FIXWEIGHT[2]", -300.0, 300.0), -216.764741590923819});
    for (bool normalise : {true, false}) {
        CAPTURE(normalise);
        ScalePolicy pol; pol.normalise = normalise;
        for (const auto& c : cases) {
            CAPTURE(c.d.label, c.x0);
            const auto plain = SearchSpace::build({c.d}, pol);
            const auto anch  = plain.with_seed_anchors({{c.x0}});
            REQUIRE(anch.anchor_count() == 1u);
            const double u0 = plain.to_internal(0, c.x0);
            const double m = std::max(std::fabs(plain.internal_lo(0)),
                                      std::fabs(plain.internal_hi(0)));
            const double ulp_box = std::nextafter(m, inf) - m;
            CHECK(anch.seed_anchor_tolerance(0, u0) == 4 * ulp_box);
            if (normalise) CHECK(ulp_box == std::nextafter(1.0, inf) - 1.0);
            std::size_t moved = 0;   // the plain map would NOT give x0 back
            for (int j = 1; j <= 4; ++j)
                for (double sign : {-1.0, 1.0}) {
                    double u = u0 + sign * j * ulp_box;
                    /* u0 + j*ulp can round when |u0| << ulp_box (e.g. 1e-20
                     * under NONE); step back onto the window if it did */
                    while (std::fabs(u - u0) > j * ulp_box)
                        u = std::nextafter(u, u0);
                    CAPTURE(j, sign, u);
                    CHECK(same_bits(anch.to_real(0, u), c.x0));
                    if (!same_bits(plain.to_real(0, u), c.x0)) ++moved;
                    /* the vector map agrees with the scalar one */
                    CHECK(same_bits(anch.to_real(std::vector<double>{u})[0], c.x0));
                }
            CHECK(moved > 0u);        // the window is not vacuous
            for (int j : {5, 6, 8, 1000})
                for (double sign : {-1.0, 1.0}) {
                    const double u = u0 + sign * j * ulp_box;
                    CAPTURE(j, sign, u);
                    CHECK(same_bits(anch.to_real(0, u), plain.to_real(0, u)));
                }
        }
    }
}

TEST_CASE("SearchSpace: within the anchor window the nearest seed wins; an "
          "exact u0 always wins", "[HPO][search_space][seed_anchor]")
{
    ScalePolicy pol; pol.normalise = true;
    const auto one = SearchSpace::build({dim("W", -300.0, 300.0)}, pol);
    const double inf = std::numeric_limits<double>::infinity();
    const double ulp1 = std::nextafter(1.0, inf) - 1.0;
    /* two seeds whose u differ by 3 ulp(1): their windows overlap */
    const double xa = 12.345678901234567;
    const double xb = xa + 4e-13;
    const double ua = one.to_internal(0, xa);
    const double ub = one.to_internal(0, xb);
    REQUIRE(ub - ua == 3 * ulp1);
    const auto anch = one.with_seed_anchors({{xa}, {xb}});
    REQUIRE(anch.anchor_count() == 2u);
    CHECK(same_bits(anch.to_real(0, ua), xa));               // exact
    CHECK(same_bits(anch.to_real(0, ub), xb));               // exact
    CHECK(same_bits(anch.to_real(0, ua - ulp1), xa));        // nearer a
    CHECK(same_bits(anch.to_real(0, ua + ulp1), xa));        // nearer a
    CHECK(same_bits(anch.to_real(0, ub - ulp1), xb));        // nearer b
    CHECK(same_bits(anch.to_real(0, ub + 4 * ulp1), xb));    // only b's window
    CHECK(same_bits(anch.to_real(0, ua - 4 * ulp1), xa));    // only a's window
    CHECK(same_bits(anch.to_real(0, ub + 5 * ulp1), one.to_real(0, ub + 5 * ulp1)));
}

/* Under OPTIM_SCALE NONE a LIN dimension maps u = x, and NLopt
 * BOBYQA/COBYLA move a start by up to 1 ulp(u0) (u -> (u/s)*s). With the
 * window capped at 2^-27 for every anchor, a seed with |x0| >= 2^26 was
 * evaluated 1 ulp off. Each anchor's window is at least 2 ulp of its own
 * u0, so such a seed stays anchored, while a small seed in the same wide
 * box keeps the capped window (the Ceres step sqrt(eps) stays outside). */
TEST_CASE("SearchSpace: a large seed under NONE is anchored within 2 ulp of "
          "itself; a small one keeps the capped window",
          "[HPO][search_space][seed_anchor]")
{
    const double inf = std::numeric_limits<double>::infinity();
    auto ulp = [&](double v) { v = std::fabs(v); return std::nextafter(v, inf) - v; };
    ScalePolicy pol; pol.normalise = false;
    struct Case { double lo, hi, x0; };
    for (const Case& c : {Case{0.0, 0x1p30, 84139161.677173063},
                          Case{0.0, 2e8, 7e7}, Case{0.0, 2e8, 1e8},
                          Case{-1e9, 1e9, 3e8}, Case{-1e9, 1e9, -3e8},
                          Case{0.0, 2e8, 4e7}}) {
        CAPTURE(c.lo, c.hi, c.x0);
        const auto plain = SearchSpace::build({dim("W", c.lo, c.hi)}, pol);
        const auto anch = plain.with_seed_anchors({{c.x0}, {1.0}});
        REQUIRE(anch.anchor_count() == 2u);
        const double u0 = plain.to_internal(0, c.x0);
        REQUIRE(u0 == c.x0);                            // NONE: identity
        const double tol = anch.seed_anchor_tolerance(0, u0);
        CHECK(tol == std::min(4 * ulp(std::max(std::fabs(c.lo), std::fabs(c.hi))),
                              std::max(kSeedAnchorMaxTol, 2 * ulp(u0))));
        CHECK(tol >= ulp(u0));
        for (double dir : {-inf, inf}) {
            const double u = std::nextafter(u0, dir);   // BOBYQA's rescaled start
            CAPTURE(u);
            CHECK(same_bits(anch.to_real(0, u), c.x0));
        }
        /* the small seed in the same box: the capped window, and the Ceres
         * kernel's smallest step maps plainly */
        CHECK(anch.seed_anchor_tolerance(0, 1.0) == kSeedAnchorMaxTol);
        const double step = std::sqrt(std::numeric_limits<double>::epsilon());
        CHECK(same_bits(anch.to_real(0, 1.0), 1.0));
        CHECK(anch.to_real(0, 1.0 + step) == 1.0 + step);
        CHECK(anch.to_real(0, 1.0 - step) == 1.0 - step);
        CHECK(same_bits(anch.to_real(0, 1.0 + 0.5 * kSeedAnchorMaxTol), 1.0));
    }
}

/* A run seeded with many points (an INIT chain, LHS/SOBOL N) anchors every
 * seed; above 24 anchors per dimension to_real binary-searches them. The
 * result must equal a brute-force "the nearest anchor (the lower one on a
 * tie) when u is within its window, else the plain map" for seeds, their
 * windows, points between them and points outside the anchors' range. */
TEST_CASE("SearchSpace: many seed anchors (binary-search path) agree with a "
          "brute-force nearest search", "[HPO][search_space][seed_anchor]")
{
    const double inf = std::numeric_limits<double>::infinity();
    for (bool normalise : {true, false}) {
        CAPTURE(normalise);
        ScalePolicy pol; pol.normalise = normalise;
        const auto plain = SearchSpace::build({dim("W", -300.0, 300.0)}, pol);
        for (std::size_t nseed : {std::size_t{5}, std::size_t{24},
                                  std::size_t{25}, std::size_t{200}}) {
            CAPTURE(nseed);
            std::vector<std::vector<double>> seeds;
            for (std::size_t k = 0; k < nseed; ++k)   // irrational spacing
                seeds.push_back({-299.9 + 599.8 * std::fmod(0.6180339887498949 *
                                                             static_cast<double>(k + 1), 1.0)});
            seeds.push_back({seeds.front()[0]});    // a duplicate: one anchor
            const auto anch = plain.with_seed_anchors(seeds);
            REQUIRE(anch.anchor_count() == nseed);
            std::vector<std::pair<double, double>> ux;   // (u0, x0), by u0
            for (std::size_t k = 0; k < nseed; ++k)
                ux.emplace_back(plain.to_internal(0, seeds[k][0]), seeds[k][0]);
            std::sort(ux.begin(), ux.end());
            auto brute = [&](double u) {     // ascending: '<' keeps the lower
                double best_d = inf, u_near = 0.0, x = 0.0;
                for (const auto& p : ux) {
                    const double d = std::fabs(u - p.first);
                    if (d < best_d) { best_d = d; u_near = p.first; x = p.second; }
                }
                return best_d <= anch.seed_anchor_tolerance(0, u_near)
                           ? x : plain.to_real(0, u);
            };
            std::vector<double> probes = {-inf, -1.0, 0.0, 0.5, 1.0, 2.0,
                                          plain.internal_lo(0), plain.internal_hi(0)};
            for (const auto& p : ux) {
                const double u0 = p.first;
                const double tol = anch.seed_anchor_tolerance(0, u0);
                for (int j : {-5, -4, -1, 0, 1, 3, 4, 5, 1000})
                    probes.push_back(u0 + j * tol / 4);
                probes.push_back(std::nextafter(u0, inf));
            }
            for (double u : probes) {
                CAPTURE(u);
                CHECK(same_bits(anch.to_real(0, u), brute(u)));
            }
            for (const auto& p : ux)
                CHECK(same_bits(anch.to_real(0, p.first), p.second));
            CHECK(std::isnan(anch.to_real(0, std::numeric_limits<double>::quiet_NaN())));
        }
    }
}

TEST_CASE("SearchSpace: seed anchors on vectors and several seeds; edge cases",
          "[HPO][search_space][seed_anchor]")
{
    ScalePolicy pol; pol.normalise = true;
    std::vector<DimSpec> dims;
    std::vector<double> seed;
    for (const auto& c : anchor_cases()) { dims.push_back(c.d); seed.push_back(c.x0); }
    const auto plain = SearchSpace::build(dims, pol);
    const std::size_t n = plain.dim();

    SECTION("vector maps: std::vector and dlib::matrix") {
        const auto anch = plain.with_seed_anchors({seed});
        CHECK(anch.anchor_count() == n);
        const auto back = anch.to_real(anch.to_internal(seed));
        dlib::matrix<double, 0, 1> m(static_cast<long>(n));
        for (std::size_t i = 0; i < n; ++i) m(static_cast<long>(i)) = seed[i];
        const auto bm = anch.to_real(anch.to_internal(m));
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(same_bits(back[i], seed[i]));
            CHECK(same_bits(bm(static_cast<long>(i)), seed[i]));
        }
    }
    SECTION("the first seed to reach a u owns it") {
        /* 0.0 (exact), 1e-20 and 1.2345678901234568e-14 all map to u = 0.5
         * in [-300, 300]; they are closer than the box resolves. */
        const DimSpec d = dim("W", -300.0, 300.0);
        const auto one = SearchSpace::build({d}, pol);
        const double u = one.to_internal(0, 0.0);
        REQUIRE(one.to_internal(0, 1e-20) == u);
        REQUIRE(one.to_internal(0, 1.2345678901234568e-14) == u);
        const auto zero_first = one.with_seed_anchors({{0.0}, {1e-20}});
        CHECK(zero_first.anchor_count() == 1u);   // the (exact) owner only
        CHECK(same_bits(zero_first.to_real(0, u), 0.0));
        const auto tiny_first = one.with_seed_anchors(
            {{1e-20}, {0.0}, {1.2345678901234568e-14}});
        CHECK(tiny_first.anchor_count() == 1u);
        CHECK(same_bits(tiny_first.to_real(0, u), 1e-20));
        /* seeds with distinct u each keep their own value */
        const auto both = one.with_seed_anchors({{-0.000287486722631648},
                                                 {108.326725555511771}});
        CHECK(both.anchor_count() == 2u);
        for (double x : {-0.000287486722631648, 108.326725555511771})
            CHECK(same_bits(both.to_real(0, both.to_internal(0, x)), x));
    }
    SECTION("unusable seeds are ignored, never thrown on") {
        std::vector<double> bad = seed;
        bad[0] = std::numeric_limits<double>::quiet_NaN();
        bad[1] = 301.0;                                   // outside its box
        bad[7] = -1.0;                                    // LOG: outside, ln undefined
        const auto anch = plain.with_seed_anchors(
            {bad, std::vector<double>(n - 1, 0.5), {}});
        CHECK(anch.anchor_count() == n - 3);
        CHECK(same_bits(anch.to_real(0, 0.5), plain.to_real(0, 0.5)));
        CHECK(same_bits(anch.to_real(1, anch.to_internal(1, seed[1])),
                        plain.to_real(1, plain.to_internal(1, seed[1]))));
        CHECK(plain.with_seed_anchors({}).anchor_count() == 0u);
    }
    SECTION("anchors are replaced, not accumulated; with_boxes drops them") {
        const auto a = plain.with_seed_anchors({seed});
        std::vector<double> other = seed;
        other[0] = -0.000123456789012345678;
        const auto b = a.with_seed_anchors({other});
        CHECK(same_bits(b.to_real(0, b.to_internal(0, other[0])), other[0]));
        const double u_old = plain.to_internal(0, seed[0]);
        CHECK(same_bits(b.to_real(0, u_old), plain.to_real(0, u_old)));
        std::vector<std::pair<double, double>> boxes;
        for (const auto& d : dims) boxes.emplace_back(d.lo, d.hi);
        CHECK(a.with_boxes(boxes).anchor_count() == 0u);
    }
    SECTION("-0.0 keeps its sign") {
        const DimSpec d = dim("W", -300.0, 300.0);
        const auto one = SearchSpace::build({d}, pol);
        const auto anch = one.with_seed_anchors({{-0.0}});
        const double back = anch.to_real(0, anch.to_internal(0, -0.0));
        CHECK(back == 0.0);
        CHECK(std::signbit(back));
    }
}

TEST_CASE("SearchSpace: the ends of the internal box map to exactly the ends "
          "of the real one, and a u inside it into [lo, hi]",
          "[HPO][search_space][fd_bounds]")
{
    /* T^-1 rounds (exp, sinh, the affine map of the unit box): LOG and
     * ASINH ends came back one ulp outside the box -- LOG [0.01, 10] at
     * u = 1 gave 10.000000000000002, an evaluation outside the box that was
     * recorded as the best when an optimiser pressed against the bound. A
     * u outside the internal box still maps through T^-1, unclamped. */
    for (const bool normalise : {true, false}) {
        CAPTURE(normalise);
        ScalePolicy pol;
        pol.normalise = normalise;
        const auto ss = SearchSpace::build(
            {dim("A", 0.01, 10.0, ScaleKind::Log), dim("B", 0.3, 3.0, ScaleKind::Log)},
            pol);
        CHECK(ss.to_real(0, ss.internal_hi(0)) == 10.0);
        CHECK(ss.to_real(0, ss.internal_lo(0)) == 0.01);
        CHECK(ss.to_real(1, ss.internal_hi(1)) == 3.0);
        CHECK(ss.to_real(1, ss.internal_lo(1)) == 0.3);
        /* just outside: unclamped (above the box, not snapped onto it) */
        const double beyond = std::nextafter(
            ss.internal_hi(0), std::numeric_limits<double>::infinity());
        CHECK(ss.to_real(0, beyond) > 10.0);
    }
    /* random boxes of every shape: the ends exactly, the inside inside */
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    const auto next = [&state]() {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return static_cast<double>(state >> 11) * 0x1p-53;
    };
    for (int k = 0; k < 3000; ++k) {
        const ScaleKind kind = k % 3 == 0 ? ScaleKind::Linear
                             : k % 3 == 1 ? ScaleKind::Log : ScaleKind::Asinh;
        double lo, hi;
        if (kind == ScaleKind::Log) {
            lo = std::pow(10.0, -6.0 + 8.0 * next());
            hi = lo * std::pow(10.0, 0.01 + 5.0 * next());
        } else {
            lo = -50.0 + 100.0 * next();
            hi = lo + 1e-3 + 50.0 * next();
        }
        ScalePolicy pol;
        pol.normalise = (k / 3) % 2 == 0;
        const auto ss = SearchSpace::build(
            {dim("X", lo, hi, kind, 0.01 + next())}, pol);
        const double ulo = ss.internal_lo(0), uhi = ss.internal_hi(0);
        CAPTURE(k, lo, hi, pol.normalise);
        CHECK(ss.to_real(0, ulo) == lo);
        CHECK(ss.to_real(0, uhi) == hi);
        for (const double w : {1e-12, 0.5, 1.0 - 1e-12}) {
            const double u = ulo + w * (uhi - ulo);
            if (!(u > ulo && u < uhi)) continue;
            const double x = ss.to_real(0, u);
            CHECK(x >= lo);
            CHECK(x <= hi);
        }
    }
}

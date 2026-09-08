/* SearchSpace -- the owner of the optimiser coordinate system. */
#include "catch2/catch.hpp"
#include <tadah/hpo/search_space.h>

#include <dlib/matrix.h>

#include <cmath>
#include <string>
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

TEST_CASE("SearchSpace: require_inside_real, fingerprint, with_boxes",
          "[HPO][search_space]")
{
    ScalePolicy pol; pol.normalise = true;
    const auto ss = SearchSpace::build(three(), pol);
    REQUIRE_NOTHROW(ss.require_inside_real({6.0, 0.1, 0.0}, "seed"));
    REQUIRE_THROWS_WITH(ss.require_inside_real({6.1, 0.1, 0.0}, "seed"),
                        Catch::Contains("LIN") && Catch::Contains("outside its OPTIM bounds"));
    REQUIRE_THROWS_WITH(ss.require_inside_real({6.0, 0.1}, "seed"),
                        Catch::Contains("expected 3"));

    const std::string f1 = ss.fingerprint();
    CHECK(f1.size() == 16);
    /* labels do not affect it; bounds/kinds/normalise do */
    auto d2 = three(); d2[0].label = "renamed";
    CHECK(SearchSpace::build(d2, pol).fingerprint() == f1);
    ScalePolicy none; none.normalise = false;
    CHECK(SearchSpace::build(three(), none).fingerprint() != f1);
    auto d3 = three(); d3[0].hi = 7.0;
    CHECK(SearchSpace::build(d3, pol).fingerprint() != f1);

    const auto grown = ss.with_boxes({{-2.0, 8.0}, {0.1, 10.0}, {-100.0, 100.0}});
    CHECK(grown.version() == 2);
    CHECK(grown.at(0).hi == 8.0);
    CHECK(grown.at(0).origin == BoxOrigin::Adapted);
    CHECK(grown.at(1).origin == BoxOrigin::Explicit);
    CHECK(grown.at(1).tf.kind == ScaleKind::Log);
    CHECK(grown.fingerprint() != f1);
}

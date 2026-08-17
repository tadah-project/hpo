/*
 * Unit tests for anneal_core.h -- the host-independent building blocks
 * shared by the TADAH annealing family (ANNEAL / ANNEAL_PT / ANNEAL_PA):
 * robust T0 estimation, the Corana sec.-4 plateau rule, mirror
 * reflection, RNG stream discipline, and Knobs parsing/validation.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

using namespace tadah::hpo;

/* ---- robust_t0 ---------------------------------------------------- */

TEST_CASE("anneal::robust_t0 solves the median uphill move for the "
          "target acceptance", "[HPO][anneal_core][t0]")
{
    /* f0 = 1, uphill deltas {1, 2, 3} -> median 2. */
    const std::vector<double> probes{2.0, 3.0, 4.0};
    const auto t0 = anneal::robust_t0(1.0, probes, std::nullopt, 0.8);
    REQUIRE(t0.has_value());
    CHECK(*t0 == Approx(2.0 / std::log(1.0 / 0.8)));
}

TEST_CASE("anneal::robust_t0 excludes FAILSCORE probes from the statistics",
          "[HPO][anneal_core][t0]")
{
    constexpr double FS = 1e16;
    /* Two finite uphill probes and two cliff probes; the cliffs would
     * have dominated the old pairwise-mean estimator. */
    const std::vector<double> probes{2.0, FS, 4.0, FS};
    const auto t0 = anneal::robust_t0(1.0, probes, FS, 0.8);
    REQUIRE(t0.has_value());
    /* median of {1, 3} = 2 */
    CHECK(*t0 == Approx(2.0 / std::log(1.0 / 0.8)));
    CHECK(*t0 < 100.0);
}

TEST_CASE("anneal::robust_t0 falls back to |dE| on pure-descent "
          "neighbourhoods", "[HPO][anneal_core][t0]")
{
    /* Every probe is downhill from f0=10: U empty, |dE| = {1,2,3}. */
    const std::vector<double> probes{9.0, 8.0, 7.0};
    const auto t0 = anneal::robust_t0(10.0, probes, std::nullopt, 0.8);
    REQUIRE(t0.has_value());
    CHECK(*t0 == Approx(2.0 / std::log(1.0 / 0.8)));
}

TEST_CASE("anneal::robust_t0 degenerate cases", "[HPO][anneal_core][t0]")
{
    constexpr double FS = 1e16;
    SECTION("no finite probe -> nullopt") {
        const std::vector<double> probes{FS, FS, FS};
        CHECK_FALSE(anneal::robust_t0(1.0, probes, FS, 0.8).has_value());
    }
    SECTION("empty probe set -> nullopt") {
        CHECK_FALSE(anneal::robust_t0(1.0, {}, FS, 0.8).has_value());
    }
    SECTION("constant landscape -> 0.0 (caller picks fallback)") {
        const std::vector<double> probes{1.0, 1.0, 1.0};
        const auto t0 = anneal::robust_t0(1.0, probes, FS, 0.8);
        REQUIRE(t0.has_value());
        CHECK(*t0 == 0.0);
    }
}

TEST_CASE("anneal::robust_t0 remains finite at floating-point extremes",
          "[HPO][anneal_core][t0][numeric]")
{
    const double tiny_p = std::numeric_limits<double>::denorm_min();
    const auto tiny = anneal::robust_t0(0.0, {1.0}, std::nullopt, tiny_p);
    REQUIRE(tiny.has_value());
    CHECK(std::isfinite(*tiny));
    CHECK(*tiny > 0.0);

    const double m = std::numeric_limits<double>::max();
    const auto wide = anneal::robust_t0(-m, {m, m}, std::nullopt, 0.5);
    REQUIRE(wide.has_value());
    CHECK(std::isfinite(*wide));
    CHECK(*wide > 0.0);
}

/* ---- loss_is_finite ------------------------------------------------ */

TEST_CASE("anneal::loss_is_finite", "[HPO][anneal_core]")
{
    constexpr double FS = 1e16;
    CHECK(anneal::loss_is_finite(1.0, FS));
    CHECK(anneal::loss_is_finite(1e15, FS));
    CHECK_FALSE(anneal::loss_is_finite(FS, FS));
    CHECK_FALSE(anneal::loss_is_finite(0.9995 * FS, FS));   // margin
    CHECK_FALSE(anneal::loss_is_finite(
        std::numeric_limits<double>::infinity(), std::nullopt));
    CHECK_FALSE(anneal::loss_is_finite(
        std::nan(""), std::nullopt));
    CHECK(anneal::loss_is_finite(1e300, std::nullopt));      // no failscore
}

TEST_CASE("anneal::loss_is_rankable preserves graded FAILSCORE ordering",
          "[HPO][anneal_core][failscore][fail_grading]")
{
    constexpr double FS = 100.0;
    CHECK(anneal::loss_is_rankable(0.9990 * FS, FS));
    CHECK(anneal::loss_is_rankable(0.9995 * FS, FS));
    CHECK(anneal::loss_is_rankable(std::nextafter(FS, 0.0), FS));
    CHECK_FALSE(anneal::loss_is_rankable(FS, FS));
    CHECK_FALSE(anneal::loss_is_rankable(
        std::numeric_limits<double>::infinity(), FS));
    CHECK_FALSE(anneal::loss_is_rankable(
        std::numeric_limits<double>::quiet_NaN(), FS));
}

TEST_CASE("anneal::stopval_reached rejects failure and non-finite losses",
          "[HPO][anneal_core][stopval][failscore][fail_grading]")
{
    const std::optional<double> stopval{99.99};
    const std::optional<double> failscore{100.0};
    CHECK(anneal::stopval_reached(1.0, stopval, failscore));
    CHECK_FALSE(anneal::stopval_reached(99.95, stopval, failscore));
    CHECK_FALSE(anneal::stopval_reached(100.0, stopval, failscore));
    CHECK_FALSE(anneal::stopval_reached(
        -std::numeric_limits<double>::infinity(), stopval, failscore));
    CHECK_FALSE(anneal::stopval_reached(1.0, std::nullopt, failscore));
}
/* ---- CoranaPlateau -------------------------------------------------- */

namespace {
StopCriteria ftol_abs_stop(double eps)
{
    StopCriteria s;
    s.ftol_abs = eps;
    return s;
}
}  // namespace

TEST_CASE("CoranaPlateau: constant end-of-rung values stop after the ring "
          "fills", "[HPO][anneal_core][plateau]")
{
    anneal::CoranaPlateau p(ftol_abs_stop(1e-6), 3);
    CHECK_FALSE(p.update(5.0, 5.0));   // ring {5}
    CHECK_FALSE(p.update(5.0, 5.0));   // ring {5,5}
    CHECK_FALSE(p.update(5.0, 5.0));   // ring {5,5,5}
    CHECK(p.update(5.0, 5.0));         // flat vs all three + best
}

TEST_CASE("CoranaPlateau: does NOT stop while the chain still explores, "
          "even when the best has stalled", "[HPO][anneal_core][plateau]")
{
    /* This is the fixed bug: the old rule watched the monotone
     * best-so-far and force-stopped after NEPS rungs once it stalled,
     * even though the end-of-rung value was still swinging. */
    anneal::CoranaPlateau p(ftol_abs_stop(0.1), 2);
    const double best = 5.0;   // stalled forever
    CHECK_FALSE(p.update(5.0, best));
    CHECK_FALSE(p.update(6.0, best));
    CHECK_FALSE(p.update(5.0, best));   // differs from previous 6.0 by 1 > eps
    CHECK_FALSE(p.update(6.0, best));
    CHECK_FALSE(p.update(5.0, best));
    CHECK_FALSE(p.update(6.0, best));
}

TEST_CASE("CoranaPlateau: flat-but-far-from-best does not stop "
          "(optimum-gap clause)", "[HPO][anneal_core][plateau]")
{
    /* End-of-rung value settled at 6.0 while the best is 5.0: the
     * chain is stuck on a shelf above the optimum -- Corana sec. 4
     * requires closeness to the optimum too. */
    anneal::CoranaPlateau p(ftol_abs_stop(0.1), 2);
    CHECK_FALSE(p.update(6.0, 5.0));
    CHECK_FALSE(p.update(6.0, 5.0));
    CHECK_FALSE(p.update(6.0, 5.0));
    CHECK_FALSE(p.update(6.0, 5.0));
}

TEST_CASE("CoranaPlateau: inactive without an FTOL tolerance",
          "[HPO][anneal_core][plateau]")
{
    anneal::CoranaPlateau p(StopCriteria{}, 2);
    CHECK_FALSE(p.active());
    for (int i = 0; i < 10; ++i) CHECK_FALSE(p.update(5.0, 5.0));
}

TEST_CASE("CoranaPlateau: FTOL_REL scales eps with |f|",
          "[HPO][anneal_core][plateau]")
{
    StopCriteria s;
    s.ftol_rel = 1e-3;
    anneal::CoranaPlateau p(s, 2);
    /* |f| = 1000 -> eps = 1; swings of 0.5 are flat. */
    CHECK_FALSE(p.update(1000.0, 1000.0));
    CHECK_FALSE(p.update(1000.5, 1000.0));
    CHECK(p.update(1000.2, 1000.0));
}

TEST_CASE("CoranaPlateau: non-finite values reset rather than satisfy plateau",
          "[HPO][anneal_core][plateau][numeric]")
{
    anneal::CoranaPlateau p(ftol_abs_stop(1e-6), 2);
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK_FALSE(p.update(std::numeric_limits<double>::infinity(), 1.0));
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK(p.update(1.0, 1.0));

    CHECK_THROWS_AS(anneal::CoranaPlateau(ftol_abs_stop(1e-6), 0),
                    std::invalid_argument);
}

TEST_CASE("CoranaPlateau excludes graded and legacy failure plateaus",
          "[HPO][anneal_core][plateau][failscore][fail_grading]")
{
    anneal::CoranaPlateau p(ftol_abs_stop(0.0), 2, 100.0);
    CHECK_FALSE(p.update(100.0, 100.0));
    CHECK_FALSE(p.update(100.0, 100.0));
    CHECK_FALSE(p.update(99.95, 99.95));
    CHECK_FALSE(p.update(99.95, 99.95));
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK_FALSE(p.update(1.0, 1.0));
    CHECK(p.update(1.0, 1.0));
}

/* ---- reflect_into ---------------------------------------------------- */

TEST_CASE("anneal::reflect_into mirrors into the box", "[HPO][anneal_core]")
{
    double v = 1.5;
    CHECK(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v == Approx(0.5));

    v = -0.25;
    CHECK(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v == Approx(0.25));

    v = 0.7;   // already inside
    CHECK_FALSE(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v == Approx(0.7));

    v = 1e9 + 0.25;   // hundreds of millions of exact folds
    CHECK(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v == Approx(0.25));

    v = -1e9 - 0.75;
    CHECK(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v == Approx(0.75));

    v = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(anneal::reflect_into(v, 0.0, 1.0),
                    std::invalid_argument);

    long double wide = static_cast<long double>(
        std::numeric_limits<double>::max()) * 2.0L + 0.25L;
    if (std::isfinite(wide)) {
        CHECK(anneal::reflect_into(wide, 0.0L, 1.0L));
        CHECK(wide >= 0.0L);
        CHECK(wide <= 1.0L);
    }
}

TEST_CASE("anneal boundary kernels redraw only after an OOB first draw",
          "[HPO][anneal_core][bounds]")
{
    double v = 1.5;
    std::vector<double> draws{-0.5, 0.375};
    std::size_t next = 0;
    CHECK(anneal::constrain_proposal(
        v, 0.0, 1.0, anneal::BoundaryMode::Redraw,
        [&] { return draws.at(next++); }));
    CHECK(next == 2);
    CHECK(v == Approx(0.375));

    v = 0.25;
    CHECK_FALSE(anneal::constrain_proposal(
        v, 0.0, 1.0, anneal::BoundaryMode::Redraw,
        [] { return 0.75; }));
    CHECK(v == Approx(0.25));

    v = 2.0;
    CHECK_THROWS_AS(anneal::constrain_proposal(
        v, 0.0, 1.0, static_cast<anneal::BoundaryMode>(99),
        [] { return 0.5; }), std::invalid_argument);

    CHECK_THROWS_WITH(anneal::constrain_proposal(
        v, 0.0, 1.0, anneal::BoundaryMode::Redraw,
        [] { return std::numeric_limits<double>::infinity(); }),
        Catch::Contains("failed to produce"));
}

TEST_CASE("anneal::adjust_corana_step implements the 0.4/0.6 table and clamps",
          "[HPO][anneal_core][corana]")
{
    CHECK(anneal::adjust_corana_step(0.3, 1.0, 0, 10, 2.0, 0.01)
          == Approx(0.1));
    CHECK(anneal::adjust_corana_step(0.3, 1.0, 4, 10, 2.0, 0.01)
          == Approx(0.3));
    CHECK(anneal::adjust_corana_step(0.3, 1.0, 5, 10, 2.0, 0.01)
          == Approx(0.3));
    CHECK(anneal::adjust_corana_step(0.3, 1.0, 6, 10, 2.0, 0.01)
          == Approx(0.3));
    CHECK(anneal::adjust_corana_step(0.3, 1.0, 10, 10, 2.0, 0.01)
          == Approx(0.9));
    CHECK(anneal::adjust_corana_step(0.9, 1.0, 10, 10, 2.0, 0.01)
          == Approx(1.0));
    CHECK(anneal::adjust_corana_step(1e-12, 1.0, 0, 10, 2.0, 1e-3)
          == Approx(1e-3));
    CHECK(anneal::adjust_corana_step(1e-12, 1.0, 0, 10, 2.0, 0.0) > 0.0);
}

/* ---- RNG streams ------------------------------------------------------ */

TEST_CASE("anneal::make_stream: streams are reproducible and decorrelated",
          "[HPO][anneal_core][rng]")
{
    auto a1 = anneal::make_stream(42, anneal::SALT_CHAIN, 0);
    auto a2 = anneal::make_stream(42, anneal::SALT_CHAIN, 0);
    auto b  = anneal::make_stream(42, anneal::SALT_CHAIN, 1);
    auto c  = anneal::make_stream(42, anneal::SALT_PROBES);

    CHECK(a1() == a2());                 // same (seed, salt, idx) -> same stream
    auto a1v = a1(); (void)a1v;
    CHECK(anneal::make_stream(42, anneal::SALT_CHAIN, 0)() != b());
    CHECK(anneal::make_stream(42, anneal::SALT_CHAIN, 0)() != c());
}

/* ---- Knobs::parse ------------------------------------------------------ */

namespace {
OptimizerSpec bare_spec()
{
    OptimizerSpec sp;
    sp.lib  = "TADAH";
    sp.algo = "ANNEAL";
    sp.low  = {-1.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    return sp;
}

using CoreVec = dlib::matrix<double, 0, 1>;

CoreVec scalar_vec(double x)
{
    CoreVec v(1);
    v(0) = x;
    return v;
}

anneal::ChainState<CoreVec> scalar_chain(double x = 0.0, double f = 0.0)
{
    anneal::ChainState<CoreVec> c;
    c.x = scalar_vec(x);
    c.f = f;
    c.V = {0.1};
    c.rng = anneal::make_stream(123, anneal::SALT_CHAIN, 0);
    c.best_x = c.x;
    c.best_f = c.f;
    return c;
}
}  // namespace

TEST_CASE("anneal::Knobs::parse defaults", "[HPO][anneal_core][knobs]")
{
    auto sp = bare_spec();
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    CHECK_FALSE(kb.corana);
    CHECK(kb.L == 50);
    CHECK(kb.NS == 5);
    CHECK(kb.NT == 5);                 // bench-derived (Kr 42-D, 2026-07-13)
    CHECK(kb.rt == Approx(0.90));
    CHECK(kb.t0_paccept == Approx(0.05));
    CHECK_FALSE(kb.t_hot.has_value());
    CHECK_FALSE(kb.t_min.has_value());
    CHECK(kb.reanneal);
    CHECK(kb.reanneal_scale == Approx(0.25));
    CHECK_FALSE(kb.reset_to_best);   // classic default
    CHECK(kb.boundary_mode == anneal::BoundaryMode::Reflect);

    sp.param["CORANA"] = 1.0;
    const auto kc = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    CHECK(kc.corana);
    CHECK(kc.reset_to_best);          // flips ON under CORANA
    sp.param["RESET_TO_BEST"] = 0.0;  // explicit override wins
    CHECK_FALSE(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial).reset_to_best);
}

TEST_CASE("anneal::Knobs::parse validation errors", "[HPO][anneal_core][knobs]")
{
    SECTION("renamed T_INIT") {
        auto sp = bare_spec();
        sp.param["T_INIT"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to T_HOT"));
    }
    SECTION("unknown key") {
        auto sp = bare_spec();
        sp.param["RTT"] = 0.9;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("unknown PARAM 'RTT'"));
    }
    SECTION("RT out of range") {
        auto sp = bare_spec();
        sp.param["RT"] = 1.5;
        REQUIRE_THROWS_AS(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                          std::runtime_error);
    }
    SECTION("T0_PROBES too small without T_HOT") {
        auto sp = bare_spec();
        sp.param["T0_PROBES"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("T0_PROBES"));
        sp.param["T_HOT"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("unused when T_HOT is set"));
        sp.param.erase("T0_PROBES");
        REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
    }
    SECTION("REANNEAL_SCALE out of range") {
        auto sp = bare_spec();
        sp.param["REANNEAL_SCALE"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("REANNEAL_SCALE"));
    }
    SECTION("NEPS validated only when plateau armed") {
        auto sp = bare_spec();
        sp.param["NEPS"] = 0.0;
        REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
        sp.stop.ftol_abs = 1e-6;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("NEPS"));
    }
    SECTION("integer knobs reject fractions and negatives before casting") {
        auto sp = bare_spec();
        sp.param["L"] = 1.5;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("non-negative integer"));

        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["INIT_MODE"] = std::ldexp(1.0, 32);
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("INIT_MODE"));
        sp.param.erase("L");
        sp.param["CORANA"] = 1.0;
        sp.param["NS"] = -1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("non-negative integer"));
    }
    SECTION("non-finite values are rejected") {
        auto sp = bare_spec();
        sp.param["RT"] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("must be finite"));
        sp.param.erase("RT");
        sp.stop.ftol_abs = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("FTOL_ABS"));
        sp.stop.ftol_abs.reset();
        sp.stop.maxtime = 0.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("MAXTIME"));
    }
    SECTION("step fractions and Corana floor have explicit ranges") {
        auto sp = bare_spec();
        sp.param["STEP_FRAC"] = 1.01;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("STEP_FRAC"));
        sp.param["STEP_FRAC"] = 0.1;
        sp.param["CORANA"] = 1.0;
        sp.param["V_FLOOR_FRAC"] = -0.1;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("V_FLOOR_FRAC"));
        sp.param["V_FLOOR_FRAC"] = 0.0;
        REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
    }
    SECTION("temperature floor cannot exceed an explicit hot temperature") {
        auto sp = bare_spec();
        sp.param["T_HOT"] = 1.0;
        sp.param["T_MIN"] = 2.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("T_MIN must be <= T_HOT"));
        sp.param["T_MIN"] = 1.0;
        REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
    }
    SECTION("opposite finite bounds must still have a finite width") {
        auto sp = bare_spec();
        sp.low[0] = -std::numeric_limits<double>::max();
        sp.high[0] = std::numeric_limits<double>::max();
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("finite bound width"));
    }
    SECTION("BOUNDARY_MODE is exactly 0 or 1") {
        auto sp = bare_spec();
        sp.param["BOUNDARY_MODE"] = 2.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("BOUNDARY_MODE"));
        sp.param["BOUNDARY_MODE"] = 1.0;
        CHECK(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial).boundary_mode
              == anneal::BoundaryMode::Redraw);
    }
    SECTION("variant-level no-op knobs are rejected") {
        auto classic = bare_spec();
        classic.param["NS"] = 2.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(classic, anneal::AlgoKind::Serial),
            Catch::Contains("requires PARAM CORANA 1"));

        auto corana = bare_spec();
        corana.param["CORANA"] = 1.0;
        corana.param["L"] = 5.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(corana, anneal::AlgoKind::Serial),
            Catch::Contains("CLASSIC variant"));

        auto pa = bare_spec();
        pa.algo = "ANNEAL_PA";
        pa.pop = 8;
        pa.param["RESET_TO_BEST"] = 1.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(pa, anneal::AlgoKind::PA),
            Catch::Contains("not defined for population annealing"));
    }
    SECTION("overridden controls are rejected") {
        auto step = bare_spec();
        step.step = 0.1;
        step.param["STEP_FRAC"] = 0.2;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(step, anneal::AlgoKind::Serial),
            Catch::Contains("mutually exclusive"));

        auto hot = bare_spec();
        hot.param["T_HOT"] = 1.0;
        hot.param["T0_PACCEPT"] = 0.5;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(hot, anneal::AlgoKind::Serial),
            Catch::Contains("unused when T_HOT"));

        auto re = bare_spec();
        re.param["REANNEAL"] = 0.0;
        re.param["REANNEAL_SCALE"] = 0.5;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(re, anneal::AlgoKind::Serial),
            Catch::Contains("unused when REANNEAL is 0"));

        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["T_COLD"] = 0.1;
        pt.param["LADDER_SPAN"] = 10.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("unused when T_COLD"));
        pt.param.erase("T_COLD");
        pt.param.erase("LADDER_SPAN");
        pt.param["T_MIN"] = 1e-3;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("static PT ladder"));
    }
    SECTION("integer parsing rejects the first unrepresentable size") {
        auto sp = bare_spec();
        sp.param["L"] = std::ldexp(
            1.0, std::numeric_limits<std::size_t>::digits);
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("non-negative integer"));
    }
}

TEST_CASE("anneal::make_initial_point validates and safely transforms seeds",
          "[HPO][anneal_core][bounds][knobs]")
{
    auto sp = bare_spec();
    sp.low = {1e308};
    sp.high = {1.1e308};
    auto mid = anneal::make_initial_point(sp, anneal::AlgoKind::Serial);
    REQUIRE(mid.size() == 1);
    CHECK(std::isfinite(mid[0]));
    CHECK(mid[0] >= sp.low[0]);
    CHECK(mid[0] <= sp.high[0]);

    sp.low = {0.0};
    sp.high = {1.0};
    sp.optim_init = {{2.0}};
    REQUIRE_THROWS_WITH(
        anneal::make_initial_point(sp, anneal::AlgoKind::Serial),
        Catch::Contains("outside its OPTIM bounds"));
    sp.optim_init = {{0.5, 0.5}};
    REQUIRE_THROWS_WITH(
        anneal::make_initial_point(sp, anneal::AlgoKind::Serial),
        Catch::Contains("expected 1"));

    sp.log_scaled = {true};
    sp.low = {std::log(1.0)};
    sp.high = {std::log(100.0)};
    sp.optim_init = {{10.0}};
    const auto logged = anneal::make_initial_point(sp, anneal::AlgoKind::Serial);
    CHECK(logged[0] == Approx(std::log(10.0)));
}

TEST_CASE("anneal schedule helpers saturate and always cool",
          "[HPO][anneal_core][numeric]")
{
    const auto many = anneal::planned_rungs(
        1.0, std::numeric_limits<double>::denorm_min(), 0.9,
        std::numeric_limits<std::uint64_t>::max() - 1, 3);
    CHECK(many > 1);
    CHECK(anneal::saturating_mul(
              std::numeric_limits<std::uint64_t>::max(), 2) ==
          std::numeric_limits<std::uint64_t>::max());
    CHECK(anneal::saturating_add(
              std::numeric_limits<std::size_t>::max() - 1, 2) ==
          std::numeric_limits<std::size_t>::max());
    const double tiny = std::numeric_limits<double>::denorm_min();
    CHECK(anneal::automatic_t_min(tiny) > 0.0);
    CHECK(anneal::cool_temperature(tiny, 0.9) == 0.0);
}

TEST_CASE("anneal::planned_rungs avoids near-UINT64 and schedule overflow",
          "[HPO][anneal_core][numeric][budget]")
{
    const auto u64max = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t left = u64max - 1;
    const std::uint64_t cost = 3;
    const std::uint64_t ceil64 = left / cost + (left % cost != 0 ? 1 : 0);
    const std::size_t expected = ceil64 > std::numeric_limits<std::size_t>::max()
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(ceil64);

    CHECK(anneal::planned_rungs(
              std::numeric_limits<double>::max(),
              std::numeric_limits<double>::denorm_min(), 1.0,
              left, cost) == expected);
    CHECK(anneal::planned_rungs(1.0, 0.5, 1.0, u64max, 1) ==
          std::numeric_limits<std::size_t>::max());

    const std::size_t extreme = anneal::planned_rungs(
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min(),
        std::nextafter(1.0, 0.0), u64max, 1);
    CHECK(extreme > std::numeric_limits<std::size_t>::max() / 2);
}

TEST_CASE("anneal::SharedBudget enforces its cap and releases reservations",
          "[HPO][anneal_core][budget]")
{
    anneal::SharedBudget budget(2);
    CHECK(budget.spent() == 0);
    CHECK(budget.left() == 2);
    CHECK(budget.try_reserve());
    CHECK(budget.try_reserve());
    CHECK_FALSE(budget.try_reserve());
    CHECK(budget.spent() == 2);
    CHECK(budget.left() == 0);

    budget.release();
    CHECK(budget.spent() == 1);
    CHECK(budget.left() == 1);
    CHECK(budget.try_reserve());
    CHECK_FALSE(budget.try_reserve());
    CHECK(budget.spent() == 2);
    budget.release();
    budget.release();
    CHECK(budget.spent() == 0);
    CHECK(budget.left() == 2);
}

TEST_CASE("anneal::metropolis_accept repairs invalid states without poisoning",
          "[HPO][anneal_core][numeric][failscore]")
{
    std::size_t coins = 0;
    auto coin = [&] { ++coins; return 0.5; };
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    CHECK(anneal::metropolis_accept(inf, 1.0, 1.0, std::nullopt, coin));
    CHECK(coins == 0);
    CHECK_FALSE(anneal::metropolis_accept(1.0, nan, 1.0,
                                          std::nullopt, coin));
    CHECK(coins == 0);
    CHECK(anneal::metropolis_accept(1.0, 0.0, 1.0, std::nullopt, coin));
    CHECK(coins == 0);

    const std::optional<double> failscore{100.0};
    CHECK(anneal::metropolis_accept(100.0, 1.0, 1.0, failscore, coin));
    CHECK_FALSE(anneal::metropolis_accept(1.0, 100.0, 1.0,
                                          failscore, coin));
    CHECK(coins == 0);

    const double m = std::numeric_limits<double>::max();
    CHECK_FALSE(anneal::metropolis_accept(-m, m, 1.0,
                                          std::nullopt, coin));
    CHECK(coins == 1);
}

TEST_CASE("anneal::metropolis_accept follows graded FAILSCORE repair signal",
          "[HPO][anneal_core][numeric][failscore][fail_grading]")
{
    const std::optional<double> failscore{100.0};
    std::size_t coins = 0;
    auto accept_coin = [&] { ++coins; return 0.0; };

    CHECK(anneal::metropolis_accept(100.0, 99.95, 1.0,
                                    failscore, accept_coin));
    CHECK(coins == 0);
    CHECK(anneal::metropolis_accept(99.95, 99.91, 1.0,
                                    failscore, accept_coin));
    CHECK(coins == 0);
    CHECK(anneal::metropolis_accept(99.91, 99.95, 1.0,
                                    failscore, accept_coin));
    CHECK(coins == 1);
    CHECK_FALSE(anneal::metropolis_accept(99.91, 100.0, 1.0,
                                          failscore, accept_coin));
    CHECK(coins == 1);
}

TEST_CASE("anneal diagonal RAM cadence accumulates every proposal",
          "[HPO][anneal_core][step_adapt]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {-10.0, -10.0};
    sp.high = {10.0, 10.0};
    auto make_chain = [] {
        anneal::ChainState<Vec> c;
        c.x = Vec(2);
        c.x = 0.0;
        c.V = {0.1, 0.2};
        return c;
    };
    const std::vector<double> z{1.0, 2.0};

    auto frozen = make_chain();
    anneal::accumulate_diagonal_ram(frozen, z, 1.0L, sp, 0);
    CHECK(frozen.V[0] == 0.1);
    CHECK(frozen.diag_ram_proposals == 0);

    auto zero_displacement = make_chain();
    const std::vector<double> zero_z{0.0, 0.0};
    anneal::accumulate_diagonal_ram(
        zero_displacement, zero_z, 0.0L, sp, 2);
    anneal::accumulate_diagonal_ram(
        zero_displacement, zero_z, 0.0L, sp, 2);
    CHECK(zero_displacement.V[0] < 0.1);
    CHECK(zero_displacement.V[1] < 0.2);

    auto every_one = make_chain();
    anneal::accumulate_diagonal_ram(every_one, z, 1.0L, sp, 1);
    CHECK(every_one.V[0] > 0.1);
    CHECK(every_one.V[1] > 0.2);
    CHECK(every_one.diag_ram_pending == 0);

    auto every_l = make_chain();
    auto every_5l = make_chain();
    for (std::size_t j = 0; j < 4; ++j) {
        anneal::accumulate_diagonal_ram(every_l, z, 1.0L, sp, 5);
        anneal::accumulate_diagonal_ram(every_5l, z, 1.0L, sp, 25);
    }
    CHECK(every_l.V[0] == 0.1);
    CHECK(every_5l.V[0] == 0.1);
    CHECK(every_l.diag_ram_pending == 4);
    CHECK(every_5l.diag_ram_pending == 4);
    REQUIRE(every_l.diag_ram_delta.size() == 2);
    REQUIRE(every_5l.diag_ram_delta.size() == 2);
    CHECK(every_l.diag_ram_delta[0] == Approx(every_5l.diag_ram_delta[0]));
    CHECK(every_l.diag_ram_delta[1] == Approx(every_5l.diag_ram_delta[1]));

    anneal::accumulate_diagonal_ram(every_l, z, 1.0L, sp, 5);
    anneal::accumulate_diagonal_ram(every_5l, z, 1.0L, sp, 25);
    CHECK(every_l.V[0] > 0.1);       // cadence L applies the batch
    CHECK(every_5l.V[0] == 0.1);     // cadence 5L retains it for later
    CHECK(every_5l.diag_ram_pending == 5);
    CHECK(every_5l.diag_ram_delta[0] > 0.0);
}

TEST_CASE("anneal scalar RAM repairs a common tenfold scale error quickly",
          "[HPO][anneal_core][step_adapt][scalar]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {-10.0, -10.0, -10.0};
    sp.high = {10.0, 10.0, 10.0};
    anneal::ChainState<Vec> c;
    c.x = Vec(3);
    c.x = 0.0;
    c.V = {1.0e-3, 1.0e-3, 1.0e-3};
    const std::vector<double> z{1.0, 1.0, 1.0};

    for (std::size_t j = 0; j < 200; ++j)
        anneal::accumulate_diagonal_ram(c, z, 1.0L, sp, 20);

    CHECK(c.V[0] > 1.0e-2);
    CHECK(c.V[0] == Approx(c.V[1]));
    CHECK(c.V[1] == Approx(c.V[2]));
    CHECK(c.diag_ram_proposals == 200);
    CHECK(c.diag_ram_pending == 0);
}

TEST_CASE("anneal diagonal RAM trust region bounds adversarial drift",
          "[HPO][anneal_core][step_adapt][trust]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {-10.0, -10.0};
    sp.high = {10.0, 10.0};
    auto make_chain = [] {
        anneal::ChainState<Vec> c;
        c.x = Vec(2);
        c.x = 0.0;
        c.V = {0.1, 0.1};
        return c;
    };
    /* All displacement in dim 0, every proposal rejected: the scalar term
     * shrinks the whole vector while the centred shape term keeps pushing
     * dim 0 below dim 1 -- the unbounded-anisotropy walk the trust region
     * exists to bound (run24 sat at sigma/width ~ 2e-10 for 70k epochs). */
    const std::vector<double> z{1.0, 0.0};
    const double F = 8.0;

    auto bounded = make_chain();
    bounded.V_baseline = bounded.V;
    for (std::size_t j = 0; j < 5000; ++j) {
        anneal::accumulate_diagonal_ram(bounded, z, 0.0L, sp, 1, F);
        if (j % 100 == 0) {
            for (std::size_t i = 0; i < 2; ++i) {
                CHECK(bounded.V[i] >= 0.1 / F * (1.0 - 1e-9));
                CHECK(bounded.V[i] <= 0.1 * F * (1.0 + 1e-9));
            }
        }
    }
    CHECK(bounded.V[0] >= 0.1 / F * (1.0 - 1e-9));
    CHECK(bounded.V[1] >= 0.1 / F * (1.0 - 1e-9));

    /* Legacy behaviour is preserved both with the factor off and with no
     * recorded baseline: the walk escapes the would-be trust bound. */
    auto unbounded = make_chain();
    unbounded.V_baseline = unbounded.V;
    for (std::size_t j = 0; j < 5000; ++j)
        anneal::accumulate_diagonal_ram(unbounded, z, 0.0L, sp, 1, 0.0);
    CHECK(unbounded.V[0] < 0.1 / F);

    auto no_baseline = make_chain();
    for (std::size_t j = 0; j < 5000; ++j)
        anneal::accumulate_diagonal_ram(no_baseline, z, 0.0L, sp, 1, F);
    CHECK(no_baseline.V[0] < 0.1 / F);
}

TEST_CASE("anneal::run_rung retains the best graded FAILSCORE repair",
          "[HPO][anneal_core][rung][failscore][fail_grading]")
{
    for (bool corana : {false, true}) {
        CAPTURE(corana);
        auto sp = bare_spec();
        sp.failscore = 100.0;
        sp.param["T_HOT"] = 1.0e300;
        if (corana) sp.param["CORANA"] = 1.0;
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
        auto c = scalar_chain(0.0, 100.0);
        anneal::SharedBudget budget(2);
        std::atomic<bool> stop{false};
        const std::function<bool()> never = [] { return false; };
        anneal::StopCtx ctx{budget, stop, never, 99.99};
        std::size_t calls = 0;
        auto host = [&](const CoreVec&) {
            return calls++ == 0 ? 99.91 : 99.95;
        };

        const auto rs = anneal::run_rung(
            c, 1.0e300, sp, kb, host, ctx, 2);
        CHECK(rs.proposed == 2);
        CHECK(rs.accepted == 2);
        CHECK(rs.improved == 1);
        CHECK(c.f == Approx(99.95));
        CHECK(c.best_f == Approx(99.91));
        CHECK(calls == 2);
        CHECK(budget.spent() == 2);
    }
}

TEST_CASE("anneal::run_rung releases reservations on pre-evaluation stops",
          "[HPO][anneal_core][rung][budget]")
{
    for (bool corana : {false, true}) {
        CAPTURE(corana);
        auto sp = bare_spec();
        sp.param["T_HOT"] = 1.0;
        if (corana) {
            sp.param["CORANA"] = 1.0;
            sp.param["NS"] = 2.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.param["L"] = 2.0;
        }
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);

        {
            INFO("zero allowance does not reserve or evaluate");
            auto c = scalar_chain();
            anneal::SharedBudget budget(4);
            std::atomic<bool> stop{false};
            const std::function<bool()> never = [] { return false; };
            anneal::StopCtx ctx{budget, stop, never, std::nullopt};
            std::size_t calls = 0;
            auto host = [&](const CoreVec&) { ++calls; return 0.0; };

            const auto rs = anneal::run_rung(c, 1.0, sp, kb, host, ctx, 0);
            CHECK(rs.stop == anneal::RungStop::Budget);
            CHECK(rs.proposed == 0);
            CHECK(calls == 0);
            CHECK(budget.spent() == 0);
        }

        {
            INFO("external stop releases the speculative reservation");
            auto c = scalar_chain();
            anneal::SharedBudget budget(4);
            std::atomic<bool> stop{true};
            const std::function<bool()> never = [] { return false; };
            anneal::StopCtx ctx{budget, stop, never, std::nullopt};
            std::size_t calls = 0;
            auto host = [&](const CoreVec&) { ++calls; return 0.0; };

            const auto rs = anneal::run_rung(c, 1.0, sp, kb, host, ctx);
            CHECK(rs.stop == anneal::RungStop::External);
            CHECK(rs.proposed == 0);
            CHECK(calls == 0);
            CHECK(budget.spent() == 0);
        }

        {
            INFO("time stop releases the speculative reservation");
            auto c = scalar_chain();
            anneal::SharedBudget budget(4);
            std::atomic<bool> stop{false};
            const std::function<bool()> elapsed = [] { return true; };
            anneal::StopCtx ctx{budget, stop, elapsed, std::nullopt};
            std::size_t calls = 0;
            auto host = [&](const CoreVec&) { ++calls; return 0.0; };

            const auto rs = anneal::run_rung(c, 1.0, sp, kb, host, ctx);
            CHECK(rs.stop == anneal::RungStop::Time);
            CHECK(rs.proposed == 0);
            CHECK(calls == 0);
            CHECK(budget.spent() == 0);
        }
    }
}

TEST_CASE("anneal::calibrate_t0 reports truncation and propagates STOPVAL",
          "[HPO][anneal_core][t0][budget]")
{
    auto sp = bare_spec();
    sp.param["T0_PROBES"] = 2.0;
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    const std::vector<double> sigma{0.1};
    const std::function<bool()> never = [] { return false; };

    SECTION("budget exhaustion is explicit and never overshoots") {
        auto x = scalar_vec(0.0);
        auto best_x = x;
        double f = 1.0, best_f = f;
        auto rng = anneal::make_stream(41, anneal::SALT_PROBES);
        anneal::SharedBudget budget(1);
        std::atomic<bool> stop{false};
        anneal::StopCtx ctx{budget, stop, never, std::nullopt};
        std::size_t calls = 0;
        auto host = [&](const CoreVec&) { ++calls; return 2.0; };

        const auto out = anneal::calibrate_t0(
            sp, kb, sigma, x, f, best_x, best_f, host, rng, ctx,
            "ANNEAL_TEST");
        CHECK(out.aborted);
        CHECK(out.budget_exhausted);
        CHECK_FALSE(out.stopval_hit);
        CHECK(out.n_probes == 1);
        CHECK(calls == 1);
        CHECK(budget.spent() == 1);
        CHECK_FALSE(stop.load());
    }

    SECTION("a probe reaching STOPVAL raises the shared stop") {
        auto x = scalar_vec(0.0);
        auto best_x = x;
        double f = 1.0, best_f = f;
        auto rng = anneal::make_stream(42, anneal::SALT_PROBES);
        anneal::SharedBudget budget(8);
        std::atomic<bool> stop{false};
        anneal::StopCtx ctx{budget, stop, never, 0.5};
        std::size_t calls = 0;
        auto host = [&](const CoreVec&) { ++calls; return 0.25; };

        const auto out = anneal::calibrate_t0(
            sp, kb, sigma, x, f, best_x, best_f, host, rng, ctx,
            "ANNEAL_TEST");
        CHECK(out.aborted);
        CHECK_FALSE(out.budget_exhausted);
        CHECK(out.stopval_hit);
        CHECK(out.n_probes == 1);
        CHECK(calls == 1);
        CHECK(budget.spent() == 1);
        CHECK(stop.load());
        CHECK(best_f == Approx(0.25));
    }

    SECTION("an already-raised shared stop evaluates nothing") {
        auto x = scalar_vec(0.0);
        auto best_x = x;
        double f = 1.0, best_f = f;
        auto rng = anneal::make_stream(44, anneal::SALT_PROBES);
        anneal::SharedBudget budget(8);
        std::atomic<bool> stop{true};
        anneal::StopCtx ctx{budget, stop, never, 0.5};
        std::size_t calls = 0;
        auto host = [&](const CoreVec&) { ++calls; return 0.25; };

        const auto out = anneal::calibrate_t0(
            sp, kb, sigma, x, f, best_x, best_f, host, rng, ctx,
            "ANNEAL_TEST");
        CHECK(out.aborted);
        CHECK_FALSE(out.stopval_hit);
        CHECK(out.external_stop);
        CHECK_FALSE(out.budget_exhausted);
        CHECK(out.n_probes == 0);
        CHECK(calls == 0);
        CHECK(budget.spent() == 0);
    }

    SECTION("negative infinity and NaN probes do not imply STOPVAL") {
        auto x = scalar_vec(0.0);
        auto best_x = x;
        double f = 1.0, best_f = f;
        auto rng = anneal::make_stream(43, anneal::SALT_PROBES);
        anneal::SharedBudget budget(8);
        std::atomic<bool> stop{false};
        anneal::StopCtx ctx{budget, stop, never, 0.5};
        std::size_t calls = 0;
        auto host = [&](const CoreVec&) {
            return calls++ == 0
                ? -std::numeric_limits<double>::infinity()
                : std::numeric_limits<double>::quiet_NaN();
        };

        const auto out = anneal::calibrate_t0(
            sp, kb, sigma, x, f, best_x, best_f, host, rng, ctx,
            "ANNEAL_TEST");
        CHECK_FALSE(out.aborted);
        CHECK_FALSE(out.budget_exhausted);
        CHECK_FALSE(out.stopval_hit);
        CHECK(out.n_probes == 2);
        CHECK(out.T0 == Approx(1.0));
        CHECK(calls == 2);
        CHECK(budget.spent() == 2);
        CHECK_FALSE(stop.load());
        CHECK(best_f == Approx(1.0));
    }
}

TEST_CASE("anneal::make_initial_steps enforces Corana floor and cap",
          "[HPO][anneal_core][corana][knobs]")
{
    auto sp = bare_spec();
    sp.param["CORANA"] = 1.0;
    sp.param["STEP_FRAC"] = 0.01;
    sp.param["V_FLOOR_FRAC"] = 0.10;
    auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    auto v = anneal::make_initial_steps(sp, kb);
    REQUIRE(v.size() == 1);
    CHECK(v[0] == Approx(0.2));

    sp.step = 10.0;
    sp.param.erase("STEP_FRAC");
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    v = anneal::make_initial_steps(sp, kb);
    CHECK(v[0] == Approx(2.0));

    sp = bare_spec();
    sp.low = {0.0};
    sp.high = {std::numeric_limits<double>::denorm_min()};
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    REQUIRE_THROWS_WITH(anneal::make_initial_steps(sp, kb),
                        Catch::Contains("proposal scale underflowed"));
    sp.step = std::numeric_limits<double>::denorm_min();
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    CHECK(anneal::make_initial_steps(sp, kb)[0] > 0.0);
}

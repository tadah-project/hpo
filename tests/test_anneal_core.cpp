/*
 * Unit tests for anneal_core.h -- the host-independent building blocks
 * shared by the TADAH annealing family (ANNEAL / ANNEAL_PT / ANNEAL_PA):
 * robust T0 estimation, the Corana sec.-4 plateau rule, mirror
 * reflection, RNG stream discipline, and Knobs parsing/validation.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/optimizer_spec.h>

#include <cmath>
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

    v = 1e9;   // pathological step -> clamped after 16 folds
    CHECK(anneal::reflect_into(v, 0.0, 1.0));
    CHECK(v >= 0.0);
    CHECK(v <= 1.0);
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
    return sp;
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
        sp.param["T_HOT"] = 1.0;   // explicit T_HOT lifts the requirement
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
}

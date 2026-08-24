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

#include <algorithm>
#include <atomic>
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

TEST_CASE("CoranaPlateau: FTOL_REL/FTOL_ABS = 0 means disabled, not "
          "bitwise-equality", "[HPO][anneal_core][plateau]")
{
    StopCriteria s;
    s.ftol_abs = 0.0;
    s.ftol_rel = 0.0;
    anneal::CoranaPlateau p(s, 2);
    CHECK_FALSE(p.active());
    for (int i = 0; i < 10; ++i) CHECK_FALSE(p.update(5.0, 5.0));
    /* NEPS=0 is only rejected when a tolerance actually arms the rule. */
    CHECK_NOTHROW(anneal::CoranaPlateau(s, 0));
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
    anneal::CoranaPlateau p(ftol_abs_stop(1e-9), 2, 100.0);
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
    /* Serial CHAIN_LENGTH/NT default to AUTO, which requires a MAXEVAL
     * budget to derive from; parse hard-errors otherwise (tested
     * explicitly below). */
    sp.stop.maxeval = 1000;
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
    CHECK(kb.chain_length == 0);       // AUTO sentinel: driver derives from MAXEVAL
    CHECK(kb.NS == 20);                // Corana 1987 NS = 20
    CHECK(kb.NT == 0);                 // AUTO sentinel
    CHECK(kb.rt == Approx(0.85));      // Corana 1987 r_T = 0.85
    CHECK(kb.t0_accept == Approx(0.05));
    CHECK_FALSE(kb.t_start.has_value());
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

    /* PA keeps static defaults (values unchanged by the redesign). */
    auto pa = bare_spec();
    pa.algo = "ANNEAL_PA";
    const auto kp = anneal::Knobs::parse(pa, anneal::AlgoKind::PA);
    CHECK(kp.chain_length == 50);
    CHECK(kp.NT == 5);
    CHECK(kp.rt == Approx(0.90));
    CHECK(kp.init_mode == anneal_enums::INIT_MODE::JITTER);

    /* PT: STEP_MODE defaults to FROZEN; the RAM trust region is
     * normalized to its 0 = off sentinel outside ADAPTIVE.
     * (STEP_ADAPT_EVERY is retired -- the RAM applies per proposal.) */
    auto pt = bare_spec();
    pt.algo = "ANNEAL_PT";
    const auto kt = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
    CHECK(kt.step_mode == anneal_enums::STEP_MODE::FROZEN);
    CHECK(kt.step_warmup_sweeps == 12);
    CHECK(kt.step_trust_factor == 0.0);
    CHECK(kt.init_mode == anneal_enums::INIT_MODE::JITTER);
}

TEST_CASE("anneal::Knobs::parse serial AUTO lengths require MAXEVAL",
          "[HPO][anneal_core][knobs]")
{
    auto sp = bare_spec();
    sp.stop.maxeval.reset();
    REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                        Catch::Contains("CHAIN_LENGTH AUTO needs MAXEVAL"));
    sp.param["CHAIN_LENGTH"] = 10.0;
    REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
    sp.param.erase("CHAIN_LENGTH");
    sp.param["CORANA"] = 1.0;
    REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                        Catch::Contains("NT AUTO needs MAXEVAL"));
    sp.param["NT"] = 3.0;
    REQUIRE_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
}

TEST_CASE("anneal::Knobs::parse enum keys take tokens",
          "[HPO][anneal_core][knobs][enum]")
{
    auto pt = bare_spec();
    pt.algo = "ANNEAL_PT";

    SECTION("numeric STEP_MODE is a hard error naming the tokens") {
        pt.param["STEP_MODE"] = 1.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
            Catch::Contains("takes a token, one of: FIXED FROZEN ADAPTIVE"));
    }
    SECTION("unknown STEP_MODE token is reported verbatim") {
        pt.param_raw["STEP_MODE"] = "SOMETIMES";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("got 'SOMETIMES'"));
    }
    SECTION("tokens map onto anneal_enums and arm the ADAPTIVE defaults") {
        pt.param_raw["STEP_MODE"] = "FIXED";
        auto kf = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
        CHECK(kf.step_mode == anneal_enums::STEP_MODE::FIXED);
        CHECK(kf.step_trust_factor == 0.0);

        pt.param_raw["STEP_MODE"] = "ADAPTIVE";
        auto ka = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
        CHECK(ka.step_mode == anneal_enums::STEP_MODE::ADAPTIVE);
        CHECK(ka.step_trust_factor == 64.0);   // registry default
    }
    SECTION("INIT_MODE tokens (PT + PA)") {
        pt.param_raw["INIT_MODE"] = "UNIFORM";
        CHECK(anneal::Knobs::parse(pt, anneal::AlgoKind::PT).init_mode ==
              anneal_enums::INIT_MODE::UNIFORM);
        auto pa = bare_spec();
        pa.algo = "ANNEAL_PA";
        pa.param["INIT_MODE"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pa, anneal::AlgoKind::PA),
                            Catch::Contains("takes a token, one of: JITTER UNIFORM"));
        pa.param.erase("INIT_MODE");
        pa.param_raw["INIT_MODE"] = "JITTER";
        CHECK(anneal::Knobs::parse(pa, anneal::AlgoKind::PA).init_mode ==
              anneal_enums::INIT_MODE::JITTER);
    }
    SECTION("token values on non-enum keys are typos, not silent zeros") {
        auto sp = bare_spec();
        sp.param_raw["STEP_FRAC"] = "BIG";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("must be a number (got 'BIG')"));
    }
}

TEST_CASE("anneal::Knobs::parse validation errors", "[HPO][anneal_core][knobs]")
{
    SECTION("wrong-kind start-temperature keys steer to the right name") {
        auto sp = bare_spec();
        sp.param["T_HOT"] = 1.0;   /* serial takes T_INIT */
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("PARAM T_INIT"));
        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["T_INIT"] = 1.0;  /* PT/PA take T_HOT */
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("use T_HOT here"));
    }
    SECTION("renamed keys are hard errors with migration hints") {
        auto sp = bare_spec();
        sp.param["L"] = 5.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to CHAIN_LENGTH"));
        sp.param.erase("L");
        sp.param["C"] = 2.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to CORANA_C"));
        sp.param.erase("C");
        sp.param["T0_PACCEPT"] = 0.1;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to T0_ACCEPT"));

        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["STEP_AUTO"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("replaced by PARAM STEP_MODE"));
        pt.param.erase("STEP_AUTO");
        pt.param["STEP_AUTO_SWEEPS"] = 8.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("renamed to STEP_WARMUP_SWEEPS"));
        pt.param.erase("STEP_AUTO_SWEEPS");
        /* STEP_ADAPT_EVERY is retired outright: the per-proposal RAM has
         * no batching cadence any more. */
        pt.param["STEP_ADAPT_EVERY"] = 5.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
            Catch::Contains("PARAM STEP_ADAPT_EVERY was removed: STEP_MODE "
                            "ADAPTIVE applies the robust-adaptive-Metropolis "
                            "update after every proposal"));
        pt.param.erase("STEP_ADAPT_EVERY");
        pt.param["LADDER_ADAPT_UPDATES"] = 4.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("was removed"));
    }
    SECTION("PT rejects cooling-schedule keys with static-ladder steering") {
        for (const char* key : {"RT", "T_MIN", "NT", "CHAIN_LENGTH"}) {
            auto pt = bare_spec();
            pt.algo = "ANNEAL_PT";
            pt.param[key] = 1.0;
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                                Catch::Contains("is not used by ANNEAL_PT"));
        }
        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["RT"] = 0.9;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("static"));
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
    SECTION("T0_PROBES too small without T_INIT") {
        auto sp = bare_spec();
        sp.param["T0_PROBES"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("T0_PROBES"));
        sp.param["T_INIT"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("unused when T_INIT is set"));
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
        sp.param["CHAIN_LENGTH"] = 1.5;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("non-negative integer"));

        auto pt = bare_spec();
        pt.algo = "ANNEAL_PT";
        pt.param["REPLICAS"] = 2.5;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(pt, anneal::AlgoKind::PT),
                            Catch::Contains("non-negative integer"));
        sp.param.erase("CHAIN_LENGTH");
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
    SECTION("temperature floor cannot exceed an explicit start temperature") {
        auto sp = bare_spec();
        sp.param["T_INIT"] = 1.0;
        sp.param["T_MIN"] = 2.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("T_MIN must be <= T_INIT"));
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
        corana.param["NT"] = 1.0;             /* satisfy the AUTO gate */
        corana.param["CHAIN_LENGTH"] = 5.0;
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
            Catch::Contains("does not use STEP"));

        auto hot = bare_spec();
        hot.param["T_INIT"] = 1.0;
        hot.param["T0_ACCEPT"] = 0.5;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(hot, anneal::AlgoKind::Serial),
            Catch::Contains("unused when T_INIT"));

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
                            Catch::Contains("static"));
    }
    SECTION("top-level STEP is a hard error for the whole family") {
        for (auto kind : {anneal::AlgoKind::Serial, anneal::AlgoKind::PT,
                          anneal::AlgoKind::PA}) {
            auto sp = bare_spec();
            sp.algo = anneal::algo_name(kind);
            if (kind == anneal::AlgoKind::PA) sp.pop = 16;
            sp.step = 0.1;
            REQUIRE_THROWS_WITH(
                anneal::Knobs::parse(sp, kind),
                Catch::Contains("does not use STEP; set PARAM STEP_FRAC"));
        }
    }
    SECTION("integer parsing rejects the first unrepresentable size") {
        auto sp = bare_spec();
        sp.param["CHAIN_LENGTH"] = std::ldexp(
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

TEST_CASE("anneal diagonal RAM applies once per proposal "
          "(STEP_ADAPT_EVERY retired)", "[HPO][anneal_core][step_adapt]")
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

    /* Every observation feeds back immediately at its own eta_j: V moves
     * on the very first call and the eta counter advances per call --
     * there is no pending batch and no cadence knob any more. */
    auto c = make_chain();
    anneal::apply_diagonal_ram(c, z, 1.0L, sp);
    CHECK(c.diag_ram_proposals == 1);
    CHECK(c.V[0] > 0.1);
    CHECK(c.V[1] > 0.2);
    const double v_after_one = c.V[0];
    anneal::apply_diagonal_ram(c, z, 1.0L, sp);
    CHECK(c.diag_ram_proposals == 2);
    CHECK(c.V[0] > v_after_one);

    /* alpha = 0 shrinks: err = -kTargetAcceptVector < 0 (zero
     * displacement leaves only the scalar term). */
    auto shrink = make_chain();
    const std::vector<double> zero_z{0.0, 0.0};
    anneal::apply_diagonal_ram(shrink, zero_z, 0.0L, sp);
    CHECK(shrink.V[0] < 0.1);
    CHECK(shrink.V[1] < 0.2);

    /* Zero-dimensional input is a no-op and does not advance eta. */
    auto zero_dim = make_chain();
    anneal::apply_diagonal_ram(zero_dim, {}, 1.0L, sp);
    CHECK(zero_dim.diag_ram_proposals == 0);
    CHECK(zero_dim.V[0] == 0.1);
}

TEST_CASE("anneal::classic_rung feeds the RAM only under STEP_MODE ADAPTIVE",
          "[HPO][anneal_core][step_adapt][rung]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {-10.0, -10.0};
    sp.high = {10.0, 10.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    auto run = [&](int step_mode) {
        anneal::Knobs kb;
        kb.chain_length = 6;
        kb.step_mode = step_mode;
        anneal::ChainState<Vec> c;
        c.x = Vec(2);
        c.x = 0.0;
        c.f = 0.0;
        c.V = {0.1, 0.1};
        c.rng = anneal::make_stream(555, anneal::SALT_CHAIN, 0);
        c.best_x = c.x;
        c.best_f = c.f;
        anneal::SharedBudget budget(100);
        std::atomic<bool> stop{false};
        const std::function<bool()> never = [] { return false; };
        anneal::StopCtx ctx{budget, stop, never, std::nullopt};
        auto host = [](const Vec&) { return 0.0; };
        anneal::classic_rung(c, 1.0, sp, kb, host, ctx);
        return c;
    };

    const auto frozen = run(anneal_enums::STEP_MODE::FROZEN);
    CHECK(frozen.diag_ram_proposals == 0);
    CHECK(frozen.V[0] == 0.1);
    CHECK(frozen.V[1] == 0.1);

    /* alpha = 1 on every proposal (f_prop <= f): the scalar term grows
     * both sigmas and the eta counter tracks the proposal count. */
    const auto adaptive = run(anneal_enums::STEP_MODE::ADAPTIVE);
    CHECK(adaptive.diag_ram_proposals == 6);
    CHECK(adaptive.V[0] > 0.1);
    CHECK(adaptive.V[1] > 0.1);
}

TEST_CASE("anneal RAM ln-V trajectory equals the Robbins-Monro closed form",
          "[HPO][anneal_core][step_adapt][lock]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {-100.0, -100.0};
    sp.high = {100.0, 100.0};
    /* z all-equal: ratio_i = 1/dim for every dim, so the centred shape
     * term is EXACTLY zero and the update is the pure scalar recursion
     *     ln V_j = ln V_{j-1} + clamp(eta_j*(alpha_j - target), +/- ln 2),
     *     eta_j = min(1, j^-kRamDecay). */
    const std::vector<double> z{1.0, 1.0};
    /* The at-target step uses the constant itself: a 0.234L literal
     * differs from (long double)kTargetAcceptVector in the low bits and
     * would make the "no move at target" branch ill-posed. */
    const std::vector<long double> alphas{
        1.0L, 0.0L, 0.8L,
        static_cast<long double>(anneal::kTargetAcceptVector),
        0.5L, 0.0L, 1.0L, 0.1L};
    const double ln2 = std::log(2.0);

    anneal::ChainState<Vec> c;
    c.x = Vec(2);
    c.x = 0.0;
    c.V = {0.1, 0.1};

    double expected = 0.1;
    for (std::size_t j = 1; j <= alphas.size(); ++j) {
        const long double alpha = alphas[j - 1];
        const double before = c.V[0];
        anneal::apply_diagonal_ram(c, z, alpha, sp);

        const long double eta = std::min<long double>(
            1.0L, std::pow(static_cast<long double>(j),
                           -static_cast<long double>(anneal::kRamDecay)));
        const long double err =
            alpha - static_cast<long double>(anneal::kTargetAcceptVector);
        const double scalar = static_cast<double>(eta * err);
        const double factor = std::clamp(scalar, -ln2, ln2);
        expected = static_cast<double>(
            static_cast<long double>(expected) *
            std::exp(static_cast<long double>(factor)));

        CAPTURE(j);
        CHECK(c.V[0] == Approx(expected).epsilon(1e-12));
        CHECK(c.V[1] == Approx(expected).epsilon(1e-12));
        /* Direction lock: alpha above target grows, below shrinks. */
        if (alpha > static_cast<long double>(anneal::kTargetAcceptVector))
            CHECK(c.V[0] > before);
        else if (alpha < static_cast<long double>(anneal::kTargetAcceptVector))
            CHECK(c.V[0] < before);
        else
            CHECK(c.V[0] == Approx(before));
    }
    CHECK(c.diag_ram_proposals == alphas.size());

    /* Trust-region clamp: the same recursion saturates at baseline*F
     * upward (alpha = 1) and baseline/F downward (alpha = 0). */
    const double F = 2.0;
    anneal::ChainState<Vec> up;
    up.x = Vec(2);
    up.x = 0.0;
    up.V = {0.1, 0.1};
    up.V_baseline = up.V;
    for (std::size_t j = 0; j < 10; ++j) {
        anneal::apply_diagonal_ram(up, z, 1.0L, sp, F);
        CHECK(up.V[0] <= 0.1 * F * (1.0 + 1e-12));
    }
    CHECK(up.V[0] == Approx(0.1 * F));
    CHECK(up.V[1] == Approx(0.1 * F));

    anneal::ChainState<Vec> down;
    down.x = Vec(2);
    down.x = 0.0;
    down.V = {0.1, 0.1};
    down.V_baseline = down.V;
    for (std::size_t j = 0; j < 40; ++j) {
        anneal::apply_diagonal_ram(down, z, 0.0L, sp, F);
        CHECK(down.V[0] >= 0.1 / F * (1.0 - 1e-12));
    }
    CHECK(down.V[0] == Approx(0.1 / F));
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
        anneal::apply_diagonal_ram(c, z, 1.0L, sp);

    CHECK(c.V[0] > 1.0e-2);
    CHECK(c.V[0] == Approx(c.V[1]));
    CHECK(c.V[1] == Approx(c.V[2]));
    CHECK(c.diag_ram_proposals == 200);
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
     * exists to bound (observed in a long run: sigma/width pinned at ~2e-10 for 70k epochs). */
    const std::vector<double> z{1.0, 0.0};
    const double F = 8.0;

    auto bounded = make_chain();
    bounded.V_baseline = bounded.V;
    for (std::size_t j = 0; j < 5000; ++j) {
        anneal::apply_diagonal_ram(bounded, z, 0.0L, sp, F);
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
        anneal::apply_diagonal_ram(unbounded, z, 0.0L, sp, 0.0);
    CHECK(unbounded.V[0] < 0.1 / F);

    auto no_baseline = make_chain();
    for (std::size_t j = 0; j < 5000; ++j)
        anneal::apply_diagonal_ram(no_baseline, z, 0.0L, sp, F);
    CHECK(no_baseline.V[0] < 0.1 / F);
}

TEST_CASE("anneal::pinned_dims tolerance is the dim's actual proposal scale",
          "[HPO][anneal_core][pinned]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    OptimizerSpec sp;
    sp.low = {0.0, 0.0};
    sp.high = {1.0, 1.0};

    SECTION("CLASSIC: tol = sigma, no step-floor clause") {
        anneal::Knobs kb;
        kb.corana = false;
        anneal::ChainState<Vec> c;
        c.x = Vec(2);
        c.x(0) = 0.05;   /* gap to LOW = 0.05 <= sigma 0.1 */
        c.x(1) = 0.5;    /* mid-box: no pin */
        c.V = {0.1, 0.1};
        const auto pins = anneal::pinned_dims(c, sp, kb);
        REQUIRE(pins.size() == 1);
        CHECK(pins[0].dim == 0);
        CHECK_FALSE(pins[0].hi);
        CHECK(pins[0].gap == Approx(0.05));
        CHECK(pins[0].tol == Approx(0.1));      /* D22: the actual sigma */

        c.V = {0.04, 0.1};   /* gap 0.05 > sigma 0.04: one step clears it */
        CHECK(anneal::pinned_dims(c, sp, kb).empty());
    }

    SECTION("CORANA: tol = the V floor, pin only when V bottomed out") {
        anneal::Knobs kb;
        kb.corana = true;
        kb.v_floor_frac = 0.01;   /* floor = 0.01 * width = 0.01 */
        anneal::ChainState<Vec> c;
        c.x = Vec(2);
        c.x(0) = 0.995;  /* gap to HIGH = 0.005 <= floor 0.01 */
        c.x(1) = 0.5;
        c.V = {0.01, 0.5};   /* dim 0 at the floor */
        const auto pins = anneal::pinned_dims(c, sp, kb);
        REQUIRE(pins.size() == 1);
        CHECK(pins[0].dim == 0);
        CHECK(pins[0].hi);
        CHECK(pins[0].gap == Approx(0.005));
        CHECK(pins[0].tol == Approx(0.01));     /* D22: the V floor */

        /* V above the floor: adaptive shrinkage is not exhausted, so the
         * dim is not reported even at the same near-bound occupancy. */
        c.V = {0.05, 0.5};
        CHECK(anneal::pinned_dims(c, sp, kb).empty());
    }
}

TEST_CASE("anneal::run_rung retains the best graded FAILSCORE repair",
          "[HPO][anneal_core][rung][failscore][fail_grading]")
{
    for (bool corana : {false, true}) {
        CAPTURE(corana);
        auto sp = bare_spec();
        sp.failscore = 100.0;
        sp.param["T_INIT"] = 1.0e300;
        if (corana) {
            sp.param["CORANA"] = 1.0;
            sp.param["NS"] = 2.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.param["CHAIN_LENGTH"] = 2.0;
        }
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
        sp.param["T_INIT"] = 1.0;
        if (corana) {
            sp.param["CORANA"] = 1.0;
            sp.param["NS"] = 2.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.param["CHAIN_LENGTH"] = 2.0;
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
    sp.param["NT"] = 1.0;
    sp.param["STEP_FRAC"] = 0.01;
    sp.param["V_FLOOR_FRAC"] = 0.10;
    auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    auto v = anneal::make_initial_steps(sp, kb);
    REQUIRE(v.size() == 1);
    CHECK(v[0] == Approx(0.2));      /* raised to the V floor 0.10 * width */

    sp.param["STEP_FRAC"] = 1.0;     /* raw 2.0 = the full width cap */
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    v = anneal::make_initial_steps(sp, kb);
    CHECK(v[0] == Approx(2.0));

    sp = bare_spec();
    sp.low = {0.0};
    sp.high = {std::numeric_limits<double>::denorm_min()};
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    REQUIRE_THROWS_WITH(anneal::make_initial_steps(sp, kb),
                        Catch::Contains("proposal scale underflowed"));
    sp.param["STEP_FRAC"] = 1.0;     /* full width is still representable */
    kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    CHECK(anneal::make_initial_steps(sp, kb)[0] > 0.0);
}

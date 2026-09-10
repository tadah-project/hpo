// test_ladder_jitter.cpp
//
// Stage 12: INIT_JITTER_LADDER -- per-slot scaling of the INITIAL jitter,
// sigma_k = sigma_d * (T_k/T_hot)^gamma, AUTO (gamma = 0.5) by default.
//
// This is the one deliberate default-trajectory change in 1.5.0, so the tests
// here have two jobs that pull in opposite directions:
//
//   * pin the law and the floor guard exactly (the kernel cases), and
//   * establish that the default EARNS the trajectory change, rather than
//     asserting it does (the [.statistical] case).
//
// The measured picture is deliberately recorded in full below, including
// where the feature LOSES, because a test suite that only encoded the
// favourable half would invite someone to "restore" the default on the
// strength of one bad benchmark.

#include "catch2/catch.hpp"

#include "anneal_test_functions.h"
#include "anneal_test_support.h"

#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace tadah::hpo;
using anneal_test::Vec;

TEST_CASE("ladder jitter: the sqrt(T) law", "[HPO][anneal][ladder_jitter]")
{
    /* gamma = 0.5 is equipartition, not a tuned constant: the equilibrium
     * coordinate spread in a locally quadratic basin goes as sqrt(T). */
    CHECK(anneal::kLadderJitterGammaAuto == 0.5);

    /* The hottest slot always draws the full sigma -- it is the reference
     * the other slots are expressed against. */
    CHECK(anneal::ladder_jitter_factor(1.0, 1.0, 0.5) == Approx(1.0));
    /* A slot at T_hot/100 draws sigma/10 at gamma = 0.5. */
    CHECK(anneal::ladder_jitter_factor(0.01, 1.0, 0.5) == Approx(0.1));
    CHECK(anneal::ladder_jitter_factor(1e-6, 1.0, 0.5) == Approx(1e-3));
    /* gamma is the exponent, so it is the knob that changes the shape. */
    CHECK(anneal::ladder_jitter_factor(0.01, 1.0, 1.0) == Approx(0.01));
    CHECK(anneal::ladder_jitter_factor(0.01, 1.0, 0.25) == Approx(0.31622777));

    /* Scaling OFF is gamma <= 0 and must be an exact no-op, not "very close
     * to 1" -- INIT_JITTER_LADDER OFF has to reproduce the pre-1.5.0
     * placement bit-for-bit, and sigma * 0.9999999 would not. */
    CHECK(anneal::ladder_jitter_factor(1e-9, 1.0, 0.0) == 1.0);
    CHECK(anneal::ladder_jitter_factor(1e-9, 1.0, -1.0) == 1.0);

    /* A slot hotter than the reference is not amplified: the law shapes the
     * ladder downward from sigma_d, it never invents a larger displacement
     * than the caller asked for. */
    CHECK(anneal::ladder_jitter_factor(10.0, 1.0, 0.5) == 1.0);
}

TEST_CASE("ladder jitter: degenerate ladders fall back to the base sigma",
          "[HPO][anneal][ladder_jitter]")
{
    /* A non-finite or non-positive temperature means the caller has a
     * broken ladder, which the ladder validator reports far more usefully
     * than a NaN sigma would. Returning 1 keeps this feature from turning
     * that into an unplaceable replica. */
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double T : {0.0, -1.0, nan, inf}) {
        INFO("T_k = " << T);
        CHECK(anneal::ladder_jitter_factor(T, 1.0, 0.5) == 1.0);
        CHECK(anneal::ladder_jitter_factor(1.0, T, 0.5) == 1.0);
    }
}

TEST_CASE("ladder jitter: the resolvable-scale floor keeps slots distinct",
          "[HPO][anneal][ladder_jitter]")
{
    /* At a very wide ladder the cold slots' scaled sigma underflows the
     * coordinate resolution and every one of them lands exactly on the
     * seed -- K replicas collapsing to a handful of distinct points, which
     * is the failure this floor exists to prevent. */
    const std::vector<double> x0{0.5, 0.5};
    const std::vector<double> lo{0.0, 0.0}, hi{1.0, 1.0};
    const std::vector<double> base{0.1, 0.1};

    const auto ok = anneal::scale_jitter_for_slot(base, x0, lo, hi,
                                                  /*T_k=*/0.01, /*T_max=*/1.0,
                                                  0.5);
    CHECK(ok.floored == 0);
    CHECK(ok.sigma[0] == Approx(0.01));

    /* T_k/T_max = 1e-300 -> factor 1e-150; 0.1 * 1e-150 is far under the
     * ULP of 0.5, so both coordinates clamp. */
    const auto floored = anneal::scale_jitter_for_slot(base, x0, lo, hi,
                                                       1e-300, 1.0, 0.5);
    CHECK(floored.floored == 2);
    for (std::size_t i = 0; i < 2; ++i) {
        const double f = anneal::resolvable_coordinate_scale(x0[i], lo[i], hi[i]);
        CHECK(floored.sigma[i] == f);
        /* The whole point: the displacement is still REPRESENTABLE, so the
         * slot is a different point from the seed rather than a copy. */
        CHECK(x0[i] + floored.sigma[i] != x0[i]);
    }
}

TEST_CASE("ladder jitter: a sigma already below the floor is not counted",
          "[HPO][anneal][ladder_jitter]")
{
    /* A base sigma that was ALREADY sub-resolvable is the jitter
     * calibration's verdict about that coordinate, not something ladder
     * scaling did. Counting it would make the journal blame this feature
     * for a collapse that happens with INIT_JITTER_LADDER OFF too, and the
     * warning's whole purpose is to say "your ladder is too wide". */
    const std::vector<double> x0{0.5};
    const std::vector<double> lo{0.0}, hi{1.0};
    const std::vector<double> base{1e-300};
    const auto r = anneal::scale_jitter_for_slot(base, x0, lo, hi,
                                                 0.01, 1.0, 0.5);
    CHECK(r.floored == 0);
    CHECK(r.sigma[0] == Approx(1e-301));
}

TEST_CASE("ladder jitter: OFF short-circuits before touching any sigma",
          "[HPO][anneal][ladder_jitter]")
{
    const std::vector<double> x0{0.5}, lo{0.0}, hi{1.0}, base{0.1};
    const auto r = anneal::scale_jitter_for_slot(base, x0, lo, hi,
                                                 1e-300, 1.0, /*gamma=*/0.0);
    CHECK(r.floored == 0);
    CHECK(r.sigma == base);          /* identical, not merely equal-ish */
}

TEST_CASE("ladder jitter: PARAM vocabulary and gating",
          "[HPO][anneal][ladder_jitter][registry]")
{
    auto base = [] {
        OptimizerSpec sp;
        sp.low = {-1.0}; sp.high = {1.0};
        sp.is_int.assign(1, false); sp.log_scaled.assign(1, false);
        sp.lib = "TADAH"; sp.algo = "ANNEAL_PT";
        sp.optim_init.assign(1, std::vector<double>{0.0});
        return sp;
    };

    SECTION("AUTO is the default and resolves to the equipartition gamma") {
        const auto kb = anneal::Knobs::parse(base(), anneal::AlgoKind::PT);
        CHECK(kb.init_jitter_ladder_auto);
        CHECK(kb.init_jitter_ladder == anneal::kLadderJitterGammaAuto);
    }
    SECTION("OFF disables it") {
        auto sp = base();
        sp.param_raw["INIT_JITTER_LADDER"] = "OFF";
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK_FALSE(kb.init_jitter_ladder_auto);
        CHECK(kb.init_jitter_ladder == 0.0);
    }
    SECTION("an explicit gamma overrides AUTO") {
        auto sp = base();
        sp.param["INIT_JITTER_LADDER"] = 1.25;
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK_FALSE(kb.init_jitter_ladder_auto);
        CHECK(kb.init_jitter_ladder == 1.25);
    }
    SECTION("a bare 0 is rejected with the 1.5.0 migration hint") {
        /* gamma = 0 happens to mean the same thing as OFF, which is exactly
         * why it must not be silently accepted: the old and the new
         * spelling would then be indistinguishable in a config file. */
        auto sp = base();
        sp.param["INIT_JITTER_LADDER"] = 0.0;
        REQUIRE_THROWS_WITH(
            anneal::Knobs::parse(sp, anneal::AlgoKind::PT),
            Catch::Contains("removed in 1.5.0") &&
            Catch::Contains("INIT_JITTER_LADDER OFF"));
    }
    SECTION("an out-of-range gamma names the accepted tokens") {
        auto sp = base();
        sp.param["INIT_JITTER_LADDER"] = 50.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::PT),
                            Catch::Contains("must be in (0,4]") &&
                            Catch::Contains("AUTO"));
    }
    SECTION("it is rejected under INIT_MODE UNIFORM, which has no jitter") {
        auto sp = base();
        sp.param_raw["INIT_MODE"] = "UNIFORM";
        sp.param_raw["INIT_JITTER_LADDER"] = "OFF";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::PT),
                            Catch::Contains("unused when INIT_MODE is UNIFORM"));
    }
    SECTION("it is a PT key: PA rejects it") {
        auto sp = base();
        sp.algo = "ANNEAL_PA";
        sp.param_raw["INIT_JITTER_LADDER"] = "AUTO";
        REQUIRE_THROWS(anneal::Knobs::parse(sp, anneal::AlgoKind::PA));
    }
}

namespace {

/* One PT run from `start`, returning the best feasible loss seen. */
double run_from(const anneal_functions::Problem& prob,
                const std::vector<double>& start, std::uint64_t seed,
                bool ladder, double span)
{
    OptimizerSpec sp;
    sp.low = prob.low; sp.high = prob.high;
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib = "TADAH"; sp.algo = "ANNEAL_PT";
    sp.optim_init.assign(1, start);
    sp.seed = seed;
    sp.num_threads = 1;
    sp.stop.maxeval = 300 * sp.low.size();
    sp.param["REPLICAS"] = 8.0;
    sp.param["SWAP_WINDOW"] = 1e6;          /* mute the per-window reports */
    sp.param["LADDER_SPAN"] = span;
    if (!ladder) sp.param_raw["INIT_JITTER_LADDER"] = "OFF";
    double best = std::numeric_limits<double>::infinity();
    anneal_test::CountingHost host{[&](const Vec& x) {
        const double f = prob.loss(x);
        if (f < best) best = f;
        return f;
    }};
    run_anneal_pt_impl<Vec>(sp, host);
    return best;
}

}  // namespace

TEST_CASE("ladder jitter: the default earns its trajectory change on a "
          "warm start", "[.statistical][HPO][anneal][ladder_jitter]")
{
    /* THE justification for turning this on by default.
     *
     * Tadah's PT runs are warm-started: OPTIM_INIT is a fitted potential,
     * and it is the best point anyone knows. Without ladder scaling every
     * non-seed slot -- the cold ones included -- is displaced by the SAME
     * sigma, so the replica whose job is to refine that potential starts as
     * far from it as the hottest explorer, and the warm start is thrown
     * away everywhere except slot 1.
     *
     * Measured here, paired by seed on a wide ladder with a good seed:
     * ellipsoid6, LADDER_SPAN 1e6, seed at 2% of the default start.
     *
     * WHERE IT LOSES (measured, recorded so nobody has to rediscover it):
     * rosenbrock6 at seed_frac 0.5 and span 1e6 is 1.22x WORSE with
     * scaling on, losing on 40/40 paired seeds. That is the honest shape of
     * the feature and not a defect -- a curved valley reached from a
     * MEDIOCRE seed is exactly the case where you want the cold slots to
     * spread out and hunt, and it is what INIT_JITTER_LADDER OFF is for. */
    const auto problems = anneal_functions::standard_set();
    const auto& prob = anneal_functions::by_name(problems, "ellipsoid6");
    std::vector<double> start(prob.low.size());
    for (std::size_t i = 0; i < start.size(); ++i) start[i] = 0.02 * prob.start[i];

    constexpr int kSeeds = 40;
    std::vector<double> on, off;
    int on_better = 0;
    for (int s = 0; s < kSeeds; ++s) {
        anneal_test::JournalCapture lr;
        const double a = run_from(prob, start, 7000 + s, true, 1e6);
        const double b = run_from(prob, start, 7000 + s, false, 1e6);
        on.push_back(a);
        off.push_back(b);
        if (a < b) ++on_better;
    }
    std::sort(on.begin(), on.end());
    std::sort(off.begin(), off.end());
    const double median_on = on[on.size() / 2];
    const double median_off = off[off.size() / 2];
    INFO("median ON=" << median_on << "  OFF=" << median_off
         << "  ratio=" << median_on / median_off
         << "  ON better on " << on_better << "/" << kSeeds << " paired seeds");
    /* Observed 2026-09-10: ratio 0.25x, ON better on 38/40. The thresholds
     * are loose enough to be a claim about the EFFECT, not a fingerprint of
     * this build: a sign flip or a collapse to parity fails, ordinary
     * seed-to-seed movement does not. */
    CHECK(median_on < 0.6 * median_off);
    CHECK(on_better >= 30);
}

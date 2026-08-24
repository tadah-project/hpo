/*
 * Layer-A validation of the parallel-tempering driver
 * ([.statistical] except the deterministic swap-log check).
 *
 *  - Swap invariance: exchanges must PRESERVE each slot's marginal
 *    (Var = T_k on a quadratic); any defect in what a swap moves shows
 *    up here as a marginal distortion.
 *  - Analytic swap acceptance: on the d=2 harmonic target the per-pair
 *    acceptance has the closed form 2R/(1+R), R = T_cold/T_hot
 *    (derived below; verified against an independent i.i.d. Monte Carlo
 *    at design time).
 *  - DEO transport: production round-trip counts are compared against
 *    an independent index-process simulator driven by the run's own
 *    measured pair acceptances, and must be closer to the DEO model
 *    than to a reversible (random-parity) variant.
 *  - FROZEN / ADAPTIVE step modes: the calibrated (and adapting)
 *    kernels still sample the right marginals.
 *
 * All samples come from the ChainObserver's pre-swap rung ends: the
 * joint chain (sweeps + exchanges) is stationary, so the state at a
 * fixed point of the cycle is a valid draw from the slot marginal.
 */

#include "catch2/catch.hpp"

#include "anneal_stats.h"
#include "anneal_test_support.h"

#include <tadah/hpo/algorithms_pt.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace tadah::hpo;
using anneal_test::Vec;

namespace {

struct SlotSeries {
    std::vector<std::vector<double>> x_by_slot;   /* first coordinate */
    std::vector<std::vector<double>> f_by_slot;
};

SlotSeries slot_series(const anneal_test::RecordingObserver& obs,
                       std::size_t K, std::size_t burn_epochs)
{
    SlotSeries s;
    s.x_by_slot.resize(K);
    s.f_by_slot.resize(K);
    /* rungs arrive K per epoch (any order within the epoch). */
    const std::size_t epochs = obs.rungs.size() / K;
    for (std::size_t e = burn_epochs; e < epochs; ++e)
        for (std::size_t j = 0; j < K; ++j) {
            const auto& r = obs.rungs[e * K + j];
            if (!r.x.empty())
                s.x_by_slot[std::size_t(r.slot)].push_back(r.x[0]);
            s.f_by_slot[std::size_t(r.slot)].push_back(r.f);
        }
    return s;
}

}  // namespace

TEST_CASE("PT exchanges preserve each slot's Gaussian marginal",
          "[.statistical][HPO][anneal_pt][equilibrium]")
{
    /* d=1 quadratic f = x^2/2: slot k must sample Var = T_k. K=4 with
     * vigorous swapping -- a swap bug (wrong field exchanged, stale f,
     * step state travelling with the walker) distorts these marginals.
     * The box +/-10*sqrt(T_hot) truncates ~1e-23 of the hottest mass. */
    anneal_test::JournalCapture lr;
    anneal_test::RecordingObserver obs;
    obs.keep_x = true;
    auto sp = anneal_test::pt_spec({-10.0}, {10.0});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.anneal_observer = &obs;
    sp.seed = 5001;
    sp.num_threads = 2;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.125;    /* geometric: 0.125, 0.25, 0.5, 1 */
    sp.param["STEP_FRAC"] = 0.06;  /* sigma = 1.2 for every slot      */
    sp.param["SWAP_WINDOW"] = 1000000.0;   /* mute periodic reports   */
    const std::size_t epochs = 30000, burn = 3000;
    sp.stop.maxeval = 4 + 4 * epochs;

    anneal_test::CountingHost host{[](const Vec& x) {
        return 0.5 * x(0) * x(0);
    }};
    run_anneal_pt_impl<Vec>(sp, host);

    const auto s = slot_series(obs, 4, burn);
    const double T[4] = {0.125, 0.25, 0.5, 1.0};
    for (std::size_t k = 0; k < 4; ++k) {
        REQUIRE(s.x_by_slot[k].size() > 20000);
        const auto m = anneal_stats::mean_of(s.x_by_slot[k]);
        CHECK_STAT("pt_slot_mean_" + std::to_string(k), m.mean, 0.0, m.se);
        const auto v = anneal_stats::var_of(s.x_by_slot[k]);
        CHECK_STAT("pt_slot_var_" + std::to_string(k), v.mean, T[k], v.se);
    }
}

TEST_CASE("PT swap log is reproducible and DEO-structured",
          "[HPO][anneal_pt][swap][determinism]")
{
    auto run_once = [](std::uint64_t seed) {
        anneal_test::JournalCapture lr;
        anneal_test::RecordingObserver obs;
        auto sp = anneal_test::pt_spec({-1.0}, {1.0});
        sp.param_raw["STEP_MODE"] = "FIXED";
        sp.anneal_observer = &obs;
        sp.seed = seed;
        sp.num_threads = 2;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        sp.stop.maxeval = 4 + 4 * 50;
        anneal_test::CountingHost host{[](const Vec& x) {
            return x(0) * x(0);
        }};
        run_anneal_pt_impl<Vec>(sp, host);
        return obs.swaps;
    };
    const auto a = run_once(5002);
    const auto b = run_once(5002);
    REQUIRE(a.size() == b.size());
    REQUIRE_FALSE(a.empty());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].epoch == b[i].epoch);
        CHECK(a[i].pair == b[i].pair);
        CHECK(a[i].accepted == b[i].accepted);
        CHECK((a[i].pair % 2 == a[i].epoch % 2));   /* DEO parity */
    }
}

TEST_CASE("PT pair swap acceptance matches the harmonic closed form",
          "[.statistical][HPO][anneal_pt][swap][equilibrium]")
{
    /* d=2 harmonic f = |x|^2/2 at temperature T: the energy is a sum of
     * two squared Gaussians, i.e. Exponential with mean T. For an
     * exchange between equilibrated slots at T_c < T_h with
     * independent f_c ~ Exp(T_c), f_h ~ Exp(T_h) and
     * dbeta = 1/T_c - 1/T_h:
     *
     *   P = P(f_h < f_c) + E[exp(-dbeta (f_h - f_c)); f_h > f_c]
     *     = (1 - a/(a+b)) + b/(a+b)          with a = 1/T_c, b = 1/T_h
     *     = 2b/(a+b) = 2R/(1+R),             R = T_c/T_h.
     *
     * (Both integrals are elementary; the closed form was additionally
     * verified against an independent i.i.d. Monte Carlo at design
     * time: R in {0.2, 0.5, 0.8, 0.215} matched to ~5e-4 at n=4e5.)
     * The production measurement uses only NON-degenerate attempts
     * after burn-in; degenerate attempts must not occur on this smooth
     * target. */
    for (const double R : {0.5, 0.8}) {
        DYNAMIC_SECTION("R=" << R) {
            anneal_test::JournalCapture lr;
            anneal_test::RecordingObserver obs;
            auto sp = anneal_test::pt_spec({-10.0, -10.0}, {10.0, 10.0});
            sp.param_raw["STEP_MODE"] = "FIXED";
            sp.anneal_observer = &obs;
            sp.seed = 5003 + std::uint64_t(R * 10);
            sp.num_threads = 2;
            sp.param["REPLICAS"] = 2.0;
            sp.param["T_HOT"] = 1.0;
            sp.param["T_COLD"] = R;
            sp.param["STEP_FRAC"] = 0.06;
            sp.param["SWAP_WINDOW"] = 1000000.0;
            const std::size_t epochs = 40000, burn = 4000;
            /* CLASSIC sweep = one whole-vector proposal, so an epoch of
             * K=2 replicas costs 2 evals; +2 for the seed and the
             * second slot's initialisation. */
            sp.stop.maxeval = 2 + 2 * epochs;

            anneal_test::CountingHost host{[](const Vec& x) {
                return 0.5 * (x(0) * x(0) + x(1) * x(1));
            }};
            run_anneal_pt_impl<Vec>(sp, host);

            std::vector<double> outcomes;
            for (const auto& s : obs.swaps) {
                if (s.epoch <= burn) continue;
                REQUIRE_FALSE(s.degenerate);
                outcomes.push_back(s.accepted ? 1.0 : 0.0);
            }
            REQUIRE(outcomes.size() > 10000);
            const auto p = anneal_stats::mean_of(outcomes);
            CHECK_STAT("pt_swap_acc_R" + std::to_string(R), p.mean,
                       2.0 * R / (1.0 + R), p.se);
        }
    }
}

TEST_CASE("DEO transport matches an independent index-process simulator",
          "[.statistical][HPO][anneal_pt][transport]")
{
    /* K=4 geometric ladder (equal pair ratio => equal pair acceptance).
     * The production run's walkers are tracked by replaying its swap
     * log; round trips (slot 0 -> slot 3 -> slot 0) are counted per
     * walker. The reference is an abstract DEO index process driven by
     * the run's own MEASURED per-pair acceptances with independent
     * Bernoulli outcomes -- an approximation (real outcomes are
     * state-correlated), so the gates are (a) the production count lies
     * within a factor band of the DEO model's mean and (b) it is closer
     * to the DEO model than to a reversible random-parity variant,
     * with the two models separated by their own spreads. */
    anneal_test::JournalCapture lr;
    anneal_test::RecordingObserver obs;
    auto sp = anneal_test::pt_spec({-10.0, -10.0}, {10.0, 10.0});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.anneal_observer = &obs;
    sp.seed = 5005;
    sp.num_threads = 2;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.125;   /* pair ratio 0.5 -> p = 2/3 */
    sp.param["STEP_FRAC"] = 0.06;
    sp.param["SWAP_WINDOW"] = 1000000.0;
    const std::size_t epochs = 40000, burn = 2000;
    sp.stop.maxeval = 4 + 4 * epochs;

    anneal_test::CountingHost host{[](const Vec& x) {
        return 0.5 * (x(0) * x(0) + x(1) * x(1));
    }};
    run_anneal_pt_impl<Vec>(sp, host);

    constexpr std::size_t K = 4;
    /* Measured per-pair acceptance (non-degenerate, post-burn). */
    std::size_t att[K - 1] = {0, 0, 0}, acc[K - 1] = {0, 0, 0};
    for (const auto& s : obs.swaps) {
        if (s.epoch <= burn) continue;
        ++att[s.pair];
        if (s.accepted) ++acc[s.pair];
    }
    double p_meas[K - 1];
    for (std::size_t i = 0; i + 1 < K; ++i) {
        REQUIRE(att[i] > 5000);
        p_meas[i] = double(acc[i]) / double(att[i]);
    }

    /* Round trips: track walkers through a sequence of (epoch, pair)
     * accepted swaps. The production replay applies EXACTLY the logged
     * outcomes (no schedule assumptions); the models generate outcomes
     * on the driver's own DEO schedule, parity = epoch %% 2 with epochs
     * counted from 1 (or random parity for the reversible variant). */
    struct WalkerTracker {
        std::vector<std::size_t> pos, at;
        std::vector<int> phase;
        double trips = 0.0;
        explicit WalkerTracker(std::size_t k)
            : pos(k), at(k), phase(k, 0)
        {
            std::iota(pos.begin(), pos.end(), 0);
            std::iota(at.begin(), at.end(), 0);
        }
        void swap_pair(std::size_t i)
        {
            std::swap(at[i], at[i + 1]);
            pos[at[i]] = i;
            pos[at[i + 1]] = i + 1;
        }
        void end_round(std::size_t k)
        {
            for (std::size_t w = 0; w < k; ++w) {
                if (pos[w] == k - 1) phase[w] = 1;
                else if (pos[w] == 0 && phase[w] == 1) {
                    phase[w] = 0;
                    trips += 1.0;
                }
            }
        }
    };

    std::size_t max_epoch = 0;
    for (const auto& s : obs.swaps)
        max_epoch = std::max(max_epoch, s.epoch);
    double prod_trips = 0.0;
    {
        WalkerTracker wt(K);
        std::size_t idx = 0;
        /* obs.swaps is in driver order: epochs ascend, pairs ascend
         * within an epoch (single-process run). */
        for (std::size_t e = 1; e <= max_epoch; ++e) {
            while (idx < obs.swaps.size() && obs.swaps[idx].epoch == e) {
                if (obs.swaps[idx].accepted)
                    wt.swap_pair(obs.swaps[idx].pair);
                ++idx;
            }
            wt.end_round(K);
        }
        prod_trips = wt.trips;
    }

    /* Independent DEO / reversible index-process models at the measured
     * rates. 400 replays give the model means and spreads. */
    auto model = [&](bool deo, std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> counts;
        for (int rep = 0; rep < 400; ++rep) {
            WalkerTracker wt(K);
            for (std::size_t e = 1; e <= max_epoch; ++e) {
                const std::size_t parity =
                    deo ? (e % 2) : (u(rng) < 0.5 ? 0 : 1);
                for (std::size_t i = parity; i + 1 < K; i += 2)
                    if (u(rng) < p_meas[i]) wt.swap_pair(i);
                wt.end_round(K);
            }
            counts.push_back(wt.trips);
        }
        double m = 0.0;
        for (double c : counts) m += c;
        m /= double(counts.size());
        double s2 = 0.0;
        for (double c : counts) s2 += (c - m) * (c - m);
        s2 /= double(counts.size() - 1);
        return std::pair<double, double>(m, std::sqrt(s2));
    };

    const auto [mu_deo, sd_deo] = model(true, 99001);
    const auto [mu_rev, sd_rev] = model(false, 99002);

    INFO("production=" << prod_trips << "  DEO=" << mu_deo << "+-" << sd_deo
                       << "  REV=" << mu_rev << "+-" << sd_rev);
    /* The two models must be distinguishable before "closer to DEO"
     * means anything. */
    REQUIRE(mu_deo - mu_rev > 3.0 * (sd_deo + sd_rev));
    CHECK(prod_trips > 0.5 * mu_deo);
    CHECK(prod_trips < 1.5 * mu_deo);
    CHECK(std::fabs(prod_trips - mu_deo) < std::fabs(prod_trips - mu_rev));
}

TEST_CASE("calibrated step kernels still sample the right marginals",
          "[.statistical][HPO][anneal_pt][equilibrium][step_auto]")
{
    /* FROZEN: warm-up calibrates per-slot sigma, then the kernel is a
     * plain fixed Metropolis kernel -- marginals at 5*SE. ADAPTIVE:
     * adaptation diminishes but never stops, so the chain is only
     * asymptotically exact -- gate at 10*SE. Explicit temperatures skip
     * the second calibration pass, keeping the references closed-form. */
    for (const bool adaptive : {false, true}) {
        DYNAMIC_SECTION((adaptive ? "ADAPTIVE" : "FROZEN")) {
            anneal_test::JournalCapture lr;
            anneal_test::RecordingObserver obs;
            obs.keep_x = true;
            auto sp = anneal_test::pt_spec({-10.0}, {10.0});
            if (adaptive) sp.param_raw["STEP_MODE"] = "ADAPTIVE";
            /* else: FROZEN default */
            sp.anneal_observer = &obs;
            sp.seed = adaptive ? 5007 : 5006;
            sp.num_threads = 2;
            sp.param["REPLICAS"] = 2.0;
            sp.param["T_HOT"] = 1.0;
            sp.param["T_COLD"] = 0.25;
            sp.param["SWAP_WINDOW"] = 1000000.0;
            const std::size_t epochs = 30000, burn = 5000;
            sp.stop.maxeval = 5000 + 2 + 2 * epochs;   /* + warm-up */

            anneal_test::CountingHost host{[](const Vec& x) {
                return 0.5 * x(0) * x(0);
            }};
            run_anneal_pt_impl<Vec>(sp, host);

            /* Production epochs only: drop everything before the first
             * observed epoch boundary (warm-up does not emit rung
             * ends), then burn. */
            const auto s = slot_series(obs, 2, burn);
            const double T[2] = {0.25, 1.0};
            for (std::size_t k = 0; k < 2; ++k) {
                REQUIRE(s.x_by_slot[k].size() > 15000);
                const auto v = anneal_stats::var_of(s.x_by_slot[k]);
                if (adaptive)
                    CHECK_STAT_RELAXED(
                        "pt_adaptive_var_" + std::to_string(k), v.mean,
                        T[k], v.se, 10.0);
                else
                    CHECK_STAT("pt_frozen_var_" + std::to_string(k),
                               v.mean, T[k], v.se);
            }
        }
    }
}

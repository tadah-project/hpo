/*
 * Unit tests for `run_anneal_pt_impl` (TADAH:ANNEAL_PT -- parallel
 * tempering over the shared annealing core, algorithms_pt.h).
 *
 * A thread-safe stub host records every parameter vector. With REPLICAS
 * fixed, a deterministic objective, and a MAXEVAL-only stop, the tests
 * assert that the same SEED gives the same SET of evaluated points for
 * any thread count (call ORDER differs; per-chain sequences are fixed).
 */

#include "catch2/catch.hpp"
#include <tadah/hpo/log_index.h>

#include <tadah/hpo/algorithms.h>      /* serial control in the double-well test */
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/core/checkpoint.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

tadah::hpo::OptimizerSpec
make_pt_spec(std::vector<double> low, std::vector<double> high)
{
    tadah::hpo::OptimizerSpec sp;
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib  = "TADAH";
    sp.algo = "ANNEAL_PT";
    /* Most tests exercise the fixed-step trajectory explicitly (no
     * calibration warm-up). Warm-up cases erase this fixture override
     * to get the FROZEN default (or set ADAPTIVE). */
    sp.param_raw["STEP_MODE"] = "FIXED";
    return sp;
}

void set_seed_point(tadah::hpo::OptimizerSpec& sp,
                    const std::vector<double>& x)
{
    sp.optim_init.assign(1, x);
}

/* Thread-safe recording stub: replicas evaluate concurrently. */
struct MtStubHost {
    std::function<double(const Vec&)> f;
    mutable std::mutex m;
    mutable std::vector<std::vector<double>> calls;
    double operator()(const Vec& x) const {
        std::vector<double> v(static_cast<std::size_t>(x.size()));
        for (long i = 0; i < x.size(); ++i) v[static_cast<std::size_t>(i)] = x(i);
        const double r = f(x);
        {
            std::lock_guard<std::mutex> lk(m);
            calls.push_back(std::move(v));
        }
        return r;
    }
    std::vector<std::vector<double>> sorted_calls() const {
        std::lock_guard<std::mutex> lk(m);
        auto c = calls;
        std::sort(c.begin(), c.end());
        return c;
    }
};

struct LoggedRun {
    std::filesystem::path path;
    LoggedRun() : path(std::filesystem::temp_directory_path() /
                       ("anneal_pt_log_" +
                        std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()) +
                        ".log"))
    {
        tadah::hpo::HPOLog::instance().reset_for_test();
        tadah::hpo::HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~LoggedRun() {
        tadah::hpo::HPOLog::instance().finish("test-end");
        tadah::hpo::HPOLog::instance().reset_for_test();
        /* Formatting aid: dump the journal before deleting it. */
        if (std::getenv("TADAH_TEST_DUMP_LOG"))
            std::cout << "\n===== " << path.string() << " =====\n" << read();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string read() const {
        std::ifstream is(path);
        std::ostringstream oss; oss << is.rdbuf();
        return oss.str();
    }
};

double quad_bowl2(const Vec& x, double cx, double cy)
{
    const double dx = x(0) - cx, dy = x(1) - cy;
    return dx * dx + dy * dy;
}

struct TempWorkingDirectory {
    std::filesystem::path old_path;
    std::filesystem::path path;
    TempWorkingDirectory()
        : old_path(std::filesystem::current_path()),
          path(std::filesystem::temp_directory_path() /
               ("anneal_pt_steps_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count())))
    {
        std::filesystem::create_directories(path);
        std::filesystem::current_path(path);
    }
    ~TempWorkingDirectory() {
        std::filesystem::current_path(old_path);
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct StepScaleRow {
    std::size_t slot = 0;
    double temperature = 0.0;
    std::string parameter;
    double sigma = 0.0;
    double fraction = 0.0;
    /* The coordinate phase's own sigma, alongside the production one --
     * every column from `acceptance` on was measured at THIS scale. */
    double coord_sigma = 0.0;
    double coord_fraction = 0.0;
    double acceptance = 0.0;
    std::string status;
    double mean_alpha = 0.0;
    std::size_t attempts = 0;
    std::size_t backoffs = 0;
    std::size_t unusable = 0;
    double pilot_multiplier = 1.0;
    int pilot_multiplier_moved = 0;   /* whether the pilot moved it */
    double pilot_acceptance = 0.0;
    double pilot_mean_alpha = 0.0;
    std::size_t pilot_attempts = 0;
    std::size_t pilot_backoffs = 0;
    std::size_t pilot_unusable = 0;
    std::string pilot_status;
};

std::vector<StepScaleRow> read_step_scales()
{
    std::ifstream input("hpo_step_scales.tsv");
    std::string line;
    std::getline(input, line);
    std::vector<StepScaleRow> rows;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream fields(line);
        StepScaleRow row;
        fields >> row.slot >> row.temperature >> row.parameter
               >> row.sigma >> row.fraction
               >> row.coord_sigma >> row.coord_fraction
               >> row.acceptance >> row.status
               >> row.mean_alpha >> row.attempts >> row.backoffs
               >> row.unusable >> row.pilot_multiplier
               >> row.pilot_multiplier_moved
               >> row.pilot_acceptance >> row.pilot_mean_alpha
               >> row.pilot_attempts >> row.pilot_backoffs
               >> row.pilot_unusable >> row.pilot_status;
        /* The `slot` column is user-facing and therefore 1-based (1.5.0);
         * convert back so the rest of the test indexes arrays with it. */
        if (fields) {
            row.slot = tadah::hpo::logidx::from_ui(
                row.slot, "hpo_step_scales.tsv slot");
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

}  // namespace

TEST_CASE("ANNEAL_PT: INIT_MODE JITTER spreads slots >= 1 for every stepper "
          "and STEP_MODE", "[HPO][anneal_pt][init][kernel]")
{
    using namespace tadah::hpo;
    /* Jitter sigma is the initial proposal scale STEP_FRAC * width. Slot 0
     * is always exactly the seed; every other slot is jittered around it --
     * for CLASSIC FIXED, CLASSIC FROZEN (the old "shared warm seed under
     * STEP_AUTO" no-op is gone) and CORANA alike, each init costing one eval.
     *
     * Since 1.5.0 the displacement is also LADDER-SCALED by default
     * (INIT_JITTER_LADDER AUTO): slot k draws from sigma*(T_k/T_hot)^0.5, so
     * the cold slots start near the seed and the hot ones explore. The
     * expected ladder is written out in closed form here rather than taken
     * from make_temperature_ladder, so this stays an independent check of
     * both the ladder and the scaling law. */
    constexpr double kJitterSigma = 0.0125 * 20.0;
    constexpr double kTHot = 1.0, kTCold = 0.1;
    constexpr std::size_t kK = 4;
    enum class Mode { ClassicFixed, ClassicFrozen, Corana };
    auto run = [](Mode mode, bool ladder) {
        auto sp = make_pt_spec({-10.0}, {10.0});
        set_seed_point(sp, {0.0});
        sp.seed = 314159;
        sp.num_threads = 2;
        sp.stop.maxeval = 4;   /* seed + 3 jittered slot inits, nothing else */
        sp.param["STEP_FRAC"] = 0.0125;
        sp.param["REPLICAS"] = static_cast<double>(kK);
        sp.param["T_HOT"] = kTHot;
        sp.param["T_COLD"] = kTCold;
        if (!ladder) sp.param_raw["INIT_JITTER_LADDER"] = "OFF";
        if (mode == Mode::Corana) {
            sp.param_raw.erase("STEP_MODE");
            sp.param_raw["CORANA"] = "ON";
            sp.param["NS"] = 1.0;
        } else if (mode == Mode::ClassicFrozen) {
            sp.param_raw["STEP_MODE"] = "FROZEN";
        }
        MtStubHost host{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, host);
        return host.sorted_calls();
    };

    /* Geometric ladder, LADDER_POWER 1: T_k = T_cold * (T_hot/T_cold)^(k/(K-1)). */
    auto slot_temperature = [](std::size_t k) {
        return kTCold * std::pow(kTHot / kTCold,
                                 static_cast<double>(k) / (kK - 1));
    };

    auto expected_starts = [&](bool ladder) {
        std::vector<std::vector<double>> expected{{0.0}};
        for (std::size_t k = 1; k < kK; ++k) {
            auto rng = anneal::make_stream(314159, anneal::SALT_CHAIN, k);
            const double unit =
                std::normal_distribution<double>(0.0, 1.0)(rng);
            const double factor =
                ladder ? std::sqrt(slot_temperature(k) / kTHot) : 1.0;
            expected.push_back({kJitterSigma * factor * unit});
        }
        std::sort(expected.begin(), expected.end());
        return expected;
    };

    SECTION("ladder scaling on (the 1.5.0 default)") {
        /* The scaling must be a real gradient, not a rounding-level effect:
         * the coldest non-seed slot draws from under half the hottest
         * slot's sigma on this ladder. Asserted so a factor that silently
         * degenerated to 1 could not pass the coordinate checks below by
         * being "close enough". */
        CHECK(std::sqrt(slot_temperature(1) / kTHot) < 0.5);
        CHECK(std::sqrt(slot_temperature(kK - 1) / kTHot) == Approx(1.0));

        for (Mode mode : {Mode::ClassicFixed, Mode::ClassicFrozen,
                          Mode::Corana}) {
            CAPTURE(static_cast<int>(mode));
            const auto expected = expected_starts(true);
            std::size_t at_seed = 0;
            for (const auto& e : expected) if (e[0] == 0.0) ++at_seed;
            CHECK(at_seed == 1);   /* every non-seed start still moved */
            const auto actual = run(mode, true);
            REQUIRE(actual.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i)
                CHECK(actual[i][0] == Approx(expected[i][0]).margin(1e-15));
        }
    }

    SECTION("INIT_JITTER_LADDER OFF reproduces the pre-1.5.0 placement") {
        /* The documented escape hatch, pinned at the tightest level there
         * is: the exact coordinates, which are sigma*unit with no factor --
         * literally the expression the pre-stage test asserted. */
        for (Mode mode : {Mode::ClassicFixed, Mode::ClassicFrozen,
                          Mode::Corana}) {
            CAPTURE(static_cast<int>(mode));
            const auto expected = expected_starts(false);
            const auto actual = run(mode, false);
            REQUIRE(actual.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i)
                CHECK(actual[i][0] == Approx(expected[i][0]).margin(1e-15));
        }
        /* ... and it is genuinely a different placement from the default. */
        CHECK(expected_starts(false) != expected_starts(true));
    }
}

TEST_CASE("ANNEAL_PT: fixed-size thread-count invariance (same SEED => same eval set)",
          "[HPO][anneal_pt][determinism]")
{
    using namespace tadah::hpo;
    auto run_with_threads = [&](unsigned threads) {
        auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
        set_seed_point(sp, {0.5, -0.5});
        sp.seed = 4242;
        sp.num_threads = threads;
        sp.stop.maxeval = 600;
        sp.param["REPLICAS"]  = 8.0;
        sp.param_raw.erase("STEP_MODE");
        sp.param_raw["CORANA"]    = "ON";
        sp.param["NS"]        = 2.0;
        sp.param["T_HOT"]     = 1.0;   /* skip probes: fixed prefix */
        sp.param["LADDER_SPAN"] = 100.0;
        MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, 0.1, -0.2); }, {}, {} };
        run_anneal_pt_impl<Vec>(sp, stub);
        return stub.sorted_calls();
    };

    const auto a = run_with_threads(1);
    const auto b = run_with_threads(2);
    const auto c = run_with_threads(8);

    REQUIRE_FALSE(a.empty());
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == c.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].size() == b[i].size());
        for (std::size_t d = 0; d < a[i].size(); ++d) {
            CHECK(a[i][d] == Approx(b[i][d]).margin(1e-15));
            CHECK(a[i][d] == Approx(c[i][d]).margin(1e-15));
        }
    }
}

TEST_CASE("ANNEAL_PT: adaptive ladder is thread-count deterministic",
          "[HPO][anneal_pt][determinism][ladder]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto run_with_threads = [&](unsigned threads) {
        auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
        set_seed_point(sp, {0.5, -0.5});
        sp.seed = 4243;
        sp.num_threads = threads;
        sp.stop.maxeval = 240;
        sp.param["REPLICAS"] = 8.0;
        sp.param_raw.erase("STEP_MODE");
        sp.param_raw["CORANA"] = "ON";
        sp.param["NS"] = 1.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["LADDER_SPAN"] = 100.0;
        sp.param["LADDER_POWER"] = 0.5;
        sp.param_raw["LADDER_ADAPT"] = "ON";
        sp.param["LADDER_ADAPT_EVERY"] = 2.0;
        MtStubHost stub{
            [&](const Vec& x) { return quad_bowl2(x, 0.1, -0.2); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, stub);
        return stub.sorted_calls();
    };

    const auto a = run_with_threads(1);
    const auto b = run_with_threads(2);
    const auto c = run_with_threads(8);
    REQUIRE(a.size() == 240);
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == c.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].size() == b[i].size());
        for (std::size_t d = 0; d < a[i].size(); ++d) {
            CHECK(a[i][d] == Approx(b[i][d]).margin(1e-15));
            CHECK(a[i][d] == Approx(c[i][d]).margin(1e-15));
        }
    }
}

TEST_CASE("ANNEAL_PT: cold replica escapes a deep double well via tempering",
          "[HPO][anneal_pt][doublewell]")
{
    using namespace tadah::hpo;
    /* f = 10*(x^2-1)^2: wells at +-1, barrier height 10 at x=0. A cold
     * chain (T=0.01) starting in the left well cannot cross (accept
     * prob ~ e^-1000); the tempering ladder must carry right-well
     * states down to the cold end. */
    auto dw = [](const Vec& x){ const double t = x(0)*x(0) - 1.0; return 10.0*t*t; };

    auto count_right_well = [&](bool use_pt) {
        MtStubHost stub{ dw, {}, {} };
        if (use_pt) {
            auto sp = make_pt_spec({-2.0}, {2.0});
            set_seed_point(sp, {-1.0});
            sp.seed = 7;
            sp.num_threads = 2;
            sp.stop.maxeval = 3000;
            sp.param["REPLICAS"] = 8.0;
            sp.param["T_HOT"]    = 10.0;
            sp.param["T_COLD"]   = 0.01;
            sp.param["STEP_FRAC"] = 0.10;
            run_anneal_pt_impl<Vec>(sp, stub);
        } else {
            /* Serial cold control: same stepper, same budget, stuck. */
            tadah::hpo::OptimizerSpec sp;
            sp.low = {-2.0}; sp.high = {2.0};
            sp.is_int.assign(1, false);
            sp.log_scaled.assign(1, false);
            sp.lib = "TADAH"; sp.algo = "ANNEAL";
            set_seed_point(sp, {-1.0});
            sp.seed = 7;
            sp.stop.maxeval = 3000;
            sp.param["CHAIN_LENGTH"] = 10.0;
            sp.param["T_INIT"]    = 0.01;
            sp.param["T_MIN"]     = 1e-5;
            sp.param["STEP_FRAC"] = 0.10;
            sp.param_raw["REANNEAL"]  = "OFF";
            run_anneal_impl<Vec>(sp, stub);
        }
        std::size_t n_right = 0;
        for (const auto& p : stub.sorted_calls())
            if (p[0] > 0.9) ++n_right;
        return n_right;
    };

    CHECK(count_right_well(/*use_pt=*/true) > 0);
    CHECK(count_right_well(/*use_pt=*/false) == 0);
}

TEST_CASE("ANNEAL_PT: MAXEVAL is a hard cap across all replicas",
          "[HPO][anneal_pt]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 3;
    sp.num_threads = 4;
    sp.stop.maxeval = 137;         /* deliberately not a multiple of K */
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"]    = 1.0;

    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, 0.0, 0.0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() == 137);
}

TEST_CASE("ANNEAL_PT automatic steps learn a 12-decade diagonal and emit TSV",
          "[HPO][anneal_pt][step_auto]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec({-1e-6, -1.0, -1e6}, {1e-6, 1.0, 1e6});
    sp.param_raw.erase("STEP_MODE");       // exercise the default-on path
    set_seed_point(sp, {0.0, 0.0, 0.0});
    sp.seed = 91001;
    sp.num_threads = 2;
    sp.stop.maxeval = 1200;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["STEP_FRAC"] = 0.1;

    MtStubHost host{[](const Vec& x) {
        const double a = x(0) / 1e-6;
        const double b = x(1);
        const double c = x(2) / 1e6;
        return a * a + b * b + c * c;
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 6);
    for (const auto& row : rows) {
        CHECK(std::isfinite(row.sigma));
        CHECK(row.sigma > 0.0);
        CHECK(row.fraction > 0.0);
        CHECK(row.acceptance >= 0.0);
        CHECK(row.acceptance <= 1.0);
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
        double min_fraction = std::numeric_limits<double>::infinity();
        double max_fraction = 0.0;
        double min_sigma = std::numeric_limits<double>::infinity();
        double max_sigma = 0.0;
        for (const auto& row : rows) if (row.slot == slot) {
            min_fraction = std::min(min_fraction, row.fraction);
            max_fraction = std::max(max_fraction, row.fraction);
            min_sigma = std::min(min_sigma, row.sigma);
            max_sigma = std::max(max_sigma, row.sigma);
        }
        CHECK(max_fraction / min_fraction < 100.0);
        CHECK(max_sigma / min_sigma > 1e10);
    }
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT[CLASSIC[FROZEN]]") != std::string::npos);
    CHECK(log.find("step calibration slot=1") != std::string::npos);
    CHECK(log.find("production epoch/swap/flow statistics reset")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT warm-up classifies a hypersensitive dim",
          "[HPO][anneal_pt][step_auto][classify]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec({-1e12}, {1e12});
    sp.param_raw.erase("STEP_MODE");
    set_seed_point(sp, {0.0});
    sp.seed = 91007;
    sp.num_threads = 2;
    sp.stop.maxeval = 800;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1e-6;
    sp.param["T_COLD"] = 1e-7;

    /* Slot 0 sits EXACTLY at the |x| cusp (JITTER pins slot 0 to the
     * seed), so acceptance collapses at every scale the 48-attempt
     * budget can reach from STEP_FRAC*width: /10 back-offs cannot cross
     * 12 decades of hypersensitivity. Slot 1 is Gaussian-jittered ~1e11
     * away from the cusp (JITTER now ALWAYS spreads slots >= 1), where
     * downhill proposals accept -- it is genuinely responsive there and
     * must NOT be classified hypersensitive. */
    MtStubHost host{[](const Vec& x) { return 1e9 * std::fabs(x(0)); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const std::string log = lr.read();
    CHECK(log.find("hypersensitive=1") != std::string::npos);
    CHECK(log.find("insensitive=0") != std::string::npos);
    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 2);
    for (const auto& row : rows) {
        if (row.slot == 0) {
            /* The cusp slot cannot find an in-band scale at any tested
             * magnitude: unconverged AND classified hypersensitive. */
            CHECK(row.status.find("unconverged") != std::string::npos);
            CHECK(row.status.find("hypersensitive") != std::string::npos);
        } else {
            /* The jittered slot is genuinely responsive away from the
             * cusp; it must not be classified hypersensitive (whether
             * its bisection fully converges depends on the warm-up
             * stream and is not the property under test). */
            CHECK(row.status.find("hypersensitive") == std::string::npos);
        }
    }
}

TEST_CASE("ANNEAL_PT warm-up accepts a tight bracket as calibrated",
          "[HPO][anneal_pt][step_auto][classify]")
{
    using namespace tadah::hpo;
    /* A 0/1 acceptance cliff defeats the two-consecutive-in-band rule at
     * 2 batches per dim (STEP_WARMUP_SWEEPS 4 -> attempt cap 8), while a
     * miss-low followed by a miss-high builds a two-sided x2 bracket
     * that the calibration accepts as calibrated.  The batch outcomes are 4-sample
     * Bernoulli draws, so a fixed deterministic seed list is scanned and
     * at least one run must certify via the bracket; each individual run
     * remains fully deterministic. */
    bool any_bracket = false;
    std::string bracket_log;
    for (unsigned seed : {91010u, 91011u, 91012u, 91013u, 91014u,
                          91015u, 91016u, 91017u, 91018u, 91019u}) {
        TempWorkingDirectory cwd;
        LoggedRun lr;
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw.erase("STEP_MODE");
        set_seed_point(sp, {0.0});
        sp.seed = seed;
        sp.num_threads = 2;
        sp.stop.maxeval = 600;
        sp.param["REPLICAS"] = 2.0;
            sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.5;
        sp.param["STEP_FRAC"] = 0.5;
        sp.param["STEP_WARMUP_SWEEPS"] = 4.0;

        MtStubHost host{[](const Vec& x) {
            return std::fabs(x(0)) < 0.45 ? 0.0 : 1e9;
        }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, host);

        const auto rows = read_step_scales();
        REQUIRE(rows.size() == 2);
        for (const auto& row : rows) {
            if (row.status.find("bracket-ok") != std::string::npos) {
                any_bracket = true;
                CHECK(row.sigma > 0.0);
            }
        }
        if (any_bracket) { bracket_log = lr.read(); break; }
    }
    REQUIRE(any_bracket);
    CHECK(bracket_log.find("bracket-ok=") != std::string::npos);
    CHECK(bracket_log.find("bracket-ok=0") == std::string::npos);
}

TEST_CASE("ANNEAL_PT trust region banner and sigma journal labels",
          "[HPO][anneal_pt][step_auto][step_adapt][trust]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec({-1.0}, {1.0});
    sp.param_raw["STEP_MODE"] = "ADAPTIVE";  // calibrate, then keep adapting
    set_seed_point(sp, {0.0});
    sp.seed = 91002;
    sp.num_threads = 2;
    sp.stop.maxeval = 1200;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["SWAP_WINDOW"] = 2.0;

    MtStubHost host{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT[CLASSIC[ADAPTIVE]]") != std::string::npos);
    CHECK(log.find("STEP_TRUST_FACTOR=64") != std::string::npos);
    /* The retired cadence knob is gone from the banner: ADAPTIVE applies
     * the RAM after every proposal. */
    CHECK(log.find("STEP_ADAPT_EVERY") == std::string::npos);
    CHECK(log.find("(instantaneous V; sigma[i]/OPTIM_width[i])") !=
          std::string::npos);
    CHECK(log.find("epochs; sigma[i]/OPTIM_width[i])") == std::string::npos);
    CHECK(log.find("ANNEAL_PT sigma dims at trust bound per slot") !=
          std::string::npos);
}

TEST_CASE("ANNEAL_PT calibrates a 174-D objective with 12 decades of "
          "per-dimension scale spread",
          "[HPO][anneal_pt][step_auto][highdim]")
{
    using namespace tadah::hpo;
    constexpr std::size_t D = 174;
    const std::vector<std::size_t> flat_dims{31, 47, 79, 105, 131, 147};
    auto is_flat = [&](std::size_t i) {
        return std::find(flat_dims.begin(), flat_dims.end(), i) !=
               flat_dims.end();
    };

    std::vector<double> low(D), high(D), seed(D, 0.0);
    for (std::size_t i = 0; i < D; ++i) {
        const double exponent = -6.0 +
            12.0 * static_cast<double>(i) / static_cast<double>(D - 1);
        const double half_width = std::pow(10.0, exponent);
        low[i] = -half_width;
        high[i] = half_width;
    }

    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec(low, high);
    sp.param_raw["STEP_MODE"] = "ADAPTIVE";  // production RAM per proposal
    sp.parameter_labels.resize(D);
    for (std::size_t i = 0; i < D; ++i)
        sp.parameter_labels[i] = "p_" + std::to_string(i);
    set_seed_point(sp, seed);
    sp.seed = 92224;
    sp.num_threads = 2;
    /* This test studies the CALIBRATION itself; budget it so the
     * MAXEVAL/4 calibration governor cannot truncate the warm-up
     * (2*STEP_WARMUP_SWEEPS*dim*K + pilot <= 17104 evals). */
    sp.stop.maxeval = 80000;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["STEP_FRAC"] = 1.0e-6;
    sp.param["STEP_WARMUP_SWEEPS"] = 24.0;
    /* Epoch = REPLICAS*1 evals now, so the post-warm-up production phase
     * runs ~1400 epochs; report every 100 to keep the journal bounded
     * while still exercising the windowed sigma blocks. */
    sp.param["SWAP_WINDOW"] = 100.0;

    MtStubHost host{[&, low, high](const Vec& x) {
        long double loss = 0.0L;
        for (std::size_t i = 0; i < D; ++i) {
            if (is_flat(i)) continue;
            const double half_width = 0.5 * (high[i] - low[i]);
            const double sensitivity = (i % 29 == 0) ? 100.0 : 1.0;
            const long double characteristic =
                static_cast<long double>(half_width) * 1.0e-3L / sensitivity;
            const long double q = static_cast<long double>(x(i)) /
                                  characteristic;
            if (i % 37 == 0 && std::fabs(q) > 4.0L)
                return std::numeric_limits<double>::infinity();
            loss += q * q;
        }
        return static_cast<double>(loss);
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 2 * D);
    std::size_t nonflat_validated = 0;
    std::size_t nonflat_total = 0;
    std::size_t flat_capped = 0;
    for (const auto& row : rows) {
        REQUIRE(row.parameter ==
                "p_" + std::to_string(
                    static_cast<std::size_t>(&row - rows.data()) % D));
        CHECK(std::isfinite(row.sigma));
        CHECK(row.sigma > 0.0);
        CHECK(row.attempts >= 24);
        CHECK(row.attempts <= 48);
        CHECK(row.pilot_attempts >= 100);
        CHECK(row.pilot_attempts <= 200);
        const std::size_t dim_index = static_cast<std::size_t>(
            std::stoul(row.parameter.substr(std::string("p_").size())));
        if (is_flat(dim_index)) {
            if (row.status.find("support-capped") != std::string::npos)
                ++flat_capped;
        } else {
            ++nonflat_total;
            if (row.status == "ok" ||
                (row.mean_alpha >= 0.29 && row.mean_alpha <= 0.59))
                ++nonflat_validated;
        }
    }
    CHECK(flat_capped == 2 * flat_dims.size());
    /* Strict status requires two consecutive valid windows. At exhaustion,
     * an unresolved coordinate deliberately uses its best tested window; it
     * is still a calibrated production scale, but remains visibly warned. */
    CHECK(nonflat_validated >=
          static_cast<std::size_t>(0.90 * nonflat_total));
    const std::string log = lr.read();
    CHECK(log.find("step calibration summary") != std::string::npos);
    CHECK(log.find("adaptive sigma/width geo-mean per slot") !=
          std::string::npos);

    /* The per-slot journal line and hpo_step_scales.tsv are two readers of
     * one gathered record. They drifted once: a field was inserted at offset
     * 1, the TSV reader was updated and the journal reader was not, so every
     * a slot line once printed pilot_attempts=0, pilot_multiplier=0,
     * pilot_status=ok while the TSV showed 200 / 13.19 / unconverged.
     * Nothing caught it because the only assertion was that the substring
     * "step calibration slot=0" existed. Pin the numbers instead. */
    /* `slot` stays 0-based here -- it indexes the gathered records and the
     * TSV rows. Only the journal KEY is rendered 1-based. */
    auto slot_line = [&](std::size_t slot) {
        const std::string key =
            "step calibration slot=" + tadah::hpo::logidx::uis(slot) + "  ";
        const auto at = log.find(key);
        REQUIRE(at != std::string::npos);
        return log.substr(at, log.find('\n', at) - at);
    };
    auto field = [](const std::string& line, const char* name) {
        const std::string key = std::string("  ") + name + "=";
        const auto at = line.find(key);
        REQUIRE(at != std::string::npos);
        const auto from = at + key.size();
        auto to = line.find(' ', from);
        if (to == std::string::npos) to = line.size();
        std::string tok = line.substr(from, to - from);
        if (!tok.empty() && tok.back() == '%') tok.pop_back();
        return tok;
    };
    std::size_t unresolved_pilots = 0;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        const StepScaleRow* any = nullptr;
        std::size_t attempts = 0, backoffs = 0, unusable = 0;
        for (const auto& row : rows) if (row.slot == slot) {
            if (!any) any = &row;
            /* Slot-wide pilot fields are replicated on every dim record. */
            CHECK(row.pilot_attempts == any->pilot_attempts);
            CHECK(row.pilot_status == any->pilot_status);
            attempts += row.attempts;
            backoffs += row.backoffs;
            unusable += row.unusable;
        }
        REQUIRE(any != nullptr);
        if (any->pilot_status != "ok") ++unresolved_pilots;
        const std::string line = slot_line(slot);
        CHECK(std::stoul(field(line, "coordinate_attempts")) == attempts);
        CHECK(std::stoul(field(line, "coordinate_backoffs")) == backoffs);
        CHECK(std::stoul(field(line, "coordinate_unusable")) == unusable);
        CHECK(std::stoul(field(line, "pilot_attempts")) == any->pilot_attempts);
        CHECK(std::stoul(field(line, "pilot_backoffs")) == any->pilot_backoffs);
        CHECK(field(line, "pilot_status") == any->pilot_status);
        CHECK(std::stod(field(line, "pilot_multiplier")) ==
              Approx(any->pilot_multiplier).epsilon(1e-3));
        CHECK(std::stod(field(line, "pilot_acc")) ==
              Approx(100.0 * any->pilot_acceptance).margin(0.05));
        CHECK(std::stod(field(line, "pilot_alpha")) ==
              Approx(100.0 * any->pilot_mean_alpha).margin(0.05));
        /* The journal must say which 1.0 this is. */
        CHECK(line.find(any->pilot_multiplier_moved ? "(moved)"
                                                    : "(never moved)")
              != std::string::npos);
    }
    CHECK(log.find("unresolved-pilots=" + std::to_string(unresolved_pilots))
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT automatic scales are robust to expanded OPTIM support",
          "[HPO][anneal_pt][step_auto][support]")
{
    using namespace tadah::hpo;
    auto learned_sigma = [](double half_width) {
        TempWorkingDirectory cwd;
        auto sp = make_pt_spec({-half_width}, {half_width});
        sp.param_raw.erase("STEP_MODE");
        set_seed_point(sp, {0.0});
        sp.seed = 91002;
        sp.num_threads = 2;
        sp.stop.maxeval = 500;
        sp.param["REPLICAS"] = 2.0;
            sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        sp.param["STEP_FRAC"] = 1e-4;
        MtStubHost host{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, host);
        const auto rows = read_step_scales();
        REQUIRE(rows.size() == 2);
        return rows[1].sigma;          // compare at the same hot slot
    };

    const double base = learned_sigma(10.0);
    const double expanded = learned_sigma(1e7);  // width expanded by 1e6
    CHECK(expanded / base > 0.01);
    CHECK(expanded / base < 100.0);
}

TEST_CASE("ANNEAL_PT automatic steps operate in transformed log coordinates",
          "[HPO][anneal_pt][step_auto][logscale]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto sp = make_pt_spec({1e-6}, {1e6});
    sp.log_threshold = 1e3;
    sp.analyse_logscale();
    REQUIRE(sp.log_scaled == std::vector<bool>{true});
    sp.param_raw.erase("STEP_MODE");
    set_seed_point(sp, {1.0});  // public starting points remain physical
    sp.seed = 91006;
    sp.num_threads = 2;
    sp.stop.maxeval = 500;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["STEP_FRAC"] = 0.1;
    MtStubHost host{[](const Vec& physical) {
        return std::pow(std::log(physical(0)), 2.0);
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 2);
    for (const auto& row : rows) {
        CHECK(std::isfinite(row.sigma));
        CHECK(row.sigma > 0.0);
    }
    const auto calls = host.sorted_calls();
    REQUIRE_FALSE(calls.empty());
    for (const auto& call : calls) {
        REQUIRE(call.size() == 1);
        CHECK(std::isfinite(call[0]));
        CHECK(call[0] > 0.0);
    }
}

TEST_CASE("ANNEAL_PT automatic warm-up flags flat support and backs off cliffs",
          "[HPO][anneal_pt][step_auto][cliff]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    sp.param_raw.erase("STEP_MODE");
    set_seed_point(sp, {0.0, 1.0});  // exercise an exact-bound warm seed
    sp.seed = 91005;
    sp.num_threads = 2;
    /* Calibration study: sized so the MAXEVAL/4 governor does not bind
     * (coordinate phase + full pilots <= 500 evals). */
    sp.stop.maxeval = 2400;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["STEP_FRAC"] = 0.5;
    MtStubHost host{[](const Vec& x) {
        if (std::fabs(x(0)) > 0.02)
            return std::numeric_limits<double>::infinity();
        return x(0) * x(0);  // dimension 1 is deliberately flat
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);

    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 4);
    bool flat_capped = false;
    for (const auto& row : rows) {
        /* 1-based: the flat coordinate is x(1), i.e. the SECOND dimension.
         * The fallback label now agrees with hpo_search_space.tsv's `dim`
         * column, which the same run writes 1-based. */
        if (row.parameter == "dim[2]" &&
            row.status.find("support-capped") != std::string::npos)
            flat_capped = true;
    }
    CHECK(flat_capped);
    unsigned floor_status = pt_detail::StepOk;
    const double floor_hi = std::nextafter(1.0, 2.0);
    const double floor_scale = pt_detail::clamp_coordinate_scale(
        0.0, 1.0, 1.0, floor_hi, floor_status);
    CHECK(floor_scale == floor_hi - 1.0);
    CHECK((floor_status & pt_detail::StepNumericalFloor) != 0);
    const std::string log = lr.read();
    const std::size_t first = log.find("backoffs=");
    REQUIRE(first != std::string::npos);
    CHECK(log.find("backoffs=0", first) == std::string::npos);
}

TEST_CASE("ANNEAL_PT coordinate cap retains the best tested fixed scale",
          "[HPO][anneal_pt][step_auto][controller]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    anneal::Knobs kb;
    kb.step_warmup_sweeps = 12;
    anneal::ChainState<Vec> c;
    c.x = Vec(1);
    c.x = 0.0;
    c.f = 0.0;
    c.V = {0.1};
    c.rng.seed(123);
    c.best_x = c.x;
    c.best_f = c.f;
    anneal::SharedBudget budget(100);
    std::atomic<bool> stop{false};
    const std::function<bool()> never = [] { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};
    pt_detail::SlotCalibration calibration;
    std::size_t call = 0;
    auto host = [&](const Vec&) {
        static const double first_eight[] =
            {-1.0, 1.0, -2.0, 1.0, -3.0, 1.0, -4.0, 1.0};
        if (call < 8) return first_eight[call++];
        ++call;
        return std::numeric_limits<double>::infinity();
    };

    pt_detail::coordinate_warmup(
        c, 1.0e-12, sp, kb, host, ctx, 100, calibration);

    REQUIRE(calibration.dims.size() == 1);
    const auto& dc = calibration.dims[0];
    CHECK(dc.attempts == 24);
    CHECK(dc.validation_mean_alpha == Approx(0.5));
    CHECK((dc.status & pt_detail::StepUnconverged) != 0);
    CHECK(c.V[0] == Approx(0.1));
    CHECK(dc.backoffs > 0);
}

TEST_CASE("ANNEAL_PT pilot changes a late failed multiplier by at most two",
          "[HPO][anneal_pt][step_auto][controller]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    anneal::Knobs kb;
    anneal::ChainState<Vec> c;
    c.x = Vec(1);
    c.x = 0.0;
    c.f = 0.0;
    c.V = {0.1};
    c.rng.seed(321);
    c.best_x = c.x;
    c.best_f = c.f;
    anneal::SharedBudget budget(250);
    std::atomic<bool> stop{false};
    const std::function<bool()> never = [] { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};
    pt_detail::SlotCalibration calibration;
    calibration.dims.resize(1);
    std::size_t call = 0;
    std::size_t downhill = 0;
    auto host = [&](const Vec&) {
        const std::size_t window = call / 20;
        const std::size_t in_window = call % 20;
        ++call;
        if (window < 2) return 0.0;  // two high-alpha factor-two windows
        if (window == 3) return 100.0;
        if (in_window < 5) return -static_cast<double>(++downhill);
        return 100.0;
    };

    pt_detail::full_vector_pilot(
        c, 1.0e-12, sp, kb, host, ctx, 250, calibration);

    CHECK(calibration.pilot_proposed == 120);
    CHECK(calibration.pilot_status == pt_detail::StepOk);
    CHECK(calibration.pilot_backoffs == 1);
    CHECK(calibration.pilot_multiplier >= 2.0);
    CHECK(calibration.pilot_multiplier <= 4.0);
    CHECK(c.V[0] == Approx(0.1 * calibration.pilot_multiplier));
}

TEST_CASE("ANNEAL_PT interrupted automatic warm-up consumes exact budget",
          "[HPO][anneal_pt][step_auto][budget]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    LoggedRun lr;
    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    sp.param_raw.erase("STEP_MODE");
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 91003;
    sp.num_threads = 4;
    sp.stop.maxeval = 137;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    MtStubHost host{[](const Vec& x) { return x(0) * x(0) + x(1) * x(1); },
                    {}, {}};
    run_anneal_pt_impl<Vec>(sp, host);
    CHECK(host.sorted_calls().size() == 137);
    const auto rows = read_step_scales();
    REQUIRE(rows.size() == 8);
    CHECK(std::any_of(rows.begin(), rows.end(), [](const StepScaleRow& row) {
        return row.status.find("interrupted") != std::string::npos;
    }));
    CHECK(lr.read().find("MAXEVAL exhausted after 137 host eval(s)")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT automatic adaptive diagonal is thread-count deterministic",
          "[HPO][anneal_pt][step_auto][determinism]")
{
    using namespace tadah::hpo;
    auto run = [](unsigned threads) {
        TempWorkingDirectory cwd;
        auto sp = make_pt_spec({-2.0, -2.0}, {2.0, 2.0});
        sp.param_raw.erase("STEP_MODE");
        set_seed_point(sp, {0.5, -0.5});
        sp.seed = 91004;
        sp.num_threads = threads;
        sp.stop.maxeval = 600;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        sp.param_raw["STEP_MODE"] = "ADAPTIVE";
        sp.param["STEP_WARMUP_SWEEPS"] = 4.0;
        MtStubHost host{[](const Vec& x) {
            return x(0) * x(0) + 10.0 * x(1) * x(1);
        }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, host);
        return host.sorted_calls();
    };
    const auto serial = run(1);
    const auto parallel = run(4);
    REQUIRE(serial.size() == 600);
    REQUIRE(serial.size() == parallel.size());
    for (std::size_t i = 0; i < serial.size(); ++i) {
        REQUIRE(serial[i].size() == parallel[i].size());
        for (std::size_t d = 0; d < serial[i].size(); ++d)
            CHECK(serial[i][d] == Approx(parallel[i][d]).margin(1e-15));
    }
}

TEST_CASE("ANNEAL_PT: MAXEVAL zero performs no host evaluation",
          "[HPO][anneal_pt][budget]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 3;
    sp.num_threads = 2;
    sp.stop.maxeval = 0;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"]    = 1.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().empty());
    CHECK(lr.read().find("after 0 host eval(s)") != std::string::npos);
}

TEST_CASE("ANNEAL_PT: layout line matches the stepper and the planning gates "
          "grade round-trip feasibility", "[HPO][anneal_pt][log][planning]")
{
    using namespace tadah::hpo;
    auto run = [](bool corana, std::size_t maxeval) {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 17;
        sp.num_threads = 1;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 1.0;
        sp.stop.maxeval = maxeval;
        if (corana) {
            sp.param_raw.erase("STEP_MODE");
            sp.param_raw["CORANA"] = "ON";
            sp.param["NS"] = 1.0;
        }

        LoggedRun lr;
        MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, stub);
        return lr.read();
    };

    /* An epoch is ONE SWEEP per replica: CLASSIC = 1 whole-vector
     * proposal, CORANA = dim single-coordinate moves. K=4 -> the
     * structural round-trip floor is 2*(K-1) = 6 exchange rounds and the
     * 25%-heuristic threshold is 24. */
    const std::string classic = run(false, 8);   /* 4 starts + 1 epoch */
    CHECK(classic.find("ANNEAL_PT layout: sweep = 1 proposal eval(s)/replica"
                       ";  epoch = REPLICAS*sweep = 4*1 = 4 evals")
          != std::string::npos);
    CHECK(classic.find("one cold<->hot ladder round trip is impossible")
          != std::string::npos);
    CHECK(classic.find("Raise MAXEVAL or reduce REPLICAS/SWAP_EVERY.")
          != std::string::npos);

    const std::string corana = run(true, 8);
    CHECK(corana.find("ANNEAL_PT layout: sweep = dim = 1 eval(s)/replica")
          != std::string::npos);
    CHECK(corana.find("one cold<->hot ladder round trip is impossible")
          != std::string::npos);

    /* 4 starts + 40 evals -> 10 planned rounds: above the structural
     * floor, below the 25%-heuristic bound -> HINT, not WARNING. */
    const std::string hinted = run(false, 44);
    CHECK(hinted.find("one cold<->hot ladder round trip is impossible")
          == std::string::npos);
    CHECK(hinted.find("< 2*(REPLICAS-1)/0.25 = 24") != std::string::npos);
    CHECK(hinted.find("Consider a larger MAXEVAL or fewer REPLICAS.")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT: T0 probe exhaustion reports MAXEVAL exactly",
          "[HPO][anneal_pt][budget]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.25});
    sp.seed = 31;
    sp.num_threads = 2;
    sp.stop.maxeval = 3;           /* seed + two of the default probes */
    sp.param["REPLICAS"] = 4.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() == 3);
    CHECK(lr.read().find("ANNEAL_PT stopped: MAXEVAL exhausted")
          != std::string::npos);
    /* Pins: the run's status "TADAH:ANNEAL_PT: MAXEVAL reached". Before the
     * fix: no status (the trailer said COMPLETED). */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PT: MAXEVAL reached");
}

TEST_CASE("ANNEAL_PT: extreme finite temperature endpoints build a finite ladder",
          "[HPO][anneal_pt][temperature]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 13;
    sp.num_threads = 1;
    sp.stop.maxeval = 3;
    sp.param["REPLICAS"] = 3.0;
    sp.param["T_HOT"] = std::numeric_limits<double>::max();
    sp.param["T_COLD"] = std::numeric_limits<double>::denorm_min();

    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    LoggedRun lr;
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 3);
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT ladder T per slot") != std::string::npos);
    CHECK(log.find(" inf") == std::string::npos);
}

TEST_CASE("ANNEAL_PT: power-shaped ladders preserve endpoints and numerical gaps",
          "[HPO][anneal_pt][temperature][ladder]")
{
    using namespace tadah::hpo::pt_detail;

    const auto geometric = make_temperature_ladder(1.0, 16.0, 5, 1.0);
    REQUIRE(geometric.size() == 5);
    CHECK(geometric[0] == 1.0);
    CHECK(geometric[1] == Approx(2.0));
    CHECK(geometric[2] == Approx(4.0));
    CHECK(geometric[3] == Approx(8.0));
    CHECK(geometric[4] == 16.0);

    const auto hot_shaped = make_temperature_ladder(1.0, 16.0, 5, 0.5);
    const auto cold_shaped = make_temperature_ladder(1.0, 16.0, 5, 2.0);
    CHECK(hot_shaped.front() == 1.0);
    CHECK(hot_shaped.back() == 16.0);
    CHECK(hot_shaped[1] > geometric[1]);
    CHECK(cold_shaped[1] < geometric[1]);

    const auto extreme = make_temperature_ladder(
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::max(), 32, 0.5);
    CHECK(extreme.front() == std::numeric_limits<double>::denorm_min());
    CHECK(extreme.back() == std::numeric_limits<double>::max());
    for (std::size_t k = 1; k < extreme.size(); ++k) {
        CHECK(std::isfinite(extreme[k]));
        CHECK(extreme[k] > extreme[k - 1]);
    }

    double narrow_hot = 1.0;
    for (int i = 0; i < 4; ++i)
        narrow_hot = std::nextafter(
            narrow_hot, std::numeric_limits<double>::infinity());
    const auto floored = make_temperature_ladder(1.0, narrow_hot, 5, 100.0);
    CHECK(floored.front() == 1.0);
    CHECK(floored.back() == narrow_hot);
    for (std::size_t k = 1; k < floored.size(); ++k)
        CHECK(floored[k] > floored[k - 1]);

    CHECK_THROWS(make_temperature_ladder(
        1.0, std::nextafter(1.0, 2.0), 3, 1.0));
}

TEST_CASE("ANNEAL_PT: synthetic bottleneck contracts while open gaps expand",
          "[HPO][anneal_pt][temperature][ladder]")
{
    using namespace tadah::hpo::pt_detail;
    const auto initial = make_temperature_ladder(1.0, 64.0, 4, 1.0);
    const std::vector<std::size_t> attempts{20, 20, 20};
    const std::vector<std::size_t> accepted{0, 10, 20};
    const auto first = adapt_temperature_ladder(
        initial, attempts, accepted, 1, 1.0, 0.6);

    CHECK(first.temperature.front() == initial.front());
    CHECK(first.temperature.back() == initial.back());
    CHECK(first.old_log_gap[0] == Approx(
        static_cast<double>(first.old_log_gap[1])));
    CHECK(first.new_log_gap[0] < first.old_log_gap[0]);
    CHECK(first.new_log_gap[2] > first.old_log_gap[2]);
    for (std::size_t k = 1; k < first.temperature.size(); ++k)
        CHECK(first.temperature[k] > first.temperature[k - 1]);

    const auto later = adapt_temperature_ladder(
        initial, attempts, accepted, 16, 1.0, 0.6);
    CHECK(later.gain < first.gain);
    CHECK(std::fabs(static_cast<double>(later.new_log_gap[0] -
                                       later.old_log_gap[0])) <
          std::fabs(static_cast<double>(first.new_log_gap[0] -
                                       first.old_log_gap[0])));

    /* A simple barrier model a_p=exp(-b_p*d_p) closes its acceptance
     * spread after one update, not merely its abstract gap spread. */
    const std::vector<double> barrier{4.0, 2.0, 1.0};
    const std::vector<std::size_t> many_attempts(3, 100000);
    std::vector<std::size_t> model_accepted(3, 0);
    std::vector<double> before_rate(3, 0.0);
    for (std::size_t p = 0; p < 3; ++p) {
        before_rate[p] = std::exp(
            -barrier[p] * static_cast<double>(first.old_log_gap[p]));
        model_accepted[p] = static_cast<std::size_t>(
            std::llround(100000.0 * before_rate[p]));
    }
    const auto equalized = adapt_temperature_ladder(
        initial, many_attempts, model_accepted, 1, 1.0, 0.6);
    std::vector<double> after_rate(3, 0.0);
    for (std::size_t p = 0; p < 3; ++p)
        after_rate[p] = std::exp(
            -barrier[p] * static_cast<double>(equalized.new_log_gap[p]));
    const auto before_range = std::minmax_element(
        before_rate.begin(), before_rate.end());
    const auto after_range = std::minmax_element(
        after_rate.begin(), after_rate.end());
    CHECK(*after_range.second - *after_range.first <
          *before_range.second - *before_range.first);
}

TEST_CASE("ANNEAL_PT: ladder floor clamps one-sided contraction",
          "[HPO][anneal_pt][temperature][ladder]")
{
    using namespace tadah::hpo::pt_detail;
    const auto initial = make_temperature_ladder(1.0, 64.0, 4, 1.0);
    const long double span = ladder_log_gap(initial.front(), initial.back());
    const std::vector<std::size_t> attempts{1000, 1000, 1000};

    /* A single update's weight change is clamped to a factor of 2, so a
     * wide floor (0.8) is needed for one update to reach it. */
    const double wide = 0.8;
    const long double wide_gap =
        static_cast<long double>(wide) * span / 3.0L;
    const auto floored = adapt_temperature_ladder(
        initial, attempts, {0, 1000, 1000}, 1, 1.0, 0.6, wide);
    REQUIRE(floored.at_floor.size() == 3);
    CHECK(floored.at_floor[0] == 1);
    CHECK(floored.at_floor[1] == 0);
    CHECK(floored.at_floor[2] == 0);
    CHECK(floored.temperature.front() == initial.front());
    CHECK(floored.temperature.back() == initial.back());
    long double total = 0.0L;
    for (std::size_t p = 0; p < 3; ++p) {
        CHECK(static_cast<double>(floored.new_log_gap[p]) >=
              static_cast<double>(wide_gap) * (1.0 - 1e-9));
        total += floored.new_log_gap[p];
    }
    CHECK(static_cast<double>(total) ==
          Approx(static_cast<double>(span)).epsilon(1e-9));

    /* Regression: re-feeding the same one-sided counters keeps
     * contracting the starved pair, but never below the floor, however
     * many updates run -- and the pair ends up flagged at the floor. */
    const double frac = 0.25;
    const long double floor_gap =
        static_cast<long double>(frac) * span / 3.0L;
    const std::vector<std::size_t> accepted{0, 500, 1000};
    auto ladder = initial;
    std::vector<char> last_at_floor;
    for (std::size_t n = 1; n <= 50; ++n) {
        const auto step = adapt_temperature_ladder(
            ladder, attempts, accepted, n, 1.0, 0.6, frac);
        ladder = step.temperature;
        last_at_floor = step.at_floor;
        CHECK(ladder.front() == initial.front());
        CHECK(ladder.back() == initial.back());
        for (std::size_t k = 1; k < ladder.size(); ++k)
            CHECK(ladder[k] > ladder[k - 1]);
        for (std::size_t p = 0; p < 3; ++p)
            CHECK(static_cast<double>(step.new_log_gap[p]) >=
                  static_cast<double>(floor_gap) * (1.0 - 1e-9));
    }
    REQUIRE(last_at_floor.size() == 3);
    CHECK(last_at_floor[0] == 1);

    /* Two starved pairs: both clamp, the third absorbs the rest. */
    const auto two = adapt_temperature_ladder(
        initial, attempts, {0, 0, 1000}, 1, 1.0, 0.6, wide);
    CHECK(two.at_floor[0] == 1);
    CHECK(two.at_floor[1] == 1);
    CHECK(two.at_floor[2] == 0);
    CHECK(static_cast<double>(two.new_log_gap[2]) ==
          Approx(static_cast<double>(span - 2.0L * wide_gap)).epsilon(1e-9));

    /* Legacy default: no floor argument means no clamps and the historic
     * renormalization. */
    const auto legacy = adapt_temperature_ladder(
        initial, attempts, accepted, 1, 1.0, 0.6);
    for (std::size_t p = 0; p < 3; ++p) CHECK(legacy.at_floor[p] == 0);

    CHECK_THROWS_AS(adapt_temperature_ladder(
        initial, attempts, accepted, 1, 1.0, 0.6, 1.0),
        std::invalid_argument);
    CHECK_THROWS_AS(adapt_temperature_ladder(
        initial, attempts, accepted, 1, 1.0, 0.6, -0.1),
        std::invalid_argument);
}

TEST_CASE("ANNEAL_PT: equal-loss plateau does not latch the ladder freeze",
          "[HPO][anneal_pt][temperature][ladder][log]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 20260817;
    sp.num_threads = 2;
    sp.stop.maxeval = 60;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param_raw["LADDER_ADAPT"] = "ON";
    sp.param["LADDER_ADAPT_EVERY"] = 1.0;

    LoggedRun lr;
    /* Constant objective: every exchange accepts through the equal-loss
     * branch. Degenerate exchanges carry no information about the pair
     * gaps, so they are EXCLUDED from adaptation evidence entirely: the
     * ladder must neither update nor freeze, adaptation stays armed
     * (mode=ongoing) for when real signal appears, the DEGENERATE%
     * table and its dominance WARNING call the situation out, and the
     * raw swap table still shows the (correct) 100% acceptance. */
    MtStubHost stub{[](const Vec&) { return 0.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ladder adaptation frozen: per-pair acceptance") ==
          std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder update=") == std::string::npos);
    CHECK(log.find("mode=ongoing") != std::string::npos);
    CHECK(log.find("DEGENERATE% (equal-loss") != std::string::npos);
    CHECK(log.find("were DEGENERATE (equal losses on both sides")
          != std::string::npos);

    /* No updates -> the ladder TSV holds the header only. */
    std::ifstream input("hpo_temperature_ladder.tsv");
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string tsv = contents.str();
    CHECK(tsv.find("\tat_floor\n") != std::string::npos);
    CHECK(tsv.rfind("\tat_floor\n") + std::string("\tat_floor\n").size()
          == tsv.size());
}

TEST_CASE("ANNEAL_PT: pooled freeze evidence holds, accumulates and resets",
          "[HPO][anneal_pt][temperature][ladder]")
{
    using namespace tadah::hpo::pt_detail;
    const std::vector<char> nofloor{0, 0, 0};
    const std::vector<std::size_t> att{20, 20, 20};

    /* Zero-information windows (all-accept / all-reject): held forever,
     * evidence untouched. */
    LadderFreezeEvidence hold;
    for (int i = 0; i < 10; ++i) {
        CHECK_FALSE(ladder_freeze_step(hold, att, {20, 20, 20}, nofloor, 100));
        CHECK_FALSE(ladder_freeze_step(hold, att, {0, 0, 0}, nofloor, 100));
    }
    REQUIRE(hold.attempt.size() == 3);
    CHECK(hold.attempt[0] == 0);

    /* Genuinely equalized noisy windows: evidence pools and the freeze
     * fires exactly when every pair reaches 100 pooled attempts. */
    LadderFreezeEvidence eq;
    const std::vector<std::vector<std::size_t>> acc_eq{
        {5, 4, 5}, {4, 5, 4}, {5, 5, 4}, {4, 4, 5}, {5, 5, 5}};
    for (std::size_t w = 0; w < acc_eq.size(); ++w) {
        const bool frozen =
            ladder_freeze_step(eq, att, acc_eq[w], nofloor, 100);
        CHECK(frozen == (w + 1 == acc_eq.size()));
    }

    /* A genuinely unequal ladder (15% vs 35%) must never freeze: pooling
     * raises the test's power until the disparity is resolved and the
     * evidence resets. */
    LadderFreezeEvidence uneq;
    for (int w = 0; w < 20; ++w)
        CHECK_FALSE(ladder_freeze_step(uneq, att, {3, 7, 3}, nofloor, 100));

    /* A floor-limited below-mean pair counts as settled. */
    LadderFreezeEvidence fl;
    const std::vector<char> floor0{1, 0, 0};
    bool frozen = false;
    for (int w = 0; w < 5; ++w)
        frozen = ladder_freeze_step(fl, att, {3, 5, 5}, floor0, 100);
    CHECK(frozen);
}

TEST_CASE("ANNEAL_PT: removed LADDER_ADAPT_UPDATES is a hard error; the "
          "equalization freeze governs the adaptation lifetime",
          "[HPO][anneal_pt][temperature][ladder][validation]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.stop.maxeval = 20;
    sp.param_raw["LADDER_ADAPT"] = "ON";
    sp.param["LADDER_ADAPT_UPDATES"] = 1.0;

    MtStubHost stub{[](const Vec&) { return 0.0; }, {}, {}};
    REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                        Catch::Contains("LADDER_ADAPT_UPDATES was removed"));
}

TEST_CASE("ANNEAL_PT: ladder adaptation stays live and writes canonical TSV",
          "[HPO][anneal_pt][temperature][ladder][log]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 20260810;
    sp.num_threads = 2;
    sp.stop.maxeval = 20;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param["LADDER_POWER"] = 0.5;
    sp.param_raw["LADDER_ADAPT"] = "ON";
    sp.param["LADDER_ADAPT_EVERY"] = 1.0;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 20);
    const std::string log = lr.read();
    CHECK(log.find("LADDER_POWER=0.5") != std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder update=1") != std::string::npos);
    /* No update-count cap exists any more: with only one pooled update the
     * equalization freeze cannot have fired, so adaptation is still live. */
    CHECK(log.find("ladder adaptation frozen") == std::string::npos);
    CHECK(log.find("ANNEAL_PT final ladder: updates=1") != std::string::npos);
    CHECK(log.find("mode=ongoing") != std::string::npos);

    std::ifstream input("hpo_temperature_ladder.tsv");
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string tsv = contents.str();
    CHECK(tsv.find("update\tepoch\tpair\tattempts") == 0);
    CHECK(static_cast<std::size_t>(std::count(tsv.begin(), tsv.end(), '\n')) == 4);
}

TEST_CASE("ANNEAL_PT: interrupted ladder window is retained without an update",
          "[HPO][anneal_pt][temperature][ladder][budget]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 20260811;
    sp.num_threads = 2;
    sp.stop.maxeval = 20;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param_raw["LADDER_ADAPT"] = "ON";
    sp.param["LADDER_ADAPT_EVERY"] = 5.0;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 20);
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT ladder update=") == std::string::npos);
    CHECK(log.find("ANNEAL_PT final ladder: updates=0") != std::string::npos);
    CHECK(log.find("pending_attempts/pair=1-2") != std::string::npos);
    std::ifstream input("hpo_temperature_ladder.tsv");
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string tsv = contents.str();
    CHECK(static_cast<std::size_t>(std::count(tsv.begin(), tsv.end(), '\n')) == 1);
}

TEST_CASE("ANNEAL_PT: exchange decision is stable at zero and extreme states",
          "[HPO][anneal_pt][temperature][swap]")
{
    using tadah::hpo::pt_detail::accept_exchange;

    int draws = 0;
    auto coin = [&](double value) {
        return [&, value]() {
            ++draws;
            return value;
        };
    };

    /* Equal beta or energy has probability one and consumes no coin. */
    CHECK(accept_exchange(1.0, 1.0, -4.0, 9.0, coin(0.99)));
    CHECK(accept_exchange(0.0, 0.0, -4.0, 9.0, coin(0.99)));
    CHECK(accept_exchange(0.0, 1.0, 3.0, 3.0, coin(0.99)));
    CHECK(draws == 0);

    /* At zero cold temperature, only a better hot state moves coldward
     * (apart from the equal-energy case above). */
    CHECK(accept_exchange(0.0, 1.0, 2.0, 1.0, coin(0.99)));
    CHECK_FALSE(accept_exchange(0.0, 1.0, 1.0, 2.0, coin(0.0)));
    CHECK(draws == 0);

    /* Normalized infinity states are ordered without inf-inf or inf*0. */
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(accept_exchange(0.25, 1.0, inf, 1.0, coin(0.99)));
    CHECK_FALSE(accept_exchange(0.25, 1.0, 1.0, inf, coin(0.0)));
    CHECK(accept_exchange(0.25, 1.0, inf, inf, coin(0.99)));
    CHECK(draws == 0);

    /* Moderate finite inputs agree with
     * exp((1/T_cold-1/T_hot)*(f_cold-f_hot)). */
    const double probability = std::exp(-0.5);
    CHECK(accept_exchange(1.0, 2.0, 0.0, 1.0,
                          coin(0.5 * probability)));
    CHECK_FALSE(accept_exchange(1.0, 2.0, 0.0, 1.0,
                                coin(0.5 * (1.0 + probability))));
    CHECK(draws == 2);

    /* Cross-check a finite, positive-temperature grid against the direct
     * replica-exchange formula away from indeterminate limit cases. */
    const double cold_grid[] = {0.125, 1.0, 3.0};
    const double ratio_grid[] = {1.25, 2.0, 10.0};
    const double loss_grid[] = {-100.0, -1.0, 0.0, 3.0};
    const double coin_grid[] = {0.01, 0.25, 0.75, 0.99};
    for (double t_cold : cold_grid) {
        for (double ratio : ratio_grid) {
            const double t_hot = t_cold * ratio;
            for (double f_cold : loss_grid) {
                for (double f_hot : loss_grid) {
                    for (double u : coin_grid) {
                        const long double delta =
                            (1.0L / static_cast<long double>(t_cold) -
                             1.0L / static_cast<long double>(t_hot)) *
                            (static_cast<long double>(f_cold) -
                             static_cast<long double>(f_hot));
                        const bool expected = delta >= 0.0L ||
                            static_cast<long double>(u) < std::exp(delta);
                        CHECK(accept_exchange(t_cold, t_hot, f_cold, f_hot,
                                              [u]() { return u; }) == expected);
                    }
                }
            }
        }
    }

    /* A positive subnormal cold endpoint remains a well-defined finite-
     * temperature limit rather than producing NaN. */
    CHECK_FALSE(accept_exchange(std::numeric_limits<double>::denorm_min(),
                                1.0, 0.0, 1.0, coin(0.5)));
    CHECK(draws == 3);

    CHECK_THROWS_AS(accept_exchange(-1.0, 1.0, 0.0, 1.0, coin(0.5)),
                    std::invalid_argument);
    CHECK_THROWS_AS(accept_exchange(2.0, 1.0, 0.0, 1.0, coin(0.5)),
                    std::invalid_argument);
    CHECK_THROWS_AS(accept_exchange(1.0, 2.0,
                                    std::numeric_limits<double>::quiet_NaN(),
                                    1.0, coin(0.5)),
                    std::invalid_argument);
    CHECK_THROWS_AS(accept_exchange(1.0, 2.0, 0.0, 1.0, coin(1.0)),
                    std::invalid_argument);
}

TEST_CASE("ANNEAL_PT: equal-energy exchange survives a subnormal cold slot",
          "[HPO][anneal_pt][temperature][swap]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 20260809;
    sp.num_threads = 1;
    sp.stop.maxeval = 8;  /* 2 starts + three 2-eval epochs */
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = std::numeric_limits<double>::denorm_min();
    sp.param["SWAP_WINDOW"] = 2.0;

    LoggedRun lr;
    /* Constant losses: every attempted exchange accepts through the
     * equal-energy branch even at the subnormal cold temperature. */
    MtStubHost stub{[](const Vec&) { return 7.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* Banner check: [FIXED] is the fixture STEP_MODE. */
    CHECK(log.find("ANNEAL_PT[CLASSIC[FIXED]]") != std::string::npos);
    /* K=2 attempts its only pair on even epoch counts; epoch 2 is a
     * SWAP_WINDOW report epoch and must show the accepted swap. */
    const std::size_t epoch2 = log.find("ANNEAL_PT epoch=2/");
    REQUIRE(epoch2 != std::string::npos);
    const std::size_t line_end = log.find("\n", epoch2);
    const std::string line = log.substr(epoch2, line_end - epoch2);
    CHECK(line.find("T=[4.941e-324, 1.000e+00]") != std::string::npos);
    CHECK(line.find("swap_acc=100.0%") != std::string::npos);
    CHECK(log.find("ANNEAL_PT swap acceptance per pair (cold->hot):  1:100%")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT: STOPVAL terminates the ladder",
          "[HPO][anneal_pt]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.9});
    sp.seed = 5;
    sp.num_threads = 2;
    sp.stop.maxeval = 5000;
    sp.stop.stopval = 1e-3;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"]    = 0.5;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() < 5000);
    CHECK(lr.read().find("ANNEAL_PT stopped: STOPVAL reached")
          != std::string::npos);
    /* Pins: the run's status "TADAH:ANNEAL_PT: STOPVAL reached". Before the
     * fix: no status. */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PT: STOPVAL reached");
}

TEST_CASE("ANNEAL_PT: cooling-schedule keys are rejected -- the ladder is static",
          "[HPO][anneal_pt][validation][temperature]")
{
    using namespace tadah::hpo;
    MtStubHost stub{ [&](const Vec&){ return 0.0; }, {}, {} };

    SECTION("RT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.5});
        sp.stop.maxeval = 10;
        sp.param["RT"] = 0.5;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("static"));
    }
    SECTION("T_MIN") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.5});
        sp.stop.maxeval = 10;
        sp.param["T_MIN"] = 1e-2;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("static"));
    }
    SECTION("NT and CHAIN_LENGTH name the sweep-per-epoch structure") {
        auto nt = make_pt_spec({-1.0}, {1.0});
        nt.param_raw["CORANA"] = "ON";
        nt.param["NT"] = 2.0;
        nt.stop.maxeval = 10;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(nt, stub),
                            Catch::Contains("NT is not used by ANNEAL_PT"));
        auto cl = make_pt_spec({-1.0}, {1.0});
        cl.param["CHAIN_LENGTH"] = 5.0;
        cl.stop.maxeval = 10;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(cl, stub),
                            Catch::Contains("one sweep per epoch"));
    }
}

TEST_CASE("ANNEAL_PT: adaptive interiors preserve the fixed ladder endpoints",
          "[HPO][anneal_pt][temperature][ladder]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory cwd;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 6;
    sp.num_threads = 2;
    sp.stop.maxeval = 40;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param_raw["LADDER_ADAPT"] = "ON";
    sp.param["LADDER_ADAPT_EVERY"] = 1.0;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* Adaptation ran, and the endpoints in the final ladder table are the
     * configured T_COLD/T_HOT exactly (only interiors moved). */
    CHECK(log.find("ANNEAL_PT ladder update=") != std::string::npos);
    const std::size_t final_at = log.find("ANNEAL_PT final ladder: updates=");
    REQUIRE(final_at != std::string::npos);
    const std::string tail = log.substr(final_at);
    CHECK(tail.find("1.000e+00") != std::string::npos);
    CHECK(tail.find("8.000e+00") != std::string::npos);
}

TEST_CASE("ANNEAL_PT: journal carries the banner, epoch lines and the "
          "per-pair swap table", "[HPO][anneal_pt][log]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 11;
    sp.num_threads = 2;
    sp.stop.maxeval = 400;
    sp.param["REPLICAS"]    = 4.0;
    sp.param_raw.erase("STEP_MODE");
    sp.param_raw["CORANA"]      = "ON";
    sp.param["NS"]          = 2.0;
    sp.param["T_HOT"]       = 1.0;
    sp.param["SWAP_WINDOW"] = 4.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, 0.0, 0.0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT[CORANA]") != std::string::npos);
    /* Journal reports fire every SWAP_WINDOW epochs now, not per epoch. */
    CHECK(log.find("ANNEAL_PT epoch=4/") != std::string::npos);
    CHECK(log.find("ANNEAL_PT epoch=1/") == std::string::npos);
    CHECK(log.find("swap acceptance per pair") != std::string::npos);
    CHECK(log.find("ANNEAL_PT stopped:") != std::string::npos);
    /* Resolved-ladder report: summary line + per-slot table (fires even
     * with a user-set T_HOT, which skips the T0 calibration). The
     * new-best histogram is NOT expected here: the stub host never calls
     * log_new_best (that lives in HPO_Host) -- the origin-tag plumbing
     * has its own HPOLog test case below. */
    CHECK(log.find("ANNEAL_PT ladder: T_HOT=") != std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder T per slot (1=coldest .. 4=hottest):")
          != std::string::npos);
    CHECK(log.find("[1-4]") != std::string::npos);
    /* Per-epoch replica-state report (world==1 path). */
    CHECK(log.find("ANNEAL_PT f per slot (pre-swap):") != std::string::npos);
    CHECK(log.find("ANNEAL_PT df per slot (epoch):") != std::string::npos);
    CHECK(log.find("ANNEAL_PT best_f per slot:") != std::string::npos);
    CHECK(log.find("ANNEAL_PT imp per slot") != std::string::npos);
    CHECK(log.find("ANNEAL_PT acc% per slot (epoch):") != std::string::npos);
    /* Flow reporting: exchange-events row, tagged post-swap walker
     * lineup, per-epoch traversal counters. */
    CHECK(log.find("ANNEAL_PT exchange this epoch") != std::string::npos);
    CHECK(log.find("ANNEAL_PT walker per slot (post-swap")
          != std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder flow: traversals cold->hot=")
          != std::string::npos);
    /* Corana V-state blocks + knobs in the banner. */
    CHECK(log.find("ANNEAL_PT V geo-mean per slot") != std::string::npos);
    CHECK(log.find("ANNEAL_PT V min per slot (min_i V[i]/width[i]):")
          != std::string::npos);
    CHECK(log.find("ANNEAL_PT V max per slot (max_i V[i]/width[i]):")
          != std::string::npos);
    CHECK(log.find("V_FLOOR_FRAC=") != std::string::npos);
    CHECK(log.find("NS=2 (sweeps per V adjustment)") != std::string::npos);
    CHECK(log.find("CORANA_C=") != std::string::npos);
    CHECK(log.find("STEP_FRAC=0.1 (Corana V half-width/width)")
          != std::string::npos);
    /* Retired banner tokens must be gone entirely. */
    CHECK(log.find("STEP_AUTO") == std::string::npos);
    CHECK(log.find(" RT=") == std::string::npos);
    CHECK(log.find("LADDER_ADAPT_UPDATES") == std::string::npos);
    CHECK(log.find("BOUNDARY_MODE=REFLECT") != std::string::npos);
    CHECK(log.find("INIT_MODE=JITTER") != std::string::npos);
    CHECK(log.find("V_floor=") != std::string::npos);
    CHECK(log.find("V_cap=") != std::string::npos);
    /* Live per-pair swap tables: cumulative each exchange epoch, windowed
     * every SWAP_WINDOW epochs. */
    CHECK(log.find("ANNEAL_PT swap acc% per pair (cumulative")
          != std::string::npos);
    CHECK(log.find("ANNEAL_PT swap acc% per pair (last 4 epoch(s)")
          != std::string::npos);
    /* Multi-line entries: starred head line, tagged continuation rows. */
    CHECK(log.find("INFO      * ANNEAL_PT ladder T per slot")
          != std::string::npos);
}

/* --- semantic coverage for the ladder-flow traversal counter -------------
 * The pre-existing journal test above only asserts that the ladder-flow
 * STRING appears. That would stay green if the counter tagged slots instead
 * of walkers, or if the world==1 guard were dropped, or if the counts were
 * simply wrong -- and these counts are the headline PT-effectiveness
 * metric, so they need pinning to arithmetic, not to a substring. */

namespace {
/* Pull the integer immediately following the LAST occurrence of `key`.
 * Returns -1 if `key` is absent. */
long last_int_after(const std::string& log, const std::string& key)
{
    const std::size_t at = log.rfind(key);
    if (at == std::string::npos) return -1;
    return std::strtol(log.c_str() + at + key.size(), nullptr, 10);
}

/* Pull the integer immediately PRECEDING the LAST occurrence of `key`
 * (the stop line reads "... (E epoch(s), R exchange round(s))", so the
 * count sits to the left of its label). Returns -1 if absent. */
long last_int_before(const std::string& log, const std::string& key)
{
    const std::size_t at = log.rfind(key);
    if (at == std::string::npos) return -1;
    std::size_t e = at;
    while (e > 0 && std::isspace(static_cast<unsigned char>(log[e - 1]))) --e;
    std::size_t b = e;
    while (b > 0 && std::isdigit(static_cast<unsigned char>(log[b - 1]))) --b;
    if (b == e) return -1;
    return std::strtol(log.substr(b, e - b).c_str(), nullptr, 10);
}
}  // namespace

TEST_CASE("ANNEAL_PT: ladder flow counts walker lineages with exact arithmetic",
          "[HPO][anneal_pt][log][flow]")
{
    using namespace tadah::hpo;

    /* K=2 makes the traversal arithmetic exactly predictable. The only
     * exchange pair is (0,1), and slot 0 is the cold END while slot 1 is the
     * hot END -- so every ACCEPTED swap moves one walker cold->hot and the
     * other hot->cold, incrementing both counters by exactly 1.
     *
     * A constant objective forces every attempted exchange to accept:
     * accept_exchange short-circuits to true when cold_loss == hot_loss
     * (algorithms_pt.h, the equal-loss/equal-temperature branch), with no
     * coin drawn.
     *
     * DEO parity is `(epoch_count / SWAP_EVERY) % 2` and epoch_count is
     * incremented BEFORE the exchange phase, so the first exchange round
     * runs at epoch_count==1 -> parity 1, which at K=2 attempts no pair at
     * all (the loop starts at i=parity and needs i+1 < K). Pairs are
     * therefore attempted on EVEN epoch_count only, giving
     *     flow_up == flow_down == rounds / 2   (floor).
     * That identity is what distinguishes walker tagging from slot tagging:
     * a slot-indexed counter cannot grow at all here, because slots 0 and 1
     * are occupied every single round by construction. */
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4242;
    sp.num_threads = 1;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_COLD"] = 1e-3;
    sp.param["T_HOT"]  = 1.0;
    sp.param["SWAP_EVERY"] = 1.0;
    /* 2 seed/init evals + E epochs x REPLICAS x 1 (CLASSIC sweep = one
     * proposal per replica). 40 epochs -> 19 accepted swaps, comfortably
     * above the 1 round trip a working ladder needs and well short of the
     * 100*(K-1)=100-round transport window, which is exercised separately
     * below. */
    sp.stop.maxeval = 2 + 40 * 2;

    LoggedRun lr;
    MtStubHost stub{[](const Vec&) { return 1.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);
    const std::string log = lr.read();

    const long rounds = last_int_before(log, "exchange round(s))");
    const long up   = last_int_after(log, "traversals cold->hot=");
    const long down = last_int_after(log, "  hot->cold=");

    REQUIRE(up   >= 0);
    REQUIRE(down >= 0);

    /* Symmetric at K=2: a walker leaving the cold end arrives at the hot end
     * in the same swap. */
    CHECK(up == down);
    /* Non-trivial: the ladder actually moved lineages. */
    CHECK(up > 0);

    /* The exact identity. `rounds` is read from the stop line
     * "... (E epoch(s), R exchange round(s))". */
    REQUIRE(rounds > 0);
    CHECK(up == rounds / 2);

    /* Whole-run mixing profile: with K=2 and every swap accepted, both slots
     * are visited by both walkers. */
    CHECK(log.find("ANNEAL_PT distinct walkers per slot over the whole run "
                   "(2 = fully mixed):") != std::string::npos);
    /* ...so the single-walker severance WARNING must NOT fire. */
    CHECK(log.find("was occupied by a single walker")
          == std::string::npos);
}

TEST_CASE("ANNEAL_PT: a cooperative shutdown still reports the whole run",
          "[HPO][anneal_pt][log][flow][shutdown]")
{
    using namespace tadah::hpo;

    /* SIGINT/SIGTERM/SIGUSR2 surface as core::ShutdownRequested thrown by
     * HPO_Host::operator() at the next evaluation. That used to unwind
     * straight out of run_anneal_pt_impl, past every end-of-run table --
     * so a PT run stopped by the wall clock (the normal fate of a
     * MAXEVAL sized above the SLURM slot) produced no whole-run mixing
     * profile, which for a transport study is the entire deliverable.
     *
     * The driver now reports first and re-throws after. Both halves are
     * asserted: the tables must be in the journal AND the exception must
     * still reach the caller, because hpo_run() writes the SHUTDOWN
     * trailer and sets the exit status from it. */
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4244;
    sp.num_threads = 1;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_COLD"] = 1e-3;
    sp.param["T_HOT"]  = 1.0;
    sp.param["SWAP_EVERY"] = 1.0;
    /* Budget far beyond the interrupt, so the ONLY way out is the throw. */
    sp.stop.maxeval = 1000000;

    LoggedRun lr;
    std::atomic<int> calls{0};
    MtStubHost stub{[&calls](const Vec&) -> double {
        /* Enough epochs for a traversal to be possible before the stop. */
        if (++calls > 200) throw tadah::core::ShutdownRequested{};
        return 1.0;
    }, {}, {}};

    REQUIRE_THROWS_AS(run_anneal_pt_impl<Vec>(sp, stub),
                      tadah::core::ShutdownRequested);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT stopped: external stop requested "
                   "(SIGINT/SIGTERM/SIGUSR2)") != std::string::npos);
    /* A signal stop records no status of its own: the run driver writes
     * the SHUTDOWN trailer. Before the fix: passes as well (a pin of the
     * signal path, which the per-reason statuses leave alone). */
    CHECK(HPOLog::instance().status().empty());
    CHECK(log.find("ANNEAL_PT distinct walkers per slot over the whole run "
                   "(2 = fully mixed):") != std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder flow: traversals cold->hot=")
          != std::string::npos);
    /* The counters describe the run that actually happened, not a blank. */
    CHECK(last_int_before(log, "exchange round(s))") > 0);
}

TEST_CASE("ANNEAL_PT: a severed ladder is reported, not just tabulated",
          "[HPO][anneal_pt][log][flow]")
{
    using namespace tadah::hpo;

    /* Same K=2 geometry, but now exchanges must never be accepted, so no
     * lineage ever reaches the opposite extreme. Two ingredients:
     *   - slot 0 starts EXACTLY at the loss minimum (JITTER leaves slot 0
     *     at the seed) and its cold chain can only stay there, while the
     *     jittered hot slot is strictly above 0 -- so cold_loss < hot_loss
     *     always and the equal-loss short circuit cannot fire;
     *   - T_COLD far below the loss scale, so the Metropolis factor for the
     *     uphill exchange direction underflows.
     * The replicas then anneal independently -- which is exactly the
     * condition can persist for many hours of wall clock without any message saying so. */
    auto sp = make_pt_spec({-1.0e4, -1.0e4}, {1.0e4, 1.0e4});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 99;
    sp.num_threads = 1;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_COLD"] = 1e-300;
    sp.param["T_HOT"]  = 1e-297;
    sp.param["SWAP_EVERY"] = 1.0;
    sp.param["STEP_FRAC"] = 1e-6;   /* stay put: keep the two losses apart */
    /* transport_window = 100*(K-1) = 100 exchange rounds; run past it
     * (CLASSIC epoch = REPLICAS * 1 evals). */
    sp.stop.maxeval = 2 + 130 * 2;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0) + x(1) * x(1); },
                    {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);
    const std::string log = lr.read();

    /* The windowed report exists and names its own window length. */
    CHECK(log.find("ANNEAL_PT ladder transport (last 100 exchange round(s)):")
          != std::string::npos);
    /* Zero transport over a measured window is a structural failure of the
     * method, so it must be a WARNING -- not a HINT, and not merely a table
     * the reader is expected to notice. */
    CHECK(log.find("WARNING") != std::string::npos);
    CHECK(log.find("no ladder traversal in the last 100 exchange round(s)")
          != std::string::npos);
    /* The message must say why the per-pair acceptance table cannot be used
     * as the diagnostic instead -- mistaking one for the other wastes long runs. */
    CHECK(log.find("adaptive ladder equalises acceptance") != std::string::npos);
    /* Cumulative counters agree with the window. */
    CHECK(log.find("traversals cold->hot=0  hot->cold=0") != std::string::npos);
    CHECK(log.find("0 round trips -- the ladder never communicated end to end")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT: final summary repeats the last completed swap window",
          "[HPO][anneal_pt][log][swap]")
{
    using namespace tadah::hpo;
    auto run = [](std::size_t maxeval) {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 91020;
        sp.num_threads = 2;
        sp.stop.maxeval = maxeval;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 8.0;
        sp.param["T_COLD"] = 1.0;
        sp.param["SWAP_WINDOW"] = 4.0;
        LoggedRun lr;
        MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, stub);
        return lr.read();
    };

    /* Enough epochs for two completed windows: the final table names the
     * last one and its end epoch. */
    const std::string log = run(40);
    CHECK(log.find("swap acc% per pair (last completed SWAP_WINDOW=4 "
                   "window, ended epoch 8") != std::string::npos);

    /* Too short for any completed window: the accumulated counters are
     * printed with an honest coverage label. */
    const std::string short_log = run(16);
    CHECK(short_log.find("last completed SWAP_WINDOW") == std::string::npos);
    CHECK(short_log.find(
              "swap acc% per pair (no completed SWAP_WINDOW; covers "
              "epochs 1-") != std::string::npos);
}

TEST_CASE("ANNEAL_PT: SWAP_WINDOW OFF suppresses window tables entirely",
          "[HPO][anneal_pt][log][swap]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 91022;
    sp.num_threads = 2;
    sp.stop.maxeval = 40;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param_raw["SWAP_WINDOW"] = "OFF";

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    /* Cumulative-only mode: no PERIODIC report fires at all (reports are
     * tied to SWAP_WINDOW boundaries now), no windowed table may appear
     * anywhere, and the final summary must not relabel whole-run counters
     * as a window -- only the whole-run per-pair summary remains. */
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT swap acceptance per pair (cold->hot):")
          != std::string::npos);
    CHECK(log.find("last completed SWAP_WINDOW") == std::string::npos);
    CHECK(log.find("no completed SWAP_WINDOW") == std::string::npos);
    CHECK(log.find("swap acc% per pair (last ") == std::string::npos);
}

TEST_CASE("ANNEAL_PT: epoch line reports the ladder endpoints of the epoch",
          "[HPO][anneal_pt][log][temperature]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 91021;
    sp.num_threads = 2;
    sp.stop.maxeval = 60;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 8.0;
    sp.param["T_COLD"] = 1.0;
    sp.param["SWAP_WINDOW"] = 2.0;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    /* The epoch line snapshots T BEFORE any in-epoch ladder mutation;
     * with a static ladder it is exactly [T_COLD, T_HOT] on every report. */
    const std::string log = lr.read();
    CHECK(log.find("T=[1.000e+00, 8.000e+00]") != std::string::npos);
}

TEST_CASE("HPOLog: best-origin tag (slot/T + probe) and per-slot histograms",
          "[HPO][anneal_pt][log]")
{
    using tadah::hpo::HPOLog;
    LoggedRun lr;
    auto& log = HPOLog::instance();

    /* A best recorded while the thread is stamped as a T0 probe; the
     * breakdown continuation line must carry the timestamp + BEST tag. */
    HPOLog::set_eval_probe();
    log.log_new_best(1, 10.0, 0,false, 0,false, 0,false, 0,false, 0,false,
                     1.0, "A=1(50%)");
    /* A better best recorded from ladder slot 3 of 32 at T=2.377. */
    HPOLog::set_eval_slot(3, 32, 2.377);
    log.log_new_best(2, 5.0, 0,false, 0,false, 0,false, 0,false, 0,false,
                     1.0);
    /* Not-a-best: must neither tag nor count. */
    log.log_new_best(3, 7.0, 0,false, 0,false, 0,false, 0,false, 0,false,
                     1.0);
    HPOLog::clear_eval_context();

    const std::string txt = lr.read();
    CHECK(txt.find("[T0 probe]") != std::string::npos);
    /* Stamped 0-based (slot 3), PRINTED 1-based (slot 4/32). The map-key
     * assertions below deliberately stay at 3: HPOLog::best_slot_* are keyed
     * by the raw slot, and shifting the stamp instead of the rendering makes
     * the whole-run "new bests per slot" tables silently print all zeros. */
    CHECK(txt.find("[slot 4/32 T=2.377e+00]") != std::string::npos);
    /* Continuation line: blank timestamp column (12 spaces) + tag. */
    CHECK(txt.find("\n            BEST       top: A=1(50%)")
          != std::string::npos);
    /* The old redundant prefix is gone. */
    CHECK(txt.find("NEW BEST") == std::string::npos);

    auto win = log.take_best_slot_window();
    REQUIRE(win.size() == 1);
    CHECK(win[0].first == 3);
    CHECK(win[0].second == 1);
    CHECK(log.take_best_slot_window().empty());   /* window drained */
    auto tot = log.best_slot_totals();
    REQUIRE(tot.size() == 1);
    CHECK(tot[0].first == 3);
    CHECK(tot[0].second == 1);
    /* The slot-3 best above landed at eval 2; the rejected
     * non-best at eval 3 must not have moved it. */
    auto last = log.best_slot_last_evals();
    REQUIRE(last.size() == 1);
    CHECK(last[0].first == 3);
    CHECK(last[0].second == 2);
    log.reset_for_test();
    CHECK(log.best_slot_last_evals().empty());
}

TEST_CASE("ANNEAL_PT: table formatters render grids and dashes",
          "[HPO][anneal_pt][log]")
{
    using namespace tadah::hpo::pt_detail;
    /* Slot-count grid: rows of 8, zero-fill for missing slots. */
    const std::string t = format_slot_count_table(
        "HDR:", 10, {{0, 17177}, {6, 157}, {9, 3}});
    CHECK(t.find("HDR:") == 0);
    /* 1-based labels: 10 slots print as 1..10, so the row labels need two
     * columns and are right-aligned. Cell spacing is unchanged. */
    CHECK(t.find("[ 1- 8]") != std::string::npos);
    CHECK(t.find("[ 9-10]") != std::string::npos);
    CHECK(t.find("17177") != std::string::npos);
    CHECK(t.find("157") != std::string::npos);
    /* the second slot is zero-filled */
    CHECK(t.find("      17177          0") != std::string::npos);

    /* Pair table: percentages, att/pair range, '-' for unattempted. */
    const std::string p = format_pair_table(
        4, "label", {10, 0, 20}, {5, 0, 10});
    CHECK(p.find("ANNEAL_PT swap acc% per pair (label, att/pair 0-20;") == 0);
    CHECK(p.find("       50") != std::string::npos);
    CHECK(p.find("        -") != std::string::npos);
}

TEST_CASE("ANNEAL_PT: validation errors", "[HPO][anneal_pt][validation]")
{
    using namespace tadah::hpo;
    MtStubHost stub{ [&](const Vec&){ return 0.0; }, {}, {} };

    SECTION("STEP_MODE defaults to FROZEN for CLASSIC PT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw.erase("STEP_MODE");
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK(kb.step_mode == anneal_enums::STEP_MODE::FROZEN);
        CHECK(kb.step_warmup_sweeps == 12);
        CHECK(kb.step_trust_factor == 0.0);    /* trust off outside ADAPTIVE */
    }
    SECTION("retired STEP_AUTO steers to STEP_MODE") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw.erase("STEP_MODE");
        sp.param["STEP_AUTO"] = 1.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("replaced by PARAM STEP_MODE"));
    }
    SECTION("STEP_WARMUP_SWEEPS must be positive when a warm-up runs") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw["STEP_MODE"] = "FROZEN";
        sp.param["STEP_WARMUP_SWEEPS"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("STEP_WARMUP_SWEEPS must be >= 1"));
    }
    SECTION("STEP_WARMUP_SWEEPS is unused under STEP_MODE FIXED") {
        auto sp = make_pt_spec({-1.0}, {1.0});   /* fixture FIXED */
        sp.param["STEP_WARMUP_SWEEPS"] = 8.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("unused when STEP_MODE is FIXED"));
    }
    SECTION("CLASSIC step controls conflict with CORANA") {
        for (const char* key : {"STEP_WARMUP_SWEEPS", "STEP_TRUST_FACTOR"}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw.erase("STEP_MODE");
            sp.param_raw["CORANA"] = "ON";
            sp.param[key] = 8.0;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("CLASSIC kernel"));
        }
        auto mode = make_pt_spec({-1.0}, {1.0});
        mode.param_raw["CORANA"] = "ON";   /* fixture sets STEP_MODE FIXED */
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(mode, stub),
                            Catch::Contains("CLASSIC kernel"));
    }
    SECTION("top-level STEP is rejected outright") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw.erase("STEP_MODE");
        sp.step = 0.1;
        REQUIRE_THROWS_WITH(
            run_anneal_pt_impl<Vec>(sp, stub),
            Catch::Contains("does not use STEP; set PARAM STEP_FRAC"));
    }
    SECTION("retired STEP_ADAPT_EVERY is removed under every STEP_MODE") {
        /* The rename gate fires before any cross-key rule, so the removal
         * message wins even where ADAPTIVE would once have accepted it. */
        for (const char* mode : {"FIXED", "FROZEN", "ADAPTIVE"}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["STEP_MODE"] = mode;
            sp.param["STEP_ADAPT_EVERY"] = 8.0;
            REQUIRE_THROWS_WITH(
                run_anneal_pt_impl<Vec>(sp, stub),
                Catch::Contains("STEP_ADAPT_EVERY was removed"));
        }
    }
    SECTION("trust factor is ADAPTIVE-only") {
        for (const char* mode : {"FIXED", "FROZEN"}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["STEP_MODE"] = mode;
            sp.param["STEP_TRUST_FACTOR"] = 8.0;
            REQUIRE_THROWS_WITH(
                run_anneal_pt_impl<Vec>(sp, stub),
                Catch::Contains("unused unless STEP_MODE is ADAPTIVE"));
        }
    }
    SECTION("STEP_TRUST_FACTOR must be > 1, or the token OFF") {
        /* 1.5.0: 0 is no longer the "off" spelling, so it joins the
         * rejected values instead of silently disabling the clamp. */
        for (double f : {1.0, 0.5, -3.0, 0.0}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["STEP_MODE"] = "ADAPTIVE";
            sp.param["STEP_TRUST_FACTOR"] = f;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("STEP_TRUST_FACTOR"));
        }
        /* The positive case is asserted at the PARSE level on purpose:
         * every other case here throws during validation and returns
         * immediately, while run_anneal_pt_impl on an accepted spec would
         * run an unbounded optimisation (make_pt_spec sets no MAXEVAL). */
        auto off = make_pt_spec({-1.0}, {1.0});
        off.param_raw["STEP_MODE"] = "ADAPTIVE";
        off.param_raw["STEP_TRUST_FACTOR"] = "OFF";
        CHECK_NOTHROW(anneal::Knobs::parse(off, anneal::AlgoKind::PT));
    }
    SECTION("LADDER_POWER is independently usable") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["LADDER_POWER"] = 0.5;
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK(kb.ladder_power == 0.5);
        CHECK_FALSE(kb.ladder_adapt);
    }
    SECTION("LADDER_POWER must be positive") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["LADDER_POWER"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("LADDER_POWER"));
    }
    SECTION("LADDER_ADAPT takes ON/OFF, not a number") {
        /* 1.5.0: numeric booleans are gone. The error must carry the
         * migration hint, because silently accepting 1 would make the old
         * and new spellings indistinguishable in a config. */
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["LADDER_ADAPT"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("LADDER_ADAPT takes ON or OFF"));

        auto one = make_pt_spec({-1.0}, {1.0});
        one.param["LADDER_ADAPT"] = 1.0;
        REQUIRE_THROWS_WITH(
            run_anneal_pt_impl<Vec>(one, stub),
            Catch::Contains("0/1 booleans were removed in 1.5.0")
                && Catch::Contains("PARAM LADDER_ADAPT ON"));

        auto word = make_pt_spec({-1.0}, {1.0});
        word.param_raw["LADDER_ADAPT"] = "yes";
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(word, stub),
                            Catch::Contains("LADDER_ADAPT takes ON or OFF"));
    }
    SECTION("adaptation controls are rejected when disabled") {
        for (const char* key : {"LADDER_ADAPT_EVERY", "LADDER_ADAPT_GAIN",
                                "LADDER_ADAPT_DECAY",
                                "LADDER_MIN_GAP", "LADDER_FREEZE"}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            /* LADDER_FREEZE is a boolean and so takes ON/OFF (1.5.0); the
             * rest are numeric. Either way the cross-key rule must fire. */
            if (std::string(key) == "LADDER_FREEZE") sp.param_raw[key] = "ON";
            else                                     sp.param[key] = 1.0;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("unused when LADDER_ADAPT is OFF"));
        }
    }
    SECTION("adaptation requires at least three resolved replicas") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["REPLICAS"] = 2.0;
        sp.param_raw["LADDER_ADAPT"] = "ON";
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("REPLICAS >= 3"));
    }
    SECTION("LADDER_ADAPT_EVERY must be positive") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw["LADDER_ADAPT"] = "ON";
        sp.param["LADDER_ADAPT_EVERY"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("LADDER_ADAPT_EVERY"));
    }
    SECTION("LADDER_ADAPT_GAIN must be in range") {
        for (double gain : {0.0, 1.01}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["LADDER_ADAPT"] = "ON";
            sp.param["LADDER_ADAPT_GAIN"] = gain;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("LADDER_ADAPT_GAIN"));
        }
    }
    SECTION("LADDER_ADAPT_DECAY must be in range") {
        for (double decay : {0.5, 1.01}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["LADDER_ADAPT"] = "ON";
            sp.param["LADDER_ADAPT_DECAY"] = decay;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("LADDER_ADAPT_DECAY"));
        }
    }
    SECTION("removed LADDER_ADAPT_UPDATES names the freeze that replaced it") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param_raw["LADDER_ADAPT"] = "ON";
        sp.param["LADDER_ADAPT_UPDATES"] = 1.5;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("LADDER_FREEZE"));
    }
    SECTION("LADDER_MIN_GAP must be in [0,1)") {
        for (double frac : {-0.1, 1.0, 2.0}) {
            auto sp = make_pt_spec({-1.0}, {1.0});
            sp.param_raw["LADDER_ADAPT"] = "ON";
            sp.param["LADDER_MIN_GAP"] = frac;
            REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                                Catch::Contains("LADDER_MIN_GAP"));
        }
    }

    SECTION("REPLICAS < 2") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["REPLICAS"] = 1.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("REPLICAS"));
    }
    SECTION("SWAP_EVERY 0") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["SWAP_EVERY"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("SWAP_EVERY"));
    }
    SECTION("numeric INIT_MODE names the tokens") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["INIT_MODE"] = 2.0;
        REQUIRE_THROWS_WITH(
            run_anneal_pt_impl<Vec>(sp, stub),
            Catch::Contains("INIT_MODE takes a token, one of: JITTER UNIFORM"));
    }
    SECTION("unknown INIT_MODE token is reported verbatim") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param_raw["INIT_MODE"] = "SCATTER";
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("got 'SCATTER'"));
    }
    SECTION("serial-only REANNEAL rejected on PT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param_raw["REANNEAL"] = "ON";
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("belongs to the serial"));
    }
    SECTION("POPULATION rejected on PT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.pop = 32;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("ANNEAL_PA"));
    }
}

TEST_CASE("ANNEAL_PT: converges on a quadratic bowl at least as well as "
          "one serial chain on the same budget", "[HPO][anneal_pt]")
{
    using namespace tadah::hpo;
    const double cx = 0.3, cy = -0.4;
    auto best_of = [&](const std::vector<std::vector<double>>& calls) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& p : calls) {
            Vec x(2); x(0) = p[0]; x(1) = p[1];
            const double f = quad_bowl2(x, cx, cy);
            if (f < best) best = f;
        }
        return best;
    };

    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {-0.8, 0.8});
    sp.seed = 99;
    sp.num_threads = 4;
    sp.stop.maxeval = 1200;
    sp.param["REPLICAS"]  = 6.0;
    sp.param_raw.erase("STEP_MODE");
    sp.param_raw["CORANA"]    = "ON";
    sp.param["NS"]        = 3.0;
    sp.param["T_HOT"]     = 2.0;
    sp.param["LADDER_SPAN"] = 1e3;

    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, cx, cy); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(best_of(stub.sorted_calls()) < 0.05);
}

/* --- FTOL best-stall stop (the PT replacement for the serial plateau) ----
 * With FTOL_REL/FTOL_ABS > 0 the run stops when the GLOBAL best has not
 * improved by more than max(FTOL_ABS, FTOL_REL*|best|) over NEPS
 * consecutive diagnostics windows (window = SWAP_WINDOW epochs;
 * SWAP_WINDOW 0 falls back to 20 for the stall detector). */

TEST_CASE("ANNEAL_PT: FTOL best-stall stops a stalled global best",
          "[HPO][anneal_pt][plateau][stall]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 91030;
    sp.num_threads = 1;
    sp.stop.maxeval = 10000;
    sp.stop.ftol_abs = 1e-6;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["SWAP_WINDOW"] = 5.0;
    sp.param["NEPS"] = 2.0;

    LoggedRun lr;
    /* Constant objective: the global best never improves after the seed.
     * Window 1 seeds the reference, windows 2 and 3 count the stall ->
     * stop at epoch 15. Evals: 2 starts + 15 epochs x 2 replicas x 1. */
    MtStubHost stub{[](const Vec&) { return 1.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 32);
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT stopped: FTOL best-stall (global best improved "
                   "by <= tol over NEPS diagnostics windows)")
          != std::string::npos);
    /* Pins: the best-stall is the run's status "TADAH:ANNEAL_PT: FTOL
     * plateau". Before the fix: no status. */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PT: FTOL plateau");
}

TEST_CASE("ANNEAL_PT: best-stall does not fire while the best keeps improving",
          "[HPO][anneal_pt][plateau][stall]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 91031;
    sp.num_threads = 1;
    sp.stop.maxeval = 100;
    sp.stop.ftol_abs = 1e-6;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["SWAP_WINDOW"] = 5.0;
    sp.param["NEPS"] = 2.0;

    LoggedRun lr;
    /* Every eval improves the global best by 1.0 >> ftol_abs. */
    std::atomic<int> k{0};
    MtStubHost stub{[&k](const Vec&) { return 1000.0 - (++k); }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 100);
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT stopped: MAXEVAL exhausted") != std::string::npos);
    CHECK(log.find("FTOL best-stall") == std::string::npos);
}

TEST_CASE("ANNEAL_PT: SWAP_WINDOW OFF falls back to a 20-epoch stall window",
          "[HPO][anneal_pt][plateau][stall][swap]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 91032;
    sp.num_threads = 1;
    sp.stop.maxeval = 10000;
    sp.stop.ftol_abs = 1e-6;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param_raw["SWAP_WINDOW"] = "OFF";
    sp.param["NEPS"] = 2.0;

    LoggedRun lr;
    MtStubHost stub{[](const Vec&) { return 1.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    /* Stall windows at epochs 20/40/60 -> stop at epoch 60:
     * 2 starts + 60 epochs x 2 replicas x 1 = 122 evals. */
    CHECK(stub.sorted_calls().size() == 122);
    CHECK(lr.read().find("FTOL best-stall") != std::string::npos);
}

/* ---- Stage 11: per-dimension Corana acceptance census ----------------
 *
 * The user-facing question was "does Corana track acceptance PER DIMENSION,
 * or one number per temperature rung?". The adaptation always did -- what a
 * reader could see was one rung-wide accept% per epoch. These two cases pin
 * the driver half of the fix: that the numbers reaching the journal and the
 * TSV really are resolved per dimension (not the rung scalar written into
 * every row), that they reconcile with the run's own proposal count, and
 * that a CLASSIC run does not emit a Corana artefact.
 */
namespace {

struct CoranaAcceptanceRow {
    std::size_t slot = 0;
    double temperature = 0.0;
    std::string parameter;
    double acceptance = 0.0;
    std::size_t accepted = 0;
    std::size_t proposed = 0;
    double sigma = 0.0;
    double fraction = 0.0;
    double window_acceptance = 0.0;
    std::size_t window_accepted = 0;
    std::size_t window_proposed = 0;
};

std::vector<CoranaAcceptanceRow> read_corana_acceptance(std::string& header)
{
    std::ifstream input("hpo_corana_acceptance.tsv");
    std::getline(input, header);
    std::vector<CoranaAcceptanceRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), '\t', ' ');
        std::istringstream fields(line);
        CoranaAcceptanceRow row;
        fields >> row.slot >> row.temperature >> row.parameter
               >> row.acceptance >> row.accepted >> row.proposed
               >> row.sigma >> row.fraction
               >> row.window_acceptance >> row.window_accepted
               >> row.window_proposed;
        if (fields) {
            /* 1-based on the way out (1.5.0), 0-based for indexing here. */
            row.slot = tadah::hpo::logidx::from_ui(
                row.slot, "hpo_corana_acceptance.tsv slot");
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

}  // namespace

TEST_CASE("ANNEAL_PT: Corana acceptance is reported per dimension",
          "[HPO][anneal_pt][corana][census]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory tmp;

    /* Two dimensions of deliberately different stiffness. x1 is soft, x2 is
     * 10^4 times stiffer, so at any temperature a step of the same size on
     * x2 costs far more and is accepted far less often. If the driver were
     * writing the RUNG-wide acceptance into every row -- the defect this
     * whole stage is about -- the two columns would be equal and the strict
     * inequality below would fail. */
    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 5150;
    sp.num_threads = 1;
    sp.param_raw.erase("STEP_MODE");
    sp.param_raw["CORANA"] = "ON";
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["NS"] = 4.0;
    sp.param["SWAP_WINDOW"] = 5.0;
    sp.stop.maxeval = 1200;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) {
        return x(0) * x(0) + 1.0e4 * x(1) * x(1);
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);
    const std::string log = lr.read();

    std::string header;
    const auto rows = read_corana_acceptance(header);
    REQUIRE(rows.size() == 4u * 2u);          /* REPLICAS x dim */
    CHECK(header.find("window_proposed") != std::string::npos);

    /* Radix: the file is 1-based, and from_ui above already refused
     * anything below 1. Check the whole slot range is covered exactly. */
    std::vector<std::size_t> per_slot(4, 0);
    std::size_t proposed_total = 0, accepted_total = 0;
    for (const auto& r : rows) {
        REQUIRE(r.slot < 4);
        ++per_slot[r.slot];
        proposed_total += r.proposed;
        accepted_total += r.accepted;
        /* The rate column is the ratio of the two count columns next to it,
         * so a reader never has to guess which window it refers to. */
        const double expect = r.proposed > 0
            ? static_cast<double>(r.accepted) / static_cast<double>(r.proposed)
            : 0.0;
        CHECK(r.acceptance == Approx(expect));
        CHECK(r.window_accepted <= r.accepted);
        CHECK(r.window_proposed <= r.proposed);
    }
    for (std::size_t k = 0; k < 4; ++k) CHECK(per_slot[k] == 2u);

    /* Reconciliation: every Corana proposal the run made is in the table.
     * Without this the census could quietly cover one slot, one window or
     * one rank and still look plausible. */
    std::size_t summary_proposed = 0;
    {
        const auto at = log.find("ANNEAL_PT summary: ");
        REQUIRE(at != std::string::npos);
        const auto key = log.find("proposed=", at);
        REQUIRE(key != std::string::npos);
        summary_proposed =
            static_cast<std::size_t>(std::stoull(log.substr(key + 9)));
    }
    CHECK(proposed_total == summary_proposed);
    CHECK(accepted_total > 0);

    /* THE per-dimension statement: the stiff coordinate is accepted strictly
     * less often than the soft one, at every slot. */
    for (std::size_t k = 0; k < 4; ++k) {
        double soft = -1.0, stiff = -1.0;
        for (const auto& r : rows) {
            if (r.slot != k) continue;
            if (r.parameter.find("[1]") != std::string::npos) soft = r.acceptance;
            if (r.parameter.find("[2]") != std::string::npos) stiff = r.acceptance;
        }
        INFO("slot " << k << " soft=" << soft << " stiff=" << stiff);
        REQUIRE(soft >= 0.0);
        REQUIRE(stiff >= 0.0);
        CHECK(stiff < soft);
    }

    /* And the journal says what the numbers are and are not. */
    CHECK(log.find("Corana per-dimension acceptance census") != std::string::npos);
    CHECK(log.find("NOT per walker") != std::string::npos);
    CHECK(log.find("hpo_corana_acceptance.tsv") != std::string::npos);
    CHECK(log.find("Corana median per-dim acceptance % per slot")
          != std::string::npos);
    /* Slot labels in the census table are 1-based like everything else. */
    CHECK(log.find("acceptance extremes, coldest slot 1 ") != std::string::npos);
    CHECK(log.find("acceptance extremes, hottest slot 4 ") != std::string::npos);
    /* Two dimensions cannot have five extremes at either end, and the
     * mirrored-pair rendering that claimed they did must not come back. */
    CHECK(log.find("lowest 5 =") == std::string::npos);
    CHECK(log.find("all 2 visited dimension(s), lowest first:")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT: a CLASSIC run writes no Corana acceptance table",
          "[HPO][anneal_pt][corana][census]")
{
    using namespace tadah::hpo;
    TempWorkingDirectory tmp;

    auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 5151;
    sp.num_threads = 1;
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    sp.stop.maxeval = 200;

    LoggedRun lr;
    MtStubHost stub{[](const Vec& x) {
        return x(0) * x(0) + 1.0e4 * x(1) * x(1);
    }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(!std::filesystem::exists("hpo_corana_acceptance.tsv"));
    CHECK(lr.read().find("Corana per-dimension acceptance census")
          == std::string::npos);
}

/* Seed/T0 calibration used to re-throw whatever the host threw as a
 * plain runtime_error(what), so SIGINT during calibration ended the run
 * with an error (non-zero exit) instead of SHUTDOWN, and a fatal
 * best-potential persistence failure lost its type. Eval 1 is the seed,
 * eval 2 the first T0 probe (T_HOT unset). */
TEST_CASE("ANNEAL_PT: calibration rethrows the host's exception unchanged",
          "[HPO][anneal_pt][failure_handling]")
{
    using namespace tadah::hpo;
    for (std::size_t throw_at : {std::size_t{1}, std::size_t{2}}) {
        CAPTURE(throw_at);
        auto make = [&] {
            auto sp = make_pt_spec({-1.0, -1.0}, {1.0, 1.0});
            set_seed_point(sp, {0.0, 0.0});
            sp.seed = 5;
            sp.num_threads = 2;
            sp.stop.maxeval = 200;
            sp.param["REPLICAS"] = 2.0;
            return sp;
        };
        {
            auto sp = make();
            std::atomic<std::size_t> n{0};
            MtStubHost stub{[&](const Vec& x) {
                if (++n == throw_at) throw tadah::core::ShutdownRequested{};
                return x(0) * x(0) + x(1) * x(1);
            }, {}, {}};
            CHECK_THROWS_AS(run_anneal_pt_impl<Vec>(sp, stub),
                            tadah::core::ShutdownRequested);
        }
        {
            auto sp = make();
            std::atomic<std::size_t> n{0};
            MtStubHost stub{[&](const Vec& x) {
                if (++n == throw_at)
                    throw durable_best::Error("best_pot.tadah: gave up");
                return x(0) * x(0) + x(1) * x(1);
            }, {}, {}};
            try {
                run_anneal_pt_impl<Vec>(sp, stub);
                FAIL("run_anneal_pt_impl returned normally");
            } catch (const durable_best::Error& e) {
                CHECK(std::string(e.what()) == "best_pot.tadah: gave up");
            }
        }
    }
}

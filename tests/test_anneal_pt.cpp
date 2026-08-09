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

#include <tadah/hpo/algorithms.h>      /* serial control in the double-well test */
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <algorithm>
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
    /* Most pre-STEP_AUTO tests exercise the legacy trajectory explicitly.
     * Automatic-mode cases erase this fixture override. */
    sp.param["STEP_AUTO"] = 0.0;
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
    double acceptance = 0.0;
    std::string status;
    double mean_alpha = 0.0;
    std::size_t attempts = 0;
    std::size_t backoffs = 0;
    std::size_t unusable = 0;
    double pilot_multiplier = 1.0;
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
               >> row.sigma >> row.fraction >> row.acceptance >> row.status
               >> row.mean_alpha >> row.attempts >> row.backoffs
               >> row.unusable >> row.pilot_multiplier
               >> row.pilot_acceptance >> row.pilot_mean_alpha
               >> row.pilot_attempts >> row.pilot_backoffs
               >> row.pilot_unusable >> row.pilot_status;
        if (fields) rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

TEST_CASE("ANNEAL_PT: INIT_MODE 0 uses Gaussian seed jitter for both steppers",
          "[HPO][anneal_pt][init][kernel]")
{
    using namespace tadah::hpo;
    auto run = [](bool corana) {
        auto sp = make_pt_spec({-10.0}, {10.0});
        set_seed_point(sp, {0.0});
        sp.seed = 314159;
        sp.num_threads = 2;
        sp.stop.maxeval = 4;
        sp.step = 0.25;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        if (corana) {
            sp.param.erase("STEP_AUTO");
            sp.param["CORANA"] = 1.0;
            sp.param["NS"] = 1.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.param["L"] = 1.0;
        }
        MtStubHost host{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, host);
        return host.sorted_calls();
    };

    for (bool corana : {false, true}) {
        CAPTURE(corana);
        std::vector<std::vector<double>> expected{{0.0}};
        for (std::size_t k = 1; k < 4; ++k) {
            auto rng = anneal::make_stream(314159, anneal::SALT_CHAIN, k);
            const double unit =
                std::normal_distribution<double>(0.0, 1.0)(rng);
            expected.push_back({0.25 * unit});
        }
        std::sort(expected.begin(), expected.end());
        const auto actual = run(corana);
        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            CHECK(actual[i][0] == Approx(expected[i][0]).margin(1e-15));
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
        sp.param.erase("STEP_AUTO");
        sp.param["CORANA"]    = 1.0;
        sp.param["NS"]        = 2.0;
        sp.param["NT"]        = 2.0;
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
            sp.param["L"]        = 10.0;
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
            sp.param["L"]         = 10.0;
            sp.param["T_HOT"]     = 0.01;
            sp.param["T_MIN"]     = 1e-5;
            sp.param["STEP_FRAC"] = 0.10;
            sp.param["REANNEAL"]  = 0.0;
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
    sp.param["L"]        = 10.0;
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
    sp.param.erase("STEP_AUTO");       // exercise the default-on path
    set_seed_point(sp, {0.0, 0.0, 0.0});
    sp.seed = 91001;
    sp.num_threads = 2;
    sp.stop.maxeval = 1200;
    sp.param["REPLICAS"] = 2.0;
    sp.param["L"] = 10.0;
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
    CHECK(log.find("ANNEAL_PT[CLASSIC[AUTO-FROZEN]]") != std::string::npos);
    CHECK(log.find("step calibration slot=0") != std::string::npos);
    CHECK(log.find("production epoch/swap/flow statistics reset")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT calibrates a run22-shaped 174-D objective",
          "[HPO][anneal_pt][step_auto][run22]")
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
    sp.param.erase("STEP_AUTO");
    sp.parameter_labels.resize(D);
    for (std::size_t i = 0; i < D; ++i)
        sp.parameter_labels[i] = "run22_param_" + std::to_string(i);
    set_seed_point(sp, seed);
    sp.seed = 92224;
    sp.num_threads = 2;
    sp.stop.maxeval = 20000;
    sp.param["REPLICAS"] = 2.0;
    sp.param["L"] = 20.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    sp.param["STEP_FRAC"] = 1.0e-6;
    sp.param["STEP_AUTO_SWEEPS"] = 24.0;
    sp.param["STEP_ADAPT_EVERY"] = 20.0;
    sp.param["SWAP_WINDOW"] = 2.0;

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
                "run22_param_" + std::to_string(
                    static_cast<std::size_t>(&row - rows.data()) % D));
        CHECK(std::isfinite(row.sigma));
        CHECK(row.sigma > 0.0);
        CHECK(row.attempts >= 24);
        CHECK(row.attempts <= 48);
        CHECK(row.pilot_attempts >= 100);
        CHECK(row.pilot_attempts <= 200);
        const std::size_t dim_index = static_cast<std::size_t>(
            std::stoul(row.parameter.substr(std::string("run22_param_").size())));
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
}

TEST_CASE("ANNEAL_PT automatic scales are robust to expanded OPTIM support",
          "[HPO][anneal_pt][step_auto][support]")
{
    using namespace tadah::hpo;
    auto learned_sigma = [](double half_width) {
        TempWorkingDirectory cwd;
        auto sp = make_pt_spec({-half_width}, {half_width});
        sp.param.erase("STEP_AUTO");
        set_seed_point(sp, {0.0});
        sp.seed = 91002;
        sp.num_threads = 2;
        sp.stop.maxeval = 500;
        sp.param["REPLICAS"] = 2.0;
        sp.param["L"] = 5.0;
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
    sp.param.erase("STEP_AUTO");
    set_seed_point(sp, {1.0});  // public starting points remain physical
    sp.seed = 91006;
    sp.num_threads = 2;
    sp.stop.maxeval = 500;
    sp.param["REPLICAS"] = 2.0;
    sp.param["L"] = 5.0;
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
    sp.param.erase("STEP_AUTO");
    set_seed_point(sp, {0.0, 1.0});  // exercise an exact-bound warm seed
    sp.seed = 91005;
    sp.num_threads = 2;
    sp.stop.maxeval = 600;
    sp.param["REPLICAS"] = 2.0;
    sp.param["L"] = 5.0;
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
        if (row.parameter == "dim[1]" &&
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
    kb.step_auto_sweeps = 12;
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
    sp.param.erase("STEP_AUTO");
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 91003;
    sp.num_threads = 4;
    sp.stop.maxeval = 137;
    sp.param["REPLICAS"] = 4.0;
    sp.param["L"] = 10.0;
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
        sp.param.erase("STEP_AUTO");
        set_seed_point(sp, {0.5, -0.5});
        sp.seed = 91004;
        sp.num_threads = threads;
        sp.stop.maxeval = 600;
        sp.param["REPLICAS"] = 4.0;
        sp.param["L"] = 5.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        sp.param["STEP_AUTO_SWEEPS"] = 4.0;
        sp.param["STEP_ADAPT_EVERY"] = 5.0;
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
    sp.param["L"]        = 2.0;
    sp.param["T_HOT"]    = 1.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().empty());
    CHECK(lr.read().find("after 0 host eval(s)") != std::string::npos);
}

TEST_CASE("ANNEAL_PT: round-trip advice matches the selected stepper variant",
          "[HPO][anneal_pt][log][planning]")
{
    using namespace tadah::hpo;
    auto run = [](bool corana) {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 17;
        sp.num_threads = 1;
        sp.param["REPLICAS"] = 4.0;
        sp.param["T_HOT"] = 1.0;
        if (corana) {
            sp.param.erase("STEP_AUTO");
            sp.stop.maxeval = 8;   /* 4 starts + one 4-eval epoch */
            sp.param["CORANA"] = 1.0;
            sp.param["NS"] = 1.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.stop.maxeval = 44;  /* 4 starts + one 40-eval epoch */
            sp.param["L"] = 10.0;
        }

        LoggedRun lr;
        MtStubHost stub{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pt_impl<Vec>(sp, stub);
        return lr.read();
    };

    const std::string classic = run(false);
    CHECK(classic.find("Reduce L (epoch = REPLICAS*L evals)")
          != std::string::npos);
    CHECK(classic.find("Reduce NS or NT") == std::string::npos);

    const std::string corana = run(true);
    CHECK(corana.find("Reduce NS or NT (epoch = REPLICAS*NT*NS*dim evals)")
          != std::string::npos);
    CHECK(corana.find("Reduce L (epoch") == std::string::npos);
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
    sp.param["L"]        = 2.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() == 3);
    CHECK(lr.read().find("ANNEAL_PT stopped: MAXEVAL exhausted")
          != std::string::npos);
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
    sp.param["L"] = 1.0;

    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    LoggedRun lr;
    run_anneal_pt_impl<Vec>(sp, stub);

    CHECK(stub.sorted_calls().size() == 3);
    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT ladder T per slot") != std::string::npos);
    CHECK(log.find(" inf") == std::string::npos);
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

TEST_CASE("ANNEAL_PT: equal-energy exchange survives cold-temperature underflow",
          "[HPO][anneal_pt][temperature][swap]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 20260809;
    sp.num_threads = 1;
    sp.stop.maxeval = 8;  /* 2 starts + three 2-eval epochs */
    sp.param["REPLICAS"] = 2.0;
    sp.param["L"] = 1.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = std::numeric_limits<double>::denorm_min();
    sp.param["RT"] = 0.5;

    LoggedRun lr;
    MtStubHost stub{[](const Vec&) { return 7.0; }, {}, {}};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    const std::size_t epoch2 = log.find("ANNEAL_PT epoch=2/");
    REQUIRE(epoch2 != std::string::npos);
    const std::size_t line_end = log.find("\n", epoch2);
    const std::string line = log.substr(epoch2, line_end - epoch2);
    CHECK(line.find("T=[0.000e+00,") != std::string::npos);
    CHECK(line.find("swap_acc=100.0%") != std::string::npos);
    CHECK(log.find("ANNEAL_PT swap acceptance per pair (cold->hot):  0:100%")
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
    sp.param["L"]        = 10.0;
    sp.param["T_HOT"]    = 0.5;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() < 5000);
    CHECK(lr.read().find("ANNEAL_PT stopped: STOPVAL reached")
          != std::string::npos);
}

TEST_CASE("ANNEAL_PT: RT<1 cools the ladder to T_MIN (hybrid mode)",
          "[HPO][anneal_pt]")
{
    using namespace tadah::hpo;
    auto sp = make_pt_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 5;
    sp.num_threads = 2;
    sp.stop.maxeval = 100000;
    sp.param["REPLICAS"] = 4.0;
    sp.param["L"]        = 5.0;
    sp.param["T_HOT"]    = 1.0;
    sp.param["T_COLD"]   = 0.1;
    sp.param["RT"]       = 0.5;
    sp.param["T_MIN"]    = 1e-2;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(lr.read().find("ANNEAL_PT stopped: T < T_MIN (hot end)")
          != std::string::npos);
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
    sp.param.erase("STEP_AUTO");
    sp.param["CORANA"]      = 1.0;
    sp.param["NS"]          = 2.0;
    sp.param["NT"]          = 2.0;
    sp.param["T_HOT"]       = 1.0;
    sp.param["SWAP_WINDOW"] = 4.0;
    sp.step = 0.123;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, 0.0, 0.0); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT[CORANA]") != std::string::npos);
    CHECK(log.find("ANNEAL_PT epoch=1/") != std::string::npos);
    CHECK(log.find("swap acceptance per pair") != std::string::npos);
    CHECK(log.find("ANNEAL_PT stopped:") != std::string::npos);
    /* Resolved-ladder report: summary line + per-slot table (fires even
     * with a user-set T_HOT, which skips the T0 calibration). The
     * new-best histogram is NOT expected here: the stub host never calls
     * log_new_best (that lives in HPO_Host) -- the origin-tag plumbing
     * has its own HPOLog test case below. */
    CHECK(log.find("ANNEAL_PT ladder: T_HOT=") != std::string::npos);
    CHECK(log.find("ANNEAL_PT ladder T per slot (0=coldest .. 3=hottest):")
          != std::string::npos);
    CHECK(log.find("[0-3]") != std::string::npos);
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
    CHECK(log.find("STEP=0.123") != std::string::npos);
    CHECK(log.find("Corana U(-V,V) half-width") != std::string::npos);
    CHECK(log.find("STEP_FRAC=") == std::string::npos);
    CHECK(log.find("BOUNDARY_MODE=0(REFLECT)") != std::string::npos);
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
    CHECK(txt.find("[slot 3/32 T=2.377e+00]") != std::string::npos);
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
}

TEST_CASE("ANNEAL_PT: validation errors", "[HPO][anneal_pt][validation]")
{
    using namespace tadah::hpo;
    MtStubHost stub{ [&](const Vec&){ return 0.0; }, {}, {} };

    SECTION("automatic controls default on for CLASSIC PT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param.erase("STEP_AUTO");
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK(kb.step_auto);
        CHECK(kb.step_auto_sweeps == 12);
        CHECK(kb.step_adapt_every == 0);
    }
    SECTION("STEP_AUTO is Boolean") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["STEP_AUTO"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("STEP_AUTO"));
    }
    SECTION("STEP_AUTO_SWEEPS must be positive") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["STEP_AUTO"] = 1.0;
        sp.param["STEP_AUTO_SWEEPS"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("STEP_AUTO_SWEEPS"));
    }
    SECTION("automatic controls are CLASSIC PT only") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param.erase("STEP_AUTO");
        sp.param["CORANA"] = 1.0;
        sp.param["NS"] = 1.0;
        sp.param["NT"] = 1.0;
        sp.param["STEP_ADAPT_EVERY"] = 1.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("CLASSIC variant"));
    }
    SECTION("explicit STEP disables automatic calibration") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param.erase("STEP_AUTO");
        sp.step = 0.1;
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        CHECK_FALSE(kb.step_auto);
        CHECK(kb.step_adapt_every == 0);
    }
    SECTION("adaptation cadence is rejected when STEP_AUTO is off") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        sp.param["STEP_AUTO"] = 0.0;
        sp.param["STEP_ADAPT_EVERY"] = 1.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("unused"));
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
    SECTION("INIT_MODE out of range") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["INIT_MODE"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains("INIT_MODE"));
    }
    SECTION("T_MIN exceeds auto-calibrated T_HOT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 100;
        sp.param["REPLICAS"] = 4.0;
        sp.param["RT"] = 0.9;
        sp.param["T_MIN"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pt_impl<Vec>(sp, stub),
                            Catch::Contains(
                                "ANNEAL_PT: resolved T_MIN must be <= resolved T_HOT"));
    }
    SECTION("serial-only REANNEAL rejected on PT") {
        auto sp = make_pt_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["REANNEAL"] = 1.0;
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
    sp.param.erase("STEP_AUTO");
    sp.param["CORANA"]    = 1.0;
    sp.param["NS"]        = 3.0;
    sp.param["NT"]        = 2.0;
    sp.param["T_HOT"]     = 2.0;
    sp.param["LADDER_SPAN"] = 1e3;

    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, cx, cy); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(best_of(stub.sorted_calls()) < 0.05);
}

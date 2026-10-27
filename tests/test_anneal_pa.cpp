/*
 * Unit tests for `run_anneal_pa_impl` (TADAH:ANNEAL_PA -- population
 * annealing over the shared annealing core, algorithms_pa.h) and the
 * pure PA math in anneal_core.h (ESS, adaptive dbeta, systematic
 * resampling).
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms_pa.h>
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
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

tadah::hpo::OptimizerSpec
make_pa_spec(std::vector<double> low, std::vector<double> high)
{
    tadah::hpo::OptimizerSpec sp;
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib  = "TADAH";
    sp.algo = "ANNEAL_PA";
    return sp;
}

void set_seed_point(tadah::hpo::OptimizerSpec& sp,
                    const std::vector<double>& x)
{
    sp.optim_init.assign(1, x);
}

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
                       ("anneal_pa_log_" +
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
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string read() const {
        std::ifstream is(path);
        std::ostringstream oss; oss << is.rdbuf();
        return oss.str();
    }
};

}  // namespace

TEST_CASE("ANNEAL_PA: INIT_MODE JITTER uses Gaussian seed jitter for both steppers",
          "[HPO][anneal_pa][init][kernel]")
{
    using namespace tadah::hpo;
    /* Jitter sigma is the proposal scale STEP_FRAC * width = 0.0125 * 20. */
    constexpr double kJitterSigma = 0.0125 * 20.0;
    auto run = [](bool corana) {
        auto sp = make_pa_spec({-10.0}, {10.0});
        set_seed_point(sp, {0.0});
        sp.seed = 271828;
        sp.num_threads = 2;
        sp.pop = 16;
        sp.stop.maxeval = 16;
        sp.param["STEP_FRAC"] = 0.0125;
        sp.param_raw["INIT_MODE"] = "JITTER";   /* the default, explicit */
        sp.param["T_HOT"] = 1.0;
        if (corana) {
            sp.param_raw["CORANA"] = "ON";
            sp.param["NS"] = 1.0;
            sp.param["NT"] = 1.0;
        } else {
            sp.param["CHAIN_LENGTH"] = 1.0;
        }
        MtStubHost host{[](const Vec& x) { return x(0) * x(0); }, {}, {}};
        run_anneal_pa_impl<Vec>(sp, host);
        return host.sorted_calls();
    };

    for (bool corana : {false, true}) {
        CAPTURE(corana);
        std::vector<std::vector<double>> expected{{0.0}};
        for (std::size_t k = 1; k < 16; ++k) {
            auto rng = anneal::make_stream(271828, anneal::SALT_CHAIN, k);
            const double unit =
                std::normal_distribution<double>(0.0, 1.0)(rng);
            expected.push_back({kJitterSigma * unit});
        }
        std::sort(expected.begin(), expected.end());
        const auto actual = run(corana);
        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            CHECK(actual[i][0] == Approx(expected[i][0]).margin(1e-15));
    }
}

/* ---- pure PA math ---------------------------------------------------- */

TEST_CASE("anneal::systematic_resample expected counts |n_i - N*w_i| <= 1",
          "[HPO][anneal_pa][resample]")
{
    using namespace tadah::hpo;
    const std::vector<double> w{0.5, 0.25, 0.125, 0.125};   /* normalised */
    const std::size_t N = w.size();
    for (double u : {0.0, 0.2, 0.5, 0.99}) {
        const auto parent = anneal::systematic_resample(w, u);
        REQUIRE(parent.size() == N);
        std::map<std::size_t, std::size_t> counts;
        for (auto p : parent) ++counts[p];
        for (std::size_t i = 0; i < N; ++i) {
            const double expect = static_cast<double>(N) * w[i];
            const double got = static_cast<double>(counts[i]);
            CHECK(std::fabs(got - expect) <= 1.0);
        }
    }
}

TEST_CASE("anneal::systematic_resample kills zero-weight walkers",
          "[HPO][anneal_pa][resample]")
{
    using namespace tadah::hpo;
    const std::vector<double> w{1.0, 0.0, 1.0, 0.0};
    const auto parent = anneal::systematic_resample(w, 0.3);
    for (auto p : parent) {
        CHECK(p != 1);
        CHECK(p != 3);
    }
}

TEST_CASE("anneal::systematic_resample skips a leading zero at u=0",
          "[HPO][anneal_pa][resample][numeric]")
{
    using namespace tadah::hpo;
    const auto parent = anneal::systematic_resample({0.0, 1.0, 0.0}, 0.0);
    const std::vector<std::size_t> expected{1, 1, 1};
    REQUIRE(parent.size() == 3);
    CHECK(parent == expected);
}

TEST_CASE("anneal::systematic_resample never selects a trailing zero",
          "[HPO][anneal_pa][resample][numeric]")
{
    using namespace tadah::hpo;
    std::vector<double> w(11, 1.0);
    w.back() = 0.0;
    const auto parent = anneal::systematic_resample(
        w, std::nextafter(1.0, 0.0));
    REQUIRE(parent.size() == w.size());
    for (std::size_t p : parent) CHECK(p < 10);
}

TEST_CASE("anneal::pa_geometric_temperature advances one subnormal level",
          "[HPO][anneal_pa][temperature][numeric]")
{
    using namespace tadah::hpo;
    const double d = std::numeric_limits<double>::denorm_min();
    const double current = 4.0 * d;
    const double rt = std::nextafter(1.0, 0.0);
    REQUIRE(current * rt == current);
    CHECK(anneal::pa_geometric_temperature(current, rt, d) == 3.0 * d);
}

TEST_CASE("anneal::reweight_ess and choose_dbeta", "[HPO][anneal_pa][ess]")
{
    using namespace tadah::hpo;
    const std::vector<double> f{1.0, 2.0, 3.0, 4.0};

    SECTION("ESS is N at dbeta=0 and decreases monotonically") {
        CHECK(anneal::reweight_ess(f, 0.0) == Approx(4.0));
        double prev = 4.0;
        for (double db : {0.1, 0.5, 1.0, 5.0}) {
            const double e = anneal::reweight_ess(f, db);
            CHECK(e <= prev + 1e-12);
            prev = e;
        }
    }
    SECTION("choose_dbeta hits the target ESS") {
        const double target = 0.5;
        const double db = anneal::choose_dbeta(f, target, 100.0);
        CHECK(anneal::reweight_ess(f, db) / 4.0 == Approx(target).margin(1e-6));
    }
    SECTION("cap wins when even the cap keeps ESS above target") {
        const double db = anneal::choose_dbeta(f, 0.5, 1e-3);
        CHECK(db == Approx(1e-3));
    }
    SECTION("degenerate all-equal population keeps ESS = N for any dbeta") {
        const std::vector<double> feq{2.0, 2.0, 2.0};
        CHECK(anneal::reweight_ess(feq, 10.0) == Approx(3.0));
        CHECK(anneal::choose_dbeta(feq, 0.5, 0.7) == Approx(0.7));
    }
    SECTION("opposite DBL_MAX losses remain numerically stable") {
        const double m = std::numeric_limits<double>::max();
        const std::vector<double> extreme{-m, m};
        CHECK(anneal::reweight_ess(extreme, 0.0) == Approx(2.0));
        CHECK(anneal::reweight_ess(extreme, 1.0) == Approx(1.0));
    }
    SECTION("choose_dbeta resolves roots far below the geometric cap") {
        const std::vector<double> wide{0.0, 1.0e300};
        const double db = anneal::choose_dbeta(wide, 0.9, 1.0);
        REQUIRE(db > 0.0);
        CHECK(db < 1.0e-299);
        CHECK(anneal::reweight_ess(wide, db) / 2.0 ==
              Approx(0.9).margin(1e-10));
    }
    SECTION("a rounded adaptive endpoint advances by one honest beta step") {
        const std::vector<double> wide{0.0, 7.0e16};
        const double db = anneal::choose_dbeta(wide, 0.9, 1.0);
        REQUIRE(db > 0.0);
        const double proposed = static_cast<double>(
            1.0L / (1.0L + static_cast<long double>(db)));
        REQUIRE(proposed == 1.0);
        const auto step = anneal::realize_pa_temperature_step(
            1.0, 0.5, proposed);
        CHECK(step.temperature == std::nextafter(1.0, 0.0));
        const long double exact =
            1.0L / static_cast<long double>(step.temperature) - 1.0L;
        CHECK(static_cast<double>(step.dbeta) ==
              Approx(static_cast<double>(exact)));
        const long double w = std::exp(-step.dbeta * 7.0e16L);
        const double achieved = static_cast<double>(
            ((1.0L + w) * (1.0L + w)) / (1.0L + w * w));
        CHECK(achieved >= 1.0);
        CHECK(achieved <= 2.0);
    }
}

/* ---- driver ----------------------------------------------------------- */

TEST_CASE("ANNEAL_PA: thread-count invariance (same SEED => same eval set)",
          "[HPO][anneal_pa][determinism]")
{
    using namespace tadah::hpo;
    auto run_with_threads = [&](unsigned threads) {
        auto sp = make_pa_spec({-1.0, -1.0}, {1.0, 1.0});
        set_seed_point(sp, {0.5, -0.5});
        sp.seed = 777;
        sp.num_threads = threads;
        sp.pop = 16;
        sp.stop.maxeval = 700;
        sp.param_raw["CORANA"] = "ON";
        sp.param["NS"]     = 2.0;
        sp.param["NT"]     = 1.0;
        sp.param["T_HOT"]  = 1.0;
        sp.param["T_MIN"]  = 1e-3;
        MtStubHost stub{ [&](const Vec& x){
            const double dx = x(0) - 0.1, dy = x(1) + 0.2;
            return dx * dx + dy * dy;
        }, {}, {} };
        run_anneal_pa_impl<Vec>(sp, stub);
        return stub.sorted_calls();
    };

    const auto a = run_with_threads(1);
    const auto b = run_with_threads(4);
    REQUIRE_FALSE(a.empty());
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t d = 0; d < a[i].size(); ++d) {
            CHECK(a[i][d] == Approx(b[i][d]).margin(1e-15));
        }
    }
}

TEST_CASE("ANNEAL_PA: FAILSCORE walkers are culled at the first resample",
          "[HPO][anneal_pa][cliff]")
{
    using namespace tadah::hpo;
    constexpr double FS = 1e16;
    /* Left half of the box is a cliff; walkers jittered there must die
     * at the first resampling and the population must concentrate on
     * the finite right half. */
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 5;
    sp.failscore = FS;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 800;
    sp.param["CHAIN_LENGTH"]         = 4.0;
    sp.param["T_HOT"]     = 0.5;
    sp.param["T_MIN"]     = 1e-3;
    sp.param_raw["INIT_MODE"] = "UNIFORM";  /* box draw -> half start on cliff */
    sp.param["STEP_FRAC"] = 0.10;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){
        return x(0) >= 0.0 ? x(0) * x(0) : FS;
    }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* After the first logged step the finite fraction must be full:
     * every cliff walker was replaced by a finite parent's copy. */
    const auto pos = log.find("finite=16/16");
    CHECK(pos != std::string::npos);
    CHECK(log.find("ANNEAL_PA stopped:") != std::string::npos);
}

/* Pins: without THREADS the journal says, as a WARNING, that the walkers
 * run on one worker thread and what THREADS does: "ANNEAL_PA: THREADS is
 * not set, so the 8 walkers run on 1 worker thread; THREADS <n> in the
 * STRATEGY block evaluates up to n of them at a time".
 * Before the fix: "ANNEAL_PA: THREADS not set -- running 8 walkers on 1
 * worker thread. Set THREADS <n> in the STRATEGY block to evaluate walkers
 * concurrently.": the check fails. */
TEST_CASE("ANNEAL_PA: without THREADS the journal says that the walkers run "
          "on one worker thread", "[HPO][anneal_pa][log]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 7;
    sp.pop = 8;
    sp.stop.maxeval = 16;
    sp.param["T_HOT"] = 0.5;
    sp.param["STEP_FRAC"] = 0.1;
    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);
    const std::string log = lr.read();
    INFO(log);
    CHECK(log.find("ANNEAL_PA: THREADS is not set, so the 8 walkers "
                   "run on 1 worker thread; THREADS <n> in the STRATEGY block "
                   "evaluates up to n of them at a time") !=
          std::string::npos);
}

TEST_CASE("ANNEAL_PA: MAXEVAL is a hard cap across the population",
          "[HPO][anneal_pa]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.2});
    sp.seed = 3;
    sp.num_threads = 4;
    sp.pop = 16;
    sp.stop.maxeval = 231;
    sp.param["CHAIN_LENGTH"]     = 5.0;
    sp.param["T_HOT"] = 1.0;

    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() == 231);
}

TEST_CASE("ANNEAL_PA: MAXEVAL zero performs no host evaluation",
          "[HPO][anneal_pa][budget]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 3;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 0;
    sp.param["CHAIN_LENGTH"]     = 2.0;
    sp.param["T_HOT"] = 1.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().empty());
    CHECK(lr.read().find("after 0 host eval(s)") != std::string::npos);
}

TEST_CASE("ANNEAL_PA: T0 probe exhaustion reports MAXEVAL exactly",
          "[HPO][anneal_pa][budget]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.25});
    sp.seed = 31;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 3;
    sp.param["CHAIN_LENGTH"] = 2.0;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);
    CHECK(stub.sorted_calls().size() == 3);
    CHECK(lr.read().find("ANNEAL_PA stopped: MAXEVAL exhausted")
          != std::string::npos);
    /* Pins: the run's status "TADAH:ANNEAL_PA: MAXEVAL reached". Before the
     * fix: no status (the trailer said COMPLETED). */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PA: MAXEVAL reached");
}

TEST_CASE("ANNEAL_PA: reaches T_MIN and reports the schedule",
          "[HPO][anneal_pa][log]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 11;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 100000;
    sp.param["CHAIN_LENGTH"]          = 2.0;
    sp.param["T_HOT"]      = 1.0;
    sp.param["T_MIN"]      = 0.05;
    sp.param["RT"]         = 0.5;
    sp.param["ESS_TARGET"] = 0.5;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PA[CLASSIC]") != std::string::npos);
    CHECK(log.find("ANNEAL_PA step=1/") != std::string::npos);
    CHECK(log.find("ESS=") != std::string::npos);
    CHECK(log.find("ANNEAL_PA stopped: T <= T_MIN") != std::string::npos);
    /* Pins: the run's status "TADAH:ANNEAL_PA: T_MIN reached". Before the
     * fix: no status. */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PA: T_MIN reached");
}

TEST_CASE("ANNEAL_PA: T_HOT equal to T_MIN runs one hot sweep only",
          "[HPO][anneal_pa][temperature]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 19;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 100;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"] = 1.0;
    sp.param["CHAIN_LENGTH"] = 1.0;

    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    LoggedRun lr;
    run_anneal_pa_impl<Vec>(sp, stub);

    /* 16 population starts (seed included) + one proposal per walker. */
    CHECK(stub.sorted_calls().size() == 32);
    CHECK(lr.read().find("ANNEAL_PA stopped: T <= T_MIN") !=
          std::string::npos);
    /* Pins: a one-temperature run ends "TADAH:ANNEAL_PA: T_MIN reached".
     * Before the fix: no status. */
    CHECK(HPOLog::instance().status() == "TADAH:ANNEAL_PA: T_MIN reached");
}

TEST_CASE("ANNEAL_PA: subnormal temperatures make geometric progress",
          "[HPO][anneal_pa][temperature]")
{
    using namespace tadah::hpo;
    const double tiny = std::numeric_limits<double>::denorm_min();
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 23;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 100;
    sp.param["T_HOT"] = 2.0 * tiny;
    sp.param["T_MIN"] = tiny;
    sp.param["RT"] = 0.5;
    sp.param["CHAIN_LENGTH"] = 1.0;

    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    LoggedRun lr;
    run_anneal_pa_impl<Vec>(sp, stub);

    /* Starts + hot sweep + exactly one sweep at the representable floor. */
    CHECK(stub.sorted_calls().size() == 48);
    CHECK(lr.read().find("ANNEAL_PA stopped: T <= T_MIN") !=
          std::string::npos);
}

TEST_CASE("ANNEAL_PA: ESS_TARGET OFF gives the pure geometric schedule",
          "[HPO][anneal_pa][log]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 11;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 100000;
    sp.param["CHAIN_LENGTH"]          = 2.0;
    sp.param["T_HOT"]      = 1.0;
    sp.param["T_MIN"]      = 0.2;
    sp.param["RT"]         = 0.5;
    sp.param_raw["ESS_TARGET"] = "OFF";

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("[geometric]") != std::string::npos);
    CHECK(log.find("ANNEAL_PA stopped: T <= T_MIN") != std::string::npos);
}

TEST_CASE("ANNEAL_PA: Corana banner and step diagnostics expose bounds",
          "[HPO][anneal_pa][log]")
{
    using namespace tadah::hpo;
    auto sp = make_pa_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.2, -0.2});
    sp.seed = 71;
    sp.num_threads = 2;
    sp.pop = 16;
    sp.stop.maxeval = 160;
    sp.param_raw["CORANA"] = "ON";
    sp.param["NS"] = 1.0;
    sp.param["NT"] = 1.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["STEP_FRAC"] = 0.123;

    LoggedRun lr;
    MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0) + x(1) * x(1); }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PA[CORANA]") != std::string::npos);
    CHECK(log.find("CORANA_C=") != std::string::npos);
    CHECK(log.find("V_FLOOR_FRAC=") != std::string::npos);
    CHECK(log.find("STEP_FRAC=0.123 (Corana V half-width/width)")
          != std::string::npos);
    CHECK(log.find("BOUNDARY_MODE=REFLECT") != std::string::npos);
    CHECK(log.find("INIT_MODE=JITTER") != std::string::npos);
    CHECK(log.find("V_floor=") != std::string::npos);
    CHECK(log.find("V_cap=") != std::string::npos);
}

/* Pins: INIT_MODE JITTER AUTO in ANNEAL_PA draws each walker's start
 * uniformly within the proposal scale of each dimension (the CLASSIC
 * sigma, the Corana initial V): ANNEAL_PA calibrates nothing and takes no
 * tolerance, and the banner says so ("INIT_MODE=JITTER AUTO (uniform
 * bound; sigma=proposal sigma per dim, not calibrated)").
 * Before the fix: JITTER AUTO without a tolerance was refused ("needs a
 * relative loss tolerance"), and with one the banner said "not calibrated:
 * tolerance 0.05 unused" (before that, "(uniform bound; sigma=AUTO per
 * dim, tol=0.05)", which names a calibration ANNEAL_PA does not run). */
TEST_CASE("ANNEAL_PA: the INIT_MODE JITTER AUTO banner names the scale the "
          "draw uses", "[HPO][anneal_pa][log]")
{
    using namespace tadah::hpo;
    for (const bool corana : {false, true}) {
        INFO("CORANA " << corana);
        auto sp = make_pa_spec({-1.0, -1.0}, {1.0, 1.0});
        set_seed_point(sp, {0.2, -0.2});
        sp.seed = 71;
        sp.num_threads = 2;
        sp.pop = 16;
        sp.stop.maxeval = 160;
        sp.param["T_HOT"] = 1.0;
        sp.param_raw["INIT_MODE"] = "JITTER";
        sp.param_tokens["INIT_MODE"] = {"JITTER", "AUTO"};
        if (corana) {
            sp.param_raw["CORANA"] = "ON";
            sp.param["NS"] = 1.0;
            sp.param["NT"] = 1.0;
        }
        LoggedRun lr;
        MtStubHost stub{ [&](const Vec& x){ return x(0) * x(0) + x(1) * x(1); }, {}, {} };
        run_anneal_pa_impl<Vec>(sp, stub);
        const std::string log = lr.read();
        INFO(log);
        CHECK(log.find(std::string("INIT_MODE=JITTER AUTO (uniform bound; "
                                   "sigma=") +
                       (corana ? "initial V" : "proposal sigma") +
                       " per dim, not calibrated)") != std::string::npos);
        CHECK(log.find("sigma=AUTO") == std::string::npos);
    }
}

TEST_CASE("ANNEAL_PA: validation errors", "[HPO][anneal_pa][validation]")
{
    using namespace tadah::hpo;
    MtStubHost stub{ [&](const Vec&){ return 0.0; }, {}, {} };

    SECTION("POPULATION too small") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.pop = 4;
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains("POPULATION"));
    }
    SECTION("ESS_TARGET out of range") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["ESS_TARGET"] = 1.0;
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains("ESS_TARGET"));
    }
    SECTION("SWEEPS 0") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["SWEEPS"] = 0.0;
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains("SWEEPS"));
    }
    SECTION("T_MIN exceeds auto-calibrated T_HOT") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 100;
        sp.pop = 16;
        sp.param["T_MIN"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains(
                                "ANNEAL_PA: resolved T_MIN must be <= resolved T_HOT"));
    }
    SECTION("PT-only knob rejected with steering") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["SWAP_EVERY"] = 2.0;
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains("ANNEAL_PT"));
    }
    /* Pins: ANNEAL_PA refuses the keys of the INIT_MODE JITTER AUTO
     * calibration of ANNEAL_PT, which it does not run, naming ANNEAL_PT,
     * and the tolerance of JITTER AUTO, which it does not use.
     * Before the fix: each key and the tolerance were accepted and had no
     * effect. */
    SECTION("the calibration keys and the AUTO tolerance of ANNEAL_PT") {
        for (const char* k : {"INIT_JITTER_MAX", "INIT_JITTER_PROBES",
                              "INIT_JITTER_MAXEVAL", "INIT_JITTER_SAMPLE",
                              "INIT_JITTER_AUDIT",
                              "INIT_JITTER_MAXTIME_FRAC"}) {
            INFO(k);
            auto sp = make_pa_spec({-1.0}, {1.0});
            set_seed_point(sp, {0.0});
            sp.stop.maxeval = 10;
            sp.pop = 16;
            /* With the tolerance ANNEAL_PT takes: the key is refused
             * before the tolerance is read. */
            sp.param_raw["INIT_MODE"] = "JITTER";
            sp.param_tokens["INIT_MODE"] = {"JITTER", "AUTO", "0.05"};
            sp.param[k] = std::string(k) == "INIT_JITTER_MAX" ||
                                  std::string(k) == "INIT_JITTER_MAXTIME_FRAC"
                              ? 0.25 : 4.0;
            REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                                /* It was "... belongs to ...". */
                                std::string("TADAH:ANNEAL_PA: PARAM ") + k +
                                    " is a key of TADAH:ANNEAL_PT");
        }
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.pop = 16;
        sp.param_raw["INIT_MODE"] = "JITTER";
        sp.param_tokens["INIT_MODE"] = {"JITTER", "AUTO", "0.05"};
        REQUIRE_THROWS_WITH(
            run_anneal_pa_impl<Vec>(sp, stub),
            "TADAH:ANNEAL_PA: PARAM INIT_MODE JITTER AUTO takes no tolerance "
            "(got '0.05'): TADAH:ANNEAL_PA measures no displacement and "
            "takes the first proposal scales as the bounds; TADAH:ANNEAL_PT "
            "measures them against the tolerance");
    }
    SECTION("RESET_TO_BEST is rejected instead of silently ignored") {
        auto sp = make_pa_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.pop = 16;
        sp.param_raw["RESET_TO_BEST"] = "ON";
        REQUIRE_THROWS_WITH(run_anneal_pa_impl<Vec>(sp, stub),
                            Catch::Contains("RESET_TO_BEST"));
    }
}

TEST_CASE("ANNEAL_PA: converges on a quadratic bowl",
          "[HPO][anneal_pa]")
{
    using namespace tadah::hpo;
    const double cx = 0.3, cy = -0.4;
    auto sp = make_pa_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {-0.8, 0.8});
    sp.seed = 99;
    sp.num_threads = 4;
    sp.pop = 16;
    sp.stop.maxeval = 1500;
    sp.param_raw["CORANA"] = "ON";
    sp.param["NS"]     = 2.0;
    sp.param["NT"]     = 1.0;
    sp.param["T_HOT"]  = 1.0;
    sp.param["T_MIN"]  = 1e-4;

    MtStubHost stub{ [&](const Vec& x){
        const double dx = x(0) - cx, dy = x(1) - cy;
        return dx * dx + dy * dy;
    }, {}, {} };
    run_anneal_pa_impl<Vec>(sp, stub);

    double best = std::numeric_limits<double>::infinity();
    for (const auto& p : stub.sorted_calls()) {
        const double dx = p[0] - cx, dy = p[1] - cy;
        best = std::min(best, dx * dx + dy * dy);
    }
    CHECK(best < 0.05);
}

/* Seed/T0 calibration used to re-throw whatever the host threw as a
 * plain runtime_error(what), so SIGINT during calibration ended the run
 * with an error (non-zero exit) instead of SHUTDOWN, and a fatal
 * best-potential persistence failure lost its type. Eval 1 is the seed,
 * eval 2 the first T0 probe (T_HOT unset). */
TEST_CASE("ANNEAL_PA: calibration rethrows the host's exception unchanged",
          "[HPO][anneal_pa][failure_handling]")
{
    using namespace tadah::hpo;
    for (std::size_t throw_at : {std::size_t{1}, std::size_t{2}}) {
        CAPTURE(throw_at);
        auto make = [&] {
            auto sp = make_pa_spec({-1.0, -1.0}, {1.0, 1.0});
            set_seed_point(sp, {0.0, 0.0});
            sp.seed = 5;
            sp.num_threads = 2;
            sp.stop.maxeval = 200;
            sp.pop = 8;
            return sp;
        };
        {
            auto sp = make();
            std::atomic<std::size_t> n{0};
            MtStubHost stub{[&](const Vec& x) {
                if (++n == throw_at) throw tadah::core::ShutdownRequested{};
                return x(0) * x(0) + x(1) * x(1);
            }, {}, {}};
            CHECK_THROWS_AS(run_anneal_pa_impl<Vec>(sp, stub),
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
                run_anneal_pa_impl<Vec>(sp, stub);
                FAIL("run_anneal_pa_impl returned normally");
            } catch (const durable_best::Error& e) {
                CHECK(std::string(e.what()) == "best_pot.tadah: gave up");
            }
        }
    }
}

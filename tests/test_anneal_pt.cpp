/*
 * Unit tests for `run_anneal_pt_impl` (TADAH:ANNEAL_PT -- parallel
 * tempering over the shared annealing core, algorithms_pt.h).
 *
 * A thread-safe stub host records every parameter vector, so the tests
 * can assert the determinism contract: same SEED => the same SET of
 * evaluated points for ANY thread count (call ORDER differs across
 * thread counts by design; per-chain sequences are fixed).
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
#include <mutex>
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

}  // namespace

TEST_CASE("ANNEAL_PT: thread-count invariance (same SEED => same eval set)",
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
    CHECK(stub.sorted_calls().size() <= 137);
    CHECK(stub.sorted_calls().size() >= 100);   /* budget actually used */
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
    sp.param["REPLICAS"] = 4.0;
    sp.param["CORANA"]   = 1.0;
    sp.param["NS"]       = 2.0;
    sp.param["NT"]       = 2.0;
    sp.param["T_HOT"]    = 1.0;

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
    CHECK(log.find("ANNEAL_PT walker per slot (pre-swap):") != std::string::npos);
    /* Corana V-state blocks (per-line INFO rows) + knobs in the banner. */
    CHECK(log.find("ANNEAL_PT V geo-mean per slot") != std::string::npos);
    CHECK(log.find("ANNEAL_PT V min per slot (min_i V[i]/width[i]):")
          != std::string::npos);
    CHECK(log.find("ANNEAL_PT V max per slot (max_i V[i]/width[i]):")
          != std::string::npos);
    CHECK(log.find("V_FLOOR_FRAC=") != std::string::npos);
    CHECK(log.find("STEP_FRAC=") != std::string::npos);
}

TEST_CASE("HPOLog: best-origin tag (slot/T + probe) and per-slot histograms",
          "[HPO][anneal_pt][log]")
{
    using tadah::hpo::HPOLog;
    LoggedRun lr;
    auto& log = HPOLog::instance();

    /* A best recorded while the thread is stamped as a T0 probe. */
    HPOLog::set_eval_probe();
    log.log_new_best(1, 10.0, 0,false, 0,false, 0,false, 0,false, 0,false,
                     1.0);
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
    sp.param["CORANA"]    = 1.0;
    sp.param["NS"]        = 3.0;
    sp.param["NT"]        = 2.0;
    sp.param["T_HOT"]     = 2.0;
    sp.param["LADDER_SPAN"] = 1e3;

    MtStubHost stub{ [&](const Vec& x){ return quad_bowl2(x, cx, cy); }, {}, {} };
    run_anneal_pt_impl<Vec>(sp, stub);
    CHECK(best_of(stub.sorted_calls()) < 0.05);
}

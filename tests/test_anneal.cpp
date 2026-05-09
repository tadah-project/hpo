/*
 * Unit tests for `run_anneal_impl` (TADAH:ANNEAL — classic
 * Kirkpatrick/Metropolis SA driver in algorithms.h).
 *
 * The tests use a tiny stub host so the SA loop can be exercised
 * without constructing a full HPO_Host (training/validation
 * StructureDBs and a LAMMPS runner).
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

/* Build a minimal OptimizerSpec with explicit bounds. The caller
 * supplies un-transformed bounds in real space; analyse_logscale-style
 * mutation isn't needed because every test runs with linear (non-log)
 * dims. */
tadah::hpo::OptimizerSpec
make_spec(std::vector<double> low, std::vector<double> high)
{
    tadah::hpo::OptimizerSpec sp;
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib  = "TADAH";
    sp.algo = "ANNEAL";
    return sp;
}

/* Set the SA starting state via the same field run_random/run_grid
 * read (`optim_init.front()`). Real space; `run_anneal_impl` calls
 * `sp.in(...)` to convert. */
void set_seed_point(tadah::hpo::OptimizerSpec& sp,
                    const std::vector<double>& x)
{
    sp.optim_init.assign(1, x);
}

/* Records every parameter vector the SA loop submits to host(). */
struct StubHost {
    std::function<double(const Vec&)> f;
    mutable std::vector<std::vector<double>> calls;
    double operator()(const Vec& x) const {
        std::vector<double> v(static_cast<std::size_t>(x.size()));
        for (long i = 0; i < x.size(); ++i) v[static_cast<std::size_t>(i)] = x(i);
        calls.push_back(v);
        return f(x);
    }
};

double quad_bowl(const Vec& x, const std::vector<double>& centre)
{
    double s = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double d = x(i) - centre[static_cast<std::size_t>(i)];
        s += d * d;
    }
    return s;
}

}  // namespace

TEST_CASE("ANNEAL: every host() call lies inside the bounds",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -2.0}, {1.0, 2.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 11;
    sp.stop.maxeval = 200;
    sp.param["L"] = 10.0;
    sp.param["RT"] = 0.9;
    sp.param["STEP_FRAC"] = 0.2;

    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.3, -0.5}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    REQUIRE_FALSE(stub.calls.empty());
    for (const auto& p : stub.calls) {
        REQUIRE(p.size() == 2);
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(p[d] >= sp.low[d] - 1e-12);
            CHECK(p[d] <= sp.high[d] + 1e-12);
        }
    }
}

TEST_CASE("ANNEAL: same seed reproduces the host()-call sequence",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 12345;
    sp.stop.maxeval = 80;
    sp.param["L"] = 8.0;

    auto run = [&]() {
        StubHost s{ [&](const Vec& x){ return quad_bowl(x, {0.2, 0.4}); }, {} };
        run_anneal_impl<Vec>(sp, s);
        return s.calls;
    };

    auto a = run();
    auto b = run();
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].size() == b[i].size());
        for (std::size_t d = 0; d < a[i].size(); ++d) {
            CHECK(a[i][d] == Approx(b[i][d]).margin(1e-15));
        }
    }
}

TEST_CASE("ANNEAL: MAXEVAL is a hard cap on total host() calls",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0, 0.0});
    sp.seed = 7;
    sp.stop.maxeval = 25;
    sp.param["L"] = 5.0;
    sp.param["T0_PROBES"] = 10.0;

    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    CHECK(stub.calls.size() <= 25);
    CHECK(stub.calls.size() >= 11);   // probes + seed at minimum
}

TEST_CASE("ANNEAL: PARAM T_INIT skips T0 calibration probes",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 99;
    sp.stop.maxeval = 30;
    sp.param["T_INIT"] = 0.5;
    sp.param["L"]      = 4.0;
    sp.param["T0_PROBES"] = 20.0;   // would otherwise eat the entire budget

    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    /* With T_INIT set, the very first host() call is the seed
     * evaluation — not a uniform-bounds probe. The seed point we set
     * was (0,0) exactly, so the first recorded call must be (0,0). */
    REQUIRE_FALSE(stub.calls.empty());
    CHECK(stub.calls[0][0] == Approx(0.0));
    CHECK(stub.calls[0][1] == Approx(0.0));
}

TEST_CASE("ANNEAL: T0 calibration falls back to 1.0 when probes return a constant",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 3;
    sp.stop.maxeval = 30;
    sp.param["L"]         = 4.0;
    sp.param["T0_PROBES"] = 5.0;

    /* Constant host: every probe returns the same value, so mean|dE|=0. */
    StubHost stub{ [&](const Vec& /*x*/){ return 1.234; }, {} };
    REQUIRE_NOTHROW(run_anneal_impl<Vec>(sp, stub));
    /* Probes (5) + seed (1) + at least one Markov chain proposal (>=1). */
    CHECK(stub.calls.size() >= 6);
}

TEST_CASE("ANNEAL: STOPVAL terminates the loop on first qualifying eval",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 1;
    sp.stop.maxeval = 100;
    sp.stop.stopval = 0.05;
    sp.param["T_INIT"] = 1.0;     // skip probes for a deterministic start
    sp.param["L"]      = 200.0;   // generous chain length

    /* Ramp loss down deterministically: kth call returns 1.0 / k. */
    std::atomic<int> k{0};
    StubHost stub{
        [&](const Vec& /*x*/){
            const int n = ++k;
            return 1.0 / static_cast<double>(n);
        }, {}
    };
    run_anneal_impl<Vec>(sp, stub);

    /* 1.0/k <= 0.05  ⇔  k >= 20. So the loop should stop somewhere
     * around k == 20 — well below MAXEVAL=100. */
    CHECK(stub.calls.size() < 100);
    CHECK(stub.calls.size() >= 20);
}

TEST_CASE("ANNEAL: rolls downhill on a smooth quadratic bowl",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.9, -0.9});
    sp.seed = 42;
    sp.stop.maxeval = 600;
    sp.param["L"]         = 30.0;
    sp.param["RT"]        = 0.85;
    sp.param["STEP_FRAC"] = 0.15;
    sp.param["T0_PROBES"] = 20.0;

    const std::vector<double> centre{0.1, -0.2};
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, centre); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    REQUIRE_FALSE(stub.calls.empty());
    /* Initial loss at the seed (0.9, -0.9) vs centre (0.1, -0.2) is
     * 0.64 + 0.49 = 1.13. We expect SA to find at least one point
     * notably better than the seed within 600 evals; loose target
     * 0.10 stays comfortably above the noise floor of the search. */
    double best = std::numeric_limits<double>::infinity();
    for (const auto& p : stub.calls) {
        Vec x(2); x(0) = p[0]; x(1) = p[1];
        const double f = quad_bowl(x, centre);
        if (f < best) best = f;
    }
    CHECK(best < 0.10);
}

TEST_CASE("ANNEAL: invalid PARAM RT throws",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.stop.maxeval = 20;
    sp.param["RT"] = 1.5;  // out of (0,1)

    StubHost stub{ [&](const Vec& /*x*/){ return 0.0; }, {} };
    REQUIRE_THROWS_AS(run_anneal_impl<Vec>(sp, stub), std::runtime_error);
}

namespace {

/* RAII helper that opens HPOLog on a temp file, runs `body()`, then
 * reads the file back. Resets HPOLog state between cases. */
struct LoggedRun {
    std::filesystem::path path;
    LoggedRun() : path(std::filesystem::temp_directory_path() /
                       ("anneal_log_" +
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

TEST_CASE("ANNEAL: per-rung summary line is emitted to the journal",
          "[HPO][anneal][log]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 42;
    sp.stop.maxeval = 80;
    sp.param["T_INIT"] = 1.0;   // skip probes for a deterministic count
    sp.param["L"]      = 10.0;
    sp.param["RT"]     = 0.7;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.1, -0.2}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* At least one rung line and the final summary + stop lines. */
    CHECK(log.find("ANNEAL rung k=0") != std::string::npos);
    CHECK(log.find("ANNEAL summary: rungs=") != std::string::npos);
    CHECK(log.find("ANNEAL stopped:") != std::string::npos);
}

TEST_CASE("ANNEAL: stop-reason line names the condition that fired",
          "[HPO][anneal][log]")
{
    using namespace tadah::hpo;

    SECTION("MAXEVAL")
    {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 1;
        sp.stop.maxeval = 6;       // tiny budget
        sp.param["T_INIT"] = 1.0;  // skip probes
        sp.param["L"]      = 100.0;

        LoggedRun lr;
        StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0}); }, {} };
        run_anneal_impl<Vec>(sp, stub);
        CHECK(lr.read().find("ANNEAL stopped: MAXEVAL exhausted")
              != std::string::npos);
    }

    SECTION("STOPVAL")
    {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 1;
        sp.stop.maxeval = 50;
        sp.stop.stopval = 1e6;     // first eval will satisfy it
        sp.param["T_INIT"] = 1.0;

        LoggedRun lr;
        StubHost stub{ [&](const Vec& /*x*/){ return 0.0; }, {} };
        run_anneal_impl<Vec>(sp, stub);
        CHECK(lr.read().find("ANNEAL stopped: STOPVAL reached")
              != std::string::npos);
    }

    SECTION("T_MIN")
    {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.seed = 1;
        sp.stop.maxeval = 1000;
        sp.param["T_INIT"] = 1.0;
        sp.param["T_MIN"]  = 0.5;     // T0=1.0; one cool to 0.7; second cool to 0.49 < T_MIN
        sp.param["RT"]     = 0.7;
        sp.param["L"]      = 4.0;

        LoggedRun lr;
        StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0}); }, {} };
        run_anneal_impl<Vec>(sp, stub);
        CHECK(lr.read().find("ANNEAL stopped: T < T_MIN")
              != std::string::npos);
    }
}

TEST_CASE("ANNEAL: T0 line reports loss range and std",
          "[HPO][anneal][log]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 5;
    sp.stop.maxeval = 50;
    sp.param["T0_PROBES"] = 8.0;
    sp.param["L"]         = 5.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL: T0=") != std::string::npos);
    CHECK(log.find("loss range=[")    != std::string::npos);
    CHECK(log.find("std=")            != std::string::npos);
    CHECK(log.find("mean|dE|=")       != std::string::npos);
}

TEST_CASE("ANNEAL: STEP override echoes uniform sigma in the banner",
          "[HPO][anneal][log]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 7;
    sp.stop.maxeval = 30;
    sp.step = 0.123;
    sp.param["T_INIT"] = 1.0;
    sp.param["L"]      = 5.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("sigma=") != std::string::npos);
    CHECK(log.find("STEP override") != std::string::npos);
    CHECK(log.find("STEP_FRAC=") == std::string::npos);
}

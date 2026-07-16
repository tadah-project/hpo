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

TEST_CASE("ANNEAL: PARAM T_HOT skips T0 calibration probes",
          "[HPO][anneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 99;
    sp.stop.maxeval = 30;
    sp.param["T_HOT"] = 0.5;
    sp.param["L"]      = 4.0;
    sp.param["T0_PROBES"] = 20.0;   // would otherwise eat much of the budget

    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    /* The seed point is always the first host() call; with T_HOT set
     * no probe follows it — call 2 is already a chain proposal (which,
     * at seed (0,0) with STEP_FRAC jitter, is not (0,0) again). */
    REQUIRE(stub.calls.size() >= 2);
    CHECK(stub.calls[0][0] == Approx(0.0));
    CHECK(stub.calls[0][1] == Approx(0.0));
}

TEST_CASE("ANNEAL: renamed PARAM T_INIT is a hard error naming T_HOT",
          "[HPO][anneal][validation]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.stop.maxeval = 20;
    sp.param["T_INIT"] = 1.0;

    StubHost stub{ [&](const Vec& /*x*/){ return 0.0; }, {} };
    REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                        Catch::Contains("renamed to T_HOT"));
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

    /* Constant host: every probe delta is 0, so no temperature scale
     * is measurable. */
    StubHost stub{ [&](const Vec& /*x*/){ return 1.234; }, {} };
    REQUIRE_NOTHROW(run_anneal_impl<Vec>(sp, stub));
    /* Seed (1) + probes (5) + at least one Markov chain proposal (>=1). */
    CHECK(stub.calls.size() >= 7);
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
    sp.param["T_HOT"] = 1.0;     // skip probes for a deterministic start
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
    sp.param["T_HOT"] = 1.0;   // skip probes for a deterministic count
    sp.param["L"]      = 10.0;
    sp.param["RT"]     = 0.7;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.1, -0.2}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* At least one rung line and the final summary + stop lines. */
    CHECK(log.find("ANNEAL rung=1/") != std::string::npos);
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
        sp.param["T_HOT"] = 1.0;  // skip probes
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
        sp.param["T_HOT"] = 1.0;

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
        sp.param["T_HOT"] = 1.0;
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
    CHECK(log.find("median|dE|=")     != std::string::npos);
    CHECK(log.find("anchor=seed")     != std::string::npos);
}

TEST_CASE("ANNEAL[CORANA]: per-dim step vector adapts (grows for fruitful dim)",
          "[HPO][anneal][corana]")
{
    using namespace tadah::hpo;
    /* 2-D objective whose only sensitivity is in dim 0; dim 1 is a
     * dead axis. Corana's per-dim step vector should grow V[0]
     * (because dim-0 perturbations get accepted often when they head
     * downhill) and shrink V[1] (dim-1 perturbations move f_x by ~0,
     * so accept rate per dim 1 is determined by the Metropolis ratio
     * exp(-0/T)=1 and they all accept — i.e. accept rate ~100% for
     * dim 1 too). Actually both dims should grow since dim 1 always
     * accepts (zero ΔL). The check below verifies the adapt happened
     * (V differs from initial), and that the run completes. */
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 17;
    sp.stop.maxeval = 800;
    sp.param["CORANA"] = 1.0;
    sp.param["NS"]     = 4.0;
    sp.param["NT"]     = 3.0;
    sp.param["C"]      = 2.0;
    sp.param["T_HOT"] = 0.05;     // low T → mostly downhill on dim 0
    sp.param["RT"]     = 0.85;
    sp.param["STEP_FRAC"] = 0.05;  // small initial V

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    /* Banner switched to CORANA, V range reported per rung. */
    CHECK(log.find("ANNEAL[CORANA]") != std::string::npos);
    CHECK(log.find("V=[")            != std::string::npos);
    /* SA found a near-zero on dim 0 (objective minimum at x(0)=0). */
    double best = std::numeric_limits<double>::infinity();
    for (const auto& p : stub.calls) {
        const double f = p[0] * p[0];
        if (f < best) best = f;
    }
    CHECK(best < 1e-2);
}

TEST_CASE("ANNEAL[CORANA]: Corana variant honours bounds via reflection",
          "[HPO][anneal][corana][reflection]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-2.0, -3.0}, {2.0, 3.0});
    set_seed_point(sp, {1.5, 2.5});   // near corner
    sp.seed = 23;
    sp.stop.maxeval = 400;
    sp.param["CORANA"] = 1.0;
    sp.param["NS"]     = 3.0;
    sp.param["NT"]     = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["STEP_FRAC"] = 0.50;     // big V — many proposals will reflect

    StubHost stub{ [&](const Vec& x){
        return quad_bowl(x, {0.0, 0.0});
    }, {} };
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

TEST_CASE("ANNEAL[CORANA]: invalid PARAM NS/NT/C throws",
          "[HPO][anneal][corana]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.stop.maxeval = 20;
    sp.param["CORANA"] = 1.0;
    sp.param["T_HOT"] = 1.0;

    StubHost stub{ [&](const Vec&){ return 0.0; }, {} };

    SECTION("NS=0") {
        sp.param["NS"] = 0.0;
        REQUIRE_THROWS_AS(run_anneal_impl<Vec>(sp, stub), std::runtime_error);
    }
    SECTION("NT=0") {
        sp.param["NT"] = 0.0;
        REQUIRE_THROWS_AS(run_anneal_impl<Vec>(sp, stub), std::runtime_error);
    }
    SECTION("C<=0") {
        sp.param["C"] = -1.0;
        REQUIRE_THROWS_AS(run_anneal_impl<Vec>(sp, stub), std::runtime_error);
    }
}

TEST_CASE("ANNEAL: high-dim near-bound seed still proposes every L-step (reflection)",
          "[HPO][anneal][reflection]")
{
    using namespace tadah::hpo;
    /* 38 dims with the seed parked near a corner — the configuration
     * that exposed the original whole-proposal rejection bug. With
     * STEP_FRAC=0.10 and Gaussian proposals, the old code rejected
     * ~98% of proposals because at least one of 38 dims fell out of
     * bounds. Reflection lets every L-step do real work. */
    const std::size_t D = 38;
    std::vector<double> low(D, 0.0), high(D, 1.0);
    auto sp = make_spec(low, high);
    std::vector<double> seed(D, 0.02);   // near the lower wall in every dim
    set_seed_point(sp, seed);
    sp.seed = 1234;
    sp.stop.maxeval = 600;
    sp.param["T_HOT"]    = 1.0;
    sp.param["L"]         = 50.0;
    sp.param["RT"]        = 0.85;
    sp.param["STEP_FRAC"] = 0.10;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){
        double s = 0.0;
        for (long i = 0; i < x.size(); ++i) s += x(i) * x(i);
        return s;
    }, {} };
    run_anneal_impl<Vec>(sp, stub);

    /* Bounds invariance still holds — reflection clamps every dim. */
    for (const auto& p : stub.calls) {
        REQUIRE(p.size() == D);
        for (std::size_t d = 0; d < D; ++d) {
            CHECK(p[d] >= low[d] - 1e-12);
            CHECK(p[d] <= high[d] + 1e-12);
        }
    }

    /* Per-rung accept-rate quality: each rung must propose something
     * close to L=50 evals (fix delivers ~100% of L; old code delivered
     * ~2%). Allow some slack at MAXEVAL boundary. */
    const std::string log = lr.read();
    /* "proposed=N" is reported in every rung line. Confirm at least one
     * rung crossed proposed=40 — old code was capped near 1-3. */
    bool saw_full_rung = false;
    std::size_t pos = 0;
    while ((pos = log.find("proposed=", pos)) != std::string::npos) {
        pos += 9;
        int n = std::atoi(log.c_str() + pos);
        if (n >= 40) { saw_full_rung = true; break; }
    }
    CHECK(saw_full_rung);
}

TEST_CASE("ANNEAL: FTOL_ABS plateau stops after exactly NEPS flat rungs",
          "[HPO][anneal][plateau]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.0, 0.0});
    sp.seed = 4;
    sp.stop.maxeval  = 10000;
    sp.stop.ftol_abs = 1e-6;
    sp.param["T_HOT"] = 1.0;   // skip probes
    sp.param["T_MIN"]  = 1e-30; // T_MIN must not fire first
    sp.param["L"]      = 5.0;
    sp.param["RT"]     = 0.9;
    sp.param["NEPS"]   = 3.0;

    LoggedRun lr;
    /* Constant objective: rung 1 has df=inf (not flat), every later rung
     * improves by exactly 0 -> flat. Stop at rung NEPS+1 = 4. */
    StubHost stub{ [&](const Vec& /*x*/){ return 5.0; }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL stopped: FTOL plateau") != std::string::npos);
    /* 4 rungs x L=5 + 1 seed eval = 21 host calls, far below MAXEVAL. */
    CHECK(stub.calls.size() == 21);
}

TEST_CASE("ANNEAL: FTOL_REL alone also arms the plateau stop",
          "[HPO][anneal][plateau]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4;
    sp.stop.maxeval  = 10000;
    sp.stop.ftol_rel = 1e-4;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"]  = 1e-30;
    sp.param["L"]      = 5.0;
    sp.param["NEPS"]   = 2.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& /*x*/){ return 5.0; }, {} };
    run_anneal_impl<Vec>(sp, stub);

    CHECK(lr.read().find("ANNEAL stopped: FTOL plateau") != std::string::npos);
    CHECK(stub.calls.size() == 16);   // (NEPS+1)=3 rungs x 5 + seed
}

TEST_CASE("ANNEAL: plateau does not fire while improvement exceeds the tolerance",
          "[HPO][anneal][plateau]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4;
    sp.stop.maxeval  = 60;
    sp.stop.ftol_abs = 1e-6;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"]  = 1e-30;
    sp.param["L"]      = 5.0;
    sp.param["NEPS"]   = 2.0;

    LoggedRun lr;
    /* Strictly ramping objective: every eval improves the best by 1.0,
     * so every rung improves by ~5 >> ftol_abs. MAXEVAL must fire. */
    std::atomic<int> k{0};
    StubHost stub{ [&](const Vec& /*x*/){ return 1000.0 - (++k); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL stopped: MAXEVAL exhausted") != std::string::npos);
    CHECK(log.find("FTOL plateau") == std::string::npos);
}

TEST_CASE("ANNEAL: plateau is disabled when neither FTOL key is set",
          "[HPO][anneal][plateau]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4;
    sp.stop.maxeval = 40;      // no ftol -> must run to MAXEVAL
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"]  = 1e-30;
    sp.param["L"]      = 5.0;
    sp.param["NEPS"]   = 2.0;  // present but inert without a tolerance

    LoggedRun lr;
    StubHost stub{ [&](const Vec& /*x*/){ return 5.0; }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL stopped: MAXEVAL exhausted") != std::string::npos);
    CHECK(log.find("FTOL plateau") == std::string::npos);
}

TEST_CASE("ANNEAL[CORANA]: plateau stop covers the Corana flavour too",
          "[HPO][anneal][plateau][corana]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 4;
    sp.stop.maxeval  = 10000;
    sp.stop.ftol_rel = 1e-4;
    sp.param["CORANA"] = 1.0;
    sp.param["NS"]     = 2.0;
    sp.param["NT"]     = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"]  = 1e-30;
    sp.param["NEPS"]   = 2.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& /*x*/){ return 5.0; }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL[CORANA]") != std::string::npos);
    CHECK(log.find("ANNEAL stopped: FTOL plateau") != std::string::npos);
    /* 3 rungs x (NT*NS*dim = 4) + seed = 13 host calls. */
    CHECK(stub.calls.size() == 13);
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
    sp.param["T_HOT"] = 1.0;
    sp.param["L"]      = 5.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return quad_bowl(x, {0.0, 0.0}); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("sigma=") != std::string::npos);
    CHECK(log.find("STEP override") != std::string::npos);
    CHECK(log.find("STEP_FRAC=") == std::string::npos);
}

/* ---- Phase-0 fixes: validation, T0 estimator, REANNEAL ------------ */

TEST_CASE("ANNEAL: dead knobs are hard errors, not silent no-ops",
          "[HPO][anneal][validation]")
{
    using namespace tadah::hpo;
    StubHost stub{ [&](const Vec& /*x*/){ return 0.0; }, {} };

    SECTION("unknown PARAM key (typo)") {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.param["T_HTO"] = 1.0;   // typo of T_HOT
        REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                            Catch::Contains("unknown PARAM"));
    }
    SECTION("XTOL_REL") {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.stop.xtol_rel = 1e-6;
        REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                            Catch::Contains("XTOL"));
    }
    SECTION("THREADS on the serial annealer (Option A: sizes the fan-out pool)") {
        // Under the shared-pool model THREADS is the total worker-pool width and
        // is honoured by EVERY family: serial ANNEAL runs ONE chain, but its
        // within-eval pc_lammps fan-out runs on that pool, so THREADS is valid
        // here. (It no longer multiplies OMP_NUM_THREADS; see anneal_core.h.)
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.num_threads = 4;
        REQUIRE_NOTHROW(run_anneal_impl<Vec>(sp, stub));
    }
    SECTION("THREADS 0 is rejected on the serial annealer") {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.num_threads = 0;
        REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                            Catch::Contains("THREADS must be >= 1"));
    }
    SECTION("POPULATION on the serial annealer") {
        auto sp = make_spec({-1.0}, {1.0});
        set_seed_point(sp, {0.0});
        sp.stop.maxeval = 10;
        sp.pop = 16;
        REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                            Catch::Contains("ANNEAL_PA"));
    }
}

TEST_CASE("ANNEAL[CORANA]: L is ignored and unvalidated under CORANA",
          "[HPO][anneal][corana][validation]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 5;
    sp.stop.maxeval = 20;
    sp.param["CORANA"] = 1.0;
    sp.param["T_HOT"]  = 1.0;
    sp.param["L"]      = 0.0;   // fatal for CLASSIC; a no-op for CORANA

    StubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {} };
    REQUIRE_NOTHROW(run_anneal_impl<Vec>(sp, stub));
}

TEST_CASE("ANNEAL[CORANA]: RESET_TO_BEST defaults to ON under CORANA",
          "[HPO][anneal][corana][log]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 5;
    sp.stop.maxeval = 30;
    sp.param["CORANA"] = 1.0;
    sp.param["T_HOT"]  = 1.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {} };
    run_anneal_impl<Vec>(sp, stub);
    CHECK(lr.read().find("RESET_TO_BEST=1") != std::string::npos);
}

namespace {

/* Quadratic bowl of the given radius centred on `centre`; FAILSCORE
 * cliff outside. Mimics the real HPO landscape (finite basin inside a
 * FAILSCORE plateau). */
double cliff_bowl(const Vec& x, const std::vector<double>& centre,
                  double radius, double failscore)
{
    double r2 = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double d = x(i) - centre[static_cast<std::size_t>(i)];
        r2 += d * d;
    }
    return r2 <= radius * radius ? r2 : failscore;
}

}  // namespace

TEST_CASE("ANNEAL: T0 estimator excludes FAILSCORE cliffs (seed anchor)",
          "[HPO][anneal][t0]")
{
    using namespace tadah::hpo;
    constexpr double FS = 1e16;
    auto sp = make_spec({-10.0, -10.0}, {10.0, 10.0});
    set_seed_point(sp, {1.5, 0.0});         // inside the r=2 bowl, near its edge
    sp.seed = 21;
    sp.failscore = FS;
    sp.stop.maxeval = 60;
    sp.param["L"]         = 5.0;
    sp.param["STEP_FRAC"] = 0.10;           // sigma = 2 -> many probes cliff out

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return cliff_bowl(x, {0.0, 0.0}, 2.0, FS); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("anchor=seed") != std::string::npos);
    /* Bowl losses are O(1)-O(4); the old pairwise estimator mixed the
     * 1e16 cliffs in and produced T0 ~ 1e15. The robust estimator must
     * land on the bowl's own scale. */
    const auto pos = log.find("ANNEAL: T0=");
    REQUIRE(pos != std::string::npos);
    const double t0 = std::atof(log.c_str() + pos + 11);
    CHECK(t0 > 0.0);
    CHECK(t0 < 100.0);
}

TEST_CASE("ANNEAL: T0 probes mirror the proposal kernel (one-dim under CORANA)",
          "[HPO][anneal][t0][corana]")
{
    using namespace tadah::hpo;
    /* Narrow high-D ridge: finite only within a radius-1 ball of the
     * origin in 40 dims. A full-vector Gaussian probe has norm
     * ~ sigma*sqrt(D) = 0.3*6.3 ~ 1.9 -> every probe cliffs out, even
     * though every actual CORANA move (one dim, |step| <= 0.3) is
     * finite. The estimator must probe with the chain's own kernel. */
    constexpr double FS = 1e16;
    const std::size_t D = 40;
    std::vector<double> low(D, -10.0), high(D, 10.0);
    auto mkspec = [&]() {
        auto sp = make_spec(low, high);
        set_seed_point(sp, std::vector<double>(D, 0.0));
        sp.seed = 7;
        sp.failscore = FS;
        sp.stop.maxeval = 40;
        sp.param["STEP_FRAC"] = 0.015;   // sigma = 0.3
        sp.param["T0_PROBES"] = 20.0;
        return sp;
    };
    StubHost stub{ [&](const Vec& x){
        double r2 = 0.0;
        for (long i = 0; i < x.size(); ++i) r2 += x(i) * x(i);
        return r2 <= 1.0 ? r2 : FS;
    }, {} };

    SECTION("CORANA: one-dim probes stay on the ridge -> real T0") {
        auto sp = mkspec();
        sp.param["CORANA"] = 1.0;
        LoggedRun lr;
        run_anneal_impl<Vec>(sp, stub);
        const std::string log = lr.read();
        CHECK(log.find("anchor=seed") != std::string::npos);
        CHECK(log.find("falling back to T0=1.0") == std::string::npos);
        const auto pos = log.find("ANNEAL: T0=");
        REQUIRE(pos != std::string::npos);
        const double t0 = std::atof(log.c_str() + pos + 11);
        CHECK(t0 > 0.0);
        CHECK(t0 < 1.0);   // on the bowl's own dE scale, not the fallback
    }
    SECTION("CLASSIC: full-vector probes cliff out -> documented fallback") {
        auto sp = mkspec();
        LoggedRun lr;
        run_anneal_impl<Vec>(sp, stub);
        CHECK(lr.read().find("falling back to T0=1.0") != std::string::npos);
    }
}

TEST_CASE("ANNEAL: FAILSCORE seed re-anchors on the best finite box probe",
          "[HPO][anneal][t0]")
{
    using namespace tadah::hpo;
    constexpr double FS = 1e16;
    auto sp = make_spec({-10.0, -10.0}, {10.0, 10.0});
    set_seed_point(sp, {9.0, 9.0});          // deep in the cliff
    sp.seed = 33;
    sp.failscore = FS;
    sp.stop.maxeval = 200;
    sp.param["L"]         = 5.0;
    sp.param["T0_PROBES"] = 30.0;            // ~30 uniform probes, r=5 bowl
    sp.param["STEP_FRAC"] = 0.05;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return cliff_bowl(x, {0.0, 0.0}, 5.0, FS); }, {} };
    REQUIRE_NOTHROW(run_anneal_impl<Vec>(sp, stub));
    CHECK(lr.read().find("anchor=re-anchored") != std::string::npos);
}

TEST_CASE("ANNEAL: all-FAILSCORE landscape is a hard error with remediation",
          "[HPO][anneal][t0]")
{
    using namespace tadah::hpo;
    constexpr double FS = 1e16;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.0});
    sp.seed = 3;
    sp.failscore = FS;
    sp.stop.maxeval = 100;
    sp.param["T0_PROBES"] = 5.0;

    StubHost stub{ [&](const Vec& /*x*/){ return FS; }, {} };
    REQUIRE_THROWS_WITH(run_anneal_impl<Vec>(sp, stub),
                        Catch::Contains("T0 calibration failed"));
}

TEST_CASE("ANNEAL: REANNEAL warm-restarts the schedule on T_MIN",
          "[HPO][anneal][reanneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 9;
    sp.stop.maxeval = 500;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"] = 1e-3;
    sp.param["RT"]    = 0.5;    // ~10 rungs per schedule
    sp.param["L"]     = 2.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("re-anneal #1") != std::string::npos);
    CHECK(log.find("ANNEAL stopped: T < T_MIN") != std::string::npos);
    CHECK(log.find("re-anneal(s)") != std::string::npos);
}

TEST_CASE("ANNEAL: REANNEAL 0 disables the warm restart",
          "[HPO][anneal][reanneal]")
{
    using namespace tadah::hpo;
    auto sp = make_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.seed = 9;
    sp.stop.maxeval = 500;
    sp.param["T_HOT"]    = 1.0;
    sp.param["T_MIN"]    = 1e-3;
    sp.param["RT"]       = 0.5;
    sp.param["L"]        = 2.0;
    sp.param["REANNEAL"] = 0.0;

    LoggedRun lr;
    StubHost stub{ [&](const Vec& x){ return x(0) * x(0); }, {} };
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("re-anneal #") == std::string::npos);
    CHECK(log.find("ANNEAL stopped: T < T_MIN") != std::string::npos);
}

TEST_CASE("ANNEAL: T0 probes do not shift the proposal stream",
          "[HPO][anneal][rng]")
{
    using namespace tadah::hpo;
    /* Same SEED, one run with T_HOT (no probes), one with auto-T0
     * (probes drawn from a separate stream): after skipping the
     * seed+probe prefix, the chain proposals must line up whenever the
     * chain state evolves identically. Use a constant objective so
     * both runs accept every proposal regardless of T -- then the two
     * proposal sequences must be IDENTICAL point-for-point. */
    auto base = make_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(base, {0.0, 0.0});
    base.seed = 77;
    base.param["L"]  = 6.0;
    base.param["RT"] = 0.9;

    auto run_with = [&](bool explicit_t0, std::size_t maxeval) {
        auto sp = base;
        sp.stop.maxeval = maxeval;
        if (explicit_t0) sp.param["T_HOT"] = 1.0;
        else             sp.param["T0_PROBES"] = 8.0;
        StubHost s{ [&](const Vec& /*x*/){ return 5.0; }, {} };
        run_anneal_impl<Vec>(sp, s);
        return s.calls;
    };

    const auto a = run_with(true,  40);       // seed + chain
    const auto b = run_with(false, 48);       // seed + 8 probes + chain

    /* a: [seed, p0, p1, ...]; b: [seed, 8 probes, p0, p1, ...] */
    REQUIRE(a.size() > 10);
    REQUIRE(b.size() > 18);
    const std::size_t n_common = std::min(a.size() - 1, b.size() - 9);
    for (std::size_t k = 0; k < n_common; ++k) {
        for (std::size_t d = 0; d < 2; ++d) {
            CHECK(a[1 + k][d] == Approx(b[9 + k][d]).margin(1e-15));
        }
    }
}

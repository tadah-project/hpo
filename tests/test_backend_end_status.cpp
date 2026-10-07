/*
 * Every optimiser driver records why its search ended (HPOLog::set_status),
 * in one vocabulary: "<LIB>:<ALGO>: <reason>" with a fixed reason text
 * (MAXEVAL reached, MAXTIME exceeded, STOPVAL reached, FTOL_ABS reached,
 * FTOL_REL reached, XTOL_REL reached, tol_gradinf reached, max_iter
 * reached, T_MIN reached, FTOL plateau, all grid points evaluated, no
 * optimisable dimensions), and the journal's stop line gives the numbers.
 * The run's trailer ("Optimiser status") is that status; a normal end
 * with none is an internal error (run_driver.h).
 *
 * DLIB:GFS, the ANNEAL family and the Ceres drivers had endings that set no
 * status, and the trailer then read COMPLETED: a spent MAXEVAL, the
 * annealers' every stop (STOPVAL and MAXTIME included), Ceres' every ending
 * but one (its convergence tests, its iteration cap, a spent MAXEVAL and a
 * genuine line-search FAILURE alike).
 *
 * The drivers run here on stub hosts through their HostT-generic
 * implementations (run_anneal_impl, run_anneal_pt_impl, run_anneal_pa_impl,
 * run_dlib_global_impl, detail::run_ceres_gradient_problem). The status
 * tests of the annealers' other stops are next to their stop-line tests
 * (test_anneal.cpp, test_anneal_pt.cpp, test_anneal_pa.cpp); DLIB:BFGS,
 * TADAH:RANDOM and TADAH:GRID, which need an HPO_Host, are tested end to
 * end in test_hpo_termination.cpp.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

namespace fs = std::filesystem;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;

struct NoRunner {};   // GradientProvider's Runner is only a type tag

/* An open journal; status() is the status the driver recorded. */
struct LoggedRun {
    fs::path path;
    LoggedRun() : path(fs::temp_directory_path() /
                       ("end_status_" +
                        std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()) +
                        ".log"))
    {
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~LoggedRun() {
        HPOLog::instance().finish("test-end");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string status() const { return HPOLog::instance().status(); }
    std::string read() const {
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

OptimizerSpec make_spec(const std::string& lib, const std::string& algo,
                        std::vector<double> low, std::vector<double> high,
                        std::vector<double> start)
{
    OptimizerSpec sp;
    sp.lib  = lib;
    sp.algo = algo;
    sp.low  = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.analyse_logscale();
    sp.optim_init = {std::move(start)};
    return sp;
}

/* Thread-safe counting stub over `f`; `sleep` before each evaluation. */
struct StubHost {
    std::function<double(const Vec&)> f;
    std::chrono::milliseconds sleep{0};
    mutable std::mutex m;
    mutable std::size_t calls = 0;
    double operator()(const Vec& x) const {
        if (sleep.count() > 0) std::this_thread::sleep_for(sleep);
        {
            std::lock_guard<std::mutex> lk(m);
            ++calls;
        }
        return f(x);
    }
    std::size_t n() const {
        std::lock_guard<std::mutex> lk(m);
        return calls;
    }
};

/* 1 + sum_i (x_i - 0.1 (i+1))^2: a smooth bowl, minimum 1 at (0.1, 0.2, ...). */
double bowl(const Vec& x)
{
    double s = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double d = x(i) - 0.1 * static_cast<double>(i + 1);
        s += d * d;
    }
    return 1.0 + s;
}

/* 1 + sum_i (x_i - 0.1 (i+1))^4: flat near its minimum, so a gradient
 * search converges slowly and a relative function tolerance ends it. */
double quartic(const Vec& x)
{
    double s = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double d = x(i) - 0.1 * static_cast<double>(i + 1);
        s += d * d * d * d;
    }
    return 1.0 + s;
}

}  // namespace

/* ======================================================================
 * TADAH:ANNEAL
 * ==================================================================== */

/* Pins: a MAXTIME that passes during a rung ends the search with the stop
 * line "ANNEAL stopped: MAXTIME exceeded" and the status
 * "TADAH:ANNEAL: MAXTIME exceeded", after fewer evaluations than MAXEVAL.
 * Before the fix: the stop line, and no status (the trailer said
 * COMPLETED). */
TEST_CASE("End status: TADAH:ANNEAL records a MAXTIME stop",
          "[HPO][stop_status][anneal]")
{
    auto sp = make_spec("TADAH", "ANNEAL", {-1.0}, {1.0}, {0.5});
    sp.seed = 3;
    sp.stop.maxeval = 1000;
    sp.stop.maxtime = 0.05;
    sp.param["T_INIT"] = 1.0;          // no T0 probes
    sp.param["CHAIN_LENGTH"] = 100.0;  // MAXTIME passes inside the first rung

    LoggedRun lr;
    StubHost host{bowl, std::chrono::milliseconds(20)};
    tadah::hpo::run_anneal_impl<Vec>(sp, host);
    CHECK(host.n() < 1000u);
    CHECK(lr.read().find("ANNEAL stopped: MAXTIME exceeded") !=
          std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL: MAXTIME exceeded");
}

/* The rung loop of serial ANNEAL tested MAXTIME after each rung and again
 * in its loop condition. A deadline that passed between the two left the
 * loop with no stop reason: "ANNEAL stopped: loop completed normally" and
 * no status. The scripted clock passes the deadline right after the
 * post-rung check of the first rung (the seed and one rung of
 * CHAIN_LENGTH 3 evaluated, then one more clock reading).
 * Pins: the search ends after those 4 evaluations, on MAXTIME: the stop
 * line "ANNEAL stopped: MAXTIME exceeded" and the status
 * "TADAH:ANNEAL: MAXTIME exceeded"; no "loop completed normally".
 * Before the fix: the same 4 evaluations, "ANNEAL stopped: loop completed
 * normally" and no status. */
TEST_CASE("End status: TADAH:ANNEAL records a MAXTIME that passes between "
          "two of its checks", "[HPO][stop_status][anneal]")
{
    constexpr std::size_t kChain = 3;
    auto sp = make_spec("TADAH", "ANNEAL", {-1.0}, {1.0}, {0.5});
    sp.seed = 3;
    sp.stop.maxeval = 1000;
    sp.stop.maxtime = 10.0;
    sp.param["T_INIT"] = 1.0;
    sp.param["CHAIN_LENGTH"] = static_cast<double>(kChain);

    LoggedRun lr;
    StubHost host{bowl};
    std::size_t readings_after_rung = 0;
    const tadah::hpo::AnnealElapsedSeconds clock = [&]() -> double {
        if (host.n() < 1 + kChain) return 0.0;   // the seed and rung 1
        return readings_after_rung++ == 0 ? 0.0 : 1e9;
    };
    tadah::hpo::run_anneal_impl<Vec>(sp, host, clock);
    CHECK(host.n() == 1 + kChain);
    const std::string journal = lr.read();
    CHECK(journal.find("ANNEAL stopped: MAXTIME exceeded") !=
          std::string::npos);
    CHECK(journal.find("loop completed normally") == std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL: MAXTIME exceeded");
}

/* With REANNEAL (the default) a schedule that cools below T_MIN with
 * budget left re-anneals from the best point. A MAXTIME that passed after
 * the schedule's last post-rung check, before the re-anneal test, ended the
 * search without a re-anneal and left its stop "T < T_MIN". The schedule:
 * T_INIT 1, RT 0.2, T_MIN 0.03, CHAIN_LENGTH 3 -- rungs at T = 1, 0.2 and
 * 0.04, then 0.008 < T_MIN -- and the first re-anneal would start at
 * T_INIT * REANNEAL_SCALE = 0.25, above T_MIN / RT = 0.15. The scripted
 * clock passes the deadline right after the post-rung check of the third
 * rung (the seed and three rungs evaluated, then one more clock reading).
 * Pins: the search ends after those 10 evaluations without a re-anneal,
 * on MAXTIME: the stop line "ANNEAL stopped: MAXTIME exceeded" and the
 * status "TADAH:ANNEAL: MAXTIME exceeded". Without the deadline the same
 * search re-anneals ("ANNEAL re-anneal #1: T_start=2.500e-01") and
 * MAXEVAL 13 ends it (a pin: as before the fix).
 * Before the fix: the same 10 evaluations, "ANNEAL stopped: T < T_MIN"
 * and the status "TADAH:ANNEAL: T_MIN reached". */
TEST_CASE("End status: TADAH:ANNEAL records the MAXTIME that prevented a "
          "re-anneal, not T_MIN", "[HPO][stop_status][anneal]")
{
    constexpr std::size_t kChain = 3;
    auto sp = make_spec("TADAH", "ANNEAL", {-1.0}, {1.0}, {0.5});
    sp.seed = 3;
    sp.stop.maxeval = 13;
    sp.stop.maxtime = 10.0;
    sp.param["T_INIT"] = 1.0;
    sp.param["T_MIN"] = 0.03;
    sp.param["RT"] = 0.2;
    sp.param["CHAIN_LENGTH"] = static_cast<double>(kChain);

    SECTION("the deadline passes before the re-anneal") {
        LoggedRun lr;
        StubHost host{bowl};
        std::size_t readings_after_schedule = 0;
        const tadah::hpo::AnnealElapsedSeconds clock = [&]() -> double {
            if (host.n() < 1 + 3 * kChain) return 0.0;  // seed, three rungs
            return readings_after_schedule++ == 0 ? 0.0 : 1e9;
        };
        tadah::hpo::run_anneal_impl<Vec>(sp, host, clock);
        CHECK(host.n() == 1 + 3 * kChain);
        const std::string journal = lr.read();
        CHECK(journal.find("ANNEAL re-anneal #") == std::string::npos);
        CHECK(journal.find("ANNEAL stopped: MAXTIME exceeded after 10 host "
                           "eval(s) (3 rung(s) completed)") !=
              std::string::npos);
        CHECK(journal.find("T < T_MIN") == std::string::npos);
        CHECK(lr.status() == "TADAH:ANNEAL: MAXTIME exceeded");
    }
    SECTION("without the deadline it re-anneals") {
        LoggedRun lr;
        StubHost host{bowl};
        const tadah::hpo::AnnealElapsedSeconds clock = [] { return 0.0; };
        tadah::hpo::run_anneal_impl<Vec>(sp, host, clock);
        CHECK(host.n() == 13u);
        const std::string journal = lr.read();
        CHECK(journal.find("ANNEAL re-anneal #1: T_start=2.500e-01") !=
              std::string::npos);
        CHECK(lr.status() == "TADAH:ANNEAL: MAXEVAL reached");
    }
}

/* ======================================================================
 * TADAH:ANNEAL_PT and TADAH:ANNEAL_PA
 * ==================================================================== */

/* Pins: a MAXTIME that passes during an epoch ends ANNEAL_PT with the stop
 * line "ANNEAL_PT stopped: MAXTIME exceeded" and the status
 * "TADAH:ANNEAL_PT: MAXTIME exceeded".
 * Before the fix: the stop line, and no status. */
TEST_CASE("End status: TADAH:ANNEAL_PT records a MAXTIME stop",
          "[HPO][stop_status][anneal_pt]")
{
    auto sp = make_spec("TADAH", "ANNEAL_PT", {-1.0}, {1.0}, {0.5});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.seed = 5;
    sp.num_threads = 1;
    sp.stop.maxeval = 100000;
    sp.stop.maxtime = 0.1;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;

    LoggedRun lr;
    StubHost host{bowl, std::chrono::milliseconds(5)};
    tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
    CHECK(host.n() < 100000u);
    CHECK(lr.read().find("ANNEAL_PT stopped: MAXTIME exceeded") !=
          std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL_PT: MAXTIME exceeded");
}

/* Pins: a loss at or below STOPVAL among the T0 calibration probes ends
 * ANNEAL_PT with the stop line "ANNEAL_PT stopped: STOPVAL reached during
 * T0 probes" and the status "TADAH:ANNEAL_PT: STOPVAL reached". The seed
 * (loss 1.25) is above STOPVAL 1.2, the bowl's minimum 1 below it.
 * Before the fix: the stop line, and no status. */
TEST_CASE("End status: TADAH:ANNEAL_PT records a STOPVAL reached during the "
          "T0 probes", "[HPO][stop_status][anneal_pt]")
{
    auto sp = make_spec("TADAH", "ANNEAL_PT", {-1.0}, {1.0}, {0.6});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.seed = 5;
    sp.num_threads = 1;
    sp.stop.maxeval = 5000;
    sp.stop.stopval = 1.2;
    sp.param["REPLICAS"] = 2.0;

    LoggedRun lr;
    StubHost host{bowl};
    tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
    CHECK(lr.read().find("ANNEAL_PT stopped: STOPVAL reached during T0 "
                         "probes") != std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL_PT: STOPVAL reached");
}

/* Pins: a population that reaches STOPVAL ends ANNEAL_PA with the stop
 * line "ANNEAL_PA stopped: STOPVAL reached" and the status
 * "TADAH:ANNEAL_PA: STOPVAL reached".
 * Before the fix: the stop line, and no status. */
TEST_CASE("End status: TADAH:ANNEAL_PA records a STOPVAL stop",
          "[HPO][stop_status][anneal_pa]")
{
    auto sp = make_spec("TADAH", "ANNEAL_PA", {-1.0}, {1.0}, {0.9});
    sp.seed = 7;
    sp.num_threads = 1;
    sp.pop = 16;
    sp.stop.maxeval = 100000;
    sp.stop.stopval = 1.0 + 1e-4;
    sp.param["T_HOT"] = 0.5;
    sp.param["CHAIN_LENGTH"] = 2.0;

    LoggedRun lr;
    StubHost host{bowl};
    tadah::hpo::run_anneal_pa_impl<Vec>(sp, host);
    CHECK(host.n() < 100000u);
    CHECK(lr.read().find("ANNEAL_PA stopped: STOPVAL reached") !=
          std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL_PA: STOPVAL reached");
}

/* Pins: a population best that stalls ends ANNEAL_PA on the Corana
 * eps-NEPS plateau rule, with the stop line "ANNEAL_PA stopped: FTOL
 * plateau" and the status "TADAH:ANNEAL_PA: FTOL plateau" (a constant
 * objective: every temperature step is flat).
 * Before the fix: the stop line, and no status. */
TEST_CASE("End status: TADAH:ANNEAL_PA records an FTOL plateau",
          "[HPO][stop_status][anneal_pa]")
{
    auto sp = make_spec("TADAH", "ANNEAL_PA", {-1.0}, {1.0}, {0.0});
    sp.seed = 9;
    sp.num_threads = 1;
    sp.pop = 16;
    sp.stop.maxeval = 100000;
    sp.stop.ftol_abs = 1e-6;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_MIN"] = 1e-12;
    sp.param["CHAIN_LENGTH"] = 1.0;

    LoggedRun lr;
    StubHost host{[](const Vec&) { return 1.0; }};
    tadah::hpo::run_anneal_pa_impl<Vec>(sp, host);
    CHECK(host.n() < 100000u);
    CHECK(lr.read().find("ANNEAL_PA stopped: FTOL plateau") !=
          std::string::npos);
    CHECK(lr.status() == "TADAH:ANNEAL_PA: FTOL plateau");
}

/* ======================================================================
 * DLIB:GFS
 * ==================================================================== */

/* Pins: a DLIB:GFS search that spends its MAXEVAL evaluates exactly
 * MAXEVAL points and ends with the stop line "DLIB:GFS stopped: MAXEVAL
 * reached after 6 evaluation(s)" and the status "DLIB:GFS: MAXEVAL
 * reached".
 * Before the fix: the stop line, and no status (STOPVAL and MAXTIME had
 * theirs). */
TEST_CASE("End status: DLIB:GFS records a spent MAXEVAL",
          "[HPO][stop_status][gfs]")
{
    auto sp = make_spec("DLIB", "GFS", {-1.0, -1.0}, {1.0, 1.0}, {0.5, -0.3});
    sp.stop.maxeval = 6;
    sp.num_threads = 1;

    LoggedRun lr;
    StubHost host{bowl};
    tadah::hpo::run_dlib_global_impl(sp, host);
    CHECK(host.n() == 6u);
    CHECK(lr.read().find("DLIB:GFS stopped: MAXEVAL reached after 6 "
                         "evaluation(s)") != std::string::npos);
    CHECK(lr.status() == "DLIB:GFS: MAXEVAL reached");
}

/* dlib takes its run time as a count of nanoseconds, which holds about
 * 292 years (9.2e9 s); MAXTIME takes any positive number of seconds.
 * Pins: with MAXTIME 1e10 s and MAXEVAL 6 the search is not cut short:
 * dlib runs for at most its own FOREVER (200 years), the search makes its
 * 6 evaluations and records "DLIB:GFS: MAXEVAL reached".
 * Before the fix: the run time MAXTIME left overflowed its conversion to
 * nanoseconds (out of range: undefined, a negative count on x86-64), so
 * dlib's deadline had passed when it started: the search ended after its
 * seed, 1 of 6 evaluations, with "DLIB:GFS: MAXTIME exceeded". */
TEST_CASE("End status: DLIB:GFS with a MAXTIME longer than dlib's run time "
          "can hold spends its MAXEVAL", "[HPO][stop_status][gfs]")
{
    auto sp = make_spec("DLIB", "GFS", {-1.0, -1.0}, {1.0, 1.0}, {0.5, -0.3});
    sp.stop.maxeval = 6;
    sp.stop.maxtime = 1e10;
    sp.num_threads = 1;

    LoggedRun lr;
    StubHost host{bowl};
    tadah::hpo::run_dlib_global_impl(sp, host);
    CHECK(host.n() == 6u);
    CHECK(lr.status() == "DLIB:GFS: MAXEVAL reached");
}

/* ======================================================================
 * CERES (GradientProblemSolver). Ceres reports its own ending
 * (CONVERGENCE, NO_CONVERGENCE, FAILURE) and a message; MAXEVAL is HPO's:
 * an evaluation refused for the budget is a failed one to Ceres, which
 * then ends with FAILURE. The status is Ceres' own reason when Ceres
 * converged or ran out of iterations, MAXEVAL when the budget made it fail,
 * and "STOPPED -- CERES:<ALGO>: FAILURE (<head>)" for any other failure,
 * the head of Ceres' message without its numbers (the stop line gives the
 * whole message).
 * The cases pin the three CONVERGENCE message prefixes of the Ceres
 * version HPO builds ("Function tolerance reached", "Parameter tolerance
 * reached", "Gradient tolerance reached") and "Maximum number of
 * iterations reached.": a Ceres upgrade that rewords them fails here.
 * Every case used to end with no status (COMPLETED): a spent MAXEVAL and a
 * genuine line-search failure as much as convergence.
 * ==================================================================== */

#ifdef TADAH_USE_CERES
namespace {

struct CeresCase {
    std::string algo;
    ceres::LineSearchDirectionType dir;
    ceres::LineSearchType ls;
};

CeresCase ceres_case(const std::string& algo)
{
    if (algo == "LBFGS")  return {algo, ceres::LBFGS, ceres::WOLFE};
    if (algo == "WOLFE")  return {algo, ceres::BFGS, ceres::WOLFE};
    if (algo == "NLCG")   return {algo, ceres::NONLINEAR_CONJUGATE_GRADIENT,
                                  ceres::WOLFE};
    return {"STEEPEST_DESCENT", ceres::STEEPEST_DESCENT, ceres::ARMIJO};
}

/* Runs CERES:<algo> on `host` and returns the journal. */
template<class Host>
std::string run_ceres(const OptimizerSpec& sp, Host& host, LoggedRun& lr)
{
    const CeresCase c = ceres_case(sp.algo);
    tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
        sp, host, c.dir, c.ls, c.algo);
    return lr.read();
}

}  // namespace

/* Pins: MAXEVAL 7 under STEEPEST_DESCENT on the 2-dimensional bowl (each
 * evaluation with its gradient costs the loss and a 4-point central
 * stencil): the second evaluation's gradient is refused for the budget,
 * Ceres ends with FAILURE, and the status is "CERES:STEEPEST_DESCENT:
 * MAXEVAL reached", with the stop line "CERES:STEEPEST_DESCENT stopped:
 * MAXEVAL reached after 7 evaluation(s)"; the host made exactly 7 calls.
 * Before the fix: no status (the trailer said COMPLETED). */
TEST_CASE("End status: CERES records a spent MAXEVAL as MAXEVAL reached",
          "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "STEEPEST_DESCENT", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 7;
    LoggedRun lr;
    StubHost host{bowl};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() == 7u);
    CHECK(journal.find("CERES:STEEPEST_DESCENT stopped: MAXEVAL reached "
                       "after 7 evaluation(s)") != std::string::npos);
    CHECK(lr.status() == "CERES:STEEPEST_DESCENT: MAXEVAL reached");
}

/* Pins: MAXEVAL 1 under LBFGS: the loss at the start is the only
 * evaluation, its gradient is refused for the budget ("Initial cost and
 * jacobian evaluation failed."), and the status is "CERES:LBFGS: MAXEVAL
 * reached".
 * Before the fix: no status. */
TEST_CASE("End status: CERES records MAXEVAL spent at its first gradient",
          "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "LBFGS", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 1;
    LoggedRun lr;
    StubHost host{bowl};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() == 1u);
    CHECK(journal.find("Initial cost and jacobian evaluation failed.") !=
          std::string::npos);
    CHECK(lr.status() == "CERES:LBFGS: MAXEVAL reached");
}

/* Pins: PARAM tol_gradinf 1e3 (Ceres' gradient_tolerance, a max-norm)
 * holds at the start: Ceres converges with "Gradient tolerance reached"
 * after the loss and its 4-point stencil, and the status is "CERES:LBFGS:
 * tol_gradinf reached"; the stop line gives the value, "PARAM tol_gradinf
 * = 1000".
 * Before the fix: no status. */
TEST_CASE("End status: CERES records its gradient tolerance as tol_gradinf "
          "reached", "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "LBFGS", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 200;
    sp.param["tol_gradinf"] = 1e3;
    LoggedRun lr;
    StubHost host{bowl};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() == 5u);
    CHECK(journal.find("CERES:LBFGS stopped: tol_gradinf reached after 5 "
                       "evaluation(s): PARAM tol_gradinf = 1000; Ceres: "
                       "Gradient tolerance reached.") != std::string::npos);
    CHECK(lr.status() == "CERES:LBFGS: tol_gradinf reached");
}

/* Pins: Ceres' function_tolerance (FTOL_REL, by default 1e-6) on the
 * quartic, whose loss falls slowly near its minimum: Ceres converges with
 * "Function tolerance reached", the status is "CERES:LBFGS: FTOL_REL
 * reached", and the stop line says the value is the default, "FTOL_REL =
 * 1e-06 (default)".
 * Before the fix: no status. */
TEST_CASE("End status: CERES records its function tolerance as FTOL_REL "
          "reached", "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "LBFGS", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 500;
    LoggedRun lr;
    StubHost host{quartic};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() < 500u);
    CHECK(journal.find("CERES:LBFGS stopped: FTOL_REL reached after ") !=
          std::string::npos);
    CHECK(journal.find(": FTOL_REL = 1e-06 (default); Ceres: Function "
                       "tolerance reached.") != std::string::npos);
    CHECK(lr.status() == "CERES:LBFGS: FTOL_REL reached");
}

/* Pins: XTOL_REL 1e-2 (Ceres' parameter_tolerance) on the quartic, with
 * FTOL_REL and PARAM tol_gradinf too small to fire: Ceres converges with
 * "Parameter tolerance reached", the status is "CERES:LBFGS: XTOL_REL
 * reached" and the stop line gives "XTOL_REL = 0.01". (On the bowl LBFGS
 * lands on the minimum, where the gradient test fires first.)
 * Before the fix: no status. */
TEST_CASE("End status: CERES records its parameter tolerance as XTOL_REL "
          "reached", "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "LBFGS", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 2000;
    sp.stop.xtol_rel = 1e-2;
    sp.stop.ftol_rel = 1e-300;
    sp.param["tol_gradinf"] = 1e-300;
    LoggedRun lr;
    StubHost host{quartic};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() < 2000u);
    CHECK(journal.find(": XTOL_REL = 0.01; Ceres: Parameter tolerance "
                       "reached.") != std::string::npos);
    CHECK(lr.status() == "CERES:LBFGS: XTOL_REL reached");
}

/* Pins: PARAM max_iter 1 (Ceres' max_num_iterations) on the quartic,
 * which no tolerance ends in one step: Ceres ends with NO_CONVERGENCE
 * "Maximum number of iterations reached.", and the status is
 * "CERES:STEEPEST_DESCENT: max_iter reached", the stop line "PARAM
 * max_iter = 1". (On the bowl the first step lands on the minimum, where
 * the gradient test fires first.)
 * Before the fix: no status. */
TEST_CASE("End status: CERES records its iteration cap as max_iter reached",
          "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "STEEPEST_DESCENT", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 200;
    sp.param["max_iter"] = 1.0;
    LoggedRun lr;
    StubHost host{quartic};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() < 200u);
    CHECK(journal.find(": PARAM max_iter = 1; Ceres: Maximum number of "
                       "iterations reached.") != std::string::npos);
    CHECK(lr.status() == "CERES:STEEPEST_DESCENT: max_iter reached");
}

/* A genuine FAILURE: on the steep bowl 1 + 100 (x - 0.1)^2 from 0.5, the
 * Armijo line search of STEEPEST_DESCENT may try one step size only
 * (PARAM max_ls_iter 1), and its first one, 1/|g| = 0.0125, overshoots to
 * -0.5, where the loss is higher: no valid step, "Numerical failure in line
 * search, failed to find a valid step size, (did not run out of
 * iterations) using initial_step_size: 1.25000e-02, initial_cost: ...",
 * with most of MAXEVAL 200 left.
 * Pins: the search ends after a handful of evaluations with the status
 * "STOPPED -- CERES:STEEPEST_DESCENT: FAILURE (Numerical failure in line
 * search)" -- the head of Ceres' message, a fixed text without its numbers,
 * the class of NLopt's "STOPPED -- nlopt roundoff-limited" -- while the
 * stop line "CERES:STEEPEST_DESCENT stopped: FAILURE after N
 * evaluation(s); Ceres: <message>" and the WARNING give the whole message.
 * Before the fix: the status quoted the whole message, the step size, cost
 * and gradient of the run included ("... FAILURE (Numerical failure in
 * line search, failed to find a valid step size, ... initial_gradient:
 * -6.40000e+03.)"); before 1.6.0's statuses, no status. */
TEST_CASE("End status: CERES records a line-search failure as STOPPED with "
          "Ceres' message", "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "STEEPEST_DESCENT", {-1.0}, {1.0}, {0.5});
    sp.stop.maxeval = 200;
    sp.param["max_ls_iter"] = 1.0;
    LoggedRun lr;
    StubHost host{[](const Vec& x) {
        const double d = x(0) - 0.1;
        return 1.0 + 100.0 * d * d;
    }};
    const std::string journal = run_ceres(sp, host, lr);
    CHECK(host.n() < 20u);
    CHECK(lr.status() == "STOPPED -- CERES:STEEPEST_DESCENT: FAILURE "
                         "(Numerical failure in line search)");
    CHECK(journal.find("CERES:STEEPEST_DESCENT stopped: FAILURE after ") !=
          std::string::npos);
    CHECK(journal.find(" evaluation(s); Ceres: Numerical failure in line "
                       "search, failed to find a valid step size, (did not "
                       "run out of iterations) using initial_step_size: ") !=
          std::string::npos);
    CHECK(journal.find("CERES:STEEPEST_DESCENT -- non-convergent: Numerical "
                       "failure in line search, failed to find a valid step "
                       "size") != std::string::npos);
}

/* Ceres checks its options before it evaluates anything and, when it
 * refuses them, ends the solve with FAILURE "Invalid configuration. ...".
 * The validators refuse every setting of HPO's that Ceres would
 * (test_hpo.cpp), so options Ceres refuses are HPO's error. The driver is
 * called here without its validator (PARAM max_ls_iter 0).
 * Pins: the driver throws std::logic_error "internal error: CERES:LBFGS:
 * Ceres refuses the options built from a validated spec: Invalid
 * configuration. ..." before any evaluation, and records no status.
 * Before the fix: Ceres' FAILURE was the search's ending, "STOPPED --
 * CERES:LBFGS: FAILURE (Invalid configuration)", as if the search had
 * stopped. */
TEST_CASE("End status: CERES options that Ceres refuses are an internal "
          "error, not an ending", "[HPO][stop_status][ceres]")
{
    auto sp = make_spec("CERES", "LBFGS", {-1.0, -1.0}, {1.0, 1.0},
                        {0.5, -0.3});
    sp.stop.maxeval = 50;
    sp.param["max_ls_iter"] = 0.0;
    LoggedRun lr;
    StubHost host{bowl};
    CHECK_THROWS_MATCHES(
        run_ceres(sp, host, lr), std::logic_error,
        Catch::Matchers::Predicate<std::logic_error>(
            [](const std::logic_error& e) {
                const std::string w = e.what();
                return w.rfind("internal error: CERES:LBFGS: Ceres refuses "
                               "the options built from a validated spec: "
                               "Invalid configuration.", 0) == 0;
            },
            "an internal error naming Ceres' refusal"));
    CHECK(host.n() == 0u);
    CHECK(lr.status().empty());
}
#endif  // TADAH_USE_CERES

/* Pins: the head of a Ceres termination message, which the status of an
 * ending that Ceres names itself quotes (detail::ceres_message_head): the
 * text before the message's first ". ", ", " or ": ", or before its closing
 * ".". For every termination message of the line-search minimizer and the
 * option check of the Ceres version HPO builds (2.2; the numbers are
 * examples of what Ceres prints) it is a fixed text without a number. A
 * text without such a stop is its own head.
 * Before the fix: does not compile (no ceres_message_head; the status
 * quoted the whole message). */
TEST_CASE("End status: the head of a Ceres message names its ending without "
          "its numbers", "[HPO][stop_status][ceres]")
{
    using tadah::hpo::detail::ceres_message_head;
    const std::pair<std::string, std::string> cases[] = {
        {"Numerical failure in line search, failed to find a valid step "
         "size, (did not run out of iterations) using initial_step_size: "
         "1.25000e-02, initial_cost: 1.70000e+01, initial_gradient: "
         "-6.40000e+03.",
         "Numerical failure in line search"},
        {"Numerical failure in line search, initial_step_size is negative: "
         "-2.00000e-01, directional_derivative: 1.00000e+00, (current_cost "
         "- previous_cost): -1.00000e-01",
         "Numerical failure in line search"},
        {"Initial cost and jacobian evaluation failed.",
         "Initial cost and jacobian evaluation failed"},
        {"Initial cost and jacobian evaluation failed. More details: "
         "projected_gradient_step = Plus(x, -gradient) failed.",
         "Initial cost and jacobian evaluation failed"},
        {"Cost and jacobian evaluation failed.",
         "Cost and jacobian evaluation failed"},
        {"Line search direction failure: specified "
         "max_num_line_search_direction_restarts: 5 reached.",
         "Line search direction failure"},
        {"Step failed to evaluate. This should not happen as the step was "
         "valid when it was selected by the line search. More details: "
         "projected_gradient_step = Plus(x, -gradient) failed.",
         "Step failed to evaluate"},
        {"Invalid configuration. Solver::Options::max_num_iterations = -1. "
         "Violated constraint: Solver::Options::max_num_iterations >= 0",
         "Invalid configuration"},
        {"Function tolerance reached. |cost_change|/cost: 9.876543e-07 <= "
         "1.000000e-06",
         "Function tolerance reached"},
        {"Parameter tolerance reached. Relative step_norm: 4.567890e-09 <= "
         "1.000000e-08.",
         "Parameter tolerance reached"},
        {"Gradient tolerance reached. Gradient max norm: 3.000000e-11 <= "
         "1.000000e-10",
         "Gradient tolerance reached"},
        {"Maximum number of iterations reached.",
         "Maximum number of iterations reached"},
        {"Maximum solver time reached.", "Maximum solver time reached"},
        {"User callback returned SOLVER_ABORT.",
         "User callback returned SOLVER_ABORT"},
        {"ceres::GradientProblemSolve was not called.",
         "ceres::GradientProblemSolve was not called"},
        {"a text without a stop", "a text without a stop"},
        {"", ""},
    };
    for (const auto& c : cases) {
        INFO("message: " << c.first);
        CHECK(ceres_message_head(c.first) == c.second);
        CHECK(c.second.find_first_of("0123456789") == std::string::npos);
    }
}

/* ======================================================================
 * The ALGO help of DLIB:BFGS
 * ==================================================================== */

/* Pins: the ALGO help says what each DLIB:BFGS stop tests: FTOL_ABS the
 * change of the loss from one iteration to the next, PARAM tol_gradinf the
 * L2 norm of the gradient (dlib's gradient_norm_stop_strategy tests
 * length(g)).
 * Before the fix: "gradient-inf-norm stop", which is not what dlib tests. */
TEST_CASE("End status: the DLIB:BFGS help says what its stops test",
          "[HPO][stop_status][caps]")
{
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    OptimizerSpec::print_caps("DLIB", "BFGS");
    std::cout.rdbuf(old);
    const std::string s = captured.str();
    CHECK(s.find("   * FTOL_ABS     (objective-delta stop, default 1e-7: the "
                 "loss\n                   changed by less from one iteration "
                 "to the next)\n") != std::string::npos);
    CHECK(s.find("   * PARAM tol_gradinf <double>  (stop once the L2 norm of "
                 "the gradient\n                   falls below it; alternative "
                 "to FTOL_ABS)\n") != std::string::npos);
    CHECK(s.find("inf-norm") == std::string::npos);
}

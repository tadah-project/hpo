/*
 * Host exceptions must cross the optimiser adapters unchanged.
 *
 * NLopt's C++ wrapper maps every exception thrown by the objective to
 * NLOPT_FAILURE, i.e. runtime_error("nlopt failure"), which the adapter
 * used to classify as a graceful self-termination: a fatal best-potential
 * persistence failure finished the run with exit code 0, RESTART never saw
 * StagnationStop, and SIGINT became "STOPPED -- nlopt failure". The
 * adapters are driven here with a stub host through their HostT-generic
 * implementations (run_nlopt_impl, detail::run_ceres_gradient_problem).
 */

#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/run_policy.h>

#include <dlib/matrix.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

namespace fs = std::filesystem;
using tadah::hpo::NloptOutcome;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::classify_nlopt_exception;

struct NoRunner {};   // GradientProvider's Runner is only a type tag

/* Counts calls; throws `thrower()` on call number `throw_at` (1-based, 0 =
 * never). The objective is a smooth bowl so every algorithm makes normal
 * progress until the throw. */
struct StubHost {
    std::size_t throw_at = 0;
    std::function<void()> thrower;
    std::size_t calls = 0;
    double operator()(const Vec& x) {
        ++calls;
        if (throw_at != 0 && calls == throw_at) thrower();
        double s = 0.0;
        for (long i = 0; i < x.size(); ++i)
            s += (x(i) - 0.1 * static_cast<double>(i + 1)) *
                 (x(i) - 0.1 * static_cast<double>(i + 1));
        return 1.0 + s;
    }
};

OptimizerSpec make_spec(const std::string& lib, const std::string& algo)
{
    OptimizerSpec sp;
    sp.lib  = lib;
    sp.algo = algo;
    sp.low  = {-1.0, -1.0};
    sp.high = { 1.0,  1.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    sp.analyse_logscale();
    sp.optim_init = {{0.5, -0.3}};
    sp.stop.maxeval = 200;
    sp.seed = 7;
    return sp;
}

/* An open journal so the adapter's status/warnings can be inspected. */
struct LoggedRun {
    fs::path path;
    LoggedRun() : path(fs::temp_directory_path() /
                       ("nlopt_host_exc_" +
                        std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()) +
                        ".log"))
    {
        tadah::hpo::HPOLog::instance().reset_for_test();
        tadah::hpo::HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~LoggedRun() {
        tadah::hpo::HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    /* Close the journal (status = last set_status) and return it. */
    std::string finish_and_read() const {
        tadah::hpo::HPOLog::instance().finish();
        std::ifstream is(path);
        std::ostringstream oss; oss << is.rdbuf();
        return oss.str();
    }
};

std::string status_line(const std::string& journal)
{
    std::istringstream in(journal);
    std::string line;
    while (std::getline(in, line))
        if (line.find("Optimiser status") != std::string::npos) return line;
    return {};
}

template<class E>
std::exception_ptr make_eptr(const E& e)
{
    return std::make_exception_ptr(e);
}

}  // namespace

TEST_CASE("classify_nlopt_exception decides by type, not by message",
          "[HPO][nlopt][failure_handling]")
{
    const char* no_msg = nullptr;
    const char* simplex = "starting step size led to simplex that was too "
                          "small in dimension 0: 1e-16 is too close to x=0.5";

    /* Algorithmic self-termination: the best result stands. */
    CHECK(classify_nlopt_exception(make_eptr(nlopt::roundoff_limited()), no_msg)
          == NloptOutcome::Graceful);
    CHECK(classify_nlopt_exception(make_eptr(std::runtime_error("nlopt failure")),
                                   no_msg) == NloptOutcome::Graceful);
    CHECK(classify_nlopt_exception(make_eptr(std::runtime_error("nlopt failure")),
                                   "") == NloptOutcome::Graceful);
    CHECK(classify_nlopt_exception(make_eptr(std::runtime_error(simplex)), simplex)
          == NloptOutcome::Graceful);

    /* A rejected setup. */
    CHECK(classify_nlopt_exception(make_eptr(std::invalid_argument("x")), no_msg)
          == NloptOutcome::InvalidArgs);

    /* Everything else fails the run. */
    CHECK(classify_nlopt_exception(make_eptr(std::runtime_error(simplex)), no_msg)
          == NloptOutcome::Failed);
    CHECK(classify_nlopt_exception(make_eptr(std::runtime_error("other")), simplex)
          == NloptOutcome::Failed);
    CHECK(classify_nlopt_exception(make_eptr(nlopt::forced_stop()), no_msg)
          == NloptOutcome::Failed);
    CHECK(classify_nlopt_exception(make_eptr(std::bad_alloc()), no_msg)
          == NloptOutcome::Failed);
    CHECK(classify_nlopt_exception(make_eptr(std::logic_error("nlopt failure")),
                                   no_msg) == NloptOutcome::Failed);
    /* A runtime_error SUBCLASS is never NLopt's, whatever its message. */
    CHECK(classify_nlopt_exception(
              make_eptr(tadah::hpo::durable_best::Error("nlopt failure")), no_msg)
          == NloptOutcome::Failed);
    CHECK(classify_nlopt_exception(std::make_exception_ptr(42), no_msg)
          == NloptOutcome::Failed);
}

TEST_CASE("NLopt: host exceptions propagate with their type and message",
          "[HPO][nlopt][failure_handling]")
{
    using tadah::hpo::run_nlopt_impl;

    SECTION("durable_best::Error at eval 3 (LN_SBPLX)") {
        LoggedRun lr;
        auto sp = make_spec("NLOPT", "LN_SBPLX");
        StubHost host;
        host.throw_at = 3;
        host.thrower = [] {
            throw tadah::hpo::durable_best::Error("best_pot.tadah: EBUSY (gave up)");
        };
        try {
            run_nlopt_impl<Vec, NoRunner>(sp, host);
            FAIL("run_nlopt_impl returned normally");
        } catch (const tadah::hpo::durable_best::Error& e) {
            CHECK(std::string(e.what()) == "best_pot.tadah: EBUSY (gave up)");
        }
        CHECK(host.calls == 3);   // the host is never called after the throw
        const std::string journal = lr.finish_and_read();
        CHECK(journal.find("nlopt failure") == std::string::npos);
        CHECK(journal.find("converged") == std::string::npos);
    }

    SECTION("StagnationStop reaches RESTART") {
        auto sp = make_spec("NLOPT", "LN_BOBYQA");
        StubHost host;
        host.throw_at = 5;
        host.thrower = [] { throw tadah::hpo::StagnationStop{}; };
        CHECK_THROWS_AS((run_nlopt_impl<Vec, NoRunner>(sp, host)),
                        tadah::hpo::StagnationStop);
        CHECK(host.calls == 5);
    }

    SECTION("ShutdownRequested reaches hpo_run") {
        auto sp = make_spec("NLOPT", "LN_NELDERMEAD");
        StubHost host;
        host.throw_at = 2;
        host.thrower = [] { throw tadah::core::ShutdownRequested{}; };
        CHECK_THROWS_AS((run_nlopt_impl<Vec, NoRunner>(sp, host)),
                        tadah::core::ShutdownRequested);
        CHECK(host.calls == 2);
    }

    SECTION("std::invalid_argument from the host keeps its message") {
        /* NLopt's wrapper would have reported NLOPT_INVALID_ARGS with its
         * own text ("nlopt invalid argument") -- a rejected SETUP. */
        auto sp = make_spec("NLOPT", "LN_SBPLX");
        StubHost host;
        host.throw_at = 1;
        host.thrower = [] { throw std::invalid_argument("x"); };
        try {
            run_nlopt_impl<Vec, NoRunner>(sp, host);
            FAIL("run_nlopt_impl returned normally");
        } catch (const std::invalid_argument& e) {
            CHECK(std::string(e.what()) == "x");
        }
    }

    SECTION("LD_LBFGS: a throw inside the NUMERICAL gradient") {
        /* Eval 1 is f at the start; evals 2.. are the gradient's central
         * differences (in-house kernel: centre, then +-h per dim). */
        for (const char* kernel : {"inhouse", "ceres"}) {
            CAPTURE(kernel);
            auto sp = make_spec("NLOPT", "LD_LBFGS");
            sp.param_raw["fd_kernel"] = kernel;
            StubHost host;
            host.throw_at = 3;
            host.thrower = [] {
                throw tadah::hpo::durable_best::Error("persist failed in gradient");
            };
            try {
                run_nlopt_impl<Vec, NoRunner>(sp, host);
                FAIL("run_nlopt_impl returned normally");
            } catch (const tadah::hpo::durable_best::Error& e) {
                CHECK(std::string(e.what()) == "persist failed in gradient");
            }
            CHECK(host.calls == 3);
        }
    }

    SECTION("G_MLSL: a throw inside the LN_SBPLX local search") {
        auto sp = make_spec("NLOPT", "G_MLSL");
        sp.pop = 2;
        sp.inner = std::make_unique<OptimizerSpec>();
        sp.inner->lib  = "NLOPT";
        sp.inner->algo = "LN_SBPLX";
        sp.inner->stop.xtol_rel = 1e-6;
        sp.inner->stop.maxeval  = 40;
        /* MLSL samples a couple of points, then local searches run: eval 12
         * lies inside the first local search. The force-stop has to cross
         * from the inner optimiser to the outer one. */
        StubHost host;
        host.throw_at = 12;
        host.thrower = [] {
            throw tadah::hpo::durable_best::Error("persist failed in local search");
        };
        try {
            run_nlopt_impl<Vec, NoRunner>(sp, host);
            FAIL("run_nlopt_impl returned normally");
        } catch (const tadah::hpo::durable_best::Error& e) {
            CHECK(std::string(e.what()) == "persist failed in local search");
        }
        CHECK(host.calls == 12);
    }
}

TEST_CASE("NLopt: normal and self-terminated runs keep their status",
          "[HPO][nlopt][failure_handling]")
{
    using tadah::hpo::run_nlopt_impl;

    SECTION("MAXEVAL_REACHED returns normally") {
        LoggedRun lr;
        auto sp = make_spec("NLOPT", "LN_SBPLX");
        sp.stop.maxeval = 20;
        StubHost host;
        REQUIRE_NOTHROW(run_nlopt_impl<Vec, NoRunner>(sp, host));
        CHECK(host.calls == 20);
        CHECK(status_line(lr.finish_and_read()).find("MAXEVAL_REACHED") !=
              std::string::npos);
    }

    SECTION("a genuine NLOPT_FAILURE is still graceful") {
        /* A 1e-16 fraction step makes NELDERMEAD's first simplex collapse
         * onto the start: NLOPT_FAILURE with a stop message -- the
         * algorithm's own self-termination, not an error of the run. */
        LoggedRun lr;
        auto sp = make_spec("NLOPT", "LN_NELDERMEAD");
        sp.step = 1e-16;
        StubHost host;
        REQUIRE_NOTHROW(run_nlopt_impl<Vec, NoRunner>(sp, host));
        const std::string status = status_line(lr.finish_and_read());
        CHECK(status.find("STOPPED -- ") != std::string::npos);
        CHECK(status.find("too small") != std::string::npos);
        CHECK(status.find("FAILED") == std::string::npos);
    }
}

TEST_CASE("MULTISTART: run-level exceptions escape a start, others do not",
          "[HPO][multistart][failure_handling]")
{
    using tadah::hpo::detail::run_start_guarded;
    using tadah::hpo::detail::StartOutcome;

    CHECK_THROWS_AS(run_start_guarded(0, 3, [] {
                        throw tadah::hpo::durable_best::Error("disk");
                    }),
                    tadah::hpo::durable_best::Error);
    CHECK_THROWS_AS(run_start_guarded(0, 3, [] {
                        throw tadah::core::ShutdownRequested{};
                    }),
                    tadah::core::ShutdownRequested);
    /* StagnationStop ends only that start (its own RESTART window); the
     * batch decides what it means (test_policy_host_exceptions.cpp). */
    CHECK(run_start_guarded(1, 3, [] { throw tadah::hpo::StagnationStop{}; }) ==
          StartOutcome::Stagnated);

    /* A failure confined to one start is logged and swallowed. */
    LoggedRun lr;
    bool ran = false;
    CHECK(run_start_guarded(1, 3, [&] {
        ran = true;
        throw std::runtime_error("start-local boom");
    }) == StartOutcome::Failed);
    CHECK(ran);
    CHECK(run_start_guarded(2, 3, [] {}) == StartOutcome::Completed);
    const std::string journal = lr.finish_and_read();
    CHECK(journal.find("MULTISTART start 2/3 failed: start-local boom") !=
          std::string::npos);
}

#ifdef TADAH_USE_CERES
TEST_CASE("CERES: a host exception aborts the solve and is rethrown",
          "[HPO][ceres][failure_handling]")
{
    using tadah::hpo::detail::run_ceres_gradient_problem;
    for (const char* kernel : {"inhouse", "ceres"}) {
        CAPTURE(kernel);
        auto sp = make_spec("CERES", "LBFGS");
        sp.param_raw["fd_kernel"] = kernel;
        StubHost host;
        host.throw_at = 5;
        host.thrower = [] {
            throw tadah::hpo::durable_best::Error("persist failed under Ceres");
        };
        try {
            run_ceres_gradient_problem<Vec, NoRunner>(
                sp, host, ceres::LBFGS, ceres::WOLFE, "LBFGS");
            FAIL("run_ceres_gradient_problem returned normally");
        } catch (const tadah::hpo::durable_best::Error& e) {
            CHECK(std::string(e.what()) == "persist failed under Ceres");
        }
        CHECK(host.calls == 5);   // zero host calls after the throw
    }

    /* Control-flow exceptions keep their type too. */
    auto sp = make_spec("CERES", "LBFGS");
    StubHost host;
    host.throw_at = 1;
    host.thrower = [] { throw tadah::core::ShutdownRequested{}; };
    CHECK_THROWS_AS((run_ceres_gradient_problem<Vec, NoRunner>(
                        sp, host, ceres::LBFGS, ceres::WOLFE, "LBFGS")),
                    tadah::core::ShutdownRequested);
    CHECK(host.calls == 1);
}
#endif

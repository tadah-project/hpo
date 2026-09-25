/*
 * How host exceptions end MULTISTART / PRESTAGE batches, what the replica
 * pool keeps when chains fail together, and how RESULT reports a best
 * potential that could not be saved.
 *
 * The batch loop (detail::run_starts) and the stagnation rule
 * (detail::report_stagnation) are generic over their callables, so they
 * run here with stub starts; the policies bind them to HPO_Host (the real
 * stack is exercised end to end by [durable_e2e] in
 * test_hpo_fixweight_e2e.cpp).
 */

#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/stagnation_stop.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::hpo::HPOLog;
using tadah::hpo::StagnationStop;
using tadah::hpo::detail::StagnationReport;
using tadah::hpo::detail::StartOutcome;
using tadah::hpo::detail::report_stagnation;
using tadah::hpo::detail::run_start_guarded;
using tadah::hpo::detail::run_starts;

/* An open journal for one test; closed and removed on exit. */
struct Journal {
    fs::path path;
    Journal() : path(fs::temp_directory_path() /
                     ("policy_host_exc_" +
                      std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch().count()) +
                      ".log"))
    {
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~Journal() {
        HPOLog::instance().finish("TEST_DONE");   // no-op if already closed
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish_and_read(const std::string& status = "") const {
        HPOLog::instance().finish(status);
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

std::string line_with(const std::string& text, const std::string& needle)
{
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(needle) != std::string::npos) return line;
    return {};
}

void new_best(double loss, std::size_t eval)
{
    HPOLog::instance().log_new_best(eval, loss, 0.0, false, 0.0, false,
                                    0.0, false, 0.0, false, 0.0, false, 0.0);
}

}  // namespace

TEST_CASE("run_start_guarded: stagnation ends a start, run-level errors escape",
          "[HPO][multistart][failure_handling]")
{
    Journal j;
    CHECK(run_start_guarded(0, 3, [] {}) == StartOutcome::Completed);
    CHECK(run_start_guarded(1, 3, [] { throw StagnationStop{}; }) ==
          StartOutcome::Stagnated);

    std::exception_ptr failure;
    CHECK(run_start_guarded(2, 3, [] { throw std::runtime_error("boom"); },
                            &failure) == StartOutcome::Failed);
    REQUIRE(failure);
    CHECK_THROWS_WITH(std::rethrow_exception(failure), "boom");

    CHECK_THROWS_AS(run_start_guarded(0, 3, [] {
                        throw tadah::hpo::durable_best::Error("disk");
                    }),
                    tadah::hpo::durable_best::Error);
    CHECK_THROWS_AS(run_start_guarded(0, 3, [] {
                        throw tadah::core::ShutdownRequested{};
                    }),
                    tadah::core::ShutdownRequested);
    const std::string journal = j.finish_and_read();
    CHECK(journal.find("MULTISTART start 3/3 failed: boom") != std::string::npos);
}

/* Under RESTART a stagnating start used to leave the whole
 * multistart, so starts 2..K practically never ran. */
TEST_CASE("MULTISTART: every start runs in its own stagnation window",
          "[HPO][multistart][failure_handling]")
{
    Journal j;
    std::vector<std::string> trace;
    const auto rep = run_starts(
        3,
        [&] { trace.push_back("reset"); },
        [&](std::size_t i) {
            trace.push_back("start " + std::to_string(i + 1));
            if (i != 1) throw StagnationStop{};
        });
    CHECK(trace == std::vector<std::string>{"reset", "start 1", "reset",
                                            "start 2", "reset", "start 3"});
    CHECK(rep.k == 3);
    CHECK(rep.n_stagnated == 2);
    CHECK(rep.n_failed == 0);
    CHECK_FALSE(rep.improved);

    const std::string journal = j.finish_and_read();
    CHECK(journal.find("MULTISTART start 1/3 done: 0 eval(s) consumed, no global "
                       "improvement") != std::string::npos);
    CHECK(line_with(journal, "MULTISTART start 1/3 done").find(
              "stopped on stagnation (RESTART window)") != std::string::npos);
    CHECK(line_with(journal, "MULTISTART start 2/3 done").find(
              "stagnation") == std::string::npos);
}

TEST_CASE("Stagnation reaches RESTART by the batch's rule",
          "[HPO][multistart][prestage][failure_handling]")
{
    /* No stagnating start: nothing to report, whatever the rule. */
    CHECK_NOTHROW(report_stagnation(StagnationReport::AnyStart, 0, false));
    CHECK_NOTHROW(report_stagnation(StagnationReport::NoImprovement, 0, false));
    /* MULTISTART: any stagnating start makes RESTART relaunch the batch. */
    CHECK_THROWS_AS(report_stagnation(StagnationReport::AnyStart, 1, true),
                    StagnationStop);
    /* PRESTAGE refinement: only without an improvement. */
    CHECK_NOTHROW(report_stagnation(StagnationReport::NoImprovement, 2, true));
    CHECK_THROWS_AS(report_stagnation(StagnationReport::NoImprovement, 2, false),
                    StagnationStop);
}

TEST_CASE("run_starts: improvement is measured over the whole batch",
          "[HPO][prestage][failure_handling]")
{
    Journal j;
    new_best(5.0, 1);
    const auto improving = run_starts(
        2, [] {},
        [&](std::size_t i) {
            if (i == 0) new_best(4.0, 2);
            throw StagnationStop{};
        });
    CHECK(improving.improved);
    CHECK(improving.n_stagnated == 2);

    const auto flat = run_starts(2, [] {}, [](std::size_t) {
        throw StagnationStop{};
    });
    CHECK_FALSE(flat.improved);
}

/* A MULTISTART whose every start failed ended COMPLETED with exit
 * code 0; a failed last start left its "FAILED -- ..." as the RESULT of an
 * otherwise successful batch. */
TEST_CASE("MULTISTART: all starts failing fails the run",
          "[HPO][multistart][failure_handling]")
{
    Journal j;
    int ran = 0;
    try {
        run_starts(3, [] {}, [&](std::size_t i) {
            ++ran;
            HPOLog::instance().set_status("FAILED -- start " + std::to_string(i + 1));
            throw std::invalid_argument("no analytical derivative " +
                                        std::to_string(i + 1));
        });
        FAIL("run_starts returned normally");
    } catch (const std::invalid_argument& e) {
        /* The FIRST failure, with its type. */
        CHECK(std::string(e.what()) == "no analytical derivative 1");
    }
    CHECK(ran == 3);
    const std::string journal = j.finish_and_read();
    CHECK(journal.find("MULTISTART start 1/3 failed: no analytical derivative 1")
          != std::string::npos);
    CHECK(journal.find("MULTISTART start 3/3 failed: no analytical derivative 3")
          != std::string::npos);
}

TEST_CASE("MULTISTART: a failed start's status does not become the result",
          "[HPO][multistart][failure_handling]")
{
    SECTION("the last start fails") {
        Journal j;
        const auto rep = run_starts(3, [] {}, [](std::size_t i) {
            if (i < 2) {
                HPOLog::instance().set_status(
                    "MAXEVAL_REACHED -- maximum evaluations exhausted");
                return;
            }
            HPOLog::instance().set_status("FAILED -- invalid args");
            throw std::runtime_error("invalid args");
        });
        CHECK(rep.n_failed == 1);
        CHECK(HPOLog::instance().status() ==
              "MAXEVAL_REACHED -- maximum evaluations exhausted "
              "(1 of 3 MULTISTART starts failed; see the WARNINGs)");
        const std::string status =
            line_with(j.finish_and_read(), "Optimiser status");
        CHECK(status.find("FAILED") == std::string::npos);
        CHECK(status.find("1 of 3 MULTISTART starts failed") != std::string::npos);
    }
    SECTION("the first start fails, the others record no status") {
        Journal j;
        HPOLog::instance().set_status("");
        const auto rep = run_starts(3, [] {}, [](std::size_t i) {
            if (i == 0) {
                HPOLog::instance().set_status("FAILED -- first");
                throw std::runtime_error("first");
            }
        });
        CHECK(rep.n_failed == 1);
        CHECK(HPOLog::instance().status() ==
              "COMPLETED (1 of 3 MULTISTART starts failed; see the WARNINGs)");
    }
}

TEST_CASE("run_starts: run-level exceptions stop the batch at once",
          "[HPO][multistart][failure_handling]")
{
    Journal j;
    int ran = 0;
    CHECK_THROWS_AS(run_starts(3, [] {}, [&](std::size_t i) {
                        ++ran;
                        if (i == 1) throw tadah::hpo::durable_best::Error("disk");
                    }),
                    tadah::hpo::durable_best::Error);
    CHECK(ran == 2);
    ran = 0;
    CHECK_THROWS_AS(run_starts(3, [] {}, [&](std::size_t) {
                        ++ran;
                        throw tadah::core::ShutdownRequested{};
                    }),
                    tadah::core::ShutdownRequested);
    CHECK(ran == 1);
}

/* The pool kept the FIRST exception of an epoch, so a
 * StagnationStop (RESTART restarts) or a ShutdownRequested (exit 0) caught
 * first masked a best potential that could not be saved. */
TEST_CASE("ReplicaPool keeps the most severe exception of an epoch",
          "[HPO][anneal][failure_handling]")
{
    using tadah::hpo::anneal::ReplicaPool;

    auto stagnation = [] { throw StagnationStop{}; };
    auto shutdown   = [] { throw tadah::core::ShutdownRequested{}; };
    auto durable    = [] { throw tadah::hpo::durable_best::Error("disk full"); };
    auto other      = [] { throw std::runtime_error("other"); };

    /* Task 0 (worker 0) runs `first` at once; task 1 (worker 1) runs
     * `later` 30 ms afterwards. */
    auto epoch = [](ReplicaPool& pool, std::function<void()> first,
                    std::function<void()> later) {
        pool.run_epoch(2, [&](std::size_t t) {
            if (t == 0) { first(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            later();
        });
    };

    std::atomic<bool> stop{false};
    ReplicaPool pool(2, stop);

    epoch(pool, stagnation, durable);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::hpo::durable_best::Error);
    CHECK_FALSE(pool.has_error());   // cleared

    epoch(pool, durable, stagnation);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::hpo::durable_best::Error);

    epoch(pool, shutdown, durable);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::hpo::durable_best::Error);

    epoch(pool, stagnation, shutdown);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::core::ShutdownRequested);

    epoch(pool, shutdown, stagnation);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::core::ShutdownRequested);

    /* Equal severity: the first one stays. */
    epoch(pool, durable, other);
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::hpo::durable_best::Error);
    epoch(pool, other, durable);
    CHECK_THROWS_WITH(pool.rethrow_if_error(), "other");

    /* The severity is reset with the error: a later lone StagnationStop is
     * kept and rethrown. */
    epoch(pool, stagnation, [] {});
    CHECK_THROWS_AS(pool.rethrow_if_error(), StagnationStop);
    epoch(pool, [] {}, [] {});
    CHECK_FALSE(pool.has_error());
    CHECK_NOTHROW(pool.rethrow_if_error());
    CHECK(stop.load());
}

/* RESULT said "no evaluation produced a usable loss" / "(none
 * written)" for a first best that could not be persisted, and named
 * best_pot.tadah for an older best while the file held the newer, unsynced
 * candidate. */
TEST_CASE("RESULT reports a best that could not be saved",
          "[hpolog][durable][failure_handling]")
{
    auto result_of = [](bool with_saved_best, bool may_be_on_disk, bool unsaved) {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        if (with_saved_best) new_best(0.25, 2);
        if (unsaved) {
            log.record_unsaved_best(3, 0.125, "cannot fsync durable-best directory",
                                    may_be_on_disk);
            /* A later, higher-loss candidate does not replace it (the
             * lowest loss is kept; see "several unsaved candidates"). */
            log.record_unsaved_best(4, 0.1875, "later", false);
        }
        const std::string journal = j.finish_and_read(
            unsaved ? "FAILED -- best-potential persistence: x" : "COMPLETED");
        return std::make_pair(line_with(journal, "Best total loss"),
                              std::make_pair(line_with(journal, "Last saved best"),
                                             line_with(journal, "Best potential saved")));
    };

    SECTION("first best, rename/dir fsync may have landed") {
        const auto r = result_of(false, true, true);
        CHECK(r.first.find("1.2500e-01") != std::string::npos);
        CHECK(r.first.find("(eval 3) -- NOT saved (best-potential persistence "
                           "failed)") != std::string::npos);
        CHECK(r.first.find("none") == std::string::npos);
        CHECK(r.second.first.empty());
        CHECK(r.second.second.find("not confirmed -- best_pot.tadah may hold eval "
                                   "3 (write not synced) or be absent; check its "
                                   "HPO_SOURCE_EVAL header") != std::string::npos);
    }
    SECTION("first best, nothing written") {
        const auto r = result_of(false, false, true);
        CHECK(r.first.find("NOT saved") != std::string::npos);
        CHECK(r.second.second.find("(none written)") != std::string::npos);
    }
    SECTION("a later best, possibly on disk") {
        const auto r = result_of(true, true, true);
        CHECK(r.first.find("(eval 3) -- NOT saved") != std::string::npos);
        CHECK(r.second.first.find("2.5000e-01") != std::string::npos);
        CHECK(r.second.first.find("(eval 2") != std::string::npos);
        CHECK(r.second.second.find("may hold eval 3 (write not synced) or eval 2")
              != std::string::npos);
    }
    SECTION("a later best, not on disk") {
        const auto r = result_of(true, false, true);
        CHECK(r.second.second.find("best_pot.tadah (eval 2)") != std::string::npos);
    }
    SECTION("no persistence failure: unchanged") {
        const auto r = result_of(true, false, false);
        CHECK(r.first.find("2.5000e-01") != std::string::npos);
        CHECK(r.first.find("NOT saved") == std::string::npos);
        CHECK(r.second.first.empty());
        CHECK(r.second.second.find(": best_pot.tadah") != std::string::npos);
        CHECK(r.second.second.find("eval") == std::string::npos);
    }
}

TEST_CASE("rethrow_if_best_unsaved turns a swallowed persistence failure "
          "into the run's error", "[HPO][durable][failure_handling]")
{
    Journal j;
    CHECK_NOTHROW(tadah::hpo::detail::rethrow_if_best_unsaved());
    HPOLog::instance().record_unsaved_best(7, 0.5, "cannot rename: EIO", true);
    CHECK_THROWS_AS(tadah::hpo::detail::rethrow_if_best_unsaved(),
                    tadah::hpo::durable_best::Error);
    CHECK_THROWS_WITH(tadah::hpo::detail::rethrow_if_best_unsaved(),
                      "cannot rename: EIO");
}

/* ---------------------------------------------------------------------------
 * Concurrent failures: lost bests, the stagnation check, the RESULT of an
 * unsaved best, the persistence-failure latch and the failed-start note.
 * ------------------------------------------------------------------------- */

TEST_CASE("run_start_guarded: a lost best ends a stagnated or failed start's "
          "run", "[HPO][multistart][failure_handling]")
{
    /* A durable_best::Error that lost to a concurrent stop (or to an
     * equal-severity error) in a parallel optimiser was swallowed with the
     * start; the batch went on optimising for its whole budget. */
    Journal j;
    HPOLog::instance().record_unsaved_best(4, 0.5, "cannot rename: EIO", true);
    CHECK_THROWS_WITH(run_start_guarded(0, 3, [] { throw StagnationStop{}; }),
                      "cannot rename: EIO");
    CHECK_THROWS_AS(run_start_guarded(1, 3, [] { throw std::runtime_error("x"); }),
                    tadah::hpo::durable_best::Error);
    CHECK(run_start_guarded(2, 3, [] {}) == StartOutcome::Completed);
}

/* Evaluations that cannot become the new best skip the best mutex,
 * so the stagnation check could read iter_at_last_best_ while an earlier
 * evaluation's new best was still being decided -- publishing under the
 * mutex, or queued for it -- and stop an improving run. The check that
 * would fire now waits until as many decisions have finished as were in
 * progress when it began (they finish in lock order, which is not FIFO; see
 * detail::throw_if_stagnated). */
TEST_CASE("stagnation check waits for new-best decisions in progress",
          "[HPO][restart][failure_handling]")
{
    using tadah::hpo::detail::BestPublications;
    using tadah::hpo::detail::PublicationDone;
    using tadah::hpo::detail::throw_if_stagnated;
    std::mutex best_mtx;
    BestPublications pubs;
    std::atomic<std::size_t> last{1};
    const std::size_t window = 4;

    /* A new-best decision as HPO_Host makes it: counted, then locked;
     * `hold` models the fsync + rename (or a backoff), `new_best` stores
     * its eval number (0: it was not a new best after all). */
    auto decision = [&](std::chrono::milliseconds queued,
                        std::chrono::milliseconds hold, std::size_t new_best) {
        pubs.started.fetch_add(1);
        std::this_thread::sleep_for(queued);   // waiting for the lock
        std::lock_guard<std::mutex> g(best_mtx);
        PublicationDone done(pubs);
        std::this_thread::sleep_for(hold);
        if (new_best) last.store(new_best, std::memory_order_relaxed);
    };

    /* Not firing: no lock, no wait (the success path). */
    {
        std::unique_lock<std::mutex> held(best_mtx);
        CHECK_NOTHROW(throw_if_stagnated(4, window, last, best_mtx, pubs));
        CHECK_NOTHROW(throw_if_stagnated(6, 0, last, best_mtx, pubs));
    }

    /* Eval 6 checks while eval 4's publication holds the lock: 6 - 1 >= 4,
     * but eval 4 is a new best, so 6 - 4 < 4 once it lands. The publisher
     * keeps the lock until the check is about to start (a bounded wait), so
     * the two overlap without relying on timing; the assertions hold
     * whatever the scheduling (the check either waits for the publication
     * or already sees eval 4). */
    {
        std::atomic<bool> checking{false};
        std::thread publisher([&] {
            pubs.started.fetch_add(1);
            std::lock_guard<std::mutex> g(best_mtx);
            PublicationDone done(pubs);
            const auto give_up =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!checking.load() && std::chrono::steady_clock::now() < give_up)
                std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            last.store(4, std::memory_order_relaxed);
        });
        while (pubs.started.load() == 0) std::this_thread::yield();   // counted
        checking.store(true);
        CHECK_NOTHROW(throw_if_stagnated(6, window, last, best_mtx, pubs));
        CHECK(last.load() == 4);
        publisher.join();
        CHECK(pubs.done.load() == 1);
    }

    /* Eval 9's new best is still QUEUED for the lock (another decision
     * holds it) when eval 12 checks: 12 - 4 >= 4, but 12 - 9 < 4. */
    {
        last = 4;
        std::thread holder(decision, std::chrono::milliseconds(0),
                           std::chrono::milliseconds(60), 0);
        std::thread queued(decision, std::chrono::milliseconds(120),
                           std::chrono::milliseconds(0), 9);
        while (pubs.started.load() < 3) std::this_thread::yield();
        CHECK_NOTHROW(throw_if_stagnated(12, window, last, best_mtx, pubs));
        CHECK(last.load() == 9);
        holder.join();
        queued.join();
    }

    /* Decisions that record no new best leave the window where it was:
     * the check fires once they are done. */
    {
        std::thread failing(decision, std::chrono::milliseconds(0),
                            std::chrono::milliseconds(50), 0);
        while (pubs.started.load() < 4) std::this_thread::yield();
        CHECK_THROWS_AS(throw_if_stagnated(14, window, last, best_mtx, pubs),
                        StagnationStop);
        failing.join();
    }
    CHECK_THROWS_AS(throw_if_stagnated(13, window, last, best_mtx, pubs),
                    StagnationStop);
    CHECK_NOTHROW(throw_if_stagnated(12, window, last, best_mtx, pubs));
    CHECK(pubs.done.load() == pubs.started.load());
}

/* (a) DLIB:GFS with THREADS > 1: two host exceptions in flight together made
 * ~dlib::thread_pool call abort() (exit 134, journal left open), and even
 * one was rethrown while other tasks still ran on find_min_global's
 * locals. The adapter now keeps them in its own slot, most severe first. */
TEST_CASE("DLIB:GFS keeps the most severe of concurrent host exceptions",
          "[HPO][dlib][failure_handling]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    struct TwoThrowsHost {
        std::function<void()> first, second;
        std::atomic<int> calls{0}, arrived{0};
        double operator()(const Vec& x)
        {
            const int n = ++calls;
            if (n <= 2 && (first || second)) {
                ++arrived;
                const auto t0 = std::chrono::steady_clock::now();
                while (arrived.load() < 2 &&
                       std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
                    std::this_thread::yield();
                if (n == 1 && first) first();
                if (n == 2 && second) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    second();
                }
            }
            double s = 0.0;
            for (long i = 0; i < x.size(); ++i) s += (x(i) - 0.3) * (x(i) - 0.3);
            return s;
        }
    };
    auto spec = [] {
        tadah::hpo::OptimizerSpec sp;
        sp.lib = "DLIB";
        sp.algo = "GFS";
        sp.low = {-1.0, -1.0};
        sp.high = {1.0, 1.0};
        sp.is_int = {false, false};
        sp.log_scaled = {false, false};
        sp.num_threads = 4;
        sp.stop.maxeval = 40;
        return sp;
    };
    auto stagnation = [] { throw StagnationStop{}; };
    auto shutdown   = [] { throw tadah::core::ShutdownRequested{}; };
    auto durable    = [] { throw tadah::hpo::durable_best::Error("disk full"); };

    Journal j;
    {
        TwoThrowsHost host;
        host.first = stagnation;
        host.second = durable;
        CHECK_THROWS_AS(tadah::hpo::run_dlib_global_impl(spec(), host),
                        tadah::hpo::durable_best::Error);
        CHECK(host.calls.load() < 40);   // the search stopped early
    }
    {
        TwoThrowsHost host;
        host.first = stagnation;
        host.second = shutdown;
        CHECK_THROWS_AS(tadah::hpo::run_dlib_global_impl(spec(), host),
                        tadah::core::ShutdownRequested);
    }
    {
        /* One StagnationStop (RESTART): rethrown once the pool drained. */
        TwoThrowsHost host;
        host.first = stagnation;
        CHECK_THROWS_AS(tadah::hpo::run_dlib_global_impl(spec(), host),
                        StagnationStop);
    }
    {
        /* No exception: the run completes its budget. */
        TwoThrowsHost host;
        CHECK_NOTHROW(tadah::hpo::run_dlib_global_impl(spec(), host));
        CHECK(host.calls.load() == 40);
    }
}

/* The unsaved-best record was first-only and always reported as the
 * run's best, even after a lower best was persisted by another chain once
 * the storage recovered. */
TEST_CASE("RESULT: an unsaved best superseded by a later saved best",
          "[hpolog][durable][failure_handling]")
{
    auto run = [](bool may_be_on_disk, double saved_loss, bool rows_lost) {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        log.record_unsaved_best(9, 0.0918, "cannot fsync: EIO", may_be_on_disk);
        new_best(saved_loss, 23);
        if (rows_lost) log.record_rows_lost(5);
        return j.finish_and_read("FAILED -- best-potential persistence: x");
    };

    SECTION("a lower loss saved later: that is the result") {
        for (bool on_disk : {false, true}) {
            const std::string journal = run(on_disk, 0.0802, true);
            INFO(journal);
            const std::string best = line_with(journal, "Best total loss");
            CHECK(best.find("8.0200e-02") != std::string::npos);
            CHECK(best.find("(eval 23") != std::string::npos);
            CHECK(best.find("NOT saved") == std::string::npos);
            CHECK(line_with(journal, "Unsaved candidate").find(
                      "9.1800e-02  (eval 9) -- could not be persisted; "
                      "superseded by eval 23") != std::string::npos);
            CHECK(line_with(journal, "Last saved best").empty());
            CHECK(line_with(journal, "Best potential saved").find(
                      ": best_pot.tadah (eval 23)") != std::string::npos);
            CHECK(journal.find("is complete") != std::string::npos);
            CHECK(journal.find("could not be persisted either") == std::string::npos);
            CHECK(line_with(journal, "Optimiser status").find("FAILED") !=
                  std::string::npos);
        }
    }
    SECTION("an equal loss: superseded only when the candidate is not on disk") {
        const std::string off = run(false, 0.0918, false);
        CHECK(line_with(off, "Best total loss").find("NOT saved") == std::string::npos);
        CHECK(line_with(off, "Unsaved candidate").find(
                  "(eval 9) -- could not be persisted; same loss as eval 23") !=
              std::string::npos);
        const std::string on = run(true, 0.0918, false);
        CHECK(line_with(on, "Best total loss").find("(eval 9) -- NOT saved") !=
              std::string::npos);
        CHECK(line_with(on, "Best potential saved").find("may hold eval 9") !=
              std::string::npos);
    }
    SECTION("a higher saved loss: the candidate stays the unsaved best") {
        const std::string journal = run(false, 0.25, true);
        CHECK(line_with(journal, "Best total loss").find("(eval 9) -- NOT saved") !=
              std::string::npos);
        CHECK(line_with(journal, "Last saved best").find("(eval 23") !=
              std::string::npos);
        CHECK(line_with(journal, "Unsaved candidate").empty());
        CHECK(journal.find("could not be persisted either") != std::string::npos);
        CHECK(journal.find("is complete") == std::string::npos);
    }
    SECTION("several unsaved candidates: the lowest is reported") {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        log.record_unsaved_best(9, 0.0918, "first error", false);
        log.record_unsaved_best(12, 0.0850, "second error", true);
        log.record_unsaved_best(14, 0.0870, "third error", false);
        const auto ub = log.unsaved_best();
        REQUIRE(ub);
        CHECK(ub->eval == 12);
        CHECK(ub->loss == 0.0850);
        CHECK(ub->what == "first error");   // the run's status
        CHECK(ub->may_be_on_disk());        // one of them
        REQUIRE(ub->maybe_on_disk.size() == 1u);
        CHECK(ub->maybe_on_disk[0].first == 12u);
        CHECK(ub->maybe_on_disk[0].second == 0.0850);
        CHECK_THROWS_WITH(tadah::hpo::detail::rethrow_if_best_unsaved(),
                          "first error");
    }
}

/* The "may hold eval N" line named the LOWEST-loss candidate while
 * the may-be-on-disk flag was ORed over every candidate: a candidate that
 * never reached rename(2) was reported as possibly in best_pot.tadah. Now
 * the line names the candidates whose rename may have landed, and a saved
 * best replaces those above its loss. */
TEST_CASE("RESULT names the candidates that may be in best_pot.tadah",
          "[hpolog][durable][failure_handling]")
{
    /* eval 9 (0.09): renamed, directory fsync failed; eval 12 (0.08):
     * failed before its rename. */
    auto run = [](std::optional<std::pair<double, std::size_t>> saved_before,
                  std::optional<std::pair<double, std::size_t>> saved_between,
                  std::optional<std::pair<double, std::size_t>> saved_after) {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        if (saved_before) new_best(saved_before->first, saved_before->second);
        log.record_unsaved_best(9, 0.09, "cannot fsync durable-best directory", true);
        if (saved_between) new_best(saved_between->first, saved_between->second);
        log.record_unsaved_best(12, 0.08, "lock busy", false);
        if (saved_after) new_best(saved_after->first, saved_after->second);
        return j.finish_and_read("FAILED -- best-potential persistence: x");
    };
    using P = std::pair<double, std::size_t>;

    SECTION("no saved best: the flagged candidate or nothing") {
        const std::string journal = run(std::nullopt, std::nullopt, std::nullopt);
        INFO(journal);
        CHECK(line_with(journal, "Best total loss").find(
                  "8.0000e-02  (eval 12) -- NOT saved") != std::string::npos);
        CHECK(line_with(journal, "Best potential saved").find(
                  ": not confirmed -- best_pot.tadah may hold eval 9 (write not "
                  "synced) or be absent; check its HPO_SOURCE_EVAL header") !=
              std::string::npos);
    }
    SECTION("an older saved best: the flagged candidate or that best") {
        const std::string journal = run(P{0.5, 2}, std::nullopt, std::nullopt);
        INFO(journal);
        CHECK(line_with(journal, "Best total loss").find("(eval 12) -- NOT saved") !=
              std::string::npos);
        CHECK(line_with(journal, "Last saved best").find("(eval 2") != std::string::npos);
        CHECK(line_with(journal, "Best potential saved").find(
                  "may hold eval 9 (write not synced) or eval 2;") != std::string::npos);
    }
    SECTION("a later saved best of the reported loss: superseded") {
        /* eval 23 (0.08) replaced eval 9's rename; eval 12 never landed. */
        const std::string journal = run(std::nullopt, std::nullopt, P{0.08, 23});
        INFO(journal);
        CHECK(line_with(journal, "Best total loss").find("8.0000e-02  (eval 23") !=
              std::string::npos);
        CHECK(line_with(journal, "Best total loss").find("NOT saved") ==
              std::string::npos);
        CHECK(line_with(journal, "Unsaved candidate").find(
                  "(eval 12) -- could not be persisted; same loss as eval 23") !=
              std::string::npos);
        CHECK(line_with(journal, "Best potential saved").find(
                  ": best_pot.tadah (eval 23)") != std::string::npos);
    }
    SECTION("a saved best between the two: it replaced the flagged one") {
        /* eval 20 (0.085) replaced eval 9's rename; eval 12 (0.08), the
         * run's best, was never written. */
        const std::string journal = run(std::nullopt, P{0.085, 20}, std::nullopt);
        INFO(journal);
        CHECK(line_with(journal, "Best total loss").find(
                  "8.0000e-02  (eval 12) -- NOT saved") != std::string::npos);
        CHECK(line_with(journal, "Last saved best").find("(eval 20") !=
              std::string::npos);
        CHECK(line_with(journal, "Best potential saved").find(
                  ": best_pot.tadah (eval 20)") != std::string::npos);
        CHECK(journal.find("may hold") == std::string::npos);
    }
    SECTION("several candidates may be on disk: lowest loss first") {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        log.record_unsaved_best(9, 0.09, "first", true);
        log.record_unsaved_best(12, 0.08, "second", true);
        const std::string journal = j.finish_and_read("FAILED -- x");
        INFO(journal);
        CHECK(line_with(journal, "Best potential saved").find(
                  "may hold eval 12 or eval 9 (write not synced) or be absent") !=
              std::string::npos);
    }
}

/* After the first best-potential publication failed for good, every
 * candidate queued on the best mutex behind it spent its own full retry
 * budget (a DLIB:GFS THREADS 8 run took 8 x 30 s to end). Once tripped, the
 * latch selects the no-backoff Options for every later publication; the
 * first failure stays the cause, and a later success does not reset it. */
TEST_CASE("durable_best::FailLatch selects the no-backoff Options once tripped",
          "[durable_best][failure_handling]")
{
    using tadah::hpo::durable_best::FailLatch;
    using tadah::hpo::durable_best::Options;
    Options normal, latched;
    latched.no_backoff = true;
    FailLatch latch;
    CHECK_FALSE(latch.failed_eval());
    CHECK(&latch.select(normal, latched) == &normal);

    latch.trip(11);
    REQUIRE(latch.failed_eval());
    CHECK(*latch.failed_eval() == 11u);
    CHECK(&latch.select(normal, latched) == &latched);
    /* a later failure (a latched candidate failed too) keeps the first */
    latch.trip(18);
    CHECK(*latch.failed_eval() == 11u);
    CHECK(&latch.select(normal, latched) == &latched);

    /* each run starts unlatched */
    latch.reset();
    CHECK_FALSE(latch.failed_eval());
    CHECK(&latch.select(normal, latched) == &normal);
    latch.trip(3);
    CHECK(*latch.failed_eval() == 3u);
}

/* Of several failed candidates, a later one's error can reach
 * hpo_run() before the one that tripped the latch (the pools keep the first
 * exception to arrive). hpo_run() reports the first recorded failure. */
TEST_CASE("hpo_run reports the first recorded persistence failure",
          "[hpolog][durable][failure_handling]")
{
    using tadah::hpo::durable_best::Error;
    using tadah::hpo::detail::durable_failure_cause;
    const std::string first =
        "cannot open durable-best lock 'best_pot.tadah.lock': Device or "
        "resource busy (gave up after 3 attempts over 1 s)";
    const std::string later =
        "cannot open durable-best lock 'best_pot.tadah.lock': Device or "
        "resource busy (gave up after 1 attempt over 0 s)";

    SECTION("nothing recorded: the error's own message") {
        Journal j;
        CHECK(durable_failure_cause(Error(later)) == later);
    }
    SECTION("a later candidate's error arrives first: the recorded one wins") {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        log.record_unsaved_best(4, 0.5, first, false);
        log.record_unsaved_best(7, 0.4, later, false);
        const std::string cause = durable_failure_cause(Error(later));
        CHECK(cause == first);
        /* hpo_run's status line carries it */
        const std::string journal =
            j.finish_and_read("FAILED -- best-potential persistence: " + cause);
        INFO(journal);
        CHECK(line_with(journal, "Optimiser status").find(
                  "FAILED -- best-potential persistence: " + first) !=
              std::string::npos);
        CHECK(journal.find("(gave up after 1 attempt over 0 s)") ==
              std::string::npos);
    }
    SECTION("the recorded error itself passes through unchanged") {
        Journal j;
        HPOLog::instance().record_unsaved_best(4, 0.5, first, false);
        CHECK(durable_failure_cause(Error(first)) == first);
    }
}

/* DIAGNOSTICS vouched "[ok] Every evaluation produced a usable loss" above
 * RESULT's "Best total loss : (none -- no evaluation produced a usable
 * loss)", and next to NaN/Inf clamps. */
TEST_CASE("DIAGNOSTICS claims usable losses only when there were",
          "[hpolog][failure_handling]")
{
    auto diagnostics = [](bool best, std::size_t nan_evals) {
        Journal j;
        HPOLog& log = HPOLog::instance();
        log.record_eval(1, 1.0, false);
        for (std::size_t i = 0; i < nan_evals; ++i) log.record_eval(2 + i, 1.0, true);
        if (best) new_best(0.5, 1);
        return j.finish_and_read("SHUTDOWN (SIGINT)");
    };
    const std::string none = diagnostics(false, 0);
    CHECK(none.find("Every evaluation produced a usable loss") == std::string::npos);
    CHECK(none.find("[info] No graded failures were recorded; no evaluation "
                    "produced a usable loss.") != std::string::npos);
    CHECK(line_with(none, "Best total loss").find("(none -- no evaluation produced "
                                                  "a usable loss)") != std::string::npos);

    const std::string nan = diagnostics(true, 1);
    CHECK(nan.find("Every evaluation produced a usable loss") == std::string::npos);
    CHECK(nan.find("(the NaN/Inf clampings above are not graded)") != std::string::npos);

    const std::string clean = diagnostics(true, 0);
    CHECK(clean.find("[ok] Every evaluation produced a usable loss.") != std::string::npos);

    /* A best that could not be saved was still a usable loss. */
    Journal j;
    HPOLog::instance().record_eval(1, 1.0, false);
    HPOLog::instance().record_unsaved_best(1, 0.5, "x", false);
    const std::string unsaved = j.finish_and_read("FAILED -- x");
    CHECK(unsaved.find("[ok] Every evaluation produced a usable loss.") !=
          std::string::npos);
    CHECK(line_with(unsaved, "Best total loss").find("NOT saved") != std::string::npos);
}

/* The failed-start note was per batch: RESTART's exhaustion status
 * dropped it, and a later batch could carry an earlier batch's note. */
TEST_CASE("MULTISTART: the failed-start note counts the whole run",
          "[HPO][multistart][failure_handling]")
{
    using tadah::hpo::detail::failed_starts_suffix;
    using tadah::hpo::detail::strip_failed_starts_note;
    Journal j;
    HPOLog& log = HPOLog::instance();

    /* Launch 1: start 2 of 3 fails, the others stagnate. */
    const auto rep1 = run_starts(3, [] {}, [](std::size_t i) {
        if (i == 1) throw std::runtime_error("rejected setup");
        HPOLog::instance().set_status("XTOL_REACHED -- x");
        throw StagnationStop{};
    });
    CHECK(rep1.n_failed == 1);
    CHECK(log.status() == "XTOL_REACHED -- x (1 of 3 MULTISTART starts failed; "
                          "see the WARNINGs)");

    /* Launch 2 (a RESTART relaunch): no failure, no status of its own. The
     * note is neither dropped nor stacked; it counts both launches. */
    const auto rep2 = run_starts(3, [] {}, [](std::size_t) {});
    CHECK(rep2.n_failed == 0);
    CHECK(log.status() == "XTOL_REACHED -- x (1 of 6 MULTISTART starts failed "
                          "across 2 launches; see the WARNINGs)");

    /* What RestartPolicy appends to its exhaustion status. */
    CHECK("STOPPED -- stagnation: no new best in 3 evals after 1 restart(s)" +
          failed_starts_suffix(log.start_tally()) ==
          "STOPPED -- stagnation: no new best in 3 evals after 1 restart(s) "
          "(1 of 6 MULTISTART starts failed across 2 launches; see the WARNINGs)");

    CHECK(strip_failed_starts_note(log.status()) == "XTOL_REACHED -- x");
    CHECK(strip_failed_starts_note("COMPLETED (see notes; see the WARNINGs)") ==
          "COMPLETED (see notes; see the WARNINGs)");
    CHECK(failed_starts_suffix(HPOLog::StartTally{}).empty());

    /* A new run starts a new tally. */
    log.reset_for_test();
    CHECK(log.start_tally().ran == 0);
    CHECK(log.start_tally().failed == 0);
}

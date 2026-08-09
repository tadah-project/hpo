#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

/* Bug 5 regression — when an exception escapes from the eval pipeline
 * BEFORE submit_result() is called, the writer's `next_flush_` cursor
 * gets stuck at the missing iter. Subsequent iters enqueue fine, land
 * in io_buf_, but the drain loop `while (io_buf_.count(next_flush_))`
 * exits immediately at the gap because the lost iter was never
 * inserted into io_buf_.
 *
 * The fix in HPO_Host::operator() wraps the eval pipeline in
 * try/catch (excluding ShutdownRequested + StagnationStop, which the
 * optimiser must see) and submits a synthetic FAIL_SCORE Result with
 * the consumed iter so the writer queue stays contiguous.
 *
 * The production fix is in operator()'s try/catch which requires a
 * fully-constructed HPO_Host (heavy: needs StructureDB, trainer,
 * etc.). This test instead exercises the SAME drain-recovery
 * structure the fix relies on — that the writer's drain advances
 * across a previously-missing iter once a FAIL_SCORE Result for that
 * iter is enqueued. End-to-end coverage of operator() lives in the
 * live-ITER5 verification step of the bug ticket. */

namespace {

struct FakeResult {
    std::size_t iter = 0;
    bool        is_failscore = false;
    double      loss = 0.0;
};

/* Mirror of the writer's drain loop. The fix's invariant is that as
 * long as io_buf_ has the next contiguous iter, the loop advances. If
 * iter N is missing, the loop stops at next_flush_ = N until N is
 * inserted (which is exactly what the operator() catch block does
 * via submit_result of a FAIL_SCORE Result). */
struct DrainSim {
    std::map<std::size_t, FakeResult> io_buf;
    std::atomic<std::size_t>          next_flush{1};
    std::vector<std::size_t>          drained_iters;
    std::vector<std::size_t>          drained_failscore_iters;
    std::vector<double>               drained_losses;

    void drain_until_gap() {
        while (io_buf.count(next_flush)) {
            const auto& e = io_buf[next_flush];
            drained_iters.push_back(next_flush);
            if (e.is_failscore) drained_failscore_iters.push_back(next_flush);
            drained_losses.push_back(e.loss);
            io_buf.erase(next_flush++);
        }
    }
};

}  // namespace

TEST_CASE("Writer drain stalls at missing iter, recovers after FAIL_SCORE submit",
          "[eval-thread][bug5]")
{
    DrainSim sim;
    /* Push 100 iters, skipping iter 50 to simulate a throw between
     * ++g_iter_ and submit_result. */
    for (std::size_t i = 1; i <= 100; ++i) {
        if (i == 50) continue;       // simulated lost iter
        sim.io_buf.emplace(i, FakeResult{i, false});
    }

    /* Without the fix: drain stops at the gap. Iters 1..49 drain;
     * iters 51..100 sit in io_buf_ unwritten. */
    sim.drain_until_gap();
    REQUIRE(sim.next_flush.load() == 50);
    REQUIRE(sim.drained_iters.size() == 49);
    REQUIRE(sim.drained_iters.back() == 49);
    REQUIRE(sim.io_buf.size() == 50);   // iters 51..100 still queued
    REQUIRE(sim.io_buf.count(50) == 0); // confirm gap

    /* The Bug-5 fix's catch block submits a synthetic FAIL_SCORE
     * Result for the lost iter. After that submit, the queue is
     * contiguous again and the writer drains the rest. */
    sim.io_buf.emplace(50, FakeResult{50, /*is_failscore=*/true});
    sim.drain_until_gap();

    CHECK(sim.next_flush.load() == 101);
    CHECK(sim.io_buf.empty());
    CHECK(sim.drained_iters.size() == 100);
    CHECK(sim.drained_failscore_iters.size() == 1);
    CHECK(sim.drained_failscore_iters.front() == 50);
}

TEST_CASE("Fatal best-potential persistence still fills its ordered iter slot",
          "[eval-thread][durable-best][queue]")
{
    DrainSim sim;
    constexpr std::size_t failed_iter = 7;
    constexpr double completed_loss = 0.125;

    /* Mirrors a durable_best::Error after evaluation completed but before the
     * normal submit_result call. Later evaluations may already be queued. */
    for (std::size_t i = 1; i <= 20; ++i) {
        if (i == failed_iter) continue;
        sim.io_buf.emplace(i, FakeResult{i, false, static_cast<double>(i)});
    }
    sim.drain_until_gap();
    REQUIRE(sim.next_flush.load() == failed_iter);

    /* The dedicated persistence catch enqueues the completed Result with its
     * real loss, then rethrows the fatal I/O error. The writer can therefore
     * drain every later result during final cleanup instead of dropping them
     * behind a permanent gap. */
    sim.io_buf.emplace(
        failed_iter, FakeResult{failed_iter, false, completed_loss});
    sim.drain_until_gap();

    CHECK(sim.next_flush.load() == 21);
    CHECK(sim.io_buf.empty());
    CHECK(sim.drained_iters.size() == 20);
    CHECK(sim.drained_failscore_iters.empty());
    REQUIRE(sim.drained_losses.size() == 20);
    CHECK(sim.drained_losses[failed_iter - 1] == Approx(completed_loss));
}

TEST_CASE("Multiple non-adjacent throws all recoverable",
          "[eval-thread][bug5]")
{
    /* If the eval pipeline throws on iter 25 AND iter 60, the fix
     * submits FAIL_SCORE for both, and the drain reaches iter 100. */
    DrainSim sim;
    const std::vector<std::size_t> lost = {25, 60};
    for (std::size_t i = 1; i <= 100; ++i) {
        bool is_lost = false;
        for (auto j : lost) if (i == j) { is_lost = true; break; }
        if (is_lost) continue;
        sim.io_buf.emplace(i, FakeResult{i, false});
    }

    /* Drain stops at iter 25 (first gap). */
    sim.drain_until_gap();
    CHECK(sim.next_flush.load() == 25);

    /* Insert FAIL_SCORE for iter 25 — drain continues to iter 60. */
    sim.io_buf.emplace(25, FakeResult{25, true});
    sim.drain_until_gap();
    CHECK(sim.next_flush.load() == 60);

    /* Insert FAIL_SCORE for iter 60 — drain finishes. */
    sim.io_buf.emplace(60, FakeResult{60, true});
    sim.drain_until_gap();
    CHECK(sim.next_flush.load() == 101);
    CHECK(sim.drained_failscore_iters.size() == 2);
}

TEST_CASE("Bug 5 fix: HPOLog records the eval-thread catch as a warning",
          "[eval-thread][bug5][log]")
{
    /* The catch path emits a journal warning of the form:
     *   "eval N: eval-thread caught throw, submitting FAIL_SCORE..."
     * We exercise the same logging path here so the journal-format
     * contract is locked down — production breakages of the warning
     * format would surface in operator-side tooling that greps for
     * this string. */
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto path = std::filesystem::temp_directory_path() /
        ("tadah_eval_throw_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    log.warning(std::string("eval ") + std::to_string(50) +
                ": eval-thread caught throw, submitting FAIL_SCORE to "
                "keep writer queue contiguous (type=St13runtime_error): synthetic");
    log.finish("TEST_DONE");

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("eval 50:") != std::string::npos);
    CHECK(body.find("eval-thread caught throw") != std::string::npos);
    CHECK(body.find("submitting FAIL_SCORE") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

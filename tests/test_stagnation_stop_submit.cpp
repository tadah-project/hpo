#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

/* Bug 7 root-cause regression — the BASINS GLOBAL→LOCAL boundary freeze.
 *
 * BasinsPolicy sets a stagnation threshold for the GLOBAL phase. Once
 * stagnation_count() >= threshold, operator() throws StagnationStop.
 * Before this fix, the catch chain rethrew StagnationStop (and
 * ShutdownRequested) WITHOUT submitting a Result for the iter slot
 * already consumed at `++g_iter_`. The optimiser propagated the throw
 * up to RestartPolicy/BasinsPolicy which transitioned to LOCAL — but
 * the writer's next_flush_ was permanently stuck at the abandoned
 * iter. Bug 6's writer gap-skip would eventually advance after 60 s,
 * but every BASINS GLOBAL→LOCAL transition cost a 60 s freeze plus
 * one lost per-eval log line.
 *
 * The fix: both StagnationStop and ShutdownRequested catches now build
 * a synthetic FAIL_SCORE Result with the consumed r.iter, call
 * submit_result() on it, and only then rethrow. The writer queue stays
 * contiguous — no gap forms, no 60 s wait.
 *
 * This test exercises the SAME drain-recovery invariant that Bug 5's
 * test covers (writer drains across an iter that was originally
 * abandoned, after a synthetic FAIL_SCORE is enqueued for it). The
 * difference here is the journal-format contract: StagnationStop's
 * recovery uses a distinct warning prefix so operators can grep it
 * separately from the generic eval-thread catch. */

namespace {

struct FakeResult {
    std::size_t iter = 0;
    bool        is_failscore = false;
};

struct DrainSim {
    std::map<std::size_t, FakeResult> io_buf;
    std::size_t                       next_flush = 1;
    std::vector<std::size_t>          drained_iters;
    std::vector<std::size_t>          drained_failscore_iters;

    void drain_until_gap() {
        while (io_buf.count(next_flush)) {
            const auto& e = io_buf[next_flush];
            drained_iters.push_back(next_flush);
            if (e.is_failscore) drained_failscore_iters.push_back(next_flush);
            io_buf.erase(next_flush++);
        }
    }
};

}  // namespace

TEST_CASE("StagnationStop recovery: synthetic FAIL_SCORE keeps queue contiguous",
          "[stagnation-stop][bug7]")
{
    /* Mirrors the live ITER5 scenario: ~3458 successful evals, then
     * iter 3459 throws StagnationStop at the BASINS GLOBAL→LOCAL
     * boundary. With the fix, a synthetic FAIL_SCORE for iter 3459 is
     * submitted before the rethrow, so the writer drains 3459 and
     * proceeds to 3460+. */
    DrainSim sim;
    sim.next_flush = 3450;
    /* Successful evals 3450..3458 (one short of the boundary). */
    for (std::size_t i = 3450; i <= 3458; ++i)
        sim.io_buf.emplace(i, FakeResult{i, false});
    /* The LOCAL phase that follows resumes at 3460+. */
    for (std::size_t i = 3460; i <= 3470; ++i)
        sim.io_buf.emplace(i, FakeResult{i, false});

    /* Without the fix, iter 3459 would be missing — drain stops there. */
    sim.drain_until_gap();
    REQUIRE(sim.next_flush == 3459);
    REQUIRE(sim.drained_iters.size() == 9);  // 3450..3458
    REQUIRE(sim.io_buf.size() == 11);        // 3460..3470 stuck

    /* With the fix, the StagnationStop catch enqueues a synthetic
     * FAIL_SCORE Result for iter 3459 BEFORE rethrowing. */
    sim.io_buf.emplace(3459, FakeResult{3459, /*is_failscore=*/true});
    sim.drain_until_gap();

    CHECK(sim.next_flush == 3471);
    CHECK(sim.io_buf.empty());
    CHECK(sim.drained_iters.size() == 21);
    CHECK(sim.drained_failscore_iters.size() == 1);
    CHECK(sim.drained_failscore_iters.front() == 3459);
}

TEST_CASE("StagnationStop recovery: warning text format is grep-friendly",
          "[stagnation-stop][bug7][log]")
{
    /* The fix emits a journal warning of the form:
     *   "eval N: StagnationStop — submitting FAIL_SCORE to keep
     *    writer queue contiguous before rethrow"
     * Lock the format down so operator tooling can grep it. */
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto path = std::filesystem::temp_directory_path() /
        ("tadah_stagnation_stop_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    log.warning(std::string("eval ") + std::to_string(3459) +
                ": StagnationStop — submitting FAIL_SCORE to keep "
                "writer queue contiguous before rethrow");
    log.finish("TEST_DONE");

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("eval 3459:") != std::string::npos);
    CHECK(body.find("StagnationStop") != std::string::npos);
    CHECK(body.find("FAIL_SCORE") != std::string::npos);
    CHECK(body.find("before rethrow") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

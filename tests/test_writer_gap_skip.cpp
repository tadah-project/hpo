#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

/* Bug 6 regression — Bug 5 hypothesised that the writer freeze was due
 * to an exception escaping the eval pipeline before submit_result, but
 * the live ITER5 evidence (writer_debug.log entries showing
 * io_buf_size growing past 1200 with next_flush_ stuck) showed the
 * eval thread was simply HUNG inside train/predict/LAMMPS — no throw
 * happened, so the operator() try/catch never fired.
 *
 * The fix in submit_result's exclusive section: after the drain loop,
 * if the writer wrote zero iters AND io_buf_ has higher keys, start a
 * "stuck since" timer. After 60 s of continuously-stuck state, advance
 * next_flush_ to the smallest available key, log a journal warning,
 * and re-drain. This recovers liveness at the cost of losing per-eval
 * logs for the hung iter slot(s).
 *
 * The production fix lives inside a static member function with many
 * dependencies (HPOTargets, contexts, StructureDB, ofstreams). This
 * test exercises the SAME state-machine pattern (stuck-timer + skip)
 * in isolation. End-to-end coverage lives in the live-ITER5
 * verification step. */

namespace {

/* Mirror of the writer's gap-skip state machine. */
struct GapSkipSim {
    std::map<std::size_t, int> io_buf;
    std::size_t                next_flush = 1;
    std::int64_t               stuck_since_ms = 0;
    std::vector<std::size_t>   drained;
    std::vector<std::size_t>   skipped_ranges;   // start-of-range entries
    /* Start clock at a non-zero value because production uses
     * steady_clock (monotonically increasing, never returns 0) and we
     * use 0 as the "not stuck" sentinel. Initial clock=1_000_000 is
     * just an arbitrary positive baseline. */
    std::int64_t               clock_ms = 1'000'000;

    void tick(std::int64_t ms) { clock_ms += ms; }

    /* One writer call. Mirrors submit_result's exclusive section
     * including the gap-skip block. */
    void run_writer(std::int64_t gap_skip_ms) {
        std::size_t written = 0;

        /* Main drain loop. */
        while (io_buf.count(next_flush)) {
            drained.push_back(next_flush);
            io_buf.erase(next_flush++);
            ++written;
        }

        /* Gap-skip recovery. */
        if (written > 0) {
            stuck_since_ms = 0;
        } else if (!io_buf.empty()) {
            const auto smallest = io_buf.begin()->first;
            if (smallest > next_flush) {
                if (stuck_since_ms == 0) {
                    stuck_since_ms = clock_ms;   // start timer
                } else if (clock_ms - stuck_since_ms > gap_skip_ms) {
                    skipped_ranges.push_back(next_flush);
                    next_flush = smallest;
                    stuck_since_ms = 0;
                    /* Re-drain after the skip. */
                    while (io_buf.count(next_flush)) {
                        drained.push_back(next_flush);
                        io_buf.erase(next_flush++);
                    }
                }
            }
        }
    }
};

}  // namespace

TEST_CASE("Writer gap-skip: stuck timer starts on first zero-progress call",
          "[writer][bug6]")
{
    GapSkipSim sim;
    /* Push iters 1..10 EXCEPT iter 5 — simulate a hung eval thread that
     * consumed slot 5 but never returned. */
    for (std::size_t i = 1; i <= 10; ++i)
        if (i != 5) sim.io_buf.emplace(i, 0);

    /* Call 1 drains 1..4 (written>0 → timer cleared, no gap-detect yet). */
    sim.run_writer(/*gap_skip_ms=*/60'000);
    CHECK(sim.next_flush == 5);
    CHECK(sim.drained.size() == 4);
    CHECK(sim.stuck_since_ms == 0);     // not yet
    CHECK(sim.skipped_ranges.empty());

    /* Call 2 finds the gap with no progress this call — timer starts. */
    sim.run_writer(60'000);
    CHECK(sim.next_flush == 5);
    CHECK(sim.stuck_since_ms != 0);     // timer running
    CHECK(sim.skipped_ranges.empty());  // no skip yet (under threshold)
}

TEST_CASE("Writer gap-skip: skip fires once timer elapses",
          "[writer][bug6]")
{
    GapSkipSim sim;
    for (std::size_t i = 1; i <= 10; ++i)
        if (i != 5) sim.io_buf.emplace(i, 0);

    /* Drain 1..4. */
    sim.run_writer(/*gap_skip_ms=*/60'000);
    REQUIRE(sim.next_flush == 5);
    REQUIRE(sim.stuck_since_ms == 0);

    /* Second call hits the gap with written=0 → timer starts. */
    sim.run_writer(60'000);
    REQUIRE(sim.next_flush == 5);
    REQUIRE(sim.stuck_since_ms != 0);

    /* Tick 30 s — still under threshold, no skip. */
    sim.tick(30'000);
    sim.run_writer(60'000);
    CHECK(sim.next_flush == 5);
    CHECK(sim.skipped_ranges.empty());

    /* Tick another 35 s — total 65 s elapsed since timer started. Skip. */
    sim.tick(35'000);
    sim.run_writer(60'000);
    CHECK(sim.next_flush == 11);  // jumped to smallest=6, then drained 6..10
    CHECK(sim.skipped_ranges.size() == 1);
    CHECK(sim.skipped_ranges.front() == 5);
    CHECK(sim.io_buf.empty());
    CHECK(sim.drained.size() == 9);  // 1..4 from first call + 6..10 after skip
}

TEST_CASE("Writer gap-skip: timer resets on resumed progress",
          "[writer][bug6]")
{
    GapSkipSim sim;
    /* Iters 1..4 first, then iter 5 STILL missing. */
    for (std::size_t i = 1; i <= 4; ++i) sim.io_buf.emplace(i, 0);
    sim.io_buf.emplace(7, 0);

    /* Drain 1..4. */
    sim.run_writer(60'000);
    REQUIRE(sim.next_flush == 5);
    REQUIRE(sim.stuck_since_ms == 0);   // not yet (made progress)

    /* Second call: gap detected, timer starts. */
    sim.run_writer(60'000);
    REQUIRE(sim.next_flush == 5);
    REQUIRE(sim.stuck_since_ms != 0);

    /* Late-arriving iter 5 — eval thread unstuck before timeout. */
    sim.io_buf.emplace(5, 0);
    sim.io_buf.emplace(6, 0);
    sim.tick(30'000);
    sim.run_writer(60'000);
    CHECK(sim.next_flush == 8);          // drained 5, 6, 7
    CHECK(sim.stuck_since_ms == 0);      // timer cleared by resumed progress
    CHECK(sim.skipped_ranges.empty());   // no skip needed
}

TEST_CASE("Writer gap-skip: multiple consecutive missing iters",
          "[writer][bug6]")
{
    /* Two adjacent hung iters — fix should skip both in one go. */
    GapSkipSim sim;
    for (std::size_t i = 1; i <= 10; ++i)
        if (i != 5 && i != 6) sim.io_buf.emplace(i, 0);

    /* Drain 1..4. */
    sim.run_writer(60'000);
    REQUIRE(sim.next_flush == 5);

    /* Start the stuck timer. */
    sim.run_writer(60'000);
    REQUIRE(sim.stuck_since_ms != 0);

    /* Tick past threshold and trigger skip. */
    sim.tick(70'000);
    sim.run_writer(60'000);
    CHECK(sim.next_flush == 11);     // jumped from 5 to 7, then drained 7..10
    CHECK(sim.skipped_ranges.size() == 1);
    CHECK(sim.io_buf.empty());
}

TEST_CASE("Writer gap-skip: warning text format is grep-friendly",
          "[writer][bug6][log]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto path = std::filesystem::temp_directory_path() /
        ("tadah_gap_skip_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    log.warning(
        std::string("HPO writer: gap-skip — advancing next_flush_ from ") +
        std::to_string(3459) + " to " + std::to_string(3460) +
        " (1 hung iter slot whose eval thread did not return within 60 s; "
        "per-eval logs lost)");
    log.finish("TEST_DONE");

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::ostringstream ss; ss << in.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("gap-skip") != std::string::npos);
    CHECK(body.find("3459") != std::string::npos);
    CHECK(body.find("3460") != std::string::npos);
    CHECK(body.find("hung iter slot") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

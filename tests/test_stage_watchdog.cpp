#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_host.h>   // pulls in stage_watchdog namespace via impl

#include <chrono>
#include <thread>

/* Bug 7 regression — the per-stage watchdog tracks each worker thread's
 * current pipeline stage so the writer's gap-skip warning can name
 * which call (train / predict / lammps / …) is hanging. Two surfaces:
 *
 *   1. enter() / leave(iter, stage, log) — log warning if leave()
 *      finds the stage took longer than the threshold.
 *   2. snapshot_stuck(min_age_ms) — read by the writer; returns a
 *      grep-friendly string describing any thread whose stage has
 *      been running longer than min_age_ms. */

namespace sw = tadah::hpo::stage_watchdog;

namespace {
struct Captured { std::vector<std::string> msgs; };
auto make_log = [](Captured& c) {
    return [&c](const std::string& m) { c.msgs.push_back(m); };
};
}

TEST_CASE("stage_watchdog: enter/leave with quick stage emits no warning",
          "[stage-watchdog][bug7]")
{
    Captured c;
    sw::enter(/*iter=*/42, "train");
    /* No sleep — leave immediately. */
    const auto took = sw::leave(42, "train", make_log(c));
    CHECK(took >= 0);
    CHECK(c.msgs.empty());
}

TEST_CASE("stage_watchdog: snapshot_stuck reports threads currently in a stage",
          "[stage-watchdog][bug7]")
{
    sw::enter(/*iter=*/100, "lammps");
    /* Active stage with age >= 0; min_age_ms=0 must include it. */
    const auto snap = sw::snapshot_stuck(0);
    CHECK(snap.find("iter=100") != std::string::npos);
    CHECK(snap.find("stage='lammps'") != std::string::npos);

    /* min_age_ms = 1 hour — the entry is far too young, snapshot empty. */
    const auto snap_old = sw::snapshot_stuck(3'600'000);
    CHECK(snap_old.empty());

    /* Cleanup — leave the stage so subsequent tests start clean. */
    Captured c;
    sw::leave(100, "lammps", make_log(c));
}

TEST_CASE("stage_watchdog: snapshot_stuck excludes idle threads",
          "[stage-watchdog][bug7]")
{
    /* enter then leave makes the entry idle (stage = nullptr). */
    sw::enter(7, "train");
    Captured c;
    sw::leave(7, "train", make_log(c));

    const auto snap = sw::snapshot_stuck(0);
    CHECK(snap.find("iter=7") == std::string::npos);   // not reported
}

TEST_CASE("stage_watchdog: separate threads each get their own slot",
          "[stage-watchdog][bug7]")
{
    /* Two worker threads, each in a different stage. snapshot_stuck
     * (min_age=0) should report both. */
    std::atomic<bool> ready_a{false}, ready_b{false}, release{false};
    std::thread t_a([&]{
        sw::enter(/*iter=*/10, "train");
        ready_a = true;
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Captured c; sw::leave(10, "train", make_log(c));
    });
    std::thread t_b([&]{
        sw::enter(/*iter=*/20, "predict_validate");
        ready_b = true;
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Captured c; sw::leave(20, "predict_validate", make_log(c));
    });
    while (!(ready_a && ready_b))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    const auto snap = sw::snapshot_stuck(0);
    CHECK(snap.find("iter=10") != std::string::npos);
    CHECK(snap.find("stage='train'") != std::string::npos);
    CHECK(snap.find("iter=20") != std::string::npos);
    CHECK(snap.find("stage='predict_validate'") != std::string::npos);

    release = true;
    t_a.join();
    t_b.join();
}

TEST_CASE("stage_watchdog: threshold env-default behaviour is sensible",
          "[stage-watchdog][bug7]")
{
    /* Default is 30 000 ms when TADAH_HPO_STAGE_WARN_MS is unset, but
     * the env capture is one-time at first call so we cannot easily
     * assert a specific value here. We only verify the function
     * returns a non-negative number. */
    const auto t = sw::threshold_ms();
    CHECK(t >= 0);
}

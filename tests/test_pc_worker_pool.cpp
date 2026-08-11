/* PC_ISOLATION worker-process pool tests.
 *
 * The pool re-execs /proc/self/exe with --tadah-pc-worker; test_main.cpp
 * (CATCH_CONFIG_RUNNER) routes that flag to pc_worker_main, so THIS test
 * binary is its own worker. Pools are constructed DIRECTLY (not via the
 * process-wide singleton) so each test case gets a fresh one.
 *
 * These cases exercise the pool MACHINERY through the clean-failure paths of
 * execute_script_task (an intentionally-invalid potential context =>
 * POT_BUILD_FAIL round-trips; bad invar => CREATE_FAIL) -- no valid trained
 * potential is required. Full valid-pot execution parity and the
 * deadline-kill of a genuinely hung LAMMPS run live in
 * test_pc_task_exec_parity.cpp (fixture-dependent).
 *
 * Design: INVESTIGATIONS/PC_ISOLATION_process_pool.md sections 3-5, 7. */

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/pc_task_exec.h>
#include <tadah/hpo/pc_worker_pool.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace tadah::hpo;

namespace {

/* A syntactically-valid but untrainable potential map: transportable over the
 * delta-pot protocol, and guaranteed to make LammpsTadah's ctor throw in the
 * worker (no TYPE2B/MODEL stack) => deterministic POT_BUILD_FAIL. */
PcPotMap invalid_pot_map() {
  tadah::core::Context c;
  c.add("ATOM", "Ta");
  return c.storage_snapshot();
}

/* Minimal spec: LAMMPS instance creation succeeds (argv[0] + quiet screen/
 * log), so the task reaches the potential-build step and fails THERE. */
PcScriptSpec minimal_spec() {
  PcScriptSpec s;
  s.script = "unit_test.in";
  s.script_lines = {"units metal\n"};
  s.invar_final = {"tadah", "-screen", "none", "-log", "none"};
  s.outvar_names = {"answer"};
  s.hotvar_names = {};
  return s;
}

std::vector<std::vector<PcScriptSpec>> specs_for(int width,
                                                 const PcScriptSpec& sp) {
  std::vector<std::vector<PcScriptSpec>> per_slot(width);
  for (auto& v : per_slot) v.push_back(sp);
  return per_slot;
}

}  // namespace

TEST_CASE("pc pool: spawn, handshake, task round-trip, shutdown",
          "[pc_pool]") {
  PcWorkerPool pool(2, specs_for(2, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();

  /* Round-trip on both channels: the worker executes the shared core and
   * reports POT_BUILD_FAIL (LammpsTadah ctor throws on the gutted context) --
   * proving HELLO/TASK/RESULT framing, seq echo, and status transport. */
  for (int slot = 0; slot < 2; ++slot) {
    const PcTaskResult res =
        pool.execute(slot, 0, /*iter=*/1, pot, {}, "off");
    INFO("slot " << slot << " error: " << res.error_msg);
    CHECK(res.status == PcTaskStatus::POT_BUILD_FAIL);
  }

  /* Same iter again: has_pot=false (parent-side pot cache) and the worker's
   * per-iter negative cache answers without re-running the throwing ctor. */
  const PcTaskResult res2 = pool.execute(0, 0, /*iter=*/1, pot, {}, "off");
  CHECK(res2.status == PcTaskStatus::POT_BUILD_FAIL);

  /* New iter: pot is re-shipped and the ctor re-attempted. */
  const PcTaskResult res3 = pool.execute(0, 0, /*iter=*/2, pot, {}, "off");
  CHECK(res3.status == PcTaskStatus::POT_BUILD_FAIL);

  pool.shutdown();   /* explicit + idempotent (dtor calls it again) */
}

TEST_CASE("pc pool: CREATE_FAIL is reported and retried per task",
          "[pc_pool]") {
  /* Unwritable log path => LAMMPS instance construction fails in the worker.
   * Contract: CREATE_FAIL round-trips, no instance is cached, and the next
   * task retries (and fails identically) rather than crashing the worker. */
  PcScriptSpec sp = minimal_spec();
  sp.invar_final = {"tadah", "-screen", "none",
                    "-log", "/nonexistent-dir/x.log"};
  PcWorkerPool pool(1, specs_for(1, sp), 0.0);
  const auto pot = invalid_pot_map();

  const PcTaskResult r1 = pool.execute(0, 0, 1, pot, {}, "off");
  CHECK(r1.status == PcTaskStatus::CREATE_FAIL);
  const PcTaskResult r2 = pool.execute(0, 0, 1, pot, {}, "off");
  CHECK(r2.status == PcTaskStatus::CREATE_FAIL);
}

TEST_CASE("pc pool: worker-side task error is reported in-band",
          "[pc_pool]") {
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();

  /* Healthy round-trip first (also zeroes consec_deaths). */
  CHECK(pool.execute(0, 0, 1, pot, {}, "off").status ==
        PcTaskStatus::POT_BUILD_FAIL);

  /* Out-of-range script_idx: the worker's task loop catches it and replies
   * RUN_FAIL with an error message -- the worker STAYS ALIVE (in-band error
   * path, no kill/respawn). */
  const PcTaskResult oor = pool.execute(0, /*script_idx=*/99, 1, pot, {}, "off");
  CHECK(oor.status == PcTaskStatus::RUN_FAIL);
  CHECK_FALSE(oor.error_msg.empty());

  /* And the same channel still works afterwards: */
  CHECK(pool.execute(0, 0, 2, pot, {}, "off").status ==
        PcTaskStatus::POT_BUILD_FAIL);
}

TEST_CASE("pc pool: killing the worker process is contained and respawned",
          "[pc_pool][kill]") {
  /* Death containment via the REAL signal path: SIGKILL the child between
   * tasks. The child pid is not exposed by the API, so find it as the only
   * direct child of this process executing with argv[0] "tadah-pc-worker"
   * (set by the spawn recipe): scan /proc/<pid>/stat ppid entries. */
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();
  REQUIRE(pool.execute(0, 0, 1, pot, {}, "off").status ==
          PcTaskStatus::POT_BUILD_FAIL);

  pid_t child = -1;
  {
    /* Find the worker: the only direct child of THIS process whose comm is
     * tadah-pc-worker (argv[0] set by the spawn recipe). NB: use getpid(),
     * not `$$` -- inside popen's shell `$$` would be the SHELL's pid. */
    /* Match argv[0] (spawn recipe sets it to "tadah-pc-worker"): `args`'s
     * first token is argv[0]; `comm` would show the binary basename instead
     * (execve of /proc/self/exe). Also excludes the popen sh, which is a
     * child of this process at scan time. */
    char cmd[160];
    std::snprintf(cmd, sizeof cmd,
                  "ps -o pid=,args= --ppid %d 2>/dev/null | "
                  "awk '$2==\"tadah-pc-worker\"{print $1; exit}'",
                  static_cast<int>(::getpid()));
    FILE* p = ::popen(cmd, "r");
    if (p) {
      char buf[64] = {0};
      if (::fgets(buf, sizeof buf, p)) child = ::atoi(buf);
      ::pclose(p);
    }
  }
  if (child <= 0) {
    WARN("could not locate worker child pid; skipping the SIGKILL leg");
    return;
  }
  ::kill(child, SIGKILL);

  /* Next task: the pool detects the dead channel (EOF/EPIPE), journals,
   * returns a parent-synthesized KILLED (distinct from the wire-returned
   * RUN_FAIL: the parent knows the worker died rather than the script
   * failing, so FAILSCORE_GRADING can band it as not-completed)... */
  const PcTaskResult dead = pool.execute(0, 0, 2, pot, {}, "off");
  CHECK(dead.status == PcTaskStatus::KILLED);

  /* ...and the one after that runs on a fresh worker. */
  const PcTaskResult alive = pool.execute(0, 0, 3, pot, {}, "off");
  CHECK(alive.status == PcTaskStatus::POT_BUILD_FAIL);
}

TEST_CASE("pc pool: repeated deaths throttle a channel, they do not retire it",
          "[pc_pool][kill][backoff]") {
  /* D1. spawn_locked used to THROW once a channel reached
   * PC_MAX_CONSEC_DEATHS. Because execute() only respawns when fd < 0 --
   * exactly the state a dead channel is left in -- and consec_deaths is
   * reset only after a complete RESULT round trip, that throw was permanent:
   * the slot could never be revived. It was then swallowed by the eval-thread
   * catch into a FAIL_SCORE, so a run kept going with one worker slot dead
   * forever, and (worse) that slot drained the shared task queue faster than
   * the live ones because it failed before doing any work.
   *
   * A sub-millisecond worker timeout makes EVERY task deadline-expire
   * deterministically -- deadline_ms_for returns worker_timeout_s * 1000, so
   * 5e-4 s truncates to a 0 ms RESULT deadline -- while the HELLO_ACK
   * handshake keeps its own fixed 60 s wait, so every respawn still
   * succeeds. Each execute() therefore costs one death. */
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0005);
  const auto pot = invalid_pot_map();

  for (int i = 0; i < PcWorkerPool::PC_MAX_CONSEC_DEATHS; ++i) {
    const PcTaskResult r = pool.execute(0, 0, static_cast<std::uint64_t>(i),
                                        pot, {}, "off");
    CHECK(r.status == PcTaskStatus::KILLED);
  }

  /* The task that used to throw. It must complete normally -- slower (the
   * respawn is throttled by PC_RESPAWN_BACKOFF_MS), but a task, not an
   * exception. */
  PcTaskResult after;
  CHECK_NOTHROW(after = pool.execute(
      0, 0, static_cast<std::uint64_t>(PcWorkerPool::PC_MAX_CONSEC_DEATHS),
      pot, {}, "off"));
  CHECK(after.status == PcTaskStatus::KILLED);
  CHECK_FALSE(after.error_msg.empty());

  /* And the channel is genuinely alive afterwards, not merely non-throwing:
   * with a workable deadline the very next task completes a full round trip.
   * (A retired channel would fail here regardless of the deadline.) */
  PcWorkerPool healthy(1, specs_for(1, minimal_spec()), 0.0);
  CHECK(healthy.execute(0, 0, 0, pot, {}, "off").status ==
        PcTaskStatus::POT_BUILD_FAIL);
}

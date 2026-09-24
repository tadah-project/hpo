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
#include <iterator>
#include <set>
#include <stdexcept>
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

/* The pid of the pool's worker: the only direct child of THIS process
 * executing with argv[0] "tadah-pc-worker" (set by the spawn recipe); -1 if
 * none is found. The pid is not exposed by the API. NB: use getpid(), not
 * `$$` -- inside popen's shell `$$` would be the SHELL's pid. `args`'s first
 * token is argv[0]; `comm` would show the binary basename instead (execve
 * of /proc/self/exe). Also excludes the popen sh, which is a child of this
 * process at scan time. */
pid_t find_worker_child() {
  char cmd[160];
  std::snprintf(cmd, sizeof cmd,
                "ps -o pid=,args= --ppid %d 2>/dev/null | "
                "awk '$2==\"tadah-pc-worker\"{print $1; exit}'",
                static_cast<int>(::getpid()));
  pid_t child = -1;
  FILE* p = ::popen(cmd, "r");
  if (p) {
    char buf[64] = {0};
    if (::fgets(buf, sizeof buf, p)) child = ::atoi(buf);
    ::pclose(p);
  }
  return child > 0 ? child : -1;
}

}  // namespace

TEST_CASE("pc pool: a width below 1 is rejected", "[pc_pool]") {
  /* THREADS is at least 1 at its line; a pool built with a width below 1
   * used to be raised to one worker without a word (and then failed the
   * per-slot table check with an unrelated message). It throws, as
   * ScriptExecutor does, before any worker is spawned. */
  for (const int width : {0, -1, -8}) {
    INFO("width " << width);
    CHECK_THROWS_AS(PcWorkerPool(width, {}, 0.0), std::invalid_argument);
    CHECK_THROWS_WITH(PcWorkerPool(width, {}, 0.0),
                      "PcWorkerPool: the width must be at least 1 worker, "
                      "got " + std::to_string(width));
  }
}

TEST_CASE("pc pool: spawn, handshake, task round-trip, shutdown",
          "[pc_pool]") {
  PcWorkerPool pool(2, specs_for(2, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();

  /* Round-trip on both channels: the worker executes the shared core and
   * reports POT_BUILD_FAIL (LammpsTadah ctor throws on the gutted context) --
   * proving HELLO/TASK/RESULT framing, seq echo, and status transport. */
  for (int slot = 0; slot < 2; ++slot) {
    const PcTaskResult res =
        pool.execute(slot, 0, /*iter=*/1, pot, {}, /*script_timeout_s=*/std::nullopt);
    INFO("slot " << slot << " error: " << res.error_msg);
    CHECK(res.status == PcTaskStatus::POT_BUILD_FAIL);
  }

  /* Same iter again: has_pot=false (parent-side pot cache) and the worker's
   * per-iter negative cache answers without re-running the throwing ctor. */
  const PcTaskResult res2 = pool.execute(0, 0, /*iter=*/1, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(res2.status == PcTaskStatus::POT_BUILD_FAIL);

  /* New iter: pot is re-shipped and the ctor re-attempted. */
  const PcTaskResult res3 = pool.execute(0, 0, /*iter=*/2, pot, {}, /*script_timeout_s=*/std::nullopt);
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

  const PcTaskResult r1 = pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(r1.status == PcTaskStatus::CREATE_FAIL);
  const PcTaskResult r2 = pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(r2.status == PcTaskStatus::CREATE_FAIL);
}

TEST_CASE("pc pool: worker-side task error is reported in-band",
          "[pc_pool]") {
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();

  /* Healthy round-trip first (also zeroes consec_deaths). */
  CHECK(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::POT_BUILD_FAIL);

  /* Out-of-range script_idx: the worker's task loop catches it and replies
   * RUN_FAIL with an error message -- the worker STAYS ALIVE (in-band error
   * path, no kill/respawn). */
  const PcTaskResult oor = pool.execute(0, /*script_idx=*/99, 1, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(oor.status == PcTaskStatus::RUN_FAIL);
  CHECK_FALSE(oor.error_msg.empty());

  /* And the same channel still works afterwards: */
  CHECK(pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::POT_BUILD_FAIL);
}

TEST_CASE("pc pool: killing the worker process is contained and respawned",
          "[pc_pool][kill]") {
  /* Death containment via the REAL signal path: SIGKILL the child between
   * tasks. */
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();
  REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
          PcTaskStatus::POT_BUILD_FAIL);

  const pid_t child = find_worker_child();
  if (child <= 0) {
    WARN("could not locate worker child pid; skipping the SIGKILL leg");
    return;
  }
  ::kill(child, SIGKILL);

  /* Next task: the pool detects the dead channel (EOF/EPIPE), journals,
   * returns a parent-synthesized KILLED (distinct from the wire-returned
   * RUN_FAIL: the parent knows the worker died rather than the script
   * failing, so FAILSCORE_GRADING can band it as not-completed)... */
  const PcTaskResult dead = pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(dead.status == PcTaskStatus::KILLED);

  /* ...and the one after that runs on a fresh worker. */
  const PcTaskResult alive = pool.execute(0, 0, 3, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(alive.status == PcTaskStatus::POT_BUILD_FAIL);
}

TEST_CASE("pc pool: a worker survives the stop signals sent to its process "
          "group", "[pc_pool][signals]") {
  /* Ctrl+C, `kill -<sig> -<pgid>` and SLURM (`scancel`, `--signal=USR2@N`)
   * deliver SIGINT, SIGTERM and SIGUSR2 to every process of the group or
   * job step, the workers included. The parent stops the run itself, after
   * the evaluations in flight; a worker killed by the signal would turn the
   * evaluation it runs into a FAILSCORE (KILLED here). SIGUSR2 used to kill
   * the worker. */
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();
  REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
          PcTaskStatus::POT_BUILD_FAIL);
  const pid_t child = find_worker_child();
  REQUIRE(child > 0);

  std::size_t iter = 2;
  for (const int signo : {SIGINT, SIGTERM, SIGUSR2}) {
    INFO("signal " << signo);
    REQUIRE(::kill(child, signo) == 0);
    /* A signal that kills the worker is fatal before it can run the next
     * task, which then comes back KILLED; an ignored one changes nothing. */
    const PcTaskResult r = pool.execute(0, 0, iter++, pot, {}, /*script_timeout_s=*/std::nullopt);
    CHECK(r.status == PcTaskStatus::POT_BUILD_FAIL);
  }
  /* Still the same worker process. */
  CHECK(find_worker_child() == child);
}

TEST_CASE("pc pool: workers ignore exactly the stop signals the parent "
          "handles", "[pc_pool][signals]") {
  /* kPcWorkerIgnoredSignals must follow core::SignalHandler: a stop signal
   * the parent handles but the workers do not ignore kills the evaluations
   * in flight when it reaches the whole group. The handled set comes from a
   * fresh process (test_main.cpp's probe); inside popen's shell
   * /proc/self/exe would be the shell, so resolve it here. */
  char exe[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
  REQUIRE(n > 0);
  const std::string cmd =
      std::string("'") + exe + "' --tadah-signal-handler-probe";
  FILE* p = ::popen(cmd.c_str(), "r");
  REQUIRE(p != nullptr);
  std::set<int> handled;
  char line[32];
  while (::fgets(line, sizeof line, p)) handled.insert(std::atoi(line));
  REQUIRE(::pclose(p) == 0);

  const std::set<int> ignored(std::begin(kPcWorkerIgnoredSignals),
                              std::end(kPcWorkerIgnoredSignals));
  CHECK(handled == ignored);
  CHECK(ignored == std::set<int>{SIGINT, SIGTERM, SIGUSR2});
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
                                        pot, {}, /*script_timeout_s=*/std::nullopt);
    CHECK(r.status == PcTaskStatus::KILLED);
  }

  /* The task that used to throw. It must complete normally -- slower (the
   * respawn is throttled by PC_RESPAWN_BACKOFF_MS), but a task, not an
   * exception. */
  PcTaskResult after;
  CHECK_NOTHROW(after = pool.execute(
      0, 0, static_cast<std::uint64_t>(PcWorkerPool::PC_MAX_CONSEC_DEATHS),
      pot, {}, /*script_timeout_s=*/std::nullopt));
  CHECK(after.status == PcTaskStatus::KILLED);
  CHECK_FALSE(after.error_msg.empty());

  /* And the channel is genuinely alive afterwards, not merely non-throwing:
   * with a workable deadline the very next task completes a full round trip.
   * (A retired channel would fail here regardless of the deadline.) */
  PcWorkerPool healthy(1, specs_for(1, minimal_spec()), 0.0);
  CHECK(healthy.execute(0, 0, 0, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::POT_BUILD_FAIL);
}

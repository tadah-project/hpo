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
 * test_pc_task_exec_parity.cpp (fixture-dependent). */

#include "pc_process_workers.h"  // before catch2/catch.hpp
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pc_task_exec.h>
#include <tadah/hpo/pc_worker_pool.h>

#include "scratch_tmpdir.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace tadah::hpo;

/* The MPI build (TADAH_HPO_MPI) has no worker-process pool: hpo_run()
 * refuses an explicit PC_ISOLATION PROCESS and runs THREAD by default, and
 * the worker entry refuses to start (pc_worker_main exits 2: a fork+exec'd
 * child of an MPI parent must not touch MPI). There every case that builds
 * a pool is Skipped (SKIP_WITHOUT_PC_PROCESS_WORKERS(), pc_process_workers.h)
 * -- also those whose workers are test_main.cpp's stand-ins, which do not
 * run pc_worker_main, and the width check, whose pools throw before they
 * spawn: the pool is code the MPI build never runs, the same in both
 * builds. "pc pool: in the MPI build the worker refuses to start" pins the
 * refusal the spawning cases would otherwise fail on; in the serial build
 * the worker of a case that does not call the macro refuses as well. */

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

/* True when `pid` is a live (not zombie) direct child of this process that
 * runs as a pool worker: argv[0] "tadah-pc-worker", set by the spawn recipe.
 * Read from /proc (the pool itself is Linux-only: /proc/self/exe, prctl).
 * It checks that PcWorkerPool::worker_pid() names the worker process the OS
 * has, not merely a number the pool keeps. */
bool is_live_worker_child(pid_t pid) {
  if (pid <= 0) return false;
  const std::string proc = "/proc/" + std::to_string(pid);
  std::ifstream stat(proc + "/stat");
  const std::string line((std::istreambuf_iterator<char>(stat)),
                         std::istreambuf_iterator<char>());
  /* "<pid> (<comm>) <state> <ppid> ...": comm may hold blanks and ')'. */
  const auto rp = line.rfind(')');
  if (rp == std::string::npos) return false;
  std::istringstream rest(line.substr(rp + 1));
  char state = 0;
  long ppid = -1;
  if (!(rest >> state >> ppid)) return false;
  if (state == 'Z' || state == 'X' || ppid != static_cast<long>(::getpid()))
    return false;
  std::ifstream cmdline(proc + "/cmdline");
  std::string argv0;
  std::getline(cmdline, argv0, '\0');
  return argv0 == "tadah-pc-worker";
}

/* The HPO journal of one case, in a scratch file; `text()` finishes it and
 * returns it (HPOLog is a process-wide singleton). */
struct Journal {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("tadah_pc_pool_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".log");
  Journal() {
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("reopen-for-test");
    log.reset_for_test();
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());
  }
  std::string text() {
    HPOLog::instance().finish("test-end");
    std::ifstream f(path);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
  }
  ~Journal() {
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("test-end");
    log.reset_for_test();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

/* Sets an environment variable for the life of one scope; the workers a
 * pool spawns inherit it. */
struct ScopedEnv {
  std::string name;
  explicit ScopedEnv(std::string n, const char* value) : name(std::move(n)) {
    REQUIRE(::setenv(name.c_str(), value, 1) == 0);
  }
  ~ScopedEnv() { ::unsetenv(name.c_str()); }
};

}  // namespace

TEST_CASE("pc pool: a width below 1 is rejected", "[pc_pool]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* THREADS is at least 1 at its line; a pool built with a width below 1
   * used to be raised to one worker without a word (and then failed the
   * per-slot table check with an unrelated message). It throws, as
   * ScriptExecutor does, before any worker is spawned.
   * Skipped in the MPI build, where it passed: it builds pools, code that
   * build never runs (the same in both builds). */
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
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
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
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
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
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
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
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* Death containment via the REAL signal path: SIGKILL the child between
   * tasks.
   *
   * Pins: the pool names its worker (worker_pid(0), a live direct child of
   * this process); after SIGKILL the next task comes back KILLED and the
   * one after it runs (POT_BUILD_FAIL) on a new worker, with another pid.
   * There is no path that skips the kill: a pid that cannot be had fails.
   *
   * Before the fix: the case looked for the pid with `ps --ppid | awk` and,
   * where that found none (no procps ps or no awk, as in a minimal
   * container), WARNed "could not locate worker child pid; skipping the
   * SIGKILL leg" and passed after the first round trip; now it does not
   * compile (PcWorkerPool::worker_pid is absent). */
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();
  REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
          PcTaskStatus::POT_BUILD_FAIL);

  const pid_t child = pool.worker_pid(0);
  REQUIRE(child > 0);
  REQUIRE(is_live_worker_child(child));
  REQUIRE(::kill(child, SIGKILL) == 0);

  /* Next task: the pool detects the dead channel (EOF/EPIPE), journals,
   * returns a parent-synthesized KILLED (distinct from the wire-returned
   * RUN_FAIL: the parent knows the worker died rather than the script
   * failing, so FAILSCORE_GRADING can band it as not-completed)... */
  const PcTaskResult dead = pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(dead.status == PcTaskStatus::KILLED);

  /* ...and the one after that runs on a fresh worker. */
  const PcTaskResult alive = pool.execute(0, 0, 3, pot, {}, /*script_timeout_s=*/std::nullopt);
  CHECK(alive.status == PcTaskStatus::POT_BUILD_FAIL);
  const pid_t respawned = pool.worker_pid(0);
  CHECK(respawned > 0);
  CHECK(respawned != child);
  CHECK(is_live_worker_child(respawned));
}

TEST_CASE("pc pool: worker_pid names the live worker of a slot and changes "
          "on respawn", "[pc_pool][kill]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* Pins: PcWorkerPool::worker_pid(slot) is the pid of the worker process
   * the pool holds for that slot: one per slot from the constructor's eager
   * spawn, each a live direct child of this process running as
   * "tadah-pc-worker"; the same after a task; still the killed one until a
   * task finds it dead, -1 once that task has reaped it, a new pid after the
   * next task respawns it, the other slot's untouched throughout; -1 for
   * every slot after shutdown(). A slot outside 0..width-1 throws
   * std::out_of_range naming the slot and the width.
   *
   * Before the fix: does not compile (PcWorkerPool::worker_pid is absent;
   * the pid was not exposed at all).
   * Skipped in the MPI build. Before it skipped there it failed there: the
   * pool's eager spawn threw "pc-worker 1: handshake failed: pcwire: recv
   * failed: Connection reset by peer" (the worker refused to start). */
  PcWorkerPool pool(2, specs_for(2, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();

  const pid_t w0 = pool.worker_pid(0);
  const pid_t w1 = pool.worker_pid(1);
  REQUIRE(w0 > 0);
  REQUIRE(w1 > 0);
  CHECK(w0 != w1);
  CHECK(is_live_worker_child(w0));
  CHECK(is_live_worker_child(w1));

  REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
          PcTaskStatus::POT_BUILD_FAIL);
  CHECK(pool.worker_pid(0) == w0);

  REQUIRE(::kill(w0, SIGKILL) == 0);
  /* The pool learns of the death from the next task on that slot. */
  CHECK(pool.worker_pid(0) == w0);
  CHECK(pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::KILLED);
  CHECK(pool.worker_pid(0) == -1);
  CHECK(pool.worker_pid(1) == w1);

  CHECK(pool.execute(0, 0, 3, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::POT_BUILD_FAIL);
  const pid_t w0_new = pool.worker_pid(0);
  CHECK(w0_new > 0);
  CHECK(w0_new != w0);
  CHECK(is_live_worker_child(w0_new));
  CHECK(pool.worker_pid(1) == w1);

  for (const int bad : {2, -1}) {
    INFO("slot " << bad);
    CHECK_THROWS_AS(pool.worker_pid(bad), std::out_of_range);
    CHECK_THROWS_WITH(pool.worker_pid(bad),
                      "PcWorkerPool::worker_pid: slot " + std::to_string(bad)
                      + " is outside 0-1 (the pool has 2 slots)");
  }

  pool.shutdown();
  CHECK(pool.worker_pid(0) == -1);
  CHECK(pool.worker_pid(1) == -1);
}

TEST_CASE("pc pool: the journal names the pid of a worker that died",
          "[pc_pool][kill][hpolog]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* A worker killed from outside (the OOM killer, a crash inside LAMMPS) is
   * journalled with its pid, which is what the kernel log names.
   *
   * Pins: after SIGKILL of the worker of slot 0, the task that finds it
   * dead journals "pc-worker 1 (pid <its pid>) killed/died: " and, on the
   * same line, the wait status "wait: signal 9".
   *
   * Before the fix: does not compile (PcWorkerPool::worker_pid is absent);
   * the warning read "pc-worker 1 killed/died: ...; wait: signal 9
   * (Killed); ...", without the pid.
   * Skipped in the MPI build. Before it skipped there it failed there, as
   * the case above did ("handshake failed: ... Connection reset by peer"). */
  Journal journal;
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const auto pot = invalid_pot_map();
  REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
          PcTaskStatus::POT_BUILD_FAIL);
  const pid_t child = pool.worker_pid(0);
  REQUIRE(child > 0);
  REQUIRE(::kill(child, SIGKILL) == 0);
  CHECK(pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::KILLED);
  pool.shutdown();

  const std::string text = journal.text();
  INFO(text);
  const std::string head =
      "pc-worker 1 (pid " + std::to_string(child) + ") killed/died: ";
  const auto at = text.find(head);
  REQUIRE(at != std::string::npos);
  const std::string line = text.substr(at, text.find('\n', at) - at);
  CHECK_THAT(line, Catch::Contains("; wait: signal 9 ("));
}

TEST_CASE("pc pool: a worker that dies inside its task is journalled with "
          "the end of its channel", "[pc_pool][kill][hpolog]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* A worker that dies while it runs a script (the kernel's out-of-memory
   * killer, a crash inside LAMMPS) has read its whole TASK frame, and the
   * parent, waiting for RESULT, finds the channel closed. test_main.cpp's
   * stand-in worker (TADAH_TEST_PC_WORKER_DIE_IN_TASK in its environment)
   * reads the first TASK and kills itself with SIGKILL.
   *
   * Pins: that task comes back KILLED, and the journal line is exactly
   * "pc-worker 1 (pid <its pid>) killed/died: RESULT recv failed: pcwire:
   * peer closed (EOF); wait: signal 9 (Killed); consecutive deaths: 1", the
   * line Nested Fitting Configuration (PC_ISOLATION) shows.
   *
   * Before the fix: passes as well (a pin of the line the docs show; they
   * showed "RESULT recv failed: pcwire: recv failed: Connection reset by
   * peer", the reason the journal gives for a worker killed between tasks
   * that the next TASK reaches, as in "the journal names the pid of a
   * worker that died" above).
   * Skipped in the MPI build, where it passed: its stand-in worker does not
   * run pc_worker_main, but the pool it builds is code that build never
   * runs (the same in both builds). */
  Journal journal;
  ScopedEnv die("TADAH_TEST_PC_WORKER_DIE_IN_TASK", "1");
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  const pid_t child = pool.worker_pid(0);
  REQUIRE(child > 0);
  CHECK(pool.execute(0, 0, 1, invalid_pot_map(), {},
                     /*script_timeout_s=*/std::nullopt).status ==
        PcTaskStatus::KILLED);
  pool.shutdown();

  const std::string text = journal.text();
  INFO(text);
  const std::string head =
      "pc-worker 1 (pid " + std::to_string(child) + ") killed/died: ";
  const auto at = text.find(head);
  REQUIRE(at != std::string::npos);
  CHECK(text.substr(at, text.find('\n', at) - at) ==
        head + "RESULT recv failed: pcwire: peer closed (EOF); wait: "
               "signal 9 (Killed); consecutive deaths: 1");
}

TEST_CASE("pc pool: a HELLO_ACK that names another pid fails the handshake",
          "[pc_pool]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* The worker reports its pid in HELLO_ACK; the pool forked it and knows
   * the pid. test_main.cpp's stand-in worker (TADAH_TEST_PC_WORKER_WRONG_ACK_PID
   * in its environment) writes its own pid <p>, the one the pool forked, to
   * the file the variable names, and answers HELLO with <p + 1>.
   *
   * Pins: the pool's constructor (its eager spawn) throws std::runtime_error
   * "pc-worker 1: handshake failed: HELLO_ACK names pid <p + 1>, but the
   * worker was forked as pid <p>", exactly, with <p> from that file.
   *
   * Before the fix: the pid in HELLO_ACK was read and discarded, so the
   * pool was built and nothing was thrown. (This file does not compile
   * before the fix: the cases above call worker_pid.) The case first
   * matched the message against "... names pid [0-9]+, but the worker was
   * forked as pid [0-9]+", which a message naming the forked pid twice
   * passed as well.
   * Skipped in the MPI build, where it passed: its stand-in worker does not
   * run pc_worker_main, but the pool it builds is code that build never
   * runs (the same in both builds). */
  const std::filesystem::path pid_file =
      std::filesystem::temp_directory_path() /
      ("tadah_pc_pool_ack_pid_" + std::to_string(::getpid()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::string what;
  bool threw = false;
  {
    ScopedEnv lie("TADAH_TEST_PC_WORKER_WRONG_ACK_PID", pid_file.c_str());
    try {
      PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
    } catch (const std::runtime_error& e) {
      threw = true;
      what = e.what();
    }
  }
  long forked = -1;
  {
    std::ifstream f(pid_file);
    f >> forked;
  }
  std::error_code ec;
  std::filesystem::remove(pid_file, ec);
  REQUIRE(threw);
  REQUIRE(forked > 0);
  CHECK(what == "pc-worker 1: handshake failed: HELLO_ACK names pid "
                    + std::to_string(forked + 1)
                    + ", but the worker was forked as pid "
                    + std::to_string(forked));
}

TEST_CASE("pc pool: a killed worker leaves no script file behind",
          "[pc_pool][kill][tempfile]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* A worker renders each script of its table once, to a file in the
   * temporary directory, and runs it through LAMMPS's Input::file(). A
   * worker killed at its hard deadline (SIGKILL), crashing inside LAMMPS,
   * or killed with its parent (PR_SET_PDEATHSIG) runs no destructor, and
   * used to leave that file behind: tadah_pc_<worker pid>_0.in, one per
   * full ctest run from the SIGKILL case above. */
  hpo_test::ScratchTmpdir tmp("pcpool");
  {
    PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
    const auto pot = invalid_pot_map();
    /* The first task renders the script in the worker. */
    REQUIRE(pool.execute(0, 0, 1, pot, {}, /*script_timeout_s=*/std::nullopt)
                .status == PcTaskStatus::POT_BUILD_FAIL);
    CHECK(tmp.count("tadah_pc_") == 0);

    const pid_t child = pool.worker_pid(0);
    REQUIRE(child > 0);
    REQUIRE(::kill(child, SIGKILL) == 0);
    CHECK(pool.execute(0, 0, 2, pot, {}, /*script_timeout_s=*/std::nullopt)
              .status == PcTaskStatus::KILLED);
    /* A fresh worker renders the script again. */
    CHECK(pool.execute(0, 0, 3, pot, {}, /*script_timeout_s=*/std::nullopt)
              .status == PcTaskStatus::POT_BUILD_FAIL);
    pool.shutdown();
  }
  CHECK(tmp.count("tadah_pc_") == 0);
}

TEST_CASE("pc pool: a worker survives the stop signals sent to its process "
          "group", "[pc_pool][signals]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
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
  const pid_t child = pool.worker_pid(0);
  REQUIRE(is_live_worker_child(child));

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
  CHECK(pool.worker_pid(0) == child);
  CHECK(is_live_worker_child(child));
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
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  /* spawn_locked used to THROW once a channel reached
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

#ifndef TADAH_HPO_MPI
namespace {
/* `cmd` run by the shell: its exit status and what it wrote to its
 * standard output (the command redirects its standard error there). */
struct CommandRun {
  int status = -1;
  std::string out;
};
CommandRun run_command(const std::string& cmd) {
  CommandRun r;
  FILE* p = ::popen(cmd.c_str(), "r");
  REQUIRE(p != nullptr);
  char buf[256];
  while (::fgets(buf, sizeof buf, p)) r.out += buf;
  r.status = ::pclose(p);
  return r;
}
/* This test binary's path: inside popen's shell /proc/self/exe would be
 * the shell. */
std::string this_executable() {
  char exe[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
  REQUIRE(n > 0);
  return exe;
}
const char* const kWorkerRefusal =
    "test_hpo --tadah-pc-worker: refused: the test case that started this "
    "PC_ISOLATION PROCESS worker did not call "
    "SKIP_WITHOUT_PC_PROCESS_WORKERS() first (tests/pc_process_workers.h); "
    "the MPI build (TADAH_HPO_MPI) has no such workers, and this worker "
    "refuses to start as its worker does\n";
}  // namespace

/* The two halves of the next case's last check, run by it in one child
 * test_hpo process in this order (hidden: CTest does not register them).
 * The first calls SKIP_WITHOUT_PC_PROCESS_WORKERS() and builds a pool; the
 * second, which does not, builds one too and has its worker refused. */
TEST_CASE("pc pool gate: a case that calls SKIP_WITHOUT_PC_PROCESS_WORKERS() "
          "starts its workers", "[.pc_pool_gate]") {
  SKIP_WITHOUT_PC_PROCESS_WORKERS();
  PcWorkerPool pool(1, specs_for(1, minimal_spec()), 0.0);
  CHECK(pool.worker_pid(0) > 0);
}

TEST_CASE("pc pool gate: the next case, which does not call it, has its "
          "workers refused", "[.pc_pool_gate]") {
  CHECK_THROWS_WITH(PcWorkerPool(1, specs_for(1, minimal_spec()), 0.0),
                    Catch::StartsWith("pc-worker 1: handshake failed: "));
}

/* Pins: in the serial build a PC_ISOLATION PROCESS worker of test_hpo that
 * a test case starts without calling SKIP_WITHOUT_PC_PROCESS_WORKERS()
 * first refuses to start, as every PROCESS worker of the MPI build does:
 *   - `test_hpo --tadah-pc-worker`, started by this case (which does not
 *     call it), exits with status 2 and writes exactly the refusal
 *     "test_hpo --tadah-pc-worker: refused: the test case that started
 *     this PC_ISOLATION PROCESS worker did not call ...";
 *   - one child test_hpo running the two "pc pool gate:" cases above, in
 *     that order, passes both: the pool of the case that calls the macro
 *     starts its worker, and the next case's pool fails its handshake
 *     ("pc-worker 1: handshake failed: ...") -- the call allows the workers
 *     of its own case only -- and the child's output has the refusal.
 * So a case that starts test_hpo PROCESS workers without the call -- a
 * pool built in a helper, a copy-initialised or std::optional pool, an
 * hpo_run whose PC_LAMMPS lines run under PROCESS by request or by the
 * serial build's default -- fails in the serial build as in the MPI build.
 * Before the fix: fails -- the worker started (pc_worker_main, exit status
 * 3 without its channel: "tadah-pc-worker: handshake failed: pcwire: recv
 * failed: Bad file descriptor"), and the second gate case's pool was built,
 * so the child failed. Not in the MPI build, whose workers all refuse. */
TEST_CASE("pc pool: in the serial build the worker of a case that did not "
          "call SKIP_WITHOUT_PC_PROCESS_WORKERS() refuses to start",
          "[pc_pool]") {
  const std::string exe = this_executable();
  {
    /* Without its channel (fd 3 closed), as the pool would never start it:
     * the refusal comes first. */
    const CommandRun w =
        run_command("'" + exe + "' --tadah-pc-worker 3<&- 2>&1");
    INFO(w.out);
    REQUIRE(WIFEXITED(w.status));
    CHECK(WEXITSTATUS(w.status) == 2);
    CHECK(w.out == kWorkerRefusal);
  }
  const CommandRun gate =
      run_command("'" + exe + "' '[pc_pool_gate]' --order decl 2>&1");
  INFO(gate.out);
  REQUIRE(WIFEXITED(gate.status));
  CHECK(WEXITSTATUS(gate.status) == 0);
  CHECK_THAT(gate.out, Catch::Contains("in 2 test cases"));
  CHECK_THAT(gate.out, Catch::Contains(kWorkerRefusal));
}
#endif

#ifdef TADAH_HPO_MPI
/* Pins: in the MPI build `test_hpo --tadah-pc-worker` (the worker entry
 * the pool re-execs, routed by test_main.cpp as tadah's main() routes it)
 * exits with status 2 and says "tadah --tadah-pc-worker: not supported in
 * the MPI build (TADAH_HPO_MPI)" on stderr, without touching MPI: the
 * reason the spawning cases above are Skipped in this build.
 * Before the fix: passes as well (a pin; the case is new with the MPI
 * build's suite). */
TEST_CASE("pc pool: in the MPI build the worker refuses to start",
          "[pc_pool]") {
  char exe[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
  REQUIRE(n > 0);
  const std::string cmd =
      std::string("'") + exe + "' --tadah-pc-worker 2>&1";
  FILE* p = ::popen(cmd.c_str(), "r");
  REQUIRE(p != nullptr);
  std::string out;
  char buf[256];
  while (::fgets(buf, sizeof buf, p)) out += buf;
  const int status = ::pclose(p);
  INFO(out);
  REQUIRE(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 2);
  CHECK(out == "tadah --tadah-pc-worker: not supported in the MPI build "
               "(TADAH_HPO_MPI)\n");
}
#endif

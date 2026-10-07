#pragma once
#ifndef TADAH_HPO_PC_WORKER_POOL_H
#define TADAH_HPO_PC_WORKER_POOL_H

/* PC_ISOLATION worker-process pool (design:
 * INVESTIGATIONS/PC_ISOLATION_process_pool.md).
 *
 * Executes pc_lammps script tasks in PERSISTENT CHILD PROCESSES (one per
 * fan-out slot, re-exec'd from this same binary with the hidden
 * `--tadah-pc-worker` flag) so that concurrent LAMMPS instances live in
 * separate address spaces and cannot race on LAMMPS process-globals
 * (thermo.cpp fmtbuf et al. -- the `free(): invalid pointer` class).
 *
 * The parent keeps ALL scheduling: ScriptExecutor threads still drive the
 * fan-out; in PROCESS mode each pool thread becomes a thin proxy that ships
 * its task to worker channel `slot` and blocks for the RESULT. The worker
 * runs the SAME execute_script_task (pc_task_exec.h) the THREAD path runs
 * in-process, so the two modes cannot diverge numerically.
 *
 * Protocol (per-slot AF_UNIX/SOCK_STREAM socketpair, worker end on fd 3):
 * length-prefixed frames {u32 magic 'TPCW', u32 type, u64 payload_len},
 * payload cap 256 MiB, host-endian (same machine, same binary). All writes
 * MSG_NOSIGNAL. See pc_worker_pool.cpp for the wire codec.
 *
 *   HELLO      P->W  once per spawn: {u32 version, u32 slot, vec<ScriptSpec>}
 *   HELLO_ACK  W->P  {u32 version_echo, u32 pid}
 *   TASK       P->W  {u64 seq, u32 script_idx, u64 iter, u8 pot_kind,
 *                     [pot ops], vec<str> inject}
 *                    pot_kind: 0 = none (worker already has this iter's pot);
 *                    1 = FULL (worker resets its pot context first);
 *                    2 = DELTA (only keys that changed vs the channel's last
 *                    shipped map -- typically the OPTIM'd grids + WEIGHTS,
 *                    ~10-20x fewer tokens than the full pot). Ops:
 *                    {u64 n; per op: str key, vstr values} -- empty values =
 *                    erase. Values are PRE-TOKENIZED; the worker applies them
 *                    via Context::storage_update (deserialize()-exact raw
 *                    semantics) so apply cost is O(changed keys), not a full
 *                    ~43 KB text parse per (worker, eval).
 *   RESULT     W->P  {u64 seq, u32 script_idx, u32 status, str error_msg,
 *                     vec<str> outvar_values, vec<str> outvar_errors,
 *                     vec<str> hotvar_values, vec<str> failvar_values,
 *                     f64 create_ms, f64 run_ms}
 *   SHUTDOWN   P->W  {}
 *
 * Failure semantics: any channel error (EOF, short/bad frame, seq mismatch,
 * deadline expiry) => SIGKILL + waitpid (decoded status journaled), the
 * in-flight task returns a synthetic KILLED PcTaskResult (parent formats
 * failscore -- graded band C, "did not complete"), and the channel respawns
 * lazily on next use. >= PC_MAX_CONSEC_DEATHS consecutive deaths on one
 * channel with no successful round trip in between throttle the next
 * respawn by PC_RESPAWN_BACKOFF_MS; the channel is never permanently
 * retired. */

#include <atomic>
#include <csignal>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <sys/types.h>   /* pid_t */

#include <tadah/hpo/pc_task_exec.h>

namespace tadah {
namespace hpo {

/* The signals a worker ignores: exactly the stop signals the parent
 * handles (core::SignalHandler::install(): SIGINT, SIGTERM, SIGUSR2). They
 * reach the workers too when they are sent to the whole process group or
 * job step -- Ctrl+C, `kill -<sig> -<pgid>`, SLURM `scancel` or
 * `--signal=USR2@<seconds>` -- and a worker that died of one would turn
 * the evaluation it is running into a FAILSCORE. The parent stops the
 * run: the evaluations in flight finish and are recorded, then the
 * workers are ended (SHUTDOWN frame / EOF). */
inline constexpr int kPcWorkerIgnoredSignals[] = {SIGINT, SIGTERM, SIGUSR2};

/* The hard deadline, in milliseconds, after which a PROCESS worker running
 * one script task is killed; -1 for none (THREAD-mode parity).
 *   - worker_timeout_s > 0 (PC_WORKER_TIMEOUT / TADAH_PC_WORKER_TIMEOUT):
 *     that, for every script;
 *   - otherwise derived from the script's own soft limit t, in seconds
 *     (PC_SCRIPT_TIMEOUT or --maxtime, parsed by parse_script_timeout):
 *     max(3t, t + 120 s), which covers multi-section scripts (`timer
 *     timeout` re-arms after every `clear`) and the time `timer timeout`
 *     cannot bound (create, pot build, neighbour lists);
 *   - no deadline when t is off/unlimited (nullopt).
 * A t <= 0 throws std::invalid_argument: parse_script_timeout rejects it
 * at its line, so it never names a deadline. So does a deadline that does
 * not fit in a long of milliseconds (pc_worker_deadline_fits); the parsers
 * of the three timeouts reject it at their line.
 * The seconds come from the parsed time spec: the string used to go
 * through strtod, which read "2:00:00" as 2 s and "00:05:00" as 0 (no
 * deadline). The deadline may exceed INT_MAX ms: the wire's readn() polls
 * it in slices (poll() takes an int). */
long pc_worker_deadline_ms(double worker_timeout_s,
                           std::optional<double> script_timeout_s);

/* True when the deadline pc_worker_deadline_ms() derives from these
 * timeouts is none or fits in a long of milliseconds (below 2^63 ms:
 * about 9.2e15 s for PC_WORKER_TIMEOUT, about 3.07e15 s for a script
 * timeout t, whose deadline is max(3t, t + 120 s)). */
bool pc_worker_deadline_fits(double worker_timeout_s,
                             std::optional<double> script_timeout_s);

/* Raw key -> value-tokens map of a trial potential (Context::storage_snapshot
 * shape). The parent snapshots it once per eval; each channel diffs it
 * against the last map it shipped. */
using PcPotMap = std::map<std::string, std::vector<std::string>>;

/* Immutable per-(slot, sim) execution statics, shipped once per spawn in
 * HELLO. invar_final is already .t<slot>-suffixed (make_worker_invar);
 * script_lines already carry the parse-time `timer timeout` injection. */
struct PcScriptSpec {
  std::string script;                      /* name (cache key + logging)   */
  std::vector<std::string> script_lines;
  std::vector<std::string> invar_final;
  std::vector<std::string> outvar_names;
  std::vector<std::string> hotvar_names;
  std::vector<std::string> failvar_names;  /* graded-FAILSCORE Tier 2 (v2) */
};

class PcWorkerPool {
public:
  /* v2: failvars; v3: the outvars' read-back errors (outvar_errors) */
  static constexpr std::uint32_t PROTOCOL_VERSION = 3;
  /* Consecutive deaths on one channel with no successful round trip in
   * between. Reaching this throttles the next respawn; it never retires the
   * channel -- a dead slot in a shared-queue pool drains tasks faster than
   * the live ones and would convert a local fault into a whole-run failure. */
  static constexpr int PC_MAX_CONSEC_DEATHS = 3;
  static constexpr long PC_RESPAWN_BACKOFF_MS = 5'000;

  /* per_slot_specs[slot][sim] -- one spec table per worker (slots differ only
   * in invar_final's .t<slot> suffix). worker_timeout_s > 0 overrides the
   * per-sim deadline formula globally; 0 = derive from each sim's maxtime. */
  PcWorkerPool(int width,
               std::vector<std::vector<PcScriptSpec>> per_slot_specs,
               double worker_timeout_s);
  ~PcWorkerPool();

  PcWorkerPool(const PcWorkerPool&) = delete;
  PcWorkerPool& operator=(const PcWorkerPool&) = delete;

  /* Execute one script task on worker channel `slot` (the ScriptExecutor
   * worker id -- 1:1 with channels). pot_map = the trial potential's raw
   * storage snapshot, taken ONCE per eval by the caller; the channel ships it
   * only when it has not seen `iter` yet -- as a DELTA vs the last map it
   * shipped (FULL on first contact / after a respawn). script_timeout_s is
   * the sim's soft limit in seconds (LammpsSim::maxtime_seconds; nullopt =
   * off) for the deadline policy (pc_worker_deadline_ms). Thread-safe
   * per channel (a mutex serializes the executor's nested-inline wid==0
   * path). Throws after PC_MAX_CONSEC_DEATHS consecutive worker deaths. */
  PcTaskResult execute(int slot, std::uint32_t script_idx,
                       std::uint64_t iter,
                       const PcPotMap& pot_map,
                       const std::vector<std::string>& inject_lines,
                       std::optional<double> script_timeout_s);

  /* Graceful stop: SHUTDOWN to every live worker, then reap. Called by the
   * destructor; idempotent. */
  void shutdown();

  /* FAILPROBE gate support: SIGKILL every worker currently executing a task
   * of evaluation `iter` (and ONLY those -- other replicas' tasks on the
   * shared pool are never touched). Mis-kill-free by construction: a
   * channel's (pid, current task iter) are published atomically inside its
   * locked execute(); kill_matching only acts on channels whose mutex is
   * BUSY (try_lock fails => an execute is in flight => the atomics describe
   * exactly that task). Missed kills are possible (task just started or just
   * finished) and benign: the gated eval's results are clamped regardless,
   * and the per-task deadline still bounds the stragglers. The killed
   * channel's own execute() observes EOF and runs its normal
   * kill/reap/journal/respawn path. */
  void kill_matching(std::uint64_t iter);

  int width() const { return width_; }

private:
  struct Channel {
    pid_t pid = -1;
    int fd = -1;
    std::mutex mtx;
    std::uint64_t task_seq = 0;
    long long last_pot_iter = -1;   /* pot generation the worker holds     */
    PcPotMap last_pot_map;          /* exact map last shipped (delta base);
                                       cleared on death => next send FULL  */
    int consec_deaths = 0;
    /* Published for kill_matching (set/cleared inside the locked execute):
     * what this channel is running RIGHT NOW. -1 = idle. */
    std::atomic<long long> current_iter{-1};
    std::atomic<pid_t>     current_pid{-1};
    /* Set by kill_matching immediately before it SIGKILLs a healthy worker,
     * consumed (and cleared) by kill_reap_locked. Marks the death as
     * requested rather than faulty, so the FAILPROBE gate cannot drive a
     * good channel into the respawn backoff. */
    std::atomic<bool>      gate_killed{false};
  };

  void spawn_locked(Channel& ch, int slot);         /* fork+exec + HELLO   */
  void kill_reap_locked(Channel& ch, int slot, const char* why);

  int width_;
  double worker_timeout_s_;
  std::vector<std::vector<PcScriptSpec>> specs_;    /* [slot][sim]         */
  std::vector<std::unique_ptr<Channel>> channels_;
  std::string self_exe_;                            /* /proc/self/exe path */
  bool shut_ = false;
};

/* Process-wide singleton (mirrors init_script_executor/script_executor).
 * pc_worker_pool_init() is called from hpo.cpp AFTER the config decides
 * PROCESS mode and BEFORE any fan-out runs; enabled() gates run_one's
 * PROCESS branch. In THREAD mode the pool is never constructed. */
void pc_worker_pool_init(int width,
                         std::vector<std::vector<PcScriptSpec>> per_slot_specs,
                         double worker_timeout_s);
bool pc_worker_pool_enabled();
PcWorkerPool& pc_worker_pool();
void pc_worker_pool_shutdown();   /* explicit teardown (end of hpo_run) */

/* Worker-process entry point: the hidden `--tadah-pc-worker` flag is
 * intercepted as the FIRST statement of main() (before any MPI/CLI init);
 * the worker's socket is inherited on fd 3. Never returns to the caller's
 * normal flow (runs the task loop, then _exit()s). */
int pc_worker_main(int argc, char** argv);

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_PC_WORKER_POOL_H

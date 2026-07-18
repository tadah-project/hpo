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
 *                     vec<str> outvar_values, vec<str> hotvar_values,
 *                     f64 create_ms, f64 run_ms}
 *   SHUTDOWN   P->W  {}
 *
 * Failure semantics: any channel error (EOF, short/bad frame, seq mismatch,
 * deadline expiry) => SIGKILL + waitpid (decoded status journaled), the
 * in-flight task returns a synthetic RUN_FAIL PcTaskResult (parent formats
 * failscore -- identical to a script failure), and the channel respawns
 * lazily on next use. >= PC_MAX_CONSEC_DEATHS consecutive deaths on one
 * channel throw (lands in the eval-thread catch -> FAIL_SCORE recovery). */

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <sys/types.h>   /* pid_t */

#include <tadah/hpo/pc_task_exec.h>

namespace tadah {
namespace hpo {

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
  static constexpr std::uint32_t PROTOCOL_VERSION = 2;  /* v2: failvars */
  static constexpr int PC_MAX_CONSEC_DEATHS = 3;

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
   * shipped (FULL on first contact / after a respawn). maxtime is the sim's
   * MAXTIME string ("off" or seconds) for the deadline policy. Thread-safe
   * per channel (a mutex serializes the executor's nested-inline wid==0
   * path). Throws after PC_MAX_CONSEC_DEATHS consecutive worker deaths. */
  PcTaskResult execute(int slot, std::uint32_t script_idx,
                       std::uint64_t iter,
                       const PcPotMap& pot_map,
                       const std::vector<std::string>& inject_lines,
                       const std::string& maxtime);

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
  };

  void spawn_locked(Channel& ch, int slot);         /* fork+exec + HELLO   */
  void kill_reap_locked(Channel& ch, int slot, const char* why);
  long deadline_ms_for(const std::string& maxtime) const;

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

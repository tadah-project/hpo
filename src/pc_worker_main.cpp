/* PC_ISOLATION worker-process entry point (`tadah --tadah-pc-worker`).
 *
 * Intercepted as the FIRST statement of main() -- before CLI parsing, MPI
 * init, or SignalHandler installation. The worker is a pure executor: it
 * receives its ScriptSpec table once (HELLO), then loops TASK -> RESULT,
 * calling the SAME execute_script_task the in-process THREAD path uses.
 * No HPOLog / bench CSV / warmstart / best publication / stage_watchdog here --
 * all of that stays parent-side (design doc section 6). Socket on fd 3. */

#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/pc_worker_wire.h>
#include <tadah/hpo/pc_task_exec.h>
#include <tadah/hpo/runners.h>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>

#include <csignal>
#include <cstdio>
#include <string>
#include <vector>

#include <unistd.h>

namespace tadah {
namespace hpo {

using pcwire::FrameType;

namespace {

std::vector<PcScriptSpec> decode_specs(pcwire::Reader& r) {
  const auto n = r.u64();
  std::vector<PcScriptSpec> specs;
  specs.reserve(static_cast<std::size_t>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    PcScriptSpec s;
    s.script        = r.str();
    s.script_lines  = r.vstr();
    s.invar_final   = r.vstr();
    s.outvar_names  = r.vstr();
    s.hotvar_names  = r.vstr();
    s.failvar_names = r.vstr();   /* protocol v2 */
    specs.push_back(std::move(s));
  }
  return specs;
}

}  // namespace

int pc_worker_main(int /*argc*/, char** /*argv*/) {
#ifdef TADAH_HPO_MPI
  /* A fork+exec'd child of an MPI parent must never touch MPI, and the
   * parent hard-errors on PC_ISOLATION PROCESS in the MPI build -- reaching
   * this code means something is badly wrong. */
  std::fprintf(stderr,
               "tadah --tadah-pc-worker: not supported in the MPI build "
               "(TADAH_HPO_MPI)\n");
  return 2;
#else
  constexpr int kFd = 3;   /* inherited socketpair end (spawn recipe dup2) */

  /* The parent's stop signals go to the whole process group or job step
   * (Ctrl+C, SLURM); shutdown must stay PARENT-driven (SHUTDOWN frame /
   * EOF) so the parent's graceful shutdown isn't polluted by a wave of
   * dead-worker failscores (kPcWorkerIgnoredSignals, pc_worker_pool.h).
   * PR_SET_PDEATHSIG(SIGKILL) from the spawn recipe covers parent crashes. */
  for (const int signo : kPcWorkerIgnoredSignals)
    std::signal(signo, SIG_IGN);

  /* Minimal init (verified sufficient): the blueprint is embedded and lazily
   * parsed; model/descriptor registries are static-initialized; the only
   * state main()'s normal path would have prepared is the periodic table
   * (TadahCLI ctor) and the log level. */
  tadah::core::Logger::instance().setLogLevel(tadah::core::LogLevel::WARNING);
  tadah::core::PeriodicTable::initialize();

  std::vector<PcScriptSpec> specs;
  std::uint32_t slot = 0;

  try {
    FrameType t;
    std::string payload;
    if (!pcwire::recv_frame(kFd, t, payload, -1)) return 3;
    if (t != FrameType::HELLO) return 3;
    pcwire::Reader r(payload);
    const auto ver = r.u32();
    if (ver != PcWorkerPool::PROTOCOL_VERSION) {
      std::fprintf(stderr, "tadah-pc-worker: protocol version mismatch\n");
      return 3;
    }
    slot = r.u32();
    specs = decode_specs(r);

    pcwire::Writer ack;
    ack.u32(PcWorkerPool::PROTOCOL_VERSION);
    ack.u32(static_cast<std::uint32_t>(::getpid()));
    pcwire::send_frame(kFd, FrameType::HELLO_ACK, ack.buf);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tadah-pc-worker: handshake failed: %s\n", e.what());
    return 3;
  }
  (void)slot;   /* informational (log names already carry .t<slot>) */

  /* Persistent per-worker execution state: the process-lifetime analogue of
   * the THREAD path's `static thread_local PcSlotState` -- LAMMPS instances
   * warm across evals, one potential per iter. */
  PcSlotState<LammpsRunner> slot_state;
  tadah::core::Context pot_ctx;
  long long pot_iter = -1;

  for (;;) {
    FrameType t;
    std::string payload;
    try {
      if (!pcwire::recv_frame(kFd, t, payload, -1)) break;
    } catch (...) {
      break;   /* EOF / parent gone: exit quietly (PDEATHSIG also covers us) */
    }
    if (t == FrameType::SHUTDOWN) break;
    if (t != FrameType::TASK) break;   /* protocol desync: die; parent respawns */

    std::uint64_t seq = 0;
    std::uint32_t script_idx = 0;
    PcTaskResult res;
    try {
      pcwire::Reader r(payload);
      seq        = r.u64();
      script_idx = r.u32();
      const auto iter = r.u64();
      const auto pot_kind = r.u8();       /* 0 none, 1 full, 2 delta */
      if (pot_kind == 1 || pot_kind == 2) {
        if (pot_kind == 1)
          pot_ctx = tadah::core::Context{};   /* FULL: reset, then set-ops */
        const std::uint64_t n_ops = r.u64();
        for (std::uint64_t k = 0; k < n_ops; ++k) {
          const std::string key = r.str();
          const std::vector<std::string> vals = r.vstr();
          /* deserialize()-exact raw write; empty vals = erase. Values arrive
           * pre-tokenized, so apply cost is O(changed keys) -- no ~43 KB
           * text parse per (worker, eval). */
          pot_ctx.storage_update(key, vals);
        }
        pot_iter = static_cast<long long>(iter);
      } else if (pot_kind != 0) {
        throw std::runtime_error("bad pot_kind");
      }
      const std::vector<std::string> inject = r.vstr();

      if (script_idx >= specs.size())
        throw std::runtime_error("script_idx out of range");
      if (pot_iter != static_cast<long long>(iter))
        throw std::runtime_error("no potential for this iter (desync)");

      const PcScriptSpec& sc = specs[script_idx];
      /* Never throws: its own guard fails the task, as in THREAD mode. */
      res = execute_script_task<LammpsRunner>(
          sc.script, sc.script_lines, sc.invar_final, sc.outvar_names,
          sc.hotvar_names, sc.failvar_names, pot_ctx, iter, inject,
          slot_state);
    } catch (const std::exception& e) {
      /* A malformed TASK (bad pot_kind, script_idx out of range, no
       * potential for this iter): reported in-band, the worker lives on. */
      res = PcTaskResult{};
      res.status = PcTaskStatus::RUN_FAIL;
      res.error_msg = e.what();
    }

    try {
      pcwire::Writer w;
      w.u64(seq);
      w.u32(script_idx);
      w.u32(static_cast<std::uint32_t>(res.status));
      w.str(res.error_msg);
      w.vstr(res.outvar_values);
      w.vstr(res.outvar_errors);    /* protocol v3 */
      w.vstr(res.hotvar_values);
      w.vstr(res.failvar_values);   /* protocol v2 */
      w.f64(res.create_ms);
      w.f64(res.run_ms);
      pcwire::send_frame(kFd, FrameType::RESULT, w.buf);
    } catch (...) {
      break;   /* parent gone mid-reply */
    }
  }
  return 0;
#endif
}

}  // namespace hpo
}  // namespace tadah

#pragma once
#ifndef TADAH_HPO_PC_TASK_EXEC_H
#define TADAH_HPO_PC_TASK_EXEC_H

/* PC_LAMMPS single-task execution core (PC_ISOLATION design, Phase 2).
 *
 * This is the LAMMPS-touching heart of HPO_Host::run_lammps's per-script
 * runner (`run_one`), extracted verbatim so that EXACTLY ONE implementation
 * is shared by:
 *   - the in-process THREAD path (today's behaviour; a ScriptExecutor worker
 *     calls execute_script_task directly), and
 *   - the PC_ISOLATION worker-process main loop (Phase 3; a re-exec'd child
 *     calls it on deserialized inputs),
 * making THREAD vs PROCESS numerical identity structural rather than hoped
 * for. See INVESTIGATIONS/PC_ISOLATION_process_pool.md.
 *
 * Division of labour (failure-parity contract):
 *   - This function performs: LAMMPS instance cache/create -> inject lines ->
 *     potential build/adopt (with per-iter negative cache) -> run -> raw
 *     variable readback -> timing stamps.
 *   - EVERYTHING ELSE stays with the caller (parent side): failscore string
 *     formatting, eval_aborted policy, hot-var parsing/validation/warmstart,
 *     EWMA/bench bookkeeping, and all journalling except the potential-build
 *     WARNING (which goes through core::Logger/stderr here, exactly as the
 *     pre-extraction code did, and is therefore visible from worker processes
 *     too).
 *
 * Semantics preserved bit-for-bit from the pre-extraction run_one:
 *   - instance cache keyed by (script + '\x01' + each invar token); a failed
 *     create leaves NO cache entry (retried on the next task). Each entry
 *     also owns the script rendered ONCE to a temp file (ScriptFile):
 *     scripts execute through LAMMPS's Input::file(), the only executor
 *     with native jump/label/next semantics (backward jumps = loops). An
 *     entry whose run fails via an exception (TIMEOUT_ABORT/SCRIPT_ERROR)
 *     is ERASED after the failvar readback: the exception skipped
 *     Input::file()'s internal cleanup, so the instance is poisoned
 *     (dangling FILE handle, nested-file counter, possibly a sticky label
 *     search). Recreation is ~free (create_ms is ~0.02% of run_ms).
 *     A TIMEOUT_CLEAN failure keeps the healthy instance, as before;
 *   - one LammpsTadah per (slot, iter), adopted by every EXTERN pair_coeff of
 *     every script this slot runs within the eval (O1a); a throwing ctor is
 *     negative-cached for the remainder of the iter on this slot;
 *   - create_ms brackets only the cache lookup/create; run_ms brackets
 *     inject + potential build + run_script (NOT the variable readback);
 *   - readback returns the raw get_variable strings ("" when undefined).
 */

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

namespace tadah {
namespace hpo {

enum class PcTaskStatus : unsigned {
  OK = 0,
  RUN_FAIL = 1,        /* run_script returned false (incl. fired timer timeout);
                          the instance is ALIVE, so failvars are still read      */
  POT_BUILD_FAIL = 2,  /* LammpsTadah ctor threw (negative-cached per iter)      */
  CREATE_FAIL = 3,     /* LAMMPS instance construction failed (infrastructure)   */
  KILLED = 4,          /* parent-synthesized only (PROCESS mode): worker died or
                          was SIGKILLed (deadline/gate) -- nothing was readable.
                          Never sent by a worker.                                */
};

struct PcTaskResult {
  PcTaskStatus status = PcTaskStatus::OK;
  std::string error_msg;                    /* POT_BUILD_FAIL: e.what()          */
  std::vector<std::string> outvar_values;   /* OK only; one per outvar name      */
  std::vector<std::string> hotvar_values;   /* OK only; one per hotvar name; ""
                                               when undefined/unreadable         */
  std::vector<std::string> failvar_values;  /* OK only; one per failvar name; ""
                                               when undefined/unreadable. A
                                               parseable value > 0 classifies
                                               the sim band B (parent-side).    */
  double create_ms = 0.0;
  double run_ms = 0.0;
};

/* Per-slot persistent execution state. In THREAD mode this lives in a
 * function-local `static thread_local` (one per ScriptExecutor worker), which
 * is exactly the lifetime of the pre-extraction `tls_lmp_cache` / `tls_lt` /
 * `tls_lt_iter` thread-locals. In PROCESS mode (Phase 3) one instance lives in
 * each worker process. */
template <class R>
struct PcSlotState {
  struct CacheEntry {
    typename R::LMPPtr lmp;
    ScriptFile script;         /* rendered once; reused every eval        */
  };
  std::unordered_map<std::string, CacheEntry> lmp_cache;
  std::shared_ptr<tadah::LammpsTadah> pot;
  long long pot_iter = -1;     /* iter the pot/pot_failed state belongs to */
  bool pot_failed = false;     /* negative cache: ctor threw for this iter */
};

/* The exact hot-value injection command the pre-extraction inject_hot_value
 * produced (`equal`-style so hot scripts may redefine same-style; %.17g so a
 * round-trip through the string preserves the double bit-for-bit). Float
 * formatting has ONE owner (the parent), in both THREAD and PROCESS modes. */
inline std::string format_hot_inject(const std::string& name, double value) {
  char cmd[256];
  std::snprintf(cmd, sizeof(cmd), "variable %s equal %.17g", name.c_str(), value);
  return std::string(cmd);
}

/* -log/-screen per-slot suffixing (verbatim from run_one): a real filename
 * gets "<rank_suffix>.t<slot>" appended so concurrent slots never clobber the
 * same file; "none" is left alone. Used by the THREAD path per call and by the
 * PROCESS handshake to finalize each worker's invar once. */
inline std::vector<std::string> make_worker_invar(
    const std::vector<std::string>& invar,
    const std::string& rank_suffix,
    int slot) {
  std::vector<std::string> out = invar;
  auto rewrite = [&](const char* flag) {
    for (std::size_t k = 0; k + 1 < out.size(); ++k) {
      if (out[k] == flag && out[k + 1] != "none") {
        out[k + 1] += rank_suffix + ".t" + std::to_string(slot);
        break;
      }
    }
  };
  rewrite("-log");
  rewrite("-screen");
  return out;
}

template <class R>
PcTaskResult execute_script_task(
    const std::string& script_name,
    const std::vector<std::string>& script_lines,
    const std::vector<std::string>& invar_final,
    const std::vector<std::string>& outvar_names,
    const std::vector<std::string>& hotvar_names,
    const std::vector<std::string>& failvar_names,
    const tadah::core::Context& pot_ctx,
    unsigned long long iter,
    const std::vector<std::string>& inject_lines,
    PcSlotState<R>& st) {
  PcTaskResult res;

  /* --- LAMMPS instance: per-slot cache keyed by (script + invar) -------- */
  std::string cache_key = script_name;
  for (const auto& s : invar_final) { cache_key += '\x01'; cache_key += s; }

  R runner;
  auto t_create0 = std::chrono::high_resolution_clock::now();
  auto cache_it = st.lmp_cache.find(cache_key);
  if (cache_it == st.lmp_cache.end()) {
    /* Serial builds: comm 0 (LAMMPS STUBS). TADAH_HPO_MPI builds:
     * MPI_COMM_SELF -- every rank runs its own single-rank LAMMPS. */
    auto fresh = runner.create_instance(hpo_lammps_comm(), invar_final);
    if (!fresh) {
      res.status = PcTaskStatus::CREATE_FAIL;
      return res;   /* no cache entry -> retried on the next task */
    }
    ScriptFile rendered;
    try {
      rendered = ScriptFile(script_lines);
    } catch (const std::exception& e) {
      /* Temp-file write failed (infrastructure, like a failed create).
       * Same classification in THREAD and PROCESS modes by construction. */
      res.status = PcTaskStatus::CREATE_FAIL;
      res.error_msg = e.what();
      return res;   /* no cache entry -> retried on the next task */
    }
    cache_it = st.lmp_cache.emplace(
        cache_key,
        typename PcSlotState<R>::CacheEntry{std::move(fresh),
                                            std::move(rendered)}).first;
  }
  auto& lmp = cache_it->second.lmp;
  auto t_create1 = std::chrono::high_resolution_clock::now();

  /* --- Injections (pre-formatted command lines; failures swallowed) ----- */
  for (const auto& cmd : inject_lines) {
    try { lmp->input->one(cmd.c_str()); } catch (...) { /* keep going */ }
  }

  /* --- Trial potential: one LammpsTadah per (slot, iter) (O1a) ---------- */
  if (st.pot_iter != static_cast<long long>(iter)) {
    st.pot_failed = false;
    try {
      st.pot = std::make_shared<tadah::LammpsTadah>(pot_ctx);
    } catch (const std::exception& e) {
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::WARNING,
          std::string("run_one: potential construction failed: ") + e.what());
      st.pot.reset();
      st.pot_failed = true;
      res.error_msg = e.what();
    }
    st.pot_iter = static_cast<long long>(iter);
  }

  const bool pot_ok = static_cast<bool>(st.pot) && !st.pot_failed;
  ScriptRunStatus run_status = ScriptRunStatus::OK;
  const bool run_ok = pot_ok &&
      runner.run_script(lmp, cache_it->second.script, st.pot, &run_status);
  auto t_run1 = std::chrono::high_resolution_clock::now();

  res.create_ms =
      std::chrono::duration<double, std::milli>(t_create1 - t_create0).count();
  res.run_ms =
      std::chrono::duration<double, std::milli>(t_run1 - t_create1).count();

  if (!run_ok) {
    res.status = pot_ok ? PcTaskStatus::RUN_FAIL : PcTaskStatus::POT_BUILD_FAIL;
    /* RUN_FAIL (in-script timer timeout / failed run) leaves the LAMMPS
     * instance ALIVE: the declared failvars -- defined EARLY by contract --
     * are still readable and carry the graded band-B violation measure.
     * POT_BUILD_FAIL never ran the script; nothing to read. */
    if (res.status == PcTaskStatus::RUN_FAIL) {
      res.failvar_values.reserve(failvar_names.size());
      for (const auto& n : failvar_names) {
        try { res.failvar_values.emplace_back(runner.get_variable(lmp, n)); }
        catch (...) { res.failvar_values.emplace_back(); }
      }
      /* Exception exits from Input::file() poison the instance (see the
       * header comment); discard AFTER the failvar readback (the Variable
       * store is still intact and band-B grading needs the values).
       * TIMEOUT_CLEAN keeps the healthy instance, as before. */
      if (run_status == ScriptRunStatus::TIMEOUT_ABORT ||
          run_status == ScriptRunStatus::SCRIPT_ERROR) {
        st.lmp_cache.erase(cache_it);
      }
    }
    return res;
  }

  /* --- Raw readback: strings only; ALL parsing/validation is caller-side */
  res.outvar_values.reserve(outvar_names.size());
  for (const auto& n : outvar_names)
    res.outvar_values.emplace_back(runner.get_variable(lmp, n));
  res.hotvar_values.reserve(hotvar_names.size());
  for (const auto& n : hotvar_names) {
    try { res.hotvar_values.emplace_back(runner.get_variable(lmp, n)); }
    catch (...) { res.hotvar_values.emplace_back(); }
  }
  res.failvar_values.reserve(failvar_names.size());
  for (const auto& n : failvar_names) {
    try { res.failvar_values.emplace_back(runner.get_variable(lmp, n)); }
    catch (...) { res.failvar_values.emplace_back(); }
  }
  return res;
}

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_PC_TASK_EXEC_H

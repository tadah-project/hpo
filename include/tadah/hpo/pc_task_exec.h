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
 *
 * Injection errors. The inject lines (the hot values, format_hot_inject)
 * run on the instance before the script. LAMMPS refuses one when the name
 * is already defined with another style -- `-var hot_a0 3.3` (or --invar)
 * on the line makes it an index-style variable when the instance is
 * created; a script that deleted the name and redefined it as a string in
 * an earlier evaluation of a cached instance -- or is not a LAMMPS name.
 * The first refused line fails the task: RUN_FAIL, error_msg "injecting
 * `variable hot_a0 equal 3.25` before the script: <LAMMPS message>", the
 * script not run and nothing read back (the instance's variables, failvars
 * included, are an earlier evaluation's), and the instance discarded, as
 * after any LAMMPS error. It was dropped, and the script ran on its own or
 * an earlier evaluation's value.
 *
 * Read-back errors. Reading a variable evaluates its formula in LAMMPS
 * after the script, and that can raise a LAMMPS error: a thermo keyword the
 * script's thermo_style does not compute (`variable P equal press` under
 * `thermo_style custom step pe`), a divide by zero for one candidate, a
 * compute that is not current. Each variable is read on its own and an
 * error stays with it -- LammpsRunner::get_variable clears the in-progress
 * marks an earlier error left, so a variable that reads an unreadable one
 * raises that one's LAMMPS error, not a false circular dependency:
 *   - an outvar (the names of --varloss and --outvar) is returned "" with
 *     its LAMMPS message in outvar_errors; which of them feed the loss is
 *     the parent's knowledge, so failing the task for one is the parent's
 *     call (apply_task_result in hpo_host_impl.hpp: a --varloss variable
 *     and the PC_FAILPROBE's S fail the task, a diagnostic --outvar is
 *     NaN);
 *   - a --hotvar or --failvar fails the task here: RUN_FAIL, error_msg
 *     "reading --hotvar <name> after the script: <LAMMPS message>" for
 *     each one that cannot be read, joined with "; " in the order they are
 *     read (the hot variables, then the fail variables), so the first
 *     names the failure; no outvar or hot value handed on (the failvars
 *     stay, for grading). Only the first was named before.
 * Every variable is read either way (after a failed run only the
 * failvars, for grading), and an instance that raised a read-back error is
 * discarded, like one whose script raised one: the error leaves evaluation
 * state that only `clear` or a redefinition resets (the variables of the
 * interrupted evaluation stay marked in progress, which get_variable
 * clears for its own reads but the next script would meet; a compute
 * interrupted mid-evaluation), and a script need not start with `clear` or
 * redefine them.
 *
 * Guard: execute_script_task never throws. An exception that escapes the
 * steps above -- each LAMMPS call in them is handled, so this is defence
 * in depth -- fails the task (RUN_FAIL with its message) and discards the
 * slot's instance for the script, in both PC_ISOLATION modes alike.
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
  RUN_FAIL = 1,        /* run_script returned false (incl. fired timer timeout),
                          a hot value could not be injected, a
                          --hotvar/--failvar could not be read, or an
                          exception ended the task; the failvars are still
                          read when the script ran and the instance allows
                          it                                                   */
  POT_BUILD_FAIL = 2,  /* LammpsTadah ctor threw (negative-cached per iter)      */
  CREATE_FAIL = 3,     /* LAMMPS instance construction, or the script's
                          rendering to its temporary file, failed
                          (infrastructure); error_msg says which and why      */
  KILLED = 4,          /* parent-synthesized only (PROCESS mode): worker died or
                          was SIGKILLed (deadline/gate) -- nothing was readable.
                          Never sent by a worker.                                */
};

struct PcTaskResult {
  PcTaskStatus status = PcTaskStatus::OK;
  std::string error_msg;                    /* why the task failed, for the
                                               journal: CREATE_FAIL "creating
                                               the LAMMPS instance: <LAMMPS
                                               message>" or "rendering the
                                               script to a temporary file:
                                               <reason>"; POT_BUILD_FAIL the
                                               potential's construction error;
                                               RUN_FAIL the LAMMPS message of
                                               an error inside the script, of a
                                               refused injection ("injecting
                                               `<command>` before the script:
                                               ..."), of a --hotvar/--failvar
                                               read-back ("reading --hotvar
                                               <name> after the script: ..."),
                                               or the exception that ended the
                                               task.
                                               Empty for a timeout, and for a
                                               POT_BUILD_FAIL the slot already
                                               reported this iteration.         */
  std::vector<std::string> outvar_values;   /* OK only; one per outvar name; ""
                                               when undefined or unreadable     */
  std::vector<std::string> outvar_errors;   /* OK only; empty when every outvar
                                               was read, else one per outvar
                                               name: the LAMMPS message of its
                                               read-back error, "" for one that
                                               was read                         */
  std::vector<std::string> hotvar_values;   /* OK only; one per hotvar name; ""
                                               when undefined                   */
  std::vector<std::string> failvar_values;  /* OK and RUN_FAIL (empty when the
                                               script did not run); one per
                                               failvar name; "" when undefined
                                               or unreadable. A parseable value
                                               > 0 classifies the sim band B
                                               (parent-side).                   */
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

/* The slot cache key of a script: its name, then '\x01' and each token of
 * its final invar. */
inline std::string pc_cache_key(const std::string& script_name,
                                const std::vector<std::string>& invar_final) {
  std::string key = script_name;
  for (const auto& s : invar_final) { key += '\x01'; key += s; }
  return key;
}

/* A message without its trailing whitespace (a LAMMPS message ends with a
 * newline). */
inline std::string pc_trim_message(std::string msg) {
  const auto end = msg.find_last_not_of(" \t\r\n");
  msg.erase(end == std::string::npos ? 0 : end + 1);
  return msg;
}

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

namespace detail {

/* Reads @p name into @p value; on a LAMMPS (or any) error, @p value is ""
 * and @p error the message. True when the variable was read. */
template <class R>
bool pc_read_variable(const R& runner, const typename R::LMPPtr& lmp,
                      const std::string& name, std::string& value,
                      std::string& error) {
  try {
    value = runner.get_variable(lmp, name);
    return true;
  } catch (const std::exception& e) {
    value.clear();
    error = pc_trim_message(e.what());
    return false;
  }
}

/* execute_script_task without its guard. */
template <class R>
PcTaskResult execute_script_task_unguarded(
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
  const std::string cache_key = pc_cache_key(script_name, invar_final);

  R runner;
  auto t_create0 = std::chrono::high_resolution_clock::now();
  auto cache_it = st.lmp_cache.find(cache_key);
  if (cache_it == st.lmp_cache.end()) {
    /* Serial builds: comm 0 (LAMMPS STUBS). TADAH_HPO_MPI builds:
     * MPI_COMM_SELF -- every rank runs its own single-rank LAMMPS. */
    std::string create_error;
    auto fresh = runner.create_instance(hpo_lammps_comm(), invar_final,
                                        &create_error);
    if (!fresh) {
      res.status = PcTaskStatus::CREATE_FAIL;
      res.error_msg = "creating the LAMMPS instance: " +
                      (create_error.empty()
                           ? std::string("no reason given")
                           : pc_trim_message(create_error));
      return res;   /* no cache entry -> retried on the next task */
    }
    ScriptFile rendered;
    try {
      rendered = ScriptFile(script_lines);
    } catch (const std::exception& e) {
      /* Temp-file write failed (infrastructure, like a failed create).
       * Same classification in THREAD and PROCESS modes by construction. */
      res.status = PcTaskStatus::CREATE_FAIL;
      res.error_msg = std::string("rendering the script to a temporary "
                                  "file: ") + pc_trim_message(e.what());
      return res;   /* no cache entry -> retried on the next task */
    }
    cache_it = st.lmp_cache.emplace(
        cache_key,
        typename PcSlotState<R>::CacheEntry{std::move(fresh),
                                            std::move(rendered)}).first;
  }
  auto& lmp = cache_it->second.lmp;
  auto t_create1 = std::chrono::high_resolution_clock::now();

  /* --- Injections (pre-formatted command lines, the hot values) ---------
   * One LAMMPS refuses fails the task (see the header comment): the script
   * does not run, and nothing is read back, since the instance's variables
   * are an earlier evaluation's. */
  for (const auto& cmd : inject_lines) {
    try {
      lmp->input->one(cmd.c_str());
    } catch (const std::exception& e) {
      res.status = PcTaskStatus::RUN_FAIL;
      res.error_msg = "injecting `" + cmd + "` before the script: " +
                      pc_trim_message(e.what());
      res.create_ms = std::chrono::duration<double, std::milli>(
                          t_create1 - t_create0).count();
      res.run_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_create1)
                       .count();
      st.lmp_cache.erase(cache_it);
      return res;
    }
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
  std::string run_error;
  const bool run_ok = pot_ok &&
      runner.run_script(lmp, cache_it->second.script, st.pot, &run_status,
                        &run_error);
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
      /* The LAMMPS message of an error inside the script, for the journal
       * (a timeout has none). */
      if (run_status == ScriptRunStatus::SCRIPT_ERROR)
        res.error_msg = pc_trim_message(run_error);
      /* A failvar that cannot be read is "" (the task has failed
       * already; its grading falls back to the failed fraction). */
      bool read_error = false;
      res.failvar_values.resize(failvar_names.size());
      for (std::size_t j = 0; j < failvar_names.size(); ++j) {
        std::string ignored;
        if (!detail::pc_read_variable(runner, lmp, failvar_names[j],
                                      res.failvar_values[j], ignored))
          read_error = true;
      }
      /* Exception exits from Input::file() poison the instance (see the
       * header comment); discard AFTER the failvar readback (the Variable
       * store is still intact and band-B grading needs the values).
       * TIMEOUT_CLEAN keeps the healthy instance, as before, unless a
       * failvar read raised an error. */
      if (run_status == ScriptRunStatus::TIMEOUT_ABORT ||
          run_status == ScriptRunStatus::SCRIPT_ERROR || read_error) {
        st.lmp_cache.erase(cache_it);
      }
    }
    return res;
  }

  /* --- Raw readback: strings only; ALL parsing/validation is caller-side.
   * Each variable on its own: a read-back error stays with its variable
   * (see the header comment). */
  bool read_error = false;
  res.outvar_values.resize(outvar_names.size());
  for (std::size_t j = 0; j < outvar_names.size(); ++j) {
    std::string err;
    if (detail::pc_read_variable(runner, lmp, outvar_names[j],
                                 res.outvar_values[j], err))
      continue;
    res.outvar_errors.resize(outvar_names.size());
    res.outvar_errors[j] = err;
    read_error = true;
  }
  /* A --hotvar/--failvar that cannot be read fails the task; the message
   * names each one, the first first. */
  std::string fatal;
  auto read_all = [&](const char* role, const std::vector<std::string>& names,
                      std::vector<std::string>& values) {
    values.resize(names.size());
    for (std::size_t j = 0; j < names.size(); ++j) {
      std::string err;
      if (detail::pc_read_variable(runner, lmp, names[j], values[j], err))
        continue;
      read_error = true;
      fatal += std::string(fatal.empty() ? "" : "; ") + "reading " + role +
               " " + names[j] + " after the script: " + err;
    }
  };
  read_all("--hotvar", hotvar_names, res.hotvar_values);
  read_all("--failvar", failvar_names, res.failvar_values);
  if (!fatal.empty()) {
    res.status = PcTaskStatus::RUN_FAIL;
    res.error_msg = fatal;
    res.outvar_values.clear();
    res.outvar_errors.clear();
    res.hotvar_values.clear();
  }
  if (read_error) st.lmp_cache.erase(cache_it);
  return res;
}

}  // namespace detail

/* One PC_LAMMPS task on slot state @p st: see the header comment. Never
 * throws: an exception that escapes the task's own handling fails the task
 * (RUN_FAIL, error_msg its message) and discards the slot's instance for
 * the script, so THREAD's run_one and the PROCESS worker, which both call
 * this, treat it alike. */
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
  std::string what;
  try {
    return detail::execute_script_task_unguarded<R>(
        script_name, script_lines, invar_final, outvar_names, hotvar_names,
        failvar_names, pot_ctx, iter, inject_lines, st);
  } catch (const std::exception& e) {
    what = pc_trim_message(e.what());
  } catch (...) {
    what = "an exception of unknown type ended the task";
  }
  PcTaskResult res;
  res.status = PcTaskStatus::RUN_FAIL;
  res.error_msg = what;
  st.lmp_cache.erase(pc_cache_key(script_name, invar_final));
  return res;
}

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_PC_TASK_EXEC_H

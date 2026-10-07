#include <tadah/hpo/runners.h>
#include <stdexcept>

#include <input.h>
#include <force.h>
#include <pair.h>
#include <timer.h>
#include <variable.h>
#include <pair_tadah.h>

#include <tadah/core/number_format.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

using LMPPtr = LammpsRunner::LMPPtr;
using LMP = LammpsRunner::LMP;

LammpsRunner::LMPPtr
LammpsRunner::create_instance(MPI_Comm sub_comm,
                              std::vector<std::string> tokens,
                              std::string* error) const {
 
  // unique_ptr initialized to nullptr
  LMPPtr lmp_ptr;

  try {
    // Construct LAMMPS; transfer ownership into unique_ptr
    lmp_ptr.reset(
      new LMP(tokens, sub_comm)
      //new LMP(argc, argv.data(), sub_comm)
    );
  } catch (const std::exception& e) {
    // On exception, leave lmp_ptr null; the LAMMPS message to the caller
    // that asked for it (an option LAMMPS refuses: "ERROR: Invalid
    // command-line argument: -sff (...)").
    if (error) *error = e.what();
  }

  return lmp_ptr;
}

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
                         const std::string& script) const {
  if (!lmp_ptr) {
    // Cannot run on invalid instance
    return false;
  }

  try {
    // Read and execute commands from the script file

    lmp_ptr->input->file(script.c_str());
  } catch (const std::exception&) {
    // Execution failure indicates invalid run
    return false;
  }

  /* A fired `timer timeout` stops run/minimize cleanly (no exception), so
   * the sim reached this point without completing its physics -- report it
   * as failed. The instance stays healthy: the next script's `clear`
   * rebuilds the Timer and re-injects the timeout. */
  if (lmp_ptr->timer->is_timeout()) return false;

  return true;
}
namespace {

using tadah::hpo::ScriptFile;
using tadah::hpo::ScriptRunStatus;

bool set_why(ScriptRunStatus* why, ScriptRunStatus v) {
  if (why) *why = v;
  return v == ScriptRunStatus::OK;
}

/* RAII staging of the EXTERN potential for the duration of one script run
 * (per thread; see PairTadah::stage_extern). An empty pot stages nothing
 * but still clears the slot on exit, so no stale potential can leak into a
 * later run on this thread. */
struct StagedExternGuard {
  explicit StagedExternGuard(std::shared_ptr<tadah::LammpsTadah> pot) {
    LAMMPS_NS::PairTadah::stage_extern(std::move(pot));
  }
  ~StagedExternGuard() { LAMMPS_NS::PairTadah::clear_staged_extern(); }
  StagedExternGuard(const StagedExternGuard&) = delete;
  StagedExternGuard& operator=(const StagedExternGuard&) = delete;
};

/* SCRIPT_ERROR, with @p msg in @p error when the caller asked for it. */
bool script_error(ScriptRunStatus* why, std::string* error,
                  const std::string& msg) {
  if (error) *error = msg;
  return set_why(why, ScriptRunStatus::SCRIPT_ERROR);
}

/* Run the rendered script through LAMMPS's own reader. Native semantics
 * for jump/label/next/if-then-"jump" -- the one()-per-line executor could
 * not loop backwards (jump's rewind() is a no-op on stdin and the sticky
 * label search then swallowed the rest of the script AND poisoned the
 * cached instance's next eval: the run26 SIGSEGV storm). */
bool execute_file(const LammpsRunner::LMPPtr& lmp_ptr, const ScriptFile& sf,
                  ScriptRunStatus* why, std::string* error) {
  try {
    lmp_ptr->input->file(sf.path().c_str());
  } catch (const std::exception& e) {
    /* The injected section-boundary guard (see prep_script_lines) aborts
     * with the sentinel as an "unknown command"; anything else is a
     * genuine script failure, whose LAMMPS message goes to @p error (the
     * sentinel is no message). Either way the exception skipped
     * Input::file()'s internal cleanup, so the instance is poisoned --
     * cache-holding callers must discard it (ScriptRunStatus contract). */
    const bool timeout =
        std::string(e.what()).find(tadah::hpo::PC_TIMEOUT_ABORT_SENTINEL) !=
        std::string::npos;
    if (timeout) return set_why(why, ScriptRunStatus::TIMEOUT_ABORT);
    return script_error(why, error, e.what());
  }
  /* A fired `timer timeout` in the FINAL section stops run/minimize
   * cleanly (no exception) and no clear follows to reset it: incomplete
   * physics, healthy instance -- the next script's clear rebuilds the
   * Timer, exactly as under the old executor. */
  if (lmp_ptr->timer->is_timeout())
    return set_why(why, ScriptRunStatus::TIMEOUT_CLEAN);
  return set_why(why, ScriptRunStatus::OK);
}

}  // namespace

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const tadah::hpo::ScriptFile& script_file,
    const tadah::core::Context& pot_ctx,
    tadah::hpo::ScriptRunStatus* why,
    std::string* error) const {
  if (!lmp_ptr) return script_error(why, error, "no LAMMPS instance");
  if (!script_file.valid())
    return script_error(why, error, "the script was not rendered");
  try {
    /* Build the potential once per run, and ONLY for scripts that adopt
     * it: EXTERN-less scripts legitimately run with an empty Context (the
     * old per-line hook rebuilt the potential on every EXTERN line). */
    StagedExternGuard staged(script_file.has_extern()
        ? std::make_shared<tadah::LammpsTadah>(pot_ctx)
        : std::shared_ptr<tadah::LammpsTadah>());
    return execute_file(lmp_ptr, script_file, why, error);
  } catch (const std::exception& e) {
    // LammpsTadah construction failed (execute_file throws nothing)
    return script_error(why, error,
                        std::string("potential construction failed: ") +
                            e.what());
  }
}

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const tadah::hpo::ScriptFile& script_file,
    const std::shared_ptr<tadah::LammpsTadah>& pot,
    tadah::hpo::ScriptRunStatus* why,
    std::string* error) const {
  if (!lmp_ptr) return script_error(why, error, "no LAMMPS instance");
  if (!pot) return script_error(why, error, "no potential");
  if (!script_file.valid())
    return script_error(why, error, "the script was not rendered");
  StagedExternGuard staged(pot);
  return execute_file(lmp_ptr, script_file, why, error);
}

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const std::vector<std::string>& script_lines,
    const tadah::core::Context& pot_ctx) const {
  if (!lmp_ptr) return false;
  try {
    const ScriptFile sf(script_lines);  // per-call render (see runners.h)
    return run_script(lmp_ptr, sf, pot_ctx, nullptr);
  } catch (const std::exception&) {
    return false;
  }
}

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const std::vector<std::string>& script_lines,
    const std::shared_ptr<tadah::LammpsTadah>& pot) const {
  if (!lmp_ptr || !pot) return false;
  try {
    const ScriptFile sf(script_lines);  // per-call render (see runners.h)
    return run_script(lmp_ptr, sf, pot, nullptr);
  } catch (const std::exception&) {
    return false;
  }
}

bool LammpsRunner::say_hello(const LMPPtr& lmp_ptr) const {
  if (!lmp_ptr) {
    // Cannot run on invalid instance
    return false;
  }

  try {
    // Read and execute commands from the script file
    LAMMPS_NS::Pair *pair = lmp_ptr->force->pair;
    LAMMPS_NS::PairTadah *pt = dynamic_cast<LAMMPS_NS::PairTadah*>(pair);
    pt->say_hello();

    return true;
  } catch (const std::exception&) {
    // Execution failure indicates invalid run
    return false;
  }

}

std::string
LammpsRunner::get_variable(const LammpsRunner::LMPPtr& lmp_ptr,
                           const std::string& varname) const {
  if (!lmp_ptr) {
    // Caller must not pass an invalid instance
    throw std::runtime_error("get_variable(): LAMMPS instance is invalid.");
  }

  LAMMPS_NS::Variable* var = lmp_ptr->input->variable;

  // Undefined variable -> "" (Variable::retrieve() would return nullptr;
  // constructing a std::string from nullptr is undefined behaviour). The
  // caller decides how to handle the missing value (typically std::stod()
  // on "" throws std::invalid_argument, which the LAMMPS sims_loss path
  // treats as a failed eval and clamps to FAIL_SCORE).
  const int ivar = var->find(varname.c_str());
  if (ivar < 0) return std::string();

  // A read is a top-level evaluation: no LAMMPS evaluation is in progress
  // when it starts. A LAMMPS error raised during an earlier one -- an
  // earlier read, or a command of the script -- left the variables of that
  // evaluation marked as being evaluated (Variable::eval_in_progress, reset
  // only by `clear` or a redefinition), and a read that reached one of them
  // would raise "Variable ...: has a circular dependency" instead of that
  // variable's own error. Clear the marks first, as `clear` does; a real
  // circular dependency is still found while this read evaluates.
  var->clear_in_progress();

  // Numeric (equal-style: EQUAL, TIMER, INTERNAL, numeric PYTHON): read the
  // DOUBLE and render it losslessly. retrieve() would format it with
  // "%.15g", losing up to ~2 digits of every observable, hot value and
  // gradient readback. compute_equal() runs the same evaluation with the
  // same circular-dependency guard (and the same underflow-to-0 rounding
  // for EQUAL); it only skips retrieve()'s string cache. equalstyle() raises
  // a LAMMPS error for a python-style variable whose function returns a
  // STRING (LAMMPS built with PYTHON): that one is text, read below.
  bool numeric = false;
  try { numeric = var->equalstyle(ivar) != 0; }
  catch (const std::exception&) { numeric = false; }
  if (numeric)
    return tadah::core::format_roundtrip(var->compute_equal(ivar));

  // Every other style (string, index, format, getenv, loop, vector, ...) is
  // text by definition: returned verbatim. nullptr (e.g. atom-style) -> "".
  const char* raw = var->retrieve(varname.c_str());
  return raw ? std::string(raw) : std::string();
}

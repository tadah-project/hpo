#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H

#include <memory>
#include <vector>
#include <string>

#include <lammps.h>
#include <input.h>
#include <variable.h>

#include <tadah/core/context.h>
#include <tadah/hpo/script_file.h>

namespace tadah { class LammpsTadah; }

namespace tadah { namespace hpo {

/* Outcome of a script run, for callers that need more than pass/fail.
 * TIMEOUT_CLEAN keeps the LAMMPS instance healthy (a fired timer in the
 * final section; run/minimize stopped cleanly, Input::file() completed).
 * TIMEOUT_ABORT and SCRIPT_ERROR left Input::file() via an exception,
 * which skips its internal cleanup (open FILE handle, nested-file
 * counter, possibly a sticky label search) -- the instance must be
 * DISCARDED by any caller that caches instances (see pc_task_exec.h). */
enum class ScriptRunStatus {
  OK = 0,
  TIMEOUT_CLEAN,   /* fired soft timeout, instance healthy               */
  TIMEOUT_ABORT,   /* section-boundary guard fired, instance poisoned    */
  SCRIPT_ERROR,    /* any other failure, instance poisoned               */
};

} }  // namespace tadah::hpo

/**
 * @class LammpsRunner
 * @brief Simplifies creation and use of LAMMPS instances.
 *
 * Instances are reusable for multiple runs.
 * An instance encountering an unrecoverable exception becomes invalid.
 */
class LammpsRunner {
 public:
  /// Alias for the LAMMPS class in the LAMMPS_NS namespace
  typedef LAMMPS_NS::LAMMPS LMP;

  /// Uniquepointer type for a LAMMPS instance
  typedef std::unique_ptr<LMP> LMPPtr;

  /**
   * @brief Creates a new LAMMPS instance.
   *
   * Builds a LAMMPS instance using provided MPI subcommunicator
   * and commandline tokens.  If construction throws, returns nullptr.
   *
   * @param sub_comm MPI communicator for this instance.
   * @param lmp_tokens Commandline arguments for LAMMPS.
   * @return Unique pointer to a valid LAMMPS instance on success; nullptr on failure.
   *
   * @code
   * // Example usage:
   * std::vector<const char*> args = {"lmp", "-in", "in.script"};
   * auto inst = runner.create_instance(comm, args);
   * if (!inst) {
   *   // handle creation failure
   * }
   * @endcode
   */
  LMPPtr create_instance(MPI_Comm sub_comm,
                         std::vector<std::string> lmp_tokens) const;

  /**
   * @brief Executes a LAMMPS input script on the given instance.
   *
   * Catches LAMMPS exceptions.  If the instance pointer is null
   * or an exception is thrown, returns false.
   *
   * @param lmp_ptr Unique pointer to a LAMMPS instance.
   * @param script Path to the LAMMPS input script.
   * @return True if execution succeeded; false otherwise.
   */
  bool run_script(const LMPPtr& lmp_ptr,
                  const std::string& script) const;

  /**
   * @brief Executes a rendered PC_LAMMPS script through Input::file().
   *
   * The whole script runs under LAMMPS's own file reader, so
   * jump/label/next and if-then-"jump" control flow (including BACKWARD
   * jumps, i.e. loops) have native LAMMPS semantics. The candidate
   * potential is staged per-thread (PairTadah::stage_extern) for the
   * duration of the run; every `pair_coeff * * EXTERN ...` the script
   * executes adopts it inside PairTadah::coeff(). The potential is built
   * from the Context once per run, and only when the script actually
   * contains an EXTERN line (EXTERN-less scripts run fine with an empty
   * Context).
   *
   * @param lmp_ptr Unique pointer to a LAMMPS instance.
   * @param script_file Rendered script (see tadah::hpo::ScriptFile).
   * @param pot_ctx Potential Context (used only for EXTERN scripts).
   * @param why Optional detailed outcome; see ScriptRunStatus for the
   *            instance-poisoning contract on TIMEOUT_ABORT/SCRIPT_ERROR.
   * @param error Optional: on SCRIPT_ERROR, why -- the LAMMPS message of
   *            an error inside the script ("ERROR: Unknown command: ..."),
   *            the potential's construction error, or the missing input.
   *            Left alone otherwise: a timeout has no message (the
   *            section guard's sentinel is not one).
   * @return True if execution succeeded; false otherwise.
   */
  bool run_script(const LMPPtr& lmp_ptr,
                  const tadah::hpo::ScriptFile& script_file,
                  const tadah::core::Context& pot_ctx,
                  tadah::hpo::ScriptRunStatus* why = nullptr,
                  std::string* error = nullptr) const;

  /**
   * @brief Executes a rendered script using a pre-built shared potential.
   *
   * Same executor as the Context overload, but the driver builds ONE
   * tadah::LammpsTadah per candidate potential (per thread) and stages it
   * directly -- no reconstruction, no Context copies. The object carries
   * per-sim scratch, so it must only be shared across sims that run
   * sequentially. @p why and @p error as for the Context overload.
   */
  bool run_script(const LMPPtr& lmp_ptr,
                  const tadah::hpo::ScriptFile& script_file,
                  const std::shared_ptr<tadah::LammpsTadah>& pot,
                  tadah::hpo::ScriptRunStatus* why = nullptr,
                  std::string* error = nullptr) const;

  /**
   * @brief Convenience wrapper: renders script_lines to a temp ScriptFile
   * per call and delegates to the ScriptFile overload. Intended for tests
   * and one-shot callers; cached/hot paths should render once and reuse
   * (pc_task_exec.h, LammpsSim::script_file()).
   */
  bool run_script(const LMPPtr& lmp_ptr,
                  const std::vector<std::string>& script_lines,
                  const tadah::core::Context& pot_ctx) const;

  /** @brief Convenience wrapper; see the Context-flavoured wrapper above. */
  bool run_script(const LMPPtr& lmp_ptr,
                  const std::vector<std::string>& script_lines,
                  const std::shared_ptr<tadah::LammpsTadah>& pot) const;

  bool say_hello(const LMPPtr& lmp_ptr) const;
  /**
   * @brief Retrieves the value of a LAMMPS variable.
   *
   * Numeric (equal-style) variables are evaluated and returned as the
   * shortest text that parses back to the identical double
   * (core::format_roundtrip) -- not LAMMPS' own "%.15g" string. Every
   * other style is returned verbatim; an undefined variable gives "".
   *
   * Each read starts by clearing the in-progress marks of LAMMPS's
   * variables (Variable::clear_in_progress, as `clear` does): it is a
   * top-level evaluation, so a mark it finds was left by a LAMMPS error in
   * an earlier evaluation -- an earlier read, or a command of the script --
   * and would turn the read of any variable that reaches the interrupted
   * one into a false "has a circular dependency". A variable that depends
   * on one that cannot be read therefore raises that one's error again.
   *
   * @param lmp_ptr Unique pointer to a LAMMPS instance.
   * @param varname Name of the variable to retrieve.
   * @return String representation of the requested variable.
   * @throws std::runtime_error if lmp_ptr is null.
   * @throws LAMMPS_NS::LAMMPSException (or LAMMPSAbortException) when
   *         evaluating the formula raises a LAMMPS error: a thermo keyword
   *         the thermo_style does not compute, a divide by zero, a compute
   *         that is not current. The error can leave the instance with
   *         state that only `clear` or a redefinition resets (the marks of
   *         the interrupted evaluation, which the next script would meet
   *         before any `clear`; a compute interrupted mid-evaluation), so a
   *         caller that caches instances discards it (pc_task_exec.h).
   */
  std::string get_variable(const LMPPtr& lmp_ptr,
                           const std::string& varname) const;
};

#endif // LAMMPS_RUNNER_H

#include <tadah/hpo/runners.h>
#include <stdexcept>

#include <input.h>
#include <force.h>
#include <pair.h>
#include <timer.h>
#include <pair_tadah.h>

#include <tadah/tadah.lammps/lammps_tadah.h>

using LMPPtr = LammpsRunner::LMPPtr;
using LMP = LammpsRunner::LMP;

LammpsRunner::LMPPtr
LammpsRunner::create_instance(MPI_Comm sub_comm,
                              std::vector<std::string> tokens) const {
 
  // unique_ptr initialized to nullptr
  LMPPtr lmp_ptr;

  try {
    // Construct LAMMPS; transfer ownership into unique_ptr
    lmp_ptr.reset(
      new LMP(tokens, sub_comm)
      //new LMP(argc, argv.data(), sub_comm)
    );
  } catch (const std::exception&) {
    // On exception, leave lmp_ptr null
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
bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const std::vector<std::string>& script_lines,
    const tadah::core::Context& pot_ctx) const {
  if (!lmp_ptr) {
    // Cannot run on invalid instance
    return false;
  }

  try {
    // Read and execute commands from the script file

    for (const auto& line : script_lines) {
      // If the line contains pair_coeff and EXTERN, load the potential
      // from the context before executing the line
      if (line.find("pair_coeff") != std::string::npos &&
          line.find("EXTERN") != std::string::npos) {
        LAMMPS_NS::Pair *pair = lmp_ptr->force->pair;
        LAMMPS_NS::PairTadah *pt = dynamic_cast<LAMMPS_NS::PairTadah*>(pair);
        // pt->load(std::move(pot_ctx));  // cannot move as pair_coeff might be called several times
        pt->load(pot_ctx);
      }
      // Execute the line
      lmp_ptr->input->one(line.c_str());
      /* Fired `timer timeout` == incomplete physics. Stop at the line
       * boundary: LAMMPS's timer only interrupts run/minimize ITERATIONS,
       * so a `run 0` probe still pays its (possibly pathological) force
       * call after a timeout -- without this check a timed-out sim would
       * execute every remaining probe line. */
      if (lmp_ptr->timer->is_timeout()) return false;
    }
  } catch (const std::exception&) {
    // Execution failure indicates invalid run
    return false;
  }

  return true;

}

bool LammpsRunner::run_script(const LMPPtr& lmp_ptr,
    const std::vector<std::string>& script_lines,
    const std::shared_ptr<tadah::LammpsTadah>& pot) const {
  if (!lmp_ptr || !pot) {
    return false;
  }

  try {
    for (const auto& line : script_lines) {
      // On the EXTERN pair_coeff line, adopt the shared pre-built potential
      // instead of reconstructing one from a Context.
      if (line.find("pair_coeff") != std::string::npos &&
          line.find("EXTERN") != std::string::npos) {
        LAMMPS_NS::Pair *pair = lmp_ptr->force->pair;
        LAMMPS_NS::PairTadah *pt = dynamic_cast<LAMMPS_NS::PairTadah*>(pair);
        pt->load(pot);
      }
      lmp_ptr->input->one(line.c_str());
      // Per-line timeout stop; see the Context overload for rationale.
      if (lmp_ptr->timer->is_timeout()) return false;
    }
  } catch (const std::exception&) {
    return false;
  }

  return true;
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

  // LAMMPS' Variable::retrieve() returns nullptr (not "") when the
  // requested variable is undefined, has not been computed yet, or its
  // expression evaluates to a non-numeric tag. Constructing a
  // std::string from nullptr is undefined behaviour (libstdc++ throws
  // std::logic_error on it). Convert nullptr to empty string so the
  // caller can decide how to handle the missing value (typically
  // std::stod() on "" throws std::invalid_argument, which the LAMMPS
  // sims_loss path treats as a failed eval and clamps to FAIL_SCORE).
  const char* raw = lmp_ptr->input->variable->retrieve(varname.c_str());
  return raw ? std::string(raw) : std::string();
}

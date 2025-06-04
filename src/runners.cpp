#include <tadah/hpo/runners.h>
#include <stdexcept>

#include <input.h>
#include <force.h>
#include <pair.h>
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

  // Retrieve variable via LAMMPS input subsystem
  return lmp_ptr->input->variable->retrieve(varname.c_str());
}

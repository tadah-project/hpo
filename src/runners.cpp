// #include <tadah/hpo/runners.h>
//
// using LMP = LammpsRunner::LMP; 
// LammpsRunner::LammpsRunner() {}
//
// LMP* LammpsRunner::run(MPI_Comm &sub_comm, const std::string &script,
//     const std::vector<const char*> &lmp_tokens) {
//   int lmpargc = static_cast<int>(lmp_tokens.size());
//   LMP* lmp = nullptr;
//
//   try {
//     lmp = new LMP(lmpargc, const_cast<char**>(lmp_tokens.data()), sub_comm);
//     lmp->input->file(script.c_str());
//   } catch (std::exception& e) {
//     //std::cout << "LAMMPS Exception: " << e.what() << std::endl;
//     delete lmp;
//     return nullptr;
//   }
//   return lmp;
// }
/**
 * @file lammps_runner.cpp
 * @brief Implementation file for the LAMMPS runner utility
 *
 * Use this implementation with "lammps_runner.h". The class
 * supports creation of a LAMMPS instance and running scripts.
 */

#include <tadah/hpo/runners.h>

using LMP = LammpsRunner::LMP; 

LammpsRunner::LammpsRunner() {
  // Nothing special here
}

LMP* LammpsRunner::create_instance(MPI_Comm &sub_comm,
                                   const std::vector<const char*> &lmp_tokens) {
  // Number of command line arguments
  int lmpargc = static_cast<int>(lmp_tokens.size());
  LMP* lmp = nullptr;

  try {
    // LAMMPS is constructed with command line args and the communicator
    lmp = new LMP(lmpargc, const_cast<char**>(lmp_tokens.data()), sub_comm);
  } catch (std::exception &e) {
    // If creation fails, release allocated object and return null
    delete lmp;
    lmp = nullptr;
  }

  return lmp;
}

bool LammpsRunner::run_script(LMP* lmp, const std::string &script) {
  if (!lmp) {
    // Cannot run if instance is invalid
    return false;
  }

  try {
    // LAMMPS reads and runs commands from the script
    lmp->input->file(script.c_str());
  } catch (std::exception &e) {
    // If a crash or error in LAMMPS is thrown, report failure
    return false;
  }

  return true;
}

std::string LammpsRunner::get_variable(LMP* lmp, const std::string &varname) {
  return lmp->input->variable->retrieve(varname.c_str());
}

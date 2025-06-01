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

// #include <tadah/hpo/runners.h>
//
// using LMP = LammpsRunner::LMP; 
//
// LammpsRunner::LammpsRunner() {
//   // Nothing special here
// }
//
// LMP* LammpsRunner::create_instance(MPI_Comm &sub_comm,
//                                    const std::vector<const char*> &lmp_tokens) {
//   // Number of command line arguments
//   int lmpargc = static_cast<int>(lmp_tokens.size());
//   LMP* lmp = nullptr;
//
//   try {
//     // LAMMPS is constructed with command line args and the communicator
//     lmp = new LMP(lmpargc, const_cast<char**>(lmp_tokens.data()), sub_comm);
//   } catch (std::exception &e) {
//     // If creation fails, release allocated object and return null
//     delete lmp;
//     lmp = nullptr;
//   }
//
//   return lmp;
// }
//
// bool LammpsRunner::run_script(LMP* lmp, const std::string &script) {
//   if (!lmp) {
//     // Cannot run if instance is invalid
//     return false;
//   }
//
//   try {
//     // LAMMPS reads and runs commands from the script
//     lmp->input->file(script.c_str());
//   } catch (std::exception &e) {
//     // If a crash or error in LAMMPS is thrown, report failure
//     return false;
//   }
//
//   return true;
// }
//
// std::string LammpsRunner::get_variable(LMP* lmp, const std::string &varname) {
//   if (!lmp) {
//     throw std::runtime_error("get_variable(): LAMMPS instance is invalid.");
//   }
//   return lmp->input->variable->retrieve(varname.c_str());
// }

#include <tadah/hpo/runners.h>
#include <stdexcept>

LammpsRunner::LMPPtr
LammpsRunner::create_instance(MPI_Comm sub_comm,
                              std::vector<std::string> tokens) const {
  // Number of command‐line arguments
  // int argc = static_cast<int>(lmp_tokens.size());

  // unique_ptr initialized to nullptr
  LMPPtr lmp_ptr;

  //   size_t n = tokens.size();
  //
  // // Own the buffers so they stay alive until f() returns
  // std::vector<std::unique_ptr<char[]>> buffers;
  // buffers.reserve(n);
  //
  // // Build argv array of mutable pointers
  // std::vector<char*> argv;
  // argv.reserve(n);
  //
  // for (auto const& s : tokens) {
  //   // Allocate one extra byte for terminating '\0'
  //   auto buf = std::make_unique<char[]>(s.size() + 1);
  //   std::copy(s.c_str(), s.c_str() + s.size() + 1, buf.get());
  //   argv.push_back(buf.get());
  //   buffers.push_back(std::move(buf));
  // }
  //
  // int argc = static_cast<int>(n);

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
// bool LammpsRunner::run_script(const LMP* lmp_ptr,
//                          const std::string& script) const {
//   if (!lmp_ptr) {
//     // Cannot run on invalid instance
//     return false;
//   }
//
//   try {
//     // Read and execute commands from the script file
//     lmp_ptr->input->file(script.c_str());
//   } catch (const std::exception&) {
//     // Execution failure indicates invalid run
//     return false;
//   }
//
//   return true;
// }

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

// std::string
// LammpsRunner::get_variable(const LammpsRunner::LMP* lmp_ptr,
//                            const std::string& varname) const {
//   if (!lmp_ptr) {
//     // Caller must not pass an invalid instance
//     throw std::runtime_error("get_variable(): LAMMPS instance is invalid.");
//   }
//
//   // Retrieve variable via LAMMPS input subsystem
//   return lmp_ptr->input->variable->retrieve(varname.c_str());
// }


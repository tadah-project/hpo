// #ifndef LAMMPS_RUNNER_H
// #define LAMMPS_RUNNER_H
//
// #include <iostream>
// #include <memory>
//
// #ifdef TADAH_BUILD_MPI
// #include <mpi.h>
// #endif
//
// #include <lammps.h>
// #include <input.h>
// #include <variable.h>
//
// /** 
//  * @class LammpsRunner
//  * @brief A class to simplify running LAMMPS simulations.
//  *
//  * To compile this file outside the Tadah library, 
//  * specify the TADAH_LAMMPS_LIB macro with the LAMMPS library name.
//  *
//  * Example:
//  * -DTADAH_LAMMPS_LIB="liblammps_single" 
//  *
//  * LAMMPS must be compiled with -DLAMMPS_EXCEPTIONS
//  * to allow Tadah! to handle simulation failures gracefully.
//  */
// class LammpsRunner {
//   public:
//
//     // Typedef for LAMMPS instance
//     typedef LAMMPS_NS::LAMMPS LMP;
//
//     // Default constructor for LammpsRunner
//     LammpsRunner();
//
//     /** Create LAMMPS instance and run a script 
//      *
//      * Return a pointer to LAMMPS instance.
//      *
//      * In case of LAMMPS throwing exception return nullptr.
//      */
//     LMP* run(MPI_Comm &sub_comm, const std::string &script,
//         const std::vector<const char*> &lmp_tokens);
//
//     /**
//      * @brief Retrieve the value of a LAMMPS variable.
//      *
//      * @param lmp Pointer to the LAMMPS instance.
//      * @param varname The name of the variable to retrieve.
//      * @return The value of the variable as a string.
//      */
//     std::string get_variable(LMP* lmp, std::string varname) {
//       return lmp->input->variable->retrieve(varname.c_str());
//     }
// };
//
// #endif // LAMMPS_RUNNER_H
/** 
 * @file lammps_runner.h
 * @brief Header file for LAMMPS runner utility
 *
 * This file declares a helper class for creating and managing
 * LAMMPS instances. Reuse of a single instance is supported.
 * LAMMPS crashes are handled by capturing exceptions (LAMMPS
 * must be compiled with -DLAMMPS_EXCEPTIONS).
 *
 * Example usage:
 * @code
 * LammpsRunner runner;
 * MPI_Comm my_comm = MPI_COMM_WORLD;
 * 
 * // Create once
 * auto *lmp = runner.create_instance(my_comm, some_invar);
 * 
 * // Run multiple scripts, handle crashes
 * if (!runner.run_script(lmp, "in.script1")) {
 *   // Crash occurred
 *   delete lmp;
 *   lmp = nullptr;
 * }
 * if (!runner.run_script(lmp, "in.script2")) {
 *   // Crash occurred
 *   delete lmp;
 *   lmp = nullptr;
 * }
 * if (lmp) {
 *   delete lmp;
 * }
 * @endcode
 */

#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

#include <lammps.h>
#include <input.h>
#include <variable.h>

/**
 * @class LammpsRunner
 * @brief A class to simplify creation and use of LAMMPS instances.
 *
 * A single instance is reusable for multiple script runs. 
 * If an unrecoverable LAMMPS exception is thrown during a script,
 * that instance is considered invalid.
 */
class LammpsRunner {
 public:
  /// Typedef for the LAMMPS namespace
  typedef LAMMPS_NS::LAMMPS LMP;

  /// Default constructor
  LammpsRunner();

  /**
   * @brief Creates a new LAMMPS instance with the given command line tokens.
   *
   * @param sub_comm MPI communicator used by this instance.
   * @param lmp_tokens Command line arguments for LAMMPS.
   * @return Valid pointer on success, nullptr if creation fails.
   */
  LMP* create_instance(MPI_Comm &sub_comm,
                       const std::vector<const char*> &lmp_tokens);

  /**
   * @brief Runs a LAMMPS input script on an existing LAMMPS instance.
   *
   * If the script causes a LAMMPS exception, the method returns false.
   *
   * @param lmp Pointer to the LAMMPS instance.
   * @param script Path to the LAMMPS input script.
   * @return True on success, false if an exception was thrown.
   */
  bool run_script(LMP* lmp, const std::string &script);

  /**
   * @brief Retrieve the value of a LAMMPS variable as a string.
   *
   * @param lmp Pointer to the LAMMPS instance.
   * @param varname Name of the variable to retrieve.
   * @return String value of the variable.
   */
  std::string get_variable(LMP* lmp, const std::string &varname);
};

#endif // LAMMPS_RUNNER_H

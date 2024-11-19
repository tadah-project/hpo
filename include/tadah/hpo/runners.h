#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H

#include <iostream>
#include <memory>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

#include <lammps.h>
#include <input.h>
#include <variable.h>

/** 
 * @class LammpsRunner
 * @brief A class to simplify running LAMMPS simulations.
 *
 * To compile this file outside the Tadah library, 
 * specify the TADAH_LAMMPS_LIB macro with the LAMMPS library name.
 *
 * Example:
 * -DTADAH_LAMMPS_LIB="liblammps_single" 
 *
 * LAMMPS must be compiled with -DLAMMPS_EXCEPTIONS
 * to allow Tadah! to handle simulation failures gracefully.
 */
class LammpsRunner {
  public:

    // Typedef for LAMMPS instance
    typedef LAMMPS_NS::LAMMPS LMP;

    // Default constructor for LammpsRunner
    LammpsRunner();

    /** Create LAMMPS instance and run a script 
     *
     * Return a pointer to LAMMPS instance.
     *
     * In case of LAMMPS throwing exception return nullptr.
     */
    LMP* run(MPI_Comm &sub_comm, const std::string &script,
        const std::vector<const char*> &lmp_tokens);

    /**
     * @brief Retrieve the value of a LAMMPS variable.
     *
     * @param lmp Pointer to the LAMMPS instance.
     * @param varname The name of the variable to retrieve.
     * @return The value of the variable as a string.
     */
    std::string get_variable(LMP* lmp, std::string varname) {
      return lmp->input->variable->retrieve(varname.c_str());
    }
};

#endif // LAMMPS_RUNNER_H

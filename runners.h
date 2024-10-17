#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H

#include <iostream>
#include <memory>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

#include TADAH_LMPA
#include TADAH_LMPB
#include TADAH_LMPC

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
    /// Typedef for LAMMPS instance
    typedef LAMMPS_NS::LAMMPS RI;

    /// Default constructor for LammpsRunner
    LammpsRunner() {}

    /**
     * @brief Create a LAMMPS instance with specified arguments.
     *
     * @param sub_comm MPI_Comm object for sub-communication.
     * @param logfile Name of the log file. Defaults to "none".
     * @param args Additional arguments for LAMMPS initialization.
     * @return Pointer to the created LAMMPS instance.
     */
    template <typename ...Args>
      RI* instance(MPI_Comm &sub_comm,
          std::string logfile="none", Args...args) {

        const char *lmpargv[] = {TADAH_LAMMPS_LIB, "-log", logfile.c_str(), args...};
        int lmpargc = sizeof(lmpargv) / sizeof(const char *);

        // Create and return a new LAMMPS instance
        return new RI(lmpargc, (char **)lmpargv, sub_comm);
      }

    /**
     * @brief Run a LAMMPS script with given arguments.
     *
     * @param sub_comm MPI_Comm object for sub-communication.
     * @param script The path to the LAMMPS script file.
     * @param logfile Name of the log file. Defaults to "log.lammps".
     * @param args Additional arguments for the LAMMPS run.
     * @return Pointer to the LAMMPS instance, or nullptr on failure.
     */
    template <typename ...Args>
      RI* run(MPI_Comm &sub_comm, std::string script,
          std::string logfile="log.lammps", Args...args) {

        RI *lmp = nullptr;
        try {
          lmp = instance(sub_comm, logfile, args...);
          lmp->input->file(script.c_str());
        }
        catch (std::exception& e) {
          std::cout << "LAMMPS Exception: " << e.what() << std::endl;
          delete lmp;
          return nullptr;
        }
        return lmp;
      }

    RI* run(MPI_Comm &sub_comm, std::string &script,
        std::vector<const char*> &lmp_tokens) {
      //std::vector<const char*> c_str_tokens;
      //c_str_tokens.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
      //for (const auto& token : lmp_tokens) {
      //  c_str_tokens.push_back(token.c_str());
      //}
      int lmpargc = static_cast<int>(lmp_tokens.size());
      RI* lmp = nullptr;
      try {
        lmp = new RI(lmpargc, const_cast<char**>(lmp_tokens.data()), sub_comm);
        lmp->input->file(script.c_str());
      } catch (std::exception& e) {
        std::cout << "LAMMPS Exception: " << e.what() << std::endl;
        delete lmp;
        return nullptr;
      }
      return lmp;
    }

    /**
     * @brief Retrieve the value of a LAMMPS variable.
     *
     * @param lmp Pointer to the LAMMPS instance.
     * @param varname The name of the variable to retrieve.
     * @return The value of the variable as a string.
     */
    std::string get_variable(RI* lmp, std::string varname) {
      return lmp->input->variable->retrieve(varname.c_str());
    }
};

#endif // LAMMPS_RUNNER_H


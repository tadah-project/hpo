#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H

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
 * @brief Simplifies creation and use of LAMMPS instances.
 *
 * Instances are reusable for multiple runs.
 * An instance encountering an unrecoverable exception becomes invalid.
 */
class LammpsRunner {
 public:
  /// Alias for the LAMMPS class in the LAMMPS_NS namespace
  typedef LAMMPS_NS::LAMMPS LMP;

  /// Unique‐pointer type for a LAMMPS instance
  typedef std::unique_ptr<LMP> LMPPtr;

  /**
   * @brief Creates a new LAMMPS instance.
   *
   * Builds a LAMMPS instance using provided MPI sub‐communicator
   * and command‐line tokens.  If construction throws, returns nullptr.
   *
   * @param sub_comm MPI communicator for this instance.
   * @param lmp_tokens Command‐line arguments for LAMMPS.
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

  bool say_hello(const LMPPtr& lmp_ptr) const;
  /**
   * @brief Retrieves the value of a LAMMPS variable.
   *
   * Throws std::runtime_error if the instance pointer is null.
   *
   * @param lmp_ptr Unique pointer to a LAMMPS instance.
   * @param varname Name of the variable to retrieve.
   * @return String representation of the requested variable.
   * @throws std::runtime_error if lmp_ptr is null.
   */
  std::string get_variable(const LMPPtr& lmp_ptr,
                           const std::string& varname) const;
};

#endif // LAMMPS_RUNNER_H

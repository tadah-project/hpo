#ifndef LAMMPS_SIM_H
#define LAMMPS_SIM_H

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>

/**
 * @class LammpsSim
 * 
 * @brief Represents a LAMMPS simulation and provides functionality
 * for serialisation and MPI communication.
 */
class LammpsSim {
  private:
    std::string cmd; /**< Command line configuration */
    std::vector<std::string> tokens; /**< Tokenised command line */
    std::string stripComments(const std::string& in,
                                     bool& in_single,
                                     bool& in_double,
                                     bool& in_triple) const;

  public:
    std::string script; /**< Name of the LAMMPS script file */
    std::vector<std::string> script_lines; /**< Lines of the LAMMPS script */
    std::vector<std::string> invar; /**< LAMMPS input variables */
    // std::vector<const char*> invar; /**< LAMMPS  variables, ptrs to sinvars */
    std::vector<std::string> n_varloss; /**< Variables names related to the loss function */
    std::vector<double> v_varloss; /**< Value of the varloss */
    std::vector<double> w_varloss; /**< Weight for varloss */
    std::vector<std::string> outvar; /**< LAMMPS output variables */
    std::vector<std::string> v_outvar; /**< LAMMPS output variables value*/

    int ncpu = 1; /**< Number of CPUs to be used */
    std::string maxtime; /**< Maximum allowed time for simulation execution */
    double failscore; /**< Score given when a simulation fails */

    LammpsSim();

    /**
     * @brief Constructor for LammpsSim class.
     * 
     * @param cmd The command line configuration.
     * @param MAX_TIME Maximum time allowed for the simulation.
     * @param FAIL_SCORE Default fail score value.
     */
    LammpsSim(std::string &cmd, std::string MAX_TIME, double FAIL_SCORE);

    /**
     * @brief Tokenises the command line configuration.
     */
    void tokenise();

    /**
     * @brief Parses the tokenised command line to set class parameters.
     */
    void parse_tokens();

    /**
     * @brief Compares two LammpsSim objects based on the number of CPUs.
     * 
     * @param lhs Left-hand side LammpsSim object.
     * @param rhs Right-hand side LammpsSim object.
     * @return True if lhs uses more CPUs than rhs, otherwise False.
     */
    static bool greater(const LammpsSim& lhs, const LammpsSim& rhs) {
      return lhs.ncpu > rhs.ncpu;
    }

    void prep_script_lines();
};
#endif // LAMMPS_SIM_H

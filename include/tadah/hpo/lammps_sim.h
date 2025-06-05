#ifndef LAMMPS_SIM_H
#define LAMMPS_SIM_H

#include <tadah/hpo/loss_function.h>

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace tadah::hpo {
  

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

    /* default for all --varloss until the user overrides it */
    tadah::LossFunction          defLoss;

  public:
    /* one loss function per --varloss                                 */
    std::vector<tadah::LossFunction> lossFn;

    std::string script; /**< Name of the LAMMPS script file */
    std::vector<std::string> script_lines; /**< Lines of the LAMMPS script */

    // Any acceptable LAMMPS command line arguments.
    std::vector<std::string> invar;     /**< LAMMPS input variables */

    // LAMMPS output variables.
    std::vector<std::string> n_outvar;  /**< names of LAMMPS output variables */
    // std::vector<std::string> v_outvar;  /**< n_outvar values */

    // LAMMPS variables for loss calculation. Not all outvar variables
    // are used for loss calculation, so we need to specify which ones
    // are used and their target values, weights. 
    // Indices in i_varloss correspond to n_outvar indices.
    std::vector<std::string> n_varloss; /**< Use this variables for loss calculation */
    std::vector<double>      t_varloss; /**< Target value for varloss property */
    std::vector<double>      w_varloss; /**< Weight for varloss */
    std::vector<size_t>      i_varloss; /**< Map to n_outvar */

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
     * @param defaultLoss Default loss function to be used.
     *    can be overridden by user using --varloss
     */
    LammpsSim(std::string&              cmd,
              const std::string&        MAX_TIME,
              double                    FAIL_SCORE,
              const tadah::LossFunction& defaultLoss);
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
} // namespace tadah::hpo
#endif // LAMMPS_SIM_H

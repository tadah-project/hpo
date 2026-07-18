#ifndef LAMMPS_SIM_H
#define LAMMPS_SIM_H

#include <tadah/hpo/loss_function.h>

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>

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

    // Index of "variable <name> <style> ..." definitions found in the expanded script
    struct VarDef {
      std::string style;        // lowercased (e.g. "equal", "string", "atom", "delete", ...)
      std::size_t logical_cmd;  // 1-based index into script_lines for diagnostics
    };

    std::unordered_map<std::string, VarDef> index_defined_variables() const;
    void validate_requested_variables_defined_equal() const;

    /* default for all --varloss until the user overrides it */
    tadah::LossFunction          defLoss;

    static std::string expand_includes(const std::filesystem::path &file);
    static std::string expand_includes_rec(const std::filesystem::path &file,
                                           std::unordered_set<std::string> &stack);

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

    /**
     * Hot-variable declaration: `--hotvar <name> <min> <max> [<initial>]`.
     *
     * A PC_LAMMPS line carrying at least one --hotvar is a *hot* script
     * and runs in phase A of the eval (in parallel with all other hot
     * scripts). After it runs, every declared <name> is read back via
     * LAMMPS' variable mechanism; if the value is in [min, max] the
     * per-(script, name) warmstart cache is updated and the value is
     * propagated as `${<name>}` to all *cold* scripts running in phase B.
     * Out-of-range hot values fail-score the iter (phase B is skipped).
     *
     * Hot scripts also receive their own `${<name>}` injection at the
     * start of each eval, sourced from the cache (or `initial`, falling
     * back to the midpoint of [min, max] if absent on the first eval),
     * so they can use the warm-started guess as a relaxation seed.
     *
     * Multiple --hotvar entries on one line are allowed; they have no
     * cross-references (hot scripts don't see each other's outputs --
     * phase A is parallel). Cold scripts see all of them.
     */
    struct HotVarDecl {
        std::string name;
        double      min   = 0.0;
        double      max   = 0.0;
        std::optional<double> initial;
    };
    std::vector<HotVarDecl> hotvars;

    /* --failvar <name> [<scale>] -- graded-FAILSCORE Tier 2 (see
     * INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md 4.8). The LAMMPS variable
     * <name> is a nonnegative violation measure: 0 = the sim vouches for
     * itself, > 0 = "failed, this badly" (natural units; scale maps it to the
     * dimensionless band-B measure). On > 0 the sim is classified failed
     * (band B), its outvars are excluded from the fit, and scale*value grades
     * the failure loss. Multiple entries per line sum. */
    struct FailVarDecl {
        std::string name;
        double      scale = 1.0;
    };
    std::vector<FailVarDecl> failvars;

    /* Set for the PC_FAILPROBE entry only: a minimiser-free potential-sanity
     * probe dispatched at the FRONT of wave 1 (exempt from LPT ordering),
     * whose first --outvar is the shittiness scalar S(theta) used to grade
     * bands C/D (and optionally to gate the eval). Never contributes loss
     * components. */
    bool is_failprobe = false;

    LammpsSim();

    /**
     * @brief Constructor for LammpsSim class.
     *
     * @param cmd The command line configuration.
     * @param MAX_TIME Maximum time allowed for the simulation.
     * @param FAIL_SCORE Default fail score value.
     * @param defaultLoss Default loss function to be used.
     *    can be overridden by user using --varloss
     * @param failprobe Construct as a PC_FAILPROBE sim: sets is_failprobe
     *    and waives the >=1 --varloss requirement (a probe grades
     *    failures; it must not shape the feasible objective).
     */
    LammpsSim(std::string&              cmd,
              const std::string&        MAX_TIME,
              double                    FAIL_SCORE,
              const tadah::LossFunction& defaultLoss,
              bool                      failprobe = false);
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

/**
 * Strict double parser. Returns true and writes to `value` only when the
 * full string `s` is consumed by std::stod (no trailing junk). On parse
 * failure, returns false and leaves `value` unchanged. Used by the eval
 * path to treat unparseable LAMMPS variable retrievals (e.g. nullptr -> ""
 * because the variable was never defined) as failed components rather
 * than letting the std::invalid_argument escape into NLopt's wrapper --
 * NLopt translates that exception to NLOPT_INVALID_ARGS and aborts the
 * whole optimisation.
 */
bool try_parse_double(const std::string& s, double& value);

} // namespace tadah::hpo
#endif // LAMMPS_SIM_H

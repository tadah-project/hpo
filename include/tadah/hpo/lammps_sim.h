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
#include <tadah/hpo/script_file.h>

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

    /**
     * @brief The script rendered once to a node-local temp file, for
     * execution through LAMMPS's Input::file() (native jump/label/next
     * semantics -- see tadah::hpo::ScriptFile).
     *
     * Lazily created on first call and shared by copies of this sim
     * (script_lines are immutable after construction). Guarded by a
     * file-local mutex in lammps_sim.cpp: a shared_ptr member (rather
     * than a once_flag/mutex member) keeps LammpsSim copyable/movable,
     * which HPOTargets::LS vector growth requires.
     */
    const tadah::hpo::ScriptFile& script_file() const;

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

    /** Per-script wall-clock limit, a LAMMPS time spec ("off", "unlimited"
     *  or a positive [[HH:]MM:]SS): the root PC_SCRIPT_TIMEOUT in force when
     *  the line was read, or this line's --maxtime. Injected as `timer
     *  timeout`. */
    std::string maxtime;
    /** maxtime in seconds (parse_script_timeout), > 0; nullopt for "off"
     *  and "unlimited". Sizes the PROCESS worker deadline
     *  (pc_worker_deadline_ms). */
    std::optional<double> maxtime_seconds;
    /** Where maxtime comes from, for the journal: "PC_SCRIPT_TIMEOUT on line
     *  11", "MAXTIME on line 11", "its --maxtime" or "the default" (set by
     *  HPOTargets; "" for a LammpsSim made without a file). */
    std::string maxtime_source;
    /** The script has a timer command of its own: the limit is not
     *  injected as `timer timeout` (prep_script_lines). */
    bool own_timer = false;
    /** The value each --varloss / --outvar variable of this line takes when
     *  its simulation fails (--failscore; default: the root FAILSCORE in
     *  force when the line is read). On a PC_LAMMPS line any finite number;
     *  on the PC_FAILPROBE line a positive one, the S at or above which the
     *  probe counts as failed. */
    double failscore;

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
        /* <scale> is written on the line: it grades the failure, which
         * FAILSCORE_GRADING ON does, and is refused under OFF
         * (validate_hpo_setup). */
        bool        scale_given = false;
    };
    std::vector<FailVarDecl> failvars;

    /* A LAMMPS variable the line defines on LAMMPS' command line: --invar
     * <name> <value>, -var <name> <values...> or -v <name> <values...>
     * (`option` as written). Each name is defined once on a line
     * (parse_tokens), and none is a --hotvar of the line, nor, on a line
     * without a --hotvar, a --hotvar of another line its script reads
     * (HPOTargets): the run injects `variable <name> equal <value>` for a
     * hot variable, which LAMMPS refuses for a variable of another style. */
    struct LineVariable {
        std::string name;
        std::string option;
    };
    std::vector<LineVariable> line_variables;

    /* Which of the settings read above the line it uses (HPOTargets: a
     * LOSS, FAILSCORE or PC_SCRIPT_TIMEOUT line that no line after it uses
     * is refused). The LOSS: a --varloss term of the line without a loss
     * function of its own. The FAILSCORE: the line has no --failscore. The
     * PC_SCRIPT_TIMEOUT: the line has no --maxtime. */
    bool reads_default_loss = false;
    bool reads_default_failscore = true;
    bool reads_default_maxtime = true;

    /* The line of the target file that holds this PC_LAMMPS or PC_FAILPROBE
     * line (its last physical line) and its text, for the refusals the
     * startup checks make of it; 0 for a LammpsSim made without a file. */
    std::size_t line = 0;
    std::string line_text;

    /* Set for the PC_FAILPROBE entry only: a minimiser-free potential-sanity
     * probe dispatched at the FRONT of wave 1 (exempt from LPT ordering),
     * whose first --outvar is the shittiness scalar S(theta) used to grade
     * bands C/D (and optionally to gate the eval). Never contributes loss
     * components. */
    bool is_failprobe = false;

    /* Lazily-rendered ScriptFile backing script_file(); shared across
     * copies. Do not touch directly. */
    mutable std::shared_ptr<tadah::hpo::ScriptFile> script_file_;

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

    void prep_script_lines();
};

/**
 * Whether the LAMMPS script `lines` reads the variable `name`: a reference
 * ${name}, $name for a one-character name, or v_name as a whole word (v_x
 * is not read by v_xb, nor by a word that ends in v_x). The one test of
 * HPOTargets (a variable a line defines is not another line's --hotvar its
 * script reads) and of the run (the hot values a cold script is injected),
 * so the two cannot disagree.
 */
bool script_reads_variable(const std::vector<std::string>& lines,
                           const std::string& name);

/**
 * Strict double parser. Returns true and writes to `value` only when the
 * full string `s` is consumed by std::stod (no trailing junk; an
 * underflowing literal is accepted, see core::parse_number). On parse
 * failure, returns false and leaves `value` unchanged. Used by the eval
 * path to treat unparseable LAMMPS variable retrievals (e.g. nullptr -> ""
 * because the variable was never defined) as failed components rather
 * than letting the std::invalid_argument escape into NLopt's wrapper --
 * NLopt translates that exception to NLOPT_INVALID_ARGS and aborts the
 * whole optimisation. It reads what LAMMPS returns (inf and nan included,
 * for the callers to classify); the numbers of the configuration are read
 * with the number grammar of strict_parse.h (parse_number).
 */
bool try_parse_double(const std::string& s, double& value);

/**
 * The journal text of a hot value `v` outside the --hotvar window of `hv`
 * (the eval's reject line): "<name>=<v> outside [<min>, <max>]", each
 * number in the shortest round-trip form (core::format_roundtrip). It used
 * std::to_string's six decimals, which print a value below 5e-7 as
 * 0.000000 and a value one ulp above a max of 3.6 as 3.600000, the max
 * itself.
 */
std::string hotvar_reject_what(const LammpsSim::HotVarDecl& hv, double v);

} // namespace tadah::hpo
#endif // LAMMPS_SIM_H

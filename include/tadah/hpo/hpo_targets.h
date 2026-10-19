#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include "tadah/core/context.h"
#include "tadah/core/number_format.h"
#include "tadah/core/utils.h"
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/observables/observable.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/validation_metric.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/mlip/training_record.h>

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <optional>
#include <sstream>
#include <filesystem>

namespace tadah {
namespace hpo {

/* RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN] -- the rho(r) sign
 * gate of a bare DM_REAM many-body term (RHOFORM SIGNED only; rho_gates.h).
 * A candidate whose rho_raw(r) on [R_LO, RCUTMB] (default window
 * [0.15 RCUTMB, RCUTMB]) has rho_min < -tol * rho_max is rejected before
 * training as a graded band-D failure. SEED: what a failing CONFIG/WARM
 * seed does at startup (ERROR, the default, stops the run; WARN evaluates
 * it as a rejected start). */
struct RhoGateSpec {
    double                tol = 0.0;     // finite, >= 0
    std::optional<double> r_lo;          // R_LO: finite, >= 0
    std::size_t           npts = 20001;  // 2 <= NPTS <= 1e7
    bool                  seed_warn = false;
};

/* RHOBARGATE <lattice> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN] --
 * the lattice rhobar(a) gate (any RHOFORM; rho_gates.h): the monatomic
 * lattice density rhobar(a) = sum_s m_s rho_raw(d_s a) must decrease
 * strictly on a in [f_lo a0, f_hi a0] (NPTS points, numpy.linspace order);
 * a candidate with a rise is rejected before training as a graded band-D
 * failure. hcp takes the ideal c/a; a0 is its in-plane constant. */
struct RhoBarGateSpec {
    std::string lattice;                 // "bcc" | "fcc" | "hcp" (lower case)
    double      a0 = 0.0;                // finite, > 0
    double      f_lo = 0.0, f_hi = 0.0;  // finite, 0 < f_lo < f_hi
    std::size_t npts = 1001;             // 3 <= NPTS <= 1e7
    bool        seed_warn = false;
};

class HPOTargets {
private:
    struct Opti {
        std::vector<bool> opt;
        /* The line of the OPTIM line that selected each token (0: none),
         * so a second line selecting it names the first. */
        std::vector<std::size_t> line;
        std::vector<std::string> config;
        std::vector<double> lowbound;
        std::vector<double> highbound;
        /* Per-token shape tag from the OPTIM line ([LIN | LOG | ASINH c]);
         * default LIN. Consumed when the SearchSpace is built. */
        std::vector<DimTransform> scale;
        /* How each box was obtained (explicit bounds or an AUTO rule). */
        std::vector<BoxOrigin> origin;

        void resize(size_t s);
    };

    size_t ui_offset(std::vector<std::string> &tokens, size_t ui);
    std::vector<std::string> parse_optim_line(std::istringstream& iss);
    /* Throws when the run cannot create the archive directory @p dir of
     * the line @p key (DUMP, BDUMP) or write into it, and creates nothing.
     * The path is taken where the operating system puts it (every symbolic
     * link followed, '..' taken from the directory reached; one the system
     * cannot resolve is refused): one that exists must be a directory the
     * run can write into; one that does not must not be a symbolic link
     * itself (mkdir does not follow it), and must be in a directory that
     * exists and the run can write into, or in @p created_first (resolved
     * likewise; empty: none), the other archive line's directory, which
     * the run creates first and which the path is resolved through as a
     * directory that exists ("archive/." is that directory, "archive/./
     * best" a directory in it); the run creates the last directory of the
     * path only. */
    static void check_archive_directory(const std::string& key,
                                        const std::string& dir,
                                        const std::filesystem::path& created_first);
    /* The DUMP and BDUMP directories, checked together once the target
     * file is read (check_archive_directory), in the order of their lines,
     * each error at its line: one inside the other's new directory is
     * accepted, whatever the order of the lines. */
    void check_archive_directories() const;
    /* The line of the last DUMP and BDUMP line (number 0: none), for the
     * errors of check_archive_directories(). */
    struct ArchiveLine { std::size_t number = 0; std::string text; };
    ArchiveLine dumpPotLine, dumpBestPotLine;
    /* Creates the archive directory @p dir of the line @p key, which
     * check_archive_directory() accepted (one that exists is kept; each is
     * named on standard output). */
    static void create_archive_directory(const std::string& key,
                                         const std::string& dir);
    /* One OPTIM line (on line @p nline of the file). The key is
     * upper-cased: OPTIM keys are case-insensitive, as configuration keys
     * are. */
    void process_optim_config(std::istringstream& iss, std::size_t nline);
    SearchSpace build_search_space() const;
    void bind_space_recursive();

    core::Context &context;
    /* The values of a -c potential that its descriptors do not read, left
     * out of `context` for the run (D_mJoin::expand_grids): an OPTIM index
     * past the values kept names them. */
    std::vector<models::D_mJoin::LeftOut> left_out_;
    SearchSpacePtr space_;
    /* OPTIM_SCALE (root key): AUTO by default -- every OPTIM dimension is
     * mapped to the unit box; NONE is the legacy escape hatch (raw units,
     * ln for LOG dims). LOG_HP lands on spec.log_threshold. */
    ScalePolicy    scale_policy_{std::nullopt, /*normalise=*/true};
    bool           scale_policy_set_ = false;
    /* RHOFORM of the -c configuration; see rho_form(). */
    std::optional<models::RhoForm> rho_form_;

public:
    /** The targets of the run whose configuration is @p c and whose
     * hpotarget file is @p fn. @p provenance: where the keys of @p c came
     * from (the task's KeyProvenance, hpo_run passes it), for
     * training_record; none: every stored key is the configuration's. */
    HPOTargets(core::Context &c, std::string fn,
               const mlip::KeyProvenance *provenance = nullptr);
    void read(const std::string &fn);
    std::map<std::string, Opti> optimap;  // used by HPO_Host
    std::vector<LammpsSim> LS;

    /* The search space built from the OPTIM lines + OPTIM_SCALE + LOG_HP
     * (shared with spec and every sub-spec via bind_space). */
    SearchSpacePtr space() const { return space_; }
    /* Positive-only OPTIM keys (geometric AUTO boxes by default), from the
     * refit hpotarget writer's key table. */
    static bool optim_key_positive_only(const std::string& key);
    /* OPTIM on structure data is refused: the density form (RHOFORM), the
     * descriptor types (TYPE2B, TYPEMB), the cutoff classes (RCTYPE2B,
     * RCTYPEMB), the knot-degree lists (KDEG2B, KDEGMB, KDEGEMB), the
     * rigid-dimer layout (DIMER), the pinned-weight indices (FIXINDEX) and
     * the model (MODEL); compared case-insensitively. Their values are
     * discrete or describe the data, so a searched value is truncated,
     * refused or meaningless. Returns the error message for such a key,
     * "" for a searchable one. (OPTIM EMBF0 has its own message.) */
    static std::string optim_structure_key_error(const std::string& key);
    /* A discrete value of a key whose other values are continuous: the
     * first OALGO value (the training algorithm; the second, the rcond M,
     * is searchable). @p index is the OPTIM index (1-based among the key's
     * numeric values). Returns the error message, "" for a searchable
     * index. */
    static std::string optim_discrete_index_error(const std::string& key,
                                                  std::size_t index);
    /* OPTIM on an internal key (the blueprint's INTERNAL_KEYS: DSIZE,
     * RCUT*MAX, ESTDEV, ...) is refused: tadah computes those values and no
     * configuration sets them (an evaluation could not write a candidate
     * into one). Returns the error message, "" for another key. */
    static std::string optim_internal_key_error(const std::string& key);
    /* OPTIM on a legacy many-body key (CGRIDMB, SGRIDMB, AMPGRIDMB,
     * CEMBFUNC, SEMBFUNC, RCUTMB) that @p ctx does not store, while its
     * many-body term is DM_Embed components (DESCRIPTOR blocks): refused
     * naming the component path of each block that holds the parameter
     * (<ID>.DENSITY.CENTRES, <ID>.DENSITY.ETA or .WIDTHS as declared, ...).
     * A stored key (a DM_REAM term beside the blocks) is searchable.
     * Returns the error message, "" otherwise. */
    static std::string optim_component_key_error(const core::Context& ctx,
                                                 const std::string& key);
    /* OPTIM on a value that the key's formats (blueprint value_type, every
     * alternative) take only as an integer -- SBASIS, the first value of
     * OALGO -- is refused: a searched value is continuous, and an integer
     * read of a candidate such as 14.47 is refused (or was truncated).
     * @p config holds the key's tokens (Opti::config), @p pos the token of
     * OPTIM index @p index. Returns the error message, "" for a value the
     * key also takes as a fraction. */
    static std::string optim_integer_index_error(
        const std::string& key, const std::vector<std::string>& config,
        std::size_t pos, std::size_t index);
    /* The OPTIM keys that arm the density gates (RHOGATE, RHOBARGATE):
     * AMPGRIDMB, CGRIDMB, SGRIDMB, RCUTMB, the dimensions that change
     * rho_raw or the window. A gate without an OPTIM dimension on one of
     * them cannot act, and read() refuses it at the gate's line. */
    static const std::vector<std::string>& density_gate_keys();
    /* The DM_REAM density form of the -c configuration: RHOFORM, mandatory
     * on a configuration with a DM_REAM block and read at the top of the
     * constructor (the migration message when absent or invalid), before
     * the OPTIM boxes and the WARM seeds that depend on it. nullopt: no
     * DM_REAM block. Under NONNEG and MONOTONE the AMPGRIDMB boxes are
     * non-negative (process_optim_config). */
    std::optional<models::RhoForm> rho_form() const { return rho_form_; }
    /* The -c configuration these targets were read against (the reference
     * the constructor holds): the run context of every InitContext, so the
     * WARM seed rules (EMBF0, RHOFORM) apply to each resolution, a RESTART
     * refresh included. */
    const core::Context& run_context() const { return context; }
    /* hpo_search_space.tsv (always) + hpo_optim_resolved.hpo (when any
     * OPTIM box was derived by an AUTO rule) into `dir`. */
    void write_search_space_files(const std::string& dir) const;
    /* Creates the archive directories of the DUMP and BDUMP lines (one
     * that exists is kept; each is named on standard output), the one
     * inside the other's after it. The parse checks them and creates
     * nothing; hpo_run() calls this once it has prepared best_pot.tadah
     * (its lock taken), just before its first evaluation, so a run that
     * stops before creates none, and `tadah hpo --check` leaves the
     * working directory as it is. */
    void create_archive_directories() const;

    /* The names of the search-space files a run writes into the working
     * directory (write_search_space_files). */
    static constexpr const char* kSearchSpaceFile = "hpo_search_space.tsv";
    static constexpr const char* kResolvedOptimFile = "hpo_optim_resolved.hpo";

    /* The names of the files a run writes into the working directory:
     * the result files and the local best potential carry the rank suffix
     * (hpo_rank_suffix()), best_pot.tadah, the canonical best potential,
     * none. The members below (lossFile, ...) hold them;
     * check_run_outputs() checks them before the run has read its target
     * file. */
    struct OutputNames {
        std::string loss, outvar, params, stats;
        std::string bestLoss, bestOutvar, bestParams, bestStats;
        std::string canonicalBestPot, localBestPot;
    };
    static OutputNames output_names();

    /* Whether the run writes the search-space files: hpo_search_space.tsv
     * when the target file has a search space, hpo_optim_resolved.hpo when
     * a box of it comes from an AUTO rule (write_search_space_files). */
    bool writes_search_space_file() const;
    bool writes_resolved_optim_file() const;

    /* Throw when the run cannot write its output files into the working
     * directory, and create nothing (`tadah hpo --check` and the run
     * alike, before any data is loaded):
     *   - check_run_outputs(journal), before the run opens its journal: the
     *     working directory must be a directory the run can write into
     *     ("the working directory '<dir>' is not writable: tadah hpo
     *     writes its journal (hpo_run.log), its result files and
     *     best_pot.tadah there"); the journal @p journal and the result
     *     files, when they exist, files the run can open to write, and the
     *     lock of each best potential one it can open to read and write,
     *     opened as the run opens them without creating or truncating
     *     them, the refusal being the open's own error ("'loss.tadah' in
     *     the working directory '<dir>' cannot be opened: Permission
     *     denied (tadah hpo rewrites it)", "... (tadah hpo opens it to
     *     lock best_pot.tadah)", the lock's own potential); a socket is
     *     refused (the run's open fails with ENXIO), a named pipe is
     *     checked for permission and not opened, except as a result file,
     *     which the run reopens for each row and which a pipe would block
     *     (refused), and a device is opened without waiting; a best
     *     potential, which the run replaces, no directory ("... is a
     *     directory: tadah hpo writes a file of that name");
     *   - check_search_space_files(), once the target file is read: the
     *     search-space files the run writes (writes_search_space_file(),
     *     writes_resolved_optim_file()), as the result files.
     * A file the run writes through that does not exist is checked where
     * the operating system would create it, every symbolic link followed:
     * a name that ends in '/' or '/.' cannot be a file, and the file must
     * be in a directory that exists and the run can write into
     * ("'loss.tadah' ... is a symbolic link to '<dir>/x/loss', which cannot
     * be created: the directory '<dir>/x' does not exist"); a path the
     * system cannot resolve is refused. A best potential, which the run
     * replaces, is checked by its own name: a link there is replaced by
     * the file, not followed. */
    static void check_run_outputs(const std::string& journal);
    void check_search_space_files() const;

    /* Typed observables (Tier 1 / Tier 2 of the PC_OBSERVABLE redesign).
     * Each entry is a polymorphic Observable* -- concrete subclasses live in
     * tadah/hpo/observables/. The host iterates this vector alongside LS
     * when computing the loss + gradient.
     * Populated by PC_OBSERVABLE lines in the .hpotarget config. */
    std::vector<std::unique_ptr<Observable>> observables;

    // int THREADS = -1; // number of threads for parallel evaluation, -1 means all available threads
    double FAIL_SCORE = std::numeric_limits<double>::max();
    std::size_t QFLUSH_THRESHOLD = 10; // number of results to flush to disk at once
    /* Per-pc_script wall-clock limit, a LAMMPS `timer timeout` time spec
     * ("off", "unlimited" or a positive [[HH:]MM:]SS), injected into every
     * PC_LAMMPS script after each `clear`. "off" (the default) and
     * "unlimited" disable enforcement -- opt in via the root
     * PC_SCRIPT_TIMEOUT key, its legacy alias MAXTIME, or per-sim --maxtime.
     * A timed-out sim scores FAILSCORE. Distinct from the OPTIMIZER-block
     * MAXTIME (whole-run stop) and from PC_WORKER_TIMEOUT (the outer
     * per-worker-task hard deadline). */
    std::string MAX_TIME = "off";
    /* MAX_TIME in seconds (parse_script_timeout), > 0: nullopt for "off"
     * and "unlimited". Sizes the PROCESS worker deadline. */
    std::optional<double> MAX_TIME_seconds;
    /* Which root key last supplied MAX_TIME, and whether each spelling was
     * seen -- consumed by the startup diagnostic line and the both-keys
     * warning. PC_SCRIPT_TIMEOUT is the self-describing name; MAXTIME is the
     * legacy alias kept so existing .hpo configs keep parsing. */
    std::string MAX_TIME_key;           // "PC_SCRIPT_TIMEOUT" | "MAXTIME" | ""
    bool MAX_TIME_seen_new    = false;  // PC_SCRIPT_TIMEOUT was given
    bool MAX_TIME_seen_legacy = false;  // legacy MAXTIME was given

    /* PC_ISOLATION: execution mode for the pc_lammps fan-out.
     *   PROCESS (default) -- persistent worker-process pool: each fan-out
     *                        slot runs its LAMMPS instances in its own child
     *                        process, eliminating races on LAMMPS process
     *                        globals (the `free(): invalid pointer` class;
     *                        INVESTIGATIONS/PC_ISOLATION_process_pool.md).
     *                        Verified: byte-identical results vs THREAD,
     *                        <=1.02x wall at 4/8/16-wide, TSan race-free,
     *                        145k-eval ASan-clean soak.
     *   THREAD            -- DEPRECATED fallback (pre-existing in-process
     *                        behaviour; concurrent LAMMPS instances share
     *                        the address space and race on its globals).
     *                        Kept one production cycle, then removed.
     * Env override: TADAH_PC_ISOLATION (read in hpo.cpp). PROCESS is
     * rejected (hard error) in MPI builds -- those default to THREAD via
     * the hpo.cpp guard. */
    std::string PC_ISOLATION = "PROCESS";
    /* true iff the config file carried an explicit PC_ISOLATION line --
     * lets MPI builds distinguish "defaulted to PROCESS" (silently fall back
     * to THREAD) from "explicitly requested PROCESS" (hard error). */
    bool PC_ISOLATION_set = false;
    /* PC_WORKER_TIMEOUT <seconds>: PROCESS mode only -- global override of
     * the per-task deadline formula max(3*maxtime, maxtime+120s); 0 =
     * derive from each sim's maxtime ("off" => no deadline, thread parity).
     * Env override: TADAH_PC_WORKER_TIMEOUT. */
    double PC_WORKER_TIMEOUT = 0.0;
    /* Strict parser shared by the PC_WORKER_TIMEOUT key (a single value)
     * and the TADAH_PC_WORKER_TIMEOUT override: a finite number >= 0, else
     * throws naming `origin`. */
    static double parse_pc_worker_timeout(const std::string& token,
                                          const std::string& origin);

    /* FAILSCORE_GRADING ON|OFF: graded ("smooth") failure losses inside the
     * band [0.999*FAILSCORE, FAILSCORE) -- severity bands A-D graded by
     * per-band violation measures so failed evals push the optimiser toward
     * feasibility instead of returning a gradient-free constant plateau.
     * OFF (default for now) = byte-identical legacy behaviour. Requires a
     * FINITE FAILSCORE when ON (startup-validated). Design:
     * INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md. Env override:
     * TADAH_FAILSCORE_GRADING. Every optimiser spec carries a copy
     * (OptimizerSpec::failscore_grading, which the finite differences
     * read): change it with set_failscore_grading(). */
    bool FAILSCORE_GRADING = false;
    /* Sets FAILSCORE_GRADING (the TADAH_FAILSCORE_GRADING override of
     * hpo_run) and its copy in the outer spec, its INNER block, the
     * PRESTAGE exploration and that one's INNER block; the specs copied
     * from them later (MULTISTART starts, the PRESTAGE refinement, RESTART
     * rounds) keep it. */
    void set_failscore_grading(bool on);

    /* BUMPGATE CANDIDATE|STRICT: how the bump-support invariant
     * (c + w/eta <= RCUT on hard Cut_Dummy channels) is enforced over the
     * OPTIM boxes.
     *   CANDIDATE (default): startup errors only when a bump's box contains
     *     NO feasible point; otherwise every sampled combination is tested
     *     per evaluation and violating candidates return a graded band-D
     *     failure before any training work.
     *   STRICT: legacy behaviour -- startup hard-errors when the worst box
     *     corner (CGRID high bound, SGRID low bound) violates, guaranteeing
     *     zero runtime rejects (strict MAXEVAL budgeting / old-run repro).
     * The seed pot's own bumps must fit in both modes. Env override:
     * TADAH_BUMPGATE. */
    bool BUMPGATE_STRICT = false;

    /* RHOGATE / RHOBARGATE: the density gates of a bare DM_REAM many-body
     * term (RhoGateSpec, RhoBarGateSpec above; rho_gates.h). Absent -- the
     * default -- means no density gate: every evaluation is as without
     * the keys. The semantic checks (a bare DM_REAM, RHOGATE under RHOFORM
     * SIGNED only, R_LO and the first RHOBARGATE shell inside the smallest
     * reachable RCUTMB) and the seed checks run in validate_hpo_setup. No
     * environment override. */
    std::optional<RhoGateSpec>    RHOGATE;
    std::optional<RhoBarGateSpec> RHOBARGATE;

    /* PC_FAILPROBE --script <f> --outvar <S> [--gate <T>] [--s0 <v0>]
     * A minimiser-free potential-sanity probe (single-point run-0 script;
     * designed to be un-failable) dispatched at the front of wave 1 every
     * eval. Its first --outvar is the shittiness scalar S(theta) >= 0:
     *   - grades bands C/D of the failure loss (measure for hangs/kills and
     *     sick-potential failures);
     *   - optional GATE: S > <T> short-circuits the eval (band D), cancels
     *     queued sims and (PROCESS mode) kills this eval's in-flight sims.
     *     The threshold changes the objective if it can fire on feasible
     *     potentials -- calibrate conservatively; fire rate is journaled.
     * <v0> is the half-saturation scale of sigma for S (default 1.0). */
    int    failprobe_idx  = -1;          /* index into LS, -1 = not defined */
    std::string failprobe_svar;          /* name of the S outvar            */
    double failprobe_gate = std::numeric_limits<double>::infinity(); /* off */
    double failprobe_s0   = 1.0;

    OptimizerSpec spec;

    int    outputDigits = 6;      // decimal digits in scientific notation columns (DIGITS key)
    /* Significant digits of the numbers in every potential file HPO writes
     * (best_pot.tadah, BDUMP best_pot_N / DUMP pot_N archives): NUMERIC,
     * resolved once in the constructor WITHOUT default insertion
     * (core::resolve_pot_digits). Absent -> 18, an exact round trip;
     * pot_digits_explicit is true iff the configuration carried NUMERIC. */
    int    pot_digits = core::kPotDigitsDefault;
    bool   pot_digits_explicit = false;
    /* The training record of every potential file HPO writes (the same
     * files): the keys a potential does not use (DBFILE, WDBFILE,
     * VALIDATION, HPOTARGET, VERBOSE, ...: mlip::training_record_keys) as
     * the run's
     * configuration and command line gave them, taken first thing in the
     * constructor, before anything inserts a default; with the
     * constructor's provenance, the blueprint defaults no one wrote are
     * left out. A comment block above the key lines, which no loader of a
     * potential reads. */
    mlip::TrainingRecord training_record;
    size_t outputEvery = 1;
    /* Cadence over NEW BESTS (not over evaluations): best_*.tadah rows are
     * written every Nth improvement, keyed by best_iter. Was parsed as a
     * number but consumed as a boolean (`> 0`), so BOUTPUT 100 wrote exactly
     * as much as BOUTPUT 1 and the key looked broken to anyone trying to
     * shrink the artifacts -- run22 produced 1.55 GB of best_*.tadah.
     * Must be >= 1; validated at the parse site. */
    size_t bestOutputEvery = 1;
    size_t dumpPotEvery = std::numeric_limits<size_t>::max(); //  archiving potentials
    size_t dumpBestPotEvery = std::numeric_limits<size_t>::max(); // archiving best potentials
    std::string dumpPotDir = "pots";
    std::string dumpBestPotDir = "best_pots";
                           
    /* Fixed output names gain a ".r<rank>" suffix on every MPI rank so
     * multi-rank runs sharing one working directory never clobber each
     * other. The unsuffixed namespace is reserved for canonical shared
     * artifacts; the suffix is empty in non-MPI builds and 1-rank runs. */
    const std::string lossFile       = output_names().loss;
    const std::string outvarFile     = output_names().outvar;
    const std::string paramsFile     = output_names().params;
    const std::string statsFile      = output_names().stats;
    const std::string bestLossFile    = output_names().bestLoss;
    const std::string bestOutvarFile  = output_names().bestOutvar;
    const std::string bestParamsFile  = output_names().bestParams;
    const std::string bestStatsFile   = output_names().bestStats;
    /* best_pot.tadah is canonical across ranks. Multi-rank runs also keep an
     * explicit local file for every rank, including best_pot.tadah.r0. */
    const std::string canonicalBestPotFile = output_names().canonicalBestPot;
    const std::string localBestPotFile = output_names().localBestPot;
    /* Prevent rank-local archive ordinals from truncating one shared file. */
    const std::string potArchiveSuffix = hpo_rank_suffix();
    LossFunction globalLoss;

    bool use_emae = false; bool use_ermse = false; bool use_errmse = false; bool use_ersq = false;
    double emae = 0, errmse = 0, ersq = 0;
    double w_emae = 0, w_errmse = 0, w_ersq = 0;
    ValidationMetricTargets ermse_targets{"PC_ERMSE"};
    ValidationMetricTargets frmse_targets{"PC_FRMSE"};
    void resolve_validation_targets(const std::vector<std::string>& files) {
        ermse_targets.resolve(files);
        frmse_targets.resolve(files);
    }
    bool use_srmse = false;
    double srmse;
    double w_srmse;
    bool use_frmse = false;

    /* Bad-pot pre-LAMMPS skip thresholds, parsed from PC_* flags
     *   PC_ERMSE  <target> <weight> [--skip-above <cap_meV>] [--skip-rel <ratio>]
     * (and analogous for EMAE / FRMSE / SRMSE / ErRMSE / ERSQ).
     *
     * abs_above: skip the LAMMPS phase when the metric exceeds <cap>
     *            (cap is in meV/atom for energy metrics, raw units for
     *            force/stress; the same units the metric reports).
     * rel_above: skip when the metric exceeds rel_above * best_seen_so_far.
     *            Only fires when a finite best has been observed.
     *
     * Replaces the legacy env vars TADAH_HPO_BAD_POT_ERMSE_MEV /
     * TADAH_HPO_BAD_POT_ERMSE_REL. Defaults are nullopt = no skip. */
    using SkipCaps = MetricSkipCaps;
    SkipCaps skip_emae, skip_errmse, skip_ersq, skip_srmse;

    /** Why two evaluations of one point can give different losses in this
     *  run, or "" when every evaluation is a function of its point -- the
     *  training (a fresh fit of the candidate) and the validation metrics
     *  are -- so that a search may answer a point it asks for again with
     *  the loss it gave (repeat_points.h). The first that applies of:
     *  a PC_LAMMPS line (its scripts run on LAMMPS instances kept from one
     *  evaluation to the next, and a --hotvar starts from the best
     *  evaluation's value), a PC_OBSERVABLE whose observable carries state
     *  (Observable::carried_state), a --skip-rel filter (it compares with
     *  the best value so far). */
    std::string repeat_reuse_blocker() const;
};
} // namespace hpo
} // namespace tadah

#endif // HPO_TARGETS_H

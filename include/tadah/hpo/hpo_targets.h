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
#include <tadah/models/descriptors/dm/rhoform.h>

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
    void handle_directory_creation(const std::string& directory);
    void process_optim_config(std::istringstream& iss);
    SearchSpace build_search_space() const;
    void bind_space_recursive();

    core::Context &context;
    SearchSpacePtr space_;
    /* OPTIM_SCALE (root key): AUTO by default -- every OPTIM dimension is
     * mapped to the unit box; NONE is the legacy escape hatch (raw units,
     * ln for LOG dims). LOG_HP lands on spec.log_threshold. */
    ScalePolicy    scale_policy_{std::nullopt, /*normalise=*/true};
    bool           scale_policy_set_ = false;
    /* RHOFORM of the -c configuration; see rho_form(). */
    std::optional<models::RhoForm> rho_form_;

public:
    HPOTargets(core::Context &c, std::string fn);
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
     * RCTYPEMB) and the knot-degree lists (KDEG2B, KDEGMB, KDEGEMB);
     * compared case-insensitively. Returns the error message for such a
     * key, "" for a searchable one. (OPTIM EMBF0 has its own message.) */
    static std::string optim_structure_key_error(const std::string& key);
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
     * TADAH_FAILSCORE_GRADING. */
    bool FAILSCORE_GRADING = false;

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
    const std::string lossFile       = "loss.tadah"        + hpo_rank_suffix();
    const std::string outvarFile     = "outvar.tadah"      + hpo_rank_suffix();
    const std::string paramsFile     = "params.tadah"      + hpo_rank_suffix();
    const std::string statsFile      = "stats.tadah"       + hpo_rank_suffix();
    const std::string bestLossFile    = "best_loss.tadah"   + hpo_rank_suffix();
    const std::string bestOutvarFile  = "best_outvar.tadah" + hpo_rank_suffix();
    const std::string bestParamsFile  = "best_params.tadah" + hpo_rank_suffix();
    const std::string bestStatsFile   = "best_stats.tadah"  + hpo_rank_suffix();
    /* best_pot.tadah is canonical across ranks. Multi-rank runs also keep an
     * explicit local file for every rank, including best_pot.tadah.r0. */
    const std::string canonicalBestPotFile = "best_pot.tadah";
    const std::string localBestPotFile =
        std::string("best_pot.tadah") + hpo_rank_suffix();
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
};
} // namespace hpo
} // namespace tadah

#endif // HPO_TARGETS_H

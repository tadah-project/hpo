#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include "tadah/core/context.h"
#include "tadah/core/utils.h"
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/observables/observable.h>
#include <tadah/hpo/optimizer_spec.h>

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

class HPOTargets {
private:
    struct Opti {
        std::vector<bool> opt;
        std::vector<std::string> config;
        std::vector<std::string> labels;
        std::vector<double> lowbound;
        std::vector<double> highbound;

        void resize(size_t s);
        void print(const std::string &key) const;
    };

    size_t ui_offset(std::vector<std::string> &tokens, size_t ui);
    std::vector<std::string> parse_optim_line(std::istringstream& iss);
    std::vector<std::pair<std::string, std::vector<double>>> get_labels(const std::string& key);
    std::vector<std::pair<std::string, std::vector<double>>> get_d_labels(const std::string& type, const std::string& prefix);
    std::pair<std::string, std::string> determine_suffix_prefix(const std::string& key);
    void process_grid_labels(const std::string& label, Opti& o, size_t& i, const std::vector<double>& params, const std::string& key);
    void process_atoms_labels(Opti& o, size_t& i, std::string& watoms);
    void create_labels();
    void handle_directory_creation(const std::string& directory);
    void process_optim_config(std::istringstream& iss, size_t nline);
    
    core::Context &context;

public:
    HPOTargets(core::Context &c, std::string fn);
    void read(const std::string &fn);
    std::map<std::string, Opti> optimap;  // used by HPO_Host
    std::vector<LammpsSim> LS;

    /* Typed observables (Tier 1 / Tier 2 of the PC_OBSERVABLE redesign).
     * Each entry is a polymorphic Observable* -- concrete subclasses live in
     * tadah/hpo/observables/. The host iterates this vector alongside LS
     * when computing the loss + gradient.
     * Populated by PC_OBSERVABLE lines in the .hpotarget config. */
    std::vector<std::unique_ptr<Observable>> observables;

    // int THREADS = -1; // number of threads for parallel evaluation, -1 means all available threads
    double FAIL_SCORE = std::numeric_limits<double>::max();
    std::size_t QFLUSH_THRESHOLD = 10; // number of results to flush to disk at once
    /* Per-pc_script wall-clock limit in SECONDS (LAMMPS `timer timeout`
     * units), injected into every PC_LAMMPS script after each `clear`. "off"
     * disables enforcement (the default -- opt in via the root
     * PC_SCRIPT_TIMEOUT key, its legacy alias MAXTIME, or per-sim --maxtime).
     * A timed-out sim scores FAILSCORE. Distinct from the OPTIMIZER-block
     * MAXTIME (whole-run stop) and from PC_WORKER_TIMEOUT (the outer
     * per-worker-task hard deadline). */
    std::string MAX_TIME = "off";
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

    /* FAILSCORE_GRADING ON|OFF: graded ("smooth") failure losses inside the
     * band [0.999*FAILSCORE, FAILSCORE) -- severity bands A-D graded by
     * per-band violation measures so failed evals push the optimiser toward
     * feasibility instead of returning a gradient-free constant plateau.
     * OFF (default for now) = byte-identical legacy behaviour. Requires a
     * FINITE FAILSCORE when ON (startup-validated). Design:
     * INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md. Env override:
     * TADAH_FAILSCORE_GRADING. */
    bool FAILSCORE_GRADING = false;

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
    double emae, ermse, errmse, ersq;
    double w_emae, w_ermse, w_errmse, w_ersq;
    bool use_srmse = false;
    double srmse;
    double w_srmse;
    bool use_frmse = false;
    double frmse;
    double w_frmse;

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
    struct SkipCaps {
      std::optional<double> abs_above;
      std::optional<double> rel_above;
    };
    SkipCaps skip_emae, skip_ermse, skip_errmse, skip_ersq, skip_frmse, skip_srmse;
};
} // namespace hpo
} // namespace tadah

#endif // HPO_TARGETS_H

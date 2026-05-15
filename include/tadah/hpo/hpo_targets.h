#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include "tadah/core/context.h"
#include "tadah/core/utils.h"
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
    std::string MAX_TIME = "10";

    OptimizerSpec spec;

    int    outputDigits = 6;      // decimal digits in scientific notation columns (DIGITS key)
    size_t outputEvery = 1;
    size_t bestOutputEvery = 1;
    size_t dumpPotEvery = std::numeric_limits<size_t>::max(); //  archiving potentials
    size_t dumpBestPotEvery = std::numeric_limits<size_t>::max(); // archiving best potentials
    std::string dumpPotDir = "pots";
    std::string dumpBestPotDir = "best_pots";
                           
    const std::string lossFile       = "loss.tadah";
    const std::string outvarFile     = "outvar.tadah";
    const std::string paramsFile     = "params.tadah";
    const std::string statsFile      = "stats.tadah";
    const std::string bestLossFile    = "best_loss.tadah";
    const std::string bestOutvarFile  = "best_outvar.tadah";
    const std::string bestParamsFile  = "best_params.tadah";
    const std::string bestStatsFile   = "best_stats.tadah";
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

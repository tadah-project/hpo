// test_validation_follows_metric_e2e.cpp
//
// Validation follows the metric (known-issues workstream 1, branch
// fix/pc-frmse-force-off; tmp/known-issues/report_frmse.md):
//
//   * FORCE and STRESS -- written in the configuration, left at their
//     default, or set by `tadah hpo -F/-S` -- decide only what training
//     fits, exactly as in `tadah train`.
//   * HPO validation predicts forces whenever PC_FRMSE is set and stresses
//     whenever PC_SRMSE is set (and only then), whatever FORCE/STRESS the
//     potential was fitted with. best_pot.tadah keeps the TRAINING
//     FORCE/STRESS -- for MODEL M_KRR too, whose get_param_file() keeps
//     them as M_BLR's does -- and the trainer's own context is left as it
//     was.
//
// Before the fix validation predicted with the trained model's context
// (HPO_Host::operator(), `pot = host.model->get_param_file()`), so:
//   - FORCE false (written or by default) + PC_FRMSE read past an empty
//     force vector in Analytics::calc_f_rmse: SIGSEGV;
//   - STRESS false + PC_SRMSE compared the reference stresses with the
//     zero-initialised prediction: SRMSE == RMS(reference), the same for
//     every candidate, and a --skip-above cap could reject them all;
//   - MODEL M_KRR, whose get_param_file() dropped FORCE and STRESS, did both
//     even when it was fitted with FORCE/STRESS true, and its
//     best_pot.tadah stated neither (so FORCE false on a retrain);
//   - FORCE/STRESS true without PC_FRMSE/PC_SRMSE predicted forces and
//     stresses no metric read;
//   - GRAD_STRATEGY ANALYTICAL was refused at startup ("PC_FRMSE needs
//     FORCE true") and HYBRID crashed like the first item;
//   - the journal never said what training fitted or what validation
//     measured.
//
// Weight 0 (branch fix/weight0-metrics, MK 2026-10-05/06): PC_FRMSE leaves
// out the validation structures stored with force weight 0 (fweight) and
// PC_SRMSE those stored with stress weight 0 (sweight), per dataset; a
// validation dataset in which every structure has that weight 0 is not
// measured (nan, left out of the average), and a run is refused at startup
// when no dataset has the data or an indexed weight above 0 targets such a
// dataset (cases 2a, 2b and 4a). Before that fix they were measured like
// any other structure.
//
// What a validation prediction used is read from the host's test hook
// HPO_Host::last_validation_switches_for_test(). It is detected at compile
// time (validation_switches_seen()): before the host had it, this file
// compiled and every hook check failed at run time.
//
// Every reference value comes from predict_with(), an independent
// prediction built the way `tadah predict` builds one
// (MLIP/src/predict_engine.cpp): DC_Selector, DescriptorsCalc and the
// model factory on a COPY of the potential context with FORCE/STRESS set,
// on neighbour lists of its own. The RMSEs are computed here, not by
// mlip::Analytics. All data are synthetic (rattled bcc Ta cells with
// made-up energies, forces and stresses): no shipped DFT dataset, so a
// change of the stored stress convention (workstream D) cannot move a
// reference value.
//
// Hidden [valmetric_e2e][.] like the other HPO_Host end-to-end groups:
// operator() fills the thread-local trainer cache and the HPOLog /
// HPO_Host statics; the group is one CTest entry (hpo_valmetric_e2e,
// tests/CMakeLists.txt):
//   ./test_hpo "[valmetric_e2e]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/candidate_context.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
using tadah::mlip::Atom;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::DM_Function_Base;
using tadah::mlip::M_Tadah_Base;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

/* FAILSCORE of every hpotarget here: a failed evaluation returns it. */
constexpr double kFailScore = 1e9;

/* ---------------------------------------------------------------------------
 * Process state and scratch files (the pattern of test_nn_exact_e2e.cpp).
 * ------------------------------------------------------------------------- */

/* @p stem, this process id and a steady-clock count: distinct across the
 * parallel ctest processes. */
std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives (the host
 * and hpo_run() write loss.tadah, stats.tadah, best_pot.tadah, hpo_run.log
 * ... there); removed with everything in it. */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() / unique_name("tadah_valmetric_e2e_");
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

/* A trainer built for this host: the thread-local trainer cache may still
 * hold another host's model. */
struct ForceRebuildGuard {
    bool restore;
    ForceRebuildGuard() {
        restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1);
    }
    ~ForceRebuildGuard() {
        if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD");
    }
};

/* hpo_run.log of a host evaluated outside hpo_run(), in a temporary file,
 * read back after finish(). */
struct JournalFile {
    fs::path path;
    JournalFile()
        : path(fs::temp_directory_path() /
               (unique_name("tadah_valmetric_e2e_log_") + ".log")) {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
    }
    ~JournalFile() {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish() {
        HPOLog::instance().finish("TEST");
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

/* ---------------------------------------------------------------------------
 * Synthetic data: 2x2x2 conventional bcc Ta cells (16 atoms), rattled.
 * ------------------------------------------------------------------------- */

/* Made-up reference force component k of atom i of cell @p id: arbitrary
 * but distinct values (RMS about 0.2 eV/A), so a force RMSE measures every
 * predicted component. */
double ref_force(int id, std::size_t i, int k) {
    return 0.3 * std::sin(1.7 * static_cast<double>(i) +
                          0.9 * static_cast<double>(k) +
                          0.37 * static_cast<double>(id));
}

/* Made-up symmetric reference stress (x, y) of cell @p id: every one of the
 * six independent components is nonzero. */
double ref_stress(int id, int x, int y) {
    return 0.02 * std::cos(0.7 * static_cast<double>(id) +
                           1.3 * static_cast<double>(x + y) +
                           0.4 * static_cast<double>(x * y));
}

/* A 2x2x2 conventional bcc cell of lattice constant @p a (edge 2a), its 16
 * atoms shifted into the cell and rattled deterministically by up to
 * 0.006 of the edge (@p phase varies the rattle). Energy: a made-up
 * E(a)/atom = -5 + 2 (a - 3.15)^2 eV (plus a small per-cell offset), so
 * cells of different volume have genuinely different energies; forces
 * ref_force(), stress ref_stress(). Every cell height 2a >= 5.9 A exceeds
 * the 4.5 A cutoff of the potentials here. */
Structure bcc_cell(int id, double a, double phase) {
    const double L = 2.0 * a;
    Structure st;
    st.label = "rattled bcc 2x2x2 cell " + std::to_string(id);
    st.cell.load(L, 0.0, 0.0,
                 0.0, L, 0.0,
                 0.0, 0.0, L);
    const tadah::core::Element ta = PeriodicTable::find_by_symbol("Ta");
    std::size_t i = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                for (double h : {0.0, 0.25}) {
                    const int n[3] = {x, y, z};
                    double r[3];
                    for (int k = 0; k < 3; ++k) {
                        const double f =
                            0.5 * n[k] + h + 0.02 +
                            0.006 * std::sin(2.3 * static_cast<double>(i) +
                                             1.1 * static_cast<double>(k) +
                                             phase);
                        r[k] = f * L;
                    }
                    st.add_atom(Atom(ta, r[0], r[1], r[2],
                                     ref_force(id, i, 0), ref_force(id, i, 1),
                                     ref_force(id, i, 2)));
                    ++i;
                }
    REQUIRE(st.natoms() == 16u);
    st.energy = (-5.0 + 2.0 * (a - 3.15) * (a - 3.15) +
                 0.01 * std::sin(static_cast<double>(id))) *
                static_cast<double>(st.natoms());
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) st.stress(x, y) = ref_stress(id, x, y);
    return st;
}

/* Training set: five volumes, a = 2.95 ... 3.35, so the energy-only design
 * matrix has well-separated rows. */
std::vector<Structure> training_cells() {
    return {bcc_cell(1, 2.95, 0.3), bcc_cell(2, 3.05, 0.9),
            bcc_cell(3, 3.15, 1.4), bcc_cell(4, 3.25, 2.2),
            bcc_cell(5, 3.35, 2.9)};
}

/* Validation set: two other cells, inside the training volumes. */
std::vector<Structure> validation_cells() {
    return {bcc_cell(11, 3.10, 1.9), bcc_cell(12, 3.20, 0.6)};
}

/* One DBFILE with every structure (dump_to_file: 18 significant digits,
 * lossless). */
void write_db(const fs::path& path, const std::vector<Structure>& sts) {
    std::ofstream os(path);
    REQUIRE(os.good());
    for (const Structure& st : sts) st.dump_to_file(os);
    REQUIRE(os.good());
}

void write_db(const fs::path& path, const StructureDB& db) {
    std::ofstream os(path);
    REQUIRE(os.good());
    for (std::size_t i = 0; i < db.size(); ++i) db(i).dump_to_file(os);
    REQUIRE(os.good());
}

/* The structures of @p path, read the way the host reads a DBFILE. */
StructureDB read_db(const fs::path& path) {
    StructureDB db;
    db.add(path.string());
    REQUIRE(db.size() > 0u);
    return db;
}

/* ---------------------------------------------------------------------------
 * Potentials: D2_BP (3 Gaussians, Cut_Cos at 4.5 A), no many-body term, no bias,
 * raw features, so a linear model has 3 weights.
 * ------------------------------------------------------------------------- */

/* The training switches as the configuration states them: nullopt = the
 * key is not written (blueprint default false). */
struct FitSwitches {
    std::optional<bool> force;
    std::optional<bool> stress;
};

const std::vector<std::string> kBLR = {"M_BLR", "BF_Linear"};
const std::vector<std::string> kKRR = {"M_KRR", "Kern_Linear", "1"};

/* The pinned weights of the fixed potential (cases 2, 2a and 2b). */
const std::array<double, 3> kPinned = {0.8, -0.35, 0.2};

Context potential_context(const std::vector<std::string>& model,
                          const FitSwitches& sw) {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(3));
    c.add("TYPE2B",   static_cast<int>(3));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "2.6 3.3 4.0");
    c.add("SGRID2B",  "1.0 1.0 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   4.5);
    /* No TYPEMB/RCTYPEMB/RCUTMB with INITMB false, as in a configuration a
     * user writes: best_pot.tadah is read back with the configuration
     * parser, which refuses "TYPEMB DM_Dummy Ta Ta" (its schema needs a
     * number after the type). */
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    if (sw.force)  c.add("FORCE",  *sw.force  ? "true" : "false");
    if (sw.stress) c.add("STRESS", *sw.stress ? "true" : "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    for (const std::string& tok : model) c.add("MODEL", tok);
    return c;
}

/* A host that fits every weight per evaluation: the normal equations with
 * a ridge (OALGO 4, LAMBDA 1e-3, the setup the analytical gradient
 * covers), a training DBFILE and, unless @p val_cfg is empty, a
 * --validation file. */
Context trained_context(const std::vector<std::string>& model,
                        const FitSwitches& sw, const fs::path& train_db,
                        const fs::path& val_cfg, const fs::path& hpo_cfg) {
    Context c = potential_context(model, sw);
    c.add("LAMBDA",     1e-3);
    c.add("OALGO",      4);
    c.add("DBFILE",     train_db.string());
    if (!val_cfg.empty()) c.add("VALIDATION", val_cfg.string());
    c.add("HPOTARGET",  hpo_cfg.string());
    return c;
}

/* The fixed potential of a no-training run: FORCE and STRESS false
 * written, every weight pinned (FIXINDEX 1-3, FIXWEIGHT kPinned), no
 * DBFILE. */
Context pinned_context(const fs::path& val_cfg, const fs::path& hpo_cfg) {
    Context c = potential_context(kBLR, FitSwitches{false, false});
    c.add("LAMBDA",   0);
    c.add("OALGO",    1);
    c.add("FIXINDEX", "1-3");
    for (double w : kPinned) c.add("FIXWEIGHT", w);
    if (!val_cfg.empty()) c.add("VALIDATION", val_cfg.string());
    c.add("HPOTARGET", hpo_cfg.string());
    return c;
}

/* The same fixed potential as a potential file states it (WEIGHTS), for
 * predict_with() before any host has run. */
Context pinned_potential() {
    Context c = potential_context(kBLR, FitSwitches{false, false});
    c.add("LAMBDA", 0);
    for (double w : kPinned) c.add("WEIGHTS", w);
    return c;
}

/* The OPTIM dimensions of the trained hosts: the third Gaussian's centre
 * and the second Gaussian's width (infinitely differentiable in both). The
 * optimiser vector is ordered by key: p(0) = CGRID2B[3], p(1) =
 * SGRID2B[2]. Neither is a cutoff, so the neighbour lists are built once
 * at RCUTMAX = 4.5. */
constexpr const char* kTrainedOptim =
    "OPTIM CGRID2B (3) 3.7 4.3\n"
    "OPTIM SGRID2B (2) 0.6 1.6\n";
const std::vector<std::string> kTrainedLabels = {"CGRID2B[3]", "SGRID2B[2]"};

/* The candidate the trained hosts evaluate: inside both boxes, and not the
 * configuration values (4.0, 1.0). */
ParamVec trained_point() {
    ParamVec p(2);
    p(0) = 3.95;
    p(1) = 1.1;
    return p;
}

/* A derivative-free optimiser block: the cases call the objective directly
 * (cases 1, 2, 2a, 3 and 6) or let hpo_run() evaluate once (cases 2b and
 * 5). */
constexpr const char* kDerivativeFree =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n";

/* A gradient-based optimiser block with GRAD_STRATEGY @p strategy. */
std::string gradient_block(const std::string& strategy) {
    return "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nGRAD_STRATEGY " +
           strategy + "\nENDOPTIMIZER\n";
}

/* The root lines every hpotarget here shares: L2 loss, a finite FAILSCORE
 * and DIGITS 15, so stats.tadah carries 16 significant digits (relative
 * rounding <= 5e-16). */
std::string target_head(const std::string& optimizer) {
    return optimizer + "LOSS L2\nFAILSCORE 1e9\nDIGITS 15\n";
}

/* ---------------------------------------------------------------------------
 * The independent reference prediction and its RMSEs.
 * ------------------------------------------------------------------------- */

/* @p db predicted with the potential @p pot_ctx the way `tadah predict`
 * does it (MLIP/src/predict_engine.cpp): on a COPY of the potential context
 * with FORCE = @p force and STRESS = @p stress, DC_Selector,
 * DescriptorsCalc and the model factory, on a copy of @p db with neighbour
 * lists of its own at the potential's RCUTMAX. Whatever FORCE/STRESS
 * @p pot_ctx states (or omits, as M_KRR's did before the fix) does not
 * matter. */
StructureDB predict_with(const Context& pot_ctx, const StructureDB& db,
                         bool force, bool stress) {
    Context c = pot_ctx;
    c.remove("FORCE");
    c.add("FORCE", force ? "true" : "false");
    c.remove("STRESS");
    c.add("STRESS", stress ? "true" : "false");
    StructureDB st = db;
    tadah::models::DC_Selector DCS(c);
    NNFinder().calc(st, c);
    DescriptorsCalc<> dc(c, *DCS.d2b, *DCS.dmb, *DCS.c2b, *DCS.cmb);
    std::unique_ptr<DM_Function_Base> fb(
        tadah::core::factory<DM_Function_Base, const Context&>(
            c.get<std::string>("MODEL", 1), c));
    std::unique_ptr<M_Tadah_Base> model(
        tadah::core::factory<M_Tadah_Base, DM_Function_Base&, Context&>(
            c.get<std::string>("MODEL", 0), *fb, c));
    StructureDB pred = model->predict(c, st, dc);
    REQUIRE(pred.size() == db.size());
    if (force)
        for (std::size_t i = 0; i < db.size(); ++i)
            REQUIRE(pred(i).atoms.size() == db(i).natoms());
    return pred;
}

/* Energy RMSE per atom in meV/atom (the unit of PC_ERMSE and stats.tadah)
 * over every structure of @p ref (one DBFILE: no per-dataset mean). */
double ermse_mev(const StructureDB& ref, const StructureDB& pred) {
    double s = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = (ref(i).energy - pred(i).energy) /
                         static_cast<double>(ref(i).natoms());
        s += d * d;
    }
    return 1000.0 * std::sqrt(s / static_cast<double>(ref.size()));
}

/* Force-component RMSE (eV/A) over every atom of every structure. */
double frmse_of(const StructureDB& ref, const StructureDB& pred) {
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(pred(i).atoms.size() == ref(i).natoms());
        for (std::size_t a = 0; a < ref(i).natoms(); ++a)
            for (int k = 0; k < 3; ++k) {
                const double d =
                    ref(i).atoms[a].force(k) - pred(i).atoms[a].force(k);
                s += d * d;
                ++n;
            }
    }
    return std::sqrt(s / static_cast<double>(n));
}

/* Stress RMSE over the six independent components (x <= y) of every
 * structure, as PC_SRMSE measures it. */
double srmse_of(const StructureDB& ref, const StructureDB& pred) {
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
        for (int x = 0; x < 3; ++x)
            for (int y = x; y < 3; ++y) {
                const double d = ref(i).stress(x, y) - pred(i).stress(x, y);
                s += d * d;
                ++n;
            }
    return std::sqrt(s / static_cast<double>(n));
}

/* frmse_of() over the structures of @p ref stored with a nonzero force
 * weight only: a structure with fweight 0 is left out as PC_FRMSE leaves
 * it out, its components counted neither in the sum nor in N. @p ref is
 * one dataset (one DBFILE), with at least one such structure. */
double frmse_over_weighted(const StructureDB& ref, const StructureDB& pred) {
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref(i).fweight == 0.0) continue;
        REQUIRE(pred(i).atoms.size() == ref(i).natoms());
        for (std::size_t a = 0; a < ref(i).natoms(); ++a)
            for (int k = 0; k < 3; ++k) {
                const double d =
                    ref(i).atoms[a].force(k) - pred(i).atoms[a].force(k);
                s += d * d;
                ++n;
            }
    }
    REQUIRE(n > 0u);
    return std::sqrt(s / static_cast<double>(n));
}

/* srmse_of() over the structures of @p ref stored with a nonzero stress
 * weight only: a structure with sweight 0 is left out as PC_SRMSE leaves
 * it out. @p ref is one dataset (one DBFILE), with at least one such
 * structure. */
double srmse_over_weighted(const StructureDB& ref, const StructureDB& pred) {
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref(i).sweight == 0.0) continue;
        for (int x = 0; x < 3; ++x)
            for (int y = x; y < 3; ++y) {
                const double d = ref(i).stress(x, y) - pred(i).stress(x, y);
                s += d * d;
                ++n;
            }
    }
    REQUIRE(n > 0u);
    return std::sqrt(s / static_cast<double>(n));
}

/* The mean of per-dataset values (one per DBFILE of the --validation
 * file): the [avg] column of stats.tadah. */
double mean_over_datasets(const std::vector<double>& per_dataset) {
    REQUIRE_FALSE(per_dataset.empty());
    double s = 0.0;
    for (double v : per_dataset) s += v;
    return s / static_cast<double>(per_dataset.size());
}

/* RMS of the six independent stress components of @p db: the SRMSE of a
 * zero prediction, the value PC_SRMSE reported before the fix. */
double rms_stress(const StructureDB& db) {
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < db.size(); ++i)
        for (int x = 0; x < 3; ++x)
            for (int y = x; y < 3; ++y) {
                s += db(i).stress(x, y) * db(i).stress(x, y);
                ++n;
            }
    return std::sqrt(s / static_cast<double>(n));
}

/* +1 / -1 by the parity of the indices: an RMSE of delta * sign() over any
 * set of components is exactly delta. */
double sign_of(std::size_t i, std::size_t j, std::size_t k) {
    return ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
}

/* The boolean @p key of @p ctx as the code reads it (a missing key is the
 * blueprint default), without touching @p ctx. */
bool effective(const Context& ctx, const std::string& key) {
    Context c = ctx;
    return c.get<bool>(key);
}

/* The token @p key stores in @p ctx, or "<not stored>"; never inserts a
 * default. A failed CHECK on it prints both sides. */
std::string stored_text(const Context& ctx, const std::string& key) {
    return ctx.stored_value(key).value_or("<not stored>");
}

/* ---------------------------------------------------------------------------
 * Reading the run's files back.
 * ------------------------------------------------------------------------- */

/* Every data row of stats.tadah, by column title (#ITERATION, SLOT,
 * ERMSE[1], ERMSE[avg], ...), in file (iteration) order. */
std::vector<std::map<std::string, double>> stats_rows(const std::string& file) {
    std::ifstream in(file);
    REQUIRE(in.good());
    std::string head;
    REQUIRE(std::getline(in, head));
    std::vector<std::string> cols;
    {
        std::istringstream hs(head);
        for (std::string c; hs >> c;) cols.push_back(c);
    }
    std::vector<std::map<std::string, double>> rows;
    for (std::string line; std::getline(in, line);) {
        const auto p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;
        std::istringstream ls(line);
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) tok.push_back(t);
        INFO("stats.tadah header: " << head << "\nrow: " << line);
        REQUIRE(tok.size() == cols.size());
        std::map<std::string, double> row;
        for (std::size_t k = 0; k < cols.size(); ++k)
            row[cols[k]] = std::stod(tok[k]);
        rows.push_back(std::move(row));
    }
    return rows;
}

double column(const std::map<std::string, double>& row,
              const std::string& title) {
    const auto it = row.find(title);
    INFO("stats.tadah column " << title);
    REQUIRE(it != row.end());
    return it->second;
}

/* A potential file parsed the way `tadah predict` parses it. */
Context read_potential(const fs::path& path) {
    INFO("potential file " << path.string());
    REQUIRE(fs::exists(path));
    return Context(tadah::core::CoreUserConfigParser().parse(path.string()));
}

/* A loss an evaluation computed: finite, positive and below the FAILSCORE
 * a failed (or skipped) evaluation returns. */
void require_real_loss(double loss) {
    CAPTURE(loss);
    REQUIRE(std::isfinite(loss));
    REQUIRE(loss > 0.0);
    REQUIRE(loss < kFailScore);
}

/* The FORCE/STRESS the most recent validation prediction on this thread
 * used, as HPO_Host::last_validation_switches_for_test() reports them. */
struct ValidationSwitchesSeen {
    bool hook = false;      // the host has the test hook
    bool recorded = false;  // it reported a prediction (nullopt before any)
    bool force = false;
    bool stress = false;
};

/* Whether H has the static test hook last_validation_switches_for_test(). */
template <typename H, typename = void>
struct has_validation_switch_hook : std::false_type {};
template <typename H>
struct has_validation_switch_hook<
    H, std::void_t<decltype(H::last_validation_switches_for_test())>>
    : std::true_type {};

/* H::last_validation_switches_for_test() when H has it; hook == false
 * otherwise, so this file compiles without the hook and a check of `hook`
 * fails at run time instead. */
template <typename H>
ValidationSwitchesSeen validation_switches_seen() {
    ValidationSwitchesSeen out;
    if constexpr (has_validation_switch_hook<H>::value) {
        out.hook = true;
        const auto sw = H::last_validation_switches_for_test();
        if (sw) {
            out.recorded = true;
            out.force    = static_cast<bool>(sw->force);
            out.stress   = static_cast<bool>(sw->stress);
        }
    }
    return out;
}

/* Checks that the hook exists and that the last validation prediction
 * used FORCE = @p force and STRESS = @p stress. */
void check_validation_switches(const ValidationSwitchesSeen& seen,
                               bool force, bool stress) {
    INFO("HPO_Host::last_validation_switches_for_test() "
         << (seen.hook ? "exists" : "does not exist"));
    CHECK(seen.hook);
    if (!seen.hook) return;
    CHECK(seen.recorded);
    if (!seen.recorded) return;
    CAPTURE(seen.force, seen.stress);
    CHECK(seen.force == force);
    CHECK(seen.stress == stress);
}

/* What one evaluation of a fresh host left behind. */
struct EvalOutcome {
    double loss = 0.0;
    Context master;                       // the -c context the host got
    std::map<std::string, double> stats;  // the stats.tadah row
    Context pot;                          // best_pot.tadah
    Context trainer_ctx;                  // the trainer's context afterwards
    std::vector<double> live_weights;     // the trainer's weights
    /* The trained model's get_param_file() after the evaluation: what
     * `tadah train` writes to pot.tadah (MLIP/src/train_engine.cpp). */
    Context model_pot;
    /* The FORCE/STRESS of the evaluation's validation prediction, read
     * right after operator() returned. */
    ValidationSwitchesSeen validation_switches;
};

/* Validates the setup like hpo_run() does, builds a fresh host on @p ctx
 * and evaluates it once at @p p (the run's first evaluation, so its best).
 * The working directory receives the run's files. */
EvalOutcome evaluate_once(Context ctx, const fs::path& hpo_cfg,
                          const fs::path& val_cfg, const ParamVec& p,
                          bool has_train_db, std::size_t n_train,
                          std::size_t n_val) {
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, has_train_db, true));
    EvalOutcome out;
    out.master = ctx;
    {
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.train_size() == n_train);
        REQUIRE(host.val_size() == n_val);
        host.prepare_best_publication();
        REQUIRE_NOTHROW(out.loss = host(p));
        out.validation_switches = validation_switches_seen<Host>();
        host.flush_remaining();
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        out.trainer_ctx = tr->context;
        const auto& w = tr->model->get_weights();
        out.live_weights.assign(w.begin(), w.end());
        out.model_pot = tr->model->get_param_file();
    }
    const auto rows = stats_rows(trg.statsFile);
    REQUIRE(rows.size() == 1u);
    out.stats = rows.front();
    out.pot = read_potential(trg.canonicalBestPotFile);
    Host::reset_globals_for_test();
    return out;
}

/* How hpo_run() ended on the fixed potential (pinned_context). */
struct RunStart {
    bool threw = false;          // hpo_run() threw
    bool runtime_error = false;  // ... a std::runtime_error
    std::string what;            // the exception's message
    int status = -1;             // what hpo_run() returned when it did
};

/* hpo_run() of the fixed potential (pinned_context) with the --validation
 * file @p val_cfg and the hpotarget @p hpo_cfg, in the working directory:
 * whether it threw (a startup error: hpo_run() records it in the journal
 * and rethrows) and the message, or its status. The journal is closed
 * either way. */
RunStart start_pinned_run(const fs::path& val_cfg, const fs::path& hpo_cfg) {
    if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
    HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    RunStart out;
    {
        ForceRebuildGuard rebuild;
        Context ctx = pinned_context(val_cfg, hpo_cfg);
        try {
            out.status = tadah::hpo::hpo_run(ctx);
        } catch (const std::runtime_error& ex) {
            out.threw = true;
            out.runtime_error = true;
            out.what = ex.what();
        } catch (const std::exception& ex) {
            out.threw = true;
            out.what = ex.what();
        }
    }
    if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
    HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    return out;
}

/* The weights an independent Trainer fits at the candidate @p p of the -c
 * context @p master (FORCE set to @p force when given): the candidate
 * context and the data are made as HPO_Host makes them
 * (update_candidate_context, apply_load_transforms, one-time lists at
 * nn_safe_rcutmax), so the arithmetic is the host's. */
std::vector<double> independent_weights(const Context& master,
                                        const fs::path& hpo_cfg,
                                        const ParamVec& p,
                                        std::optional<bool> force = {}) {
    Context m = master;
    if (force) {
        m.remove("FORCE");
        m.add("FORCE", *force ? "true" : "false");
    }
    HPOTargets trg(m, hpo_cfg.string());
    Context cand = m;
    tadah::hpo::detail::update_candidate_context(cand, trg, p, false, false);
    StructureDB db(m);
    (void)tadah::mlip::apply_load_transforms(db, m);
    NNFinder().calc(db, tadah::hpo::detail::nn_safe_rcutmax(m, trg));
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer tr(cand, ws);
    const StructureDB& cdb = db;
    tr.train(cdb);
    const auto& w = tr.model->get_weights();
    return std::vector<double>(w.begin(), w.end());
}

/* The weights of a potential file. */
std::vector<double> weights_of(const Context& pot) {
    Context c = pot;
    std::vector<double> w(c.size("WEIGHTS"));
    c.get("WEIGHTS", w);
    return w;
}

/* The POTENTIAL SETUP section of a journal: from its header to the next
 * section header. */
std::string potential_setup(const std::string& journal) {
    const auto at = journal.find("-- POTENTIAL SETUP");
    if (at == std::string::npos) return {};
    const auto end = journal.find("\n-- ", at + 1);
    return journal.substr(at, end == std::string::npos ? std::string::npos
                                                       : end - at);
}

/* The value of the journal entry @p key in @p text ("  <key>   : <value>"):
 * the text after the first ':' that follows the key, leading blanks
 * removed; "" when no line holds the key. */
std::string entry_value(const std::string& text, const std::string& key) {
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        const auto at = line.find(key);
        if (at == std::string::npos) continue;
        const auto colon = line.find(':', at + key.size());
        if (colon == std::string::npos) continue;
        const auto v = line.find_first_not_of(' ', colon + 1);
        return v == std::string::npos ? std::string() : line.substr(v);
    }
    return {};
}

}  // namespace

/* Case 1. Energy-only training (FORCE false written, FORCE absent, FORCE
 * false with STRESS true) validated with PC_ERMSE + PC_FRMSE: the
 * evaluation is a real one, its ERMSE and FRMSE (stats.tadah, 16
 * significant digits) and its loss equal those of predict_with() on the
 * evaluation's own potential (best_pot.tadah, 18 significant digits, so
 * bit-exact weights) to 1e-12 relative -- the two run the same model
 * arithmetic on the same weights and lists and differ only in summation
 * order (~1e-15) and stats.tadah rounding (5e-16). best_pot.tadah and the
 * trainer's context keep the training FORCE/STRESS, and the weights are
 * bitwise those of an independent Trainer fitted with that FORCE (which a
 * FORCE-true fit would not give). The validation prediction used FORCE
 * true and STRESS false (the test hook): no stresses, even where training
 * fitted them, since no PC_SRMSE reads them.
 * Before the fix: SIGSEGV in Analytics::calc_f_rmse (validation predicted
 * no forces because the training FORCE is false). Before the host had
 * last_validation_switches_for_test(), the hook check failed. */
TEST_CASE("valmetric e2e: an energy-only fit is validated on forces when "
          "PC_FRMSE asks for them", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    FitSwitches sw;
    SECTION("FORCE false written in the configuration") {
        sw.force = false;
    }
    SECTION("no FORCE key (the default false)") {}
    SECTION("FORCE false with STRESS true") {
        sw.force = false;
        sw.stress = true;
    }
    const bool fit_stress = sw.stress.value_or(false);
    CAPTURE(sw.force.has_value(), fit_stress);

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_db(train_db, training_cells());
    write_db(val_db, validation_cells());
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
    std::ofstream(hpo_cfg) << target_head(kDerivativeFree)
                           << "PC_ERMSE 0 0.0001\nPC_FRMSE 0 1\n"
                           << kTrainedOptim;

    const ParamVec p = trained_point();
    const EvalOutcome e =
        evaluate_once(trained_context(kBLR, sw, train_db, val_cfg, hpo_cfg),
                      hpo_cfg, val_cfg, p, true, 5u, 2u);
    require_real_loss(e.loss);

    /* The reported metrics are those of the evaluation's own potential,
     * predicted with forces. */
    const StructureDB vdb = read_db(val_db);
    const StructureDB pred = predict_with(e.pot, vdb, /*force=*/true,
                                          /*stress=*/false);
    const double ermse = ermse_mev(vdb, pred);
    const double frmse = frmse_of(vdb, pred);
    CAPTURE(ermse, frmse);
    REQUIRE(std::isfinite(frmse));
    REQUIRE(frmse > 0.0);
    /* ERMSE: also an absolute 1e-9 meV/atom (1e-12 eV/atom, ~1e3 ulp of
     * the 5 eV/atom energies) for a residual small enough that a
     * last-bit difference of the total energy shows at 1e-12. */
    CHECK(column(e.stats, "ERMSE[avg]") ==
          Approx(ermse).epsilon(1e-12).margin(1e-9));
    CHECK(column(e.stats, "FRMSE[avg]") == Approx(frmse).epsilon(1e-12));
    /* L2: the mean of 1e-4 ERMSE^2 and FRMSE^2. */
    CHECK(e.loss == Approx((1e-4 * ermse * ermse + frmse * frmse) / 2.0)
                        .epsilon(1e-12));

    /* Validation predicted forces (PC_FRMSE) and no stresses (no
     * PC_SRMSE), whatever training fitted. */
    check_validation_switches(e.validation_switches, /*force=*/true,
                              /*stress=*/false);

    /* best_pot.tadah keeps the training switches. */
    CHECK_FALSE(effective(e.pot, "FORCE"));
    CHECK(effective(e.pot, "STRESS") == fit_stress);
    if (sw.force) CHECK(e.pot.has_stored("FORCE"));

    /* The trainer's context is left with the training switches. */
    CHECK_FALSE(effective(e.trainer_ctx, "FORCE"));
    CHECK(effective(e.trainer_ctx, "STRESS") == fit_stress);
    if (sw.force)
        CHECK(e.trainer_ctx.stored_value("FORCE") ==
              std::optional<std::string>("false"));
    if (sw.stress)
        CHECK(e.trainer_ctx.stored_value("STRESS") ==
              std::optional<std::string>("true"));

    /* The fit is the energy-only one: bitwise the weights of an
     * independent Trainer with FORCE false, live and in best_pot.tadah.
     * A force fit gives other weights, so the comparison tells them
     * apart. */
    const std::vector<double> w_energy = independent_weights(e.master, hpo_cfg, p);
    const std::vector<double> w_force =
        independent_weights(e.master, hpo_cfg, p, /*force=*/true);
    REQUIRE(w_energy.size() == 3u);
    REQUIRE(w_force.size() == w_energy.size());
    REQUIRE(w_force != w_energy);
    const std::vector<double> w_pot = weights_of(e.pot);
    REQUIRE(e.live_weights.size() == w_energy.size());
    REQUIRE(w_pot.size() == w_energy.size());
    for (std::size_t k = 0; k < w_energy.size(); ++k) {
        CAPTURE(k, e.live_weights[k], w_pot[k], w_energy[k]);
        CHECK(same_bits(e.live_weights[k], w_energy[k]));
        CHECK(same_bits(w_pot[k], w_energy[k]));
    }
}

/* Case 2. A potential that is not trained (no training DBFILE, every
 * weight pinned by FIXINDEX/FIXWEIGHT) with FORCE and STRESS false
 * written. The validation references are the potential's own prediction
 * (predict_with, before the run) plus delta * (+-1) on every force
 * component and/or delta_s * (+-1) on every independent stress component,
 * so the exact FRMSE is delta and the exact SRMSE is delta_s.
 *   - FRMSE == delta (0.01 eV/A) and SRMSE == delta_s to 1e-10 relative:
 *     the references are written with 18 significant digits, so
 *     ref - pred = delta +- ulp(|pred|), a relative error ulp(|pred|) /
 *     delta: below 1e-12 for forces under 10 eV/A against delta = 0.01,
 *     and below 1e-13 for stresses (|pred| within a few RMS, delta_s = 1%
 *     of it);
 *   - both also equal predict_with() on best_pot.tadah to 1e-12;
 *   - delta_s is 1% of the RMS of the predicted stresses, so RMS(reference)
 *     >> delta_s, and 'PC_SRMSE 0 1 --skip-above 10 delta_s' accepts the
 *     candidate;
 *   - the validation prediction used FORCE = PC_FRMSE set and STRESS =
 *     PC_SRMSE set (the test hook).
 * The stress-only SECTION runs first, so that before the fix the group
 * failed there without a crash.
 * Before the fix: SRMSE == RMS(reference) (~100 delta_s), above the skip
 * cap, so the candidate was skipped (loss = FAILSCORE); with PC_FRMSE,
 * SIGSEGV in Analytics::calc_f_rmse. Before the host had
 * last_validation_switches_for_test(), the hook check failed. */
TEST_CASE("valmetric e2e: pinned weights are validated on forces and "
          "stresses that were not fitted", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    bool use_f = false, use_s = false;
    SECTION("PC_SRMSE with STRESS false") {
        use_s = true;
    }
    SECTION("PC_FRMSE with FORCE false") {
        use_f = true;
    }
    SECTION("PC_FRMSE and PC_SRMSE with FORCE and STRESS false") {
        use_f = true;
        use_s = true;
    }
    CAPTURE(use_f, use_s);

    ScratchDir sd;
    const fs::path raw_db  = sd.dir / "cells.db";
    const fs::path val_db  = sd.dir / "val.db";
    const fs::path val_cfg = sd.dir / "config.val";
    const fs::path hpo_cfg = sd.dir / "config.hpo";

    /* The cells as the host will read them, predicted by the fixed
     * potential with what the metrics ask for. */
    write_db(raw_db, validation_cells());
    StructureDB cells = read_db(raw_db);
    const StructureDB pred0 = predict_with(pinned_potential(), cells, use_f, use_s);

    const double delta_f = 0.01;
    double delta_s = 0.0;
    if (use_s) {
        const double rms_pred = rms_stress(pred0);
        CAPTURE(rms_pred);
        REQUIRE(std::isfinite(rms_pred));
        REQUIRE(rms_pred > 0.0);
        delta_s = 1e-2 * rms_pred;
    }
    for (std::size_t i = 0; i < cells.size(); ++i) {
        Structure& st = cells(i);
        if (use_f)
            for (std::size_t a = 0; a < st.natoms(); ++a)
                for (int k = 0; k < 3; ++k)
                    st.atoms[a].force(k) =
                        pred0(i).atoms[a].force(k) +
                        delta_f * sign_of(i, a, static_cast<std::size_t>(k));
        if (use_s)
            for (int x = 0; x < 3; ++x)
                for (int y = x; y < 3; ++y) {
                    st.stress(x, y) =
                        pred0(i).stress(x, y) +
                        delta_s * sign_of(i, static_cast<std::size_t>(x),
                                          static_cast<std::size_t>(y));
                    st.stress(y, x) = st.stress(x, y);
                }
    }
    write_db(val_db, cells);
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
    {
        std::ofstream os(hpo_cfg);
        os << target_head(kDerivativeFree);
        if (use_f) os << "PC_FRMSE 0 1\n";
        if (use_s)
            os << "PC_SRMSE 0 1 --skip-above " << std::setprecision(17)
               << 10.0 * delta_s << "\n";
        os << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
    }

    /* FIXWEIGHT (1) evaluated at its pinned value: the weights are
     * kPinned. */
    ParamVec p(1);
    p(0) = kPinned[0];
    const EvalOutcome e = evaluate_once(pinned_context(val_cfg, hpo_cfg),
                                        hpo_cfg, val_cfg, p, false, 0u, 2u);
    require_real_loss(e.loss);

    /* The fixture: the host's potential predicts what the references were
     * built from (same weights, same model), and the stresses are far
     * from zero. */
    const StructureDB vdb = read_db(val_db);
    const StructureDB pred = predict_with(e.pot, vdb, use_f, use_s);
    for (std::size_t i = 0; i < vdb.size(); ++i) {
        if (use_f)
            for (std::size_t a = 0; a < vdb(i).natoms(); ++a)
                for (int k = 0; k < 3; ++k)
                    REQUIRE(pred(i).atoms[a].force(k) ==
                            Approx(pred0(i).atoms[a].force(k))
                                .epsilon(1e-12).margin(1e-14));
        if (use_s)
            for (int x = 0; x < 3; ++x)
                for (int y = x; y < 3; ++y)
                    REQUIRE(pred(i).stress(x, y) ==
                            Approx(pred0(i).stress(x, y))
                                .epsilon(1e-12).margin(1e-14));
    }

    double expected_loss = 0.0;
    int n_terms = 0;
    if (use_f) {
        const double frmse = frmse_of(vdb, pred);
        const double reported = column(e.stats, "FRMSE[avg]");
        CAPTURE(frmse, reported, delta_f);
        CHECK(frmse == Approx(delta_f).epsilon(1e-10));
        CHECK(reported == Approx(delta_f).epsilon(1e-10));
        CHECK(reported == Approx(frmse).epsilon(1e-12));
        expected_loss += delta_f * delta_f;
        ++n_terms;
    }
    if (use_s) {
        const double rms_ref = rms_stress(vdb);
        const double srmse = srmse_of(vdb, pred);
        const double reported = column(e.stats, "SRMSE[avg]");
        CAPTURE(rms_ref, srmse, reported, delta_s);
        REQUIRE(rms_ref > 50.0 * delta_s);
        CHECK(srmse == Approx(delta_s).epsilon(1e-10));
        CHECK(reported == Approx(delta_s).epsilon(1e-10));
        CHECK(reported == Approx(srmse).epsilon(1e-12));
        CHECK(reported < 10.0 * delta_s);   // under the --skip-above cap
        expected_loss += delta_s * delta_s;
        ++n_terms;
    }
    CHECK(e.loss == Approx(expected_loss / n_terms).epsilon(1e-10));

    /* Validation predicted what the metrics measure. */
    check_validation_switches(e.validation_switches, use_f, use_s);

    /* Nothing was fitted with forces or stresses, and nothing says so. */
    CHECK(e.pot.stored_value("FORCE") == std::optional<std::string>("false"));
    CHECK(e.pot.stored_value("STRESS") == std::optional<std::string>("false"));
    CHECK(e.trainer_ctx.stored_value("FORCE") ==
          std::optional<std::string>("false"));
    CHECK(e.trainer_ctx.stored_value("STRESS") ==
          std::optional<std::string>("false"));
    REQUIRE(e.live_weights.size() == kPinned.size());
    for (std::size_t k = 0; k < kPinned.size(); ++k)
        CHECK(same_bits(e.live_weights[k], kPinned[k]));
}

/* Case 2a. Pins: PC_FRMSE and PC_SRMSE leave out, per dataset, the
 * validation structures stored with weight 0. The fixed potential of case
 * 2 (no training DBFILE, every weight pinned, FORCE and STRESS false
 * written), whose prediction is predict_with() on pinned_potential(), is
 * validated with PC_ERMSE + PC_FRMSE + PC_SRMSE on two datasets of three
 * cells. One cell of each, never the first (the last of dataset 1, at the
 * dataset boundary; the middle one of dataset 2), is stored with fweight 0
 * and sweight 0 and garbage references: every force component 50 eV/A and
 * every stress entry 5 eV/A^3. The other cells keep their references and
 * weight 1; eweight is 1 throughout.
 *   - FRMSE[d] and SRMSE[d] (stats.tadah) equal the RMSEs over the cells of
 *     weight 1 of dataset d (frmse_over_weighted, srmse_over_weighted), and
 *     FRMSE[avg] and SRMSE[avg] their mean over the datasets, to 1e-12
 *     relative (the same arithmetic on the same weights, as in case 1). The
 *     garbage moves each all-cells RMSE (frmse_of, srmse_of) by more than 1
 *     (eV/A, eV/A^3; hence 5 eV/A^3 rather than a realistic stress), so a
 *     metric that kept those cells cannot pass;
 *   - the cells of weight 0 stay in the validation set (six structures)
 *     and in PC_ERMSE (eweight 1): ERMSE[d], ERMSE[avg] and the L2 loss of
 *     the three metrics agree to 1e-12;
 *   - the evaluation used the pinned weights, and validation predicted
 *     forces and stresses (the test hook).
 * Before the fix: the cells of weight 0 were measured, so FRMSE[d] and
 * SRMSE[d] were the all-cells values, off by more than 1. */
TEST_CASE("valmetric e2e: PC_FRMSE and PC_SRMSE leave out validation "
          "structures with weight 0", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    const double garbage_force  = 50.0;  // eV/A, every component
    const double garbage_stress = 5.0;   // eV/A^3, every entry

    ScratchDir sd;
    const std::vector<fs::path> val_dbs = {sd.dir / "val_1.db",
                                           sd.dir / "val_2.db"};
    const fs::path val_cfg = sd.dir / "config.val";
    const fs::path hpo_cfg = sd.dir / "config.hpo";

    /* Two datasets of three cells; the cell of weight 0 is the last of
     * dataset 1 and the middle one of dataset 2. */
    std::vector<std::vector<Structure>> sets(2);
    sets[0] = {bcc_cell(11, 3.10, 1.9), bcc_cell(12, 3.20, 0.6),
               bcc_cell(13, 3.00, 2.6)};
    sets[1] = {bcc_cell(14, 3.05, 0.4), bcc_cell(15, 3.15, 2.1),
               bcc_cell(16, 3.30, 1.2)};
    const std::array<std::size_t, 2> zero_at = {2u, 1u};
    for (std::size_t d = 0; d < sets.size(); ++d) {
        Structure& st = sets[d][zero_at[d]];
        st.fweight = 0.0;
        st.sweight = 0.0;
        for (std::size_t a = 0; a < st.natoms(); ++a)
            for (int k = 0; k < 3; ++k) st.atoms[a].force(k) = garbage_force;
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y) st.stress(x, y) = garbage_stress;
        write_db(val_dbs[d], sets[d]);
    }
    {
        std::ofstream os(val_cfg);
        for (const fs::path& db : val_dbs)
            os << "DBFILE " << db.string() << "\n";
    }
    std::ofstream(hpo_cfg) << target_head(kDerivativeFree)
                           << "PC_ERMSE 0 0.0001\nPC_FRMSE 0 1\n"
                              "PC_SRMSE 0 1\n"
                           << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";

    /* FIXWEIGHT (1) evaluated at its pinned value: the weights are
     * kPinned. All six structures stay in the validation set. */
    ParamVec p(1);
    p(0) = kPinned[0];
    const EvalOutcome e = evaluate_once(pinned_context(val_cfg, hpo_cfg),
                                        hpo_cfg, val_cfg, p, false, 0u, 6u);
    require_real_loss(e.loss);
    REQUIRE(e.live_weights.size() == kPinned.size());
    for (std::size_t k = 0; k < kPinned.size(); ++k)
        CHECK(same_bits(e.live_weights[k], kPinned[k]));
    check_validation_switches(e.validation_switches, /*force=*/true,
                              /*stress=*/true);

    /* The references, per dataset, on the data as the host reads it (the
     * weights included). */
    std::vector<double> ermse, frmse, srmse, frmse_all, srmse_all;
    for (std::size_t d = 0; d < val_dbs.size(); ++d) {
        const StructureDB vdb = read_db(val_dbs[d]);
        REQUIRE(vdb.size() == 3u);
        for (std::size_t i = 0; i < vdb.size(); ++i) {
            const double w = (i == zero_at[d]) ? 0.0 : 1.0;
            CAPTURE(d, i);
            REQUIRE(vdb(i).eweight == 1.0);
            REQUIRE(vdb(i).fweight == w);
            REQUIRE(vdb(i).sweight == w);
        }
        const StructureDB pred = predict_with(pinned_potential(), vdb,
                                              /*force=*/true, /*stress=*/true);
        ermse.push_back(ermse_mev(vdb, pred));
        frmse.push_back(frmse_over_weighted(vdb, pred));
        srmse.push_back(srmse_over_weighted(vdb, pred));
        frmse_all.push_back(frmse_of(vdb, pred));
        srmse_all.push_back(srmse_of(vdb, pred));
    }

    for (std::size_t d = 0; d < val_dbs.size(); ++d) {
        const std::string n = "[" + std::to_string(d + 1) + "]";
        const double f_reported = column(e.stats, "FRMSE" + n);
        const double s_reported = column(e.stats, "SRMSE" + n);
        CAPTURE(d, frmse[d], frmse_all[d], f_reported, srmse[d],
                srmse_all[d], s_reported);
        /* The fixture: the garbage would move both metrics by more than
         * 1. */
        REQUIRE(std::fabs(frmse_all[d] - frmse[d]) > 1.0);
        REQUIRE(std::fabs(srmse_all[d] - srmse[d]) > 1.0);
        CHECK(f_reported == Approx(frmse[d]).epsilon(1e-12));
        CHECK(s_reported == Approx(srmse[d]).epsilon(1e-12));
        /* ERMSE: also an absolute 1e-9 meV/atom, as in case 1. */
        CHECK(column(e.stats, "ERMSE" + n) ==
              Approx(ermse[d]).epsilon(1e-12).margin(1e-9));
    }

    const double e_avg = mean_over_datasets(ermse);
    const double f_avg = mean_over_datasets(frmse);
    const double s_avg = mean_over_datasets(srmse);
    CAPTURE(e_avg, f_avg, s_avg);
    CHECK(column(e.stats, "ERMSE[avg]") ==
          Approx(e_avg).epsilon(1e-12).margin(1e-9));
    CHECK(column(e.stats, "FRMSE[avg]") == Approx(f_avg).epsilon(1e-12));
    CHECK(column(e.stats, "SRMSE[avg]") == Approx(s_avg).epsilon(1e-12));
    /* L2: the mean of 1e-4 ERMSE^2, FRMSE^2 and SRMSE^2. */
    CHECK(e.loss == Approx((1e-4 * e_avg * e_avg + f_avg * f_avg +
                            s_avg * s_avg) / 3.0)
                        .epsilon(1e-12));
}

/* Case 2b. Pins (MK, 2026-10-06): a validation dataset in which every
 * structure has force weight 0 has no PC_FRMSE ("not measured"), and one
 * in which every structure has stress weight 0 no PC_SRMSE. On the fixed
 * potential of case 2, through hpo_run():
 *   - a run in which no dataset has the data is refused at startup: a
 *     std::runtime_error naming the metric, the DBFILE as the --validation
 *     file gives it and the weight ("force weight 0" / "stress weight 0"),
 *     before any evaluation (stats.tadah has no row);
 *   - with a global PC_FRMSE (and PC_SRMSE), the second of two datasets
 *     without forces is not measured: the run returns 0, FRMSE[2] is nan
 *     and FRMSE[avg] is FRMSE[1], bit for bit, while both datasets have a
 *     stress (SRMSE[2] finite);
 *   - with indexed PC_FRMSE, weight 0 on that dataset runs the same, and a
 *     weight above 0 on it is refused at startup, naming the metric, the
 *     dataset and the force weight;
 *   - a weight no metric reads refuses nothing: with PC_SRMSE alone and
 *     force weight 0 everywhere, or PC_FRMSE alone and stress weight 0
 *     everywhere, the run measures its metric (finite and positive).
 * Before the fix: every dataset without the data was refused at startup
 * (the three run SECTIONs failed); before that, no dataset was refused and
 * the zeros stored for the missing data were measured. */
TEST_CASE("valmetric e2e: a validation dataset whose structures all have "
          "weight 0 is not measured", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    std::string metrics;                  // the PC_ lines of the hpotarget
    bool zero_f = false, zero_s = false;  // the weights of the last dataset
    bool weighted_first = false;          // a dataset of weight 1 before it
    std::string refused_by, weight_named; // empty: the run starts
    std::string measured;                 // the column a started run fills
    bool second_unmeasured = false;       // FRMSE[2] nan, FRMSE[avg] = FRMSE[1]
    SECTION("PC_FRMSE and no dataset with forces") {
        metrics = "PC_FRMSE 0 1\n";
        zero_f = true;
        refused_by = "PC_FRMSE";
        weight_named = "force weight 0";
    }
    SECTION("PC_SRMSE and no dataset with a stress tensor") {
        metrics = "PC_SRMSE 0 1\n";
        zero_s = true;
        refused_by = "PC_SRMSE";
        weight_named = "stress weight 0";
    }
    SECTION("global PC_FRMSE and PC_SRMSE, the second of two datasets "
            "without forces") {
        metrics = "PC_FRMSE 0 1\nPC_SRMSE 0 1\n";
        zero_f = true;
        weighted_first = true;
        measured = "SRMSE[2]";
        second_unmeasured = true;
    }
    SECTION("indexed PC_FRMSE, weight 0 on the dataset without forces") {
        metrics = "PC_FRMSE (1) 0 1\nPC_FRMSE (2) 0 0\n";
        zero_f = true;
        weighted_first = true;
        measured = "FRMSE[1]";
        second_unmeasured = true;
    }
    SECTION("indexed PC_FRMSE, weight 0.5 on the dataset without forces") {
        metrics = "PC_FRMSE (1) 0 1\nPC_FRMSE (2) 0 0.5\n";
        zero_f = true;
        weighted_first = true;
        refused_by = "PC_FRMSE";
        weight_named = "force weight 0";
    }
    SECTION("PC_SRMSE alone and every structure with force weight 0") {
        metrics = "PC_SRMSE 0 1\n";
        zero_f = true;
        measured = "SRMSE[avg]";
    }
    SECTION("PC_FRMSE alone and every structure with stress weight 0") {
        metrics = "PC_FRMSE 0 1\n";
        zero_s = true;
        measured = "FRMSE[avg]";
    }
    CAPTURE(metrics, zero_f, zero_s, weighted_first);

    ScratchDir sd;
    const fs::path weighted_db = sd.dir / "val_weighted.db";
    const fs::path zero_db     = sd.dir / "val_zero.db";
    const fs::path val_cfg     = sd.dir / "config.val";
    const fs::path hpo_cfg     = sd.dir / "config.hpo";
    std::vector<Structure> zero = validation_cells();
    for (Structure& st : zero) {
        if (zero_f) st.fweight = 0.0;
        if (zero_s) st.sweight = 0.0;
    }
    write_db(zero_db, zero);
    {
        std::ofstream os(val_cfg);
        if (weighted_first) {
            const std::vector<Structure> weighted = {bcc_cell(13, 3.00, 2.6),
                                                     bcc_cell(14, 3.30, 1.2)};
            write_db(weighted_db, weighted);
            os << "DBFILE " << weighted_db.string() << "\n";
        }
        os << "DBFILE " << zero_db.string() << "\n";
    }
    std::ofstream(hpo_cfg) << target_head(kDerivativeFree) << metrics
                           << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";

    /* The fixture: the weights as the host reads them. */
    {
        const StructureDB db = read_db(zero_db);
        REQUIRE(db.size() == 2u);
        for (std::size_t i = 0; i < db.size(); ++i) {
            CAPTURE(i);
            REQUIRE(db(i).fweight == (zero_f ? 0.0 : 1.0));
            REQUIRE(db(i).sweight == (zero_s ? 0.0 : 1.0));
        }
    }

    const RunStart r = start_pinned_run(val_cfg, hpo_cfg);
    INFO("hpo_run() " << (r.threw ? "threw: " + r.what
                                  : "returned " + std::to_string(r.status)));
    if (!refused_by.empty()) {
        CHECK(r.threw);
        CHECK(r.runtime_error);
        CHECK(contains(r.what, refused_by));
        CHECK(contains(r.what, zero_db.string()));
        CHECK(contains(r.what, weight_named));
        /* At startup: no evaluation wrote a stats.tadah row. */
        const fs::path stats = sd.dir / "stats.tadah";
        if (fs::exists(stats)) CHECK(stats_rows(stats.string()).empty());
        return;
    }

    /* Not refused: the run started and measured its metric. */
    CHECK_FALSE(r.threw);
    REQUIRE(r.status == 0);
    const auto rows = stats_rows((sd.dir / "stats.tadah").string());
    REQUIRE_FALSE(rows.empty());
    const double v = column(rows.front(), measured);
    CAPTURE(measured, v);
    CHECK(std::isfinite(v));
    CHECK(v > 0.0);
    if (second_unmeasured) {
        const double f1 = column(rows.front(), "FRMSE[1]");
        const double f2 = column(rows.front(), "FRMSE[2]");
        const double favg = column(rows.front(), "FRMSE[avg]");
        CAPTURE(f1, f2, favg);
        CHECK(std::isnan(f2));
        CHECK(std::isfinite(f1));
        CHECK(same_bits(favg, f1));
    }
}

/* Case 3. MODEL M_KRR Kern_Linear 1 (FORCE and STRESS both written in the
 * configuration), fitted with STRESS true and validated with PC_ERMSE +
 * PC_SRMSE, and fitted with FORCE true and validated with PC_ERMSE +
 * PC_FRMSE:
 *   - ERMSE, SRMSE / FRMSE and the loss equal those of predict_with() on
 *     the evaluation's best_pot.tadah to 1e-12 relative (as in case 1). The
 *     SRMSE of the fixture differs from RMS(reference) by more than 0.1%, so
 *     the zero prediction cannot pass;
 *   - the validation prediction used FORCE = use_f, STRESS = use_s (the
 *     test hook);
 *   - best_pot.tadah, and the model's own get_param_file() (what `tadah
 *     train` writes to pot.tadah), store the training FORCE and STRESS, as
 *     for M_BLR: retraining from best_pot.tadah fits what HPO fitted.
 * The STRESS SECTION runs first, so that before the fix the group failed
 * there without a crash.
 * Before the fix: SRMSE == RMS(reference) (the prediction stayed zero: the
 * potential context had no STRESS); with FORCE true + PC_FRMSE, SIGSEGV in
 * Analytics::calc_f_rmse (the potential context had no FORCE). Before
 * M_KRR::get_param_file() kept FORCE and STRESS, best_pot.tadah and
 * get_param_file() stored neither key (stored_text() "<not stored>"), also
 * once validation followed the metrics; before the host had
 * last_validation_switches_for_test(), the hook check failed. */
TEST_CASE("valmetric e2e: MODEL M_KRR Kern_Linear validates the forces "
          "and stresses it fitted", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    FitSwitches sw;
    bool use_f = false, use_s = false;
    SECTION("STRESS true and PC_SRMSE") {
        sw = FitSwitches{false, true};
        use_s = true;
    }
    SECTION("FORCE true and PC_FRMSE") {
        sw = FitSwitches{true, false};
        use_f = true;
    }
    CAPTURE(use_f, use_s);

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_db(train_db, training_cells());
    write_db(val_db, validation_cells());
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
    {
        std::ofstream os(hpo_cfg);
        os << target_head(kDerivativeFree) << "PC_ERMSE 0 0.0001\n";
        if (use_f) os << "PC_FRMSE 0 1\n";
        if (use_s) os << "PC_SRMSE 0 1\n";
        os << kTrainedOptim;
    }

    const ParamVec p = trained_point();
    const EvalOutcome e =
        evaluate_once(trained_context(kKRR, sw, train_db, val_cfg, hpo_cfg),
                      hpo_cfg, val_cfg, p, true, 5u, 2u);
    require_real_loss(e.loss);
    {
        Context pot = e.pot;
        REQUIRE(pot.get<std::string>("MODEL", 0) == "M_KRR");
        REQUIRE(pot.get<std::string>("MODEL", 1) == "Kern_Linear");
    }

    const StructureDB vdb = read_db(val_db);
    const StructureDB pred = predict_with(e.pot, vdb, use_f, use_s);
    const double ermse = ermse_mev(vdb, pred);
    CAPTURE(ermse);
    /* ERMSE: also an absolute 1e-9 meV/atom (1e-12 eV/atom, ~1e3 ulp of
     * the 5 eV/atom energies) for a residual small enough that a
     * last-bit difference of the total energy shows at 1e-12. */
    CHECK(column(e.stats, "ERMSE[avg]") ==
          Approx(ermse).epsilon(1e-12).margin(1e-9));
    double expected_loss = 1e-4 * ermse * ermse;
    if (use_f) {
        const double frmse = frmse_of(vdb, pred);
        CAPTURE(frmse);
        REQUIRE(frmse > 0.0);
        CHECK(column(e.stats, "FRMSE[avg]") == Approx(frmse).epsilon(1e-12));
        expected_loss += frmse * frmse;
    }
    if (use_s) {
        const double srmse = srmse_of(vdb, pred);
        const double rms_ref = rms_stress(vdb);
        CAPTURE(srmse, rms_ref);
        REQUIRE(std::fabs(srmse - rms_ref) > 1e-3 * rms_ref);
        CHECK(column(e.stats, "SRMSE[avg]") == Approx(srmse).epsilon(1e-12));
        expected_loss += srmse * srmse;
    }
    CHECK(e.loss == Approx(expected_loss / 2.0).epsilon(1e-12));

    /* Validation predicted what the metrics measure. */
    check_validation_switches(e.validation_switches, use_f, use_s);

    /* The trainer's context keeps the training switches. */
    CHECK(effective(e.trainer_ctx, "FORCE") == use_f);
    CHECK(effective(e.trainer_ctx, "STRESS") == use_s);

    /* best_pot.tadah and the model's potential context state the training
     * switches, both written in the configuration (sw.force == use_f,
     * sw.stress == use_s): M_KRR keeps them as M_BLR does. */
    REQUIRE(sw.force.has_value());
    REQUIRE(sw.stress.has_value());
    const std::string fit_f = use_f ? "true" : "false";
    const std::string fit_s = use_s ? "true" : "false";
    CHECK(stored_text(e.pot, "FORCE") == fit_f);
    CHECK(stored_text(e.pot, "STRESS") == fit_s);
    CHECK(stored_text(e.model_pot, "FORCE") == fit_f);
    CHECK(stored_text(e.model_pot, "STRESS") == fit_s);
    /* ALPHA, BETA and DBFILE stay out of an M_KRR potential. */
    CHECK_FALSE(e.pot.has_stored("DBFILE"));
    CHECK_FALSE(e.model_pot.has_stored("DBFILE"));
    CHECK_FALSE(e.model_pot.has_stored("ALPHA"));
    CHECK_FALSE(e.model_pot.has_stored("BETA"));
}

/* Case 4. GRAD_STRATEGY ANALYTICAL and HYBRID with FORCE false + PC_ERMSE
 * + PC_FRMSE (OALGO 4, LAMBDA 1e-3, BF_Linear, D2_BP; OPTIM a Gaussian
 * centre and a Gaussian width):
 *   - startup validation accepts the run;
 *   - the analytical gradient routes no parameter to finite differences;
 *   - each component equals the Richardson-extrapolated central difference
 *     of the host loss, (4 D(h) - D(2h)) / 3 with h = 1e-3, to 1e-5
 *     relative. Its truncation error is O(h^4) ~ 1e-12 relative for
 *     Gaussians of width ~1 A, and the loss of a 3-weight ridge fit to five
 *     distinct cells carries ~1e-14 relative rounding, ~1e-11 after
 *     division by h: both far below 1e-5. A component much smaller than
 *     the largest is held to 1e-5 of the largest (margin).
 * PC_ERMSE has weight 1e-6 so the force term dominates the loss (checked
 * on eval 1): the gradient cannot pass on its energy part alone. The
 * journal no longer says "PC_FRMSE needs FORCE true". The ANALYTICAL
 * SECTION runs first, so that before the fix the group failed there
 * without a crash.
 * Before the fix: ANALYTICAL was refused by validate_hpo_setup ("PC_FRMSE
 * needs FORCE true"); HYBRID passed startup and the first evaluation got
 * SIGSEGV in Analytics::calc_f_rmse. */
TEST_CASE("valmetric e2e: the analytical gradient covers PC_FRMSE with "
          "FORCE false", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    std::string strategy;
    SECTION("GRAD_STRATEGY ANALYTICAL") {
        strategy = "ANALYTICAL";
    }
    SECTION("GRAD_STRATEGY HYBRID") {
        strategy = "HYBRID";
    }
    CAPTURE(strategy);

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_db(train_db, training_cells());
    write_db(val_db, validation_cells());
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
    std::ofstream(hpo_cfg) << target_head(gradient_block(strategy))
                           << "PC_ERMSE 0 0.000001\nPC_FRMSE 0 1\n"
                           << kTrainedOptim;

    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    Context ctx = trained_context(kBLR, FitSwitches{false, false}, train_db,
                                  val_cfg, hpo_cfg);
    JournalFile j;
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.use_frmse);
    REQUIRE(trg.spec.parameter_labels == kTrainedLabels);
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));

    const ParamVec p = trained_point();
    const double h = 1e-3;
    ParamVec g;
    std::vector<int> needs_fd;
    std::vector<double> richardson(2, 0.0);
    {
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.train_size() == 5u);
        REQUIRE(host.val_size() == 2u);
        double f0 = 0.0;
        REQUIRE_NOTHROW(f0 = host(p));
        require_real_loss(f0);
        bool ok = false;
        REQUIRE_NOTHROW(ok = host.analytical_gradient(p, g, needs_fd));
        REQUIRE(ok);
        REQUIRE(g.size() == 2);

        auto central = [&](long i, double step) {
            ParamVec up = p, down = p;
            up(i) += step;
            down(i) -= step;
            double fu = 0.0, fd = 0.0;
            REQUIRE_NOTHROW(fu = host(up));
            REQUIRE_NOTHROW(fd = host(down));
            require_real_loss(fu);
            require_real_loss(fd);
            return (fu - fd) / (2.0 * step);
        };
        for (long i = 0; i < 2; ++i) {
            const double d2h = central(i, 2.0 * h);
            const double dh  = central(i, h);
            richardson[static_cast<std::size_t>(i)] = (4.0 * dh - d2h) / 3.0;
        }
        host.flush_remaining();
    }
    const std::string journal = j.finish();
    Host::reset_globals_for_test();

    /* Eval 1 (at p): the force term dominates the loss. */
    {
        const auto rows = stats_rows(trg.statsFile);
        REQUIRE_FALSE(rows.empty());
        const double ermse = column(rows.front(), "ERMSE[avg]");
        const double frmse = column(rows.front(), "FRMSE[avg]");
        CAPTURE(ermse, frmse);
        REQUIRE(frmse * frmse > 1e-6 * ermse * ermse);
    }

    CHECK(needs_fd.empty());
    double gmax = 0.0;
    for (double d : richardson) gmax = std::max(gmax, std::fabs(d));
    CAPTURE(gmax);
    REQUIRE(gmax > 0.0);
    for (long i = 0; i < 2; ++i) {
        const double fd = richardson[static_cast<std::size_t>(i)];
        const std::string& label = kTrainedLabels[static_cast<std::size_t>(i)];
        const double analytical = g(i);
        CAPTURE(label, analytical, fd);
        CHECK(std::isfinite(g(i)));
        CHECK(g(i) == Approx(fd).epsilon(1e-5).margin(1e-5 * gmax));
    }

    INFO(journal);
    CHECK_FALSE(contains(journal, "PC_FRMSE needs FORCE true"));
}

/* Case 4a. Pins: the analytical gradient of a global PC_FRMSE whose second
 * validation dataset is not measured (every structure with force weight 0,
 * forces far from any prediction) equals the Richardson-extrapolated
 * central difference of the host loss, as in case 4 (ANALYTICAL, FORCE
 * false, PC_ERMSE weight 1e-6): the average and its derivative run over the
 * measured dataset only (the slope of dataset 1 is L'(FRMSE[avg]) / 1, not
 * / 2), and the unmeasured dataset has no force cells. FRMSE[2] is nan and
 * FRMSE[avg] is FRMSE[1].
 * Before the fix: the run was refused at startup (the dataset has no
 * structure with forces). */
TEST_CASE("valmetric e2e: the analytical gradient of PC_FRMSE averages the "
          "measured datasets", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path nof_db   = sd.dir / "val_noforces.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_db(train_db, training_cells());
    write_db(val_db, validation_cells());
    {
        std::vector<Structure> nof = {bcc_cell(17, 3.12, 0.9),
                                      bcc_cell(18, 3.24, 2.2)};
        for (Structure& st : nof) {
            st.fweight = 0.0;
            for (std::size_t a = 0; a < st.natoms(); ++a)
                for (int k = 0; k < 3; ++k) st.atoms[a].force(k) = 50.0;
        }
        write_db(nof_db, nof);
    }
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n"
                           << "DBFILE " << nof_db.string() << "\n";
    std::ofstream(hpo_cfg) << target_head(gradient_block("ANALYTICAL"))
                           << "PC_ERMSE 0 0.000001\nPC_FRMSE 0 1\n"
                           << kTrainedOptim;

    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    Context ctx = trained_context(kBLR, FitSwitches{false, false}, train_db,
                                  val_cfg, hpo_cfg);
    JournalFile j;
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.use_frmse);
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));

    const ParamVec p = trained_point();
    const double h = 1e-3;
    ParamVec g;
    std::vector<int> needs_fd;
    std::vector<double> richardson(2, 0.0);
    {
        std::unique_ptr<Host> host;
        REQUIRE_NOTHROW(host = std::make_unique<Host>(ctx, trg, val_cfg.string()));
        REQUIRE(host->val_size() == 4u);
        double f0 = 0.0;
        REQUIRE_NOTHROW(f0 = (*host)(p));
        require_real_loss(f0);
        bool ok = false;
        REQUIRE_NOTHROW(ok = host->analytical_gradient(p, g, needs_fd));
        REQUIRE(ok);
        REQUIRE(g.size() == 2);
        auto central = [&](long i, double step) {
            ParamVec up = p, down = p;
            up(i) += step;
            down(i) -= step;
            double fu = 0.0, fd = 0.0;
            REQUIRE_NOTHROW(fu = (*host)(up));
            REQUIRE_NOTHROW(fd = (*host)(down));
            require_real_loss(fu);
            require_real_loss(fd);
            return (fu - fd) / (2.0 * step);
        };
        for (long i = 0; i < 2; ++i) {
            const double d2h = central(i, 2.0 * h);
            const double dh  = central(i, h);
            richardson[static_cast<std::size_t>(i)] = (4.0 * dh - d2h) / 3.0;
        }
        host->flush_remaining();
    }
    (void)j.finish();
    Host::reset_globals_for_test();

    {
        const auto rows = stats_rows(trg.statsFile);
        REQUIRE_FALSE(rows.empty());
        const double f1 = column(rows.front(), "FRMSE[1]");
        const double f2 = column(rows.front(), "FRMSE[2]");
        const double favg = column(rows.front(), "FRMSE[avg]");
        const double ermse = column(rows.front(), "ERMSE[avg]");
        CAPTURE(f1, f2, favg, ermse);
        CHECK(std::isnan(f2));
        CHECK(same_bits(favg, f1));
        REQUIRE(favg * favg > 1e-6 * ermse * ermse);
    }
    CHECK(needs_fd.empty());
    double gmax = 0.0;
    for (double d : richardson) gmax = std::max(gmax, std::fabs(d));
    CAPTURE(gmax);
    REQUIRE(gmax > 0.0);
    for (long i = 0; i < 2; ++i) {
        const double fd = richardson[static_cast<std::size_t>(i)];
        CAPTURE(kTrainedLabels[static_cast<std::size_t>(i)], g(i), fd);
        CHECK(std::isfinite(g(i)));
        CHECK(g(i) == Approx(fd).epsilon(1e-5).margin(1e-5 * gmax));
    }
}

/* Case 5. hpo_run(ctx) without key provenance on a one-evaluation run:
 * POTENTIAL SETUP states what training fits and what validation predicts.
 *   - "Fit forces (FORCE)" / "Fit stresses (STRESS)": the value starts with
 *     yes/no; its source is "(source not recorded)" for a key the context
 *     holds and "(default)" for one it does not;
 *   - "Validation predicts": "energies" plus "forces (PC_FRMSE)" and/or
 *     "stresses (PC_SRMSE)" exactly when those metrics are set, or "no
 *     validation set".
 * Key phrases only (substrings), never whole lines. The SECTION with no
 * FORCE key and PC_FRMSE runs last, because before the fix it crashed.
 * Before the fix: none of the three lines existed; the last SECTION got
 * SIGSEGV in Analytics::calc_f_rmse. */
TEST_CASE("valmetric e2e: the journal states what training fits and what "
          "validation predicts", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    FitSwitches sw;
    std::string metrics;
    bool with_validation = true;
    bool predicts_forces = false, predicts_stresses = false;
    std::string force_yes_no, force_source, stress_yes_no, stress_source;
    SECTION("FORCE true written and PC_FRMSE") {
        sw.force = true;
        metrics = "PC_ERMSE 0 0.0001\nPC_FRMSE 0 1\n";
        force_yes_no = "yes";
        force_source = "(source not recorded)";
        stress_yes_no = "no";
        stress_source = "(default)";
        predicts_forces = true;
    }
    SECTION("FORCE and STRESS false written and PC_SRMSE") {
        sw = FitSwitches{false, false};
        metrics = "PC_ERMSE 0 0.0001\nPC_SRMSE 0 1\n";
        force_yes_no = "no";
        force_source = "(source not recorded)";
        stress_yes_no = "no";
        stress_source = "(source not recorded)";
        predicts_stresses = true;
    }
    SECTION("no validation set") {
        with_validation = false;
    }
    SECTION("no FORCE key and PC_FRMSE") {
        metrics = "PC_ERMSE 0 0.0001\nPC_FRMSE 0 1\n";
        force_yes_no = "no";
        force_source = "(default)";
        stress_yes_no = "no";
        stress_source = "(default)";
        predicts_forces = true;
    }

    if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
    HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    std::string journal;
    {
        ForceRebuildGuard rebuild;
        ScratchDir sd;
        const fs::path hpo_cfg = sd.dir / "config.hpo";
        Context ctx;
        if (with_validation) {
            const fs::path train_db = sd.dir / "train.db";
            const fs::path val_db   = sd.dir / "val.db";
            const fs::path val_cfg  = sd.dir / "config.val";
            write_db(train_db, training_cells());
            write_db(val_db, validation_cells());
            std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
            std::ofstream(hpo_cfg) << target_head(kDerivativeFree) << metrics
                                   << kTrainedOptim;
            ctx = trained_context(kBLR, sw, train_db, val_cfg, hpo_cfg);
        } else {
            /* No dataset at all: the fixed potential and a cohesive-energy
             * observable (the PC_OBSERVABLE run of test_nn_exact_e2e.cpp;
             * GRAD_STRATEGY HYBRID: an unset one resolves to ANALYTICAL
             * with PC_OBSERVABLE, which needs training data). */
            std::ofstream(hpo_cfg)
                << target_head(gradient_block("HYBRID"))
                << "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
                   "--weight 1 --supercell 2 2 2 --maxiter 50 --maxeval 200 "
                   "--hotvar a0 2.7 3.9 3.3\n"
                << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
            ctx = pinned_context(fs::path(), hpo_cfg);
        }
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(ctx));
        /* The observable may fail to relax with this potential; the run's
         * status then says so, and the journal is what this case reads. */
        if (with_validation) CHECK(status == 0);
        journal = read_text(sd.dir / "hpo_run.log");
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
    }
    Host::reset_globals_for_test();

    const std::string setup = potential_setup(journal);
    INFO("POTENTIAL SETUP:\n" << setup);
    REQUIRE_FALSE(setup.empty());
    const std::string predicts = entry_value(setup, "Validation predicts");
    CAPTURE(predicts);
    REQUIRE_FALSE(predicts.empty());
    if (!with_validation) {
        CHECK(contains(predicts, "no validation set"));
        return;
    }
    CHECK(contains(predicts, "energies"));
    CHECK(contains(predicts, "forces (PC_FRMSE)") == predicts_forces);
    CHECK(contains(predicts, "stresses (PC_SRMSE)") == predicts_stresses);
    CHECK_FALSE(contains(predicts, "no validation set"));

    const std::string fit_f = entry_value(setup, "Fit forces (FORCE)");
    const std::string fit_s = entry_value(setup, "Fit stresses (STRESS)");
    CAPTURE(fit_f, fit_s);
    REQUIRE_FALSE(fit_f.empty());
    REQUIRE_FALSE(fit_s.empty());
    CHECK(starts_with(fit_f, force_yes_no));
    CHECK(contains(fit_f, force_source));
    CHECK(starts_with(fit_s, stress_yes_no));
    CHECK(contains(fit_s, stress_source));
}

/* Case 6. A fit of forces and stresses (MODEL M_BLR BF_Linear, FORCE true
 * and STRESS true written) validated on energies alone (PC_ERMSE only):
 * validation predicts neither forces nor stresses -- PC_FRMSE and PC_SRMSE
 * decide, and only they -- while training is unchanged. Two evaluations of
 * the same fit at the same candidate, each by a fresh host:
 *   - a control with PC_ERMSE + PC_FRMSE + PC_SRMSE: the test hook reports
 *     FORCE true / STRESS true, so the value read after the energy-only
 *     evaluation is that evaluation's own;
 *   - PC_ERMSE only: the hook reports FORCE false / STRESS false. The loss
 *     is a real one, and ERMSE and the loss equal those of predict_with()
 *     on its best_pot.tadah (energies only) to 1e-12 relative;
 *   - best_pot.tadah and the trainer's context store FORCE true and STRESS
 *     true, and both evaluations fitted the same weights (to 1e-12): what
 *     validation measures does not change what training fits.
 * Nothing else can tell: no metric reads the extra forces or stresses, and
 * the journal row "Validation predicts" is built from the targets.
 * Before the fix: the host had no last_validation_switches_for_test(), so
 * both hook checks failed (the rest held). Before validation followed the
 * metrics, this prediction used the training FORCE/STRESS (true, true):
 * the energy-only hook check fails for that behaviour. */
TEST_CASE("valmetric e2e: a force and stress fit validated on energies alone "
          "predicts neither forces nor stresses", "[valmetric_e2e][.]")
{
    PeriodicTable::initialize();
    const FitSwitches sw{true, true};
    const ParamVec p = trained_point();

    struct Run {
        EvalOutcome e;
        StructureDB vdb;
    };
    auto run = [&](const std::string& metrics) {
        ScratchDir sd;
        const fs::path train_db = sd.dir / "train.db";
        const fs::path val_db   = sd.dir / "val.db";
        const fs::path val_cfg  = sd.dir / "config.val";
        const fs::path hpo_cfg  = sd.dir / "config.hpo";
        write_db(train_db, training_cells());
        write_db(val_db, validation_cells());
        std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
        std::ofstream(hpo_cfg) << target_head(kDerivativeFree) << metrics
                               << kTrainedOptim;
        Run r{evaluate_once(
                  trained_context(kBLR, sw, train_db, val_cfg, hpo_cfg),
                  hpo_cfg, val_cfg, p, true, 5u, 2u),
              read_db(val_db)};
        return r;
    };

    /* The control: every metric, so validation predicts everything. */
    const Run all = run("PC_ERMSE 0 0.0001\nPC_FRMSE 0 1\nPC_SRMSE 0 1\n");
    {
        INFO("control: PC_ERMSE + PC_FRMSE + PC_SRMSE");
        require_real_loss(all.e.loss);
        check_validation_switches(all.e.validation_switches, /*force=*/true,
                                  /*stress=*/true);
    }

    /* Energies alone. */
    const Run energy = run("PC_ERMSE 0 0.0001\n");
    const EvalOutcome& e = energy.e;
    INFO("PC_ERMSE only");
    require_real_loss(e.loss);
    check_validation_switches(e.validation_switches, /*force=*/false,
                              /*stress=*/false);

    /* The loss is that of the evaluation's own potential. */
    const StructureDB pred = predict_with(e.pot, energy.vdb, /*force=*/false,
                                          /*stress=*/false);
    const double ermse = ermse_mev(energy.vdb, pred);
    CAPTURE(ermse);
    REQUIRE(ermse > 0.0);
    /* ERMSE: also an absolute 1e-9 meV/atom, as in case 1. */
    CHECK(column(e.stats, "ERMSE[avg]") ==
          Approx(ermse).epsilon(1e-12).margin(1e-9));
    /* L2 with one term: 1e-4 ERMSE^2. */
    CHECK(e.loss == Approx(1e-4 * ermse * ermse).epsilon(1e-12));

    /* Training is unchanged: the switches it fitted with are kept, in
     * best_pot.tadah and in the trainer's context ... */
    CHECK(stored_text(e.pot, "FORCE") == "true");
    CHECK(stored_text(e.pot, "STRESS") == "true");
    CHECK(stored_text(e.trainer_ctx, "FORCE") == "true");
    CHECK(stored_text(e.trainer_ctx, "STRESS") == "true");

    /* ... and the fit is the control's (same data, candidate and
     * switches). */
    REQUIRE(all.e.live_weights.size() == 3u);
    REQUIRE(e.live_weights.size() == all.e.live_weights.size());
    for (std::size_t k = 0; k < e.live_weights.size(); ++k) {
        CAPTURE(k, e.live_weights[k], all.e.live_weights[k]);
        CHECK(e.live_weights[k] ==
              Approx(all.e.live_weights[k]).epsilon(1e-12));
    }
}

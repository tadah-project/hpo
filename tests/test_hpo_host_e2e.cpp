// HPO_Host end-to-end integration test (Phase 3 follow-up).
//
// Validates production wiring of HPO_Host<ParamVec, LammpsRunner>::
// analytical_gradient — specifically the SimsGradProvider lambda
// construction at hpo_host_impl.hpp:~2186. Prior coverage was indirect:
// `[live_trainer][sims_grad_provider]` (test_hpo_analytical.cpp) tested
// `compute_analytical_gradient_from_trainer`'s ability to fold in a
// synthetic provider, and `[phase3_orchestrator]` reproduced the
// lambda's body inline on a real LAMMPS instance. Neither exercised
// HPO_Host's own construction of the lambda.
//
// Two cases:
//  1. No-LAMMPS path: trg_.LS empty ⇒ sims_grad_provider stays nullptr,
//     analytical_gradient goes through the descriptor-side path only.
//     Verifies HPO_Host constructs and operator() populates the trainer
//     cache that analytical_gradient consumes.
//  2. PC_LAMMPS path: trg_.LS has one cohesive-E script ⇒ provider is
//     wired; calling analytical_gradient runs the full LAMMPS lifecycle
//     and folds dO/dW into the analytical gradient.
//
// Both cases are hidden (`[.][!benchmark]`) because they require a real
// pot fixture and LAMMPS in-process — the same skip-on-no-pot pattern
// used by `[lammps_hessian]` / `[phase3_pilot]`.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/warmstart_cache.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() /
              ("tadah_hpo_e2e_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        /* HPO_Host's constructor writes loss.tadah / outvar.tadah /
         * params.tadah / stats.tadah + best_* in CWD. chdir into the
         * scratch dir so those files don't pollute the test working
         * directory. Restore CWD in dtor. */
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
};

/* Write a minimal 4-atom BCC Ta structure DB. Cell ≈ 1×1×2 supercell at
 * a=3.301 Å (matches test_lammps_hessian.cpp's fixture). offset_x lets
 * the val DBFILE differ from train so the descriptor distribution isn't
 * algebraically aligned with the trained W (same trick as
 * make_h3_stdb in test_hpo_analytical.cpp).
 *
 * Structure::read format:
 *   <label>
 *   <eweight> <fweight> <sweight>
 *   <energy> <T>
 *   <3x3 cell>
 *   <3x3 stress>
 *   <symbol> <x> <y> <z> <fx> <fy> <fz>
 *   ...
 *   <blank line>
 */
void write_minimal_ta_dbfile(const fs::path& path, double offset_x, double energy = -0.001) {
    std::ofstream os(path);
    os << "test_bcc_ta\n"
       << "1.0 1.0 1.0\n"
       << energy << " 0.0\n"
       << "3.301 0.0 0.0\n"
       << "0.0 3.301 0.0\n"
       << "0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n"
       << "\n";
}

/* Build a Trainer-able master context. Mirrors
 * make_minimal_trainer_context from test_hpo_analytical.cpp but adapted
 * for Ta (single species, single DBFILE). The keys here must be enough
 * for: Trainer construction (descriptor stack), HPO_Host construction
 * (DBFILE + RCUTMAX + load transforms), and HPOTargets OPTIM parser
 * (CGRID2B must already be declared so OPTIM CGRID2B (1) resolves).
 */
tadah::core::Context make_ta_master_context(
    const fs::path& dbfile,
    const fs::path& hpo_cfg,
    const fs::path& val_cfg)
{
    tadah::core::Context c;
    c.add("DBFILE",   dbfile.string());

    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);

    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");

    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(2));
    c.add("TYPE2B",   static_cast<int>(2));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");

    c.add("CGRID2B",  "0.5 1.4");
    c.add("SGRID2B",  "3.0 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   5.0);

    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   5.0);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");

    /* RCUTMAX is an internal key auto-derived from RCUT2B/RCUTMB by
     * Context::updateInternalKeys() (triggered on every add()). Don't
     * set it manually — would throw "Setting of internal keys is not
     * allowed". */

    c.add("FORCE",    "false");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);

    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);

    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);

    c.add("HPOTARGET",  hpo_cfg.string());
    c.add("VALIDATION", val_cfg.string());
    return c;
}

}  // namespace

TEST_CASE("HPO_Host::analytical_gradient end-to-end on a Trainer with no "
          "PC_LAMMPS line (Phase 3 follow-up)",
          "[hpo_host_e2e][phase3][.][!benchmark]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";

    write_minimal_ta_dbfile(train_db, /*offset_x=*/0.0);
    write_minimal_ta_dbfile(val_db,   /*offset_x=*/0.07);

    {
        std::ofstream os(val_cfg);
        os << "DBFILE " << val_db.string() << "\n";
    }
    {
        std::ofstream os(hpo_cfg);
        /* No PC_LAMMPS line ⇒ trg.LS stays empty ⇒
         * analytical_gradient's lambda construction is skipped. */
        os <<
            "OPTIMIZER\n"
            "LIB DLIB\n"
            "ALGO BFGS\n"
            "MAXEVAL 1\n"
            "ENDOPTIMIZER\n"
            "LOSS L2\n"
            "PC_EMAE 0 1\n"
            "OPTIM CGRID2B (1) 0.0 4.0\n";
    }

    tadah::core::Context ctx = make_ta_master_context(train_db, hpo_cfg, val_cfg);
    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.LS.empty());

    /* If anything in fixture construction throws, we want a clean test
     * failure with the message — not a SEGFAULT in some downstream stage. */
    Host host(ctx, trg, val_cfg.string());
    REQUIRE(host.train_size() == 1u);
    REQUIRE(host.val_size()   == 1u);

    /* Populate the trainer cache so analytical_gradient has something
     * to chain through. */
    ParamVec p(1);
    p(0) = 0.5;   /* CGRID2B[1] initial */
    double f0 = 0.0;
    REQUIRE_NOTHROW(f0 = host(p));
    CHECK(std::isfinite(f0));

    /* The analytical-gradient hook for GRAD_STRATEGY=ANALYTICAL/HYBRID. */
    ParamVec g(1);
    std::vector<int> needs_fd;
    bool ok = host.analytical_gradient(p, g, needs_fd);
    REQUIRE(ok);
    CHECK(std::isfinite(g(0)));
    /* needs_fd may be empty (analytical path engaged) or contain index 0
     * (helper routed to FD — e.g. OALGO/MODEL gate). Both are valid
     * behaviour for the wrapper; we just assert the call doesn't blow up. */

    Host::reset_globals_for_test();
}

/* PC_OBSERVABLE end-to-end (D4c integration).
 *
 * Mirrors the PC_LAMMPS case below but with a typed observable
 * (`PC_OBSERVABLE cohesive_energy ...`) instead. Validates that:
 *   - HPOTargets parses PC_OBSERVABLE and populates trg.observables
 *   - HPO_Host::operator() runs obs_loss(r), populates
 *     r.v_obs_value / r.v_obs_loss, and folds them into r.loss
 *   - HPO_Host::analytical_gradient builds the ObservableGradProvider
 *     lambda and either folds the analytical chain into g, or routes
 *     to FD (all-or-nothing fall-back). Either is valid wiring.
 *
 * Declared BEFORE the PC_LAMMPS case so the LAMMPS in-process global
 * state (left after the PC_LAMMPS test's LMP destructor runs) does
 * not contaminate the PC_OBSERVABLE relaxer. Tests in this file are
 * hidden `[.][!benchmark]` and won't run together by default.
 *
 * The minimal 4-atom Ta DB and BF_Linear/D2_BP[2,2] pot used in the
 * fixture produces a low-expressivity potential; the relaxer inside
 * the observable may not converge to a physical Ecoh for arbitrary θ.
 * The test accepts either a finite loss or FAIL_SCORE.
 */
TEST_CASE("HPO_Host::analytical_gradient end-to-end with one PC_OBSERVABLE "
          "cohesive_energy entry (D4c)",
          "[hpo_host_e2e][pc_observable][.][!benchmark]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";

    write_minimal_ta_dbfile(train_db, /*offset_x=*/0.0);
    write_minimal_ta_dbfile(val_db,   /*offset_x=*/0.07);

    {
        std::ofstream os(val_cfg);
        os << "DBFILE " << val_db.string() << "\n";
    }
    {
        std::ofstream os(hpo_cfg);
        os <<
            "OPTIMIZER\n"
            "LIB DLIB\n"
            "ALGO BFGS\n"
            "MAXEVAL 1\n"
            "ENDOPTIMIZER\n"
            "LOSS L2\n"
            "FAILSCORE 1e8\n"
            "PC_EMAE 0 1\n"
            "PC_OBSERVABLE cohesive_energy --species Ta "
            "--target 8.1 --weight 1 "
            "--supercell 2 2 2 "
            "--hotvar a0 2.7 3.9 3.3\n"
            "OPTIM CGRID2B (1) 0.0 4.0\n";
    }

    tadah::core::Context ctx = make_ta_master_context(train_db, hpo_cfg, val_cfg);
    /* Cohesive_energy's NativeRelaxer drives forces via fpredict, which
     * needs FORCE=true in ctx so DescriptorsCalc::calc populates fd_ij.
     * The minimal DBFILE has zero forces so FORCE=true is safe for the
     * training fit. */
    ctx.remove("FORCE");
    ctx.add("FORCE", "true");

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.LS.empty());
    REQUIRE(trg.observables.size() == 1u);
    REQUIRE(trg.observables[0]->kind() == "cohesive_energy");

    Host host(ctx, trg, val_cfg.string());
    REQUIRE(host.train_size() == 1u);

    ParamVec p(1);
    p(0) = 0.5;
    double f0 = std::numeric_limits<double>::quiet_NaN();
    /* operator() runs fit_loss + obs_loss. obs_loss evaluates the
     * cohesive_energy observable; a diverging relax clamps its loss
     * component to FAIL_SCORE per the contract. f0 must be finite. */
    REQUIRE_NOTHROW(f0 = host(p));
    CHECK(std::isfinite(f0));

    /* The real production wiring: analytical_gradient should construct
     * an ObservableGradProvider (because trg_.observables is non-empty)
     * and either fold the chain into g (analytical path) or route to FD
     * (all-or-nothing fall-back). Either is valid. The test asserts the
     * call returns cleanly. */
    ParamVec g(1);
    std::vector<int> needs_fd;
    bool ok = false;
    REQUIRE_NOTHROW(ok = host.analytical_gradient(p, g, needs_fd));
    REQUIRE(ok);
    CHECK(std::isfinite(g(0)));

    Host::reset_globals_for_test();
}

/* Companion case: same fixture but with a PC_LAMMPS line pointing at a
 * cohesive-E pilot script (mirrors the simple_bcc_pilot_script from
 * test_hpo_phase3_pilot.cpp). This case verifies that:
 *   - HPO_Host constructs with a non-empty trg_.LS
 *   - operator()'s LAMMPS-eval path executes (PC_LAMMPS scripts run)
 *   - analytical_gradient builds the SimsGradProvider lambda
 *     (verified indirectly: the call returns true, doesn't crash, and
 *     g remains finite)
 *
 * The fixture's trainer's pot may not produce a numerically reasonable
 * relaxation for arbitrary scripts — the trainer is fit on a single 4-
 * atom Ta cell with D2_BP[2,2]+BF_Linear, which is a low-expressivity
 * potential. If the per-script populate_sims_grad_input_from_lammps
 * returns false (FAIL_SCORE-clamped observable from a bad relax), the
 * all-or-nothing contract routes all theta to FD and analytical_gradient
 * still returns true. The test is structured to accept either outcome:
 * analytical path engaged, or graceful FD fall-back. The wiring is what
 * matters.
 */
TEST_CASE("HPO_Host::analytical_gradient end-to-end with one PC_LAMMPS "
          "cohesive-E script (Phase 3 follow-up)",
          "[hpo_host_e2e][phase3][.][!benchmark]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();

    ScratchDir sd;
    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    const fs::path script   = sd.dir / "cohesive_e.in";

    write_minimal_ta_dbfile(train_db, /*offset_x=*/0.0);
    write_minimal_ta_dbfile(val_db,   /*offset_x=*/0.07);

    {
        std::ofstream os(val_cfg);
        os << "DBFILE " << val_db.string() << "\n";
    }

    /* Mirrors simple_bcc_pilot_script from test_hpo_phase3_pilot.cpp.
     * 16-atom BCC cell, box/relax iso, cohesive E per atom. */
    {
        std::ofstream os(script);
        os <<
            "units metal\n"
            "boundary p p p\n"
            "atom_style atomic\n"
            "atom_modify map array\n"
            "lattice bcc 3.301\n"
            "region box block 0 2 0 2 0 2 units lattice\n"
            "create_box 1 box\n"
            "create_atoms 1 region box\n"
            "mass 1 180.94788\n"
            "pair_style tadah\n"
            "pair_coeff * * EXTERN Ta\n"
            "neighbor 2.0 bin\n"
            "neigh_modify delay 10 check yes\n"
            "compute eng all pe/atom\n"
            "compute eatoms all reduce sum c_eng\n"
            "fix relax all box/relax iso 0.0 vmax 0.001\n"
            "min_style cg\n"
            "min_modify dmax 0.01 line quadratic\n"
            "minimize 1e-12 1e-12 1000 1000\n"
            "unfix relax\n"
            "variable e_per equal $(c_eatoms/atoms)\n"
            "run 0\n";
    }

    {
        std::ofstream os(hpo_cfg);
        os <<
            "OPTIMIZER\n"
            "LIB DLIB\n"
            "ALGO BFGS\n"
            "MAXEVAL 1\n"
            "ENDOPTIMIZER\n"
            "LOSS L2\n"
            "FAILSCORE 1e8\n"
            "PC_EMAE 0 1\n"
            "PC_LAMMPS --script " << script.string()
            << " --varloss e_per 0 1\n"
            "OPTIM CGRID2B (1) 0.0 4.0\n";
    }

    tadah::core::Context ctx = make_ta_master_context(train_db, hpo_cfg, val_cfg);
    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.LS.size() == 1u);

    Host host(ctx, trg, val_cfg.string());
    REQUIRE(host.train_size() == 1u);

    ParamVec p(1);
    p(0) = 0.5;
    double f0 = std::numeric_limits<double>::quiet_NaN();
    /* operator() may FAIL_SCORE the script if the trainer's minimal pot
     * gives a degenerate relaxation. The test accepts either: a finite
     * loss (LAMMPS path succeeded) or FAIL_SCORE (graceful clamp). */
    REQUIRE_NOTHROW(f0 = host(p));
    CHECK(std::isfinite(f0));

    /* The real production wiring: analytical_gradient should construct
     * a SimsGradProvider (because trg_.LS is non-empty) and either fold
     * sims_grad into g (analytical path) or route to FD (all-or-nothing
     * fall-back). Either is valid. The test asserts the call returns
     * cleanly. */
    ParamVec g(1);
    std::vector<int> needs_fd;
    bool ok = false;
    REQUIRE_NOTHROW(ok = host.analytical_gradient(p, g, needs_fd));
    REQUIRE(ok);
    CHECK(std::isfinite(g(0)));

    Host::reset_globals_for_test();
}

namespace {
struct DatasetHostGuard {
    DatasetHostGuard() { Host::reset_best_state_for_fresh_run(); }
    ~DatasetHostGuard() { Host::reset_best_state_for_fresh_run(); }
};
std::string read_dataset_file(const fs::path& path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::vector<double> last_dataset_row(const fs::path& path) {
    std::ifstream in(path);
    std::string line;
    std::vector<double> row;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        row.clear();
        std::istringstream values(line);
        double v;
        while (values >> v) row.push_back(v);
    }
    return row;
}
const std::string dataset_host_head =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
    "LOSS L2\nQFLUSH 1\nDIGITS 12\nOPTIM CGRID2B (1) 0.0 4.0\n";
}

TEST_CASE("PC dataset host loss and reports match resolved targets", "[pc_dataset][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", first = sd.dir / "first.db", second = sd.dir / "second.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(first, 0.07, -0.04);
    write_minimal_ta_dbfile(second, 0.13, -0.16);
    const auto second_body = read_dataset_file(second);
    std::ofstream(second, std::ios::app) << second_body; // second dataset has two structures
    std::ofstream(val) << "DBFILE " << first.string() << "\nDBFILE " << second.string() << "\n";
    bool global_energy = false, global_force = false;
    std::string lines;
    SECTION("individually weighted energy and forces") {
        lines = "PC_ERMSE (2) 20 3\nPC_ERMSE (1) 10 2\n"
                "PC_FRMSE (1) 0.1 2\nPC_FRMSE (2) 0.2 0.5\n";
    }
    SECTION("indexed energy with global forces") {
        global_force = true;
        lines = "PC_ERMSE (2) 20 3\nPC_ERMSE (1) 10 2\nPC_FRMSE 0.1 2\n";
    }
    SECTION("legacy global targets") {
        global_energy = global_force = true;
        lines = "PC_ERMSE 10 2\nPC_FRMSE 0.1 2\n";
    }
    std::ofstream(hpo) << dataset_host_head << lines;
    auto ctx = make_ta_master_context(train, hpo, val);
    ctx.remove("FORCE"); ctx.add("FORCE", "true");
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
    host.prepare_best_publication();
    REQUIRE(host.val_size() == 3);
    ParamVec p(1); p(0) = 0.5;
    const double total = host(p);
    host.flush_remaining();
    const auto stats = last_dataset_row("stats.tadah");
    const auto losses = last_dataset_row("loss.tadah");
    REQUIRE(stats.size() == 8); // iteration, slot, E[1,2,avg], F[1,2,avg]
    REQUIRE(losses.size() == 9);
    CHECK(stats[4] == Approx((stats[2] + stats[3])/2));
    CHECK(stats[7] == Approx((stats[5] + stats[6])/2));
    const auto l2 = [](double x, double t, double w) { return (x-t)*(x-t)*w; };
    const double e1 = l2(stats[2], 10, 2);
    const double e2 = l2(stats[3], global_energy ? 10 : 20, global_energy ? 2 : 3);
    const double f1 = l2(stats[5], 0.1, 2);
    const double f2 = l2(stats[6], global_force ? 0.1 : 0.2, global_force ? 2 : 0.5);
    const double e = global_energy ? l2(stats[4], 10, 2) : (e1+e2)/2;
    const double f = global_force ? l2(stats[7], 0.1, 2) : (f1+f2)/2;
    CHECK(total == Approx((e+f)/2).epsilon(1e-9));
    CHECK(losses[2] == Approx(e1)); CHECK(losses[3] == Approx(e2)); CHECK(losses[4] == Approx(e));
    CHECK(losses[5] == Approx(f1)); CHECK(losses[6] == Approx(f2)); CHECK(losses[7] == Approx(f));
    CHECK(losses[8] == Approx(total));
    const auto best = read_dataset_file(trg.canonicalBestPotFile);
    CHECK(best.find(global_energy ? "global (loss of average metric)" : "indexed (average of dataset losses)")
          != std::string::npos);
    if (!global_energy) {
        CHECK(best.find("[2] " + second.string() + " target=20 weight=3") != std::string::npos);
    }
    auto& log = tadah::hpo::HPOLog::instance();
    log.open((sd.dir / "journal.log").string(), "test", sd.dir.string(), 1);
    log.write_loss_section(trg);
    log.finish("TEST");
    const auto journal = read_dataset_file(sd.dir / "journal.log");
    CHECK(journal.find("target=10 weight=2") != std::string::npos);
    CHECK(journal.find("target=10000") == std::string::npos);
}

TEST_CASE("PC dataset host rejects incomplete coverage and empty validation files", "[pc_dataset][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", first = sd.dir / "first.db", second = sd.dir / "second.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(first, 0.07);
    write_minimal_ta_dbfile(second, 0.13);
    std::string lines = "PC_ERMSE (1-2) 0 1\n";
    std::string filter, message;
    SECTION("missing assignment") {
        lines = "PC_ERMSE (1) 0 1\n";
        message = "dataset 2 (" + second.string() + ")";
    }
    SECTION("out of range") {
        lines = "PC_FRMSE (1-3) 0 1\n";
        message = "outside validation DBFILE range 1-2";
    }
    SECTION("empty file") {
        std::ofstream(second).close();
        message = "Empty validation dataset 2";
    }
    SECTION("filter empties a file") {
        write_minimal_ta_dbfile(second, 0.13, -10);
        filter = "EFILTER -1 1\n";
        message = "Empty validation dataset 2";
    }
    std::ofstream(val) << "DBFILE " << first.string() << "\nDBFILE " << second.string() << "\n" << filter;
    std::ofstream(hpo) << dataset_host_head << lines;
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    CHECK_THROWS_WITH(Host(ctx, trg, val.string()), Catch::Matchers::Contains(message));
}

/* FORCE and STRESS decide only what training fits (the -c configuration,
 * or -F/-S); validation predicts forces whenever PC_FRMSE is set and
 * stresses whenever PC_SRMSE is set. A FORCE or STRESS key in the
 * --validation file therefore has no role: the host refuses it at its line
 * (through CoreUserConfigParser::refuse_keys), naming the key, the line and
 * the file, instead of ignoring it. The host is only constructed: nothing
 * is trained or evaluated. */
namespace {
const std::string kRefusalTarget = dataset_host_head + "PC_ERMSE 0 1\n";

/* Constructs the host on the --validation file `val`. */
void construct_host_on_validation_file(const fs::path& train,
                                       const fs::path& hpo,
                                       const fs::path& val) {
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
}
}  // namespace

/* Pins: `FORCE true` on line 2 of the --validation file (after its DBFILE
 * line) makes the HPO_Host constructor throw a message that names FORCE,
 * says "not accepted", names line 2 and the validation file. Before the
 * fix the key was parsed into the validation context and silently ignored,
 * so the constructor did not throw. */
TEST_CASE("A FORCE key in the --validation file is refused at its line",
          "[validation_file][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << vdb.string() << "\nFORCE true\n";
    std::ofstream(hpo) << kRefusalTarget;
    CHECK_THROWS_WITH(construct_host_on_validation_file(train, hpo, val),
                      Catch::Matchers::Contains("FORCE") &&
                      Catch::Matchers::Contains("not accepted") &&
                      Catch::Matchers::Contains("line 2") &&
                      Catch::Matchers::Contains(val.string()));
}

/* Pins: the same for `STRESS true` on line 2. Before the fix the key was
 * silently ignored and the constructor did not throw. */
TEST_CASE("A STRESS key in the --validation file is refused at its line",
          "[validation_file][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << vdb.string() << "\nSTRESS true\n";
    std::ofstream(hpo) << kRefusalTarget;
    CHECK_THROWS_WITH(construct_host_on_validation_file(train, hpo, val),
                      Catch::Matchers::Contains("STRESS") &&
                      Catch::Matchers::Contains("not accepted") &&
                      Catch::Matchers::Contains("line 2") &&
                      Catch::Matchers::Contains(val.string()));
}

/* Pins: a FORCE key in a file that the --validation file includes with
 * CONFIG (on line 2 of the validation file) is refused too, and the
 * message names the included file and the key's line there (line 3: a
 * comment and a blank line come first), not the validation file's CONFIG
 * line. Before the fix the included FORCE was merged into the validation
 * context and silently ignored, so the constructor did not throw. */
TEST_CASE("A FORCE key in a file included by the --validation file is "
          "refused at its line", "[validation_file][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    const auto inc = sd.dir / "shared_val_settings.tadah";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::ofstream(inc) << "# shared validation settings\n\nFORCE true\n";
    std::ofstream(val) << "DBFILE " << vdb.string() << "\nCONFIG "
                       << inc.string() << "\n";
    std::ofstream(hpo) << kRefusalTarget;
    CHECK_THROWS_WITH(construct_host_on_validation_file(train, hpo, val),
                      Catch::Matchers::Contains("FORCE") &&
                      Catch::Matchers::Contains("not accepted") &&
                      Catch::Matchers::Contains("line 3") &&
                      Catch::Matchers::Contains(inc.string()));
}

TEST_CASE("PC dataset zero weight retains its dataset skip filter", "[pc_dataset][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", first = sd.dir / "first.db", second = sd.dir / "second.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(first, 0.07, -0.04);
    write_minimal_ta_dbfile(second, 0.13, -10);
    std::ofstream(val) << "DBFILE " << first.string() << "\nDBFILE " << second.string() << "\n";
    std::ofstream(hpo) << dataset_host_head
        << "FAILSCORE 1000000\nPC_ERMSE (1) 0 1\nPC_ERMSE (2) 0 0 --skip-above 1\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
    host.prepare_best_publication();
    ParamVec p(1); p(0) = 0.5;
    CHECK(host(p) == trg.FAIL_SCORE);
    host.flush_remaining();
    const auto losses = last_dataset_row("loss.tadah");
    REQUIRE(losses.size() == 6);
    CHECK(losses[3] == 0);
}

TEST_CASE("params.tadah columns: iteration then slot then OPTIM dimensions "
          "sorted by key and index", "[HPO][host][params_order]")
{
    /* The optimiser vector -- and with it params.tadah, best_params.tadah,
     * hpo_search_space.tsv and the journal -- is ordered by the OPTIM
     * key (std::map, i.e. byte-wise alphabetical), then by index within
     * the key, regardless of the order of the OPTIM lines. */
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", valdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(valdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo)
        << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
           "LOSS L2\nQFLUSH 1\n"
           "OPTIM SGRID2B (2) 4.0 6.0\n"
           "OPTIM CGRID2B (2) 1.0 2.0\n"
           "OPTIM CGRID2B (1) 0.1 0.9\n"
           "PC_ERMSE 10 1\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());

    const std::vector<std::string> expected = {"CGRID2B[1]", "CGRID2B[2]",
                                               "SGRID2B[2]"};
    CHECK(trg.spec.parameter_labels == expected);

    trg.write_search_space_files(sd.dir.string());
    {
        std::ifstream tsv(sd.dir / "hpo_search_space.tsv");
        std::vector<std::string> labels;
        std::string line;
        bool header = true;
        while (std::getline(tsv, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (header) { header = false; continue; }
            std::istringstream is(line);
            std::string dim, label;
            is >> dim >> label;
            labels.push_back(label);
        }
        CHECK(labels == expected);
    }

    Host host(ctx, trg, val.string());
    host.prepare_best_publication();
    ParamVec p(3); p(0) = 0.5; p(1) = 1.4; p(2) = 5.0;
    (void)host(p);
    host.flush_remaining();

    for (const auto* file : {&trg.paramsFile, &trg.bestParamsFile}) {
        INFO(*file);
        std::ifstream in(*file);
        std::string head;
        REQUIRE(std::getline(in, head));
        std::istringstream hs(head);
        std::vector<std::string> cols;
        for (std::string c; hs >> c;) cols.push_back(c);
        const std::vector<std::string> want = {"#ITERATION", "SLOT",
                                               "CGRID2B[1]", "CGRID2B[2]",
                                               "SGRID2B[2]"};
        CHECK(cols == want);
        const auto row = last_dataset_row(*file);
        REQUIRE(row.size() == 5);
        /* params.tadah: global evaluation number (process-wide counter);
         * best_params.tadah: improvement ordinal -- 1 for the first best
         * of a fresh run. */
        if (file == &trg.bestParamsFile) CHECK(row[0] == 1.0);
        else                             CHECK(row[0] >= 1.0);
        CHECK(row[2] == Approx(0.5));
        CHECK(row[3] == Approx(1.4));
        CHECK(row[4] == Approx(5.0));
    }
}

TEST_CASE("A second run in the same process restarts the best comparator "
          "and keeps the evaluation counter", "[HPO][host][durable_best]")
{
    /* prepare_best_publication() starts every run with
     * reset_best_state_for_fresh_run(). An embedding that runs HPO twice in
     * one process must not judge the second run against the first run's
     * best -- its first feasible candidate owns the new best_pot.tadah even
     * when the first run did better -- while the evaluation counter (and
     * with it the ordered result writer) continues monotonically and
     * stagnation counts from the new start. */
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", valdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(valdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << dataset_host_head << "PC_ERMSE 10 1\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
    namespace db = tadah::hpo::durable_best;

    /* First run: two distinct candidates, then the worse one again, which
     * is not an improvement and so leaves a stagnation count behind. */
    host.prepare_best_publication();
    ParamVec a(1); a(0) = 0.5;
    ParamVec b(1); b(0) = 1.2;
    const double la = host(a);
    const double lb = host(b);
    REQUIRE(std::isfinite(la));
    REQUIRE(std::isfinite(lb));
    REQUIRE(la != lb);
    const ParamVec worse = la > lb ? a : b;
    const double worse_loss = std::max(la, lb);
    const double best_loss  = std::min(la, lb);
    REQUIRE(host(worse) == worse_loss);
    host.flush_remaining();

    const std::size_t g = Host::g_iter_value();
    REQUIRE(g >= 3u);
    CHECK(Host::current_best() == best_loss);
    CHECK(Host::current_best_iter() >= 1u);
    CHECK(Host::stagnation_count() >= 1u);
    const auto first = db::read_key_file(trg.canonicalBestPotFile);
    REQUIRE(first);
    CHECK(first->loss == best_loss);

    /* Second run in the same process. */
    host.prepare_best_publication();
    CHECK(Host::current_best() == std::numeric_limits<double>::max());
    CHECK(Host::current_best_iter() == 0u);
    CHECK(Host::g_iter_value() == g);
    CHECK(Host::current_last_best() == g);
    CHECK(Host::stagnation_count() == 0u);
    CHECK_FALSE(fs::exists(trg.canonicalBestPotFile));

    /* Its first candidate is its best, although the first run did better. */
    REQUIRE(host(worse) == worse_loss);
    host.flush_remaining();
    CHECK(Host::g_iter_value() == g + 1);
    CHECK(Host::current_best() == worse_loss);
    CHECK(Host::current_best_iter() == 1u);
    CHECK(Host::current_last_best() == g + 1);
    CHECK(Host::stagnation_count() == 0u);
    const auto second = db::read_key_file(trg.canonicalBestPotFile);
    REQUIRE(second);
    CHECK(second->loss == worse_loss);
    CHECK(second->eval == g + 1);
    const auto best_row = last_dataset_row(trg.bestParamsFile);
    REQUIRE(best_row.size() == 3);   // improvement ordinal, slot, CGRID2B[1]
    CHECK(best_row[0] == 1.0);
    CHECK(best_row[2] == Approx(worse(0)));
}

/* Pins: prepare_best_publication(), which starts every run, forgets the
 * per-chain --hotvar values an earlier run in the process converged to and
 * keeps the WARM seeds: after a first run left chain 0's hot_a0 = 3.31 and
 * chain 3's b = 7.0 (warmstart::set, as each evaluation stores its
 * converged value) and the second run's WARM init primed b = 6.5
 * (warmstart::prime), chain 0 has no hot_a0 (its script starts from the
 * --hotvar initial value), chain 3's b is WARM's 6.5, and the best-HOT
 * snapshot is empty.
 * Before the fix: chain 0 read 3.31 and chain 3 read 7.0, the first run's
 * values (the run boundary reset only the best-HOT snapshot), so the
 * second run of a task file with two TASK hpo blocks started its
 * relaxations where the first run's last evaluation ended. */
TEST_CASE("A second run in the same process starts its --hotvar values from "
          "their declared start or the WARM seed",
          "[HPO][host][warmstart]")
{
    namespace ws = tadah::hpo::warmstart;
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", valdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(valdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << dataset_host_head << "PC_ERMSE 10 1\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());

    ws::clear();
    ws::set(0, "hot.in", "hot_a0", 3.31);
    ws::set(3, "hot.in", "b", 7.0);
    ws::prime("hot.in", "b", 6.5);
    host.prepare_best_publication();
    CHECK_FALSE(ws::get(0, "hot.in", "hot_a0").has_value());
    const auto b = ws::get(3, "hot.in", "b");
    REQUIRE(b.has_value());
    CHECK(*b == 6.5);
    CHECK_FALSE(ws::best_get("hot.in", "b").has_value());
    ws::clear();
}

/* ---------------------------------------------------------------------
 * PC_OBSERVABLE column labels and journal names.
 *
 * The k-th PC_OBSERVABLE line (k 1-based, in configuration order) owns the
 * loss.tadah column L(<tag>)[k] and the outvar.tadah column <tag>[k], with
 * <tag> the observable's column tag (cohesive_energy: Ecoh), like the
 * per-dataset L(ERMSE)[i]. The journal names it <kind>[k]. The index used
 * to be 0-based and the column L(cohesive_energy[0]) was cut at every
 * DIGITS (the title budget is max(DIGITS + 9, 15) - 4, 11 to 20
 * characters) to "L(cohesive*".
 * ------------------------------------------------------------------- */
namespace {

std::vector<std::string> header_columns(const std::string& file)
{
    std::ifstream in(file);
    std::string head;
    std::getline(in, head);
    std::istringstream is(head);
    std::vector<std::string> cols;
    for (std::string c; is >> c;) cols.push_back(c);
    return cols;
}

std::string first_line(const std::string& file)
{
    std::ifstream in(file);
    std::string head;
    std::getline(in, head);
    return head;
}

std::string observable_lines(std::size_t n)
{
    std::string s;
    for (std::size_t k = 0; k < n; ++k)
        s += "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
             "--weight 1 --supercell 2 2 2 --hotvar a0 2.7 3.9 3.3\n";
    return s;
}

/* An observable-only hpotarget on the minimal Ta fixture. */
void write_observable_target(const fs::path& hpo, std::size_t n, int digits)
{
    std::ofstream(hpo)
        << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
        << "LOSS L2\nFAILSCORE 1e8\nDIGITS " << digits << "\n"
        << observable_lines(n) << "OPTIM CGRID2B (1) 0.0 4.0\n";
}

}  // namespace

TEST_CASE("PC_OBSERVABLE columns are L(Ecoh)[k] and Ecoh[k] with a 1-based "
          "k, never truncated at any DIGITS",
          "[HPO][host][pc_observable][observable_labels]")
{
    tadah::core::PeriodicTable::initialize();
    DatasetHostGuard guard;
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        for (int digits = 1; digits <= 15; ++digits) {
            INFO(n << " PC_OBSERVABLE line(s), DIGITS " << digits);
            ScratchDir sd;
            const auto train = sd.dir / "train.db", hpo = sd.dir / "config.hpo";
            write_minimal_ta_dbfile(train, 0);
            write_observable_target(hpo, n, digits);
            auto ctx = make_ta_master_context(train, hpo, sd.dir / "unused.val");
            tadah::hpo::HPOTargets trg(ctx, hpo.string());
            Host host(ctx, trg, "");

            std::vector<std::string> want_loss = {"#ITERATION", "SLOT"};
            std::vector<std::string> want_outvar = {"#ITERATION", "IT-TIME/ms",
                                                    "SLOT"};
            for (std::size_t k = 1; k <= n; ++k) {
                want_loss.push_back("L(Ecoh)[" + std::to_string(k) + "]");
                want_outvar.push_back("Ecoh[" + std::to_string(k) + "]");
            }
            want_loss.push_back("TOTAL_LOSS/N");
            CHECK(header_columns(trg.lossFile) == want_loss);
            CHECK(header_columns(trg.bestLossFile) == want_loss);
            CHECK(header_columns(trg.outvarFile) == want_outvar);
            CHECK(header_columns(trg.bestOutvarFile) == want_outvar);
            for (const auto* file :
                 {&trg.lossFile, &trg.bestLossFile, &trg.outvarFile,
                  &trg.bestOutvarFile, &trg.paramsFile, &trg.bestParamsFile,
                  &trg.statsFile, &trg.bestStatsFile}) {
                INFO(*file << ": " << first_line(*file));
                CHECK(first_line(*file).find('*') == std::string::npos);
            }
        }
    }
    Host::reset_globals_for_test();
}

TEST_CASE("PC_OBSERVABLE columns past k = 99 keep their index when DIGITS "
          "leaves no room for the whole label",
          "[HPO][host][pc_observable][observable_labels]")
{
    /* L(Ecoh)[100] is 12 characters: DIGITS 1..6 give an 11-character
     * title, so the tag is shortened and the index kept (the params.tadah
     * rule); DIGITS 7 gives 12 characters and the label fits. */
    tadah::core::PeriodicTable::initialize();
    DatasetHostGuard guard;
    for (const int digits : {1, 7}) {
        INFO("DIGITS " << digits);
        ScratchDir sd;
        const auto train = sd.dir / "train.db", hpo = sd.dir / "config.hpo";
        write_minimal_ta_dbfile(train, 0);
        write_observable_target(hpo, 100, digits);
        auto ctx = make_ta_master_context(train, hpo, sd.dir / "unused.val");
        tadah::hpo::HPOTargets trg(ctx, hpo.string());
        REQUIRE(trg.observables.size() == 100u);
        Host host(ctx, trg, "");

        const auto loss = header_columns(trg.lossFile);
        const auto outvar = header_columns(trg.outvarFile);
        REQUIRE(loss.size() == 2u + 100u + 1u);
        REQUIRE(outvar.size() == 3u + 100u);
        CHECK(loss[2] == "L(Ecoh)[1]");
        CHECK(loss[2 + 98] == "L(Ecoh)[99]");
        CHECK(loss[2 + 99] == (digits == 1 ? "L(Eco*[100]" : "L(Ecoh)[100]"));
        CHECK(outvar[3] == "Ecoh[1]");
        CHECK(outvar[3 + 99] == "Ecoh[100]");

        /* The journal's loss section names the columns as the files do. */
        const auto journal = sd.dir / "journal.log";
        auto& log = tadah::hpo::HPOLog::instance();
        log.open(journal.string(), "test", sd.dir.string(), 1);
        log.write_loss_section(trg);
        log.finish("TEST");
        std::string entry;
        {
            std::istringstream lines(read_dataset_file(journal));
            for (std::string line; std::getline(lines, line);)
                if (line.find("cohesive_energy[100]") != std::string::npos) {
                    entry = line;
                    break;
                }
        }
        INFO(entry);
        CHECK(entry.find("columns " + loss[2 + 99] + ", " + outvar[3 + 99]) !=
              std::string::npos);
    }
    Host::reset_globals_for_test();
}

TEST_CASE("The journal names the k-th PC_OBSERVABLE cohesive_energy[k], "
          "1-based, in the loss section and the new-best breakdown",
          "[HPO][host][pc_observable][observable_labels]")
{
    tadah::core::PeriodicTable::initialize();
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", hpo = sd.dir / "config.hpo";
    const auto journal = sd.dir / "journal.log";
    write_minimal_ta_dbfile(train, 0);
    write_observable_target(hpo, 2, 6);
    auto ctx = make_ta_master_context(train, hpo, sd.dir / "unused.val");
    /* The relaxation inside the observable needs forces. */
    ctx.remove("FORCE");
    ctx.add("FORCE", "true");
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, "");

    auto& log = tadah::hpo::HPOLog::instance();
    log.open(journal.string(), "test", sd.dir.string(), 1);
    log.write_loss_section(trg);
    host.prepare_best_publication();
    ParamVec p(1);
    p(0) = 0.5;
    const double loss = host(p);
    host.flush_remaining();
    log.finish("TEST");
    REQUIRE(std::isfinite(loss));
    REQUIRE(loss > 0.0);

    const auto text = read_dataset_file(journal);
    INFO(text);
    CHECK(text.find("cohesive_energy[1]") != std::string::npos);
    CHECK(text.find("cohesive_energy[2]") != std::string::npos);
    CHECK(text.find("cohesive_energy[0]") == std::string::npos);

    /* The new-best line of the first evaluation lists the observable
     * terms by the same names. */
    std::istringstream lines(text);
    std::string top;
    for (std::string line; std::getline(lines, line);)
        if (line.find("top: ") != std::string::npos) { top = line; break; }
    REQUIRE_FALSE(top.empty());
    CHECK(top.find("cohesive_energy[1]=") != std::string::npos);
    CHECK(top.find("cohesive_energy[2]=") != std::string::npos);
    Host::reset_globals_for_test();
}

/* ---------------------------------------------------------------------
 * PC_OBSERVABLE analytical gradient.
 *
 * GRAD_STRATEGY ANALYTICAL/HYBRID differentiate the host loss -- the mean
 * of every term, validation metrics and PC_OBSERVABLE lines alike -- with
 * respect to the OPTIM parameters. A cohesive energy depends on a
 * descriptor parameter twice: through the refitted weights, and directly,
 * through the descriptors of the relaxed crystal and of the isolated atom
 * at fixed weights (the relaxation itself adds nothing at a stationary
 * geometry). The model is the examples' shared Ta potential with the
 * settings the analytical gradient needs (OALGO 4 and a ridge LAMBDA),
 * i.e. the configuration of example 22_pc_observable.
 * ------------------------------------------------------------------- */
namespace {

const fs::path& examples_common_dir()
{
    static const fs::path dir = fs::path(TADAH_HPO_EXAMPLES_DIR) / "common";
    return dir;
}

tadah::core::Context examples_ta_context()
{
    const fs::path& common = examples_common_dir();
    tadah::core::Context c(tadah::core::CoreUserConfigParser().parse(
        (common / "config.tadah").string()));
    c.remove("DBFILE");
    c.add("DBFILE", (common / "data" / "elastic.tadahdb").string());
    c.add("DBFILE", (common / "data" / "eos_bcc.tadahdb").string());
    c.remove("OALGO");
    c.add("OALGO", 4);
    c.remove("LAMBDA");
    c.add("LAMBDA", 1e-8);
    return c;
}

}  // namespace

TEST_CASE("PC_OBSERVABLE analytical gradient matches central differences of "
          "the host loss",
          "[HPO][host][pc_observable][analytical_gradient]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    DatasetHostGuard guard;
    std::string metric;
    SECTION("the observable alone") {}
    SECTION("the observable and a validation metric") {
        metric = "PC_ERMSE 0 0.0001\n";
    }
    INFO("metric line: '" << metric << "'");

    ScratchDir sd;
    const auto hpo = sd.dir / "config.hpo", val = sd.dir / "config.val";
    std::ofstream(val) << "DBFILE "
                       << (examples_common_dir() / "data" / "elastic.tadahdb").string()
                       << "\n";
    std::ofstream(hpo)
        << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nGRAD_STRATEGY HYBRID\n"
           "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\n"
        << metric
        << "PC_OBSERVABLE cohesive_energy --species Ta --target 8.10 "
           "--weight 800 --hotvar a0 2.5 4.5 3.302 --maxiter 500 "
           "--maxeval 500 --etol 1e-12\n"
           "OPTIM RCUT2B (1) 4.0 6.0\n"
           "OPTIM SGRIDMB (1) 0.1 0.4\n";
    auto ctx = examples_ta_context();
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    REQUIRE(trg.spec.parameter_labels ==
            std::vector<std::string>{"RCUT2B", "SGRIDMB"});
    Host host(ctx, trg, metric.empty() ? std::string() : val.string());
    host.prepare_best_publication();

    ParamVec p(2);
    p(0) = 5.6;   // the configuration values
    p(1) = 0.2;
    const double f0 = host(p);
    /* A converged relaxation: not the FAILSCORE, not the unconverged
     * penalty (Ecoh = 100 eV/atom, a loss of about 3e6). */
    REQUIRE(std::isfinite(f0));
    REQUIRE(f0 < 1.0);

    ParamVec g;
    std::vector<int> needs_fd;
    REQUIRE(host.analytical_gradient(p, g, needs_fd));
    REQUIRE(needs_fd.empty());
    REQUIRE(g.size() == 2);

    const double step[2] = {1e-4, 1e-5};
    for (long i = 0; i < 2; ++i) {
        ParamVec up = p, down = p;
        up(i) += step[i];
        down(i) -= step[i];
        const double fd = (host(up) - host(down)) / (2.0 * step[i]);
        INFO(trg.spec.parameter_labels[static_cast<std::size_t>(i)]
             << ": analytical " << g(i) << ", central difference " << fd);
        CHECK(g(i) == Approx(fd).epsilon(1e-2).margin(1e-6));
    }
    host.flush_remaining();
    Host::reset_globals_for_test();
}

/* ---- two atoms at the same position -------------------------------------
 * NNFinder refuses a structure with two atoms at one position, naming the
 * structure and the atoms (report_leftovers.md section 3, option A), and
 * HPO_Host builds the neighbour lists of the validation and training sets
 * in its constructor. So `tadah hpo` stops at startup with that message
 * instead of failing every evaluation ("Error in DGELSD: info = -4" in
 * training, a NaN force metric in validation) or measuring an energy
 * silently. The host is only constructed. */
namespace {
/* Appends to @p path a 3-atom Ta structure in a 3.301 A cube labelled
 * @p label, with atoms 1 and 3 both at the origin and atom 2 at the cube
 * centre. */
void append_coincident_ta(const fs::path& path, const std::string& label) {
    std::ofstream os(path, std::ios::app);
    os << label << "\n"
       << "1.0 1.0 1.0\n"
       << "-0.03 0.0\n"
       << "3.301 0.0 0.0\n"
       << "0.0 3.301 0.0\n"
       << "0.0 0.0 3.301\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "Ta 0.0 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta 1.6505 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta 0.0 0.0 0.0  0.0 0.0 0.0\n"
       << "\n";
}
}  // namespace

/* Pins: a validation DBFILE, or a training DBFILE, whose second structure
 * has two Ta atoms at the same position makes the HPO_Host constructor
 * throw std::invalid_argument naming the structure ("structure 2 of 2" and
 * its label) and the atoms ("atoms 1 and 3 (Ta, Ta)", "same position").
 * Before the fix: the constructor built the lists with the pair at
 * distance 0 and did not throw, so the CHECK_THROWS_AS and the
 * CHECK_THROWS_WITH failed ("no exception was thrown") in both sections. */
TEST_CASE("A structure with two atoms at the same position stops the host at "
          "startup, naming the structure and the atoms",
          "[coincident][host]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::string label;
    SECTION("in the validation set") {
        label = "Ta3 coincident (validation)";
        append_coincident_ta(vdb, label);
    }
    SECTION("in the training set") {
        label = "Ta3 coincident (training)";
        append_coincident_ta(train, label);
    }
    std::ofstream(val) << "DBFILE " << vdb.string() << "\n";
    std::ofstream(hpo) << kRefusalTarget;
    CHECK_THROWS_AS(construct_host_on_validation_file(train, hpo, val),
                    std::invalid_argument);
    CHECK_THROWS_WITH(construct_host_on_validation_file(train, hpo, val),
                      Catch::Matchers::Contains("structure 2 of 2") &&
                      Catch::Matchers::Contains("'" + label + "'") &&
                      Catch::Matchers::Contains("atoms 1 and 3 (Ta, Ta)") &&
                      Catch::Matchers::Contains("same position"));
}

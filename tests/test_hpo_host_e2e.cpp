// HPO_Host end-to-end integration test.
//
// Validates production wiring of HPO_Host<ParamVec, LammpsRunner>::
// analytical_gradient — specifically the SimsGradProvider lambda
// construction at hpo_host_impl.hpp:~2186. Prior coverage was indirect:
// `[live_trainer][sims_grad_provider]` (test_hpo_analytical.cpp) tested
// `compute_analytical_gradient_from_trainer`'s ability to fold in a
// synthetic provider, and `[adjoint_orchestrator]` reproduced the
// lambda's body inline on a real LAMMPS instance. Neither exercised
// HPO_Host's own construction of the lambda.
//
// Three cases (tag [hpo_host_e2e]):
//  1. No-LAMMPS path: trg_.LS empty ⇒ sims_grad_provider stays nullptr,
//     analytical_gradient goes through the descriptor-side path only.
//     Verifies HPO_Host constructs and operator() populates the trainer
//     cache that analytical_gradient consumes.
//  2. PC_OBSERVABLE path: one cohesive_energy observable ⇒ the
//     ObservableGradProvider is wired.
//  3. PC_LAMMPS path: trg_.LS has one cohesive-E script ⇒ provider is
//     wired; calling analytical_gradient runs the full LAMMPS lifecycle
//     in-process. Its inputs carry no direct θ term, so the chain sends
//     every θ to finite differences (pinned below).
//
// The three are hidden (`[.][!benchmark]`), so not discovered: they run in
// the CTest entry hpo_host_e2e (label bench), `test_hpo "[hpo_host_e2e]"`,
// one process. They train on the minimal 4-atom Ta structure written
// below and read no potential file; they need no skip.

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
#include <map>
#include <regex>
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
    /* ESIGMA 250 meV/atom: the energy row of a 4-atom cell of
     * write_minimal_ta_dbfile keeps the factor 1 (1/(4 x 0.25 eV)),
     * so the fits are those the cases were written for. */
    c.add("ESIGMA", 250.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);

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
          "PC_LAMMPS line",
          "[hpo_host_e2e][adjoint][.][!benchmark]")
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
 * not contaminate the PC_OBSERVABLE relaxer: the three hidden cases run
 * in one process, in declaration order (CTest hpo_host_e2e).
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
 * test_hpo_adjoint_pilot.cpp). This case verifies that:
 *   - HPO_Host constructs with a non-empty trg_.LS
 *   - operator()'s LAMMPS-eval path executes (PC_LAMMPS scripts run)
 *   - analytical_gradient builds the SimsGradProvider lambda, and the
 *     chain sends every theta to FD, as the provider's inputs carry no
 *     direct theta term (below)
 *
 * The fixture's trainer's pot is fit on a single 4-atom Ta cell with
 * D2_BP[2,2]+BF_Linear, a low-expressivity potential; the script runs
 * with it (the provider's script does not fail here: with the chain
 * through W alone, before the direct theta term, the analytical path
 * was taken).
 */
TEST_CASE("HPO_Host::analytical_gradient end-to-end with one PC_LAMMPS "
          "cohesive-E script",
          "[hpo_host_e2e][adjoint][.][!benchmark]")
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

    /* Mirrors simple_bcc_pilot_script from test_hpo_adjoint_pilot.cpp.
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
        /* A gradient-free optimiser: a PC_LAMMPS line refuses a
         * gradient-based one, DLIB:BFGS among them (it was DLIB:BFGS). The
         * case calls analytical_gradient directly. Before the fix: passes as
         * well (a pin). */
        std::ofstream os(hpo_cfg);
        os <<
            "OPTIMIZER\n"
            "LIB NLOPT\n"
            "ALGO LN_SBPLX\n"
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

    /* The real production wiring: analytical_gradient constructs a
     * SimsGradProvider (trg_.LS is non-empty), which runs the script and
     * fills the chain's input. That input carries no direct θ term
     * (SimsGradScriptInput::direct_theta: the adjoint snapshot holds no
     * derivative with respect to θ), and CGRID2B (1) changes the
     * descriptors, so compute_sims_grad_dWchain sends every θ to finite
     * differences and the orchestrator falls back to FD for all of them.
     * Pins: the call returns true, g is finite, and needs_fd lists every
     * θ: flat index 0, the one OPTIM dimension.
     * Before the fix (src/sims_grad.cpp of v1-dev, the chain through W
     * alone): needs_fd was empty, and the case run alone printed the
     * diagnostic "analytical=1 | FD-fallback=0": the chain was taken
     * without the direct θ term. */
    ParamVec g(1);
    std::vector<int> needs_fd;
    bool ok = false;
    REQUIRE_NOTHROW(ok = host.analytical_gradient(p, g, needs_fd));
    REQUIRE(ok);
    CHECK(std::isfinite(g(0)));
    CHECK(needs_fd == std::vector<int>{0});

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

TEST_CASE("The new-best breakdown names indexed PC_ERMSE and PC_FRMSE terms "
          "per dataset", "[pc_dataset][host]") {
    /* Pins: the journal's new-best "top:" line lists an indexed PC_ERMSE
     * or PC_FRMSE term per dataset, ERMSE[i] and FRMSE[i] (1-based, the
     * index of PC_ERMSE (i) and of the L(ERMSE)[i] column), and no
     * "_global" label; the global form keeps ERMSE_global and
     * FRMSE_global. Each part is shown, as every part of the line is, as
     * its contribution to the run's loss, the mean over the loss terms:
     * ERMSE[1] is L(ERMSE)[1] / 2 datasets / 2 terms (PC_ERMSE and
     * PC_FRMSE), its share of the average of the dataset losses only when
     * the indexed metric is the one loss term, and the printed parts (all
     * four are listed: the line keeps the five largest) add up to the loss.
     * Before the fix: the indexed line read "ERMSE_global=... FRMSE_global=..."
     * (the average of the dataset losses under the global label), so the
     * first SECTION fails; the second SECTION passes as well (a pin of the
     * global labels), and so does the sum of the parts in both (a pin of
     * the display scaling the docs describe). */
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", first = sd.dir / "first.db", second = sd.dir / "second.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(first, 0.07, -0.04);
    write_minimal_ta_dbfile(second, 0.13, -0.16);
    std::ofstream(val) << "DBFILE " << first.string() << "\nDBFILE " << second.string() << "\n";
    bool indexed = false;
    std::string lines;
    SECTION("indexed energy and forces") {
        indexed = true;
        lines = "PC_ERMSE (2) 20 3\nPC_ERMSE (1) 10 2\n"
                "PC_FRMSE (1) 0.1 2\nPC_FRMSE (2) 0.2 0.5\n";
    }
    SECTION("global energy and forces") {
        lines = "PC_ERMSE 10 2\nPC_FRMSE 0.1 2\n";
    }
    std::ofstream(hpo) << dataset_host_head << lines;
    auto ctx = make_ta_master_context(train, hpo, val);
    ctx.remove("FORCE"); ctx.add("FORCE", "true");
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
    auto& log = tadah::hpo::HPOLog::instance();
    log.open((sd.dir / "journal.log").string(), "test", sd.dir.string(), 1);
    log.write_loss_section(trg);
    host.prepare_best_publication();
    ParamVec p(1); p(0) = 0.5;
    const double total = host(p);
    host.flush_remaining();
    log.finish("TEST");
    REQUIRE(total > 0.0);
    const auto losses = last_dataset_row("loss.tadah");
    REQUIRE(losses.size() == 9); // iteration, slot, L(E)[1,2,avg], L(F)[1,2,avg], total

    const auto journal = read_dataset_file(sd.dir / "journal.log");
    std::istringstream in(journal);
    std::string top;
    for (std::string line; std::getline(in, line);)
        if (line.find("top: ") != std::string::npos) { top = line; break; }
    INFO(journal);
    REQUIRE_FALSE(top.empty());
    if (indexed) {
        for (const char* label : {"ERMSE[1]=", "ERMSE[2]=", "FRMSE[1]=", "FRMSE[2]="}) {
            INFO(label);
            CHECK(top.find(label) != std::string::npos);
        }
        CHECK(top.find("_global") == std::string::npos);
        /* The printed value of ERMSE[1] (%.2e) is its share of the mean
         * loss: L(ERMSE)[1] / 2 datasets / 2 terms. */
        std::smatch m;
        REQUIRE(std::regex_search(top, m, std::regex(R"(ERMSE\[1\]=([0-9.eE+-]+)\()")));
        CHECK(std::stod(m[1]) == Approx(losses[2] / 2 / 2).epsilon(0.01));
    } else {
        CHECK(top.find("ERMSE_global=") != std::string::npos);
        CHECK(top.find("FRMSE_global=") != std::string::npos);
        CHECK(top.find("ERMSE[1]=") == std::string::npos);
        CHECK(top.find("FRMSE[1]=") == std::string::npos);
    }
    /* The printed parts (%.2e) add up to the run's loss. */
    const std::regex part(R"(([A-Za-z_]+(\[[0-9]+\])?)=([0-9.eE+-]+)\()");
    double parts = 0.0;
    std::size_t n_parts = 0;
    for (auto it = std::sregex_iterator(top.begin(), top.end(), part);
         it != std::sregex_iterator(); ++it, ++n_parts)
        parts += std::stod((*it)[3]);
    CHECK(n_parts == (indexed ? 4u : 2u));
    CHECK(parts == Approx(total).epsilon(0.01));
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

/* Pins: a --validation file whose DBFILE lists one dataset twice -- the
 * same path on two lines, or the path and another spelling of it
 * ("<dir>/./first.db") after a second dataset -- makes the HPO_Host
 * constructor throw a message that names the validation file, says
 * "DBFILE lists one dataset twice", names both entries ("entry 1 (<path>)
 * and entry 2 (<path>)"; "entry 1 (...) and entry 3 (<dir>/./first.db)")
 * and points to the indexed PC_ERMSE (i) / PC_FRMSE (i) lines; the
 * validation files of two datasets the other host cases construct pass.
 * Before the fix: the constructor did not throw, and every validation
 * metric took the duplicated dataset twice into its average over the
 * datasets. */
TEST_CASE("The host refuses a validation DBFILE that lists one dataset twice",
          "[pc_dataset][host][validation_file]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", first = sd.dir / "first.db",
               second = sd.dir / "second.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(first, 0.07);
    write_minimal_ta_dbfile(second, 0.13);
    const std::string respelt = (sd.dir / "." / "first.db").string();
    std::string lines, named;
    SECTION("the same path on two lines") {
        lines = "DBFILE " + first.string() + "\nDBFILE " + first.string() + "\n";
        named = "entry 1 (" + first.string() + ") and entry 2 (" +
                first.string() + ")";
    }
    SECTION("another spelling of the path after a second dataset") {
        lines = "DBFILE " + first.string() + "\nDBFILE " + second.string() +
                "\nDBFILE " + respelt + "\n";
        named = "entry 1 (" + first.string() + ") and entry 3 (" + respelt +
                ")";
    }
    std::ofstream(val) << lines;
    std::ofstream(hpo) << dataset_host_head << "PC_ERMSE 0 1\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    CHECK_THROWS_WITH(Host(ctx, trg, val.string()),
                      Catch::Matchers::Contains(val.string()) &&
                          Catch::Matchers::Contains(
                              "DBFILE lists one dataset twice") &&
                          Catch::Matchers::Contains(named) &&
                          Catch::Matchers::Contains("PC_ERMSE (i)"));
}

namespace {

/* The opening and end lines of the training record of a potential file.
 * The opening line said "(not read by tadah)" before it was corrected:
 * refit --edit reads the block, to copy it. */
const std::string kRecordOpen =
    "# --- training record (no key is read from it) ---";
const std::string kRecordEnd = "# --- end of training record ---";

/* The training record of the potential text @p text: its lines without
 * their leading "# " -- those of the keys (lines), then the fit balance
 * table, from its "Fit balance:" line on (balance), which best_pot.tadah
 * ends its record with and the DUMP and BDUMP archives do not have --,
 * whether its opening and end lines were found, and the indices of the end
 * line and of the first key line after it. */
struct PotRecord {
    bool opened = false, closed = false;
    std::size_t end_at = 0, first_key = 0;
    std::vector<std::string> lines;
    std::vector<std::string> balance;
};
PotRecord pot_record(const std::string& text) {
    PotRecord r;
    std::istringstream in(text);
    std::size_t n = 0;
    for (std::string l; std::getline(in, l); ++n) {
        if (!r.opened) {
            r.opened = l == kRecordOpen;
            continue;
        }
        if (r.closed) {
            std::istringstream ls(l);
            std::string key;
            if ((ls >> key) && key[0] != '#') { r.first_key = n; break; }
            continue;
        }
        if (l == kRecordEnd) {
            r.closed = true;
            r.end_at = n;
            continue;
        }
        const std::string line = l.rfind("# ", 0) == 0 ? l.substr(2) : l;
        if (!r.balance.empty() || line.rfind("Fit balance:", 0) == 0)
            r.balance.push_back(line);
        else
            r.lines.push_back(line);
    }
    return r;
}

/* Whether @p balance is a fit balance table of one quantity in one
 * dataset: its first line "Fit balance: LAMBDA ...", the line of its error
 * scale, the column header, the row of the quantity over every dataset and
 * the prior's row (MLIP fit_balance_lines). */
bool one_row_balance_table(const std::vector<std::string>& balance) {
    return balance.size() == 5 &&
           balance[0].rfind("Fit balance: LAMBDA ", 0) == 0 &&
           balance[1].rfind("  error scales: ", 0) == 0 &&
           balance[2].rfind("  quantity  dataset", 0) == 0 &&
           balance[3].find(" all ") != std::string::npos &&
           balance[4].rfind("  prior", 0) == 0;
}

/* The tokens of each key line of the potential text @p text (comment
 * lines skipped; not the parser's reading). */
std::map<std::string, std::vector<std::string>> key_lines(const std::string& text) {
    std::map<std::string, std::vector<std::string>> lines;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) {
        std::istringstream ls(l);
        std::string key;
        if (!(ls >> key) || key[0] == '#') continue;
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) tok.push_back(t);
        lines[key] = tok;
    }
    return lines;
}

}  // namespace

/* Pins: the potentials one evaluation writes -- best_pot.tadah, the BDUMP
 * archive (best_pots/) and the DUMP archive (pots/), one file each -- have
 * no key line of an input of the run: no HPOTARGET and no VALIDATION line
 * (the -c context has both, as `tadah hpo --hpotarget ... --validation
 * ...` puts them there), no VERBOSE (0 in the context, as -v gives it), no
 * AUDIT or HEALTH_LOG ("off" in the context) and no DBFILE, for MODEL
 * M_BLR BF_Linear and M_KRR Kern_Linear; each keeps the model's settings
 * and fit (its MODEL, LAMBDA, the D2_BP TYPE2B line and two WEIGHTS); and
 * each has, as the comment block right above its first key line, the
 * training record of the run: "DBFILE <train.db>", "VALIDATION
 * <config.val>", "HPOTARGET <config.hpo>", "VERBOSE 0", "AUDIT off",
 * "HEALTH_LOG off" (the context's values, in that order; the HPOTargets
 * here is built without the KeyProvenance of a command line, so every
 * stored key counts), and best_pot.tadah's record ends with the balance
 * table of the evaluation's fit (its first line, the energies' error
 * scale, the column header, the energies' row and the prior's row), which
 * the archives do not have. The
 * DATA section of best_pot.tadah's header gives the
 * training dataset's statistics without its path: the one line of each
 * file that names train.db is the record's DBFILE line.
 * Before the fix: no file had a training record (config.hpo and
 * config.val appeared nowhere in it), and the DATA section's training line
 * ended with the path of train.db.
 * Before the training record: the case was
 * named "The potentials of an evaluation record no input of the run", and
 * it checked that the paths of config.hpo and config.val appear nowhere in
 * the files. Before the potential-content fix: each of the three files had
 * the key lines "HPOTARGET <config.hpo>", "VALIDATION <config.val>" and
 * "VERBOSE 0".
 * Before the opening line was corrected: it read
 * "# --- training record (not read by tadah) ---": REQUIRE(opened) stops
 * the case. */
TEST_CASE("The potentials of an evaluation keep the inputs of the run in "
          "their training record, not as key lines",
          "[pc_dataset][host][potential_content][training_record]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << vdb.string() << "\n";
    std::vector<std::string> model;
    SECTION("M_BLR BF_Linear") { model = {"M_BLR", "BF_Linear"}; }
    SECTION("M_KRR Kern_Linear") { model = {"M_KRR", "Kern_Linear", "1"}; }
    CAPTURE(model);
    std::ofstream(hpo) << dataset_host_head << "PC_ERMSE 0 1\n"
                       << "DUMP 1 " << (sd.dir / "pots").string() << "\n"
                       << "BDUMP 1 " << (sd.dir / "best_pots").string()
                       << "\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    ctx.remove("MODEL");
    for (const std::string& m : model) ctx.add("MODEL", m);
    ctx.add("AUDIT", "off");
    ctx.add("HEALTH_LOG", "off");
    REQUIRE(ctx.has_stored("HPOTARGET"));
    REQUIRE(ctx.has_stored("VALIDATION"));
    REQUIRE(ctx.has_stored("VERBOSE"));
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    {
        Host host(ctx, trg, val.string());
        auto& log = tadah::hpo::HPOLog::instance();
        log.open((sd.dir / "journal.log").string(), "test",
                 sd.dir.string(), 1);
        log.write_loss_section(trg);
        /* The run creates the DUMP and BDUMP directories before its
         * first evaluation (hpo_run()), not its evaluation host. */
        trg.create_archive_directories();
        host.prepare_best_publication();
        ParamVec p(1); p(0) = 0.5;
        const double total = host(p);
        host.flush_remaining();
        log.finish("TEST");
        REQUIRE(std::isfinite(total));
    }
    std::vector<fs::path> files = {sd.dir / "best_pot.tadah"};
    for (const char* dir : {"best_pots", "pots"}) {
        std::vector<fs::path> in_dir;
        for (const auto& e : fs::directory_iterator(sd.dir / dir))
            in_dir.push_back(e.path());
        INFO("directory " << dir);
        REQUIRE(in_dir.size() == 1u);
        files.push_back(in_dir.front());
    }
    for (const fs::path& f : files) {
        const std::string text = read_dataset_file(f);
        INFO(f.string() << ":\n" << text);
        REQUIRE_FALSE(text.empty());
        /* The tokens of each key line (not the parser's reading: it refuses
         * the DM_Dummy TYPEMB line of this context). */
        std::map<std::string, std::vector<std::string>> lines;
        {
            std::istringstream in(text);
            for (std::string l; std::getline(in, l);) {
                std::istringstream ls(l);
                std::string key;
                if (!(ls >> key) || key[0] == '#') continue;
                std::vector<std::string> tok;
                for (std::string t; ls >> t;) tok.push_back(t);
                lines[key] = tok;
            }
        }
        for (const char* key : {"HPOTARGET", "VALIDATION", "VERBOSE", "AUDIT",
                                "HEALTH_LOG", "DBFILE"}) {
            INFO("key " << key);
            CHECK(lines.count(key) == 0u);
        }
        /* The training record: the lines between its opening and end
         * lines, right above the first key line; best_pot.tadah's ends with
         * the balance table of the evaluation's fit, the archives' do
         * not. */
        const PotRecord r = pot_record(text);
        REQUIRE(r.opened);
        REQUIRE(r.closed);
        CHECK(r.first_key == r.end_at + 1);
        CHECK(r.lines == std::vector<std::string>{
                             "DBFILE " + train.string(),
                             "VALIDATION " + val.string(),
                             "HPOTARGET " + hpo.string(), "VERBOSE 0",
                             "AUDIT off", "HEALTH_LOG off"});
        if (f == files.front())
            CHECK(one_row_balance_table(r.balance));
        else
            CHECK(r.balance.empty());
        /* train.db is named by the record only (best_pot.tadah's DATA
         * section gives its statistics without the path). */
        {
            std::istringstream in(text);
            std::size_t naming = 0;
            for (std::string l; std::getline(in, l);) {
                if (l.find(train.string()) == std::string::npos) continue;
                ++naming;
                INFO("line naming train.db: " << l);
                CHECK(l.rfind("# DBFILE ", 0) == 0);
            }
            CHECK(naming == 1u);
        }
        REQUIRE(lines.count("MODEL") == 1u);
        CHECK(lines.at("MODEL") == model);
        CHECK(lines.count("LAMBDA") == 1u);
        REQUIRE(lines.count("TYPE2B") == 1u);
        CHECK(lines.at("TYPE2B").front() == "D2_BP");
        REQUIRE(lines.count("WEIGHTS") == 1u);
        CHECK(lines.at("WEIGHTS").size() == 2u);
    }
}

/* Pins: the potentials one
 * evaluation writes -- best_pot.tadah, the BDUMP archive and the DUMP
 * archive -- keep the keys that prepared the training data in their
 * training record, not as key lines: with EFILTER -1 1, FFILTER 100,
 * ESHIFT_DBATOM true, ZERO_COM_FORCE true and WDBFILE 1 in the -c context
 * (they keep the one training structure: E/N is -0.00025 eV), each file has
 * no key line of the five, the record "DBFILE <train.db>", "EFILTER -1 1",
 * "FFILTER 100", "ESHIFT_DBATOM true", "ZERO_COM_FORCE true", "WDBFILE 1",
 * "VALIDATION <config.val>", "HPOTARGET <config.hpo>", "VERBOSE 0" (the
 * context's values, the order of the record; best_pot.tadah's then ends
 * with the balance table of the evaluation's fit, the archives' not), and
 * the ESHIFT key line that
 * ESHIFT_DBATOM resolved on the training data (one value, not 0), the
 * model's energy reference. (The retired WDBFILE_AUTO and EWEIGHT_TEMP,
 * which this case had too, are refused by a fit.)
 * Before the fix: they were key lines of every file, and the record
 * listed DBFILE, VALIDATION, HPOTARGET and VERBOSE alone; the opening line
 * read "# --- training record (not read by tadah) ---", so REQUIRE(opened)
 * stops the case. */
TEST_CASE("The potentials of an evaluation keep the keys that prepared the "
          "training data in their training record",
          "[pc_dataset][host][potential_content][training_record]"
          "[dataset_keys]") {
    DatasetHostGuard guard;
    ScratchDir sd;
    const auto train = sd.dir / "train.db", vdb = sd.dir / "val.db";
    const auto val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train, 0);
    write_minimal_ta_dbfile(vdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << vdb.string() << "\n";
    std::ofstream(hpo) << dataset_host_head << "PC_ERMSE 0 1\n"
                       << "DUMP 1 " << (sd.dir / "pots").string() << "\n"
                       << "BDUMP 1 " << (sd.dir / "best_pots").string()
                       << "\n";
    auto ctx = make_ta_master_context(train, hpo, val);
    const std::vector<std::pair<std::string, std::string>> dataset_keys = {
        {"EFILTER", "-1 1"},        {"FFILTER", "100"},
        {"ESHIFT_DBATOM", "true"},  {"ZERO_COM_FORCE", "true"},
        {"WDBFILE", "1"}};
    for (const auto& [key, values] : dataset_keys) {
        std::istringstream vs(values);
        for (std::string v; vs >> v;) ctx.add(key, v);
    }
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    {
        Host host(ctx, trg, val.string());
        auto& log = tadah::hpo::HPOLog::instance();
        log.open((sd.dir / "journal.log").string(), "test",
                 sd.dir.string(), 1);
        log.write_loss_section(trg);
        /* The run creates the DUMP and BDUMP directories before its
         * first evaluation (hpo_run()), not its evaluation host. */
        trg.create_archive_directories();
        host.prepare_best_publication();
        ParamVec p(1); p(0) = 0.5;
        const double total = host(p);
        host.flush_remaining();
        log.finish("TEST");
        REQUIRE(std::isfinite(total));
    }
    std::vector<fs::path> files = {sd.dir / "best_pot.tadah"};
    for (const char* dir : {"best_pots", "pots"}) {
        std::vector<fs::path> in_dir;
        for (const auto& e : fs::directory_iterator(sd.dir / dir))
            in_dir.push_back(e.path());
        INFO("directory " << dir);
        REQUIRE(in_dir.size() == 1u);
        files.push_back(in_dir.front());
    }
    for (const fs::path& f : files) {
        const std::string text = read_dataset_file(f);
        INFO(f.string() << ":\n" << text);
        REQUIRE_FALSE(text.empty());
        const auto lines = key_lines(text);
        for (const auto& kv : dataset_keys) {
            INFO("key " << kv.first);
            CHECK(lines.count(kv.first) == 0u);
        }
        const PotRecord r = pot_record(text);
        REQUIRE(r.opened);
        REQUIRE(r.closed);
        CHECK(r.first_key == r.end_at + 1);
        CHECK(r.lines == std::vector<std::string>{
                             "DBFILE " + train.string(), "EFILTER -1 1",
                             "FFILTER 100", "ESHIFT_DBATOM true",
                             "ZERO_COM_FORCE true", "WDBFILE 1",
                             "VALIDATION " + val.string(),
                             "HPOTARGET " + hpo.string(), "VERBOSE 0"});
        if (f == files.front())
            CHECK(one_row_balance_table(r.balance));
        else
            CHECK(r.balance.empty());
        REQUIRE(lines.count("ESHIFT") == 1u);
        REQUIRE(lines.at("ESHIFT").size() == 1u);
        CHECK(std::stod(lines.at("ESHIFT").front()) != 0.0);
    }
}

/* FORCE and STRESS decide only what training fits (the -c configuration,
 * or -F/-S); validation predicts forces whenever PC_FRMSE is set and
 * stresses whenever PC_SRMSE_GPA is set. A FORCE or STRESS key in the
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
 * second run in one process started its relaxations where the first run's
 * last evaluation ended. */
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

/* ---------------------------------------------------------------------
 * The titles of the metric columns.
 *
 * Every title of loss.tadah, outvar.tadah and stats.tadah (and their
 * best_ files) is written whole, with at least one blank before it, at
 * every DIGITS and with the default (no DIGITS line), so that splitting a
 * header on blanks gives one title per value. The longest of them,
 * L(SRMSE_GPA)[avg], has 17 characters.
 * ------------------------------------------------------------------- */
namespace {

/* The fields of the first line of @p file that is not a header or a
 * comment. */
std::vector<std::string> first_row_fields(const std::string& file)
{
    std::ifstream in(file);
    std::vector<std::string> fields;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        for (std::string f; is >> f;) fields.push_back(f);
        break;
    }
    return fields;
}

/* A target of PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA, with a DIGITS line
 * unless @p digits is 0. */
void write_metric_target(const fs::path& hpo, int digits)
{
    std::ofstream os(hpo);
    os << "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
       << "LOSS L2\nFAILSCORE 1e8\n";
    if (digits > 0) os << "DIGITS " << digits << "\n";
    os << "PC_ERMSE 0 1\nPC_FRMSE 0 1\nPC_SRMSE_GPA 0 1\n"
       << "OPTIM CGRID2B (1) 0.0 4.0\n";
}

}  // namespace

/* Pins: with PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA and two validation
 * datasets, the headers split on blanks into the titles L(ERMSE)[1] ...
 * L(SRMSE_GPA)[avg], TOTAL_LOSS/N (loss), ERMSE, FRMSE, SRMSE_GPA
 * (outvar) and ERMSE[1] ... SRMSE_GPA[avg] (stats), at DIGITS 1 to 15 and
 * the default; with the default, the first row of each file has one value
 * per title.
 * Before the fix: at a column width of 15 (DIGITS 6 or less) the loss
 * titles L(SRMSE_GPA)[1], [2] and [avg] ran together (9 titles for 12
 * values) and stats.tadah cut them to SRMSE_GPA[*, SRMSE_GPA~2 and
 * SRMSE_GPA~3. */
TEST_CASE("Every metric column title is whole and keeps a blank before it "
          "at every DIGITS",
          "[HPO][host][column_titles]")
{
    tadah::core::PeriodicTable::initialize();
    DatasetHostGuard guard;
    const std::vector<std::string> metrics = {"ERMSE", "FRMSE", "SRMSE_GPA"};
    std::vector<std::string> want_loss = {"#ITERATION", "SLOT"};
    std::vector<std::string> want_stats = {"#ITERATION", "SLOT"};
    std::vector<std::string> want_outvar = {"#ITERATION", "IT-TIME/ms",
                                            "SLOT"};
    for (const std::string& m : metrics) {
        for (const char* i : {"[1]", "[2]", "[avg]"}) {
            want_loss.push_back("L(" + m + ")" + i);
            want_stats.push_back(m + i);
        }
        want_outvar.push_back(m);
    }
    want_loss.push_back("TOTAL_LOSS/N");

    for (int digits = 0; digits <= 15; ++digits) {
        INFO("DIGITS " << (digits == 0 ? std::string("default")
                                       : std::to_string(digits)));
        ScratchDir sd;
        const auto train = sd.dir / "train.db", hpo = sd.dir / "config.hpo";
        const auto val_cfg = sd.dir / "config.val";
        const auto val1 = sd.dir / "val1.db", val2 = sd.dir / "val2.db";
        write_minimal_ta_dbfile(train, 0);
        write_minimal_ta_dbfile(val1, 0.07);
        write_minimal_ta_dbfile(val2, 0.05, -0.002);
        std::ofstream(val_cfg) << "DBFILE " << val1.string() << "\n"
                               << "DBFILE " << val2.string() << "\n";
        write_metric_target(hpo, digits);
        auto ctx = make_ta_master_context(train, hpo, val_cfg);
        tadah::hpo::HPOTargets trg(ctx, hpo.string());
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.val_size() == 2u);

        CHECK(header_columns(trg.lossFile) == want_loss);
        CHECK(header_columns(trg.bestLossFile) == want_loss);
        CHECK(header_columns(trg.statsFile) == want_stats);
        CHECK(header_columns(trg.bestStatsFile) == want_stats);
        CHECK(header_columns(trg.outvarFile) == want_outvar);
        CHECK(header_columns(trg.bestOutvarFile) == want_outvar);
        if (digits != 0) continue;

        host.prepare_best_publication();
        ParamVec p(1);
        p(0) = 0.5;
        REQUIRE_NOTHROW(host(p));
        host.flush_remaining();
        for (const std::string* file :
             {&trg.lossFile, &trg.statsFile, &trg.outvarFile}) {
            INFO(*file << ": " << first_line(*file));
            CHECK(first_row_fields(*file).size() ==
                  header_columns(*file).size());
        }
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
 * geometry). The model is the shared Ta model of the tests
 * (tests/data/ta/common/config.tadah) with the settings the analytical
 * gradient needs (OALGO 4 and a ridge LAMBDA).
 * ------------------------------------------------------------------- */
namespace {

const fs::path& ta_common_dir()
{
    static const fs::path dir = fs::path(TADAH_HPO_TA_DATA_DIR) / "common";
    return dir;
}

tadah::core::Context ta_context()
{
    const fs::path& common = ta_common_dir();
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
                       << (ta_common_dir() / "data" / "elastic.tadahdb").string()
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
    auto ctx = ta_context();
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
 * structure and the atoms, and
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

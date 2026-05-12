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
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

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
void write_minimal_ta_dbfile(const fs::path& path, double offset_x) {
    std::ofstream os(path);
    os << "test_bcc_ta\n"
       << "1.0 1.0 1.0\n"
       << "-0.001 0.0\n"
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

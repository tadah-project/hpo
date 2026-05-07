#include "catch2/catch.hpp"

#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/core/context.h>

#include <chrono>
#include <filesystem>
#include <fstream>

/* HPO integration tests
 *
 * Full regression coverage of HPO_Host requires actual training data
 * (StructureDB files) which are not bundled with the source tree.
 * The key invariants validated by unit tests in MODELS/tests/test_reinit.cpp
 * already cover the bugs found during the reinit() implementation:
 *
 *   - D2_mJoin::reinit() / DM_mJoin::reinit() outer fcut updated
 *     when RCUT changes (Bug 2 — the tadah-predict mismatch).
 *   - M_Core::reset_trained() correctly clears the trained flag (Bug 3).
 *
 * What belongs here when data is available
 * -----------------------------------------
 * 1. HPO_Host self-consistency test
 *    Run 5 iterations of HPO on a tiny dataset (e.g. 3 structures).
 *    For each best iteration recorded in best_outvar, load the
 *    corresponding best_pot from BestPots/ and verify that
 *    tadah-predict gives the same ERSQ as HPO reported.
 *    This is the integration-level regression for the RCUT prefilter bug.
 *
 * 2. flush_remaining() test
 *    Start HPO, interrupt at iteration N by throwing from the objective
 *    function, then call flush_remaining() and verify that all N results
 *    were written to the output files (no truncated output).
 *
 * 3. Standalone optimiser parser coverage
 *    Keep a self-contained config for DLIB:BFGS so the parser/validator
 *    contract stays locked down independently of larger integration data.
 */

namespace {

struct ScopedTempFile {
  std::filesystem::path path;

  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_hpo_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
    os.close();
    REQUIRE(os.good());
  }

  ~ScopedTempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

}  // namespace

TEST_CASE("HPO placeholder — see comments above for planned tests", "[HPO]")
{
    // No self-contained integration tests yet.
    // All unit-level regression tests are in MODELS/tests/test_reinit.cpp.
    SUCCEED();
}

TEST_CASE("HPO parses standalone DLIB BFGS optimizer blocks",
          "[HPO][optim][dlib][bfgs]")
{
    ScopedTempFile input(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 25
PARAM fd_step 1e-6
PARAM tol_gradinf 1e-5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.lib == "DLIB");
    CHECK(trg.spec.algo == "BFGS");
    REQUIRE(trg.spec.stop.maxeval.has_value());
    CHECK(*trg.spec.stop.maxeval == 25u);
    REQUIRE(trg.spec.param.count("fd_step") == 1);
    CHECK(trg.spec.param.at("fd_step") == Approx(1.0e-6));
    REQUIRE(trg.spec.param.count("tol_gradinf") == 1);
    CHECK(trg.spec.param.at("tol_gradinf") == Approx(1.0e-5));
}

TEST_CASE("DLIB BFGS validator rejects INNER blocks",
          "[HPO][optim][dlib][bfgs][validation]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.low = {-1.0, -2.0};
    sp.high = {1.0, 2.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.0, 0.5}};
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_BOBYQA";

    CHECK_THROWS_WITH(tadah::hpo::detail::validate_dlib_bfgs_spec(sp),
                      Catch::Contains("standalone optimiser") &&
                      Catch::Contains("INNER"));
}

TEST_CASE("DLIB BFGS validator accepts standalone bounded specs",
          "[HPO][optim][dlib][bfgs][validation]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.low = {-1.0, -2.0};
    sp.high = {1.0, 2.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.0, 0.5}};
    sp.stop.maxeval = 10;
    sp.param["fd_step"] = 1.0e-6;

    CHECK_NOTHROW(tadah::hpo::detail::validate_dlib_bfgs_spec(sp));
}

TEST_CASE("HPO parses INIT block",
          "[HPO][optim][init_strategy]")
{
    ScopedTempFile input(R"(OPTIMIZER
INIT
  STRATEGY CONFIG
  STRATEGY RANDOM 4
  K_TOTAL 5
  SEED 12345
  INCLUDE_CONFIG false
  WARM_ON_MISMATCH skip
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 25
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    const auto& isp = trg.spec.init_strategy_spec;
    REQUIRE(isp.chain.size() == 2);
    CHECK(isp.chain[0].method == "CONFIG");
    CHECK(isp.chain[1].method == "RANDOM");
    REQUIRE(isp.chain[1].args.size() == 1);
    CHECK(isp.chain[1].args[0] == "4");
    REQUIRE(isp.k_total.has_value());
    CHECK(*isp.k_total == 5u);
    REQUIRE(isp.seed.has_value());
    CHECK(*isp.seed == 12345u);
    CHECK(isp.include_config == false);
    CHECK(isp.warm_on_mismatch == "skip");
}

TEST_CASE("HPO INCLUDE_CONFIG accepts a range of boolean spellings",
          "[HPO][optim][init_strategy]")
{
    auto make_cfg = [](const std::string& val) {
        return std::string("OPTIMIZER\nINIT\n  INCLUDE_CONFIG ") + val
            + "\nENDINIT\nSTRATEGY\n  TYPE PLAIN\n  LIB DLIB\n  ALGO BFGS\nENDSTRATEGY\nENDOPTIMIZER\n";
    };
    for (const std::string val : {"true","TRUE","1","yes","YES","on","ON"}) {
        ScopedTempFile input(make_cfg(val));
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        CHECK(trg.spec.init_strategy_spec.include_config == true);
    }
    for (const std::string val : {"false","FALSE","0","no","NO","off","OFF"}) {
        ScopedTempFile input(make_cfg(val));
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        CHECK(trg.spec.init_strategy_spec.include_config == false);
    }
}

TEST_CASE("HPO WARM_ON_MISMATCH rejects invalid values",
          "[HPO][optim][init_strategy]")
{
    ScopedTempFile bad(R"(OPTIMIZER
INIT
  WARM_ON_MISMATCH bogus
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, bad.path.string()));
}

TEST_CASE("HPO unknown INIT.STRATEGY method is rejected at resolve time",
          "[HPO][optim][init_strategy]")
{
    ScopedTempFile input(R"(OPTIMIZER
INIT
  STRATEGY BOGUS
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
}

TEST_CASE("HPO ConfigInitStrategy reproduces empty-spec defaults",
          "[HPO][optim][init_strategy]")
{
    /* No OPTIM lines → bounds are empty → ConfigInitStrategy returns
     * exactly one zero-dimensional point. */
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    REQUIRE(trg.spec.optim_init.size() == 1);
    CHECK(trg.spec.optim_init.front().empty());
}

TEST_CASE("HPO parser accepts ROOT-level keys after ENDOPTIMIZER",
          "[HPO][parser][regression]")
{
    /* Regression for the post-STRATEGY-rewrite bug where ROOT-level keys
     * (LOSS, FAILSCORE, OUTPUT, BOUTPUT, DIGITS, OPTIM, PC_*, BDUMP)
     * threw `unknown key '<X>' in this block`. The parser's fall-through
     * guard had become `if (block == OUTER) { fall through }` (matching
     * only the OPTIMIZER scope), so once `ENDOPTIMIZER` set
     * `block = ROOT`, every subsequent ROOT-level key hit the catch-all
     * throw before reaching its handler. Real-world configs (every
     * config.hpo under examples/ and the BASINS workdir) failed to
     * parse with `Error on line N: unknown key 'LOSS' in this block`.
     *
     * This test exercises the most common ROOT-level keys to ensure
     * they all parse without throwing. */
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 6
ENDSTRATEGY
ENDOPTIMIZER

LOSS L2
FAILSCORE 1e16
OUTPUT 1
BOUTPUT 1
DIGITS 6

PC_ERMSE 0 0.0001
PC_FRMSE 0 1
)");
    tadah::core::Context ctx;
    /* Pre-fix: this throws.  Post-fix: parses cleanly. */
    CHECK_NOTHROW(tadah::hpo::HPOTargets(ctx, input.path.string()));
}

TEST_CASE("HPO parser still rejects unknown keys inside OPTIMIZER blocks",
          "[HPO][parser][regression]")
{
    /* Companion check: keys that genuinely have no home should still
     * error at parse time, not silently fall through to the ROOT
     * handler (where they would also fail, but with a less specific
     * "unknown key" message). */
    SECTION("unknown key inside STRATEGY") {
        ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  BOGUS_KEY 42
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        CHECK_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
    SECTION("unknown key inside INIT") {
        ScopedTempFile input(R"(OPTIMIZER
INIT
  STRATEGY RANDOM 1
  BOGUS_KEY 42
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        CHECK_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
    SECTION("unknown key inside NOISE") {
        ScopedTempFile input(R"(OPTIMIZER
NOISE
  BOGUS_KEY 42
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        CHECK_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
}

TEST_CASE("LammpsSim validates dynamic varloss and outvar names built from --invar values",
          "[HPO][lammps_sim][dynamic_vars]")
{
    ScopedTempFile script(R"(
variable P_GPa equal ${P}
variable P_is_zero equal abs(v_P_GPa) < 1.0e-12
variable P_int equal round(v_P_GPa)
variable P_is_integer equal abs(v_P_GPa-v_P_int) < 1.0e-12
variable P_mGPa equal round(v_P_GPa*1000.0)
variable P_int_str format P_int %.0f
variable P_mGPa_str format P_mGPa %.0f

if "${P_is_zero} != 0" then "variable quantity string E" else "variable quantity string H"
if "${P_is_integer} != 0" then "variable Ptag string ${P_int_str}" else "variable Ptag string ${P_mGPa_str}mGPa"

variable a0 equal $(lx)
clear

if "${phase} == fcc" then "jump SELF phase_fcc"
if "${phase} == hcp" then "jump SELF phase_hcp"

label phase_fcc
variable dummy equal v_a0
jump SELF finalize

label phase_hcp
variable dummy equal v_a0
jump SELF finalize

label finalize
variable d${quantity}_${phase}_bcc_${Ptag} equal v_dummy-v_a0
variable pen_${phase}_P${Ptag} equal v_d${quantity}_${phase}_bcc_${Ptag}
)");

    SECTION("integer pressure preserves legacy P0 names") {
        std::string cmd = "--script " + script.path.string() +
                          " --varloss pen_fcc_P0 0.0 100 --invar P 0 --invar phase fcc --outvar dE_fcc_bcc_0";
        CHECK_NOTHROW(tadah::hpo::LammpsSim(cmd, "1", 1.0e16, tadah::LossFunction{}));
    }

    SECTION("non-integer pressure resolves milli-GPa dynamic names") {
        std::string cmd = "--script " + script.path.string() +
                          " --varloss pen_fcc_P12500mGPa 0.0 100 --invar P 12.5 --invar phase fcc --outvar dH_fcc_bcc_12500mGPa";
        CHECK_NOTHROW(tadah::hpo::LammpsSim(cmd, "1", 1.0e16, tadah::LossFunction{}));
    }
}

TEST_CASE("LammpsSim still rejects dynamically-resolved variables with wrong style",
          "[HPO][lammps_sim][dynamic_vars][validation]")
{
    ScopedTempFile script(R"(
variable P_GPa equal ${P}
variable P_is_zero equal abs(v_P_GPa) < 1.0e-12
variable P_int equal round(v_P_GPa)
variable P_is_integer equal abs(v_P_GPa-v_P_int) < 1.0e-12
variable P_mGPa equal round(v_P_GPa*1000.0)
variable P_int_str format P_int %.0f
variable P_mGPa_str format P_mGPa %.0f

if "${P_is_zero} != 0" then "variable quantity string E" else "variable quantity string H"
if "${P_is_integer} != 0" then "variable Ptag string ${P_int_str}" else "variable Ptag string ${P_mGPa_str}mGPa"

variable d${quantity}_${phase}_bcc_${Ptag} equal 1.23
variable pen_${phase}_P${Ptag} string nope
)");

    std::string cmd = "--script " + script.path.string() +
                      " --varloss pen_fcc_P0 0.0 100 --invar P 0 --invar phase fcc --outvar dE_fcc_bcc_0";

    CHECK_THROWS_WITH(tadah::hpo::LammpsSim(cmd, "1", 1.0e16, tadah::LossFunction{}),
                      Catch::Contains("pen_fcc_P0") && Catch::Contains("style 'string'"));
}

TEST_CASE("LammpsSim validation does not eagerly evaluate formula substitutions",
          "[HPO][lammps_sim][validation]")
{
    ScopedTempFile script(R"(
clear
units metal
dimension 3
boundary f f f
atom_style atomic

region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0 0 0

compute eng all pe/atom
compute eatoms all reduce sum c_eng
run 0

variable tmp equal c_eatoms
variable Eiso equal ${tmp}

clear
units metal
dimension 3
boundary p p p
atom_style atomic

lattice bcc ${hot_a0}
region box block 0 4 0 4 0 4 units lattice
create_box 1 box
create_atoms 1 region box

compute eng all pe/atom
compute eatoms all reduce sum c_eng

variable lxnow equal lx
variable a0 equal ${lxnow}/4.0
variable Ecoh equal 0.0
)");

    std::string cmd = "--script " + script.path.string() +
                      " --varloss Ecoh 8.10 800 --invar hot_a0 3.30";

    CHECK_NOTHROW(tadah::hpo::LammpsSim(cmd, "1", 1.0e16, tadah::LossFunction{}));
}

#include "catch2/catch.hpp"

#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/core/context.h>

#include <atomic>
#include <cmath>
#include <dlib/matrix.h>

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

/* =========================================================================
 * CERES adapter tests
 * ========================================================================= */

namespace {
tadah::hpo::OptimizerSpec make_minimal_ceres_spec(const std::string& algo)
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib  = "CERES";
    sp.algo = algo;
    sp.low  = {-1.0, -2.0};
    sp.high = { 1.0,  2.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.0, 0.5}};
    sp.stop.maxeval = 25;
    return sp;
}
}  // namespace

TEST_CASE("HPO parses standalone CERES LBFGS optimizer blocks",
          "[HPO][optim][ceres][lbfgs][parse]")
{
    ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO LBFGS
MAXEVAL 30
FTOL_REL 1e-7
XTOL_REL 1e-9
PARAM fd_step 1e-6
PARAM tol_gradinf 1e-8
PARAM max_lbfgs_rank 10
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.lib  == "CERES");
    CHECK(trg.spec.algo == "LBFGS");
    REQUIRE(trg.spec.stop.maxeval.has_value());
    CHECK(*trg.spec.stop.maxeval == 30u);
    CHECK(trg.spec.stop.ftol_rel == Approx(1e-7));
    CHECK(trg.spec.stop.xtol_rel == Approx(1e-9));
    CHECK(trg.spec.param.at("fd_step")        == Approx(1e-6));
    CHECK(trg.spec.param.at("tol_gradinf")    == Approx(1e-8));
    CHECK(trg.spec.param.at("max_lbfgs_rank") == Approx(10.0));
}

TEST_CASE("HPO parses CERES WOLFE/NLCG/STEEPEST_DESCENT",
          "[HPO][optim][ceres][parse]")
{
    for (const std::string algo : {"WOLFE", "NLCG", "STEEPEST_DESCENT"}) {
        ScopedTempFile input("OPTIMIZER\nLIB CERES\nALGO " + algo +
                             "\nMAXEVAL 5\nENDOPTIMIZER\n");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        CHECK(trg.spec.lib  == "CERES");
        CHECK(trg.spec.algo == algo);
    }
}

TEST_CASE("HPO parses GRAD_STRATEGY",
          "[HPO][optim][grad_strategy][parse]")
{
    /* default (unset) */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        CHECK_FALSE(trg.spec.grad_strategy.has_value());
    }
    /* NUMERICAL */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
GRAD_STRATEGY NUMERICAL
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        REQUIRE(trg.spec.grad_strategy.has_value());
        CHECK(*trg.spec.grad_strategy == tadah::hpo::GradStrategy::Numerical);
    }
    /* AUTODIFF */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO LBFGS
GRAD_STRATEGY AUTODIFF
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        REQUIRE(trg.spec.grad_strategy.has_value());
        CHECK(*trg.spec.grad_strategy == tadah::hpo::GradStrategy::Autodiff);
    }
    /* invalid */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
GRAD_STRATEGY BOGUS
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        CHECK_THROWS_WITH(tadah::hpo::HPOTargets(ctx, input.path.string()),
                          Catch::Contains("GRAD_STRATEGY"));
    }
}

TEST_CASE("CERES caps registry exposes the four GradientProblemSolver algos",
          "[HPO][optim][ceres][caps]")
{
    for (const std::string algo : {"LBFGS", "WOLFE", "NLCG", "STEEPEST_DESCENT"}) {
        const auto& caps = tadah::hpo::optimizer_caps("CERES", algo);
        CHECK(caps.is_gradient_based);
        CHECK_FALSE(caps.is_global);
        CHECK_FALSE(caps.is_population);
        CHECK_FALSE(caps.accepts_inner);
        CHECK(caps.supports_maxeval);
    }
}

TEST_CASE("CERES NLLS family is deferred and rejected at dispatch",
          "[HPO][optim][ceres][deferred]")
{
    /* Caps registry deliberately does NOT carry these names — confirm. */
    for (const std::string algo : {"LEVENBERG_MARQUARDT", "DOGLEG", "SUBSPACE_DOGLEG"}) {
        CHECK_THROWS_WITH(tadah::hpo::optimizer_caps("CERES", algo),
                          Catch::Contains("not registered"));
    }
    /* The early-reject helper used by run_inner emits the tailored
     * "use CERES:LBFGS" message. */
    for (const std::string algo : {"LEVENBERG_MARQUARDT", "DOGLEG", "SUBSPACE_DOGLEG"}) {
        auto sp = make_minimal_ceres_spec(algo);
        CHECK_THROWS_WITH(tadah::hpo::reject_deferred_ceres_nlls(sp),
                          Catch::Contains("deferred") &&
                          Catch::Contains("CERES:LBFGS"));
    }
    /* Non-deferred algos pass the helper untouched. */
    for (const std::string algo : {"LBFGS", "WOLFE", "NLCG", "STEEPEST_DESCENT"}) {
        auto sp = make_minimal_ceres_spec(algo);
        CHECK_NOTHROW(tadah::hpo::reject_deferred_ceres_nlls(sp));
    }
    /* Non-CERES libs are unaffected. */
    {
        tadah::hpo::OptimizerSpec sp;
        sp.lib  = "DLIB";
        sp.algo = "LEVENBERG_MARQUARDT";  // hypothetical typo
        CHECK_NOTHROW(tadah::hpo::reject_deferred_ceres_nlls(sp));
    }
}

TEST_CASE("CERES validators accept minimal bounded specs",
          "[HPO][optim][ceres][validate]")
{
    {
        auto sp = make_minimal_ceres_spec("LBFGS");
        CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp));
    }
    {
        auto sp = make_minimal_ceres_spec("WOLFE");
        CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_wolfe_spec(sp));
    }
    {
        auto sp = make_minimal_ceres_spec("NLCG");
        CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_nlcg_spec(sp));
    }
    {
        auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
        CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_steepest_spec(sp));
    }
}

TEST_CASE("CERES validators reject INNER blocks",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("LBFGS");
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_BOBYQA";
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp),
                      Catch::Contains("standalone optimiser") &&
                      Catch::Contains("INNER"));
}

TEST_CASE("CERES validators reject integer-valued parameters",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("WOLFE");
    sp.is_int = {true, false};
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_wolfe_spec(sp),
                      Catch::Contains("integer-valued"));
}

TEST_CASE("CERES validators reject FTOL_ABS / XTOL_ABS",
          "[HPO][optim][ceres][validate]")
{
    {
        auto sp = make_minimal_ceres_spec("LBFGS");
        sp.stop.ftol_abs = 1e-6;
        CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp),
                          Catch::Contains("FTOL_REL"));
    }
    {
        auto sp = make_minimal_ceres_spec("LBFGS");
        sp.stop.xtol_abs = 1e-8;
        CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp),
                          Catch::Contains("XTOL_REL"));
    }
}

TEST_CASE("CERES validators reject unsupported PARAMs",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("LBFGS");
    sp.param["bogus_param"] = 1.0;
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp),
                      Catch::Contains("does not support PARAM"));
}

TEST_CASE("CERES LBFGS validator rejects non-positive max_lbfgs_rank",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("LBFGS");
    sp.param["max_lbfgs_rank"] = 0.0;
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp),
                      Catch::Contains("max_lbfgs_rank"));
}

TEST_CASE("CERES NLCG validator rejects bad nlcg_type",
          "[HPO][optim][ceres][validate][nlcg]")
{
    auto sp = make_minimal_ceres_spec("NLCG");
    sp.param["nlcg_type"] = 5.0;
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_nlcg_spec(sp),
                      Catch::Contains("nlcg_type"));
}

/* =========================================================================
 * GradientProvider tests — exercise the unified gradient seam against a
 * stub host. These cover Ceres-NumericDiff parity vs. analytic, MAXEVAL
 * accounting, bad-eval propagation, and the AUTODIFF stub.
 * ========================================================================= */

namespace {
using DlibVec = dlib::matrix<double, 0, 1>;

/* Stub host: f(x, y) = sin(x) + y².
 * Gradient at (x, y) = (cos(x), 2y). */
struct StubHostSinXY2 {
    mutable std::atomic<unsigned long> call_count{0};
    double operator()(const DlibVec& x) const {
        call_count.fetch_add(1);
        return std::sin(x(0)) + x(1) * x(1);
    }
};

/* Stub host that is non-finite at one specific point — used to test
 * bad-eval propagation (gradient returns false). */
struct StubHostBadAtOrigin {
    double operator()(const DlibVec& x) const {
        if (std::abs(x(0)) < 1e-9 && std::abs(x(1)) < 1e-9)
            return std::numeric_limits<double>::quiet_NaN();
        return x(0) * x(0) + x(1) * x(1);
    }
};

tadah::hpo::OptimizerSpec make_provider_test_spec() {
    tadah::hpo::OptimizerSpec sp;
    sp.lib  = "CERES";
    sp.algo = "LBFGS";
    sp.low  = {-5.0, -5.0};
    sp.high = { 5.0,  5.0};
    sp.is_int = {false, false};
    sp.log_scaled = {false, false};   // sp.out is identity
    return sp;
}
}  // namespace

#ifdef TADAH_USE_CERES
TEST_CASE("GradientProvider Numerical matches analytic on f=sin(x)+y^2",
          "[HPO][grad_provider][parity]")
{
    auto sp = make_provider_test_spec();
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0);

    DlibVec x;
    x.set_size(2);
    x(0) = 1.0;
    x(1) = 0.5;

    /* Cost parity. */
    const double y = gp.cost(x);
    CHECK(y == Approx(std::sin(1.0) + 0.25));

    /* Gradient parity vs. analytic d/dx (sin(x)+y²) = (cos(x), 2y). */
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    REQUIRE(g.size() == 2);
    CHECK(g(0) == Approx(std::cos(1.0)).margin(1e-6));
    CHECK(g(1) == Approx(1.0)         .margin(1e-6));
}

TEST_CASE("GradientProvider Numerical respects MAXEVAL",
          "[HPO][grad_provider][maxeval]")
{
    auto sp = make_provider_test_spec();
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};
    const unsigned long cap = 3;

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, cap);

    DlibVec x;
    x.set_size(2);
    x(0) = 1.0;
    x(1) = 0.5;

    /* Two cost calls + at least one gradient (which would normally need
     * 2*N=4 cost calls under CENTRAL diff). With MAXEVAL=3, at least
     * the second cost or the gradient must signal exhaustion. */
    CHECK(std::isfinite(gp.cost(x)));   // counter -> 1
    CHECK(std::isfinite(gp.cost(x)));   // counter -> 2
    CHECK(std::isfinite(gp.cost(x)));   // counter -> 3

    /* Counter is now at the cap; further cost returns +inf. */
    CHECK_FALSE(std::isfinite(gp.cost(x)));
    CHECK(counter.load() <= cap + 1);  // tolerate one race-window overshoot

    /* Gradient hits the cap (returns false) without spawning further
     * host calls beyond the gate. */
    DlibVec g;
    CHECK_FALSE(gp.gradient(x, g));
}

TEST_CASE("GradientProvider Numerical propagates non-finite host eval",
          "[HPO][grad_provider][bad_eval]")
{
    auto sp = make_provider_test_spec();
    StubHostBadAtOrigin host;
    std::atomic<unsigned long> counter{0};

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0);

    DlibVec x;
    x.set_size(2);
    x(0) = 0.0;
    x(1) = 0.0;

    /* Cost at the bad point returns +inf. */
    CHECK_FALSE(std::isfinite(gp.cost(x)));

    /* Gradient at the bad point returns false (NumericDiff sees NaN). */
    DlibVec g;
    CHECK_FALSE(gp.gradient(x, g));
}
#endif  // TADAH_USE_CERES

TEST_CASE("GradientProvider Autodiff strategy throws not-implemented",
          "[HPO][grad_provider][autodiff]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Autodiff;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0);

    DlibVec x;
    x.set_size(2);
    x(0) = 1.0;
    x(1) = 0.5;

    /* Cost still works (only gradient is stubbed). */
    CHECK(std::isfinite(gp.cost(x)));

    /* Gradient throws the planned not-yet-implemented diagnostic. */
    DlibVec g;
    CHECK_THROWS_WITH(gp.gradient(x, g),
                      Catch::Contains("AUTODIFF") &&
                      Catch::Contains("NUMERICAL"));
}

#include "catch2/catch.hpp"

#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/refit/refit_parser.h>
#include <tadah/core/context.h>

#include <atomic>
#include <cmath>
#include <dlib/matrix.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

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

TEST_CASE("ANNEAL capability help exposes corrected public contracts",
          "[HPO][anneal][help]")
{
    auto capture = [](const std::string& algo) {
        std::ostringstream out;
        std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
        tadah::hpo::OptimizerSpec::print_caps("TADAH", algo);
        std::cout.rdbuf(old);
        return out.str();
    };

    /* The key listings are generated from the registry
     * (blueprint/anneal_keys.toml), so these checks pin the emitter
     * format and a few registry facts, not hand-written prose.
     * The help is text for a user: the keys a family does not read are
     * "refused" ("Dead knobs are hard errors" before the help was
     * rewritten), ANNEAL_PA "accepts and does not use" the INIT_JITTER
     * calibration keys of ANNEAL_PT ("Accepted as an inactive setting by
     * the current ANNEAL_PA driver"), and no key's help gives history
     * (INIT_JITTER_LADDER OFF was "(pre-1.5.0 behaviour)"); the help says
     * what a key does, not what the reader is to do ("Adapts ...", not
     * "Adapt ..."; POPULATION is read by TADAH:ANNEAL_PA, not "use
     * TADAH:ANNEAL_PA"): these checks fail on the text before that
     * rewrite. */
    const std::string serial = capture("ANNEAL");
    CHECK(serial.find("PARAM T_INIT") != std::string::npos);
    CHECK(serial.find("PARAM CHAIN_LENGTH (default auto)")
          != std::string::npos);
    CHECK(serial.find("PARAM RT (default 0.85)") != std::string::npos);
    CHECK(serial.find("Keys it does not read are refused: STEP")
          != std::string::npos);
    CHECK(serial.find("POPULATION, which TADAH:ANNEAL_PA reads")
          != std::string::npos);

    const std::string pt = capture("ANNEAL_PT");
    CHECK(pt.find("PARAM SWAP_WINDOW (default 20)") != std::string::npos);
    CHECK(pt.find("PARAM STEP_MODE <FIXED|FROZEN|ADAPTIVE> (default FROZEN)")
          != std::string::npos);
    CHECK(pt.find("best-stall") != std::string::npos);
    CHECK(pt.find("With REPLICAS fixed") != std::string::npos);
    CHECK(pt.find("PARAM RT") == std::string::npos);
    CHECK(pt.find("OFF = every slot uses the same sigma.")
          != std::string::npos);
    CHECK(pt.find("pre-1.5.0") == std::string::npos);
    for (const char* doc :
         {"Adapts interior ladder temperatures",
          "Freezes ladder adaptation once",
          "ANNEAL_PT INIT_MODE JITTER AUTO measures at most",
          "ANNEAL_PT INIT_MODE JITTER AUTO re-probes at most",
          "Scales each slot's initial jitter"}) {
        INFO(doc);
        CHECK(pt.find(doc) != std::string::npos);
    }

    const std::string pa = capture("ANNEAL_PA");
    CHECK(pa.find("all-fail skips") != std::string::npos);
    CHECK(pa.find("POPULATION <R> population size") != std::string::npos);
    CHECK(pa.find("PARAM INIT_MODE <JITTER|UNIFORM> (default JITTER)")
          != std::string::npos);
    /* INIT_JITTER_MAX, _PROBES, _MAXEVAL, _MAXTIME_FRAC, _SAMPLE and
     * _AUDIT, each listed once */
    std::size_t unused = 0;
    for (auto at = pa.find("ANNEAL_PA accepts it and does not use it.");
         at != std::string::npos;
         at = pa.find("ANNEAL_PA accepts it and does not use it.", at + 1))
        ++unused;
    CHECK(unused == 6u);
    CHECK(pa.find("driver") == std::string::npos);
}

TEST_CASE("HPO placeholder — see comments above for planned tests", "[HPO]")
{
    // No self-contained integration tests yet.
    // All unit-level regression tests are in MODELS/tests/test_reinit.cpp.
    SUCCEED();
}

TEST_CASE("HPOTargets accepts OPTIM WATOM with pair-triple input",
          "[HPO][watom][pair_triple][optim]")
{
    /* Regression for the stod-on-symbol crash: when WATOM is in
     * pair-triple format (`sym sym val ...`), HPOTargets::get_labels
     * must not call stod on the symbol tokens. arg_pos counts the
     * numeric (weight) entries only, so (1) selects the first triple's
     * value, (2) the second, etc. Bounds apply to that value. */
    ScopedTempFile input(R"(OPTIM WATOM (1) 0.1 5.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("ATOM", "C"); ctx.add("ATOM", "H");
    ctx.add("WATOM", "C"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
    ctx.add("WATOM", "H"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
    ctx.add("WATOM", "C"); ctx.add("WATOM", "C"); ctx.add("WATOM", 1.0);

    REQUIRE_NOTHROW(tadah::hpo::HPOTargets(ctx, input.path.string()));

    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    REQUIRE(trg.optimap.count("WATOM") == 1);
    const auto& w = trg.optimap.at("WATOM");

    /* Pair-triple config keeps symbols at positions 0,1,3,4,6,7 and
     * weights at 2,5,8. (1) targets the first weight slot (idx 2). */
    REQUIRE(w.opt.size() == 9);
    CHECK(w.opt[0] == false);  // "C" symbol
    CHECK(w.opt[1] == false);  // "H" symbol
    CHECK(w.opt[2] == true);   // C-H weight — optimised
    CHECK(w.opt[3] == false);
    CHECK(w.opt[5] == false);  // H-H weight — not optimised
    CHECK(w.opt[8] == false);  // C-C weight — not optimised
    CHECK(w.lowbound[2] == Approx(0.1));
    CHECK(w.highbound[2] == Approx(5.0));
}

TEST_CASE("HPOTargets pair-triple WATOM with three OPTIM indices",
          "[HPO][watom][pair_triple][optim]")
{
    /* Optimise all three pair weights with the same bounds. Indices are
     * comma-separated inside the parens; arg_pos counts the numeric
     * (weight) entries only. (1,2,3) or the equivalent range (1-3)
     * selects the three weight slots. */
    ScopedTempFile input(R"(OPTIM WATOM (1-3) 0.1 5.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("ATOM", "C"); ctx.add("ATOM", "H");
    ctx.add("WATOM", "C"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
    ctx.add("WATOM", "H"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
    ctx.add("WATOM", "C"); ctx.add("WATOM", "C"); ctx.add("WATOM", 1.0);

    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    const auto& w = trg.optimap.at("WATOM");
    REQUIRE(w.opt.size() == 9);
    CHECK(w.opt[2] == true);
    CHECK(w.opt[5] == true);
    CHECK(w.opt[8] == true);
    /* Symbol slots stay inactive. */
    CHECK(w.opt[0] == false);
    CHECK(w.opt[1] == false);
    CHECK(w.opt[3] == false);
    CHECK(w.opt[4] == false);
    CHECK(w.opt[6] == false);
    CHECK(w.opt[7] == false);
}

TEST_CASE("OPTIM FIXWEIGHT indices name model weights (mapped through FIXINDEX)",
          "[HPO][fixweight][optim]")
{
    /* FIXINDEX 4,1,7: slot 1 pins w4, slot 2 pins w1, slot 3 pins w7.
     * OPTIM FIXWEIGHT (7) selects the weight named 7, i.e. slot 3. */
    ScopedTempFile input(R"(OPTIM FIXWEIGHT (7) -1.0 1.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("FIXINDEX", "4,1,7");
    ctx.add("FIXWEIGHT", 0.4);
    ctx.add("FIXWEIGHT", 0.1);
    ctx.add("FIXWEIGHT", 0.7);

    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    const auto& o = trg.optimap.at("FIXWEIGHT");
    REQUIRE(o.opt.size() == 3);
    CHECK(o.opt[0] == false);
    CHECK(o.opt[1] == false);
    CHECK(o.opt[2] == true);
    CHECK(o.lowbound[2] == Approx(-1.0));
    CHECK(o.highbound[2] == Approx(1.0));
    /* The label names the pinned MODEL WEIGHT (the OPTIM index), not the
     * slot: slot 3 pins weight 7. */
    REQUIRE(trg.spec.parameter_labels.size() == 1);
    CHECK(trg.spec.parameter_labels[0] == "FIXWEIGHT[7]");
    REQUIRE(trg.spec.space);
    CHECK(trg.spec.space->at(0).user_index == 7);
}

TEST_CASE("OPTIM FIXWEIGHT refit back-compat: FIXINDEX 1-K, all searched",
          "[HPO][fixweight][optim]")
{
    /* `tadah refit` pins FIXINDEX 1-K, so weight index == slot and its
     * emitted per-coefficient OPTIM lines must select every slot. */
    ScopedTempFile input(R"(OPTIM FIXWEIGHT (1-5) -1.0 1.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("FIXINDEX", "1-5");
    for (int i = 0; i < 5; ++i) ctx.add("FIXWEIGHT", 0.1 * (i + 1));

    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    const auto& o = trg.optimap.at("FIXWEIGHT");
    REQUIRE(o.opt.size() == 5);
    for (size_t i = 0; i < 5; ++i) CHECK(o.opt[i] == true);
    REQUIRE(trg.spec.parameter_labels.size() == 5);
    CHECK(trg.spec.parameter_labels[0] == "FIXWEIGHT[1]");
    CHECK(trg.spec.parameter_labels[4] == "FIXWEIGHT[5]");
}

TEST_CASE("OPTIM FIXWEIGHT rejects weights not pinned by FIXINDEX",
          "[HPO][fixweight][optim]")
{
    ScopedTempFile input(R"(OPTIM FIXWEIGHT (3) -1.0 1.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("FIXINDEX", "4,1");
    ctx.add("FIXWEIGHT", 0.4);
    ctx.add("FIXWEIGHT", 0.1);

    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, input.path.string()),
                        Catch::Contains("not pinned by FIXINDEX"));
}

TEST_CASE("OPTIM FIXWEIGHT without FIXINDEX in the training config throws",
          "[HPO][fixweight][optim]")
{
    ScopedTempFile input(R"(OPTIM FIXWEIGHT (1) -1.0 1.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("FIXWEIGHT", 0.4);

    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, input.path.string()),
                        Catch::Contains("requires FIXINDEX"));
}

TEST_CASE("OPTIM WEIGHTS is rejected with a FIXWEIGHT hint",
          "[HPO][fixweight][optim]")
{
    ScopedTempFile input(R"(OPTIM WEIGHTS (1) -1.0 1.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    ctx.add("WEIGHTS", 0.4);

    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, input.path.string()),
                        Catch::Contains("not a valid search target"));
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
    CHECK(isp.warm_on_mismatch == "skip");
}

TEST_CASE("HPO INCLUDE_CONFIG keyword is rejected with a helpful message",
          "[HPO][optim][init_strategy]")
{
    /* INCLUDE_CONFIG was removed: the in-config point is now included
     * only when the user lists `METHOD CONFIG` explicitly. The old
     * keyword must fail with a message pointing at METHOD CONFIG (the
     * canonical INIT keyword) rather than a generic "unknown key" error. */
    ScopedTempFile input(R"(OPTIMIZER
INIT
  INCLUDE_CONFIG false
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    bool caught = false;
    std::string msg;
    try {
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
    } catch (const std::exception& e) {
        caught = true;
        msg = e.what();
    }
    CHECK(caught);
    CHECK(msg.find("METHOD CONFIG") != std::string::npos);
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
     * config.hpo under examples/) failed to
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

TEST_CASE("HPO parses standalone CERES STEEPEST_DESCENT optimizer blocks",
          "[HPO][optim][ceres][parse]")
{
    /* Pins: CERES:STEEPEST_DESCENT's block with its stops and PARAMs.
     * Before the fix: passes as well (a pin; the case parsed CERES:LBFGS
     * with PARAM max_lbfgs_rank 10, which is refused since). */
    ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO STEEPEST_DESCENT
MAXEVAL 30
FTOL_REL 1e-7
XTOL_REL 1e-9
PARAM fd_step 1e-6
PARAM tol_gradinf 1e-8
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.lib  == "CERES");
    CHECK(trg.spec.algo == "STEEPEST_DESCENT");
    REQUIRE(trg.spec.stop.maxeval.has_value());
    CHECK(*trg.spec.stop.maxeval == 30u);
    CHECK(trg.spec.stop.ftol_rel == Approx(1e-7));
    CHECK(trg.spec.stop.xtol_rel == Approx(1e-9));
    CHECK(trg.spec.param.at("fd_step")        == Approx(1e-6));
    CHECK(trg.spec.param.at("tol_gradinf")    == Approx(1e-8));
}

TEST_CASE("CERES:LBFGS, WOLFE and NLCG are refused at their block's line: "
          "their Wolfe line search stalls at a point outside the OPTIM box",
          "[HPO][optim][ceres][parse][out_of_box]")
{
    /* A point outside the OPTIM box is scored, not evaluated (box_penalty.h),
     * and Ceres' Wolfe line search
     * shrinks its step to nothing at that score (optimizer_caps.cpp).
     * Pins: the block naming CERES:LBFGS, WOLFE or NLCG is refused at its
     * closing line, "CERES:<ALGO> is refused: Ceres' GradientProblemSolver
     * has no bounds, ...", naming NLOPT:LD_LBFGS, DLIB:BFGS and
     * CERES:STEEPEST_DESCENT; the lookup gives the same text.
     * Before the fix: all three parsed and ran (and evaluated their points
     * outside the box). */
    for (const std::string algo : {"LBFGS", "WOLFE", "NLCG"}) {
        INFO(algo);
        ScopedTempFile input("OPTIMIZER\nLIB CERES\nALGO " + algo +
                             "\nMAXEVAL 5\nENDOPTIMIZER\n");
        tadah::core::Context ctx;
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, input.path.string()),
            Catch::Contains("CERES:" + algo + " is refused: Ceres' "
                            "GradientProblemSolver has no bounds") &&
                Catch::Contains("The recommended replacements are "
                                "NLOPT:LD_LBFGS (L-BFGS within the box), "
                                "DLIB:BFGS and CERES:STEEPEST_DESCENT"));
        CHECK_THROWS_WITH(tadah::hpo::optimizer_caps("CERES", algo),
                          Catch::StartsWith("CERES:" + algo + " is refused: "));
    }
}

TEST_CASE("HPO parses GRAD_STRATEGY",
          "[HPO][optim][grad_strategy][parse]")
{
    /* Before the fix: passes as well (a pin; its CERES blocks named
     * CERES:LBFGS, which is refused since). */
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
    /* ANALYTICAL */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO STEEPEST_DESCENT
GRAD_STRATEGY ANALYTICAL
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        REQUIRE(trg.spec.grad_strategy.has_value());
        CHECK(*trg.spec.grad_strategy == tadah::hpo::GradStrategy::Analytical);
    }
    /* HYBRID */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO STEEPEST_DESCENT
GRAD_STRATEGY HYBRID
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, input.path.string());
        REQUIRE(trg.spec.grad_strategy.has_value());
        CHECK(*trg.spec.grad_strategy == tadah::hpo::GradStrategy::Hybrid);
    }
    /* AUTODIFF — an early keyword, removed with a tailored
     * migration message pointing at ANALYTICAL/HYBRID. */
    {
        ScopedTempFile input(R"(OPTIMIZER
LIB CERES
ALGO STEEPEST_DESCENT
GRAD_STRATEGY AUTODIFF
MAXEVAL 5
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        CHECK_THROWS_WITH(tadah::hpo::HPOTargets(ctx, input.path.string()),
                          Catch::Contains("AUTODIFF") &&
                          Catch::Contains("ANALYTICAL"));
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

TEST_CASE("CERES caps registry exposes STEEPEST_DESCENT, the GradientProblemSolver "
          "algo HPO runs", "[HPO][optim][ceres][caps]")
{
    /* Before the fix: passes as well (a pin; it also looked up CERES:LBFGS,
     * WOLFE and NLCG, refused since). */
    const std::string library = "CERES";
    for (const std::string algo : {"STEEPEST_DESCENT"}) {
        const auto& caps = tadah::hpo::optimizer_caps(library, algo);
        CHECK(caps.is_gradient_based);
        CHECK_FALSE(caps.is_global);
        CHECK_FALSE(caps.reads_population);
        CHECK_FALSE(caps.accepts_inner);
        CHECK(caps.supports_maxeval);
    }
}

TEST_CASE("CERES NLLS family is refused by the caps registry",
          "[HPO][optim][ceres][deferred]")
{
    /* Pins: Ceres' least-squares solvers are refused through the caps
     * registry's refusal table (optimizer_caps.cpp): the lookup throws
     * "CERES:<ALGO> is refused: <reason>", the reason naming
     * CERES:STEEPEST_DESCENT, the Ceres solver HPO runs, and the
     * bound-constrained gradient searches NLOPT:LD_LBFGS and DLIB:BFGS, so
     * every check that looks the backend up -- the parser's at the block's
     * closing line, validate_hpo_setup, run_inner -- reports the same text;
     * reject_refused_backend() throws it with the phase in front;
     * CERES:STEEPEST_DESCENT and another library's LEVENBERG_MARQUARDT (a
     * typo, unknown) are not refused.
     * Before the fix: the reason named "CERES:LBFGS, CERES:WOLFE,
     * CERES:NLCG or CERES:STEEPEST_DESCENT", three solvers refused since
     * (their Wolfe line search stalls at a point outside the OPTIM box).
     * The reason names the command, tadah hpo ("HPO" before the help was
     * rewritten for a user). */
    for (const std::string algo : {"LEVENBERG_MARQUARDT", "DOGLEG", "SUBSPACE_DOGLEG"}) {
        INFO(algo);
        CHECK_THROWS_WITH(tadah::hpo::optimizer_caps("CERES", algo),
                          Catch::StartsWith("CERES:" + algo + " is refused: ") &&
                          Catch::Contains("The recommended replacements are "
                                          "CERES:STEEPEST_DESCENT, Ceres' "
                                          "line-search solver of a scalar loss "
                                          "that tadah hpo runs, and the "
                                          "bound-constrained gradient searches "
                                          "NLOPT:LD_LBFGS and DLIB:BFGS"));
        CHECK_THROWS_WITH(tadah::hpo::reject_refused_backend("CERES", algo,
                                                             "the EXPLORE block"),
                          Catch::StartsWith("the EXPLORE block: CERES:" + algo +
                                            " is refused: "));
    }
    /* The registered solver is not refused. */
    for (const std::string algo : {"STEEPEST_DESCENT"}) {
        INFO(algo);
        CHECK_NOTHROW(tadah::hpo::reject_refused_backend("CERES", algo));
        CHECK_NOTHROW(tadah::hpo::optimizer_caps("CERES", algo));
    }
    /* Another library's name is not refused: it is unknown. */
    CHECK_NOTHROW(tadah::hpo::reject_refused_backend("DLIB", "LEVENBERG_MARQUARDT"));
    CHECK_THROWS_WITH(tadah::hpo::optimizer_caps("DLIB", "LEVENBERG_MARQUARDT"),
                      Catch::StartsWith("unknown optimiser DLIB:LEVENBERG_MARQUARDT: "));
}

/* The CERES validators: CERES:STEEPEST_DESCENT's, the Ceres solver HPO runs
 * (CERES:LBFGS, WOLFE and NLCG are refused by the caps registry). Before
 * the fix the cases below ran the same checks through the LBFGS, WOLFE and
 * NLCG validators as well, which are gone with those solvers; each passes
 * before the fix as well (a pin), except the case on the PARAMs of the
 * refused solvers. */
TEST_CASE("CERES validators accept minimal bounded specs",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
    CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_steepest_spec(sp));
}

TEST_CASE("CERES validators reject INNER blocks",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_BOBYQA";
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_steepest_spec(sp),
                      Catch::Contains("standalone optimiser") &&
                      Catch::Contains("INNER"));
}

TEST_CASE("CERES validators reject integer-valued parameters",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
    sp.is_int = {true, false};
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_steepest_spec(sp),
                      Catch::Contains("integer-valued"));
}

TEST_CASE("CERES validators reject FTOL_ABS / XTOL_ABS",
          "[HPO][optim][ceres][validate]")
{
    {
        auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
        sp.stop.ftol_abs = 1e-6;
        CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_steepest_spec(sp),
                          Catch::Contains("FTOL_REL"));
    }
    {
        auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
        sp.stop.xtol_abs = 1e-8;
        CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_steepest_spec(sp),
                          Catch::Contains("XTOL_REL"));
    }
}

TEST_CASE("CERES validators reject unsupported PARAMs",
          "[HPO][optim][ceres][validate]")
{
    auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
    sp.param["bogus_param"] = 1.0;
    CHECK_THROWS_WITH(tadah::hpo::detail::validate_ceres_steepest_spec(sp),
                      Catch::Contains("does not support PARAM"));
}

TEST_CASE("CERES:STEEPEST_DESCENT refuses PARAM max_lbfgs_rank and nlcg_type, "
          "the PARAMs of the refused CERES:LBFGS and CERES:NLCG",
          "[HPO][optim][ceres][validate]")
{
    /* Pins: PARAM max_lbfgs_rank and PARAM nlcg_type, which
     * CERES:STEEPEST_DESCENT does not read, are refused as any other PARAM
     * it does not read ("does not support PARAM max_lbfgs_rank; supported:
     * fd_step, fd_kernel, diff_method, tol_gradinf, max_iter,
     * max_ls_iter").
     * Before the fix: both were accepted on every CERES solver, read by
     * CERES:LBFGS and CERES:NLCG only, and the list of supported PARAMs
     * named them. */
    for (const std::string key : {"max_lbfgs_rank", "nlcg_type"}) {
        INFO(key);
        auto sp = make_minimal_ceres_spec("STEEPEST_DESCENT");
        sp.param[key] = 1.0;
        CHECK_THROWS_WITH(
            tadah::hpo::detail::validate_ceres_steepest_spec(sp),
            Catch::Equals("CERES:STEEPEST_DESCENT does not support PARAM " +
                          key + "; supported: fd_step, fd_kernel, diff_method, "
                          "tol_gradinf, max_iter, max_ls_iter"));
    }
}

/* Ceres refuses a negative function_tolerance (FTOL_REL) or
 * parameter_tolerance (XTOL_REL), max_num_iterations (PARAM max_iter)
 * below 0 and max_num_line_search_step_size_iterations (PARAM max_ls_iter)
 * below 1 before it evaluates anything, and the integer PARAMs (max_iter,
 * max_ls_iter) are Ceres ints.
 * Pins: the validator refuses FTOL_REL -1 and XTOL_REL -0.001 ("requires
 * FTOL_REL >= 0", "requires XTOL_REL >= 0"), PARAM max_iter -1, 1.5 and
 * 3e9 ("requires PARAM max_iter to be an integer from 0 to 2147483647"),
 * PARAM max_ls_iter 0 and 2.5 ("... from 1 to 2147483647"), naming the
 * value; the edges are accepted: FTOL_REL 0, XTOL_REL 0, max_iter 0 and
 * 2147483647, max_ls_iter 1.
 * Before the fix: passes as well (a pin; it also checked CERES:LBFGS's
 * max_lbfgs_rank and CERES:NLCG's nlcg_type, PARAMs refused since). */
TEST_CASE("CERES validators refuse the tolerances and integer PARAMs Ceres "
          "cannot take", "[HPO][optim][ceres][validate]")
{
    const auto validate = [](const tadah::hpo::OptimizerSpec& sp) {
        tadah::hpo::detail::validate_ceres_steepest_spec(sp);
    };
    for (const std::string algo : {"STEEPEST_DESCENT"}) {
        INFO("CERES:" << algo);
        const std::string who = "CERES:" + algo + " requires ";
        {
            auto sp = make_minimal_ceres_spec(algo);
            sp.stop.ftol_rel = -1.0;
            CHECK_THROWS_WITH(validate(sp),
                              Catch::StartsWith(who + "FTOL_REL >= 0") &&
                              Catch::EndsWith(", got -1"));
        }
        {
            auto sp = make_minimal_ceres_spec(algo);
            sp.stop.xtol_rel = -1e-3;
            CHECK_THROWS_WITH(validate(sp),
                              Catch::StartsWith(who + "XTOL_REL >= 0") &&
                              Catch::EndsWith(", got -0.001"));
        }
        for (const auto& bad : {std::make_pair(-1.0, "-1"),
                                std::make_pair(1.5, "1.5"),
                                std::make_pair(3e9, "3e+09")}) {
            auto sp = make_minimal_ceres_spec(algo);
            sp.param["max_iter"] = bad.first;
            CHECK_THROWS_WITH(validate(sp),
                              Catch::StartsWith(
                                  who + "PARAM max_iter to be an integer "
                                        "from 0 to 2147483647") &&
                              Catch::EndsWith(std::string(", got ") +
                                              bad.second));
        }
        for (const auto& bad : {std::make_pair(0.0, "0"),
                                std::make_pair(2.5, "2.5")}) {
            auto sp = make_minimal_ceres_spec(algo);
            sp.param["max_ls_iter"] = bad.first;
            CHECK_THROWS_WITH(validate(sp),
                              Catch::StartsWith(
                                  who + "PARAM max_ls_iter to be an integer "
                                        "from 1 to 2147483647") &&
                              Catch::EndsWith(std::string(", got ") +
                                              bad.second));
        }
        auto edges = make_minimal_ceres_spec(algo);
        edges.stop.ftol_rel = 0.0;
        edges.stop.xtol_rel = 0.0;
        edges.param["max_iter"] = 0.0;
        edges.param["max_ls_iter"] = 1.0;
        CHECK_NOTHROW(validate(edges));
        edges.param["max_iter"] = 2147483647.0;
        CHECK_NOTHROW(validate(edges));
    }
}

/* =========================================================================
 * GradientProvider tests — exercise the unified gradient seam against a
 * stub host. These cover Ceres-NumericDiff parity vs. analytic, MAXEVAL
 * accounting, bad-eval propagation, and the ANALYTICAL/HYBRID dispatch.
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

/* NUMERICAL and HYBRID take their finite differences in Tadah
 * (fd_stencil.h), Ceres' step rule included: no Ceres build needed. */
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

    /* Gradient at the bad point returns false (the loss at the centre is
     * not finite). */
    DlibVec g;
    CHECK_FALSE(gp.gradient(x, g));
}

TEST_CASE("GradientProvider Analytical without callback throws wiring error",
          "[HPO][grad_provider][analytical]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    /* No analytical_grad_fn passed -> the gradient lambda must throw a
     * wiring-error diagnostic, not silently fall back. */
    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    CHECK(std::isfinite(gp.cost(x)));
    DlibVec g;
    CHECK_THROWS_WITH(gp.gradient(x, g),
                      Catch::Contains("ANALYTICAL") &&
                      Catch::Contains("internal wiring"));
}

TEST_CASE("GradientProvider Analytical with full callback matches analytic",
          "[HPO][grad_provider][analytical][parity]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    /* Callback fills both indices analytically: d/dx (sin(x)+y²) = (cos x, 2y). */
    auto agf = [](const DlibVec& x, DlibVec& g, std::vector<int>& nfd) -> bool {
        g.set_size(2);
        g(0) = std::cos(x(0));
        g(1) = 2.0 * x(1);
        (void)nfd;  // empty
        return true;
    };

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0, agf);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    CHECK(g(0) == Approx(std::cos(1.0)).margin(1e-12));
    CHECK(g(1) == Approx(1.0)         .margin(1e-12));
    /* One eval counted for the entire analytical pass. */
    CHECK(counter.load() == 1u);
}

TEST_CASE("GradientProvider Analytical with leftover needs_fd throws strict diagnostic",
          "[HPO][grad_provider][analytical][strict]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Analytical;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    /* Callback fills index 0 but punts index 1 onto needs_fd. Strict
     * Analytical mode must throw, naming the missing index. */
    auto agf = [](const DlibVec& x, DlibVec& g, std::vector<int>& nfd) -> bool {
        g.set_size(2);
        g(0) = std::cos(x(0));
        g(1) = 0.0;        // placeholder
        nfd.push_back(1);
        return true;
    };

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0, agf);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    /* The message names the index and the possible causes, and points at
     * HYBRID; it used to send the user to a "dtheta_registry" that does
     * not exist. */
    CHECK_THROWS_WITH(gp.gradient(x, g),
                      Catch::Contains("ANALYTICAL") &&
                      Catch::Contains("HYBRID") &&
                      Catch::Contains("indices: [1]") &&
                      Catch::Contains("no analytical gradient for 1 "
                                      "parameter") &&
                      Catch::Contains("a loss term") &&
                      !Catch::Contains("dtheta_registry"));
}

TEST_CASE("GradientProvider Hybrid runs masked FD on needs_fd indices",
          "[HPO][grad_provider][hybrid][parity]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    /* Callback fills index 0 analytically, punts index 1 to FD. */
    auto agf = [](const DlibVec& x, DlibVec& g, std::vector<int>& nfd) -> bool {
        g.set_size(2);
        g(0) = std::cos(x(0));
        g(1) = 0.0;        // placeholder
        nfd.push_back(1);
        return true;
    };

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0, agf);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    CHECK(g(0) == Approx(std::cos(1.0)).margin(1e-12));
    /* FD on index 1 with default fd_step=1e-6 hits 2y = 1.0 to FD precision. */
    CHECK(g(1) == Approx(1.0).margin(1e-5));
    /* 1 (the loss at the point, which gradient() evaluates first: its kind
     * decides every difference) + 1 (analytical pass) + 2 (central-diff for
     * index 1) = 4. Before the fix: 3, without the loss at the point. */
    CHECK(counter.load() == 4u);
}

TEST_CASE("GradientProvider Hybrid respects MAXEVAL on FD pass",
          "[HPO][grad_provider][hybrid][maxeval]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    auto agf = [](const DlibVec& x, DlibVec& g, std::vector<int>& nfd) -> bool {
        g.set_size(2);
        g(0) = std::cos(x(0));
        g(1) = 0.0;
        nfd.push_back(1);
        return true;
    };

    /* Cap = 2: the loss at the point and the analytical pass consume them;
     * no FD sample fits before the cap → gradient returns false (graceful
     * termination).
     * Before the fix: passes as well (a pin; the analytical pass and one of
     * the two FD samples consumed the cap: gradient() did not evaluate the
     * loss at the point first). */
    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/2, agf);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    CHECK_FALSE(gp.gradient(x, g));
}

TEST_CASE("GradientProvider Hybrid takes finite differences on every "
          "parameter when none has an analytical derivative",
          "[HPO][grad_provider][hybrid][all_fd]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    /* Callback returns success but routes EVERY index to needs_fd -- a
     * setup the analytical gradient does not cover (OALGO other than 4,
     * FIXINDEX, ...), a candidate rejected before training, a PC_LAMMPS
     * chain that failed. HYBRID skipped the finite differences there and
     * returned g = 0, so DLIB:BFGS ended COMPLETED at its start after two
     * evaluations (example 01 without PC_LAMMPS, GRAD_STRATEGY HYBRID).
     * The fix: difference every parameter, as the host's journal line
     * already said it would. */
    auto agf = [](const DlibVec& x, DlibVec& g, std::vector<int>& nfd) -> bool {
        g.set_size(2);
        g(0) = 0.0;
        g(1) = 0.0;
        nfd.push_back(0);
        nfd.push_back(1);
        (void)x;
        return true;
    };

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0, agf);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    REQUIRE(gp.gradient(x, g));
    CHECK(g(0) == Approx(std::cos(1.0)).margin(1e-6));
    CHECK(g(1) == Approx(1.0).margin(1e-6));
    /* The loss at the point, the analytical pass and the central pair of
     * each parameter. Before the fix: 5, without the loss at the point. */
    CHECK(counter.load() == 6u);
}

TEST_CASE("GradientProvider Hybrid without callback throws wiring error",
          "[HPO][grad_provider][hybrid]")
{
    auto sp = make_provider_test_spec();
    sp.grad_strategy = tadah::hpo::GradStrategy::Hybrid;
    StubHostSinXY2 host;
    std::atomic<unsigned long> counter{0};

    auto gp = tadah::hpo::make_gradient_provider<DlibVec, void>(
        sp, host, &counter, /*maxeval=*/0);

    DlibVec x; x.set_size(2); x(0)=1.0; x(1)=0.5;
    DlibVec g;
    CHECK_THROWS_WITH(gp.gradient(x, g),
                      Catch::Contains("HYBRID") &&
                      Catch::Contains("internal wiring"));
}

/* ---- tadah refit stage-pipeline parser (INIT/EXPLORE/REFINE/FUSE) ---- */
namespace {
// Mirror what CORE's config parser deposits: each stage block becomes a
// whitespace-split STRINGS Context key. Add token-by-token (these are valid
// blueprint STRINGS keys, so Context::add accepts them).
void add_block(tadah::core::Context &c, const std::string &key,
               const std::string &body) {
    std::istringstream iss(body);
    std::string t;
    while (iss >> t) c.add(key, t);
}
}  // namespace

TEST_CASE("RefitParser parses the INIT/EXPLORE/REFINE/FUSE pipeline") {
    using namespace tadah::hpo::refit;
    tadah::core::Context c;
    add_block(c, "INIT", "PHI XMIN PIN F XMAX PIN RHO MANUAL 2.55 1.5");
    add_block(c, "EXPLORE",
              "MAX_ADD 8 OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 60000 "
              "INNER LIB NLOPT ALGO LN_SBPLX MAXEVAL 20000 FTOL_REL 1e-13 "
              "XTOL_REL 1e-11 ENDINNER ENDOPTIMIZER");
    add_block(c, "REFINE",
              "OPTIMIZER LIB CERES ALGO LM MAXEVAL 1000 FTOL_REL 1e-15 "
              "GTOL 1e-14 ENDOPTIMIZER");
    add_block(c, "FUSE",
              "OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 2000 FTOL_REL 1e-15 "
              "ENDOPTIMIZER");

    RefitConfig rc = RefitParser::parse(c);

    REQUIRE(rc.init.size() == 3);
    CHECK(rc.init[0].channel == InitBasis::PHI);
    CHECK(rc.init[0].placement == PinMode::XMin);
    CHECK(rc.init[0].pinned);
    CHECK(rc.init[1].channel == InitBasis::F);
    CHECK(rc.init[1].placement == PinMode::XMax);
    CHECK(rc.init[1].pinned);
    CHECK(rc.init[2].channel == InitBasis::RHO);
    CHECK(rc.init[2].placement == PinMode::Manual);
    CHECK(rc.init[2].mu == Approx(2.55));
    CHECK(rc.init[2].has_eta);
    CHECK(rc.init[2].eta == Approx(1.5));
    CHECK_FALSE(rc.init[2].pinned);

    CHECK(rc.explore.has_opt);
    // a single MAX_ADD value fills all three per-function slots [phi, rho, F]
    CHECK(rc.explore.max_add[0] == 8);
    CHECK(rc.explore.max_add[1] == 8);
    CHECK(rc.explore.max_add[2] == 8);
    CHECK(refit_explore_outer_algo(rc.explore.opt) == "MLSL");
    REQUIRE(rc.explore.opt.inner);
    CHECK(refit_explore_inner_algo(*rc.explore.opt.inner) == "SBPLX");
    REQUIRE(rc.explore.opt.stop.maxeval.has_value());
    CHECK(rc.explore.opt.stop.maxeval.value() == 60000);
    REQUIRE(rc.explore.opt.inner->stop.maxeval.has_value());
    CHECK(rc.explore.opt.inner->stop.maxeval.value() == 20000);

    CHECK(refit_refine_algo(rc.refine.opt) == "CERES_LM");
    CHECK(rc.refine.opt.stop.maxeval.value() == 1000);  // MAXEVAL -> ceres iters
    REQUIRE(rc.refine.opt.stop.ftol_rel.has_value());    // FTOL_REL -> function_tolerance
    CHECK(rc.refine.opt.stop.ftol_rel.value() == Approx(1e-15));
    CHECK(rc.refine.opt.param.at("GTOL") == Approx(1e-14));

    CHECK(refit_fuse_algo(rc.fuse.opt) == "CERES_DOGLEG");
    CHECK(rc.fuse.opt.stop.maxeval.value() == 2000);
}

TEST_CASE("RefitParser rejects old grammar and illegal stage algorithms") {
    using namespace tadah::hpo::refit;

    SECTION("a top-level OPTIMIZER block (old grammar) is rejected") {
        tadah::core::Context c;
        add_block(c, "OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL");
        CHECK_THROWS(RefitParser::parse(c));
    }
    SECTION("FUSE only accepts least-squares Ceres") {
        tadah::hpo::OptimizerSpec s; s.lib = "CERES"; s.algo = "LBFGS";
        CHECK_THROWS(refit_fuse_algo(s));
    }
    SECTION("REFINE rejects FTOL_ABS (Ceres function_tolerance is relative)") {
        tadah::core::Context c;
        add_block(c, "REFINE",
                  "OPTIMIZER LIB CERES ALGO LM FTOL_ABS 1e-12 ENDOPTIMIZER");
        CHECK_THROWS(RefitParser::parse(c));
    }
    SECTION("FUSE rejects the bare FTOL alias") {
        tadah::core::Context c;
        add_block(c, "FUSE",
                  "OPTIMIZER LIB CERES ALGO DOGLEG FTOL 1e-12 ENDOPTIMIZER");
        CHECK_THROWS(RefitParser::parse(c));
    }
    SECTION("EXPLORE outer must be a global search") {
        tadah::hpo::OptimizerSpec s; s.lib = "NLOPT"; s.algo = "LN_SBPLX";
        CHECK_THROWS(refit_explore_outer_algo(s));
    }
    SECTION("UNIFORM disables the global seed (accepted)") {
        tadah::hpo::OptimizerSpec s; s.algo = "UNIFORM";
        CHECK(refit_explore_outer_algo(s) == "UNIFORM");
    }
    SECTION("XMAX on PHI is parsed (engine skips it at resolution time)") {
        tadah::core::Context c;
        add_block(c, "INIT", "PHI XMAX PIN");
        RefitConfig rc = RefitParser::parse(c);
        REQUIRE(rc.init.size() == 1);
        CHECK(rc.init[0].placement == PinMode::XMax);
    }
}

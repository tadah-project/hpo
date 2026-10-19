// test_pc_observable_parser.cpp
//
// The .hpotarget parser exposes
// PC_OBSERVABLE entries via the typed Observable interface. Verifies
// that a config line containing a cohesive_energy directive is parsed
// into an Observable with the expected fields, and that error cases
// (missing flags, malformed values, unknown kinds) are flagged.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/strategy_rules.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

/* Minimal Context — HPOTargets needs a few keys present to instantiate. */
tadah::core::Context make_context_for_targets() {
    tadah::core::Context c;
    c.add("ATOM",   "Ta");
    c.add("WATOM",  1);
    c.add("INIT2B", "false");
    c.add("INITMB", "false");
    c.add("BIAS",   "false");
    c.add("NORM",   "false");
    c.add("MODEL",  "M_BLR");
    c.add("MODEL",  "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO",  4);
    c.add("FORCE",  "false");
    c.add("STRESS", "false");
    c.add("VERBOSE", 0);
    return c;
}

/* Helper: write an .hpotarget snippet to a temp file and return its path.
 * The name must be unique ACROSS PROCESSES, not just within one: ctest runs
 * every TEST_CASE as its own process (so a static counter restarts at 1 in
 * each) and `ctest -j` runs them concurrently -- a counter-only name made
 * parallel cases overwrite each other's configs (flaky no-throw failures). */
/* "Error on line N: " for the first line N of `body` that reads `keyword`
 * (leading blanks aside): the line HPOTargets names for a refusal of the
 * block that line closes. */
std::string error_at_block_end(const std::string& body,
                               const std::string& keyword) {
  std::istringstream in(body);
  std::string text;
  for (std::size_t n = 1; std::getline(in, text); ++n) {
    const auto p = text.find_first_not_of(' ');
    if (p != std::string::npos && text.substr(p) == keyword)
      return "Error on line " + std::to_string(n) + ": ";
  }
  return "(no line reads " + keyword + ")";
}

std::filesystem::path write_temp_config(const std::string& body) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path()
           / ("test_pc_observable_parser_"
              + std::to_string(::getpid()) + "_"
              + std::to_string(++counter) + ".cfg");
    std::ofstream f(p);
    f << body;
    f.close();
    return p;
}

}  // namespace

/* ============================ unit-level ============================ */

TEST_CASE("parse_pc_observable_line: cohesive_energy with all required flags",
          "[hpo][pc_observable_parser][cohesive_energy]")
{
    tadah::core::PeriodicTable::initialize();

    /* Minimal — all required + initial value. */
    auto obs = tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 3.3036");

    REQUIRE(obs != nullptr);
    REQUIRE(obs->kind() == "cohesive_energy");
    CHECK(std::abs(obs->target() - 8.100) < 1e-12);
    CHECK(std::abs(obs->weight() - 1000.0) < 1e-12);

    auto* cohs = dynamic_cast<tadah::hpo::CohesiveEnergyObservable*>(obs.get());
    REQUIRE(cohs != nullptr);
    CHECK(std::abs(cohs->current_a0() - 3.3036) < 1e-12);
}

TEST_CASE("parse_pc_observable_line: cohesive_energy hotvar without init seeds midpoint",
          "[hpo][pc_observable_parser][cohesive_energy][hotvar]")
{
    tadah::core::PeriodicTable::initialize();

    auto obs = tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9");

    auto* cohs = dynamic_cast<tadah::hpo::CohesiveEnergyObservable*>(obs.get());
    REQUIRE(cohs != nullptr);
    CHECK(std::abs(cohs->current_a0() - 0.5 * (2.7 + 3.9)) < 1e-12);
}

TEST_CASE("parse_pc_observable_line: --supercell sets nx/ny/nz",
          "[hpo][pc_observable_parser][cohesive_energy][supercell]")
{
    tadah::core::PeriodicTable::initialize();

    /* Parser accepts --supercell; we don't expose nx/ny/nz on the public
     * interface in MVP, but we can at least confirm the parser doesn't
     * reject the syntax and the constructor accepts it. */
    CHECK_NOTHROW(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --supercell 5 5 5"));
}

TEST_CASE("parse_pc_observable_line: --etol sets relax_etol",
          "[hpo][pc_observable_parser][cohesive_energy][etol]")
{
    tadah::core::PeriodicTable::initialize();

    /* Smoke: syntax accepted, doesn't throw. The production runs target
     * 1e-15 to match LAMMPS Ecoh_fast.in. */
    CHECK_NOTHROW(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --etol 1e-15"));
}

TEST_CASE("parse_pc_observable_line: --unconverged const accepts a value",
          "[hpo][pc_observable_parser][cohesive_energy][unconverged]")
{
    tadah::core::PeriodicTable::initialize();

    /* `const 100.0` matches Ecoh_fast.in's eq_ok=false branch (cliff). */
    CHECK_NOTHROW(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --unconverged const 100.0"));
}

TEST_CASE("parse_pc_observable_line: --unconverged force-scaled accepts a slope",
          "[hpo][pc_observable_parser][cohesive_energy][unconverged]")
{
    tadah::core::PeriodicTable::initialize();

    /* `force-scaled <slope>` gives local optimisers a smooth gradient
     * toward fmax→0 instead of a FAIL_SCORE cliff. */
    CHECK_NOTHROW(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --unconverged force-scaled 10.0"));
}

TEST_CASE("parse_pc_observable_line: --unconverged unknown kind throws",
          "[hpo][pc_observable_parser][cohesive_energy][unconverged][error]")
{
    tadah::core::PeriodicTable::initialize();

    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --unconverged ridiculous 100.0"),
        std::runtime_error);
    /* Legacy --unconverged-penalty syntax no longer accepted. */
    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --unconverged-penalty 100.0"),
        std::runtime_error);
}

TEST_CASE("parse_pc_observable_line: missing required flags throw",
          "[hpo][pc_observable_parser][error]")
{
    tadah::core::PeriodicTable::initialize();

    /* No --species. */
    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --target 8.100 --weight 1000 --hotvar a0 2.7 3.9"),
        std::runtime_error);
    /* No --target. */
    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --weight 1000 --hotvar a0 2.7 3.9"),
        std::runtime_error);
    /* No --weight. */
    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --hotvar a0 2.7 3.9"),
        std::runtime_error);
    /* No --hotvar. */
    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000"),
        std::runtime_error);
}

TEST_CASE("parse_pc_observable_line: unknown observable kind throws",
          "[hpo][pc_observable_parser][error][unknown_kind]")
{
    tadah::core::PeriodicTable::initialize();

    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "this_does_not_exist --target 1.0 --weight 1.0"),
        std::runtime_error);
}

TEST_CASE("parse_pc_observable_line: unknown flag throws",
          "[hpo][pc_observable_parser][error][unknown_flag]")
{
    tadah::core::PeriodicTable::initialize();

    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8.100 --weight 1000 "
        "--hotvar a0 2.7 3.9 --bogus_flag whatever"),
        std::runtime_error);
}

TEST_CASE("parse_pc_observable_line: malformed value throws",
          "[hpo][pc_observable_parser][error][malformed]")
{
    tadah::core::PeriodicTable::initialize();

    CHECK_THROWS_AS(tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target not_a_number --weight 1000 "
        "--hotvar a0 2.7 3.9"),
        std::runtime_error);
}

/* ============================ end-to-end ============================ */

TEST_CASE("HPOTargets parses PC_OBSERVABLE into observables vector",
          "[hpo][pc_observable_parser][hpotargets][e2e]")
{
    tadah::core::PeriodicTable::initialize();

    auto ctx = make_context_for_targets();
    auto cfg = write_temp_config(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036
)");
    tadah::hpo::HPOTargets trg(ctx, cfg.string());

    REQUIRE(trg.observables.size() == 1);
    CHECK(trg.observables[0]->kind() == "cohesive_energy");
    CHECK(std::abs(trg.observables[0]->target() - 8.100) < 1e-12);
    CHECK(std::abs(trg.observables[0]->weight() - 1000.0) < 1e-12);

    /* PC_OBSERVABLE-only run: LS is empty by construction. PC_OBSERVABLE
     * and PC_LAMMPS are now mutually exclusive paths (see coexistence
     * test below); side-by-side declaration is rejected at parse time. */
    CHECK(trg.LS.empty());

    std::filesystem::remove(cfg);
}

TEST_CASE("HPOTargets rejects PC_OBSERVABLE with GRAD_STRATEGY NUMERICAL",
          "[hpo][pc_observable_parser][hpotargets][always_analytical]")
{
    tadah::core::PeriodicTable::initialize();

    auto ctx = make_context_for_targets();

    /* OUTER NUMERICAL — must reject. */
    {
        auto cfg = write_temp_config(R"(OPTIMIZER
LIB NLOPT
ALGO LD_LBFGS
MAXEVAL 1
GRAD_STRATEGY NUMERICAL
ENDOPTIMIZER
LOSS L2
PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036
)");
        /* At the line that closes the block (ENDOPTIMIZER, line 6). Before
         * the fix: the message without a line. */
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::StartsWith(
                "Error on line 6: --hpotarget defines PC_OBSERVABLE but "
                "OPTIMIZER's GRAD_STRATEGY=NUMERICAL is set.")
            && Catch::Matchers::EndsWith("\nLine: ENDOPTIMIZER"));
        std::filesystem::remove(cfg);
    }

    /* HYBRID — must accept (PC_OBSERVABLE redundant-but-equivalent path). */
    {
        auto cfg = write_temp_config(R"(OPTIMIZER
LIB NLOPT
ALGO LD_LBFGS
MAXEVAL 1
GRAD_STRATEGY HYBRID
ENDOPTIMIZER
LOSS L2
PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036
)");
        REQUIRE_NOTHROW(tadah::hpo::HPOTargets(ctx, cfg.string()));
        std::filesystem::remove(cfg);
    }

    /* ANALYTICAL — must accept. */
    {
        auto cfg = write_temp_config(R"(OPTIMIZER
LIB NLOPT
ALGO LD_LBFGS
MAXEVAL 1
GRAD_STRATEGY ANALYTICAL
ENDOPTIMIZER
LOSS L2
PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036
)");
        REQUIRE_NOTHROW(tadah::hpo::HPOTargets(ctx, cfg.string()));
        std::filesystem::remove(cfg);
    }

    /* INNER NUMERICAL inside an MLSL outer — must also reject. INNER
     * lives inside STRATEGY per the config grammar. An INNER block takes
     * no GRAD_STRATEGY at all, so the line is refused where it stands,
     * before the PC_OBSERVABLE rules run. */
    {
        auto cfg = write_temp_config(R"(OPTIMIZER
  STRATEGY
    LIB NLOPT
    ALGO GN_MLSL_LDS
    MAXEVAL 1
    INNER
      LIB NLOPT
      ALGO LD_LBFGS
      GRAD_STRATEGY NUMERICAL
    ENDINNER
  ENDSTRATEGY
ENDOPTIMIZER
LOSS L2
PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036
)");
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::StartsWith(
                "Error on line 9: GRAD_STRATEGY is not accepted inside an "
                "INNER block: the enclosing block's GRAD_STRATEGY applies "
                "to its local search; set GRAD_STRATEGY on the enclosing "
                "STRATEGY block"));
        std::filesystem::remove(cfg);
    }
}

/* Pins (among the refusals of this case): GRAD_STRATEGY NUMERICAL, HYBRID
 * or ANALYTICAL beside PC_LAMMPS is refused with a message that names
 * PC_LAMMPS and GRAD_STRATEGY and says "PC_LAMMPS does not support
 * analytical/numerical/hybrid gradients. Drop the GRAD_STRATEGY line.",
 * with no date ("2026") in it: no release supported these gradients with
 * PC_LAMMPS (v1.3.0-beta.1 and 1.5.0 refused them as now).
 * Before the fix: the message said "PC_LAMMPS no longer supports
 * analytical/numerical/hybrid gradients (2026-05-13). Drop the
 * GRAD_STRATEGY line.": the check of the sentence fails. */
TEST_CASE("HPOTargets rejects PC_LAMMPS with gradient-based optimisers",
          "[hpo][pc_observable_parser][hpotargets][pc_lammps_grad_drop]")
{
    tadah::core::PeriodicTable::initialize();

    /* Stub LAMMPS script (required by LammpsSim's --script path check). */
    auto cwd = std::filesystem::current_path();
    auto script_path = cwd / "test_pc_lammps_grad_drop_stub.in";
    {
        std::ofstream f(script_path);
        f << "# stub script\nprint \"hi\"\nvariable xyz equal 0.0\n";
    }
    auto ctx = make_context_for_targets();
    const std::string pc_lammps_line =
        "PC_LAMMPS --script " + script_path.string() +
        " --varloss xyz 1.0 100 --outvar xyz\n";

    /* GRAD_STRATEGY rejected with PC_LAMMPS, regardless of value. */
    for (const char* gs : {"NUMERICAL", "HYBRID", "ANALYTICAL"}) {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\n") +
            "LIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 1\n" +
            "GRAD_STRATEGY " + gs + "\n" +
            "ENDOPTIMIZER\nLOSS L2\n" + pc_lammps_line);
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains("PC_LAMMPS")
            && Catch::Matchers::Contains("GRAD_STRATEGY"));
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains(
                "GRAD_STRATEGY is set. PC_LAMMPS does not support "
                "analytical/numerical/hybrid gradients. Drop the "
                "GRAD_STRATEGY line.")
            && !Catch::Matchers::Contains("2026"));
        std::filesystem::remove(cfg);
    }

    /* Gradient-based NLopt LD_LBFGS rejected. */
    {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\nLIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n")
            + pc_lammps_line);
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains("PC_LAMMPS")
            && Catch::Matchers::Contains("LD_LBFGS")
            && Catch::Matchers::Contains("gradient-free"));
        std::filesystem::remove(cfg);
    }

    /* CERES:STEEPEST_DESCENT rejected (gradient-based, no special-case;
     * the case named CERES:LBFGS, refused since: a pin). */
    {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\nLIB CERES\nALGO STEEPEST_DESCENT\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n")
            + pc_lammps_line);
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains("PC_LAMMPS")
            && Catch::Matchers::Contains("CERES"));
        std::filesystem::remove(cfg);
    }

    /* DLIB:BFGS ACCEPTED — the explicit exception (per-θ-per-iter
     * dtheta, not a full-vector gradient). */
    {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n")
            + pc_lammps_line);
        REQUIRE_NOTHROW(tadah::hpo::HPOTargets(ctx, cfg.string()));
        std::filesystem::remove(cfg);
    }

    /* NLopt gradient-free GN_/LN_ accepted. */
    for (const char* algo : {"GN_MLSL_LDS", "GN_CRS2_LM", "LN_SBPLX", "LN_BOBYQA"}) {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\nLIB NLOPT\nALGO ") + algo +
            "\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps_line);
        REQUIRE_NOTHROW(tadah::hpo::HPOTargets(ctx, cfg.string()));
        std::filesystem::remove(cfg);
    }

    /* INNER LD_LBFGS inside MLSL: also rejected. */
    {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\n  STRATEGY\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n")
            + "    MAXEVAL 1\n    INNER\n      LIB NLOPT\n      ALGO LD_LBFGS\n"
              "    ENDINNER\n"
            + "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps_line);
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains("PC_LAMMPS")
            && Catch::Matchers::Contains("INNER")
            && Catch::Matchers::Contains("LD_LBFGS"));
        std::filesystem::remove(cfg);
    }

    std::filesystem::remove(script_path);
}

TEST_CASE("HPOTargets rejects PC_OBSERVABLE and PC_LAMMPS side by side",
          "[hpo][pc_observable_parser][hpotargets][coexistence]")
{
    tadah::core::PeriodicTable::initialize();

    /* LammpsSim's parser validates that --script points at a real file,
     * so write an empty stub alongside the .hpotarget. The error we
     * test fires AFTER all lines are parsed (post-parse validation), so
     * the LammpsSim parse step still runs and needs a real path. */
    auto cwd = std::filesystem::current_path();
    auto script_path = cwd / "test_pc_observable_parser_stub.in";
    {
        std::ofstream f(script_path);
        f << "# stub LAMMPS script for coexistence-rejection test\n"
          << "print \"hello\"\n"
          << "variable xyz equal 0.0\n";
    }

    auto ctx = make_context_for_targets();
    auto cfg = write_temp_config(std::string{} +
        "OPTIMIZER\n"
        "LIB DLIB\n"
        "ALGO BFGS\n"
        "MAXEVAL 1\n"
        "ENDOPTIMIZER\n"
        "LOSS L2\n"
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight 1000 --hotvar a0 2.7 3.9 3.3036\n"
        "PC_LAMMPS --script " + script_path.string() +
        " --varloss xyz 1.0 100 --outvar xyz\n");

    /* PC_OBSERVABLE is mutually exclusive with PC_LAMMPS. The post-parse
     * validation in HPOTargets::read fires a clear runtime_error mentioning
     * both counts.
     * Pins: the message states the rule, without a development state.
     * Before the fix: "... these paths are mutually exclusive in the
     * current MVP. ...". */
    CHECK_THROWS_WITH(
        tadah::hpo::HPOTargets(ctx, cfg.string()),
        Catch::Matchers::Contains(
            "--hpotarget defines both PC_OBSERVABLE (1 entries) and "
            "PC_LAMMPS (1 entries); PC_OBSERVABLE and PC_LAMMPS lines are "
            "mutually exclusive. Remove one set of lines and rerun."));

    std::filesystem::remove(cfg);
    std::filesystem::remove(script_path);
}

/* Pins: with PC_LAMMPS, an NLopt MLSL or AUGLAG is judged by its local
 * search, as obtains_gradients judges it: MLSL and AUGLAG evaluate their
 * own points without a gradient, and the D of GD_MLSL, GD_MLSL_LDS,
 * LD_AUGLAG and LD_AUGLAG_EQ only picks NLopt's default local optimiser,
 * LD_MMA, which an INNER block replaces.
 *   - accepted: each of the four with a derivative-free INNER block
 *     (LN_BOBYQA, LN_SBPLX, LN_COBYLA), at the STRATEGY level and as the
 *     PRESTAGE exploration;
 *   - refused, with a message that names the local search: GD_MLSL and
 *     LD_AUGLAG without an INNER block ("--hpotarget defines PC_LAMMPS but
 *     OPTIMIZER's optimiser NLOPT:GD_MLSL takes a gradient: NLopt's MLSL
 *     and AUGLAG take one for their local search, and NLopt's default
 *     local optimiser, NLOPT:LD_MMA, is gradient-based; an INNER block with
 *     a derivative-free one takes none. PC_LAMMPS requires a gradient-free
 *     optimiser ...");
 *   - refused as before: GN_MLSL_LDS with an INNER LD_LBFGS, at the INNER
 *     block ("... INNER's optimiser NLOPT:LD_LBFGS is gradient-based.").
 * Each refusal names the line that closes the block it refuses
 * (ENDOPTIMIZER, ENDINNER), as the parser's refusals do; the pins read the
 * message alone before that fix, which it was (no line).
 * Before the fix: the four with a derivative-free INNER block were refused
 * ("OPTIMIZER's optimiser NLOPT:GD_MLSL is gradient-based"), and the two
 * without one with that message, which does not say why. */
TEST_CASE("HPOTargets judges an MLSL or AUGLAG under PC_LAMMPS by its local "
          "search", "[hpo][pc_observable_parser][hpotargets][pc_lammps_grad_drop]")
{
    tadah::core::PeriodicTable::initialize();
    auto script_path = std::filesystem::current_path() /
                       "test_pc_lammps_local_search_stub.in";
    {
        std::ofstream f(script_path);
        f << "# stub script\nprint \"hi\"\nvariable xyz equal 0.0\n";
    }
    auto ctx = make_context_for_targets();
    const std::string pc_lammps =
        "PC_LAMMPS --script " + script_path.string() +
        " --varloss xyz 1.0 100 --outvar xyz\n";
    auto error = [&](const std::string& body) {
        auto cfg = write_temp_config(body);
        std::string what;
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
        } catch (const std::exception& e) {
            what = e.what();
        }
        std::filesystem::remove(cfg);
        return what;
    };
    auto plain = [&](const std::string& algo, const std::string& inner) {
        std::string b = "OPTIMIZER\n  STRATEGY\n    LIB NLOPT\n    ALGO " +
                        algo + "\n    MAXEVAL 1\n";
        if (!inner.empty())
            b += "    INNER\n      LIB NLOPT\n      ALGO " + inner +
                 "\n    ENDINNER\n";
        return b + "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps;
    };

    struct Accepted { std::string algo, inner; };
    for (const auto& c : std::vector<Accepted>{{"GD_MLSL", "LN_BOBYQA"},
                                               {"GD_MLSL_LDS", "LN_SBPLX"},
                                               {"LD_AUGLAG", "LN_BOBYQA"},
                                               {"LD_AUGLAG_EQ", "LN_COBYLA"}}) {
        INFO(c.algo << " with an INNER " << c.inner);
        CHECK(error(plain(c.algo, c.inner)) == "");
    }
    {
        INFO("EXPLORE GD_MLSL_LDS with an INNER LN_SBPLX");
        CHECK(error("OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n"
                    "    LIB NLOPT\n    ALGO LN_BOBYQA\n    MAXEVAL 1\n"
                    "    EXPLORE\n      LIB NLOPT\n      ALGO GD_MLSL_LDS\n"
                    "      MAXEVAL 1\n      INNER\n        LIB NLOPT\n"
                    "        ALGO LN_SBPLX\n      ENDINNER\n    ENDEXPLORE\n"
                    "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps) == "");
    }
    for (const std::string algo : {"GD_MLSL", "LD_AUGLAG"}) {
        INFO(algo << " without an INNER block");
        CHECK_THAT(error(plain(algo, "")),
                   Catch::StartsWith(
                       error_at_block_end(plain(algo, ""), "ENDOPTIMIZER") +
                       "--hpotarget defines PC_LAMMPS but OPTIMIZER's "
                       "optimiser NLOPT:" + algo + " takes a gradient: "
                       "NLopt's MLSL and AUGLAG take one for their local "
                       "search, and NLopt's default local optimiser, "
                       "NLOPT:LD_MMA, is gradient-based; an INNER block "
                       "with a derivative-free one takes none. PC_LAMMPS "
                       "requires a gradient-free optimiser"));
    }
    {
        INFO("GN_MLSL_LDS with an INNER LD_LBFGS");
        CHECK_THAT(error(plain("GN_MLSL_LDS", "LD_LBFGS")),
                   Catch::StartsWith(
                       error_at_block_end(plain("GN_MLSL_LDS", "LD_LBFGS"),
                                          "ENDINNER") +
                       "--hpotarget defines PC_LAMMPS but INNER's optimiser "
                       "NLOPT:LD_LBFGS is gradient-based. PC_LAMMPS requires "
                       "a gradient-free optimiser") &&
                   Catch::EndsWith("\nLine:     ENDINNER"));
    }
    std::filesystem::remove(script_path);
}

/* Pins: with PC_LAMMPS, the refusal of a gradient-based optimiser in an
 * INNER block names what an INNER block takes there: an INNER search is a
 * local NLopt algorithm (require_inner_backend(), at startup), so of the
 * advice of the other blocks only the derivative-free local NLopt
 * algorithms remain, every one, from the registry:
 *   "--hpotarget defines PC_LAMMPS but INNER's optimiser NLOPT:LD_MMA is
 *   gradient-based. PC_LAMMPS requires a gradient-free optimiser, and an
 *   INNER search is a local NLopt algorithm: name a derivative-free one in
 *   the INNER block (NLOPT:LN_AUGLAG, NLOPT:LN_AUGLAG_EQ, NLOPT:LN_BOBYQA,
 *   NLOPT:LN_COBYLA, NLOPT:LN_NELDERMEAD, NLOPT:LN_PRAXIS,
 *   NLOPT:LN_SBPLX)."
 * at the INNER block's ENDINNER line -- for the G_MLSL_LDS search of
 * data/ta/target_files/mlsl_inner.hpo with an
 * INNER LD_MMA, an INNER GD_STOGO (gradient-based and global), an INNER
 * CERES:LBFGS, and an EXPLORE block's INNER LD_LBFGS; an INNER LD_AUGLAG,
 * whose NLopt default local optimiser LD_MMA takes the gradient, is
 * refused naming LD_MMA, without the advice of an INNER block of its own
 * (an INNER block has none). None of these names NLOPT GN_*, TADAH or
 * DLIB:BFGS. Each algorithm the advice names is accepted in that INNER
 * block by the parser and by validate_run_policy() (startup's INNER
 * rules). The STRATEGY-level and EXPLORE blocks name every optimiser that
 * takes no gradient, from the registry (ASA:ASA, DLIB:GFS, the NLopt
 * derivative-free algorithms, TADAH:ANNEAL, ANNEAL_PA, ANNEAL_PT, GRID and
 * RANDOM), "or DLIB:BFGS, the one gradient-based optimiser it accepts";
 * they named "NLOPT GN_*\/LN_*, TADAH RANDOM/GRID/ANNEAL" (no DLIB:GFS,
 * ASA, ANNEAL_PT or ANNEAL_PA) and said that DLIB:BFGS "uses
 * per-theta-per-iter dtheta, not a full gradient", while it takes the
 * central-difference gradient of the loss, as an NLopt LD_* algorithm
 * would (the pin of that text is replaced).
 * Before the fix: every INNER refusal gave the other blocks' advice,
 * "PC_LAMMPS requires a gradient-free optimiser (NLOPT GN_*\/LN_*, TADAH
 * RANDOM/GRID/ANNEAL). The only gradient-based exception is DLIB:BFGS
 * (...)", and the INNER LD_AUGLAG one "...; an INNER block with a
 * derivative-free one takes none. ...": an INNER GN_* is refused at
 * startup as a global INNER search, an INNER TADAH or DLIB block as not
 * NLopt, so the advice led to another refusal; the checks of the INNER
 * messages fail, the accepted list and the other blocks' advice pass as
 * well (pins). */
TEST_CASE("HPOTargets names, for an INNER block under PC_LAMMPS, the "
          "derivative-free local NLopt algorithms an INNER block takes",
          "[hpo][pc_observable_parser][hpotargets][pc_lammps_grad_drop]"
          "[no_nested_global]")
{
    tadah::core::PeriodicTable::initialize();
    auto script_path = std::filesystem::current_path() /
                       "test_pc_lammps_inner_advice_stub.in";
    {
        std::ofstream f(script_path);
        f << "# stub script\nprint \"hi\"\nvariable xyz equal 0.0\n";
    }
    auto ctx = make_context_for_targets();
    const std::string pc_lammps =
        "PC_LAMMPS --script " + script_path.string() +
        " --varloss xyz 1.0 100 --outvar xyz\n";
    /* The refusal of the parse of `body`, and, when the parse accepts it,
     * that of validate_run_policy() on its spec ("" when both accept it). */
    auto error = [&](const std::string& body, bool startup) {
        auto cfg = write_temp_config(body);
        std::string what;
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            if (startup) tadah::hpo::validate_run_policy(trg.spec);
        } catch (const std::exception& e) {
            what = e.what();
        }
        std::filesystem::remove(cfg);
        return what;
    };
    /* The search of data/ta/target_files/mlsl_inner.hpo: G_MLSL_LDS with an
     * INNER block of `lib`:`algo`. */
    auto mlsl = [&](const std::string& lib, const std::string& algo) {
        return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
               "    ALGO G_MLSL_LDS\n    MAXEVAL 120\n    POPULATION 5\n"
               "    INNER\n      LIB " + lib + "\n      ALGO " + algo +
               "\n    ENDINNER\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" +
               pc_lammps;
    };
    const std::string inner_advice =
        " PC_LAMMPS requires a gradient-free optimiser, and an INNER search "
        "is a local NLopt algorithm: name a derivative-free one in the INNER "
        "block (NLOPT:LN_AUGLAG, NLOPT:LN_AUGLAG_EQ, NLOPT:LN_BOBYQA, "
        "NLOPT:LN_COBYLA, NLOPT:LN_NELDERMEAD, NLOPT:LN_PRAXIS, "
        "NLOPT:LN_SBPLX).";

    struct Case { std::string what, body, message, line; };
    std::vector<Case> cases;
    for (const auto& [lib, algo] :
         std::vector<std::pair<std::string, std::string>>{
             {"NLOPT", "LD_MMA"}, {"NLOPT", "GD_STOGO"}})
        cases.push_back({"an INNER " + lib + ":" + algo, mlsl(lib, algo),
                         "--hpotarget defines PC_LAMMPS but INNER's optimiser "
                         + lib + ":" + algo + " is gradient-based." +
                             inner_advice,
                         "    ENDINNER"});
    cases.push_back({"an INNER NLOPT:LD_AUGLAG", mlsl("NLOPT", "LD_AUGLAG"),
                     "--hpotarget defines PC_LAMMPS but INNER's optimiser "
                     "NLOPT:LD_AUGLAG takes a gradient: NLopt's MLSL and "
                     "AUGLAG take one for their local search, and NLopt's "
                     "default local optimiser, NLOPT:LD_MMA, is "
                     "gradient-based." + inner_advice,
                     "    ENDINNER"});
    cases.push_back(
        {"an EXPLORE block's INNER NLOPT:LD_LBFGS",
         "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB NLOPT\n"
         "    ALGO LN_BOBYQA\n    MAXEVAL 1\n    EXPLORE\n      LIB NLOPT\n"
         "      ALGO GN_MLSL_LDS\n      MAXEVAL 1\n      INNER\n"
         "        LIB NLOPT\n        ALGO LD_LBFGS\n      ENDINNER\n"
         "    ENDEXPLORE\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps,
         "--hpotarget defines PC_LAMMPS but EXPLORE INNER's optimiser "
         "NLOPT:LD_LBFGS is gradient-based." + inner_advice,
         "      ENDINNER"});
    for (const auto& c : cases) {
        INFO(c.what);
        const std::string got = error(c.body, false);
        CHECK(got == error_at_block_end(c.body, "ENDINNER") + c.message +
                         "\nLine: " + c.line);
        CHECK_THAT(got, !(Catch::Contains("GN_*") || Catch::Contains("TADAH") ||
                          Catch::Contains("DLIB:BFGS")));
    }

    /* An INNER CERES:LBFGS, a refused optimiser of another LIB: the INNER
     * rule refuses it at startup (an INNER search is a local NLopt
     * algorithm), and neither the parser nor the PC_LAMMPS rule gives it
     * the CERES refusal, whose advice (DLIB:BFGS, CERES:STEEPEST_DESCENT)
     * an INNER block does not take. Before the fix: the parser refused it
     * at ENDINNER with that refusal ("... CERES:LBFGS is refused: ...
     * Use NLOPT:LD_LBFGS (L-BFGS within the box) or DLIB:BFGS, or
     * CERES:STEEPEST_DESCENT ..."): the parse check fails. */
    {
        const std::string body = mlsl("CERES", "LBFGS");
        CHECK(error(body, false) == "");
        const std::string got = error(body, true);
        CHECK_THAT(got, Catch::Contains("names CERES:LBFGS, but "
                                        "NLOPT:G_MLSL_LDS runs its INNER "
                                        "search with NLopt: set LIB NLOPT "
                                        "and an NLopt ALGO"));
        CHECK_THAT(got, !(Catch::Contains("is refused") ||
                          Catch::Contains("DLIB:BFGS") ||
                          Catch::Contains("STEEPEST_DESCENT")));
    }

    /* Each algorithm the advice names is accepted in that INNER block, by
     * the parser and at startup. */
    for (const std::string algo : {"LN_AUGLAG", "LN_AUGLAG_EQ", "LN_BOBYQA",
                                   "LN_COBYLA", "LN_NELDERMEAD", "LN_PRAXIS",
                                   "LN_SBPLX"}) {
        INFO("an INNER NLOPT:" << algo);
        CHECK(error(mlsl("NLOPT", algo), true) == "");
    }

    /* The STRATEGY-level and EXPLORE blocks name every optimiser that takes
     * no gradient, and DLIB:BFGS. */
    const auto check_other = [](const std::string& got) {
        INFO(got);
        CHECK_THAT(got, Catch::Contains(
                            " PC_LAMMPS requires a gradient-free optimiser "
                            "(ASA:ASA, DLIB:GFS, NLOPT:"));
        for (const char* name :
             {"NLOPT:GN_CRS2_LM", "NLOPT:GN_DIRECT_L", "NLOPT:LN_BOBYQA",
              "NLOPT:LN_SBPLX", "TADAH:ANNEAL", "TADAH:ANNEAL_PA",
              "TADAH:ANNEAL_PT", "TADAH:GRID", "TADAH:RANDOM"})
            CHECK_THAT(got, Catch::Contains(name));
        CHECK_THAT(got, Catch::Contains(
                            "TADAH:RANDOM) or DLIB:BFGS, the one "
                            "gradient-based optimiser it accepts; an NLopt "
                            "MLSL or AUGLAG algorithm is accepted with a "
                            "derivative-free INNER block."));
        /* The advice, after the refused optimiser's name. */
        const auto at = got.find("PC_LAMMPS requires");
        REQUIRE(at != std::string::npos);
        const std::string advice = got.substr(at);
        CHECK_THAT(advice, !(Catch::Contains("per-theta") ||
                             Catch::Contains("NLOPT:LD_") ||
                             Catch::Contains("NLOPT:GD_") ||
                             Catch::Contains("CERES")));
    };
    {
        const std::string body =
            "OPTIMIZER\n  STRATEGY\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
            "    MAXEVAL 1\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps;
        const std::string got = error(body, false);
        CHECK_THAT(got, Catch::StartsWith(
                            error_at_block_end(body, "ENDOPTIMIZER") +
                            "--hpotarget defines PC_LAMMPS but OPTIMIZER's "
                            "optimiser NLOPT:LD_LBFGS is gradient-based. "
                            "PC_LAMMPS requires"));
        CHECK_THAT(got, Catch::EndsWith("\nLine: ENDOPTIMIZER"));
        check_other(got);
    }
    {
        const std::string body =
            "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB NLOPT\n"
            "    ALGO LN_BOBYQA\n    MAXEVAL 1\n    EXPLORE\n      LIB NLOPT\n"
            "      ALGO LD_MMA\n      MAXEVAL 1\n    ENDEXPLORE\n"
            "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + pc_lammps;
        const std::string got = error(body, false);
        CHECK_THAT(got, Catch::StartsWith(
                            error_at_block_end(body, "ENDEXPLORE") +
                            "--hpotarget defines PC_LAMMPS but EXPLORE's "
                            "optimiser NLOPT:LD_MMA is gradient-based. "
                            "PC_LAMMPS requires"));
        CHECK_THAT(got, Catch::EndsWith("\nLine:     ENDEXPLORE"));
        check_other(got);
    }
    /* Every optimiser the advice names runs with PC_LAMMPS as written, in
     * a STRATEGY block of its own: accepted by the parser and at startup
     * (validate_run_policy). Before the fix the advice named NLOPT:AUGLAG,
     * AUGLAG_EQ, G_MLSL and G_MLSL_LDS, which are refused without an INNER
     * block ("needs an INNER block: NLopt has no default local optimiser
     * for it"). */
    {
        const std::string got = error(
            "OPTIMIZER\n  STRATEGY\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
            "    MAXEVAL 1\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" +
                pc_lammps,
            false);
        const std::string open = "gradient-free optimiser (";
        const auto from = got.find(open);
        const auto to = got.find(") or DLIB:BFGS");
        REQUIRE(from != std::string::npos);
        REQUIRE(to != std::string::npos);
        std::istringstream names(got.substr(from + open.size(),
                                            to - from - open.size()));
        std::size_t n = 0;
        for (std::string name; std::getline(names >> std::ws, name, ',');) {
            ++n;
            const auto colon = name.find(':');
            REQUIRE(colon != std::string::npos);
            INFO(name);
            const std::string body =
                "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB " +
                name.substr(0, colon) + "\n    ALGO " +
                name.substr(colon + 1) + "\n    MAXEVAL 4\n  ENDSTRATEGY\n"
                "ENDOPTIMIZER\nLOSS L2\n" + pc_lammps;
            CHECK(error(body, true) == "");
        }
        CHECK(n > 10);
    }
    /* What the advice names is accepted with PC_LAMMPS (pins). */
    for (const std::string& lib_algo :
         {std::string("DLIB\n    ALGO GFS"), std::string("ASA\n    ALGO ASA"),
          std::string("TADAH\n    ALGO ANNEAL_PT"),
          std::string("DLIB\n    ALGO BFGS")}) {
        INFO(lib_algo);
        const std::string body =
            "OPTIMIZER\n  STRATEGY\n    LIB " + lib_algo +
            "\n    MAXEVAL 4\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" +
            pc_lammps;
        CHECK_THAT(error(body, false),
                   !Catch::Contains("PC_LAMMPS requires"));
    }
    std::filesystem::remove(script_path);
}

TEST_CASE("HPOTargets applies the PC_LAMMPS and PC_OBSERVABLE gradient rules "
          "to the EXPLORE block and its INNER block",
          "[hpo][pc_observable_parser][hpotargets][pc_lammps_grad_drop]")
{
    /* The rules above were checked for the STRATEGY-level optimiser and its
     * INNER block only. Under TYPE PRESTAGE the EXPLORE block (and an
     * INNER block inside it) configures an optimiser as well: NLOPT
     * LD_LBFGS there ran with PC_LAMMPS, failed at once inside NLopt
     * ("nlopt failure") and the run reported success after a wasted
     * exploration; a GRAD_STRATEGY there failed only after the first
     * evaluation. Each refusal names the line that closes the block it
     * refuses (ENDEXPLORE, the EXPLORE block's ENDINNER); the pins read the
     * message alone before that fix, which it was (no line). */
    tadah::core::PeriodicTable::initialize();
    auto script_path = std::filesystem::current_path() /
                       "test_pc_lammps_grad_explore_stub.in";
    {
        std::ofstream f(script_path);
        f << "# stub script\nprint \"hi\"\nvariable xyz equal 0.0\n";
    }
    auto ctx = make_context_for_targets();
    const std::string pc_lammps =
        "PC_LAMMPS --script " + script_path.string() +
        " --varloss xyz 1.0 100 --outvar xyz\n";
    const std::string pc_observable =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 "
        "--weight 1000 --hotvar a0 2.7 3.9 3.3036\n";

    /* A PRESTAGE run: the refinement at STRATEGY level, then `explore`. */
    auto prestage = [](const std::string& refine, const std::string& explore) {
        return "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n" + refine +
               "    MAXEVAL 1\n    EXPLORE\n" + explore +
               "    ENDEXPLORE\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n";
    };
    const std::string bobyqa = "    LIB NLOPT\n    ALGO LN_BOBYQA\n";
    const std::string mlsl_inner =
        "      LIB NLOPT\n      ALGO GN_MLSL_LDS\n      MAXEVAL 1\n"
        "      INNER\n        LIB NLOPT\n";
    auto error = [&](const std::string& body) {
        auto cfg = write_temp_config(body);
        std::string what;
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
        } catch (const std::exception& e) {
            what = e.what();
        }
        std::filesystem::remove(cfg);
        return what;
    };

    struct Case { std::string explore, loss_line, message, closing; };
    for (const auto& c : std::vector<Case>{
             /* PC_LAMMPS: a gradient-based optimiser ... */
             {"      LIB NLOPT\n      ALGO LD_LBFGS\n      MAXEVAL 1\n",
              pc_lammps,
              "--hpotarget defines PC_LAMMPS but EXPLORE's optimiser "
              "NLOPT:LD_LBFGS is gradient-based. PC_LAMMPS requires a "
              "gradient-free optimiser", "ENDEXPLORE"},
             {mlsl_inner + "        ALGO LD_LBFGS\n      ENDINNER\n",
              pc_lammps,
              "--hpotarget defines PC_LAMMPS but EXPLORE INNER's optimiser "
              "NLOPT:LD_LBFGS is gradient-based. PC_LAMMPS requires a "
              "gradient-free optimiser", "ENDINNER"},
             /* ... and any GRAD_STRATEGY. */
             {"      LIB DLIB\n      ALGO BFGS\n      MAXEVAL 1\n"
              "      GRAD_STRATEGY ANALYTICAL\n",
              pc_lammps,
              "--hpotarget defines PC_LAMMPS but EXPLORE's GRAD_STRATEGY is "
              "set.", "ENDEXPLORE"},
             /* PC_OBSERVABLE: GRAD_STRATEGY NUMERICAL. */
             {"      LIB NLOPT\n      ALGO LD_LBFGS\n      MAXEVAL 1\n"
              "      GRAD_STRATEGY NUMERICAL\n",
              pc_observable,
              "--hpotarget defines PC_OBSERVABLE but EXPLORE's "
              "GRAD_STRATEGY=NUMERICAL is set.", "ENDEXPLORE"}}) {
        const std::string body = prestage(bobyqa, c.explore) + c.loss_line;
        INFO(body);
        CHECK_THAT(error(body),
                   Catch::StartsWith(error_at_block_end(body, c.closing) +
                                     c.message) &&
                   Catch::Contains("\nLine: ") &&
                   Catch::EndsWith(c.closing));
    }

    /* An INNER block takes no GRAD_STRATEGY: whatever the loss, the line
     * is refused where it stands (line 14 of these files), before either
     * rule runs. */
    for (const auto& c : std::vector<Case>{
             {mlsl_inner + "        ALGO LN_SBPLX\n"
                           "        GRAD_STRATEGY HYBRID\n      ENDINNER\n",
              pc_lammps, "", ""},
             {mlsl_inner + "        ALGO LD_LBFGS\n"
                           "        GRAD_STRATEGY NUMERICAL\n      ENDINNER\n",
              pc_observable, "", ""}}) {
        const std::string body = prestage(bobyqa, c.explore) + c.loss_line;
        INFO(body);
        CHECK_THAT(error(body), Catch::StartsWith(
            "Error on line 14: GRAD_STRATEGY is not accepted inside an INNER "
            "block: the enclosing block's GRAD_STRATEGY applies to its local "
            "search; set GRAD_STRATEGY on the enclosing EXPLORE block"));
    }

    /* What the rules allow in the EXPLORE block is still accepted: with
     * PC_LAMMPS a gradient-free optimiser and DLIB:BFGS (its own
     * per-theta differences), with PC_OBSERVABLE the analytical modes. */
    for (const std::string& explore :
         {std::string("      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
                      "      MAXEVAL 1\n"),
          std::string("      LIB DLIB\n      ALGO BFGS\n      MAXEVAL 1\n"),
          mlsl_inner + "        ALGO LN_SBPLX\n      ENDINNER\n"}) {
        const std::string body = prestage(bobyqa, explore) + pc_lammps;
        INFO(body);
        CHECK(error(body) == "");
    }
    for (const std::string gs : {"HYBRID", "ANALYTICAL"}) {
        const std::string body =
            prestage(bobyqa, "      LIB NLOPT\n      ALGO LD_LBFGS\n"
                             "      MAXEVAL 1\n      GRAD_STRATEGY " + gs +
                             "\n") + pc_observable;
        INFO(body);
        CHECK(error(body) == "");
    }
    std::filesystem::remove(script_path);
}

/* An unset GRAD_STRATEGY means NUMERICAL, which PC_OBSERVABLE rejects when
 * it is written out; the analytical gradient is what PC_OBSERVABLE is
 * built for. The parser therefore resolves an unset GRAD_STRATEGY to
 * ANALYTICAL in every block that runs an optimiser and obtains gradients
 * -- the STRATEGY-level optimiser and EXPLORE -- and leaves the
 * derivative-free ones alone (ANNEAL and ASA reject any GRAD_STRATEGY).
 * An INNER block runs no optimiser of its own: NLopt's MLSL and AUGLAG
 * hand their local search the gradient of the enclosing block's
 * objective, so the enclosing block's GRAD_STRATEGY decides how it is
 * obtained and an INNER block is not resolved. An explicit HYBRID or
 * ANALYTICAL is kept, NUMERICAL is still rejected, and without
 * PC_OBSERVABLE nothing changes. */
TEST_CASE("HPOTargets resolves an unset GRAD_STRATEGY to ANALYTICAL in the "
          "gradient-based blocks of a PC_OBSERVABLE run",
          "[hpo][pc_observable_parser][hpotargets][grad_strategy_resolution]")
{
    using tadah::hpo::GradStrategy;
    tadah::core::PeriodicTable::initialize();
    const std::string observable =
        "LOSS L2\nPC_OBSERVABLE cohesive_energy --species Ta --target 8.100 "
        "--weight 1000 --hotvar a0 2.7 3.9 3.3036\n";

    struct Parsed {
        std::optional<GradStrategy> outer, inner, explore, explore_inner;
    };
    auto parse = [](const std::string& body) {
        auto ctx = make_context_for_targets();
        auto cfg = write_temp_config(body);
        Parsed out;
        {
            INFO(body);
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            const auto& sp = trg.spec;
            out.outer = sp.grad_strategy;
            if (sp.inner) out.inner = sp.inner->grad_strategy;
            if (sp.prestage) {
                out.explore = sp.prestage->grad_strategy;
                if (sp.prestage->inner)
                    out.explore_inner = sp.prestage->inner->grad_strategy;
            }
        }
        std::filesystem::remove(cfg);
        return out;
    };
    auto plain = [](const std::string& lib, const std::string& algo,
                    const std::string& extra = "") {
        return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB " + lib +
               "\n    ALGO " + algo + "\n    MAXEVAL 1\n" + extra +
               "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const std::optional<GradStrategy> unset;
    const auto analytical = std::optional<GradStrategy>(GradStrategy::Analytical);

    SECTION("gradient-based backends resolve to ANALYTICAL") {
        CHECK(parse(plain("DLIB", "BFGS") + observable).outer == analytical);
        CHECK(parse(plain("NLOPT", "LD_LBFGS") + observable).outer == analytical);
        CHECK(parse(plain("NLOPT", "LD_MMA", "    FTOL_REL 1e-6\n") +
                    observable).outer == analytical);
        /* CERES:STEEPEST_DESCENT (it was CERES:LBFGS, refused since; before
         * the fix the case passes as well, a pin) */
        CHECK(parse(plain("CERES", "STEEPEST_DESCENT") + observable).outer ==
              analytical);
        /* GD_MLSL_LDS's and LD_AUGLAG's default local search is LD_MMA
         * (their own points are evaluated without a gradient). */
        CHECK(parse(plain("NLOPT", "GD_MLSL_LDS") + observable).outer ==
              analytical);
        CHECK(parse(plain("NLOPT", "LD_AUGLAG") + observable).outer ==
              analytical);
        /* LN_AUGLAG's default local search is LN_COBYLA. */
        CHECK(parse(plain("NLOPT", "LN_AUGLAG") + observable).outer == unset);
    }
    SECTION("an explicit HYBRID or ANALYTICAL is kept") {
        CHECK(parse(plain("DLIB", "BFGS", "    GRAD_STRATEGY HYBRID\n") +
                    observable).outer == GradStrategy::Hybrid);
        CHECK(parse(plain("DLIB", "BFGS", "    GRAD_STRATEGY ANALYTICAL\n") +
                    observable).outer == analytical);
    }
    SECTION("an explicit NUMERICAL is still rejected") {
        auto ctx = make_context_for_targets();
        auto cfg = write_temp_config(
            plain("DLIB", "BFGS", "    GRAD_STRATEGY NUMERICAL\n") + observable);
        /* The message names both alternatives and what sets them apart:
         * ANALYTICAL (also what an unset GRAD_STRATEGY resolves to) needs
         * the settings of the analytical gradient; HYBRID uses finite
         * differences where it is unavailable. They used to be called
         * equivalent. */
        CHECK_THROWS_WITH(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          Catch::Matchers::Contains(
                              "GRAD_STRATEGY=NUMERICAL is set") &&
                          Catch::Matchers::Contains(
                              "an unset GRAD_STRATEGY resolves to it") &&
                          Catch::Matchers::Contains("finite differences") &&
                          !Catch::Matchers::Contains("equivalent"));
        std::filesystem::remove(cfg);
    }
    SECTION("derivative-free backends are left alone") {
        CHECK(parse(plain("TADAH", "ANNEAL") + observable).outer == unset);
        CHECK(parse(plain("ASA", "ASA") + observable).outer == unset);
        CHECK(parse(plain("NLOPT", "LN_BOBYQA") + observable).outer == unset);
        CHECK(parse(plain("DLIB", "GFS") + observable).outer == unset);
        CHECK(parse(plain("TADAH", "RANDOM") + observable).outer == unset);
    }
    SECTION("INNER: the block that hands a gradient-based local search its "
            "gradient resolves; the INNER block has no GRAD_STRATEGY") {
        const auto mlsl = [](const std::string& outer_extra,
                             const std::string& algo,
                             const std::string& inner_algo,
                             const std::string& inner_extra) {
            return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
                   "    ALGO " + algo + "\n    MAXEVAL 1\n" + outer_extra +
                   "    INNER\n      LIB NLOPT\n      ALGO " + inner_algo +
                   "\n" + inner_extra + "    ENDINNER\n  ENDSTRATEGY\n"
                   "ENDOPTIMIZER\n";
        };
        const std::string hybrid = "    GRAD_STRATEGY HYBRID\n";
        const auto hyb = std::optional<GradStrategy>(GradStrategy::Hybrid);

        /* Neither block sets one: the enclosing block resolves. */
        const auto neither = parse(mlsl("", "GN_MLSL_LDS", "LD_LBFGS", "") +
                                   observable);
        CHECK(neither.outer == analytical);
        CHECK(neither.inner == unset);
        /* The enclosing block sets one: it is kept, INNER stays unset. */
        const auto outer_only = parse(
            mlsl(hybrid, "GN_MLSL_LDS", "LD_LBFGS", "") + observable);
        CHECK(outer_only.outer == hyb);
        CHECK(outer_only.inner == unset);
        /* One written in INNER, with or without one in the enclosing
         * block, is refused at its line: the enclosing block's
         * GRAD_STRATEGY is the one that serves the local search. */
        for (const std::string& outer_extra : {std::string(), hybrid}) {
            auto ctx = make_context_for_targets();
            const std::string body =
                mlsl(outer_extra, "GN_MLSL_LDS", "LD_LBFGS", "  " + hybrid) +
                observable;
            INFO(body);
            const std::string at_line =
                outer_extra.empty() ? "Error on line 10: "
                                    : "Error on line 11: ";
            auto cfg = write_temp_config(body);
            CHECK_THROWS_WITH(tadah::hpo::HPOTargets(ctx, cfg.string()),
                              Catch::Matchers::StartsWith(
                                  at_line + "GRAD_STRATEGY is not accepted "
                                  "inside an INNER block"));
            std::filesystem::remove(cfg);
        }

        /* A derivative-free local search obtains no gradients. */
        const auto free = parse(mlsl("", "GN_MLSL_LDS", "LN_SBPLX", "") +
                                observable);
        CHECK(free.inner == unset);
        CHECK(free.outer == unset);
        /* AUGLAG with a gradient-based or a derivative-free INNER. */
        const auto auglag = parse(mlsl("", "AUGLAG", "LD_LBFGS", "") +
                                  observable);
        CHECK(auglag.outer == analytical);
        CHECK(auglag.inner == unset);
        CHECK(parse(mlsl("", "AUGLAG", "LN_BOBYQA", "") + observable).outer ==
              unset);
        /* Pins: the same whatever the D of the enclosing algorithm, which
         * only picks NLopt's default local optimiser (LD_MMA): an INNER
         * block replaces it, and MLSL and AUGLAG evaluate their own points
         * without a gradient.
         * Before the review fix: both resolved to ANALYTICAL
         * (obtains_gradients() answered from their own caps), so startup
         * checked that the analytical gradient can be formed for a run
         * that takes no gradient. */
        CHECK(parse(mlsl("", "GD_MLSL_LDS", "LN_SBPLX", "") + observable)
                  .outer == unset);
        CHECK(parse(mlsl("", "LD_AUGLAG", "LN_BOBYQA", "") + observable)
                  .outer == unset);
    }
    SECTION("EXPLORE resolves like the STRATEGY-level block, its INNER "
            "like an INNER block") {
        const auto prestage = [](const std::string& explore) {
            return "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB DLIB\n"
                   "    ALGO BFGS\n    MAXEVAL 1\n    EXPLORE\n" + explore +
                   "    ENDEXPLORE\n  ENDSTRATEGY\nENDOPTIMIZER\n";
        };
        const auto ld = parse(prestage("      LIB NLOPT\n      ALGO LD_LBFGS\n"
                                       "      MAXEVAL 1\n") + observable);
        CHECK(ld.outer == analytical);
        CHECK(ld.explore == analytical);
        const auto crs = parse(prestage("      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
                                        "      MAXEVAL 1\n") + observable);
        CHECK(crs.outer == analytical);
        CHECK(crs.explore == unset);
        const auto mlsl = parse(prestage(
            "      LIB NLOPT\n      ALGO GN_MLSL_LDS\n      MAXEVAL 1\n"
            "      INNER\n        LIB NLOPT\n        ALGO LD_LBFGS\n"
            "      ENDINNER\n") + observable);
        CHECK(mlsl.explore == analytical);
        CHECK(mlsl.explore_inner == unset);
    }
    SECTION("without PC_OBSERVABLE an unset GRAD_STRATEGY stays unset "
            "(NUMERICAL)") {
        CHECK(parse(plain("DLIB", "BFGS") + "LOSS L2\nPC_ERMSE 0 1\n").outer ==
              unset);
    }
}

TEST_CASE("The journal records a GRAD_STRATEGY resolved for PC_OBSERVABLE",
          "[hpo][pc_observable_parser][hpotargets][grad_strategy_resolution]")
{
    tadah::core::PeriodicTable::initialize();
    const std::string observable =
        "LOSS L2\nPC_OBSERVABLE cohesive_energy --species Ta --target 8.100 "
        "--weight 1000 --hotvar a0 2.7 3.9 3.3036\n";
    auto journal_of = [](const std::string& body) {
        auto ctx = make_context_for_targets();
        auto cfg = write_temp_config(body);
        const auto journal = std::filesystem::temp_directory_path() /
            ("test_pc_observable_journal_" + std::to_string(::getpid()) +
             ".log");
        {
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            auto& log = tadah::hpo::HPOLog::instance();
            log.open(journal.string(), "test",
                     std::filesystem::temp_directory_path().string(), 1);
            log.write_optimizer_section(trg.spec, trg.spec.parameter_labels);
            log.finish("TEST");
        }
        std::ifstream in(journal);
        std::stringstream ss;
        ss << in.rdbuf();
        std::filesystem::remove(cfg);
        std::filesystem::remove(journal);
        return ss.str();
    };
    auto line_with = [](const std::string& text, const std::string& what) {
        std::istringstream is(text);
        for (std::string line; std::getline(is, line);)
            if (line.find(what) != std::string::npos) return line;
        return std::string();
    };

    const std::string resolved = journal_of(
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n" + observable);
    INFO(resolved);
    CHECK_THAT(line_with(resolved, "GRAD_STRATEGY"),
               Catch::Matchers::Contains("ANALYTICAL (resolved for PC_OBSERVABLE)"));

    const std::string hybrid = journal_of(
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nGRAD_STRATEGY HYBRID\n"
        "ENDOPTIMIZER\n" + observable);
    INFO(hybrid);
    CHECK_THAT(line_with(hybrid, "GRAD_STRATEGY"),
               Catch::Matchers::Contains("HYBRID"));
    CHECK(hybrid.find("resolved for PC_OBSERVABLE") == std::string::npos);

    /* MLSL with a gradient-based INNER search: the enclosing block's line
     * is the one that counts. An INNER block takes no GRAD_STRATEGY, so
     * the journal has no line for one. */
    const auto mlsl = [](const std::string& inner_extra) {
        return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
               "    ALGO GN_MLSL_LDS\n    MAXEVAL 1\n    INNER\n"
               "      LIB NLOPT\n      ALGO LD_LBFGS\n" + inner_extra +
               "    ENDINNER\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const std::string inner_unset = journal_of(mlsl("") + observable);
    INFO(inner_unset);
    CHECK_THAT(line_with(inner_unset, "GRAD_STRATEGY"),
               Catch::Matchers::Contains("ANALYTICAL (resolved for PC_OBSERVABLE)"));
    CHECK(line_with(inner_unset, "Inner GRAD_STRATEGY").empty());
    CHECK(inner_unset.find("not used") == std::string::npos);

    /* Pins: GD_MLSL_LDS with a derivative-free INNER search obtains no
     * gradient, so the journal has no GRAD_STRATEGY line for it.
     * Before the review fix: "GRAD_STRATEGY : ANALYTICAL (resolved for
     * PC_OBSERVABLE)". */
    const std::string free_inner = journal_of(
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
        "    ALGO GD_MLSL_LDS\n    MAXEVAL 1\n    INNER\n      LIB NLOPT\n"
        "      ALGO LN_SBPLX\n    ENDINNER\n  ENDSTRATEGY\nENDOPTIMIZER\n" +
        observable);
    INFO(free_inner);
    CHECK(line_with(free_inner, "GRAD_STRATEGY").empty());

    /* PRESTAGE: the EXPLORE block's line; its INNER block has none. */
    const auto prestage = [](const std::string& explore) {
        return "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB DLIB\n"
               "    ALGO BFGS\n    MAXEVAL 1\n    GRAD_STRATEGY HYBRID\n"
               "    EXPLORE\n" + explore + "    ENDEXPLORE\n  ENDSTRATEGY\n"
               "ENDOPTIMIZER\n";
    };
    const std::string explore = journal_of(prestage(
        "      LIB NLOPT\n      ALGO GN_MLSL_LDS\n      MAXEVAL 1\n"
        "      INNER\n        LIB NLOPT\n        ALGO LD_LBFGS\n"
        "      ENDINNER\n") + observable);
    INFO(explore);
    CHECK_THAT(line_with(explore, "EXPLORE GRAD_STRATEGY"),
               Catch::Matchers::Contains("ANALYTICAL (resolved for PC_OBSERVABLE)"));
    CHECK(line_with(explore, "EXPLORE INNER GRAD_STRATEGY").empty());
    CHECK(explore.find("not used") == std::string::npos);
}

// test_pc_observable_parser.cpp
//
// Phase D4a (PC_OBSERVABLE redesign): the .hpotarget parser exposes
// PC_OBSERVABLE entries via the typed Observable interface. Verifies
// that a config line containing a cohesive_energy directive is parsed
// into an Observable with the expected fields, and that error cases
// (missing flags, malformed values, unknown kinds) are flagged.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/observables/observable_parser.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

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

/* Helper: write an .hpotarget snippet to a temp file and return its path. */
std::filesystem::path write_temp_config(const std::string& body) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path()
           / ("test_pc_observable_parser_" + std::to_string(++counter) + ".cfg");
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
        CHECK_THROWS_WITH(
            tadah::hpo::HPOTargets(ctx, cfg.string()),
            Catch::Matchers::Contains("PC_OBSERVABLE")
            && Catch::Matchers::Contains("NUMERICAL"));
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
     * lives inside STRATEGY per the config grammar. */
    {
        auto cfg = write_temp_config(R"(OPTIMIZER
  STRATEGY
    LIB NLOPT
    ALGO GN_MLSL_LDS
    MAXEVAL 1
    INNER
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
            Catch::Matchers::Contains("PC_OBSERVABLE")
            && Catch::Matchers::Contains("NUMERICAL")
            && Catch::Matchers::Contains("INNER"));
        std::filesystem::remove(cfg);
    }
}

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

    /* CERES:LBFGS rejected (gradient-based, no special-case). */
    {
        auto cfg = write_temp_config(
            std::string("OPTIMIZER\nLIB CERES\nALGO LBFGS\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n")
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
            + "    MAXEVAL 1\n    INNER\n      ALGO LD_LBFGS\n    ENDINNER\n"
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

    /* 2026-05-13 directive: PC_OBSERVABLE is mutually exclusive with
     * PC_LAMMPS. The post-parse validation in HPOTargets::read fires
     * a clear runtime_error mentioning both counts. */
    CHECK_THROWS_WITH(
        tadah::hpo::HPOTargets(ctx, cfg.string()),
        Catch::Matchers::Contains("PC_OBSERVABLE")
        && Catch::Matchers::Contains("PC_LAMMPS")
        && Catch::Matchers::Contains("mutually exclusive"));

    std::filesystem::remove(cfg);
    std::filesystem::remove(script_path);
}

// test_failgrade_parse.cpp
//
// Graded-FAILSCORE config surface (FAILGRADE Tiers 2+3):
//   - PC_LAMMPS --failvar <name> [<scale>] parsing + validation;
//   - FAILSCORE_GRADING ON|OFF root key;
//   - PC_FAILPROBE line: flag stripping (--gate/--s0), probe marking,
//     and every parse-time rejection (varloss / missing outvar /
//     hotvar / failvar / duplicate probe).
// Design: INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md sections 4.6-4.8.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

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

/* Unique-across-processes temp names: ctest runs each TEST_CASE as its own
 * process and `ctest -j` runs them concurrently (see the note in
 * test_pc_observable_parser.cpp). */
std::filesystem::path temp_path(const std::string& stem) {
    static int counter = 0;
    return std::filesystem::temp_directory_path()
         / ("test_failgrade_parse_" + stem + "_"
            + std::to_string(::getpid()) + "_"
            + std::to_string(++counter));
}

std::filesystem::path write_file(const std::string& stem,
                                 const std::string& body) {
    auto p = temp_path(stem);
    std::ofstream f(p);
    f << body;
    return p;
}

/* A minimal LAMMPS script defining every variable the test configs
 * request: LammpsSim reads the script at construction and statically
 * validates that all --varloss/--outvar/--failvar names are defined as
 * equal-style variables. */
std::filesystem::path write_script() {
    return write_file("script.in",
                      "clear\nrun 0\n"
                      "variable A equal 0.0\n"
                      "variable B equal 0.0\n"
                      "variable S equal 0.0\n"
                      "variable fv equal 0.0\n"
                      "variable fv_p equal 0.0\n"
                      "variable fv_q equal 0.0\n");
}

std::string config_head() {
    return "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
           "LOSS L2\nFAILSCORE 1e6\n";
}

}  // namespace

TEST_CASE("PC_LAMMPS --failvar: name-only and name+scale parse onto "
          "LammpsSim",
          "[failgrade][parse][failvar]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();

    auto cfg = write_file("cfg", config_head()
        + "PC_LAMMPS --script " + script.string()
        + " --outvar A --varloss A 1.0 1.0 --failvar fv\n"
        + "PC_LAMMPS --script " + script.string()
        + " --outvar B --varloss B 1.0 1.0 --failvar fv_p 2.5 --failvar fv_q\n");
    tadah::hpo::HPOTargets trg(ctx, cfg.string());

    REQUIRE(trg.LS.size() == 2);
    REQUIRE(trg.LS[0].failvars.size() == 1);
    CHECK(trg.LS[0].failvars[0].name == "fv");
    CHECK(trg.LS[0].failvars[0].scale == Approx(1.0));
    REQUIRE(trg.LS[1].failvars.size() == 2);
    CHECK(trg.LS[1].failvars[0].name == "fv_p");
    CHECK(trg.LS[1].failvars[0].scale == Approx(2.5));
    CHECK(trg.LS[1].failvars[1].name == "fv_q");
    CHECK(trg.LS[1].failvars[1].scale == Approx(1.0));
    CHECK_FALSE(trg.LS[0].is_failprobe);
}

TEST_CASE("PC_LAMMPS --failvar: rejections", "[failgrade][parse][failvar]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    const std::string base = config_head()
        + "PC_LAMMPS --script " + script.string()
        + " --outvar A --varloss A 1.0 1.0 ";

    SECTION("missing name") {
        auto cfg = write_file("cfg", base + "--failvar\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("flag where the name should be") {
        auto cfg = write_file("cfg", base + "--failvar --outvar X\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("non-positive scale") {
        auto cfg = write_file("cfg", base + "--failvar fv 0.0\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("failvar not defined in the script (static validation)") {
        /* A typo'd failvar must be a parse-time error, not a silent
         * "no violation ever" at runtime. */
        auto cfg = write_file("cfg", base + "--failvar fv_typo\n");
        REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, cfg.string()),
                            Catch::Contains("fv_typo"));
    }
}

TEST_CASE("FAILSCORE_GRADING: ON/OFF/garbage", "[failgrade][parse]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    const std::string sim = "PC_LAMMPS --script " + script.string()
                          + " --outvar A --varloss A 1.0 1.0\n";

    {
        auto cfg = write_file("cfg", config_head()
                              + "FAILSCORE_GRADING ON\n" + sim);
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        CHECK(trg.FAILSCORE_GRADING);
    }
    {
        auto cfg = write_file("cfg", config_head()
                              + "FAILSCORE_GRADING OFF\n" + sim);
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        CHECK_FALSE(trg.FAILSCORE_GRADING);
    }
    {   /* default is OFF */
        auto cfg = write_file("cfg", config_head() + sim);
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        CHECK_FALSE(trg.FAILSCORE_GRADING);
    }
    {
        auto cfg = write_file("cfg", config_head()
                              + "FAILSCORE_GRADING yes\n" + sim);
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
}

TEST_CASE("PC_FAILPROBE: valid line, flags stripped, sim marked",
          "[failgrade][parse][failprobe]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    const std::string sim = "PC_LAMMPS --script " + script.string()
                          + " --outvar A --varloss A 1.0 1.0\n";

    SECTION("defaults: gate off (inf), s0 = 1") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string() + " --outvar S\n");
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        REQUIRE(trg.failprobe_idx >= 0);
        REQUIRE(static_cast<std::size_t>(trg.failprobe_idx) < trg.LS.size());
        const auto& probe = trg.LS[trg.failprobe_idx];
        CHECK(probe.is_failprobe);
        CHECK(trg.failprobe_svar == "S");
        CHECK_FALSE(std::isfinite(trg.failprobe_gate));   /* gate off */
        CHECK(trg.failprobe_s0 == Approx(1.0));
        CHECK(probe.n_varloss.empty());                   /* no loss shaping */
    }
    SECTION("--gate and --s0 parsed and stripped") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --gate 50 --s0 0.5\n");
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        CHECK(trg.failprobe_gate == Approx(50.0));
        CHECK(trg.failprobe_s0 == Approx(0.5));
    }
    SECTION("failprobe_idx is the probe's configuration position") {
        /* LS keeps configuration order and failprobe_idx is recorded
         * against it at parse time. Declare the probe FIRST and mix hot +
         * cold sims after it: the index must point at the probe (an index
         * that pointed elsewhere was a real bug: the probe journal named
         * vac_bcc_P_fast.in and a mis-dispatched wave-1 "probe" corrupted
         * every eval). */
        auto cfg = write_file("cfg", config_head()
            + "PC_FAILPROBE --script " + script.string() + " --outvar S\n"
            + "PC_LAMMPS --script " + script.string()
            + " --outvar A --varloss A 1.0 1.0 --hotvar h 0 1 0.5\n"
            + "PC_LAMMPS --script " + script.string()
            + " --outvar B --varloss B 1.0 1.0\n");
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        REQUIRE(trg.failprobe_idx == 0);
        REQUIRE(trg.LS.size() == 3);
        CHECK(trg.LS[trg.failprobe_idx].is_failprobe);
        CHECK(trg.LS[1].n_varloss == std::vector<std::string>{"A"});
        CHECK(trg.LS[2].n_varloss == std::vector<std::string>{"B"});
        /* exactly one probe entry in LS */
        int n_probe = 0;
        for (const auto& s : trg.LS) n_probe += s.is_failprobe ? 1 : 0;
        CHECK(n_probe == 1);
    }
}

TEST_CASE("PC_FAILPROBE: rejections", "[failgrade][parse][failprobe]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    const std::string sim = "PC_LAMMPS --script " + script.string()
                          + " --outvar A --varloss A 1.0 1.0\n";
    const std::string probe_ok =
        "PC_FAILPROBE --script " + script.string() + " --outvar S\n";

    SECTION("must not declare --varloss") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --varloss S 0.0 1.0\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("requires --outvar") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string() + "\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("must not declare --hotvar") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --hotvar h 1 2 1.5\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("must not declare --failvar") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --failvar fv\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("only one probe allowed") {
        auto cfg = write_file("cfg", config_head() + sim
                              + probe_ok + probe_ok);
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("--gate without a value") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --gate\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("--s0 must be positive") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --s0 0\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
    SECTION("--s0 must be finite") {
        auto cfg = write_file("cfg", config_head() + sim
            + "PC_FAILPROBE --script " + script.string()
            + " --outvar S --s0 inf\n");
        REQUIRE_THROWS_AS(tadah::hpo::HPOTargets(ctx, cfg.string()),
                          std::runtime_error);
    }
}

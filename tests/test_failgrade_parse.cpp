// test_failgrade_parse.cpp
//
// Graded-FAILSCORE config surface (FAILGRADE Tiers 2+3):
//   - PC_LAMMPS --failvar <name> [<scale>] parsing + validation;
//   - FAILSCORE_GRADING ON|OFF root key;
//   - PC_FAILPROBE line: flag stripping (--gate/--s0), probe marking,
//     and every parse-time rejection (varloss / missing outvar /
//     hotvar / failvar / duplicate probe).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/phase_specs.h>

#include "flush_subnormals.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
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

/* Every file this process writes, removed when it exits (they were left
 * behind in TMPDIR, one set per test process). */
struct WrittenFiles {
    std::vector<std::filesystem::path> paths;
    ~WrittenFiles() {
        for (const auto& p : paths) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
    }
};

std::filesystem::path write_file(const std::string& stem,
                                 const std::string& body) {
    static WrittenFiles written;
    auto p = temp_path(stem);
    written.paths.push_back(p);
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

/* A gradient-free optimiser: a PC_LAMMPS line refuses a gradient-based one,
 * DLIB:BFGS among them (the head named DLIB:BFGS). Before the fix: the
 * cases pass as well (a pin: the optimiser does not enter what they test). */
std::string config_head() {
    return "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
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

TEST_CASE("FAILSCORE_GRADING: every optimiser spec carries it, through the "
          "override and every spec copy", "[failgrade][parse][grad_provider]")
{
    /* The finite differences read FAILSCORE_GRADING from the optimiser
     * spec (OptimizerSpec::failscore_grading, gradient_provider.h): at a
     * graded failure they take the graded differences, where a failed loss
     * at the point fails the gradient without grading.
     * Pins: the spec of a PRESTAGE run, its INNER block, the EXPLORE spec
     * and that one's INNER block carry FAILSCORE_GRADING as the file sets
     * it (ON, OFF, the default OFF); set_failscore_grading (the
     * TADAH_FAILSCORE_GRADING override of hpo_run) sets FAILSCORE_GRADING
     * and every one of them; the hand-written copy constructor and copy
     * assignment keep it (the MULTISTART starts, the PRESTAGE refinement
     * and each RESTART round, RestartPolicy::run, are copies), and so do
     * multistart_start_spec and prestage_refinement_spec.
     * Before the fix: the specs had no such flag (the gradient read none);
     * built with the field declared but neither set nor copied, every
     * spec reads false.
     * The file has no LOSS line: no line of it reads one, and such a line
     * is refused (it had LOSS L2; the case passes before that fix as well,
     * a pin). */
    tadah::core::PeriodicTable::initialize();
    tadah::core::Context ctx;
    const auto config = [](const std::string& grading) {
        return std::string(
                   "OPTIMIZER\n"
                   "STRATEGY\n"
                   "  TYPE PRESTAGE\n"
                   "  LIB NLOPT\n"
                   "  ALGO AUGLAG\n"
                   "  MAXEVAL 50\n"
                   "  TOPK 2\n"
                   "  INNER\n"
                   "    LIB NLOPT\n"
                   "    ALGO LD_LBFGS\n"
                   "  ENDINNER\n"
                   "  EXPLORE\n"
                   "    LIB NLOPT\n"
                   "    ALGO G_MLSL_LDS\n"
                   "    MAXEVAL 80\n"
                   "    INNER\n"
                   "      LIB NLOPT\n"
                   "      ALGO LD_LBFGS\n"
                   "    ENDINNER\n"
                   "  ENDEXPLORE\n"
                   "ENDSTRATEGY\n"
                   "ENDOPTIMIZER\n"
                   "FAILSCORE 1e6\n") +
               grading;
    };
    /* Every spec the run starts from carries `on`. */
    const auto all = [](const tadah::hpo::OptimizerSpec& sp, bool on) {
        REQUIRE(sp.inner);
        REQUIRE(sp.prestage);
        REQUIRE(sp.prestage->inner);
        CHECK(sp.failscore_grading == on);
        CHECK(sp.inner->failscore_grading == on);
        CHECK(sp.prestage->failscore_grading == on);
        CHECK(sp.prestage->inner->failscore_grading == on);
    };
    SECTION("FAILSCORE_GRADING ON") {
        auto cfg = write_file("cfg", config("FAILSCORE_GRADING ON\n"));
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        CHECK(trg.FAILSCORE_GRADING);
        all(trg.spec, true);

        const tadah::hpo::OptimizerSpec copy(trg.spec);
        all(copy, true);
        tadah::hpo::OptimizerSpec assigned;
        assigned = trg.spec;
        all(assigned, true);

        tadah::hpo::MultiStartPlan plan;
        plan.starts = 1;
        CHECK(tadah::hpo::multistart_start_spec(trg.spec, plan, 0)
                  .failscore_grading);
        CHECK(tadah::hpo::prestage_refinement_spec(trg.spec, {{0.5}})
                  .failscore_grading);

        trg.set_failscore_grading(false);
        CHECK_FALSE(trg.FAILSCORE_GRADING);
        all(trg.spec, false);
    }
    SECTION("FAILSCORE_GRADING OFF, and the default") {
        for (const std::string line : {"FAILSCORE_GRADING OFF\n", ""}) {
            CAPTURE(line);
            auto cfg = write_file("cfg", config(line));
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            CHECK_FALSE(trg.FAILSCORE_GRADING);
            all(trg.spec, false);
            trg.set_failscore_grading(true);
            CHECK(trg.FAILSCORE_GRADING);
            all(trg.spec, true);
            all(tadah::hpo::OptimizerSpec(trg.spec), true);
        }
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

namespace {

/* The LOSS FUNCTION section of the journal HPOLog writes for `trg`. */
std::string loss_section(const tadah::hpo::HPOTargets& trg) {
    auto& log = tadah::hpo::HPOLog::instance();
    if (log.is_open()) log.finish("closed");
    const auto path = temp_path("journal.log");
    log.open(path.string(), "test", "/tmp", 1);
    log.write_loss_section(trg);
    log.finish("TEST");
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    const std::string journal = ss.str();
    const auto at = journal.find("-- LOSS FUNCTION");
    REQUIRE(at != std::string::npos);
    return journal.substr(at, journal.find("\n-- ", at + 1) - at);
}

/* The section's line that holds `label` ("" when none does). */
std::string line_of(const std::string& text, const std::string& label) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(label) != std::string::npos) return line;
    return "";
}

}  // namespace

TEST_CASE("The journal's Fail score line names what scores FAILSCORE, "
          "under FAILSCORE_GRADING OFF and ON", "[failgrade][hpo_log]")
{
    /* Pins: under FAILSCORE_GRADING OFF the LOSS FUNCTION section's Fail
     * score line names the failures that score FAILSCORE in this run: a
     * NaN/Inf loss, an exception, a skip cap, a failed term of the run --
     * a failed PC_LAMMPS simulation when it has PC_LAMMPS lines, a failed
     * PC_OBSERVABLE term when it has PC_OBSERVABLE lines (the two cannot
     * be combined), neither otherwise -- and a gate reject. Under ON it
     * names only the NaN/Inf loss and the exception, and a "Fail grading"
     * line names the same failures as scoring a graded loss in [0.999, 1)
     * x FAILSCORE (graded_eval_loss), as their rows in loss.tadah show.
     * Before the fix: one line whatever FAILSCORE_GRADING, "(the loss of a
     * failed evaluation: a NaN/Inf loss, an exception, a skip cap, a failed
     * script or a gate reject; the ceiling of every loss)", which under ON
     * named three failures that do not score it, and no "Fail grading"
     * line; then (until 2026-10-12) every run's lines named a failed
     * PC_LAMMPS simulation and none a failed PC_OBSERVABLE term, which
     * scores FAILSCORE, or band B under ON, since an evaluation with one
     * fails: a PC_OBSERVABLE run's journal named a cause it cannot have
     * and left out the one it had. Then the lines named "a skip cap" and
     * "a rejection by BUMPGATE, RHOGATE or RHOBARGATE" and left out the
     * other rejections of a potential as unusable that the startup line of
     * FAILSCORE_GRADING ON lists, from the same list of the run
     * (unusable_potential_causes): a PC_FAILPROBE rejection, a refused fit
     * and a potential LAMMPS cannot build. Each line names them as "a
     * potential rejected as unusable (by ...)": every case fails on it. */
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    struct Case {
        std::string name, lines, causes;
    };
    const std::string rejected = "a potential rejected as unusable (by ";
    const std::string d_list =
        "a --skip-above or --skip-rel cap, BUMPGATE, RHOGATE or RHOBARGATE, "
        "a refused fit";
    const std::string lammps =
        "PC_ERMSE 0 1\nPC_LAMMPS --script " + script.string() +
        " --outvar A --varloss A 1.0 1.0\n";
    const std::vector<Case> cases = {
        {"PC_ERMSE alone", "PC_ERMSE 0 1\n",
         rejected + "a --skip-above or --skip-rel cap, BUMPGATE, RHOGATE or "
                    "RHOBARGATE, or a refused fit)"},
        {"PC_LAMMPS", lammps,
         "a failed PC_LAMMPS simulation or " + rejected + d_list +
             ", or a potential LAMMPS cannot build)"},
        {"PC_LAMMPS and PC_FAILPROBE",
         lammps + "PC_FAILPROBE --script " + script.string() +
             " --outvar S\n",
         "a failed PC_LAMMPS simulation or " + rejected + "PC_FAILPROBE, " +
             d_list + ", or a potential LAMMPS cannot build)"},
        {"PC_OBSERVABLE",
         "PC_ERMSE 0 1\nPC_OBSERVABLE cohesive_energy --species Ta "
         "--target 8.1 --weight 1 --hotvar a0 3.0 3.35 3.3\n",
         "a failed PC_OBSERVABLE term or " + rejected +
             "a --skip-above or --skip-rel cap, BUMPGATE, RHOGATE or "
             "RHOBARGATE, or a refused fit)"}};
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.name << ", FAILSCORE_GRADING OFF") {
            auto cfg = write_file("cfg", config_head() + c.lines);
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            REQUIRE_FALSE(trg.FAILSCORE_GRADING);
            const std::string loss = loss_section(trg);
            INFO(loss);
            CHECK_THAT(line_of(loss, "Fail score"),
                       Catch::EndsWith("1.0000e+06  (the loss of a failed "
                                       "evaluation: a NaN/Inf loss, an "
                                       "exception, " + c.causes + "; the "
                                       "ceiling of every loss)"));
            CHECK(line_of(loss, "Fail grading").empty());
        }
        DYNAMIC_SECTION(c.name << ", FAILSCORE_GRADING ON") {
            auto cfg = write_file("cfg", config_head() + c.lines +
                                             "FAILSCORE_GRADING ON\n");
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            REQUIRE(trg.FAILSCORE_GRADING);
            const std::string loss = loss_section(trg);
            INFO(loss);
            CHECK_THAT(line_of(loss, "Fail score"),
                       Catch::EndsWith("1.0000e+06  (the loss of an "
                                       "evaluation with a NaN/Inf loss or an "
                                       "exception; the ceiling of every "
                                       "loss)"));
            CHECK_THAT(line_of(loss, "Fail grading"),
                       Catch::EndsWith(": ON -- " + c.causes +
                                       " scores a graded loss in [0.999, 1) "
                                       "x Fail score"));
        }
    }
}

/* Pins: the refusal of a FAILSCORE too small for the four grading bands of
 * FAILSCORE_GRADING ON gives the value, the width of a band, the reason
 * and a bound that suffices (graded_failscore_refusal, which the startup
 * checks put at the FAILSCORE line: test_pc_lines_losses.cpp): "FAILSCORE
 * 1e-307: the four grading bands of FAILSCORE_GRADING ON, each 0.00025 x
 * FAILSCORE wide, are not representable at this magnitude; a FAILSCORE of
 * 1e-290 or more has them". */
TEST_CASE("FAILSCORE_GRADING: a FAILSCORE too small for the grading bands is "
          "refused with its value and a bound that suffices",
          "[failgrade][parse][failscore_bound]")
{
    CHECK(tadah::hpo::graded_failscore_refusal(1e-307) ==
          "FAILSCORE 1e-307: the four grading bands of FAILSCORE_GRADING "
          "ON, each 0.00025 x FAILSCORE wide, are not representable at this "
          "magnitude; a FAILSCORE of 1e-290 or more has them");
    CHECK(tadah::hpo::kGradedFailscoreSufficient == 1e-290);
}

/* Pins: the bound of that refusal holds where the refusal can happen, in a
 * process that flushes subnormal numbers to zero (the flush-to-zero and
 * denormals-are-zero bits set, as the start-up code of an executable linked
 * with -ffast-math sets them; FlushSubnormals sets them here):
 * graded_failscore_is_ordered is false for 1e-307, whose band widths are
 * subnormal, and true for every power of ten from kGradedFailscoreSufficient
 * (1e-290) to 1e308. In IEEE arithmetic it is true for every power of ten
 * from 1e-307, the smallest normal one, to 1e308: the parser refuses a
 * subnormal FAILSCORE, so no FAILSCORE is refused there.
 * Before the fix: passes as well (a pin of a bound that was right). The
 * check of the bound ran in the flags of the test alone, without flushing
 * (compiling with -ffast-math flushes nothing; neither test_hpo nor
 * test_tadah is linked with it), where every normal FAILSCORE has its
 * bands: with kGradedFailscoreSufficient set to 1e-307 it passed, and the
 * checks under FlushSubnormals fail. */
TEST_CASE("FAILSCORE_GRADING: every FAILSCORE from the stated bound has its "
          "grading bands where subnormal numbers are flushed to zero",
          "[failgrade][parse][failscore_bound]")
{
    using tadah::hpo::graded_failscore_is_ordered;
    using tadah::hpo::test::FlushSubnormals;
    using tadah::hpo::test::subnormals_are_flushed;
    const bool flushed_before = subnormals_are_flushed();
    if (flushed_before)
        WARN("this process flushes subnormal numbers to zero: the IEEE "
             "arithmetic is not checked");
    else
        for (int e = -307; e <= 308; ++e) {
            const double fs = std::stod("1e" + std::to_string(e));
            INFO("IEEE arithmetic, FAILSCORE 1e" << e);
            CHECK(graded_failscore_is_ordered(fs));
        }
    bool active = false, flushed = false;
    std::vector<int> unordered;
    std::vector<int> ordered_below_bound;
    {
        FlushSubnormals ftz;
        active = ftz.active();
        flushed = subnormals_are_flushed();
        for (int e = -307; e <= 308; ++e) {
            const double fs = std::stod("1e" + std::to_string(e));
            const bool ok = graded_failscore_is_ordered(fs);
            if (fs >= tadah::hpo::kGradedFailscoreSufficient && !ok)
                unordered.push_back(e);
            if (e == -307 && ok) ordered_below_bound.push_back(e);
        }
    }
    CHECK(subnormals_are_flushed() == flushed_before);   /* restored */
    if (!active) {
        WARN("no flush-to-zero control on this architecture: the bound is "
             "checked in IEEE arithmetic only");
        return;
    }
    REQUIRE(flushed);
    INFO("powers of ten from the bound without their bands (exponents)");
    CHECK(unordered.empty());
    INFO("1e-307 has its bands");
    CHECK(ordered_below_bound.empty());
}

/* Pins: the LOSS FUNCTION section of the journal opens with the loss
 * function of the metric lines, its name and parameters as the last LOSS
 * line gives them: "Metric loss function : HUBER 50" for LOSS HUBER 50,
 * "L2" for LOSS L2 (no parameter). A --varloss term and a PC_OBSERVABLE
 * term take the LOSS line above their line, so their lines give their own:
 * with LOSS L1, a PC_LAMMPS line with --varloss A 1.0 1.0, LOSS HUBER 50 and
 * PC_ERMSE 0 1, the section reads "Metric loss function : HUBER 50" and
 * "A : weight=1.00  target=1.000e+00  loss=L1"; a PC_OBSERVABLE line under
 * LOSS TUKEY 3 reads "weight=1.00  target=8.100e+00  loss=TUKEY 3  columns
 * ...". A file without a metric line has no "Metric loss function" line.
 * Before the fix: the first line read "Loss function : HUBER 50" (v1-dev
 * had no such line), given as the loss function of the run although the
 * terms above the last LOSS line take another, and the --varloss and
 * PC_OBSERVABLE lines named no loss function: the checks of the label and
 * of the terms' loss functions fail. */
TEST_CASE("The journal's LOSS FUNCTION section names the loss function of "
          "the metric lines and of every other term",
          "[failgrade][hpo_log][loss]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script();
    const std::string head =
        "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n";
    const std::string lammps = "PC_LAMMPS --script " + script.string() +
                               " --outvar A --varloss A 1.0 1.0\n";
    /* the line of the section whose label is @p label */
    auto labelled = [](const std::string& section, const std::string& label) {
        std::istringstream in(section);
        std::string line;
        while (std::getline(in, line)) {
            const auto colon = line.find(": ");
            if (colon == std::string::npos) continue;
            std::string l = line.substr(0, colon);
            l.erase(0, l.find_first_not_of(' '));
            l.erase(l.find_last_not_of(' ') + 1);
            if (l == label) return line;
        }
        return std::string();
    };
    SECTION("the metric lines take the last LOSS line") {
        struct Case { std::string loss, shown; };
        for (const Case& c : std::vector<Case>{{"LOSS HUBER 50", "HUBER 50"},
                                               {"LOSS L2", "L2"}}) {
            INFO(c.loss);
            auto cfg = write_file("cfg", head + c.loss + "\nPC_ERMSE 0 1\n");
            tadah::hpo::HPOTargets trg(ctx, cfg.string());
            const std::string loss = loss_section(trg);
            INFO(loss);
            const std::string line = labelled(loss, "Metric loss function");
            CHECK_THAT(line, Catch::EndsWith(": " + c.shown));
            /* the first line of the section */
            std::istringstream in(loss);
            std::string first, second;
            std::getline(in, first);
            std::getline(in, second);
            CHECK(second == line);
            CHECK(labelled(loss, "Loss function").empty());
        }
    }
    SECTION("a --varloss term takes the LOSS line above its line") {
        auto cfg = write_file("cfg", head + "LOSS L1\n" + lammps +
                                         "LOSS HUBER 50\nPC_ERMSE 0 1\n");
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        const std::string loss = loss_section(trg);
        INFO(loss);
        CHECK_THAT(labelled(loss, "Metric loss function"),
                   Catch::EndsWith(": HUBER 50"));
        CHECK_THAT(labelled(loss, "A"),
                   Catch::EndsWith(": weight=1.00  target=1.000e+00  "
                                   "loss=L1"));
    }
    SECTION("a file without a metric line") {
        auto cfg = write_file("cfg", head + "LOSS L1\n" + lammps);
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        const std::string loss = loss_section(trg);
        INFO(loss);
        CHECK(labelled(loss, "Metric loss function").empty());
        CHECK(labelled(loss, "Loss function").empty());
        CHECK_THAT(labelled(loss, "A"), Catch::EndsWith("  loss=L1"));
    }
    SECTION("a PC_OBSERVABLE term takes the LOSS line above its line") {
        auto cfg = write_file(
            "cfg", head + "LOSS TUKEY 3\n"
                   "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
                   "--weight 1 --hotvar a0 2.5 4.5\n"
                   "LOSS HUBER 50\nPC_ERMSE 0 1\n");
        tadah::hpo::HPOTargets trg(ctx, cfg.string());
        REQUIRE(trg.observables.size() == 1u);
        const std::string loss = loss_section(trg);
        INFO(loss);
        CHECK_THAT(labelled(loss, "Metric loss function"),
                   Catch::EndsWith(": HUBER 50"));
        CHECK_THAT(loss, Catch::Contains(": weight=1.00  target=8.100e+00  "
                                         "loss=TUKEY 3  columns "));
    }
}

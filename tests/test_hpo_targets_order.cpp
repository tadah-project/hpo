// test_hpo_targets_order.cpp
//
// PC_LAMMPS lines keep their configuration order inside HPOTargets::LS:
// the outvar/loss column headers, the per-evaluation rows and the
// potential-file header are all emitted in LS order and are only
// identifiable by position once labels are truncated, and failprobe_idx is
// recorded against that order at parse time. Nothing reorders LS after
// parsing (24 lines: beyond the size where a sort could still happen to
// keep an all-equal order).
//
// PC_LAMMPS --ncpu (the per-sim rank count of the legacy MPI build, removed
// in 1.6.0) is rejected at parse time with the removal hint and the line
// number, rather than being ignored or passed to LAMMPS. So is any other
// option that starts with "--" and is not one of HPO's: LAMMPS has none.
// A single-dash option LAMMPS refuses is an error at its line too, with
// LAMMPS' message.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>

#include <algorithm>
#include <cstdio>
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

std::filesystem::path temp_path(const std::string& stem) {
    static int counter = 0;
    return std::filesystem::temp_directory_path()
         / ("test_hpo_targets_order_" + stem + "_"
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

std::string var_name(int k) {          // k is 1-based
    char buf[8];
    std::snprintf(buf, sizeof(buf), "V%02d", k);
    return buf;
}

/* A minimal LAMMPS script defining every variable the config lines
 * request (LammpsSim validates the names at construction). */
std::filesystem::path write_script(int n) {
    std::string body = "clear\nrun 0\n";
    for (int k = 1; k <= n; ++k)
        body += "variable " + var_name(k) + " equal 0.0\n";
    return write_file("script.in", body);
}

std::string config_head() {
    return "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
           "LOSS L2\nFAILSCORE 1e6\n"
           "OPTIM WATOM (1) 0.1 5.0\n";
}

constexpr int kLines = 24;   // > 16: beyond libstdc++'s insertion-sort threshold

}  // namespace

TEST_CASE("HPOTargets: PC_LAMMPS lines keep configuration order",
          "[hpo_targets][order]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(kLines);

    std::string cfg = config_head();
    for (int k = 1; k <= kLines; ++k)
        cfg += "PC_LAMMPS --script " + script.string()
             + " --varloss " + var_name(k) + " 1.0 1.0\n";
    const auto cfg_path = write_file("cfg_equal", cfg);

    tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
    REQUIRE(trg.LS.size() == static_cast<std::size_t>(kLines));

    std::vector<std::string> observed, expected;
    for (int k = 1; k <= kLines; ++k) {
        expected.push_back(var_name(k));
        REQUIRE(trg.LS[k - 1].n_varloss.size() == 1);
        observed.push_back(trg.LS[k - 1].n_varloss[0]);
    }
    CHECK(observed == expected);
}

TEST_CASE("HPOTargets: PC_FAILPROBE keeps its configuration position",
          "[hpo_targets][order][failprobe]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(kLines);

    /* the probe sits between PC_LAMMPS lines 12 and 13 */
    const int probe_after = 12;
    std::string cfg = config_head();
    for (int k = 1; k <= kLines; ++k) {
        cfg += "PC_LAMMPS --script " + script.string()
             + " --varloss " + var_name(k) + " 1.0 1.0\n";
        if (k == probe_after)
            cfg += "PC_FAILPROBE --script " + script.string()
                 + " --outvar " + var_name(1) + "\n";
    }
    const auto cfg_path = write_file("cfg_probe", cfg);

    tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
    REQUIRE(trg.LS.size() == static_cast<std::size_t>(kLines + 1));
    CHECK(trg.failprobe_idx == probe_after);
    std::vector<std::string> observed, expected;
    for (int k = 1; k <= kLines; ++k) expected.push_back(var_name(k));
    for (std::size_t i = 0; i < trg.LS.size(); ++i) {
        const auto& sim = trg.LS[i];
        CHECK(sim.is_failprobe == (static_cast<int>(i) == probe_after));
        if (sim.is_failprobe) continue;
        REQUIRE(sim.n_varloss.size() == 1);
        observed.push_back(sim.n_varloss[0]);
    }
    CHECK(observed == expected);
}

TEST_CASE("HPOTargets: PC_LAMMPS --ncpu is rejected with the removal hint",
          "[hpo_targets][parse][ncpu]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(2);
    const std::string ok_line = "PC_LAMMPS --script " + script.string()
                              + " --varloss " + var_name(1) + " 1.0 1.0";

    /* `--ncpu 1` was the default: it is rejected too, like any use. */
    struct Case { std::string name, line; };
    const std::vector<Case> cases = {
        {"--ncpu 2 on PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --varloss "
             + var_name(2) + " 1.0 1.0 --ncpu 2"},
        {"--ncpu 1 on PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --ncpu 1 --varloss "
             + var_name(2) + " 1.0 1.0"},
        {"--ncpu without a value",
         "PC_LAMMPS --script " + script.string() + " --varloss "
             + var_name(2) + " 1.0 1.0 --ncpu"},
        {"--ncpu on PC_FAILPROBE",
         "PC_FAILPROBE --script " + script.string() + " --outvar "
             + var_name(2) + " --ncpu 4"},
    };
    for (const auto& c : cases) {
        INFO(c.name);
        const std::string head = config_head() + ok_line + "\n";
        const long bad_line = std::count(head.begin(), head.end(), '\n') + 1;
        const auto cfg_path = write_file("cfg_ncpu", head + c.line + "\n");
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
            FAIL_CHECK("--ncpu was accepted");
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            CHECK_THAT(msg, Catch::Contains(
                "Error on line " + std::to_string(bad_line) + ":"));
            CHECK_THAT(msg, Catch::Contains(
                "--ncpu was removed in 1.6.0 (legacy MPI build)"));
            CHECK_THAT(msg, Catch::Contains("THREADS / PC_ISOLATION"));
            CHECK_THAT(msg, Catch::Contains("Line: " + c.line));
        }
    }
}

/* Pins: an option that starts with "--" and is not one of HPO's (a typo
 * such as --hotvars, or PC_FAILPROBE's --gate on a PC_LAMMPS line) is an
 * error at its line naming HPO's options, with the closest one when it is
 * a near miss; on a PC_FAILPROBE line the list and the closest one are the
 * options a probe line takes: --gate and --s0, and not --varloss, --hotvar
 * or --failvar, which it refuses. LAMMPS has no "--" option
 * (src/lammps.cpp at the pinned commit).
 * Before the fix: the constructor accepted the --hotvars, --gate and --foo
 * lines and passed the token and its values to LAMMPS as command-line
 * arguments; every evaluation then failed ("ERROR: Invalid command-line
 * argument: --hotvars") and the run spent its whole MAXEVAL before ending
 * with no usable loss. The --varlos line, which then had no --varloss, was
 * refused with "Error on line 10: LammpsSim: --varloss variable is not
 * defined.".
 * Before the review fix: the PC_FAILPROBE list also named --varloss,
 * --hotvar and --failvar, and --hotvars there was answered "did you mean
 * '--hotvar'?", which the probe line refuses ("PC_FAILPROBE must not
 * declare --hotvar/--failvar"). */
TEST_CASE("HPOTargets: an unknown -- option on a PC_LAMMPS or PC_FAILPROBE "
          "line is rejected at its line", "[hpo_targets][parse][lammps_sim]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(2);
    const std::string ok_line = "PC_LAMMPS --script " + script.string()
                              + " --varloss " + var_name(1) + " 1.0 1.0";
    const std::string hpo_options =
        "--script, --varloss, --invar, --outvar, --failscore, --maxtime, "
        "--hotvar and --failvar";
    const std::string probe_options =
        "--script, --invar, --outvar, --failscore, --maxtime, --gate and "
        "--s0";
    const std::string lammps_note =
        "; LAMMPS command-line options start with a single '-'";

    struct Case { std::string name, line, message; };
    const std::vector<Case> cases = {
        {"--hotvars on PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --varloss "
             + var_name(2) + " 1.0 1.0 --hotvars a0 1 2",
         "PC_LAMMPS: unknown option '--hotvars' (did you mean '--hotvar'?): "
         "HPO's options are " + hpo_options + lammps_note},
        {"--varlos on PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --varlos "
             + var_name(2) + " 1 1",
         "PC_LAMMPS: unknown option '--varlos' (did you mean '--varloss'?): "
         "HPO's options are " + hpo_options + lammps_note},
        {"--gate on PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --varloss "
             + var_name(2) + " 1.0 1.0 --gate 0.5",
         "PC_LAMMPS: unknown option '--gate' (a PC_FAILPROBE option): HPO's "
         "options are " + hpo_options + lammps_note},
        {"--foo on PC_FAILPROBE",
         "PC_FAILPROBE --script " + script.string() + " --outvar "
             + var_name(2) + " --foo",
         "PC_FAILPROBE: unknown option '--foo': HPO's options are " +
             probe_options + lammps_note},
        {"--hotvars on PC_FAILPROBE",
         "PC_FAILPROBE --script " + script.string() + " --outvar "
             + var_name(2) + " --s0 0.5 --hotvars h 1 2",
         "PC_FAILPROBE: unknown option '--hotvars': HPO's options are " +
             probe_options + lammps_note},
    };
    for (const auto& c : cases) {
        INFO(c.name);
        const std::string head = config_head() + ok_line + "\n";
        const long bad_line = std::count(head.begin(), head.end(), '\n') + 1;
        const auto cfg_path = write_file("cfg_unknown_option",
                                         head + c.line + "\n");
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
            FAIL_CHECK("the unknown option was accepted");
        } catch (const std::runtime_error& e) {
            CHECK(std::string(e.what()) ==
                  "Error on line " + std::to_string(bad_line) + ": " +
                      c.message + "\nLine: " + c.line);
        }
    }
}

/* Pins: a single-dash option LAMMPS refuses on a PC_LAMMPS or PC_FAILPROBE
 * line (the typo -sff) is an error at its line: "Error on line N:
 * PC_LAMMPS: LAMMPS cannot start with this line's command-line options: "
 * (PC_FAILPROBE on a probe line), LAMMPS' own message ("ERROR: Invalid
 * command-line argument: -sff (...)"), then "Line: " and the line.
 * Before the fix: the constructor accepted both lines, and every
 * evaluation then failed to create its LAMMPS instance (FAIL_CHECK
 * fires). */
TEST_CASE("HPOTargets: a command-line option LAMMPS refuses is rejected at "
          "its line", "[hpo_targets][parse][lammps_sim]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(2);
    const std::string ok_line = "PC_LAMMPS --script " + script.string()
                              + " --varloss " + var_name(1) + " 1.0 1.0";
    struct Case { std::string kind, line; };
    const std::vector<Case> cases = {
        {"PC_LAMMPS",
         "PC_LAMMPS --script " + script.string() + " --varloss "
             + var_name(2) + " 1.0 1.0 -sff omp"},
        {"PC_FAILPROBE",
         "PC_FAILPROBE --script " + script.string() + " --outvar "
             + var_name(2) + " -sff omp"},
    };
    for (const auto& c : cases) {
        INFO(c.kind);
        const std::string head = config_head() + ok_line + "\n";
        const long bad_line = std::count(head.begin(), head.end(), '\n') + 1;
        const auto cfg_path = write_file("cfg_lammps_option",
                                         head + c.line + "\n");
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
            FAIL_CHECK("the option LAMMPS refuses was accepted");
        } catch (const std::runtime_error& e) {
            CHECK_THAT(std::string(e.what()),
                       Catch::StartsWith(
                           "Error on line " + std::to_string(bad_line) +
                           ": " + c.kind + ": LAMMPS cannot start with this "
                           "line's command-line options: ERROR: Invalid "
                           "command-line argument: -sff (") &&
                       Catch::EndsWith(")\nLine: " + c.line));
        }
    }
}

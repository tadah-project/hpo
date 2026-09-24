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
// in 1.5.1) is rejected at parse time with the removal hint and the line
// number, rather than being ignored or passed to LAMMPS.

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

std::filesystem::path write_file(const std::string& stem,
                                 const std::string& body) {
    auto p = temp_path(stem);
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
    return "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
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
                "--ncpu was removed in 1.5.1 (legacy MPI build)"));
            CHECK_THAT(msg, Catch::Contains("THREADS / PC_ISOLATION"));
            CHECK_THAT(msg, Catch::Contains("Line: " + c.line));
        }
    }
}

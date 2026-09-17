// test_hpo_targets_order.cpp
//
// PC_LAMMPS lines must keep their configuration order inside
// HPOTargets::LS. The ncpu ordering applied after parsing exists for
// scheduling only; sims with equal ncpu (the default for every line) have
// to stay in the order they were written, because the outvar/loss column
// headers, the per-evaluation rows and the potential-file header are all
// emitted in LS order and are only identifiable by position once labels
// are truncated. A comparator on an all-equal key gives an unspecified
// permutation under std::sort for more than 16 elements.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
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

TEST_CASE("HPOTargets keeps PC_LAMMPS configuration order for equal ncpu",
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

TEST_CASE("HPOTargets orders PC_LAMMPS lines by descending ncpu and keeps "
          "configuration order within equal ncpu",
          "[hpo_targets][order]")
{
    tadah::core::PeriodicTable::initialize();
    auto ctx = make_context_for_targets();
    const auto script = write_script(kLines);

    /* three lines ask for 2 cpus; every other line keeps the default 1 */
    const std::vector<int> wide = {5, 13, 20};
    std::string cfg = config_head();
    for (int k = 1; k <= kLines; ++k) {
        cfg += "PC_LAMMPS --script " + script.string()
             + " --varloss " + var_name(k) + " 1.0 1.0";
        for (int w : wide) if (w == k) cfg += " --ncpu 2";
        cfg += "\n";
    }
    const auto cfg_path = write_file("cfg_mixed", cfg);

    tadah::hpo::HPOTargets trg(ctx, cfg_path.string());
    REQUIRE(trg.LS.size() == static_cast<std::size_t>(kLines));

    std::vector<std::string> expected;
    for (int w : wide) expected.push_back(var_name(w));
    for (int k = 1; k <= kLines; ++k) {
        bool is_wide = false;
        for (int w : wide) is_wide |= (w == k);
        if (!is_wide) expected.push_back(var_name(k));
    }
    std::vector<std::string> observed;
    for (const auto& sim : trg.LS) {
        REQUIRE(sim.n_varloss.size() == 1);
        observed.push_back(sim.n_varloss[0]);
    }
    CHECK(observed == expected);
}

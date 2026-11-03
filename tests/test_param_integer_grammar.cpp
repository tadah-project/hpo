// test_param_integer_grammar.cpp
//
// A PARAM key that takes a whole number -- integral in the target-file
// registry (TADAH:GRID's GRID_POINTS, the whole-number options of
// ASA:ASA, CERES' iteration caps, diff_method), a count of the annealers
// (anneal_keys.toml)
// -- reads the integer syntax of every whole-number key of the target file:
// decimal digits, with a '-' for a negative value. A value written with an
// exponent, a '+' or a decimal point is refused when tadah hpo starts, at
// the END line of its optimiser, with the key's message and the value as
// the file writes it, as a top-level integer key (MAXEVAL 2e1) is refused
// when the file is read.
#include "catch2/catch.hpp"
#include "target_file_text.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

/* The smallest configuration whose model builds (validate_hpo_setup builds
 * the model of the -c configuration). */
Context make_ctx()
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP 2 2 H H");
    c.add("CGRID2B", "1.0 2.0");
    c.add("SGRID2B", "0.5 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    c.add("OALGO", 4);
    c.add("LAMBDA", 1e-3);
    return c;
}

/* A PLAIN search of LIB:ALGO with the line `param` (lines 1-2: an OPTIM
 * dimension and a loss; the block closes on line 11). */
std::string plain(const std::string& lib, const std::string& algo,
                  const std::string& param)
{
    return "OPTIM CGRID2B (1) 0.5 1.5\n"   // 1
           "PC_ERMSE 0 1\n"                // 2
           "OPTIMIZER\n"                   // 3
           "  STRATEGY\n"                  // 4
           "    TYPE PLAIN\n"              // 5
           "    LIB " + lib + "\n"         // 6
           "    ALGO " + algo + "\n"       // 7
           "    MAXEVAL 20\n"              // 8
           "    " + param + "\n"           // 9
           "  ENDSTRATEGY\n"               // 10
           "ENDOPTIMIZER\n";               // 11
}

/* What the parse and the startup checks of tadah hpo make of `text`: ""
 * when they accept it, else the refusal, with the target file written as
 * <target> (hpo_test::as_target). */
std::string refusal(const std::string& text)
{
    tadah::core::PeriodicTable::initialize();
    static unsigned counter = 0;
    const fs::path file =
        fs::temp_directory_path() /
        ("tadah_param_integer_" + std::to_string(::getpid()) + "_" +
         std::to_string(counter++) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".hpo");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << text;
    }
    Context ctx = make_ctx();
    std::string what;
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        const fs::path journal = fs::path(file.string() + ".log");
        log.open(journal.string(), "test", "test_cwd", 0);
        std::ostringstream sink;
        std::streambuf* old_err = std::cerr.rdbuf(sink.rdbuf());
        std::streambuf* old_out = std::cout.rdbuf(sink.rdbuf());
        try {
            HPOTargets trg(ctx, file.string());
            tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
        } catch (const std::exception& e) {
            what = hpo_test::as_target(e.what(), file.string());
        }
        std::cerr.rdbuf(old_err);
        std::cout.rdbuf(old_out);
        log.finish("test-end");
        log.reset_for_test();
        std::error_code ec;
        fs::remove(journal, ec);
    }
    std::error_code ec;
    fs::remove(file, ec);
    return what;
}

/* The refusal at the END line of the block of the STRATEGY-level optimiser
 * (line 11). */
std::string at_end(const std::string& what)
{
    return "line 11 of <target>: the STRATEGY-level optimiser" + what +
           "\nLine: ENDOPTIMIZER";
}

}  // namespace

/* Pins: a whole-number PARAM key written with an exponent, a '+' or a
 * decimal point is refused at startup, at the line its block closes at,
 * with the key's message and the value as written:
 *  - TADAH:GRID, PARAM GRID_POINTS 1e1, +10 and 10.0: "TADAH:GRID: PARAM
 *    GRID_POINTS must be an integer from 2 to 4294967295, got '1e1' (the
 *    points of each dimension include both of its bounds)";
 *  - ASA:ASA, PARAM Limit_Acceptances 1e3: "ASA:ASA: PARAM
 *    Limit_Acceptances must be an integer (got '1e3')";
 *  - CERES:STEEPEST_DESCENT, PARAM max_iter 1e1: "CERES:STEEPEST_DESCENT
 *    requires PARAM max_iter to be an integer from 0 to 2147483647
 *    (Ceres' max_num_iterations), got '1e1'";
 *  - TADAH:ANNEAL, PARAM CHAIN_LENGTH 5e1: "TADAH:ANNEAL: PARAM
 *    CHAIN_LENGTH must be a non-negative integer (got '5e1')";
 *  - NLOPT:LD_LBFGS, PARAM diff_method 1.0 and 3: "PARAM diff_method must
 *    be 0 (CENTRAL, the default) or 1 (FORWARD) (got '1.0')".
 * Every refusal quotes the value as the file writes it, as the refusal of a
 * word does (got 'central').
 * The same keys written as decimal digits (GRID_POINTS 10, Limit_Acceptances
 * 1000, max_iter 10, CHAIN_LENGTH 50, diff_method 1) start.
 * Before the fix: each value of the first list was read as the number it
 * writes (1e1 as 10) and tadah hpo started; the annealers' refusal of a
 * fraction gave no value ("must be a non-negative integer"); the refusals
 * of ASA:ASA, CERES and diff_method gave the value without quotes
 * (got 1e3). */
TEST_CASE("startup validation: a PARAM key that takes a whole number reads "
          "decimal digits only", "[startup][param][integer]")
{
    struct Case {
        const char* lib;
        const char* algo;
        const char* line;
        std::string what;   // after "the STRATEGY-level optimiser"
    };
    const auto grid = [](const std::string& v) {
        return ": TADAH:GRID: PARAM GRID_POINTS must be an integer from 2 to "
               "4294967295, got '" + v + "' (the points of each dimension "
               "include both of its bounds)";
    };
    const std::vector<Case> refused = {
        {"TADAH", "GRID", "PARAM GRID_POINTS 1e1", grid("1e1")},
        {"TADAH", "GRID", "PARAM GRID_POINTS +10", grid("+10")},
        {"TADAH", "GRID", "PARAM GRID_POINTS 10.0", grid("10.0")},
        {"ASA", "ASA", "PARAM Limit_Acceptances 1e3",
         ": ASA:ASA: PARAM Limit_Acceptances must be an integer (got '1e3')"},
        {"CERES", "STEEPEST_DESCENT", "PARAM max_iter 1e1",
         ": CERES:STEEPEST_DESCENT requires PARAM max_iter to be an integer "
         "from 0 to 2147483647 (Ceres' max_num_iterations), got '1e1'"},
        {"TADAH", "ANNEAL", "PARAM CHAIN_LENGTH 5e1",
         ": TADAH:ANNEAL: PARAM CHAIN_LENGTH must be a non-negative integer "
         "(got '5e1')"},
        {"NLOPT", "LD_LBFGS", "PARAM diff_method 1.0",
         " NLOPT:LD_LBFGS: PARAM diff_method must be 0 (CENTRAL, the "
         "default) or 1 (FORWARD) (got '1.0')"},
        {"NLOPT", "LD_LBFGS", "PARAM diff_method 3",
         " NLOPT:LD_LBFGS: PARAM diff_method must be 0 (CENTRAL, the "
         "default) or 1 (FORWARD) (got '3')"},
    };
    for (const Case& c : refused) {
        DYNAMIC_SECTION(c.lib << ":" << c.algo << ", " << c.line) {
            CHECK(refusal(plain(c.lib, c.algo, c.line)) == at_end(c.what));
        }
    }
    const std::vector<Case> accepted = {
        {"TADAH", "GRID", "PARAM GRID_POINTS 10", ""},
        {"ASA", "ASA", "PARAM Limit_Acceptances 1000", ""},
        {"CERES", "STEEPEST_DESCENT", "PARAM max_iter 10", ""},
        {"TADAH", "ANNEAL", "PARAM CHAIN_LENGTH 50", ""},
        {"NLOPT", "LD_LBFGS", "PARAM diff_method 1", ""},
    };
    for (const Case& c : accepted) {
        DYNAMIC_SECTION(c.lib << ":" << c.algo << ", " << c.line) {
            CHECK(refusal(plain(c.lib, c.algo, c.line)) == "");
        }
    }
}

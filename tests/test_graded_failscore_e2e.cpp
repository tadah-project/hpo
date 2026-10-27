/* Graded failures near the largest double (FAILSCORE_GRADING ON).
 *
 * A graded failure scores in [0.999 FAILSCORE, FAILSCORE) (pc_fail_grading.h),
 * so without a FAILSCORE line -- FAILSCORE the largest double -- every
 * failed evaluation reports a loss near 1.8e308. These cases run the whole
 * hpo_run() stack on the Ta model of the tests (tests/data/ta/common: the
 * elastic and bcc equation-of-state datasets for training, the elastic set
 * for validation), RCUT2B in [4, 6] and SGRIDMB in [0.1, 0.4], with
 * PC_ERMSE 0 0.0001 --skip-above 8: the potentials whose ERMSE is above 8
 * meV/atom, a large part of the box, are rejected as graded band-D
 * failures. Each search starts either at a usable point (RCUT2B 5.6,
 * SGRIDMB 0.2: ERMSE 5.26) or inside the failure region (RCUT2B 4.2,
 * SGRIDMB 0.3). An evaluation takes a few milliseconds. */

#include "catch2/catch.hpp"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/runners.h>

#include "finish_within.h"

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

const fs::path& ta_common_dir()
{
    static const fs::path dir = fs::path(TADAH_HPO_TA_DATA_DIR) / "common";
    return dir;
}

/* A scratch working directory: hpo_run() writes its files into the current
 * directory. */
struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir()
    {
        dir = fs::temp_directory_path() /
              ("tadah_graded_failscore_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir()
    {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

/* Where a search starts: RCUT2B and SGRIDMB of the training configuration. */
struct Start { const char* name; double rcut2b, sgridmb; };
const Start kUsable{"a usable start", 5.6, 0.2};
const Start kFailing{"a start in the failure region", 4.2, 0.3};

/* What one run left behind. */
struct Outcome {
    int status = -1;
    std::string journal;                       // hpo_run.log
    /* The search parameters of each row of params.tadah (its columns
     * after the evaluation number and the worker slot). */
    std::vector<std::vector<std::string>> params;
    std::vector<double> losses;                // loss.tadah, last column
};

std::string read_file(const fs::path& path)
{
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/* Whitespace-separated tokens of every data row (not '#', not blank). */
std::vector<std::vector<std::string>> data_rows(const fs::path& path)
{
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;
        std::istringstream is(line);
        std::vector<std::string> tok;
        for (std::string t; is >> t;) tok.push_back(t);
        rows.push_back(std::move(tok));
    }
    return rows;
}

/* One hpo_run() of a PLAIN search of `lib`:`algo` (with `extra` lines in
 * its STRATEGY block) from `start`, under FAILSCORE_GRADING ON with the
 * FAILSCORE line `failscore` ("" for none: the largest double). */
Outcome run(const std::string& lib, const std::string& algo,
            const std::string& extra, const std::string& failscore,
            const Start& start, int maxeval)
{
    tadah::core::PeriodicTable::initialize();
    tadah::hpo::HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    ScratchDir sd;
    const fs::path& common = ta_common_dir();
    const fs::path hpo = sd.dir / "config.hpo";
    const fs::path val = sd.dir / "config.val";
    std::ofstream(hpo)
        << "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB " << lib
        << "\n    ALGO " << algo << "\n    MAXEVAL " << maxeval << "\n"
        << extra << "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\nDIGITS 15\n"
        << "FAILSCORE_GRADING ON\n"
        << (failscore.empty() ? std::string()
                              : "FAILSCORE " + failscore + "\n")
        << "PC_ERMSE 0 0.0001 --skip-above 8\n"
        << "OPTIM RCUT2B (1) 4.0 6.0\nOPTIM SGRIDMB (1) 0.1 0.4\n";
    std::ofstream(val) << "DBFILE "
                       << (common / "data" / "elastic.tadahdb").string()
                       << "\n";
    tadah::core::Context ctx(tadah::core::CoreUserConfigParser().parse(
        (common / "config.tadah").string()));
    ctx.remove("DBFILE");
    ctx.add("DBFILE", (common / "data" / "elastic.tadahdb").string());
    ctx.add("DBFILE", (common / "data" / "eos_bcc.tadahdb").string());
    ctx.remove("RCUT2B");
    ctx.add("RCUT2B", start.rcut2b);
    ctx.remove("SGRIDMB");
    ctx.add("SGRIDMB", start.sgridmb);
    ctx.add("HPOTARGET", hpo.string());
    ctx.add("VALIDATION", val.string());
    Outcome out;
    hpo_test::finish_within(std::chrono::seconds(300), "a Ta-data run", [&] {
        out.status = tadah::hpo::hpo_run(ctx);
    });
    out.journal = read_file("hpo_run.log");
    for (const auto& row : data_rows("params.tadah"))
        out.params.emplace_back(row.begin() + 2, row.end());
    for (const auto& row : data_rows("loss.tadah"))
        out.losses.push_back(std::stod(row.back()));
    return out;
}

/* The "Optimiser status" value of the journal's RESULT section. */
std::string journal_status(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(Optimiser status\s*: ([^\n]*))")))
        return m[1];
    return "(no RESULT section)";
}

/* The bit pattern of a double: the build compiles with -ffast-math, under
 * which std::isfinite and std::isnan may be folded away. */
bool finite_bits(double v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

constexpr double kLargest = std::numeric_limits<double>::max();

}  // namespace

/* Pins: FAILSCORE_GRADING ON without a FAILSCORE line -- FAILSCORE the
 * largest double, so every graded failure reports a loss near 1.8e308 --
 * runs through the optimisers of each family, from a usable start and
 * from a start in the failure region: NLopt's LN_SBPLX, LN_BOBYQA,
 * LN_COBYLA, LN_NELDERMEAD, LD_LBFGS and GN_CRS2_LM, DLIB:GFS, DLIB:BFGS,
 * TADAH:RANDOM, TADAH:ANNEAL, TADAH:ANNEAL_PT, TADAH:ANNEAL_PA and
 * CERES:STEEPEST_DESCENT. For each run: status 0 and an Optimiser status
 * other than FAILED or STOPPED; every loss of loss.tadah is a finite
 * double, either usable or a graded failure in [0.999 FAILSCORE,
 * FAILSCORE) (none is FAILSCORE itself); the run has a usable loss, and
 * from the start in the failure region a graded failure. The optimisers
 * that compare losses alone (LN_SBPLX, LN_NELDERMEAD, GN_CRS2_LM,
 * TADAH:RANDOM and the annealers) evaluate the same points as under
 * FAILSCORE 1e299, which the startup checks accepted. (A search of the
 * gradient-based or model-based optimisers can take another path at
 * another FAILSCORE: the finite stand-in of a failure before the first
 * usable loss is FAILSCORE capped at 1e100, and rounding in its slope
 * changes the steps. ASA:ASA refuses, under FAILSCORE_GRADING ON, a
 * FAILSCORE whose graded failures reach the cost it takes for an invalid
 * state: the case "startup validation: FAILSCORE_GRADING ON takes any
 * FAILSCORE, and ASA:ASA one whose graded failures stay below its largest
 * valid cost".) CERES:STEEPEST_DESCENT's Armijo line search takes the
 * squared length of the gradient as its first slope: its lines in the
 * journal ("CERES:STEEPEST_DESCENT -- termination: ...", "--
 * non-convergent: ...") name no infinite or NaN number, and the search
 * runs more than one iteration.
 * Before the fix: every run without a FAILSCORE line was refused at
 * startup, "FAILSCORE_GRADING ON grades failures in bands below FAILSCORE,
 * which needs a FAILSCORE above 0 and below 1e300 (without a FAILSCORE
 * line, FAILSCORE is the largest double)". Before Ceres was given a
 * failed evaluation's loss scaled to at most 1e100: CERES:STEEPEST_DESCENT
 * from the start in the failure region ran 1 iteration, its line search
 * failing on a slope that is not a finite double ("termination: FAILURE |
 * iters=1", "Numerical failure in line search ... initial_gradient: -inf";
 * on these data from a FAILSCORE of about 1e160 on, below 1e299 too), and
 * most of its evaluations were graded failures (50 of 60 under MAXEVAL 60,
 * 11 under FAILSCORE 1e9). */
TEST_CASE("Graded failures near the largest double: FAILSCORE_GRADING ON "
          "without a FAILSCORE line runs through the optimiser families",
          "[HPO][failscore][grading][e2e]")
{
    struct Case {
        std::string lib, algo, extra;
        int maxeval;
        bool compares_only;   // the same points under FAILSCORE 1e299
    };
    const std::vector<Case> cases = {
        {"NLOPT", "LN_SBPLX", "", 40, true},
        {"NLOPT", "LN_BOBYQA", "", 40, false},
        {"NLOPT", "LN_COBYLA", "", 40, false},
        {"NLOPT", "LN_NELDERMEAD", "", 40, true},
        {"NLOPT", "LD_LBFGS", "", 40, false},
        {"NLOPT", "GN_CRS2_LM", "    SEED 5\n", 40, true},
        {"DLIB", "GFS", "", 40, false},
        {"DLIB", "BFGS", "", 40, false},
#ifdef TADAH_USE_CERES
        {"CERES", "STEEPEST_DESCENT", "", 40, false},
#endif
        {"TADAH", "RANDOM", "    SEED 5\n", 40, true},
        {"TADAH", "ANNEAL", "    SEED 5\n", 60, true},
        {"TADAH", "ANNEAL_PT", "    SEED 5\n", 120, true},
        {"TADAH", "ANNEAL_PA", "    SEED 5\n", 120, true},
    };
    for (const Case& c : cases) {
        for (const Start* start : {&kUsable, &kFailing}) {
            INFO(c.lib << ":" << c.algo << " from " << start->name);
            const Outcome o =
                run(c.lib, c.algo, c.extra, "", *start, c.maxeval);
            INFO(o.journal);
            CHECK(o.status == 0);
            const std::string status = journal_status(o.journal);
            CHECK(status.rfind("FAILED", 0) == std::string::npos);
            CHECK(status.rfind("STOPPED", 0) == std::string::npos);
            REQUIRE_FALSE(o.losses.empty());
            std::size_t graded = 0, usable = 0;
            for (double l : o.losses) {
                INFO("loss " << l);
                REQUIRE(finite_bits(l));
                CHECK(l < kLargest);
                if (l >= 0.999 * kLargest) ++graded;
                else ++usable;
            }
            /* A start in the failure region is a graded failure. */
            if (start == &kFailing) CHECK(graded > 0u);
            CHECK(usable > 0u);
            if (c.lib == "CERES") {
                std::size_t ceres_lines = 0;
                bool ended = false;
                std::istringstream in(o.journal);
                for (std::string line; std::getline(in, line);) {
                    if (line.find("CERES:" + c.algo + " -- ") ==
                        std::string::npos)
                        continue;
                    ++ceres_lines;
                    INFO(line);
                    CHECK_FALSE(std::regex_search(
                        line, std::regex(R"((^|[^A-Za-z])(inf|nan)\b)",
                                         std::regex::icase)));
                    std::smatch m;
                    if (std::regex_search(line, m,
                                          std::regex(R"(iters=(\d+))"))) {
                        ended = true;
                        CHECK(std::stoul(m[1]) > 1u);
                    }
                }
                CHECK(ended);
                CHECK(ceres_lines > 0u);
            }
            if (c.compares_only) {
                const Outcome near =
                    run(c.lib, c.algo, c.extra, "1e299", *start, c.maxeval);
                CHECK(near.status == 0);
                CHECK(near.params == o.params);
            }
        }
    }
}

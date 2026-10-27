/* The first steps of a search on the Ta test data (tests/data/ta/common):
 * what NLopt is given for a failed evaluation, where LN_BOBYQA starts next
 * to a bound, and the scale DLIB:BFGS runs on.
 *
 * Each case runs the whole hpo_run() stack in a scratch directory on the
 * shared Ta model of the tests (config.tadah, the elastic and bcc
 * equation-of-state datasets for training, the elastic set for validation)
 * with RCUT2B in [4, 6] and SGRIDMB in [0.1, 0.4]; an evaluation takes a
 * few milliseconds. */

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
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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
              ("tadah_search_steps_" +
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

/* One hpo_run() of the target file `target` on the Ta model, with RCUT2B
 * set to `rcut2b` in the training configuration. */
struct TaDataRun {
    ScratchDir sd;
    tadah::core::Context ctx;
    int status = -1;

    TaDataRun(const std::string& target, double rcut2b = 5.6)
    {
        tadah::core::PeriodicTable::initialize();
        tadah::hpo::HPOLog::instance().reset_for_test();
        Host::reset_globals_for_test();
        const fs::path& common = ta_common_dir();
        const fs::path hpo = sd.dir / "config.hpo";
        const fs::path val = sd.dir / "config.val";
        std::ofstream(hpo) << target;
        std::ofstream(val) << "DBFILE "
                           << (common / "data" / "elastic.tadahdb").string()
                           << "\n";
        ctx = tadah::core::Context(tadah::core::CoreUserConfigParser().parse(
            (common / "config.tadah").string()));
        ctx.remove("DBFILE");
        ctx.add("DBFILE", (common / "data" / "elastic.tadahdb").string());
        ctx.add("DBFILE", (common / "data" / "eos_bcc.tadahdb").string());
        ctx.remove("RCUT2B");
        ctx.add("RCUT2B", rcut2b);
        ctx.add("HPOTARGET", hpo.string());
        ctx.add("VALIDATION", val.string());
        hpo_test::finish_within(std::chrono::seconds(300), "a Ta-data run",
                                [&] { status = tadah::hpo::hpo_run(ctx); });
    }
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

/* The parameters of each row of params.tadah (RCUT2B, SGRIDMB). */
std::vector<std::vector<double>> params_rows()
{
    std::vector<std::vector<double>> out;
    for (const auto& row : data_rows("params.tadah")) {
        std::vector<double> p;
        for (std::size_t i = 2; i < row.size(); ++i)
            p.push_back(std::stod(row[i]));
        out.push_back(std::move(p));
    }
    return out;
}

/* The rows of params.tadah whose parameters repeat an earlier row's. */
std::size_t repeated_rows()
{
    std::set<std::vector<std::string>> seen;
    std::size_t n = 0;
    for (const auto& row : data_rows("params.tadah")) {
        const std::vector<std::string> params(row.begin() + 2, row.end());
        if (!seen.insert(params).second) ++n;
    }
    return n;
}

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
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

/* The target file of a PLAIN search of `strategy_body` on RCUT2B in [4, 6]
 * and SGRIDMB in [0.1, 0.4], with the loss lines `loss`. */
std::string ta_target(const std::string& strategy_body,
                      const std::string& loss)
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n" + strategy_body +
           "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\nDIGITS 15\n" + loss +
           "OPTIM RCUT2B (1) 4.0 6.0\nOPTIM SGRIDMB (1) 0.1 0.4\n";
}

}  // namespace

/* Pins: LN_BOBYQA and LN_COBYLA (MAXEVAL 60) with PC_ERMSE 0 0.0001
 * --skip-above 5.5, which rejects part of the box (not the start), under
 * the default FAILSCORE (the largest double), under FAILSCORE 1e300 and
 * under FAILSCORE 1000: each run makes 60 evaluations, no params.tadah row
 * repeats another and none was answered from the record (no "Repeated
 * points reused" in DIAGNOSTICS), it ends MAXEVAL_REACHED, and loss.tadah
 * records the FAILSCORE of every rejected evaluation; the journal says once
 * that NLopt is given a finite stand-in, and the three runs evaluate the
 * same points (the start is usable, so every failure is given the worst
 * usable loss so far plus a margin, whatever FAILSCORE is).
 * Before the fix: NLopt was given FAILSCORE itself, and the journal had no
 * stand-in line. Under the default FAILSCORE and 1e300 LN_BOBYQA made 49
 * evaluations, 2 of them repeating an earlier row, and its other calls were
 * answered from the record; LN_COBYLA asked for points whose coordinates
 * were not numbers under the default, 7 rows of NaN (the 6 repeated rows
 * the check counted, 'Failed converting "-nan"' in the log), and ended
 * after 3 evaluations under 1e300. Under 1000 both made 60
 * evaluations without a repeat, along other points than under the
 * default. */
TEST_CASE("NLopt failed evaluations: LN_BOBYQA and LN_COBYLA on the Ta data "
          "with a --skip-above cap make every evaluation of MAXEVAL",
          "[HPO][nlopt][failscore][e2e]")
{
    for (const std::string algo : {"LN_BOBYQA", "LN_COBYLA"}) {
        std::vector<std::vector<std::vector<double>>> runs;
        for (const std::string failscore :
             {std::string(), std::string("1e300"), std::string("1000")}) {
            INFO(algo << (failscore.empty() ? ", default FAILSCORE"
                                            : ", FAILSCORE " + failscore));
            TaDataRun run(ta_target(
                "    LIB NLOPT\n    ALGO " + algo + "\n    MAXEVAL 60\n",
                (failscore.empty() ? std::string()
                                   : "FAILSCORE " + failscore + "\n") +
                    "PC_ERMSE 0 0.0001 --skip-above 5.5\n"));
            CHECK(run.status == 0);
            const std::string journal = read_file("hpo_run.log");
            const auto loss = data_rows("loss.tadah");
            const std::string fail_row =
                failscore.empty() ? "1.797693134862316e+308"
                : failscore == "1e300" ? "1.000000000000000e+300"
                                       : "1.000000000000000e+03";
            std::size_t failed = 0;
            for (const auto& row : loss)
                if (row.back() == fail_row) ++failed;
            INFO("rejected evaluations " << failed);
            CHECK(loss.size() == 60u);
            CHECK(failed > 0u);
            CHECK(repeated_rows() == 0u);
            CHECK(journal.find("Repeated points reused") == std::string::npos);
            CHECK(journal_status(journal) ==
                  "MAXEVAL_REACHED -- maximum evaluations exhausted");
            CHECK(count_of(journal, "NLOPT:" + algo +
                                        ": NLopt is given a finite stand-in "
                                        "for each failed evaluation of this "
                                        "search") == 1u);
            runs.push_back(params_rows());
        }
        INFO(algo);
        CHECK(runs[1] == runs[0]);
        CHECK(runs[2] == runs[0]);
    }
}

/* Pins: LN_BOBYQA (MAXEVAL 30) from the configuration's RCUT2B 4.005, 0.005
 * above its lower bound 4, with STEP unset and with STEP 0.25: the start
 * moves onto the bound (the gap is below half the step, 0.5), the first
 * evaluation is at RCUT2B 4, and the search reaches beyond RCUT2B 4.1. The
 * journal gives the old and the new start in an INFO line and warns that
 * the CONFIG start moved.
 * Before the fix: BOBYQA started at 4.005, shrank the step of RCUT2B to
 * 0.75 of the gap and kept it as that coordinate's scale: no evaluation
 * reached beyond RCUT2B 4.01006 in either run, and neither journal had the
 * two lines. */
TEST_CASE("LN_BOBYQA: a start next to a bound moves by at most half a step "
          "and the search keeps its full step",
          "[HPO][nlopt][step_plan][e2e]")
{
    for (const std::string step : {std::string(), std::string("0.25")}) {
        DYNAMIC_SECTION((step.empty() ? std::string("STEP unset")
                                      : "STEP " + step)) {
            TaDataRun run(
                ta_target("    LIB NLOPT\n    ALGO LN_BOBYQA\n    MAXEVAL 30\n" +
                              (step.empty() ? std::string()
                                            : "    STEP " + step + "\n"),
                          "FAILSCORE 1e16\nPC_ERMSE 0 0.0001\n"),
                4.005);
            CHECK(run.status == 0);
            const auto pts = params_rows();
            REQUIRE(pts.size() == 30u);
            CHECK(pts.front()[0] == 4.0);
            double reach = 0.0;
            for (const auto& p : pts) reach = std::max(reach, p[0]);
            INFO("largest RCUT2B evaluated " << reach);
            CHECK(reach > 4.1);
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("NLOPT:LN_BOBYQA: RCUT2B starts at 4 instead "
                               "of 4.005") != std::string::npos);
            CHECK(journal.find("WARNING   NLOPT:LN_BOBYQA: the CONFIG start "
                               "moves") != std::string::npos);
        }
    }
}

/* Pins: LN_COBYLA (MAXEVAL 30, STEP unset) from the configuration's
 * RCUT2B 4.005, 0.005 above its lower bound 4: it is given a step of 0.25
 * of the width of each dimension, keeps it as the scale of the coordinate,
 * and the search reaches beyond RCUT2B 4.1; the start stays (the first
 * evaluation is at 4.005), and the journal has no line of a moved start.
 * Before the fix: NLopt's own default shrank the step of RCUT2B to 0.75 of
 * the gap, and COBYLA kept it: no evaluation went beyond RCUT2B 4.0091. */
TEST_CASE("LN_COBYLA: a start next to a bound keeps the full step with STEP "
          "unset", "[HPO][nlopt][step_plan][e2e]")
{
    TaDataRun run(ta_target("    LIB NLOPT\n    ALGO LN_COBYLA\n    MAXEVAL 30\n",
                            "FAILSCORE 1e16\nPC_ERMSE 0 0.0001\n"),
                  4.005);
    CHECK(run.status == 0);
    const auto pts = params_rows();
    REQUIRE(pts.size() == 30u);
    CHECK(pts.front()[0] == Approx(4.005).epsilon(1e-12));
    double reach = 0.0;
    for (const auto& p : pts) reach = std::max(reach, p[0]);
    INFO("largest RCUT2B evaluated " << reach);
    CHECK(reach > 4.1);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("starts at") == std::string::npos);
}

/* Pins: LN_AUGLAG (MAXEVAL 30) with an INNER LN_BOBYQA and STEP unset from
 * the configuration's RCUT2B 4.005, 0.005 above its lower bound 4: AUGLAG
 * runs its one local search from its start, and that search is given 0.25
 * of each width and moves the start onto the bound, as an LN_BOBYQA search
 * of its own does: the first evaluation is at RCUT2B 4, the search reaches
 * beyond RCUT2B 4.1, and the journal gives the moved start and warns that
 * the CONFIG start moved.
 * Before the fix: the local search took NLopt's default from the start,
 * which shrank the step of RCUT2B to 0.75 of the gap, and kept it as the
 * scale of the coordinate: the first evaluation was at 4.005, none went
 * beyond RCUT2B 4.0101, and the journal had neither line. */
TEST_CASE("LN_AUGLAG with an INNER LN_BOBYQA: a start next to a bound moves "
          "by at most half a step and the search keeps its full step",
          "[HPO][nlopt][step_plan][e2e]")
{
    TaDataRun run(ta_target("    LIB NLOPT\n    ALGO LN_AUGLAG\n    MAXEVAL 30\n"
                            "    INNER\n      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                            "    ENDINNER\n",
                            "FAILSCORE 1e16\nPC_ERMSE 0 0.0001\n"),
                  4.005);
    CHECK(run.status == 0);
    const auto pts = params_rows();
    REQUIRE(pts.size() >= 20u);
    CHECK(pts.front()[0] == 4.0);
    double reach = 0.0;
    for (const auto& p : pts) reach = std::max(reach, p[0]);
    INFO("largest RCUT2B evaluated " << reach);
    CHECK(reach > 4.1);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("NLOPT:LN_AUGLAG (its local search LN_BOBYQA): RCUT2B "
                       "starts at 4 instead of 4.005") != std::string::npos);
    CHECK(journal.find("WARNING   NLOPT:LN_AUGLAG (its local search "
                       "LN_BOBYQA): the CONFIG start moves") !=
          std::string::npos);
}

/* Pins: DLIB:BFGS (MAXEVAL 60, FAILSCORE 1e300) from RCUT2B 5.6, SGRIDMB
 * 0.2 with PC_ERMSE 0 <w> and FTOL_ABS 1e-7 w at the loss-term weights w =
 * 1e-4, 1 and 1e6 -- the same problem with its loss, and the tolerance on
 * it, in units 1e4 times smaller and 1e6 times larger: every run reaches
 * the optimum at RCUT2B 6 (its upper bound) and ends there, and the
 * numbers of evaluations of the three runs differ by at most one line
 * search (a trial and the 4 points of a central difference in two
 * dimensions); the journal gives the length of the first gradient each run
 * divides its loss by.
 * Before the fix: dlib's first step is the gradient itself, so the path
 * followed the unit of the loss: at 1e-4 and 1 BFGS crept along RCUT2B and
 * made 60 and 59 evaluations; at 1e6 its first step went to the corner
 * (6, 0.1), backtracked, and the run stopped at RCUT2B 5.604 on FTOL_ABS
 * after 24; the journal had no such line. */
TEST_CASE("DLIB:BFGS: the path does not depend on the scale of the loss",
          "[HPO][dlib_bfgs][e2e]")
{
    std::vector<std::size_t> evals;
    for (const auto& [weight, ftol] :
         std::vector<std::pair<std::string, std::string>>{
             {"1e-4", "1e-11"}, {"1", "1e-7"}, {"1e6", "1e-1"}}) {
        INFO("PC_ERMSE weight " << weight << ", FTOL_ABS " << ftol);
        TaDataRun run(ta_target("    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 60\n"
                                "    FTOL_ABS " + ftol + "\n",
                                "FAILSCORE 1e300\nPC_ERMSE 0 " + weight + "\n"));
        CHECK(run.status == 0);
        const auto best = data_rows("best_params.tadah");
        REQUIRE_FALSE(best.empty());
        CHECK(std::stod(best.back().at(2)) == 6.0);
        evals.push_back(data_rows("loss.tadah").size());
        const std::string journal = read_file("hpo_run.log");
        CHECK(journal.find("DLIB:BFGS: the search runs on the loss divided by "
                           "the length of its first gradient, ") !=
              std::string::npos);
    }
    INFO("evaluations " << evals[0] << ", " << evals[1] << ", " << evals[2]);
    const auto [lo, hi] = std::minmax_element(evals.begin(), evals.end());
    CHECK(*hi - *lo <= 5u);
}

/* Pins: DLIB:BFGS (MAXEVAL 60, FTOL_ABS 1e-11) from RCUT2B 5.6, SGRIDMB 0.2
 * with PC_ERMSE 0 0.0001 --skip-above 6.0, which rejects part of the box,
 * under the default FAILSCORE (the largest double): the length of the first
 * gradient the search divides its loss by is below 1 (0.0072), the journal
 * says once that dlib is given a finite stand-in for a failed evaluation,
 * at most 2 of the 60 evaluations fail, and the best loss is below 1.2e-3
 * (the start's is 2.77e-3, and Tadah! 1.5.0, which runs dlib's loop on the
 * loss itself, reaches 2.29e-3 here).
 * Before the fix: dlib was given FAILSCORE divided by that length, +inf;
 * its line search took a step that is not a number, which the box put on a
 * corner: 55 of the 60 evaluations failed, the best loss was the start's
 * (2.767e-3), and no journal line named a stand-in. */
TEST_CASE("DLIB:BFGS: failed evaluations under the default FAILSCORE with a "
          "first gradient shorter than 1", "[HPO][dlib_bfgs][failscore][e2e]")
{
    TaDataRun run(ta_target("    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 60\n"
                            "    FTOL_ABS 1e-11\n",
                            "PC_ERMSE 0 0.0001 --skip-above 6.0\n"));
    CHECK(run.status == 0);
    const auto loss = data_rows("loss.tadah");
    REQUIRE(loss.size() == 60u);
    std::size_t failed = 0;
    double best = 1e300;
    for (const auto& row : loss) {
        if (row.back() == "1.797693134862316e+308") ++failed;
        else best = std::min(best, std::stod(row.back()));
    }
    INFO("failed evaluations " << failed << ", best " << best);
    CHECK(failed <= 2u);
    CHECK(best < 1.2e-3);
    const std::string journal = read_file("hpo_run.log");
    std::smatch m;
    REQUIRE(std::regex_search(
        journal, m,
        std::regex("DLIB:BFGS: the search runs on the loss divided by the "
                   "length of its first gradient, ([0-9.e+-]+) ")));
    CHECK(std::stod(m[1]) < 1.0);
    CHECK(count_of(journal, "DLIB:BFGS: dlib is given a finite stand-in for "
                            "each failed evaluation of this search") == 1u);
}

/* What NLopt is given for a failed evaluation (nlopt_adapter.h,
 * run_nlopt_impl; failed_loss_stand_in.h).
 *
 * A failed evaluation -- one rejected by a --skip-above cap, a failed
 * script, an exception -- reports FAILSCORE, by default the largest
 * double. LN_BOBYQA and LN_COBYLA build a model of the loss from the
 * values they are given; a value that large overflows it, and the search
 * then asks for points it has evaluated, or stops early. NLopt is given a
 * finite stand-in for a failed evaluation instead, as DLIB:GFS is: the
 * worst usable loss of the search so far plus a margin, or, before any,
 * FAILSCORE capped at 1e100. The records keep the loss the evaluation
 * reported.
 *
 * The stub host records every REAL point asked for. Its loss is
 * Rosenbrock's valley; a point whose loss is above a cap is rejected and
 * reports FAILSCORE, as --skip-above makes it. */

#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;
namespace fs = std::filesystem;

struct NoRunner {};   // the GradientProvider's Runner is only a type tag

constexpr double kDblMax = std::numeric_limits<double>::max();

bool same_bits(const Vec& a, const Vec& b)
{
    if (a.size() != b.size()) return false;
    for (long i = 0; i < a.size(); ++i)
        if (std::memcmp(&a(i), &b(i), sizeof(double)) != 0) return false;
    return true;
}

/* The points of `asked` that equal an earlier one, bit for bit. */
std::size_t repeats_in(const std::vector<Vec>& asked)
{
    std::size_t n = 0;
    for (std::size_t k = 0; k < asked.size(); ++k)
        for (std::size_t j = 0; j < k; ++j)
            if (same_bits(asked[k], asked[j])) {
                ++n;
                break;
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

/* The number on the DIAGNOSTICS line `label` ("" when there is none). */
std::string diagnostics_count(const std::string& journal,
                              const std::string& label)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(label + R"(\s*: ([0-9]+))")))
        return m[1];
    return "";
}

/* An open journal; `text()` closes it and returns it. */
struct Journal {
    fs::path path = fs::temp_directory_path() /
                    ("tadah_nlopt_failed_" + std::to_string(::getpid()) + "_" +
                     std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch().count()) +
                     ".log");
    Journal() {
        auto& log = HPOLog::instance();
        log.reset_for_test();
        log.open(path.string(), "test", "/", 0);
    }
    std::string text() {
        HPOLog::instance().finish();
        std::ifstream f(path);
        std::ostringstream s;
        s << f.rdbuf();
        return s.str();
    }
    ~Journal() {
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* Rosenbrock's valley over [0, 1]^2 (real = internal coordinates): x =
 * 4A - 2, y = 4B - 1, (1 - x)^2 + 100 (y - x^2)^2, its minimum 0 at A =
 * 0.75, B = 0.5. A point whose loss is above `cap` (100) -- high on the
 * walls of the valley -- is rejected and reports `failscore`, as a
 * --skip-above cap makes it. Every evaluation is a function of its point
 * (repeat_reuse_blocker() is ""), so a point asked for again is answered
 * from the search's record when its evaluation was usable. */
struct CappedValley {
    std::vector<Vec> asked;
    std::vector<double> reported;
    double failscore = kDblMax;
    double cap = 100.0;
    double operator()(const Vec& v) {
        asked.push_back(v);
        const double x = 4.0 * v(0) - 2.0, y = 4.0 * v(1) - 1.0;
        const double loss =
            (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
        reported.push_back(loss > cap ? failscore : loss);
        return reported.back();
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

OptimizerSpec spec(const std::string& algo, double failscore)
{
    OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = algo;
    sp.low = {0.0, 0.0};
    sp.high = {1.0, 1.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    sp.analyse_logscale();
    sp.parameter_labels = {"A", "B"};
    sp.optim_init = {{0.3, 0.5}};
    sp.stop.maxeval = 60;
    sp.failscore = failscore;
    return sp;
}

const std::string kStandInNote =
    ": NLopt is given a finite stand-in for each failed evaluation of this "
    "search";

}  // namespace

/* Pins: LN_BOBYQA and LN_COBYLA (MAXEVAL 60, from A = 0.3, B = 0.5, where
 * the loss is 16.2) on the capped valley, whose rejected points report
 * FAILSCORE: under the default FAILSCORE (the largest double), under 1e300
 * and under 1000 each search makes its 60 evaluations, asks for no point
 * twice -- none is answered from the record (no "Repeated points reused"
 * in DIAGNOSTICS) -- and ends MAXEVAL_REACHED with a usable best; the
 * three searches ask for the same points (the start is usable, so every
 * failure is given the worst usable loss so far plus a margin, whatever
 * FAILSCORE is). The host reported FAILSCORE for every rejected point (the
 * records keep it), and the journal says once, at the first failed
 * evaluation, that NLopt is given a finite stand-in.
 * Before the fix: NLopt was given FAILSCORE itself, and the journal had no
 * stand-in line. Under the largest double and under 1e300 LN_BOBYQA made 47
 * evaluations (best 14.47) and spent its other calls asking for points it
 * had evaluated; LN_COBYLA asked four times for a point whose
 * coordinates were not numbers (NaN; the 3 repeats the check counted) under
 * the largest double, and ended after 3 evaluations under 1e300 (best 16.2,
 * the start). Under 1000 both made 60 evaluations without a repeat (best
 * 0.437 and 1.64), and the three FAILSCOREs gave three different
 * searches. */
TEST_CASE("NLopt failed evaluations: LN_BOBYQA and LN_COBYLA reach MAXEVAL "
          "without asking for a point twice under any FAILSCORE",
          "[HPO][nlopt][failscore]")
{
    for (const std::string algo : {"LN_BOBYQA", "LN_COBYLA"}) {
        std::vector<std::vector<Vec>> searches;
        for (const double failscore : {kDblMax, 1e300, 1000.0}) {
            INFO(algo << ", FAILSCORE " << failscore);
            const OptimizerSpec sp = spec(algo, failscore);
            CappedValley host;
            host.failscore = failscore;
            Journal j;
            REQUIRE_NOTHROW(
                tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
            const std::string journal = j.text();
            std::size_t failed = 0;
            double best = kDblMax;
            for (const double l : host.reported) {
                if (l == failscore) ++failed;
                else best = std::min(best, l);
            }
            INFO("failed evaluations " << failed << ", best " << best);
            CHECK(host.asked.size() == 60u);
            CHECK(repeats_in(host.asked) == 0u);
            CHECK(diagnostics_count(journal, "Repeated points reused").empty());
            CHECK(journal.find("NLopt finished: MAXEVAL_REACHED -- maximum "
                               "evaluations exhausted after 60 of MAXEVAL 60 "
                               "evaluation(s)\n") != std::string::npos);
            CHECK(failed > 0u);
            CHECK(best < 16.2);
            CHECK(count_of(journal, "NLOPT:" + algo + kStandInNote) == 1u);
            searches.push_back(host.asked);
        }
        INFO(algo);
        for (std::size_t k = 1; k < searches.size(); ++k) {
            INFO("FAILSCORE " << k + 1 << " of 3 against the largest double");
            bool same = searches[k].size() == searches[0].size();
            for (std::size_t i = 0; same && i < searches[0].size(); ++i)
                same = same_bits(searches[k][i], searches[0][i]);
            CHECK(same);
        }
    }
}

/* Pins: a search without a failed evaluation gives NLopt its losses and
 * journals no stand-in (LN_BOBYQA on the valley without a cap).
 * Before the fix: passes as well (a pin: NLopt was given every loss as the
 * evaluation reported it). */
TEST_CASE("NLopt failed evaluations: a search without a failed evaluation "
          "journals no stand-in", "[HPO][nlopt][failscore]")
{
    const OptimizerSpec sp = spec("LN_BOBYQA", kDblMax);
    CappedValley host;
    host.cap = kDblMax;
    Journal j;
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    const std::string journal = j.text();
    for (const double l : host.reported) CHECK(l < kDblMax);
    CHECK(count_of(journal, kStandInNote) == 0u);
}

/* Pins: LN_NELDERMEAD (FTOL_ABS 1e-6, MAXEVAL 60) on the valley whose
 * every point is rejected (cap -1), under the default FAILSCORE: NLopt is
 * given the same stand-in for every evaluation, its simplex is flat, and
 * the search ends FTOL_REACHED; the journal's ending line gives as fmin the
 * loss the evaluation of NLopt's best point reported, the largest double
 * ("NLopt finished: FTOL_REACHED -- function tolerance converged
 * fmin=1.79769e+308"), not the stand-in. On the valley without a cap
 * (MAXEVAL 400) the search ends FTOL_REACHED too, and the line gives the
 * best usable loss, the loss NLopt was given (a pin).
 * Before the fix: the line gave the stand-in NLopt was given, a loss no
 * evaluation reported ("fmin=1e+100"); the run without a cap passed as
 * well. */
TEST_CASE("NLopt failed evaluations: the ending line gives the loss the "
          "evaluation reported as fmin", "[HPO][nlopt][failscore]")
{
    const auto fmin_of = [](const std::string& journal) {
        std::smatch m;
        if (std::regex_search(
                journal, m,
                std::regex(R"(NLopt finished: FTOL_REACHED -- function )"
                           R"(tolerance converged  fmin=([^\n]*))")))
            return std::string(m[1]);
        return std::string("(no FTOL_REACHED line)");
    };
    SECTION("every evaluation rejected") {
        auto sp = spec("LN_NELDERMEAD", kDblMax);
        sp.stop.ftol_abs = 1e-6;
        CappedValley host;
        host.cap = -1.0;
        Journal j;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        const std::string journal = j.text();
        for (const double l : host.reported) CHECK(l == kDblMax);
        CHECK(host.asked.size() < 60u);
        CHECK(fmin_of(journal) == "1.79769e+308");
    }
    SECTION("no evaluation rejected") {
        auto sp = spec("LN_NELDERMEAD", kDblMax);
        sp.stop.ftol_abs = 1e-6;
        sp.stop.maxeval = 400;
        CappedValley host;
        host.cap = kDblMax;
        Journal j;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        const std::string journal = j.text();
        const double best =
            *std::min_element(host.reported.begin(), host.reported.end());
        std::ostringstream shown;
        shown << best;
        CHECK(host.asked.size() < 400u);
        CHECK(fmin_of(journal) == shown.str());
    }
}

/* Pins: a loss NLopt asks for without its gradient is given the stand-in
 * for every failed evaluation, an ungraded one included, also in a
 * gradient-based search: GD_MLSL (its default local search LD_MMA,
 * MAXEVAL 60, POPULATION 4) on the capped valley under the default
 * FAILSCORE and FAILSCORE_GRADING OFF evaluates its random samples without
 * a gradient (mlsl.c), and samples high on the walls of the valley fail;
 * NLopt is given the stand-in for them, and the journal says so once. (A
 * loss NLopt asks for with its gradient is given the stand-in only for a
 * graded failure with a usable gradient: an ungraded failure fails the
 * gradient, and the point is rejected.)
 * Before the fix: passes as well (a pin of the rule the documentation of
 * FAILSCORE states). */
TEST_CASE("NLopt failed evaluations: a gradient-based search is given the "
          "stand-in for a failed loss asked for without its gradient",
          "[HPO][nlopt][failscore]")
{
    auto sp = spec("GD_MLSL", kDblMax);
    sp.pop = 4;
    CappedValley host;
    Journal j;
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    const std::string journal = j.text();
    std::size_t failed = 0;
    for (const double l : host.reported)
        if (l == kDblMax) ++failed;
    INFO("evaluations " << host.reported.size() << ", failed " << failed);
    CHECK(failed > 0u);
    CHECK(count_of(journal, "NLOPT:GD_MLSL" + kStandInNote) == 1u);
}

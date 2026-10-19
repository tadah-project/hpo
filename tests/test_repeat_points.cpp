/* Repeated points (repeat_points.h, nlopt_adapter.h, dlib_adapter.h).
 *
 * An optimiser can ask for a point it has evaluated before, bit for bit:
 * NLopt's LD_MMA and LD_CCSAQ re-solve their subproblem with a larger rho
 * and land on the same clamped corner, LN_SBPLX and LN_NELDERMEAD reflect
 * onto it again, and DLIB:BFGS's line search evaluates the point it starts
 * from when the step, held to the OPTIM box, cannot move it. Each repeat
 * was a full evaluation (a fit, and a LAMMPS run under PC_LAMMPS), and
 * DLIB:BFGS spent the rest of MAXEVAL on one such point -- or, under PARAM
 * tol_gradinf, ran without end.
 *
 * A point whose earlier evaluation gave a usable loss is answered with that
 * loss (and its gradient, when the search formed one there) when the host
 * says every evaluation is a function of its point (repeat_reuse_blocker()
 * is ""), and evaluated again otherwise. HPO_Host keeps one record for the
 * run (test_hpo_termination.cpp, test_repeat_record.cpp); the stub hosts
 * here have none, so each search keeps its own. DLIB:BFGS does not run a
 * line search that cannot move the point, whatever the host, ends a search
 * that no step can move ("no step within the OPTIM box"), and ends one that
 * an iteration without an evaluation took back to a point where it stood
 * ("back at an earlier point").
 *
 * The stub hosts record every REAL point asked for; g_iter_value() numbers
 * the evaluations as HPO_Host's counter does. */

#include "catch2/catch.hpp"
#include "finish_within.h"

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/periodic_table.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>
#include <dlib/optimization.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::GradStrategy;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;
namespace fs = std::filesystem;

struct NoRunner {};   // GradientProvider's Runner is only a type tag

Vec point(double a, double b)
{
    Vec v(2);
    v(0) = a;
    v(1) = b;
    return v;
}

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

/* `asked` without the points that equal an earlier one, in order. */
std::vector<Vec> first_asks(const std::vector<Vec>& asked)
{
    std::vector<Vec> out;
    for (const Vec& p : asked) {
        bool seen = false;
        for (const Vec& q : out) seen = seen || same_bits(p, q);
        if (!seen) out.push_back(p);
    }
    return out;
}

bool same_sequence(const std::vector<Vec>& a, const std::vector<Vec>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!same_bits(a[k], b[k])) return false;
    return true;
}

/* An open journal; `text()` closes it and returns it. */
struct Journal {
    fs::path path = fs::temp_directory_path() /
                    ("tadah_repeat_points_" + std::to_string(::getpid()) + "_" +
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

/* The journal trailer's status. */
std::string status_of(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(Optimiser status\s*: ([^\n]*))")))
        return m[1];
    return "(no status)";
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

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
}

/* 100 ((a - 1.3)^2 + (b - 1.1)^2) on [0, 1]^2 (real = internal coordinates):
 * its minimum lies outside the box, beyond the corner (1, 1), where the
 * searches end. `reason` is what repeat_reuse_blocker() says: "" when every
 * evaluation is a function of its point. */
struct CornerBowl {
    std::vector<Vec> asked;
    std::string reason;
    double scale = 1.0;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double a = x(0) - 1.3, b = x(1) - 1.1;
        return scale * (a * a + b * b);
    }
    std::string repeat_reuse_blocker() const { return reason; }
    std::size_t g_iter_value() const { return asked.size(); }
};

const std::string kBlocker =
    "PC_LAMMPS: its scripts run on LAMMPS instances kept from one "
    "evaluation to the next (test)";

OptimizerSpec nlopt_spec(const std::string& algo, unsigned maxeval)
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
    sp.optim_init = {{0.5, 0.5}};
    sp.stop.maxeval = maxeval;
    sp.stop.xtol_rel = 1e-8;
    sp.seed = 7;
    sp.param["fd_step"] = 1e-6;
    sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = std::numeric_limits<double>::max();
    return sp;
}

/* One NLopt search of `algo` on `host`; returns the journal. */
template <class Host>
std::string run_nlopt(const OptimizerSpec& sp, Host& host)
{
    Journal j;
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    return j.text();
}

}  // namespace

/* ======================================================================
 * NLopt
 * ==================================================================== */

/* Pins: LD_MMA and LD_CCSAQ (NUMERICAL, CENTRAL), MAXEVAL 60, on the bowl
 * whose minimum lies beyond the corner (1, 1), with a host whose every
 * evaluation is a function of its point: the host is asked for no point
 * twice (10 fewer than with a host that evaluates the repeats: MMA asks
 * for the corner and its one-sided stencil three times), the points asked
 * for are those of that host, in its order, without its repeats -- the same
 * search -- and both end XTOL_REACHED. The journal has the INFO line of the
 * first reuse ("NLOPT:LD_MMA asked again for the point of eval 16 (A = 1,
 * B = 1) and its gradient: the loss and gradient of eval 16 are reused,
 * without an evaluation") and DIAGNOSTICS "Repeated points reused : 2" (two
 * calls, each the corner's loss and its 4-point gradient).
 * Before the fix: the reusing host was asked for 30 points (LD_MMA) and 25
 * (LD_CCSAQ), 10 of them repeats, and the journal had neither line. */
TEST_CASE("Repeated points: LD_MMA and LD_CCSAQ evaluate no point twice "
          "when every evaluation is a function of its point, and keep their "
          "search", "[HPO][repeat_points][nlopt]")
{
    for (const std::string algo : {"LD_MMA", "LD_CCSAQ"}) {
        DYNAMIC_SECTION(algo) {
            CornerBowl again{{}, kBlocker};
            const std::string j_again = run_nlopt(nlopt_spec(algo, 60), again);
            CornerBowl reuse{{}, ""};
            const std::string j_reuse = run_nlopt(nlopt_spec(algo, 60), reuse);

            REQUIRE(repeats_in(again.asked) == 10u);
            CHECK(repeats_in(reuse.asked) == 0u);
            CHECK(reuse.asked.size() == again.asked.size() - 10u);
            CHECK(same_sequence(reuse.asked, first_asks(again.asked)));
            CHECK(status_of(j_again) ==
                  "XTOL_REACHED -- parameter tolerance converged");
            CHECK(status_of(j_reuse) == status_of(j_again));
            CHECK(j_reuse.find("NLOPT:" + algo +
                               " asked again for the point of eval ") !=
                  std::string::npos);
            CHECK(j_reuse.find("(A = 1, B = 1) and its gradient: the loss and "
                               "gradient of eval ") != std::string::npos);
            CHECK(j_reuse.find("are reused, without an evaluation") !=
                  std::string::npos);
            CHECK(count_of(j_reuse, " asked again for the point of eval ") ==
                  1u);
            CHECK(diagnostics_count(j_reuse, "Repeated points reused") == "2");
        }
    }
}

/* Pins: a host that says why its evaluations can differ (here a PC_LAMMPS
 * reason) has every repeat evaluated: LD_MMA asks it for the corner and
 * its stencil three times, 30 points, 10 of them repeats, as before; the
 * journal says so once, with the reason ("NLOPT:LD_MMA asked again for
 * the point of eval 16 (A = 1, B = 1) and its gradient; it is evaluated
 * again: PC_LAMMPS: ..."), and DIAGNOSTICS has "Repeats evaluated again :
 * 2" (the two calls) with the reason.
 * Before the fix: the same 30 points; neither journal line. */
TEST_CASE("Repeated points: a host whose evaluations can differ has every "
          "repeat evaluated, and the journal says why",
          "[HPO][repeat_points][nlopt]")
{
    CornerBowl host{{}, kBlocker};
    const std::string journal = run_nlopt(nlopt_spec("LD_MMA", 60), host);
    CHECK(host.asked.size() == 30u);
    CHECK(repeats_in(host.asked) == 10u);
    CHECK(journal.find("NLOPT:LD_MMA asked again for the point of eval 16 "
                       "(A = 1, B = 1) and its gradient; it is evaluated "
                       "again: " + kBlocker) != std::string::npos);
    CHECK(count_of(journal, "it is evaluated again: ") == 1u);
    CHECK(diagnostics_count(journal, "Repeats evaluated again") == "2");
    CHECK(journal.find("Repeats evaluated again : 2 (" + kBlocker + ")") !=
          std::string::npos);
    CHECK(diagnostics_count(journal, "Repeated points reused").empty());
}

/* Pins: a reuse is not an evaluation, so MAXEVAL does not count it. LD_MMA
 * makes 20 distinct evaluations on the bowl (the 30 calls of the case
 * above less its 10 repeats), all before its first repeat; at MAXEVAL 20
 * the reusing host is asked for those 20, the repeats after them are
 * answered from the record, and the search ends on its own test,
 * XTOL_REACHED. A host that evaluates the repeats is asked for the same 20,
 * and MAXEVAL ends its search at the first repeat, MAXEVAL_REACHED.
 * Before the fix: the reusing host too: 20 evaluations, MAXEVAL_REACHED. */
TEST_CASE("Repeated points: MAXEVAL does not count a reuse",
          "[HPO][repeat_points][nlopt][maxeval]")
{
    CornerBowl reuse{{}, ""};
    const std::string j_reuse = run_nlopt(nlopt_spec("LD_MMA", 20), reuse);
    CHECK(reuse.asked.size() == 20u);
    CHECK(repeats_in(reuse.asked) == 0u);
    CHECK(status_of(j_reuse) == "XTOL_REACHED -- parameter tolerance converged");

    CornerBowl again{{}, kBlocker};
    const std::string j_again = run_nlopt(nlopt_spec("LD_MMA", 20), again);
    CHECK(again.asked.size() == 20u);
    CHECK(repeats_in(again.asked) == 0u);
    CHECK(same_sequence(reuse.asked, again.asked));
    CHECK(status_of(j_again) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
}

/* Pins: LN_SBPLX (derivative-free), MAXEVAL 60, on the bowl: NLopt counts
 * its calls, a reused one among them, so the search still ends after 60
 * calls, MAXEVAL_REACHED, on the same points as with a host that evaluates
 * the repeats (60 points, 22 of them repeats), and the reusing host is
 * asked for 38. The journal's MAXEVAL line says so: "NLopt finished:
 * MAXEVAL_REACHED -- maximum evaluations exhausted after 38 of MAXEVAL 60
 * evaluation(s); NLopt counts its calls, and 22 of them asked for points
 * the run had evaluated and were answered without an evaluation" (it said
 * "asked again for a point" until a call answered from an earlier
 * search's evaluation was named as such).
 * Before the fix: 60 evaluations, 22 repeats, and "after 60 of MAXEVAL 60
 * evaluation(s)". */
TEST_CASE("Repeated points: a derivative-free NLopt search still ends after "
          "MAXEVAL of NLopt's calls, with fewer evaluations",
          "[HPO][repeat_points][nlopt][maxeval]")
{
    CornerBowl again{{}, kBlocker};
    const std::string j_again = run_nlopt(nlopt_spec("LN_SBPLX", 60), again);
    REQUIRE(again.asked.size() == 60u);
    REQUIRE(repeats_in(again.asked) == 22u);

    CornerBowl reuse{{}, ""};
    const std::string j_reuse = run_nlopt(nlopt_spec("LN_SBPLX", 60), reuse);
    CHECK(reuse.asked.size() == 38u);
    CHECK(repeats_in(reuse.asked) == 0u);
    CHECK(same_sequence(reuse.asked, first_asks(again.asked)));
    CHECK(status_of(j_reuse) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    CHECK(j_reuse.find("NLopt finished: MAXEVAL_REACHED -- maximum evaluations "
                       "exhausted after 38 of MAXEVAL 60 evaluation(s); NLopt "
                       "counts its calls, and 22 of them asked for points the "
                       "run had evaluated and were answered without an "
                       "evaluation") !=
          std::string::npos);
    CHECK(j_again.find("after 60 of MAXEVAL 60 evaluation(s)\n") !=
          std::string::npos);
}

namespace {

/* The bowl with a failed evaluation (FAILSCORE, the largest double) at the
 * corner (1, 1) itself. */
struct FailingCorner : CornerBowl {
    double operator()(const Vec& x) {
        asked.push_back(x);
        if (x(0) == 1.0 && x(1) == 1.0)
            return std::numeric_limits<double>::max();
        const double a = x(0) - 1.3, b = x(1) - 1.1;
        return a * a + b * b;
    }
};

std::size_t corner_asks(const std::vector<Vec>& asked)
{
    std::size_t n = 0;
    for (const Vec& p : asked) n += (p(0) == 1.0 && p(1) == 1.0) ? 1 : 0;
    return n;
}

}  // namespace

/* Pins: a failed evaluation is not reused. LN_SBPLX, MAXEVAL 60, asks for
 * the failed corner 20 times; the reusing host is asked for it 20 times
 * as well, and for no other point twice (the 4 other repeats are reused):
 * 56 points, 19 repeats, all of the corner.
 * Before the fix: 60 points, 23 repeats. */
TEST_CASE("Repeated points: a failed evaluation is evaluated again when its "
          "point comes again", "[HPO][repeat_points][nlopt]")
{
    FailingCorner host;
    host.reason = "";
    run_nlopt(nlopt_spec("LN_SBPLX", 60), host);
    CHECK(corner_asks(host.asked) == 20u);
    CHECK(repeats_in(host.asked) == 19u);
    CHECK(host.asked.size() == 56u);
}

namespace {

/* The bowl with its exact gradient under ANALYTICAL, formed, like
 * HPO_Host::analytical_gradient, from the last evaluation the host made:
 * each call records whether its point is that evaluation's. */
struct AnalyticalBowl : CornerBowl {
    std::size_t analytical = 0;
    std::size_t off_point = 0;   // gradients asked at another point
    bool analytical_gradient(const Vec& x, Vec& g, std::vector<int>& nfd) {
        ++analytical;
        if (asked.empty() || !same_bits(asked.back(), x)) ++off_point;
        g.set_size(2);
        g(0) = 2.0 * (x(0) - 1.3);
        g(1) = 2.0 * (x(1) - 1.1);
        nfd.clear();
        return true;
    }
};

}  // namespace

/* Pins the contract a reuse keeps: HPO_Host::analytical_gradient
 * differentiates the thread's last evaluation without checking its point,
 * so a gradient is asked for only at the point evaluated last. LD_MMA, and
 * G_MLSL_LDS (POPULATION 4) with INNER LD_MMA, whose local searches start
 * from sampled points evaluated without a gradient, under ANALYTICAL at
 * MAXEVAL 60 with a reusing host: no analytical gradient at another point,
 * and LD_MMA asks for no point twice.
 * Before the fix: passes as well for the contract (a pin: no loss was
 * reused, so every gradient followed its point's evaluation); LD_MMA asked
 * for the corner three times. */
TEST_CASE("Repeated points: an analytical gradient is formed only after an "
          "evaluation of its point", "[HPO][repeat_points][nlopt][analytical]")
{
    SECTION("LD_MMA") {
        AnalyticalBowl host;
        auto sp = nlopt_spec("LD_MMA", 60);
        sp.grad_strategy = GradStrategy::Analytical;
        sp.param.clear();
        sp.param_raw.clear();
        run_nlopt(sp, host);
        CHECK(host.analytical > 0u);
        CHECK(host.off_point == 0u);
        CHECK(repeats_in(host.asked) == 0u);
    }
    SECTION("G_MLSL_LDS, INNER LD_MMA") {
        AnalyticalBowl host;
        auto sp = nlopt_spec("G_MLSL_LDS", 60);
        sp.stop.xtol_rel.reset();
        sp.pop = 4;
        sp.grad_strategy = GradStrategy::Analytical;
        sp.param.clear();
        sp.param_raw.clear();
        OptimizerSpec in;
        in.lib = "NLOPT";
        in.algo = "LD_MMA";
        in.stop.xtol_rel = 1e-4;
        sp.inner = std::make_unique<OptimizerSpec>(in);
        run_nlopt(sp, host);
        CHECK(host.analytical > 0u);
        CHECK(host.off_point == 0u);
    }
}

/* Pins: G_MLSL_LDS (POPULATION 4) with INNER LD_MMA under GRAD_STRATEGY
 * NUMERICAL (the default), MAXEVAL 60, on the bowl with a reusing host:
 * MLSL evaluates its samples without a gradient, and a local search that
 * starts from one asks for the gradient there; the recorded loss is the
 * loss at the point that the finite differences take, so only the stencil
 * is evaluated: 60 evaluations, none of a point asked for before. The
 * journal says so once ("NLOPT:G_MLSL_LDS asked again for the point of
 * eval 2 (A = 0.59375, B = 0.96875) and its gradient: the loss of eval 2 is
 * reused, without an evaluation, as the loss at the point that the
 * gradient's finite differences take, and their stencil is evaluated"),
 * and DIAGNOSTICS has no "Repeats evaluated again".
 * Before the fix: 3 of the 60 evaluations repeated an earlier one (evals
 * 6, 21 and 40 those of 2, 3 and 36, local starts), each "evaluated again:
 * a gradient was asked for where the earlier evaluation formed none", and
 * DIAGNOSTICS had "Repeats evaluated again : 3". */
TEST_CASE("Repeated points: under NUMERICAL a gradient at a recorded point "
          "takes its recorded loss", "[HPO][repeat_points][nlopt]")
{
    CornerBowl host{{}, ""};
    auto sp = nlopt_spec("G_MLSL_LDS", 60);
    sp.stop.xtol_rel.reset();
    sp.pop = 4;
    OptimizerSpec in;
    in.lib = "NLOPT";
    in.algo = "LD_MMA";
    in.stop.xtol_rel = 1e-4;
    sp.inner = std::make_unique<OptimizerSpec>(in);
    const std::string journal = run_nlopt(sp, host);
    CHECK(host.asked.size() == 60u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(journal.find("NLOPT:G_MLSL_LDS asked again for the point of eval 2 "
                       "(A = 0.59375, B = 0.96875) and its gradient: the loss "
                       "of eval 2 is reused, without an evaluation, as the "
                       "loss at the point that the gradient's finite "
                       "differences take, and their stencil is evaluated") !=
          std::string::npos);
    CHECK(diagnostics_count(journal, "Repeats evaluated again").empty());
    CHECK_FALSE(diagnostics_count(journal, "Repeated points reused").empty());
}

/* ======================================================================
 * DLIB:BFGS
 * ==================================================================== */

namespace {

OptimizerSpec bfgs_spec(std::vector<double> low, std::vector<double> high,
                        std::vector<double> start)
{
    OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    const std::size_t n = low.size();
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(n, false);
    sp.log_scaled.assign(n, false);
    sp.analyse_logscale();
    sp.parameter_labels = n == 1 ? std::vector<std::string>{"A"}
                                 : std::vector<std::string>{"A", "B"};
    sp.optim_init = {std::move(start)};
    sp.param["fd_step"] = 1e-6;
    sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = std::numeric_limits<double>::max();
    return sp;
}

/* 1 - u on [0, 1]: the loss falls towards the upper bound, where the
 * search starts. With `noisy`, a point asked for again gives a loss one ulp
 * above its last one each time -- what the warm-started relaxation of a
 * PC_LAMMPS --hotvar does to the RCUT2B bound of the search of
 * data/ta/target_files/init_warm.hpo (~1e-13 relative), with a
 * host that says its evaluations can differ. */
struct Slope {
    std::vector<Vec> asked;
    bool noisy = false;
    std::string reason;
    std::map<double, double> last;   // u -> the loss it gave last
    double operator()(const Vec& x) {
        asked.push_back(x);
        double y = 1.0 - x(0);
        if (noisy) {
            const auto it = last.find(x(0));
            if (it != last.end())
                y = std::nextafter(it->second,
                                   std::numeric_limits<double>::infinity());
            last[x(0)] = y;
        }
        return y;
    }
    std::string repeat_reuse_blocker() const { return reason; }
    std::size_t g_iter_value() const { return asked.size(); }
};

template <class Host>
std::string run_bfgs(const OptimizerSpec& sp, Host& host)
{
    Journal j;
    REQUIRE_NOTHROW(tadah::hpo::run_dlib_bfgs_impl<Vec, NoRunner>(sp, host));
    return j.text();
}

}  // namespace

/* Pins: DLIB:BFGS started on the upper bound of [0, 1], the loss falling
 * towards it (1 - u): the start and its one-sided stencil, 3 evaluations;
 * the step, held to the box, cannot move the point, so the line search is
 * not run and the loss does not change: "DLIB:BFGS: FTOL_ABS reached"
 * after 3 evaluations. The journal says so once (INFO "DLIB:BFGS: the step
 * from A = 1 cannot move the point within the OPTIM box: the line search
 * is not run"), DIAGNOSTICS has "Line searches not run : 1". The same with
 * a host whose re-evaluated loss rises by an ulp each time (a PC_LAMMPS
 * --hotvar relaxation), at MAXEVAL 30.
 * Before the fix (dlib's find_min_box_constrained): the line search
 * evaluated the start again and BFGS took its gradient again, 6
 * evaluations, 3 of them repeats, FTOL_ABS; with the rising loss it failed
 * its decrease test at every re-evaluation of the bound and evaluated it
 * until MAXEVAL: 30 evaluations, 27 of them at u = 1, "DLIB:BFGS: MAXEVAL
 * reached" (the search of data/ta/target_files/init_warm.hpo: 28 of 30). */
TEST_CASE("Repeated points: DLIB:BFGS runs no line search that cannot move "
          "the point", "[HPO][repeat_points][dlib_bfgs]")
{
    for (const bool noisy : {false, true}) {
        DYNAMIC_SECTION((noisy ? "a re-evaluated loss rises by an ulp"
                               : "a function of the point")) {
            Slope host;
            host.noisy = noisy;
            host.reason = noisy ? kBlocker : "";
            auto sp = bfgs_spec({0.0}, {1.0}, {1.0});
            sp.stop.maxeval = 30;
            const std::string journal = run_bfgs(sp, host);
            CHECK(host.asked.size() == 3u);
            CHECK(repeats_in(host.asked) == 0u);
            CHECK(status_of(journal) == "DLIB:BFGS: FTOL_ABS reached");
            CHECK(journal.find("DLIB:BFGS stopped: FTOL_ABS reached after 3 "
                               "evaluation(s)") != std::string::npos);
            CHECK(journal.find("DLIB:BFGS: the step from A = 1 cannot move "
                               "the point within the OPTIM box: the line "
                               "search is not run") != std::string::npos);
            CHECK(diagnostics_count(journal, "Line searches not run") == "1");
        }
    }
}

/* Pins: PARAM tol_gradinf, whose test reads the whole gradient, does not
 * end a search held at the corner of the box: on the steep bowl (scale
 * 100) from (0.5, 0.5) at MAXEVAL 40, BFGS reaches the corner, takes its
 * gradient there, cannot move, and its steepest-descent step (the BFGS
 * update resets after a step of zero) cannot either: the search ends
 * "DLIB:BFGS: no step within the OPTIM box", with the stop line "DLIB:BFGS
 * stopped: no step within the OPTIM box after N evaluation(s): A at its
 * upper bound 1, B at its upper bound 1, each with a gradient that points
 * out of the box", well within MAXEVAL, and no point evaluated twice
 * (a reusing host).
 * Before the fix: the corner and its gradient were evaluated again and
 * again: 40 evaluations, "DLIB:BFGS: MAXEVAL reached"; without MAXEVAL the
 * search never ended. */
TEST_CASE("Repeated points: DLIB:BFGS ends a search that no step can move",
          "[HPO][repeat_points][dlib_bfgs]")
{
    CornerBowl host{{}, "", 100.0};
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.stop.maxeval = 40;
    sp.param["tol_gradinf"] = 1e-12;
    const std::string journal = run_bfgs(sp, host);
    CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
    CHECK(host.asked.size() < 20u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(journal.find("DLIB:BFGS stopped: no step within the OPTIM box after " +
                       std::to_string(host.asked.size()) +
                       " evaluation(s): A at its upper bound 1, B at its upper "
                       "bound 1, each with a gradient that points out of the "
                       "box") != std::string::npos);
}

/* Pins: on the steep bowl (scale 100) from (0.5, 0.5) at MAXEVAL 60, the
 * first line search's trials are clamped onto the corner (1, 1) until its
 * decrease test passes there. A reusing host is asked for the corner once:
 * the later trials get its loss; a host that evaluates the repeats is asked
 * for it on every trial, and for nothing else twice. Both reach the corner
 * and end "DLIB:BFGS: FTOL_ABS reached" without a line search at the
 * corner.
 * Before the fix: both hosts were asked for the corner on every trial and
 * again by the line search from the corner, which also took its gradient
 * again (5 repeats more than now on the host that evaluates the repeats). */
TEST_CASE("Repeated points: DLIB:BFGS reuses a line-search trial the box "
          "put on a point it evaluated", "[HPO][repeat_points][dlib_bfgs]")
{
    CornerBowl again{{}, kBlocker, 100.0};
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.stop.maxeval = 60;
    const std::string j_again = run_bfgs(sp, again);
    CornerBowl reuse{{}, "", 100.0};
    const std::string j_reuse = run_bfgs(sp, reuse);

    const std::size_t trial_repeats = repeats_in(again.asked);
    CHECK(trial_repeats > 0u);
    CHECK(corner_asks(again.asked) == trial_repeats + 1u);
    CHECK(repeats_in(reuse.asked) == 0u);
    CHECK(same_sequence(reuse.asked, first_asks(again.asked)));
    CHECK(status_of(j_again) == "DLIB:BFGS: FTOL_ABS reached");
    CHECK(status_of(j_reuse) == "DLIB:BFGS: FTOL_ABS reached");
    CHECK(diagnostics_count(j_reuse, "Repeated points reused") ==
          std::to_string(trial_repeats));
    CHECK(diagnostics_count(j_again, "Repeats evaluated again") ==
          std::to_string(trial_repeats));
}

/* Pins: how each backend's MAXEVAL takes a reuse, as the journal says it.
 * NLopt (LN_SBPLX, MAXEVAL 60): the first reuse's INFO line says "a reuse
 * is no evaluation: HPO's MAXEVAL does not count it, but NLopt counts it
 * among the calls its own MAXEVAL bounds", which is what ends the search
 * after 38 evaluations ("NLopt counts its calls, and 22 of them asked
 * again"); DLIB:BFGS (the steep bowl, whose line-search trials the box puts
 * on the corner): "a reuse is no evaluation, and MAXEVAL counts
 * evaluations". DIAGNOSTICS says both: "points the run had evaluated that a
 * search asked for, ... HPO's MAXEVAL does not count them, NLopt's own
 * MAXEVAL counts them among its calls, and the RESTART window counts each
 * as one evaluation without a new best".
 * Before the fix: both INFO lines said "(a reuse is no evaluation, and
 * MAXEVAL counts evaluations; ...)" and DIAGNOSTICS "not evaluated again,
 * not counted in MAXEVAL", while NLopt's own MAXEVAL counted the 22 reuses
 * and ended the search 22 evaluations early; until the run kept one record
 * (2026-10-12) DIAGNOSTICS said "points a search asked for again", the
 * record being the search's. */
TEST_CASE("Repeated points: the journal says how MAXEVAL takes a reuse",
          "[HPO][repeat_points][nlopt][dlib_bfgs][maxeval]")
{
    CornerBowl sbplx{{}, ""};
    const std::string j_nlopt = run_nlopt(nlopt_spec("LN_SBPLX", 60), sbplx);
    CHECK(sbplx.asked.size() == 38u);
    CHECK(j_nlopt.find("NLOPT:LN_SBPLX asked again for the point of eval ") !=
          std::string::npos);
    CHECK(j_nlopt.find("is reused, without an evaluation (a reuse is no "
                       "evaluation: HPO's MAXEVAL does not count it, but NLopt "
                       "counts it among the calls its own MAXEVAL bounds; "
                       "DIAGNOSTICS counts reuses)") != std::string::npos);
    CHECK(j_nlopt.find("NLopt counts its calls, and 22 of them asked for "
                       "points the run had evaluated") !=
          std::string::npos);
    const std::string diag =
        "points the run had evaluated that a search asked for, answered "
        "with the\n    loss -- and gradient -- of their evaluation, not "
        "evaluated again: HPO's MAXEVAL\n    does not count them, NLopt's own "
        "MAXEVAL counts them among its calls, and the\n    RESTART window "
        "counts each as one evaluation without a new best)";
    CHECK(j_nlopt.find(diag) != std::string::npos);
    CHECK(j_nlopt.find("not counted in\n    MAXEVAL") == std::string::npos);

    CornerBowl bowl{{}, "", 100.0};
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.stop.maxeval = 60;
    const std::string j_bfgs = run_bfgs(sp, bowl);
    CHECK(j_bfgs.find("DLIB:BFGS asked again for the point of eval ") !=
          std::string::npos);
    CHECK(j_bfgs.find("is reused, without an evaluation (a reuse is no "
                      "evaluation, and MAXEVAL counts evaluations; DIAGNOSTICS "
                      "counts reuses)") != std::string::npos);
    CHECK(j_bfgs.find(diag) != std::string::npos);
}

/* Pins: the loop DLIB:BFGS runs (detail::bfgs_box_search) is dlib's
 * find_min_box_constrained wherever every step moves the point: on
 * Rosenbrock's valley in [-2, 2]^2 from (-1.2, 1), with its exact gradient,
 * BFGS and the objective-delta stop (1e-9), both ask for the same losses
 * and gradients at the same points, bit for bit, in the same order, and end
 * on the same point and loss.
 * Before the fix: does not compile (bfgs_box_search is new); a pin that
 * the loop replacing dlib's keeps dlib's trajectory. */
TEST_CASE("Repeated points: DLIB:BFGS's loop takes dlib's steps",
          "[HPO][repeat_points][dlib_bfgs]")
{
    const auto valley = [](const Vec& x) {
        const double a = 1.0 - x(0), b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
    };
    const auto valley_grad = [](const Vec& x) {
        Vec g(2);
        const double b = x(1) - x(0) * x(0);
        g(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * b;
        g(1) = 200.0 * b;
        return g;
    };
    const Vec lo = point(-2.0, -2.0), hi = point(2.0, 2.0);

    std::vector<Vec> dlib_f, dlib_g;
    Vec xd = point(-1.2, 1.0);
    const double fd = dlib::find_min_box_constrained(
        dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-9),
        [&](const Vec& x) { dlib_f.push_back(x); return valley(x); },
        [&](const Vec& x) { dlib_g.push_back(x); return valley_grad(x); },
        xd, lo, hi);

    std::vector<Vec> ours_f, ours_g;
    Vec xo = point(-1.2, 1.0);
    double fo = 0.0;
    const auto r = tadah::hpo::detail::bfgs_box_search(
        dlib::bfgs_search_strategy(), dlib::objective_delta_stop_strategy(1e-9),
        [&](const Vec& x) { ours_f.push_back(x); return valley(x); },
        [&](const Vec& x) { ours_g.push_back(x); return valley_grad(x); },
        xo, lo, hi, fo, [](const Vec&) {});
    CHECK(r.end == tadah::hpo::detail::BfgsBoxEnd::Stop);
    CHECK(r.line_searches_not_run == 0u);
    CHECK(dlib_f.size() > 20u);
    CHECK(same_sequence(ours_f, dlib_f));
    CHECK(same_sequence(ours_g, dlib_g));
    CHECK(same_bits(xo, xd));
    CHECK(std::memcmp(&fo, &fd, sizeof fo) == 0);
}

namespace {

/* A host whose analytical gradient, like HPO_Host::analytical_gradient,
 * belongs to its last evaluation; it counts gradients asked at another
 * point. */
struct LastEvaluationGradient {
    std::vector<Vec> asked;
    std::size_t off_point = 0;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return x(0) * x(0) + x(1) * x(1);
    }
    bool analytical_gradient(const Vec& x, Vec& g, std::vector<int>& nfd) {
        if (asked.empty() || !same_bits(asked.back(), x)) ++off_point;
        g.set_size(2);
        g(0) = 2.0 * x(0);
        g(1) = 2.0 * x(1);
        nfd.clear();
        return true;
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

}  // namespace

/* Pins: DLIB:BFGS's objective (detail::BfgsObjective) under ANALYTICAL
 * with a reusing host: the losses of P = (0.5, 0.5) and Q = (0.25, 0.75),
 * then P's again -- reused, no evaluation -- and then P's gradient: the
 * host's last evaluation is Q's, so P is evaluated first (3 evaluations in
 * all, the third P) and the gradient is P's, (1, 1); a second gradient of P
 * is the recorded one, without an evaluation.
 * Before the fix: does not compile (BfgsObjective is new). The objective's
 * gradient of a point that was not its last evaluation used to call the
 * provider's gradient(x), which under ANALYTICAL formed the gradient of the
 * host's last evaluation without one of x; dlib's loop never asked for
 * such a gradient, a reuse can. */
TEST_CASE("Repeated points: DLIB:BFGS evaluates a point before an analytical "
          "gradient there when its last evaluation was another point",
          "[HPO][repeat_points][dlib_bfgs][analytical]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.grad_strategy = GradStrategy::Analytical;
    sp.param.clear();
    sp.param_raw.clear();
    LastEvaluationGradient host;
    Journal j;
    tadah::hpo::detail::BfgsObjective<Vec, NoRunner, LastEvaluationGradient>
        objective(sp, host, 0);
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
    CHECK(objective.loss(P) == 0.5);
    CHECK(objective.loss(Q) == 0.625);
    CHECK(objective.loss(P) == 0.5);
    CHECK(host.asked.size() == 2u);
    const Vec g = objective.gradient(P);
    CHECK(host.asked.size() == 3u);
    CHECK(same_bits(host.asked.back(), P));
    CHECK(host.off_point == 0u);
    CHECK(g(0) == 1.0);
    CHECK(g(1) == 1.0);
    const Vec g2 = objective.gradient(P);
    CHECK(host.asked.size() == 3u);
    CHECK(same_bits(g2, g));
    CHECK(objective.evaluations() == 4u);   // 3 losses, 1 analytical gradient
}

/* Pins: DLIB:BFGS's objective (detail::BfgsObjective) under NUMERICAL
 * (CENTRAL, fd_step 1e-6) with a reusing host: the losses of P = (0.5,
 * 0.5) and Q = (0.25, 0.75), then P's gradient: the recorded loss of P is
 * the loss at the point that the finite differences take, so the host is
 * asked for P, Q and P's 4-point stencil, P not again (6 evaluations); the
 * gradient is P's, (1, 1) to 1e-6; a second gradient of P is the recorded
 * one, without an evaluation.
 * Before the fix: P was evaluated again before its stencil (7 evaluations,
 * the third P), "it is evaluated again: a gradient was asked for where the
 * earlier evaluation formed none". */
TEST_CASE("Repeated points: DLIB:BFGS takes the recorded loss of a point as "
          "the centre of its finite differences",
          "[HPO][repeat_points][dlib_bfgs]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    LastEvaluationGradient host;
    Journal j;
    tadah::hpo::detail::BfgsObjective<Vec, NoRunner, LastEvaluationGradient>
        objective(sp, host, 0);
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
    CHECK(objective.loss(P) == 0.5);
    CHECK(objective.loss(Q) == 0.625);
    const Vec g = objective.gradient(P);
    CHECK(host.asked.size() == 6u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(std::abs(g(0) - 1.0) < 1e-6);
    CHECK(std::abs(g(1) - 1.0) < 1e-6);
    const Vec g2 = objective.gradient(P);
    CHECK(host.asked.size() == 6u);
    CHECK(same_bits(g2, g));
    CHECK(objective.evaluations() == 6u);
    const std::string journal = j.text();
    CHECK(journal.find("DLIB:BFGS asked again for the point of eval 1 (A = "
                       "0.5, B = 0.5) and its gradient: the loss of eval 1 is "
                       "reused, without an evaluation, as the loss at the "
                       "point that the gradient's finite differences take") !=
          std::string::npos);
    CHECK(diagnostics_count(journal, "Repeats evaluated again").empty());
}

namespace {

/* The stop strategy `Inner` of a bfgs_box_search that ends the search
 * after `cap` iterations and counts them: a loop that would not end makes
 * the case fail instead of hang. */
template <class Inner>
struct CappedStop {
    Inner inner;
    std::size_t cap;
    std::size_t* iterations;
    template <class T>
    bool should_continue_search(const T& x, double f, const T& g)
    {
        if (++*iterations > cap) return false;
        return inner.should_continue_search(x, f, g);
    }
};

constexpr std::size_t kIterationCap = 100000;

/* 1 for u < 0.5 and 2 from 0.5 on: a jump (a FAILSCORE edge, a neighbour
 * shell): from 1e-9 left of it the central difference (h = 1e-6) is
 * 1/(2h), while the loss is flat. */
double jump_loss(double u) { return u < 0.5 ? 1.0 : 2.0; }
const double kJumpStart = 0.5 - 1e-9;

struct JumpHost {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return jump_loss(x(0));
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

/* f(A, B) = 1 + 1000 (6 - A) + q(B) on [4, 6] x [0, 1], from A one ulp
 * below 6 (inside dlib's 1e-8 gap of its bound, with a steep gradient out
 * of the box) and B = 0.6; q is 0.4 (B - 0.5) above 0.5, 0 on [0.45, 0.5)
 * and 100 (0.45 - B) below: BFGS ends B's line search on the flat part. */
double gap_q(double b)
{
    if (b >= 0.5) return 0.4 * (b - 0.5);
    if (b >= 0.45) return 0.0;
    return 100.0 * (0.45 - b);
}
double gap_loss(const Vec& x) { return 1.0 + 1000.0 * (6.0 - x(0)) + gap_q(x(1)); }
Vec gap_grad(const Vec& x)
{
    Vec g(2);
    g(0) = -1000.0;
    g(1) = x(1) >= 0.5 ? 0.4 : (x(1) >= 0.45 ? 0.0 : -100.0);
    return g;
}
Vec gap_start() { return point(std::nextafter(6.0, 0.0), 0.6); }

struct GapHost {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return gap_loss(x);
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

/* 1 + 6 (u - 0.5) above 0.5, 1 + 4 (0.5 - u) below: the minimum at the
 * start 0.5, where the central difference is 1 though the loss rises on
 * both sides, so every step along it comes back. `reason` as CornerBowl's. */
struct VHost {
    std::vector<Vec> asked;
    std::string reason;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double u = x(0);
        return u >= 0.5 ? 1.0 + 6.0 * (u - 0.5) : 1.0 + 4.0 * (0.5 - u);
    }
    std::string repeat_reuse_blocker() const { return reason; }
    std::size_t g_iter_value() const { return asked.size(); }
};

/* run_bfgs, which must return within a minute: a search that asks for no
 * evaluation could spin. */
template <class Host>
std::string run_bfgs_within(const OptimizerSpec& sp, Host& host,
                            const std::string& what)
{
    Journal j;
    hpo_test::finish_within(std::chrono::seconds(60), what, [&] {
        tadah::hpo::run_dlib_bfgs_impl<Vec, NoRunner>(sp, host);
    });
    return j.text();
}

}  // namespace

/* Pins: bfgs_box_search at the jump, from 1e-9 left of it, with the
 * central difference of the jump (5e5) as the gradient, under PARAM
 * tol_gradinf (1e-12) and under FTOL_ABS 0: the first line search shrinks
 * its step until its trial is the start itself, and the next iteration's
 * steepest-descent step from there cannot move the point at the farthest
 * first step dlib takes from it (its step length, ten times the largest
 * step whose decrease the loss's rounding hides, ~4e-26, against ~1e-22
 * that moves A): the search ends NoStep within 3 iterations, the point
 * where it started, without the line search from R (came_back false).
 * Before the fix: the iterations after the first asked for no loss and
 * their step length cycled below the step that moves the point: the cap of
 * 100000 iterations ended the search (BfgsBoxEnd::Stop); dlib's
 * find_min_box_constrained evaluated the start at each of its iterations
 * until MAXEVAL. */
TEST_CASE("Repeated points: DLIB:BFGS ends a search held next to a jump in "
          "the loss", "[HPO][repeat_points][dlib_bfgs]")
{
    const double h = 1e-6;
    std::size_t evaluations = 0;
    const auto f = [&](const Vec& x) {
        ++evaluations;
        return jump_loss(x(0));
    };
    const auto der = [&](const Vec& x) {
        Vec g(1);
        g(0) = (f(Vec(x + h)) - f(Vec(x - h))) / (2 * h);
        return g;
    };
    Vec lo(1), hi(1);
    lo = 0.0;
    hi = 1.0;
    const auto check = [&](auto stop) {
        evaluations = 0;
        std::size_t iterations = 0;
        Vec x(1);
        x = kJumpStart;
        double fv = 0.0;
        const auto r = tadah::hpo::detail::bfgs_box_search(
            dlib::bfgs_search_strategy(),
            CappedStop<decltype(stop)>{stop, kIterationCap, &iterations}, f,
            der, x, lo, hi, fv, [](const Vec&) {});
        CHECK(r.end == tadah::hpo::detail::BfgsBoxEnd::NoStep);
        CHECK(iterations <= 3u);
        CHECK_FALSE(r.came_back);
        CHECK(x(0) == kJumpStart);
        CHECK(fv == 1.0);
    };
    SECTION("PARAM tol_gradinf") {
        check(dlib::gradient_norm_stop_strategy(1e-12));
    }
    SECTION("FTOL_ABS 0") {
        check(dlib::objective_delta_stop_strategy(0.0));
    }
}

/* Pins: run_dlib_bfgs_impl at the jump (fd_step 1e-6, inhouse, MAXEVAL 50,
 * a reusing host), under PARAM tol_gradinf 1e-12 and under FTOL_ABS 0:
 * the search returns, after 49 evaluations (the start, its stencil, and
 * the first line search's trials left of the start), with "DLIB:BFGS: no
 * step within the OPTIM box" and the stop line "... after 49
 * evaluation(s): A = 0.499999999, whose step is below the rounding of its
 * value".
 * Before the fix: it never returned (the host made its 49 evaluations,
 * and the iterations after them asked for none), and a minute ended the
 * case. */
TEST_CASE("Repeated points: DLIB:BFGS held next to a jump returns, under "
          "PARAM tol_gradinf", "[HPO][repeat_points][dlib_bfgs]")
{
    JumpHost host;
    auto sp = bfgs_spec({0.0}, {1.0}, {kJumpStart});
    sp.stop.maxeval = 50;
    sp.param["tol_gradinf"] = 1e-12;
    const std::string journal =
        run_bfgs_within(sp, host, "DLIB:BFGS at the jump, tol_gradinf");
    CHECK(host.asked.size() == 49u);
    CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
    CHECK(journal.find("DLIB:BFGS stopped: no step within the OPTIM box after "
                       "49 evaluation(s): A = 0.499999999, whose step is "
                       "below the rounding of its value") != std::string::npos);
}

/* Pins: the same under FTOL_ABS 0 (no change of the loss passes it).
 * Before the fix: as under tol_gradinf, it never returned. */
TEST_CASE("Repeated points: DLIB:BFGS held next to a jump returns, under "
          "FTOL_ABS 0", "[HPO][repeat_points][dlib_bfgs]")
{
    JumpHost host;
    auto sp = bfgs_spec({0.0}, {1.0}, {kJumpStart});
    sp.stop.maxeval = 50;
    sp.stop.ftol_abs = 0.0;
    const std::string journal =
        run_bfgs_within(sp, host, "DLIB:BFGS at the jump, FTOL_ABS 0");
    CHECK(host.asked.size() == 49u);
    CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
}

/* Pins: bfgs_box_search with A one ulp below its upper bound 6, inside
 * dlib's 1e-8 gap, its gradient -1000 pointing out of the box, and B's
 * first line search ending on the flat part of q (exact gradients), under
 * PARAM tol_gradinf and FTOL_ABS 0: the next iterations cannot move the
 * point (dlib's gap step for A is 6 - A, 8.9e-16, and the farthest first
 * step dlib takes from there, ~0.12 of it, rounds back onto A; B's gradient
 * is 0): the search ends NoStep within 10 iterations, at A one ulp below 6,
 * B = 0.48.
 * Before the fix: the iterations after B's line search asked for no loss,
 * and their step length cycled between ~0.01 and ~0.1, below the ~0.5 that
 * moves A: the cap of 100000 iterations ended the search (dlib's
 * find_min_box_constrained reached the same point and evaluated it until
 * MAXEVAL). */
TEST_CASE("Repeated points: DLIB:BFGS ends a search held in dlib's gap of "
          "a bound", "[HPO][repeat_points][dlib_bfgs]")
{
    Vec lo(2), hi(2);
    lo = 4.0, 0.0;
    hi = 6.0, 1.0;
    const auto check = [&](auto stop) {
        std::size_t iterations = 0;
        Vec x = gap_start();
        double fv = 0.0;
        const auto r = tadah::hpo::detail::bfgs_box_search(
            dlib::bfgs_search_strategy(),
            CappedStop<decltype(stop)>{stop, kIterationCap, &iterations},
            [](const Vec& v) { return gap_loss(v); },
            [](const Vec& v) { return gap_grad(v); }, x, lo, hi, fv,
            [](const Vec&) {});
        CHECK(r.end == tadah::hpo::detail::BfgsBoxEnd::NoStep);
        CHECK(iterations <= 10u);
        CHECK_FALSE(r.came_back);
        CHECK(x(0) == std::nextafter(6.0, 0.0));
        CHECK(std::abs(x(1) - 0.48) < 1e-12);
    };
    SECTION("PARAM tol_gradinf") {
        check(dlib::gradient_norm_stop_strategy(1e-12));
    }
    SECTION("FTOL_ABS 0") {
        check(dlib::objective_delta_stop_strategy(0.0));
    }
}

/* Pins: run_dlib_bfgs_impl on the gap case (MAXEVAL 50, PARAM tol_gradinf
 * 1e-12, a reusing host): it returns after 22 evaluations, "DLIB:BFGS: no
 * step within the OPTIM box", the stop line "... after 22 evaluation(s): A
 * at its upper bound 6, with a gradient that points out of the box; B =
 * 0.48, with a gradient of zero".
 * Before the fix: it never returned after its 22 evaluations, and a minute
 * ended the case. */
TEST_CASE("Repeated points: DLIB:BFGS held in dlib's gap of a bound returns",
          "[HPO][repeat_points][dlib_bfgs]")
{
    GapHost host;
    const Vec start = gap_start();
    auto sp = bfgs_spec({4.0, 0.0}, {6.0, 1.0}, {start(0), start(1)});
    sp.stop.maxeval = 50;
    sp.param["tol_gradinf"] = 1e-12;
    const std::string journal =
        run_bfgs_within(sp, host, "DLIB:BFGS in the gap of a bound");
    CHECK(host.asked.size() == 22u);
    CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
    CHECK(journal.find("DLIB:BFGS stopped: no step within the OPTIM box after "
                       "22 evaluation(s): A at its upper bound 6, with a "
                       "gradient that points out of the box; B = 0.48, with "
                       "a gradient of zero") != std::string::npos);
}

/* Pins: run_dlib_bfgs_impl from the minimum of VHost (MAXEVAL 200, PARAM
 * tol_gradinf 1e-12): the first line search comes back to the start (17
 * trials left of it, the last one the start itself); from there the first
 * step cannot move the point but the farthest step dlib takes from it,
 * 5.55e-14 of the steepest-descent step, can: the line search from it is
 * run once (4 trials), comes back, and the search ends "DLIB:BFGS: no step
 * within the OPTIM box", the stop line "... after 24 evaluation(s): from A
 * = 0.5 the line search from the farthest step BFGS takes there
 * (5.551115122e-14 of the steepest-descent step) found no lower loss and
 * came back to the point". A reusing host is asked for 24 points, none
 * twice; a host that evaluates repeats for 26, the start twice more (the
 * two trials that were the start).
 * Before the fix: after the first line search the step length grew without
 * an evaluation until a trial moved the point, whose line search came back
 * again: with a reusing host the same trials every time, all reused, and
 * the search never returned (a minute ended the case); a host that
 * evaluates repeats spent MAXEVAL 200, "DLIB:BFGS: MAXEVAL reached". */
TEST_CASE("Repeated points: DLIB:BFGS runs the line search from its farthest "
          "step once at a stalled point", "[HPO][repeat_points][dlib_bfgs]")
{
    auto sp = bfgs_spec({0.0}, {1.0}, {0.5});
    sp.stop.maxeval = 200;
    sp.param["tol_gradinf"] = 1e-12;
    const std::string stop_line =
        "DLIB:BFGS stopped: no step within the OPTIM box after N "
        "evaluation(s): from A = 0.5 the line search from the farthest step "
        "BFGS takes there (5.551115122e-14 of the steepest-descent step) "
        "found no lower loss and came back to the point";
    const auto with_count = [&](std::size_t n) {
        std::string line = stop_line;
        line.replace(line.find(" N "), 3, " " + std::to_string(n) + " ");
        return line;
    };
    SECTION("a host whose every evaluation is a function of its point") {
        VHost host{{}, ""};
        const std::string journal =
            run_bfgs_within(sp, host, "DLIB:BFGS at a stalled point (reuse)");
        CHECK(host.asked.size() == 24u);
        CHECK(repeats_in(host.asked) == 0u);
        CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
        CHECK(journal.find(with_count(24)) != std::string::npos);
    }
    SECTION("a host that evaluates repeats") {
        VHost host{{}, kBlocker};
        const std::string journal = run_bfgs_within(
            sp, host, "DLIB:BFGS at a stalled point (evaluated again)");
        CHECK(host.asked.size() == 26u);
        CHECK(repeats_in(host.asked) == 2u);
        CHECK(status_of(journal) == "DLIB:BFGS: no step within the OPTIM box");
        CHECK(journal.find(with_count(26)) != std::string::npos);
    }
}

namespace {

/* 23207.610197404942 + 0.0034262213715514594 |u - 0.89349363830995211| on
 * [0, 1]: a V whose kink term, near the kink, is below the rounding of the
 * offset -- a plateau of the loss's rounding, where BFGS's steps all pass
 * their decrease test (the decrease asked for is lost in the rounding) and
 * the loss comes back equal at different points. Found by a fuzz of
 * DLIB:BFGS over random losses on [0, 1]^n (14 of 3000 searches never
 * ended, every call answered from the record). */
struct PlateauV {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return 23207.610197404942 +
               0.0034262213715514594 * std::fabs(x(0) - 0.89349363830995211);
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

/* 66.758239935890771 + 53.903606516122373 (A + 0.26236689683470005)^2
 * + 33.466698031961712 (B - 0.23311259478514823)^2 on [0, 1]^2: a smooth
 * bowl whose minimum over the box has A held at its lower bound 0 (A's
 * gradient there is 28, so PARAM tol_gradinf never holds) and B inside,
 * where the B term near its minimum is below the rounding of the rest. The
 * fuzz's other search that never ended. */
struct BoundBowl {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double a = x(0) - (-0.26236689683470005);
        const double b = x(1) - 0.23311259478514823;
        double q = 0.0;
        q += 53.903606516122373 * a * a;
        q += 33.466698031961712 * b * b;
        return 66.758239935890771 + q;
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

/* The journal's "DLIB:BFGS stopped:" line (its text after the stamp). */
std::string bfgs_stop_line(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"((DLIB:BFGS stopped: [^\n]*))")))
        return m[1];
    return "(no stop line)";
}

}  // namespace

/* Pins: run_dlib_bfgs_impl on the plateau (PlateauV, from u =
 * 0.96678801528256708, fd_step 1e-6 inhouse, FTOL_ABS 0, MAXEVAL 400, a
 * reusing host) ends after 111 evaluations, none of a point asked for
 * before, "DLIB:BFGS: back at an earlier point", with the stop line
 * "DLIB:BFGS stopped: back at an earlier point after 111 evaluation(s): the
 * iteration from A = 0.89349363928454106 (the point of eval 109) made no
 * evaluation: it asked only for points the run had evaluated, and took the
 * search back to A = 0.89349363928412684 (the point of eval 98), where it
 * had stood with the same loss and gradient": an iteration that moves the
 * point without an evaluation ends the search at the next stop test; the
 * best, eval 95, is the one the search would find.
 * Before the fix: the search went on from there, made 12 evaluations more
 * (123; none lower than eval 95), then went round points it had evaluated
 * (the loss 23207.610197404945 at each), every call answered from the
 * record, without end: MAXEVAL, which counts evaluations, never fired, and
 * a minute ended the case. */
TEST_CASE("Repeated points: DLIB:BFGS on a plateau of the loss's rounding "
          "ends back at an earlier point, under FTOL_ABS 0",
          "[HPO][repeat_points][dlib_bfgs][revisit]")
{
    PlateauV host;
    auto sp = bfgs_spec({0.0}, {1.0}, {0.96678801528256708});
    sp.stop.maxeval = 400;
    sp.stop.ftol_abs = 0.0;
    const std::string journal =
        run_bfgs_within(sp, host, "DLIB:BFGS on the rounding plateau");
    CHECK(host.asked.size() == 111u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(status_of(journal) == "DLIB:BFGS: back at an earlier point");
    CHECK(bfgs_stop_line(journal) ==
          "DLIB:BFGS stopped: back at an earlier point after 111 "
          "evaluation(s): the iteration from A = 0.89349363928454106 (the "
          "point of eval 109) made no evaluation: it asked only for points "
          "the run had evaluated, and took the search back to A = "
          "0.89349363928412684 (the point of eval 98), where it had stood "
          "with the same loss and gradient");
}

/* Pins: run_dlib_bfgs_impl on BoundBowl from (0.40137104310117261, 0)
 * under PARAM tol_gradinf 1e-12 (MAXEVAL 400, a reusing host): A reaches
 * its lower bound and stays (its gradient, 28, points out of the box, so
 * the gradient norm never falls below tol_gradinf), B goes to its minimum,
 * and the search ends after 96 evaluations, none repeated, "DLIB:BFGS: back
 * at an earlier point", the stop line naming the point it came back to
 * ("... took the search back to A = 0, B = 0.233112594... (the point of
 * eval 47) ...").
 * Before the fix: after the same 96 evaluations BFGS went round points of
 * B's plateau without an evaluation and never returned (a minute ended the
 * case; 17 million calls in 4 s answered from the record). */
TEST_CASE("Repeated points: DLIB:BFGS held at a bound under PARAM "
          "tol_gradinf ends back at an earlier point",
          "[HPO][repeat_points][dlib_bfgs][revisit]")
{
    BoundBowl host;
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.40137104310117261, 0.0});
    sp.stop.maxeval = 400;
    sp.param["tol_gradinf"] = 1e-12;
    const std::string journal = run_bfgs_within(
        sp, host, "DLIB:BFGS held at a bound under tol_gradinf");
    CHECK(host.asked.size() == 96u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(status_of(journal) == "DLIB:BFGS: back at an earlier point");
    const std::string line = bfgs_stop_line(journal);
    CHECK_THAT(line,
               Catch::StartsWith("DLIB:BFGS stopped: back at an earlier "
                                 "point after 96 evaluation(s): the "
                                 "iteration from A = 0, B = "));
    CHECK_THAT(line, Catch::Contains(", and took the search back to A = 0, "
                                     "B = 0.233112594") &&
                         Catch::EndsWith("(the point of eval 47), where it "
                                         "had stood with the same loss and "
                                         "gradient"));
}

namespace {

/* 1 + (a + 0.3)^2 + (b + 0.1)^2 on [0, 1]^2: the minimum lies beyond the
 * lower corner (0, 0), where LD_MMA ends up, both coordinates at u = 0. */
struct LowerCornerBowl {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double a = x(0) + 0.3, b = x(1) + 0.1;
        return 1.0 + a * a + b * b;
    }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return asked.size(); }
};

}  // namespace

/* Pins: an NLopt search whose calls are answered from the record is
 * bounded by NLopt's own MAXEVAL, which counts its calls: LD_MMA (MAXEVAL
 * 60, XTOL_REL 1e-8) on the bowl whose minimum lies beyond the lower corner
 * reaches the corner (0, 0), where XTOL_REL's test, relative to |x| = 0,
 * never holds, and asks for the corner and its gradient again and again;
 * every such call is answered from the record, and the search ends after 60
 * calls, MAXEVAL_REACHED ("NLopt finished: MAXEVAL_REACHED -- maximum
 * evaluations exhausted after 20 of MAXEVAL 60 evaluation(s); NLopt counts
 * its calls, and 56 of them asked for points the run had evaluated ..."): 4
 * calls evaluated (each the loss and a 4-point stencil) and 56 answered,
 * none of the 20 evaluations repeating another.
 * Before the fix: passes as well (a pin: NLopt's count of calls bounds a
 * search that only asks again; DLIB:BFGS had no such count). Before the
 * line named a call answered from an earlier search's evaluation as such,
 * it said "56 of them asked again for a point": the regex fails there. */
TEST_CASE("Repeated points: NLopt's own MAXEVAL ends a search whose calls "
          "are answered from the record",
          "[HPO][repeat_points][nlopt][maxeval][revisit]")
{
    LowerCornerBowl host;
    const std::string journal = run_nlopt(nlopt_spec("LD_MMA", 60), host);
    CHECK(status_of(journal) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    CHECK(repeats_in(host.asked) == 0u);
    std::smatch m;
    REQUIRE(std::regex_search(
        journal, m,
        std::regex(R"(after (\d+) of MAXEVAL 60 evaluation\(s\); NLopt counts )"
                   R"(its calls, and (\d+) of them asked for points the run )"
                   R"(had evaluated)")));
    CHECK(std::stoul(m[1]) == host.asked.size());
    CHECK(host.asked.size() == 20u);
    CHECK(std::stoul(m[2]) == 56u);
}

/* Pins: a gradient that takes a recorded loss as the centre of its finite
 * differences (RepeatUse::Centre) evaluates the stencil, so it is counted
 * in MAXEVAL and refused once MAXEVAL is spent: BfgsObjective under
 * NUMERICAL with MAXEVAL 2 and a reusing host, the losses of P and Q (2
 * evaluations), then P's gradient: MaxEvalReached, and the host is asked for
 * nothing more.
 * Before the fix: passes as well (a pin: a Centre call is an evaluation,
 * so no stretch of calls without one includes it). */
TEST_CASE("Repeated points: a gradient centred on a recorded loss counts in "
          "MAXEVAL", "[HPO][repeat_points][dlib_bfgs][maxeval][revisit]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    LastEvaluationGradient host;
    Journal j;
    tadah::hpo::detail::BfgsObjective<Vec, NoRunner, LastEvaluationGradient>
        objective(sp, host, 2);
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
    CHECK(objective.loss(P) == 0.5);
    CHECK(objective.loss(Q) == 0.625);
    CHECK_THROWS_AS(objective.gradient(P),
                    tadah::hpo::detail::MaxEvalReached);
    CHECK(host.asked.size() == 2u);
    CHECK(objective.evaluations() == 2u);
}

/* Pins: dlib asks for the gradient of the point its line search ended on
 * right after its loss; when that loss was answered from the record, the
 * gradient takes it as an evaluation's (NUMERICAL: the centre of the finite
 * differences) and is no second reuse. BfgsObjective under NUMERICAL with a
 * reusing host: the losses of P and Q, P's loss again (answered from the
 * record), then P's gradient: the host is asked for P, Q and P's 4-point
 * stencil, and DIAGNOSTICS counts one reuse ("Repeated points reused : 1"),
 * as the evaluation of P that the call stands for is one.
 * Before the fix: the gradient call was counted as a second reuse
 * ("Repeated points reused : 2"), and in the RESTART window as a second
 * evaluation without a new best. */
TEST_CASE("Repeated points: a gradient right after a reused loss is no "
          "second reuse", "[HPO][repeat_points][dlib_bfgs]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    LastEvaluationGradient host;
    Journal j;
    tadah::hpo::detail::BfgsObjective<Vec, NoRunner, LastEvaluationGradient>
        objective(sp, host, 0);
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
    CHECK(objective.loss(P) == 0.5);
    CHECK(objective.loss(Q) == 0.625);
    CHECK(objective.loss(P) == 0.5);
    const Vec g = objective.gradient(P);
    CHECK(host.asked.size() == 6u);
    CHECK(repeats_in(host.asked) == 0u);
    CHECK(std::abs(g(0) - 1.0) < 1e-6);
    CHECK(std::abs(g(1) - 1.0) < 1e-6);
    CHECK(diagnostics_count(j.text(), "Repeated points reused") == "1");
}

namespace {

/* LastEvaluationGradient with a RESTART window: it counts the calls the
 * search answers from the record (HPO_Host::count_reused_repeat). */
struct WindowedLastEvaluationGradient : LastEvaluationGradient {
    mutable std::size_t window_calls = 0;
    void count_reused_repeat(const tadah::hpo::RecordedEval<Vec>&) const
    {
        ++window_calls;
    }
};

using WindowedObjective =
    tadah::hpo::detail::BfgsObjective<Vec, NoRunner,
                                      WindowedLastEvaluationGradient>;

}  // namespace

/* Pins: under GRAD_STRATEGY ANALYTICAL and HYBRID a gradient at a point
 * the search formed none at evaluates the point (HPO_Host's analytical
 * gradient is the last evaluation's), so a loss there answered from the
 * record right before it saved no evaluation: the two calls are one
 * evaluation, as dlib asks for them at its start and at the point each line
 * search ends on. BfgsObjective with a host that has a RESTART window: the
 * losses of P and Q, P's loss again (answered from the record, the host
 * not asked), then P's gradient: P is evaluated (3 evaluations, the third
 * P) and its gradient is (1, 1); no call is counted in the window or in
 * DIAGNOSTICS as a reuse ("Repeated points reused" absent, no "is reused"
 * line), and the evaluation is a repeat evaluated again ("Repeats evaluated
 * again : 1", "DLIB:BFGS asked again for the point of eval 1 (A = 0.5, B =
 * 0.5) and its gradient; it is evaluated again: a gradient was asked for
 * where this search formed none, and GRAD_STRATEGY ANALYTICAL [HYBRID]
 * differentiates the last evaluation").
 * Before the fix: the loss call was counted as a reuse as well, once in
 * the window and in DIAGNOSTICS ("Repeated points reused : 1", "the loss of
 * eval 1 is reused, without an evaluation"), and the evaluation of P once
 * more: a PRESTAGE refinement's start counted twice in the RESTART window,
 * which ended the refinement at its first evaluation under AFTER_STAGNATION
 * 2. */
TEST_CASE("Repeated points: under ANALYTICAL and HYBRID a reused loss and "
          "the evaluation its gradient makes are one call",
          "[HPO][repeat_points][dlib_bfgs][analytical]")
{
    for (const GradStrategy strategy :
         {GradStrategy::Analytical, GradStrategy::Hybrid}) {
        const std::string name =
            strategy == GradStrategy::Analytical ? "ANALYTICAL" : "HYBRID";
        DYNAMIC_SECTION(name)
        {
            auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
            sp.grad_strategy = strategy;
            WindowedLastEvaluationGradient host;
            Journal j;
            WindowedObjective objective(sp, host, 0);
            const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
            CHECK(objective.loss(P) == 0.5);
            CHECK(objective.loss(Q) == 0.625);
            CHECK(objective.loss(P) == 0.5);
            CHECK(host.asked.size() == 2u);
            const Vec g = objective.gradient(P);
            CHECK(host.asked.size() == 3u);
            CHECK(same_bits(host.asked.back(), P));
            CHECK(host.off_point == 0u);
            CHECK(g(0) == 1.0);
            CHECK(g(1) == 1.0);
            CHECK(host.window_calls == 0u);
            const std::string journal = j.text();
            INFO("journal:\n" << journal);
            CHECK(diagnostics_count(journal, "Repeated points reused").empty());
            CHECK(diagnostics_count(journal, "Repeats evaluated again") == "1");
            CHECK(journal.find("is reused") == std::string::npos);
            CHECK(journal.find("DLIB:BFGS asked again for the point of eval 1 "
                               "(A = 0.5, B = 0.5) and its gradient; it is "
                               "evaluated again: a gradient was asked for "
                               "where this search formed none, and "
                               "GRAD_STRATEGY " + name +
                               " differentiates the last evaluation") !=
                  std::string::npos);
        }
    }
}

/* Pins: under ANALYTICAL a loss answered from the record that no gradient
 * evaluation answers is a reuse, counted once in the window and in
 * DIAGNOSTICS ("the loss of eval 1 is reused, without an evaluation"):
 *   - followed by the loss of another point R (evaluated): window 1,
 *     "Repeated points reused : 1" (a pin: passes before the fix as well);
 *   - followed by P's gradient once MAXEVAL (2) is spent: the gradient
 *     throws MaxEvalReached without an evaluation, the host asked for P and
 *     Q only, window 1, "Repeated points reused : 1", and no repeat
 *     evaluated again ("Repeats evaluated again" absent).
 * Before the fix: the second leg counted and journalled P's gradient call
 * as a repeat evaluated again ("Repeats evaluated again : 1", "it is
 * evaluated again: ...") although MAXEVAL refused its evaluation. */
TEST_CASE("Repeated points: under ANALYTICAL a reused loss that no gradient "
          "evaluation answers is a reuse",
          "[HPO][repeat_points][dlib_bfgs][analytical]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.grad_strategy = GradStrategy::Analytical;
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75), R = point(0.5, 0.25);
    SECTION("followed by another point's loss")
    {
        WindowedLastEvaluationGradient host;
        Journal j;
        WindowedObjective objective(sp, host, 0);
        CHECK(objective.loss(P) == 0.5);
        CHECK(objective.loss(Q) == 0.625);
        CHECK(objective.loss(P) == 0.5);
        CHECK(objective.loss(R) == 0.3125);
        CHECK(host.asked.size() == 3u);
        CHECK(host.window_calls == 1u);
        const std::string journal = j.text();
        INFO("journal:\n" << journal);
        CHECK(diagnostics_count(journal, "Repeated points reused") == "1");
        CHECK(journal.find("the loss of eval 1 is reused, without an "
                           "evaluation") != std::string::npos);
    }
    SECTION("followed by its gradient once MAXEVAL is spent")
    {
        WindowedLastEvaluationGradient host;
        Journal j;
        WindowedObjective objective(sp, host, 2);
        CHECK(objective.loss(P) == 0.5);
        CHECK(objective.loss(Q) == 0.625);
        CHECK(objective.loss(P) == 0.5);
        CHECK_THROWS_AS(objective.gradient(P),
                        tadah::hpo::detail::MaxEvalReached);
        CHECK(host.asked.size() == 2u);
        CHECK(host.window_calls == 1u);
        const std::string journal = j.text();
        INFO("journal:\n" << journal);
        CHECK(diagnostics_count(journal, "Repeated points reused") == "1");
        CHECK(diagnostics_count(journal, "Repeats evaluated again").empty());
        CHECK(journal.find("it is evaluated again") == std::string::npos);
    }
}

/* Pins: a loss answered from the record that the search's last call asked
 * for (no call followed it) is counted when the search ends
 * (BfgsObjective::settle, which run_dlib_bfgs_impl calls when the loop
 * returns): under ANALYTICAL, P, Q, then P's loss again answered from the
 * record: before settle() the window has counted nothing, after it 1, and
 * DIAGNOSTICS "Repeated points reused : 1"; a second settle() counts
 * nothing more.
 * Before the fix: does not compile (BfgsObjective::settle is new); the
 * loss call was counted when it was answered. */
TEST_CASE("Repeated points: a reused loss the search ended on is counted "
          "when it ends", "[HPO][repeat_points][dlib_bfgs][analytical]")
{
    auto sp = bfgs_spec({0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5});
    sp.grad_strategy = GradStrategy::Analytical;
    const Vec P = point(0.5, 0.5), Q = point(0.25, 0.75);
    WindowedLastEvaluationGradient host;
    Journal j;
    WindowedObjective objective(sp, host, 0);
    CHECK(objective.loss(P) == 0.5);
    CHECK(objective.loss(Q) == 0.625);
    CHECK(objective.loss(P) == 0.5);
    CHECK(host.window_calls == 0u);
    objective.settle();
    CHECK(host.window_calls == 1u);
    objective.settle();
    CHECK(host.window_calls == 1u);
    CHECK(host.asked.size() == 2u);
    CHECK(diagnostics_count(j.text(), "Repeated points reused") == "1");
}

/* Pins: an iteration of DLIB:BFGS whose line search is not run asks for no
 * evaluation, so it checks the stop flag as an evaluation does: with a
 * SIGTERM raised before the search, on the Slope host (which does not
 * check it) started on its upper bound, the start and its stencil are
 * evaluated (3) and the first iteration, which cannot move the point,
 * throws core::ShutdownRequested. A hidden case ([.shutdown_bfgs_not_run])
 * with a CTest entry -- a process -- of its own: the flag is never
 * lowered.
 * Before the fix: the search ended "DLIB:BFGS: FTOL_ABS reached" after 3
 * evaluations without a word of the signal. */
TEST_CASE("Repeated points: an iteration of DLIB:BFGS that asks for no "
          "evaluation checks the stop flag", "[.shutdown_bfgs_not_run]")
{
    tadah::core::SignalHandler::install();
    std::raise(SIGTERM);
    REQUIRE(tadah::core::SignalHandler::stop_requested());
    Slope host;
    auto sp = bfgs_spec({0.0}, {1.0}, {1.0});
    sp.stop.maxeval = 30;
    Journal j;
    CHECK_THROWS_AS((tadah::hpo::run_dlib_bfgs_impl<Vec, NoRunner>(sp, host)),
                    tadah::core::ShutdownRequested);
    CHECK(host.asked.size() == 3u);
}

/* ======================================================================
 * When every evaluation is a function of its point
 * ==================================================================== */

namespace {

tadah::core::Context targets_context()
{
    tadah::core::Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "false");
    c.add("INITMB", "false");
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    c.add("VERBOSE", 0);
    return c;
}

/* The blocker of an HPOTargets read from `body` (a scratch file, unique
 * across processes: ctest runs cases concurrently). */
std::string blocker_of(const std::string& body)
{
    static int counter = 0;
    const fs::path p = fs::temp_directory_path() /
                       ("test_repeat_points_" + std::to_string(::getpid()) +
                        "_" + std::to_string(++counter) + ".hpo");
    std::ofstream(p) << body;
    tadah::core::PeriodicTable::initialize();
    tadah::core::Context ctx = targets_context();
    tadah::hpo::HPOTargets trg(ctx, p.string());
    fs::remove(p);
    return trg.repeat_reuse_blocker();
}

const std::string kHead =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nENDOPTIMIZER\nLOSS L2\n";

}  // namespace

/* Pins: HPOTargets::repeat_reuse_blocker() is "" for a loss of fit metrics
 * (PC_ERMSE, with an absolute --skip-above cap too), and names why an
 * evaluation can depend on earlier ones otherwise: a PC_LAMMPS line ("its
 * scripts run on LAMMPS instances kept from one evaluation to the next",
 * and with a --hotvar "each --hotvar starts from the value of the best
 * evaluation so far"), a PC_OBSERVABLE whose observable keeps state
 * (cohesive_energy: "its relaxation starts from the lattice constant the
 * previous evaluation relaxed to") and a --skip-rel filter ("compares with
 * the best value so far").
 * Before the fix: does not compile (repeat_reuse_blocker is new). */
TEST_CASE("Repeated points: the run says when an evaluation is a function "
          "of its point", "[HPO][repeat_points][hpotargets]")
{
    CHECK(blocker_of(kHead + "PC_ERMSE 0 1\n") == "");
    CHECK(blocker_of(kHead + "PC_ERMSE 0 1 --skip-above 50\n") == "");

    const std::string rel = blocker_of(kHead + "PC_ERMSE 0 1 --skip-rel 5\n");
    CHECK(rel.find("PC_ERMSE --skip-rel") != std::string::npos);
    CHECK(rel.find("compares with the best value so far") != std::string::npos);

    const std::string obs = blocker_of(
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\nLOSS L2\n"
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.100 --weight "
        "1000 --hotvar a0 2.7 3.9 3.3036\n");
    CHECK(obs.find("PC_OBSERVABLE cohesive_energy") != std::string::npos);
    CHECK(obs.find("its relaxation starts from the lattice constant the "
                   "previous evaluation relaxed to") != std::string::npos);

    const fs::path script = fs::temp_directory_path() /
                            ("test_repeat_points_" +
                             std::to_string(::getpid()) + ".in");
    std::ofstream(script) << "variable xyz equal 0.0\n";
    const std::string lmp = blocker_of(
        kHead + "PC_LAMMPS --script " + script.string() +
        " --varloss xyz 1.0 100\n");
    CHECK(lmp.find("PC_LAMMPS: its scripts run on LAMMPS instances kept from "
                   "one evaluation to the next") != std::string::npos);
    CHECK(lmp.find("--hotvar") == std::string::npos);
    const std::string hot = blocker_of(
        kHead + "PC_LAMMPS --script " + script.string() +
        " --hotvar xyz 0 1 0.5 --varloss xyz 1.0 100\n");
    CHECK(hot.find("each --hotvar starts from the value of the best "
                   "evaluation so far") != std::string::npos);
    fs::remove(script);
}

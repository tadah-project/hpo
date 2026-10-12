/* NLopt's MAXEVAL counts every evaluation of the run, as DLIB:BFGS and the
 * CERES drivers count (nlopt_adapter.h, run_nlopt_impl).
 *
 * NLopt counts objective calls, and a call with a finite-difference
 * gradient makes the loss and the gradient's evaluations (CENTRAL: 2n in n
 * dimensions, after the centre). The GradientProvider kept a MAXEVAL of
 * its own for the gradient's evaluations, so once it was spent every later
 * gradient failed (GradientFailure::Budget), the objective returned +inf for
 * points it had evaluated, and the search ran on to NLopt's own count
 * (LD_MMA: 120 evaluations for MAXEVAL 60) or ended as if it had converged
 * (LD_LBFGS: "STOPPED -- nlopt failure (algorithm converged ...)").
 * GN_CRS2_LM, GN_ORIG_DIRECT(_L) and AUGLAG passed NLopt's own MAXEVAL too.
 *
 * A gradient takes the loss its caller has just evaluated at the point as
 * its centre (GradientProvider::gradient_at, gradient_provider.h): the
 * NLopt objective, Ceres' FirstOrderFunction and DLIB:BFGS evaluate the
 * loss and then ask for the gradient there, and NUMERICAL evaluated the
 * point once more.
 *
 * The stub hosts record every REAL point asked for. */

#include "catch2/catch.hpp"

#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>

#ifdef TADAH_USE_CERES
#include <tadah/hpo/ceres_adapter.h>
#endif

#include <dlib/matrix.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using tadah::hpo::GradStrategy;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;
namespace fs = std::filesystem;

struct NoRunner {};   // GradientProvider's Runner is only a type tag

const std::string kMaxevalReached =
    "MAXEVAL_REACHED -- maximum evaluations exhausted";
const std::string kStopvalReached =
    "STOPVAL_REACHED -- objective below stopval threshold";

/* The journal line of a search that MAXEVAL ended. */
std::string maxeval_line(unsigned maxeval)
{
    return "NLopt finished: " + kMaxevalReached + " after " +
           std::to_string(maxeval) + " of MAXEVAL " + std::to_string(maxeval) +
           " evaluation(s)";
}

double valley(const Vec& x)
{
    const double a = 1.0 - x(0), b = x(1) - x(0) * x(0);
    return a * a + 100.0 * b * b;
}

/* Rosenbrock's valley in [-2, 2]^2 (real = internal coordinates), from
 * (-1.2, 1), where the loss is 24.2: no search reaches its minimum in a few
 * dozen evaluations, so MAXEVAL ends every search here. The analytical
 * gradient is the exact one, made not finite (NaN) at analytical call
 * number `nan_at` (0: never). */
struct Valley {
    std::vector<Vec> asked;
    std::size_t analytical = 0;   // analytical gradients formed
    std::size_t nan_at = 0;
    double operator()(const Vec& x) {
        asked.push_back(x);
        return valley(x);
    }
    bool analytical_gradient(const Vec& x, Vec& g, std::vector<int>& nfd) {
        ++analytical;
        g.set_size(2);
        const double b = x(1) - x(0) * x(0);
        g(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * b;
        g(1) = 200.0 * b;
        if (analytical == nan_at)
            g(1) = std::numeric_limits<double>::quiet_NaN();
        nfd.clear();
        return true;
    }
    /* Points asked for twice in a row, bit for bit. */
    std::size_t repeats() const {
        std::size_t n = 0;
        for (std::size_t k = 1; k < asked.size(); ++k)
            if (std::memcmp(&asked[k](0), &asked[k - 1](0), sizeof(double)) == 0 &&
                std::memcmp(&asked[k](1), &asked[k - 1](1), sizeof(double)) == 0)
                ++n;
        return n;
    }
};

OptimizerSpec nlopt_spec(const std::string& algo, unsigned maxeval)
{
    OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = algo;
    sp.low = {-2.0, -2.0};
    sp.high = {2.0, 2.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    sp.analyse_logscale();
    sp.optim_init = {{-1.2, 1.0}};
    sp.stop.maxeval = maxeval;
    sp.seed = 7;
    sp.param["fd_step"] = 1e-6;
    sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = std::numeric_limits<double>::max();
    return sp;
}

/* An INNER block running `algo` (XTOL_REL 1e-4: a stop of its own). */
void set_inner(OptimizerSpec& sp, const std::string& algo)
{
    OptimizerSpec in;
    in.lib = "NLOPT";
    in.algo = algo;
    in.stop.xtol_rel = 1e-4;
    sp.inner = std::make_unique<OptimizerSpec>(in);
}

/* An open journal; `text()` closes it and returns it. */
struct Journal {
    fs::path path = fs::temp_directory_path() /
                    ("tadah_nlopt_maxeval_" +
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
    std::istringstream in(journal);
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find("Optimiser status");
        if (p == std::string::npos) continue;
        const auto c = line.find(": ", p);
        return c == std::string::npos ? line : line.substr(c + 2);
    }
    return "(no status)";
}

}  // namespace

/* Pins: each NLopt gradient search on the valley, NUMERICAL (CENTRAL), at
 * MAXEVAL 60 -- every local LD_* family, GD_STOGO, LD_AUGLAG (its default
 * local search LD_MMA), AUGLAG with INNER LD_LBFGS and G_MLSL_LDS
 * (POPULATION 4) with INNER LD_LBFGS -- asks the host for exactly 60 points,
 * ends MAXEVAL_REACHED, and the journal says "NLopt finished:
 * MAXEVAL_REACHED -- maximum evaluations exhausted after 60 of MAXEVAL 60
 * evaluation(s)". GD_STOGO ignores NLopt's forced stop: NLopt's own MAXEVAL
 * ends it, and the objective evaluates none of its calls after the budget.
 * Before the fix: LD_LBFGS made 94 evaluations, LD_TNEWTON 99,
 * LD_TNEWTON_PRECOND_RESTART and LD_VAR2 94, AUGLAG with INNER LD_LBFGS 94
 * and G_MLSL_LDS 99, each ending "STOPPED -- nlopt failure (algorithm
 * converged ...)"; LD_SLSQP, LD_MMA and LD_CCSAQ 120 and LD_AUGLAG 121,
 * ending MAXEVAL_REACHED; GD_STOGO 120, ending "SUCCESS -- generic
 * success". No journal line said that MAXEVAL ended a search. */
TEST_CASE("NLopt MAXEVAL: a gradient search makes exactly MAXEVAL "
          "evaluations, the loss and each finite-difference point counted, "
          "and ends MAXEVAL_REACHED", "[HPO][nlopt][maxeval][fd_gradient]")
{
    struct Case { std::string algo, inner, label; };
    const std::vector<Case> cases = {
        {"LD_LBFGS", "", "LD_LBFGS"},
        {"LD_TNEWTON", "", "LD_TNEWTON"},
        {"LD_TNEWTON_PRECOND_RESTART", "", "LD_TNEWTON_PRECOND_RESTART"},
        {"LD_VAR2", "", "LD_VAR2"},
        {"LD_SLSQP", "", "LD_SLSQP"},
        {"LD_MMA", "", "LD_MMA"},
        {"LD_CCSAQ", "", "LD_CCSAQ"},
        {"GD_STOGO", "", "GD_STOGO"},
        {"LD_AUGLAG", "", "LD_AUGLAG (NLopt's default LD_MMA)"},
        {"AUGLAG", "LD_LBFGS", "AUGLAG, INNER LD_LBFGS"},
        {"G_MLSL_LDS", "LD_LBFGS", "G_MLSL_LDS, INNER LD_LBFGS"}};
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.label) {
            Journal j;
            auto sp = nlopt_spec(c.algo, 60);
            if (!c.inner.empty()) set_inner(sp, c.inner);
            if (c.algo == "G_MLSL_LDS") sp.pop = 4;
            Valley host;
            REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
            CHECK(host.asked.size() == 60u);
            const std::string journal = j.text();
            CHECK(status_of(journal) == kMaxevalReached);
            CHECK(journal.find(maxeval_line(60)) != std::string::npos);
        }
    }
}

/* Pins: LD_LBFGS and LD_MMA on the valley at MAXEVAL 1, 2, 5, 7 and 13 --
 * a budget that ends at the start's loss, within its gradient (the loss
 * and 4 central-difference points in 2 dimensions), at its end, and within
 * a later gradient -- ask the host for exactly MAXEVAL points and end
 * MAXEVAL_REACHED.
 * Before the fix: LD_LBFGS made 2, 3, 17, 19 and 26 evaluations,
 * ending "SUCCESS -- generic success" at MAXEVAL 1 and 2 and "STOPPED --
 * nlopt failure (algorithm converged ...)" above; LD_MMA 2, 4, 10, 14 and
 * 26, ending MAXEVAL_REACHED. */
TEST_CASE("NLopt MAXEVAL: the budget holds at any MAXEVAL, within a "
          "gradient too", "[HPO][nlopt][maxeval][fd_gradient]")
{
    for (const std::string algo : {"LD_LBFGS", "LD_MMA"}) {
        for (unsigned maxeval : {1u, 2u, 5u, 7u, 13u}) {
            DYNAMIC_SECTION(algo << " MAXEVAL " << maxeval) {
                Journal j;
                Valley host;
                REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(
                    nlopt_spec(algo, maxeval), host));
                CHECK(host.asked.size() == maxeval);
                CHECK(status_of(j.text()) == kMaxevalReached);
            }
        }
    }
}

/* Pins: on the valley at MAXEVAL 60, GN_CRS2_LM (SEED 7), GN_ORIG_DIRECT,
 * GN_ORIG_DIRECT_L and AUGLAG with INNER LN_BOBYQA ask the host for exactly
 * 60 points; LN_BOBYQA and LN_SBPLX, which NLopt's own count ends, do too.
 * Each ends MAXEVAL_REACHED with the journal line "NLopt finished:
 * MAXEVAL_REACHED -- maximum evaluations exhausted after 60 of MAXEVAL 60
 * evaluation(s)".
 * Before the fix: GN_CRS2_LM made 61 evaluations (it tests MAXEVAL
 * only after a trial that does not beat its worst point), GN_ORIG_DIRECT
 * 75 and GN_ORIG_DIRECT_L 71 (once per iteration), AUGLAG 61 (it evaluates
 * the point its local search returned once more), all ending
 * MAXEVAL_REACHED; every search, LN_BOBYQA and LN_SBPLX with their 60
 * included, wrote "NLopt finished: MAXEVAL_REACHED -- maximum evaluations
 * exhausted  fmin=<best>" instead of the line. */
TEST_CASE("NLopt MAXEVAL: a derivative-free search, and AUGLAG, make no "
          "evaluation beyond it", "[HPO][nlopt][maxeval]")
{
    struct Case { std::string algo, inner; };
    for (const Case& c : {Case{"GN_CRS2_LM", ""}, Case{"GN_ORIG_DIRECT", ""},
                          Case{"GN_ORIG_DIRECT_L", ""},
                          Case{"AUGLAG", "LN_BOBYQA"}, Case{"LN_BOBYQA", ""},
                          Case{"LN_SBPLX", ""}}) {
        DYNAMIC_SECTION(c.algo << (c.inner.empty() ? "" : ", INNER " + c.inner)) {
            Journal j;
            auto sp = nlopt_spec(c.algo, 60);
            if (!c.inner.empty()) set_inner(sp, c.inner);
            Valley host;
            REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
            CHECK(host.asked.size() == 60u);
            const std::string journal = j.text();
            CHECK(status_of(journal) == kMaxevalReached);
            CHECK(journal.find(maxeval_line(60)) != std::string::npos);
        }
    }
}

/* Pins: LD_LBFGS under GRAD_STRATEGY ANALYTICAL on the valley at MAXEVAL
 * 30 makes 15 losses and 15 analytical gradients, 30 evaluations, and
 * ends MAXEVAL_REACHED: each analytical gradient counts as one evaluation,
 * as it does under DLIB:BFGS and the CERES drivers (owner decision,
 * 2026-10-07).
 * Before the fix: 28 losses and 28 analytical gradients, 56
 * evaluations, ending "SUCCESS -- generic success": NLopt counted its 28
 * calls. */
TEST_CASE("NLopt MAXEVAL: an analytical gradient counts as one evaluation, "
          "as under DLIB:BFGS and CERES", "[HPO][nlopt][maxeval][analytical]")
{
    Journal j;
    auto sp = nlopt_spec("LD_LBFGS", 30);
    sp.grad_strategy = GradStrategy::Analytical;
    sp.param.clear();
    sp.param_raw.clear();
    Valley host;
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    CHECK(host.asked.size() == 15u);
    CHECK(host.analytical == 15u);
    CHECK(status_of(j.text()) == kMaxevalReached);
}

/* Pins: a loss at or below STOPVAL that the objective evaluated but did not
 * give NLopt -- NLopt tests STOPVAL first -- ends the search
 * STOPVAL_REACHED, with no evaluation after it (owner decision,
 * 2026-10-07):
 *   - the budget: LD_LBFGS at MAXEVAL 1 and STOPVAL 100 evaluates the start
 *     (24.2), whose gradient MAXEVAL cuts short;
 *   - a failed stencil evaluation: LD_LBFGS (NUMERICAL) at STOPVAL 100
 *     from the start, whose stencil point (-1.2, 1 - 1e-6) fails (the
 *     default FAILSCORE): the journal line names the cause;
 *   - a gradient that is not finite at a later point: LD_LBFGS under
 *     ANALYTICAL at STOPVAL 1, whose fourth point (loss 0.19) is the first
 *     at or below it and gets a NaN analytical gradient: 4 losses and 4
 *     analytical gradients.
 * A failed loss is no STOPVAL success (anneal::stopval_reached): at
 * STOPVAL = the largest double, the start scoring FAILSCORE still ends
 * "STOPPED -- no usable gradient", after one evaluation: its failed loss
 * fails its gradient before the stencil is evaluated (since 2026-10-07; it
 * evaluated the stencil, the failed (-1.2 + 1.2e-6, 1) and the usable
 * (-1.2 - 1.2e-6, 1), and the difference overflowed: 3 evaluations, and
 * the status said "gradient not finite"). The journal line of a withheld
 * loss says "its point has no usable gradient: <cause>" (it said "its
 * gradient is not finite: <cause>").
 * Before the fix: the budget: 2 evaluations (the start again, as the
 * gradient's centre), ending "SUCCESS -- generic success"; the failed
 * stencil evaluation: "STOPPED -- gradient not finite: ..." (HPO's stop at
 * a starting point without a finite gradient); the later point: NLopt was
 * given +inf there and went on, to its own STOPVAL_REACHED at a later
 * point, after 6 losses and 6 analytical gradients. No journal line said
 * that a loss NLopt was not given met STOPVAL. */
TEST_CASE("NLopt STOPVAL: a loss at or below it that NLopt was not given "
          "ends the search STOPVAL_REACHED", "[HPO][nlopt][maxeval][stopval]")
{
    const std::string stopval_line = "NLopt finished: " + kStopvalReached;
    SECTION("MAXEVAL cut its gradient short") {
        Journal j;
        auto sp = nlopt_spec("LD_LBFGS", 1);
        sp.stop.stopval = 100.0;
        Valley host;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        CHECK(host.asked.size() == 1u);
        const std::string journal = j.text();
        CHECK(status_of(journal) == kStopvalReached);
        CHECK(journal.find(stopval_line + " after 1 evaluation(s): the loss "
                           "24.2 is at or below STOPVAL 100, and NLopt was "
                           "not given it: its gradient could not be "
                           "completed within MAXEVAL") != std::string::npos);
    }
    SECTION("a failed stencil evaluation at the start") {
        struct Cliff : Valley {
            double operator()(const Vec& x) {
                asked.push_back(x);
                return x(1) < 1.0 - 1e-7 ? std::numeric_limits<double>::max()
                                         : valley(x);
            }
        };
        Journal j;
        auto sp = nlopt_spec("LD_LBFGS", 60);
        sp.stop.stopval = 100.0;
        Cliff host;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        REQUIRE_FALSE(host.asked.empty());
        /* the last evaluation is the stencil point that failed */
        CHECK(host.asked.back()(0) == -1.2);
        CHECK(host.asked.back()(1) < 1.0);
        const std::string journal = j.text();
        CHECK(status_of(journal) == kStopvalReached);
        CHECK(journal.find(stopval_line) != std::string::npos);
        CHECK(journal.find("the loss 24.2 is at or below STOPVAL 100, and "
                           "NLopt was not given it: its point has no usable "
                           "gradient: a failed evaluation in the "
                           "finite-difference stencil") != std::string::npos);
        CHECK(journal.find("stopped: no usable gradient") == std::string::npos);
    }
    SECTION("an analytical gradient not finite at a later point") {
        Journal j;
        auto sp = nlopt_spec("LD_LBFGS", 60);
        sp.grad_strategy = GradStrategy::Analytical;
        sp.param.clear();
        sp.param_raw.clear();
        sp.stop.stopval = 1.0;
        Valley host;
        host.nan_at = 4;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        CHECK(host.asked.size() == 4u);
        CHECK(host.analytical == 4u);
        const std::string journal = j.text();
        CHECK(status_of(journal) == kStopvalReached);
        CHECK(journal.find("NLopt was not given it: its point has no usable "
                           "gradient: the analytical gradient is not finite "
                           "at this point") != std::string::npos);
    }
    SECTION("a failed loss is no STOPVAL success") {
        struct FailRight : Valley {
            double operator()(const Vec& x) {
                asked.push_back(x);
                return x(0) >= -1.2 ? std::numeric_limits<double>::max()
                                    : valley(x);
            }
        };
        Journal j;
        auto sp = nlopt_spec("LD_LBFGS", 60);
        sp.stop.stopval = std::numeric_limits<double>::max();
        FailRight host;
        REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
        CHECK(host.asked.size() == 1u);
        const std::string journal = j.text();
        CHECK(status_of(journal) ==
              "STOPPED -- no usable gradient: a failed evaluation in the "
              "finite-difference stencil");
        CHECK(journal.find(stopval_line) == std::string::npos);
    }
}

/* Pins: a host exception raised by the evaluation that spends MAXEVAL (the
 * fifth of LD_LBFGS at MAXEVAL 5, a finite-difference point of the start's
 * gradient) crosses run_nlopt_impl with its own type: the objective's
 * exception comes before the budget's ending.
 * Before the fix: passes as well (a pin of the order: the exception was
 * rethrown the same way). */
TEST_CASE("NLopt MAXEVAL: a host exception at the evaluation that spends it "
          "keeps its own type", "[HPO][nlopt][maxeval][failure_handling]")
{
    struct HostBroke : std::runtime_error {
        HostBroke() : std::runtime_error("host broke") {}
    };
    struct Throws : Valley {
        double operator()(const Vec& x) {
            asked.push_back(x);
            if (asked.size() == 5) throw HostBroke();
            return valley(x);
        }
    };
    Journal j;
    Throws host;
    CHECK_THROWS_AS((tadah::hpo::run_nlopt_impl<Vec, NoRunner>(
                        nlopt_spec("LD_LBFGS", 5), host)),
                    HostBroke);
    CHECK(host.asked.size() == 5u);
}

/* Pins: the masked-diagnostic warning of LN_SBPLX (its first simplex too
 * small next to x = 1e13: NLopt reports XTOL_REACHED) gives the
 * evaluations made, "after 1 evaluation(s)"; for a derivative-free search
 * NLopt's count of its calls is the same number.
 * Before the fix: the warning read "after 1 eval(s)", NLopt's count of
 * objective calls, which is not the evaluations of a gradient search. */
TEST_CASE("NLopt MAXEVAL: the masked-diagnostic warning counts evaluations",
          "[HPO][nlopt][maxeval]")
{
    OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "LN_SBPLX";
    sp.low = {1e13, -2.0};
    sp.high = {1e13 + 4.0, 2.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    sp.analyse_logscale();
    sp.optim_init = {{1e13 + 1.0, 1.0}};
    sp.stop.maxeval = 60;
    sp.stop.xtol_rel = 1e-4;
    struct Bowl {
        std::size_t calls = 0;
        double operator()(const Vec& x) {
            ++calls;
            return (x(0) - 1e13 - 2.0) * (x(0) - 1e13 - 2.0) + x(1) * x(1);
        }
    };
    Journal j;
    Bowl host;
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    CHECK(host.calls == 1u);
    const std::string journal = j.text();
    CHECK(journal.find("LN_SBPLX: NLopt internal diagnostic was set after 1 "
                       "evaluation(s) -- 'starting step size led to simplex "
                       "that was too small") != std::string::npos);
}

/* Pins: the journal's "Inner stop criteria" of a block whose INNER search
 * is gradient-based (G_MLSL_LDS, INNER LD_LBFGS with MAXEVAL 10) says that
 * the INNER MAXEVAL counts each local search's NLopt calls, and that the
 * enclosing block's MAXEVAL counts every evaluation of the run (owner
 * decision, 2026-10-07); a derivative-free INNER search (LN_BOBYQA), whose
 * calls are its evaluations, gets no such line.
 * Before the fix: the block listed "Max evaluations : 10" alone. */
TEST_CASE("NLopt MAXEVAL: the journal says what an INNER MAXEVAL counts",
          "[HPO][nlopt][maxeval][journal]")
{
    const std::string line =
        "(MAXEVAL here counts the NLopt calls of each local search, not its "
        "evaluations: a call evaluates the loss and its gradient; the "
        "MAXEVAL of the enclosing block counts every evaluation of the run, "
        "each loss, each finite-difference point and each analytical "
        "gradient)";
    for (const std::string inner : {"LD_LBFGS", "LN_BOBYQA"}) {
        DYNAMIC_SECTION("INNER " << inner) {
            Journal j;
            auto sp = nlopt_spec("G_MLSL_LDS", 60);
            set_inner(sp, inner);
            sp.inner->stop.maxeval = 10;
            HPOLog::instance().write_optimizer_section(sp, {"p0", "p1"});
            const std::string journal = j.text();
            const auto at = journal.find("Inner stop criteria:");
            REQUIRE(at != std::string::npos);
            CHECK(journal.find("Max evaluations", at) != std::string::npos);
            CHECK((journal.find(line, at) != std::string::npos) ==
                  (inner == "LD_LBFGS"));
        }
    }
}

/* Pins: on the valley at MAXEVAL 60 (NUMERICAL, CENTRAL), NLopt's LD_LBFGS,
 * LD_MMA and LD_TNEWTON and CERES:LBFGS (WOLFE) ask the host for 60 points
 * and never for one point twice in a row: each gradient's first evaluation
 * is a stencil point, not the point whose loss its caller has just
 * evaluated.
 * Before the fix: each gradient evaluated its point once more first:
 * each of the four searches asked for 10 points twice in a row (10
 * gradients of 6 evaluations: the loss, the point again, 4 stencil
 * points). */
TEST_CASE("Finite differences: NLopt and Ceres evaluate no point twice in a "
          "row", "[HPO][grad_provider][fd_gradient][maxeval]")
{
    for (const std::string algo : {"LD_LBFGS", "LD_MMA", "LD_TNEWTON"}) {
        DYNAMIC_SECTION("NLOPT:" << algo) {
            Journal j;
            Valley host;
            REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(
                nlopt_spec(algo, 60), host));
            CHECK(host.asked.size() == 60u);
            CHECK(host.repeats() == 0u);
        }
    }
#ifdef TADAH_USE_CERES
    SECTION("CERES:LBFGS") {
        Journal j;
        auto sp = nlopt_spec("LBFGS", 60);
        sp.lib = "CERES";
        sp.seed.reset();
        Valley host;
        REQUIRE_NOTHROW(
            tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
                sp, host, ceres::LBFGS, ceres::WOLFE, "LBFGS"));
        CHECK(host.asked.size() == 60u);
        CHECK(host.repeats() == 0u);
    }
#endif
}

/* Pins: GradientProvider (NUMERICAL, fd_step 1e-2) on f = u0 + 2 u1 in
 * [0, 1]^2: gradient_at(x, f(x), g) evaluates 4 points for CENTRAL at
 * (0.5, 0.5) and for the one-sided differences at (0, 1), both on a bound,
 * 2 for FORWARD at (0.5, 0.5) and at (0.5, 1) (backward in u1), none of
 * them x, all counted, and the exact gradient (1, 2); gradient(x, g)
 * evaluates x first, one more.
 * Before the fix: gradient_at evaluated x first as well: 5, 5, 3 and 3
 * points. */
TEST_CASE("Finite differences: a gradient takes the loss its caller has just "
          "evaluated as its centre", "[HPO][grad_provider][fd_gradient]")
{
    auto sp = nlopt_spec("LD_LBFGS", 1);
    sp.stop.maxeval.reset();
    sp.low = {0.0, 0.0};
    sp.high = {1.0, 1.0};
    sp.param["fd_step"] = 1e-2;
    struct Line {
        std::vector<Vec> asked;
        double operator()(const Vec& x) {
            asked.push_back(x);
            return x(0) + 2.0 * x(1);
        }
    };
    const auto f = [](double a, double b) { return a + 2.0 * b; };
    for (const auto& [u0, u1, method, evals] :
         std::vector<std::tuple<double, double, int, std::size_t>>{
             {0.5, 0.5, 0, 4u},    // central
             {0.0, 1.0, 0, 4u},    // one-sided at both bounds
             {0.5, 0.5, 1, 2u},    // forward
             {0.5, 1.0, 1, 2u}}) { // backward at the upper bound
        DYNAMIC_SECTION("(" << u0 << ", " << u1 << ") "
                            << (method ? "FORWARD" : "CENTRAL")) {
            auto s = sp;
            s.param["diff_method"] = method;
            Vec x(2);
            x(0) = u0;
            x(1) = u1;
            Vec g;
            Line host;
            std::atomic<unsigned long> counter{0};
            auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                s, host, &counter, 0);
            REQUIRE(gp.gradient_at(x, f(u0, u1), g));
            CHECK(counter.load() == evals);
            CHECK(host.asked.size() == evals);
            for (const Vec& p : host.asked)
                CHECK_FALSE((p(0) == u0 && p(1) == u1));
            CHECK(g(0) == Approx(1.0).epsilon(1e-9));
            CHECK(g(1) == Approx(2.0).epsilon(1e-9));
            /* without the caller's loss: x first */
            Line host2;
            std::atomic<unsigned long> counter2{0};
            auto gp2 = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
                s, host2, &counter2, 0);
            REQUIRE(gp2.gradient(x, g));
            CHECK(counter2.load() == evals + 1);
            REQUIRE_FALSE(host2.asked.empty());
            CHECK((host2.asked[0](0) == u0 && host2.asked[0](1) == u1));
            CHECK(g(0) == Approx(1.0).epsilon(1e-9));
            CHECK(g(1) == Approx(2.0).epsilon(1e-9));
        }
    }
}

/* Pins: gradient_at(x, +inf, g) (NUMERICAL) fails with a Stencil failure
 * naming the loss at x, and evaluates nothing: the caller's loss is the one
 * checked.
 * Before the fix: it evaluated x again (24.2 on the valley) and its 4
 * stencil points and returned a gradient. */
TEST_CASE("Finite differences: a caller's loss that is not finite fails the "
          "gradient without an evaluation", "[HPO][grad_provider][fd_gradient]")
{
    auto sp = nlopt_spec("LD_LBFGS", 1);
    sp.stop.maxeval.reset();
    Valley host;
    std::atomic<unsigned long> counter{0};
    auto gp = tadah::hpo::make_gradient_provider<Vec, NoRunner>(
        sp, host, &counter, 0);
    Vec x(2);
    x(0) = 0.5;
    x(1) = 0.5;
    Vec g;
    CHECK_FALSE(
        gp.gradient_at(x, std::numeric_limits<double>::infinity(), g));
    CHECK(gp.failure->kind == tadah::hpo::GradientFailure::Stencil);
    CHECK(gp.failure->detail.find("is inf") != std::string::npos);
    CHECK(counter.load() == 0u);
    CHECK(host.asked.empty());
}

namespace {

/* The valley, with a failed loss just below x(1) = 0: the stencil point
 * (0, -1e-6) of GD_STOGO's first gradient, at the centre of [-2, 2]^2
 * (loss 1), fails. */
struct CentreCliff : Valley {
    double operator()(const Vec& x) {
        asked.push_back(x);
        return (x(1) < -1e-7 && x(1) > -1e-5)
                   ? std::numeric_limits<double>::max()
                   : valley(x);
    }
};

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

/* Pins: GD_STOGO ignores NLopt's forced stop. At MAXEVAL 60, MAXTIME 60
 * and STOPVAL 2, its first gradient, at the centre of [-2, 2]^2 (loss 1),
 * has a stencil point that fails, (0, -1e-6): the loss 1 is withheld and
 * ends the search STOPVAL_REACHED with no evaluation after that stencil
 * point, though StoGO calls the objective on until NLopt's own MAXEVAL ends
 * it: the objective refuses those calls, and run_nlopt_impl returns in
 * well under 10 s, far from MAXTIME. That is why GD_STOGO needs MAXEVAL
 * (backend_rules; the next test case).
 * Before the fix: passes as well (a pin of the path the startup rule
 * relies on: NLopt's own MAXEVAL ends a StoGO search the objective
 * stopped, at once). */
TEST_CASE("NLopt STOPVAL: GD_STOGO, which ignores the forced stop, ends at a "
          "withheld loss through NLopt's own MAXEVAL",
          "[HPO][nlopt][maxeval][stopval]")
{
    Journal j;
    auto sp = nlopt_spec("GD_STOGO", 60);
    sp.stop.stopval = 2.0;
    sp.stop.maxtime = 60.0;
    CentreCliff host;
    const auto t0 = std::chrono::steady_clock::now();
    REQUIRE_NOTHROW(tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
    CHECK(seconds_since(t0) < 10.0);
    REQUIRE_FALSE(host.asked.empty());
    CHECK(host.asked.size() < 60u);
    CHECK(host.asked.back()(0) == 0.0);
    CHECK(host.asked.back()(1) == -1e-6);
    const std::string journal = j.text();
    CHECK(status_of(journal) == kStopvalReached);
    CHECK(journal.find("NLopt finished: " + kStopvalReached + " after " +
                       std::to_string(host.asked.size()) +
                       " evaluation(s): the loss 1 is at or below STOPVAL 2") !=
          std::string::npos);
}

/* Pins: GD_STOGO with MAXTIME 2 and STOPVAL 2 but no MAXEVAL is refused
 * before it evaluates anything (run_nlopt_impl -> make_nlopt ->
 * require_stop): "NLOPT:GD_STOGO needs a stop: ...; set MAXEVAL (it honours
 * no tolerance)", within a second. Once HPO stops a StoGO search (here the
 * withheld loss 1 at STOPVAL 2 of the case above; a signal, an error or the
 * RESTART stagnation stop alike), StoGO calls on and the objective refuses
 * each call: only NLopt's own MAXEVAL ends that promptly.
 * Before the fix: the run was accepted, the search stopped after 5
 * evaluations, and the objective refused StoGO's calls at 100% CPU for the
 * rest of MAXTIME: run_nlopt_impl returned after about 2 s, STOPVAL_REACHED
 * (and with MAXTIME 3600, after an hour). The code before the withheld
 * STOPVAL rule gave StoGO +inf there, and StoGO went on evaluating until
 * its own STOPVAL test, on later losses below 2 it was given, ended it
 * after 54 evaluations. */
TEST_CASE("NLopt STOPVAL: GD_STOGO without MAXEVAL is refused before it "
          "evaluates, instead of waiting out MAXTIME once HPO stops it",
          "[HPO][nlopt][maxeval][stopval]")
{
    for (const std::string algo : {"GD_STOGO", "GD_STOGO_RAND"}) {
        DYNAMIC_SECTION(algo) {
            Journal j;
            auto sp = nlopt_spec(algo, 60);
            sp.stop.maxeval.reset();
            sp.stop.maxtime = 2.0;
            sp.stop.stopval = 2.0;
            CentreCliff host;
            const auto t0 = std::chrono::steady_clock::now();
            CHECK_THROWS_WITH(
                (tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host)),
                Catch::StartsWith("NLOPT:" + algo + " needs a stop: ") &&
                    Catch::EndsWith("; set MAXEVAL (it honours no "
                                    "tolerance)"));
            CHECK(seconds_since(t0) < 1.0);
            CHECK(host.asked.empty());
        }
    }
}

namespace {

/* Whether a search asked for a point it stepped to: a point after the first
 * `before` -- its start, and an MLSL's samples -- farther than 1e-3 in some
 * coordinate from every point asked for before it. The finite-difference
 * stencil of a point (fd_step 1e-6) and a point asked for again are not. */
bool stepped(const std::vector<Vec>& asked, std::size_t before)
{
    for (std::size_t k = before; k < asked.size(); ++k) {
        bool near = false;
        for (std::size_t j = 0; j < k && !near; ++j)
            near = std::abs(asked[k](0) - asked[j](0)) <= 1e-3 &&
                   std::abs(asked[k](1) - asked[j](1)) <= 1e-3;
        if (!near) return true;
    }
    return false;
}

}  // namespace

/* Pins: the startup WARNING of a gradient search whose budget takes no step
 * (gradient_budget_warnings, backend_rules.h) is given exactly at the
 * budgets whose run takes no step: on the valley, for every MAXEVAL m from
 * 1 to the first budget that steps, the warning is given at m if and only
 * if the run at m (run_nlopt_impl, the host asked for m points) asks for no
 * point it stepped to (see stepped()):
 *   - LD_LBFGS (NUMERICAL, CENTRAL in 2-D): the start, its 4-point stencil,
 *     a first step at evaluation 6;
 *   - LD_LBFGS under ANALYTICAL: the start's loss and analytical gradient, a
 *     first step at evaluation 3;
 *   - AUGLAG with INNER LD_LBFGS: its local search from the start at once,
 *     a first step at evaluation 6;
 *   - GD_MLSL with INNER LD_LBFGS (SEED 7, NLopt's 4 samples per round):
 *     the start and 4 samples, the local search's start again and its
 *     stencil, a first step at evaluation 11;
 *   - G_MLSL_LDS with POPULATION 3 and INNER LD_LBFGS: a first step at
 *     evaluation 10.
 * Before the fix: the warning took an MLSL for a search that takes the
 * gradient at its start (no warning from MAXEVAL 6 on), so the GD_MLSL
 * checks at MAXEVAL 6 to 10 and the G_MLSL_LDS ones at 6 to 9 fail; the
 * other searches pass as well (pins). */
TEST_CASE("NLopt MAXEVAL: the startup warning of a gradient budget that "
          "takes no step is given exactly when the run takes none",
          "[HPO][nlopt][maxeval][gradient_budget]")
{
    struct Case {
        std::string label, algo, inner;
        bool analytical;
        unsigned pop;            // 0: NLopt's default
        std::size_t before;      // the start, and an MLSL's samples
        unsigned first_step;     // the evaluation of the first step
    };
    const std::vector<Case> cases = {
        {"LD_LBFGS", "LD_LBFGS", "", false, 0, 1, 6},
        {"LD_LBFGS, ANALYTICAL", "LD_LBFGS", "", true, 0, 1, 3},
        {"AUGLAG, INNER LD_LBFGS", "AUGLAG", "LD_LBFGS", false, 0, 1, 6},
        {"GD_MLSL, INNER LD_LBFGS", "GD_MLSL", "LD_LBFGS", false, 0, 5, 11},
        {"G_MLSL_LDS, POPULATION 3, INNER LD_LBFGS", "G_MLSL_LDS", "LD_LBFGS",
         false, 3, 4, 10}};
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.label) {
            for (unsigned m = 1; m <= c.first_step; ++m) {
                INFO(c.label << " MAXEVAL " << m);
                auto sp = nlopt_spec(c.algo, m);
                if (!c.inner.empty()) set_inner(sp, c.inner);
                if (c.pop) sp.pop = c.pop;
                if (c.analytical) {
                    sp.grad_strategy = GradStrategy::Analytical;
                    sp.param.clear();
                    sp.param_raw.clear();
                }
                const std::vector<std::string> warnings =
                    tadah::hpo::gradient_budget_warnings(sp);
                INFO("warnings: " << (warnings.empty() ? "(none)"
                                                        : warnings.front()));
                Journal j;
                Valley host;
                REQUIRE_NOTHROW(
                    tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, host));
                (void)j.text();
                CHECK(host.asked.size() <= m);
                const bool step = stepped(host.asked, c.before);
                CHECK(step == (m == c.first_step));
                CHECK(warnings.empty() == step);
            }
        }
    }
}

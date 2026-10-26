// test_gradient_budget.cpp
//
// A gradient search evaluates the loss at its starting point and the
// gradient there before it evaluates any point it steps to. A MAXEVAL that
// cannot pay for both and one more evaluation ends the search where it
// started: with GRAD_STRATEGY NUMERICAL and central differences in n
// dimensions the start costs 1 + 2n evaluations, so MAXEVAL 2n + 1 or less
// takes no step (LD_LBFGS and DLIB:BFGS in 2-D, and CERES:LBFGS, refused
// since: MAXEVAL 4 and 5 evaluate the start and its stencil, MAXEVAL 6 a
// first step). Such a budget
// can be deliberate (a probe of the start), so validate_hpo_setup warns, for
// every phase that has a MAXEVAL of its own, naming what the gradient costs
// and the budget that takes a step.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

std::string unique_name(const std::string& stem)
{
    static unsigned counter = 0;
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

std::string slurp(const fs::path& p)
{
    std::ifstream in(p);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

/* The journal of one startup check, in a file of its own. */
struct JournalCapture {
    fs::path path = fs::temp_directory_path() /
                    (unique_name("tadah_budget_journal_") + ".log");
    JournalCapture()
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        REQUIRE(log.is_open());
    }
    std::string text() const { return slurp(path); }
    ~JournalCapture()
    {
        HPOLog::instance().finish("test-end");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
};

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

/* Two OPTIM dimensions and a loss. PC_ERMSE alone: every evaluation is a
 * function of its point (HPOTargets::repeat_reuse_blocker() is ""), so a
 * search answers a point it asks for again from its record
 * (repeat_points.h). */
const std::string kRoot =
    "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\nPC_ERMSE 0 1\n";

/* The same with a --skip-rel filter, which compares with the best value so
 * far: two evaluations of one point can differ, and every point asked for
 * again is evaluated again. */
const std::string kRootSkipRel =
    "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
    "PC_ERMSE 0 1 --skip-rel 5\n";

std::string strategy(const std::string& body, const std::string& root = kRoot)
{
    return root + "OPTIMIZER\n  STRATEGY\n" + body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

std::string plain(const std::string& lib, const std::string& algo,
                  const std::string& maxeval, const std::string& extra = "")
{
    return strategy("    TYPE PLAIN\n    LIB " + lib + "\n    ALGO " + algo +
                    "\n    MAXEVAL " + maxeval + "\n" + extra);
}

/* What validate_hpo_setup() journals for `config`: it must accept it. */
std::string startup_journal(const std::string& config)
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_budget_") + ".hpo");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << config;
    }
    Context ctx = make_ctx();
    std::string journal;
    {
        JournalCapture j;
        INFO("config:\n" << config);
        HPOTargets trg(ctx, file.string());
        REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, true, true));
        journal = j.text();
    }
    std::error_code ec;
    fs::remove(file, ec);
    return journal;
}

/* What validate_hpo_setup() refuses `config` with, its line included ("" when
 * it accepts it); the parser must accept it. */
std::string startup_refusal(const std::string& config)
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_budget_") + ".hpo");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << config;
    }
    Context ctx = make_ctx();
    std::string what;
    {
        JournalCapture j;
        INFO("config:\n" << config);
        HPOTargets trg(ctx, file.string());
        try {
            tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
        } catch (const std::exception& e) {
            what = e.what();
        }
    }
    std::error_code ec;
    fs::remove(file, ec);
    return what;
}

std::size_t count_of(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto p = text.find(needle); p != std::string::npos;
         p = text.find(needle, p + needle.size()))
        ++n;
    return n;
}

const std::string kNoStep = "takes no step";

/* The tail every NUMERICAL central 2-D warning has. */
std::string central2(const std::string& budget, const std::string& needs)
{
    return ": " + budget + " takes no step: its gradient search evaluates "
           "the loss at a starting point (1 evaluation) and the gradient there "
           "(4 evaluations: central differences in 2 dimensions, "
           "GRAD_STRATEGY NUMERICAL) before it evaluates a point it steps to, "
           "which needs " + needs;
}

/* The same when the search's start is a point the run has evaluated (a
 * PRESTAGE refinement's, an exploration candidate; the seed point that
 * NOISE_CALIBRATION evaluated) and the run answers it from its record: the
 * loss there is the recorded one, the centre of the finite differences,
 * and only their stencil is evaluated. */
std::string record_central2(const std::string& budget, const std::string& needs)
{
    return ": " + budget + " takes no step: its gradient search takes the "
           "loss at its starting point from the run's record (no evaluation) "
           "and evaluates the gradient there (4 evaluations: central "
           "differences in 2 dimensions, GRAD_STRATEGY NUMERICAL) before it "
           "evaluates a point it steps to, which needs " + needs;
}

}  // namespace

/* Pins: a gradient search in 2-D with GRAD_STRATEGY NUMERICAL (central
 * differences, the default) whose MAXEVAL is 4 or 5 -- below the 1 + 4 + 1
 * evaluations of its start, the gradient there and one point it steps to --
 * gets one startup WARNING, naming the phase, the optimiser, the gradient's
 * cost and the MAXEVAL that takes a step: "the STRATEGY-level optimiser
 * NLOPT:LD_LBFGS: MAXEVAL 4 takes no step: its gradient search evaluates
 * the loss at a starting point (1 evaluation) and the gradient there (4
 * evaluations: central differences in 2 dimensions, GRAD_STRATEGY
 * NUMERICAL) before it evaluates a point it steps to, which needs MAXEVAL 6
 * or more", for NLOPT:LD_LBFGS, DLIB:BFGS and CERES:STEEPEST_DESCENT (it
 * was CERES:LBFGS, refused since: that row a pin); MAXEVAL 6 gets
 * none (the sixth evaluation is the first step). The CERES row runs in a
 * build with Ceres only: without it (TADAH_ENABLE_CERES OFF) startup
 * refuses LIB CERES ("CERES support is not built"), which failed the case
 * there.
 * Before the fix: no WARNING for any of them (the search ended
 * MAXEVAL_REACHED at its starting point). */
TEST_CASE("startup validation: a gradient search whose MAXEVAL cannot pay for "
          "one step is warned about",
          "[startup_validation][gradient_budget]")
{
    std::vector<std::pair<std::string, std::string>> searches = {
        {"NLOPT", "LD_LBFGS"}, {"DLIB", "BFGS"}};
#ifdef TADAH_USE_CERES
    searches.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    for (const auto& [lib, algo] : searches) {
        for (const char* maxeval : {"4", "5"}) {
            INFO(lib << ":" << algo << " MAXEVAL " << maxeval);
            const std::string journal =
                startup_journal(plain(lib, algo, maxeval));
            CHECK(count_of(journal,
                           "the STRATEGY-level optimiser " + lib + ":" + algo +
                               central2(std::string("MAXEVAL ") + maxeval,
                                        "MAXEVAL 6 or more")) == 1u);
            CHECK(count_of(journal, kNoStep) == 1u);
        }
        INFO(lib << ":" << algo << " MAXEVAL 6");
        CHECK(count_of(startup_journal(plain(lib, algo, "6")), kNoStep) == 0u);
    }
}

/* Pins: the warning counts what the configured gradient costs:
 *   - FORWARD differences (PARAM diff_method 1): n evaluations after the
 *     loss, so 2-D needs MAXEVAL 4 ("(2 evaluations: forward differences in
 *     2 dimensions, GRAD_STRATEGY NUMERICAL)"), and MAXEVAL 4 gets none;
 *   - GRAD_STRATEGY ANALYTICAL: one counted analytical gradient, so MAXEVAL
 *     2 warns and needs 3 ("(1 evaluation: one analytical gradient,
 *     GRAD_STRATEGY ANALYTICAL)"), and MAXEVAL 3 gets none;
 *   - GRAD_STRATEGY HYBRID: at least the analytical gradient, so MAXEVAL 2
 *     warns, needing 3 and more for each parameter it differences, and
 *     MAXEVAL 3 gets none (the parameters it differences are known only at
 *     a point);
 *   - a derivative-free optimiser takes no gradient: LN_BOBYQA with MAXEVAL
 *     2 gets none.
 * Before the fix: no WARNING (the ANALYTICAL and HYBRID ones and the
 * FORWARD one fail; the rest pass as well, pins). */
TEST_CASE("startup validation: the gradient budget warning counts the "
          "configured gradient",
          "[startup_validation][gradient_budget]")
{
    {
        const std::string journal = startup_journal(
            plain("NLOPT", "LD_LBFGS", "3", "    PARAM diff_method 1\n"));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL "
                       "3 takes no step: its gradient search evaluates the "
                       "loss at a starting point (1 evaluation) and the "
                       "gradient there (2 evaluations: forward differences "
                       "in 2 dimensions, GRAD_STRATEGY NUMERICAL) before it "
                       "evaluates a point it steps to, which needs MAXEVAL 4 "
                       "or more") == 1u);
        CHECK(count_of(startup_journal(plain("NLOPT", "LD_LBFGS", "4",
                                             "    PARAM diff_method 1\n")),
                       kNoStep) == 0u);
    }
    {
        const std::string journal = startup_journal(plain(
            "NLOPT", "LD_LBFGS", "2", "    GRAD_STRATEGY ANALYTICAL\n"));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL "
                       "2 takes no step: its gradient search evaluates the "
                       "loss at a starting point (1 evaluation) and the "
                       "gradient there (1 evaluation: one analytical "
                       "gradient, GRAD_STRATEGY ANALYTICAL) before it "
                       "evaluates a point it steps to, which needs MAXEVAL 3 "
                       "or more") == 1u);
        CHECK(count_of(startup_journal(plain("NLOPT", "LD_LBFGS", "3",
                                             "    GRAD_STRATEGY ANALYTICAL\n")),
                       kNoStep) == 0u);
    }
    {
        const std::string journal = startup_journal(
            plain("DLIB", "BFGS", "2", "    GRAD_STRATEGY HYBRID\n"));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser DLIB:BFGS: MAXEVAL 2 "
                       "takes no step: its gradient search evaluates the loss "
                       "at a starting point (1 evaluation) and the gradient "
                       "there (at least 1 evaluation: the analytical "
                       "gradient, and the finite differences of each "
                       "parameter it does not cover, GRAD_STRATEGY HYBRID) "
                       "before it evaluates a point it steps to, which needs "
                       "MAXEVAL 3 or more, and more for each parameter it "
                       "differences") == 1u);
        CHECK(count_of(startup_journal(plain("DLIB", "BFGS", "3",
                                             "    GRAD_STRATEGY HYBRID\n")),
                       kNoStep) == 0u);
    }
    CHECK(count_of(startup_journal(plain("NLOPT", "LN_BOBYQA", "2")),
                   kNoStep) == 0u);
}

/* Pins: the warning is given for every phase with a MAXEVAL of its own:
 *   - each MULTISTART start, with its BUDGET_PER_START ("each MULTISTART
 *     start NLOPT:LD_LBFGS: BUDGET_PER_START 4 takes no step: ... which
 *     needs BUDGET_PER_START 6 or more") or its share of MAXEVAL ("MAXEVAL
 *     8 over 2 starts, 4 each, takes no step: ... which needs 6 evaluations
 *     or more per start"); a MAXEVAL 12 over 2 starts gets none;
 *   - the PRESTAGE refinement, whose start, an exploration candidate, the
 *     run has evaluated and answers from its record ("the PRESTAGE
 *     refinement NLOPT:LD_LBFGS: MAXEVAL 4 takes no step: its gradient
 *     search takes the loss at its starting point from the run's record (no
 *     evaluation) and evaluates the gradient there (4 evaluations: ...)
 *     ..., which needs MAXEVAL 5 or more"; MAXEVAL 5 gets none), and not
 *     the derivative-free exploration; with TOPK 2 each candidate's share
 *     ("MAXEVAL 8 over 2 TOPK candidates, 4 each, takes no step: ... which
 *     needs 5 evaluations or more per candidate"; MAXEVAL 10 gets none);
 *   - an INNER block, whose MAXEVAL NLopt counts in its own calls, each the
 *     loss and the gradient at a point: INNER LD_LBFGS MAXEVAL 1 under
 *     GD_MLSL, whose local searches start at points MLSL evaluated, whose
 *     loss the run's record answers (every evaluation of this run is a
 *     function of its point; NUMERICAL takes it as the centre of the
 *     finite differences, repeat_use, repeat_points.h): "the INNER block of
 *     the STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL 1 takes no step:
 *     NLopt counts an INNER block's MAXEVAL in its calls, each the loss and
 *     the gradient at a point (4 evaluations: central differences in 2
 *     dimensions, GRAD_STRATEGY NUMERICAL; MLSL's local searches start at
 *     points it evaluated, whose loss is taken from the run's record),
 *     so each local search evaluates the gradient at its starting point and
 *     ends there, which needs MAXEVAL 2 or more"; under a --skip-rel filter,
 *     where every repeat is evaluated again, "(1 + 4 evaluations: central
 *     differences in 2 dimensions, GRAD_STRATEGY NUMERICAL), so each local
 *     search evaluates its starting point and ends there", and so under
 *     AUGLAG, whose local search starts at the phase's start, a point the
 *     search has not evaluated; MAXEVAL 2 gets none, and so does an INNER
 *     LN_BOBYQA MAXEVAL 1, which takes no gradient.
 * Before the fix: the INNER LD_LBFGS MAXEVAL 1 check under GD_MLSL without
 * --skip-rel fails: its text was the --skip-rel one, "(1 + 4 evaluations:
 * ...), so each local search evaluates its starting point", which counted
 * a loss the record answers; every other check passes as well (pins; until
 * the warning was added none was given).
 * Before the run kept one record (fix/repeat-followups): the PRESTAGE
 * refinement was warned with the start evaluated again, "its gradient
 * search evaluates the loss at a starting point (1 evaluation) ...,
 * which needs MAXEVAL 6 or more" ("6 evaluations or more per candidate"),
 * and MAXEVAL 5 (10 over 2 candidates) as well, although the run's
 * refinement steps at its fifth evaluation; the texts said "from the
 * search's record": those checks fail. */
TEST_CASE("startup validation: the gradient budget warning covers each "
          "MULTISTART start, the PRESTAGE refinement and an INNER block",
          "[startup_validation][gradient_budget]")
{
    const std::string init =
        "  INIT\n    METHOD LHS 2\n    SEED 3\n  ENDINIT\n";
    const auto multistart = [&](const std::string& keys) {
        return kRoot + "OPTIMIZER\n" + init +
               "  STRATEGY\n    TYPE MULTISTART\n    LIB NLOPT\n"
               "    ALGO LD_LBFGS\n    N_STARTS 2\n" + keys +
               "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    CHECK(count_of(startup_journal(multistart(
                       "    MAXEVAL 100\n    BUDGET_PER_START 4\n")),
                   "each MULTISTART start NLOPT:LD_LBFGS" +
                       central2("BUDGET_PER_START 4",
                                "BUDGET_PER_START 6 or more")) == 1u);
    CHECK(count_of(startup_journal(multistart("    MAXEVAL 8\n")),
                   "each MULTISTART start NLOPT:LD_LBFGS" +
                       central2("MAXEVAL 8 over 2 starts, 4 each,",
                                "6 evaluations or more per start")) == 1u);
    CHECK(count_of(startup_journal(multistart("    MAXEVAL 12\n")), kNoStep) ==
          0u);

    {
        const std::string journal = startup_journal(strategy(
            "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
            "    MAXEVAL 4\n    EXPLORE\n      LIB NLOPT\n"
            "      ALGO GN_DIRECT\n      MAXEVAL 10\n    ENDEXPLORE\n"));
        CHECK(count_of(journal, "the PRESTAGE refinement NLOPT:LD_LBFGS" +
                                    record_central2("MAXEVAL 4",
                                                    "MAXEVAL 5 or more")) ==
              1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    CHECK(count_of(startup_journal(strategy(
                       "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
                       "    MAXEVAL 5\n    EXPLORE\n      LIB NLOPT\n"
                       "      ALGO GN_DIRECT\n      MAXEVAL 10\n"
                       "    ENDEXPLORE\n")),
                   kNoStep) == 0u);

    /* With TOPK 2 the refinement runs once from each candidate, each with
     * its share of MAXEVAL (plan_multistart, ceil division). */
    {
        const std::string journal = startup_journal(strategy(
            "    TYPE PRESTAGE\n    TOPK 2\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
            "    MAXEVAL 8\n    EXPLORE\n      LIB NLOPT\n"
            "      ALGO GN_DIRECT\n      MAXEVAL 10\n    ENDEXPLORE\n"));
        CHECK(count_of(journal,
                       "the PRESTAGE refinement NLOPT:LD_LBFGS" +
                           record_central2("MAXEVAL 8 over 2 TOPK candidates, "
                                           "4 each,",
                                           "5 evaluations or more per "
                                           "candidate")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    CHECK(count_of(startup_journal(strategy(
                       "    TYPE PRESTAGE\n    TOPK 2\n    LIB NLOPT\n"
                       "    ALGO LD_LBFGS\n    MAXEVAL 10\n    EXPLORE\n"
                       "      LIB NLOPT\n      ALGO GN_DIRECT\n"
                       "      MAXEVAL 10\n    ENDEXPLORE\n")),
                   kNoStep) == 0u);

    const auto mlsl = [](const std::string& inner_algo,
                         const std::string& inner_maxeval,
                         const std::string& algo = "GD_MLSL",
                         const std::string& root = kRoot) {
        return strategy("    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
                            "\n    MAXEVAL 100\n    INNER\n      LIB NLOPT\n"
                            "      ALGO " + inner_algo + "\n      MAXEVAL " +
                            inner_maxeval + "\n    ENDINNER\n",
                        root);
    };
    const std::string inner_head =
        "the INNER block of the STRATEGY-level optimiser NLOPT:LD_LBFGS: "
        "MAXEVAL 1 takes no step: NLopt counts an INNER block's MAXEVAL in "
        "its calls, each the loss and the gradient at a point ";
    const std::string inner_evaluated =
        inner_head + "(1 + 4 evaluations: central differences in 2 "
        "dimensions, GRAD_STRATEGY NUMERICAL), so each local search "
        "evaluates its starting point and ends there, which needs MAXEVAL 2 "
        "or more";
    {
        const std::string journal = startup_journal(mlsl("LD_LBFGS", "1"));
        CHECK(count_of(journal,
                       inner_head + "(4 evaluations: central differences in "
                       "2 dimensions, GRAD_STRATEGY NUMERICAL; MLSL's local "
                       "searches start at points it evaluated, whose loss is "
                       "taken from the run's record), so each local "
                       "search evaluates the gradient at its starting point "
                       "and ends there, which needs MAXEVAL 2 or more") ==
              1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    {
        const std::string journal = startup_journal(
            mlsl("LD_LBFGS", "1", "GD_MLSL", kRootSkipRel));
        CHECK(count_of(journal, inner_evaluated) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    {
        const std::string journal =
            startup_journal(mlsl("LD_LBFGS", "1", "AUGLAG"));
        CHECK(count_of(journal, inner_evaluated) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    CHECK(count_of(startup_journal(mlsl("LD_LBFGS", "2")), kNoStep) == 0u);
    CHECK(count_of(startup_journal(mlsl("LN_BOBYQA", "1")), kNoStep) == 0u);
}

namespace {

/* The beginning of an MLSL warning's tail in 2-D: MLSL evaluates its start
 * and `samples` points without a gradient. */
std::string mlsl_head(const std::string& budget, unsigned samples,
                      const std::string& population)
{
    return ": " + budget + " takes no step of a local search: MLSL evaluates "
           "its starting point and its first " + std::to_string(samples) +
           " samples without a gradient (" + std::to_string(1 + samples) +
           " evaluations; " + population + "), then its first local search, "
           "from one of them, ";
}

/* The tail of an MLSL warning in 2-D under NUMERICAL central differences
 * when the run evaluates a point asked for again: its first local search
 * evaluates the loss at its start again and the gradient there. */
std::string mlsl_central2(const std::string& budget, unsigned samples,
                          const std::string& population,
                          const std::string& needs)
{
    return mlsl_head(budget, samples, population) +
           "evaluates the loss there again (1 evaluation) and the gradient "
           "there (4 evaluations: central differences in 2 dimensions, "
           "GRAD_STRATEGY NUMERICAL) before it evaluates a point it steps to, "
           "which needs " + needs;
}

/* The same when the run answers a point asked for again from the search's
 * record: the loss at the local search's start is the recorded one, the
 * centre of the finite differences, and only their stencil is evaluated. */
std::string mlsl_record_central2(const std::string& budget, unsigned samples,
                                 const std::string& population,
                                 const std::string& needs)
{
    return mlsl_head(budget, samples, population) +
           "takes the loss there from the run's record (no evaluation) "
           "and evaluates the gradient there (4 evaluations: central "
           "differences in 2 dimensions, GRAD_STRATEGY NUMERICAL) before it "
           "evaluates a point it steps to, which needs " + needs;
}

std::string mlsl_plain(const std::string& algo, const std::string& maxeval,
                       const std::string& keys,
                       const std::string& root = kRoot)
{
    return strategy("    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
                        "\n    MAXEVAL " + maxeval + "\n" + keys,
                    root);
}

const std::string kInnerLbfgs =
    "    INNER\n      LIB NLOPT\n      ALGO LD_LBFGS\n    ENDINNER\n";

}  // namespace

/* Pins: an MLSL whose local search takes gradients spends, before its first
 * local search can step, its starting point and its first N samples, both
 * without a gradient (mlsl.c: grad NULL; N is POPULATION, NLopt's default
 * 4), then the gradient at that search's start, one of these points. Here
 * every evaluation is a function of its point (PC_ERMSE alone), so the
 * search answers the start's loss from its record, and under NUMERICAL
 * that loss is the centre of the finite differences (repeat_use,
 * RepeatUse::Centre, repeat_points.h): in 2-D under central differences 1 +
 * 4 + 4 evaluations, so MAXEVAL 9 or less lets no local search take a step
 * (a run before the fix at MAXEVAL 9: the start, 4 samples, the 4-point
 * stencil of the local start, "Repeated points reused : 1"; at MAXEVAL 10
 * the 10th evaluation is the first step). The WARNING says so:
 *   - GD_MLSL with INNER LD_LBFGS, MAXEVAL 5 and 9: "the STRATEGY-level
 *     optimiser NLOPT:GD_MLSL: MAXEVAL 9 takes no step of a local search:
 *     MLSL evaluates its starting point and its first 4 samples without a
 *     gradient (5 evaluations; POPULATION 4, NLopt's default), then its
 *     first local search, from one of them, takes the loss there from the
 *     run's record (no evaluation) and evaluates the gradient there (4
 *     evaluations: central differences in 2 dimensions, GRAD_STRATEGY
 *     NUMERICAL) before it evaluates a point it steps to, which needs
 *     MAXEVAL 10 or more"; MAXEVAL 10 gets none;
 *   - POPULATION 3: 4 evaluations before the local search, so MAXEVAL 8
 *     warns ("POPULATION 3"), needing 9, and MAXEVAL 9 gets none;
 *   - GD_MLSL without INNER runs NLopt's default local search, LD_MMA,
 *     which takes gradients: MAXEVAL 9 warns alike;
 *   - AUGLAG goes straight into its local search (HPO sets no constraint;
 *     auglag.c): AUGLAG with INNER LD_LBFGS keeps the local rule, MAXEVAL 5
 *     warns with "takes no step: its gradient search evaluates the loss at a
 *     starting point" and MAXEVAL 6 gets none;
 *   - GN_MLSL, whose default local search LN_COBYLA takes no gradient, gets
 *     none at MAXEVAL 3.
 * Where the run evaluates the start again (a --skip-rel filter, ANALYTICAL,
 * HYBRID) MLSL needs one evaluation more: the next test case.
 * Before the fix: the warning counted the local start's loss as evaluated
 * again ("evaluates the loss there again (1 evaluation)"), so GD_MLSL
 * MAXEVAL 5 and 9 were warned about needing MAXEVAL 11, MAXEVAL 10 -- whose
 * 10th evaluation is a step -- was warned about too, and POPULATION 3
 * needed 10 (MAXEVAL 9, which steps, was warned about): the GD_MLSL checks
 * fail; the AUGLAG and GN_MLSL ones pass as well (pins). (fix/hpo-
 * optimiser-hygiene3 pinned MAXEVAL 10 and POPULATION 3 MAXEVAL 9 as
 * warned, on stub hosts that never reuse a point.) */
TEST_CASE("startup validation: the gradient budget warning of an MLSL counts "
          "its samples and its local search's start",
          "[startup_validation][gradient_budget]")
{
    for (const char* maxeval : {"5", "9"}) {
        INFO("GD_MLSL MAXEVAL " << maxeval);
        const std::string journal =
            startup_journal(mlsl_plain("GD_MLSL", maxeval, kInnerLbfgs));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_record_central2(
                               std::string("MAXEVAL ") + maxeval, 4,
                               "POPULATION 4, NLopt's default",
                               "MAXEVAL 10 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    CHECK(count_of(startup_journal(mlsl_plain("GD_MLSL", "10", kInnerLbfgs)),
                   kNoStep) == 0u);

    {
        const std::string journal = startup_journal(mlsl_plain(
            "GD_MLSL", "8", "    POPULATION 3\n" + kInnerLbfgs));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_record_central2("MAXEVAL 8", 3,
                                                "POPULATION 3",
                                                "MAXEVAL 9 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(count_of(startup_journal(mlsl_plain(
                           "GD_MLSL", "9", "    POPULATION 3\n" + kInnerLbfgs)),
                       kNoStep) == 0u);
    }

    {
        const std::string journal =
            startup_journal(mlsl_plain("GD_MLSL", "9", ""));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_record_central2(
                               "MAXEVAL 9", 4, "POPULATION 4, NLopt's default",
                               "MAXEVAL 10 or more")) == 1u);
    }

    {
        const std::string journal =
            startup_journal(mlsl_plain("AUGLAG", "5", kInnerLbfgs));
        CHECK(count_of(journal, "the STRATEGY-level optimiser NLOPT:AUGLAG" +
                                    central2("MAXEVAL 5",
                                             "MAXEVAL 6 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(count_of(startup_journal(mlsl_plain("AUGLAG", "6", kInnerLbfgs)),
                       kNoStep) == 0u);
    }

    CHECK(count_of(startup_journal(mlsl_plain("GN_MLSL", "3", "")), kNoStep) ==
          0u);
}

/* Pins: where the run evaluates the start of an MLSL's first local search
 * again, the warning counts that loss (1 evaluation), as the run spends it
 * (RepeatPoints::use and the warning share repeat_use, repeat_points.h):
 *   - a --skip-rel filter compares with the best value so far, so two
 *     evaluations of one point can differ and every repeat is evaluated
 *     again (HPOTargets::repeat_reuse_blocker): GD_MLSL with INNER LD_LBFGS
 *     in 2-D, MAXEVAL 10: "... then its first local search, from one of
 *     them, evaluates the loss there again (1 evaluation) and the gradient
 *     there (4 evaluations: central differences in 2 dimensions,
 *     GRAD_STRATEGY NUMERICAL) before it evaluates a point it steps to,
 *     which needs MAXEVAL 11 or more"; MAXEVAL 11 gets none
 *     (a run before the fix at MAXEVAL 10: the local start is evaluated
 *     again, eval 6, then its stencil; at MAXEVAL 11 the 11th evaluation is
 *     the first step); with POPULATION 3, MAXEVAL 9 warns needing 10;
 *   - GRAD_STRATEGY ANALYTICAL differentiates the last evaluation, so its
 *     gradient follows an evaluation of the point: MAXEVAL 7 warns, "...
 *     evaluates the loss there again (1 evaluation) and the gradient there
 *     (1 evaluation: one analytical gradient, GRAD_STRATEGY ANALYTICAL)
 *     before it evaluates a point it steps to, which needs MAXEVAL 8 or
 *     more" (5 + 1 + 1 + a step), and MAXEVAL 8 gets none;
 *   - GRAD_STRATEGY HYBRID likewise: MAXEVAL 7 warns "... (at least 1
 *     evaluation: the analytical gradient, and the finite differences of
 *     each parameter it does not cover, GRAD_STRATEGY HYBRID) ... which
 *     needs MAXEVAL 8 or more, and more for each parameter it differences",
 *     and MAXEVAL 8 gets none.
 * Before the fix: passes as well (a pin: the warning counted the local
 * start's loss whatever the run). */
TEST_CASE("startup validation: the gradient budget warning of an MLSL counts "
          "its local search's start where the run evaluates it again",
          "[startup_validation][gradient_budget]")
{
    {
        const std::string journal = startup_journal(
            mlsl_plain("GD_MLSL", "10", kInnerLbfgs, kRootSkipRel));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_central2("MAXEVAL 10", 4,
                                         "POPULATION 4, NLopt's default",
                                         "MAXEVAL 11 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(count_of(startup_journal(mlsl_plain("GD_MLSL", "11", kInnerLbfgs,
                                                  kRootSkipRel)),
                       kNoStep) == 0u);
    }
    {
        const std::string journal = startup_journal(
            mlsl_plain("GD_MLSL", "9", "    POPULATION 3\n" + kInnerLbfgs,
                       kRootSkipRel));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_central2("MAXEVAL 9", 3, "POPULATION 3",
                                         "MAXEVAL 10 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }

    const std::string analytical =
        "    GRAD_STRATEGY ANALYTICAL\n" + kInnerLbfgs;
    {
        const std::string journal =
            startup_journal(mlsl_plain("GD_MLSL", "7", analytical));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_head("MAXEVAL 7", 4,
                                     "POPULATION 4, NLopt's default") +
                           "evaluates the loss there again (1 evaluation) "
                           "and the gradient there (1 evaluation: one "
                           "analytical gradient, GRAD_STRATEGY ANALYTICAL) "
                           "before it evaluates a point it steps to, which "
                           "needs MAXEVAL 8 or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(count_of(startup_journal(mlsl_plain("GD_MLSL", "8", analytical)),
                       kNoStep) == 0u);
    }

    const std::string hybrid = "    GRAD_STRATEGY HYBRID\n" + kInnerLbfgs;
    {
        const std::string journal =
            startup_journal(mlsl_plain("GD_MLSL", "7", hybrid));
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                           mlsl_head("MAXEVAL 7", 4,
                                     "POPULATION 4, NLopt's default") +
                           "evaluates the loss there again (1 evaluation) "
                           "and the gradient there (at least 1 evaluation: "
                           "the analytical gradient, and the finite "
                           "differences of each parameter it does not cover, "
                           "GRAD_STRATEGY HYBRID) before it evaluates a point "
                           "it steps to, which needs MAXEVAL 8 or more, and "
                           "more for each parameter it differences") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(count_of(startup_journal(mlsl_plain("GD_MLSL", "8", hybrid)),
                       kNoStep) == 0u);
    }
}

namespace {

/* kRoot with a PC_OBSERVABLE: cohesive_energy, which starts its relaxation
 * from the lattice constant the previous evaluation relaxed to
 * (Observable::carried_state), so the run evaluates every point asked for
 * again (HPOTargets::repeat_reuse_blocker). */
const std::string kRootObservable =
    kRoot +
    "PC_OBSERVABLE cohesive_energy --species H --target 8.1 --weight 1.0 "
    "--lattice bcc --hotvar a0 2.5 4.5 3.3\n";

}  // namespace

/* Pins: a run with a PC_OBSERVABLE obtains its gradients under
 * GRAD_STRATEGY ANALYTICAL or HYBRID, never NUMERICAL, so the budget
 * warning of its MLSL is ANALYTICAL's (or HYBRID's), not the NUMERICAL one:
 *   - GD_MLSL with INNER LD_LBFGS and GRAD_STRATEGY NUMERICAL is refused
 *     ("--hpotarget defines PC_OBSERVABLE but OPTIMIZER's
 *     GRAD_STRATEGY=NUMERICAL is set", HPOTargets);
 *   - with no GRAD_STRATEGY, which resolves to ANALYTICAL with a
 *     PC_OBSERVABLE, MAXEVAL 7 warns "... then its first local search, from
 *     one of them, evaluates the loss there again (1 evaluation) and the
 *     gradient there (1 evaluation: one analytical gradient, GRAD_STRATEGY
 *     ANALYTICAL) before it evaluates a point it steps to, which needs
 *     MAXEVAL 8 or more" (5 + 1 + 1 + a step), and MAXEVAL 8 gets none.
 * Before the fix: passes as well (a pin: the warning was right; the docs
 * fixed with it -- the Tadah.MLIP CHANGELOG and the MAXEVAL row of the
 * Configuration File -- gave a PC_OBSERVABLE that keeps state the NUMERICAL
 * bound, MAXEVAL 10 or less in 2-D, and named it beside --skip-rel as what
 * makes a NUMERICAL run evaluate the start again). */
TEST_CASE("startup validation: the gradient budget warning of an MLSL in a "
          "PC_OBSERVABLE run is ANALYTICAL's",
          "[startup_validation][gradient_budget]")
{
    tadah::core::PeriodicTable::initialize();
    {
        const fs::path file = fs::temp_directory_path() /
                              (unique_name("tadah_budget_") + ".hpo");
        {
            std::ofstream os(file);
            REQUIRE(os.good());
            os << mlsl_plain("GD_MLSL", "8",
                             "    GRAD_STRATEGY NUMERICAL\n" + kInnerLbfgs,
                             kRootObservable);
        }
        Context ctx = make_ctx();
        CHECK_THROWS_WITH(HPOTargets(ctx, file.string()),
                          Catch::Contains("--hpotarget defines PC_OBSERVABLE "
                                          "but OPTIMIZER's "
                                          "GRAD_STRATEGY=NUMERICAL is set"));
        std::error_code ec;
        fs::remove(file, ec);
    }

    const std::string journal = startup_journal(
        mlsl_plain("GD_MLSL", "7", kInnerLbfgs, kRootObservable));
    CHECK(count_of(journal,
                   "the STRATEGY-level optimiser NLOPT:GD_MLSL" +
                       mlsl_head("MAXEVAL 7", 4,
                                 "POPULATION 4, NLopt's default") +
                       "evaluates the loss there again (1 evaluation) and the "
                       "gradient there (1 evaluation: one analytical "
                       "gradient, GRAD_STRATEGY ANALYTICAL) before it "
                       "evaluates a point it steps to, which needs MAXEVAL 8 "
                       "or more") == 1u);
    CHECK(count_of(journal, kNoStep) == 1u);
    CHECK(count_of(startup_journal(mlsl_plain("GD_MLSL", "8", kInnerLbfgs,
                                              kRootObservable)),
                   kNoStep) == 0u);
}

namespace {

/* A PRESTAGE run whose refinement is the STRATEGY-level `algo_lines` (LIB,
 * ALGO and its other keys) with MAXEVAL `maxeval`, after a GN_DIRECT
 * exploration of 10 evaluations. */
std::string prestage(const std::string& algo_lines, const std::string& maxeval,
                     const std::string& root = kRoot)
{
    return strategy("    TYPE PRESTAGE\n" + algo_lines + "    MAXEVAL " +
                        maxeval +
                        "\n    EXPLORE\n      LIB NLOPT\n      ALGO GN_DIRECT\n"
                        "      MAXEVAL 10\n    ENDEXPLORE\n",
                    root);
}

/* `strategy_body` under an OPTIMIZER block whose NOISE block sets
 * CALIBRATION `n`. */
std::string with_calibration(const std::string& n,
                             const std::string& strategy_body)
{
    return kRoot + "OPTIMIZER\n  NOISE\n    CALIBRATION " + n +
           "\n  ENDNOISE\n  STRATEGY\n" + strategy_body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

const std::string kLbfgs = "    LIB NLOPT\n    ALGO LD_LBFGS\n";

}  // namespace

/* Pins: the run keeps one record of its usable evaluations (repeat_points.h,
 * HPO_Host::recorded), which NLopt and DLIB:BFGS answer a call from, so a
 * search of theirs that starts at a point the run has evaluated takes the
 * loss there as the run spends it (repeat_use): from the record under
 * NUMERICAL (the centre of the finite differences), evaluated again under
 * ANALYTICAL or HYBRID or without a reuse. Two starts are such points:
 *   - a PRESTAGE refinement's, an exploration candidate (a usable
 *     evaluation): in 2-D under NUMERICAL central differences 0 + 4 + 1
 *     evaluations, "the PRESTAGE refinement DLIB:BFGS: MAXEVAL 4 takes no
 *     step: its gradient search takes the loss at its starting point from
 *     the run's record (no evaluation) and evaluates the gradient there (4
 *     evaluations: ...) ..., which needs MAXEVAL 5 or more", and MAXEVAL 5
 *     gets none; under ANALYTICAL the start is evaluated again, "the
 *     PRESTAGE refinement DLIB:BFGS: MAXEVAL 2 takes no step: its gradient
 *     search evaluates the loss at its starting point again (1 evaluation)
 *     and the gradient there (1 evaluation: one analytical gradient,
 *     GRAD_STRATEGY ANALYTICAL) ..., which needs MAXEVAL 3 or more", and so
 *     under a --skip-rel filter (no reuse), NLOPT:LD_LBFGS MAXEVAL 5
 *     needing 6; an MLSL refinement takes its starting point's loss from
 *     the record too, "MLSL takes the loss at its starting point from the
 *     run's record and evaluates its first 4 samples without a gradient (4
 *     evaluations; ...), then its first local search, from one of them,
 *     takes the loss there from the run's record (no evaluation) ...,
 *     which needs MAXEVAL 9 or more" (MAXEVAL 8; 9 gets none); an AUGLAG
 *     refinement's local search starts at the candidate, so its INNER
 *     LD_LBFGS MAXEVAL 1 reads "(4 evaluations: ...; its local search
 *     starts at a point the run has evaluated, whose loss is taken from the
 *     run's record), so each local search evaluates the gradient at its
 *     starting point and ends there";
 *   - under NOISE CALIBRATION 3 the run's first search starts at the seed
 *     point the calibration evaluated: the PLAIN NLOPT:LD_LBFGS reads "the
 *     STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL 4 takes no step: its
 *     gradient search takes the loss at its starting point from the run's
 *     record ..., which needs MAXEVAL 5 or more", and MAXEVAL 5 gets none;
 *     CALIBRATION 0, which runs no calibration, keeps "evaluates the loss at
 *     a starting point", needing 6 at MAXEVAL 5 (a pin); of two
 *     MULTISTART starts the first begins at the seed point and the second
 *     elsewhere, so BUDGET_PER_START 5 is warned for "each MULTISTART start
 *     after the first", needing 6 (the first steps: the next test case).
 * The refinement from INIT points that a PRESTAGE whose exploration has no
 * usable candidate falls back on starts at points the run need not have
 * evaluated, which the warning, before any evaluation, cannot know.
 * Before the fix: every such start was counted as evaluated ("evaluates the
 * loss at a starting point (1 evaluation)"): the DLIB:BFGS refinement
 * MAXEVAL 4 was warned about needing 6 and MAXEVAL 5, which steps at its
 * fifth evaluation, was warned about too; the ANALYTICAL and --skip-rel
 * texts said "at a starting point"; the MLSL refinement needed 10, the
 * AUGLAG INNER text was "(1 + 4 evaluations: ...), so each local search
 * evaluates its starting point", and the calibrated PLAIN run MAXEVAL 4 and
 * 5 needed 6: those checks fail; the CALIBRATION 0 check passes as well.
 * Before the first MULTISTART start was named apart:
 * the MULTISTART check fails, its WARNING being "each MULTISTART start
 * NLOPT:LD_LBFGS: BUDGET_PER_START 5 ...", which claimed the first start,
 * which steps. */
TEST_CASE("startup validation: the gradient budget warning takes a start the "
          "run has evaluated from the run's record",
          "[startup_validation][gradient_budget][repeat_points]")
{
    const std::string dlib = "    LIB DLIB\n    ALGO BFGS\n";
    CHECK(count_of(startup_journal(prestage(dlib, "4")),
                   "the PRESTAGE refinement DLIB:BFGS" +
                       record_central2("MAXEVAL 4", "MAXEVAL 5 or more")) ==
          1u);
    CHECK(count_of(startup_journal(prestage(dlib, "5")), kNoStep) == 0u);

    const std::string analytical = dlib + "    GRAD_STRATEGY ANALYTICAL\n";
    CHECK(count_of(startup_journal(prestage(analytical, "2")),
                   "the PRESTAGE refinement DLIB:BFGS: MAXEVAL 2 takes no "
                   "step: its gradient search evaluates the loss at its "
                   "starting point again (1 evaluation) and the gradient "
                   "there (1 evaluation: one analytical gradient, "
                   "GRAD_STRATEGY ANALYTICAL) before it evaluates a point it "
                   "steps to, which needs MAXEVAL 3 or more") == 1u);
    CHECK(count_of(startup_journal(prestage(analytical, "3")), kNoStep) == 0u);

    CHECK(count_of(startup_journal(prestage(kLbfgs, "5", kRootSkipRel)),
                   "the PRESTAGE refinement NLOPT:LD_LBFGS: MAXEVAL 5 takes "
                   "no step: its gradient search evaluates the loss at its "
                   "starting point again (1 evaluation) and the gradient "
                   "there (4 evaluations: central differences in 2 "
                   "dimensions, GRAD_STRATEGY NUMERICAL) before it evaluates "
                   "a point it steps to, which needs MAXEVAL 6 or more") ==
          1u);

    const std::string mlsl =
        "    LIB NLOPT\n    ALGO GD_MLSL\n" + kInnerLbfgs;
    CHECK(count_of(startup_journal(prestage(mlsl, "8")),
                   "the PRESTAGE refinement NLOPT:GD_MLSL: MAXEVAL 8 takes no "
                   "step of a local search: MLSL takes the loss at its "
                   "starting point from the run's record and evaluates its "
                   "first 4 samples without a gradient (4 evaluations; "
                   "POPULATION 4, NLopt's default), then its first local "
                   "search, from one of them, takes the loss there from the "
                   "run's record (no evaluation) and evaluates the gradient "
                   "there (4 evaluations: central differences in 2 "
                   "dimensions, GRAD_STRATEGY NUMERICAL) before it evaluates "
                   "a point it steps to, which needs MAXEVAL 9 or more") ==
          1u);
    CHECK(count_of(startup_journal(prestage(mlsl, "9")), kNoStep) == 0u);

    const std::string auglag =
        "    LIB NLOPT\n    ALGO AUGLAG\n    INNER\n      LIB NLOPT\n"
        "      ALGO LD_LBFGS\n      MAXEVAL 1\n    ENDINNER\n";
    CHECK(count_of(startup_journal(prestage(auglag, "100")),
                   "the INNER block of the PRESTAGE refinement NLOPT:LD_LBFGS: "
                   "MAXEVAL 1 takes no step: NLopt counts an INNER block's "
                   "MAXEVAL in its calls, each the loss and the gradient at a "
                   "point (4 evaluations: central differences in 2 "
                   "dimensions, GRAD_STRATEGY NUMERICAL; its local search "
                   "starts at a point the run has evaluated, whose loss is "
                   "taken from the run's record), so each local search "
                   "evaluates the gradient at its starting point and ends "
                   "there, which needs MAXEVAL 2 or more") == 1u);

    const std::string plain_lbfgs = "    TYPE PLAIN\n" + kLbfgs;
    CHECK(count_of(startup_journal(with_calibration(
                       "3", plain_lbfgs + "    MAXEVAL 4\n")),
                   "the STRATEGY-level optimiser NLOPT:LD_LBFGS" +
                       record_central2("MAXEVAL 4", "MAXEVAL 5 or more")) ==
          1u);
    CHECK(count_of(startup_journal(with_calibration(
                       "3", plain_lbfgs + "    MAXEVAL 5\n")),
                   kNoStep) == 0u);
    CHECK(count_of(startup_journal(with_calibration(
                       "0", plain_lbfgs + "    MAXEVAL 5\n")),
                   "the STRATEGY-level optimiser NLOPT:LD_LBFGS" +
                       central2("MAXEVAL 5", "MAXEVAL 6 or more")) == 1u);
    CHECK(count_of(startup_journal(
                       kRoot +
                       "OPTIMIZER\n  NOISE\n    CALIBRATION 3\n  ENDNOISE\n"
                       "  INIT\n    METHOD LHS 2\n    SEED 3\n  ENDINIT\n"
                       "  STRATEGY\n    TYPE MULTISTART\n" + kLbfgs +
                       "    N_STARTS 2\n    MAXEVAL 100\n"
                       "    BUDGET_PER_START 5\n  ENDSTRATEGY\nENDOPTIMIZER\n"),
                   "each MULTISTART start after the first NLOPT:LD_LBFGS" +
                       central2("BUDGET_PER_START 5",
                                "BUDGET_PER_START 6 or more")) == 1u);
}

namespace {

/* The tail of a 2-D NUMERICAL central warning of a gradient search that
 * does not start at the point it is given (NLopt's StoGO samples its own
 * box, global.cc): it evaluates a point of its own and the gradient there,
 * whatever the run has recorded at its starting point. */
std::string own_central2(const std::string& budget, const std::string& needs)
{
    return ": " + budget + " takes no step: its gradient search, which does "
           "not start at the point it is given, evaluates the loss at a point "
           "of its own (1 evaluation) and the gradient there (4 evaluations: "
           "central differences in 2 dimensions, GRAD_STRATEGY NUMERICAL) "
           "before it evaluates a point it steps to, which needs " + needs;
}

}  // namespace

/* Pins: NLopt's StoGO (GD_STOGO, GD_STOGO_RAND) does not start at the point
 * it is given: stogo_minimize never reads x0, and its first trial is the
 * centre of its box (global.cc, FillRegular: the midpoint is the last trial
 * added and the first taken; GD_STOGO_RAND with NLopt's default population
 * as well), where it takes its first gradient. So a StoGO search starting at
 * a point the run has evaluated -- the seed point of NOISE CALIBRATION --
 * takes no loss from the run's record: in 2-D under NUMERICAL central
 * differences it spends 1 + 4 evaluations before its first step, as from
 * any start:
 *   - the PLAIN GD_STOGO and GD_STOGO_RAND under NOISE CALIBRATION 3 at
 *     MAXEVAL 5, and without a calibration: "the STRATEGY-level optimiser
 *     NLOPT:GD_STOGO: MAXEVAL 5 takes no step: its gradient search, which
 *     does not start at the point it is given, evaluates the loss at a
 *     point of its own (1 evaluation) and the gradient there (4
 *     evaluations: ...) ..., which needs MAXEVAL 6 or more"; MAXEVAL 6 gets
 *     none;
 *   - the PRESTAGE refinement on GD_STOGO or GD_STOGO_RAND is refused at
 *     startup (it would never use the exploration's candidate;
 *     test_startup_validation_strategy.cpp has the whole text): "STRATEGY
 *     TYPE PRESTAGE cannot refine with NLOPT:GD_STOGO: it does not start at
 *     the point it is given ...";
 *   - StoGO is a global optimiser, so an INNER block does not take it (no
 *     nested global search): the PLAIN
 *     NLOPT:AUGLAG with an INNER GD_STOGO under NOISE CALIBRATION 3, the
 *     same AUGLAG as the PRESTAGE refinement and GD_MLSL with an INNER
 *     GD_STOGO are refused at startup, at the ENDINNER line: "the INNER
 *     block of the STRATEGY-level optimiser: NLOPT:AUGLAG cannot run
 *     NLOPT:GD_STOGO as its INNER search: an INNER search is a local
 *     search, ..." (test_startup_validation_strategy.cpp has the whole
 *     text).
 * Before the no-nested-global fix: the PLAIN AUGLAG and the GD_MLSL with
 * an INNER GD_STOGO were accepted, and these legs pinned two WARNINGs of
 * each: the AUGLAG's at MAXEVAL 5 as the PLAIN GD_STOGO's, and its INNER
 * GD_STOGO MAXEVAL 1's "(1 + 4 evaluations: ...), so each local search,
 * which does not start at the point it is given, evaluates a point of its
 * own and ends there"; GD_MLSL's at MAXEVAL 10 "MLSL evaluates its starting
 * point and its first 4 samples without a gradient (5 evaluations; ...),
 * then its first local search, which does not start at the point MLSL
 * gives it, evaluates the loss at a point of its own (1 evaluation) and
 * the gradient there (4 evaluations: ...) ..., which needs MAXEVAL 11 or
 * more", and the INNER text above; the AUGLAG refinement was refused with
 * "the INNER block of the PRESTAGE refinement: NLOPT:AUGLAG runs its INNER
 * search NLOPT:GD_STOGO from the point it is given ...": the three refusal
 * checks fail.
 * Until the refinement was refused the
 * PRESTAGE refinement legs pinned the warning ("the PRESTAGE refinement
 * NLOPT:GD_STOGO: MAXEVAL 5 takes no step: ...", "the PRESTAGE refinement
 * NLOPT:AUGLAG ...", "the INNER block of the PRESTAGE refinement
 * NLOPT:GD_STOGO: ..."), which the PLAIN legs under NOISE CALIBRATION 3 now
 * pin; GD_STOGO_RAND was not run as PLAIN, nor the PLAIN legs at MAXEVAL 6.
 * Before the refusal: the refinements were accepted (the refusal checks
 * fail); the warning legs pass as well (pins).
 * Before the fix of the warning: start_is_recorded took every NLopt
 * search's start as its first call, so the refinement and the calibrated
 * PLAIN run at MAXEVAL 5, which take no step, were not warned (they "took
 * the loss at its starting point from the run's record", needing 5; the
 * runs evaluated the box centre and its stencil, no step), the
 * uncalibrated PLAIN text said "evaluates the loss at a starting point",
 * the AUGLAG INNER text said its local search
 * starts at a point the run has evaluated, and the GD_MLSL one at MAXEVAL
 * 10 was not warned (needing 10) and its INNER text said MLSL's local
 * searches start at points it evaluated. */
TEST_CASE("startup validation: the gradient budget warning of a search that "
          "does not start at the point it is given counts a point of its own",
          "[startup_validation][gradient_budget][stogo]")
{
    for (const char* algo : {"GD_STOGO", "GD_STOGO_RAND"}) {
        INFO(algo);
        const std::string lines =
            std::string("    LIB NLOPT\n    ALGO ") + algo + "\n";
        for (const char* n : {"3", "0"}) {
            INFO("NOISE CALIBRATION " << n);
            const std::string journal = startup_journal(with_calibration(
                n, "    TYPE PLAIN\n" + lines + "    MAXEVAL 5\n"));
            CHECK(count_of(journal,
                           std::string("the STRATEGY-level optimiser NLOPT:") +
                               algo +
                               own_central2("MAXEVAL 5", "MAXEVAL 6 or more")) ==
                  1u);
            CHECK(count_of(journal, kNoStep) == 1u);
            CHECK(count_of(startup_journal(with_calibration(
                               n, "    TYPE PLAIN\n" + lines +
                                      "    MAXEVAL 6\n")),
                           kNoStep) == 0u);
        }
        CHECK_THAT(startup_refusal(prestage(lines, "5")),
                   Catch::StartsWith("Error on line ") &&
                       Catch::Contains(
                           std::string("STRATEGY TYPE PRESTAGE cannot refine "
                                       "with NLOPT:") +
                           algo + ": it does not start at the point it is "
                                  "given"));
    }

    {
        const std::string auglag =
            "    LIB NLOPT\n    ALGO AUGLAG\n    INNER\n      LIB NLOPT\n"
            "      ALGO GD_STOGO\n      MAXEVAL 1\n    ENDINNER\n";
        CHECK_THAT(startup_refusal(with_calibration(
                       "3", "    TYPE PLAIN\n" + auglag + "    MAXEVAL 5\n")),
                   Catch::StartsWith("Error on line ") &&
                       Catch::Contains("the INNER block of the STRATEGY-level "
                                       "optimiser: NLOPT:AUGLAG cannot run "
                                       "NLOPT:GD_STOGO as its INNER search: "
                                       "an INNER search is a local search") &&
                       Catch::EndsWith("ENDINNER"));
        CHECK_THAT(startup_refusal(prestage(auglag, "5")),
                   Catch::StartsWith("Error on line ") &&
                       Catch::Contains("the INNER block of the PRESTAGE "
                                       "refinement: NLOPT:AUGLAG cannot run "
                                       "NLOPT:GD_STOGO as its INNER search: "
                                       "an INNER search is a local search") &&
                       Catch::EndsWith("ENDINNER"));
    }
    CHECK_THAT(startup_refusal(mlsl_plain(
                   "GD_MLSL", "10",
                   "    INNER\n      LIB NLOPT\n      ALGO GD_STOGO\n"
                   "      MAXEVAL 1\n    ENDINNER\n")),
               Catch::StartsWith("Error on line ") &&
                   Catch::Contains("the INNER block of the STRATEGY-level "
                                   "optimiser: NLOPT:GD_MLSL cannot run "
                                   "NLOPT:GD_STOGO as its INNER search: an "
                                   "INNER search is a local search, which "
                                   "NLOPT:GD_MLSL starts at the points it "
                                   "samples in the OPTIM box") &&
                   Catch::EndsWith("ENDINNER"));
}

namespace {

/* MULTISTART over INIT LHS 2 (SEED 3) under NOISE CALIBRATION `n`: the
 * calibration evaluates the first INIT point, the first start's. */
std::string calibrated_multistart(const std::string& n,
                                  const std::string& algo_lines,
                                  const std::string& budget_per_start)
{
    return kRoot + "OPTIMIZER\n  NOISE\n    CALIBRATION " + n +
           "\n  ENDNOISE\n  INIT\n    METHOD LHS 2\n    SEED 3\n  ENDINIT\n"
           "  STRATEGY\n    TYPE MULTISTART\n" + algo_lines +
           "    N_STARTS 2\n    MAXEVAL 100\n    BUDGET_PER_START " +
           budget_per_start + "\n  ENDSTRATEGY\nENDOPTIMIZER\n";
}

}  // namespace

/* Pins: under NOISE CALIBRATION the first MULTISTART start begins at the
 * seed point the calibration evaluated (the first INIT point), and the run
 * answers its loss from the run's record; the later starts begin at INIT
 * points the run has not evaluated. So in 2-D under NUMERICAL central
 * differences the first start needs 0 + 4 + 1 evaluations for a step and
 * every later one 1 + 4 + 1, and the warning names each where they differ:
 *   - LD_LBFGS, BUDGET_PER_START 5: one WARNING, "each MULTISTART start
 *     after the first NLOPT:LD_LBFGS: BUDGET_PER_START 5 takes no step: its
 *     gradient search evaluates the loss at a starting point (1 evaluation)
 *     ..., which needs BUDGET_PER_START 6 or more"; the first start steps;
 *   - BUDGET_PER_START 4: that one (needing 6) and "the first MULTISTART
 *     start NLOPT:LD_LBFGS: BUDGET_PER_START 4 takes no step: its gradient
 *     search takes the loss at its starting point from the run's record (no
 *     evaluation) ..., which needs BUDGET_PER_START 5 or more";
 *   - BUDGET_PER_START 6: none;
 *   - DLIB:BFGS, which answers a call from the record too, the same at 5;
 *   - under GRAD_STRATEGY ANALYTICAL, where the first start's loss is
 *     evaluated again, every start costs the same: one WARNING for "each
 *     MULTISTART start" at BUDGET_PER_START 2, needing 3;
 *   - CALIBRATION 0 runs no calibration: one WARNING for "each MULTISTART
 *     start" at 5, needing 6.
 * Before the fix: start_is_recorded counted the calibration seed only for a
 * MULTISTART that is a single run, so every start was warned at
 * BUDGET_PER_START 5 ("each MULTISTART start NLOPT:LD_LBFGS:
 * BUDGET_PER_START 5 takes no step ..., which needs BUDGET_PER_START 6"),
 * although the first one steps (a run before the fix: its
 * stencil, then a step, "start 1/2 done: 3 eval(s) consumed, improved
 * global best" in 1-D): the checks of the CALIBRATION 3 LD_LBFGS and
 * DLIB:BFGS cases at 5 and 4 fail; the others pass as well (pins). */
TEST_CASE("startup validation: the gradient budget warning names the first "
          "MULTISTART start under NOISE CALIBRATION apart",
          "[startup_validation][gradient_budget][repeat_points]")
{
    const std::string later = "each MULTISTART start after the first ";
    const std::string first = "the first MULTISTART start ";
    for (const auto& [name, lines] :
         std::vector<std::pair<std::string, std::string>>{
             {"NLOPT:LD_LBFGS", kLbfgs},
             {"DLIB:BFGS", "    LIB DLIB\n    ALGO BFGS\n"}}) {
        INFO(name);
        {
            const std::string journal =
                startup_journal(calibrated_multistart("3", lines, "5"));
            CHECK(count_of(journal,
                           later + name +
                               central2("BUDGET_PER_START 5",
                                        "BUDGET_PER_START 6 or more")) == 1u);
            CHECK(count_of(journal, first) == 0u);
            CHECK(count_of(journal, kNoStep) == 1u);
        }
        {
            const std::string journal =
                startup_journal(calibrated_multistart("3", lines, "4"));
            CHECK(count_of(journal,
                           first + name +
                               record_central2("BUDGET_PER_START 4",
                                               "BUDGET_PER_START 5 or more")) ==
                  1u);
            CHECK(count_of(journal,
                           later + name +
                               central2("BUDGET_PER_START 4",
                                        "BUDGET_PER_START 6 or more")) == 1u);
            CHECK(count_of(journal, kNoStep) == 2u);
        }
        CHECK(count_of(startup_journal(calibrated_multistart("3", lines, "6")),
                       kNoStep) == 0u);
    }
    {
        const std::string journal = startup_journal(calibrated_multistart(
            "3", kLbfgs + "    GRAD_STRATEGY ANALYTICAL\n", "2"));
        CHECK(count_of(journal,
                       "each MULTISTART start NLOPT:LD_LBFGS: BUDGET_PER_START "
                       "2 takes no step: its gradient search evaluates the "
                       "loss at a starting point (1 evaluation) and the "
                       "gradient there (1 evaluation: one analytical "
                       "gradient, GRAD_STRATEGY ANALYTICAL) before it "
                       "evaluates a point it steps to, which needs "
                       "BUDGET_PER_START 3 or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
    {
        const std::string journal =
            startup_journal(calibrated_multistart("0", kLbfgs, "5"));
        CHECK(count_of(journal,
                       "each MULTISTART start NLOPT:LD_LBFGS" +
                           central2("BUDGET_PER_START 5",
                                    "BUDGET_PER_START 6 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }
}

/* Pins: OptimizerCaps::starts_at_x0 records which backends ask first for the
 * point they are given (nlopt 2.10 sources): not NLopt's StoGO (stogo.cc
 * never reads x0), DIRECT in each form (cdirect.c starts at the centre of
 * the box; DIRect.c, GN_ORIG_DIRECT*) or AGS (ags.cc), which return x as
 * their result only; the local NLopt algorithms, MLSL (mlsl.c evaluates x0
 * first), AUGLAG, CRS2, ISRES and ESCH (x0 is the first member of their
 * population) and DLIB:BFGS do. The gradient budget warning reads it
 * (start_is_recorded, backend_rules.cpp).
 * Before the fix: does not compile (the flag is new with the StoGO budget
 * fix; this pin of its values comes with it). */
TEST_CASE("optimizer caps: starts_at_x0 marks the backends whose first call "
          "is at the point they are given",
          "[optimizer_caps][gradient_budget][stogo]")
{
    using tadah::hpo::optimizer_caps;
    for (const char* algo :
         /* GN_DIRECT_NOSCAL and GN_DIRECT_L_RAND_NOSCAL, which NLopt runs
          * as GN_DIRECT_L_NOSCAL, are refused (they were in this list). */
         {"GD_STOGO", "GD_STOGO_RAND", "GN_DIRECT", "GN_DIRECT_L",
          "GN_DIRECT_L_RAND", "GN_DIRECT_L_NOSCAL", "GN_ORIG_DIRECT",
          "GN_ORIG_DIRECT_L", "GN_AGS"}) {
        INFO(algo);
        CHECK_FALSE(optimizer_caps("NLOPT", algo).starts_at_x0);
    }
    for (const char* algo :
         {"LD_LBFGS", "LD_MMA", "LD_SLSQP", "LN_SBPLX", "LN_BOBYQA",
          "GD_MLSL", "G_MLSL_LDS", "AUGLAG", "LD_AUGLAG", "GN_CRS2_LM",
          "GN_ISRES", "GN_ESCH"}) {
        INFO(algo);
        CHECK(optimizer_caps("NLOPT", algo).starts_at_x0);
    }
    CHECK(optimizer_caps("DLIB", "BFGS").starts_at_x0);
}

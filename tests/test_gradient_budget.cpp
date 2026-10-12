// test_gradient_budget.cpp
//
// A gradient search evaluates the loss at its starting point and the
// gradient there before it evaluates any point it steps to. A MAXEVAL that
// cannot pay for both and one more evaluation ends the search where it
// started: with GRAD_STRATEGY NUMERICAL and central differences in n
// dimensions the start costs 1 + 2n evaluations, so MAXEVAL 2n + 1 or less
// takes no step (LD_LBFGS, DLIB:BFGS and CERES:LBFGS in 2-D: MAXEVAL 4 and 5
// evaluate the start and its stencil, MAXEVAL 6 a first step). Such a budget
// can be deliberate (a probe of the start), so validate_hpo_setup warns, for
// every phase that has a MAXEVAL of its own, naming what the gradient costs
// and the budget that takes a step.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
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
 * or more", for NLOPT:LD_LBFGS, DLIB:BFGS and CERES:LBFGS; MAXEVAL 6 gets
 * none (tmp/repro lbfgs6, dlib6, ceres6: the sixth evaluation is the first
 * step).
 * Before the fix: no WARNING for any of them (the search ended
 * MAXEVAL_REACHED at its starting point: tmp/repro a2_lbfgs3, a2_dlib3). */
TEST_CASE("startup validation: a gradient search whose MAXEVAL cannot pay for "
          "one step is warned about",
          "[startup_validation][gradient_budget]")
{
    for (const auto& [lib, algo] : std::vector<std::pair<std::string,
                                                         std::string>>{
             {"NLOPT", "LD_LBFGS"}, {"DLIB", "BFGS"}, {"CERES", "LBFGS"}}) {
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
 *   - the PRESTAGE refinement ("the PRESTAGE refinement NLOPT:LD_LBFGS:
 *     MAXEVAL 4 takes no step"), and not the derivative-free exploration;
 *     with TOPK 2 each candidate's share ("MAXEVAL 8 over 2 TOPK
 *     candidates, 4 each, takes no step: ... which needs 6 evaluations or
 *     more per candidate");
 *   - an INNER block, whose MAXEVAL NLopt counts in its own calls, each the
 *     loss and the gradient at a point: INNER LD_LBFGS MAXEVAL 1 under
 *     GD_MLSL, whose local searches start at points MLSL evaluated, whose
 *     loss the search's record answers (every evaluation of this run is a
 *     function of its point; NUMERICAL takes it as the centre of the
 *     finite differences, repeat_use, repeat_points.h): "the INNER block of
 *     the STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL 1 takes no step:
 *     NLopt counts an INNER block's MAXEVAL in its calls, each the loss and
 *     the gradient at a point (4 evaluations: central differences in 2
 *     dimensions, GRAD_STRATEGY NUMERICAL; MLSL's local searches start at
 *     points it evaluated, whose loss is taken from the search's record),
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
 * the warning was added none was given). */
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
                                    central2("MAXEVAL 4",
                                             "MAXEVAL 6 or more")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }

    /* With TOPK 2 the refinement runs once from each candidate, each with
     * its share of MAXEVAL (plan_multistart, ceil division). */
    {
        const std::string journal = startup_journal(strategy(
            "    TYPE PRESTAGE\n    TOPK 2\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
            "    MAXEVAL 8\n    EXPLORE\n      LIB NLOPT\n"
            "      ALGO GN_DIRECT\n      MAXEVAL 10\n    ENDEXPLORE\n"));
        CHECK(count_of(journal,
                       "the PRESTAGE refinement NLOPT:LD_LBFGS" +
                           central2("MAXEVAL 8 over 2 TOPK candidates, 4 "
                                    "each,",
                                    "6 evaluations or more per "
                                    "candidate")) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
    }

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
                       "taken from the search's record), so each local "
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
           "takes the loss there from the search's record (no evaluation) "
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
 * (lanes/D/tmp/br/runs_before/m9_plain: the start, 4 samples, the 4-point
 * stencil of the local start, "Repeated points reused : 1"; m10_plain: the
 * 10th evaluation is the first step). The WARNING says so:
 *   - GD_MLSL with INNER LD_LBFGS, MAXEVAL 5 and 9: "the STRATEGY-level
 *     optimiser NLOPT:GD_MLSL: MAXEVAL 9 takes no step of a local search:
 *     MLSL evaluates its starting point and its first 4 samples without a
 *     gradient (5 evaluations; POPULATION 4, NLopt's default), then its
 *     first local search, from one of them, takes the loss there from the
 *     search's record (no evaluation) and evaluates the gradient there (4
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
 *     (lanes/D/tmp/br/runs_before/m10_skiprel: the local start is evaluated
 *     again, eval 6, then its stencil; m11_skiprel: the 11th evaluation is
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

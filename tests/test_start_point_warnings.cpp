// test_start_point_warnings.cpp
//
// NLopt's StoGO, DIRECT and AGS never read the point HPO hands them
// (OptimizerCaps::starts_at_x0; their first call is the centre of the OPTIM
// box, or a point of AGS's curve near it). A PRESTAGE refinement and a
// MULTISTART start on one are refused at startup (strategy_rules.h); where
// such a search is accepted -- the STRATEGY-level optimiser under PLAIN, the
// PRESTAGE exploration -- two settings that are about its starting points
// have no effect on it, and validate_hpo_setup warns, once each, naming the
// line of the block:
//  - a RESTART block, when the search also draws no random numbers before
//    it asks for a point of its own (StoGO, DIRECT and AGS; GN_DIRECT_L_RAND
//    in one dimension; GD_STOGO_RAND below POPULATION 2n + 1 until its
//    first local search, from the box centre, ends): every round asks for
//    round 0's points again, whatever points it starts from (the run's
//    record answers their losses; StoGO's gradients, which a round forms
//    itself, are evaluated again), so no round after round 0 searches anew;
//  - a RESTART block with METHOD lines on such a search that draws random
//    numbers (GN_DIRECT_L_RAND in two or more dimensions, GD_STOGO_RAND at
//    POPULATION 2n + 1 or more): no round evaluates the block's points;
//  - an INIT block with a METHOD: the search never evaluates its points.
// A search that starts at its point gets no RESTART warning; an INIT block
// without a METHOD (its SEED only), a RESTART block without METHOD lines on
// a search that draws random numbers and an inactive RESTART block get
// none.
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
                    (unique_name("tadah_start_journal_") + ".log");
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

/* Lines 1-3: two OPTIM dimensions and a loss whose every evaluation is a
 * function of its point (the run answers a point it has evaluated from its
 * record); OPTIMIZER is line 4. */
const std::string kRoot =
    "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\nPC_ERMSE 0 1\n";
/* The same with a --skip-rel filter, which compares with the best value so
 * far: two evaluations of one point can differ, and a point asked for again
 * is evaluated again. */
const std::string kRootSkipRel =
    "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
    "PC_ERMSE 0 1 --skip-rel 5\n";
/* One OPTIM dimension (lines 1-2): OPTIMIZER is line 3. */
const std::string kRoot1 = "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\n";

/* What validate_hpo_setup() writes for `config` -- the journal and
 * standard error of the parse and of the check -- which must accept it. */
struct Startup {
    std::string journal;
    std::string err;
};
Startup startup(const std::string& config)
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_start_") + ".hpo");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << config;
    }
    Context ctx = make_ctx();
    Startup out;
    {
        JournalCapture j;
        INFO("config:\n" << config);
        std::ostringstream err;
        std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
        try {
            HPOTargets trg(ctx, file.string());
            tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
        } catch (...) {
            std::cerr.rdbuf(old);
            std::error_code ec;
            fs::remove(file, ec);
            throw;
        }
        std::cerr.rdbuf(old);
        out.err = err.str();
        out.journal = j.text();
    }
    std::error_code ec;
    fs::remove(file, ec);
    return out;
}

std::size_t count_of(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto p = text.find(needle); p != std::string::npos;
         p = text.find(needle, p + needle.size()))
        ++n;
    return n;
}

/* A PLAIN run of NLOPT:`algo` (MAXEVAL 20) under an active RESTART block,
 * which opens on line 10 (line 9 with kRoot1), with METHOD RANDOM 1 unless
 * `method` is false; `extra` is STRATEGY lines after the block. */
std::string plain_restart(const std::string& algo,
                          const std::string& root = kRoot,
                          const std::string& max = "1",
                          const std::string& extra = "",
                          bool method = true)
{
    return root +
           "OPTIMIZER\n"
           "  STRATEGY\n"
           "    TYPE PLAIN\n"
           "    LIB NLOPT\n"
           "    ALGO " + algo + "\n"
           "    MAXEVAL 20\n"
           "    RESTART\n"
           "      AFTER_STAGNATION 3\n"
           "      MAX " + max + "\n" +
           (method ? "      METHOD RANDOM 1\n" : "") +
           "      SEED 5\n"
           "    ENDRESTART\n" + extra +
           "  ENDSTRATEGY\n"
           "ENDOPTIMIZER\n";
}

/* The clause that says where `algo` starts instead (the registry's). */
std::string note(const std::string& algo)
{
    return tadah::hpo::optimizer_caps("NLOPT", algo).start_note;
}

const std::string kAnew =
    " For rounds that search anew, use an optimiser that starts at the point "
    "it is given.";

/* The RESTART warning of a PLAIN run whose STRATEGY-level optimiser is
 * NLOPT:`algo` (`draws`: how it draws no random numbers), RESTART on line
 * `line`, the run answering repeats from its record (a search that
 * obtains no gradient). */
std::string plain_restart_warning(const std::string& algo, int line = 10,
                                  const std::string& draws =
                                      " and draws no random numbers")
{
    return "the RESTART block on line " + std::to_string(line) +
           ": the STRATEGY-level optimiser NLOPT:" + algo +
           " does not start at the point it is given (" + note(algo) + ")" +
           draws +
           ", so every RESTART round, whatever points it starts from, asks "
           "for round 0's points again, which the run's record answers: no "
           "round after round 0 evaluates a new point, and the RESTART "
           "block's only effect is to end the run once round 0 has had no "
           "new best in AFTER_STAGNATION 3 evaluations." + kAnew;
}

/* What a round of a gradient search (StoGO) asks for at round 0's points
 * under GRAD_STRATEGY NUMERICAL, the default: the record answers the
 * losses, the stencils are evaluated again. */
const std::string kStencils =
    ", and their gradients: the run's record answers their losses, but a "
    "round forms its gradients itself and evaluates their finite-difference "
    "stencils again (GRAD_STRATEGY NUMERICAL)";

/* The RESTART warning of a PLAIN run of NLOPT:`algo`, a start-ignoring
 * search that draws random numbers, whose RESTART block (line `line`) has
 * METHOD lines. */
std::string method_lines_warning(const std::string& algo, int line = 10)
{
    return "the RESTART block on line " + std::to_string(line) +
           ": the STRATEGY-level optimiser NLOPT:" + algo +
           " does not start at the point it is given (" + note(algo) +
           "), so no RESTART round evaluates the points of the block's "
           "METHOD lines: a round's search differs from round 0's only in "
           "the random numbers it draws. To start each round at the block's "
           "points, use an optimiser that starts at the point it is given.";
}

}  // namespace

/* Pins: under STRATEGY TYPE PLAIN with an active RESTART block (line 10),
 * each NLopt search that does not start at the point it is given, draws
 * no random numbers and obtains no gradient -- GN_DIRECT, GN_DIRECT_L, the
 * three _NOSCAL forms (NLopt 2.10 runs all three as DIRECT-L,
 * GN_DIRECT_L_RAND_NOSCAL included), GN_ORIG_DIRECT, GN_ORIG_DIRECT_L,
 * GN_AGS -- gets exactly one startup WARNING, in the journal and on
 * standard error ("[HPO] WARNING: "): "the RESTART block on line 10: the
 * STRATEGY-level optimiser NLOPT:GN_DIRECT_L does not start at the point
 * it is given (DIRECT evaluates the centre of the OPTIM box first and then
 * divides the whole box; NLopt never reads that point) and draws no random
 * numbers, so every RESTART round, whatever points it starts from, asks for
 * round 0's points again, which the run's record answers: no round after
 * round 0 evaluates a new point, and the RESTART block's only effect is to
 * end the run once round 0 has had no new best in AFTER_STAGNATION 3
 * evaluations. For rounds that search anew, use an optimiser that starts
 * at the point it is given." (each with its own start note). GN_DIRECT_L_RAND
 * in one OPTIM dimension (RESTART on line 9) gets it with "and, in one
 * OPTIM dimension, draws no random numbers (it draws them only to choose
 * between equally long sides of a box)". With a --skip-rel filter, which
 * makes the run evaluate a repeated point again, the warning says "asks
 * for round 0's points again as long as their losses are those of round 0
 * (the run evaluates them again: PC_ERMSE --skip-rel: ...)". An AUGLAG
 * whose INNER search is GN_DIRECT_L names both ("NLOPT:LN_AUGLAG runs its
 * INNER search NLOPT:GN_DIRECT_L from the point it is given, and
 * NLOPT:GN_DIRECT_L does not start ..."); a PRESTAGE exploration on
 * GN_DIRECT_L gets the exploration's text (RESTART on line 15). No warning:
 * LN_SBPLX and GN_CRS2_LM, which start at their point (a RESTART of MAX
 * 0 is refused at its line). The gradient searches (StoGO), the RAND forms in two
 * dimensions and the METHOD lines of a RESTART block are the next case's.
 * Before the fix: none of these warnings was given (a RESTART run: round 1
 * asked for round 0's points, answered from the record, without a word at
 * startup): every count is 0. */
TEST_CASE("startup validation: RESTART on a search that does not start at "
          "its point and draws no random numbers warns that its rounds "
          "repeat round 0",
          "[startup][restart][start_point]")
{
    SECTION("each search that draws no random numbers")
    {
        for (const char* algo :
             /* GN_DIRECT_NOSCAL and GN_DIRECT_L_RAND_NOSCAL are refused
              * (they were in this list). */
             {"GN_DIRECT", "GN_DIRECT_L", "GN_DIRECT_L_NOSCAL",
              "GN_ORIG_DIRECT", "GN_ORIG_DIRECT_L", "GN_AGS"}) {
            INFO("NLOPT:" << algo);
            const Startup s = startup(plain_restart(algo));
            INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
            const std::string w = plain_restart_warning(algo);
            CHECK(count_of(s.journal, w) == 1);
            CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
    }

    SECTION("GN_DIRECT_L_RAND draws no random numbers in one dimension")
    {
        const Startup one = startup(plain_restart("GN_DIRECT_L_RAND", kRoot1));
        INFO("journal:\n" << one.journal);
        CHECK(count_of(one.journal,
                       plain_restart_warning(
                           "GN_DIRECT_L_RAND", 9,
                           " and, in one OPTIM dimension, draws no random "
                           "numbers (it draws them only to choose between "
                           "equally long sides of a box)")) == 1);
        CHECK(count_of(one.journal, "the RESTART block on line") == 1);
    }

    SECTION("a run that evaluates a repeated point again")
    {
        const Startup s = startup(plain_restart("GN_DIRECT_L", kRootSkipRel));
        INFO("journal:\n" << s.journal);
        CHECK(count_of(
                  s.journal,
                  "the RESTART block on line 10: the STRATEGY-level optimiser "
                  "NLOPT:GN_DIRECT_L does not start at the point it is given "
                  "(" + note("GN_DIRECT_L") + ") and draws no random numbers, "
                  "so every RESTART round, whatever points it starts from, "
                  "asks for round 0's points again as long as their losses "
                  "are those of round 0 (the run evaluates them again: "
                  "PC_ERMSE --skip-rel: the filter compares with the best "
                  "value so far, which a later evaluation can lower)." +
                      kAnew) == 1);
    }

    /* An AUGLAG whose INNER search is such a search: refused at startup
     * since no INNER search is a global one (no nested global search:
     * require_inner_backend), so the RESTART
     * warning start_point_warnings() would give for it (the AUGLAG recursion
     * of search_ignoring_start) is never given. The configuration this
     * section warned about is refused at its ENDINNER line instead. */
    SECTION("an AUGLAG whose INNER search is such a search is refused")
    {
        const std::string config =
            kRoot +
            "OPTIMIZER\n"            // 4
            "  STRATEGY\n"           // 5
            "    TYPE PLAIN\n"
            "    LIB NLOPT\n"
            "    ALGO LN_AUGLAG\n"
            "    MAXEVAL 20\n"
            "    INNER\n"            // 10
            "      LIB NLOPT\n"
            "      ALGO GN_DIRECT_L\n"
            "      MAXEVAL 10\n"
            "    ENDINNER\n"
            "    RESTART\n"          // 15
            "      AFTER_STAGNATION 3\n"
            "      MAX 1\n"
            "    ENDRESTART\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n";
        CHECK_THROWS_WITH(
            startup(config),
            Catch::Contains("Error on line 14: the INNER block of the "
                            "STRATEGY-level optimiser: NLOPT:LN_AUGLAG cannot "
                            "run NLOPT:GN_DIRECT_L as its INNER search: an "
                            "INNER search is a local search"));
    }

    SECTION("a PRESTAGE exploration")
    {
        const std::string config =
            kRoot +
            "OPTIMIZER\n"            // 4
            "  STRATEGY\n"
            "    TYPE PRESTAGE\n"
            "    LIB NLOPT\n"
            "    ALGO LN_BOBYQA\n"
            "    MAXEVAL 10\n"
            "    EXPLORE\n"          // 10
            "      LIB NLOPT\n"
            "      ALGO GN_DIRECT_L\n"
            "      MAXEVAL 8\n"
            "    ENDEXPLORE\n"
            "    RESTART\n"          // 15
            "      AFTER_STAGNATION 3\n"
            "      MAX 1\n"
            "    ENDRESTART\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n";
        const Startup s = startup(config);
        INFO("journal:\n" << s.journal);
        CHECK(count_of(
                  s.journal,
                  "the RESTART block on line 15: the PRESTAGE exploration "
                  "NLOPT:GN_DIRECT_L does not start at the point it is given "
                  "(" + note("GN_DIRECT_L") + ") and draws no random numbers, "
                  "so the exploration of every RESTART round, whatever "
                  "points it starts from, asks for round 0's exploration "
                  "points again, which the run's record answers: a round's "
                  "refinement starts from candidates among round 0's "
                  "exploration points. For explorations that search anew, "
                  "use an exploration optimiser that starts at the point it "
                  "is given.") == 1);
    }

    SECTION("no warning: a search that starts at its point")
    {
        /* An inactive RESTART (MAX 0), the other leg of this section, is
         * refused at its MAX line (the case "startup validation: a RESTART
         * block that never restarts the search is refused"). */
        for (const char* algo : {"LN_SBPLX", "GN_CRS2_LM"}) {
            INFO("NLOPT:" << algo);
            const Startup s = startup(plain_restart(algo));
            CHECK(count_of(s.journal, "the RESTART block on line") == 0);
        }
    }
}

/* Pins: the RESTART warning of a start-ignoring search says what its
 * rounds cost and what they never read (RESTART on line 10, PLAIN, 2
 * OPTIM dimensions):
 *  - GD_STOGO, which obtains gradients: "the RESTART block on line 10: the
 *    STRATEGY-level optimiser NLOPT:GD_STOGO does not start at the point it
 *    is given (StoGO evaluates ...; NLopt never reads that point) and draws
 *    no random numbers, so every RESTART round, whatever points it starts
 *    from, asks for round 0's points again, and their gradients: the run's
 *    record answers their losses, but a round forms its gradients itself
 *    and evaluates their finite-difference stencils again (GRAD_STRATEGY
 *    NUMERICAL): no round after round 0 evaluates a new point, and each
 *    round evaluates up to AFTER_STAGNATION 3 of round 0's points again
 *    before it ends without a new best. For rounds that search anew, use an
 *    optimiser that starts at the point it is given."; under GRAD_STRATEGY
 *    HYBRID "..., and their gradients: the run's record answers their
 *    losses where a round asks for no gradient, but each point at which it
 *    asks for one is evaluated again, with its gradient (GRAD_STRATEGY
 *    HYBRID differentiates an evaluation of the point): ..."; as a PRESTAGE
 *    exploration (RESTART on line 15) "... asks for round 0's exploration
 *    points again, and their gradients: ... (GRAD_STRATEGY NUMERICAL): a
 *    round's refinement starts from candidates among round 0's exploration
 *    points. ...";
 *  - GD_STOGO_RAND below POPULATION 2n + 1 = 5, whose first trial point is
 *    the box centre and whose first local search, from it, draws nothing:
 *    "... (StoGO evaluates the centre ...) and, with POPULATION 4 (NLopt's
 *    default, 2n for n = 2 OPTIM dimensions), less than 2n + 1 = 5, takes
 *    the centre of the box as its first trial point and runs a local
 *    search from it before any of its random trial points, so every
 *    RESTART round, whatever points it starts from, asks first for the
 *    points of round 0's first local search again, and their gradients:
 *    ... (GRAD_STRATEGY NUMERICAL): until that search ends a round
 *    evaluates no new point, and it reaches its random trial points only if
 *    the search ends before the round has had AFTER_STAGNATION 3
 *    evaluations without a new best. For rounds that search anew, use an
 *    optimiser that starts at the point it is given, or POPULATION 5 (2n +
 *    1) or more, whose trial points are all random." (POPULATION 3: "with
 *    POPULATION 3, less than 2n + 1 = 5 (n = 2 OPTIM dimensions), ...");
 *  - a search that draws random numbers, GN_DIRECT_L_RAND in 2 dimensions
 *    and GD_STOGO_RAND at POPULATION 5, under a RESTART block with METHOD
 *    lines: "the RESTART block on line 10: the STRATEGY-level optimiser
 *    NLOPT:GN_DIRECT_L_RAND does not start at the point it is given (...),
 *    so no RESTART round evaluates the points of the block's METHOD lines:
 *    a round's search differs from round 0's only in the random numbers it
 *    draws. To start each round at the block's points, use an optimiser
 *    that starts at the point it is given."; a PRESTAGE exploration on
 *    GN_DIRECT_L_RAND that takes the round's points (its EXPLORE block has
 *    no INIT block; RESTART on line 15) "the RESTART block on line 15: the
 *    PRESTAGE exploration NLOPT:GN_DIRECT_L_RAND, which is given each
 *    round's points (its EXPLORE block has no INIT block), does not start
 *    at the point it is given (...), so no round's exploration evaluates
 *    the points of the block's METHOD lines, and a round's refinement
 *    starts from them only when no evaluation of its exploration has a
 *    usable loss: a round's exploration differs from round 0's only in the
 *    random numbers it draws. To start each round's exploration at them,
 *    use an exploration optimiser that starts at the point it is given."
 * Each is the one RESTART warning of its run. No RESTART warning: a
 * RESTART block without METHOD lines on GN_DIRECT_L_RAND in 2 dimensions
 * (its rounds draw new numbers from a fresh seed), an exploration with an
 * INIT block of its own (the round's points are not the exploration's).
 * Before the fix: GD_STOGO got the text of the searches that obtain no
 * gradient ("..., which the run's record answers: no round after round 0
 * evaluates a new point, and the RESTART block's only effect is to end the
 * run ...", although each round evaluated round 0's stencils again:
 * 4 evaluations after round 0), and GD_STOGO_RAND,
 * GN_DIRECT_L_RAND in 2 dimensions and the METHOD lines got no warning
 * (GD_STOGO_RAND was taken to draw random numbers from its first trial
 * point): every exact-text count is 0. */
TEST_CASE("startup validation: the RESTART warning says what a round of a "
          "gradient search costs, when GD_STOGO_RAND repeats round 0 and "
          "which METHOD lines no round reads",
          "[startup][restart][start_point]")
{
    SECTION("GD_STOGO evaluates round 0's stencils again")
    {
        const std::string tail =
            ": no round after round 0 evaluates a new point, and each round "
            "evaluates up to AFTER_STAGNATION 3 of round 0's points again "
            "before it ends without a new best." + kAnew;
        const std::string head =
            "the RESTART block on line 10: the STRATEGY-level optimiser "
            "NLOPT:GD_STOGO does not start at the point it is given (" +
            note("GD_STOGO") + ") and draws no random numbers, so every "
            "RESTART round, whatever points it starts from, asks for round "
            "0's points again";
        {
            const Startup s = startup(plain_restart("GD_STOGO"));
            INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
            const std::string w = head + kStencils + tail;
            CHECK(count_of(s.journal, w) == 1);
            CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
        {
            const Startup s = startup(plain_restart(
                "GD_STOGO", kRoot, "1", "    GRAD_STRATEGY HYBRID\n"));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(s.journal,
                           head +
                               ", and their gradients: the run's record "
                               "answers their losses where a round asks for "
                               "no gradient, but each point at which it asks "
                               "for one is evaluated again, with its "
                               "gradient (GRAD_STRATEGY HYBRID differentiates "
                               "an evaluation of the point)" + tail) == 1);
        }
    }

    SECTION("a PRESTAGE exploration on GD_STOGO")
    {
        const std::string config =
            kRoot +
            "OPTIMIZER\n"            // 4
            "  STRATEGY\n"
            "    TYPE PRESTAGE\n"
            "    LIB NLOPT\n"
            "    ALGO LN_BOBYQA\n"
            "    MAXEVAL 10\n"
            "    EXPLORE\n"          // 10
            "      LIB NLOPT\n"
            "      ALGO GD_STOGO\n"
            "      MAXEVAL 8\n"
            "    ENDEXPLORE\n"
            "    RESTART\n"          // 15
            "      AFTER_STAGNATION 3\n"
            "      MAX 1\n"
            "    ENDRESTART\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n";
        const Startup s = startup(config);
        INFO("journal:\n" << s.journal);
        CHECK(count_of(
                  s.journal,
                  "the RESTART block on line 15: the PRESTAGE exploration "
                  "NLOPT:GD_STOGO does not start at the point it is given (" +
                      note("GD_STOGO") + ") and draws no random numbers, so "
                      "the exploration of every RESTART round, whatever "
                      "points it starts from, asks for round 0's exploration "
                      "points again" + kStencils +
                      ": a round's refinement starts from candidates among "
                      "round 0's exploration points. For explorations that "
                      "search anew, use an exploration optimiser that starts "
                      "at the point it is given.") == 1);
    }

    SECTION("GD_STOGO_RAND below POPULATION 2n + 1 repeats round 0's first "
            "local search")
    {
        const std::string rest =
            ", takes the centre of the box as its first trial point and runs "
            "a local search from it before any of its random trial points, "
            "so every RESTART round, whatever points it starts from, asks "
            "first for the points of round 0's first local search again" +
            kStencils +
            ": until that search ends a round evaluates no new point, and it "
            "reaches its random trial points only if the search ends before "
            "the round has had AFTER_STAGNATION 3 evaluations without a new "
            "best. For rounds that search anew, use an optimiser that starts "
            "at the point it is given, or POPULATION 5 (2n + 1) or more, "
            "whose trial points are all random.";
        const std::string head =
            "the RESTART block on line 10: the STRATEGY-level optimiser "
            "NLOPT:GD_STOGO_RAND does not start at the point it is given (" +
            note("GD_STOGO_RAND") + ") and, with POPULATION ";
        {
            const Startup s = startup(plain_restart("GD_STOGO_RAND"));
            INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
            const std::string w =
                head + "4 (NLopt's default, 2n for n = 2 OPTIM dimensions), "
                       "less than 2n + 1 = 5" + rest;
            CHECK(count_of(s.journal, w) == 1);
            CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
        {
            const Startup s = startup(plain_restart(
                "GD_STOGO_RAND", kRoot, "1", "    POPULATION 3\n"));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(s.journal,
                           head + "3, less than 2n + 1 = 5 (n = 2 OPTIM "
                                  "dimensions)" + rest) == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
    }

    SECTION("a search that draws random numbers never reads the RESTART "
            "block's METHOD lines")
    {
        {
            const Startup s = startup(plain_restart("GN_DIRECT_L_RAND"));
            INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
            const std::string w = method_lines_warning("GN_DIRECT_L_RAND");
            CHECK(count_of(s.journal, w) == 1);
            CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
        {
            const Startup s = startup(plain_restart(
                "GD_STOGO_RAND", kRoot, "1", "    POPULATION 5\n"));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(s.journal, method_lines_warning("GD_STOGO_RAND")) ==
                  1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
        {
            // without METHOD lines: nothing the rounds do not read
            const Startup s = startup(
                plain_restart("GN_DIRECT_L_RAND", kRoot, "1", "", false));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(s.journal, "the RESTART block on line") == 0);
        }

        const auto prestage = [](const std::string& explore_init) {
            return kRoot +
                   "OPTIMIZER\n"            // 4
                   "  STRATEGY\n"
                   "    TYPE PRESTAGE\n"
                   "    LIB NLOPT\n"
                   "    ALGO LN_BOBYQA\n"
                   "    MAXEVAL 10\n"
                   "    EXPLORE\n"          // 10
                   "      LIB NLOPT\n"
                   "      ALGO GN_DIRECT_L_RAND\n"
                   "      MAXEVAL 8\n" + explore_init +
                   "    ENDEXPLORE\n"
                   "    RESTART\n"          // 15 (19 with an EXPLORE INIT)
                   "      AFTER_STAGNATION 3\n"
                   "      MAX 1\n"
                   "      METHOD RANDOM 1\n"
                   "    ENDRESTART\n"
                   "  ENDSTRATEGY\n"
                   "ENDOPTIMIZER\n";
        };
        {
            const Startup s = startup(prestage(""));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(
                      s.journal,
                      "the RESTART block on line 15: the PRESTAGE exploration "
                      "NLOPT:GN_DIRECT_L_RAND, which is given each round's "
                      "points (its EXPLORE block has no INIT block), does not "
                      "start at the point it is given (" +
                          note("GN_DIRECT_L_RAND") +
                          "), so no round's exploration evaluates the points "
                          "of the block's METHOD lines, and a round's "
                          "refinement starts from them only when no "
                          "evaluation of its exploration has a usable loss: a "
                          "round's exploration differs from round 0's only in "
                          "the random numbers it draws. To start each round's "
                          "exploration at them, use an exploration optimiser "
                          "that starts at the point it is given.") == 1);
            CHECK(count_of(s.journal, "the RESTART block on line") == 1);
        }
        {
            const Startup s = startup(prestage(
                "      INIT\n        METHOD RANDOM 2\n        SEED 3\n"
                "      ENDINIT\n"));
            INFO("journal:\n" << s.journal);
            CHECK(count_of(s.journal, "the RESTART block on line") == 0);
            CHECK(count_of(s.journal, "the INIT block on line 14") == 1);
        }
    }
}

/* Pins: an INIT block with a METHOD on a phase whose search does not start
 * at the point it is given gets exactly one startup WARNING naming the
 * line of the block (INIT):
 *  - PLAIN, the STRATEGY-level optimiser NLOPT:GN_DIRECT_L, INIT on line 5
 *    (METHOD LHS 3): "the INIT block on line 5: the STRATEGY-level
 *    optimiser NLOPT:GN_DIRECT_L does not start at the point it is given
 *    (DIRECT ...; NLopt never reads that point), so the search never
 *    evaluates the block's INIT points. To start the search at them, use an
 *    optimiser that starts at the point it is given."; standard error has
 *    no "INIT chain produced 3 seeds but NLOPT:GN_DIRECT_L consumes only
 *    the first" (it consumes none); under NOISE CALIBRATION 2 the sentence
 *    adds "(NOISE CALIBRATION evaluates the first of them 2 times)";
 *  - PRESTAGE, the EXPLORE block's own INIT block (line 14): "... the
 *    PRESTAGE exploration NLOPT:GN_DIRECT_L does not start ..., so the
 *    exploration never evaluates the block's INIT points. To start the
 *    exploration at them, use an exploration optimiser that starts at the
 *    point it is given." -- an OPTIMIZER-level INIT block beside it, which
 *    only the refinement's fallback and NOISE CALIBRATION read, gets not
 *    this WARNING but the one that the search evaluates its points only in
 *    the refinement ("the INIT block on line 5 gives 3 starting points,
 *    ..."; it got none until that WARNING was added);
 *  - PRESTAGE, an OPTIMIZER-level INIT block (line 5) that the exploration
 *    takes (its EXPLORE block has none): "... the PRESTAGE exploration
 *    NLOPT:GN_DIRECT_L, which is given the block's INIT points (its EXPLORE
 *    block has no INIT block), does not start ..., so the exploration never
 *    evaluates them; the refinement starts from them only when no
 *    evaluation of the exploration has a usable loss. ...".
 * No warning: no INIT block; an INIT block without a METHOD (SEED only);
 * LN_BOBYQA, which starts at its first INIT point (and keeps the
 * WARNING that it "starts from the first only" for METHOD LHS 3, which
 * said "consumes only the first" before the case "startup validation: the
 * INIT points a PLAIN search does not use are warned about in the
 * journal").
 * Before the fix: no INIT warning was given, and standard error said that
 * GN_DIRECT_L "consumes only the first" of the 3 seeds: the counts fail. */
TEST_CASE("startup validation: an INIT block on a search that does not "
          "start at its point warns that its points are never evaluated",
          "[startup][init][start_point]")
{
    const auto plain = [](const std::string& algo, const std::string& init,
                          const std::string& noise = "") {
        return kRoot +
               "OPTIMIZER\n" + noise +  // 4
               init +                   // INIT on line 5 (no NOISE block)
               "  STRATEGY\n"
               "    TYPE PLAIN\n"
               "    LIB NLOPT\n"
               "    ALGO " + algo + "\n"
               "    MAXEVAL 20\n"
               "  ENDSTRATEGY\n"
               "ENDOPTIMIZER\n";
    };
    const std::string lhs =
        "  INIT\n    METHOD LHS 3\n    SEED 7\n  ENDINIT\n";
    const std::string tail =
        " To start the search at them, use an optimiser that starts at the "
        "point it is given.";
    const std::string head =
        ": the STRATEGY-level optimiser NLOPT:GN_DIRECT_L does not start at "
        "the point it is given (" + note("GN_DIRECT_L") +
        "), so the search never evaluates the block's INIT points";

    {
        const Startup s = startup(plain("GN_DIRECT_L", lhs));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        const std::string w = "the INIT block on line 5" + head + "." + tail;
        CHECK(count_of(s.journal, w) == 1);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.err, "consumes only the first") == 0);
    }
    {
        const Startup s = startup(plain(
            "GN_DIRECT_L", lhs, "  NOISE\n    CALIBRATION 2\n  ENDNOISE\n"));
        INFO("journal:\n" << s.journal);
        CHECK(count_of(s.journal,
                       "the INIT block on line 8" + head +
                           " (NOISE CALIBRATION evaluates the first of them 2 "
                           "times)." + tail) == 1);
    }

    const std::string explore_head =
        ": the PRESTAGE exploration NLOPT:GN_DIRECT_L does not start at the "
        "point it is given (" + note("GN_DIRECT_L") + ")";
    const std::string explore_tail =
        " To start the exploration at them, use an exploration optimiser "
        "that starts at the point it is given.";
    const auto prestage = [](const std::string& outer_init,
                             const std::string& explore_init) {
        return kRoot +
               "OPTIMIZER\n" + outer_init +  // 4
               "  STRATEGY\n"
               "    TYPE PRESTAGE\n"
               "    LIB NLOPT\n"
               "    ALGO LN_BOBYQA\n"
               "    MAXEVAL 10\n"
               "    EXPLORE\n"
               "      LIB NLOPT\n"
               "      ALGO GN_DIRECT_L\n"
               "      MAXEVAL 8\n" + explore_init +
               "    ENDEXPLORE\n"
               "  ENDSTRATEGY\n"
               "ENDOPTIMIZER\n";
    };
    const std::string explore_init =
        "      INIT\n        METHOD RANDOM 2\n        SEED 3\n      ENDINIT\n";
    {
        // INIT on line 14: the EXPLORE block's own
        const Startup s = startup(prestage("", explore_init));
        INFO("journal:\n" << s.journal);
        CHECK(count_of(s.journal,
                       "the INIT block on line 14" + explore_head +
                           ", so the exploration never evaluates the block's "
                           "INIT points." + explore_tail) == 1);
        CHECK(count_of(s.journal, "the INIT block on line") == 1);
        // with an OPTIMIZER-level INIT block too (lines 5-8): the
        // exploration's own (now on line 18) is warned about as above, the
        // OPTIMIZER-level one as the refinement's fallback
        const Startup both = startup(prestage(lhs, explore_init));
        INFO("journal:\n" << both.journal);
        CHECK(count_of(both.journal,
                       "the INIT block on line 18" + explore_head +
                           ", so the exploration never evaluates the block's "
                           "INIT points." + explore_tail) == 1);
        CHECK(count_of(both.journal, "the INIT block on line 18") == 1);
        CHECK(count_of(both.journal, "the INIT block on line 5 gives 3 "
                                     "starting points, and the PRESTAGE "
                                     "exploration (EXPLORE block) starts from "
                                     "those of its own INIT block, on line "
                                     "18,") == 1);
        CHECK(count_of(both.journal, "the INIT block on line") == 2);
    }
    {
        // the OPTIMIZER-level block (line 5), which the exploration takes
        const Startup s = startup(prestage(lhs, ""));
        INFO("journal:\n" << s.journal);
        CHECK(count_of(
                  s.journal,
                  "the INIT block on line 5: the PRESTAGE exploration "
                  "NLOPT:GN_DIRECT_L, which is given the block's INIT points "
                  "(its EXPLORE block has no INIT block), does not start at "
                  "the point it is given (" + note("GN_DIRECT_L") +
                      "), so the exploration never evaluates them; the "
                      "refinement starts from them only when no evaluation "
                      "of the exploration has a usable loss." +
                      explore_tail) == 1);
    }

    // no warning
    for (const auto& [algo, init] :
         std::vector<std::pair<std::string, std::string>>{
             {"GN_DIRECT_L", ""},
             {"GN_DIRECT_L", "  INIT\n    SEED 7\n  ENDINIT\n"},
             {"LN_BOBYQA", lhs}}) {
        INFO("NLOPT:" << algo << ", INIT block '" << init << "'");
        const Startup s = startup(plain(algo, init));
        /* This case's WARNING ("the INIT block on line 5: ..."), not the
         * one about the points a search does not evaluate ("the INIT block
         * on line 5 gives 3 starting points, ..."). */
        CHECK(count_of(s.journal, "the INIT block on line 5: ") == 0);
        CHECK(count_of(s.journal, "does not start at the point") == 0);
        if (algo == "LN_BOBYQA")
            CHECK(count_of(s.err, "starts from the first only") == 1);
    }
}

/* Pins: a PLAIN search with an INIT block of METHOD LHS 3 (line 5), whose
 * optimiser takes one starting point: the startup WARNING, on standard
 * error and in the journal, names the INIT block's line, the phase and its
 * optimiser, counts the points it does not evaluate and names only what
 * takes them for that optimiser. For NLOPT:LN_BOBYQA, a local optimiser
 * that STRATEGY TYPE MULTISTART runs from each point: "the INIT block on
 * line 5 gives 3 starting points, and the STRATEGY-level optimiser
 * NLOPT:LN_BOBYQA starts from the first only: the other 2 are not
 * evaluated. STRATEGY TYPE MULTISTART with N_STARTS 3 starts
 * NLOPT:LN_BOBYQA from each of them, and an optimiser that takes several
 * starting points in one run (DLIB:GFS) takes them all, also as the
 * exploration of STRATEGY TYPE PRESTAGE." For a global optimiser that keeps
 * its starting point and that MULTISTART refuses (NLOPT:GN_CRS2_LM,
 * ASA:ASA, TADAH:ANNEAL): "... the other 2 are not evaluated. An optimiser
 * that takes several starting points in one run (DLIB:GFS) takes them all,
 * also as the exploration of STRATEGY TYPE PRESTAGE.", which names no
 * MULTISTART; the same file under TYPE MULTISTART with N_STARTS 3 is
 * refused ("STRATEGY TYPE MULTISTART cannot run ..."), while the LN_BOBYQA
 * one gets no such WARNING. No WARNING says anything of seeds.
 * Before the fix: the WARNING came when the file was read ("INIT chain
 * produced 3 starting points but NLOPT:LN_BOBYQA starts from the first
 * only; the other 2 are not used. ..."), naming neither the INIT block's
 * line nor the phase (the texts fail). Before that: every optimiser was
 * advised "STRATEGY TYPE MULTISTART with N_STARTS 3 or TYPE PRESTAGE uses
 * them, and so does an optimiser that takes several starting points in
 * one run (DLIB:GFS).". */
TEST_CASE("startup validation: the INIT points a PLAIN search does not use "
          "are warned about in the journal", "[startup][init][start_point]")
{
    const std::string lhs =
        "  INIT\n    METHOD LHS 3\n    SEED 7\n  ENDINIT\n";
    const auto run = [&](const std::string& strategy, const std::string& lib,
                         const std::string& algo) {
        return kRoot + "OPTIMIZER\n" + lhs + "  STRATEGY\n" + strategy +
               "    LIB " + lib + "\n    ALGO " + algo +
               "\n    MAXEVAL 20\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const std::string head =
        "the INIT block on line 5 gives 3 starting points, and the "
        "STRATEGY-level optimiser ";
    const std::string unused =
        " starts from the first only: the other 2 are not evaluated. ";
    const std::string takes_all =
        "optimiser that takes several starting points in one run (DLIB:GFS) "
        "takes them all, also as the exploration of STRATEGY TYPE PRESTAGE.";
    {
        const std::string w =
            head + "NLOPT:LN_BOBYQA" + unused + "STRATEGY TYPE MULTISTART "
            "with N_STARTS 3 starts NLOPT:LN_BOBYQA from each of them, and an " +
            takes_all;
        const Startup s =
            startup(run("    TYPE PLAIN\n", "NLOPT", "LN_BOBYQA"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
        CHECK(count_of(s.err, "seed(s)") == 0);
    }
    {
        const Startup s = startup(
            run("    TYPE MULTISTART\n    N_STARTS 3\n", "NLOPT", "LN_BOBYQA"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "starts from the first only") == 0);
        CHECK(count_of(s.journal, "starts from the first only") == 0);
        CHECK(count_of(s.journal, "are not evaluated") == 0);
    }
    for (const auto& [lib, algo] :
         std::vector<std::pair<std::string, std::string>>{
             {"NLOPT", "GN_CRS2_LM"}, {"ASA", "ASA"}, {"TADAH", "ANNEAL"}}) {
        const std::string name = lib + ":" + algo;
        INFO(name);
        const std::string w = head + name + unused + "An " + takes_all;
        const Startup s = startup(run("    TYPE PLAIN\n", lib, algo));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
        CHECK(count_of(s.err, "MULTISTART") == 0);
        CHECK(count_of(s.journal, "MULTISTART") == 0);
        CHECK_THROWS_WITH(
            startup(run("    TYPE MULTISTART\n    N_STARTS 3\n", lib, algo)),
            Catch::Contains("STRATEGY TYPE MULTISTART cannot run " + name));
    }
}

/* Pins: the startup WARNING about INIT points a phase does not evaluate,
 * on standard error and in the journal, for the phases that take fewer
 * points than an INIT block gives:
 *  - STRATEGY TYPE PRESTAGE whose exploration (NLOPT:GN_CRS2_LM) takes one
 *    starting point, given the OPTIMIZER-level block's 7 points (METHOD
 *    CONFIG and METHOD LHS 6, the block on line 5): "the INIT block on line
 *    5 gives 7 starting points, and the PRESTAGE exploration (EXPLORE
 *    block) NLOPT:GN_CRS2_LM starts from the first only: the other 6 are
 *    not evaluated (the refinement starts from them only when no evaluation
 *    of the exploration has a usable loss). An exploration optimiser that
 *    takes several starting points in one run (DLIB:GFS) takes them all.";
 *  - the same with the EXPLORE block's own INIT block of METHOD RANDOM 5
 *    (line 14): "the INIT block on line 14 gives 5 starting points, and the
 *    PRESTAGE exploration (EXPLORE block) NLOPT:GN_CRS2_LM starts from the
 *    first only: the other 4 are not evaluated. An exploration optimiser
 *    ...";
 *  - STRATEGY TYPE MULTISTART with N_STARTS 3 and METHOD LHS 7: "the INIT
 *    block on line 5 gives 7 starting points, and STRATEGY TYPE MULTISTART
 *    runs NLOPT:LN_BOBYQA from N_STARTS 3 of them: the other 4 are not
 *    evaluated. N_STARTS 7 starts it from each of them.".
 * No such WARNING: an exploration on DLIB:GFS, which takes every point;
 * MULTISTART with as many starts as points.
 * Before the fix: a PRESTAGE and a MULTISTART of N_STARTS 2 or more counted
 * as taking every point, and nothing was said (the counts are 0). */
TEST_CASE("startup validation: the INIT points a PRESTAGE exploration or a "
          "MULTISTART does not evaluate are warned about",
          "[startup][init][start_point]")
{
    const std::string chain7 =
        "  INIT\n    METHOD CONFIG\n    METHOD LHS 6\n    SEED 7\n"
        "  ENDINIT\n";  // lines 5-9
    const auto prestage = [](const std::string& outer_init,
                             const std::string& lib, const std::string& algo,
                             const std::string& explore_init) {
        return kRoot + "OPTIMIZER\n" + outer_init +
               "  STRATEGY\n"
               "    TYPE PRESTAGE\n"
               "    LIB NLOPT\n"
               "    ALGO LN_BOBYQA\n"
               "    MAXEVAL 10\n"
               "    EXPLORE\n"
               "      LIB " + lib + "\n"
               "      ALGO " + algo + "\n"
               "      MAXEVAL 8\n" + explore_init +
               "    ENDEXPLORE\n"
               "  ENDSTRATEGY\n"
               "ENDOPTIMIZER\n";
    };
    const std::string takes_all =
        " An exploration optimiser that takes several starting points in one "
        "run (DLIB:GFS) takes them all.";
    {
        const std::string w =
            "the INIT block on line 5 gives 7 starting points, and the "
            "PRESTAGE exploration (EXPLORE block) NLOPT:GN_CRS2_LM starts from "
            "the first only: the other 6 are not evaluated (the refinement "
            "starts from them only when no evaluation of the exploration has "
            "a usable loss)." + takes_all;
        const Startup s = startup(prestage(chain7, "NLOPT", "GN_CRS2_LM", ""));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
    }
    {
        // the EXPLORE block's own INIT block, lines 14-17
        const std::string w =
            "the INIT block on line 14 gives 5 starting points, and the "
            "PRESTAGE exploration (EXPLORE block) NLOPT:GN_CRS2_LM starts from "
            "the first only: the other 4 are not evaluated." + takes_all;
        const Startup s = startup(prestage(
            "", "NLOPT", "GN_CRS2_LM",
            "      INIT\n        METHOD RANDOM 5\n        SEED 3\n"
            "      ENDINIT\n"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
    }
    {
        const Startup s = startup(prestage(chain7, "DLIB", "GFS", ""));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.journal, "are not evaluated") == 0);
        CHECK(count_of(s.err, "are not evaluated") == 0);
    }
    const auto multistart = [](const std::string& k) {
        return kRoot + "OPTIMIZER\n"
               "  INIT\n    METHOD LHS " + k + "\n    SEED 7\n  ENDINIT\n"
               "  STRATEGY\n"
               "    TYPE MULTISTART\n"
               "    N_STARTS 3\n"
               "    LIB NLOPT\n"
               "    ALGO LN_BOBYQA\n"
               "    MAXEVAL 30\n"
               "  ENDSTRATEGY\n"
               "ENDOPTIMIZER\n";
    };
    {
        const std::string w =
            "the INIT block on line 5 gives 7 starting points, and STRATEGY "
            "TYPE MULTISTART runs NLOPT:LN_BOBYQA from N_STARTS 3 of them: the "
            "other 4 are not evaluated. N_STARTS 7 starts it from each of "
            "them.";
        const Startup s = startup(multistart("7"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
    }
    {
        const Startup s = startup(multistart("3"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.journal, "are not evaluated") == 0);
    }
}

/* Pins: under STRATEGY TYPE PRESTAGE whose EXPLORE block has an INIT block
 * of its own (METHOD RANDOM 5, line 19), the OPTIMIZER-level INIT block
 * (METHOD CONFIG and METHOD LHS 6, line 5) gives the refinement its
 * starting points only when no evaluation of the exploration has a usable
 * loss, and the startup WARNING, on standard error and in the journal, says
 * so: "the INIT block on line 5 gives 7 starting points, and the PRESTAGE
 * exploration (EXPLORE block) starts from those of its own INIT block, on
 * line 19, so the search evaluates them only in the refinement, when no
 * evaluation of the exploration has a usable loss. Without an INIT block
 * of its own, the exploration takes them."; whatever the exploration's
 * optimiser (NLOPT:GN_CRS2_LM, which also leaves 4 of its own 5 points
 * unevaluated, and DLIB:GFS, which takes them all); under NOISE
 * CALIBRATION 3 it adds " (NOISE CALIBRATION evaluates the first of them 3
 * times)" after "usable loss". An exploration without an INIT block of its
 * own, which takes the OPTIMIZER-level points, gives no such WARNING.
 * Before the fix: nothing was said of the OPTIMIZER-level block (the
 * counts are 0). */
TEST_CASE("startup validation: the OPTIMIZER-level INIT points of a "
          "PRESTAGE whose exploration has its own are warned about",
          "[startup][init][start_point]")
{
    const std::string chain7 =
        "  INIT\n    METHOD CONFIG\n    METHOD LHS 6\n    SEED 7\n"
        "  ENDINIT\n";  // lines 5-9
    const std::string own =
        "      INIT\n        METHOD RANDOM 5\n        SEED 3\n"
        "      ENDINIT\n";  // lines 19-22
    const auto prestage = [&](const std::string& lib, const std::string& algo,
                              const std::string& explore_init,
                              const std::string& noise) {
        return kRoot + "OPTIMIZER\n" + chain7 +
               "  STRATEGY\n"
               "    TYPE PRESTAGE\n"
               "    LIB NLOPT\n"
               "    ALGO LN_BOBYQA\n"
               "    MAXEVAL 10\n"
               "    EXPLORE\n"
               "      LIB " + lib + "\n"
               "      ALGO " + algo + "\n"
               "      MAXEVAL 8\n" + explore_init +
               "    ENDEXPLORE\n"
               "  ENDSTRATEGY\n" + noise +
               "ENDOPTIMIZER\n";
    };
    const auto warning = [](const std::string& calibration) {
        return "the INIT block on line 5 gives 7 starting points, and the "
               "PRESTAGE exploration (EXPLORE block) starts from those of its "
               "own INIT block, on line 19, so the search evaluates them only "
               "in the refinement, when no evaluation of the exploration has "
               "a usable loss" + calibration + ". Without an INIT block of "
               "its own, the exploration takes them.";
    };
    for (const auto& [lib, algo] :
         {std::pair<std::string, std::string>{"NLOPT", "GN_CRS2_LM"},
          std::pair<std::string, std::string>{"DLIB", "GFS"}}) {
        INFO(lib << ":" << algo);
        const Startup s = startup(prestage(lib, algo, own, ""));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        CHECK(count_of(s.err, "[HPO] WARNING: " + warning("") + "\n") == 1);
        CHECK(count_of(s.journal, warning("")) == 1);
        const Startup none = startup(prestage(lib, algo, "", ""));
        INFO("journal:\n" << none.journal);
        CHECK(count_of(none.journal, "starts from those of its own INIT "
                                     "block") == 0);
    }
    {
        const Startup s = startup(prestage(
            "DLIB", "GFS", own, "  NOISE\n    CALIBRATION 3\n  ENDNOISE\n"));
        INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
        const std::string w =
            warning(" (NOISE CALIBRATION evaluates the first of them 3 times)");
        CHECK(count_of(s.err, "[HPO] WARNING: " + w + "\n") == 1);
        CHECK(count_of(s.journal, w) == 1);
    }
}

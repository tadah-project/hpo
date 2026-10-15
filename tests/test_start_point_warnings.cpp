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
//  - a RESTART block, when the search also draws no random numbers: every
//    round asks for round 0's points again, whatever points it starts from
//    (the run's record answers them), so no round after round 0 searches
//    anew;
//  - an INIT block with a METHOD: the search never evaluates its points.
// A search that starts at its point, or one that draws random numbers
// (GD_STOGO_RAND; GN_DIRECT_L_RAND in two or more dimensions), gets neither
// RESTART warning; an INIT block without a METHOD (its SEED only) and an
// inactive RESTART block get none.
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
 * which opens on line 10 (line 9 with kRoot1). */
std::string plain_restart(const std::string& algo,
                          const std::string& root = kRoot,
                          const std::string& max = "1")
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
           "      MAX " + max + "\n"
           "      METHOD RANDOM 1\n"
           "      SEED 5\n"
           "    ENDRESTART\n"
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
 * `line`, the run answering repeats from its record. */
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

}  // namespace

/* Pins: under STRATEGY TYPE PLAIN with an active RESTART block (line 10),
 * each NLopt search that does not start at the point it is given and draws
 * no random numbers -- GD_STOGO, GN_DIRECT, GN_DIRECT_L, the three _NOSCAL
 * forms (NLopt 2.10 runs all three as DIRECT-L, GN_DIRECT_L_RAND_NOSCAL
 * included), GN_ORIG_DIRECT, GN_ORIG_DIRECT_L, GN_AGS -- gets exactly one
 * startup WARNING, in the journal and on standard error ("[HPO] WARNING:
 * "): "the RESTART block on line 10: the STRATEGY-level optimiser
 * NLOPT:GN_DIRECT_L does not start at the point it is given (DIRECT
 * evaluates the centre of the OPTIM box first and then divides the whole
 * box; NLopt never reads that point) and draws no random numbers, so every
 * RESTART round, whatever points it starts from, asks for round 0's points
 * again, which the run's record answers: no round after round 0 evaluates
 * a new point, and the RESTART block's only effect is to end the run once
 * round 0 has had no new best in AFTER_STAGNATION 3 evaluations. For
 * rounds that search anew, use an optimiser that starts at the point it is
 * given." (each with its own start note). GN_DIRECT_L_RAND in one OPTIM
 * dimension (RESTART on line 9) gets it with "and, in one OPTIM dimension,
 * draws no random numbers (it draws them only to choose between equally
 * long sides of a box)"; in two it is not warned about, nor is
 * GD_STOGO_RAND. With a --skip-rel filter, which makes the run evaluate a
 * repeated point again, the warning says "asks for round 0's points again
 * as long as their losses are those of round 0 (the run evaluates them
 * again: PC_ERMSE --skip-rel: ...)". An AUGLAG whose INNER search is
 * GN_DIRECT_L names both ("NLOPT:LN_AUGLAG runs its INNER search
 * NLOPT:GN_DIRECT_L from the point it is given, and NLOPT:GN_DIRECT_L does
 * not start ..."); a PRESTAGE exploration on GN_DIRECT_L gets the
 * exploration's text (RESTART on line 15). No warning: LN_SBPLX and
 * GN_CRS2_LM, which start at their point, and an inactive RESTART (MAX 0).
 * Before the fix: none of these warnings was given (tmp/sr_probe/
 * run_restart_direct: round 1 asked for round 0's points, answered from the
 * record, without a word at startup): every count is 0. */
TEST_CASE("startup validation: RESTART on a search that does not start at "
          "its point and draws no random numbers warns that its rounds "
          "repeat round 0",
          "[startup][restart][start_point]")
{
    SECTION("each search that draws no random numbers")
    {
        for (const char* algo :
             {"GD_STOGO", "GN_DIRECT", "GN_DIRECT_L", "GN_DIRECT_NOSCAL",
              "GN_DIRECT_L_NOSCAL", "GN_DIRECT_L_RAND_NOSCAL",
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

    SECTION("GN_DIRECT_L_RAND draws random numbers only in two or more "
            "dimensions; GD_STOGO_RAND always")
    {
        const Startup one = startup(plain_restart("GN_DIRECT_L_RAND", kRoot1));
        INFO("journal:\n" << one.journal);
        CHECK(count_of(one.journal,
                       plain_restart_warning(
                           "GN_DIRECT_L_RAND", 9,
                           " and, in one OPTIM dimension, draws no random "
                           "numbers (it draws them only to choose between "
                           "equally long sides of a box)")) == 1);
        for (const char* algo : {"GN_DIRECT_L_RAND", "GD_STOGO_RAND"}) {
            INFO("NLOPT:" << algo << " in 2 dimensions");
            const Startup two = startup(plain_restart(algo));
            CHECK(count_of(two.journal, "the RESTART block on line") == 0);
        }
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

    SECTION("an AUGLAG whose INNER search is such a search")
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
        const Startup s = startup(config);
        INFO("journal:\n" << s.journal);
        CHECK(count_of(
                  s.journal,
                  "the RESTART block on line 15: the STRATEGY-level optimiser "
                  "NLOPT:LN_AUGLAG runs its INNER search NLOPT:GN_DIRECT_L "
                  "from the point it is given, and NLOPT:GN_DIRECT_L does not "
                  "start at the point it is given (" + note("GN_DIRECT_L") +
                      ") and draws no random numbers, so every RESTART "
                      "round, whatever points it starts from, asks for round "
                      "0's points again, which the run's record answers: no "
                      "round after round 0 evaluates a new point, and the "
                      "RESTART block's only effect is to end the run once "
                      "round 0 has had no new best in AFTER_STAGNATION 3 "
                      "evaluations." + kAnew) == 1);
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

    SECTION("no warning: a search that starts at its point, an inactive "
            "RESTART")
    {
        for (const char* algo : {"LN_SBPLX", "GN_CRS2_LM"}) {
            INFO("NLOPT:" << algo);
            const Startup s = startup(plain_restart(algo));
            CHECK(count_of(s.journal, "the RESTART block on line") == 0);
        }
        const Startup off = startup(plain_restart("GN_DIRECT_L", kRoot, "0"));
        CHECK(count_of(off.journal, "the RESTART block on line") == 0);
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
 *    only the refinement's fallback and NOISE CALIBRATION read, gets none;
 *  - PRESTAGE, an OPTIMIZER-level INIT block (line 5) that the exploration
 *    takes (its EXPLORE block has none): "... the PRESTAGE exploration
 *    NLOPT:GN_DIRECT_L, which is given the block's INIT points (its EXPLORE
 *    block has no INIT block), does not start ..., so the exploration never
 *    evaluates them; the refinement starts from them only when no
 *    evaluation of the exploration has a usable loss. ...".
 * No warning: no INIT block; an INIT block without a METHOD (SEED only);
 * LN_BOBYQA, which starts at its first INIT point (and keeps the read-time
 * "consumes only the first" WARNING for METHOD LHS 3).
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
        // with an OPTIMIZER-level INIT block too (lines 5-8): only the
        // exploration's own (now on line 18) is warned about
        const Startup both = startup(prestage(lhs, explore_init));
        INFO("journal:\n" << both.journal);
        CHECK(count_of(both.journal,
                       "the INIT block on line 18" + explore_head +
                           ", so the exploration never evaluates the block's "
                           "INIT points." + explore_tail) == 1);
        CHECK(count_of(both.journal, "the INIT block on line") == 1);
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
        CHECK(count_of(s.journal, "the INIT block on line") == 0);
        if (algo == "LN_BOBYQA")
            CHECK(count_of(s.err, "consumes only the first") == 1);
    }
}

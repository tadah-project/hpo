// test_hpotarget_block_structure.cpp
//
// The block structure of the target file (--hpotarget) that
// HPOTargets::read() accepts: one OPTIMIZER block, at the top level of the
// file, and every block closed by its END line before the block around it
// closes and before the file ends. Each refusal names the line it is
// about:
//  - a second OPTIMIZER block, at its OPTIMIZER line, naming the line of
//    the first;
//  - an OPTIMIZER line inside a block, at its line, naming the innermost
//    open block and the line that opens it;
//  - an END line that does not close the innermost open block, at its
//    line, naming that block, the line that opens it and the END line that
//    closes it (an ENDOPTIMIZER inside a STRATEGY block closed the whole
//    OPTIMIZER block, a second OPTIMIZER block reopened the first and an
//    OPTIMIZER line inside a block went back to the OPTIMIZER level);
//  - an END line with no block open: "ENDX without matching X", for
//    ENDOPTIMIZER as for the other blocks;
//  - a block the file leaves open, at the line that opens it, the
//    innermost open block first (a file without ENDOPTIMIZER, or without
//    the END line of a block at its end, was read as complete);
//  - a line that opens or closes a block and holds anything after its
//    key, at its line, naming the first extra token (`OPTIMIZER LIB
//    NLOPT` opened the block and dropped `LIB NLOPT`, and `ENDOPTIMIZER
//    junk here` closed it).
// A repeated INIT, NOISE, EXPLORE, REFINE or RESTART block continues the
// first: its lines set the same settings, a METHOD line adds to the same
// chain and an INIT, NOISE or RESTART key set in both takes the later
// value. An optimiser key (LIB, ALGO, MAXEVAL, SEED, PARAM <name>, ...) is
// set once for its optimiser: a second line is refused at its line, naming
// the first, whether it stands in the same block, a repeated block or
// another block that configures the same optimiser (OPTIMIZER, STRATEGY
// and REFINE configure the STRATEGY-level one); a REFINE line replaced the
// STRATEGY block's value without a word.
// A NOISE block takes AUTO and CALIBRATION: an optimiser key inside it is
// refused at its line, as every other key (it set the optimiser of the
// OPTIMIZER block around it).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/optimizer_spec.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;

using Lines = std::vector<std::string>;

/* A per-process unique name: ctest may run the cases of this file in
 * parallel processes. */
std::string unique_name(const std::string& stem)
{
    static unsigned counter = 0;
    return stem + std::to_string(::getpid()) + "_" + std::to_string(counter++) +
           "_" + std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch().count());
}

/* The file whose lines are `lines`, line 1 first. */
struct ScopedTempFile {
    fs::path path;
    explicit ScopedTempFile(const Lines& lines)
        : path(fs::temp_directory_path() / (unique_name("tadah_hbs_") + ".hpo"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        for (const auto& l : lines) os << l << "\n";
    }
    ~ScopedTempFile() { std::error_code ec; fs::remove(path, ec); }
};

/* The smallest training configuration whose model builds. */
Context make_minimal_ctx()
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
    return c;
}

/* The message HPOTargets gives for the file `lines`, "" when it reads it. */
std::string parse_error(const Lines& lines)
{
    PeriodicTable::initialize();
    ScopedTempFile tf(lines);
    Context ctx = make_minimal_ctx();
    try {
        HPOTargets trg(ctx, tf.path.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* A refusal at line `n`, whose text is `text`. */
std::string at_line(std::size_t n, const std::string& what,
                    const std::string& text)
{
    return "Error on line " + std::to_string(n) + ": " + what +
           "\nLine: " + text;
}

/* `lines` followed by `more`. */
Lines operator+(Lines lines, const Lines& more)
{
    lines.insert(lines.end(), more.begin(), more.end());
    return lines;
}

/* `lines` with `text` inserted as line n + 1. */
Lines insert_after(Lines lines, std::size_t n, const std::string& text)
{
    lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(n), text);
    return lines;
}

/* Lines 1 to n of `lines`. */
Lines first_lines(const Lines& lines, std::size_t n)
{
    return Lines(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(n));
}

/* `lines` with `more` appended to line n. */
Lines append_to_line(Lines lines, std::size_t n, const std::string& more)
{
    lines[n - 1] += more;
    return lines;
}

/* Lines 1-2 of every file below: top-level keys. */
const Lines kRoot = {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1"};

/* A complete file with a block of one type between two known lines. */
struct Frame {
    std::string block;   // its name; "END" + block closes it
    Lines lines;         // the whole file
    std::size_t open;    // the line that opens the block
    std::size_t end;     // the line that closes it
};

std::vector<Frame> frames()
{
    const Lines explore = {"    EXPLORE", "      LIB DLIB", "      ALGO GFS",
                           "      MAXEVAL 12", "    ENDEXPLORE"};
    return {
        {"OPTIMIZER",
         kRoot + Lines{"OPTIMIZER", "  LIB DLIB", "  ALGO BFGS",
                       "ENDOPTIMIZER"},
         3, 6},
        {"INIT",
         kRoot + Lines{"OPTIMIZER", "  LIB DLIB", "  ALGO BFGS", "  INIT",
                       "    METHOD CONFIG", "  ENDINIT", "ENDOPTIMIZER"},
         6, 8},
        {"NOISE",
         kRoot + Lines{"OPTIMIZER", "  LIB DLIB", "  ALGO BFGS", "  NOISE",
                       "    CALIBRATION 0", "  ENDNOISE", "ENDOPTIMIZER"},
         6, 8},
        {"STRATEGY",
         kRoot + Lines{"OPTIMIZER", "  STRATEGY", "    LIB DLIB",
                       "    ALGO BFGS", "  ENDSTRATEGY", "ENDOPTIMIZER"},
         4, 7},
        {"EXPLORE",
         kRoot + Lines{"OPTIMIZER", "  STRATEGY", "    TYPE PRESTAGE",
                       "    LIB DLIB", "    ALGO BFGS"} +
             explore + Lines{"  ENDSTRATEGY", "ENDOPTIMIZER"},
         8, 12},
        {"REFINE",
         kRoot + Lines{"OPTIMIZER", "  STRATEGY", "    TYPE PRESTAGE"} +
             explore +
             Lines{"    REFINE", "      LIB DLIB", "      ALGO BFGS",
                   "    ENDREFINE", "  ENDSTRATEGY", "ENDOPTIMIZER"},
         11, 14},
        {"INNER",
         kRoot + Lines{"OPTIMIZER", "  STRATEGY", "    LIB NLOPT",
                       "    ALGO G_MLSL_LDS", "    MAXEVAL 60", "    INNER",
                       "      LIB NLOPT", "      ALGO LN_SBPLX",
                       "      MAXEVAL 5", "    ENDINNER", "  ENDSTRATEGY",
                       "ENDOPTIMIZER"},
         8, 12},
        {"RESTART",
         kRoot + Lines{"OPTIMIZER", "  STRATEGY", "    LIB DLIB",
                       "    ALGO BFGS", "    RESTART",
                       "      AFTER_STAGNATION 5", "      MAX 2",
                       "    ENDRESTART", "  ENDSTRATEGY", "ENDOPTIMIZER"},
         7, 10},
    };
}

/* The END line of every block type. */
const std::vector<std::string> kEnds = {
    "ENDOPTIMIZER", "ENDINIT", "ENDNOISE", "ENDSTRATEGY",
    "ENDEXPLORE",   "ENDREFINE", "ENDINNER", "ENDRESTART"};

}  // namespace

TEST_CASE("hpotarget block structure: the files of these tests are read "
          "with each block closed",
          "[hpotarget][block_structure]")
{
    /* The refusals below come from the one line each case changes. */
    for (const auto& f : frames()) {
        INFO(f.block);
        REQUIRE(f.lines.size() >= f.end);
        CHECK(f.lines[f.open - 1].find(f.block) != std::string::npos);
        CHECK(f.lines[f.end - 1].find("END" + f.block) != std::string::npos);
        CHECK(parse_error(f.lines) == "");
    }
}

TEST_CASE("hpotarget block structure: a second OPTIMIZER block is refused at "
          "its line, naming the line of the first",
          "[hpotarget][block_structure]")
{
    const Lines first = kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                      "  ALGO BFGS", "ENDOPTIMIZER"};
    const std::string twice =
        "OPTIMIZER block given more than once (the first opens on line 3); "
        "write every optimiser setting in one block";
    SECTION("right after the first") {
        CHECK(parse_error(first + Lines{"OPTIMIZER", "  MAXEVAL 5",
                                        "ENDOPTIMIZER"}) ==
              at_line(7, twice, "OPTIMIZER"));
    }
    SECTION("after a top-level key, with a STRATEGY block") {
        /* The second block's STRATEGY block was refused as a second
         * STRATEGY block only when the first had one. */
        CHECK(parse_error(first + Lines{"OUTPUT 10", "OPTIMIZER",
                                        "  STRATEGY", "    TYPE PLAIN",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(8, twice, "OPTIMIZER"));
    }
    SECTION("an empty second block") {
        CHECK(parse_error(first + Lines{"OPTIMIZER", "ENDOPTIMIZER"}) ==
              at_line(7, twice, "OPTIMIZER"));
    }
}

TEST_CASE("hpotarget block structure: an OPTIMIZER line inside a block is "
          "refused at its line, naming the innermost open block",
          "[hpotarget][block_structure]")
{
    SECTION("inside each block type, after its first line") {
        for (const auto& f : frames()) {
            INFO(f.block);
            CHECK(parse_error(insert_after(f.lines, f.open, "OPTIMIZER")) ==
                  at_line(f.open + 1,
                          "OPTIMIZER block must appear at the top level of the "
                          "file (it is inside the " + f.block +
                              " block opened on line " +
                              std::to_string(f.open) + ")",
                          "OPTIMIZER"));
        }
    }
    SECTION("inside each block type, before its END line") {
        for (const auto& f : frames()) {
            INFO(f.block);
            CHECK(parse_error(insert_after(f.lines, f.end - 1, "  OPTIMIZER")) ==
                  at_line(f.end,
                          "OPTIMIZER block must appear at the top level of the "
                          "file (it is inside the " + f.block +
                              " block opened on line " +
                              std::to_string(f.open) + ")",
                          "  OPTIMIZER"));
        }
    }
}

TEST_CASE("hpotarget block structure: an END line that does not close the "
          "innermost open block is refused at its line, naming that block",
          "[hpotarget][block_structure]")
{
    SECTION("ENDOPTIMIZER while a STRATEGY block is open") {
        /* It closed the whole OPTIMIZER block. */
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "ENDOPTIMIZER"}) ==
              at_line(7,
                      "ENDOPTIMIZER does not close the innermost open block, "
                      "the STRATEGY block opened on line 4 (ENDSTRATEGY "
                      "closes it)",
                      "ENDOPTIMIZER"));
    }
    SECTION("ENDSTRATEGY while an INIT block inside it is open") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "    INIT", "      METHOD CONFIG",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(9,
                      "ENDSTRATEGY does not close the innermost open block, "
                      "the INIT block opened on line 7 (ENDINIT closes it)",
                      "  ENDSTRATEGY"));
    }
    SECTION("each other END line inside each block type") {
        for (const auto& f : frames()) {
            for (const auto& end : kEnds) {
                if (end == "END" + f.block) continue;
                INFO(end << " inside " << f.block);
                CHECK(parse_error(insert_after(f.lines, f.open, "  " + end)) ==
                      at_line(f.open + 1,
                              end + " does not close the innermost open "
                              "block, the " + f.block + " block opened on "
                              "line " + std::to_string(f.open) + " (END" +
                                  f.block + " closes it)",
                              "  " + end));
            }
        }
    }
}

TEST_CASE("hpotarget block structure: an END line with no block open is "
          "refused at its line",
          "[hpotarget][block_structure]")
{
    const Frame f = frames().front();   // the OPTIMIZER block, lines 3-6
    for (const auto& end : kEnds) {
        INFO(end);
        const std::string what =
            end + " without matching " + end.substr(3);
        /* After the OPTIMIZER block (a second ENDOPTIMIZER was ignored). */
        CHECK(parse_error(f.lines + Lines{end}) == at_line(7, what, end));
        /* Before it. */
        CHECK(parse_error(insert_after(f.lines, 2, end)) ==
              at_line(3, what, end));
    }
}

TEST_CASE("hpotarget block structure: a block the file leaves open is refused "
          "at the line that opens it, the innermost first",
          "[hpotarget][block_structure]")
{
    SECTION("a file without ENDOPTIMIZER") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS"}) ==
              at_line(3, "OPTIMIZER block not closed with ENDOPTIMIZER",
                      "OPTIMIZER"));
    }
    SECTION("each block type, the file ending before its END line") {
        /* Every block around it is open as well: the innermost is named. */
        for (const auto& f : frames()) {
            INFO(f.block);
            CHECK(parse_error(first_lines(f.lines, f.end - 1)) ==
                  at_line(f.open,
                          f.block + " block not closed with END" + f.block,
                          f.lines[f.open - 1]));
        }
    }
    SECTION("a closed INNER block inside an open STRATEGY block") {
        const Frame inner = frames()[6];
        REQUIRE(inner.block == "INNER");
        CHECK(parse_error(first_lines(inner.lines, inner.end)) ==
              at_line(4, "STRATEGY block not closed with ENDSTRATEGY",
                      "  STRATEGY"));
    }
    SECTION("comments and blank lines after the last line") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "  ENDSTRATEGY", "",
                                        "# ENDOPTIMIZER", ""}) ==
              at_line(3, "OPTIMIZER block not closed with ENDOPTIMIZER",
                      "OPTIMIZER"));
    }
}

TEST_CASE("hpotarget block structure: a line that opens or closes a block "
          "holds its key alone, and anything after the key is refused at "
          "its line",
          "[hpotarget][block_structure]")
{
    SECTION("each opening line, with a key after the block's name") {
        /* `OPTIMIZER LIB NLOPT` opened the block and dropped `LIB NLOPT`. */
        for (const auto& f : frames()) {
            INFO(f.block);
            const Lines lines = append_to_line(f.lines, f.open, " LIB NLOPT");
            CHECK(parse_error(lines) ==
                  at_line(f.open,
                          f.block + " opens a block and takes no value "
                          "(unexpected 'LIB'); write each key on its own "
                          "line inside the block",
                          lines[f.open - 1]));
        }
    }
    SECTION("each END line, with words after it") {
        /* `ENDOPTIMIZER junk here` closed the block. */
        for (const auto& f : frames()) {
            INFO(f.block);
            const Lines lines = append_to_line(f.lines, f.end, " junk here");
            CHECK(parse_error(lines) ==
                  at_line(f.end,
                          "END" + f.block + " closes the " + f.block +
                              " block and takes no value (unexpected 'junk')",
                          lines[f.end - 1]));
        }
    }
    SECTION("a comment or blanks after the key are not part of the line") {
        for (const auto& f : frames()) {
            INFO(f.block);
            CHECK(parse_error(append_to_line(
                      append_to_line(f.lines, f.open, "   # opens it"), f.end,
                      " \t")) == "");
        }
    }
    SECTION("STRATEGY inside INIT and RESTART is the METHOD key, with its "
            "values") {
        PeriodicTable::initialize();
        Context ctx = make_minimal_ctx();
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "    INIT", "      STRATEGY RANDOM 2",
                                        "    ENDINIT", "    RESTART",
                                        "      STRATEGY LHS 3",
                                        "    ENDRESTART", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        const auto& init = trg.spec.init_strategy_spec;
        REQUIRE(init.chain.size() == 1);
        CHECK(init.chain[0].method == "RANDOM");
        const auto& restart = trg.spec.restart.init_strategy;
        REQUIRE(restart.chain.size() == 1);
        CHECK(restart.chain[0].method == "LHS");
    }
}

TEST_CASE("hpotarget block structure: a repeated INIT, NOISE, EXPLORE, "
          "REFINE or RESTART block continues the first",
          "[hpotarget][block_structure]")
{
    PeriodicTable::initialize();
    Context ctx = make_minimal_ctx();
    SECTION("INIT") {
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  INIT",
                                        "    METHOD CONFIG", "    SEED 1",
                                        "  ENDINIT", "  INIT",
                                        "    METHOD RANDOM 2", "    SEED 7",
                                        "  ENDINIT", "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        const auto& init = trg.spec.init_strategy_spec;
        REQUIRE(init.chain.size() == 2);
        CHECK(init.chain[0].method == "CONFIG");
        CHECK(init.chain[0].line == 7);
        CHECK(init.chain[1].method == "RANDOM");
        CHECK(init.chain[1].line == 11);
        CHECK(init.seed == 7u);
        /* The block's line is the first block's. */
        CHECK(init.line == 6);
    }
    SECTION("INIT inside OPTIMIZER and inside STRATEGY") {
        /* Both set the starting points of the STRATEGY-level optimiser. */
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  INIT",
                                        "    METHOD CONFIG", "  ENDINIT",
                                        "  STRATEGY", "    LIB DLIB",
                                        "    ALGO BFGS", "    INIT",
                                        "      METHOD RANDOM 2",
                                        "    ENDINIT", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        const auto& init = trg.spec.init_strategy_spec;
        REQUIRE(init.chain.size() == 2);
        CHECK(init.chain[0].method == "CONFIG");
        CHECK(init.chain[1].method == "RANDOM");
        CHECK(init.line == 4);
    }
    SECTION("NOISE") {
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  NOISE",
                                        "    CALIBRATION 3", "  ENDNOISE",
                                        "  NOISE", "    CALIBRATION 4",
                                        "  ENDNOISE", "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        CHECK(trg.spec.noise.calibration_n == 4u);
        CHECK_FALSE(trg.spec.noise.autodetect);
    }
    SECTION("EXPLORE") {
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    TYPE PRESTAGE", "    LIB DLIB",
                                        "    ALGO BFGS", "    EXPLORE",
                                        "      LIB DLIB", "      ALGO GFS",
                                        "    ENDEXPLORE",
                                        "    EXPLORE", "      MAXEVAL 20",
                                        "    ENDEXPLORE", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE(trg.spec.prestage);
        CHECK(trg.spec.prestage->lib == "DLIB");
        CHECK(trg.spec.prestage->algo == "GFS");
        CHECK(trg.spec.prestage->stop.maxeval == 20u);
    }
    SECTION("REFINE") {
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    TYPE PRESTAGE", "    EXPLORE",
                                        "      LIB DLIB", "      ALGO GFS",
                                        "      MAXEVAL 12", "    ENDEXPLORE",
                                        "    REFINE", "      LIB DLIB",
                                        "      ALGO BFGS",
                                        "    ENDREFINE", "    REFINE",
                                        "      MAXEVAL 7", "    ENDREFINE",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        CHECK(trg.spec.lib == "DLIB");
        CHECK(trg.spec.algo == "BFGS");
        CHECK(trg.spec.stop.maxeval == 7u);
    }
    SECTION("RESTART") {
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "    RESTART",
                                        "      AFTER_STAGNATION 5",
                                        "      METHOD RANDOM 1",
                                        "    ENDRESTART", "    RESTART",
                                        "      MAX 2", "      METHOD LHS 2",
                                        "    ENDRESTART", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        const auto& restart = trg.spec.restart;
        CHECK(restart.after_stagnation == 5u);
        CHECK(restart.max == 2u);
        REQUIRE(restart.init_strategy.chain.size() == 2);
        CHECK(restart.init_strategy.chain[0].method == "RANDOM");
        CHECK(restart.init_strategy.chain[1].method == "LHS");
        CHECK(restart.line == 7);
    }
}

/* Pins and fixes: an optimiser key is set once for its optimiser. A second
 * line is refused at its line, naming the line and block of the first:
 *  - in a REFINE block, after the STRATEGY block set it: "ALGO is set on
 *    line 7 in the STRATEGY block and again here in the REFINE block; both
 *    blocks configure the same optimiser, so give it once";
 *  - in a STRATEGY block, after the OPTIMIZER block set it;
 *  - twice in one block, or in a repeated block: "MAXEVAL is set twice in
 *    the EXPLORE block (first on line 12); give it once";
 *  - PARAM with one name twice ("PARAM STEP_FRAC is set twice ..."), the
 *    block-wide STEP twice, and a key set twice in an INNER block.
 * The same key for two optimisers (the STRATEGY-level one and the EXPLORE
 * or INNER one), and PARAM lines of two names, are read.
 * Before the fix: each later line replaced the first's value without a
 * word (REFINE's ALGO LN_BOBYQA replaced the STRATEGY block's LN_SBPLX). */
TEST_CASE("hpotarget block structure: an optimiser key set twice for one "
          "optimiser is refused at the second line",
          "[hpotarget][block_structure]")
{
    const Lines explore = {"    EXPLORE", "      LIB DLIB", "      ALGO GFS",
                           "      MAXEVAL 12", "    ENDEXPLORE"};
    SECTION("STRATEGY and REFINE") {
        const Lines f = kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                      "    TYPE PRESTAGE", "    LIB NLOPT",
                                      "    ALGO LN_SBPLX", "    MAXEVAL 20"} +
                        explore +
                        Lines{"    REFINE", "      ALGO LN_BOBYQA",
                              "    ENDREFINE", "  ENDSTRATEGY",
                              "ENDOPTIMIZER"};
        CHECK(parse_error(f) ==
              at_line(15, "ALGO is set on line 7 in the STRATEGY block and "
                          "again here in the REFINE block; both blocks "
                          "configure the same optimiser, so give it once",
                      "      ALGO LN_BOBYQA"));
    }
    SECTION("OPTIMIZER and STRATEGY") {
        const Lines f = kRoot + Lines{"OPTIMIZER", "  MAXEVAL 5", "  STRATEGY",
                                      "    LIB DLIB", "    ALGO BFGS",
                                      "    MAXEVAL 9", "  ENDSTRATEGY",
                                      "ENDOPTIMIZER"};
        CHECK(parse_error(f) ==
              at_line(8, "MAXEVAL is set on line 4 in the OPTIMIZER block "
                         "and again here in the STRATEGY block; both blocks "
                         "configure the same optimiser, so give it once",
                      "    MAXEVAL 9"));
    }
    SECTION("twice in one block") {
        const Lines f = kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                      "    LIB TADAH", "    ALGO RANDOM",
                                      "    MAXEVAL 9", "    SEED 1",
                                      "    SEED 2", "  ENDSTRATEGY",
                                      "ENDOPTIMIZER"};
        CHECK(parse_error(f) ==
              at_line(9, "SEED is set twice in the STRATEGY block (first on "
                         "line 8); give it once",
                      "    SEED 2"));
    }
    SECTION("in a repeated EXPLORE block") {
        const Lines f = kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                      "    TYPE PRESTAGE", "    LIB DLIB",
                                      "    ALGO BFGS"} +
                        explore +
                        Lines{"    EXPLORE", "      MAXEVAL 20",
                              "    ENDEXPLORE", "  ENDSTRATEGY",
                              "ENDOPTIMIZER"};
        CHECK(parse_error(f) ==
              at_line(14, "MAXEVAL is set twice in the EXPLORE block (first "
                          "on line 11); give it once",
                      "      MAXEVAL 20"));
    }
    SECTION("PARAM of one name, the block-wide STEP, a key of an INNER "
            "block") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB TADAH", "    ALGO ANNEAL",
                                        "    MAXEVAL 9",
                                        "    PARAM STEP_FRAC 0.1",
                                        "    PARAM T0_PROBES 10",
                                        "    PARAM STEP_FRAC 0.2",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(10, "PARAM STEP_FRAC is set twice in the STRATEGY "
                          "block (first on line 8); give it once",
                      "    PARAM STEP_FRAC 0.2"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB NLOPT", "    ALGO LN_SBPLX",
                                        "    MAXEVAL 9", "    STEP 0.1",
                                        "    STEP 0.2", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"}) ==
              at_line(9, "STEP is set twice in the STRATEGY block (first on "
                         "line 8); give it once",
                      "    STEP 0.2"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB NLOPT",
                                        "    ALGO G_MLSL_LDS", "    MAXEVAL 60",
                                        "    INNER", "      LIB NLOPT",
                                        "      ALGO LN_SBPLX", "      MAXEVAL 5",
                                        "      MAXEVAL 6", "    ENDINNER",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(12, "MAXEVAL is set twice in the INNER block (first on "
                          "line 11); give it once",
                      "      MAXEVAL 6"));
    }
    SECTION("the same key for two optimisers, PARAM of two names") {
        PeriodicTable::initialize();
        Context ctx = make_minimal_ctx();
        ScopedTempFile tf(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    TYPE PRESTAGE", "    LIB TADAH",
                                        "    ALGO ANNEAL", "    MAXEVAL 9",
                                        "    PARAM STEP_FRAC 0.1",
                                        "    PARAM T0_PROBES 10"} +
                          explore + Lines{"  ENDSTRATEGY", "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        CHECK(trg.spec.algo == "ANNEAL");
        CHECK(trg.spec.stop.maxeval == 9u);
        REQUIRE(trg.spec.prestage);
        CHECK(trg.spec.prestage->algo == "GFS");
        CHECK(trg.spec.prestage->stop.maxeval == 12u);
        CHECK(trg.spec.param.at("STEP_FRAC") == 0.1);
        CHECK(trg.spec.param.at("T0_PROBES") == 10);
    }
}

TEST_CASE("hpotarget block structure: an optimiser key inside a NOISE block "
          "is refused at its line, naming the keys NOISE takes",
          "[hpotarget][block_structure]")
{
    const Frame noise = frames()[2];   // NOISE on line 6, after LIB DLIB
    REQUIRE(noise.block == "NOISE");
    const auto refused = [&](const std::string& text) {
        return parse_error(insert_after(noise.lines, noise.open, text));
    };
    const auto unknown = [&](const std::string& key, const std::string& text) {
        return at_line(noise.open + 1,
                       "unknown key '" + key + "' inside NOISE block (only "
                       "AUTO and CALIBRATION are valid)",
                       text);
    };
    SECTION("each optimiser key") {
        /* Each set the optimiser of the OPTIMIZER block around it: `LIB
         * NLOPT` here replaced its `LIB DLIB`. */
        const std::map<std::string, std::string> value = {
            {"LIB", "NLOPT"},          {"ALGO", "LN_SBPLX"},
            {"MAXEVAL", "5"},          {"MAXTIME", "10"},
            {"STOPVAL", "0"},          {"FTOL_REL", "1e-6"},
            {"FTOL_ABS", "1e-6"},      {"XTOL_REL", "1e-6"},
            {"XTOL_ABS", "1e-6"},      {"POPULATION", "4"},
            {"STEP", "0.1"},           {"SEED", "3"},
            {"VECTOR_ARR", "3"},       {"THREADS", "1"},
            {"GRAD_STRATEGY", "NUMERICAL"},
            {"PARAM", "fd_step 1e-3"}};
        for (const auto& key : tadah::hpo::opt_attr_keys()) {
            INFO(key);
            const auto v = value.find(key);
            REQUIRE(v != value.end());
            const std::string text = "    " + key + " " + v->second;
            CHECK(refused(text) == unknown(key, text));
        }
    }
    SECTION("any other key") {
        for (const std::string text : {"    LOG_HP 10", "    BOGUS 1"}) {
            INFO(text);
            CHECK(refused(text) ==
                  unknown(text.substr(4, text.find(' ', 4) - 4), text));
        }
    }
    SECTION("AUTO and CALIBRATION") {
        PeriodicTable::initialize();
        Context ctx = make_minimal_ctx();
        ScopedTempFile tf(insert_after(noise.lines, noise.open,
                                       "    AUTO false"));
        HPOTargets trg(ctx, tf.path.string());
        CHECK_FALSE(trg.spec.noise.autodetect);
        CHECK(trg.spec.noise.calibration_n == 0u);
        CHECK(trg.spec.lib == "DLIB");
        CHECK(trg.spec.algo == "BFGS");
    }
}

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
// first: its lines set the same settings and a METHOD line adds to the same
// chain. Every key is set once for what it configures: a second line is
// refused at its line, naming the first, whether it stands in the same
// block, a repeated block or another block that configures the same thing
// -- an optimiser key (LIB, ALGO, MAXEVAL, SEED, PARAM <name>, ...) for its
// optimiser (OPTIMIZER, STRATEGY and REFINE configure the STRATEGY-level
// one), an INIT key for the starting points of its optimiser, a NOISE,
// RESTART or STRATEGY key for the run, and a top-level key that takes one
// value (QFLUSH, OUTPUT, DUMP, ...). A later line replaced the first's
// value without a word. METHOD lines add to their chain and STEP <KEY>
// overrides of other indices layer over the block-wide STEP and over each
// other: neither is a repeat. FAILSCORE, LOSS and PC_SCRIPT_TIMEOUT are set
// again only after a PC_LAMMPS or PC_FAILPROBE line has taken them.
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
#include <optional>
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

/* Pins: a last line that ends with the continuation backslash is refused
 * at that line, with its text joined so far and its final backslash, as
 * "the line ends with \, which continues it on the next line, and the file
 * has no next line", the frame naming the last physical line (a line whose
 * continuation ends with a backslash too); a target file that cannot be
 * opened is named with the path as given and --hpotarget.
 * Before the fix: the first two give "unfinished line <n>" with no frame,
 * and the missing file "HPO file <path> not found". */
TEST_CASE("hpotarget block structure: a continued last line is refused at its "
          "line, and a target file that cannot be opened names the path and "
          "--hpotarget",
          "[hpotarget][block_structure]")
{
    const std::string what =
        "the line ends with \\, which continues it on the next line, and the "
        "file has no next line";
    SECTION("the last line ends with a backslash") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "ENDOPTIMIZER",
                                        "FAILSCORE \\"}) ==
              at_line(kRoot.size() + 5, what, "FAILSCORE \\"));
    }
    SECTION("a continued line whose last line ends with a backslash too") {
        /* The frame names the last physical line of the logical line. */
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "ENDOPTIMIZER",
                                        "FAILSCORE \\", "  \\"}) ==
              at_line(kRoot.size() + 6, what, "FAILSCORE   \\"));
    }
    SECTION("a target file that cannot be opened") {
        PeriodicTable::initialize();
        Context ctx = make_minimal_ctx();
        const std::string path =
            (fs::temp_directory_path() / (unique_name("tadah_hbs_none_") +
                                          ".hpo")).string();
        std::string msg;
        try {
            HPOTargets trg(ctx, path);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        CHECK(msg == "the target file " + path +
                         " (--hpotarget) cannot be opened");
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
                                        "      AFTER_STAGNATION 5",
                                        "      MAX 2",
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
                                        "    METHOD CONFIG", "    K_TOTAL 2",
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
        CHECK(init.k_total == std::optional<std::size_t>(2));
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
                                        "    CALIBRATION 4", "  ENDNOISE",
                                        "  NOISE", "    AUTO true",
                                        "  ENDNOISE", "ENDOPTIMIZER"});
        HPOTargets trg(ctx, tf.path.string());
        CHECK(trg.spec.noise.calibration_n == 4u);
        CHECK(trg.spec.noise.autodetect);
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
 *    blocks configure the same optimiser, so each key is given once";
 *  - in a STRATEGY block, after the OPTIMIZER block set it;
 *  - twice in one block, or in a repeated block: "MAXEVAL is set twice in
 *    the EXPLORE block (first on line 12); each key is given once";
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
                          "configure the same optimiser, so each key is given once",
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
                         "configure the same optimiser, so each key is given once",
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
                         "line 8); each key is given once",
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
                          "on line 11); each key is given once",
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
                          "block (first on line 8); each key is given once",
                      "    PARAM STEP_FRAC 0.2"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB NLOPT", "    ALGO LN_SBPLX",
                                        "    MAXEVAL 9", "    STEP 0.1",
                                        "    STEP 0.2", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"}) ==
              at_line(9, "STEP is set twice in the STRATEGY block (first on "
                         "line 8); each key is given once",
                      "    STEP 0.2"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    LIB NLOPT",
                                        "    ALGO G_MLSL_LDS", "    MAXEVAL 60",
                                        "    INNER", "      LIB NLOPT",
                                        "      ALGO LN_SBPLX", "      MAXEVAL 5",
                                        "      MAXEVAL 6", "    ENDINNER",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(12, "MAXEVAL is set twice in the INNER block (first on "
                          "line 11); each key is given once",
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

/* Pins and fixes: a key of an INIT, NOISE, RESTART or STRATEGY block is set
 * once for what it configures. A second line is refused at its line, naming
 * the line of the first ("SEED is set twice in the INIT block (first on
 * line 7); each key is given once"): in one block, in a repeated block, and
 * in the INIT blocks inside OPTIMIZER and inside STRATEGY, which both set
 * the starting points of the STRATEGY-level optimiser; LOG_HP, which the
 * OPTIMIZER and STRATEGY blocks take, once for the search space. METHOD
 * lines in two INIT blocks add to one chain (the case above).
 * Before the fix: the later line replaced the first's value without a
 * word. */
TEST_CASE("hpotarget block structure: a key of an INIT, NOISE, RESTART or "
          "STRATEGY block set twice is refused at the second line",
          "[hpotarget][block_structure]")
{
    const std::string once = "; each key is given once";
    SECTION("INIT: in one block, in a repeated block, inside OPTIMIZER and "
            "inside STRATEGY") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  INIT",
                                        "    SEED 1", "    SEED 2",
                                        "  ENDINIT", "ENDOPTIMIZER"}) ==
              at_line(8, "SEED is set twice in the INIT block (first on "
                         "line 7)" + once, "    SEED 2"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  INIT",
                                        "    K_TOTAL 2", "  ENDINIT", "  INIT",
                                        "    K_TOTAL 3", "  ENDINIT",
                                        "ENDOPTIMIZER"}) ==
              at_line(10, "K_TOTAL is set twice in the INIT block (first on "
                          "line 7)" + once, "    K_TOTAL 3"));
        /* two INIT blocks inside two blocks: both named (it said "set
         * twice in the INIT block", as for one block) */
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  INIT",
                                        "    WARM_ON_MISMATCH warn",
                                        "  ENDINIT", "  STRATEGY",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "    INIT",
                                        "      WARM_ON_MISMATCH error",
                                        "    ENDINIT", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"}) ==
              at_line(11, "WARM_ON_MISMATCH is set on line 5 in the INIT "
                          "block inside OPTIMIZER and again here in the INIT "
                          "block inside STRATEGY; both blocks configure the "
                          "same starting points, so each key is given once",
                      "      WARM_ON_MISMATCH error"));
        /* the INIT block inside REFINE sets the same starting points */
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  INIT",
                                        "    SEED 4", "  ENDINIT",
                                        "  STRATEGY", "    TYPE PRESTAGE",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "    EXPLORE", "      LIB DLIB",
                                        "      ALGO GFS", "      MAXEVAL 4",
                                        "    ENDEXPLORE", "    REFINE",
                                        "      INIT", "        SEED 5",
                                        "      ENDINIT", "    ENDREFINE",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(18, "SEED is set on line 5 in the INIT block inside "
                          "OPTIMIZER and again here in the INIT block inside "
                          "REFINE; both blocks configure the same starting "
                          "points, so each key is given once",
                      "        SEED 5"));
    }
    SECTION("NOISE") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  NOISE",
                                        "    CALIBRATION 3", "  ENDNOISE",
                                        "  NOISE", "    CALIBRATION 4",
                                        "  ENDNOISE", "ENDOPTIMIZER"}) ==
              at_line(10, "CALIBRATION is set twice in the NOISE block "
                          "(first on line 7)" + once, "    CALIBRATION 4"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LIB DLIB",
                                        "  ALGO BFGS", "  NOISE",
                                        "    AUTO true", "    AUTO false",
                                        "  ENDNOISE", "ENDOPTIMIZER"}) ==
              at_line(8, "AUTO is set twice in the NOISE block (first on "
                         "line 7)" + once, "    AUTO false"));
    }
    SECTION("RESTART") {
        const Lines head = kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                         "    LIB DLIB", "    ALGO BFGS",
                                         "    RESTART",
                                         "      AFTER_STAGNATION 5"};
        CHECK(parse_error(head + Lines{"      MAX 2", "      MAX 3",
                                       "    ENDRESTART", "  ENDSTRATEGY",
                                       "ENDOPTIMIZER"}) ==
              at_line(10, "MAX is set twice in the RESTART block (first on "
                          "line 9)" + once, "      MAX 3"));
        CHECK(parse_error(head + Lines{"    ENDRESTART", "    RESTART",
                                       "      AFTER_STAGNATION 6",
                                       "    ENDRESTART", "  ENDSTRATEGY",
                                       "ENDOPTIMIZER"}) ==
              at_line(11, "AFTER_STAGNATION is set twice in the RESTART "
                          "block (first on line 8)" + once,
                      "      AFTER_STAGNATION 6"));
        CHECK(parse_error(head + Lines{"      SEED 4", "      SEED 5",
                                       "    ENDRESTART", "  ENDSTRATEGY",
                                       "ENDOPTIMIZER"}) ==
              at_line(10, "SEED is set twice in the RESTART block (first on "
                          "line 9)" + once, "      SEED 5"));
    }
    SECTION("STRATEGY") {
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    TYPE MULTISTART",
                                        "    TYPE PLAIN", "    LIB DLIB",
                                        "    ALGO BFGS", "  ENDSTRATEGY",
                                        "ENDOPTIMIZER"}) ==
              at_line(6, "TYPE is set twice in the STRATEGY block (first on "
                         "line 5)" + once, "    TYPE PLAIN"));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                        "    TYPE MULTISTART",
                                        "    N_STARTS 2", "    N_STARTS 3",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(7, "N_STARTS is set twice in the STRATEGY block (first "
                         "on line 6)" + once, "    N_STARTS 3"));
    }
    SECTION("LOG_HP, a search-space key of the OPTIMIZER and STRATEGY "
            "blocks") {
        /* It warned 'LOG_HP given more than once; the last value (...)
         * wins'. */
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "  LOG_HP 10",
                                        "  STRATEGY", "    LOG_HP 100",
                                        "    LIB DLIB", "    ALGO BFGS",
                                        "  ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(6, "LOG_HP is set on line 4 in the OPTIMIZER block and "
                         "again here in the STRATEGY block; both blocks "
                         "configure the same search space, so each key is "
                         "given once", "    LOG_HP 100"));
    }
}

/* Fixes: a top-level key that takes one value is set once, as a key of a
 * block is: QFLUSH, OUTPUT, BOUTPUT, DIGITS, DUMP, BDUMP,
 * FAILSCORE_GRADING, BUMPGATE, PC_ISOLATION and PC_WORKER_TIMEOUT. A second
 * line is refused at its line, naming the first ("QFLUSH is set twice
 * (first on line 3); each key is given once").
 * Before the fix: the later line replaced the first without a word. */
TEST_CASE("hpotarget block structure: a top-level key that takes one value "
          "set twice is refused at the second line",
          "[hpotarget][block_structure]")
{
    const fs::path dir = fs::temp_directory_path() / unique_name("tadah_hbs_d");
    fs::create_directories(dir);   // DUMP and BDUMP create their directory
    struct Case { std::string key, first, second; };
    for (const auto& c : std::vector<Case>{
             {"QFLUSH", "4", "8"}, {"OUTPUT", "2", "3"},
             {"BOUTPUT", "2", "3"}, {"DIGITS", "6", "7"},
             {"DUMP", "1 " + (dir / "a").string(), "2 " + (dir / "b").string()},
             {"BDUMP", "1 " + (dir / "c").string(), "2 " + (dir / "d").string()},
             {"FAILSCORE_GRADING", "ON", "OFF"},
             {"BUMPGATE", "STRICT", "CANDIDATE"},
             {"PC_ISOLATION", "THREAD", "THREAD"},
             {"PC_WORKER_TIMEOUT", "60", "70"}}) {
        INFO(c.key);
        const std::string second = c.key + " " + c.second;
        CHECK(parse_error(kRoot + Lines{c.key + " " + c.first, "OPTIMIZER",
                                        "  LIB DLIB", "  ALGO BFGS",
                                        "ENDOPTIMIZER", second}) ==
              at_line(8, c.key + " is set twice (first on line 3); each key "
                             "is given once", second));
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

/* Fixes: FAILSCORE, LOSS and PC_SCRIPT_TIMEOUT (its alias MAXTIME) are set
 * once for each line that reads them: a second line is read only after
 * such a line has read the first, and is refused otherwise, naming the
 * first and the lines that read the key ("LOSS is set twice (first on
 * line 3) and no line between reads the first (a --varloss or
 * PC_OBSERVABLE term that names no loss function); each key is given
 * once"). The run-wide uses (the validation metrics' loss, the ceiling of
 * every loss) take the last line.
 * Before the fix: a second line with no PC_LAMMPS line between replaced
 * the first without a word, for the validation metrics and the run's
 * ceiling too; then the message said "and no PC_LAMMPS or PC_OBSERVABLE
 * (PC_FAILPROBE) line between takes the first". */
TEST_CASE("hpotarget block structure: FAILSCORE, LOSS or PC_SCRIPT_TIMEOUT "
          "set again before a PC_LAMMPS line takes it is refused",
          "[hpotarget][block_structure]")
{
    const fs::path script =
        fs::temp_directory_path() / (unique_name("tadah_hbs_s") + ".in");
    { std::ofstream os(script); os << "variable xyz equal 0.0\n"; }
    const std::string pc_lammps = "PC_LAMMPS --script " + script.string() +
                                  " --varloss xyz 1.0 100 --outvar xyz";
    const Lines block = {"OPTIMIZER", "  LIB DLIB", "  ALGO GFS",
                         "  MAXEVAL 4", "ENDOPTIMIZER"};
    struct Case { std::string first, second, message; };
    for (const auto& c : std::vector<Case>{
             {"LOSS L1", "LOSS L2",
              "LOSS is set twice (first on line 3)"},
             {"FAILSCORE 1e9", "FAILSCORE 1e6",
              "FAILSCORE is set twice (first on line 3)"},
             {"PC_SCRIPT_TIMEOUT 60", "PC_SCRIPT_TIMEOUT 90",
              "PC_SCRIPT_TIMEOUT is set twice (first on line 3)"},
             {"PC_SCRIPT_TIMEOUT 60", "MAXTIME 90",
              "MAXTIME is set twice (first on line 3, as "
              "PC_SCRIPT_TIMEOUT)"}}) {
        INFO(c.second);
        /* LOSS is read by a term that names no loss function, FAILSCORE
         * and PC_SCRIPT_TIMEOUT by a PC_LAMMPS or PC_FAILPROBE line without
         * its own --failscore or --maxtime. */
        const std::string readers =
            c.first.rfind("LOSS", 0) == 0
                ? "a --varloss or PC_OBSERVABLE term that names no loss "
                  "function"
                : c.first.rfind("FAILSCORE", 0) == 0
                      ? "a PC_LAMMPS or PC_FAILPROBE line without --failscore"
                      : "a PC_LAMMPS or PC_FAILPROBE line without --maxtime";
        CHECK(parse_error(kRoot + Lines{c.first, c.second} + block) ==
              at_line(4, c.message + " and no line between reads the first (" +
                             readers + "); each key is given once",
                      c.second));
        /* a PC_LAMMPS line between reads the first */
        CHECK(parse_error(kRoot + Lines{c.first, pc_lammps, c.second,
                                        pc_lammps} + block) == "");
    }
    std::error_code ec;
    fs::remove(script, ec);
}

/* A potential's values its descriptors do not read are left out of it for
 * the run (D_mJoin::expand_grids): a single descriptor keeps its largest
 * RCUT2B value, its cutoff, so OPTIM indices address the values kept. An
 * index past them is refused saying so, naming the values left out and
 * the index the kept value has: a 1.5.0 target file's 'OPTIM RCUT2B (2)'
 * on 'RCUT2B 4 6' searched the cutoff 6, which is index 1.
 * Before the fix: 'OPTIM RCUT2B (2): Out of bound index: 2', without why. */
TEST_CASE("hpotarget: an OPTIM index past the values kept of a potential "
          "says why", "[hpotarget][potential]")
{
    PeriodicTable::initialize();
    Context ctx = make_minimal_ctx();
    ctx.remove("RCUT2B");
    ctx.add("RCUT2B", "4 6");
    ctx.add("WEIGHTS", "0.1 0.2");
    ScopedTempFile tf(Lines{"OPTIM RCUT2B (2) 5.5 6.5", "PC_ERMSE 0 1",
                            "OPTIMIZER", "  LIB NLOPT", "  ALGO LN_SBPLX",
                            "  MAXEVAL 3", "ENDOPTIMIZER"});
    std::string what;
    try {
        HPOTargets trg(ctx, tf.path.string());
    } catch (const std::exception& e) {
        what = e.what();
    }
    /* Before the fix: "OPTIM RCUT2B (2): Out of bound index: 2 (RCUT2B has 1
     * value: of the potential's RCUT2B 4 6, ...". */
    CHECK_THAT(what, Catch::Contains(
                         "OPTIM RCUT2B (2): RCUT2B has 1 value that the "
                         "descriptors read, so the index of OPTIM RCUT2B is 1 "
                         "(of the potential's RCUT2B 4 6, the descriptors do "
                         "not read 4, which was left out, and OPTIM indices "
                         "address the values kept, 6; the potential's value "
                         "at index 2, 6, is index 1)"));
}

/* The grids of a single DM_Poly line whose channel function reads SEMBFUNC
 * are expanded before the OPTIM lines are read: OPTIM CGRIDMB (2) of
 * 'CGRIDMB LIN 3 2 4.5' starts at the second centre, 3.25.
 * Before the fix: the line kept its generator as written, so CGRIDMB (2)
 * was the generator's start, 2, and the search moved it. */
TEST_CASE("hpotarget: OPTIM addresses the expanded grid of a DM_PS_SQRT "
          "line", "[hpotarget][dm_poly]")
{
    PeriodicTable::initialize();
    Context ctx;
    ctx.add("ATOM", "Ta");
    ctx.add("WATOM", 1.0);
    ctx.add("DBFILE", "dummy.db");
    ctx.add("INIT2B", "false");
    ctx.add("INITMB", "true");
    ctx.add("TYPEMB", "DM_PS_SQRT 1 3 3 Ta Ta");
    ctx.add("CGRIDMB", "LIN 3 2.0 4.5");
    ctx.add("SGRIDMB", "1.0 1.0 1.0");
    ctx.add("RCTYPEMB", "Cut_Cos");
    ctx.add("RCUTMB", 5.0);
    ctx.add("VERBOSE", 0);
    ctx.add("MODEL", "M_BLR BF_Linear");
    ScopedTempFile tf(Lines{"OPTIM CGRIDMB (2) 1.0 5.0", "PC_ERMSE 0 1",
                            "OPTIMIZER", "  LIB NLOPT", "  ALGO LN_SBPLX",
                            "  MAXEVAL 3", "ENDOPTIMIZER"});
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.optim_init.size() == 1);
    REQUIRE(trg.spec.optim_init.front().size() == 1);
    CHECK(trg.spec.optim_init.front()[0] == 3.25);
    REQUIRE(ctx.size("CGRIDMB") == 3);
    CHECK(ctx.get<double>("CGRIDMB", 2) == 4.5);
}

/* Fixes: a STEP <KEY> override is set once for its key and indices: an
 * identical one is refused at its line, naming the first ("STEP WATOM(1)
 * is set twice in the STRATEGY block (first on line 8); each key is given
 * once"; "WATOM(1)" and "WATOM (1)" are one override). An override of
 * other or no indices layers over it, and the block-wide STEP under both.
 * Before the fix: the later override replaced the first without a word. */
TEST_CASE("hpotarget block structure: a STEP <KEY> override given twice for "
          "the same indices is refused at the second line",
          "[hpotarget][block_structure][step]")
{
    const Lines head = kRoot + Lines{"OPTIMIZER", "  STRATEGY",
                                     "    LIB NLOPT", "    ALGO LN_SBPLX",
                                     "    MAXEVAL 9"};   // lines 3-7
    const Lines tail = {"  ENDSTRATEGY", "ENDOPTIMIZER"};
    CHECK(parse_error(head + Lines{"    STEP WATOM(1) 0.05",
                                   "    STEP WATOM (1) 0.3"} + tail) ==
          at_line(9, "STEP WATOM(1) is set twice in the STRATEGY block "
                     "(first on line 8); each key is given once",
                  "    STEP WATOM (1) 0.3"));
    CHECK(parse_error(head + Lines{"    STEP WATOM 0.2", "    STEP WATOM 0.3"} +
                      tail) ==
          at_line(9, "STEP WATOM is set twice in the STRATEGY block (first "
                     "on line 8); each key is given once",
                  "    STEP WATOM 0.3"));
    /* layering: the block-wide STEP, all indices of WATOM, then index 1 */
    CHECK(parse_error(head + Lines{"    STEP 0.1", "    STEP WATOM 0.2",
                                   "    STEP WATOM(1) 0.05"} + tail) == "");
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
    /* The message names the line the block opened on and its END line
     * (it was "unknown key 'K' inside NOISE block (only AUTO and
     * CALIBRATION are valid)"). */
    const auto unknown = [&](const std::string& key, const std::string& text) {
        return at_line(noise.open + 1,
                       "unknown key '" + key + "' inside the NOISE block "
                       "opened on line " + std::to_string(noise.open) +
                       " (only AUTO and CALIBRATION are valid; ENDNOISE "
                       "closes it)",
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

// test_hpotarget_engine.cpp
//
// The parser of the target file (src/hpotarget_parser.cpp and
// src/hpotarget_handlers.cpp), the registry-driven one, which `tadah hpo`
// reads with: each file below is read by HPOTargets and gives the refusal
// the order of the parse gives (blueprint/README.md, the `[structure]`
// orders), or a description of what it read (describe_parse,
// hpo_targets_testing.h).
//
// The orders the cases pin, each where a line or a file has two faults:
//   - the steps of a key's line by its place (structure::line_steps_*,
//     Key::steps): a repeated line whose value is wrong, for a top-level
//     key, a metric line, a key of the optimiser group and the keys of
//     INIT, RESTART, NOISE, STRATEGY and LOG_HP;
//   - the check order of a line (structure::check_order): a removed
//     keyword inside INNER is refused as removed; LOG_HP inside INIT is an
//     unknown key of INIT, inside INNER and EXPLORE and at the top level
//     its own refusal;
//   - the checks of ENDSTRATEGY (Block::close_steps): a key of another
//     TYPE, a MULTISTART without N_STARTS and a RESTART block that never
//     restarts, the first refused;
//   - a PC_FAILPROBE line whose --gate value and a LAMMPS option are both
//     wrong: the --gate value is read first;
//   - the checks at the end of the file (structure::eof_order): two faults
//     of two checks next to each other in the order.
// And: the registry-driven parser is the one parser of the target file
// (ParsedTarget has no read(), the hand-written parser, and the test seam
// chooses no parser).
//
// Tag [hpotarget_engine].
// Before the fix: ParsedTarget::read() exists and hpo_targets_testing.h
// chooses between it and the registry-driven parser; the case "hpotarget
// engine: the registry-driven parser is the one parser of the target file"
// fails.

#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_targets_testing.h>

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/strict_parse.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using Lines = std::vector<std::string>;

/* A scratch directory of this process, removed at exit. */
const fs::path& scratch()
{
    static const struct Dir {
        fs::path path;
        Dir()
            : path(fs::temp_directory_path() /
                   ("tadah_hpotarget_engine_" + std::to_string(::getpid()) +
                    "_" +
                    std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count())))
        {
            fs::create_directories(path);
        }
        ~Dir() { std::error_code ec; fs::remove_all(path, ec); }
    } dir;
    return dir.path;
}

/* The smallest training configuration whose model builds. */
Context minimal_ctx()
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

/* The target file the cases write and read, which the refusals name. */
const std::string& target_path()
{
    static const std::string p = (scratch() / "target.hpo").string();
    return p;
}

void write_target(const Lines& lines)
{
    std::ofstream os(target_path());
    REQUIRE(os.good());
    for (const auto& l : lines) os << l << "\n";
}

/* What the parser gives for the file `lines`: its refusal, or "" and the
 * description of what it read. Each read takes its own copy of the
 * training configuration (a key the parse reads gets its default). */
struct Outcome {
    std::string error;
    std::string dump;
};

Outcome read_file(const Lines& lines, const Context& base = minimal_ctx())
{
    tadah::core::PeriodicTable::initialize();
    write_target(lines);
    Context ctx = base;
    Outcome out;
    try {
        const tadah::hpo::HPOTargets t(ctx, target_path());
        out.dump = tadah::hpo::testing::describe_parse(t);
    } catch (const std::exception& e) {
        out.error = e.what();
    }
    return out;
}

/* The refusal of line n of the file, whose text is `text`. */
std::string at_line(std::size_t n, const std::string& what,
                    const std::string& text)
{
    return tadah::hpo::config_line_error(target_path(), n, what, text);
}

/* The parser refuses the file `lines` with `expected`. */
void refuses(const Lines& lines, const std::string& expected,
             const Context& ctx = minimal_ctx())
{
    CAPTURE(lines);
    CHECK(read_file(lines, ctx).error == expected);
}

/* The refusal the parser gives for `lines` ("" when it reads the file). */
std::string refusal(const Lines& lines, const Context& ctx = minimal_ctx())
{
    CAPTURE(lines);
    return read_file(lines, ctx).error;
}

Lines operator+(Lines a, const Lines& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

/* Lines 1-2 of most files below: a search parameter and a loss. */
const Lines kRoot = {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1"};
const Lines kOptimizer = {"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                          "MAXEVAL 10", "ENDOPTIMIZER"};

/* A LAMMPS script that sets the variables of the lines below, and one that
 * reads the variable a. */
const std::string& probe_script()
{
    static const std::string p = [] {
        const fs::path f = scratch() / "engine_probe.in";
        std::ofstream os(f);
        os << "variable E equal 1.0\nvariable S equal 0.0\n"
              "variable E2 equal 2.0\n";
        return f.string();
    }();
    return p;
}

const std::string& reads_a_script()
{
    static const std::string p = [] {
        const fs::path f = scratch() / "engine_reads_a.in";
        std::ofstream os(f);
        os << "variable E3 equal v_a+1\n";
        return f.string();
    }();
    return p;
}

/* A fault of a target file: line `index` (from 0) is `faulty` where the
 * fault is, and `fixed` where it is not. */
struct Fault {
    std::size_t index;
    std::string faulty;
    std::string fixed;
};

Lines with_faults(Lines base, const std::vector<Fault>& on,
                  const std::vector<Fault>& off)
{
    for (const Fault& f : off) base.at(f.index) = f.fixed;
    for (const Fault& f : on) base.at(f.index) = f.faulty;
    return base;
}

/* Two faults of one file, the first refused by an earlier check than the
 * second: the parser refuses the file with both as the file with the
 * first alone, and the file with the second alone with another refusal. */
struct TwoFaults {
    std::string label;
    Lines base;
    std::vector<Fault> first;
    std::vector<Fault> second;
};

void check_first_wins(const TwoFaults& t)
{
    CAPTURE(t.label);
    const std::string first = refusal(with_faults(t.base, t.first, t.second));
    const std::string second =
        refusal(with_faults(t.base, t.second, t.first));
    REQUIRE(!first.empty());
    REQUIRE(!second.empty());
    std::vector<Fault> both = t.first;
    both.insert(both.end(), t.second.begin(), t.second.end());
    refuses(with_faults(t.base, both, {}), first);
}

}  // namespace

/* Pins: the steps of a key's line follow its place. A top-level key set
 * once and FAILSCORE, given again with a wrong value or a value too many,
 * are refused as given again (repeat, then value); a metric line without
 * indices reads its target and weight, then its repeat rule, then its
 * options (PC_EMAE: Key::steps), and a metric line with a dataset index
 * set reads its options before ValidationMetricTargets::add refuses a
 * repeated set; a key of the optimiser group reads its value, then its
 * repeat rule, then the leftover check; the keys of INIT, RESTART, NOISE,
 * STRATEGY and LOG_HP read their value with its leftover check, then their
 * repeat rule.
 * Before the fix: the registry-driven parser does not exist. */
TEST_CASE("hpotarget engine: a repeated line whose value is wrong is "
          "refused by the first step of its place", "[hpotarget_engine]")
{
    /* The top level: repeat, value. */
    refuses(kRoot + Lines{"QFLUSH 1", "QFLUSH x"} + kOptimizer,
            at_line(4, "QFLUSH is set twice (first on line 3); each key "
                       "is given once", "QFLUSH x"));
    refuses(kRoot + Lines{"QFLUSH 1", "QFLUSH 1 2"} + kOptimizer,
            at_line(4, "QFLUSH is set twice (first on line 3); each key "
                       "is given once", "QFLUSH 1 2"));
    refuses(kRoot + Lines{"FAILSCORE 10", "FAILSCORE x"} + kOptimizer,
            at_line(4, "FAILSCORE is set twice (first on line 3) and no "
                       "line between reads the first (a PC_LAMMPS or "
                       "PC_FAILPROBE line without --failscore); each key "
                       "is given once", "FAILSCORE x"));
    refuses(kRoot + Lines{"MAXTIME 60", "PC_SCRIPT_TIMEOUT x"} +
                kOptimizer,
            at_line(4, "PC_SCRIPT_TIMEOUT is set twice (first on line 3, "
                       "as MAXTIME) and no line between reads the first "
                       "(a PC_LAMMPS or PC_FAILPROBE line without "
                       "--maxtime); the key is given again only after such "
                       "a line",
                    "PC_SCRIPT_TIMEOUT x"));

    /* A metric line: value, repeat, options. */
    refuses(kRoot + Lines{"PC_EMAE 1 1", "PC_EMAE x 1"} + kOptimizer,
            at_line(4, "PC_EMAE <target> must be a finite number, got "
                       "'x'", "PC_EMAE x 1"));
    refuses(kRoot + Lines{"PC_EMAE 1 1", "PC_EMAE 1 1 --bogus 1"} +
                kOptimizer,
            at_line(4, "PC_EMAE is set twice (first on line 3); each key "
                       "is given once", "PC_EMAE 1 1 --bogus 1"));
    refuses(kRoot + Lines{"PC_ERMSE 0 1 --bogus"} + kOptimizer,
            at_line(3, "PC_ERMSE: unknown flag '--bogus'",
                    "PC_ERMSE 0 1 --bogus"));

    /* A key of the optimiser group: value, repeat, leftover. */
    const Lines strategy_head = kRoot + Lines{"OPTIMIZER", "STRATEGY",
                                              "LIB NLOPT", "ALGO LN_SBPLX"};
    const Lines strategy_tail = {"ENDSTRATEGY", "ENDOPTIMIZER"};
    refuses(strategy_head + Lines{"MAXEVAL 5", "MAXEVAL x"} +
                strategy_tail,
            at_line(8, "MAXEVAL must be an integer from 1 to 2147483647, "
                       "got 'x'", "MAXEVAL x"));
    refuses(strategy_head + Lines{"MAXEVAL 5", "MAXEVAL 5 6"} +
                strategy_tail,
            at_line(8, "MAXEVAL is set twice in the STRATEGY block (first "
                       "on line 7); each key is given once",
                    "MAXEVAL 5 6"));
    refuses(strategy_head + Lines{"MAXEVAL 5 6"} + strategy_tail,
            at_line(7, "MAXEVAL takes no further value here (unexpected "
                       "'6')", "MAXEVAL 5 6"));

    /* INIT: value with its leftover, then repeat. */
    const Lines init_head = kRoot + Lines{"OPTIMIZER", "LIB NLOPT",
                                          "ALGO LN_SBPLX", "MAXEVAL 10",
                                          "INIT", "METHOD RANDOM 2"};
    const Lines init_tail = {"ENDINIT", "ENDOPTIMIZER"};
    refuses(init_head + Lines{"SEED 1", "SEED x"} + init_tail,
            at_line(10, "SEED must be an integer >= 0, got 'x'",
                    "SEED x"));
    refuses(init_head + Lines{"SEED 1", "SEED 1 2"} + init_tail,
            at_line(10, "SEED takes one value (unexpected '2')",
                    "SEED 1 2"));
    refuses(init_head + Lines{"SEED 1", "SEED 2"} + init_tail,
            at_line(10, "SEED is set twice in the INIT block (first on "
                        "line 9); each key is given once", "SEED 2"));

    /* NOISE and RESTART. */
    refuses(kRoot + Lines{"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                          "MAXEVAL 10", "NOISE", "CALIBRATION 2",
                          "CALIBRATION 1", "ENDNOISE", "ENDOPTIMIZER"},
            at_line(9, "CALIBRATION in the NOISE block must be 0 (off) or "
                       ">= 2, got '1'",
                    "CALIBRATION 1"));
    refuses(strategy_head + Lines{"MAXEVAL 10", "RESTART",
                                  "AFTER_STAGNATION 5", "MAX 2", "MAX 0",
                                  "ENDRESTART"} + strategy_tail,
            at_line(11, "MAX must be an integer >= 1, got '0' (0 never "
                        "restarts the search; a search without restarts "
                        "has no RESTART block)", "MAX 0"));

    /* LOG_HP, set for the search space by OPTIMIZER and STRATEGY. */
    refuses(kRoot + Lines{"OPTIMIZER", "LOG_HP 10", "STRATEGY",
                          "LIB NLOPT", "ALGO LN_SBPLX", "MAXEVAL 10",
                          "LOG_HP 0.5", "ENDSTRATEGY", "ENDOPTIMIZER"},
            at_line(9, "LOG_HP must be a finite ratio >= 1 (a dimension "
                       "is log-scaled when high/low >= LOG_HP and low > "
                       "0), got '0.5'", "LOG_HP 0.5"));
    refuses(kRoot + Lines{"OPTIMIZER", "LOG_HP 10", "STRATEGY",
                          "LIB NLOPT", "ALGO LN_SBPLX", "MAXEVAL 10",
                          "LOG_HP 20", "ENDSTRATEGY", "ENDOPTIMIZER"},
            at_line(9, "LOG_HP is set on line 4 in the OPTIMIZER block "
                       "and again here in the STRATEGY block; both "
                       "blocks configure the same search space, so each "
                       "key is given once", "LOG_HP 20"));

    /* STRATEGY. */
    refuses(strategy_head + Lines{"MAXEVAL 10", "TYPE PRESTAGE", "TOPK 2",
                                  "TOPK 0"} + strategy_tail,
            at_line(10, "TOPK must be an integer from 1 to " +
                            std::to_string(tadah::hpo::kMaxStartingPoints) +
                            ", got '0'", "TOPK 0"));
}

/* Pins: a removed keyword is refused as removed in every block, INNER
 * included, before the refusals of INNER; a key INNER refuses is refused
 * with its reason. Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: a removed keyword inside INNER is refused as "
          "removed", "[hpotarget_engine]")
{
    const Lines head = kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB NLOPT",
                                     "ALGO G_MLSL_LDS", "MAXEVAL 10",
                                     "INNER", "LIB NLOPT"};
    const Lines tail = {"ALGO LN_BOBYQA", "ENDINNER", "ENDSTRATEGY",
                        "ENDOPTIMIZER"};
    refuses(head + Lines{"LOCAL"} + tail,
            at_line(10, "'LOCAL' was removed from the grammar: renamed "
                        "to INNER, the block of the local search that an "
                        "NLopt MLSL or AUGLAG algorithm runs.", "LOCAL"));
    refuses(head + Lines{"INIT_K 2"} + tail,
            at_line(10, "'INIT_K' was removed from the grammar: renamed "
                        "to K_TOTAL inside an INIT block.", "INIT_K 2"));
    refuses(head + Lines{"SEED 1"} + tail,
            at_line(10, "SEED is not accepted inside an INNER block: "
                        "NLopt draws the INNER search from the random "
                        "numbers of the enclosing run; the enclosing "
                        "STRATEGY block takes SEED", "SEED 1"));
    refuses(head + Lines{"FOO 1"} + tail,
            at_line(10, "unknown key 'FOO' inside the INNER block opened "
                        "on line 8 (only LIB/ALGO/MAXEVAL/MAXTIME/FTOL_REL"
                        "/FTOL_ABS/XTOL_REL/XTOL_ABS/POPULATION/STEP/"
                        "VECTOR_ARR/THREADS/PARAM are valid; ENDINNER "
                        "closes it)", "FOO 1"));
}

/* Pins: LOG_HP inside INIT is an unknown key of INIT (INIT refuses its
 * unknown keys before the LOG_HP check); inside INNER and EXPLORE and at
 * the top level it gets its own refusal, which names the block.
 * Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: LOG_HP in INIT, INNER, EXPLORE and at the top "
          "level", "[hpotarget_engine]")
{
    const std::string written = "LOG_HP is written once, inside OPTIMIZER or "
                                "STRATEGY: it holds for the whole search space";
    refuses(kRoot + Lines{"LOG_HP 10"} + kOptimizer,
            at_line(3, written + " and is not a top-level key",
                    "LOG_HP 10"));
    refuses(kRoot + Lines{"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                          "MAXEVAL 10", "INIT", "LOG_HP 10", "ENDINIT",
                          "ENDOPTIMIZER"},
            at_line(8, "unknown key 'LOG_HP' inside the INIT block opened "
                       "on line 7 (ENDINIT closes it)", "LOG_HP 10"));
    refuses(kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB NLOPT",
                          "ALGO G_MLSL_LDS", "MAXEVAL 10", "INNER",
                          "LOG_HP 10", "ENDINNER", "ENDSTRATEGY",
                          "ENDOPTIMIZER"},
            at_line(9, written + ", and the INNER block does not take it",
                    "LOG_HP 10"));
    refuses(kRoot + Lines{"OPTIMIZER", "STRATEGY", "TYPE PRESTAGE",
                          "LIB NLOPT", "ALGO LN_SBPLX", "MAXEVAL 10",
                          "EXPLORE", "LOG_HP 10", "ENDEXPLORE",
                          "ENDSTRATEGY", "ENDOPTIMIZER"},
            at_line(10, written + ", and the EXPLORE block does not take "
                                  "it", "LOG_HP 10"));
}

/* Pins: ENDSTRATEGY checks the keys of one TYPE, then a MULTISTART
 * without N_STARTS, then a RESTART block that never restarts; a file with
 * the three faults is refused at the key of the other TYPE (its line), one
 * without it at the TYPE line, one with N_STARTS at the last ENDRESTART
 * line. Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: ENDSTRATEGY reports the first of a TYPE key, "
          "a MULTISTART without N_STARTS and a RESTART that never "
          "restarts", "[hpotarget_engine]")
{
    const Lines base = kRoot + Lines{"OPTIMIZER", "STRATEGY",
                                     "TYPE MULTISTART", "#", "LIB NLOPT",
                                     "ALGO LN_SBPLX", "MAXEVAL 10", "TOPK 3",
                                     "RESTART", "AFTER_STAGNATION 5",
                                     "ENDRESTART", "ENDSTRATEGY",
                                     "ENDOPTIMIZER"};
    refuses(base,
            at_line(10, "TOPK is not accepted with STRATEGY TYPE "
                        "MULTISTART: only TYPE PRESTAGE forwards TOPK "
                        "candidates to its refinement, so it would have "
                        "no effect; TOPK takes effect under STRATEGY TYPE "
                        "PRESTAGE", "TOPK 3"));
    Lines no_topk = base;
    no_topk[9] = "#";
    refuses(no_topk,
            at_line(5, "STRATEGY TYPE MULTISTART needs N_STARTS 2 or "
                       "more: without N_STARTS it runs one start, which "
                       "is STRATEGY TYPE PLAIN", "TYPE MULTISTART"));
    Lines restart_only = no_topk;
    restart_only[5] = "N_STARTS 3";
    refuses(restart_only,
            at_line(13, "the RESTART block never restarts the search: a "
                        "restart needs AFTER_STAGNATION and MAX, and the "
                        "block has no MAX line; a search without "
                        "restarts has no RESTART block", "ENDRESTART"));
}

/* Pins: the values of --gate and --s0 of a PC_FAILPROBE line are read
 * before the line's other options: a wrong --gate value is refused even
 * where an unknown option stands before it on the line.
 * Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: a PC_FAILPROBE line reads --gate before its "
          "other options", "[hpotarget_engine]")
{
    const std::string probe = "PC_FAILPROBE --bogus --script " +
                              probe_script() + " --outvar S --gate ";
    refuses(kRoot + Lines{probe + "x"} + kOptimizer,
            at_line(3, "PC_FAILPROBE --gate must be a finite number, got "
                       "'x'", probe + "x"));
    const std::string unknown = refusal(kRoot + Lines{probe + "5"} +
                                        kOptimizer);
    CHECK(unknown.find("PC_FAILPROBE: unknown option '--bogus'") !=
          std::string::npos);
}

/* Pins: the checks at the end of the file run in the order of
 * structure::eof_order: a file with two faults of two checks next to each
 * other in the order is refused as the file with the fault of the earlier
 * check alone, by the parser. The pairs no file meets together are
 * those of test_hpotarget_registry_bindings.cpp.
 * Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: two faults at the end of the file are "
          "reported in the order of the checks", "[hpotarget_engine]")
{
    const std::string bad_dump =
        "DUMP 1 " + (scratch() / "no_such_dir" / "pots").string();
    const Lines plain = {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1", "#",
                         "#", "#", "OPTIMIZER", "LIB NLOPT",
                         "ALGO LN_SBPLX", "MAXEVAL 10", "INIT",
                         "METHOD RANDOM 2", "K_TOTAL 1", "ENDINIT",
                         "ENDOPTIMIZER"};
    std::vector<Fault> no_optimizer;
    for (std::size_t i = 5; i < plain.size(); ++i)
        no_optimizer.push_back({i, "#", plain[i]});
    const std::string observable =
        "PC_OBSERVABLE cohesive_energy --species H --target 2.0 --weight 1 "
        "--hotvar a0 1.5 2.5";
    const std::string script_line =
        "PC_LAMMPS --script " + probe_script() + " --varloss E 1 1";
    const Lines with_observable = {"OPTIM WATOM (1) 0.1 5.0", "#",
                                   observable, "OPTIMIZER", "LIB NLOPT",
                                   "ALGO LD_LBFGS", "MAXEVAL 10", "#",
                                   "#", "ENDOPTIMIZER"};
    std::vector<Fault> no_observable_optimizer;
    for (std::size_t i = 3; i < with_observable.size(); ++i)
        no_observable_optimizer.push_back({i, "#", with_observable[i]});
    const std::string cold_plain =
        "PC_LAMMPS --script " + reads_a_script() + " --varloss E3 1 1";
    const std::string cold = cold_plain + " --invar a 1";
    const Lines with_scripts = {
        "OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1",
        "PC_LAMMPS --script " + probe_script() +
            " --varloss E 1 1 --hotvar a 1 2",
        cold_plain, "#", "OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
        "MAXEVAL 10", "ENDOPTIMIZER"};
    const std::string hot_again = "PC_LAMMPS --script " + probe_script() +
                                  " --varloss E2 1 1 --hotvar a 1 2";
    const std::vector<TwoFaults> pairs = {
        {"an open block and a K_TOTAL that caps nothing", plain,
         {{13, "#", "ENDOPTIMIZER"}}, {{11, "K_TOTAL 5", "K_TOTAL 1"}}},
        {"a K_TOTAL that caps nothing and an unread PC_SCRIPT_TIMEOUT",
         plain, {{11, "K_TOTAL 5", "K_TOTAL 1"}},
         {{2, "PC_SCRIPT_TIMEOUT 60", "#"}}},
        {"an unread PC_SCRIPT_TIMEOUT and an unread LOSS", plain,
         {{2, "PC_SCRIPT_TIMEOUT 60", "#"}},
         {{1, "#", "PC_ERMSE 0 1"}, {3, "LOSS L2", "#"}}},
        {"an unread PC_SCRIPT_TIMEOUT and a target LOSS MAPE cannot "
         "measure", plain, {{2, "PC_SCRIPT_TIMEOUT 60", "#"}},
         {{3, "LOSS MAPE", "#"}}},
        {"a target LOSS MAPE cannot measure and a RHOGATE no OPTIM line "
         "arms", plain, {{3, "LOSS MAPE", "#"}}, {{4, "RHOGATE 0", "#"}}},
        {"a RHOGATE no OPTIM line arms and a DUMP directory that cannot be "
         "made", plain, {{4, "RHOGATE 0", "#"}}, {{2, bad_dump, "#"}}},
        {"a DUMP directory that cannot be made and no OPTIMIZER block",
         plain, {{2, bad_dump, "#"}}, no_optimizer},
        {"no OPTIMIZER block and PC_OBSERVABLE beside PC_LAMMPS",
         with_observable, no_observable_optimizer, {{1, script_line, "#"}}},
        {"PC_OBSERVABLE beside PC_LAMMPS and GRAD_STRATEGY NUMERICAL",
         with_observable, {{1, script_line, "#"}},
         {{7, "GRAD_STRATEGY NUMERICAL", "#"}}},
        {"PC_OBSERVABLE beside PC_LAMMPS and a script that reads a hot "
         "variable", with_scripts, {{4, observable, "#"}},
         {{3, cold, cold_plain}}},
        {"a script that reads a hot variable and a hot variable of two "
         "lines", with_scripts, {{3, cold, cold_plain}},
         {{4, hot_again, "#"}}},
        {"a hot variable of two lines and a gradient-based optimiser",
         with_scripts, {{4, hot_again, "#"}},
         {{7, "ALGO LD_LBFGS", "ALGO LN_SBPLX"}}},
    };
    for (const TwoFaults& t : pairs) check_first_wins(t);
}

/* Pins: a file with every block and their keys, the top-level keys, the
 * metric lines with their options, the PC_LAMMPS and PC_FAILPROBE lines
 * with theirs, the PC_OBSERVABLE line, the OPTIM lines and continued lines
 * is read, and the description of the parse (describe_parse) holds the
 * settings of its lines. (Every input of the frozen corpus,
 * test_hpotarget_corpus.cpp, is compared with its whole description.)
 * Before the fix: the parser does not exist. */
TEST_CASE("hpotarget engine: a file of every block and line is read",
          "[hpotarget_engine]")
{
    const std::string pots = (scratch() / "pots").string();
    const std::string best = (scratch() / "best_pots").string();
    const std::vector<Lines> files = {
        /* The OPTIMIZER block with every inner block but EXPLORE. */
        {"OPTIM WATOM (1) 0.1 5.0 LOG", "OPTIM CGRID2B (1,2) AUTO",
         "OPTIM_SCALE NONE", "PC_ERMSE 0 1 --skip-above 50",
         "PC_EMAE 0 2 --skip-rel 3", "PC_ErRMSE 0 1", "PC_ERSQ 1 0.5",
         "PC_FRMSE 0 1", "PC_SRMSE_GPA 0 1", "LOSS HUBER 20",
         "FAILSCORE 1e5", "QFLUSH 3", "OUTPUT 2", "BOUTPUT 4", "DIGITS 8",
         "DUMP 5 " + pots, "BDUMP 2 " + best, "FAILSCORE_GRADING ON",
         "BUMPGATE STRICT", "PC_ISOLATION THREAD",
         "OPTIMIZER", "LOG_HP 100", "NOISE", "AUTO false",
         "CALIBRATION 3", "ENDNOISE", "INIT", "METHOD CONFIG",
         "STRATEGY RANDOM 4", "SEED 11", "K_TOTAL 3",
         "WARM_ON_MISMATCH skip", "ENDINIT", "STRATEGY",
         "TYPE MULTISTART", "N_STARTS 3", "BUDGET_PER_START 20",
         "LIB NLOPT", "ALGO LN_BOBYQA", "MAXEVAL 60", "MAXTIME 100",
         "STOPVAL -1", "FTOL_REL 1e-6", "FTOL_ABS 1e-8", "XTOL_REL 1e-4",
         "XTOL_ABS 1e-6", "STEP 0.1", "STEP WATOM (1) 0.2", "SEED 42",
         "THREADS 2", "PARAM fd_step 1e-3", "RESTART",
         "AFTER_STAGNATION 10", "MAX 2", "METHOD RANDOM 2", "ENDRESTART",
         "ENDSTRATEGY", "ENDOPTIMIZER"},
        /* PRESTAGE with EXPLORE, REFINE and their INNER blocks. */
        {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1", "OPTIMIZER",
         "STRATEGY", "TYPE PRESTAGE", "TOPK 2", "EXPLORE", "LIB NLOPT",
         "ALGO G_MLSL_LDS", "MAXEVAL 30", "POPULATION 4", "INIT",
         "METHOD SOBOL 4", "ENDINIT", "INNER", "LIB NLOPT",
         "ALGO LN_SBPLX", "MAXEVAL 5", "ENDINNER", "ENDEXPLORE", "REFINE",
         "LIB NLOPT", "ALGO LN_BOBYQA", "MAXEVAL 20", "VECTOR_ARR 3",
         "ENDREFINE", "ENDSTRATEGY", "ENDOPTIMIZER"},
        /* The PC_LAMMPS and PC_FAILPROBE lines, with the settings they
         * read and a continued line. */
        {"OPTIM WATOM (1) 0.1 5.0", "LOSS L1", "FAILSCORE 50",
         "PC_SCRIPT_TIMEOUT 1:30",
         "PC_LAMMPS --script " + probe_script() + " --varloss E 1 2 \\",
         "  --outvar E2 --hotvar a 1 2 1.5 --failvar S 2 -var x 3",
         "PC_FAILPROBE --script " + probe_script() +
             " --outvar S --gate 0.5 --s0 2 --maxtime 30",
         "MAXTIME 20", "PC_LAMMPS --script " + probe_script() +
                           " --varloss E2 1 1 HUBER 3 --failscore 7",
         "PC_WORKER_TIMEOUT 600", "OPTIMIZER", "LIB NLOPT",
         "ALGO LN_SBPLX", "MAXEVAL 10", "ENDOPTIMIZER"},
        /* A PC_OBSERVABLE line and a gradient-based optimiser. */
        {"OPTIM WATOM (1) 0.1 5.0",
         "PC_OBSERVABLE cohesive_energy --species H --target 2.0 "
         "--weight 1 --hotvar a0 1.5 2.5 2.0 --supercell 2 2 2 "
         "--maxiter 50 --unconverged const 10",
         "OPTIMIZER", "LIB NLOPT", "ALGO LD_LBFGS", "MAXEVAL 10",
         "ENDOPTIMIZER"},
    };
    /* Settings of each file in its description. */
    const std::vector<std::vector<std::string>> settings = {
        {"scale_policy.normalise = false", "globalLoss.type = HUBER",
         "FAIL_SCORE = 100000", "QFLUSH_THRESHOLD = 3", "outputEvery = 2",
         "bestOutputEvery = 4", "outputDigits = 8",
         "FAILSCORE_GRADING = true", "BUMPGATE_STRICT = true",
         "PC_ISOLATION = \"THREAD\"", "skip_emae.rel_above = 3",
         "spec.strategy_type = MULTISTART", "spec.multistart.n_starts = 3",
         "spec.multistart.budget_per_start = 20",
         "spec.algo = \"LN_BOBYQA\"", "spec.stop.maxeval = 60",
         "spec.seed = 42", "spec.log_threshold = 100",
         "spec.noise.calibration_n = 3", "spec.restart.max = 2",
         "spec.restart.after_stagnation = 10",
         "spec.init_strategy_spec.k_total = 3",
         "spec.init_strategy_spec.warm_on_mismatch = \"skip\""},
        {"spec.strategy_type = PRESTAGE", "spec.stage_topk = 2",
         "spec.algo = \"LN_BOBYQA\"", "spec.vector_storage = 3",
         "spec.prestage.algo = \"G_MLSL_LDS\"", "spec.prestage.pop = 4",
         "spec.prestage.inner.algo = \"LN_SBPLX\""},
        {"LS.size = 3", "failprobe_gate = 0.5", "failprobe_s0 = 2",
         "PC_WORKER_TIMEOUT = 600", "globalLoss.type = L1",
         "LS[0].hotvars[0].name = \"a\"", "LS[2].failscore = 7"},
        {"observables.size = 1", "spec.algo = \"LD_LBFGS\""},
    };
    REQUIRE(settings.size() == files.size());
    for (std::size_t i = 0; i < files.size(); ++i) {
        CAPTURE(files[i]);
        const Outcome o = read_file(files[i]);
        REQUIRE(o.error == "");
        for (const std::string& line : settings[i]) {
            CAPTURE(line);
            CHECK(("\n" + o.dump).find("\n" + line + "\n") !=
                  std::string::npos);
        }
    }
}

/* Detects a member function read(const std::string&) of T. */
template <typename T, typename = void>
struct has_read : std::false_type {};
template <typename T>
struct has_read<T, std::void_t<decltype(std::declval<T&>().read(
                       std::declval<const std::string&>()))>>
    : std::true_type {};

/* Pins: the registry-driven parser is the one parser of the target file:
 * ParsedTarget has no read(), the hand-written parser, and the test seam
 * (hpo_targets_testing.h) neither chooses a parser for the targets
 * (TADAH_HPOTARGET_PARSER_CHOICE) nor compares two (the parser shadow,
 * TADAH_HPOTARGET_PARSER_SHADOW_SEAM).
 * Before the fix: fails, ParsedTarget::read() exists and the seam chooses
 * between it and the registry-driven parser. */
TEST_CASE("hpotarget engine: the registry-driven parser is the one parser "
          "of the target file", "[hpotarget_engine]")
{
    CHECK_FALSE(has_read<tadah::hpo::ParsedTarget>::value);
#if defined(TADAH_HPOTARGET_PARSER_CHOICE)
    FAIL("hpo_targets_testing.h chooses the parser of the targets "
         "(TADAH_HPOTARGET_PARSER_CHOICE)");
#endif
#if defined(TADAH_HPOTARGET_PARSER_SHADOW_SEAM)
    FAIL("hpo_targets_testing.h compares two parsers "
         "(TADAH_HPOTARGET_PARSER_SHADOW_SEAM)");
#endif
}


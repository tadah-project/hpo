// test_pc_lines_losses.cpp
//
// The PC lines of a target file (PC_LAMMPS, PC_FAILPROBE, PC_OBSERVABLE),
// the loss functions of their terms and of the metric lines, and the
// startup checks of both, as `tadah hpo` reads a target file: every input
// it accepts changes the run, and one that would not, or would fail every
// evaluation, is refused at its line.
//
//   - a gradient-based optimiser, DLIB:BFGS included, is refused with a
//     PC_LAMMPS or a PC_FAILPROBE line;
//   - a file whose only PC line is PC_FAILPROBE has no loss source;
//   - a LOSS, FAILSCORE or PC_SCRIPT_TIMEOUT (MAXTIME) line that no line
//     after it reads is refused: a term that names its loss function, and
//     a line with its own --failscore or --maxtime, read none;
//   - an option that takes one value, given twice on one line, and a
//     variable declared twice on one line, are refused;
//   - a LAMMPS variable a line defines (--invar, -var, -v) that is also a
//     --hotvar the line's script would be injected is refused;
//   - a `quit` command in a PC_LAMMPS or PC_FAILPROBE script is refused;
//   - -sc and -l are LAMMPS' -screen and -log;
//   - a target that MAPE (0) or RMSLE (at or below -1e-12) cannot measure
//     is refused at its line; MAPE divides by the target;
//   - a PC_OBSERVABLE term takes the loss function of the LOSS line above
//     it, and --loss takes the parameter of HUBER and TUKEY; GRAD_STRATEGY
//     ANALYTICAL accepts every loss function of such a term; the --check
//     summary gives the loss function of every term;
//   - PC_OBSERVABLE cohesive_energy refuses a --species that is no element
//     (at its line) or one the potential lacks (at startup), and a
//     --hotvar a0 start outside its window;
//   - under FAILSCORE_GRADING OFF the settings that only grade a failure
//     (PC_FAILPROBE --s0, a --failvar scale) are refused.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/analytical_gate.h>
#include <tadah/hpo/check_summary.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using tadah::hpo::analytical_loss_term_blocker;
using tadah::LossFunction;

/* A scratch directory, the cwd while it lives: a target file names its
 * scripts by relative paths, and the messages quote them as written. */
struct Scratch {
    fs::path dir, original_cwd;
    Scratch() {
        dir = fs::temp_directory_path() /
              ("tadah_pc_lines_" + std::to_string(::getpid()) + "_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~Scratch() {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
    void write(const std::string& name, const std::string& text) const {
        std::ofstream(dir / name) << text;
    }
};

/* The script every PC line of these cases runs: the variables the lines
 * name, equal-style. */
const std::string kScript =
    "variable E equal 1.0\nvariable E2 equal 2.0\nvariable S equal 0.0\n"
    "variable bad equal 0.0\nvariable worse equal 0.0\n";

/* A one-species two-body model that builds without data (Ta). */
Context make_ctx() {
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", 2);
    c.add("TYPE2B", 2);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("CGRID2B", "1.0 2.0");
    c.add("SGRID2B", "0.5 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("RCUTMB", 5.0);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    return c;
}

/* An optimiser and a search parameter: six lines, the PC lines of a case
 * start on line 7. */
const std::string kSbplx =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
    "OPTIM CGRID2B (1) 0.5 1.5\n";

/* "" when the target file @p body is read; otherwise the error. */
std::string read_error(const std::string& body) {
    std::ofstream("config.hpo") << body;
    Context ctx = make_ctx();
    try {
        HPOTargets trg(ctx, "config.hpo");
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* "" when the target file @p body is read and passes the startup checks
 * (with a validation configuration); otherwise the error. */
std::string startup_error(const std::string& body) {
    std::ofstream("config.hpo") << body;
    Context ctx = make_ctx();
    try {
        HPOTargets trg(ctx, "config.hpo");
        tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* The error of a target file at line @p line, as the parser frames it. */
std::string at_line(std::size_t line, const std::string& what,
                    const std::string& text) {
    return "Error on line " + std::to_string(line) + ": " + what +
           "\nLine: " + text;
}

/* "" when LammpsSim reads the PC_LAMMPS (PC_FAILPROBE) line @p cmd under
 * the default loss @p loss; otherwise the error. */
std::string sim_error(std::string cmd, bool failprobe = false,
                      const LossFunction& loss = LossFunction{}) {
    try {
        tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, loss, failprobe);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

/* Pins: a PC_LAMMPS line, or a PC_FAILPROBE line alone, refuses DLIB:BFGS
 * as it refuses every gradient-based optimiser, in the STRATEGY block and
 * in a PRESTAGE EXPLORE block: "--hpotarget defines <line> but OPTIMIZER's
 * optimiser DLIB:BFGS is gradient-based. A PC_LAMMPS or PC_FAILPROBE line
 * requires a gradient-free optimiser (" and the list, with no exception for
 * DLIB:BFGS in the message; a gradient-free optimiser is still accepted.
 * The refusals name the line the file has: PC_LAMMPS when it has one, and
 * PC_FAILPROBE for a file whose only script line is the probe -- for a
 * gradient-based optimiser, an MLSL whose default local optimiser takes
 * gradients, such an INNER block and a GRAD_STRATEGY line alike.
 * Before the fix: DLIB:BFGS was the one gradient-based optimiser accepted
 * (read_error was "" for the three files), and the refusal of the others
 * said "or DLIB:BFGS, the one gradient-based optimiser it accepts". Before
 * the line was named: every refusal said "--hpotarget defines PC_LAMMPS
 * but" and "PC_LAMMPS requires" (or "PC_LAMMPS does not support"), also
 * for a file without a PC_LAMMPS line. */
TEST_CASE("PC lines: PC_LAMMPS and PC_FAILPROBE refuse DLIB:BFGS",
          "[HPO][pc_lines][gradient]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    const std::string bfgs =
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n";
    const std::string lammps = "PC_LAMMPS --script p.in --varloss E 1 1\n";
    const std::string probe =
        "PC_ERMSE 0 1\nPC_FAILPROBE --script p.in --outvar S\n";
    struct Case { std::string lines, named; };
    for (const Case& c : std::vector<Case>{
             {lammps, "PC_LAMMPS"},
             {probe, "PC_FAILPROBE"},
             {probe + lammps, "PC_LAMMPS"}}) {
        INFO(c.lines);
        const std::string e = read_error(bfgs + c.lines);
        CHECK_THAT(e, Catch::StartsWith(
                          "Error on line 5: --hpotarget defines " + c.named +
                          " but OPTIMIZER's optimiser DLIB:BFGS is "
                          "gradient-based. A PC_LAMMPS or PC_FAILPROBE line "
                          "requires a gradient-free optimiser (") &&
                          Catch::EndsWith("Line: ENDOPTIMIZER"));
        CHECK_THAT(e, !Catch::Contains("DLIB:BFGS, the one"));
        CHECK(read_error(kSbplx + c.lines) == "");
    }
    const std::string explore =
        "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB NLOPT\n"
        "    ALGO LN_BOBYQA\n    MAXEVAL 1\n    EXPLORE\n      LIB DLIB\n"
        "      ALGO BFGS\n      MAXEVAL 1\n    ENDEXPLORE\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n";
    CHECK_THAT(read_error(explore + lammps),
               Catch::StartsWith(
                   "Error on line 11: --hpotarget defines PC_LAMMPS but "
                   "EXPLORE's optimiser DLIB:BFGS is gradient-based. "
                   "A PC_LAMMPS or PC_FAILPROBE line requires a "
                   "gradient-free optimiser ("));
    CHECK_THAT(read_error(explore + probe),
               Catch::StartsWith(
                   "Error on line 11: --hpotarget defines PC_FAILPROBE but "
                   "EXPLORE's optimiser DLIB:BFGS is gradient-based. "
                   "A PC_LAMMPS or PC_FAILPROBE line requires a "
                   "gradient-free optimiser ("));

    /* The other refusals of a gradient with a script line, for a file
     * whose only script line is the probe. */
    CHECK(read_error(
              "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\n"
              "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\n" + probe) ==
          "Error on line 6: --hpotarget defines PC_FAILPROBE but "
          "OPTIMIZER's GRAD_STRATEGY is set. A PC_LAMMPS or PC_FAILPROBE "
          "line does not support analytical/numerical/hybrid gradients. "
          "Drop the GRAD_STRATEGY line.\nLine: ENDOPTIMIZER");
    CHECK_THAT(read_error("OPTIMIZER\nLIB NLOPT\nALGO GD_MLSL\nMAXEVAL 1\n"
                          "ENDOPTIMIZER\n" + probe),
               Catch::StartsWith(
                   "Error on line 5: --hpotarget defines PC_FAILPROBE but "
                   "OPTIMIZER's optimiser NLOPT:GD_MLSL takes a gradient: "
                   "NLopt's MLSL and AUGLAG take one for their local "
                   "search, and NLopt's default local optimiser, "
                   "NLOPT:LD_MMA, is gradient-based; an INNER block with a "
                   "derivative-free one takes none. A PC_LAMMPS or "
                   "PC_FAILPROBE line requires a gradient-free optimiser ("));
    CHECK_THAT(read_error("OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LN_AUGLAG\n"
                          "MAXEVAL 1\nINNER\nLIB NLOPT\nALGO LD_LBFGS\n"
                          "ENDINNER\nENDSTRATEGY\nENDOPTIMIZER\n" + probe),
               Catch::StartsWith(
                   "Error on line 9: --hpotarget defines PC_FAILPROBE but "
                   "INNER's optimiser NLOPT:LD_LBFGS is gradient-based. A "
                   "PC_LAMMPS or PC_FAILPROBE line requires a gradient-free "
                   "optimiser, and an INNER search is a local NLopt "
                   "algorithm: name a derivative-free one in the INNER "
                   "block ("));
}

/* Pins: a file whose only PC line is PC_FAILPROBE has no loss source and
 * is refused at startup, naming PC_FAILPROBE as a line that gives no loss
 * term; with a PC_ERMSE line, or a PC_LAMMPS line, it starts.
 * Before the fix: the PC_FAILPROBE line counted as a loss source, the file
 * passed the startup checks, and every evaluation scored 0
 * (startup_error was ""). */
TEST_CASE("PC lines: a file whose only PC line is PC_FAILPROBE has no loss "
          "source", "[HPO][pc_lines][loss_source]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    const std::string probe = "PC_FAILPROBE --script p.in --outvar S\n";
    CHECK(startup_error(kSbplx + probe) ==
          "No loss source: a loss term comes from a PC_EMAE, PC_ERMSE, "
          "PC_ErRMSE, PC_ERSQ, PC_FRMSE or PC_SRMSE_GPA line (with "
          "--validation), a PC_OBSERVABLE line or a PC_LAMMPS line, and the "
          "target file has none (a PC_FAILPROBE line gives no loss term)");
    CHECK(startup_error(kSbplx + probe + "PC_ERMSE 0 1\n") == "");
    CHECK(startup_error(kSbplx + probe +
                        "PC_LAMMPS --script p.in --varloss E 1 1\n") == "");
}

/* Pins: a line reads LOSS, FAILSCORE or PC_SCRIPT_TIMEOUT (MAXTIME) when it
 * uses the value: a --varloss or PC_OBSERVABLE term that names no loss
 * function reads LOSS, and a PC_LAMMPS or PC_FAILPROBE line without
 * --failscore (--maxtime) reads FAILSCORE (the script timeout). A
 * PC_SCRIPT_TIMEOUT (or MAXTIME) line that no line after it reads, and a
 * LOSS line that no term after it reads in a file without a metric line,
 * are refused at their line ("... it would have no effect"): one with no
 * PC line after it, and one whose following lines all give their own
 * value (--maxtime; a loss function on every --varloss, --loss on the
 * PC_OBSERVABLE line). A second line of one of the three keys with no line
 * between that reads the first is refused as set twice, naming the lines
 * that read it; a line between with its own value does not read the
 * first. A LOSS line after the last PC line is read by the metric lines, a
 * timeout before a PC_FAILPROBE line by it, and a LOSS line before a
 * PC_LAMMPS line one of whose terms names no loss function by that term.
 * Before the fix: a line read the three keys whatever it gave: the files
 * whose lines give their own values were read (read_error ""), and the
 * messages said "the PC_LAMMPS and PC_FAILPROBE lines after it, and no
 * such line follows it", "the PC_LAMMPS and PC_OBSERVABLE lines after it"
 * and "no PC_LAMMPS or PC_OBSERVABLE line between takes the first". */
TEST_CASE("PC lines: a LOSS, FAILSCORE or PC_SCRIPT_TIMEOUT line that no "
          "line reads is refused", "[HPO][pc_lines][line_settings]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    const std::string lammps = "PC_LAMMPS --script p.in --varloss E 1 1\n";
    const std::string obs =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
        "--hotvar a0 2.5 4.5";
    for (const char* key : {"PC_SCRIPT_TIMEOUT", "MAXTIME"}) {
        INFO(key);
        const std::string unread =
            std::string(key) +
            " sets the script timeout of the PC_LAMMPS and PC_FAILPROBE lines "
            "after it that give no --maxtime, and no such line follows it; "
            "it would have no effect";
        CHECK(read_error(kSbplx + lammps + key + " 5\n") ==
              at_line(8, unread, std::string(key) + " 5"));
        CHECK(read_error(kSbplx + key + " 5\nPC_ERMSE 0 1\n") ==
              at_line(7, unread, std::string(key) + " 5"));
        CHECK(read_error(kSbplx + key +
                         " 5\nPC_LAMMPS --script p.in --varloss E 1 1 "
                         "--maxtime 10\n") ==
              at_line(7, unread, std::string(key) + " 5"));
        CHECK(read_error(kSbplx + key + " 5\nPC_ERMSE 0 1\nPC_FAILPROBE "
                         "--script p.in --outvar S --maxtime 10\n") ==
              at_line(7, unread, std::string(key) + " 5"));
        CHECK(read_error(kSbplx + "PC_ERMSE 0 1\n" + key +
                         " 5\nPC_FAILPROBE --script p.in --outvar S\n") ==
              "");
        CHECK(read_error(kSbplx + key + " 5\nPC_LAMMPS --script p.in "
                         "--varloss E 1 1 --maxtime 10\n" + key + " 7\n" +
                         lammps) ==
              at_line(9,
                      std::string(key) + " is set twice (first on line 7) "
                      "and no line between reads the first (a PC_LAMMPS or "
                      "PC_FAILPROBE line without --maxtime); each key is "
                      "given once",
                      std::string(key) + " 7"));
    }
    CHECK(read_error(kSbplx + "FAILSCORE 1e9\nPC_LAMMPS --script p.in "
                     "--varloss E 1 1 --failscore 5\nFAILSCORE 1e10\n" +
                     lammps) ==
          at_line(9,
                  "FAILSCORE is set twice (first on line 7) and no line "
                  "between reads the first (a PC_LAMMPS or PC_FAILPROBE line "
                  "without --failscore); each key is given once",
                  "FAILSCORE 1e10"));
    CHECK(read_error(kSbplx + "FAILSCORE 1e9\n" + lammps +
                     "FAILSCORE 1e10\n" + lammps) == "");
    const std::string loss_unread =
        "LOSS sets the loss function of the --varloss and PC_OBSERVABLE "
        "terms after it that name none, and of the metric lines (PC_EMAE, "
        "PC_ERMSE, PC_ErRMSE, PC_ERSQ, PC_FRMSE, PC_SRMSE_GPA), and the file "
        "has no metric line and no such term after it; it would have no "
        "effect";
    CHECK(read_error(kSbplx + "LOSS L1\n" + lammps + "LOSS HUBER\n") ==
          at_line(9, loss_unread, "LOSS HUBER"));
    CHECK(read_error(kSbplx + lammps +
                     "LOSS L1\nPC_FAILPROBE --script p.in --outvar S\n") ==
          at_line(8, loss_unread, "LOSS L1"));
    CHECK(read_error(kSbplx + "LOSS HUBER 10\nPC_LAMMPS --script p.in "
                     "--varloss E 8.1 800 L1\n") ==
          at_line(7, loss_unread, "LOSS HUBER 10"));
    CHECK(read_error(kSbplx + "LOSS MAPE\n" + obs + " --loss L2\n") ==
          at_line(7, loss_unread, "LOSS MAPE"));
    CHECK(read_error(kSbplx + "LOSS L1\n" + lammps + "LOSS HUBER 10\n"
                     "PC_LAMMPS --script p.in --varloss E 8.1 800 L2\n") ==
          at_line(9, loss_unread, "LOSS HUBER 10"));
    CHECK(read_error(kSbplx + "LOSS L1\n" + lammps +
                     "LOSS HUBER\nPC_ERMSE 0 1\n") == "");
    CHECK(read_error(kSbplx + "LOSS L1\nPC_LAMMPS --script p.in --varloss E "
                     "1 1 L2 --varloss E2 2 1\n") == "");
    CHECK(read_error(kSbplx + "LOSS MAPE\n" + obs + "\n") == "");
    const std::string loss_twice =
        "LOSS is set twice (first on line 7) and no line between reads the "
        "first (a --varloss or PC_OBSERVABLE term that names no loss "
        "function); each key is given once";
    CHECK(read_error(kSbplx + "LOSS L1\nPC_FAILPROBE --script p.in "
                     "--outvar S\nLOSS L2\n" + lammps) ==
          at_line(9, loss_twice, "LOSS L2"));
    CHECK(read_error(kSbplx + "LOSS L1\nPC_LAMMPS --script p.in --varloss E "
                     "1 1 L2\nLOSS HUBER\nPC_ERMSE 0 1\n") ==
          at_line(9, loss_twice, "LOSS HUBER"));
}

/* Pins: an option of a PC_LAMMPS or PC_FAILPROBE line that takes one value
 * (--script, --failscore, --maxtime; --gate, --s0) given twice on the line
 * is refused at the line, "<option> is given twice on this line; each
 * option is given once" ("PC_FAILPROBE --gate ..." for the probe's own); a
 * --hotvar or --failvar variable declared twice, and a LAMMPS variable
 * defined twice (--invar, -var, -v), are refused naming the variable.
 * Before the fix: each line was read, the later value or declaration
 * kept (read_error ""). */
TEST_CASE("PC lines: a one-value option or a variable given twice on a "
          "PC_LAMMPS or PC_FAILPROBE line is refused",
          "[HPO][pc_lines][repeat]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    s.write("q.in", kScript);
    const std::string base = "PC_LAMMPS --script p.in --varloss E 1 1 ";
    struct Case { std::string tail, what; };
    for (const Case& c : std::vector<Case>{
             {"--script q.in",
              "--script is given twice on this line; each option is given "
              "once"},
             {"--failscore 10 --failscore 20",
              "--failscore is given twice on this line; each option is given "
              "once"},
             {"--maxtime 10 --maxtime 20",
              "--maxtime is given twice on this line; each option is given "
              "once"},
             {"--hotvar E2 0 3 --hotvar E2 0 4",
              "--hotvar E2 is declared twice on this line; each variable is "
              "declared once"},
             {"--failvar bad --failvar bad",
              "--failvar bad is declared twice on this line; each variable "
              "is declared once"},
             {"--invar x 1 --invar x 2",
              "--invar x: the variable x is defined twice on this line "
              "(first by --invar x); each variable is defined once"},
             {"--invar x 1 -var x 2",
              "-var x: the variable x is defined twice on this line (first "
              "by --invar x); each variable is defined once"},
             {"-v x 1 --invar x 2",
              "--invar x: the variable x is defined twice on this line "
              "(first by -v x); each variable is defined once"}}) {
        INFO(c.tail);
        const std::string line = base + c.tail;
        CHECK(read_error(kSbplx + "PC_ERMSE 0 1\n" + line + "\n") ==
              at_line(8, c.what, line));
    }
    const std::string probe = "PC_FAILPROBE --script p.in --outvar S ";
    for (const Case& c : std::vector<Case>{
             {"--gate 1 --gate 3",
              "PC_FAILPROBE --gate is given twice on this line; each option "
              "is given once"},
             {"--s0 0.5 --s0 2",
              "PC_FAILPROBE --s0 is given twice on this line; each option is "
              "given once"},
             {"--maxtime 10 --maxtime 20",
              "--maxtime is given twice on this line; each option is given "
              "once"}}) {
        INFO(c.tail);
        const std::string line = probe + c.tail;
        CHECK(read_error(kSbplx + "FAILSCORE_GRADING ON\nFAILSCORE 1e9\n"
                         "PC_ERMSE 0 1\n" + line + "\n") ==
              at_line(10, c.what, line));
    }
}

/* Pins: an option of a PC_OBSERVABLE cohesive_energy line given twice is
 * refused at the line, "PC_OBSERVABLE cohesive_energy: <option> is given
 * twice on this line; each option is given once", for each of its 13
 * options; and an OPTIM AUTO FLOOR or GROUP_FLOOR given twice: "OPTIM
 * CGRID2B: AUTO FLOOR is given twice on this line; each option is given
 * once".
 * Before the fix: the later value was kept (read_error ""). */
TEST_CASE("PC lines: an option given twice on a PC_OBSERVABLE or an OPTIM "
          "AUTO line is refused", "[HPO][pc_lines][repeat]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    const std::string obs =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
        "--hotvar a0 2.5 4.5 ";
    struct Case { std::string tail, option; };
    for (const Case& c : std::vector<Case>{
             {"--species Ta", "--species"},
             {"--target 8.2", "--target"},
             {"--weight 2", "--weight"},
             {"--hotvar a0 2.6 4.4", "--hotvar"},
             {"--lattice bcc --lattice bcc", "--lattice"},
             {"--supercell 2 2 2 --supercell 3 3 3", "--supercell"},
             {"--loss L1 --loss L2", "--loss"},
             {"--maxiter 10 --maxiter 20", "--maxiter"},
             {"--maxeval 10 --maxeval 20", "--maxeval"},
             {"--max-wall-time 1 --max-wall-time 2", "--max-wall-time"},
             {"--etol 1e-8 --etol 1e-9", "--etol"},
             {"--fmax-converged 0.1 --fmax-converged 0.2",
              "--fmax-converged"},
             {"--unconverged const 100 --unconverged const 50",
              "--unconverged"}}) {
        INFO(c.tail);
        const std::string line = obs + c.tail;
        CHECK(read_error(kSbplx + line + "\n") ==
              at_line(7, "PC_OBSERVABLE cohesive_energy: " + c.option +
                             " is given twice on this line; each option is "
                             "given once",
                      line));
    }
    for (const char* opt : {"FLOOR", "GROUP_FLOOR"}) {
        INFO(opt);
        const std::string line = std::string("OPTIM CGRID2B (2) AUTO FRAC 0.1 ") +
                                 opt + " 0.2 " + opt + " 0.3";
        CHECK(read_error(kSbplx + "PC_ERMSE 0 1\n" + line + "\n") ==
              at_line(8, std::string("OPTIM CGRID2B: AUTO ") + opt +
                             " is given twice on this line; each option is "
                             "given once",
                      line));
    }
}

/* Pins: a --hotvar a0 <init> outside its window [<lo>, <hi>] is refused at
 * the PC_OBSERVABLE line; one at the edge of the window is accepted.
 * Before the fix: the start was clamped into the window without a word
 * (read_error ""). */
TEST_CASE("PC lines: a PC_OBSERVABLE --hotvar a0 start outside its window is "
          "refused", "[HPO][pc_lines][observable]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    const std::string head =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
        "--hotvar a0 2.5 4.5 ";
    for (const char* init : {"5.0", "2.4"}) {
        INFO(init);
        CHECK(read_error(kSbplx + head + init + "\n") ==
              at_line(7, std::string("PC_OBSERVABLE cohesive_energy: "
                                     "--hotvar a0 <init> ") +
                             init + " is outside its window [2.5, 4.5]",
                      head + init));
    }
    CHECK(read_error(kSbplx + head + "4.5\n") == "");
}

/* Pins: PC_OBSERVABLE cohesive_energy --species refuses, at its line, a
 * word that is the symbol of no element ("Xx"), and at startup an element
 * the potential has none of (W, the potential's elements Ta), at the line
 * of the observable; Ta is accepted.
 * Before the fix: neither was checked (read_error and startup_error ""):
 * every evaluation failed on Xx ("Unknown symbol: Xx"), and W gave a
 * cohesive energy of 0 for every candidate. */
TEST_CASE("PC lines: PC_OBSERVABLE --species is an element of the potential",
          "[HPO][pc_lines][observable]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    auto line = [](const std::string& el) {
        return "PC_OBSERVABLE cohesive_energy --species " + el +
               " --target 8.1 --weight 1 --hotvar a0 2.5 4.5";
    };
    CHECK(read_error(kSbplx + line("Xx") + "\n") ==
          at_line(7, "PC_OBSERVABLE cohesive_energy: --species Xx is not the "
                     "symbol of an element",
                  line("Xx")));
    CHECK(startup_error(kSbplx + "PC_ERMSE 0 1\n" + line("W") + "\n") ==
          at_line(8, "PC_OBSERVABLE cohesive_energy --species W: the "
                     "potential has no element W (its elements: Ta)",
                  line("W")));
    CHECK(startup_error(kSbplx + "PC_ERMSE 0 1\n" + line("Ta") + "\n") == "");
}

/* Pins: -sc and -l, LAMMPS' short spellings of -screen and -log, are read
 * as them: the line's LAMMPS arguments hold one -screen and one -log, the
 * line's own (-log x.log), and no "-log none" or "-screen none" after them.
 * Before the fix: the line's "-l x.log" was followed by the "-log none"
 * HPO adds to a line without -log, which LAMMPS applies last: no log file
 * was written. */
TEST_CASE("PC lines: -sc and -l are -screen and -log",
          "[HPO][pc_lines][lammps_options]") {
    Scratch s;
    s.write("p.in", kScript);
    auto count = [](const std::vector<std::string>& v, const std::string& t) {
        std::size_t n = 0;
        for (const auto& x : v) n += x == t;
        return n;
    };
    std::string cmd = "--script p.in --varloss E 1 1 -l x.log -sc none";
    tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, LossFunction{});
    INFO([&] { std::string a; for (const auto& t : sim.invar) a += t + " "; return a; }());
    CHECK(count(sim.invar, "-log") == 1);
    CHECK(count(sim.invar, "-screen") == 1);
    CHECK(count(sim.invar, "-l") == 0);
    CHECK(count(sim.invar, "-sc") == 0);
    const auto log = std::find(sim.invar.begin(), sim.invar.end(), "-log");
    REQUIRE(log != sim.invar.end());
    REQUIRE(log + 1 != sim.invar.end());
    CHECK(*(log + 1) == "x.log");
}

/* Pins: a target that the loss function of its term cannot measure is
 * refused at the line of the term: MAPE divides by the target, so a target
 * of 0 is refused; RMSLE takes ln(target + 1e-12), so a target at or below
 * -1e-12 is refused. The sites: a --varloss term (its own loss function or
 * the line's LOSS), a PC_OBSERVABLE term (--loss, or the LOSS line above
 * it), and a metric line under the last LOSS line, which may follow it.
 * Before the fix: every file was read (read_error ""); under MAPE every
 * candidate scored 1 for a target of 0 (|m - 0| / |m|), and under RMSLE
 * every evaluation failed (ln of a negative number). */
TEST_CASE("PC lines: a target MAPE or RMSLE cannot measure is refused at its "
          "line", "[HPO][pc_lines][loss_target]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    const std::string mape = "MAPE divides by the target, and the target is 0";
    const std::string rmsle =
        "RMSLE takes ln(target + 1e-12), and the target -8.1 is at or below "
        "-1e-12";
    struct Case { std::string lines, what, text; std::size_t at; };
    for (const Case& c : std::vector<Case>{
             {"PC_LAMMPS --script p.in --varloss E 0 1 MAPE\n",
              "--varloss E 0: " + mape,
              "PC_LAMMPS --script p.in --varloss E 0 1 MAPE", 7},
             {"LOSS MAPE\nPC_LAMMPS --script p.in --varloss E 0 1\n",
              "--varloss E 0: " + mape,
              "PC_LAMMPS --script p.in --varloss E 0 1", 8},
             {"PC_LAMMPS --script p.in --varloss E -8.1 800 RMSLE\n",
              "--varloss E -8.1: " + rmsle,
              "PC_LAMMPS --script p.in --varloss E -8.1 800 RMSLE", 7},
             {"PC_OBSERVABLE cohesive_energy --species Ta --target 0 "
              "--weight 1 --hotvar a0 2.5 4.5 --loss MAPE\n",
              "PC_OBSERVABLE cohesive_energy --target 0: " + mape,
              "PC_OBSERVABLE cohesive_energy --species Ta --target 0 "
              "--weight 1 --hotvar a0 2.5 4.5 --loss MAPE", 7},
             {"LOSS RMSLE\nPC_OBSERVABLE cohesive_energy --species Ta "
              "--target -8.1 --weight 1 --hotvar a0 2.5 4.5\n",
              "PC_OBSERVABLE cohesive_energy --target -8.1: " + rmsle,
              "PC_OBSERVABLE cohesive_energy --species Ta --target -8.1 "
              "--weight 1 --hotvar a0 2.5 4.5", 8},
             {"PC_ERMSE 0 1\nLOSS MAPE\n",
              "PC_ERMSE 0: the metric lines take the loss function of the "
              "last LOSS line (line 8): " + mape,
              "PC_ERMSE 0 1", 7},
             {"LOSS RMSLE\nPC_ERMSE (2) -8.1 1\nPC_ERMSE (1) 1 1\n",
              "PC_ERMSE (2) -8.1: the metric lines take the loss function of "
              "the last LOSS line (line 7): " + rmsle,
              "PC_ERMSE (2) -8.1 1", 8},
             {"LOSS MAPE\nPC_EMAE 0 1\n",
              "PC_EMAE 0: the metric lines take the loss function of the "
              "last LOSS line (line 7): " + mape,
              "PC_EMAE 0 1", 8}}) {
        INFO(c.lines);
        CHECK(read_error(kSbplx + c.lines) == at_line(c.at, c.what, c.text));
    }
    /* Measurable targets are read. */
    CHECK(read_error(kSbplx + "LOSS MAPE\nPC_ERMSE 5 1\n") == "");
    CHECK(read_error(kSbplx + "LOSS RMSLE\nPC_ERMSE 0 1\n") == "");
    CHECK(read_error(kSbplx +
                     "PC_LAMMPS --script p.in --varloss E 8.1 1 MAPE\n") == "");
}

/* Pins: MAPE is |measured - target| / |target| times the weight: target
 * 8.1, measured 4.05 costs 0.5 and measured 16.2 costs 1, so the cost
 * grows with the error; its derivative with respect to the measured value
 * is sign(measured - target) w / |target|.
 * Before the fix: MAPE divided by the measured value: 4.05 cost 1 and
 * 16.2 cost 0.5. */
TEST_CASE("PC lines: MAPE divides by the target", "[HPO][pc_lines][loss_target]") {
    const LossFunction mape(LossFunction::Type::MAPE);
    CHECK(mape.loss(4.05, 8.1) == Approx(0.5).epsilon(1e-14));
    CHECK(mape.loss(16.2, 8.1) == Approx(1.0).epsilon(1e-14));
    CHECK(mape.loss(16.2, 8.1, 3.0) == Approx(3.0).epsilon(1e-14));
    CHECK(mape.loss_prime_observed(16.2, 8.1, 3.0) ==
          Approx(3.0 / 8.1).epsilon(1e-14));
    CHECK(mape.loss_prime_observed(4.05, 8.1, 3.0) ==
          Approx(-3.0 / 8.1).epsilon(1e-14));
}

/* Pins: the loss function of a PC_OBSERVABLE term is that of the LOSS line
 * above it when the line gives no --loss (HUBER 10 here), as for a
 * --varloss term, and --loss <name> <parameter> takes the parameter of
 * HUBER and TUKEY; a LOSS line after it does not change it (L2, the LOSS
 * line, read by the metric line). The metric line of the --loss HUBER 5
 * case reads its LOSS L1 line, which the term, naming its own loss
 * function, does not read.
 * Before the fix: the term was L2 whatever LOSS said, and "--loss HUBER 5"
 * was refused ("PC_OBSERVABLE cohesive_energy: unknown flag '5'"). */
TEST_CASE("PC lines: a PC_OBSERVABLE term takes the LOSS line above it and a "
          "--loss parameter", "[HPO][pc_lines][observable][loss]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    const std::string obs =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
        "--hotvar a0 2.5 4.5";
    struct Case {
        std::string lines;
        LossFunction::Type type;
        std::vector<double> params;
    };
    for (const Case& c : std::vector<Case>{
             {"LOSS HUBER 10\n" + obs + "\n", LossFunction::Type::HUBER, {10.0}},
             {"LOSS L1\nPC_ERMSE 0 1\n" + obs + " --loss HUBER 5\n",
              LossFunction::Type::HUBER, {5.0}},
             {obs + " --loss tukey 2\n", LossFunction::Type::TUKEY, {2.0}},
             {obs + "\nPC_ERMSE 0 1\nLOSS L1\n", LossFunction::Type::L2, {}}}) {
        INFO(c.lines);
        std::ofstream("config.hpo") << kSbplx + c.lines;
        Context ctx = make_ctx();
        std::unique_ptr<HPOTargets> trg;
        REQUIRE_NOTHROW(trg = std::make_unique<HPOTargets>(ctx, "config.hpo"));
        REQUIRE(trg->observables.size() == 1u);
        const LossFunction& fn = trg->observables[0]->lossfn();
        CHECK(fn.type() == c.type);
        CHECK(fn.params() == c.params);
    }
    CHECK_THAT(read_error(kSbplx + obs + " --loss HUBER 0\n"),
               Catch::StartsWith(
                   "Error on line 7: loss HUBER parameter (delta) must be "
                   "finite and > 0 (got 0); omit it to use the default 50"));
}

/* Pins: GRAD_STRATEGY ANALYTICAL accepts a PC_OBSERVABLE term of every
 * loss function, wherever it comes from: the LOSS line above the line
 * (MAPE) or its own --loss (RMSLE), as a residual loss (HUBER): each has a
 * derivative with respect to the value (MAPE w sign(m - t) / |t|, RMSLE
 * 2 w (ln(m + 1e-12) - ln(t + 1e-12)) / (m + 1e-12)), the one the metric
 * lines take.
 * Before the fix: MAPE and RMSLE were refused, "the PC_OBSERVABLE term
 * cohesive_energy[1] uses the loss function MAPE, which has no derivative
 * with respect to the value" (RMSLE alike). */
TEST_CASE("PC lines: ANALYTICAL accepts a PC_OBSERVABLE term of every loss "
          "function, from LOSS or --loss",
          "[HPO][pc_lines][observable][loss]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    const std::string obs =
        "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
        "--hotvar a0 2.5 4.5";
    for (const std::string& lines :
         {"LOSS MAPE\n" + obs + "\n", obs + " --loss RMSLE\n",
          "LOSS HUBER 10\n" + obs + "\n"}) {
        INFO(lines);
        std::ofstream("config.hpo") << kSbplx + lines;
        Context ctx = make_ctx();
        std::unique_ptr<HPOTargets> trg;
        REQUIRE_NOTHROW(trg = std::make_unique<HPOTargets>(ctx, "config.hpo"));
        CHECK(analytical_loss_term_blocker(*trg) == "");
    }
}

/* Pins: the --check summary gives the loss function of every term, after
 * its target and weight: a metric line the last LOSS line's (L1), a
 * PC_LAMMPS term its line's (HUBER 10, the LOSS line above it) or its own
 * (TUKEY 2), and the heading names no LOSS line.
 * Before the fix: the heading was "Loss terms: LOSS L1", the LOSS line
 * read last, above terms of other loss functions, and the rows had none. */
TEST_CASE("PC lines: the --check summary gives the loss function of every "
          "term", "[HPO][pc_lines][check_summary]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    std::ofstream("config.hpo")
        << kSbplx
        << "LOSS HUBER 10\nPC_LAMMPS --script p.in --varloss E 1 2 "
           "--varloss E2 3 4 TUKEY 2\nLOSS L1\nPC_ERMSE 0 1\n";
    Context ctx = make_ctx();
    HPOTargets trg(ctx, "config.hpo");
    std::ostringstream out;
    tadah::hpo::write_check_summary(out, "config.hpo", trg, 1, 1, {});
    const std::string text = out.str();
    INFO(text);
    CHECK_THAT(text, Catch::Contains(
        "Loss terms:\n"
        "  PC_ERMSE  target 0, weight 1, loss L1\n"
        "  PC_LAMMPS p.in: E  target 1, weight 2, loss HUBER 10\n"
        "  PC_LAMMPS p.in: E2  target 3, weight 4, loss TUKEY 2\n"));
}

/* Pins: a LAMMPS variable a PC_LAMMPS line defines (--invar, -var, -v) and
 * a --hotvar of the same name are refused: on one line ("--hotvar x: ...
 * defined on this line by --invar x"), and across lines when the line
 * that defines it has no --hotvar and its script reads the hot variable,
 * so that the run would inject `variable x equal <value>` into it, which
 * LAMMPS refuses for a variable of another style. A script that reads
 * only v_xb (another name that starts with x) or does not read x is
 * accepted: it is not injected x. So is a line with a --hotvar of its own
 * (y) whose script reads x and which defines x: a script with a --hotvar
 * is injected its own hot values only, and reads the x of its line.
 * Before the fix: every refused file was read (read_error ""), and every
 * evaluation failed ("Cannot redefine variable x with a different
 * style"). The line with a --hotvar of its own passes as well (a pin of
 * the rule the refusal follows). */
TEST_CASE("PC lines: a variable a line defines is not a --hotvar the line's "
          "script is injected", "[HPO][pc_lines][hotvar]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("hot.in", kScript + "variable x equal 3.3\n");
    s.write("reads.in", kScript + "variable F equal ${x}+1\n");
    s.write("readsv.in", kScript + "variable F equal v_x+1\n");
    s.write("readsxb.in", kScript + "variable xb equal 2\n"
                                    "variable F equal v_xb+1\n");
    s.write("none.in", kScript);
    s.write("hotreads.in", kScript + "variable y equal ${x}+1\n");
    const std::string hot =
        "PC_LAMMPS --script hot.in --varloss E 1 1 --hotvar x 2.5 4.5\n";
    for (const char* opt : {"--invar", "-var", "-v"}) {
        INFO(opt);
        const std::string line =
            std::string("PC_LAMMPS --script hot.in --varloss E 1 1 ") + opt +
            " x 3.3 --hotvar x 2.5 4.5";
        CHECK(read_error(kSbplx + line + "\n") ==
              at_line(7, std::string("--hotvar x: the variable x is defined "
                                     "on this line by ") +
                             opt +
                             " x too; a --hotvar variable is injected as "
                             "`variable x equal <value>`, which LAMMPS "
                             "refuses for a variable of another style",
                      line));
        for (const char* script : {"reads.in", "readsv.in"}) {
            INFO(script);
            const std::string cold = std::string("PC_LAMMPS --script ") +
                                     script + " --varloss F 1 1 " + opt +
                                     " x 3.3";
            /* The cold line before and after the hot line. */
            CHECK(read_error(kSbplx + cold + "\n" + hot) ==
                  at_line(7, std::string("PC_LAMMPS: the script ") + script +
                                 " reads x, the --hotvar of line 8, and this "
                                 "line defines x (" + opt +
                                 " x): the run injects `variable x equal "
                                 "<value>` into the script, which LAMMPS "
                                 "refuses for a variable of another style",
                          cold));
            CHECK(read_error(kSbplx + hot + cold + "\n") ==
                  at_line(8, std::string("PC_LAMMPS: the script ") + script +
                                 " reads x, the --hotvar of line 7, and this "
                                 "line defines x (" + opt +
                                 " x): the run injects `variable x equal "
                                 "<value>` into the script, which LAMMPS "
                                 "refuses for a variable of another style",
                          cold));
        }
        for (const char* script : {"readsxb.in", "none.in"}) {
            INFO(script);
            const std::string cold = std::string("PC_LAMMPS --script ") +
                                     script + " --varloss E 1 1 " + opt +
                                     " x 3.3";
            CHECK(read_error(kSbplx + hot + cold + "\n") == "");
        }
        const std::string own_hot =
            std::string("PC_LAMMPS --script hotreads.in --varloss E 1 1 "
                        "--hotvar y 2.5 4.5 ") + opt + " x 3.3";
        CHECK(read_error(kSbplx + hot + own_hot + "\n") == "");
        CHECK(read_error(kSbplx + own_hot + "\n" + hot) == "");
    }
}

/* Pins: script_reads_variable, which says whether a script reads a
 * variable (the run injects a hot value into a script without --hotvar
 * that reads it, and HPOTargets refuses a line that defines such a value):
 * ${x}; $x for a one-character name, and $xy too (LAMMPS reads it as $x,
 * then y); v_x as a whole word, also v_x[1]; in any line of the script.
 * Not v_xb, xv_x, ${xb}, $ab for the name ab (LAMMPS reads $a, then b), a
 * script without the name, nor an empty name.
 * Before the fix: passes as well (a pin of the rule the run and the
 * refusal use; the substring test the run used before counted v_xb and
 * xv_x as x, and not $x). */
TEST_CASE("PC lines: the variables a script reads",
          "[HPO][pc_lines][hotvar]") {
    using tadah::hpo::script_reads_variable;
    auto reads = [](const std::string& line, const std::string& name) {
        return script_reads_variable({line}, name);
    };
    CHECK(reads("variable F equal ${x}+1", "x"));
    CHECK(reads("variable F equal $x+1", "x"));
    CHECK(reads("variable F equal $xy", "x"));
    CHECK(reads("variable F equal v_x+1", "x"));
    CHECK(reads("variable F equal v_x[1]", "x"));
    CHECK(reads("v_x", "x"));
    CHECK(reads("variable F equal v_a0*2", "a0"));
    CHECK(reads("print \"${a0}\"", "a0"));
    CHECK(script_reads_variable({"variable E equal 1", "print $x"}, "x"));
    CHECK_FALSE(reads("variable F equal v_xb+1", "x"));
    CHECK_FALSE(reads("variable F equal xv_x", "x"));
    CHECK_FALSE(reads("variable F equal ${xb}", "x"));
    CHECK_FALSE(reads("variable F equal $ab", "ab"));
    CHECK_FALSE(reads("variable F equal 1", "x"));
    CHECK_FALSE(reads("variable F equal v_x", ""));
    CHECK_FALSE(script_reads_variable({}, "x"));
}

/* Pins: a quit command in the script of a PC_LAMMPS or PC_FAILPROBE line,
 * as a command or as a command of an if (then, elif and else), is refused
 * when the line is read, naming the script and the command; text that is
 * no command (print "quit") is accepted.
 * Before the fix: every script was accepted (sim_error ""); quit ended
 * tadah hpo (a segmentation fault under PC_ISOLATION THREAD) or its worker
 * (PROCESS). */
TEST_CASE("PC lines: a quit command in a PC_LAMMPS or PC_FAILPROBE script is "
          "refused", "[HPO][pc_lines][quit]") {
    Scratch s;
    const std::string quit_tail =
        ": quit ends the process that runs the script (tadah hpo itself "
        "under PC_ISOLATION THREAD)";
    struct Case { std::string line, command; };
    for (const Case& c : std::vector<Case>{
             {"quit 0", "quit 0"},
             {"quit", "quit"},
             {"if \"1 > 0\" then \"quit 0\"", "if \"1 > 0\" then \"quit 0\""},
             {"if \"${E} > 0\" then \"print ok\" \"quit\"",
              "if \"${E} > 0\" then \"print ok\" \"quit\""},
             {"if \"1 < 0\" then \"print a\" elif \"1 > 2\" \"quit\" else "
              "\"print b\"",
              "if \"1 < 0\" then \"print a\" elif \"1 > 2\" \"quit\" else "
              "\"print b\""},
             {"if \"1 < 0\" then \"print a\" else 'quit 2'",
              "if \"1 < 0\" then \"print a\" else 'quit 2'"}}) {
        INFO(c.line);
        s.write("q.in", kScript + c.line + "\n");
        /* The command is counted in words (it was "(logical command #6
         * in expanded script: <command>)"). */
        const std::string where = "the script q.in has a quit command, "
                                  "command 6 of the script with its include "
                                  "files read in (" + c.command + ")";
        CHECK(sim_error("--script q.in --varloss E 1 1") ==
              "PC_LAMMPS: " + where + quit_tail);
        CHECK(sim_error("--script q.in --outvar S", true) ==
              "PC_FAILPROBE: " + where + quit_tail);
    }
    s.write("ok.in", kScript + "print \"quit\"\n"
                               "if \"1 > 0\" then \"print quit\"\n");
    CHECK(sim_error("--script ok.in --varloss E 1 1") == "");
}

/* Pins: under FAILSCORE_GRADING OFF (the default) the settings that only
 * grade a failure are refused at startup at their line: PC_FAILPROBE --s0
 * and the <scale> of a --failvar; under ON both are accepted, and a
 * --failvar without a scale (which fails an evaluation under OFF too) is
 * accepted under OFF.
 * Before the fix: both were accepted under OFF and had no effect
 * (startup_error ""). */
TEST_CASE("PC lines: under FAILSCORE_GRADING OFF the settings that only "
          "grade are refused", "[HPO][pc_lines][failure_signals]") {
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    s.write("p.in", kScript);
    const std::string s0 = "PC_FAILPROBE --script p.in --outvar S --s0 2";
    const std::string scale =
        "PC_LAMMPS --script p.in --varloss E 1 1 --failvar bad 2";
    CHECK(startup_error(kSbplx + "PC_ERMSE 0 1\n" + s0 + "\n") ==
          at_line(8,
                  "PC_FAILPROBE --s0 grades failed evaluations, which "
                  "FAILSCORE_GRADING ON does; under FAILSCORE_GRADING OFF "
                  "every failed evaluation scores FAILSCORE, so --s0 would "
                  "have no effect",
                  s0));
    CHECK(startup_error(kSbplx + "PC_ERMSE 0 1\n" + scale + "\n") ==
          at_line(8,
                  "--failvar bad 2: the scale grades the failure, which "
                  "FAILSCORE_GRADING ON does; under FAILSCORE_GRADING OFF a "
                  "failed evaluation scores FAILSCORE, so the scale would "
                  "have no effect (--failvar bad alone fails the evaluation "
                  "when bad > 0)",
                  scale));
    const std::string on = "FAILSCORE_GRADING ON\nFAILSCORE 1e9\n";
    CHECK(startup_error(kSbplx + on + "PC_ERMSE 0 1\n" + s0 + "\n") == "");
    CHECK(startup_error(kSbplx + on + "PC_ERMSE 0 1\n" + scale + "\n") == "");
    CHECK(startup_error(kSbplx + "PC_ERMSE 0 1\n"
                        "PC_LAMMPS --script p.in --varloss E 1 1 --failvar "
                        "bad\n") == "");
}

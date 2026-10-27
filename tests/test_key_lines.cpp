// test_key_lines.cpp
//
// A refusal of one key of an optimiser block names the line of that key,
// not the line the block closes at: STEP (the block-wide STEP, a STEP
// <KEY> override), ALGO (a gradient-based optimiser beside a PC_LAMMPS or
// PC_FAILPROBE line, which the message names with its line) and
// GRAD_STRATEGY (beside a PC_LAMMPS, PC_FAILPROBE or PC_OBSERVABLE line). A
// refusal of a combination of keys, such as STEP <KEY> overrides that cover
// only some dimensions without a block-wide STEP, keeps the END line of the
// block. The parser's refusals carry the frame "Error on line N: <what>\n
// Line: <text>" (config_line_error).
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;

std::string unique_name(const std::string& stem)
{
    static unsigned counter = 0;
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A one-species model whose CGRID2B has two values: OPTIM CGRID2B (1) and
 * (2) name two search dimensions. */
Context make_ctx()
{
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1.0);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP 2 2 Ta Ta");
    c.add("CGRID2B", "1.0 2.0");
    c.add("SGRID2B", "0.5 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    c.add("OALGO", 4);
    c.add("LAMBDA", 1e-3);
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    return c;
}

/* A LAMMPS script a PC_LAMMPS or PC_FAILPROBE line can name: it sets the
 * variable the line reads. */
struct Script {
    fs::path path = fs::temp_directory_path() /
                    (unique_name("tadah_key_lines_") + ".in");
    Script()
    {
        std::ofstream f(path);
        f << "variable e equal 1.0\nvariable s equal 0.0\n";
    }
    ~Script()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* The message HPOTargets refuses `config` with ("" when it accepts it). */
std::string refusal(const std::string& config)
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_key_lines_") + ".hpo");
    {
        std::ofstream os(file);
        os << config;
    }
    std::string what;
    try {
        Context ctx = make_ctx();
        HPOTargets trg(ctx, file.string());
    } catch (const std::exception& e) {
        what = e.what();
    }
    std::error_code ec;
    fs::remove(file, ec);
    return what;
}

/* "Error on line <n>: <what>\nLine: <text>" */
std::string framed(std::size_t n, const std::string& what,
                   const std::string& text)
{
    return "Error on line " + std::to_string(n) + ": " + what + "\nLine: " +
           text;
}

bool ends_with(const std::string& text, const std::string& tail)
{
    return text.size() >= tail.size() &&
           text.compare(text.size() - tail.size(), tail.size(), tail) == 0;
}

const std::string kSpace = "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n";

}  // namespace

/* Pins: DLIB:GFS, which does not use STEP, refused at the STEP line, whose
 * text the frame echoes: "Error on line 5: DLIB:GFS does not use STEP;
 * MaxLIPO+TR sizes its own trust region from the samples; remove STEP\n
 * Line: STEP 0.1" with STEP after ALGO (line 5), and on line 3 with STEP
 * before LIB and ALGO; a STEP <KEY> override (no block-wide STEP) on line 5
 * names that line ("Line: STEP CGRID2B 0.1"); with both, the first of them
 * in the file. The block is named by its line, so the message has no
 * "(OUTER)".
 * Before the fix: every case named the ENDOPTIMIZER line ("Error on line
 * 8: DLIB:GFS (OUTER) does not use STEP; ...\nLine: ENDOPTIMIZER"). */
TEST_CASE("key lines: a STEP an optimiser does not use is refused at the "
          "STEP line", "[hpo][key_lines][step]")
{
    const std::string what =
        "DLIB:GFS does not use STEP; MaxLIPO+TR sizes its own trust region "
        "from the samples; remove STEP";
    const auto block = [](const std::string& before,
                          const std::string& after) {
        return "OPTIMIZER\nSTRATEGY\n" + before + "LIB DLIB\nALGO GFS\n" +
               after + "MAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n" + kSpace;
    };
    CHECK(refusal(block("", "STEP 0.1\n")) ==
          framed(5, what, "STEP 0.1"));
    CHECK(refusal(block("STEP 0.1\n", "")) ==
          framed(3, what, "STEP 0.1"));
    CHECK(refusal(block("", "STEP CGRID2B 0.1\n")) ==
          framed(5, what, "STEP CGRID2B 0.1"));
    CHECK(refusal(block("STEP CGRID2B(2) 0.2\n", "STEP 0.1\n")) ==
          framed(3, what, "STEP CGRID2B(2) 0.2"));
}

/* Pins: the value of a STEP is refused at its line: "Error on line 5:
 * NLOPT:LN_SBPLX: STEP must be a fraction of the OPTIM width in (0, 0.5]
 * (got 0.7); ...\nLine: STEP 0.7" for the block-wide STEP, and the
 * override's own line for "NLOPT:LN_SBPLX STEP CGRID2B: STEP must be > 0
 * and finite (got -0.1)" (line 6, after a valid block-wide STEP on line 5);
 * an override that names a key without an OPTIM line ("STEP override names
 * OPTIM key 'RCUT2B' which is not optimised (no OPTIM line for it)") and an
 * index that is not searched ("STEP CGRID2B(3): index 3 is not an OPTIM'd
 * index of CGRID2B") are refused at the override's line. Overrides that
 * cover only some dimensions without a block-wide STEP are a combination of
 * lines, refused at ENDOPTIMIZER (line 8) as before.
 * Before the fix: every one named ENDOPTIMIZER, and the value refusals said
 * "NLOPT:LN_SBPLX (OUTER)". */
TEST_CASE("key lines: a STEP value or a STEP <KEY> override is refused at "
          "its line", "[hpo][key_lines][step]")
{
    const auto block = [](const std::string& steps) {
        return "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LN_SBPLX\n" + steps +
               "MAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n" + kSpace;
    };
    CHECK(refusal(block("STEP 0.7\n")) ==
          framed(5,
                 "NLOPT:LN_SBPLX: STEP must be a fraction of the OPTIM width "
                 "in (0, 0.5] (got 0.7); values above 0.5 cannot satisfy "
                 "BOBYQA's width >= 2*step rule and exceed NLopt's own 0.25 "
                 "default",
                 "STEP 0.7"));
    CHECK(refusal(block("STEP 0.1\nSTEP CGRID2B -0.1\n")) ==
          framed(6,
                 "NLOPT:LN_SBPLX STEP CGRID2B: STEP must be > 0 and finite "
                 "(got -0.1)",
                 "STEP CGRID2B -0.1"));
    CHECK(refusal(block("STEP 0.1\nSTEP RCUT2B 0.2\n")) ==
          framed(6,
                 "STEP override names OPTIM key 'RCUT2B' which is not "
                 "optimised (no OPTIM line for it)",
                 "STEP RCUT2B 0.2"));
    CHECK(refusal(block("STEP CGRID2B(1,2) 0.2\nSTEP CGRID2B(3) 0.1\n")) ==
          framed(6,
                 "STEP CGRID2B(3): index 3 is not an OPTIM'd index of CGRID2B",
                 "STEP CGRID2B(3) 0.1"));
    CHECK(refusal(block("STEP CGRID2B(1) 0.2\n")) ==
          framed(8,
                 "STEP overrides cover only some dimensions and no block-wide "
                 "STEP is set; add `STEP <frac>` or override every OPTIM'd "
                 "dimension",
                 "ENDOPTIMIZER"));
}

/* Pins: beside a PC_LAMMPS line (line 11), a gradient-based optimiser is
 * refused at its ALGO line, and the message names the PC_LAMMPS line:
 * "Error on line 4: --hpotarget defines PC_LAMMPS on line 11 but
 * OPTIMIZER's optimiser NLOPT:LD_LBFGS is gradient-based. ...\nLine: ALGO
 * LD_LBFGS" -- in the STRATEGY block, in an EXPLORE block ("EXPLORE's", the
 * EXPLORE block's ALGO line) and in an INNER block ("INNER's", the INNER
 * block's ALGO line); a GRAD_STRATEGY line beside a PC_FAILPROBE line (and
 * no PC_LAMMPS line) is refused at the GRAD_STRATEGY line: "--hpotarget
 * defines PC_FAILPROBE on line 12 but OPTIMIZER's GRAD_STRATEGY is set.
 * ...".
 * Before the fix: each named the END line of its block (ENDOPTIMIZER,
 * ENDEXPLORE, ENDINNER) and no PC line ("--hpotarget defines PC_LAMMPS
 * but ..."). */
TEST_CASE("key lines: a gradient beside a PC_LAMMPS line is refused at the "
          "ALGO or GRAD_STRATEGY line, naming the PC_LAMMPS line",
          "[hpo][key_lines][pc_lammps]")
{
    const Script script;
    const std::string pc = "PC_LAMMPS --script " + script.path.string() +
                           " --varloss e 1.0 100\n";
    const std::string free_list =
        " A PC_LAMMPS or PC_FAILPROBE line requires a gradient-free "
        "optimiser (";
    {
        // lines 1-8 the OPTIMIZER block, 9-10 OPTIM, 11 PC_LAMMPS
        const std::string config =
            "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 10\n"
            "ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n" + kSpace + pc;
        const std::string got = refusal(config);
        INFO(got);
        CHECK(got.rfind("Error on line 4: --hpotarget defines PC_LAMMPS on "
                        "line 11 but OPTIMIZER's optimiser NLOPT:LD_LBFGS is "
                        "gradient-based." + free_list, 0) == 0);
        CHECK(ends_with(got, "\nLine: ALGO LD_LBFGS"));
    }
    {
        // EXPLORE on lines 6-10, its ALGO on line 8; PC_LAMMPS on line 17
        const std::string config =
            "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_SBPLX\n"
            "EXPLORE\nLIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 10\nENDEXPLORE\n"
            "MAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n" + kSpace + "LOSS L2\n" +
            pc;
        const std::string got = refusal(config);
        INFO(got);
        CHECK(got.rfind("Error on line 8: --hpotarget defines PC_LAMMPS on "
                        "line 17 but EXPLORE's optimiser NLOPT:LD_LBFGS is "
                        "gradient-based." + free_list, 0) == 0);
        CHECK(ends_with(got, "\nLine: ALGO LD_LBFGS"));
    }
    {
        // INNER on lines 6-9, its ALGO on line 8; PC_LAMMPS on line 15
        const std::string config =
            "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO GN_MLSL_LDS\nMAXEVAL 10\n"
            "INNER\nLIB NLOPT\nALGO LD_LBFGS\nENDINNER\nENDSTRATEGY\n"
            "ENDOPTIMIZER\nLOSS L2\n" + kSpace + pc;
        const std::string got = refusal(config);
        INFO(got);
        CHECK(got.rfind("Error on line 8: --hpotarget defines PC_LAMMPS on "
                        "line 15 but INNER's optimiser NLOPT:LD_LBFGS is "
                        "gradient-based. A PC_LAMMPS or PC_FAILPROBE line "
                        "requires a gradient-free optimiser, and an INNER "
                        "search is a local NLopt algorithm: name a "
                        "derivative-free one in the INNER block (", 0) == 0);
        CHECK(ends_with(got, "\nLine: ALGO LD_LBFGS"));
    }
    {
        // GRAD_STRATEGY on line 5, PC_FAILPROBE on line 12
        const std::string config =
            "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LN_SBPLX\n"
            "GRAD_STRATEGY NUMERICAL\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n" +
            kSpace + "LOSS L2\nPC_FAILPROBE --script " +
            script.path.string() + " --outvar s\nPC_ERMSE 0 1\n";
        CHECK(refusal(config) ==
              framed(5,
                     "--hpotarget defines PC_FAILPROBE on line 12 but "
                     "OPTIMIZER's GRAD_STRATEGY is set. A PC_LAMMPS or "
                     "PC_FAILPROBE line does not support analytical/numerical/"
                     "hybrid gradients. Drop the GRAD_STRATEGY line.",
                     "GRAD_STRATEGY NUMERICAL"));
    }
}

/* Pins: GRAD_STRATEGY NUMERICAL beside a PC_OBSERVABLE line is refused at
 * the GRAD_STRATEGY line (line 5): "Error on line 5: --hpotarget defines
 * PC_OBSERVABLE but OPTIMIZER's GRAD_STRATEGY=NUMERICAL is set. ...\nLine:
 * GRAD_STRATEGY NUMERICAL".
 * Before the fix: "Error on line 8: ...\nLine: ENDOPTIMIZER". */
TEST_CASE("key lines: GRAD_STRATEGY NUMERICAL beside a PC_OBSERVABLE line is "
          "refused at the GRAD_STRATEGY line",
          "[hpo][key_lines][pc_observable]")
{
    const std::string config =
        "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LD_LBFGS\n"
        "GRAD_STRATEGY NUMERICAL\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n"
        "LOSS L2\nPC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
        "--weight 1000 --hotvar a0 2.7 3.9 3.3\n" + kSpace;
    const std::string got = refusal(config);
    INFO(got);
    CHECK(got.rfind("Error on line 5: --hpotarget defines PC_OBSERVABLE but "
                    "OPTIMIZER's GRAD_STRATEGY=NUMERICAL is set.", 0) == 0);
    CHECK(ends_with(got, "\nLine: GRAD_STRATEGY NUMERICAL"));
}

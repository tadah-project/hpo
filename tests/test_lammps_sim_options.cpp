// test_lammps_sim_options.cpp
//
// LammpsSim::parse_tokens() reads HPO's options on a PC_LAMMPS line
// (--script, --varloss, --invar, --outvar, --failscore, --maxtime, --hotvar,
// --failvar) and passes every other option, with the values that follow
// it, to LAMMPS as command-line arguments. Every LAMMPS command-line option
// starts with a single '-' (src/lammps.cpp at the pinned commit), so an
// option that starts with "--" and is not one of HPO's is a typo: it is an
// error naming HPO's options, not an argument every LAMMPS run rejects.

#include "catch2/catch.hpp"

#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct TempScript {
    std::string path;
    TempScript() {
        static int counter = 0;
        path = (std::filesystem::temp_directory_path()
                / ("test_lammps_sim_options_" + std::to_string(::getpid())
                   + "_" + std::to_string(++counter) + ".in")).string();
        std::ofstream os(path);
        os << "clear\nrun 0\nvariable E equal 0.0\nvariable S equal 0.0\n";
    }
    ~TempScript() { std::remove(path.c_str()); }
};

std::string sim_error(std::string cmd, bool failprobe = false)
{
    try {
        tadah::hpo::LammpsSim sim(cmd, "1", 1.0e6, tadah::LossFunction{},
                                  failprobe);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

const std::string kHpoOptions =
    "--script, --varloss, --invar, --outvar, --failscore, --maxtime, "
    "--hotvar and --failvar";
const std::string kLammpsNote =
    "; LAMMPS command-line options start with a single '-'";

}  // namespace

/* Pins: the exact message for an unknown "--" option: the line's kind, the
 * option, the closest HPO option when it is within two edits, HPO's
 * options (PC_FAILPROBE's --gate and --s0 on a probe line), and where
 * LAMMPS options go; --gate and --s0 on a PC_LAMMPS line are named as
 * PC_FAILPROBE options.
 * Before the fix: sim_error() was "" for --hotvars, --foo and --gate (the
 * option and its values went to LAMMPS, and every evaluation failed with
 * "Invalid command-line argument"), and "LammpsSim: --varloss variable is
 * not defined." for --varlos. */
TEST_CASE("PC_LAMMPS: an unknown -- option is an error naming HPO's options",
          "[HPO][lammps_sim][parse]")
{
    TempScript s;
    const std::string head = "--script " + s.path + " ";
    struct Case { std::string cmd; bool failprobe; std::string message; };
    for (const auto& c : std::vector<Case>{
             {head + "--varloss E 1 1 --hotvars a0 1 2", false,
              "PC_LAMMPS: unknown option '--hotvars' (did you mean "
              "'--hotvar'?): HPO's options are " + kHpoOptions + kLammpsNote},
             {head + "--varlos E 1 1", false,
              "PC_LAMMPS: unknown option '--varlos' (did you mean "
              "'--varloss'?): HPO's options are " + kHpoOptions + kLammpsNote},
             {head + "--varloss E 1 1 --outvars S", false,
              "PC_LAMMPS: unknown option '--outvars' (did you mean "
              "'--outvar'?): HPO's options are " + kHpoOptions + kLammpsNote},
             {head + "--varloss E 1 1 --foo 3", false,
              "PC_LAMMPS: unknown option '--foo': HPO's options are " +
                  kHpoOptions + kLammpsNote},
             {head + "--varloss E 1 1 --gate 0.5", false,
              "PC_LAMMPS: unknown option '--gate' (a PC_FAILPROBE option): "
              "HPO's options are " + kHpoOptions + kLammpsNote},
             {head + "--varloss E 1 1 --s0 0.5", false,
              "PC_LAMMPS: unknown option '--s0' (a PC_FAILPROBE option): "
              "HPO's options are " + kHpoOptions + kLammpsNote},
             /* After a LAMMPS option's values too. */
             {head + "--varloss E 1 1 -sf omp --maxtme 10", false,
              "PC_LAMMPS: unknown option '--maxtme' (did you mean "
              "'--maxtime'?): HPO's options are " + kHpoOptions +
                  kLammpsNote},
             {head + "--outvar S --foo", true,
              "PC_FAILPROBE: unknown option '--foo': HPO's options are "
              "--script, --varloss, --invar, --outvar, --failscore, "
              "--maxtime, --hotvar, --failvar, --gate and --s0" +
                  kLammpsNote},
             {head + "--outvar S --gates 1", true,
              "PC_FAILPROBE: unknown option '--gates' (did you mean "
              "'--gate'?): HPO's options are --script, --varloss, --invar, "
              "--outvar, --failscore, --maxtime, --hotvar, --failvar, --gate "
              "and --s0" + kLammpsNote}}) {
        INFO(c.cmd);
        CHECK(sim_error(c.cmd, c.failprobe) == c.message);
    }
}

/* Pins: a single-dash option and every value up to the next option go to
 * LAMMPS unchanged, in order, before the -screen and -var maxtime that
 * HPO adds (-log is the user's here, so HPO adds none). LAMMPS judges its
 * own options: HPO keeps no copy of LAMMPS's option table.
 * Before the fix: passes as well (a pin of the arguments the new "--"
 * check must leave alone). */
TEST_CASE("PC_LAMMPS: single-dash options and their values go to LAMMPS",
          "[HPO][lammps_sim][parse]")
{
    TempScript s;
    std::string cmd = "--script " + s.path +
                      " --varloss E 1 1 -sf omp -var x 1 2 -log none";
    tadah::hpo::LammpsSim sim(cmd, "1", 1.0e6, tadah::LossFunction{});
    REQUIRE_FALSE(sim.invar.empty());
    const std::vector<std::string> expected = {
        sim.invar.front(), "-sf", "omp", "-var", "x", "1", "2", "-log", "none",
        "-screen", "none", "-var", "maxtime", "1 every 1"};
    CHECK(sim.invar == expected);
}

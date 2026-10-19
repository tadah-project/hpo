// test_sim_timeout.cpp
//
// Per-sim wall-clock enforcement (root MAXTIME / per-sim --maxtime,
// SECONDS). LammpsSim injects `timer timeout ${maxtime}` after every
// `clear` (and at line 0 when the script does not start with one); the
// runner reports a fired timeout as a failed sim; a cached instance stays
// healthy for the next script. Enforcement is opt-in: the default
// MAX_TIME "off" injects nothing, and a script issuing its own `timer`
// command keeps full control.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/hpo_targets.h>   // PC_SCRIPT_TIMEOUT / MAXTIME keyword parse

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct ScopedTempFile {
  std::filesystem::path path;

  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_hpo_timeout_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
    os.close();
    REQUIRE(os.good());
  }

  ~ScopedTempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

std::size_t count_timer_lines(const std::vector<std::string>& lines) {
  std::size_t n = 0;
  for (const auto& l : lines)
    if (l.rfind("timer timeout", 0) == 0) ++n;
  return n;
}

int index_of_prefix(const std::vector<std::string>& lines,
                    const std::string& prefix) {
  for (std::size_t i = 0; i < lines.size(); ++i)
    if (lines[i].rfind(prefix, 0) == 0) return static_cast<int>(i);
  return -1;
}

const char* SIMPLE_BODY = R"(clear
units lj
variable foo equal 1.0
print "ok"
)";

}  // namespace

TEST_CASE("LammpsSim injects timer timeout after clear when maxtime is set",
          "[lammps_sim][timeout]") {
    ScopedTempFile script(SIMPLE_BODY);
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "5", 1.0e6, tadah::LossFunction{});

    REQUIRE(count_timer_lines(sim.script_lines) == 1);
    const int i_clear = index_of_prefix(sim.script_lines, "clear");
    const int i_timer = index_of_prefix(sim.script_lines, "timer timeout");
    REQUIRE(i_clear == 0);
    REQUIRE(i_timer == 1);           // immediately after the leading clear
    REQUIRE(sim.script_lines[i_timer].find("${maxtime}") != std::string::npos);
}

TEST_CASE("LammpsSim injects at line 0 when the script does not start with clear",
          "[lammps_sim][timeout]") {
    ScopedTempFile script("units lj\nvariable foo equal 1.0\nclear\n"
                          "units lj\nvariable foo equal 1.0\nprint \"x\"\n");
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "5", 1.0e6, tadah::LossFunction{});

    /* one at index 0 + one after the mid-script clear */
    REQUIRE(count_timer_lines(sim.script_lines) == 2);
    REQUIRE(sim.script_lines[0].rfind("timer timeout", 0) == 0);
    const int i_clear = index_of_prefix(sim.script_lines, "clear");
    REQUIRE(sim.script_lines[i_clear + 1].rfind("timer timeout", 0) == 0);
}

TEST_CASE("LammpsSim does not inject when maxtime is off (default)",
          "[lammps_sim][timeout]") {
    ScopedTempFile script(SIMPLE_BODY);
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, tadah::LossFunction{});
    REQUIRE(count_timer_lines(sim.script_lines) == 0);
}

TEST_CASE("LammpsSim does not inject when maxtime is unlimited",
          "[lammps_sim][timeout]") {
    /* 'unlimited' is LAMMPS' other spelling of no limit. Injected, `timer
     * timeout unlimited` leaves $(timeremain) at 0.0, so the section guard
     * aborted every multi-section script as timed out. */
    ScopedTempFile script(SIMPLE_BODY);
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0";
    tadah::hpo::LammpsSim root(cmd, "unlimited", 1.0e6, tadah::LossFunction{});
    CHECK(count_timer_lines(root.script_lines) == 0);

    std::string own = cmd + " --maxtime unlimited";
    tadah::hpo::LammpsSim line(own, "5", 1.0e6, tadah::LossFunction{});
    CHECK(count_timer_lines(line.script_lines) == 0);
}

TEST_CASE("LammpsSim does not inject when the script has its own timer",
          "[lammps_sim][timeout]") {
    ScopedTempFile script("clear\nunits lj\ntimer timeout 30 every 10\n"
                          "variable foo equal 1.0\n");
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "5", 1.0e6, tadah::LossFunction{});

    /* exactly the user's own line, no ${maxtime} injection */
    std::size_t n_injected = 0;
    for (const auto& l : sim.script_lines)
        if (l.find("${maxtime}") != std::string::npos) ++n_injected;
    REQUIRE(n_injected == 0);
}

TEST_CASE("fired timer timeout fails the sim and leaves the instance healthy",
          "[lammps_sim][timeout][.][slow]") {
    /* Long NVE run on an LJ melt: guaranteed to outlive a 1 s budget.
     * `timer timeout 1 every 1` is checked every step and ends the run
     * cleanly, so run_script must return false via Timer::is_timeout(). */
    ScopedTempFile script(R"(clear
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 12 0 12 0 12
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify delay 0 every 20 check no
fix 1 all nve
variable foo equal 1.0
run 500000000
)");
    std::string cmd = "--script " + script.path.string() +
                      " --varloss foo 0.0 1.0 --maxtime 1";
    tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, tadah::LossFunction{});
    REQUIRE(count_timer_lines(sim.script_lines) == 1);

    LammpsRunner runner;
    /* Same -var registration the HPO host performs (sim.invar carries it,
     * here passed explicitly): ${maxtime} -> "1 every 1". */
    auto lmp = runner.create_instance(
        {"lmp", "-log", "none", "-screen", "none",
            "-var", "maxtime", "1 every 1"});
    REQUIRE(lmp != nullptr);

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = runner.run_script(lmp, sim.script_lines,
                                      tadah::core::Context{});
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    REQUIRE_FALSE(ok);        // timeout reported as a failed sim
    REQUIRE(secs < 30.0);     // fired near 1 s, not after 500M steps

    /* Cached-instance reuse: the next script's clear rebuilds the Timer. */
    const std::vector<std::string> next = {
        "clear", "units lj", "print \"healthy\""};
    REQUIRE(runner.run_script(lmp, next, tadah::core::Context{}));
}

/* ---------------------------------------------------------------------------
 * Config-keyword parse: PC_SCRIPT_TIMEOUT is the self-describing name for the
 * per-pc_script wall-clock limit; MAXTIME is the legacy alias. Both feed the
 * same MAX_TIME string; if both appear the later line wins. (The OPTIMIZER-
 * block MAXTIME is a different setting and is not exercised here.)
 * ------------------------------------------------------------------------- */
namespace { const char* MIN_OPT = "OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n"; }

TEST_CASE("PC_SCRIPT_TIMEOUT sets the per-script limit",
          "[lammps_sim][timeout][config]") {
    ScopedTempFile cfg(std::string(MIN_OPT) + "PC_SCRIPT_TIMEOUT 5\n");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.MAX_TIME == "5");
    REQUIRE(trg.MAX_TIME_seen_new);
    REQUIRE_FALSE(trg.MAX_TIME_seen_legacy);
    REQUIRE(trg.MAX_TIME_key == "PC_SCRIPT_TIMEOUT");
}

TEST_CASE("legacy MAXTIME still sets the per-script limit (alias)",
          "[lammps_sim][timeout][config]") {
    ScopedTempFile cfg(std::string(MIN_OPT) + "MAXTIME 7\n");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.MAX_TIME == "7");
    REQUIRE(trg.MAX_TIME_seen_legacy);
    REQUIRE_FALSE(trg.MAX_TIME_seen_new);
    REQUIRE(trg.MAX_TIME_key == "MAXTIME");
}

TEST_CASE("PC_SCRIPT_TIMEOUT and legacy MAXTIME set an identical value",
          "[lammps_sim][timeout][config]") {
    ScopedTempFile a(std::string(MIN_OPT) + "PC_SCRIPT_TIMEOUT 3\n");
    ScopedTempFile b(std::string(MIN_OPT) + "MAXTIME 3\n");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets ta(ctx, a.path.string());
    tadah::hpo::HPOTargets tb(ctx, b.path.string());
    REQUIRE(ta.MAX_TIME == tb.MAX_TIME);   // downstream timer injection identical
}

/* MAXTIME and PC_SCRIPT_TIMEOUT are one key: set again only after a
 * PC_LAMMPS line has taken the first value, else refused naming the first
 * (it was: the later line replaced the first). With a PC_LAMMPS line
 * between, that line keeps the first value and the later line is the
 * run-wide value; both flags are set. */
TEST_CASE("both timeout keys: one key, set again after a PC_LAMMPS line "
          "takes the first",
          "[lammps_sim][timeout][config]") {
    tadah::core::Context ctx;
    ScopedTempFile twice(std::string(MIN_OPT) +
                         "MAXTIME 7\nPC_SCRIPT_TIMEOUT 5\n");
    REQUIRE_THROWS_WITH(
        tadah::hpo::HPOTargets(ctx, twice.path.string()),
        Catch::Contains("Error on line 6: PC_SCRIPT_TIMEOUT is set twice "
                        "(first on line 5, as MAXTIME) and no PC_LAMMPS or "
                        "PC_FAILPROBE line between takes the first"));
    ScopedTempFile script(SIMPLE_BODY);
    ScopedTempFile cfg(std::string(MIN_OPT) + "MAXTIME 7\nPC_LAMMPS --script " +
                       script.path.string() +
                       " --varloss foo 0.0 1.0\nPC_SCRIPT_TIMEOUT 5\n");
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.LS.size() == 1);
    REQUIRE(trg.LS[0].maxtime == "7");     // the line takes the first
    REQUIRE(trg.MAX_TIME == "5");          // the run-wide value: the later
    REQUIRE(trg.MAX_TIME_seen_new);
    REQUIRE(trg.MAX_TIME_seen_legacy);
    REQUIRE(trg.MAX_TIME_key == "PC_SCRIPT_TIMEOUT");
}

TEST_CASE("no timeout key: defaults to off, no flags",
          "[lammps_sim][timeout][config]") {
    ScopedTempFile cfg(MIN_OPT);
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.MAX_TIME == "off");
    REQUIRE_FALSE(trg.MAX_TIME_seen_new);
    REQUIRE_FALSE(trg.MAX_TIME_seen_legacy);
}

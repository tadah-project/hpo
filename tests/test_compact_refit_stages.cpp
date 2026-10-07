// `tadah refit` compact mode: what a stage block's keys set, end to end
// through the HPO refit engine on the synthesised setfl of
// compact_refit_harness.h (Blip5, small budgets).
#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include "compact_refit_harness.h"

#include <sstream>
#include <string>
#include <vector>

using tadah::core::Context;
using namespace compact_refit_test;

namespace {

// The lines of refit_opt.log containing `needle`, without their timestamps
// ("[2026-10-07 06:45:11] ").
std::vector<std::string> log_lines(const std::string &needle) {
  std::vector<std::string> out;
  std::istringstream in(slurp("refit_opt.log"));
  for (std::string line; std::getline(in, line);) {
    if (line.find(needle) == std::string::npos) continue;
    const auto stamp = line.find("] ");
    out.push_back(line.rfind("[", 0) == 0 && stamp != std::string::npos
                      ? line.substr(stamp + 2)
                      : line);
  }
  return out;
}

// The FUSE line of function `fn` (phi, rho, F) among `lines`, "" if none.
std::string fuse_line(const std::vector<std::string> &lines,
                      const std::string &fn) {
  for (const auto &l : lines)
    if (l.rfind("FUSE[" + fn + "] curve score", 0) == 0) return l;
  return "";
}

// A curve-mode refit with REFINE ALGO NONE (so the ladder leaves the FUSE
// refine something to do) and the FUSE block `fuse`: the FUSE lines of
// refit_opt.log and the written pot.tadah.
struct CurveRun {
  std::vector<std::string> fuse;
  std::string pot;
};
CurveRun curve_run(const std::string &fuse) {
  Context c = compact_context("SIGNED", "OPTIMIZER ALGO NONE ENDOPTIMIZER",
                              fuse);
  c.add("REFIT_OBJECTIVE", std::string("curve"));
  const auto [what, logged] = run_compact(c);
  INFO(logged);
  REQUIRE(what.empty());
  return {log_lines("FUSE["), slurp("pot.tadah")};
}
std::vector<std::string> curve_fuse_lines(const std::string &fuse) {
  return curve_run(fuse).fuse;
}

}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact, curve mode: the FUSE refine of each "
                 "function runs with FUSE's MAXEVAL, FTOL_REL and GTOL",
                 "[compact][refit][fuse_curve]") {
  /* Pins: under REFIT_OBJECTIVE curve the FUSE curve refine of phi, rho
   * and F runs with FUSE's own Ceres settings, each function with the
   * whole FUSE MAXEVAL: with MAXEVAL 1 every FUSE[phi|rho|F] line logs
   * its own refine's term, NO_CONVERGENCE after 2 iterations (the start
   * and one step) at maxiter, and the lines and the written pot.tadah
   * differ from those of MAXEVAL 200; FUSE FTOL_REL 0.99 ends every refine by its function tolerance
   * (ftol); FUSE GTOL 1e10 ends every one at its start by its gradient
   * tolerance (CONVERGENCE(1it, gtol); the radial ones in their edge
   * parametrisation); the optimiser: line prints FUSE's gtol.
   * Before the fix: the curve FUSE refine ran with REFINE's MAXEVAL/
   * MAXITER, FTOL_REL and GTOL (here REFINE's defaults: 1000 iterations,
   * 1e-15, 1e-14), so the FUSE lines and pot.tadah of MAXEVAL 1 and 200
   * were identical and the lines logged no term, and FUSE GTOL was an error ("FUSE: GTOL is not
   * used by a tadah refit stage (...)"). */
  SECTION("FUSE MAXEVAL is each function's Ceres iteration budget") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const CurveRun run1 =
        curve_run("OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 1 ENDOPTIMIZER");
    const auto optimiser = log_lines("optimiser: ");
    const CurveRun run200 =
        curve_run("OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 200 ENDOPTIMIZER");
    const auto &one = run1.fuse, &many = run200.fuse;
    INFO("MAXEVAL 1:\n" << fuse_line(one, "phi") << "\n"
                        << fuse_line(one, "rho") << "\n"
                        << fuse_line(one, "F") << "\nMAXEVAL 200:\n"
                        << fuse_line(many, "phi") << "\n"
                        << fuse_line(many, "rho") << "\n"
                        << fuse_line(many, "F"));
    REQUIRE(one.size() == 3u);
    REQUIRE(many.size() == 3u);
    CHECK(one != many);
    REQUIRE_FALSE(run1.pot.empty());
    CHECK(run1.pot != run200.pot);  // the fitted potential, not just the log
    CHECK(fuse_line(one, "phi").find("(NO_CONVERGENCE(2it, maxiter, edges))") !=
          std::string::npos);
    CHECK(fuse_line(one, "rho").find("(NO_CONVERGENCE(2it, maxiter, edges))") !=
          std::string::npos);
    CHECK(fuse_line(one, "F").find("(NO_CONVERGENCE(2it, maxiter))") !=
          std::string::npos);
    REQUIRE(optimiser.size() == 1u);
    CHECK(optimiser[0].find("FUSE{algo=CERES_DOGLEG maxiter=1 ftol_rel=1e-15 "
                            "gtol=1e-14}") != std::string::npos);
  }
  SECTION("FUSE FTOL_REL is each refine's function tolerance") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const auto lines = curve_fuse_lines(
        "OPTIMIZER LIB CERES ALGO LM FTOL_REL 0.99 ENDOPTIMIZER");
    REQUIRE(lines.size() == 3u);
    for (const char *fn : {"phi", "rho", "F"}) {
      INFO(fuse_line(lines, fn));
      CHECK(fuse_line(lines, fn).find("(CONVERGENCE(") != std::string::npos);
      CHECK(fuse_line(lines, fn).find("it, ftol") != std::string::npos);
    }
  }
  SECTION("FUSE GTOL is each refine's gradient tolerance") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const auto lines = curve_fuse_lines(
        "OPTIMIZER LIB CERES ALGO LM GTOL 1e10 ENDOPTIMIZER");
    REQUIRE(lines.size() == 3u);
    CHECK(fuse_line(lines, "phi").find("(CONVERGENCE(1it, gtol, edges))") !=
          std::string::npos);
    CHECK(fuse_line(lines, "rho").find("(CONVERGENCE(1it, gtol, edges))") !=
          std::string::npos);
    CHECK(fuse_line(lines, "F").find("(CONVERGENCE(1it, gtol))") !=
          std::string::npos);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact, energy mode: FUSE GTOL is the joint "
                 "solve's gradient tolerance",
                 "[compact][refit][fuse_gtol]") {
  /* Pins: FUSE reads GTOL in energy mode too: with GTOL 1e10 the joint
   * solve ends at its start by its gradient tolerance (the joint line's
   * term is CONVERGENCE(1it, gtol, ...)); without it the joint solve
   * takes steps (its default gradient tolerance is 1e-14, as before).
   * Before the fix: FUSE GTOL was an error ("FUSE: GTOL is not used by a
   * tadah refit stage (...)"), and the joint solve's gradient tolerance
   * was the constant 1e-14. */
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  const auto [what, logged] = run_compact(compact_context(
      "SIGNED", kDefaultRefine,
      "OPTIMIZER LIB CERES ALGO DOGLEG GTOL 1e10 ENDOPTIMIZER"));
  INFO(logged);
  REQUIRE(what.empty());
  const auto joint = log_lines("joint: max |dE|/tol");
  REQUIRE_FALSE(joint.empty());
  INFO(joint.back());
  CHECK(joint.back().find("term=CONVERGENCE(1it, gtol, cost ") !=
        std::string::npos);
  {
    ScopedTempDir dflt;
    write_synth_eam("synth.eam.alloy");
    const auto [w, l] = run_compact(compact_context("SIGNED"));
    REQUIRE(w.empty());
    const auto j = log_lines("joint: max |dE|/tol");
    REQUIRE_FALSE(j.empty());
    INFO(j.back());
    CHECK(j.back().find("term=CONVERGENCE(1it, gtol") == std::string::npos);
  }
}

#endif  // TADAH_USE_CERES

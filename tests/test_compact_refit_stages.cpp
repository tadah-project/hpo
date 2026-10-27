// `tadah refit` compact mode: what a stage block's keys set, end to end
// through the HPO refit engine on the synthesised setfl of
// compact_refit_harness.h (Blip5, small budgets).
#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include "compact_refit_harness.h"

#include <tadah/hpo/refit/refit_parser.h>
#include <tadah/models/eam/eam_refit.h>
#include <tadah/models/eam/eam_setfl_io.h>

#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
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

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the optimiser line gives the keys "
                 "each stage's ALGO reads",
                 "[compact][refit]") {
  /* Pins: refit_opt.log's optimiser: line gives, for each stage, the
   * settings of the algorithm that runs it and only those: the MLSL seed
   * its budget and its local LN_SBPLX searches' (the INNER block's
   * MAXEVAL, FTOL_REL, XTOL_REL), an NLopt INNER search maxeval ftol_rel
   * xtol_rel, dlib's LBFGS (INNER, REFINE) maxeval ftol_abs, NLopt's
   * LD_LBFGS REFINE maxeval ftol_rel, Ceres maxiter ftol_rel gtol, and
   * ALGO NONE nothing.
   * Before the fix: every stage printed every setting of its block
   * (REFINE's gtol with LD_LBFGS, which never reads it, the MLSL seed
   * without its local searches), and FUSE ALGO NONE its unused ones; the
   * second run is refused (FTOL_ABS is not accepted). */
  SECTION("MLSL, an NLopt INNER search, LD_LBFGS and DOGLEG") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const auto [what, logged] = run_compact(compact_context(
        "SIGNED", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS MAXEVAL 50 ENDOPTIMIZER"));
    INFO(logged);
    REQUIRE(what.empty());
    const auto line = log_lines("optimiser: ");
    REQUIRE(line.size() == 1u);
    CHECK(line[0] ==
          "optimiser: EXPLORE.seed{algo=MLSL maxeval=4000 local=LN_SBPLX{"
          "maxeval=1000 ftol_rel=1e-13 xtol_rel=1e-11}}  EXPLORE.inner{"
          "algo=SBPLX maxeval=1000 ftol_rel=1e-13 xtol_rel=1e-11}  REFINE{"
          "algo=NLOPT_LBFGS maxeval=50 ftol_rel=1e-15}  FUSE{"
          "algo=CERES_DOGLEG maxiter=200 ftol_rel=1e-15 gtol=1e-14}");
  }
  SECTION("MLSL with an INNER dlib LBFGS: the seed's local searches take "
          "the INNER FTOL_REL and XTOL_REL") {
    /* Before the fix: passes as well (a pin: under GN_MLSL_LDS an INNER
     * dlib LBFGS block's FTOL_REL and XTOL_REL set MLSL's local LN_SBPLX
     * searches, and dlib's LBFGS keeps its FTOL_ABS default). */
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context(
        "SIGNED", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS MAXEVAL 50 ENDOPTIMIZER");
    c.remove("EXPLORE");
    add_block(c, "EXPLORE",
              "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 4000 INNER LIB "
              "DLIB ALGO LBFGS MAXEVAL 1000 FTOL_REL 1e-7 XTOL_REL 1e-6 "
              "ENDINNER ENDOPTIMIZER");
    const auto [what, logged] = run_compact(c);
    INFO(logged);
    REQUIRE(what.empty());
    const auto line = log_lines("optimiser: ");
    REQUIRE(line.size() == 1u);
    CHECK(line[0].rfind(
              "optimiser: EXPLORE.seed{algo=MLSL maxeval=4000 local=LN_SBPLX{"
              "maxeval=1000 ftol_rel=1e-07 xtol_rel=1e-06}}  EXPLORE.inner{"
              "algo=DLIB_LBFGS maxeval=1000 ftol_abs=1e-13}  REFINE{",
              0) == 0);
  }
  SECTION("CRS2, dlib's LBFGS in INNER and REFINE, FUSE NONE") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context(
        "SIGNED",
        "OPTIMIZER LIB DLIB ALGO LBFGS MAXEVAL 50 FTOL_ABS 1e-9 ENDOPTIMIZER",
        "OPTIMIZER ALGO NONE ENDOPTIMIZER");
    c.remove("EXPLORE");
    add_block(c, "EXPLORE",
              "OPTIMIZER LIB NLOPT ALGO GN_CRS2_LM MAXEVAL 400 INNER LIB DLIB "
              "ALGO LBFGS MAXEVAL 100 FTOL_ABS 1e-10 ENDINNER ENDOPTIMIZER");
    const auto [what, logged] = run_compact(c);
    INFO(logged);
    REQUIRE(what.empty());
    const auto line = log_lines("optimiser: ");
    REQUIRE(line.size() == 1u);
    CHECK(line[0] ==
          "optimiser: EXPLORE.seed{algo=CRS2 maxeval=400}  EXPLORE.inner{"
          "algo=DLIB_LBFGS maxeval=100 ftol_abs=1e-10}  REFINE{"
          "algo=DLIB_LBFGS maxeval=50 ftol_abs=1e-09}  FUSE{algo=NONE}");
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the configuration numbers are read "
                 "strictly, before any fitting",
                 "[compact][refit][strict_numbers]") {
  /* Pins: the compact refit reads its numbers with the strict reader of
   * MLIP (read_refit_numbers): REFIT_COMPACT_NMAX '+3' is refused naming
   * the token, before any ladder runs; REFIT_COMPACT_SEED 3000000000, an
   * unsigned seed, runs; a REFIT_COMPACT_NMAX whose curve sample grid
   * (4 x NMAX points per function) does not fit an int is refused naming
   * the grid, before any ladder runs.
   * Before the fix: std::stoi read '+3' as 3 and the refit ran; SEED was
   * read as an int and 3000000000 failed converting; 4 * NMAX overflowed
   * an int (undefined behaviour) and the refit ran with an 800-point grid. */
  SECTION("REFIT_COMPACT_NMAX +3") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("SIGNED");
    c.remove("REFIT_COMPACT_NMAX");
    c.add("REFIT_COMPACT_NMAX", std::string("+3"));
    const auto [what, logged] = run_compact(c);
    CHECK(what == "tadah refit: REFIT_COMPACT_NMAX must be an integer from 1 "
                  "to 2147483647, got '+3'");
    CHECK(log_lines("---- phi ladder ----").empty());
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
  SECTION("REFIT_COMPACT_SEED 3000000000") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("SIGNED");
    c.add("REFIT_COMPACT_SEED", std::string("3000000000"));
    const auto [what, logged] = run_compact(c);
    INFO(logged);
    CHECK(what == "");
    CHECK(fs::exists("pot.tadah"));
  }
  SECTION("REFIT_COMPACT_NMAX 536870912: a sample grid above INT_MAX") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("SIGNED");
    c.remove("REFIT_COMPACT_NMAX");
    c.add("REFIT_COMPACT_NMAX", std::string("536870912"));
    const auto [what, logged] = run_compact(c);
    CHECK(what == "tadah refit: REFIT_COMPACT_NMAX 536870912 needs a curve "
                  "sample grid of 4 x 536870912 = 2147483648 points per "
                  "function (REFIT_COMPACT_SAMPLES is raised to at least 4 x "
                  "REFIT_COMPACT_NMAX), above 2147483647; lower "
                  "REFIT_COMPACT_NMAX");
    CHECK(log_lines("---- phi ladder ----").empty());
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
}

namespace {
// The mode rule's refusal of `key` under REFIT_MODE compact (MLIP's
// check_refit_mode_keys); `defaulted`: no REFIT_MODE key.
std::string dense_key_refused(const std::string &key, bool defaulted) {
  return "tadah refit: " + key + " has no effect under REFIT_MODE compact" +
         (defaulted ? " (the default)" : "") +
         ": only REFIT_MODE dense reads it; remove " + key +
         ", or set REFIT_MODE dense";
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a key that only REFIT_MODE dense reads "
                 "is refused before any fitting",
                 "[compact][refit][mode_keys]") {
  /* Pins: the HPO refit engine (the one tadah runs) refuses, in validate()
   * and in run() before any ladder runs, a key of the dense fit under
   * REFIT_MODE compact, written or the default: REFIT_LEVEL, N2B,
   * NRHO_BASIS, NF_BASIS, REFIT_MAX_ESCALATIONS, REFIT_PLACEMENT and
   * REFIT_CONSTRAIN_F0 ("tadah refit: N2B has no effect under REFIT_MODE
   * compact: only REFIT_MODE dense reads it; remove N2B, or set REFIT_MODE
   * dense"), and writes nothing.
   * Before the fix: the compact fit never read them and ran to the end
   * (no exception, pot.tadah written; with N2B 50, REFIT_LEVEL accurate
   * and REFIT_MAX_ESCALATIONS 9 the potential was byte-identical to the
   * one without them). */
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_LEVEL", "accurate"},
           {"N2B", "50"},
           {"NRHO_BASIS", "40"},
           {"NF_BASIS", "30"},
           {"REFIT_MAX_ESCALATIONS", "9"},
           {"REFIT_PLACEMENT", "uniform"},
           {"REFIT_CONSTRAIN_F0", "false"}}) {
    for (const bool defaulted : {false, true}) {
      INFO(key << " " << value << (defaulted ? " (default mode)" : ""));
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("SIGNED");
      if (defaulted) c.remove("REFIT_MODE");
      c.add(key, value);
      tadah::mlip::Task task("refit", c);
      CHECK_THROWS_WITH(tadah::hpo::engines::HPORefitEngine().validate(task),
                        dense_key_refused(key, defaulted));
      const auto [what, logged] = run_compact(c);
      CHECK(what == dense_key_refused(key, defaulted));
      CHECK(log_lines("---- phi ladder ----").empty());
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a top-level OPTIMIZER block is refused "
                 "with RefitParser's migration text before the table is read",
                 "[compact][refit][mode_keys]") {
  /* Pins: the HPO refit engine refuses a top-level OPTIMIZER block (the
   * grammar before the stage blocks) under REFIT_MODE compact, written or
   * the default, in validate() and in run() before it reads the EAM file
   * (absent here), with the text RefitParser::parse gives the same
   * configuration ("tadah refit: a top-level OPTIMIZER block is no longer
   * supported. ..."), and writes nothing.
   * Before the fix: validate() accepted the block (the mode rule took it
   * for a key of compact mode) and run() failed reading the absent EAM
   * file ("EAMSetflIO: setfl file does not exist: absent.eam.alloy"):
   * only RefitParser::parse refused the block, after the table was read. */
  for (const bool defaulted : {false, true}) {
    INFO((defaulted ? "REFIT_MODE compact (the default)"
                    : "REFIT_MODE compact"));
    ScopedTempDir tmp;
    Context c = compact_context("SIGNED");
    c.remove("EAM");
    c.add("EAM", std::string("absent.eam.alloy"));
    if (defaulted) c.remove("REFIT_MODE");
    add_block(c, "OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL");
    std::string parser_text;
    try {
      (void)tadah::hpo::refit::RefitParser::parse(c);
    } catch (const std::exception &e) {
      parser_text = e.what();
    }
    REQUIRE(parser_text.rfind("tadah refit: a top-level OPTIMIZER block is "
                              "no longer supported.",
                              0) == 0);
    tadah::mlip::Task task("refit", c);
    CHECK_THROWS_WITH(tadah::hpo::engines::HPORefitEngine().validate(task),
                      parser_text);
    const auto [what, logged] = run_compact(c);
    CHECK(what == parser_text);
    CHECK_FALSE(fs::exists("refit_opt.log"));
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: REFIT_CURVE_WEIGHTING is refused under "
                 "REFIT_OBJECTIVE energy and read under curve",
                 "[compact][refit][mode_keys]") {
  /* Pins: under REFIT_OBJECTIVE energy, written or the default,
   * REFIT_CURVE_WEIGHTING (true or false) is refused before any fitting
   * ("tadah refit: REFIT_CURVE_WEIGHTING has no effect under
   * REFIT_OBJECTIVE energy (the default): only REFIT_OBJECTIVE curve reads
   * it; remove REFIT_CURVE_WEIGHTING, or set REFIT_OBJECTIVE curve"); under
   * curve, REFIT_CURVE_WEIGHTING false runs with uniform weights
   * (refit_opt.log: "curve weighting: OFF (uniform)").
   * Before the fix: under energy the key was ignored and the fit ran to the
   * end (no exception), although the help said it was rejected there; the
   * curve case passes as well (a pin). */
  const std::string head = "tadah refit: REFIT_CURVE_WEIGHTING has no effect "
                           "under REFIT_OBJECTIVE energy";
  const std::string tail = ": only REFIT_OBJECTIVE curve reads it; remove "
                           "REFIT_CURVE_WEIGHTING, or set REFIT_OBJECTIVE "
                           "curve";
  for (const char *value : {"false", "true"}) {
    for (const bool explicit_energy : {false, true}) {
      INFO("REFIT_CURVE_WEIGHTING " << value
                                    << (explicit_energy ? " (REFIT_OBJECTIVE "
                                                          "energy)"
                                                        : ""));
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("SIGNED");
      c.add("REFIT_CURVE_WEIGHTING", std::string(value));
      if (explicit_energy) c.add("REFIT_OBJECTIVE", std::string("energy"));
      const auto [what, logged] = run_compact(c);
      CHECK(what == head + (explicit_energy ? "" : " (the default)") + tail);
      CHECK(log_lines("---- phi ladder ----").empty());
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
  }
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  Context c = compact_context("SIGNED");
  c.add("REFIT_OBJECTIVE", std::string("curve"));
  c.add("REFIT_CURVE_WEIGHTING", std::string("false"));
  const auto [what, logged] = run_compact(c);
  INFO(logged);
  CHECK(what == "");
  CHECK(log_lines("curve weighting: OFF (uniform)").size() == 1);
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: an EXPLORE UNIFORM without INIT points "
                 "is a warning that says what gives a global search",
                 "[compact][refit][stages]") {
  /* Pins: EXPLORE with ALGO UNIFORM and no INIT block is the WARNING
   * "tadah refit: EXPLORE stage 0 is UNIFORM (no global search) and INIT is
   * empty, so no global structure search runs; a global EXPLORE ALGO, such
   * as GN_MLSL_LDS, or INIT points give one."
   * Before the fix: "tadah refit: EXPLORE rung 0 is UNIFORM (no global
   * search) and INIT is empty — the global structure search is disabled.
   * Set a global EXPLORE ALGO (e.g. GN_MLSL_LDS) or provide INIT guesses
   * for robustness.": the check fails. */
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  Context c = compact_context("SIGNED");
  c.remove("EXPLORE");
  add_block(c, "EXPLORE", "OPTIMIZER ALGO UNIFORM ENDOPTIMIZER");
  const auto [what, logged] = run_compact(c);
  INFO(logged);
  CHECK(what == "");
  CHECK(logged.find("tadah refit: EXPLORE stage 0 is UNIFORM (no global "
                    "search) and INIT is empty, so no global structure "
                    "search runs; a global EXPLORE ALGO, such as GN_MLSL_LDS, "
                    "or INIT points give one.") != std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: above tolerance, the log names the "
                 "compact fit's keys, not the dense fit's",
                 "[compact][refit][mode_keys]") {
  /* Pins: a compact fit whose crystal self-test stays above the tolerances
   * (REFIT_TOL_AMBIENT and REFIT_TOL_COMPRESSED 1e-6 meV/atom) ends with
   * the WARNING "still above tolerance — raise REFIT_COMPACT_NMAX or
   * EXPLORE MAX_ADD (a larger basis), or relax REFIT_TOL_* if this regime
   * is irrelevant for your use.", and its window report is "tadah
   * refit: fit windows (reachability-driven):" with "(control:
   * REFIT_REACH_SMIN, REFIT_REACH_MARGIN)"; the log names none of the dense
   * fit's keys or its uniform tail basis (REFIT_MAX_ESCALATIONS,
   * N2B/NRHO_BASIS/NF_BASIS, REFIT_PLACEMENT, "small uniform basis").
   * Before the fix: the WARNING was the dense fit's ("still above
   * tolerance after escalation — raise REFIT_MAX_ESCALATIONS, pin a larger
   * N2B/NRHO_BASIS/NF_BASIS, ...") and the window report said "windowed
   * basis placement", "the unreachable tail keeps a small uniform basis"
   * and "(control: REFIT_PLACEMENT window|uniform, ...)", although the
   * compact fit reads none of those keys. */
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  Context c = compact_context("SIGNED");
  c.add("REFIT_TOL_AMBIENT", std::string("1e-6"));
  c.add("REFIT_TOL_COMPRESSED", std::string("1e-6"));
  const auto [what, logged] = run_compact(c);
  INFO(logged);
  REQUIRE(what == "");
  CHECK(logged.find("still above tolerance — raise REFIT_COMPACT_NMAX or "
                    "EXPLORE MAX_ADD (a larger basis), or relax REFIT_TOL_* "
                    "if this regime is irrelevant for your use.") !=
        std::string::npos);
  CHECK(logged.find("tadah refit: fit windows (reachability-driven):") !=
        std::string::npos);
  CHECK(logged.find("(control: REFIT_REACH_SMIN, REFIT_REACH_MARGIN)") !=
        std::string::npos);
  for (const char *dense : {"REFIT_MAX_ESCALATIONS", "N2B/NRHO_BASIS/NF_BASIS",
                            "REFIT_PLACEMENT", "small uniform basis",
                            "after escalation"}) {
    INFO(dense);
    CHECK(logged.find(dense) == std::string::npos);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: refit_range_report.dat's F row "
                 "describes the compact fit, not the dense fit's tail basis",
                 "[compact][refit][mode_keys]") {
  /* Pins: after a compact fit the F row of refit_range_report.dat has the
   * window of the log's "embedding fit window rho <= X", [0, rho_reach],
   * and the note "above fit_window_hi the setfl carries the ORIGINAL table
   * (offset-matched); control: REFIT_REACH_SMIN, REFIT_REACH_MARGIN" (the
   * compact fit joins refit.eam to the table there, EAMRefit::set_rho_join),
   * and the report names no "small uniform basis".
   * Before the fix: the F note was the dense fit's, "tail keeps a small
   * uniform basis; control: REFIT_REACH_MARGIN", a basis the compact fit
   * does not have. */
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  const auto [what, logged] = run_compact(compact_context("SIGNED"));
  INFO(logged);
  REQUIRE(what == "");
  // the log's window, printed to 4 decimals
  const std::string mark = "embedding fit window rho <= ";
  const auto at = logged.find(mark);
  REQUIRE(at != std::string::npos);
  const double rho_reach = std::stod(logged.substr(at + mark.size()));
  const std::string report = slurp("refit_range_report.dat");
  INFO(report);
  std::istringstream in(report);
  std::string F_row;
  for (std::string line; std::getline(in, line);)
    if (line.rfind("F ", 0) == 0) F_row = line;
  REQUIRE_FALSE(F_row.empty());
  std::istringstream ls(F_row.substr(1));
  double full_lo = -1, full_hi = -1, win_lo = -1, win_hi = -1;
  ls >> full_lo >> full_hi >> win_lo >> win_hi;
  CHECK(full_lo == 0.0);
  CHECK(win_lo == 0.0);
  CHECK(win_hi == Approx(rho_reach).margin(1e-4));
  CHECK(win_hi < full_hi);
  const auto hash = F_row.find("# ");
  REQUIRE(hash != std::string::npos);
  CHECK(F_row.substr(hash + 2) ==
        "above fit_window_hi the setfl carries the ORIGINAL table "
        "(offset-matched); control: REFIT_REACH_SMIN, REFIT_REACH_MARGIN");
  CHECK(report.find("small uniform basis") == std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the HPO engine has the compact fitter "
                 "and accepts the ends of the compact numbers' ranges",
                 "[compact][refit][mode_keys][strict_numbers]") {
  /* Pins: HPORefitEngine::validate() accepts REFIT_MODE compact, written or
   * the default (the engine has the compact fitter), and the ends of the
   * compact keys' ranges (REFIT_COMPACT_SEED 0, 3000000000 and 4294967295,
   * REFIT_COMPACT_JITTERS 0 and 2147483646, REFIT_COMPACT_NMAX 536870911,
   * REFIT_CURVE_DERIV_WEIGHT 0, REFIT_CURVE_RELAX 1, these two under
   * REFIT_OBJECTIVE curve, since energy refuses them). (MLIP's
   * RefitEngine, which has no compact fitter, refuses REFIT_MODE compact
   * once these numbers pass.)
   * Before the fix: passes as well (a pin: MLIP's test checked these ends
   * through its own engine, in a dense configuration). Until the curve
   * knobs were refused under REFIT_OBJECTIVE energy the two were checked
   * under energy, the configuration's default objective. */
  ScopedTempDir tmp;
  for (const bool defaulted : {false, true}) {
    Context c = compact_context("SIGNED");
    if (defaulted) c.remove("REFIT_MODE");
    tadah::mlip::Task task("refit", c);
    INFO((defaulted ? "default mode" : "REFIT_MODE compact"));
    CHECK_NOTHROW(tadah::hpo::engines::HPORefitEngine().validate(task));
  }
  for (const auto &[key, tok] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_COMPACT_SEED", "0"},
           {"REFIT_COMPACT_SEED", "3000000000"},
           {"REFIT_COMPACT_SEED", "4294967295"},
           {"REFIT_COMPACT_JITTERS", "0"},
           {"REFIT_COMPACT_JITTERS", "2147483646"},
           {"REFIT_COMPACT_NMAX", "536870911"},
           {"REFIT_CURVE_DERIV_WEIGHT", "0"},
           {"REFIT_CURVE_RELAX", "1"}}) {
    INFO(key << " " << tok);
    Context c = compact_context("SIGNED");
    if (key.rfind("REFIT_CURVE_", 0) == 0)
      c.add("REFIT_OBJECTIVE", std::string("curve"));
    c.remove(key);
    c.add(key, tok);
    tadah::mlip::Task task("refit", c);
    CHECK_NOTHROW(tadah::hpo::engines::HPORefitEngine().validate(task));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: REFIT_CURVE_RELAX and "
                 "REFIT_CURVE_DERIV_WEIGHT are refused under REFIT_OBJECTIVE "
                 "energy, REFIT_CURVE_RELAX under REFIT_CURVE_WEIGHTING false",
                 "[compact][refit][mode_keys]") {
  /* Pins: through the HPO refit engine, REFIT_CURVE_RELAX and
   * REFIT_CURVE_DERIV_WEIGHT under REFIT_OBJECTIVE energy, written or the
   * default, are refused in validate() and in run() before any ladder runs
   * ("tadah refit: REFIT_CURVE_RELAX has no effect under REFIT_OBJECTIVE
   * energy (the default): only REFIT_OBJECTIVE curve reads it; remove
   * REFIT_CURVE_RELAX, or set REFIT_OBJECTIVE curve"), and nothing is
   * written; under curve with REFIT_CURVE_WEIGHTING false REFIT_CURVE_RELAX
   * is refused ("tadah refit: REFIT_CURVE_RELAX has no effect under
   * REFIT_CURVE_WEIGHTING false: only REFIT_CURVE_WEIGHTING true (the
   * default) reads it; remove REFIT_CURVE_RELAX, or set
   * REFIT_CURVE_WEIGHTING true"); under curve with the weighting on, the
   * default, REFIT_CURVE_RELAX 0.1 is the weighting's floor (refit_opt.log:
   * "curve weighting: ON (pressure-relaxed) ... relax_floor=0.100").
   * Before the fix: under energy the two were accepted with no effect and
   * the fit ran to the end (no exception, pot.tadah written), and so did
   * REFIT_CURVE_RELAX under REFIT_CURVE_WEIGHTING false; the curve run
   * passes as well (a pin). */
  const auto under_energy = [](const std::string &key, bool defaulted) {
    return "tadah refit: " + key + " has no effect under REFIT_OBJECTIVE "
           "energy" + std::string(defaulted ? " (the default)" : "") +
           ": only REFIT_OBJECTIVE curve reads it; remove " + key +
           ", or set REFIT_OBJECTIVE curve";
  };
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_CURVE_RELAX", "0.1"}, {"REFIT_CURVE_DERIV_WEIGHT", "0.1"}}) {
    for (const bool defaulted : {true, false}) {
      INFO(key << " " << value
               << (defaulted ? "" : " (REFIT_OBJECTIVE energy)"));
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("SIGNED");
      c.add(key, value);
      if (!defaulted) c.add("REFIT_OBJECTIVE", std::string("energy"));
      tadah::mlip::Task task("refit", c);
      CHECK_THROWS_WITH(tadah::hpo::engines::HPORefitEngine().validate(task),
                        under_energy(key, defaulted));
      const auto [what, logged] = run_compact(c);
      CHECK(what == under_energy(key, defaulted));
      CHECK(log_lines("---- phi ladder ----").empty());
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
  }
  {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("SIGNED");
    c.add("REFIT_OBJECTIVE", std::string("curve"));
    c.add("REFIT_CURVE_WEIGHTING", std::string("false"));
    c.add("REFIT_CURVE_RELAX", std::string("0.1"));
    const auto [what, logged] = run_compact(c);
    CHECK(what == "tadah refit: REFIT_CURVE_RELAX has no effect under "
                  "REFIT_CURVE_WEIGHTING false: only REFIT_CURVE_WEIGHTING "
                  "true (the default) reads it; remove REFIT_CURVE_RELAX, or "
                  "set REFIT_CURVE_WEIGHTING true");
    CHECK(log_lines("---- phi ladder ----").empty());
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  Context c = compact_context("SIGNED");
  c.add("REFIT_OBJECTIVE", std::string("curve"));
  c.add("REFIT_CURVE_RELAX", std::string("0.1"));
  const auto [what, logged] = run_compact(c);
  INFO(logged);
  CHECK(what == "");
  const auto on = log_lines("curve weighting: ON (pressure-relaxed)");
  REQUIRE(on.size() == 1);
  CHECK(on[0].find("relax_floor=0.100") != std::string::npos);
}

namespace {
// The row of `function` in refit_range_report.dat: full_lo, full_hi,
// fit_window_lo, fit_window_hi.
std::array<double, 4> report_row(const std::string &function) {
  std::istringstream in(slurp("refit_range_report.dat"));
  for (std::string line; std::getline(in, line);) {
    std::istringstream ls(line);
    std::string name;
    if (!(ls >> name) || name != function) continue;
    std::array<double, 4> row{};
    ls >> row[0] >> row[1] >> row[2] >> row[3];
    return row;
  }
  FAIL("refit_range_report.dat has no " << function << " row");
  return {};
}

// The number of rows of refit_curve_<fn>.dat whose in_window flag is not
// that of x in [lo, hi] (rows within 1e-9 of an edge are not judged).
int wrong_window_flags(const std::string &f, double lo, double hi) {
  std::istringstream in(slurp(f));
  int rows = 0, wrong = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::vector<double> cols;
    for (double v; ls >> v;) cols.push_back(v);
    if (cols.size() != 9) return -1;
    ++rows;
    const double x = cols[0];
    if (std::fabs(x - lo) < 1e-9 || std::fabs(x - hi) < 1e-9) continue;
    if ((cols[8] != 0.0) != (x >= lo && x <= hi)) ++wrong;
  }
  return rows > 100 ? wrong : -1;
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: refit_range_report.dat and the curve "
                 "files give the window the compact fit used",
                 "[compact][refit]") {
  /* Pins: after a compact fit the phi and rho rows of
   * refit_range_report.dat have fit_window_lo = the lower edge of the
   * compact fit's radial window, max(0.8 r_reach, REFIT_R_LO_FRAC rcut,
   * dr), as refit_opt.log's "windows: r in [<lo>, <rcut>]" prints it (to
   * 4 decimals; the synthetic fcc table: 0.8 x 0.75 x 3.6/sqrt 2 = 1.527 A),
   * equal to the row's full_lo (the fit's r_lo), and fit_window_hi = rcut;
   * refit_curve_phi.dat and refit_curve_rho.dat flag in_window exactly the
   * x in [fit_window_lo, rcut].
   * Before the fix: fit_window_lo was 0.9 r_reach (1.718 A), neither the
   * compact fit's window nor the dense fit's, and the curve files flagged
   * in_window from it (x in [1.527, 1.718) flagged 0). */
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  const auto [what, logged] = run_compact(compact_context("SIGNED"));
  INFO(logged);
  REQUIRE(what == "");
  const auto windows = log_lines("windows: r in [");
  REQUIRE(windows.size() == 1);
  const double window_lo =
      std::stod(windows[0].substr(windows[0].find('[') + 1));
  CHECK(window_lo == Approx(0.8 * 0.75 * 3.6 / std::sqrt(2.0)).margin(1e-4));
  INFO(slurp("refit_range_report.dat"));
  for (const char *fn : {"phi", "rho"}) {
    INFO(fn);
    const auto row = report_row(fn);
    CHECK(row[2] == Approx(window_lo).margin(1e-4));
    CHECK(row[2] == Approx(row[0]));
    CHECK(row[3] == Approx(5.0));
    CHECK(wrong_window_flags(std::string("refit_curve_") + fn + ".dat",
                             window_lo, 5.0) == 0);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the knot-family keys are refused under "
                 "the Blip families",
                 "[compact][refit][mode_keys]") {
  /* Pins: through the HPO refit engine, REFIT_COMPACT_MERGE,
   * REFIT_COMPACT_OPT_EVERY and REFIT_COMPACT_INSERT_TOPK, which the
   * compact fitter applies to the one-sided knot families only, are refused
   * under REFIT_BASIS Blip5 in validate() and in run() before any ladder
   * runs ("tadah refit: REFIT_COMPACT_MERGE has no effect under REFIT_BASIS
   * Blip5: only the knot families (REFIT_BASIS Knot5, the compact default,
   * Knot3 or KnotMix) read it; remove REFIT_COMPACT_MERGE, or set
   * REFIT_BASIS Knot5"), and nothing is written; under REFIT_BASIS Knot5 and
   * without REFIT_BASIS (the compact default, Knot5) validate() accepts
   * each of them. A Blip5 fit without them logs the schedule it applies,
   * "merge_guard=off" (refit_opt.log's schedule line), REFIT_COMPACT_MERGE
   * auto under REFIT_OBJECTIVE energy notwithstanding.
   * Before the fix: under Blip5 they were accepted with no effect and the
   * fit ran to the end (no exception, pot.tadah written; the schedule line
   * printed the values written, opt_every=3 or insert_topk=3, which the
   * Blip fit does not apply), and the schedule line of a Blip5 fit said
   * "merge_guard=on" (auto under energy), a guard the fit never applies to
   * the Blip families; the knot cases pass as well (a pin). */
  const auto refused = [](const std::string &key) {
    return "tadah refit: " + key + " has no effect under REFIT_BASIS Blip5: "
           "only the knot families (REFIT_BASIS Knot5, the compact default, "
           "Knot3 or KnotMix) read it; remove " + key +
           ", or set REFIT_BASIS Knot5";
  };
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_COMPACT_MERGE", "true"},
           {"REFIT_COMPACT_OPT_EVERY", "3"},
           {"REFIT_COMPACT_INSERT_TOPK", "3"}}) {
    INFO(key << " " << value);
    {
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("SIGNED");
      c.add(key, value);
      tadah::mlip::Task task("refit", c);
      CHECK_THROWS_WITH(tadah::hpo::engines::HPORefitEngine().validate(task),
                        refused(key));
      const auto [what, logged] = run_compact(c);
      CHECK(what == refused(key));
      CHECK(log_lines("---- phi ladder ----").empty());
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
    for (const bool defaulted : {false, true}) {
      INFO((defaulted ? "the default basis" : "REFIT_BASIS Knot5"));
      Context c = compact_context("SIGNED");
      c.remove("REFIT_BASIS");
      if (!defaulted) c.add("REFIT_BASIS", std::string("Knot5"));
      c.add(key, value);
      tadah::mlip::Task task("refit", c);
      CHECK_NOTHROW(tadah::hpo::engines::HPORefitEngine().validate(task));
    }
  }
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  const auto [what, logged] = run_compact(compact_context("SIGNED"));
  INFO(logged);
  REQUIRE(what == "");
  const auto schedule = log_lines("schedule: opt_every=");
  REQUIRE(schedule.size() == 1);
  CHECK(schedule[0].find("opt_every=1 ") != std::string::npos);
  CHECK(schedule[0].find("insert_topk=1 ") != std::string::npos);
  CHECK(schedule[0].find("merge_guard=off") != std::string::npos);
}

/* Pins: the curve-mode note of refit_opt.log's header gives the knob of the
 * fit's family for a costly Ceres REFINE at high NMAX: under REFIT_BASIS
 * Blip5 (a Blip family, which refuses REFIT_COMPACT_OPT_EVERY) it names
 * REFINE's MAXEVAL and REFIT_COMPACT_NMAX and not REFIT_COMPACT_OPT_EVERY;
 * under REFIT_BASIS Knot5 it keeps "use REFIT_COMPACT_OPT_EVERY".
 * Before the fix: every curve fit's note said "cap its MAXEVAL / use
 * REFIT_COMPACT_OPT_EVERY (or the gentle SBPLX-only arm) at high NMAX",
 * under Blip and Blip5 too, where adding REFIT_COMPACT_OPT_EVERY is refused
 * ("REFIT_COMPACT_OPT_EVERY has no effect under REFIT_BASIS Blip5"). */
TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact, curve mode: refit_opt.log advises the "
                 "keys the basis family reads",
                 "[compact][refit][curve_advice]") {
  for (const char *basis : {"Blip5", "Knot5"}) {
    INFO("REFIT_BASIS " << basis);
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("SIGNED");
    c.remove("REFIT_BASIS");
    c.add("REFIT_BASIS", std::string(basis));
    c.add("REFIT_OBJECTIVE", std::string("curve"));
    const auto [what, logged] = run_compact(c);
    INFO(logged);
    REQUIRE(what.empty());
    const auto note = log_lines("# REFIT_OBJECTIVE=curve:");
    REQUIRE(note.size() == 1u);
    if (std::string(basis) == "Knot5") {
      CHECK(note[0].find("use REFIT_COMPACT_OPT_EVERY") != std::string::npos);
    } else {
      CHECK(note[0].find("REFIT_COMPACT_OPT_EVERY") == std::string::npos);
      CHECK(note[0].find("lower REFIT_COMPACT_NMAX") != std::string::npos);
    }
  }
}

namespace {

/* The compact fitter's hook, as RefitEngine::run() calls it (protected):
 * the seam by which a test hands it a configuration no `tadah refit` run
 * can reach. */
struct CompactFitHook : tadah::hpo::engines::HPORefitEngine {
  using tadah::hpo::engines::HPORefitEngine::compact_fit;
};

}  // namespace

/* Pins: HPORefitEngine::compact_fit() given a reachability analysis whose
 * r_reach is less than the step dr of the table's radial grid --
 * REFIT_REACH_SMIN 0 on the synthetic fcc table (a0 = 3.6 A, dr = 0.0125313
 * A), so r_reach = 0 x the nearest-neighbour distance -- throws
 * std::logic_error "tadah refit[compact]: internal error: the reachability
 * analysis gave r_reach = 0 A (REFIT_REACH_SMIN 0 x the nearest-neighbour
 * distance of the fcc lattice of a0 = 3.6 A), less than the step of the
 * table's radial grid, dr = 0.0125313 A, although the crystal gate of
 * tadah refit (eam_lattice::require_crystal_on_grid) refuses every such
 * table; this is a bug, please report it" and fits nothing (no EAMRefit).
 * No `tadah refit` run reaches it: run()'s crystal gate refuses a table
 * whose crystal's nearest neighbours at REFIT_REACH_SMIN are less than dr
 * apart (the next test), so only a caller of the hook can hand it one
 * (an internal error, not a fallback).
 * Before the fix: the internal error said "the reachable pair-distance
 * window is empty (r_reach = 0.0000 A, from REFIT_REACH_SMIN 0.0000 and the
 * fcc lattice of a0 = 3.6000 A), although the crystal gate (a known
 * lattice, a0 > 0) and the range of REFIT_REACH_SMIN (0.1 to 1) make it
 * positive for every configuration; ..." (a gate that a subnormal a0
 * passes, the next test): the text check fails. */
TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: an r_reach below the table's dr is an "
                 "internal error, not a fallback to the dense fit",
                 "[compact][refit][reachability]") {
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  const Context c = compact_context("SIGNED");
  tadah::models::EAMSetflIO eam;
  eam.read_setfl("synth.eam.alloy");
  eam.build_splines();
  tadah::models::EAMRefitConfig cfg;
  cfg.basis = "Blip5";
  cfg.reach_smin = 0.0;
  std::unique_ptr<tadah::models::EAMRefit> refit;
  CompactFitHook::SelfTest st;
  CompactFitHook hook;
  std::string what;
  bool logic_error = false, returned = false;
  try {
    (void)hook.compact_fit(c, eam, cfg, refit, st);
    returned = true;
  } catch (const std::logic_error &e) {
    logic_error = true;
    what = e.what();
  } catch (const std::exception &e) {
    what = e.what();
  }
  INFO("what: " << what);
  CHECK_FALSE(returned);
  CHECK(logic_error);
  CHECK(what ==
        "tadah refit[compact]: internal error: the reachability analysis "
        "gave r_reach = 0 A (REFIT_REACH_SMIN 0 x the nearest-neighbour "
        "distance of the fcc lattice of a0 = 3.6 A), less than the step of "
        "the table's radial grid, dr = 0.0125313 A, although the crystal "
        "gate of tadah refit (eam_lattice::require_crystal_on_grid) refuses "
        "every such table; this is a bug, please report it");
  CHECK(refit == nullptr);
}

/* Pins: a compact `tadah refit` (REFIT_REACH_SMIN 0.1) of the synthetic fcc
 * table whose element line gives a0 = 1e-323 (a subnormal, which the setfl
 * reader reads) is refused by RefitEngine::run()'s crystal gate, a
 * std::runtime_error and not the internal error: "tadah refit: the lattice
 * constant of the EAM header, a0 = 1e-323 A, puts the nearest neighbours of
 * the fcc crystal 0 A apart at the strongest compression the refit tests
 * (a/a0 = REFIT_REACH_SMIN 0.1), less than the step of the table's radial
 * grid, dr = 0.0125313 A: the table describes no pair distance of that
 * crystal. Edit the element line of the setfl file (`<Z> <mass> <a0>
 * <lattice>`): set the equilibrium lattice constant, in Angstrom."; it
 * writes no pot.tadah.
 * Before the fix: the gate asked for a known lattice and a0 > 0 only, and
 * compact_fit's r_reach = 0.1 x a0 / sqrt(2) underflowed to 0:
 * std::logic_error "tadah refit[compact]: internal error: the reachable
 * pair-distance window is empty (r_reach = 0.0000 A, from REFIT_REACH_SMIN
 * 0.1000 and the fcc lattice of a0 = 0.0000 A), ...; this is a bug, please
 * report it". */
TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a subnormal lattice constant is "
                 "refused by the crystal gate, not the internal error",
                 "[compact][refit][reachability]") {
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  {
    std::string body = slurp("synth.eam.alloy");
    const std::string line = "\n29 63.55 3.6 fcc\n";
    const auto at = body.find(line);
    REQUIRE(at != std::string::npos);
    body.replace(at, line.size(), "\n29 63.55 1e-323 fcc\n");
    std::ofstream("synth.eam.alloy") << body;
  }
  Context c = compact_context("SIGNED");
  c.add("REFIT_REACH_SMIN", 0.1);
  tadah::mlip::Task task("refit", c);
  std::string what;
  bool logic_error = false, runtime_error = false;
  try {
    tadah::hpo::engines::HPORefitEngine().run(task);
  } catch (const std::logic_error &e) {
    logic_error = true;
    what = e.what();
  } catch (const std::runtime_error &e) {
    runtime_error = true;
    what = e.what();
  }
  INFO("what: " << what);
  CHECK_FALSE(logic_error);
  CHECK(runtime_error);
  CHECK(what ==
        "tadah refit: the lattice constant of the EAM header, a0 = 1e-323 A, "
        "puts the nearest neighbours of the fcc crystal 0 A apart at the "
        "strongest compression the refit tests (a/a0 = REFIT_REACH_SMIN "
        "0.1), less than the step of the table's radial grid, dr = "
        "0.0125313 A: the table describes no pair distance of that crystal. "
        "Edit the element line of the setfl file (`<Z> <mass> <a0> "
        "<lattice>`): set the equilibrium lattice constant, in Angstrom.");
  CHECK_FALSE(fs::exists("pot.tadah"));
}

/* Pins: a compact `tadah refit` of the synthetic fcc table (rcut 5 A)
 * whose element line gives a0 = 361 (copper's lattice constant in
 * picometres) is refused by RefitEngine::run()'s crystal gate, before any
 * fitting, with the std::runtime_error "tadah refit: the lattice constant
 * of the EAM header, a0 = 361 A, puts the nearest neighbours of the fcc
 * crystal 255.26554800834364 A apart, at or beyond the table's cutoff,
 * rcut = 5 A, and LAMMPS (pair_style eam/alloy) counts a pair only below
 * rcut: the crystal has no interaction in LAMMPS. Edit the element line of
 * the setfl file (`<Z> <mass> <a0> <lattice>`): set the equilibrium
 * lattice constant, in Angstrom (a value in picometres is 100 times the
 * value in Angstrom)."; no ladder runs and no pot.tadah is written.
 * Before the fix: the refusal said "...: the crystal has no pair within
 * the cutoff." (tadah itself counts a pair at exactly rcut), so the text
 * check fails. Before the crystal gate took the cutoff, the compact fit
 * started its windows at 0.8 x r_reach = 153 A, above rcut, ran its phi
 * ladder and failed inside the fitter: "compact_fitter
 * clamp_bump_support_params: radial bump 1 (mu 153.159329) has width eta
 * -0.006749 <= 0; every radial bump width must be > 0". */
TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a lattice constant whose nearest "
                 "neighbours are beyond the cutoff is refused by the crystal "
                 "gate",
                 "[compact][refit][reachability]") {
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  {
    std::string body = slurp("synth.eam.alloy");
    const std::string line = "\n29 63.55 3.6 fcc\n";
    const auto at = body.find(line);
    REQUIRE(at != std::string::npos);
    body.replace(at, line.size(), "\n29 63.55 361 fcc\n");
    std::ofstream("synth.eam.alloy") << body;
  }
  const auto [what, logged] = run_compact(compact_context("SIGNED"));
  INFO(logged);
  CHECK(what ==
        "tadah refit: the lattice constant of the EAM header, a0 = 361 A, "
        "puts the nearest neighbours of the fcc crystal 255.26554800834364 A "
        "apart, at or beyond the table's cutoff, rcut = 5 A, and LAMMPS "
        "(pair_style eam/alloy) counts a pair only below rcut: the crystal "
        "has no interaction in LAMMPS. Edit the element line of the setfl "
        "file "
        "(`<Z> <mass> <a0> <lattice>`): set the equilibrium lattice constant, "
        "in Angstrom (a value in picometres is 100 times the value in "
        "Angstrom).");
  CHECK(log_lines("---- phi ladder ----").empty());
  CHECK_FALSE(fs::exists("pot.tadah"));
}

#endif  // TADAH_USE_CERES

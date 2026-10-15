// `tadah refit` REFIT_MODE compact with a DESCRIPTOR block (DM_Embed): the
// component compact fit of the HPO refit engine (compact_embed_fit), on the
// synthesised single-element setfl of the compact refit tests.
#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include "compact_refit_harness.h"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/engines/embed_refit.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <tadah/models/eam/eam_refit.h>
#include <tadah/models/eam/eam_setfl_io.h>

#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <unistd.h>

using tadah::core::Context;
using namespace compact_refit_test;

namespace {
// A component compact refit of synth.eam.alloy: density WendlandC4 in the
// form `form` with amplitudes to fit, Gaussian embedding, Cut_Poly;
// `extra` adds key lines.
Context component_context(const std::string &form,
                          const std::string &extra = "") {
  std::istringstream in(
      "EAM synth.eam.alloy\nREFIT_MODE compact\nREFIT_PAIR_BASIS Blip5\n"
      "REFIT_TOL_AMBIENT 5\nREFIT_TOL_COMPRESSED 20\n" + extra +
      "DESCRIPTOR cu DM_Embed\nPAIRS Cu Cu\nDENSITY\nBASIS WendlandC4\n"
      "FORM " + form + "\nCENTRES 2 3\nWIDTHS 1.3 1.1\nAMPLITUDES AUTO\n"
      "CUTOFF\nTYPE Cut_Poly\nRADIUS AUTO\nENDCUTOFF\nENDDENSITY\n"
      "EMBEDDING\nBASIS Gaussian\nCENTRES AUTO 12\nWIDTHS AUTO\n"
      "REFERENCE ZERO\nENDEMBEDDING\nENDDESCRIPTOR\n");
  return tadah::core::CoreUserConfigParser().parse(in);
}

// The refit engine with its compact hook exposed.
struct Engine : tadah::hpo::engines::HPORefitEngine {
  using HPORefitEngine::compact_fit;
};

// stderr of `f`, through the file descriptor (Ceres logs through glog).
template <class F> std::string stderr_of(F &&f) {
  std::fflush(stderr);
  const std::string path = "stderr_capture.txt";
  const int saved = ::dup(2);
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  REQUIRE(fd >= 0);
  ::dup2(fd, 2);
  ::close(fd);
  try {
    f();
  } catch (...) {
    std::fflush(stderr);
    ::dup2(saved, 2);
    ::close(saved);
    throw;
  }
  std::fflush(stderr);
  ::dup2(saved, 2);
  ::close(saved);
  return slurp(path);
}
}  // namespace

/* Pins: a component compact refit in FORM NONNEG of a table whose density
 * is negative inside the fit window is refused with EAMRefit's own message
 * (the component's form cannot follow the table, naming r), as the legacy
 * compact and every dense refit refuse it. Before the fix the Ceres
 * objective caught the refusal at every point, so the solve failed at its
 * start and the refit reported "DM_Embed compact refit: Ceres
 * GradientProblemSolver Report: Iterations: 0, ... Termination: FAILURE". */
TEST_CASE_METHOD(PTFixture,
                 "Component compact refit reports a precondition refusal "
                 "with its own message",
                 "[unified-dm][compact][refit]") {
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy", [](double r) {
    const double x = 1.0 - r / 5.0;
    return x * x - 0.02;
  });
  const auto [what, logged] = run_compact(component_context("NONNEG"));
  INFO(logged);
  CHECK(what.find("the density component CU's FORM NONNEG cannot fit this "
                  "table") != std::string::npos);
  CHECK(what.find("Ceres") == std::string::npos);
  CHECK_FALSE(fs::exists("pot.tadah"));
}

/* Pins: the compact hook's internal invariant r_reach >= dr (the crystal
 * gate of run() guarantees it) is checked for a component fit too: a table
 * whose radial step dr = 1.25 A exceeds r_reach (REFIT_REACH_SMIN 0.75 x
 * the nearest-neighbour distance 1.556 A of fcc a0 = 2.2 A), which reaches
 * the hook only when called directly, is std::logic_error naming r_reach.
 * Before the fix the component branch returned before the check and fitted
 * the table. */
TEST_CASE_METHOD(PTFixture,
                 "Component compact fit checks the reachability invariant",
                 "[unified-dm][compact][refit]") {
  ScopedTempDir tmp;
  {  // the synthetic functions on nr = 5 points to rcut 5 (dr = 1.25 A)
    const int nr = 5, nrho = 400;
    const double rcut = 5.0, dr = rcut / (nr - 1);
    const double rhomax = 60.0, drho = rhomax / (nrho - 1);
    std::ofstream os("coarse.eam.alloy");
    os << "# coarse synthetic EAM\n# comment 2\n# comment 3\n1 Cu\n"
       << nrho << " " << drho << " " << nr << " " << dr << " " << rcut
       << "\n29 63.55 2.2 fcc\n" << std::setprecision(12);
    for (int i = 0; i < nrho; ++i)
      os << (-2.0 * std::sqrt(i * drho) + 0.05 * i * drho) << "\n";
    for (int i = 0; i < nr; ++i) {
      const double x = 1.0 - i * dr / rcut;
      os << x * x << "\n";
    }
    for (int i = 0; i < nr; ++i) {
      const double r = i * dr, x = 1.0 - r / rcut;
      os << 3.0 * x * x * x * r << "\n";
    }
  }
  tadah::models::EAMSetflIO eam;
  eam.read_setfl("coarse.eam.alloy");
  eam.build_splines();
  REQUIRE(eam.dr() == Approx(1.25));
  Context c = component_context("SIGNED");
  tadah::models::EAMRefitConfig cfg;
  tadah::mlip::engines::prepare_embed_refit(c, eam, cfg);
  REQUIRE(cfg.reach_smin * 2.2 / std::sqrt(2.0) < eam.dr());
  std::unique_ptr<tadah::models::EAMRefit> refit;
  tadah::mlip::engines::RefitEngine::SelfTest st;
  Engine engine;
  REQUIRE_THROWS_AS(engine.compact_fit(c, eam, cfg, refit, st),
                    std::logic_error);
  REQUIRE_THROWS_WITH(engine.compact_fit(c, eam, cfg, refit, st),
                      Catch::Contains("r_reach"));
}

/* Pins: the component compact fit logs nothing through Ceres (its progress
 * lines went to stderr whatever VERBOSE was), and when its centre/scale
 * optimisation stops before converging (REFIT_COMPONENT_ITERATIONS 2) it
 * says so in a WARNING naming the stop and the key, and still writes the
 * potential it reached. Before the fix the iteration limit was a fixed 60
 * that no key set, a run that stopped at it was accepted without a word,
 * and Ceres printed one 'callbacks.cc' line per iteration on stderr. */
TEST_CASE_METHOD(PTFixture,
                 "Component compact fit is silent through Ceres and warns "
                 "when it stops before converging",
                 "[unified-dm][compact][refit]") {
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  std::pair<std::string, std::string> result;
  const std::string err = stderr_of([&] {
    result = run_compact(
        component_context("SIGNED", "REFIT_COMPONENT_ITERATIONS 2\n"));
  });
  INFO(result.second);
  INFO(err);
  REQUIRE(result.first.empty());
  CHECK(err.find("callbacks.cc") == std::string::npos);
  CHECK(result.second.find("WARNING") != std::string::npos);
  CHECK(result.second.find("stopped after 2 iterations without converging") !=
        std::string::npos);
  CHECK(result.second.find("REFIT_COMPONENT_ITERATIONS") != std::string::npos);
  CHECK(fs::exists("pot.tadah"));
}

#endif  // TADAH_USE_CERES

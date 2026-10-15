// `tadah refit` compact mode in every RHOFORM, end to end through the HPO
// refit engine (the compact fitter's ladders, FUSE and the EAMRefit
// handoff), on a synthesised single-element setfl with small optimiser
// budgets. The density is fitted in the declared form: SIGNED (the fit
// before RHOFORM), NONNEG (every amplitude >= 0) and MONOTONE (integrated-bump steps,
// every amplitude >= 0).
#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include "compact_refit_harness.h"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/models/descriptors/dm/dm_ream.h>
#include <tadah/models/descriptors/dm/embed_kernel.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using tadah::core::Context;
using tadah::models::RhoForm;
using namespace compact_refit_test;
namespace fs = std::filesystem;

namespace {
// The crystal self-test's max error (meV/atom) from refit_opt.log's
// "    <label>    RMSE=... max=..." line (label ambient | compressed).
double logged_max(const std::string &log, const std::string &label) {
  const auto at = log.find("\n    " + label + " ");
  REQUIRE(at != std::string::npos);
  const auto m = log.find("max=", at);
  REQUIRE(m != std::string::npos);
  return std::stod(log.substr(m + 4));
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact fits, self-tests and writes every "
                 "RHOFORM", "[compact][refit][rhoform]") {
  for (const char *form : {"SIGNED", "NONNEG", "MONOTONE"}) {
    INFO("RHOFORM " << form);
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const auto [what, logged] = run_compact(compact_context(form));
    INFO(logged);
    REQUIRE(what.empty());
    const RhoForm f = tadah::models::parse_rhoform(form);

    // the refit log records the form, and the compact path ran
    const std::string olog = slurp("refit_opt.log");
    CHECK(olog.find("compact mode") != std::string::npos);
    CHECK(olog.find(std::string("density: RHOFORM ") + form + " — ") !=
          std::string::npos);
    // SIGNED names the release that introduced the forms
    if (f == RhoForm::Signed)
      CHECK(olog.find("with signed amplitudes (the density before tadah "
                      "1.6.0)") != std::string::npos);
    CHECK(olog.find("self-test (crystal E-V") != std::string::npos);

    // The joint refinement optimises the radial bumps in their outer support
    // edges, bounded at rcut, so the post-FUSE bump-support projection has
    // nothing to move and the result is the joint optimum in every form: no
    // linear re-solve, no held re-run, and the crystal self-test within its
    // tolerances (5 / 20 meV/atom). With the old post-solve projection on
    // this table SIGNED failed the self-test, and the positive forms needed
    // the held re-run, which traded the density away.
    CHECK(olog.find("back inside rcut") == std::string::npos);
    CHECK(olog.find("moved bumps held") == std::string::npos);
    CHECK(logged_max(olog, "ambient") <= 5.0);
    CHECK(logged_max(olog, "compressed") <= 20.0);

    // the written potential declares the form and loads in it
    Context pot = tadah::core::CoreUserConfigParser().parse("pot.tadah");
    auto specs=tadah::models::embed_specs(pot);REQUIRE(specs.size()==1);
    CHECK(specs[0].form == f);
    std::vector<double> amps=specs[0].density.amplitudes;
    REQUIRE_FALSE(amps.empty());
    if (tadah::models::rhoform_is_positive(f))
      for (double a : amps) REQUIRE(a >= 0.0);
    pot.add("VERBOSE", 0);
    tadah::models::EmbedEvaluator dm(specs[0]);
    REQUIRE(dm.spec().form == f);

    // the density the potential evaluates (Cut_Dummy: f_c = 1 below rcut)
    // follows the table (1 - r/5)^2 on the sampled shells, and carries the
    // form's guarantees on [0, rcut)
    double err = 0.0, prev = dm.density(0.0).value, term_sum = 0.0;
    for (double a : amps) term_sum += std::fabs(a);
    int rises = 0;
    for (int i = 1; i < 5000; ++i) {
      const double r = 5.0 * i / 5000.0;
      const double v = dm.density(r).value;
      if (r >= 2.2) {
        const double x = 1.0 - r / 5.0;
        err = std::max(err, std::fabs(v - x * x));
      }
      if (f != RhoForm::Signed) REQUIRE(v >= 0.0);
      if (v - prev > 8.0 * std::numeric_limits<double>::epsilon() * term_sum)
        ++rises;
      prev = v;
    }
    INFO("max |rho - rho_tab| on [2.2, 5) = " << err << ", rises = " << rises);
    // A coarse fit (small budgets; |rho| <= 0.31 there), in every form.
    CHECK(err < 1e-2);
    if (f == RhoForm::Monotone) CHECK(rises == 0);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the RHOFORM table pre-checks and the "
                 "MONOTONE INIT width rule", "[compact][refit][rhoform]") {
  SECTION("NONNEG refuses a negative table before any fitting") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy", [](double r) {
      const double x = 1.0 - r / 5.0;
      return x * x - 0.02;
    });
    const auto [what, logged] = run_compact(compact_context("NONNEG"));
    CHECK(what.find("RHOFORM NONNEG cannot fit this table") !=
          std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
    CHECK(slurp("refit_opt.log").find("---- phi ladder ----") ==
          std::string::npos);
  }
  SECTION("MONOTONE refuses a rising table before any fitting") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy", [](double r) {
      const double x = 1.0 - r / 5.0, g = (r - 3.5) / 0.2;
      return x * x + 0.05 * std::exp(-g * g);
    });
    const auto [what, logged] = run_compact(compact_context("MONOTONE"));
    CHECK(what.find("RHOFORM MONOTONE cannot fit this table: the tabulated "
                    "density rises") != std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
  SECTION("MONOTONE: a density INIT anchor with eta <= 0 is an error") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    for (const char *eta : {"-0.5", "0"}) {
      INFO("eta " << eta);
      Context c = compact_context("MONOTONE");
      add_block(c, "INIT", std::string("RHO MANUAL 3.0 ") + eta + " PIN");
      const auto [what, logged] = run_compact(c);
      CHECK(what.find("tadah refit INIT [rho]: the basis at mu 3") !=
            std::string::npos);
      CHECK(what.find("RHOFORM MONOTONE fits the density in integrated-bump "
                      "steps, which need eta > 0") != std::string::npos);
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a radial INIT anchor with eta <= 0 is "
                 "an error in every RHOFORM",
                 "[compact][refit][rhoform][bump_width]") {
  // Owner decision (2026-09-29, C8): a radial bump needs a width > 0 in
  // every form. INIT PHI MANUAL 5.0 -1.0 PIN was fitted as its mirror (the
  // bump is even in eta) and the bump-support checks skipped it.
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  for (const char *form : {"SIGNED", "NONNEG"}) {
    for (const char *ch : {"PHI", "RHO"}) {
      // the value as written, and as the message prints it (shortest
      // round trip)
      for (const auto &[eta, shown] :
           {std::pair<const char *, const char *>{"-1.0", "-1"},
            std::pair<const char *, const char *>{"0", "0"}}) {
        INFO(form << " INIT " << ch << " eta " << eta);
        Context c = compact_context(form);
        add_block(c, "INIT", std::string(ch) + " MANUAL 3.0 2.0 " + ch +
                                 " MANUAL 3.5 " + eta + " PIN");
        const auto [what, logged] = run_compact(c);
        INFO("error: " << what);
        const std::string lower = std::string(ch) == "PHI" ? "phi" : "rho";
        CHECK(what.find("tadah refit INIT [" + lower +
                        "]: the basis at mu 3.5 (INIT " + ch +
                        " anchor 2) has eta " + shown + " <= 0") !=
              std::string::npos);
        CHECK(what.find("a radial bump needs a width eta > 0") !=
              std::string::npos);
        CHECK_FALSE(fs::exists("pot.tadah"));
      }
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: RHOFORM x REFIT_BASIS, the knot "
                 "families and the defaulted Knot5",
                 "[compact][refit][rhoform]") {
  /* Pins: under REFIT_MODE compact, written or the default, through the
   * HPO engine (which has the compact fitter):
   *  - RHOFORM MONOTONE with the defaulted basis is refused before any
   *    fitting, naming it as the compact default ("RHOFORM MONOTONE is not
   *    defined for REFIT_BASIS Knot5 (the REFIT_MODE compact default, a
   *    one-sided knot family): use a bump basis (Blip, Blip5) or RHOFORM
   *    NONNEG"), and with a chosen Knot3 as "REFIT_BASIS Knot3 (the
   *    one-sided knot family)"; nothing is written;
   *  - RHOFORM NONNEG with a chosen Knot5 logs the convex-lock HINT ("HINT:
   *    RHOFORM NONNEG with REFIT_BASIS Knot5: a knot density with every
   *    amplitude >= 0 is non-increasing and convex"), not the compact
   *    default's WARNING, and the compact fit writes a NONNEG potential;
   *  - RHOFORM NONNEG with the defaulted Knot5 logs the WARNING naming
   *    REFIT_BASIS Blip5 ("tadah refit: RHOFORM NONNEG with the REFIT_MODE
   *    compact default basis Knot5: ... Set REFIT_BASIS Blip5 (a
   *    non-negative bump density without the convex lock), or use RHOFORM
   *    SIGNED."), not the HINT, and the compact fit writes a NONNEG
   *    potential.
   * Before the fix: passes as well (a pin: these cases ran in MLIP's
   * RefitEngine tests, which has no compact fitter, through the fallback
   * to the dense fit that REFIT_MODE compact's refusal replaced there). */
  SECTION("MONOTONE: the knot families are refused, a defaulted one as the "
          "compact default") {
    for (const bool defaulted : {false, true}) {
      INFO((defaulted ? "default mode" : "REFIT_MODE compact"));
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("MONOTONE");
      c.remove("REFIT_BASIS");
      if (defaulted) c.remove("REFIT_MODE");
      const auto [what, logged] = run_compact(c);
      CHECK(what.find("RHOFORM MONOTONE is not defined for REFIT_BASIS Knot5 "
                      "(the REFIT_MODE compact default, a one-sided knot "
                      "family): use a bump basis (Blip, Blip5) or RHOFORM "
                      "NONNEG") != std::string::npos);
      CHECK_FALSE(fs::exists("pot.tadah"));
      CHECK_FALSE(fs::exists("refit_opt.log"));
    }
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("MONOTONE");
    c.remove("REFIT_BASIS");
    c.add("REFIT_BASIS", std::string("Knot3"));
    const auto [what, logged] = run_compact(c);
    CHECK(what.find("REFIT_BASIS Knot3 (the one-sided knot family)") !=
          std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
  SECTION("NONNEG with a chosen Knot5: the convex-lock HINT") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    Context c = compact_context("NONNEG");
    c.remove("REFIT_BASIS");
    c.add("REFIT_BASIS", std::string("Knot5"));
    const auto [what, logged] = run_compact(c);
    INFO(logged);
    REQUIRE(what.empty());
    CHECK(logged.find("WARNING: HINT: RHOFORM NONNEG with REFIT_BASIS Knot5: "
                      "a knot density with every amplitude >= 0 is "
                      "non-increasing and convex") != std::string::npos);
    CHECK(logged.find("compact default") == std::string::npos);
    CHECK(slurp("refit_opt.log").find("compact mode") != std::string::npos);
    CHECK(tadah::models::stored_rhoform(
              tadah::core::CoreUserConfigParser().parse("pot.tadah")) ==
          RhoForm::NonNeg);
  }
  SECTION("NONNEG with the defaulted Knot5: the WARNING naming Blip5") {
    for (const bool defaulted : {false, true}) {
      INFO((defaulted ? "default mode" : "REFIT_MODE compact"));
      ScopedTempDir tmp;
      write_synth_eam("synth.eam.alloy");
      Context c = compact_context("NONNEG");
      c.remove("REFIT_BASIS");
      if (defaulted) c.remove("REFIT_MODE");
      const auto [what, logged] = run_compact(c);
      INFO(logged);
      REQUIRE(what.empty());
      CHECK(logged.find("WARNING: tadah refit: RHOFORM NONNEG with the "
                        "REFIT_MODE compact default basis Knot5: a knot "
                        "density with every amplitude >= 0 is non-increasing "
                        "and convex") != std::string::npos);
      CHECK(logged.find("Set REFIT_BASIS Blip5 (a non-negative bump density "
                        "without the convex lock), or use RHOFORM SIGNED.") !=
            std::string::npos);
      CHECK(logged.find("HINT: RHOFORM NONNEG") == std::string::npos);
      CHECK(slurp("refit_opt.log").find("basis=Knot5") != std::string::npos);
      CHECK(tadah::models::stored_rhoform(
                tadah::core::CoreUserConfigParser().parse("pot.tadah")) ==
            RhoForm::NonNeg);
    }
  }
}

#endif  // TADAH_USE_CERES

// test_bump_width.cpp
//
// A radial bump -- SGRID2B of a
// D2_Blip/D2_Blip5 pair term, SGRIDMB of a DM_REAM Blip/Blip5 density --
// needs a width eta > 0 in every RHOFORM. A bump spans
// [c - w/eta, c + w/eta]: at eta <= 0 it has no support edge (the bump is
// even in eta, so a negative width evaluates as its mirror), and the
// bump-support checks, which skipped such widths, could not hold it inside
// the cutoff. tadah hpo refuses:
//   * an OPTIM box whose low bound on a bump width is <= 0, at its line
//     (models::optim_low_bound_rule(key, ctx), the rule check_bounds uses);
//   * a seed potential whose width is <= 0, at startup, naming the key, the
//     bump index and the value;
// and BUMPGATE counts a candidate width <= 0 (an optimiser that leaves the
// box) as a violation instead of skipping it.

#include "catch2/catch.hpp"

#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
namespace bg = tadah::hpo::detail;

struct WidthTempFile {
  fs::path path;
  explicit WidthTempFile(const std::string& body)
  : path(fs::temp_directory_path() /
         ("tadah_bump_width_" +
          std::to_string(std::chrono::steady_clock::now()
                             .time_since_epoch().count()) + ".in")) {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~WidthTempFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

constexpr const char* kWidthOptimizer =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

/* A two-bump D2_Blip (w = 2) pair term; seed supports end at RCUT2B = 5
 * (4.0 + 2/2.0 = 3.0 + 2/1.0 = 5.0). */
Context blip_ctx(const char* sgrid = "2.0 1.0",
                 const char* cutoff = "Cut_Dummy") {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "4.0 3.0");
  c.add("SGRID2B", sgrid);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", cutoff);
  c.add("INITMB", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  return c;
}

/* TYPE2B D2_mJoin of a D2_BP (centres 0 0, widths SGRID2B 1-2) and a
 * two-bump D2_Blip (centres 4.0 3.0, widths SGRID2B 3-4 = `blip_sgrid`),
 * each with RCUT2B 5 and the cutoff `cutoff`. */
Context joined_blip_ctx(const char* blip_sgrid = "2.0 1.0",
                        const char* cutoff = "Cut_Dummy") {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_mJoin D2_BP 2 2 Ta Ta D2_Blip 2 2 Ta Ta");
  c.add("CGRID2B", "0 0 4.0 3.0");
  c.add("SGRID2B", std::string("0.5 1.0 ") + blip_sgrid);
  c.add("RCUT2B", "5.0 5.0");
  c.add("RCTYPE2B", std::string(cutoff) + " " + cutoff);
  c.add("INITMB", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  return c;
}

/* The error HPOTargets(ctx, target) throws ("" when it does not). */
std::string parse_error(Context ctx, const std::string& target) {
  WidthTempFile tf(target);
  try {
    HPOTargets trg(ctx, tf.path.string());
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

/* The error validate_hpo_setup throws ("" when it does not). */
std::string startup_error(Context ctx, const std::string& target) {
  WidthTempFile tf(target);
  HPOTargets trg(ctx, tf.path.string());
  try {
    tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

struct Vec {
  std::vector<double> v;
  double operator()(long i) const { return v[static_cast<std::size_t>(i)]; }
};

}  // namespace

TEST_CASE("an OPTIM box whose low bound on a radial bump width is <= 0 is "
          "refused at its line, in every form",
          "[HPO][optim][bump_width]") {
  tadah::core::PeriodicTable::initialize();
  const std::string tail = std::string("PC_ERMSE 0 1\n") + kWidthOptimizer;

  SECTION("SGRID2B of a D2_Blip pair term, hard or smooth cutoff") {
    for (const char* cutoff : {"Cut_Dummy", "Cut_Cos"}) {
      INFO(cutoff);
      const std::string e =
          parse_error(blip_ctx("2.0 1.0", cutoff),
                      "OPTIM CGRID2B (1) 3.0 3.5\nOPTIM SGRID2B (2) 0 3\n" + tail);
      CHECK_THAT(e, Catch::Contains("Error on line 2") &&
                        Catch::Contains("OPTIM SGRID2B (2)") &&
                        Catch::Contains("the low bound 0 is not > 0") &&
                        Catch::Contains("the D2_Blip pair basis requires every "
                                        "SGRID2B width > 0"));
      CHECK_THAT(parse_error(blip_ctx("2.0 1.0", cutoff),
                             "OPTIM sgrid2b (1) -1 3\n" + tail),
                 Catch::Contains("OPTIM SGRID2B (1)") &&
                     Catch::Contains("the low bound -1 is not > 0"));
      // an AUTO FRAC box reaching below 0 (seed 2, half-width 3)
      CHECK_THAT(parse_error(blip_ctx("2.0 1.0", cutoff),
                             "OPTIM SGRID2B (1) AUTO FRAC 1.5\n" + tail),
                 Catch::Contains("OPTIM SGRID2B (1)") &&
                     Catch::Contains("every SGRID2B width > 0"));
      CHECK(parse_error(blip_ctx("2.0 1.0", cutoff),
                        "OPTIM SGRID2B (2) 0.5 3\n" + tail).empty());
    }
  }
  SECTION("SGRIDMB of a DM_REAM Blip5 density, every RHOFORM") {
    for (const char* form : {"SIGNED", "NONNEG", "MONOTONE"}) {
      INFO(form);
      CHECK_THAT(parse_error(rhoform_test::make_ream_ctx(form),
                             "OPTIM SGRIDMB (1) 0 4\n" + tail),
                 Catch::Contains("OPTIM SGRIDMB (1)") &&
                     Catch::Contains("requires every SGRIDMB width > 0"));
      CHECK(parse_error(rhoform_test::make_ream_ctx(form),
                        "OPTIM SGRIDMB (1) 0.5 4\n" + tail).empty());
    }
    CHECK_THAT(parse_error(rhoform_test::make_ream_ctx("SIGNED"),
                           "OPTIM SGRIDMB (1) 0 4\n" + tail),
               Catch::Contains("the DM_REAM Blip5 density basis requires"));
  }
}

TEST_CASE("a seed potential with a radial bump width <= 0 is refused at "
          "startup, naming the key, the bump and the value",
          "[HPO][startup][bump_width]") {
  tadah::core::PeriodicTable::initialize();
  const std::string target =
      std::string("OPTIM WATOM (1) 0.5 2.0\nPC_ERMSE 0 1\n") + kWidthOptimizer;

  SECTION("SGRID2B, hard cutoff (BUMPGATE skipped the bump)") {
    const std::string e = startup_error(blip_ctx("2.0 -1.0"), target);
    CHECK_THAT(e, Catch::Contains("SGRID2B (2) = -1") &&
                      Catch::Contains("the D2_Blip pair basis requires every "
                                      "SGRID2B width > 0"));
  }
  SECTION("SGRID2B, smooth cutoff; every offending bump is named") {
    const std::string e =
        startup_error(blip_ctx("0 -0.5", "Cut_Cos"), target);
    CHECK_THAT(e, Catch::Contains("SGRID2B (1) = 0") &&
                      Catch::Contains("SGRID2B (2) = -0.5"));
  }
  SECTION("SGRIDMB of a DM_REAM Blip5 density under SIGNED") {
    Context ctx = rhoform_test::make_ream_ctx("SIGNED");
    ctx.remove("SGRIDMB");
    ctx.add("SGRIDMB", "3.0 -3.0 3.0");
    const std::string e = startup_error(
        ctx, std::string("OPTIM CEMBFUNC (1) -0.3 -0.1\nPC_ERMSE 0 1\n") +
                 kWidthOptimizer);
    CHECK_THAT(e, Catch::Contains("SGRIDMB (2) = -3") &&
                      Catch::Contains("the DM_REAM Blip5 density basis "
                                      "requires every SGRIDMB width > 0"));
  }
  SECTION("positive widths pass") {
    CHECK(startup_error(blip_ctx("2.0 1.0"), target).empty());
  }
}

TEST_CASE("BUMPGATE: a candidate width <= 0 is a violation, not a skip",
          "[HPO][bumpgate][bump_width]") {
  tadah::core::PeriodicTable::initialize();

  SECTION("the per-candidate measure continues through eta = 0") {
    Context ctx = blip_ctx();
    WidthTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                     std::string(kWidthOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    const auto tbl = bg::build_bump_gate_table(ctx, trg);
    REQUIRE(tbl.size() == 1);
    REQUIRE(tbl[0].bumps.size() == 1);
    REQUIRE(tbl[0].bumps[0].s_flat == 0);
    // bump 1: c = 4, w = 2, rcut = 5. For eta > 0 the overhang measure is
    // (c + w/eta - rcut) eta / w = 1 - (rcut - c) eta / w -> 1 as eta -> 0+;
    // for eta <= 0 it is 1 - eta rcut / w: 1 at 0, growing below it.
    CHECK(bg::bump_gate_violation(tbl, Vec{{2.0}}) == 0.0);
    const double v_small = bg::bump_gate_violation(tbl, Vec{{1e-9}});
    CHECK(v_small == Approx(1.0));
    CHECK(bg::bump_gate_violation(tbl, Vec{{0.0}}) == 1.0);
    CHECK(bg::bump_gate_violation(tbl, Vec{{-0.5}}) == Approx(2.25));
    CHECK(bg::bump_gate_violation(tbl, Vec{{-1.0}}) == Approx(3.5));
    CHECK(bg::bump_gate_violation(tbl, Vec{{0.0}}) > v_small);
    CHECK(bg::bump_gate_violation(tbl, Vec{{std::nan("")}}) ==
          std::numeric_limits<double>::infinity());
  }
  SECTION("a seed width <= 0 never reaches the gate table") {
    // validate_hpo_setup refuses it first; the table refuses it too
    Context ctx = blip_ctx("2.0 -1.0");
    WidthTempFile tf("OPTIM CGRID2B (2) 2.0 3.5\nPC_ERMSE 0 1\n" +
                     std::string(kWidthOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_AS(bg::build_bump_gate_table(ctx, trg), std::logic_error);
    CHECK_THROWS_WITH(bg::build_bump_gate_table(ctx, trg),
                      Catch::Contains("SGRID2B (2) = -1"));
  }
}

TEST_CASE("BUMPGATE's reject text prints the violation losslessly",
          "[HPO][bumpgate][bumpgate_text]") {
  // std::to_string printed six decimals: a violation below 5e-7 read
  // "violation=0.000000" and every value lost its digits. The text now
  // carries the shortest round-trip form (core::format_roundtrip), as
  // hpo_optim_resolved.hpo does.
  CHECK(bg::bump_gate_reject_what(2.5e-7) ==
        "bump-support gate reject (c + w/eta past the cutoff, "
        "violation=2.5e-07, graded band D)");
  CHECK(bg::bump_gate_reject_what(0.8) ==
        "bump-support gate reject (c + w/eta past the cutoff, "
        "violation=0.8, graded band D)");
  const double third = 1.0 / 3.0;
  const std::string t = bg::bump_gate_reject_what(third);
  const auto at = t.find("violation=") + 10;
  CHECK(std::stod(t.substr(at, t.find(',', at) - at)) == third);
  CHECK(bg::bump_gate_reject_what(std::numeric_limits<double>::infinity())
            .find("violation=inf") != std::string::npos);
}

TEST_CASE("BUMPGATE: grids that cannot be read are an error, not a skipped "
          "channel", "[HPO][bumpgate][model_probe]") {
  /* The gate table and the startup gate skipped a bump channel whose
   * CGRID/SGRID could not be read, and the binding cutoff fell back to the
   * first RCUT token when the per-pair list could not be read. Such a
   * channel cannot build its descriptor (validate_hpo_setup refuses the
   * configuration first); the gate says so instead of skipping it. */
  tadah::core::PeriodicTable::initialize();
  SECTION("CGRID2B") {
    Context ctx = blip_ctx();
    WidthTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                     std::string(kWidthOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    ctx.storage_update("CGRID2B", {"4.0", "x"});
    CHECK_THROWS_WITH(bg::build_bump_gate_table(ctx, trg),
                      Catch::Contains("CGRID2B"));
  }
  SECTION("RCUT2B") {
    Context ctx = blip_ctx();
    WidthTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                     std::string(kWidthOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    ctx.storage_update("RCUT2B", {"5.0", "x"});
    CHECK_THROWS_WITH(bg::derive_binding_rcut(ctx, trg, "2B"),
                      Catch::Contains("RCUT2B"));
  }
}

/* A bump centre beyond the cutoff only keeps its inner shoulder, which
 * startup warns about for the configuration's cutoff (smooth cutoff; a hard
 * one is an error). With OPTIM RCUT2B the search tries cutoffs down to the
 * box's low bound: a centre beyond that low bound is warned about too,
 * naming the bump, its centre and the low bound ("CGRID2B bump 1 centre 4
 * lies beyond the OPTIM RCUT2B low bound 3.5: ..."), and a centre below it
 * is not.
 * Before the fix: the check read the configuration's RCUT2B (5) only, so
 * nothing warned. */
TEST_CASE("a bump centre beyond the low bound of an OPTIM RCUT2B box is "
          "warned about at startup", "[HPO][startup][bump_width]") {
  tadah::core::PeriodicTable::initialize();
  const auto warnings = [](const std::string& target) {
    WidthTempFile tf(target);
    Context ctx = blip_ctx("2.0 1.0", "Cut_Cos");
    HPOTargets trg(ctx, tf.path.string());
    /* A startup WARNING of the run: on standard error ("[HPO] WARNING: ")
     * and in the journal (test_startup_warnings_journal.cpp). It went
     * through core::Logger to standard output. */
    std::ostringstream captured;
    auto* old = std::cerr.rdbuf(captured.rdbuf());
    try {
      tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
    } catch (...) {
      std::cerr.rdbuf(old);
      throw;
    }
    std::cerr.rdbuf(old);
    return captured.str();
  };
  const std::string tail = std::string("PC_ERMSE 0 1\n") + kWidthOptimizer;
  {
    const std::string w = warnings("OPTIM RCUT2B (1) 3.5 6.0\n" + tail);
    INFO(w);
    CHECK_THAT(w, Catch::Contains(
                      "[HPO] WARNING: "
                      "CGRID2B bump 1 centre 4 lies beyond the OPTIM RCUT2B "
                      "low bound 3.5: a candidate whose RCUT2B is below the "
                      "centre keeps only the bump's inner shoulder "
                      "(pathological); with the OPTIM RCUT2B low bound "
                      "above 4, or the centre below 3.5, no candidate "
                      "does."));
    CHECK(w.find("CGRID2B bump 2") == std::string::npos);
  }
  {
    /* every centre below the low bound: no such warning */
    const std::string w = warnings("OPTIM RCUT2B (1) 4.5 6.0\n" + tail);
    INFO(w);
    CHECK(w.find("lies beyond") == std::string::npos);
  }
}

/* Pins: the widths of a D2_Blip descriptor of a D2_mJoin line are held > 0
 * as those of a line of one D2_Blip: an OPTIM SGRID2B box from 0 is refused
 * at its line ("the low bound 0 is not > 0, but the D2_Blip pair basis of
 * the D2_mJoin line requires every SGRID2B width > 0"), and a seed width of
 * -1 at startup ("SGRID2B (4) = -1: ..."), under a hard or a smooth cutoff.
 * Before the fix: both accepted (no width rule for a D2_mJoin line; the
 * bump-support checks skipped the joined descriptor). */
TEST_CASE("the widths of a D2_Blip of a D2_mJoin line are held > 0, at the "
          "OPTIM line and at startup",
          "[HPO][optim][startup][bump_width][join]") {
  tadah::core::PeriodicTable::initialize();
  const std::string tail = std::string("PC_ERMSE 0 1\n") + kWidthOptimizer;
  for (const char* cutoff : {"Cut_Dummy", "Cut_Cos"}) {
    INFO(cutoff);
    const std::string e = parse_error(joined_blip_ctx("2.0 1.0", cutoff),
                                      "OPTIM SGRID2B (4) 0 3\n" + tail);
    CHECK_THAT(e, Catch::Contains("Error on line 1") &&
                      Catch::Contains("OPTIM SGRID2B (4)") &&
                      Catch::Contains("the low bound 0 is not > 0") &&
                      Catch::Contains("the D2_Blip pair basis of the D2_mJoin "
                                      "line requires every SGRID2B width > 0"));
    CHECK(parse_error(joined_blip_ctx("2.0 1.0", cutoff),
                      "OPTIM SGRID2B (4) 0.5 3\n" + tail).empty());
    const std::string s = startup_error(
        joined_blip_ctx("2.0 -1.0", cutoff),
        std::string("OPTIM WATOM (1) 0.5 2.0\n") + tail);
    CHECK_THAT(s, Catch::Contains("SGRID2B (4) = -1") &&
                      Catch::Contains("the D2_Blip pair basis of the D2_mJoin "
                                      "line requires every SGRID2B width > 0"));
    CHECK(startup_error(joined_blip_ctx("2.0 1.0", cutoff),
                        std::string("OPTIM WATOM (1) 0.5 2.0\n") + tail)
              .empty());
  }
}

// test_bump_width.cpp
//
// Owner decision (2026-09-29, C8): a radial bump -- SGRID2B of a
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
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cmath>
#include <filesystem>
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

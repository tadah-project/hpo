// Catch2 v2 tests of the low-bound rule of the OPTIM boxes as tadah hpo
// reads them (HPOTargets): an explicit box below the lower limit that the
// descriptor reading the key declares is refused at its line, an AUTO box
// that reaches below it is clamped at it and searched clamped, and
// hpo_optim_resolved.hpo records the clamped box.

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp
#include "catch2/catch.hpp"
#include "target_file_text.h"
#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/search_space.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace rhoform_test;
using tadah::hpo::BoxOrigin;

namespace {

/* D2_BP 2 2 H H and DM_mBlipEnv 0 1 1 2 2 H H (CGRIDMB 0, SGRIDMB 0.2,
 * CEMBFUNC 1 2, SEMBFUNC 1 1, RCUTENV 1.5), the training configuration of
 * the registry's lines_blipenv. */
Context blipenv_ctx() {
  tadah::core::PeriodicTable::initialize();
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1.0);
  c.add("DBFILE", "dummy.db");
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_BP 2 2 H H");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "0.5 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("TYPEMB", "DM_mBlipEnv 0 1 1 2 2 H H");
  c.add("CGRIDMB", 0.0);
  c.add("SGRIDMB", 0.2);
  c.add("CEMBFUNC", "1.0 2.0");
  c.add("SEMBFUNC", "1.0 1.0");
  c.add("RCUTENV", 1.5);
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUTMB", 5.0);
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR BF_Linear");
  c.add("OALGO", 4);
  c.add("LAMBDA", 1e-3);
  return c;
}

/* hpo_optim_resolved.hpo of the target file @p target read against
 * @p ctx. */
std::string resolved_of(Context ctx, const std::string& target) {
  TempFile tf(target);
  HPOTargets trg(ctx, tf.path.string());
  const fs::path dir =
      fs::temp_directory_path() / unique_name("tadah_low_bound_ss_");
  fs::create_directories(dir);
  trg.write_search_space_files(dir.string());
  std::ifstream is(dir / "hpo_optim_resolved.hpo");
  std::ostringstream oss;
  oss << is.rdbuf();
  std::error_code ec;
  fs::remove_all(dir, ec);
  return oss.str();
}

}  // namespace

/* Pins: tadah hpo searches an AUTO box clamped at the lower limit of its
 * key and records it: OPTIM RCUTENV (1) AUTO FRAC 1.5 (seed 1.5, x0 - d =
 * -0.75) is the dimension [0.75, 3.75] with the origin "AUTO FRAC>0" (an
 * AUTO FRAC box clamped at half its seed, the key's values being > 0), and
 * hpo_optim_resolved.hpo has "OPTIM RCUTENV (1) 0.75 3.75   # AUTO
 * FRAC>0"; OPTIM CGRIDMB (1) AUTO FRAC 0 FLOOR 0.5 (seed 0) is [0, 0.5],
 * "AUTO FRAC>=0"; OPTIM RCUTENV (1) AUTO is the RATIO 2 box [0.75, 3],
 * "AUTO RATIO", RCUTENV being a key whose values stay > 0.
 * Before the fix: the RCUTENV dimension was [-0.75, 3.75] ("AUTO FRAC"),
 * whose part below 0 every evaluation refused ("Cutoff distance cannot be
 * negative"), the CGRIDMB dimension [-0.5, 0.5], and OPTIM RCUTENV (1)
 * AUTO the FRAC 0.5 box [0.75, 2.25]. Before the fix of the clamp: the
 * RCUTENV dimension was [1e-6, 3.75] and the line "OPTIM RCUTENV (1) 1e-06
 * 3.75   # AUTO FRAC>0". */
TEST_CASE("tadah hpo searches and records an AUTO box clamped at the lower "
          "limit of its key",
          "[hpo][optim][low_bound]") {
  const std::string tail = std::string("PC_ERMSE 0 1\n") + kOptimizer;
  auto dims = [&](const std::string& line) {
    Context ctx = blipenv_ctx();
    TempFile tf(line + "\n" + tail);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.space());
    std::vector<tadah::hpo::DimSpec> d;
    for (std::size_t i = 0; i < trg.space()->dim(); ++i)
      d.push_back(trg.space()->at(i));
    return d;
  };
  const auto r = dims("OPTIM RCUTENV (1) AUTO FRAC 1.5");
  REQUIRE(r.size() == 1);
  CHECK(r[0].lo == 0.75);
  CHECK(r[0].hi == 3.75);
  CHECK(r[0].origin == BoxOrigin::AutoLinearPositive);
  CHECK(std::string(tadah::hpo::box_origin_name(r[0].origin)) ==
        "AUTO FRAC>0");
  CHECK_THAT(resolved_of(blipenv_ctx(),
                         "OPTIM RCUTENV (1) AUTO FRAC 1.5\n" + tail),
             Catch::Contains("OPTIM RCUTENV (1) 0.75 3.75   # AUTO FRAC>0"));
  const auto c = dims("OPTIM CGRIDMB (1) AUTO FRAC 0 FLOOR 0.5");
  REQUIRE(c.size() == 1);
  CHECK(c[0].lo == 0.0);
  CHECK(c[0].hi == 0.5);
  CHECK(std::string(tadah::hpo::box_origin_name(c[0].origin)) ==
        "AUTO FRAC>=0");
  const auto g = dims("OPTIM RCUTENV (1) AUTO");
  REQUIRE(g.size() == 1);
  CHECK(g[0].lo == 0.75);
  CHECK(g[0].hi == 3.0);
  CHECK(std::string(tadah::hpo::box_origin_name(g[0].origin)) ==
        "AUTO RATIO");
}

/* Pins: tadah hpo refuses an explicit box below the lower limit of its
 * key at its line: "line 1 of <target>: OPTIM RCUTENV (1): the low bound
 * -1 is not > 0, but the DM_mBlipEnv descriptor requires every RCUTENV
 * value > 0, so the low bound of its box is > 0 (an AUTO FRAC box is
 * clamped at half its seed)", and the centre box CGRID2B (1) -0.5 1.5
 * ("... is negative, but the D2_BP descriptor requires every CGRID2B value
 * >= 0, ..."); OPTIM RCUTENV (1) 0.5 3 is read.
 * Before the fix: both boxes were read. Before the fix of the clamp: the
 * message ended "(an AUTO FRAC box is clamped at 1e-06)". */
TEST_CASE("tadah hpo refuses an explicit box below the lower limit of its "
          "key at its line",
          "[hpo][optim][low_bound]") {
  const std::string tail = std::string("PC_ERMSE 0 1\n") + kOptimizer;
  CHECK_THAT(targets_error(blipenv_ctx(), "OPTIM RCUTENV (1) -1 7\n" + tail),
             Catch::Contains(
                 "line 1 of <target>: OPTIM RCUTENV (1): the low bound -1 is "
                 "not > 0, but the DM_mBlipEnv descriptor requires every "
                 "RCUTENV value > 0, so the low bound of its box is > 0 (an "
                 "AUTO FRAC box is clamped at half its seed)"));
  CHECK_THAT(targets_error(blipenv_ctx(), "OPTIM CGRID2B (1) -0.5 1.5\n" + tail),
             Catch::Contains("line 1 of <target>: OPTIM CGRID2B (1): the low "
                             "bound -0.5 is negative, but the D2_BP "
                             "descriptor requires every CGRID2B value >= 0"));
  CHECK(targets_error(blipenv_ctx(), "OPTIM RCUTENV (1) 0.5 3\n" + tail) == "");
}

/* Pins: under LOG_HP 100 the clamped AUTO FRAC box of a value > 0 keeps
 * the shape LIN: OPTIM RCUTENV (1) AUTO FRAC 1.5 (seed 1.5) and OPTIM
 * SEMBFUNC (1) AUTO FRAC 2 (seed 1) are [0.75, 3.75] and [0.5, 3], whose
 * high ends are 5 and 6 times their low ends, below the ratio 100 at
 * which LOG_HP gives an untagged dimension the shape LOG; the search stays
 * around the seed.
 * Before the fix of the clamp: the boxes were [1e-6, 3.75] and [1e-6, 3],
 * which LOG_HP 100 made LOG over more than six decades, most of the
 * search below 0.01 against seeds of 1.5 and 1. */
TEST_CASE("tadah hpo keeps a clamped AUTO FRAC box near its seed under "
          "LOG_HP",
          "[hpo][optim][low_bound]") {
  Context ctx = blipenv_ctx();
  TempFile tf(std::string("OPTIM RCUTENV (1) AUTO FRAC 1.5\n"
                          "OPTIM SEMBFUNC (1) AUTO FRAC 2\n"
                          "PC_ERMSE 0 1\n"
                          "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                          "LOG_HP 100\nENDOPTIMIZER\n"));
  HPOTargets trg(ctx, tf.path.string());
  REQUIRE(trg.space());
  REQUIRE(trg.space()->dim() == 2);
  const auto& r = trg.space()->at(0);
  const auto& e = trg.space()->at(1);
  CHECK(r.lo == 0.75);
  CHECK(r.hi == 3.75);
  CHECK_FALSE(r.auto_log);
  CHECK(e.lo == 0.5);
  CHECK(e.hi == 3.0);
  CHECK_FALSE(e.auto_log);
}

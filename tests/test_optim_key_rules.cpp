// test_optim_key_rules.cpp
//
// The key of an OPTIM line is a configuration key, and configuration keys
// are case-insensitive (core::Context stores them upper-case). HPOTargets
// upper-cases it when the line is parsed, so every rule and lookup keyed
// on it applies whatever the case it was typed in: the WEIGHTS refusal,
// the FIXWEIGHT slot mapping through FIXINDEX, the bump-support gate, the
// neighbour-list cutoff and the labels of params.tadah and the journal.
// An index an earlier OPTIM line of the same key already selects is an
// error at the second line, in any case.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;

struct KeyRulesTempFile {
  fs::path path;
  explicit KeyRulesTempFile(const std::string& body)
  : path(fs::temp_directory_path() /
         ("tadah_optim_key_rules_" +
          std::to_string(std::chrono::steady_clock::now()
                             .time_since_epoch().count()) + ".in")) {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~KeyRulesTempFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

constexpr const char* kKeyRulesOptimizer =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

/* A hard-cutoff (Cut_Dummy) two-bump D2_Blip channel (w = 2) whose seed
 * supports end exactly at RCUT2B = 5: 4.0 + 2/2.0 = 3.0 + 2/1.0 = 5.0. */
Context key_rules_gate_ctx() {
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
  c.add("SGRID2B", "2.0 1.0");
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("INITMB", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  return c;
}

}  // namespace

TEST_CASE("OPTIM keys are upper-cased at parse: dimensions and labels",
          "[HPO][optim][optim_key_case]") {
  Context ctx;
  ctx.add("AMPGRIDMB", "0.5 0.6");
  ctx.add("CGRIDMB", "1.0 2.0");
  KeyRulesTempFile tf("OPTIM ampgridmb (1) 0 1\nOPTIM CGRIDMB (2) 3 4\n" +
                      std::string(kKeyRulesOptimizer));
  HPOTargets trg(ctx, tf.path.string());
  /* one entry per configuration key, keyed upper-case */
  CHECK(trg.optimap.count("AMPGRIDMB") == 1);
  CHECK(trg.optimap.count("ampgridmb") == 0);
  /* the search space in key order, labelled upper-case (params.tadah,
   * the journal and hpo_search_space.tsv print these labels) */
  REQUIRE(trg.spec.parameter_labels.size() == 2);
  CHECK(trg.spec.parameter_labels[0] == "AMPGRIDMB[1]");
  CHECK(trg.spec.parameter_labels[1] == "CGRIDMB[2]");
  REQUIRE(trg.space());
  CHECK(trg.space()->at(0).key == "AMPGRIDMB");
  CHECK(trg.space()->at(1).key == "CGRIDMB");
}

TEST_CASE("OPTIM weights is refused like OPTIM WEIGHTS",
          "[HPO][optim][optim_key_case][fixweight]") {
  Context ctx;
  ctx.add("WEIGHTS", 0.4);
  KeyRulesTempFile tf("OPTIM weights (1) -1.0 1.0\n" +
                      std::string(kKeyRulesOptimizer));
  REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                      Catch::Contains("Error on line 1") &&
                      Catch::Contains("OPTIM WEIGHTS is not a valid search "
                                      "target"));
}

TEST_CASE("OPTIM fixweight maps its indices through FIXINDEX",
          "[HPO][optim][optim_key_case][fixweight]") {
  /* FIXINDEX 4,1,7: weight 7 is pinned by slot 3. */
  Context ctx;
  ctx.add("FIXINDEX", "4,1,7");
  ctx.add("FIXWEIGHT", 0.4);
  ctx.add("FIXWEIGHT", 0.1);
  ctx.add("FIXWEIGHT", 0.8);
  KeyRulesTempFile tf("OPTIM fixweight (7) 0.5 1.5\n" +
                      std::string(kKeyRulesOptimizer));
  HPOTargets trg(ctx, tf.path.string());
  REQUIRE(trg.spec.parameter_labels.size() == 1);
  CHECK(trg.spec.parameter_labels[0] == "FIXWEIGHT[7]");
  const auto it = trg.optimap.find("FIXWEIGHT");
  REQUIRE(it != trg.optimap.end());
  REQUIRE(it->second.opt.size() == 3);
  CHECK_FALSE(it->second.opt[0]);
  CHECK_FALSE(it->second.opt[1]);
  CHECK(it->second.opt[2]);
}

TEST_CASE("lower-case OPTIM cgrid/sgrid keys arm the bump-support gate",
          "[HPO][optim][optim_key_case][bumpgate]") {
  tadah::core::PeriodicTable::initialize();
  namespace bg = tadah::hpo::detail;

  SECTION("CANDIDATE: the per-candidate table") {
    Context ctx = key_rules_gate_ctx();
    KeyRulesTempFile tf("OPTIM cgrid2b (1) 3.0 4.8\nOPTIM sgrid2b (1) 1.0 3.0\n"
                        "PC_ERMSE 0 1\n" + std::string(kKeyRulesOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    const auto tbl = bg::build_bump_gate_table(ctx, trg);
    REQUIRE(tbl.size() == 1);
    REQUIRE(tbl[0].bumps.size() == 1);
    CHECK(tbl[0].bumps[0].c_flat == 0);
    CHECK(tbl[0].bumps[0].s_flat == 1);
  }
  SECTION("STRICT: the startup box-corner error") {
    Context ctx = key_rules_gate_ctx();
    KeyRulesTempFile tf("BUMPGATE STRICT\nOPTIM cgrid2b (1) 3.0 4.8\n"
                        "PC_ERMSE 0 1\n" + std::string(kKeyRulesOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(tadah::hpo::validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("bump-support gate") &&
                      Catch::Contains("bump 1 of CGRID2B"));
  }
  SECTION("a lower-case OPTIM rcut2b bounds the neighbour list") {
    Context ctx = key_rules_gate_ctx();   // RCUTMAX 5, from RCUT2B
    KeyRulesTempFile tf("OPTIM rcut2b (1) 4.0 7.0\nPC_ERMSE 0 1\n" +
                        std::string(kKeyRulesOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    CHECK(bg::nn_safe_rcutmax(ctx, trg) == 7.0);
  }
}

TEST_CASE("an OPTIM index selected twice is an error at the second line",
          "[HPO][optim][optim_key_case]") {
  Context ctx;
  ctx.add("CGRIDMB", "1.0 2.0 3.0");

  SECTION("the same key in two cases, the same index") {
    KeyRulesTempFile tf("OPTIM cgridmb (1) 0 1\nOPTIM CGRIDMB (1) 0.5 2\n" +
                        std::string(kKeyRulesOptimizer));
    REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                        Catch::Contains("Error on line 2") &&
                        Catch::Contains("OPTIM CGRIDMB (1)") &&
                        Catch::Contains("line 1"));
  }
  SECTION("the same key in one case, overlapping selections") {
    KeyRulesTempFile tf("OPTIM CGRIDMB (1-2) 0 3\nOPTIM CGRIDMB (2,3) 0 4\n" +
                        std::string(kKeyRulesOptimizer));
    REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                        Catch::Contains("Error on line 2") &&
                        Catch::Contains("OPTIM CGRIDMB (2)") &&
                        Catch::Contains("line 1"));
  }
  SECTION("the same key in two cases, other indices: one key, merged") {
    KeyRulesTempFile tf("OPTIM cgridmb (1) 0 1.5\nOPTIM CGRIDMB (3) 2 4\n" +
                        std::string(kKeyRulesOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    CHECK(trg.optimap.size() == 1);
    REQUIRE(trg.spec.parameter_labels.size() == 2);
    CHECK(trg.spec.parameter_labels[0] == "CGRIDMB[1]");
    CHECK(trg.spec.parameter_labels[1] == "CGRIDMB[3]");
    REQUIRE(trg.space());
    CHECK(trg.space()->at(0).lo == 0.0);
    CHECK(trg.space()->at(0).hi == 1.5);
    CHECK(trg.space()->at(1).lo == 2.0);
    CHECK(trg.space()->at(1).hi == 4.0);
  }
}

TEST_CASE("OPTIM on DIMER, FIXINDEX, MODEL and the OALGO algorithm is refused "
          "at its line, as on the other structure keys",
          "[HPO][optim][optim_structure_keys]") {
  /* Each refused line comes second, after a valid one, so the error must
   * name line 2. */
  const std::string first = "OPTIM WATOM (1) 0.5 2.0\n";
  auto base_ctx = [] {
    Context c;
    c.add("WATOM", 1.0);
    return c;
  };
  struct Case {
    const char* line;
    const char* key;
    const char* what;
  };
  const Case cases[] = {
    {"OPTIM DIMER (1) 1.0 1.2", "DIMER", "rigid-dimer"},
    {"OPTIM dimer (1) 1.0 1.2", "DIMER", "rigid-dimer"},
    {"OPTIM FIXINDEX (1) 1 5", "FIXINDEX", "OPTIM FIXWEIGHT"},
    {"OPTIM MODEL (1) 1 3", "MODEL", "model"},
    {"OPTIM Model (1) 1 3", "MODEL", "model"},
  };
  for (const Case& c : cases) {
    INFO(c.line);
    Context ctx = base_ctx();
    ctx.add("DIMER", "true 1.104 true");
    ctx.add("FIXINDEX", 4);
    ctx.add("FIXINDEX", 1);
    ctx.add("FIXINDEX", 7);
    ctx.add("FIXWEIGHT", "0.4 0.1 0.8");
    ctx.add("MODEL", "M_BLR BF_Polynomial2 2");
    KeyRulesTempFile tf(first + c.line + "\n" + kKeyRulesOptimizer);
    CHECK_THROWS_WITH(
        HPOTargets(ctx, tf.path.string()),
        Catch::Contains("Error on line 2") &&
        Catch::Contains(std::string("OPTIM ") + c.key +
                        " is not a valid search target") &&
        Catch::Contains(c.what) &&
        Catch::Contains("structure data rather than a continuous parameter"));
  }

  SECTION("OALGO: the algorithm (its first value) is refused") {
    for (const char* line : {"OPTIM OALGO (1) 1 4", "OPTIM oalgo (1,2) 1 4"}) {
      INFO(line);
      Context ctx = base_ctx();
      ctx.add("OALGO", "1 1e-8");
      KeyRulesTempFile tf(first + line + "\n" + kKeyRulesOptimizer);
      CHECK_THROWS_WITH(
          HPOTargets(ctx, tf.path.string()),
          Catch::Contains("Error on line 2") &&
          Catch::Contains("OPTIM OALGO (1) is not a valid search target") &&
          Catch::Contains("training algorithm") &&
          Catch::Contains("OPTIM OALGO (2)"));
    }
  }
  SECTION("OALGO: the rcond M of options 1 and 3 (its second value) is a "
          "continuous parameter") {
    Context ctx = base_ctx();
    ctx.add("OALGO", "1 1e-8");
    KeyRulesTempFile tf(first + "OPTIM OALGO (2) 1e-10 1e-6 LOG\n" +
                        kKeyRulesOptimizer);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.parameter_labels.size() == 2);
    CHECK(trg.spec.parameter_labels[0] == "OALGO[2]");
    CHECK(trg.spec.parameter_labels[1] == "WATOM");
  }
}

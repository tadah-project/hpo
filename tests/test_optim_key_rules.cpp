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

namespace {

/* A D2_BP Ta model (2 Gaussians, Cut_Cos 4.5 A, no many-body term) with
 * the MODEL tokens @p model, MPARAMS @p mparams and SBASIS 3 for a kernel
 * that draws a basis; a training DBFILE the startup checks never open. */
Context mparams_ctx(const std::string& model, const std::string& mparams) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_BP 2 2 Ta Ta");
  c.add("CGRID2B", "2.6 3.3");
  c.add("SGRID2B", "1.0 1.0");
  c.add("RCUT2B", 4.5);
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("INITMB", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", model);
  c.add("MPARAMS", mparams);
  if (model.rfind("M_KRR", 0) == 0 && model != "M_KRR Kern_Linear")
    c.add("SBASIS", 3);
  c.add("DBFILE", "dummy.db");
  return c;
}

/* The message validate_hpo_setup throws for @p ctx and the hpotarget
 * line @p line, "" when it accepts them. */
std::string startup_error(Context ctx, const std::string& line) {
  KeyRulesTempFile tf(line + "\nPC_ERMSE 0 1\n" + kKeyRulesOptimizer);
  try {
    HPOTargets trg(ctx, tf.path.string());
    tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

}  // namespace

/* Pins: OPTIM MPARAMS is refused at startup on a function that reads no
 * MPARAMS (BF_Linear, BF_Polynomial2, Kern_Linear, Kern_Quadratic,
 * Kern_LQ), naming OPTIM MPARAMS (1), the function and that it reads none.
 * Before the fix: accepted; every evaluation trained the same model, so
 * the search changed nothing. */
TEST_CASE("OPTIM MPARAMS is refused on a function that reads none",
          "[HPO][optim][optim_mparams]") {
  tadah::core::PeriodicTable::initialize();
  for (const char* model : {"M_BLR BF_Linear", "M_BLR BF_Polynomial2",
                            "M_KRR Kern_Linear", "M_KRR Kern_Quadratic",
                            "M_KRR Kern_LQ"}) {
    INFO(model);
    const std::string msg =
        startup_error(mparams_ctx(model, "1.0"), "OPTIM MPARAMS (1) 0.5 2.0");
    INFO("message: " << msg);
    CHECK(has(msg, "OPTIM MPARAMS (1) is not a valid search target"));
    CHECK(has(msg, std::string(model).substr(6)));
    CHECK(has(msg, "reads no MPARAMS"));
  }
}

/* Pins: OPTIM MPARAMS of Kern_Polynomial (d gamma c): (1), the degree d,
 * is refused at startup as an integer, naming the degree; (2) gamma and
 * (3) c are accepted.
 * Before the fix: (1) was accepted, and every candidate degree was read as
 * its integer prefix (3.84 trained the degree 3). */
TEST_CASE("OPTIM MPARAMS of the Kern_Polynomial degree is refused",
          "[HPO][optim][optim_mparams]") {
  tadah::core::PeriodicTable::initialize();
  const Context c = mparams_ctx("M_KRR Kern_Polynomial", "2 0.5 1.0");
  const std::string msg = startup_error(c, "OPTIM MPARAMS (1) 1 4");
  INFO("message: " << msg);
  CHECK(has(msg, "OPTIM MPARAMS (1) is not a valid search target"));
  CHECK(has(msg, "Kern_Polynomial"));
  CHECK(has(msg, "d, an integer"));
  CHECK(startup_error(c, "OPTIM MPARAMS (2) 0.1 1.0").empty());
  CHECK(startup_error(c, "OPTIM MPARAMS (3) 0.0 2.0").empty());
}

/* Pins: Kern_RBF reads one MPARAMS value, sigma: OPTIM MPARAMS (2) is
 * refused at startup naming the one value it reads; OPTIM MPARAMS (1) with
 * a box that is not positive (-1 2, 0 2) is refused naming sigma and the
 * bound; (1) with 0.3 3.0 is accepted.
 * Before the fix: all accepted; (2) changed nothing, and a box through 0
 * evaluated sigma 0 (gamma = inf) and negative widths. */
TEST_CASE("OPTIM MPARAMS of Kern_RBF: one positive value",
          "[HPO][optim][optim_mparams]") {
  tadah::core::PeriodicTable::initialize();
  const Context c = mparams_ctx("M_KRR Kern_RBF", "1.0 2.0");
  const std::string beyond = startup_error(c, "OPTIM MPARAMS (2) 0.5 3.0");
  INFO("(2): " << beyond);
  CHECK(has(beyond, "OPTIM MPARAMS (2) is not a valid search target"));
  CHECK(has(beyond, "Kern_RBF reads 1 MPARAMS value"));
  for (const char* box : {"-1 2", "0 2"}) {
    const std::string msg =
        startup_error(c, std::string("OPTIM MPARAMS (1) ") + box);
    INFO("box " << box << ": " << msg);
    CHECK(has(msg, "OPTIM MPARAMS (1)"));
    CHECK(has(msg, "sigma"));
    CHECK(has(msg, "must be positive"));
  }
  CHECK(startup_error(c, "OPTIM MPARAMS (1) 0.3 3.0").empty());
}

/* Pins: OPTIM on an integer value of a key (every format of the key types
 * it INT or UINT: SBASIS) is refused at its line, naming the key, its
 * index and that it is an integer; OPTIM on an internal key (ESTDEV, which
 * tadah computes) is refused at its line naming it; OPTIM OALGO (1) keeps
 * its own message (the training algorithm).
 * Before the fix: OPTIM SBASIS was accepted (14.47 trained SBASIS 14) and
 * OPTIM ESTDEV too (every evaluation then failed: "Attempt to add internal
 * key"). */
TEST_CASE("OPTIM on an integer key or an internal key is refused at its line",
          "[HPO][optim][optim_structure_keys]") {
  const std::string first = "OPTIM WATOM (1) 0.5 2.0\n";
  Context ctx;
  ctx.add("WATOM", 1.0);
  ctx.add("SBASIS", 10);
  ctx.add("OALGO", "1 1e-8");
  {
    KeyRulesTempFile tf(first + "OPTIM SBASIS (1) 5 20\n" + kKeyRulesOptimizer);
    CHECK_THROWS_WITH(
        HPOTargets(ctx, tf.path.string()),
        Catch::Contains("Error on line 2") &&
        Catch::Contains("OPTIM SBASIS (1) is not a valid search target") &&
        Catch::Contains("integer"));
  }
  {
    KeyRulesTempFile tf(first + "OPTIM ESTDEV (1) 0 1\n" + kKeyRulesOptimizer);
    CHECK_THROWS_WITH(
        HPOTargets(ctx, tf.path.string()),
        Catch::Contains("Error on line 2") &&
        Catch::Contains("OPTIM ESTDEV is not a valid search target") &&
        Catch::Contains("internal"));
  }
  {
    KeyRulesTempFile tf(first + "OPTIM OALGO (1) 1 4\n" + kKeyRulesOptimizer);
    CHECK_THROWS_WITH(
        HPOTargets(ctx, tf.path.string()),
        Catch::Contains("OPTIM OALGO (1) is not a valid search target") &&
        Catch::Contains("training algorithm"));
  }
}

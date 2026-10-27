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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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
                      Catch::Contains("OPTIM WEIGHTS: WEIGHTS holds the "
                                      "coefficients each evaluation fits"));
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

/* Pins: a FIXWEIGHT with one value names its model weight as a FIXWEIGHT
 * with several does: FIXINDEX 26, FIXWEIGHT 0.4 and OPTIM FIXWEIGHT (26)
 * label the dimension FIXWEIGHT[26] (the params.tadah column, the
 * journal and hpo_search_space.tsv), the index being a model weight, not
 * a position among the values of the key.
 * Before the fix: "FIXWEIGHT", the label of a key with one value. */
TEST_CASE("OPTIM FIXWEIGHT with one value is labelled with its model weight",
          "[HPO][optim][fixweight]") {
  Context ctx;
  ctx.add("FIXINDEX", "26");
  ctx.add("FIXWEIGHT", 0.4);
  KeyRulesTempFile tf("OPTIM FIXWEIGHT (26) 0 1\n" +
                      std::string(kKeyRulesOptimizer));
  HPOTargets trg(ctx, tf.path.string());
  REQUIRE(trg.spec.parameter_labels.size() == 1);
  CHECK(trg.spec.parameter_labels[0] == "FIXWEIGHT[26]");
  REQUIRE(trg.spec.space);
  CHECK(trg.spec.space->at(0).label == "FIXWEIGHT[26]");
  CHECK(trg.spec.space->at(0).user_index == 26);
}

/* Pins: OPTIM FIXWEIGHT (14) with FIXINDEX given as separate values,
 * 'FIXINDEX 13 14' (the values '13' and '14', as `FIXINDEX 13 14` in a
 * configuration gives them) and FIXWEIGHT 14.0 2500, selects FIXWEIGHT
 * slot 2 (weight 14), labelled FIXWEIGHT[14] with user index 14 in the
 * search space, as 'FIXINDEX 13,14' does; 'FIXINDEX 14-13' is refused as a
 * descending range, and 'FIXINDEX 13,13,14' (two FIXWEIGHT values, one
 * per distinct weight) as weight 13 named twice, when the OPTIM line is
 * read.
 * Before the fix: HPOTargets read FIXINDEX with the lenient
 * core::parse_indices, which dropped '13 14' (the values joined into one
 * malformed segment) and expanded '14-13' to nothing, so both were refused
 * with the self-contradicting "OPTIM FIXWEIGHT (14): weight 14 is not
 * pinned by FIXINDEX (13 14)" ("(14-13)"); '13,13,14' was deduplicated to
 * 13, 14 and accepted ('13, 14' was read, its joined text a comma list).
 * After core::parse_fixindex first read FIXINDEX, '13,13,14' was refused
 * as "FIXINDEX 13,13,14 has 3 indices but FIXWEIGHT has 2 values" (this
 * case pinned that): it compared the count before it looked for a
 * repeat. */
TEST_CASE("OPTIM FIXWEIGHT maps through FIXINDEX given as separate values",
          "[HPO][optim][fixweight][fixindex]") {
  auto ctx_with = [](const std::string& fixindex, std::size_t n_values) {
    Context c;
    c.add("FIXINDEX", fixindex);
    const double v[3] = {14.0, 2500.0, 7.0};
    for (std::size_t k = 0; k < n_values; ++k) c.add("FIXWEIGHT", v[k]);
    return c;
  };
  const std::string line =
      "OPTIM FIXWEIGHT (14) 0.5 5000\n" + std::string(kKeyRulesOptimizer);
  for (const char* fi : {"13,14", "13 14", "13, 14"}) DYNAMIC_SECTION(
      "FIXINDEX " << fi) {
    Context ctx = ctx_with(fi, 2);
    KeyRulesTempFile tf(line);
    HPOTargets trg(ctx, tf.path.string());
    const auto it = trg.optimap.find("FIXWEIGHT");
    REQUIRE(it != trg.optimap.end());
    REQUIRE(it->second.opt.size() == 2);
    CHECK_FALSE(it->second.opt[0]);
    CHECK(it->second.opt[1]);
    REQUIRE(trg.spec.parameter_labels.size() == 1);
    CHECK(trg.spec.parameter_labels[0] == "FIXWEIGHT[14]");
    REQUIRE(trg.spec.space);
    CHECK(trg.spec.space->at(0).user_index == 14);
  }
  SECTION("FIXINDEX 14-13") {
    Context ctx = ctx_with("14-13", 2);
    KeyRulesTempFile tf(line);
    CHECK_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                      Catch::Contains("FIXINDEX: index '14-13' is a "
                                      "descending range") &&
                          !Catch::Contains("not pinned"));
  }
  SECTION("FIXINDEX 13,13,14") {
    Context ctx = ctx_with("13,13,14", 2);
    KeyRulesTempFile tf(line);
    CHECK_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                      Catch::Contains("FIXINDEX 13,13,14 names weight 13 "
                                      "twice"));
  }
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
  /* It ended "; set it in the -c potential/configuration." */
  CHECK(has(msg, "d, an integer, a discrete value rather than a continuous "
                 "parameter; the training configuration sets it."));
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

namespace {

/* A single-element (Ta) setfl file with cutoff 5 (100-point tables:
 * F(rho) = -sqrt(rho), rho(r) = (1 - r/5)^2, r phi(r) = r (1 - r/5)^3),
 * removed when the object goes. */
struct TaSetfl {
  fs::path path;
  TaSetfl()
  : path(fs::temp_directory_path() /
         ("tadah_optim_key_rules_" +
          std::to_string(std::chrono::steady_clock::now()
                             .time_since_epoch().count()) + ".eam.alloy")) {
    const int n = 100;
    const double rcut = 5.0, dr = rcut / (n - 1), drho = 50.0 / (n - 1);
    std::ofstream os(path);
    REQUIRE(os.good());
    os << "# synthetic Ta setfl\n# 2\n# 3\n1 Ta\n"
       << n << " " << drho << " " << n << " " << dr << " " << rcut << "\n"
       << "73 180.95 3.304 bcc\n" << std::setprecision(12);
    for (int i = 0; i < n; ++i) os << -std::sqrt(i * drho) << "\n";
    for (int i = 0; i < n; ++i) {
      const double x = 1.0 - i * dr / rcut;
      os << x * x << "\n";
    }
    for (int i = 0; i < n; ++i) {
      const double x = 1.0 - i * dr / rcut;
      os << i * dr * x * x * x << "\n";
    }
  }
  ~TaSetfl() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

/* A Ta model with the two-body term @p type2b (TYPE2B tokens), its
 * cutoffs @p rcut2b, @p rctype2b, and no many-body term, or (@p type2b
 * empty) the many-body term @p typemb with RCUTMB 5; SETFL @p setfl. */
Context eam_ctx(const std::string& setfl, const std::string& type2b,
                const std::string& rcut2b, const std::string& rctype2b,
                const std::string& typemb = "") {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("VERBOSE", 0);
  c.add("SETFL", setfl);
  if (!type2b.empty()) {
    c.add("INIT2B", "true");
    c.add("TYPE2B", type2b);
    c.add("RCUT2B", rcut2b);
    c.add("RCTYPE2B", rctype2b);
    c.add("INITMB", "false");
  } else {
    c.add("INIT2B", "false");
    c.add("INITMB", "true");
    c.add("TYPEMB", typemb);
    c.add("RCUTMB", 5.0);
    c.add("RCTYPEMB", "Cut_Cos");
  }
  c.add("MODEL", "M_BLR BF_Linear");
  c.add("DBFILE", "dummy.db");
  return c;
}

}  // namespace

/* Pins: OPTIM on a cutoff value that a descriptor takes from its setfl
 * file (D_Base::cutoff_fixed_by) is refused at startup, naming the OPTIM
 * key and index and the file's cutoff:
 *   - OPTIM RCUT2B (1) with TYPE2B D2_EAM: "OPTIM RCUT2B (1) is not a
 *     valid search target: D2_EAM takes the cutoff 5 of its setfl file
 *     '<path>' (SETFL) and refuses another value";
 *   - OPTIM RCUTMB (1) with TYPEMB DM_EAM: the same, naming DM_EAM;
 *   - TYPE2B D2_mJoin D2_BP ... D2_EAM ... (RCUT2B 4.5 5): OPTIM RCUT2B (2),
 *     D2_EAM's, is refused, and OPTIM RCUT2B (1), D2_BP's, is accepted.
 * Before the fix: all accepted; every candidate but the seed then failed in
 * the D2_EAM/DM_EAM constructor ("RCUT2B ... differs from the cutoff ...
 * of the setfl file"), FAILSCORE, and the run ended with exit status 0. */
TEST_CASE("OPTIM on a cutoff the setfl file fixes is refused at startup",
          "[HPO][optim][startup][eam]") {
  tadah::core::PeriodicTable::initialize();
  TaSetfl setfl;
  const std::string file = setfl.path.string();
  /* The message ends with what refusing another value does (it ended
   * "...; leave RCUT2B (1) out of OPTIM."). */
  const std::string fails = ", so every candidate but the value of the "
                            "training configuration would fail.";
  const std::string fixed2b =
      "D2_EAM takes the cutoff 5 of its setfl file '" + file +
      "' (SETFL) and refuses another value" + fails;
  {
    const std::string msg =
        startup_error(eam_ctx(file, "D2_EAM 0 Ta Ta", "5.0", "Cut_Cos"),
                      "OPTIM RCUT2B (1) 4.5 5.5");
    INFO(msg);
    CHECK(has(msg, "OPTIM RCUT2B (1) is not a valid search target: " +
                       fixed2b));
  }
  {
    const std::string msg = startup_error(
        eam_ctx(file, "", "", "", "DM_EAM 0 Ta Ta"),
        "OPTIM RCUTMB (1) 4.5 5.5");
    INFO(msg);
    CHECK(has(msg, "OPTIM RCUTMB (1) is not a valid search target: "
                   "DM_EAM takes the cutoff 5 of its setfl file '" + file +
                   "' (SETFL) and refuses another value" + fails));
  }
  {
    Context c = eam_ctx(file, "D2_mJoin D2_BP 2 2 Ta Ta D2_EAM 0 Ta Ta",
                        "4.5 5.0", "Cut_Cos Cut_Cos");
    c.add("CGRID2B", "2.6 3.3");
    c.add("SGRID2B", "1.0 1.0");
    const std::string msg = startup_error(c, "OPTIM RCUT2B (2) 4.5 5.5");
    INFO(msg);
    CHECK(has(msg, "OPTIM RCUT2B (2) is not a valid search target: " +
                       fixed2b));
    CHECK(startup_error(c, "OPTIM RCUT2B (1) 4.0 5.0").empty());
  }
}

/* Pins: OPTIM of a key the training configuration does not give is refused
 * at its line by HPOTargets, naming what OPTIM needs, and the Logger
 * writes no ERROR line about it:
 *   - OPTIM FIXWEIGHT without FIXINDEX and FIXWEIGHT: "OPTIM FIXWEIGHT
 *     requires FIXINDEX and FIXWEIGHT in the training configuration
 *     (FIXINDEX names the coefficients the FIXWEIGHT values hold, and OPTIM
 *     FIXWEIGHT searches those values).";
 *   - with FIXINDEX and without FIXWEIGHT: "OPTIM FIXWEIGHT requires
 *     FIXWEIGHT in the training configuration (the FIXWEIGHT values hold
 *     the coefficients FIXINDEX names, and OPTIM FIXWEIGHT searches those
 *     values).";
 *   - OPTIM CGRIDMB of a configuration without many-body keys: "OPTIM
 *     CGRIDMB: the training configuration has no CGRIDMB line, and OPTIM
 *     CGRIDMB searches the values the training configuration gives";
 *   - OPTIM of a word no key has (CGRIDMBX): "OPTIM CGRIDMBX: CGRIDMBX is
 *     not a key of the training configuration (closest key: CGRIDMB)".
 * Before the fix: the Logger wrote "ERROR: Context::size() -> Key not
 * found: FIXWEIGHT" (and "Key not found: CGRIDMB", "Did you mean:
 * CGRIDMB") to standard output, and the refusals were "Key not found in
 * the Context object: FIXWEIGHT", "... CGRIDMB" and "Key not allowed:
 * CGRIDMBX". */
TEST_CASE("OPTIM of a key the training configuration does not give names "
          "what OPTIM needs, with no Logger line",
          "[HPO][optim][optim_structure_keys]") {
  struct CoutCapture {
    std::ostringstream text;
    std::streambuf* old;
    CoutCapture() : old(std::cout.rdbuf(text.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(old); }
  };
  auto refusal = [](Context ctx, const std::string& line,
                    std::string& out) {
    KeyRulesTempFile tf(line + "\n" + kKeyRulesOptimizer);
    CoutCapture cap;
    std::string what;
    try {
      HPOTargets trg(ctx, tf.path.string());
    } catch (const std::exception& e) {
      what = e.what();
    }
    out = cap.text.str();
    return what;
  };
  Context base;
  base.add("WATOM", 1.0);
  struct Case {
    const char* fixindex;  // "" for none
    const char* line;
    const char* what;
  };
  const Case cases[] = {
      {"", "OPTIM FIXWEIGHT (1) 0 2",
       "OPTIM FIXWEIGHT requires FIXINDEX and FIXWEIGHT in the training "
       "configuration (FIXINDEX names the coefficients the FIXWEIGHT values "
       "hold, and OPTIM FIXWEIGHT searches those values)."},
      {"2", "OPTIM FIXWEIGHT (2) 0 2",
       "OPTIM FIXWEIGHT requires FIXWEIGHT in the training configuration (the "
       "FIXWEIGHT values hold the coefficients FIXINDEX names, and OPTIM "
       "FIXWEIGHT searches those values)."},
      {"", "OPTIM CGRIDMB (1) 0 2",
       "OPTIM CGRIDMB: the training configuration has no CGRIDMB line, and "
       "OPTIM CGRIDMB searches the values the training configuration gives"},
      {"", "OPTIM CGRIDMBX (1) 0 2",
       "OPTIM CGRIDMBX: CGRIDMBX is not a key of the training configuration "
       "(closest key: CGRIDMB)"},
  };
  for (const Case& c : cases) {
    INFO(c.line << " FIXINDEX '" << c.fixindex << "'");
    Context ctx = base;
    if (*c.fixindex) ctx.add("FIXINDEX", c.fixindex);
    std::string out;
    const std::string what = refusal(ctx, c.line, out);
    CHECK(what == std::string("Error on line 1: ") + c.what + "\nLine: " +
                      c.line);
    INFO("standard output: " << out);
    CHECK(out.find("Context::size()") == std::string::npos);
    CHECK(out.find("Did you mean") == std::string::npos);
  }
}

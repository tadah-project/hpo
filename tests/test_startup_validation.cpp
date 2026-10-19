// test_startup_validation.cpp
//
// Fail-fast rules enforced by validate_hpo_setup() before HPO_Host
// construction: FIXINDEX/FIXWEIGHT pairing and coverage, loss-source
// presence, validation-file requirements, the NORM restriction and the
// OPTIM EMBF0 rejection; plus derive_raw_embedding_f0 (EMBF0 journal).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/models/descriptors/basis_dispatch.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
using tadah::hpo::RawEmbeddingF0;
using tadah::hpo::derive_raw_embedding_f0;
using tadah::hpo::derive_weight_count;
using tadah::hpo::validate_hpo_setup;

struct ScopedTempFile {
  std::filesystem::path path;

  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_sv_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
    os.close();
  }

  ~ScopedTempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

constexpr const char* kOptimizerBlock =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

std::string target_with(const std::string& loss_lines) {
  return "OPTIM WATOM (1) 0.1 5.0\n" + loss_lines + kOptimizerBlock;
}

/* Minimal context: enough for HPOTargets + OPTIM WATOM, but no model
 * keys: the rules checked before the model is built (the loss sources)
 * can use it; validate_hpo_setup refuses it after them (the model of the
 * -c configuration must build). */
Context make_minimal_ctx() {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1.0);
  return c;
}

/* Full D2_BP + BF_Linear model context: derive_weight_count > 0. */
Context make_model_ctx() {
  Context c = make_minimal_ctx();
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "0.5 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

void pin_all_weights(Context& c, std::size_t K) {
  c.add("FIXINDEX", "1-" + std::to_string(K));
  for (std::size_t k = 0; k < K; ++k)
    c.add("FIXWEIGHT", 0.1 * static_cast<double>(k + 1));
}

}  // namespace

TEST_CASE("startup validation: FIXINDEX/FIXWEIGHT pairing",
          "[startup_validation]") {
  /* On a model context: without model keys the configuration is refused
   * before these rules (validate_hpo_setup builds the model). */
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));

  SECTION("lone FIXINDEX throws") {
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "2");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("requires FIXWEIGHT"));
  }
  SECTION("size mismatch throws") {
    /* FIXINDEX 1,2 for one FIXWEIGHT value on the 2-weight model. (The
     * section had '1,2,3', which also names weight 3 of the 2 weights:
     * core::parse_fixindex checks the bounds as it reads each entry, so
     * that is now refused as "index 3 is outside 1-2" first.) */
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "1,2");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("pair 1:1"));
  }
}

TEST_CASE("startup validation: K-dependent rules on a real model context",
          "[startup_validation]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));

  SECTION("out-of-bounds FIXINDEX throws") {
    /* core::parse_fixindex's message, the one every site gives (before
     * the fix: "FIXINDEX entry 99 is out of range: this model has 2
     * weights (valid range 1..2).", which this section checked for "out
     * of range"). */
    Context ctx = make_model_ctx();
    const std::size_t K = derive_weight_count(ctx);
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "99");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        "FIXINDEX: index 99 is outside 1-" +
                            std::to_string(K) + " (the model has " +
                            std::to_string(K) + " weights)");
  }
  SECTION("no training DBFILE with partial coverage throws") {
    Context ctx = make_model_ctx();
    ctx.add("FIXINDEX", "1");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                        Catch::Contains("have no training data"));
  }
  SECTION("no training DBFILE and no pinning throws") {
    Context ctx = make_model_ctx();
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                        Catch::Contains("cannot be determined"));
  }
  SECTION("NORM true with FIXINDEX throws") {
    Context ctx = make_model_ctx();
    const std::size_t K = derive_weight_count(ctx);
    REQUIRE(K > 0);
    pin_all_weights(ctx, K);
    ctx.remove("NORM");
    ctx.add("NORM", "true");
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("NORM"));
  }
}

/* Pins: validate_hpo_setup reads FIXINDEX with core::parse_fixindex, as
 * the trainers do, on the D2_BP model (2 weights) with a training DBFILE:
 *  - FIXINDEX given as separate values, '1 2' (FIXWEIGHT 0.5 0.6), passes,
 *    as '1,2' does;
 *  - '1,1,2' is refused as weight 1 named twice, with 2 FIXWEIGHT values
 *    (one per distinct weight) as with 3;
 *  - '2-1' is refused as a descending range.
 * Before the fix: the lenient core::parse_indices dropped '1 2' and
 * expanded '2-1' to nothing ("FIXINDEX names 0 weights but FIXWEIGHT
 * provides 2 values"), and deduplicated '1,1,2' to weights 1, 2: with 2
 * values the setup passed (the run then trained with it), with 3 it was
 * "FIXINDEX names 2 weights but FIXWEIGHT provides 3 values". After
 * core::parse_fixindex first read FIXINDEX, '1,1,2' with 2 values was
 * refused as "FIXINDEX 1,1,2 has 3 indices but FIXWEIGHT has 2 values"
 * (this case pinned that): it compared the count before it looked for a
 * repeat. */
TEST_CASE("startup validation: FIXINDEX is read strictly, separate values as "
          "one list",
          "[startup_validation][fixindex]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  auto ctx_with = [](const std::string& fixindex,
                     const std::vector<double>& fixweight) {
    Context c = make_model_ctx();
    c.add("DBFILE", "dummy.db");
    c.add("FIXINDEX", fixindex);
    for (double v : fixweight) c.add("FIXWEIGHT", v);
    return c;
  };
  REQUIRE(derive_weight_count(make_model_ctx()) == 2);
  SECTION("separate values pass, as the comma list does") {
    for (const char* fi : {"1,2", "1 2"}) {
      INFO("FIXINDEX " << fi);
      Context ctx = ctx_with(fi, {0.5, 0.6});
      HPOTargets trg(ctx, tf.path.string());
      CHECK_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    }
  }
  SECTION("a weight named twice is refused") {
    Context two = ctx_with("1,1,2", {0.5, 0.6});
    HPOTargets trg2(two, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(two, trg2, true, true),
                      Catch::Contains("FIXINDEX 1,1,2 names weight 1 twice"));
    Context three = ctx_with("1,1,2", {0.5, 0.6, 0.7});
    HPOTargets trg3(three, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(three, trg3, true, true),
                      Catch::Contains("FIXINDEX 1,1,2 names weight 1 twice"));
  }
  SECTION("a descending range is refused") {
    Context ctx = ctx_with("2-1", {0.5, 0.6});
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("FIXINDEX: index '2-1' is a descending "
                                      "range"));
  }
}

/* Pins: validate_hpo_setup refuses FIXINDEX/FIXWEIGHT with M_KRR
 * implementation 2 (MODEL M_KRR Kern_Linear 2 on the D2_BP model, a
 * training DBFILE, FIXINDEX 1 / FIXWEIGHT 0.5), whose solve cannot pin
 * weights: "MODEL M_KRR Kern_Linear 2 cannot pin weights", at startup;
 * the same configuration with implementation 1 passes.
 * Before the fix: the setup passed (classify_weight_source took it for
 * implementation 1), and every evaluation trained with the pin ignored. */
TEST_CASE("startup validation: M_KRR implementation 2 with FIXINDEX is "
          "refused",
          "[startup_validation][fixindex][krr]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  for (const char* impl : {"1", "2"}) DYNAMIC_SECTION("implementation " << impl) {
    Context ctx = make_model_ctx();
    ctx.remove("MODEL");
    ctx.add("MODEL", "M_KRR");
    ctx.add("MODEL", "Kern_Linear");
    ctx.add("MODEL", impl);
    ctx.add("SBASIS", 2);
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "1");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    if (std::string(impl) == "1")
      CHECK_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    else
      CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::StartsWith("MODEL M_KRR Kern_Linear 2 cannot "
                                          "pin weights"));
  }
}

/* Pins: a -c configuration whose model has no weights (INIT2B true with a
 * D2_Dummy two-body descriptor, BIAS false: descriptor size 0) is refused
 * at startup as a model that does not build ("The descriptor size is 0"),
 * before any FIXINDEX rule, with or without FIXINDEX/FIXWEIGHT: the weight
 * count K the FIXINDEX rules use is never 0.
 * Before the fix: passes as well (a pin: probe_model refuses the model
 * first; validate_hpo_setup's branch for K = 0, which checked FIXINDEX
 * without the weight count, was unreachable and is removed). */
TEST_CASE("startup validation: a model without weights is refused before the "
          "FIXINDEX rules",
          "[startup_validation][fixindex][model_probe]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  auto no_weights = [] {
    Context c = make_model_ctx();
    c.remove("TYPE2B");
    c.add("TYPE2B", "D2_Dummy H H");
    c.add("DBFILE", "dummy.db");
    return c;
  };
  CHECK_THROWS_WITH(derive_weight_count(no_weights()),
                    Catch::Contains("The descriptor size is 0"));
  for (const bool pinned : {false, true}) {
    INFO("FIXINDEX/FIXWEIGHT: " << pinned);
    Context ctx = no_weights();
    if (pinned) {
      ctx.add("FIXINDEX", "1 2");
      ctx.add("FIXWEIGHT", "0.5 0.6");
    }
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("The descriptor size is 0"));
  }
}

/* Pins: for M_KRR with a non-linear kernel (Kern_RBF, SBASIS 4) the
 * weight count K that startup checks FIXINDEX/FIXWEIGHT against is SBASIS
 * (derive_weight_count == 4), the number of basis vectors training draws,
 * one weight each: FIXINDEX 5 is refused ("the model has 4 weights"),
 * FIXINDEX 4 passes; without a training DBFILE the configuration is
 * refused even with every weight pinned (the basis is drawn from the
 * training data: "draws its SBASIS basis vectors from the training
 * data"); SBASIS 1 is refused at startup, naming SBASIS.
 * Before the fix: K was the kernel's design-matrix width before training,
 * 0 (no basis yet), so derive_weight_count returned 0 and only the
 * pairing of FIXINDEX and FIXWEIGHT was checked: FIXINDEX 5 passed
 * startup (and failed every evaluation), as did the configuration without
 * a DBFILE and the one with SBASIS 1. Before fix/fixindex-strict the
 * refusal said "this model has 4 weights" (MLIP's own bound message; the
 * bound is now core::parse_fixindex's, "the model has 4 weights"). */
TEST_CASE("startup validation: FIXINDEX of a non-linear M_KRR kernel is "
          "checked against SBASIS",
          "[startup_validation][nonlinear]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  auto krr_ctx = [] {
    Context c = make_model_ctx();
    c.remove("MODEL");
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_RBF");
    c.add("MODEL", 1);
    c.add("MPARAMS", 1.0);
    c.add("SBASIS", 4);
    return c;
  };
  SECTION("the weight count is SBASIS") {
    CHECK(derive_weight_count(krr_ctx()) == 4);
  }
  SECTION("an index past SBASIS is refused") {
    Context ctx = krr_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "5");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("the model has 4 weights"));
  }
  SECTION("an index within SBASIS passes") {
    Context ctx = krr_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "4");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("no training DBFILE is refused, every weight pinned") {
    Context ctx = krr_ctx();
    pin_all_weights(ctx, 4);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                      Catch::Contains("draws its SBASIS basis vectors from "
                                      "the training data"));
  }
  SECTION("SBASIS 1 is refused") {
    Context ctx = krr_ctx();
    ctx.remove("SBASIS");
    ctx.add("SBASIS", 1);
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("SBASIS 1"));
  }
}

/* Pins: M_KRR Kern_Linear implementation 2 draws SBASIS basis vectors for
 * its solve, so startup checks its SBASIS as a non-linear kernel's: with a
 * training DBFILE, SBASIS 1 is refused naming "does not build a model",
 * "M_KRR Kern_Linear 2", "SBASIS 1" and "at least 2 basis vectors", without
 * the word "non-linear"; a configuration without SBASIS is refused naming
 * "M_KRR Kern_Linear 2 needs SBASIS"; SBASIS 3 passes, and its weight count
 * is DSIZE, 2 (derive_weight_count: the descriptor weights of the basis
 * vectors, not SBASIS; a pin).
 * Before the fix: SBASIS 1 and the configuration without SBASIS passed
 * startup (the count, DSIZE, was returned before any SBASIS check), and
 * every evaluation threw ("SBASIS 1: a non-linear M_KRR model needs at
 * least 2 basis vectors"; "Key not found in the Context object: SBASIS"). */
TEST_CASE("startup validation: SBASIS of M_KRR Kern_Linear 2 is checked",
          "[startup_validation][krr2]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  auto krr2_ctx = [](int sbasis) {
    Context c = make_model_ctx();
    c.remove("MODEL");
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_Linear");
    c.add("MODEL", 2);
    if (sbasis > 0) c.add("SBASIS", sbasis);
    c.add("DBFILE", "dummy.db");
    return c;
  };
  SECTION("SBASIS 1 is refused") {
    Context ctx = krr2_ctx(1);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("M_KRR Kern_Linear 2") &&
                          Catch::Contains("SBASIS 1") &&
                          Catch::Contains("at least 2 basis vectors") &&
                          !Catch::Contains("non-linear"));
  }
  SECTION("no SBASIS is refused") {
    Context ctx = krr2_ctx(0);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("M_KRR Kern_Linear 2 needs SBASIS"));
  }
  SECTION("SBASIS 3 passes; the weight count is DSIZE") {
    Context ctx = krr2_ctx(3);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    CHECK(derive_weight_count(krr2_ctx(3)) == 2);
  }
}

TEST_CASE("startup validation: a -c configuration that does not build a "
          "model is refused", "[startup_validation][model_probe]") {
  /* A construction error of the model (a DIMER potential with element-
   * named TYPE2B pairs: DIMER evaluates every atom as Z=1) failed every
   * evaluation. With FIXINDEX the weight-count probe met it and printed a
   * WARNING ("FIXINDEX bounds/coverage checks are deferred to the first
   * training evaluation"); without, startup said nothing. The run then
   * ended "no evaluation produced a usable loss". */
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  auto dimer_ctx = [] {
    Context c = make_model_ctx();
    c.remove("DIMER");
    c.add("DIMER", "true 1.104 true");
    c.add("DBFILE", "dummy.db");
    return c;
  };
  SECTION("with FIXINDEX/FIXWEIGHT (the weight-count probe)") {
    Context ctx = dimer_ctx();
    ctx.add("FIXINDEX", "1");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("DIMER evaluates every atom as Z=1") &&
                          Catch::Contains("Every evaluation would fail"));
  }
  SECTION("without them") {
    Context ctx = dimer_ctx();
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("DIMER evaluates every atom as Z=1"));
  }
  SECTION("derive_weight_count reports the error instead of 0") {
    CHECK_THROWS_WITH(derive_weight_count(dimer_ctx()),
                      Catch::Contains("DIMER evaluates every atom as Z=1"));
  }
}

/* Pins: the startup refusal of a -c configuration whose model does not
 * build ends the model's reason with one full stop before "Every
 * evaluation would fail": a reason that ends in '.' (M_KRR Kern_RBF with
 * SBASIS 1: "... Set SBASIS to 2 or more.") is followed by " Every
 * evaluation would fail" with no ".." anywhere in the message; a reason
 * that ends in ')' (DIMER with element-named TYPE2B pairs: "... does not
 * apply to every element pair)") gets its '.': "element pair). Every
 * evaluation would fail", no "..".
 * Before the fix: the SBASIS refusal read "Set SBASIS to 2 or more.. Every
 * evaluation would fail" (". " was appended to every reason); the DIMER
 * one passed as well (a pin). */
TEST_CASE("startup validation: a model refusal ends its reason with one "
          "full stop", "[startup_validation][model_probe]") {
  PeriodicTable::initialize();
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
  SECTION("a reason that ends in a full stop") {
    Context ctx = make_model_ctx();
    ctx.remove("MODEL");
    ctx.add("MODEL", "M_KRR");
    ctx.add("MODEL", "Kern_RBF");
    ctx.add("MODEL", 1);
    ctx.add("MPARAMS", 1.0);
    ctx.add("SBASIS", 1);
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not build a model") &&
                          Catch::Contains("Set SBASIS to 2 or more. Every "
                                          "evaluation would fail") &&
                          !Catch::Contains(".."));
  }
  SECTION("a reason that ends in a parenthesis") {
    Context ctx = make_model_ctx();
    ctx.remove("DIMER");
    ctx.add("DIMER", "true 1.104 true");
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("does not apply to every element "
                                      "pair). Every evaluation would fail") &&
                          !Catch::Contains(".."));
  }
}

TEST_CASE("startup validation: loss-source rules", "[startup_validation]") {
  SECTION("no loss source at all throws") {
    ScopedTempFile tf(target_with(""));
    Context ctx = make_minimal_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("No loss source"));
  }
  SECTION("validation metrics without --validation throw") {
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    Context ctx = make_minimal_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, false),
                        Catch::Contains("--validation"));
  }
}

TEST_CASE("startup validation: valid setups pass", "[startup_validation]") {
  PeriodicTable::initialize();

  SECTION("DBFILE + validation metrics") {
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("no DBFILE, all weights pinned, observable-only loss") {
    /* GRAD_STRATEGY HYBRID: an unset one resolves to ANALYTICAL with
     * PC_OBSERVABLE, which needs training data (see the ANALYTICAL rules
     * below). */
    ScopedTempFile tf(
        "OPTIM WATOM (1) 0.1 5.0\n"
        "PC_OBSERVABLE cohesive_energy --species H --target 8.1 "
        "--weight 1.0 --lattice bcc --hotvar a0 2.5 4.5 3.3\n"
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nGRAD_STRATEGY HYBRID\n"
        "ENDOPTIMIZER\n");
    Context ctx = make_model_ctx();
    const std::size_t K = derive_weight_count(ctx);
    REQUIRE(K > 0);
    pin_all_weights(ctx, K);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, false));
  }
  SECTION("no DBFILE, all weights pinned, validation metrics") {
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    Context ctx = make_model_ctx();
    const std::size_t K = derive_weight_count(ctx);
    REQUIRE(K > 0);
    pin_all_weights(ctx, K);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
  }
}

/* GRAD_STRATEGY ANALYTICAL -- written out, or resolved from an unset
 * GRAD_STRATEGY in a PC_OBSERVABLE run -- differentiates the least-squares
 * fit analytically. That needs training data with every weight fitted, the
 * Cholesky normal equations (OALGO 4) with a fixed ridge LAMBDA >= 0, raw
 * features (NORM false), a linear basis (BF_Linear or Kern_Linear) and no
 * stress rows (STRESS false, no PC_SRMSE). PC_FRMSE does not need FORCE
 * true: FORCE decides only what training fits, and the validation force
 * rows of the gradient are formed whenever PC_FRMSE is set. A setup
 * without the settings used to pass startup and stop at the first gradient
 * with "no registered analytical derivative". HYBRID falls back to finite
 * differences and is not affected. */
TEST_CASE("startup validation: GRAD_STRATEGY ANALYTICAL needs the settings "
          "of the analytical gradient", "[startup_validation][grad_strategy]") {
  PeriodicTable::initialize();
  const auto target = [](const std::string& grad, const std::string& loss) {
    return "OPTIM WATOM (1) 0.1 5.0\n" + loss +
           "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n" + grad +
           "ENDOPTIMIZER\n";
  };
  const std::string analytical = "GRAD_STRATEGY ANALYTICAL\n";
  const std::string ermse = "PC_ERMSE 0 1\n";
  /* A model context that satisfies every rule; each section breaks one. */
  const auto good_ctx = [] {
    Context c = make_model_ctx();
    c.add("DBFILE", "dummy.db");
    c.remove("OALGO");
    c.add("OALGO", 4);
    c.remove("LAMBDA");
    c.add("LAMBDA", 1e-8);
    return c;
  };
  const auto set = [](Context& c, const std::string& key,
                      const std::string& value) {
    c.remove(key);
    c.add(key, value);
  };
  const auto rejected = [&](Context& ctx, const std::string& body,
                            bool has_train_db = true) {
    ScopedTempFile tf(body);
    HPOTargets trg(ctx, tf.path.string());
    std::string what;
    try {
      validate_hpo_setup(ctx, trg, has_train_db, true);
    } catch (const std::exception& e) {
      what = e.what();
    }
    return what;
  };

  SECTION("a setup with the settings passes") {
    Context ctx = good_ctx();
    CHECK(rejected(ctx, target(analytical, ermse)) == "");
  }
  SECTION("OALGO other than 4") {
    Context ctx = good_ctx();
    set(ctx, "OALGO", "1");
    const auto what = rejected(ctx, target(analytical, ermse));
    CHECK_THAT(what, Catch::Contains("GRAD_STRATEGY ANALYTICAL") &&
                     Catch::Contains("OALGO 4") &&
                     Catch::Contains("GRAD_STRATEGY HYBRID"));
  }
  SECTION("LAMBDA -1 (evidence approximation) or negative") {
    for (const char* lambda : {"-1", "-0.5"}) {
      Context ctx = good_ctx();
      set(ctx, "LAMBDA", lambda);
      INFO("LAMBDA " << lambda);
      CHECK_THAT(rejected(ctx, target(analytical, ermse)),
                 Catch::Contains("LAMBDA"));
    }
  }
  /* NORM true with BF_Linear needs BIAS true (a startup rule of its own,
   * mlip::check_bias_constraints, as in tadah train), so the setup has BIAS
   * true: the refusal is the analytical gradient's. Pins (passes before and
   * after that rule): the reason is "NORM is true". Before the rule the
   * setup had BIAS false. */
  SECTION("NORM true") {
    Context ctx = good_ctx();
    set(ctx, "NORM", "true");
    set(ctx, "BIAS", "true");
    CHECK_THAT(rejected(ctx, target(analytical, ermse)),
               Catch::Contains("GRAD_STRATEGY ANALYTICAL") &&
               Catch::Contains("NORM is true"));
  }
  SECTION("STRESS true") {
    Context ctx = good_ctx();
    set(ctx, "STRESS", "true");
    CHECK_THAT(rejected(ctx, target(analytical, ermse)),
               Catch::Contains("STRESS"));
  }
  SECTION("a non-linear basis") {
    Context ctx = good_ctx();
    ctx.remove("MODEL");
    ctx.add("MODEL", "M_BLR");
    ctx.add("MODEL", "BF_Polynomial2");
    CHECK_THAT(rejected(ctx, target(analytical, ermse)),
               Catch::Contains("BF_Polynomial2") &&
               Catch::Contains("BF_Linear"));
  }
  SECTION("PC_SRMSE") {
    Context ctx = good_ctx();
    CHECK_THAT(rejected(ctx, target(analytical, "PC_SRMSE 0 1\n")),
               Catch::Contains("PC_SRMSE"));
  }
  /* A behaviour change, not a weakening: this SECTION used to assert that
   * ANALYTICAL + PC_FRMSE + FORCE false is refused with "PC_FRMSE needs
   * FORCE true". FORCE now decides only what training fits; validation
   * predicts forces whenever PC_FRMSE is set, and the analytical gradient
   * forms its validation force rows from PC_FRMSE alone (its parity with
   * central differences is pinned in test_hpo_analytical.cpp). So the setup
   * is accepted with FORCE false, with FORCE true, and with no FORCE key
   * (the default, false). Before the fix the FORCE false and the
   * absent-key runs were refused by analytical_gradient_blocker ("PC_FRMSE
   * needs FORCE true"). */
  SECTION("PC_FRMSE is accepted whatever FORCE is") {
    for (const char* force : {"false", "true", ""}) {
      Context ctx = good_ctx();
      if (*force == '\0')
        ctx.remove("FORCE");
      else
        set(ctx, "FORCE", force);
      INFO("FORCE " << (*force == '\0' ? "(absent)" : force));
      CHECK(rejected(ctx, target(analytical, "PC_FRMSE 0 1\n")) == "");
      CHECK(rejected(ctx, target(analytical, ermse + "PC_FRMSE 0 1\n")) ==
            "");
    }
  }
  SECTION("FIXINDEX") {
    Context ctx = good_ctx();
    ctx.add("FIXINDEX", "1");
    ctx.add("FIXWEIGHT", 0.5);
    CHECK_THAT(rejected(ctx, target(analytical, ermse)),
               Catch::Contains("FIXINDEX"));
  }
  SECTION("no training DBFILE") {
    Context ctx = good_ctx();
    ctx.remove("DBFILE");
    const std::size_t K = derive_weight_count(ctx);
    REQUIRE(K > 0);
    pin_all_weights(ctx, K);
    CHECK_THAT(rejected(ctx, target(analytical, ermse), false),
               Catch::Contains("training DBFILE"));
  }
  /* Pins: a setup without a training DBFILE whose FIXINDEX and FIXWEIGHT
   * pin every weight -- the only one startup takes without a DBFILE
   * (classify_weight_source refuses the others, ending with the potential
   * advice) -- is refused under ANALYTICAL for "there is no training
   * DBFILE", advising GRAD_STRATEGY HYBRID, and its message has no
   * potential advice ("A potential has no DBFILE line", and not its
   * wording before the training record, "A potential records no
   * DBFILE"): with the DBFILE line that advice would add, the same setup
   * is refused for "FIXINDEX pins model weights", so a DBFILE line does
   * not make the analytical gradient formable there; under HYBRID it
   * passes.
   * Before the fix: passes as well (a pin of why this no-DBFILE refusal,
   * unlike the others of tadah hpo, does not end with the advice; the
   * check of the advice's wording since the training record is added
   * beside the old one). */
  SECTION("no training DBFILE with every weight pinned: HYBRID, not a "
          "DBFILE line") {
    Context pinned = good_ctx();
    pinned.remove("DBFILE");
    const std::size_t K = derive_weight_count(pinned);
    REQUIRE(K > 0);
    pin_all_weights(pinned, K);
    const std::string what =
        rejected(pinned, target(analytical, ermse), false);
    CHECK_THAT(what, Catch::Contains("there is no training DBFILE") &&
                     Catch::Contains("GRAD_STRATEGY HYBRID"));
    CHECK_THAT(what, !Catch::Contains("A potential records no DBFILE"));
    CHECK_THAT(what, !Catch::Contains("A potential has no DBFILE line"));
    Context with_db = pinned;
    with_db.add("DBFILE", "dummy.db");
    CHECK_THAT(rejected(with_db, target(analytical, ermse)),
               Catch::Contains("FIXINDEX pins model weights"));
    CHECK(rejected(pinned, target("GRAD_STRATEGY HYBRID\n", ermse), false) ==
          "");
  }
  /* BIAS true for NORM true, as above. Pins (passes before and after the
   * BIAS rule); before it the setup had BIAS false. */
  SECTION("HYBRID is not affected") {
    Context ctx = good_ctx();
    set(ctx, "OALGO", "1");
    set(ctx, "NORM", "true");
    set(ctx, "BIAS", "true");
    CHECK(rejected(ctx, target("GRAD_STRATEGY HYBRID\n", ermse)) == "");
  }
  SECTION("an unset GRAD_STRATEGY resolved to ANALYTICAL for PC_OBSERVABLE") {
    const std::string observable =
        "PC_OBSERVABLE cohesive_energy --species H --target 8.1 "
        "--weight 1.0 --lattice bcc --hotvar a0 2.5 4.5 3.3\n";
    Context ctx = good_ctx();
    CHECK(rejected(ctx, target("", observable)) == "");
    set(ctx, "OALGO", "1");
    CHECK_THAT(rejected(ctx, target("", observable)),
               Catch::Contains("resolved for PC_OBSERVABLE") &&
               Catch::Contains("OALGO 4"));
    /* No training data: all weights pinned, observable-only loss. */
    Context pinned = good_ctx();
    pinned.remove("DBFILE");
    const std::size_t K = derive_weight_count(pinned);
    REQUIRE(K > 0);
    pin_all_weights(pinned, K);
    CHECK_THAT(rejected(pinned, target("", observable), false),
               Catch::Contains("resolved for PC_OBSERVABLE") &&
               Catch::Contains("training DBFILE"));
  }
}

namespace {

/* A model context that satisfies every setting the analytical gradient
 * needs (analytical_gradient_blocker): training DBFILE, OALGO 4, a fixed
 * LAMBDA >= 0, NORM/STRESS false, BF_Linear. */
Context analytical_ready_ctx() {
  Context c = make_model_ctx();
  c.add("DBFILE", "dummy.db");
  c.remove("OALGO");
  c.add("OALGO", 4);
  c.remove("LAMBDA");
  c.add("LAMBDA", 1e-8);
  return c;
}

/* The message validate_hpo_setup stops with, or "" when it passes. */
/* The 1-based number of the first line of `config` that reads `keyword`
 * (leading blanks aside); 0 when none does. */
std::size_t line_of(const std::string& config, const std::string& keyword) {
  std::istringstream in(config);
  std::string text;
  for (std::size_t n = 1; std::getline(in, text); ++n) {
    const auto p = text.find_first_not_of(' ');
    if (p != std::string::npos && text.substr(p) == keyword) return n;
  }
  return 0;
}

std::string startup_error(Context& ctx, const std::string& body) {
  ScopedTempFile tf(body);
  HPOTargets trg(ctx, tf.path.string());
  try {
    validate_hpo_setup(ctx, trg, true, true);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

const std::string kObservable =
    "PC_OBSERVABLE cohesive_energy --species H --target 8.1 --weight 1.0 "
    "--lattice bcc --hotvar a0 2.5 4.5 3.3";

}  // namespace

/* Loss terms whose value has no analytical derivative -- at every
 * evaluation (a PC_OBSERVABLE --loss MAPE or RMSLE) or at some (an
 * unconverged relaxation under --unconverged force-scaled) -- stop an
 * ANALYTICAL run at the first such gradient. Startup refuses ANALYTICAL
 * for them, written out or resolved for PC_OBSERVABLE, naming the line;
 * HYBRID fills them with finite differences. They used to pass startup
 * and stop the run after its first evaluation with "no registered
 * analytical derivative ... dtheta_registry". */
TEST_CASE("startup validation: GRAD_STRATEGY ANALYTICAL needs loss terms "
          "with an analytical derivative",
          "[startup_validation][grad_strategy]") {
  PeriodicTable::initialize();
  const auto target = [](const std::string& grad, const std::string& loss) {
    return "OPTIM WATOM (1) 0.1 5.0\n" + loss +
           "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n" + grad +
           "ENDOPTIMIZER\n";
  };
  const std::string analytical = "GRAD_STRATEGY ANALYTICAL\n";
  Context ctx = analytical_ready_ctx();

  SECTION("--loss MAPE and RMSLE have no derivative with respect to the "
          "value") {
    for (const char* loss : {"MAPE", "RMSLE"}) {
      INFO("--loss " << loss);
      const std::string line = kObservable + " --loss " + loss + "\n";
      for (const std::string& grad : {analytical, std::string()}) {
        INFO("GRAD_STRATEGY line: '" << grad << "'");
        CHECK_THAT(startup_error(ctx, target(grad, line)),
                   Catch::Contains("GRAD_STRATEGY ANALYTICAL") &&
                   Catch::Contains("cohesive_energy[1]") &&
                   Catch::Contains(std::string("--loss ") + loss) &&
                   Catch::Contains("GRAD_STRATEGY HYBRID"));
      }
      CHECK(startup_error(ctx, target("GRAD_STRATEGY HYBRID\n", line)) == "");
    }
    /* The residual losses have one. */
    for (const char* loss : {"L1", "L2", "HUBER", "TUKEY", "LOG_COSH"}) {
      INFO("--loss " << loss);
      CHECK(startup_error(ctx, target(analytical, kObservable + " --loss " +
                                                      loss + "\n")) == "");
    }
  }
  SECTION("--unconverged force-scaled has none at an unconverged "
          "relaxation") {
    const std::string line = kObservable + " --unconverged force-scaled 1.0\n";
    const auto what = startup_error(ctx, target("", line));
    CHECK_THAT(what, Catch::Contains("resolved for PC_OBSERVABLE") &&
                     Catch::Contains("cohesive_energy[1]") &&
                     Catch::Contains("--unconverged force-scaled") &&
                     Catch::Contains("GRAD_STRATEGY HYBRID"));
    CHECK_THAT(startup_error(ctx, target(analytical, line)),
               Catch::Contains("--unconverged force-scaled"));
    CHECK(startup_error(ctx, target("GRAD_STRATEGY HYBRID\n", line)) == "");
    /* The constant penalty contributes a zero gradient. */
    CHECK(startup_error(ctx, target(analytical, kObservable +
                                    " --unconverged const 100\n")) == "");
  }
  SECTION("the message names the PC_OBSERVABLE line by its 1-based index") {
    const std::string two = kObservable + "\n" + kObservable + " --loss MAPE\n";
    CHECK_THAT(startup_error(ctx, target(analytical, two)),
               Catch::Contains("cohesive_energy[2]") &&
               !Catch::Contains("cohesive_energy[1]"));
  }
}

/* NLopt's MLSL and AUGLAG run their INNER search inside their own run and
 * hand it the gradient of their objective, which the enclosing block's
 * GRAD_STRATEGY obtains; an INNER block takes no GRAD_STRATEGY (a parse
 * error at its line). The ANALYTICAL rule therefore checks the blocks that
 * run an optimiser -- the STRATEGY-level one and EXPLORE -- and not an
 * INNER block. A setup the analytical gradient does not cover (OALGO 1,
 * the default) with HYBRID on the enclosing block and nothing in INNER
 * used to be refused on the INNER block, whose resolved ANALYTICAL was
 * never used.
 * The refusal names the line that closes the block it refuses, as the
 * parser's refusals do: "Error on line N: OPTIMIZER block: GRAD_STRATEGY
 * ANALYTICAL ...\nLine: ENDOPTIMIZER" ("EXPLORE block: ..." at
 * ENDEXPLORE). Before that fix the pins below read the message alone,
 * "OPTIMIZER block: GRAD_STRATEGY ANALYTICAL (resolved ...", which the
 * refusal was (no line). */
TEST_CASE("startup validation: the ANALYTICAL rule applies to the block "
          "whose gradient an INNER search uses",
          "[startup_validation][grad_strategy]") {
  PeriodicTable::initialize();
  Context ctx = analytical_ready_ctx();
  ctx.remove("OALGO");
  ctx.add("OALGO", 1);  /* not covered by the analytical gradient */
  const std::string hybrid = "    GRAD_STRATEGY HYBRID\n";
  const auto mlsl = [](const std::string& outer_extra,
                       const std::string& inner_extra) {
    return "OPTIM WATOM (1) 0.1 5.0\n" + kObservable + "\n"
           "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
           "    ALGO GN_MLSL_LDS\n    MAXEVAL 5\n" + outer_extra +
           "    INNER\n      LIB NLOPT\n      ALGO LD_LBFGS\n" + inner_extra +
           "    ENDINNER\n  ENDSTRATEGY\nENDOPTIMIZER\n";
  };

  SECTION("HYBRID on the enclosing block serves the INNER search") {
    CHECK(startup_error(ctx, mlsl(hybrid, "")) == "");
  }
  SECTION("an unset enclosing block resolves to ANALYTICAL and is "
          "refused, naming it") {
    const std::string config = mlsl("", "");
    const auto neither = startup_error(ctx, config);
    CHECK_THAT(neither, Catch::StartsWith(
                            "Error on line " +
                            std::to_string(line_of(config, "ENDOPTIMIZER")) +
                            ": OPTIMIZER block: GRAD_STRATEGY ANALYTICAL "
                            "(resolved") &&
                        Catch::EndsWith("\nLine: ENDOPTIMIZER") &&
                        Catch::Contains("OALGO 4") &&
                        !Catch::Contains("INNER"));
  }
  SECTION("a GRAD_STRATEGY in INNER stops the parse before startup "
          "validation") {
    for (const std::string& outer : {std::string(), hybrid}) {
      INFO(outer);
      ScopedTempFile tf(mlsl(outer, "  " + hybrid));
      CHECK_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                        Catch::Contains("GRAD_STRATEGY is not accepted "
                                        "inside an INNER block"));
    }
  }
  SECTION("EXPLORE is checked like the STRATEGY-level block, its INNER like "
          "an INNER block") {
    const auto prestage = [](const std::string& explore) {
      return "OPTIM WATOM (1) 0.1 5.0\n" + kObservable + "\n"
             "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB DLIB\n"
             "    ALGO BFGS\n    MAXEVAL 5\n    GRAD_STRATEGY HYBRID\n"
             "    EXPLORE\n" + explore + "    ENDEXPLORE\n  ENDSTRATEGY\n"
             "ENDOPTIMIZER\n";
    };
    const std::string explore_mlsl =
        "      LIB NLOPT\n      ALGO GN_MLSL_LDS\n      MAXEVAL 5\n";
    const std::string explore_inner =
        "      INNER\n        LIB NLOPT\n        ALGO LD_LBFGS\n"
        "      ENDINNER\n";
    /* "Error on line N: " for the ENDEXPLORE line N of `config`. */
    const auto at_endexplore = [](const std::string& config) {
      return "Error on line " + std::to_string(line_of(config, "ENDEXPLORE")) +
             ": ";
    };
    const std::string with_inner = prestage(explore_mlsl + explore_inner);
    CHECK_THAT(startup_error(ctx, with_inner),
               Catch::StartsWith(at_endexplore(with_inner) +
                                 "EXPLORE block: GRAD_STRATEGY ANALYTICAL "
                                 "(resolved") &&
               Catch::EndsWith("\nLine:     ENDEXPLORE"));
    CHECK(startup_error(ctx, prestage(explore_mlsl +
                                      "      GRAD_STRATEGY HYBRID\n" +
                                      explore_inner)) == "");
    const std::string lbfgs = prestage(
        "      LIB NLOPT\n      ALGO LD_LBFGS\n      MAXEVAL 5\n");
    CHECK_THAT(startup_error(ctx, lbfgs),
               Catch::StartsWith(at_endexplore(lbfgs) +
                                 "EXPLORE block: GRAD_STRATEGY ANALYTICAL") &&
               Catch::EndsWith("\nLine:     ENDEXPLORE"));
  }
}

/* FAILSCORE_GRADING ON bands the failure losses below FAILSCORE, so it
 * needs a finite FAILSCORE. The rule used to live in hpo_run() alone, so a
 * setup checked with validate_hpo_setup() -- the target-file parse cases
 * (test_ta_target_files.cpp), any embedding -- passed it. */
TEST_CASE("startup validation: FAILSCORE_GRADING ON needs a finite FAILSCORE",
          "[startup_validation][failscore_grading]") {
  const std::string head = "OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n"
                           "FAILSCORE_GRADING ON\n";
  PeriodicTable::initialize();
  SECTION("the default FAILSCORE is rejected") {
    ScopedTempFile tf(head + kOptimizerBlock);
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("FAILSCORE_GRADING ON requires a "
                                        "finite positive FAILSCORE"));
  }
  SECTION("a finite FAILSCORE passes") {
    ScopedTempFile tf(head + "FAILSCORE 1e9\n" + kOptimizerBlock);
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
}

namespace {

/* Bump-family (D2_Blip, w = 2) + DM_REAM bump density context. Every bump
 * support ends EXACTLY at its channel's rcut (c + 2/eta == RCUT): compliant
 * with the bump-support gate, and any loosening spills. `rctype` switches
 * the cutoff kind (the gate applies to discontinuous cutoffs only). */
Context make_bump_ctx(const char* rctype = "Cut_Dummy") {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "4.0 3.0 4.2");
  c.add("SGRID2B", "2.0 1.0 2.5");
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", rctype);
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 0);  // bcode Blip
  c.add("TYPEMB", 2); c.add("TYPEMB", 2); c.add("TYPEMB", 2);
  c.add("TYPEMB", 3); c.add("TYPEMB", 3);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.8 2.5");
  c.add("SGRIDMB", "1.0 1.1");
  c.add("AMPGRIDMB", "0.01 -1.2");
  c.add("RHOFORM", "SIGNED");  // a negative amplitude
  c.add("CEMBFUNC", "0.9 2.9 0.1");
  c.add("SEMBFUNC", "1.5 2.0 1.8");
  c.add("RCUTMB", 4.8);
  c.add("RCTYPEMB", rctype);
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  return c;
}

}  // namespace

TEST_CASE("startup validation: bump-support gate",
          "[startup_validation][bumpgate]") {
  PeriodicTable::initialize();
  ScopedTempFile plain(target_with("PC_ERMSE 0 1\n"));

  SECTION("compliant edge seeds pass (support exactly at rcut)") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, plain.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    REQUIRE(ctx.get<double>("RCUT2B") == 5.0);  // never modified
  }
  SECTION("seed spill errors, listing EVERY violating index + eta fix") {
    Context ctx = make_bump_ctx();
    ctx.remove("SGRID2B");
    ctx.add("SGRID2B", "1.5 0.8 2.5");  // bump 1 -> 5.33, bump 2 -> 5.5
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, plain.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("bump-support gate") &&
            Catch::Contains("CGRID2B bump 1") &&
            Catch::Contains("CGRID2B bump 2") &&
            Catch::Contains("set SGRID2B (2) >= 1"));
  }
  SECTION("partially feasible box arms the per-candidate gate (CANDIDATE)") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    // e_lo 1.0 -> worst corner 4.0 + 2/1.0 = 6.0 > 5, but the best corner
    // 4.0 + 2/3.0 = 4.67 <= 5 is feasible: accepted at full width under
    // the default BUMPGATE CANDIDATE (violating trials are rejected per
    // evaluation instead).
    ScopedTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("BUMPGATE STRICT: legacy box-corner error with the SGRID fix") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    // e_lo 1.0 -> corner 4.0 + 2/1.0 = 6.0 > 5; need lo >= 2/(5-4) = 2
    ScopedTempFile tf("BUMPGATE STRICT\nOPTIM SGRID2B (1) 1.0 3.0\n"
                      "PC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("bump-support gate") &&
            Catch::Contains("low bound >= w/(RCUT2B - c_hi) = 2"));
  }
  SECTION("CGRID box with a feasible sliver arms the gate; STRICT errors") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    // Seed eta 2.0 (static): best corner 4.5 + 2/2.0 = 5.5 > 5 would be
    // infeasible -- see the "no feasible point" SECTION below. Here the
    // low bound 3.9 keeps a feasible sliver (3.9 + 1.0 = 4.9 <= 5).
    ScopedTempFile tf("OPTIM CGRID2B (1) 3.9 5.5\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    ScopedTempFile tf2("BUMPGATE STRICT\nOPTIM CGRID2B (1) 3.9 5.5\n"
                       "PC_ERMSE 0 1\n" +
                       std::string(kOptimizerBlock));
    HPOTargets trg2(ctx, tf2.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg2, true, true),
        Catch::Contains("impossible") &&
            Catch::Contains("lower the CGRID2B high bound"));
  }
  SECTION("no feasible point in the box is still a hard error (CANDIDATE)") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    // Seed eta 2.0 (static): even the best corner 4.5 + 2/2.0 = 5.5 > 5.
    ScopedTempFile tf("OPTIM CGRID2B (1) 4.5 5.5\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("bump-support gate") &&
            Catch::Contains("no feasible point") &&
            Catch::Contains("CGRID2B"));
  }
  SECTION("MB density bump gate fires against RCUTMB") {
    Context ctx = make_bump_ctx();
    ctx.remove("CGRIDMB");
    ctx.add("CGRIDMB", "4.0 2.5");  // 4.0 + 2/1.0 = 6.0 > 4.8
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, plain.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("CGRIDMB bump 1"));
  }
  SECTION("smooth cutoff: identical spilling config passes (no gate)") {
    Context ctx = make_bump_ctx("Cut_Poly");
    ctx.remove("SGRID2B");
    ctx.add("SGRID2B", "1.5 0.8 2.5");  // spills under Cut_Dummy
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf("OPTIM CGRID2B (1) 3.9 5.5\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("OPTIM'd RCUT2B: reachable high bound keeps the box feasible") {
    Context ctx = make_bump_ctx();  // edge seeds reach exactly 5.0
    ctx.add("DBFILE", "dummy.db");
    // Sampled RCUT2B in [4.5, 6.0]: candidates below 5.0 make the static
    // edge bumps spill and are per-candidate rejects; the seed itself is
    // valid at its stated cutoff, so startup accepts (armed gate).
    ScopedTempFile tf("OPTIM RCUT2B (1) 4.5 6.0\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("OPTIM'd RCUT2B: STRICT keeps the LOW bound as the binding rcut") {
    Context ctx = make_bump_ctx();  // edge seeds reach exactly 5.0
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf("BUMPGATE STRICT\nOPTIM RCUT2B (1) 4.5 6.0\n"
                      "PC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("OPTIM RCUT2B low bound"));
  }
  SECTION("OPTIM'd RCUT2B: high bound below the seeds' reach hard-errors") {
    Context ctx = make_bump_ctx();  // edge seeds reach exactly 5.0
    ctx.add("DBFILE", "dummy.db");
    // Even the largest sampled cutoff (4.6) cannot fit the static bumps:
    // no feasible point anywhere in the search space.
    ScopedTempFile tf("OPTIM RCUT2B (1) 4.0 4.6\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("bump-support gate") &&
                            Catch::Contains("no feasible point"));
  }
}

namespace {

/* make_bump_ctx + the model keys a probe Trainer needs: D2_Blip (3) + DM_REAM
 * Blip embedding CEMBFUNC 0.9 2.9 0.1 / SEMBFUNC 1.5 2.0 1.8, so bumps 1 and 3
 * cover rho = 0 and bump 2 does not. K = 3 + 3 weights, BIAS false. */
Context make_ream_model_ctx() {
  Context c = make_bump_ctx();
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

const std::vector<double> kReamWeights = {0.8, -0.3, 0.1, -0.5, 2.0, 1.5};

/* sum_m w_m g_m(0) over the embedding weights (4..6), from the basis
 * primitive itself. */
double expected_raw_f0() {
  using tadah::models::BasisKind;
  using tadah::models::embedding_value;
  const double cemb[] = {0.9, 2.9, 0.1};
  const double semb[] = {1.5, 2.0, 1.8};
  double acc = 0.0;
  for (int m = 0; m < 3; ++m)
    acc += kReamWeights[3 + m] * embedding_value(BasisKind::Blip, 0.0,
                                                 semb[m], cemb[m]);
  return acc;
}

}  // namespace

TEST_CASE("startup validation: OPTIM EMBF0 is rejected",
          "[startup_validation][embf0]") {
  PeriodicTable::initialize();

  SECTION("at parse, before the key is looked up in the context") {
    for (const char* key : {"EMBF0", "embf0"}) {
      Context ctx = make_ream_model_ctx();
      ScopedTempFile tf(std::string("OPTIM ") + key + " (1) 0 1\n" +
                        "PC_ERMSE 0 1\n" + kOptimizerBlock);
      /* The error names its line, like every hpotarget error. */
      REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                          Catch::StartsWith("Error on line 1: ") &&
                              Catch::Contains("OPTIM EMBF0 is not a valid "
                                              "search target") &&
                              Catch::Contains("-c potential/configuration"));
      /* The rejection comes before context.size(k): the default is not
       * materialised into the run context. */
      CHECK_FALSE(ctx.has_stored("EMBF0"));
    }
  }
  SECTION("mirrored in validate_hpo_setup") {
    Context ctx = make_ream_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    HPOTargets trg(ctx, tf.path.string());
    trg.optimap["EMBF0"];  // a search space assembled without the parser
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("OPTIM EMBF0 is not a valid "
                                        "search target"));
  }
  SECTION("a top-level EMBF0 line in the hpotarget is an unknown key") {
    Context ctx = make_ream_model_ctx();
    ScopedTempFile tf("EMBF0 FREE\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                        Catch::StartsWith("Error on line 1: ") &&
                            Catch::Contains("unknown key \"EMBF0\""));
  }
  SECTION("an invalid stored EMBF0 of a DM_REAM configuration fails fast") {
    Context ctx = make_ream_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("EMBF0", "MAYBE");
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("EMBF0 must be ZERO or FREE"));
  }
  SECTION("a valid DM_REAM setup without the key passes, key not inserted") {
    Context ctx = make_ream_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    CHECK_FALSE(ctx.has_stored("EMBF0"));
  }
}

/* Pins: derive_raw_embedding_f0 takes FIXINDEX given as separate values
 * ('1 2 3 4 5 6', one value each) that pin every weight of the DM_REAM
 * model as the comma list: the value is provisioned from FIXWEIGHT
 * (source "FIXWEIGHT", sum w g(0)).
 * Before the fix: the lenient core::parse_indices dropped the joined
 * values, so FIXINDEX named no weight, the pins were not seen as covering
 * the 6 weights, and the value was determined per evaluation
 * (Kind::PerEvaluation, no source). */
TEST_CASE("derive_raw_embedding_f0: FIXINDEX given as separate values pins "
          "every weight",
          "[startup_validation][embf0][fixindex]") {
  PeriodicTable::initialize();
  using Kind = RawEmbeddingF0::Kind;
  Context ctx = make_ream_model_ctx();
  REQUIRE(derive_weight_count(ctx) == kReamWeights.size());
  ctx.add("FIXINDEX", "1 2 3 4 5 6");
  REQUIRE(ctx.size("FIXINDEX") == 6);
  for (double w : kReamWeights) ctx.add("FIXWEIGHT", w);
  const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
  CHECK(r.kind == Kind::Value);
  CHECK(r.source == "FIXWEIGHT");
  CHECK(r.value == Approx(expected_raw_f0()).epsilon(1e-13).margin(1e-15));
}

TEST_CASE("derive_raw_embedding_f0: raw F(0) of the -c potential",
          "[startup_validation][embf0]") {
  PeriodicTable::initialize();
  using Kind = RawEmbeddingF0::Kind;
  const double expected = expected_raw_f0();
  REQUIRE(expected != 0.0);  // bumps 1 and 3 cover rho = 0

  SECTION("all weights pinned: provisioned FIXWEIGHT, sum w g(0)") {
    for (const char* mode : {"", "ZERO", "FREE"}) {
      Context ctx = make_ream_model_ctx();
      if (*mode) ctx.add("EMBF0", mode);
      REQUIRE(derive_weight_count(ctx) == kReamWeights.size());
      ctx.add("FIXINDEX", "1-6");
      for (double w : kReamWeights) ctx.add("FIXWEIGHT", w);
      const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
      REQUIRE(r.kind == Kind::Value);
      CHECK(r.source == "FIXWEIGHT");
      /* The raw value does not depend on the embedding reference. */
      CHECK(r.value == Approx(expected).epsilon(1e-13).margin(1e-15));
      /* Probe on a copy: nothing inserted into the caller's context. */
      CHECK(ctx.has_stored("EMBF0") == (*mode != '\0'));
    }
  }
  SECTION("stored WEIGHTS of the model size") {
    Context ctx = make_ream_model_ctx();
    for (double w : kReamWeights) ctx.add("WEIGHTS", w);
    const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
    REQUIRE(r.kind == Kind::Value);
    CHECK(r.source == "WEIGHTS");
    CHECK(r.value == Approx(expected).epsilon(1e-13).margin(1e-15));
  }
  SECTION("partial pins and no WEIGHTS: determined per evaluation") {
    Context ctx = make_ream_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "4");
    ctx.add("FIXWEIGHT", 0.5);
    CHECK(derive_raw_embedding_f0(ctx).kind == Kind::PerEvaluation);
  }
  SECTION("WEIGHTS of the wrong size are not used") {
    Context ctx = make_ream_model_ctx();
    ctx.add("WEIGHTS", 1.0);
    CHECK(derive_raw_embedding_f0(ctx).kind == Kind::PerEvaluation);
  }
  SECTION("a model not linear in the descriptor: n/a") {
    Context ctx = make_ream_model_ctx();
    ctx.remove("MODEL");
    ctx.add("MODEL", "M_BLR");
    ctx.add("MODEL", "BF_Polynomial2");
    for (double w : kReamWeights) ctx.add("WEIGHTS", w);
    const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
    CHECK(r.kind == Kind::NotApplicable);
    CHECK(std::isnan(r.value));
    CHECK_THAT(r.reason, Catch::Contains("not linear"));
  }
  SECTION("NORM true: a linear model's stored WEIGHTS are raw-descriptor "
          "weights (to_raw_descriptors), so the value is reported") {
    Context ctx = make_ream_model_ctx();
    ctx.remove("NORM");
    ctx.add("NORM", "true");
    for (double w : kReamWeights) ctx.add("WEIGHTS", w);
    const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
    REQUIRE(r.kind == Kind::Value);
    CHECK(r.value == Approx(expected).epsilon(1e-13).margin(1e-15));
  }
  SECTION("no DM_REAM block: n/a") {
    Context ctx = make_model_ctx();
    const RawEmbeddingF0 r = derive_raw_embedding_f0(ctx);
    CHECK(r.kind == Kind::NotApplicable);
    CHECK_THAT(r.reason, Catch::Contains("DM_REAM"));
  }
}

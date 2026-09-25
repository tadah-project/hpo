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
 * keys — derive_weight_count returns 0 (light-path validation). */
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

TEST_CASE("startup validation: FIXINDEX/FIXWEIGHT pairing (no model keys)",
          "[startup_validation]") {
  ScopedTempFile tf(target_with("PC_ERMSE 0 1\n"));

  SECTION("lone FIXINDEX throws") {
    Context ctx = make_minimal_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "2");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("requires FIXWEIGHT"));
  }
  SECTION("size mismatch throws") {
    Context ctx = make_minimal_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "1,2,3");
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
    Context ctx = make_model_ctx();
    ctx.add("DBFILE", "dummy.db");
    ctx.add("FIXINDEX", "99");
    ctx.add("FIXWEIGHT", 0.5);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("out of range"));
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
    Context ctx = make_minimal_ctx();
    ctx.add("DBFILE", "dummy.db");
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("no DBFILE, all weights pinned, observable-only loss") {
    ScopedTempFile tf(target_with(
        "PC_OBSERVABLE cohesive_energy --species H --target 8.1 "
        "--weight 1.0 --lattice bcc --hotvar a0 2.5 4.5 3.3\n"));
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
      REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
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
          "weights (convert_to_nweights), so the value is reported") {
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

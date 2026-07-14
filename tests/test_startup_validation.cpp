// test_startup_validation.cpp
//
// Fail-fast rules enforced by validate_hpo_setup() before HPO_Host
// construction: FIXINDEX/FIXWEIGHT pairing and coverage, loss-source
// presence, validation-file requirements, and the NORM restriction.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
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
  SECTION("box corner spill errors with the exact SGRID low bound") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    // e_lo 1.0 -> corner 4.0 + 2/1.0 = 6.0 > 5; need lo >= 2/(5-4) = 2
    ScopedTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("bump-support gate") &&
            Catch::Contains("low bound >= w/(RCUT2B - c_hi) = 2"));
  }
  SECTION("CGRID high bound at/above rcut is called impossible") {
    Context ctx = make_bump_ctx();
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf("OPTIM CGRID2B (1) 4.5 5.5\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("impossible") &&
            Catch::Contains("lower the CGRID2B high bound"));
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
    ScopedTempFile tf("OPTIM CGRID2B (1) 4.5 5.5\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("OPTIM'd RCUT2B: its LOW bound is the binding rcut") {
    Context ctx = make_bump_ctx();  // edge seeds reach exactly 5.0
    ctx.add("DBFILE", "dummy.db");
    ScopedTempFile tf("OPTIM RCUT2B (1) 4.5 6.0\nPC_ERMSE 0 1\n" +
                      std::string(kOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("OPTIM RCUT2B low bound"));
  }
}

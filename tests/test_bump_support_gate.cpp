// test_bump_support_gate.cpp
//
// Per-candidate bump-support gate (BUMPGATE CANDIDATE, the default):
//  * unit tests for detail::build_bump_gate_table /
//    detail::bump_gate_violation / detail::derive_binding_rcut
//    (bump_support_gate.h) — table arming rules, flat-index wiring,
//    violation measure monotonicity, sampled-RCUT unlock, and the graded
//    band-D loss mapping;
//  * a hidden HPO_Host end-to-end case ([bumpgate_e2e][.]): an infeasible
//    candidate returns a graded band-D loss without training, a feasible
//    candidate trains normally, FAILSCORE_GRADING OFF returns the flat
//    FAILSCORE, and BUMPGATE STRICT restores the startup box-corner error.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>

#include <dlib/matrix.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::FailBand;
using tadah::hpo::HPOTargets;
using tadah::hpo::PcFailInfo;
using tadah::hpo::graded_eval_loss;
using tadah::hpo::loss_is_feasible;
using tadah::hpo::validate_hpo_setup;
namespace bg = tadah::hpo::detail;

struct BgTempFile {
  fs::path path;
  explicit BgTempFile(const std::string& body)
  : path(fs::temp_directory_path() /
         ("tadah_bg_" +
          std::to_string(std::chrono::steady_clock::now()
                             .time_since_epoch().count()) + ".in")) {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~BgTempFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

constexpr const char* kBgOptimizerBlock =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

/* Two-bump D2_Blip (w = 2) channel under Cut_Dummy. Both seed supports
 * end EXACTLY at RCUT2B = 5: 4.0 + 2/2.0 = 3.0 + 2/1.0 = 5.0. */
Context make_gate_ctx() {
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

/* Minimal ParamVec: the gate reads p(flat_idx) only. */
struct Vec {
  std::vector<double> v;
  double operator()(long i) const { return v[static_cast<std::size_t>(i)]; }
};

}  // namespace

TEST_CASE("bump-support gate table: arming rules and flat-index wiring",
          "[bump_support_gate][bumpgate]") {
  PeriodicTable::initialize();

  SECTION("both OPT'd, partially infeasible box: armed with both indices") {
    Context ctx = make_gate_ctx();
    BgTempFile tf("OPTIM CGRID2B (1) 3.0 4.8\nOPTIM SGRID2B (1) 1.0 3.0\n"
                  "PC_ERMSE 0 1\n" + std::string(kBgOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    const auto tbl = bg::build_bump_gate_table(ctx, trg);
    REQUIRE(tbl.size() == 1);
    const auto& ch = tbl[0];
    CHECK(ch.ckey == "CGRID2B");
    CHECK(ch.w == 2.0);
    CHECK(ch.rcut.fixed_max == 5.0);
    CHECK(ch.rcut.flat_idx.empty());
    /* bump 2 is static and seed-compliant: never armed */
    REQUIRE(ch.bumps.size() == 1);
    CHECK(ch.bumps[0].index == 0);
    /* flat order: CGRID2B[0] = 0, SGRID2B[0] = 1 (lexicographic keys) */
    CHECK(ch.bumps[0].c_flat == 0);
    CHECK(ch.bumps[0].s_flat == 1);

    /* violation: zero on feasible points, positive and monotone past the
     * boundary (measure = summed overhang / half-width) */
    CHECK(bg::bump_gate_violation(tbl, Vec{{4.0, 2.0}}) == 0.0);  // edge
    CHECK(bg::bump_gate_violation(tbl, Vec{{3.0, 1.0}}) == 0.0);  // 3+2=5
    const double v_bad = bg::bump_gate_violation(tbl, Vec{{4.8, 1.0}});
    CHECK(v_bad == Approx(0.9));                    // (6.8-5)*1.0/2
    const double v_less_c = bg::bump_gate_violation(tbl, Vec{{4.4, 1.0}});
    const double v_more_eta = bg::bump_gate_violation(tbl, Vec{{4.8, 2.0}});
    CHECK(v_less_c > 0.0);
    CHECK(v_less_c < v_bad);      // smaller centre -> smaller violation
    CHECK(v_more_eta > 0.0);
    CHECK(v_more_eta < v_bad);    // narrower bump -> smaller violation
  }

  SECTION("only eta OPT'd: centre is carried as the fixed seed value") {
    Context ctx = make_gate_ctx();
    BgTempFile tf("OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" +
                  std::string(kBgOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    const auto tbl = bg::build_bump_gate_table(ctx, trg);
    REQUIRE(tbl.size() == 1);
    REQUIRE(tbl[0].bumps.size() == 1);
    CHECK(tbl[0].bumps[0].c_flat == -1);
    CHECK(tbl[0].bumps[0].c_fix == 4.0);
    CHECK(tbl[0].bumps[0].s_flat == 0);
    CHECK(bg::bump_gate_violation(tbl, Vec{{2.0}}) == 0.0);
    CHECK(bg::bump_gate_violation(tbl, Vec{{1.0}}) == Approx(0.5));
  }

  SECTION("fully feasible box: nothing is armed (empty table)") {
    Context ctx = make_gate_ctx();
    BgTempFile tf("OPTIM SGRID2B (1) 2.0 8.0\nPC_ERMSE 0 1\n" +
                  std::string(kBgOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    CHECK(bg::build_bump_gate_table(ctx, trg).empty());
  }

  SECTION("no OPTIM lines at all: empty table") {
    Context ctx = make_gate_ctx();
    BgTempFile tf("OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n" +
                  std::string(kBgOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    CHECK(bg::build_bump_gate_table(ctx, trg).empty());
  }

  SECTION("OPT'd RCUT arms static edge bumps; sampled cutoff unlocks") {
    Context ctx = make_gate_ctx();
    BgTempFile tf("OPTIM RCUT2B (1) 4.0 6.0\nPC_ERMSE 0 1\n" +
                  std::string(kBgOptimizerBlock));
    HPOTargets trg(ctx, tf.path.string());
    const auto rc = bg::derive_binding_rcut(ctx, trg, "2B");
    REQUIRE(rc.flat_idx.size() == 1);
    CHECK(rc.flat_idx[0] == 0);
    CHECK(rc.min_reachable == 4.0);
    CHECK(rc.max_reachable == 6.0);
    const auto tbl = bg::build_bump_gate_table(ctx, trg);
    REQUIRE(tbl.size() == 1);
    /* both static bumps reach 5.0 > min_reachable 4.0: armed */
    REQUIRE(tbl[0].bumps.size() == 2);
    CHECK(tbl[0].bumps[0].c_flat == -1);
    CHECK(tbl[0].bumps[0].s_flat == -1);
    /* candidate carrying rcut >= 5.0 is feasible ... */
    CHECK(bg::bump_gate_violation(tbl, Vec{{6.0}}) == 0.0);
    CHECK(bg::bump_gate_violation(tbl, Vec{{5.0}}) == 0.0);
    /* ... one below the seeds' reach is not: bump1 (eta 2) contributes
     * 1.0*2/2, bump2 (eta 1) contributes 1.0*1/2 */
    CHECK(bg::bump_gate_violation(tbl, Vec{{4.0}}) == Approx(1.5));
  }

  SECTION("graded band-D mapping: in [0.999*FS, FS), never feasible") {
    const double fs = 1.0e9;
    PcFailInfo info;
    info.band = static_cast<int>(FailBand::D);
    info.vD = 0.9;
    const double L = graded_eval_loss(info, fs, 1.0);
    CHECK(L >= 0.999 * fs);
    CHECK(L < fs);                       // rankable: strictly below FS
    CHECK_FALSE(loss_is_feasible(L, fs));
    /* monotone in the violation within the band */
    info.vD = 1.8;
    CHECK(graded_eval_loss(info, fs, 1.0) > L);
  }
}

/* Hidden host-level e2e (thread-local trainer cache; run explicitly):
 *   ./test_hpo "[bumpgate_e2e]" */
TEST_CASE("HPO_Host per-candidate bump gate: graded reject before training",
          "[bumpgate_e2e][.][!benchmark]") {
  using ParamVec = dlib::matrix<double, 0, 1>;
  using Host = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
  PeriodicTable::initialize();
  Host::reset_globals_for_test();

  struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
      dir = fs::temp_directory_path() /
            ("tadah_bumpgate_e2e_" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch().count()));
      fs::create_directories(dir);
      original_cwd = fs::current_path();
      fs::current_path(dir);
    }
    ~ScratchDir() {
      std::error_code ec;
      fs::current_path(original_cwd);
      fs::remove_all(dir, ec);
    }
  } sd;

  const fs::path train_db = sd.dir / "train.db";
  const fs::path val_db   = sd.dir / "val.db";
  const fs::path val_cfg  = sd.dir / "config.val";
  const fs::path hpo_cfg  = sd.dir / "config.hpo";

  auto write_db = [](const fs::path& p, double offset_x) {
    std::ofstream os(p);
    os << "test_bcc_ta\n"
       << "1.0 1.0 1.0\n"
       << "-0.001 0.0\n"
       << "3.301 0.0 0.0\n0.0 3.301 0.0\n0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n"
       << "\n";
  };
  write_db(train_db, 0.0);
  write_db(val_db, 0.07);
  {
    std::ofstream os(val_cfg);
    os << "DBFILE " << val_db.string() << "\n";
  }

  const double fs_score = 1.0e9;
  const std::string base_target =
      "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
      "LOSS L2\n"
      "PC_EMAE 0 1\n"
      "FAILSCORE 1e9\n"
      "OPTIM CGRID2B (1) 3.0 4.8\n"
      "OPTIM SGRID2B (1) 1.0 3.0\n";
  {
    std::ofstream os(hpo_cfg);
    os << base_target << "FAILSCORE_GRADING ON\n";
  }

  /* Trainer-able master context: D2_Blip (w = 2) under Cut_Dummy, both
   * seed supports ending exactly at RCUT2B = 5. */
  Context ctx;
  ctx.add("DBFILE", train_db.string());
  ctx.add("ATOM", "Ta");
  ctx.add("WATOM", 1);
  ctx.add("INIT2B", "true");
  ctx.add("INITMB", "false");
  ctx.add("TYPE2B", "D2_Blip");
  ctx.add("TYPE2B", 2);
  ctx.add("TYPE2B", 2);
  ctx.add("TYPE2B", "Ta");
  ctx.add("TYPE2B", "Ta");
  ctx.add("CGRID2B", "4.0 3.0");
  ctx.add("SGRID2B", "2.0 1.0");
  ctx.add("RCTYPE2B", "Cut_Dummy");
  ctx.add("RCUT2B", 5.0);
  ctx.add("TYPEMB", "DM_Dummy");
  ctx.add("TYPEMB", "Ta");
  ctx.add("TYPEMB", "Ta");
  ctx.add("RCTYPEMB", "Cut_Cos");
  ctx.add("RCUTMB", 5.0);
  ctx.add("DIMER", "false");
  ctx.add("DIMER", 0);
  ctx.add("DIMER", "false");
  ctx.add("FORCE", "false");
  ctx.add("STRESS", "false");
  ctx.add("EWEIGHT", 1.0);
  ctx.add("FWEIGHT", 1.0);
  ctx.add("SWEIGHT", 1.0);
  ctx.add("BIAS", "false");
  ctx.add("NORM", "false");
  ctx.add("VERBOSE", 0);
  ctx.add("MODEL", "M_BLR");
  ctx.add("MODEL", "BF_Linear");
  ctx.add("LAMBDA", 1e-3);
  ctx.add("OALGO", 4);
  ctx.add("HPOTARGET", hpo_cfg.string());
  ctx.add("VALIDATION", val_cfg.string());

  SECTION("graded reject, no training; feasible point trains normally") {
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE(trg.FAILSCORE_GRADING);
    REQUIRE_FALSE(trg.BUMPGATE_STRICT);
    /* the partially infeasible box passes startup under CANDIDATE */
    Context vctx = ctx;
    REQUIRE_NOTHROW(validate_hpo_setup(vctx, trg, true, true));

    Host host(ctx, trg, val_cfg.string());

    ParamVec p_bad(2);
    p_bad(0) = 4.8;   /* CGRID2B[1] */
    p_bad(1) = 1.0;   /* SGRID2B[1] -> reach 6.8 > 5 */
    double loss_bad = 0.0;
    REQUIRE_NOTHROW(loss_bad = host(p_bad));
    CHECK(loss_bad >= 0.999 * fs_score);
    CHECK(loss_bad < fs_score);                       // graded, rankable
    CHECK_FALSE(loss_is_feasible(loss_bad, fs_score));

    /* no training happened: the gradient hook must refuse the (absent /
     * stale) trainer cache and route every dim to FD */
    ParamVec g(2);
    std::vector<int> needs_fd;
    REQUIRE(host.analytical_gradient(p_bad, g, needs_fd));
    CHECK(needs_fd.size() == 2);

    ParamVec p_good(2);
    p_good(0) = 4.0;
    p_good(1) = 2.0;  /* reach 5.0 == RCUT2B: feasible edge */
    double loss_good = 0.0;
    REQUIRE_NOTHROW(loss_good = host(p_good));
    CHECK(std::isfinite(loss_good));
    CHECK(loss_is_feasible(loss_good, fs_score));
  }

  SECTION("FAILSCORE_GRADING OFF: flat FAILSCORE sentinel") {
    const fs::path hpo_cfg_off = sd.dir / "config_off.hpo";
    {
      std::ofstream os(hpo_cfg_off);
      os << base_target;
    }
    Context ctx_off = ctx;
    ctx_off.remove("HPOTARGET");
    ctx_off.add("HPOTARGET", hpo_cfg_off.string());
    HPOTargets trg(ctx_off, hpo_cfg_off.string());
    REQUIRE_FALSE(trg.FAILSCORE_GRADING);
    Host host(ctx_off, trg, val_cfg.string());
    ParamVec p_bad(2);
    p_bad(0) = 4.8;
    p_bad(1) = 1.0;
    double loss_bad = 0.0;
    REQUIRE_NOTHROW(loss_bad = host(p_bad));
    CHECK(loss_bad == fs_score);
  }

  SECTION("BUMPGATE STRICT: the same box is a startup error") {
    const fs::path hpo_cfg_strict = sd.dir / "config_strict.hpo";
    {
      std::ofstream os(hpo_cfg_strict);
      os << base_target << "BUMPGATE STRICT\n";
    }
    Context ctx_strict = ctx;
    ctx_strict.remove("HPOTARGET");
    ctx_strict.add("HPOTARGET", hpo_cfg_strict.string());
    HPOTargets trg(ctx_strict, hpo_cfg_strict.string());
    REQUIRE(trg.BUMPGATE_STRICT);
    REQUIRE_THROWS_WITH(
        validate_hpo_setup(ctx_strict, trg, true, true),
        Catch::Contains("bump-support gate"));
  }

  Host::reset_globals_for_test();
}

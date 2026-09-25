// F2: knot-family RCUT auto-derivation in the HPO loop.
//  * detail::apply_knot_rcut_derivation — per-eval re-derivation hook
//  * detail::nn_safe_rcutmax — one-time NN cutoff incl. knot-grid bounds
//  * validate_hpo_setup — startup normalisation + OPTIM RCUT warnings
//  * hidden [rcut_e2e][.] — an OPTIM'd knot pushed beyond the stated RCUT
//    evaluates identically to a host whose master context already sits at
//    that cutoff (proves the prebuilt NN list covers the grown cutoff and
//    the per-eval hook applies it).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
using tadah::hpo::detail::apply_knot_rcut_derivation;
using tadah::hpo::detail::nn_safe_rcutmax;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() /
              ("tadah_hpo_rcut_" +
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
};

struct ForceRebuildGuard {
    bool restore;
    ForceRebuildGuard() {
        restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1);
    }
    ~ForceRebuildGuard() { if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD"); }
};

/* Same minimal 4-atom BCC Ta fixture as test_hpo_host_e2e.cpp. */
void write_minimal_ta_dbfile(const fs::path& path, double offset_x) {
    std::ofstream os(path);
    os << "test_bcc_ta\n"
       << "1.0 1.0 1.0\n"
       << "-0.001 0.0\n"
       << "3.301 0.0 0.0\n"
       << "0.0 3.301 0.0\n"
       << "0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n"
       << "\n";
}

/* Knot-family (D2_Knot5 + DM_REAM bcode 2) Ta context in the refit shape.
 * first_knot parameterises the largest pair knot; RCUT2B is emitted
 * consistently (= the largest knot) unless stale_rcut2b overrides it. */
Context make_knot_ctx(double first_knot = 5.0, double stale_rcut2b = -1.0) {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1.0);
    c.add("INIT2B",   "true");
    c.add("TYPE2B",   "D2_Knot5");
    c.add("TYPE2B",   3);
    c.add("TYPE2B",   0);
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  first_knot);
    c.add("CGRID2B",  3.0);
    c.add("CGRID2B",  4.2);
    c.add("RCUT2B",   stale_rcut2b > 0 ? stale_rcut2b
                                       : std::max(first_knot, 4.2));
    c.add("RCTYPE2B", "Cut_Dummy");
    c.add("INITMB",   "true");
    c.add("TYPEMB",   "DM_REAM");
    c.add("TYPEMB",   2);
    c.add("TYPEMB",   2); c.add("TYPEMB", 0); c.add("TYPEMB", 2);
    c.add("TYPEMB",   3); c.add("TYPEMB", 0);
    c.add("TYPEMB",   0); c.add("TYPEMB", 0);
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("CGRIDMB",  4.8);
    c.add("CGRIDMB",  2.5);
    c.add("AMPGRIDMB", 0.01);
    c.add("AMPGRIDMB", -1.2);
    c.add("CEMBFUNC", "0.9 -2.9 0.1");
    c.add("RCUTMB",   4.8);
    c.add("RCTYPEMB", "Cut_Dummy");
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "false");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

/* Blip twin: D2_Blip / DM_REAM bcode 0 with width keys. Every bump support
 * ends EXACTLY at its channel's rcut (c + 2/eta == RCUT) so the fixture
 * passes the bump-support startup gate (spilling seeds are a hard error
 * since the gate landed). */
Context make_blip_ctx() {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1.0);
    c.add("INIT2B",   "true");
    c.add("TYPE2B",   "D2_Blip");
    c.add("TYPE2B",   3);
    c.add("TYPE2B",   3);
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "4.0 3.0 4.2");
    c.add("SGRID2B",  "2.0 1.0 2.5");
    c.add("RCUT2B",   5.0);
    c.add("RCTYPE2B", "Cut_Dummy");
    c.add("INITMB",   "true");
    c.add("TYPEMB",   "DM_REAM");
    c.add("TYPEMB",   0);
    c.add("TYPEMB",   2); c.add("TYPEMB", 2); c.add("TYPEMB", 2);
    c.add("TYPEMB",   3); c.add("TYPEMB", 3);
    c.add("TYPEMB",   0); c.add("TYPEMB", 0);
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("CGRIDMB",  "2.8 2.5");
    c.add("SGRIDMB",  "1.0 1.1");
    c.add("AMPGRIDMB", "0.01 -1.2");
    c.add("CEMBFUNC", "0.9 2.9 0.1");
    c.add("SEMBFUNC", "1.5 2.0 1.8");
    c.add("RCUTMB",   4.8);
    c.add("RCTYPEMB", "Cut_Dummy");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    return c;
}

struct ScopedTempFile {
    fs::path path;
    explicit ScopedTempFile(const std::string& body)
    : path(fs::temp_directory_path() /
           ("tadah_rcut_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ".in")) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~ScopedTempFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

constexpr const char* kOptimizerBlock =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n";

}  // namespace

TEST_CASE("rcut derivation: per-eval hook tracks the largest knot",
          "[rcut_autoderive]") {
    tadah::core::PeriodicTable::initialize();
    Context ctx = make_knot_ctx();

    SECTION("no change -> no churn") {
        REQUIRE_FALSE(apply_knot_rcut_derivation(ctx, true, true));
        REQUIRE(ctx.get<double>("RCUT2B") == 5.0);
    }
    SECTION("an OPTIM'd knot moves the derived cutoff with it") {
        ctx.remove("CGRID2B");
        ctx.add("CGRID2B", 5.7);   // the optimiser pushed knot 1 outward
        ctx.add("CGRID2B", 3.0);
        ctx.add("CGRID2B", 4.2);
        REQUIRE(apply_knot_rcut_derivation(ctx, true, true));
        REQUIRE(ctx.get<double>("RCUT2B") == Approx(5.7));
        REQUIRE(ctx.get<double>("RCUT2BMAX") == Approx(5.7));
        REQUIRE(ctx.get<double>("RCUTMAX") == Approx(5.7));
        // shrinking works too (prediction-exact for knots)
        ctx.remove("CGRID2B");
        ctx.add("CGRID2B", "4.5 3.0 4.2");
        REQUIRE(apply_knot_rcut_derivation(ctx, true, true));
        REQUIRE(ctx.get<double>("RCUT2B") == Approx(4.5));
    }
    SECTION("flags off (bump channels) -> untouched") {
        Context blip = make_blip_ctx();
        REQUIRE_FALSE(apply_knot_rcut_derivation(blip, false, false));
        REQUIRE(blip.get<double>("RCUT2B") == 5.0);
    }
}

TEST_CASE("rcut derivation: exact, so the cutoff is a function of theta "
          "alone (no evaluation-history hysteresis)",
          "[rcut_autoderive][lossless]") {
    /* The trainer context persists per thread, so the RCUT2B the hook sees
     * is the previous evaluation's. A tolerance there (was 1e-12 relative)
     * kept a stale cutoff for knot moves below it: the same theta got a
     * cutoff that depended on which theta that thread evaluated before. */
    tadah::core::PeriodicTable::initialize();
    const double target = 5.0 + 3e-12;
    auto set_knot = [](Context& c, double k) {
        c.remove("CGRID2B");
        c.add("CGRID2B", k);
        c.add("CGRID2B", 3.0);
        c.add("CGRID2B", 4.2);
    };
    SECTION("a sub-1e-12 knot move is applied exactly") {
        Context ctx = make_knot_ctx();
        set_knot(ctx, target);
        REQUIRE(apply_knot_rcut_derivation(ctx, true, true));
        REQUIRE(ctx.get<double>("RCUT2B") == target);
        REQUIRE(ctx.get<double>("RCUT2BMAX") == target);
        REQUIRE_FALSE(apply_knot_rcut_derivation(ctx, true, true));
    }
    SECTION("two histories ending at the same theta agree bit-for-bit") {
        Context a = make_knot_ctx();   // thread A: 5.0 -> target
        set_knot(a, 5.0);
        apply_knot_rcut_derivation(a, true, true);
        set_knot(a, target);
        apply_knot_rcut_derivation(a, true, true);

        Context b = make_knot_ctx();   // thread B: 5+1e-11 -> target
        set_knot(b, 5.0 + 1e-11);
        apply_knot_rcut_derivation(b, true, true);
        set_knot(b, target);
        apply_knot_rcut_derivation(b, true, true);

        REQUIRE(a.get<double>("RCUT2B") == b.get<double>("RCUT2B"));
        REQUIRE(a.get<double>("RCUT2BMAX") == b.get<double>("RCUT2BMAX"));
        REQUIRE(a.get<double>("RCUT2B") == target);
    }
}

TEST_CASE("rcut derivation: NN-safe cutoff covers OPTIM'd knot bounds",
          "[rcut_autoderive]") {
    tadah::core::PeriodicTable::initialize();

    SECTION("knot channel: CGRID2B highbound extends the NN cutoff") {
        Context ctx = make_knot_ctx();
        ScopedTempFile tf("OPTIM CGRID2B (1) 5.5 6.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE(nn_safe_rcutmax(ctx, trg) == Approx(6.0));
    }
    SECTION("legacy behaviour: OPTIM RCUT2B highbound still counts") {
        Context ctx = make_knot_ctx();
        ScopedTempFile tf("OPTIM RCUT2B (1) 5.0 5.6\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE(nn_safe_rcutmax(ctx, trg) == Approx(5.6));
    }
    SECTION("bump channel: CGRID2B bounds do NOT extend the cutoff") {
        Context ctx = make_blip_ctx();
        ScopedTempFile tf("OPTIM CGRID2B (1) 5.5 6.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE(nn_safe_rcutmax(ctx, trg) == Approx(5.0));
    }
    SECTION("no OPTIM lines on cutoff-relevant keys -> stated RCUTMAX") {
        Context ctx = make_knot_ctx();
        ScopedTempFile tf("OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE(nn_safe_rcutmax(ctx, trg) == Approx(5.0));
    }
}

TEST_CASE("rcut derivation: startup validation normalises stale knot RCUT",
          "[rcut_autoderive]") {
    tadah::core::PeriodicTable::initialize();

    SECTION("stale RCUT2B is corrected before the host is built") {
        // table-cutoff style: stated 5.3 vs largest knot 5.0; RCUTMB is
        // stale too (4.9 vs largest density knot 4.8)
        Context ctx = make_knot_ctx(5.0, /*stale_rcut2b=*/5.3);
        ctx.remove("RCUTMB");
        ctx.add("RCUTMB", 4.9);
        ctx.add("DBFILE", "dummy.db");
        ScopedTempFile tf("OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        validate_hpo_setup(ctx, trg, true, true);
        REQUIRE(ctx.get<double>("RCUT2B") == Approx(5.0));
        REQUIRE(ctx.get<double>("RCUTMB") == Approx(4.8));
        REQUIRE(ctx.get<double>("RCUTMAX") == Approx(5.0));
    }
    SECTION("OPTIM RCUT2B on a knot channel warns but proceeds") {
        Context ctx = make_knot_ctx();
        ctx.add("DBFILE", "dummy.db");
        ScopedTempFile tf("OPTIM RCUT2B (1) 4.5 5.5\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    }
    SECTION("bump channels: RCUT never modified; spilling boxes hard-error") {
        // A CGRID2B box crossing RCUT is no longer a warning: the
        // bump-support gate rejects it (the overhang would be
        // hard-truncated at the cutoff).
        Context ctx = make_blip_ctx();
        ctx.add("DBFILE", "dummy.db");
        ScopedTempFile tf("OPTIM CGRID2B (1) 5.5 6.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                            Catch::Contains("bump-support gate"));
        REQUIRE(ctx.get<double>("RCUT2B") == 5.0);  // stays authoritative
    }
    SECTION("bump channels: compliant boxes pass, RCUT untouched") {
        Context ctx = make_blip_ctx();
        ctx.add("DBFILE", "dummy.db");
        // binding eta = min(seed 2.0, lowbound 2.0) = 2.0; corner
        // 4.0 + 2/2.0 = 5.0 == RCUT2B (exact edge passes)
        ScopedTempFile tf("OPTIM CGRID2B (1) 3.7 4.0\n"
                          "OPTIM SGRID2B (1) 2.0 8.0\nPC_ERMSE 0 1\n" +
                          std::string(kOptimizerBlock));
        HPOTargets trg(ctx, tf.path.string());
        REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
        REQUIRE(ctx.get<double>("RCUT2B") == 5.0);
    }
}

/* Hidden e2e like the other HPO_Host cases (thread-local trainer cache):
 *   ./test_hpo "[rcut_e2e]" */
TEST_CASE("rcut e2e: OPTIM'd knot beyond the stated RCUT evaluates exactly",
          "[rcut_e2e][.]")
{
    tadah::core::PeriodicTable::initialize();
    ForceRebuildGuard _rebuild;
    ScratchDir sd;

    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train_db, 0.0);
    write_minimal_ta_dbfile(val_db,   0.07);
    { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }
    {
        std::ofstream os(hpo_cfg);
        os << "OPTIM CGRID2B (1) 5.5 6.0\nPC_ERMSE 0 1\n" << kOptimizerBlock;
    }

    /* Host A: master knot 1 sits at 5.0 (RCUT2B 5.0) but the OPTIM bound
     * reaches 6.0 — the bcc fixture HAS neighbour pairs between 5.0 and 6.0
     * (e.g. 5.717), so an undersized NN list would corrupt the design
     * matrix silently. */
    double loss_a;
    {
        Host::reset_globals_for_test();
        Context ctx = make_knot_ctx(5.0);
        ctx.add("DBFILE", train_db.string());
        ctx.add("HPOTARGET", hpo_cfg.string());
        HPOTargets trg(ctx, hpo_cfg.string());
        validate_hpo_setup(ctx, trg, true, true);
        Host host(ctx, trg, val_cfg.string());
        ParamVec p(1);
        p(0) = 6.0;
        loss_a = host(p);
        REQUIRE(std::isfinite(loss_a));
        // the per-eval hook stamped the derived cutoff into the eval ctx
        const auto *tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        REQUIRE(tr->context.get<double>("RCUT2B") == Approx(6.0));
        REQUIRE(tr->context.get<double>("RCUT2BMAX") == Approx(6.0));
    }

    /* Host B: reference — the master context already sits at knot 6.0 with
     * a consistent RCUT2B, so its NN list is exact by construction. */
    double loss_b;
    {
        Host::reset_globals_for_test();
        Context ctx = make_knot_ctx(6.0);
        ctx.add("DBFILE", train_db.string());
        ctx.add("HPOTARGET", hpo_cfg.string());
        HPOTargets trg(ctx, hpo_cfg.string());
        validate_hpo_setup(ctx, trg, true, true);
        Host host(ctx, trg, val_cfg.string());
        ParamVec p(1);
        p(0) = 6.0;
        loss_b = host(p);
        REQUIRE(std::isfinite(loss_b));
    }

    REQUIRE(loss_a == Approx(loss_b).epsilon(1e-12));
}

// FIXWEIGHT end-to-end host tests: mixed LS/fixed/OPTIM weights (case 2)
// and no-training-DBFILE operation (case 3).
//
// Hidden `[.]` like the other HPO_Host e2e cases: operator() populates the
// thread-local trainer cache, and interleaving with other host tests in one
// binary can leave stale descriptor state behind. Run explicitly with:
//   ./test_hpo "[fixweight_e2e]"
// Every test here sets TADAH_HPO_FORCE_REBUILD for hygiene.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() /
              ("tadah_hpo_fixw_" +
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

/* Ta D2_BP[5,5] + BF_Linear master context (K = 5 weights — the user
 * scenario: 5 weights, 1-3 LS, 4 pinned, 5 OPTIM'd). DBFILE / VALIDATION /
 * FIXINDEX / FIXWEIGHT are added by the individual tests. */
tadah::core::Context make_ta_base_context(const fs::path& hpo_cfg)
{
    tadah::core::Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(5));
    c.add("TYPE2B",   static_cast<int>(5));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "0.3 0.6 1.0 1.4 1.8");
    c.add("SGRID2B",  "2.0 2.7 3.4 4.2 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   5.0);
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   5.0);
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
    c.add("HPOTARGET", hpo_cfg.string());
    return c;
}

constexpr const char* kOptimizerBlock =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n";

}  // namespace

/* Case 2 (the user scenario, K=4 variant): weights 1..K-2 LS-fitted,
 * weight K-1 pinned at a constant, weight K searched by the optimiser. */
TEST_CASE("fixweight mixed mode: LS + pinned + OPTIM'd weights",
          "[fixweight_e2e][.]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ForceRebuildGuard _rebuild;
    ScratchDir sd;

    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train_db, 0.0);
    write_minimal_ta_dbfile(val_db,   0.07);
    { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }

    tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    REQUIRE(K >= 3);

    const double pinned_val = 0.125;
    ctx.add("DBFILE", train_db.string());
    ctx.add("VALIDATION", val_cfg.string());
    ctx.add("FIXINDEX", std::to_string(K - 1) + "," + std::to_string(K));
    ctx.add("FIXWEIGHT", pinned_val);   // pins weight K-1 (constant)
    ctx.add("FIXWEIGHT", 0.02);         // pins weight K (searched below)

    {
        std::ofstream os(hpo_cfg);
        os << kOptimizerBlock
           << "LOSS L2\nPC_EMAE 0 1\n"
           << "OPTIM FIXWEIGHT (" << K << ") -1.0 1.0\n";
    }

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, true, true));
    Host host(ctx, trg, val_cfg.string());

    std::vector<double> free_w_first;   // weights 1..K-2 at v=0.02
    bool first_eval = true;
    for (double v : {0.02, 0.35}) {
        if (!first_eval) {
            /* Route the second evaluation through the CACHED trainer
             * (update_context + reinit + re-train) — the production
             * path. Regression: the reduced LS re-solve shrinks the
             * model weight vector (K -> K-2), which used to trip a
             * heap overflow in aed_type::resize. */
            ::unsetenv("TADAH_HPO_FORCE_REBUILD");
        }
        first_eval = false;
        ParamVec p(1);
        p(0) = v;
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = host(p));
        CHECK(std::isfinite(loss));

        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        const auto& w = tr->model->get_weights();
        REQUIRE(w.size() == K);

        /* Pinned constant survives every eval verbatim; the OPTIM'd
         * weight equals the optimiser's live value (string round-trip
         * through the Context, hence Approx). */
        CHECK(w[K - 2] == Approx(pinned_val).epsilon(1e-12));
        CHECK(w[K - 1] == Approx(v).epsilon(1e-12));

        if (free_w_first.empty()) {
            for (std::size_t i = 0; i + 2 < K; ++i)
                free_w_first.push_back(w[i]);
        } else {
            /* The free weights are re-fitted against the residual
             * target, so moving the OPTIM'd weight must move them. */
            bool any_changed = false;
            for (std::size_t i = 0; i + 2 < K; ++i)
                any_changed |= std::abs(w[i] - free_w_first[i]) > 1e-14;
            CHECK(any_changed);
        }

        /* Round-trip: the per-eval pot context carries the live
         * FIXWEIGHT values AND the matching trained WEIGHTS, so a
         * best_pot.tadah re-seeds the next HPO run consistently. */
        auto pot = const_cast<tadah::mlip::Trainer*>(tr)->get_param_file();
        REQUIRE(pot.contains("FIXWEIGHT"));
        std::vector<double> fw(pot.size("FIXWEIGHT"));
        pot.get("FIXWEIGHT", fw);
        REQUIRE(fw.size() == 2);
        CHECK(fw[1] == Approx(v).epsilon(1e-12));
        std::vector<double> wpot(pot.size("WEIGHTS"));
        pot.get("WEIGHTS", wpot);
        REQUIRE(wpot.size() == K);
        CHECK(wpot[K - 1] == Approx(v).epsilon(1e-12));
    }

    /* Analytical/HYBRID gradients: FIXINDEX present -> every dim must be
     * routed onto the masked-FD path. */
    ParamVec p(1); p(0) = 0.02;
    ParamVec g(1);
    std::vector<int> needs_fd;
    REQUIRE(host.analytical_gradient(p, g, needs_fd));
    CHECK(needs_fd.size() == 1u);

    Host::reset_globals_for_test();
}

/* All-fixed with a DBFILE present (the `tadah refit` coefficient-search
 * workload): the fast path provisions weights directly and skips the
 * per-eval design-matrix build. */
TEST_CASE("fixweight all-fixed with DBFILE: DM-skip fast path",
          "[fixweight_e2e][.]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ForceRebuildGuard _rebuild;
    ScratchDir sd;

    const fs::path train_db = sd.dir / "train.db";
    const fs::path val_db   = sd.dir / "val.db";
    const fs::path val_cfg  = sd.dir / "config.val";
    const fs::path hpo_cfg  = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(train_db, 0.0);
    write_minimal_ta_dbfile(val_db,   0.07);
    { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }

    tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    REQUIRE(K > 0);

    ctx.add("DBFILE", train_db.string());
    ctx.add("VALIDATION", val_cfg.string());
    ctx.add("FIXINDEX", "1-" + std::to_string(K));
    for (std::size_t k = 0; k < K; ++k)
        ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));

    {
        std::ofstream os(hpo_cfg);
        os << kOptimizerBlock
           << "LOSS L2\nPC_EMAE 0 1\n"
           << "OPTIM FIXWEIGHT (1-" << K << ") -1.0 1.0\n";
    }

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, true, true));
    Host host(ctx, trg, val_cfg.string());

    ParamVec p(static_cast<long>(K));
    for (std::size_t k = 0; k < K; ++k)
        p(static_cast<long>(k)) = 0.05 * static_cast<double>(k + 1);
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = host(p));
    CHECK(std::isfinite(loss));

    const auto* tr = Host::tls_trainer_for_test();
    REQUIRE(tr != nullptr);
    REQUIRE(tr->model->is_trained());
    const auto& w = tr->model->get_weights();
    REQUIRE(w.size() == K);
    for (std::size_t k = 0; k < K; ++k)
        CHECK(w[k] == Approx(0.05 * static_cast<double>(k + 1)).epsilon(1e-12));

    /* Fast path: no design matrix was built for the training set. */
    CHECK(tr->dm.layout().blocks.empty());

    Host::reset_globals_for_test();
}

/* Case 3a: no training DBFILE, no --validation; loss from PC_OBSERVABLE. */
TEST_CASE("case 3: no DBFILE, observable-only loss",
          "[fixweight_e2e][.]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ForceRebuildGuard _rebuild;
    ScratchDir sd;

    const fs::path hpo_cfg = sd.dir / "config.hpo";
    tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    REQUIRE(K > 0);

    ctx.add("FIXINDEX", "1-" + std::to_string(K));
    for (std::size_t k = 0; k < K; ++k)
        ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));

    {
        /* GRAD_STRATEGY HYBRID: an unset one resolves to ANALYTICAL with
         * PC_OBSERVABLE, which needs training data (validate_hpo_setup
         * refuses it); HYBRID uses finite differences. */
        std::ofstream os(hpo_cfg);
        os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
              "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\n"
           << "LOSS L2\nFAILSCORE 1e8\n"
           << "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
              "--weight 1 --supercell 2 2 2 --maxiter 50 --maxeval 200 "
              "--hotvar a0 2.7 3.9 3.3\n"
           << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
    }

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, false));
    Host host(ctx, trg, /*val_cfg=*/"");
    CHECK(host.train_size() == 0u);
    CHECK(host.val_size() == 0u);

    ParamVec p(1);
    p(0) = 0.01;
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = host(p));
    /* The low-expressivity fixture pot may not relax to a physical Ecoh;
     * both a genuine loss and the FAIL_SCORE clamp are valid outcomes —
     * what matters is that the no-data pipeline completes. */
    CHECK(std::isfinite(loss));

    const auto* tr = Host::tls_trainer_for_test();
    REQUIRE(tr != nullptr);
    REQUIRE(tr->model->is_trained());
    const auto& w = tr->model->get_weights();
    REQUIRE(w.size() == K);
    CHECK(w[0] == Approx(0.01).epsilon(1e-12));  // OPTIM'd
    CHECK(w[1] == Approx(0.02).epsilon(1e-12));  // pinned

    Host::reset_globals_for_test();
}

/* Case 3b: no training DBFILE, --validation present; loss from val metrics. */
TEST_CASE("case 3: no DBFILE, validation-metric loss",
          "[fixweight_e2e][.]")
{
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ForceRebuildGuard _rebuild;
    ScratchDir sd;

    const fs::path val_db  = sd.dir / "val.db";
    const fs::path val_cfg = sd.dir / "config.val";
    const fs::path hpo_cfg = sd.dir / "config.hpo";
    write_minimal_ta_dbfile(val_db, 0.07);
    { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }

    tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    REQUIRE(K > 0);

    ctx.add("VALIDATION", val_cfg.string());
    ctx.add("FIXINDEX", "1-" + std::to_string(K));
    for (std::size_t k = 0; k < K; ++k)
        ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));

    {
        std::ofstream os(hpo_cfg);
        os << kOptimizerBlock
           << "LOSS L2\nPC_EMAE 0 1\n"
           << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
    }

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, true));
    Host host(ctx, trg, val_cfg.string());
    CHECK(host.train_size() == 0u);
    REQUIRE(host.val_size() == 1u);

    /* The loss must respond to the OPTIM'd weight: predictions on the
     * validation set change when w1 changes. */
    ParamVec p(1);
    p(0) = 0.01;
    double loss_a = 0.0, loss_b = 0.0;
    REQUIRE_NOTHROW(loss_a = host(p));
    p(0) = 0.5;
    REQUIRE_NOTHROW(loss_b = host(p));
    CHECK(std::isfinite(loss_a));
    CHECK(std::isfinite(loss_b));
    CHECK(loss_a != loss_b);

    Host::reset_globals_for_test();
}

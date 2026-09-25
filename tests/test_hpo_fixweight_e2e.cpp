// FIXWEIGHT end-to-end host tests: mixed LS/fixed/OPTIM weights (case 2),
// no-training-DBFILE operation (case 3), best_pot precision and exact
// CONFIG/WARM seed evaluation ([seed_anchor]).
// Best-potential persistence failures and the run's exit status
// ([durable_e2e]).
//
// Hidden `[.]` like the other HPO_Host e2e cases: operator() populates the
// thread-local trainer cache, and interleaving with other host tests in one
// binary can leave stale descriptor state behind. Run explicitly with:
//   ./test_hpo "[fixweight_e2e]"
// Every test here sets TADAH_HPO_FORCE_REBUILD for hygiene.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/io_retry.h>
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

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
         * weight equals the optimiser's live value EXACTLY (the Context
         * stores values losslessly). */
        CHECK(w[K - 2] == pinned_val);
        CHECK(w[K - 1] == v);

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
        CHECK(fw[1] == v);
        std::vector<double> wpot(pot.size("WEIGHTS"));
        pot.get("WEIGHTS", wpot);
        REQUIRE(wpot.size() == K);
        CHECK(wpot[K - 1] == v);
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
        std::ofstream os(hpo_cfg);
        os << kOptimizerBlock
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

/* Helpers shared by the end-to-end cases below. */
namespace {

tadah::core::Context make_validation_context(const fs::path& dir)
{
    tadah::core::Context ctx = make_ta_base_context(dir / "config.hpo");
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    ctx.add("VALIDATION", (dir / "config.val").string());
    ctx.add("FIXINDEX", "1-" + std::to_string(K));
    for (std::size_t k = 0; k < K; ++k)
        ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));
    return ctx;
}

std::string read_text(const fs::path& p)
{
    std::ifstream in(p);
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

std::string line_with(const std::string& text, const std::string& needle)
{
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(needle) != std::string::npos) return line;
    return {};
}

/* "  Total evaluations      : N" */
std::size_t total_evaluations(const std::string& journal)
{
    const std::string l = line_with(journal, "Total evaluations");
    const auto colon = l.find(':');
    return colon == std::string::npos
               ? 0 : static_cast<std::size_t>(std::stoul(l.substr(colon + 1)));
}

}  // namespace

/* The live best_pot.tadah the host publishes. Every non-integer
 * number is written with NUMERIC significant digits -- default 18, so the
 * file re-parses to the identical doubles the evaluation used -- and
 * NUMERIC itself is never written (the 1.5.0 writer's
 * setprecision(get("NUMERIC")) default-inserted and wrote 'NUMERIC 12').
 * An explicit NUMERIC 12 is honoured in the file only: the evaluation is
 * independent of it. The machine-readable header is unchanged. */
TEST_CASE("best_pot.tadah numbers follow NUMERIC and never carry it",
          "[fixweight_e2e][lossless][.]")
{
    using tadah::core::format_sig;
    tadah::core::PeriodicTable::initialize();

    struct Run {
        double loss = 0.0;
        std::string text;             // whole best_pot.tadah
        tadah::core::Context pot;     // its body, parsed like WARM does
        std::vector<double> weights;  // the model's live weights
        std::optional<tadah::hpo::durable_best::Key> key;
    };
    const double w1 = 1.0 / 3.0;      // the OPTIM'd weight: not 12-digit exact

    auto run_one = [&](int numeric /* 0 = absent */) {
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
        REQUIRE(K > 1);
        ctx.add("VALIDATION", val_cfg.string());
        ctx.add("FIXINDEX", "1-" + std::to_string(K));
        for (std::size_t k = 0; k < K; ++k)
            ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));
        if (numeric > 0) ctx.add("NUMERIC", numeric);
        {
            std::ofstream os(hpo_cfg);
            os << kOptimizerBlock
               << "LOSS L2\nPC_EMAE 0 1\n"
               << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
        }

        tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
        REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, true));
        Host host(ctx, trg, val_cfg.string());

        Run out;
        ParamVec p(1);
        p(0) = w1;
        REQUIRE_NOTHROW(out.loss = host(p));
        REQUIRE(std::isfinite(out.loss));
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        const auto& w = tr->model->get_weights();
        out.weights.assign(w.begin(), w.end());

        std::ifstream in(sd.dir / "best_pot.tadah");
        REQUIRE(in.good());
        out.text.assign(std::istreambuf_iterator<char>(in),
                        std::istreambuf_iterator<char>());
        std::istringstream lines(out.text);
        std::string line, body;
        while (std::getline(lines, line))
            if (!line.empty() && line[0] != '#') body += line + "\n";
        out.pot.deserialize(std::vector<char>(body.begin(), body.end()));
        out.key = tadah::hpo::durable_best::read_key_file(
            (sd.dir / "best_pot.tadah").string());
        Host::reset_globals_for_test();
        return out;
    };

    auto token_of = [](const std::string& text, const std::string& key) {
        std::istringstream lines(text);
        std::string line, k, v;
        while (std::getline(lines, line)) {
            std::istringstream ls(line);
            if (ls >> k && k == key && ls >> v) return v;
        }
        return std::string();
    };
    auto same_bits = [](double a, double b) {
        return std::memcmp(&a, &b, sizeof a) == 0;
    };
    auto has_numeric_line = [](const std::string& text) {
        std::istringstream lines(text);
        std::string line, k;
        while (std::getline(lines, line)) {
            std::istringstream ls(line);
            if (ls >> k && k == "NUMERIC") return true;
        }
        return false;
    };

    const Run exact = run_one(0);
    CHECK_FALSE(has_numeric_line(exact.text));
    /* Numbers carry 18 significant digits and re-parse bit-exactly. */
    CHECK(token_of(exact.text, "FIXWEIGHT") == format_sig(w1, 18));
    CHECK(token_of(exact.text, "LAMBDA") == format_sig(1e-3, 18));
    std::vector<double> fw(exact.pot.size("FIXWEIGHT"));
    exact.pot.get("FIXWEIGHT", fw);
    REQUIRE(fw.size() == exact.weights.size());
    CHECK(same_bits(fw[0], w1));
    for (std::size_t k = 1; k < fw.size(); ++k)
        CHECK(same_bits(fw[k], 0.01 * static_cast<double>(k + 1)));
    std::vector<double> wpot(exact.pot.size("WEIGHTS"));
    exact.pot.get("WEIGHTS", wpot);
    REQUIRE(wpot.size() == exact.weights.size());
    for (std::size_t k = 0; k < wpot.size(); ++k)
        CHECK(same_bits(wpot[k], exact.weights[k]));
    CHECK(same_bits(exact.weights[0], w1));   // the evaluated weight itself
    /* The machine-readable header still round-trips the loss exactly. */
    REQUIRE(exact.key.has_value());
    CHECK(same_bits(exact.key->loss, exact.loss));

    const Run twelve = run_one(12);
    CHECK_FALSE(has_numeric_line(twelve.text));
    CHECK(token_of(twelve.text, "FIXWEIGHT") == format_sig(w1, 12));
    CHECK(token_of(twelve.text, "FIXWEIGHT") == "0.333333333333");
    /* The file is limited; the evaluation is not. */
    CHECK(same_bits(twelve.loss, exact.loss));
    REQUIRE(twelve.weights.size() == exact.weights.size());
    for (std::size_t k = 0; k < exact.weights.size(); ++k)
        CHECK(same_bits(twelve.weights[k], exact.weights[k]));
    REQUIRE(twelve.key.has_value());
    CHECK(same_bits(twelve.key->loss, twelve.loss));
}

/* ---------------------------------------------------------------------------
 * A CONFIG or WARM seed of an OPTIM'd dimension is evaluated at EXACTLY
 * its loaded double, whatever the box. Under OPTIM_SCALE AUTO the seed goes
 * x0 -> u0 = (x0 - lo)/(hi - lo) -> lo + u0*(hi - lo), which is not the
 * identity (a small weight in a wide explicit box came back 3e-11 off, a
 * FLOOR-box grid centre 1 ulp off); run_inner() now anchors u0 to x0. Each
 * run: NLOPT:LN_SBPLX, MAXEVAL 1 -- the one evaluation is the seed -- then
 * the live model weights and best_pot.tadah's FIXWEIGHT are compared with
 * the seed bit for bit, and the plain map is shown to miss at least one.
 * ------------------------------------------------------------------------- */
namespace {

/* Small, mid-size and 1e-14 weights that the plain map rounds in a
 * [-300, 300] box; 1.2 also rounds in a LOG box. */
const std::vector<double> kAnchorSeed = {
    -0.000287486722631648, 108.326725555511771, -5.55896250563770522,
    1.2345678901234568e-14, 1.2};

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

std::string fixweight_line(const std::vector<double>& w)
{
    std::string s = "FIXWEIGHT";
    for (double v : w) s += " " + tadah::core::format_sig(v, 18);
    return s + "\n";
}

std::vector<double> pot_fixweight(const std::string& text)
{
    std::istringstream lines(text);
    std::string line, body;
    while (std::getline(lines, line))
        if (!line.empty() && line[0] != '#') body += line + "\n";
    tadah::core::Context pot;
    pot.deserialize(std::vector<char>(body.begin(), body.end()));
    std::vector<double> fw(pot.size("FIXWEIGHT"));
    pot.get("FIXWEIGHT", fw);
    return fw;
}

struct AnchorRun {
    std::vector<double> weights;     // the evaluated model's weights
    std::vector<double> pot_fw;      // best_pot.tadah FIXWEIGHT, re-parsed
    std::string         pot_text;
    std::size_t         plain_misses = 0;   // seeds the plain map would round
};

/* One MAXEVAL-1 run through run_inner(). `config_fw` is the FIXWEIGHT line
 * of the configuration (the CONFIG seed); `warm_text`, when non-empty, is
 * written to seed.tadah and the run starts from INIT STRATEGY WARM. */
AnchorRun run_anchor_case(const std::string& optim_lines,
                          const std::vector<double>& config_fw,
                          const std::string& warm_text,
                          const std::vector<double>& expect_seed)
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
    std::string init;
    if (!warm_text.empty()) {
        { std::ofstream os(sd.dir / "seed.tadah"); os << warm_text; }
        init = "INIT\nSTRATEGY WARM " + (sd.dir / "seed.tadah").string() +
               "\nWARM_ON_MISMATCH error\nENDINIT\n";
    }
    {
        std::ofstream os(hpo_cfg);
        os << "OPTIMIZER\n" << init
           << "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
           << "LOSS L2\nPC_EMAE 0 1\n" << optim_lines;
    }

    tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
    const std::size_t K = tadah::hpo::derive_weight_count(ctx);
    REQUIRE(K == config_fw.size());
    ctx.add("VALIDATION", val_cfg.string());
    ctx.add("FIXINDEX", "1-" + std::to_string(K));
    for (double v : config_fw) ctx.add("FIXWEIGHT", v);

    tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, true));
    REQUIRE(trg.spec.optim_init.size() == 1u);
    REQUIRE(trg.space());
    const tadah::hpo::SearchSpace& ss = *trg.space();
    REQUIRE(ss.dim() == expect_seed.size());

    AnchorRun out;
    for (std::size_t i = 0; i < ss.dim(); ++i) {
        CAPTURE(i);
        CHECK(bits_equal(trg.spec.optim_init.front()[i], expect_seed[i]));
        if (!bits_equal(ss.to_real(i, ss.to_internal(i, expect_seed[i])), expect_seed[i]))
            ++out.plain_misses;
    }

    Host host(ctx, trg, val_cfg.string());
    REQUIRE_NOTHROW(tadah::hpo::run_inner<ParamVec, ::LammpsRunner>(trg.spec, host));

    const auto* tr = Host::tls_trainer_for_test();
    REQUIRE(tr != nullptr);
    const auto& w = tr->model->get_weights();
    out.weights.assign(w.begin(), w.end());
    out.pot_text = read_text(sd.dir / "best_pot.tadah");
    REQUIRE_FALSE(out.pot_text.empty());
    out.pot_fw = pot_fixweight(out.pot_text);
    Host::reset_globals_for_test();
    return out;
}

void check_seed_evaluated(const AnchorRun& r, const std::vector<double>& seed)
{
    REQUIRE(r.weights.size() == seed.size());
    REQUIRE(r.pot_fw.size() == seed.size());
    for (std::size_t k = 0; k < seed.size(); ++k) {
        CAPTURE(k, seed[k], r.weights[k], r.pot_fw[k]);
        CHECK(bits_equal(r.weights[k], seed[k]));
        CHECK(bits_equal(r.pot_fw[k], seed[k]));
    }
}

}  // namespace

TEST_CASE("seed anchor: a CONFIG seed is evaluated exactly in every box kind",
          "[fixweight_e2e][seed_anchor][.]")
{
    struct Box { const char* name; std::string optim; std::size_t min_misses; };
    const std::vector<Box> boxes = {
        {"explicit wide LIN box",
         "OPTIM FIXWEIGHT (1-5) -300.0 300.0\n", 4},
        {"AUTO FRAC 0 FLOOR 0.5 + explicit LOG box",
         "OPTIM FIXWEIGHT (1-4) AUTO FRAC 0 FLOOR 0.5\n"
         "OPTIM FIXWEIGHT (5) 0.5 2.0 LOG\n", 1},
        {"AUTO GROUP_FLOOR + AUTO RATIO LOG",
         "OPTIM FIXWEIGHT (1-4) AUTO FRAC 0.5 GROUP_FLOOR 0.1\n"
         "OPTIM FIXWEIGHT (5) AUTO RATIO 1.5 LOG\n", 1},
        {"explicit ASINH box",
         "OPTIM FIXWEIGHT (1-5) -300.0 300.0 ASINH 0.01\n", 1},
    };
    for (const auto& b : boxes) {
        DYNAMIC_SECTION(b.name) {
            const AnchorRun r = run_anchor_case(b.optim, kAnchorSeed, "", kAnchorSeed);
            CHECK(r.plain_misses >= b.min_misses);
            check_seed_evaluated(r, kAnchorSeed);
        }
    }
}

TEST_CASE("seed anchor: a WARM seed is evaluated exactly and re-seeding is a no-op",
          "[fixweight_e2e][seed_anchor][.]")
{
    const std::vector<double> config_fw = {0.01, 0.02, 0.03, 0.04, 0.05};
    const std::string optim = "OPTIM FIXWEIGHT (1-5) -300.0 300.0\n";
    const AnchorRun first = run_anchor_case(optim, config_fw,
                                            fixweight_line(kAnchorSeed), kAnchorSeed);
    CHECK(first.plain_misses >= 4u);
    check_seed_evaluated(first, kAnchorSeed);

    /* WARM again from the best_pot.tadah the first run wrote. */
    const AnchorRun again = run_anchor_case(optim, config_fw, first.pot_text, kAnchorSeed);
    check_seed_evaluated(again, kAnchorSeed);
    CHECK(pot_fixweight(again.pot_text) == pot_fixweight(first.pot_text));
}

/* The same through the real executable (hpo_run -> policy -> run_inner):
 * `tadah hpo` WARM-started from a seed file with MAXEVAL 1 writes a
 * best_pot.tadah whose OPTIM'd FIXWEIGHT tokens re-parse to the seed.
 * Needs TADAH_BINARY; skipped otherwise. */
TEST_CASE("seed anchor: tadah hpo WARM + MAXEVAL 1 reproduces the seed file",
          "[fixweight_e2e][seed_anchor][.]")
{
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin) {
        SUCCEED("Skipped -- set TADAH_BINARY to the built tadah executable.");
        return;
    }
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd;
    write_minimal_ta_dbfile(sd.dir / "val.db", 0.07);
    {
        std::ofstream os(sd.dir / "config.val");
        os << "DBFILE " << (sd.dir / "val.db").string() << "\n";
    }
    { std::ofstream os(sd.dir / "seed.tadah"); os << fixweight_line(kAnchorSeed); }
    {
        std::ofstream os(sd.dir / "config.hpo");
        os << "OPTIMIZER\nINIT\nSTRATEGY WARM seed.tadah\n"
              "WARM_ON_MISMATCH error\nENDINIT\n"
              "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
              "LOSS L2\nPC_EMAE 0 1\n"
              "OPTIM FIXWEIGHT (1-5) -300.0 300.0\n";
    }
    {
        tadah::core::Context ctx = make_validation_context(sd.dir);
        for (const char* k : {"HPOTARGET", "VALIDATION",
                              "TYPEMB", "RCTYPEMB", "RCUTMB"})
            ctx.remove(k);
        std::ofstream os(sd.dir / "config");
        os << ctx;
    }
    const std::string cmd =
        "cd '" + sd.dir.string() + "' && TADAH_HPO_FORCE_REBUILD=1 '" + bin +
        "' hpo -c config --hpotarget config.hpo --validation config.val"
        " > tadah.stdout 2> tadah.stderr";
    const int rc = std::system(cmd.c_str());
    INFO(read_text(sd.dir / "tadah.stderr"));
    REQUIRE(WIFEXITED(rc));
    REQUIRE(WEXITSTATUS(rc) == 0);
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(total_evaluations(journal) == 1u);
    const std::vector<double> fw = pot_fixweight(read_text(sd.dir / "best_pot.tadah"));
    REQUIRE(fw.size() == kAnchorSeed.size());
    for (std::size_t k = 0; k < fw.size(); ++k) {
        CAPTURE(k, fw[k], kAnchorSeed[k]);
        CHECK(bits_equal(fw[k], kAnchorSeed[k]));
    }
}

/* NLopt BOBYQA/COBYLA/PRAXIS round the start point before evaluating it
 * (a few ulp off u0; see kSeedAnchorUlps): the seed-anchor window still
 * evaluates -- and writes -- the WARM seed bit for bit. The seed is
 * example 17's FIXWEIGHT (2-11) in explicit [-300, 300] boxes on an
 * 11-weight D2_BP model; DUMP keeps the
 * first evaluation's potential (PRAXIS makes a second one even at
 * MAXEVAL 1). Needs TADAH_BINARY; skipped otherwise. */
TEST_CASE("seed anchor: tadah hpo NLopt BOBYQA/COBYLA/PRAXIS evaluate the WARM "
          "seed exactly", "[fixweight_e2e][seed_anchor][.]")
{
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin) {
        SUCCEED("Skipped -- set TADAH_BINARY to the built tadah executable.");
        return;
    }
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    const std::vector<double> seed = {
        0.0,
        -216.764741590923819, 108.326725555511771, -43.3325794917411713,
        8.56326925583498166,  11.7937235812636061, -8.12615760784936469,
        -26.560116371452601,  -27.7050793839341019, -16.8854852768916643,
        -5.55896250563770522};
    for (const char* algo : {"LN_BOBYQA", "LN_COBYLA", "LN_PRAXIS"}) {
        DYNAMIC_SECTION("NLOPT:" << algo) {
            ScratchDir sd;
            write_minimal_ta_dbfile(sd.dir / "val.db", 0.07);
            {
                std::ofstream os(sd.dir / "config.val");
                os << "DBFILE " << (sd.dir / "val.db").string() << "\n";
            }
            { std::ofstream os(sd.dir / "seed.tadah"); os << fixweight_line(seed); }
            {
                std::ofstream os(sd.dir / "config.hpo");
                os << "OPTIMIZER\nINIT\nSTRATEGY WARM seed.tadah\n"
                      "WARM_ON_MISMATCH error\nENDINIT\n"
                      "LIB NLOPT\nALGO " << algo << "\nMAXEVAL 1\nENDOPTIMIZER\n"
                      "LOSS L2\nPC_EMAE 0 1\n"
                      "OPTIM FIXWEIGHT (2-11) -300.0 300.0\n"
                      "DUMP 1 dump\n";
            }
            {
                tadah::core::Context ctx = make_ta_base_context(sd.dir / "config.hpo");
                for (const char* k : {"HPOTARGET", "TYPE2B", "CGRID2B", "SGRID2B",
                                      "TYPEMB", "RCTYPEMB", "RCUTMB"})
                    ctx.remove(k);
                ctx.add("TYPE2B", "D2_BP");
                ctx.add("TYPE2B", static_cast<int>(seed.size()));
                ctx.add("TYPE2B", static_cast<int>(seed.size()));
                ctx.add("TYPE2B", "Ta");
                ctx.add("TYPE2B", "Ta");
                for (std::size_t k = 0; k < seed.size(); ++k) {
                    ctx.add("CGRID2B", 0.2 + 0.2 * static_cast<double>(k));
                    ctx.add("SGRID2B", 2.0 + 0.3 * static_cast<double>(k));
                }
                REQUIRE(tadah::hpo::derive_weight_count(ctx) == seed.size());
                ctx.add("FIXINDEX", "1-" + std::to_string(seed.size()));
                /* weight 1 is not OPTIM'd: the configuration holds the seed's */
                for (std::size_t k = 0; k < seed.size(); ++k)
                    ctx.add("FIXWEIGHT", k == 0 ? seed[0]
                                                : 0.01 * static_cast<double>(k + 1));
                std::ofstream os(sd.dir / "config");
                os << ctx;
            }
            const std::string cmd =
                "cd '" + sd.dir.string() + "' && TADAH_HPO_FORCE_REBUILD=1 '" + bin +
                "' hpo -c config --hpotarget config.hpo --validation config.val"
                " > tadah.stdout 2> tadah.stderr";
            const int rc = std::system(cmd.c_str());
            INFO(read_text(sd.dir / "tadah.stderr"));
            REQUIRE(WIFEXITED(rc));
            REQUIRE(WEXITSTATUS(rc) == 0);
            const std::string journal = read_text(sd.dir / "hpo_run.log");
            INFO(journal);
            /* the first evaluation IS the seed */
            const std::vector<double> first =
                pot_fixweight(read_text(sd.dir / "dump" / "pot_1.tadah"));
            REQUIRE(first.size() == seed.size());
            for (std::size_t k = 0; k < seed.size(); ++k) {
                CAPTURE(k, first[k], seed[k]);
                CHECK(bits_equal(first[k], seed[k]));
            }
            /* ... and so is best_pot.tadah whenever it is the best one */
            if (line_with(journal, "Best total loss").find("(eval 1)") !=
                std::string::npos) {
                const std::vector<double> fw =
                    pot_fixweight(read_text(sd.dir / "best_pot.tadah"));
                REQUIRE(fw.size() == seed.size());
                for (std::size_t k = 0; k < seed.size(); ++k) {
                    CAPTURE(k, fw[k], seed[k]);
                    CHECK(bits_equal(fw[k], seed[k]));
                }
            } else {
                CHECK(std::string(algo) == "LN_PRAXIS");   // only PRAXIS runs 2
            }
        }
    }
}

/* The failvar and hot-value readbacks are parsed with core::to_double
 * (std::stod's acceptance, but an underflowing literal is a value), like
 * every other reader of numbers: a failvar > 0 still fails the sim and a
 * hot value outside its range is still rejected; graded FAILSCORE puts a
 * failed evaluation in [0.999*FAILSCORE, FAILSCORE). A subnormal cannot
 * reach these parses from LAMMPS today: compute_equal flushes |v| < DBL_MIN
 * to 0 and LAMMPS' literal parser rejects e.g. 1e-310 (pinned in
 * test_lammps_variable_readback.cpp), so the subnormal case is checked as
 * LAMMPS delivers it. */
TEST_CASE("PC_LAMMPS readback: failvar violations and hot-value range checks",
          "[fixweight_e2e][readback][.]")
{
    struct Case {
        const char* name;
        std::string flags;      // --failvar / --hotvar part of PC_LAMMPS
        std::string tail;       // script lines after `run 0`
        bool fails;
    };
    const std::vector<Case> cases = {
        {"failvar 0", "--failvar fv", "variable fv equal 0\n", false},
        {"failvar 1e-3", "--failvar fv", "variable fv equal 1e-3\n", true},
        {"failvar with a subnormal result (flushed to 0 by LAMMPS)",
         "--failvar fv", "variable fv equal 1e-200*1e-110\n", false},
        {"hot value in range", "--hotvar hot 0.0 1.0 0.5",
         "variable hot equal 0.25\n", false},
        {"hot value below its range", "--hotvar hot 0.5 1.0 0.75",
         "variable hot equal 0.25\n", true},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            tadah::core::PeriodicTable::initialize();
            Host::reset_globals_for_test();
            ForceRebuildGuard _rebuild;
            ScratchDir sd;
            const fs::path val_db  = sd.dir / "val.db";
            const fs::path val_cfg = sd.dir / "config.val";
            const fs::path hpo_cfg = sd.dir / "config.hpo";
            const fs::path script  = sd.dir / "probe.in";
            write_minimal_ta_dbfile(val_db, 0.07);
            { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }
            {
                std::ofstream os(script);
                os << "units metal\nboundary p p p\natom_style atomic\n"
                      "lattice bcc 3.301\n"
                      "region box block 0 2 0 2 0 2 units lattice\n"
                      "create_box 1 box\ncreate_atoms 1 region box\n"
                      "mass 1 180.94788\npair_style tadah\n"
                      "pair_coeff * * EXTERN Ta\n"
                      "run 0\n"
                      "variable e equal pe/atoms\n"
                   << c.tail;
            }
            {
                std::ofstream os(hpo_cfg);
                os << "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
                      "LOSS L2\nFAILSCORE 1e8\nFAILSCORE_GRADING ON\n"
                      "PC_LAMMPS --script " << script.string()
                   << " --varloss e 0 1 " << c.flags << "\n"
                      "OPTIM FIXWEIGHT (5) -300.0 300.0\n";
            }
            tadah::core::Context ctx = make_ta_base_context(hpo_cfg);
            const std::size_t K = tadah::hpo::derive_weight_count(ctx);
            ctx.add("VALIDATION", val_cfg.string());
            ctx.add("FIXINDEX", "1-" + std::to_string(K));
            for (std::size_t k = 0; k < K; ++k)
                ctx.add("FIXWEIGHT", 0.01 * static_cast<double>(k + 1));
            tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
            REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, true));
            REQUIRE(trg.LS.size() == 1u);
            Host host(ctx, trg, val_cfg.string());
            ParamVec p(1);
            p(0) = 0.05;
            double f = std::numeric_limits<double>::quiet_NaN();
            REQUIRE_NOTHROW(f = host(p));
            CAPTURE(f);
            REQUIRE(std::isfinite(f));
            if (c.fails) CHECK(f >= 0.999e8);
            else         CHECK(f < 0.999e8);
            Host::reset_globals_for_test();
        }
    }
}

/* ---------------------------------------------------------------------------
 * End to end: a best-potential persistence failure fails the RUN --
 * "FAILED -- best-potential persistence: ..." in hpo_run.log, the error
 * reaches the caller (non-zero exit) -- while a transient failure that
 * clears within the retry budget leaves only a DIAGNOSTICS line. NLopt used
 * to turn the durable error into "nlopt failure" = graceful, exit code 0.
 *
 * hpo_run() runs in a re-executed copy of this binary: the fault-injection
 * and backoff environment knobs are read once per process, and hpo_run()
 * installs process-wide state (signal handlers, the script executor).
 * Run explicitly with:  ./test_hpo "[durable_e2e]"
 * ------------------------------------------------------------------------- */
extern char** environ;

namespace {

/* Case 3b (no DBFILE, validation-metric loss) under NLOPT:LN_SBPLX. */
constexpr const char* kNloptBlock =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\nENDOPTIMIZER\n";

void write_durable_fixture(const fs::path& dir)
{
    write_minimal_ta_dbfile(dir / "val.db", 0.07);
    {
        std::ofstream os(dir / "config.val");
        os << "DBFILE " << (dir / "val.db").string() << "\n";
    }
    std::ofstream os(dir / "config.hpo");
    os << kNloptBlock
       << "LOSS L2\nPC_EMAE 0 1\n"
       << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
}

std::size_t data_rows(const fs::path& p)
{
    std::ifstream in(p);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line))
        if (!line.empty() && line[0] != '#') ++n;
    return n;
}

bool has_key_line(const std::string& text, const std::string& key)
{
    std::istringstream lines(text);
    std::string line, k;
    while (std::getline(lines, line)) {
        std::istringstream ls(line);
        if (ls >> k && k == key) return true;
    }
    return false;
}

struct ChildRun { int status = -1; std::string output; };

/* Start this binary on `test_spec` in `cwd` with `extra_env` ("K=V")
 * added; its output goes to cwd/child.out. Everything is prepared before
 * fork(); the child then only makes async-signal-safe calls (another thread
 * of this process may hold a lock). */
pid_t spawn_self(const std::string& test_spec, const fs::path& cwd,
                 const std::vector<std::string>& extra_env)
{
    std::vector<std::string> env_store;
    for (char** e = environ; *e; ++e) {
        const std::string kv(*e);
        bool replaced = false;
        for (const auto& x : extra_env)
            if (kv.compare(0, x.find('=') + 1, x, 0, x.find('=') + 1) == 0)
                replaced = true;
        if (!replaced) env_store.push_back(kv);
    }
    env_store.insert(env_store.end(), extra_env.begin(), extra_env.end());
    std::vector<char*> envp;
    for (auto& s : env_store) envp.push_back(s.data());
    envp.push_back(nullptr);
    std::string arg0 = "test_hpo", arg1 = test_spec;
    char* argv[] = {arg0.data(), arg1.data(), nullptr};
    const std::string dir = cwd.string();
    const std::string out = (cwd / "child.out").string();

    std::cout.flush();
    std::fflush(nullptr);   // or the child re-emits our buffered output
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        const int fd = ::open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) ::_exit(126);
        ::dup2(fd, 1);
        ::dup2(fd, 2);
        if (::chdir(dir.c_str()) != 0) ::_exit(125);
        ::execve("/proc/self/exe", argv, envp.data());
        ::_exit(127);
    }
    return pid;
}

/* spawn_self() and wait for it. */
ChildRun run_self(const std::string& test_spec, const fs::path& cwd,
                  const std::vector<std::string>& extra_env)
{
    const pid_t pid = spawn_self(test_spec, cwd, extra_env);
    ChildRun r;
    REQUIRE(::waitpid(pid, &r.status, 0) == pid);
    r.output = read_text(cwd / "child.out");
    return r;
}

const char* const kFaultPersistent =
    "TADAH_HPO_TEST_IO_FAULT=durable.lock_open,EBUSY,skip=1,count=always";
const char* const kShortBackoff = "TADAH_HPO_TEST_IO_BACKOFF_MS=5,20,200,0";

}  // namespace

/* The re-executed half: hpo_run() on the fixture in the working directory.
 * TADAH_TEST_DURABLE_E2E_EXPECT=throw|fail|ok|any says which outcome is
 * required (throw: a durable_best::Error; fail: any exception; any: the
 * parent inspects the journal).
 * TADAH_TEST_DURABLE_E2E_STALE_STATE=1 first arms the host's process-wide
 * stagnation threshold and top-K tracking, as a policy that ended by an
 * exception in an earlier run of this process would leave them. */
TEST_CASE("durable e2e child: hpo_run in the working directory",
          "[.durable_e2e_child]")
{
    tadah::core::PeriodicTable::initialize();
    const char* expect = std::getenv("TADAH_TEST_DURABLE_E2E_EXPECT");
    REQUIRE(expect != nullptr);
    if (const char* stale = std::getenv("TADAH_TEST_DURABLE_E2E_STALE_STATE");
        stale && std::string(stale) == "1") {
        Host::set_stagnation_threshold(1);
        Host::set_track_topk(2);
    }
    tadah::core::Context ctx = make_validation_context(fs::current_path());
    if (std::string(expect) == "throw") {
        CHECK_THROWS_AS(tadah::hpo::hpo_run(ctx),
                        tadah::hpo::durable_best::Error);
    } else if (std::string(expect) == "fail") {
        CHECK_THROWS(tadah::hpo::hpo_run(ctx));
    } else if (std::string(expect) == "any") {
        try { (void)tadah::hpo::hpo_run(ctx); } catch (...) {}
    } else {
        CHECK(tadah::hpo::hpo_run(ctx) == 0);
    }
}

TEST_CASE("durable best: a persistent failure fails the run",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    /* skip=1: the fresh-run reset takes the lock; every later lock open
     * (the first publication, at eval 1) fails with EBUSY until the 200 ms
     * budget is spent. */
    const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                {kFaultPersistent, kShortBackoff,
                                 "TADAH_TEST_DURABLE_E2E_EXPECT=throw",
                                 "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(r.output.find("All tests passed") != std::string::npos);

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    const std::string status = line_with(journal, "Optimiser status");
    CHECK(status.find("FAILED -- best-potential persistence: ") != std::string::npos);
    CHECK(status.find("gave up after") != std::string::npos);
    CHECK(journal.find("converged") == std::string::npos);
    CHECK(journal.find("nlopt failure") == std::string::npos);
    /* Every retried attempt was journalled. */
    CHECK(journal.find("best-potential I/O: durable.lock_open") != std::string::npos);

    /* The failed evaluation still reached loss.tadah: the writer queue
     * stays contiguous and the final flush drains it. */
    const std::size_t n = total_evaluations(journal);
    CHECK(n >= 1);
    CHECK(data_rows(sd.dir / "loss.tadah") == n);
    CHECK_FALSE(fs::exists(sd.dir / "best_pot.tadah"));
}

TEST_CASE("durable best: a transient failure is retried and reported",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const ChildRun r = run_self(
        "[.durable_e2e_child]", sd.dir,
        {"TADAH_HPO_TEST_IO_FAULT=durable.lock_open,EBUSY,skip=1,count=3",
         kShortBackoff, "TADAH_TEST_DURABLE_E2E_EXPECT=ok",
         "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(r.output.find("All tests passed") != std::string::npos);

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(line_with(journal, "Optimiser status").find("FAILED") == std::string::npos);
    CHECK(line_with(journal, "Transient I/O retries").find(
              ": 3 (best-potential 3, per-eval logs/archives 0); rows lost 0; "
              "archives lost 0") != std::string::npos);

    const std::size_t n = total_evaluations(journal);
    CHECK(n == 5);
    CHECK(data_rows(sd.dir / "loss.tadah") == n);
    const std::string pot = read_text(sd.dir / "best_pot.tadah");
    REQUIRE_FALSE(pot.empty());
    CHECK(has_key_line(pot, "FIXWEIGHT"));
    CHECK_FALSE(has_key_line(pot, "NUMERIC"));
}

/* The same through the real executable: `tadah hpo` must exit non-zero.
 * Needs TADAH_BINARY (the built `tadah`); skipped otherwise. */
TEST_CASE("durable best: tadah hpo exits non-zero on a persistence failure",
          "[durable_e2e][failure_handling][.]")
{
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin) {
        SUCCEED("Skipped -- set TADAH_BINARY to the built tadah executable.");
        return;
    }
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    {
        /* The CLI supplies HPOTARGET and VALIDATION from its flags; the
         * in-memory DM_Dummy many-body term (INITMB false) has no file
         * syntax. */
        tadah::core::Context ctx = make_validation_context(sd.dir);
        for (const char* k : {"HPOTARGET", "VALIDATION",
                              "TYPEMB", "RCTYPEMB", "RCUTMB"})
            ctx.remove(k);
        std::ofstream os(sd.dir / "config");
        os << ctx;
    }
    const std::string cmd =
        "cd '" + sd.dir.string() + "' && " + kFaultPersistent + " " +
        kShortBackoff + " TADAH_HPO_FORCE_REBUILD=1 '" + bin +
        "' hpo -c config --hpotarget config.hpo --validation config.val"
        " > tadah.stdout 2> tadah.stderr";
    const int rc = std::system(cmd.c_str());
    INFO(read_text(sd.dir / "tadah.stderr"));
    REQUIRE(WIFEXITED(rc));
    CHECK(WEXITSTATUS(rc) != 0);
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(line_with(journal, "Optimiser status")
              .find("FAILED -- best-potential persistence: ") != std::string::npos);
    CHECK(journal.find("converged") == std::string::npos);
}

/* A shutdown signal during the durable backoff (default 30 s
 * budget) ends the run after at most the 2 s stop grace -- still FAILED,
 * since the candidate was not persisted -- instead of after the whole
 * budget. */
TEST_CASE("durable best: SIGTERM ends the persistence backoff within the grace",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const pid_t pid = spawn_self("[.durable_e2e_child]", sd.dir,
                                 {kFaultPersistent,
                                  "TADAH_TEST_DURABLE_E2E_EXPECT=throw",
                                  "TADAH_HPO_FORCE_REBUILD=1"});
    /* Wait until the child is backing off (its first retry WARNING). */
    const auto journal_path = sd.dir / "hpo_run.log";
    bool backing_off = false;
    for (int i = 0; i < 1200 && !backing_off; ++i) {   // <= 60 s
        backing_off = read_text(journal_path).find(
                          "best-potential I/O: durable.lock_open") != std::string::npos;
        if (!backing_off) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!backing_off) {
        ::kill(pid, SIGKILL);
        int st = 0;
        ::waitpid(pid, &st, 0);
        FAIL("the child never reached the durable backoff");
    }
    const auto t_signal = std::chrono::steady_clock::now();
    REQUIRE(::kill(pid, SIGTERM) == 0);
    int status = 0;
    REQUIRE(::waitpid(pid, &status, 0) == pid);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_signal).count();
    const std::string output = read_text(sd.dir / "child.out");
    INFO(output);
    INFO("signal -> exit: " << seconds << " s");
    CHECK(seconds < 5.0);   // was up to ~28 s (the rest of the budget); now <= the 2 s grace
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    CHECK(output.find("All tests passed") != std::string::npos);

    const std::string journal = read_text(journal_path);
    INFO(journal);
    const std::string st = line_with(journal, "Optimiser status");
    CHECK(st.find("FAILED -- best-potential persistence: ") != std::string::npos);
    CHECK(st.find("(retry abandoned: shutdown requested)") != std::string::npos);
    CHECK(journal.find("is complete") == std::string::npos);
}

/* All per-eval files of one drain share ONE retry budget
 * (it used to be one per file: ~8x the budget on the first failing drain). */
TEST_CASE("per-eval logs: one retry budget per drain, shared by every file",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const ChildRun r = run_self(
        "[.durable_e2e_child]", sd.dir,
        {"TADAH_HPO_TEST_IO_FAULT=writer.append,EIO,always",
         "TADAH_HPO_TEST_IO_BACKOFF_MS=10,200,1000,0",
         "TADAH_TEST_DURABLE_E2E_EXPECT=ok", "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(r.output.find("All tests passed") != std::string::npos);

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    /* MAXEVAL 5 < QFLUSH 10: one drain (the final flush) with rows for all
     * eight files. The failure-path time of all "gave up" lines together
     * stays within about one 1 s budget. */
    double spent = 0.0;
    std::size_t gave_up = 0, spent_lines = 0;
    std::istringstream in(journal);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("HPO writer: ") == std::string::npos) continue;
        if (line.find("(not retried: this drain's retry budget was spent)") !=
            std::string::npos)
            ++spent_lines;
        const auto p = line.find("(gave up after ");
        if (p == std::string::npos) continue;
        ++gave_up;
        const auto over = line.find(" over ", p);
        REQUIRE(over != std::string::npos);
        spent += std::stod(line.substr(over + 6));
    }
    CHECK(gave_up >= 1);
    CHECK(spent < 1.6);
    CHECK(spent_lines >= 1);
    CHECK(gave_up + spent_lines == 8);
    CHECK(line_with(journal, "Optimiser status").find("FAILED") == std::string::npos);
    CHECK(line_with(journal, "Transient I/O retries").find("rows lost") !=
          std::string::npos);
}

/* Once a drain's budget was spent, a file whose permanent error
 * (ENOSPC) still got its one immediate re-attempt was reported as "not
 * retried: this drain's retry budget was spent" right after "permanent
 * error, re-trying once". Slow failures (latency 150 ms) spend the 1 s
 * budget part-way through the eight files. */
TEST_CASE("per-eval logs: a re-attempted permanent error is not 'not retried'",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const ChildRun r = run_self(
        "[.durable_e2e_child]", sd.dir,
        {"TADAH_HPO_TEST_IO_FAULT=writer.append,ENOSPC,always,latency=150",
         "TADAH_HPO_TEST_IO_BACKOFF_MS=10,200,1000,0",
         "TADAH_TEST_DURABLE_E2E_EXPECT=ok", "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(r.output.find("All tests passed") != std::string::npos);
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    std::size_t lost = 0, reattempted = 0;
    std::istringstream in(journal);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("permanent error, re-trying once") != std::string::npos)
            ++reattempted;
        if (line.find("HPO writer: ") != std::string::npos &&
            line.find(" lost: writer.append failed with ENOSPC") != std::string::npos) {
            ++lost;
            CHECK(line.find("(not retried:") == std::string::npos);
            CHECK(line.find("(gave up after 2 attempts over ") != std::string::npos);
        }
    }
    CHECK(lost == 8);
    CHECK(reattempted == 8);
}

/* ---------------------------------------------------------------------------
 * How a run ends -- RESULT status, best-loss report,
 * exit status -- through the real policy stack (hpo_run in a re-executed
 * child, as above). The objective is the durable fixture's one OPTIM'd
 * weight; RESTART refreshes draw from std::random_device, so the checks are
 * structural (order of journal lines), not eval counts.
 * ------------------------------------------------------------------------- */
namespace {

void write_policy_fixture(const fs::path& dir, const std::string& optimizer)
{
    write_durable_fixture(dir);
    std::ofstream os(dir / "config.hpo");
    os << optimizer << "LOSS L2\nPC_EMAE 0 1\n"
       << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";
}

std::vector<std::string> lines_of(const std::string& text)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

bool contains(const std::string& s, const std::string& needle)
{
    return s.find(needle) != std::string::npos;
}

/* "... N eval(s) consumed ..." -> N */
std::size_t evals_consumed(const std::string& line)
{
    const auto p = line.find(" eval(s) consumed");
    REQUIRE(p != std::string::npos);
    auto b = line.rfind(' ', p - 1);
    return static_cast<std::size_t>(std::stoul(line.substr(b + 1, p - b - 1)));
}

ChildRun run_policy_child(const fs::path& dir, const char* expect)
{
    return run_self("[.durable_e2e_child]", dir,
                    {std::string("TADAH_TEST_DURABLE_E2E_EXPECT=") + expect,
                     "TADAH_HPO_FORCE_REBUILD=1"});
}

}  // namespace

/* After a stop request retrying used to stop at once, so ONE
 * transient blip right after SIGINT/SIGTERM/SIGUSR2 (SLURM's time-limit
 * warning) failed the run (rc 1, the in-flight best lost) and dropped the
 * final drain's rows. With the 2 s grace both survive. The child stops
 * itself just before each durable rename (the test-only SIGSTOP hook); at
 * the first stop the parent sends SIGINT, so the stop request is up when
 * that rename fails once (slowly: latency 300 ms), and the final drain's
 * first append fails once too. */
TEST_CASE("durable best: a blip right after SIGINT still saves the best",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const pid_t pid = spawn_self(
        "[.durable_e2e_child]", sd.dir,
        {"TADAH_HPO_TEST_IO_FAULT=durable.rename,EBUSY,count=1,latency=300;"
         "writer.append,EIO,count=1",
         "TADAH_HPO_TEST_STOP_DURABLE_BEST_BEFORE_RENAME=1",
         "TADAH_TEST_DURABLE_E2E_EXPECT=ok", "TADAH_HPO_FORCE_REBUILD=1"});
    int status = 0, stops = 0;
    for (;;) {
        REQUIRE(::waitpid(pid, &status, WUNTRACED) == pid);
        if (!WIFSTOPPED(status)) break;
        if (++stops == 1) REQUIRE(::kill(pid, SIGINT) == 0);
        REQUIRE(::kill(pid, SIGCONT) == 0);
    }
    const std::string output = read_text(sd.dir / "child.out");
    INFO(output);
    CHECK(stops == 2);   // the failed rename and its retry
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    CHECK(contains(output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(contains(line_with(journal, "Optimiser status"), ": SHUTDOWN (SIG"));
    CHECK_FALSE(contains(journal, "retry abandoned"));
    CHECK(contains(journal, "best-potential I/O: durable.rename"));
    CHECK(contains(line_with(journal, "Transient I/O retries"),
                   ": 2 (best-potential 1, per-eval logs/archives 1); rows lost 0; "
                   "archives lost 0"));
    CHECK(line_with(journal, "Best potential saved").find(": best_pot.tadah") !=
          std::string::npos);
    CHECK(contains(read_text(sd.dir / "best_pot.tadah"), "# HPO_SOURCE_EVAL 1"));
    const std::size_t n = total_evaluations(journal);
    CHECK(n >= 1);
    CHECK(data_rows(sd.dir / "loss.tadah") == n);
}

/* An exhausted RESTART budget left whatever an inner run recorded
 * last (COMPLETED, or a stale per-start MAXEVAL_REACHED) as the status. */
TEST_CASE("RESTART: running out of restarts is the run's status",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO LN_SBPLX\n"
        "MAXEVAL 60\nRESTART\nAFTER_STAGNATION 2\nMAX 1\n"
        "STRATEGY RANDOM 1\nENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n");
    const ChildRun r = run_policy_child(sd.dir, "ok");
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    /* A 1-D bowl stagnates long before 60 evaluations. */
    REQUIRE(contains(journal, "RESTART: stagnation reached, max restarts (1) used"));
    CHECK(contains(r.output,
                   "[HPO] RESTART: stagnation reached, max restarts (1) used; stopping"));
    CHECK(line_with(journal, "Optimiser status").find(
              ": STOPPED -- stagnation: no new best in 2 evals after 1 restart(s)")
          != std::string::npos);
    CHECK(total_evaluations(journal) < 120u);
}

/* Under RESTART the first stagnating start restarted the whole
 * multistart, so starts 2..K practically never ran. Now every launch runs
 * all K starts, each in its own window, and a restart follows the last. */
TEST_CASE("MULTISTART under RESTART: every launch runs all its starts",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nINIT\nSTRATEGY LHS 3\nSEED 11\nENDINIT\n"
        "STRATEGY\nTYPE MULTISTART\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 45\n"
        "N_STARTS 3\nRESTART\nAFTER_STAGNATION 3\nMAX 1\n"
        "STRATEGY LHS 3\nENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n");
    const ChildRun r = run_policy_child(sd.dir, "ok");
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    /* Walk the journal: within a launch the starts run 1, 2, 3, each ends
     * with its "done" line, and a RESTART only follows a finished 3/3. */
    std::size_t next_start = 1, launches = 0, restarts = 0, stagnated = 0;
    bool in_start = false;
    for (const std::string& l : lines_of(journal)) {
        if (contains(l, "MULTISTART start ") && contains(l, " -- running inner optimiser")) {
            CHECK_FALSE(in_start);
            CHECK(contains(l, "MULTISTART start " + std::to_string(next_start) + "/3"));
            if (next_start == 1) ++launches;
            in_start = true;
        } else if (contains(l, "MULTISTART start ") && contains(l, " done: ")) {
            CHECK(in_start);
            CHECK(contains(l, "MULTISTART start " + std::to_string(next_start) + "/3"));
            if (contains(l, "stopped on stagnation")) ++stagnated;
            in_start = false;
            next_start = next_start % 3 + 1;
        } else if (contains(l, "INFO") && contains(l, "RESTART 1/1 -- fresh init")) {
            CHECK_FALSE(in_start);
            CHECK(next_start == 1);   // the batch finished: 3/3 done
            ++restarts;
        }
    }
    CHECK(next_start == 1);
    CHECK(launches == 2);
    CHECK(restarts == 1);
    CHECK(stagnated >= 2);
    CHECK(contains(journal, "MULTISTART: "));
    CHECK(contains(journal, "start(s) stopped on stagnation; reporting stagnation to RESTART"));
    CHECK(line_with(journal, "Optimiser status").find(
              ": STOPPED -- stagnation: no new best in 3 evals after 1 restart(s)")
          != std::string::npos);
}

/* After a stagnating PRESTAGE the refinement's first evaluation
 * re-fired StagnationStop (the window was never reset), so no seed was
 * refined and RESTART re-ran the exploration at once. */
TEST_CASE("PRESTAGE under RESTART: every refinement seed gets a fair run",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_SBPLX\n"
        "MAXEVAL 30\nTOPK 3\nEXPLORE\nLIB NLOPT\nALGO GN_CRS2_LM\n"
        "MAXEVAL 40\nSEED 1717\nENDEXPLORE\nRESTART\nAFTER_STAGNATION 4\n"
        "MAX 1\nSTRATEGY RANDOM 1\nENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n");
    const ChildRun r = run_policy_child(sd.dir, "ok");
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    REQUIRE(contains(journal, "REFINEMENT: refining top-3 seed(s)"));
    std::size_t done = 0, done_before_restart = 0;
    bool restarted = false;
    for (const std::string& l : lines_of(journal)) {
        if (contains(l, "MULTISTART start ") && contains(l, " done: ")) {
            ++done;
            if (!restarted) ++done_before_restart;
            CAPTURE(l);
            CHECK(evals_consumed(l) > 1u);   // was 1 (or 0) per seed
        } else if (contains(l, "INFO") && contains(l, "RESTART 1/1 -- fresh init")) {
            restarted = true;
        }
    }
    /* The whole first refinement ran before any restart. */
    CHECK(done_before_restart == 3u);
    CHECK(done >= 3u);
    const std::string status = line_with(journal, "Optimiser status");
    CHECK(status.find("FAILED") == std::string::npos);
    if (contains(journal, "max restarts (1) used"))
        CHECK(contains(status, "STOPPED -- stagnation: no new best in 4 evals "
                               "after 1 restart(s)"));
}

/* A MULTISTART whose every start failed ended COMPLETED with exit
 * code 0 (the same error under PLAIN fails the run). */
TEST_CASE("MULTISTART: every start failing fails the run",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    /* No analytical derivative is registered for FIXWEIGHT: each start's
     * first gradient request throws. */
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nINIT\nSTRATEGY LHS 3\nSEED 11\nENDINIT\n"
        "STRATEGY\nTYPE MULTISTART\nLIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 30\n"
        "N_STARTS 3\nGRAD_STRATEGY ANALYTICAL\nENDSTRATEGY\nENDOPTIMIZER\n");
    const ChildRun r = run_policy_child(sd.dir, "fail");
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));
    /* Each start's failure also reaches the console. */
    CHECK(contains(r.output, "[HPO] WARNING: MULTISTART start 3/3 failed: "
                             "GRAD_STRATEGY=ANALYTICAL"));
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    for (int i = 1; i <= 3; ++i)
        CHECK(contains(journal, "MULTISTART start " + std::to_string(i) +
                                "/3 failed: GRAD_STRATEGY=ANALYTICAL"));
    CHECK(contains(line_with(journal, "Optimiser status"),
                   ": FAILED -- GRAD_STRATEGY=ANALYTICAL"));
}

/* The stagnation threshold and top-K tracking are process-wide
 * statics; one left armed (a policy that ended by an exception) stopped
 * the next PLAIN run in the process as FAILED. hpo_run disarms both. */
TEST_CASE("hpo_run starts with the stagnation window disarmed",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);   // PLAIN NLOPT:LN_SBPLX, MAXEVAL 5
    const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                {"TADAH_TEST_DURABLE_E2E_EXPECT=ok",
                                 "TADAH_TEST_DURABLE_E2E_STALE_STATE=1",
                                 "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(total_evaluations(journal) == 5u);
    CHECK_FALSE(contains(journal, "StagnationStop fired"));
    CHECK(line_with(journal, "Optimiser status").find("FAILED") == std::string::npos);
}

/* RESULT claimed "(none -- no evaluation produced a usable loss)"
 * for a first best that could not be persisted, and "(none written)" while
 * its renamed file sat on disk unsynced. */
TEST_CASE("RESULT names a best that could not be persisted",
          "[durable_e2e][failure_handling][.]")
{
    struct Case {
        const char* fault;
        bool        file_expected;
        const char* saved_line;
    };
    const std::vector<Case> cases = {
        /* The rename lands, the directory fsync keeps failing (EIO). */
        {"TADAH_HPO_TEST_IO_FAULT=durable.dir_fsync,EIO,always", true,
         "not confirmed -- best_pot.tadah may hold eval 1 (write not synced) "
         "or be absent; check its HPO_SOURCE_EVAL header"},
        /* The rename itself fails on a read-only filesystem: it provably
         * did not happen. */
        {"TADAH_HPO_TEST_IO_FAULT=durable.rename,EROFS,always", false,
         ": (none written)"},
        /* The rename fails with an errno it can report after landing (a
         * network filesystem): the file may hold the candidate. */
        {"TADAH_HPO_TEST_IO_FAULT=durable.rename,EIO,always", false,
         "not confirmed -- best_pot.tadah may hold eval 1 (write not synced) "
         "or be absent"},
        /* Nothing reaches the rename (the temporary cannot be written). */
        {"TADAH_HPO_TEST_IO_FAULT=durable.write,ENOSPC,always", false,
         ": (none written)"},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.fault) {
            ScratchDir sd;
            write_durable_fixture(sd.dir);
            const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                        {c.fault, kShortBackoff,
                                         "TADAH_TEST_DURABLE_E2E_EXPECT=throw",
                                         "TADAH_HPO_FORCE_REBUILD=1"});
            INFO(r.output);
            REQUIRE(WIFEXITED(r.status));
            REQUIRE(WEXITSTATUS(r.status) == 0);
            REQUIRE(contains(r.output, "All tests passed"));
            const std::string journal = read_text(sd.dir / "hpo_run.log");
            INFO(journal);
            CHECK(contains(line_with(journal, "Optimiser status"),
                           "FAILED -- best-potential persistence: "));
            const std::string best = line_with(journal, "Best total loss");
            CHECK(contains(best, "(eval 1) -- NOT saved (best-potential "
                                 "persistence failed)"));
            CHECK_FALSE(contains(journal, "no evaluation produced a usable loss"));
            CHECK(line_with(journal, "Last saved best").empty());
            CHECK(contains(line_with(journal, "Best potential saved"), c.saved_line));
            CHECK(fs::exists(sd.dir / "best_pot.tadah") == c.file_expected);
            if (c.file_expected)
                CHECK(contains(read_text(sd.dir / "best_pot.tadah"),
                               "# HPO_SOURCE_EVAL 1"));
        }
    }
}

/* A later best whose directory fsync fails: RESULT used to name
 * best_pot.tadah with the older best's loss while the file held the newer,
 * unsynced candidate. */
TEST_CASE("RESULT distinguishes the last saved best from the unsaved one",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    /* The first best publishes (1 directory fsync); every later one fails. */
    const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                {"TADAH_HPO_TEST_IO_FAULT=durable.dir_fsync,EIO,"
                                 "skip=1,always",
                                 kShortBackoff, "TADAH_TEST_DURABLE_E2E_EXPECT=any",
                                 "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));
    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    const std::string status = line_with(journal, "Optimiser status");
    if (!contains(status, "FAILED")) {
        /* MAXEVAL 5 produced a single new best: nothing to check. */
        WARN("no second new best within MAXEVAL 5");
        return;
    }
    const std::string best = line_with(journal, "Best total loss");
    const std::string last = line_with(journal, "Last saved best");
    const std::string saved = line_with(journal, "Best potential saved");
    CHECK(contains(best, "-- NOT saved"));
    REQUIRE(contains(last, "(eval "));
    const auto eval_of = [](const std::string& l) {
        const auto p = l.find("(eval ");
        return std::stoul(l.substr(p + 6));
    };
    const std::size_t unsaved = eval_of(best), kept = eval_of(last);
    CHECK(kept < unsaved);
    CHECK(contains(saved, "may hold eval " + std::to_string(unsaved) +
                          " (write not synced) or eval " + std::to_string(kept)));
}

/* Once the first best-potential publication had failed for good, every
 * candidate of a parallel optimiser queued behind it on the best mutex spent
 * its own full retry budget, one after another: a DLIB:GFS THREADS 8 run
 * with persistently failing storage took 231 s to end FAILED instead of
 * about one 30 s budget. After the failure (durable_best::FailLatch) those
 * candidates get a single attempt without backoff. Default backoff policy
 * (30 s budget), so this case takes about half a minute. */
TEST_CASE("durable best: later candidates get a single attempt after a fatal failure",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nLIB DLIB\nALGO GFS\nMAXEVAL 400\nTHREADS 8\nSEED 1234\n"
        "ENDOPTIMIZER\n");
    /* skip=3: the fresh-run reset and the first two publications take the
     * lock; every later publication fails with EBUSY. */
    const auto t0 = std::chrono::steady_clock::now();
    const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                {"TADAH_HPO_TEST_IO_FAULT=durable.lock_open,EBUSY,"
                                 "skip=3,count=always",
                                 "TADAH_TEST_DURABLE_E2E_EXPECT=throw",
                                 "TADAH_HPO_FORCE_REBUILD=1"});
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    INFO(r.output);
    INFO("run time: " << seconds << " s");
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    /* The failure is journalled under the best mutex, before any later
     * candidate's attempt, so an in-order parse is exact. */
    std::size_t spent = 0, single = 0, first_eval = 0;
    for (const std::string& l : lines_of(journal)) {
        if (contains(l, "fatal best-potential persistence failure at eval ")) {
            ++spent;
            CHECK(contains(l, "(gave up after "));
            CHECK_FALSE(contains(l, "(gave up after 1 attempt over "));
            const auto p = l.find(" at eval ") + 9;
            first_eval = std::stoul(l.substr(p));
        } else if (contains(l, "best-potential persistence failure at eval ")) {
            ++single;
            CHECK(first_eval != 0);   // after the first failure
            CHECK(contains(l, " (no backoff after the failure at eval " +
                              std::to_string(first_eval) + "): "));
            CHECK(contains(l, "(gave up after 1 attempt over "));
        }
    }
    /* One budget spent; the candidates queued behind it made one attempt. */
    CHECK(spent == 1u);
    CHECK(single >= 1u);
    CHECK(seconds < 45.0);   // one 30 s budget; was one budget per candidate
    CHECK_FALSE(contains(journal, "saved after the persistence failure"));
    /* The first failure is the run's cause, whichever error left last. */
    const std::string status = line_with(journal, "Optimiser status");
    CHECK(contains(status, "FAILED -- best-potential persistence: cannot open "
                           "durable-best lock"));
    CHECK(contains(status, "(gave up after "));
    CHECK_FALSE(contains(status, "(gave up after 1 attempt over "));
    CHECK(contains(line_with(journal, "Best total loss"), "-- NOT saved"));
    const std::size_t n = total_evaluations(journal);
    CHECK(data_rows(sd.dir / "loss.tadah") == n);
}

/* The re-executed half of the next case: one host, evaluated point by point
 * (deterministic order), with faults installed and removed in between. A
 * scan first measures the loss of 21 weights; then, in a fresh run of the
 * same host, six of them, each a new best when it is evaluated unless noted:
 *   eval A (worst)  saved;
 *   eval B (3rd)    the lock is busy for good: the budget is spent, fatal;
 *   eval C (2nd)    still busy (better than A only): a single attempt,
 *                   fails at once;
 *   eval D (4th)    the storage has recovered: saved;
 *   eval E (5th)    one ENOSPC: fails at once, no re-attempt;
 *   eval F (best)   a rename that landed but reported ENOENT: the one
 *                   immediate re-attempt finds it in place and saves it.
 * hpo_run's durable_best::Error handler is replayed here, not run (the
 * real one runs in "durable best: hpo_run ends FAILED after storage
 * recovers"); the parent checks that these assertions passed. */
TEST_CASE("durable latch child: later candidates after a fatal failure",
          "[.durable_latch_child]")
{
    using tadah::hpo::HPOLog;
    namespace durable_best = tadah::hpo::durable_best;
    using tadah::hpo::io::testing::ScopedFaults;
    tadah::core::PeriodicTable::initialize();
    const fs::path dir = fs::current_path();
    HPOLog& log = HPOLog::instance();
    log.open((dir / "scan.log").string(), "test", dir.string(), 0);
    tadah::core::Context ctx = make_validation_context(dir);
    tadah::hpo::HPOTargets trg(ctx, (dir / "config.hpo").string());
    Host host(ctx, trg, (dir / "config.val").string());

    auto at = [](double w) { ParamVec p(1); p(0) = w; return p; };
    std::vector<std::pair<double, double>> scan;   // (loss, weight)
    host.prepare_best_publication(false);
    for (int k = 0; k <= 20; ++k) {
        const double w = -1.0 + 0.1 * k;
        const double loss = host(at(w));
        if (std::isfinite(loss) && loss < trg.FAIL_SCORE) scan.emplace_back(loss, w);
    }
    host.flush_remaining();
    log.finish("SCAN");
    std::sort(scan.begin(), scan.end(), std::greater<>());
    scan.erase(std::unique(scan.begin(), scan.end(),
                           [](const auto& a, const auto& b) { return a.first == b.first; }),
               scan.end());
    REQUIRE(scan.size() >= 6u);
    const std::size_t m = scan.size() - 1;
    const std::size_t q[6] = {0, m / 5, 2 * m / 5, 3 * m / 5, 4 * m / 5, m};
    for (int k = 0; k < 5; ++k) REQUIRE(scan[q[k]].first > scan[q[k + 1]].first);
    const double w_worst = scan[q[0]].second, w_2nd = scan[q[1]].second,
                 w_3rd   = scan[q[2]].second, w_4th = scan[q[3]].second,
                 w_5th   = scan[q[4]].second, w_best = scan[q[5]].second;

    log.reset_for_test();
    log.open((dir / "hpo_run.log").string(), "test", dir.string(), 0);
    host.prepare_best_publication(false);   // a fresh run of the same host

    (void)host(at(w_worst));
    const std::size_t A = Host::current_last_best();
    REQUIRE(A > 0u);
    const std::size_t B = A + 1, C = A + 2, D = A + 3, E = A + 4, F = A + 5;
    std::string first_error;
    {
        ScopedFaults busy("durable.lock_open,EBUSY,always");
        try {
            (void)host(at(w_3rd));
            FAIL("eval B was saved");
        } catch (const durable_best::Error& e) {
            first_error = e.what();
        }
        INFO(first_error);
        CHECK(contains(first_error, "(gave up after "));
        CHECK_FALSE(contains(first_error, "(gave up after 1 attempt over "));
        const auto calls_before = busy.calls("durable.lock_open");
        try {
            (void)host(at(w_2nd));
            FAIL("eval C was saved");
        } catch (const durable_best::Error& e) {
            CHECK(contains(e.what(), "(gave up after 1 attempt over "));
        }
        CHECK(busy.calls("durable.lock_open") == calls_before + 1);   // one attempt
    }
    (void)host(at(w_4th));                   // recovered: a single attempt saves it
    CHECK(Host::current_last_best() == D);
    {
        ScopedFaults full("durable.write,ENOSPC,count=1");
        try {
            (void)host(at(w_5th));
            FAIL("eval E was saved");
        } catch (const durable_best::Error& e) {
            CHECK(contains(e.what(), "No space left on device"));
            CHECK_FALSE(contains(e.what(), "gave up"));
        }
        CHECK(full.calls("durable.write") == 1u);   // no re-attempt
    }
    CHECK(Host::current_last_best() == D);
    {
        ScopedFaults landed("durable.rename,ENOENT,count=1,land;"
                            "durable.temp_create,EIO,skip=1000000");
        (void)host(at(w_best));              // the re-attempt finds it in place
        CHECK(landed.calls("durable.rename") == 1u);
        CHECK(landed.calls("durable.temp_create") == 1u);   // not written twice
    }
    CHECK(Host::current_last_best() == F);

    /* hpo_run's durable_best::Error handler (fail_persistence), replayed:
     * the first recorded failure is the cause. */
    const std::string cause =
        tadah::hpo::detail::durable_failure_cause(durable_best::Error("later"));
    CHECK(cause == first_error);
    log.warning("Best-potential persistence failed: " + cause);
    log.record_best_persistence_failure();
    host.flush_remaining(/*retry_io=*/false);
    log.finish("FAILED -- best-potential persistence: " + cause);

    const std::string journal = read_text(dir / "hpo_run.log");
    INFO(journal);
    const auto S = [](std::size_t v) { return std::to_string(v); };
    const auto lines = lines_of(journal);
    auto index_of = [&](const std::string& needle) {
        for (std::size_t i = 0; i < lines.size(); ++i)
            if (contains(lines[i], needle)) return i;
        return lines.size();
    };
    const std::size_t fatal =
        index_of("fatal best-potential persistence failure at eval " + S(B) + ": ");
    const std::size_t single =
        index_of("best-potential persistence failure at eval " + S(C) +
                 " (no backoff after the failure at eval " + S(B) + "): ");
    REQUIRE(fatal < lines.size());
    REQUIRE(single < lines.size());
    CHECK(fatal < single);
    CHECK(contains(lines[single], "(gave up after 1 attempt over "));
    CHECK(index_of("fatal best-potential persistence failure at eval " + S(C)) ==
          lines.size());
    CHECK(index_of("best potential of eval " + S(D) + " saved after the persistence "
                   "failure at eval " + S(B) + " (no backoff); the run still "
                   "ends FAILED") < lines.size());
    const std::size_t enospc =
        index_of("best-potential persistence failure at eval " + S(E) +
                 " (no backoff after the failure at eval " + S(B) + "): ");
    REQUIRE(enospc < lines.size());
    CHECK(contains(lines[enospc], "No space left on device"));
    CHECK_FALSE(contains(lines[enospc], "gave up"));
    CHECK(index_of("best potential of eval " + S(F) + " saved after the persistence "
                   "failure at eval " + S(B)) < lines.size());
    CHECK(index_of("best-potential I/O: durable.rename (possible landed rename) "
                   "on '") < lines.size());
    CHECK(index_of("; re-trying once, no backoff") < lines.size());
    CHECK(index_of("permanent error") == lines.size());

    /* RESULT: the later, lower saved best is the result; the lowest unsaved
     * candidate (E) is listed as superseded by it. */
    const std::string best = line_with(journal, "Best total loss");
    CHECK(contains(best, "(eval " + S(F)));
    CHECK_FALSE(contains(best, "NOT saved"));
    CHECK(contains(line_with(journal, "Unsaved candidate"),
                   "(eval " + S(E) + ") -- could not be persisted; superseded by "
                   "eval " + S(F)));
    CHECK(contains(line_with(journal, "Best potential saved"),
                   ": best_pot.tadah (eval " + S(F) + ")"));
    CHECK(contains(read_text(dir / "best_pot.tadah"), "# HPO_SOURCE_EVAL " + S(F) + "\n"));
    Host::reset_globals_for_test();
}

/* Before, every candidate after the first fatal failure failed without an
 * attempt, so storage that had recovered could never save a later, lower
 * best. Deterministic: one host is evaluated point by point in a child
 * process with a 200 ms budget. */
TEST_CASE("durable best: storage that recovers after a fatal failure saves a "
          "later best",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_durable_fixture(sd.dir);
    const ChildRun r = run_self("[.durable_latch_child]", sd.dir,
                                {kShortBackoff, "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    CHECK(WEXITSTATUS(r.status) == 0);
    CHECK(contains(r.output, "All tests passed"));
}

/* The same through hpo_run itself: a parallel optimiser whose storage
 * recovers after the first fatal failure. The lock is busy for 5 opens
 * (skip=3: the fresh-run reset and the first two publications take it),
 * which the first failing publication's 150 ms budget spends; later
 * candidates then get their single attempt against working storage. hpo_run
 * must still throw (the process would exit non-zero), with the first
 * failure as the FAILED status, and best_pot.tadah must hold the last
 * candidate saved after it. Which RESULT form appears depends on the
 * completion order of the 4 threads, so both are accepted. */
TEST_CASE("durable best: hpo_run ends FAILED after storage recovers",
          "[durable_e2e][failure_handling][.]")
{
    ScratchDir sd;
    write_policy_fixture(sd.dir,
        "OPTIMIZER\nLIB DLIB\nALGO GFS\nMAXEVAL 400\nTHREADS 4\nSEED 1234\n"
        "ENDOPTIMIZER\n");
    const ChildRun r = run_self("[.durable_e2e_child]", sd.dir,
                                {"TADAH_HPO_TEST_IO_FAULT=durable.lock_open,EBUSY,"
                                 "skip=3,count=5",
                                 "TADAH_HPO_TEST_IO_BACKOFF_MS=20,40,150,0",
                                 "TADAH_TEST_DURABLE_E2E_EXPECT=throw",
                                 "TADAH_HPO_FORCE_REBUILD=1"});
    INFO(r.output);
    REQUIRE(WIFEXITED(r.status));
    REQUIRE(WEXITSTATUS(r.status) == 0);
    REQUIRE(contains(r.output, "All tests passed"));

    const std::string journal = read_text(sd.dir / "hpo_run.log");
    INFO(journal);
    const std::string fatal_tag = "fatal best-potential persistence failure at eval ";
    const std::string saved_tag = " saved after the persistence failure at eval ";
    std::size_t n_fatal = 0, fatal_line = 0, first_saved_line = 0, last_saved = 0;
    std::string first_error;
    const auto lines = lines_of(journal);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        if (contains(l, fatal_tag)) {
            ++n_fatal;
            fatal_line = i;
            const auto colon = l.find(": ", l.find(fatal_tag) + fatal_tag.size());
            REQUIRE(colon != std::string::npos);
            first_error = l.substr(colon + 2);
        } else if (contains(l, saved_tag)) {
            if (!first_saved_line) first_saved_line = i;
            const std::string tag = "best potential of eval ";
            const auto p = l.find(tag);
            REQUIRE(p != std::string::npos);
            last_saved = std::stoul(l.substr(p + tag.size()));
        }
    }
    REQUIRE(n_fatal == 1u);
    CHECK(contains(first_error, "(gave up after "));
    /* hpo_run's own status line carries the first failure. */
    const std::string status = line_with(journal, "Optimiser status");
    CHECK(status.substr(status.find(':') + 2) ==
          "FAILED -- best-potential persistence: " + first_error);
    CHECK_FALSE(contains(journal, "converged"));
    CHECK(data_rows(sd.dir / "loss.tadah") == total_evaluations(journal));
    if (last_saved == 0) {
        WARN("no new best was saved after the persistence failure");
        return;
    }
    CHECK(fatal_line < first_saved_line);
    const std::string S = std::to_string(last_saved);
    CHECK(contains(read_text(sd.dir / "best_pot.tadah"),
                   "# HPO_SOURCE_EVAL " + S + "\n"));
    CHECK((contains(line_with(journal, "Unsaved candidate"), "superseded by eval " + S) ||
           contains(line_with(journal, "Last saved best"), "(eval " + S)));
    CHECK(contains(line_with(journal, "Best potential saved"),
                   ": best_pot.tadah (eval " + S + ")"));
}

// FIXWEIGHT end-to-end host tests: mixed LS/fixed/OPTIM weights (case 2),
// no-training-DBFILE operation (case 3), best_pot precision and exact
// CONFIG/WARM seed evaluation ([seed_anchor]).
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
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
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

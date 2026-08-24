/*
 * Layer-B: feature-isolated optimization performance matrix
 * ([.statistical]). Each config toggles ONE feature against its base so
 * a regression names its feature. Success on a (config, function) cell
 * is counted over N=50 fixed seeds with a fixed per-dimension budget;
 * counts are compared EXACTLY against the recorded baseline
 * (tests/data/anneal_perf_baseline.tsv) -- runs are deterministic, so
 * any difference is a real behavior change. When a deliberate change
 * shifts trajectories, regenerate with
 *     benchmarks/annealing/regen_perf_baseline.py
 * which reruns the matrix (TADAH_PERF_RECORD=1) and prints per-cell
 * Wilson 95% intervals of old vs new -- the review criterion for
 * accepting the new baseline is non-inferiority, not blind refresh.
 *
 * Invariant gates (always-true claims with margin) run regardless of
 * the baseline: every config solves sphere6; the FAILSCORE cliff never
 * crashes and always ends feasible.
 */

#include "catch2/catch.hpp"

#include "anneal_test_functions.h"
#include "anneal_test_support.h"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_pt.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;
using anneal_test::Vec;

namespace {

struct MatrixConfig {
    std::string name;
    bool pt = false;
    std::map<std::string, double> param;
    std::map<std::string, std::string> param_raw;
};

/* Feature isolation: each PT config differs from pt-plain in ONE
 * feature; the two serial configs differ in the kernel; the -gate-
 * configs exist to decide registry defaults from recorded data. */
std::vector<MatrixConfig> matrix_configs()
{
    std::vector<MatrixConfig> cs;
    cs.push_back({"serial-classic", false, {}, {}});
    cs.push_back({"serial-corana", false, {{"CORANA", 1.0}}, {}});
    cs.push_back({"serial-t0acc080-gate", false,
                  {{"T0_ACCEPT", 0.80}}, {}});
    cs.push_back({"pt-plain", true, {}, {{"STEP_MODE", "FIXED"}}});
    cs.push_back({"pt-ladder-adapt", true, {{"LADDER_ADAPT", 1.0}},
                  {{"STEP_MODE", "FIXED"}}});
    cs.push_back({"pt-step-frozen", true, {}, {}});   /* FROZEN default */
    cs.push_back({"pt-step-adaptive", true, {}, {{"STEP_MODE", "ADAPTIVE"}}});
    cs.push_back({"pt-corana", true, {{"CORANA", 1.0}}, {}});
    cs.push_back({"pt-defaults", true, {}, {}});
    cs.push_back({"pt-adapt-combo-gate", true, {{"LADDER_ADAPT", 1.0}},
                  {{"STEP_MODE", "ADAPTIVE"}}});
    return cs;
}

double run_cell(const MatrixConfig& cfg,
                const anneal_functions::Problem& prob, std::uint64_t seed)
{
    OptimizerSpec sp;
    sp.low = prob.low;
    sp.high = prob.high;
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib = "TADAH";
    sp.algo = cfg.pt ? "ANNEAL_PT" : "ANNEAL";
    sp.optim_init.assign(1, prob.start);
    sp.seed = seed;
    sp.num_threads = 1;
    sp.failscore = prob.failscore;
    const std::size_t dim = prob.low.size();
    sp.stop.maxeval = 300 * dim;   /* fixed per-dimension budget */
    for (const auto& [k, v] : cfg.param) sp.param[k] = v;
    for (const auto& [k, v] : cfg.param_raw) sp.param_raw[k] = v;
    if (cfg.pt) {
        sp.param["REPLICAS"] = 8.0;
        sp.param["SWAP_WINDOW"] = 1000000.0;   /* mute reports */
    }

    double best = std::numeric_limits<double>::infinity();
    anneal_test::CountingHost host{[&](const Vec& x) {
        const double f = prob.loss(x);
        const bool feasible =
            !prob.failscore || f < 0.999 * *prob.failscore;
        if (feasible && f < best) best = f;
        return f;
    }};
    if (cfg.pt)
        run_anneal_pt_impl<Vec>(sp, host);
    else
        run_anneal_impl<Vec>(sp, host);
    return best;
}

const std::vector<std::string>& matrix_functions()
{
    static const std::vector<std::string> fs = {
        "sphere2",  "sphere6",    "sphere20",  "ellipsoid6",
        "rosenbrock6", "rastrigin2", "rastrigin6", "ackley10",
        "corana4",  "double_well1", "boundary_optimum6",
        "failscore_cliff6"};
    return fs;
}

std::string baseline_path()
{
    return std::string(__FILE__).substr(
               0, std::string(__FILE__).rfind('/')) +
           "/data/anneal_perf_baseline.tsv";
}

}  // namespace

TEST_CASE("annealing performance matrix: success counts match the "
          "recorded baseline",
          "[.statistical][HPO][anneal][perf_matrix]")
{
    const auto problems = anneal_functions::standard_set();
    const auto configs = matrix_configs();
    constexpr int n_seeds = 50;
    const bool record = std::getenv("TADAH_PERF_RECORD") != nullptr;

    /* Baseline: config \t function \t success \t median_best.
     * Only `success` is gated (exact); the median is the reviewer's
     * context in regen diffs. */
    std::map<std::string, int> baseline;
    if (!record) {
        std::ifstream in(baseline_path());
        REQUIRE(in.good());
        std::string cfg_name, fn;
        int succ;
        double med;
        while (in >> cfg_name >> fn >> succ >> med)
            baseline[cfg_name + "\t" + fn] = succ;
        REQUIRE_FALSE(baseline.empty());
    }

    std::ostringstream out;
    for (const auto& cfg : configs) {
        for (const auto& fname : matrix_functions()) {
            const auto& prob =
                anneal_functions::by_name(problems, fname);
            int succ = 0;
            std::vector<double> bests;
            for (int s = 0; s < n_seeds; ++s) {
                anneal_test::JournalCapture lr;
                const double best =
                    run_cell(cfg, prob, 1000 + std::uint64_t(s));
                bests.push_back(best);
                if (best - prob.f_opt < prob.success_tol) ++succ;
            }
            std::sort(bests.begin(), bests.end());
            const double median = bests[bests.size() / 2];

            /* Invariant gates. The sphere bound is a RESOLUTION
             * claim: any working annealer refines a smooth quadratic
             * to well below 1% of the box scale in 300*dim evals. The
             * fixed-sigma kernels (CLASSIC without step adaptation)
             * legitimately plateau near their step resolution
             * (~2e-3 at STEP_FRAC 0.1), so a strict 1e-3 SUCCESS count
             * differentiates configs in the baseline instead of gating
             * here. */
            if (fname == "sphere6") {
                INFO(cfg.name << " sphere6 success=" << succ
                              << "/50 median=" << median);
                CHECK(median < 1e-2);
            }
            if (fname == "failscore_cliff6") {
                INFO(cfg.name << " failscore_cliff6 median=" << median);
                CHECK(median < 1.0);   /* feasible: far below the cliff */
            }

            out << cfg.name << "\t" << fname << "\t" << succ << "\t"
                << median << "\n";
            if (!record) {
                const auto it = baseline.find(cfg.name + "\t" + fname);
                INFO("cell " << cfg.name << " / " << fname
                             << " success=" << succ << "/50 median="
                             << median);
                REQUIRE(it != baseline.end());
                CHECK(succ == it->second);
            }
        }
    }
    if (record) {
        std::ofstream o(baseline_path());
        o << out.str();
        WARN("baseline recorded to " << baseline_path());
    }
}

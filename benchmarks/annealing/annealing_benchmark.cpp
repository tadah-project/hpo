/* Cross-revision annealing benchmark.
 *
 * Runs the feature-isolated config matrix (mirroring
 * tests/test_anneal_perf_matrix.cpp) over the shared standard test
 * functions and writes one TSV row per (config, function):
 *
 *   config  function  seeds  success  median_best  wall_ms
 *
 * Built OPT-IN via inject.cmake (see README.md) against any TADAH
 * snapshot, so two revisions can be compared A/B with summarize.py.
 * Wall time excludes output I/O; it is a driver-overhead indicator on
 * microsecond objectives, not a substitute for real workload timings.
 *
 * Usage:  hpo_annealing_benchmark [n_seeds] > results.tsv
 */

#include "../../tests/anneal_test_functions.h"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_pt.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <mutex>
#include <streambuf>
#include <string>
#include <vector>

using namespace tadah::hpo;
using Vec = anneal_functions::Vec;

namespace {

struct Config {
    std::string name;
    bool pt = false;
    std::map<std::string, double> param;
    std::map<std::string, std::string> param_raw;
};

std::vector<Config> configs()
{
    return {
        {"serial-classic", false, {}, {}},
        {"serial-corana", false, {{"CORANA", 1.0}}, {}},
        {"pt-plain", true, {}, {{"STEP_MODE", "FIXED"}}},
        {"pt-ladder-adapt", true, {{"LADDER_ADAPT", 1.0}},
         {{"STEP_MODE", "FIXED"}}},
        {"pt-step-frozen", true, {}, {}},
        {"pt-step-adaptive", true, {}, {{"STEP_MODE", "ADAPTIVE"}}},
        {"pt-corana", true, {{"CORANA", 1.0}}, {}},
    };
}

struct Host {
    explicit Host(const anneal_functions::Problem* p) : prob(p) {}
    const anneal_functions::Problem* prob;
    mutable std::mutex m;
    mutable double best = std::numeric_limits<double>::infinity();
    double operator()(const Vec& x) const
    {
        const double f = prob->loss(x);
        const bool feasible =
            !prob->failscore || f < 0.999 * *prob->failscore;
        std::lock_guard<std::mutex> lg(m);
        if (feasible && f < best) best = f;
        return f;
    }
};

}  // namespace

int main(int argc, char** argv)
{
    const int n_seeds = argc > 1 ? std::atoi(argv[1]) : 50;
    /* The drivers echo per-run banners to std::cout (console progress in the
     * production CLI).  The TSV goes out via std::printf (C stdout), so
     * silencing std::cout's streambuf keeps the output machine-readable
     * without touching production code. */
    struct NullBuf : std::streambuf {
        int overflow(int c) override { return traits_type::not_eof(c); }
    } cout_sink;
    std::cout.rdbuf(&cout_sink);
    const auto problems = anneal_functions::standard_set();
    std::printf("config\tfunction\tseeds\tsuccess\tmedian_best\twall_ms\n");
    for (const auto& cfg : configs()) {
        for (const auto& prob : problems) {
            int succ = 0;
            std::vector<double> bests;
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < n_seeds; ++s) {
                OptimizerSpec sp;
                sp.low = prob.low;
                sp.high = prob.high;
                sp.is_int.assign(sp.low.size(), false);
                sp.log_scaled.assign(sp.low.size(), false);
                sp.lib = "TADAH";
                sp.algo = cfg.pt ? "ANNEAL_PT" : "ANNEAL";
                sp.optim_init.assign(1, prob.start);
                sp.seed = 1000 + std::uint64_t(s);
                sp.num_threads = 1;
                sp.failscore = prob.failscore;
                sp.stop.maxeval = 300 * prob.low.size();
                for (const auto& [k, v] : cfg.param) sp.param[k] = v;
                for (const auto& [k, v] : cfg.param_raw)
                    sp.param_raw[k] = v;
                if (cfg.pt) {
                    sp.param["REPLICAS"] = 8.0;
                    sp.param["SWAP_WINDOW"] = 1000000.0;
                }
                Host host{&prob};
                if (cfg.pt)
                    run_anneal_pt_impl<Vec>(sp, host);
                else
                    run_anneal_impl<Vec>(sp, host);
                bests.push_back(host.best);
                if (host.best - prob.f_opt < prob.success_tol) ++succ;
            }
            const double wall_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
            std::sort(bests.begin(), bests.end());
            std::printf("%s\t%s\t%d\t%d\t%g\t%.1f\n", cfg.name.c_str(),
                        prob.name.c_str(), n_seeds, succ,
                        bests[bests.size() / 2], wall_ms);
            std::fflush(stdout);
        }
    }
    return 0;
}

/* Corana et al. (1987) protocol replication -- external oracle.
 *
 * Re-runs the paper's own experimental protocol (their algorithm
 * parameters, start points, and termination rule) through the CORANA
 * variant of ANNEAL and compares against the paper's published Table II:
 * every run must reach the paper's reported accuracy and each row's mean
 * function-evaluation count must lie within [0.5x, 2x] of the paper's
 * figure.
 *
 * The reference numbers are TRANSCRIBED from the paper into
 * tests/data/corana1987_protocol.tsv (provenance notes there); they are
 * deliberately not hard-coded so that nothing in this file can drift
 * from the primary source.  While the table has no data rows the test
 * reports itself as SKIPPED.
 *
 * The multiminima function q_n itself is the shared, transcription-
 * verified definition anneal_functions::corana1987_q (see the notes in
 * anneal_test_functions.h, incl. the cross-check against the paper's
 * reported local-minimum values 3.375E-3 / 5.4E-4); this oracle is what
 * pins that shared definition -- also used by the performance matrix
 * and the benchmark harness -- to the paper's published results.
 *
 * Protocol-fidelity settings applied to every run (paper, Section 2):
 *   BOUNDARY_MODE REDRAW -- out-of-domain coordinates are redrawn (Step 2);
 *   REANNEAL 0       -- the paper has no re-annealing;
 *   T_MIN 1e-10      -- the paper has no temperature floor; the plateau
 *                       rule (eps over N_eps reductions, Step 7) is the
 *                       stop, so the floor is parked far below it;
 *   V_FLOOR_FRAC 1e-12 -- the paper's step vector has no lower clamp;
 *                       the production default (1e-3) would forbid the
 *                       ~1e-6-scale steps the final refinement needs.
 */

#include "catch2/catch.hpp"

#include "anneal_test_functions.h"
#include "anneal_test_support.h"

#include <tadah/hpo/algorithms.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;
using anneal_test::CountingHost;
using anneal_test::Vec;

namespace {

/* Rosenbrock, classic signs per the Corrigenda (+(1 - x_k)^2). Table I
 * cannot be transcribed (its termination epsilon is unpublished), so no
 * reference rows use this yet; kept should that epsilon ever surface. */
double rosenbrock(const Vec& x, double, double)
{
    double s = 0.0;
    for (long i = 0; i + 1 < x.size(); ++i) {
        const double a = x(i), b = x(i + 1);
        s += 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
    }
    return s;
}

/* The paper's multiminima function q_n lives in the shared function
 * header (anneal_functions::corana1987_q, single source of truth used
 * by the performance matrix and benchmark harness too); this oracle
 * pins that definition to the paper's published results. */
double staircase(const Vec& x, double s, double t)
{
    return anneal_functions::corana1987_q(x, s, t);
}

struct ProtocolRow {
    std::string function;
    std::size_t dim = 0;
    double low = 0.0, high = 0.0;
    std::vector<double> start;
    double fs = 0.0, ft = 0.0;  /* staircase grid step / hole half-width */
    double v0 = -1.0;           /* < 0: implementation default */
    double t_init = 0.0, rt = 0.0;
    unsigned ns = 0, nt = 0, neps = 0;
    double eps = 0.0, accuracy = 0.0;
    double neval = 0.0;
};

std::string data_path()
{
    return std::string(__FILE__).substr(0,
               std::string(__FILE__).rfind('/')) +
           "/data/corana1987_protocol.tsv";
}

std::vector<ProtocolRow> load_rows()
{
    std::ifstream in(data_path());
    REQUIRE(in.good());
    std::vector<ProtocolRow> rows;
    std::string line;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::vector<std::string> f;
        std::string tok;
        while (std::getline(ls, tok, '\t')) f.push_back(tok);
        if (!saw_header) {
            REQUIRE(f.size() == 16);
            REQUIRE(f.front() == "function");
            REQUIRE(f.back() == "neval");
            saw_header = true;
            continue;
        }
        REQUIRE(f.size() == 16);
        ProtocolRow r;
        r.function = f[0];
        r.dim = std::stoul(f[1]);
        r.low = std::stod(f[2]);
        r.high = std::stod(f[3]);
        {
            std::istringstream ss(f[4]);
            std::string c;
            while (std::getline(ss, c, ',')) r.start.push_back(std::stod(c));
            REQUIRE(!r.start.empty());
            if (r.start.size() == 1) r.start.assign(r.dim, r.start[0]);
            REQUIRE(r.start.size() == r.dim);
        }
        r.fs = (f[5] == "-") ? 0.0 : std::stod(f[5]);
        r.ft = (f[6] == "-") ? 0.0 : std::stod(f[6]);
        r.v0 = (f[7] == "-") ? -1.0 : std::stod(f[7]);
        r.t_init = std::stod(f[8]);
        r.rt = std::stod(f[9]);
        r.ns = unsigned(std::stoul(f[10]));
        r.nt = unsigned(std::stoul(f[11]));
        r.neps = unsigned(std::stoul(f[12]));
        r.eps = std::stod(f[13]);
        r.accuracy = std::stod(f[14]);
        r.neval = std::stod(f[15]);
        rows.push_back(std::move(r));
    }
    REQUIRE(saw_header);
    return rows;
}

double (*objective(const std::string& name))(const Vec&, double, double)
{
    if (name == "rosenbrock") return &rosenbrock;
    if (name == "staircase") return &staircase;
    FAIL("unknown protocol function: " << name);
    return nullptr;
}

}  // namespace

TEST_CASE("Corana 1987 protocol replication against the paper's Table II",
          "[.corana1987][HPO][anneal]")
{
    const auto rows = load_rows();
    if (rows.empty()) {
        WARN("SKIPPED: tests/data/corana1987_protocol.tsv has no data "
             "rows yet. Transcribe the paper's tables (instructions in "
             "that file) to arm this oracle.");
        SUCCEED();
        return;
    }

    /* The paper ran each starting point once; two seeds per row double
     * the evidence while keeping the whole oracle inside the
     * statistical-label budget. */
    constexpr int n_seeds = 2;
    /* Paper outcome, 4-D block: 8 of 10 runs found the global minimum
     * (~E-7), 2 stopped at the nearest local minimum 3.375E-3 (p. 276).
     * Require at least half of our (staircase, 4-D) runs to match the
     * global outcome; the per-run accuracy gate already forbids anything
     * worse than the paper's worst. */
    int global4_hits = 0, global4_runs = 0;

    /* No Catch2 sections here: each section would re-run the whole test
     * case, resetting the 4-D block accumulator and re-running every
     * other row's optimizations 26 times over.  One pass, INFO context
     * per row. */
    for (std::size_t ri = 0; ri < rows.size(); ++ri) {
        const auto& r = rows[ri];
        {
            INFO("row " << ri << ": " << r.function << " n=" << r.dim
                        << " start=[" << r.start.front() << ", ...]");
            auto* fn = objective(r.function);
            double total_evals = 0.0;
            double row_best_lo = std::numeric_limits<double>::infinity();
            double row_best_hi = -row_best_lo;
            for (int s = 0; s < n_seeds; ++s) {
                auto sp = anneal_test::serial_spec(
                    std::vector<double>(r.dim, r.low),
                    std::vector<double>(r.dim, r.high));
                sp.optim_init.assign(1, r.start);
                sp.seed = 1000 + 10 * std::uint64_t(ri) + std::uint64_t(s);
                sp.param_raw["CORANA"] = "ON";
                sp.param["T_INIT"] = r.t_init;
                sp.param["RT"] = r.rt;
                sp.param["NS"] = double(r.ns);
                sp.param["NT"] = double(r.nt);
                sp.param["NEPS"] = double(r.neps);
                sp.param_raw["BOUNDARY_MODE"] = "REDRAW";
                sp.param_raw["REANNEAL"] = "OFF";
                sp.param["T_MIN"] = 1e-10;
                sp.param["V_FLOOR_FRAC"] = 1e-12;
                if (r.v0 > 0.0)
                    sp.param["STEP_FRAC"] = r.v0 / (r.high - r.low);
                sp.stop.ftol_abs = r.eps;
                /* Runaway guard only; the paper's own plateau rule must
                 * fire long before it (the eval-band gate would fail
                 * anyway if it did not). */
                sp.stop.maxeval = std::uint64_t(4.0 * r.neval);

                double best = std::numeric_limits<double>::infinity();
                std::mutex bm;
                const double fs = r.fs, ft = r.ft;
                CountingHost host{[&, fs, ft](const Vec& x) {
                    const double f = fn(x, fs, ft);
                    std::lock_guard<std::mutex> lg(bm);
                    if (f < best) best = f;
                    return f;
                }};
                run_anneal_impl<Vec>(sp, host);
                total_evals += double(host.n);
                row_best_lo = std::min(row_best_lo, best);
                row_best_hi = std::max(row_best_hi, best);
                INFO("seed " << s << ": best=" << best
                             << " evals=" << host.n);
                CHECK(best <= r.accuracy);
                if (r.function == "staircase" && r.dim == 4) {
                    ++global4_runs;
                    if (best <= 1e-6) ++global4_hits;
                }
            }
            const double mean_evals = total_evals / n_seeds;
            INFO("mean evals = " << mean_evals
                                 << ", paper = " << r.neval);
            CHECK(mean_evals >= 0.5 * r.neval);
            CHECK(mean_evals <= 2.0 * r.neval);
            /* Optional audit trail for a human comparing against the
             * paper's Table II (never printed in CI). */
            if (std::getenv("TADAH_CORANA1987_REPORT"))
                std::printf("corana1987 row=%2zu %s n=%-2zu  best=[%.2e"
                            ", %.2e]  mean_evals=%.0f  paper=%.0f  "
                            "ratio=%.2f\n",
                            ri, r.function.c_str(), r.dim, row_best_lo,
                            row_best_hi, mean_evals, r.neval,
                            mean_evals / r.neval);
        }
    }

    if (global4_runs > 0) {
        INFO("4-D global-minimum outcomes: " << global4_hits << "/"
             << global4_runs << " (paper: 8/10)");
        CHECK(2 * global4_hits >= global4_runs);
    }
}

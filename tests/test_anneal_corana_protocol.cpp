/* Corana et al. (1987) protocol replication -- external oracle.
 *
 * Re-runs the paper's own experimental protocol (their algorithm
 * parameters, start points, and termination rule) through the CORANA
 * variant of ANNEAL and compares against the paper's published results:
 * every seed must reach the paper's reported accuracy and the mean
 * function-evaluation count must lie within [0.5x, 2x] of the paper's
 * figure.
 *
 * The reference numbers are TRANSCRIBED from the paper into
 * tests/data/corana1987_protocol.tsv (see the instructions there);
 * they are deliberately not hard-coded so that nothing in this file can
 * drift from the primary source.  While the table has no data rows the
 * test reports itself as SKIPPED.
 *
 * The test-function definitions below follow the paper's test set as it
 * is reproduced across the secondary literature; the transcriber filling
 * the table must verify them against the paper (and its Corrigenda) and
 * correct any discrepancy here.
 */

#include "catch2/catch.hpp"

#include "anneal_test_support.h"

#include <tadah/hpo/algorithms.h>

#include <cmath>
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

/* Weight cycle shared by the paraboloid and the staircase. */
constexpr double kD[4] = {1.0, 1000.0, 10.0, 100.0};

double paraboloid(const Vec& x)
{
    double s = 0.0;
    for (long i = 0; i < x.size(); ++i) s += kD[i % 4] * x(i) * x(i);
    return s;
}

double rosenbrock(const Vec& x)
{
    double s = 0.0;
    for (long i = 0; i + 1 < x.size(); ++i) {
        const double a = x(i), b = x(i + 1);
        s += 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
    }
    return s;
}

double staircase(const Vec& x)
{
    constexpr double s = 0.2, t = 0.05, c = 0.15;
    double sum = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double xi = x(i);
        const double zi = std::floor(std::fabs(xi / s) + 0.49999) *
                          (xi >= 0.0 ? 1.0 : -1.0) * s;
        if (std::fabs(xi - zi) < t)
            sum += c * kD[i % 4] * zi * zi;
        else
            sum += kD[i % 4] * xi * xi;
    }
    return sum;
}

struct ProtocolRow {
    std::string function;
    std::size_t dim = 0;
    double low = 0.0, high = 0.0;
    std::vector<double> start;
    double v0 = -1.0;  /* < 0: implementation default */
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
            REQUIRE(f.size() == 14);
            REQUIRE(f.front() == "function");
            REQUIRE(f.back() == "neval");
            saw_header = true;
            continue;
        }
        REQUIRE(f.size() == 14);
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
        r.v0 = (f[5] == "-") ? -1.0 : std::stod(f[5]);
        r.t_init = std::stod(f[6]);
        r.rt = std::stod(f[7]);
        r.ns = unsigned(std::stoul(f[8]));
        r.nt = unsigned(std::stoul(f[9]));
        r.neps = unsigned(std::stoul(f[10]));
        r.eps = std::stod(f[11]);
        r.accuracy = std::stod(f[12]);
        r.neval = std::stod(f[13]);
        rows.push_back(std::move(r));
    }
    REQUIRE(saw_header);
    return rows;
}

double (*objective(const std::string& name))(const Vec&)
{
    if (name == "paraboloid") return &paraboloid;
    if (name == "rosenbrock") return &rosenbrock;
    if (name == "staircase") return &staircase;
    FAIL("unknown protocol function: " << name);
    return nullptr;
}

}  // namespace

TEST_CASE("Corana 1987 protocol replication against the paper's tables",
          "[HPO][anneal][corana1987]")
{
    const auto rows = load_rows();
    if (rows.empty()) {
        WARN("SKIPPED: tests/data/corana1987_protocol.tsv has no data "
             "rows yet. Transcribe the paper's tables (instructions in "
             "that file) to arm this oracle.");
        SUCCEED();
        return;
    }

    constexpr int n_seeds = 20;
    for (const auto& r : rows) {
        DYNAMIC_SECTION(r.function << " n=" << r.dim) {
            auto* fn = objective(r.function);
            int reached = 0;
            double total_evals = 0.0;
            for (int s = 0; s < n_seeds; ++s) {
                auto sp = anneal_test::serial_spec(
                    std::vector<double>(r.dim, r.low),
                    std::vector<double>(r.dim, r.high));
                sp.optim_init.assign(1, r.start);
                sp.seed = 1000 + std::uint64_t(s);
                sp.param["CORANA"] = 1.0;
                sp.param["T_INIT"] = r.t_init;
                sp.param["RT"] = r.rt;
                sp.param["NS"] = double(r.ns);
                sp.param["NT"] = double(r.nt);
                sp.param["NEPS"] = double(r.neps);
                if (r.v0 > 0.0)
                    sp.param["STEP_FRAC"] = r.v0 / (r.high - r.low);
                sp.stop.ftol_abs = r.eps;
                /* Runaway guard only; the paper's own termination rule
                 * (CoranaPlateau at eps/NEPS) must fire well before it. */
                sp.stop.maxeval =
                    std::uint64_t(20.0 * r.neval) + 100000;

                double best = std::numeric_limits<double>::infinity();
                std::mutex bm;
                CountingHost host{[&](const Vec& x) {
                    const double f = fn(x);
                    std::lock_guard<std::mutex> lg(bm);
                    if (f < best) best = f;
                    return f;
                }};
                run_anneal_impl<Vec>(sp, host);
                total_evals += double(host.n);
                if (best <= r.accuracy) ++reached;
                INFO("seed " << s << ": best=" << best
                             << " evals=" << host.n);
                CHECK(best <= r.accuracy);
            }
            const double mean_evals = total_evals / n_seeds;
            INFO("mean evals = " << mean_evals
                                 << ", paper = " << r.neval);
            CHECK(reached == n_seeds);
            CHECK(mean_evals >= 0.5 * r.neval);
            CHECK(mean_evals <= 2.0 * r.neval);
        }
    }
}

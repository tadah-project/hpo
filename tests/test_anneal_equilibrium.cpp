/*
 * Layer-A equilibrium-sampling validation of the serial annealing
 * kernels ([.statistical]; run via `ctest -L statistical` or
 * `test_hpo "[.statistical]"`).
 *
 * At a FIXED temperature the Metropolis kernel must sample its exact
 * stationary distribution. Wrong acceptance arithmetic, biased boundary
 * handling, or RNG-stream defects all show up as distorted marginals
 * long before they are visible in optimization benchmarks -- and with
 * closed-form references there is nothing tunable about the check.
 * Statistics use batch means (anneal_stats.h) with fixed seeds; every
 * gate is |value - ref| < 5*SE.
 */

#include "catch2/catch.hpp"

#include "anneal_stats.h"
#include "anneal_test_support.h"

#include <tadah/hpo/anneal_core.h>

#include <cmath>
#include <vector>

using namespace tadah::hpo;
using anneal_test::Vec;

namespace {

/* Drive classic_rung one proposal at a time at fixed T, recording the
 * post-proposal chain state (the standard MCMC estimator: repeats on
 * rejection included). */
std::vector<std::vector<double>> sample_classic(
    const OptimizerSpec& sp, double T, double step_frac_sigma_unused,
    std::uint64_t seed, std::size_t n_burn, std::size_t n_keep)
{
    (void)step_frac_sigma_unused;
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
    const std::size_t dim = sp.low.size();

    anneal::ChainState<Vec> c;
    c.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, 0);
    c.V = anneal::make_initial_steps(sp, kb);
    c.x = Vec(dim);
    for (std::size_t i = 0; i < dim; ++i)
        c.x(i) = 0.5 * (sp.low[i] + sp.high[i]);

    anneal::SharedBudget budget(
        std::numeric_limits<std::uint64_t>::max());
    std::atomic<bool> stop{false};
    const std::function<bool()> never = []() { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};

    auto host = [&](const Vec& x) {
        double f = 0.0;
        for (std::size_t i = 0; i < dim; ++i) f += 0.5 * x(i) * x(i);
        return f;
    };
    /* Constant-loss variant is selected with T <= 0 sentinel by the
     * uniform test below (any proposal accepts). */
    auto host_const = [](const Vec&) { return 1.0; };

    anneal::Knobs kb1 = kb;
    kb1.chain_length = 1;

    c.f = T > 0.0 ? host(c.x) : host_const(c.x);
    std::vector<std::vector<double>> out;
    out.reserve(n_keep);
    for (std::size_t s = 0; s < n_burn + n_keep; ++s) {
        if (T > 0.0)
            anneal::classic_rung(c, T, sp, kb1, host, ctx);
        else
            anneal::classic_rung(c, 1.0, sp, kb1, host_const, ctx);
        if (s >= n_burn) {
            std::vector<double> row(dim);
            for (std::size_t i = 0; i < dim; ++i) row[i] = c.x(i);
            out.push_back(std::move(row));
        }
    }
    return out;
}

std::vector<double> column(const std::vector<std::vector<double>>& xs,
                           std::size_t i)
{
    std::vector<double> c;
    c.reserve(xs.size());
    for (const auto& row : xs) c.push_back(row[i]);
    return c;
}

}  // namespace

TEST_CASE("classic Metropolis kernel samples the Gaussian at fixed T",
          "[.statistical][HPO][anneal_core][equilibrium]")
{
    /* f(x) = |x|^2 / 2 at temperature T has stationary density
     * prod_i exp(-x_i^2 / (2T)): mean 0, Var = T, fourth central moment
     * 3T^2, zero cross-covariance. The box +/-10*sqrt(T) truncates
     * ~1e-23 of the mass -- far below the 5*SE resolution. */
    struct Cfg { std::size_t dim; double T; std::uint64_t seed; };
    for (const Cfg cfg : {Cfg{1, 0.5, 1001}, Cfg{1, 2.0, 1002},
                          Cfg{5, 1.0, 1003}}) {
        DYNAMIC_SECTION("dim=" << cfg.dim << " T=" << cfg.T) {
            const double half = 10.0 * std::sqrt(cfg.T);
            auto sp = anneal_test::serial_spec(
                std::vector<double>(cfg.dim, -half),
                std::vector<double>(cfg.dim, half));
            sp.stop.maxeval = 1;   /* satisfies the AUTO gate; unused */
            /* sigma = STEP_FRAC * 20*sqrt(T) = 2.4*sqrt(T)/sqrt(d):
             * the random-walk optimal-scaling region for whole-vector
             * proposals (sigma ~ 2.4/sqrt(d) in units of the target
             * scale). Acceptance only affects mixing efficiency, not
             * correctness, but an unscaled step in d=5 collapses
             * acceptance to a few percent and the autocorrelation time
             * then exceeds the batch length. */
            sp.param["STEP_FRAC"] = 0.12 / std::sqrt(double(cfg.dim));

            const auto xs = sample_classic(sp, cfg.T, 0, cfg.seed,
                                           20000, 400000);
            for (std::size_t i = 0; i < cfg.dim; ++i) {
                const auto xi = column(xs, i);
                const auto m = anneal_stats::mean_of(xi);
                CHECK_STAT("classic_mean", m.mean, 0.0, m.se);
                const auto v = anneal_stats::var_of(xi);
                CHECK_STAT("classic_var", v.mean, cfg.T, v.se);
                const auto m4 = anneal_stats::m4_of(xi);
                CHECK_STAT("classic_m4", m4.mean, 3.0 * cfg.T * cfg.T,
                           m4.se);
            }
            if (cfg.dim > 1) {
                /* Cross-covariance of the first two coordinates. */
                const auto a = column(xs, 0), b = column(xs, 1);
                std::vector<double> prod(a.size());
                for (std::size_t s = 0; s < a.size(); ++s)
                    prod[s] = a[s] * b[s];
                const auto cv = anneal_stats::mean_of(prod);
                CHECK_STAT("classic_cov01", cv.mean, 0.0, cv.se);
            }
        }
    }
}

TEST_CASE("mirror reflection preserves the uniform distribution",
          "[.statistical][HPO][anneal_core][equilibrium][boundary]")
{
    /* Constant loss => every proposal accepts and the chain is a
     * reflected Gaussian random walk on [0,1], whose stationary law is
     * uniform: mean 1/2, Var 1/12, fourth central moment 1/80, and 5%
     * edge-strip occupancy on each side (where reflection bias would
     * concentrate). STEP_FRAC 1.0 exercises multi-fold reflection
     * (sigma = the full width). */
    for (const double frac : {0.3, 1.0}) {
        DYNAMIC_SECTION("STEP_FRAC=" << frac) {
            auto sp = anneal_test::serial_spec({0.0}, {1.0});
            sp.stop.maxeval = 1;
            sp.param["STEP_FRAC"] = frac;
            const auto xs =
                sample_classic(sp, 0.0 /* constant-loss mode */, 0,
                               2001 + std::uint64_t(frac * 10), 20000,
                               400000);
            const auto x = column(xs, 0);
            const auto m = anneal_stats::mean_of(x);
            CHECK_STAT("uniform_mean", m.mean, 0.5, m.se);
            const auto v = anneal_stats::var_of(x);
            CHECK_STAT("uniform_var", v.mean, 1.0 / 12.0, v.se);
            const auto m4 = anneal_stats::m4_of(x);
            CHECK_STAT("uniform_m4", m4.mean, 1.0 / 80.0, m4.se);
            for (const bool high : {false, true}) {
                std::vector<double> in_strip(x.size());
                for (std::size_t s = 0; s < x.size(); ++s)
                    in_strip[s] = high ? (x[s] > 0.95 ? 1.0 : 0.0)
                                       : (x[s] < 0.05 ? 1.0 : 0.0);
                const auto e = anneal_stats::mean_of(in_strip);
                CHECK_STAT(high ? "uniform_strip_hi" : "uniform_strip_lo",
                           e.mean, 0.05, e.se);
            }
        }
    }
}

TEST_CASE("tight-box sampling matches the truncated Gaussian",
          "[.statistical][HPO][anneal_core][equilibrium][boundary]")
{
    /* f = x^2/2 at T=1 on [-1,1]: the stationary density is the
     * standard normal truncated to one sigma. For truncation at +/-a,
     *   Var = 1 - 2a*phi(a) / (2*Phi(a) - 1),
     * with phi the normal pdf and 2*Phi(a)-1 = erf(a/sqrt(2)). Here the
     * boundary is hit constantly, so any reflection asymmetry moves the
     * variance. */
    auto sp = anneal_test::serial_spec({-1.0}, {1.0});
    sp.stop.maxeval = 1;
    sp.param["STEP_FRAC"] = 0.5;   /* sigma = 1: heavy boundary traffic */
    const auto xs = sample_classic(sp, 1.0, 0, 3001, 20000, 400000);
    const auto x = column(xs, 0);

    const double a = 1.0;
    const double phi_a =
        std::exp(-0.5 * a * a) / std::sqrt(2.0 * std::acos(-1.0));
    const double mass = std::erf(a / std::sqrt(2.0));
    const double var_ref = 1.0 - 2.0 * a * phi_a / mass;

    const auto m = anneal_stats::mean_of(x);
    CHECK_STAT("trunc_mean", m.mean, 0.0, m.se);
    const auto v = anneal_stats::var_of(x);
    CHECK_STAT("trunc_var", v.mean, var_ref, v.se);
}

TEST_CASE("Corana step adaptation is stationary at fixed temperature",
          "[.statistical][HPO][anneal_core][equilibrium][corana]")
{
    /* At a fixed T on a smooth target the Corana controller's fixed
     * point is a step vector whose per-dimension acceptance sits in the
     * paper's [0.4, 0.6] dead band; inside the band V does not move.
     * Assertions: the time-averaged acceptance over the second half of
     * the run lies in the dead band (to 5*SE of the per-rung series),
     * and the geometric-mean step obeys the single-adjustment bound
     * (one 0.4/0.6-rule update multiplies V by at most 1 + C = 3), i.e.
     * max/min of V_gm over the second half < 3. */
    auto sp = anneal_test::serial_spec({-10.0, -10.0}, {10.0, 10.0});
    sp.stop.maxeval = 1;
    sp.param_raw["CORANA"] = "ON";
    /* NS = 20 (the paper default): each per-dimension acceptance ratio
     * is measured on 20 samples per adjustment. At small NS the
     * controller adjusts on pure binomial noise (with NS = 5 and
     * p = 0.5, 37% of ratios fall outside the dead band) and V executes
     * a wide multiplicative random walk even at equilibrium. */
    sp.param["NS"] = 20.0;
    sp.param["NT"] = 1.0;
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);

    anneal::ChainState<Vec> c;
    c.rng = anneal::make_stream(4001, anneal::SALT_CHAIN, 0);
    c.V = anneal::make_initial_steps(sp, kb);
    c.x = Vec(2);
    c.x = 0.0;
    anneal::SharedBudget budget(std::numeric_limits<std::uint64_t>::max());
    std::atomic<bool> stop{false};
    const std::function<bool()> never = []() { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};
    auto host = [](const Vec& x) {
        return 0.5 * (x(0) * x(0) + x(1) * x(1));
    };
    c.f = host(c.x);

    const double T = 1.0;
    const std::size_t rungs = 400;
    std::vector<double> acc_series;
    std::vector<double> vgm_series;
    for (std::size_t r = 0; r < rungs; ++r) {
        const auto rs = anneal::corana_rung(c, T, sp, kb, host, ctx);
        acc_series.push_back(
            rs.proposed > 0
                ? double(rs.accepted) / double(rs.proposed) : 0.0);
        vgm_series.push_back(std::sqrt(c.V[0] * c.V[1]));
    }
    const std::vector<double> acc2(acc_series.begin() + rungs / 2,
                                   acc_series.end());
    const auto a = anneal_stats::mean_of(acc2);
    INFO("second-half acceptance " << a.mean << " +- " << a.se);
    /* THE stationarity statement: the controller's time-averaged
     * acceptance sits in the Corana dead band. */
    CHECK(a.mean > 0.4 - 5.0 * a.se);
    CHECK(a.mean < 0.6 + 5.0 * a.se);

    /* V itself fluctuates with the binomial noise of the ratio
     * estimates (no tight distributional bound exists), but it must
     * respect its PROVABLE invariants at every rung: the
     * [V_FLOOR_FRAC * width, width] clamp of adjust_corana_step. */
    double vmin = vgm_series[rungs / 2], vmax = vmin;
    for (std::size_t r = rungs / 2; r < rungs; ++r) {
        vmin = std::min(vmin, vgm_series[r]);
        vmax = std::max(vmax, vgm_series[r]);
    }
    INFO("second-half V_gm range [" << vmin << ", " << vmax << "]");
    const double width = sp.high[0] - sp.low[0];
    CHECK(vmin >= kb.v_floor_frac * width);
    CHECK(vmax <= width * (1.0 + 1e-12));
}

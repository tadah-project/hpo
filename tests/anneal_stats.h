#pragma once
/* Batch-means statistics for the annealing validation suite.
 *
 * MCMC samples are autocorrelated, so a naive standard error
 * sd(x)/sqrt(n) understates the uncertainty and would make 5-sigma
 * gates fire on correct code. Batch means need no independence
 * assumption: split the series into B = ceil(sqrt(n)) contiguous
 * batches (each much longer than the autocorrelation time for the
 * sample sizes used here), compute the statistic per batch, and take
 * SE = sd(batch statistics)/sqrt(B).
 *
 * Every statistical assertion in the suite has the form
 *     |value - ref| < 5 * SE
 * with a FIXED seed, so CI is deterministic; the 5-sigma margin exists
 * so the tests survive legitimate RNG-stream changes without retuning
 * (a correct implementation re-seeded arbitrarily fails a given
 * statistic with p ~ 5.7e-7). References must be closed-form,
 * quadrature-derived, or produced by an independent in-test model --
 * never by the code under test.
 */

#include "catch2/catch.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace anneal_stats {

struct Estimate {
    double mean = 0.0;   /* grand mean of the per-batch statistic */
    double se = 0.0;     /* batch-means standard error            */
    std::size_t batches = 0;
};

/* Per-batch statistic -> batch-means estimate of its mean and SE. */
inline Estimate batch_stat(
    const std::vector<double>& xs,
    const std::function<double(const double*, std::size_t)>& stat)
{
    Estimate e;
    const std::size_t n = xs.size();
    if (n < 16) return e;
    const std::size_t B =
        static_cast<std::size_t>(std::ceil(std::sqrt(double(n))));
    const std::size_t len = n / B;         /* drop the remainder */
    if (len == 0) return e;
    std::vector<double> bs;
    bs.reserve(B);
    for (std::size_t b = 0; b < B; ++b)
        bs.push_back(stat(xs.data() + b * len, len));
    double m = 0.0;
    for (double v : bs) m += v;
    m /= double(bs.size());
    double s2 = 0.0;
    for (double v : bs) s2 += (v - m) * (v - m);
    s2 /= double(bs.size() - 1);
    e.mean = m;
    e.se = std::sqrt(s2 / double(bs.size()));
    e.batches = bs.size();
    return e;
}

inline double stat_mean(const double* x, std::size_t n)
{
    double m = 0.0;
    for (std::size_t i = 0; i < n; ++i) m += x[i];
    return m / double(n);
}

inline double stat_var(const double* x, std::size_t n)
{
    const double m = stat_mean(x, n);
    double v = 0.0;
    for (std::size_t i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
    return v / double(n);
}

/* Fourth central moment (for kurtosis references). */
inline double stat_m4(const double* x, std::size_t n)
{
    const double m = stat_mean(x, n);
    double v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = x[i] - m;
        v += d * d * d * d;
    }
    return v / double(n);
}

inline Estimate mean_of(const std::vector<double>& xs)
{
    return batch_stat(xs, &stat_mean);
}

/* Central moments are computed around the GLOBAL mean, with batches
 * used only for the standard error: centering each batch on its own
 * mean would bias the estimate low by O(tau/batch_len) (the classic
 * within-batch variance bias), which at 5-sigma resolution is visible
 * for the correlated chains sampled here. Around the global mean the
 * per-batch statistic is an (asymptotically) unbiased mean of
 * (x - mu)^k terms and batch means applies exactly. */
inline Estimate var_of(const std::vector<double>& xs)
{
    Estimate g = batch_stat(xs, &stat_mean);
    const double mu = g.mean;
    return batch_stat(xs, [mu](const double* x, std::size_t n) {
        double v = 0.0;
        for (std::size_t i = 0; i < n; ++i) v += (x[i] - mu) * (x[i] - mu);
        return v / double(n);
    });
}
inline Estimate m4_of(const std::vector<double>& xs)
{
    Estimate g = batch_stat(xs, &stat_mean);
    const double mu = g.mean;
    return batch_stat(xs, [mu](const double* x, std::size_t n) {
        double v = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = x[i] - mu;
            v += d * d * d * d;
        }
        return v / double(n);
    });
}

}  // namespace anneal_stats

/* The gate. Prints a machine-readable STAT line (the record-over-time
 * hook) and applies the 5-sigma batch-means criterion. */
#define CHECK_STAT(name, value, ref, se_)                                   \
    do {                                                                    \
        const double cs_v = (value), cs_r = (ref), cs_se = (se_);           \
        INFO("STAT " << (name) << " value=" << cs_v << " ref=" << cs_r      \
                     << " se=" << cs_se << " z="                            \
                     << (cs_se > 0 ? (cs_v - cs_r) / cs_se : 0.0));         \
        REQUIRE(cs_se > 0.0);                                               \
        CHECK(std::fabs(cs_v - cs_r) < 5.0 * cs_se);                        \
    } while (0)

/* Relaxed variant for phases that are only asymptotically exact
 * (diminishing adaptation). */
#define CHECK_STAT_RELAXED(name, value, ref, se_, k)                        \
    do {                                                                    \
        const double cs_v = (value), cs_r = (ref), cs_se = (se_);           \
        INFO("STAT " << (name) << " value=" << cs_v << " ref=" << cs_r      \
                     << " se=" << cs_se << " z="                            \
                     << (cs_se > 0 ? (cs_v - cs_r) / cs_se : 0.0));         \
        REQUIRE(cs_se > 0.0);                                               \
        CHECK(std::fabs(cs_v - cs_r) < (k) * cs_se);                        \
    } while (0)

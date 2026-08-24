#pragma once
/* Shared standard test functions for the annealing performance matrix
 * and the cross-revision benchmark harness. Every function is stated
 * with its known optimum; several are normalised so a fixed temperature
 * remains meaningful across functions. Nothing here is derived from any
 * user workload.
 *
 * References: sphere/ellipsoid/Rosenbrock/Rastrigin/Ackley are the
 * standard global-optimization test set; corana1987_q below is the
 * multiminima test function q_n of Corana, Marchesi, Martini & Ridella,
 * ACM TOMS 13(3), 1987 (the paper whose annealing algorithm the CORANA
 * variant implements), transcription-verified and guarded by the
 * Table II replication oracle in test_anneal_corana_protocol.cpp. */

#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlib/matrix.h>

namespace anneal_functions {

using Vec = dlib::matrix<double, 0, 1>;

struct Problem {
    std::string name;
    std::vector<double> low;
    std::vector<double> high;
    std::vector<double> start;
    std::optional<double> failscore;
    double f_opt = 0.0;               /* known global minimum value */
    double success_tol = 1e-3;        /* success := best - f_opt < tol */
    std::function<double(const Vec&)> loss;
};

/* ---- Corana et al. (1987) multiminima function q_n -------------------
 *
 * Transcribed from the paper (p. 272) on 2026-08-24 and cross-checked
 * against its reported local-minimum values (0.15*(s-t)^2 * d_i along a
 * d_i = 1 axis: 3.375E-3 for s=.2/t=.05 and 5.4E-4 for s=.1/t=.04).
 * The Table II replication oracle (test_anneal_corana_protocol.cpp)
 * exercises this exact definition, so any drift here breaks a test with
 * an external anchor.
 *
 * NOTE q_n is NOT the separable per-coordinate staircase circulated in
 * secondary implementations: the local-minimum holes are open PRODUCT
 * regions -- a point is inside one iff EVERY coordinate lies strictly
 * within t of a grid node k_i*s and the node vector k is not
 * identically zero -- and the plateau value uses the hole's inner
 * corner z_i = k_i*s - sign(k_i)*t, not the node itself.  Everywhere
 * else q is the paraboloid sum_i d_i x_i^2. */

/* Steepness vectors d (paper p. 272): part of the definition, per n. */
inline const std::vector<double>& corana1987_steepness(std::size_t dim)
{
    static const std::vector<double> d2{1.0, 1000.0};
    static const std::vector<double> d4{1.0, 1000.0, 10.0, 100.0};
    static const std::vector<double> d10{1.0, 1000.0, 10.0, 100.0, 1.0,
                                         10.0, 100.0, 1000.0, 1.0, 10.0};
    if (dim == 2) return d2;
    if (dim == 4) return d4;
    if (dim == 10) return d10;
    throw std::runtime_error(
        "corana1987_q: no steepness vector for dim " + std::to_string(dim));
}

inline double corana1987_q(const Vec& x, double s, double t)
{
    constexpr double cr = 0.15;
    const auto& d = corana1987_steepness(std::size_t(x.size()));
    bool in_hole = true, all_zero = true;
    double hole = 0.0, parab = 0.0;
    for (long i = 0; i < x.size(); ++i) {
        const double xi = x(i);
        parab += d[std::size_t(i)] * xi * xi;
        if (!in_hole) continue;
        const double k = std::nearbyint(xi / s);
        if (std::fabs(xi - k * s) >= t) { in_hole = false; continue; }
        if (k != 0.0) {
            all_zero = false;
            const double zi = k * s - (k > 0.0 ? t : -t);
            hole += d[std::size_t(i)] * zi * zi;
        }
    }
    return (in_hole && !all_zero) ? cr * hole : parab;
}

inline std::vector<Problem> standard_set()
{
    const double pi = std::acos(-1.0);
    std::vector<Problem> ps;

    auto add_sphere = [&](std::size_t dim) {
        ps.push_back(Problem{
            "sphere" + std::to_string(dim), std::vector<double>(dim, -1.0),
            std::vector<double>(dim, 1.0), std::vector<double>(dim, 0.8),
            std::nullopt, 0.0, 1e-3,
            [dim](const Vec& x) {
                double s = 0.0;
                for (long i = 0; i < x.size(); ++i) s += x(i) * x(i);
                return s / double(dim);
            }});
    };
    add_sphere(2);
    add_sphere(6);
    add_sphere(20);

    {
        constexpr std::size_t dim = 6;
        std::vector<double> start(dim);
        for (std::size_t i = 0; i < dim; ++i)
            start[i] = (i % 2 == 0) ? 0.75 : -0.50;
        ps.push_back(Problem{
            "ellipsoid6", std::vector<double>(dim, -1.0),
            std::vector<double>(dim, 1.0), start, std::nullopt, 0.0, 1e-3,
            [](const Vec& x) {
                constexpr std::size_t d = 6;
                double weighted = 0.0, weights = 0.0;
                for (std::size_t i = 0; i < d; ++i) {
                    const double w =
                        std::pow(1000.0, double(i) / double(d - 1));
                    weighted += w * x(long(i)) * x(long(i));
                    weights += w;
                }
                return weighted / weights;
            }});
    }

    {
        constexpr std::size_t dim = 6;
        std::vector<double> start(dim);
        for (std::size_t i = 0; i < dim; ++i)
            start[i] = (i % 2 == 0) ? -1.2 : 1.0;
        ps.push_back(Problem{
            "rosenbrock6", std::vector<double>(dim, -2.0),
            std::vector<double>(dim, 2.0), start, std::nullopt, 0.0, 1e-2,
            [](const Vec& x) {
                constexpr std::size_t d = 6;
                double s = 0.0;
                for (std::size_t i = 0; i + 1 < d; ++i) {
                    const double a = x(long(i)), b = x(long(i + 1));
                    s += 100.0 * (b - a * a) * (b - a * a) +
                         (1.0 - a) * (1.0 - a);
                }
                return s / (101.0 * double(d - 1));
            }});
    }

    auto add_rastrigin = [&](std::size_t dim) {
        std::vector<double> start(dim);
        for (std::size_t i = 0; i < dim; ++i)
            start[i] = (i % 2 == 0) ? 3.2 : -2.7;
        ps.push_back(Problem{
            "rastrigin" + std::to_string(dim),
            std::vector<double>(dim, -5.12), std::vector<double>(dim, 5.12),
            start, std::nullopt, 0.0, 5e-3,
            [dim, pi](const Vec& x) {
                double s = 0.0;
                for (long i = 0; i < x.size(); ++i)
                    s += x(i) * x(i) - 10.0 * std::cos(2.0 * pi * x(i)) +
                         10.0;
                return s / (20.0 * double(dim));
            }});
    };
    add_rastrigin(2);
    add_rastrigin(6);

    {
        constexpr std::size_t dim = 10;
        ps.push_back(Problem{
            "ackley10", std::vector<double>(dim, -32.768),
            std::vector<double>(dim, 32.768),
            std::vector<double>(dim, 20.0), std::nullopt, 0.0, 5e-2,
            [pi](const Vec& x) {
                const double d = double(x.size());
                double s2 = 0.0, sc = 0.0;
                for (long i = 0; i < x.size(); ++i) {
                    s2 += x(i) * x(i);
                    sc += std::cos(2.0 * pi * x(i));
                }
                return -20.0 * std::exp(-0.2 * std::sqrt(s2 / d)) -
                       std::exp(sc / d) + 20.0 + std::exp(1.0);
            }});
    }

    {
        /* The paper's q_4 at its standard parameters (s=0.2, t=0.05,
         * domain [-10000, 10000]^4, 10^20 local minima).  The cell name
         * carries the provenance so cross-revision A/B tools never
         * silently compare it against the retired separable variant
         * ("corana4"). */
        constexpr std::size_t dim = 4;
        ps.push_back(Problem{
            "corana1987_q4", std::vector<double>(dim, -10000.0),
            std::vector<double>(dim, 10000.0),
            {999.0, -1007.0, 1001.0, -903.0}, std::nullopt, 0.0, 1e-6,
            [](const Vec& x) { return corana1987_q(x, 0.2, 0.05); }});
    }

    {
        /* Double well 10*(x^2-1)^2: minima at +/-1 (f=0), barrier 10 at
         * the origin. Started in the LEFT well; success = finding the
         * (equally deep) right well is not required, so the tolerance
         * gates depth only -- the escape-rate contrast between serial
         * and tempering uses the well-occupancy statistic directly. */
        ps.push_back(Problem{
            "double_well1", {-2.0}, {2.0}, {-1.0}, std::nullopt, 0.0, 1e-3,
            [](const Vec& x) {
                const double u = x(0) * x(0) - 1.0;
                return 10.0 * u * u;
            }});
    }

    {
        constexpr std::size_t dim = 6;
        ps.push_back(Problem{
            "boundary_optimum6", std::vector<double>(dim, -1.0),
            std::vector<double>(dim, 1.0), std::vector<double>(dim, -0.5),
            std::nullopt, 0.0, 1e-3,
            [](const Vec& x) {
                double s = 0.0;
                for (long i = 0; i < x.size(); ++i) {
                    const double d = (x(i) - 1.0) / 2.0;
                    s += d * d;
                }
                return s / double(x.size());
            }});
    }

    {
        constexpr std::size_t dim = 6;
        constexpr double cliff = 1.0e6;
        ps.push_back(Problem{
            "failscore_cliff6", std::vector<double>(dim, -1.0),
            std::vector<double>(dim, 1.0), std::vector<double>(dim, 0.15),
            cliff, 0.0, 1e-3,
            [](const Vec& x) {
                for (long i = 0; i < x.size(); ++i)
                    if (x(i) < 0.0) return cliff;
                double s = 0.0;
                for (long i = 0; i < x.size(); ++i) {
                    const double d = x(i) - 0.65;
                    s += d * d;
                }
                return s / double(x.size());
            }});
    }

    return ps;
}

inline const Problem& by_name(const std::vector<Problem>& ps,
                              const std::string& name)
{
    for (const auto& p : ps)
        if (p.name == name) return p;
    throw std::runtime_error("unknown test function: " + name);
}

}  // namespace anneal_functions

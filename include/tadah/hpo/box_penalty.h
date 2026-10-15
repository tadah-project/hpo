#pragma once
/*
 * box_penalty.h -- points outside the OPTIM box (owner decision, MK,
 * 2026-10-15).
 *
 * The OPTIM box is the configuration's contract: no point outside it is
 * evaluated, and none can be the run's best. A point an optimiser proposes
 * outside the box -- the CERES drivers' iterates and line-search trials
 * (Ceres' GradientProblemSolver has no bounds); every other registered
 * backend keeps to the box -- is not trained; HPO_Host::operator() scores
 * it with out_of_box_score() of its distance to the box, a failure graded by
 * that distance, and GradientProvider gives it the slope of that score
 * (out_of_box_slope), whatever GRAD_STRATEGY, without evaluating anything.
 * Both decide with outside_box() on the same REAL point, so they agree.
 *
 * The rule (outside_box). A coordinate within rounding outside its box is
 * evaluated on the bound (SearchSpace::snapped_onto_box, owner decision
 * D10) and is not outside; a NaN coordinate is not outside either (its
 * evaluation fails as such). Any other coordinate below lo or above hi is.
 *
 * The distance is taken in the search's own coordinates
 * (SearchSpace::to_internal): under OPTIM_SCALE AUTO a fraction of each
 * box's width, under NONE the raw value (ln x on a LOG dimension). Per
 * dimension the signed excess e_k = u_k - internal_hi(k) above the box,
 * u_k - internal_lo(k) below it (negative), 0 inside; the distance is
 * d = ||e||_2, +inf when an internal coordinate is not finite (x <= 0 on a
 * LOG dimension, which only a direct host call can ask for).
 *
 * The score (out_of_box_score) is graded_failscore(FS, FailBand::D, d, 1),
 *
 *     L(d) = 0.999 FS + 0.001 FS (0.75 + 0.25 d / (1 + d)),
 *
 * the range of band D -- where the gates' rejects before training score --
 * graded by the distance whatever FAILSCORE_GRADING says: it is the box's
 * penalty, not a model failure. It is always a failure (>= 0.99975 FS:
 * loss_is_feasible() is false for every FAILSCORE the parser takes, the
 * default, the largest double, included), so it is never the best, a WARM
 * seed or a PRESTAGE candidate. It lies below FAILSCORE itself, the loss of
 * an ungraded in-box failure, where no gradient search goes on (such a
 * point has no usable gradient); under FAILSCORE_GRADING ON it ranks above
 * every band A-C failure and with the band-D failures by their measure. A
 * lower place (band A, the hot-variable range band) would make a step out of
 * the box downhill for a search started in a band B-D failure region; a
 * place above FAILSCORE does not exist for the default. L grows with d
 * (strictly until the measure saturates, d ~ 1e9), so its slope points away
 * from the box and a descent step back into it.
 *
 * The slope (out_of_box_slope), in internal coordinates:
 *     dL/du_k = 0.001 FS * 0.25 * sigma'(d) * e_k / d,
 *     sigma'(d) = 1 / (1 + d)^2;
 * 0 when d is not finite.
 */

#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/search_space.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace tadah::hpo {

/** How a point lies outside its OPTIM box (outside_box). */
struct OutOfBox {
    /** Per dimension, the signed excess in the search's own coordinates:
     *  > 0 above the box, < 0 below it, 0 inside. */
    std::vector<double> excess;
    /** ||excess||_2; +inf when an internal coordinate is not finite. */
    double      distance = 0.0;
    /** The first dimension outside the box. */
    std::size_t first = 0;
    /** How many dimensions lie outside the box. */
    std::size_t count = 0;
};

namespace detail {

/* The distance of an excess vector (overflow-safe). */
inline double excess_norm(const std::vector<double>& e)
{
    double scale = 0.0;
    for (double v : e) {
        if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
        scale = std::max(scale, std::fabs(v));
    }
    if (scale == 0.0) return 0.0;
    double s = 0.0;
    for (double v : e) {
        const double r = v / scale;
        s += r * r;
    }
    return scale * std::sqrt(s);
}

}  // namespace detail

/** Whether the REAL point @p p lies outside the OPTIM box of @p space (see
 *  the file comment): std::nullopt when every coordinate lies inside, within
 *  rounding of a bound (evaluated on it) or is NaN. */
template<class V>
std::optional<OutOfBox> outside_box(const SearchSpace& space, const V& p)
{
    const std::size_t n =
        std::min(space.dim(), ss_detail::len(p));
    OutOfBox o;
    for (std::size_t k = 0; k < n; ++k) {
        const double x = ss_detail::elem(p, k);
        const DimSpec& d = space.at(k);
        if (!(x < d.lo || x > d.hi)) continue;          /* inside, or NaN */
        if (space.snapped_onto_box(k, x)) continue;     /* on the bound */
        if (o.count++ == 0) {
            o.first = k;
            o.excess.assign(n, 0.0);
        }
        const double u = space.to_internal(k, x);
        o.excess[k] = x > d.hi ? u - space.internal_hi(k)
                               : u - space.internal_lo(k);
        if (!std::isfinite(u))
            o.excess[k] = x > d.hi ? std::numeric_limits<double>::infinity()
                                   : -std::numeric_limits<double>::infinity();
    }
    if (o.count == 0) return std::nullopt;
    o.distance = detail::excess_norm(o.excess);
    return o;
}

/** The same for an internal point @p u of a spec without a SearchSpace:
 *  its internal bounds @p lo, @p hi are the box (no snap). */
template<class V>
std::optional<OutOfBox> outside_internal_box(const std::vector<double>& lo,
                                             const std::vector<double>& hi,
                                             const V& u)
{
    const std::size_t n =
        std::min({lo.size(), hi.size(), ss_detail::len(u)});
    OutOfBox o;
    for (std::size_t k = 0; k < n; ++k) {
        const double x = ss_detail::elem(u, k);
        if (!(x < lo[k] || x > hi[k])) continue;
        if (o.count++ == 0) {
            o.first = k;
            o.excess.assign(n, 0.0);
        }
        o.excess[k] = x > hi[k] ? x - hi[k] : x - lo[k];
    }
    if (o.count == 0) return std::nullopt;
    o.distance = detail::excess_norm(o.excess);
    return o;
}

/** The score of a point at distance @p distance outside the box, for the
 *  FAILSCORE @p failscore (finite, > 0): band D's range of graded failures,
 *  graded by the distance (see the file comment). */
inline double out_of_box_score(double failscore, double distance)
{
    return graded_failscore(failscore, FailBand::D, distance, 1.0);
}

/** Fills @p g (resized to the excess' size) with the slope of
 *  out_of_box_score() at the point @p o describes, in internal
 *  coordinates. */
template<class V>
void out_of_box_slope(const OutOfBox& o, double failscore, V& g)
{
    const std::size_t n = o.excess.size();
    if constexpr (std::is_same_v<V, std::vector<double>>) g.assign(n, 0.0);
    else {
        g.set_size(static_cast<long>(n));
        for (std::size_t k = 0; k < n; ++k) ss_detail::elem(g, k) = 0.0;
    }
    const double d = o.distance;
    if (!std::isfinite(d) || !(d > 0.0)) return;
    const double w = 1.0e-3 * failscore;
    const double ds = 1.0 / ((1.0 + d) * (1.0 + d));   /* sigma'(d), v0 = 1 */
    const double c = 0.25 * w * ds / d;
    for (std::size_t k = 0; k < n; ++k)
        ss_detail::elem(g, k) = c * o.excess[k];
}

}  // namespace tadah::hpo

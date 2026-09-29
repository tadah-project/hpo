#pragma once
/*
 * fd_stencil.h -- the finite-difference stencil of HPO's in-house kernel,
 * and which kernel the gradients of a spec use (gradient_provider.h).
 *
 * Kernels. A GRAD_STRATEGY NUMERICAL gradient is taken by the in-house
 * kernel (PARAM fd_kernel inhouse; the default under OPTIM_SCALE AUTO) or
 * by Ceres' numeric differentiation (fd_kernel ceres; the default under
 * OPTIM_SCALE NONE). HYBRID's finite-difference pass -- the parameters
 * without an analytical derivative -- always uses the in-house stencil;
 * ANALYTICAL takes no finite difference.
 *
 * Step. Along internal coordinate u (what the optimiser iterates on) the
 * in-house step is h = fd_step * max(|u|, 1): PARAM fd_step of the box
 * width in the unit box of OPTIM_SCALE AUTO, absolute below |u| = 1 and
 * relative above under OPTIM_SCALE NONE.
 *
 * Stencil. A point u in its OPTIM box [lo, hi] (the internal bounds,
 * OptimizerSpec::low/high) is differenced without evaluating outside the
 * box:
 *
 *   central   (f(u+h) - f(u-h)) / (2h)                  u-h >= lo, u+h <= hi
 *   forward   (4 (f(u+h) - f(u)) - (f(u+2h) - f(u))) / (2h)   u-h < lo
 *   backward  the forward formula with -h:   u+h > hi
 *             (4 (f(u-h) - f(u)) - (f(u-2h) - f(u))) / (-2h)
 *
 * The one-sided formulas are (-3 f(u) + 4 f(u+h) - f(u+2h)) / (2h) and its
 * mirror, written with differences so that two large, close losses do not
 * overflow. All three are second order: the errors are h^2 f'''/6
 * (central) and -h^2 f'''/3 (one-sided, twice the central constant). The
 * conditions are tested on the very points evaluated (u - h, u + 2.0*h,
 * ... as computed), so a chosen stencil never leaves the box, and away
 * from the bounds the central stencil is the one used before 1.6.0, bit
 * for bit.
 *
 * Cost. A one-sided stencil needs f(u), which the NUMERICAL in-house kernel
 * evaluates anyway (the centre, for its bad-evaluation contract): two
 * evaluations per dimension either way. HYBRID's analytical pass does not
 * evaluate the loss, so a HYBRID gradient with a finite-difference
 * coordinate within h of a bound evaluates the centre once: one evaluation
 * more than the central stencil would take.
 *
 * A point outside its box -- only an optimiser that ignores the bounds
 * (OptimizerCaps::ignores_bounds: the CERES drivers, NLopt LN_NEWUOA)
 * evaluates one -- keeps the central stencil: the search itself is outside
 * the box there. Ceres' numeric differentiation cannot be bounded: it
 * perturbs each coordinate by +-delta (CENTRAL, RIDDERS) or +delta
 * (FORWARD) inside Ceres (internal/numeric_diff.h), and NumericDiffOptions
 * has no bounds.
 *
 * Narrow boxes. Every point of a box has a stencil that fits when
 * lo + 3H <= hi, H = fd_step * max(|lo|, |hi|, 1) being the largest step
 * in it (fd_box_error, with the rounding of the computed points included).
 * A narrower box is refused -- at startup (validate_hpo_setup, for every
 * phase whose gradients use the in-house stencil) and by the gradient of a
 * GradientProvider before it evaluates anything -- naming the dimension,
 * its box and the largest fd_step it takes: a derivative whose step is a third of the box
 * or more says little about the slope, and shrinking the step for that
 * dimension alone would silently change what PARAM fd_step means (and
 * the noise it was chosen to step over: the NOISE block's AUTO).
 */

#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah::hpo {

/* The kernel of a GRAD_STRATEGY NUMERICAL gradient. */
enum class FdKernel : std::uint8_t { InHouse, Ceres };

/* PARAM fd_kernel of `sp`, or the default of its OPTIM_SCALE: InHouse in
 * the unit box (OPTIM_SCALE AUTO), Ceres otherwise (OPTIM_SCALE NONE, and
 * a spec without a normalised space). Throws std::runtime_error for
 * another value. */
inline FdKernel fd_kernel_of(const OptimizerSpec& sp)
{
    FdKernel k = (sp.space && sp.space->normalised()) ? FdKernel::InHouse
                                                      : FdKernel::Ceres;
    if (auto it = sp.param_raw.find("fd_kernel"); it != sp.param_raw.end()) {
        if (it->second == "ceres" || it->second == "CERES")
            k = FdKernel::Ceres;
        else if (it->second == "inhouse" || it->second == "INHOUSE")
            k = FdKernel::InHouse;
        else
            throw std::runtime_error(
                "PARAM fd_kernel must be inhouse (default) or ceres (got '" +
                it->second + "')");
    }
    return k;
}

/* PARAM fd_step of the in-house stencil (default 1e-6). Throws
 * std::runtime_error unless it is positive and finite. */
inline double fd_step_of(const OptimizerSpec& sp)
{
    double fd_step = 1.0e-6;
    if (auto it = sp.param.find("fd_step"); it != sp.param.end())
        fd_step = it->second;
    if (!(fd_step > 0.0) || !std::isfinite(fd_step))
        throw std::runtime_error(
            "PARAM fd_step must be a positive finite fraction of the box width");
    return fd_step;
}

/* Whether the gradients of a run of `sp` (GradientProvider) take finite
 * differences with the in-house stencil: GRAD_STRATEGY NUMERICAL with the
 * in-house kernel, or HYBRID. Throws for an invalid PARAM fd_kernel under
 * NUMERICAL, as the provider does. */
inline bool uses_in_house_stencil(const OptimizerSpec& sp)
{
    const GradStrategy g = sp.grad_strategy.value_or(GradStrategy::Numerical);
    if (g == GradStrategy::Hybrid) return true;
    return g == GradStrategy::Numerical &&
           fd_kernel_of(sp) == FdKernel::InHouse;
}

/* Whether they take them with Ceres' numeric differentiation, whose steps
 * are not bounded: GRAD_STRATEGY NUMERICAL with the Ceres kernel. */
inline bool uses_ceres_kernel(const OptimizerSpec& sp)
{
    return sp.grad_strategy.value_or(GradStrategy::Numerical) ==
               GradStrategy::Numerical &&
           fd_kernel_of(sp) == FdKernel::Ceres;
}

/* The in-house step at internal coordinate u. */
inline double fd_step_at(double fd_step, double u)
{
    return fd_step * std::max(std::abs(u), 1.0);
}

/* Which stencil differences a coordinate (see the file comment). */
enum class FdSide : std::uint8_t { Central, Forward, Backward };

/* The stencil of internal coordinate u with step h in the box [lo, hi]:
 * central when u - h and u + h lie in it, forward (u + h, u + 2.0*h) when
 * u - h lies below it and u + 2.0*h does not lie above, backward
 * (u - h, u - 2.0*h) in the mirror case; nullopt when u lies in the box
 * and none fits (a box narrower than the stencil: fd_box_error refuses
 * one). A u outside the box, or NaN, is Central. Pass an infinite box for
 * a dimension without one. */
inline std::optional<FdSide> fd_side(double u, double h, double lo, double hi)
{
    if (!(u >= lo && u <= hi)) return FdSide::Central;
    const bool below = u - h < lo;
    const bool above = u + h > hi;
    if (!below && !above) return FdSide::Central;
    if (below && !above && u + 2.0 * h <= hi) return FdSide::Forward;
    if (above && !below && u - 2.0 * h >= lo) return FdSide::Backward;
    return std::nullopt;
}

/* Whether the in-house stencil with PARAM fd_step fits at every point of
 * the box [lo, hi]: lo + 3H <= hi and hi - 3H >= lo, H = fd_step *
 * max(|lo|, |hi|, 1), each rounded outwards (the step at any u of the box
 * is at most H; u - h < lo as computed implies u < lo + h exactly, so
 * u + 2h < lo + 3H, and rounding is monotone). */
inline bool fd_box_fits(double lo, double hi, double fd_step)
{
    const double H = fd_step_at(fd_step, std::max(std::abs(lo), std::abs(hi)));
    const double inf = std::numeric_limits<double>::infinity();
    return std::isfinite(H) &&
           std::nextafter(std::fma(3.0, H, lo), inf) <= hi &&
           std::nextafter(std::fma(-3.0, H, hi), -inf) >= lo;
}

/* "" when the in-house stencil with `fd_step` fits every OPTIM box of `sp`
 * (its internal bounds low/high; fd_box_fits), else the error: each
 * dimension that does not, with its box, the stencil's span there and the
 * largest fd_step it takes. */
inline std::string fd_box_error(const OptimizerSpec& sp, double fd_step)
{
    const auto num = [](double v) {
        std::ostringstream os;
        os << std::setprecision(10) << v;
        return os.str();
    };
    std::string bad;
    const std::size_t n = std::min(sp.low.size(), sp.high.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double lo = sp.low[i], hi = sp.high[i];
        if (fd_box_fits(lo, hi, fd_step)) continue;
        const double m = std::max({std::abs(lo), std::abs(hi), 1.0});
        const std::string label =
            (i < sp.parameter_labels.size() && !sp.parameter_labels[i].empty())
                ? sp.parameter_labels[i] : logidx::dim_label(i);
        bad += "\n  - " + label + ": box [" + num(lo) + ", " + num(hi) +
               "], width " + num(hi - lo) + " < 3 x " +
               num(fd_step_at(fd_step, m)) + " (fd_step below " +
               num((hi - lo) / (3.0 * m)) + " fits it)";
    }
    if (bad.empty()) return "";
    return "PARAM fd_step " + num(fd_step) + ": the finite-difference "
           "stencil takes three steps of fd_step x max(|u|, 1) inside the "
           "OPTIM box, in internal coordinates (the unit box under "
           "OPTIM_SCALE AUTO), so that it never evaluates outside it; these "
           "boxes are narrower:" + bad +
           "\n  Lower PARAM fd_step (set it by hand when the NOISE block's "
           "AUTO derived it) or widen these boxes.";
}

}  // namespace tadah::hpo

#pragma once
/*
 * fd_stencil.h -- the finite-difference stencil of HPO's gradients
 * (gradient_provider.h: GRAD_STRATEGY NUMERICAL, and HYBRID's pass over the
 * parameters without an analytical derivative), and the box checks that
 * go with it.
 *
 * Settings (fd_settings_of). Three PARAMs of the block that takes the
 * gradient:
 *   fd_kernel    the step rule: inhouse (FdRule::Width) or ceres
 *                (FdRule::Ceres). The default is inhouse under
 *                OPTIM_SCALE AUTO and for HYBRID, ceres for NUMERICAL
 *                under OPTIM_SCALE NONE (and a spec without a
 *                normalised space);
 *   fd_step      the step (default 1e-6), positive and finite; the
 *                NOISE block's AUTO may have derived it
 *                (OptimizerSpec::fd_step_from_noise_auto);
 *   diff_method  0 CENTRAL (default) or 1 FORWARD. 2 (RIDDERS) is
 *                refused: Ridders' extrapolation ignored fd_step, started
 *                from a step of 0.32 x max(|u|, 1) and could not be kept
 *                inside the box.
 *
 * Step along internal coordinate u (what the optimiser iterates on):
 *   Width   h = fd_step * max(|u|, 1)       -- fd_step of the box width in
 *                                              the unit box (OPTIM_SCALE
 *                                              AUTO), absolute below |u| = 1
 *                                              and relative above under NONE;
 *   Ceres   h = max(2^-26, |u| * fd_step)   -- the rule of Ceres'
 *                                              NumericDiff (relative_step_
 *                                              size = fd_step; 2^-26 is
 *                                              sqrt(DBL_EPSILON)).
 * The step is never shrunk or grown; only its direction changes.
 *
 * Stencil (fd_plan) of u in its OPTIM box [a, b] (internal bounds,
 * OptimizerSpec::low/high). The conditions are tested on the very points
 * evaluated, as computed, so a chosen stencil never leaves the box:
 *   CENTRAL   u + h, u - h, when both lie in the box:
 *               Width  (f(u+h) - f(u-h)) / (2h)
 *               Ceres  (f(u+h) - f(u-h)) * ((1/h) / 2)
 *             otherwise one-sided into the box, towards the side with
 *             room: u + s, u + 2s (s = +h or -h), with f(u):
 *               (4 (f(u+s) - f(u)) - (f(u+2s) - f(u))) / (2s)
 *             = (-3 f(u) + 4 f(u+s) - f(u+2s)) / (2s), written with
 *             differences so that two large, close losses do not overflow;
 *   FORWARD   u + h, or u - h when u + h lies above the box (SciPy's
 *             2-point flip), with f(u):
 *               Width  (f(u+s) - f(u)) / s
 *               Ceres  (f(u+s) - f(u)) * (1/s)
 * In the interior each rule is its old kernel bit for bit: the in-house
 * stencil of Tadah 1.5, and Ceres' DynamicNumericDiffCostFunction (the
 * same points in the same order -- the centre, then per dimension +h and
 * -h -- and the same arithmetic; the naive (f+ - f-) / (2h) differs in
 * the last bit in about a quarter of the components). The Ceres library
 * is not used.
 *
 * Accuracy. The central and the one-sided 3-point differences are second
 * order: truncation error h^2 f'''/6 and -h^2 f'''/3. Noise sigma on the
 * loss becomes 0.71 sigma/h (central) and sqrt(26)/2 sigma/h = 2.55
 * sigma/h (one-sided). The forward difference is first order (h f''/2,
 * 1.41 sigma/h). Cost: the same number of evaluations as the interior
 * stencil: the loss at u, which a one-sided stencil uses, is the one the
 * caller has just evaluated (GradientProvider::gradient_at), else
 * evaluated once per gradient.
 *
 * A point outside the box is not differenced: a gradient search that
 * ignores the bounds (OptimizerCaps::ignores_bounds: the CERES drivers)
 * gets the slope of its out-of-box score there (box_penalty.h,
 * GradientProvider; owner decision, MK, 2026-10-15, which retires D4, the
 * difference towards the box from outside it), and a coordinate within
 * rounding outside its box is differenced on the bound, where it is
 * evaluated (D10). fd_plan refuses a centre outside the box.
 *
 * Kinds. A difference never mixes a usable loss with a failure, nor a graded
 * failure with a usable loss or an ungraded failure (owner decision, MK,
 * 2026-10-15): GradientProvider takes, per component, the side of the
 * stencil whose losses are of the kind of the loss at the point, the
 * stencil a bound there would give (gradient_provider.h, fd_component).
 *
 * Narrow boxes (fd_fit, fd_box_error). A box in which some u has no
 * stencil is refused at startup (validate_hpo_setup, every phase that
 * takes finite-difference gradients) and by the gradient before it
 * evaluates anything, naming the largest fd_step that fits
 * (fd_largest_fitting_step): shrinking the step at run time would change
 * what PARAM fd_step means, and the noise it was chosen to step over.
 * Under OPTIM_SCALE AUTO a CENTRAL step fits exactly when fd_step <= 1/3,
 * a FORWARD one when fd_step <= 1/2 (the in-house rule; Ceres' rule: 1/2
 * and 1); under NONE the check runs over the whole box. It is sound -- a
 * box it accepts gives every u a stencil. Its bound on the computed step
 * is exact where the step is (a power-of-two fd_step), else within an ulp
 * of fd_step; but it treats u +- h as exact reals, where fd_plan tests
 * them rounded, so in a box only hundreds or thousands of ulps wide it can
 * refuse an fd_step that every double of the box accepts, up to a few per
 * cent below the true limit ([16820343.156662896, 16820343.156663369],
 * FORWARD, fd_step 2^-46: 1.6%). The AUTO limits above are exact.
 *
 * ASA's probes (asa_probe_fit): the reannealing and exit tangents of
 * external/asa (cost_derivatives) evaluate (1 + Delta) x, or (1 - Delta) x
 * when the first lies outside the box, and the exit curvature (Curvature_0
 * 0 or -1) x +- Delta |x|, one-sided x -+ Delta |x|, x -+ 2 Delta |x| next
 * to a bound, and elsewhere x +- Delta |x|, the points it tests
 * (TADAH-PATCH 8 in external/asa/TADAH-PATCHES.md: upstream evaluated
 * (1 +- Delta) x there, which can round an ulp beyond a bound whatever the
 * Delta); algorithms_asa.cpp refuses a box in which they cannot fit, as
 * ASA computes them.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace tadah::hpo {

/* The step rule (PARAM fd_kernel). */
enum class FdRule : std::uint8_t {
    Width,  /* fd_kernel inhouse: h = fd_step * max(|u|, 1) */
    Ceres   /* fd_kernel ceres:   h = max(2^-26, |u| * fd_step) */
};

/* PARAM diff_method. */
enum class FdScheme : std::uint8_t { Central, Forward };

/* Where fd_step came from. */
enum class FdStepSource : std::uint8_t {
    Default,    /* not set: 1e-6 */
    Param,      /* PARAM fd_step */
    NoiseAuto   /* the NOISE block's AUTO derived it */
};

/* Ceres' smallest step, sqrt(DBL_EPSILON) = 2^-26 (numeric_diff.h:
 * min_step_size). */
inline constexpr double kCeresMinStep = 0x1p-26;

/* The finite-difference settings of a spec (fd_settings_of). */
struct FdSettings {
    FdRule       rule    = FdRule::Width;
    FdScheme     scheme  = FdScheme::Central;
    double       fd_step = 1.0e-6;
    FdStepSource source  = FdStepSource::Default;
    bool         rule_written = false;   /* PARAM fd_kernel was given */
};

/* The finite-difference settings of `sp` (see the file comment): PARAM
 * fd_kernel, fd_step and diff_method, with the defaults of its
 * GRAD_STRATEGY and OPTIM_SCALE. Throws std::runtime_error for an invalid
 * value, and for diff_method 2 (RIDDERS). */
FdSettings fd_settings_of(const OptimizerSpec& sp);

/* Whether `key` is one of the finite-difference PARAMs, fd_step,
 * fd_kernel and diff_method: fd_settings_of checks them, with the same
 * message on every backend, and a backend's own PARAM list passes them
 * on. */
inline bool is_fd_param(const std::string& key)
{
    return key == "fd_step" || key == "fd_kernel" || key == "diff_method";
}

/* Whether `sp` sets PARAM fd_step, fd_kernel or diff_method (or the NOISE
 * block's AUTO derived fd_step); "" when not, else their names, "PARAM
 * fd_step and diff_method". */
std::string fd_params_written(const OptimizerSpec& sp);

/* The step of `rule` at internal coordinate u. Ceres' arithmetic:
 * std::max(min_step_size, |u| * relative_step_size). */
inline double fd_step_at(FdRule rule, double fd_step, double u)
{
    if (rule == FdRule::Width) return fd_step * std::max(std::abs(u), 1.0);
    return std::max(kCeresMinStep, std::abs(u) * fd_step);
}

/* One dimension's stencil (fd_plan). */
struct FdStencil {
    enum class Kind : std::uint8_t {
        Central,   /* p1 = u + h, p2 = u - h */
        OneSided,  /* p1 = u + s, p2 = u + 2.0 * s, with f(u) */
        Forward    /* p1 = u + s, with f(u) */
    };
    Kind   kind = Kind::Central;
    double h  = 0.0;   /* the step, > 0 */
    double s  = 0.0;   /* the signed step of OneSided / Forward (+h or -h) */
    double p1 = 0.0;   /* the coordinates evaluated, in this order */
    double p2 = 0.0;
    /* A stencil a bound turned: one-sided into the box instead of central,
     * or backward instead of forward. */
    bool   turned = false;

    bool needs_centre() const { return kind != Kind::Central; }
};

/* The stencil of internal coordinate u in the box [a, b] with `rule`,
 * `scheme` and `fd_step` (see the file comment); std::nullopt when none
 * fits (fd_fit refuses such a box). A NaN u gets the central stencil (its
 * evaluations fail). Throws std::logic_error for a u outside the box,
 * which is not differenced (see the file comment). */
std::optional<FdStencil> fd_plan(double u, double a, double b, FdRule rule,
                                 FdScheme scheme, double fd_step);

/* The derivative from the stencil's losses: y0 = f(u) (unused by Central),
 * y1 = f(p1), y2 = f(p2) (unused by Forward). The interior arithmetic of
 * each rule's old kernel; see the file comment. */
inline double fd_combine(const FdStencil& st, FdRule rule, double y0,
                         double y1, double y2)
{
    switch (st.kind) {
        case FdStencil::Kind::Central:
            if (rule == FdRule::Width) return (y1 - y2) / (2.0 * st.h);
            {
                /* Ceres (numeric_diff.h, EvaluateJacobianColumn):
                 * residuals -= temp; one_over_delta /= 2;
                 * residuals *= one_over_delta. */
                double one_over_delta = 1.0 / st.h;
                one_over_delta /= 2;
                return (y1 - y2) * one_over_delta;
            }
        case FdStencil::Kind::OneSided:
            return (4.0 * (y1 - y0) - (y2 - y0)) / (2.0 * st.s);
        case FdStencil::Kind::Forward:
            if (rule == FdRule::Width) return (y1 - y0) / st.s;
            return (y1 - y0) * (1.0 / st.s);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/* Whether every u of the box [a, b] gets a stencil (fd_plan) with `rule`,
 * `scheme` and `fd_step`; when not, `where` is a u of the box without
 * one. */
struct FdFit {
    bool   ok = true;
    double where = std::numeric_limits<double>::quiet_NaN();
};
FdFit fd_fit(double a, double b, FdRule rule, FdScheme scheme, double fd_step);

/* The largest fd_step for which fd_fit accepts the box, or std::nullopt
 * when none does (a box narrower than three of Ceres' smallest steps,
 * 3 x 2^-26, somewhere). */
std::optional<double> fd_largest_fitting_step(double a, double b,
                                              FdRule rule, FdScheme scheme);

/* "" when every OPTIM box of `sp` (its internal bounds) fits the stencil of
 * `s`, else the error: each dimension that does not, with its box, a
 * point without a stencil and the largest fd_step that fits it, the step's
 * source, and what to change. */
std::string fd_box_error(const OptimizerSpec& sp, const FdSettings& s);

/* What the finite differences of `s` are, for the journal's OPTIMISER
 * section and the messages: "CENTRAL, h = 1e-06 of each box width (PARAM
 * fd_step; in-house rule); next to a bound: one-sided 3-point, same h,
 * same cost". `normalised`: OPTIM_SCALE AUTO; `hybrid`: HYBRID's pass. */
std::string fd_describe(const FdSettings& s, bool normalised, bool hybrid);

/* `v` with `digits` significant digits, rounded towards zero, so that the
 * printed value itself fits where `v` is the largest that does. */
std::string fd_floor_text(double v, int digits = 6);

/* ASA's finite-difference probes of a coordinate in [lo, hi] with Delta
 * `delta` (external/asa cost_derivatives), as ASA computes them in double:
 * the tangents, (1 + Delta) x or, when that lies outside the box, (1 -
 * Delta) x -- also the rule of the off-diagonal curvature -- and with
 * `curvature` (Curvature_0 0 or -1) the diagonal curvature, x +- Delta
 * |x|, or next to a bound x -+ Delta |x| and x -+ 2 Delta |x| (the middle
 * branch evaluates the x +- Delta |x| it tested: TADAH-PATCH 8). Sound: a
 * box it accepts keeps these probes in it; within an ulp of Delta of
 * exact. `where`: an x of the box where a probe could lie outside it. */
FdFit asa_probe_fit(double lo, double hi, double delta, bool curvature);

/* The largest Delta asa_probe_fit accepts for [lo, hi] (std::nullopt: none
 * does). */
std::optional<double> asa_largest_fitting_delta(double lo, double hi,
                                                bool curvature);

}  // namespace tadah::hpo

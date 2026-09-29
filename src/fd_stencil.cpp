#include <tadah/hpo/fd_stencil.h>

#include <tadah/hpo/log_index.h>
#include <tadah/hpo/search_space.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>

namespace tadah::hpo {

namespace {

std::string num(double v)
{
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

std::string label_of(const OptimizerSpec& sp, std::size_t i)
{
    return (i < sp.parameter_labels.size() && !sp.parameter_labels[i].empty())
               ? sp.parameter_labels[i]
               : logidx::dim_label(i);
}

/* ---- the fit check ----------------------------------------------------
 *
 * The computed step h^(u) of each rule is bounded above, at every double u,
 * by a piecewise linear H(u) = c0 + c1 u:
 *   Width  f for |u| <= 1, where max(|u|, 1) = 1 and f * 1.0 is exact, and
 *          F |u| for |u| > 1;
 *   Ceres  2^-26 for |u| below u*, the smallest double with
 *          fl(f u*) > 2^-26, and F |u| from u* on.
 * F = upper_factor(f) bounds the rounded product: f itself when f is a
 * normal power of two, whose products are exact, else nextafter(f), which
 * is not below the rounded f |u|. The pieces meet at doubles (1, u*), and
 * the gap between a piece's end and the next double holds no double. A
 * double u of the box without a stencil (fd_plan, on computed doubles)
 * then satisfies the strict real inequalities below with H in place of
 * h^: rounding to nearest is monotone and a, b are doubles, so fl(u - h^)
 * < a implies u - h^ < a, and u - H <= u - h^. On each linear piece the set
 * where a conjunction of two strict inequalities holds is an open interval
 * whose ends are among the piece's ends and the roots of the linear
 * functions; it is found by testing the midpoint of each pair of
 * consecutive such candidates, in long double. So the check is sound -- a
 * box it accepts gives every double u a stencil -- and exact where the
 * step is (a power-of-two fd_step: 1/2 and 1 with Ceres' rule in the unit
 * box), else within the ulp by which F exceeds f. */

using LD = long double;

/* An upper bound F of the factor f of a rounded product: fl(f x) <= F x for
 * every double x with |f x| >= DBL_MIN (a normal result). f itself when f
 * is a normal power of two (the product is then exact), else the next
 * double above f: fl(f x) exceeds f x by at most half an ulp of f x, which
 * is at most ulp(f) x. */
double upper_factor(double f)
{
    int e = 0;
    if (f >= std::numeric_limits<double>::min() && std::frexp(f, &e) == 0.5)
        return f;
    return std::nextafter(f, std::numeric_limits<double>::infinity());
}

/* The smallest double u >= 0 with fl(f u) > m (f > 0), or +inf when there
 * is none: fl(f u) grows with u, so bisect on the bit patterns of the
 * non-negative doubles, which order them. */
double first_product_above(double f, double m)
{
    const double top = std::numeric_limits<double>::max();
    if (!(f * top > m)) return std::numeric_limits<double>::infinity();
    std::uint64_t lo = 0, hi = 0;
    std::memcpy(&hi, &top, sizeof hi);
    /* invariant: fl(f from_bits(hi)) > m; fl(f from_bits(lo)) <= m (u = 0) */
    while (hi - lo > 1) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        double u;
        std::memcpy(&u, &mid, sizeof u);
        if (f * u > m) hi = mid;
        else lo = mid;
    }
    double u;
    std::memcpy(&u, &hi, sizeof u);
    return u;
}

struct Piece {
    LD s, t;      /* the piece, clipped to the box */
    LD c0, c1;    /* H(u) = c0 + c1 u >= the computed step on [s, t] */
};

std::vector<Piece> step_pieces(double a, double b, FdRule rule, double f)
{
    const LD inf = std::numeric_limits<LD>::infinity();
    const LD F = upper_factor(f);
    std::vector<Piece> raw;
    if (rule == FdRule::Width) {
        /* the doubles |u| > 1 start at nextafter(1) */
        const LD n1 = std::nextafter(1.0, 2.0);
        raw = {{-inf, -n1, 0, -F}, {-1, 1, static_cast<LD>(f), 0},
               {n1, inf, 0, F}};
    } else {
        const double m = kCeresMinStep;
        const double us = first_product_above(f, m);
        if (std::isinf(us)) {
            raw = {{-inf, inf, static_cast<LD>(m), 0}};
        } else {
            /* the doubles below u* have fl(f |u|) <= 2^-26 */
            const LD below = std::nextafter(us, 0.0);
            const LD U = us;
            raw = {{-inf, -U, 0, -F}, {-below, below, static_cast<LD>(m), 0},
                   {U, inf, 0, F}};
        }
    }
    std::vector<Piece> out;
    for (Piece p : raw) {
        p.s = std::max(p.s, static_cast<LD>(a));
        p.t = std::min(p.t, static_cast<LD>(b));
        if (p.s <= p.t) out.push_back(p);
    }
    return out;
}

/* One conjunction of two linear conditions, g1(u) < 0 and g2(u) > 0 (each
 * g(u) = k u + d; a condition "> 0" is passed negated). */
struct Lin {
    LD k, d;
    LD at(LD u) const { return k * u + d; }
};

/* The candidates of a piece: its ends and the roots of `fns` inside it,
 * sorted. */
std::vector<LD> candidates(LD s, LD t, const std::vector<Lin>& fns)
{
    std::vector<LD> c = {s, t};
    for (const Lin& g : fns)
        if (g.k != 0) {
            const LD r = -g.d / g.k;
            if (r > s && r < t) c.push_back(r);
        }
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

/* A point where one of the conjunctions (g1 < 0 and g2 < 0) holds
 * strictly -- the midpoints of consecutive candidates, and the single
 * point of a degenerate piece -- or NaN when none does. */
LD first_failure(const std::vector<LD>& cand,
                 const std::vector<std::pair<Lin, Lin>>& conj)
{
    std::vector<LD> probe;
    if (cand.size() == 1) probe.push_back(cand[0]);
    for (std::size_t k = 1; k < cand.size(); ++k)
        probe.push_back((cand[k - 1] + cand[k]) / 2);
    for (const auto& c : conj)
        for (LD u : probe)
            if (c.first.at(u) < 0 && c.second.at(u) < 0) return u;
    return std::numeric_limits<LD>::quiet_NaN();
}

FdFit fit_pieces(double a, double b, FdRule rule, FdScheme scheme, double f)
{
    FdFit fit;
    const LD A = a, B = b;
    for (const Piece& p : step_pieces(a, b, rule, f)) {
        /* g(u) < 0 forms:
         *   below   u - H - a < 0          (u - h under the box)
         *   above   b - u - H < 0          (u + h over it)
         *   up      b - u - 2H < 0         (u + 2h over it: no room above)
         *   down    u - 2H - a < 0         (u - 2h under it: no room below) */
        const Lin below{1 - p.c1, -p.c0 - A};
        const Lin above{-1 - p.c1, B - p.c0};
        const Lin up{-1 - 2 * p.c1, B - 2 * p.c0};
        const Lin down{1 - 2 * p.c1, -2 * p.c0 - A};
        std::vector<std::pair<Lin, Lin>> conj;
        if (scheme == FdScheme::Forward)
            conj = {{above, below}};
        else
            conj = {{below, up}, {above, down}};
        const LD w = first_failure(
            candidates(p.s, p.t, {below, above, up, down}), conj);
        if (!std::isnan(w)) {
            fit.ok = false;
            fit.where = std::clamp(static_cast<double>(w), a, b);
            return fit;
        }
    }
    return fit;
}

std::uint64_t bits_of(double v)
{
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

double from_bits(std::uint64_t b)
{
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

/* The largest positive double v for which fits(v) holds, fits being
 * monotone (true below a threshold); std::nullopt when not even the
 * smallest positive double fits. */
template<class Fits>
std::optional<double> largest_fitting(const Fits& fits)
{
    double lo = std::numeric_limits<double>::denorm_min();
    if (!fits(lo)) return std::nullopt;
    double hi = 1.0;
    while (fits(hi)) {
        lo = hi;
        hi *= 2.0;
        if (!std::isfinite(hi)) return std::numeric_limits<double>::max();
    }
    std::uint64_t L = bits_of(lo), H = bits_of(hi);
    while (H - L > 1) {
        const std::uint64_t M = L + (H - L) / 2;
        if (fits(from_bits(M))) L = M;
        else H = M;
    }
    return from_bits(L);
}

bool finite_box(double a, double b)
{
    return std::isfinite(a) && std::isfinite(b) && a <= b;
}

}  // namespace

/* ---- settings ---------------------------------------------------------- */

FdSettings fd_settings_of(const OptimizerSpec& sp)
{
    FdSettings s;
    const bool hybrid = sp.grad_strategy.value_or(GradStrategy::Numerical) ==
                        GradStrategy::Hybrid;
    const bool unit = sp.space && sp.space->normalised();
    s.rule = (hybrid || unit) ? FdRule::Width : FdRule::Ceres;

    if (const auto it = sp.param.find("fd_kernel"); it != sp.param.end())
        throw std::runtime_error("PARAM fd_kernel must be inhouse or ceres "
                                 "(got " + num(it->second) + ")");
    if (const auto it = sp.param_raw.find("fd_kernel");
        it != sp.param_raw.end()) {
        if (it->second == "ceres" || it->second == "CERES")
            s.rule = FdRule::Ceres;
        else if (it->second == "inhouse" || it->second == "INHOUSE")
            s.rule = FdRule::Width;
        else
            throw std::runtime_error("PARAM fd_kernel must be inhouse or "
                                     "ceres (got '" + it->second + "')");
        s.rule_written = true;
    }

    if (const auto it = sp.param_raw.find("fd_step"); it != sp.param_raw.end())
        throw std::runtime_error("PARAM fd_step must be a positive finite "
                                 "number (got '" + it->second + "')");
    if (const auto it = sp.param.find("fd_step"); it != sp.param.end()) {
        if (!(it->second > 0.0) || !std::isfinite(it->second))
            throw std::runtime_error("PARAM fd_step must be a positive finite "
                                     "number (got " + num(it->second) + ")");
        s.fd_step = it->second;
        s.source = sp.fd_step_from_noise_auto ? FdStepSource::NoiseAuto
                                              : FdStepSource::Param;
    }

    const std::string dm_rule =
        "PARAM diff_method must be 0 (CENTRAL, the default) or 1 (FORWARD)";
    if (const auto it = sp.param_raw.find("diff_method");
        it != sp.param_raw.end())
        throw std::runtime_error(dm_rule + " (got '" + it->second + "')");
    if (const auto it = sp.param.find("diff_method"); it != sp.param.end()) {
        const double v = it->second;
        if (v == 0.0) {
            s.scheme = FdScheme::Central;
        } else if (v == 1.0) {
            s.scheme = FdScheme::Forward;
        } else if (v == 2.0) {
            throw std::runtime_error(
                "PARAM diff_method 2 (RIDDERS) is refused: Ridders' "
                "extrapolation ignored PARAM fd_step, started from a step of "
                "0.32 x max(|u|, 1) -- outside most OPTIM boxes -- and took "
                "up to 20 evaluations per parameter; use diff_method 0 "
                "(CENTRAL, the default) or 1 (FORWARD)");
        } else {
            throw std::runtime_error(dm_rule + " (got " + num(v) + ")");
        }
    }
    return s;
}

std::string fd_params_written(const OptimizerSpec& sp)
{
    std::vector<std::string> names;
    for (const char* k : {"fd_step", "fd_kernel", "diff_method"})
        if (sp.param.count(k) || sp.param_raw.count(k)) names.push_back(k);
    if (names.empty()) return "";
    std::string out = "PARAM " + names[0];
    for (std::size_t i = 1; i < names.size(); ++i)
        out += (i + 1 == names.size() ? " and " : ", ") + names[i];
    return out;
}

/* ---- the stencil ------------------------------------------------------- */

std::optional<FdStencil> fd_plan(double u, double a, double b, FdRule rule,
                                 FdScheme scheme, double fd_step)
{
    FdStencil st;
    const double h = fd_step_at(rule, fd_step, u);
    st.h = h;
    if (std::isnan(u)) {
        st.p1 = u + h;
        st.p2 = u - h;
        return st;
    }
    if (u < a || u > b) {
        /* Outside the box (an optimiser that ignores the bounds): towards
         * it (owner decision D4). */
        st.outside = true;
        st.turned = true;
        st.s = u < a ? h : -h;
        st.p1 = u + st.s;
        if (scheme == FdScheme::Central) {
            st.kind = FdStencil::Kind::OneSided;
            st.p2 = u + 2.0 * st.s;
        } else {
            st.kind = FdStencil::Kind::Forward;
        }
        const auto out_by = [a, b](double x) {
            return x < a ? a - x : (x > b ? x - b : 0.0);
        };
        const double d = out_by(u);
        if (out_by(st.p1) > d ||
            (st.kind == FdStencil::Kind::OneSided && out_by(st.p2) > d))
            throw std::logic_error(
                "fd_plan: the finite-difference stencil of u = " + num(u) +
                ", outside the box [" + num(a) + ", " + num(b) +
                "], reaches further outside than u with h = " + num(h) +
                " (a box fd_fit accepts rules this out: the stencil of its "
                "bound of larger magnitude fits inside it, which bounds "
                "fd_step by 1 for CENTRAL and by 2 for FORWARD)");
        return st;
    }
    if (scheme == FdScheme::Forward) {
        st.kind = FdStencil::Kind::Forward;
        const double xp = u + h;
        if (xp <= b) {
            st.s = h;
            st.p1 = xp;
            return st;
        }
        const double xm = u - h;
        if (xm >= a) {
            st.s = -h;
            st.p1 = xm;
            st.turned = true;
            return st;
        }
        return std::nullopt;
    }
    const double xp = u + h;
    const double xm = u - h;
    if (xm >= a && xp <= b) {
        st.p1 = xp;
        st.p2 = xm;
        return st;
    }
    const bool below = xm < a;
    const bool above = xp > b;
    st.kind = FdStencil::Kind::OneSided;
    st.turned = true;
    if (below && !above) {
        const double x2 = u + 2.0 * h;
        if (x2 <= b) {
            st.s = h;
            st.p1 = xp;
            st.p2 = x2;
            return st;
        }
    }
    if (above && !below) {
        const double x2 = u + 2.0 * -h;
        if (x2 >= a) {
            st.s = -h;
            st.p1 = xm;
            st.p2 = x2;
            return st;
        }
    }
    return std::nullopt;
}

/* ---- the box checks ---------------------------------------------------- */

FdFit fd_fit(double a, double b, FdRule rule, FdScheme scheme, double fd_step)
{
    if (!finite_box(a, b) || !(fd_step > 0.0) || !std::isfinite(fd_step))
        return {false, a};
    return fit_pieces(a, b, rule, scheme, fd_step);
}

std::optional<double> fd_largest_fitting_step(double a, double b,
                                              FdRule rule, FdScheme scheme)
{
    if (!finite_box(a, b)) return std::nullopt;
    return largest_fitting([&](double f) {
        return fit_pieces(a, b, rule, scheme, f).ok;
    });
}

std::string fd_floor_text(double v, int digits)
{
    if (!(v > 0.0) || !std::isfinite(v) || digits < 1) return num(v);
    int e = static_cast<int>(std::floor(std::log10(v)));
    /* m x 10^(e - digits + 1), exactly as decimal text */
    const auto text = [digits](long long m, int exp10) {
        return std::to_string(m) + "e" + std::to_string(exp10 - digits + 1);
    };
    /* the nearest `digits`-digit decimal, then down until it is <= v */
    long long m = 0;
    const long long top =
        static_cast<long long>(std::llround(std::pow(10.0, digits)));
    for (int k = 0; k < 3; ++k) {
        const LD scale = std::pow(static_cast<LD>(10), e - digits + 1);
        m = std::llround(static_cast<LD>(v) / scale);
        if (m >= top) { ++e; continue; }
        if (m < top / 10) { --e; continue; }
        break;
    }
    double p = std::strtod(text(m, e).c_str(), nullptr);
    while (p > v && m > 1) {
        --m;
        if (m < top / 10) {   /* keep `digits` digits: 99999 -> 999999 */
            m = m * 10 + 9;
            --e;
        }
        p = std::strtod(text(m, e).c_str(), nullptr);
    }
    char out[64];
    std::snprintf(out, sizeof out, "%.*g", digits, p);
    return out;
}

namespace {

/* "h = 0.4 of each box width" and its variants (fd_describe). */
std::string step_rule_text(const FdSettings& s, bool normalised)
{
    const std::string f = num(s.fd_step);
    if (s.rule == FdRule::Width)
        return "h = " + f + (normalised ? " of each box width"
                                        : " x max(|u|, 1) in internal units");
    return "h = max(2^-26, " + f + " x |u|)" +
           (normalised ? " in the unit box" : " in internal units");
}

std::string source_text(FdStepSource src)
{
    switch (src) {
        case FdStepSource::Default:   return "default fd_step";
        case FdStepSource::Param:     return "PARAM fd_step";
        case FdStepSource::NoiseAuto: return "fd_step from NOISE AUTO";
    }
    return "";
}

}  // namespace

std::string fd_describe(const FdSettings& s, bool normalised, bool hybrid)
{
    std::string d = std::string(s.scheme == FdScheme::Central ? "CENTRAL"
                                                              : "FORWARD") +
                    ", " + step_rule_text(s, normalised) + " (" +
                    source_text(s.source) + "; " +
                    (s.rule == FdRule::Width ? "in-house rule" : "Ceres' rule") +
                    ")";
    if (s.scheme == FdScheme::Central)
        d += "; next to a bound: one-sided 3-point into the box, same h, "
             "same cost";
    else
        d += "; next to the upper bound: backward, same h, same cost";
    if (hybrid)
        d += "; HYBRID: for the parameters without an analytical derivative, "
             "and a one-sided or forward difference takes the loss at the "
             "point itself -- the optimiser's own when it has just evaluated "
             "that point, else one evaluation more";
    return d;
}

std::string fd_box_error(const OptimizerSpec& sp, const FdSettings& s)
{
    const std::size_t n = std::min(sp.low.size(), sp.high.size());
    const bool normalised = sp.space && sp.space->normalised();
    std::string bad;
    std::optional<double> smallest_fit;
    bool none_fits = false, decades = false;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = sp.low[i], b = sp.high[i];
        const FdFit fit = fd_fit(a, b, s.rule, s.scheme, s.fd_step);
        if (fit.ok) continue;
        const std::optional<double> best =
            fd_largest_fitting_step(a, b, s.rule, s.scheme);
        const std::string lbl = label_of(sp, i);
        std::string box = "[" + num(a) + ", " + num(b) + "]";
        double where = fit.where;
        if (sp.space && i < sp.space->dim()) {
            const DimSpec& d = sp.space->at(i);
            const std::string real = "[" + num(d.lo) + ", " + num(d.hi) + "]";
            if (real != box) box = real + " (internal " + box + ")";
            where = sp.space->to_real(i, fit.where);
            if (d.tf.kind == ScaleKind::Linear && d.lo > 0.0 &&
                d.hi >= 10.0 * d.lo)
                decades = true;
        }
        bad += "\n  - " + lbl + ": OPTIM box " + box + "; no stencil fits at " +
               lbl + " = " + num(where);
        if (best) {
            bad += "; fd_step up to " + fd_floor_text(*best) + " fits it";
            smallest_fit = smallest_fit ? std::min(*smallest_fit, *best) : *best;
        } else {
            bad += "; no fd_step fits it: Ceres' rule never steps below "
                   "2^-26 = 1.49e-08, and this box is narrower than three "
                   "such steps";
            none_fits = true;
        }
    }
    if (bad.empty()) return "";

    std::string step;
    switch (s.source) {
        case FdStepSource::Default:
            step = "the default fd_step " + num(s.fd_step);
            break;
        case FdStepSource::Param:
            step = "PARAM fd_step " + num(s.fd_step);
            break;
        case FdStepSource::NoiseAuto:
            step = "fd_step " + num(s.fd_step) + ", derived by the NOISE "
                   "block's AUTO from the box diagonal,";
            break;
    }
    std::string msg =
        step + " does not fit these OPTIM boxes: no finite-difference point "
        "is evaluated outside the box, and next to a bound the " +
        std::string(s.scheme == FdScheme::Central
                        ? "CENTRAL stencil turns one-sided and needs two steps"
                        : "FORWARD stencil steps backward and needs one step") +
        " of " + step_rule_text(s, normalised) + " (" +
        (s.rule == FdRule::Width ? "in-house rule" : "Ceres' rule") +
        ") inside it; the step is not shrunk:" + bad + "\n  ";
    std::vector<std::string> fixes;
    if (smallest_fit) {
        std::string f = "set PARAM fd_step to at most " +
                        fd_floor_text(*smallest_fit);
        if (s.source == FdStepSource::NoiseAuto)
            f += " (a PARAM fd_step overrides NOISE AUTO)";
        fixes.push_back(f);
    }
    if (none_fits && s.rule == FdRule::Ceres)
        fixes.push_back("use PARAM fd_kernel inhouse, whose step has no "
                        "floor");
    if (!normalised) {
        fixes.push_back("use OPTIM_SCALE AUTO (fd_step is then a fraction "
                        "of each box width)");
        if (decades)
            fixes.push_back("give a positive parameter that spans decades a "
                            "LOG box");
    }
    for (std::size_t k = 0; k < fixes.size(); ++k) {
        if (k == 0) {
            fixes[k][0] = static_cast<char>(std::toupper(
                static_cast<unsigned char>(fixes[k][0])));
            msg += fixes[k];
        } else {
            msg += "; or " + fixes[k];
        }
    }
    return msg + ".";
}

/* ---- ASA's probes ------------------------------------------------------ */

namespace {

FdFit asa_fit(double lo, double hi, double delta, bool curvature)
{
    /* The probes as ASA computes them, in double (asa.c), bounded by real
     * linear functions of x as fit_pieces bounds the step, so that a
     * computed probe outside the box implies a strict real inequality: the
     * tangents are fl(E x) with E = fl(1 + Delta), or fl(E' x) with E' =
     * fl(1 - Delta) -- the factors ASA multiplies by, computed here the same
     * way, so fl(E x) > hi implies E x > hi. The curvature adds t = fl(Delta
     * |x|), and 2t, to x; t <= D |x| + c with D = upper_factor(Delta), where
     * c = 2^-1075, half the smallest double, covers a subnormal product (|x|
     * below DBL_MIN / Delta). The strict inequalities are then decided as in
     * fit_pieces. Not covered: ASA's third curvature branch tests x +-
     * Delta |x| and evaluates (1 +- Delta) x, which round differently by an
     * ulp at a point or two of some boxes (about one in six), whatever the
     * Delta; the host counts such an evaluation and warns at the first
     * (fd_stencil.h). */
    FdFit fit;
    const LD L = lo, Hh = hi;
    const LD E = 1.0 + delta, Em = 1.0 - delta;
    /* g(x) < 0 forms, with P = E x, M = E' x. */
    const Lin p_over{-E, Hh};      /* hi - P < 0: P over the box */
    const Lin p_under{E, -L};      /* P - lo < 0: P under it */
    const Lin m_over{-Em, Hh};
    const Lin m_under{Em, -L};
    {
        const std::vector<std::pair<Lin, Lin>> conj = {
            {p_over, m_over}, {p_over, m_under},
            {p_under, m_over}, {p_under, m_under}};
        const LD w = first_failure(
            candidates(L, Hh, {p_over, p_under, m_over, m_under}), conj);
        if (!std::isnan(w)) {
            fit.ok = false;
            fit.where = std::clamp(static_cast<double>(w), lo, hi);
            return fit;
        }
    }
    if (!curvature) return fit;
    /* The diagonal curvature, on x <= 0 and x >= 0 (|x| = sg x): Q+ = x +
     * t, Q- = x - t, Q2+ = x + 2t, Q2- = x - 2t, each bounded with t <= D
     * |x| + c; c only where the product can be subnormal, |x| < DBL_MIN /
     * Delta (a piece of its own, overlapping the other at its end). Next to
     * the upper bound (Q+ over it) ASA evaluates Q2- and Q-, next to the
     * lower one (Q- under it) Q2+ and Q+. */
    const LD D = upper_factor(delta);
    const LD c = std::ldexp(static_cast<LD>(1), -1075);
    const LD tiny = static_cast<LD>(std::numeric_limits<double>::min()) /
                    static_cast<LD>(delta) * (1 + std::ldexp(static_cast<LD>(1), -60));
    struct Part { LD sg, s, t, c; };
    std::vector<Part> parts;
    for (const LD sg : {static_cast<LD>(-1), static_cast<LD>(1)}) {
        /* sg x in [0, tiny] with c, [tiny, inf) without */
        const LD inf = std::numeric_limits<LD>::infinity();
        for (const auto& [from, to, cc] :
             {std::tuple<LD, LD, LD>{0, tiny, c},
              std::tuple<LD, LD, LD>{tiny, inf, 0}}) {
            LD s0 = sg < 0 ? -to : from, t0 = sg < 0 ? -from : to;
            s0 = std::max(s0, L);
            t0 = std::min(t0, Hh);
            if (s0 <= t0) parts.push_back({sg, s0, t0, cc});
        }
    }
    for (const Part& q : parts) {
        const LD k = D * q.sg;
        const Lin qp_over{-(1 + k), Hh - q.c};         /* hi - (x + t) */
        const Lin qm_under{1 - k, -L - q.c};           /* x - t - lo */
        const Lin q2m_under{1 - 2 * k, -L - 2 * q.c};  /* x - 2t - lo */
        const Lin q2p_over{-(1 + 2 * k), Hh - 2 * q.c}; /* hi - (x + 2t) */
        const std::vector<std::pair<Lin, Lin>> conj = {
            {qp_over, q2m_under}, {qp_over, qm_under}, {qm_under, q2p_over}};
        const LD w = first_failure(
            candidates(q.s, q.t, {qp_over, qm_under, q2m_under, q2p_over}),
            conj);
        if (!std::isnan(w)) {
            fit.ok = false;
            fit.where = std::clamp(static_cast<double>(w), lo, hi);
            return fit;
        }
    }
    return fit;
}

}  // namespace

FdFit asa_probe_fit(double lo, double hi, double delta, bool curvature)
{
    if (!finite_box(lo, hi) || !(delta >= 0.0) || !std::isfinite(delta))
        return {false, lo};
    return asa_fit(lo, hi, delta, curvature);
}

std::optional<double> asa_largest_fitting_delta(double lo, double hi,
                                                bool curvature)
{
    if (!finite_box(lo, hi)) return std::nullopt;
    return largest_fitting([&](double d) {
        return asa_fit(lo, hi, d, curvature).ok;
    });
}

}  // namespace tadah::hpo

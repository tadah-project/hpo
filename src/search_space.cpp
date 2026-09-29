#include <tadah/hpo/search_space.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tadah::hpo {

const char* scale_kind_name(ScaleKind k)
{
    switch (k) {
        case ScaleKind::Linear: return "LIN";
        case ScaleKind::Log:    return "LOG";
        case ScaleKind::Asinh:  return "ASINH";
    }
    return "?";
}

const char* box_origin_name(BoxOrigin o)
{
    switch (o) {
        case BoxOrigin::Explicit:      return "OPTIM";
        case BoxOrigin::AutoLinear:    return "AUTO FRAC";
        case BoxOrigin::AutoGeometric: return "AUTO RATIO";
        case BoxOrigin::Adapted:       return "ADAPTED";
        case BoxOrigin::AutoLinearNonneg: return "AUTO FRAC>=0";
    }
    return "?";
}

double DimTransform::fwd(double x) const
{
    switch (kind) {
        case ScaleKind::Linear: return x;
        case ScaleKind::Log:    return std::log(x);
        case ScaleKind::Asinh:  return std::asinh(x / c);
    }
    return x;
}

double DimTransform::inv(double t) const
{
    switch (kind) {
        case ScaleKind::Linear: return t;
        case ScaleKind::Log:    return std::exp(t);
        case ScaleKind::Asinh:  return c * std::sinh(t);
    }
    return t;
}

double DimTransform::dinv(double t) const
{
    switch (kind) {
        case ScaleKind::Linear: return 1.0;
        case ScaleKind::Log:    return std::exp(t);
        case ScaleKind::Asinh:  return c * std::cosh(t);
    }
    return 1.0;
}

SearchSpace SearchSpace::build(std::vector<DimSpec> dims, const ScalePolicy& policy)
{
    SearchSpace s;
    s.policy_ = policy;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        DimSpec& d = dims[i];
        const std::string& lbl = d.label.empty() ? d.key : d.label;
        if (!std::isfinite(d.lo) || !std::isfinite(d.hi) || !(d.hi > d.lo)) {
            std::ostringstream oss;
            oss << "OPTIM " << lbl << ": bounds must be finite with high > low (got ["
                << d.lo << ", " << d.hi << "])";
            throw std::runtime_error(oss.str());
        }
        /* LOG_HP rule on untagged dims (an explicit tag always wins). */
        if (d.tf.kind == ScaleKind::Linear && !d.auto_log && policy.log_ratio &&
            d.lo > 0.0 && d.hi / d.lo >= *policy.log_ratio) {
            d.tf.kind = ScaleKind::Log;
            d.auto_log = true;
        }
        if (d.tf.kind == ScaleKind::Log && !(d.lo > 0.0)) {
            std::ostringstream oss;
            oss << "OPTIM " << lbl << ": LOG scale requires low > 0 (got " << d.lo
                << "); use ASINH <c> for sign-crossing ranges";
            throw std::runtime_error(oss.str());
        }
        if (d.tf.kind == ScaleKind::Asinh && !(std::isfinite(d.tf.c) && d.tf.c > 0.0)) {
            std::ostringstream oss;
            oss << "OPTIM " << lbl << ": ASINH requires a positive linear scale c (got "
                << d.tf.c << ")";
            throw std::runtime_error(oss.str());
        }
        if (d.is_int && policy.normalise)
            throw std::runtime_error("OPTIM " + lbl +
                ": integer dimensions cannot be normalised (OPTIM_SCALE NONE)");
        s.t_lo_.push_back(d.tf.fwd(d.lo));
        s.t_hi_.push_back(d.tf.fwd(d.hi));
        if (!(s.t_hi_.back() > s.t_lo_.back()) ||
            !std::isfinite(s.t_lo_.back()) || !std::isfinite(s.t_hi_.back())) {
            std::ostringstream oss;
            oss << "OPTIM " << lbl << ": transformed bounds are degenerate ("
                << scale_kind_name(d.tf.kind) << ": [" << s.t_lo_.back() << ", "
                << s.t_hi_.back() << "])";
            throw std::runtime_error(oss.str());
        }
        s.labels_.push_back(lbl);
    }
    s.dims_ = std::move(dims);
    return s;
}

double SearchSpace::to_internal(std::size_t i, double x) const
{
    const double t = dims_[i].tf.fwd(x);
    if (!policy_.normalise) return t;
    return (t - t_lo_[i]) / (t_hi_[i] - t_lo_[i]);
}

namespace {

/* The spacing of the doubles at |m| (the smallest positive double at 0). */
double ulp_at(double m)
{
    m = std::fabs(m);
    return std::nextafter(m, std::numeric_limits<double>::infinity()) - m;
}

}  // namespace

std::optional<double> SearchSpace::snapped_onto_box(std::size_t i,
                                                    double x) const
{
    const DimSpec& d = dims_[i];
    if (!(x < d.lo || x > d.hi)) return std::nullopt;   /* inside, or NaN */
    const double bound = x < d.lo ? d.lo : d.hi;
    const double tol = static_cast<double>(kSnapUlps);
    /* real coordinates (x and the bound are close: the difference is
     * exact) */
    const double scale = d.tf.kind == ScaleKind::Linear
                             ? std::max(std::fabs(d.lo), std::fabs(d.hi))
                             : std::fabs(bound);
    if (std::fabs(x - bound) <= tol * ulp_at(scale)) return bound;
    /* the optimiser's coordinates */
    const double u = to_internal(i, x);
    const double a = internal_lo(i), b = internal_hi(i);
    const double ub = x < d.lo ? a : b;
    if (std::isfinite(u) &&
        std::fabs(u - ub) <= tol * ulp_at(std::max(std::fabs(a), std::fabs(b))))
        return bound;
    return std::nullopt;
}

double SearchSpace::to_real(std::size_t i, double u) const
{
    if (!anc_off_.empty() && anc_off_[i] != anc_off_[i + 1]) {
        /* The anchor nearest to u (the lower one on a tie) maps it to its
         * seed when u lies within that anchor's window: an exact u0
         * (distance 0) always, else the seed as a start-point rounding
         * moved it (seed_anchor_tolerance, search_space.h). Dim i's
         * anchors are sorted by u and distinct (+0 for both zeros). A NaN
         * u matches nothing and falls through. */
        const std::size_t b = anc_off_[i], e = anc_off_[i + 1];
        std::size_t k = b, stop = e;
        if (e - b > 24) {
            /* many anchors (a run seeded with many INIT points): only the
             * two neighbours of u can be nearest. Up to 24, the linear
             * scan's predictable branches are cheaper than the search. */
            k = static_cast<std::size_t>(
                std::lower_bound(anc_u_.begin() + static_cast<std::ptrdiff_t>(b),
                                 anc_u_.begin() + static_cast<std::ptrdiff_t>(e), u) -
                anc_u_.begin());
            stop = std::min(e, k + 1);
            if (k > b) --k;
        }
        std::size_t best = e;
        double best_d = std::numeric_limits<double>::infinity();
        for (; k < stop; ++k) {         /* branches rarely taken: predictable */
            const double d = std::fabs(u - anc_u_[k]);
            if (d < best_d) { best = k; best_d = d; }
        }
        if (best != e && best_d <= anc_tol_[best]) return anc_x_[best];
    }
    /* The ends of the internal box map to exactly the ends of the real
     * one, and a u inside it into [lo, hi]: T^-1 rounds (exp, sinh, the
     * affine map of the unit box), and LOG / ASINH ends used to come back
     * one ulp outside the box (LOG [0.01, 10] at u = 1: 10.000000000000002)
     * -- an evaluation outside the box, recorded as it was. A u outside the
     * internal box (where the optimiser put it) maps as it always did.
     * (Owner decision D6, 2026-09-29.) */
    const DimSpec& d = dims_[i];
    const double ulo = internal_lo(i), uhi = internal_hi(i);
    if (u == ulo) return d.lo;
    if (u == uhi) return d.hi;
    const double t = policy_.normalise ? t_lo_[i] + u * (t_hi_[i] - t_lo_[i]) : u;
    const double x = d.tf.inv(t);
    if (u > ulo && u < uhi) return std::clamp(x, d.lo, d.hi);
    return x;
}

double SearchSpace::seed_anchor_tolerance(std::size_t i, double u0) const
{
    const auto ulp = [](double v) {
        v = std::fabs(v);
        return std::nextafter(v, std::numeric_limits<double>::infinity()) - v;
    };
    const double m = std::max(std::fabs(internal_lo(i)), std::fabs(internal_hi(i)));
    return std::min(kSeedAnchorUlps * ulp(m),
                    std::max(kSeedAnchorMaxTol, kSeedAnchorSeedUlps * ulp(u0)));
}

double SearchSpace::dreal_dinternal(std::size_t i, double u) const
{
    const double t = policy_.normalise ? t_lo_[i] + u * (t_hi_[i] - t_lo_[i]) : u;
    const double dt_du = policy_.normalise ? (t_hi_[i] - t_lo_[i]) : 1.0;
    return dims_[i].tf.dinv(t) * dt_du;
}

double SearchSpace::internal_lo(std::size_t i) const
{ return policy_.normalise ? 0.0 : t_lo_[i]; }

double SearchSpace::internal_hi(std::size_t i) const
{ return policy_.normalise ? 1.0 : t_hi_[i]; }

std::vector<double> SearchSpace::internal_low() const
{
    std::vector<double> v(dims_.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = internal_lo(i);
    return v;
}

std::vector<double> SearchSpace::internal_high() const
{
    std::vector<double> v(dims_.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = internal_hi(i);
    return v;
}

std::string SearchSpace::describe_dim(std::size_t i) const
{
    const DimSpec& d = dims_[i];
    std::string s = scale_kind_name(d.tf.kind);
    if (d.tf.kind == ScaleKind::Log && d.auto_log) s += " (auto)";
    if (d.tf.kind == ScaleKind::Asinh) {
        std::ostringstream oss;
        oss << " c=" << std::setprecision(4) << d.tf.c;
        s += oss.str();
    }
    return s;
}

void SearchSpace::require_inside_real(const std::vector<double>& x,
                                      const std::string& what) const
{
    if (x.size() != dims_.size()) {
        std::ostringstream oss;
        oss << what << ": point has " << x.size() << " coordinate(s), expected "
            << dims_.size();
        throw std::runtime_error(oss.str());
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        const DimSpec& d = dims_[i];
        if (!std::isfinite(x[i])) {
            std::ostringstream oss;
            oss << what << ": " << labels_[i] << " = " << x[i] << " is not finite";
            throw std::runtime_error(oss.str());
        }
        if (x[i] < d.lo || x[i] > d.hi) {
            std::ostringstream oss;
            oss << what << ": " << labels_[i] << " = " << x[i]
                << " lies outside its OPTIM bounds [" << d.lo << ", " << d.hi
                << "]; move the seed inside the box or widen the box";
            throw std::runtime_error(oss.str());
        }
    }
}

SearchSpace SearchSpace::with_boxes(const std::vector<std::pair<double,double>>& boxes) const
{
    if (boxes.size() != dims_.size())
        throw std::runtime_error("SearchSpace::with_boxes: box count mismatch");
    std::vector<DimSpec> dims = dims_;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (boxes[i].first != dims[i].lo || boxes[i].second != dims[i].hi) {
            dims[i].lo = boxes[i].first;
            dims[i].hi = boxes[i].second;
            dims[i].origin = BoxOrigin::Adapted;
        }
        /* an auto-LOG tag was decided on the old box; re-decide from scratch */
        if (dims[i].auto_log) { dims[i].tf.kind = ScaleKind::Linear; dims[i].auto_log = false; }
    }
    SearchSpace s = build(std::move(dims), policy_);
    s.version_ = version_ + 1;
    return s;
}

SearchSpace SearchSpace::with_seed_anchors(
    const std::vector<std::vector<double>>& seeds) const
{
    SearchSpace s = *this;
    s.anc_off_.clear();          // s maps through the plain formula below
    s.anc_u_.clear();
    s.anc_x_.clear();
    s.anc_tol_.clear();
    const std::size_t n = dims_.size();
    struct Entry { double u, x; };
    std::vector<Entry> ent;
    std::vector<std::size_t> off(n + 1, 0);
    std::vector<double> au, ax;
    for (std::size_t i = 0; i < n; ++i) {
        off[i] = au.size();
        ent.clear();
        for (const auto& p : seeds) {
            if (p.size() != n) continue;
            const double x = p[i];
            if (!std::isfinite(x) || x < dims_[i].lo || x > dims_[i].hi) continue;
            const double u = s.to_internal(i, x);
            if (!std::isfinite(u)) continue;
            /* key: +0 for both zeros (to_real's lookup compares with ==).
             * A coordinate whose plain round trip is exact is anchored as
             * well: only the anchor window maps a PERTURBED u0 back to it. */
            ent.push_back({u == 0.0 ? 0.0 : u, x});
        }
        /* Stable: within one u the seeds keep their order, so the first
         * entry of each run of equal u is the owner. */
        std::stable_sort(ent.begin(), ent.end(),
                         [](const Entry& a, const Entry& b) { return a.u < b.u; });
        for (std::size_t k = 0; k < ent.size();) {
            const Entry& own = ent[k];
            au.push_back(own.u);
            ax.push_back(own.x);
            while (k < ent.size() && ent[k].u == own.u) ++k;
        }
    }
    off[n] = au.size();
    if (!au.empty()) {
        std::vector<double> tol(au.size());
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = off[i]; k < off[i + 1]; ++k)
                tol[k] = s.seed_anchor_tolerance(i, au[k]);
        s.anc_off_ = std::move(off);
        s.anc_u_   = std::move(au);
        s.anc_x_   = std::move(ax);
        s.anc_tol_ = std::move(tol);
    }
    return s;
}

}  // namespace tadah::hpo

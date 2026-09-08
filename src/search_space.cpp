#include <tadah/hpo/search_space.h>

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

double SearchSpace::to_real(std::size_t i, double u) const
{
    const double t = policy_.normalise ? t_lo_[i] + u * (t_hi_[i] - t_lo_[i]) : u;
    return dims_[i].tf.inv(t);
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

std::string SearchSpace::fingerprint() const
{
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << fingerprint_u64();
    return oss.str();
}

std::uint64_t SearchSpace::fingerprint_u64() const
{
    /* FNV-1a over the exact bit patterns of (kind, c, lo, hi) per dim plus
     * the normalise flag: stable across processes and compilers for the
     * same config, sensitive to any coordinate-system change. */
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, std::size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    const unsigned char nrm = policy_.normalise ? 1 : 0;
    mix(&nrm, 1);
    for (const DimSpec& d : dims_) {
        const unsigned char k = static_cast<unsigned char>(d.tf.kind);
        mix(&k, 1);
        mix(&d.tf.c, sizeof d.tf.c);
        mix(&d.lo, sizeof d.lo);
        mix(&d.hi, sizeof d.hi);
    }
    return h;
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

}  // namespace tadah::hpo

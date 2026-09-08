#pragma once
/*
 * SearchSpace -- the one owner of the optimiser coordinate system.
 *
 * Every OPTIM dimension i has a REAL box [lo_i, hi_i] (what the user wrote
 * and what the model sees) and an INTERNAL coordinate u_i (what every
 * optimiser backend iterates on). The map is
 *
 *     t_i = T_i(x_i)                         per-dim shape (LIN / LOG / ASINH c)
 *     u_i = (t_i - T_i(lo_i)) / (T_i(hi_i) - T_i(lo_i))   when normalised (OPTIM_SCALE AUTO)
 *     u_i = t_i                                          when not (OPTIM_SCALE NONE)
 *
 * so under AUTO (the default) every dimension spans [0,1] and one STEP /
 * XTOL_ABS / fd_step means the same fraction of every box; under NONE the
 * internal coordinates are exactly the pre-1.5 ones (raw values, ln for
 * LOG_HP dims), which reproduces every legacy trajectory byte-identically.
 *
 * Shapes:
 *   LIN      t = x
 *   LOG      t = ln x          (requires lo > 0; opt-in per line or via LOG_HP)
 *   ASINH c  t = asinh(x / c)  (sign-crossing, multi-decade dims; slope 1/c
 *                               near 0, log-like beyond |x| ~ c; C-infinity)
 *
 * The object is immutable after build() and shared (shared_ptr<const>)
 * between the outer OptimizerSpec, every sub-spec / stage-spec clone,
 * HPO_Host, the basin pools and the journal writer. It is a pure function
 * of the configuration, so every MPI rank builds the identical space.
 *
 * Coordinate contract for the rest of HPO:
 *   - OptimizerSpec::low/high are INTERNAL bounds (filled by bind_space);
 *   - OptimizerSpec::in()/out() map REAL <-> INTERNAL through the space;
 *   - host(...) always receives REAL vectors (adapters call sp.out(u));
 *   - persisted artefacts (best_*.tadah, params, WARM seeds, basin pools)
 *     are REAL-space.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tadah::hpo {

enum class ScaleKind : std::uint8_t { Linear, Log, Asinh };
const char* scale_kind_name(ScaleKind k);

/* Per-dimension shape T_i (see file comment). */
struct DimTransform {
    ScaleKind kind = ScaleKind::Linear;
    double    c    = 1.0;          // ASINH linear scale (unused otherwise)

    double fwd (double x) const;   // T(x)
    double inv (double t) const;   // T^{-1}(t)
    double dinv(double t) const;   // d T^{-1} / dt
};

/* How a real box came to be (journal "origin" column; AUTO boxes and
 * BOUND_ADAPT arrive in later phases). */
enum class BoxOrigin : std::uint8_t { Explicit, AutoLinear, AutoGeometric, Adapted };
const char* box_origin_name(BoxOrigin o);

struct DimSpec {
    std::string  key;                         // OPTIM key (e.g. "CGRID2B")
    std::size_t  user_index = 0;              // 1-based OPTIM index as written (0 = scalar key)
    std::string  label;                       // journal label ("CGRID2B[3]", "RCUT2B")
    double       lo = 0.0, hi = 0.0;          // REAL bounds
    DimTransform tf;                          // shape
    bool         auto_log = false;            // shape chosen by the LOG_HP rule
    BoxOrigin    origin = BoxOrigin::Explicit;
    double       x0 = std::numeric_limits<double>::quiet_NaN();  // CONFIG seed, NaN if unknown
    bool         is_int = false;
};

/* Search-space policy parsed from the config:
 *   log_ratio  -- LOG_HP: untagged dims with lo > 0 and hi/lo >= ratio get LOG
 *   normalise  -- OPTIM_SCALE AUTO (true) | NONE (false). The struct
 *   default is NONE for hand-built specs (analyse_logscale shim);
 *   HPOTargets defaults to AUTO. */
struct ScalePolicy {
    std::optional<double> log_ratio;
    bool                  normalise = false;
};

namespace ss_detail {
/* One element accessor for dlib::matrix (operator()) and std::vector
 * (operator[]) so the vector maps share a single implementation. */
template<class V>
auto elem(V& v, std::size_t i) -> decltype(v(static_cast<long>(i)))
{ return v(static_cast<long>(i)); }
template<class V>
auto elem(V& v, std::size_t i) -> decltype(v[i])
{ return v[i]; }
template<class V>
std::size_t len(const V& v) { return static_cast<std::size_t>(v.size()); }
}  // namespace ss_detail

class SearchSpace final {
public:
    /* Validates (finite hi > lo; LOG needs lo > 0; ASINH needs finite c > 0;
     * integer dims cannot be normalised) and throws std::runtime_error
     * naming the dimension label on the first violation. Applies the
     * LOG_HP rule to untagged dims. */
    static SearchSpace build(std::vector<DimSpec> dims, const ScalePolicy& policy);

    std::size_t                      dim()        const noexcept { return dims_.size(); }
    const DimSpec&                   at(std::size_t i) const     { return dims_.at(i); }
    const std::vector<std::string>&  labels()     const noexcept { return labels_; }
    bool                             normalised() const noexcept { return policy_.normalise; }
    const ScalePolicy&               policy()     const noexcept { return policy_; }
    std::uint32_t                    version()    const noexcept { return version_; }

    /* scalar maps */
    double to_internal    (std::size_t i, double x) const;
    double to_real        (std::size_t i, double u) const;
    double dreal_dinternal(std::size_t i, double u) const;   // dx/du at u
    double internal_lo    (std::size_t i) const;
    double internal_hi    (std::size_t i) const;
    double internal_width (std::size_t i) const { return internal_hi(i) - internal_lo(i); }
    double real_width     (std::size_t i) const { return dims_[i].hi - dims_[i].lo; }
    /* The real displacement produced by an internal step du taken at u. */
    double real_step_at   (std::size_t i, double u, double du) const
    { return to_real(i, u + du) - to_real(i, u); }

    bool        is_nonlinear(std::size_t i) const { return dims_[i].tf.kind != ScaleKind::Linear; }
    std::string describe_dim(std::size_t i) const;   // "LIN" | "LOG (auto)" | "ASINH c=1e-2"

    std::vector<double> internal_low () const;
    std::vector<double> internal_high() const;

    /* vector maps (std::vector<double> or dlib column matrix) */
    template<class V> V to_internal(const V& x) const
    {
        V u = x;
        for (std::size_t i = 0; i < ss_detail::len(u) && i < dims_.size(); ++i)
            ss_detail::elem(u, i) = to_internal(i, ss_detail::elem(x, i));
        return u;
    }
    template<class V> V to_real(const V& u) const
    {
        V x = u;
        for (std::size_t i = 0; i < ss_detail::len(x) && i < dims_.size(); ++i)
            ss_detail::elem(x, i) = to_real(i, ss_detail::elem(u, i));
        return x;
    }
    /* g_u[i] = g_x[i] * dx_i/du_i  (chain rule for a gradient taken in real space) */
    template<class V> V pull_gradient(const V& u, const V& g_real) const
    {
        V g = g_real;
        for (std::size_t i = 0; i < ss_detail::len(g) && i < dims_.size(); ++i)
            ss_detail::elem(g, i) = ss_detail::elem(g_real, i) *
                                    dreal_dinternal(i, ss_detail::elem(u, i));
        return g;
    }

    /* Throws std::runtime_error naming the dim when a REAL point lies
     * outside its box (bounds inclusive) or is non-finite. */
    void require_inside_real(const std::vector<double>& x, const std::string& what) const;

    /* Stable identity of the coordinate system (kinds, c, lo, hi,
     * normalise) -- used to tag basin-pool checkpoints. Labels excluded. */
    std::string   fingerprint() const;       // 16 hex chars
    std::uint64_t fingerprint_u64() const;   // same value, for checkpoints

    /* Same shapes/labels, new real boxes (version + 1). BOUND_ADAPT. */
    SearchSpace with_boxes(const std::vector<std::pair<double,double>>& boxes) const;

private:
    SearchSpace() = default;
    std::vector<DimSpec>      dims_;
    std::vector<std::string>  labels_;
    std::vector<double>       t_lo_, t_hi_;   // T_i(lo_i), T_i(hi_i)
    ScalePolicy               policy_;
    std::uint32_t             version_ = 1;
};

using SearchSpacePtr = std::shared_ptr<const SearchSpace>;

}  // namespace tadah::hpo

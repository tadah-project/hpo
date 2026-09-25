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
 * LOG dims). Runs under NONE are not byte-identical to 1.5.0 and earlier:
 * numbers are no longer rounded on their way through the code, and seed
 * points are evaluated exactly (anchors below).
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
 *
 * Seed anchors. A REAL start point x0 (CONFIG or WARM value, a pool or
 * top-K point) enters an optimiser as u0 = to_internal(x0), and the plain
 * map need not give x0 back: lo + u0*(hi - lo) rounds (a small |x0| in a
 * wide LIN box comes back as a multiple of ulp(lo)), LOG and ASINH add
 * exp/log rounding. Some optimisers also move the start by rounding
 * before evaluating it (NLopt BOBYQA/COBYLA rescale it, u -> (u/s)*s;
 * PRAXIS maps it through atanh/tanh). with_seed_anchors() returns a copy
 * of the space whose to_real() maps each such u0 to its x0 -- exactly
 * equal u first, then any u within the anchor's window of a few units in
 * the last place (seed_anchor_tolerance()) -- so the start point is
 * evaluated at precisely the loaded doubles and every coordinate an
 * optimiser has not moved keeps its seed value. Every other u maps as
 * before. run_inner() anchors each optimiser run on its own
 * spec.optim_init; the anchored copy is immutable like every space.
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

/* Seed-anchor window: the half-width, in u, around an anchored seed's u0
 * within which to_real() returns the seed's exact x0. Per anchor
 * (seed_anchor_tolerance(i, u0)):
 *
 *   min(kSeedAnchorUlps * ulp(max(|internal lo|, |internal hi|)),
 *       max(kSeedAnchorMaxTol, kSeedAnchorSeedUlps * ulp(u0)))
 *
 * i.e. 4 ulp of the dimension's internal box (ulp(1) = 2.2e-16 under
 * OPTIM_SCALE AUTO), capped at 2^-27 -- but never below 2 ulp of the
 * seed's own u0.
 *
 * Why the box and not ulp(u0): the start-point rounding of NLopt PRAXIS
 * (x_bound_inv/x_bound: v = (u - mid)/half, atanh, tanh, mid + v*half) is
 * a few ulp of the box's midpoint/half-width, not of u0 -- near the lower
 * bound of the unit box (u0 ~ 1e-8) it is ~1e6 ulp(u0). Measured with
 * NLopt 2.10's code and glibc's atanh/tanh over 1e7 random boxes and
 * seeds: at most 3 ulp(box) (0.625 ulp(1) under AUTO); BOBYQA/COBYLA's
 * rescaling (u/s)*s is at most 1 ulp(u0) <= 1 ulp(box). 4 keeps one ulp
 * of margin over the worst measured case (3 ulp, once in 1e7) for other
 * libm implementations.
 *
 * Which points it remaps: every u with 0 < |u - u0| <= the window maps to
 * x0 (when u0 is the nearest anchor). Where
 * x = T^-1(T(lo) + u*(T(hi) - T(lo))) is only accurate to about ulp(box)
 * anyway, that changes nothing the map could resolve. It does replace a
 * resolvable move where the plain map is finer than the box: near t = 0
 * when T(lo) = 0 (a LIN box [0, W] near its lower bound keeps the relative
 * precision of u), and under OPTIM_SCALE NONE on every LIN dimension (the
 * identity map).
 *
 * Why that is harmless: only a point within the window of a seed -- an
 * optimiser's rounded start, or a step smaller than the window -- is
 * affected, and no optimiser takes such a step on purpose. Derivative-free
 * steps and STEP fractions are orders of magnitude larger. The finite-
 * difference kernels step by fd_step*max(|u|, 1) (in-house) or
 * max(sqrt(eps), fd_step*|u|) with sqrt(eps) = 2^-26 (Ceres); with the
 * default fd_step 1e-6 (NOISE_AUTO never goes below it) both steps exceed
 * the window (8.9e-16 under AUTO, at most the cap under NONE unless the
 * seed is large, 2 ulp(u0) for a large one), so no difference is ever taken
 * between two points that both map to the seed. Only a PARAM fd_step set
 * by hand below window/max(|u|, 1) -- under NONE on a LIN box of 2^24 or
 * more with fd_kernel inhouse, below ~7.5e-9 -- can put a step inside the
 * window, and that gradient component is then 0. A stochastic proposal
 * that small (annealers, ASA; the annealers' resolvable-step floor
 * max(ulp(u), eps*width) is 2.2e-16 under AUTO) made from a seed evaluates
 * the seed again: a wasted proposal at a converged scale, not a frozen
 * dimension. A seed NLopt deliberately moves (PRAXIS clamps a start ON a
 * bound 1e-12 inside; BOBYQA shifts a start closer than its initial step
 * to a bound) stays outside the window and maps as any other point. */
constexpr int kSeedAnchorUlps = 4;

/* Cap of the box term: half the Ceres finite-difference kernel's smallest
 * step sqrt(eps) = 2^-26. It binds only under OPTIM_SCALE NONE on a LIN
 * dimension with max(|lo|, |hi|) >= 2^24 (~1.7e7), where 4 ulp(box) would
 * reach that step: the Ceres kernel (the default under NONE) would then
 * difference two points that both map to a small seed, a zero gradient
 * component that keeps the dimension at its seed for the whole local run.
 * With the cap, PRAXIS's start rounding (up to 3 ulp(box): it scales with
 * the box's midpoint and half-width) can exceed the window of a seed of
 * smaller magnitude than the box on such a dimension -- not only of a
 * small one -- and its first evaluation then lands a few ulp(box) off the
 * seed. */
constexpr double kSeedAnchorMaxTol = 0x1p-27;

/* Floor of an anchor's window, in ulp of its own u0: BOBYQA/COBYLA's start
 * rescaling moves u0 by up to 1 ulp(u0), which exceeds the cap once
 * |u0| >= 2^26 (~6.7e7; only under NONE can u0 be that large), so such a
 * seed is still anchored. The floor widens a window from |u0| >= 2^25
 * on; both finite-difference kernels step by at least fd_step*|u0| there,
 * above 2 ulp(u0) for any fd_step > 4.4e-16. */
constexpr int kSeedAnchorSeedUlps = 2;

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

    /* scalar maps (to_real honours the seed anchors, if any) */
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

    /* Same shapes/labels, new real boxes (version + 1). BOUND_ADAPT.
     * Seed anchors are dropped: they belong to the old boxes. */
    SearchSpace with_boxes(const std::vector<std::pair<double,double>>& boxes) const;

    /* Copy of this space anchored on `seeds` (REAL points; see the file
     * comment). Every in-box coordinate of every seed gets an anchor --
     * also one whose plain round trip is exact, because an optimiser that
     * perturbs its start by rounding (BOBYQA, COBYLA, PRAXIS) evaluates a
     * u next to u0, which the plain map would move off x0. Per dimension,
     * the first seed that reaches a given u owns it: a later seed with a
     * different value but the same u (the two are closer than the box
     * resolves) is evaluated at the owner's value. to_real() returns an
     * anchor's x0 for its exact u0 and, failing every exact match, for u
     * whose nearest anchor u0 is within seed_anchor_tolerance(i, u0) of
     * it. Seeds of the wrong size
     * and non-finite or out-of-box values are ignored. The anchors replace
     * any this space had; they do not enter fingerprint() or version().
     * to_real() pays one test per dimension plus, on an anchored
     * dimension with A anchors (A is 1 per seed of the run), one
     * distance compare per anchor for A <= 24, else a binary search and
     * two compares; no allocation. */
    SearchSpace with_seed_anchors(const std::vector<std::vector<double>>& seeds) const;
    std::size_t anchor_count() const noexcept { return anc_u_.size(); }
    /* min(kSeedAnchorUlps * ulp(max(|internal_lo(i)|, |internal_hi(i)|)),
     *     max(kSeedAnchorMaxTol, kSeedAnchorSeedUlps * ulp(u0))): the
     * half-width, in u (absolute), of the window of an anchor at u0 on
     * dimension i. */
    double seed_anchor_tolerance(std::size_t i, double u0) const;

private:
    SearchSpace() = default;
    std::vector<DimSpec>      dims_;
    std::vector<std::string>  labels_;
    std::vector<double>       t_lo_, t_hi_;   // T_i(lo_i), T_i(hi_i)
    ScalePolicy               policy_;
    std::uint32_t             version_ = 1;
    /* Seed anchors, CSR by dimension: dim i owns [anc_off_[i],
     * anc_off_[i+1]) of anc_u_/anc_x_/anc_tol_ (sorted by u); anc_tol_[k]
     * is anchor k's window half-width (seed_anchor_tolerance). All four
     * are empty when the space has no anchor at all. */
    std::vector<std::size_t>  anc_off_;
    std::vector<double>       anc_u_, anc_x_, anc_tol_;
};

using SearchSpacePtr = std::shared_ptr<const SearchSpace>;

}  // namespace tadah::hpo

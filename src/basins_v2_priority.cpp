#include <tadah/hpo/basins_v2_priority.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tadah {
namespace hpo {

double state_bonus(LifecycleState s)
{
    switch (s) {
        case LifecycleState::PROMISING:   return  1.0;
        case LifecycleState::ACTIVE:      return  0.0;
        case LifecycleState::INACTIVE:    return -0.5;
        case LifecycleState::SUSPECT_DUP: return -2.0;
        case LifecycleState::MERGED:
        case LifecycleState::KILLED:      return -std::numeric_limits<double>::infinity();
    }
    return 0.0;
}

double min_max_normalise(double x, double lo, double hi)
{
    const double range = hi - lo;
    if (range <= 0.0) return 0.0;     // degenerate -- all equal
    return (x - lo) / range;
}

namespace {

struct Range {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    void observe(double x) {
        if (!std::isfinite(x)) return;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    bool empty() const { return !std::isfinite(lo) || !std::isfinite(hi); }
};

double safe_log10(double x) {
    if (!(x > 0.0)) return -1e300;
    return std::log10(x);
}

}   // namespace

std::vector<double>
compute_priorities(const std::vector<BasinV2>&  basins,
                   const std::vector<double>&   min_dist_to_other_active,
                   const PriorityContext&       ctx,
                   const PriorityWeights&       w)
{
    const std::size_t N = basins.size();
    std::vector<double> scores(N, -std::numeric_limits<double>::infinity());
    if (N == 0) return scores;

    /* Compute per-component ranges over ACTIVE-family basins for min-max
     * normalisation. Archived basins are excluded entirely. */
    Range r_imp, r_neg, r_dist, r_tol, r_age;
    for (std::size_t i = 0; i < N; ++i) {
        if (is_archive_state(basins[i].state)) continue;
        const auto& b = basins[i];
        r_imp .observe(b.improvement_ema);
        r_neg .observe(static_cast<double>(ctx.current_iteration - b.last_iteration));
        if (i < min_dist_to_other_active.size())
            r_dist.observe(min_dist_to_other_active[i]);
        r_tol .observe(safe_log10(b.current_tol_cursor) - safe_log10(ctx.target_tol));
        r_age .observe(static_cast<double>(ctx.current_iteration - b.birth_iteration));
    }

    for (std::size_t i = 0; i < N; ++i) {
        if (is_archive_state(basins[i].state)) {
            scores[i] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const auto& b = basins[i];
        const double n_imp  = r_imp.empty()  ? 0.0
            : min_max_normalise(b.improvement_ema, r_imp.lo, r_imp.hi);
        const double n_neg  = r_neg.empty()  ? 0.0
            : min_max_normalise(static_cast<double>(ctx.current_iteration - b.last_iteration),
                                r_neg.lo, r_neg.hi);
        const double dist_i = (i < min_dist_to_other_active.size())
            ? min_dist_to_other_active[i] : 0.0;
        const double n_dist = r_dist.empty() ? 0.0
            : min_max_normalise(dist_i, r_dist.lo, r_dist.hi);
        const double tol_gap = safe_log10(b.current_tol_cursor) - safe_log10(ctx.target_tol);
        const double n_tol  = r_tol.empty()  ? 0.0
            : min_max_normalise(tol_gap, r_tol.lo, r_tol.hi);
        const double n_age  = r_age.empty()  ? 0.0
            : min_max_normalise(static_cast<double>(ctx.current_iteration - b.birth_iteration),
                                r_age.lo, r_age.hi);

        const double sb = state_bonus(b.state);

        scores[i] = w.improve  * n_imp
                  + w.neglect  * n_neg
                  + w.distance * n_dist
                  - w.tol_gap  * n_tol
                  + w.state    * sb
                  - w.age      * n_age;
    }
    return scores;
}

int argmax_priority(const std::vector<BasinV2>& basins,
                    const std::vector<double>&  scores)
{
    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < basins.size() && i < scores.size(); ++i) {
        if (is_archive_state(basins[i].state)) continue;
        if (scores[i] > best_score) {
            best_score = scores[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

std::size_t
adaptive_refine_threshold(double      tol,
                          std::size_t dim,
                          std::size_t min_evals,
                          std::size_t max_evals,
                          double      plateau_eps_factor)
{
    if (plateau_eps_factor <= 0.0) plateau_eps_factor = 0.5;
    const std::size_t dim_floor =
        100u * std::max<std::size_t>(dim, 1u);
    const double tol_term =
        50.0 * std::abs(std::log10(std::max(tol, 1.0e-300)))
        * plateau_eps_factor;
    std::size_t threshold = std::max<std::size_t>({
        min_evals,
        dim_floor,
        static_cast<std::size_t>(std::max(tol_term, 1.0))
    });
    if (max_evals > 0)
        threshold = std::min(threshold, max_evals);
    return threshold;
}

}   // namespace hpo
}   // namespace tadah

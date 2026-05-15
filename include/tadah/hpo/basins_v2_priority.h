#ifndef TADAH_HPO_BASINS_V2_PRIORITY_H
#define TADAH_HPO_BASINS_V2_PRIORITY_H

/*
 *  basins_v2_priority -- pure-function priority scoring for BASINSv2.
 *
 *  Decoupled from BasinPoolV2 / BasinsV2Policy so the formula is unit-
 *  testable in isolation. The policy supplies the inputs (basins +
 *  precomputed min-distance vector + iteration + target tol) and gets
 *  back one score per basin.
 *
 *  Formula (single source of truth):
 *
 *    priority(b) =
 *        W_improve  * norm(b.improvement_ema)
 *      + W_neglect  * norm(iter - b.last_iteration)
 *      + W_distance * norm(min_dist_to_other_active(b))
 *      - W_tol_gap  * norm(log10(b.current_tol_cursor) - log10(target_tol))
 *      + W_state    * state_bonus(b.state)
 *      - W_age      * norm(iter - b.birth_iteration)
 *
 *  where norm(x) is min-max normalisation across the active pool, and
 *  state_bonus is { PROMISING:+1.0, ACTIVE:0.0, INACTIVE:-0.5,
 *  SUSPECT_DUP:-2.0, MERGED/KILLED:-inf }.
 *
 *  Starvation backstop (caller's responsibility, not part of this
 *  module): if max_neglect > starvation_cap, force-pick the most-
 *  neglected basin once, regardless of its priority score.
 */

#include <tadah/hpo/basin_pool_v2.h>

#include <cstddef>
#include <vector>

namespace tadah {
namespace hpo {

struct PriorityWeights {
    double improve  = 2.0;
    double neglect  = 0.5;
    double distance = 0.3;
    double tol_gap  = 0.5;
    double state    = 1.0;
    double age      = 0.05;
};

struct PriorityContext {
    int    current_iteration = 0;
    double target_tol        = 1.0e-9;
};

/* Pure function. Returns a vector of scores aligned with `basins`.
 *
 * `min_dist_to_other_active[i]` MUST be precomputed by the caller and
 * be the minimum normalised L2 distance from basin i to any other
 * ACTIVE-family basin (or +inf if i has no peers). The caller can use
 * BasinPoolV2::relative_l2 to build this.
 *
 * Archived basins (MERGED/KILLED) get a score of -inf and are never
 * picked by argmax.
 *
 * Score for basins not in any active state still reflects the formula's
 * non-state terms -- useful for observability but never used for
 * scheduling. */
std::vector<double>
compute_priorities(const std::vector<BasinV2>&  basins,
                   const std::vector<double>&   min_dist_to_other_active,
                   const PriorityContext&       ctx,
                   const PriorityWeights&       w);

/* Argmax helper. Returns the index of the highest-scoring ACTIVE-family
 * basin, or -1 if none exist. */
int argmax_priority(const std::vector<BasinV2>& basins,
                    const std::vector<double>&  scores);

/* Adaptive REFINE plateau threshold. Returns the value to pass to
 * HPO_Host::set_stagnation_threshold() -- the optimiser bails when no
 * new global best is found within this many evals. The formula is
 * derived from the basin's current tol cursor + dim + user knobs:
 *
 *   threshold = clamp( max(min_evals, 100*dim, 50*|log10(tol)|*plateau_eps_factor),
 *                      0,
 *                      max_evals )
 *
 * - min_evals    : refine_budget_min_evals (lower bound)
 * - 100 * dim    : protects slow exponential descent in higher dim
 * - 50 * |log10(tol)| * plateau_eps_factor : tighter tol -> patience
 *                                            grows; user can soften
 *                                            via plateau_eps_factor
 * - max_evals    : refine_budget_max_evals (hard cap)
 *
 * Pure free function -- exposed here so unit tests can exercise the
 * formula without instantiating BasinsV2Policy. */
std::size_t
adaptive_refine_threshold(double      tol,
                          std::size_t dim,
                          std::size_t min_evals,
                          std::size_t max_evals,
                          double      plateau_eps_factor);

/* Min/max normalisation helper exported for diagnostics tests. */
double min_max_normalise(double x, double lo, double hi);

/* state_bonus exported for tests. */
double state_bonus(LifecycleState s);

}   // namespace hpo
}   // namespace tadah

#endif

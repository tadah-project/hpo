#ifndef TADAH_HPO_BASINS_V2_PATH_H
#define TADAH_HPO_BASINS_V2_PATH_H

/*
 *  basins_v2_path -- coarse-trajectory helpers for BASINSv2 (Phase 4).
 *
 *  Pure functions over PathPoint vectors. The data layout lives in
 *  basin_pool_v2.h (PathPoint, BasinV2::coarse_path) so the pool can
 *  serialise paths without a circular include; the *operations* live
 *  here so they can be unit-tested in isolation.
 *
 *  Design notes:
 *    - prune_path keeps endpoints (first + last) and drops the
 *      intermediate snapshot whose loss is closest to its neighbours
 *      -- i.e. the least-informative point. Repeated until under cap.
 *      This preserves diversity at both ends of the trajectory.
 *    - path_overlap measures: for each point on path B, is there a
 *      point on path A within `dist_thresh` (using a caller-provided
 *      distance functor -- typically BasinPoolV2::relative_l2)?
 *      Returns the overlap fraction in [0, 1]. Used by Phase 2's
 *      merge predicate when paths are available (Phase 4 enhancement).
 *    - should_snapshot decides whether the policy should record a new
 *      PathPoint, based on per-basin loss/distance thresholds.
 */

#include <tadah/hpo/basin_pool_v2.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace tadah {
namespace hpo {

/* Prune `path` down to at most `cap` points by repeatedly dropping the
 * intermediate snapshot whose loss is closest to its two neighbours
 * (mean-loss thinning). Endpoints are preserved unless the path itself
 * is shorter than 2. No-op if path.size() <= cap. */
void prune_path(std::vector<PathPoint>& path, std::size_t cap);

/* Decide whether to record a new snapshot. Returns true iff:
 *   - the path is empty (always record the first point), OR
 *   - relative loss improvement vs the last snapshot >= loss_eps, OR
 *   - normalised parameter distance vs the last snapshot >= dist_eps
 *
 * `dist_fn` is the caller's normalised-L2 functor (typically a
 * lambda capturing BasinPoolV2::relative_l2). */
bool should_snapshot(const std::vector<PathPoint>& path,
                     const std::vector<double>&    new_params,
                     double                        new_loss,
                     double                        loss_eps,
                     double                        dist_eps,
                     const std::function<double(const std::vector<double>&,
                                                const std::vector<double>&)>& dist_fn);

/* Fraction of path B's points that have at least one point on path A
 * within dist_thresh. Returns 0.0 if either path is empty. Symmetric
 * version is max(overlap(A,B), overlap(B,A)) -- left to the caller. */
double path_overlap(const std::vector<PathPoint>& a,
                    const std::vector<PathPoint>& b,
                    double dist_thresh,
                    const std::function<double(const std::vector<double>&,
                                               const std::vector<double>&)>& dist_fn);

}   // namespace hpo
}   // namespace tadah

#endif

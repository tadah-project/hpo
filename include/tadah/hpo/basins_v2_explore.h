#ifndef TADAH_HPO_BASINS_V2_EXPLORE_H
#define TADAH_HPO_BASINS_V2_EXPLORE_H

/*
 *  basins_v2_explore — Sobol generator + path-aware survivor filter
 *  for BASINSv2's EXPLORE pipeline.
 *
 *  Pure helpers, no policy / host / runner dependencies — keeps the
 *  pipeline unit-testable in isolation.
 *
 *  Pipeline (called by BasinsV2Policy::do_explore):
 *    1. sobol_unit_points(K, dim, seed_skip)   → K points in [0,1]^dim
 *    2. caller maps to real space, evaluates them via host()
 *    3. filter_far_from_pool(candidates, pool, explore_dist_floor)
 *       → indices of survivors not within explore_dist_floor of any
 *         known basin (or its coarse path, when present in Phase 4)
 *    4. caller picks top-M by loss and runs brief local refinement
 *
 *  The Sobol generator wraps boost::random::sobol with the same
 *  semantics as init_strategies.cpp's SobolInitStrategy: deterministic
 *  sequence, optional seed_skip discards the first N points so
 *  different runs/iterations sample different sub-sequences.
 */

#include <cstdint>
#include <cstddef>
#include <vector>

namespace tadah {
namespace hpo {

class BasinPoolV2;

/* Generate K Sobol points in [0,1]^dim. Returns a flat vector of K
 * sub-vectors. seed_skip discards the first seed_skip*dim Sobol values
 * so successive EXPLORE invocations sample different subsequences. */
std::vector<std::vector<double>>
sobol_unit_points(std::size_t K, std::size_t dim, std::uint64_t seed_skip);

/* Filter step: returns the indices of `candidates` that are NOT within
 * `explore_dist_floor` (relative L2, normalised by pool's bounding box)
 * of any active-family basin in the pool. Indices are returned in
 * ascending order; archived basins (MERGED/KILLED) are still consulted
 * by design (they answer "have we explored this region?"). */
std::vector<std::size_t>
filter_far_from_pool(const std::vector<std::vector<double>>& candidates,
                     const BasinPoolV2& pool,
                     double explore_dist_floor);

/* Pick the top-M (lowest-loss) indices from `losses`. Returns indices
 * into the input vector, sorted ascending by loss. M is capped at the
 * input size. */
std::vector<std::size_t>
top_m_by_loss(const std::vector<double>& losses, std::size_t M);

/* Mean pairwise normalised L2 distance among candidate points. Used
 * by tests to assert Sobol diversity vs random. */
double mean_pairwise_distance(const std::vector<std::vector<double>>& points);

}   // namespace hpo
}   // namespace tadah

#endif

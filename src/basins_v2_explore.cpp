#include <tadah/hpo/basins_v2_explore.h>
#include <tadah/hpo/basin_pool_v2.h>

#include <boost/random/sobol.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tadah {
namespace hpo {

std::vector<std::vector<double>>
sobol_unit_points(std::size_t K, std::size_t dim, std::uint64_t seed_skip)
{
    std::vector<std::vector<double>> out;
    out.reserve(K);
    if (dim == 0 || K == 0) {
        out.assign(K, std::vector<double>{});
        return out;
    }
    boost::random::sobol engine(static_cast<unsigned>(dim));
    if (seed_skip > 0) {
        /* Discard seed_skip*dim values to land on a different
         * subsequence -- same trick init_strategies.cpp uses. */
        engine.discard(static_cast<std::size_t>(seed_skip * dim));
    }
    const auto sobol_max = static_cast<double>(engine.max());
    const auto sobol_min = static_cast<double>(engine.min());
    const double range = sobol_max - sobol_min + 1.0;

    for (std::size_t i = 0; i < K; ++i) {
        std::vector<double> u(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            const auto raw = static_cast<double>(engine());
            u[d] = (raw - sobol_min) / range;
        }
        out.push_back(std::move(u));
    }
    return out;
}

std::vector<std::size_t>
filter_far_from_pool(const std::vector<std::vector<double>>& candidates,
                     const BasinPoolV2& pool,
                     double explore_dist_floor)
{
    std::vector<std::size_t> survivors;
    survivors.reserve(candidates.size());

    /* Empty pool -> every candidate survives (nothing to compare against). */
    if (pool.empty() || explore_dist_floor <= 0.0) {
        survivors.resize(candidates.size());
        std::iota(survivors.begin(), survivors.end(), std::size_t{0});
        return survivors;
    }

    const auto& basins = pool.view();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        bool too_close = false;
        for (const auto& b : basins) {
            /* Both active and archived basins are consulted -- archived
             * basins answer "have we explored here before?" (the design
             * doc's path/region-coverage query). */
            const double d = pool.relative_l2(candidates[i], b.params);
            if (d < explore_dist_floor) { too_close = true; break; }
            /* Phase 4 hook: also check distance to any path point. */
            for (const auto& pp : b.coarse_path) {
                const double dp = pool.relative_l2(candidates[i], pp.params);
                if (dp < explore_dist_floor) { too_close = true; break; }
            }
            if (too_close) break;
        }
        if (!too_close) survivors.push_back(i);
    }
    return survivors;
}

std::vector<std::size_t>
top_m_by_loss(const std::vector<double>& losses, std::size_t M)
{
    std::vector<std::size_t> idx(losses.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    if (M >= losses.size()) {
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b){ return losses[a] < losses[b]; });
        return idx;
    }
    std::partial_sort(idx.begin(), idx.begin() + M, idx.end(),
                      [&](std::size_t a, std::size_t b){ return losses[a] < losses[b]; });
    idx.resize(M);
    return idx;
}

double mean_pairwise_distance(const std::vector<std::vector<double>>& points)
{
    if (points.size() < 2) return 0.0;
    double sum = 0.0;
    std::size_t pairs = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            if (points[i].size() != points[j].size()) continue;
            double sq = 0.0;
            for (std::size_t d = 0; d < points[i].size(); ++d) {
                const double diff = points[i][d] - points[j][d];
                sq += diff * diff;
            }
            sum += std::sqrt(sq);
            ++pairs;
        }
    }
    return pairs > 0 ? sum / static_cast<double>(pairs) : 0.0;
}

}   // namespace hpo
}   // namespace tadah

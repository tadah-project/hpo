#include <tadah/hpo/basins_v2_path.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tadah {
namespace hpo {

void prune_path(std::vector<PathPoint>& path, std::size_t cap)
{
    if (path.size() <= cap || cap == 0) return;
    if (cap == 1) {
        /* Edge case: keep only the lowest-loss point. */
        const auto best = std::min_element(
            path.begin(), path.end(),
            [](const PathPoint& a, const PathPoint& b){ return a.loss < b.loss; });
        PathPoint keep = *best;
        path.clear();
        path.push_back(std::move(keep));
        return;
    }

    /* Repeatedly drop the intermediate point whose loss is closest to
     * the mean of its two neighbours' losses. Tie-broken by smaller
     * eval_index (older point dropped first). Endpoints (first + last)
     * are sacrosanct. */
    while (path.size() > cap) {
        std::size_t worst = 1;   // any interior position
        double worst_score = std::numeric_limits<double>::infinity();
        for (std::size_t i = 1; i + 1 < path.size(); ++i) {
            const double mid = 0.5 * (path[i-1].loss + path[i+1].loss);
            const double score = std::abs(path[i].loss - mid);
            if (score < worst_score ||
                (score == worst_score && path[i].eval_index < path[worst].eval_index)) {
                worst_score = score;
                worst = i;
            }
        }
        path.erase(path.begin() + worst);
    }
}

bool should_snapshot(const std::vector<PathPoint>& path,
                     const std::vector<double>&    new_params,
                     double                        new_loss,
                     double                        loss_eps,
                     double                        dist_eps,
                     const std::function<double(const std::vector<double>&,
                                                const std::vector<double>&)>& dist_fn)
{
    if (path.empty()) return true;
    const PathPoint& last = path.back();
    if (loss_eps > 0.0 && last.loss != 0.0) {
        const double rel_imp = (last.loss - new_loss) / std::abs(last.loss);
        if (rel_imp >= loss_eps) return true;
    }
    if (dist_eps > 0.0 && dist_fn) {
        const double d = dist_fn(last.params, new_params);
        if (d >= dist_eps) return true;
    }
    return false;
}

double path_overlap(const std::vector<PathPoint>& a,
                    const std::vector<PathPoint>& b,
                    double dist_thresh,
                    const std::function<double(const std::vector<double>&,
                                               const std::vector<double>&)>& dist_fn)
{
    if (a.empty() || b.empty() || !dist_fn) return 0.0;
    std::size_t hits = 0;
    for (const auto& bp : b) {
        for (const auto& ap : a) {
            if (dist_fn(ap.params, bp.params) <= dist_thresh) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(b.size());
}

}   // namespace hpo
}   // namespace tadah

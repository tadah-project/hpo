#include "catch2/catch.hpp"

#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/basins_v2_path.h>

#include <cmath>
#include <vector>

using tadah::hpo::PathPoint;
using tadah::hpo::path_overlap;
using tadah::hpo::prune_path;
using tadah::hpo::should_snapshot;

namespace {
PathPoint mk(std::vector<double> p, double loss, std::size_t e = 0)
{
    PathPoint pp;
    pp.params = std::move(p);
    pp.loss = loss;
    pp.eval_index = e;
    return pp;
}
}   // namespace

TEST_CASE("path: prune_path keeps endpoints",
          "[basins_v2][path]")
{
    std::vector<PathPoint> path = {
        mk({0.0}, 1.0, 1),
        mk({0.5}, 0.49, 2),   // close to mean(1.0, 0.5) = 0.75 → not closest
        mk({1.0}, 0.5,  3),   // closer to mean(0.49, 0.4) = 0.445 → drop
        mk({1.5}, 0.4,  4),
        mk({2.0}, 0.2,  5),
    };
    prune_path(path, 3);
    REQUIRE(path.size() == 3u);
    REQUIRE(path.front().eval_index == 1u);
    REQUIRE(path.back().eval_index  == 5u);
}

TEST_CASE("path: prune_path is no-op below cap",
          "[basins_v2][path]")
{
    std::vector<PathPoint> path = { mk({0.0}, 1.0), mk({1.0}, 0.5) };
    prune_path(path, 5);
    REQUIRE(path.size() == 2u);
}

TEST_CASE("path: prune_path cap=1 keeps lowest-loss point",
          "[basins_v2][path]")
{
    std::vector<PathPoint> path = {
        mk({0.0}, 1.0, 1), mk({1.0}, 0.5, 2), mk({2.0}, 0.3, 3)
    };
    prune_path(path, 1);
    REQUIRE(path.size() == 1u);
    REQUIRE(path[0].loss == Approx(0.3));
}

TEST_CASE("path: should_snapshot accepts empty path",
          "[basins_v2][path]")
{
    std::vector<PathPoint> empty;
    REQUIRE(should_snapshot(empty, {1.0}, 0.5, 0.01, 0.01,
                            [](const std::vector<double>& a,
                               const std::vector<double>& b) {
                                double s = 0;
                                for (std::size_t i = 0; i < a.size(); ++i) {
                                    const double d = a[i] - b[i];
                                    s += d * d;
                                }
                                return std::sqrt(s);
                            }));
}

TEST_CASE("path: should_snapshot triggers on loss improvement",
          "[basins_v2][path]")
{
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i]; s += d * d;
        }
        return std::sqrt(s);
    };
    std::vector<PathPoint> path = { mk({1.0}, 1.0) };
    /* 5% improvement, threshold 1% → snapshot. */
    REQUIRE(should_snapshot(path, {1.0}, 0.95, 0.01, 0.10, dist));
    /* 0.5% improvement, threshold 1% AND no movement → no snapshot. */
    REQUIRE_FALSE(should_snapshot(path, {1.0}, 0.995, 0.01, 0.10, dist));
}

TEST_CASE("path: should_snapshot triggers on parameter movement",
          "[basins_v2][path]")
{
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i]; s += d * d;
        }
        return std::sqrt(s);
    };
    std::vector<PathPoint> path = { mk({0.0}, 1.0) };
    /* No loss change but moved 0.5 → snapshot. */
    REQUIRE(should_snapshot(path, {0.5}, 1.0, 0.01, 0.1, dist));
    /* No loss change, moved 0.05 → below dist threshold → no snapshot. */
    REQUIRE_FALSE(should_snapshot(path, {0.05}, 1.0, 0.01, 0.1, dist));
}

TEST_CASE("path: path_overlap returns fraction of B in A's neighbourhood",
          "[basins_v2][path][merge]")
{
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i]; s += d * d;
        }
        return std::sqrt(s);
    };
    /* Path A = {(0), (1), (2)}; Path B = {(0.05), (1.5), (5)} */
    std::vector<PathPoint> a = { mk({0.0}, 1), mk({1.0}, 0.5), mk({2.0}, 0.3) };
    std::vector<PathPoint> b = { mk({0.05}, 1), mk({1.5}, 0.4), mk({5.0}, 1) };
    /* threshold 0.1 → only first two of B match (0 and 1.5 within 0.1
     * of A points 0 and ... wait, 1.5 is 0.5 from both A's 1 and 2.)
     * Adjust threshold to 0.55 so first and second match. */
    const double frac = path_overlap(a, b, 0.55, dist);
    REQUIRE(frac == Approx(2.0 / 3.0));
}

TEST_CASE("path: path_overlap is 0 for empty paths",
          "[basins_v2][path]")
{
    auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
        return std::abs(a[0] - b[0]);
    };
    std::vector<PathPoint> empty;
    std::vector<PathPoint> nonempty = { mk({1.0}, 0.5) };
    REQUIRE(path_overlap(empty, nonempty, 1.0, dist) == 0.0);
    REQUIRE(path_overlap(nonempty, empty, 1.0, dist) == 0.0);
}

TEST_CASE("path: append_path_point + prune via pool API",
          "[basins_v2][path]")
{
    using tadah::hpo::BasinPoolV2;
    BasinPoolV2 pool(0.0, 4);
    const int id = pool.add_or_merge({1.0}, 1.0, 1);
    for (int k = 0; k < 10; ++k) {
        PathPoint pp;
        pp.params = {1.0 + 0.1 * k};
        pp.loss = 1.0 - 0.05 * k;
        pp.eval_index = static_cast<std::size_t>(k);
        pool.append_path_point(id, std::move(pp), /*cap=*/4);
    }
    REQUIRE(pool.by_id(id)->coarse_path.size() == 4u);
    /* Endpoints survived. */
    REQUIRE(pool.by_id(id)->coarse_path.front().eval_index == 0u);
    REQUIRE(pool.by_id(id)->coarse_path.back().eval_index == 9u);
}

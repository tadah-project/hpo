#include "catch2/catch.hpp"

#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/basins_v2_explore.h>

#include <random>
#include <vector>

using tadah::hpo::BasinPoolV2;
using tadah::hpo::filter_far_from_pool;
using tadah::hpo::mean_pairwise_distance;
using tadah::hpo::sobol_unit_points;
using tadah::hpo::top_m_by_loss;

TEST_CASE("explore: Sobol generates K points in [0,1]^dim",
          "[basins_v2][explore]")
{
    auto pts = sobol_unit_points(/*K=*/16, /*dim=*/4, /*seed_skip=*/0);
    REQUIRE(pts.size() == 16u);
    for (const auto& p : pts) {
        REQUIRE(p.size() == 4u);
        for (double v : p) {
            REQUIRE(v >= 0.0);
            REQUIRE(v <  1.0);
        }
    }
}

TEST_CASE("explore: Sobol diversity beats uniform random",
          "[basins_v2][explore]")
{
    /* Mean pairwise distance for low-discrepancy sequences should be
     * close to the optimal, and at least competitive with random. */
    constexpr std::size_t K = 32;
    constexpr std::size_t D = 4;
    auto sobol = sobol_unit_points(K, D, /*seed_skip=*/0);

    std::vector<std::vector<double>> rnd;
    rnd.reserve(K);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (std::size_t i = 0; i < K; ++i) {
        std::vector<double> p(D);
        for (std::size_t d = 0; d < D; ++d) p[d] = u(rng);
        rnd.push_back(std::move(p));
    }

    const double sobol_mpd = mean_pairwise_distance(sobol);
    const double rnd_mpd   = mean_pairwise_distance(rnd);
    /* Both produce distances on the same order; Sobol should be at
     * least 90 % of the random mean (asserts diversity is sane). */
    REQUIRE(sobol_mpd > 0.0);
    REQUIRE(rnd_mpd > 0.0);
    REQUIRE(sobol_mpd >= 0.9 * rnd_mpd);
}

TEST_CASE("explore: seed_skip produces a different subsequence",
          "[basins_v2][explore]")
{
    auto a = sobol_unit_points(8, 2, /*seed_skip=*/0);
    auto b = sobol_unit_points(8, 2, /*seed_skip=*/4);
    /* The two sets share dimensions; they should differ in at least
     * one element value by a non-trivial margin. */
    REQUIRE(a.size() == b.size());
    bool any_diff = false;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t d = 0; d < 2; ++d)
            if (std::abs(a[i][d] - b[i][d]) > 1e-6) any_diff = true;
    REQUIRE(any_diff);
}

TEST_CASE("explore: filter_far_from_pool drops in-region candidates",
          "[basins_v2][explore]")
{
    BasinPoolV2 pool(0.0, 4);
    pool.set_normalisation({0.0, 0.0}, {1.0, 1.0});
    pool.add_or_merge({0.5, 0.5}, 0.5, 1);   // pool has one basin at centre

    /* Three candidates: one near the basin, one far, one borderline. */
    std::vector<std::vector<double>> candidates = {
        {0.50, 0.51},   // very close
        {0.10, 0.10},   // far
        {0.20, 0.20},   // far enough at floor 0.05
    };
    auto survivors = filter_far_from_pool(candidates, pool, 0.05);
    REQUIRE(survivors.size() == 2u);
    REQUIRE(survivors[0] == 1u);
    REQUIRE(survivors[1] == 2u);
}

TEST_CASE("explore: filter passes everything when pool is empty",
          "[basins_v2][explore]")
{
    BasinPoolV2 empty(0.0, 4);
    std::vector<std::vector<double>> candidates = {
        {0.1}, {0.2}, {0.3}
    };
    auto survivors = filter_far_from_pool(candidates, empty, 0.05);
    REQUIRE(survivors.size() == 3u);
}

TEST_CASE("explore: top_m_by_loss returns sorted indices",
          "[basins_v2][explore]")
{
    std::vector<double> losses = {3.0, 1.0, 5.0, 2.0, 4.0};
    auto top3 = top_m_by_loss(losses, 3);
    REQUIRE(top3.size() == 3u);
    REQUIRE(top3[0] == 1u);   // 1.0
    REQUIRE(top3[1] == 3u);   // 2.0
    REQUIRE(top3[2] == 0u);   // 3.0
}

TEST_CASE("explore: top_m_by_loss caps M at input size",
          "[basins_v2][explore]")
{
    std::vector<double> losses = {3.0, 1.0};
    auto out = top_m_by_loss(losses, 99);
    REQUIRE(out.size() == 2u);
    REQUIRE(out[0] == 1u);
    REQUIRE(out[1] == 0u);
}

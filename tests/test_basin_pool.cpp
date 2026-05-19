#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/basin_pool.h>

#include <filesystem>
#include <vector>

using tadah::core::Checkpoint;
using tadah::hpo::Basin;
using tadah::hpo::BasinPool;

TEST_CASE("BasinPool: insert, sort by loss, capacity", "[basin_pool]")
{
    BasinPool p(/*dedup=*/0.0, /*max=*/3);
    REQUIRE(p.empty());

    REQUIRE(p.add_or_merge({1.0}, 0.5, 1) == 0);
    REQUIRE(p.add_or_merge({2.0}, 0.3, 1) == 1);
    REQUIRE(p.add_or_merge({3.0}, 0.7, 1) == 2);

    auto h = p.heads();
    REQUIRE(h.size() == 3);
    REQUIRE(h[0].loss == Approx(0.3));
    REQUIRE(h[1].loss == Approx(0.5));
    REQUIRE(h[2].loss == Approx(0.7));
}

TEST_CASE("BasinPool: capacity eviction prefers worst-loss basin", "[basin_pool]")
{
    BasinPool p(0.0, 2);
    p.add_or_merge({1.0}, 0.5, 1);
    p.add_or_merge({2.0}, 0.9, 1);

    /* Insert 0.7 — must evict 0.9 */
    int i = p.add_or_merge({3.0}, 0.7, 2);
    REQUIRE(i >= 0);
    auto h = p.heads();
    REQUIRE(h.size() == 2);
    REQUIRE(h[0].loss == Approx(0.5));
    REQUIRE(h[1].loss == Approx(0.7));

    /* Insert 1.0 — worse than every resident, refused. */
    i = p.add_or_merge({4.0}, 1.0, 3);
    REQUIRE(i == -1);
    REQUIRE(p.size() == 2);
}

TEST_CASE("BasinPool: dedup merges within distance, keeps better loss",
          "[basin_pool]")
{
    BasinPool p(/*dedup=*/0.05, /*max=*/8);
    p.set_normalisation({0.0, 0.0}, {10.0, 10.0});

    /* Two close points: relative L2 = sqrt((0.5/10)^2 + 0) = 0.05 */
    REQUIRE(p.add_or_merge({1.0, 1.0}, 0.5, 1) == 0);
    /* Within threshold (0.05) — should merge. */
    int i = p.add_or_merge({1.5, 1.0}, 0.3, 2);
    REQUIRE(i == 0);
    REQUIRE(p.size() == 1);
    /* Better loss wins; params updated to the better point. */
    auto h = p.heads();
    REQUIRE(h[0].loss == Approx(0.3));
    REQUIRE(h[0].params[0] == Approx(1.5));

    /* Far away: 5/10 = 0.5 — not merged. */
    int j = p.add_or_merge({6.0, 1.0}, 0.6, 2);
    REQUIRE(j == 1);
    REQUIRE(p.size() == 2);
}

TEST_CASE("BasinPool: not_exhausted_at filters by mark_exhausted", "[basin_pool]")
{
    BasinPool p(0.0, 8);
    p.add_or_merge({1.0}, 0.5, 1);
    p.add_or_merge({2.0}, 0.6, 1);
    p.add_or_merge({3.0}, 0.7, 1);

    p.mark_exhausted(0, 1e-3);   // basin 0 has already converged to 1e-3
    p.mark_exhausted(1, 1e-5);   // basin 1 has converged tighter, to 1e-5

    /* At query tol=1e-4 (tighter than 1e-3, looser than 1e-5):
     *   basin 0 (1e-3) is *not* tight enough — still due.
     *   basin 1 (1e-5) is already tighter than 1e-4 — not due.
     *   basin 2 was never refined — still due. */
    auto due = p.not_exhausted_at(1e-4);
    REQUIRE(due.size() == 2);
    REQUIRE(std::find(due.begin(), due.end(), 0u) != due.end());
    REQUIRE(std::find(due.begin(), due.end(), 2u) != due.end());

    /* At very loose query tol=1e-2: every previously exhausted basin
     * is already tight enough; only basin 2 (never refined) is due. */
    auto loose = p.not_exhausted_at(1e-2);
    REQUIRE(loose.size() == 1);
    REQUIRE(loose[0] == 2u);
}

TEST_CASE("BasinPool: checkpoint round-trip", "[basin_pool][checkpoint]")
{
    BasinPool p(0.07, 4);
    p.set_normalisation({0.0, -1.0}, {10.0, 1.0});
    p.add_or_merge({1.0,  0.0}, 0.5, 1);
    p.add_or_merge({2.0,  0.5}, 0.3, 1);
    p.mark_exhausted(0, 1e-3);
    p.mark_attempted(1, 1e-4);

    auto path = std::filesystem::temp_directory_path() / "tadah_bp.tadah";
    std::filesystem::remove(path);
    {
        Checkpoint cp(path.string());
        p.save(cp);
        cp.commit();
    }

    BasinPool p2(0.0, 1);
    Checkpoint cp(path.string());
    REQUIRE(cp.load_from_disk());
    p2.load(cp);

    REQUIRE(p2.dedup_dist() == Approx(0.07));
    REQUIRE(p2.pool_max()   == 4u);
    REQUIRE(p2.size()       == 2u);
    auto h = p2.heads();
    REQUIRE(h[0].loss == Approx(0.3));
    REQUIRE(h[0].params.size() == 2);
    REQUIRE(h[0].params[1] == Approx(0.5));
    REQUIRE(h[1].exhausted_at_tol.has_value());
    REQUIRE(h[1].exhausted_at_tol.value() == Approx(1e-3));
    /* basin 1 (loss 0.3) had attempted=1e-4. */
    REQUIRE(h[0].last_tol_attempted.has_value());
    REQUIRE(h[0].last_tol_attempted.value() == Approx(1e-4));

    std::filesystem::remove(path);
}

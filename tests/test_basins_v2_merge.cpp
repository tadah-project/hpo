#include "catch2/catch.hpp"

#include <tadah/hpo/basin_pool_v2.h>

#include <vector>

using tadah::hpo::BasinV2;
using tadah::hpo::BasinPoolV2;
using tadah::hpo::LifecycleState;

TEST_CASE("merge: find_merge_candidates returns unique pairs",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(/*dedup=*/0.0, /*max=*/8);
    p.set_normalisation({0.0, 0.0}, {10.0, 10.0});

    /* Two close pairs (a~b, c~d) and one isolated (e). */
    const int a = p.add_or_merge({1.00, 1.00}, 0.50, 1);
    const int b = p.add_or_merge({1.05, 1.00}, 0.51, 1);
    const int c = p.add_or_merge({5.00, 5.00}, 0.30, 1);
    const int d = p.add_or_merge({5.05, 5.00}, 0.31, 1);
    const int e = p.add_or_merge({9.00, 9.00}, 0.70, 1);
    (void)a; (void)b; (void)c; (void)d; (void)e;

    auto pairs = p.find_merge_candidates(/*dist=*/0.10, /*loss=*/0.10);
    REQUIRE(pairs.size() == 2u);
}

TEST_CASE("merge: candidates respect loss threshold",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(/*dedup=*/0.0, /*max=*/4);
    p.set_normalisation({0.0}, {10.0});
    p.add_or_merge({1.0}, 0.50, 1);
    p.add_or_merge({1.05}, 0.90, 1);   // close in distance, very different loss
    auto pairs = p.find_merge_candidates(/*dist=*/0.10, /*loss=*/0.10);
    REQUIRE(pairs.empty());
}

TEST_CASE("merge: flag_suspect_pair transitions both basins",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(0.0, 4);
    const int a = p.add_or_merge({1.0}, 0.5, 1);
    const int b = p.add_or_merge({2.0}, 0.5, 1);

    REQUIRE(p.flag_suspect_pair(a, b, /*iter=*/10));
    REQUIRE(p.by_id(a)->state == LifecycleState::SUSPECT_DUP);
    REQUIRE(p.by_id(b)->state == LifecycleState::SUSPECT_DUP);

    /* Idempotent. */
    REQUIRE_FALSE(p.flag_suspect_pair(a, b, /*iter=*/11));
}

TEST_CASE("merge: confirm_merge moves loser to MERGED with sibling_root_id",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(0.0, 4);
    const int a = p.add_or_merge({1.0}, 0.30, 1);   // winner (lower loss)
    const int b = p.add_or_merge({1.5}, 0.50, 1);   // loser
    p.flag_suspect_pair(a, b, 5);

    p.confirm_merge(/*loser=*/b, /*winner=*/a, /*iter=*/12);

    REQUIRE(p.by_id(b)->state == LifecycleState::MERGED);
    REQUIRE(p.by_id(b)->sibling_root_id == a);
    REQUIRE(p.by_id(a)->state == LifecycleState::SUSPECT_DUP);   // winner state untouched here
    REQUIRE(p.active_count() == 1u);
    REQUIRE(p.archive_count() == 1u);
}

TEST_CASE("merge: revert_suspect_pair restores ACTIVE on both",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(0.0, 4);
    const int a = p.add_or_merge({1.0}, 0.5, 1);
    const int b = p.add_or_merge({2.0}, 0.5, 1);
    p.flag_suspect_pair(a, b, 5);

    p.revert_suspect_pair(a, b, 12);
    REQUIRE(p.by_id(a)->state == LifecycleState::ACTIVE);
    REQUIRE(p.by_id(b)->state == LifecycleState::ACTIVE);
    REQUIRE(p.by_id(a)->last_iteration == 12);
}

TEST_CASE("merge: confirm_merge accumulates loser eval counts into winner",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(0.0, 4);
    const int a = p.add_or_merge({1.0}, 0.5, 1);
    const int b = p.add_or_merge({2.0}, 0.6, 1);
    p.on_refinement_complete(a, {1.0}, 0.4, 80, 2, 1e-3, false);
    p.on_refinement_complete(b, {2.0}, 0.6, 60, 2, 1e-3, true);

    REQUIRE(p.by_id(a)->evals_consumed_total == 80u);
    REQUIRE(p.by_id(b)->evals_consumed_total == 60u);

    p.confirm_merge(/*loser=*/b, /*winner=*/a, /*iter=*/5);
    REQUIRE(p.by_id(a)->evals_consumed_total == 140u);
}

TEST_CASE("merge: terminal basins are not candidates",
          "[basins_v2][merge]")
{
    BasinPoolV2 p(0.0, 4);
    const int a = p.add_or_merge({1.0}, 0.5, 1);
    const int b = p.add_or_merge({1.05}, 0.5, 1);
    p.set_normalisation({0.0}, {10.0});
    p.set_state(b, LifecycleState::KILLED, 5);

    auto pairs = p.find_merge_candidates(0.10, 0.10);
    REQUIRE(pairs.empty());
    /* Sanity: a is still ACTIVE. */
    REQUIRE(p.by_id(a)->state == LifecycleState::ACTIVE);
}

#include "catch2/catch.hpp"

#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/basins_v2_priority.h>

#include <limits>
#include <vector>

using tadah::hpo::BasinV2;
using tadah::hpo::LifecycleState;
using tadah::hpo::PriorityContext;
using tadah::hpo::PriorityWeights;
using tadah::hpo::compute_priorities;
using tadah::hpo::argmax_priority;
using tadah::hpo::min_max_normalise;
using tadah::hpo::state_bonus;

namespace {
BasinV2 make_basin(int id, double loss, LifecycleState s,
                   double ema, int last_iter, int birth_iter,
                   double tol_cursor)
{
    BasinV2 b;
    b.id = id;
    b.sibling_root_id = id;
    b.loss = loss;
    b.state = s;
    b.improvement_ema = ema;
    b.last_iteration = last_iter;
    b.birth_iteration = birth_iter;
    b.current_tol_cursor = tol_cursor;
    return b;
}
}   // namespace

TEST_CASE("priority: state_bonus matches design", "[basins_v2][priority]")
{
    REQUIRE(state_bonus(LifecycleState::PROMISING)   == Approx( 1.0));
    REQUIRE(state_bonus(LifecycleState::ACTIVE)      == Approx( 0.0));
    REQUIRE(state_bonus(LifecycleState::INACTIVE)    == Approx(-0.5));
    REQUIRE(state_bonus(LifecycleState::SUSPECT_DUP) == Approx(-2.0));
    REQUIRE(std::isinf(state_bonus(LifecycleState::MERGED)));
    REQUIRE(std::isinf(state_bonus(LifecycleState::KILLED)));
}

TEST_CASE("priority: min_max_normalise behaves at edges",
          "[basins_v2][priority]")
{
    REQUIRE(min_max_normalise(0.5, 0.0, 1.0) == Approx(0.5));
    REQUIRE(min_max_normalise(1.0, 1.0, 1.0) == Approx(0.0));   // degenerate
    REQUIRE(min_max_normalise(2.0, 0.0, 4.0) == Approx(0.5));
}

TEST_CASE("priority: PROMISING beats ACTIVE with otherwise-identical inputs",
          "[basins_v2][priority]")
{
    std::vector<BasinV2> basins;
    basins.push_back(make_basin(0, 1.0, LifecycleState::ACTIVE,    0.0, 0, 0, 1e-3));
    basins.push_back(make_basin(1, 1.0, LifecycleState::PROMISING, 0.0, 0, 0, 1e-3));

    std::vector<double> dists = {1.0, 1.0};
    PriorityContext ctx; ctx.current_iteration = 10; ctx.target_tol = 1e-9;
    PriorityWeights w;   /* defaults */

    auto scores = compute_priorities(basins, dists, ctx, w);
    REQUIRE(scores.size() == 2u);
    REQUIRE(scores[1] > scores[0]);   // PROMISING wins
    REQUIRE(argmax_priority(basins, scores) == 1);
}

TEST_CASE("priority: archived basins score -inf",
          "[basins_v2][priority]")
{
    std::vector<BasinV2> basins;
    basins.push_back(make_basin(0, 1.0, LifecycleState::ACTIVE, 0.0, 0, 0, 1e-3));
    basins.push_back(make_basin(1, 0.5, LifecycleState::KILLED, 1.0, 0, 0, 1e-9));
    basins.push_back(make_basin(2, 0.5, LifecycleState::MERGED, 1.0, 0, 0, 1e-9));

    std::vector<double> dists(3, 1.0);
    PriorityContext ctx; ctx.current_iteration = 10;
    PriorityWeights w;
    auto scores = compute_priorities(basins, dists, ctx, w);
    REQUIRE(std::isinf(scores[1]));
    REQUIRE(std::isinf(scores[2]));
    REQUIRE(scores[0] > -std::numeric_limits<double>::infinity());
    REQUIRE(argmax_priority(basins, scores) == 0);
}

TEST_CASE("priority: neglect raises a basin's score over time",
          "[basins_v2][priority]")
{
    /* Two ACTIVE basins; one was last refined long ago. With default
     * weights it should overtake the freshly-touched one as iteration
     * advances. */
    std::vector<BasinV2> basins;
    basins.push_back(make_basin(0, 1.0, LifecycleState::ACTIVE, 0.0,
                                /*last=*/9, 0, 1e-3));   // fresh
    basins.push_back(make_basin(1, 1.0, LifecycleState::ACTIVE, 0.0,
                                /*last=*/0, 0, 1e-3));   // neglected

    std::vector<double> dists = {1.0, 1.0};
    PriorityContext ctx; ctx.current_iteration = 10; ctx.target_tol = 1e-9;
    PriorityWeights w;

    auto scores = compute_priorities(basins, dists, ctx, w);
    REQUIRE(scores[1] > scores[0]);
    REQUIRE(argmax_priority(basins, scores) == 1);
}

TEST_CASE("priority: distance preference picks isolated basin",
          "[basins_v2][priority]")
{
    /* Two ACTIVE basins; basin 1 is far from peers, basin 0 is close.
     * Distance term boosts basin 1. */
    std::vector<BasinV2> basins;
    basins.push_back(make_basin(0, 1.0, LifecycleState::ACTIVE, 0.0, 0, 0, 1e-3));
    basins.push_back(make_basin(1, 1.0, LifecycleState::ACTIVE, 0.0, 0, 0, 1e-3));
    std::vector<double> dists = {0.01, 0.5};
    PriorityContext ctx; ctx.current_iteration = 0;
    PriorityWeights w;
    auto scores = compute_priorities(basins, dists, ctx, w);
    REQUIRE(scores[1] > scores[0]);
}

TEST_CASE("priority: tol_gap penalises basins far from target_tol",
          "[basins_v2][priority]")
{
    /* basin 0 is at 1e-1 (far from 1e-9 target).
     * basin 1 is at 1e-7 (close to target).
     * Negative weight on tol_gap → basin 1 wins. */
    std::vector<BasinV2> basins;
    basins.push_back(make_basin(0, 1.0, LifecycleState::ACTIVE, 0.0, 0, 0, 1e-1));
    basins.push_back(make_basin(1, 1.0, LifecycleState::ACTIVE, 0.0, 0, 0, 1e-7));
    std::vector<double> dists = {1.0, 1.0};
    PriorityContext ctx; ctx.current_iteration = 0; ctx.target_tol = 1e-9;
    PriorityWeights w;
    auto scores = compute_priorities(basins, dists, ctx, w);
    REQUIRE(scores[1] > scores[0]);
}

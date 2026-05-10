#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/basin_pool_v2.h>

#include <filesystem>
#include <vector>

using tadah::core::Checkpoint;
using tadah::hpo::BasinV2;
using tadah::hpo::BasinPoolV2;
using tadah::hpo::LifecycleState;
using tadah::hpo::PathPoint;
using tadah::hpo::is_archive_state;
using tadah::hpo::lifecycle_name;

TEST_CASE("BasinPoolV2: insert returns stable id, not vector index",
          "[basin_pool_v2]")
{
    BasinPoolV2 p(/*dedup=*/0.0, /*pool_max=*/3, /*archive_max=*/4);
    REQUIRE(p.empty());

    const int id_a = p.add_or_merge({1.0}, 0.5, 1);
    const int id_b = p.add_or_merge({2.0}, 0.3, 1);
    const int id_c = p.add_or_merge({3.0}, 0.7, 1);
    REQUIRE(id_a == 0);
    REQUIRE(id_b == 1);
    REQUIRE(id_c == 2);
    REQUIRE(p.size() == 3u);
    REQUIRE(p.active_count() == 3u);
    REQUIRE(p.archive_count() == 0u);
}

TEST_CASE("BasinPoolV2: dedup merges within distance, keeps better loss",
          "[basin_pool_v2]")
{
    BasinPoolV2 p(0.05, 8);
    p.set_normalisation({0.0, 0.0}, {10.0, 10.0});

    const int id1 = p.add_or_merge({1.0, 1.0}, 0.5, 1);
    const int id2 = p.add_or_merge({1.5, 1.0}, 0.3, 2);   // within 0.05
    REQUIRE(id1 == id2);
    REQUIRE(p.size() == 1u);

    const BasinV2* b = p.by_id(id1);
    REQUIRE(b != nullptr);
    REQUIRE(b->loss == Approx(0.3));
    REQUIRE(b->params[0] == Approx(1.5));
    /* Improvement EMA picks up the merge. */
    REQUIRE(b->improvement_ema > 0.0);
}

TEST_CASE("BasinPoolV2: capacity eviction prefers worst-loss active basin",
          "[basin_pool_v2]")
{
    BasinPoolV2 p(0.0, /*pool_max=*/2, /*archive_max=*/2);
    p.add_or_merge({1.0}, 0.5, 1);
    p.add_or_merge({2.0}, 0.9, 1);

    /* New 0.7 must evict 0.9 (worst). */
    const int id3 = p.add_or_merge({3.0}, 0.7, 2);
    REQUIRE(id3 >= 0);
    REQUIRE(p.active_count() == 2u);

    /* New 1.0 worse than every resident → refused. */
    const int id4 = p.add_or_merge({4.0}, 1.0, 3);
    REQUIRE(id4 == -1);
    REQUIRE(p.active_count() == 2u);
}

TEST_CASE("BasinPoolV2: lifecycle transitions update last_iteration",
          "[basin_pool_v2][lifecycle]")
{
    BasinPoolV2 p(0.0, 4);
    const int id = p.add_or_merge({1.0}, 0.5, /*iter=*/3);

    p.set_state(id, LifecycleState::INACTIVE, /*iter=*/10);
    const BasinV2* b = p.by_id(id);
    REQUIRE(b->state == LifecycleState::INACTIVE);
    REQUIRE(b->last_iteration == 10);

    /* MERGED is terminal — set_state into ACTIVE silently rejected. */
    p.set_state(id, LifecycleState::KILLED, /*iter=*/12);
    REQUIRE(p.by_id(id)->state == LifecycleState::KILLED);
    REQUIRE(is_archive_state(LifecycleState::KILLED));
    p.set_state(id, LifecycleState::ACTIVE, /*iter=*/13);
    /* Still KILLED — terminal transition refused. */
    REQUIRE(p.by_id(id)->state == LifecycleState::KILLED);

    REQUIRE(p.active_count() == 0u);
    REQUIRE(p.archive_count() == 1u);
}

TEST_CASE("BasinPoolV2: on_refinement_complete updates EMA and counters",
          "[basin_pool_v2][lifecycle]")
{
    BasinPoolV2 p(0.0, 4);
    const int id = p.add_or_merge({1.0}, 1.0, 1);

    /* Improvement event. */
    p.on_refinement_complete(id, {1.1}, /*loss=*/0.5,
                             /*evals=*/100, /*iter=*/2,
                             /*tol=*/1e-3, /*converged=*/false);
    const BasinV2* b = p.by_id(id);
    REQUIRE(b->loss == Approx(0.5));
    REQUIRE(b->improvement_ema > 0.0);
    REQUIRE(b->evals_consumed_total == 100u);
    REQUIRE(b->evals_since_improvement == 0u);
    REQUIRE(b->last_improve_iteration == 2);
    REQUIRE(b->last_attempted_tol.has_value());
    REQUIRE(*b->last_attempted_tol == Approx(1e-3));
    /* Not exhausted because no convergence and improvement happened. */
    REQUIRE_FALSE(b->last_exhausted_tol.has_value());

    /* Stagnation event. */
    p.on_refinement_complete(id, {1.1}, /*loss=*/0.5,
                             /*evals=*/50, /*iter=*/3,
                             /*tol=*/1e-3, /*converged=*/true);
    b = p.by_id(id);
    REQUIRE(b->evals_consumed_total == 150u);
    REQUIRE(b->evals_since_improvement == 50u);
    REQUIRE(b->last_exhausted_tol.has_value());
    REQUIRE(*b->last_exhausted_tol == Approx(1e-3));
}

TEST_CASE("BasinPoolV2: on_verify_attempt revives or counts toward kill",
          "[basin_pool_v2][lifecycle]")
{
    BasinPoolV2 p(0.0, 4);
    const int id = p.add_or_merge({1.0}, 0.5, 1);
    p.set_state(id, LifecycleState::INACTIVE, 10);

    p.on_verify_attempt(id, /*improved=*/false, /*iter=*/20);
    const BasinV2* b = p.by_id(id);
    REQUIRE(b->state == LifecycleState::INACTIVE);
    REQUIRE(b->verify_attempts == 1);

    /* Successful tickle — revived. */
    p.on_verify_attempt(id, /*improved=*/true, /*iter=*/30);
    b = p.by_id(id);
    REQUIRE(b->state == LifecycleState::ACTIVE);
    REQUIRE(b->verify_passes == 1);
    REQUIRE(b->evals_since_improvement == 0u);
}

TEST_CASE("BasinPoolV2: archive_max evicts oldest terminal basins",
          "[basin_pool_v2][archive]")
{
    BasinPoolV2 p(0.0, /*pool_max=*/4, /*archive_max=*/2);
    const int a = p.add_or_merge({1.0}, 0.1, /*iter=*/1);
    const int b = p.add_or_merge({2.0}, 0.2, /*iter=*/2);
    const int c = p.add_or_merge({3.0}, 0.3, /*iter=*/3);
    p.set_state(a, LifecycleState::KILLED, 10);
    p.set_state(b, LifecycleState::KILLED, 11);
    p.set_state(c, LifecycleState::KILLED, 12);
    /* Archive cap = 2 → oldest (a, born iter 1) evicted. */
    REQUIRE(p.archive_count() == 2u);
    REQUIRE(p.by_id(a) == nullptr);
    REQUIRE(p.by_id(b) != nullptr);
    REQUIRE(p.by_id(c) != nullptr);
}

TEST_CASE("BasinPoolV2: checkpoint round-trip preserves all fields",
          "[basin_pool_v2][checkpoint]")
{
    BasinPoolV2 p(0.07, 4, 8);
    p.set_normalisation({0.0, -1.0}, {10.0, 1.0});
    const int id1 = p.add_or_merge({1.0,  0.0}, 0.5, 1);
    const int id2 = p.add_or_merge({2.0,  0.5}, 0.3, 1);
    p.on_refinement_complete(id1, {1.0, 0.0}, 0.4, 80, 2, 1e-3, false);
    p.set_state(id2, LifecycleState::PROMISING, 5);
    PathPoint pp; pp.params = {1.05, 0.0}; pp.loss = 0.42; pp.eval_index = 42;
    p.append_path_point(id1, pp, /*cap=*/4);

    auto path = std::filesystem::temp_directory_path() / "tadah_bp_v2.tadah";
    std::filesystem::remove(path);
    {
        Checkpoint cp(path.string());
        p.save(cp);
        cp.commit();
    }

    BasinPoolV2 p2(0.0, 1, 1);
    Checkpoint cp(path.string());
    REQUIRE(cp.load_from_disk());
    p2.load(cp);

    REQUIRE(p2.dedup_dist()  == Approx(0.07));
    REQUIRE(p2.pool_max()    == 4u);
    REQUIRE(p2.archive_max() == 8u);
    REQUIRE(p2.size()        == 2u);

    const BasinV2* b1 = p2.by_id(id1);
    const BasinV2* b2 = p2.by_id(id2);
    REQUIRE(b1 != nullptr);
    REQUIRE(b2 != nullptr);
    REQUIRE(b1->loss == Approx(0.4));
    REQUIRE(b1->evals_consumed_total == 80u);
    REQUIRE(b1->coarse_path.size() == 1u);
    REQUIRE(b1->coarse_path[0].loss == Approx(0.42));
    REQUIRE(b1->coarse_path[0].eval_index == 42u);
    REQUIRE(b2->state == LifecycleState::PROMISING);

    std::filesystem::remove(path);
}

TEST_CASE("BasinPoolV2: lifecycle_name covers all states",
          "[basin_pool_v2][lifecycle]")
{
    REQUIRE(std::string(lifecycle_name(LifecycleState::ACTIVE))      == "ACTIVE");
    REQUIRE(std::string(lifecycle_name(LifecycleState::PROMISING))   == "PROMISING");
    REQUIRE(std::string(lifecycle_name(LifecycleState::INACTIVE))    == "INACTIVE");
    REQUIRE(std::string(lifecycle_name(LifecycleState::SUSPECT_DUP)) == "SUSPECT_DUP");
    REQUIRE(std::string(lifecycle_name(LifecycleState::MERGED))      == "MERGED");
    REQUIRE(std::string(lifecycle_name(LifecycleState::KILLED))      == "KILLED");
    REQUIRE(is_archive_state(LifecycleState::MERGED));
    REQUIRE(is_archive_state(LifecycleState::KILLED));
    REQUIRE_FALSE(is_archive_state(LifecycleState::ACTIVE));
    REQUIRE_FALSE(is_archive_state(LifecycleState::PROMISING));
    REQUIRE_FALSE(is_archive_state(LifecycleState::INACTIVE));
    REQUIRE_FALSE(is_archive_state(LifecycleState::SUSPECT_DUP));
}

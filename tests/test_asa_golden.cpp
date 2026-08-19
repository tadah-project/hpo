/* The ASA golden-neutrality gate (see external/asa/TADAH-PATCHES.md).
 *
 * Every compile-time gate enabled in tadah_asa_config.h claims to be
 * trajectory-neutral at its default option values. This test PROVES it:
 * the featured build and the pristine all-gates-off build run the same
 * deterministic problem with the same explicit LCG, and the recorded
 * trajectories must match bitwise. If this test fails after a gate
 * change: flip the offending gate off, remove its PARAM keys, document
 * the exclusion (the verify-or-drop procedure). */

#include "catch2/catch.hpp"

#include "asa_golden_common.h"
#include <tadah/hpo/asa_testing.h>

TEST_CASE("ASA golden gate: featured == pristine at default options",
          "[HPO][asa][golden]")
{
    constexpr std::size_t dim = 3;
    const double lo[dim] = {-4.0, -4.0, -4.0};
    const double hi[dim] = {4.0, 4.0, 4.0};
    const double x0[dim] = {2.5, -1.5, 0.5};
    const long limit_generated = 2000;

    asa_golden::reset(777);
    const auto p = run_golden_pristine(dim, lo, hi, x0, limit_generated);
    const asa_golden::Trajectory ptraj = asa_golden::traj();  /* copy */

    asa_golden::reset(777);
    const auto f = tadah::hpo::asa_testing::run_raw_featured(
        &asa_golden::golden_cost, &asa_golden::golden_rng, dim, lo, hi, x0,
        limit_generated);
    const asa_golden::Trajectory& ftraj = asa_golden::traj();

    INFO("pristine evals=" << ptraj.f.size()
         << " featured evals=" << ftraj.f.size());
    REQUIRE(ptraj.f.size() > 100);  /* the run actually ran */
    REQUIRE(ptraj.f.size() == ftraj.f.size());
    CHECK(ptraj.flat_x == ftraj.flat_x);  /* bitwise double compare */
    CHECK(ptraj.f == ftraj.f);
    CHECK(p.exit_status == f.exit_status);
    CHECK(p.n_accepted == f.n_accepted);
    CHECK(p.n_generated == f.n_generated);
    CHECK(p.best_x == f.best_x);
    CHECK(p.final_cost == f.final_cost);
}

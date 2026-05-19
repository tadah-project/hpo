#include "catch2/catch.hpp"

#include <tadah/hpo/basins_v2_priority.h>

using tadah::hpo::adaptive_refine_threshold;

TEST_CASE("adaptive_refine_threshold: dim floor dominates at loose tol",
          "[basins_v2][plateau]")
{
    /* tol = 1e-3, dim = 8 → tol_term = 50*3*0.5 = 75
     *   dim_floor = 800 → wins. */
    const std::size_t got =
        adaptive_refine_threshold(/*tol=*/1.0e-3, /*dim=*/8,
                                   /*min=*/30, /*max=*/100000,
                                   /*plateau_eps_factor=*/0.5);
    REQUIRE(got == 800u);
}

TEST_CASE("adaptive_refine_threshold: tol_term grows with tighter tol",
          "[basins_v2][plateau]")
{
    /* dim = 1, plateau_eps_factor = 1.0; tol_term = 50*|log10(tol)|.
     *   tol = 1e-2  → tol_term = 100; dim_floor = 100; max(...) = 100
     *   tol = 1e-6  → tol_term = 300
     *   tol = 1e-9  → tol_term = 450 */
    REQUIRE(adaptive_refine_threshold(1e-2, 1, 30, 100000, 1.0) == 100u);
    REQUIRE(adaptive_refine_threshold(1e-6, 1, 30, 100000, 1.0) == 300u);
    REQUIRE(adaptive_refine_threshold(1e-9, 1, 30, 100000, 1.0) == 450u);
}

TEST_CASE("adaptive_refine_threshold: respects min and max caps",
          "[basins_v2][plateau]")
{
    /* min_evals dominates if both other terms are smaller. */
    REQUIRE(adaptive_refine_threshold(1.0, 1, /*min=*/2000, /*max=*/100000, 0.5)
            == 2000u);
    /* max_evals caps the result. */
    REQUIRE(adaptive_refine_threshold(1e-9, 100, /*min=*/30, /*max=*/500, 0.5)
            == 500u);
}

TEST_CASE("adaptive_refine_threshold: zero plateau_eps_factor uses default 0.5",
          "[basins_v2][plateau]")
{
    /* With factor = 0 we silently fall back to 0.5 (defensive). */
    const std::size_t got =
        adaptive_refine_threshold(1.0e-6, 1, 30, 100000, /*factor=*/0.0);
    REQUIRE(got == 150u);   // 50*6*0.5 = 150
}

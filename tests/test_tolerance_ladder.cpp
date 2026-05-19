#include "catch2/catch.hpp"

#include <tadah/hpo/tolerance_ladder.h>

using tadah::hpo::ToleranceLadder;

TEST_CASE("ToleranceLadder: explicit lists are sorted descending",
          "[tolerance_ladder]")
{
    auto t = ToleranceLadder::from_explicit({1e-4, 1e-2, 1e-3},
                                            {1e-5, 1e-3});
    REQUIRE(t.glo_ladder().size() == 3);
    REQUIRE(t.glo_ladder()[0] == Approx(1e-2));
    REQUIRE(t.glo_ladder()[1] == Approx(1e-3));
    REQUIRE(t.glo_ladder()[2] == Approx(1e-4));
    REQUIRE(t.loc_ladder().size() == 2);
    REQUIRE(t.loc_ladder()[0] == Approx(1e-3));
    REQUIRE(t.loc_ladder()[1] == Approx(1e-5));
}

TEST_CASE("ToleranceLadder: glo() clamps to last entry past length",
          "[tolerance_ladder]")
{
    auto t = ToleranceLadder::from_explicit({1e-2, 1e-4}, {1e-2, 1e-4});
    REQUIRE(t.glo(1) == Approx(1e-2));
    REQUIRE(t.glo(2) == Approx(1e-4));
    REQUIRE(t.glo(99) == Approx(1e-4));   // clamps
    REQUIRE(t.glo(0) == Approx(1e-2));    // round=0 -> idx 0
}

TEST_CASE("ToleranceLadder: cumulative loc rungs grow with round",
          "[tolerance_ladder]")
{
    auto t = ToleranceLadder::from_explicit({1e-3}, {1e-3, 1e-4, 1e-5, 1e-6});
    REQUIRE(t.loc_steps_for(1) == 1);
    REQUIRE(t.loc_steps_for(2) == 2);
    REQUIRE(t.loc_steps_for(3) == 3);
    REQUIRE(t.loc_steps_for(4) == 4);
    REQUIRE(t.loc_steps_for(5) == 4);   // capped at ladder size

    /* Round 1 only sees the first rung (loose). */
    REQUIRE(t.loc(1, 1) == Approx(1e-3));
    /* Round 4 sees four rungs cumulatively. */
    REQUIRE(t.loc(4, 1) == Approx(1e-3));
    REQUIRE(t.loc(4, 2) == Approx(1e-4));
    REQUIRE(t.loc(4, 3) == Approx(1e-5));
    REQUIRE(t.loc(4, 4) == Approx(1e-6));
    /* Round 2 cap: requesting step 4 clamps to step 2. */
    REQUIRE(t.loc(2, 4) == Approx(1e-4));
}

TEST_CASE("ToleranceLadder: build_default uses noise floor",
          "[tolerance_ladder]")
{
    auto t = ToleranceLadder::build_default(/*noise_rel_spread=*/1e-5);
    /* Floor = 10 * 1e-5 = 1e-4. Default ladder cuts off there. */
    REQUIRE_FALSE(t.glo_ladder().empty());
    for (double v : t.glo_ladder()) REQUIRE(v >= 1e-4);
    REQUIRE(t.glo_ladder().front() == Approx(1e-3));
}

TEST_CASE("ToleranceLadder: build_default with no calibration uses 1e-9 floor",
          "[tolerance_ladder]")
{
    auto t = ToleranceLadder::build_default(std::nullopt);
    /* Default 6 rungs starting at 1e-3 -> 1e-8. */
    REQUIRE(t.glo_ladder().size() == 6);
    REQUIRE(t.glo_ladder().front() == Approx(1e-3));
    REQUIRE(t.glo_ladder().back()  == Approx(1e-8));
    /* loc has up to 2 extra rungs (max_rungs=8) but the loop breaks
     * once cur < floor (1e-9), giving 7 entries: 1e-3 .. 1e-9. */
    REQUIRE(t.loc_ladder().size() == 7);
    REQUIRE(t.loc_ladder().front() == Approx(1e-3));
    REQUIRE(t.loc_ladder().back()  == Approx(1e-9));
}

TEST_CASE("ToleranceLadder: trim_below drops sub-floor rungs",
          "[tolerance_ladder][noise_clamp]")
{
    auto t = ToleranceLadder::from_explicit(
            {1e-3, 1e-4, 1e-5, 1e-6}, {1e-3, 1e-4, 1e-5, 1e-6, 1e-7});
    /* floor = 1e-6 → drop rungs strictly less than 1e-6 (i.e. 1e-7
     * from loc only). 1e-6 itself is retained. */
    t.trim_below(1e-6);
    REQUIRE(t.glo_ladder().size() == 4);
    REQUIRE(t.glo_ladder().back()  == Approx(1e-6));
    REQUIRE(t.loc_ladder().size() == 4);
    REQUIRE(t.loc_ladder().back()  == Approx(1e-6));
}

TEST_CASE("ToleranceLadder: trim_below at the loosest rung leaves single entry",
          "[tolerance_ladder][noise_clamp]")
{
    auto t = ToleranceLadder::from_explicit(
            {1e-3, 1e-4, 1e-5}, {1e-3, 1e-4, 1e-5, 1e-6});
    /* floor = 1e-3 → drop everything strictly below; only 1e-3
     * remains in both. */
    t.trim_below(1e-3);
    REQUIRE(t.glo_ladder().size() == 1);
    REQUIRE(t.glo_ladder().front() == Approx(1e-3));
    REQUIRE(t.loc_ladder().size() == 1);
    REQUIRE(t.loc_ladder().front() == Approx(1e-3));
}

TEST_CASE("ToleranceLadder: trim_below above the loosest rung pushes floor",
          "[tolerance_ladder][noise_clamp]")
{
    auto t = ToleranceLadder::from_explicit({1e-5, 1e-6}, {1e-5, 1e-6});
    /* floor = 1.0 — strictly greater than every rung — both ladders
     * empty out, then the never-empty rule pushes a single floor rung. */
    t.trim_below(1.0);
    REQUIRE(t.glo_ladder().size() == 1);
    REQUIRE(t.glo_ladder().front() == Approx(1.0));
    REQUIRE(t.loc_ladder().size() == 1);
    REQUIRE(t.loc_ladder().front() == Approx(1.0));
}

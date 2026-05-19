#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/basins_policy.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/run_policy.h>

#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

struct ScopedTempFile {
    std::filesystem::path path;
    explicit ScopedTempFile(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_basins_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".in"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~ScopedTempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

}  // namespace

TEST_CASE("BASINS parser: minimal config picks defaults",
          "[basins][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.strategy_type == tadah::hpo::StrategyType::BASINS);
    /* Defaults from BasinsSpec: NLopt CRS2_LM + LN_BOBYQA. */
    REQUIRE(trg.spec.basins.explore);
    REQUIRE(trg.spec.basins.refine);
    CHECK(trg.spec.basins.explore->lib  == "NLOPT");
    CHECK(trg.spec.basins.explore->algo == "GN_CRS2_LM");
    CHECK(trg.spec.basins.refine->lib   == "NLOPT");
    CHECK(trg.spec.basins.refine->algo  == "LN_BOBYQA");
    CHECK(trg.spec.basins.basin_pool_max         == 32u);
    CHECK(trg.spec.basins.stop_no_improve_rounds == 5u);
}

TEST_CASE("BASINS parser: explicit overrides land in BasinsSpec",
          "[basins][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  EXPLORE
    LIB NLOPT
    ALGO GN_DIRECT_L
    FTOL_LADDER 1e-2 1e-3 1e-4
    BUDGET_MAX_EVALS 10000
    BUDGET_MIN_EVALS 100
    BUDGET_RATE_FACTOR 0.25
  ENDEXPLORE
  REFINE
    LIB NLOPT
    ALGO LN_BOBYQA
    FTOL_LADDER 1e-3 1e-5 1e-7
  ENDREFINE
  BASIN_POOL_MAX 64
  BASIN_DEDUP_DIST 0.1
  STOP_NO_IMPROVE_ROUNDS 8
  CHECKPOINT_EVERY 30s
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    const auto& b = trg.spec.basins;

    REQUIRE(b.explore);
    REQUIRE(b.refine);
    CHECK(b.explore->algo == "GN_DIRECT_L");
    CHECK(b.refine->algo  == "LN_BOBYQA");
    REQUIRE(b.explore_ftol_ladder.size() == 3);
    /* Parser stores the user's order; ToleranceLadder sorts internally. */
    CHECK(b.explore_ftol_ladder[0] == Approx(1e-2));
    CHECK(b.explore_ftol_ladder[2] == Approx(1e-4));
    CHECK(b.basin_pool_max               == 64u);
    CHECK(b.basin_dedup_dist             == Approx(0.1));
    CHECK(b.explore_budget_rate_factor   == Approx(0.25));
    CHECK(b.explore_budget_min_evals     == 100u);
    CHECK(b.explore_budget_max_evals     == 10000u);
    CHECK(b.stop_no_improve_rounds       == 8u);
    CHECK(b.checkpoint_seconds           == Approx(30.0));
    CHECK(b.checkpoint_evals             == 0u);
}

TEST_CASE("BASINS parser: CHECKPOINT_EVERY accepts both seconds and evals",
          "[basins][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  CHECKPOINT_EVERY 250
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.basins.checkpoint_seconds == Approx(0.0));
    CHECK(trg.spec.basins.checkpoint_evals   == 250u);
}

TEST_CASE("BASINS parser: bad inputs are rejected", "[basins][optim]")
{
    SECTION("empty FTOL_LADDER rejected") {
        ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  EXPLORE
    FTOL_LADDER
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
    SECTION("non-positive rung rejected") {
        ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  REFINE
    FTOL_LADDER 1e-3 -1
  ENDREFINE
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
    SECTION("zero pool max rejected") {
        ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  BASIN_POOL_MAX 0
ENDSTRATEGY
ENDOPTIMIZER
)");
        tadah::core::Context ctx;
        REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
    }
}

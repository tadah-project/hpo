#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <filesystem>
#include <fstream>

namespace {
struct ScopedTempFile {
    std::filesystem::path path;
    explicit ScopedTempFile(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_basins_v2_parser_" +
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
}   // namespace

TEST_CASE("BASINSV2 parser: minimal config picks defaults",
          "[basins_v2][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINSV2
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    REQUIRE(trg.spec.strategy_type == tadah::hpo::StrategyType::BASINSV2);
    REQUIRE(trg.spec.basins_v2.explore);
    REQUIRE(trg.spec.basins_v2.refine);
    CHECK(trg.spec.basins_v2.explore->lib  == "NLOPT");
    CHECK(trg.spec.basins_v2.explore->algo == "GN_CRS2_LM");
    CHECK(trg.spec.basins_v2.refine->lib   == "NLOPT");
    CHECK(trg.spec.basins_v2.refine->algo  == "LN_BOBYQA");
    CHECK(trg.spec.basins_v2.basin_pool_max == 24u);
    CHECK(trg.spec.basins_v2.verify_cadence == 30u);
    CHECK(trg.spec.basins_v2.merge_cadence  == 25u);
    CHECK(trg.spec.basins_v2.summary_cadence== 25u);
    /* Default priority weights. */
    CHECK(trg.spec.basins_v2.w_improve  == Approx(2.0));
    CHECK(trg.spec.basins_v2.w_neglect  == Approx(0.5));
    CHECK(trg.spec.basins_v2.w_distance == Approx(0.3));
    CHECK(trg.spec.basins_v2.w_tol_gap  == Approx(0.5));
    CHECK(trg.spec.basins_v2.w_state    == Approx(1.0));
    CHECK(trg.spec.basins_v2.w_age      == Approx(0.05));
}

TEST_CASE("BASINSV2 parser: explicit overrides land in BasinsV2Spec",
          "[basins_v2][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINSV2
  EXPLORE
    LIB NLOPT
    ALGO GN_CRS2_LM
    FTOL_LADDER 1e-2 1e-3 1e-4 1e-5 1e-6
    BUDGET_MAX_EVALS 1500
  ENDEXPLORE
  REFINE
    LIB NLOPT
    ALGO LN_BOBYQA
    FTOL_LADDER 1e-4 1e-5 1e-6 1e-7 1e-8
  ENDREFINE
  BASIN_POOL_MAX 8
  BASIN_DEDUP_DIST 0.10
  BASIN_MERGE_DIST 0.20
  BASIN_MERGE_LOSS_THRESHOLD 0.02
  ARCHIVE_MAX 16
  INACTIVE_AFTER_EVALS 800
  VERIFY_CADENCE 50
  VERIFY_MAX_ATTEMPTS 3
  VERIFY_EVALS_CAP 64
  MERGE_CADENCE 30
  EXPLORE_KICK_CADENCE 75
  EXPLORE_DIST_FLOOR 0.04
  SUMMARY_CADENCE 10
  STARVATION_CAP 100
  STOP_NO_IMPROVE_ITERATIONS 250
  PLATEAU_EPS_FACTOR 0.25
  PATH_MAX_POINTS 32
  PATH_LOSS_EPS 0.005
  PATH_DIST_EPS 0.02
  PRIORITY_WEIGHTS 3.0 1.0 0.5 0.7 1.5 0.1
  VERBOSE_DIAGNOSTICS true
  CHECKPOINT_EVERY 45s
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());
    const auto& b = trg.spec.basins_v2;

    REQUIRE(b.explore_ftol_ladder.size() == 5);
    REQUIRE(b.refine_ftol_ladder.size()  == 5);
    CHECK(b.explore_ftol_ladder.front() == Approx(1e-2));
    CHECK(b.refine_ftol_ladder.back()   == Approx(1e-8));
    CHECK(b.explore_budget_max_evals    == 1500u);
    CHECK(b.basin_pool_max              == 8u);
    CHECK(b.basin_dedup_dist            == Approx(0.10));
    CHECK(b.basin_merge_dist            == Approx(0.20));
    CHECK(b.basin_merge_loss_threshold  == Approx(0.02));
    CHECK(b.archive_max                 == 16u);
    CHECK(b.inactive_after_evals        == 800u);
    CHECK(b.verify_cadence              == 50u);
    CHECK(b.verify_max_attempts         == 3u);
    CHECK(b.verify_evals_cap            == 64u);
    CHECK(b.merge_cadence               == 30u);
    CHECK(b.explore_kick_cadence        == 75u);
    CHECK(b.explore_dist_floor          == Approx(0.04));
    CHECK(b.summary_cadence             == 10u);
    CHECK(b.starvation_cap              == 100u);
    CHECK(b.stop_no_improve_iterations  == 250u);
    CHECK(b.plateau_eps_factor          == Approx(0.25));
    CHECK(b.path_max_points             == 32u);
    CHECK(b.path_loss_eps               == Approx(0.005));
    CHECK(b.path_dist_eps               == Approx(0.02));
    CHECK(b.w_improve  == Approx(3.0));
    CHECK(b.w_neglect  == Approx(1.0));
    CHECK(b.w_distance == Approx(0.5));
    CHECK(b.w_tol_gap  == Approx(0.7));
    CHECK(b.w_state    == Approx(1.5));
    CHECK(b.w_age      == Approx(0.1));
    CHECK(b.verbose_diagnostics);
    CHECK(b.checkpoint_seconds == Approx(45.0));
}

TEST_CASE("BASINSV2 parser: BASINS-only key on BASINSV2 is rejected",
          "[basins_v2][optim]")
{
    /* SEED_TOP_N is BASINS-only. */
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINSV2
  SEED_TOP_N 3
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
}

TEST_CASE("BASINSV2 parser: BASINSV2-only key on BASINS is rejected",
          "[basins_v2][optim]")
{
    /* PRIORITY_WEIGHTS is BASINSV2-only. */
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  PRIORITY_WEIGHTS 1 1 1 1 1 1
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
}

TEST_CASE("BASINSV2 parser: PRIORITY_WEIGHTS short value list rejected",
          "[basins_v2][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINSV2
  PRIORITY_WEIGHTS 1 1 1
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    REQUIRE_THROWS(tadah::hpo::HPOTargets(ctx, input.path.string()));
}

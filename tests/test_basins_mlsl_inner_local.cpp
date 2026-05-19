/* test_basins_mlsl_inner_local.cpp ------------------------------------------
 *
 * Regression test: BASINS' build_stage_spec must preserve the EXPLORE-block's
 * INNER sub-spec. The NLopt adapter consumes `s.inner` via
 * `set_local_optimizer(make_nlopt(*sp.inner, dim))` for MLSL/AUGLAG-family
 * globals (G_MLSL, G_MLSL_LDS, AUGLAG, AUGLAG_EQ), and silently ignores it
 * for algorithms that don't need it.
 *
 * The test:
 *   1. Parses a config with STRATEGY { TYPE BASINS } and an EXPLORE block
 *      containing G_MLSL_LDS + an INNER sub-block (LN_SBPLX).
 *   2. Constructs a BasinsPolicy from the parsed spec.
 *   3. Calls build_stage_spec_for_test(true, tol) — the public test hook
 *      that delegates to build_stage_spec.
 *   4. Verifies the returned OptimizerSpec carries the inner block:
 *        - s.inner non-null
 *        - s.inner->lib  == "NLOPT"
 *        - s.inner->algo == "LN_SBPLX"
 * --------------------------------------------------------------------- */
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/basins_policy.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

struct ScopedTempFile {
    std::filesystem::path path;
    explicit ScopedTempFile(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_basins_mlsl_" +
            std::to_string(std::chrono::steady_clock::now()
                           .time_since_epoch().count()) +
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

TEST_CASE("BASINS+MLSL: build_stage_spec preserves EXPLORE INNER sub-block",
          "[basins][mlsl][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  EXPLORE
    LIB  NLOPT
    ALGO G_MLSL_LDS
    INNER
      LIB  NLOPT
      ALGO LN_SBPLX
      FTOL_REL 1e-4
      MAXEVAL 200
    ENDINNER
  ENDEXPLORE
  REFINE
    LIB  NLOPT
    ALGO LN_BOBYQA
  ENDREFINE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    /* Sanity: parser populated explore->inner. */
    REQUIRE(trg.spec.basins.explore);
    REQUIRE(trg.spec.basins.explore->inner);
    CHECK(trg.spec.basins.explore->inner->lib  == "NLOPT");
    CHECK(trg.spec.basins.explore->inner->algo == "LN_SBPLX");

    using V = dlib::matrix<double, 0, 1>;
    tadah::hpo::BasinsPolicy<V, ::LammpsRunner> policy(trg.spec, &trg);

    SECTION("EXPLORE stage carries the INNER block (MLSL needs it)") {
        const auto s = policy.build_stage_spec_for_test(/*global=*/true, 1e-3);
        CHECK(s.lib  == "NLOPT");
        CHECK(s.algo == "G_MLSL_LDS");
        REQUIRE(s.inner);
        CHECK(s.inner->lib  == "NLOPT");
        CHECK(s.inner->algo == "LN_SBPLX");
        REQUIRE(s.inner->stop.ftol_rel);
        CHECK(*s.inner->stop.ftol_rel == Approx(1e-4));
        REQUIRE(s.inner->stop.maxeval);
        CHECK(*s.inner->stop.maxeval  == 200u);
    }

    SECTION("REFINE stage carries no INNER (none was set)") {
        const auto s = policy.build_stage_spec_for_test(/*global=*/false, 1e-4);
        CHECK(s.lib  == "NLOPT");
        CHECK(s.algo == "LN_BOBYQA");
        CHECK_FALSE(s.inner);
    }
}

TEST_CASE("BASINS: build_stage_spec is a no-op on s.inner when none was set",
          "[basins][mlsl][optim]")
{
    ScopedTempFile input(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
  EXPLORE
    LIB NLOPT
    ALGO GN_CRS2_LM
  ENDEXPLORE
  REFINE
    LIB NLOPT
    ALGO LN_BOBYQA
  ENDREFINE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    /* No INNER blocks in the config -> explore/refine inner stay null. */
    REQUIRE(trg.spec.basins.explore);
    REQUIRE(trg.spec.basins.refine);
    REQUIRE_FALSE(trg.spec.basins.explore->inner);
    REQUIRE_FALSE(trg.spec.basins.refine->inner);

    using V = dlib::matrix<double, 0, 1>;
    tadah::hpo::BasinsPolicy<V, ::LammpsRunner> policy(trg.spec, &trg);

    const auto s_explore = policy.build_stage_spec_for_test(true,  1e-3);
    const auto s_refine  = policy.build_stage_spec_for_test(false, 1e-4);
    CHECK_FALSE(s_explore.inner);
    CHECK_FALSE(s_refine.inner);
}

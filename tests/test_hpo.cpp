#include "catch2/catch.hpp"

#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/core/context.h>

#include <chrono>
#include <filesystem>
#include <fstream>

/* HPO integration tests
 *
 * Full regression coverage of HPO_Host requires actual training data
 * (StructureDB files) which are not bundled with the source tree.
 * The key invariants validated by unit tests in MODELS/tests/test_reinit.cpp
 * already cover the bugs found during the reinit() implementation:
 *
 *   - D2_mJoin::reinit() / DM_mJoin::reinit() outer fcut updated
 *     when RCUT changes (Bug 2 — the tadah-predict mismatch).
 *   - M_Core::reset_trained() correctly clears the trained flag (Bug 3).
 *
 * What belongs here when data is available
 * -----------------------------------------
 * 1. HPO_Host self-consistency test
 *    Run 5 iterations of HPO on a tiny dataset (e.g. 3 structures).
 *    For each best iteration recorded in best_outvar, load the
 *    corresponding best_pot from BestPots/ and verify that
 *    tadah-predict gives the same ERSQ as HPO reported.
 *    This is the integration-level regression for the RCUT prefilter bug.
 *
 * 2. flush_remaining() test
 *    Start HPO, interrupt at iteration N by throwing from the objective
 *    function, then call flush_remaining() and verify that all N results
 *    were written to the output files (no truncated output).
 *
 * 3. Standalone optimiser parser coverage
 *    Keep a self-contained config for DLIB:BFGS so the parser/validator
 *    contract stays locked down independently of larger integration data.
 */

namespace {

struct ScopedTempFile {
  std::filesystem::path path;

  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_hpo_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
    os.close();
    REQUIRE(os.good());
  }

  ~ScopedTempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

}  // namespace

TEST_CASE("HPO placeholder — see comments above for planned tests", "[HPO]")
{
    // No self-contained integration tests yet.
    // All unit-level regression tests are in MODELS/tests/test_reinit.cpp.
    SUCCEED();
}

TEST_CASE("HPO parses standalone DLIB BFGS optimizer blocks",
          "[HPO][optim][dlib][bfgs]")
{
    ScopedTempFile input(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 25
PARAM fd_step 1e-6
PARAM tol_gradinf 1e-5
ENDOPTIMIZER
)");

    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, input.path.string());

    CHECK(trg.spec.lib == "DLIB");
    CHECK(trg.spec.algo == "BFGS");
    REQUIRE(trg.spec.stop.maxeval.has_value());
    CHECK(*trg.spec.stop.maxeval == 25u);
    REQUIRE(trg.spec.param.count("fd_step") == 1);
    CHECK(trg.spec.param.at("fd_step") == Approx(1.0e-6));
    REQUIRE(trg.spec.param.count("tol_gradinf") == 1);
    CHECK(trg.spec.param.at("tol_gradinf") == Approx(1.0e-5));
}

TEST_CASE("DLIB BFGS validator rejects LOCAL blocks",
          "[HPO][optim][dlib][bfgs][validation]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.low = {-1.0, -2.0};
    sp.high = {1.0, 2.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.0, 0.5}};
    sp.local = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.local->lib = "NLOPT";
    sp.local->algo = "LN_BOBYQA";

    CHECK_THROWS_WITH(tadah::hpo::detail::validate_dlib_bfgs_spec(sp),
                      Catch::Contains("standalone optimiser") &&
                      Catch::Contains("LOCAL"));
}

TEST_CASE("DLIB BFGS validator accepts standalone bounded specs",
          "[HPO][optim][dlib][bfgs][validation]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.low = {-1.0, -2.0};
    sp.high = {1.0, 2.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.0, 0.5}};
    sp.stop.maxeval = 10;
    sp.param["fd_step"] = 1.0e-6;

    CHECK_NOTHROW(tadah::hpo::detail::validate_dlib_bfgs_spec(sp));
}

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <filesystem>
#include <fstream>

/* Bug 3 regression — NOISE_CALIBRATION must be skipped when the run is
 * resumed from a checkpoint. The prior run already calibrated noise;
 * re-running charges N extra LAMMPS evals against the resumed counter,
 * pollutes the per-eval logs, and (combined with Bug 1's missing
 * g_best_ persistence) sets up a spurious post-resume "NEW BEST".
 *
 * The gating logic was extracted into should_run_noise_calibration()
 * so the predicate is unit-testable without invoking the full
 * optimiser. End-to-end coverage of the ordering (try_load() runs
 * before noise calibration) lives in test_resume_idempotence. */

namespace {

struct ScopedTempFile {
    std::filesystem::path path;
    explicit ScopedTempFile(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_noise_cal_" +
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

TEST_CASE("NOISE_CALIBRATION gate skips on resume",
          "[noise][resume][bug3]")
{
    /* The bug: original code was `if (sp.noise.calibration_n >= 2)`
     * with no resumed-flag check. Now the gate is two-armed: knob set AND
     * not resumed. */
    using tadah::hpo::should_run_noise_calibration;

    SECTION("fresh run with N>=2 → run") {
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/2));
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/10));
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/100));
    }
    SECTION("fresh run with N<2 → skip") {
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/0));
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/1));
    }
    SECTION("resumed run with N>=2 → skip (Bug 3 fix)") {
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/true, /*noise_n=*/2));
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/true, /*noise_n=*/10));
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/true, /*noise_n=*/100));
    }
    SECTION("resumed run with N<2 → skip") {
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/true, /*noise_n=*/0));
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/true, /*noise_n=*/1));
    }
}

TEST_CASE("HPOTargets parses NOISE.CALIBRATION value into spec",
          "[noise][resume][bug3]")
{
    /* Sanity check: the parser populates spec.noise.calibration_n so the
     * gate has a value to inspect at hpo_run() time. */
    ScopedTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION 5
ENDNOISE
STRATEGY
  TYPE BASINS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.calibration_n == 5u);
}

TEST_CASE("HPOTargets default NOISE.CALIBRATION is below the gate",
          "[noise][resume][bug3]")
{
    ScopedTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE BASINS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    /* Default should not trigger calibration. */
    using tadah::hpo::should_run_noise_calibration;
    CHECK_FALSE(should_run_noise_calibration(
        /*resumed=*/false,
        trg.spec.noise.calibration_n));
}

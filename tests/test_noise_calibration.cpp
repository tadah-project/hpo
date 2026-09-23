#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <filesystem>
#include <fstream>

/* NOISE_CALIBRATION gate -- hpo_run() repeats the seed point N times
 * before the optimiser starts when the knob asks for at least two
 * evaluations (one evaluation has no spread to measure). The predicate
 * lives in should_run_noise_calibration() so the gating is unit-testable
 * without invoking the full optimiser. */

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

TEST_CASE("NOISE_CALIBRATION gate runs for N >= 2 only",
          "[noise]")
{
    using tadah::hpo::should_run_noise_calibration;

    SECTION("N >= 2 -> run") {
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/2));
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/10));
        CHECK(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/100));
    }
    SECTION("N < 2 -> skip") {
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/0));
        CHECK_FALSE(should_run_noise_calibration(/*resumed=*/false, /*noise_n=*/1));
    }
}

TEST_CASE("HPOTargets parses NOISE.CALIBRATION value into spec",
          "[noise]")
{
    /* Sanity check: the parser populates spec.noise.calibration_n so the
     * gate has a value to inspect at hpo_run() time. */
    ScopedTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION 5
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB NLOPT
  ALGO LN_BOBYQA
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.calibration_n == 5u);
    using tadah::hpo::should_run_noise_calibration;
    CHECK(should_run_noise_calibration(
        /*resumed=*/false,
        trg.spec.noise.calibration_n));
}

TEST_CASE("HPOTargets default NOISE.CALIBRATION is below the gate",
          "[noise]")
{
    ScopedTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB NLOPT
  ALGO LN_BOBYQA
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

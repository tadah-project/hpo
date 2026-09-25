/* What DLIB:GFS is given for an evaluation's loss (detail::GfsLossForDlib,
 * dlib_adapter.h).
 *
 * dlib's find_min_global crashed (SIGSEGV in its trust-region step) once
 * the dims + 2 samples it held all sat at or above 1e300, which a failed
 * evaluation's default FAILSCORE, the largest double, does. run_dlib_global
 * hands dlib a finite stand-in for a failed evaluation instead: the worst
 * usable loss so far plus a margin, or, before any usable loss, FAILSCORE
 * capped at 1e100. The end-to-end runs are in test_hpo_termination.cpp
 * ("DLIB:GFS: ..."); these cases pin the values and run dlib itself on
 * them. */

#include "catch2/catch.hpp"

#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/pc_fail_grading.h>

#include <dlib/global_optimization.h>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace {

using tadah::hpo::detail::GfsLossForDlib;
constexpr double kDblMax = std::numeric_limits<double>::max();
constexpr double kCeiling = GfsLossForDlib::kCeiling;

}  // namespace

TEST_CASE("GFS loss for dlib: before any usable loss a failure is FAILSCORE, "
          "at most 1e100", "[HPO][gfs][dlib]")
{
    SECTION("the default FAILSCORE, the largest double") {
        GfsLossForDlib to_dlib(kDblMax);
        CHECK(to_dlib.failed(kDblMax));
        CHECK(to_dlib(kDblMax) == kCeiling);
        CHECK(to_dlib(std::numeric_limits<double>::infinity()) == kCeiling);
        CHECK(to_dlib(std::numeric_limits<double>::quiet_NaN()) == kCeiling);
        CHECK(to_dlib.not_evaluated() == kCeiling);
    }
    SECTION("a finite FAILSCORE below the ceiling is kept, graded failures "
            "too") {
        const double fs = 1e9;
        GfsLossForDlib to_dlib(fs);
        CHECK(to_dlib(fs) == fs);
        const double graded = tadah::hpo::kFailBandFloor * fs + 0.25e6;
        REQUIRE(to_dlib.failed(graded));
        CHECK(to_dlib(graded) == Approx(graded).epsilon(1e-15));
        CHECK(to_dlib.not_evaluated() == fs);
    }
    SECTION("a FAILSCORE above the ceiling maps its failure band onto "
            "[0.999, 1] x 1e100") {
        const double fs = 1e200;
        GfsLossForDlib to_dlib(fs);
        CHECK(to_dlib(fs) == Approx(kCeiling));
        const double band_bottom = tadah::hpo::kFailBandFloor * fs;
        CHECK(to_dlib(band_bottom) ==
              Approx(tadah::hpo::kFailBandFloor * kCeiling));
        const double mid = band_bottom + 0.5 * (fs - band_bottom);
        CHECK(to_dlib(mid) == Approx(0.9995 * kCeiling));
    }
}

TEST_CASE("GFS loss for dlib: after usable losses a failure is the worst "
          "usable loss plus a margin", "[HPO][gfs][dlib]")
{
    const double fs = 1e9;
    GfsLossForDlib to_dlib(fs);
    /* Usable losses reach dlib as they are. */
    CHECK_FALSE(to_dlib.failed(5.0));
    CHECK(to_dlib(5.0) == 5.0);
    CHECK(to_dlib(3.0) == 3.0);
    /* W = 5, B = 3: m = max(W - B, |W|) = 5. An ungraded failure is
     * W + 2m, a graded one W + m (1 + g) with g its place in the band. */
    CHECK(to_dlib(fs) == 15.0);
    CHECK(to_dlib.not_evaluated() == 15.0);
    const double band_bottom = tadah::hpo::kFailBandFloor * fs;
    CHECK(to_dlib(band_bottom) == Approx(10.0));
    CHECK(to_dlib(band_bottom + 0.5 * (fs - band_bottom)) == Approx(12.5));
    /* Failures do not move the usable range; a worse usable loss does. */
    CHECK(to_dlib(fs) == 15.0);
    CHECK(to_dlib(7.0) == 7.0);
    CHECK(to_dlib(fs) == 7.0 + 2.0 * 7.0);
}

TEST_CASE("GFS loss for dlib: the margin of degenerate usable ranges",
          "[HPO][gfs][dlib]")
{
    SECTION("losses of exactly 0: a margin of 1") {
        GfsLossForDlib to_dlib(kDblMax);
        CHECK(to_dlib(0.0) == 0.0);
        CHECK(to_dlib(kDblMax) == 2.0);
    }
    SECTION("negative losses: the spread") {
        GfsLossForDlib to_dlib(kDblMax);
        CHECK(to_dlib(-5.0) == -5.0);
        CHECK(to_dlib(-1.0) == -1.0);
        /* W = -1, B = -5: m = max(4, 1) = 4. */
        CHECK(to_dlib(kDblMax) == 7.0);
    }
    SECTION("usable losses beyond the ceiling are clamped, and so is the "
            "stand-in") {
        GfsLossForDlib to_dlib(kDblMax);
        CHECK(to_dlib(1e250) == kCeiling);
        CHECK(to_dlib(kDblMax) == kCeiling);
        CHECK(to_dlib(-1e250) == -kCeiling);
    }
}

TEST_CASE("GFS loss for dlib: dlib's global search reaches its budget when "
          "evaluations fail with the largest double", "[HPO][gfs][dlib]")
{
    /* Three dimensions: dlib takes its first trust-region step with five
     * samples. Handed the largest double for every failure, it crashed
     * there when those five had all failed. */
    const long dims = 3;
    const dlib::matrix<double, 0, 1> lo = dlib::zeros_matrix<double>(dims, 1);
    const dlib::matrix<double, 0, 1> hi = dlib::ones_matrix<double>(dims, 1);
    const std::vector<bool> is_int(dims, false);

    for (const std::size_t first_failures : {std::size_t{40}, std::size_t{8}}) {
        DYNAMIC_SECTION("the first " << first_failures
                        << " of 40 evaluations fail") {
            GfsLossForDlib to_dlib(kDblMax);
            std::size_t calls = 0;
            std::size_t usable = 0;
            const auto loss = [&](const dlib::matrix<double, 0, 1>& x) {
                ++calls;
                const double l = calls <= first_failures
                    ? kDblMax
                    : dlib::sum(dlib::squared(x - 0.3));
                if (!to_dlib.failed(l)) ++usable;
                return to_dlib(l);
            };
            dlib::thread_pool tp(1);
            const auto best = dlib::find_min_global(
                tp, loss, lo, hi, is_int, dlib::max_function_calls(40),
                dlib::FOREVER, 1e-8);
            CHECK(calls == 40u);
            CHECK(usable == 40u - std::min<std::size_t>(first_failures, 40));
            CHECK(std::isfinite(best.y));
            if (usable > 0) {
                CHECK(best.y < 1.0);
            } else {
                CHECK(best.y == kCeiling);
            }
        }
    }
}

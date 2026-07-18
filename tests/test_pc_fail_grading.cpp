/* Graded-FAILSCORE arithmetic tests (pc_fail_grading.h).
 * Properties under test are exactly the design invariants
 * (INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md 4.2/4.7):
 *   - every failure loss lives in [0.999*FS, FS)  (separation + filters);
 *   - within a band, less violation => strictly lower loss (direction);
 *   - every band-X value < every band-(X+1) value (repair-path ordering);
 *   - sigma is bounded, monotone, overflow/NaN-safe;
 *   - worst band wins in the per-eval accumulator;
 *   - infra failures are constant (directionless) at the top of band C. */

#include "catch2/catch.hpp"

#include <tadah/hpo/pc_fail_grading.h>

#include <cmath>
#include <limits>

using namespace tadah::hpo;

namespace {
constexpr double FS = 1.0e9;
constexpr double LO = 0.999 * FS;   /* failure-band floor */
}

TEST_CASE("fail_sigma: bounded, monotone, safe", "[fail_grading]") {
  CHECK(fail_sigma(0.0, 1.0) == 0.0);
  CHECK(fail_sigma(-5.0, 1.0) == 0.0);                       /* clamped   */
  CHECK(fail_sigma(std::nan(""), 1.0) == 0.0);               /* NaN-safe  */
  CHECK(fail_sigma(std::numeric_limits<double>::infinity(), 1.0) < 1.0);
  CHECK(fail_sigma(1.0, 1.0) == Approx(0.5));
  /* monotone */
  double prev = -1.0;
  for (double v : {0.0, 0.1, 1.0, 10.0, 1e6, 1e300}) {
    const double s = fail_sigma(v, 1.0);
    CHECK(s >= prev);
    CHECK(s < 1.0);
    prev = s;
  }
  /* bad v0 falls back sanely */
  CHECK(fail_sigma(1.0, -3.0) == Approx(0.5));
  CHECK(fail_sigma(1.0, std::nan("")) == Approx(0.5));
}

TEST_CASE("graded_failscore: band ranges and separation", "[fail_grading]") {
  const double W = 1e-3 * FS;
  for (auto band : {FailBand::A, FailBand::B, FailBand::C, FailBand::D}) {
    const double lo = graded_failscore(FS, band, 0.0, 1.0);
    const double hi = graded_failscore(
        FS, band, std::numeric_limits<double>::infinity(), 1.0);
    INFO("band " << fail_band_name(band));
    /* inside the reserved range, never touching FS */
    CHECK(lo >= LO);
    CHECK(hi < FS);
    /* band width is a quarter of the range */
    CHECK(hi - lo <= 0.25 * W + 1.0);
    /* still >= 0.999*FS => loss_is_finite-style filters see a failure */
    CHECK(lo >= 0.999 * FS);
  }
  /* strict cross-band ordering: top of X < bottom of X+1 */
  const double a_top = graded_failscore(FS, FailBand::A,
      std::numeric_limits<double>::infinity(), 1.0);
  const double b_lo  = graded_failscore(FS, FailBand::B, 0.0, 1.0);
  const double b_top = graded_failscore(FS, FailBand::B,
      std::numeric_limits<double>::infinity(), 1.0);
  const double c_lo  = graded_failscore(FS, FailBand::C, 0.0, 1.0);
  const double c_top = graded_failscore(FS, FailBand::C,
      std::numeric_limits<double>::infinity(), 1.0);
  const double d_lo  = graded_failscore(FS, FailBand::D, 0.0, 1.0);
  CHECK(a_top < b_lo);
  CHECK(b_top < c_lo);
  CHECK(c_top < d_lo);
}

TEST_CASE("graded_failscore: monotone within a band (direction)",
          "[fail_grading]") {
  double prev = 0.0;
  for (double v : {0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 100.0}) {
    const double l = graded_failscore(FS, FailBand::A, v, 1.0);
    CHECK(l > prev);   /* strictly increasing with violation */
    prev = l;
  }
  /* a proposal that reduces the violation is strictly downhill */
  CHECK(graded_failscore(FS, FailBand::A, 0.05, 1.0) <
        graded_failscore(FS, FailBand::A, 0.10, 1.0));
}

TEST_CASE("PcFailState: worst band wins; measures accumulate",
          "[fail_grading]") {
  PcFailState st;
  CHECK_FALSE(st.failed());

  st.record(FailBand::A, 0.1);
  CHECK(st.band() == FailBand::A);
  st.record(FailBand::B, 0.2);
  CHECK(st.band() == FailBand::B);
  st.record(FailBand::A, 0.3);              /* lower band cannot demote */
  CHECK(st.band() == FailBand::B);

  const PcFailInfo info = st.snapshot();
  CHECK(info.band == static_cast<int>(FailBand::B));
  CHECK(info.vA == Approx(0.4));
  CHECK(info.vB == Approx(0.2));
}

TEST_CASE("graded_eval_loss: band measures and fallbacks", "[fail_grading]") {
  PcFailInfo i;

  SECTION("band A graded by vA") {
    i.band = static_cast<int>(FailBand::A);
    i.vA = 0.5;
    const double l = graded_eval_loss(i, FS, 1.0);
    CHECK(l == Approx(graded_failscore(FS, FailBand::A, 0.5, 1.0)));
  }
  SECTION("band C graded by S when available, saturated when not") {
    i.band = static_cast<int>(FailBand::C);
    i.S = 2.0;
    const double with_s = graded_eval_loss(i, FS, 1.0);
    CHECK(with_s == Approx(graded_failscore(FS, FailBand::C, 2.0, 1.0)));
    i.S = -1.0;   /* no probe */
    const double without = graded_eval_loss(i, FS, 1.0);
    CHECK(without > with_s);          /* unknown grades worse than known */
    CHECK(without < FS);
  }
  SECTION("band D prefers the explicit vD (skip caps) over S") {
    i.band = static_cast<int>(FailBand::D);
    i.S = 100.0;
    i.vD = 0.25;
    CHECK(graded_eval_loss(i, FS, 1.0) ==
          Approx(graded_failscore(FS, FailBand::D, 0.25, 1.0)));
  }
  SECTION("infra failure = constant top of band C") {
    i.band = static_cast<int>(FailBand::C);
    i.infra = true;
    i.S = 0.001;   /* even a good S must NOT grade an infra failure */
    const double l = graded_eval_loss(i, FS, 1.0);
    CHECK(l == Approx(graded_failscore(
        FS, FailBand::C, std::numeric_limits<double>::infinity(), 1.0)));
  }
  SECTION("feasible eval returns FS only as a caller-bug guard") {
    i.band = 0;
    CHECK(graded_eval_loss(i, FS, 1.0) == FS);
  }
}

TEST_CASE("PcFailState: gate and S recording", "[fail_grading]") {
  PcFailState st;
  st.set_S(3.5);
  CHECK(st.S() == Approx(3.5));
  CHECK_FALSE(st.failed());          /* recording S alone is not a failure */
  st.set_gated();
  CHECK(st.gated());
  CHECK(st.band() == FailBand::D);
  const PcFailInfo info = st.snapshot();
  CHECK(info.gated);
  CHECK(info.S == Approx(3.5));

  /* garbage S is rejected */
  PcFailState st2;
  st2.set_S(std::nan(""));
  CHECK(st2.S() < 0.0);
  st2.set_S(-1.0);
  CHECK(st2.S() < 0.0);
}

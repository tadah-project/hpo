// FAILSCORE text in outvar slots.
//
// A failed sim has every outvar slot clamped to its FAILSCORE, written as
// TEXT (r.v_outvar holds the LAMMPS readback strings) and parsed back by
// try_parse_double / std::stod downstream -- including the failprobe check
// "S < failscore" that tells a real S(theta) from a failed probe. The old
// std::to_string (%f, 6 decimals) broke that round trip: 1.0000004 came
// back as 1.0 (BELOW the failscore, so a failed probe passed as a real S)
// and a failscore below 5e-7 came back as 0.

#include "catch2/catch.hpp"

#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/pc_fail_grading.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

using tadah::hpo::failprobe_value_is_real;
using tadah::hpo::failscore_token;
using tadah::hpo::try_parse_double;

namespace {

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof a) == 0;
}

}  // namespace

TEST_CASE("failscore_token parses back to the identical double",
          "[HPO][failscore][lossless]")
{
  for (double fs : {1e9, 1e8, 0.999e8, DBL_MAX, 0.1234567891, 1.0000004,
                    1e-7, 3.0, 0.1, 1.0 / 3.0, DBL_MIN}) {
    INFO("failscore = " << fs);
    const std::string tok = failscore_token(fs);
    double v = 0.0;
    REQUIRE(try_parse_double(tok, v));
    CHECK(same_bits(v, fs));
    CHECK(same_bits(std::stod(tok), fs));
  }
  /* Short, human-readable text where the value allows it. */
  CHECK(failscore_token(1e9) == "1000000000");
  CHECK(failscore_token(0.1234567891) == "0.1234567891");
  CHECK(failscore_token(DBL_MAX) == "1.7976931348623157e+308");

  /* What the legacy rendering did to the same values. */
  CHECK(std::stod(std::to_string(1.0000004)) == 1.0);
  CHECK(std::stod(std::to_string(1e-7)) == 0.0);
  CHECK_FALSE(same_bits(std::stod(std::to_string(0.1234567891)),
                        0.1234567891));
}

TEST_CASE("a failed failprobe (outvars clamped to FAILSCORE) never reads as "
          "a real S", "[HPO][failscore][failprobe][lossless]")
{
  for (double fs : {1.0000004, 1e9, DBL_MAX, 0.1234567891}) {
    INFO("failscore = " << fs);
    const double S = std::stod(failscore_token(fs));
    CHECK_FALSE(failprobe_value_is_real(S, fs));
    CHECK(failprobe_value_is_real(0.5 * fs, fs));
  }
  /* The legacy token made exactly this case pass as a real S = 1.0. */
  CHECK(failprobe_value_is_real(std::stod(std::to_string(1.0000004)),
                                1.0000004));

  const double fs = 10.0;
  CHECK(failprobe_value_is_real(0.0, fs));
  CHECK_FALSE(failprobe_value_is_real(-1e-300, fs));
  CHECK_FALSE(failprobe_value_is_real(
      std::numeric_limits<double>::quiet_NaN(), fs));
  CHECK_FALSE(failprobe_value_is_real(
      std::numeric_limits<double>::infinity(), fs));
  CHECK_FALSE(failprobe_value_is_real(fs, fs));
}

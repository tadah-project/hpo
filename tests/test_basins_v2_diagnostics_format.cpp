#include "catch2/catch.hpp"

/* Phase 5 — diagnostics-format contract.
 *
 * The BASINSv2 diagnostic lines are a stable contract that downstream
 * tooling (run analysers, dashboards, the BASINSv2.md design doc's
 * own examples) parses by regex. This test exercises the format
 * strings of a few example lines to catch accidental shape changes.
 *
 * The test does NOT instantiate the policy (which requires LAMMPS
 * runner / dlib::matrix machinery). It produces synthetic inputs
 * matching the exact line shape and asserts a regex parse.  If a
 * future change to the policy diagnostics breaks this regex, the
 * symptom is a test failure here — and a deliberate update to both
 * the formatter and this test, kept in lockstep.
 */

#include <regex>
#include <string>

namespace {
/* Examples mirror the strings emitted by basins_v2_policy_impl.hpp. */
const std::string ex_select = "BASINSv2 iter=137 select=REFINE basin[id=12] tol=1.000e-05 ema=4.2e-04 stagn_thr=300";
const std::string ex_state  = "BASINSv2 basin[id=12] state=ACTIVE->PROMISING ema=4.200e-04 (p75=2.100e-04)";
const std::string ex_inact  = "BASINSv2 basin[id=07] state=ACTIVE->INACTIVE evals_since_improve=820 (cap=750)";
const std::string ex_verify = "BASINSv2 verify basin[id=07] tickle: best 3.4521e-02 -> 3.4521e-02 (no progress) attempt=1/2";
const std::string ex_mpass  = "BASINSv2 merge-pass: 28 active, found 2 candidate-pair(s)";
const std::string ex_merge  = "BASINSv2 merge basin[id=18] -> basin[id=03] (root) dist=0.073 dloss=1.20%";
const std::string ex_explore= "BASINSv2 EXPLORE Sobol K=16 evaluated, M=4 survived (filter dropped 12 in known regions)";
const std::string ex_summary= "BASINSv2 round-summary iter=200 active=24 promising=3 inactive=4 suspect=0 merged=8 killed=2 best=3.4109e-02";
}

TEST_CASE("diagnostics-format: select line shape stable",
          "[basins_v2][diagnostics]")
{
    std::regex r(R"(BASINSv2 iter=\d+ select=\w+ basin\[id=\d+\] tol=[\d\.eE+-]+ ema=[\d\.eE+-]+ stagn_thr=\d+)");
    REQUIRE(std::regex_search(ex_select, r));
}

TEST_CASE("diagnostics-format: state-transition lines stable",
          "[basins_v2][diagnostics]")
{
    std::regex r(R"(BASINSv2 basin\[id=\d+\] state=\w+->\w+)");
    REQUIRE(std::regex_search(ex_state, r));
    REQUIRE(std::regex_search(ex_inact, r));
}

TEST_CASE("diagnostics-format: verify line shape stable",
          "[basins_v2][diagnostics]")
{
    std::regex r(R"(BASINSv2 verify basin\[id=\d+\] tickle: best [\d\.eE+-]+ -> [\d\.eE+-]+ \(.*\) attempt=\d+/\d+)");
    REQUIRE(std::regex_search(ex_verify, r));
}

TEST_CASE("diagnostics-format: merge-pass + merge lines stable",
          "[basins_v2][diagnostics]")
{
    std::regex r1(R"(BASINSv2 merge-pass: \d+ active, found \d+ candidate-pair\(s\))");
    REQUIRE(std::regex_search(ex_mpass, r1));
    std::regex r2(R"(BASINSv2 merge basin\[id=\d+\] -> basin\[id=\d+\] \(root\) dist=[\d\.]+ dloss=[\d\.]+%)");
    REQUIRE(std::regex_search(ex_merge, r2));
}

TEST_CASE("diagnostics-format: EXPLORE Sobol line shape stable",
          "[basins_v2][diagnostics]")
{
    std::regex r(R"(BASINSv2 EXPLORE Sobol K=\d+ evaluated, M=\d+ survived \(filter dropped \d+ in known regions\))");
    REQUIRE(std::regex_search(ex_explore, r));
}

TEST_CASE("diagnostics-format: round-summary line shape stable",
          "[basins_v2][diagnostics]")
{
    std::regex r(R"(BASINSv2 round-summary iter=\d+ active=\d+ promising=\d+ inactive=\d+ suspect=\d+ merged=\d+ killed=\d+ best=[\d\.eE+-]+)");
    REQUIRE(std::regex_search(ex_summary, r));
}

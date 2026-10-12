// test_restart_seeds.cpp
//
// The RESTART block's METHOD chain draws the starting points of every
// restart round. Its seeds -- METHOD WARM <file> and METHOD CONFIG, the
// user's starting points as opposed to sampled ones -- were read only when
// a stagnation drew them: `METHOD WARM missing.tadah` failed at the first
// stagnation, a WARM file whose EMBF0 or RHOFORM the run refuses likewise,
// and the density gates' SEED rule never saw them. HPOTargets resolves the
// RESTART chain once at startup (RestartSpec::seed_points), so these are
// startup errors, and validate_density_gates checks the restart seeds as it
// checks the INIT seeds.

#include "catch2/catch.hpp"

#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rho_gates.h>

#include <regex>
#include <sstream>
#include <string>

namespace {

using tadah::core::Context;
using tadah::hpo::HPOTargets;
using rhoform_test::make_ream_ctx;
using rhoform_test::ream_seed;
using rhoform_test::targets_error;
using rhoform_test::TempFile;

/* An OPTIMIZER block whose STRATEGY carries a RESTART block with the
 * METHOD lines `methods`. */
std::string restart_block(const std::string& methods) {
  return "OPTIMIZER\nSTRATEGY\n  TYPE PLAIN\n  LIB DLIB\n  ALGO BFGS\n"
         "  MAXEVAL 5\n  RESTART\n    AFTER_STAGNATION 5\n    MAX 1\n" +
         methods + "  ENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n";
}

}  // namespace

TEST_CASE("RESTART METHOD WARM: the seed file is read at startup",
          "[HPO][restart][restart_seeds]") {
  tadah::core::PeriodicTable::initialize();
  const std::string head = "OPTIM AMPGRIDMB (1) 0 1\nPC_ERMSE 0 1\n";

  SECTION("a missing file is a startup error") {
    CHECK_THAT(targets_error(make_ream_ctx("SIGNED"),
                             head + restart_block(
                                 "    METHOD WARM no_such_restart_seed.tadah\n")),
               Catch::Contains("WARM: cannot open seed file "
                               "'no_such_restart_seed.tadah'"));
  }
  SECTION("the WARM RHOFORM rule applies to a RESTART seed") {
    TempFile seed(ream_seed("0.4 0.3 0.1", "RHOFORM NONNEG\n"));
    CHECK_THAT(targets_error(make_ream_ctx("SIGNED"),
                             head + restart_block("    METHOD WARM " +
                                                  seed.path.string() + "\n")),
               Catch::Contains("WARM seed " + seed.path.string()) &&
                   Catch::Contains("RHOFORM"));
  }
  SECTION("the WARM EMBF0 rule applies to a RESTART seed (a WARNING)") {
    TempFile seed(ream_seed("0.4 0.3 0.1") + "EMBF0 FREE\n");
    rhoform_test::Journal j;
    CHECK(targets_error(make_ream_ctx("SIGNED"),
                        head + restart_block("    METHOD WARM " +
                                             seed.path.string() + "\n"))
              .empty());
    CHECK_THAT(j.finish(),
               Catch::Contains("WARM seed " + seed.path.string() +
                               " declares EMBF0 FREE; this run evaluates "
                               "EMBF0 ZERO"));
  }
  SECTION("a valid seed: resolved as a restart seed, sampled points are not") {
    TempFile seed(ream_seed("0.4 0.3 0.1"));
    Context ctx = make_ream_ctx("SIGNED");
    TempFile target(head + restart_block("    METHOD WARM " +
                                         seed.path.string() +
                                         "\n    METHOD RANDOM 2\n"));
    HPOTargets trg(ctx, target.path.string());
    const auto& rs = trg.spec.restart;
    REQUIRE(rs.seed_points.size() == 1u);
    REQUIRE(rs.seed_origins.size() == 1u);
    CHECK(rs.seed_points[0] == std::vector<double>{0.4});
    CHECK(rs.seed_origins[0].kind == tadah::hpo::InitOrigin::Kind::Warm);
  }
  SECTION("an inactive RESTART block draws nothing") {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile target(head +
        "OPTIMIZER\nSTRATEGY\n  TYPE PLAIN\n  LIB DLIB\n  ALGO BFGS\n"
        "  MAXEVAL 5\n  RESTART\n    AFTER_STAGNATION 0\n    MAX 1\n"
        "    METHOD WARM no_such_restart_seed.tadah\n  ENDRESTART\n"
        "ENDSTRATEGY\nENDOPTIMIZER\n");
    HPOTargets trg(ctx, target.path.string());
    CHECK(trg.spec.restart.seed_points.empty());
  }
}

TEST_CASE("RESTART seeds meet the density gates' SEED rule",
          "[HPO][restart][restart_seeds][rhogate]") {
  tadah::core::PeriodicTable::initialize();
  namespace dt = tadah::hpo::detail;
  TempFile bad(ream_seed("0.5 0.3 -0.2"));   // rho(r) dips below 0
  const std::string head = "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\n";
  auto gates_error = [&](const std::string& gate) {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile target(head + gate + restart_block("    METHOD WARM " +
                                                bad.path.string() + "\n"));
    HPOTargets trg(ctx, target.path.string());
    try {
      dt::validate_density_gates(ctx, trg);
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  CHECK_THAT(gates_error("RHOGATE 1e-5\n"),
             Catch::Contains("RHOGATE: seed RESTART WARM " + bad.path.string()) &&
                 Catch::Contains("fails the rho(r) sign gate"));
  rhoform_test::Journal j;
  CHECK(gates_error("RHOGATE 1e-5 SEED WARN\n").empty());
  CHECK_THAT(j.finish(),
             Catch::Contains("RHOGATE: seed RESTART WARM " + bad.path.string()) &&
                 Catch::Contains("SEED WARN: the run proceeds"));
}

namespace {

/* The 1-based number of the first line of `config` that is `text`, 0 when
 * none is. */
std::size_t line_number(const std::string& config, const std::string& text) {
  std::istringstream in(config);
  std::size_t n = 0;
  for (std::string l; std::getline(in, l);) {
    ++n;
    if (l == text) return n;
  }
  return 0;
}

/* Checks that `error` is the parser's form of a refusal of the line
 * `method_line` of `config` ("Error on line N: <what>\nLine: <text>") and
 * returns <what>. */
std::string refusal_at(const std::string& config, const std::string& error,
                       const std::string& method_line) {
  INFO("config:\n" << config << "error: " << error);
  const std::size_t n = line_number(config, method_line);
  REQUIRE(n > 0);
  std::smatch m;
  REQUIRE(std::regex_match(
      error, m, std::regex(R"(Error on line (\d+): ([\s\S]*)\nLine: ([^\n]*))")));
  CHECK(std::stoul(m[1]) == n);
  CHECK(m[3] == method_line);
  return m[2];
}

}  // namespace

/* An INIT or RESTART block's METHOD chain is resolved at startup, after
 * the file has been read: its refusals named no line, while the parser's
 * refusals of the same lines (an arity, K) name the METHOD line.
 * Pins: a WARM file that is missing (in the OPTIMIZER block's INIT block,
 * an EXPLORE block's INIT block and the RESTART block), a WARM seed without
 * the OPTIM parameter under WARM_ON_MISMATCH error, and a WARM seed whose
 * RHOFORM the run refuses are refused as "Error on line N: <what>\nLine:
 * <the METHOD line>", N and the text those of the METHOD WARM line, <what>
 * the refusal as it was ("WARM: cannot open seed file ...", "WARM: 1
 * parameter(s) not found in ...: AMPGRIDMB[1]", "WARM seed ... declares
 * RHOFORM NONNEG, but ...").
 * Before the fix: the refusals were <what> alone, no line. */
TEST_CASE("A METHOD chain's refusal names the line of its METHOD",
          "[HPO][init][restart][restart_seeds][refusal_lines]") {
  tadah::core::PeriodicTable::initialize();
  const std::string head = "OPTIM AMPGRIDMB (1) 0 1\nPC_ERMSE 0 1\n";
  const std::string plain =
      "  STRATEGY\n    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
      "    MAXEVAL 5\n  ENDSTRATEGY\n";
  const std::string missing = "WARM: cannot open seed file 'no_such_seed.tadah'";

  SECTION("the OPTIMIZER block's INIT block") {
    const std::string line = "    METHOD WARM no_such_seed.tadah";
    const std::string config = head + "OPTIMIZER\n  INIT\n" + line +
                               "\n  ENDINIT\n" + plain + "ENDOPTIMIZER\n";
    CHECK_THAT(refusal_at(config, targets_error(make_ream_ctx("SIGNED"), config),
                          line),
               Catch::StartsWith(missing));
  }
  SECTION("an EXPLORE block's INIT block") {
    const std::string line = "        METHOD WARM no_such_seed.tadah";
    const std::string config =
        head + "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB DLIB\n"
        "    ALGO BFGS\n    MAXEVAL 5\n    EXPLORE\n      INIT\n" + line +
        "\n      ENDINIT\n      LIB DLIB\n      ALGO GFS\n      MAXEVAL 5\n"
        "    ENDEXPLORE\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    CHECK_THAT(refusal_at(config, targets_error(make_ream_ctx("SIGNED"), config),
                          line),
               Catch::StartsWith(missing));
  }
  SECTION("the RESTART block") {
    const std::string line = "    METHOD WARM no_such_seed.tadah";
    const std::string config = head + restart_block(line + "\n");
    CHECK_THAT(refusal_at(config, targets_error(make_ream_ctx("SIGNED"), config),
                          line),
               Catch::StartsWith(missing));
  }
  SECTION("WARM_ON_MISMATCH error: a parameter the seed does not have") {
    TempFile seed("RCUT2B 5.5\n");
    const std::string line = "    METHOD WARM " + seed.path.string();
    const std::string config =
        head + "OPTIMIZER\n  INIT\n" + line +
        "\n    WARM_ON_MISMATCH error\n  ENDINIT\n" + plain + "ENDOPTIMIZER\n";
    CHECK(refusal_at(config, targets_error(make_ream_ctx("SIGNED"), config),
                     line) ==
          "WARM: 1 parameter(s) not found in " + seed.path.string() +
              ": AMPGRIDMB[1]");
  }
  SECTION("a WARM seed whose RHOFORM the run refuses") {
    TempFile seed(ream_seed("0.4 0.3 0.1", "RHOFORM NONNEG\n"));
    const std::string line = "    METHOD WARM " + seed.path.string();
    const std::string config = head + "OPTIMIZER\n  INIT\n" + line +
                               "\n  ENDINIT\n" + plain + "ENDOPTIMIZER\n";
    CHECK_THAT(refusal_at(config, targets_error(make_ream_ctx("SIGNED"), config),
                          line),
               Catch::StartsWith("WARM seed " + seed.path.string() +
                                 " declares RHOFORM NONNEG, but this run "
                                 "evaluates RHOFORM SIGNED"));
  }
}

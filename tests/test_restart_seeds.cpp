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

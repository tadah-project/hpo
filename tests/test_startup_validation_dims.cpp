// test_startup_validation_dims.cpp
//
// A configuration without any OPTIM line describes a zero-dimensional
// search space. The TADAH-native optimisers evaluate the INIT start point
// once in that case; NLOPT, DLIB and CERES cannot work with zero
// dimensions and only fail once the optimiser is entered, after every
// setup step has run. validate_hpo_setup must reject such a setup for
// those libraries at startup with a message that names OPTIM, and must
// keep accepting it for the native optimisers.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;

struct ScopedTempFile {
  std::filesystem::path path;
  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_svd_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~ScopedTempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};

Context make_minimal_ctx() {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1.0);
  c.add("DBFILE", "dummy.db");
  return c;
}

const std::string kLoss = "PC_ERMSE 0 1\n";
const std::string kOptim = "OPTIM WATOM (1) 0.1 5.0\n";

const std::string kNlopt =
    "OPTIMIZER\n  STRATEGY\n    THREADS 1\n    TYPE PLAIN\n    LIB NLOPT\n"
    "    ALGO LN_SBPLX\n    MAXEVAL 1\n    STEP 0.1\n  ENDSTRATEGY\nENDOPTIMIZER\n";
const std::string kDlib =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";
const std::string kTadah =
    "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
    "    MAXEVAL 1\n    SEED 1234\n  ENDSTRATEGY\nENDOPTIMIZER\n";

}  // namespace

TEST_CASE("startup validation: zero-dimensional search space",
          "[startup_validation][dims]")
{
  PeriodicTable::initialize();

  SECTION("LIB NLOPT without any OPTIM line is rejected, naming OPTIM") {
    ScopedTempFile tf(kLoss + kNlopt);
    Context ctx = make_minimal_ctx();
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.low.empty());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("OPTIM"));
  }
  SECTION("LIB DLIB without any OPTIM line is rejected, naming OPTIM") {
    ScopedTempFile tf(kLoss + kDlib);
    Context ctx = make_minimal_ctx();
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.low.empty());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("OPTIM"));
  }
  SECTION("LIB TADAH without any OPTIM line passes (single evaluation)") {
    ScopedTempFile tf(kLoss + kTadah);
    Context ctx = make_minimal_ctx();
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.low.empty());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
  SECTION("LIB NLOPT with one OPTIM line passes") {
    ScopedTempFile tf(kOptim + kLoss + kNlopt);
    Context ctx = make_minimal_ctx();
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.low.size() == 1);
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
  }
}

/* Config surface for the search-space / STEP policy through HPOTargets:
 * LOG_HP placement, PARAM GRID_POINTS, STEP override grammar, strict OPTIM
 * grammar, registry-driven STEP rejection, seed-in-box validation. */
#include "catch2/catch.hpp"
#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/scale_validation.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

struct ScopedTempFile {
  std::filesystem::path path;
  explicit ScopedTempFile(const std::string& body)
  : path(std::filesystem::temp_directory_path() /
         ("tadah_scale_parser_" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~ScopedTempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};

/* WATOM in pair-triple form: one optimisable weight (value 1.0). */
tadah::core::Context watom_ctx()
{
  tadah::core::Context ctx;
  ctx.add("ATOM", "C"); ctx.add("ATOM", "H");
  ctx.add("WATOM", "C"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
  return ctx;
}

const char* kOptim = "OPTIM WATOM (1) 0.1 5.0\n";

}  // namespace

TEST_CASE("LOG_HP is accepted at OPTIMIZER and STRATEGY level",
          "[HPO][scale][parser][log_hp]")
{
  SECTION("OUTER short form") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
LOG_HP 100
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.log_threshold.has_value());
    CHECK(*trg.spec.log_threshold == Approx(100.0));
    CHECK(trg.spec.parameter_labels.size() == 1);
    /* pair-triple WATOM: the weight sits at token position 2 */
    CHECK(trg.spec.parameter_labels[0] == "WATOM[2]");
  }
  SECTION("STRATEGY block") {
    ScopedTempFile in(std::string("OPTIM_SCALE NONE\n") + kOptim + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LOG_HP 10
    LIB DLIB
    ALGO BFGS
    MAXEVAL 5
  ENDSTRATEGY
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.log_threshold.has_value());
    CHECK(*trg.spec.log_threshold == Approx(10.0));
    /* 5.0/0.1 = 50 >= 10 and low > 0 -> the dim is log-scaled */
    REQUIRE(trg.spec.log_scaled.size() == 1);
    CHECK(trg.spec.log_scaled[0]);
  }
  SECTION("inside INNER is rejected") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB NLOPT
    ALGO G_MLSL_LDS
    MAXEVAL 5
    INNER
      LIB NLOPT
      ALGO LN_BOBYQA
      LOG_HP 10
    ENDINNER
  ENDSTRATEGY
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                        Catch::Contains("search-space property"));
  }
  SECTION("ratio below 1 is rejected") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
LOG_HP 0.5
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                        Catch::Contains("LOG_HP must be a finite ratio >= 1"));
  }
}

TEST_CASE("TADAH:GRID takes PARAM GRID_POINTS and rejects STEP",
          "[HPO][scale][parser][grid]")
{
  SECTION("GRID_POINTS parsed") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB TADAH
    ALGO GRID
    PARAM GRID_POINTS 10
  ENDSTRATEGY
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.param.count("GRID_POINTS") == 1);
    CHECK(trg.spec.param.at("GRID_POINTS") == Approx(10.0));
  }
  SECTION("STEP under GRID is rejected at startup") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB TADAH
    ALGO GRID
    STEP 10
  ENDSTRATEGY
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                        Catch::Contains("does not use STEP") &&
                        Catch::Contains("GRID_POINTS"));
  }
}

TEST_CASE("STEP override grammar: STEP <KEY>[(indices)] <frac>",
          "[HPO][scale][parser][step]")
{
  ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB NLOPT
    ALGO LN_SBPLX
    MAXEVAL 5
    STEP 0.1
    STEP WATOM(1) 0.05
    STEP WATOM 0.2
    STEP WATOM (1) 0.3
  ENDSTRATEGY
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  tadah::hpo::HPOTargets trg(ctx, in.path.string());
  REQUIRE(trg.spec.step.has_value());
  CHECK(*trg.spec.step == Approx(0.1));
  REQUIRE(trg.spec.step_overrides.size() == 3);
  CHECK(trg.spec.step_overrides[0].key == "WATOM");
  REQUIRE(trg.spec.step_overrides[0].indices.size() == 1);
  CHECK(trg.spec.step_overrides[0].indices[0] == 1);
  CHECK(trg.spec.step_overrides[0].frac == Approx(0.05));
  CHECK(trg.spec.step_overrides[1].indices.empty());
  CHECK(trg.spec.step_overrides[1].frac == Approx(0.2));
  /* "KEY (i)" with a space before the parenthesis is the same form */
  REQUIRE(trg.spec.step_overrides[2].indices.size() == 1);
  CHECK(trg.spec.step_overrides[2].indices[0] == 1);
  CHECK(trg.spec.step_overrides[2].frac == Approx(0.3));
}

TEST_CASE("STEP with a non-numeric value is a parse error",
          "[HPO][scale][parser][step]")
{
  ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB NLOPT
    ALGO LN_SBPLX
    STEP WATOM abc
  ENDSTRATEGY
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                      Catch::Contains("expects a numeric fraction"));
}

TEST_CASE("OPTIM grammar is strict: stray tokens are an error",
          "[HPO][scale][parser][optim]")
{
  ScopedTempFile in(R"(OPTIM WATOM (1) 0.1 5.0 junk
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                      Catch::Contains("unexpected token") &&
                      Catch::Contains("junk"));
}

TEST_CASE("STEP on a backend that ignores it is rejected at startup",
          "[HPO][scale][parser][step]")
{
  ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO GFS
    MAXEVAL 5
    STEP 0.1
  ENDSTRATEGY
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                      Catch::Contains("DLIB:GFS") &&
                      Catch::Contains("does not use STEP"));
}

TEST_CASE("CONFIG seed outside the OPTIM box is rejected at startup",
          "[HPO][scale][parser][seed]")
{
  /* WATOM value is 1.0; the box excludes it. */
  ScopedTempFile in(R"(OPTIM WATOM (1) 2.0 5.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  /* Parsing succeeds (the box is a valid config); the seed rule fires in
   * the startup validation pass (validate_hpo_setup -> run_scale_validation). */
  tadah::hpo::HPOTargets trg(ctx, in.path.string());
  REQUIRE_THROWS_WITH(tadah::hpo::run_scale_validation(trg.spec, true),
                      Catch::Contains("WATOM") &&
                      Catch::Contains("outside its OPTIM bounds"));
}

TEST_CASE("INNER unknown-key message lists the shared attribute vocabulary",
          "[HPO][scale][parser][inner]")
{
  ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB NLOPT
    ALGO G_MLSL_LDS
    MAXEVAL 5
    INNER
      LIB NLOPT
      ALGO LN_BOBYQA
      FOO 1
    ENDINNER
  ENDSTRATEGY
ENDOPTIMIZER
)");
  auto ctx = watom_ctx();
  REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                      Catch::Contains("inside INNER block") &&
                      Catch::Contains("GRAD_STRATEGY"));
}

TEST_CASE("OPTIM shape tags and OPTIM_SCALE build the search space",
          "[HPO][scale][parser][search_space]")
{
  SECTION("default: LIN, AUTO -> internal bounds are the unit box") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.space);
    CHECK(trg.spec.space->normalised());
    CHECK(trg.spec.low[0] == 0.0);
    CHECK(trg.spec.high[0] == 1.0);
    CHECK(trg.spec.space->at(0).lo == Approx(0.1));
    CHECK(trg.spec.space->at(0).hi == Approx(5.0));
    CHECK(trg.spec.space->at(0).x0 == Approx(1.0));      /* CONFIG seed */
    CHECK(trg.spec.space->at(0).user_index == 1);
    CHECK(trg.spec.space->describe_dim(0) == "LIN");
  }
  SECTION("OPTIM_SCALE NONE: internal bounds are the real bounds (legacy)") {
    ScopedTempFile in(std::string("OPTIM_SCALE NONE\n") + kOptim + R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK_FALSE(trg.spec.space->normalised());
    CHECK(trg.spec.low[0] == Approx(0.1));
    CHECK(trg.spec.high[0] == Approx(5.0));
  }
  SECTION("LOG tag: internal bounds are ln under NONE, seed maps through the space") {
    ScopedTempFile in(R"(OPTIM_SCALE NONE
OPTIM WATOM (1) 0.1 5.0 LOG
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.log_scaled == std::vector<bool>{true});
    CHECK(trg.spec.low[0] == Approx(std::log(0.1)));
    CHECK(trg.spec.space->describe_dim(0) == "LOG");
    REQUIRE(trg.spec.optim_init.size() == 1);
    CHECK(trg.spec.in(trg.spec.optim_init.front())[0] == Approx(0.0).margin(1e-15));
  }
  SECTION("ASINH tag with its scale") {
    ScopedTempFile in(R"(OPTIM_SCALE NONE
OPTIM WATOM (1) -5.0 5.0 ASINH 0.5
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK(trg.spec.space->at(0).tf.kind == tadah::hpo::ScaleKind::Asinh);
    CHECK(trg.spec.space->at(0).tf.c == Approx(0.5));
    CHECK_FALSE(trg.spec.log_scaled[0]);
    CHECK(trg.spec.low[0] == Approx(std::asinh(-10.0)));
  }
  SECTION("OPTIM_SCALE AUTO: unit box, seed at its fraction") {
    ScopedTempFile in(R"(OPTIM_SCALE AUTO
OPTIM WATOM (1) 0.0 4.0
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.space->normalised());
    CHECK(trg.spec.low[0] == 0.0);
    CHECK(trg.spec.high[0] == 1.0);
    CHECK(trg.spec.in(trg.spec.optim_init.front())[0] == Approx(0.25));
    CHECK(trg.spec.out(std::vector<double>{0.5})[0] == Approx(2.0));
  }
  SECTION("bad tag / bad ASINH argument / duplicate OPTIM_SCALE / LOG with low <= 0") {
    auto ctx = watom_ctx();
    {
      ScopedTempFile in("OPTIM WATOM (1) 0.1 5.0 CUBIC\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                          Catch::Contains("unexpected token") && Catch::Contains("ASINH <c>"));
    }
    {
      ScopedTempFile in("OPTIM WATOM (1) 0.1 5.0 ASINH\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                          Catch::Contains("ASINH needs one argument"));
    }
    {
      ScopedTempFile in("OPTIM_SCALE AUTO\nOPTIM_SCALE NONE\nOPTIM WATOM (1) 0.1 5.0\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                          Catch::Contains("OPTIM_SCALE given more than once"));
    }
    {
      ScopedTempFile in("OPTIM WATOM (1) -1.0 5.0 LOG\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()),
                          Catch::Contains("LOG scale requires low > 0"));
    }
  }
  SECTION("BASINSV2 stage specs are bound to the space") {
    ScopedTempFile in(std::string(kOptim) + R"(OPTIMIZER
  STRATEGY
    TYPE BASINSV2
    EXPLORE
      MAXEVAL 5
    ENDEXPLORE
    REFINE
      MAXEVAL 5
    ENDREFINE
  ENDSTRATEGY
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.basins_v2.explore);
    CHECK(trg.spec.basins_v2.explore->space == trg.spec.space);
    CHECK(trg.spec.basins_v2.refine->low.size() == 1);
  }
}

TEST_CASE("OPTIM AUTO boxes are derived from the CONFIG seed",
          "[HPO][scale][parser][autobox]")
{
  /* WATOM value is 1.0 and WATOM is not a positive-only key -> linear
   * rule by default: x0 +/- max(FRAC*|x0|, FLOOR, GROUP_FLOOR*max|x0|). */
  SECTION("default linear FRAC 0.5") {
    ScopedTempFile in(R"(OPTIM WATOM (1) AUTO
OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 5
ENDOPTIMIZER
)");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    REQUIRE(trg.spec.space->dim() == 1);
    CHECK(trg.spec.space->at(0).lo == Approx(0.5));
    CHECK(trg.spec.space->at(0).hi == Approx(1.5));
    CHECK(trg.spec.space->at(0).origin == tadah::hpo::BoxOrigin::AutoLinear);
  }
  SECTION("explicit FRAC + FLOOR") {
    ScopedTempFile in("OPTIM WATOM (1) AUTO FRAC 0.1 FLOOR 0.3\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK(trg.spec.space->at(0).lo == Approx(0.7));   /* max(0.1, 0.3) = 0.3 */
    CHECK(trg.spec.space->at(0).hi == Approx(1.3));
  }
  SECTION("FRAC 0 + FLOOR: constant absolute half-width (grid-centre boxes)") {
    ScopedTempFile in("OPTIM WATOM (1) AUTO FRAC 0 FLOOR 0.25\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK(trg.spec.space->at(0).lo == Approx(0.75));   /* max(0*1, 0.25) = 0.25 */
    CHECK(trg.spec.space->at(0).hi == Approx(1.25));
    CHECK(trg.spec.space->at(0).origin == tadah::hpo::BoxOrigin::AutoLinear);
  }
  SECTION("FRAC 0 without a floor is the zero-width error; negative FRAC is rejected") {
    auto ctx = watom_ctx();
    {
      ScopedTempFile in("OPTIM WATOM (1) AUTO FRAC 0\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()), Catch::Contains("zero-width box"));
    }
    {
      ScopedTempFile in("OPTIM WATOM (1) AUTO FRAC -0.1\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()), Catch::Contains("FRAC must be >= 0"));
    }
  }
  SECTION("geometric RATIO with a tag") {
    ScopedTempFile in("OPTIM WATOM (1) AUTO RATIO 4 LOG\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK(trg.spec.space->at(0).lo == Approx(0.25));
    CHECK(trg.spec.space->at(0).hi == Approx(4.0));
    CHECK(trg.spec.space->at(0).origin == tadah::hpo::BoxOrigin::AutoGeometric);
    CHECK(trg.spec.space->at(0).tf.kind == tadah::hpo::ScaleKind::Log);
  }
  SECTION("errors: RATIO and FRAC together, unknown option, RATIO <= 1, cutoff key") {
    auto ctx = watom_ctx();
    {
      ScopedTempFile in("OPTIM WATOM (1) AUTO RATIO 2 FRAC 0.5\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()), Catch::Contains("mutually exclusive"));
    }
    {
      ScopedTempFile in("OPTIM WATOM (1) AUTO WIDTH 2\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()), Catch::Contains("unexpected AUTO option"));
    }
    {
      ScopedTempFile in("OPTIM WATOM (1) AUTO RATIO 1\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, in.path.string()), Catch::Contains("RATIO must be > 1"));
    }
    {
      tadah::core::Context c2 = watom_ctx();
      c2.add("RCUT2B", 5.0);
      ScopedTempFile in("OPTIM RCUT2B (1) AUTO\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
      REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(c2, in.path.string()), Catch::Contains("cutoff keys"));
    }
  }
  SECTION("positive-only keys (writer table) default to the geometric rule") {
    CHECK(tadah::hpo::HPOTargets::optim_key_positive_only("SGRID2B"));
    CHECK(tadah::hpo::HPOTargets::optim_key_positive_only("SGRIDMB"));
    CHECK(tadah::hpo::HPOTargets::optim_key_positive_only("SEMBFUNC"));
    CHECK_FALSE(tadah::hpo::HPOTargets::optim_key_positive_only("CGRID2B"));
    CHECK_FALSE(tadah::hpo::HPOTargets::optim_key_positive_only("FIXWEIGHT"));
    CHECK_FALSE(tadah::hpo::HPOTargets::optim_key_positive_only("WATOM"));
  }
  SECTION("sidecar files are written; resolved block only for AUTO") {
    ScopedTempFile in("OPTIM WATOM (1) AUTO FRAC 0.2\nOPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
    auto ctx = watom_ctx();
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    const auto dir = std::filesystem::temp_directory_path() /
        ("tadah_ss_sidecar_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    trg.write_search_space_files(dir.string());
    REQUIRE(std::filesystem::exists(dir / "hpo_search_space.tsv"));
    REQUIRE(std::filesystem::exists(dir / "hpo_optim_resolved.hpo"));
    std::ifstream f(dir / "hpo_optim_resolved.hpo"); std::string line, all;
    while (std::getline(f, line)) all += line + "\n";
    CHECK(all.find("OPTIM WATOM (1) 0.8 1.2") != std::string::npos);
    std::filesystem::remove_all(dir);
  }
}

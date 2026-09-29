// test_hpo_rhoform.cpp
//
// RHOFORM (the DM_REAM density form) in tadah hpo:
//   - HPOTargets reads the mandatory key first and refuses OPTIM on
//     structure data (RHOFORM, TYPE*, RCTYPE*, KDEG2B, KDEGMB, KDEGEMB);
//   - NONNEG / MONOTONE boxes: the non-negative AMPGRIDMB AUTO class
//     (implicit GROUP_FLOOR, lo = max(0, x0 - d), "AUTO FRAC>=0", the RATIO
//     and zero-seed rules) and the explicit-box errors (AMPGRIDMB lo < 0,
//     WATOM lo <= 0, MONOTONE SGRIDMB lo <= 0);
//   - WARM: a seed whose RHOFORM is absent or differs is an ERROR when a
//     density key is OPTIM'd, else a WARNING;
//   - INIT provenance (OptimizerSpec::optim_init_origin);
//   - validate_hpo_setup validates a bare DM_REAM density, and the probes
//     that swallow construction errors rethrow models::DensityFormError;
//   - HPO_Host evaluates a NONNEG / MONOTONE candidate projected onto
//     AMPGRIDMB >= 0 (hidden host-level cases, run explicitly with
//     ./test_hpo "[rhoform_e2e]").

#include "catch2/catch.hpp"
#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/candidate_context.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <dlib/matrix.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::BoxOrigin;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using tadah::hpo::InitOrigin;
using tadah::hpo::validate_hpo_setup;
using tadah::models::DensityFormError;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using rhoform_test::ForceRebuildGuard;
using rhoform_test::Journal;
using rhoform_test::kOptimizer;
using rhoform_test::make_ream_ctx;
using rhoform_test::ream_seed;
using rhoform_test::ScratchDir;
using rhoform_test::targets_error;
using rhoform_test::TempFile;
using rhoform_test::unique_name;
using rhoform_test::write_ta_db;

std::vector<InitOrigin::Kind> kinds(const std::vector<InitOrigin>& o) {
  std::vector<InitOrigin::Kind> k;
  for (const auto& e : o) k.push_back(e.kind);
  return k;
}

}  // namespace

TEST_CASE("RHOFORM: HPOTargets reads the mandatory key first",
          "[hpo][rhoform]") {
  PeriodicTable::initialize();
  const std::string target = "OPTIM CEMBFUNC (1) -0.5 0.1\nPC_ERMSE 0 1\n" +
                             std::string(kOptimizer);

  SECTION("absent on a DM_REAM configuration: the migration message") {
    const std::string e = targets_error(make_ream_ctx(nullptr), target);
    CHECK_THAT(e, Catch::Contains("tadah hpo: DM_REAM: RHOFORM is required") &&
                      Catch::Contains("add 'RHOFORM SIGNED'"));
  }
  SECTION("an invalid value") {
    const std::string e = targets_error(make_ream_ctx("SQUARE"), target);
    CHECK_THAT(e, Catch::Contains("RHOFORM 'SQUARE' is not a density form"));
  }
  SECTION("every form, case-insensitive") {
    for (const char* f : {"SIGNED", "nonneg", "Monotone"}) {
      Context ctx = make_ream_ctx(f);
      TempFile tf(target);
      HPOTargets trg(ctx, tf.path.string());
      REQUIRE(trg.rho_form());
      CHECK(*trg.rho_form() == tadah::models::parse_rhoform(f));
    }
  }
  SECTION("no DM_REAM block: no key needed, none reported") {
    Context ctx;
    ctx.add("WATOM", 1.0);
    TempFile tf("OPTIM WATOM (1) 0.1 5.0\n" + std::string(kOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    CHECK_FALSE(trg.rho_form());
  }
}

TEST_CASE("RHOFORM: OPTIM on structure keys is refused",
          "[hpo][rhoform]") {
  PeriodicTable::initialize();
  for (const char* key : {"RHOFORM", "rhoform", "TYPEMB", "TYPE2B", "RCTYPE2B",
                          "RCTYPEMB", "KDEG2B", "kdeg2b", "KDEGMB", "KDEGEMB",
                          "kdegmb"}) {
    CAPTURE(key);
    const std::string e = targets_error(
        make_ream_ctx("SIGNED"),
        std::string("OPTIM ") + key + " (1) 0 1\nPC_ERMSE 0 1\n" + kOptimizer);
    /* at its line, as every hpotarget error */
    std::string up(key);
    for (auto& ch : up) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    CHECK_THAT(e, Catch::StartsWith("Error on line 1: OPTIM " + up +
                                    " is not a valid search target") &&
                      Catch::Contains("structure data") &&
                      Catch::Contains("-c potential/configuration"));
  }
  SECTION("refused before the key is looked up: nothing materialised") {
    Context ctx;
    ctx.add("WATOM", 1.0);
    TempFile tf("OPTIM RHOFORM (1) 0 1\n" + std::string(kOptimizer));
    REQUIRE_THROWS_WITH(HPOTargets(ctx, tf.path.string()),
                        Catch::Contains("OPTIM RHOFORM is not a valid"));
    CHECK_FALSE(ctx.has_stored("RHOFORM"));
  }
  SECTION("other keys are searchable") {
    for (const char* key : {"AMPGRIDMB", "CGRIDMB", "WATOM", "EMBF0X"})
      CHECK(HPOTargets::optim_structure_key_error(key).empty());
  }
  SECTION("mirrored in validate_hpo_setup") {
    for (const char* key : {"KDEGEMB", "KDEG2B"}) {
      CAPTURE(key);
      Context ctx = make_ream_ctx("SIGNED");
      ctx.add("DBFILE", "dummy.db");
      TempFile tf("OPTIM CEMBFUNC (1) -0.5 0.1\nPC_ERMSE 0 1\n" +
                  std::string(kOptimizer));
      HPOTargets trg(ctx, tf.path.string());
      trg.optimap[key];  // assembled without the parser
      REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                          Catch::Contains(std::string("OPTIM ") + key +
                                          " is not a valid"));
    }
  }
  SECTION("KDEG2B names the two-body knot degrees") {
    CHECK_THAT(HPOTargets::optim_structure_key_error("kdeg2b"),
               Catch::StartsWith("OPTIM KDEG2B is not a valid search "
                                 "target: KDEG2B lists the two-body knot "
                                 "degrees (3|5)"));
  }
}

TEST_CASE("RHOFORM NONNEG/MONOTONE: non-negative AMPGRIDMB AUTO boxes",
          "[hpo][rhoform][auto]") {
  PeriodicTable::initialize();
  auto boxes = [](const char* form, const char* amps, const std::string& optim) {
    Context ctx = make_ream_ctx(form, amps);
    TempFile tf(optim + "PC_ERMSE 0 1\n" + kOptimizer);
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.space());
    std::vector<tadah::hpo::DimSpec> d;
    for (std::size_t i = 0; i < trg.space()->dim(); ++i)
      d.push_back(trg.space()->at(i));
    return d;
  };

  SECTION("default FRAC 0.5 + implicit GROUP_FLOOR 1e-3, lo clamped at 0") {
    for (const char* form : {"NONNEG", "MONOTONE"}) {
      CAPTURE(form);
      const auto d = boxes(form, "0.0 0.3 0.1", "OPTIM AMPGRIDMB (1-3) AUTO\n");
      REQUIRE(d.size() == 3);
      /* seed 0: d = 1e-3 * max|x0| = 3e-4 -> [0, 3e-4] */
      CHECK(d[0].lo == 0.0);
      CHECK(d[0].hi == 1e-3 * 0.3);
      CHECK(d[1].lo == 0.3 - 0.5 * 0.3);
      CHECK(d[1].hi == 0.3 + 0.5 * 0.3);
      CHECK(d[2].lo == 0.1 - 0.5 * 0.1);
      for (const auto& e : d) {
        CHECK(e.origin == BoxOrigin::AutoLinearNonneg);
        CHECK(std::string(tadah::hpo::box_origin_name(e.origin)) == "AUTO FRAC>=0");
      }
    }
  }
  SECTION("x0 - d < 0 is clamped; SIGNED keeps the symmetric box") {
    const auto nn = boxes("NONNEG", "0.5 0.3 0.1", "OPTIM AMPGRIDMB (2) AUTO FRAC 2\n");
    REQUIRE(nn.size() == 1);
    CHECK(nn[0].lo == 0.0);
    CHECK(nn[0].hi == 0.3 + 2.0 * 0.3);
    const auto sg = boxes("SIGNED", "0.5 0.3 0.1", "OPTIM AMPGRIDMB (2) AUTO FRAC 2\n");
    REQUIRE(sg.size() == 1);
    CHECK(sg[0].lo == 0.3 - 2.0 * 0.3);
    CHECK(sg[0].origin == BoxOrigin::AutoLinear);
  }
  SECTION("SIGNED: unchanged rules (a 0 seed is a zero-width box)") {
    const std::string e = targets_error(
        make_ream_ctx("SIGNED", "0.0 0.3 0.1"),
        "OPTIM AMPGRIDMB (1) AUTO\nPC_ERMSE 0 1\n" + std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("AUTO FRAC gives a zero-width box"));
  }
  SECTION("case-insensitive key") {
    const auto d = boxes("NONNEG", "0.0 0.3 0.1", "OPTIM ampgridmb (1) AUTO FLOOR 0.2\n");
    REQUIRE(d.size() == 1);
    CHECK(d[0].lo == 0.0);
    CHECK(d[0].hi == 0.2);
    CHECK(d[0].origin == BoxOrigin::AutoLinearNonneg);
  }
  SECTION("every seed on the line 0 and no floor: an error with a hint") {
    const std::string e = targets_error(
        make_ream_ctx("NONNEG", "0.0 0.3 0.1"),
        "OPTIM AMPGRIDMB (1) AUTO\nPC_ERMSE 0 1\n" + std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("every seed on this line is 0") &&
                      Catch::Contains("RHOFORM NONNEG") &&
                      Catch::Contains("set FLOOR <a> for the box [0, a]"));
    /* FLOOR gives the absolute box [0, a] */
    const auto d = boxes("NONNEG", "0.0 0.3 0.1", "OPTIM AMPGRIDMB (1) AUTO FLOOR 0.2\n");
    CHECK(d[0].lo == 0.0);
    CHECK(d[0].hi == 0.2);
  }
  SECTION("an explicit GROUP_FLOOR replaces the implicit one") {
    const std::string e = targets_error(
        make_ream_ctx("NONNEG", "0.0 0.3 0.1"),
        "OPTIM AMPGRIDMB (1-3) AUTO GROUP_FLOOR 0\nPC_ERMSE 0 1\n" +
            std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("AUTO FRAC gives a zero-width box"));
  }
  SECTION("RATIO only when every seed on the line is > 0") {
    const std::string e = targets_error(
        make_ream_ctx("NONNEG", "0.0 0.3 0.1"),
        "OPTIM AMPGRIDMB (1-3) AUTO RATIO 2\nPC_ERMSE 0 1\n" +
            std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("AUTO RATIO under RHOFORM NONNEG needs every "
                                  "seed on the line > 0"));
    const auto d = boxes("NONNEG", "0.0 0.3 0.1", "OPTIM AMPGRIDMB (2-3) AUTO RATIO 2\n");
    REQUIRE(d.size() == 2);
    CHECK(d[0].lo == 0.15);
    CHECK(d[0].hi == 0.6);
    CHECK(d[0].origin == BoxOrigin::AutoGeometric);
  }
  SECTION("the clamp precedes the LOG check") {
    const std::string e = targets_error(
        make_ream_ctx("NONNEG"),
        "OPTIM AMPGRIDMB (2) AUTO FRAC 2 LOG\nPC_ERMSE 0 1\n" +
            std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("is not positive but the dim is tagged LOG"));
  }
  SECTION("a negative seed is refused") {
    const std::string e = targets_error(
        make_ream_ctx("MONOTONE", "-0.1 0.3 0.1"),
        "OPTIM AMPGRIDMB (1) AUTO\nPC_ERMSE 0 1\n" + std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("is negative, but RHOFORM MONOTONE requires "
                                  "every AMPGRIDMB amplitude >= 0"));
  }
  SECTION("hpo_optim_resolved.hpo pins [0, hi]") {
    Context ctx = make_ream_ctx("NONNEG", "0.0 0.3 0.1");
    TempFile tf("OPTIM AMPGRIDMB (1) AUTO FLOOR 0.25\nPC_ERMSE 0 1\n" +
                std::string(kOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    const fs::path dir = fs::temp_directory_path() / unique_name("tadah_rhoform_ss_");
    fs::create_directories(dir);
    trg.write_search_space_files(dir.string());
    std::ifstream is(dir / "hpo_optim_resolved.hpo");
    std::ostringstream oss;
    oss << is.rdbuf();
    std::error_code ec;
    fs::remove_all(dir, ec);
    CHECK_THAT(oss.str(), Catch::Contains("OPTIM AMPGRIDMB (1) 0 0.25   # AUTO FRAC>=0"));
  }
}

TEST_CASE("RHOFORM NONNEG/MONOTONE: explicit boxes that reach an invalid "
          "density are refused", "[hpo][rhoform]") {
  PeriodicTable::initialize();
  auto err = [](const char* form, const std::string& optim) {
    return targets_error(make_ream_ctx(form),
                         optim + "PC_ERMSE 0 1\n" + kOptimizer);
  };
  SECTION("AMPGRIDMB lo < 0 (lo = 0 is allowed)") {
    CHECK_THAT(err("NONNEG", "OPTIM AMPGRIDMB (1) -0.1 0.5\n"),
               Catch::Contains("is negative, but RHOFORM NONNEG requires every "
                               "AMPGRIDMB amplitude >= 0"));
    CHECK_THAT(err("MONOTONE", "OPTIM AMPGRIDMB (1) -0.1 0.5\n"),
               Catch::Contains("RHOFORM MONOTONE"));
    CHECK(err("NONNEG", "OPTIM AMPGRIDMB (1) 0 0.5\n").empty());
    CHECK(err("SIGNED", "OPTIM AMPGRIDMB (1) -0.1 0.5\n").empty());
  }
  SECTION("WATOM lo <= 0, explicit or AUTO") {
    CHECK_THAT(err("NONNEG", "OPTIM WATOM (1) 0 2\n"),
               Catch::Contains("density pair weight W > 0"));
    CHECK_THAT(err("MONOTONE", "OPTIM WATOM (1) AUTO FRAC 1\n"),
               Catch::Contains("density pair weight W > 0"));
    CHECK(err("NONNEG", "OPTIM WATOM (1) 0.5 2\n").empty());
    CHECK(err("SIGNED", "OPTIM WATOM (1) 0 2\n").empty());
  }
  SECTION("MONOTONE: SGRIDMB lo <= 0") {
    CHECK_THAT(err("MONOTONE", "OPTIM SGRIDMB (2) 0 4\n"),
               Catch::Contains("requires every SGRIDMB width > 0"));
    CHECK_THAT(err("MONOTONE", "OPTIM SGRIDMB (2) -1 4\n"),
               Catch::Contains("requires every SGRIDMB width > 0"));
    CHECK(err("MONOTONE", "OPTIM SGRIDMB (2) 1 4\n").empty());
    CHECK(err("NONNEG", "OPTIM SGRIDMB (2) 0 4\n").empty());
  }
}

TEST_CASE("RHOFORM NONNEG: the SGRIDMB box of a Gaussian density needs a "
          "low bound > 0", "[hpo][rhoform]") {
  // The Gaussian exp(-s (r - c)^2) grows without bound for s < 0, so
  // DM_REAM refuses s <= 0 under NONNEG (ReamDensity::validate); a box
  // reaching there passed startup and every such candidate then failed
  // when its model was built.
  PeriodicTable::initialize();
  auto gaussian = [](const char* form) {
    Context c = make_ream_ctx(form);
    std::vector<std::string> t(c.size("TYPEMB"));
    c.get("TYPEMB", t);
    t[1] = "1";  // the Gaussian basis code
    c.remove("TYPEMB");
    for (const auto& tok : t) c.add("TYPEMB", tok);
    return c;
  };
  auto err = [&](const char* form, const std::string& optim) {
    return targets_error(gaussian(form), optim + "PC_ERMSE 0 1\n" + kOptimizer);
  };
  CHECK_THAT(err("NONNEG", "OPTIM SGRIDMB (1) -0.5 2.0\n"),
             Catch::Contains("RHOFORM NONNEG requires every SGRIDMB width > 0 "
                             "for the Gaussian basis"));
  CHECK_THAT(err("NONNEG", "OPTIM SGRIDMB (2) 0 4\n"),
             Catch::Contains("for the Gaussian basis"));
  CHECK(err("NONNEG", "OPTIM SGRIDMB (1) 0.5 2.0\n").empty());
  CHECK_THAT(err("MONOTONE", "OPTIM SGRIDMB (1) -0.5 2.0\n"),
             Catch::Contains("requires every SGRIDMB width > 0"));
}

TEST_CASE("RHOFORM: WARM seed form vs the run's", "[hpo][rhoform][warm]") {
  PeriodicTable::initialize();
  auto warm_target = [](const fs::path& seed, const std::string& optim,
                        const char* policy = nullptr) {
    return "OPTIMIZER\nINIT\nSTRATEGY WARM " + seed.string() + "\n" +
           (policy ? std::string("WARM_ON_MISMATCH ") + policy + "\n" : "") +
           "ENDINIT\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n" + optim +
           "PC_ERMSE 0 1\n";
  };

  SECTION("density key OPTIM'd: absent, different or invalid form is an ERROR") {
    struct Case { const char* line; const char* what; };
    for (const Case& c : {Case{"", "does not declare RHOFORM"},
                          Case{"RHOFORM NONNEG\n", "declares RHOFORM NONNEG"},
                          Case{"RHOFORM FOO\n", "declares an invalid RHOFORM value 'FOO'"}}) {
      CAPTURE(c.line);
      for (const char* optim : {"OPTIM AMPGRIDMB (1) 0 1\n",
                                "OPTIM cgridmb (1) 2.5 3.0\n"}) {
        for (const char* policy : {"warn", "skip"}) {
          TempFile seed(ream_seed("0.4 0.3 0.1", c.line));
          const std::string e = targets_error(
              make_ream_ctx("SIGNED"), warm_target(seed.path, optim, policy));
          CHECK_THAT(e, Catch::Contains(c.what) &&
                            Catch::Contains("this run evaluates RHOFORM SIGNED") &&
                            Catch::Contains("a potential written before tadah "
                                            "1.6.0 is SIGNED: add 'RHOFORM "
                                            "SIGNED' to it"));
        }
      }
    }
  }
  SECTION("a positive run: a seed from before the forms cannot seed it") {
    TempFile seed(ream_seed("0.4 0.3 0.1", ""));
    const std::string e =
        targets_error(make_ream_ctx("NONNEG"),
                      warm_target(seed.path, "OPTIM AMPGRIDMB (1) 0 1\n"));
    CHECK_THAT(e, Catch::Contains("does not declare RHOFORM") &&
                      Catch::Contains("this run evaluates RHOFORM NONNEG") &&
                      Catch::Contains("a potential written before tadah 1.6.0 "
                                      "is SIGNED and cannot seed a NONNEG "
                                      "density; refit it by tadah refit with "
                                      "RHOFORM NONNEG"));
  }
  SECTION("no density key OPTIM'd: a WARNING, the seed is used") {
    Journal j;
    TempFile seed(ream_seed("0.4 0.3 0.1", "RHOFORM MONOTONE\n"));
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf(warm_target(seed.path, "OPTIM CEMBFUNC (1) -0.5 0.1\n"));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.optim_init.size() == 1);
    CHECK(trg.spec.optim_init.front()[0] == -0.2);
    const std::string txt = j.finish();
    CHECK_THAT(txt, Catch::Contains("WARNING") &&
                        Catch::Contains("declares RHOFORM MONOTONE; this run "
                                        "evaluates RHOFORM SIGNED") &&
                        Catch::Contains("the seed's density is not read"));
  }
  SECTION("matching form: silent") {
    Journal j;
    TempFile seed(ream_seed("0.4 0.3 0.1", "RHOFORM signed\n"));
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf(warm_target(seed.path, "OPTIM AMPGRIDMB (1) 0 1\n"));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.optim_init.size() == 1);
    CHECK(trg.spec.optim_init.front()[0] == 0.4);
    CHECK(j.finish().find("RHOFORM") == std::string::npos);
  }
}

TEST_CASE("INIT provenance: optim_init_origin parallel to optim_init",
          "[hpo][rhoform][init_strategy]") {
  PeriodicTable::initialize();
  using K = InitOrigin::Kind;
  const std::string optim = "OPTIM AMPGRIDMB (1-3) -1 1\nPC_ERMSE 0 1\n";
  auto run = [&](const std::string& init, const std::string& strategy =
                                                 "LIB DLIB\nALGO BFGS\nMAXEVAL 5\n") {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf("OPTIMIZER\n" + init + strategy + "ENDOPTIMIZER\n" + optim);
    auto trg = std::make_unique<HPOTargets>(ctx, tf.path.string());
    REQUIRE(trg->spec.optim_init_origin.size() == trg->spec.optim_init.size());
    return trg;
  };
  TempFile warm(ream_seed("0.4 0.3 0.1", "RHOFORM SIGNED\n"));
  TempFile warm_cfg(ream_seed("0.5 0.3 0.1", "RHOFORM SIGNED\n"));  // == CONFIG

  SECTION("no INIT block: CONFIG") {
    auto trg = run("");
    CHECK(kinds(trg->spec.optim_init_origin) == std::vector<K>{K::Config});
    CHECK(trg->spec.optim_init_origin.front().is_seed());
    CHECK(trg->spec.optim_init_origin.front().name() == "CONFIG");
  }
  SECTION("each point tagged with the strategy that produced it") {
    auto trg = run("INIT\nSTRATEGY WARM " + warm.path.string() +
                   "\nSTRATEGY CONFIG\nSTRATEGY LHS 3\nSTRATEGY SOBOL 2\n"
                   "STRATEGY RANDOM 1\nSEED 5\nENDINIT\n");
    const auto& o = trg->spec.optim_init_origin;
    CHECK(kinds(o) == std::vector<K>{K::Warm, K::Config, K::Lhs, K::Lhs, K::Lhs,
                                     K::Sobol, K::Sobol, K::Random});
    CHECK(o[0].file == warm.path.string());
    CHECK(o[0].name() == "WARM " + warm.path.string());
    CHECK(o[0].is_seed());
    CHECK_FALSE(o[2].is_seed());
    CHECK(trg->spec.optim_init[0][0] == 0.4);
    CHECK(trg->spec.optim_init[1][0] == 0.5);
  }
  SECTION("a de-duplicated point keeps the first origin") {
    auto a = run("INIT\nSTRATEGY CONFIG\nSTRATEGY WARM " + warm_cfg.path.string() +
                 "\nENDINIT\n");
    CHECK(kinds(a->spec.optim_init_origin) == std::vector<K>{K::Config});
    auto b = run("INIT\nSTRATEGY WARM " + warm_cfg.path.string() +
                 "\nSTRATEGY CONFIG\nENDINIT\n");
    CHECK(kinds(b->spec.optim_init_origin) == std::vector<K>{K::Warm});
  }
  SECTION("K_TOTAL truncates points and origins together") {
    auto trg = run("INIT\nSTRATEGY CONFIG\nSTRATEGY RANDOM 4\nK_TOTAL 3\n"
                   "SEED 9\nENDINIT\n");
    CHECK(kinds(trg->spec.optim_init_origin) ==
          std::vector<K>{K::Config, K::Random, K::Random});
  }
  SECTION("generate_tagged returns exactly generate()'s points") {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf("OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n" + optim);
    HPOTargets trg(ctx, tf.path.string());
    tadah::hpo::InitStrategySpec ispec;
    ispec.chain = {{"CONFIG", {}}, {"SOBOL", {"3"}}, {"RANDOM", {"2"}},
                   {"LHS", {"2"}}, {"CONFIG", {}}};
    auto strat = tadah::hpo::build_init_strategy(ispec);
    tadah::hpo::InitContext ic{trg.spec, trg, 77, &ctx};
    std::vector<InitOrigin> o;
    const auto tagged = strat->generate_tagged(ic, 6, o);
    CHECK(tagged == strat->generate(ic, 6));
    CHECK(kinds(o) == std::vector<K>{K::Config, K::Sobol, K::Sobol, K::Sobol,
                                     K::Random, K::Random});
  }
  SECTION("PRESTAGE: inherited with the OUTER's points, or its own chain") {
    const std::string prestage_head =
        "STRATEGY\nTYPE PRESTAGE\nLIB DLIB\nALGO BFGS\nMAXEVAL 8\nTOPK 2\n";
    auto inherited = run("INIT\nSTRATEGY WARM " + warm.path.string() +
                             "\nSTRATEGY CONFIG\nENDINIT\n",
                         prestage_head + "EXPLORE\nLIB DLIB\nALGO GFS\nMAXEVAL 20\n"
                                         "ENDEXPLORE\nENDSTRATEGY\n");
    REQUIRE(inherited->spec.prestage);
    CHECK(kinds(inherited->spec.prestage->optim_init_origin) ==
          std::vector<K>{K::Warm, K::Config});
    auto own = run("", prestage_head + "EXPLORE\nLIB DLIB\nALGO GFS\nMAXEVAL 20\n"
                                       "INIT\nSTRATEGY SOBOL 4\nSEED 3\nENDINIT\n"
                                       "ENDEXPLORE\nENDSTRATEGY\n");
    REQUIRE(own->spec.prestage);
    CHECK(kinds(own->spec.prestage->optim_init_origin) ==
          std::vector<K>(4, K::Sobol));
    CHECK(kinds(own->spec.optim_init_origin) == std::vector<K>{K::Config});
  }
}

TEST_CASE("RHOFORM: validate_hpo_setup validates the -c density, probes "
          "rethrow DensityFormError", "[hpo][rhoform][startup_validation]") {
  PeriodicTable::initialize();
  const std::string target = "OPTIM CEMBFUNC (1) -0.5 0.1\nPC_ERMSE 0 1\n" +
                             std::string(kOptimizer);
  auto validate_error = [&](Context ctx) {
    ctx.add("DBFILE", "dummy.db");
    TempFile tf(target);
    HPOTargets trg(ctx, tf.path.string());
    try {
      validate_hpo_setup(ctx, trg, true, true);
    } catch (const DensityFormError& e) {
      return std::string("DensityFormError: ") + e.what();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  SECTION("valid densities pass under every form") {
    for (const char* f : {"SIGNED", "NONNEG", "MONOTONE"})
      CHECK(validate_error(make_ream_ctx(f)).empty());
    CHECK(validate_error(make_ream_ctx("SIGNED", "-0.5 0.3 0.1")).empty());
  }
  SECTION("NONNEG: a negative amplitude") {
    CHECK_THAT(validate_error(make_ream_ctx("NONNEG", "0.5 -0.3 0.1")),
               Catch::StartsWith("DensityFormError: ") &&
                   Catch::Contains("requires every AMPGRIDMB entry to be finite "
                                   "and >= 0; entry 2 is -0.3"));
  }
  SECTION("NONNEG: WATOM 0") {
    Context ctx = make_ream_ctx("NONNEG");
    ctx.remove("WATOM");
    ctx.add("WATOM", 0.0);
    CHECK_THAT(validate_error(ctx),
               Catch::StartsWith("DensityFormError: ") &&
                   Catch::Contains("every density pair weight must be finite and > 0"));
  }
  SECTION("MONOTONE: a cutoff off the whitelist") {
    Context ctx = make_ream_ctx("MONOTONE");
    ctx.remove("RCTYPEMB");
    ctx.add("RCTYPEMB", "Cut_SinInv");
    CHECK_THAT(validate_error(ctx),
               Catch::StartsWith("DensityFormError: ") &&
                   Catch::Contains("RCTYPEMB must be one of") &&
                   Catch::Contains("got Cut_SinInv"));
  }
  SECTION("the key removed after parsing: the migration message") {
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("DBFILE", "dummy.db");
    TempFile tf(target);
    HPOTargets trg(ctx, tf.path.string());
    ctx.remove("RHOFORM");
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, true, true),
                        Catch::Contains("tadah hpo: DM_REAM: RHOFORM is required"));
  }
  SECTION("derive_weight_count rethrows (it returned 0 before)") {
    CHECK(tadah::hpo::derive_weight_count(make_ream_ctx("NONNEG")) == 9);
    REQUIRE_THROWS_AS(
        tadah::hpo::derive_weight_count(make_ream_ctx("NONNEG", "0.5 -0.3 0.1")),
        DensityFormError);
    /* other construction errors still give K = 0 */
    Context bad = make_ream_ctx("SIGNED");
    bad.remove("CEMBFUNC");
    CHECK(tadah::hpo::derive_weight_count(bad) == 0);
  }
  SECTION("derive_raw_embedding_f0 rethrows (it reported n/a before)") {
    REQUIRE_THROWS_AS(
        tadah::hpo::derive_raw_embedding_f0(make_ream_ctx("MONOTONE", "0.5 -0.3 0.1")),
        DensityFormError);
    CHECK(tadah::hpo::derive_raw_embedding_f0(make_ream_ctx("MONOTONE")).kind ==
          tadah::hpo::RawEmbeddingF0::Kind::Value);
  }
}

TEST_CASE("RHOFORM NONNEG/MONOTONE: one WARNING when an optimiser can leave "
          "the box", "[hpo][rhoform][startup_validation]") {
  PeriodicTable::initialize();
  auto journal = [](const char* form, const std::string& optimizer,
                    const char* optim = "OPTIM AMPGRIDMB (3) 0 0.5\n") {
    Journal j;
    Context ctx = make_ream_ctx(form);
    ctx.add("DBFILE", "dummy.db");
    TempFile tf(std::string(optim) + "PC_ERMSE 0 1\n" + optimizer);
    HPOTargets trg(ctx, tf.path.string());
    validate_hpo_setup(ctx, trg, true, true);
    return j.finish();
  };
  const std::string needle = "can evaluate AMPGRIDMB amplitudes outside the "
                             "non-negative OPTIM box";

  SECTION("an ignores_bounds optimiser") {
    const std::string txt = journal(
        "NONNEG", "OPTIMIZER\nLIB CERES\nALGO LBFGS\nMAXEVAL 5\nENDOPTIMIZER\n");
    CHECK_THAT(txt, Catch::Contains("RHOFORM NONNEG: the STRATEGY-level "
                                    "optimiser CERES:LBFGS (ignores the OPTIM "
                                    "bounds) " + needle) &&
                        Catch::Contains("projected onto AMPGRIDMB >= 0"));
  }
  SECTION("named by the phase that runs it (phase_specs)") {
    const std::string txt = journal(
        "NONNEG", "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB CERES\nALGO LBFGS\n"
                  "MAXEVAL 5\nTOPK 1\nEXPLORE\nLIB DLIB\nALGO GFS\nMAXEVAL 10\n"
                  "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n");
    CHECK_THAT(txt, Catch::Contains("RHOFORM NONNEG: the PRESTAGE refinement "
                                    "CERES:LBFGS (ignores the OPTIM bounds) " +
                                    needle));
    CHECK(txt.find("PRESTAGE exploration (EXPLORE block) DLIB:GFS") ==
          std::string::npos);   // GFS keeps to the box
  }
  SECTION("PARAM fd_kernel ceres on a gradient-based optimiser") {
    const std::string txt = journal(
        "MONOTONE", "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                    "PARAM fd_kernel ceres\nENDOPTIMIZER\n");
    CHECK_THAT(txt, Catch::Contains("RHOFORM MONOTONE: the STRATEGY-level "
                                    "optimiser DLIB:BFGS (PARAM fd_kernel ceres") &&
                        Catch::Contains(needle));
  }
  SECTION("PARAM fd_kernel ceres on an MLSL whose local search takes gradients") {
    /* G_MLSL_LDS itself is derivative-free, but NLopt hands its local
     * search the gradient the MLSL objective computes (inner_run_spec,
     * obtains_gradients). */
    const std::string txt = journal(
        "NONNEG", "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO G_MLSL_LDS\n"
                  "MAXEVAL 50\nPARAM fd_kernel ceres\nINNER\nLIB NLOPT\n"
                  "ALGO LD_LBFGS\nMAXEVAL 10\nENDINNER\nENDSTRATEGY\n"
                  "ENDOPTIMIZER\n");
    CHECK_THAT(txt, Catch::Contains("RHOFORM NONNEG: the STRATEGY-level "
                                    "optimiser NLOPT:G_MLSL_LDS (PARAM fd_kernel "
                                    "ceres") &&
                        Catch::Contains(needle));
  }
  SECTION("none: bounded optimiser, SIGNED, or no OPTIM AMPGRIDMB") {
    CHECK(journal("NONNEG", kOptimizer).find(needle) == std::string::npos);
    CHECK(journal("SIGNED", "OPTIMIZER\nLIB CERES\nALGO LBFGS\nMAXEVAL 5\n"
                            "ENDOPTIMIZER\n").find(needle) == std::string::npos);
    CHECK(journal("NONNEG", "OPTIMIZER\nLIB CERES\nALGO LBFGS\nMAXEVAL 5\n"
                            "ENDOPTIMIZER\n",
                  "OPTIM CEMBFUNC (1) -0.5 0.1\n").find(needle) == std::string::npos);
  }
}

TEST_CASE("update_candidate_context is HPO_Host::update_context",
          "[hpo][rhoform]") {
  PeriodicTable::initialize();
  Context ctx = make_ream_ctx("SIGNED");
  TempFile tf("OPTIM AMPGRIDMB (1,3) -1 1\nOPTIM CEMBFUNC (2) -1 1\n"
              "PC_ERMSE 0 1\n" + std::string(kOptimizer));
  HPOTargets trg(ctx, tf.path.string());
  ParamVec p(3);
  p(0) = 0.25; p(1) = -0.125; p(2) = 0.75;   // AMPGRIDMB 1, 3; CEMBFUNC 2
  Context c = ctx;
  tadah::hpo::detail::update_candidate_context(c, trg, p, false, false);
  std::vector<double> a(c.size("AMPGRIDMB")), e(c.size("CEMBFUNC"));
  c.get("AMPGRIDMB", a);
  c.get("CEMBFUNC", e);
  CHECK(a == std::vector<double>{0.25, 0.3, -0.125});
  CHECK(e == std::vector<double>{-0.2, 0.75, 0.8, 1.3, 1.8});
}

/* ------------------------------------------------------------------------
 * Host-level cases (thread-local trainer cache, like the other HPO_Host
 * e2e cases): ./test_hpo "[rhoform_e2e]"
 * ------------------------------------------------------------------------ */
namespace {

/* Rows of a per-eval table (loss.tadah, params.tadah): the numeric columns
 * of every non-comment line. */
std::vector<std::vector<double>> table_rows(const fs::path& path) {
  std::vector<std::vector<double>> rows;
  std::ifstream is(path);
  std::string line;
  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::vector<double> r;
    double v;
    while (ls >> v) r.push_back(v);
    rows.push_back(r);
  }
  return rows;
}

double pot_amplitude(const tadah::mlip::Trainer* tr, std::size_t i) {
  auto pot = const_cast<tadah::mlip::Trainer*>(tr)->get_param_file();
  std::vector<double> a(pot.size("AMPGRIDMB"));
  pot.get("AMPGRIDMB", a);
  return a.at(i);
}

/* AMPGRIDMB entry `i` of a potential file (best_pot.tadah). */
double file_amplitude(const fs::path& path, std::size_t i) {
  std::ifstream is(path);
  std::string line;
  while (std::getline(is, line)) {
    std::istringstream ls(line);
    std::string key;
    if (!(ls >> key) || key != "AMPGRIDMB") continue;
    std::vector<double> a;
    double v;
    while (ls >> v) a.push_back(v);
    return a.at(i);
  }
  FAIL("no AMPGRIDMB line in " << path);
  return 0.0;
}

}  // namespace

TEST_CASE("HPO_Host: NONNEG/MONOTONE candidates are evaluated projected onto "
          "AMPGRIDMB >= 0", "[rhoform_e2e][rhoform_projection_e2e][.]") {
  PeriodicTable::initialize();
  ScratchDir sd("tadah_rhoform_e2e_");
  const fs::path val_db  = sd.dir / "val.db";
  const fs::path val_cfg = sd.dir / "config.val";
  write_ta_db(val_db);
  { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }
  const double fs_score = 1.0e9;
  auto target = [&](const std::string& optimizer) {
    const fs::path p = sd.dir / "config.hpo";
    std::ofstream os(p);
    os << optimizer << "LOSS L2\nPC_EMAE 0 1\nFAILSCORE 1e9\n"
       << "OPTIM AMPGRIDMB (3) 0 0.5\n";
    return p;
  };
  const std::string one_eval =
      "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n";

  SECTION("an out-of-box probe is evaluated and recorded at its projection") {
    for (const char* form : {"NONNEG", "MONOTONE"}) {
      CAPTURE(form);
      Journal j;
      Host::reset_globals_for_test();
      ForceRebuildGuard rebuild;
      const fs::path hpo_cfg = target(one_eval);
      Context ctx = make_ream_ctx(form, "0.5 0.3 0.0");
      ctx.add("HPOTARGET", hpo_cfg.string());
      HPOTargets trg(ctx, hpo_cfg.string());
      REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
      double loss_probe = 0.0, loss_zero = 0.0, loss_in = 0.0;
      {
        Host host(ctx, trg, val_cfg.string());
        ParamVec p(1);
        p(0) = -1e-6;   // a central-difference probe at the lower bound
        REQUIRE_NOTHROW(loss_probe = host(p));
        CHECK(std::isfinite(loss_probe));
        CHECK(loss_probe < fs_score);
        CHECK(HPOLog::instance().form_projections() == 1);
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        CHECK(pot_amplitude(tr, 2) == 0.0);
        CHECK(file_amplitude("best_pot.tadah", 2) == 0.0);   // the first best

        p(0) = 0.0;
        REQUIRE_NOTHROW(loss_zero = host(p));
        CHECK(loss_zero == loss_probe);   // the same point, bit for bit
        p(0) = 0.25;
        REQUIRE_NOTHROW(loss_in = host(p));
        CHECK(pot_amplitude(Host::tls_trainer_for_test(), 2) == 0.25);
        CHECK(HPOLog::instance().form_projections() == 1);   // in-box: no-op
        host.flush_remaining();
      }
      const auto rows = table_rows("params.tadah");
      REQUIRE(rows.size() == 3);
      CHECK(rows[0].back() == 0.0);   // the recorded point is the projection
      CHECK(rows[2].back() == 0.25);
      const std::string txt = j.finish();
      CHECK_THAT(txt, Catch::Contains(std::string("eval 1: RHOFORM ") + form +
                                      ": AMPGRIDMB[3] = "
                                      "-1.0000e-06 below 0 is evaluated and "
                                      "recorded as 0") &&
                          Catch::Contains("Positive-form projection: 1 evaluation(s)"));
      /* the best sits on the form's zero lower bound: a hard bound, no
       * widen-that-side advisory */
      REQUIRE(loss_probe <= loss_in);
      CHECK_THAT(txt, Catch::Contains("LO*   0.0000e+00    0.0000e+00") &&
                          Catch::Contains("(* a hard lower bound: RHOFORM "
                                          "NONNEG/MONOTONE require AMPGRIDMB >= 0"));
      CHECK(txt.find("advisory") == std::string::npos);
      Host::reset_globals_for_test();
    }
  }

  SECTION("SIGNED: no projection, the negative amplitude is evaluated") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path hpo_cfg = sd.dir / "config_signed.hpo";
    {
      std::ofstream os(hpo_cfg);
      os << one_eval << "LOSS L2\nPC_EMAE 0 1\nFAILSCORE 1e9\n"
         << "OPTIM AMPGRIDMB (3) -0.5 0.5\n";
    }
    Context ctx = make_ream_ctx("SIGNED", "0.5 0.3 0.0");
    ctx.add("HPOTARGET", hpo_cfg.string());
    HPOTargets trg(ctx, hpo_cfg.string());
    Host host(ctx, trg, val_cfg.string());
    ParamVec p(1);
    p(0) = -1e-6;
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = host(p));
    CHECK(std::isfinite(loss));
    CHECK(pot_amplitude(Host::tls_trainer_for_test(), 2) == -1e-6);
    CHECK(HPOLog::instance().form_projections() == 0);
    CHECK(j.finish().find("Positive-form projection") == std::string::npos);
    Host::reset_globals_for_test();
  }

  SECTION("a gradient-based optimiser from a zero amplitude: no FAIL_SCORE") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path hpo_cfg = target(
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 8\nENDOPTIMIZER\n");
    Context ctx = make_ream_ctx("NONNEG", "0.5 0.3 0.0");
    ctx.add("HPOTARGET", hpo_cfg.string());
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
    REQUIRE(trg.spec.optim_init.size() == 1);
    REQUIRE(trg.spec.optim_init.front()[0] == 0.0);   // the seed is on the bound
    {
      Host host(ctx, trg, val_cfg.string());
      REQUIRE_NOTHROW(tadah::hpo::run_inner<ParamVec, ::LammpsRunner>(trg.spec, host));
      host.flush_remaining();
    }
    CHECK(HPOLog::instance().form_projections() >= 1);
    const auto loss_rows = table_rows("loss.tadah");
    const auto par_rows  = table_rows("params.tadah");
    REQUIRE(loss_rows.size() >= 3);
    REQUIRE(par_rows.size() == loss_rows.size());
    for (std::size_t i = 0; i < loss_rows.size(); ++i) {
      CAPTURE(i);
      CHECK(loss_rows[i].back() < fs_score);   // never the failure sentinel
      CHECK(par_rows[i].back() >= 0.0);        // never a negative amplitude
    }
    const std::string txt = j.finish();
    CHECK(txt.find("eval-thread caught throw") == std::string::npos);
    CHECK_THAT(txt, Catch::Contains("Positive-form projection") &&
                        Catch::Contains("[ok] No NaN/Inf clampings were recorded."));
    Host::reset_globals_for_test();
  }
}

TEST_CASE("HPO_Host: the all-weights-fixed probe rethrows DensityFormError",
          "[rhoform_e2e][.]") {
  PeriodicTable::initialize();
  Host::reset_globals_for_test();
  ScratchDir sd("tadah_rhoform_e2e_");
  const fs::path train_db = sd.dir / "train.db";
  const fs::path val_db   = sd.dir / "val.db";
  const fs::path val_cfg  = sd.dir / "config.val";
  const fs::path hpo_cfg  = sd.dir / "config.hpo";
  write_ta_db(train_db);
  write_ta_db(val_db);
  { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }
  {
    std::ofstream os(hpo_cfg);
    os << kOptimizer << "LOSS L2\nPC_EMAE 0 1\nOPTIM CEMBFUNC (1) -0.5 0.1\n";
  }
  /* A NONNEG -c potential with a negative amplitude, built without
   * validate_hpo_setup: the host's FIXINDEX probe is the first to construct
   * DM_REAM, and must not read the failure as "not all weights fixed". */
  Context ctx = make_ream_ctx("NONNEG", "0.5 -0.3 0.1");
  ctx.add("DBFILE", train_db.string());
  ctx.add("HPOTARGET", hpo_cfg.string());
  HPOTargets trg(ctx, hpo_cfg.string());
  REQUIRE_THROWS_AS(Host(ctx, trg, val_cfg.string()), DensityFormError);
  Host::reset_globals_for_test();
}

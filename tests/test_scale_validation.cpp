/* scale_validation -- registry-driven STEP policy, start-point checks and
 * the search-space width audit, on hand-built OptimizerSpecs (no host). */
#include "catch2/catch.hpp"
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_log.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;

namespace {

OptimizerSpec make_spec(const std::string& lib, const std::string& algo,
                        std::vector<double> low = {0.0, 0.0},
                        std::vector<double> high = {1.0, 1.0})
{
    OptimizerSpec sp;
    sp.lib = lib;
    sp.algo = algo;
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    std::vector<double> mid(sp.low.size());
    for (std::size_t i = 0; i < mid.size(); ++i)
        mid[i] = 0.5 * (sp.low[i] + sp.high[i]);
    sp.optim_init = {mid};
    return sp;
}

std::filesystem::path scratch_log_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_scale_validation_" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch().count()) + ".log");
}

std::string slurp(const std::filesystem::path& p)
{
    std::ifstream f(p);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

/* HPOLog is a process-wide singleton whose open() cannot re-open an
 * already-open journal (the ofstream keeps the old file), so the fixture
 * closes any journal a previous test left behind, opens its own, and
 * closes it again on exit (finish() writes the trailer and closes). */
struct JournalCapture {
    std::filesystem::path path = scratch_log_path();
    JournalCapture()
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        REQUIRE(log.is_open());
    }
    std::string text() const { return slurp(path); }
    ~JournalCapture()
    {
        HPOLog::instance().finish("test-end");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

}  // namespace

TEST_CASE("scale_validation: STEP is rejected on every non-consuming backend",
          "[HPO][scale][step]")
{
    struct Row { const char* lib; const char* algo; const char* hint; };
    const std::vector<Row> rows = {
        {"DLIB",  "GFS",        "trust region"},
        {"TADAH", "RANDOM",     "remove STEP"},
        {"TADAH", "GRID",       "GRID_POINTS"},
        {"NLOPT", "GN_CRS2_LM", "never reads an initial step"},
        {"NLOPT", "LD_LBFGS",   "never reads an initial step"},
        {"TADAH", "ANNEAL",     "STEP_FRAC"},
        {"TADAH", "ANNEAL_PT",  "STEP_FRAC"},
        {"ASA",   "ASA",        "temperatures"},
        {"DLIB",  "BFGS",       "fd_step"},
        {"CERES", "LBFGS",      "max_line_search_step_size"},
    };
    for (const auto& r : rows) {
        INFO(r.lib << ":" << r.algo);
        auto sp = make_spec(r.lib, r.algo);
        sp.step = 0.1;
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                            Catch::Contains("does not use STEP") &&
                            Catch::Contains(r.hint));
    }
}

TEST_CASE("scale_validation: STEP override alone is rejected on non-consumers",
          "[HPO][scale][step]")
{
    auto sp = make_spec("DLIB", "GFS");
    sp.step_overrides.push_back({"RCUT2B", {}, 0.1});
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                        Catch::Contains("does not use STEP"));
}

TEST_CASE("scale_validation: STEP accepted on NLopt consumers",
          "[HPO][scale][step]")
{
    for (const char* algo : {"LN_SBPLX", "LN_NELDERMEAD", "LN_BOBYQA",
                             "LN_COBYLA", "LN_NEWUOA_BOUND", "LD_MMA"}) {
        INFO(algo);
        auto sp = make_spec("NLOPT", algo);
        sp.step = 0.1;
        sp.step_overrides.push_back({"RCUT2B", {1}, 0.05});
        REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER"));
    }
}

TEST_CASE("scale_validation: STEP value must be > 0 and finite",
          "[HPO][scale][step]")
{
    auto sp = make_spec("NLOPT", "LN_SBPLX");
    for (double v : {0.0, -0.1, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
        sp.step = v;
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                            Catch::Contains("STEP must be > 0 and finite"));
    }
    sp.step = 0.1;
    sp.step_overrides.push_back({"RCUT2B", {}, 0.0});
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                        Catch::Contains("STEP must be > 0 and finite"));
}

TEST_CASE("scale_validation: OUTER STEP overriding an INNER STEP is journaled",
          "[HPO][scale][step][journal]")
{
    JournalCapture j;
    auto sp = make_spec("NLOPT", "G_MLSL_LDS");
    sp.step = 0.1;
    sp.inner = std::make_unique<OptimizerSpec>(make_spec("NLOPT", "LN_BOBYQA"));
    sp.inner->step = 0.05;
    REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER"));
    const std::string t = j.text();
    CHECK(t.find("WARNING") != std::string::npos);
    CHECK(t.find("overrides the INNER") != std::string::npos);
}

TEST_CASE("scale_validation: bounds-ignoring backends warn at startup",
          "[HPO][scale][bounds][journal]")
{
    JournalCapture j;
    auto sp = make_spec("NLOPT", "LN_NEWUOA");
    REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER"));
    CHECK(j.text().find("called without bounds") != std::string::npos);

    auto ce = make_spec("CERES", "LBFGS");
    REQUIRE_NOTHROW(validate_optimizer_scaling(ce, "OUTER"));
    CHECK(j.text().find("no box constraints") != std::string::npos);
}

TEST_CASE("scale_validation: start points must lie inside the OPTIM box",
          "[HPO][scale][seed]")
{
    auto sp = make_spec("NLOPT", "LN_BOBYQA", {0.0, 1.0}, {1.0, 2.0});
    sp.parameter_labels = {"RCUT2B", "SGRIDMB[1]"};

    SECTION("inside") {
        sp.optim_init = {{0.5, 1.5}, {0.0, 2.0}};   /* bounds inclusive */
        REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER"));
    }
    SECTION("outside names the dimension and the real bounds") {
        sp.optim_init = {{0.5, 2.5}};
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                            Catch::Contains("SGRIDMB[1]") &&
                            Catch::Contains("outside its OPTIM bounds") &&
                            Catch::Contains("[1, 2]"));
    }
    SECTION("size mismatch") {
        sp.optim_init = {{0.5}};
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                            Catch::Contains("expected 2"));
    }
    SECTION("non-finite") {
        sp.optim_init = {{0.5, std::numeric_limits<double>::quiet_NaN()}};
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                            Catch::Contains("not finite"));
    }
    SECTION("run path skips seed checks") {
        sp.optim_init = {{0.5, 2.5}};
        REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "run", false));
    }
}

TEST_CASE("scale_validation: log-scaled dims need positive seeds in real space",
          "[HPO][scale][seed][log]")
{
    /* One log-scaled dim with real box [0.1, 10]: internal bounds are ln. */
    auto sp = make_spec("NLOPT", "LN_BOBYQA", {std::log(0.1)}, {std::log(10.0)});
    sp.log_scaled = {true};
    sp.parameter_labels = {"SGRID2B"};

    sp.optim_init = {{1.0}};        /* ln(1) = 0 inside [ln 0.1, ln 10] */
    REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER"));

    sp.optim_init = {{0.0}};
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                        Catch::Contains("must be > 0 on a log-scaled"));

    sp.optim_init = {{20.0}};       /* positive but outside the real box */
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                        Catch::Contains("SGRID2B") &&
                        Catch::Contains("outside its OPTIM bounds"));
}

TEST_CASE("scale_validation: STRATEGY-level STEP under BASINS has no consumer",
          "[HPO][scale][step][basins]")
{
    OptimizerSpec root = make_spec("NOT SET", "NOT SET");
    root.strategy_type = StrategyType::BASINS;
    basins_spec_ensure_defaults(root.basins);
    root.basins.explore->low = root.low;   root.basins.explore->high = root.high;
    root.basins.refine->low  = root.low;   root.basins.refine->high  = root.high;
    root.step = 0.1;
    REQUIRE_THROWS_WITH(run_scale_validation(root),
                        Catch::Contains("no consumer under TYPE BASINS"));

    root.step.reset();
    root.basins.refine->step = 0.1;   /* REFINE LN_BOBYQA consumes it */
    REQUIRE_NOTHROW(run_scale_validation(root));
}

TEST_CASE("scale_validation: width audit journals ratio and pinned dims",
          "[HPO][scale][journal]")
{
    JournalCapture j;
    auto sp = make_spec("NLOPT", "LN_SBPLX", {0.0, 0.0, 1.0}, {1.0, 1e-3, 1.0 + 1e-15});
    sp.parameter_labels = {"CGRID2B[1]", "AMPGRIDMB[5]", "SEMBFUNC[13]"};
    audit_search_space_widths(sp);
    const std::string t = j.text();
    CHECK(t.find("max/min ratio") != std::string::npos);
    CHECK(t.find("CGRID2B[1]") != std::string::npos);    /* widest */
    CHECK(t.find("SEMBFUNC[13]") != std::string::npos);  /* narrowest + pinned */
    CHECK(t.find("effectively pinned") != std::string::npos);
}

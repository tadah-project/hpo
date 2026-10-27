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

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
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
        {"TADAH", "RANDOM",     "has no initial step"},
        {"TADAH", "GRID",       "GRID_POINTS"},
        {"NLOPT", "GN_CRS2_LM", "never reads an initial step"},
        {"NLOPT", "LD_LBFGS",   "never reads an initial step"},
        {"TADAH", "ANNEAL",     "STEP_FRAC"},
        {"TADAH", "ANNEAL_PT",  "STEP_FRAC"},
        {"ASA",   "ASA",        "parameter temperature"},
        {"DLIB",  "BFGS",       "fd_step"},
        /* was CERES:LBFGS, refused since; before the fix the case passes
         * as well (a pin) */
        {"CERES", "STEEPEST_DESCENT", "max_ls_iter"},
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
    /* One algorithm of each consumption mode; LN_PRAXIS stands for the
     * MinCollapsed ones in place of LN_NEWUOA_BOUND, which is refused.
     * Before the fix: passes as well (a pin; LN_NEWUOA_BOUND passed too). */
    for (const char* algo : {"LN_SBPLX", "LN_NELDERMEAD", "LN_BOBYQA",
                             "LN_COBYLA", "LN_PRAXIS", "LD_MMA"}) {
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
    /* Once per run and backend: the check runs at parse time, at startup
     * and when each phase starts, and it printed the same WARNING each
     * time (three times in a plain run). The text says what a point the
     * optimiser proposes outside the box does -- it is not evaluated: it is
     * scored as a failure graded by its distance to the box, counts
     * against MAXEVAL and is never the best (box_penalty.h) --, and which
     * optimisers keep to the box. CERES:STEEPEST_DESCENT is the only
     * registered backend that ignores the bounds: NLOPT:LN_NEWUOA, which
     * did, is refused, so the check throws its refusal and warns nothing.
     * Pins: that WARNING once, for CERES:STEEPEST_DESCENT; the slope of
     * the score points away from the box (box_penalty.h: g_k = c e_k, c > 0,
     * e_k > 0 above the box), so a descent step leads back into it.
     * Before the fix: the WARNING said such a point is evaluated there and
     * can become the best ("... its iterates can leave the box ... and are
     * evaluated there, and such a point can become the best ..."), for
     * CERES:LBFGS, refused since; then (review) it gave the slope the wrong
     * sign, "... graded by its distance to the box, whose slope points back
     * into it, it counts against MAXEVAL ...": the count of this text is 0. */
    JournalCapture j;
    auto sp = make_spec("NLOPT", "LN_NEWUOA");
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER"),
                        Catch::StartsWith("NLOPT:LN_NEWUOA is refused: "));
    CHECK(count_of(j.text(), "ignores the OPTIM box") == 0u);

    auto ce = make_spec("CERES", "STEEPEST_DESCENT");
    for (const char* where : {"OUTER", "run"})
        REQUIRE_NOTHROW(validate_optimizer_scaling(ce, where));
    const std::string ceres =
        "the STRATEGY-level optimiser: CERES:STEEPEST_DESCENT ignores the "
        "OPTIM box: Ceres' "
        "line-search solver has no bounds, so its line searches can "
        "propose points outside the box (a LOG dimension stays positive, a "
        "linear one is unbounded). Such a point is not evaluated: it is "
        "scored as a failure graded by its distance to the box, whose slope "
        "points away from the box, so a descent step leads back into it; it "
        "counts against MAXEVAL and it is never the best; the first is "
        "reported with a WARNING and the DIAGNOSTICS section of hpo_run.log "
        "counts them all. DLIB:BFGS and the NLopt LD_* algorithms keep to the "
        "box";
    CHECK(count_of(j.text(), ceres) == 1u);
    CHECK(count_of(j.text(), "ignores the OPTIM box") == 1u);
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
                            Catch::Contains("outside its OPTIM box") &&
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
                        Catch::Contains("outside its OPTIM box"));
}

TEST_CASE("scale_validation: the tree walk checks the PRESTAGE EXPLORE stage and its INNER",
          "[HPO][scale][step][prestage]")
{
    OptimizerSpec root = make_spec("NLOPT", "LN_BOBYQA");
    root.strategy_type = StrategyType::PRESTAGE;
    root.prestage = std::make_unique<OptimizerSpec>(make_spec("NLOPT", "GN_CRS2_LM"));
    REQUIRE_NOTHROW(run_scale_validation(root));

    root.prestage->step = 0.1;       /* GN_CRS2_LM never reads a step */
    REQUIRE_THROWS_WITH(run_scale_validation(root),
                        Catch::Contains("the PRESTAGE exploration (EXPLORE block): "
                                        "NLOPT:GN_CRS2_LM does not use "
                                        "STEP"));

    root.prestage = std::make_unique<OptimizerSpec>(make_spec("NLOPT", "GN_MLSL_LDS"));
    root.prestage->inner = std::make_unique<OptimizerSpec>(make_spec("NLOPT", "LD_LBFGS"));
    root.prestage->inner->step = 0.1;   /* LD_LBFGS never reads a step */
    REQUIRE_THROWS_WITH(run_scale_validation(root),
                        Catch::Contains("the INNER search of the EXPLORE block: "
                                        "NLOPT:LD_LBFGS does not use "
                                        "STEP"));

    root.prestage->inner = std::make_unique<OptimizerSpec>(make_spec("NLOPT", "LN_SBPLX"));
    root.prestage->inner->step = 0.1;   /* LN_SBPLX consumes it */
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

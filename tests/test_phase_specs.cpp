// test_phase_specs.cpp
//
// phase_specs.h derives the spec each phase of a STRATEGY runs with: each
// MULTISTART start (one INIT point, BUDGET_PER_START or its share of
// MAXEVAL, SEED + i), the PRESTAGE exploration and refinement, and the
// INNER blocks. MultiStartPolicy and StagedPolicy run the phases with these
// functions, and validate_hpo_setup() checks each phase's backend settings
// on them (backend_rules.h), so the startup checks see what the run runs.
//
// backend_rules.h: validate_backend() calls each backend's own validator,
// and require_maxeval_budget() names a backend that cannot stop without
// MAXEVAL.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::hpo::HPOTargets;
using tadah::hpo::MultiStartPlan;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::PhaseSpec;

struct ScopedTempFile {
    fs::path path;
    explicit ScopedTempFile(const std::string& body)
        : path(fs::temp_directory_path() /
               ("tadah_phase_specs_" + std::to_string(::getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()) + ".hpo"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~ScopedTempFile() { std::error_code ec; fs::remove(path, ec); }
};

/* The parsed spec of an OPTIMIZER block over one OPTIM dimension. */
OptimizerSpec parse_spec(const std::string& optimizer)
{
    tadah::core::PeriodicTable::initialize();
    ScopedTempFile tf("OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n" + optimizer);
    tadah::core::Context ctx;
    ctx.add("ATOM", "H");
    ctx.add("WATOM", 1.0);
    HPOTargets trg(ctx, tf.path.string());
    return trg.spec;
}

/* A bound local spec with `n` INIT points, for the plan functions. */
OptimizerSpec local_spec(std::size_t n_init)
{
    OptimizerSpec s;
    s.lib = "DLIB";
    s.algo = "BFGS";
    s.low = {0.0, 0.0};
    s.high = {1.0, 1.0};
    s.analyse_logscale();
    for (std::size_t i = 0; i < n_init; ++i)
        s.optim_init.push_back({0.1 * static_cast<double>(i + 1), 0.5});
    return s;
}

std::vector<std::string> names(const std::vector<PhaseSpec>& phases)
{
    std::vector<std::string> out;
    for (const auto& p : phases) out.push_back(p.name);
    return out;
}

std::string error_of(const OptimizerSpec& s)
{
    try {
        tadah::hpo::validate_backend(s);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("phase specs: the MULTISTART plan and each start's spec",
          "[HPO][phase_specs][multistart]")
{
    using tadah::hpo::multistart_start_spec;
    using tadah::hpo::plan_multistart;

    SECTION("the outer MAXEVAL is split over the starts, rounded up") {
        auto s = local_spec(4);
        s.stop.maxeval = 20;
        s.seed = 7;
        const MultiStartPlan plan = plan_multistart(s, 3, std::nullopt);
        CHECK_FALSE(plan.single_run);
        CHECK(plan.starts == 3u);
        CHECK(plan.per_start_maxeval == std::optional<std::uint64_t>(7));
        CHECK_FALSE(plan.from_budget_per_start);
        for (std::size_t i = 0; i < 3; ++i) {
            INFO("start " << i);
            const OptimizerSpec sub = multistart_start_spec(s, plan, i);
            CHECK(sub.optim_init ==
                  std::vector<std::vector<double>>{s.optim_init[i]});
            CHECK(sub.stop.maxeval == std::optional<std::uint64_t>(7));
            CHECK(sub.seed == std::optional<std::uint64_t>(7 + i));
            CHECK(sub.lib == "DLIB");
            CHECK(sub.algo == "BFGS");
        }
    }
    SECTION("BUDGET_PER_START is each start's MAXEVAL") {
        auto s = local_spec(4);
        s.stop.maxeval = 20;
        const MultiStartPlan plan = plan_multistart(s, 3, std::size_t{5});
        CHECK(plan.per_start_maxeval == std::optional<std::uint64_t>(5));
        CHECK(plan.from_budget_per_start);
        const OptimizerSpec sub = multistart_start_spec(s, plan, 2);
        CHECK(sub.stop.maxeval == std::optional<std::uint64_t>(5));
        CHECK_FALSE(sub.seed);
    }
    SECTION("fewer INIT points than N_STARTS: one start per point") {
        auto s = local_spec(2);
        s.stop.maxeval = 9;
        const MultiStartPlan plan = plan_multistart(s, 5, std::nullopt);
        CHECK(plan.starts == 2u);
        CHECK(plan.per_start_maxeval == std::optional<std::uint64_t>(5));
    }
    SECTION("without MAXEVAL or BUDGET_PER_START a start has no budget") {
        const MultiStartPlan plan =
            plan_multistart(local_spec(3), 3, std::nullopt);
        CHECK_FALSE(plan.per_start_maxeval);
        CHECK_FALSE(
            multistart_start_spec(local_spec(3), plan, 0).stop.maxeval);
    }
    SECTION("a multi-seed backend, or no INIT point: one run of the spec") {
        auto gfs = local_spec(3);
        gfs.algo = "GFS";
        REQUIRE(tadah::hpo::optimizer_caps(gfs).accepts_multi_seed);
        CHECK(plan_multistart(gfs, 3, std::size_t{5}).single_run);
        CHECK(plan_multistart(local_spec(0), 3, std::nullopt).single_run);
    }
    SECTION("the split does not overflow near the top of the range") {
        auto s = local_spec(2);
        s.stop.maxeval = std::numeric_limits<std::uint64_t>::max();
        CHECK(plan_multistart(s, 2, std::nullopt).per_start_maxeval ==
              std::optional<std::uint64_t>(
                  std::numeric_limits<std::uint64_t>::max() / 2 + 1));
    }
}

TEST_CASE("phase specs: the PRESTAGE refinement starts from the candidates",
          "[HPO][phase_specs][prestage]")
{
    auto s = local_spec(1);
    s.strategy_type = tadah::hpo::StrategyType::PRESTAGE;
    s.stop.maxeval = 30;
    s.prestage = std::make_unique<OptimizerSpec>(local_spec(1));
    const std::vector<std::vector<double>> candidates = {{0.2, 0.3},
                                                         {0.4, 0.6}};
    const OptimizerSpec refine =
        tadah::hpo::prestage_refinement_spec(s, candidates);
    CHECK_FALSE(refine.prestage);
    CHECK(refine.optim_init == candidates);
    CHECK(refine.stop.maxeval == std::optional<std::uint64_t>(30));
    CHECK(refine.lib == "DLIB");
}

TEST_CASE("phase specs: every phase of each STRATEGY TYPE",
          "[HPO][phase_specs]")
{
    using tadah::hpo::strategy_phase_specs;

    SECTION("PLAIN with an INNER block") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO G_MLSL_LDS\n"
            "MAXEVAL 40\nINNER\nLIB NLOPT\nALGO LN_BOBYQA\nMAXEVAL 5\n"
            "ENDINNER\nENDSTRATEGY\nENDOPTIMIZER\n"));
        CHECK(names(phases) ==
              std::vector<std::string>{
                  "the STRATEGY-level optimiser",
                  "the INNER block of the STRATEGY-level optimiser"});
        REQUIRE(phases.size() == 2u);
        CHECK(phases[0].spec.stop.maxeval == std::optional<std::uint64_t>(40));
        CHECK(phases[1].spec.algo == "LN_BOBYQA");
        CHECK(phases[1].spec.stop.maxeval == std::optional<std::uint64_t>(5));
    }
    SECTION("MULTISTART: a start with its share of MAXEVAL and SEED") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nINIT\nMETHOD RANDOM 2\nSEED 3\nENDINIT\n"
            "STRATEGY\nTYPE MULTISTART\nLIB DLIB\nALGO BFGS\nMAXEVAL 9\n"
            "N_STARTS 2\nENDSTRATEGY\nENDOPTIMIZER\n"));
        REQUIRE(phases.size() == 1u);
        CHECK(phases[0].name == "each MULTISTART start");
        CHECK(phases[0].spec.stop.maxeval == std::optional<std::uint64_t>(5));
        CHECK(phases[0].spec.optim_init.size() == 1u);
    }
    SECTION("MULTISTART with BUDGET_PER_START and no MAXEVAL") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nINIT\nMETHOD RANDOM 2\nSEED 3\nENDINIT\n"
            "STRATEGY\nTYPE MULTISTART\nLIB NLOPT\nALGO LN_SBPLX\n"
            "N_STARTS 2\nBUDGET_PER_START 6\nSEED 11\nENDSTRATEGY\n"
            "ENDOPTIMIZER\n"));
        REQUIRE(phases.size() == 1u);
        CHECK(phases[0].spec.stop.maxeval == std::optional<std::uint64_t>(6));
        CHECK(phases[0].spec.seed == std::optional<std::uint64_t>(11));
    }
    SECTION("MULTISTART over a multi-seed backend: one run") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nINIT\nMETHOD RANDOM 3\nSEED 3\nENDINIT\n"
            "STRATEGY\nTYPE MULTISTART\nLIB DLIB\nALGO GFS\nMAXEVAL 12\n"
            "N_STARTS 3\nENDSTRATEGY\nENDOPTIMIZER\n"));
        REQUIRE(phases.size() == 1u);
        CHECK(phases[0].name == "the MULTISTART run");
        CHECK(phases[0].spec.stop.maxeval == std::optional<std::uint64_t>(12));
        CHECK(phases[0].spec.optim_init.size() == 3u);
    }
    SECTION("PRESTAGE with TOPK 3: the exploration, a refinement start with "
            "its share of MAXEVAL, and the exploration's INNER block") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB DLIB\nALGO BFGS\n"
            "MAXEVAL 20\nTOPK 3\nEXPLORE\nLIB NLOPT\nALGO G_MLSL_LDS\n"
            "MAXEVAL 12\nINNER\nLIB NLOPT\nALGO LN_SBPLX\nENDINNER\n"
            "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n"));
        CHECK(names(phases) ==
              std::vector<std::string>{
                  "the PRESTAGE exploration (EXPLORE block)",
                  "the PRESTAGE refinement",
                  "the INNER block of the PRESTAGE exploration (EXPLORE "
                  "block)"});
        REQUIRE(phases.size() == 3u);
        CHECK(phases[0].spec.algo == "G_MLSL_LDS");
        CHECK(phases[0].spec.stop.maxeval == std::optional<std::uint64_t>(12));
        CHECK(phases[1].spec.algo == "BFGS");
        CHECK_FALSE(phases[1].spec.prestage);
        CHECK(phases[1].spec.stop.maxeval == std::optional<std::uint64_t>(7));
        CHECK(phases[1].spec.optim_init.size() == 1u);
        CHECK(phases[2].spec.algo == "LN_SBPLX");
    }
    SECTION("PRESTAGE with TOPK 1: the refinement runs once, with MAXEVAL") {
        const auto phases = strategy_phase_specs(parse_spec(
            "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nEXPLORE\nLIB TADAH\n"
            "ALGO RANDOM\nMAXEVAL 3\nENDEXPLORE\nREFINE\nLIB NLOPT\n"
            "ALGO LN_SBPLX\nMAXEVAL 20\nENDREFINE\nENDSTRATEGY\n"
            "ENDOPTIMIZER\n"));
        REQUIRE(phases.size() == 2u);
        CHECK(phases[1].name == "the PRESTAGE refinement");
        CHECK(phases[1].spec.stop.maxeval == std::optional<std::uint64_t>(20));
        CHECK(phases[1].spec.optim_init.size() == 1u);
    }
}

TEST_CASE("backend rules: validate_backend runs each backend's own checks",
          "[HPO][backend_rules]")
{
    auto with = [](const std::string& lib, const std::string& algo) {
        auto s = local_spec(1);
        s.lib = lib;
        s.algo = algo;
        s.stop.maxeval = 10;
        return s;
    };
    /* A spec each backend accepts. */
    for (const auto& la : std::vector<std::pair<std::string, std::string>>{
             {"TADAH", "RANDOM"}, {"TADAH", "GRID"}, {"TADAH", "ANNEAL"},
             {"TADAH", "ANNEAL_PT"}, {"TADAH", "ANNEAL_PA"}, {"ASA", "ASA"},
             {"DLIB", "GFS"}, {"DLIB", "BFGS"}, {"NLOPT", "LN_SBPLX"},
             {"CERES", "LBFGS"}}) {
        INFO(la.first << ":" << la.second);
        CHECK(error_of(with(la.first, la.second)) == "");
    }
    /* No optimiser named: require_driver() reports it. */
    CHECK(error_of(OptimizerSpec{}) == "");

    struct Case { std::string lib, algo; void (*set)(OptimizerSpec&);
                  std::string message; };
    for (const auto& c : std::vector<Case>{
             {"DLIB", "BFGS", [](OptimizerSpec& s) { s.stop.maxtime = 60; },
              "DLIB:BFGS does not support MAXTIME"},
             {"CERES", "LBFGS", [](OptimizerSpec& s) { s.stop.stopval = 1; },
              "CERES:LBFGS does not support STOPVAL"},
             {"TADAH", "ANNEAL",
              [](OptimizerSpec& s) { s.param["NOT_A_KEY"] = 1; },
              "TADAH:ANNEAL: unknown PARAM 'NOT_A_KEY'"},
             {"TADAH", "ANNEAL_PT",
              [](OptimizerSpec& s) {
                  s.param["REPLICAS"] = 2;
                  s.param_raw["LADDER_ADAPT"] = "ON";
              },
              "TADAH:ANNEAL_PT: LADDER_ADAPT requires REPLICAS >= 3"},
             {"ASA", "ASA", [](OptimizerSpec& s) { s.stop.xtol_rel = 1e-3; },
              "ASA:ASA does not implement XTOL_REL/XTOL_ABS"},
             {"TADAH", "GRID", [](OptimizerSpec& s) { s.param["FOO"] = 1; },
              "TADAH:GRID: unknown PARAM 'FOO'"},
             {"TADAH", "GRID",
              [](OptimizerSpec& s) { s.param_raw["GRID_POINTS"] = "AUTO"; },
              "TADAH:GRID PARAM GRID_POINTS must be an integer >= 2 (got "
              "'AUTO')"},
             {"NLOPT", "LN_SBPLX",
              [](OptimizerSpec& s) { s.stop.maxeval = 4294967299u; },
              "NLopt spec with MAXEVAL 4294967299"},
             {"CERES", "LEVENBERG_MARQUARDT", [](OptimizerSpec&) {},
              "CERES:LEVENBERG_MARQUARDT is deferred"},
             {"NLOPT", "NOT_AN_ALGO", [](OptimizerSpec&) {},
              "unknown optimizer NLOPT:NOT_AN_ALGO"}}) {
        INFO(c.lib << ":" << c.algo << " -> " << c.message);
        auto s = with(c.lib, c.algo);
        c.set(s);
        CHECK_THAT(error_of(s), Catch::Contains(c.message));
    }
}

TEST_CASE("backend rules: a backend without a stopping rule of its own needs "
          "MAXEVAL", "[HPO][backend_rules][budget]")
{
    /* The caps registry says which: TADAH:RANDOM and DLIB:GFS. */
    std::vector<std::string> needing;
    for (const auto* c : tadah::hpo::filter_optimizer_caps(
             [](const tadah::hpo::OptimizerCaps& k) { return k.needs_maxeval; }))
        needing.push_back(c->lib + ":" + c->algo);
    std::sort(needing.begin(), needing.end());
    CHECK(needing == std::vector<std::string>{"DLIB:GFS", "TADAH:RANDOM"});

    auto s = local_spec(1);
    s.lib = "TADAH";
    s.algo = "RANDOM";
    CHECK(error_of(s) ==
          "TADAH:RANDOM needs MAXEVAL: it stops only when its MAXEVAL "
          "evaluations are spent (it reads neither MAXTIME nor STOPVAL); set "
          "MAXEVAL (an integer from 1 to 2147483647)");
    s.stop.maxtime = 60;   // not read by RANDOM
    CHECK_THAT(error_of(s), Catch::StartsWith("TADAH:RANDOM needs MAXEVAL"));
    s.stop.maxeval = 3;
    CHECK(error_of(s) == "");

    s.lib = "DLIB";
    s.algo = "GFS";
    s.stop = {};
    CHECK(error_of(s) ==
          "DLIB:GFS needs MAXEVAL: dlib's global search has no stopping rule "
          "of its own and reads neither MAXTIME nor STOPVAL; set MAXEVAL (an "
          "integer from 1 to 2147483647)");

    /* Every other backend runs without an evaluation limit: GRID, the
     * local optimisers, ASA (its own limits), the annealers with explicit
     * lengths. */
    for (const auto& la : std::vector<std::pair<std::string, std::string>>{
             {"TADAH", "GRID"}, {"DLIB", "BFGS"}, {"NLOPT", "LN_SBPLX"},
             {"NLOPT", "GN_DIRECT"}, {"CERES", "LBFGS"}, {"ASA", "ASA"},
             {"TADAH", "ANNEAL_PT"}, {"TADAH", "ANNEAL_PA"}}) {
        INFO(la.first << ":" << la.second);
        auto t = local_spec(1);
        t.lib = la.first;
        t.algo = la.second;
        CHECK(error_of(t) == "");
    }
    auto anneal = local_spec(1);
    anneal.lib = "TADAH";
    anneal.algo = "ANNEAL";
    CHECK_THAT(error_of(anneal),
               Catch::Contains("CHAIN_LENGTH AUTO needs MAXEVAL"));
    anneal.param["CHAIN_LENGTH"] = 5;
    CHECK(error_of(anneal) == "");
}

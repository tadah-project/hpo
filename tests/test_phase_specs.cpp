// test_phase_specs.cpp
//
// phase_specs.h derives the spec each phase of a STRATEGY runs with: each
// MULTISTART start (one INIT point, BUDGET_PER_START or its share of
// MAXEVAL, SEED + i), the PRESTAGE exploration and refinement, and the
// INNER blocks (the local search NLopt's MLSL and AUGLAG run,
// inner_run_spec()). MultiStartPolicy and StagedPolicy run the phases with
// these functions, and validate_hpo_setup() checks each phase's backend
// settings on them (backend_rules.h), so the startup checks see what the run
// runs.
//
// backend_rules.h: validate_backend() calls each backend's own validator,
// and require_stop() names a backend that cannot end without a stop.

#include "catch2/catch.hpp"
#include "ceres_build.h"

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
#include <iostream>
#include <optional>
#include <sstream>
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
        tadah::hpo::validate_backend(s, tadah::hpo::RestartStop::Off);
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
    SECTION("an NLopt spec seeded by its INIT block's SEED: start i takes "
            "that SEED + i") {
        /* Pins: NLopt takes the INIT block's SEED when its block has no SEED
         * (make_nlopt), so start i of such a spec gets the INIT SEED + i
         * (7, 8, 9) and no SEED; a DLIB spec, which no INIT SEED seeds, and
         * an NLopt spec with a SEED of its own (SEED + i) keep the INIT SEED.
         * Before the fix: every start of the NLopt spec kept the INIT SEED 7
         * (the NLopt checks fail). */
        auto s = local_spec(3);
        s.lib = "NLOPT";
        s.algo = "LN_SBPLX";
        s.init_strategy_spec.seed = 7;
        const MultiStartPlan plan = plan_multistart(s, 3, std::nullopt);
        REQUIRE(plan.starts == 3u);
        for (std::size_t i = 0; i < 3; ++i) {
            INFO("start " << i);
            const OptimizerSpec sub = multistart_start_spec(s, plan, i);
            CHECK(sub.init_strategy_spec.seed ==
                  std::optional<std::uint64_t>(7 + i));
            CHECK_FALSE(sub.seed);
        }
        auto d = local_spec(3);
        d.init_strategy_spec.seed = 7;
        CHECK(multistart_start_spec(d, plan, 2).init_strategy_spec.seed ==
              std::optional<std::uint64_t>(7));
        s.seed = 3;
        const OptimizerSpec own = multistart_start_spec(s, plan, 2);
        CHECK(own.seed == std::optional<std::uint64_t>(5));
        CHECK(own.init_strategy_spec.seed == std::optional<std::uint64_t>(7));
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

/* Pins: the starting points of a phase keep their INIT origin (CONFIG,
 * WARM, a sampled point) only where it holds, so that an NLopt search whose
 * start moves warns for a user's own point (nlopt_adapter.h): a MULTISTART
 * start keeps the origin of its INIT point, and a PRESTAGE refinement,
 * which starts from exploration candidates, carries no origin, also when it
 * has as many candidates as the spec had INIT points.
 * Before the fix: the refinement kept the spec's origins -- one CONFIG
 * origin for one candidate here, so an LN_BOBYQA refinement whose candidate
 * moved named it a CONFIG start. The MULTISTART part fails on the code
 * before a start kept the origin of its point (the change came with the
 * moving LN_BOBYQA starts, before this case): multistart_start_spec copied
 * the spec's origins, two for its one start ("2 == 1"), and an NLopt search
 * then took the origins for stale and named no CONFIG start ("LN_BOBYQA
 * under MULTISTART: a moved CONFIG start gets the warning ...",
 * test_step_plan.cpp, runs it). */
TEST_CASE("phase specs: a phase's starting points keep their INIT origin "
          "only where it holds", "[HPO][phase_specs][prestage][multistart]")
{
    using tadah::hpo::InitOrigin;
    auto s = local_spec(2);
    s.optim_init_origin = {InitOrigin{InitOrigin::Kind::Config, {}},
                           InitOrigin{InitOrigin::Kind::Random, {}}};
    const MultiStartPlan plan = tadah::hpo::plan_multistart(s, 2, std::nullopt);
    for (std::size_t i = 0; i < 2; ++i) {
        INFO("MULTISTART start " << i + 1);
        const OptimizerSpec sub = tadah::hpo::multistart_start_spec(s, plan, i);
        REQUIRE(sub.optim_init_origin.size() == 1u);
        CHECK(sub.optim_init_origin.front().kind ==
              s.optim_init_origin[i].kind);
    }

    auto one = local_spec(1);
    one.optim_init_origin = {InitOrigin{InitOrigin::Kind::Config, {}}};
    one.strategy_type = tadah::hpo::StrategyType::PRESTAGE;
    one.prestage = std::make_unique<OptimizerSpec>(local_spec(1));
    const OptimizerSpec refine =
        tadah::hpo::prestage_refinement_spec(one, {{0.2, 0.3}});
    CHECK(refine.optim_init.size() == 1u);
    CHECK(refine.optim_init_origin.empty());
}

/* A PRESTAGE exploration whose optimiser reads SEED and that has no seed of
 * its own takes the STRATEGY-level SEED, as the refinement's starts and the
 * RESTART rounds derive theirs from it (the refinement draws streams of its
 * own key, phase_stream_seed, so the two phases never share numbers). Its
 * own seeds come first: the EXPLORE block's SEED and, for NLopt, its INIT
 * block's SEED. An exploration whose optimiser reads no SEED (DLIB:GFS)
 * takes none, and without a STRATEGY-level SEED the exploration has none.
 * Before the fix: the exploration never had a seed it did not set itself,
 * so a run with SEED in its OPTIMIZER or STRATEGY block drew the
 * exploration from NLopt's time seed (random_device for TADAH:RANDOM and
 * the annealers) and did not repeat. */
TEST_CASE("phase specs: a PRESTAGE exploration without a SEED takes the "
          "STRATEGY-level SEED", "[HPO][phase_specs][prestage][seed]")
{
    const auto prestage = [](const std::string& strategy_seed,
                             const std::string& explore) {
        return parse_spec(
            "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_BOBYQA\n"
            "MAXEVAL 20\n" + strategy_seed + "EXPLORE\n" + explore +
            "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n");
    };
    const std::string crs2 = "LIB NLOPT\nALGO GN_CRS2_LM\nMAXEVAL 20\n";
    SECTION("an NLopt exploration takes it") {
        const OptimizerSpec s = prestage("SEED 7\n", crs2);
        REQUIRE(s.prestage);
        CHECK(s.seed == std::optional<std::uint64_t>(7));
        CHECK(s.prestage->seed == std::optional<std::uint64_t>(7));
        /* the refinement keeps the SEED */
        const OptimizerSpec refine =
            tadah::hpo::prestage_refinement_spec(s, {{1.0}});
        CHECK(refine.seed == std::optional<std::uint64_t>(7));
    }
    SECTION("SEED in the OPTIMIZER block, outside STRATEGY") {
        const OptimizerSpec s = parse_spec(
            "OPTIMIZER\nSEED 9\nSTRATEGY\nTYPE PRESTAGE\nLIB NLOPT\n"
            "ALGO LN_BOBYQA\nMAXEVAL 20\nEXPLORE\n" + crs2 +
            "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n");
        REQUIRE(s.prestage);
        CHECK(s.prestage->seed == std::optional<std::uint64_t>(9));
    }
    SECTION("TADAH:RANDOM and TADAH:ANNEAL explorations take it") {
        for (const std::string algo : {"RANDOM", "ANNEAL"}) {
            INFO(algo);
            const OptimizerSpec s = prestage(
                "SEED 7\n", "LIB TADAH\nALGO " + algo + "\nMAXEVAL 20\n");
            REQUIRE(s.prestage);
            CHECK(s.prestage->seed == std::optional<std::uint64_t>(7));
        }
    }
    SECTION("the EXPLORE block's SEED comes first") {
        const OptimizerSpec s = prestage("SEED 7\n", crs2 + "SEED 3\n");
        REQUIRE(s.prestage);
        CHECK(s.prestage->seed == std::optional<std::uint64_t>(3));
    }
    SECTION("an NLopt exploration seeded by its INIT block keeps that seed") {
        const OptimizerSpec s = prestage(
            "SEED 7\n",
            crs2 + "INIT\nMETHOD RANDOM 2\nSEED 5\nENDINIT\n");
        REQUIRE(s.prestage);
        CHECK_FALSE(s.prestage->seed);
        CHECK(s.prestage->init_strategy_spec.seed ==
              std::optional<std::uint64_t>(5));
        CHECK(s.prestage->nlopt_seed_from_init());
    }
    SECTION("an exploration whose optimiser reads no SEED takes none") {
        const OptimizerSpec s = prestage(
            "SEED 7\n", "LIB DLIB\nALGO GFS\nMAXEVAL 20\n");
        REQUIRE(s.prestage);
        CHECK_FALSE(s.prestage->seed);
    }
    SECTION("without a STRATEGY-level SEED the exploration has none") {
        const OptimizerSpec s = prestage("", crs2);
        REQUIRE(s.prestage);
        CHECK_FALSE(s.prestage->seed);
    }
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

/* DLIB:GFS reads FTOL_ABS as the accuracy of dlib's solver, which dlib
 * requires to be >= 0: a negative one is refused at startup, naming the
 * phase and the value ("the STRATEGY-level optimiser: DLIB:GFS: FTOL_ABS is
 * the accuracy of dlib's solver and must be >= 0 (got -0.001)"), and 0 is
 * accepted.
 * Before the fix: startup accepted it, and the search failed inside dlib
 * after the training data had been read ("Failing expression was 0 <=
 * eps"). */
TEST_CASE("backend rules: DLIB:GFS refuses a negative FTOL_ABS at startup",
          "[HPO][backend_rules]")
{
    auto s = local_spec(1);
    s.lib = "DLIB";
    s.algo = "GFS";
    s.stop.maxeval = 10;
    s.stop.ftol_abs = -1e-3;
    CHECK(error_of(s) == "DLIB:GFS: FTOL_ABS is the accuracy of dlib's "
                         "solver and must be >= 0 (got -0.001)");
    s.stop.ftol_abs = 0.0;
    CHECK(error_of(s) == "");

    const OptimizerSpec parsed = parse_spec(
        "OPTIMIZER\nLIB DLIB\nALGO GFS\nMAXEVAL 10\nFTOL_ABS -1e-3\n"
        "ENDOPTIMIZER\n");
    std::string what;
    try {
        tadah::hpo::validate_phase_backends(parsed);
    } catch (const std::exception& e) {
        what = e.what();
    }
    CHECK_THAT(what, Catch::Contains(
                         "the STRATEGY-level optimiser: DLIB:GFS: FTOL_ABS is "
                         "the accuracy of dlib's solver and must be >= 0 "
                         "(got -0.001)"));
}

/* CERES:STEEPEST_DESCENT stands in for CERES:LBFGS, refused since (a pin).
 * A build without Ceres refuses every CERES:STEEPEST_DESCENT spec here as
 * not built (ceres_build.h), the deferred CERES:LEVENBERG_MARQUARDT with its own
 * message first: a pin of that refusal. Before this test change such a
 * build failed the CERES rows, expecting the Ceres validator's result. */
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
             {"CERES", "STEEPEST_DESCENT"}}) {
        INFO(la.first << ":" << la.second);
        CHECK(error_of(with(la.first, la.second)) ==
              (la.first == "CERES" ? hpo_test::ceres_expect("")
                                   : std::string()));
    }
    /* No optimiser named: require_driver() reports it. */
    CHECK(error_of(OptimizerSpec{}) == "");

    struct Case { std::string lib, algo; void (*set)(OptimizerSpec&);
                  std::string message; };
    for (const auto& c : std::vector<Case>{
             {"DLIB", "BFGS", [](OptimizerSpec& s) { s.stop.maxtime = 60; },
              "DLIB:BFGS does not support MAXTIME"},
             {"CERES", "STEEPEST_DESCENT", [](OptimizerSpec& s) { s.stop.stopval = 1; },
              hpo_test::ceres_expect("CERES:STEEPEST_DESCENT does not support STOPVAL")},
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
              /* It was "ASA:ASA does not implement XTOL_REL/XTOL_ABS". */
              "ASA:ASA does not take XTOL_REL or XTOL_ABS"},
             {"TADAH", "GRID", [](OptimizerSpec& s) { s.param["FOO"] = 1; },
              "TADAH:GRID: unknown PARAM 'FOO'"},
             {"TADAH", "GRID",
              [](OptimizerSpec& s) { s.param_raw["GRID_POINTS"] = "AUTO"; },
              /* It was "TADAH:GRID PARAM GRID_POINTS must be an integer
               * >= 2 (got 'AUTO')". */
              "TADAH:GRID: PARAM GRID_POINTS must be an integer from 2 to "
              "4294967295, got 'AUTO' (the points of each dimension include "
              "both of its bounds)"},
             {"NLOPT", "LN_SBPLX",
              [](OptimizerSpec& s) { s.stop.maxeval = 4294967299u; },
              "NLopt spec with MAXEVAL 4294967299"},
             /* Refused by the caps registry, and unknown: "CERES:
              * LEVENBERG_MARQUARDT is deferred" and "unknown optimizer
              * NLOPT:NOT_AN_ALGO (not registered in optimizer_caps)" until
              * the least-squares solvers joined the refusal table and the
              * lookup said what is wrong. */
             {"CERES", "LEVENBERG_MARQUARDT", [](OptimizerSpec&) {},
              "CERES:LEVENBERG_MARQUARDT is refused: "},
             {"NLOPT", "NOT_AN_ALGO", [](OptimizerSpec&) {},
              "unknown optimiser NLOPT:NOT_AN_ALGO: LIB NLOPT has no ALGO "
              "NOT_AN_ALGO"}}) {
        INFO(c.lib << ":" << c.algo << " -> " << c.message);
        auto s = with(c.lib, c.algo);
        c.set(s);
        CHECK_THAT(error_of(s), Catch::Contains(c.message));
    }

    /* Pins: validate_backend() -- the check each driver applies when it
     * starts, for a spec built outside the parser -- refuses a key the
     * backend never reads: POPULATION on TADAH:RANDOM, SEED on TADAH:GRID
     * and CERES, VECTOR_ARR on DLIB:GFS, a tolerance NLopt's GN_ESCH never
     * tests, and an INNER spec's STOPVAL, which NLopt replaces with the
     * enclosing run's.
     * Before the fix: error_of() was "" for each (the driver ran with the
     * key and ignored it). */
    for (const auto& c : std::vector<Case>{
             {"TADAH", "RANDOM", [](OptimizerSpec& s) { s.pop = 8; },
              "TADAH:RANDOM does not use POPULATION; remove it (it is read "
              "by NLOPT:GD_MLSL, "},
             {"TADAH", "GRID", [](OptimizerSpec& s) { s.seed = 3; },
              "TADAH:GRID does not use SEED: it draws no random numbers; "
              "remove it"},
             {"DLIB", "GFS", [](OptimizerSpec& s) { s.vector_storage = 5; },
              "DLIB:GFS does not use VECTOR_ARR; remove it (it is read by "
              "NLOPT:LD_LBFGS, "},
             {"NLOPT", "GN_ESCH",
              [](OptimizerSpec& s) { s.stop.ftol_rel = 1e-6; },
              "NLOPT:GN_ESCH does not support FTOL_REL: NLopt's global search "
              "tests no tolerance; stop it with MAXEVAL, MAXTIME or STOPVAL"},
             {"NLOPT", "AUGLAG",
              [](OptimizerSpec& s) {
                  s.inner = std::make_unique<OptimizerSpec>();
                  s.inner->lib = "NLOPT";
                  s.inner->algo = "LN_BOBYQA";
                  s.inner->stop.stopval = 0.1;
              },
              "NLopt INNER spec with a STOPVAL: NLopt's MLSL and AUGLAG "
              "replace the INNER search's STOPVAL with the enclosing "
              "block's; set STOPVAL on the enclosing block"},
             {"CERES", "STEEPEST_DESCENT", [](OptimizerSpec& s) { s.seed = 3; },
              hpo_test::ceres_expect("CERES:STEEPEST_DESCENT does not support SEED")}}) {
        INFO(c.lib << ":" << c.algo << " -> " << c.message);
        auto s = with(c.lib, c.algo);
        c.set(s);
        CHECK_THAT(error_of(s), Catch::StartsWith(c.message));
    }

    /* Pins: a tolerance on G_MLSL without an INNER block -- a spec built
     * outside the parser, which require_inner_backend() refuses at startup
     * and in run_inner() -- names the INNER block it needs, with no INNER
     * search to look at.
     * Before the review fix: "...; set it in the INNER block, whose local
     * search tests it", of a block the spec does not have. */
    auto g_mlsl = with("NLOPT", "G_MLSL");
    g_mlsl.stop.ftol_rel = 1e-6;
    CHECK(error_of(g_mlsl) ==
          "NLOPT:G_MLSL does not support FTOL_REL: MLSL's multistart loop "
          "tests no tolerance, and NLopt passes the block's tolerances to no "
          "INNER search; NLOPT:G_MLSL needs an INNER block, and only that "
          "block's search can test it");
}

TEST_CASE("backend rules: a backend with no convergence test and no finite "
          "schedule needs a stop", "[HPO][backend_rules][budget]")
{
    /* The caps registry says which (the phase 4c probes ran each without a
     * stop until it was killed). */
    std::vector<std::string> needing;
    for (const auto* c : tadah::hpo::filter_optimizer_caps(
             [](const tadah::hpo::OptimizerCaps& k) { return k.needs_stop; }))
        needing.push_back(c->lib + ":" + c->algo);
    std::sort(needing.begin(), needing.end());
    CHECK(needing == std::vector<std::string>{
                         "DLIB:GFS", "NLOPT:GD_MLSL", "NLOPT:GD_MLSL_LDS",
                         "NLOPT:GD_STOGO", "NLOPT:GD_STOGO_RAND",
                         "NLOPT:GN_AGS", "NLOPT:GN_CRS2_LM", "NLOPT:GN_DIRECT",
                         "NLOPT:GN_DIRECT_L", "NLOPT:GN_DIRECT_L_NOSCAL",
                         "NLOPT:GN_DIRECT_L_RAND", "NLOPT:GN_ESCH",
                         "NLOPT:GN_ISRES", "NLOPT:GN_MLSL",
                         "NLOPT:GN_MLSL_LDS", "NLOPT:G_MLSL",
                         "NLOPT:G_MLSL_LDS", "NLOPT:LD_CCSAQ", "NLOPT:LD_MMA",
                         "NLOPT:LD_SLSQP", "TADAH:ANNEAL_PT", "TADAH:RANDOM"});

    /* The stops each honours, as its message lists them: MAXEVAL, MAXTIME,
     * STOPVAL, and the tolerances its stop tests read. */
    struct Stops { std::string lib, algo, list; };
    for (const auto& c : std::vector<Stops>{
             {"TADAH", "RANDOM", "MAXEVAL, MAXTIME or STOPVAL (it honours no "
                                 "tolerance)"},
             {"DLIB", "GFS", "MAXEVAL, MAXTIME or STOPVAL (it honours no "
                             "tolerance)"},
             {"TADAH", "ANNEAL_PT", "MAXEVAL, MAXTIME, STOPVAL or a positive "
                                    "tolerance it honours (FTOL_REL, "
                                    "FTOL_ABS)"},
             {"NLOPT", "GN_CRS2_LM", "MAXEVAL, MAXTIME, STOPVAL or a positive "
                                     "tolerance it honours (FTOL_REL, "
                                     "FTOL_ABS, XTOL_REL, XTOL_ABS)"},
             {"NLOPT", "GN_DIRECT_L", "MAXEVAL, MAXTIME, STOPVAL or a positive "
                                      "tolerance it honours (FTOL_REL, "
                                      "FTOL_ABS, XTOL_REL, XTOL_ABS); DIRECT's "
                                      "XTOL_REL and XTOL_ABS end it only once "
                                      "every rectangle is that small, after "
                                      "roughly (1/XTOL_REL)^n evaluations in n "
                                      "dimensions"},
             {"NLOPT", "GN_ESCH", "MAXEVAL, MAXTIME or STOPVAL (it honours no "
                                  "tolerance)"},
             {"NLOPT", "GN_MLSL_LDS", "MAXEVAL, MAXTIME or STOPVAL (it honours "
                                      "no tolerance)"}}) {
        INFO(c.lib << ":" << c.algo);
        auto s = local_spec(1);
        s.lib = c.lib;
        s.algo = c.algo;
        const std::string err = error_of(s);
        CHECK_THAT(err, Catch::StartsWith(c.lib + ":" + c.algo +
                                          " needs a stop: ") &&
                        Catch::EndsWith("; set " + c.list));
        /* Each stop it honours is enough on its own. */
        for (const auto& set : std::vector<void (*)(OptimizerSpec&)>{
                 [](OptimizerSpec& t) { t.stop.maxeval = 3; },
                 [](OptimizerSpec& t) { t.stop.maxtime = 60; },
                 [](OptimizerSpec& t) { t.stop.stopval = 0.1; }}) {
            auto t = s;
            set(t);
            CHECK(error_of(t) == "");
        }
    }
    auto random = local_spec(1);
    random.lib = "TADAH";
    random.algo = "RANDOM";
    CHECK(error_of(random) ==
          "TADAH:RANDOM needs a stop: uniform random search has no "
          "convergence test and runs until it is interrupted; set MAXEVAL, "
          "MAXTIME or STOPVAL (it honours no tolerance)");
    /* Pins: a tolerance a backend does not test is no stop, and it is
     * refused for itself, before the missing stop, naming the stops the
     * backend honours.
     * Before the fix: it was accepted and counted as no stop, so the spec
     * was refused as "TADAH:RANDOM needs a stop: ...", and with MAXEVAL it
     * ran, ignoring it. */
    random.stop.ftol_rel = 1e-6;
    CHECK(error_of(random) ==
          "TADAH:RANDOM does not support FTOL_REL: uniform random search has "
          "no convergence test to apply it to; stop it with MAXEVAL, MAXTIME "
          "or STOPVAL");
    random.stop.maxeval = 5;
    CHECK(error_of(random) ==
          "TADAH:RANDOM does not support FTOL_REL: uniform random search has "
          "no convergence test to apply it to; stop it with MAXEVAL, MAXTIME "
          "or STOPVAL");
    auto pt = local_spec(1);
    pt.lib = "TADAH";
    pt.algo = "ANNEAL_PT";
    pt.stop.ftol_abs = 1e-8;   // the plateau rule's
    CHECK(error_of(pt) == "");

    /* A tolerance counts only above 0: NLopt switches one of 0 or below
     * off, and ANNEAL_PT's best-stall rule arms only above 0. */
    pt.stop.ftol_abs = 0.0;
    CHECK_THAT(error_of(pt), Catch::StartsWith("TADAH:ANNEAL_PT needs a stop"));
    auto crs = local_spec(1);
    crs.lib = "NLOPT";
    crs.algo = "GN_CRS2_LM";
    for (const double tol : {0.0, -1.0}) {
        INFO("tolerance " << tol);
        for (const auto& set :
             std::vector<void (*)(OptimizerSpec&, double)>{
                 [](OptimizerSpec& t, double v) { t.stop.ftol_rel = v; },
                 [](OptimizerSpec& t, double v) { t.stop.ftol_abs = v; },
                 [](OptimizerSpec& t, double v) { t.stop.xtol_rel = v; },
                 [](OptimizerSpec& t, double v) { t.stop.xtol_abs = v; }}) {
            auto t = crs;
            set(t, tol);
            CHECK_THAT(error_of(t),
                       Catch::StartsWith("NLOPT:GN_CRS2_LM needs a stop"));
            CHECK_FALSE(tadah::hpo::sets_own_stop(t));
            set(t, 1e-6);
            CHECK(error_of(t) == "");
            CHECK(tadah::hpo::sets_own_stop(t));
        }
    }

    /* An active RESTART's stagnation stop ends every phase it wraps: a
     * backend that needs a stop runs under it without one of its own. */
    using tadah::hpo::RestartStop;
    for (const auto& la : std::vector<std::pair<std::string, std::string>>{
             {"TADAH", "RANDOM"}, {"DLIB", "GFS"}, {"TADAH", "ANNEAL_PT"},
             {"NLOPT", "GN_DIRECT_L"}, {"NLOPT", "GN_ESCH"},
             {"NLOPT", "LD_MMA"}, {"NLOPT", "GN_MLSL_LDS"}}) {
        INFO(la.first << ":" << la.second << " under RESTART");
        auto t = local_spec(1);
        t.lib = la.first;
        t.algo = la.second;
        CHECK_THROWS(tadah::hpo::require_stop(t, RestartStop::Off));
        CHECK_NOTHROW(tadah::hpo::require_stop(t, RestartStop::On));
        CHECK_NOTHROW(tadah::hpo::validate_backend(t, RestartStop::On));
        CHECK_FALSE(tadah::hpo::sets_own_stop(t));
    }
    /* restart_stop_of(): active only with AFTER_STAGNATION and MAX > 0. */
    OptimizerSpec root;
    CHECK(tadah::hpo::restart_stop_of(root) == RestartStop::Off);
    root.restart.after_stagnation = 30;
    CHECK(tadah::hpo::restart_stop_of(root) == RestartStop::Off);
    root.restart.max = 2;
    CHECK(tadah::hpo::restart_stop_of(root) == RestartStop::On);
    root.restart.after_stagnation = 0;
    CHECK(tadah::hpo::restart_stop_of(root) == RestartStop::Off);

    /* Every other backend ends on its own: GRID after its grid, the local
     * optimisers on their convergence tests, ASA on its own limits, the
     * original DIRECT after 100000 evaluations, and ANNEAL and ANNEAL_PA at
     * the end of their schedules (ANNEAL's AUTO lengths need MAXEVAL). */
    for (const auto& la : std::vector<std::pair<std::string, std::string>>{
             {"TADAH", "GRID"}, {"DLIB", "BFGS"}, {"NLOPT", "LN_SBPLX"},
             {"NLOPT", "LN_COBYLA"}, {"NLOPT", "LD_LBFGS"},
             {"NLOPT", "GN_ORIG_DIRECT_L"}, {"NLOPT", "LN_AUGLAG"},
             {"CERES", "STEEPEST_DESCENT"}, {"ASA", "ASA"}, {"TADAH", "ANNEAL_PA"}}) {
        INFO(la.first << ":" << la.second);
        auto t = local_spec(1);
        t.lib = la.first;
        t.algo = la.second;
        /* without Ceres, CERES:STEEPEST_DESCENT is refused as not built
         * (ceres_build.h; a pin: before this test change such a build failed
         * here; the row named CERES:LBFGS, refused since) */
        CHECK(error_of(t) == (la.first == "CERES" ? hpo_test::ceres_expect("")
                                                  : std::string()));
    }
    auto anneal = local_spec(1);
    anneal.lib = "TADAH";
    anneal.algo = "ANNEAL";
    CHECK_THAT(error_of(anneal),
               /* It was "CHAIN_LENGTH AUTO needs MAXEVAL to derive the
                * per-rung budget; set PARAM CHAIN_LENGTH explicitly". */
               Catch::Contains("CHAIN_LENGTH AUTO spreads MAXEVAL over the "
                               "temperature steps, and this search has no "
                               "MAXEVAL"));
    anneal.param["CHAIN_LENGTH"] = 5;
    CHECK(error_of(anneal) == "");
}

/* Pins: GD_STOGO and GD_STOGO_RAND need MAXEVAL: require_stop refuses
 * STOPVAL alone, MAXTIME alone (with STOPVAL too) and an active RESTART,
 * naming "MAXEVAL (it honours no tolerance)", and sets_own_stop() is false
 * with STOPVAL, MAXTIME or both; MAXEVAL is enough, alone, beside MAXTIME
 * and under a RESTART. StoGO ignores NLopt's forced stop, by which HPO ends
 * a search (a signal, an error, the RESTART stagnation stop, a loss at or
 * below STOPVAL that NLopt was not given): it then calls on, each call
 * refused, until NLopt's own count ends it -- or its clock, only once the
 * rest of MAXTIME has passed.
 * Before the fix: MAXTIME alone was accepted (the message read "; set
 * MAXEVAL or MAXTIME (it honours no tolerance)"), and such a run waited out
 * the rest of its MAXTIME once HPO stopped it. */
TEST_CASE("backend rules: GD_STOGO needs MAXEVAL, the stop that ends it "
          "promptly when HPO stops it", "[HPO][backend_rules][budget]")
{
    using tadah::hpo::RestartStop;
    for (const std::string algo : {"GD_STOGO", "GD_STOGO_RAND"}) {
        INFO(algo);
        auto s = local_spec(1);
        s.lib = "NLOPT";
        s.algo = algo;
        const std::string ends = "; set MAXEVAL (it honours no tolerance)";
        CHECK_THAT(error_of(s),
                   Catch::StartsWith("NLOPT:" + algo + " needs a stop: ") &&
                   Catch::EndsWith(ends));
        CHECK_THROWS(tadah::hpo::require_stop(s, RestartStop::On));
        for (const auto& set : std::vector<void (*)(OptimizerSpec&)>{
                 [](OptimizerSpec& u) { u.stop.stopval = 0.1; },
                 [](OptimizerSpec& u) { u.stop.maxtime = 60; },
                 [](OptimizerSpec& u) {
                     u.stop.maxtime = 60;
                     u.stop.stopval = 0.1;
                 }}) {
            auto u = s;
            set(u);
            CHECK_THAT(error_of(u), Catch::EndsWith(ends));
            CHECK_FALSE(tadah::hpo::sets_own_stop(u));
            CHECK_THROWS(tadah::hpo::require_stop(u, RestartStop::On));
        }
        for (const auto& set : std::vector<void (*)(OptimizerSpec&)>{
                 [](OptimizerSpec& u) { u.stop.maxeval = 3; },
                 [](OptimizerSpec& u) {
                     u.stop.maxeval = 3;
                     u.stop.maxtime = 60;
                 }}) {
            auto u = s;
            u.stop.stopval = 0.1;
            set(u);
            CHECK(error_of(u) == "");
            CHECK(tadah::hpo::sets_own_stop(u));
            CHECK_NOTHROW(tadah::hpo::require_stop(u, RestartStop::On));
        }
    }
}

TEST_CASE("phase specs: the local search NLopt's MLSL and AUGLAG run",
          "[HPO][phase_specs][budget]")
{
    using tadah::hpo::inner_run_spec;
    const auto nlopt = [](const std::string& algo) {
        OptimizerSpec s;
        s.lib = "NLOPT";
        s.algo = algo;
        return s;
    };
    /* No local search: an algorithm without one, and one that needs an
     * INNER block it does not have. */
    CHECK_FALSE(inner_run_spec(nlopt("LN_SBPLX")));
    CHECK_FALSE(inner_run_spec(nlopt("AUGLAG")));
    CHECK_FALSE(inner_run_spec(nlopt("G_MLSL_LDS")));

    /* An INNER block: the tighter MAXEVAL and MAXTIME, the enclosing
     * STOPVAL (its own is replaced), its own tolerances. */
    auto auglag = nlopt("AUGLAG");
    auglag.stop.maxeval = 100;
    auglag.stop.maxtime = 30;
    auglag.stop.ftol_rel = 1e-3;  // the enclosing block's: not the INNER's
    auglag.inner = std::make_unique<OptimizerSpec>(nlopt("GN_DIRECT_L"));
    auglag.inner->stop.maxeval = 40;
    auglag.inner->stop.stopval = 0.5;
    auglag.inner->stop.xtol_rel = 1e-4;
    auto run = inner_run_spec(auglag);
    REQUIRE(run);
    CHECK(run->algo == "GN_DIRECT_L");
    CHECK(run->stop.maxeval == std::optional<std::uint64_t>(40));
    CHECK(run->stop.maxtime == std::optional<double>(30));
    CHECK_FALSE(run->stop.stopval);
    CHECK(run->stop.xtol_rel == std::optional<double>(1e-4));
    CHECK_FALSE(run->stop.ftol_rel);
    auglag.stop.stopval = 0.25;
    auglag.inner->stop.maxeval.reset();
    run = inner_run_spec(auglag);
    REQUIRE(run);
    CHECK(run->stop.maxeval == std::optional<std::uint64_t>(100));
    CHECK(run->stop.stopval == std::optional<double>(0.25));

    /* No INNER block: NLopt's default local optimiser, with the block's
     * tolerances. */
    auto ld = nlopt("LD_AUGLAG");
    ld.stop.ftol_rel = 1e-8;
    run = inner_run_spec(ld);
    REQUIRE(run);
    CHECK(run->lib == "NLOPT");
    CHECK(run->algo == "LD_MMA");
    CHECK(run->stop.ftol_rel == std::optional<double>(1e-8));
    run = inner_run_spec(nlopt("GN_MLSL_LDS"));
    REQUIRE(run);
    CHECK(run->algo == "LN_COBYLA");

    /* The INNER phase of the startup checks runs with it: EXPLORE's MAXEVAL
     * reaches its LN_SBPLX. */
    const auto phases = tadah::hpo::strategy_phase_specs(parse_spec(
        "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB DLIB\nALGO BFGS\n"
        "MAXEVAL 20\nEXPLORE\nLIB NLOPT\nALGO G_MLSL_LDS\n"
        "MAXEVAL 12\nINNER\nLIB NLOPT\nALGO LN_SBPLX\nENDINNER\n"
        "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n"));
    REQUIRE(phases.size() == 3u);
    CHECK(phases[2].spec.algo == "LN_SBPLX");
    CHECK(phases[2].spec.stop.maxeval == std::optional<std::uint64_t>(12));

    /* A local search that needs a stop, with none of its own or of the
     * enclosing block's. */
    auto bare = nlopt("AUGLAG");
    bare.low = {0.0};
    bare.high = {1.0};
    bare.analyse_logscale();
    bare.inner = std::make_unique<OptimizerSpec>(nlopt("GN_ESCH"));
    CHECK(error_of(bare) ==
          "the INNER search of NLOPT:AUGLAG (NLOPT:GN_ESCH) needs a stop: "
          "NLopt's global search has no convergence test of its own and runs "
          "until it is interrupted; set MAXEVAL or MAXTIME in the INNER block, "
          "or MAXEVAL, MAXTIME or STOPVAL in the NLOPT:AUGLAG block, which "
          "NLopt hands on to it");
    bare.stop.stopval = 0.1;
    CHECK(error_of(bare) == "");
    /* The RESTART stagnation stop ends the local search too. */
    bare.stop.stopval.reset();
    CHECK_THROWS(tadah::hpo::require_inner_run_stop(
        bare, tadah::hpo::RestartStop::Off));
    CHECK_NOTHROW(tadah::hpo::require_inner_run_stop(
        bare, tadah::hpo::RestartStop::On));
}

TEST_CASE("phase specs: the single local search a run is",
          "[HPO][phase_specs][auglag]")
{
    using tadah::hpo::single_local_search;
    const auto spec = [](const std::string& lib, const std::string& algo) {
        OptimizerSpec s;
        s.lib = lib;
        s.algo = algo;
        return s;
    };
    const auto nlopt = [&](const std::string& algo) {
        return spec("NLOPT", algo);
    };
    /* A local backend without a local search of its own is one. */
    for (const auto& s : {nlopt("LD_LBFGS"), nlopt("LN_SBPLX"),
                          spec("DLIB", "BFGS")}) {
        const auto local = single_local_search(s);
        REQUIRE(local);
        CHECK(local->lib == s.lib);
        CHECK(local->algo == s.algo);
    }
    /* A global backend is not: MLSL starts a local search from each of
     * many points, GD_STOGO samples the box. */
    auto mlsl = nlopt("G_MLSL_LDS");
    mlsl.inner = std::make_unique<OptimizerSpec>(nlopt("LD_LBFGS"));
    CHECK_FALSE(single_local_search(mlsl));
    CHECK_FALSE(single_local_search(nlopt("GD_MLSL_LDS")));
    CHECK_FALSE(single_local_search(nlopt("GD_STOGO")));

    /* AUGLAG (HPO sets no constraint) runs its local search once: the run
     * is that search, as inner_run_spec derives it. */
    auto local = single_local_search(nlopt("LD_AUGLAG"));
    REQUIRE(local);
    CHECK(local->algo == "LD_MMA");
    auto auglag = nlopt("AUGLAG");
    auglag.stop.maxeval = 60;
    auglag.inner = std::make_unique<OptimizerSpec>(nlopt("LD_SLSQP"));
    auglag.inner->stop.maxeval = 7;
    local = single_local_search(auglag);
    REQUIRE(local);
    CHECK(local->algo == "LD_SLSQP");
    CHECK(local->stop.maxeval == std::optional<std::uint64_t>(7));
    /* An AUGLAG inside it runs its own once in turn. */
    auglag.inner = std::make_unique<OptimizerSpec>(nlopt("LD_AUGLAG"));
    local = single_local_search(auglag);
    REQUIRE(local);
    CHECK(local->algo == "LD_MMA");
    CHECK(local->stop.maxeval == std::optional<std::uint64_t>(60));
    /* Not a local search: a global INNER, or none (NLopt refuses it). */
    auglag.inner = std::make_unique<OptimizerSpec>(nlopt("GN_ESCH"));
    CHECK_FALSE(single_local_search(auglag));
    CHECK_FALSE(single_local_search(nlopt("AUGLAG")));
}

/* Pins: obtains_gradients() is true exactly when a run asks the host for a
 * gradient. A backend without a local search of its own answers from its
 * caps (LD_*, GD_STOGO, DLIB:BFGS, CERES); NLopt's MLSL and AUGLAG evaluate
 * their own points without one (mlsl.c, auglag.c: grad NULL) and hand the
 * objective only the gradient their local search asks for, so their local
 * search decides (inner_run_spec): an INNER block, which replaces the
 * default the D or N of their name picks (LD_MMA or LN_COBYLA), or that
 * default. An INNER MLSL or AUGLAG of a default local search of its own is
 * decided by that one in turn.
 * Before the fix: true for GD_MLSL, GD_MLSL_LDS, LD_AUGLAG and LD_AUGLAG_EQ
 * with a derivative-free INNER block (their caps are is_gradient_based),
 * and for GD_MLSL_LDS with an INNER LN_AUGLAG; the other cases pass as
 * well (a pin of the rule the fix keeps). */
TEST_CASE("phase specs: a run obtains gradients when its search takes them",
          "[HPO][phase_specs][gradients]")
{
    using tadah::hpo::obtains_gradients;
    const auto spec = [](const std::string& lib, const std::string& algo) {
        OptimizerSpec s;
        s.lib = lib;
        s.algo = algo;
        return s;
    };
    const auto nlopt = [&](const std::string& algo) {
        return spec("NLOPT", algo);
    };
    const auto with_inner = [&](const std::string& algo,
                                const std::string& inner) {
        OptimizerSpec s = nlopt(algo);
        s.inner = std::make_unique<OptimizerSpec>(nlopt(inner));
        return s;
    };
    /* A backend without a local search of its own. */
    CHECK(obtains_gradients(nlopt("LD_LBFGS")));
    CHECK(obtains_gradients(nlopt("GD_STOGO")));
    CHECK(obtains_gradients(spec("DLIB", "BFGS")));
    CHECK(obtains_gradients(spec("CERES", "STEEPEST_DESCENT")));   /* was LBFGS */
    CHECK_FALSE(obtains_gradients(nlopt("LN_SBPLX")));
    CHECK_FALSE(obtains_gradients(nlopt("GN_CRS2_LM")));
    CHECK_FALSE(obtains_gradients(spec("TADAH", "RANDOM")));
    CHECK_FALSE(obtains_gradients(spec("DLIB", "GFS")));
    /* MLSL and AUGLAG without an INNER block: NLopt's default local
     * optimiser; AUGLAG has none (refused at startup). */
    for (const std::string algo :
         {"GD_MLSL", "GD_MLSL_LDS", "LD_AUGLAG", "LD_AUGLAG_EQ"}) {
        INFO(algo);
        CHECK(obtains_gradients(nlopt(algo)));
    }
    for (const std::string algo :
         {"GN_MLSL", "GN_MLSL_LDS", "LN_AUGLAG", "LN_AUGLAG_EQ", "AUGLAG",
          "G_MLSL_LDS"}) {
        INFO(algo);
        CHECK_FALSE(obtains_gradients(nlopt(algo)));
    }
    /* With an INNER block, that search alone, whatever the D or N of the
     * enclosing algorithm. */
    for (const std::string algo :
         {"GD_MLSL", "GD_MLSL_LDS", "LD_AUGLAG", "LD_AUGLAG_EQ", "GN_MLSL_LDS",
          "G_MLSL_LDS", "AUGLAG", "LN_AUGLAG"}) {
        INFO(algo);
        CHECK_FALSE(obtains_gradients(with_inner(algo, "LN_BOBYQA")));
        CHECK(obtains_gradients(with_inner(algo, "LD_LBFGS")));
    }
    /* An INNER MLSL or AUGLAG with a default local search: that one. */
    CHECK(obtains_gradients(with_inner("AUGLAG", "LD_AUGLAG")));
    CHECK_FALSE(obtains_gradients(with_inner("GD_MLSL_LDS", "LN_AUGLAG")));
    CHECK_FALSE(obtains_gradients(with_inner("LD_AUGLAG", "GN_MLSL_LDS")));
    CHECK(obtains_gradients(with_inner("GN_MLSL_LDS", "GD_MLSL")));
}

TEST_CASE("backend rules: the ALGO help lists the stops that end a run",
          "[HPO][backend_rules][help]")
{
    const auto help = [](const std::string& lib, const std::string& algo) {
        std::ostringstream out;
        std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
        OptimizerSpec::print_caps(lib, algo);
        std::cout.rdbuf(old);
        return out.str();
    };
    /* DLIB:GFS read no SOLVER_EPS or REL_NOISE PARAM, which its help
     * listed: it takes no PARAM, and FTOL_ABS is its solver epsilon. */
    const std::string gfs = help("DLIB", "GFS");
    CHECK_THAT(gfs, Catch::Contains("No PARAM") &&
                    Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                                    "STOPVAL\n") &&
                    Catch::Contains("One of them is required") &&
                    !Catch::Contains("SOLVER_EPS"));
    CHECK_THAT(help("TADAH", "RANDOM"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL\n") &&
               Catch::Contains("One of them is required"));
    CHECK_THAT(help("TADAH", "GRID"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL\n") &&
               !Catch::Contains("One of them is required"));
    CHECK_THAT(help("NLOPT", "GN_ESCH"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL\n") &&
               Catch::Contains("One of them is required"));
    /* DIRECT's XTOL is a stop, but a costly one; a backend that needs a
     * stop names the RESTART alternative, "a RESTART block" (every RESTART
     * block the parser takes restarts the search; it said "an active
     * RESTART"). */
    CHECK_THAT(help("NLOPT", "GN_DIRECT_L"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL, FTOL_REL, FTOL_ABS, XTOL_REL, "
                               "XTOL_ABS\n     DIRECT's XTOL_REL and XTOL_ABS "
                               "end it only once every rectangle is that "
                               "small, after roughly (1/XTOL_REL)^n "
                               "evaluations in n dimensions.\n") &&
               Catch::Contains("One of them is required (a tolerance above "
                               "0), or a RESTART block: "));
    CHECK_THAT(help("TADAH", "RANDOM"),
               Catch::Contains("One of them is required, or a RESTART "
                               "block: "));
    CHECK_THAT(help("NLOPT", "LN_SBPLX"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL, FTOL_REL, FTOL_ABS, XTOL_REL, "
                               "XTOL_ABS\n") &&
               !Catch::Contains("One of them is required"));
    /* Pins: GD_STOGO's help requires MAXEVAL and says why.
     * Before the fix: "MAXEVAL or MAXTIME is required (STOPVAL and an
     * active RESTART do not end it)". */
    CHECK_THAT(help("NLOPT", "GD_STOGO"),
               Catch::Contains("Stops that end a run: MAXEVAL, MAXTIME, "
                               "STOPVAL\n") &&
               Catch::Contains("MAXEVAL is required (MAXTIME, STOPVAL and "
                               "a RESTART block are not enough): ") &&
               !Catch::Contains("One of them is required") &&
               !Catch::Contains("MAXEVAL or MAXTIME"));
}

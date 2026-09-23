// test_hpo_parser_contracts.cpp
//
// Parser and message contracts of the --hpotarget grammar that the user
// documentation (Tadah.MLIP/docs/source/nfp_configuration.rst) states:
//   - LOSS HUBER / TUKEY defaults and validation, shared with --varloss
//     through LossFunction::resolve_params (regression: a bare
//     `LOSS HUBER` used to set delta = 0, i.e. an identically zero loss);
//   - the MULTISTART / PRESTAGE rejection messages name the real
//     constraint (regression: "set N_STARTS 1" although the check fires
//     at N_STARTS 1, and a reference to the removed PRESTAGE block);
//   - every PARAM key named in a CERES / DLIB:BFGS STEP hint is accepted
//     by that adapter (regression: the CERES hint named
//     max_line_search_step_size, which the adapter rejects);
//   - INIT METHOD (canonical) and its STRATEGY alias;
//   - OUTPUT / BOUTPUT / DUMP / BDUMP cadences reject 0;
//   - PC_WORKER_TIMEOUT parsing.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/run_policy.h>

#include <dlib/matrix.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TempFile {
    fs::path path;
    explicit TempFile(const std::string& body, const std::string& ext = ".hpo")
    : path(fs::temp_directory_path() /
           ("tadah_hpo_contract_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ext))
    {
        std::ofstream os(path);
        os << body;
        REQUIRE(os.good());
    }
    ~TempFile() { std::error_code ec; fs::remove_all(path, ec); }
};

const std::string kOptimizer =
    "OPTIMIZER\n"
    "STRATEGY\n"
    "  TYPE PLAIN\n"
    "  LIB DLIB\n"
    "  ALGO BFGS\n"
    "  MAXEVAL 5\n"
    "ENDSTRATEGY\n"
    "ENDOPTIMIZER\n";

/* Parse `root` (ROOT-level lines) after a minimal OPTIMIZER block. */
tadah::hpo::HPOTargets parse_root(const std::string& root)
{
    TempFile cfg(kOptimizer + root);
    tadah::core::Context ctx;
    return tadah::hpo::HPOTargets(ctx, cfg.path.string());
}

std::string parse_error(const std::string& body)
{
    TempFile cfg(body);
    tadah::core::Context ctx;
    try {
        tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

using tadah::LossFunction;

}  // namespace

/* ------------------------------------------------------------------ */
/*  LOSS parameters                                                    */
/* ------------------------------------------------------------------ */

TEST_CASE("LOSS HUBER/TUKEY without a parameter use the shared defaults",
          "[HPO][hpo_targets][loss]")
{
    {
        auto trg = parse_root("LOSS HUBER\n");
        CHECK(trg.globalLoss.type() == LossFunction::Type::HUBER);
        CHECK(trg.globalLoss.params() ==
              LossFunction::default_params(LossFunction::Type::HUBER));
        REQUIRE(trg.globalLoss.params().size() == 1);
        CHECK(trg.globalLoss.params()[0] == 50.0);
        /* delta = 50: residual 1 is in the quadratic region, 0.5*r^2.
         * The pre-fix parser set delta = 0 and returned 0 here. */
        CHECK(trg.globalLoss.loss(1.0, 0.0) == Approx(0.5));
    }
    {
        auto trg = parse_root("LOSS TUKEY\n");
        CHECK(trg.globalLoss.params() ==
              LossFunction::default_params(LossFunction::Type::TUKEY));
        REQUIRE(trg.globalLoss.params().size() == 1);
        CHECK(trg.globalLoss.params()[0] == 4.685);
        CHECK(trg.globalLoss.loss(1.0, 0.0) > 0.0);
    }
}

TEST_CASE("LOSS accepts an explicit positive parameter",
          "[HPO][hpo_targets][loss]")
{
    auto trg = parse_root("LOSS HUBER 2.5\n");
    REQUIRE(trg.globalLoss.params().size() == 1);
    CHECK(trg.globalLoss.params()[0] == 2.5);
    /* |r| = 4 > delta: linear branch delta*(|r| - delta/2). */
    CHECK(trg.globalLoss.loss(4.0, 0.0) == Approx(2.5 * (4.0 - 1.25)));

    auto trg2 = parse_root("LOSS TUKEY 3\n");
    CHECK(trg2.globalLoss.params() == std::vector<double>{3.0});
}

TEST_CASE("LOSS rejects non-positive / non-numeric / surplus parameters",
          "[HPO][hpo_targets][loss]")
{
    CHECK_THAT(parse_error(kOptimizer + "LOSS HUBER 0\n"),
               Catch::Contains("HUBER") && Catch::Contains("> 0"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS HUBER -1\n"),
               Catch::Contains("> 0"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS TUKEY 0\n"),
               Catch::Contains("TUKEY") && Catch::Contains("> 0"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS HUBER abc\n"),
               Catch::Contains("not a number"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS HUBER 1 2\n"),
               Catch::Contains("at most 1"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS L2 1\n"),
               Catch::Contains("takes no parameter"));
    CHECK_THAT(parse_error(kOptimizer + "LOSS\n"),
               Catch::Contains("LOSS requires a loss type"));
}

TEST_CASE("LossFunction::resolve_params is the single source of defaults",
          "[HPO][loss]")
{
    using T = LossFunction::Type;
    CHECK(LossFunction::resolve_params(T::HUBER, {}) ==
          LossFunction::default_params(T::HUBER));
    CHECK(LossFunction::resolve_params(T::TUKEY, {}) ==
          LossFunction::default_params(T::TUKEY));
    CHECK(LossFunction::resolve_params(T::L2, {}).empty());
    CHECK(LossFunction(T::HUBER).params() == LossFunction::default_params(T::HUBER));
    CHECK_THROWS_WITH(LossFunction(T::HUBER, {0.0}), Catch::Contains("> 0"));
    CHECK_THROWS_WITH(LossFunction(T::TUKEY,
                                   {std::numeric_limits<double>::infinity()}),
                      Catch::Contains("finite"));
    CHECK_THROWS_WITH(LossFunction(T::MAPE, {1.0}),
                      Catch::Contains("takes no parameter"));
}

TEST_CASE("PC_LAMMPS --varloss loss override shares the LOSS defaults and "
          "validation", "[HPO][lammps_sim][loss]")
{
    TempFile script("variable x equal 1.0\n", ".in");
    const std::string head = "--script " + script.path.string() +
                             " --varloss x 0.0 1.0 ";
    /* LammpsSim takes the command line by non-const reference. */
    auto make_sim = [](std::string cmd, const LossFunction& root) {
        return tadah::hpo::LammpsSim(cmd, "off", 1e6, root);
    };
    {
        auto sim = make_sim(head + "HUBER", LossFunction{});
        REQUIRE(sim.lossFn.size() == 1);
        CHECK(sim.lossFn[0].type() == LossFunction::Type::HUBER);
        CHECK(sim.lossFn[0].params() ==
              LossFunction::default_params(LossFunction::Type::HUBER));
    }
    {
        auto sim = make_sim(head + "TUKEY 2", LossFunction{});
        CHECK(sim.lossFn[0].params() == std::vector<double>{2.0});
    }
    CHECK_THROWS_WITH(make_sim(head + "HUBER 0", LossFunction{}),
                      Catch::Contains("> 0"));
    /* Without an override the sim inherits the root LOSS verbatim. */
    {
        auto sim = make_sim(head, LossFunction(LossFunction::Type::HUBER, {7.0}));
        CHECK(sim.lossFn[0].params() == std::vector<double>{7.0});
    }
}

/* ------------------------------------------------------------------ */
/*  Strategy rejection messages                                        */
/* ------------------------------------------------------------------ */

TEST_CASE("MULTISTART rejection of a single-seed global states the real "
          "constraint", "[HPO][run_policy][multistart]")
{
    using V = dlib::matrix<double, 0, 1>;
    for (const auto& la : std::vector<std::pair<std::string, std::string>>{
             {"NLOPT", "GN_DIRECT"}, {"TADAH", "RANDOM"}, {"TADAH", "ANNEAL"},
             {"ASA", "ASA"}}) {
        INFO(la.first << ":" << la.second);
        tadah::hpo::OptimizerSpec sp;
        sp.lib  = la.first;
        sp.algo = la.second;
        sp.strategy_type = tadah::hpo::StrategyType::MULTISTART;
        sp.multistart.n_starts = 1;   // the check fires even here
        std::string msg;
        try {
            tadah::hpo::MultiStartPolicy<V, int>::validate(sp);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        REQUIRE_FALSE(msg.empty());
        CHECK_THAT(msg, Catch::Contains("any N_STARTS"));
        CHECK_THAT(msg, Catch::Contains("TYPE PLAIN"));
        CHECK_THAT(msg, !Catch::Contains("set N_STARTS 1"));
    }
    /* A multi-seed global (DLIB:GFS) stays accepted. */
    tadah::hpo::OptimizerSpec gfs;
    gfs.lib = "DLIB"; gfs.algo = "GFS";
    CHECK_NOTHROW(tadah::hpo::MultiStartPolicy<V, int>::validate(gfs));
}

TEST_CASE("PRESTAGE without EXPLORE names the current grammar",
          "[HPO][run_policy][prestage]")
{
    using V = dlib::matrix<double, 0, 1>;
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "NLOPT"; sp.algo = "LN_SBPLX";
    sp.strategy_type = tadah::hpo::StrategyType::PRESTAGE;
    std::string msg;
    try {
        tadah::hpo::StagedPolicy<V, int>::validate(sp);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    REQUIRE_FALSE(msg.empty());
    CHECK_THAT(msg, Catch::Contains("EXPLORE ... ENDEXPLORE"));
    CHECK_THAT(msg, !Catch::Contains("ENDPRESTAGE"));

    sp.prestage = std::make_unique<tadah::hpo::OptimizerSpec>();
    CHECK_NOTHROW(tadah::hpo::StagedPolicy<V, int>::validate(sp));
}

TEST_CASE("Removed PRESTAGE block keyword points at STRATEGY TYPE PRESTAGE",
          "[HPO][hpo_targets][prestage]")
{
    const std::string msg = parse_error(
        "OPTIMIZER\nPRESTAGE\nENDPRESTAGE\nENDOPTIMIZER\n");
    CHECK_THAT(msg, Catch::Contains("TYPE PRESTAGE"));
    CHECK_THAT(msg, Catch::Contains("REFINE is optional"));
}

/* ------------------------------------------------------------------ */
/*  STEP hints name PARAM keys the adapter accepts                     */
/* ------------------------------------------------------------------ */

namespace {
std::vector<std::string> params_named_in(const std::string& note)
{
    /* Every "PARAM <key>" occurrence, wherever it sits in the prose. */
    std::vector<std::string> out;
    const std::string tag = "PARAM ";
    for (auto pos = note.find(tag); pos != std::string::npos;
         pos = note.find(tag, pos + tag.size())) {
        std::size_t b = pos + tag.size(), e = b;
        while (e < note.size() &&
               (std::isalnum(static_cast<unsigned char>(note[e])) || note[e] == '_'))
            ++e;
        if (e > b) out.push_back(note.substr(b, e - b));
    }
    return out;
}

tadah::hpo::OptimizerSpec bounded_spec(const std::string& lib,
                                       const std::string& algo)
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = lib; sp.algo = algo;
    sp.low = {0.0, 0.0}; sp.high = {1.0, 1.0};
    sp.is_int = {false, false};
    sp.optim_init = {{0.5, 0.5}};
    return sp;
}
}  // namespace

TEST_CASE("Every PARAM key named in a CERES / DLIB:BFGS STEP hint is "
          "accepted by that adapter", "[HPO][optimizer_caps][step]")
{
    for (const char* algo : {"LBFGS", "WOLFE", "NLCG", "STEEPEST_DESCENT"}) {
        INFO("CERES:" << algo);
        const auto& caps = tadah::hpo::optimizer_caps("CERES", algo);
        CHECK_THAT(caps.step_note, !Catch::Contains("max_line_search_step_size"));
        const auto keys = params_named_in(caps.step_note);
        REQUIRE_FALSE(keys.empty());
        for (const auto& k : keys) {
            INFO("PARAM " << k);
            auto sp = bounded_spec("CERES", algo);
            sp.param[k] = 5.0;
            const std::string a = algo;
            if (a == "LBFGS")
                CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_lbfgs_spec(sp));
            else if (a == "WOLFE")
                CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_wolfe_spec(sp));
            else if (a == "NLCG")
                CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_nlcg_spec(sp));
            else
                CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_steepest_spec(sp));
        }
    }
    {
        /* The STEP rejection itself must not advertise a rejected key. */
        auto sp = bounded_spec("CERES", "LBFGS");
        sp.step = 0.1;
        std::string msg;
        try {
            tadah::hpo::detail::validate_ceres_lbfgs_spec(sp);
        } catch (const std::exception& e) { msg = e.what(); }
        CHECK_THAT(msg, Catch::Contains("does not support STEP"));
        CHECK_THAT(msg, Catch::Contains("max_ls_iter"));
        CHECK_THAT(msg, !Catch::Contains("max_line_search_step_size"));
    }
    {
        const auto& caps = tadah::hpo::optimizer_caps("DLIB", "BFGS");
        const auto keys = params_named_in(caps.step_note);
        REQUIRE_FALSE(keys.empty());
        for (const auto& k : keys) {
            INFO("DLIB:BFGS PARAM " << k);
            auto sp = bounded_spec("DLIB", "BFGS");
            sp.param[k] = 1.0e-4;
            CHECK_NOTHROW(tadah::hpo::detail::validate_dlib_bfgs_spec(sp));
        }
    }
}

/* ------------------------------------------------------------------ */
/*  INIT METHOD                                                        */
/* ------------------------------------------------------------------ */

TEST_CASE("INIT METHOD is the canonical chain keyword and STRATEGY its alias",
          "[HPO][hpo_targets][init_strategy]")
{
    const std::string body =
        "OPTIMIZER\n"
        "INIT\n"
        "  METHOD CONFIG\n"
        "  STRATEGY RANDOM 3\n"
        "  METHOD LHS 2\n"
        "ENDINIT\n"
        "STRATEGY\n"
        "  TYPE PLAIN\n  LIB DLIB\n  ALGO BFGS\n"
        "  RESTART\n"
        "    AFTER_STAGNATION 10\n"
        "    MAX 2\n"
        "    METHOD SOBOL 4\n"
        "  ENDRESTART\n"
        "ENDSTRATEGY\n"
        "ENDOPTIMIZER\n";
    TempFile cfg(body);
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());

    const auto& chain = trg.spec.init_strategy_spec.chain;
    REQUIRE(chain.size() == 3);
    CHECK(chain[0].method == "CONFIG");
    CHECK(chain[1].method == "RANDOM");
    CHECK(chain[1].args == std::vector<std::string>{"3"});
    CHECK(chain[2].method == "LHS");
    CHECK(chain[2].args == std::vector<std::string>{"2"});

    const auto& rs = trg.spec.restart;
    CHECK(rs.after_stagnation == 10u);
    CHECK(rs.max == 2u);
    REQUIRE(rs.init_strategy.chain.size() == 1);
    CHECK(rs.init_strategy.chain[0].method == "SOBOL");

    CHECK_THAT(parse_error("OPTIMIZER\nINIT\n  METHOD\nENDINIT\nENDOPTIMIZER\n"),
               Catch::Contains("METHOD requires a method name"));
}

/* ------------------------------------------------------------------ */
/*  Output cadences                                                    */
/* ------------------------------------------------------------------ */

TEST_CASE("OUTPUT / BOUTPUT / DUMP / BDUMP reject a zero cadence",
          "[HPO][hpo_targets][output]")
{
    CHECK_THAT(parse_error(kOptimizer + "OUTPUT 0\n"),
               Catch::Contains("OUTPUT must be >= 1"));
    CHECK_THAT(parse_error(kOptimizer + "BOUTPUT 0\n"),
               Catch::Contains("BOUTPUT must be >= 1"));
    CHECK_THAT(parse_error(kOptimizer + "DUMP 0 pots\n"),
               Catch::Contains("DUMP requires a cadence >= 1"));
    CHECK_THAT(parse_error(kOptimizer + "BDUMP 0 best_pots\n"),
               Catch::Contains("BDUMP requires a cadence >= 1"));
    /* The directory is required, not defaulted. */
    CHECK_THAT(parse_error(kOptimizer + "DUMP 5\n"),
               Catch::Contains("and a directory"));
}

TEST_CASE("Output cadence defaults: every eval / every best / no archives",
          "[HPO][hpo_targets][output]")
{
    auto trg = parse_root("");
    CHECK(trg.outputEvery == 1u);
    CHECK(trg.bestOutputEvery == 1u);
    CHECK(trg.dumpPotEvery == std::numeric_limits<std::size_t>::max());
    CHECK(trg.dumpBestPotEvery == std::numeric_limits<std::size_t>::max());

    auto trg2 = parse_root("OUTPUT 7\nBOUTPUT 3\n");
    CHECK(trg2.outputEvery == 7u);
    CHECK(trg2.bestOutputEvery == 3u);
}

/* ------------------------------------------------------------------ */
/*  PC_WORKER_TIMEOUT                                                  */
/* ------------------------------------------------------------------ */

TEST_CASE("PC_WORKER_TIMEOUT parses seconds and rejects invalid values",
          "[HPO][hpo_targets][pc_worker_timeout]")
{
    CHECK(parse_root("").PC_WORKER_TIMEOUT == 0.0);
    CHECK(parse_root("PC_WORKER_TIMEOUT 45.5\n").PC_WORKER_TIMEOUT == 45.5);
    CHECK(parse_root("PC_WORKER_TIMEOUT 0\n").PC_WORKER_TIMEOUT == 0.0);
    CHECK_THAT(parse_error(kOptimizer + "PC_WORKER_TIMEOUT -1\n"),
               Catch::Contains("PC_WORKER_TIMEOUT must be a finite number"));
    CHECK_THAT(parse_error(kOptimizer + "PC_WORKER_TIMEOUT soon\n"),
               Catch::Contains("got 'soon'"));
    CHECK_THAT(parse_error(kOptimizer + "PC_WORKER_TIMEOUT\n"),
               Catch::Contains("requires a value"));
    /* The environment override uses the same parser. */
    CHECK(tadah::hpo::HPOTargets::parse_pc_worker_timeout("12", "X") == 12.0);
    CHECK_THROWS_WITH(
        tadah::hpo::HPOTargets::parse_pc_worker_timeout("inf",
                                                        "TADAH_PC_WORKER_TIMEOUT"),
        Catch::Contains("TADAH_PC_WORKER_TIMEOUT"));
}

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
//   - PC_WORKER_TIMEOUT parsing;
//   - STRATEGY TYPE is PLAIN, MULTISTART or PRESTAGE: the undocumented
//     BASINS / BASINSV2 strategies, their knobs and their pre-release
//     keyword hints are gone, no error text names them, and
//     build_run_policy still wraps the remaining strategies in
//     RestartPolicy when RESTART is set.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <algorithm>
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

/* ------------------------------------------------------------------ */
/*  STRATEGY TYPE: PLAIN, MULTISTART, PRESTAGE (BASINS removed)         */
/* ------------------------------------------------------------------ */

namespace {

/* The parser's own wording: the "Error on line N: ..." diagnostic without
 * the echoed config line and without the user's quoted token, so a check
 * on it cannot be satisfied or broken by what the user typed. */
std::string own_wording(const std::string& msg, const std::string& token)
{
    std::string s = msg.substr(0, msg.find("\nLine: "));
    const std::string quoted = "'" + token + "'";
    for (auto p = s.find(quoted); p != std::string::npos; p = s.find(quoted))
        s.erase(p, quoted.size());
    return s;
}

bool names_basins(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s.find("BASIN") != std::string::npos;
}

std::string strategy_block(const std::string& type, const std::string& body)
{
    return "OPTIMIZER\nSTRATEGY\n  TYPE " + type + "\n" + body +
           "ENDSTRATEGY\nENDOPTIMIZER\n";
}

/* PRESTAGE with an EXPLORE block; `explore` and `refine` are extra lines
 * appended inside the respective block. */
std::string prestage_block(const std::string& explore,
                           const std::string& refine = "")
{
    std::string s =
        "OPTIMIZER\n"                     // 1
        "STRATEGY\n"                      // 2
        "  TYPE PRESTAGE\n"               // 3
        "  EXPLORE\n"                     // 4
        "    LIB NLOPT\n"                 // 5
        "    ALGO GN_CRS2_LM\n"           // 6
        "    MAXEVAL 20\n"                // 7
        + explore +                       // 8...
        "  ENDEXPLORE\n";
    if (!refine.empty())
        s += "  REFINE\n    LIB NLOPT\n    ALGO LN_BOBYQA\n" + refine +
             "  ENDREFINE\n";
    return s + "ENDSTRATEGY\nENDOPTIMIZER\n";
}

/* Pre-release keywords (never accepted by a tagged release) whose
 * "removed from the grammar" hints pointed only at BASINS keys. */
const std::vector<std::string> kDroppedBasinsEraKeys = {
    "GLOBAL_LIB", "GLOBAL_ALGO", "LOCAL_LIB", "LOCAL_ALGO",
    "TOL_GLO_LADDER", "TOL_LOC_LADDER",
    "STAGE_RATE_FACTOR", "STAGE_MIN_EVALS", "STAGE_MAX_EVALS",
    "EMERGENCY_NO_IMPROVE_ROUNDS", "GLOBAL_SEED_TOP_N",
    "CLAMP_LADDER_TO_NOISE_FLOOR"};

/* STRATEGY-level knobs shared by the removed BASINS / BASINSV2 strategies.
 * Before the removal the parser recognised them under every TYPE and
 * answered "<KEY> requires TYPE BASINS or BASINSV2" outside those two. */
const std::vector<std::string> kRemovedStrategyKnobs = {
    "BASIN_POOL_MAX 8", "BASIN_DEDUP_DIST 0.05", "STOP_NO_IMPROVE_ROUNDS 3",
    "SEED_TOP_N 2", "CLAMP_TO_NOISE_FLOOR true", "CHECKPOINT_EVERY 60s"};

/* BASINSV2-only knobs. The parser read them only after TYPE BASINSV2, so
 * under PLAIN / MULTISTART / PRESTAGE they already got the generic
 * unknown-key error before the removal: these entries guard against their
 * return, they do not demonstrate the removal. */
const std::vector<std::string> kRemovedBasinsV2OnlyKnobs = {
    "BASIN_MERGE_DIST 0.1", "STOP_NO_IMPROVE_ITERATIONS 200",
    "PRIORITY_WEIGHTS 2 0.5 0.3 0.5 1 0.05", "VERBOSE_DIAGNOSTICS true"};

std::string first_token(const std::string& line)
{
    return line.substr(0, line.find(' '));
}

}  // namespace

TEST_CASE("STRATEGY TYPE BASINS and BASINSV2 are rejected with the "
          "PLAIN/MULTISTART/PRESTAGE list", "[HPO][hpo_targets][strategy]")
{
    for (const std::string t : {"BASINS", "BASINSV2", "BOGUS"}) {
        INFO("TYPE " << t);
        const std::string msg = parse_error(strategy_block(t, ""));
        CHECK_THAT(msg, Catch::Contains("Error on line 3"));
        CHECK_THAT(msg, Catch::Contains(
            "STRATEGY TYPE must be PLAIN, MULTISTART, or PRESTAGE (got '" +
            t + "')"));
        CHECK_FALSE(names_basins(own_wording(msg, t)));
    }

    using tadah::hpo::StrategyType;
    const std::vector<std::pair<std::string, StrategyType>> accepted = {
        {"PLAIN", StrategyType::PLAIN},
        {"MULTISTART", StrategyType::MULTISTART},
        {"PRESTAGE", StrategyType::PRESTAGE}};
    for (const auto& [name, type] : accepted) {
        INFO("TYPE " << name);
        TempFile cfg(strategy_block(name, "  LIB NLOPT\n  ALGO LN_BOBYQA\n"));
        tadah::core::Context ctx;
        tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
        CHECK(trg.spec.strategy_type == type);
    }
}

TEST_CASE("LOCAL keeps its rename hint to INNER without naming BASINS",
          "[HPO][hpo_targets][strategy]")
{
    /* LOCAL/ENDLOCAL shipped in v1.0.0 and v1.1.1, so their hints stay. */
    const std::string local = parse_error("OPTIMIZER\nLOCAL\nENDOPTIMIZER\n");
    CHECK_THAT(local, Catch::Contains(
        "'LOCAL' was removed from the grammar: renamed to INNER"));
    CHECK_THAT(local, Catch::Contains("MLSL/AUGLAG"));
    CHECK_FALSE(names_basins(own_wording(local, "LOCAL")));

    const std::string endlocal =
        parse_error("OPTIMIZER\nENDLOCAL\nENDOPTIMIZER\n");
    CHECK_THAT(endlocal, Catch::Contains(
        "'ENDLOCAL' was removed from the grammar: renamed to ENDINNER"));
}

TEST_CASE("Pre-release BASINS-era keywords get the generic unknown-key error",
          "[HPO][hpo_targets][strategy]")
{
    for (const auto& k : kDroppedBasinsEraKeys) {
        INFO(k);
        const std::string outer =
            parse_error("OPTIMIZER\n" + k + " 1\nENDOPTIMIZER\n");
        CHECK_THAT(outer, Catch::Contains("Error on line 2"));
        CHECK_THAT(outer, Catch::Contains("unknown key '" + k + "' in this block"));
        CHECK_THAT(outer, !Catch::Contains("was removed from the grammar"));

        const std::string in_strategy =
            parse_error(strategy_block("PLAIN", "  " + k + " 1\n"));
        CHECK_THAT(in_strategy, Catch::Contains(
            "unknown key '" + k + "' inside STRATEGY block"));
    }
}

TEST_CASE("EXPLORE and REFINE blocks name TYPE PRESTAGE as their only home",
          "[HPO][hpo_targets][strategy][prestage]")
{
    for (const std::string t : {"PLAIN", "MULTISTART"}) {
        INFO("TYPE " << t);
        const std::string explore =
            parse_error(strategy_block(t, "  EXPLORE\n  ENDEXPLORE\n"));
        CHECK_THAT(explore, Catch::Contains("Error on line 4"));
        CHECK_THAT(explore, Catch::Contains(
            "EXPLORE block is only valid with TYPE PRESTAGE "
            "(declare TYPE before EXPLORE)"));
        const std::string refine =
            parse_error(strategy_block(t, "  REFINE\n  ENDREFINE\n"));
        CHECK_THAT(refine, Catch::Contains(
            "REFINE block is only valid with TYPE PRESTAGE "
            "(declare TYPE before REFINE)"));
    }
    /* Without a TYPE line the strategy is PLAIN. */
    CHECK_THAT(parse_error("OPTIMIZER\nSTRATEGY\n  EXPLORE\n  ENDEXPLORE\n"
                           "ENDSTRATEGY\nENDOPTIMIZER\n"),
               Catch::Contains("EXPLORE block is only valid with TYPE PRESTAGE"));
}

TEST_CASE("BASINS ladder/budget/pool knobs are unknown keys",
          "[HPO][hpo_targets][strategy][prestage]")
{
    SECTION("FTOL_LADDER and BUDGET_* inside a PRESTAGE EXPLORE block") {
        for (const std::string line : {"FTOL_LADDER 1e-3 1e-4",
                                       "BUDGET_MAX_EVALS 100",
                                       "BUDGET_MIN_EVALS 10",
                                       "BUDGET_RATE_FACTOR 0.5"}) {
            INFO(line);
            const std::string k = first_token(line);
            const std::string msg =
                parse_error(prestage_block("    " + line + "\n"));
            CHECK_THAT(msg, Catch::Contains("Error on line 8"));
            CHECK_THAT(msg, Catch::Contains(
                "unknown key '" + k + "' inside EXPLORE block"));
        }
    }
    SECTION("FTOL_LADDER and BUDGET_* inside a PRESTAGE REFINE block") {
        for (const std::string line : {"FTOL_LADDER 1e-5",
                                       "BUDGET_MAX_EVALS 100"}) {
            INFO(line);
            const std::string k = first_token(line);
            const std::string msg =
                parse_error(prestage_block("", "    " + line + "\n"));
            CHECK_THAT(msg, Catch::Contains(
                "unknown key '" + k + "' inside REFINE block"));
        }
    }
    SECTION("pool, stop and checkpoint knobs inside STRATEGY") {
        for (const std::string t : {"PLAIN", "MULTISTART", "PRESTAGE"}) {
            for (const auto* knobs : {&kRemovedStrategyKnobs,
                                      &kRemovedBasinsV2OnlyKnobs}) {
                for (const auto& line : *knobs) {
                    INFO("TYPE " << t << ": " << line);
                    const std::string k = first_token(line);
                    const std::string msg =
                        parse_error(strategy_block(t, "  " + line + "\n"));
                    CHECK_THAT(msg, Catch::Contains("Error on line 4"));
                    CHECK_THAT(msg, Catch::Contains(
                        "unknown key '" + k + "' inside STRATEGY block"));
                }
            }
        }
    }
}

TEST_CASE("No parser error text names the removed BASINS strategies",
          "[HPO][hpo_targets][strategy]")
{
    /* Every strategy-related rejection the grammar can produce, each paired
     * with the token it echoes back to the user. */
    std::vector<std::pair<std::string, std::string>> cases = {
        {strategy_block("BASINS", ""), "BASINS"},
        {strategy_block("BASINSV2", ""), "BASINSV2"},
        {strategy_block("BOGUS", ""), "BOGUS"},
        {"OPTIMIZER\nLOCAL\nENDOPTIMIZER\n", "LOCAL"},
        {"OPTIMIZER\nENDLOCAL\nENDOPTIMIZER\n", "ENDLOCAL"},
        {strategy_block("PLAIN", "  EXPLORE\n  ENDEXPLORE\n"), "EXPLORE"},
        {strategy_block("MULTISTART", "  REFINE\n  ENDREFINE\n"), "REFINE"},
        {"OPTIMIZER\nEXPLORE\nENDEXPLORE\nENDOPTIMIZER\n", "EXPLORE"},
        {"OPTIMIZER\nREFINE\nENDREFINE\nENDOPTIMIZER\n", "REFINE"},
        {"OPTIMIZER\nINNER\nENDINNER\nENDOPTIMIZER\n", "INNER"},
        {prestage_block("    FTOL_LADDER 1e-3\n"), "FTOL_LADDER"},
        {prestage_block("", "    BUDGET_RATE_FACTOR 0.5\n"), "BUDGET_RATE_FACTOR"},
    };
    for (const auto& k : kDroppedBasinsEraKeys)
        cases.emplace_back("OPTIMIZER\n" + k + " 1\nENDOPTIMIZER\n", k);
    for (const auto* knobs : {&kRemovedStrategyKnobs, &kRemovedBasinsV2OnlyKnobs})
        for (const auto& line : *knobs)
            cases.emplace_back(strategy_block("PLAIN", "  " + line + "\n"),
                               first_token(line));

    /* Every case is checked: one that parses (no error to inspect) fails
     * without hiding the wording checks of the cases after it. */
    for (const auto& [body, token] : cases) {
        INFO(body);
        const std::string msg = parse_error(body);
        CHECK_FALSE(msg.empty());
        if (msg.empty()) continue;
        CHECK_FALSE(names_basins(own_wording(msg, token)));
    }
}

TEST_CASE("build_run_policy wraps PLAIN/MULTISTART/PRESTAGE in RestartPolicy "
          "when RESTART is set", "[HPO][run_policy][restart]")
{
    /* This pins the policy composition. That a restart then fires under
     * NLopt, MULTISTART and PRESTAGE is checked end to end by the
     * [restart][termination] cases in test_hpo_termination.cpp. */
    using V = dlib::matrix<double, 0, 1>;
    using R = ::LammpsRunner;
    const std::string restart =
        "  RESTART\n"
        "    AFTER_STAGNATION 10\n"
        "    MAX 2\n"
        "    METHOD RANDOM 1\n"
        "  ENDRESTART\n";
    const std::string explore =
        "  EXPLORE\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n    MAXEVAL 20\n"
        "  ENDEXPLORE\n";
    for (const std::string t : {"PLAIN", "MULTISTART", "PRESTAGE"}) {
        INFO("TYPE " << t);
        const std::string common = "  LIB NLOPT\n  ALGO LN_SBPLX\n  MAXEVAL 50\n" +
            std::string(t == "PRESTAGE" ? explore : "");
        {
            TempFile cfg(strategy_block(t, common + restart));
            tadah::core::Context ctx;
            tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
            CHECK(trg.spec.restart.after_stagnation == 10);
            CHECK(trg.spec.restart.max == 2);
            REQUIRE(trg.spec.restart.init_strategy.chain.size() == 1);
            CHECK(trg.spec.restart.init_strategy.chain.front().method == "RANDOM");
            const auto policy = tadah::hpo::build_run_policy<V, R>(trg.spec, &trg);
            REQUIRE(policy);
            CHECK(dynamic_cast<tadah::hpo::RestartPolicy<V, R>*>(policy.get()));
        }
        {
            /* Without RESTART the strategy's own policy is returned. */
            TempFile cfg(strategy_block(t, common));
            tadah::core::Context ctx;
            tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
            const auto policy = tadah::hpo::build_run_policy<V, R>(trg.spec, &trg);
            REQUIRE(policy);
            CHECK_FALSE(dynamic_cast<tadah::hpo::RestartPolicy<V, R>*>(policy.get()));
            if (t == "PLAIN")
                CHECK(dynamic_cast<tadah::hpo::BasePolicy<V, R>*>(policy.get()));
            if (t == "MULTISTART")
                CHECK(dynamic_cast<tadah::hpo::MultiStartPolicy<V, R>*>(policy.get()));
            if (t == "PRESTAGE")
                CHECK(dynamic_cast<tadah::hpo::StagedPolicy<V, R>*>(policy.get()));
        }
    }
}

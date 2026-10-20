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
//   - strict numbers: every numeric key (root, STRATEGY, RESTART, INIT,
//     NOISE, the optimiser attributes, OPTIM bounds, PC_FAILPROBE and the
//     PC_LAMMPS options) rejects a malformed, non-finite, signed,
//     fractional or out-of-range value, a missing value and an extra
//     token with its own message at its own line; the timeouts are LAMMPS
//     time specs and a limit <= 0 is rejected ('off' and 'unlimited' are
//     no limit), and so is one whose worker deadline does not fit in
//     64-bit milliseconds; THREADS is at least 1 in every optimiser block;
//     a PC_LAMMPS option takes only its own values, and the LAMMPS
//     variable maxtime is reserved for the script timeout; a number too
//     small for a double says so, and PC_<METRIC> errors name the key;
//     `tadah refit` OPTIMIZER blocks reject SEED, and EXPLORE MAX_ADD is
//     an integer >= 0;
//   - the density gates RHOGATE and RHOBARGATE read their numbers in the
//     same grammar (NPTS an integer, the rest finite decimal numbers);
//   - STRATEGY TYPE is PLAIN, MULTISTART or PRESTAGE: the undocumented
//     BASINS / BASINSV2 strategies, their knobs and their pre-release
//     keyword hints are gone, no error text names them, and
//     build_run_policy still wraps the remaining strategies in
//     RestartPolicy when RESTART is set.

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/refit/refit_parser.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <tuple>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

/* A directory created at unique_temp_path(@p ext), as the operating system
 * names it (symbolic links in the temporary directory's path resolved):
 * the archive-directory messages print physical paths. */
fs::path physical_temp_dir(const std::string& ext);

/* Unique per process and per call: ctest runs the cases of this file as
 * concurrent processes. Removed on scope exit. */
fs::path unique_temp_path(const std::string& ext)
{
    static unsigned counter = 0;
    return fs::temp_directory_path() /
           ("tadah_hpo_contract_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter++) + "_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ext);
}

fs::path physical_temp_dir(const std::string& ext)
{
    const fs::path dir = unique_temp_path(ext);
    fs::create_directories(dir);
    return fs::canonical(dir);
}

struct TempFile {
    fs::path path;
    explicit TempFile(const std::string& body, const std::string& ext = ".hpo")
    : path(unique_temp_path(ext))
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
    sp.prestage->lib = "DLIB";
    sp.prestage->algo = "GFS";
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
    /* Before the fix: passes as well (a pin; it also checked CERES:LBFGS,
     * WOLFE and NLCG, refused since). */
    for (const char* algo : {"STEEPEST_DESCENT"}) {
        INFO("CERES:" << algo);
        const auto& caps = tadah::hpo::optimizer_caps("CERES", algo);
        CHECK_THAT(caps.step_note, !Catch::Contains("max_line_search_step_size"));
        const auto keys = params_named_in(caps.step_note);
        REQUIRE_FALSE(keys.empty());
        for (const auto& k : keys) {
            INFO("PARAM " << k);
            auto sp = bounded_spec("CERES", algo);
            sp.param[k] = 5.0;
            CHECK_NOTHROW(tadah::hpo::detail::validate_ceres_steepest_spec(sp));
        }
    }
    {
        /* The STEP rejection itself must not advertise a rejected key. */
        auto sp = bounded_spec("CERES", "STEEPEST_DESCENT");
        sp.step = 0.1;
        std::string msg;
        try {
            tadah::hpo::detail::validate_ceres_steepest_spec(sp);
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
               Catch::Contains("OUTPUT must be an integer >= 1, got '0'"));
    CHECK_THAT(parse_error(kOptimizer + "BOUTPUT 0\n"),
               Catch::Contains("BOUTPUT must be an integer >= 1, got '0'"));
    CHECK_THAT(parse_error(kOptimizer + "DUMP 0 pots\n"),
               Catch::Contains("DUMP cadence must be an integer >= 1, got '0'"));
    CHECK_THAT(parse_error(kOptimizer + "BDUMP 0 best_pots\n"),
               Catch::Contains("BDUMP cadence must be an integer >= 1, got '0'"));
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

/* ------------------------------------------------------------------ */
/*  Strict numbers                                                     */
/* ------------------------------------------------------------------ */
/* Every numeric key rejects what it cannot hold, with its own message at
 * its own line ("Error on line N: ..."): a non-number, trailing junk, a
 * value the key's type cannot represent (a sign, fraction or exponent for
 * an integer, inf/nan/overflow for a real), a value outside its range, a
 * missing value and an extra token. `iss >> v` accepted most of these
 * silently: `FAILSCORE abc` scored every failure 0, `MAX abc` disabled
 * RESTART, `N_STARTS` without a value read an uninitialised variable,
 * `N_STARTS -2` wrapped to 2^64-2, `MAXEVAL 1e5` read 1 then complained
 * about 'e5'. */

namespace {

/* `KEY <value>` on a line of its own inside a whole configuration. */
struct KeyHome {
    std::string key;
    std::string before;   // everything up to the key's line
    std::string after;    // everything after it

    std::string config(const std::string& value) const
    {
        return before + key + (value.empty() ? "" : " " + value) + "\n" +
               after;
    }
    std::size_t line() const
    {
        return static_cast<std::size_t>(
                   std::count(before.begin(), before.end(), '\n')) + 1;
    }
    /* "Error on line N: " */
    std::string at() const
    {
        return "Error on line " + std::to_string(line()) + ": ";
    }
};

KeyHome root_home(const std::string& key) { return {key, kOptimizer, ""}; }

/* A key of the STRATEGY block, after TYPE/LIB/ALGO (line 6). */
KeyHome strategy_home(const std::string& key, const std::string& type,
                      const std::string& lib, const std::string& algo,
                      const std::string& tail = "")
{
    return {key,
            "OPTIMIZER\nSTRATEGY\nTYPE " + type + "\nLIB " + lib +
                "\nALGO " + algo + "\n",
            tail + "ENDSTRATEGY\nENDOPTIMIZER\n"};
}

KeyHome restart_home(const std::string& key)
{
    return {key,
            "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO LN_SBPLX\n"
            "MAXEVAL 5\nRESTART\n",
            "ENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n"};
}

const std::string kStrategyBlock =
    "STRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\n"
    "ENDSTRATEGY\n";

KeyHome init_home(const std::string& key)
{
    return {key, "OPTIMIZER\nINIT\nMETHOD RANDOM 2\n",
            "ENDINIT\n" + kStrategyBlock + "ENDOPTIMIZER\n"};
}

KeyHome noise_home(const std::string& key)
{
    return {key, "OPTIMIZER\nNOISE\n",
            "ENDNOISE\n" + kStrategyBlock + "ENDOPTIMIZER\n"};
}

/* An optimiser attribute: NLopt honours every stop criterion. */
KeyHome attr_home(const std::string& key)
{
    return strategy_home(key, "PLAIN", "NLOPT", "LN_SBPLX");
}

/* `key` in every block that configures an optimiser: the short form
 * directly inside OPTIMIZER, the STRATEGY level, EXPLORE, REFINE and
 * INNER. */
std::vector<KeyHome> optimizer_block_homes(const std::string& key)
{
    return {
        {key, "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\n", "ENDOPTIMIZER\n"},
        attr_home(key),
        {key,
         "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB DLIB\nALGO BFGS\n"
         "EXPLORE\nLIB DLIB\nALGO GFS\n",
         "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n"},
        {key,
         "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\n"
         "EXPLORE\nLIB DLIB\nALGO GFS\nENDEXPLORE\n"
         "REFINE\nLIB DLIB\nALGO BFGS\n",
         "ENDREFINE\nENDSTRATEGY\nENDOPTIMIZER\n"},
        {key,
         "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO G_MLSL_LDS\n"
         "INNER\nLIB NLOPT\nALGO LN_SBPLX\n",
         "ENDINNER\nENDSTRATEGY\nENDOPTIMIZER\n"}};
}

/* The leftover-token error of the optimiser attributes (line-framed).
 * "Error on line N: " names the line once. */
std::string attr_extra(const KeyHome& h, const std::string& tok)
{
    return h.at() + h.key + " takes no further value here (unexpected '" +
           tok + "')";
}

std::string single_extra(const KeyHome& h, const std::string& tok)
{
    return h.at() + h.key + " takes a single value (unexpected '" + tok + "')";
}

/* Tokens no integer key accepts. */
const std::vector<std::string> kNotUnsigned = {
    "abc", "5x", "2.5", "-1", "+5", "1e3", "0x10", "99999999999999999999999"};

/* Tokens no real key accepts. A hexadecimal float is not in the number
 * grammar (std::stod read `0x1p4` as 16). */
const std::vector<std::string> kNotFinite = {
    "abc", "5x", "1.5.2", "inf", "-inf", "nan", "1e400", "--5", "0x1p4",
    "-0x10"};

/* An integer key: `bound` is how the messages state its range ("an integer
 * >= 1", "an integer from 1 to 15"). */
void check_uint_key(const KeyHome& h, const std::string& bound,
                    const std::vector<std::string>& accepted,
                    const std::vector<std::string>& out_of_range,
                    bool line_framed_attr = false)
{
    INFO("key " << h.key << " on line " << h.line());
    for (const auto& v : accepted) {
        INFO("accepted value '" << v << "'");
        CHECK(parse_error(h.config(v)) == "");
    }
    std::vector<std::string> rejected = kNotUnsigned;
    rejected.insert(rejected.end(), out_of_range.begin(), out_of_range.end());
    for (const auto& tok : rejected) {
        INFO("rejected value '" << tok << "'");
        CHECK_THAT(parse_error(h.config(tok)),
                   Catch::StartsWith(h.at() + h.key + " must be " + bound +
                                     ", got '" + tok + "'"));
    }
    INFO("missing value");
    CHECK_THAT(parse_error(h.config("")),
               Catch::StartsWith(h.at() + h.key + " requires a value"));
    INFO("extra token");
    CHECK_THAT(parse_error(h.config(accepted.front() + " 7")),
               Catch::StartsWith(line_framed_attr ? attr_extra(h, "7")
                                                  : single_extra(h, "7")));
}

/* A real key: `kind` is how the messages state what it holds ("a finite
 * number", "a positive finite number"), and `out_of_range` lists numbers
 * in the grammar the key refuses with that message. */
void check_finite_key(const KeyHome& h,
                      const std::vector<std::string>& accepted,
                      bool line_framed_attr = false,
                      const std::string& kind = "a finite number",
                      const std::vector<std::string>& out_of_range = {})
{
    INFO("key " << h.key << " on line " << h.line());
    for (const auto& v : accepted) {
        INFO("accepted value '" << v << "'");
        CHECK(parse_error(h.config(v)) == "");
    }
    std::vector<std::string> rejected = kNotFinite;
    rejected.insert(rejected.end(), out_of_range.begin(), out_of_range.end());
    for (const auto& tok : rejected) {
        INFO("rejected value '" << tok << "'");
        CHECK_THAT(parse_error(h.config(tok)),
                   Catch::StartsWith(h.at() + h.key + " must be " + kind +
                                     ", got '" + tok + "'"));
    }
    INFO("missing value");
    CHECK_THAT(parse_error(h.config("")),
               Catch::StartsWith(h.at() + h.key + " requires a value"));
    INFO("extra token");
    CHECK_THAT(parse_error(h.config(accepted.front() + " 7")),
               Catch::StartsWith(line_framed_attr ? attr_extra(h, "7")
                                                  : single_extra(h, "7")));
}

tadah::hpo::HPOTargets parse_config(const std::string& body)
{
    TempFile cfg(body);
    tadah::core::Context ctx;
    return tadah::hpo::HPOTargets(ctx, cfg.path.string());
}

}  // namespace

TEST_CASE("Strict numbers: FAILSCORE is a positive finite number",
          "[HPO][hpo_targets][strict_numbers][failscore]")
{
    /* A FAILSCORE of 0 or below has no failure band: every finite loss,
     * FAILSCORE itself included, counted as usable. A run whose every
     * evaluation failed ended COMPLETED with "[ok] Every evaluation
     * produced a usable loss" and a best_pot.tadah that held a failed
     * evaluation, and without a skip cap the ceiling clipped every
     * positive loss to FAILSCORE. Such a value is an error at its line. */
    const auto h = root_home("FAILSCORE");
    check_finite_key(h, {"1e16", "0.5", "1e-300"}, false,
                     "a positive finite number",
                     {"0", "-0", "+0", "0.0", "-5", "-1e-300", "-1e16"});
    CHECK(parse_config(h.config("1e16")).FAIL_SCORE == 1e16);
    CHECK(parse_config(h.config("0.5")).FAIL_SCORE == 0.5);
    CHECK_THAT(parse_error(h.config("0")),
               Catch::StartsWith(h.at() + "FAILSCORE must be a positive "
                                 "finite number, got '0'"));

    /* The default is still the largest double. */
    CHECK(parse_config(kOptimizer).FAIL_SCORE ==
          std::numeric_limits<double>::max());
}

TEST_CASE("Strict numbers: QFLUSH, DIGITS, OUTPUT and BOUTPUT are integers "
          "in range", "[HPO][hpo_targets][strict_numbers]")
{
    check_uint_key(root_home("QFLUSH"), "an integer >= 0", {"25", "0"}, {});
    CHECK(parse_config(root_home("QFLUSH").config("25")).QFLUSH_THRESHOLD == 25u);
    CHECK(parse_config(root_home("QFLUSH").config("0")).QFLUSH_THRESHOLD == 0u);

    check_uint_key(root_home("DIGITS"), "an integer from 1 to 15",
                   {"7", "1", "15"}, {"0", "16"});
    CHECK(parse_config(root_home("DIGITS").config("15")).outputDigits == 15);

    check_uint_key(root_home("OUTPUT"), "an integer >= 1", {"7", "1"}, {"0"});
    check_uint_key(root_home("BOUTPUT"), "an integer >= 1", {"3", "1"}, {"0"});
    CHECK(parse_config(root_home("OUTPUT").config("7")).outputEvery == 7u);
}

/* Pins also: parsing creates no directory. The archive directory is the
 * run's output: the run creates it once every startup check passed, before
 * its first evaluation (hpo_run(); test_hpo_host_e2e's DUMP and BDUMP cases
 * create it as the run does, and write into one only a target file names),
 * and `tadah hpo --check` (test_hpo_check.cpp) leaves the working
 * directory as it is.
 * Before the change: the parse created the directory. */
TEST_CASE("Strict numbers: DUMP and BDUMP take a cadence and a directory",
          "[HPO][hpo_targets][strict_numbers]")
{
    const fs::path dir = unique_temp_path("_dump");
    struct RemoveDir {
        fs::path p;
        ~RemoveDir() { std::error_code ec; fs::remove_all(p, ec); }
    } cleanup{dir};
    for (const std::string key : {"DUMP", "BDUMP"}) {
        INFO(key);
        const auto h = root_home(key);
        for (const std::string tok : {"5x", "0", "-1", "2.5", "+5", "1e3", "abc"}) {
            INFO("cadence '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok + " " + dir.string())),
                       Catch::StartsWith(h.at() + key + " cadence must be an "
                                         "integer >= 1, got '" + tok + "'"));
        }
        CHECK_THAT(parse_error(h.config("")),
                   Catch::StartsWith(h.at() + key +
                                     " requires a cadence and a directory"));
        CHECK_THAT(parse_error(h.config("5")),
                   Catch::StartsWith(h.at() + key +
                                     " requires a cadence and a directory"));
        CHECK_THAT(parse_error(h.config("5 " + dir.string() + " extra")),
                   Catch::StartsWith(h.at() + key + " takes a cadence and a "
                                     "directory (unexpected 'extra')"));
        CHECK_FALSE(fs::exists(dir));
        const auto trg = parse_config(h.config("5 " + dir.string()));
        CHECK((key == "DUMP" ? trg.dumpPotEvery : trg.dumpBestPotEvery) == 5u);
        CHECK((key == "DUMP" ? trg.dumpPotDir : trg.dumpBestPotDir) ==
              dir.string());
        CHECK_FALSE(fs::exists(dir));
        fs::remove_all(dir);
    }
}

/* Pins: the directory of a DUMP or BDUMP line is checked once the target
 * file is read (for `tadah hpo --check` and the run alike), the error at
 * its line, and the check creates nothing: a directory in a directory that
 * does not exist ("DUMP directory '<root>/missing/pots' cannot be created:
 * the directory '<root>/missing' does not exist (tadah hpo creates only the
 * last directory of the path)"), a path that is a file ("... '<root>/a_file'
 * exists and is not a directory"), a directory in a file ("... cannot be
 * created: '<root>/a_file' is not a directory") and a symbolic link to a
 * path that does not exist ("... '<root>/dangling' is a symbolic link to
 * '<root>/nowhere', which does not exist") are refused, at the line; an
 * existing directory, a new one in an existing directory, the same written
 * with a trailing '/', and a relative one in the working directory are
 * accepted, and none is created. A trailing '/' or '/.' is taken as mkdir
 * takes it: "<dangling>/" is refused as the link, "<root>/pots/." with pots
 * missing is refused ("... cannot be resolved: it walks '.' in
 * '<root>/pots', which does not exist"), "<root>/fresh/" is accepted and
 * created by the run.
 * Before the trailing-separator rule: "<dangling>/" and "<root>/pots/."
 * were accepted (the run then failed creating them).
 * Before the check: the parse accepted the first three (the run stopped
 * when it created the directory, after it had loaded the data, or wrote no
 * archive into the file): the refusal checks fail.
 * Before the symbolic-link check: the parse accepted the dangling link,
 * and the run stopped when it created the directory ("could not be
 * created: File exists"): its refusal check fails. */
TEST_CASE("DUMP and BDUMP refuse a directory the run cannot create, at "
          "their line, and create nothing",
          "[HPO][hpo_targets][archive_directories]")
{
    const fs::path root = physical_temp_dir("_archive");
    struct RemoveDir {
        fs::path p;
        ~RemoveDir() { std::error_code ec; fs::remove_all(p, ec); }
    } cleanup{root};
    const fs::path file = root / "a_file";
    std::ofstream(file) << "not a directory\n";
    for (const std::string key : {"DUMP", "BDUMP"}) {
        INFO(key);
        const auto h = root_home(key);
        const std::string what = key + " directory '";
        const fs::path missing = root / "missing" / "pots";
        CHECK_THAT(parse_error(h.config("5 " + missing.string())),
                   Catch::StartsWith(
                       h.at() + what + missing.string() +
                       "' cannot be created: the directory '" +
                       (root / "missing").string() +
                       "' does not exist (tadah hpo creates only the last "
                       "directory of the path)"));
        CHECK_THAT(parse_error(h.config("5 " + file.string())),
                   Catch::StartsWith(h.at() + what + file.string() +
                                     "' exists and is not a directory"));
        const fs::path in_file = file / "pots";
        CHECK_THAT(parse_error(h.config("5 " + in_file.string())),
                   Catch::StartsWith(h.at() + what + in_file.string() +
                                     "' cannot be created: '" +
                                     file.string() +
                                     "' is not a directory"));
        CHECK_FALSE(fs::exists(root / "missing"));
        const fs::path dangling = root / "dangling";
        if (!fs::is_symlink(dangling))
            fs::create_symlink(root / "nowhere", dangling);
        CHECK_THAT(parse_error(h.config("5 " + dangling.string())),
                   Catch::StartsWith(h.at() + what + dangling.string() +
                                     "' is a symbolic link to '" +
                                     (root / "nowhere").string() +
                                     "', which does not exist"));
        CHECK_FALSE(fs::exists(root / "nowhere"));
        /* A trailing '/' or '/.' as mkdir takes it: "<dangling>/" is the
         * link itself (refused as above), "<root>/pots/." walks '.' in a
         * directory that does not exist (mkdir fails), "<root>/fresh/" is
         * the directory fresh (accepted, created by the run). */
        CHECK_THAT(parse_error(h.config("5 " + dangling.string() + "/")),
                   Catch::StartsWith(h.at() + what + dangling.string() +
                                     "/' is a symbolic link to '" +
                                     (root / "nowhere").string() +
                                     "', which does not exist"));
        const std::string dot = (root / "pots").string() + "/.";
        CHECK_THAT(parse_error(h.config("5 " + dot)),
                   Catch::StartsWith(h.at() + what + dot +
                                     "' cannot be resolved: it walks '.' in '" +
                                     (root / "pots").string() +
                                     "', which does not exist"));
        {
            const std::string fresh = (root / "fresh").string() + "/";
            const auto trg = parse_config(h.config("5 " + fresh));
            CHECK_FALSE(fs::exists(root / "fresh"));
            trg.create_archive_directories();
            CHECK(fs::is_directory(root / "fresh"));
            fs::remove_all(root / "fresh");
        }
        CHECK_FALSE(fs::exists(root / "pots"));

        const auto existing = parse_config(h.config("5 " + root.string()));
        CHECK((key == "DUMP" ? existing.dumpPotDir
                             : existing.dumpBestPotDir) == root.string());
        for (const std::string dir :
             {(root / "new").string(), (root / "new").string() + "/",
              std::string("tadah_no_such_archive")}) {
            INFO("directory " << dir);
            const auto trg = parse_config(h.config("5 " + dir));
            CHECK((key == "DUMP" ? trg.dumpPotDir : trg.dumpBestPotDir) ==
                  dir);
            CHECK_FALSE(fs::exists(dir));
        }
    }
}

/* Pins: the DUMP and BDUMP directories are checked together: one inside
 * the other's new directory is accepted, whatever the order of the lines,
 * and HPOTargets::create_archive_directories() creates the outer one
 * first:
 *   - "DUMP 5 <root>/archive" then "BDUMP 5 <root>/archive/best";
 *   - "BDUMP 5 <root>/archive/best" then "DUMP 5 <root>/archive";
 *   - "BDUMP 5 <root>/archive" then "DUMP 5 <root>/archive/all";
 * each parses, creates nothing, and create_archive_directories() then
 * creates both. A directory two levels inside the other's ("BDUMP 5
 * <root>/archive/x/best") is refused at its line ("... cannot be created:
 * the directory '<root>/archive/x' does not exist ..."), and so is an outer
 * directory that cannot be created (DUMP in <root>/missing): the error is
 * at the line of that directory's key, the inner one being accepted; and
 * so is an inner directory whose path walks through a missing directory
 * with '..' ("BDUMP 5 <root>/archive/x/../best"), which mkdir cannot
 * create although the path names <root>/archive/best once normalised
 * (accepted before the '..' rule).
 * Before the change: each line was checked alone, so the inner directory
 * was refused when the outer one did not exist yet ("the directory
 * '<root>/archive' does not exist"); 1.5.0 created the DUMP directory at
 * its line and accepted the first layout only. */
TEST_CASE("DUMP and BDUMP: a directory inside the other's is accepted in "
          "either order and created after it",
          "[HPO][hpo_targets][archive_directories]")
{
    const fs::path root = physical_temp_dir("_archive_nested");
    struct RemoveDir {
        fs::path p;
        ~RemoveDir() { std::error_code ec; fs::remove_all(p, ec); }
    } cleanup{root};
    const fs::path archive = root / "archive";
    struct Layout {
        const char* name;
        std::string first, second;   // the two lines, in file order
        fs::path dump, bdump;        // the directories they name
    };
    const std::vector<Layout> layouts = {
        {"DUMP outer, first", "DUMP 5 " + archive.string(),
         "BDUMP 5 " + (archive / "best").string(), archive, archive / "best"},
        {"DUMP outer, second", "BDUMP 5 " + (archive / "best").string(),
         "DUMP 5 " + archive.string(), archive, archive / "best"},
        {"BDUMP outer, first", "BDUMP 5 " + archive.string(),
         "DUMP 5 " + (archive / "all").string(), archive / "all", archive},
    };
    for (const Layout& l : layouts) {
        INFO(l.name);
        const auto trg =
            parse_config(kOptimizer + l.first + "\n" + l.second + "\n");
        CHECK(trg.dumpPotDir == l.dump.string());
        CHECK(trg.dumpBestPotDir == l.bdump.string());
        CHECK_FALSE(fs::exists(archive));
        trg.create_archive_directories();
        CHECK(fs::is_directory(l.dump));
        CHECK(fs::is_directory(l.bdump));
        fs::remove_all(archive);
    }

    const std::size_t line = static_cast<std::size_t>(
        std::count(kOptimizer.begin(), kOptimizer.end(), '\n')) + 2;
    const std::string at = "Error on line " + std::to_string(line) + ": ";
    CHECK_THAT(parse_error(kOptimizer + "DUMP 5 " + archive.string() +
                           "\nBDUMP 5 " + (archive / "x" / "best").string() +
                           "\n"),
               Catch::StartsWith(at + "BDUMP directory '" +
                                 (archive / "x" / "best").string() +
                                 "' cannot be created: the directory '" +
                                 (archive / "x").string() +
                                 "' does not exist"));
    const fs::path missing = root / "missing" / "archive";
    CHECK_THAT(parse_error(kOptimizer + "BDUMP 5 " +
                           (missing / "best").string() + "\nDUMP 5 " +
                           missing.string() + "\n"),
               Catch::StartsWith(at + "DUMP directory '" + missing.string() +
                                 "' cannot be created: the directory '" +
                                 (root / "missing").string() +
                                 "' does not exist"));
    CHECK_FALSE(fs::exists(archive));
    CHECK_FALSE(fs::exists(root / "missing"));

    /* '..' walks the directories the operating system finds: "BDUMP 5
     * <root>/archive/x/../best" with "DUMP 5 <root>/archive" is refused at
     * the BDUMP line, as mkdir cannot walk '..' out of '<root>/archive/x',
     * which does not exist ("... cannot be resolved: it walks '..' out of
     * '<root>/archive/x', which does not exist"). */
    const std::string through = (archive / "x" / ".." / "best").string();
    CHECK_THAT(parse_error(kOptimizer + "DUMP 5 " + archive.string() +
                           "\nBDUMP 5 " + through + "\n"),
               Catch::StartsWith(at + "BDUMP directory '" + through +
                                 "' cannot be resolved: it walks '..' out of '" +
                                 (archive / "x").string() +
                                 "', which does not exist"));
    CHECK_FALSE(fs::exists(archive));

    /* A link followed by '..' leads where the link's target is: with lnk a
     * link to <root>/elsewhere/deep, "DUMP 5 <root>/lnk/../archive" is
     * <root>/elsewhere/archive, so "BDUMP 5 <root>/archive/best" is refused
     * at its line (the directory '<root>/archive' does not exist); and a
     * DUMP directory reached through a link into a directory that does not
     * exist (lnk2 -> <root>/missing2) is refused naming that directory. */
    fs::create_directories(root / "elsewhere" / "deep");
    fs::create_symlink(root / "elsewhere" / "deep", root / "lnk");
    CHECK_THAT(parse_error(kOptimizer + "DUMP 5 " +
                           (root / "lnk" / ".." / "archive").string() +
                           "\nBDUMP 5 " + (archive / "best").string() + "\n"),
               Catch::StartsWith(at + "BDUMP directory '" +
                                 (archive / "best").string() +
                                 "' cannot be created: the directory '" +
                                 archive.string() + "' does not exist"));
    fs::create_symlink(root / "missing2", root / "lnk2");
    const std::size_t line9 = line - 1;
    CHECK_THAT(parse_error(kOptimizer + "DUMP 5 " +
                           (root / "lnk2" / "pots").string() + "\n"),
               Catch::StartsWith("Error on line " + std::to_string(line9) +
                                 ": DUMP directory '" +
                                 (root / "lnk2" / "pots").string() +
                                 "' cannot be created: the directory '" +
                                 (root / "missing2").string() +
                                 "' does not exist"));
    CHECK_FALSE(fs::exists(root / "elsewhere" / "archive"));
    CHECK_FALSE(fs::exists(root / "missing2"));

    /* The creation order follows the resolved paths: lnk3 a link to
     * <root>/archive, which does not exist yet, "BDUMP 5 <root>/archive"
     * with "DUMP 5 <root>/lnk3/all", in either order: accepted, nothing
     * created, and create_archive_directories() creates <root>/archive
     * first, then <root>/archive/all through the link (in the order of the
     * lines as written, DUMP first, mkdir of lnk3/all fails). */
    fs::create_symlink(archive, root / "lnk3");
    for (const bool bdump_first : {true, false}) {
        INFO("BDUMP line first: " << bdump_first);
        const std::string bdump = "BDUMP 5 " + archive.string() + "\n";
        const std::string dump =
            "DUMP 5 " + (root / "lnk3" / "all").string() + "\n";
        const auto trg = parse_config(
            kOptimizer + (bdump_first ? bdump + dump : dump + bdump));
        CHECK_FALSE(fs::exists(archive));
        trg.create_archive_directories();
        CHECK(fs::is_directory(archive / "all"));
        fs::remove_all(archive);
    }

    /* '.' in the other line's directory, which the run creates first, is
     * taken in that directory: "DUMP 5 <root>/archive" with "BDUMP 5
     * <root>/archive/." (the same directory) or "BDUMP 5
     * <root>/archive/./best", in either order: accepted, nothing created,
     * and create_archive_directories() creates both (archive, then
     * archive/best). Before: refused at the BDUMP line ("it walks '.' in
     * '<root>/archive', which does not exist"). */
    for (const std::string inner : {archive.string() + "/.",
                                    archive.string() + "/./best"}) {
        for (const bool dump_first : {true, false}) {
            INFO("BDUMP 5 " << inner << ", DUMP line first: " << dump_first);
            const std::string dump = "DUMP 5 " + archive.string() + "\n";
            const std::string bdump = "BDUMP 5 " + inner + "\n";
            const auto trg = parse_config(
                kOptimizer + (dump_first ? dump + bdump : bdump + dump));
            CHECK_FALSE(fs::exists(archive));
            trg.create_archive_directories();
            CHECK(fs::is_directory(archive));
            if (inner.size() > archive.string().size() + 2)
                CHECK(fs::is_directory(archive / "best"));
            fs::remove_all(archive);
        }
    }

    /* "BDUMP 5 <root>/best/." with "DUMP 5 <root>/best/pots": the BDUMP
     * line walks '.' in <root>/best, which does not exist (mkdir fails),
     * and is refused at its line. */
    {
        const std::string best_dot = (root / "best").string() + "/.";
        CHECK_THAT(parse_error(kOptimizer + "BDUMP 5 " + best_dot +
                               "\nDUMP 5 " + (root / "best" / "pots").string() +
                               "\n"),
                   Catch::StartsWith("Error on line " +
                                     std::to_string(line - 1) +
                                     ": BDUMP directory '" + best_dot +
                                     "' cannot be resolved: it walks '.' in '" +
                                     (root / "best").string() +
                                     "', which does not exist"));
        CHECK_FALSE(fs::exists(root / "best"));
    }

    /* "DUMP 5 ../archive" with "BDUMP 5 ../archive/best", relative to a
     * working directory <root>/work, in either order: accepted, and both
     * created, the outer first. */
    fs::create_directories(root / "work");
    struct Cwd {
        fs::path old = fs::current_path();
        ~Cwd() { std::error_code ec; fs::current_path(old, ec); }
    } restore;
    fs::current_path(root / "work");
    for (const bool dump_first : {true, false}) {
        INFO("DUMP line first: " << dump_first);
        const std::string dump = "DUMP 5 ../archive\n";
        const std::string bdump = "BDUMP 5 ../archive/best\n";
        const auto trg = parse_config(kOptimizer + (dump_first ? dump + bdump
                                                               : bdump + dump));
        CHECK_FALSE(fs::exists(archive));
        trg.create_archive_directories();
        CHECK(fs::is_directory(archive / "best"));
        fs::remove_all(archive);
    }
}

/* Pins: the directory of a DUMP or BDUMP line must be one the run can
 * write into: an existing directory without write permission ("DUMP
 * directory '<root>/ro' is not writable") and a new one in such a
 * directory ("... '<root>/ro/pots' cannot be created: the directory
 * '<root>/ro' is not writable") are refused at the line.
 * A directory under one without any permission (mode 000) cannot be
 * resolved: "... cannot be resolved: '<root>/locked/pots' cannot be read:
 * Permission denied" (it was reported as a file type).
 * Before the check: both were accepted, and the run wrote no archive (each
 * one a WARNING in the journal) or stopped when it created the directory.
 * Skipped as root, whose writes no permission stops. */
TEST_CASE("DUMP and BDUMP refuse a directory the run cannot write into",
          "[HPO][hpo_targets][archive_directories]")
{
    if (::geteuid() == 0)
        TADAH_SKIP("running as root: file permissions are not enforced");
    const fs::path root = physical_temp_dir("_archive_ro");
    const fs::path ro = root / "ro";
    fs::create_directories(ro);
    struct RemoveDir {
        fs::path p, ro;
        ~RemoveDir()
        {
            std::error_code ec;
            fs::permissions(ro, fs::perms::owner_all, ec);
            fs::remove_all(p, ec);
        }
    } cleanup{root, ro};
    fs::permissions(ro, fs::perms::owner_read | fs::perms::owner_exec);
    const fs::path locked = root / "locked";
    fs::create_directories(locked);
    struct Unlock {
        fs::path p;
        ~Unlock() { std::error_code ec; fs::permissions(p, fs::perms::owner_all, ec); }
    } unlock{locked};
    fs::permissions(locked, fs::perms::none);
    for (const std::string key : {"DUMP", "BDUMP"}) {
        INFO(key);
        const auto h = root_home(key);
        CHECK_THAT(parse_error(h.config("5 " + (locked / "pots").string())),
                   Catch::StartsWith(h.at() + key + " directory '" +
                                     (locked / "pots").string() +
                                     "' cannot be resolved: '" +
                                     (locked / "pots").string() +
                                     "' cannot be read: Permission denied"));
        CHECK_THAT(parse_error(h.config("5 " + ro.string())),
                   Catch::StartsWith(h.at() + key + " directory '" +
                                     ro.string() + "' is not writable"));
        CHECK_THAT(parse_error(h.config("5 " + (ro / "pots").string())),
                   Catch::StartsWith(h.at() + key + " directory '" +
                                     (ro / "pots").string() +
                                     "' cannot be created: the directory '" +
                                     ro.string() + "' is not writable"));
    }
}

TEST_CASE("Strict numbers: STRATEGY N_STARTS, PARALLEL, BUDGET_PER_START and "
          "TOPK are integers >= 1", "[HPO][hpo_targets][strict_numbers]")
{
    for (const std::string key : {"N_STARTS", "PARALLEL"}) {
        const auto h = strategy_home(key, "MULTISTART", "DLIB", "BFGS");
        check_uint_key(h, "an integer >= 1", {"4", "1"}, {"0"});
    }
    /* BUDGET_PER_START becomes each start's MAXEVAL: the same int budget. */
    check_uint_key(strategy_home("BUDGET_PER_START", "MULTISTART", "DLIB",
                                 "BFGS"),
                   "an integer from 1 to 2147483647", {"4", "1", "2147483647"},
                   {"0", "2147483648", "4294967299"});
    const auto ms = parse_config(
        strategy_home("N_STARTS", "MULTISTART", "DLIB", "BFGS").config("4"));
    CHECK(ms.spec.multistart.n_starts == 4u);

    const auto topk = strategy_home("TOPK", "PRESTAGE", "DLIB", "BFGS",
                                    "EXPLORE\nLIB DLIB\nALGO GFS\nENDEXPLORE\n");
    /* TOPK is a count of starting points, at most 1000000
     * (init_strategy.h, kMaxStartingPoints); it was read up to 2^64 - 1. */
    check_uint_key(topk, "an integer from 1 to 1000000",
                   {"3", "1", "1000000"}, {"0", "1000001"});
    CHECK(parse_config(topk.config("3")).spec.stage_topk == 3u);
}

TEST_CASE("Strict numbers: RESTART AFTER_STAGNATION and MAX are integers >= 0",
          "[HPO][hpo_targets][strict_numbers]")
{
    check_uint_key(restart_home("AFTER_STAGNATION"), "an integer >= 0",
                   {"10", "0"}, {});
    check_uint_key(restart_home("MAX"), "an integer >= 0", {"2", "0"}, {});
    const auto trg = parse_config(restart_home("MAX").config("2"));
    CHECK(trg.spec.restart.max == 2u);

    /* The RESTART block's SEED seeds its METHOD chain (restart r draws
     * with SEED + r): an integer >= 0, like an INIT block's SEED. */
    check_uint_key(restart_home("SEED"), "an integer >= 0",
                   {"42", "0", "18446744073709551615"},
                   {"18446744073709551616"});
    CHECK(parse_config(restart_home("SEED").config("42"))
              .spec.restart.init_strategy.seed ==
          std::optional<std::uint64_t>(42));
}

TEST_CASE("Strict numbers: INIT K_TOTAL, SEED and the METHOD K",
          "[HPO][hpo_targets][strict_numbers]")
{
    /* K_TOTAL and K are counts of starting points, at most 1000000
     * (kMaxStartingPoints); they were read up to 2^64 - 1. */
    check_uint_key(init_home("K_TOTAL"), "an integer from 0 to 1000000",
                   {"4", "0", "1000000"}, {"1000001"});
    check_uint_key(init_home("SEED"), "an integer >= 0",
                   {"11", "0", "18446744073709551615"},
                   {"18446744073709551616"});
    CHECK(parse_config(init_home("SEED").config("18446744073709551615"))
              .spec.init_strategy_spec.seed ==
          std::optional<std::uint64_t>(18446744073709551615ull));

    /* METHOD RANDOM|LHS|SOBOL <K>, checked at its own line (line 3). LHS
     * and SOBOL draw at least one point: K 0 is out of their range, in the
     * same message (it was "INIT METHOD LHS K must be > 0"). */
    for (const std::string method : {"RANDOM", "LHS", "SOBOL"}) {
        const KeyHome h{"METHOD " + method, "OPTIMIZER\nINIT\n",
                        "ENDINIT\n" + kStrategyBlock + "ENDOPTIMIZER\n"};
        INFO(h.key);
        CHECK(parse_error(h.config("3")) == "");
        const bool random = method == "RANDOM";
        std::vector<std::string> rejected = kNotUnsigned;
        rejected.push_back("1000001");
        if (random)
            CHECK(parse_error(h.config("0")) == "");
        else
            rejected.push_back("0");
        for (const auto& tok : rejected) {
            INFO("K '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + "INIT METHOD " + method +
                                         " K must be an integer from " +
                                         (random ? "0" : "1") +
                                         " to 1000000, got '" + tok + "'"));
        }
        CHECK_THAT(parse_error(h.config("")),
                   Catch::StartsWith(h.at() + "INIT METHOD " + method +
                                     " expects exactly one arg: K"));
        CHECK_THAT(parse_error(h.config("3 4")),
                   Catch::StartsWith(h.at() + "INIT METHOD " + method +
                                     " expects exactly one arg: K"));
    }
}

TEST_CASE("Strict numbers: NOISE CALIBRATION and LOG_HP",
          "[HPO][hpo_targets][strict_numbers]")
{
    const auto cal = noise_home("CALIBRATION");
    /* N evaluations, at most the int budget MAXEVAL is; it was read up
     * to 2^64 - 1. */
    check_uint_key(cal, "an integer from 0 to 2147483647",
                   {"5", "0", "2", "2147483647"}, {"2147483648"});
    CHECK_THAT(parse_error(cal.config("1")),
               Catch::StartsWith(cal.at() + "NOISE.CALIBRATION must be 0 (off) "
                                 "or >= 2"));
    CHECK(parse_config(cal.config("5")).spec.noise.calibration_n == 5u);

    const auto log_hp = attr_home("LOG_HP");
    check_finite_key(log_hp, {"10", "1"});
    CHECK_THAT(parse_error(log_hp.config("0.5")),
               Catch::StartsWith(log_hp.at() + "LOG_HP must be a finite ratio "
                                 ">= 1"));
}

TEST_CASE("Strict numbers: the optimiser attributes",
          "[HPO][hpo_targets][strict_numbers][optimizer_block]")
{
    /* MAXEVAL 1e5 used to read 1 and then report the 'e5' as a stray
     * value; -1 wrapped to 2^64-1; abc stored 0. MAXEVAL 0 is no budget
     * (its own case below). */
    check_uint_key(attr_home("MAXEVAL"), "an integer from 1 to 2147483647",
                   {"30", "1", "2147483647"},
                   {"0", "2147483648", "4294967299"},
                   /*line_framed_attr=*/true);
    CHECK(parse_config(attr_home("MAXEVAL").config("30")).spec.stop.maxeval ==
          std::optional<std::uint64_t>(30));
    check_uint_key(attr_home("SEED"), "an integer >= 0",
                   {"1717", "0", "18446744073709551615"},
                   {"18446744073709551616"}, true);
    for (const std::string key : {"POPULATION", "VECTOR_ARR"})
        check_uint_key(attr_home(key), "an integer from 0 to 4294967295",
                       {"4", "0", "4294967295"}, {"4294967296"}, true);
    /* THREADS sizes the worker pool (an int): at least one worker. */
    check_uint_key(attr_home("THREADS"), "an integer from 1 to 2147483647",
                   {"4", "1", "2147483647"}, {"0", "2147483648", "4294967295"},
                   true);
    CHECK(parse_config(attr_home("THREADS").config("4")).spec.num_threads ==
          std::optional<unsigned>(4));

    /* MAXTIME is a positive number of seconds (its own case below). */
    for (const std::string key : {"STOPVAL", "FTOL_REL", "FTOL_ABS",
                                  "XTOL_REL", "XTOL_ABS"})
        check_finite_key(attr_home(key), {"1e-6", "0.5"}, true);
    CHECK(parse_config(attr_home("STOPVAL").config("-3.5")).spec.stop.stopval ==
          std::optional<double>(-3.5));

    /* LIB and ALGO need their value too. */
    for (const std::string key : {"LIB", "ALGO"}) {
        const KeyHome h{key, "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\n" +
                                 std::string(key == "LIB" ? "ALGO LN_SBPLX\n"
                                                          : "LIB NLOPT\n"),
                        "ENDSTRATEGY\nENDOPTIMIZER\n"};
        CHECK_THAT(parse_error(h.config("")),
                   Catch::StartsWith(h.at() + key + " requires a value"));
    }
}

TEST_CASE("Strict numbers: MAXTIME in an optimiser block is a positive "
          "number of seconds", "[HPO][hpo_targets][strict_numbers]"
          "[optimizer_block]")
{
    /* The optimiser's whole-run limit. 0 or a negative value meant no
     * limit to NLopt and TADAH and a startup error to ASA and the
     * annealers; the key is left out for no limit. (Root-level MAXTIME is
     * the alias of PC_SCRIPT_TIMEOUT, a LAMMPS time spec.) */
    for (const auto& h : optimizer_block_homes("MAXTIME")) {
        INFO(h.config("60"));
        for (const std::string v : {"60", "0.5", "1e-6", "+2", "86400"}) {
            INFO("accepted '" << v << "'");
            CHECK(parse_error(h.config(v)) == "");
        }
        for (const std::string tok :
             {"-5", "0", "-0", "0.0", "-1e-9", "abc", "5x", "inf", "nan",
              "1e400", "0x1p4", "1:00"}) {
            INFO("rejected '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + "MAXTIME must be a positive "
                                         "number of seconds, got '" + tok +
                                         "'") &&
                       Catch::EndsWith("Line: MAXTIME " + tok));
        }
        CHECK_THAT(parse_error(h.config("")),
                   Catch::StartsWith(h.at() + "MAXTIME requires a value"));
        CHECK_THAT(parse_error(h.config("60 7")),
                   Catch::StartsWith(attr_extra(h, "7")));
    }
    CHECK(parse_config(attr_home("MAXTIME").config("60")).spec.stop.maxtime ==
          std::optional<double>(60.0));
    CHECK_FALSE(parse_config(attr_home("MAXEVAL").config("5"))
                    .spec.stop.maxtime);
}

TEST_CASE("Strict numbers: MAXEVAL is an integer from 1 to 2147483647 in "
          "every optimiser block", "[HPO][hpo_targets][strict_numbers]"
          "[optimizer_block]")
{
    /* NLopt counts evaluations in an int (nlopt::opt::set_maxeval), and the
     * adapter passed MAXEVAL on with an implicit conversion: MAXEVAL
     * 4294967299 ran 3 evaluations (MAXEVAL_REACHED), and 2147483648 became
     * a negative budget, which NLopt reads as no limit. MAXEVAL 0 meant a
     * different thing to every backend: no limit to NLopt, no evaluation
     * to TADAH:RANDOM, the annealers and ASA, the first start point to
     * DLIB:GFS, an error when DLIB:BFGS and CERES started, and nothing to
     * TADAH:GRID. The key is left out for no limit. */
    const std::string bound = "an integer from 1 to 2147483647";
    for (const auto& h : optimizer_block_homes("MAXEVAL")) {
        INFO(h.config("30"));
        for (const std::string v : {"1", "30", "2147483647"}) {
            INFO("accepted '" << v << "'");
            CHECK(parse_error(h.config(v)) == "");
        }
        for (const std::string tok :
             {"0", "-0", "00", "2147483648", "4294967299", "-1"}) {
            INFO("rejected '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + "MAXEVAL must be " + bound +
                                         ", got '" + tok + "'") &&
                       Catch::EndsWith("Line: MAXEVAL " + tok));
        }
    }
    CHECK(parse_config(attr_home("MAXEVAL").config("2147483647"))
              .spec.stop.maxeval == std::optional<std::uint64_t>(2147483647));
    CHECK(parse_config(attr_home("MAXEVAL").config("1")).spec.stop.maxeval ==
          std::optional<std::uint64_t>(1));
    /* Without MAXEVAL the spec has no budget. */
    CHECK_FALSE(parse_config(attr_home("MAXTIME").config("60"))
                    .spec.stop.maxeval);
}

TEST_CASE("Strict numbers: single-value keys reject an extra token",
          "[HPO][hpo_targets][strict_numbers]")
{
    struct Case { KeyHome home; std::string value; };
    for (const auto& c : std::vector<Case>{
             {{"TYPE", "OPTIMIZER\nSTRATEGY\n",
               "LIB NLOPT\nALGO LN_SBPLX\nENDSTRATEGY\nENDOPTIMIZER\n"},
              "PLAIN"},
             {root_home("PC_ISOLATION"), "THREAD"},
             {root_home("FAILSCORE_GRADING"), "ON"},
             {root_home("BUMPGATE"), "STRICT"},
             {root_home("OPTIM_SCALE"), "AUTO"},
             {root_home("PC_WORKER_TIMEOUT"), "5"},
             {init_home("WARM_ON_MISMATCH"), "warn"},
             {noise_home("AUTO"), "true"}}) {
        INFO(c.home.key);
        CHECK(parse_error(c.home.config(c.value)) == "");
        CHECK_THAT(parse_error(c.home.config(c.value + " x")),
                   Catch::StartsWith(single_extra(c.home, "x")));
    }
}

TEST_CASE("Strict numbers: PC_SCRIPT_TIMEOUT and MAXTIME are LAMMPS time "
          "specs", "[HPO][hpo_targets][strict_numbers][timeout]")
{
    for (const std::string key : {"PC_SCRIPT_TIMEOUT", "MAXTIME"}) {
        const auto h = root_home(key);
        INFO(key);
        for (const std::string v : {"00:05:00", "2:00:00", "1:30", "90", "0.5",
                                    "off", "unlimited"}) {
            INFO("accepted '" << v << "'");
            CHECK(parse_config(h.config(v)).MAX_TIME == v);
        }
        for (const std::string tok : {"bad", "5x", "1:2:3:4", "1::2", ":5",
                                      "5:", "inf", "nan", "1:x:3", "OFF"}) {
            INFO("rejected '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + key + " must be 'off', "
                                         "'unlimited' or a time [[HH:]MM:]SS, "
                                         "got '" + tok + "'"));
        }
        CHECK_THAT(parse_error(h.config("")),
                   Catch::StartsWith(h.at() + key + " requires a value"));
        CHECK_THAT(parse_error(h.config("90 7")),
                   Catch::StartsWith(single_extra(h, "7")));
    }
}

TEST_CASE("Strict numbers: OPTIM explicit bounds are finite numbers",
          "[HPO][hpo_targets][strict_numbers][optim]")
{
    auto optim_error = [](const std::string& line) {
        TempFile cfg(kOptimizer + line + "\n");
        tadah::core::Context ctx;
        ctx.add("ATOM", "Ta");
        ctx.add("WATOM", 1.0);
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    CHECK(optim_error("OPTIM WATOM (1) 0.1 5.0") == "");
    /* "Error on line 9: " names the line; the message does not repeat it. */
    CHECK_THAT(optim_error("OPTIM WATOM (1) 0.1 6.0x"),
               Catch::StartsWith("Error on line 9: OPTIM WATOM: <high> must "
                                 "be a finite number (got '6.0x')"));
    CHECK_THAT(optim_error("OPTIM WATOM (x) 0.1 5.0"),
               Catch::StartsWith("Error on line 9: OPTIM WATOM: bad index "
                                 "list (x): ") &&
               !Catch::Contains("process_optim_config"));
    CHECK_THAT(optim_error("OPTIM WATOM (3) 0.1 5.0"),
               Catch::StartsWith("Error on line 9: OPTIM WATOM (3): ") &&
               !Catch::Contains("process_optim_config"));
    for (const std::string tok : {"0.1x", "abc", "inf", "nan", "1e400"}) {
        INFO("low '" << tok << "'");
        CHECK_THAT(optim_error("OPTIM WATOM (1) " + tok + " 5.0"),
                   Catch::StartsWith("Error on line 9: ") &&
                   Catch::Contains("<low> must be a finite number (got '" +
                                   tok + "')"));
        INFO("high '" << tok << "'");
        CHECK_THAT(optim_error("OPTIM WATOM (1) 0.1 " + tok),
                   Catch::StartsWith("Error on line 9: ") &&
                   Catch::Contains("<high> must be a finite number (got '" +
                                   tok + "')"));
    }
}

TEST_CASE("Strict numbers: PC_FAILPROBE --gate and --s0 are finite numbers",
          "[HPO][hpo_targets][strict_numbers][failprobe]")
{
    /* --gate is any finite number, --s0 a positive one (a scale). */
    for (const std::string flag : {"--gate", "--s0"}) {
        const std::string kind =
            flag == "--gate" ? "a finite number" : "a positive number";
        for (const std::string tok : {"5x", "abc", "inf", "nan"}) {
            INFO(flag << " '" << tok << "'");
            CHECK_THAT(parse_error(kOptimizer + "PC_FAILPROBE --script p.in "
                                   "--outvar S " + flag + " " + tok + "\n"),
                       Catch::StartsWith("Error on line 9: PC_FAILPROBE " +
                                         flag + " must be " + kind + ", "
                                         "got '" + tok + "'"));
        }
    }
}

namespace {

/* A LAMMPS script that defines the variables the PC_LAMMPS cases use. */
const std::string kSimScript =
    "clear\nrun 0\nvariable E equal 0.0\nvariable F equal 0.0\n"
    "variable a0 equal 3.3\n";

std::string lammps_sim_error(std::string cmd)
{
    try {
        tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, tadah::LossFunction{});
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("Strict numbers: PC_LAMMPS option values",
          "[HPO][lammps_sim][parse][strict_numbers]")
{
    TempFile script(kSimScript, ".in");
    const std::string head = "--script " + script.path.string() + " ";
    struct Case { std::string options, message; };
    const std::vector<Case> cases = {
        {"--varloss E 8.1x 1", "--varloss E <target> must be a finite number, got '8.1x'"},
        {"--varloss E abc 1", "--varloss E <target> must be a finite number, got 'abc'"},
        {"--varloss E inf 1", "--varloss E <target> must be a finite number, got 'inf'"},
        {"--varloss E 8.1 1x", "--varloss E <weight> must be a finite number, got '1x'"},
        {"--varloss E 8.1 nan", "--varloss E <weight> must be a finite number, got 'nan'"},
        {"--varloss E 8.1 1 HUBER 5x",
         "--varloss E HUBER parameter must be a finite number, got '5x'"},
        {"--varloss E 8.1 1 --failscore 1e6x",
         "--failscore must be a finite number, got '1e6x'"},
        {"--varloss E 8.1 1 --failscore abc",
         "--failscore must be a finite number, got 'abc'"},
        {"--varloss E 8.1 1 --failscore inf",
         "--failscore must be a finite number, got 'inf'"},
        {"--varloss E 8.1 1 --hotvar a0 2.5x 4.5",
         "--hotvar a0 <min> must be a finite number, got '2.5x'"},
        {"--varloss E 8.1 1 --hotvar a0 2.5 4.5x",
         "--hotvar a0 <max> must be a finite number, got '4.5x'"},
        /* An unparsable <initial> used to be passed on to LAMMPS as a
         * command-line argument. */
        {"--varloss E 8.1 1 --hotvar a0 2.5 4.5 3.3x",
         "--hotvar a0 <initial> must be a finite number, got '3.3x'"},
        {"--varloss E 8.1 1 --hotvar a0 2.5 4.5 -3x",
         "--hotvar a0 <initial> must be a finite number, got '-3x'"},
        {"--varloss E 8.1 1 --failvar F 0.5x",
         "--failvar F <scale> must be a positive number, got '0.5x'"},
        {"--varloss E 8.1 1 --maxtime bad",
         "--maxtime must be 'off', 'unlimited' or a time [[HH:]MM:]SS, got 'bad'"},
        {"--varloss E 8.1 1 --maxtime 5x",
         "--maxtime must be 'off', 'unlimited' or a time [[HH:]MM:]SS, got '5x'"},
    };
    for (const auto& c : cases) {
        INFO(c.options);
        /* LammpsSim on its own ... */
        CHECK_THAT(lammps_sim_error(head + c.options),
                   Catch::StartsWith(c.message));
        /* ... and through the parser, at the PC_LAMMPS line. */
        const std::string line = "PC_LAMMPS " + head + c.options;
        CHECK_THAT(parse_error(kOptimizer + line + "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message) &&
                   Catch::EndsWith("Line: " + line));
    }

    /* Accepted: negative targets and weights, time specs, a value after
     * an option is still the option's. */
    std::string ok = head + "--varloss E -8.1 -2 --hotvar a0 2.5 4.5 3.3 "
                     "--failvar F 0.5 --maxtime 00:05:00 --failscore -1e6";
    tadah::hpo::LammpsSim sim(ok, "off", 1.0e6, tadah::LossFunction{});
    CHECK(sim.t_varloss.at(0) == -8.1);
    CHECK(sim.w_varloss.at(0) == -2.0);
    CHECK(sim.maxtime == "00:05:00");
    CHECK(sim.failscore == -1e6);
}

TEST_CASE("Strict numbers: a script timeout must be positive, 'off' or "
          "'unlimited'", "[HPO][hpo_targets][lammps_sim][strict_numbers][timeout]")
{
    /* LAMMPS accepts 0 and a negative time spec, but $(timeremain) is then
     * 0, so the section guard stopped every multi-section script and each
     * evaluation scored FAILSCORE; a limit of 0 also stops a script at its
     * first timer check. PC_SCRIPT_TIMEOUT, its alias MAXTIME and
     * --maxtime reject such a limit at their line. */
    const auto positive = [](const std::string& what, const std::string& tok) {
        return what + " must be a positive time, 'off' or 'unlimited' (a "
               "limit <= 0 stops every script), got '" + tok + "'";
    };
    const std::vector<std::string> not_positive = {
        "0", "-0", "0.0", "-5", "0:00", "00:00:00", "-1:30", "0:-5"};

    for (const std::string key : {"PC_SCRIPT_TIMEOUT", "MAXTIME"}) {
        const auto h = root_home(key);
        INFO(key);
        for (const auto& tok : not_positive) {
            INFO("rejected '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + positive(key, tok)) &&
                       Catch::EndsWith("Line: " + key + " " + tok));
        }
        /* The smallest positive limits and no limit are still accepted. */
        for (const std::string v : {"0.001", "0:01", "off", "unlimited"}) {
            INFO("accepted '" << v << "'");
            CHECK(parse_error(h.config(v)) == "");
        }
    }

    TempFile script(kSimScript, ".in");
    const std::string head =
        "--script " + script.path.string() + " --varloss E 8.1 1 --maxtime ";
    for (const auto& tok : not_positive) {
        INFO("--maxtime '" << tok << "'");
        /* LammpsSim on its own ... */
        CHECK_THAT(lammps_sim_error(head + tok),
                   Catch::StartsWith(positive("--maxtime", tok)));
        /* ... and through the parser, at the PC_LAMMPS line. */
        const std::string line = "PC_LAMMPS " + head + tok;
        CHECK_THAT(parse_error(kOptimizer + line + "\n"),
                   Catch::StartsWith("Error on line 9: " +
                                     positive("--maxtime", tok)) &&
                   Catch::EndsWith("Line: " + line));
    }
    for (const std::string v : {"0.001", "off", "unlimited"}) {
        INFO("--maxtime '" << v << "' accepted");
        CHECK(lammps_sim_error(head + v) == "");
    }
}

TEST_CASE("Strict numbers: a timeout whose worker deadline does not fit in "
          "64-bit milliseconds is rejected at its line",
          "[HPO][hpo_targets][lammps_sim][strict_numbers][timeout]")
{
    /* The PROCESS worker deadline is kept in milliseconds (a long):
     * max(3t, t + 120 s) for a script timeout t, PC_WORKER_TIMEOUT as
     * given. A larger value overflowed the conversion (undefined
     * behaviour); it is far beyond any meaningful limit (about 95 million
     * years for t), so it is an error at its line. */
    for (const std::string key : {"PC_SCRIPT_TIMEOUT", "MAXTIME"}) {
        const auto h = root_home(key);
        INFO(key);
        CHECK(parse_error(h.config("3e15")) == "");
        for (const std::string tok : {"4e15", "1e300", "1e12:0:0"}) {
            INFO("rejected '" << tok << "'");
            CHECK_THAT(parse_error(h.config(tok)),
                       Catch::StartsWith(h.at() + key + " is too large for "
                                         "the worker deadline, got '" + tok +
                                         "'"));
        }
    }
    const auto w = root_home("PC_WORKER_TIMEOUT");
    CHECK(parse_error(w.config("9e15")) == "");
    CHECK_THAT(parse_error(w.config("1e16")),
               Catch::StartsWith(w.at() + "PC_WORKER_TIMEOUT is too large for "
                                 "the worker deadline, got '1e16'"));
    CHECK_THROWS_WITH(
        tadah::hpo::HPOTargets::parse_pc_worker_timeout(
            "1e300", "TADAH_PC_WORKER_TIMEOUT"),
        Catch::StartsWith("TADAH_PC_WORKER_TIMEOUT is too large for the "
                          "worker deadline, got '1e300'"));

    TempFile script(kSimScript, ".in");
    const std::string line = "PC_LAMMPS --script " + script.path.string() +
                             " --varloss E 8.1 1 --maxtime 4e15";
    CHECK_THAT(parse_error(kOptimizer + line + "\n"),
               Catch::StartsWith("Error on line 9: --maxtime is too large for "
                                 "the worker deadline, got '4e15'"));
}

TEST_CASE("Strict numbers: a number too small for a double says so",
          "[HPO][hpo_targets][strict_numbers]")
{
    /* 1e-320 is finite, but below the smallest normal double std::stod
     * reports it out of range; the message said it was not a finite
     * number (or not a time) without saying why. */
    const std::string note = "(too small: its magnitude is below the smallest "
                             "normal double, 2.2250738585072014e-308)";
    const auto fs = root_home("FAILSCORE");
    CHECK_THAT(parse_error(fs.config("1e-320")),
               Catch::StartsWith(fs.at() + "FAILSCORE must be a positive "
                                 "finite number, got '1e-320' " + note));
    const auto ts = root_home("PC_SCRIPT_TIMEOUT");
    CHECK_THAT(parse_error(ts.config("1e-320")),
               Catch::StartsWith(ts.at() + "PC_SCRIPT_TIMEOUT must be 'off', "
                                 "'unlimited' or a time [[HH:]MM:]SS, got "
                                 "'1e-320' " + note));
    /* An overflow is not "too small". */
    CHECK_THAT(parse_error(fs.config("1e400")),
               !Catch::Contains("too small"));
    /* PC_<METRIC> reads the same grammar: std::from_chars took 1e-320 as a
     * subnormal target that every other key rejected. */
    CHECK_THAT(parse_error(kOptimizer + "PC_ERMSE 1e-320 1\n"),
               Catch::StartsWith("Error on line 9: PC_ERMSE <target> must be "
                                 "a finite number, got '1e-320' " + note));
}

TEST_CASE("Strict numbers: a PC_<METRIC> number error names the key and the "
          "value", "[HPO][hpo_targets][strict_numbers]")
{
    struct Case { std::string line, message; };
    for (const auto& c : std::vector<Case>{
             {"PC_ERMSE abc 1", "PC_ERMSE <target> must be a finite number, "
                                "got 'abc'"},
             {"PC_ERMSE 0 1x", "PC_ERMSE <weight> must be a finite number, "
                               "got '1x'"},
             {"PC_EMAE 0 inf", "PC_EMAE <weight> must be a finite number, "
                               "got 'inf'"},
             {"PC_FRMSE 0 1 --skip-above 2x",
              "PC_FRMSE --skip-above must be a positive number, got '2x'"}}) {
        INFO(c.line);
        CHECK_THAT(parse_error(kOptimizer + c.line + "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message));
    }
}

TEST_CASE("Strict numbers: a PC_LAMMPS option takes only its own values",
          "[HPO][lammps_sim][parse][strict_numbers]")
{
    /* A bare token after an HPO option's values fell through to the LAMMPS
     * command line: `--maxtime 1:30 00` (a typo for 1:30:00) loaded, and
     * every LAMMPS run failed on 'Invalid command-line argument: 00'. A
     * bare token belongs to a LAMMPS option (-var x 1 2, -echo none) or is an
     * error naming the option it follows. */
    TempFile script(kSimScript, ".in");
    const std::string head = "--script " + script.path.string() + " ";
    const std::string vl = "--varloss E 8.1 1 ";
    struct Case { std::string options, message; };
    const std::vector<Case> cases = {
        {vl + "--maxtime 1:30 00",
         "--maxtime takes a single value (unexpected '00')"},
        {vl + "--failscore 100 5",
         "--failscore takes a single value (unexpected '5')"},
        {vl + "--outvar F extra",
         "--outvar takes a single value (unexpected 'extra')"},
        {vl + "--invar x 1 2",
         "--invar takes <name> <value> (unexpected '2')"},
        {vl + "--hotvar a0 2.5 4.5 3.3 9",
         "--hotvar takes <name> <min> <max> [<initial>] (unexpected '9')"},
        {vl + "--failvar F 0.5 7",
         "--failvar takes <name> [<scale>] (unexpected '7')"},
        {"extra --varloss E 8.1 1",
         "--script takes a single value (unexpected 'extra')"},
    };
    for (const auto& c : cases) {
        INFO(c.options);
        CHECK_THAT(lammps_sim_error(head + c.options),
                   Catch::StartsWith(c.message));
        const std::string line = "PC_LAMMPS " + head + c.options;
        CHECK_THAT(parse_error(kOptimizer + line + "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message) &&
                   Catch::EndsWith("Line: " + line));
    }
    /* A bare token before any option. */
    CHECK_THAT(lammps_sim_error("junk " + head + vl),
               Catch::StartsWith("unexpected 'junk': a PC_LAMMPS token is an "
                                 "option or a value of one"));

    /* LAMMPS options keep every value that follows them. (They were
     * "-var x 1 2 -sf omp -pk omp 4 neigh no -var y -5" until LAMMPS judged
     * them when the line is read: a LAMMPS built without the OPENMP
     * package, as this test's may be, cannot start with -sf omp or -pk omp,
     * so they are options every build starts with.) */
    std::string cmd = head + vl + "-var x 1 2 -echo none -nocite "
                      "-var y -5";
    tadah::hpo::LammpsSim sim(cmd, "off", 1.0e6, tadah::LossFunction{});
    const std::vector<std::string> expected = {
        "-var", "x", "1", "2", "-echo", "none", "-nocite",
        "-var", "y", "-5"};
    const auto it = std::search(sim.invar.begin(), sim.invar.end(),
                                expected.begin(), expected.end());
    CHECK(it != sim.invar.end());
}

TEST_CASE("PC_LAMMPS: the variable maxtime is reserved for the script "
          "timeout", "[HPO][lammps_sim][parse][timeout]")
{
    /* HPO passes -var maxtime "<spec> every 1" and injects `timer timeout
     * ${maxtime}`. A user's -var maxtime replaced it unchecked: with
     * PC_SCRIPT_TIMEOUT 60 and `-var maxtime 0` the journal reported 60 s
     * and a 180 s deadline while every script ran with timeout 0. */
    TempFile script(kSimScript, ".in");
    const std::string head =
        "--script " + script.path.string() + " --varloss E 8.1 1 ";
    for (const std::string opt : {"-var maxtime 0", "-v maxtime 30",
                                  "--invar maxtime 0"}) {
        INFO(opt);
        const std::string option = opt.substr(0, opt.find(' '));
        CHECK_THAT(lammps_sim_error(head + opt),
                   Catch::StartsWith(option + " maxtime: the variable maxtime "
                                     "is reserved for the script timeout") &&
                   Catch::Contains("--maxtime"));
        const std::string line = "PC_LAMMPS " + head + opt;
        CHECK_THAT(parse_error(kOptimizer + "PC_SCRIPT_TIMEOUT 60\n" + line +
                               "\n"),
                   Catch::StartsWith("Error on line 10: " + option +
                                     " maxtime: the variable maxtime is "
                                     "reserved"));
    }
    /* A variable whose value is the word maxtime is not the variable. */
    std::string cmd = head + "-var label maxtime";
    tadah::hpo::LammpsSim sim(cmd, "60", 1.0e6, tadah::LossFunction{});
    const std::vector<std::string> injected = {"-var", "maxtime", "60 every 1"};
    CHECK(std::search(sim.invar.begin(), sim.invar.end(), injected.begin(),
                      injected.end()) != sim.invar.end());
}

namespace {

/* What CORE's config parser deposits for a `tadah refit` stage block: its
 * whitespace-split tokens under the block's key. */
tadah::core::Context refit_context(const std::string& key,
                                   const std::string& body)
{
    tadah::core::Context c;
    std::istringstream iss(body);
    for (std::string t; iss >> t;) c.add(key, t);
    return c;
}

std::string refit_error(const std::string& key, const std::string& body)
{
    try {
        (void)tadah::hpo::refit::RefitParser::parse(refit_context(key, body));
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("tadah refit: SEED in an OPTIMIZER block is rejected with the "
          "REFIT_COMPACT_SEED hint", "[HPO][refit][strict_numbers]")
{
    /* The refit never seeded an optimiser from SEED: REFIT_COMPACT_SEED
     * seeds it. An accepted SEED did nothing. */
    /* The message names the block: an INNER block's is "EXPLORE INNER",
     * like every other error of that block. */
    struct Case { std::string key, body, block; };
    for (const auto& c : std::vector<Case>{
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 "
                         "SEED 7 ENDOPTIMIZER", "EXPLORE"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 "
                         "INNER LIB NLOPT ALGO LN_SBPLX SEED 7 ENDINNER "
                         "ENDOPTIMIZER", "EXPLORE INNER"},
             {"REFINE", "OPTIMIZER LIB CERES ALGO LM SEED 7 ENDOPTIMIZER",
              "REFINE"},
             {"FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG SEED 7 ENDOPTIMIZER",
              "FUSE"}}) {
        INFO(c.key << ": " << c.body);
        CHECK_THAT(refit_error(c.key, c.body),
                   Catch::StartsWith(c.block + ": SEED is not accepted in a "
                                     "tadah refit OPTIMIZER block") &&
                   Catch::Contains("REFIT_COMPACT_SEED"));
    }
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 ENDOPTIMIZER") == "");
}

/* A tadah refit stage sets each key once, as a target file does: a key
 * given twice in an OPTIMIZER block, in its INNER block, or as MAXEVAL and
 * its alias MAXITER, is refused, naming the block and the key as written
 * ("REFINE: MAXEVAL is set twice in the OPTIMIZER block (as MAXEVAL and as
 * MAXITER, its alias); each key is given once"; "REFINE: MAXITER is set
 * twice ..."), and so are a second OPTIMIZER block in a stage, a second
 * INNER block in an OPTIMIZER block and a stage key given twice (ENABLED,
 * MAX_ADD). A PARAM of another name is no repeat.
 * Before the fix: the later value replaced the first without a word (a
 * second OPTIMIZER block replaced the first's settings). */
TEST_CASE("tadah refit: a key set twice in a stage is refused",
          "[HPO][refit][strict_numbers]")
{
    const std::string once = "; each key is given once";
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 5 "
                                "MAXEVAL 9 ENDOPTIMIZER") ==
          "REFINE: MAXEVAL is set twice in the OPTIMIZER block" + once);
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 5 "
                                "MAXITER 9 ENDOPTIMIZER") ==
          "REFINE: MAXEVAL is set twice in the OPTIMIZER block (as MAXEVAL "
          "and as MAXITER, its alias)" + once);
    /* the key as written: MAXITER twice is no MAXEVAL (it said "MAXEVAL
     * is set twice"); mixed spellings in the order written */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXITER 5 "
                                "MAXITER 9 ENDOPTIMIZER") ==
          "REFINE: MAXITER is set twice in the OPTIMIZER block" + once);
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG MAXITER 5 "
                              "MAXITER 9 ENDOPTIMIZER") ==
          "FUSE: MAXITER is set twice in the OPTIMIZER block" + once);
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXITER 5 "
                                "MAXEVAL 9 ENDOPTIMIZER") ==
          "REFINE: MAXEVAL is set twice in the OPTIMIZER block (as MAXITER, "
          "its alias, and as MAXEVAL)" + once);
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG GTOL 1e-6 "
                              "GTOL 1e-8 ENDOPTIMIZER") ==
          "FUSE: GTOL is set twice in the OPTIMIZER block" + once);
    /* a second INNER block (it replaced the first) */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT ALGO LN_SBPLX "
                                 "MAXEVAL 5 ENDINNER INNER LIB NLOPT "
                                 "ALGO LN_BOBYQA MAXEVAL 7 ENDINNER "
                                 "ENDOPTIMIZER") ==
          "EXPLORE: the INNER block is given twice; an OPTIMIZER block takes "
          "one INNER block");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT ALGO LN_SBPLX "
                                 "MAXEVAL 5 MAXEVAL 6 ENDINNER "
                                 "ENDOPTIMIZER") ==
          "EXPLORE INNER: MAXEVAL is set twice in the INNER block" + once);
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM ENDOPTIMIZER "
                                "OPTIMIZER LIB CERES ALGO DOGLEG "
                                "ENDOPTIMIZER") ==
          "REFINE: the OPTIMIZER block is given twice; a stage takes one "
          "OPTIMIZER block");
    CHECK(refit_error("REFINE", "ENABLED true ENABLED false") ==
          "REFINE: ENABLED is set twice" + once);
    CHECK(refit_error("EXPLORE", "MAX_ADD 2 MAX_ADD 3 OPTIMIZER LIB NLOPT "
                                 "ALGO GN_MLSL_LDS MAXEVAL 100 "
                                 "ENDOPTIMIZER") ==
          "EXPLORE: MAX_ADD is set twice" + once);
    /* the same key in the OPTIMIZER and the INNER block: two optimisers */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT ALGO LN_SBPLX "
                                 "MAXEVAL 5 ENDINNER ENDOPTIMIZER") == "");
}

TEST_CASE("tadah refit: OPTIMIZER block numbers are strict",
          "[HPO][refit][strict_numbers]")
{
    /* MAXEVAL and its alias MAXITER are the int budgets the refit hands
     * NLopt and Ceres (static_cast<int>, which wrapped 4294967299 to 3),
     * and at least 1: MAXEVAL 0 ran the Ceres stages with no iteration and
     * NLopt with no limit. A malformed number names the stage block it is
     * in. */
    const std::string budget = "an integer from 1 to 2147483647";
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 1e5 "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: MAXEVAL must be " + budget +
                                 ", got '1e5'"));
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL -1 "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: MAXEVAL must be " + budget +
                                 ", got '-1'"));
    for (const std::string key : {"MAXEVAL", "MAXITER"}) {
        INFO(key << " 0");
        CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM " + key +
                                        " 0 ENDOPTIMIZER") ==
              "REFINE: " + key + " must be " + budget + ", got '0'");
    }
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG MAXITER 0 "
                              "ENDOPTIMIZER") ==
          "FUSE: MAXITER must be " + budget + ", got '0'");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 0 ENDOPTIMIZER") ==
          "EXPLORE: MAXEVAL must be " + budget + ", got '0'");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS INNER "
                                 "LIB NLOPT ALGO LN_SBPLX MAXEVAL 0 ENDINNER "
                                 "ENDOPTIMIZER") ==
          "EXPLORE INNER: MAXEVAL must be " + budget + ", got '0'");
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXITER 5x "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: MAXITER must be " + budget +
                                 ", got '5x'"));
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM GTOL 1e-14x "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: GTOL must be a finite number, got "
                                 "'1e-14x'"));
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM FTOL_REL nan "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: FTOL_REL must be a finite number, "
                                 "got 'nan'"));
    struct Case { std::string key, body, message; };
    for (const auto& c : std::vector<Case>{
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                         "MAXEVAL 4294967299 ENDOPTIMIZER",
              "EXPLORE: MAXEVAL must be " + budget + ", got '4294967299'"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS INNER "
                         "LIB NLOPT ALGO LN_SBPLX MAXEVAL 2147483648 "
                         "ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: MAXEVAL must be " + budget +
                  ", got '2147483648'"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS INNER "
                         "LIB NLOPT ALGO LN_SBPLX FTOL_REL 1e-7x ENDINNER "
                         "ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_REL must be a finite number, got '1e-7x'"},
             {"REFINE", "OPTIMIZER LIB CERES ALGO LM MAXITER 4294967299 "
                        "ENDOPTIMIZER",
              "REFINE: MAXITER must be " + budget + ", got '4294967299'"},
             {"FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 2147483648 "
                      "ENDOPTIMIZER",
              "FUSE: MAXEVAL must be " + budget + ", got '2147483648'"},
             {"FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG FTOL_REL inf "
                      "ENDOPTIMIZER",
              "FUSE: FTOL_REL must be a finite number, got 'inf'"},
             /* The block's other errors name it too. */
             {"REFINE", "OPTIMIZER LIB CERES ALGO LM",
              "REFINE: OPTIMIZER block not closed with ENDOPTIMIZER"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS INNER "
                         "LIB NLOPT ALGO LN_SBPLX",
              "EXPLORE INNER: INNER block not closed with ENDINNER"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS INNER "
                         "LIB NLOPT ALGO LN_SBPLX FTOLREL 1e-7 ENDINNER "
                         "ENDOPTIMIZER",
              "EXPLORE INNER: unknown key 'FTOLREL' in the INNER block"}}) {
        INFO(c.key << ": " << c.body);
        CHECK(refit_error(c.key, c.body) == c.message);
    }
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 2147483647 INNER LIB NLOPT ALGO "
                                 "LN_SBPLX MAXEVAL 2147483647 ENDINNER "
                                 "ENDOPTIMIZER") == "");
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 1 "
                                "ENDOPTIMIZER") == "");
    /* MAXEVAL and its alias MAXITER are one key, given once (a block with
     * both is refused: the case "a key set twice in a stage"). */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM "
                                "FTOL_REL 1e-14 GTOL 1e-14 MAXITER 50 "
                                "ENDOPTIMIZER") == "");
}

TEST_CASE("tadah refit: a negative tolerance is rejected in every stage block",
          "[HPO][refit][strict_numbers]")
{
    /* Pins: FTOL_REL, XTOL_REL and GTOL are finite numbers >= 0 in every
     * refit block that reads them (REFINE, FUSE, EXPLORE INNER), whatever
     * the ALGO: a negative one is an error at the key naming the block,
     * the key and the token as written ("REFINE: GTOL must be a number >=
     * 0, got '-1'"); 0 is accepted (it switches the test off) and so is a
     * positive value; a malformed token keeps its "must be a finite
     * number" message. Ceres refuses a negative tolerance in its option
     * check and returns FAILURE without iterating, so REFINE and FUSE did
     * nothing (refine=FAILURE(0it, Invalid configuration)); NLopt switches
     * one below 0 off, and dlib requires one >= 0.
     * Before the fix: every negative case parses ("" instead of the
     * message). */
    struct Case { std::string key, body, message; };
    for (const auto& c : std::vector<Case>{
             {"REFINE", "OPTIMIZER LIB CERES ALGO LM GTOL -1 ENDOPTIMIZER",
              "REFINE: GTOL must be a number >= 0, got '-1'"},
             {"REFINE", "OPTIMIZER LIB CERES ALGO DOGLEG FTOL_REL -1e-3 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_REL must be a number >= 0, got '-1e-3'"},
             {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS FTOL_REL -1 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_REL must be a number >= 0, got '-1'"},
             {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS FTOL_REL -.5 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_REL must be a number >= 0, got '-.5'"},
             {"FUSE", "OPTIMIZER LIB CERES ALGO LM FTOL_REL -1 ENDOPTIMIZER",
              "FUSE: FTOL_REL must be a number >= 0, got '-1'"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 "
                         "INNER LIB NLOPT ALGO LN_SBPLX XTOL_REL -1e-6 "
                         "ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: XTOL_REL must be a number >= 0, got '-1e-6'"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 "
                         "INNER LIB NLOPT ALGO LN_SBPLX FTOL_REL -2 "
                         "ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_REL must be a number >= 0, got '-2'"}}) {
        INFO(c.key << ": " << c.body);
        CHECK(refit_error(c.key, c.body) == c.message);
    }
    /* 0 and positive tolerances are accepted. */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM FTOL_REL 0 "
                                "GTOL 0 ENDOPTIMIZER") == "");
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG FTOL_REL 0 "
                              "ENDOPTIMIZER") == "");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT ALGO LN_SBPLX "
                                 "FTOL_REL 0 XTOL_REL 1e-6 ENDINNER "
                                 "ENDOPTIMIZER") == "");
    /* A malformed tolerance keeps the number-grammar message. */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM GTOL -1x "
                                "ENDOPTIMIZER") ==
          "REFINE: GTOL must be a finite number, got '-1x'");
}

TEST_CASE("tadah refit: a negative FTOL_ABS is rejected where dlib's LBFGS "
          "reads it", "[HPO][refit][strict_numbers]")
{
    /* Pins: FTOL_ABS, the absolute stop of dlib's LBFGS (EXPLORE INNER and
     * REFINE), is a finite number >= 0 like the other tolerances: a
     * negative one is an error at the key naming the block and the token as
     * written ("REFINE: FTOL_ABS must be a number >= 0, got '-1e-9'"); 0 and
     * a positive value are accepted. dlib requires a stop >= 0.
     * Before the fix: passes as well (a pin: RefitParser's tolerance reader
     * refused a negative FTOL_ABS with the other tolerances, but no test
     * covered it). */
    struct Case { std::string key, body, message; };
    for (const auto& c : std::vector<Case>{
             {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS FTOL_ABS -1e-9 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_ABS must be a number >= 0, got '-1e-9'"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_CRS2_LM MAXEVAL 100 "
                         "INNER LIB DLIB ALGO LBFGS FTOL_ABS -1 ENDINNER "
                         "ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS must be a number >= 0, got '-1'"}}) {
        INFO(c.key << ": " << c.body);
        CHECK(refit_error(c.key, c.body) == c.message);
    }
    CHECK(refit_error("REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS FTOL_ABS 0 "
                                "ENDOPTIMIZER") == "");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_CRS2_LM "
                                 "MAXEVAL 100 INNER LIB DLIB ALGO LBFGS "
                                 "FTOL_ABS 1e-12 ENDINNER ENDOPTIMIZER") == "");
}

TEST_CASE("tadah refit: a top-level OPTIMIZER block is refused with its "
          "migration text, which names the Refitting page",
          "[HPO][refit]")
{
    /* Pins: RefitParser::parse refuses a top-level OPTIMIZER block (the
     * grammar before the stage blocks) with the migration text, which ends
     * by naming the Stage blocks section of the Refitting page.
     * Before the fix: the text ended "See HPO_REFIT_2B_EAM/REDESIGN/configs
     * for examples.", a directory that is not shipped. */
    CHECK(refit_error("OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL") ==
          "tadah refit: a top-level OPTIMIZER block is no longer supported. "
          "The stage pipeline is now INIT -> EXPLORE -> REFINE -> FUSE, each "
          "its own block, with the OPTIMIZER block NESTED inside "
          "EXPLORE/REFINE/FUSE. Migration: GLOBAL folded into EXPLORE rung 0 "
          "(set its global ALGO in EXPLORE's OPTIMIZER, with an INNER local); "
          "POLISH -> REFINE; JOINT -> FUSE. The Refitting page of the "
          "documentation has examples (Stage blocks).");
}

TEST_CASE("tadah refit: EXPLORE MAX_ADD is an integer >= 0",
          "[HPO][refit][strict_numbers]")
{
    /* MAX_ADD was read with std::stoi: '+3' was 3 and a negative value
     * silently meant no cap. Without MAX_ADD the ladder is not capped. */
    const std::string opt =
        " OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 ENDOPTIMIZER";
    for (const std::string v : {"0", "45", "1 2 3"}) {
        INFO("MAX_ADD " << v);
        CHECK(refit_error("EXPLORE", "MAX_ADD " + v + opt) == "");
    }
    for (const std::string tok : {"-1", "+3", "2.5", "1e3", "2147483648"}) {
        INFO("MAX_ADD " << tok);
        CHECK_THAT(refit_error("EXPLORE", "MAX_ADD " + tok + opt),
                   Catch::StartsWith("EXPLORE MAX_ADD must be an integer from "
                                     "0 to 2147483647, got '" + tok + "'"));
    }
    CHECK_THAT(refit_error("EXPLORE", "MAX_ADD 1 -2 3" + opt),
               Catch::StartsWith("EXPLORE MAX_ADD must be an integer from 0 to "
                                 "2147483647, got '-2'"));
    const auto cfg = tadah::hpo::refit::RefitParser::parse(
        refit_context("EXPLORE", "MAX_ADD 1 2 3" + opt));
    CHECK(cfg.explore.max_add == std::array<int, 3>{1, 2, 3});
}

TEST_CASE("Strict numbers: THREADS 0 is rejected in every optimiser block",
          "[HPO][hpo_targets][strict_numbers][optimizer_block]")
{
    /* THREADS 0 used to be accepted: the pool width silently became one
     * worker at the OPTIMIZER / STRATEGY level (max(1, THREADS) in
     * hpo.cpp), and most backends rejected it only once the run started. */
    const std::string message =
        "THREADS must be an integer from 1 to 2147483647, got '0'";
    const std::vector<KeyHome> homes = {
        /* The short form, directly inside OPTIMIZER (line 4). */
        {"THREADS", "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\n", "ENDOPTIMIZER\n"},
        /* The STRATEGY-level optimiser (line 6). */
        attr_home("THREADS"),
        /* EXPLORE, REFINE and INNER. */
        {"THREADS",
         "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB DLIB\nALGO BFGS\n"
         "EXPLORE\nLIB DLIB\nALGO GFS\n",
         "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n"},
        {"THREADS",
         "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\n"
         "EXPLORE\nLIB DLIB\nALGO GFS\nENDEXPLORE\n"
         "REFINE\nLIB DLIB\nALGO BFGS\n",
         "ENDREFINE\nENDSTRATEGY\nENDOPTIMIZER\n"},
        {"THREADS",
         "OPTIMIZER\nSTRATEGY\nTYPE PLAIN\nLIB NLOPT\nALGO G_MLSL_LDS\n"
         "INNER\nLIB NLOPT\nALGO LN_SBPLX\n",
         "ENDINNER\nENDSTRATEGY\nENDOPTIMIZER\n"}};
    for (const auto& h : homes) {
        INFO(h.config("0"));
        CHECK_THAT(parse_error(h.config("0")),
                   Catch::StartsWith(h.at() + message) &&
                   Catch::EndsWith("Line: THREADS 0"));
        CHECK(parse_error(h.config("1")) == "");
    }

    /* No `tadah refit` stage uses THREADS, so any THREADS line there is an
     * error ("tadah refit: a stage OPTIMIZER block rejects the keys the
     * stage does not use"). */
}

TEST_CASE("tadah refit: INIT MANUAL operands are finite numbers",
          "[HPO][refit][strict_numbers]")
{
    CHECK(refit_error("INIT", "PHI MANUAL 2.55 1.5") == "");
    CHECK_THAT(refit_error("INIT", "PHI MANUAL inf"),
               Catch::StartsWith("INIT PHI MANUAL <mu> must be a finite "
                                 "number, got 'inf'"));
    CHECK_THAT(refit_error("INIT", "RHO MANUAL 2.55 nan"),
               Catch::StartsWith("INIT RHO MANUAL <eta> must be a finite "
                                 "number, got 'nan'"));
}

TEST_CASE("Strict numbers: PC_OBSERVABLE values are finite numbers and "
          "integers that fit", "[HPO][hpo_targets][strict_numbers][observable]")
{
    const std::string base =
        "cohesive_energy --species Ta --hotvar a0 2.7 3.9 3.3 ";
    auto error = [](const std::string& cmd) {
        try {
            (void)tadah::hpo::parse_pc_observable_line(cmd);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    CHECK(error(base + "--target 8.1 --weight 100") == "");
    CHECK_THAT(error(base + "--target inf --weight 100"),
               Catch::Contains("flag '--target' expected a finite number, "
                               "got 'inf'"));
    CHECK_THAT(error(base + "--target 8.1 --weight nan"),
               Catch::Contains("flag '--weight' expected a finite number, "
                               "got 'nan'"));
    /* The integer flags take decimal digits only, within an int: stoll
     * then a narrowing cast turned 4294967300 into 4, and a sign was read
     * (`--maxiter -0` was 0; a negative value got a range error that did
     * not name it). --supercell takes at least one cell per axis. */
    const std::string ok = base + "--target 8.1 --weight 1 ";
    CHECK(error(ok + "--supercell 3 3 3 --maxiter 0 --maxeval 100") == "");
    struct Case { std::string flags, flag, bound, tok; };
    for (const auto& c : std::vector<Case>{
             {"--supercell 4294967300 4 4", "--supercell nx", "1", "4294967300"},
             {"--supercell -0 4 4", "--supercell nx", "1", "-0"},
             {"--supercell 4 0 4", "--supercell ny", "1", "0"},
             {"--supercell 4 4 +4", "--supercell nz", "1", "+4"},
             {"--supercell 4 4 -2", "--supercell nz", "1", "-2"},
             {"--maxiter 3000000000", "--maxiter", "0", "3000000000"},
             {"--maxiter -0", "--maxiter", "0", "-0"},
             {"--maxiter -5", "--maxiter", "0", "-5"},
             {"--maxeval -0", "--maxeval", "0", "-0"},
             {"--maxeval +3", "--maxeval", "0", "+3"},
             {"--maxeval 2.5", "--maxeval", "0", "2.5"}}) {
        INFO(c.flags);
        CHECK_THAT(error(ok + c.flags),
                   Catch::Contains("PC_OBSERVABLE: flag '" + c.flag +
                                   "' must be an integer from " + c.bound +
                                   " to 2147483647, got '" + c.tok + "'"));
    }
}

TEST_CASE("Strict numbers: a value outside its range is named in the error",
          "[HPO][hpo_targets][strict_numbers]")
{
    /* A number in the grammar but out of the key's range used to give a
     * message without the value ("PC_EMAE weight must be nonnegative",
     * "PC_FAILPROBE --s0 must be finite and > 0", "--failvar F: <scale>
     * must be > 0"). */
    struct Case { std::string line, message; };
    for (const auto& c : std::vector<Case>{
             {"PC_EMAE 0 -1", "PC_EMAE <weight> must be a number >= 0, got '-1'"},
             {"PC_ERMSE 0 -0.5",
              "PC_ERMSE <weight> must be a number >= 0, got '-0.5'"},
             {"PC_ERMSE (1) 0 -2",
              "PC_ERMSE <weight> must be a number >= 0, got '-2'"},
             {"PC_EMAE 0 1 --skip-above -1",
              "PC_EMAE --skip-above must be a positive number, got '-1'"},
             {"PC_SRMSE 0 1 --skip-rel 0",
              "PC_SRMSE --skip-rel must be a positive number, got '0'"},
             {"PC_FAILPROBE --script p.in --outvar S --s0 -1",
              "PC_FAILPROBE --s0 must be a positive number, got '-1'"},
             {"PC_FAILPROBE --script p.in --outvar S --s0 0",
              "PC_FAILPROBE --s0 must be a positive number, got '0'"}}) {
        INFO(c.line);
        CHECK_THAT(parse_error(kOptimizer + c.line + "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message) &&
                   Catch::EndsWith("Line: " + c.line));
    }
    CHECK(parse_error(kOptimizer + "PC_EMAE 0 0 --skip-above 5\n") == "");

    TempFile script(kSimScript, ".in");
    const std::string head =
        "--script " + script.path.string() + " --varloss E 8.1 1 ";
    for (const std::string tok : {"-1", "0", "-0"}) {
        INFO("--failvar F " << tok);
        const std::string message =
            "--failvar F <scale> must be a positive number, got '" + tok + "'";
        CHECK_THAT(lammps_sim_error(head + "--failvar F " + tok),
                   Catch::StartsWith(message));
        CHECK_THAT(parse_error(kOptimizer + "PC_LAMMPS " + head +
                               "--failvar F " + tok + "\n"),
                   Catch::StartsWith("Error on line 9: " + message));
    }
    CHECK(lammps_sim_error(head + "--failvar F 0.5") == "");
}

TEST_CASE("Strict numbers: PC_FAILPROBE --failscore is a positive finite "
          "number; PC_LAMMPS --failscore any finite number",
          "[HPO][hpo_targets][lammps_sim][strict_numbers][failscore]"
          "[failprobe]")
{
    /* On the PC_FAILPROBE line --failscore is a failure score: the value
     * the probe's S reads when the probe itself fails, and an S at or above
     * it is taken for a failed probe (HPO_Host::run_lammps). S is a
     * nonnegative measure, so with 0 or below every probe read as failed --
     * band D for every evaluation under FAILSCORE_GRADING ON. It takes the
     * rule of FAILSCORE, at its line. */
    TempFile probe("clear\nrun 0\nvariable S equal 0.0\n", ".in");
    const std::string head =
        "PC_FAILPROBE --script " + probe.path.string() + " --outvar S ";
    for (const std::string tok : {"0", "-0", "-1", "-1e9", "abc", "5x",
                                  "inf", "0x1p4"}) {
        INFO("PC_FAILPROBE --failscore " << tok);
        const std::string line = head + "--failscore " + tok;
        CHECK_THAT(parse_error(kOptimizer + line + "\n"),
                   Catch::StartsWith("Error on line 9: PC_FAILPROBE "
                                     "--failscore must be a positive finite "
                                     "number, got '" + tok + "'") &&
                   Catch::EndsWith("Line: " + line));
    }
    /* A positive value; without the option, the root FAILSCORE in force. */
    const auto own = parse_root("FAILSCORE 1e9\n" + head + "--failscore 1e6\n");
    REQUIRE(own.failprobe_idx >= 0);
    CHECK(own.LS.at(static_cast<std::size_t>(own.failprobe_idx)).failscore ==
          1e6);
    const auto dflt = parse_root("FAILSCORE 1e9\n" + head + "\n");
    REQUIRE(dflt.failprobe_idx >= 0);
    CHECK(dflt.LS.at(static_cast<std::size_t>(dflt.failprobe_idx)).failscore ==
          1e9);

    /* On a PC_LAMMPS line --failscore is no failure score: it is the value
     * substituted for the line's variables when its simulation fails, which
     * then goes through the loss against the target like a measured value
     * (a variable of either sign). 0 and negative values stay accepted, next
     * to a negative-weight --varloss term. */
    TempFile script(kSimScript, ".in");
    for (const std::string tok : {"0", "-1e6"}) {
        INFO("PC_LAMMPS --failscore " << tok);
        const auto trg = parse_root("FAILSCORE 1e9\nPC_LAMMPS --script " +
                                    script.path.string() +
                                    " --varloss E -8.1 -2 --failscore " + tok +
                                    "\n");
        REQUIRE(trg.LS.size() == 1u);
        CHECK(trg.LS.front().failscore == std::stod(tok));
        CHECK(trg.LS.front().w_varloss.at(0) == -2.0);
        CHECK(trg.FAIL_SCORE == 1e9);
    }
}

TEST_CASE("Strict numbers: RHOGATE and RHOBARGATE", "[HPO][hpo_targets][strict_numbers][rhogate]")
{
    /* The density gates read their numbers in the grammar of every other
     * key: a real one whole finite decimal token (no hexadecimal float, no
     * inf or nan, no subnormal), NPTS an integer in decimal digits only (no
     * sign, fraction or exponent). Their first reader took strtod's forms
     * (`RHOGATE 0x1p-10`, a subnormal R_LO) and an NPTS written as a real
     * (`NPTS 2e3`, `NPTS +100`, `NPTS 100.0`). Errors name the line. */
    const std::string at = "Error on line 9: ";
    const auto err = [](const std::string& line) {
        return parse_error(kOptimizer + line + "\n");
    };
    SECTION("accepted") {
        /* a gate needs an OPTIM line that arms it (a density key) */
        const std::string arm = "OPTIM AMPGRIDMB (1) 0 1\n";
        auto armed = [&](const std::string& lines) {
            TempFile cfg(kOptimizer + lines + arm);
            tadah::core::Context ctx;
            ctx.add("AMPGRIDMB", 0.5);
            return std::make_unique<tadah::hpo::HPOTargets>(
                ctx, cfg.path.string());
        };
        for (const std::string line :
                 {"RHOGATE 1e-5", "RHOGATE 0 R_LO 1.7168 NPTS 2 SEED WARN",
                  "RHOGATE +1e-5 npts 10000000", "RHOGATE 1E-5 R_LO 0",
                  "RHOBARGATE bcc 3.304 0.98 1.1",
                  "RHOBARGATE fcc 4.05 .9 1.10 NPTS 3 SEED ERROR"}) {
            INFO(line);
            CHECK_NOTHROW(armed(line + "\n"));
        }
        auto trg = armed("RHOGATE 1e-5 NPTS 2000\n"
                         "RHOBARGATE hcp 2.9 0.9 1.1 NPTS 3501\n");
        REQUIRE(trg->RHOGATE);
        CHECK(trg->RHOGATE->npts == 2000u);
        REQUIRE(trg->RHOBARGATE);
        CHECK(trg->RHOBARGATE->npts == 3501u);
    }
    SECTION("a real number is one finite decimal token") {
        std::vector<std::string> bad = kNotFinite;
        bad.insert(bad.end(), {"0x1p-17", "1e-320"});
        for (const auto& tok : bad) {
            INFO(tok);
            CHECK_THAT(err("RHOGATE " + tok),
                       Catch::StartsWith(at + "RHOGATE tol must be a finite "
                                         "number, got '" + tok + "'"));
            CHECK_THAT(err("RHOGATE 1e-5 R_LO " + tok),
                       Catch::StartsWith(at + "RHOGATE R_LO must be a finite "
                                         "number, got '" + tok + "'"));
            CHECK_THAT(err("RHOBARGATE bcc " + tok + " 0.9 1.1"),
                       Catch::StartsWith(at + "RHOBARGATE a0 must be a finite "
                                         "number, got '" + tok + "'"));
            CHECK_THAT(err("RHOBARGATE bcc 3.3 " + tok + " 1.1"),
                       Catch::StartsWith(at + "RHOBARGATE f_lo must be a "
                                         "finite number, got '" + tok + "'"));
            CHECK_THAT(err("RHOBARGATE bcc 3.3 0.9 " + tok),
                       Catch::StartsWith(at + "RHOBARGATE f_hi must be a "
                                         "finite number, got '" + tok + "'"));
        }
        CHECK_THAT(err("RHOGATE 1e-320"), Catch::Contains("(too small"));
    }
    SECTION("NPTS is an integer in decimal digits within its range") {
        std::vector<std::string> bad = kNotUnsigned;
        bad.insert(bad.end(), {"100.0", "2e3", "+500", "10000001"});
        for (const auto& tok : bad) {
            INFO(tok);
            CHECK_THAT(err("RHOGATE 1e-5 NPTS " + tok),
                       Catch::StartsWith(at + "RHOGATE NPTS must be an "
                                         "integer from 2 to 10000000, got '" +
                                         tok + "'"));
            CHECK_THAT(err("RHOBARGATE bcc 3.3 0.9 1.1 NPTS " + tok),
                       Catch::StartsWith(at + "RHOBARGATE NPTS must be an "
                                         "integer from 3 to 10000000, got '" +
                                         tok + "'"));
        }
        CHECK_THAT(err("RHOGATE 1e-5 NPTS 1"),
                   Catch::StartsWith(at + "RHOGATE NPTS must be an integer "
                                     "from 2 to 10000000, got '1'"));
        CHECK_THAT(err("RHOBARGATE bcc 3.3 0.9 1.1 NPTS 2"),
                   Catch::StartsWith(at + "RHOBARGATE NPTS must be an integer "
                                     "from 3 to 10000000, got '2'"));
    }
    SECTION("a missing value and an extra token") {
        CHECK_THAT(err("RHOGATE"),
                   Catch::StartsWith(at + "RHOGATE: the tolerance is missing"));
        CHECK_THAT(err("RHOGATE 1e-5 NPTS"),
                   Catch::StartsWith(at + "RHOGATE: NPTS needs a value"));
        CHECK_THAT(err("RHOGATE 1e-5 7"),
                   Catch::StartsWith(at + "RHOGATE: unexpected token '7'"));
        CHECK_THAT(err("RHOBARGATE bcc 3.3 0.9"),
                   Catch::StartsWith(at + "RHOBARGATE: the lattice, a0, f_lo "
                                     "and f_hi are required"));
        CHECK_THAT(err("RHOBARGATE bcc 3.3 0.9 1.1 7"),
                   Catch::StartsWith(at + "RHOBARGATE: unexpected token '7'"));
    }
}

TEST_CASE("Strict numbers: a hexadecimal float is an error for every key",
          "[HPO][hpo_targets][strict_numbers][hex]")
{
    /* One number grammar for every key, the one LAMMPS reads: a sign,
     * decimal digits with at most one '.', an exponent. std::stod also read
     * C99 hexadecimal floats, so `FAILSCORE 0x1p40` loaded as 1.0995e+12
     * and `STEP 0x1p-3` as 0.125, while PC_<METRIC> (std::from_chars)
     * rejected them. */
    struct Case { std::string line, message; };
    for (const auto& c : std::vector<Case>{
             {"FAILSCORE 0x1p40",
              "FAILSCORE must be a positive finite number, got '0x1p40'"},
             {"LOSS HUBER 0x1p-3",
              "LOSS HUBER: parameter '0x1p-3' is not a number"},
             {"PC_WORKER_TIMEOUT 0x10",
              "PC_WORKER_TIMEOUT must be a finite number of seconds >= 0 (0 "
              "derives the deadline from each script's timeout), got '0x10'"},
             {"PC_SCRIPT_TIMEOUT 0x10",
              "PC_SCRIPT_TIMEOUT must be 'off', 'unlimited' or a time "
              "[[HH:]MM:]SS, got '0x10'"},
             {"PC_SCRIPT_TIMEOUT 1:0x10",
              "PC_SCRIPT_TIMEOUT must be 'off', 'unlimited' or a time "
              "[[HH:]MM:]SS, got '1:0x10'"},
             {"MAXTIME 0x1p3",
              "MAXTIME must be 'off', 'unlimited' or a time [[HH:]MM:]SS, "
              "got '0x1p3'"},
             {"PC_ERMSE 0x0 1",
              "PC_ERMSE <target> must be a finite number, got '0x0'"},
             {"PC_ERMSE 0 0x1p-4",
              "PC_ERMSE <weight> must be a finite number, got '0x1p-4'"},
             {"PC_EMAE 0 1 --skip-rel 0x1p1",
              "PC_EMAE --skip-rel must be a positive number, got '0x1p1'"},
             {"PC_FAILPROBE --script p.in --outvar S --gate 0x1p1",
              "PC_FAILPROBE --gate must be a finite number, got '0x1p1'"},
             {"PC_FAILPROBE --script p.in --outvar S --s0 0x1p1",
              "PC_FAILPROBE --s0 must be a positive number, got '0x1p1'"}}) {
        INFO(c.line);
        CHECK_THAT(parse_error(kOptimizer + c.line + "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message));
    }

    /* The optimiser attributes: the block-wide and per-key STEP, a numeric
     * PARAM value, and the real-valued stop criteria (check_finite_key
     * covers those with its hexadecimal tokens). */
    const auto step = attr_home("STEP");
    CHECK_THAT(parse_error(step.config("0x1p-3")),
               Catch::StartsWith(step.at() + "STEP must be a finite number, "
                                 "got '0x1p-3'"));
    CHECK_THAT(parse_error(step.config("CGRID2B 0x1p-3")),
               Catch::StartsWith(step.at() + "STEP CGRID2B must be a finite "
                                 "number, got '0x1p-3'"));
    const auto param = strategy_home("PARAM", "PLAIN", "DLIB", "BFGS");
    CHECK_THAT(parse_error(param.config("fd_step 0x1p-7")),
               Catch::StartsWith(param.at() + "PARAM fd_step must be a finite "
                                 "number, got '0x1p-7'"));
    CHECK(parse_config(param.config("fd_step 0.0078125"))
              .spec.param.at("fd_step") == 0.0078125);

    /* OPTIM: explicit bounds, the AUTO options and ASINH's scale. */
    auto optim_error = [](const std::string& line) {
        TempFile cfg(kOptimizer + line + "\n");
        tadah::core::Context ctx;
        ctx.add("ATOM", "Ta");
        ctx.add("WATOM", 1.0);
        try {
            tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    for (const auto& c : std::vector<Case>{
             {"OPTIM WATOM (1) 0x1p-3 5.0",
              "OPTIM WATOM: <low> must be a finite number (got '0x1p-3')"},
             {"OPTIM WATOM (1) 0.1 0x5",
              "OPTIM WATOM: <high> must be a finite number (got '0x5')"},
             {"OPTIM WATOM (1) AUTO RATIO 0x2",
              "OPTIM WATOM: AUTO RATIO must be a finite number (got '0x2')"},
             {"OPTIM WATOM (1) 0.1 5.0 ASINH 0x1",
              "OPTIM WATOM: ASINH <c> must be a finite number (got '0x1')"}}) {
        INFO(c.line);
        CHECK_THAT(optim_error(c.line),
                   Catch::StartsWith("Error on line 9: " + c.message));
    }

    /* The PC_LAMMPS options, through LammpsSim and at the PC_LAMMPS line. */
    TempFile script(kSimScript, ".in");
    const std::string head = "--script " + script.path.string() + " ";
    for (const auto& c : std::vector<Case>{
             {"--varloss E 0x1p3 1",
              "--varloss E <target> must be a finite number, got '0x1p3'"},
             {"--varloss E 8.1 0x1",
              "--varloss E <weight> must be a finite number, got '0x1'"},
             {"--varloss E 8.1 1 HUBER 0x1p-3",
              "--varloss E HUBER parameter must be a finite number, got "
              "'0x1p-3'"},
             {"--varloss E 8.1 1 --failscore 0x1p40",
              "--failscore must be a finite number, got '0x1p40'"},
             {"--varloss E 8.1 1 --hotvar a0 0x1p1 4.5",
              "--hotvar a0 <min> must be a finite number, got '0x1p1'"},
             {"--varloss E 8.1 1 --hotvar a0 2.5 0x1p2",
              "--hotvar a0 <max> must be a finite number, got '0x1p2'"},
             {"--varloss E 8.1 1 --hotvar a0 2.5 4.5 0x1.8p1",
              "--hotvar a0 <initial> must be a finite number, got '0x1.8p1'"},
             {"--varloss E 8.1 1 --failvar F 0x1p-1",
              "--failvar F <scale> must be a positive number, got '0x1p-1'"},
             {"--varloss E 8.1 1 --maxtime 0x10",
              "--maxtime must be 'off', 'unlimited' or a time [[HH:]MM:]SS, "
              "got '0x10'"}}) {
        INFO(c.line);
        CHECK_THAT(lammps_sim_error(head + c.line),
                   Catch::StartsWith(c.message));
        CHECK_THAT(parse_error(kOptimizer + "PC_LAMMPS " + head + c.line +
                               "\n"),
                   Catch::StartsWith("Error on line 9: " + c.message));
    }

    /* PC_OBSERVABLE values. */
    auto observable_error = [](const std::string& cmd) {
        try {
            (void)tadah::hpo::parse_pc_observable_line(cmd);
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string obs =
        "cohesive_energy --species Ta --hotvar a0 2.7 3.9 3.3 ";
    CHECK_THAT(observable_error(obs + "--target 0x8 --weight 100"),
               Catch::Contains("flag '--target' expected a finite number, got "
                               "'0x8'"));
    CHECK_THAT(observable_error(obs + "--target 8.1 --weight 0x64"),
               Catch::Contains("flag '--weight' expected a finite number, got "
                               "'0x64'"));

    /* The environment override of PC_WORKER_TIMEOUT. */
    CHECK_THROWS_WITH(tadah::hpo::HPOTargets::parse_pc_worker_timeout(
                          "0x10", "TADAH_PC_WORKER_TIMEOUT"),
                      Catch::StartsWith("TADAH_PC_WORKER_TIMEOUT must be a "
                                        "finite number of seconds >= 0"));

    /* The `tadah refit` numbers. */
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM GTOL 0x1p-40 "
                                     "ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: GTOL must be a finite number, got "
                                 "'0x1p-40'"));
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM FTOL_REL "
                                     "0x1p-40 ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: FTOL_REL must be a finite number, "
                                 "got '0x1p-40'"));
    CHECK_THAT(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                      "MAXEVAL 100 INNER LIB NLOPT ALGO "
                                      "LN_SBPLX XTOL_REL 0x1p-20 ENDINNER "
                                      "ENDOPTIMIZER"),
               Catch::StartsWith("EXPLORE INNER: XTOL_REL must be a finite "
                                 "number, got '0x1p-20'"));
    CHECK_THAT(refit_error("INIT", "PHI MANUAL 0x1p1"),
               Catch::StartsWith("INIT PHI MANUAL <mu> must be a finite "
                                 "number, got '0x1p1'"));
    CHECK_THAT(refit_error("INIT", "RHO MANUAL 2.55 0x1p0"),
               Catch::StartsWith("INIT RHO MANUAL <eta> must be a finite "
                                 "number, got '0x1p0'"));
}

namespace {

/* The keys each `tadah refit` stage reads, as its messages list them. */
const std::string kExploreReads = "LIB, ALGO, MAXEVAL, INNER";
/* FTOL_ABS: dlib's LBFGS stops on it (an INNER block's and REFINE's). */
const std::string kExploreInnerReads =
    "LIB, ALGO, MAXEVAL, FTOL_REL, FTOL_ABS, XTOL_REL";
const std::string kRefineReads =
    "LIB, ALGO, MAXEVAL/MAXITER, FTOL_REL, FTOL_ABS, GTOL";
const std::string kFuseReads =
    "LIB, ALGO, MAXEVAL/MAXITER, FTOL_REL, GTOL";

}  // namespace

TEST_CASE("tadah refit: a stage OPTIMIZER block rejects the keys the stage "
          "does not use", "[HPO][refit][strict_numbers]")
{
    /* Every optimiser attribute of `tadah hpo` was parsed in a refit
     * OPTIMIZER block, but each stage reads only a few (refit.rst): THREADS,
     * MAXTIME, STOPVAL, POPULATION, STEP, ... were accepted and did
     * nothing. FUSE reads GTOL (its Ceres gradient tolerance; it was
     * refused here, while the curve FUSE refine ran with REFINE's). */
    struct Stage {
        std::string key;      // the stage block
        std::string head;     // up to the key under test
        std::string tail;     // after it
        std::string prefix;   // the message's
        std::string reads;
        std::vector<std::string> unused;
    };
    const std::vector<std::string> common_unused = {
        "MAXTIME 60", "THREADS 4", "THREADS 0", "STOPVAL 0.1",
        "XTOL_ABS 1e-6", "POPULATION 10", "STEP 0.1", "VECTOR_ARR 2",
        "GRAD_STRATEGY NUMERICAL", "PARAM fd_step 0.01"};
    auto with = [&](std::vector<std::string> extra) {
        extra.insert(extra.begin(), common_unused.begin(), common_unused.end());
        return extra;
    };
    const std::vector<Stage> stages = {
        {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 ",
         " ENDOPTIMIZER", "EXPLORE", kExploreReads,
         with({"FTOL_REL 1e-6", "FTOL_ABS 1e-6", "XTOL_REL 1e-6",
               "GTOL 1e-8", "MAXITER 50"})},
        {"EXPLORE",
         "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER LIB NLOPT "
         "ALGO LN_SBPLX ",
         " ENDINNER ENDOPTIMIZER", "EXPLORE INNER", kExploreInnerReads,
         with({"GTOL 1e-8", "MAXITER 50"})},
        {"REFINE", "OPTIMIZER LIB CERES ALGO LM ", " ENDOPTIMIZER", "REFINE",
         kRefineReads,
         with({"XTOL_REL 1e-6", "INNER LIB NLOPT ALGO LN_SBPLX ENDINNER"})},
        {"FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG ", " ENDOPTIMIZER", "FUSE",
         kFuseReads,
         with({"XTOL_REL 1e-6", "INNER LIB NLOPT ALGO LN_SBPLX ENDINNER"})}};
    for (const auto& s : stages) {
        for (const auto& kv : s.unused) {
            const std::string key = kv.substr(0, kv.find(' '));
            INFO(s.key << ": " << s.head << kv << s.tail);
            CHECK(refit_error(s.key, s.head + kv + s.tail) ==
                  s.prefix + ": " + key + " is not used by a tadah refit "
                  "stage (it reads " + s.reads + ")");
        }
    }

    /* The keys a stage reads are accepted. */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT ALGO LN_SBPLX "
                                 "MAXEVAL 50 FTOL_REL 1e-7 XTOL_REL 1e-6 "
                                 "ENDINNER ENDOPTIMIZER") == "");
    /* MAXEVAL or its alias MAXITER, one of them (a block with both is
     * refused: the case "a key set twice in a stage"). */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXITER 900 "
                                "FTOL_REL 1e-14 GTOL 1e-14 "
                                "ENDOPTIMIZER") == "");
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG MAXITER 2000 "
                              "FTOL_REL 1e-14 ENDOPTIMIZER") == "");
    CHECK(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 2000 "
                              "FTOL_REL 1e-14 GTOL 1e-12 ENDOPTIMIZER") == "");

    /* SEED and FTOL_ABS keep the messages that say what to use instead. */
    CHECK_THAT(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG SEED 7 "
                                   "ENDOPTIMIZER"),
               Catch::StartsWith("FUSE: SEED is not accepted in a tadah refit "
                                 "OPTIMIZER block") &&
               Catch::Contains("REFIT_COMPACT_SEED"));
    CHECK_THAT(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM FTOL_ABS "
                                     "1e-12 ENDOPTIMIZER"),
               Catch::StartsWith("REFINE: FTOL_ABS is not supported") &&
               Catch::Contains("Use FTOL_REL"));
    /* A word that is no optimiser key at all is still an unknown key. */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVALS 5 "
                                "ENDOPTIMIZER") ==
          "REFINE: unknown key 'MAXEVALS' in the OPTIMIZER block");
    /* The bare FTOL of the Ceres stages points to FTOL_REL. */
    CHECK_THAT(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO DOGLEG FTOL "
                                   "1e-12 ENDOPTIMIZER"),
               Catch::StartsWith("FUSE: 'FTOL' is not supported") &&
               Catch::Contains("Use FTOL_REL"));
}

TEST_CASE("tadah refit: every stage OPTIMIZER block and INNER block names "
          "its ALGO", "[HPO][refit]")
{
    using tadah::hpo::refit::RefitParser;
    /* An INNER block without ALGO meant no local search (NONE), while an
     * EXPLORE optimiser without an INNER block runs the default one
     * (LN_SBPLX). ALGO is what the block is read for, so it is required;
     * NONE is written out. */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER LIB NLOPT MAXEVAL 50 "
                                 "ENDINNER ENDOPTIMIZER") ==
          "EXPLORE INNER: ALGO is not set; set ALGO in the INNER block (NONE "
          "for no local search)");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER ENDINNER ENDOPTIMIZER") ==
          "EXPLORE INNER: ALGO is not set; set ALGO in the INNER block (NONE "
          "for no local search)");
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT MAXEVAL 100 "
                                 "ENDOPTIMIZER") ==
          "EXPLORE: ALGO is not set; set ALGO in the OPTIMIZER block "
          "(UNIFORM for no global search)");
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES MAXEVAL 100 "
                                "ENDOPTIMIZER") ==
          "REFINE: ALGO is not set; set ALGO in the OPTIMIZER block (NONE to "
          "skip the stage)");
    CHECK(refit_error("FUSE", "OPTIMIZER MAXEVAL 100 ENDOPTIMIZER") ==
          "FUSE: ALGO is not set; set ALGO in the OPTIMIZER block (NONE to "
          "skip the stage)");

    /* NONE is the explicit spelling, without LIB (it runs no optimiser). */
    const auto none = RefitParser::parse(refit_context(
        "EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER "
                   "ALGO NONE ENDINNER ENDOPTIMIZER"));
    REQUIRE(none.explore.opt.inner);
    CHECK(tadah::hpo::refit::refit_explore_inner_algo(
              *none.explore.opt.inner) == "NONE");
    const auto sbplx = RefitParser::parse(refit_context(
        "EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER LIB "
                   "NLOPT ALGO LN_SBPLX ENDINNER ENDOPTIMIZER"));
    REQUIRE(sbplx.explore.opt.inner);
    CHECK(tadah::hpo::refit::refit_explore_inner_algo(
              *sbplx.explore.opt.inner) == "SBPLX");
}

TEST_CASE("tadah refit: a stage block that runs an optimiser names the "
          "library of its ALGO in LIB", "[HPO][refit]")
{
    /* As in `tadah hpo`, where every block, INNER included, sets LIB and
     * ALGO, a refit stage block that runs an optimiser names its library.
     * LIB used to be optional wherever the ALGO named the optimiser
     * (GN_MLSL_LDS, LN_SBPLX, LM, DLIB_GLOBAL, LD_LBFGS in an INNER block)
     * and required elsewhere (LD_LBFGS in REFINE), and a LIB that did not
     * match the ALGO was ignored (LIB CERES ALGO GN_MLSL_LDS ran MLSL). */
    using tadah::hpo::refit::RefitParser;
    using tadah::hpo::refit::refit_explore_inner_algo;
    using tadah::hpo::refit::refit_explore_outer_algo;
    using tadah::hpo::refit::refit_fuse_algo;
    using tadah::hpo::refit::refit_refine_algo;

    /* Every ALGO spelling of a stage, the library it belongs to and the
     * optimiser it selects. (GN_MLSL, GN_DIRECT and GN_CRS2, which ran
     * GN_MLSL_LDS, GN_DIRECT_L and GN_CRS2_LM, are refused: see "ALGO
     * GN_MLSL, GN_DIRECT and GN_CRS2 are refused".) */
    struct Algo { std::string stage, algo, lib, canonical; };
    const std::vector<Algo> algos = {
        {"EXPLORE", "GN_MLSL_LDS", "NLOPT", "MLSL"},
        {"EXPLORE", "MLSL", "NLOPT", "MLSL"},
        {"EXPLORE", "GN_DIRECT_L", "NLOPT", "DIRECT"},
        {"EXPLORE", "DIRECT", "NLOPT", "DIRECT"},
        {"EXPLORE", "GN_CRS2_LM", "NLOPT", "CRS2"},
        {"EXPLORE", "CRS2", "NLOPT", "CRS2"},
        {"EXPLORE", "GLOBAL", "DLIB", "DLIB_GLOBAL"},
        {"EXPLORE", "DLIB_GLOBAL", "DLIB", "DLIB_GLOBAL"},
        {"EXPLORE INNER", "LN_SBPLX", "NLOPT", "SBPLX"},
        {"EXPLORE INNER", "SBPLX", "NLOPT", "SBPLX"},
        {"EXPLORE INNER", "LN_NELDERMEAD", "NLOPT", "NELDERMEAD"},
        {"EXPLORE INNER", "NELDERMEAD", "NLOPT", "NELDERMEAD"},
        {"EXPLORE INNER", "LN_BOBYQA", "NLOPT", "BOBYQA"},
        {"EXPLORE INNER", "BOBYQA", "NLOPT", "BOBYQA"},
        {"EXPLORE INNER", "LN_PRAXIS", "NLOPT", "PRAXIS"},
        {"EXPLORE INNER", "PRAXIS", "NLOPT", "PRAXIS"},
        {"EXPLORE INNER", "LD_LBFGS", "NLOPT", "NLOPT_LBFGS"},
        {"EXPLORE INNER", "NLOPT_LBFGS", "NLOPT", "NLOPT_LBFGS"},
        {"EXPLORE INNER", "LBFGS", "DLIB", "DLIB_LBFGS"},
        {"EXPLORE INNER", "DLIB_LBFGS", "DLIB", "DLIB_LBFGS"},
        {"REFINE", "LM", "CERES", "CERES_LM"},
        {"REFINE", "CERES_LM", "CERES", "CERES_LM"},
        {"REFINE", "DOGLEG", "CERES", "CERES_DOGLEG"},
        {"REFINE", "CERES_DOGLEG", "CERES", "CERES_DOGLEG"},
        {"REFINE", "LD_LBFGS", "NLOPT", "NLOPT_LBFGS"},
        {"REFINE", "NLOPT_LBFGS", "NLOPT", "NLOPT_LBFGS"},
        {"REFINE", "LBFGS", "DLIB", "DLIB_LBFGS"},
        {"REFINE", "DLIB_LBFGS", "DLIB", "DLIB_LBFGS"},
        {"FUSE", "LM", "CERES", "CERES_LM"},
        {"FUSE", "CERES_LM", "CERES", "CERES_LM"},
        {"FUSE", "DOGLEG", "CERES", "CERES_DOGLEG"},
        {"FUSE", "CERES_DOGLEG", "CERES", "CERES_DOGLEG"}};
    const auto library = [](const std::string& lib) -> std::string {
        if (lib == "NLOPT") return "an NLopt algorithm";
        if (lib == "DLIB") return "a dlib algorithm";
        return "a Ceres algorithm";
    };
    /* The algorithms of each stage in each library, as the stage's legal
     * list names them: a LIB that does not match the ALGO may be the one
     * meant, so the message names them too. */
    const std::map<std::pair<std::string, std::string>, std::string> in_lib = {
        {{"EXPLORE", "NLOPT"}, "GN_MLSL_LDS|GN_DIRECT_L|GN_CRS2_LM"},
        {{"EXPLORE", "DLIB"}, "GLOBAL"},
        {{"EXPLORE INNER", "NLOPT"},
         "LN_SBPLX|LN_NELDERMEAD|LN_BOBYQA|LN_PRAXIS|LD_LBFGS"},
        {{"EXPLORE INNER", "DLIB"}, "LBFGS"},
        {{"REFINE", "CERES"}, "LM|DOGLEG"},
        {{"REFINE", "NLOPT"}, "LD_LBFGS"},
        {{"REFINE", "DLIB"}, "LBFGS"},
        {{"FUSE", "CERES"}, "LM|DOGLEG"}};
    const auto or_with = [&](const std::string& stage,
                             const std::string& lib) -> std::string {
        const auto it = in_lib.find({stage, lib});
        return it == in_lib.end()
                   ? ""
                   : ", or with LIB " + lib + " set ALGO " + it->second;
    };
    /* `lib_line` is "" or "LIB <x> "; the stage block around the ALGO. */
    const auto block = [](const Algo& a, const std::string& lib_line) {
        if (a.stage == "EXPLORE")
            return std::string("OPTIMIZER ") + lib_line + "ALGO " + a.algo +
                   " MAXEVAL 100 ENDOPTIMIZER";
        if (a.stage == "EXPLORE INNER")
            return "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER " +
                   lib_line + "ALGO " + a.algo + " MAXEVAL 50 ENDINNER "
                   "ENDOPTIMIZER";
        return std::string("OPTIMIZER ") + lib_line + "ALGO " + a.algo +
               " MAXEVAL 100 ENDOPTIMIZER";
    };
    const auto key = [](const Algo& a) {
        return a.stage == "EXPLORE INNER" ? std::string("EXPLORE") : a.stage;
    };
    const auto canonical = [&](const Algo& a, const std::string& lib_line) {
        const auto rc =
            RefitParser::parse(refit_context(key(a), block(a, lib_line)));
        if (a.stage == "EXPLORE") return refit_explore_outer_algo(rc.explore.opt);
        if (a.stage == "EXPLORE INNER")
            return refit_explore_inner_algo(*rc.explore.opt.inner);
        if (a.stage == "REFINE") return refit_refine_algo(rc.refine.opt);
        return refit_fuse_algo(rc.fuse.opt);
    };
    for (const auto& a : algos) {
        INFO(a.stage << ": ALGO " << a.algo);
        /* With its library the ALGO selects its optimiser, in any case. */
        CHECK(canonical(a, "LIB " + a.lib + " ") == a.canonical);
        std::string lower = a.lib;
        for (char& ch : lower) ch = static_cast<char>(std::tolower(ch));
        CHECK(canonical(a, "LIB " + lower + " ") == a.canonical);
        /* Without LIB, or with another library, the block is rejected. */
        CHECK(refit_error(key(a), block(a, "")) ==
              a.stage + ": LIB is not set; set LIB " + a.lib + " (ALGO " +
                  a.algo + " is " + library(a.lib) + ")");
        for (const std::string other : {"NLOPT", "DLIB", "CERES", "TADAH"}) {
            if (other == a.lib) continue;
            INFO("LIB " << other);
            CHECK(refit_error(key(a), block(a, "LIB " + other + " ")) ==
                  a.stage + ": LIB " + other + " does not match ALGO " +
                      a.algo + ", " + library(a.lib) + "; set LIB " + a.lib +
                      or_with(a.stage, other));
        }
    }

    /* LBFGS is dlib's L-BFGS in the refit (in `tadah hpo` LBFGS is a
     * CERES algorithm, and NLopt's is LD_LBFGS): with another LIB the
     * message also names that library's algorithms of the stage, whatever
     * the case of LIB. */
    CHECK(refit_error("REFINE", "OPTIMIZER LIB NLOPT ALGO LBFGS ENDOPTIMIZER") ==
          "REFINE: LIB NLOPT does not match ALGO LBFGS, a dlib algorithm; set "
          "LIB DLIB, or with LIB NLOPT set ALGO LD_LBFGS");
    CHECK(refit_error("REFINE", "OPTIMIZER LIB nlopt ALGO LBFGS ENDOPTIMIZER") ==
          "REFINE: LIB nlopt does not match ALGO LBFGS, a dlib algorithm; set "
          "LIB DLIB, or with LIB NLOPT set ALGO LD_LBFGS");
    CHECK(refit_error("REFINE", "OPTIMIZER LIB CERES ALGO LBFGS ENDOPTIMIZER") ==
          "REFINE: LIB CERES does not match ALGO LBFGS, a dlib algorithm; set "
          "LIB DLIB, or with LIB CERES set ALGO LM|DOGLEG");
    CHECK(refit_error("EXPLORE",
                      "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER "
                      "LIB NLOPT ALGO LBFGS ENDINNER ENDOPTIMIZER") ==
          "EXPLORE INNER: LIB NLOPT does not match ALGO LBFGS, a dlib "
          "algorithm; set LIB DLIB, or with LIB NLOPT set ALGO "
          "LN_SBPLX|LN_NELDERMEAD|LN_BOBYQA|LN_PRAXIS|LD_LBFGS");
    /* A stage without an algorithm of the LIB given, or a LIB that is no
     * library, gets the library of the ALGO only. */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB CERES ALGO GN_MLSL_LDS "
                                 "ENDOPTIMIZER") ==
          "EXPLORE: LIB CERES does not match ALGO GN_MLSL_LDS, an NLopt "
          "algorithm; set LIB NLOPT");
    CHECK(refit_error("REFINE", "OPTIMIZER LIB FOO ALGO LD_LBFGS "
                                "ENDOPTIMIZER") ==
          "REFINE: LIB FOO does not match ALGO LD_LBFGS, an NLopt algorithm; "
          "set LIB NLOPT");
    /* `tadah hpo`'s name of dlib's global search, GFS, is not a refit
     * ALGO (the refit's is GLOBAL): the stage's legal list names it. */
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB DLIB ALGO GFS ENDOPTIMIZER") ==
          "EXPLORE outer: unsupported optimiser LIB 'DLIB' ALGO 'GFS'. Legal: "
          "NLOPT GN_MLSL_LDS|GN_DIRECT_L|GN_CRS2_LM, DLIB GLOBAL, or UNIFORM");

    /* ALGO NONE runs no optimiser, so a LIB there would have no effect. */
    struct None { std::string key, body, message; };
    for (const auto& n : std::vector<None>{
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 "
                         "INNER LIB NLOPT ALGO NONE ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: LIB has no effect with ALGO NONE (no local "
              "search runs); remove LIB"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO NONE ENDOPTIMIZER",
              "EXPLORE: LIB has no effect with ALGO NONE (no global search "
              "runs); remove LIB"},
             {"REFINE", "OPTIMIZER LIB CERES ALGO NONE ENDOPTIMIZER",
              "REFINE: LIB has no effect with ALGO NONE (the stage is "
              "skipped); remove LIB"},
             {"FUSE", "OPTIMIZER LIB CERES ALGO NONE ENDOPTIMIZER",
              "FUSE: LIB has no effect with ALGO NONE (the stage is "
              "skipped); remove LIB"}}) {
        INFO(n.key << ": " << n.body);
        CHECK(refit_error(n.key, n.body) == n.message);
    }
    CHECK(refit_error("REFINE", "OPTIMIZER ALGO NONE ENDOPTIMIZER") == "");
    CHECK(refit_error("FUSE", "OPTIMIZER ALGO NONE ENDOPTIMIZER") == "");
    CHECK(refit_explore_outer_algo(
              RefitParser::parse(refit_context(
                  "EXPLORE", "OPTIMIZER ALGO NONE ENDOPTIMIZER"))
                  .explore.opt) == "UNIFORM");

    /* UNIFORM, the refit's own uniform start, runs no optimiser either:
     * a LIB there is rejected as it is with NONE (it used to be neither
     * required nor checked, even LIB FOO). */
    CHECK(refit_explore_outer_algo(
              RefitParser::parse(refit_context(
                  "EXPLORE", "OPTIMIZER ALGO UNIFORM ENDOPTIMIZER"))
                  .explore.opt) == "UNIFORM");
    for (const std::string lib : {"NLOPT", "DLIB", "nlopt", "FOO"}) {
        INFO("LIB " << lib << " ALGO UNIFORM");
        CHECK(refit_error("EXPLORE", "OPTIMIZER LIB " + lib +
                                         " ALGO UNIFORM ENDOPTIMIZER") ==
              "EXPLORE: LIB has no effect with ALGO UNIFORM (no global search "
              "runs); remove LIB");
    }

    /* An ALGO the stage does not run is still named with the stage's
     * legal list, whatever the LIB. */
    CHECK_THAT(refit_error("FUSE", "OPTIMIZER LIB CERES ALGO LBFGS "
                                   "ENDOPTIMIZER"),
               Catch::StartsWith("FUSE: unsupported optimiser LIB 'CERES' "
                                 "ALGO 'LBFGS'"));
    CHECK_THAT(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO LN_SBPLX "
                                      "ENDOPTIMIZER"),
               Catch::StartsWith("EXPLORE outer: unsupported optimiser LIB "
                                 "'NLOPT' ALGO 'LN_SBPLX'"));
}

TEST_CASE("tadah refit: ALGO GN_MLSL, GN_DIRECT and GN_CRS2 are refused, "
          "naming the algorithm the refit runs", "[HPO][refit]")
{
    /* Pins: an EXPLORE OPTIMIZER block with ALGO GN_MLSL, GN_DIRECT or
     * GN_CRS2 is an error that says the refit does not run it, what it is,
     * and the algorithm the refit runs for it, whatever the LIB (NLOPT,
     * another, none) and the case of the ALGO (named as written). The
     * refit's MLSL, DIRECT and CRS2 are GN_MLSL_LDS, GN_DIRECT_L and
     * GN_CRS2_LM, still accepted with their short names; GN_MLSL (pseudo-
     * random starts) and GN_DIRECT (no local bias) are other NLopt
     * algorithms, which tadah hpo runs as named, and GN_CRS2 is no NLopt
     * algorithm at all.
     * Before the fix: each spelling was an alias of the refit's algorithm
     * (refit_error returns "" with LIB NLOPT, and the LIB message without
     * or with another LIB), so GN_MLSL ran GN_MLSL_LDS, GN_DIRECT
     * GN_DIRECT_L and GN_CRS2 GN_CRS2_LM. */
    struct Spelling { std::string algo, what, runs, short_name; };
    const std::vector<Spelling> refused = {
        {"GN_MLSL", "NLopt's MLSL with pseudo-random starts", "GN_MLSL_LDS",
         "MLSL"},
        {"GN_DIRECT", "NLopt's DIRECT without the local bias of DIRECT_L",
         "GN_DIRECT_L", "DIRECT"},
        {"GN_CRS2", "not an NLopt algorithm", "GN_CRS2_LM", "CRS2"}};
    for (const auto& r : refused) {
        std::string lower = r.algo;
        for (char& ch : lower) ch = static_cast<char>(std::tolower(ch));
        for (const std::string& algo : {r.algo, lower}) {
            for (const std::string lib : {"LIB NLOPT ", "", "LIB DLIB "}) {
                INFO(lib << "ALGO " << algo);
                CHECK(refit_error("EXPLORE", "OPTIMIZER " + lib + "ALGO " +
                                                 algo +
                                                 " MAXEVAL 100 ENDOPTIMIZER") ==
                      "EXPLORE: ALGO " + algo + " is not run by tadah refit (" +
                          r.what + "); its " + r.short_name + " is " + r.runs +
                          ": set ALGO " + r.runs);
            }
        }
        /* The algorithm the refit runs, and its short name, stay. */
        for (const std::string& ok : {r.runs, r.short_name}) {
            INFO("ALGO " << ok);
            CHECK(tadah::hpo::refit::refit_explore_outer_algo(
                      tadah::hpo::refit::RefitParser::parse(
                          refit_context("EXPLORE", "OPTIMIZER LIB NLOPT ALGO " +
                                                       ok +
                                                       " MAXEVAL 100 "
                                                       "ENDOPTIMIZER"))
                          .explore.opt) == r.short_name);
        }
    }
}

TEST_CASE("Strict numbers: the PARAM INIT_MODE JITTER numbers are checked at "
          "their line", "[HPO][hpo_targets][strict_numbers][hex]")
{
    /* PARAM INIT_MODE JITTER <fraction> | JITTER AUTO <tolerance>: a token
     * shaped like a number is the value, read with the number grammar at
     * its line. A hexadecimal fraction was taken as a number (JITTER
     * 0x1p-3 ran with 0.125), inf or a malformed tolerance failed only
     * when the annealer started, and `JITTER 5x` was reported as a stray
     * token. */
    const auto h = strategy_home("PARAM", "PLAIN", "TADAH", "ANNEAL_PT");
    struct Case { std::string value, message; };
    for (const auto& c : std::vector<Case>{
             {"INIT_MODE JITTER 0x1p-3",
              "PARAM INIT_MODE JITTER <fraction> must be a finite number, got "
              "'0x1p-3'"},
             {"INIT_MODE JITTER 5x",
              "PARAM INIT_MODE JITTER <fraction> must be a finite number, got "
              "'5x'"},
             {"INIT_MODE JITTER inf",
              "PARAM INIT_MODE JITTER <fraction> must be a finite number, got "
              "'inf'"},
             {"INIT_MODE JITTER AUTO 0x1p-4",
              "PARAM INIT_MODE JITTER AUTO <tolerance> must be a finite "
              "number, got '0x1p-4'"},
             {"INIT_MODE JITTER AUTO abc",
              "PARAM INIT_MODE JITTER AUTO <tolerance> must be a finite "
              "number, got 'abc'"}}) {
        INFO(c.value);
        CHECK_THAT(parse_error(h.config(c.value)),
                   Catch::StartsWith(h.at() + c.message) &&
                   Catch::EndsWith("Line: PARAM " + c.value));
    }
    for (const std::string v : {"INIT_MODE JITTER 0.125", "INIT_MODE JITTER",
                                "INIT_MODE JITTER AUTO 5e-2",
                                "INIT_MODE UNIFORM"}) {
        INFO(v);
        CHECK(parse_error(h.config(v)) == "");
    }
    /* A word after JITTER is not its value: the next key, or a stray
     * token on the line. */
    CHECK_THAT(parse_error(h.config("INIT_MODE JITTER rubbish")),
               Catch::StartsWith(h.at() + "PARAM takes no further value "
                                 "here (unexpected 'rubbish')"));
}

namespace {

/* A `tadah refit` Context with several stage blocks, and REFIT_OBJECTIVE
 * when `objective` is not empty. */
tadah::core::Context refit_stages(
    const std::vector<std::pair<std::string, std::string>>& blocks,
    const std::string& objective = "")
{
    tadah::core::Context c;
    for (const auto& [key, body] : blocks) {
        std::istringstream iss(body);
        for (std::string t; iss >> t;) c.add(key, t);
    }
    if (!objective.empty()) c.add("REFIT_OBJECTIVE", objective);
    return c;
}

std::string refit_stages_error(
    const std::vector<std::pair<std::string, std::string>>& blocks,
    const std::string& objective = "")
{
    try {
        (void)tadah::hpo::refit::RefitParser::parse(
            refit_stages(blocks, objective));
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("tadah refit: a key that has no effect with ALGO NONE or UNIFORM "
          "is rejected", "[HPO][refit]")
{
    /* A block with ALGO NONE or UNIFORM runs no optimiser, so its stop
     * criteria stop nothing: they were accepted and ignored. They are
     * rejected, as LIB is, naming what NONE means in the block. The
     * spelling the block used (MAXITER, the alias of MAXEVAL) is named. */
    struct Case { std::string key, body, message; };
    for (const auto& c : std::vector<Case>{
             {"EXPLORE", "OPTIMIZER ALGO UNIFORM MAXEVAL 100 ENDOPTIMIZER",
              "EXPLORE: MAXEVAL has no effect with ALGO UNIFORM (no global "
              "search runs); remove MAXEVAL"},
             {"EXPLORE", "OPTIMIZER ALGO NONE MAXEVAL 100 ENDOPTIMIZER",
              "EXPLORE: MAXEVAL has no effect with ALGO NONE (no global "
              "search runs); remove MAXEVAL"},
             {"EXPLORE", "OPTIMIZER MAXEVAL 100 ALGO uniform ENDOPTIMIZER",
              "EXPLORE: MAXEVAL has no effect with ALGO UNIFORM (no global "
              "search runs); remove MAXEVAL"},
             {"REFINE", "OPTIMIZER ALGO NONE MAXEVAL 100 ENDOPTIMIZER",
              "REFINE: MAXEVAL has no effect with ALGO NONE (the stage is "
              "skipped); remove MAXEVAL"},
             {"REFINE", "OPTIMIZER ALGO NONE MAXITER 100 ENDOPTIMIZER",
              "REFINE: MAXITER has no effect with ALGO NONE (the stage is "
              "skipped); remove MAXITER"},
             {"REFINE", "OPTIMIZER ALGO NONE FTOL_REL 1e-12 ENDOPTIMIZER",
              "REFINE: FTOL_REL has no effect with ALGO NONE (the stage is "
              "skipped); remove FTOL_REL"},
             {"REFINE", "OPTIMIZER ALGO NONE GTOL 1e-12 ENDOPTIMIZER",
              "REFINE: GTOL has no effect with ALGO NONE (the stage is "
              "skipped); remove GTOL"},
             {"FUSE", "OPTIMIZER ALGO NONE MAXITER 100 ENDOPTIMIZER",
              "FUSE: MAXITER has no effect with ALGO NONE (the stage is "
              "skipped); remove MAXITER"},
             {"FUSE", "OPTIMIZER ALGO NONE FTOL_REL 1e-12 ENDOPTIMIZER",
              "FUSE: FTOL_REL has no effect with ALGO NONE (the stage is "
              "skipped); remove FTOL_REL"},
             {"FUSE", "OPTIMIZER ALGO NONE GTOL 1e-12 ENDOPTIMIZER",
              "FUSE: GTOL has no effect with ALGO NONE (the stage is "
              "skipped); remove GTOL"}}) {
        INFO(c.key << ": " << c.body);
        CHECK(refit_error(c.key, c.body) == c.message);
    }

    /* EXPLORE's INNER block configures the per-rung local search whatever
     * the EXPLORE optimiser, so UNIFORM takes one. */
    CHECK(refit_error("EXPLORE", "OPTIMIZER ALGO UNIFORM INNER LIB NLOPT ALGO "
                                 "LN_SBPLX MAXEVAL 50 FTOL_REL 1e-7 ENDINNER "
                                 "ENDOPTIMIZER") == "");

    /* INNER ALGO NONE runs no local search per rung. Its MAXEVAL,
     * FTOL_REL and XTOL_REL also set the local searches of the EXPLORE
     * optimiser GN_MLSL_LDS (LN_SBPLX), so only there they have an
     * effect. */
    for (const std::string outer :
         {"LIB NLOPT ALGO GN_DIRECT_L MAXEVAL 100",
          "LIB NLOPT ALGO GN_CRS2_LM MAXEVAL 100",
          "LIB DLIB ALGO GLOBAL MAXEVAL 100", "ALGO UNIFORM"}) {
        for (const std::string kv :
             {"MAXEVAL 50", "FTOL_REL 1e-7", "XTOL_REL 1e-6"}) {
            const std::string key = kv.substr(0, kv.find(' '));
            INFO(outer << " / INNER ALGO NONE " << kv);
            CHECK(refit_error("EXPLORE", "OPTIMIZER " + outer +
                                             " INNER ALGO NONE " + kv +
                                             " ENDINNER ENDOPTIMIZER") ==
                  "EXPLORE INNER: " + key + " has no effect with ALGO NONE (no "
                  "local search runs); remove " + key);
        }
    }
    CHECK(refit_error("EXPLORE", "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS "
                                 "MAXEVAL 100 INNER ALGO NONE MAXEVAL 50 "
                                 "FTOL_REL 1e-7 XTOL_REL 1e-6 ENDINNER "
                                 "ENDOPTIMIZER") == "");

    /* The FUSE curve refine of REFIT_OBJECTIVE curve runs with FUSE's own
     * MAXEVAL/MAXITER, FTOL_REL and GTOL, so REFINE NONE's stop criteria
     * have no effect in either mode, whether FUSE runs or not. (It ran
     * with REFINE's, so they were accepted under REFIT_OBJECTIVE curve
     * while FUSE ran: the first two cases returned "".) */
    const std::pair<std::string, std::string> refine_none{
        "REFINE", "OPTIMIZER ALGO NONE MAXITER 100 FTOL_REL 1e-12 GTOL 1e-12 "
                  "ENDOPTIMIZER"};
    const std::string skipped =
        "REFINE: MAXITER has no effect with ALGO NONE (the stage is skipped); "
        "remove MAXITER";
    CHECK(refit_stages_error({refine_none}, "curve") == skipped);
    CHECK(refit_stages_error({refine_none, {"FUSE", "OPTIMIZER LIB CERES ALGO "
                                                    "LM ENDOPTIMIZER"}},
                             "CURVE") == skipped);
    CHECK(refit_stages_error({refine_none}) == skipped);
    CHECK(refit_stages_error({refine_none}, "energy") == skipped);
    CHECK(refit_stages_error({refine_none, {"FUSE", "OPTIMIZER ALGO NONE "
                                                    "ENDOPTIMIZER"}},
                             "curve") == skipped);
    CHECK(refit_stages_error({refine_none, {"FUSE", "ENABLED false"}},
                             "curve") == skipped);
}

TEST_CASE("tadah refit: a key the block's ALGO does not read is rejected",
          "[HPO][refit]")
{
    /* Pins: a stage block's stop criterion that its ALGO does not read is
     * an error naming the key as written, the ALGO and the keys it reads:
     * GTOL (Ceres' gradient tolerance) with REFINE LD_LBFGS or LBFGS,
     * XTOL_REL with an EXPLORE INNER dlib LBFGS, FTOL_ABS with an NLopt
     * search; MAXITER (Ceres iterations) with an L-BFGS REFINE names
     * MAXEVAL; FTOL_REL with dlib's LBFGS names FTOL_ABS, the absolute
     * change of the objective dlib tests, which that ALGO reads; FTOL_ABS
     * with a Ceres ALGO keeps its pointer to FTOL_REL. The one exemption:
     * under the EXPLORE optimiser GN_MLSL_LDS the INNER block's MAXEVAL,
     * FTOL_REL and XTOL_REL also set MLSL's local LN_SBPLX searches, so
     * they are accepted whatever the INNER ALGO, and a key neither reads
     * (FTOL_ABS with INNER ALGO NONE) is an error naming the MLSL searches
     * as what reads the others.
     * Before the fix of the MLSL message: FTOL_ABS with INNER ALGO NONE
     * under GN_MLSL_LDS said "has no effect with ALGO NONE (it reads
     * MAXEVAL, FTOL_REL, XTOL_REL)", MLSL's keys given as NONE's.
     * Before the fix: the keys were checked per stage, not per ALGO, so
     * GTOL with LD_LBFGS/LBFGS, XTOL_REL with an INNER LBFGS, MAXITER with
     * LD_LBFGS/LBFGS (read as MAXEVAL) and FTOL_REL with LBFGS (dlib's
     * absolute stop) were accepted (""), FTOL_ABS was refused everywhere
     * ("... is not supported — Ceres' ..." in REFINE, "... is not used by
     * a tadah refit stage ..." in an INNER block), with an NLopt REFINE
     * too. */
    const std::string crs2 =
        "OPTIMIZER LIB NLOPT ALGO GN_CRS2_LM MAXEVAL 100 INNER ";
    const std::string mlsl =
        "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 100 INNER ";
    struct Case { std::string key, body, message; };
    for (const auto& c : std::vector<Case>{
             {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS GTOL 1e-3 "
                        "ENDOPTIMIZER",
              "REFINE: GTOL has no effect with ALGO LD_LBFGS (it reads "
              "MAXEVAL, FTOL_REL); remove GTOL"},
             {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS GTOL 1e-3 "
                        "ENDOPTIMIZER",
              "REFINE: GTOL has no effect with ALGO LBFGS (it reads MAXEVAL, "
              "FTOL_ABS); remove GTOL"},
             {"EXPLORE", crs2 + "LIB DLIB ALGO LBFGS MAXEVAL 40 XTOL_REL "
                                "1e-3 ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: XTOL_REL has no effect with ALGO LBFGS (it "
              "reads MAXEVAL, FTOL_ABS); remove XTOL_REL"},
             {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS MAXITER 20 "
                        "ENDOPTIMIZER",
              "REFINE: MAXITER counts Ceres iterations; ALGO LD_LBFGS "
              "counts objective evaluations: write MAXEVAL"},
             {"REFINE", "OPTIMIZER LIB DLIB ALGO lbfgs MAXITER 20 "
                        "ENDOPTIMIZER",
              "REFINE: MAXITER counts Ceres iterations; ALGO LBFGS counts "
              "objective evaluations: write MAXEVAL"},
             {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS FTOL_REL 1e-9 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_REL has no effect with ALGO LBFGS: dlib's L-BFGS "
              "stops on an absolute change of the objective; write "
              "FTOL_ABS"},
             {"EXPLORE", crs2 + "LIB DLIB ALGO LBFGS FTOL_REL 1e-9 ENDINNER "
                                "ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_REL has no effect with ALGO LBFGS: dlib's "
              "L-BFGS stops on an absolute change of the objective; write "
              "FTOL_ABS"},
             {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS FTOL_ABS 1e-9 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_ABS has no effect with ALGO LD_LBFGS (it reads "
              "MAXEVAL, FTOL_REL); remove FTOL_ABS"},
             {"EXPLORE", crs2 + "LIB NLOPT ALGO LN_SBPLX FTOL_ABS 1e-9 "
                                "ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS has no effect with ALGO LN_SBPLX (it "
              "reads MAXEVAL, FTOL_REL, XTOL_REL); remove FTOL_ABS"},
             {"EXPLORE", mlsl + "LIB DLIB ALGO LBFGS GTOL 1e-9 ENDINNER "
                                "ENDOPTIMIZER",
              "EXPLORE INNER: GTOL is not used by a tadah refit stage (it "
              "reads LIB, ALGO, MAXEVAL, FTOL_REL, FTOL_ABS, XTOL_REL)"},
             /* INNER ALGO NONE runs no search; under GN_MLSL_LDS (or its
              * short name MLSL) the block sets MLSL's local LN_SBPLX
              * searches, which the message names as theirs, not NONE's. */
             {"EXPLORE", crs2 + "ALGO NONE FTOL_ABS 1e-9 ENDINNER "
                                "ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS has no effect with ALGO NONE (no "
              "local search runs); remove FTOL_ABS"},
             {"EXPLORE", mlsl + "ALGO NONE FTOL_ABS 1e-9 ENDINNER "
                                "ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS has no effect with ALGO NONE (no "
              "local search runs; under ALGO GN_MLSL_LDS the block sets "
              "MLSL's local LN_SBPLX searches, which read MAXEVAL, "
              "FTOL_REL, XTOL_REL); remove FTOL_ABS"},
             {"EXPLORE", "OPTIMIZER LIB NLOPT ALGO mlsl MAXEVAL 100 INNER "
                         "ALGO none FTOL_ABS 1e-9 ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS has no effect with ALGO NONE (no "
              "local search runs; under ALGO MLSL the block sets MLSL's "
              "local LN_SBPLX searches, which read MAXEVAL, FTOL_REL, "
              "XTOL_REL); remove FTOL_ABS"},
             /* An NLopt INNER search reads what LN_SBPLX reads: the message
              * gives its own keys. */
             {"EXPLORE", mlsl + "LIB NLOPT ALGO LN_SBPLX FTOL_ABS 1e-9 "
                                "ENDINNER ENDOPTIMIZER",
              "EXPLORE INNER: FTOL_ABS has no effect with ALGO LN_SBPLX (it "
              "reads MAXEVAL, FTOL_REL, XTOL_REL); remove FTOL_ABS"},
             /* A Ceres ALGO keeps the pointer to FTOL_REL. */
             {"REFINE", "OPTIMIZER LIB CERES ALGO LM FTOL_ABS 1e-12 "
                        "ENDOPTIMIZER",
              "REFINE: FTOL_ABS is not supported — Ceres' function_tolerance "
              "is RELATIVE and has no absolute form. Use FTOL_REL."}}) {
        INFO(c.key << ": " << c.body);
        CHECK(refit_error(c.key, c.body) == c.message);
    }
    /* What the ALGO reads is accepted; under GN_MLSL_LDS the INNER block
     * also takes the keys of MLSL's local LN_SBPLX, whatever its ALGO. */
    for (const auto& [key, body] : std::vector<std::pair<std::string,
                                                         std::string>>{
             {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS MAXEVAL 20 FTOL_ABS "
                        "1e-9 ENDOPTIMIZER"},
             {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS MAXEVAL 20 "
                        "FTOL_REL 1e-9 ENDOPTIMIZER"},
             {"EXPLORE", crs2 + "LIB DLIB ALGO LBFGS MAXEVAL 40 FTOL_ABS 1e-9 "
                                "ENDINNER ENDOPTIMIZER"},
             {"EXPLORE", mlsl + "LIB DLIB ALGO LBFGS MAXEVAL 40 XTOL_REL 1e-3 "
                                "ENDINNER ENDOPTIMIZER"},
             {"EXPLORE", mlsl + "LIB DLIB ALGO LBFGS MAXEVAL 40 FTOL_REL 1e-7 "
                                "XTOL_REL 1e-3 FTOL_ABS 1e-9 ENDINNER "
                                "ENDOPTIMIZER"}}) {
        INFO(key << ": " << body);
        CHECK(refit_error(key, body) == "");
    }

    SECTION("every stage, ALGO and stop criterion: accepted iff the ALGO "
            "reads it") {
        /* The read set of each (stage, ALGO), beyond LIB and ALGO. */
        const std::vector<std::string> ceres = {"MAXEVAL", "MAXITER",
                                                "FTOL_REL", "GTOL"};
        const std::vector<std::string> nlopt_local = {"MAXEVAL", "FTOL_REL",
                                                      "XTOL_REL"};
        const std::vector<std::string> dlib_lbfgs = {"MAXEVAL", "FTOL_ABS"};
        const std::vector<std::string> none = {};
        struct Cell {
            std::string key;        // the stage block
            std::string head, tail; // around the key under test
            std::vector<std::string> reads;
        };
        const auto lib_algo = [](const std::string& lib,
                                 const std::string& algo) {
            return (lib.empty() ? std::string() : "LIB " + lib + " ") +
                   "ALGO " + algo + " ";
        };
        std::vector<Cell> cells;
        for (const auto& [lib, algo] :
             std::vector<std::pair<std::string, std::string>>{
                 {"NLOPT", "GN_MLSL_LDS"}, {"NLOPT", "GN_DIRECT_L"},
                 {"NLOPT", "GN_CRS2_LM"}, {"DLIB", "GLOBAL"}})
            cells.push_back({"EXPLORE", "OPTIMIZER " + lib_algo(lib, algo),
                             " ENDOPTIMIZER", {"MAXEVAL"}});
        cells.push_back({"EXPLORE", "OPTIMIZER ALGO UNIFORM ", " ENDOPTIMIZER",
                         none});
        const std::vector<std::tuple<std::string, std::string,
                                     std::vector<std::string>>>
            inner = {{"NLOPT", "LN_SBPLX", nlopt_local},
                     {"NLOPT", "LN_NELDERMEAD", nlopt_local},
                     {"NLOPT", "LN_BOBYQA", nlopt_local},
                     {"NLOPT", "LN_PRAXIS", nlopt_local},
                     {"NLOPT", "LD_LBFGS", nlopt_local},
                     {"DLIB", "LBFGS", dlib_lbfgs},
                     {"", "NONE", none}};
        for (const auto& [lib, algo, reads] : inner) {
            cells.push_back({"EXPLORE", crs2 + lib_algo(lib, algo),
                             " ENDINNER ENDOPTIMIZER", reads});
            std::vector<std::string> joined = reads;
            for (const auto& k : nlopt_local)
                if (std::find(joined.begin(), joined.end(), k) == joined.end())
                    joined.push_back(k);
            cells.push_back({"EXPLORE", mlsl + lib_algo(lib, algo),
                             " ENDINNER ENDOPTIMIZER", joined});
        }
        for (const auto& [lib, algo, reads] :
             std::vector<std::tuple<std::string, std::string,
                                    std::vector<std::string>>>{
                 {"CERES", "LM", ceres},
                 {"CERES", "DOGLEG", ceres},
                 {"NLOPT", "LD_LBFGS", {"MAXEVAL", "FTOL_REL"}},
                 {"DLIB", "LBFGS", dlib_lbfgs},
                 {"", "NONE", none}})
            cells.push_back({"REFINE", "OPTIMIZER " + lib_algo(lib, algo),
                             " ENDOPTIMIZER", reads});
        for (const auto& [lib, algo, reads] :
             std::vector<std::tuple<std::string, std::string,
                                    std::vector<std::string>>>{
                 {"CERES", "LM", ceres},
                 {"CERES", "DOGLEG", ceres},
                 {"", "NONE", none}})
            cells.push_back({"FUSE", "OPTIMIZER " + lib_algo(lib, algo),
                             " ENDOPTIMIZER", reads});
        for (const auto& cell : cells) {
            for (const std::string kv :
                 {"MAXEVAL 10", "MAXITER 10", "FTOL_REL 1e-9", "FTOL_ABS 1e-9",
                  "XTOL_REL 1e-9", "GTOL 1e-9"}) {
                const std::string k = kv.substr(0, kv.find(' '));
                const bool reads = std::find(cell.reads.begin(),
                                             cell.reads.end(),
                                             k) != cell.reads.end();
                const std::string body = cell.head + kv + cell.tail;
                INFO(cell.key << ": " << body << (reads ? " (read)" : ""));
                const std::string err = refit_error(cell.key, body);
                if (reads)
                    CHECK(err == "");
                else
                    CHECK(err != "");
            }
        }
    }
}

TEST_CASE("tadah refit: ENABLED false with an OPTIMIZER block is rejected",
          "[HPO][refit]")
{
    /* Pins: REFINE or FUSE with ENABLED false and an OPTIMIZER block, in
     * either order and whatever its ALGO (NONE included), is an error
     * naming the stage and the two ways to write it; a bare ENABLED false
     * and ENABLED true with an OPTIMIZER block are accepted.
     * Before the fix: the block was accepted and ignored (the stage was
     * skipped with only a WARNING that the configured optimiser is
     * ignored), so every rejecting case returns "". */
    const std::string tail =
        " skips the stage, so its OPTIMIZER block has no effect; remove the "
        "OPTIMIZER block, or write ALGO NONE in it instead of ENABLED false";
    for (const std::string stage : {"REFINE", "FUSE"}) {
        for (const std::string body :
             {"ENABLED false OPTIMIZER LIB CERES ALGO LM ENDOPTIMIZER",
              "OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 7 ENDOPTIMIZER "
              "ENABLED false",
              "ENABLED off OPTIMIZER ALGO NONE ENDOPTIMIZER"}) {
            INFO(stage << ": " << body);
            CHECK(refit_stages_error({{stage, body}}) ==
                  stage + ": ENABLED false" + tail);
            CHECK(refit_stages_error({{stage, body}}, "curve") ==
                  stage + ": ENABLED false" + tail);
        }
        CHECK(refit_stages_error({{stage, "ENABLED false"}}) == "");
        CHECK(refit_stages_error({{stage, "ENABLED true OPTIMIZER LIB CERES "
                                          "ALGO LM ENDOPTIMIZER"}}) == "");
    }
}

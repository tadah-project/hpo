// test_startup_validation_strategy.cpp
//
// Which STRATEGY TYPE can drive which optimiser depends on the
// configuration alone, so validate_hpo_setup() rejects an incompatible
// combination at startup -- before HPO_Host is built, any data is loaded
// or any evaluation runs -- instead of the policy finding out once the run
// (under PRESTAGE, the whole EXPLORE phase) is under way:
//  - MULTISTART runs its optimiser from each INIT point, which a global
//    optimiser that consumes a single starting point cannot do, for any
//    N_STARTS;
//  - the PRESTAGE refinement does the same over the TOPK candidates, so a
//    configured TOPK > 1 needs a refinement optimiser MULTISTART accepts;
//  - TYPE PRESTAGE needs an EXPLORE block;
//  - an INNER block needs an optimiser that runs one (NLopt's MLSL and
//    AUGLAG), and names an NLopt algorithm, which is what they run.
// MultiStartPolicy and StagedPolicy apply the same rules again when they
// run (a safety net for specs built outside HPOTargets), with the same
// text, and so does run_inner() for an INNER block.
//
// The parser rejects what is wrong in the text itself:
//  - a phase without LIB or ALGO (the STRATEGY-level optimiser, the
//    PRESTAGE refinement, an EXPLORE block, an INNER block), naming the
//    block;
//  - a STRATEGY key that has no effect under the block's TYPE (N_STARTS,
//    PARALLEL and BUDGET_PER_START outside MULTISTART, TOPK outside
//    PRESTAGE), at the key's own line, whether TYPE comes before or after
//    the key; likewise an EXPLORE or REFINE block the block's last TYPE
//    line takes out of PRESTAGE;
//  - a second STRATEGY block, which could change the TYPE the first one
//    was checked against.
//
// The alternatives these messages list come from the caps registry in
// sorted order, so the text is the same on every run.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/strategy_rules.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using V = dlib::matrix<double, 0, 1>;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
using tadah::hpo::OptimizerCaps;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::StrategyType;
using tadah::hpo::validate_hpo_setup;

/* A per-process unique name: ctest may run the cases of this file in
 * parallel processes. */
std::string unique_name(const std::string& stem)
{
    static unsigned counter = 0;
    return stem + std::to_string(::getpid()) + "_" + std::to_string(counter++) +
           "_" + std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch().count());
}

struct ScopedTempFile {
    fs::path path;
    explicit ScopedTempFile(const std::string& body)
        : path(fs::temp_directory_path() / (unique_name("tadah_svs_") + ".hpo"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~ScopedTempFile() { std::error_code ec; fs::remove(path, ec); }
};

Context make_minimal_ctx()
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
  /* the smallest configuration whose model builds: validate_hpo_setup
   * builds the model of the -c configuration (a construction error fails
   * every evaluation) */
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP 2 2 H H");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "0.5 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR BF_Linear");
    return c;
}

/* Lines 1-2; OPTIMIZER is line 3, STRATEGY line 4, and the STRATEGY body
 * starts at line 5. */
const std::string kRoot = "OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n";

std::string strategy_config(const std::string& strategy_body)
{
    return kRoot + "OPTIMIZER\n  STRATEGY\n" + strategy_body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

template <class F>
std::string error_of(F&& f)
{
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* The rule the running policy applies to the parsed spec. */
std::string policy_error(const OptimizerSpec& sp)
{
    switch (sp.strategy_type) {
        case StrategyType::MULTISTART:
            return error_of([&] {
                tadah::hpo::MultiStartPolicy<V, int>::validate(sp);
            });
        case StrategyType::PRESTAGE:
            return error_of([&] {
                tadah::hpo::StagedPolicy<V, int>::validate(sp);
            });
        case StrategyType::PLAIN:
            break;
    }
    return "";
}

/* The parser must accept `config`; returns what validate_hpo_setup() and
 * the policy say about it ("" when they accept it). */
struct Verdict {
    std::string startup;
    std::string policy;
};

Verdict judge(const std::string& config)
{
    PeriodicTable::initialize();
    ScopedTempFile tf(config);
    Context ctx = make_minimal_ctx();
    std::unique_ptr<HPOTargets> trg;
    const std::string parse_error = error_of(
        [&] { trg = std::make_unique<HPOTargets>(ctx, tf.path.string()); });
    INFO("config:\n" << config);
    REQUIRE(parse_error == "");
    Verdict v;
    v.startup = error_of([&] { validate_hpo_setup(ctx, *trg, true, true); });
    v.policy  = policy_error(trg->spec);
    return v;
}

/* What the parser says about `config` ("" when it accepts it). */
std::string parse_error(const std::string& config)
{
    PeriodicTable::initialize();
    ScopedTempFile tf(config);
    Context ctx = make_minimal_ctx();
    return error_of([&] { HPOTargets trg(ctx, tf.path.string()); });
}

bool runs_from_each_start(const OptimizerCaps& c)
{
    return !c.is_global || c.accepts_multi_seed;
}

/* Every backend runs_from_each_start() accepts, as LIB:ALGO, sorted. */
std::vector<std::string> compatible_backends()
{
    std::vector<std::string> out;
    for (const auto* c : tadah::hpo::filter_optimizer_caps(&runs_from_each_start))
        out.push_back(c->lib + ":" + c->algo);
    std::sort(out.begin(), out.end());
    return out;
}

/* The alternatives a rejection lists: the "(A, B, ..., ... (N more))" group
 * after `lead`. They must be the first compatible backends in sorted
 * order. */
void check_alternatives(const std::string& msg, const std::string& lead)
{
    INFO("message: " << msg);
    const auto open = msg.find(lead + " (");
    REQUIRE(open != std::string::npos);
    const auto from = open + lead.size() + 2;
    const auto close = msg.find(')', msg.find(" more", from) == std::string::npos
                                         ? from : msg.find(" more", from));
    REQUIRE(close != std::string::npos);
    std::string list = msg.substr(from, close - from);
    const auto more = list.find(", ... (");
    if (more != std::string::npos) list.resize(more);
    std::vector<std::string> shown;
    std::istringstream is(list);
    for (std::string item; std::getline(is, item, ',');) {
        item.erase(0, item.find_first_not_of(' '));
        shown.push_back(item);
    }
    const auto all = compatible_backends();
    REQUIRE_FALSE(shown.empty());
    REQUIRE(shown.size() <= all.size());
    for (std::size_t i = 0; i < shown.size(); ++i) {
        INFO("alternative " << i);
        CHECK(shown[i] == all[i]);
    }
}

std::string multistart_body(const std::string& lib, const std::string& algo,
                            const std::string& n_starts,
                            const std::string& extra = "")
{
    return "    TYPE MULTISTART\n    LIB " + lib + "\n    ALGO " + algo +
           "\n    MAXEVAL 20\n    N_STARTS " + n_starts + "\n" + extra;
}

const std::string kExploreGfs =
    "    EXPLORE\n      LIB DLIB\n      ALGO GFS\n      MAXEVAL 12\n"
    "    ENDEXPLORE\n";

/* An active RESTART: AFTER_STAGNATION and MAX both positive. */
const std::string kRestart =
    "    RESTART\n      AFTER_STAGNATION 60\n      MAX 1\n"
    "      METHOD RANDOM 1\n      SEED 5\n    ENDRESTART\n";

std::string prestage_body(const std::string& lib, const std::string& algo,
                          const std::string& topk_line)
{
    return "    TYPE PRESTAGE\n    LIB " + lib + "\n    ALGO " + algo +
           "\n    MAXEVAL 20\n" + topk_line + kExploreGfs;
}

}  // namespace

TEST_CASE("startup validation: MULTISTART rejects a single-seed global "
          "optimiser for any N_STARTS",
          "[startup_validation][strategy][multistart]")
{
    SECTION("single-seed globals are rejected with the policy's text") {
        struct Case { std::string lib, algo, n_starts, extra; };
        for (const auto& c : std::vector<Case>{
                 {"NLOPT", "GN_DIRECT", "1", ""},
                 {"NLOPT", "G_MLSL_LDS", "2",
                  "    INNER\n      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                  "      MAXEVAL 5\n    ENDINNER\n"},
                 {"TADAH", "ANNEAL", "3", ""},
                 {"TADAH", "RANDOM", "2", ""},
                 {"ASA", "ASA", "2", ""}}) {
            INFO(c.lib << ":" << c.algo << " N_STARTS " << c.n_starts);
            const auto v =
                judge(strategy_config(multistart_body(c.lib, c.algo,
                                                      c.n_starts, c.extra)));
            CHECK_THAT(v.startup,
                       Catch::Contains("STRATEGY TYPE MULTISTART cannot run " +
                                       c.lib + ":" + c.algo + ":") &&
                       Catch::Contains("any N_STARTS") &&
                       Catch::Contains("STRATEGY TYPE PLAIN to run " + c.lib +
                                       ":" + c.algo + " once"));
            CHECK(v.policy == v.startup);
            if (!v.startup.empty())
                check_alternatives(v.startup, "multi-seed optimiser");
        }
    }
    SECTION("local and multi-seed optimisers pass") {
        for (const auto& la : std::vector<std::pair<std::string, std::string>>{
                 {"DLIB", "BFGS"}, {"NLOPT", "LN_SBPLX"}, {"DLIB", "GFS"}}) {
            INFO(la.first << ":" << la.second);
            const auto v = judge(strategy_config(
                multistart_body(la.first, la.second, "3")));
            CHECK(v.startup == "");
            CHECK(v.policy == "");
        }
    }
}

TEST_CASE("startup validation: PRESTAGE with TOPK > 1 rejects a single-seed "
          "global refinement with a message of its own",
          "[startup_validation][strategy][prestage]")
{
    SECTION("TOPK > 1 with a single-seed global refinement is rejected") {
        struct Case { std::string lib, algo, topk, refine_block; };
        for (const auto& c : std::vector<Case>{
                 {"TADAH", "ANNEAL", "3", ""},
                 {"NLOPT", "GN_CRS2_LM", "2", ""}}) {
            INFO(c.lib << ":" << c.algo << " TOPK " << c.topk);
            const auto v = judge(strategy_config(
                prestage_body(c.lib, c.algo, "    TOPK " + c.topk + "\n")));
            CHECK_THAT(v.startup,
                       Catch::Contains("STRATEGY TYPE PRESTAGE with TOPK " +
                                       c.topk + " cannot refine with " +
                                       c.lib + ":" + c.algo + ":") &&
                       Catch::Contains("TOPK 1") &&
                       !Catch::Contains("TYPE MULTISTART"));
            CHECK(v.policy == v.startup);
            if (!v.startup.empty())
                check_alternatives(v.startup, "multi-seed refinement optimiser");
        }
    }
    SECTION("a REFINE block is the same refinement optimiser") {
        const auto v = judge(strategy_config(
            "    TYPE PRESTAGE\n    TOPK 3\n" + kExploreGfs +
            "    REFINE\n      LIB TADAH\n      ALGO ANNEAL\n"
            "      MAXEVAL 20\n    ENDREFINE\n"));
        CHECK_THAT(v.startup,
                   Catch::Contains("STRATEGY TYPE PRESTAGE with TOPK 3 cannot "
                                   "refine with TADAH:ANNEAL:"));
        CHECK(v.policy == v.startup);
    }
    SECTION("TOPK 1, a local or a multi-seed refinement passes") {
        for (const auto& body : std::vector<std::string>{
                 prestage_body("TADAH", "ANNEAL", "    TOPK 1\n"),
                 prestage_body("NLOPT", "GN_CRS2_LM", ""),   // TOPK default 1
                 prestage_body("DLIB", "BFGS", "    TOPK 3\n"),
                 prestage_body("NLOPT", "LN_SBPLX", "    TOPK 3\n"),
                 prestage_body("DLIB", "GFS", "    TOPK 3\n")}) {
            INFO(body);
            const auto v = judge(strategy_config(body));
            CHECK(v.startup == "");
            CHECK(v.policy == "");
        }
    }
}

TEST_CASE("startup validation: TYPE PRESTAGE without an EXPLORE block",
          "[startup_validation][strategy][prestage]")
{
    const auto v = judge(strategy_config(
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n"));
    CHECK_THAT(v.startup, Catch::Contains("EXPLORE ... ENDEXPLORE"));
    CHECK(v.policy == v.startup);
}

TEST_CASE("startup validation: a phase without LIB or ALGO is rejected by "
          "the parser naming the block",
          "[startup_validation][strategy][driver]")
{
    SECTION("PLAIN without LIB and ALGO") {
        /* ENDOPTIMIZER is line 8. */
        const auto msg = parse_error(strategy_config(
            "    TYPE PLAIN\n    MAXEVAL 5\n"));
        CHECK_THAT(msg, Catch::StartsWith("Error on line 8: ") &&
                        Catch::Contains("the OPTIMIZER block names no "
                                        "optimiser: LIB and ALGO are not set"));
    }
    SECTION("the short form without ALGO") {
        const auto msg = parse_error(
            kRoot + "OPTIMIZER\nLIB DLIB\nMAXEVAL 5\nENDOPTIMIZER\n");
        CHECK_THAT(msg, Catch::StartsWith("Error on line 6: ") &&
                        Catch::Contains("the OPTIMIZER block names no "
                                        "optimiser: ALGO is not set"));
    }
    SECTION("MULTISTART without LIB") {
        const auto msg = parse_error(strategy_config(
            "    TYPE MULTISTART\n    ALGO BFGS\n    N_STARTS 2\n"));
        CHECK_THAT(msg, Catch::Contains("STRATEGY TYPE MULTISTART names no "
                                        "optimiser to run from each start: "
                                        "LIB is not set"));
    }
    SECTION("the PRESTAGE refinement without LIB and ALGO") {
        const auto msg = parse_error(strategy_config(
            "    TYPE PRESTAGE\n    MAXEVAL 20\n" + kExploreGfs));
        CHECK_THAT(msg, Catch::Contains("STRATEGY TYPE PRESTAGE names no "
                                        "refinement optimiser: LIB and ALGO "
                                        "are not set") &&
                        Catch::Contains("REFINE block"));
    }
    SECTION("an EXPLORE block without ALGO, at its ENDEXPLORE line") {
        /* EXPLORE opens on line 9 and closes on line 12. */
        const auto msg = parse_error(strategy_config(
            "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n"
            "    EXPLORE\n      LIB DLIB\n      MAXEVAL 12\n    ENDEXPLORE\n"));
        CHECK_THAT(msg, Catch::StartsWith("Error on line 12: ") &&
                        Catch::Contains("the EXPLORE block names no "
                                        "optimiser: ALGO is not set"));
    }
    SECTION("a configuration without an OPTIMIZER block") {
        const auto msg = parse_error(kRoot);
        CHECK_THAT(msg, Catch::Contains("names no optimiser: LIB and ALGO are "
                                        "not set"));
    }
    SECTION("an OPTIMIZER block that is never closed") {
        const auto msg = parse_error(kRoot + "OPTIMIZER\n  MAXEVAL 5\n");
        CHECK_THAT(msg, Catch::Contains("names no optimiser: LIB and ALGO are "
                                        "not set"));
    }
    SECTION("a spec built outside the parser gets the same text from the "
            "policies") {
        OptimizerSpec ms;
        ms.strategy_type = StrategyType::MULTISTART;
        const auto m = policy_error(ms);
        CHECK_THAT(m, Catch::Contains("STRATEGY TYPE MULTISTART names no "
                                      "optimiser to run from each start: LIB "
                                      "and ALGO are not set") &&
                      !Catch::Contains("unknown optimizer"));

        OptimizerSpec ps;
        ps.strategy_type = StrategyType::PRESTAGE;
        ps.lib = "DLIB";
        ps.algo = "BFGS";
        ps.prestage = std::make_unique<OptimizerSpec>();
        ps.prestage->lib = "DLIB";
        const auto p = policy_error(ps);
        CHECK_THAT(p, Catch::Contains("the EXPLORE block names no optimiser: "
                                      "ALGO is not set"));
    }
}

TEST_CASE("startup validation: an INNER block without LIB or ALGO is "
          "rejected at its ENDINNER line naming the block",
          "[startup_validation][strategy][driver][inner]")
{
    /* An INNER block without ALGO used to pass the parser and fail inside
     * NLopt once the run had started; one without LIB skipped the STEP
     * checks. The rule is the one of the other blocks: LIB and ALGO name
     * the optimiser, checked where the block closes. */
    const std::string head = kRoot + "OPTIMIZER\n  STRATEGY\n";   // lines 1-4
    const std::string tail = "  ENDSTRATEGY\nENDOPTIMIZER\n";
    const std::string mlsl =
        "    LIB NLOPT\n    ALGO G_MLSL_LDS\n    MAXEVAL 60\n";
    const auto inner = [](const std::string& body) {
        return "    INNER\n" + body + "      MAXEVAL 20\n    ENDINNER\n";
    };

    SECTION("at ENDINNER, naming the enclosing block") {
        struct Case {
            std::string name;
            std::string strategy_body;   // with the INNER block
            std::string enclosing;
            std::string missing;
        };
        for (const auto& c : std::vector<Case>{
                 {"STRATEGY, no ALGO",
                  "    TYPE PLAIN\n" + mlsl + inner("      LIB NLOPT\n"),
                  "STRATEGY", "ALGO is not set"},
                 {"STRATEGY, no LIB",
                  "    TYPE PLAIN\n" + mlsl + inner("      ALGO LN_SBPLX\n"),
                  "STRATEGY", "LIB is not set"},
                 {"STRATEGY, neither", "    TYPE PLAIN\n" + mlsl + inner(""),
                  "STRATEGY", "LIB and ALGO are not set"},
                 {"EXPLORE, no ALGO",
                  "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                  "    EXPLORE\n" + mlsl + inner("      LIB NLOPT\n") +
                      "    ENDEXPLORE\n",
                  "EXPLORE", "ALGO is not set"},
                 {"REFINE, no LIB",
                  "    TYPE PRESTAGE\n" + kExploreGfs + "    REFINE\n" + mlsl +
                      inner("      ALGO LN_BOBYQA\n") + "    ENDREFINE\n",
                  "REFINE", "LIB is not set"}}) {
            const std::string text = head + c.strategy_body + tail;
            const std::string upto = text.substr(0, text.find("    ENDINNER\n"));
            const std::size_t endinner_line =
                static_cast<std::size_t>(
                    std::count(upto.begin(), upto.end(), '\n')) + 1;
            INFO(c.name << ", config:\n" << text);
            CHECK_THAT(parse_error(text),
                       Catch::StartsWith(
                           "Error on line " + std::to_string(endinner_line) +
                           ": the INNER block inside " + c.enclosing +
                           " names no optimiser: " + c.missing +
                           "; set LIB and ALGO inside the INNER block") &&
                       Catch::EndsWith("Line:     ENDINNER"));
        }
    }
    SECTION("an INNER block that is never closed, at the end of the file") {
        const auto msg = parse_error(head + "    TYPE PLAIN\n" + mlsl +
                                     "    INNER\n      LIB NLOPT\n");
        CHECK_THAT(msg, Catch::Contains("the INNER block inside STRATEGY names "
                                        "no optimiser: ALGO is not set"));
    }
    SECTION("an INNER block that names LIB and ALGO passes") {
        CHECK(parse_error(head + "    TYPE PLAIN\n" + mlsl +
                          inner("      LIB NLOPT\n      ALGO LN_SBPLX\n") +
                          tail) == "");
    }
}

TEST_CASE("startup validation: a STRATEGY key of another TYPE is rejected at "
          "its own line",
          "[startup_validation][strategy][knobs]")
{
    struct Case {
        std::string type_line;   // "" = no TYPE line (PLAIN by default)
        std::string knob;
        std::string type_named;
        std::string home;
    };
    for (const auto& c : std::vector<Case>{
             {"    TYPE PLAIN\n", "N_STARTS 3", "TYPE PLAIN", "TYPE MULTISTART"},
             {"    TYPE PLAIN\n", "PARALLEL 2", "TYPE PLAIN", "TYPE MULTISTART"},
             {"    TYPE PLAIN\n", "BUDGET_PER_START 10", "TYPE PLAIN",
              "TYPE MULTISTART"},
             {"    TYPE MULTISTART\n", "TOPK 3", "TYPE MULTISTART",
              "TYPE PRESTAGE"},
             {"    TYPE PLAIN\n", "TOPK 2", "TYPE PLAIN", "TYPE PRESTAGE"},
             {"", "N_STARTS 2", "TYPE PLAIN (the default", "TYPE MULTISTART"}}) {
        INFO(c.type_line << c.knob);
        /* The knob sits on line 5 or 6, before LIB/ALGO. */
        const std::size_t knob_line = c.type_line.empty() ? 5 : 6;
        const std::string key = c.knob.substr(0, c.knob.find(' '));
        const auto msg = parse_error(strategy_config(
            c.type_line + "    " + c.knob + "\n    LIB DLIB\n    ALGO BFGS\n"
            "    MAXEVAL 20\n"));
        CHECK_THAT(msg,
                   Catch::StartsWith("Error on line " +
                                     std::to_string(knob_line) + ": " + key +
                                     " is not accepted with STRATEGY " +
                                     c.type_named) &&
                   Catch::Contains("set " + c.home + " or remove " + key) &&
                   Catch::EndsWith("Line:     " + c.knob));
    }

    SECTION("PARALLEL under TYPE PRESTAGE") {
        const auto msg = parse_error(strategy_config(
            "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n    PARALLEL 2\n" +
            kExploreGfs));
        CHECK_THAT(msg, Catch::StartsWith("Error on line 8: PARALLEL is not "
                                          "accepted with STRATEGY TYPE "
                                          "PRESTAGE"));
    }
    SECTION("TYPE written after the key decides") {
        PeriodicTable::initialize();
        ScopedTempFile tf(strategy_config(
            "    N_STARTS 3\n    TOPK 1\n    TYPE PRESTAGE\n    LIB DLIB\n"
            "    ALGO BFGS\n" + kExploreGfs));
        Context ctx = make_minimal_ctx();
        const auto msg = error_of([&] { HPOTargets trg(ctx, tf.path.string()); });
        CHECK_THAT(msg, Catch::StartsWith("Error on line 5: N_STARTS is not "
                                          "accepted with STRATEGY TYPE "
                                          "PRESTAGE"));

        ScopedTempFile ok(strategy_config(
            "    N_STARTS 3\n    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"));
        HPOTargets trg(ctx, ok.path.string());
        CHECK(trg.spec.multistart.n_starts == 3);
    }
    SECTION("the last TYPE line decides") {
        const auto msg = parse_error(strategy_config(
            "    TYPE MULTISTART\n    N_STARTS 3\n    TYPE PLAIN\n"
            "    LIB DLIB\n    ALGO BFGS\n"));
        CHECK_THAT(msg, Catch::StartsWith("Error on line 6: N_STARTS is not "
                                          "accepted with STRATEGY TYPE PLAIN"));
    }
}

TEST_CASE("startup validation: an EXPLORE or REFINE block that a later TYPE "
          "line takes out of PRESTAGE is rejected at its own line",
          "[startup_validation][strategy][knobs][prestage]")
{
    /* EXPLORE and REFINE need TYPE PRESTAGE before them; a TYPE line after
     * them used to switch the strategy silently: the EXPLORE block was
     * dropped (the run was PLAIN) and a REFINE block configured the PLAIN
     * optimiser. The TYPE the block ends with decides, as for the keys of
     * one TYPE. Lines 1-4 are the root keys, OPTIMIZER and STRATEGY. */
    const std::string head = "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                             "    MAXEVAL 3\n";   // lines 5-8
    SECTION("EXPLORE, then TYPE PLAIN") {
        const auto msg = parse_error(strategy_config(
            head + kExploreGfs + "    TYPE PLAIN\n"));
        CHECK_THAT(msg,
                   Catch::StartsWith("Error on line 9: EXPLORE block is only "
                                     "valid with TYPE PRESTAGE, but the "
                                     "STRATEGY block ends with TYPE PLAIN") &&
                   Catch::Contains("set TYPE PRESTAGE or remove the EXPLORE "
                                   "block") &&
                   Catch::EndsWith("Line:     EXPLORE"));
    }
    SECTION("REFINE, then TYPE MULTISTART") {
        const auto msg = parse_error(strategy_config(
            "    TYPE PRESTAGE\n"
            "    REFINE\n      LIB DLIB\n      ALGO BFGS\n    ENDREFINE\n"
            "    TYPE MULTISTART\n"));
        /* REFINE opens on line 6. */
        CHECK_THAT(msg,
                   Catch::StartsWith("Error on line 6: REFINE block is only "
                                     "valid with TYPE PRESTAGE, but the "
                                     "STRATEGY block ends with TYPE "
                                     "MULTISTART") &&
                   Catch::EndsWith("Line:     REFINE"));
    }
    SECTION("a TYPE PRESTAGE line repeated after the blocks is accepted") {
        CHECK(parse_error(strategy_config(head + kExploreGfs +
                                          "    TYPE PRESTAGE\n")) == "");
    }
}

TEST_CASE("startup validation: a second STRATEGY block is rejected at its "
          "line",
          "[startup_validation][strategy][knobs]")
{
    /* A second block reopened the first: its TYPE silently replaced the
     * one the first block's keys had been checked against (a MULTISTART
     * block with N_STARTS 3 followed by a PLAIN block ran PLAIN). */
    const std::string first = "    TYPE MULTISTART\n    N_STARTS 3\n"
                              "    LIB DLIB\n    ALGO BFGS\n";   // lines 5-8
    SECTION("in the same OPTIMIZER block") {
        CHECK(parse_error(strategy_config(first)) == "");
        const auto twice = parse_error(
            kRoot + "OPTIMIZER\n  STRATEGY\n" + first + "  ENDSTRATEGY\n"
            "  STRATEGY\n    TYPE PLAIN\n  ENDSTRATEGY\nENDOPTIMIZER\n");
        /* The second STRATEGY opens on line 10. */
        CHECK_THAT(twice,
                   Catch::StartsWith("Error on line 10: STRATEGY block given "
                                     "more than once (the first opens on line "
                                     "4)") &&
                   Catch::EndsWith("Line:   STRATEGY"));
    }
    SECTION("in a second OPTIMIZER block") {
        const auto twice = parse_error(
            kRoot + "OPTIMIZER\n  STRATEGY\n" + first + "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\nOPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n"
            "  ENDSTRATEGY\nENDOPTIMIZER\n");
        CHECK_THAT(twice, Catch::StartsWith("Error on line 12: STRATEGY block "
                                            "given more than once"));
    }
}

TEST_CASE("startup validation: an INNER block needs an optimiser that runs "
          "one and an NLopt algorithm",
          "[startup_validation][strategy][inner]")
{
    /* Only NLopt's MLSL and AUGLAG run an INNER search, and they run it
     * with NLopt (build_nlopt hands the INNER ALGO to nlopt::opt). Both
     * rules depend on the configuration alone, so they are startup errors:
     * an INNER block under another optimiser was rejected only once the
     * host had loaded the data (run_inner), and a non-NLopt INNER
     * optimiser failed inside NLopt once the run had started ("wrong
     * algorithm string"). */
    const auto inner = [](const std::string& lib, const std::string& algo) {
        return "    INNER\n      LIB " + lib + "\n      ALGO " + algo +
               "\n      MAXEVAL 5\n    ENDINNER\n";
    };
    const std::string mlsl = "    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
                             "    MAXEVAL 60\n";

    SECTION("a non-NLopt INNER optimiser under MLSL") {
        for (const auto& la : std::vector<std::pair<std::string, std::string>>{
                 {"DLIB", "BFGS"}, {"DLIB", "GFS"}, {"TADAH", "ANNEAL"}}) {
            INFO(la.first << ":" << la.second);
            const auto v = judge(strategy_config(
                "    TYPE PLAIN\n" + mlsl + inner(la.first, la.second)));
            CHECK_THAT(v.startup,
                       Catch::StartsWith("the INNER block of the STRATEGY-level "
                                         "optimiser names " + la.first + ":" +
                                         la.second + ", but NLOPT:G_MLSL_LDS "
                                         "runs its INNER search with NLopt") &&
                       Catch::Contains("set LIB NLOPT and an NLopt ALGO"));
        }
    }
    SECTION("an INNER block under an optimiser that runs none") {
        struct Case { std::string body, phase, name; };
        for (const auto& c : std::vector<Case>{
                 {"    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n" +
                      inner("NLOPT", "LN_SBPLX"),
                  "the STRATEGY-level optimiser", "DLIB:BFGS"},
                 {"    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n" +
                      inner("NLOPT", "LN_SBPLX"),
                  "the STRATEGY-level optimiser", "TADAH:ANNEAL"},
                 {"    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                  "    EXPLORE\n      LIB NLOPT\n      ALGO GN_DIRECT_L\n" +
                      inner("NLOPT", "LN_SBPLX") + "    ENDEXPLORE\n",
                  "the EXPLORE block", "NLOPT:GN_DIRECT_L"},
                 {"    TYPE PRESTAGE\n" + kExploreGfs +
                      "    REFINE\n      LIB NLOPT\n      ALGO LN_BOBYQA\n" +
                      inner("NLOPT", "LN_SBPLX") + "    ENDREFINE\n",
                  "the PRESTAGE refinement", "NLOPT:LN_BOBYQA"}}) {
            INFO(c.name << " (" << c.phase << ")");
            const auto v = judge(strategy_config(c.body));
            CHECK_THAT(v.startup,
                       Catch::StartsWith(c.name + " does not accept an INNER "
                                         "sub-optimiser (the INNER block of " +
                                         c.phase + ")") &&
                       Catch::Contains("NLOPT:AUGLAG") &&
                       Catch::Contains("remove the INNER block"));
        }
    }
    SECTION("the same rules for a spec built outside the parser") {
        OptimizerSpec sp;
        sp.lib = "DLIB";
        sp.algo = "BFGS";
        sp.inner = std::make_unique<OptimizerSpec>();
        sp.inner->lib = "NLOPT";
        sp.inner->algo = "LN_SBPLX";
        CHECK_THAT(error_of([&] { tadah::hpo::validate_run_policy(sp); }),
                   Catch::StartsWith("DLIB:BFGS does not accept an INNER "
                                     "sub-optimiser (the INNER block of the "
                                     "STRATEGY-level optimiser): only NLopt"));
        sp.lib = "NLOPT";
        sp.algo = "AUGLAG";
        sp.inner->lib = "DLIB";
        sp.inner->algo = "BFGS";
        CHECK_THAT(error_of([&] { tadah::hpo::validate_run_policy(sp); }),
                   Catch::StartsWith("the INNER block of the STRATEGY-level "
                                     "optimiser names DLIB:BFGS, but "
                                     "NLOPT:AUGLAG runs its INNER search with "
                                     "NLopt"));
    }
    SECTION("an NLopt INNER optimiser under MLSL or AUGLAG passes") {
        for (const std::string algo : {"G_MLSL_LDS", "AUGLAG"}) {
            INFO(algo);
            const auto v = judge(strategy_config(
                "    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
                "\n    MAXEVAL 60\n" + inner("NLOPT", "LN_BOBYQA")));
            CHECK(v.startup == "");
        }
    }
}

TEST_CASE("startup validation: the caps lists in messages are sorted",
          "[startup_validation][strategy][optimizer_caps]")
{
    using tadah::hpo::filter_optimizer_caps;
    using tadah::hpo::format_caps_list;
    auto all = filter_optimizer_caps([](const OptimizerCaps&) { return true; });
    REQUIRE(all.size() > 10);
    const std::string listed = format_caps_list(all, all.size());

    std::vector<std::string> names;
    for (const auto* c : all) names.push_back(c->lib + ":" + c->algo);
    std::sort(names.begin(), names.end());
    std::string sorted;
    for (const auto& n : names) sorted += (sorted.empty() ? "" : ", ") + n;
    CHECK(listed == sorted);

    /* The order the entries come in does not matter. */
    std::reverse(all.begin(), all.end());
    CHECK(format_caps_list(all, all.size()) == sorted);
    CHECK(format_caps_list(all, 3) ==
          names[0] + ", " + names[1] + ", " + names[2] + ", ... (" +
              std::to_string(names.size() - 3) + " more)");
}

/* ------------------------------------------------------------------ */
/*  End to end: hpo_run() stops before it builds the host              */
/* ------------------------------------------------------------------ */

namespace {

using Host = tadah::hpo::HPO_Host<V, ::LammpsRunner>;

/* hpo_run() writes every output file into the current directory. */
struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir()
        : dir(fs::temp_directory_path() / unique_name("tadah_svs_run_")),
          original_cwd(fs::current_path())
    {
        fs::create_directories(dir);
        fs::current_path(dir);
    }
    ~ScratchDir()
    {
        auto& log = tadah::hpo::HPOLog::instance();
        if (log.is_open()) log.finish("test end");
        log.reset_for_test();
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

/* The four-atom BCC Ta structure of test_hpo_host_e2e.cpp. */
void write_ta_db(const fs::path& path, double offset_x, double energy)
{
    std::ofstream os(path);
    os << "test_bcc_ta\n1.0 1.0 1.0\n" << energy << " 0.0\n"
       << "3.301 0.0 0.0\n0.0 3.301 0.0\n0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n\n";
}

/* hpo_run() on the no-LAMMPS D2_BP Ta setup with the given OPTIMIZER
 * block; returns what it throws. */
std::string run_hpo(const std::string& optimizer)
{
    PeriodicTable::initialize();
    tadah::hpo::HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    ScratchDir sd;
    const fs::path train = sd.dir / "train.db";
    const fs::path valdb = sd.dir / "val.db";
    const fs::path val = sd.dir / "config.val";
    const fs::path hpo = sd.dir / "config.hpo";
    write_ta_db(train, 0.0, -0.001);
    write_ta_db(valdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << optimizer
                       << "LOSS L2\nFAILSCORE 1e9\nPC_ERMSE 0 1\n"
                          "OPTIM CGRID2B (1) 0.1 0.9\n"
                          "OPTIM CGRID2B (2) 1.0 2.0\n"
                          "OPTIM SGRID2B (2) 4.0 6.0\n";
    Context c;
    c.add("DBFILE", train.string());
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", 2);
    c.add("TYPE2B", 2);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "0.5 1.4");
    c.add("SGRID2B", "3.0 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", 5.0);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    c.add("EWEIGHT", 1.0);
    c.add("FWEIGHT", 1.0);
    c.add("SWEIGHT", 1.0);
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    c.add("HPOTARGET", hpo.string());
    c.add("VALIDATION", val.string());

    const std::string err = error_of([&] { tadah::hpo::hpo_run(c); });

    /* Nothing was evaluated, and the journal stops before the setup
     * sections that follow startup validation (FILES, POTENTIAL SETUP). */
    CHECK(Host::g_iter_value() == 0);
    std::ifstream in("hpo_run.log");
    std::stringstream journal;
    journal << in.rdbuf();
    CHECK_THAT(journal.str(), !Catch::Contains("-- FILES ") &&
                              !Catch::Contains("-- POTENTIAL SETUP "));
    CHECK_FALSE(fs::exists("loss.tadah"));
    /* The rejection comes before the worker pool is sized or a PC_LAMMPS
     * worker process is spawned (their journal lines never appear), and
     * the journal records it. */
    CHECK_THAT(journal.str(), !Catch::Contains("ScriptExecutor:") &&
                              !Catch::Contains("PC_ISOLATION"));
    CHECK_FALSE(tadah::hpo::pc_worker_pool_enabled());
    INFO("journal:\n" << journal.str());
    REQUIRE_FALSE(err.empty());
    CHECK_THAT(journal.str(), Catch::Contains("ERROR     startup: " + err));
    return err;
}

}  // namespace

TEST_CASE("startup validation: hpo_run rejects a strategy before it loads "
          "any data",
          "[startup_validation][strategy][e2e]")
{
    struct ForceRebuild {
        bool restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        ForceRebuild() { if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1); }
        ~ForceRebuild() { if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD"); }
    } force_rebuild;

    SECTION("MULTISTART with TADAH:ANNEAL") {
        const auto err = run_hpo(
            "OPTIMIZER\n  INIT\n    METHOD RANDOM 2\n    SEED 3\n  ENDINIT\n"
            "  STRATEGY\n    TYPE MULTISTART\n    LIB TADAH\n    ALGO ANNEAL\n"
            "    MAXEVAL 10\n    N_STARTS 2\n    SEED 5\n  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("STRATEGY TYPE MULTISTART cannot run "
                                        "TADAH:ANNEAL:"));
    }
    SECTION("PRESTAGE with TOPK 3 and a TADAH:ANNEAL refinement, before the "
            "exploration") {
        const auto err = run_hpo(
            "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n    LIB TADAH\n"
            "    ALGO ANNEAL\n    MAXEVAL 10\n    SEED 5\n    TOPK 3\n"
            "    EXPLORE\n      LIB DLIB\n      ALGO GFS\n      MAXEVAL 12\n"
            "    ENDEXPLORE\n  ENDSTRATEGY\nENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("STRATEGY TYPE PRESTAGE with TOPK 3 "
                                        "cannot refine with TADAH:ANNEAL:"));
    }
    SECTION("MULTISTART with NLOPT:GN_DIRECT under THREADS 2 and "
            "PC_ISOLATION PROCESS, before the pool is built") {
        const auto err = run_hpo(
            "PC_ISOLATION PROCESS\n"
            "OPTIMIZER\n  STRATEGY\n    TYPE MULTISTART\n    LIB NLOPT\n"
            "    ALGO GN_DIRECT\n    MAXEVAL 10\n    N_STARTS 2\n"
            "    THREADS 2\n  ENDSTRATEGY\nENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("STRATEGY TYPE MULTISTART cannot run "
                                        "NLOPT:GN_DIRECT:"));
    }
}

/* ======================================================================
 * Each phase's backend checks.
 *
 * A backend's own attribute and PARAM checks (DLIB:BFGS does not support
 * MAXTIME, TADAH:ANNEAL's PARAM registry, ASA's options, GRID's
 * GRID_POINTS, ...) ran only when that optimiser started: after the data
 * were loaded, and for a PRESTAGE refinement after the whole exploration;
 * under MULTISTART each start failed on its own, with a warning, and the
 * run ended without an evaluation. validate_hpo_setup() now runs them on
 * the spec each phase runs with -- the STRATEGY-level optimiser, each
 * MULTISTART start (with its BUDGET_PER_START or its share of MAXEVAL),
 * the PRESTAGE exploration and refinement, and every INNER block -- and a
 * backend that cannot stop without MAXEVAL is named when it has none.
 * ==================================================================== */

TEST_CASE("startup validation: each phase's backend checks run at startup",
          "[startup_validation][strategy][backend]")
{
    struct Case { std::string what, body, message; };
    for (const auto& c : std::vector<Case>{
             {"PLAIN DLIB:BFGS with MAXTIME",
              "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n"
              "    MAXTIME 60\n",
              "the STRATEGY-level optimiser: DLIB:BFGS does not support "
              "MAXTIME"},
             {"each MULTISTART start: DLIB:BFGS with SEED",
              multistart_body("DLIB", "BFGS", "2", "    SEED 3\n"),
              "each MULTISTART start: DLIB:BFGS does not support SEED"},
             {"the PRESTAGE refinement (REFINE block): DLIB:BFGS with MAXTIME",
              "    TYPE PRESTAGE\n    EXPLORE\n      LIB TADAH\n"
              "      ALGO RANDOM\n      MAXEVAL 3\n    ENDEXPLORE\n"
              "    REFINE\n      LIB DLIB\n      ALGO BFGS\n"
              "      MAXTIME 60\n    ENDREFINE\n",
              "the PRESTAGE refinement: DLIB:BFGS does not support MAXTIME"},
             {"the PRESTAGE exploration: CERES:LBFGS with STOPVAL",
              "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
              "    MAXEVAL 20\n    EXPLORE\n      LIB CERES\n"
              "      ALGO LBFGS\n      MAXEVAL 12\n      STOPVAL 0.1\n"
              "    ENDEXPLORE\n",
              "the PRESTAGE exploration (EXPLORE block): CERES:LBFGS does not "
              "support STOPVAL"},
             {"TADAH:ANNEAL with an unknown PARAM",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n"
              "    MAXEVAL 20\n    PARAM NOT_A_KEY 1\n",
              "the STRATEGY-level optimiser: TADAH:ANNEAL: unknown PARAM "
              "'NOT_A_KEY'"},
             {"TADAH:ANNEAL_PT with LADDER_ADAPT on two replicas",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PT\n"
              "    MAXEVAL 20\n    PARAM REPLICAS 2\n"
              "    PARAM LADDER_ADAPT ON\n",
              "TADAH:ANNEAL_PT: LADDER_ADAPT requires REPLICAS >= 3"},
             {"ASA:ASA with FTOL_REL",
              "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n    MAXEVAL 20\n"
              "    FTOL_REL 1e-6\n",
              "ASA:ASA does not implement FTOL_REL/FTOL_ABS"},
             {"TADAH:GRID with an unknown PARAM",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
              "    PARAM GRID_STEPS 5\n",
              "TADAH:GRID: unknown PARAM 'GRID_STEPS'"},
             {"TADAH:GRID with a one-point grid",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
              "    PARAM GRID_POINTS 1\n",
              "TADAH:GRID PARAM GRID_POINTS must be an integer >= 2"},
             /* A backend with no convergence test and no finite schedule
              * needs a stop (the cases of its own test case below). */
             {"TADAH:RANDOM without a stop",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n",
              "the STRATEGY-level optimiser: TADAH:RANDOM needs a stop"},
             {"DLIB:GFS without a stop",
              "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n",
              "the STRATEGY-level optimiser: DLIB:GFS needs a stop"},
             {"DLIB:GFS under MULTISTART without a stop (one run from every "
              "INIT point)",
              "    TYPE MULTISTART\n    LIB DLIB\n    ALGO GFS\n"
              "    N_STARTS 2\n",
              "the MULTISTART run: DLIB:GFS needs a stop"},
             {"the PRESTAGE exploration: TADAH:RANDOM without a stop",
              "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
              "    MAXEVAL 20\n    EXPLORE\n      LIB TADAH\n"
              "      ALGO RANDOM\n    ENDEXPLORE\n",
              "the PRESTAGE exploration (EXPLORE block): TADAH:RANDOM needs "
              "a stop"},
             {"TADAH:ANNEAL without MAXEVAL (AUTO chain lengths)",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n"
              "    MAXTIME 60\n",
              "TADAH:ANNEAL: CHAIN_LENGTH AUTO needs MAXEVAL"}}) {
        INFO(c.what);
        const auto v = judge(strategy_config(c.body));
        CHECK_THAT(v.startup, Catch::Contains(c.message));
    }
    /* The message says why, and which stops the backend honours. */
    CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB TADAH\n"
                                "    ALGO RANDOM\n")).startup ==
          "the STRATEGY-level optimiser: TADAH:RANDOM needs a stop: uniform "
          "random search has no convergence test and runs until it is "
          "interrupted; set MAXEVAL, MAXTIME or STOPVAL (it honours no "
          "tolerance)");
}

TEST_CASE("startup validation: the backend checks reject no valid "
          "configuration", "[startup_validation][strategy][backend]")
{
    /* Leaving MAXEVAL out is no evaluation limit for a backend that stops
     * on its own: a convergence test, a finite schedule, MAXTIME or
     * STOPVAL, its own limits (ASA). */
    for (const std::string& body : std::vector<std::string>{
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
             "    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n",
             "    TYPE PLAIN\n    LIB CERES\n    ALGO LBFGS\n",
             "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
             "    PARAM GRID_POINTS 4\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n"
             "    PARAM CHAIN_LENGTH 5\n    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PT\n"
             "    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PA\n",
             /* MULTISTART: BUDGET_PER_START is each start's budget. */
             "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
             "    N_STARTS 2\n    BUDGET_PER_START 5\n",
             multistart_body("DLIB", "GFS", "3"),
             multistart_body("DLIB", "BFGS", "3"),
             /* The PRESTAGE refinement over TOPK candidates, each with its
              * share of MAXEVAL. */
             prestage_body("DLIB", "BFGS", "    TOPK 3\n"),
             prestage_body("NLOPT", "LN_SBPLX", "    TOPK 3\n"),
             prestage_body("TADAH", "ANNEAL", "    TOPK 1\n"),
             /* INNER blocks. */
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
             "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
             "      ALGO LN_BOBYQA\n      MAXEVAL 5\n    ENDINNER\n",
             "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
             "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
             "      ALGO G_MLSL_LDS\n      MAXEVAL 12\n      INNER\n"
             "        LIB NLOPT\n        ALGO LN_SBPLX\n      ENDINNER\n"
             "    ENDEXPLORE\n",
             /* A backend without a convergence test stops on any stop it
              * honours: MAXTIME or STOPVAL alone, or a tolerance its
              * algorithm tests. */
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
             "    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
             "    STOPVAL 0.1\n",
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n    STOPVAL 0.1\n",
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n    MAXEVAL 20\n"
             "    FTOL_ABS 1e-3\n",
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PT\n"
             "    FTOL_REL 1e-6\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
             "    FTOL_REL 1e-6\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_DIRECT_L\n"
             "    XTOL_ABS 1e-4\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_ISRES\n"
             "    STOPVAL 0.1\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_ESCH\n"
             "    MAXTIME 60\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_MMA\n"
             "    FTOL_REL 1e-8\n",
             /* The algorithms that end on their own. */
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_ORIG_DIRECT_L\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_COBYLA\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_LBFGS\n",
             /* AUGLAG ends when its local search ends: with NLopt's default
              * LN_COBYLA, or an INNER search that has a stop of its own or
              * one of the enclosing block's. */
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_AUGLAG\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n"
             "    FTOL_REL 1e-8\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
             "      ALGO GN_DIRECT_L\n    ENDINNER\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    STOPVAL 0.1\n    INNER\n      LIB NLOPT\n"
             "      ALGO GN_DIRECT_L\n    ENDINNER\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    INNER\n      LIB NLOPT\n      ALGO GN_DIRECT_L\n"
             "      XTOL_REL 1e-4\n    ENDINNER\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    INNER\n      LIB NLOPT\n      ALGO LN_BOBYQA\n"
             "    ENDINNER\n",
             /* MLSL hands its stop on to the INNER search. */
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
             "    MAXTIME 60\n    INNER\n      LIB NLOPT\n"
             "      ALGO GN_DIRECT_L\n    ENDINNER\n",
             /* Each MULTISTART start with its share of MAXEVAL, or its
              * BUDGET_PER_START. */
             "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LD_MMA\n"
             "    MAXEVAL 20\n    N_STARTS 2\n",
             "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LD_MMA\n"
             "    N_STARTS 2\n    BUDGET_PER_START 5\n",
             "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
             "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
             "      ALGO GN_CRS2_LM\n      MAXTIME 60\n    ENDEXPLORE\n",
             /* An active RESTART (AFTER_STAGNATION and MAX > 0) ends every
              * phase it wraps: once its MAX restarts are used, the next
              * AFTER_STAGNATION evaluations without a new best end the
              * run. */
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_DIRECT_L\n" +
                 kRestart,
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PT\n" + kRestart,
             "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n" + kRestart,
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n" + kRestart,
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n" +
                 kRestart,
             "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LD_MMA\n"
             "    N_STARTS 2\n" + kRestart,
             "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
             "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
             "      ALGO GN_CRS2_LM\n    ENDEXPLORE\n" + kRestart,
             "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO GN_DIRECT_L\n"
             "    EXPLORE\n      LIB DLIB\n      ALGO GFS\n"
             "      MAXEVAL 12\n    ENDEXPLORE\n" + kRestart,
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    INNER\n      LIB NLOPT\n      ALGO GN_DIRECT_L\n"
             "    ENDINNER\n" + kRestart,
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n" +
                 kRestart}) {
        INFO(body);
        const auto v = judge(strategy_config(body));
        CHECK(v.startup == "");
    }
}

/* ======================================================================
 * A backend with no convergence test and no finite schedule needs a stop:
 * MAXEVAL, MAXTIME, STOPVAL or a tolerance it honours. Without one it ran
 * until it was interrupted (the probes of phase 4c: NLopt's global
 * searches, MLSL, LD_MMA/LD_CCSAQ/LD_SLSQP, TADAH:ANNEAL_PT, TADAH:RANDOM,
 * DLIB:GFS).
 * ==================================================================== */

namespace {

const std::string kAnyTolerance =
    "; set MAXEVAL, MAXTIME, STOPVAL or a positive tolerance it honours "
    "(FTOL_REL, FTOL_ABS, XTOL_REL, XTOL_ABS)";
const std::string kNoTolerance =
    "; set MAXEVAL, MAXTIME or STOPVAL (it honours no tolerance)";
/* DIRECT tests XTOL on every rectangle it divides, the largest included:
 * an XTOL ends it, but only once the whole box is divided that finely. */
const std::string kDirectXtol =
    "; DIRECT's XTOL_REL and XTOL_ABS end it only once every rectangle is "
    "that small, after roughly (1/XTOL_REL)^n evaluations in n dimensions";

}  // namespace

TEST_CASE("startup validation: a backend with no convergence test and no "
          "finite schedule needs a stop", "[startup_validation][strategy][stop]")
{
    struct Case { std::string what, body, starts, ends; };
    const std::string plain = "    TYPE PLAIN\n";
    const auto nlopt = [&](const std::string& algo, const std::string& extra) {
        return plain + "    LIB NLOPT\n    ALGO " + algo + "\n" + extra;
    };
    const std::string top = "the STRATEGY-level optimiser: ";
    std::vector<Case> cases;
    for (const std::string algo :
         {"GN_DIRECT", "GN_DIRECT_L", "GN_DIRECT_L_RAND", "GN_DIRECT_NOSCAL",
          "GN_DIRECT_L_NOSCAL", "GN_DIRECT_L_RAND_NOSCAL"})
        cases.push_back({"NLOPT:" + algo + " without a stop", nlopt(algo, ""),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kAnyTolerance + kDirectXtol});
    for (const std::string algo :
         {"GN_CRS2_LM", "GN_ISRES", "LD_MMA", "LD_CCSAQ", "LD_SLSQP"})
        cases.push_back({"NLOPT:" + algo + " without a stop", nlopt(algo, ""),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kAnyTolerance});
    /* A tolerance of 0 or below is no stop: NLopt switches such a
     * tolerance off, and ANNEAL_PT's best-stall rule arms only above 0.
     * (FTOL_REL 0 is a common way to write "no tolerance".) */
    for (const std::string& tol :
         {std::string("FTOL_REL 0"), std::string("FTOL_ABS 0"),
          std::string("XTOL_REL -1"), std::string("XTOL_ABS 0")}) {
        cases.push_back({"NLOPT:GN_DIRECT_L with " + tol,
                         nlopt("GN_DIRECT_L", "    " + tol + "\n"),
                         top + "NLOPT:GN_DIRECT_L needs a stop: ",
                         kAnyTolerance + kDirectXtol});
        cases.push_back({"NLOPT:LD_MMA with " + tol,
                         nlopt("LD_MMA", "    " + tol + "\n"),
                         top + "NLOPT:LD_MMA needs a stop: ", kAnyTolerance});
        cases.push_back({"NLOPT:GN_CRS2_LM with " + tol,
                         nlopt("GN_CRS2_LM", "    " + tol + "\n"),
                         top + "NLOPT:GN_CRS2_LM needs a stop: ",
                         kAnyTolerance});
    }
    for (const std::string& tol :
         {std::string("FTOL_REL 0"), std::string("FTOL_ABS 0"),
          std::string("FTOL_REL 0\n    FTOL_ABS 0")})
        cases.push_back({"TADAH:ANNEAL_PT with " + tol,
                         plain + "    LIB TADAH\n    ALGO ANNEAL_PT\n    " +
                             tol + "\n",
                         top + "TADAH:ANNEAL_PT needs a stop: ",
                         "; set MAXEVAL, MAXTIME, STOPVAL or a positive "
                         "tolerance it honours (FTOL_REL, FTOL_ABS)"});
    /* A RESTART that is not active (MAX 0, or no AFTER_STAGNATION) arms no
     * stagnation stop. */
    for (const std::string& restart :
         {std::string("      AFTER_STAGNATION 60\n      MAX 0\n"),
          std::string("      MAX 2\n")})
        cases.push_back({"NLOPT:GN_DIRECT_L under an inactive RESTART",
                         nlopt("GN_DIRECT_L", "    RESTART\n" + restart +
                                                  "      METHOD RANDOM 1\n"
                                                  "    ENDRESTART\n"),
                         top + "NLOPT:GN_DIRECT_L needs a stop: ",
                         kAnyTolerance + kDirectXtol});
    for (const std::string algo :
         {"GN_ESCH", "GN_AGS", "GD_STOGO", "GD_STOGO_RAND"})
        cases.push_back({"NLOPT:" + algo + " without a stop", nlopt(algo, ""),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    /* A tolerance the algorithm does not test is no stop. */
    cases.push_back({"NLOPT:GN_ESCH with FTOL_REL",
                     nlopt("GN_ESCH", "    FTOL_REL 1e-6\n"),
                     top + "NLOPT:GN_ESCH needs a stop: ", kNoTolerance});
    /* MLSL hands its tolerances to its local searches only. */
    for (const std::string algo :
         {"GN_MLSL", "GN_MLSL_LDS", "GD_MLSL", "GD_MLSL_LDS"})
        cases.push_back({"NLOPT:" + algo + " with XTOL_REL",
                         nlopt(algo, "    XTOL_REL 1e-4\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    for (const std::string algo : {"G_MLSL", "G_MLSL_LDS"})
        cases.push_back({"NLOPT:" + algo + " with an INNER block",
                         nlopt(algo, "    FTOL_REL 1e-6\n    INNER\n"
                                     "      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                                     "      MAXEVAL 5\n    ENDINNER\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    cases.push_back({"TADAH:ANNEAL_PT without a stop",
                     plain + "    LIB TADAH\n    ALGO ANNEAL_PT\n",
                     top + "TADAH:ANNEAL_PT needs a stop: ",
                     "; set MAXEVAL, MAXTIME, STOPVAL or a positive tolerance "
                     "it honours (FTOL_REL, FTOL_ABS)"});
    cases.push_back({"TADAH:RANDOM with a tolerance only",
                     plain + "    LIB TADAH\n    ALGO RANDOM\n"
                             "    FTOL_REL 1e-6\n",
                     top + "TADAH:RANDOM needs a stop: ", kNoTolerance});
    /* DLIB:GFS reads FTOL_ABS as its solver's precision, which stops
     * nothing. */
    cases.push_back({"DLIB:GFS with FTOL_ABS only",
                     plain + "    LIB DLIB\n    ALGO GFS\n"
                             "    FTOL_ABS 1e-3\n",
                     top + "DLIB:GFS needs a stop: ", kNoTolerance});
    /* Each phase, named. */
    cases.push_back({"each MULTISTART start: NLOPT:LD_MMA without a stop",
                     "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LD_MMA\n"
                     "    N_STARTS 2\n",
                     "each MULTISTART start: NLOPT:LD_MMA needs a stop: ",
                     kAnyTolerance});
    cases.push_back({"the PRESTAGE exploration: NLOPT:GN_CRS2_LM without a stop",
                     "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                     "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                     "      ALGO GN_CRS2_LM\n    ENDEXPLORE\n",
                     "the PRESTAGE exploration (EXPLORE block): "
                     "NLOPT:GN_CRS2_LM needs a stop: ", kAnyTolerance});
    cases.push_back({"the PRESTAGE refinement: NLOPT:GN_DIRECT_L without a stop",
                     "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO GN_DIRECT_L\n"
                     "    EXPLORE\n      LIB DLIB\n      ALGO GFS\n"
                     "      MAXEVAL 12\n    ENDEXPLORE\n",
                     "the PRESTAGE refinement: NLOPT:GN_DIRECT_L needs a stop: ",
                     kAnyTolerance + kDirectXtol});
    /* AUGLAG ends when its local search ends: an INNER search that needs a
     * stop takes MAXEVAL, MAXTIME and STOPVAL from the enclosing block
     * (NLopt hands them on, and replaces the INNER block's own STOPVAL),
     * and its own MAXEVAL, MAXTIME and tolerances. */
    const std::string auglag_direct =
        "; set MAXEVAL, MAXTIME or a positive tolerance it honours (FTOL_REL, "
        "FTOL_ABS, XTOL_REL, XTOL_ABS) in the INNER block, or MAXEVAL, "
        "MAXTIME or STOPVAL in the NLOPT:AUGLAG block, which NLopt hands on "
        "to it" + kDirectXtol;
    cases.push_back({"NLOPT:AUGLAG with an INNER NLOPT:GN_DIRECT_L, no stop",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_DIRECT_L\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG "
                           "(NLOPT:GN_DIRECT_L) needs a stop: ",
                     auglag_direct});
    cases.push_back({"NLOPT:AUGLAG with an INNER STOPVAL (NLopt replaces it)",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_DIRECT_L\n"
                                     "      STOPVAL 0.1\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG "
                           "(NLOPT:GN_DIRECT_L) needs a stop: ",
                     auglag_direct});
    cases.push_back({"NLOPT:AUGLAG with an INNER XTOL_REL of 0",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_DIRECT_L\n"
                                     "      XTOL_REL 0\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG "
                           "(NLOPT:GN_DIRECT_L) needs a stop: ",
                     auglag_direct});
    cases.push_back({"NLOPT:AUGLAG with an INNER NLOPT:GN_ESCH and a tolerance",
                     nlopt("AUGLAG", "    FTOL_REL 1e-6\n    INNER\n"
                                     "      LIB NLOPT\n      ALGO GN_ESCH\n"
                                     "      FTOL_REL 1e-6\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG (NLOPT:GN_ESCH) "
                           "needs a stop: ",
                     "; set MAXEVAL or MAXTIME in the INNER block, or MAXEVAL, "
                     "MAXTIME or STOPVAL in the NLOPT:AUGLAG block, which "
                     "NLopt hands on to it"});
    /* Without an INNER block NLopt runs its default local optimiser, which
     * takes the block's tolerances too: LD_MMA for LD_AUGLAG. */
    for (const std::string& extra :
         {std::string(""), std::string("    FTOL_REL 0\n")})
        cases.push_back({"NLOPT:LD_AUGLAG without an INNER block or a stop "
                         "(block lines: '" + extra + "')",
                         nlopt("LD_AUGLAG", extra),
                         top + "the local search of NLOPT:LD_AUGLAG (NLopt's "
                               "default NLOPT:LD_MMA, as the block has no "
                               "INNER block) needs a stop: ",
                         "; set MAXEVAL, MAXTIME, STOPVAL or a positive "
                         "tolerance it honours (FTOL_REL, FTOL_ABS, XTOL_REL, "
                         "XTOL_ABS) in the NLOPT:LD_AUGLAG block, which NLopt "
                         "hands on to it"});
    for (const auto& c : cases) {
        INFO(c.what << "\n" << c.body);
        const auto v = judge(strategy_config(c.body));
        CHECK_THAT(v.startup,
                   Catch::StartsWith(c.starts) && Catch::EndsWith(c.ends));
    }

    /* The whole message of one NLopt global search. */
    CHECK(judge(strategy_config(nlopt("GN_DIRECT", ""))).startup ==
          top + "NLOPT:GN_DIRECT needs a stop: NLopt's global search has no "
                "convergence test of its own and runs until it is "
                "interrupted" + kAnyTolerance + kDirectXtol);
}

/* ======================================================================
 * TADAH:RANDOM, TADAH:GRID and DLIB:GFS: the stops they honour, and what
 * they reject. RANDOM and GRID honour MAXTIME and STOPVAL (they used to
 * accept them and ignore them), GFS through dlib's own run-time limit and
 * stop condition; GFS rejects the tolerances it cannot apply, and each
 * rejects a PARAM it does not take, naming the ones it does.
 * ==================================================================== */

TEST_CASE("startup validation: TADAH:RANDOM/GRID and DLIB:GFS reject what "
          "they cannot honour", "[startup_validation][strategy][stop]")
{
    struct Case { std::string what, body, message; };
    const std::string top = "the STRATEGY-level optimiser: ";
    const std::string random = "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                               "    MAXEVAL 5\n";
    const std::string grid = "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n";
    const std::string gfs = "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
                            "    MAXEVAL 5\n";
    const std::string gfs_why =
        ": dlib's global search has no convergence test to apply it to; stop "
        "it with MAXEVAL, MAXTIME or STOPVAL";
    for (const auto& c : std::vector<Case>{
             {"RANDOM PARAM", random + "    PARAM FOO 1\n",
              top + "TADAH:RANDOM: unknown PARAM 'FOO' (it takes none)"},
             {"RANDOM word PARAM", random + "    PARAM FOO AUTO\n",
              top + "TADAH:RANDOM: unknown PARAM 'FOO' (it takes none)"},
             {"GFS PARAM", gfs + "    PARAM solver_eps 1e-3\n",
              top + "DLIB:GFS: unknown PARAM 'solver_eps' (it takes none)"},
             {"GRID PARAM", grid + "    PARAM GRID_STEPS 5\n",
              top + "TADAH:GRID: unknown PARAM 'GRID_STEPS' (it takes "
                    "GRID_POINTS)"},
             {"GRID word PARAM", grid + "    PARAM FOO AUTO\n",
              top + "TADAH:GRID: unknown PARAM 'FOO' (it takes GRID_POINTS)"},
             {"GFS FTOL_REL", gfs + "    FTOL_REL 1e-6\n",
              top + "DLIB:GFS does not support FTOL_REL" + gfs_why},
             {"GFS XTOL_REL", gfs + "    XTOL_REL 1e-6\n",
              top + "DLIB:GFS does not support XTOL_REL" + gfs_why},
             {"GFS XTOL_ABS", gfs + "    XTOL_ABS 1e-6\n",
              top + "DLIB:GFS does not support XTOL_ABS" + gfs_why}}) {
        INFO(c.what << "\n" << c.body);
        CHECK(judge(strategy_config(c.body)).startup == c.message);
    }
    /* The stops they honour. */
    for (const std::string& body : std::vector<std::string>{
             random + "    MAXTIME 60\n    STOPVAL 0.1\n",
             grid + "    MAXTIME 60\n    STOPVAL 0.1\n",
             gfs + "    MAXTIME 60\n    STOPVAL 0.1\n    FTOL_ABS 1e-3\n"}) {
        INFO(body);
        CHECK(judge(strategy_config(body)).startup == "");
    }
}

/* ======================================================================
 * MULTISTART runs a multi-seed backend (DLIB:GFS) once, from every INIT
 * point: BUDGET_PER_START, each start's budget, has no effect there.
 * ==================================================================== */

TEST_CASE("startup validation: BUDGET_PER_START with a multi-seed backend is "
          "rejected", "[startup_validation][strategy][multistart]")
{
    const std::string message =
        "BUDGET_PER_START has no effect with DLIB:GFS, which runs every start "
        "in one optimiser; use MAXEVAL";
    for (const std::string& extra :
         {std::string("    MAXEVAL 40\n    BUDGET_PER_START 5\n"),
          std::string("    BUDGET_PER_START 5\n")}) {
        INFO(extra);
        const auto v = judge(strategy_config(
            "    TYPE MULTISTART\n    LIB DLIB\n    ALGO GFS\n"
            "    N_STARTS 2\n" + extra));
        CHECK(v.startup == message);
        CHECK(v.policy == message);
    }
    /* A backend that runs from each start takes it. */
    const auto v = judge(strategy_config(
        "    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
        "    N_STARTS 2\n    BUDGET_PER_START 5\n"));
    CHECK(v.startup == "");
    CHECK(v.policy == "");
}

TEST_CASE("startup validation: hpo_run rejects a backend setting of a "
          "later phase before it loads any data",
          "[startup_validation][strategy][backend][e2e]")
{
    struct ForceRebuild {
        bool restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        ForceRebuild() { if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1); }
        ~ForceRebuild() { if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD"); }
    } force_rebuild;

    SECTION("the PRESTAGE refinement's MAXTIME, before the exploration") {
        /* The exploration's 3 evaluations used to run first. */
        const auto err = run_hpo(
            "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n"
            "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
            "      MAXEVAL 3\n      SEED 1\n    ENDEXPLORE\n"
            "    REFINE\n      LIB DLIB\n      ALGO BFGS\n"
            "      MAXTIME 60\n    ENDREFINE\n  ENDSTRATEGY\nENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("DLIB:BFGS does not support MAXTIME"));
    }
    SECTION("each MULTISTART start's SEED, instead of a run without an "
            "evaluation") {
        const auto err = run_hpo(
            "OPTIMIZER\n  INIT\n    METHOD RANDOM 2\n    SEED 3\n  ENDINIT\n"
            "  STRATEGY\n    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
            "    MAXEVAL 10\n    N_STARTS 2\n    SEED 5\n  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("DLIB:BFGS does not support SEED"));
    }
}

TEST_CASE("startup validation: finite differences fit every OPTIM box, "
          "with valid settings that take effect",
          "[startup_validation][strategy][fd_stencil]")
{
    /* No finite-difference point is evaluated outside the OPTIM box, with
     * either step rule (fd_stencil.h): next to a bound a CENTRAL stencil is
     * one-sided and needs two steps inside the box, a FORWARD one steps
     * back. A box in which some point has no stencil is refused at startup
     * on each phase that takes finite-difference gradients with its own
     * settings, naming the step's source and the largest fd_step that fits
     * (owner decision D1(a)); the step is not shrunk. PARAM fd_step,
     * fd_kernel and diff_method must be valid there (diff_method 2,
     * RIDDERS, is refused: D3(a)), and a phase that takes no
     * finite-difference gradients refuses them: they never took effect
     * (D5(a)). The first attempt named only the in-house stencil and its
     * three steps of the largest step anywhere in the box; the owner's
     * decisions D1-D5 change the messages and the cases below. */
    const std::string bfgs =
        "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n";
    const std::string auto_refusal =
        "PARAM fd_step 0.4 does not fit these OPTIM boxes: no "
        "finite-difference point is evaluated outside the box, and next to a "
        "bound the CENTRAL stencil turns one-sided and needs two steps of "
        "h = 0.4 of each box width (in-house rule) inside it; the step is not "
        "shrunk:\n  - WATOM: OPTIM box [0.1, 5] (internal [0, 1]); no stencil "
        "fits at WATOM = 1.57; fd_step up to 0.333333 fits it\n  Set PARAM "
        "fd_step to at most 0.333333.";
    SECTION("OPTIM_SCALE AUTO: fd_step up to 1/3 (CENTRAL), 1/2 (FORWARD)") {
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_step 0.4\n")).startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: " + auto_refusal);
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_step 0.33\n")).startup ==
              "");
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_step "
                                    "0.3333333333333333\n")).startup == "");
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_step 0.5\n"
                                    "    PARAM diff_method 1\n")).startup == "");
        CHECK_THAT(judge(strategy_config(bfgs + "    PARAM fd_step 0.51\n"
                                         "    PARAM diff_method 1\n")).startup,
                   Catch::StartsWith("the STRATEGY-level optimiser DLIB:BFGS: "
                                     "PARAM fd_step 0.51 does not fit these "
                                     "OPTIM boxes: no finite-difference point "
                                     "is evaluated outside the box, and next "
                                     "to a bound the FORWARD stencil steps "
                                     "backward and needs one step of h = 0.51 "
                                     "of each box width (in-house rule) inside "
                                     "it") &&
                       Catch::EndsWith("fd_step up to 0.5 fits it\n  Set PARAM "
                                       "fd_step to at most 0.5."));
        /* HYBRID: any parameter may need finite differences */
        CHECK(judge(strategy_config(bfgs + "    GRAD_STRATEGY HYBRID\n"
                                    "    PARAM fd_step 0.4\n")).startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: " + auto_refusal);
        /* Ceres' rule is bounded too (D2(a)); in the unit box its step,
         * fd_step x |u|, is largest at the upper bound, where the one-sided
         * stencil needs 1 - 2 fd_step >= 0: up to 1/2, and 1/2 itself -- the
         * step 0.5 u is exact, and 1 - 2 (0.5 x 1) = 0 lies on the bound.
         * The check refused 0.5, with a margin of an ulp or two on every
         * step, and named 0.499999 as the largest fd_step that fits
         * (review 2, finding 3). */
        CHECK_THAT(judge(strategy_config(bfgs + "    PARAM fd_kernel ceres\n"
                                         "    PARAM fd_step 0.51\n")).startup,
                   Catch::Contains("h = max(2^-26, 0.51 x |u|) in the unit box "
                                   "(Ceres' rule)") &&
                       Catch::EndsWith("fd_step up to 0.5 fits it\n  Set "
                                       "PARAM fd_step to at most 0.5."));
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_kernel ceres\n"
                                    "    PARAM fd_step 0.5\n")).startup == "");
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_kernel ceres\n"
                                    "    PARAM fd_step 0.49\n")).startup == "");
        /* FORWARD with Ceres' rule: up to 1, the backward step from u to
         * u - u = 0 */
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_kernel ceres\n"
                                    "    PARAM diff_method 1\n"
                                    "    PARAM fd_step 1\n")).startup == "");
        CHECK_THAT(judge(strategy_config(bfgs + "    PARAM fd_kernel ceres\n"
                                         "    PARAM diff_method 1\n"
                                         "    PARAM fd_step 1.01\n")).startup,
                   Catch::EndsWith("fd_step up to 1 fits it\n  Set PARAM "
                                   "fd_step to at most 1."));
    }
    SECTION("each phase that takes finite-difference gradients") {
        const std::string inner =
            "    INNER\n      LIB NLOPT\n      ALGO LD_LBFGS\n"
            "      MAXEVAL 10\n    ENDINNER\n";
        const std::vector<std::pair<std::string, std::string>> phases = {
            {"the STRATEGY-level optimiser DLIB:BFGS",
             bfgs + "    PARAM fd_step 0.4\n"},
            {"each MULTISTART start DLIB:BFGS",
             "    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n"
             "    N_STARTS 2\n    PARAM fd_step 0.4\n"},
            {"the PRESTAGE refinement DLIB:BFGS",
             "    TYPE PRESTAGE\n    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
             "      MAXEVAL 3\n      SEED 1\n    ENDEXPLORE\n    REFINE\n"
             "      LIB DLIB\n      ALGO BFGS\n      MAXEVAL 20\n"
             "      PARAM fd_step 0.4\n    ENDREFINE\n"},
            {"the PRESTAGE exploration (EXPLORE block) NLOPT:LD_LBFGS",
             "    TYPE PRESTAGE\n    EXPLORE\n      LIB NLOPT\n      ALGO LD_LBFGS\n"
             "      MAXEVAL 5\n      PARAM fd_step 0.4\n    ENDEXPLORE\n"
             "    REFINE\n      LIB DLIB\n      ALGO BFGS\n      MAXEVAL 20\n"
             "    ENDREFINE\n"},
            {"the STRATEGY-level optimiser NLOPT:G_MLSL_LDS",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n    MAXEVAL 50\n"
             "    PARAM fd_step 0.4\n" + inner},
            {"the STRATEGY-level optimiser NLOPT:AUGLAG",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n    MAXEVAL 50\n"
             "    PARAM fd_step 0.4\n" + inner}};
        for (const auto& [name, body] : phases) {
            INFO(body);
            CHECK(judge(strategy_config(body)).startup ==
                  name + ": " + auto_refusal);
        }
        /* RIDDERS on each of them */
        for (const auto& [name, body] : phases) {
            std::string b = body;
            b.replace(b.find("PARAM fd_step 0.4"), 17, "PARAM diff_method 2");
            INFO(b);
            CHECK(judge(strategy_config(b)).startup ==
                  name + ": PARAM diff_method 2 (RIDDERS) is refused: Ridders' "
                  "extrapolation ignored PARAM fd_step, started from a step of "
                  "0.32 x max(|u|, 1) -- outside most OPTIM boxes -- and took "
                  "up to 20 evaluations per parameter; use diff_method 0 "
                  "(CENTRAL, the default) or 1 (FORWARD)");
        }
    }
    SECTION("OPTIM_SCALE NONE: Ceres' rule, the default, over the whole box") {
        const auto none = [&](const std::string& optim,
                              const std::string& extra) {
            return "OPTIM_SCALE NONE\n" + optim + "PC_ERMSE 0 1\nOPTIMIZER\n"
                   "  STRATEGY\n" + bfgs + extra + "  ENDSTRATEGY\n"
                   "ENDOPTIMIZER\n";
        };
        /* a box of relative width 1e-6 ([1000, 1000.001] alike; the seed
         * WATOM = 1 lies in this one) */
        CHECK(judge(none("OPTIM WATOM (1) 1 1.000001\n", "")).startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: the default fd_step "
              "1e-06 does not fit these OPTIM boxes: no finite-difference "
              "point is evaluated outside the box, and next to a bound the "
              "CENTRAL stencil turns one-sided and needs two steps of h = "
              "max(2^-26, 1e-06 x |u|) in internal units (Ceres' rule) inside "
              "it; the step is not shrunk:\n  - WATOM: OPTIM box [1, "
              "1.000001]; no stencil fits at WATOM = 1.0000005; fd_step up to "
              "3.33333e-07 fits it\n  Set PARAM fd_step to at most "
              "3.33333e-07; or use OPTIM_SCALE AUTO (fd_step is then a "
              "fraction of each box width).");
        CHECK(judge(none("OPTIM WATOM (1) 1 1.000001\n",
                         "    PARAM fd_step 3.3e-7\n")).startup == "");
        /* the in-house rule under NONE, h = fd_step x max(|u|, 1): the
         * exact limit of [0.1, 5] is 4.9 / 10.1 = 0.48515 (it was refused
         * above 0.3267, three steps of the largest step anywhere) */
        CHECK_THAT(judge(none("OPTIM WATOM (1) 0.1 5.0\n",
                              "    PARAM fd_kernel inhouse\n"
                              "    PARAM fd_step 0.5\n")).startup,
                   Catch::Contains("\n  - WATOM: OPTIM box [0.1, 5]; no stencil "
                                   "fits at WATOM = ") &&
                       Catch::Contains("; fd_step up to 0.485148 fits it\n  Set "
                                       "PARAM fd_step to at most 0.485148; or "
                                       "use OPTIM_SCALE AUTO (fd_step is then "
                                       "a fraction of each box width); or give "
                                       "a positive parameter that spans "
                                       "decades a LOG box."));
        CHECK(judge(none("OPTIM WATOM (1) 0.1 5.0\n",
                         "    PARAM fd_kernel inhouse\n"
                         "    PARAM fd_step 0.48\n")).startup == "");
        /* the step the NOISE block's AUTO derived: 1e-3 of the RMS width,
         * 0.0707107, far wider than the CGRID2B box */
        const std::string noise =
            "OPTIM_SCALE NONE\nOPTIM WATOM (1) 0.1 100.1\n"
            "OPTIM CGRID2B (1) 1.0 1.001\nPC_ERMSE 0 1\nOPTIMIZER\n"
            "  NOISE\n    AUTO true\n  ENDNOISE\n  STRATEGY\n" + bfgs +
            "  ENDSTRATEGY\nENDOPTIMIZER\n";
        CHECK_THAT(judge(noise).startup,
                   Catch::StartsWith("the STRATEGY-level optimiser DLIB:BFGS: "
                                     "fd_step 0.07071067812, derived by the "
                                     "NOISE block's AUTO from the box "
                                     "diagonal, does not fit these OPTIM "
                                     "boxes") &&
                       Catch::Contains("\n  - CGRID2B[1]: OPTIM box [1, 1.001]; "
                                       "no stencil fits at CGRID2B[1] = ") &&
                       Catch::Contains("  Set PARAM fd_step to at most ") &&
                       Catch::Contains(" (a PARAM fd_step overrides NOISE "
                                       "AUTO); or use OPTIM_SCALE AUTO") &&
                       !Catch::Contains("WATOM"));
    }
    SECTION("invalid settings") {
        CHECK(judge(strategy_config(bfgs + "    PARAM fd_kernel fast\n"))
                  .startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: PARAM fd_kernel must "
              "be inhouse or ceres (got 'fast')");
        /* HYBRID ignored an invalid fd_kernel (review item 5) */
        CHECK(judge(strategy_config(bfgs + "    GRAD_STRATEGY HYBRID\n"
                                    "    PARAM fd_kernel fast\n")).startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: PARAM fd_kernel must "
              "be inhouse or ceres (got 'fast')");
        CHECK(judge(strategy_config(bfgs + "    PARAM diff_method 3\n"))
                  .startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: PARAM diff_method must "
              "be 0 (CENTRAL, the default) or 1 (FORWARD) (got 3)");
        CHECK(judge(strategy_config(
                        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
                        "    MAXEVAL 20\n    PARAM fd_step 0\n")).startup ==
              "the STRATEGY-level optimiser NLOPT:LD_LBFGS: PARAM fd_step must "
              "be a positive finite number (got 0)");
        /* a number for fd_kernel: the same message on every backend. DLIB:BFGS
         * said it "does not support PARAM fd_kernel" while listing it as
         * supported, and the CERES drivers left it out of their list (review
         * 2, finding 2). */
        for (const std::string& backend :
             {std::string("DLIB:BFGS"), std::string("CERES:LBFGS"),
              std::string("NLOPT:LD_LBFGS")}) {
            const std::string lib = backend.substr(0, backend.find(':'));
            const std::string algo = backend.substr(backend.find(':') + 1);
            INFO(backend);
            CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB " + lib +
                                        "\n    ALGO " + algo +
                                        "\n    MAXEVAL 20\n"
                                        "    PARAM fd_kernel 1\n")).startup ==
                  "the STRATEGY-level optimiser " + backend + ": PARAM "
                  "fd_kernel must be inhouse or ceres (got 1)");
        }
    }
    SECTION("settings that never take effect are refused") {
        /* an INNER block takes the gradient of the block it runs in: its
         * own PARAM was accepted and ignored (NLopt set_param) */
        const std::string mlsl =
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
            "    MAXEVAL 50\n";
        CHECK(judge(strategy_config(mlsl +
                                    "    INNER\n      LIB NLOPT\n"
                                    "      ALGO LD_LBFGS\n      MAXEVAL 10\n"
                                    "      PARAM fd_step 0.4\n"
                                    "      PARAM fd_kernel ceres\n"
                                    "    ENDINNER\n")).startup ==
              "the INNER block of the STRATEGY-level optimiser NLOPT:LD_LBFGS: "
              "PARAM fd_step and fd_kernel never take effect: an INNER search "
              "takes the gradient of the block it runs in, with that block's "
              "finite-difference settings; set them in the enclosing block.");
        /* a derivative-free optimiser */
        CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB NLOPT\n"
                                    "    ALGO LN_SBPLX\n    MAXEVAL 20\n"
                                    "    PARAM fd_step 0.4\n")).startup ==
              "the STRATEGY-level optimiser NLOPT:LN_SBPLX: PARAM fd_step "
              "never takes effect: NLOPT:LN_SBPLX takes no gradient; remove "
              "it.");
        /* GRAD_STRATEGY ANALYTICAL takes no finite differences */
        CHECK(judge(strategy_config(bfgs + "    GRAD_STRATEGY ANALYTICAL\n"
                                    "    PARAM diff_method 1\n")).startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: PARAM diff_method never "
              "takes effect: GRAD_STRATEGY ANALYTICAL takes no finite "
              "differences; remove it.");
    }
    SECTION("ASA's probes fit its boxes") {
        const std::string asa =
            "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n    MAXEVAL 20\n";
        CHECK(judge(strategy_config(asa)).startup == "");
        CHECK(judge("OPTIM_SCALE NONE\nOPTIM WATOM (1) 1 1.000001\n"
                    "PC_ERMSE 0 1\nOPTIMIZER\n  STRATEGY\n" + asa +
                    "  ENDSTRATEGY\nENDOPTIMIZER\n").startup ==
              "the STRATEGY-level optimiser ASA:ASA: ASA's finite-difference "
              "probes -- the reannealing and exit tangents at (1 + Delta) x, "
              "or (1 - Delta) x when that lies outside the box -- would be "
              "evaluated outside the OPTIM box of these dimensions:\n  - "
              "WATOM: OPTIM box [1, 1.000001], the default Delta_X 0.001: a "
              "probe of "
              "WATOM = 1.0000005 lies outside the box; Delta_X up to "
              "4.99999e-07 fits it\n  Lower the Delta; or use OPTIM_SCALE "
              "AUTO, where every box is the unit box (the tangents fit for "
              "Delta <= 1).");
        /* the message said "PARAM Delta_X 0.001" of the default too (review
         * 2, finding 9); a written one is a PARAM */
        CHECK_THAT(judge("OPTIM_SCALE NONE\nOPTIM WATOM (1) 1 1.000001\n"
                         "PC_ERMSE 0 1\nOPTIMIZER\n  STRATEGY\n" + asa +
                         "    PARAM Delta_X 0.002\n  ENDSTRATEGY\n"
                         "ENDOPTIMIZER\n").startup,
                   Catch::Contains("\n  - WATOM: OPTIM box [1, 1.000001], "
                                   "PARAM Delta_X 0.002: a probe of WATOM = "));
        /* the unit box: tangents up to Delta 1, the exit curvature up to
         * 0.5 */
        CHECK(judge(strategy_config(asa + "    PARAM Delta_X 1\n")).startup == "");
        CHECK(judge(strategy_config(asa + "    PARAM Delta_X 1.5\n")).startup !=
              "");
        CHECK(judge(strategy_config(asa + "    PARAM Delta_X 0.5\n"
                                    "    PARAM Curvature_0 0\n")).startup == "");
        CHECK_THAT(judge(strategy_config(asa + "    PARAM Delta_X 0.6\n"
                                         "    PARAM Curvature_0 0\n")).startup,
                   Catch::Contains("and the exit curvature at x +- Delta |x|, "
                                   "or x -+ Delta |x| and x -+ 2 Delta |x| next "
                                   "to a bound (Curvature_0 0)") &&
                       Catch::Contains("Delta_X up to 0.5 fits it"));
    }
}

TEST_CASE("startup validation: NOISE AUTO derives fd_step where the "
          "STRATEGY-level phase takes finite differences, and is refused "
          "where no slope is measured",
          "[startup_validation][strategy][fd_stencil][noise_auto]")
{
    /* The NOISE block's AUTO derives PARAM fd_step for the finite
     * differences of the STRATEGY-level phase (each MULTISTART start, the
     * PRESTAGE refinement). It follows the rule of the finite-difference
     * PARAMs (owner decision D5(a)): used where that phase takes
     * finite-difference gradients -- an MLSL or AUGLAG whose local search
     * takes gradients included, which were left without it -- and refused
     * at startup where it measures no slopes: a derivative-free optimiser,
     * or GRAD_STRATEGY ANALYTICAL, written or resolved. A derivative-free
     * optimiser printed "NOISE_AUTO derives PARAM fd_step for
     * gradient-based optimisers only ...; ignored" to stderr and ran (owner
     * decision D7, 2026-09-29). */
    struct Run {
        std::unique_ptr<HPOTargets> trg;
        std::string startup;
        std::string journal;
    };
    const auto run = [](const std::string& config) {
        PeriodicTable::initialize();
        ScopedTempFile tf(config);
        Context ctx = make_minimal_ctx();
        auto& log = tadah::hpo::HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        const fs::path jp = fs::temp_directory_path() /
                            (unique_name("tadah_svs_journal_") + ".log");
        log.open(jp.string(), "test", "test_cwd", 0);
        Run r;
        const std::string parse = error_of([&] {
            r.trg = std::make_unique<HPOTargets>(ctx, tf.path.string());
        });
        INFO("config:\n" << config);
        REQUIRE(parse == "");
        r.startup = error_of([&] { validate_hpo_setup(ctx, *r.trg, true, true); });
        log.finish("test-end");
        log.reset_for_test();
        std::ifstream is(jp);
        std::ostringstream os;
        os << is.rdbuf();
        r.journal = os.str();
        std::error_code ec;
        fs::remove(jp, ec);
        return r;
    };
    const auto noise = [](const std::string& strategy_body,
                          const std::string& root = kRoot) {
        return root + "OPTIMIZER\n  NOISE\n    AUTO true\n  ENDNOISE\n"
                      "  STRATEGY\n" + strategy_body +
               "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const std::string inner =
        "    INNER\n      LIB NLOPT\n      ALGO LD_LBFGS\n      MAXEVAL 10\n"
        "    ENDINNER\n";
    SECTION("derived where the STRATEGY-level phase takes finite "
            "differences") {
        /* WATOM [0.1, 5] is the unit box under OPTIM_SCALE AUTO: 1e-3 of
         * the RMS width */
        for (const std::string& body :
             {std::string("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 20\n"),
              std::string("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_LBFGS\n"
                          "    MAXEVAL 20\n"),
              "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
              "    MAXEVAL 50\n" + inner,
              "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
              "    MAXEVAL 50\n" + inner}) {
            INFO(body);
            const Run r = run(noise(body));
            CHECK(r.startup == "");
            CHECK(r.trg->spec.param.count("fd_step") == 1u);
            if (r.trg->spec.param.count("fd_step"))
                CHECK(r.trg->spec.param.at("fd_step") == 1e-3);
            CHECK(r.trg->spec.fd_step_from_noise_auto);
            CHECK_THAT(r.journal,
                       Catch::Contains("NOISE_AUTO: PARAM fd_step set to 0.001 "
                                       "(diag=1, dim=1)"));
        }
    }
    SECTION("refused where no slope is measured") {
        const Run sbplx = run(noise("    TYPE PLAIN\n    LIB NLOPT\n"
                                    "    ALGO LN_SBPLX\n    MAXEVAL 20\n"));
        CHECK(sbplx.startup ==
              "the STRATEGY-level optimiser NLOPT:LN_SBPLX: NOISE AUTO has no "
              "effect: NLOPT:LN_SBPLX measures no slopes; remove AUTO from "
              "the NOISE block.");
        CHECK(sbplx.trg->spec.param.count("fd_step") == 0u);
        CHECK(sbplx.journal.find("NOISE_AUTO") == std::string::npos);
        /* each MULTISTART start; the PRESTAGE refinement, whatever its
         * exploration takes */
        CHECK(run(noise("    TYPE MULTISTART\n    LIB NLOPT\n"
                        "    ALGO LN_NELDERMEAD\n    MAXEVAL 20\n"
                        "    N_STARTS 2\n"))
                  .startup ==
              "each MULTISTART start NLOPT:LN_NELDERMEAD: NOISE AUTO has no "
              "effect: NLOPT:LN_NELDERMEAD measures no slopes; remove AUTO "
              "from the NOISE block.");
        CHECK(run(noise("    TYPE PRESTAGE\n    EXPLORE\n      LIB NLOPT\n"
                        "      ALGO LD_LBFGS\n      MAXEVAL 5\n    ENDEXPLORE\n"
                        "    REFINE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                        "      MAXEVAL 20\n    ENDREFINE\n"))
                  .startup ==
              "the PRESTAGE refinement NLOPT:LN_SBPLX: NOISE AUTO has no "
              "effect: NLOPT:LN_SBPLX measures no slopes; remove AUTO from "
              "the NOISE block.");
        /* ASA takes its slopes with probes of its own */
        CHECK(run(noise("    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
                        "    MAXEVAL 20\n"))
                  .startup ==
              "the STRATEGY-level optimiser ASA:ASA: NOISE AUTO has no "
              "effect: ASA:ASA measures its slopes with probes of its own "
              "(PARAM Delta_X), not with fd_step; remove AUTO from the NOISE "
              "block.");
    }
    SECTION("refused under GRAD_STRATEGY ANALYTICAL, written or resolved") {
        const Run written = run(noise("    TYPE PLAIN\n    LIB DLIB\n"
                                      "    ALGO BFGS\n    MAXEVAL 20\n"
                                      "    GRAD_STRATEGY ANALYTICAL\n"));
        CHECK(written.startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: NOISE AUTO has no "
              "effect: GRAD_STRATEGY ANALYTICAL takes no finite differences; "
              "remove AUTO from the NOISE block.");
        CHECK(written.trg->spec.param.count("fd_step") == 0u);
        /* an unset GRAD_STRATEGY resolves to ANALYTICAL with PC_OBSERVABLE
         * (the check comes before the one that ANALYTICAL needs training
         * data) */
        const Run resolved = run(noise(
            "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n",
            "OPTIM WATOM (1) 0.1 5.0\nPC_EMAE 0 1\nPC_OBSERVABLE "
            "cohesive_energy --species Ta --target 8.1 --weight 1 --hotvar a0 "
            "2.7 3.9 3.3\n"));
        CHECK(resolved.startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: NOISE AUTO has no "
              "effect: GRAD_STRATEGY ANALYTICAL takes no finite differences "
              "(the resolution of an unset GRAD_STRATEGY with PC_OBSERVABLE); "
              "remove AUTO from the NOISE block.");
        CHECK(resolved.trg->spec.param.count("fd_step") == 0u);
    }
}

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

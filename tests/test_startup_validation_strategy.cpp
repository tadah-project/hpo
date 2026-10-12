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
#include "ceres_build.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/backend_rules.h>
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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
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

bool starts_with(const std::string& s, const std::string& head)
{
    return s.compare(0, head.size(), head) == 0;
}

/* Whether `message` is the refusal of one phase's settings: it starts with
 * the phase's name (strategy_phase_specs: "the STRATEGY-level optimiser",
 * "each MULTISTART start", "the MULTISTART run", "the PRESTAGE
 * exploration (EXPLORE block)", "the PRESTAGE refinement", "the INNER
 * block of <phase>"; validate_run_policy: "the EXPLORE block"; the check
 * of the GRAD_STRATEGY ANALYTICAL setup: "OPTIMIZER block", "EXPLORE
 * block"). */
bool names_phase(const std::string& message)
{
    for (const char* name : {"the STRATEGY-level optimiser",
                             "each MULTISTART start", "the MULTISTART run",
                             "the PRESTAGE exploration (EXPLORE block)",
                             "the EXPLORE block", "the PRESTAGE refinement",
                             "the INNER block of ", "OPTIMIZER block",
                             "EXPLORE block"})
        if (starts_with(message, name)) return true;
    return false;
}

/* The line of `config` that closes the block of the phase `message`
 * names, and its text, as the refusal of that phase's settings names it
 * (OptimizerSpec::block_line): the INNER block's ENDINNER for "the INNER
 * block of ..." (inside the EXPLORE block for the exploration's; without
 * an INNER block, NLopt's default local search, its phase's line),
 * ENDEXPLORE for the exploration, ENDOPTIMIZER otherwise; {0, ""} when the
 * configuration has no such line. */
std::pair<std::size_t, std::string> phase_line(const std::string& config,
                                               const std::string& message)
{
    std::vector<std::string> lines;
    {
        std::istringstream in(config);
        for (std::string l; std::getline(in, l);) lines.push_back(l);
    }
    const auto keyword = [&](std::size_t i) {
        const auto p = lines[i].find_first_not_of(' ');
        return p == std::string::npos ? std::string() : lines[i].substr(p);
    };
    std::size_t explore_begin = lines.size(), explore_end = lines.size();
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (keyword(i) == "EXPLORE" && explore_begin == lines.size())
            explore_begin = i;
        if (keyword(i) == "ENDEXPLORE" && explore_end == lines.size())
            explore_end = i;
    }
    const auto find = [&](const std::string& key, bool in_explore)
        -> std::pair<std::size_t, std::string> {
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const bool inside = i > explore_begin && i < explore_end;
            if (keyword(i) == key && inside == in_explore)
                return {i + 1, lines[i]};
        }
        return {0, ""};
    };
    const std::string inner_of = "the INNER block of ";
    const bool inner = starts_with(message, inner_of);
    const std::string parent = inner ? message.substr(inner_of.size())
                                     : message;
    const bool explore =
        starts_with(parent, "the PRESTAGE exploration (EXPLORE block)") ||
        starts_with(parent, "the EXPLORE block") ||
        starts_with(parent, "EXPLORE block");
    if (inner) {
        const auto at = find("ENDINNER", explore);
        if (at.first > 0) return at;
    }
    if (explore) return find("ENDEXPLORE", false);
    return find("ENDOPTIMIZER", false);
}

/* The parser must accept `config`; returns what validate_hpo_setup() and
 * the policy say about it ("" when they accept it). A refusal of the
 * OPTIMIZER block's settings names the line of the block it is about, in
 * the parser's form "Error on line N: <what>\nLine: <text>"
 * (with_block_line, phase_specs.h): `startup_raw` is the refusal as
 * validate_hpo_setup() threw it, `startup` the message without that line,
 * and judge() checks the line of every refusal that names a phase against
 * the configuration (phase_line), so the cases that compare `startup`
 * check it too; a refusal of the STRATEGY TYPE ("STRATEGY TYPE MULTISTART
 * cannot run ...") names no phase, and its line is checked by "startup
 * validation: a phase's settings refused at startup name the line that
 * closes its block". */
struct Verdict {
    std::string startup;
    std::string policy;
    std::string startup_raw;
};

/* `raw`, what validate_hpo_setup() threw for `config`, without the line it
 * names; when its message names a phase, checks that the line is the one
 * that closes the phase's block in `config` (phase_line), or that it names
 * none when `config` has no such line. */
std::string checked_refusal(const std::string& config, const std::string& raw)
{
    std::string message = raw;
    std::smatch m;
    const bool at_line = std::regex_match(
        raw, m, std::regex(R"(Error on line (\d+): ([\s\S]*)\nLine: ([^\n]*))"));
    if (at_line) message = m[2];
    if (names_phase(message)) {
        const auto expected = phase_line(config, message);
        INFO("startup refusal: " << raw);
        if (expected.first == 0) {
            CHECK_FALSE(at_line);
        } else {
            REQUIRE(at_line);
            CHECK(std::stoul(m[1]) == expected.first);
            CHECK(m[3] == expected.second);
        }
    }
    return message;
}

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
    v.startup_raw =
        error_of([&] { validate_hpo_setup(ctx, *trg, true, true); });
    v.startup = checked_refusal(config, v.startup_raw);
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
                      /* the lookup's text, "unknown optimizer" until it
                       * became "unknown optimiser ...: ..." */
                      !Catch::Contains("unknown optimi"));

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

/* How the journal writes the event `msg` with `tag`, without the
 * timestamp of its first line (HPOLog::event_locked): a multi-line message
 * -- a parse error, "Error on line N: ...\nLine: ..." -- is a block whose
 * first line carries a star and whose continuation lines repeat the tag. */
std::string journal_event(const std::string& tag, const std::string& msg)
{
    std::string padded = tag;
    if (padded.size() < 8) padded.resize(8, ' ');
    const bool multi = msg.find('\n') != std::string::npos;
    std::string out;
    std::size_t start = 0;
    for (bool first = true;; first = false) {
        const std::size_t nl = msg.find('\n', start);
        if (!first) out += "\n" + std::string(12, ' ');
        out += padded + "  " + (multi ? (first ? "* " : "  ") : "") +
               msg.substr(start, nl == std::string::npos ? std::string::npos
                                                         : nl - start);
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    return out;
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
    CHECK_THAT(journal.str(),
               Catch::Contains(journal_event("ERROR", "startup: " + err)));
    return err;
}

}  // namespace

TEST_CASE("startup validation: hpo_run rejects a strategy before it loads "
          "any data",
          "[startup_validation][strategy][e2e]")
{
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
 * NLOPT:LN_NEWUOA_BOUND is refused (the refusal table of the caps
 * registry, optimizer_caps.cpp). NLopt 2.10 plans every step of it with a
 * nested LD_MMA run that has no time limit and sees no stop request; once a
 * non-finite value reaches its model -- HUGE_VAL, NLopt's convention for a
 * forced stop, which HPO's objective returns after every stop it makes
 * (a signal, a RESTART stagnation stop, an error that ends the run), or a
 * model that overflows on a flat loss -- each step takes 10-60 s without
 * evaluating the loss, so Ctrl-C, SIGTERM, MAXTIME and RESTART wait that
 * long (docs/issues/nlopt_newuoa_bound_stall.md has the repro). The stall
 * itself is not a CTest: it is third-party code and takes that long.
 * ==================================================================== */

TEST_CASE("startup validation: NLOPT:LN_NEWUOA_BOUND is refused at the line "
          "that closes its block, in every phase",
          "[startup_validation][strategy][backend][refused]")
{
    /* Pins: the parser refuses NLOPT:LN_NEWUOA_BOUND at the lines where the
     * 1.6.0 rule checks a block's optimiser: ENDOPTIMIZER for the
     * STRATEGY-level optimiser, whether the STRATEGY block, a REFINE block
     * or the OPTIMIZER block itself names it (PLAIN, the short form,
     * MULTISTART, the PRESTAGE refinement; ENDSTRATEGY and ENDREFINE check
     * no optimiser), ENDEXPLORE for the PRESTAGE exploration, ENDINNER for
     * an INNER block (under G_MLSL_LDS and LN_AUGLAG, inside STRATEGY,
     * EXPLORE and REFINE), and the end of the file for an OPTIMIZER block
     * never closed -- naming the phase, as the other messages of these
     * blocks do, with the caps registry's reason and replacement. A spec
     * built outside the parser gets the refusal from validate_backend() (no
     * phase), validate_run_policy() and the MULTISTART policy (the
     * STRATEGY-level optimiser), validate_phase_backends() and
     * require_inner_backend() (its INNER block).
     * Before the fix: parse_error() was "" for every configuration (the run
     * started and NLopt ran the algorithm), and every validator accepted
     * the specs (""). */
    const std::string refused = "NLOPT:LN_NEWUOA_BOUND is refused: ";
    const std::string inner =
        "    INNER\n      LIB NLOPT\n      ALGO LN_NEWUOA_BOUND\n"
        "      MAXEVAL 20\n    ENDINNER\n";
    struct Case {
        std::string what;
        std::string config;
        std::size_t line;      // the line that closes the refused block
        std::string closing;   // its text
        std::string phase;     // what the message names
    };
    const std::string top = "the STRATEGY-level optimiser";
    for (const auto& c : std::vector<Case>{
             {"PLAIN",
              strategy_config("    TYPE PLAIN\n    LIB NLOPT\n"
                              "    ALGO LN_NEWUOA_BOUND\n    MAXEVAL 400\n"),
              10, "ENDOPTIMIZER", top},
             {"the short form inside OPTIMIZER",
              kRoot + "OPTIMIZER\nLIB NLOPT\nALGO LN_NEWUOA_BOUND\n"
                      "MAXEVAL 400\nENDOPTIMIZER\n",
              7, "ENDOPTIMIZER", top},
             {"MULTISTART",
              strategy_config(multistart_body("NLOPT", "LN_NEWUOA_BOUND", "2")),
              11, "ENDOPTIMIZER", top},
             {"the PRESTAGE refinement at STRATEGY level",
              strategy_config("    TYPE PRESTAGE\n    LIB NLOPT\n"
                              "    ALGO LN_NEWUOA_BOUND\n    MAXEVAL 20\n" +
                              kExploreGfs),
              15, "ENDOPTIMIZER", "the PRESTAGE refinement"},
             {"a REFINE block, at ENDOPTIMIZER (not ENDREFINE)",
              strategy_config("    TYPE PRESTAGE\n" + kExploreGfs +
                              "    REFINE\n      LIB NLOPT\n"
                              "      ALGO LN_NEWUOA_BOUND\n"
                              "      MAXEVAL 20\n    ENDREFINE\n"),
              17, "ENDOPTIMIZER", "the PRESTAGE refinement"},
             {"an EXPLORE block",
              strategy_config("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                              "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                              "      ALGO LN_NEWUOA_BOUND\n      MAXEVAL 12\n"
                              "    ENDEXPLORE\n"),
              13, "    ENDEXPLORE", "the EXPLORE block"},
             {"an INNER block under G_MLSL_LDS",
              strategy_config("    TYPE PLAIN\n    LIB NLOPT\n"
                              "    ALGO G_MLSL_LDS\n    MAXEVAL 60\n" + inner),
              13, "    ENDINNER", "the INNER block inside STRATEGY"},
             {"an INNER block under LN_AUGLAG",
              strategy_config("    TYPE PLAIN\n    LIB NLOPT\n"
                              "    ALGO LN_AUGLAG\n    MAXEVAL 60\n" + inner),
              13, "    ENDINNER", "the INNER block inside STRATEGY"},
             {"an INNER block inside EXPLORE",
              strategy_config("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                              "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                              "      ALGO G_MLSL_LDS\n      MAXEVAL 60\n" +
                              inner + "    ENDEXPLORE\n"),
              17, "    ENDINNER", "the INNER block inside EXPLORE"},
             {"an INNER block inside REFINE",
              strategy_config("    TYPE PRESTAGE\n" + kExploreGfs +
                              "    REFINE\n      LIB NLOPT\n"
                              "      ALGO G_MLSL_LDS\n      MAXEVAL 60\n" +
                              inner + "    ENDREFINE\n"),
              19, "    ENDINNER", "the INNER block inside REFINE"}}) {
        INFO(c.what << ", config:\n" << c.config);
        CHECK_THAT(parse_error(c.config),
                   Catch::StartsWith("Error on line " + std::to_string(c.line) +
                                     ": " + c.phase + ": " + refused) &&
                   Catch::Contains("Use NLOPT:LN_BOBYQA") &&
                   Catch::EndsWith("\nLine: " + c.closing));
    }
    /* An OPTIMIZER block the file never closes: at the end of the file,
     * with no line, as its missing LIB or ALGO. */
    CHECK_THAT(parse_error(kRoot + "OPTIMIZER\n  LIB NLOPT\n"
                                   "  ALGO LN_NEWUOA_BOUND\n  MAXEVAL 5\n"),
               Catch::StartsWith(top + ": " + refused));

    /* A spec built outside the parser. */
    const auto bound = [](const std::string& algo, std::uint64_t maxeval) {
        OptimizerSpec s;
        s.lib = "NLOPT";
        s.algo = algo;
        s.low = {0.0, 0.0};
        s.high = {1.0, 1.0};
        s.analyse_logscale();
        s.stop.maxeval = maxeval;
        return s;
    };
    OptimizerSpec sp = bound("LN_NEWUOA_BOUND", 5);
    CHECK_THAT(error_of([&] {
                   tadah::hpo::validate_backend(sp,
                                                tadah::hpo::RestartStop::Off);
               }),
               Catch::StartsWith(refused) &&
               Catch::Contains("Use NLOPT:LN_BOBYQA"));
    CHECK_THAT(error_of([&] { tadah::hpo::validate_run_policy(sp); }),
               Catch::StartsWith(top + ": " + refused));
    sp.strategy_type = StrategyType::MULTISTART;
    sp.multistart.n_starts = 2;
    CHECK_THAT(policy_error(sp), Catch::StartsWith(top + ": " + refused));

    OptimizerSpec root = bound("G_MLSL_LDS", 60);
    root.inner = std::make_unique<OptimizerSpec>(bound("LN_NEWUOA_BOUND", 20));
    const std::string of_top = "the INNER block of " + top + ": " + refused;
    CHECK_THAT(error_of([&] { tadah::hpo::validate_phase_backends(root); }),
               Catch::StartsWith(of_top));
    CHECK_THAT(error_of([&] { tadah::hpo::validate_run_policy(root); }),
               Catch::StartsWith(of_top));
    /* run_inner()'s safety net for a spec built at run time names no
     * phase. */
    CHECK_THAT(error_of([&] { tadah::hpo::require_inner_backend(root, ""); }),
               Catch::StartsWith("the INNER block of NLOPT:G_MLSL_LDS: " +
                                 refused));
}

/* ======================================================================
 * NLOPT:LN_NEWUOA and Ceres' least-squares solvers are refused, and an
 * unknown optimiser is an error, at the line that closes their block.
 * ==================================================================== */

TEST_CASE("startup validation: NLOPT:LN_NEWUOA, the Ceres least-squares "
          "solvers and an unknown optimiser are refused at the line that "
          "closes their block",
          "[startup_validation][strategy][backend][refused]")
{
    /* Pins: the parser refuses, at the line where it checks a block's
     * optimiser (ENDOPTIMIZER, ENDEXPLORE, ENDINNER) and naming the phase as
     * for LN_NEWUOA_BOUND (above):
     *   - NLOPT:LN_NEWUOA, which NLopt runs without the OPTIM box, in 1-D
     *     and in 2-D, written in the STRATEGY block or in the short form,
     *     in an EXPLORE block and in an INNER block, naming NLOPT:LN_BOBYQA;
     *   - CERES:LEVENBERG_MARQUARDT, DOGLEG and SUBSPACE_DOGLEG, Ceres'
     *     least-squares solvers, naming the four Ceres solvers HPO runs (in
     *     the STRATEGY block and in an EXPLORE block);
     *   - a LIB:ALGO HPO does not know -- an ALGO its library does not have,
     *     or a LIB that does not exist -- with what is wrong and `tadah
     *     explain ALGO`, in the STRATEGY block, an EXPLORE and an INNER
     *     block.
     * A spec built outside the parser gets the refusal from
     * validate_backend() too.
     * Before the fix: parse_error() was "" for LN_NEWUOA (accepted with a
     * startup WARNING; in 1-D the run then ended "FAILED -- dimension 1
     * must be >= 2" after 0 evaluations), and for the others it was the
     * HPOTargets constructor's lookup after the parse, with no line and no
     * alternatives: "unknown optimizer CERES:LEVENBERG_MARQUARDT (not
     * registered in optimizer_caps)", "unknown optimizer NLOPT:LD_FOO (not
     * registered in optimizer_caps)"; validate_backend() accepted
     * LN_NEWUOA and gave the Ceres solvers "CERES:LEVENBERG_MARQUARDT is
     * deferred to a later phase ...". */
    const std::string root2 =
        "OPTIM WATOM (1) 0.1 5.0\nOPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\n";
    const auto plain = [](const std::string& lib, const std::string& algo) {
        return "    TYPE PLAIN\n    LIB " + lib + "\n    ALGO " + algo +
               "\n    MAXEVAL 25\n";
    };
    const auto explore = [](const std::string& lib, const std::string& algo) {
        return "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
               "    MAXEVAL 20\n    EXPLORE\n      LIB " + lib +
               "\n      ALGO " + algo + "\n      MAXEVAL 12\n    ENDEXPLORE\n";
    };
    const auto inner = [](const std::string& lib, const std::string& algo) {
        return "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
               "    MAXEVAL 60\n    INNER\n      LIB " + lib +
               "\n      ALGO " + algo + "\n      MAXEVAL 20\n    ENDINNER\n";
    };
    const std::string newuoa =
        "NLOPT:LN_NEWUOA is refused: NLopt 2.10 runs LN_NEWUOA without the "
        "OPTIM box";
    const std::string bobyqa = "Use NLOPT:LN_BOBYQA";
    const std::string ceres_use =
        "Use CERES:LBFGS, CERES:WOLFE, CERES:NLCG or CERES:STEEPEST_DESCENT";
    const std::string explain = "; `tadah explain ALGO` lists the optimisers";
    const std::string top = "the STRATEGY-level optimiser";
    struct Case {
        std::string what;
        std::string config;
        std::size_t line;     // the line that closes the block
        std::string closing;  // its text
        std::string head;     // the message after "Error on line N: "
        std::string also;     // a part of the reason
    };
    std::vector<Case> cases = {
        {"LN_NEWUOA in 1-D", strategy_config(plain("NLOPT", "LN_NEWUOA")), 10,
         "ENDOPTIMIZER", top + ": " + newuoa, bobyqa},
        {"LN_NEWUOA in 2-D",
         root2 + "OPTIMIZER\n  STRATEGY\n" + plain("NLOPT", "LN_NEWUOA") +
             "  ENDSTRATEGY\nENDOPTIMIZER\n",
         11, "ENDOPTIMIZER", top + ": " + newuoa, bobyqa},
        {"LN_NEWUOA in the short form inside OPTIMIZER",
         kRoot + "OPTIMIZER\nLIB NLOPT\nALGO LN_NEWUOA\nMAXEVAL 25\n"
                 "ENDOPTIMIZER\n",
         7, "ENDOPTIMIZER", top + ": " + newuoa, bobyqa},
        {"LN_NEWUOA in an EXPLORE block",
         strategy_config(explore("NLOPT", "LN_NEWUOA")), 13, "    ENDEXPLORE",
         "the EXPLORE block: " + newuoa, bobyqa},
        {"LN_NEWUOA in an INNER block",
         strategy_config(inner("NLOPT", "LN_NEWUOA")), 13, "    ENDINNER",
         "the INNER block inside STRATEGY: " + newuoa, bobyqa},
        {"CERES:DOGLEG in an EXPLORE block",
         strategy_config(explore("CERES", "DOGLEG")), 13, "    ENDEXPLORE",
         "the EXPLORE block: CERES:DOGLEG is refused: ", ceres_use},
        {"an ALGO NLOPT does not have",
         strategy_config(plain("NLOPT", "LD_FOO")), 10, "ENDOPTIMIZER",
         top + ": unknown optimiser NLOPT:LD_FOO: LIB NLOPT has no ALGO "
               "LD_FOO" + explain, ""},
        {"a LIB that does not exist",
         strategy_config(plain("FOO", "BFGS")), 10, "ENDOPTIMIZER",
         top + ": unknown optimiser FOO:BFGS: there is no LIB FOO (LIB is "
               "ASA, CERES, DLIB, NLOPT or TADAH)" + explain, ""},
        {"an unknown ALGO in an EXPLORE block",
         strategy_config(explore("DLIB", "LBFGS")), 13, "    ENDEXPLORE",
         "the EXPLORE block: unknown optimiser DLIB:LBFGS: LIB DLIB has no "
         "ALGO LBFGS" + explain, ""},
        {"an unknown ALGO in an INNER block",
         strategy_config(inner("NLOPT", "LD_FOO")), 13, "    ENDINNER",
         "the INNER block inside STRATEGY: unknown optimiser NLOPT:LD_FOO: "
         "LIB NLOPT has no ALGO LD_FOO" + explain, ""}};
    for (const char* algo :
         {"LEVENBERG_MARQUARDT", "DOGLEG", "SUBSPACE_DOGLEG"})
        cases.push_back({std::string("CERES:") + algo,
                         strategy_config(plain("CERES", algo)), 10,
                         "ENDOPTIMIZER",
                         top + ": CERES:" + algo + " is refused: ", ceres_use});
    for (const auto& c : cases) {
        INFO(c.what << ", config:\n" << c.config);
        const std::string msg = parse_error(c.config);
        CHECK_THAT(msg,
                   Catch::StartsWith("Error on line " + std::to_string(c.line) +
                                     ": " + c.head) &&
                   Catch::Contains(c.also) &&
                   Catch::EndsWith("\nLine: " + c.closing));
    }

    /* A spec built outside the parser. */
    const auto spec = [](const std::string& lib, const std::string& algo) {
        OptimizerSpec s;
        s.lib = lib;
        s.algo = algo;
        s.low = {0.0, 0.0};
        s.high = {1.0, 1.0};
        s.analyse_logscale();
        s.stop.maxeval = 20;
        return s;
    };
    const auto backend_error = [](const OptimizerSpec& s) {
        return error_of([&] {
            tadah::hpo::validate_backend(s, tadah::hpo::RestartStop::Off);
        });
    };
    CHECK_THAT(backend_error(spec("NLOPT", "LN_NEWUOA")),
               Catch::StartsWith(newuoa) && Catch::Contains(bobyqa));
    CHECK_THAT(backend_error(spec("CERES", "LEVENBERG_MARQUARDT")),
               Catch::StartsWith("CERES:LEVENBERG_MARQUARDT is refused: ") &&
               Catch::Contains(ceres_use));
    CHECK_THAT(backend_error(spec("NLOPT", "LD_FOO")),
               Catch::StartsWith("unknown optimiser NLOPT:LD_FOO: "));
}

TEST_CASE("startup validation: validate_phase_backends refuses a backend for "
          "its own phase in every STRATEGY shape",
          "[startup_validation][strategy][backend][refused]")
{
    /* Pins: validate_phase_backends() reports a refused backend first, for
     * its own phase, with the text validate_run_policy() throws for it, in
     * every shape of the STRATEGY (specs built outside the parser, which
     * refuses them earlier): the STRATEGY-level optimiser under PLAIN and
     * under MULTISTART, without INIT points and with two (the starts are
     * planned from them); the PRESTAGE refinement with TOPK 1 and TOPK 3 (a
     * MULTISTART over three candidates); the EXPLORE block; the INNER block
     * of an LN_AUGLAG STRATEGY-level optimiser under MULTISTART with INIT
     * points and under PRESTAGE with TOPK 3, and of a G_MLSL_LDS
     * exploration.
     * Before the fix: strategy_phase_specs() looked up the caps of the
     * refused backend to plan the MULTISTART starts from the INIT points and
     * the TOPK 3 refinement, and the lookup threw the refusal with no phase
     * ("NLOPT:LN_NEWUOA_BOUND is refused: ..."); the other shapes named the
     * phase as strategy_phase_specs() does, not as validate_run_policy():
     * "the MULTISTART run" without INIT points, "the PRESTAGE exploration
     * (EXPLORE block)", "the INNER block of each MULTISTART start", "the
     * INNER block of the PRESTAGE exploration (EXPLORE block)". PLAIN, the
     * TOPK 1 refinement and the INNER block of the TOPK 3 refinement were
     * named as now. */
    const std::string refused = "NLOPT:LN_NEWUOA_BOUND is refused: ";
    const auto nlopt = [](const std::string& algo) {
        OptimizerSpec s;
        s.lib = "NLOPT";
        s.algo = algo;
        s.low = {0.0, 0.0};
        s.high = {1.0, 1.0};
        s.analyse_logscale();
        s.stop.maxeval = 20;
        return s;
    };
    const auto with_inner = [&](OptimizerSpec s) {
        s.inner = std::make_unique<OptimizerSpec>(nlopt("LN_NEWUOA_BOUND"));
        return s;
    };
    const auto multistart = [](OptimizerSpec s, bool init) {
        s.strategy_type = StrategyType::MULTISTART;
        s.multistart.n_starts = 2;
        if (init) s.optim_init = {{0.2, 0.2}, {0.4, 0.4}};
        return s;
    };
    const auto prestage = [&](OptimizerSpec s, std::size_t topk,
                              OptimizerSpec explore) {
        s.strategy_type = StrategyType::PRESTAGE;
        s.stage_topk = topk;
        s.prestage = std::make_unique<OptimizerSpec>(std::move(explore));
        return s;
    };
    struct Shape {
        std::string what;
        OptimizerSpec root;
        std::string phase;   // what both validators name
    };
    const std::string top = "the STRATEGY-level optimiser";
    const std::string refinement = "the PRESTAGE refinement";
    std::vector<Shape> shapes;
    shapes.push_back({"PLAIN", nlopt("LN_NEWUOA_BOUND"), top});
    shapes.push_back({"MULTISTART without INIT points",
                      multistart(nlopt("LN_NEWUOA_BOUND"), false), top});
    shapes.push_back({"MULTISTART from two INIT points",
                      multistart(nlopt("LN_NEWUOA_BOUND"), true), top});
    shapes.push_back({"the PRESTAGE refinement, TOPK 1",
                      prestage(nlopt("LN_NEWUOA_BOUND"), 1, nlopt("GN_DIRECT")),
                      refinement});
    shapes.push_back({"the PRESTAGE refinement, TOPK 3",
                      prestage(nlopt("LN_NEWUOA_BOUND"), 3, nlopt("GN_DIRECT")),
                      refinement});
    shapes.push_back({"the EXPLORE block",
                      prestage(nlopt("LN_BOBYQA"), 1, nlopt("LN_NEWUOA_BOUND")),
                      "the EXPLORE block"});
    shapes.push_back({"the INNER block under MULTISTART from INIT points",
                      multistart(with_inner(nlopt("LN_AUGLAG")), true),
                      "the INNER block of " + top});
    shapes.push_back({"the INNER block of the TOPK 3 refinement",
                      prestage(with_inner(nlopt("LN_AUGLAG")), 3,
                               nlopt("GN_DIRECT")),
                      "the INNER block of " + refinement});
    shapes.push_back({"the INNER block of the exploration",
                      prestage(nlopt("LN_BOBYQA"), 1,
                               with_inner(nlopt("G_MLSL_LDS"))),
                      "the INNER block of the EXPLORE block"});
    for (const Shape& c : shapes) {
        INFO(c.what);
        const std::string policy =
            error_of([&] { tadah::hpo::validate_run_policy(c.root); });
        CHECK_THAT(policy, Catch::StartsWith(c.phase + ": " + refused));
        CHECK(error_of([&] { tadah::hpo::validate_phase_backends(c.root); }) ==
              policy);
    }
}

TEST_CASE("strategy rules: reject_refused_backends refuses the first refused "
          "block in validate_run_policy's order and looks up no caps",
          "[startup_validation][strategy][backend][refused]")
{
    /* Pins: reject_refused_backends() throws the refusal of the first
     * refused backend in validate_run_policy()'s order -- the EXPLORE block
     * (under PRESTAGE), the STRATEGY-level optimiser, its INNER block, the
     * EXPLORE block's INNER block -- named as validate_run_policy() names
     * it; an unknown LIB:ALGO is refused the same way, with its phase
     * ("the STRATEGY-level optimiser: unknown optimiser NLOPT:NOT_AN_ALGO:
     * ...", require_registered_backend()); it accepts a spec whose backends
     * are known and not refused, an unset LIB:ALGO (require_driver()'s to
     * report), and an EXPLORE block left on a spec whose TYPE is not
     * PRESTAGE (no phase runs it). It looks up no caps (optimizer_caps(),
     * which throws with no phase), so validate_phase_backends() runs it
     * before strategy_phase_specs().
     * Added with the fix, which introduces it. Until unknown backends were
     * refused with their phase, it accepted NLOPT:NOT_AN_ALGO (""), and
     * validate_phase_backends() reported one through strategy_phase_specs()
     * with no phase, where that looks the backend up. */
    const std::string refused = "NLOPT:LN_NEWUOA_BOUND is refused: ";
    const auto nlopt = [](const std::string& algo) {
        OptimizerSpec s;
        s.lib = "NLOPT";
        s.algo = algo;
        return s;
    };
    const auto with_inner = [&](OptimizerSpec s, const std::string& algo) {
        s.inner = std::make_unique<OptimizerSpec>(nlopt(algo));
        return s;
    };
    const auto with_explore = [](OptimizerSpec s, StrategyType type,
                                 OptimizerSpec explore) {
        s.strategy_type = type;
        s.prestage = std::make_unique<OptimizerSpec>(std::move(explore));
        return s;
    };
    const auto error = [](const OptimizerSpec& s) {
        return error_of([&] { tadah::hpo::reject_refused_backends(s); });
    };
    const std::string bad = "LN_NEWUOA_BOUND";
    const std::string top = "the STRATEGY-level optimiser: ";
    const std::string refinement = "the PRESTAGE refinement: ";

    CHECK_THAT(error(nlopt(bad)), Catch::StartsWith(top + refused));
    OptimizerSpec ms = nlopt(bad);
    ms.strategy_type = StrategyType::MULTISTART;
    CHECK_THAT(error(ms), Catch::StartsWith(top + refused));
    CHECK_THAT(error(with_explore(nlopt(bad), StrategyType::PRESTAGE,
                                  nlopt("GN_DIRECT"))),
               Catch::StartsWith(refinement + refused));
    CHECK_THAT(error(with_explore(with_inner(nlopt(bad), bad),
                                  StrategyType::PRESTAGE, nlopt(bad))),
               Catch::StartsWith("the EXPLORE block: " + refused));
    CHECK_THAT(error(with_inner(nlopt(bad), bad)),
               Catch::StartsWith(top + refused));
    CHECK_THAT(error(with_inner(nlopt("LN_AUGLAG"), bad)),
               Catch::StartsWith("the INNER block of " + top + refused));
    CHECK_THAT(error(with_explore(with_inner(nlopt("LN_AUGLAG"), bad),
                                  StrategyType::PRESTAGE,
                                  with_inner(nlopt("G_MLSL_LDS"), bad))),
               Catch::StartsWith("the INNER block of " + refinement +
                                 refused));
    CHECK_THAT(error(with_explore(nlopt("LN_BOBYQA"), StrategyType::PRESTAGE,
                                  with_inner(nlopt("G_MLSL_LDS"), bad))),
               Catch::StartsWith("the INNER block of the EXPLORE block: " +
                                 refused));

    CHECK(error(with_explore(with_inner(nlopt("LN_AUGLAG"), "LN_BOBYQA"),
                             StrategyType::PRESTAGE,
                             with_inner(nlopt("G_MLSL_LDS"), "LN_COBYLA"))) ==
          "");
    const std::string unknown =
        "unknown optimiser NLOPT:NOT_AN_ALGO: LIB NLOPT has no ALGO "
        "NOT_AN_ALGO";
    CHECK_THAT(error(with_inner(nlopt("NOT_AN_ALGO"), "NOT_AN_ALGO")),
               Catch::StartsWith(top + unknown));
    CHECK_THAT(error(with_inner(nlopt("LN_AUGLAG"), "NOT_AN_ALGO")),
               Catch::StartsWith("the INNER block of " + top + unknown));
    CHECK(error(OptimizerSpec{}) == "");
    CHECK(error(with_explore(nlopt("LN_BOBYQA"), StrategyType::PLAIN,
                             nlopt(bad))) == "");
}

TEST_CASE("startup validation: NLOPT:LN_NEWUOA_BOUND is refused by hpo_run "
          "before it loads any data",
          "[startup_validation][strategy][e2e][refused]")
{
    /* Pins: hpo_run() stops at the ENDOPTIMIZER line (line 8 of this
     * config.hpo), before any data is loaded or any evaluation runs, and
     * journals the refusal, a two-line block with the line's text (run_hpo
     * checks both).
     * Before the fix: the run started and NLopt made 5 evaluations (the
     * harness has three OPTIM dimensions, so NEWUOA's 2n+1 = 7 starting
     * interpolation points exceed MAXEVAL 5 and no trust-region step ran)
     * and ended MAXEVAL_REACHED: hpo_run threw nothing, and run_hpo's checks
     * failed (5 evaluations, the FILES, POTENTIAL SETUP and PC_ISOLATION
     * lines in the journal, loss.tadah written, no error). */
    const auto err = run_hpo(
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
        "    ALGO LN_NEWUOA_BOUND\n    MAXEVAL 5\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    CHECK_THAT(err, Catch::StartsWith("Error on line 8: the STRATEGY-level "
                                      "optimiser: NLOPT:LN_NEWUOA_BOUND is "
                                      "refused: ") &&
                    Catch::Contains("Use NLOPT:LN_BOBYQA"));
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
 * In a build without Ceres a CERES row is refused as not built, for its
 * phase (ceres_build.h): a pin of that refusal. Before this test change
 * such a build failed the row, expecting the Ceres validator's message.
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
              hpo_test::ceres_expect(
                  "the PRESTAGE exploration (EXPLORE block): CERES:LBFGS "
                  "does not support STOPVAL",
                  "the PRESTAGE exploration (EXPLORE block): ")},
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

namespace {

/* The first line of `config` whose text, without its indentation, is
 * `keyword` (1-based), and that line as written: the line a phase's
 * startup refusal names (ENDOPTIMIZER, ENDEXPLORE, ENDINNER). */
std::pair<std::size_t, std::string> line_of(const std::string& config,
                                            const std::string& keyword)
{
    std::istringstream in(config);
    std::string text;
    for (std::size_t n = 1; std::getline(in, text); ++n) {
        const auto p = text.find_first_not_of(' ');
        if (p != std::string::npos && text.substr(p) == keyword)
            return {n, text};
    }
    return {0, ""};
}

}  // namespace

/* The caps registry's refusal of a backend comes from the parser where its
 * block closes, "Error on line N: <phase>: ...\nLine: ENDOPTIMIZER"; the
 * refusals of a phase's settings at startup -- each backend's own checks
 * (validate_phase_backends: the CERES validators, DLIB:BFGS, TADAH:ANNEAL,
 * NLopt's, a backend that needs a stop), the per-phase checks of
 * validate_hpo_setup (GRAD_STRATEGY on a phase that takes no gradient, the
 * finite differences) and the STRATEGY TYPE and INNER rules
 * (validate_run_policy) -- named the phase and no line.
 * Pins: each names the line read() checks the phase's optimiser at, in the
 * parser's form ("Error on line N: <phase>: <message>" and "\nLine:
 * <text>"): ENDOPTIMIZER for the STRATEGY-level optimiser, a MULTISTART
 * start and the PRESTAGE refinement (a REFINE block's included),
 * ENDEXPLORE for the exploration, ENDINNER for an INNER block, the short
 * form's ENDOPTIMIZER; a STRATEGY TYPE rule the ENDOPTIMIZER line, a block
 * that needs an INNER block its own closing line; a spec built outside the
 * parser has no line, and its refusal is the phase's message alone. In a
 * build without Ceres the two CERES rows are refused as not built, at the
 * same lines (ceres_build.h; a pin of that refusal: before this test
 * change such a build failed them, expecting the Ceres validators').
 * Before the fix: "<phase>: <message>" with no line for every case. */
TEST_CASE("startup validation: a phase's settings refused at startup name the "
          "line that closes its block",
          "[startup_validation][strategy][backend]")
{
    struct Case {
        std::string what, config, closing, message;
    };
    const std::string top = "the STRATEGY-level optimiser: ";
    const std::vector<Case> cases = {
        {"CERES:LBFGS FTOL_REL -1",
         strategy_config("    TYPE PLAIN\n    LIB CERES\n    ALGO LBFGS\n"
                         "    MAXEVAL 20\n    FTOL_REL -1\n"),
         "ENDOPTIMIZER",
         top + hpo_test::ceres_expect("CERES:LBFGS requires FTOL_REL >= 0 "
                                      "(Ceres' function_tolerance), got -1")},
        {"an EXPLORE block's CERES:WOLFE PARAM max_iter 1.5",
         strategy_config("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                         "    MAXEVAL 20\n    EXPLORE\n      LIB CERES\n"
                         "      ALGO WOLFE\n      MAXEVAL 12\n"
                         "      PARAM max_iter 1.5\n    ENDEXPLORE\n"),
         "ENDEXPLORE",
         "the PRESTAGE exploration (EXPLORE block): " +
             hpo_test::ceres_expect(
                 "CERES:WOLFE requires PARAM max_iter to be an integer from "
                 "0 to 2147483647 (Ceres' max_num_iterations), got 1.5")},
        {"a REFINE block's DLIB:BFGS MAXTIME",
         strategy_config("    TYPE PRESTAGE\n" + kExploreGfs +
                         "    REFINE\n      LIB DLIB\n      ALGO BFGS\n"
                         "      MAXTIME 60\n    ENDREFINE\n"),
         "ENDOPTIMIZER",
         "the PRESTAGE refinement: DLIB:BFGS does not support MAXTIME"},
        {"each MULTISTART start: DLIB:BFGS SEED",
         strategy_config(multistart_body("DLIB", "BFGS", "2", "    SEED 3\n")),
         "ENDOPTIMIZER",
         "each MULTISTART start: DLIB:BFGS does not support SEED"},
        {"TADAH:ANNEAL with an unknown PARAM",
         strategy_config("    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n"
                         "    MAXEVAL 20\n    PARAM NOT_A_KEY 1\n"),
         "ENDOPTIMIZER",
         top + "TADAH:ANNEAL: unknown PARAM 'NOT_A_KEY'"},
        {"TADAH:RANDOM without a stop",
         strategy_config("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"),
         "ENDOPTIMIZER", top + "TADAH:RANDOM needs a stop"},
        {"an INNER block's LN_BOBYQA POPULATION",
         strategy_config("    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
                         "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
                         "      ALGO LN_BOBYQA\n      POPULATION 10\n"
                         "    ENDINNER\n"),
         "ENDINNER",
         "the INNER block of the STRATEGY-level optimiser: NLOPT:LN_BOBYQA "
         "does not use POPULATION"},
        {"GRAD_STRATEGY on a derivative-free search",
         strategy_config("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                         "    MAXEVAL 20\n    GRAD_STRATEGY NUMERICAL\n"),
         "ENDOPTIMIZER",
         "the STRATEGY-level optimiser NLOPT:LN_SBPLX: GRAD_STRATEGY never "
         "takes effect: NLOPT:LN_SBPLX takes no gradient"},
        {"the short form inside OPTIMIZER",
         kRoot + "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 20\nMAXTIME 60\n"
                 "ENDOPTIMIZER\n",
         "ENDOPTIMIZER", top + "DLIB:BFGS does not support MAXTIME"},
        {"STRATEGY TYPE MULTISTART with a global optimiser",
         strategy_config(multistart_body("NLOPT", "GN_DIRECT", "2")),
         "ENDOPTIMIZER",
         "STRATEGY TYPE MULTISTART cannot run NLOPT:GN_DIRECT"},
        {"an INNER block that names DLIB:BFGS",
         strategy_config("    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
                         "    MAXEVAL 60\n    INNER\n      LIB DLIB\n"
                         "      ALGO BFGS\n    ENDINNER\n"),
         "ENDINNER",
         "the INNER block of the STRATEGY-level optimiser names DLIB:BFGS"},
        {"an EXPLORE block's AUGLAG without an INNER block",
         strategy_config("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                         "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                         "      ALGO AUGLAG\n      MAXEVAL 12\n"
                         "    ENDEXPLORE\n"),
         "ENDEXPLORE", "NLOPT:AUGLAG needs an INNER block"}};
    for (const auto& c : cases) {
        INFO(c.what << ", config:\n" << c.config);
        const auto at = line_of(c.config, c.closing);
        REQUIRE(at.first > 0);
        CHECK_THAT(judge(c.config).startup_raw,
                   Catch::StartsWith("Error on line " +
                                     std::to_string(at.first) + ": " +
                                     c.message) &&
                   Catch::EndsWith("\nLine: " + at.second));
    }

    /* A spec built outside the parser: no line. */
    OptimizerSpec hand;
    hand.lib = "DLIB";
    hand.algo = "BFGS";
    hand.low = {0.0};
    hand.high = {1.0};
    hand.analyse_logscale();
    hand.stop.maxeval = 20;
    hand.stop.maxtime = 60.0;
    CHECK(error_of([&] { tadah::hpo::validate_phase_backends(hand); }) ==
          top + "DLIB:BFGS does not support MAXTIME");
}

/* The startup refusals of a phase's settings that still named no line:
 * the check of the GRAD_STRATEGY ANALYTICAL setup (validate_hpo_setup:
 * "OPTIMIZER block: ...", "EXPLORE block: ..."), the rule that NLOPT, DLIB
 * and CERES cannot search a space without an OPTIM line, and the
 * multi-rank dispatch refusal of a parsed spec (hpo_mpi_dispatch_refusal).
 * Pins: each is "Error on line N: <what>\nLine: <text>", N the line that
 * closes its block -- ENDOPTIMIZER for the STRATEGY-level block (each
 * MULTISTART start, the PRESTAGE refinement), ENDEXPLORE for the EXPLORE
 * block; the zero-dimension rule applies to every phase, the EXPLORE
 * block's included, and names it with its backend ("the PRESTAGE
 * exploration (EXPLORE block): NLOPT:GN_DIRECT cannot optimise a
 * zero-dimensional search space ..."); TADAH:RANDOM without an OPTIM line
 * passes startup (a pin: it evaluates its INIT point).
 * Before the fix: the ANALYTICAL refusals ("OPTIMIZER block: GRAD_STRATEGY
 * ANALYTICAL ...") and the multi-rank refusal named no line; the
 * zero-dimension rule was checked for the STRATEGY-level LIB only, "No
 * OPTIM line: the search space has zero dimensions, and LIB NLOPT cannot
 * optimise a zero-dimensional space ..." without a line, so an EXPLORE
 * NLOPT:GN_DIRECT or DLIB:GFS passed startup (the run failed in the
 * optimiser: "NULL args to nlopt_optimize_", dlib's "The vectors can't be
 * empty") and an EXPLORE DLIB:BFGS was refused as "... DLIB:BFGS requires
 * non-empty, same-sized low/high bounds". */
TEST_CASE("startup validation: the GRAD_STRATEGY ANALYTICAL check, the "
          "zero-dimension rule and the multi-rank refusal name the line that "
          "closes their block", "[startup_validation][strategy][block_line]")
{
    struct Case {
        std::string what, config, closing, message;
    };
    const std::string analytical =
        "GRAD_STRATEGY ANALYTICAL needs the analytical gradient of the loss, "
        "which this setup cannot form: ";
    const std::string zero_dim =
        " cannot optimise a zero-dimensional search space (there is no "
        "OPTIM line";
    /* No OPTIM line: a zero-dimensional search space. */
    const auto no_optim = [](const std::string& body) {
        return "PC_ERMSE 0 1\nOPTIMIZER\n  STRATEGY\n" + body +
               "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const auto explore = [](const std::string& lib, const std::string& algo) {
        return "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
               "    MAXEVAL 2\n    EXPLORE\n      LIB " + lib +
               "\n      ALGO " + algo + "\n      MAXEVAL 3\n    ENDEXPLORE\n";
    };
    const std::vector<Case> cases = {
        {"GRAD_STRATEGY ANALYTICAL on the STRATEGY-level DLIB:BFGS",
         strategy_config("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                         "    MAXEVAL 20\n    GRAD_STRATEGY ANALYTICAL\n"),
         "ENDOPTIMIZER", "OPTIMIZER block: " + analytical},
        {"GRAD_STRATEGY ANALYTICAL on the EXPLORE block's NLOPT:LD_LBFGS",
         strategy_config("    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                         "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                         "      ALGO LD_LBFGS\n      MAXEVAL 12\n"
                         "      GRAD_STRATEGY ANALYTICAL\n    ENDEXPLORE\n"),
         "ENDEXPLORE", "EXPLORE block: " + analytical},
        {"PLAIN NLOPT:LN_SBPLX without an OPTIM line",
         no_optim("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                  "    MAXEVAL 20\n"),
         "ENDOPTIMIZER",
         "the STRATEGY-level optimiser: NLOPT:LN_SBPLX" + zero_dim},
        {"PLAIN DLIB:BFGS without an OPTIM line",
         no_optim("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                  "    MAXEVAL 20\n"),
         "ENDOPTIMIZER", "the STRATEGY-level optimiser: DLIB:BFGS" + zero_dim},
        {"each MULTISTART start's NLOPT:LN_SBPLX without an OPTIM line",
         no_optim(multistart_body("NLOPT", "LN_SBPLX", "2")),
         "ENDOPTIMIZER", "each MULTISTART start: NLOPT:LN_SBPLX" + zero_dim},
        {"the PRESTAGE refinement's NLOPT:LN_SBPLX without an OPTIM line",
         no_optim("    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                  "    MAXEVAL 20\n    EXPLORE\n      LIB TADAH\n"
                  "      ALGO RANDOM\n      MAXEVAL 3\n    ENDEXPLORE\n"),
         "ENDOPTIMIZER", "the PRESTAGE refinement: NLOPT:LN_SBPLX" + zero_dim},
        {"an EXPLORE block's NLOPT:GN_DIRECT without an OPTIM line",
         no_optim(explore("NLOPT", "GN_DIRECT")), "ENDEXPLORE",
         "the PRESTAGE exploration (EXPLORE block): NLOPT:GN_DIRECT" + zero_dim},
        {"an EXPLORE block's DLIB:GFS without an OPTIM line",
         no_optim(explore("DLIB", "GFS")), "ENDEXPLORE",
         "the PRESTAGE exploration (EXPLORE block): DLIB:GFS" + zero_dim},
        {"an EXPLORE block's DLIB:BFGS without an OPTIM line",
         no_optim(explore("DLIB", "BFGS")), "ENDEXPLORE",
         "the PRESTAGE exploration (EXPLORE block): DLIB:BFGS" + zero_dim}};
    /* A section each: judge() stops a case whose refusal names no line. */
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.what) {
            INFO("config:\n" << c.config);
            const auto at = line_of(c.config, c.closing);
            REQUIRE(at.first > 0);
            CHECK_THAT(judge(c.config).startup_raw,
                       Catch::StartsWith("Error on line " +
                                         std::to_string(at.first) + ": " +
                                         c.message) &&
                       Catch::EndsWith("\nLine: " + at.second));
        }
    }
    /* A TADAH-native optimiser evaluates its INIT point. */
    CHECK(judge(no_optim("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                         "    MAXEVAL 2\n")).startup_raw == "");

    SECTION("the multi-rank dispatch refusal of a parsed spec") {
        PeriodicTable::initialize();
        for (const auto& c : std::vector<Case>{
                 {"TADAH:RANDOM on 2 ranks",
                  strategy_config("    TYPE PLAIN\n    LIB TADAH\n"
                                  "    ALGO RANDOM\n    MAXEVAL 2\n"),
                  "ENDOPTIMIZER",
                  "multi-rank HPO supports only STRATEGY TYPE PLAIN with LIB "
                  "TADAH and ALGO ANNEAL_PT or ANNEAL_PA; got TADAH:RANDOM "
                  "on 2 ranks"},
                 {"TADAH:ANNEAL_PT with RESTART on 2 ranks",
                  strategy_config("    TYPE PLAIN\n    LIB TADAH\n"
                                  "    ALGO ANNEAL_PT\n    MAXEVAL 20\n"
                                  "    RESTART\n      AFTER_STAGNATION 10\n"
                                  "      MAX 1\n    ENDRESTART\n"),
                  "ENDOPTIMIZER",
                  "multi-rank HPO does not support RESTART (AFTER_STAGNATION "
                  "10, MAX 1 on 2 ranks)"}}) {
            INFO(c.what << ", config:\n" << c.config);
            ScopedTempFile tf(c.config);
            Context ctx = make_minimal_ctx();
            std::unique_ptr<HPOTargets> trg;
            REQUIRE(error_of([&] {
                trg = std::make_unique<HPOTargets>(ctx, tf.path.string());
            }) == "");
            const auto at = line_of(c.config, c.closing);
            REQUIRE(at.first > 0);
            CHECK_THAT(tadah::hpo::hpo_mpi_dispatch_refusal(2, trg->spec),
                       Catch::StartsWith("Error on line " +
                                         std::to_string(at.first) + ": " +
                                         c.message) &&
                       Catch::EndsWith("\nLine: " + at.second));
            CHECK(tadah::hpo::hpo_mpi_dispatch_refusal(1, trg->spec).empty());
        }
    }
}

/* The STEP rules (scale_validation.h: run_scale_validation, applied by the
 * parser at the end of the file and again at startup) refuse a phase's
 * STEP setting: a backend that does not use STEP, a value outside (0,
 * 0.5], overrides that cover only some dimensions.
 * Pins: each refusal is "Error on line N: <what>\nLine: <text>", N the
 * line that closes the block whose STEP it refuses: ENDOPTIMIZER for the
 * STRATEGY-level block, ENDEXPLORE for the EXPLORE block, ENDINNER for an
 * INNER block (the STRATEGY-level one's and the EXPLORE block's).
 * Before the fix: the same text without a line ("DLIB:BFGS (OUTER) does
 * not use STEP; ..."). */
TEST_CASE("startup validation: the STEP rules name the line that closes the "
          "block whose STEP they refuse",
          "[startup_validation][strategy][block_line][step]")
{
    struct Case {
        std::string what, config, closing, message;
    };
    const std::string two_dims =
        "OPTIM WATOM (1) 0.1 5.0\nOPTIM RCUT2B (1) 4.0 6.0\nPC_ERMSE 0 1\n";
    const std::vector<Case> cases = {
        {"STEP on the STRATEGY-level DLIB:BFGS",
         strategy_config("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                         "    MAXEVAL 20\n    STEP 0.1\n"),
         "ENDOPTIMIZER", "DLIB:BFGS (OUTER) does not use STEP"},
        {"STEP 0.9 on the STRATEGY-level NLOPT:LN_SBPLX",
         strategy_config("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                         "    MAXEVAL 20\n    STEP 0.9\n"),
         "ENDOPTIMIZER",
         "NLOPT:LN_SBPLX (OUTER): STEP must be a fraction of the OPTIM width "
         "in (0, 0.5] (got 0.9)"},
        {"a STEP override that covers one of two dimensions",
         two_dims + "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
                    "    ALGO LN_SBPLX\n    MAXEVAL 20\n    STEP WATOM 0.1\n"
                    "  ENDSTRATEGY\nENDOPTIMIZER\n",
         "ENDOPTIMIZER",
         "STEP overrides cover only some dimensions and no block-wide STEP "
         "is set"},
        {"STEP on the EXPLORE block's NLOPT:GN_CRS2_LM",
         strategy_config("    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                         "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                         "      ALGO GN_CRS2_LM\n      MAXEVAL 12\n"
                         "      STEP 0.1\n    ENDEXPLORE\n"),
         "ENDEXPLORE", "NLOPT:GN_CRS2_LM (PRESTAGE) does not use STEP"},
        {"STEP on the STRATEGY-level INNER block's NLOPT:LD_LBFGS",
         strategy_config("    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
                         "    MAXEVAL 60\n    INNER\n      LIB NLOPT\n"
                         "      ALGO LD_LBFGS\n      STEP 0.1\n"
                         "    ENDINNER\n"),
         "ENDINNER", "NLOPT:LD_LBFGS (INNER) does not use STEP"},
        {"STEP on the EXPLORE block's INNER block's NLOPT:LD_LBFGS",
         strategy_config("    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                         "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                         "      ALGO G_MLSL_LDS\n      MAXEVAL 60\n"
                         "      INNER\n        LIB NLOPT\n"
                         "        ALGO LD_LBFGS\n        STEP 0.1\n"
                         "      ENDINNER\n    ENDEXPLORE\n"),
         "ENDINNER", "NLOPT:LD_LBFGS (PRESTAGE.INNER) does not use STEP"}};
    for (const auto& c : cases) {
        INFO(c.what << ", config:\n" << c.config);
        const auto at = line_of(c.config, c.closing);
        REQUIRE(at.first > 0);
        CHECK_THAT(parse_error(c.config),
                   Catch::StartsWith("Error on line " +
                                     std::to_string(at.first) + ": " +
                                     c.message) &&
                   Catch::EndsWith("\nLine: " + at.second));
    }
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
    /* CERES:LBFGS ends on its own as well. A build without Ceres refuses
     * every LIB CERES as not built (ceres_build.h): a pin of that refusal,
     * for the phase. Before this test change: such a build failed the case
     * here, expecting no refusal. */
    CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB CERES\n"
                                "    ALGO LBFGS\n")).startup ==
          hpo_test::ceres_expect("", "the STRATEGY-level optimiser: "));
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
/* GD_STOGO ignores NLopt's forced stop: once HPO has to end it, only
 * NLopt's own MAXEVAL ends it promptly (MAXTIME after the rest of its
 * time). */
const std::string kStogoNote =
    "NLopt's StoGO has no convergence test of its own and ignores NLopt's "
    "forced stop, by which HPO ends a search (on a signal, an error, the "
    "RESTART stagnation stop, or a loss at or below STOPVAL that NLopt was "
    "not given): it then calls on, each call refused, and only NLopt's own "
    "evaluation count ends it promptly (MAXTIME only once the rest of its "
    "time has passed)";
const std::string kStogoStops = "; set MAXEVAL (it honours no tolerance)";
/* DIRECT tests XTOL on every rectangle it divides, the largest included:
 * an XTOL ends it, but only once the whole box is divided that finely. */
const std::string kDirectXtol =
    "; DIRECT's XTOL_REL and XTOL_ABS end it only once every rectangle is "
    "that small, after roughly (1/XTOL_REL)^n evaluations in n dimensions";
/* The backends that read POPULATION and VECTOR_ARR, as the refusal of a
 * backend that does not lists them (sorted): ANNEAL_PA's walkers, and
 * NLopt's stochastic population (POP() in nlopt 2.10's optimize.c: CRS2,
 * ISRES, ESCH, the six MLSL forms, GD_STOGO_RAND) and the Luksan
 * algorithms' vector storage (LD_LBFGS, LD_VAR1/2, LD_TNEWTON*). */
const std::string kPopulationReaders =
    "NLOPT:GD_MLSL, NLOPT:GD_MLSL_LDS, NLOPT:GD_STOGO_RAND, NLOPT:GN_CRS2_LM, "
    "NLOPT:GN_ESCH, NLOPT:GN_ISRES, NLOPT:GN_MLSL, NLOPT:GN_MLSL_LDS, "
    "NLOPT:G_MLSL, NLOPT:G_MLSL_LDS, TADAH:ANNEAL_PA";
const std::string kVectorStorageReaders =
    "NLOPT:LD_LBFGS, NLOPT:LD_TNEWTON, NLOPT:LD_TNEWTON_PRECOND, "
    "NLOPT:LD_TNEWTON_PRECOND_RESTART, NLOPT:LD_TNEWTON_RESTART, "
    "NLOPT:LD_VAR1, NLOPT:LD_VAR2";

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
    for (const std::string algo : {"GN_ESCH", "GN_AGS"})
        cases.push_back({"NLOPT:" + algo + " without a stop", nlopt(algo, ""),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    /* Pins: GD_STOGO and GD_STOGO_RAND ignore NLopt's forced stop, by
     * which HPO ends a search; once HPO has to end one (a signal, an error,
     * the RESTART stagnation stop, a loss at or below STOPVAL that NLopt was
     * not given), the objective refuses every call and only NLopt's own
     * MAXEVAL ends it promptly. Neither STOPVAL, MAXTIME (the run waits out
     * the rest of it) nor an active RESTART is enough, and the message names
     * MAXEVAL only, as does the INNER search's under AUGLAG.
     * Before the fix: MAXTIME alone was accepted as its stop, in its own
     * block and in either block of AUGLAG and its INNER search, and the
     * messages read "; set MAXEVAL or MAXTIME (it honours no tolerance)" and
     * "; set MAXEVAL or MAXTIME in the INNER block, or MAXEVAL or MAXTIME in
     * the NLOPT:AUGLAG block, ..."; such a run, once HPO stopped it, waited
     * out the rest of MAXTIME at 100% CPU. */
    const std::string stogo_inner =
        "; set MAXEVAL in the INNER block, or MAXEVAL in the NLOPT:AUGLAG "
        "block, which NLopt hands on to it";
    for (const std::string algo : {"GD_STOGO", "GD_STOGO_RAND"}) {
        cases.push_back({"NLOPT:" + algo + " without a stop", nlopt(algo, ""),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
        cases.push_back({"NLOPT:" + algo + " with STOPVAL only",
                         nlopt(algo, "    STOPVAL 0.1\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
        cases.push_back({"NLOPT:" + algo + " with MAXTIME only",
                         nlopt(algo, "    MAXTIME 60\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
        cases.push_back({"NLOPT:" + algo + " with MAXTIME and STOPVAL",
                         nlopt(algo, "    MAXTIME 60\n    STOPVAL 0.1\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
        cases.push_back({"NLOPT:" + algo + " under an active RESTART",
                         nlopt(algo, kRestart),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
        cases.push_back({"NLOPT:" + algo + " with MAXTIME under an active "
                         "RESTART",
                         nlopt(algo, "    MAXTIME 60\n" + kRestart),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kStogoStops});
    }
    cases.push_back({"NLOPT:AUGLAG with an INNER NLOPT:GD_STOGO and STOPVAL",
                     nlopt("AUGLAG", "    STOPVAL 0.1\n    INNER\n"
                                     "      LIB NLOPT\n      ALGO GD_STOGO\n"
                                     "    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG (NLOPT:GD_STOGO) "
                           "needs a stop: ",
                     stogo_inner});
    cases.push_back({"NLOPT:AUGLAG with MAXTIME and an INNER NLOPT:GD_STOGO",
                     nlopt("AUGLAG", "    MAXTIME 60\n    INNER\n"
                                     "      LIB NLOPT\n      ALGO GD_STOGO\n"
                                     "    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG (NLOPT:GD_STOGO) "
                           "needs a stop: ",
                     stogo_inner});
    cases.push_back({"NLOPT:AUGLAG with an INNER NLOPT:GD_STOGO with MAXTIME",
                     nlopt("AUGLAG", "    INNER\n"
                                     "      LIB NLOPT\n      ALGO GD_STOGO\n"
                                     "      MAXTIME 60\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG (NLOPT:GD_STOGO) "
                           "needs a stop: ",
                     stogo_inner});
    /* MLSL hands its tolerances to its local searches only: without an
     * INNER block to NLopt's default local optimiser. */
    for (const std::string algo :
         {"GN_MLSL", "GN_MLSL_LDS", "GD_MLSL", "GD_MLSL_LDS"})
        cases.push_back({"NLOPT:" + algo + " with XTOL_REL",
                         nlopt(algo, "    XTOL_REL 1e-4\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    /* The INNER block's MAXEVAL ends each local search, not the MLSL
     * loop. The block has no tolerance: one on an MLSL block with an
     * INNER block is refused for itself (the Exact cases below), so these
     * cases dropped their FTOL_REL 1e-6, which was no stop either. Before
     * the fix: passes as well (a pin of the MLSL loop's stop, with or
     * without that FTOL_REL). */
    for (const std::string algo : {"G_MLSL", "G_MLSL_LDS"})
        cases.push_back({"NLOPT:" + algo + " with an INNER block",
                         nlopt(algo, "    INNER\n"
                                     "      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                                     "      MAXEVAL 5\n    ENDINNER\n"),
                         top + "NLOPT:" + algo + " needs a stop: ",
                         kNoTolerance});
    cases.push_back({"TADAH:ANNEAL_PT without a stop",
                     plain + "    LIB TADAH\n    ALGO ANNEAL_PT\n",
                     top + "TADAH:ANNEAL_PT needs a stop: ",
                     "; set MAXEVAL, MAXTIME, STOPVAL or a positive tolerance "
                     "it honours (FTOL_REL, FTOL_ABS)"});
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
    cases.push_back({"NLOPT:AUGLAG with an INNER XTOL_REL of 0",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_DIRECT_L\n"
                                     "      XTOL_REL 0\n    ENDINNER\n"),
                     top + "the INNER search of NLOPT:AUGLAG "
                           "(NLOPT:GN_DIRECT_L) needs a stop: ",
                     auglag_direct});
    /* An INNER search that tests no tolerance takes MAXEVAL or MAXTIME
     * only. Pins: this message without the tolerances the case below
     * wrote. Before the fix: passes as well (a pin of the INNER search's
     * stop, which the tolerances no longer reach). */
    cases.push_back({"NLOPT:AUGLAG with an INNER NLOPT:GN_ESCH, no stop",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_ESCH\n    ENDINNER\n"),
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

    /* Pins: a tolerance the backend never tests is refused for itself, with
     * the stops it honours, and an INNER block's STOPVAL at its line (the
     * test case "NLopt rejects a tolerance, POPULATION or VECTOR_ARR its
     * algorithm never reads" below has every backend).
     * Before the fix: each was accepted, so it counted as no stop: the
     * first four were refused as "NLOPT:GN_ESCH needs a stop: ...",
     * "NLOPT:G_MLSL needs a stop: ...", "TADAH:RANDOM needs a stop: ..."
     * and "the INNER search of NLOPT:AUGLAG (NLOPT:GN_ESCH) needs a stop:
     * ...", and the INNER STOPVAL parsed, replaced by the enclosing
     * block's, and the run was refused as "the INNER search of NLOPT:AUGLAG
     * (NLOPT:GN_DIRECT_L) needs a stop: ...". */
    struct Exact { std::string what, body, message; };
    const std::string mlsl_inner =
        " does not support FTOL_REL with an INNER block: MLSL's multistart "
        "loop tests no tolerance, and NLopt passes the block's tolerances to "
        "no INNER search; set it in the INNER block, whose local search "
        "tests it";
    for (const auto& c : std::vector<Exact>{
             {"NLOPT:GN_ESCH with FTOL_REL",
              nlopt("GN_ESCH", "    FTOL_REL 1e-6\n"),
              top + "NLOPT:GN_ESCH does not support FTOL_REL: NLopt's global "
                    "search tests no tolerance; stop it with MAXEVAL, MAXTIME "
                    "or STOPVAL"},
             {"NLOPT:G_MLSL with an INNER block and FTOL_REL",
              nlopt("G_MLSL", "    FTOL_REL 1e-6\n    INNER\n"
                              "      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                              "      MAXEVAL 5\n    ENDINNER\n"),
              top + "NLOPT:G_MLSL" + mlsl_inner},
             {"NLOPT:G_MLSL_LDS with an INNER block and FTOL_REL",
              nlopt("G_MLSL_LDS", "    FTOL_REL 1e-6\n    INNER\n"
                                  "      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                                  "      MAXEVAL 5\n    ENDINNER\n"),
              top + "NLOPT:G_MLSL_LDS" + mlsl_inner},
             {"TADAH:RANDOM with a tolerance only",
              plain + "    LIB TADAH\n    ALGO RANDOM\n    FTOL_REL 1e-6\n",
              top + "TADAH:RANDOM does not support FTOL_REL: uniform random "
                    "search has no convergence test to apply it to; stop it "
                    "with MAXEVAL, MAXTIME or STOPVAL"},
             /* Pins: where the INNER search does not test the tolerance
              * either, the message names the stops that search honours
              * instead of sending the tolerance into the INNER block,
              * which would refuse it there.
              * Before the review fix: "...; set it in the INNER block,
              * whose local search tests it" for every INNER algorithm
              * (the case wrote FTOL_REL in its INNER GN_ESCH block too,
              * which is refused there; that block is the case "INNER
              * GN_ESCH FTOL_REL under AUGLAG" of the test case below). */
             {"NLOPT:AUGLAG with an INNER NLOPT:GN_ESCH and a tolerance",
              nlopt("AUGLAG", "    FTOL_REL 1e-6\n    INNER\n"
                              "      LIB NLOPT\n      ALGO GN_ESCH\n"
                              "      MAXEVAL 5\n    ENDINNER\n"),
              top + "NLOPT:AUGLAG does not support FTOL_REL with an INNER "
                    "block: AUGLAG ends when its INNER search ends (HPO sets "
                    "no constraint) and tests the block's tolerances only to "
                    "name that ending, and NLopt passes them to no INNER "
                    "search; the INNER block's NLOPT:GN_ESCH does not test "
                    "it either: stop that search with MAXEVAL or MAXTIME in "
                    "the INNER block, or MAXEVAL, MAXTIME or STOPVAL in the "
                    "NLOPT:AUGLAG block, which NLopt hands on to it"},
             /* GD_STOGO ignores NLopt's forced stop: only MAXEVAL ends it,
              * in either block (backend rules: GD_STOGO needs MAXEVAL).
              * Before the merge with that rule: "...; stop that search with
              * MAXEVAL or MAXTIME in the INNER block, or MAXEVAL, MAXTIME or
              * STOPVAL in the NLOPT:GN_MLSL_LDS block, ...". */
             {"NLOPT:GN_MLSL_LDS with XTOL_ABS and an INNER NLOPT:GD_STOGO",
              nlopt("GN_MLSL_LDS", "    MAXEVAL 20\n    XTOL_ABS 1e-4\n"
                                   "    INNER\n      LIB NLOPT\n"
                                   "      ALGO GD_STOGO\n      MAXEVAL 5\n"
                                   "    ENDINNER\n"),
              top + "NLOPT:GN_MLSL_LDS does not support XTOL_ABS with an "
                    "INNER block: MLSL's multistart loop tests no tolerance, "
                    "and NLopt passes the block's tolerances to no INNER "
                    "search; the INNER block's NLOPT:GD_STOGO does not test "
                    "it either: stop that search with MAXEVAL in the INNER "
                    "block, or MAXEVAL in the NLOPT:GN_MLSL_LDS block, which "
                    "NLopt hands on to it"},
             /* The original DIRECT tests XTOL_REL only: FTOL_REL is not
              * sent into its INNER block, XTOL_REL is (the second case
              * passes before the review fix as well: a pin of the advice
              * the fix keeps). */
             {"NLOPT:GN_MLSL_LDS with FTOL_REL and an INNER "
              "NLOPT:GN_ORIG_DIRECT",
              nlopt("GN_MLSL_LDS", "    MAXEVAL 20\n    FTOL_REL 1e-4\n"
                                   "    INNER\n      LIB NLOPT\n"
                                   "      ALGO GN_ORIG_DIRECT\n"
                                   "      MAXEVAL 5\n    ENDINNER\n"),
              top + "NLOPT:GN_MLSL_LDS does not support FTOL_REL with an "
                    "INNER block: MLSL's multistart loop tests no tolerance, "
                    "and NLopt passes the block's tolerances to no INNER "
                    "search; the INNER block's NLOPT:GN_ORIG_DIRECT does not "
                    "test it either: stop that search with MAXEVAL, MAXTIME "
                    "or a positive tolerance it honours (XTOL_REL) in the "
                    "INNER block, or MAXEVAL, MAXTIME or STOPVAL in the "
                    "NLOPT:GN_MLSL_LDS block, which NLopt hands on to it"},
             {"NLOPT:GN_MLSL_LDS with XTOL_REL and an INNER "
              "NLOPT:GN_ORIG_DIRECT",
              nlopt("GN_MLSL_LDS", "    MAXEVAL 20\n    XTOL_REL 1e-4\n"
                                   "    INNER\n      LIB NLOPT\n"
                                   "      ALGO GN_ORIG_DIRECT\n"
                                   "      MAXEVAL 5\n    ENDINNER\n"),
              top + "NLOPT:GN_MLSL_LDS does not support XTOL_REL with an "
                    "INNER block: MLSL's multistart loop tests no tolerance, "
                    "and NLopt passes the block's tolerances to no INNER "
                    "search; set it in the INNER block, whose local search "
                    "tests it"}}) {
        INFO(c.what << "\n" << c.body);
        CHECK(judge(strategy_config(c.body)).startup == c.message);
    }
    const std::string inner_stopval =
        nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n      ALGO GN_DIRECT_L\n"
                        "      STOPVAL 0.1\n    ENDINNER\n");
    /* Lines 1-4 are kRoot, OPTIMIZER and STRATEGY; the INNER block's
     * STOPVAL is line 11. */
    CHECK(parse_error(strategy_config(inner_stopval)) ==
          "Error on line 11: STOPVAL is not accepted inside an INNER block: "
          "NLopt's MLSL and AUGLAG replace the INNER search's STOPVAL with the "
          "enclosing block's; set STOPVAL on the enclosing STRATEGY block\n"
          "Line:       STOPVAL 0.1");

    /* The whole message of one NLopt global search. */
    CHECK(judge(strategy_config(nlopt("GN_DIRECT", ""))).startup ==
          top + "NLOPT:GN_DIRECT needs a stop: NLopt's global search has no "
                "convergence test of its own and runs until it is "
                "interrupted" + kAnyTolerance + kDirectXtol);
    /* GD_STOGO's, and the stop that ends it: MAXEVAL, in its own block or
     * (an INNER search) in the enclosing one, with MAXTIME or not. */
    CHECK(judge(strategy_config(nlopt("GD_STOGO", "    STOPVAL 0.1\n")))
              .startup ==
          top + "NLOPT:GD_STOGO needs a stop: " + kStogoNote + kStogoStops);
    CHECK(judge(strategy_config(nlopt("GD_STOGO", "    MAXTIME 60\n")))
              .startup ==
          top + "NLOPT:GD_STOGO needs a stop: " + kStogoNote + kStogoStops);
    for (const std::string& stop :
         {std::string("    MAXEVAL 30\n"),
          std::string("    MAXEVAL 30\n    MAXTIME 60\n"),
          std::string("    MAXEVAL 30\n") + kRestart}) {
        INFO(stop);
        CHECK(judge(strategy_config(nlopt("GD_STOGO", stop))).startup == "");
        CHECK(judge(strategy_config(nlopt("AUGLAG", stop +
                                          "    INNER\n      LIB NLOPT\n"
                                          "      ALGO GD_STOGO\n"
                                          "    ENDINNER\n")))
                  .startup == "");
    }
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
    /* Pins: RANDOM and GRID refuse every tolerance, POPULATION and
     * VECTOR_ARR, GRID also SEED, and GFS POPULATION and VECTOR_ARR -- keys
     * neither run_random, run_grid nor find_min_global reads -- with the
     * stops they honour or the backends that read the key.
     * Before the fix: each was accepted (startup ""), the journal listed it
     * as if it applied (`ftol_rel : 1.0e-03` among RANDOM's stop criteria,
     * `Random seed : 7` for GRID, `Population size : 10` for GFS), and it
     * did nothing. */
    const std::string random_why =
        ": uniform random search has no convergence test to apply it to; "
        "stop it with MAXEVAL, MAXTIME or STOPVAL";
    const std::string grid_why =
        ": the grid search has no convergence test to apply it to; stop it "
        "with MAXEVAL, MAXTIME or STOPVAL";
    const std::string population =
        " does not use POPULATION; remove it (it is read by " +
        kPopulationReaders + ")";
    const std::string vector_storage =
        " does not use VECTOR_ARR; remove it (it is read by " +
        kVectorStorageReaders + ")";
    std::vector<Case> unread;
    for (const std::string key :
         {"FTOL_REL", "FTOL_ABS", "XTOL_REL", "XTOL_ABS"}) {
        unread.push_back({"RANDOM " + key, random + "    " + key + " 1e-3\n",
                          top + "TADAH:RANDOM does not support " + key +
                              random_why});
        unread.push_back({"GRID " + key, grid + "    " + key + " 1e-3\n",
                          top + "TADAH:GRID does not support " + key +
                              grid_why});
    }
    for (const auto& b : std::vector<std::pair<std::string, std::string>>{
             {"TADAH:RANDOM", random}, {"TADAH:GRID", grid},
             {"DLIB:GFS", gfs}}) {
        unread.push_back({b.first + " POPULATION",
                          b.second + "    POPULATION 10\n",
                          top + b.first + population});
        unread.push_back({b.first + " VECTOR_ARR",
                          b.second + "    VECTOR_ARR 5\n",
                          top + b.first + vector_storage});
    }
    unread.push_back({"GRID SEED", grid + "    SEED 7\n",
                      top + "TADAH:GRID does not use SEED: it draws no random "
                            "numbers; remove it"});
    for (const auto& c : unread) {
        INFO(c.what << "\n" << c.body);
        CHECK(judge(strategy_config(c.body)).startup == c.message);
    }
    /* The stops they honour, RANDOM's SEED, and DLIB:GFS's SEED, which
     * find_min_global does not read either: it is reported with a warning
     * when the run starts (existing configurations set it), not refused. */
    for (const std::string& body : std::vector<std::string>{
             random + "    MAXTIME 60\n    STOPVAL 0.1\n",
             grid + "    MAXTIME 60\n    STOPVAL 0.1\n",
             gfs + "    MAXTIME 60\n    STOPVAL 0.1\n    FTOL_ABS 1e-3\n",
             random + "    SEED 7\n", gfs + "    SEED 7\n"}) {
        INFO(body);
        CHECK(judge(strategy_config(body)).startup == "");
    }
}

/* ======================================================================
 * NLopt applies every attribute to nlopt::opt (build_nlopt), and an
 * algorithm ignores those it does not read. A tolerance its search never
 * tests (nlopt 2.10: esch.c and ags.cc test none, StoGO's NLopt-local path
 * is compiled out, direct_optimize() takes only XTOL_REL, mlsl.c tests
 * none and auglag.c tests them only to name an ending its local search
 * decides, and NLopt hands the block's tolerances only to the local
 * optimiser it creates without an INNER block), and POPULATION or
 * VECTOR_ARR on an algorithm without one (optimize.c's POP() and
 * vector_storage) are refused at startup, naming the phase.
 * ==================================================================== */

TEST_CASE("startup validation: NLopt rejects a tolerance, POPULATION or "
          "VECTOR_ARR its algorithm never reads",
          "[startup_validation][strategy][stop]")
{
    /* Pins: the exact message for each algorithm class and key, at the
     * STRATEGY level and in an INNER block; the keys an algorithm reads
     * stay accepted.
     * Before the fix: every rejected configuration was accepted (startup
     * ""), and the key did nothing (the probes ran each pair with and
     * without loose tolerances: byte-identical losses; an AUGLAG tolerance
     * relabelled a MAXEVAL ending XTOL_REACHED). */
    struct Case { std::string what, body, message; };
    const std::string top = "the STRATEGY-level optimiser: ";
    const auto nlopt = [](const std::string& algo, const std::string& extra) {
        return "    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
               "\n    MAXEVAL 20\n" + extra;
    };
    const std::string inner_bobyqa =
        "    INNER\n      LIB NLOPT\n      ALGO LN_BOBYQA\n"
        "      MAXEVAL 5\n    ENDINNER\n";
    const std::vector<std::string> tolerances = {"FTOL_REL", "FTOL_ABS",
                                                 "XTOL_REL", "XTOL_ABS"};
    std::vector<Case> cases;
    for (const std::string algo : {"GN_ESCH", "GN_AGS"})
        for (const auto& key : tolerances)
            cases.push_back({algo + " " + key,
                             nlopt(algo, "    " + key + " 1e-4\n"),
                             top + "NLOPT:" + algo + " does not support " +
                                 key + ": NLopt's global search tests no "
                                 "tolerance; stop it with MAXEVAL, MAXTIME or "
                                 "STOPVAL"});
    /* GD_STOGO and GD_STOGO_RAND ignore NLopt's forced stop, so the stop
     * the message names is MAXEVAL only (backend rules: GD_STOGO needs
     * MAXEVAL). Before the merge with that rule: "...; stop it with
     * MAXEVAL, MAXTIME or STOPVAL", as for GN_ESCH. */
    for (const std::string algo : {"GD_STOGO", "GD_STOGO_RAND"})
        for (const auto& key : tolerances)
            cases.push_back({algo + " " + key,
                             nlopt(algo, "    " + key + " 1e-4\n"),
                             top + "NLOPT:" + algo + " does not support " +
                                 key + ": NLopt's global search tests no "
                                 "tolerance; stop it with MAXEVAL"});
    for (const std::string algo : {"GN_ORIG_DIRECT", "GN_ORIG_DIRECT_L"})
        for (const std::string key : {"FTOL_REL", "FTOL_ABS", "XTOL_ABS"})
            cases.push_back({algo + " " + key,
                             nlopt(algo, "    " + key + " 1e-4\n"),
                             top + "NLOPT:" + algo + " does not support " +
                                 key + ": the original DIRECT tests only "
                                 "XTOL_REL, its volume tolerance; stop it with "
                                 "MAXEVAL, MAXTIME, STOPVAL or a positive "
                                 "tolerance it honours (XTOL_REL)"});
    for (const std::string algo : {"GN_MLSL_LDS", "G_MLSL_LDS"})
        for (const auto& key : tolerances)
            cases.push_back({algo + " " + key + " with an INNER block",
                             nlopt(algo, "    " + key + " 1e-4\n" +
                                             inner_bobyqa),
                             top + "NLOPT:" + algo + " does not support " +
                                 key + " with an INNER block: MLSL's "
                                 "multistart loop tests no tolerance, and "
                                 "NLopt passes the block's tolerances to no "
                                 "INNER search; set it in the INNER block, "
                                 "whose local search tests it"});
    for (const std::string algo : {"AUGLAG", "LN_AUGLAG"})
        for (const auto& key : tolerances)
            cases.push_back({algo + " " + key + " with an INNER block",
                             nlopt(algo, "    " + key + " 1e-4\n" +
                                             inner_bobyqa),
                             top + "NLOPT:" + algo + " does not support " +
                                 key + " with an INNER block: AUGLAG ends "
                                 "when its INNER search ends (HPO sets no "
                                 "constraint) and tests the block's "
                                 "tolerances only to name that ending, and "
                                 "NLopt passes them to no INNER search; set "
                                 "it in the INNER block, whose local search "
                                 "tests it"});
    /* An INNER block's own tolerance, named with the phase; the INNER
     * search's STOPVAL is the enclosing block's. */
    cases.push_back({"INNER GN_ESCH FTOL_REL under AUGLAG",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GN_ESCH\n"
                                     "      FTOL_REL 1e-6\n    ENDINNER\n"),
                     "the INNER block of the STRATEGY-level optimiser: "
                     "NLOPT:GN_ESCH does not support FTOL_REL: NLopt's global "
                     "search tests no tolerance; stop it with MAXEVAL or "
                     "MAXTIME in the INNER block, or STOPVAL in the enclosing "
                     "block"});
    /* Pins: an INNER GD_STOGO's own tolerance names MAXEVAL in the INNER
     * block only: STOPVAL and MAXTIME are no stop of a search that ignores
     * NLopt's forced stop. Before the merge with the GD_STOGO rule: "...;
     * stop it with MAXEVAL or MAXTIME in the INNER block, or STOPVAL in
     * the enclosing block". */
    cases.push_back({"INNER GD_STOGO FTOL_REL under AUGLAG",
                     nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n"
                                     "      ALGO GD_STOGO\n"
                                     "      FTOL_REL 1e-6\n    ENDINNER\n"),
                     "the INNER block of the STRATEGY-level optimiser: "
                     "NLOPT:GD_STOGO does not support FTOL_REL: NLopt's global "
                     "search tests no tolerance; stop it with MAXEVAL in the "
                     "INNER block"});
    const std::string population =
        " does not use POPULATION; remove it (it is read by " +
        kPopulationReaders + ")";
    const std::string vector_storage =
        " does not use VECTOR_ARR; remove it (it is read by " +
        kVectorStorageReaders + ")";
    cases.push_back({"LN_BOBYQA POPULATION",
                     nlopt("LN_BOBYQA", "    POPULATION 10\n"),
                     top + "NLOPT:LN_BOBYQA" + population});
    cases.push_back({"LD_LBFGS POPULATION",
                     nlopt("LD_LBFGS", "    POPULATION 10\n"),
                     top + "NLOPT:LD_LBFGS" + population});
    cases.push_back({"AUGLAG POPULATION",
                     nlopt("AUGLAG", "    POPULATION 10\n" + inner_bobyqa),
                     top + "NLOPT:AUGLAG" + population});
    cases.push_back({"LN_BOBYQA VECTOR_ARR",
                     nlopt("LN_BOBYQA", "    VECTOR_ARR 5\n"),
                     top + "NLOPT:LN_BOBYQA" + vector_storage});
    cases.push_back({"LD_MMA VECTOR_ARR",
                     nlopt("LD_MMA", "    VECTOR_ARR 5\n"),
                     top + "NLOPT:LD_MMA" + vector_storage});
    cases.push_back({"INNER LN_BOBYQA POPULATION under GN_MLSL_LDS",
                     nlopt("GN_MLSL_LDS", "    INNER\n      LIB NLOPT\n"
                                          "      ALGO LN_BOBYQA\n"
                                          "      POPULATION 10\n"
                                          "    ENDINNER\n"),
                     "the INNER block of the STRATEGY-level optimiser: "
                     "NLOPT:LN_BOBYQA" + population});
    cases.push_back({"each MULTISTART start: LN_SBPLX VECTOR_ARR",
                     "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                     "    MAXEVAL 20\n    N_STARTS 2\n    VECTOR_ARR 5\n",
                     "each MULTISTART start: NLOPT:LN_SBPLX" + vector_storage});
    for (const auto& c : cases) {
        INFO(c.what << "\n" << c.body);
        CHECK(judge(strategy_config(c.body)).startup == c.message);
    }

    /* What an algorithm reads stays accepted: the tolerances of DIRECT,
     * CRS2, ISRES and LD_MMA, of MLSL and AUGLAG without an INNER block
     * (NLopt hands them to its default local optimiser: with GN_MLSL_LDS,
     * XTOL_REL 1e-4 and 0.5 gave different trajectories), the original
     * DIRECT's XTOL_REL, POPULATION on a population-based search (example
     * 14's G_MLSL_LDS), VECTOR_ARR on L-BFGS, and SEED on any NLopt
     * algorithm (which ones draw random numbers is not checked). */
    std::vector<std::string> accepted = {
        nlopt("GN_DIRECT_L", "    XTOL_ABS 1e-4\n"),
        nlopt("GN_MLSL_LDS", "    XTOL_REL 1e-4\n"),
        nlopt("GD_MLSL", "    FTOL_REL 1e-6\n    FTOL_ABS 1e-6\n"),
        nlopt("LN_AUGLAG", "    FTOL_REL 1e-6\n"),
        nlopt("LD_AUGLAG", "    XTOL_ABS 1e-4\n"),
        nlopt("GN_ORIG_DIRECT", "    XTOL_REL 1e-4\n"),
        nlopt("GN_CRS2_LM", "    POPULATION 10\n"),
        nlopt("GD_STOGO_RAND", "    POPULATION 10\n"),
        nlopt("GN_ISRES", "    POPULATION 10\n"),
        nlopt("LD_LBFGS", "    VECTOR_ARR 5\n"),
        nlopt("LD_TNEWTON_PRECOND", "    VECTOR_ARR 5\n"),
        nlopt("LN_BOBYQA", "    SEED 3\n"),
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n    MAXEVAL 120\n"
        "    POPULATION 5\n    INNER\n      LIB NLOPT\n      ALGO LN_BOBYQA\n"
        "    ENDINNER\n",
        nlopt("AUGLAG", "    INNER\n      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
                        "      POPULATION 10\n      FTOL_REL 1e-6\n"
                        "    ENDINNER\n")};
    for (const std::string algo : {"GN_CRS2_LM", "GN_ISRES", "LD_MMA"})
        for (const auto& key : tolerances)
            accepted.push_back(nlopt(algo, "    " + key + " 1e-4\n"));
    for (const auto& body : accepted) {
        INFO(body);
        CHECK(judge(strategy_config(body)).startup == "");
    }
}

/* ======================================================================
 * NLopt's AUGLAG, AUGLAG_EQ, G_MLSL and G_MLSL_LDS have no default local
 * optimiser (OptimizerCaps::default_inner): NLopt refuses them without a
 * local optimiser ("local optimizer must be specified for AUGLAG",
 * optimize.c), and an INNER block runs no INNER block of its own.
 * ==================================================================== */

TEST_CASE("startup validation: NLopt's AUGLAG, AUGLAG_EQ, G_MLSL and "
          "G_MLSL_LDS need an INNER block",
          "[startup_validation][strategy][inner]")
{
    /* Pins: each of the four without an INNER block, in every phase, and
     * one of them named in an INNER block, is refused at startup with the
     * phase and the alternatives; the algorithms with a default local
     * optimiser run without one; run_inner()'s safety net refuses a spec
     * built at run time.
     * Before the fix: startup "" for every refused case; NLopt refused the
     * run once it had started (exit 1 with "HPOEngine error: local
     * optimizer must be specified for AUGLAG", after the data were loaded,
     * and under PRESTAGE after the whole exploration), and an INNER
     * G_MLSL_LDS or AUGLAG failed after some evaluations with only "nlopt
     * invalid argument"; require_inner_backend() returned at once without
     * an INNER block. */
    const std::string alternatives =
        "; add INNER ... ENDINNER with LIB NLOPT and a local ALGO such as "
        "LN_BOBYQA, or use one of NLOPT:GD_MLSL, NLOPT:GD_MLSL_LDS, "
        "NLOPT:GN_MLSL, NLOPT:GN_MLSL_LDS, NLOPT:LD_AUGLAG, "
        "NLOPT:LD_AUGLAG_EQ, NLOPT:LN_AUGLAG, NLOPT:LN_AUGLAG_EQ, which run "
        "LD_MMA or LN_COBYLA without one";
    const auto needs = [&](const std::string& algo, const std::string& phase) {
        return "NLOPT:" + algo + " needs an INNER block (" + phase +
               "): NLopt has no default local optimiser for it" + alternatives;
    };
    struct Case { std::string what, body, message; };
    std::vector<Case> cases;
    for (const std::string algo : {"AUGLAG", "AUGLAG_EQ", "G_MLSL", "G_MLSL_LDS"})
        cases.push_back({"PLAIN " + algo,
                         "    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
                             "\n    MAXEVAL 20\n",
                         needs(algo, "the STRATEGY-level optimiser")});
    cases.push_back({"MULTISTART AUGLAG",
                     "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO AUGLAG\n"
                     "    MAXEVAL 20\n    N_STARTS 2\n",
                     needs("AUGLAG", "the STRATEGY-level optimiser")});
    cases.push_back({"PRESTAGE refinement AUGLAG",
                     "    TYPE PRESTAGE\n    EXPLORE\n      LIB NLOPT\n"
                     "      ALGO GN_CRS2_LM\n      MAXEVAL 20\n    ENDEXPLORE\n"
                     "    REFINE\n      LIB NLOPT\n      ALGO AUGLAG\n"
                     "      MAXEVAL 10\n    ENDREFINE\n",
                     needs("AUGLAG", "the PRESTAGE refinement")});
    cases.push_back({"EXPLORE G_MLSL",
                     "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                     "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
                     "      ALGO G_MLSL\n      MAXEVAL 20\n    ENDEXPLORE\n",
                     needs("G_MLSL", "the EXPLORE block")});
    for (const std::string algo : {"G_MLSL_LDS", "AUGLAG"})
        cases.push_back({"INNER " + algo + " under GN_MLSL_LDS",
                         "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
                         "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
                         "      ALGO " + algo + "\n    ENDINNER\n",
                         "the INNER block of the STRATEGY-level optimiser "
                         "names NLOPT:" + algo + ", which needs an INNER "
                         "block of its own; INNER blocks do not nest: pick a "
                         "local algorithm such as LN_BOBYQA"});
    for (const auto& c : cases) {
        INFO(c.what << "\n" << c.body);
        const auto v = judge(strategy_config(c.body));
        CHECK(v.startup == c.message);
    }

    /* With a default local optimiser (LN_COBYLA or LD_MMA) they run
     * without an INNER block, and an INNER LN_AUGLAG runs its own. */
    for (const std::string& body : std::vector<std::string>{
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_AUGLAG\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n"
             "    FTOL_REL 1e-8\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
             "    MAXEVAL 20\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
             "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
             "      ALGO LN_AUGLAG\n    ENDINNER\n"}) {
        INFO(body);
        CHECK(judge(strategy_config(body)).startup == "");
    }

    /* run_inner() applies the rule again to every spec it runs, with no
     * phase to name. */
    OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "AUGLAG";
    sp.stop.maxeval = 20;
    CHECK(error_of([&] { tadah::hpo::require_inner_backend(sp, ""); }) ==
          "NLOPT:AUGLAG needs an INNER block: NLopt has no default local "
          "optimiser for it" + alternatives);
    sp.inner = std::make_unique<OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "G_MLSL";
    CHECK(error_of([&] { tadah::hpo::require_inner_backend(sp, ""); }) ==
          "the INNER block of NLOPT:AUGLAG names NLOPT:G_MLSL, which needs an "
          "INNER block of its own; INNER blocks do not nest: pick a local "
          "algorithm such as LN_BOBYQA");
    sp.inner->algo = "LN_BOBYQA";
    CHECK(error_of([&] { tadah::hpo::require_inner_backend(sp, ""); }) == "");
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
    SECTION("the PRESTAGE refinement's NLOPT:AUGLAG without an INNER block, "
            "before the exploration") {
        /* Pins: hpo_run() stops at startup, before any data is loaded or
         * any evaluation runs (run_hpo checks both).
         * Before the fix: the exploration's 20 evaluations ran (loss.tadah
         * had 20 rows), then NLopt refused the refinement: "local
         * optimizer must be specified for AUGLAG". */
        const auto err = run_hpo(
            "OPTIMIZER\n  STRATEGY\n    TYPE PRESTAGE\n"
            "    EXPLORE\n      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
            "      MAXEVAL 20\n      SEED 1\n    ENDEXPLORE\n"
            "    REFINE\n      LIB NLOPT\n      ALGO AUGLAG\n"
            "      MAXEVAL 10\n    ENDREFINE\n  ENDSTRATEGY\nENDOPTIMIZER\n");
        CHECK_THAT(err, Catch::Contains("NLOPT:AUGLAG needs an INNER block "
                                        "(the PRESTAGE refinement): NLopt has "
                                        "no default local optimiser for it"));
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
                       /* not "set PARAM fd_step", which NOISE AUTO now
                        * refuses beside it (owner decision D8) */
                       Catch::Contains("\n  Remove AUTO from the NOISE block "
                                       "and set PARAM fd_step to at most ") &&
                       Catch::Contains("; or use OPTIM_SCALE AUTO") &&
                       !Catch::Contains("overrides NOISE AUTO") &&
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
         * 2, finding 2). A build without Ceres refuses CERES:LBFGS as not
         * built first (ceres_build.h; a pin of that refusal; before this
         * test change such a build failed here, expecting the fd_kernel
         * message). */
        for (const std::string& backend :
             {std::string("DLIB:BFGS"), std::string("CERES:LBFGS"),
              std::string("NLOPT:LD_LBFGS")}) {
            const std::string lib = backend.substr(0, backend.find(':'));
            const std::string algo = backend.substr(backend.find(':') + 1);
            const std::string message =
                "the STRATEGY-level optimiser " + backend + ": PARAM "
                "fd_kernel must be inhouse or ceres (got 1)";
            INFO(backend);
            CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB " + lib +
                                        "\n    ALGO " + algo +
                                        "\n    MAXEVAL 20\n"
                                        "    PARAM fd_kernel 1\n")).startup ==
                  (lib == "CERES"
                       ? hpo_test::ceres_expect(message,
                                                "the STRATEGY-level optimiser: ")
                       : message));
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
        /* Pins: NLopt's MLSL or AUGLAG whose local search is
         * derivative-free takes no gradient, whatever the D of its name
         * (it only picks the default local optimiser, LD_MMA, which the
         * INNER block replaces), and says why.
         * Before the review fix: startup "" (obtains_gradients() answered
         * from GD_MLSL_LDS's own caps), and the step was never used. */
        CHECK(judge(strategy_config("    TYPE PLAIN\n    LIB NLOPT\n"
                                    "    ALGO GD_MLSL_LDS\n    MAXEVAL 12\n"
                                    "    PARAM fd_step 0.01\n    INNER\n"
                                    "      LIB NLOPT\n      ALGO LN_BOBYQA\n"
                                    "      MAXEVAL 4\n    ENDINNER\n"))
                  .startup ==
              "the STRATEGY-level optimiser NLOPT:GD_MLSL_LDS: PARAM fd_step "
              "never takes effect: NLOPT:GD_MLSL_LDS takes no gradient: "
              "NLopt's MLSL and AUGLAG take a gradient only for their local "
              "search, and the INNER block's NLOPT:LN_BOBYQA takes none; "
              "remove it.");
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

TEST_CASE("startup validation: GRAD_STRATEGY on a phase that takes no "
          "gradient is refused", "[startup_validation][strategy][fd_stencil]")
{
    /* Pins: GRAD_STRATEGY written on a phase whose optimiser never asks for
     * a gradient (obtains_gradients(): a derivative-free NLopt algorithm,
     * TADAH:RANDOM, TADAH:GRID, DLIB:GFS) is refused at startup, naming the
     * phase, like the finite-difference PARAMs there; a phase that obtains
     * gradients -- its own, or its local search's -- keeps it.
     * Before the fix: each was accepted (startup ""), the run ignored it
     * and the journal did not mention it. */
    struct Case { std::string body, who; };
    for (const auto& c : std::vector<Case>{
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
              "    MAXEVAL 20\n    GRAD_STRATEGY NUMERICAL\n",
              "the STRATEGY-level optimiser NLOPT:LN_SBPLX"},
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_BOBYQA\n"
              "    MAXEVAL 20\n    GRAD_STRATEGY HYBRID\n",
              "the STRATEGY-level optimiser NLOPT:LN_BOBYQA"},
             {"    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
              "    MAXEVAL 5\n    GRAD_STRATEGY NUMERICAL\n",
              "the STRATEGY-level optimiser TADAH:RANDOM"},
             {"    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
              "    GRAD_STRATEGY NUMERICAL\n",
              "the STRATEGY-level optimiser TADAH:GRID"},
             {"    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
              "    MAXEVAL 5\n    GRAD_STRATEGY ANALYTICAL\n",
              "the STRATEGY-level optimiser DLIB:GFS"},
             {"    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
              "    MAXEVAL 20\n    EXPLORE\n      LIB TADAH\n"
              "      ALGO RANDOM\n      MAXEVAL 3\n"
              "      GRAD_STRATEGY NUMERICAL\n    ENDEXPLORE\n",
              "the PRESTAGE exploration (EXPLORE block) TADAH:RANDOM"}}) {
        INFO(c.body);
        const std::string backend = c.who.substr(c.who.rfind(' ') + 1);
        CHECK(judge(strategy_config(c.body)).startup ==
              c.who + ": GRAD_STRATEGY never takes effect: " + backend +
                  " takes no gradient; remove it.");
    }
    /* Pins: NLopt's MLSL and AUGLAG take a gradient only for their local
     * search (mlsl.c and auglag.c evaluate their own points with none), so
     * one whose local search is derivative-free -- a derivative-free INNER
     * block, or NLopt's default LN_COBYLA -- refuses GRAD_STRATEGY in every
     * phase, whatever the D or N of its name (the D only picks the default
     * LD_MMA, which an INNER block replaces), and says which search takes
     * none.
     * Before the review fix: the GD_/LD_ cases were accepted (startup "":
     * obtains_gradients() answered from their own caps), the journal printed
     * "GRAD_STRATEGY : NUMERICAL" and the losses were byte-identical to a run
     * without it; GN_MLSL_LDS and LN_AUGLAG without an INNER block were
     * refused without the reason ("... NLOPT:GN_MLSL_LDS takes no gradient;
     * remove it."). */
    const auto inner = [](const std::string& indent, const std::string& algo) {
        return indent + "INNER\n" + indent + "  LIB NLOPT\n" + indent +
               "  ALGO " + algo + "\n" + indent + "  MAXEVAL 4\n" + indent +
               "ENDINNER\n";
    };
    const std::string the_inner = "the INNER block's ";
    const std::string the_default = "NLopt's default local optimiser, ";
    struct LocalCase { std::string body, who, local; };
    for (const auto& c : std::vector<LocalCase>{
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO GD_MLSL_LDS\n"
              "    MAXEVAL 12\n    SEED 3\n    GRAD_STRATEGY NUMERICAL\n" +
                  inner("    ", "LN_BOBYQA"),
              "the STRATEGY-level optimiser NLOPT:GD_MLSL_LDS",
              the_inner + "NLOPT:LN_BOBYQA"},
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n"
              "    MAXEVAL 12\n    GRAD_STRATEGY NUMERICAL\n" +
                  inner("    ", "LN_BOBYQA"),
              "the STRATEGY-level optimiser NLOPT:LD_AUGLAG",
              the_inner + "NLOPT:LN_BOBYQA"},
             {"    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LD_AUGLAG_EQ\n"
              "    MAXEVAL 20\n    N_STARTS 2\n    GRAD_STRATEGY HYBRID\n" +
                  inner("    ", "LN_SBPLX"),
              "each MULTISTART start NLOPT:LD_AUGLAG_EQ",
              the_inner + "NLOPT:LN_SBPLX"},
             {"    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
              "    MAXEVAL 20\n    EXPLORE\n      LIB NLOPT\n"
              "      ALGO GD_MLSL\n      MAXEVAL 12\n"
              "      GRAD_STRATEGY NUMERICAL\n" +
                  inner("      ", "LN_COBYLA") + "    ENDEXPLORE\n",
              "the PRESTAGE exploration (EXPLORE block) NLOPT:GD_MLSL",
              the_inner + "NLOPT:LN_COBYLA"},
             {"    TYPE PRESTAGE\n    EXPLORE\n      LIB TADAH\n"
              "      ALGO RANDOM\n      MAXEVAL 3\n    ENDEXPLORE\n"
              "    REFINE\n      LIB NLOPT\n      ALGO LD_AUGLAG\n"
              "      MAXEVAL 10\n      GRAD_STRATEGY HYBRID\n" +
                  inner("      ", "LN_BOBYQA") + "    ENDREFINE\n",
              "the PRESTAGE refinement NLOPT:LD_AUGLAG",
              the_inner + "NLOPT:LN_BOBYQA"},
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
              "    MAXEVAL 20\n    GRAD_STRATEGY NUMERICAL\n",
              "the STRATEGY-level optimiser NLOPT:GN_MLSL_LDS",
              the_default + "NLOPT:LN_COBYLA,"},
             {"    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_AUGLAG\n"
              "    MAXEVAL 20\n    GRAD_STRATEGY HYBRID\n",
              "the STRATEGY-level optimiser NLOPT:LN_AUGLAG",
              the_default + "NLOPT:LN_COBYLA,"}}) {
        INFO(c.body);
        const std::string backend = c.who.substr(c.who.rfind(' ') + 1);
        CHECK(judge(strategy_config(c.body)).startup ==
              c.who + ": GRAD_STRATEGY never takes effect: " + backend +
                  " takes no gradient: NLopt's MLSL and AUGLAG take a "
                  "gradient only for their local search, and " + c.local +
                  " takes none; remove it.");
    }
    /* A phase that obtains gradients keeps it: MLSL or AUGLAG with a
     * gradient-based INNER search (its GRAD_STRATEGY obtains the gradient
     * that search uses), whatever the N of its name, or with NLopt's
     * default LD_MMA, DLIB:BFGS. */
    for (const std::string& body : std::vector<std::string>{
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_MLSL_LDS\n"
             "    MAXEVAL 20\n    GRAD_STRATEGY HYBRID\n    INNER\n"
             "      LIB NLOPT\n      ALGO LD_LBFGS\n      MAXEVAL 5\n"
             "    ENDINNER\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
             "    MAXEVAL 20\n    GRAD_STRATEGY NUMERICAL\n" +
                 inner("    ", "LD_SLSQP"),
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n"
             "    MAXEVAL 20\n    GRAD_STRATEGY NUMERICAL\n",
             "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GD_MLSL_LDS\n"
             "    MAXEVAL 20\n    GRAD_STRATEGY HYBRID\n",
             "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 20\n"
             "    GRAD_STRATEGY HYBRID\n"}) {
        INFO(body);
        CHECK(judge(strategy_config(body)).startup == "");
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
     * decision D7, 2026-09-29). Beside an explicit PARAM fd_step it has no
     * effect either, and is refused too (owner decision D8). */
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
        r.startup = checked_refusal(
            config,
            error_of([&] { validate_hpo_setup(ctx, *r.trg, true, true); }));
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
        /* Pins: NLopt's MLSL or AUGLAG whose local search is
         * derivative-free measures no slopes, whatever the D of its name,
         * and the message says which search takes no gradient.
         * Before the review fix: startup "", and fd_step was derived
         * (NOISE_AUTO: PARAM fd_step set to 0.001) for finite differences
         * no evaluation took. */
        const Run auglag = run(noise(
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_AUGLAG\n"
            "    MAXEVAL 20\n    INNER\n      LIB NLOPT\n"
            "      ALGO LN_BOBYQA\n      MAXEVAL 10\n    ENDINNER\n"));
        CHECK(auglag.startup ==
              "the STRATEGY-level optimiser NLOPT:LD_AUGLAG: NOISE AUTO has no "
              "effect: NLOPT:LD_AUGLAG measures no slopes: NLopt's MLSL and "
              "AUGLAG take a gradient only for their local search, and the "
              "INNER block's NLOPT:LN_BOBYQA takes none; remove AUTO from the "
              "NOISE block.");
        CHECK(auglag.trg->spec.param.count("fd_step") == 0u);
        CHECK(auglag.journal.find("NOISE_AUTO") == std::string::npos);
        /* ASA takes its slopes with probes of its own */
        CHECK(run(noise("    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
                        "    MAXEVAL 20\n"))
                  .startup ==
              "the STRATEGY-level optimiser ASA:ASA: NOISE AUTO has no "
              "effect: ASA:ASA measures its slopes with probes of its own "
              "(PARAM Delta_X), not with fd_step; remove AUTO from the NOISE "
              "block.");
    }
    SECTION("refused beside an explicit PARAM fd_step") {
        /* The explicit step won, with "[HPO] NOISE_AUTO: PARAM fd_step
         * already set explicitly; user value retained" on stderr (owner
         * decision D8, 2026-09-29: refused on the phase concerned). */
        const Run plain = run(noise("    TYPE PLAIN\n    LIB DLIB\n"
                                    "    ALGO BFGS\n    MAXEVAL 20\n"
                                    "    PARAM fd_step 1e-4\n"));
        CHECK(plain.startup ==
              "the STRATEGY-level optimiser DLIB:BFGS: NOISE AUTO has no "
              "effect: PARAM fd_step 0.0001 is set; remove one of them.");
        CHECK(plain.journal.find("NOISE_AUTO") == std::string::npos);
        CHECK(run(noise("    TYPE PRESTAGE\n    EXPLORE\n      LIB TADAH\n"
                        "      ALGO RANDOM\n      MAXEVAL 3\n      SEED 1\n"
                        "    ENDEXPLORE\n    REFINE\n      LIB DLIB\n"
                        "      ALGO BFGS\n      MAXEVAL 20\n"
                        "      PARAM fd_step 1e-4\n    ENDREFINE\n"))
                  .startup ==
              "the PRESTAGE refinement DLIB:BFGS: NOISE AUTO has no effect: "
              "PARAM fd_step 0.0001 is set; remove one of them.");
        CHECK(run(noise("    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
                        "    MAXEVAL 50\n    PARAM fd_step 1e-4\n" + inner))
                  .startup ==
              "the STRATEGY-level optimiser NLOPT:G_MLSL_LDS: NOISE AUTO has "
              "no effect: PARAM fd_step 0.0001 is set; remove one of them.");
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

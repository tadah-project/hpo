// test_param_arity.cpp
//
// PARAM value arity (1.5.0).
//
// `PARAM <name> <value>` used to be parsed with `iss >> p >> raw`, so every
// token past the first was silently discarded: `PARAM INIT_MODE JITTER AUTO
// 0.05` ran at the STEP_FRAC default with no diagnostic. Two rules now apply:
//
//   * a key may declare a value arity > 1 (param_max_values); the full token
//     vector lands in OptimizerSpec::param_tokens, while param/param_raw keep
//     carrying token 0 so pre-existing consumers are untouched;
//   * on the LINE-framed surface (HPOTargets::read) anything left on the line
//     after a key has taken its values is a hard error.
//
// The second rule is deliberately NOT applied to read_optimizer_block(): it
// receives a flat token stream spanning the whole block, so "leftovers" there
// are simply the next key. That case is regression-tested below, because a
// greedy PARAM read passes every line-framed test and then silently eats the
// rest of a refit OPTIMIZER block.

#include "catch2/catch.hpp"

#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/optimizer_spec.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

using tadah::hpo::OptimizerSpec;
using tadah::hpo::dispatch_opt_attr;
using tadah::hpo::param_takes_another_value;
using tadah::hpo::read_optimizer_block;

namespace {

/* Parse one line the way HPOTargets::read frames it, and report whatever the
 * handler left behind (that residue is what the line-framed caller rejects). */
OptimizerSpec parse_line(const std::string& line, std::string& leftover) {
    OptimizerSpec sp;
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    REQUIRE(dispatch_opt_attr(key, iss, sp));
    leftover.clear();
    std::string t;
    if (iss >> t) leftover = t;
    return sp;
}

std::vector<std::string> tokens(std::initializer_list<const char*> ts) {
    return std::vector<std::string>(ts.begin(), ts.end());
}

}  // namespace

TEST_CASE("param_takes_another_value: the INIT_MODE grammar",
          "[hpo][param][arity]")
{
    const std::vector<std::string> none;

    /* Every PARAM takes at least one value, whatever the key. */
    CHECK(param_takes_another_value("STEP_FRAC", none, "0.15"));
    CHECK(param_takes_another_value("INIT_MODE", none, "JITTER"));
    CHECK(param_takes_another_value("NOT_A_KEY", none, "anything"));

    /* ...and no ordinary key takes a second. */
    CHECK_FALSE(param_takes_another_value("STEP_FRAC", {"0.15"}, "0.3"));
    CHECK_FALSE(param_takes_another_value("NOT_A_KEY", {"x"}, "y"));

    /* UNIFORM is complete on its own. */
    CHECK_FALSE(param_takes_another_value("INIT_MODE", {"UNIFORM"}, "AUTO"));

    /* JITTER optionally takes AUTO <p>, or a bare fraction. */
    CHECK(param_takes_another_value("INIT_MODE", {"JITTER"}, "AUTO"));
    CHECK(param_takes_another_value("INIT_MODE", {"JITTER"}, "0.01"));
    CHECK(param_takes_another_value("INIT_MODE", {"JITTER", "AUTO"}, "0.05"));

    /* The decisive case: a following KEY is not a value. This is what stops
     * read_optimizer_block()'s flat token stream being eaten. */
    CHECK_FALSE(param_takes_another_value("INIT_MODE", {"JITTER"}, "SEED"));
    CHECK_FALSE(param_takes_another_value("INIT_MODE", {"JITTER"}, "rubbish"));

    /* Nothing follows a completed form. */
    CHECK_FALSE(param_takes_another_value("INIT_MODE", {"JITTER", "0.01"}, "9"));
    CHECK_FALSE(
        param_takes_another_value("INIT_MODE", {"JITTER", "AUTO", "0.05"}, "9"));
}

TEST_CASE("PARAM: a single value leaves param_tokens empty",
          "[hpo][param][arity]")
{
    std::string leftover;
    const auto sp = parse_line("PARAM STEP_FRAC 0.15", leftover);

    CHECK(leftover.empty());
    CHECK(sp.param.at("STEP_FRAC") == Approx(0.15));
    CHECK(sp.param_tokens.empty());     // no extras -> nothing recorded
    CHECK(sp.param_raw.empty());        // numeric -> param, not param_raw
}

TEST_CASE("PARAM: a token value still lands in param_raw",
          "[hpo][param][arity]")
{
    std::string leftover;
    const auto sp = parse_line("PARAM INIT_MODE UNIFORM", leftover);

    CHECK(leftover.empty());
    CHECK(sp.param_raw.at("INIT_MODE") == "UNIFORM");
    CHECK(sp.param_tokens.empty());
}

TEST_CASE("PARAM INIT_MODE JITTER AUTO 0.05: extras reach param_tokens",
          "[hpo][param][arity]")
{
    std::string leftover;
    const auto sp = parse_line("PARAM INIT_MODE JITTER AUTO 0.05", leftover);

    CHECK(leftover.empty());
    /* token 0 still drives the existing enum parse (penum reads param_raw) */
    CHECK(sp.param_raw.at("INIT_MODE") == "JITTER");
    REQUIRE(sp.param_tokens.count("INIT_MODE") == 1u);
    CHECK(sp.param_tokens.at("INIT_MODE")
          == tokens({"JITTER", "AUTO", "0.05"}));
}

TEST_CASE("PARAM INIT_MODE JITTER 0.01: the two-token form parses",
          "[hpo][param][arity]")
{
    std::string leftover;
    const auto sp = parse_line("PARAM INIT_MODE JITTER 0.01", leftover);

    CHECK(leftover.empty());
    CHECK(sp.param_raw.at("INIT_MODE") == "JITTER");
    CHECK(sp.param_tokens.at("INIT_MODE") == tokens({"JITTER", "0.01"}));
}

TEST_CASE("PARAM: a stray token on a single-value key is left on the line",
          "[hpo][param][arity]")
{
    /* dispatch_opt_attr consumes only what the key defines; HPOTargets::read
     * turns this residue into "line N: PARAM takes no further value here". */
    std::string leftover;
    const auto sp = parse_line("PARAM STEP_FRAC 0.15 0.3", leftover);

    CHECK(sp.param.at("STEP_FRAC") == Approx(0.15));
    CHECK(leftover == "0.3");           // <- detected, not discarded
}

TEST_CASE("PARAM: a fourth INIT_MODE token is left on the line",
          "[hpo][param][arity]")
{
    std::string leftover;
    parse_line("PARAM INIT_MODE JITTER AUTO 0.05 rubbish", leftover);
    CHECK(leftover == "rubbish");
}

TEST_CASE("PARAM INIT_MODE JITTER <garbage>: garbage is left on the line",
          "[hpo][param][arity]")
{
    /* `rubbish` is neither AUTO nor a number, so the grammar declines it and
     * the line-framed caller reports it instead of silently ignoring it. */
    std::string leftover;
    const auto sp = parse_line("PARAM INIT_MODE JITTER rubbish", leftover);
    CHECK(sp.param_raw.at("INIT_MODE") == "JITTER");
    CHECK(sp.param_tokens.empty());
    CHECK(leftover == "rubbish");
}

TEST_CASE("read_optimizer_block: a bounded PARAM read does not swallow the "
          "rest of the block",
          "[hpo][param][arity][refit]")
{
    /* The regression guard. read_optimizer_block() sees the whole block as one
     * flat token stream, so a PARAM that consumed "everything after the value"
     * would eat SEED, MAXEVAL and ENDOPTIMIZER, and `pos` would run past the
     * terminator. Each PARAM below is followed by another key on purpose. */
    const auto tok = tokens({
        "OPTIMIZER",
        "LIB", "TADAH",
        "ALGO", "ANNEAL_PT",
        "PARAM", "INIT_MODE", "JITTER",     // arity 3, only 1 value present
        "SEED", "1234",
        "PARAM", "STEP_FRAC", "0.15",
        "MAXEVAL", "260",
        "ENDOPTIMIZER",
        "TRAILING_SENTINEL",
    });

    OptimizerSpec sp;
    std::size_t pos = 0;
    REQUIRE_NOTHROW(read_optimizer_block(tok, pos, sp));

    /* Everything after the block must be untouched. */
    REQUIRE(pos == tok.size() - 1);
    CHECK(tok[pos] == "TRAILING_SENTINEL");

    CHECK(sp.lib == "TADAH");
    CHECK(sp.algo == "ANNEAL_PT");
    CHECK(sp.param_raw.at("INIT_MODE") == "JITTER");
    CHECK(sp.param.at("STEP_FRAC") == Approx(0.15));
    REQUIRE(sp.seed.has_value());
    CHECK(*sp.seed == 1234u);
    REQUIRE(sp.stop.maxeval.has_value());
    CHECK(*sp.stop.maxeval == 260u);
    /* Only one value token was present, so nothing was recorded as "extra". */
    CHECK(sp.param_tokens.empty());
}

TEST_CASE("read_optimizer_block: INIT_MODE extras are consumed, next key intact",
          "[hpo][param][arity][refit]")
{
    const auto tok = tokens({
        "OPTIMIZER",
        "PARAM", "INIT_MODE", "JITTER", "AUTO", "0.05",
        "SEED", "7",
        "ENDOPTIMIZER",
    });

    OptimizerSpec sp;
    std::size_t pos = 0;
    REQUIRE_NOTHROW(read_optimizer_block(tok, pos, sp));

    CHECK(pos == tok.size());
    CHECK(sp.param_tokens.at("INIT_MODE")
          == tokens({"JITTER", "AUTO", "0.05"}));
    REQUIRE(sp.seed.has_value());
    CHECK(*sp.seed == 7u);      // SEED was NOT eaten by the PARAM read
}

TEST_CASE("OptimizerSpec copy carries param_tokens", "[hpo][param][arity]")
{
    std::string leftover;
    const auto sp = parse_line("PARAM INIT_MODE JITTER AUTO 0.05", leftover);
    const OptimizerSpec copy(sp);
    CHECK(copy.param_tokens.at("INIT_MODE")
          == tokens({"JITTER", "AUTO", "0.05"}));
}

// test_optimizer_attr_reader.cpp
//
// The reader of the keys of the optimiser group, dispatch_opt_attr()
// (optimizer_block.h), which `tadah refit` calls through
// read_optimizer_block() on the flat token stream of a stage's OPTIMIZER
// block and the target-file parsers on one line: what it reads for each
// input below, the settings of the optimiser or the refusal, is pinned as
// it was read before the reader took its keys and their values from the
// generated tables of the registry (hpotarget_registry_gen.h), so the
// reader gives `tadah refit` the same results on its tables.
//
// Tag [hpotarget_refit]. The expected outcomes were taken from the reader
// of the hand-written tables (the commit that adds this file).

#include "catch2/catch.hpp"

#include <tadah/core/number_format.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::hpo::OptimizerSpec;

template <class T, class F>
std::string opt(const std::optional<T>& v, F f)
{
    return v ? f(*v) : std::string("-");
}

std::string num(double v) { return tadah::core::format_roundtrip(v); }

template <class T>
std::string whole(T v) { return std::to_string(v); }

template <class M>
std::vector<std::string> keys_of(const M& m)
{
    std::vector<std::string> k;
    for (const auto& kv : m) k.push_back(kv.first);
    std::sort(k.begin(), k.end());
    return k;
}

/* The settings of an optimiser the reader sets, on one line. */
std::string settings(const OptimizerSpec& s)
{
    std::string out = "lib=" + s.lib + " algo=" + s.algo;
    out += " maxeval=" + opt(s.stop.maxeval, whole<std::uint64_t>);
    out += " maxtime=" + opt(s.stop.maxtime, num);
    out += " stopval=" + opt(s.stop.stopval, num);
    out += " ftol_rel=" + opt(s.stop.ftol_rel, num);
    out += " ftol_abs=" + opt(s.stop.ftol_abs, num);
    out += " xtol_rel=" + opt(s.stop.xtol_rel, num);
    out += " xtol_abs=" + opt(s.stop.xtol_abs, num);
    out += " pop=" + opt(s.pop, whole<unsigned>);
    out += " step=" + opt(s.step, num);
    out += " overrides=[";
    for (const auto& o : s.step_overrides) {
        out += o.key + "(";
        for (std::size_t i = 0; i < o.indices.size(); ++i)
            out += (i ? "," : "") + std::to_string(o.indices[i]);
        out += ")=" + num(o.frac) + ";";
    }
    out += "] seed=" + opt(s.seed, whole<std::uint64_t>);
    out += " vector=" + opt(s.vector_storage, whole<unsigned>);
    out += " threads=" + opt(s.num_threads, whole<unsigned>);
    out += " grad=" +
           (s.grad_strategy ? std::string(tadah::hpo::grad_strategy_name(
                                  *s.grad_strategy))
                            : std::string("-"));
    out += " param={";
    for (const auto& k : keys_of(s.param))
        out += k + ":" + num(s.param.at(k)) + ";";
    out += "} raw={";
    for (const auto& k : keys_of(s.param_raw))
        out += k + ":" + s.param_raw.at(k) + ";";
    out += "} tokens={";
    for (const auto& k : keys_of(s.param_tokens)) {
        out += k + ":";
        for (const auto& t : s.param_tokens.at(k)) out += t + ",";
        out += ";";
    }
    out += "}";
    if (s.inner) out += " inner{" + settings(*s.inner) + "}";
    return out;
}

std::vector<std::string> tokens_of(const std::string& text)
{
    std::istringstream in(text);
    std::vector<std::string> out;
    for (std::string t; in >> t;) out.push_back(t);
    return out;
}

/* A stage block read as `tadah refit` reads it (no hook of a stage): the
 * settings and the position after the block, or the refusal. */
std::string read_block(const std::string& text)
{
    const auto tok = tokens_of(text);
    std::size_t pos = 0;
    OptimizerSpec spec;
    try {
        tadah::hpo::read_optimizer_block(tok, pos, spec, nullptr, nullptr,
                                         "W");
    } catch (const std::exception& e) {
        return std::string("error: ") + e.what();
    }
    return settings(spec) + " pos=" + std::to_string(pos);
}

/* One line read with dispatch_opt_attr(): whether it read the key, what
 * it left on the line and the settings, or the refusal. */
std::string read_line(const std::string& text)
{
    std::istringstream iss(text);
    std::string key;
    iss >> key;
    OptimizerSpec spec;
    bool read = false;
    try {
        read = tadah::hpo::dispatch_opt_attr(key, iss, spec);
    } catch (const std::exception& e) {
        return std::string("error: ") + e.what();
    }
    std::string left;
    for (std::string t; iss >> t;) left += t + " ";
    return std::string(read ? "read" : "not read") + " left=[" + left +
           "] " + settings(spec);
}

struct Case {
    const char* input;
    const char* expected;
};

}  // namespace

/* Pins: the stage blocks of `tadah refit` give the settings or the
 * refusals they gave on the hand-written tables, keys in any case,
 * each value read with the number and integer grammar of its key, STEP,
 * PARAM and GRAD_STRATEGY with their own grammars, a key given twice, an
 * unknown key, an INNER block and a block left open. */
TEST_CASE("dispatch_opt_attr: the stage blocks of tadah refit read as on "
          "the hand-written tables", "[hpotarget_refit]")
{
    const std::vector<Case> cases = {
#include "data/optimizer_attr_reader_blocks.inc"
    };
    for (const Case& c : cases) {
        CAPTURE(c.input);
        CHECK(read_block(c.input) == c.expected);
    }
}

/* Pins: one line of a target file read with dispatch_opt_attr(): a key
 * of the group in its case only, the values it reads and what it leaves
 * on the line for the line's leftover check. */
TEST_CASE("dispatch_opt_attr: one line reads as on the hand-written "
          "tables", "[hpotarget_refit]")
{
    const std::vector<Case> cases = {
#include "data/optimizer_attr_reader_lines.inc"
    };
    for (const Case& c : cases) {
        CAPTURE(c.input);
        CHECK(read_line(c.input) == c.expected);
    }
}

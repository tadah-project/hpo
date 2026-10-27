// test_hpotarget_registry_bindings.cpp
//
// The generated tables of the target-file registry
// (include/tadah/hpo/hpotarget_registry_gen.h and hpo_param_registry_gen.h,
// written by blueprint/gen_hpotarget.py) against the tables the parser and
// the startup checks hold by hand, which a parser built on the generated
// tables replaces: each generated table must equal the hand-written one,
// in order where the order shows in a message.
//
// Where the hand-written table can be read (a function of the library:
// opt_attr_keys(), the OPTIM refusals of HPOTargets, LossFunction,
// hpotarget_group_specs(), the ASA help table), the two are compared entry
// by entry. Where it is local to a function (the block names, the removed
// keywords, the INNER refusals, the top-level and once-set keys of
// HPOTargets::read, the options of the gate lines, line_options() of
// lammps_sim.cpp, the options of PC_OBSERVABLE cohesive_energy, the INIT
// methods, the STRATEGY TYPE keys, kDescs and kExcluded of
// asa_options.cpp, the PARAM lists of the adapters, the reader messages of
// strict_parse.cpp), each generated entry is put through the code that
// reads it,
// and the code's answer, a refusal with its message, is compared with the
// entry's message filled in. A name the code knows without a generated
// entry is the coverage check's (hpo_hpotarget_registry_coverage, which
// reads these functions, the set of is_top_level_key() among them); the
// OPTIM refusals are compared both ways over every key of the blueprint.
//
// Tag [hpotarget_gen]. The file compiles with or without the generated
// headers (__has_include): without them it holds one case, which fails.
// The build compiles it again when the headers appear, change or go
// (tests/CMakeLists.txt, hpo_depend_on_has_include(); CTest
// hpo_has_include_rebuild checks the mechanism).
//
// Before the fix: the headers and their generator do not exist; the one
// case "hpotarget generated tables: the generated headers exist" fails.

#include "catch2/catch.hpp"

#if __has_include(<tadah/hpo/hpotarget_registry_gen.h>) && \
    __has_include(<tadah/hpo/hpo_param_registry_gen.h>)
#define TADAH_HPOTARGET_TABLES 1
#else
#define TADAH_HPOTARGET_TABLES 0
#endif

#if !TADAH_HPOTARGET_TABLES

TEST_CASE("hpotarget generated tables: the generated headers exist",
          "[hpotarget_gen]")
{
    FAIL("include/tadah/hpo/hpotarget_registry_gen.h or "
         "hpo_param_registry_gen.h is missing: blueprint/gen_hpotarget.py "
         "--header writes them (build target "
         "hpo_regenerate_hpotarget_registry)");
}

#else

#include <tadah/hpo/hpo_param_registry_gen.h>
#include <tadah/hpo/hpotarget_registry_gen.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/mlip/engines/hpotarget_writer.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace R = tadah::hpo::hpotarget_registry;
namespace P = tadah::hpo::param_registry;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using tadah::hpo::OptimizerSpec;
using Lines = std::vector<std::string>;

std::string str(std::string_view s) { return std::string(s); }

/* A number as the parser's messages write a value they read. */
std::string core_text(double v) { return tadah::core::format_roundtrip(v); }

/* A number as a stream writes it by default (the loss functions'
 * messages). */
std::string stream_text(double v)
{
    std::ostringstream os;
    os << v;
    return os.str();
}

/* The least double above x that is not subnormal: the first value a bound
 * "above x" takes. A subnormal is left out, as a build with -ffast-math
 * may read one as zero. */
double least_above(double x)
{
    if (x >= 0.0 && x < std::numeric_limits<double>::min())
        return std::numeric_limits<double>::min();
    return std::nextafter(x, std::numeric_limits<double>::infinity());
}

/* The message `f` throws, "" when it returns. */
std::string thrown(const std::function<void()>& f)
{
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* "a, b and c": a list as the parser's messages write it. */
std::string and_list(const std::vector<std::string>& v)
{
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i)
        out += (i == 0 ? "" : i + 1 == v.size() ? " and " : ", ") + v[i];
    return out;
}

std::string joined(const std::vector<std::string>& v, const std::string& sep)
{
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i)
        out += (i == 0 ? "" : sep) + v[i];
    return out;
}

/* A startup message of the registry without the "{phase}: " the startup
 * checks put before the refusal of a validator. */
std::string_view without_phase(std::string_view tmpl)
{
    constexpr std::string_view phase = "{phase}: ";
    return tmpl.substr(0, phase.size()) == phase ? tmpl.substr(phase.size())
                                                 : tmpl;
}

/* Whether `actual` is the message of `tmpl` with `values` put in, every
 * other placeholder any text: the registry's loader reads an error so
 * (hpotarget_registry.template_regex). For a message a part of which the
 * code builds and the registry does not hold. */
bool matches(const std::string& actual, std::string_view tmpl,
             std::initializer_list<R::Value> values)
{
    static const std::string special = "\\^$.|?*+()[]{}/";
    const auto escape = [&](std::string_view s) {
        std::string out;
        for (char c : s) {
            if (special.find(c) != std::string::npos) out += '\\';
            out += c;
        }
        return out;
    };
    std::string re;
    std::size_t i = 0, lit = 0;
    const auto head = [](char c) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    };
    const auto tail = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    while (i < tmpl.size()) {
        if (tmpl[i] == '{' && (i == 0 || tmpl[i - 1] != '$') &&
            i + 1 < tmpl.size() && head(tmpl[i + 1])) {
            std::size_t j = i + 2;
            while (j < tmpl.size() && tail(tmpl[j])) ++j;
            if (j < tmpl.size() && tmpl[j] == '}') {
                re += escape(tmpl.substr(lit, i - lit));
                const std::string_view name = tmpl.substr(i + 1, j - i - 1);
                const R::Value* v = nullptr;
                for (const R::Value& x : values)
                    if (x.name == name) v = &x;
                re += v ? escape(v->text) : std::string("[\\s\\S]*?");
                i = lit = j + 1;
                continue;
            }
        }
        ++i;
    }
    re += escape(tmpl.substr(lit));
    return std::regex_match(actual, std::regex(re));
}

/* Whether `actual` is one of `templates` filled with `values` (a template
 * with a placeholder `values` does not give is not that message). */
bool one_of(const std::string& actual, R::Span<std::string_view> templates,
            std::initializer_list<R::Value> values,
            bool strip_phase = false)
{
    for (std::string_view t : templates) {
        try {
            if (R::fill(strip_phase ? without_phase(t) : t, values) == actual)
                return true;
        } catch (const std::invalid_argument&) {
        }
    }
    return false;
}

const R::Key& key(R::KeyId id)
{
    return R::kKeys[static_cast<std::size_t>(id)];
}

const R::Key& root_key(std::string_view name)
{
    const R::Key* k = R::find_key(name, R::BlockId::ROOT);
    REQUIRE(k != nullptr);
    return *k;
}

const R::Block& block(std::string_view name)
{
    const R::Block* b = R::find_block(name);
    REQUIRE(b != nullptr);
    return *b;
}

std::vector<std::string> names(R::Span<R::KeyId> ids)
{
    std::vector<std::string> out;
    for (R::KeyId id : ids) out.push_back(str(key(id).name));
    return out;
}

/* ------------------------------------------------ target files ---- */

/* A scratch directory of this process, removed at exit. */
const fs::path& scratch()
{
    static const struct Dir {
        fs::path path;
        Dir()
            : path(fs::temp_directory_path() /
                   ("tadah_hpotarget_gen_" + std::to_string(::getpid()) + "_" +
                    std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count())))
        {
            fs::create_directories(path);
        }
        ~Dir() { std::error_code ec; fs::remove_all(path, ec); }
    } dir;
    return dir.path;
}

/* The smallest training configuration whose model builds. */
Context minimal_ctx()
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
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

/* A training configuration whose many-body term is a DM_Embed
 * component, which has no CGRIDMB, SGRIDMB, ... key. */
Context embed_ctx()
{
    std::istringstream in(
        "VERBOSE 0\nATOM Ta\nMODEL M_BLR BF_Linear\nINIT2B false\n"
        "INITMB true\nDESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\n"
        "BASIS Blip5\nFORM NONNEG\nCENTRES 1 2\nWIDTHS 0.5 0.75\n"
        "AMPLITUDES 0.4 0.2\nCUTOFF\nTYPE Cut_Poly\nRADIUS 6\nENDCUTOFF\n"
        "ENDDENSITY\nEMBEDDING\nBASIS Gaussian\nCENTRES 0 1\n"
        "WIDTHS 0.5 0.75\nENVELOPE\nTYPE Smoothstep2\nCUTOFF 2\nENDENVELOPE\n"
        "ENDEMBEDDING\nENDDESCRIPTOR\n");
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* The message HPOTargets gives for the target file `lines`, "" when it
 * reads it. */
std::string parse_error(const Lines& lines, Context ctx = minimal_ctx())
{
    tadah::core::PeriodicTable::initialize();
    static unsigned n = 0;
    const fs::path file = scratch() / ("target_" + std::to_string(n++) + ".hpo");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        for (const auto& l : lines) os << l << "\n";
    }
    return thrown([&] { HPOTargets t(ctx, file.string()); });
}

/* A refusal of line n, whose text is `text`. */
std::string at_line(std::size_t n, const std::string& what,
                    const std::string& text)
{
    return tadah::hpo::config_line_error(n, what, text);
}

Lines operator+(Lines a, const Lines& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

/* Lines 1-2 of the files below: a search parameter and a loss. */
const Lines kRoot = {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1"};
const Lines kOptimizer = {"OPTIMIZER", "LIB DLIB", "ALGO BFGS",
                          "ENDOPTIMIZER"};

/* --------------------------------------------- optimiser specs ---- */

OptimizerSpec spec_of(const std::string& lib, const std::string& algo)
{
    OptimizerSpec sp;
    sp.lib = lib;
    sp.algo = algo;
    sp.low = {0.0};
    sp.high = {1.0};
    sp.stop.maxeval = 10;
    return sp;
}

}  // namespace

/* Pins: fill() puts a value where a template writes {name} (a letter or
 * '_', then letters, digits and '_'), leaves ${name} and every other brace
 * as text, as the registry's loader reads a placeholder, and refuses a
 * placeholder without a value. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: fill() reads placeholders as the "
          "registry's loader does", "[hpotarget_gen]")
{
    CHECK(R::fill("a {x} ${y} {z_1}", {{"x", "1"}, {"z_1", "2"}}) ==
          "a 1 ${y} 2");
    CHECK(R::fill("{{x}} { x} {1x} {}", {{"x", "v"}}) == "{v} { x} {1x} {}");
    CHECK(R::fill("no placeholder", {}) == "no placeholder");
    CHECK_THROWS_AS(R::fill("{x} and {w}", {{"x", "1"}}),
                    std::invalid_argument);
}

/* Pins: the frame of a line's refusal is lexing::error_frame
 * (config_line_error), and each message of [reading] is the one the value
 * reader gives: parse_uint (with and without a maximum, and its note),
 * parse_finite and parse_positive (with the too-small note), next_value,
 * single_value, parse_bool, parse_lammps_timespec, parse_script_timeout;
 * the words of lexing::bool_true and bool_false are the booleans
 * parse_bool reads, in any case, and timespec_words the words of no
 * limit. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the frame and the reader messages "
          "are those of strict_parse", "[hpotarget_gen]")
{
    using namespace tadah::hpo;
    CHECK(config_line_error(7, "W", "T") ==
          R::fill(R::lexing::error_frame,
                  {{"line", "7"}, {"what", "W"}, {"text", "T"}}));
    CHECK(thrown([] { parse_uint("x", "W", 1); }) ==
          R::fill(R::reading::integer_msg,
                  {{"what", "W"}, {"min", "1"}, {"tok", "x"}}));
    CHECK(thrown([] { parse_uint("x", "W", 1, 5); }) ==
          R::fill(R::reading::integer_range_msg,
                  {{"what", "W"}, {"min", "1"}, {"max", "5"}, {"tok", "x"}}));
    CHECK(thrown([] {
              parse_uint("0", "W", 1,
                         std::numeric_limits<std::uint64_t>::max(), "N");
          }) == R::fill(R::reading::integer_msg,
                        {{"what", "W"}, {"min", "1"}, {"tok", "0"}}) +
                    R::fill(R::reading::note_msg, {{"note", "N"}}));
    CHECK(thrown([] { parse_finite("x", "W"); }) ==
          R::fill(R::reading::finite_msg,
                  {{"what", "W"}, {"tok", "x"}, {"underflow", ""}}));
    CHECK(thrown([] { parse_finite("1e-320", "W"); }) ==
          R::fill(R::reading::finite_msg,
                  {{"what", "W"}, {"tok", "1e-320"},
                   {"underflow", R::reading::underflow_note}}));
    CHECK(thrown([] { parse_positive("0", "W", "noun"); }) ==
          R::fill(R::reading::positive_msg, {{"what", "W"}, {"noun", "noun"},
                                             {"tok", "0"}, {"underflow", ""}}));
    CHECK(thrown([] {
              std::istringstream iss("");
              next_value(iss, "K");
          }) == R::fill(R::reading::requires_value_msg, {{"key", "K"}}));
    CHECK(thrown([] {
              std::istringstream iss("a b");
              single_value(iss, "K");
          }) == R::fill(R::reading::single_value_msg,
                        {{"key", "K"}, {"tok", "b"}}));
    CHECK(thrown([] { parse_bool("x", "W"); }) ==
          R::fill(R::reading::bool_msg, {{"what", "W"}, {"tok", "x"}}));
    for (std::string_view w : R::lexing::bool_true) {
        std::string upper(w);
        for (char& c : upper)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        CHECK(parse_bool(str(w), "W"));
        CHECK(parse_bool(upper, "W"));
    }
    for (std::string_view w : R::lexing::bool_false)
        CHECK(!parse_bool(str(w), "W"));
    CHECK(thrown([] { parse_lammps_timespec("x", "W"); }) ==
          R::fill(R::reading::timespec_msg,
                  {{"what", "W"}, {"tok", "x"}, {"underflow", ""}}));
    for (std::string_view w : R::lexing::timespec_words)
        CHECK(!parse_lammps_timespec(str(w), "W").has_value());
    CHECK(thrown([] { parse_script_timeout("0", "W"); }) ==
          R::fill(R::reading::script_timeout_msg,
                  {{"what", "W"}, {"tok", "0"}}));
    CHECK(thrown([] { parse_script_timeout("1e16", "W"); }) ==
          R::fill(R::reading::script_timeout_large_msg,
                  {{"what", "W"}, {"tok", "1e16"}}));
}

/* Pins: the keys of the optimiser group are opt_attr_keys(), in its
 * order; the INNER block takes them less the keys it refuses, and the list
 * its unknown-key message gives is those keys, in that order, joined by
 * '/'; each refused key is refused inside INNER with refuse_msg and its
 * reason, naming the enclosing STRATEGY block; each other key of the group
 * is read there (a line of the key alone is not refused as a key of
 * INNER). Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the optimiser group keys are "
          "opt_attr_keys(), and INNER takes them but its refusals",
          "[hpotarget_gen]")
{
    std::vector<std::string> group;
    for (const R::Key& k : R::kKeys)
        if (k.group == "OPTIMIZER") group.push_back(str(k.name));
    CHECK(group == tadah::hpo::opt_attr_keys());

    const R::Block& inner = block("INNER");
    const Lines head = kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB NLOPT",
                                     "ALGO GN_MLSL", "MAXEVAL 10", "INNER",
                                     "LIB NLOPT", "ALGO LN_SBPLX"};
    const Lines tail = {"ENDINNER", "ENDSTRATEGY", "ENDOPTIMIZER"};
    REQUIRE(parse_error(head + tail) == "");
    const std::size_t at = head.size() + 1;
    const std::string inner_line = std::to_string(head.size() - 2);

    CHECK(parse_error(head + Lines{"ZZZ 1"} + tail) ==
          at_line(at, R::fill(inner.unknown_msg,
                              {{"key", "ZZZ"}, {"open_line", inner_line},
                               {"keys", joined(names(inner.keys), "/")}}),
                  "ZZZ 1"));
    std::vector<std::string> taken = names(inner.keys);
    for (const R::Refusal& r : inner.refuse) {
        CAPTURE(r.key);
        CHECK(std::find(taken.begin(), taken.end(), str(r.key)) == taken.end());
        const std::string line = str(r.key) + " 1";
        CHECK(parse_error(head + Lines{line} + tail) ==
              at_line(at, R::fill(inner.refuse_msg,
                                  {{"key", r.key}, {"why", r.why},
                                   {"enclosing", "STRATEGY"}}),
                      line));
    }
    for (const std::string& k : taken) {
        CAPTURE(k);
        const std::string err = parse_error(head + Lines{k} + tail);
        CHECK(err.find("is not accepted inside an INNER block") ==
              std::string::npos);
        CHECK(err.find("unknown key '" + k + "'") == std::string::npos);
    }
}

/* Pins: the blocks are the parser's: the END line of each, with no block
 * open, is refused with end_without_open_msg naming it; each block whose
 * parents leave out the top level is refused there with its
 * misplaced_msg; OPTIMIZER inside OPTIMIZER is refused with its own,
 * naming the block it is in. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the blocks are the parser's",
          "[hpotarget_gen]")
{
    for (const R::Block& b : R::kBlocks) {
        CAPTURE(b.name);
        CHECK(b.end == "END" + str(b.name));
        CHECK(parse_error(kRoot + Lines{str(b.end)}) ==
              at_line(3, R::fill(R::structure::end_without_open_msg,
                                 {{"end", b.end}, {"name", b.name}}),
                      str(b.end)));
        if (b.parents & R::block_bit(R::BlockId::ROOT)) continue;
        CHECK(parse_error(kRoot + Lines{str(b.name)}) ==
              at_line(3, R::fill(b.misplaced_msg, {}), str(b.name)));
    }
    const R::Block& opt = block("OPTIMIZER");
    CHECK(parse_error(kRoot + Lines{"OPTIMIZER", "OPTIMIZER"}) ==
          at_line(4, R::fill(opt.misplaced_msg,
                             {{"parent", "OPTIMIZER"}, {"parent_line", "3"}}),
                  "OPTIMIZER"));
}

/* Pins: each removed keyword is the parser's, refused with removed_msg
 * and its hint. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the removed keywords are the "
          "parser's, each with its hint", "[hpotarget_gen]")
{
    for (const R::Removed& r : R::kRemoved) {
        CAPTURE(r.name);
        CHECK(parse_error(kRoot + Lines{str(r.name)}) ==
              at_line(3, R::fill(R::structure::removed_msg,
                                 {{"key", r.name}, {"hint", r.hint}}),
                      str(r.name)));
    }
}

/* Pins: kTopLevelKeys are the keys a block refuses as top-level keys
 * (is_top_level_key): each, inside INIT, is refused with
 * root_key_in_block_msg; a key of the top level that is not one of them
 * (MAXTIME, a key of the optimiser group too) is an unknown key there. A
 * name of the code's set that is no key of the top level is the coverage
 * check's, which reads the set. Before the fix: the headers do not
 * exist. */
TEST_CASE("hpotarget generated tables: the top-level keys are those a "
          "block refuses as such", "[hpotarget_gen]")
{
    const Lines head = kRoot + Lines{"OPTIMIZER", "LIB DLIB", "ALGO BFGS",
                                     "INIT"};
    const Lines tail = {"ENDINIT", "ENDOPTIMIZER"};
    const std::vector<std::string> top(R::kTopLevelKeys.begin(),
                                       R::kTopLevelKeys.end());
    for (const std::string& k : top) {
        CAPTURE(k);
        CHECK(parse_error(head + Lines{k} + tail) ==
              at_line(7, R::fill(R::structure::root_key_in_block_msg,
                                 {{"key", k}, {"name", "INIT"},
                                  {"open_line", "6"}, {"end", "ENDINIT"}}),
                      k));
    }
    for (const R::Key& k : R::kKeys) {
        if (!(k.blocks & R::block_bit(R::BlockId::ROOT))) continue;
        if (std::find(top.begin(), top.end(), str(k.name)) != top.end())
            continue;
        CAPTURE(k.name);
        CHECK(parse_error(head + Lines{str(k.name)} + tail) ==
              at_line(7, R::fill(block("INIT").unknown_msg,
                                 {{"key", k.name}, {"open_line", "6"}}),
                      str(k.name)));
    }
}

/* Pins: a top-level key whose repeat is error, given twice, is refused at
 * the second line with its repeat_msg naming the first: the once-set keys
 * of HPOTargets::read (set_once), FAILSCORE, PC_SCRIPT_TIMEOUT, MAXTIME
 * and LOSS (set_for_lines: no line between reads the first), and the
 * metric and PC_FAILPROBE lines, which refuse a repeat themselves. Each
 * such key has a valid line below; a key without one fails the case.
 * Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the top-level keys set once refuse "
          "a second line with their repeat message", "[hpotarget_gen]")
{
    const fs::path script = scratch() / "probe.in";
    {
        std::ofstream os(script);
        os << "variable S equal 1.0\n";
    }
    const std::map<std::string, std::string> valid = {
        {"FAILSCORE", "FAILSCORE 10"},
        {"PC_SCRIPT_TIMEOUT", "PC_SCRIPT_TIMEOUT 60"},
        {"MAXTIME", "MAXTIME 60"},
        {"PC_ISOLATION", "PC_ISOLATION THREAD"},
        {"PC_WORKER_TIMEOUT", "PC_WORKER_TIMEOUT 10"},
        {"FAILSCORE_GRADING", "FAILSCORE_GRADING OFF"},
        {"BUMPGATE", "BUMPGATE CANDIDATE"},
        {"RHOGATE", "RHOGATE 0"},
        {"RHOBARGATE", "RHOBARGATE bcc 3.3 0.8 1.2"},
        {"QFLUSH", "QFLUSH 5"},
        {"OUTPUT", "OUTPUT 1"},
        {"BOUTPUT", "BOUTPUT 1"},
        {"DIGITS", "DIGITS 6"},
        {"DUMP", "DUMP 1 dump_dir"},
        {"BDUMP", "BDUMP 1 bdump_dir"},
        {"OPTIM_SCALE", "OPTIM_SCALE AUTO"},
        {"PC_EMAE", "PC_EMAE 0 1"},
        {"PC_ErRMSE", "PC_ErRMSE 0 1"},
        {"PC_ERSQ", "PC_ERSQ 0 1"},
        {"PC_FAILPROBE",
         "PC_FAILPROBE --script " + script.string() + " --outvar S"},
        {"LOSS", "LOSS L2"}};
    for (const R::Key& k : R::kKeys) {
        if (!(k.blocks & R::block_bit(R::BlockId::ROOT)) ||
            k.repeat != R::Repeat::Error)
            continue;
        CAPTURE(k.name);
        const auto it = valid.find(str(k.name));
        if (it == valid.end())
            FAIL("no valid line of " << k.name << " in this test: add one");
        const std::string& line = it->second;
        CHECK(parse_error(kRoot + Lines{line, line}) ==
              at_line(4, R::fill(k.repeat_msg, {{"first_line", "3"}}), line));
    }
}

/* Pins: the STRATEGY TYPE keys of the registry are kStrategyTypeKeys of
 * strategy_rules.cpp, each with its home TYPE: is_strategy_type_key()
 * names them and no other key of the STRATEGY block; require_strategy_key()
 * passes one under its TYPE and refuses it under another with the
 * STRATEGY block's close check (its {effect} is the code's text). The
 * TYPE words are strategy_type_name()'s, in order. Before the fix: the
 * headers do not exist. */
TEST_CASE("hpotarget generated tables: the STRATEGY TYPE keys are "
          "kStrategyTypeKeys, each under its TYPE", "[hpotarget_gen]")
{
    using tadah::hpo::StrategyType;
    const std::vector<StrategyType> types = {
        StrategyType::PLAIN, StrategyType::MULTISTART, StrategyType::PRESTAGE};
    std::vector<std::string> words;
    for (const R::StrategyType& t : R::kStrategyTypes)
        words.push_back(str(t.name));
    std::vector<std::string> code;
    for (StrategyType t : types)
        code.push_back(tadah::hpo::strategy_type_name(t));
    REQUIRE(words == code);

    const R::Block& strategy = block("STRATEGY");
    REQUIRE(strategy.close_checks.size == 1);
    std::set<std::string> type_keys;
    for (std::size_t h = 0; h < types.size(); ++h) {
        for (R::KeyId id : R::kStrategyTypes[h].keys) {
            const std::string k = str(key(id).name);
            CAPTURE(k);
            type_keys.insert(k);
            CHECK(tadah::hpo::is_strategy_type_key(k));
            for (std::size_t u = 0; u < types.size(); ++u) {
                const std::string err = thrown([&] {
                    tadah::hpo::require_strategy_key(k, types[u], true);
                });
                if (u == h) {
                    CHECK(err == "");
                    continue;
                }
                CHECK(matches(err, strategy.close_checks[0],
                              {{"key", k}, {"type", words[u]}, {"note", ""},
                               {"home", words[h]}}));
            }
        }
    }
    for (const R::Key& k : R::kKeys) {
        if (!(k.blocks & R::block_bit(R::BlockId::STRATEGY))) continue;
        if (type_keys.count(str(k.name))) continue;
        CAPTURE(k.name);
        CHECK(!tadah::hpo::is_strategy_type_key(str(k.name)));
    }
}

/* Pins: the options of a PC_LAMMPS and of a PC_FAILPROBE line are
 * line_options() of lammps_sim.cpp: the unknown-option refusal of each
 * line (a check of its key) lists the line's options of the registry, the
 * refused one left out, in registry order; an option of PC_FAILPROBE
 * alone on a PC_LAMMPS line is named as one. Before the fix: the headers
 * do not exist. */
TEST_CASE("hpotarget generated tables: the options of PC_LAMMPS and "
          "PC_FAILPROBE are line_options(), in order", "[hpotarget_gen]")
{
    for (const std::string line : {"PC_LAMMPS", "PC_FAILPROBE"}) {
        CAPTURE(line);
        std::vector<std::string> options;
        for (const R::LineOption& o : R::kLineOptions)
            if (!o.refused && std::find(o.lines.begin(), o.lines.end(),
                                        line) != o.lines.end())
                options.push_back(str(o.option));
        std::string cmd = "--zzzzzzzzzz";
        const std::string err = thrown([&] {
            tadah::hpo::LammpsSim(cmd, "off", 1.0, tadah::LossFunction{},
                                  line == "PC_FAILPROBE");
        });
        CHECK(one_of(err, root_key(line).checks,
                     {{"key", line}, {"option", "--zzzzzzzzzz"}, {"note", ""},
                      {"list", and_list(options)}}));
    }
    for (const R::LineOption& o : R::kLineOptions) {
        const auto on = [&](std::string_view l) {
            return std::find(o.lines.begin(), o.lines.end(), l) != o.lines.end();
        };
        if (o.refused || !on("PC_FAILPROBE") || on("PC_LAMMPS")) continue;
        CAPTURE(o.option);
        std::string cmd = str(o.option) + " 1";
        const std::string err = thrown([&] {
            tadah::hpo::LammpsSim(cmd, "off", 1.0, tadah::LossFunction{},
                                  false);
        });
        CHECK(err.find("unknown option '" + str(o.option) +
                       "' (a PC_FAILPROBE option)") != std::string::npos);
    }
}

/* Pins: the options of a PC_OBSERVABLE cohesive_energy line are those
 * parse_pc_observable_line reads: each, without its value, is refused as
 * an option that requires one, and another word as an unknown option (the
 * form's checks); the observable kinds are the words of the
 * PC_OBSERVABLE forms, and another word is refused with the kind
 * argument's message. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the options of PC_OBSERVABLE "
          "cohesive_energy are the parser's", "[hpotarget_gen]")
{
    using tadah::hpo::parse_pc_observable_line;
    for (const R::Form& f : R::kForms) {
        if (f.line != "PC_OBSERVABLE") continue;
        CAPTURE(f.word);
        const std::string form_id = "PC_OBSERVABLE." + str(f.word);
        for (const R::LineOption& o : R::kLineOptions) {
            if (std::find(o.lines.begin(), o.lines.end(), form_id) ==
                o.lines.end())
                continue;
            CAPTURE(o.option);
            CHECK(one_of(thrown([&] {
                             parse_pc_observable_line(str(f.word) + " " +
                                                      str(o.option));
                         }),
                         f.checks, {{"option", o.option}}));
        }
        CHECK(one_of(thrown([&] {
                         parse_pc_observable_line(str(f.word) + " --zzz");
                     }),
                     f.checks, {{"tok", "--zzz"}}));
    }
    const R::Key& obs = root_key("PC_OBSERVABLE");
    REQUIRE(obs.args.size >= 1);
    CHECK(thrown([] { parse_pc_observable_line("zzz"); }) ==
          R::fill(obs.args[0].msg, {{"tok", "zzz"}}));
}

/* Pins: the loss functions are LossFunction's, in the order of its Type:
 * each name reads (in any case) as its Type and is written back by
 * to_string(), its parameters are default_params(); a parameter at its gt
 * bound is refused with the LOSS check of a parameter, which names the
 * loss function, the parameter and its default, the least double above gt
 * that is not subnormal is taken (so gt is the code's bound exactly), and
 * so is its default; the unknown-loss refusal (a check of LOSS) lists the
 * names in order. Before the fix: the headers do not exist. With them, a
 * gt below the code's bound or another parameter name fails the case (a
 * pin of the tables against LossFunction). */
TEST_CASE("hpotarget generated tables: the loss functions are "
          "LossFunction's, in order, with their parameters", "[hpotarget_gen]")
{
    using tadah::LossFunction;
    std::vector<std::string> code;
    for (int i = 0; i < 64; ++i) {
        const std::string n =
            LossFunction::to_string(static_cast<LossFunction::Type>(i));
        if (n == "<invalid>") break;
        code.push_back(n);
    }
    std::vector<std::string> gen;
    for (const R::LossFunction& f : R::kLossFunctions) gen.push_back(str(f.name));
    REQUIRE(gen == code);
    for (std::size_t i = 0; i < gen.size(); ++i) {
        const R::LossFunction& f = R::kLossFunctions[i];
        const auto t = static_cast<LossFunction::Type>(i);
        CAPTURE(f.name);
        std::string lower(f.name);
        for (char& c : lower)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        CHECK(LossFunction::from_string(str(f.name)) == t);
        CHECK(LossFunction::from_string(lower) == t);
        std::vector<double> defaults;
        for (const R::LossParam& p : f.params) defaults.push_back(p.default_value);
        CHECK(LossFunction::default_params(t) == defaults);
        for (const R::LossParam& p : f.params) {
            CAPTURE(p.name, p.gt);
            REQUIRE(p.has_gt);
            CHECK(one_of(thrown([&] { LossFunction::resolve_params(t, {p.gt}); }),
                         root_key("LOSS").checks,
                         {{"name", f.name}, {"key", p.name},
                          {"value", stream_text(p.gt)},
                          {"default", stream_text(p.default_value)}}));
            CHECK(thrown([&] {
                      LossFunction::resolve_params(t, {least_above(p.gt)});
                  }) == "");
            CHECK(LossFunction::resolve_params(t, {p.default_value}) ==
                  std::vector<double>{p.default_value});
        }
    }
    const std::string err =
        thrown([] { LossFunction::from_name("zzz", "W"); });
    CHECK(one_of(err, root_key("LOSS").checks, {{"what", "W"}, {"tok", "zzz"}}));
    CHECK(err.find(and_list(gen)) != std::string::npos);
}

/* Pins: the INIT methods are make_one()'s (validate_init_entry): each
 * takes as many values as its arguments, a value more is refused with
 * the method's check, a K below its min with the integer reader's range
 * message (its what, min and max); an unknown method is refused with
 * METHOD's message, listing the methods in order. Before the fix: the
 * headers do not exist. */
TEST_CASE("hpotarget generated tables: the INIT methods are make_one()'s, "
          "with their arguments", "[hpotarget_gen]")
{
    using tadah::hpo::validate_init_entry;
    using Entry = tadah::hpo::InitStrategySpec::Entry;
    std::vector<std::string> methods;
    for (const R::InitMethod& m : R::kInitMethods) {
        methods.push_back(str(m.name));
        CAPTURE(m.name);
        Entry e;
        e.method = str(m.name);
        for (const R::Arg& a : m.args)
            e.args.push_back(a.kind == R::ArgKind::Integer
                                 ? std::to_string(a.min.whole_value)
                                 : std::string("start.tadah"));
        CHECK(thrown([&] { validate_init_entry(e); }) == "");
        Entry more = e;
        more.args.push_back("1");
        CHECK(one_of(thrown([&] { validate_init_entry(more); }), m.checks,
                     {{"key", "METHOD"}}));
        for (std::size_t i = 0; i < m.args.size; ++i) {
            const R::Arg& a = m.args[i];
            if (a.kind != R::ArgKind::Integer || !a.min.set ||
                a.min.whole_value == 0)
                continue;
            REQUIRE(a.max.set);
            Entry low = e;
            low.args[i] = std::to_string(a.min.whole_value - 1);
            CHECK(thrown([&] { validate_init_entry(low); }) ==
                  R::fill(R::reading::integer_range_msg,
                          {{"what", R::fill(a.what, {{"key", "METHOD"}})},
                           {"min", std::to_string(a.min.whole_value)},
                           {"max", std::to_string(a.max.whole_value)},
                           {"tok", low.args[i]}}));
        }
    }
    Entry unknown;
    unknown.method = "ZZZ";
    const R::Key& method = key(R::KeyId::INIT_METHOD);
    REQUIRE(method.args.size >= 1);
    const std::string err = thrown([&] { validate_init_entry(unknown); });
    CHECK(err == R::fill(method.args[0].msg, {{"key", "METHOD"}, {"tok", "ZZZ"}}));
    CHECK(err.find(and_list(methods)) != std::string::npos);
}

/* Pins: the OPTIM targets are the parser's refusal tables: a refused
 * target with a message of its own is the refusal of the OPTIM refusals
 * of HPOTargets (fitted, structure, retired, switch and startup-only
 * keys), and conversely every key of the blueprint those refuse is such a
 * target; a refused target with indices is optim_discrete_index_error of
 * them; a DM_REAM many-body key a DM_Embed potential lacks is
 * optim_component_key_error (its {list} the code's paths); an internal
 * key is refused with the [optim] check of internal keys; the fit-value
 * rules (optim_fit_value_rule) are the special targets' messages, and the
 * FIXWEIGHT index refusal is FIXWEIGHT's. Before the fix: the headers do
 * not exist. */
TEST_CASE("hpotarget generated tables: the OPTIM targets are the parser's "
          "refusal tables", "[hpotarget_gen]")
{
    const auto refusal = [](const std::string& k) {
        for (const auto& f : {HPOTargets::optim_fitted_key_error,
                              HPOTargets::optim_structure_key_error,
                              HPOTargets::optim_retired_key_error,
                              HPOTargets::optim_switch_key_error,
                              HPOTargets::optim_startup_only_key_error})
            if (std::string e = f(k); !e.empty()) return e;
        return std::string();
    };
    const auto has_placeholder = [](std::string_view m) {
        return m.find('{') != std::string_view::npos;
    };
    std::set<std::string> plain;
    const Context embed = embed_ctx();
    for (const R::OptimTarget& t : R::kOptimTargets) {
        const std::string k = str(t.key);
        CAPTURE(k);
        if (t.status == R::OptimTargetStatus::Refused && t.indices.empty() &&
            !has_placeholder(t.msg)) {
            plain.insert(k);
            CHECK(refusal(k) == str(t.msg));
        } else if (t.status == R::OptimTargetStatus::Refused &&
                   !t.indices.empty()) {
            for (int i : t.indices)
                CHECK(HPOTargets::optim_discrete_index_error(
                          k, static_cast<std::size_t>(i)) == str(t.msg));
            CHECK(refusal(k) == "");
        } else if (t.status == R::OptimTargetStatus::Refused) {
            const std::string err =
                HPOTargets::optim_component_key_error(embed, k);
            CHECK(matches(err, t.msg, {}));
            CHECK(HPOTargets::optim_component_key_error(minimal_ctx(), k) == "");
        } else if (k == "FIXWEIGHT") {
            Context c = minimal_ctx();
            c.add("FIXINDEX", "1 2");
            c.add("FIXWEIGHT", "0.5 0.5");
            const std::string line = "OPTIM FIXWEIGHT (3) 0 1";
            const std::string err = parse_error(Lines{line, "PC_ERMSE 0 1"} +
                                                kOptimizer, c);
            CHECK(matches(err, at_line(1, str(t.msg), line), {{"index", "3"}}));
        } else {
            const auto rule = HPOTargets::optim_fit_value_rule(k, 1);
            REQUIRE(rule.has_value());
            const std::string low = rule->strict ? "0" : "-1";
            CHECK(R::fill(t.msg, {{"index", "1"}, {"low", low}})
                      .find(rule->why) != std::string::npos);
        }
    }
    for (const auto& kv : tadah::core::BlueprintParser::getKeysMapRef()) {
        CAPTURE(kv.first);
        CHECK(plain.count(kv.first) == (refusal(kv.first).empty() ? 0u : 1u));
    }
    for (const auto& kv :
         tadah::core::BlueprintParser::getInternalKeysMapRef()) {
        CAPTURE(kv.first);
        CHECK(one_of(HPOTargets::optim_internal_key_error(kv.first),
                     R::optim::checks, {{"key", kv.first}}));
    }
}

/* Pins: the special targets are read with their rule by the parser: an
 * OPTIM line on one, with a low bound its rule refuses, is refused with
 * the target's message (FSIGMA and SSIGMA at index 1; index 2, the
 * relative part, with RULE.optim_relative_error_scale). Before the fix:
 * the headers do not exist. */
TEST_CASE("hpotarget generated tables: the special OPTIM targets are "
          "refused with their messages", "[hpotarget_gen]")
{
    Context c = minimal_ctx();
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", "0.1 0.1");
    c.add("SSIGMA", "0.1 0.1");
    c.add("LAMBDA", 0.0);
    c.add("WDBFILE", 1.0);
    for (const R::OptimTarget& t : R::kOptimTargets) {
        if (t.status != R::OptimTargetStatus::Special || t.key == "FIXWEIGHT")
            continue;
        CAPTURE(t.key);
        const auto rule = HPOTargets::optim_fit_value_rule(str(t.key), 1);
        REQUIRE(rule.has_value());
        const std::string low = rule->strict ? "0" : "-1";
        const std::string line = "OPTIM " + str(t.key) + " (1) " + low + " 1";
        CHECK(parse_error(Lines{line, "PC_ERMSE 0 1"} + kOptimizer, c) ==
              at_line(1, R::fill(t.msg, {{"index", "1"}, {"low", low}}), line));
    }
    const R::Rule* relative = nullptr;
    for (const R::Rule& r : R::kRules)
        if (r.id_text == "RULE.optim_relative_error_scale") relative = &r;
    REQUIRE(relative != nullptr);
    for (const std::string k : {"FSIGMA", "SSIGMA"}) {
        const std::string line = "OPTIM " + k + " (2) -1 1";
        CHECK(parse_error(Lines{line, "PC_ERMSE 0 1"} + kOptimizer, c) ==
              at_line(1, R::fill(relative->msg, {{"key", k}, {"index", "2"},
                                                 {"low", "-1"}}),
                      line));
    }
}

/* Pins: the density keys the arming rule names are
 * HPOTargets::density_gate_keys(): a RHOGATE line no OPTIM line arms is
 * refused at its line with RULE.density_key_arming, {keys} those keys
 * ("a, b, c or d"; the registry holds no list of them). Before the fix:
 * the headers do not exist. */
TEST_CASE("hpotarget generated tables: the density gate rule names "
          "density_gate_keys()", "[hpotarget_gen]")
{
    const R::Rule* rule = nullptr;
    for (const R::Rule& r : R::kRules)
        if (r.id_text == "RULE.density_key_arming") rule = &r;
    REQUIRE(rule != nullptr);
    const auto& dk = HPOTargets::density_gate_keys();
    std::string keys;
    for (std::size_t i = 0; i < dk.size(); ++i)
        keys += (i == 0 ? "" : i + 1 == dk.size() ? " or " : ", ") + dk[i];
    CHECK(parse_error(kRoot + Lines{"RHOGATE 0"} + kOptimizer) ==
          at_line(3, R::fill(rule->msg, {{"key", "RHOGATE"}, {"keys", keys}}),
                  "RHOGATE 0"));
}

/* Pins: [optim].positive_only is the positive-only keys of
 * hpotarget_group_specs() (MLIP), which optim_key_positive_only() reads.
 * Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the positive-only OPTIM keys are "
          "those of hpotarget_group_specs()", "[hpotarget_gen]")
{
    std::set<std::string> gen(R::optim::positive_only.begin(),
                              R::optim::positive_only.end());
    std::set<std::string> code;
    for (const auto& g : tadah::mlip::engines::hpotarget_group_specs()) {
        CHECK(HPOTargets::optim_key_positive_only(g.key) == g.positive_only);
        if (g.positive_only) code.insert(g.key);
    }
    CHECK(gen == code);
}

/* Pins: the gate lines take the options of the registry: each option of
 * RHOGATE and of RHOBARGATE without its value is refused with its check;
 * an option of the other gate alone is an unexpected token there (a check
 * of the gate's key); NPTS below its min is refused with the integer
 * reader's range message (its what, min and max), R_LO below its min, a
 * SEED word and a lattice word outside their words with their arguments'
 * messages. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the gate lines take the options of "
          "the registry", "[hpotarget_gen]")
{
    const auto refused_with = [](const std::string& line,
                                 R::Span<std::string_view> templates,
                                 std::initializer_list<R::Value> values) {
        const std::string err = parse_error(kRoot + Lines{line} + kOptimizer);
        for (std::string_view t : templates) {
            try {
                if (err == at_line(3, R::fill(t, values), line)) return true;
            } catch (const std::invalid_argument&) {
            }
        }
        return false;
    };
    const std::map<std::string, std::string> start = {
        {"RHOGATE", "RHOGATE 0"}, {"RHOBARGATE", "RHOBARGATE bcc 3.3 0.8 1.2"}};
    for (const auto& [gate, head] : start) {
        CAPTURE(gate);
        std::set<std::string> own;
        for (const R::LineOption& o : R::kLineOptions) {
            if (std::find(o.lines.begin(), o.lines.end(), gate) == o.lines.end())
                continue;
            own.insert(str(o.option));
            CAPTURE(o.option);
            const std::string bare = head + " " + str(o.option);
            CHECK(refused_with(bare, o.checks, {}));
            REQUIRE(o.args.size == 1);
            const R::Arg& a = o.args[0];
            if (a.kind == R::ArgKind::Integer) {
                REQUIRE(a.min.whole_value > 0);
                const std::string tok = std::to_string(a.min.whole_value - 1);
                const std::string l = bare + " " + tok;
                CHECK(parse_error(kRoot + Lines{l} + kOptimizer) ==
                      at_line(3, R::fill(R::reading::integer_range_msg,
                                         {{"what", a.what}, {"min", a.min.text},
                                          {"max", std::to_string(a.max.whole_value)},
                                          {"tok", tok}}),
                              l));
            } else if (a.kind == R::ArgKind::Number && a.min.set) {
                const std::string tok = core_text(a.min.value - 1);
                const std::string l = bare + " " + tok;
                CHECK(parse_error(kRoot + Lines{l} + kOptimizer) ==
                      at_line(3, R::fill(a.msg, {{"tok", tok}}), l));
            } else if (a.kind == R::ArgKind::Word) {
                const std::string l = bare + " ZZZ";
                CHECK(parse_error(kRoot + Lines{l} + kOptimizer) ==
                      at_line(3, R::fill(a.msg, {{"tok", "ZZZ"}}), l));
            }
        }
        for (const R::LineOption& o : R::kLineOptions) {
            const auto on = [&](std::string_view l) {
                return std::find(o.lines.begin(), o.lines.end(), l) !=
                       o.lines.end();
            };
            if (!(on("RHOGATE") || on("RHOBARGATE")) || own.count(str(o.option)))
                continue;
            CAPTURE(o.option);
            CHECK(refused_with(head + " " + str(o.option) + " 1",
                               root_key(gate).checks, {{"tok", o.option}}));
        }
    }
    const R::Key& rhobar = root_key("RHOBARGATE");
    const std::string l = "RHOBARGATE zzz 3.3 0.8 1.2";
    CHECK(parse_error(kRoot + Lines{l} + kOptimizer) ==
          at_line(3, R::fill(rhobar.args[0].msg, {{"tok", "zzz"}}), l));
}

/* Pins: ASA:ASA's PARAM keys are kDescs of asa_options.cpp, in order
 * (print_asa_param_table), each with its per-dimension form ([/i]) and
 * its default; its ranges are those parse_asa_options checks: a value
 * below min, at an open min, above max, and with a fraction for an
 * integral key, is refused with a check of the key; each excluded key is
 * refused with the policy's check and its reason; an unknown key with the
 * policy's message, the keys in order. Before the fix: the headers do not
 * exist. */
TEST_CASE("hpotarget generated tables: ASA:ASA's PARAM keys are kDescs, "
          "with their ranges, and kExcluded", "[hpotarget_gen]")
{
    using tadah::core::format_roundtrip;
    std::vector<const P::Param*> asa;
    for (const P::Param& p : P::kParams)
        if (std::find(p.optimizers.begin(), p.optimizers.end(), "ASA:ASA") !=
            p.optimizers.end())
            asa.push_back(&p);
    const P::ParamPolicy* policy = nullptr;
    for (const P::ParamPolicy& p : P::kParamPolicies)
        if (p.id_text == "PARAM.ASA:ASA") policy = &p;
    REQUIRE(policy != nullptr);

    std::ostringstream printed;
    tadah::hpo::print_asa_param_table(printed);
    std::istringstream table(printed.str());
    std::vector<std::string> rows;
    for (std::string row; std::getline(table, row);) rows.push_back(row);
    REQUIRE(rows.size() == asa.size());
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < asa.size(); ++i) {
        const P::Param& p = *asa[i];
        keys.push_back(str(p.name));
        CAPTURE(p.name);
        std::string head = "  PARAM " + str(p.name) + (p.indexed ? "[/i]" : "") +
                           "  (default ";
        CHECK(rows[i].substr(0, head.size()) == head);
        if (p.has_default_literal)
            CHECK(rows[i].substr(head.size(), p.default_literal.size() + 1) ==
                  str(p.default_literal) + ")");
    }
    const auto refused = [](const std::string& name, double v) {
        OptimizerSpec sp = spec_of("ASA", "ASA");
        sp.param[name] = v;
        return thrown([&] { tadah::hpo::parse_asa_options(sp, 2); });
    };
    for (const P::Param* p : asa) {
        CAPTURE(p->name);
        REQUIRE(p->args.size == 1);
        const R::Arg& a = p->args[0];
        const std::string name = str(p->name);
        const auto check = [&](double v) {
            const std::string err = refused(name, v);
            CAPTURE(v, err);
            CHECK(one_of(err, p->checks,
                         {{"key", name}, {"value", format_roundtrip(v)},
                          {"min", a.min.text}, {"max", a.max.text}},
                         true));
        };
        if (a.min.set) check(a.min_open ? a.min.value : a.min.value - 1);
        if (a.max.set) check(a.max.value + 1);
        if (a.integral) check((a.min.set ? a.min.value : 0.0) + 0.5);
        if (!a.integral)
            CHECK(refused(name, (a.min.set ? a.min.value : 0.0) + 0.5) == "");
    }
    for (const P::ParamExcluded& x : P::kParamExcluded) {
        if (std::find(x.optimizers.begin(), x.optimizers.end(), "ASA:ASA") ==
            x.optimizers.end())
            continue;
        CAPTURE(x.name);
        CHECK(one_of(refused(str(x.name), 1.0), policy->checks,
                     {{"name", x.name}, {"why", x.why}}, true));
    }
    CHECK(refused("Zzz", 1.0) ==
          R::fill(without_phase(policy->msg),
                  {{"key", "Zzz"}, {"keys", joined(keys, " ")}}));
}

/* Pins: the PARAM keys of DLIB:BFGS, CERES:STEEPEST_DESCENT, TADAH:GRID,
 * TADAH:RANDOM and DLIB:GFS are those their validators take: an unknown
 * key is refused with the policy's message, and each key of the
 * optimiser and each finite-difference key (the PARAM keys of every
 * optimiser, is_fd_param(), and no other) is not; TADAH:RANDOM and
 * DLIB:GFS take none. Before the fix: the headers do not exist. */
TEST_CASE("hpotarget generated tables: the PARAM keys of the adapters and "
          "the finite-difference keys", "[hpotarget_gen]")
{
    using namespace tadah::hpo;
    std::vector<std::string> fd;
    for (const P::Param& p : P::kParams) {
        const bool every = std::find(p.optimizers.begin(), p.optimizers.end(),
                                     "*") != p.optimizers.end();
        CAPTURE(p.name);
        CHECK(is_fd_param(str(p.name)) == every);
        if (every) fd.push_back(str(p.name));
    }
    CHECK(!fd.empty());
    struct Adapter {
        std::string policy, lib, algo;
        std::function<void(const OptimizerSpec&)> validate;
        bool takes_fd;
    };
    const std::vector<Adapter> adapters = {
        {"PARAM.DLIB:BFGS", "DLIB", "BFGS",
         [](const OptimizerSpec& s) {
             tadah::hpo::detail::validate_dlib_bfgs_spec(s);
         },
         true},
        {"PARAM.CERES", "CERES", "STEEPEST_DESCENT",
         [](const OptimizerSpec& s) {
             tadah::hpo::detail::validate_ceres_steepest_spec(s);
         },
         true},
        {"PARAM.TADAH:GRID", "TADAH", "GRID",
         [](const OptimizerSpec& s) { validate_grid_spec(s); }, false},
        {"PARAM.TADAH:RANDOM", "TADAH", "RANDOM",
         [](const OptimizerSpec& s) {
             validate_random_spec(s, RestartStop::Off);
         },
         false},
        {"PARAM.DLIB:GFS", "DLIB", "GFS",
         [](const OptimizerSpec& s) {
             validate_dlib_gfs_spec(s, RestartStop::Off);
         },
         false}};
    for (const Adapter& ad : adapters) {
        CAPTURE(ad.policy);
        const P::ParamPolicy* policy = nullptr;
        for (const P::ParamPolicy& p : P::kParamPolicies)
            if (p.id_text == ad.policy) policy = &p;
        REQUIRE(policy != nullptr);
        const std::string name = ad.lib + ":" + ad.algo;
        OptimizerSpec bogus = spec_of(ad.lib, ad.algo);
        bogus.param["zzz_bogus"] = 1.0;
        CHECK(thrown([&] { ad.validate(bogus); }) ==
              R::fill(without_phase(policy->msg),
                      {{"key", "zzz_bogus"}, {"algo", ad.algo}}));
        for (const P::Param& p : P::kParams) {
            const bool own = std::find(p.optimizers.begin(), p.optimizers.end(),
                                       name) != p.optimizers.end();
            const bool every = std::find(p.optimizers.begin(),
                                         p.optimizers.end(), "*") !=
                               p.optimizers.end();
            if (!own && !(every && ad.takes_fd)) continue;
            CAPTURE(p.name);
            OptimizerSpec sp = spec_of(ad.lib, ad.algo);
            const R::Arg& a = p.args[0];
            if (a.kind == R::ArgKind::Word)
                sp.param_raw[str(p.name)] = str(a.words[0]);
            else
                sp.param[str(p.name)] =
                    a.min.set ? a.min.value + (a.min_open ? 1.0 : 0.0) : 1.0;
            const std::string err = thrown([&] { ad.validate(sp); });
            CHECK(err.find("PARAM " + str(p.name) + ";") == std::string::npos);
            CHECK(err.find("unknown PARAM") == std::string::npos);
        }
    }
}

/* Pins: each environment override names a key that names it back
 * (Key::env); TADAH_OPTIM_SCALE, which HPOTargets reads, refuses a value
 * outside its words with its message. The overrides tadah hpo reads when
 * it starts (TADAH_PC_ISOLATION, TADAH_PC_WORKER_TIMEOUT,
 * TADAH_FAILSCORE_GRADING, TADAH_BUMPGATE) are compared with the code by
 * the registry's conformance cases, which run them. Before the fix: the
 * headers do not exist. */
TEST_CASE("hpotarget generated tables: the environment overrides name "
          "their keys", "[hpotarget_gen]")
{
    for (const R::Env& e : R::kEnvs) {
        CAPTURE(e.name);
        CHECK(key(e.key).env == e.name);
        for (std::string_view w : e.values) {
            const R::Arg& a = key(e.key).args[0];
            CHECK(std::find(a.words.begin(), a.words.end(), w) != a.words.end());
        }
    }
    const R::Env* scale = nullptr;
    for (const R::Env& e : R::kEnvs)
        if (e.name == "TADAH_OPTIM_SCALE") scale = &e;
    REQUIRE(scale != nullptr);
    const char* old = std::getenv("TADAH_OPTIM_SCALE");
    const std::optional<std::string> saved =
        old ? std::optional<std::string>(old) : std::nullopt;
    ::setenv("TADAH_OPTIM_SCALE", "zzz", 1);
    const std::string err = parse_error(kRoot + kOptimizer);
    if (saved) ::setenv("TADAH_OPTIM_SCALE", saved->c_str(), 1);
    else ::unsetenv("TADAH_OPTIM_SCALE");
    CHECK(err == R::fill(scale->msg, {{"tok", "zzz"}}));
}

#endif  // TADAH_HPOTARGET_TABLES

// test_hpotarget_registry_bindings.cpp
//
// The generated tables of the target-file registry
// (include/tadah/hpo/hpotarget_registry_gen.h and hpo_param_registry_gen.h,
// written by blueprint/gen_hpotarget.py) against the tables the parser and
// the startup checks hold by hand: each table the cases compare (listed
// below) must equal the hand-written one, in order where the order shows
// in a message.
//
// Where the hand-written table can be read (a function of the library:
// opt_attr_keys(), the OPTIM refusals of HPOTargets, LossFunction,
// hpotarget_group_specs(), the ASA help table), the two are compared entry
// by entry. Where it is local to a function (the block names, the removed
// keywords, the INNER refusals, the top-level and once-set keys, the
// options of the gate lines, the INIT methods, kDescs and kExcluded of
// asa_options.cpp, the PARAM lists of the adapters, the reader messages of
// strict_parse.cpp), each generated entry is put through the code that
// reads it (the parser of the target file, HPOTargets), and the code's
// answer, a refusal with its message, is compared with the entry's message
// filled in. The
// keys of the optimiser group, the options of PC_LAMMPS, PC_FAILPROBE and
// PC_OBSERVABLE cohesive_energy, the STRATEGY TYPE keys and the loss
// functions the code takes from the generated tables themselves (the
// handlers of hpotarget_handlers.cpp read them): their cases hold what the
// code does with them, the messages, the order and the TYPE of each key. A
// name the code knows without a generated entry is the coverage check's
// (hpo_hpotarget_registry_coverage, which reads these functions); the
// OPTIM refusals are compared both ways over every key of the blueprint.
//
// The tables the parser's engine reads besides (the header defines
// TADAH_HPOTARGET_ENGINE_TABLES once it holds them) are compared through
// the parser's refusals: the order of the steps of a key's line
// (structure::line_steps_*, Key::steps), of a line that opens a block and
// of an END line (open_order, close_order), of the checks of an END line
// (Block::close_steps) and of the end of the file (eof_order), each by a
// file that two steps refuse, refused as the earlier step refuses it; the
// line a close check names (Block::close_checks_at); the keys given again
// only after a line reads them (Repeat::ErrorUntilRead); the texts the
// parser puts into messages (Key::effect against require_strategy_key(),
// Rule::what); and the text of a PARAM key in `tadah explain ASA:ASA`
// (Param::explain against print_asa_param_table). The case of words
// (any_case) is compared by check_hpotarget_generated.py, check `case`.
//
// Tag [hpotarget_gen]. The file compiles with or without the generated
// headers (__has_include): without them it holds one case, which fails;
// with headers without TADAH_HPOTARGET_ENGINE_TABLES, the cases of the
// engine's tables are one case, which fails.
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
/* The target file parse_error() writes and reads, which its refusals name:
 * one file, written again for each case. */
const std::string& target_path()
{
    static const std::string p = (scratch() / "target.hpo").string();
    return p;
}

std::string parse_error(const Lines& lines, Context ctx = minimal_ctx())
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file = target_path();
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        for (const auto& l : lines) os << l << "\n";
    }
    return thrown([&] { HPOTargets t(ctx, file.string()); });
}

/* A refusal of line n of the file parse_error() reads, whose text is
 * `text`. */
std::string at_line(std::size_t n, const std::string& what,
                    const std::string& text)
{
    return tadah::hpo::config_line_error(target_path(), n, what, text);
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
    CHECK(config_line_error("t.hpo", 7, "W", "T") ==
          R::fill(R::lexing::error_frame, {{"line", "7"}, {"file", "t.hpo"},
                                           {"what", "W"}, {"text", "T"}}));
    /* A line without a file (a spec a program builds) is named by its
     * number alone. */
    CHECK(config_line_error("", 7, "W", "T") == "line 7: W\nLine: T");
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
 * (MAXTIME, a key of the optimiser group too) is an unknown key there.
 * The parser takes the set from the generated tables
 * (src/hpotarget_parser.cpp). Before the fix: the headers do not exist. */
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
 * of the top level, and the metric and PC_FAILPROBE lines,
 * which refuse a repeat themselves. (FAILSCORE, PC_SCRIPT_TIMEOUT, MAXTIME
 * and LOSS, set_for_lines, are of repeat error_until_read: the case "a key
 * given again after a line reads it" below.) Each such key has a valid
 * line below; a key without one fails the case.
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
         "PC_FAILPROBE --script " + script.string() + " --outvar S"}};
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

/* Pins: the STRATEGY TYPE keys of the registry are those of
 * strategy_rules.cpp, each with its home TYPE: require_strategy_key()
 * passes one under its TYPE and refuses it under another with the
 * STRATEGY block's close check (its {effect} is the code's text), and
 * passes every other key of the STRATEGY block under every TYPE. The TYPE
 * words are strategy_type_name()'s, in order. Before the fix: the headers
 * do not exist. */
TEST_CASE("hpotarget generated tables: the STRATEGY TYPE keys are "
          "strategy_rules.cpp's, each under its TYPE", "[hpotarget_gen]")
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
        for (StrategyType t : types)
            CHECK(thrown([&] {
                      tadah::hpo::require_strategy_key(str(k.name), t, true);
                  }) == "");
    }
}

/* Pins: the options of a PC_LAMMPS and of a PC_FAILPROBE line are those
 * LammpsSim reads (lammps_sim.cpp): the unknown-option refusal of each
 * line (a check of its key) lists the line's options of the registry, the
 * refused one left out, in registry order; an option of PC_FAILPROBE
 * alone on a PC_LAMMPS line is named as one. Before the fix: the headers
 * do not exist. */
TEST_CASE("hpotarget generated tables: the options of PC_LAMMPS and "
          "PC_FAILPROBE are LammpsSim's, in order", "[hpotarget_gen]")
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

#if __has_include(<tadah/mlip/hpotarget_optim.h>)

#include <tadah/mlip/hpotarget_optim.h>
#include <tadah/mlip/hpotarget_syntax.h>

/* Pins: the words MLIP's OPTIM reader holds by hand (tadah hpo and tadah
 * analysis check_bounds read their OPTIM lines with it) are the
 * registry's: the options of the lines OPTIM.AUTO are
 * optim_auto_options() (RATIO, FRAC, FLOOR, GROUP_FLOOR) and those of the
 * lines OPTIM optim_shape_tags() (LIN, LOG, ASINH), as sets; the word of
 * the form OPTIM.AUTO is kOptimAutoWord; [optim] auto_excluded_prefix is
 * kOptimAutoExcludedPrefix; [lexing] comment, continuation and
 * unfinished_msg are kTargetComment, kTargetContinuation and
 * kTargetUnfinishedLineMsg; [reading] underflow_note is
 * mlip::number_underflow_note of 1e-400. (The refusal tables of the OPTIM
 * keys, MLIP's too, are compared above through HPOTargets' forwards.)
 * Before the fix: MLIP has no OPTIM reader (the header does not exist). */
TEST_CASE("hpotarget generated tables: the words of MLIP's OPTIM reader are "
          "the registry's", "[hpotarget_gen]")
{
    namespace M = tadah::mlip;
    std::set<std::string> auto_options, tags;
    for (const R::LineOption& o : R::kLineOptions)
        for (std::string_view l : o.lines) {
            if (l == "OPTIM.AUTO") auto_options.insert(std::string(o.option));
            if (l == "OPTIM") tags.insert(std::string(o.option));
        }
    CHECK(auto_options == std::set<std::string>(M::optim_auto_options().begin(),
                                                M::optim_auto_options().end()));
    CHECK(tags == std::set<std::string>(M::optim_shape_tags().begin(),
                                        M::optim_shape_tags().end()));
    const R::Form* auto_form = nullptr;
    for (const R::Form& f : R::kForms)
        if (f.id_text == "OPTIM.AUTO") auto_form = &f;
    REQUIRE(auto_form != nullptr);
    CHECK(auto_form->word == M::kOptimAutoWord);
    CHECK(R::optim::auto_excluded_prefix == M::kOptimAutoExcludedPrefix);
    CHECK(R::lexing::comment == std::string(1, M::kTargetComment));
    CHECK(R::lexing::continuation == std::string(1, M::kTargetContinuation));
    CHECK(R::lexing::unfinished_msg ==
          std::string_view(M::kTargetUnfinishedLineMsg));
    CHECK(R::reading::underflow_note == M::number_underflow_note("1e-400"));
}

#else

TEST_CASE("hpotarget generated tables: the words of MLIP's OPTIM reader are "
          "the registry's", "[hpotarget_gen]")
{
    FAIL("tadah/mlip/hpotarget_optim.h does not exist: MLIP has no OPTIM "
         "reader");
}

#endif  // __has_include(<tadah/mlip/hpotarget_optim.h>)

#if defined(TADAH_HPOTARGET_ENGINE_TABLES)

/* ------------------------------------------ the order of the parse ---- */

namespace {

/* The line a refusal of the file parse_error() reads names, 0 for none. */
std::size_t line_of(const std::string& err)
{
    std::size_t n = 0;
    if (err.rfind("line ", 0) != 0) return 0;
    for (std::size_t i = 5; i < err.size() && std::isdigit(
                                static_cast<unsigned char>(err[i]));
         ++i)
        n = n * 10 + static_cast<std::size_t>(err[i] - '0');
    return n;
}

/* The {what} of a refusal of the file parse_error() reads: the text
 * between its file and its line; the whole text of a refusal that names
 * no line; "" for none. */
std::string what_of(const std::string& err)
{
    const std::string head = target_path() + ": ";
    const auto at = err.find(head);
    const auto end = err.rfind("\nLine: ");
    if (at == std::string::npos || end == std::string::npos || end < at)
        return err;
    return err.substr(at + head.size(), end - at - head.size());
}

/* A refusal as the line it names and its {what}: two refusals of one
 * line whose texts differ (a word more on the line) are one. */
using Refusal = std::pair<std::size_t, std::string>;

Refusal refusal_of(const std::string& err)
{
    return {line_of(err), what_of(err)};
}

template <class Step>
std::size_t position(R::Span<Step> order, Step step)
{
    for (std::size_t i = 0; i < order.size; ++i)
        if (order[i] == step) return i;
    FAIL("a step is not in its order");
    return 0;
}

/* The steps of the line of the key k: its own, else those of its place. */
R::Span<R::LineStep> steps_of(const R::Key& k)
{
    if (!k.steps.empty()) return k.steps;
    if (!k.group.empty()) return R::structure::line_steps_group;
    if (k.blocks & R::block_bit(R::BlockId::ROOT))
        return R::structure::line_steps_root;
    return R::structure::line_steps_block;
}

/* A fault of a target file: line `index` (from 0) is `faulty` where the
 * fault is, and `fixed` where it is not. */
struct Fault {
    std::size_t index;
    std::string faulty;
    std::string fixed;
};

Lines with_faults(Lines base, const std::vector<Fault>& on,
                  const std::vector<Fault>& off)
{
    for (const Fault& f : off) base.at(f.index) = f.fixed;
    for (const Fault& f : on) base.at(f.index) = f.faulty;
    return base;
}

/* Two faults of one file, each refused by one step of an order: the file
 * with both is refused as the file with the fault of the earlier step
 * alone. */
template <class Step>
struct TwoFaults {
    std::string label;
    Step a;
    Step b;
    Lines base;
    std::vector<Fault> fa;
    std::vector<Fault> fb;
};

template <class Step>
void check_two_faults(const TwoFaults<Step>& t, R::Span<Step> order)
{
    CAPTURE(t.label);
    const std::string ea = parse_error(with_faults(t.base, t.fa, t.fb));
    const std::string eb = parse_error(with_faults(t.base, t.fb, t.fa));
    CAPTURE(ea, eb);
    REQUIRE(!ea.empty());
    REQUIRE(!eb.empty());
    REQUIRE(refusal_of(ea) != refusal_of(eb));
    std::vector<Fault> both = t.fa;
    both.insert(both.end(), t.fb.begin(), t.fb.end());
    const std::string got = parse_error(with_faults(t.base, both, {}));
    const bool a_first = position(order, t.a) < position(order, t.b);
    CHECK(refusal_of(got) == refusal_of(a_first ? ea : eb));
}

/* One line that two steps of an order refuse, each with its refusal (line
 * 0 and "" for a step that reads the line). */
template <class Step>
struct OneLine {
    std::string label;
    Step a;
    Step b;
    Lines file;
    Refusal ra;
    Refusal rb;
};

template <class Step>
void check_one_line(const OneLine<Step>& t, R::Span<Step> order)
{
    CAPTURE(t.label);
    REQUIRE(t.ra != t.rb);
    const bool a_first = position(order, t.a) < position(order, t.b);
    CHECK(refusal_of(parse_error(t.file)) == (a_first ? t.ra : t.rb));
}

/* The optimiser block of a file that runs LAMMPS scripts: a
 * gradient-free one. */
const Lines kFreeOptimizer = {"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                              "MAXEVAL 10", "ENDOPTIMIZER"};

/* A LAMMPS script that sets the variables of the lines below, and one that
 * reads the variable a. */
const std::string& probe_script()
{
    static const std::string p = [] {
        const fs::path f = scratch() / "engine_probe.in";
        std::ofstream os(f);
        os << "variable E equal 1.0\nvariable S equal 0.0\n"
              "variable E2 equal 2.0\n";
        return f.string();
    }();
    return p;
}

const std::string& reads_a_script()
{
    static const std::string p = [] {
        const fs::path f = scratch() / "engine_reads_a.in";
        std::ofstream os(f);
        os << "variable E3 equal v_a+1\n";
        return f.string();
    }();
    return p;
}

/* The refusal of the key `key` unknown in the block b opened on line
 * `open_line`. */
std::string fill_unknown(const R::Block& b, const std::string& key,
                         std::size_t open_line)
{
    return R::fill(b.unknown_msg, {{"key", key},
                                   {"open_line", std::to_string(open_line)}});
}

const R::Rule& rule(std::string_view id)
{
    for (const R::Rule& r : R::kRules)
        if (r.id_text == id) return r;
    FAIL("no rule " << id);
    return R::kRules[0];
}

}  // namespace

/* Pins: structure::line_steps_root, _group and _block, and the steps of a
 * key of its own (Key::steps), are the parser's order of the steps of a
 * key's line: a key given twice whose second line has a wrong value, a
 * value too many or a wrong option is refused as the step of the two
 * that comes first refuses it -- the repeat rule (the first line's refusal
 * with only the second written twice) or the value, leftover or option
 * (the refusal of the second line alone). Each step after repeat in some
 * list, and each before it, has a case. Before the fix: the header has no
 * order of the steps (the case "the generated header holds the tables of
 * the engine" fails). */
TEST_CASE("hpotarget generated tables: the steps of a key's line are the "
          "parser's order", "[hpotarget_gen]")
{
    struct StepCase {
        std::string key;
        R::BlockId place;
        Lines before, after;
        std::string first, second;
        R::LineStep step;      // the step that refuses `second` alone
    };
    const Lines in_opt = kRoot + Lines{"OPTIMIZER", "LIB DLIB", "ALGO BFGS"};
    const Lines in_strategy =
        kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB DLIB", "ALGO BFGS"};
    const std::vector<StepCase> cases = {
        {"QFLUSH", R::BlockId::ROOT, kRoot, kOptimizer, "QFLUSH 5",
         "QFLUSH x", R::LineStep::Value},
        {"FAILSCORE", R::BlockId::ROOT, kRoot, kOptimizer, "FAILSCORE 10",
         "FAILSCORE 0", R::LineStep::Value},
        {"LOSS", R::BlockId::ROOT, kRoot, kOptimizer, "LOSS L2", "LOSS FOO",
         R::LineStep::Value},
        {"RHOGATE", R::BlockId::ROOT, kRoot, kOptimizer, "RHOGATE 0",
         "RHOGATE 0 NPTS 1", R::LineStep::Options},
        {"PC_EMAE", R::BlockId::ROOT, kRoot, kOptimizer, "PC_EMAE 0 1",
         "PC_EMAE x 1", R::LineStep::Value},
        {"PC_EMAE", R::BlockId::ROOT, kRoot, kOptimizer, "PC_EMAE 0 1",
         "PC_EMAE 0 1 --bogus 1", R::LineStep::Options},
        {"PC_ErRMSE", R::BlockId::ROOT, kRoot, kOptimizer, "PC_ErRMSE 0 1",
         "PC_ErRMSE 0 -1", R::LineStep::Value},
        {"PC_ErRMSE", R::BlockId::ROOT, kRoot, kOptimizer, "PC_ErRMSE 0 1",
         "PC_ErRMSE 0 1 --skip-above x", R::LineStep::Options},
        {"PC_ERSQ", R::BlockId::ROOT, kRoot, kOptimizer, "PC_ERSQ 0 1",
         "PC_ERSQ 0 x", R::LineStep::Value},
        {"PC_ERSQ", R::BlockId::ROOT, kRoot, kOptimizer, "PC_ERSQ 0 1",
         "PC_ERSQ 0 1 --skip-rel 1", R::LineStep::Options},
        {"MAXEVAL", R::BlockId::OPTIMIZER, in_opt, {"ENDOPTIMIZER"},
         "MAXEVAL 5", "MAXEVAL 0", R::LineStep::Value},
        {"MAXEVAL", R::BlockId::OPTIMIZER, in_opt, {"ENDOPTIMIZER"},
         "MAXEVAL 5", "MAXEVAL 5 6", R::LineStep::Leftover},
        {"LOG_HP", R::BlockId::OPTIMIZER, in_opt, {"ENDOPTIMIZER"},
         "LOG_HP 10", "LOG_HP 0.5", R::LineStep::Value},
        {"SEED", R::BlockId::INIT, in_opt + Lines{"INIT"},
         {"ENDINIT", "ENDOPTIMIZER"}, "SEED 1", "SEED x",
         R::LineStep::Value},
        {"CALIBRATION", R::BlockId::NOISE, in_opt + Lines{"NOISE"},
         {"ENDNOISE", "ENDOPTIMIZER"}, "CALIBRATION 5", "CALIBRATION 1",
         R::LineStep::Value},
        {"TYPE", R::BlockId::STRATEGY, in_strategy,
         {"ENDSTRATEGY", "ENDOPTIMIZER"}, "TYPE PLAIN", "TYPE FOO",
         R::LineStep::Value},
        {"MAX", R::BlockId::RESTART,
         in_strategy + Lines{"RESTART", "AFTER_STAGNATION 5"},
         {"ENDRESTART", "ENDSTRATEGY", "ENDOPTIMIZER"}, "MAX 2", "MAX 0",
         R::LineStep::Value},
    };
    std::set<std::pair<const R::LineStep*, R::LineStep>> covered;
    for (const StepCase& c : cases) {
        CAPTURE(c.key, c.second);
        const R::Key* k = R::find_key(c.key, c.place);
        REQUIRE(k != nullptr);
        const R::Span<R::LineStep> steps = steps_of(*k);
        covered.insert({steps.data, c.step});
        const std::size_t n = c.before.size() + 2;
        const std::string twice =
            parse_error(c.before + Lines{c.first, c.first} + c.after);
        const std::string alone =
            parse_error(c.before + Lines{"#", c.second} + c.after);
        REQUIRE(line_of(twice) == n);
        REQUIRE(line_of(alone) == n);
        REQUIRE(what_of(twice) != what_of(alone));
        const std::string got =
            parse_error(c.before + Lines{c.first, c.second} + c.after);
        CHECK(line_of(got) == n);
        CHECK(what_of(got) ==
              (position(steps, R::LineStep::Repeat) <
                       position(steps, c.step)
                   ? what_of(twice)
                   : what_of(alone)));
    }
    /* Each step of each list of steps of a place, and of a key's own, has
     * a case. */
    std::vector<R::Span<R::LineStep>> lists = {
        R::structure::line_steps_root, R::structure::line_steps_group,
        R::structure::line_steps_block};
    for (const R::Key& k : R::kKeys)
        if (!k.steps.empty()) lists.push_back(k.steps);
    for (const R::Span<R::LineStep>& l : lists)
        for (R::LineStep s : l) {
            if (s == R::LineStep::Repeat) continue;
            CAPTURE(static_cast<int>(s));
            CHECK(covered.count({l.data, s}) == 1);
        }
}

/* Pins: the keys of repeat error_until_read (FAILSCORE, PC_SCRIPT_TIMEOUT
 * and MAXTIME, LOSS) are given again only after a line that reads them: a
 * second line right after the first is refused with the key's
 * repeat_msg, and one after a PC_LAMMPS line that reads the value (no
 * --failscore, no --maxtime, a --varloss term without a loss function) is
 * read; a key of repeat error is refused with a reader between. Before
 * the fix: the header has no repeat rule error_until_read. */
TEST_CASE("hpotarget generated tables: a key given again after a line "
          "reads it is the parser's", "[hpotarget_gen]")
{
    const std::string reader =
        "PC_LAMMPS --script " + probe_script() + " --varloss E 1 1";
    const std::map<std::string, std::pair<std::string, std::string>> lines = {
        {"FAILSCORE", {"FAILSCORE 10", "FAILSCORE 20"}},
        {"PC_SCRIPT_TIMEOUT", {"PC_SCRIPT_TIMEOUT 60", "PC_SCRIPT_TIMEOUT 90"}},
        {"MAXTIME", {"MAXTIME 60", "MAXTIME 90"}},
        {"LOSS", {"LOSS L2", "LOSS L1"}}};
    std::size_t n = 0;
    for (const R::Key& k : R::kKeys) {
        if (k.repeat != R::Repeat::ErrorUntilRead) continue;
        CAPTURE(k.id_text);
        ++n;
        REQUIRE(k.blocks == R::block_bit(R::BlockId::ROOT));
        const auto it = lines.find(str(k.name));
        if (it == lines.end())
            FAIL("no lines of " << k.name << " in this test: add them");
        const auto& [l1, l2] = it->second;
        CHECK(parse_error(kRoot + Lines{l1, l2, reader} + kFreeOptimizer) ==
              at_line(4, R::fill(k.repeat_msg, {{"first_line", "3"}}), l2));
        CHECK(parse_error(kRoot + Lines{l1, reader, l2, reader} +
                          kFreeOptimizer) == "");
    }
    CHECK(n == lines.size());
    const R::Key& once = root_key("QFLUSH");
    REQUIRE(once.repeat == R::Repeat::Error);
    CHECK(parse_error(kRoot + Lines{"QFLUSH 5", reader, "QFLUSH 6"} +
                      kFreeOptimizer) ==
          at_line(5, R::fill(once.repeat_msg, {{"first_line", "3"}}),
                  "QFLUSH 6"));
}

/* Pins: Key::effect of each STRATEGY TYPE key is the {effect} of the
 * close check of STRATEGY as require_strategy_key() of strategy_rules.cpp
 * writes it: require_strategy_key() refuses the key under another TYPE
 * with the check's message filled with the key, the TYPE, no note, the
 * effect and the key's TYPE; a key of no TYPE has no effect. Before the
 * fix: the header has no effect of a key. */
TEST_CASE("hpotarget generated tables: the effect of a STRATEGY TYPE key "
          "is require_strategy_key()'s", "[hpotarget_gen]")
{
    using tadah::hpo::StrategyType;
    const std::vector<StrategyType> types = {
        StrategyType::PLAIN, StrategyType::MULTISTART, StrategyType::PRESTAGE};
    REQUIRE(std::size(R::kStrategyTypes) == types.size());
    const R::Block& strategy = block("STRATEGY");
    std::set<std::string> with_type;
    for (std::size_t h = 0; h < types.size(); ++h)
        for (R::KeyId id : R::kStrategyTypes[h].keys) {
            const R::Key& k = key(id);
            CAPTURE(k.name);
            with_type.insert(str(k.id_text));
            REQUIRE(!k.effect.empty());
            for (std::size_t u = 0; u < types.size(); ++u) {
                if (u == h) continue;
                CHECK(thrown([&] {
                          tadah::hpo::require_strategy_key(str(k.name),
                                                           types[u], true);
                      }) ==
                      R::fill(strategy.close_checks[0],
                              {{"key", k.name},
                               {"type", R::kStrategyTypes[u].name},
                               {"note", ""},
                               {"effect", k.effect},
                               {"home", R::kStrategyTypes[h].name}}));
            }
        }
    for (const R::Key& k : R::kKeys)
        if (!with_type.count(str(k.id_text))) CHECK(k.effect.empty());
}

/* Pins: Param::explain of each PARAM key of ASA:ASA is its text in the
 * PARAM table of `tadah explain ASA:ASA` (print_asa_param_table, which
 * takes it from the registry for each row of kDescs of asa_options.cpp),
 * the end of its row. Before the fix: the header has no explain of a PARAM
 * key. */
TEST_CASE("hpotarget generated tables: the explain of an ASA PARAM key is "
          "its row of the PARAM table", "[hpotarget_gen]")
{
    std::ostringstream printed;
    tadah::hpo::print_asa_param_table(printed);
    std::istringstream table(printed.str());
    std::vector<std::string> rows;
    for (std::string row; std::getline(table, row);) rows.push_back(row);
    std::size_t i = 0;
    for (const P::Param& p : P::kParams) {
        if (std::find(p.optimizers.begin(), p.optimizers.end(), "ASA:ASA") ==
            p.optimizers.end())
            continue;
        CAPTURE(p.name);
        REQUIRE(i < rows.size());
        REQUIRE(!p.explain.empty());
        const std::string tail = ")  " + str(p.explain);
        CHECK(rows[i].size() > tail.size());
        CHECK(rows[i].substr(rows[i].size() - std::min(tail.size(),
                                                       rows[i].size())) ==
              tail);
        CHECK(rows[i].rfind("  PARAM " + str(p.name), 0) == 0);
        ++i;
    }
    CHECK(i == rows.size());
}

/* Pins: Rule::what of RULE.set_once_for_optimiser is the {what} of its
 * message for each entry it gives: a line of the optimiser group, or
 * LOG_HP, in the OPTIMIZER block and again in its STRATEGY block is
 * refused with the rule's message filled with it. Before the fix: the
 * header has no what of a rule. */
TEST_CASE("hpotarget generated tables: the what of a set-once rule is the "
          "parser's", "[hpotarget_gen]")
{
    const R::Rule& r = rule("RULE.set_once_for_optimiser");
    const std::map<std::string, std::string> line_of_entry = {
        {"GROUP.OPTIMIZER", "MAXEVAL 10"}, {"OPTIMIZER.LOG_HP", "LOG_HP 10"}};
    REQUIRE(r.what.size == line_of_entry.size());
    for (const R::RuleWhat& w : r.what) {
        CAPTURE(w.id_text);
        const auto it = line_of_entry.find(str(w.id_text));
        if (it == line_of_entry.end())
            FAIL("no line of " << w.id_text << " in this test: add one");
        const std::string& line = it->second;
        const std::string k = line.substr(0, line.find(' '));
        CHECK(parse_error(kRoot + Lines{"OPTIMIZER", line, "STRATEGY",
                                        "LIB DLIB", "ALGO BFGS", line,
                                        "ENDSTRATEGY", "ENDOPTIMIZER"}) ==
              at_line(8,
                      R::fill(r.msg, {{"key", k},
                                      {"first_line", "4"},
                                      {"open", "OPTIMIZER"},
                                      {"block", "STRATEGY"},
                                      {"what", w.text}}),
                      line));
    }
}

/* Pins: structure::open_order and close_order are the parser's order of
 * the steps of a line that opens a block and of an END line: a line that
 * two steps refuse (or that one reads and the other refuses: STRATEGY
 * inside INIT is an alias of METHOD) is refused by the step of the two
 * that comes first. The pairs of steps one line can meet have a case each
 * (no block has both a repeat rule and `requires`). Before the fix: the
 * header has no order of the steps. */
TEST_CASE("hpotarget generated tables: the steps of a block's lines are "
          "the parser's order", "[hpotarget_gen]")
{
    using O = R::OpenStep;
    using C = R::CloseStep;
    const auto fill_of = [](std::string_view t,
                            std::initializer_list<R::Value> v) {
        return R::fill(t, v);
    };
    const R::Block& opt = block("OPTIMIZER");
    const R::Block& strategy = block("STRATEGY");
    const R::Block& noise = block("NOISE");
    const R::Block& explore = block("EXPLORE");
    const std::string_view open_alone = R::structure::open_takes_no_value_msg;
    REQUIRE(explore.requires_checks.size == 1);
    const Lines head = kRoot + Lines{"OPTIMIZER", "STRATEGY"};
    const std::vector<OneLine<O>> open = {
        {"OPTIMIZER inside STRATEGY", O::Parents, O::Repeat,
         head + Lines{"OPTIMIZER"},
         {5, fill_of(opt.misplaced_msg,
                     {{"parent", "STRATEGY"}, {"parent_line", "4"}})},
         {5, fill_of(opt.repeat_msg, {{"first_line", "3"}})}},
        {"a second STRATEGY block with a word", O::Repeat, O::KeyAlone,
         kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB NLOPT", "ALGO LN_SBPLX",
                       "ENDSTRATEGY", "STRATEGY X"},
         {8, fill_of(strategy.repeat_msg, {{"first_line", "4"}})},
         {8, fill_of(open_alone, {{"name", "STRATEGY"}, {"tok", "X"}})}},
        {"NOISE AUTO inside STRATEGY", O::Parents, O::KeyAlone,
         head + Lines{"NOISE AUTO"},
         {5, str(noise.misplaced_msg)},
         {5, fill_of(open_alone, {{"name", "NOISE"}, {"tok", "AUTO"}})}},
        {"EXPLORE inside OPTIMIZER", O::Parents, O::Requires,
         kRoot + Lines{"OPTIMIZER", "EXPLORE"},
         {4, str(explore.misplaced_msg)},
         {4, str(explore.requires_checks[0])}},
        {"EXPLORE X without TYPE PRESTAGE", O::Requires, O::KeyAlone,
         head + Lines{"EXPLORE X"},
         {5, str(explore.requires_checks[0])},
         {5, fill_of(open_alone, {{"name", "EXPLORE"}, {"tok", "X"}})}},
        {"STRATEGY RANDOM 2 inside INIT", O::Alias, O::Parents,
         kRoot + Lines{"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                       "MAXEVAL 10", "INIT", "STRATEGY RANDOM 2", "ENDINIT",
                       "ENDOPTIMIZER"},
         {0, ""},
         {8, str(strategy.misplaced_msg)}},
        {"STRATEGY RANDOM 2 inside INIT", O::Alias, O::KeyAlone,
         kRoot + Lines{"OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
                       "MAXEVAL 10", "INIT", "STRATEGY RANDOM 2", "ENDINIT",
                       "ENDOPTIMIZER"},
         {0, ""},
         {8, fill_of(open_alone, {{"name", "STRATEGY"}, {"tok", "RANDOM"}})}},
    };
    for (const auto& t : open) check_one_line(t, R::structure::open_order);

    const std::vector<OneLine<C>> close = {
        {"ENDOPTIMIZER X inside INIT", C::Match, C::KeyAlone,
         kRoot + Lines{"OPTIMIZER", "INIT", "ENDOPTIMIZER X"},
         {5, fill_of(R::structure::end_mismatch_msg,
                     {{"end", "ENDOPTIMIZER"},
                      {"open", "INIT"},
                      {"open_line", "4"}})},
         {5, fill_of(R::structure::end_takes_no_value_msg,
                     {{"end", "ENDOPTIMIZER"},
                      {"name", "OPTIMIZER"},
                      {"tok", "X"}})}},
    };
    for (const auto& t : close) check_one_line(t, R::structure::close_order);

    /* The END line holds its key alone before its block's checks: TOPK
     * under TYPE PLAIN and a word after ENDSTRATEGY. */
    const Lines topk = kRoot + Lines{"OPTIMIZER", "STRATEGY", "LIB NLOPT",
                                     "ALGO LN_SBPLX", "MAXEVAL 10", "TOPK 3",
                                     "ENDSTRATEGY", "ENDOPTIMIZER"};
    check_two_faults(TwoFaults<C>{"ENDSTRATEGY X and TOPK under PLAIN",
                                  C::KeyAlone, C::CloseSteps, topk,
                                  {{8, "ENDSTRATEGY X", "ENDSTRATEGY"}},
                                  {{7, "TOPK 3", "#"}}},
                     R::structure::close_order);
}

/* Pins: Block::close_steps are the checks the END line of each block runs,
 * in the parser's order, and Block::close_checks_at the line each close
 * check's refusal names. ENDSTRATEGY checks the keys of one TYPE, then a
 * MULTISTART without N_STARTS, then a RESTART block that never restarts:
 * of two such faults the earlier step's is refused, and a key the
 * STRATEGY block does not know after the fault (before ENDSTRATEGY) is
 * refused first, as the checks wait for the END line of the block that
 * lists them. The refusal of a TYPE key names the key's line
 * (key_line), that of a RESTART block its last ENDRESTART line
 * (last_end_line), that of an OPTIMIZER block without LIB and ALGO its END
 * line (end). Before the fix: the header has no close steps. */
TEST_CASE("hpotarget generated tables: the checks of an END line are the "
          "parser's", "[hpotarget_gen]")
{
    using S = R::BlockCloseStep;
    const R::Block* owner_of[6] = {};
    for (const R::Block& b : R::kBlocks)
        for (S s : b.close_steps) {
            REQUIRE(owner_of[static_cast<int>(s)] == nullptr);
            owner_of[static_cast<int>(s)] = &b;
        }
    for (const R::Block* b : owner_of) REQUIRE(b != nullptr);
    const R::Block& strategy = block("STRATEGY");
    REQUIRE(owner_of[static_cast<int>(S::StrategyTypeKeys)] == &strategy);
    /* Lines 5-17 of the STRATEGY block: its TYPE and a line for N_STARTS,
     * a line for TOPK (a PRESTAGE key), and two RESTART blocks, the first
     * with a line for MAX; the line before ENDSTRATEGY is free. Each fault
     * is on a line of its own. */
    const Lines base =
        kRoot + Lines{"OPTIMIZER", "STRATEGY", "TYPE PLAIN", "#",
                      "LIB NLOPT", "ALGO LN_SBPLX", "MAXEVAL 10", "#",
                      "RESTART", "AFTER_STAGNATION 5", "MAX 2", "ENDRESTART",
                      "RESTART", "ENDRESTART", "#", "ENDSTRATEGY",
                      "ENDOPTIMIZER"};
    Lines multi = base;
    multi[4] = "TYPE MULTISTART";
    multi[5] = "N_STARTS 3";
    const std::vector<Fault> no_n_starts = {{5, "#", "N_STARTS 3"}};
    const std::vector<Fault> topk = {{9, "TOPK 3", "#"}};
    const std::vector<Fault> no_max = {{12, "#", "MAX 2"}};
    const auto steps = strategy.close_steps;
    check_two_faults(TwoFaults<S>{"TOPK and MULTISTART without N_STARTS",
                                  S::StrategyTypeKeys, S::MultistartNStarts,
                                  multi, topk, no_n_starts},
                     steps);
    check_two_faults(TwoFaults<S>{"MULTISTART without N_STARTS and a "
                                  "RESTART without MAX",
                                  S::MultistartNStarts, S::RestartRestarts,
                                  multi, no_n_starts, no_max},
                     steps);
    check_two_faults(TwoFaults<S>{"TOPK under PLAIN and a RESTART without "
                                  "MAX",
                                  S::StrategyTypeKeys, S::RestartRestarts,
                                  base, topk, no_max},
                     steps);

    /* Each check waits for the END line of the block that lists it: an
     * unknown key on line 17, after the fault, comes first when that block
     * is STRATEGY. */
    const std::string foo =
        at_line(17, fill_unknown(strategy, "FOO", 4), "FOO 1");
    struct Waits {
        S step;
        const Lines* file;
        std::vector<Fault> fault;
    };
    for (const Waits& w : std::vector<Waits>{
             {S::StrategyTypeKeys, &base, topk},
             {S::MultistartNStarts, &multi, no_n_starts},
             {S::RestartRestarts, &base, no_max}}) {
        CAPTURE(static_cast<int>(w.step));
        std::vector<Fault> with_foo = w.fault;
        with_foo.push_back({16, "FOO 1", "#"});
        const std::string alone = parse_error(with_faults(*w.file, w.fault,
                                                          {}));
        REQUIRE(!alone.empty());
        CHECK(parse_error(with_faults(*w.file, with_foo, {})) ==
              (owner_of[static_cast<int>(w.step)] == &strategy ? foo
                                                               : alone));
    }

    /* The line each close check names. */
    const auto line_at = [](R::CheckAt at, std::size_t end,
                            std::size_t key_line, std::size_t last_end) {
        return at == R::CheckAt::End       ? end
               : at == R::CheckAt::KeyLine ? key_line
                                           : last_end;
    };
    REQUIRE(strategy.close_checks_at.size == 1);
    CHECK(line_of(parse_error(with_faults(base, topk, {}))) ==
          line_at(strategy.close_checks_at[0], 18, 10, 0));
    const R::Block& restart = block("RESTART");
    REQUIRE(restart.close_checks_at.size == 1);
    CHECK(line_of(parse_error(with_faults(base, no_max, {}))) ==
          line_at(restart.close_checks_at[0], 18, 0, 16));
    const R::Block& opt = block("OPTIMIZER");
    REQUIRE(opt.close_checks_at.size >= 1);
    CHECK(line_of(parse_error(kRoot + Lines{"OPTIMIZER", "MAXEVAL 10",
                                            "ENDOPTIMIZER"})) ==
          line_at(opt.close_checks_at[0], 5, 0, 0));
    for (const R::Block& b : R::kBlocks)
        CHECK(b.close_checks_at.size == b.close_checks.size);
}

/* Pins: structure::eof_order is the parser's order of the checks at the
 * end of the file: a file with two faults that two checks refuse is
 * refused as the file with the fault of the earlier check alone, for
 * every two checks next to each other in the order, but those no file
 * meets together (a LOSS no line reads needs a file without a metric
 * line, the LOSS of a metric target one with; PC_OBSERVABLE beside
 * PC_LAMMPS is refused before the rules of either; the resolution of an
 * unset GRAD_STRATEGY refuses nothing). Before the fix: the header has no
 * order of the checks at the end of the file. */
TEST_CASE("hpotarget generated tables: the checks at the end of the file "
          "are the parser's order", "[hpotarget_gen]")
{
    using E = R::EofStep;
    const std::string bad_dump =
        "DUMP 1 " + (scratch() / "no_such_dir" / "pots").string();
    /* Lines 3-5 hold a fault each where a case puts one. */
    const Lines plain = {"OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1", "#",
                         "#", "#", "OPTIMIZER", "LIB NLOPT",
                         "ALGO LN_SBPLX", "MAXEVAL 10", "INIT",
                         "METHOD RANDOM 2", "K_TOTAL 1", "ENDINIT",
                         "ENDOPTIMIZER"};
    std::vector<Fault> no_optimizer;
    for (std::size_t i = 5; i < plain.size(); ++i)
        no_optimizer.push_back({i, "#", plain[i]});
    const std::string observable =
        "PC_OBSERVABLE cohesive_energy --species H --target 2.0 --weight 1 "
        "--hotvar a0 1.5 2.5";
    const std::string script_line =
        "PC_LAMMPS --script " + probe_script() + " --varloss E 1 1";
    /* A PC_OBSERVABLE line, a slot for a PC_LAMMPS line (line 2) and one
     * for GRAD_STRATEGY (line 9). */
    const Lines with_observable = {"OPTIM WATOM (1) 0.1 5.0", "#",
                                   observable, "OPTIMIZER", "LIB NLOPT",
                                   "ALGO LD_LBFGS", "MAXEVAL 10", "#",
                                   "#", "ENDOPTIMIZER"};
    std::vector<Fault> no_observable_optimizer;
    for (std::size_t i = 3; i < with_observable.size(); ++i)
        no_observable_optimizer.push_back({i, "#", with_observable[i]});
    /* A PC_LAMMPS line with --hotvar a, a line without one whose script
     * reads a (line 4: the fault defines a there, --invar a) and a slot for
     * a third line (line 5). */
    const std::string cold_plain =
        "PC_LAMMPS --script " + reads_a_script() + " --varloss E3 1 1";
    const std::string cold = cold_plain + " --invar a 1";
    const Lines with_scripts = {
        "OPTIM WATOM (1) 0.1 5.0", "PC_ERMSE 0 1",
        "PC_LAMMPS --script " + probe_script() +
            " --varloss E 1 1 --hotvar a 1 2",
        cold_plain, "#", "OPTIMIZER", "LIB NLOPT", "ALGO LN_SBPLX",
        "MAXEVAL 10", "ENDOPTIMIZER"};
    const std::string hot_again = "PC_LAMMPS --script " + probe_script() +
                                  " --varloss E2 1 1 --hotvar a 1 2";
    const std::vector<TwoFaults<E>> pairs = {
        {"an open block and a K_TOTAL that caps nothing", E::OpenBlock,
         E::KTotalCaps, plain, {{13, "#", "ENDOPTIMIZER"}},
         {{11, "K_TOTAL 5", "K_TOTAL 1"}}},
        {"a K_TOTAL that caps nothing and an unread PC_SCRIPT_TIMEOUT",
         E::KTotalCaps, E::ScriptTimeoutUnread, plain,
         {{11, "K_TOTAL 5", "K_TOTAL 1"}},
         {{2, "PC_SCRIPT_TIMEOUT 60", "#"}}},
        {"an unread PC_SCRIPT_TIMEOUT and an unread LOSS",
         E::ScriptTimeoutUnread, E::LossUnread, plain,
         {{2, "PC_SCRIPT_TIMEOUT 60", "#"}},
         {{1, "#", "PC_ERMSE 0 1"}, {3, "LOSS L2", "#"}}},
        {"an unread PC_SCRIPT_TIMEOUT and a target LOSS MAPE cannot "
         "measure",
         E::ScriptTimeoutUnread, E::MetricTargetLoss, plain,
         {{2, "PC_SCRIPT_TIMEOUT 60", "#"}}, {{3, "LOSS MAPE", "#"}}},
        {"a target LOSS MAPE cannot measure and a RHOGATE no OPTIM line "
         "arms",
         E::MetricTargetLoss, E::DensityKeyArming, plain,
         {{3, "LOSS MAPE", "#"}}, {{4, "RHOGATE 0", "#"}}},
        {"a RHOGATE no OPTIM line arms and a DUMP directory that cannot "
         "be made",
         E::DensityKeyArming, E::ArchiveDirectories, plain,
         {{4, "RHOGATE 0", "#"}}, {{2, bad_dump, "#"}}},
        {"a DUMP directory that cannot be made and no OPTIMIZER block",
         E::ArchiveDirectories, E::OptimizerRequired, plain,
         {{2, bad_dump, "#"}}, no_optimizer},
        {"no OPTIMIZER block and PC_OBSERVABLE beside PC_LAMMPS",
         E::OptimizerRequired, E::ObservableWithScripts, with_observable,
         no_observable_optimizer, {{1, script_line, "#"}}},
        {"PC_OBSERVABLE beside PC_LAMMPS and GRAD_STRATEGY NUMERICAL",
         E::ObservableWithScripts, E::ObservableNumerical, with_observable,
         {{1, script_line, "#"}}, {{7, "GRAD_STRATEGY NUMERICAL", "#"}}},
        {"PC_OBSERVABLE beside PC_LAMMPS and a script that reads a hot "
         "variable",
         E::ObservableWithScripts, E::HotvarInjected, with_scripts,
         {{4, observable, "#"}}, {{3, cold, cold_plain}}},
        {"a script that reads a hot variable and a hot variable of two "
         "lines",
         E::HotvarInjected, E::HotvarUnique, with_scripts,
         {{3, cold, cold_plain}}, {{4, hot_again, "#"}}},
        {"a hot variable of two lines and a gradient-based optimiser",
         E::HotvarUnique, E::ScriptGradients, with_scripts,
         {{4, hot_again, "#"}}, {{7, "ALGO LD_LBFGS", "ALGO LN_SBPLX"}}},
    };
    const std::set<std::pair<E, E>> apart = {
        {E::LossUnread, E::MetricTargetLoss},
        {E::HotvarUnique, E::ObservableNumerical},
        {E::ObservableNumerical, E::ObservableAnalytical},
        {E::ObservableAnalytical, E::ScriptGradients}};
    std::set<std::pair<E, E>> covered;
    for (const auto& t : pairs) {
        check_two_faults(t, R::structure::eof_order);
        covered.insert({t.a, t.b});
        covered.insert({t.b, t.a});
    }
    const auto order = R::structure::eof_order;
    REQUIRE(order.size >= 2);
    for (std::size_t i = 0; i + 1 < order.size; ++i) {
        const std::pair<E, E> next{order[i], order[i + 1]};
        CAPTURE(static_cast<int>(next.first), static_cast<int>(next.second));
        CHECK((covered.count(next) == 1 || apart.count(next) == 1 ||
               apart.count({next.second, next.first}) == 1));
    }
}

#else

TEST_CASE("hpotarget generated tables: the generated header holds the "
          "tables of the engine", "[hpotarget_gen]")
{
    FAIL("include/tadah/hpo/hpotarget_registry_gen.h does not define "
         "TADAH_HPOTARGET_ENGINE_TABLES: it has no order of the steps of "
         "the parse, no case of words and no texts of messages "
         "(blueprint/gen_hpotarget.py --header writes them)");
}

#endif  // TADAH_HPOTARGET_ENGINE_TABLES

#endif  // TADAH_HPOTARGET_TABLES

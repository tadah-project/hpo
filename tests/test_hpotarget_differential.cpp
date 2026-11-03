// test_hpotarget_differential.cpp
//
// The two parsers of the target file compared: ParsedTarget::read (the
// parser of `tadah hpo`) and the registry-driven parser
// (src/hpotarget_parser.cpp). Pins:
//  - the comparison of the parser shadow (testing::read_shadowed,
//    hpo_targets_testing.h), which a validation build with the CMake option
//    TADAH_HPO_PARSER_SHADOW makes on every read of a target file: two
//    parsers that read a file alike give the targets of the first; a
//    difference in the description of the parse (describe_parse), in the
//    refusal (its text or its type), one parser refusing a file the other
//    reads, or a difference in what the parsers write to standard output
//    or standard error is refused with a std::logic_error naming both
//    parsers and the difference; equal refusals give the refusal of the
//    first parser; the second parser reads a copy of the training
//    configuration and writes nothing;
//  - the shadow is wired: in a build with the option (tests/CMakeLists.txt
//    passes it as TADAH_HPO_PARSER_SHADOW_BUILD) each construction of
//    HPOTargets compares the two parsers once, and in a build without it,
//    which is the program's, never;
//  - [.hpotarget_differential] (CTest hpo_hpotarget_differential): every
//    case of the file TADAH_HPOTARGET_DIFFERENTIAL_CASES names
//    (tests/hpotarget_differential.py writes it: every example and invalid
//    case of the registry, every case check_hpotarget_generated.py
//    generates, every input of the corpus of tests/hpotarget_corpus.py),
//    read by each parser in a directory of its own on a training
//    configuration of its own, gives the same description or the same
//    refusal, and the same refusal of the startup checks
//    (validate_hpo_setup) where the case runs them.
// Before the fix: hpo_targets_testing.h has no shadow seam
// (TADAH_HPOTARGET_PARSER_SHADOW_SEAM) and the one case of this file
// fails.

#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_targets_testing.h>

#ifndef TADAH_HPO_PARSER_SHADOW_BUILD
#error "TADAH_HPO_PARSER_SHADOW_BUILD is 1 in a build with the CMake option TADAH_HPO_PARSER_SHADOW, else 0 (tests/CMakeLists.txt)"
#endif

#ifdef TADAH_HPOTARGET_PARSER_SHADOW_SEAM

#include "tools/hpotarget_cases.h"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using tadah::hpo::ParsedTarget;
using tadah::hpo::testing::NamedParser;
using tadah::hpo::testing::TargetParser;
using tadah::hpo::testing::describe_parse;
using tadah::hpo::testing::make_targets;
using tadah::hpo::testing::parser_of;
using tadah::hpo::testing::shadow_reads;

/* The training configuration of the registry's context default
 * (blueprint/hpotarget.toml). */
const char* const kConfig =
    "ATOM H\nWATOM 1.0\nDBFILE dummy.db\nINIT2B true\nINITMB false\n"
    "TYPE2B D2_BP 2 2 H H\nCGRID2B 1.0 2.0\nSGRID2B 0.5 1.0\n"
    "RCTYPE2B Cut_Cos\nRCUT2B 5.0\nVERBOSE 0\nMODEL M_BLR BF_Linear\n"
    "OALGO 4\nLAMBDA 1e-3\n";

/* The registry's root wrapper with `top` at its slot. */
std::string root(const std::string& top = "")
{
    return top + "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\n"
                 "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
                 "ENDSTRATEGY\nENDOPTIMIZER\n";
}

/* A fresh working directory holding target.hpo, removed on exit. */
class Scratch {
public:
    explicit Scratch(const std::string& file)
    {
        static unsigned n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_parser_shadow_" + std::to_string(::getpid()) + "_" +
                std::to_string(n++) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()));
        fs::create_directories(dir_);
        old_ = fs::current_path();
        fs::current_path(dir_);
        std::ofstream("target.hpo") << file;
        tadah::core::PeriodicTable::initialize();
    }
    ~Scratch()
    {
        std::error_code ec;
        fs::current_path(old_, ec);
        fs::remove_all(dir_, ec);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

private:
    fs::path dir_, old_;
};

Context config_of(const std::string& text)
{
    std::istringstream in(text);
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* Standard output and standard error, recorded while it lives. */
class Captured {
public:
    Captured()
        : out_(std::cout.rdbuf(out.rdbuf())),
          err_(std::cerr.rdbuf(err.rdbuf()))
    {
    }
    ~Captured()
    {
        std::cout.rdbuf(out_);
        std::cerr.rdbuf(err_);
    }
    Captured(const Captured&) = delete;
    Captured& operator=(const Captured&) = delete;
    std::ostringstream out, err;

private:
    std::streambuf* out_;
    std::streambuf* err_;
};

/* The parser of `tadah hpo` under another name, and the same parser with
 * one change after it has read the file. */
NamedParser old_as(const std::string& name)
{
    return {parser_of(TargetParser::Old).read, name};
}

NamedParser changed(const std::string& name,
                    void (*change)(ParsedTarget&))
{
    return {[change](ParsedTarget& t, const std::string& fn) {
                parser_of(TargetParser::Old).read(t, fn);
                change(t);
            },
            name};
}

NamedParser refusing(const std::string& name, const std::string& what)
{
    return {[what](ParsedTarget&, const std::string&) {
                throw std::runtime_error(what);
            },
            name};
}

/* The message of the std::logic_error the shadowed read of `file` throws
 * ("" when it throws none or another exception). */
std::string shadow_difference(const NamedParser& primary,
                              const NamedParser& shadow,
                              const std::string& file = root())
{
    Scratch s(file);
    Context ctx = config_of(kConfig);
    try {
        make_targets(ctx, "target.hpo", primary, shadow);
    } catch (const std::logic_error& e) {
        return e.what();
    } catch (const std::exception&) {
    }
    return "";
}

}  // namespace

TEST_CASE("parser shadow: two parsers that read a file alike give the "
          "targets of the first, read on its own configuration",
          "[hpotarget_shadow]")
{
    std::string alone;
    {
        Scratch s(root("FAILSCORE 1000\n"));
        Context ctx = config_of(kConfig);
        alone = describe_parse(*make_targets(ctx, "target.hpo",
                                             TargetParser::Old));
    }
    for (const auto& [primary, shadow] :
         {std::pair{TargetParser::Old, TargetParser::New},
          std::pair{TargetParser::New, TargetParser::Old}}) {
        Scratch s(root("FAILSCORE 1000\n"));
        Context ctx = config_of(kConfig);
        const std::size_t before = shadow_reads();
        const auto trg = make_targets(ctx, "target.hpo", parser_of(primary),
                                      parser_of(shadow));
        CHECK(shadow_reads() > before);
        /* The context of the targets is the one the first parser read:
         * the second inserted nothing into it. */
        CHECK(describe_parse(*trg) == alone);
    }
}

TEST_CASE("parser shadow: a difference in the description of the parse is "
          "refused naming both parsers and the first different line",
          "[hpotarget_shadow]")
{
    const std::string e = shadow_difference(
        old_as("parser A"), changed("parser B", [](ParsedTarget& t) {
            t.target_file += "x";
        }));
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("target.hpo"));
    CHECK_THAT(e, Catch::Contains("parser A: target_file = \"target.hpo\""));
    CHECK_THAT(e, Catch::Contains("parser B: target_file = \"target.hpox\""));
}

TEST_CASE("parser shadow: one parser that refuses a file the other reads is "
          "refused naming both parsers and the refusal",
          "[hpotarget_shadow]")
{
    const std::string why = "line 1 of target.hpo: no";
    std::string e = shadow_difference(old_as("parser A"),
                                      refusing("parser B", why));
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("parser A: reads the file"));
    CHECK_THAT(e, Catch::Contains("parser B: refuses it: " + why));

    e = shadow_difference(refusing("parser A", why), old_as("parser B"));
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("parser A: refuses it: " + why));
    CHECK_THAT(e, Catch::Contains("parser B: reads the file"));
}

TEST_CASE("parser shadow: two refusals with different texts or types are "
          "refused naming both", "[hpotarget_shadow]")
{
    std::string e = shadow_difference(refusing("parser A", "line 1: one"),
                                      refusing("parser B", "line 1: two"));
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("parser A: refuses it: line 1: one"));
    CHECK_THAT(e, Catch::Contains("parser B: refuses it: line 1: two"));

    const NamedParser invalid{[](ParsedTarget&, const std::string&) {
                                  throw std::invalid_argument("line 1: one");
                              },
                              "parser B"};
    e = shadow_difference(refusing("parser A", "line 1: one"), invalid);
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("parser A"));
    CHECK_THAT(e, Catch::Contains("parser B"));
    CHECK_THAT(e, Catch::Contains("type"));
}

TEST_CASE("parser shadow: two equal refusals give the refusal of the first "
          "parser", "[hpotarget_shadow]")
{
    /* A file both parsers refuse: an unknown key at the top level. */
    const std::string file = root("NOSUCHKEY 1\n");
    std::string alone;
    {
        Scratch s(file);
        Context ctx = config_of(kConfig);
        try {
            HPOTargets trg(ctx, "target.hpo");
        } catch (const std::runtime_error& e) {
            alone = e.what();
        }
    }
    REQUIRE_THAT(alone, Catch::StartsWith("line 1 of target.hpo: "));
    Scratch s(file);
    Context ctx = config_of(kConfig);
    REQUIRE_THROWS_MATCHES(
        make_targets(ctx, "target.hpo", parser_of(TargetParser::Old),
                     parser_of(TargetParser::New)),
        std::runtime_error, Catch::Message(alone));
}

TEST_CASE("parser shadow: what the parsers write is compared, and the second "
          "parser's is not written", "[hpotarget_shadow]")
{
    const auto writing = [](const std::string& name, void (*note)()) {
        return NamedParser{[note](ParsedTarget& t, const std::string& fn) {
                               parser_of(TargetParser::Old).read(t, fn);
                               note();
                           },
                           name};
    };
    const auto note = [] { std::cerr << "a note\n"; };
    const auto other = [] { std::cerr << "another note\n"; };
    const auto out = [] { std::cout << "a note\n"; };
    {
        Scratch s(root());
        Context ctx = config_of(kConfig);
        Captured cap;
        make_targets(ctx, "target.hpo", writing("parser A", note),
                     writing("parser B", note));
        const std::string err = cap.err.str();
        CHECK(err.find("a note\n") != std::string::npos);
        CHECK(err.find("a note\n") == err.rfind("a note\n"));
    }
    std::string e;
    {
        Captured cap;
        e = shadow_difference(writing("parser A", note),
                              writing("parser B", other));
    }
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("standard error"));
    CHECK_THAT(e, Catch::Contains("another note"));
    {
        Captured cap;
        e = shadow_difference(writing("parser A", out), old_as("parser B"));
    }
    REQUIRE_THAT(e, Catch::Contains("TADAH_HPO_PARSER_SHADOW"));
    CHECK_THAT(e, Catch::Contains("standard output"));
}

TEST_CASE("parser shadow: a build with TADAH_HPO_PARSER_SHADOW compares the "
          "two parsers on every read of a target file, the program's build "
          "never", "[hpotarget_shadow]")
{
    const std::size_t per_read = TADAH_HPO_PARSER_SHADOW_BUILD ? 1 : 0;
    Scratch s(root());
    {
        Context ctx = config_of(kConfig);
        const std::size_t before = shadow_reads();
        HPOTargets trg(ctx, "target.hpo");
        CHECK(shadow_reads() - before == per_read);
    }
    for (const auto parser : {TargetParser::Old, TargetParser::New}) {
        Context ctx = config_of(kConfig);
        const std::size_t before = shadow_reads();
        make_targets(ctx, "target.hpo", parser);
        CHECK(shadow_reads() - before == per_read);
    }
}

namespace {

/* What one parser made of a case: the description of the parse or its
 * refusal, and the refusal of the startup checks. */
struct Outcome {
    std::string parse_error, dump, startup_error;
};

Outcome run_case(const hpotarget_cases::Case& c, TargetParser parser)
{
    Outcome o;
    tadah::core::PeriodicTable::initialize();
    const hpotarget_cases::CaseDir dir(c);
    Captured quiet;
    Context ctx = hpotarget_cases::case_config(c);
    std::unique_ptr<HPOTargets> trg;
    try {
        trg = make_targets(ctx, dir.target(), parser);
    } catch (const std::exception& e) {
        o.parse_error = e.what();
        if (o.parse_error.empty()) o.parse_error = "(empty message)";
        return o;
    }
    o.dump = describe_parse(*trg);
    if (c.mode == "startup" || c.mode == "check") {
        try {
            tadah::hpo::validate_hpo_setup(ctx, *trg, c.has_train_db,
                                           c.has_validation);
        } catch (const std::exception& e) {
            o.startup_error = e.what();
            if (o.startup_error.empty()) o.startup_error = "(empty message)";
        }
    }
    return o;
}

/* The first line where two texts differ, as "line N: a | b". */
std::string first_difference(const std::string& a, const std::string& b)
{
    std::istringstream ia(a), ib(b);
    std::string la, lb;
    for (std::size_t n = 1;; ++n) {
        const bool ga = static_cast<bool>(std::getline(ia, la));
        const bool gb = static_cast<bool>(std::getline(ib, lb));
        if (!ga && !gb) return "";
        if (!ga) la = "(end)";
        if (!gb) lb = "(end)";
        if (!ga || !gb || la != lb)
            return "line " + std::to_string(n) + ": " + la + " | " + lb;
    }
}

}  // namespace

TEST_CASE("parser differential: every case of the registry, of the generated "
          "checks and of the corpus is read alike by both parsers",
          "[.hpotarget_differential]")
{
    const char* path = std::getenv("TADAH_HPOTARGET_DIFFERENTIAL_CASES");
    if (!path || !*path)
        FAIL("TADAH_HPOTARGET_DIFFERENTIAL_CASES names no file of cases "
             "(tests/hpotarget_differential.py cases OUT writes one; CTest "
             "hpo_hpotarget_differential_cases)");
    const std::vector<hpotarget_cases::Case> cases =
        hpotarget_cases::read_cases(path);
    std::map<std::string, std::size_t> per_source;
    std::vector<std::string> differences;
    for (const auto& c : cases) {
        ++per_source[c.id.substr(0, c.id.find(':'))];
        const Outcome a = run_case(c, TargetParser::Old);
        const Outcome b = run_case(c, TargetParser::New);
        std::string why;
        if (a.parse_error != b.parse_error)
            why = "refusal: old: " + (a.parse_error.empty() ? "(reads it)"
                                                           : a.parse_error) +
                  "\n  new: " + (b.parse_error.empty() ? "(reads it)"
                                                      : b.parse_error);
        else if (a.dump != b.dump)
            why = "description, " + first_difference(a.dump, b.dump);
        else if (a.startup_error != b.startup_error)
            why = "startup refusal: old: " + a.startup_error +
                  "\n  new: " + b.startup_error;
        if (!why.empty()) differences.push_back(c.id + ": " + why);
    }
    for (const char* source : {"registry-example", "registry-invalid",
                               "generated", "corpus"}) {
        INFO("source " << source);
        CHECK(per_source[source] > 0);
    }
    std::ostringstream report;
    for (const auto& [s, n] : per_source) report << s << ": " << n << "\n";
    for (std::size_t i = 0; i < differences.size() && i < 50; ++i)
        report << differences[i] << "\n";
    INFO(report.str());
    CHECK(differences.size() == 0);
}

#else

TEST_CASE("parser shadow: hpo_targets_testing.h has the shadow seam",
          "[hpotarget_shadow]")
{
    FAIL("hpo_targets_testing.h has no read_shadowed(): the parser shadow "
         "(TADAH_HPO_PARSER_SHADOW) is not implemented");
}

#endif

// test_hpotarget_corpus.cpp
//
// The frozen corpus of target files, tests/data/hpotarget_corpus/: the
// target files of the Tadah! repositories (every *.hpo of HPO's tests/data,
// every example of the target-file registry in its wrapper, the target
// files of the tests of Tadah.MLIP and of its documentation), each with its
// training configuration and its files, and what the parser of
// `tadah hpo` gave for it when the corpus was written: the description of
// the parse (tadah::hpo::testing::describe_parse, hpo_targets_testing.h)
// or the message that refused the file. tests/hpotarget_corpus.py write
// collects the inputs, freezes them into the corpus and writes what the
// parser gives (its docstring describes the corpus).
//
// Pins: each input of corpus.toml, read by the program's parser
// (HPOTargets) in a fresh directory as the corpus describes it, gives
// exactly what the corpus holds for it, byte for byte, and the corpus holds
// both kinds of result.
//
// Each [[input]] of corpus.toml:
//   id, origin     the input and where it was taken from
//   target         a file of the corpus whose text is the target file,
//                  written as target.hpo in the case's directory, and
//   config         a file of the corpus whose text is the training
//                  configuration; or
//   target_file,   the target file and the training configuration read in
//   config_file    place, paths relative to the working directory
//   tree           optional: a directory of the corpus copied into the
//                  case's directory first (the files the target file
//                  reads)
//   cwd, copies, config_drop, config_add, env
//                  as the cases of hpotarget_case_runner
//                  (tools/hpotarget_case_runner.cpp)
//   expected       the file of the corpus that holds the result: <x>.dump
//                  the description of the parse, <x>.error the refusal
//
// Tag [hpotarget_corpus]. The corpus directory is
// TADAH_HPOTARGET_CORPUS_DIR (tests/CMakeLists.txt).
// Before the expectations: the corpus has no corpus.toml and the case
// fails.

#include "catch2/catch.hpp"

#include "tools/hpotarget_cases.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/hpo_targets_testing.h>

#include <toml.hpp>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TADAH_HPOTARGET_CORPUS_DIR
#error "TADAH_HPOTARGET_CORPUS_DIR names tests/data/hpotarget_corpus (tests/CMakeLists.txt)"
#endif

namespace {

namespace fs = std::filesystem;
using hpotarget_cases::Case;

const fs::path& corpus_dir()
{
    static const fs::path d = TADAH_HPOTARGET_CORPUS_DIR;
    return d;
}

/* The text of the corpus file `rel`. */
std::string corpus_text(const std::string& rel)
{
    const fs::path p = corpus_dir() / rel;
    std::ifstream in(p, std::ios::binary);
    if (!in)
        throw std::runtime_error("the corpus has no file " + rel);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

struct Input {
    std::string origin;
    std::string expected;  // the corpus file of the result
    Case c;
};

std::string field(const toml::value& t, const std::string& k)
{
    return hpotarget_cases::str(t, k);
}

/* The inputs of corpus.toml, as cases of hpotarget_cases.h. */
std::vector<Input> read_corpus()
{
    const toml::value doc =
        toml::parse((corpus_dir() / "corpus.toml").string());
    std::vector<Input> out;
    if (!doc.contains("input")) return out;
    for (const auto& t : doc.at("input").as_array()) {
        Input in;
        Case& c = in.c;
        c.id = field(t, "id");
        c.mode = "describe";
        in.origin = field(t, "origin");
        in.expected = field(t, "expected");
        if (in.expected.empty())
            throw std::runtime_error("input " + c.id + " names no expected "
                                     "result");
        const std::string target = field(t, "target");
        const std::string config = field(t, "config");
        c.target_file = field(t, "target_file");
        c.config_file = field(t, "config_file");
        if (target.empty() == c.target_file.empty() ||
            config.empty() == c.config_file.empty())
            throw std::runtime_error("input " + c.id + ": one of target and "
                                     "target_file, and one of config and "
                                     "config_file");
        if (!target.empty()) c.file = corpus_text(target);
        if (!config.empty()) c.config = corpus_text(config);
        const std::string tree = field(t, "tree");
        if (!tree.empty()) {
            if (!fs::is_directory(corpus_dir() / tree))
                throw std::runtime_error("input " + c.id + ": the corpus "
                                         "has no directory " + tree);
            c.tree = (corpus_dir() / tree).string();
        }
        c.cwd = field(t, "cwd");
        if (t.contains("config_drop"))
            for (const auto& k : t.at("config_drop").as_array())
                c.config_drop.push_back(k.as_string().str);
        if (t.contains("config_add"))
            for (const auto& [k, vs] : t.at("config_add").as_table())
                for (const auto& v : vs.as_array())
                    c.config_add[k].push_back(v.as_string().str);
        if (t.contains("env"))
            for (const auto& [k, v] : t.at("env").as_table())
                c.env[k] = v.as_string().str;
        if (t.contains("copies"))
            for (const auto& f : t.at("copies").as_array())
                c.copies.push_back({hpotarget_cases::str(f, "from"),
                                    hpotarget_cases::str(f, "to")});
        out.push_back(std::move(in));
    }
    return out;
}

/* What the parser of `tadah hpo` gives for the input: "<x>.dump" and the
 * description of the parse, or "<x>.error" and the refusal. */
std::pair<std::string, std::string> read_input(const Case& c)
{
    tadah::core::PeriodicTable::initialize();
    const hpotarget_cases::CaseDir dir(c);
    tadah::core::Context ctx = hpotarget_cases::case_config(c);
    std::unique_ptr<tadah::hpo::HPOTargets> t;
    try {
        t = std::make_unique<tadah::hpo::HPOTargets>(ctx, dir.target());
    } catch (const std::exception& e) {
        return {".error", e.what()};
    }
    return {".dump", tadah::hpo::testing::describe_parse(*t)};
}

bool ends_with(const std::string& s, const std::string& end)
{
    return s.size() >= end.size() &&
           s.compare(s.size() - end.size(), end.size(), end) == 0;
}

/* The first line where `want` and `got` differ, both shown. */
std::string first_difference(const std::string& want, const std::string& got)
{
    std::istringstream a(want), b(got);
    std::string la, lb;
    for (std::size_t n = 1;; ++n) {
        const bool ha = static_cast<bool>(std::getline(a, la));
        const bool hb = static_cast<bool>(std::getline(b, lb));
        if (!ha && !hb)
            return "they differ in their last newline";
        if (!ha || !hb || la != lb)
            return "line " + std::to_string(n) + ": the corpus holds " +
                   (ha ? "'" + la + "'" : "nothing") + ", the parser gives " +
                   (hb ? "'" + lb + "'" : "nothing");
    }
}

}  // namespace

TEST_CASE("hpotarget corpus: every input of the frozen corpus reads as the "
          "corpus holds", "[hpotarget_corpus]")
{
    if (!fs::exists(corpus_dir() / "corpus.toml"))
        FAIL("the frozen corpus " << corpus_dir().string()
             << " has no corpus.toml: tests/hpotarget_corpus.py write "
                "writes it");
    const std::vector<Input> inputs = read_corpus();
    REQUIRE(!inputs.empty());
    std::size_t dumps = 0, errors = 0;
    std::vector<std::string> wrong;
    for (const Input& in : inputs) {
        const auto [kind, got] = read_input(in.c);
        if (!ends_with(in.expected, ".dump") &&
            !ends_with(in.expected, ".error")) {
            wrong.push_back(in.c.id + ": the expected result " +
                            in.expected + " is neither a .dump nor an "
                            ".error file");
            continue;
        }
        const std::string want = corpus_text(in.expected);
        (ends_with(in.expected, ".dump") ? dumps : errors) += 1;
        if (!ends_with(in.expected, kind)) {
            wrong.push_back(in.c.id + " (" + in.origin + "): the corpus "
                            "holds " + in.expected + ", the parser " +
                            (kind == ".dump" ? "reads the file"
                                             : "refuses it: " + got));
        } else if (got != want) {
            wrong.push_back(in.c.id + " (" + in.origin + "): " +
                            in.expected + ", " +
                            first_difference(want, got));
        }
    }
    INFO(inputs.size() << " inputs, " << dumps << " read, " << errors
                       << " refused");
    for (const std::string& w : wrong) UNSCOPED_INFO(w);
    CHECK(wrong.empty());
    CHECK(dumps > 0);
    CHECK(errors > 0);
}

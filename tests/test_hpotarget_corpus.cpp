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
// exactly what the corpus holds for it, byte for byte: the description or
// the refusal, and what the constructor prints on std::cout and on
// std::cerr (nothing when the corpus names no file for the stream); the
// corpus holds both kinds of result; an environment override of the registry
// (hpotarget_registry_gen.h, kEnvs) set where the test runs does not reach
// an input that does not set it (each input is read with the environment
// the corpus gives it).
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
//   stdout, stderr optional: the file of the corpus that holds what the
//                  constructor printed on the stream (<x>.stdout,
//                  <x>.stderr); without it, it printed nothing there
//
// Tag [hpotarget_corpus]. The corpus directory is
// TADAH_HPOTARGET_CORPUS_DIR (tests/CMakeLists.txt).
// Before the expectations: the corpus has no corpus.toml and the case
// fails. Before the fix (the printed output in the corpus): fails, the
// inputs whose parse prints a line have no stdout or stderr file.

#include "catch2/catch.hpp"

#include "tools/hpotarget_cases.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/hpo_targets_testing.h>
#include <tadah/hpo/hpotarget_registry_gen.h>

#include <toml.hpp>

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <string>
#include <utility>
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
    std::string out, err;  // the corpus files of the printed output
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
        in.out = field(t, "stdout");
        in.err = field(t, "stderr");
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

/* The environment overrides of the registry that the input `c` does not
 * set, unset while it lives (and restored after). */
class OverridesUnset {
public:
    explicit OverridesUnset(const Case& c)
    {
        for (const auto& e : tadah::hpo::hpotarget_registry::kEnvs) {
            const std::string name(e.name);
            if (c.env.count(name)) continue;
            const char* old = std::getenv(name.c_str());
            if (!old) continue;
            saved_.emplace_back(name, old);
            ::unsetenv(name.c_str());
        }
    }
    ~OverridesUnset()
    {
        for (const auto& [k, v] : saved_) ::setenv(k.c_str(), v.c_str(), 1);
    }
    OverridesUnset(const OverridesUnset&) = delete;
    OverridesUnset& operator=(const OverridesUnset&) = delete;

private:
    std::vector<std::pair<std::string, std::string>> saved_;
};

/* std::cout and std::cerr written into strings while it lives. */
class Printed {
public:
    Printed()
        : old_out_(std::cout.rdbuf(out_.rdbuf())),
          old_err_(std::cerr.rdbuf(err_.rdbuf()))
    {}
    ~Printed()
    {
        std::cout.rdbuf(old_out_);
        std::cerr.rdbuf(old_err_);
    }
    Printed(const Printed&) = delete;
    Printed& operator=(const Printed&) = delete;
    std::string out() const { return out_.str(); }
    std::string err() const { return err_.str(); }

private:
    std::ostringstream out_, err_;
    std::streambuf* old_out_;
    std::streambuf* old_err_;
};

/* What the parser of `tadah hpo` gives for the input. */
struct Read {
    std::string kind;  // ".dump" or ".error"
    std::string text;  // the description of the parse, or the refusal
    std::string out, err;  // what the constructor printed
};

Read read_input(const Case& c)
{
    tadah::core::PeriodicTable::initialize();
    const OverridesUnset unset(c);
    const hpotarget_cases::CaseDir dir(c);
    tadah::core::Context ctx = hpotarget_cases::case_config(c);
    std::unique_ptr<tadah::hpo::HPOTargets> t;
    Read r;
    {
        const Printed printed;
        try {
            t = std::make_unique<tadah::hpo::HPOTargets>(ctx, dir.target());
        } catch (const std::exception& e) {
            r.kind = ".error";
            r.text = e.what();
        }
        r.out = printed.out();
        r.err = printed.err();
    }
    if (t) {
        r.kind = ".dump";
        r.text = tadah::hpo::testing::describe_parse(*t);
    }
    return r;
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

/* The variable `name` set to `value` while it lives (and restored). */
class EnvSet {
public:
    EnvSet(std::string name, const std::string& value) : name_(std::move(name))
    {
        if (const char* old = std::getenv(name_.c_str())) old_ = old;
        ::setenv(name_.c_str(), value.c_str(), 1);
    }
    ~EnvSet()
    {
        if (old_) ::setenv(name_.c_str(), old_->c_str(), 1);
        else      ::unsetenv(name_.c_str());
    }
    EnvSet(const EnvSet&) = delete;
    EnvSet& operator=(const EnvSet&) = delete;

private:
    std::string name_;
    std::optional<std::string> old_;
};

/* Every input of the corpus read and compared with what the corpus holds:
 * the inputs whose result differs, and the number of each result. */
void check_corpus()
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
        const Read r = read_input(in.c);
        const std::string& kind = r.kind;
        const std::string& got = r.text;
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
        const std::pair<const char*, std::pair<std::string, std::string>>
            streams[] = {{"stdout", {in.out, r.out}},
                         {"stderr", {in.err, r.err}}};
        for (const auto& [name, files] : streams) {
            const auto& [file, printed] = files;
            const std::string held = file.empty() ? "" : corpus_text(file);
            if (printed != held)
                wrong.push_back(in.c.id + " (" + in.origin + "): " + name +
                                (file.empty() ? " (the corpus holds "
                                                "nothing)"
                                              : " (" + file + ")") +
                                ", " + first_difference(held, printed));
        }
    }
    INFO(inputs.size() << " inputs, " << dumps << " read, " << errors
                       << " refused");
    const std::size_t shown = 25;
    for (std::size_t i = 0; i < wrong.size() && i < shown; ++i)
        UNSCOPED_INFO(wrong[i]);
    if (wrong.size() > shown)
        UNSCOPED_INFO("... and " << wrong.size() - shown << " more inputs");
    CHECK(wrong.empty());
    CHECK(dumps > 0);
    CHECK(errors > 0);
}

}  // namespace

TEST_CASE("hpotarget corpus: every input of the frozen corpus reads as the "
          "corpus holds", "[hpotarget_corpus]")
{
    check_corpus();
}

/* Pins: TADAH_OPTIM_SCALE=NONE (an environment override the constructor of
 * the targets reads) set where the test runs changes no result: the inputs
 * that do not set it read without it.
 * Before the fix: fails, every input without the override reads with
 * OPTIM_SCALE NONE (scale_policy.normalise = false). */
TEST_CASE("hpotarget corpus: an environment override set outside the corpus "
          "does not reach its inputs", "[hpotarget_corpus]")
{
    const EnvSet scale("TADAH_OPTIM_SCALE", "NONE");
    check_corpus();
}

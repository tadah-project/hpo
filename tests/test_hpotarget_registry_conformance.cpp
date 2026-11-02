// test_hpotarget_registry_conformance.cpp
//
// The target-file registry (blueprint/hpotarget.toml and the part files its
// meta.parts lists) against the parser: every `examples` entry of the
// registry, put into its wrapper, is accepted by HPOTargets (and, unless the
// entry says check = "parse", by validate_hpo_setup), and every `invalid`
// entry is refused with exactly its `error`, framed as the parser frames it
// ("line N of target.hpo: ...\nLine: ...") at the line its `at` names, or
// unframed with at = "none". blueprint/README.md ("The target-file
// registry") describes the fields.
//
// The registry is read at run time with toml11, so a change to it needs no
// rebuild. TADAH_HPOTARGET_REGISTRY_DIR names another directory to read it
// from. One test case per file, each with a tag of its own, so one file can
// be checked alone (the tags: [hpotarget_main] for hpotarget.toml and its
// wrappers, [hpotarget_optimizer], [hpotarget_lines], [hpotarget_params];
// [hpotarget_registry] selects every case of this file):
//     test_hpo "[hpotarget_lines]"
//     ctest -R "hpotarget registry conformance: hpotarget_lines.toml"
// A part file is read together with hpotarget.toml (its wrappers, training
// configurations and fixture files). A file with no example or no invalid
// entry fails: it checks nothing.
//
// Each case runs in a fresh scratch directory, the working directory while
// it runs: the fixture files of the files read are written there, and so is
// the target file, target.hpo.

#include "catch2/catch.hpp"

#include <toml.hpp>

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/startup_validation.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <variant>
#include <vector>

#ifndef TADAH_HPO_BLUEPRINT_DIR
#error "TADAH_HPO_BLUEPRINT_DIR names HPO/blueprint (tests/CMakeLists.txt)"
#endif

namespace {

namespace fs = std::filesystem;

const char* const kMainFile = "hpotarget.toml";
/* The target file of every case, as the cases name it to HPOTargets and
 * the errors name it ({file} of the error frame). */
const char* const kTargetFile = "target.hpo";
/* The part files this test has a case for, in meta.parts order. */
const std::vector<std::string> kParts = {
    "hpotarget_optimizer.toml", "hpotarget_lines.toml",
    "hpotarget_params.toml"};

fs::path registry_dir()
{
    if (const char* d = std::getenv("TADAH_HPOTARGET_REGISTRY_DIR"))
        if (*d) return d;
    return TADAH_HPO_BLUEPRINT_DIR;
}

struct Wrapper {
    std::string text;
    std::string context = "default";
};
struct TrainingConfig {
    std::string config;
    bool has_train_db = true;
    bool has_validation = true;
};
struct Fixture {
    std::string path;
    std::string text;
};

struct Case {
    std::string part;
    std::string owner;   // the TOML path of the entry
    std::string kind;    // "example" or "invalid"
    std::size_t index = 0;
    std::string wrap;
    std::string text;
    std::string check;   // "parse" or "startup"
    std::optional<std::string> context;
    std::map<std::string, std::string> env;
    std::string error;
    std::variant<long, std::string> at = 1L;

    std::string label() const
    {
        return part + ": " + owner + " " + kind + " " +
               std::to_string(index) + " (wrap " + wrap + ")";
    }
};

struct Registry {
    std::string error_frame;
    std::vector<std::string> parts;   // meta.parts
    std::map<std::string, Wrapper> wrappers;
    std::map<std::string, TrainingConfig> contexts;
    std::map<std::string, Fixture> fixtures;
    std::vector<Case> cases;
};

const toml::value* find(const toml::value& t, const std::string& k)
{
    if (!t.is_table() || !t.contains(k)) return nullptr;
    return &t.at(k);
}

std::string str(const toml::value& t, const std::string& k,
                const std::string& dflt = "")
{
    const toml::value* v = find(t, k);
    if (!v) return dflt;
    if (!v->is_string())
        throw std::runtime_error("field '" + k + "' is not a string");
    return v->as_string().str;
}

bool flag(const toml::value& t, const std::string& k, bool dflt)
{
    const toml::value* v = find(t, k);
    if (!v) return dflt;
    if (!v->is_boolean())
        throw std::runtime_error("field '" + k + "' is not a boolean");
    return v->as_boolean();
}

/* The table's keys, sorted (toml11 keeps a table in a hash map). */
std::vector<std::string> sorted_keys(const toml::value& t)
{
    std::vector<std::string> out;
    for (const auto& kv : t.as_table()) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

Case read_case(const toml::value& t, const std::string& part,
               const std::string& owner, const std::string& kind,
               std::size_t index)
{
    Case c;
    c.part = part;
    c.owner = owner;
    c.kind = kind;
    c.index = index;
    c.wrap = str(t, "wrap");
    c.text = str(t, "text");
    c.check = str(t, "check", kind == "example" ? "startup" : "parse");
    if (find(t, "context")) c.context = str(t, "context");
    if (const toml::value* env = find(t, "env"))
        for (const auto& k : sorted_keys(*env))
            c.env[k] = str(*env, k);
    c.error = str(t, "error");
    if (const toml::value* at = find(t, "at")) {
        if (at->is_integer()) c.at = static_cast<long>(at->as_integer());
        else c.at = at->as_string().str;
    }
    return c;
}

/* Every examples and invalid entry under `v`, in file order for arrays and
 * name order for tables. */
void collect_cases(const toml::value& v, const std::string& path,
                   const std::string& part, std::vector<Case>& out)
{
    if (v.is_array()) {
        const auto& a = v.as_array();
        for (std::size_t i = 0; i < a.size(); ++i)
            collect_cases(a[i], path + "[" + std::to_string(i) + "]", part,
                          out);
        return;
    }
    if (!v.is_table()) return;
    std::string owner = path;
    if (const toml::value* id = find(v, "id"); id && id->is_string())
        owner += " " + id->as_string().str;
    else if (const toml::value* nm = find(v, "name"); nm && nm->is_string())
        owner += " " + nm->as_string().str;
    for (const char* kind : {"examples", "invalid"}) {
        const toml::value* list = find(v, kind);
        if (!list) continue;
        const auto& a = list->as_array();
        for (std::size_t i = 0; i < a.size(); ++i)
            out.push_back(read_case(a[i], part, owner,
                                    std::string(kind) == "examples"
                                        ? "example" : "invalid", i));
    }
    for (const auto& k : sorted_keys(v)) {
        if (k == "examples" || k == "invalid") continue;
        if (path.empty() && (k == "wrapper" || k == "context" ||
                             k == "fixture" || k == "meta"))
            continue;
        collect_cases(v.at(k), path.empty() ? k : path + "." + k, part, out);
    }
}

toml::value parse_file(const std::string& name)
{
    const fs::path p = registry_dir() / name;
    if (!fs::exists(p))
        throw std::runtime_error(p.string() + ": file not found");
    return toml::parse(p.string());
}

/* hpotarget.toml and the part files `parts`; the cases of `case_files`
 * only. */
Registry load(const std::vector<std::string>& parts,
              const std::vector<std::string>& case_files)
{
    Registry reg;
    std::vector<std::string> files = {kMainFile};
    files.insert(files.end(), parts.begin(), parts.end());
    for (const auto& file : files) {
        const toml::value doc = parse_file(file);
        if (file == kMainFile) {
            reg.error_frame = str(doc.at("lexing"), "error_frame");
            for (const auto& p : doc.at("meta").at("parts").as_array())
                reg.parts.push_back(p.as_string().str);
        }
        if (const toml::value* w = find(doc, "wrapper"))
            for (const auto& name : sorted_keys(*w))
                reg.wrappers[name] = {str(w->at(name), "text"),
                                      str(w->at(name), "context", "default")};
        if (const toml::value* c = find(doc, "context"))
            for (const auto& name : sorted_keys(*c))
                reg.contexts[name] = {str(c->at(name), "config"),
                                      flag(c->at(name), "has_train_db", true),
                                      flag(c->at(name), "has_validation",
                                           true)};
        if (const toml::value* f = find(doc, "fixture"))
            for (const auto& name : sorted_keys(*f))
                reg.fixtures[name] = {str(f->at(name), "path"),
                                      str(f->at(name), "text")};
        if (std::find(case_files.begin(), case_files.end(), file) !=
            case_files.end())
            collect_cases(doc, "", file, reg.cases);
    }
    return reg;
}

/* ---- the target file of a case ---------------------------------------- */

std::vector<std::string> split_lines(std::string s)
{
    if (!s.empty() && s.back() == '\n') s.pop_back();
    std::vector<std::string> out;
    std::string::size_type from = 0;
    while (true) {
        const auto nl = s.find('\n', from);
        out.push_back(s.substr(from, nl == std::string::npos
                                         ? std::string::npos : nl - from));
        if (nl == std::string::npos) break;
        from = nl + 1;
    }
    return out;
}

struct Composed {
    std::string file;        // the target file's text
    std::size_t slot = 1;    // physical line of the first line of the text
    std::string context;
};

Composed compose(const Registry& reg, const Case& c)
{
    Composed out;
    if (c.wrap == "none") {
        out.file = c.text;
        out.context = c.context.value_or("default");
    } else {
        const auto w = reg.wrappers.find(c.wrap);
        if (w == reg.wrappers.end())
            throw std::runtime_error("wrapper '" + c.wrap + "' is not in the "
                                     "files read");
        const auto lines = split_lines(w->second.text);
        const auto slot = std::find(lines.begin(), lines.end(), "{line}");
        if (slot == lines.end())
            throw std::runtime_error("wrapper '" + c.wrap + "' has no slot");
        out.slot = static_cast<std::size_t>(slot - lines.begin()) + 1;
        std::string text;
        for (auto it = lines.begin(); it != lines.end(); ++it) {
            if (it != lines.begin()) text += "\n";
            text += it == slot ? c.text : *it;
        }
        out.file = text;
        out.context = c.context.value_or(w->second.context);
    }
    if (out.file.empty() || out.file.back() != '\n') out.file += "\n";
    return out;
}

/* A logical line as HPOTargets::read() reads it: its first and last
 * physical lines, its text (comments cut, trailing blanks removed,
 * continued lines joined) and its key. */
struct Logical {
    std::size_t first = 0;
    std::size_t last = 0;
    std::string text;
    std::string key;
};

std::vector<Logical> logical_lines(const std::string& file)
{
    auto cut = [](std::string& s) {
        if (const auto p = s.find('#'); p != std::string::npos) s.resize(p);
        while (!s.empty() &&
               std::isspace(static_cast<unsigned char>(s.back())))
            s.pop_back();
    };
    std::vector<Logical> out;
    std::istringstream in(file);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line)) {
        ++n;
        cut(line);
        if (line.empty()) continue;
        Logical l;
        l.first = n;
        /* A last line that ends with the backslash keeps it, as the
         * frame of its refusal shows the line as written
         * (hpotarget_registry.logical_lines). */
        while (!line.empty() && line.back() == '\\') {
            std::string more;
            if (!std::getline(in, more)) break;
            line.pop_back();
            ++n;
            cut(more);
            line += more;
        }
        l.last = n;
        l.text = line;
        std::istringstream ks(line);
        ks >> l.key;
        if (!l.key.empty()) out.push_back(l);
    }
    return out;
}

std::string replace_all(std::string s, const std::string& from,
                        const std::string& to)
{
    for (auto p = s.find(from); p != std::string::npos;
         p = s.find(from, p + to.size()))
        s.replace(p, from.size(), to);
    return s;
}

/* The message the parser gives for an invalid case. */
std::string expected_error(const Registry& reg, const Case& c,
                           const Composed& comp)
{
    if (std::holds_alternative<std::string>(c.at) &&
        std::get<std::string>(c.at) == "none")
        return c.error;
    const auto lines = logical_lines(comp.file);
    const Logical* hit = nullptr;
    long k = 0;
    for (const auto& l : lines) {
        if (l.first < comp.slot) continue;
        if (std::holds_alternative<long>(c.at)) {
            if (++k == std::get<long>(c.at)) { hit = &l; break; }
        } else if (l.key == std::get<std::string>(c.at)) {
            hit = &l;
            break;
        }
    }
    if (!hit)
        throw std::runtime_error("at names no line of the file");
    std::string msg = reg.error_frame;
    msg = replace_all(msg, "{line}", std::to_string(hit->last));
    msg = replace_all(msg, "{file}", kTargetFile);
    msg = replace_all(msg, "{text}", hit->text);
    return replace_all(msg, "{what}", c.error);
}

/* ---- running a case --------------------------------------------------- */

/* A fresh directory, the working directory while it lives. */
class ScratchCwd {
public:
    ScratchCwd()
    {
        static unsigned long n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_hpotarget_registry_" + std::to_string(::getpid()) +
                "_" + std::to_string(n++) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()));
        fs::create_directories(dir_);
        old_ = fs::current_path();
        fs::current_path(dir_);
    }
    ~ScratchCwd()
    {
        std::error_code ec;
        fs::current_path(old_, ec);
        fs::remove_all(dir_, ec);
    }
    ScratchCwd(const ScratchCwd&) = delete;
    ScratchCwd& operator=(const ScratchCwd&) = delete;

private:
    fs::path dir_, old_;
};

/* The environment variables of a case, restored when it ends. */
class ScopedEnv {
public:
    explicit ScopedEnv(const std::map<std::string, std::string>& env)
    {
        for (const auto& [k, v] : env) {
            const char* old = std::getenv(k.c_str());
            saved_.emplace_back(k, old ? std::optional<std::string>(old)
                                       : std::nullopt);
            ::setenv(k.c_str(), v.c_str(), 1);
        }
    }
    ~ScopedEnv()
    {
        for (const auto& [k, v] : saved_) {
            if (v) ::setenv(k.c_str(), v->c_str(), 1);
            else   ::unsetenv(k.c_str());
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

struct Outcome {
    std::string parse_error;     // "" when HPOTargets reads the file
    std::string startup_error;   // "" when validate_hpo_setup accepts it
};

/* What hpo_run with CHECK true (`tadah hpo --check`) throws for the file
 * of `comp`, "" when it returns: an invalid case with check = "check"
 * shows a refusal the run makes before it reads its data (the validation
 * configuration has the DBFILE lines of the training configuration, whose
 * data the check never reaches: a validation configuration holds DBFILE,
 * EFILTER and FFILTER). */
std::string run_check(const Registry& reg, const Composed& comp,
                      const std::map<std::string, std::string>& env)
{
    const auto ctx_it = reg.contexts.find(comp.context);
    if (ctx_it == reg.contexts.end())
        throw std::runtime_error("context '" + comp.context + "' is not in "
                                 "the files read");
    tadah::core::PeriodicTable::initialize();
    ScratchCwd cwd;
    for (const auto& [name, f] : reg.fixtures) {
        const fs::path p(f.path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        std::ofstream os(p);
        os << f.text;
    }
    {
        std::ofstream os(kTargetFile);
        os << comp.file;
    }
    {
        std::ofstream os("config.val");
        std::istringstream lines(ctx_it->second.config);
        for (std::string line; std::getline(lines, line);) {
            std::istringstream words(line);
            std::string key;
            if (words >> key && key == "DBFILE") os << line << "\n";
        }
    }
    ScopedEnv scoped(env);
    std::istringstream config(ctx_it->second.config);
    tadah::core::Context ctx = tadah::core::CoreUserConfigParser().parse(config);
    ctx.add("HPOTARGET", kTargetFile);
    ctx.add("VALIDATION", "config.val");
    ctx.add("CHECK", "true");
    try {
        tadah::hpo::hpo_run(ctx);
    } catch (const std::exception& e) {
        return e.what()[0] ? e.what() : "(empty message)";
    }
    return "";
}

Outcome run_file(const Registry& reg, const Composed& comp, bool startup,
                 const std::map<std::string, std::string>& env)
{
    const auto ctx_it = reg.contexts.find(comp.context);
    if (ctx_it == reg.contexts.end())
        throw std::runtime_error("context '" + comp.context + "' is not in "
                                 "the files read");
    const TrainingConfig& tc = ctx_it->second;
    tadah::core::PeriodicTable::initialize();
    ScratchCwd cwd;
    for (const auto& [name, f] : reg.fixtures) {
        const fs::path p(f.path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        std::ofstream os(p);
        os << f.text;
    }
    {
        std::ofstream os(kTargetFile);
        os << comp.file;
    }
    ScopedEnv scoped(env);
    std::istringstream config(tc.config);
    tadah::core::Context ctx;
    try {
        ctx = tadah::core::CoreUserConfigParser().parse(config);
    } catch (const std::exception& e) {
        throw std::runtime_error("context '" + comp.context +
                                 "' does not parse: " + e.what());
    }
    Outcome out;
    std::optional<tadah::hpo::HPOTargets> trg;
    try {
        trg.emplace(ctx, kTargetFile);
    } catch (const std::exception& e) {
        out.parse_error = e.what();
        if (out.parse_error.empty()) out.parse_error = "(empty message)";
        return out;
    }
    if (!startup) return out;
    try {
        tadah::hpo::validate_hpo_setup(ctx, *trg, tc.has_train_db,
                                       tc.has_validation);
    } catch (const std::exception& e) {
        out.startup_error = e.what();
        if (out.startup_error.empty()) out.startup_error = "(empty message)";
    }
    return out;
}

/* "" when the parser treats the case as the registry says, else why not. */
std::string conformance_failure(const Registry& reg, const Case& c)
{
    try {
        const Composed comp = compose(reg, c);
        const bool startup = c.check == "startup";
        const std::string file = "\n--- target file ---\n" + comp.file +
                                 "-------------------";
        if (c.check == "check") {
            if (c.kind == "example")
                return "check = \"check\" is for an invalid case" + file;
            const std::string want = expected_error(reg, c, comp);
            const std::string got = run_check(reg, comp, c.env);
            if (got.empty())
                return "tadah hpo --check accepts the invalid case; "
                       "expected:\n" + want + file;
            if (got != want)
                return "expected:\n" + want + "\ngot:\n" + got + file;
            return "";
        }
        const Outcome got = run_file(reg, comp, startup, c.env);
        if (c.kind == "example") {
            if (!got.parse_error.empty())
                return "the parser refuses the example:\n" + got.parse_error +
                       file;
            if (!got.startup_error.empty())
                return "the startup checks refuse the example:\n" +
                       got.startup_error + file;
            return "";
        }
        const std::string want = expected_error(reg, c, comp);
        const std::string& actual =
            startup ? got.startup_error : got.parse_error;
        if (startup && !got.parse_error.empty())
            return "the parser refuses the file before the startup checks:\n" +
                   got.parse_error + "\nexpected from the startup checks:\n" +
                   want + file;
        if (actual.empty())
            return std::string(startup ? "the startup checks accept"
                                       : "the parser accepts") +
                   " the invalid line; expected:\n" + want + file;
        if (actual != want)
            return "expected:\n" + want + "\ngot:\n" + actual + file;
        return "";
    } catch (const std::exception& e) {
        return std::string("cannot run: ") + e.what();
    }
}

void check_file(const std::string& file)
{
    const std::vector<std::string> parts =
        file == kMainFile ? std::vector<std::string>{}
                          : std::vector<std::string>{file};
    const Registry reg = load(parts, {file});
    std::size_t examples = 0, invalid = 0, failed = 0;
    for (const Case& c : reg.cases) {
        (c.kind == "example" ? examples : invalid) += 1;
        const std::string why = conformance_failure(reg, c);
        if (!why.empty()) {
            ++failed;
            FAIL_CHECK(c.label() + ":\n" + why);
        }
    }
    INFO(file << ": " << examples << " examples, " << invalid
              << " invalid lines, " << failed << " failed");
    CHECK(failed == 0);
    if (examples == 0)
        FAIL_CHECK(file + " holds no example: nothing of it is checked");
    if (invalid == 0)
        FAIL_CHECK(file + " holds no invalid line: nothing of it is checked");
}

}  // namespace

/* meta.parts lists the part files this test has a case for: a new part
 * file needs a test case of its own (and a tag), a removed one loses it. */
TEST_CASE("hpotarget registry: meta.parts lists the part files checked "
          "here", "[hpotarget_registry]")
{
    const Registry reg = load({}, {});
    CHECK(reg.parts == kParts);
}

/* Each wrapper, with its slot left empty, is a target file the parser and
 * the startup checks accept: an example fails for what its text does, not
 * for its wrapper. */
TEST_CASE("hpotarget registry: every wrapper is a complete target file",
          "[hpotarget_registry][hpotarget_main]")
{
    const Registry reg = load(kParts, {});
    REQUIRE_FALSE(reg.wrappers.empty());
    for (const auto& [name, w] : reg.wrappers) {
        Case c;
        c.part = "wrapper";
        c.owner = name;
        c.kind = "example";
        c.wrap = name;
        c.check = "startup";
        const std::string why = conformance_failure(reg, c);
        if (!why.empty()) FAIL_CHECK("wrapper " + name + ":\n" + why);
    }
}

TEST_CASE("hpotarget registry conformance: hpotarget.toml",
          "[hpotarget_registry][hpotarget_main]")
{
    check_file(kMainFile);
}

TEST_CASE("hpotarget registry conformance: hpotarget_optimizer.toml",
          "[hpotarget_registry][hpotarget_optimizer]")
{
    check_file("hpotarget_optimizer.toml");
}

TEST_CASE("hpotarget registry conformance: hpotarget_lines.toml",
          "[hpotarget_registry][hpotarget_lines]")
{
    check_file("hpotarget_lines.toml");
}

TEST_CASE("hpotarget registry conformance: hpotarget_params.toml",
          "[hpotarget_registry][hpotarget_params]")
{
    check_file("hpotarget_params.toml");
}

/* ---- MLIP's OPTIM reader against tadah hpo ------------------------------
 *
 * tadah hpo reads the OPTIM lines with MLIP's OPTIM reader
 * (tadah/mlip/hpotarget_optim.h), and tadah analysis check_bounds reads a
 * target file with its file reader, read_optim_lines(), which reads the
 * logical lines with its own lexer and the configuration as
 * prepare_optim_seed() prepares it. Every case of the registry (examples
 * and invalid lines, put into their wrappers) and every wrapper is read
 * both ways, each on its own context parsed from the case's training
 * configuration: HPOTargets (the parse, not the startup checks) and
 * prepare_optim_seed + read_optim_lines. The two agree when
 *   - HPOTargets reads the file: read_optim_lines reads it, and every value
 *     an OPTIM line selects is one of its boxes with the same bounds (bit
 *     for bit), origin, shape (and ASINH c), tag and line, the key's
 *     values the same, and it has no other box;
 *   - HPOTargets refuses an OPTIM line, or a last line that ends with a
 *     backslash: read_optim_lines refuses the same line with the same
 *     reason (config_line_error of its line, reason and text is HPOTargets'
 *     message), except the refusal of an OPTIM line inside a block ("OPTIM
 *     is a top-level key, and the ..."), which read_optim_lines does not
 *     make: it then refuses no earlier line;
 *   - HPOTargets refuses another line: read_optim_lines refuses no earlier
 *     line;
 *   - HPOTargets' error names no line: it is the refusal of the seed's
 *     preparation, which prepare_optim_seed makes with the same text, or
 *     one after the OPTIM lines were read, and read_optim_lines reads the
 *     file.
 * Tag [hpotarget_mlip]: one case per registry file and one for the
 * wrappers. */

#if __has_include(<tadah/mlip/hpotarget_optim.h>)

#include <tadah/core/number_format.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/mlip/hpotarget_optim.h>

#include <cstring>

namespace {

namespace M = tadah::mlip;

/* What read_optim_lines did with a file. */
struct MlipRead {
    std::string seed_error;   // prepare_optim_seed's refusal
    bool refused = false;     // read_optim_lines refused a line
    std::size_t line = 0;
    std::string what, text;
    std::vector<M::OptimTargetLine> lines;
};

/* An error HPOTargets frames, "line N of target.hpo: what\nLine: text"
 * (it was "Error on line N: what\nLine: text", without the file). */
struct Frame {
    std::size_t line = 0;
    std::string what, text;
};

std::optional<Frame> frame_of(const std::string& e)
{
    const std::string head = "line ";
    const std::string of = std::string(" of ") + kTargetFile + ": ";
    if (e.rfind(head, 0) != 0) return std::nullopt;
    const auto digits_end = e.find_first_not_of("0123456789", head.size());
    if (digits_end == head.size() || digits_end == std::string::npos ||
        e.compare(digits_end, of.size(), of) != 0)
        return std::nullopt;
    const auto colon = digits_end + of.size() - 2;
    const auto at = e.rfind("\nLine: ");
    if (at == std::string::npos || at < colon) return std::nullopt;
    Frame f;
    f.line = std::stoul(e.substr(head.size(), digits_end - head.size()));
    f.what = e.substr(colon + 2, at - colon - 2);
    f.text = e.substr(at + 7);
    return f;
}

bool same_bits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

const char* origin_name(M::OptimBoxOrigin o)
{
    switch (o) {
        case M::OptimBoxOrigin::Explicit:         return "explicit";
        case M::OptimBoxOrigin::AutoLinear:       return "AutoLinear";
        case M::OptimBoxOrigin::AutoGeometric:    return "AutoGeometric";
        case M::OptimBoxOrigin::AutoLinearNonneg: return "AutoLinearNonneg";
    }
    return "?";
}

const char* origin_name(tadah::hpo::BoxOrigin o)
{
    switch (o) {
        case tadah::hpo::BoxOrigin::Explicit:         return "explicit";
        case tadah::hpo::BoxOrigin::AutoLinear:       return "AutoLinear";
        case tadah::hpo::BoxOrigin::AutoGeometric:    return "AutoGeometric";
        case tadah::hpo::BoxOrigin::AutoLinearNonneg: return "AutoLinearNonneg";
        case tadah::hpo::BoxOrigin::Adapted:          return "Adapted";
    }
    return "?";
}

const char* shape_name(M::OptimShape s)
{
    switch (s) {
        case M::OptimShape::Linear: return "LIN";
        case M::OptimShape::Log:    return "LOG";
        case M::OptimShape::Asinh:  return "ASINH";
    }
    return "?";
}

const char* shape_name(tadah::hpo::ScaleKind s)
{
    switch (s) {
        case tadah::hpo::ScaleKind::Linear: return "LIN";
        case tadah::hpo::ScaleKind::Log:    return "LOG";
        case tadah::hpo::ScaleKind::Asinh:  return "ASINH";
    }
    return "?";
}

/* "" when the boxes of `m` are those of `trg`, else the first difference. */
std::string box_difference(const tadah::hpo::HPOTargets& trg,
                           const std::vector<M::OptimTargetLine>& m,
                           std::size_t& compared)
{
    struct At { const M::OptimLine* line; const M::OptimBox* box; };
    std::map<std::pair<std::string, std::size_t>, At> boxes;
    std::map<std::string, const M::OptimLine*> keys;
    for (const auto& l : m) {
        keys[l.optim.key] = &l.optim;
        for (const auto& b : l.optim.boxes)
            if (!boxes.emplace(std::make_pair(l.optim.key, b.slot),
                               At{&l.optim, &b}).second)
                return "MLIP's reader gives " + l.optim.key + " slot " +
                       std::to_string(b.slot) + " two boxes";
    }
    std::size_t n = 0;
    for (const auto& [key, o] : trg.optimap) {
        const auto k = keys.find(key);
        if (k == keys.end())
            return "MLIP's reader has no line of " + key;
        if (k->second->config != o.config)
            return "the values of " + key + " differ";
        for (std::size_t i = 0; i < o.opt.size(); ++i) {
            if (!o.opt[i]) continue;
            ++n;
            const auto it = boxes.find({key, i});
            const std::string what = key + " slot " + std::to_string(i);
            if (it == boxes.end())
                return "MLIP's reader has no box of " + what;
            const M::OptimLine& l = *it->second.line;
            const M::OptimBox& b = *it->second.box;
            if (!same_bits(b.lo, o.lowbound[i]) ||
                !same_bits(b.hi, o.highbound[i]))
                return what + ": the box [" +
                       tadah::core::format_roundtrip(o.lowbound[i]) + ", " +
                       tadah::core::format_roundtrip(o.highbound[i]) +
                       "] of tadah hpo is [" +
                       tadah::core::format_roundtrip(b.lo) + ", " +
                       tadah::core::format_roundtrip(b.hi) +
                       "] in MLIP's reader";
            if (std::string(origin_name(b.origin)) !=
                origin_name(o.origin[i]))
                return what + ": origin " + origin_name(o.origin[i]) +
                       " of tadah hpo is " + origin_name(b.origin);
            if (std::string(shape_name(l.shape)) !=
                    shape_name(o.scale[i].kind) ||
                (l.shape == M::OptimShape::Asinh &&
                 !same_bits(l.c, o.scale[i].c)))
                return what + ": the shape differs";
            if (l.tagged != o.tagged[i]) return what + ": the tag differs";
            if (l.line != o.line[i]) return what + ": the line differs";
        }
    }
    if (n != boxes.size())
        return "MLIP's reader gives " + std::to_string(boxes.size()) +
               " boxes, tadah hpo " + std::to_string(n);
    compared += n;
    return "";
}

struct MlipCounts {
    std::size_t boxes = 0;     // boxes compared bit for bit
    std::size_t refusals = 0;  // OPTIM refusals compared
};

/* "" when MLIP's reader reads the case as tadah hpo reads it, else why
 * not. */
std::string mlip_failure(const Registry& reg, const Case& c,
                         MlipCounts& counts)
{
    try {
        const Composed comp = compose(reg, c);
        const auto ctx_it = reg.contexts.find(comp.context);
        if (ctx_it == reg.contexts.end())
            throw std::runtime_error("context '" + comp.context +
                                     "' is not in the files read");
        const std::string file = "\n--- target file ---\n" + comp.file +
                                 "-------------------";
        tadah::core::PeriodicTable::initialize();
        ScratchCwd cwd;
        for (const auto& [name, f] : reg.fixtures) {
            const fs::path p(f.path);
            if (p.has_parent_path()) fs::create_directories(p.parent_path());
            std::ofstream os(p);
            os << f.text;
        }
        {
            std::ofstream os("target.hpo");
            os << comp.file;
        }
        ScopedEnv scoped(c.env);
        auto parse = [&]() {
            std::istringstream config(ctx_it->second.config);
            return tadah::core::CoreUserConfigParser().parse(config);
        };
        tadah::core::Context hctx = parse();
        tadah::core::Context mctx = parse();
        std::string h_error;
        std::optional<tadah::hpo::HPOTargets> trg;
        try {
            trg.emplace(hctx, "target.hpo");
        } catch (const std::exception& e) {
            h_error = e.what()[0] ? e.what() : "(empty message)";
        }
        MlipRead m;
        std::optional<M::OptimSeed> seed;
        try {
            seed = M::prepare_optim_seed(mctx, "tadah hpo");
        } catch (const std::exception& e) {
            m.seed_error = e.what()[0] ? e.what() : "(empty message)";
        }
        if (seed) {
            std::istringstream in(comp.file);
            try {
                m.lines = M::read_optim_lines(in, mctx, *seed);
            } catch (const M::TargetLineError& e) {
                m.refused = true;
                m.line = e.line;
                m.what = e.what();
                m.text = e.text;
            }
        }
        const std::string m_refusal =
            m.refused ? tadah::hpo::config_line_error(kTargetFile, m.line,
                                                      m.what, m.text)
                      : std::string();

        if (h_error.empty()) {
            if (!m.seed_error.empty())
                return "tadah hpo reads the file; prepare_optim_seed "
                       "refuses it:\n" + m.seed_error + file;
            if (m.refused)
                return "tadah hpo reads the file; MLIP's reader refuses "
                       "it:\n" + m_refusal + file;
            const std::string d = box_difference(*trg, m.lines, counts.boxes);
            return d.empty() ? "" : d + file;
        }
        const std::optional<Frame> f = frame_of(h_error);
        if (!f) {
            if (!m.seed_error.empty())
                return m.seed_error == h_error
                           ? ""
                           : "expected (tadah hpo):\n" + h_error +
                                 "\ngot (prepare_optim_seed):\n" +
                                 m.seed_error + file;
            if (m.refused)
                return "tadah hpo's refusal names no line:\n" + h_error +
                       "\nand MLIP's reader refuses a line:\n" + m_refusal +
                       file;
            return "";
        }
        if (!m.seed_error.empty())
            return "tadah hpo refuses line " + std::to_string(f->line) +
                   " and prepare_optim_seed the file:\n" + m.seed_error + file;
        std::istringstream in(comp.file);
        const std::vector<M::TargetLine> logical =
            M::target_logical_lines(in);
        const auto at = std::find_if(
            logical.begin(), logical.end(), [&](const M::TargetLine& l) {
                return l.last == f->line && l.text == f->text;
            });
        if (at == logical.end())
            return "tadah hpo's refusal names a line MLIP's lexer does not "
                   "read:\n" + h_error + file;
        const bool optim_line = at->key == "OPTIM" ||
                                f->what == M::kTargetUnfinishedLineMsg;
        const bool in_block =
            f->what.rfind("OPTIM is a top-level key, and the ", 0) == 0;
        if (optim_line && !in_block) {
            ++counts.refusals;
            if (!m.refused)
                return "expected (tadah hpo):\n" + h_error +
                       "\nMLIP's reader reads the file" + file;
            return m_refusal == h_error
                       ? ""
                       : "expected (tadah hpo):\n" + h_error +
                             "\ngot (MLIP's reader):\n" + m_refusal + file;
        }
        if (m.refused && m.line < f->line)
            return "tadah hpo refuses line " + std::to_string(f->line) +
                   ":\n" + h_error + "\nand MLIP's reader an earlier one:\n" +
                   m_refusal + file;
        return "";
    } catch (const std::exception& e) {
        return std::string("cannot run: ") + e.what();
    }
}

void check_file_mlip(const std::string& file)
{
    const std::vector<std::string> parts =
        file == kMainFile ? std::vector<std::string>{}
                          : std::vector<std::string>{file};
    const Registry reg = load(parts, {file});
    MlipCounts counts;
    std::size_t failed = 0;
    for (const Case& c : reg.cases) {
        const std::string why = mlip_failure(reg, c, counts);
        if (!why.empty()) {
            ++failed;
            FAIL_CHECK(c.label() + ":\n" + why);
        }
    }
    INFO(file << ": " << reg.cases.size() << " cases, " << counts.boxes
              << " boxes and " << counts.refusals
              << " OPTIM refusals compared, " << failed << " failed");
    CHECK(failed == 0);
    /* Every wrapper has an OPTIM line, so every file's cases compare
     * boxes; hpotarget_lines.toml holds the invalid OPTIM lines. */
    CHECK(counts.boxes > 0);
    if (file == "hpotarget_lines.toml") CHECK(counts.refusals > 0);
}

}  // namespace

/* Pins: MLIP's OPTIM reader (read_optim_lines, prepare_optim_seed) reads
 * every case of each registry file, and every wrapper, as tadah hpo reads
 * it (the rules above).
 * Before the fix: MLIP has no OPTIM reader (the header does not exist). */
TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads every wrapper as "
          "tadah hpo", "[hpotarget_mlip]")
{
    const Registry reg = load(kParts, {});
    REQUIRE_FALSE(reg.wrappers.empty());
    MlipCounts counts;
    for (const auto& [name, w] : reg.wrappers) {
        Case c;
        c.part = "wrapper";
        c.owner = name;
        c.kind = "example";
        c.wrap = name;
        const std::string why = mlip_failure(reg, c, counts);
        if (!why.empty()) FAIL_CHECK("wrapper " + name + ":\n" + why);
    }
    CHECK(counts.boxes > 0);
}

TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads hpotarget.toml as "
          "tadah hpo", "[hpotarget_mlip]")
{
    check_file_mlip(kMainFile);
}

TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads "
          "hpotarget_optimizer.toml as tadah hpo", "[hpotarget_mlip]")
{
    check_file_mlip("hpotarget_optimizer.toml");
}

TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads "
          "hpotarget_lines.toml as tadah hpo", "[hpotarget_mlip]")
{
    check_file_mlip("hpotarget_lines.toml");
}

TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads "
          "hpotarget_params.toml as tadah hpo", "[hpotarget_mlip]")
{
    check_file_mlip("hpotarget_params.toml");
}

#else

TEST_CASE("hpotarget registry: MLIP's OPTIM reader reads every wrapper as "
          "tadah hpo", "[hpotarget_mlip]")
{
    FAIL("tadah/mlip/hpotarget_optim.h does not exist: MLIP has no OPTIM "
         "reader");
}

#endif

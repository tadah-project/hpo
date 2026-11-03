#pragma once
/* hpotarget_cases.h -- the cases of target files that hpotarget_case_runner
 * (tools/hpotarget_case_runner.cpp) reads, a TOML file of [[case]] tables,
 * each run in a fresh directory with its files, its training configuration
 * and its environment (hpotarget_case_runner.cpp describes the fields),
 * and the directory a case runs in, which the frozen corpus of target files
 * (test_hpotarget_corpus.cpp) runs its inputs in too. */

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>

#include <toml.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace hpotarget_cases {

namespace fs = std::filesystem;

struct Fixture { std::string path, text; };
struct Copy { std::string from, to; };

struct Case {
    std::string id, mode, file, target_file, config, config_file,
        validation, tree, cwd;
    bool has_train_db = true, has_validation = true;
    std::vector<std::string> config_drop;
    std::map<std::string, std::vector<std::string>> config_add;
    std::map<std::string, std::string> env;
    std::vector<Copy> copies;
    std::vector<Fixture> fixtures;
};

inline std::string str(const toml::value& t, const std::string& k,
                       const std::string& dflt = "")
{
    if (!t.contains(k)) return dflt;
    return t.at(k).as_string().str;
}

inline bool flag(const toml::value& t, const std::string& k, bool dflt)
{
    return t.contains(k) ? t.at(k).as_boolean() : dflt;
}

/* The cases of the TOML file `path`; throws std::runtime_error for a case
 * whose fields contradict each other. */
inline std::vector<Case> read_cases(const std::string& path)
{
    const toml::value doc = toml::parse(path);
    std::vector<Case> out;
    if (!doc.contains("case")) return out;
    for (const auto& t : doc.at("case").as_array()) {
        Case c;
        c.id = str(t, "id");
        c.mode = str(t, "mode", "parse");
        c.file = str(t, "file");
        c.target_file = str(t, "target_file");
        c.config = str(t, "config");
        c.config_file = str(t, "config_file");
        c.validation = str(t, "validation");
        c.tree = str(t, "tree");
        c.cwd = str(t, "cwd");
        c.has_train_db = flag(t, "has_train_db", true);
        c.has_validation = flag(t, "has_validation", true);
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
                c.copies.push_back({str(f, "from"), str(f, "to")});
        if (t.contains("fixtures"))
            for (const auto& f : t.at("fixtures").as_array())
                c.fixtures.push_back({str(f, "path"), str(f, "text")});
        if (c.mode != "parse" && c.mode != "startup" && c.mode != "check" &&
            c.mode != "defaults" && c.mode != "describe")
            throw std::runtime_error("case " + c.id + ": mode '" + c.mode +
                                     "' is not parse, startup, check, "
                                     "defaults or describe");
        if (!c.target_file.empty() && !c.file.empty())
            throw std::runtime_error("case " + c.id + ": file and "
                                     "target_file are both given");
        if (!c.config_file.empty() && !c.config.empty())
            throw std::runtime_error("case " + c.id + ": config and "
                                     "config_file are both given");
        if (c.mode == "check" &&
            (!c.target_file.empty() || !c.config_file.empty() ||
             !c.config_drop.empty() || !c.config_add.empty()))
            throw std::runtime_error("case " + c.id + ": mode check takes "
                                     "the target file and the training "
                                     "configuration as text (file, "
                                     "config)");
        out.push_back(std::move(c));
    }
    return out;
}

/* A fresh directory, the working directory while it lives: a copy of
 * `tree` when one is named, and its directory `cwd` when one is named. */
class ScratchCwd {
public:
    ScratchCwd(const std::string& tree, const std::string& cwd)
    {
        static unsigned long n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_hpotarget_cases_" + std::to_string(::getpid()) + "_" +
                std::to_string(n++) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()));
        fs::create_directories(dir_);
        if (!tree.empty())
            fs::copy(tree, dir_, fs::copy_options::recursive);
        const fs::path work = cwd.empty() ? dir_ : dir_ / cwd;
        if (!fs::is_directory(work))
            throw std::runtime_error("the directory " + cwd + " of the case "
                                     "does not exist");
        old_ = fs::current_path();
        fs::current_path(work);
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

inline void write_file(const fs::path& p, const std::string& text)
{
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream os(p);
    os << text;
}

inline tadah::core::Context parse_config(const std::string& text)
{
    std::istringstream in(text);
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* The training configuration of a case that HPOTargets reads. */
inline tadah::core::Context case_config(const Case& c)
{
    tadah::core::Context ctx =
        c.config_file.empty()
            ? parse_config(c.config)
            : tadah::core::CoreUserConfigParser().parse(c.config_file);
    for (const auto& key : c.config_drop) ctx.remove(key);
    for (const auto& [key, values] : c.config_add)
        for (const auto& value : values) ctx.add(key, value);
    return ctx;
}

/* The directory of a case, its working directory while it lives, with
 * the case's tree, copies and fixture files, and the target file, which
 * target() names; the case's environment is set while it lives. */
class CaseDir {
public:
    explicit CaseDir(const Case& c)
        : cwd_(c.tree, c.cwd),
          target_(c.target_file.empty() ? std::string("target.hpo")
                                        : c.target_file)
    {
        for (const auto& f : c.copies) fs::copy_file(f.from, f.to);
        for (const auto& f : c.fixtures) write_file(f.path, f.text);
        if (c.target_file.empty()) write_file(target_, c.file);
        env_.emplace(c.env);
    }
    const std::string& target() const { return target_; }

private:
    ScratchCwd cwd_;
    std::string target_;
    std::optional<ScopedEnv> env_;
};

}  // namespace hpotarget_cases

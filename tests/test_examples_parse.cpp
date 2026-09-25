/*
 * Guards the worked examples (HPO/examples) against grammar and startup-rule
 * drift: every examples/NN_name/config.hpo must parse and pass
 * validate_hpo_setup() the way its driver runs it.
 *
 * Each example is a test case of its own. tests/CMakeLists.txt generates the
 * list (examples_parse_cases.inc) from every examples/<NN_name>/config.hpo
 * when it configures, and CONFIGURE_DEPENDS re-globs it on every build, so a new
 * example gets a case -- which fails until examples/tools/parse_guard.manifest
 * names what the example needs. The run itself (LAMMPS, fingerprints) stays
 * with run_reference_suite.sh; this guard needs no data beyond the examples'
 * own files.
 *
 * For each example the guard copies common/ and the example's config.hpo,
 * config.tadah and config.val into a scratch tree, applies the manifest's
 * directives there, and in the example's scratch directory -- relative paths
 * such as ../common/lammps/Ecoh.in resolve as they do under run.sh --
 * parses config.hpo against the local config.tadah (else
 * ../common/config.tadah), resolves the validation targets against the
 * local config.val (else ../common/config.val) and runs validate_hpo_setup.
 * An example whose driver passes another file of its own as the training
 * configuration (`tadah hpo -c <file>`) names it with the `config`
 * directive; the guard copies that file and parses config.hpo against it.
 */

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

#ifndef TADAH_HPO_EXAMPLES_DIR
#error "TADAH_HPO_EXAMPLES_DIR must name HPO/examples (tests/CMakeLists.txt)"
#endif

namespace {

namespace fs = std::filesystem;

const fs::path& examples_dir()
{
    static const fs::path dir = TADAH_HPO_EXAMPLES_DIR;
    return dir;
}

const fs::path& manifest_path()
{
    static const fs::path p = examples_dir() / "tools" / "parse_guard.manifest";
    return p;
}

std::string trim(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_ws(const std::string& s)
{
    std::istringstream is(s);
    std::vector<std::string> out;
    for (std::string t; is >> t;) out.push_back(t);
    return out;
}

/* One manifest entry: the directives, each split into words. */
using Directives = std::vector<std::vector<std::string>>;

/* The manifest, keyed by example. A malformed line, an unknown directive
 * or a repeated example is an error naming the line. */
std::map<std::string, Directives> read_manifest()
{
    std::ifstream in(manifest_path());
    if (!in)
        throw std::runtime_error("cannot open " + manifest_path().string());
    std::map<std::string, Directives> out;
    std::string line;
    std::size_t nline = 0;
    while (std::getline(in, line)) {
        ++nline;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        const std::string where = manifest_path().string() + ":" +
                                  std::to_string(nline) + ": ";
        const auto bar = t.find('|');
        if (bar == std::string::npos)
            throw std::runtime_error(where + "expected '<example> | <directives>'");
        const std::string name = trim(t.substr(0, bar));
        const std::string rest = trim(t.substr(bar + 1));
        if (name.empty() || name.find_first_of(" \t") != std::string::npos)
            throw std::runtime_error(where + "bad example name '" + name + "'");
        if (out.count(name))
            throw std::runtime_error(where + "'" + name + "' is listed twice");
        Directives dirs;
        if (rest != "-") {
            std::string item;
            std::istringstream items(rest);
            while (std::getline(items, item, ';')) {
                auto words = split_ws(item);
                if (words.empty())
                    throw std::runtime_error(where + "empty directive");
                const std::string& d = words.front();
                const bool ok =
                    (d == "copy" && words.size() == 3) ||
                    (d == "config" && words.size() == 2) ||
                    (d == "no-validation" && words.size() == 1) ||
                    (d == "drop" && words.size() == 2) ||
                    (d == "add" && words.size() >= 3);
                if (!ok)
                    throw std::runtime_error(where + "bad directive '" +
                                             trim(item) + "'");
                dirs.push_back(std::move(words));
            }
        }
        out.emplace(name, std::move(dirs));
    }
    return out;
}

/* Example directories that hold a config.hpo, sorted. */
std::vector<std::string> example_dirs_on_disk()
{
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(examples_dir()))
        if (e.is_directory() && fs::is_regular_file(e.path() / "config.hpo"))
            out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

/* The examples this binary was built with (one test case each). */
const std::vector<std::string>& compiled_examples()
{
    static const std::vector<std::string> v = {
#define TADAH_EXAMPLE_PARSE_CASE(name) name,
#include "examples_parse_cases.inc"
#undef TADAH_EXAMPLE_PARSE_CASE
    };
    return v;
}

/* A scratch tree holding common/ and the example; removed on exit, the
 * working directory restored. */
struct ScratchTree {
    fs::path root;
    fs::path original_cwd = fs::current_path();
    explicit ScratchTree(const std::string& name)
    : root(fs::temp_directory_path() /
           ("tadah_examples_parse_" + name + "_" + std::to_string(::getpid()) +
            "_" + std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch().count())))
    {
        fs::create_directories(root);
    }
    ~ScratchTree() {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(root, ec);
    }
};

void check_example(const std::string& name)
{
    tadah::core::PeriodicTable::initialize();
    const auto manifest = read_manifest();
    const auto entry = manifest.find(name);
    if (entry == manifest.end())
        FAIL("examples/" << name << " has no entry in " << manifest_path().string()
             << ": add '" << name << " | -' (or the directives it needs)");

    const fs::path src = examples_dir() / name;
    ScratchTree tree(name);
    fs::copy(examples_dir() / "common", tree.root / "common",
             fs::copy_options::recursive);
    const fs::path dir = tree.root / name;
    fs::create_directories(dir);
    for (const char* f : {"config.hpo", "config.tadah", "config.val"})
        if (fs::exists(src / f)) fs::copy_file(src / f, dir / f);
    fs::current_path(dir);

    bool validation = true;
    std::string config_file;
    std::vector<std::string> drops;
    std::vector<std::vector<std::string>> adds;
    for (const auto& d : entry->second) {
        if (d[0] == "copy") {
            REQUIRE(fs::exists(d[1]));
            fs::copy_file(d[1], d[2], fs::copy_options::overwrite_existing);
        } else if (d[0] == "config") {
            /* The example's own file its driver passes with -c. */
            INFO("config directive: examples/" << name << "/" << d[1]);
            REQUIRE(fs::exists(src / d[1]));
            fs::copy_file(src / d[1], d[1], fs::copy_options::overwrite_existing);
            config_file = d[1];
        } else if (d[0] == "no-validation") {
            validation = false;
        } else if (d[0] == "drop") {
            drops.push_back(d[1]);
        } else {
            adds.push_back(d);
        }
    }

    const std::string train_cfg =
        !config_file.empty()        ? config_file
        : fs::exists("config.tadah") ? "config.tadah"
                                     : "../common/config.tadah";
    tadah::core::Context ctx(tadah::core::CoreUserConfigParser().parse(train_cfg));
    for (const auto& key : drops) ctx.remove(key);
    for (const auto& a : adds)
        for (std::size_t i = 2; i < a.size(); ++i) ctx.add(a[1], a[i]);

    INFO("examples/" << name << ": training configuration " << train_cfg
         << (validation ? "" : ", no --validation"));
    tadah::hpo::HPOTargets trg(ctx, "config.hpo");

    if (validation) {
        const std::string val_cfg =
            fs::exists("config.val") ? "config.val" : "../common/config.val";
        INFO("validation file " << val_cfg);
        tadah::core::Context val(tadah::core::CoreUserConfigParser().parse(val_cfg));
        std::vector<std::string> files;
        if (val.contains("DBFILE")) files = val("DBFILE");
        for (const auto& f : files) CHECK(fs::exists(f));
        trg.resolve_validation_targets(files);
    }
    const bool has_train_db = ctx.contains("DBFILE") && ctx.size("DBFILE") > 0;
    tadah::hpo::validate_hpo_setup(ctx, trg, has_train_db, validation);
}

}  // namespace

#define TADAH_EXAMPLE_PARSE_CASE(name)                                        \
    TEST_CASE("HPO example " name ": config.hpo parses and passes the "      \
              "startup rules", "[HPO][examples][parse_guard]")               \
    {                                                                         \
        check_example(name);                                                  \
    }
#include "examples_parse_cases.inc"
#undef TADAH_EXAMPLE_PARSE_CASE

TEST_CASE("HPO examples: the parse guard covers every examples/*/config.hpo",
          "[HPO][examples][parse_guard]")
{
    /* The case list is generated when the build configures; an example
     * directory added or removed since then needs a rebuild (which
     * re-configures on its own) before the guard covers it. */
    CHECK(compiled_examples() == example_dirs_on_disk());
}

TEST_CASE("HPO examples: parse_guard.manifest is well formed and lists only "
          "existing examples", "[HPO][examples][parse_guard]")
{
    std::map<std::string, Directives> manifest;
    REQUIRE_NOTHROW(manifest = read_manifest());
    const auto on_disk = example_dirs_on_disk();
    for (const auto& kv : manifest) {
        INFO("manifest entry " << kv.first);
        CHECK(std::find(on_disk.begin(), on_disk.end(), kv.first) !=
              on_disk.end());
    }
}

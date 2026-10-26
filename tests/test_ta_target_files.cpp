/*
 * Parse cases over the target files of tests/data/ta/target_files: each
 * file parses against its training configuration, has its validation
 * targets resolved against its validation configuration, and passes the
 * startup validation (validate_hpo_setup()), the rules `tadah hpo` and
 * `tadah hpo --check` apply to the configuration before the data are read. Every file is a test case of its own. The coverage
 * case fails when a target file of that directory has no case, when a case
 * has no file, and when another file there is no case's training or
 * validation configuration.
 *
 * The files name the shared Ta set by relative paths (../common/lammps/...,
 * ../common/data/... through the configurations), so each case copies
 * tests/data/ta into a scratch directory and works in its target_files/
 * directory: the paths resolve there as in the source tree, and what a case
 * creates (the WARM seed seed_pot.tadah) or a parse writes stays out of the
 * source tree. By default a file parses against ../common/config.tadah and
 * validates on ../common/config.val; parse_case() names the cases that
 * differ: another training configuration, no validation (`tadah hpo`
 * without --validation), a WARM seed to create, or keys removed from or
 * added to the training configuration.
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
#include <set>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef TADAH_HPO_TA_DATA_DIR
#error "TADAH_HPO_TA_DATA_DIR must name HPO/tests/data/ta (tests/CMakeLists.txt)"
#endif

/* Every target file of tests/data/ta/target_files, without ".hpo". */
#define TADAH_TA_TARGET_FILES(X)                                              \
    X(anneal) X(anneal_corana) X(anneal_pa) X(anneal_pt) X(asa)               \
    X(dlib_bfgs) X(dm_ream_embf0) X(failprobe_grading) X(fixweight_mixed)     \
    X(fixweight_no_dbfile) X(init_chain) X(init_lhs_multistart)               \
    X(init_random) X(init_sobol_gfs) X(init_warm) X(invar_outvar)             \
    X(mlsl_inner) X(nlopt_sbplx_step) X(optim_auto) X(pc_isolation)           \
    X(pc_observable) X(per_dataset_targets) X(prestage) X(restart)            \
    X(skip_above) X(tadah_grid) X(tadah_random)

namespace {

namespace fs = std::filesystem;

const fs::path& ta_data_dir()
{
    static const fs::path dir = TADAH_HPO_TA_DATA_DIR;
    return dir;
}

/* How a target file is read; paths are relative to target_files/. */
struct ParseCase {
    std::string training = "../common/config.tadah";
    std::string validation = "../common/config.val";  // "": no --validation
    std::string warm_seed;  // copied to seed_pot.tadah, the WARM seed
    std::vector<std::string> drop;  // keys removed from the training config
    std::vector<std::pair<std::string, std::vector<std::string>>> add;
};

ParseCase parse_case(const std::string& name)
{
    ParseCase c;
    if (name == "init_warm" || name == "init_chain") {
        /* WARM reads the searched keys (RCUT2B) from its seed potential;
         * the shared training configuration carries them. */
        c.warm_seed = "../common/config.tadah";
    } else if (name == "fixweight_mixed") {
        c.training = "fixweight_mixed.tadah";
    } else if (name == "pc_observable") {
        c.training = "pc_observable.tadah";
    } else if (name == "per_dataset_targets") {
        c.validation = "per_dataset_targets.val";
    } else if (name == "fixweight_no_dbfile") {
        /* The shared model's 14 weights pinned, no training data, no
         * validation: the loss is the LAMMPS cohesive energy alone. The
         * values of the pinned weights do not matter to the parse. Without
         * a training DBFILE, ATOM declares the element set, which the
         * startup rules require (no data gives it). */
        c.validation.clear();
        c.drop = {"DBFILE"};
        c.add = {{"ATOM", {"Ta"}},
                 {"FIXINDEX", {"1-14"}},
                 {"FIXWEIGHT", std::vector<std::string>(14, "0")}};
    } else if (name == "dm_ream_embf0") {
        /* The DM_REAM seed pins every weight and has no DBFILE. */
        c.training = "../ta2_refit_seed.tadah";
        c.validation.clear();
    }
    return c;
}

std::vector<std::string> case_names()
{
#define TADAH_TA_TARGET_FILE_NAME(name) #name,
    std::vector<std::string> v = {TADAH_TA_TARGET_FILES(TADAH_TA_TARGET_FILE_NAME)};
#undef TADAH_TA_TARGET_FILE_NAME
    std::sort(v.begin(), v.end());
    return v;
}

/* A scratch copy of tests/data/ta, the working directory its target_files/
 * while it lives; removed on exit, the working directory restored. */
struct ScratchCopy {
    fs::path root;
    fs::path original_cwd = fs::current_path();
    explicit ScratchCopy(const std::string& name)
    : root(fs::temp_directory_path() /
           ("tadah_ta_target_file_" + name + "_" + std::to_string(::getpid()) +
            "_" + std::to_string(std::chrono::steady_clock::now()
                                     .time_since_epoch().count())))
    {
        fs::create_directories(root);
        fs::copy(ta_data_dir(), root, fs::copy_options::recursive);
        fs::current_path(root / "target_files");
    }
    ~ScratchCopy()
    {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(root, ec);
    }
};

void check_target_file(const std::string& name)
{
    tadah::core::PeriodicTable::initialize();
    const ParseCase c = parse_case(name);
    ScratchCopy tree(name);
    if (!c.warm_seed.empty()) {
        REQUIRE(fs::exists(c.warm_seed));
        fs::copy_file(c.warm_seed, "seed_pot.tadah");
    }

    INFO("target_files/" << name << ".hpo: training configuration "
         << c.training << ", "
         << (c.validation.empty() ? "no --validation"
                                  : "validation " + c.validation));
    REQUIRE(fs::exists(c.training));
    tadah::core::Context ctx(
        tadah::core::CoreUserConfigParser().parse(c.training));
    for (const auto& key : c.drop) ctx.remove(key);
    for (const auto& [key, values] : c.add)
        for (const auto& value : values) ctx.add(key, value);

    tadah::hpo::HPOTargets trg(ctx, name + ".hpo");

    const bool validation = !c.validation.empty();
    if (validation) {
        REQUIRE(fs::exists(c.validation));
        tadah::core::Context val(
            tadah::core::CoreUserConfigParser().parse(c.validation));
        std::vector<std::string> files;
        if (val.contains("DBFILE")) files = val("DBFILE");
        REQUIRE_FALSE(files.empty());
        for (const auto& f : files) CHECK(fs::exists(f));
        trg.resolve_validation_targets(files);
    }
    const bool has_train_db = ctx.contains("DBFILE") && ctx.size("DBFILE") > 0;
    tadah::hpo::validate_hpo_setup(ctx, trg, has_train_db, validation);
}

}  // namespace

#define TADAH_TA_TARGET_FILE_CASE(name)                                       \
    TEST_CASE("HPO target file " #name ".hpo parses and passes the startup " \
              "rules", "[HPO][target_files]")                                 \
    {                                                                         \
        check_target_file(#name);                                             \
    }
TADAH_TA_TARGET_FILES(TADAH_TA_TARGET_FILE_CASE)
#undef TADAH_TA_TARGET_FILE_CASE

/* Pins: the startup rules (validate_hpo_setup()) refuse a training
 * configuration without a training DBFILE and without ATOM, which declares
 * the element set no data gives: the fixweight_no_dbfile case without its
 * ATOM line is refused with "No training DBFILE: the ATOM key is required
 * ...", as `tadah hpo` and `tadah hpo --check` refuse it, and the case
 * with it passes (above).
 * Before the fix: validate_hpo_setup() accepted it (the rule was in
 * HPOEngine::run alone, which the parse case does not reach), and this
 * parse case passed without ATOM, which no run accepts. */
TEST_CASE("HPO target files: without a training DBFILE the startup rules "
          "require ATOM",
          "[HPO][target_files]")
{
    tadah::core::PeriodicTable::initialize();
    ParseCase c = parse_case("fixweight_no_dbfile");
    ScratchCopy tree("fixweight_no_dbfile_no_atom");
    tadah::core::Context ctx(
        tadah::core::CoreUserConfigParser().parse(c.training));
    for (const auto& key : c.drop) ctx.remove(key);
    for (const auto& [key, values] : c.add)
        if (key != "ATOM")
            for (const auto& value : values) ctx.add(key, value);
    REQUIRE_FALSE(ctx.has_stored("DBFILE"));
    REQUIRE_FALSE(ctx.has_stored("ATOM"));
    tadah::hpo::HPOTargets trg(ctx, "fixweight_no_dbfile.hpo");
    std::string what;
    try {
        tadah::hpo::validate_hpo_setup(ctx, trg, false, false);
    } catch (const std::exception& e) {
        what = e.what();
    }
    INFO(what);
    CHECK(what.find("No training DBFILE: the ATOM key is required") !=
          std::string::npos);
}

TEST_CASE("HPO target files: every file of tests/data/ta/target_files is a "
          "parse case or the training or validation configuration of one",
          "[HPO][target_files]")
{
    const std::vector<std::string> cases = case_names();
    std::set<std::string> configurations;
    for (const auto& name : cases) {
        const ParseCase c = parse_case(name);
        for (const std::string& f : {c.training, c.validation})
            if (!f.empty() && f.rfind("../", 0) != 0) configurations.insert(f);
    }

    std::vector<std::string> target_files;
    std::vector<std::string> unused;
    for (const auto& e : fs::directory_iterator(ta_data_dir() / "target_files")) {
        const std::string file = e.path().filename().string();
        if (e.path().extension() == ".hpo")
            target_files.push_back(e.path().stem().string());
        else if (!configurations.count(file))
            unused.push_back(file);
    }
    std::sort(target_files.begin(), target_files.end());
    CHECK(target_files == cases);
    INFO("files of target_files/ that no case reads");
    CHECK(unused == std::vector<std::string>{});
}

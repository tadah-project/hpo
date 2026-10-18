// test_hpo_check.cpp
//
// `tadah hpo --check` (CHECK true in the context of hpo_run): every startup
// check of the run, in the run's order, then a summary on standard output
// and status 0, without an evaluation and without a file -- no journal, no
// result file, no search-space sidecar, no DUMP or BDUMP directory. An error
// is the run's: hpo_run() with CHECK true throws what hpo_run() without it
// throws for the same setup, a DUMP or BDUMP directory that cannot be
// created included. The WARNING and HINT lines the run writes to its
// journal at startup are in the summary. CHECK is an instruction to
// hpo_run(), not a key of the model: hpo_run() takes it out of the context,
// so a run with CHECK false writes potentials without a CHECK line. A run
// creates its DUMP and BDUMP directories once every startup check passed:
// one that stops at startup creates none.
//
// The fixture is the no-LAMMPS D2_BP Ta setup of test_hpo_termination.cpp
// (a four-atom training structure, one validation dataset, PC_ERMSE), so a
// run costs milliseconds and the whole hpo_run() stack runs in-process.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

/* A scratch working directory: hpo_run() writes every output file into the
 * current directory. */
struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    explicit ScratchDir(const std::string& tag)
    {
        dir = fs::temp_directory_path() /
              ("tadah_hpo_check_" + tag + "_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir()
    {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

/* Standard output into a string while it lives: the summary of a check is
 * written there. */
struct CaptureCout {
    std::ostringstream text;
    std::streambuf* old;
    CaptureCout() : old(std::cout.rdbuf(text.rdbuf())) {}
    ~CaptureCout() { std::cout.rdbuf(old); }
    CaptureCout(const CaptureCout&) = delete;
    CaptureCout& operator=(const CaptureCout&) = delete;
};

/* Four-atom BCC Ta structure (test_hpo_host_e2e.cpp's fixture). */
void write_ta_db(const fs::path& path, double offset_x, double energy)
{
    std::ofstream os(path);
    os << "test_bcc_ta\n"
       << "1.0 1.0 1.0\n"
       << energy << " 0.0\n"
       << "3.301 0.0 0.0\n"
       << "0.0 3.301 0.0\n"
       << "0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n"
       << "\n";
}

/* The training configuration of test_hpo_termination.cpp, with the target
 * file and, if `val_cfg` is not empty, the validation configuration. */
tadah::core::Context make_ta_context(const std::string& dbfile,
                                     const std::string& hpo_cfg,
                                     const std::string& val_cfg)
{
    tadah::core::Context c;
    c.add("DBFILE",   dbfile);
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(2));
    c.add("TYPE2B",   static_cast<int>(2));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "0.5 1.4");
    c.add("SGRID2B",  "3.0 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   5.0);
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   5.0);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "false");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    c.add("HPOTARGET", hpo_cfg);
    if (!val_cfg.empty()) c.add("VALIDATION", val_cfg);
    return c;
}

const std::string kOptimizer =
    "OPTIMIZER\n"
    "  STRATEGY\n"
    "    TYPE PLAIN\n"
    "    LIB NLOPT\n"
    "    ALGO LN_SBPLX\n"
    "    MAXEVAL 5\n"
    "  ENDSTRATEGY\n"
    "ENDOPTIMIZER\n";

/* The objective and the three-dimensional search space, and archives of
 * every potential (DUMP) and every new best (BDUMP): a run creates their
 * directories, a check does not. */
const std::string kRootLines =
    "LOSS L2\n"
    "FAILSCORE 1e9\n"
    "PC_ERMSE 0 1\n"
    "OPTIM CGRID2B (1) 0.1 0.9\n"
    "OPTIM CGRID2B (2) 1.0 2.0\n"
    "OPTIM SGRID2B (2) 4.0 6.0\n"
    "DUMP 1 pots\n"
    "BDUMP 1 best_pots\n";

/* The input files of one setup, written into the current directory (paths
 * relative to it): train.db, val.db, config.val and config.hpo. */
void write_inputs(const std::string& hpo_text)
{
    write_ta_db("train.db", 0.0, -0.001);
    write_ta_db("val.db", 0.07, -0.04);
    std::ofstream("config.val") << "DBFILE val.db\n";
    std::ofstream("config.hpo") << hpo_text;
}

/* Every entry under `dir`, relative to it. */
std::set<std::string> entries(const fs::path& dir)
{
    std::set<std::string> out;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        out.insert(fs::relative(e.path(), dir).string());
    return out;
}

/* Starts the process-wide HPOLog and HPO_Host state as a new process
 * would. */
void fresh_process_state()
{
    tadah::core::PeriodicTable::initialize();
    tadah::hpo::HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
}

/* What hpo_run(ctx) threw, "" when it returned. */
std::string error_of(const std::function<int()>& run)
{
    CaptureCout quiet;
    try {
        run();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

bool contains(const std::string& s, const std::string& what)
{
    return s.find(what) != std::string::npos;
}

/* The width of the residual OpenMP kernels at @p n while it lives (the
 * OpenMP runtime's, else OMP_NUM_THREADS, as hpo_run() reads it): a run
 * without THREADS warns at startup when it is above 1. */
struct ResidualOmpWidth {
#ifdef TADAH_ENABLE_OPENMP
    int old;
    explicit ResidualOmpWidth(int n) : old(omp_get_max_threads())
    {
        omp_set_num_threads(n);
    }
    ~ResidualOmpWidth() { omp_set_num_threads(old); }
    int width() const { return omp_get_max_threads(); }
#else
    std::optional<std::string> old;
    explicit ResidualOmpWidth(int n)
    {
        if (const char* e = std::getenv("OMP_NUM_THREADS")) old = e;
        ::setenv("OMP_NUM_THREADS", std::to_string(n).c_str(), 1);
    }
    ~ResidualOmpWidth()
    {
        if (old) ::setenv("OMP_NUM_THREADS", old->c_str(), 1);
        else     ::unsetenv("OMP_NUM_THREADS");
    }
    int width() const { return std::atoi(std::getenv("OMP_NUM_THREADS")); }
#endif
    ResidualOmpWidth(const ResidualOmpWidth&) = delete;
    ResidualOmpWidth& operator=(const ResidualOmpWidth&) = delete;
};

/* The first line of each WARNING and HINT entry of the journal @p path
 * before its PROGRESS section, the startup's: "[HH:MM:SS]  WARNING   text"
 * (the tag in eight columns, then two blanks; "* " before the first line
 * of an entry of several lines). */
std::vector<std::string> startup_journal_notes(const fs::path& path)
{
    std::vector<std::string> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("-- PROGRESS", 0) == 0) break;
        if (line.empty() || line[0] != '[') continue;
        const auto close = line.find("]  ");
        if (close == std::string::npos) continue;
        const std::string rest = line.substr(close + 3);
        if (rest.size() < 10) continue;
        std::string tag = rest.substr(0, 8);
        tag.erase(tag.find_last_not_of(' ') + 1);
        if (tag != "WARNING" && tag != "HINT") continue;
        std::string text = rest.substr(10);
        if (text.rfind("* ", 0) == 0) text = text.substr(2);
        out.push_back(text);
    }
    return out;
}

}  // namespace

/* Pins: hpo_run() with CHECK true on a valid setup returns 0 and writes the
 * summary -- the search parameters with their boxes and scales, the
 * optimiser with its stops, the loss terms and the data loaded -- on
 * standard output; the working directory holds the inputs alone (no
 * hpo_run.log, loss.tadah, hpo_search_space.tsv, best_pot.tadah, no pots/ or
 * best_pots/ directory), no evaluation was made (the process-wide evaluation
 * counter is where it was), no journal is open, and CHECK is no longer in
 * the context.
 * Before the feature: CHECK is no configuration key; the context refuses
 * it ("Context::add"). */
TEST_CASE("hpo --check: a valid setup passes, with a summary, no evaluation "
          "and no file",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("valid");
    write_inputs(kOptimizer + kRootLines);
    const auto before = entries(sd.dir);
    const std::size_t evaluations = Host::g_iter_value();

    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    ctx.add("CHECK", "true");
    int status = -1;
    std::string summary;
    {
        CaptureCout out;
        status = tadah::hpo::hpo_run(ctx);
        summary = out.text.str();
    }
    INFO("summary:\n" << summary);
    CHECK(status == 0);
    CHECK(entries(sd.dir) == before);
    CHECK_FALSE(fs::exists("hpo_run.log"));
    CHECK_FALSE(fs::exists("loss.tadah"));
    CHECK_FALSE(fs::exists("hpo_search_space.tsv"));
    CHECK_FALSE(fs::exists("best_pot.tadah"));
    CHECK_FALSE(fs::exists("pots"));
    CHECK_FALSE(fs::exists("best_pots"));
    CHECK(Host::g_iter_value() == evaluations);
    CHECK_FALSE(tadah::hpo::HPOLog::instance().is_open());
    CHECK_FALSE(ctx.has_stored("CHECK"));

    CHECK(contains(summary, "tadah hpo --check: config.hpo passed the "
                            "startup checks"));
    CHECK(contains(summary, "Search parameters: 3"));
    CHECK(contains(summary, "CGRID2B[1]  [0.1, 0.9]  LIN"));
    CHECK(contains(summary, "CGRID2B[2]  [1, 2]  LIN"));
    CHECK(contains(summary, "SGRID2B[2]  [4, 6]  LIN"));
    CHECK(contains(summary, "Optimiser: STRATEGY TYPE PLAIN"));
    CHECK(contains(summary, "STRATEGY  NLOPT:LN_SBPLX; stops: MAXEVAL 5"));
    CHECK(contains(summary, "Loss terms: LOSS L2"));
    CHECK(contains(summary, "PC_ERMSE  target 0, weight 1"));
    CHECK(contains(summary, "Data: 1 training structure, 1 validation "
                            "structure"));
    CHECK(contains(summary, "No candidate was evaluated and no file was "
                            "written."));
}

/* Pins: each startup error of the run is the check's: for a target file
 * with an unknown key, a PC_LAMMPS line whose script does not exist, a
 * PC_ERMSE without a validation configuration (validate_hpo_setup), a
 * training dataset that does not exist (loaded when the run builds its
 * evaluation host), a DUMP or a BDUMP directory in a directory that does
 * not exist, and a DUMP directory that is a file, hpo_run() with CHECK
 * true throws the message hpo_run() without it throws, and writes no file
 * (the run writes its journal, which the check does not open). The check
 * runs first and the run after it, in the same directory: a message can
 * name a file by its absolute path. The run stops before it loads the data
 * (it writes no search-space sidecar, which it writes before) for every
 * error but the dataset's: a DUMP or BDUMP directory is checked at its
 * line, as the target file is read.
 * Before the feature: CHECK is no configuration key; the context refuses
 * it, and the messages differ.
 * Before the DUMP and BDUMP directories were checked at their line: the
 * check passed such a setup (no message), and the run stopped when it
 * created the directory, after it had loaded the data (with a sidecar). */
TEST_CASE("hpo --check: a startup error is the run's, and no file is "
          "written",
          "[HPO][hpo_check]")
{
    struct Case {
        const char* name;
        std::string hpo;
        std::string dbfile;
        bool validation;
        const char* says;  // a part of the run's message
        bool before_data;  // the run stops before it loads the data
    };
    const std::vector<Case> cases = {
        {"unknown key", kOptimizer + kRootLines + "NOSUCHKEY 1\n",
         "train.db", true, "NOSUCHKEY", true},
        {"missing LAMMPS script",
         kOptimizer + kRootLines +
             "PC_LAMMPS --script missing.in --varloss E 0 1\n",
         "train.db", true, "missing.in", true},
        {"PC_ERMSE without validation", kOptimizer + kRootLines, "train.db",
         false, "PC_ERMSE", true},
        {"missing training dataset", kOptimizer + kRootLines, "absent.db",
         true, "absent.db", false},
        {"DUMP directory in a missing directory",
         kOptimizer + kRootLines + "DUMP 1 missing/pots\n", "train.db", true,
         "missing/pots", true},
        {"BDUMP directory in a missing directory",
         kOptimizer + kRootLines + "BDUMP 1 missing/best_pots\n", "train.db",
         true, "missing/best_pots", true},
        {"DUMP directory that is a file",
         kOptimizer + kRootLines + "DUMP 1 config.val\n", "train.db", true,
         "config.val", true},
    };
    for (const Case& c : cases) {
        INFO("case: " << c.name);
        std::string run_error, check_error;
        fresh_process_state();
        ScratchDir sd("error");
        write_inputs(c.hpo);
        const auto before = entries(sd.dir);
        {
            tadah::core::Context ctx = make_ta_context(
                c.dbfile, "config.hpo", c.validation ? "config.val" : "");
            ctx.add("CHECK", "true");
            check_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
            CHECK(entries(sd.dir) == before);
        }
        {
            fresh_process_state();
            tadah::core::Context ctx = make_ta_context(
                c.dbfile, "config.hpo", c.validation ? "config.val" : "");
            run_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
            CHECK(fs::exists("hpo_run.log"));
            CHECK(fs::exists("hpo_search_space.tsv") != c.before_data);
            CHECK_FALSE(fs::exists("missing"));
        }
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(contains(run_error, c.says));
        CHECK(check_error == run_error);
    }
}

/* Pins: a run that stops at startup after it has loaded its data -- here
 * the pre-flight audit, under AUDIT error, refuses RCUT2B 1 (no neighbour
 * in any structure) -- creates neither its DUMP nor its BDUMP directory:
 * they are created once every startup check passed, before the first
 * evaluation. The check stops with the same message, and writes nothing.
 * Before the change: the run's evaluation host created both directories
 * when it was built, before the pre-flight audit: the run left pots/ and
 * best_pots/ behind. */
TEST_CASE("hpo --check: a run that stops at startup creates no DUMP or "
          "BDUMP directory",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("audit_stop");
    write_inputs(kOptimizer + kRootLines);
    const auto before = entries(sd.dir);
    auto audited = [] {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        ctx.remove("RCUT2B");
        ctx.add("RCUT2B", 1.0);
        ctx.add("AUDIT", "error");
        return ctx;
    };
    std::string check_error, run_error;
    {
        tadah::core::Context ctx = audited();
        ctx.add("CHECK", "true");
        check_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        CHECK(entries(sd.dir) == before);
    }
    {
        fresh_process_state();
        tadah::core::Context ctx = audited();
        run_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
    }
    INFO("run: " << run_error << "\ncheck: " << check_error);
    CHECK(contains(run_error, "preflight_audit"));
    CHECK(check_error == run_error);
    CHECK(fs::exists("hpo_run.log"));
    CHECK_FALSE(fs::exists("pots"));
    CHECK_FALSE(fs::exists("best_pots"));
}

/* Pins: the check prints the WARNING and HINT lines the run writes to its
 * journal at startup, which a check, without a journal, would otherwise
 * drop: for a setup without THREADS whose residual OpenMP width is 2, and
 * with PC_ISOLATION THREAD, the summary's first line counts them ("passed
 * the startup checks, with 2 warnings.") and its section "Warnings and
 * hints" lists each with its tag: "THREADS not set: ..." and, in the
 * serial build, "PC_ISOLATION: THREAD (deprecated fallback) -- ...". The
 * same setup as a run writes each of its startup WARNING and HINT lines
 * (those before the PROGRESS section of hpo_run.log) into the journal, and
 * the first line of each is in the check's summary.
 * Before the change: the check reported "passed the startup checks." with
 * no warning (it skipped the worker settings, where both are written, and
 * every journal line of its startup went nowhere). */
TEST_CASE("hpo --check prints the startup warnings and hints the run writes "
          "to its journal",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("warnings");
    write_inputs(kOptimizer + kRootLines + "PC_ISOLATION THREAD\n");
    const ResidualOmpWidth omp(2);
    REQUIRE(omp.width() == 2);
    std::string summary;
    {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        ctx.add("CHECK", "true");
        int status = -1;
        {
            CaptureCout out;
            status = tadah::hpo::hpo_run(ctx);
            summary = out.text.str();
        }
        INFO("summary:\n" << summary);
        REQUIRE(status == 0);
    }
    INFO("summary:\n" << summary);
#ifdef TADAH_HPO_MPI
    const std::size_t n_warnings = 1;
#else
    const std::size_t n_warnings = 2;
    CHECK(contains(summary, "\n  WARNING  PC_ISOLATION: THREAD (deprecated "
                            "fallback) -- "));
#endif
    CHECK(contains(summary, "tadah hpo --check: config.hpo passed the "
                            "startup checks, with " +
                                std::to_string(n_warnings) + " warning" +
                                (n_warnings == 1 ? "" : "s") + ".\n"));
    CHECK(contains(summary, "\nWarnings and hints, which a run writes to "
                            "its journal (hpo_run.log):\n"));
    CHECK(contains(summary, "\n  WARNING  THREADS not set: the pc_lammps "
                            "fan-out and MLIP train/predict run on a SINGLE "
                            "worker."));

    fresh_process_state();
    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    int status = -1;
    {
        CaptureCout quiet;
        status = tadah::hpo::hpo_run(ctx);
    }
    REQUIRE(status == 0);
    const std::vector<std::string> notes =
        startup_journal_notes(sd.dir / "hpo_run.log");
    CHECK(notes.size() == n_warnings);
    for (const std::string& n : notes) {
        INFO("journal: " << n);
        CHECK(contains(summary, n));
    }
}

/* Pins: a check after a run that stopped at startup -- which leaves its
 * journal open, the singleton's -- writes nothing into that journal and
 * closes it: hpo_run.log of the stopped run keeps its size.
 * Before the feature: CHECK is no configuration key; the context refuses
 * it. */
TEST_CASE("hpo --check writes nothing into a journal an earlier run left "
          "open",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("open_journal");
    write_inputs(kOptimizer + kRootLines);
    {
        tadah::core::Context ctx =
            make_ta_context("absent.db", "config.hpo", "config.val");
        const std::string error =
            error_of([&] { return tadah::hpo::hpo_run(ctx); });
        INFO("run: " << error);
        REQUIRE_FALSE(error.empty());
    }
    REQUIRE(tadah::hpo::HPOLog::instance().is_open());
    const auto journal_size = fs::file_size("hpo_run.log");

    fs::create_directories("check");
    fs::current_path("check");
    write_inputs(kOptimizer + kRootLines);
    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    ctx.add("CHECK", "true");
    int status = -1;
    {
        CaptureCout quiet;
        status = tadah::hpo::hpo_run(ctx);
    }
    fs::current_path(sd.dir);
    CHECK(status == 0);
    CHECK_FALSE(tadah::hpo::HPOLog::instance().is_open());
    CHECK(fs::file_size("hpo_run.log") == journal_size);
}

/* Pins: hpo_run() with CHECK false runs (status 0, a best_pot.tadah), and
 * no potential it writes -- best_pot.tadah, the DUMP and BDUMP archives --
 * has a CHECK line: CHECK is not a key of the model.
 * Before the feature: CHECK is no configuration key; the context refuses
 * it. */
TEST_CASE("hpo --check: a run with CHECK false writes no CHECK line into "
          "its potentials",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("check_false");
    write_inputs(kOptimizer + kRootLines);
    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    ctx.add("CHECK", "false");
    int status = -1;
    {
        CaptureCout quiet;
        status = tadah::hpo::hpo_run(ctx);
    }
    REQUIRE(status == 0);
    REQUIRE(fs::exists("best_pot.tadah"));
    REQUIRE(fs::is_directory("pots"));
    REQUIRE(fs::is_directory("best_pots"));
    std::vector<fs::path> potentials = {"best_pot.tadah"};
    for (const char* dir : {"pots", "best_pots"})
        for (const auto& e : fs::directory_iterator(dir))
            potentials.push_back(e.path());
    REQUIRE(potentials.size() >= 3);
    for (const auto& p : potentials) {
        INFO(p.string());
        std::ifstream in(p);
        std::string line;
        bool has_check = false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string key;
            if ((ls >> key) && key == "CHECK") has_check = true;
        }
        CHECK_FALSE(has_check);
    }
}

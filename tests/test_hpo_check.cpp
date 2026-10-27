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
// so a run with CHECK false writes potentials without a CHECK line or a
// CHECK line in their training record. Both check, creating nothing, that
// the run can write its journal, result files and best_pot.tadah into the
// working directory. A run creates its DUMP and BDUMP directories, the one
// inside the other's after it, once it has prepared best_pot.tadah, just
// before its first evaluation: one that stops before creates none.
//
// The fixture is the no-LAMMPS D2_BP Ta setup of test_hpo_termination.cpp
// (a four-atom training structure, one validation dataset, PC_ERMSE), so a
// run costs milliseconds and the whole hpo_run() stack runs in-process.

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/io_retry.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
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
#include <thread>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
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
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);
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

/* Every entry under `dir`, relative to it (lexically: a symbolic link in
 * a loop is listed, not followed). */
std::set<std::string> entries(const fs::path& dir)
{
    std::set<std::string> out;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        out.insert(e.path().lexically_relative(dir).string());
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

/* Standard error into a string while it lives. */
struct CaptureCerr {
    std::ostringstream text;
    std::streambuf* old;
    CaptureCerr() : old(std::cerr.rdbuf(text.rdbuf())) {}
    ~CaptureCerr() { std::cerr.rdbuf(old); }
    CaptureCerr(const CaptureCerr&) = delete;
    CaptureCerr& operator=(const CaptureCerr&) = delete;
};

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
    /* Each term with its loss function (the heading named the last LOSS
     * line, "Loss terms: LOSS L2", and the rows had none). */
    CHECK(contains(summary,
                   "Loss terms:\n  PC_ERMSE  target 0, weight 1, loss L2\n"));
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
 * not exist, a DUMP directory that is a file, and one that is a symbolic
 * link to no file, hpo_run() with CHECK
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
 * created the directory, after it had loaded the data (with a sidecar).
 * Before the symbolic-link check: the check passed the dangling link, and
 * the run stopped when it created the directory ("File exists"). */
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
        const char* dangling = nullptr;  // a symbolic link to no file
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
        {"DUMP directory that is a dangling symbolic link",
         kOptimizer + kRootLines + "DUMP 1 dangling\n", "train.db", true,
         "dangling", true, "dangling"},
    };
    for (const Case& c : cases) {
        INFO("case: " << c.name);
        std::string run_error, check_error;
        fresh_process_state();
        ScratchDir sd("error");
        write_inputs(c.hpo);
        if (c.dangling) fs::create_symlink("nowhere", c.dangling);
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

/* An environment variable set (or, with nullopt, unset) while this lives,
 * as it was afterwards. */
struct ScopedEnv {
    std::string name;
    std::optional<std::string> saved;
    ScopedEnv(std::string n, const std::optional<std::string>& value)
        : name(std::move(n))
    {
        if (const char* e = std::getenv(name.c_str())) saved = e;
        if (value) ::setenv(name.c_str(), value->c_str(), 1);
        else       ::unsetenv(name.c_str());
    }
    ~ScopedEnv()
    {
        if (saved) ::setenv(name.c_str(), saved->c_str(), 1);
        else       ::unsetenv(name.c_str());
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
};

/* Pins: PC_WORKER_TIMEOUT, the deadline of a script task of a worker
 * process, is refused under PC_ISOLATION THREAD, where the scripts run
 * inside the tadah process without a worker deadline: at its line
 * ("Error on line 18: PC_WORKER_TIMEOUT sets the deadline of a script task
 * in a worker process (PC_ISOLATION PROCESS); under PC_ISOLATION THREAD,
 * set on line 17, the scripts run inside the tadah process without a
 * worker deadline, so it would have no effect"), for any value (0 too),
 * with THREAD from the file or from TADAH_PC_ISOLATION ("..., set by
 * TADAH_PC_ISOLATION, ..."); TADAH_PC_WORKER_TIMEOUT is refused naming
 * itself and its value ("TADAH_PC_WORKER_TIMEOUT (got '60') sets ...").
 * The check and the run stop alike, before any evaluation. Accepted: the
 * key under PROCESS (the serial build's default, or TADAH_PC_ISOLATION
 * PROCESS over a file's THREAD), and an empty TADAH_PC_WORKER_TIMEOUT,
 * which sets nothing. In the MPI build THREAD is the only mode: the key is
 * refused there without a PC_ISOLATION line ("..., the only mode of the
 * MPI build of tadah, ...").
 * Before the fix: every one of these setups passed, and the key had no
 * effect under THREAD. */
TEST_CASE("hpo --check: PC_WORKER_TIMEOUT and TADAH_PC_WORKER_TIMEOUT are "
          "refused under PC_ISOLATION THREAD",
          "[HPO][hpo_check][pc_isolation]")
{
    const std::string why =
        " sets the deadline of a script task in a worker process "
        "(PC_ISOLATION PROCESS); under PC_ISOLATION THREAD, ";
    const std::string tail =
        ", the scripts run inside the tadah process without a worker "
        "deadline, so it would have no effect";
    /* kOptimizer and kRootLines are lines 1-16. */
    const auto check_error = [](const std::string& extra) {
        fresh_process_state();
        ScratchDir sd("worker_timeout");
        write_inputs(kOptimizer + kRootLines + extra);
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        ctx.add("CHECK", "true");
        return error_of([&] { return tadah::hpo::hpo_run(ctx); });
    };
    ScopedEnv no_isolation("TADAH_PC_ISOLATION", std::nullopt);
    ScopedEnv no_timeout("TADAH_PC_WORKER_TIMEOUT", std::nullopt);
    for (const std::string v : {"60", "0"}) {
        INFO("PC_WORKER_TIMEOUT " << v);
        CHECK(check_error("PC_ISOLATION THREAD\nPC_WORKER_TIMEOUT " + v +
                          "\n") ==
              "Error on line 18: PC_WORKER_TIMEOUT" + why +
                  "set on line 17" + tail + "\nLine: PC_WORKER_TIMEOUT " + v);
    }
    /* The key before the PC_ISOLATION line. */
    CHECK(check_error("PC_WORKER_TIMEOUT 60\nPC_ISOLATION THREAD\n") ==
          "Error on line 17: PC_WORKER_TIMEOUT" + why + "set on line 18" +
              tail + "\nLine: PC_WORKER_TIMEOUT 60");
    {
        ScopedEnv thread("TADAH_PC_ISOLATION", std::string("THREAD"));
        CHECK(check_error("PC_WORKER_TIMEOUT 60\n") ==
              "Error on line 17: PC_WORKER_TIMEOUT" + why +
                  "set by TADAH_PC_ISOLATION" + tail +
                  "\nLine: PC_WORKER_TIMEOUT 60");
    }
    {
        ScopedEnv timeout("TADAH_PC_WORKER_TIMEOUT", std::string("60"));
        CHECK(check_error("PC_ISOLATION THREAD\n") ==
              "TADAH_PC_WORKER_TIMEOUT (got '60')" + why + "set on line 17" +
                  tail);
    }
    {
        /* An empty value sets nothing. */
        ScopedEnv timeout("TADAH_PC_WORKER_TIMEOUT", std::string());
        CHECK(check_error("PC_ISOLATION THREAD\n") == "");
    }
    /* The run stops as the check does, before any evaluation. */
    {
        fresh_process_state();
        ScratchDir sd("worker_timeout_run");
        write_inputs(kOptimizer + kRootLines +
                     "PC_ISOLATION THREAD\nPC_WORKER_TIMEOUT 60\n");
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        CHECK(error_of([&] { return tadah::hpo::hpo_run(ctx); }) ==
              "Error on line 18: PC_WORKER_TIMEOUT" + why + "set on line 17" +
                  tail + "\nLine: PC_WORKER_TIMEOUT 60");
        CHECK_FALSE(fs::exists("loss.tadah"));
    }
#ifndef TADAH_HPO_MPI
    CHECK(check_error("PC_WORKER_TIMEOUT 60\n") == "");
    {
        ScopedEnv process("TADAH_PC_ISOLATION", std::string("PROCESS"));
        CHECK(check_error("PC_ISOLATION THREAD\nPC_WORKER_TIMEOUT 60\n") ==
              "");
    }
#else
    CHECK(check_error("PC_WORKER_TIMEOUT 60\n") ==
          "Error on line 17: PC_WORKER_TIMEOUT" + why +
              "the only mode of the MPI build of tadah" + tail +
              "\nLine: PC_WORKER_TIMEOUT 60");
#endif
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
 * hints" lists each with its tag: "THREADS is not set: ..." and, in the
 * serial build, "PC_ISOLATION THREAD is deprecated in the serial build:
 * ..." (they were "THREADS not set: the pc_lammps fan-out ..." and
 * "PC_ISOLATION: THREAD (deprecated fallback) -- ..."). The
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
    CHECK(contains(summary, "\n  WARNING  PC_ISOLATION THREAD is deprecated "
                            "in the serial build: "));
#endif
    CHECK(contains(summary, "tadah hpo --check: config.hpo passed the "
                            "startup checks, with " +
                                std::to_string(n_warnings) + " warning" +
                                (n_warnings == 1 ? "" : "s") + ".\n"));
    CHECK(contains(summary, "\nWarnings and hints, which a run writes to "
                            "its journal (hpo_run.log):\n"));
    CHECK(contains(summary, "\n  WARNING  THREADS is not set: one worker runs "
                            "the work of the evaluations"));

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
 * has a CHECK line: CHECK is not a key of the model; nor does its training
 * record list CHECK ("# CHECK false"): hpo_run() takes CHECK out of the
 * context before the record is taken (the record of a tadah train
 * potential lists the CHECK line of its configuration, which tadah train
 * does not read).
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
        bool has_check = false, records_check = false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string key;
            if ((ls >> key) && key == "CHECK") has_check = true;
            if (line.rfind("# CHECK", 0) == 0) records_check = true;
        }
        CHECK_FALSE(has_check);
        CHECK_FALSE(records_check);
    }
}

/* Pins: a check that stops at a startup error after the run's startup has
 * warned writes those WARNING and HINT lines to standard error before the
 * error propagates, as a run has them in its journal: for a setup without
 * THREADS whose residual OpenMP width is 2, refused by the pre-flight audit
 * (AUDIT error, RCUT2B 1), standard error reads "tadah hpo --check: the
 * startup checks stopped at the error below.", then "Warnings and hints,
 * which a run writes to its journal (hpo_run.log):" and "  WARNING  THREADS
 * not set: ...", and hpo_run() throws the audit's error.
 * Before the change: nothing was written (the check dropped every journal
 * line of its startup). */
TEST_CASE("hpo --check: the startup warnings before an error are written "
          "to standard error",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("warnings_error");
    write_inputs(kOptimizer + kRootLines);
    const ResidualOmpWidth omp(2);
    REQUIRE(omp.width() == 2);
    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    ctx.remove("RCUT2B");
    ctx.add("RCUT2B", 1.0);
    ctx.add("AUDIT", "error");
    ctx.add("CHECK", "true");
    std::string error, err;
    {
        CaptureCerr capture;
        error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        err = capture.text.str();
    }
    INFO("error: " << error << "\nstandard error:\n" << err);
    CHECK(contains(error, "preflight_audit"));
    CHECK(contains(err, "tadah hpo --check: the startup checks stopped at "
                        "the error below.\nWarnings and hints, which a run "
                        "writes to its journal (hpo_run.log):\n"));
    CHECK(contains(err, "\n  WARNING  THREADS is not set: one worker runs the "
                        "work of the evaluations"));
}

namespace {

/* The permissions of @p dir at read and search alone while it lives (the
 * owner's), restored to the owner's all after. */
struct ReadOnlyDir {
    fs::path dir;
    explicit ReadOnlyDir(fs::path d) : dir(std::move(d))
    {
        fs::permissions(dir, fs::perms::owner_read | fs::perms::owner_exec);
    }
    ~ReadOnlyDir()
    {
        std::error_code ec;
        fs::permissions(dir, fs::perms::owner_all, ec);
    }
    ReadOnlyDir(const ReadOnlyDir&) = delete;
    ReadOnlyDir& operator=(const ReadOnlyDir&) = delete;
};

/* The names of the entries of @p dir whose name starts with @p prefix. */
std::size_t count_prefixed(const fs::path& dir, const std::string& prefix)
{
    std::size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().rfind(prefix, 0) == 0) ++n;
    return n;
}

}  // namespace

/* Pins: a working directory the run cannot write into (read and search
 * permission only) stops the check and the run with one message, before
 * the journal and before any data is loaded: "the working directory
 * '<dir>' is not writable: tadah hpo writes its journal (hpo_run.log), its
 * result files and best_pot.tadah there", and nothing is created -- no
 * file in the working directory, and no DUMP directory elsewhere (the
 * target file names one in a writable directory).
 * Before the change: the check passed (status 0), and the run stopped at
 * the lock of best_pot.tadah ("cannot open durable-best lock
 * 'best_pot.tadah.lock': Permission denied") after it had created its
 * DUMP directory. Skipped as root, whose writes no permission stops. */
TEST_CASE("hpo --check: a working directory the run cannot write into "
          "stops the check and the run alike",
          "[HPO][hpo_check]")
{
    if (::geteuid() == 0)
        TADAH_SKIP("running as root: file permissions are not enforced");
    fresh_process_state();
    ScratchDir sd("read_only");
    const fs::path elsewhere = sd.dir / "elsewhere";
    fs::create_directories(elsewhere);
    const fs::path work = sd.dir / "work";
    fs::create_directories(work);
    fs::current_path(work);
    write_inputs(kOptimizer + kRootLines + "DUMP 1 " +
                 (elsewhere / "pots").string() + "\n");
    const auto before = entries(sd.dir);
    std::string check_error, run_error;
    {
        const ReadOnlyDir ro(work);
        {
            tadah::core::Context ctx =
                make_ta_context("train.db", "config.hpo", "config.val");
            ctx.add("CHECK", "true");
            check_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        }
        fresh_process_state();
        {
            tadah::core::Context ctx =
                make_ta_context("train.db", "config.hpo", "config.val");
            run_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        }
    }
    fs::current_path(sd.dir);
    INFO("run: " << run_error << "\ncheck: " << check_error);
    CHECK(contains(run_error, "the working directory '" +
                                  fs::canonical(work).string() +
                                  "' is not writable: tadah hpo writes its "
                                  "journal (hpo_run.log), its result files "
                                  "and best_pot.tadah there"));
    CHECK(check_error == run_error);
    CHECK(entries(sd.dir) == before);
    CHECK_FALSE(fs::exists(elsewhere / "pots"));
}

/* Pins: the run creates its DUMP and BDUMP directories once it has prepared
 * best_pot.tadah (its lock taken), just before its first evaluation: a run
 * whose lock cannot be opened (an injected EACCES at durable.lock_open, a
 * local filesystem) stops with that error ("cannot open durable-best lock
 * 'best_pot.tadah.lock'") and leaves neither pots/ nor best_pots/ behind.
 * Before the change: the run created both directories before it prepared
 * best_pot.tadah, and left them behind. */
TEST_CASE("hpo --check: a run that stops at the lock of best_pot.tadah "
          "creates no DUMP or BDUMP directory",
          "[HPO][hpo_check]")
{
    fresh_process_state();
    ScratchDir sd("lock_stop");
    write_inputs(kOptimizer + kRootLines);
    std::string run_error;
    {
        tadah::hpo::io::testing::ScopedNetworkFs local(false);
        tadah::hpo::io::testing::ScopedFaults faults(
            "durable.lock_open,EACCES,always");
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        run_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
    }
    INFO("run: " << run_error);
    CHECK(contains(run_error, "durable-best lock"));
    CHECK(fs::exists("hpo_run.log"));
    CHECK_FALSE(fs::exists("pots"));
    CHECK_FALSE(fs::exists("best_pots"));
}

/* Pins: a DUMP directory inside the BDUMP directory, or the reverse, is
 * accepted by the check and by the run, in either order of the lines, and
 * the run creates the outer directory first and writes into both: for
 * "DUMP 1 archive" with "BDUMP 1 archive/best", and for "BDUMP 1 archive"
 * before "DUMP 1 archive/all", the check returns 0 and creates nothing, and
 * the run returns 0 with archives pot_<n>.tadah in the DUMP directory and
 * best_pot_<n>.tadah in the BDUMP directory.
 * Before the change: the inner directory's line was refused ("the
 * directory 'archive' does not exist"), by the check and the run alike. */
TEST_CASE("hpo --check: a DUMP directory inside the BDUMP directory, or the "
          "reverse, is accepted and written into",
          "[HPO][hpo_check]")
{
    struct Layout {
        const char* name;
        std::string lines;
        const char* dump;
        const char* bdump;
    };
    const std::vector<Layout> layouts = {
        {"DUMP outer", "DUMP 1 archive\nBDUMP 1 archive/best\n", "archive",
         "archive/best"},
        {"BDUMP outer, first", "BDUMP 1 archive\nDUMP 1 archive/all\n",
         "archive/all", "archive"},
    };
    for (const Layout& l : layouts) {
        INFO(l.name);
        fresh_process_state();
        ScratchDir sd("nested");
        write_inputs(kOptimizer +
                     "LOSS L2\nFAILSCORE 1e9\nPC_ERMSE 0 1\n"
                     "OPTIM CGRID2B (1) 0.1 0.9\n" + l.lines);
        const auto before = entries(sd.dir);
        std::string check_error;
        {
            tadah::core::Context ctx =
                make_ta_context("train.db", "config.hpo", "config.val");
            ctx.add("CHECK", "true");
            check_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        }
        INFO("check: " << check_error);
        CHECK(check_error.empty());
        CHECK(entries(sd.dir) == before);
        fresh_process_state();
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        int status = -1;
        std::string run_error;
        {
            CaptureCout quiet;
            try {
                status = tadah::hpo::hpo_run(ctx);
            } catch (const std::exception& e) {
                run_error = e.what();
            }
        }
        INFO("run: " << run_error);
        REQUIRE(status == 0);
        REQUIRE(fs::is_directory(l.dump));
        REQUIRE(fs::is_directory(l.bdump));
        CHECK(count_prefixed(l.dump, "pot_") >= 1u);
        CHECK(count_prefixed(l.bdump, "best_pot_") >= 1u);
    }
}

namespace {

/* Owner read permission alone for the file @p path. */
void read_only_file(const fs::path& path)
{
    fs::permissions(path, fs::perms::owner_read);
}

/* What the check and the run of the setup in the working directory throw
 * ("" when one returns), in that order; each starts as a new process
 * would. @p edit changes the training configuration of both. */
std::pair<std::string, std::string> check_and_run_errors(
    const std::function<void(tadah::core::Context&)>& edit = {})
{
    std::pair<std::string, std::string> errors;
    fresh_process_state();
    {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        if (edit) edit(ctx);
        ctx.add("CHECK", "true");
        errors.first = error_of([&] { return tadah::hpo::hpo_run(ctx); });
    }
    fresh_process_state();
    {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        if (edit) edit(ctx);
        errors.second = error_of([&] { return tadah::hpo::hpo_run(ctx); });
    }
    return errors;
}

/* "'<file>' in the working directory '<cwd>' " */
std::string in_working_directory(const std::string& file)
{
    return "'" + file + "' in the working directory '" +
           fs::current_path().string() + "' ";
}

}  // namespace

/* Pins: an existing output file of the run is opened as the run opens it
 * (to write, the lock to read and write), without creating or truncating
 * it, and one the run cannot open -- loss.tadah, hpo_run.log and
 * best_pot.tadah.lock with owner read permission alone, and
 * best_pot.tadah.lock with owner write permission alone -- stops the check
 * and the run with one message, the open's own error, before the journal
 * and before any data is loaded: "'loss.tadah' in the working directory
 * '<dir>' cannot be opened: Permission denied (tadah hpo rewrites it)"
 * (the same for hpo_run.log), "'best_pot.tadah.lock' ... cannot be opened:
 * Permission denied (tadah hpo opens it to lock best_pot.tadah)"; the run
 * writes no hpo_run.log (the one that exists keeps its line) and no
 * search-space file, and the working directory is as it was.
 * Before the open: the messages read "is not writable: tadah hpo rewrites
 * it", and a write-only lock, which the run cannot open to read, passed
 * the check (the run stopped at it after writing its journal).
 * Before the change: the result files and the lock were checked after the
 * journal was opened (the run wrote hpo_run.log, its error in it), and the
 * lock's message said "tadah hpo rewrites it". Before the output files were
 * checked at all: the check passed, and the run lost the rows of loss.tadah
 * or stopped at the lock. Skipped as root, whose writes no permission
 * stops. */
TEST_CASE("hpo --check: an output file the run cannot write stops the "
          "check and the run alike, before the journal",
          "[HPO][hpo_check]")
{
    if (::geteuid() == 0)
        TADAH_SKIP("running as root: file permissions are not enforced");
    struct Case {
        const char* file;
        fs::perms perms;
        const char* says;
    };
    const std::string denied = "cannot be opened: Permission denied ";
    const std::vector<Case> cases = {
        {"loss.tadah", fs::perms::owner_read,
         "(tadah hpo rewrites it)"},
        {"hpo_run.log", fs::perms::owner_read, "(tadah hpo rewrites it)"},
        {"best_pot.tadah.lock", fs::perms::owner_read,
         "(tadah hpo opens it to lock best_pot.tadah)"},
        {"best_pot.tadah.lock", fs::perms::owner_write,
         "(tadah hpo opens it to lock best_pot.tadah)"},
    };
    for (const Case& c : cases) {
        INFO("file: " << c.file << ", owner "
                      << (c.perms == fs::perms::owner_read ? "read" : "write")
                      << " permission alone");
        ScratchDir sd("read_only_output");
        write_inputs(kOptimizer + kRootLines);
        std::ofstream(c.file) << "kept\n";
        fs::permissions(c.file, c.perms);
        const auto before = entries(sd.dir);
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory(c.file) + denied + c.says);
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
        CHECK_FALSE(fs::exists("hpo_search_space.tsv"));
        fs::permissions(c.file, fs::perms::owner_read);
        std::ifstream in(c.file);
        std::string line;
        std::getline(in, line);
        CHECK(line == "kept");
        if (std::string(c.file) != "hpo_run.log")
            CHECK_FALSE(fs::exists("hpo_run.log"));
    }
}

/* Pins: the files the run writes through (here loss.tadah) are checked
 * where the operating system puts them, every symbolic link followed, and
 * best_pot.tadah, which the run replaces, by its own name:
 *   - a directory named best_pot.tadah stops the check and the run with
 *     one message, before the journal: "'best_pot.tadah' in the working
 *     directory '<dir>' is a directory: tadah hpo writes a file of that
 *     name";
 *   - loss.tadah a symbolic link into a directory that does not exist
 *     (nowhere/loss), directly or through a second link (chain ->
 *     nowhere/loss), stops both: "'loss.tadah' in the working directory
 *     '<dir>' is a symbolic link to '<dir>/nowhere/loss', which cannot be
 *     created: the directory '<dir>/nowhere' does not exist";
 *   - loss.tadah in a loop of links (loss.tadah -> loop -> loss.tadah)
 *     stops both with the open's own error: "... cannot be opened: Too
 *     many levels of symbolic links (tadah hpo rewrites it)";
 *   - loss.tadah a link to a name that ends in '/' (lost/) stops both: a
 *     file cannot be created there ("... is a symbolic link to
 *     '<dir>/lost', which cannot be created: a file name cannot end in '/'
 *     or '/.'");
 *   - loss.tadah a link to a file not yet created in a directory that
 *     exists, directly (out/loss) or through a second link (chain ->
 *     out/loss), is accepted: the run creates out/loss through the links
 *     and writes its rows there;
 *   - best_pot.tadah with owner read permission alone, a link into a
 *     directory that does not exist and a link to a directory are
 *     accepted: the run replaces the name with a regular file (no longer
 *     "old"; the link's target is left as it was).
 * Before the change: a chain was followed one link only (accepted into a
 * missing directory: the run lost every row of loss.tadah), a loop was
 * refused as not writable, and best_pot.tadah was checked through its link
 * (both links refused, though the run replaces them). */
TEST_CASE("hpo --check: best_pot.tadah as a directory and the symbolic "
          "links of the output files",
          "[HPO][hpo_check]")
{
    {
        INFO("best_pot.tadah is a directory");
        ScratchDir sd("best_pot_directory");
        write_inputs(kOptimizer + kRootLines);
        fs::create_directories("best_pot.tadah");
        const auto before = entries(sd.dir);
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory("best_pot.tadah") +
                               "is a directory: tadah hpo writes a file of "
                               "that name");
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
    }
    {
        for (const bool chain : {false, true}) {
            INFO("loss.tadah links into a directory that does not exist"
                 << (chain ? ", through a second link" : ""));
            ScratchDir sd("dangling_result");
            write_inputs(kOptimizer + kRootLines);
            if (chain) {
                fs::create_symlink("nowhere/loss", "chain");
                fs::create_symlink("chain", "loss.tadah");
            } else {
                fs::create_symlink("nowhere/loss", "loss.tadah");
            }
            const auto before = entries(sd.dir);
            const auto [check_error, run_error] = check_and_run_errors();
            INFO("run: " << run_error << "\ncheck: " << check_error);
            const std::string cwd = fs::current_path().string();
            CHECK(run_error == in_working_directory("loss.tadah") +
                                   "is a symbolic link to '" + cwd +
                                   "/nowhere/loss', which cannot be created: "
                                   "the directory '" + cwd +
                                   "/nowhere' does not exist");
            CHECK(check_error == run_error);
            CHECK(entries(sd.dir) == before);
        }
    }
    {
        INFO("loss.tadah in a loop of links");
        ScratchDir sd("loop_result");
        write_inputs(kOptimizer + kRootLines);
        fs::create_symlink("loss.tadah", "loop");
        fs::create_symlink("loop", "loss.tadah");
        const auto before = entries(sd.dir);
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory("loss.tadah") +
                               "cannot be opened: Too many levels of "
                               "symbolic links (tadah hpo rewrites it)");
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
    }
    {
        INFO("loss.tadah links to a name that ends in '/'");
        ScratchDir sd("slash_result");
        write_inputs(kOptimizer + kRootLines);
        fs::create_symlink("lost/", "loss.tadah");
        const auto before = entries(sd.dir);
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory("loss.tadah") +
                               "is a symbolic link to '" +
                               fs::current_path().string() +
                               "/lost', which cannot be created: a file name "
                               "cannot end in '/' or '/.'");
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
    }
    for (const bool chain : {false, true}) {
        INFO("loss.tadah links to a file not yet created"
             << (chain ? ", through a second link" : ""));
        ScratchDir sd("link_result");
        write_inputs(kOptimizer + kRootLines);
        fs::create_directories("out");
        if (chain) {
            fs::create_symlink("out/loss", "chain");
            fs::create_symlink("chain", "loss.tadah");
        } else {
            fs::create_symlink("out/loss", "loss.tadah");
        }
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(check_error.empty());
        CHECK(run_error.empty());
        REQUIRE(fs::exists("out/loss"));
        CHECK(fs::file_size("out/loss") > 0u);
    }
    for (const char* target : {"nowhere/best", "a_directory"}) {
        INFO("best_pot.tadah links to " << target);
        ScratchDir sd("link_best");
        write_inputs(kOptimizer + kRootLines);
        fs::create_directories("a_directory");
        fs::create_symlink(target, "best_pot.tadah");
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(check_error.empty());
        CHECK(run_error.empty());
        CHECK(fs::is_regular_file(fs::symlink_status("best_pot.tadah")));
        CHECK(fs::is_empty("a_directory"));
        CHECK_FALSE(fs::exists("nowhere"));
    }
    {
        INFO("best_pot.tadah with owner read permission alone");
        ScratchDir sd("read_only_best");
        write_inputs(kOptimizer + kRootLines);
        std::ofstream("best_pot.tadah") << "old\n";
        read_only_file("best_pot.tadah");
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(check_error.empty());
        CHECK(run_error.empty());
        std::ifstream in("best_pot.tadah");
        std::string line;
        std::getline(in, line);
        CHECK(line != "old");
    }
}

/* Pins: hpo_optim_resolved.hpo, which the run writes only when a box of
 * the search space comes from an AUTO rule, is checked only then: with
 * explicit boxes alone a read-only one (owner read permission) is
 * accepted -- the check passes, the run returns 0 and leaves it as it was
 * ("# stale") --, and with an AUTO box (OPTIM SGRID2B (2) AUTO) the check
 * and the run refuse it with one message, "'hpo_optim_resolved.hpo' in the
 * working directory '<dir>' cannot be opened: Permission denied (tadah hpo
 * rewrites it)".
 * Before the change: it was refused whatever the boxes (v1-dev ran with
 * explicit boxes and left it alone). Skipped as root, whose writes no
 * permission stops. */
TEST_CASE("hpo --check: hpo_optim_resolved.hpo is checked only when a box "
          "comes from an AUTO rule",
          "[HPO][hpo_check]")
{
    if (::geteuid() == 0)
        TADAH_SKIP("running as root: file permissions are not enforced");
    {
        INFO("explicit boxes");
        ScratchDir sd("resolved_explicit");
        write_inputs(kOptimizer + kRootLines);
        std::ofstream("hpo_optim_resolved.hpo") << "# stale\n";
        read_only_file("hpo_optim_resolved.hpo");
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(check_error.empty());
        CHECK(run_error.empty());
        std::ifstream in("hpo_optim_resolved.hpo");
        std::string line;
        std::getline(in, line);
        CHECK(line == "# stale");
    }
    {
        INFO("an AUTO box");
        ScratchDir sd("resolved_auto");
        write_inputs(kOptimizer +
                     "LOSS L2\nFAILSCORE 1e9\nPC_ERMSE 0 1\n"
                     "OPTIM CGRID2B (1) 0.1 0.9\nOPTIM SGRID2B (2) AUTO\n");
        std::ofstream("hpo_optim_resolved.hpo") << "# stale\n";
        read_only_file("hpo_optim_resolved.hpo");
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory("hpo_optim_resolved.hpo") +
                               "cannot be opened: Permission denied (tadah "
                               "hpo rewrites it)");
        CHECK(check_error == run_error);
    }
}

/* Pins the agreement of the ranks on a startup step that one rank can fail
 * alone (its rank-suffixed hpo_run.log.r1, loss.tadah.r1 or
 * best_pot.tadah.r1.lock): agreed_startup_error(), the error each rank
 * throws once the ranks' failure flags are combined, is the rank's own
 * error where the step failed on it, "the check of the output files failed
 * on another MPI rank" where it failed on another rank alone, and "" (no
 * error) where it failed on no rank; run_on_every_rank() of a step that
 * throws throws its message (a serial run is one rank), and of a step that
 * does not, nothing. What the agreement prevents -- a rank waiting in a
 * later collective for a peer that stopped -- needs the MPI build
 * (TADAH_HPO_MPI) and several ranks.
 * Before the change: the steps had no agreement; neither function existed
 * (the case does not compile). */
TEST_CASE("hpo --check: the ranks agree on a startup step one rank fails",
          "[HPO][hpo_check][mpi_agreement]")
{
    using tadah::hpo::agreed_startup_error;
    const std::string what = "the check of the output files";
    CHECK(agreed_startup_error("loss.tadah.r1 is not writable", true, what) ==
          "loss.tadah.r1 is not writable");
    CHECK(agreed_startup_error("", true, what) ==
          "the check of the output files failed on another MPI rank");
    CHECK(agreed_startup_error("", false, what).empty());

    std::string thrown;
    try {
        tadah::hpo::run_on_every_rank(what, [] {
            throw std::runtime_error("loss.tadah is not writable");
        });
    } catch (const std::exception& e) {
        thrown = e.what();
    }
    CHECK(thrown == "loss.tadah is not writable");
    CHECK_NOTHROW(tadah::hpo::run_on_every_rank(what, [] {}));
}

/* Pins: an output file that is a named pipe is not opened by the check,
 * which would end the stream of its reader, and the run writes into it as
 * into a file: with hpo_run.log a FIFO and a reader waiting on it (a
 * thread of this process), a child process (this test binary, the hidden
 * case below) makes the check, then the run, of a valid setup; both
 * return 0, and the reader receives the run's journal ("Run Journal"), not
 * an end of file at the check. A regression that hangs is ended after
 * 120 s and fails.
 * Before the change: the check opened and closed the FIFO's write end, so
 * the reader got an end of file and left, and the run's own open of the
 * journal could not complete (the run was refused, or blocked). */
TEST_CASE("hpo --check: a named pipe as hpo_run.log is not opened by the "
          "check, and the run writes its journal into it",
          "[HPO][hpo_check]")
{
    ScratchDir sd("fifo_journal");
    write_inputs(kOptimizer + kRootLines);
    REQUIRE(::mkfifo("hpo_run.log", 0600) == 0);
    std::string journal;
    std::atomic<bool> read_done{false};
    std::thread reader([&] {
        std::ifstream in("hpo_run.log");  // waits for a writer
        std::ostringstream text;
        text << in.rdbuf();
        journal = text.str();
        read_done = true;
    });
    const fs::path out = sd.dir / "child.out";
    std::cout.flush();
    std::fflush(nullptr);
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        std::FILE* f = std::freopen(out.c_str(), "w", stdout);
        (void)f;
        ::dup2(::fileno(stdout), 2);
        ::execl("/proc/self/exe", "test_hpo", "[.hpo_check_fifo_child]",
                static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    bool finished = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(child, &status, WNOHANG) == child) {
            finished = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!finished) {
        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
    }
    if (!read_done) {
        /* End the reader's wait: a writer that closes at once. */
        const int fd = ::open("hpo_run.log", O_WRONLY | O_NONBLOCK);
        if (fd >= 0) ::close(fd);
    }
    reader.join();
    std::ifstream child_out(out);
    std::ostringstream child_text;
    child_text << child_out.rdbuf();
    INFO("child:\n" << child_text.str() << "\njournal:\n" << journal);
    CHECK(finished);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    CHECK(contains(journal, "Run Journal"));
}

/* The child of the case above: in the working directory it inherits, the
 * check and then the run of the setup there; each must return 0. */
TEST_CASE("hpo --check: the check and the run of a setup whose journal is a "
          "named pipe", "[.hpo_check_fifo_child]")
{
    fresh_process_state();
    {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        ctx.add("CHECK", "true");
        CaptureCout quiet;
        REQUIRE(tadah::hpo::hpo_run(ctx) == 0);
    }
    fresh_process_state();
    {
        tadah::core::Context ctx =
            make_ta_context("train.db", "config.hpo", "config.val");
        CaptureCout quiet;
        REQUIRE(tadah::hpo::hpo_run(ctx) == 0);
    }
}

namespace {

/* A Unix socket bound at @p name (the socket file stays once the socket
 * is closed). */
void make_socket_file(const std::string& name)
{
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    REQUIRE(name.size() < sizeof(addr.sun_path));
    std::copy(name.begin(), name.end(), addr.sun_path);
    const int bound =
        ::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    ::close(fd);
    REQUIRE(bound == 0);
}

}  // namespace

/* Pins: the special files among the output files are checked as the run
 * will use them:
 *   - a socket as best_pot.tadah.lock or loss.tadah stops the check and the
 *     run alike, the run's open failing with ENXIO: "'best_pot.tadah.lock'
 *     in the working directory '<dir>' cannot be opened: No such device or
 *     address (tadah hpo opens it to lock best_pot.tadah)", "'loss.tadah'
 *     ... cannot be opened: No such device or address (tadah hpo rewrites
 *     it)";
 *   - a named pipe as loss.tadah, a file the run reopens for each row it
 *     writes, stops both: "'loss.tadah' ... is a named pipe: tadah hpo
 *     reopens the file for each row it writes, and a pipe would block it"
 *     (the run is made only once the check has refused it: a run that took
 *     the pipe would wait for ever);
 *   - loss.tadah a symbolic link to /dev/null, a device the run can open,
 *     is accepted: the check passes and the run returns 0.
 * (A named pipe as hpo_run.log, which the run opens once, is accepted: the
 * case above.)
 * Before the change: a socket and a named pipe passed the check (their
 * permission alone was checked); the run then stopped at the lock socket
 * or wrote no history, and waited for ever on the pipe. */
TEST_CASE("hpo --check: sockets, named pipes and devices among the output "
          "files", "[HPO][hpo_check]")
{
    const std::string nxio = "cannot be opened: No such device or address ";
    struct Case {
        const char* file;
        const char* purpose;
    };
    for (const Case& c :
         {Case{"best_pot.tadah.lock",
               "(tadah hpo opens it to lock best_pot.tadah)"},
          Case{"loss.tadah", "(tadah hpo rewrites it)"}}) {
        INFO("a socket as " << c.file);
        ScratchDir sd("socket_output");
        write_inputs(kOptimizer + kRootLines);
        make_socket_file(c.file);
        const auto before = entries(sd.dir);
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(run_error == in_working_directory(c.file) + nxio + c.purpose);
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
    }
    {
        INFO("a named pipe as loss.tadah");
        ScratchDir sd("fifo_result");
        write_inputs(kOptimizer + kRootLines);
        REQUIRE(::mkfifo("loss.tadah", 0600) == 0);
        const auto before = entries(sd.dir);
        std::string check_error, run_error;
        fresh_process_state();
        {
            tadah::core::Context ctx =
                make_ta_context("train.db", "config.hpo", "config.val");
            ctx.add("CHECK", "true");
            check_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        }
        INFO("check: " << check_error);
        REQUIRE_FALSE(check_error.empty());
        fresh_process_state();
        {
            tadah::core::Context ctx =
                make_ta_context("train.db", "config.hpo", "config.val");
            run_error = error_of([&] { return tadah::hpo::hpo_run(ctx); });
        }
        INFO("run: " << run_error);
        CHECK(run_error == in_working_directory("loss.tadah") +
                               "is a named pipe: tadah hpo reopens the file "
                               "for each row it writes, and a pipe would "
                               "block it");
        CHECK(check_error == run_error);
        CHECK(entries(sd.dir) == before);
    }
    {
        INFO("loss.tadah a link to /dev/null");
        ScratchDir sd("dev_null_result");
        write_inputs(kOptimizer + kRootLines);
        fs::create_symlink("/dev/null", "loss.tadah");
        const auto [check_error, run_error] = check_and_run_errors();
        INFO("run: " << run_error << "\ncheck: " << check_error);
        CHECK(check_error.empty());
        CHECK(run_error.empty());
    }
}

/* Pins: tadah hpo hands the critical-OPTIM audit the entries its OPTIM
 * lines search (HPOTargets::optimap's opt flags), so a joined line is
 * judged by each descriptor's own entries: for TYPE2B D2_mJoin of two D2_BP
 * (RCUT2B 5 5) under AUDIT warn, OPTIM RCUT2B (1) and OPTIM SGRID2B (2),
 * the check prints the warning "RCUT2B(2) is a critical parameter that the
 * seed fixes and no OPTIM line searches" for the second D2_BP, and none
 * for RCUT2B(1), which the search holds.
 * Before the fix: tadah hpo passed the key names, so OPTIM RCUT2B (1)
 * counted as a search of the second descriptor's RCUT2B too: no warning. */
TEST_CASE("hpo --check: the critical-OPTIM audit judges a joined line by "
          "the entries the OPTIM lines search",
          "[HPO][hpo_check][join]")
{
    fresh_process_state();
    ScratchDir sd("audit_entries");
    write_inputs("LOSS L2\nFAILSCORE 1e9\nPC_ERMSE 0 1\n"
                 "OPTIM RCUT2B (1) 4.5 5.5\nOPTIM SGRID2B (2) 4.0 6.0\n" +
                 kOptimizer);
    tadah::core::Context ctx =
        make_ta_context("train.db", "config.hpo", "config.val");
    ctx.remove("TYPE2B");
    ctx.add("TYPE2B", "D2_mJoin D2_BP 1 1 Ta Ta D2_BP 1 1 Ta Ta");
    ctx.remove("RCTYPE2B");
    ctx.add("RCTYPE2B", "Cut_Cos Cut_Cos");
    ctx.remove("RCUT2B");
    ctx.add("RCUT2B", "5.0 5.0");
    ctx.add("AUDIT", "warn");
    ctx.add("CHECK", "true");
    /* The audit's findings are WARNING lines of core::Logger, which the
     * tadah command line prints (its level) and a test binary does not (its
     * level is ERROR). */
    struct LoggerLevel {
        tadah::core::LogLevel old = tadah::core::Logger::instance().getLogLevel();
        LoggerLevel() {
            tadah::core::Logger::instance().setLogLevel(
                tadah::core::LogLevel::WARNING);
        }
        ~LoggerLevel() { tadah::core::Logger::instance().setLogLevel(old); }
    } level;
    std::string out;
    int status = -1;
    {
        CaptureCout capture;
        status = tadah::hpo::hpo_run(ctx);
        out = capture.text.str();
    }
    INFO("output:\n" << out);
    REQUIRE(status == 0);
    const std::string fixed =
        " is a critical parameter that the seed fixes and no OPTIM line "
        "searches";
    CHECK(contains(out, "D2_BP.RCUT2B: RCUT2B(2)" + fixed));
    CHECK_FALSE(contains(out, "RCUT2B(1)" + fixed));
}

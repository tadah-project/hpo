/* How an HPO run ends: what it leaves on disk and what it reports.
 *
 * The contract (nfp_configuration.rst, "Stopping a run"):
 *  - best_pot.tadah always holds the incumbent: the lowest feasible loss of
 *    every evaluation that completed, whatever stopped the run;
 *  - the history files are complete: loss.tadah has one row per evaluation,
 *    numbered 1, 2, 3, ... without a gap, and best_params.tadah one row per
 *    improvement, the last improvement's being the incumbent's params.tadah
 *    row;
 *  - the journal trailer ("Optimiser status") states why the run ended, and
 *    a run stopped by SIGINT/SIGTERM/SIGUSR2 returns 128 + the signal
 *    number (130, 143, 140 on Linux);
 *  - a signal is honoured promptly by every backend and strategy: after it
 *    arrives, only the evaluations already running are completed.
 *
 * The fixture is the no-LAMMPS D2_BP Ta setup of test_hpo_host_e2e.cpp (a
 * four-atom training structure, one validation dataset, PC_ERMSE), so an
 * evaluation costs milliseconds and the whole hpo_run() stack runs
 * in-process. QFLUSH 1000 keeps every result queued until the run's final
 * flush, so a missing flush shows up as missing rows.
 *
 * The real-signal cases are hidden ([.shutdown_*]) and each is registered as
 * its own CTest entry (tests/CMakeLists.txt): CORE's stop flag can only be
 * raised by a real signal and is never lowered, so every such case needs a
 * fresh process. */

#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>      // core::ShutdownRequested
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/result_drain.h>
#include <tadah/hpo/run_driver.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/run_stop.h>
#include <tadah/hpo/runners.h>

#include "anneal_test_support.h"

#include <dlib/matrix.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

constexpr double kFailScore = 1e9;

/* A scratch working directory: hpo_run() writes every output file into the
 * current directory. */
struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir()
    {
        dir = fs::temp_directory_path() /
              ("tadah_hpo_termination_" +
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

tadah::core::Context make_ta_context(const fs::path& dbfile,
                                     const fs::path& hpo_cfg,
                                     const fs::path& val_cfg)
{
    tadah::core::Context c;
    c.add("DBFILE",   dbfile.string());
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
    c.add("HPOTARGET",  hpo_cfg.string());
    c.add("VALIDATION", val_cfg.string());
    return c;
}

/* One hpo_run() setup in a scratch directory. `optimizer` is the complete
 * OPTIMIZER ... ENDOPTIMIZER block; the objective and the three-dimensional
 * search space are shared by every case. */
struct TaRun {
    /* Rebuild the per-thread trainer on every evaluation, as the host
     * end-to-end tests do: a trainer cached on this thread by another test
     * in the same process must not stand in for this configuration. */
    struct ForceRebuild {
        bool restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        ForceRebuild()  { if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1); }
        ~ForceRebuild() { if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD"); }
    } force_rebuild;
    ScratchDir sd;
    tadah::core::Context ctx;
    fs::path hpo, val;

    explicit TaRun(const std::string& optimizer)
    {
        tadah::core::PeriodicTable::initialize();
        /* HPOLog and the HPO_Host statics are process-wide; a run in the
         * same process as another test must start as a new process would. */
        tadah::hpo::HPOLog::instance().reset_for_test();
        Host::reset_globals_for_test();
        const fs::path train = sd.dir / "train.db";
        const fs::path valdb = sd.dir / "val.db";
        val = sd.dir / "config.val";
        hpo = sd.dir / "config.hpo";
        write_ta_db(train, 0.0, -0.001);
        write_ta_db(valdb, 0.07, -0.04);
        std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
        std::ofstream(hpo)
            << optimizer
            << "LOSS L2\n"
               "FAILSCORE 1e9\n"
               "QFLUSH 1000\n"
               "DIGITS 15\n"
               "PC_ERMSE 0 1\n"
               "OPTIM CGRID2B (1) 0.1 0.9\n"
               "OPTIM CGRID2B (2) 1.0 2.0\n"
               "OPTIM SGRID2B (2) 4.0 6.0\n";
        ctx = make_ta_context(train, hpo, val);
    }
};

std::string plain_block(const std::string& strategy_body)
{
    return "OPTIMIZER\n  STRATEGY\n" + strategy_body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

std::string read_file(const fs::path& path)
{
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/* Whitespace-separated tokens of every data row (not '#', not blank). */
std::vector<std::vector<std::string>> data_rows(const fs::path& path)
{
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find_first_not_of(" \t");
        if (p == std::string::npos || line[p] == '#') continue;
        std::istringstream is(line);
        std::vector<std::string> tok;
        for (std::string t; is >> t;) tok.push_back(t);
        rows.push_back(std::move(tok));
    }
    return rows;
}

/* The "Optimiser status" value of the journal's RESULT section. */
std::string journal_status(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(Optimiser status\s*: ([^\n]*))")))
        return m[1];
    return "(no RESULT section)";
}

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
}

/* The evaluations at which a stagnation stop fired, in journal order (one
 * line per stagnation event), with the journal offset of each line. */
std::vector<std::pair<std::size_t, std::size_t>>
stagnation_stops(const std::string& journal)
{
    std::vector<std::pair<std::size_t, std::size_t>> stops;
    const std::regex stop_re(R"(StagnationStop fired at eval (\d+))");
    for (auto it = std::sregex_iterator(journal.begin(), journal.end(),
                                        stop_re);
         it != std::sregex_iterator(); ++it)
        stops.emplace_back(std::stoull((*it)[1]),
                           static_cast<std::size_t>(it->position(0)));
    return stops;
}

/* The invariants every ending must leave behind, read from the files in
 * the current directory. */
void check_run_files()
{
    const auto loss = data_rows("loss.tadah");
    REQUIRE_FALSE(loss.empty());

    /* One row per evaluation the run made (none is missing, none is
     * extra), numbered 1, 2, 3, ... */
    CHECK(loss.size() == Host::g_iter_value());
    for (std::size_t i = 0; i < loss.size(); ++i) {
        INFO("loss.tadah row " << i);
        CHECK(std::stoull(loss[i].at(0)) == i + 1);
    }

    /* The incumbent: the lowest feasible loss. loss.tadah prints 16
     * significant digits (DIGITS 15), one short of a double's exact round
     * trip, so two losses closer than that cannot be ordered from the file
     * (best_pot.tadah's key compares full doubles): every row within the
     * print resolution of the lowest printed loss is a candidate, and
     * best_pot.tadah must hold one of them. With a single candidate this
     * is the exact incumbent. */
    struct Row { std::size_t iter, slot; double loss; };
    std::vector<Row> feasible;
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto& row : loss) {
        const double l = std::stod(row.back());
        if (!(std::isfinite(l) && l < kFailScore)) continue;
        feasible.push_back({std::stoull(row.at(0)), std::stoull(row.at(1)), l});
        lowest = std::min(lowest, l);
    }
    REQUIRE_FALSE(feasible.empty());
    constexpr double kPrintResolution = 2e-15;   // relative, 16 digits
    std::vector<Row> candidates;
    for (const auto& r : feasible)
        if (std::fabs(r.loss - lowest) <= kPrintResolution * std::fabs(lowest))
            candidates.push_back(r);

    const auto key = tadah::hpo::durable_best::read_key_file("best_pot.tadah");
    REQUIRE(key);
    const Row* winner = nullptr;
    for (const auto& r : candidates)
        if (r.iter == key->eval) winner = &r;
    INFO("best_pot.tadah holds eval " << key->eval << ", loss " << key->loss
         << "; lowest loss.tadah value " << lowest << " ("
         << candidates.size() << " candidate row(s))");
    REQUIRE(winner != nullptr);
    CHECK(key->slot + 1 == winner->slot);   // the files print 1-based slots
    CHECK(key->loss == Approx(winner->loss).epsilon(kPrintResolution));

    /* best_params.tadah has one row per improvement (BOUTPUT 1), with the
     * improvement ordinal k = 1, 2, ... in its first column, and the row of
     * the last improvement is the incumbent's params.tadah row (same slot
     * and values). Rows are written in evaluation order, so with concurrent
     * evaluations a later improvement can precede an earlier one: the last
     * improvement is the highest k, not necessarily the last row. */
    const auto params  = data_rows("params.tadah");
    const auto bparams = data_rows("best_params.tadah");
    REQUIRE_FALSE(bparams.empty());
    std::vector<std::size_t> ordinals;
    const std::vector<std::string>* last_best = nullptr;
    for (const auto& row : bparams) {
        ordinals.push_back(std::stoull(row.at(0)));
        if (!last_best || ordinals.back() > std::stoull(last_best->at(0)))
            last_best = &row;
    }
    std::sort(ordinals.begin(), ordinals.end());
    for (std::size_t i = 0; i < ordinals.size(); ++i) {
        INFO("best_params.tadah ordinals " << ::Catch::Detail::stringify(ordinals));
        CHECK(ordinals[i] == i + 1);
    }
    const std::vector<std::string>* incumbent_row = nullptr;
    for (const auto& row : params)
        if (std::stoull(row.at(0)) == winner->iter) incumbent_row = &row;
    REQUIRE(incumbent_row != nullptr);
    REQUIRE(last_best->size() == incumbent_row->size());
    for (std::size_t c = 1; c < last_best->size(); ++c) {
        INFO("best_params.tadah column " << c);
        CHECK((*last_best)[c] == (*incumbent_row)[c]);
    }

    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("dropping iter") == std::string::npos);
    CHECK(journal.find("stranded") == std::string::npos);
    CHECK_FALSE(fs::exists("checkpoint.tadah"));
}

/* ---- real signals --------------------------------------------------- */

struct SignalOutcome {
    int                status = -1;
    std::exception_ptr error;
    bool               raised = false;
    std::size_t        evals_at_raise = 0;
    std::size_t        evals_final = 0;
};

/* Runs hpo_run(ctx) on a worker thread and raises `signo` on this thread
 * once three evaluations have started (so SignalHandler::install() has
 * run). None of the configurations can end by itself within the time this
 * takes: each runs for many seconds, or thousands of evaluations, without
 * the signal, so the signal is what ends the run. */
SignalOutcome run_and_signal(tadah::core::Context& ctx, int signo)
{
    SignalOutcome out;
    std::atomic<bool> finished{false};
    std::thread worker([&] {
        try {
            out.status = tadah::hpo::hpo_run(ctx);
        } catch (...) {
            out.error = std::current_exception();
        }
        finished.store(true);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::minutes(4);
    while (!finished.load() && Host::g_iter_value() < 3 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    if (!finished.load() && Host::g_iter_value() >= 3) {
        /* The handler runs on this thread before raise() returns, so every
         * evaluation that starts after the count below is taken sees the
         * stop flag. */
        std::raise(signo);
        out.evals_at_raise = Host::g_iter_value();
        out.raised = true;
    }
    worker.join();
    out.evals_final = Host::g_iter_value();
    return out;
}

std::string describe(const std::exception_ptr& e)
{
    if (!e) return "none";
    try {
        std::rethrow_exception(e);
    } catch (const std::exception& x) {
        return std::string("std::exception: ") + x.what();
    } catch (...) {
        return "non-standard exception";
    }
}

/* `in_flight`: evaluations that may legitimately start after the signal --
 * those that had passed the stop-flag check (at the start of an
 * evaluation) but not yet taken their number when the count was sampled,
 * i.e. at most one per concurrent evaluation thread. */
void check_signal_stop(const SignalOutcome& o, int signo,
                       const std::string& name, int expected_status,
                       std::size_t in_flight)
{
    REQUIRE(o.raised);
    INFO("hpo_run exception: " << describe(o.error));
    CHECK_FALSE(o.error);
    CHECK(o.status == expected_status);
    CHECK(o.status == 128 + signo);
    CHECK(tadah::core::SignalHandler::signal_seen() == signo);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == "SHUTDOWN (" + name + ")");
    INFO("evaluations when the signal was raised: " << o.evals_at_raise);
    CHECK(o.evals_final <= o.evals_at_raise + in_flight);
    check_run_files();
}

/* `more` runs extra checks while the run's files still exist. */
void signal_case(const std::string& optimizer, int signo,
                 const std::string& name, int expected_status,
                 std::size_t in_flight = 1,
                 const std::function<void()>& more = {})
{
    TaRun run(optimizer);
    const SignalOutcome o = run_and_signal(run.ctx, signo);
    check_signal_stop(o, signo, name, expected_status, in_flight);
    if (more) more();
}

/* DLIB BFGS that never stops by itself here: its gradient-norm test cannot
 * be met, so it runs until MAXEVAL (about 7000 evaluations per second on
 * this fixture). */
const std::string kBfgsLong =
    "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 100000000\n"
    "    PARAM tol_gradinf 1e-300\n";

/* A real HPO_Host on the fixture, with an open journal, outside hpo_run():
 * for the parts of a run tested on their own. */
struct HostFixture {
    TaRun run;
    struct Journal {
        explicit Journal(const fs::path& dir)
        {
            tadah::hpo::HPOLog::instance().open(
                (dir / "hpo_run.log").string(), "test", dir.string(), 1);
        }
        /* No-op when the case already wrote the trailer. */
        ~Journal() { tadah::hpo::HPOLog::instance().finish("test-end"); }
    } journal;
    tadah::hpo::HPOTargets trg;
    Host host;

    HostFixture()
        : run(plain_block(kBfgsLong)),
          journal(run.sd.dir),
          trg(run.ctx, run.hpo.string()),
          host(run.ctx, trg, run.val.string())
    {
        host.prepare_best_publication();
    }

    /* k distinct points inside the OPTIM boxes, as an optimiser would
     * propose them (CGRID2B[1], CGRID2B[2], SGRID2B[2]). */
    void evaluate(std::size_t k) const
    {
        for (std::size_t i = 0; i < k; ++i) {
            const double t = static_cast<double>(i + 1) /
                             static_cast<double>(k + 1);
            ParamVec p(3);
            p(0) = 0.1 + 0.8 * t;
            p(1) = 2.0 - 1.0 * t;
            p(2) = 4.0 + 2.0 * t * t;
            (void)host(p);
        }
    }

    tadah::hpo::HPOLog& log() const { return tadah::hpo::HPOLog::instance(); }
};

}  // namespace

/* ======================================================================
 * Stop criteria: an ordinary end leaves the same invariants.
 * ==================================================================== */

TEST_CASE("Termination: DLIB BFGS MAXEVAL ends with best_pot.tadah = the "
          "best loss.tadah row", "[HPO][termination][stop_criteria]")
{
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 9\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(data_rows("loss.tadah").size() <= 9u);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("DLIB:BFGS -- MAXEVAL=9 reached") != std::string::npos);
}

TEST_CASE("Termination: NLopt MAXEVAL ends with best_pot.tadah = the best "
          "loss.tadah row", "[HPO][termination][stop_criteria]")
{
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 12\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(data_rows("loss.tadah").size() == 12u);
    CHECK(journal_status(read_file("hpo_run.log")) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
}

TEST_CASE("Termination: ANNEAL MAXEVAL ends with best_pot.tadah = the best "
          "loss.tadah row", "[HPO][termination][stop_criteria]")
{
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL\n    MAXEVAL 30\n"
        "    SEED 5\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(data_rows("loss.tadah").size() == 30u);
    CHECK(read_file("hpo_run.log").find("ANNEAL stopped: MAXEVAL")
          != std::string::npos);
}

TEST_CASE("Termination: NLopt STOPVAL ends with best_pot.tadah = the best "
          "loss.tadah row", "[HPO][termination][stop_criteria]")
{
    /* Every feasible loss of this objective is below 1e8, so STOPVAL is
     * reached by the first evaluation. */
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 50\n"
        "    STOPVAL 1e8\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(data_rows("loss.tadah").size() == 1u);
    CHECK(journal_status(read_file("hpo_run.log")) ==
          "STOPVAL_REACHED -- objective below stopval threshold");
}

TEST_CASE("Termination: under PRESTAGE the trailer is the refinement's, "
          "not the exploration's", "[HPO][termination][stop_criteria]")
{
    /* The NLopt exploration records MAXEVAL_REACHED; the DLIB refinement,
     * which ends the run, records no reason of its own. The trailer used to
     * repeat the exploration's status. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
        "    MAXEVAL 20\n    TOPK 1\n"
        "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
        "      MAXEVAL 10\n    ENDEXPLORE\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("REFINEMENT: refining top-1") != std::string::npos);
    CHECK(journal_status(journal) == "COMPLETED");
    check_run_files();
}

/* ======================================================================
 * RESTART fires through every strategy and backend.
 * ==================================================================== */

TEST_CASE("Termination: RESTART fires under NLopt",
          "[HPO][termination][restart]")
{
    /* NLopt used to turn the stagnation stop into an ordinary
     * "nlopt failure" end, so RESTART never fired with an NLopt backend. */
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 200\n"
        "    RESTART\n      AFTER_STAGNATION 3\n      MAX 2\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 1/2") != std::string::npos);
    CHECK(journal.find("nlopt failure") == std::string::npos);
    check_run_files();
}

TEST_CASE("Termination: RESTART fires when the stop is raised inside an "
          "NLopt MLSL INNER local search", "[HPO][termination][restart]")
{
    /* MLSL runs the INNER optimiser on the same objective; with POPULATION
     * 2 the stagnation window closes during the first SBPLX local search
     * (MLSL's own samples are the first three evaluations). The forced stop
     * reaches the local run through NLopt's force_stop_child chain. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL_LDS\n"
        "    MAXEVAL 300\n    POPULATION 2\n"
        "    INNER\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
        "      MAXEVAL 60\n    ENDINNER\n"
        "    RESTART\n      AFTER_STAGNATION 4\n      MAX 2\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 1/2") != std::string::npos);
    CHECK(journal.find("RESTART 2/2") != std::string::npos);
    CHECK(journal.find("max restarts (2) used") != std::string::npos);
    CHECK(journal.find("nlopt failure") == std::string::npos);
    /* The RESTART budget ended the run, not NLopt. */
    CHECK(journal_status(journal) ==
          "STOPPED -- stagnation, max restarts (2) used");
    check_run_files();
}

TEST_CASE("Termination: RESTART re-initialises the whole MULTISTART",
          "[HPO][termination][restart]")
{
    /* MULTISTART used to catch the stagnation stop as a failed start and
     * move on, so RESTART never fired; the later starts then stopped after
     * a single evaluation each (the stagnation window was never reset). */
    TaRun run(
        "OPTIMIZER\n"
        "  INIT\n    METHOD LHS 3\n    SEED 11\n  ENDINIT\n"
        "  STRATEGY\n"
        "    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
        "    N_STARTS 3\n    BUDGET_PER_START 200\n"
        "    RESTART\n      AFTER_STAGNATION 3\n      MAX 1\n"
        "      METHOD RANDOM 3\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 1/1") != std::string::npos);
    CHECK(journal.find("failed: HPO stagnation threshold reached")
          == std::string::npos);
    /* The restart begins the multistart again, from its first start. */
    CHECK(count_of(journal, "MULTISTART start 1/3 -- running inner optimiser")
          == 2u);
    check_run_files();
}

TEST_CASE("Termination: under PRESTAGE, RESTART re-runs exploration and "
          "refinement, and the refinement gets its own stagnation window",
          "[HPO][termination][restart]")
{
    /* A stagnating exploration hands its top-K to the refinement early. The
     * refinement used to inherit the exploration's stagnation count, so its
     * first evaluation already crossed the threshold; and MULTISTART (the
     * TOPK > 1 refinement) then swallowed the stop start by start, so
     * RESTART never fired. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
        "    MAXEVAL 400\n    TOPK 2\n"
        "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
        "      MAXEVAL 200\n      SEED 5\n    ENDEXPLORE\n"
        "    RESTART\n      AFTER_STAGNATION 4\n      MAX 1\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 1/1") != std::string::npos);
    CHECK(journal.find("max restarts (1) used") != std::string::npos);
    CHECK(journal.find("failed: HPO stagnation threshold reached")
          == std::string::npos);
    CHECK(journal_status(journal) ==
          "STOPPED -- stagnation, max restarts (1) used");

    /* One stagnation stop per phase and round -- exploration, refinement,
     * exploration, refinement -- each a full window after the previous. */
    const auto stops = stagnation_stops(journal);
    REQUIRE(stops.size() == 4u);
    for (std::size_t i = 1; i < stops.size(); ++i) {
        INFO("stagnation stop " << i << " at eval " << stops[i].first
             << ", previous at " << stops[i - 1].first);
        CHECK(stops[i].first >= stops[i - 1].first + 4);
    }
    CHECK(count_of(journal, "REFINEMENT: refining top-2") == 2u);
    check_run_files();
}

TEST_CASE("Termination: a restarted PRESTAGE explores from the fresh points",
          "[HPO][termination][restart]")
{
    /* The exploration has no INIT block, so it starts from the
     * OPTIMIZER-level INIT points (here the configuration's own point).
     * RESTART used to replace only the refinement's copy of those, so every
     * restart replayed the exploration from the original point and the
     * RESTART points were never used. LN_SBPLX evaluates its start point
     * first. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
        "    MAXEVAL 60\n    TOPK 1\n"
        "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
        "      MAXEVAL 40\n    ENDEXPLORE\n"
        "    RESTART\n      AFTER_STAGNATION 4\n      MAX 1\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    REQUIRE(read_file("hpo_run.log").find("RESTART 1/1") != std::string::npos);

    /* SBPLX is deterministic: from the same start point it proposes the
     * same first points, so an exploration restarted from the original
     * point shows up as a second run of evaluations 1, 2, 3. */
    const auto params = data_rows("params.tadah");
    auto x_of = [&](std::size_t row) {
        return std::vector<std::string>(params.at(row).begin() + 2,
                                        params.at(row).end());
    };
    constexpr std::size_t kFirst = 3;
    REQUIRE(params.size() > 2 * kFirst);
    REQUIRE(x_of(0) == std::vector<std::string>{"5.000000000000000e-01",
                                                "1.400000000000000e+00",
                                                "5.000000000000000e+00"});
    std::vector<std::size_t> replayed_at;
    for (std::size_t r = kFirst; r + kFirst <= params.size(); ++r) {
        bool same = true;
        for (std::size_t k = 0; k < kFirst && same; ++k)
            same = x_of(r + k) == x_of(k);
        if (same) replayed_at.push_back(r + 1);
    }
    INFO("evaluations 1-3 repeated from evaluation(s) "
         << ::Catch::Detail::stringify(replayed_at));
    CHECK(replayed_at.empty());
    check_run_files();
}

TEST_CASE("Termination: RESTART under ASA block parallelism ends with the "
          "RESTART budget as its trailer", "[HPO][termination][restart]")
{
    /* Every evaluation of a block that ends after the window closed throws
     * the stagnation stop. When the MAX restarts were used up, the trailer
     * used to be the stopped ASA run's "ASA: exception raised during
     * evaluation". */
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n    MAXEVAL 600\n"
        "    THREADS 4\n    SEED 7\n"
        "    PARAM Gener_Block 4\n    PARAM Gener_Block_Max 4\n"
        "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 2/2") != std::string::npos);
    CHECK(journal.find("max restarts (2) used") != std::string::npos);
    CHECK(journal_status(journal) ==
          "STOPPED -- stagnation, max restarts (2) used");
    check_run_files();
}

TEST_CASE("Termination: RESTART fires under DLIB GFS with four threads",
          "[HPO][termination][restart]")
{
    /* Several of GFS's concurrent evaluations throw the stagnation stop.
     * dlib's thread pool used to rethrow the first one while the others
     * were still running and abort the process on the second. */
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n    MAXEVAL 400\n"
        "    THREADS 4\n"
        "    RESTART\n      AFTER_STAGNATION 5\n      MAX 3\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 3/3") != std::string::npos);
    CHECK(journal.find("max restarts (3) used") != std::string::npos);
    CHECK(journal_status(journal) ==
          "STOPPED -- stagnation, max restarts (3) used");
    check_run_files();
}

TEST_CASE("Termination: RESTART fires when a four-thread DLIB GFS "
          "exploration of a PRESTAGE stagnates", "[HPO][termination][restart]")
{
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
        "    MAXEVAL 100\n    TOPK 1\n"
        "    EXPLORE\n      LIB DLIB\n      ALGO GFS\n"
        "      MAXEVAL 200\n      THREADS 4\n    ENDEXPLORE\n"
        "    RESTART\n      AFTER_STAGNATION 5\n      MAX 2\n"
        "      METHOD RANDOM 1\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("RESTART 2/2") != std::string::npos);
    CHECK(journal.find("max restarts (2) used") != std::string::npos);
    CHECK(count_of(journal, "REFINEMENT: refining top-1") == 3u);
    check_run_files();
}

namespace {

/* An inner policy that ends the way an optimiser run does when a signal,
 * an error or the stagnation watchdog stops it. */
template<class E>
struct ThrowingPolicy : tadah::hpo::RunPolicy<ParamVec, ::LammpsRunner> {
    E           error;
    std::size_t runs = 0;
    std::size_t armed_threshold = 0;
    explicit ThrowingPolicy(E e) : error(std::move(e)) {}
    void run(const tadah::hpo::OptimizerSpec&, Host&) override
    {
        ++runs;
        armed_threshold = Host::get_stagnation_threshold();
        throw error;
    }
};

}  // namespace

TEST_CASE("Termination: RESTART disarms the stagnation watchdog however "
          "its run ends", "[HPO][termination][restart]")
{
    /* The watchdog used to be disarmed only after RestartPolicy's loop, so
     * a run ended by an exception left it armed: a later run in the same
     * process then stopped at a window it never configured. Nothing resets
     * the host's globals between a run and its check here. */
    HostFixture fx;
    tadah::hpo::OptimizerSpec spec;
    constexpr std::size_t kWindow = 7;

    auto ends_with = [&](auto error, const char* what) {
        using E = decltype(error);
        INFO("the inner run ends with " << what);
        auto inner = std::make_unique<ThrowingPolicy<E>>(error);
        auto* probe = inner.get();
        tadah::hpo::RestartPolicy<ParamVec, ::LammpsRunner> restart(
            std::move(inner), kWindow, 2, nullptr, nullptr);
        CHECK_THROWS_AS(restart.run(spec, fx.host), E);
        CHECK(probe->runs == 1u);
        CHECK(probe->armed_threshold == kWindow);
        CHECK(Host::get_stagnation_threshold() == 0u);
    };
    ends_with(tadah::core::ShutdownRequested{}, "a signal stop");
    ends_with(tadah::hpo::durable_best::Error("cannot save best_pot.tadah"),
              "a failure to save best_pot.tadah");
    ends_with(std::runtime_error("optimiser broke"), "an error");

    /* ... and when the MAX restarts are used up. */
    auto inner = std::make_unique<ThrowingPolicy<Host::StagnationStop>>(
        Host::StagnationStop{});
    auto* probe = inner.get();
    tadah::hpo::RestartPolicy<ParamVec, ::LammpsRunner> restart(
        std::move(inner), kWindow, 2, nullptr, nullptr);
    CHECK_NOTHROW(restart.run(spec, fx.host));
    CHECK(probe->runs == 3u);   // the first run and two restarts
    CHECK(Host::get_stagnation_threshold() == 0u);
}

/* ======================================================================
 * RESTART round r (1..MAX) draws new random numbers: every phase with a
 * SEED runs with a seed derived from it -- the outer spec with SEED + r*S,
 * where S is the number of starts the strategy seeds from SEED (N_STARTS
 * under MULTISTART, TOPK under PRESTAGE, 1 under PLAIN), an EXPLORE spec
 * with SEED + r -- while round 0 keeps SEED. (An INNER block takes no
 * SEED; see below.) A restart used to keep every SEED, so a seeded
 * stochastic optimiser replayed round 0's random sequence from its new
 * start point.
 *
 * NLopt draws from the SEED of the phase it runs, and from the phase's INIT
 * SEED only when the phase has no SEED: an INIT SEED used to take
 * precedence, so neither the MULTISTART start seeds nor the derived restart
 * seeds reached NLopt.
 * ==================================================================== */

namespace {

using Point = std::vector<std::string>;

/* The evaluated points of params.tadah, by evaluation number (1-based):
 * the columns after the evaluation number and the slot. */
std::vector<Point> evaluated_points()
{
    const auto params = data_rows("params.tadah");
    std::vector<Point> pts(params.size());
    for (const auto& row : params) {
        const std::size_t eval = std::stoull(row.at(0));
        REQUIRE(eval >= 1);
        REQUIRE(eval <= pts.size());
        pts[eval - 1] = Point(row.begin() + 2, row.end());
    }
    return pts;
}

/* Evaluations [first, last] (1-based, inclusive) of `pts`. */
std::vector<Point> evals_between(const std::vector<Point>& pts,
                                 std::size_t first, std::size_t last)
{
    REQUIRE(first >= 1);
    REQUIRE(last <= pts.size());
    return std::vector<Point>(pts.begin() + (first - 1), pts.begin() + last);
}

/* The points after the first one (the start point). */
std::vector<Point> drawn(const std::vector<Point>& pts)
{
    REQUIRE_FALSE(pts.empty());
    return std::vector<Point>(pts.begin() + 1, pts.end());
}

/* The points `optimizer` evaluates in a run of its own, without RESTART,
 * from the configuration's point. */
std::vector<Point> reference_points(const std::string& optimizer)
{
    TaRun ref(optimizer);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(ref.ctx));
    REQUIRE(status == 0);
    return evaluated_points();
}

/* An inner policy that records the seeds of every phase each round runs
 * with, and ends the round with a stagnation stop. */
struct SeedRecorder : tadah::hpo::RunPolicy<ParamVec, ::LammpsRunner> {
    struct Seeds {
        std::optional<std::uint64_t> outer, inner, explore, explore_inner;
    };
    std::vector<Seeds> rounds;
    void run(const tadah::hpo::OptimizerSpec& spec, Host&) override
    {
        Seeds s;
        s.outer = spec.seed;
        if (spec.inner) s.inner = spec.inner->seed;
        if (spec.prestage) {
            s.explore = spec.prestage->seed;
            if (spec.prestage->inner)
                s.explore_inner = spec.prestage->inner->seed;
        }
        rounds.push_back(s);
        throw Host::StagnationStop{};
    }
};

/* Every "MULTISTART start i/k (seed=X)" line of the journal -- the starts
 * of a MULTISTART, or of a PRESTAGE refinement over TOPK > 1 candidates --
 * has X = base + stride*r + (i-1), r being the RESTART round the line
 * belongs to, and no two starts share a seed. Each of the `n_rounds`
 * rounds runs at least `min_starts` starts. */
void check_start_seeds(const std::string& journal, std::uint64_t base,
                       std::uint64_t stride, std::size_t n_rounds,
                       std::size_t min_starts = 1)
{
    const std::regex restart_re(R"(RESTART (\d+)/\d+ -- fresh init)");
    const std::regex start_re(
        R"(MULTISTART start (\d+)/(\d+) -- running inner optimiser \(seed=(\d+)\))");
    std::uint64_t round = 0;
    std::set<std::uint64_t> seeds;
    std::vector<std::size_t> starts_per_round(n_rounds, 0);
    std::istringstream lines(journal);
    for (std::string line; std::getline(lines, line);) {
        std::smatch m;
        if (std::regex_search(line, m, restart_re)) {
            round = std::stoull(m[1]);
            REQUIRE(round < starts_per_round.size());
        } else if (std::regex_search(line, m, start_re)) {
            const std::uint64_t i = std::stoull(m[1]) - 1;
            const std::uint64_t seed = std::stoull(m[3]);
            INFO("round " << round << ", start " << (i + 1) << ": " << line);
            CHECK(seed == base + stride * round + i);
            CHECK(seeds.insert(seed).second);
            ++starts_per_round[round];
        }
    }
    INFO("starts per round " << ::Catch::Detail::stringify(starts_per_round));
    for (const std::size_t n : starts_per_round) CHECK(n >= min_starts);
}

}  // namespace

TEST_CASE("Termination: each RESTART round derives the seed of every "
          "seeded phase from its SEED", "[HPO][termination][restart][seed]")
{
    HostFixture fx;
    using tadah::hpo::OptimizerSpec;

    /* Runs `spec` under RESTART MAX 2 with an inner run that always
     * stagnates: round 0 and two restarts. */
    auto rounds_of = [&](const OptimizerSpec& spec) {
        auto inner = std::make_unique<SeedRecorder>();
        auto* recorder = inner.get();
        tadah::hpo::RestartPolicy<ParamVec, ::LammpsRunner> restart(
            std::move(inner), 5, 2, nullptr, nullptr);
        REQUIRE_NOTHROW(restart.run(spec, fx.host));
        REQUIRE(recorder->rounds.size() == 3u);
        return recorder->rounds;
    };

    SECTION("a MULTISTART with an INNER and a PRESTAGE-style EXPLORE with "
            "an INNER: no round seeds an INNER")
    {
        /* An INNER block takes no SEED -- NLopt draws the INNER search
         * from the enclosing run's random stream, and make_nlopt refuses
         * an INNER spec with a seed -- so a restart must not give it one
         * either: in every round both INNER specs stay unseeded while the
         * outer and EXPLORE specs run with their derived seeds. This is
         * the RESTART half of the rule; it held before INNER SEED was
         * rejected too, since no INNER here has a SEED to derive from. The
         * parser and make_nlopt cases below pin the rejection itself. */
        OptimizerSpec spec;
        spec.strategy_type = tadah::hpo::StrategyType::MULTISTART;
        spec.multistart.n_starts = 3;
        spec.seed = 10;
        spec.inner = std::make_unique<OptimizerSpec>();
        spec.prestage = std::make_unique<OptimizerSpec>();
        spec.prestage->seed = 30;
        spec.prestage->inner = std::make_unique<OptimizerSpec>();

        const auto rounds = rounds_of(spec);
        for (std::uint64_t r = 0; r < rounds.size(); ++r) {
            INFO("round " << r);
            REQUIRE(rounds[r].outer);
            REQUIRE(rounds[r].explore);
            CHECK(*rounds[r].outer == 10 + 3 * r);
            CHECK_FALSE(rounds[r].inner);
            CHECK(*rounds[r].explore == 30 + r);
            CHECK_FALSE(rounds[r].explore_inner);
        }

        /* MULTISTART gives start i of a round that round's SEED + i
         * (MultiStartPolicy), so no two starts of the run share a seed. */
        std::set<std::uint64_t> start_seeds;
        for (const auto& round : rounds)
            for (std::uint64_t i = 0; i < 3; ++i) {
                INFO("start seed " << *round.outer + i);
                CHECK(start_seeds.insert(*round.outer + i).second);
            }

        /* The run's own spec keeps its seeds. */
        CHECK(spec.seed == std::optional<std::uint64_t>(10));
        CHECK(spec.prestage->seed == std::optional<std::uint64_t>(30));
    }

    SECTION("a PRESTAGE steps by TOPK: its refinement starts never share "
            "a seed")
    {
        /* The owner's worked example: SEED 7, TOPK 3, MAX 2. The
         * refinement multistarts the TOPK candidates, candidate i with the
         * round's SEED + i (StagedPolicy -> MultiStartPolicy). Only
         * MULTISTART runs N_STARTS starts (the parser rejects N_STARTS under
         * TYPE PRESTAGE), so a hand-built spec's N_STARTS does not set the
         * PRESTAGE stride. */
        OptimizerSpec spec;
        spec.strategy_type = tadah::hpo::StrategyType::PRESTAGE;
        spec.stage_topk = 3;
        spec.multistart.n_starts = 5;
        spec.seed = 7;
        spec.prestage = std::make_unique<OptimizerSpec>();
        spec.prestage->seed = 1717;

        const auto rounds = rounds_of(spec);
        std::set<std::uint64_t> start_seeds;
        const std::vector<std::vector<std::uint64_t>> expected = {
            {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
        for (std::uint64_t r = 0; r < rounds.size(); ++r) {
            INFO("round " << r);
            REQUIRE(rounds[r].outer);
            REQUIRE(rounds[r].explore);
            CHECK(*rounds[r].outer == 7 + 3 * r);
            CHECK(*rounds[r].explore == 1717 + r);
            for (std::uint64_t i = 0; i < 3; ++i) {
                INFO("refinement start " << (i + 1) << " seed "
                     << *rounds[r].outer + i);
                CHECK(*rounds[r].outer + i == expected[r][i]);
                CHECK(start_seeds.insert(*rounds[r].outer + i).second);
            }
        }
    }

    SECTION("a PLAIN run steps by one; a phase without SEED stays unseeded")
    {
        /* One run per round, whatever N_STARTS says: only MULTISTART runs
         * N_STARTS starts. */
        OptimizerSpec spec;
        spec.seed = 5;
        spec.multistart.n_starts = 4;
        spec.prestage = std::make_unique<OptimizerSpec>();

        const auto rounds = rounds_of(spec);
        for (std::uint64_t r = 0; r < rounds.size(); ++r) {
            INFO("round " << r);
            REQUIRE(rounds[r].outer);
            CHECK(*rounds[r].outer == 5 + r);
            CHECK_FALSE(rounds[r].inner);
            CHECK_FALSE(rounds[r].explore);
        }
    }
}

TEST_CASE("Termination: a restarted seeded optimiser draws new points",
          "[HPO][termination][restart][seed]")
{
    /* TADAH:RANDOM evaluates its start point, then points drawn from its
     * seed; each round ends at a stagnation stop. */
    const auto random_search = [](std::size_t maxeval, std::uint64_t seed,
                                  const std::string& restart) {
        return plain_block(
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL " + std::to_string(maxeval) + "\n"
            "    SEED " + std::to_string(seed) + "\n" + restart);
    };
    std::vector<std::vector<Point>> rounds;
    {
        TaRun run(random_search(
            5000, 5,
            "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
            "      METHOD RANDOM 1\n    ENDRESTART\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        REQUIRE(journal_status(journal) ==
                "STOPPED -- stagnation, max restarts (2) used");

        /* The seed each round's search reports. */
        std::vector<std::string> seeds;
        const std::regex seed_re(R"(RANDOM search: seed=(\d+))");
        for (auto it = std::sregex_iterator(journal.begin(), journal.end(),
                                            seed_re);
             it != std::sregex_iterator(); ++it)
            seeds.push_back((*it)[1]);
        CHECK(seeds == std::vector<std::string>{"5", "6", "7"});
        /* The RESTART line names the derived seed. */
        CHECK(journal.find("RESTART 1/2 -- fresh init from RANDOM; SEED 6\n")
              != std::string::npos);
        CHECK(journal.find("RESTART 2/2 -- fresh init from RANDOM; SEED 7\n")
              != std::string::npos);

        const auto stops = stagnation_stops(journal);
        REQUIRE(stops.size() == 3u);
        const auto pts = evaluated_points();
        std::size_t first = 1;
        for (const auto& stop : stops) {
            rounds.push_back(evals_between(pts, first, stop.first));
            first = stop.first + 1;
        }
        CHECK(first == pts.size() + 1);
        check_run_files();
    }

    /* Round 0 is exactly the run SEED 5 makes without RESTART. */
    CHECK(rounds[0] == reference_points(random_search(rounds[0].size(), 5, "")));

    /* Round r starts at its fresh RESTART point and then draws the points
     * SEED 5 + r draws, which no earlier round drew. */
    std::set<Point> earlier(rounds[0].begin(), rounds[0].end());
    for (std::uint64_t r = 1; r < rounds.size(); ++r) {
        INFO("round " << r << ": " << rounds[r].size() << " evaluation(s)");
        const auto ref = reference_points(
            random_search(rounds[r].size(), 5 + r, ""));
        REQUIRE(ref.size() == rounds[r].size());
        CHECK(drawn(rounds[r]) == drawn(ref));
        for (const auto& p : drawn(rounds[r])) {
            INFO("point " << ::Catch::Detail::stringify(p));
            CHECK(earlier.count(p) == 0u);
        }
        earlier.insert(rounds[r].begin(), rounds[r].end());
    }
}

namespace {

/* One PLAIN TADAH:RANDOM run with SEED 5 under RESTART MAX 2, whose rounds
 * each end at a stagnation stop. `restart_seed` is the RESTART block's SEED
 * line ("" for none). */
struct RestartRounds {
    std::vector<std::vector<Point>> rounds;
    std::string journal;
};

std::string random_search_block(std::size_t maxeval, std::uint64_t seed,
                                const std::string& restart)
{
    return plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
        "    MAXEVAL " + std::to_string(maxeval) + "\n"
        "    SEED " + std::to_string(seed) + "\n" + restart);
}

RestartRounds run_restart_rounds(const std::string& restart_seed)
{
    RestartRounds out;
    TaRun run(random_search_block(
        5000, 5,
        "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
        "      METHOD RANDOM 1\n" + restart_seed + "    ENDRESTART\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    out.journal = read_file("hpo_run.log");
    REQUIRE(journal_status(out.journal) ==
            "STOPPED -- stagnation, max restarts (2) used");
    const auto stops = stagnation_stops(out.journal);
    REQUIRE(stops.size() == 3u);
    const auto pts = evaluated_points();
    std::size_t first = 1;
    for (const auto& stop : stops) {
        out.rounds.push_back(evals_between(pts, first, stop.first));
        first = stop.first + 1;
    }
    CHECK(first == pts.size() + 1);
    check_run_files();
    return out;
}

}  // namespace

TEST_CASE("Termination: a RESTART SEED draws restart round r's points with "
          "SEED + r", "[HPO][termination][restart][seed]")
{
    /* The RESTART block's SEED used to be ignored: every restart drew its
     * point from std::random_device, so no run with RESTART could be
     * repeated. Round r now draws it with SEED + r, next to the optimiser
     * seeds RESTART derives for the round (SEED 5 + r here). */
    const std::string seed_line = "      SEED 42\n";
    const auto a = run_restart_rounds(seed_line);
    CHECK(a.journal.find("RESTART 1/2 -- fresh init from RANDOM; "
                         "RESTART SEED 43, SEED 6\n") != std::string::npos);
    CHECK(a.journal.find("RESTART 2/2 -- fresh init from RANDOM; "
                         "RESTART SEED 44, SEED 7\n") != std::string::npos);

    /* Round 0 is the run SEED 5 makes without RESTART. */
    CHECK(a.rounds[0] ==
          reference_points(random_search_block(a.rounds[0].size(), 5, "")));

    for (std::uint64_t r = 1; r < a.rounds.size(); ++r) {
        INFO("round " << r << ": " << a.rounds[r].size() << " evaluation(s)");
        /* The round starts at the point an INIT block's METHOD RANDOM 1
         * draws with SEED 42 + r ... */
        const auto start = reference_points(
            "OPTIMIZER\n"
            "  INIT\n    METHOD RANDOM 1\n"
            "    SEED " + std::to_string(42 + r) + "\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 1\n    SEED 5\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        REQUIRE(start.size() == 1u);
        CHECK(a.rounds[r].front() == start.front());
        /* ... and then draws the points of the derived SEED 5 + r. */
        const auto ref = reference_points(
            random_search_block(a.rounds[r].size(), 5 + r, ""));
        REQUIRE(ref.size() == a.rounds[r].size());
        CHECK(drawn(a.rounds[r]) == drawn(ref));
    }

    /* The whole run repeats. */
    const auto b = run_restart_rounds(seed_line);
    CHECK(b.rounds == a.rounds);
}

TEST_CASE("Termination: without a RESTART SEED the restart points come from "
          "std::random_device", "[HPO][termination][restart][seed]")
{
    /* The documented default: without a SEED in the RESTART block each
     * restart draws its point from std::random_device, and the journal's
     * RESTART line lists only the optimiser seeds. Round 0 and the drawn
     * points still follow SEED 5 + r. */
    const auto a = run_restart_rounds("");
    const auto b = run_restart_rounds("");
    for (const auto* run : {&a, &b}) {
        CHECK(run->journal.find("RESTART 1/2 -- fresh init from RANDOM; "
                                "SEED 6\n") != std::string::npos);
        CHECK(run->journal.find("RESTART SEED") == std::string::npos);
    }
    CHECK(a.rounds[0] == b.rounds[0]);
    CHECK(a.rounds[1].front() != b.rounds[1].front());
    CHECK(a.rounds[2].front() != b.rounds[2].front());
}

TEST_CASE("RESTART: a SEED no METHOD of the chain draws from is reported at "
          "startup", "[HPO][restart][seed]")
{
    /* Like an INIT block's SEED: CONFIG and WARM take no random numbers,
     * and without a METHOD line a restart starts from CONFIG. */
    const auto startup_stderr = [](const std::string& restart_body,
                                   const std::string& after_stagnation = "6") {
        TaRun run(random_search_block(
            50, 5,
            "    RESTART\n      AFTER_STAGNATION " + after_stagnation +
                "\n      MAX 2\n" + restart_body + "    ENDRESTART\n"));
        std::ostringstream err;
        struct Redirect {
            std::streambuf* old;
            explicit Redirect(std::streambuf* to) : old(std::cerr.rdbuf(to)) {}
            ~Redirect() { std::cerr.rdbuf(old); }
        } redirect(err.rdbuf());
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        return err.str();
    };
    const std::string unused = "RESTART: SEED is set but the METHOD chain";
    CHECK_THAT(startup_stderr("      METHOD CONFIG\n      SEED 42\n"),
               Catch::Contains(unused + " (CONFIG) contains no stochastic "
                               "method, so the seed does not affect the "
                               "restart points"));
    CHECK_THAT(startup_stderr("      SEED 42\n"),
               Catch::Contains(unused + " (empty -> CONFIG fallback)"));
    CHECK_THAT(startup_stderr("      METHOD CONFIG\n      METHOD LHS 2\n"
                              "      SEED 42\n"),
               !Catch::Contains(unused));
    CHECK_THAT(startup_stderr("      METHOD CONFIG\n"),
               !Catch::Contains(unused));
    /* An inactive RESTART block (AFTER_STAGNATION 0) draws no restart
     * points at all, so its SEED is not reported. */
    CHECK_THAT(startup_stderr("      METHOD CONFIG\n      SEED 42\n", "0"),
               !Catch::Contains(unused));
}

TEST_CASE("Termination: a restarted seeded exploration draws a new "
          "population", "[HPO][termination][restart][seed]")
{
    /* A PRESTAGE whose NLopt GN_CRS2_LM exploration has SEED 1717. CRS2
     * evaluates its start point, then a random population drawn from the
     * seed. The restarted exploration used to draw round 0's population
     * again and largely find the same top-K. AFTER_STAGNATION 25 exceeds
     * the exploration's MAXEVAL 20, so each exploration runs all 20
     * evaluations and only the refinement (a random search) stagnates. */
    constexpr std::size_t kExplore = 20;
    std::vector<Point> round0, round1;
    {
        TaRun run(
            "OPTIMIZER\n"
            "  STRATEGY\n"
            "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 5000\n    TOPK 1\n"
            "    EXPLORE\n      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
            "      MAXEVAL 20\n      SEED 1717\n    ENDEXPLORE\n"
            "    RESTART\n      AFTER_STAGNATION 25\n      MAX 1\n"
            "      METHOD RANDOM 1\n    ENDRESTART\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        REQUIRE(journal.find("RESTART 1/1") != std::string::npos);
        CHECK(journal.find("RESTART 1/1 -- fresh init from RANDOM; "
                           "EXPLORE SEED 1718\n") != std::string::npos);
        CHECK(count_of(journal, "REFINEMENT: refining top-1") == 2u);

        /* One stagnation stop per round, in its refinement: round 1's
         * exploration begins after round 0's. */
        const auto stops = stagnation_stops(journal);
        REQUIRE(stops.size() == 2u);
        const auto pts = evaluated_points();
        round0 = evals_between(pts, 1, kExplore);
        round1 = evals_between(pts, stops[0].first + 1,
                               stops[0].first + kExplore);
        check_run_files();
    }

    const auto crs2 = [](std::uint64_t seed) {
        return plain_block(
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
            "    MAXEVAL 20\n    SEED " + std::to_string(seed) + "\n");
    };
    /* Round 0 explores with SEED 1717, round 1 with 1718. */
    CHECK(drawn(round0) == drawn(reference_points(crs2(1717))));
    CHECK(drawn(round1) == drawn(reference_points(crs2(1718))));
    const std::set<Point> population0(round0.begin(), round0.end());
    for (const auto& p : drawn(round1)) {
        INFO("point " << ::Catch::Detail::stringify(p));
        CHECK(population0.count(p) == 0u);
    }
}

TEST_CASE("Termination: the MULTISTART starts of a restarted run never "
          "share a seed", "[HPO][termination][restart][seed]")
{
    /* MULTISTART runs start i with SEED + i. A restart used to keep SEED,
     * so the starts of every round reused round 0's seeds; round r now
     * runs start i with SEED + r*N_STARTS + i. */
    TaRun run(
        "OPTIMIZER\n"
        "  INIT\n    METHOD LHS 3\n  ENDINIT\n"
        "  STRATEGY\n"
        "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    N_STARTS 3\n    BUDGET_PER_START 200\n    SEED 10\n"
        "    RESTART\n      AFTER_STAGNATION 3\n      MAX 2\n"
        "      METHOD RANDOM 3\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    REQUIRE(journal_status(journal) ==
            "STOPPED -- stagnation, max restarts (2) used");
    CHECK(journal.find("RESTART 1/2 -- fresh init from RANDOM; SEED 13\n")
          != std::string::npos);
    CHECK(journal.find("RESTART 2/2 -- fresh init from RANDOM; SEED 16\n")
          != std::string::npos);

    check_start_seeds(journal, 10, 3, 3);
    check_run_files();
}

TEST_CASE("Termination: the PRESTAGE refinement starts of a restarted run "
          "never share a seed", "[HPO][termination][restart][seed]")
{
    /* The refinement multistarts the TOPK candidates, candidate i with
     * SEED + i. Round r now runs it with SEED + r*TOPK + i; it stepped by
     * N_STARTS (1 here), so round 1's candidates reused seeds 8 and 9 of
     * round 0. The owner's worked example: SEED 7, TOPK 3, MAX 2 gives
     * {7,8,9}, {10,11,12}, {13,14,15}.
     *
     * The run is deterministic: every round starts from the configuration's
     * point (RESTART METHOD CONFIG) and explores with a seeded random
     * search, which stagnates early and hands over its top 3. Each
     * refinement start re-evaluates its candidate once (MAXEVAL 3 split
     * over TOPK 3), so the third start's evaluation is the third without a
     * new best and restarts the PRESTAGE: every round runs all three
     * starts. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    MAXEVAL 3\n    TOPK 3\n    SEED 7\n"
        "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
        "      MAXEVAL 20\n      SEED 1717\n    ENDEXPLORE\n"
        "    RESTART\n      AFTER_STAGNATION 3\n      MAX 2\n"
        "      METHOD CONFIG\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    REQUIRE(journal_status(journal) ==
            "STOPPED -- stagnation, max restarts (2) used");
    CHECK(journal.find("RESTART 1/2 -- fresh init from CONFIG; SEED 10, "
                       "EXPLORE SEED 1718\n") != std::string::npos);
    CHECK(journal.find("RESTART 2/2 -- fresh init from CONFIG; SEED 13, "
                       "EXPLORE SEED 1719\n") != std::string::npos);
    check_start_seeds(journal, 7, 3, 3, 3);
    check_run_files();
}

/* ======================================================================
 * NLopt draws from the SEED of the phase it runs. The phase's INIT-block
 * SEED draws the INIT points, and seeds NLopt only when the phase has no
 * SEED of its own.
 * ==================================================================== */

namespace {

/* A 2-D quadratic that records every point NLopt evaluates. */
double recording_quadratic(unsigned n, const double* x, double*, void* data)
{
    static_cast<std::vector<std::vector<double>>*>(data)->emplace_back(
        x, x + n);
    return (x[0] - 0.3) * (x[0] - 0.3) + (x[1] - 0.6) * (x[1] - 0.6);
}

/* The points `opt` (GN_CRS2_LM) evaluates in [0, 1]^2 from the NLopt
 * random state as it stands: its start point, then a population drawn from
 * NLopt's random number generator. */
std::vector<std::vector<double>> crs2_draws(nlopt::opt opt)
{
    std::vector<std::vector<double>> xs;
    opt.set_lower_bounds(std::vector<double>{0.0, 0.0});
    opt.set_upper_bounds(std::vector<double>{1.0, 1.0});
    opt.set_min_objective(recording_quadratic, &xs);
    std::vector<double> x{0.5, 0.5};
    double f = 0.0;
    opt.optimize(x, f);
    return xs;
}

/* PLAIN NLOPT GN_CRS2_LM with MAXEVAL and SEED, from the configuration's
 * point. CRS2 evaluates its start point, then a population of
 * 10*(dim+1) = 40 points drawn from NLopt's random number generator,
 * whatever the start point. */
constexpr std::size_t kCrs2Population = 40;

std::string crs2_block(std::size_t maxeval, std::uint64_t seed)
{
    return plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
        "    MAXEVAL " + std::to_string(maxeval) + "\n"
        "    SEED " + std::to_string(seed) + "\n");
}

/* The first `k` points of `pts`. */
std::vector<Point> first_points(const std::vector<Point>& pts, std::size_t k)
{
    REQUIRE(k <= pts.size());
    return std::vector<Point>(pts.begin(), pts.begin() + k);
}

}  // namespace

TEST_CASE("NLopt takes its random seed from SEED, and from the INIT "
          "block's SEED only without one", "[HPO][seed][nlopt]")
{
    tadah::hpo::OptimizerSpec base;
    base.lib = "NLOPT";
    base.algo = "GN_CRS2_LM";
    base.stop.maxeval = 12;
    const auto drawn_with = [&](std::optional<std::uint64_t> seed,
                                std::optional<std::uint64_t> init_seed) {
        tadah::hpo::OptimizerSpec sp = base;
        sp.seed = seed;
        sp.init_strategy_spec.seed = init_seed;
        return crs2_draws(tadah::hpo::make_nlopt(sp, 2));
    };
    const auto reference = [](unsigned long seed) {
        nlopt::opt opt("GN_CRS2_LM", 2);
        opt.set_maxeval(12);
        nlopt::srand(seed);
        return crs2_draws(opt);
    };
    const auto seed_1717 = reference(1717);
    const auto seed_99 = reference(99);
    REQUIRE(seed_1717.size() == 12u);
    REQUIRE(seed_1717 != seed_99);

    CHECK(drawn_with(1717, std::nullopt) == seed_1717);
    /* The INIT SEED used to take precedence. */
    CHECK(drawn_with(1717, 99) == seed_1717);
    CHECK(drawn_with(std::nullopt, 99) == seed_99);
}

TEST_CASE("Termination: a restarted NLopt run with an INIT SEED draws from "
          "each round's SEED", "[HPO][termination][restart][seed][nlopt]")
{
    /* The INIT block's SEED 99 draws the start point; SEED 1717 seeds
     * NLopt, so round r draws the population of SEED 1717 + r. The INIT
     * SEED used to seed NLopt in its place, and it is not derived, so
     * every round drew SEED 99's population again. */
    std::vector<std::vector<Point>> rounds;
    {
        TaRun run(
            "OPTIMIZER\n"
            "  INIT\n    METHOD RANDOM 1\n    SEED 99\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
            "    MAXEVAL 5000\n    SEED 1717\n"
            "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
            "      METHOD RANDOM 1\n    ENDRESTART\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        REQUIRE(journal_status(journal) ==
                "STOPPED -- stagnation, max restarts (2) used");
        CHECK(journal.find("RESTART 1/2 -- fresh init from RANDOM; "
                           "SEED 1718\n") != std::string::npos);
        CHECK(journal.find("RESTART 2/2 -- fresh init from RANDOM; "
                           "SEED 1719\n") != std::string::npos);

        const auto stops = stagnation_stops(journal);
        REQUIRE(stops.size() == 3u);
        const auto pts = evaluated_points();
        std::size_t first = 1;
        for (const auto& stop : stops) {
            rounds.push_back(evals_between(pts, first, stop.first));
            first = stop.first + 1;
        }
        CHECK(first == pts.size() + 1);
        check_run_files();
    }

    /* The population part of each round (CRS2's evolution after it
     * depends on the start point). */
    std::set<Point> earlier;
    for (std::uint64_t r = 0; r < rounds.size(); ++r) {
        const std::size_t k = std::min(rounds[r].size(), kCrs2Population);
        INFO("round " << r << ": " << rounds[r].size()
             << " evaluation(s), comparing the first " << k);
        REQUIRE(k >= 2u);
        const auto ours = drawn(first_points(rounds[r], k));
        CHECK(ours == drawn(reference_points(crs2_block(k, 1717 + r))));
        if (r == 0)
            CHECK(ours != drawn(reference_points(crs2_block(k, 99))));
        for (const auto& p : ours) {
            INFO("point " << ::Catch::Detail::stringify(p));
            CHECK(earlier.count(p) == 0u);
        }
        earlier.insert(rounds[r].begin(), rounds[r].end());
    }
}

namespace {

/* A MULTISTART of NLOPT AUGLAG with a GN_CRS2_LM INNER over three LHS
 * start points: `init_seed` is the INIT block's SEED line (or ""),
 * `seed` the STRATEGY block's and `inner_seed` the INNER block's.
 * MULTISTART takes only local NLopt algorithms; AUGLAG with a GN_CRS2_LM
 * INNER draws a CRS2 population from NLopt's random number generator from
 * its second evaluation on (LN_PRAXIS, the local algorithm that draws,
 * does so only once it stalls). */
constexpr std::size_t kAuglagBudget = 25;
static_assert(kAuglagBudget <= kCrs2Population,
              "every evaluation of a start is part of CRS2's population");

std::string auglag_crs2_multistart(const std::string& init_seed,
                                   const std::string& seed,
                                   const std::string& inner_seed)
{
    return "OPTIMIZER\n"
           "  INIT\n    METHOD LHS 3\n" + init_seed + "  ENDINIT\n"
           "  STRATEGY\n"
           "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO AUGLAG\n"
           "    N_STARTS 3\n    BUDGET_PER_START 25\n" + seed +
           "    INNER\n      LIB NLOPT\n      ALGO GN_CRS2_LM\n" + inner_seed +
           "    ENDINNER\n"
           "  ENDSTRATEGY\n"
           "ENDOPTIMIZER\n";
}

/* Checks the finished auglag_crs2_multistart() run in the current
 * directory: start i ran with seed base + i and drew the CRS2 population
 * of SEED base + i, and no two starts share a point. */
void check_auglag_crs2_starts(std::uint64_t base)
{
    const std::string journal = read_file("hpo_run.log");
    check_start_seeds(journal, base, 3, 1);

    /* Each start's evaluations, from the "done" line of each start. */
    const std::regex done_re(
        R"(MULTISTART start (\d+)/3 done: (\d+) eval\(s\) consumed)");
    std::vector<std::size_t> evals;
    for (auto it = std::sregex_iterator(journal.begin(), journal.end(),
                                        done_re);
         it != std::sregex_iterator(); ++it) {
        CHECK(std::stoull((*it)[1]) == evals.size() + 1);
        evals.push_back(std::stoull((*it)[2]));
    }
    REQUIRE(evals.size() == 3u);
    const auto pts = evaluated_points();
    std::size_t first = 1;
    std::set<Point> earlier;
    for (std::uint64_t i = 0; i < evals.size(); ++i) {
        INFO("start " << (i + 1) << ": " << evals[i] << " evaluation(s)");
        /* The CRS2 run takes the start's whole budget; AUGLAG then
         * evaluates the best point CRS2 found once more. */
        REQUIRE(evals[i] >= kAuglagBudget);
        const auto start = evals_between(pts, first, first + evals[i] - 1);
        first += evals[i];
        const auto ours = drawn(first_points(start, kAuglagBudget));
        CHECK(ours ==
              drawn(reference_points(crs2_block(kAuglagBudget, base + i))));
        for (const auto& p : ours) {
            INFO("point " << ::Catch::Detail::stringify(p));
            CHECK(earlier.count(p) == 0u);
        }
        earlier.insert(start.begin(), start.end());
    }
    CHECK(first == pts.size() + 1);
}

}  // namespace

TEST_CASE("Termination: NLopt MULTISTART starts with an INIT SEED each draw "
          "a population of their own", "[HPO][termination][seed][nlopt]")
{
    /* MULTISTART runs start i with SEED + i. The INIT block's SEED 99
     * draws the LHS start points. It used to seed NLopt in place of
     * SEED + i, so every start drew SEED 99's population. */
    TaRun run(auglag_crs2_multistart("    SEED 99\n", "    SEED 10\n", ""));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_auglag_crs2_starts(10);
}

TEST_CASE("NLopt: an unused INIT SEED is reported as NLopt's seed only "
          "when the optimiser has no SEED", "[HPO][seed][nlopt]")
{
    /* An INIT block with a SEED and no RANDOM/LHS/SOBOL method draws no
     * point from it; the startup warning says so, and used to add that the
     * seed "still feeds NLopt's global RNG" even when the optimiser's SEED
     * takes its place. */
    const auto startup_stderr = [](const std::string& seed_line) {
        TaRun run(
            "OPTIMIZER\n"
            "  INIT\n    SEED 99\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
            "    MAXEVAL 20\n" + seed_line +
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        std::ostringstream err;
        struct Redirect {
            std::streambuf* old;
            explicit Redirect(std::streambuf* to) : old(std::cerr.rdbuf(to)) {}
            ~Redirect() { std::cerr.rdbuf(old); }
        } redirect(err.rdbuf());
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        return err.str();
    };
    const std::string unused = "contains no stochastic method";
    const std::string unseeded = startup_stderr("");
    CHECK(unseeded.find(unused) != std::string::npos);
    CHECK(unseeded.find("Without a SEED of its own, the NLopt optimiser "
                        "still takes its random seed from it")
          != std::string::npos);
    const std::string seeded = startup_stderr("    SEED 1717\n");
    CHECK(seeded.find(unused) != std::string::npos);
    CHECK(seeded.find("NLopt") == std::string::npos);
}

TEST_CASE("Termination: an NLopt phase without SEED draws from its INIT "
          "block's SEED", "[HPO][termination][seed][nlopt]")
{
    constexpr std::size_t kEvals = 20;
    std::vector<Point> pts;
    {
        TaRun run(
            "OPTIMIZER\n"
            "  INIT\n    METHOD RANDOM 1\n    SEED 99\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_CRS2_LM\n"
            "    MAXEVAL 20\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        pts = evaluated_points();
        check_run_files();
    }
    REQUIRE(pts.size() == kEvals);
    CHECK(drawn(pts) == drawn(reference_points(crs2_block(kEvals, 99))));
}

/* ======================================================================
 * An INNER block takes no SEED. MLSL and AUGLAG run their INNER search
 * inside the enclosing block's NLopt run, and NLopt has one random stream
 * per run, which the enclosing block's SEED seeds (make_nlopt). An INNER
 * SEED used to seed that stream again, in place of the enclosing block's
 * SEED, so neither that SEED nor the MULTISTART, PRESTAGE and RESTART
 * seeds derived from it reached NLopt. It is now an error at startup that
 * names its line.
 * ==================================================================== */

namespace {

/* The error HPOTargets reports when it loads `optimizer` (with TaRun's
 * objective and search space), or "" when the configuration loads. */
std::string startup_error(const std::string& optimizer)
{
    TaRun run(optimizer);
    try {
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

/* The 1-based number of the one line of `text` that equals `line`. */
std::size_t line_number(const std::string& text, const std::string& line)
{
    std::size_t found = 0, n = 0;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) {
        ++n;
        if (l != line) continue;
        REQUIRE(found == 0u);
        found = n;
    }
    REQUIRE(found != 0u);
    return found;
}

/* An NLopt block with an INNER, in each block the grammar allows an INNER
 * in. config(seed, inner) places the line(s) `seed` in the block that runs
 * NLopt and `inner` in its INNER block; `indent` is that block's
 * indentation, the INNER block's lines have two spaces more. */
struct InnerContext {
    std::string enclosing;
    std::string indent;
    std::function<std::string(const std::string& seed,
                              const std::string& inner)> config;
    /* The parsed spec of the block that runs NLopt. */
    std::function<const tadah::hpo::OptimizerSpec&(
        const tadah::hpo::OptimizerSpec&)> block;
};

std::vector<InnerContext> inner_contexts()
{
    using tadah::hpo::OptimizerSpec;
    return {
        {"STRATEGY", "    ",
         [](const std::string& seed, const std::string& inner) {
             return plain_block(
                 "    TYPE PLAIN\n    LIB NLOPT\n    ALGO G_MLSL\n"
                 "    MAXEVAL 400\n" + seed +
                 "    INNER\n      LIB NLOPT\n      ALGO LN_PRAXIS\n" +
                 inner + "    ENDINNER\n");
         },
         [](const OptimizerSpec& s) -> const OptimizerSpec& { return s; }},
        {"EXPLORE", "      ",
         [](const std::string& seed, const std::string& inner) {
             return plain_block(
                 "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                 "    MAXEVAL 40\n"
                 "    EXPLORE\n      LIB NLOPT\n      ALGO G_MLSL\n"
                 "      MAXEVAL 60\n" + seed +
                 "      INNER\n        LIB NLOPT\n        ALGO LN_PRAXIS\n" +
                 inner + "      ENDINNER\n"
                 "    ENDEXPLORE\n");
         },
         [](const OptimizerSpec& s) -> const OptimizerSpec& {
             REQUIRE(s.prestage);
             return *s.prestage;
         }},
        {"REFINE", "      ",
         [](const std::string& seed, const std::string& inner) {
             return plain_block(
                 "    TYPE PRESTAGE\n    TOPK 3\n"
                 "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
                 "      MAXEVAL 20\n    ENDEXPLORE\n"
                 "    REFINE\n      LIB NLOPT\n      ALGO AUGLAG\n"
                 "      MAXEVAL 60\n" + seed +
                 "      INNER\n        LIB NLOPT\n        ALGO GN_CRS2_LM\n" +
                 inner + "      ENDINNER\n"
                 "    ENDREFINE\n");
         },
         [](const OptimizerSpec& s) -> const OptimizerSpec& { return s; }},
    };
}

std::string inner_seed_error(const std::string& enclosing)
{
    return "SEED is not accepted inside an INNER block: NLopt draws the "
           "INNER search from the enclosing run's random stream; set SEED "
           "on the enclosing " + enclosing + " block";
}

}  // namespace

TEST_CASE("NLopt: SEED inside an INNER block is rejected at startup, naming "
          "its line", "[HPO][seed][nlopt][hpo_targets]")
{
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        const std::string inner_seed = c.indent + "  SEED 9";

        /* Whether or not the enclosing block has a SEED of its own. */
        for (const std::string seed : {"", "SEED 5"}) {
            const std::string cfg = c.config(
                seed.empty() ? "" : c.indent + seed + "\n", inner_seed + "\n");
            INFO(cfg);
            const std::string msg = startup_error(cfg);
            CHECK_THAT(msg, Catch::StartsWith(
                "Error on line " +
                std::to_string(line_number(cfg, inner_seed)) + ": " +
                inner_seed_error(c.enclosing)));
            CHECK_THAT(msg, Catch::EndsWith("\nLine: " + inner_seed));
        }

        /* Moved to the enclosing block, the SEED is the NLopt run's. */
        TaRun run(c.config(c.indent + "SEED 9\n", ""));
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        const auto& block = c.block(trg.spec);
        CHECK(block.seed == std::optional<std::uint64_t>(9));
        REQUIRE(block.inner);
        CHECK_FALSE(block.inner->seed);
    }
}

TEST_CASE("NLopt: the unknown-key error of an INNER block lists every "
          "optimiser attribute but SEED", "[HPO][seed][nlopt][hpo_targets]")
{
    std::string keys;
    for (const auto& k : tadah::hpo::opt_attr_keys())
        if (k != "SEED") keys += (keys.empty() ? "" : "/") + k;
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        const std::string bogus = c.indent + "  BOGUS 1";
        const std::string cfg = c.config("", bogus + "\n");
        CHECK_THAT(startup_error(cfg), Catch::StartsWith(
            "Error on line " + std::to_string(line_number(cfg, bogus)) +
            ": unknown key 'BOGUS' inside INNER block (only " + keys +
            " are valid)"));
    }
}

TEST_CASE("NLopt: an INIT block inside INNER is rejected at startup, so no "
          "INIT SEED reaches an INNER spec", "[HPO][seed][nlopt][hpo_targets]")
{
    /* A pin, not a change: the grammar has never accepted INIT inside
     * INNER. */
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        const std::string in = c.indent + "  ";
        const std::string init = in + "INIT";
        const std::string cfg = c.config(
            "", init + "\n" + in + "  METHOD RANDOM 1\n" + in + "  SEED 9\n" +
                    in + "ENDINIT\n");
        CHECK_THAT(startup_error(cfg), Catch::StartsWith(
            "Error on line " + std::to_string(line_number(cfg, init)) +
            ": INIT block must appear inside OPTIMIZER, STRATEGY, EXPLORE, "
            "or REFINE"));
    }
}

TEST_CASE("NLopt: make_nlopt seeds a run with an INNER once, from the block "
          "that runs it, and refuses an INNER seed", "[HPO][seed][nlopt]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "AUGLAG";
    sp.stop.maxeval = 12;
    sp.seed = 1717;
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "GN_CRS2_LM";

    /* The same AUGLAG run with the stream seeded by hand. */
    const auto reference = [](unsigned long seed) {
        nlopt::opt opt("AUGLAG", 2);
        opt.set_maxeval(12);
        opt.set_local_optimizer(nlopt::opt("GN_CRS2_LM", 2));
        nlopt::srand(seed);
        return crs2_draws(opt);
    };
    const auto seed_1717 = reference(1717);
    REQUIRE(seed_1717.size() >= 3u);
    REQUIRE(seed_1717 != reference(99));
    CHECK(crs2_draws(tadah::hpo::make_nlopt(sp, 2)) == seed_1717);

    /* An INNER spec with a seed can only come from outside the parser. It
     * used to reseed the stream in place of SEED 1717. */
    const auto refused = Catch::Contains(
        "NLopt draws the INNER search from the enclosing run's random "
        "stream");
    SECTION("an INNER SEED") {
        sp.inner->seed = 99;
        CHECK_THROWS_WITH(tadah::hpo::make_nlopt(sp, 2), refused);
    }
    SECTION("an INNER INIT SEED") {
        sp.inner->init_strategy_spec.seed = 99;
        CHECK_THROWS_WITH(tadah::hpo::make_nlopt(sp, 2), refused);
    }
}

TEST_CASE("Termination: NLopt MULTISTART starts draw from the STRATEGY "
          "block's SEED, and an INNER SEED is rejected",
          "[HPO][termination][seed][nlopt]")
{
    /* The owner's example: SEED 10 on the STRATEGY block and SEED 20 on the
     * INNER. The INNER SEED used to seed NLopt in place of SEED + i, so all
     * three starts drew SEED 20's population. */
    const std::string inner_seed = "      SEED 20";
    const std::string rejected =
        auglag_crs2_multistart("", "    SEED 10\n", inner_seed + "\n");
    CHECK_THAT(startup_error(rejected), Catch::StartsWith(
        "Error on line " + std::to_string(line_number(rejected, inner_seed)) +
        ": " + inner_seed_error("STRATEGY")));

    /* With the SEED on the STRATEGY block only, start i draws the CRS2
     * population of SEED 10 + i: 10, 11 and 12, three distinct
     * populations. */
    TaRun run(auglag_crs2_multistart("", "    SEED 10\n", ""));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_auglag_crs2_starts(10);
}

/* ======================================================================
 * A fatal best-potential persistence failure ends the run as an error
 * under every strategy and backend: the incumbent can no longer be kept.
 * ==================================================================== */

namespace {

struct PersistFailure {
    int                status = -1;
    std::exception_ptr error;
};

/* Runs hpo_run(ctx) on a worker thread and makes the run directory
 * read-only once two evaluations have started: the next new best cannot be
 * published (durable_best writes a temporary file there and renames it),
 * while the existing history files can still be appended to. The
 * configurations keep improving for hundreds of evaluations, so a new best
 * follows the switch. */
PersistFailure run_until_unpublishable(tadah::core::Context& ctx,
                                       const fs::path& dir)
{
    PersistFailure out;
    std::atomic<bool> finished{false};
    std::thread worker([&] {
        try {
            out.status = tadah::hpo::hpo_run(ctx);
        } catch (...) {
            out.error = std::current_exception();
        }
        finished.store(true);
    });
    while (!finished.load() && Host::g_iter_value() < 2)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    fs::permissions(dir, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace);
    worker.join();
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace);
    return out;
}

void check_persist_failure(const PersistFailure& o)
{
    INFO("hpo_run returned " << o.status);
    REQUIRE(o.error);
    CHECK_THROWS_AS(std::rethrow_exception(o.error),
                    tadah::hpo::durable_best::Error);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal).rfind("STOPPED with error: ", 0) == 0);
    CHECK(journal.find("fatal best-potential persistence failure at eval")
          != std::string::npos);
}

}  // namespace

TEST_CASE("Termination: MULTISTART ends at a fatal best-potential "
          "persistence failure", "[HPO][termination][durable_best]")
{
    /* MULTISTART used to log the failure as a failed start and run the
     * next one, which failed the same way; the run then ended COMPLETED.
     * Each SBPLX start improves for about 200 evaluations. */
    TaRun run(
        "OPTIMIZER\n"
        "  INIT\n    METHOD LHS 3\n    SEED 11\n  ENDINIT\n"
        "  STRATEGY\n"
        "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    N_STARTS 3\n    BUDGET_PER_START 100000\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    const PersistFailure o = run_until_unpublishable(run.ctx, run.sd.dir);
    check_persist_failure(o);
    CHECK(read_file("hpo_run.log").find("MULTISTART start 2/3")
          == std::string::npos);
}

TEST_CASE("Termination: NLopt ends at a fatal best-potential persistence "
          "failure", "[HPO][termination][durable_best]")
{
    /* NLopt used to report the failure as its own "nlopt failure"
     * self-termination, and the run ended as if converged. */
    TaRun run(plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    MAXEVAL 100000\n"));
    const PersistFailure o = run_until_unpublishable(run.ctx, run.sd.dir);
    check_persist_failure(o);
    CHECK(read_file("hpo_run.log").find("nlopt failure") == std::string::npos);
}

/* ======================================================================
 * The journal status of one run does not leak into the next.
 * ==================================================================== */

TEST_CASE("Termination: a new journal does not inherit the previous run's "
          "status", "[HPO][termination][hpolog]")
{
    ScratchDir sd;
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    log.open((sd.dir / "first.log").string(), "test", sd.dir.string(), 1);
    log.set_status("MAXEVAL_REACHED -- maximum evaluations exhausted");
    log.finish();
    log.open((sd.dir / "second.log").string(), "test", sd.dir.string(), 1);
    log.finish();
    CHECK(journal_status(read_file(sd.dir / "first.log")) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    CHECK(journal_status(read_file(sd.dir / "second.log")) == "COMPLETED");
    log.reset_for_test();
}

TEST_CASE("Termination: a new journal is written even when the previous "
          "one was never finished", "[HPO][termination][hpolog]")
{
    /* A run that fails during its setup leaves its journal open, without a
     * trailer. The next run in the same process used to open its journal
     * on the still-open stream, which failed silently: nothing of that run
     * was ever written. */
    ScratchDir sd;
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    log.open((sd.dir / "failed_setup.log").string(), "test", sd.dir.string(), 1);
    log.info("setup of the first run");
    log.open((sd.dir / "next.log").string(), "test", sd.dir.string(), 1);
    log.info("the next run");
    log.finish();
    const std::string first = read_file(sd.dir / "failed_setup.log");
    const std::string next  = read_file(sd.dir / "next.log");
    CHECK(first.find("setup of the first run") != std::string::npos);
    CHECK(first.find("the next run") == std::string::npos);
    CHECK(journal_status(first) == "(no RESULT section)");
    CHECK(next.find("Run Journal") != std::string::npos);
    CHECK(next.find("the next run") != std::string::npos);
    CHECK(journal_status(next) == "COMPLETED");
    log.reset_for_test();
}

/* ======================================================================
 * The parallel annealing drivers keep the type of a stop raised during the
 * seed evaluation or the T0 calibration (they used to convert it into a
 * std::runtime_error carrying its message).
 * ==================================================================== */

namespace {

struct ThrowOnCall {
    std::size_t k;             // 1-based call that throws
    mutable std::atomic<std::size_t> n{0};
    double operator()(const ParamVec& x) const
    {
        if (++n == k) throw tadah::core::ShutdownRequested{};
        return x(0) * x(0) + 1.0;
    }
};

tadah::hpo::OptimizerSpec anneal_spec(const char* algo)
{
    auto sp = anneal_test::pt_spec({-1.0}, {1.0});
    sp.algo = algo;
    sp.seed = 17;
    sp.num_threads = 1;
    sp.stop.maxeval = 1000;
    if (std::string(algo) == "ANNEAL_PT") sp.param["REPLICAS"] = 2.0;
    else                                   sp.pop = 8;
    return sp;
}

/* The stop is raised on every call from the seed evaluation (1) through
 * the T0 calibration (the next T0_PROBES = 20), ANNEAL_PT's second
 * calibration pass and warm-up, into the production sweeps. */
constexpr std::size_t kSweepCalls = 400;

}  // namespace

TEST_CASE("Termination: ANNEAL_PT keeps ShutdownRequested raised in any "
          "phase, including the seed evaluation and the T0 calibration",
          "[HPO][termination][anneal_pt]")
{
    anneal_test::JournalCapture jc;
    const auto sp = anneal_spec("ANNEAL_PT");
    for (std::size_t k = 1; k <= kSweepCalls; ++k) {
        INFO("the host throws on call " << k);
        ThrowOnCall host{k};
        CHECK_THROWS_AS(tadah::hpo::run_anneal_pt_impl<ParamVec>(sp, host),
                        tadah::core::ShutdownRequested);
        CHECK(host.n.load() == k);
    }
}

TEST_CASE("Termination: ANNEAL_PA keeps ShutdownRequested raised in any "
          "phase, including the seed evaluation and the T0 calibration",
          "[HPO][termination][anneal_pa]")
{
    anneal_test::JournalCapture jc;
    const auto sp = anneal_spec("ANNEAL_PA");
    for (std::size_t k = 1; k <= kSweepCalls; ++k) {
        INFO("the host throws on call " << k);
        ThrowOnCall host{k};
        CHECK_THROWS_AS(tadah::hpo::run_anneal_pa_impl<ParamVec>(sp, host),
                        tadah::core::ShutdownRequested);
        CHECK(host.n.load() == k);
    }
}

TEST_CASE("Termination: the annealing worker pool keeps the exception that "
          "decides how the run ends", "[HPO][termination][anneal_pool]")
{
    /* When several workers throw in one epoch, the pool used to keep the
     * first: a failure to save best_pot.tadah that came after a signal or
     * a stagnation stop was dropped, and the run ended as a shutdown (exit
     * 130) or restarted instead of failing. One worker runs the tasks in
     * order, so the order of the throws is fixed. */
    std::atomic<bool> stop{false};
    tadah::hpo::anneal::ReplicaPool pool(1, stop);

    pool.run_epoch(3, [](std::size_t t) {
        if (t == 0) throw Host::StagnationStop{};
        if (t == 1) throw tadah::core::ShutdownRequested{};
        throw tadah::hpo::durable_best::Error("cannot save best_pot.tadah");
    });
    CHECK(stop.load());
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::hpo::durable_best::Error);
    CHECK_FALSE(pool.has_error());

    /* A stagnation stop never hides a signal stop... */
    pool.run_epoch(2, [](std::size_t t) {
        if (t == 0) throw Host::StagnationStop{};
        throw tadah::core::ShutdownRequested{};
    });
    CHECK_THROWS_AS(pool.rethrow_if_error(), tadah::core::ShutdownRequested);

    /* ... and a signal stop never hides an error. */
    pool.run_epoch(2, [](std::size_t t) {
        if (t == 0) throw std::runtime_error("evaluation broke");
        throw tadah::core::ShutdownRequested{};
    });
    CHECK_THROWS_WITH(pool.rethrow_if_error(), "evaluation broke");

    /* Among errors, the first one is kept. */
    pool.run_epoch(2, [](std::size_t t) {
        throw std::runtime_error(t == 0 ? "first" : "second");
    });
    CHECK_THROWS_WITH(pool.rethrow_if_error(), "first");
}

/* ======================================================================
 * Real signals: hidden, one process each (tests/CMakeLists.txt).
 * ==================================================================== */

TEST_CASE("Termination: SIGINT stops DLIB BFGS", "[.shutdown_dlib]")
{
    signal_case(plain_block(kBfgsLong), SIGINT, "SIGINT", 130);
}

TEST_CASE("Termination: SIGTERM stops DLIB BFGS", "[.shutdown_sigterm]")
{
    signal_case(plain_block(kBfgsLong), SIGTERM, "SIGTERM", 143);
}

TEST_CASE("Termination: SIGUSR2 stops DLIB BFGS", "[.shutdown_sigusr2]")
{
#ifdef __linux__
    signal_case(plain_block(kBfgsLong), SIGUSR2, "SIGUSR2", 140);
#else
    signal_case(plain_block(kBfgsLong), SIGUSR2, "SIGUSR2", 128 + SIGUSR2);
#endif
}

TEST_CASE("Termination: SIGINT stops NLopt SBPLX", "[.shutdown_nlopt]")
{
    /* SBPLX ends by itself after about 220 evaluations on this fixture. */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    MAXEVAL 100000\n"), SIGINT, "SIGINT", 130);
}

TEST_CASE("Termination: SIGINT stops MULTISTART and its later starts",
          "[.shutdown_multistart]")
{
    signal_case(
        "OPTIMIZER\n"
        "  INIT\n    METHOD LHS 3\n    SEED 11\n  ENDINIT\n"
        "  STRATEGY\n"
        "    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
        "    N_STARTS 3\n    BUDGET_PER_START 100000000\n"
        "    PARAM tol_gradinf 1e-300\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n", SIGINT, "SIGINT", 130, 1, [] {
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("MULTISTART start 1/3 -- running")
                  != std::string::npos);
            CHECK(journal.find("MULTISTART start 2/3") == std::string::npos);
        });
}

TEST_CASE("Termination: SIGINT stops PRESTAGE during the exploration",
          "[.shutdown_prestage]")
{
    signal_case(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
        "    MAXEVAL 100\n    TOPK 2\n"
        "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
        "      MAXEVAL 100000000\n      SEED 5\n    ENDEXPLORE\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n", SIGINT, "SIGINT", 130, 1, [] {
            /* The refinement does not run. */
            CHECK(read_file("hpo_run.log").find("REFINEMENT:")
                  == std::string::npos);
        });
}

TEST_CASE("Termination: SIGINT stops DLIB GFS with four threads",
          "[.shutdown_gfs]")
{
    /* Every GFS evaluation that starts after the signal throws; dlib's
     * thread pool used to abort the process on the second one. */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
        "    MAXEVAL 100000000\n    THREADS 4\n"), SIGINT, "SIGINT", 130, 4);
}

TEST_CASE("Termination: SIGINT stops ANNEAL_PT with two threads",
          "[.shutdown_anneal_pt]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PT\n"
        "    MAXEVAL 100000\n    THREADS 2\n    SEED 7\n"
        "    PARAM REPLICAS 2\n"), SIGINT, "SIGINT", 130, 2);
}

TEST_CASE("Termination: SIGINT stops ANNEAL_PA", "[.shutdown_anneal_pa]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PA\n"
        "    MAXEVAL 100000\n    POPULATION 8\n    SEED 7\n"),
        SIGINT, "SIGINT", 130);
}

TEST_CASE("Termination: SIGINT stops ANNEAL_PA with four threads",
          "[.shutdown_anneal_pa_threads]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO ANNEAL_PA\n"
        "    MAXEVAL 100000000\n    POPULATION 8\n    THREADS 4\n"
        "    SEED 7\n"), SIGINT, "SIGINT", 130, 4);
}

TEST_CASE("Termination: SIGINT stops ASA", "[.shutdown_asa]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000\n    SEED 7\n"), SIGINT, "SIGINT", 130);
}

TEST_CASE("Termination: SIGINT stops ASA block-parallel with four threads",
          "[.shutdown_asa_block]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000000\n    THREADS 4\n    SEED 7\n"
        "    PARAM Gener_Block 4\n    PARAM Gener_Block_Max 4\n"),
        SIGINT, "SIGINT", 130, 4);
}

#ifdef TADAH_USE_CERES
TEST_CASE("Termination: SIGINT stops CERES LBFGS", "[.shutdown_ceres]")
{
    /* With these tolerances LBFGS ends by itself after about 150
     * evaluations on this fixture. */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB CERES\n    ALGO LBFGS\n"
        "    MAXEVAL 100000\n    FTOL_REL 1e-300\n    XTOL_REL 1e-300\n"
        "    PARAM tol_gradinf 1e-300\n"), SIGINT, "SIGINT", 130);
}
#endif

TEST_CASE("Termination: SIGINT during NOISE_CALIBRATION",
          "[.shutdown_noise]")
{
    /* The calibration evaluations run before the optimiser; a signal
     * there used to escape hpo_run() without the final flush or the
     * journal trailer. */
    signal_case(
        "OPTIMIZER\n"
        "  NOISE\n    CALIBRATION 1000000\n  ENDNOISE\n"
        "  STRATEGY\n" + kBfgsLong + "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n", SIGINT, "SIGINT", 130, 1, [] {
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("NOISE_CALIBRATION: running 1000000 evals")
                  != std::string::npos);
            /* Stopped inside the calibration: it wrote no summary, and the
             * optimiser never started (its PROGRESS section opens it). */
            CHECK(journal.find("NOISE CALIBRATION") == std::string::npos);
            CHECK(journal.find("-- PROGRESS ") == std::string::npos);
        });
}

/* ======================================================================
 * drive_run(): every ending flushes the queued results and writes one
 * trailer. The body evaluates known points through a real HPO_Host
 * (QFLUSH 1000: nothing reaches the history files before the final flush)
 * and then ends the way an optimiser can.
 * ==================================================================== */

namespace {

constexpr std::size_t kDriverEvals = 7;

void check_driver_files(const std::string& expected_status)
{
    CHECK(journal_status(read_file("hpo_run.log")) == expected_status);
    CHECK(data_rows("loss.tadah").size() == kDriverEvals);
    check_run_files();
}

}  // namespace

TEST_CASE("Termination: drive_run completes a normal end with the "
          "optimiser's status", "[HPO][termination][run_driver]")
{
    HostFixture fx;
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] { fx.evaluate(kDriverEvals); }, fx.host, fx.log()));
    CHECK(status == 0);
    check_driver_files("COMPLETED");
}

TEST_CASE("Termination: drive_run ends a shutdown with the SHUTDOWN trailer "
          "and the signal status", "[HPO][termination][run_driver]")
{
    HostFixture fx;
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw tadah::core::ShutdownRequested{};
        }, fx.host, fx.log()));
    /* No real signal in this process: the status is the no-signal case of
     * shutdown_exit_status (the signal cases run in [.shutdown_*]). */
    CHECK(status == tadah::hpo::shutdown_exit_status(
                        tadah::core::SignalHandler::signal_seen()));
    check_driver_files("SHUTDOWN (" + tadah::core::SignalHandler::reason() +
                       ")");
}

TEST_CASE("Termination: drive_run rethrows an error after its trailer",
          "[HPO][termination][run_driver]")
{
    HostFixture fx;
    CHECK_THROWS_WITH(tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw std::runtime_error("optimiser broke");
        }, fx.host, fx.log()), "optimiser broke");
    check_driver_files("STOPPED with error: optimiser broke");
}

TEST_CASE("Termination: drive_run rethrows a non-standard exception after "
          "its trailer", "[HPO][termination][run_driver]")
{
    HostFixture fx;
    CHECK_THROWS_AS(tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw 42;
        }, fx.host, fx.log()), int);
    check_driver_files("STOPPED with unknown error");
}

TEST_CASE("Termination: drive_run ends a stagnation stop that no RESTART "
          "consumed as an ordinary stop", "[HPO][termination][run_driver]")
{
    HostFixture fx;
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw Host::StagnationStop{};
        }, fx.host, fx.log()));
    CHECK(status == 0);
    check_driver_files(
        "STOPPED -- stagnation (no new best within the stagnation window)");
}

TEST_CASE("Termination: a signal stop exits with 128 + the signal number",
          "[HPO][termination][run_driver]")
{
    using tadah::hpo::shutdown_exit_status;
    CHECK(shutdown_exit_status(SIGINT) == 130);
    CHECK(shutdown_exit_status(SIGTERM) == 143);
    CHECK(shutdown_exit_status(SIGUSR2) == 128 + SIGUSR2);
#ifdef __linux__
    CHECK(shutdown_exit_status(SIGUSR2) == 140);
#endif
    /* A ShutdownRequested nobody's signal raised: not a success, and not a
     * made-up signal either. */
    CHECK(shutdown_exit_status(0) == EXIT_FAILURE);
}

/* ======================================================================
 * The final flush writes every result, including those stranded behind
 * an iteration that never arrived (the in-run writer must wait there).
 * ==================================================================== */

TEST_CASE("Termination: the final flush writes results stranded behind a "
          "missing iteration, with a warning",
          "[HPO][termination][flush]")
{
    struct Row { std::size_t iter; double loss; };
    std::map<std::size_t, Row> buf;
    std::atomic<std::size_t> next{1};
    for (std::size_t it : {1u, 2u, 4u}) buf.emplace(it, Row{it, 0.5 * it});

    std::vector<std::size_t> written;
    std::vector<std::string> warnings;
    const auto rep = tadah::hpo::drain_remaining_results(
        buf, next,
        [&](const Row& r) { written.push_back(r.iter); },
        [&](const std::string& w) { warnings.push_back(w); });

    CHECK(written == std::vector<std::size_t>{1, 2, 4});
    CHECK(buf.empty());
    CHECK(next.load() == 5u);
    CHECK(rep.in_order == 2u);
    CHECK(rep.stranded == 1u);
    CHECK(rep.late == 0u);
    CHECK(rep.dropped == 0u);
    CHECK(rep.n_missing == 1u);
    REQUIRE(warnings.size() == 1u);
    CHECK(warnings[0] ==
          "HPO flush_remaining: 1 result(s) stranded behind missing "
          "iteration 3; writing them in iteration order (the missing "
          "iteration has no row)");
}

TEST_CASE("Termination: the final flush writes late and stranded results and "
          "reports every row it drops", "[HPO][termination][flush]")
{
    struct Row { std::size_t iter; };
    std::map<std::size_t, Row> buf;
    /* The in-run writer already skipped past 3 (gap-skip), so 3 is late;
     * 7 and 9 have no result; 12 fails to write. */
    std::atomic<std::size_t> next{6};
    for (std::size_t it : {3u, 6u, 8u, 10u, 11u, 12u}) buf.emplace(it, Row{it});

    std::vector<std::size_t> written;
    std::vector<std::string> warnings;
    const auto rep = tadah::hpo::drain_remaining_results(
        buf, next,
        [&](const Row& r) {
            if (r.iter == 12) throw std::runtime_error("disk full");
            written.push_back(r.iter);
        },
        [&](const std::string& w) { warnings.push_back(w); });

    CHECK(written == std::vector<std::size_t>{6, 3, 8, 10, 11});
    CHECK(buf.empty());
    CHECK(next.load() == 13u);
    CHECK(rep.in_order == 1u);
    CHECK(rep.stranded == 3u);
    CHECK(rep.late == 1u);
    CHECK(rep.dropped == 1u);
    CHECK(rep.n_missing == 2u);
    REQUIRE(warnings.size() == 3u);
    CHECK(warnings[0] ==
          "HPO flush_remaining: 4 result(s) stranded behind missing "
          "iterations 7, 9; writing them in iteration order (the missing "
          "iterations have no row)");
    CHECK(warnings[1] ==
          "HPO flush_remaining: 1 late result(s) for iteration(s) 3 arrived "
          "after the writer had skipped them; writing them out of sequence");
    CHECK(warnings[2] ==
          "HPO flush_remaining: dropping iter=12 after exception: disk full");
}

/* ======================================================================
 * NLopt: which end of optimize() is the algorithm stopping itself.
 * ==================================================================== */

TEST_CASE("Termination: NLopt self-termination is told apart from failure",
          "[HPO][termination][nlopt]")
{
    using tadah::hpo::nlopt_self_terminated;
    /* NLOPT_FAILURE without and with an algorithm stop message. */
    CHECK(nlopt_self_terminated(std::runtime_error("nlopt failure"), nullptr));
    CHECK(nlopt_self_terminated(std::runtime_error("nlopt failure"), ""));
    CHECK(nlopt_self_terminated(
        std::runtime_error("starting step size led to simplex that was too "
                           "small in dimension 0"),
        "starting step size led to simplex that was too small in "
        "dimension 0"));
    /* NLOPT_ROUNDOFF_LIMITED: nlopt::roundoff_limited says
     * "nlopt roundoff-limited", which a message test never matched. */
    CHECK(nlopt_self_terminated(nlopt::roundoff_limited(), nullptr));
    /* Failures. */
    CHECK_FALSE(nlopt_self_terminated(nlopt::forced_stop(), nullptr));
    CHECK_FALSE(nlopt_self_terminated(
        std::invalid_argument("nlopt invalid argument"), nullptr));
    CHECK_FALSE(nlopt_self_terminated(std::bad_alloc(), nullptr));
    CHECK_FALSE(nlopt_self_terminated(std::runtime_error("something else"),
                                      "a different stop message"));
}

/* ======================================================================
 * Which exception ends a run when concurrent evaluations throw several
 * (run_stop.h): an error, then a signal stop, then a stagnation stop.
 * ==================================================================== */

TEST_CASE("Termination: an error outranks a signal stop, which outranks a "
          "stagnation stop", "[HPO][termination][stash]")
{
    using tadah::hpo::ExceptionStash;
    using tadah::hpo::run_stop_rank;
    const auto stagnation =
        std::make_exception_ptr(tadah::hpo::StagnationStop{});
    const auto signal =
        std::make_exception_ptr(tadah::core::ShutdownRequested{});
    const auto persist = std::make_exception_ptr(
        tadah::hpo::durable_best::Error("cannot save best_pot.tadah"));

    CHECK(run_stop_rank(nullptr) == -1);
    CHECK(run_stop_rank(stagnation) == 0);
    CHECK(run_stop_rank(signal) == 1);
    CHECK(run_stop_rank(persist) == 2);
    CHECK(run_stop_rank(std::make_exception_ptr(42)) == 2);
    /* HPO_Host's stagnation stop is the same type for every host. */
    CHECK(run_stop_rank(std::make_exception_ptr(Host::StagnationStop{})) == 0);

    ExceptionStash stash;
    CHECK_FALSE(stash.has());
    CHECK(stash.take() == nullptr);
    CHECK_NOTHROW(stash.rethrow_if_any());

    stash.offer(stagnation);
    stash.offer(signal);
    stash.offer(stagnation);
    CHECK_THROWS_AS(stash.rethrow_if_any(), tadah::core::ShutdownRequested);
    CHECK_FALSE(stash.has());

    stash.offer(signal);
    stash.offer(persist);
    stash.offer(signal);
    stash.offer(std::make_exception_ptr(std::runtime_error("later error")));
    REQUIRE(stash.has());
    CHECK_THROWS_AS(stash.rethrow_if_any(), tadah::hpo::durable_best::Error);

    /* A stash emptied by take() ranks from scratch. */
    stash.offer(persist);
    (void)stash.take();
    stash.offer(stagnation);
    CHECK_THROWS_AS(stash.rethrow_if_any(), tadah::hpo::StagnationStop);
}

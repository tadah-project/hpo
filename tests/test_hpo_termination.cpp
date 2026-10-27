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

#include "pc_process_workers.h"  // before catch2/catch.hpp
#include "catch2/catch.hpp"

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <tadah/core/checkpoint.h>      // core::ShutdownRequested
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/box_penalty.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>         // hpo_rank_suffix
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/result_drain.h>
#include <tadah/hpo/run_driver.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/run_stop.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/parallel.h>

#include "anneal_test_support.h"
#include "finish_within.h"
#include "rhoform_test_support.h"

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
#include <iomanip>
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
#include <tuple>
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
    /* The energy error scale of EWEIGHT 1 for the 4-atom
     * Ta cells these fits train on: an energy row factor 1. */
    c.add("ESIGMA", 250.0);
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
    c.add("HPOTARGET",  hpo_cfg.string());
    c.add("VALIDATION", val_cfg.string());
    return c;
}

/* The root lines of config.hpo after the OPTIMIZER block that most cases
 * share: the objective and the three-dimensional search space. */
const std::string kRootLines =
    "LOSS L2\n"
    "FAILSCORE 1e9\n"
    "QFLUSH 1000\n"
    "DIGITS 15\n"
    "PC_ERMSE 0 1\n"
    "OPTIM CGRID2B (1) 0.1 0.9\n"
    "OPTIM CGRID2B (2) 1.0 2.0\n"
    "OPTIM SGRID2B (2) 4.0 6.0\n";

/* One hpo_run() setup in a scratch directory. `optimizer` is the complete
 * OPTIMIZER ... ENDOPTIMIZER block and `root` the rest of config.hpo (by
 * default the objective and search space every case shares). */
struct TaRun {
    ScratchDir sd;
    tadah::core::Context ctx;
    fs::path hpo, val;

    /* `new_process`: reset the process-wide HPOLog and HPO_Host state
     * first, as a new process would start; false runs as a later run of an
     * embedding in the same process, with only hpo_run()'s own per-run
     * resets. */
    explicit TaRun(const std::string& optimizer,
                   const std::string& root = kRootLines,
                   bool new_process = true)
    {
        tadah::core::PeriodicTable::initialize();
        /* HPOLog and the HPO_Host statics are process-wide; a run in the
         * same process as another test must start as a new process would. */
        if (new_process) {
            tadah::hpo::HPOLog::instance().reset_for_test();
            Host::reset_globals_for_test();
        }
        const fs::path train = sd.dir / "train.db";
        const fs::path valdb = sd.dir / "val.db";
        val = sd.dir / "config.val";
        hpo = sd.dir / "config.hpo";
        write_ta_db(train, 0.0, -0.001);
        write_ta_db(valdb, 0.07, -0.04);
        std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
        std::ofstream(hpo) << optimizer << root;
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

/* The trailer of a run whose RESTART used up its MAX restarts: the
 * stagnation window (AFTER_STAGNATION) and the restarts made. */
std::string restarts_used_status(std::size_t window, std::size_t restarts)
{
    return "STOPPED -- stagnation: no new best in " + std::to_string(window) +
           " evals after " + std::to_string(restarts) + " restart(s)";
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

/* A stagnation stop: the latest evaluation when it fired, and the calls of
 * its window answered from the run's record (repeat_points.h), which the
 * window counts as evaluations without a new best. */
struct StagnationWindow {
    std::size_t eval = 0;
    std::size_t reused = 0;
};

/* Every stagnation stop of the journal, in order: one at an evaluation
 * ("StagnationStop fired at eval N; ... (threshold=W evals without
 * improvement[, K of them call(s) answered from the run's record])") or at
 * a call answered from the record ("StagnationStop fired at a repeat of
 * eval M answered from the run's record, after eval N; ... K of them
 * ..."). */
std::vector<StagnationWindow> stagnation_windows(const std::string& journal)
{
    std::vector<StagnationWindow> out;
    const std::regex re(
        R"(StagnationStop fired at (?:eval (\d+)|a repeat of [^\n]*? )"
        R"(answered from the run's record, after eval (\d+));[^\n]*?)"
        R"(evals without improvement(?:, (\d+) of them call\(s\) answered )"
        R"(from the run's record)?\))");
    for (auto it = std::sregex_iterator(journal.begin(), journal.end(), re);
         it != std::sregex_iterator(); ++it) {
        const auto& m = *it;
        StagnationWindow w;
        w.eval = std::stoull(m[1].matched ? m[1].str() : m[2].str());
        w.reused = m[3].matched ? std::stoull(m[3].str()) : 0;
        out.push_back(w);
    }
    return out;
}

/* The seed S a RESTART block without SEED drew from std::random_device,
 * as the journal names it ("RESTART: the RESTART block has no SEED, so its
 * restart points are drawn from std::random_device: restart r draws with S
 * + r ..."); none without the line. */
std::optional<std::uint64_t> drawn_restart_seed(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(
            journal, m,
            std::regex(R"(RESTART: the RESTART block has no SEED, so its )"
                       R"(restart points are drawn from std::random_device: )"
                       R"(restart r draws with (\d+) \+ r \(SEED (\d+) in )"
                       R"(the RESTART block draws the same points again\))")) &&
        m[1] == m[2])
        return std::stoull(m[1]);
    return std::nullopt;
}

/* The journal line of restart `r` of `max` of a RESTART block without
 * SEED: "RESTART r/max -- fresh init from <method>; RESTART SEED <S + r>"
 * and the optimisers' seeds `seeds` ("SEED 6"; "" for none), S the drawn
 * seed (drawn_restart_seed); "(no drawn RESTART seed)" without one. */
std::string drawn_round_line(const std::string& journal, std::size_t r,
                             std::size_t max, const std::string& method,
                             const std::string& seeds)
{
    const auto s = drawn_restart_seed(journal);
    if (!s) return "(no drawn RESTART seed)";
    return "RESTART " + std::to_string(r) + "/" + std::to_string(max) +
           " -- fresh init from " + method + "; RESTART SEED " +
           std::to_string(*s + r) + (seeds.empty() ? "" : ", " + seeds) + "\n";
}

/* A loss.tadah value. The default FAILSCORE, the largest double, prints as
 * 1.797693134862316e+308 with DIGITS 15, which rounds above it: std::stod
 * throws on it, and this reads it as +inf, which is no feasible loss
 * either. */
double loss_value(const std::string& token)
{
    char* end = nullptr;
    const double v = std::strtod(token.c_str(), &end);
    INFO("loss.tadah value '" << token << "'");
    REQUIRE(end != token.c_str());
    REQUIRE(*end == '\0');
    return v;
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
        const double l = loss_value(row.back());
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
    bool               stalled = false;   /* evaluations stopped first */
    std::size_t        evals_at_raise = 0;
    std::size_t        evals_final = 0;
};

/* Runs hpo_run(ctx) on a worker thread and raises `signo` on this thread
 * once `raise_at` (at least three) evaluations have started, so
 * SignalHandler::install() has run. None of the configurations can end by
 * itself within the time this takes: each runs for many seconds, or
 * thousands of evaluations, without the signal, so the signal is what ends
 * the run.
 *
 * An evaluation on this fixture takes milliseconds, so a run whose count of
 * evaluations stops moving for kStalledAfter before `raise_at` is stuck. If
 * three evaluations have started, the signal is raised at once, as a user
 * would, and `stalled` records it. However it was raised, hpo_run() must
 * return within kHonouredWithin of the signal. A run that does not can be
 * neither interrupted nor joined, so the process reports it and exits with
 * EXIT_FAILURE: the case's CTest entry fails with a message about 80 s after
 * the run stalls (kStalledAfter + kHonouredWithin) instead of at the CTest
 * timeout. A run that stalls before its third evaluation, when the handler
 * may not be installed yet, or that neither ends nor reaches `raise_at`
 * within kReachWithin, is reported the same way without a signal. */
SignalOutcome run_and_signal(tadah::core::Context& ctx, int signo,
                             std::size_t raise_at = 3)
{
    using Clock = std::chrono::steady_clock;
    constexpr auto kStalledAfter   = std::chrono::seconds(20);
    constexpr auto kHonouredWithin = std::chrono::seconds(60);
    constexpr auto kReachWithin    = std::chrono::minutes(4);
    constexpr std::size_t kHandlerInstalled = 3;
    raise_at = std::max<std::size_t>(raise_at, kHandlerInstalled);

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
    const auto deadline = Clock::now() + kReachWithin;
    std::size_t seen = 0;
    auto seen_since = Clock::now();
    while (!finished.load() && Host::g_iter_value() < raise_at &&
           Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        const std::size_t n = Host::g_iter_value();
        if (n != seen) {
            seen = n;
            seen_since = Clock::now();
        } else if (Clock::now() - seen_since >= kStalledAfter) {
            out.stalled = true;
            break;
        }
    }
    const std::size_t reached = Host::g_iter_value();
    if (!finished.load() && reached < raise_at &&
        (!out.stalled || reached < kHandlerInstalled)) {
        std::cerr << "FAILED: hpo_run() " << (out.stalled
                         ? "stopped evaluating for "
                               + std::to_string(kStalledAfter.count())
                               + " s at "
                         : "neither ended nor reached the signal's "
                               + std::to_string(raise_at)
                               + " evaluations within "
                               + std::to_string(kReachWithin.count())
                               + " minutes; it stood at ")
                  << reached << " evaluation(s), before the signal could be "
                  << "raised. Its thread cannot be joined, so the test "
                  << "process exits now\n"
                  << std::flush;
        std::_Exit(EXIT_FAILURE);
    }
    if (!finished.load()) {
        /* The handler runs on this thread before raise() returns, so every
         * evaluation that starts after the count below is taken sees the
         * stop flag. */
        std::raise(signo);
        out.evals_at_raise = Host::g_iter_value();
        out.raised = true;
        const auto honoured_by = Clock::now() + kHonouredWithin;
        while (!finished.load() && Clock::now() < honoured_by)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!finished.load()) {
            std::cerr << "FAILED: hpo_run() did not return within "
                      << kHonouredWithin.count() << " s of the signal; "
                      << "evaluations " << (out.stalled ? "had stopped at "
                                                        : "stood at ")
                      << out.evals_at_raise << " when it was raised. The "
                      << "run is stuck where the stop flag is never "
                      << "polled, and its thread cannot be joined, so the "
                      << "test process exits now\n"
                      << std::flush;
            std::_Exit(EXIT_FAILURE);
        }
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
                       std::size_t in_flight, std::size_t raise_at)
{
    REQUIRE(o.raised);
    INFO("evaluations stopped before the signal: " << o.stalled
         << " (at " << o.evals_at_raise << ", signal due at " << raise_at
         << ")");
    CHECK_FALSE(o.stalled);
    CHECK(o.evals_at_raise >= raise_at);
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

/* `more` runs extra checks while the run's files still exist; the signal
 * is raised once `raise_at` evaluations have started. */
void signal_case(const std::string& optimizer, int signo,
                 const std::string& name, int expected_status,
                 std::size_t in_flight = 1,
                 const std::function<void()>& more = {},
                 std::size_t raise_at = 3)
{
    TaRun run(optimizer);
    const SignalOutcome o = run_and_signal(run.ctx, signo, raise_at);
    check_signal_stop(o, signo, name, expected_status, in_flight, raise_at);
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

    explicit HostFixture(const std::string& root = kRootLines,
                         const std::string& optimizer = plain_block(kBfgsLong))
        : run(optimizer, root),
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

/* A PC_LAMMPS script in the run's directory: a bcc Ta cell run once with
 * the candidate potential, the variable `var` its energy per atom (or the
 * equal-style `formula` given). */
void write_pe_script(const fs::path& path, const std::string& var,
                     const std::string& formula = "pe/atoms")
{
    std::ofstream(path)
        << "clear\nunits metal\nboundary p p p\natom_style atomic\n"
           "lattice bcc 3.30\nregion box block 0 2 0 2 0 2 units lattice\n"
           "create_box 1 box\ncreate_atoms 1 region box\nmass 1 180.95\n"
           "pair_style tadah\npair_coeff * * EXTERN Ta\n"
           "thermo_style custom step pe\nrun 0 post no\n"
           "variable " << var << " equal " << formula << "\n";
}

/* The values of column `name` of a history file (outvar.tadah, ...), one
 * per data row, by the header line's column names. */
std::vector<double> column_of(const fs::path& path, const std::string& name)
{
    std::ifstream in(path);
    std::string header;
    REQUIRE(std::getline(in, header));
    std::istringstream hs(header);
    std::vector<std::string> names;
    for (std::string t; hs >> t;) names.push_back(t);
    const auto it = std::find(names.begin(), names.end(), name);
    INFO("header of " << path << ": " << header);
    REQUIRE(it != names.end());
    const auto col = static_cast<std::size_t>(it - names.begin());
    std::vector<double> out;
    for (const auto& row : data_rows(path)) {
        REQUIRE(row.size() == names.size());
        out.push_back(std::stod(row[col]));
    }
    return out;
}

/* TADAH_PC_ISOLATION unset while this lives (it overrides the
 * configuration's PC_ISOLATION), as the case found it afterwards. */
struct PcIsolationEnvUnset {
    std::optional<std::string> saved;
    PcIsolationEnvUnset()
    {
        if (const char* e = std::getenv("TADAH_PC_ISOLATION")) saved = e;
        ::unsetenv("TADAH_PC_ISOLATION");
    }
    ~PcIsolationEnvUnset()
    {
        if (saved) ::setenv("TADAH_PC_ISOLATION", saved->c_str(), 1);
        else       ::unsetenv("TADAH_PC_ISOLATION");
    }
};

/* The journal lines that contain `what`. */
std::vector<std::string> lines_with(const std::string& text,
                                    const std::string& what)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);)
        if (l.find(what) != std::string::npos) out.push_back(l);
    return out;
}

}  // namespace

/* ======================================================================
 * Stop criteria: an ordinary end leaves the same invariants.
 * ==================================================================== */

/* PROCESS is the serial build's default PC_ISOLATION. With no PC_LAMMPS
 * line there is nothing to isolate, and the journal said PROCESS was
 * "requested" whether or not the configuration (or TADAH_PC_ISOLATION)
 * asked for it. */
TEST_CASE("Journal: without PC_LAMMPS lines the PC_ISOLATION line says "
          "whether PROCESS was requested or is the default",
          "[HPO][pc_isolation][journal]")
{
    /* The environment override counts as a request; this case controls
     * it. */
    struct NoIsolationEnv {
        std::optional<std::string> saved;
        NoIsolationEnv()
        {
            if (const char* e = std::getenv("TADAH_PC_ISOLATION"))
                saved = e;
            ::unsetenv("TADAH_PC_ISOLATION");
        }
        ~NoIsolationEnv()
        {
            /* Leave the variable as the case found it: a later case in
             * this process must not see the PROCESS a section set. */
            if (saved) ::setenv("TADAH_PC_ISOLATION", saved->c_str(), 1);
            else       ::unsetenv("TADAH_PC_ISOLATION");
        }
    } no_env;
    const std::string optimizer = plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 1\n"
        "    SEED 1\n");
    /* The journal's line names the mode and then says what it does (it
     * was "PC_ISOLATION: <mode> -- ..."); the echo of the target file has
     * the key and its value alone. */
    const auto isolation_line = [](const std::string& journal) {
        std::istringstream in(journal);
        std::string found;
        for (std::string l; std::getline(in, l);) {
            const auto p = l.find("PC_ISOLATION ");
            if (p != std::string::npos &&
                l.find(':', p) != std::string::npos) {
                REQUIRE(found.empty());
                found = l;
            }
        }
        return found;
    };

#ifndef TADAH_HPO_MPI
    SECTION("PROCESS by default") {
        TaRun run(optimizer);
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
        const std::string line = isolation_line(read_file("hpo_run.log"));
        CHECK_THAT(line, Catch::EndsWith(
            "PC_ISOLATION PROCESS (the default): the target file has no "
            "PC_LAMMPS or PC_FAILPROBE line, so no worker process starts"));
    }
    SECTION("PROCESS written in the configuration") {
        TaRun run(optimizer + "PC_ISOLATION PROCESS\n");
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
        CHECK_THAT(isolation_line(read_file("hpo_run.log")), Catch::EndsWith(
            "PC_ISOLATION PROCESS: the target file has no PC_LAMMPS or "
            "PC_FAILPROBE line, so no worker process starts"));
    }
    SECTION("PROCESS set by TADAH_PC_ISOLATION") {
        ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);
        TaRun run(optimizer);
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
        CHECK_THAT(isolation_line(read_file("hpo_run.log")), Catch::EndsWith(
            "PC_ISOLATION PROCESS: the target file has no PC_LAMMPS or "
            "PC_FAILPROBE line, so no worker process starts"));
    }
#else
    /* The MPI build has no PROCESS pool (pc_worker_main refuses to start
     * in it): its default is THREAD, said so in the journal, and a PROCESS
     * request -- the configuration's or TADAH_PC_ISOLATION's -- is a
     * startup error. Before this case knew the MPI build it asserted the
     * serial build's lines there, and failed. */
    const std::string refusal =
        "PC_ISOLATION PROCESS is not supported in the MPI build "
        "(TADAH_HPO_MPI): the worker processes cannot initialize MPI. Use "
        "the serial build, or PC_ISOLATION THREAD.";
    SECTION("THREAD by default in the MPI build") {
        TaRun run(optimizer);
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
        CHECK_THAT(isolation_line(read_file("hpo_run.log")), Catch::EndsWith(
            "PC_ISOLATION THREAD: the MPI build of tadah runs the scripts "
            "inside the tadah process"));
    }
    SECTION("PROCESS written in the configuration is refused in the MPI "
            "build") {
        TaRun run(optimizer + "PC_ISOLATION PROCESS\n");
        CHECK_THROWS_WITH(tadah::hpo::hpo_run(run.ctx), refusal);
    }
    SECTION("PROCESS set by TADAH_PC_ISOLATION is refused in the MPI build") {
        ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);
        TaRun run(optimizer);
        CHECK_THROWS_WITH(tadah::hpo::hpo_run(run.ctx), refusal);
    }
#endif
}

/* Pins: the startup journal lines on OpenMP and on FAILSCORE_GRADING ON
 * say what they set: "OpenMP: at most <n> threads (OMP_NUM_THREADS=<v>),
 * for the OpenMP parts (neighbour lists, data statistics), not for the
 * LAMMPS scripts", "OpenMP: OMP_WAIT_POLICY=<v>; it sets how the threads
 * of the OpenMP parts wait between them (the workers of the evaluations
 * wait without spinning); 'passive' suits a shared node" (in a build with
 * OpenMP), and, under FAILSCORE_GRADING ON without a PC_FAILPROBE line,
 * "FAILSCORE_GRADING ON: a failed evaluation scores from 0.999 FAILSCORE
 * up to FAILSCORE, in four bands from the lowest: A, ...; no PC_FAILPROBE
 * line: band C scores the top of its range, and band D is graded only for
 * a --skip-above or --skip-rel cap and for BUMPGATE, RHOGATE and
 * RHOBARGATE".
 * Before the fix: "OpenMP: max threads = <n>  (OMP_NUM_THREADS=<v>);
 * drives only the residual OpenMP kernels (NN lists, data-stats), NOT the
 * pc_lammps fan-out", "...; affects only the residual OMP kernels now
 * (pool idles on a futex); prefer 'passive' on shared nodes" and
 * "FAILSCORE_GRADING: ON -- failed evals graded in [0.999*FS, FS), bands
 * A-D (see docs); no PC_FAILPROBE (bands C/D constant)": the checks
 * fail. */
TEST_CASE("Journal: the OpenMP and FAILSCORE_GRADING lines say what they "
          "set", "[HPO][journal][termination]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                          "    MAXEVAL 1\n    SEED 1\n"),
              kRootLines + "FAILSCORE_GRADING ON\n");
    REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
#ifdef TADAH_ENABLE_OPENMP
    const auto openmp = lines_with(journal, "OpenMP: at most ");
    REQUIRE(openmp.size() == 1u);
    CHECK_THAT(openmp[0], Catch::EndsWith(
        "), for the OpenMP parts (neighbour lists, data statistics), not for "
        "the LAMMPS scripts"));
    /* A count of 1 takes the singular (it was "at most 1 threads"). */
    const int omp_max = omp_get_max_threads();
    CHECK(openmp[0].find("OpenMP: at most " +
                         (omp_max == 1 ? std::string("1 thread (")
                                       : std::to_string(omp_max) +
                                             " threads (")) !=
          std::string::npos);
    const auto wait = lines_with(journal, "OpenMP: OMP_WAIT_POLICY=");
    REQUIRE(wait.size() == 1u);
    CHECK_THAT(wait[0], Catch::EndsWith(
        "; it sets how the threads of the OpenMP parts wait between them (the "
        "workers of the evaluations wait without spinning); 'passive' suits a "
        "shared node"));
#endif
    const auto grading = lines_with(
        journal, "FAILSCORE_GRADING ON: a failed evaluation scores from 0.999 "
                 "FAILSCORE up to FAILSCORE, in four bands from the lowest: "
                 "A, a --hotvar value outside its range; B, a script or "
                 "PC_OBSERVABLE term that failed; C, a simulation that did "
                 "not finish; D, a potential rejected as unusable (by "
                 "PC_FAILPROBE --gate, a --skip-above or --skip-rel cap, "
                 "BUMPGATE, RHOGATE or RHOBARGATE, a refused fit, or a "
                 "potential LAMMPS cannot build); ");
    REQUIRE(grading.size() == 1u);
    CHECK_THAT(grading[0], Catch::EndsWith(
        "; no PC_FAILPROBE line: band C scores the top of its range, and band "
        "D is graded only for a --skip-above or --skip-rel cap and for "
        "BUMPGATE, RHOGATE and RHOBARGATE"));
    /* Without THREADS one worker runs the evaluations, in the singular (it
     * was "THREADS: 1 workers run the work of the evaluations"). */
    CHECK(lines_with(journal, "THREADS: 1 worker runs the work of the "
                              "evaluations").size() == 1u);
}

/* Pins: under FAILSCORE_GRADING ON with a PC_FAILPROBE line, the startup
 * journal line ends with the probe's script and what its --outvar grades:
 * "...; PC_FAILPROBE probe.in: its --outvar S grades band C and the band-D
 * failures of --gate and of a potential LAMMPS cannot build (--gate 0.5)",
 * and "(no --gate)" for a line without --gate. Each run stops when it
 * loads its validation data (the validation configuration names a dataset
 * that does not exist), after its startup lines: the case needs no
 * evaluation.
 * Before the fix: "FAILSCORE_GRADING: ON -- failed evals graded in
 * [0.999*FS, FS), bands A-D (see docs); PC_FAILPROBE 'probe.in' S=S
 * gate=0.500000" (and "gate=off" without --gate), which the case does not
 * find; that probe note after the words of the fix fails the EndsWith
 * check. */
TEST_CASE("Journal: the FAILSCORE_GRADING ON line names the PC_FAILPROBE "
          "line and what its value grades", "[HPO][journal][termination]")
{
    PcIsolationEnvUnset no_env;
    const std::string optimizer = plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 1\n"
        "    SEED 1\n");
    struct Case { std::string gate, note; };
    for (const Case& c : std::vector<Case>{{" --gate 0.5", "(--gate 0.5)"},
                                           {"", "(no --gate)"}}) {
        INFO("PC_FAILPROBE options:" << c.gate);
        TaRun run(optimizer,
                  kRootLines + "FAILSCORE_GRADING ON\nPC_ISOLATION THREAD\n"
                               "PC_FAILPROBE --script probe.in --outvar S" +
                      c.gate + "\n");
        std::ofstream(run.sd.dir / "probe.in") << "variable S equal 0.0\n";
        std::ofstream(run.val)
            << "DBFILE " << (run.sd.dir / "missing.db").string() << "\n";
        CHECK_THROWS(tadah::hpo::hpo_run(run.ctx));
        const std::string journal = read_file("hpo_run.log");
        INFO(journal);
        const auto grading = lines_with(
            journal, "FAILSCORE_GRADING ON: a failed evaluation scores from "
                     "0.999 FAILSCORE up to FAILSCORE");
        REQUIRE(grading.size() == 1u);
        CHECK_THAT(grading[0], Catch::EndsWith(
            "; PC_FAILPROBE probe.in: its --outvar S grades band C and the "
            "band-D failures of --gate and of a potential LAMMPS cannot "
            "build " + c.note));
    }
}

/* Pins: a journal that cannot be opened is a WARNING on standard error
 * naming its path: "[HPO] WARNING: the journal <path> cannot be opened,
 * so the run writes no journal".
 * Before the fix: "[HPOLog] Cannot open log file: <path>": the check
 * fails. */
TEST_CASE("Journal: a journal that cannot be opened is a warning naming its "
          "path", "[HPO][journal]")
{
    ScratchDir sd;
    const std::string path = (sd.dir / "missing" / "hpo_run.log").string();
    tadah::hpo::HPOLog::instance().reset_for_test();
    std::ostringstream err;
    std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
    tadah::hpo::HPOLog::instance().open(path, "test", sd.dir.string(), 0);
    std::cerr.rdbuf(old);
    CHECK(err.str() == "[HPO] WARNING: the journal " + path +
                           " cannot be opened, so the run writes no "
                           "journal\n");
    tadah::hpo::HPOLog::instance().reset_for_test();
}

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
    /* Pins: the spent MAXEVAL is the run's status. Before the fix: the
     * trailer said COMPLETED. */
    CHECK(journal_status(journal) == "DLIB:BFGS: MAXEVAL reached");
}

namespace {

/* The root lines of a run with one dimension, SGRID2B (2) in [4, 6], whose
 * loss falls towards the lower bound 4: a gradient search ends on it. The
 * loss is PC_ERMSE alone, so every evaluation is a function of its point
 * (HPOTargets::repeat_reuse_blocker() is ""). */
const std::string kLowerBoundRoot =
    "LOSS L2\n"
    "FAILSCORE 1e9\n"
    "QFLUSH 1000\n"
    "DIGITS 15\n"
    "PC_ERMSE 0 1\n"
    "OPTIM SGRID2B (2) 4.0 6.0\n";

/* The rows of params.tadah whose parameters repeat an earlier row's. */
std::size_t repeated_rows()
{
    std::set<std::vector<std::string>> seen;
    std::size_t n = 0;
    for (const auto& row : data_rows("params.tadah")) {
        const std::vector<std::string> params(row.begin() + 2, row.end());
        if (!seen.insert(params).second) ++n;
    }
    return n;
}

/* The number on the DIAGNOSTICS line `label` ("" when there is none). */
std::string diagnostics_count(const std::string& journal,
                              const std::string& label)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(label + R"(\s*: ([0-9]+))")))
        return m[1];
    return "";
}

}  // namespace

/* Pins: LD_MMA (MAXEVAL 60, XTOL_REL 1e-8) on the run whose loss falls
 * towards SGRID2B (2) = 4 evaluates no point twice: held at the bound (u =
 * 0, where NLopt's XTOL_REL test is vacuous) MMA asks for it and its
 * gradient again and again, and each repeat gets the loss and gradient of
 * the bound's evaluation, eval 16: 18 rows, none repeating another, and
 * NLopt's own MAXEVAL, which counts its 60 calls, ends the search ("NLopt
 * finished: MAXEVAL_REACHED -- maximum evaluations exhausted after 18 of
 * MAXEVAL 60 evaluation(s); NLopt counts its calls, and 54 of them asked
 * for points the run had evaluated ..."); the journal counts the reuses at
 * 10 ("NLOPT:LD_MMA: 10 calls of this search have asked for points the run
 * had evaluated") and DIAGNOSTICS has "Repeated points reused : 54".
 * Before the fix: 60 rows, 42 of them repeats of the bound and its
 * stencil (20 calls), MAXEVAL_REACHED; until the run kept one record
 * (2026-10-12) the count line said "points it evaluated"; until a call
 * answered from an earlier search's evaluation was named as such (the
 * review of fix/repeat-followups) both lines said "asked again for". */
TEST_CASE("Repeated points: an NLopt run of fit metrics evaluates no point "
          "twice", "[HPO][termination][repeat_points]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_MMA\n"
                          "    MAXEVAL 60\n    XTOL_REL 1e-8\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("params.tadah").size() == 18u);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    CHECK(journal.find("NLopt finished: MAXEVAL_REACHED -- maximum evaluations "
                       "exhausted after 18 of MAXEVAL 60 evaluation(s); NLopt "
                       "counts its calls, and 54 of them asked for points "
                       "the run had evaluated") != std::string::npos);
    CHECK(journal.find("NLOPT:LD_MMA asked again for the point of eval 16 "
                       "(SGRID2B[2] = 4) and its gradient") !=
          std::string::npos);
    CHECK(journal.find("NLOPT:LD_MMA: 10 calls of this search have asked "
                       "for points the run had evaluated") !=
          std::string::npos);
    CHECK(journal.find("NLOPT:LD_MMA: 100 calls") == std::string::npos);
    CHECK(diagnostics_count(journal, "Repeated points reused") == "54");
}

/* Pins: DLIB:BFGS (MAXEVAL 60) on the same run reaches SGRID2B (2) = 4,
 * takes its one-sided gradient there, and its next step, held to the box,
 * cannot move the point: the line search is not run, the loss has not
 * changed, and the run ends "DLIB:BFGS: FTOL_ABS reached" after 15
 * evaluations, no row repeating another; DIAGNOSTICS has "Line searches not
 * run : 1".
 * Before the fix: 18 rows, the last 3 the bound and its stencil again (the
 * line search evaluated the bound, and BFGS took its gradient again), then
 * FTOL_ABS. */
TEST_CASE("Repeated points: a DLIB:BFGS run that reaches a bound evaluates "
          "it once", "[HPO][termination][repeat_points]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 60\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("params.tadah").size() == 15u);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == "DLIB:BFGS: FTOL_ABS reached");
    CHECK(diagnostics_count(journal, "Line searches not run") == "1");
}

/* Pins: under PARAM tol_gradinf, whose test reads the whole gradient,
 * DLIB:BFGS (MAXEVAL 60) held at SGRID2B (2) = 4 ends "DLIB:BFGS: no step
 * within the OPTIM box", the stop line naming the bound ("SGRID2B[2] at
 * its lower bound 4, with a gradient that points out of the box"), after
 * 15 evaluations, no row repeating another.
 * Before the fix: the bound and its gradient were evaluated until MAXEVAL,
 * 60 rows, "DLIB:BFGS: MAXEVAL reached"; without MAXEVAL the run never
 * ended. */
TEST_CASE("Repeated points: a DLIB:BFGS run that no step can move ends",
          "[HPO][termination][repeat_points]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 60\n    PARAM tol_gradinf 1e-12\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("params.tadah").size() == 15u);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == "DLIB:BFGS: no step within the OPTIM box");
    CHECK(journal.find("SGRID2B[2] at its lower bound 4, with a gradient that "
                       "points out of the box") != std::string::npos);
}

namespace {

/* The RESTART block of the cases below: a window of 5, one restart from a
 * RANDOM point (seeded). */
const std::string kRepeatRestart =
    "    RESTART\n      AFTER_STAGNATION 5\n      MAX 1\n"
    "      METHOD RANDOM 1\n      SEED 5\n    ENDRESTART\n";

/* hpo_run() of `run`, which must return within two minutes (a search that
 * only asks for points it has makes no evaluation and could spin). */
int repeat_run_within(TaRun& run, const std::string& what)
{
    int status = -1;
    hpo_test::finish_within(std::chrono::seconds(120), what, [&] {
        status = tadah::hpo::hpo_run(run.ctx);
    });
    return status;
}

}  // namespace

/* Pins: LD_MMA held at SGRID2B (2) = 4 (u = 0, where NLopt's XTOL_REL test
 * is vacuous) with RESTART as its only stop (AFTER_STAGNATION 5, MAX 1):
 * each call that asks for the bound again is answered from the record, and
 * the window counts it as one evaluation without a new best, so the window
 * ends the search after 3 reused calls and eval 18 ("StagnationStop fired
 * at a repeat of eval 16 answered from the run's record, after eval 18;
 * ... 3 of them call(s) answered from the run's record"), RESTART
 * relaunches it once from a RANDOM point ("RESTART 1/1 --
 * fresh init from RANDOM"), and the run ends "STOPPED -- stagnation: no new
 * best in 5 evals after 1 restart(s)" after 23 evaluations.
 * Before the fix: the reused calls never reached the window, nothing else
 * stops LD_MMA, and the search spun without an evaluation ("100000000
 * calls of this search have asked again ..."): the run did not return
 * within two minutes (v1-dev, which evaluated every repeat: 26 rows, the
 * same status); until the run kept one record (2026-10-12) the line said
 * "answered from the search's record" and "each repeat answered from the
 * record counted as one". */
TEST_CASE("Repeated points: the RESTART window counts a reused repeat",
          "[HPO][termination][repeat_points][restart]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_MMA\n" +
                          kRepeatRestart),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(
                        run, "LD_MMA at a bound under RESTART alone"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal_status(journal) == restarts_used_status(5, 1));
    CHECK(journal.find("RESTART 1/1 -- fresh init from RANDOM") !=
          std::string::npos);
    CHECK(journal.find("StagnationStop fired at a repeat of eval 16 answered "
                       "from the run's record, after eval 18") !=
          std::string::npos);
    CHECK(journal.find("5 evals without improvement, 3 of them call(s) "
                       "answered from the run's record)") != std::string::npos);
    CHECK(diagnostics_count(journal, "Repeated points reused") == "3");
    CHECK(data_rows("loss.tadah").size() == 23u);
}

/* Pins: the same search with MAXEVAL 60: the window ends the search held at
 * the bound before NLopt's count of calls does, RESTART relaunches it once
 * ("RESTART 1/1 -- fresh init from RANDOM"), and the run ends "STOPPED --
 * stagnation: no new best in 5 evals after 1 restart(s)" after 23
 * evaluations, as without MAXEVAL.
 * Before the fix: no restart: the reused calls never reached the window,
 * and NLopt's count of 60 calls ended the search after 18 evaluations, 54
 * of its calls reused, MAXEVAL_REACHED (v1-dev: 26 rows, a restart, the
 * stagnation status). */
TEST_CASE("Repeated points: the RESTART window ends a search that only "
          "asks again before NLopt's MAXEVAL does",
          "[HPO][termination][repeat_points][restart]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LD_MMA\n"
                          "    MAXEVAL 60\n" + kRepeatRestart),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(
                        run, "LD_MMA at a bound under RESTART and MAXEVAL"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal.find("RESTART 1/1 -- fresh init from RANDOM") !=
          std::string::npos);
    CHECK(journal_status(journal) == restarts_used_status(5, 1));
    CHECK(data_rows("loss.tadah").size() == 23u);
}

/* ======================================================================
 * One record for the run: a point an earlier search of the run evaluated
 * is answered from its evaluation (HPO_Host::recorded), whichever phase
 * asks -- a PRESTAGE refinement, a MULTISTART start, a RESTART round.
 * ==================================================================== */

namespace {

/* The params.tadah rows whose single parameter is `value`. */
std::size_t rows_at(double value)
{
    std::size_t n = 0;
    for (const auto& row : data_rows("params.tadah"))
        if (row.size() == 3 && std::stod(row[2]) == value) ++n;
    return n;
}

}  // namespace

namespace {

/* The root lines of a two-dimensional run, CGRID2B (1) in [0.1, 0.9] and
 * SGRID2B (2) in [4, 6], whose loss is PC_ERMSE with `flags`. Without flags
 * every evaluation is a function of its point
 * (HPOTargets::repeat_reuse_blocker() is ""). */
std::string budget_root(const std::string& flags)
{
    return "LOSS L2\nFAILSCORE 1e9\nQFLUSH 1000\nDIGITS 15\nPC_ERMSE 0 1" +
           flags + "\nOPTIM CGRID2B (1) 0.1 0.9\nOPTIM SGRID2B (2) 4.0 6.0\n";
}

/* GD_MLSL with INNER LD_LBFGS (NLopt's 4 samples per round), SEED 7, and
 * the block's `keys` (a GRAD_STRATEGY line). */
std::string mlsl_budget_block(unsigned maxeval, const std::string& keys = "")
{
    return plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO GD_MLSL\n"
                       "    MAXEVAL " + std::to_string(maxeval) +
                       "\n    SEED 7\n" + keys + "    INNER\n      LIB NLOPT\n"
                       "      ALGO LD_LBFGS\n    ENDINNER\n");
}

/* The points of params.tadah by evaluation number (1-based, at index
 * eval - 1): the columns after the evaluation number and the slot. */
std::vector<std::vector<double>> params_points()
{
    const auto params = data_rows("params.tadah");
    std::vector<std::vector<double>> points(params.size());
    for (const auto& row : params) {
        const std::size_t eval = std::stoull(row.at(0));
        REQUIRE(eval >= 1);
        REQUIRE(eval <= points.size());
        for (std::size_t c = 2; c < row.size(); ++c)
            points[eval - 1].push_back(std::stod(row[c]));
    }
    return points;
}

/* Whether `b` lies within a finite-difference stencil of `a` (their
 * coordinates differ by 1e-4 at most; the stencil's step is about 1e-6 of
 * an OPTIM width, a search's step is far larger), or is `a`. */
bool near_point(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > 1e-4) return false;
    return true;
}

/* The first of the first `before` points near which every point from
 * `from` to `to` (0-based, inclusive) lies: the start of the local search
 * whose stencil those points are. */
std::optional<std::size_t> stencil_centre(
    const std::vector<std::vector<double>>& pts, std::size_t before,
    std::size_t from, std::size_t to)
{
    for (std::size_t k = 0; k < before && k < pts.size(); ++k) {
        bool all = to < pts.size();
        for (std::size_t j = from; all && j <= to; ++j)
            all = near_point(pts[k], pts[j]);
        if (all) return k;
    }
    return std::nullopt;
}

/* The startup WARNING of the MLSL of mlsl_budget_block() at `maxeval`, the
 * loss at its first local search's start taken from the search's record
 * (`recorded`) or evaluated again. */
std::string mlsl_budget_warning(unsigned maxeval, bool recorded)
{
    return "the STRATEGY-level optimiser NLOPT:GD_MLSL: MAXEVAL " +
           std::to_string(maxeval) +
           " takes no step of a local search: MLSL evaluates its starting "
           "point and its first 4 samples without a gradient (5 evaluations; "
           "POPULATION 4, NLopt's default), then its first local search, "
           "from one of them, " +
           (recorded ? std::string("takes the loss there from the run's "
                                   "record (no evaluation) and evaluates the "
                                   "gradient there")
                     : std::string("evaluates the loss there again (1 "
                                   "evaluation) and the gradient there")) +
           " (4 evaluations: central differences in 2 dimensions, "
           "GRAD_STRATEGY NUMERICAL) before it evaluates a point it steps to, "
           "which needs MAXEVAL " + (recorded ? "10" : "11") + " or more";
}

}  // namespace

/* Pins: a PRESTAGE refinement starts from the exploration's best point,
 * which the exploration evaluated: on the run whose loss falls towards
 * SGRID2B (2) = 4, an LN_SBPLX exploration (MAXEVAL 8) finds the bound at
 * eval 4 and hands it to a DLIB:BFGS refinement (TOPK 1, MAXEVAL 30),
 * whose first call, the loss of its start, is answered from the run's
 * record ("DLIB:BFGS asked for the point of eval 4 (SGRID2B[2] = 4), which
 * an earlier search of the run evaluated: the loss of eval 4 is reused,
 * without an evaluation"); the refinement evaluates its gradient's stencil
 * alone ("DLIB:BFGS stopped: FTOL_ABS reached after 2 evaluation(s)"): 10
 * rows, none repeating another.
 * Before the fix: the refinement evaluated its start again, its record
 * being its own: 11 rows, the 9th repeating the 4th, "after 3
 * evaluation(s)". */
TEST_CASE("Run record: a PRESTAGE refinement does not evaluate its start "
          "point again", "[HPO][termination][repeat_points][prestage]")
{
    TaRun run(plain_block("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 30\n    TOPK 1\n"
                          "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                          "      MAXEVAL 8\n    ENDEXPLORE\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "PRESTAGE refinement"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("params.tadah").size() == 10u);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 4 (SGRID2B[2] "
                       "= 4), which an earlier search of the run evaluated: "
                       "the loss of eval 4 is reused, without an evaluation") !=
          std::string::npos);
    CHECK(journal.find("DLIB:BFGS stopped: FTOL_ABS reached after 2 "
                       "evaluation(s)") != std::string::npos);
}

/* Pins: under GRAD_STRATEGY ANALYTICAL a PRESTAGE refinement's start is
 * evaluated (its analytical gradient is that of the host's last
 * evaluation), and that evaluation answers the loss call at the start as
 * well: the run of the case above with a DLIB:BFGS ANALYTICAL refinement
 * and RESTART { AFTER_STAGNATION 2, MAX 1, METHOD CONFIG }. The
 * refinement's start, the exploration's eval 4 at SGRID2B (2) = 4, is
 * evaluated once more (one params row repeats another), counted once in
 * the window, and the refinement takes its gradient and ends on its own
 * stop test ("DLIB:BFGS stopped: FTOL_ABS reached after 2 evaluation(s)"):
 * no StagnationStop in the refinement and no restart; the journal names the
 * start as an earlier search's evaluation evaluated again ("DLIB:BFGS asked
 * for the point of eval 4 (SGRID2B[2] = 4) and its gradient, which an
 * earlier search of the run evaluated; it is evaluated again: ...
 * GRAD_STRATEGY ANALYTICAL ..."), with no reuse line for it.
 * Before the fix: the loss call at the start was counted as a reuse
 * ("DLIB:BFGS asked for the point of eval 4 ..., which an earlier search
 * of the run evaluated: the loss of eval 4 is reused, without an
 * evaluation") and its evaluation (eval 7, after the window had ended the
 * exploration at eval 6) once more, so the window of 2 ended the
 * refinement at that evaluation ("StagnationStop fired at eval 7"), before
 * it took its gradient, and RESTART ran a round ("RESTART 1/1 -- fresh
 * init from CONFIG"), whose refinement stagnated too. */
TEST_CASE("Run record: a PRESTAGE refinement under ANALYTICAL evaluates its "
          "start once and counts it once in the RESTART window",
          "[HPO][termination][repeat_points][prestage][analytical]")
{
    TaRun run(plain_block("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                          "    GRAD_STRATEGY ANALYTICAL\n"
                          "    MAXEVAL 30\n    TOPK 1\n"
                          "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                          "      MAXEVAL 8\n    ENDEXPLORE\n"
                          "    RESTART\n      AFTER_STAGNATION 2\n"
                          "      MAX 1\n      METHOD CONFIG\n"
                          "    ENDRESTART\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(
                        run, "PRESTAGE refinement under ANALYTICAL"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(repeated_rows() == 1u);
    CHECK(rows_at(4.0) == 2u);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 4 (SGRID2B[2] "
                       "= 4) and its gradient, which an earlier search of the "
                       "run evaluated; it is evaluated again: a gradient was "
                       "asked for where this search formed none, and "
                       "GRAD_STRATEGY ANALYTICAL differentiates the last "
                       "evaluation") != std::string::npos);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 4 (SGRID2B[2] "
                       "= 4), which an earlier search of the run evaluated: "
                       "the loss of eval 4 is reused") == std::string::npos);
    CHECK(journal.find("DLIB:BFGS stopped: FTOL_ABS reached after 2 "
                       "evaluation(s)") != std::string::npos);
    CHECK(journal.find("RESTART 1/1") == std::string::npos);
}

/* Pins: MULTISTART starts reach points an earlier start evaluated: two
 * DLIB:BFGS starts (INIT LHS 2, SEED 3; BUDGET_PER_START 30, FTOL_ABS
 * 1e-12) on the run whose loss falls towards SGRID2B (2) = 4 both end at
 * the bound. Start 1 evaluates it (eval 7) and its one-sided stencil;
 * start 2's line-search trial clamped onto 4 is answered from the run's
 * record ("DLIB:BFGS asked for the point of eval 7 (SGRID2B[2] = 4), which
 * an earlier search of the run evaluated"), so the bound is evaluated once
 * (one params row at 4). Start 2 evaluates its own gradient's stencil
 * there (the gradients are each search's: 2 rows repeat start 1's
 * stencil). BUDGET_PER_START charges each start its own evaluations: 9 and
 * 14 "eval(s) consumed", the run's 23 rows.
 * Before the fix: each start evaluated the bound (two rows at 4, 3 rows
 * repeating another, 24 rows). */
TEST_CASE("Run record: a MULTISTART start does not evaluate a point an "
          "earlier start evaluated",
          "[HPO][termination][repeat_points][multistart]")
{
    TaRun run("OPTIMIZER\n"
              "  INIT\n    METHOD LHS 2\n    SEED 3\n  ENDINIT\n"
              "  STRATEGY\n"
              "    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
              "    N_STARTS 2\n    BUDGET_PER_START 30\n"
              "    FTOL_ABS 1e-12\n"
              "  ENDSTRATEGY\n"
              "ENDOPTIMIZER\n",
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "MULTISTART starts"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(rows_at(4.0) == 1u);
    CHECK(repeated_rows() == 2u);
    CHECK(data_rows("loss.tadah").size() == 23u);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 7 (SGRID2B[2] = "
                       "4), which an earlier search of the run evaluated") !=
          std::string::npos);
    CHECK(journal.find("MULTISTART start 1/2 done: 9 eval(s) consumed") !=
          std::string::npos);
    CHECK(journal.find("MULTISTART start 2/2 done: 14 eval(s) consumed") !=
          std::string::npos);
}

/* Pins: a RESTART round that starts where round 0 started takes round 0's
 * path, every point of which the run has: LN_SBPLX on the run whose loss
 * falls towards SGRID2B (2) = 4, with RESTART { AFTER_STAGNATION 5, MAX 1,
 * METHOD CONFIG } as its only stop, stagnates at eval 9 and restarts from
 * the configuration's point; round 1's calls are answered from the record
 * ("NLOPT:LN_SBPLX asked for the point of eval 1 (SGRID2B[2] = 5), which an
 * earlier search of the run evaluated"), so the window ends it after 5
 * calls without an evaluation ("StagnationStop fired at a repeat of eval 5
 * answered from the run's record, after eval 9; ... (threshold=5 evals
 * without improvement, 5 of them call(s) answered from the run's
 * record)"), and the run ends "STOPPED -- stagnation: no new best in 5
 * evals after 1 restart(s)" with round 0's 9 evaluations, none repeated,
 * DIAGNOSTICS "Repeated points reused : 5".
 * Before the fix: round 1 evaluated round 0's first 5 points again: 14
 * rows, 5 of them repeating earlier ones. */
TEST_CASE("Run record: a RESTART round does not evaluate an earlier round's "
          "points again", "[HPO][termination][repeat_points][restart]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
                          "    RESTART\n      AFTER_STAGNATION 5\n"
                          "      MAX 1\n      METHOD CONFIG\n"
                          "    ENDRESTART\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "RESTART from CONFIG"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal_status(journal) == restarts_used_status(5, 1));
    CHECK(journal.find("RESTART 1/1 -- fresh init from CONFIG") !=
          std::string::npos);
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("loss.tadah").size() == 9u);
    CHECK(journal.find("NLOPT:LN_SBPLX asked for the point of eval 1 "
                       "(SGRID2B[2] = 5), which an earlier search of the run "
                       "evaluated") != std::string::npos);
    CHECK(journal.find("StagnationStop fired at a repeat of eval 5 answered "
                       "from the run's record, after eval 9; ") !=
          std::string::npos);
    CHECK(journal.find("(threshold=5 evals without improvement, 5 of them "
                       "call(s) answered from the run's record)") !=
          std::string::npos);
    CHECK(diagnostics_count(journal, "Repeated points reused") == "5");
}

/* Pins: a PRESTAGE under RESTART { AFTER_STAGNATION 3, MAX 1, METHOD
 * CONFIG } whose round-1 exploration (LN_SBPLX, MAXEVAL 6, from the
 * configuration's point) takes round 0's path: its first three calls are
 * answered from the record, and the window ends it there ("StagnationStop
 * fired at a repeat of eval 3 answered from the run's record, after eval
 * 8"); the points those calls stand for are its candidates, so the DLIB:BFGS
 * refinement (TOPK 1) refines one of them, SGRID2B (2) = 4.5 ("DLIB:BFGS
 * asked for the point of eval 3 (SGRID2B[2] = 4.5), which an earlier search
 * of the run evaluated"); both rounds refine ("REFINEMENT: refining top-1"
 * twice), the run ends "STOPPED -- stagnation: no new best in 3 evals after
 * 1 restart(s)" after 10 evaluations, none repeated, and no exploration is
 * an error for asking for no point.
 * Before the fix: round 1's exploration evaluated round 0's points again:
 * 15 rows, 5 of them repeating earlier ones. (With the run's record but
 * without its reused calls as candidates, that exploration would make no
 * evaluation and leave the refinement no candidate.) */
TEST_CASE("Run record: a PRESTAGE exploration answered from the record "
          "still hands its candidates on",
          "[HPO][termination][repeat_points][prestage][restart]")
{
    TaRun run(plain_block("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 40\n    TOPK 1\n"
                          "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                          "      MAXEVAL 6\n    ENDEXPLORE\n"
                          "    RESTART\n      AFTER_STAGNATION 3\n"
                          "      MAX 1\n      METHOD CONFIG\n"
                          "    ENDRESTART\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "PRESTAGE under RESTART"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal.find("RESTART 1/1 -- fresh init from CONFIG") !=
          std::string::npos);
    CHECK(journal.find("StagnationStop fired at a repeat of eval 3 answered "
                       "from the run's record, after eval 8") !=
          std::string::npos);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 3 (SGRID2B[2] = "
                       "4.5), which an earlier search of the run evaluated") !=
          std::string::npos);
    CHECK(count_of(journal, "REFINEMENT: refining top-1") == 2u);
    CHECK(journal.find("asked for no point") == std::string::npos);
    CHECK(repeated_rows() == 0u);
    CHECK(data_rows("loss.tadah").size() == 10u);
    CHECK(journal_status(journal) == restarts_used_status(3, 1));
}

/* Pins: the same run with TOPK 5: round 1's exploration answers its three
 * calls from the record and has 3 candidates, which the journal counts
 * ("PRESTAGE: 3 usable candidate(s) for TOPK 5 (the exploration made 0
 * evaluation(s), 0 without a usable loss, and answered 3 call(s) from the
 * run's record of earlier evaluations); refining 3"), and the refinement
 * refines those 3.
 * Before the fix: round 1's exploration evaluated its three points again
 * ("... (the exploration made 3 evaluation(s), 0 without a usable loss);
 * refining 3"). */
TEST_CASE("Run record: a PRESTAGE exploration's calls answered from the "
          "record are counted among its candidates",
          "[HPO][termination][repeat_points][prestage][restart]")
{
    TaRun run(plain_block("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 40\n    TOPK 5\n"
                          "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                          "      MAXEVAL 6\n    ENDEXPLORE\n"
                          "    RESTART\n      AFTER_STAGNATION 3\n"
                          "      MAX 1\n      METHOD CONFIG\n"
                          "    ENDRESTART\n"),
              kLowerBoundRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "PRESTAGE TOPK 5"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal.find("PRESTAGE: 3 usable candidate(s) for TOPK 5 (the "
                       "exploration made 0 evaluation(s), 0 without a usable "
                       "loss, and answered 3 call(s) from the run's record of "
                       "earlier evaluations); refining 3") !=
          std::string::npos);
    CHECK(count_of(journal, "REFINEMENT: refining top-3") == 1u);
    CHECK(journal_status(journal) == restarts_used_status(3, 1));
}

/* Pins: on the real host (hpo_run, the Ta fixture in 2-D), the startup
 * WARNING of an MLSL whose budget lets no local search step is given
 * exactly when the run takes no step, with the run's own reuse of its
 * local search's start (validate_hpo_setup passes
 * HPOTargets::repeat_reuse_blocker() to gradient_budget_warnings; the
 * NLopt objective asks HPO_Host::repeat_reuse_blocker(), which is the same):
 *   - PC_ERMSE alone (every evaluation a function of its point), MAXEVAL 9:
 *     the WARNING "... then its first local search, from one of them, takes
 *     the loss there from the search's record (no evaluation) and evaluates
 *     the gradient there (4 evaluations: ...) ... which needs MAXEVAL 10 or
 *     more", once; 9 rows: the start, 4 samples, and the 4-point stencil of
 *     one of those five (no row repeated, "Repeated points reused : 1"):
 *     no step;
 *   - the same at MAXEVAL 10: no WARNING; row 10 is a step, far from every
 *     earlier point;
 *   - with PC_ERMSE --skip-rel 1e6, which compares with the best value so
 *     far, so every repeat is evaluated again, MAXEVAL 10: the WARNING "...
 *     evaluates the loss there again (1 evaluation) and the gradient there
 *     ... which needs MAXEVAL 11 or more", once; 10 rows, row 6 repeating
 *     one of rows 1-5 and rows 7-10 its stencil: no step.
 * Before the fix: the MAXEVAL 9 leg fails (its WARNING said "evaluates the
 * loss there again (1 evaluation)" and "needs MAXEVAL 11 or more"), and
 * the MAXEVAL 10 leg fails (that WARNING was given although row 10 is a
 * step); the --skip-rel leg passes as well (a pin). */
TEST_CASE("Gradient budget: an MLSL's startup warning follows the run's "
          "reuse of its local search's start",
          "[HPO][termination][repeat_points][gradient_budget]")
{
    const std::string kNoStep = "takes no step";
    SECTION("reused, MAXEVAL 9: warned, no step") {
        TaRun run(mlsl_budget_block(9), budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, mlsl_budget_warning(9, true)) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        const auto pts = params_points();
        REQUIRE(pts.size() == 9u);
        CHECK(repeated_rows() == 0u);
        CHECK(diagnostics_count(journal, "Repeated points reused") == "1");
        CHECK(stencil_centre(pts, 5, 5, 8).has_value());
    }
    SECTION("reused, MAXEVAL 10: not warned, a step") {
        TaRun run(mlsl_budget_block(10), budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, kNoStep) == 0u);
        const auto pts = params_points();
        REQUIRE(pts.size() == 10u);
        CHECK(repeated_rows() == 0u);
        CHECK(stencil_centre(pts, 5, 5, 8).has_value());
        bool far = true;
        for (std::size_t j = 0; j < 9; ++j)
            far = far && !near_point(pts[j], pts[9]);
        CHECK(far);
    }
    SECTION("evaluated again (--skip-rel), MAXEVAL 10: warned, no step") {
        TaRun run(mlsl_budget_block(10), budget_root(" --skip-rel 1e6"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, mlsl_budget_warning(10, false)) == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        const auto pts = params_points();
        REQUIRE(pts.size() == 10u);
        CHECK(repeated_rows() == 1u);
        const auto k = stencil_centre(pts, 5, 5, 9);
        REQUIRE(k.has_value());
        const auto rows = data_rows("params.tadah");
        CHECK(std::vector<std::string>(rows[5].begin() + 2, rows[5].end()) ==
              std::vector<std::string>(rows[*k].begin() + 2, rows[*k].end()));
    }
}

/* Pins: on the real host (hpo_run, the Ta fixture in 2-D), an MLSL under
 * GRAD_STRATEGY ANALYTICAL evaluates its first local search's start again
 * whatever the run's reuse -- an analytical gradient differentiates the last
 * evaluation, so it follows an evaluation of its point -- and the analytical
 * gradient there costs 1 evaluation, so GD_MLSL with INNER LD_LBFGS needs
 * MAXEVAL 8 for a local step (1 + 4 + 1 + 1 + 1): N + 3 or less, 7, lets
 * none step, N its 4 samples (under NUMERICAL central differences 2n + N +
 * 1 or less, 9, with the loss there from the record, the case above, and
 * 2n + N + 2 or less, 10, with it evaluated again); the startup WARNING is
 * given exactly when the run takes no step:
 *   - ANALYTICAL, MAXEVAL 7: the WARNING "... then its first local search,
 *     from one of them, evaluates the loss there again (1 evaluation) and
 *     the gradient there (1 evaluation: one analytical gradient,
 *     GRAD_STRATEGY ANALYTICAL) before it evaluates a point it steps to,
 *     which needs MAXEVAL 8 or more", once; 6 rows: the start, 4 samples
 *     and row 6 repeating one of rows 1-5 exactly ("Repeats evaluated
 *     again : 1"): no step;
 *   - ANALYTICAL, MAXEVAL 8: no WARNING; 7 rows, row 6 repeating one of rows
 *     1-5 and row 7 a step, far from every earlier point;
 *   - HYBRID, MAXEVAL 8: the analytical gradient covers both OPTIM
 *     parameters here (OALGO 4; the journal's "Gradient-pipeline diagnostic
 *     (first eval): GRAD_STRATEGY=HYBRID | total_theta=2 | analytical=2 |
 *     FD-fallback=0"), so HYBRID differences none and takes a step at row 7,
 *     with no WARNING: HYBRID needs ANALYTICAL's budget and more for each
 *     parameter it differences.
 * Before the fix: passes as well (a pin: the warning and the run agreed;
 * the docs fixed with it -- the Tadah.MLIP CHANGELOG and the MAXEVAL row of
 * the Configuration File -- gave ANALYTICAL and HYBRID the NUMERICAL bound,
 * 2n + N + 2: MAXEVAL 10 or less lets no local search step). */
TEST_CASE("Gradient budget: an MLSL's startup warning under GRAD_STRATEGY "
          "ANALYTICAL counts its local search's start evaluated again and one "
          "analytical gradient",
          "[HPO][termination][repeat_points][gradient_budget]")
{
    const std::string kNoStep = "takes no step";
    /* the row of rows 1-5 that row 6 repeats exactly, -1 for none */
    const auto start_again = [] {
        const auto rows = data_rows("params.tadah");
        long k = -1;
        for (std::size_t j = 0; j < 5 && rows.size() > 5; ++j)
            if (std::vector<std::string>(rows[5].begin() + 2, rows[5].end()) ==
                std::vector<std::string>(rows[j].begin() + 2, rows[j].end()))
                k = static_cast<long>(j);
        return k;
    };
    SECTION("ANALYTICAL, MAXEVAL 7: warned, no step") {
        TaRun run(mlsl_budget_block(7, "    GRAD_STRATEGY ANALYTICAL\n"),
                  budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_MLSL: MAXEVAL 7 "
                       "takes no step of a local search: MLSL evaluates its "
                       "starting point and its first 4 samples without a "
                       "gradient (5 evaluations; POPULATION 4, NLopt's "
                       "default), then its first local search, from one of "
                       "them, evaluates the loss there again (1 evaluation) "
                       "and the gradient there (1 evaluation: one analytical "
                       "gradient, GRAD_STRATEGY ANALYTICAL) before it "
                       "evaluates a point it steps to, which needs MAXEVAL 8 "
                       "or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(params_points().size() == 6u);
        CHECK(repeated_rows() == 1u);
        CHECK(start_again() >= 0);
        CHECK(diagnostics_count(journal, "Repeats evaluated again") == "1");
    }
    for (const char* strategy : {"ANALYTICAL", "HYBRID"}) {
        DYNAMIC_SECTION(strategy << ", MAXEVAL 8: not warned, a step") {
            TaRun run(mlsl_budget_block(8, std::string("    GRAD_STRATEGY ") +
                                               strategy + "\n"),
                      budget_root(""));
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            CHECK(status == 0);
            check_run_files();
            const std::string journal = read_file("hpo_run.log");
            INFO("journal:\n" << journal);
            CHECK(count_of(journal, kNoStep) == 0u);
            /* the analytical gradient covers both parameters: HYBRID
             * differences none */
            CHECK(count_of(journal, std::string("GRAD_STRATEGY=") + strategy +
                                        " | total_theta=2 | analytical=2 | "
                                        "FD-fallback=0") == 1u);
            const auto pts = params_points();
            REQUIRE(pts.size() == 7u);
            CHECK(repeated_rows() == 1u);
            CHECK(start_again() >= 0);
            bool far = true;
            for (std::size_t j = 0; j < 6; ++j)
                far = far && !near_point(pts[j], pts[6]);
            CHECK(far);
        }
    }
}

/* Pins: on the real host (hpo_run, the Ta fixture in 2-D, PC_ERMSE alone),
 * the startup WARNING of a gradient search whose start is a point the run
 * has evaluated is given exactly when the run takes no step: the run's
 * record answers the start's loss, the centre of the finite differences
 * (NUMERICAL), so the start costs its 4-point stencil alone:
 *   - a PRESTAGE refinement (after a GN_DIRECT exploration of 10
 *     evaluations, TOPK 1), NLOPT:LD_LBFGS or DLIB:BFGS at MAXEVAL 5: no
 *     WARNING; rows 11-14 are the stencil of an exploration point (the
 *     candidate, not evaluated again) and row 15 is a step, near none of
 *     rows 1-14;
 *   - NLOPT:LD_LBFGS at MAXEVAL 4: the WARNING "the PRESTAGE refinement
 *     NLOPT:LD_LBFGS: MAXEVAL 4 takes no step: its gradient search takes the
 *     loss at its starting point from the run's record (no evaluation) ...,
 *     which needs MAXEVAL 5 or more", and 14 rows, the last 4 the stencil;
 *   - a PLAIN NLOPT:LD_LBFGS with NOISE CALIBRATION 2 at MAXEVAL 5, whose
 *     start is the seed point the calibration evaluated twice (rows 1-2):
 *     no WARNING, rows 3-6 the stencil of the seed and row 7 a step.
 * Before the fix: the warning counted each start as evaluated again (the
 * run's record is new to this branch; v1-dev's was each search's), so the
 * MAXEVAL 5 runs, which step, were warned that they take "no step ...,
 * which needs MAXEVAL 6 or more", and the MAXEVAL 4 text was "evaluates the
 * loss at a starting point (1 evaluation) ... needs MAXEVAL 6". */
TEST_CASE("Run record: the startup budget warning of a search that starts "
          "at a point the run evaluated follows the run",
          "[HPO][termination][repeat_points][gradient_budget]")
{
    const std::string kNoStep = "takes no step";
    const auto prestage = [](const std::string& lib, const std::string& algo,
                             unsigned maxeval) {
        return plain_block("    TYPE PRESTAGE\n    LIB " + lib + "\n    ALGO " +
                           algo + "\n    MAXEVAL " + std::to_string(maxeval) +
                           "\n    TOPK 1\n    EXPLORE\n      LIB NLOPT\n"
                           "      ALGO GN_DIRECT\n      MAXEVAL 10\n"
                           "    ENDEXPLORE\n");
    };
    /* Whether row `step` (0-based) is near none of the rows before it, and
     * rows `from` to `step - 1` are the stencil of one of the first `from`
     * rows. */
    const auto stencil_then_step = [](std::size_t from, std::size_t step) {
        const auto pts = params_points();
        REQUIRE(pts.size() == step + 1);
        CHECK(stencil_centre(pts, from, from, step - 1).has_value());
        bool far = true;
        for (std::size_t j = 0; j < step; ++j)
            far = far && !near_point(pts[j], pts[step]);
        CHECK(far);
    };
    for (const auto& [lib, algo] :
         std::vector<std::pair<std::string, std::string>>{
             {"NLOPT", "LD_LBFGS"}, {"DLIB", "BFGS"}}) {
        DYNAMIC_SECTION("the PRESTAGE refinement " << lib << ":" << algo
                                                   << ", MAXEVAL 5") {
            TaRun run(prestage(lib, algo, 5), budget_root(""));
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            CHECK(status == 0);
            check_run_files();
            const std::string journal = read_file("hpo_run.log");
            INFO("journal:\n" << journal);
            CHECK(count_of(journal, kNoStep) == 0u);
            CHECK(repeated_rows() == 0u);
            stencil_then_step(10, 14);
        }
    }
    SECTION("the PRESTAGE refinement NLOPT:LD_LBFGS, MAXEVAL 4") {
        TaRun run(prestage("NLOPT", "LD_LBFGS", 4), budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal,
                       "the PRESTAGE refinement NLOPT:LD_LBFGS: MAXEVAL 4 "
                       "takes no step: its gradient search takes the loss at "
                       "its starting point from the run's record (no "
                       "evaluation) and evaluates the gradient there (4 "
                       "evaluations: central differences in 2 dimensions, "
                       "GRAD_STRATEGY NUMERICAL) before it evaluates a point "
                       "it steps to, which needs MAXEVAL 5 or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        const auto pts = params_points();
        REQUIRE(pts.size() == 14u);
        CHECK(stencil_centre(pts, 10, 10, 13).has_value());
    }
    SECTION("NOISE CALIBRATION 2, PLAIN NLOPT:LD_LBFGS, MAXEVAL 5") {
        TaRun run("OPTIMIZER\n  NOISE\n    CALIBRATION 2\n  ENDNOISE\n"
                  "  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
                  "    ALGO LD_LBFGS\n    MAXEVAL 5\n  ENDSTRATEGY\n"
                  "ENDOPTIMIZER\n",
                  budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, kNoStep) == 0u);
        CHECK(repeated_rows() == 1u);
        stencil_then_step(1, 6);
    }
}

/* Pins: on the real host (hpo_run, the Ta fixture in 2-D, PC_ERMSE alone),
 * the startup WARNING follows two starts the warning took for evaluated
 * ones or not:
 *   - NLopt's StoGO does not start at the point it is given: its first
 *     trial is the centre of its box (stogo/global.cc), here (0.5, 5.5)
 *     with SGRID2B (2) in [4, 7]. A PLAIN NLOPT:GD_STOGO under NOISE
 *     CALIBRATION 2, whose calibration evaluates the configuration's point
 *     (0.5, 5) twice (rows 1-2), evaluates that centre (row 3, at no
 *     earlier row's point) and its 4-point stencil (rows 4-7) before any
 *     step, not the seed point: at MAXEVAL 5 the WARNING "the
 *     STRATEGY-level optimiser NLOPT:GD_STOGO: MAXEVAL 5 takes no step: its
 *     gradient search, which does not start at the point it is given,
 *     evaluates the loss at a point of its own (1 evaluation) ..., which
 *     needs MAXEVAL 6 or more", once, and 7 rows; at MAXEVAL 6 none, and
 *     row 8 is a step, near none of rows 1-7;
 *   - a PRESTAGE refinement NLOPT:GD_STOGO (TOPK 1, after an LN_SBPLX
 *     exploration) is refused at startup: hpo_run throws "Error on line
 *     ...: STRATEGY TYPE PRESTAGE cannot refine with NLOPT:GD_STOGO: it
 *     does not start at the point it is given ...", the journal records it
 *     ("startup: Error on line ...", its first line) and nothing is
 *     evaluated (no loss.tadah);
 *   - under NOISE CALIBRATION 2 the first MULTISTART start (LD_LBFGS, INIT
 *     LHS 2, SEED 3, N_STARTS 2) begins at the seed point the calibration
 *     evaluated (rows 1-2) and takes its loss from the run's record: at
 *     BUDGET_PER_START 5 it evaluates the stencil (rows 3-6) and steps (row
 *     7, "MULTISTART start 1/2 done: 5 eval(s) consumed"), and the second
 *     start evaluates its point and stencil (rows 8-12) and takes no step:
 *     one WARNING, "each MULTISTART start after the first NLOPT:LD_LBFGS:
 *     BUDGET_PER_START 5 takes no step: its gradient search evaluates the
 *     loss at a starting point (1 evaluation) ..., which needs
 *     BUDGET_PER_START 6 or more", and none for the first start.
 * Until the refinement on StoGO was refused the StoGO legs ran it as that
 * refinement, after an LN_SBPLX
 * exploration of 4 evaluations from the configuration's point (rows 1-4;
 * the centre row 5, its stencil rows 6-9; MAXEVAL 5 warned "the PRESTAGE
 * refinement NLOPT:GD_STOGO: MAXEVAL 5 ...", 9 rows; MAXEVAL 6 a step at
 * row 10); the PLAIN legs under NOISE CALIBRATION check the same warning
 * rule, and pass before the refusal as well (pins); the refusal leg fails
 * before it (hpo_run ran the refinement, status 0).
 * Before the fix of the warning: the StoGO refinement at MAXEVAL 5 was not
 * warned (the warning took the loss at its start from the run's record,
 * needing 5; a run before the fix: no warning, the box
 * centre and its stencil, no step), and the MULTISTART WARNING was "each
 * MULTISTART start NLOPT:LD_LBFGS: BUDGET_PER_START 5 ...", which claimed
 * the first start, which steps. */
TEST_CASE("Run record: the startup budget warning of NLopt's StoGO and of "
          "the MULTISTART starts under NOISE CALIBRATION follows the run",
          "[HPO][termination][repeat_points][gradient_budget][stogo]")
{
    const std::string kNoStep = "takes no step";
    /* SGRID2B (2) in [4, 7]: the box centre (0.5, 5.5) is not the
     * configuration's point (0.5, 5), the exploration's start. */
    const std::string root =
        "LOSS L2\nFAILSCORE 1e9\nQFLUSH 1000\nDIGITS 15\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.1 0.9\nOPTIM SGRID2B (2) 4.0 7.0\n";
    const auto stogo = [&](unsigned maxeval) {
        return "OPTIMIZER\n  NOISE\n    CALIBRATION 2\n  ENDNOISE\n"
               "  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
               "    ALGO GD_STOGO\n    MAXEVAL " +
               std::to_string(maxeval) + "\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    /* Rows 1-2 are the calibration's, at the configuration's point; the
     * search's first row is the box centre, at no earlier row's point, and
     * the next 4 rows its stencil. */
    const auto centre_first = [](const std::vector<std::vector<double>>& pts) {
        REQUIRE(pts.size() >= 7u);
        for (std::size_t j = 0; j < 2; ++j) {
            CHECK(std::fabs(pts[j][0] - 0.5) < 1e-12);
            CHECK(std::fabs(pts[j][1] - 5.0) < 1e-12);
        }
        CHECK(std::fabs(pts[2][0] - 0.5) < 1e-12);
        CHECK(std::fabs(pts[2][1] - 5.5) < 1e-12);
        for (std::size_t j = 0; j < 2; ++j)
            CHECK_FALSE(near_point(pts[j], pts[2]));
        CHECK(stencil_centre(pts, 3, 3, 6) == std::optional<std::size_t>(2));
    };
    SECTION("NOISE CALIBRATION 2, PLAIN NLOPT:GD_STOGO, MAXEVAL 5: warned, "
            "no step") {
        TaRun run(stogo(5), root);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal,
                       "the STRATEGY-level optimiser NLOPT:GD_STOGO: MAXEVAL 5 "
                       "takes no step: its gradient search, which does not "
                       "start at the point it is given, evaluates the loss at "
                       "a point of its own (1 evaluation) and the gradient "
                       "there (4 evaluations: central differences in 2 "
                       "dimensions, GRAD_STRATEGY NUMERICAL) before it "
                       "evaluates a point it steps to, which needs MAXEVAL 6 "
                       "or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        const auto pts = params_points();
        CHECK(pts.size() == 7u);
        centre_first(pts);
    }
    SECTION("NOISE CALIBRATION 2, PLAIN NLOPT:GD_STOGO, MAXEVAL 6: not "
            "warned, a step") {
        TaRun run(stogo(6), root);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, kNoStep) == 0u);
        const auto pts = params_points();
        REQUIRE(pts.size() == 8u);
        centre_first(pts);
        bool far = true;
        for (std::size_t j = 0; j < 7; ++j)
            far = far && !near_point(pts[j], pts[7]);
        CHECK(far);
    }
    SECTION("the PRESTAGE refinement NLOPT:GD_STOGO is refused at startup") {
        TaRun run(plain_block("    TYPE PRESTAGE\n    LIB NLOPT\n"
                              "    ALGO GD_STOGO\n    MAXEVAL 5\n"
                              "    TOPK 1\n    EXPLORE\n      LIB NLOPT\n"
                              "      ALGO LN_SBPLX\n      MAXEVAL 4\n"
                              "    ENDEXPLORE\n"),
                  root);
        std::string what;
        try {
            (void)tadah::hpo::hpo_run(run.ctx);
        } catch (const std::exception& e) {
            what = e.what();
        }
        INFO("refusal: " << what);
        CHECK_THAT(what, Catch::StartsWith("Error on line ") &&
                             Catch::Contains(
                                 "STRATEGY TYPE PRESTAGE cannot refine with "
                                 "NLOPT:GD_STOGO: it does not start at the "
                                 "point it is given") &&
                             Catch::EndsWith("\nLine: ENDOPTIMIZER"));
        /* The journal writes each line of a refusal as a line of its own. */
        CHECK(read_file("hpo_run.log")
                  .find("startup: " + what.substr(0, what.find('\n'))) !=
              std::string::npos);
        CHECK_FALSE(fs::exists("loss.tadah"));
    }
    SECTION("NOISE CALIBRATION 2, MULTISTART NLOPT:LD_LBFGS, "
            "BUDGET_PER_START 5: the later start warned") {
        TaRun run("OPTIMIZER\n  NOISE\n    CALIBRATION 2\n  ENDNOISE\n"
                  "  INIT\n    METHOD LHS 2\n    SEED 3\n  ENDINIT\n"
                  "  STRATEGY\n    TYPE MULTISTART\n    LIB NLOPT\n"
                  "    ALGO LD_LBFGS\n    N_STARTS 2\n"
                  "    BUDGET_PER_START 5\n  ENDSTRATEGY\nENDOPTIMIZER\n",
                  budget_root(""));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal,
                       "each MULTISTART start after the first NLOPT:LD_LBFGS: "
                       "BUDGET_PER_START 5 takes no step: its gradient search "
                       "evaluates the loss at a starting point (1 evaluation) "
                       "and the gradient there (4 evaluations: central "
                       "differences in 2 dimensions, GRAD_STRATEGY NUMERICAL) "
                       "before it evaluates a point it steps to, which needs "
                       "BUDGET_PER_START 6 or more") == 1u);
        CHECK(count_of(journal, kNoStep) == 1u);
        CHECK(journal.find("MULTISTART start 1/2 done: 5 eval(s) consumed") !=
              std::string::npos);
        const auto pts = params_points();
        REQUIRE(pts.size() == 12u);
        /* start 1: the calibrated seed (rows 1-2), its stencil, a step */
        CHECK(pts[0] == pts[1]);
        CHECK(stencil_centre(pts, 1, 2, 5) == std::optional<std::size_t>(0));
        bool far = true;
        for (std::size_t j = 0; j < 6; ++j)
            far = far && !near_point(pts[j], pts[6]);
        CHECK(far);
        /* start 2: its point and its stencil, no step */
        CHECK(stencil_centre(pts, 8, 8, 11) == std::optional<std::size_t>(7));
    }
}

/* Pins: where every repeat is evaluated again (a --skip-rel filter compares
 * with the best so far), the run's record still knows the earlier search's
 * evaluation, and the journal names it as such: a PRESTAGE refinement
 * (DLIB:BFGS, TOPK 1) after an LN_SBPLX exploration (MAXEVAL 8) on the run
 * whose loss falls towards SGRID2B (2) = 4 evaluates its start, the
 * exploration's eval 4, again ("DLIB:BFGS asked for the point of eval 4
 * (SGRID2B[2] = 4), which an earlier search of the run evaluated; it is
 * evaluated again: PC_ERMSE --skip-rel: ..."), one params row repeating
 * another, DIAGNOSTICS "Repeats evaluated again : 1".
 * Before the fix: the line said "DLIB:BFGS asked again for the point of
 * eval 4 (SGRID2B[2] = 4); it is evaluated again: ...", though DLIB:BFGS
 * had not asked for it before. */
TEST_CASE("Run record: under a reuse blocker an earlier search's point is "
          "named as such when it is evaluated again",
          "[HPO][termination][repeat_points][prestage]")
{
    std::string root = kLowerBoundRoot;
    const std::string loss = "PC_ERMSE 0 1\n";
    REQUIRE(root.find(loss) != std::string::npos);
    root.replace(root.find(loss), loss.size(), "PC_ERMSE 0 1 --skip-rel 5\n");
    TaRun run(plain_block("    TYPE PRESTAGE\n    LIB DLIB\n    ALGO BFGS\n"
                          "    MAXEVAL 30\n    TOPK 1\n"
                          "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                          "      MAXEVAL 8\n    ENDEXPLORE\n"),
              root);
    int status = -1;
    REQUIRE_NOTHROW(status = repeat_run_within(run, "blocked PRESTAGE"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal.find("DLIB:BFGS asked for the point of eval 4 (SGRID2B[2] "
                       "= 4), which an earlier search of the run evaluated; "
                       "it is evaluated again: PC_ERMSE --skip-rel: the "
                       "filter compares with the best") != std::string::npos);
    CHECK(journal.find("asked again for the point of eval 4") ==
          std::string::npos);
    CHECK(repeated_rows() == 1u);
    CHECK(diagnostics_count(journal, "Repeats evaluated again") == "1");
}

TEST_CASE("Two runs in one process: each journal carries its own "
          "once-per-run notes", "[HPO][termination][two_runs]")
{
    /* The notes a run prints once (the DLIB:BFGS MAXEVAL preamble, the
     * gradient-pipeline diagnostic, the no-analytical-gradient fallback)
     * were function statics: the second run of an embedding, or of a test
     * process, printed none of them, and the rate-limited counts (a
     * pc-worker error's occurrence, the FAILPROBE "total gated=N") went
     * on from the first run's. The second run here has only hpo_run()'s
     * own per-run resets. */
    for (const char* oalgo : {"4", "1"}) {
        CAPTURE(oalgo);
        const std::string optimizer = plain_block(
            "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 9\n"
            "    GRAD_STRATEGY HYBRID\n");
        std::vector<std::string> journals;
        for (int run = 0; run < 2; ++run) {
            TaRun r(optimizer, kRootLines, /*new_process=*/run == 0);
            r.ctx.remove("OALGO");
            r.ctx.add("OALGO", std::stoi(oalgo));
            REQUIRE(tadah::hpo::hpo_run(r.ctx) == 0);
            journals.push_back(read_file("hpo_run.log"));
        }
        /* OALGO 4: the analytical gradient, and BFGS reaches MAXEVAL;
         * OALGO 1: finite differences (the fallback note), and BFGS
         * converges before MAXEVAL */
        for (const std::string& j : journals) {
            INFO("journal:\n" << j);
            if (std::string(oalgo) == "4") {
                CHECK(j.find("DLIB:BFGS -- MAXEVAL is enforced externally") !=
                      std::string::npos);
                CHECK(j.find("Gradient-pipeline diagnostic (first eval)") !=
                      std::string::npos);
            } else {
                CHECK(j.find("No analytical gradient for this setup") !=
                      std::string::npos);
            }
        }
    }
}

namespace {

/* The value of the journal line `label : value` (its first occurrence),
 * "" without the line. */
std::string journal_value(const std::string& journal, const std::string& label)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex("\n  " + label + R"(\s*: ([^\n]*))")))
        return m[1];
    return "";
}

/* The N of "(eval N" in the value of the journal line `label`, -1 without
 * it. */
long journal_eval(const std::string& journal, const std::string& label)
{
    std::smatch m;
    const std::string v = journal_value(journal, label);
    if (std::regex_search(v, m, std::regex(R"(\(eval (\d+))")))
        return std::stol(m[1]);
    return -1;
}

/* A PC_LAMMPS script whose --hotvar a0 it leaves as it was given, read back
 * as --outvar A, and whose E is the energy per atom. */
void write_hot_script(const fs::path& path, const std::string& a0_line = "")
{
    std::ofstream(path)
        << "clear\nunits metal\nboundary p p p\natom_style atomic\n"
           "lattice bcc 3.30\nregion box block 0 2 0 2 0 2 units lattice\n"
           "create_box 1 box\ncreate_atoms 1 region box\nmass 1 180.95\n"
           "pair_style tadah\npair_coeff * * EXTERN Ta\n"
           "thermo_style custom step pe\nrun 0 post no\n"
           "variable E equal pe/atoms\n" << a0_line
        << "variable A equal v_a0\n";
}

}  // namespace

/* Pins: the journal of a second run in one process (TADAH:RANDOM, SEED 3)
 * counts, times and reports that run alone, as the same run made in a
 * process of its own does, though its evaluations are numbered on from the
 * first run's (3 evaluations, then the second run's 4, ...):
 *   - a second run whose every evaluation fails (MAXEVAL 4, a PC_ERMSE skip
 *     cap of 1e-30): "Total evaluations : 4", "Unusable losses : 4", "[!]
 *     No evaluation produced a usable loss.", "Best total loss : (none --
 *     no evaluation produced a usable loss)", no "Best ERMSE", its fastest
 *     (and slowest, when printed) evaluation among its own (evals 4 to 7),
 *     as when it is run alone (evals 1 to 4);
 *   - a second run with a worse best than the first (PC_ERMSE weight 2,
 *     MAXEVAL 2): "Total evaluations : 2", its own BEST lines, and "Best
 *     total loss" its best loss at its evaluation (4 or 5), the
 *     HPO_BEST_LOSS of its best_pot.tadah, as when run alone.
 * Before the fix: HPOLog::open reset only the status, the start tally, the
 * EMBF0, density and occupancy reports, the notes and the FD counters, so
 * the second journal counted both runs ("Total evaluations : 7",
 * "Unusable losses : 4 (57.1%)", "[!] The majority of evaluations
 * failed"), timed the first run's evaluations, and reported the first
 * run's best ("Best total loss : ... (eval 2)", "Best ERMSE"); with a worse
 * second run it had no BEST line, as no loss was below the first run's
 * best. */
TEST_CASE("Two runs in one process: the second journal counts, times and "
          "reports its own run", "[HPO][termination][two_runs]")
{
    const auto random = [](int maxeval) {
        return plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                           "    MAXEVAL " + std::to_string(maxeval) +
                           "\n    SEED 3\n");
    };
    const auto root_with = [](const std::string& ermse_line) {
        std::string root = kRootLines;
        const std::string from = "PC_ERMSE 0 1\n";
        root.replace(root.find(from), from.size(), ermse_line);
        return root;
    };
    struct Second {
        std::string name, root;
        int maxeval;
        int status;
    };
    for (const Second& second :
         {Second{"no usable loss", root_with("PC_ERMSE 0 1 --skip-above 1e-30\n"),
                 4, 1},
          Second{"a worse best", root_with("PC_ERMSE 0 2\n"), 2, 0}}) {
        DYNAMIC_SECTION("a second run with " << second.name) {
            std::string alone;
            {
                TaRun run(random(second.maxeval), second.root);
                REQUIRE(tadah::hpo::hpo_run(run.ctx) == second.status);
                alone = read_file("hpo_run.log");
            }
            {
                TaRun first(random(3));
                REQUIRE(tadah::hpo::hpo_run(first.ctx) == 0);
            }
            TaRun run(random(second.maxeval), second.root,
                      /*new_process=*/false);
            REQUIRE(tadah::hpo::hpo_run(run.ctx) == second.status);
            const std::string journal = read_file("hpo_run.log");
            INFO("second journal:\n" << journal);
            INFO("the same run alone:\n" << alone);
            const long evals = second.maxeval;
            CHECK(journal_value(journal, "Total evaluations") ==
                  std::to_string(evals));
            CHECK(journal_value(journal, "Total evaluations") ==
                  journal_value(alone, "Total evaluations"));
            CHECK(journal_value(journal, "Unusable losses") ==
                  journal_value(alone, "Unusable losses"));
            CHECK(journal_value(journal, "NaN/Inf clampings") ==
                  journal_value(alone, "NaN/Inf clampings"));
            /* (The slowest is printed only when it is slower than the
             * fastest.) */
            REQUIRE_FALSE(journal_value(journal, "Fastest eval").empty());
            for (const char* label : {"Fastest eval", "Slowest eval"}) {
                if (journal_value(journal, label).empty()) continue;
                INFO(label);
                const long e = journal_eval(journal, label);
                CHECK(e >= 4);
                CHECK(e <= 3 + evals);
            }
            CHECK(journal_value(journal, "Optimiser status") ==
                  journal_value(alone, "Optimiser status"));
            if (second.status == 1) {
                CHECK_THAT(journal, Catch::Contains("[!] No evaluation "
                                                    "produced a usable loss."));
                CHECK(journal_value(journal, "Best total loss") ==
                      "(none -- no evaluation produced a usable loss)");
                CHECK(journal_value(journal, "Best ERMSE").empty());
                CHECK_FALSE(fs::exists("best_pot.tadah"));
            } else {
                CHECK_FALSE(lines_with(journal, "BEST     * eval=").empty());
                const long best_eval = journal_eval(journal, "Best total loss");
                CHECK(best_eval == journal_eval(alone, "Best total loss") + 3);
                const std::string best = journal_value(journal,
                                                       "Best total loss");
                CHECK(best.substr(0, best.find(' ')) ==
                      journal_value(alone, "Best total loss")
                          .substr(0, best.find(' ')));
                CHECK(journal_value(journal, "Best ERMSE") ==
                      journal_value(alone, "Best ERMSE"));
                REQUIRE(fs::exists("best_pot.tadah"));
                std::smatch m;
                const std::string pot = read_file("best_pot.tadah");
                REQUIRE(std::regex_search(pot, m, std::regex(
                            R"(HPO_SOURCE_EVAL\s+(\d+))")));
                CHECK(std::stol(m[1]) == best_eval);
            }
        }
    }
}

/* Pins: the rate limit of the hot-variable "rejecting" WARNING counts the
 * run's own rejects: a first run whose 5 evaluations each reject the
 * --hotvar a0 (the script sets it to 5, outside [0, 1]), then a second run
 * in the process (MAXEVAL 1) whose evaluation rejects it too: the second
 * journal has its "rejecting (... total rejects=1)" WARNING.
 * Before the fix: the counter (warmstart::reject_count) was the process's,
 * at 6 for the second run's reject, which the <= 5-or-a-power-of-2 limit
 * dropped: the second journal had no rejecting WARNING. */
TEST_CASE("Two runs in one process: the second run's hot-variable rejects "
          "are counted from 1", "[HPO][termination][two_runs]")
{
    PcIsolationEnvUnset no_env;
    const auto random = [](int maxeval) {
        return plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                           "    MAXEVAL " + std::to_string(maxeval) +
                           "\n    SEED 3\n");
    };
    const std::string root = kRootLines +
        "PC_LAMMPS --script hot.in --varloss E 0 1 --outvar A "
        "--hotvar a0 0 1 0.5\nPC_ISOLATION THREAD\n";
    {
        TaRun first(random(5), root);
        write_hot_script(first.sd.dir / "hot.in", "variable a0 equal 5.0\n");
        REQUIRE(tadah::hpo::hpo_run(first.ctx) == 1);
        const std::string journal = read_file("hpo_run.log");
        INFO("first journal:\n" << journal);
        CHECK(lines_with(journal, ": rejecting (").size() == 5u);
    }
    TaRun run(random(1), root, /*new_process=*/false);
    write_hot_script(run.sd.dir / "hot.in", "variable a0 equal 5.0\n");
    REQUIRE(tadah::hpo::hpo_run(run.ctx) == 1);
    const std::string journal = read_file("hpo_run.log");
    INFO("second journal:\n" << journal);
    const auto rejects = lines_with(journal, ": rejecting (");
    REQUIRE(rejects.size() == 1u);
    CHECK_THAT(rejects[0], Catch::Contains("total rejects=1)"));
}

/* Pins: a run's --hotvar values start from their declared initial value
 * or from its own WARM seed, never from another run's WARM seed: three
 * runs in one process with a PC_LAMMPS script that never sets its --hotvar
 * a0 (0 100 5.0) and reads it back as --outvar A -- a run that writes a
 * best_pot.tadah, a run that WARM-starts from that potential with "# HOT
 * <script> a0 42.0" prepended (its A is 42), then a run without WARM: its
 * A is 5, the declared initial value, as in a process of its own.
 * Before the fix: the third run's A was 42 -- warmstart::reset_for_fresh_run
 * kept warm_prime_map, which a chain's first evaluation falls back to, so
 * the second run's WARM seed seeded the third run. */
TEST_CASE("Two runs in one process: a run without WARM does not start its "
          "--hotvar from an earlier run's WARM seed",
          "[HPO][termination][two_runs][warm]")
{
    PcIsolationEnvUnset no_env;
    const std::string strategy =
        "  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
        "    MAXEVAL 1\n    SEED 3\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    const auto root = [](const fs::path& script) {
        return kRootLines + "PC_LAMMPS --script " + script.string() +
               " --varloss E 0 1 --outvar A --hotvar a0 0 100 5.0\n"
               "PC_ISOLATION THREAD\n";
    };
    ScratchDir shared;
    const fs::path script = shared.dir / "hot.in";
    write_hot_script(script);
    const fs::path warm = shared.dir / "warm.tadah";
    {
        TaRun seed("OPTIMIZER\n" + strategy, root(script));
        REQUIRE(tadah::hpo::hpo_run(seed.ctx) == 0);
        REQUIRE(fs::exists("best_pot.tadah"));
        /* The potential's own HOT line (a0 = 5) replaced by the seed's. */
        std::ofstream out(warm);
        out << "# HOT " << script.string() << " a0 42.0\n";
        std::istringstream pot(read_file("best_pot.tadah"));
        for (std::string line; std::getline(pot, line);)
            if (line.rfind("# HOT ", 0) != 0) out << line << "\n";
    }
    {
        TaRun warmed("OPTIMIZER\n  INIT\n    METHOD WARM " + warm.string() +
                         "\n  ENDINIT\n" + strategy,
                     root(script), /*new_process=*/false);
        REQUIRE(tadah::hpo::hpo_run(warmed.ctx) == 0);
        const auto a = column_of("outvar.tadah", "A");
        REQUIRE_FALSE(a.empty());
        CHECK(a.front() == 42.0);
    }
    TaRun plain("OPTIMIZER\n" + strategy, root(script), /*new_process=*/false);
    REQUIRE(tadah::hpo::hpo_run(plain.ctx) == 0);
    const auto a = column_of("outvar.tadah", "A");
    REQUIRE(a.size() == 1u);
    CHECK(a.front() == 5.0);
}

/* Pins: a run that fails after it started its PC_ISOLATION PROCESS worker
 * processes stops them: the first run (one PC_LAMMPS line, PC_ISOLATION
 * PROCESS, THREADS 1) starts its worker ("PC_ISOLATION PROCESS: the
 * PC_LAMMPS and PC_FAILPROBE scripts run in 1 worker process" in its
 * journal; it was "PC_ISOLATION: PROCESS -- pc_lammps tasks execute in 1
 * persistent worker processes") and then fails building its host (its validation DBFILE does
 * not exist: hpo_run throws, the journal has "ERROR" and "startup: "); no
 * worker pool is up after it, and MLIP has no parallel backend installed.
 * The second run in the process, the same configuration with an existing
 * validation file, starts its own worker, returns 0 with a best_pot.tadah,
 * and leaves no pool up either.
 * Before the fix: the pool was shut down only after a run that returned,
 * so it was still up after the failed run (pc_worker_pool_enabled()), and
 * the second run threw 'pc_worker_pool_init: pool already initialized'.
 * Skipped in the MPI build (TADAH_HPO_MPI), which has no PROCESS workers
 * (pc_process_workers.h). Before it skipped there it failed there: hpo_run
 * refuses PC_ISOLATION PROCESS at startup in that build, so the first
 * journal had no "PC_ISOLATION PROCESS: ..." line. */
TEST_CASE("Two runs in one process: a run that fails after starting its "
          "PC_ISOLATION PROCESS workers stops them, and the next run starts "
          "its own", "[HPO][termination][two_runs][pc_isolation]")
{
    SKIP_WITHOUT_PC_PROCESS_WORKERS();
    PcIsolationEnvUnset no_env;
    const std::string optimizer = plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 2\n"
        "    SEED 1\n    THREADS 1\n");
    const std::string root = kRootLines +
        "PC_LAMMPS --script pe.in --varloss E 0 1\nPC_ISOLATION PROCESS\n";
    const std::string started =
        "PC_ISOLATION PROCESS: the PC_LAMMPS and PC_FAILPROBE scripts run in "
        "1 worker process,";
    {
        TaRun run(optimizer, root);
        write_pe_script(run.sd.dir / "pe.in", "E");
        std::ofstream(run.val)
            << "DBFILE " << (run.sd.dir / "missing.db").string() << "\n";
        CHECK_THROWS(tadah::hpo::hpo_run(run.ctx));
        const std::string journal = read_file("hpo_run.log");
        INFO("first journal:\n" << journal);
        REQUIRE(lines_with(journal, started).size() == 1u);
        CHECK(lines_with(journal, "startup: ").size() == 1u);
        CHECK_FALSE(tadah::hpo::pc_worker_pool_enabled());
        CHECK_FALSE(tadah::mlip::has_parallel_backend());
    }
    {
        TaRun run(optimizer, root, /*new_process=*/false);
        write_pe_script(run.sd.dir / "pe.in", "E");
        int status = -1;
        CHECK_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        const std::string journal = read_file("hpo_run.log");
        INFO("second journal:\n" << journal);
        CHECK(status == 0);
        CHECK(lines_with(journal, started).size() == 1u);
        CHECK(fs::exists("best_pot.tadah"));
        CHECK_FALSE(tadah::hpo::pc_worker_pool_enabled());
        CHECK_FALSE(tadah::mlip::has_parallel_backend());
    }
}

/* Pins: every run gets a worker pool of its own THREADS and stops it when
 * it ends: two runs in one process, THREADS 2 then 3, each journal says
 * "THREADS: <its THREADS> workers run the work of the evaluations" once
 * (it was "ScriptExecutor: <n> workers = THREADS ..."), and after each
 * run MLIP has no parallel backend installed (a later caller in the
 * process, a training after hpo_run(), runs its kernels on OpenMP
 * again).
 * Before the fix: the backend stayed installed after the run (the journal
 * lines were right: they printed the THREADS asked for, while the pool kept
 * the first run's 2 workers -- test_script_executor.cpp's
 * "init_script_executor gives the executor the width it asks for"). */
TEST_CASE("Two runs in one process: each run's worker pool has the run's "
          "THREADS, and the run stops it", "[HPO][termination][two_runs]")
{
    for (int threads : {2, 3}) {
        CAPTURE(threads);
        TaRun run(plain_block(
                      "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                      "    MAXEVAL 4\n    SEED 1\n    THREADS " +
                      std::to_string(threads) + "\n"),
                  kRootLines, /*new_process=*/threads == 2);
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(lines_with(journal, "THREADS: " + std::to_string(threads) +
                                      " workers run the work of the "
                                      "evaluations").size() == 1u);
        CHECK_FALSE(tadah::mlip::has_parallel_backend());
    }
}

/* Pins: a second run with a smaller THREADS under PC_ISOLATION PROCESS runs
 * its PC_LAMMPS tasks on its own pool: the first run (THREADS 3, no
 * PC_LAMMPS) and then a run with THREADS 1 and three PC_LAMMPS lines (one
 * worker process) return 0, the second run's journal has no "bad slot"
 * and no failed task, and every one of its loss.tadah rows is feasible
 * (below FAILSCORE 1e9).
 * Before the fix: the second run's tasks ran on the first run's 3 executor
 * workers, and a task handed to worker 1 or 2 asked the one-worker pool
 * for a channel it does not have ('PcWorkerPool::execute: bad slot'),
 * failing its evaluation ("eval-thread caught throw, submitting
 * FAIL_SCORE"). Which worker takes a task is not fixed; in each of 8 runs
 * every evaluation failed so, and the run ended "STOPPED -- no evaluation
 * produced a usable loss" with status 1.
 * Skipped in the MPI build (TADAH_HPO_MPI), which has no PROCESS workers
 * (pc_process_workers.h). Before it skipped there it failed there: the
 * second hpo_run threw "PC_ISOLATION PROCESS is not supported in the MPI
 * build (TADAH_HPO_MPI): ...". */
TEST_CASE("Two runs in one process: a smaller THREADS under PC_ISOLATION "
          "PROCESS runs on the run's own pool",
          "[HPO][termination][two_runs][pc_isolation]")
{
    SKIP_WITHOUT_PC_PROCESS_WORKERS();
    PcIsolationEnvUnset no_env;
    {
        TaRun run(plain_block(
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 3\n"
            "    SEED 1\n    THREADS 3\n"));
        REQUIRE(tadah::hpo::hpo_run(run.ctx) == 0);
    }
    TaRun run(plain_block(
                  "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                  "    MAXEVAL 4\n    SEED 1\n    THREADS 1\n"),
              kRootLines +
                  "PC_LAMMPS --script pe1.in --varloss E1 0 1\n"
                  "PC_LAMMPS --script pe2.in --varloss E2 0 1\n"
                  "PC_LAMMPS --script pe3.in --varloss E3 0 1\n"
                  "PC_ISOLATION PROCESS\n",
              /*new_process=*/false);
    write_pe_script(run.sd.dir / "pe1.in", "E1");
    write_pe_script(run.sd.dir / "pe2.in", "E2");
    write_pe_script(run.sd.dir / "pe3.in", "E3");
    int status = -1;
    CHECK_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(status == 0);
    CHECK(lines_with(journal, "bad slot").empty());
    CHECK(lines_with(journal, "task failed").empty());
    const auto loss = data_rows("loss.tadah");
    CHECK(loss.size() == 4u);
    for (const auto& row : loss) {
        INFO("loss.tadah row " << row.at(0));
        CHECK(loss_value(row.back()) < kFailScore);
    }
}

/* Pins: under PC_ISOLATION THREAD each run's LAMMPS instances end with the
 * run, and a later run in the process runs its own script: two runs
 * (THREADS 1, TADAH:RANDOM MAXEVAL 2 SEED 1, one line "PC_LAMMPS --script
 * pe.in --varloss E 0 1 -log lmp.log"), the second with pe.in rewritten from
 * "variable E equal pe/atoms" to "variable E equal pe/atoms+1". Each run
 * returns 0 with 2 outvar.tadah rows and the same candidates (params.tadah
 * without ITERATION); its LAMMPS log of executor worker 0 (lmp.log.t0) has
 * LAMMPS's "Total wall time" line, which LAMMPS writes when the instance is
 * destroyed, once the run has returned; and every E of the second run is
 * the first run's E of the same row plus 1.
 * Before the fix: the executor was a function-local static whose workers
 * lived until process exit, and with them their thread-local LAMMPS
 * instance and the script it rendered once (PcSlotState, keyed by the
 * script's name and the line's LAMMPS options): after the first run its
 * log had no "Total wall time" line, and the second run ran the first
 * run's pe.in on that instance: its E were the first run's (about 1e-8),
 * not those plus 1. */
TEST_CASE("Two runs in one process: under PC_ISOLATION THREAD a run's LAMMPS "
          "instances end with the run, and the next run runs its own script",
          "[HPO][termination][two_runs][pc_isolation]")
{
    PcIsolationEnvUnset no_env;
    const std::string optimizer = plain_block(
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 2\n"
        "    SEED 1\n    THREADS 1\n");
    const std::string root = kRootLines +
        "PC_LAMMPS --script pe.in --varloss E 0 1 -log lmp.log\n"
        "PC_ISOLATION THREAD\n";
    const std::string lmp_log =
        "lmp.log" + tadah::hpo::hpo_rank_suffix() + ".t0";
    std::vector<std::vector<double>> energies;
    std::vector<std::vector<std::vector<std::string>>> candidates;
    for (const char* formula : {"pe/atoms", "pe/atoms+1"}) {
        CAPTURE(formula);
        TaRun run(optimizer, root, /*new_process=*/energies.empty());
        write_pe_script(run.sd.dir / "pe.in", "E", formula);
        int status = -1;
        CHECK_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string log = read_file(lmp_log);
        INFO(lmp_log << ":\n" << log);
        CHECK(log.find("Total wall time") != std::string::npos);
        energies.push_back(column_of("outvar.tadah", "E"));
        REQUIRE(energies.back().size() == 2u);
        auto params = data_rows("params.tadah");
        for (auto& row : params) row.erase(row.begin());
        candidates.push_back(params);
    }
    REQUIRE(candidates[1] == candidates[0]);
    for (std::size_t k = 0; k < 2; ++k) {
        CAPTURE(k, energies[0][k], energies[1][k]);
        CHECK(energies[1][k] == Approx(energies[0][k] + 1.0).epsilon(1e-12));
    }
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
    /* LN_SBPLX asks for the point of eval 6 again twice among its 12
     * calls, and each repeat is answered with eval 6's loss (every
     * evaluation of this run is a function of its point): 10 evaluations,
     * and NLopt's own MAXEVAL, which counts its calls, ends the search.
     * Until 2026-10-12 both repeats were evaluated again: 12 rows. */
    CHECK(data_rows("loss.tadah").size() == 10u);
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
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("ANNEAL stopped: MAXEVAL") != std::string::npos);
    /* Pins: the spent MAXEVAL is the run's status. Before the fix: the
     * trailer said COMPLETED. */
    CHECK(journal_status(journal) == "TADAH:ANNEAL: MAXEVAL reached");
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
     * which ends the run, converges on the bound SGRID2B[2] = 4, where its
     * next step cannot move the point. The trailer used to repeat the
     * exploration's status.
     * Pins: the trailer is the refinement's own status, "DLIB:BFGS:
     * FTOL_ABS reached" ("DLIB:BFGS: MAXEVAL reached" until 2026-10-12: its
     * line search evaluated the bound again and BFGS took its gradient
     * there again, until its MAXEVAL 20 was spent).
     * Before the fix: COMPLETED (DLIB:BFGS recorded no status). */
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
    CHECK(journal_status(journal) == "DLIB:BFGS: FTOL_ABS reached");
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
          restarts_used_status(4, 2));
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
          "refinement, and each refinement seed gets its own stagnation "
          "window", "[HPO][termination][restart]")
{
    /* A stagnating exploration hands its top-K to the refinement early. The
     * refinement used to inherit the exploration's stagnation count, so its
     * first evaluation already crossed the threshold; and MULTISTART (the
     * TOPK > 1 refinement) then swallowed the stop start by start, so
     * RESTART never fired. Now each refinement seed runs in a window of its
     * own, and after the last seed a refinement in which a seed stagnated
     * restarts the whole PRESTAGE. */
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
    CHECK(count_of(journal, "REFINEMENT: 2 of 2 seed(s) stopped on "
                            "stagnation") == 2u);
    CHECK(journal_status(journal) ==
          restarts_used_status(4, 1));

    /* One stagnation stop per phase, seed and round -- exploration, seed 1,
     * seed 2, exploration, seed 1, seed 2 -- each a full window after the
     * previous: 4 calls without a new best, evaluations or calls answered
     * from the run's record (a refinement seed's start, a point the
     * exploration evaluated, is one: the four seeds' starts among them).
     * Until the run kept one record (2026-10-12) each window was 4
     * evaluations: every seed evaluated its start again. */
    const auto stops = stagnation_windows(journal);
    REQUIRE(stops.size() == 6u);
    std::size_t reused = 0;
    for (std::size_t i = 1; i < stops.size(); ++i) {
        INFO("stagnation stop " << i << " at eval " << stops[i].eval << " ("
             << stops[i].reused << " call(s) answered from the record), "
             << "previous at " << stops[i - 1].eval);
        CHECK(stops[i].eval - stops[i - 1].eval + stops[i].reused >= 4);
        reused += stops[i].reused;
    }
    CHECK(reused >= 1u);
    /* Each of the four refinement seeds starts at a point the exploration
     * evaluated, answered from the run's record. */
    CHECK(count_of(journal, "which an earlier search of the run evaluated") ==
          4u);
    CHECK(count_of(journal, "REFINEMENT: refining top-2") == 2u);
    check_run_files();
}

TEST_CASE("Termination: under PRESTAGE, a refinement that improved the "
          "global best and then stagnated restarts the PRESTAGE",
          "[HPO][termination][restart]")
{
    /* The short exploration hands over its best point; the refinement
     * improves on it, then runs out of its stagnation window. The run used
     * to end there, with a status the stopped refinement never set
     * (COMPLETED, exit status 0) and its two restarts unused. A stagnating
     * refinement now restarts the PRESTAGE whether or not it improved the
     * global best, as a PLAIN run does, and the run ends with RESTART's
     * status once its MAX restarts are used. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    MAXEVAL 400\n    TOPK 1\n"
        "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
        "      MAXEVAL 3\n      SEED 5\n    ENDEXPLORE\n"
        "    RESTART\n      AFTER_STAGNATION 5\n      MAX 2\n"
        "      METHOD RANDOM 1\n      SEED 11\n    ENDRESTART\n"
        "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    /* Round 0's refinement improved on the exploration, then stagnated,
     * and the PRESTAGE restarted. */
    const auto improved_then_stagnated = journal.find(
        "REFINEMENT: 1 of 1 seed(s) stopped on stagnation (the refinement "
        "improved the global best); reporting stagnation to RESTART");
    const auto restart1 = journal.find("RESTART 1/2 -- fresh init");
    REQUIRE(improved_then_stagnated != std::string::npos);
    REQUIRE(restart1 != std::string::npos);
    CHECK(improved_then_stagnated < restart1);
    CHECK(journal.find("RESTART 2/2 -- fresh init") != std::string::npos);
    CHECK(count_of(journal, "REFINEMENT: refining top-1") == 3u);
    CHECK(journal_status(journal) == restarts_used_status(5, 2));
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
          restarts_used_status(6, 2));
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
          restarts_used_status(5, 3));
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

namespace {

/* What RecordingPolicy saw: its runs, and the INIT points and their
 * origins of the last run. Outlives the policy, which RestartPolicy owns. */
struct RunRecord {
    std::size_t runs = 0;
    std::vector<std::vector<double>> init;
    std::vector<tadah::hpo::InitOrigin> origin;
};

/* Stagnates on its first run and records every run in `rec`. */
struct RecordingPolicy : tadah::hpo::RunPolicy<ParamVec, ::LammpsRunner> {
    RunRecord& rec;
    explicit RecordingPolicy(RunRecord& r) : rec(r) {}
    void run(const tadah::hpo::OptimizerSpec& spec, Host&) override
    {
        ++rec.runs;
        rec.init = spec.optim_init;
        rec.origin = spec.optim_init_origin;
        if (rec.runs == 1) throw Host::StagnationStop{};
    }
};

}  // namespace

TEST_CASE("Termination: a RESTART refresh applies the run's WARM density-form "
          "rule and tags the fresh points", "[HPO][termination][restart][rhoform]")
{
    /* The refresh draws its points with the run's -c configuration in its
     * InitContext, so a WARM seed meets the RHOFORM (and EMBF0) rules there
     * as it does at startup, and the fresh points carry their origin
     * (optim_init_origin), as the INIT chains' points do. */
    HostFixture fx;   // RestartPolicy uses the host's statics only
    tadah::core::Context ctx = rhoform_test::make_ream_ctx("SIGNED");
    rhoform_test::TempFile target(std::string("OPTIM AMPGRIDMB (1) 0 1\n"
                                              "PC_ERMSE 0 1\n") +
                                  rhoform_test::kOptimizer);
    tadah::hpo::HPOTargets trg(ctx, target.path.string());
    REQUIRE(trg.spec.optim_init_origin.size() == 1);
    REQUIRE(trg.spec.optim_init_origin[0].kind ==
            tadah::hpo::InitOrigin::Kind::Config);

    auto restart_from = [&](const std::string& rhoform_line, RunRecord& rec) {
        rhoform_test::TempFile seed(
            rhoform_test::ream_seed("0.4 0.3 0.1", rhoform_line));
        tadah::hpo::InitStrategySpec is;
        is.chain.push_back({"WARM", {seed.path.string()}});
        tadah::hpo::RestartPolicy<ParamVec, ::LammpsRunner> restart(
            std::make_unique<RecordingPolicy>(rec), 5, 1,
            tadah::hpo::build_init_strategy(is), &trg);
        restart.run(trg.spec, fx.host);
    };
    SECTION("a seed without RHOFORM is refused when a density key is searched") {
        RunRecord rec;
        REQUIRE_THROWS_WITH(restart_from("", rec),
                            Catch::Contains("does not declare RHOFORM") &&
                                Catch::Contains("this run evaluates RHOFORM "
                                                "SIGNED"));
        CHECK(rec.runs == 1u);
    }
    SECTION("a seed in the run's form: the fresh point is tagged WARM") {
        RunRecord rec;
        REQUIRE_NOTHROW(restart_from("RHOFORM SIGNED\n", rec));
        CHECK(rec.runs == 2u);
        REQUIRE(rec.init.size() == 1u);
        CHECK(rec.init[0][0] == 0.4);
        REQUIRE(rec.origin.size() == 1u);
        CHECK(rec.origin[0].kind == tadah::hpo::InitOrigin::Kind::Warm);
    }
}

TEST_CASE("Termination: a RESTART refresh draws its WARM seed under the "
          "RESTART block's WARM_ON_MISMATCH", "[HPO][termination][restart][warm]")
{
    /* Pins: a RESTART round draws its WARM seed under the WARM_ON_MISMATCH
     * of the RESTART block (build_init_strategy binds it to the chain's
     * WARM entries), not under the INIT block of the spec it restarts,
     * which has the default `warn` here. The seed's AMPGRIDMB[1] 1.5 lies
     * above the box's 1. With `error` the first restart throws, naming the
     * clamping, before a second run; with `skip` the seed is dropped, the
     * chain produces no point, and the round starts from the previous
     * round's points (the CONFIG point 0.5) with a WARNING that says so;
     * with `warn` the round starts from the clamped seed, 1.
     * Before the fix: the refresh took the restarted spec's INIT policy
     * (`warn`) whatever the RESTART block said, so `error` and `skip`
     * restarted from the clamped seed 1 (no throw), and an empty refresh
     * restarted from the previous points without a word. The `warn`
     * SECTION passes as well (a pin). */
    HostFixture fx;   // RestartPolicy uses the host's statics only
    tadah::core::Context ctx = rhoform_test::make_ream_ctx("SIGNED");
    rhoform_test::TempFile target(std::string("OPTIM AMPGRIDMB (1) 0 1\n"
                                              "PC_ERMSE 0 1\n") +
                                  rhoform_test::kOptimizer);
    tadah::hpo::HPOTargets trg(ctx, target.path.string());
    REQUIRE(trg.spec.init_strategy_spec.warm_on_mismatch == "warn");
    REQUIRE(trg.spec.optim_init.size() == 1u);
    REQUIRE(trg.spec.optim_init[0].size() == 1u);
    REQUIRE(trg.spec.optim_init[0][0] == 0.5);
    rhoform_test::TempFile seed(rhoform_test::ream_seed("1.5 0.3 0.1"));

    const auto restart_with = [&](const char* policy, RunRecord& rec) {
        tadah::hpo::InitStrategySpec is;
        is.chain.push_back({"WARM", {seed.path.string()}});
        is.warm_on_mismatch = policy;
        tadah::hpo::RestartPolicy<ParamVec, ::LammpsRunner> restart(
            std::make_unique<RecordingPolicy>(rec), 5, 1,
            tadah::hpo::build_init_strategy(is), &trg);
        restart.run(trg.spec, fx.host);
    };
    RunRecord rec;
    SECTION("WARM_ON_MISMATCH error: the restart fails on the clamped seed") {
        REQUIRE_THROWS_WITH(restart_with("error", rec),
                            Catch::Contains("clamped into OPTIM bounds"));
        CHECK(rec.runs == 1u);
    }
    SECTION("WARM_ON_MISMATCH skip: the round keeps the previous points, "
            "and the journal says so") {
        REQUIRE_NOTHROW(restart_with("skip", rec));
        CHECK(rec.runs == 2u);
        REQUIRE(rec.init.size() == 1u);
        REQUIRE(rec.init[0].size() == 1u);
        CHECK(rec.init[0][0] == 0.5);
        tadah::hpo::HPOLog::instance().finish("TEST");
        CHECK_THAT(read_file(fx.run.sd.dir / "hpo_run.log"),
                   Catch::Contains("RESTART 1/1: the METHOD chain (WARM) "
                                   "produced no starting point; the round "
                                   "starts from the previous round's "
                                   "starting points"));
    }
    SECTION("WARM_ON_MISMATCH warn: the round starts from the clamped seed") {
        REQUIRE_NOTHROW(restart_with("warn", rec));
        CHECK(rec.runs == 2u);
        REQUIRE(rec.init.size() == 1u);
        REQUIRE(rec.init[0].size() == 1u);
        CHECK(rec.init[0][0] == 1.0);
    }
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
 * with (and the SEED of its INIT block), and ends the round with a
 * stagnation stop. */
struct SeedRecorder : tadah::hpo::RunPolicy<ParamVec, ::LammpsRunner> {
    struct Seeds {
        std::optional<std::uint64_t> outer, inner, explore, explore_inner;
        std::optional<std::uint64_t> outer_init, explore_init;
    };
    std::vector<Seeds> rounds;
    void run(const tadah::hpo::OptimizerSpec& spec, Host&) override
    {
        Seeds s;
        s.outer = spec.seed;
        s.outer_init = spec.init_strategy_spec.seed;
        if (spec.inner) s.inner = spec.inner->seed;
        if (spec.prestage) {
            s.explore = spec.prestage->seed;
            s.explore_init = spec.prestage->init_strategy_spec.seed;
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
        /* A worked example: SEED 7, TOPK 3, MAX 2. The
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

    SECTION("an NLopt phase seeded by its INIT block's SEED: that SEED is "
            "derived")
    {
        /* Pins: NLopt takes an INIT block's SEED when its block has no SEED
         * (make_nlopt), so RESTART derives that SEED as it derives SEED: the
         * outer spec's INIT SEED 7 under MULTISTART N_STARTS 3 is 7, 10, 13
         * (SEED + r*S) and the EXPLORE spec's INIT SEED 30 is 30, 31, 32,
         * neither spec getting a SEED. Nothing else reads an INIT block's
         * SEED once its points are drawn at startup: a DLIB phase keeps its
         * INIT SEED 7 in every round, and so does an NLopt phase with a SEED
         * of its own (5, 6, 7), whose SEED NLopt takes. The run's own spec
         * keeps its seeds.
         * Before the fix: derive_restart_seeds derived SEED only, so an NLopt
         * search seeded by its INIT block drew round 0's random numbers in
         * every round: outer_init 7 and explore_init 30 in every round (the
         * NLopt checks fail). */
        OptimizerSpec spec;
        spec.lib = "NLOPT";
        spec.algo = "LN_SBPLX";
        spec.strategy_type = tadah::hpo::StrategyType::MULTISTART;
        spec.multistart.n_starts = 3;
        spec.init_strategy_spec.seed = 7;
        spec.prestage = std::make_unique<OptimizerSpec>();
        spec.prestage->lib = "NLOPT";
        spec.prestage->algo = "GN_CRS2_LM";
        spec.prestage->init_strategy_spec.seed = 30;

        const auto rounds = rounds_of(spec);
        for (std::uint64_t r = 0; r < rounds.size(); ++r) {
            INFO("round " << r);
            CHECK(rounds[r].outer_init == std::optional<std::uint64_t>(7 + 3 * r));
            CHECK(rounds[r].explore_init == std::optional<std::uint64_t>(30 + r));
            CHECK_FALSE(rounds[r].outer);
            CHECK_FALSE(rounds[r].explore);
        }
        CHECK(spec.init_strategy_spec.seed == std::optional<std::uint64_t>(7));
        CHECK(spec.prestage->init_strategy_spec.seed ==
              std::optional<std::uint64_t>(30));

        OptimizerSpec dlib;
        dlib.lib = "DLIB";
        dlib.algo = "BFGS";
        dlib.init_strategy_spec.seed = 7;
        for (const auto& round : rounds_of(dlib)) {
            CHECK(round.outer_init == std::optional<std::uint64_t>(7));
            CHECK_FALSE(round.outer);
        }

        OptimizerSpec own;
        own.lib = "NLOPT";
        own.algo = "GN_CRS2_LM";
        own.seed = 5;
        own.init_strategy_spec.seed = 7;
        const auto own_rounds = rounds_of(own);
        for (std::uint64_t r = 0; r < own_rounds.size(); ++r) {
            INFO("round " << r);
            CHECK(own_rounds[r].outer == std::optional<std::uint64_t>(5 + r));
            CHECK(own_rounds[r].outer_init == std::optional<std::uint64_t>(7));
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
                restarts_used_status(6, 2));

        /* The seed each round's search reports. */
        std::vector<std::string> seeds;
        const std::regex seed_re(R"(RANDOM search: seed=(\d+))");
        for (auto it = std::sregex_iterator(journal.begin(), journal.end(),
                                            seed_re);
             it != std::sregex_iterator(); ++it)
            seeds.push_back((*it)[1]);
        CHECK(seeds == std::vector<std::string>{"5", "6", "7"});
        /* The RESTART line names the seed the round's point was drawn with
         * (the RESTART block has none: S + r, S drawn from
         * std::random_device) and the derived seed; until 2026-10-12 it
         * named only the derived seed ("RESTART 1/2 -- fresh init from
         * RANDOM; SEED 6"). */
        CHECK(journal.find(drawn_round_line(journal, 1, 2, "RANDOM",
                                            "SEED 6")) != std::string::npos);
        CHECK(journal.find(drawn_round_line(journal, 2, 2, "RANDOM",
                                            "SEED 7")) != std::string::npos);

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
            restarts_used_status(6, 2));
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

/* The RESTART block's SEED used to be ignored: every restart drew its
 * point from std::random_device, so no run with RESTART could be repeated.
 * Round r now draws it with SEED + r, next to the optimiser seeds RESTART
 * derives for the round (SEED 5 + r here), from a stream of its own.
 * Pins: with RESTART SEED 42 (METHOD RANDOM 1, MAX 2) the journal names
 * RESTART SEED 43 and 44 for rounds 1 and 2; round 0 is the run SEED 5
 * makes without RESTART; round r starts at the point a run with RESTART
 * SEED 41 starts its round r + 1 from (both drawn with 42 + r), which is
 * not the point an INIT block's METHOD RANDOM 1 draws with SEED 42 + r, and
 * then draws the points of the derived SEED 5 + r; the whole run repeats.
 * Before the fix: round r started at the INIT block's point for SEED 42 +
 * r (the RANDOM method drew std::mt19937_64(SEED + r) in both blocks, so a
 * round whose RESTART SEED + r was an INIT block's SEED started where that
 * block's round 0 did). */
TEST_CASE("Termination: a RESTART SEED draws restart round r's points with "
          "SEED + r", "[HPO][termination][restart][seed]")
{
    const std::string seed_line = "      SEED 42\n";
    const auto a = run_restart_rounds(seed_line);
    CHECK(a.journal.find("RESTART 1/2 -- fresh init from RANDOM; "
                         "RESTART SEED 43, SEED 6\n") != std::string::npos);
    CHECK(a.journal.find("RESTART 2/2 -- fresh init from RANDOM; "
                         "RESTART SEED 44, SEED 7\n") != std::string::npos);

    /* Round 0 is the run SEED 5 makes without RESTART. */
    CHECK(a.rounds[0] ==
          reference_points(random_search_block(a.rounds[0].size(), 5, "")));

    /* A run whose round r + 1 draws with 41 + (r + 1) = 42 + r. */
    const auto shifted = run_restart_rounds("      SEED 41\n");
    for (std::uint64_t r = 1; r < a.rounds.size(); ++r) {
        INFO("round " << r << ": " << a.rounds[r].size() << " evaluation(s)");
        if (r + 1 < shifted.rounds.size())
            CHECK(a.rounds[r].front() == shifted.rounds[r + 1].front());
        /* ... not the point an INIT block's METHOD RANDOM 1 draws with
         * SEED 42 + r ... */
        const auto init_point = reference_points(
            "OPTIMIZER\n"
            "  INIT\n    METHOD RANDOM 1\n"
            "    SEED " + std::to_string(42 + r) + "\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 1\n    SEED 5\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        REQUIRE(init_point.size() == 1u);
        CHECK(a.rounds[r].front() != init_point.front());
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

namespace {

/* Whether no two of `pts` are the same point. */
bool all_distinct(const std::vector<Point>& pts)
{
    return std::set<Point>(pts.begin(), pts.end()).size() == pts.size();
}

}  // namespace

/* TADAH:RANDOM draws a stream of its own for its SEED (seed_stream.h), not
 * the numbers of the starting-point method RANDOM for the same seed: both
 * seeded std::mt19937_64 with the integer and drew uniform numbers in the
 * box, so with an INIT block's SEED equal to the optimiser's the search
 * evaluated its start point (the method's first point) again as its first
 * draw -- the reviewer's EXPLORE INIT RANDOM 2 SEED 2 with TADAH:RANDOM
 * SEED 2 evaluated (4.626487, 2.700472) twice.
 * Pins: with INIT { METHOD RANDOM 1, SEED 2 } and TADAH:RANDOM SEED 2
 * (MAXEVAL 4) the four evaluated points are distinct, and so are the three
 * of a PRESTAGE exploration with EXPLORE INIT { METHOD RANDOM 2, SEED 2 }
 * and TADAH:RANDOM SEED 2 (MAXEVAL 3).
 * Before the fix: evaluations 1 and 2 were the same point in both. */
TEST_CASE("Termination: TADAH:RANDOM does not draw the points of a RANDOM "
          "method with the same seed", "[HPO][termination][seed]")
{
    const std::string init_random =
        "INIT\n    METHOD RANDOM 1\n    SEED 2\n  ENDINIT\n";
    SECTION("the OPTIMIZER block's INIT block") {
        TaRun run("OPTIMIZER\n  " + init_random +
                  "  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
                  "    ALGO RANDOM\n    MAXEVAL 4\n    SEED 2\n"
                  "  ENDSTRATEGY\nENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const auto pts = evaluated_points();
        REQUIRE(pts.size() == 4u);
        INFO(::Catch::Detail::stringify(pts));
        CHECK(all_distinct(pts));
    }
    SECTION("the EXPLORE block's INIT block") {
        TaRun run(plain_block(
            "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
            "    MAXEVAL 4\n"
            "    EXPLORE\n"
            "      INIT\n        METHOD RANDOM 2\n        SEED 2\n"
            "      ENDINIT\n"
            "      LIB TADAH\n      ALGO RANDOM\n      MAXEVAL 3\n"
            "      SEED 2\n"
            "    ENDEXPLORE\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const auto pts = evaluated_points();
        REQUIRE(pts.size() >= 3u);
        const auto explored = evals_between(pts, 1, 3);
        INFO(::Catch::Detail::stringify(explored));
        CHECK(all_distinct(explored));
    }
}

/* A RESTART round draws its fresh point with RESTART SEED + r and runs a
 * seeded TADAH:RANDOM with SEED + r (under PLAIN): with the same SEED in
 * both blocks the two seeds are equal every round, and the search's first
 * draw of the round was the round's start point again (the documented
 * "evaluated twice").
 * Pins: with SEED 5 in the RESTART block and on TADAH:RANDOM, no round
 * (MAX 2, each ended by a stagnation stop) evaluates a point twice.
 * Before the fix: the first two evaluations of rounds 1 and 2 were the
 * same point. */
TEST_CASE("Termination: a RESTART round's fresh RANDOM point is not the "
          "seeded TADAH:RANDOM's first draw of the round",
          "[HPO][termination][restart][seed]")
{
    const auto a = run_restart_rounds("      SEED 5\n");
    CHECK(a.journal.find("RESTART 1/2 -- fresh init from RANDOM; "
                         "RESTART SEED 6, SEED 6\n") != std::string::npos);
    REQUIRE(a.rounds.size() == 3u);
    for (std::size_t r = 0; r < a.rounds.size(); ++r) {
        INFO("round " << r << ": "
             << ::Catch::Detail::stringify(a.rounds[r]));
        REQUIRE(a.rounds[r].size() >= 2u);
        CHECK(all_distinct(a.rounds[r]));
    }
}

namespace {

/* How many evaluations of `pts` repeat a point evaluated before. */
std::size_t repeated_points(const std::vector<Point>& pts)
{
    std::set<Point> seen;
    std::size_t n = 0;
    for (const auto& p : pts)
        if (!seen.insert(p).second) ++n;
    return n;
}

}  // namespace

/* A PRESTAGE refinement draws random numbers of its own (seed_stream.h):
 * its TADAH:RANDOM drew the exploration's numbers for an equal SEED, and
 * the seeds RESTART derives for a round -- the refinement SEED + r*S, the
 * exploration EXPLORE SEED + r -- met for SEEDs that differ, so a round's
 * refinement drew an earlier exploration's points again. A refinement
 * evaluates its start point, a candidate the exploration evaluated, again
 * (TADAH:RANDOM evaluates its INIT point first); no other evaluation of
 * these runs repeats a point.
 * Pins: (a) STRATEGY and EXPLORE TADAH:RANDOM with SEED 7 and MAXEVAL 5:
 * of the 10 evaluations only the 6th (the refinement's start point)
 * repeats an earlier point; (b) SEED 7, EXPLORE SEED 8, MAXEVAL 6 each,
 * RESTART { AFTER_STAGNATION 3, MAX 1, METHOD RANDOM 1, SEED 100 }: the
 * journal reads "RESTART 1/1 -- fresh init from RANDOM; RESTART SEED 101,
 * SEED 8, EXPLORE SEED 9" (round 1's refinement runs with round 0's
 * EXPLORE SEED), and the evaluations that repeat a point are as many as the
 * refinements ("REFINEMENT: refining" lines), their start points.
 * Before the fix: (a) evaluations 7-10 were evaluations 2-5 (5 repeats);
 * (b) round 1's refinement drew round 0's exploration points again (more
 * repeats than refinements). */
TEST_CASE("Termination: a PRESTAGE refinement does not draw its "
          "exploration's random numbers", "[HPO][termination][prestage][seed]")
{
    SECTION("equal SEEDs") {
        TaRun run(plain_block(
            "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 5\n    SEED 7\n"
            "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
            "      MAXEVAL 5\n      SEED 7\n    ENDEXPLORE\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const auto pts = evaluated_points();
        REQUIRE(pts.size() == 10u);
        INFO(::Catch::Detail::stringify(pts));
        CHECK(repeated_points(pts) == 1u);
        CHECK(repeated_points(evals_between(pts, 1, 6)) == 1u);
    }
    SECTION("the seeds RESTART derives for a round") {
        TaRun run(plain_block(
            "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 6\n    SEED 7\n"
            "    RESTART\n      AFTER_STAGNATION 3\n      MAX 1\n"
            "      METHOD RANDOM 1\n      SEED 100\n    ENDRESTART\n"
            "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
            "      MAXEVAL 6\n      SEED 8\n    ENDEXPLORE\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        REQUIRE(journal.find("RESTART 1/1 -- fresh init from RANDOM; "
                             "RESTART SEED 101, SEED 8, EXPLORE SEED 9\n") !=
                std::string::npos);
        const std::size_t refinements =
            count_of(journal, "REFINEMENT: refining");
        REQUIRE(refinements == 2u);
        const auto pts = evaluated_points();
        INFO(::Catch::Detail::stringify(pts));
        CHECK(repeated_points(pts) == refinements);
    }
}

TEST_CASE("Termination: METHOD SOBOL K starts the INIT round and each "
          "restart round at the first point of block SEED",
          "[HPO][termination][restart][seed][sobol]")
{
    /* A Sobol seed used to skip `seed` points. With METHOD SOBOL 3 and SEED
     * 42 in both blocks, the INIT block drew points 42-44, restart 1 (seed
     * 43) points 43-45 and restart 2 (seed 44) points 44-46: two of each
     * round's three fresh points were the previous round's. Seed s now
     * draws block s, points 3s ... 3s + 2, and PLAIN starts each round from
     * the first of them: this run checks that the seeds reach the blocks
     * (the rounds start at points 126, 129 and 132, not 42, 43 and 44).
     * That blocks share no point is SobolInitStrategy's (test_init_
     * strategies.cpp); a PLAIN run evaluates only a block's first point. */
    const std::string strategy =
        "  STRATEGY\n"
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
        "    MAXEVAL 5000\n    SEED 5\n"
        "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
        "      METHOD SOBOL 3\n      SEED 42\n    ENDRESTART\n"
        "  ENDSTRATEGY\n";
    std::vector<std::vector<Point>> rounds;
    {
        TaRun run("OPTIMIZER\n  INIT\n    METHOD SOBOL 3\n    SEED 42\n"
                  "  ENDINIT\n" + strategy + "ENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        REQUIRE(journal_status(journal) ==
                restarts_used_status(6, 2));
        CHECK(journal.find("RESTART 1/2 -- fresh init from SOBOL; "
                           "RESTART SEED 43, SEED 6\n") != std::string::npos);
        CHECK(journal.find("RESTART 2/2 -- fresh init from SOBOL; "
                           "RESTART SEED 44, SEED 7\n") != std::string::npos);
        const auto stops = stagnation_stops(journal);
        REQUIRE(stops.size() == 3u);
        const auto pts = evaluated_points();
        std::size_t first = 1;
        for (const auto& stop : stops) {
            rounds.push_back(evals_between(pts, first, stop.first));
            first = stop.first + 1;
        }
        check_run_files();
    }

    /* Sobol point n is what METHOD SOBOL 1 draws with SEED n: a block of
     * one point is the point itself. */
    const auto sobol_point = [](std::uint64_t n) {
        const auto pts = reference_points(
            "OPTIMIZER\n"
            "  INIT\n    METHOD SOBOL 1\n"
            "    SEED " + std::to_string(n) + "\n  ENDINIT\n"
            "  STRATEGY\n"
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 1\n    SEED 5\n"
            "  ENDSTRATEGY\n"
            "ENDOPTIMIZER\n");
        REQUIRE(pts.size() == 1u);
        return pts.front();
    };
    REQUIRE(rounds.size() == 3u);
    CHECK(rounds[0].front() == sobol_point(3 * 42));   // INIT, seed 42
    CHECK(rounds[1].front() == sobol_point(3 * 43));   // restart 1
    CHECK(rounds[2].front() == sobol_point(3 * 44));   // restart 2
}

/* Without a SEED in the RESTART block the restart points come from
 * std::random_device: one draw per run, S, at the first restart; round r
 * draws with S + r, as a block with SEED S does, and the journal names S.
 * Pins: two runs without a RESTART SEED (METHOD RANDOM 1, MAX 2) each
 * journal "RESTART: the RESTART block has no SEED, so its restart points
 * are drawn from std::random_device: restart r draws with S + r (SEED S in
 * the RESTART block draws the same points again)" and the round lines
 * "RESTART 1/2 -- fresh init from RANDOM; RESTART SEED <S + 1>, SEED 6"
 * and "... 2/2 ...; RESTART SEED <S + 2>, SEED 7"; their S differ, and so do
 * their restart points; round 0 and the drawn points still follow SEED 5 +
 * r; a run with SEED S in its RESTART block starts its rounds at the same
 * points.
 * Before the fix: no such line; each round drew std::random_device() + r
 * and the round lines named only the optimiser seeds ("RESTART 1/2 --
 * fresh init from RANDOM; SEED 6"), so a run's restart points could not be
 * drawn again. */
TEST_CASE("Termination: without a RESTART SEED the restart points come from "
          "std::random_device, and the journal names the seed",
          "[HPO][termination][restart][seed]")
{
    const auto a = run_restart_rounds("");
    const auto b = run_restart_rounds("");
    for (const auto* run : {&a, &b}) {
        INFO("journal:\n" << run->journal);
        REQUIRE(drawn_restart_seed(run->journal));
        CHECK(run->journal.find(drawn_round_line(run->journal, 1, 2, "RANDOM",
                                                 "SEED 6")) !=
              std::string::npos);
        CHECK(run->journal.find(drawn_round_line(run->journal, 2, 2, "RANDOM",
                                                 "SEED 7")) !=
              std::string::npos);
    }
    REQUIRE(drawn_restart_seed(a.journal));
    CHECK(drawn_restart_seed(a.journal) != drawn_restart_seed(b.journal));
    CHECK(a.rounds[0] == b.rounds[0]);
    CHECK(a.rounds[1].front() != b.rounds[1].front());
    CHECK(a.rounds[2].front() != b.rounds[2].front());

    /* The SEED the journal names draws the same restart points. */
    const auto again = run_restart_rounds(
        "      SEED " + std::to_string(*drawn_restart_seed(a.journal)) + "\n");
    CHECK(again.rounds[1].front() == a.rounds[1].front());
    CHECK(again.rounds[2].front() == a.rounds[2].front());
}

/* Pins: a PLAIN NLOPT:GN_CRS2_LM run (MAXEVAL 200) seeded only by its INIT
 * block's SEED 7 (no METHOD: it starts at the configuration's point), under
 * RESTART (AFTER_STAGNATION 6, MAX 2, METHOD RANDOM 1, SEED 5), evaluates
 * exactly the points of the same run seeded by the STRATEGY block's SEED 7,
 * round after round: round r seeds NLopt with 7 + r either way. The journal
 * names the derived seed: "RESTART 1/2 -- fresh init from RANDOM; RESTART
 * SEED 6, INIT SEED 8" and "... 2/2 ...; RESTART SEED 7, INIT SEED 9" (the
 * STRATEGY-seeded run: "..., SEED 8", "..., SEED 9"). Each of the three
 * rounds ends on stagnation, and rounds 1 and 2 evaluate more new points
 * than their two start points.
 * Before the fix: the INIT block's SEED was not derived: round r seeded
 * NLopt with 7 again, so CRS2 drew round 0's population again and every
 * call of rounds 1 and 2 after their start point asked for a point of
 * round 0, answered from the record ("StagnationStop fired at a repeat of
 * eval 6 answered from the run's record"); the RESTART line named no
 * optimiser seed ("RESTART 1/2 -- fresh init from RANDOM; RESTART SEED
 * 6"): the journal checks, the comparison with the STRATEGY-seeded run and
 * the new-point count (2, the start points) fail. */
TEST_CASE("Termination: an NLopt search seeded by its INIT block's SEED "
          "draws new random numbers in each RESTART round",
          "[HPO][termination][restart][seed]")
{
    const auto crs = [](const std::string& init, const std::string& seed) {
        return "OPTIMIZER\n" + init +
               "  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n"
               "    ALGO GN_CRS2_LM\n    MAXEVAL 200\n" + seed +
               "    RESTART\n      AFTER_STAGNATION 6\n      MAX 2\n"
               "      METHOD RANDOM 1\n      SEED 5\n    ENDRESTART\n"
               "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    struct Run {
        std::vector<Point> pts;
        std::string journal;
        std::vector<StagnationWindow> stops;
    };
    const auto run = [](const std::string& optimizer) {
        Run out;
        TaRun r(optimizer);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(r.ctx));
        CHECK(status == 0);
        out.journal = read_file("hpo_run.log");
        out.pts = evaluated_points();
        out.stops = stagnation_windows(out.journal);
        return out;
    };
    const Run init = run(crs("  INIT\n    SEED 7\n  ENDINIT\n", ""));
    const Run own = run(crs("", "    SEED 7\n"));
    INFO("INIT-seeded journal:\n" << init.journal);
    CHECK(init.journal.find("RESTART 1/2 -- fresh init from RANDOM; RESTART "
                            "SEED 6, INIT SEED 8\n") != std::string::npos);
    CHECK(init.journal.find("RESTART 2/2 -- fresh init from RANDOM; RESTART "
                            "SEED 7, INIT SEED 9\n") != std::string::npos);
    CHECK(own.journal.find("RESTART 1/2 -- fresh init from RANDOM; RESTART "
                           "SEED 6, SEED 8\n") != std::string::npos);
    CHECK(init.pts == own.pts);
    REQUIRE(init.stops.size() == 3u);
    const std::size_t round0 = init.stops.front().eval;
    REQUIRE(round0 <= init.pts.size());
    const std::set<Point> earlier(init.pts.begin(), init.pts.begin() + round0);
    std::size_t fresh = 0;
    for (std::size_t i = round0; i < init.pts.size(); ++i)
        fresh += earlier.count(init.pts[i]) == 0u;
    CHECK(fresh > 2u);
}

/* Pins: MULTISTART (and the PRESTAGE refinement it runs) gives start i of an
 * NLopt search seeded only by its INIT block's SEED that SEED + i, as it
 * gives a SEED: N_STARTS 2 of NLOPT:LN_SBPLX with an INIT block METHOD
 * RANDOM 2, SEED 7 journals "MULTISTART start 1/2 -- running inner optimiser
 * (INIT seed=7)" and "MULTISTART start 2/2 -- running inner optimiser (INIT
 * seed=8)".
 * Before the fix: every start took the INIT block's SEED 7, and the start
 * lines named no seed ("MULTISTART start 2/2 -- running inner optimiser"):
 * the checks fail. */
TEST_CASE("Termination: MULTISTART gives each start of an NLopt search "
          "seeded by its INIT block the INIT SEED + i",
          "[HPO][termination][multistart][seed]")
{
    TaRun run("OPTIMIZER\n  INIT\n    METHOD RANDOM 2\n    SEED 7\n"
              "  ENDINIT\n  STRATEGY\n    TYPE MULTISTART\n    LIB NLOPT\n"
              "    ALGO LN_SBPLX\n    MAXEVAL 8\n    N_STARTS 2\n"
              "  ENDSTRATEGY\nENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK(journal.find("MULTISTART start 1/2 -- running inner optimiser "
                       "(INIT seed=7)\n") != std::string::npos);
    CHECK(journal.find("MULTISTART start 2/2 -- running inner optimiser "
                       "(INIT seed=8)\n") != std::string::npos);
}

/* Pins: a MULTISTART of N_STARTS 2 whose INIT chain gives one starting
 * point (no INIT block: the CONFIG point) runs one start and journals
 * "MULTISTART: running 1 start", after the WARNING that counts the one
 * starting point and advises a setting the parser takes: "MULTISTART
 * N_STARTS 2 requested but the resolved INIT chain produced only 1
 * starting point; running 1 start, which is STRATEGY TYPE PLAIN. A chain
 * of 2 or more starting points (METHOD LHS 2, RANDOM 2 or SOBOL 2) runs 2
 * starts." It advises no lower N_STARTS, which would be N_STARTS 1, refused
 * at its line ("one start is STRATEGY TYPE PLAIN").
 * Before the fix: the WARNING ended "running 1 start(s). Add INIT points
 * (e.g. METHOD LHS/RANDOM/SOBOL with K >= 2) or lower N_STARTS." (the text
 * fails); the run line said "running 1 start" already. */
TEST_CASE("Termination: MULTISTART with one starting point journals that it "
          "runs 1 start", "[HPO][termination][multistart]")
{
    TaRun run("OPTIMIZER\n  STRATEGY\n    TYPE MULTISTART\n    N_STARTS 2\n"
              "    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 3\n"
              "  ENDSTRATEGY\nENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK(journal.find("MULTISTART: running 1 start\n") != std::string::npos);
    CHECK(journal.find("1 starts") == std::string::npos);
    CHECK(journal.find(
              "MULTISTART N_STARTS 2 requested but the resolved INIT chain "
              "produced only 1 starting point; running 1 start, which is "
              "STRATEGY TYPE PLAIN. A chain of 2 or more starting points "
              "(METHOD LHS 2, RANDOM 2 or SOBOL 2) runs 2 starts.\n") !=
          std::string::npos);
    CHECK(journal.find("lower N_STARTS") == std::string::npos);
}

/* Pins: a MULTISTART of N_STARTS 3 whose INIT chain gives two starting
 * points (METHOD RANDOM 2) runs two starts and journals "MULTISTART:
 * running 2 starts", after the WARNING that names the N_STARTS that runs
 * them, which the parser takes: "MULTISTART N_STARTS 3 requested but the
 * resolved INIT chain produced only 2 starting points; running 2 starts. A
 * chain of 3 or more starting points (METHOD LHS 3, RANDOM 3 or SOBOL 3)
 * runs 3 starts, and N_STARTS 2 asks for the 2 that run."
 * Before the fix: the WARNING read "... produced only 2 starting point(s);
 * running 2 start(s). Add INIT points (e.g. METHOD LHS/RANDOM/SOBOL with K
 * >= 3) or lower N_STARTS." (the text fails). */
TEST_CASE("Termination: MULTISTART with fewer starting points than N_STARTS "
          "names the N_STARTS that runs them", "[HPO][termination][multistart]")
{
    TaRun run("OPTIMIZER\n  INIT\n    METHOD RANDOM 2\n    SEED 4\n"
              "  ENDINIT\n  STRATEGY\n    TYPE MULTISTART\n    N_STARTS 3\n"
              "    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 4\n"
              "  ENDSTRATEGY\nENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK(journal.find("MULTISTART: running 2 starts\n") != std::string::npos);
    CHECK(journal.find(
              "MULTISTART N_STARTS 3 requested but the resolved INIT chain "
              "produced only 2 starting points; running 2 starts. A chain of "
              "3 or more starting points (METHOD LHS 3, RANDOM 3 or SOBOL 3) "
              "runs 3 starts, and N_STARTS 2 asks for the 2 that run.\n") !=
          std::string::npos);
}

/* Pins: a MULTISTART whose INIT chain has fewer starting points than
 * N_STARTS because K_TOTAL keeps fewer names that K_TOTAL in the WARNING,
 * and the advice depends on the points of the METHOD lines (LHS 4, 3 and
 * 3 here, K_TOTAL 2):
 *  - more than N_STARTS 3: "... produced only 2 starting points (K_TOTAL
 *    2); running 2 starts. With K_TOTAL 3, or without K_TOTAL, 3 starts
 *    run, and N_STARTS 2 asks for the 2 that run.";
 *  - as many as N_STARTS 3 (a K_TOTAL 3 would cap nothing, refused when
 *    the file is read): "... Without K_TOTAL, 3 starts run, ...";
 *  - fewer than N_STARTS 4: "... A chain of 4 or more starting points
 *    (METHOD LHS 4, RANDOM 4 or SOBOL 4) without K_TOTAL runs 4 starts,
 *    ...".
 * Before the fix: the WARNING named no K_TOTAL and advised METHOD lines
 * the block already had ("A chain of 3 or more starting points (METHOD LHS
 * 3, RANDOM 3 or SOBOL 3) runs 3 starts" beside METHOD LHS 4; the text
 * fails). */
TEST_CASE("Termination: MULTISTART with fewer starting points than N_STARTS "
          "because of K_TOTAL names K_TOTAL", "[HPO][termination][multistart]")
{
    struct Case { std::string lhs, n_starts, advice; };
    const std::vector<Case> cases = {
        {"4", "3", "With K_TOTAL 3, or without K_TOTAL, 3 starts run"},
        {"3", "3", "Without K_TOTAL, 3 starts run"},
        {"3", "4", "A chain of 4 or more starting points (METHOD LHS 4, "
                   "RANDOM 4 or SOBOL 4) without K_TOTAL runs 4 starts"},
    };
    for (const auto& c : cases) {
        INFO("METHOD LHS " << c.lhs << ", N_STARTS " << c.n_starts);
        TaRun run("OPTIMIZER\n  INIT\n    METHOD LHS " + c.lhs +
                  "\n    K_TOTAL 2\n    SEED 1\n  ENDINIT\n  STRATEGY\n"
                  "    TYPE MULTISTART\n    N_STARTS " + c.n_starts +
                  "\n    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 4\n"
                  "  ENDSTRATEGY\nENDOPTIMIZER\n");
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        INFO(journal);
        CHECK(journal.find("MULTISTART: running 2 starts\n") !=
              std::string::npos);
        CHECK(journal.find(
                  "MULTISTART N_STARTS " + c.n_starts + " requested but the "
                  "resolved INIT chain produced only 2 starting points "
                  "(K_TOTAL 2); running 2 starts. " + c.advice +
                  ", and N_STARTS 2 asks for the 2 that run.\n") !=
              std::string::npos);
    }
}

/* Pins: what the RESTART rounds of NLopt's StoGO evaluate, as the startup
 * WARNING says (PLAIN, 3 OPTIM dimensions, MAXEVAL 60, STRATEGY SEED 11,
 * RESTART AFTER_STAGNATION 4, MAX 1, METHOD RANDOM 1, SEED 5): round 1
 * makes evaluations, and
 *  - GD_STOGO: each is at a point of round 0 (the box centre's loss comes
 *    from the record, its stencil is evaluated again); the WARNING names
 *    the stencils ("evaluates their finite-difference stencils again
 *    (GRAD_STRATEGY NUMERICAL): no round after round 0 evaluates a new
 *    point");
 *  - GD_STOGO_RAND at the default POPULATION 6 (2n, below 2n + 1 = 7): each
 *    is at a point of round 0 (its first trial point is the box centre);
 *    the WARNING says "asks first for the points of round 0's first local
 *    search again";
 *  - GD_STOGO_RAND at POPULATION 7: its first evaluation is a point round 0
 *    did not evaluate (its trial points are all random, drawn with SEED
 *    12); the WARNING says "so no RESTART round evaluates the points of the
 *    block's METHOD lines".
 * Before the fix: the evaluations are the same (a pin of the behaviour),
 * but GD_STOGO's WARNING said "which the run's record answers: no round
 * after round 0 evaluates a new point, and the RESTART block's only effect
 * is to end the run ..." and GD_STOGO_RAND got none: the WARNING checks
 * fail. */
TEST_CASE("Termination: a RESTART round of StoGO evaluates what the startup "
          "warning says",
          "[HPO][termination][restart][start_point]")
{
    const auto stogo = [](const std::string& algo, const std::string& pop) {
        return plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO " + algo +
                           "\n    MAXEVAL 60\n    SEED 11\n" + pop +
                           "    RESTART\n      AFTER_STAGNATION 4\n"
                           "      MAX 1\n      METHOD RANDOM 1\n"
                           "      SEED 5\n    ENDRESTART\n");
    };
    for (const auto& [algo, pop, warning, repeats] :
         std::vector<std::tuple<std::string, std::string, std::string, bool>>{
             {"GD_STOGO", "",
              "evaluates their finite-difference stencils again "
              "(GRAD_STRATEGY NUMERICAL): no round after round 0 evaluates a "
              "new point", true},
             {"GD_STOGO_RAND", "",
              "asks first for the points of round 0's first local search "
              "again", true},
             {"GD_STOGO_RAND", "    POPULATION 7\n",
              "so no RESTART round evaluates the points of the block's METHOD "
              "lines", false}}) {
        INFO("NLOPT:" << algo << " " << pop);
        TaRun run(stogo(algo, pop));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const std::string journal = read_file("hpo_run.log");
        INFO(journal);
        CHECK(journal.find(warning) != std::string::npos);
        const auto pts = evaluated_points();
        const auto stops = stagnation_windows(journal);
        REQUIRE(stops.size() == 2u);
        const std::size_t round0 = stops.front().eval;
        REQUIRE(round0 < pts.size());  // round 1 makes evaluations
        const std::set<Point> earlier(pts.begin(), pts.begin() + round0);
        if (repeats) {
            for (std::size_t i = round0; i < pts.size(); ++i) {
                INFO("eval " << (i + 1));
                CHECK(earlier.count(pts[i]) == 1u);
            }
        } else {
            CHECK(earlier.count(pts[round0]) == 0u);
        }
    }
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
    /* An inactive RESTART block (AFTER_STAGNATION 0), whose SEED was not
     * reported (it drew no restart points), is refused at its
     * AFTER_STAGNATION line ("startup validation: a RESTART block that
     * never restarts the search is refused"). */
}

/* best_pot.tadah.lock, the file the publications of the best potential
 * lock (durable_best.h), goes when the run ends: no rank publishes after
 * the run's last collective. A run that ends normally leaves
 * best_pot.tadah and no lock file, and so does a second run in the same
 * process and directory.
 * Before the fix: an empty best_pot.tadah.lock stayed in the directory
 * after every run. */
TEST_CASE("Termination: a run leaves no best_pot.tadah.lock",
          "[HPO][termination][durable_best]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                          "    MAXEVAL 3\n    SEED 1\n"));
    for (int pass = 0; pass < 2; ++pass) {
        INFO("run " << pass + 1);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        CHECK(fs::exists("best_pot.tadah"));
        CHECK_FALSE(fs::exists("best_pot.tadah.lock"));
    }
}

/* A PRESTAGE run with SEED 7 in its STRATEGY block whose GN_CRS2_LM
 * exploration (MAXEVAL 20) has no SEED of its own: the exploration takes
 * the STRATEGY-level SEED, so two runs evaluate the same points, the points
 * of the run with SEED 7 in its EXPLORE block too, and the journal names
 * the exploration's seed and where it comes from ("EXPLORE SEED      : 7
 * (the STRATEGY-level SEED)"; "EXPLORE SEED      : 7" when the EXPLORE
 * block sets it). The TADAH:RANDOM refinement (MAXEVAL 3) draws from the
 * SEED's refinement streams in both runs.
 * Before the fix: the exploration had no seed, so CRS2 drew its population
 * from NLopt's generator unseeded (its time seed, or the state an earlier
 * run left), the two runs differed and the journal named no exploration
 * seed. */
TEST_CASE("Termination: a PRESTAGE run seeded at the STRATEGY level "
          "repeats its exploration", "[HPO][termination][prestage][seed]")
{
    const auto prestage = [](const std::string& explore_seed) {
        return plain_block(
            "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
            "    MAXEVAL 3\n    SEED 7\n    TOPK 1\n"
            "    EXPLORE\n      LIB NLOPT\n      ALGO GN_CRS2_LM\n"
            "      MAXEVAL 20\n" + explore_seed + "    ENDEXPLORE\n");
    };
    struct Run {
        std::vector<Point> pts;
        std::string journal;
    };
    const auto run = [](const std::string& optimizer) {
        Run out;
        TaRun r(optimizer);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(r.ctx));
        CHECK(status == 0);
        out.journal = read_file("hpo_run.log");
        out.pts = evaluated_points();
        return out;
    };
    const Run a = run(prestage(""));
    const Run b = run(prestage(""));
    const Run own = run(prestage("      SEED 7\n"));
    INFO("journal:\n" << a.journal);
    REQUIRE(a.pts.size() >= 20u);
    CHECK(a.journal.find("    EXPLORE SEED      : 7 (the STRATEGY-level "
                         "SEED)\n") != std::string::npos);
    CHECK(own.journal.find("    EXPLORE SEED      : 7\n") !=
          std::string::npos);
    CHECK(evals_between(a.pts, 1, 20) == evals_between(b.pts, 1, 20));
    CHECK(evals_between(a.pts, 1, 20) == evals_between(own.pts, 1, 20));
    CHECK(a.pts == b.pts);
    CHECK(a.pts == own.pts);
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
        /* Until 2026-10-12 without "RESTART SEED <S + 1>, " (S drawn from
         * std::random_device, now named). */
        CHECK(journal.find(drawn_round_line(journal, 1, 1, "RANDOM",
                                            "EXPLORE SEED 1718")) !=
              std::string::npos);
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
            restarts_used_status(3, 2));
    /* Until 2026-10-12 without "RESTART SEED <S + r>, " (S drawn from
     * std::random_device, now named). */
    CHECK(journal.find(drawn_round_line(journal, 1, 2, "RANDOM", "SEED 13")) !=
          std::string::npos);
    CHECK(journal.find(drawn_round_line(journal, 2, 2, "RANDOM", "SEED 16")) !=
          std::string::npos);

    check_start_seeds(journal, 10, 3, 3);
    check_run_files();
}

TEST_CASE("Termination: the PRESTAGE refinement starts of a restarted run "
          "never share a seed", "[HPO][termination][restart][seed]")
{
    /* The refinement multistarts the TOPK candidates, candidate i with
     * SEED + i. Round r now runs it with SEED + r*TOPK + i; it stepped by
     * N_STARTS (1 here), so round 1's candidates reused seeds 8 and 9 of
     * round 0. A worked example: SEED 7, TOPK 3, MAX 2 gives
     * {7,8,9}, {10,11,12}, {13,14,15}.
     *
     * The run is deterministic: every round starts from the configuration's
     * point (RESTART METHOD CONFIG) and explores with a seeded random
     * search, which stagnates early and hands over its top 3. Each
     * refinement start (MAXEVAL 12 split over TOPK 3) runs in a stagnation
     * window of its own; after the third start a refinement in which a
     * start stagnated restarts the PRESTAGE: every round runs all three
     * starts. */
    TaRun run(
        "OPTIMIZER\n"
        "  STRATEGY\n"
        "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
        "    MAXEVAL 12\n    TOPK 3\n    SEED 7\n"
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
            restarts_used_status(3, 2));
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

/* The points `opt` evaluates in [0, 1]^2 from (0.5, 0.5) and the NLopt
 * random state as it stands: for GN_CRS2_LM its start point, then a
 * population drawn from NLopt's random number generator; for GN_MLSL its
 * start point, its random samples and its local searches. */
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
        return crs2_draws(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off));
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
                restarts_used_status(6, 2));
        /* Until 2026-10-12 without "RESTART SEED <S + r>, " (S drawn from
         * std::random_device, now named). */
        CHECK(journal.find(drawn_round_line(journal, 1, 2, "RANDOM",
                                            "SEED 1718")) != std::string::npos);
        CHECK(journal.find(drawn_round_line(journal, 2, 2, "RANDOM",
                                            "SEED 1719")) != std::string::npos);

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

/* A MULTISTART of NLOPT AUGLAG with an LN_PRAXIS INNER over three LHS
 * start points: `init_seed` is the INIT block's SEED line (or ""), `seed`
 * the STRATEGY block's and `inner_seed` the INNER block's. MULTISTART
 * takes only local optimisers, and an INNER block only a local search (no
 * nested global search); of the local
 * NLopt algorithms LN_PRAXIS alone draws from NLopt's random number
 * generator, for its random steps once it stalls: on this loss a PLAIN run
 * from the configuration's point under SEED 10 and SEED 20 evaluates the
 * same 32 points first and differs from the 33rd; of the three starts
 * below (BUDGET_PER_START 150), the second and the third reach their random
 * steps, the first, on a flat part of the loss, does not. (These runs were an
 * AUGLAG with an INNER GN_CRS2_LM, a global search, which an INNER block
 * no longer takes; its population did not depend on the start point, so
 * each start's 25 points were compared with a PLAIN CRS2 run of SEED + i.
 * LN_PRAXIS's points depend on the start point: the checks below compare
 * the starts of two runs that share their start points, and the test case
 * "NLopt: MULTISTART start i draws from SEED + i" compares one start with
 * the NLopt run of SEED + i.) */
constexpr std::size_t kAuglagBudget = 150;

std::string auglag_praxis_multistart(const std::string& init_seed,
                                     const std::string& seed,
                                     const std::string& inner_seed)
{
    return "OPTIMIZER\n"
           "  INIT\n    METHOD LHS 3\n" + init_seed + "  ENDINIT\n"
           "  STRATEGY\n"
           "    TYPE MULTISTART\n    LIB NLOPT\n    ALGO AUGLAG\n"
           "    N_STARTS 3\n    BUDGET_PER_START " +
           std::to_string(kAuglagBudget) + "\n" + seed +
           "    INNER\n      LIB NLOPT\n      ALGO LN_PRAXIS\n" + inner_seed +
           "    ENDINNER\n"
           "  ENDSTRATEGY\n"
           "ENDOPTIMIZER\n";
}

/* Runs auglag_praxis_multistart(init_seed, seed) and returns the evaluated
 * points of each start, after checking that start i ran with seed base +
 * i, that its evaluations are the run's, in order, and that no two starts
 * share a point. */
std::vector<std::vector<Point>> auglag_praxis_starts(
    const std::string& init_seed, std::uint64_t base)
{
    TaRun run(auglag_praxis_multistart(
        init_seed, "    SEED " + std::to_string(base) + "\n", ""));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    REQUIRE(status == 0);
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
    std::vector<std::vector<Point>> starts;
    for (std::uint64_t i = 0; i < evals.size(); ++i) {
        INFO("start " << (i + 1) << ": " << evals[i] << " evaluation(s)");
        REQUIRE(evals[i] >= 1u);
        const auto start = evals_between(pts, first, first + evals[i] - 1);
        first += evals[i];
        for (const auto& p : start) {
            INFO("point " << ::Catch::Detail::stringify(p));
            CHECK(earlier.count(p) == 0u);
        }
        earlier.insert(start.begin(), start.end());
        starts.push_back(start);
    }
    CHECK(first == pts.size() + 1);
    return starts;
}

}  // namespace

/* Pins: MULTISTART runs start i with SEED + i, and the INIT block's SEED 99
 * draws the LHS start points only: two runs under INIT SEED 99, with
 * STRATEGY SEED 10 and 20, start each of their three starts at the same
 * point (its first evaluation), and the runs evaluate different points,
 * since NLopt draws LN_PRAXIS's random steps from SEED 10 + i and 20 + i
 * (the starts that take a random step). The INIT SEED used to seed NLopt
 * in place of SEED + i, so every start drew SEED 99's numbers: both runs
 * would then evaluate the same points.
 * Before the fix: passes as well (a pin; the runs were an AUGLAG with an
 * INNER GN_CRS2_LM, whose populations were compared with those of PLAIN
 * CRS2 runs of SEED 10 + i). */
TEST_CASE("Termination: NLopt MULTISTART starts with an INIT SEED each draw "
          "from a seed of their own", "[HPO][termination][seed][nlopt]")
{
    const auto a = auglag_praxis_starts("    SEED 99\n", 10);
    const auto b = auglag_praxis_starts("    SEED 99\n", 20);
    REQUIRE(a.size() == 3u);
    REQUIRE(b.size() == 3u);
    for (std::size_t i = 0; i < a.size(); ++i) {
        INFO("start " << (i + 1));
        CHECK(a[i].front() == b[i].front());
    }
    CHECK(a != b);
}

/* Pins: MULTISTART start i's NLopt search draws from SEED + i: the spec of
 * start i (multistart_start_spec(), which MultiStartPolicy runs) of an
 * AUGLAG with an INNER LN_PRAXIS and SEED 10 draws, from a fixed start in
 * [0, 1]^2, the points of the same run without a SEED whose NLopt
 * generator is seeded with 10 + i by hand, and the three starts draw
 * three different sets. This is the check
 * the runs of an AUGLAG with an INNER GN_CRS2_LM made end to end (start i
 * drew the CRS2 population of SEED + i); an INNER block no longer takes a
 * global search, and LN_PRAXIS's points
 * depend on the start point, which MULTISTART takes from the INIT chain.
 * Before the fix: passes as well (a pin). */
TEST_CASE("NLopt: MULTISTART start i draws from SEED + i",
          "[HPO][seed][nlopt][multistart]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "AUGLAG";
    sp.stop.maxeval = 200;
    sp.seed = 10;
    sp.optim_init = {{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}};
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_PRAXIS";
    const tadah::hpo::MultiStartPlan plan =
        tadah::hpo::plan_multistart(sp, 3, std::nullopt);
    REQUIRE(plan.starts == 3u);
    /* The same NLopt run without a SEED, whose generator is seeded by
     * hand. */
    const auto reference = [&](unsigned long seed) {
        tadah::hpo::OptimizerSpec unseeded =
            tadah::hpo::multistart_start_spec(sp, plan, 0);
        unseeded.seed.reset();
        nlopt::opt opt = tadah::hpo::make_nlopt(unseeded, 2,
                                                tadah::hpo::RestartStop::Off);
        nlopt::srand(seed);
        return crs2_draws(opt);
    };
    REQUIRE(reference(10) != reference(11));
    std::vector<std::vector<std::vector<double>>> drew;
    for (std::size_t i = 0; i < plan.starts; ++i) {
        INFO("start " << (i + 1));
        const tadah::hpo::OptimizerSpec start =
            tadah::hpo::multistart_start_spec(sp, plan, i);
        REQUIRE(start.seed);
        CHECK(*start.seed == 10u + i);
        drew.push_back(crs2_draws(
            tadah::hpo::make_nlopt(start, 2, tadah::hpo::RestartStop::Off)));
        CHECK(drew.back() == reference(10 + i));
    }
    CHECK(drew[0] != drew[1]);
    CHECK(drew[1] != drew[2]);
    CHECK(drew[0] != drew[2]);
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
 * indentation, the INNER block's lines have two spaces more. Each INNER
 * search is LN_PRAXIS, a local one (the REFINE block's was GN_CRS2_LM, a
 * global search, which an INNER block no longer takes; the cases only
 * parse the configuration). */
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
                 "      INNER\n        LIB NLOPT\n        ALGO LN_PRAXIS\n" +
                 inner + "      ENDINNER\n"
                 "    ENDREFINE\n");
         },
         [](const OptimizerSpec& s) -> const OptimizerSpec& { return s; }},
    };
}

/* The refusals name the block that takes the key (they ended "; set
 * <KEY> on the enclosing <block> block"). */
std::string inner_seed_error(const std::string& enclosing)
{
    return "SEED is not accepted inside an INNER block: NLopt draws the "
           "INNER search from the enclosing run's random stream; the "
           "enclosing " + enclosing + " block takes SEED";
}

std::string inner_grad_strategy_error(const std::string& enclosing)
{
    return "GRAD_STRATEGY is not accepted inside an INNER block: the "
           "enclosing block's GRAD_STRATEGY applies to its local search; "
           "the enclosing " + enclosing + " block takes GRAD_STRATEGY";
}

std::string inner_stopval_error(const std::string& enclosing)
{
    return "STOPVAL is not accepted inside an INNER block: NLopt's MLSL and "
           "AUGLAG replace the INNER search's STOPVAL with the enclosing "
           "block's; the enclosing " + enclosing + " block takes STOPVAL";
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

/* NLopt's MLSL and AUGLAG run their INNER search inside their own run and
 * hand it the gradient of their own objective, which the enclosing block's
 * GRAD_STRATEGY obtains (nlopt_adapter.h). A GRAD_STRATEGY written in the
 * INNER block was accepted and never used; it is a startup error at its
 * line, like an INNER SEED. */
TEST_CASE("NLopt: GRAD_STRATEGY inside an INNER block is rejected at "
          "startup, naming its line",
          "[HPO][grad_strategy][nlopt][hpo_targets]")
{
    using tadah::hpo::GradStrategy;
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        for (const std::string value : {"NUMERICAL", "ANALYTICAL", "HYBRID"}) {
            const std::string inner_gs =
                c.indent + "  GRAD_STRATEGY " + value;
            /* Whether or not the enclosing block sets one of its own. */
            for (const std::string outer : {"", "GRAD_STRATEGY HYBRID"}) {
                const std::string cfg = c.config(
                    outer.empty() ? "" : c.indent + outer + "\n",
                    inner_gs + "\n");
                INFO(cfg);
                const std::string msg = startup_error(cfg);
                CHECK_THAT(msg, Catch::StartsWith(
                    "Error on line " +
                    std::to_string(line_number(cfg, inner_gs)) + ": " +
                    inner_grad_strategy_error(c.enclosing)));
                CHECK_THAT(msg, Catch::EndsWith("\nLine: " + inner_gs));
            }
        }

        /* Moved to the enclosing block, the GRAD_STRATEGY is that block's,
         * and the INNER spec carries none. */
        TaRun run(c.config(c.indent + "GRAD_STRATEGY HYBRID\n", ""));
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        const auto& block = c.block(trg.spec);
        CHECK(block.grad_strategy ==
              std::optional<GradStrategy>(GradStrategy::Hybrid));
        REQUIRE(block.inner);
        CHECK_FALSE(block.inner->grad_strategy);
    }
}

/* NLopt's MLSL and AUGLAG set their local search's stopval to the enclosing
 * run's (mlsl.c, auglag.c; HPO sets AUGLAG no constraint), so an INNER
 * block's STOPVAL never takes effect: inner_run_spec() models the
 * replacement. Pins: STOPVAL in an INNER block is a startup error at its
 * line, naming the enclosing block, in each block that can hold an INNER
 * block; moved to the enclosing block it is the NLopt run's.
 * Before the fix: the INNER STOPVAL parsed (startup_error ""), the journal
 * printed it ("Inner stop criteria: stopval 1.0"), and a STOPVAL above
 * every loss did not stop the run. */
TEST_CASE("NLopt: STOPVAL inside an INNER block is rejected at startup, "
          "naming its line", "[HPO][stop][nlopt][hpo_targets]")
{
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        const std::string inner_stopval = c.indent + "  STOPVAL 1";

        /* Whether or not the enclosing block has a STOPVAL of its own. */
        for (const std::string outer : {"", "STOPVAL 0.5"}) {
            const std::string cfg = c.config(
                outer.empty() ? "" : c.indent + outer + "\n",
                inner_stopval + "\n");
            INFO(cfg);
            const std::string msg = startup_error(cfg);
            CHECK(msg == "Error on line " +
                             std::to_string(line_number(cfg, inner_stopval)) +
                             ": " + inner_stopval_error(c.enclosing) +
                             "\nLine: " + inner_stopval);
        }

        /* Moved to the enclosing block, the STOPVAL is the NLopt run's. */
        TaRun run(c.config(c.indent + "STOPVAL 1\n", ""));
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        const auto& block = c.block(trg.spec);
        CHECK(block.stop.stopval == std::optional<double>(1.0));
        REQUIRE(block.inner);
        CHECK_FALSE(block.inner->stop.stopval);
    }
}

/* Pins: the unknown-key error inside INNER lists the optimiser attributes
 * an INNER block takes: every one but SEED, GRAD_STRATEGY and STOPVAL,
 * which belong to the enclosing block.
 * Before the fix: the list named STOPVAL, which an INNER block accepted
 * (the test case was "... lists every optimiser attribute but SEED and
 * GRAD_STRATEGY"). */
TEST_CASE("NLopt: the unknown-key error of an INNER block lists every "
          "optimiser attribute but SEED, GRAD_STRATEGY and STOPVAL",
          "[HPO][seed][nlopt][hpo_targets]")
{
    std::string keys;
    for (const auto& k : tadah::hpo::opt_attr_keys())
        if (k != "SEED" && k != "GRAD_STRATEGY" && k != "STOPVAL")
            keys += (keys.empty() ? "" : "/") + k;
    REQUIRE(keys.find("GRAD_STRATEGY") == std::string::npos);
    REQUIRE(keys.find("STOPVAL") == std::string::npos);
    REQUIRE(keys.find("MAXEVAL") != std::string::npos);
    for (const auto& c : inner_contexts()) {
        INFO("INNER inside " << c.enclosing);
        const std::string bogus = c.indent + "  BOGUS 1";
        const std::string cfg = c.config("", bogus + "\n");
        /* The message names the line the block opened on and its END line
         * (it was "unknown key 'BOGUS' inside INNER block (only ... are
         * valid)"). */
        const std::string upto = cfg.substr(0, cfg.find(c.indent + "INNER\n"));
        const std::size_t inner_line =
            static_cast<std::size_t>(std::count(upto.begin(), upto.end(),
                                                '\n')) + 1;
        CHECK_THAT(startup_error(cfg), Catch::StartsWith(
            "Error on line " + std::to_string(line_number(cfg, bogus)) +
            ": unknown key 'BOGUS' inside the INNER block opened on line " +
            std::to_string(inner_line) + " (only " + keys + " are valid; "
            "ENDINNER closes it)"));
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

/* The run is NLopt's GN_MLSL, whose samples come from NLopt's random number
 * generator, with an INNER LN_BOBYQA. It was an AUGLAG with an INNER
 * GN_CRS2_LM, a global search, which an INNER block no longer takes.
 * Before that fix: passes as well (a pin). */
TEST_CASE("NLopt: make_nlopt seeds a run with an INNER once, from the block "
          "that runs it, and refuses an INNER seed", "[HPO][seed][nlopt]")
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "GN_MLSL";
    sp.stop.maxeval = 12;
    sp.seed = 1717;
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_BOBYQA";

    /* The same MLSL run with the stream seeded by hand. */
    const auto reference = [](unsigned long seed) {
        nlopt::opt opt("GN_MLSL", 2);
        opt.set_maxeval(12);
        opt.set_local_optimizer(nlopt::opt("LN_BOBYQA", 2));
        nlopt::srand(seed);
        return crs2_draws(opt);
    };
    const auto seed_1717 = reference(1717);
    REQUIRE(seed_1717.size() >= 3u);
    REQUIRE(seed_1717 != reference(99));
    CHECK(crs2_draws(tadah::hpo::make_nlopt(
              sp, 2, tadah::hpo::RestartStop::Off)) == seed_1717);

    /* An INNER spec with a seed can only come from outside the parser. It
     * used to reseed the stream in place of SEED 1717. */
    const auto refused = Catch::Contains(
        "NLopt draws the INNER search from the enclosing run's random "
        "stream");
    SECTION("an INNER SEED") {
        sp.inner->seed = 99;
        CHECK_THROWS_WITH(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off),
            refused);
    }
    SECTION("an INNER INIT SEED") {
        sp.inner->init_strategy_spec.seed = 99;
        CHECK_THROWS_WITH(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off),
            refused);
    }
    /* Pins: an INNER spec with an INIT chain or an INIT K_TOTAL (from
     * outside the parser, which refuses INIT inside INNER) is refused:
     * MLSL and AUGLAG choose the INNER search's starting points, and
     * nothing resolves an INNER chain.
     * Before the fix: nothing threw; the chain was ignored. */
    const auto chain_refused = Catch::StartsWith(
        "NLopt INNER spec with an INIT chain: MLSL and AUGLAG choose the "
        "INNER search's starting points");
    SECTION("an INNER INIT chain") {
        sp.inner->init_strategy_spec.chain = {{"RANDOM", {"1"}}};
        CHECK_THROWS_WITH(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off),
            chain_refused);
    }
    SECTION("an INNER INIT K_TOTAL") {
        sp.inner->init_strategy_spec.k_total = 2;
        CHECK_THROWS_WITH(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off),
            chain_refused);
    }
}

TEST_CASE("NLopt: make_nlopt refuses an INNER spec with a GRAD_STRATEGY",
          "[HPO][grad_strategy][nlopt]")
{
    /* The parser rejects GRAD_STRATEGY inside INNER; an INNER spec with
     * one can only come from outside it, and its value would not be used:
     * the enclosing run hands its INNER search the gradient its own
     * GRAD_STRATEGY obtains. */
    using tadah::hpo::GradStrategy;
    /* The INNER search is LN_BOBYQA, a local one (it was GN_CRS2_LM, a
     * global search, which an INNER block no longer takes; before that
     * fix: passes as well, a pin). */
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "AUGLAG";
    sp.stop.maxeval = 12;
    sp.grad_strategy = GradStrategy::Hybrid;
    sp.inner = std::make_unique<tadah::hpo::OptimizerSpec>();
    sp.inner->lib = "NLOPT";
    sp.inner->algo = "LN_BOBYQA";
    REQUIRE_NOTHROW(
        tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off));

    for (const auto g : {GradStrategy::Numerical, GradStrategy::Analytical,
                         GradStrategy::Hybrid}) {
        INFO(tadah::hpo::grad_strategy_name(g));
        sp.inner->grad_strategy = g;
        CHECK_THROWS_WITH(
            tadah::hpo::make_nlopt(sp, 2, tadah::hpo::RestartStop::Off),
            Catch::StartsWith("NLopt INNER spec with a GRAD_STRATEGY: the "
                              "enclosing block's GRAD_STRATEGY applies to "
                              "its local search; set GRAD_STRATEGY on the "
                              "enclosing block"));
    }
}

TEST_CASE("Termination: NLopt MULTISTART starts draw from the STRATEGY "
          "block's SEED, and an INNER SEED is rejected",
          "[HPO][termination][seed][nlopt]")
{
    /* An example: SEED 10 on the STRATEGY block and SEED 20 on the
     * INNER. The INNER SEED used to seed NLopt in place of SEED + i, so all
     * three starts drew SEED 20's numbers. The INNER search is LN_PRAXIS
     * (it was GN_CRS2_LM, a global search, which an INNER block no longer
     * takes; before that fix the test passes as well, a pin). */
    const std::string inner_seed = "      SEED 20";
    const std::string rejected =
        auglag_praxis_multistart("", "    SEED 10\n", inner_seed + "\n");
    CHECK_THAT(startup_error(rejected), Catch::StartsWith(
        "Error on line " + std::to_string(line_number(rejected, inner_seed)) +
        ": " + inner_seed_error("STRATEGY")));

    /* With the SEED on the STRATEGY block only, start i runs with SEED 10
     * + i: 10, 11 and 12, and no two starts evaluate one point; that NLopt
     * draws from that seed is the test case "NLopt: MULTISTART start i
     * draws from SEED + i". */
    CHECK(auglag_praxis_starts("", 10).size() == 3u);
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
    CHECK(journal_status(journal).rfind(
              "FAILED -- best-potential persistence: ", 0) == 0);
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
    /* Pins: a journal finished with no status says so. Before the fix:
     * HPOLog::finish() wrote COMPLETED, a status nothing recorded. */
    CHECK(journal_status(read_file(sd.dir / "second.log")) ==
          "(no status recorded)");
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
    /* Before the fix: COMPLETED, which nothing recorded. */
    CHECK(journal_status(next) == "(no status recorded)");
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

/* Pins: ASA's stop line names the signal stop as a shutdown request with
 * its signal. Before the fix: "ASA stopped: exception raised during
 * evaluation after N host eval(s)". */
void check_asa_names_the_signal()
{
    CHECK_THAT(read_file("hpo_run.log"),
               Catch::Contains("ASA stopped: shutdown requested (SIGINT) "
                               "after "));
    CHECK_THAT(read_file("hpo_run.log"),
               !Catch::Contains("exception raised during evaluation"));
}

TEST_CASE("Termination: SIGINT stops ASA", "[.shutdown_asa]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000\n    SEED 7\n"), SIGINT, "SIGINT", 130, 1,
        check_asa_names_the_signal);
}

TEST_CASE("Termination: SIGINT stops ASA block-parallel with four threads",
          "[.shutdown_asa_block]")
{
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000000\n    THREADS 4\n    SEED 7\n"
        "    PARAM Gener_Block 4\n    PARAM Gener_Block_Max 4\n"),
        SIGINT, "SIGINT", 130, 4, check_asa_names_the_signal);
}

TEST_CASE("Termination: SIGTERM stops ASA after the periodic test that "
          "follows a new best", "[.shutdown_asa_periodic]")
{
    /* With SEED 8 and ASA's default options, the periodic test at
     * acceptance 100 (evaluation 167 on this fixture) follows a new best.
     * It averaged the count of states generated since that best (0) into
     * the block of generated states and truncated the block to 0 states:
     * asa() then looped for ever without evaluating, and the signal, which
     * the next evaluation would have seen, was never honoured. The signal
     * is raised well past that point. */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000\n    SEED 8\n"),
        SIGTERM, "SIGTERM", 143, 1, {}, 500);
}

TEST_CASE("Termination: SIGINT stops ASA inside its initial cost-temperature "
          "sampling with no invalid-state limit", "[.shutdown_asa_sampling]")
{
    /* The signal lands among the 1000000 states ASA samples for its initial
     * cost temperature. Every callback after it returns an invalid state
     * without evaluating, and the sampling loop, which upstream bounds only
     * by Limit_Invalid_Generated_States and in which it ignores
     * Immediate_Exit, drew states for ever with the limit off (-1). */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB ASA\n    ALGO ASA\n"
        "    MAXEVAL 100000000\n    SEED 7\n"
        "    PARAM Number_Cost_Samples 1000000\n"
        "    PARAM Limit_Invalid_Generated_States -1\n"),
        SIGINT, "SIGINT", 130, 1, [] {
            const std::string journal = read_file("hpo_run.log");
            /* Still sampling: the main loop's progress lines, on the
             * accepted-states cadence, never started. */
            CHECK(journal.find("ASA gen=") == std::string::npos);
            CHECK(journal.find("(asa exit: IMMEDIATE_EXIT)") !=
                  std::string::npos);
        });
}

#ifdef TADAH_USE_CERES
TEST_CASE("Termination: SIGINT stops CERES STEEPEST_DESCENT",
          "[.shutdown_ceres]")
{
    /* The signal lands at the third evaluation. CERES:STEEPEST_DESCENT is
     * the Ceres solver HPO runs (CERES:LBFGS, refused since 2026-10-15,
     * was the case before). Before the fix: passes as well (a pin). */
    signal_case(plain_block(
        "    TYPE PLAIN\n    LIB CERES\n    ALGO STEEPEST_DESCENT\n"
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

namespace {

/* The peak virtual memory of this process so far, in kB (VmPeak of
 * /proc/self/status), or -1 when it cannot be read. */
long vm_peak_kb()
{
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("VmPeak:", 0) == 0)
            return std::stol(line.substr(7));
    return -1;
}

}  // namespace

TEST_CASE("Termination: SIGINT during a NOISE_CALIBRATION of 2147483647 "
          "evaluations, which keeps the losses of the evaluations made only",
          "[.shutdown_noise_max]")
{
    /* NOISE CALIBRATION takes up to 2147483647 evaluations, as MAXEVAL
     * does. Before the fix the calibration reserved room for all N losses
     * before its first evaluation, 16 GiB for this N: the run stopped
     * with "vector::reserve" where that much memory was not available,
     * and took it otherwise (VmPeak rose by 16 GiB). */
    const long before = vm_peak_kb();
    REQUIRE(before > 0);
    signal_case(
        "OPTIMIZER\n"
        "  NOISE\n    CALIBRATION 2147483647\n  ENDNOISE\n"
        "  STRATEGY\n" + kBfgsLong + "  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n", SIGINT, "SIGINT", 130, 1, [&] {
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("NOISE_CALIBRATION: running 2147483647 evals")
                  != std::string::npos);
            const long grew = vm_peak_kb() - before;
            INFO("VmPeak grew by " << grew << " kB");
            CHECK(grew < 4L * 1024 * 1024);
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

/* A body that ends normally records its status, as every optimiser does
 * (run_end.h); one that records none is an internal error (below).
 * Pins: the trailer is the status the body recorded, and the run returns 0.
 * Before the fix: passes as well (a pin; a body without a status ended
 * COMPLETED, which this case used to pin). */
TEST_CASE("Termination: drive_run completes a normal end with the "
          "optimiser's status", "[HPO][termination][run_driver]")
{
    HostFixture fx;
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            fx.log().set_status("TEST: done");
        }, fx.host, fx.log()));
    CHECK(status == 0);
    check_driver_files("TEST: done");
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
    check_driver_files("FAILED -- optimiser broke");
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
    check_driver_files("FAILED -- non-standard exception");
}

/* Best-potential persistence is one rule: a new best that could not be
 * saved fails the run -- trailer "FAILED -- best-potential persistence:
 * <cause>", the durable_best::Error rethrown (tadah hpo exits 1) -- however
 * the run ends: by the error itself, or, when the error was outranked or
 * swallowed on its way up, by a normal end, a signal stop or a stagnation
 * stop. The cause is the first recorded failure. */
TEST_CASE("Termination: drive_run fails a run whose best potential could "
          "not be saved, however it ends", "[HPO][termination][run_driver]"
          "[durable_best]")
{
    const std::string kCause = "cannot fsync durable-best directory";
    const std::string kTrailer = "FAILED -- best-potential persistence: " + kCause;
    auto check_failed = [&](const std::function<void(HostFixture&)>& end) {
        HostFixture fx;
        CHECK_THROWS_MATCHES(
            tadah::hpo::drive_run([&] {
                fx.evaluate(kDriverEvals);
                fx.log().record_unsaved_best(kDriverEvals, 0.5, kCause, false);
                end(fx);
            }, fx.host, fx.log()),
            tadah::hpo::durable_best::Error,
            Catch::Matchers::Message(kCause));
        check_driver_files(kTrailer);
        const std::string journal = read_file("hpo_run.log");
        CHECK(journal.find("Best-potential persistence failed: " + kCause) !=
              std::string::npos);
    };
    SECTION("the error itself") {
        check_failed([&](HostFixture&) {
            throw tadah::hpo::durable_best::Error(kCause);
        });
    }
    SECTION("a later candidate's error arrives first: the first failure is "
            "the cause") {
        check_failed([&](HostFixture&) {
            throw tadah::hpo::durable_best::Error("a later candidate's error");
        });
    }
    SECTION("a normal end after a policy swallowed the error") {
        check_failed([&](HostFixture&) {});
    }
    SECTION("a signal stop that outranked nothing") {
        check_failed([&](HostFixture&) {
            throw tadah::core::ShutdownRequested{};
        });
    }
    SECTION("a stagnation stop that no RESTART consumed") {
        check_failed([&](HostFixture&) { throw Host::StagnationStop{}; });
    }
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

/* ======================================================================
 * The journal writes a PRESTAGE block without MAXEVAL as MAXEVAL=none: no
 * evaluation limit. It wrote MAXEVAL=0, which is not a budget a block can
 * have (MAXEVAL 0 is an error at its line).
 * ==================================================================== */

TEST_CASE("Termination: the journal writes a PRESTAGE block without MAXEVAL "
          "as MAXEVAL=none", "[HPO][termination][prestage][journal]")
{
    /* TOPK 2 over a GRID exploration (1 + 2^3 points) and an SBPLX
     * refinement; `explore` and `refine` are the stop criteria of each. */
    const auto prestage = [](const std::string& explore,
                             const std::string& refine) {
        return plain_block(
            "    TYPE PRESTAGE\n    TOPK 2\n"
            "    EXPLORE\n      LIB TADAH\n      ALGO GRID\n"
            "      PARAM GRID_POINTS 2\n" + explore + "    ENDEXPLORE\n"
            "    REFINE\n      LIB NLOPT\n      ALGO LN_SBPLX\n" + refine +
            "    ENDREFINE\n");
    };
    SECTION("EXPLORE without MAXEVAL") {
        TaRun run(prestage("", "      MAXEVAL 6\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal,
                   Catch::Contains("    PRESTAGE EXPLORE  : TADAH:GRID  "
                                   "MAXEVAL=none  TOPK=2\n") &&
                   Catch::Contains("    PRESTAGE REFINE   : NLOPT:LN_SBPLX  "
                                   "MAXEVAL=6\n") &&
                   Catch::Contains("PRESTAGE: exploration phase TADAH:GRID "
                                   "(MAXEVAL=none: no evaluation limit)\n"));
        CHECK(journal.find("MAXEVAL=0") == std::string::npos);
    }
    SECTION("REFINE without MAXEVAL") {
        TaRun run(prestage("      MAXEVAL 5\n", "      FTOL_REL 1e-2\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal,
                   Catch::Contains("    PRESTAGE EXPLORE  : TADAH:GRID  "
                                   "MAXEVAL=5  TOPK=2\n") &&
                   Catch::Contains("    PRESTAGE REFINE   : NLOPT:LN_SBPLX  "
                                   "MAXEVAL=none\n") &&
                   Catch::Contains("PRESTAGE: exploration phase TADAH:GRID "
                                   "(MAXEVAL=5)\n"));
        CHECK(journal.find("MAXEVAL=0") == std::string::npos);
    }
}

/* ======================================================================
 * TADAH:GRID honours MAXEVAL: it stops after MAXEVAL points in the order
 * it visits them -- the INIT point, then the grid nodes, the first
 * dimension varying fastest. It used to ignore MAXEVAL without a word and
 * evaluate the whole grid. Without MAXEVAL it still does.
 * ==================================================================== */

/* The GRID messages count what they count: the banner names "1
 * dimension" for one OPTIM dimension (and "3 dimensions" for three), and
 * the end of the whole sweep names the INIT point apart from the grid
 * points ("TADAH:GRID finished: the INIT point and all 3 grid points
 * evaluated"): the INIT point is evaluated first and is no grid point.
 * The cases below pin the MAXEVAL, STOPVAL and MAXTIME forms ("after 5 of
 * 28 evaluations (the INIT point and 27 grid points)").
 * Before the fix: "1 dimensions." and "all 4 grid points evaluated" (the
 * INIT point counted as a grid point), and "after 5 of 28 grid points". */
TEST_CASE("Termination: the TADAH:GRID messages count dimensions and grid "
          "points", "[HPO][termination][grid]")
{
    const std::string one_dim =
        "LOSS L2\nFAILSCORE 1e9\nQFLUSH 1000\nDIGITS 15\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.1 0.9\n";
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
                          "    PARAM GRID_POINTS 3\n"),
              one_dim);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    CHECK(evaluated_points().size() == 4u);
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK_THAT(journal, Catch::Contains(
                            "GRID search: 3 points in each dimension (PARAM "
                            "GRID_POINTS; spacing = width/2), 1 dimension.\n"));
    CHECK_THAT(journal, Catch::Contains("TADAH:GRID finished: the INIT point "
                                        "and all 3 grid points evaluated"));
    CHECK(journal.find("1 dimensions") == std::string::npos);

    TaRun three(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
                            "    PARAM GRID_POINTS 2\n"));
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(three.ctx));
    CHECK(status == 0);
    CHECK_THAT(read_file("hpo_run.log"),
               Catch::Contains("spacing = width/1), 3 dimensions.\n"));
}

TEST_CASE("Termination: TADAH:GRID stops after MAXEVAL points in grid order",
          "[HPO][termination][stop_criteria][grid]")
{
    const std::string grid =
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
        "    PARAM GRID_POINTS 3\n";

    /* The whole grid: the INIT point and the 3^3 nodes of the three OPTIM
     * dimensions. */
    std::vector<Point> full;
    {
        TaRun run(plain_block(grid));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        full = evaluated_points();
        CHECK(full.size() == 28u);
        CHECK(read_file("hpo_run.log").find("GRID: MAXEVAL") ==
              std::string::npos);
    }
    REQUIRE(full.size() == 28u);

    SECTION("MAXEVAL below the grid size: the first MAXEVAL points") {
        TaRun run(plain_block(grid + "    MAXEVAL 5\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const auto pts = evaluated_points();
        CHECK(pts.size() == 5u);
        CHECK(pts == evals_between(full, 1, 5));
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal,
                   Catch::Contains("GRID: MAXEVAL 5 < 28 evaluations (the "
                                   "INIT point, then 27 = 3^3 grid nodes)") &&
                   Catch::Contains("evaluating the first 5 in grid order"));
        /* The journal says which stop ended the sweep, and it is the
         * run's status.
         * Pins: "TADAH:GRID: MAXEVAL reached" (here and below). Before the
         * fix: COMPLETED for a spent MAXEVAL or the whole grid. */
        CHECK_THAT(journal, Catch::Contains("TADAH:GRID stopped: MAXEVAL "
                                            "reached after 5 of 28 "
                                            "evaluations (the INIT point "
                                            "and 27 grid points)"));
        CHECK(journal_status(journal) == "TADAH:GRID: MAXEVAL reached");
    }
    SECTION("MAXEVAL 1: the INIT point only") {
        TaRun run(plain_block(grid + "    MAXEVAL 1\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        const auto pts = evaluated_points();
        CHECK(pts == evals_between(full, 1, 1));
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains("TADAH:GRID stopped: MAXEVAL "
                                            "reached after 1 of 28 "
                                            "evaluations (the INIT point "
                                            "and 27 grid points)"));
        CHECK(journal_status(journal) == "TADAH:GRID: MAXEVAL reached");
    }
    SECTION("MAXEVAL at or above the grid size: the whole grid, no note") {
        for (const std::string maxeval : {"28", "1000"}) {
            INFO("MAXEVAL " << maxeval);
            TaRun run(plain_block(grid + "    MAXEVAL " + maxeval + "\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            CHECK(status == 0);
            CHECK(evaluated_points() == full);
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("GRID: MAXEVAL") == std::string::npos);
            CHECK_THAT(journal, Catch::Contains("TADAH:GRID finished: the "
                                                "INIT point and all 27 grid "
                                                "points evaluated"));
            /* Pins: the whole grid is the status "TADAH:GRID: all grid
             * points evaluated". */
            CHECK(journal_status(journal) ==
                  "TADAH:GRID: all grid points evaluated");
        }
    }
}

/* ======================================================================
 * The startup checks see the spec each phase runs with: MultiStartPolicy
 * derives each start's spec with the functions strategy_phase_specs()
 * (phase_specs.h) derives the startup phases from.
 * ==================================================================== */

TEST_CASE("Termination: each MULTISTART start runs with the MAXEVAL the "
          "startup checks see", "[HPO][termination][multistart][phase_specs]")
{
    /* `init_k` INIT points, the STRATEGY lines `strategy`; the journal's
     * per-start line, and the startup phase's MAXEVAL, must both be
     * `per_start`. */
    const auto multistart = [](std::size_t init_k, const std::string& strategy) {
        return "OPTIMIZER\n  INIT\n    METHOD RANDOM " + std::to_string(init_k) +
               "\n    SEED 3\n  ENDINIT\n  STRATEGY\n    TYPE MULTISTART\n" +
               strategy + "  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const auto startup_maxeval = [](TaRun& run) {
        const tadah::hpo::HPOTargets trg(run.ctx, run.hpo.string());
        const auto phases = tadah::hpo::strategy_phase_specs(trg.spec);
        REQUIRE(phases.size() == 1u);
        CHECK(phases[0].name == "each MULTISTART start");
        return phases[0].spec.stop.maxeval;
    };

    SECTION("MAXEVAL split over N_STARTS") {
        TaRun run(multistart(2, "    LIB DLIB\n    ALGO BFGS\n    MAXEVAL 9\n"
                                "    N_STARTS 2\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains("MULTISTART per-start MAXEVAL = 5 "
                                            "(split from outer MAXEVAL)"));
        CHECK(data_rows("loss.tadah").size() <= 10u);
        CHECK(startup_maxeval(run) == std::optional<std::uint64_t>(5));
    }
    /* The INIT chain yields fewer points than N_STARTS: MAXEVAL is split
     * over the starts that run (3), not over N_STARTS (4). SBPLX spends
     * every evaluation of its 4-evaluation budget. */
    SECTION("MAXEVAL split over the starts the INIT points allow") {
        TaRun run(multistart(3, "    LIB NLOPT\n    ALGO LN_SBPLX\n"
                                "    MAXEVAL 12\n    N_STARTS 4\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains("MULTISTART per-start MAXEVAL = 4 "
                                            "(split from outer MAXEVAL)") &&
                            Catch::Contains("MULTISTART start 3/3"));
        CHECK(data_rows("loss.tadah").size() == 12u);
        CHECK(startup_maxeval(run) == std::optional<std::uint64_t>(4));
    }
    /* BUDGET_PER_START takes precedence over the outer MAXEVAL's split. */
    SECTION("BUDGET_PER_START next to MAXEVAL") {
        TaRun run(multistart(2, "    LIB NLOPT\n    ALGO LN_SBPLX\n"
                                "    MAXEVAL 12\n    N_STARTS 2\n"
                                "    BUDGET_PER_START 3\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains("MULTISTART per-start MAXEVAL = 3 "
                                            "(from BUDGET_PER_START)"));
        CHECK(data_rows("loss.tadah").size() == 6u);
        CHECK(startup_maxeval(run) == std::optional<std::uint64_t>(3));
    }
}

/* ======================================================================
 * TADAH:RANDOM, TADAH:GRID and DLIB:GFS honour MAXTIME and STOPVAL: a run
 * ends after the first evaluation whose loss is at or below STOPVAL, or
 * once MAXTIME seconds have passed (no evaluation starts after that), and
 * the journal and the trailer say which. They used to accept both keys and
 * ignore them, and RANDOM and GFS could not run without MAXEVAL.
 * ==================================================================== */

namespace {

/* The losses of loss.tadah, by evaluation number (1-based). */
std::vector<double> evaluated_losses()
{
    const auto loss = data_rows("loss.tadah");
    std::vector<double> out(loss.size());
    for (const auto& row : loss) {
        const std::size_t eval = std::stoull(row.at(0));
        REQUIRE(eval >= 1);
        REQUIRE(eval <= out.size());
        out[eval - 1] = std::stod(row.back());
    }
    return out;
}

/* The first evaluation after the first one whose loss improves on every
 * earlier loss, and a STOPVAL halfway between the two: the first loss at
 * or below that STOPVAL is that evaluation's. */
struct FirstImprovement {
    std::size_t eval;
    std::string stopval;
};

FirstImprovement first_improvement(const std::vector<double>& losses)
{
    REQUIRE_FALSE(losses.empty());
    double best = losses.front();
    for (std::size_t i = 1; i < losses.size(); ++i) {
        if (losses[i] < best) {
            std::ostringstream os;
            os << std::setprecision(17) << 0.5 * (losses[i] + best);
            return {i + 1, os.str()};
        }
    }
    FAIL("no evaluation improves on the first one");
    return {0, ""};
}

/* The evaluated points and losses of a run of `optimizer`, and its
 * journal. */
struct Reference {
    std::vector<Point> points;
    std::vector<double> losses;
    std::string journal;
};

Reference reference_run(const std::string& optimizer)
{
    TaRun run(optimizer);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    REQUIRE(status == 0);
    return {evaluated_points(), evaluated_losses(), read_file("hpo_run.log")};
}

/* hpo_run() of `run`, which must end within `limit`: a run that ignores
 * its stop would not. */
int run_within(TaRun& run, std::chrono::seconds limit, const std::string& what)
{
    int status = -1;
    hpo_test::finish_within(limit, what, [&] {
        status = tadah::hpo::hpo_run(run.ctx);
    });
    return status;
}

}  // namespace

TEST_CASE("Termination: TADAH:RANDOM stops at the first loss at or below "
          "STOPVAL", "[HPO][termination][stop_criteria][random]")
{
    const std::string random =
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    SEED 11\n";
    const Reference ref = reference_run(plain_block(random + "    MAXEVAL 30\n"));
    REQUIRE(ref.points.size() == 30u);
    /* The reference run spends its MAXEVAL, and the journal and the
     * status say so.
     * Pins: "TADAH:RANDOM: MAXEVAL reached". Before the fix: COMPLETED. */
    CHECK_THAT(ref.journal, Catch::Contains("TADAH:RANDOM stopped: MAXEVAL "
                                            "reached after 30 "
                                            "evaluation(s)"));
    CHECK(journal_status(ref.journal) == "TADAH:RANDOM: MAXEVAL reached");
    const FirstImprovement stop = first_improvement(ref.losses);
    INFO("STOPVAL " << stop.stopval << ", first reached at evaluation "
         << stop.eval);

    /* With MAXEVAL, and on its own: STOPVAL is a stop of its own. */
    for (const std::string maxeval : {"    MAXEVAL 30\n", ""}) {
        INFO("MAXEVAL line: '" << maxeval << "'");
        TaRun run(plain_block(random + maxeval + "    STOPVAL " +
                              stop.stopval + "\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                            "TADAH:RANDOM with STOPVAL"));
        CHECK(status == 0);
        check_run_files();
        CHECK(evaluated_points() == evals_between(ref.points, 1, stop.eval));
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains(
                                "TADAH:RANDOM stopped: STOPVAL reached after " +
                                std::to_string(stop.eval) + " evaluation(s)"));
        CHECK(journal_status(journal) == "TADAH:RANDOM: STOPVAL reached");
    }
}

TEST_CASE("Termination: TADAH:RANDOM stops once MAXTIME has passed",
          "[HPO][termination][stop_criteria][random]")
{
    /* On its own, and next to a MAXEVAL it would take far longer to
     * spend. */
    for (const std::string maxeval : {"", "    MAXEVAL 100000000\n"}) {
        INFO("MAXEVAL line: '" << maxeval << "'");
        TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                              "    SEED 5\n    MAXTIME 1\n" + maxeval));
        int status = -1;
        REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(90),
                                            "TADAH:RANDOM with MAXTIME 1"));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains(
                                "TADAH:RANDOM stopped: MAXTIME exceeded after "));
        CHECK(journal_status(journal) == "TADAH:RANDOM: MAXTIME exceeded");
    }
}

TEST_CASE("Termination: TADAH:GRID stops at the first loss at or below "
          "STOPVAL in grid order", "[HPO][termination][stop_criteria][grid]")
{
    const std::string grid =
        "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
        "    PARAM GRID_POINTS 3\n";
    const Reference ref = reference_run(plain_block(grid));
    REQUIRE(ref.points.size() == 28u);
    const FirstImprovement stop = first_improvement(ref.losses);
    INFO("STOPVAL " << stop.stopval << ", first reached at grid point "
         << stop.eval);

    TaRun run(plain_block(grid + "    STOPVAL " + stop.stopval + "\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "TADAH:GRID with STOPVAL"));
    CHECK(status == 0);
    check_run_files();
    CHECK(evaluated_points() == evals_between(ref.points, 1, stop.eval));
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains(
                            "TADAH:GRID stopped: STOPVAL reached after " +
                            std::to_string(stop.eval) + " of 28 evaluations "
                            "(the INIT point and 27 grid points)"));
    CHECK(journal_status(journal) == "TADAH:GRID: STOPVAL reached");
}

TEST_CASE("Termination: TADAH:GRID stops once MAXTIME has passed",
          "[HPO][termination][stop_criteria][grid]")
{
    /* 1 + 100^3 grid points: far more than a second's evaluations. */
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
                          "    PARAM GRID_POINTS 100\n    MAXTIME 1\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(90),
                                        "TADAH:GRID with MAXTIME 1"));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains("TADAH:GRID stopped: MAXTIME exceeded "
                                        "after ") &&
                        Catch::Contains(" of 1000001 evaluations (the INIT "
                                        "point and 1000000 grid points)"));
    CHECK(journal_status(journal) == "TADAH:GRID: MAXTIME exceeded");
    CHECK(data_rows("loss.tadah").size() < 1000001u);
}

TEST_CASE("Termination: DLIB:GFS stops at the first loss at or below STOPVAL",
          "[HPO][termination][stop_criteria][gfs]")
{
    /* One worker: dlib evaluates one point at a time, so no evaluation
     * runs after the one that reaches STOPVAL. */
    const std::string gfs =
        "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n    THREADS 1\n";
    const Reference ref = reference_run(plain_block(gfs + "    MAXEVAL 30\n"));
    REQUIRE(ref.points.size() == 30u);
    CHECK_THAT(ref.journal, Catch::Contains("DLIB:GFS stopped: MAXEVAL "
                                            "reached after 30 "
                                            "evaluation(s)"));
    /* Pins: "DLIB:GFS: MAXEVAL reached". Before the fix: COMPLETED. */
    CHECK(journal_status(ref.journal) == "DLIB:GFS: MAXEVAL reached");
    const FirstImprovement stop = first_improvement(ref.losses);
    INFO("STOPVAL " << stop.stopval << ", first reached at evaluation "
         << stop.eval);

    for (const std::string maxeval : {"    MAXEVAL 30\n", ""}) {
        INFO("MAXEVAL line: '" << maxeval << "'");
        TaRun run(plain_block(gfs + maxeval + "    STOPVAL " + stop.stopval +
                              "\n"));
        int status = -1;
        REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                            "DLIB:GFS with STOPVAL"));
        CHECK(status == 0);
        check_run_files();
        CHECK(evaluated_points() == evals_between(ref.points, 1, stop.eval));
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains(
                                "DLIB:GFS stopped: STOPVAL reached after " +
                                std::to_string(stop.eval) + " evaluation(s)"));
        CHECK(journal_status(journal) == "DLIB:GFS: STOPVAL reached");
    }
}

TEST_CASE("Termination: DLIB:GFS stops once MAXTIME has passed",
          "[HPO][termination][stop_criteria][gfs]")
{
    for (const std::string maxeval : {"", "    MAXEVAL 100000000\n"}) {
        INFO("MAXEVAL line: '" << maxeval << "'");
        TaRun run(plain_block("    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
                              "    MAXTIME 1\n" + maxeval));
        int status = -1;
        REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(90),
                                            "DLIB:GFS with MAXTIME 1"));
        CHECK(status == 0);
        check_run_files();
        const std::string journal = read_file("hpo_run.log");
        CHECK_THAT(journal, Catch::Contains(
                                "DLIB:GFS stopped: MAXTIME exceeded after "));
        CHECK(journal_status(journal) == "DLIB:GFS: MAXTIME exceeded");
    }
}

TEST_CASE("Termination: the journal says how a run without stop criteria "
          "ends", "[HPO][termination][stop_criteria][journal]")
{
    /* A backend without a stop criterion either ends on its own or is
     * refused at startup (it needs a stop), so "may run indefinitely" no
     * longer holds of any run that starts. */
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
                          "    PARAM GRID_POINTS 2\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal,
               Catch::Contains("(none set -- TADAH:GRID ends on its own: a "
                               "convergence test, a finite schedule or "
                               "limits of its own)") &&
               !Catch::Contains("may run indefinitely"));
    CHECK_THAT(journal, Catch::Contains("TADAH:GRID finished: the INIT "
                                        "point and all 8 grid points "
                                        "evaluated"));
}

/* ======================================================================
 * MAXTIME is a stop only while another evaluation would start: a run whose
 * last evaluation spends MAXEVAL (or the grid) ends on MAXEVAL, with the
 * status "<backend>: MAXEVAL reached" (it was COMPLETED), however long that
 * evaluation took -- as DLIB:GFS reports it.
 * ==================================================================== */

TEST_CASE("Termination: TADAH:RANDOM and TADAH:GRID report MAXEVAL when "
          "their last evaluation spends it after MAXTIME has passed",
          "[HPO][termination][stop_criteria][random][grid]")
{
    /* MAXTIME 0.0001 s has passed after the first evaluation. */
    struct Case { std::string what, body, maxeval_line, maxtime_line; };
    for (const auto& c : std::vector<Case>{
             {"TADAH:RANDOM",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    SEED 1\n",
              "TADAH:RANDOM stopped: MAXEVAL reached after 1 evaluation(s)",
              "TADAH:RANDOM stopped: MAXTIME exceeded after 1 evaluation(s)"},
             {"TADAH:GRID",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO GRID\n"
              "    PARAM GRID_POINTS 2\n",
              "TADAH:GRID stopped: MAXEVAL reached after 1 of 9 evaluations "
              "(the INIT point and 8 grid points)",
              "TADAH:GRID stopped: MAXTIME exceeded after 1 of 9 evaluations "
              "(the INIT point and 8 grid points)"}}) {
        INFO(c.what);
        {
            TaRun run(plain_block(c.body + "    MAXEVAL 1\n"
                                           "    MAXTIME 0.0001\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(90),
                                                c.what + " MAXEVAL 1"));
            CHECK(status == 0);
            check_run_files();
            CHECK(data_rows("loss.tadah").size() == 1u);
            const std::string journal = read_file("hpo_run.log");
            CHECK_THAT(journal, Catch::Contains(c.maxeval_line) &&
                                !Catch::Contains("MAXTIME exceeded"));
            /* Pins: "<backend>: MAXEVAL reached". Before the fix:
             * COMPLETED. */
            CHECK(journal_status(journal) == c.what + ": MAXEVAL reached");
        }
        /* With evaluations left, the same MAXTIME ends the run after the
         * first one. */
        {
            TaRun run(plain_block(c.body + "    MAXEVAL 5\n"
                                           "    MAXTIME 0.0001\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(90),
                                                c.what + " MAXEVAL 5"));
            CHECK(status == 0);
            CHECK(data_rows("loss.tadah").size() == 1u);
            const std::string journal = read_file("hpo_run.log");
            CHECK_THAT(journal, Catch::Contains(c.maxtime_line));
            CHECK(journal_status(journal) == c.what + ": MAXTIME exceeded");
        }
    }
}

/* ======================================================================
 * An active RESTART (AFTER_STAGNATION and MAX > 0) is a stop of every
 * phase it wraps: once its MAX restarts are used, the next
 * AFTER_STAGNATION evaluations without a new best end the run. A backend
 * that needs a stop runs under it without one of its own -- at startup,
 * and when its driver checks its spec again at run time (the PRESTAGE
 * exploration runs its EXPLORE block's spec, which has no RESTART).
 * ==================================================================== */

TEST_CASE("Termination: the RESTART stagnation stop ends a search that has "
          "no stop of its own", "[HPO][termination][restart][stop_criteria]")
{
    const std::string restart =
        "    RESTART\n      AFTER_STAGNATION 20\n      MAX 1\n"
        "      METHOD RANDOM 1\n      SEED 5\n    ENDRESTART\n";
    struct Case { std::string what, body; bool ends_on_stagnation; };
    for (const auto& c : std::vector<Case>{
             {"NLOPT:GN_DIRECT_L",
              "    TYPE PLAIN\n    LIB NLOPT\n    ALGO GN_DIRECT_L\n",
              true},
             {"TADAH:RANDOM",
              "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
              "    SEED 3\n",
              true},
             /* LD_MMA, a local search whose only convergence test is the
              * tolerances (it was GN_DIRECT_L, a global search, which an
              * INNER block no longer takes; before that fix: passes as
              * well, a pin). */
             {"NLOPT:AUGLAG over an INNER NLOPT:LD_MMA",
              "    TYPE PLAIN\n    LIB NLOPT\n    ALGO AUGLAG\n"
              "    INNER\n      LIB NLOPT\n      ALGO LD_MMA\n"
              "    ENDINNER\n",
              true},
             /* The exploration ends at its first stagnation and hands its
              * best point to the refinement, which ends on its own. */
             {"the PRESTAGE exploration NLOPT:GN_CRS2_LM",
              "    TYPE PRESTAGE\n    LIB NLOPT\n    ALGO LN_SBPLX\n"
              "    MAXEVAL 15\n    TOPK 1\n    EXPLORE\n      LIB NLOPT\n"
              "      ALGO GN_CRS2_LM\n      SEED 7\n    ENDEXPLORE\n",
              false}}) {
        DYNAMIC_SECTION(c.what) {
            TaRun run(plain_block(c.body + restart));
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(180),
                                                c.what + " under RESTART"));
            CHECK(status == 0);
            check_run_files();
            const std::string journal = read_file("hpo_run.log");
            if (c.ends_on_stagnation) {
                CHECK_THAT(journal, Catch::Contains("RESTART 1/1"));
                CHECK(journal_status(journal) ==
                      restarts_used_status(20, 1));
                /* The journal says what ends a run without a stop of its
                 * own. */
                if (c.what == "NLOPT:GN_DIRECT_L")
                    CHECK_THAT(journal,
                               Catch::Contains(
                                   "(none set -- the RESTART stagnation stop "
                                   "ends NLOPT:GN_DIRECT_L: once its MAX 1 "
                                   "restart(s) are used, AFTER_STAGNATION 20 "
                                   "evaluations without a new best)"));
            } else {
                CHECK_THAT(journal,
                           Catch::Contains("REFINEMENT: refining top-1"));
            }
        }
    }
}

/* ======================================================================
 * Every driver records why it ended (HPOLog::set_status, run_end.h): the
 * trailer is "<LIB>:<ALGO>: <reason>" (NLopt and ASA keep their own
 * strings) and the journal's stop line gives the numbers. DLIB:BFGS
 * recorded a status only when a gradient was not finite: a spent MAXEVAL
 * and its convergence tests ended the run COMPLETED.
 * ==================================================================== */

/* DLIB:BFGS ends on MAXEVAL or on its stop strategy, which is
 * objective_delta_stop_strategy(FTOL_ABS, 1e-7 by default): the loss
 * changed by less than FTOL_ABS from one iteration to the next; with
 * PARAM tol_gradinf, gradient_norm_stop_strategy: the L2 norm of the
 * gradient (dlib's length(g)) fell below it.
 * Pins (NUMERICAL central differences, three dimensions):
 *   MAXEVAL 3 -- 3 evaluations, "DLIB:BFGS -- MAXEVAL=3 reached" (the
 *     MAXEVAL stop line, unchanged) and the status "DLIB:BFGS: MAXEVAL
 *     reached";
 *   MAXEVAL 200 -- 44 evaluations, the stop line "DLIB:BFGS stopped:
 *     FTOL_ABS reached after 44 evaluation(s): the loss changed by less
 *     than FTOL_ABS = 1e-07 (default) from one iteration to the next", no
 *     MAXEVAL line, the status "DLIB:BFGS: FTOL_ABS reached" (51 until
 *     2026-10-12: at the bound SGRID2B[2] = 4, where its step cannot move
 *     the point, its line search evaluated the point again and BFGS took
 *     its gradient there again, 7 evaluations);
 *   FTOL_ABS 1e-3 -- 14 evaluations, "FTOL_ABS = 0.001" without
 *     "(default)";
 *   PARAM tol_gradinf 1e6 -- 1 + 2 x 3 = 7 evaluations (the loss at the
 *     start and its gradient's stencil, which takes that loss as its
 *     centre), the stop line "DLIB:BFGS stopped: tol_gradinf reached after
 *     7 evaluation(s): the largest component of the gradient fell below
 *     PARAM tol_gradinf = 1e+06" ("the L2 norm of the gradient" until
 *     tol_gradinf tested the largest component) and the status "DLIB:BFGS:
 *     tol_gradinf reached".
 * Before the fix: the same evaluations, no stop line for a convergence
 * test, and the trailer COMPLETED in every case. */
TEST_CASE("Termination: DLIB:BFGS records why it ended",
          "[HPO][termination][stop_criteria][stop_status]")
{
    const std::string bfgs = "    TYPE PLAIN\n    LIB DLIB\n    ALGO BFGS\n";
    struct Case {
        std::string what, lines;
        std::size_t evals;
        std::string status, stop_line;
    };
    for (const auto& c : std::vector<Case>{
             {"MAXEVAL 3", "    MAXEVAL 3\n", 3u,
              "DLIB:BFGS: MAXEVAL reached",
              "DLIB:BFGS -- MAXEVAL=3 reached; preserving best-so-far"},
             {"the default FTOL_ABS", "    MAXEVAL 200\n", 44u,
              "DLIB:BFGS: FTOL_ABS reached",
              "DLIB:BFGS stopped: FTOL_ABS reached after 44 evaluation(s): "
              "the loss changed by less than FTOL_ABS = 1e-07 (default) "
              "from one iteration to the next"},
             {"FTOL_ABS 1e-3", "    MAXEVAL 200\n    FTOL_ABS 1e-3\n", 14u,
              "DLIB:BFGS: FTOL_ABS reached",
              "DLIB:BFGS stopped: FTOL_ABS reached after 14 evaluation(s): "
              "the loss changed by less than FTOL_ABS = 0.001 from one "
              "iteration to the next"},
             {"PARAM tol_gradinf 1e6",
              "    MAXEVAL 200\n    PARAM tol_gradinf 1e6\n", 7u,
              "DLIB:BFGS: tol_gradinf reached",
              "DLIB:BFGS stopped: tol_gradinf reached after 7 "
              "evaluation(s): the largest component of the gradient fell "
              "below PARAM tol_gradinf = 1e+06"}}) {
        DYNAMIC_SECTION(c.what) {
            TaRun run(plain_block(bfgs + c.lines));
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            CHECK(status == 0);
            check_run_files();
            CHECK(data_rows("loss.tadah").size() == c.evals);
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal, c.stop_line) == 1u);
            if (c.evals != 3u)
                CHECK(journal.find("MAXEVAL=200 reached") ==
                      std::string::npos);
            CHECK(journal_status(journal) == c.status);
        }
    }
}

/* Every registered backend (the optimizer_caps registry; CERES when it is
 * built; NLOPT:LN_NEWUOA_BOUND is refused and not registered) records why
 * its run ended under every STRATEGY TYPE and under RESTART: a run that
 * ends normally has a status of its own, never one nothing recorded. A
 * driver that forgot it would now fail the run (drive_run's internal
 * error), so this sweep is the guard that no backend and strategy does.
 * Each run spends a small MAXEVAL on the three-dimensional fixture:
 *   PLAIN       the backend alone;
 *   MULTISTART  two starts from two INIT points (the backends that run
 *               from each start, runs_from_each_start(); a global one is
 *               refused at startup);
 *   PRESTAGE    a TADAH:RANDOM exploration, then the backend as the
 *               refinement of TOPK 2 candidates (TOPK 1 for a global
 *               backend; one that does not start at the point it is given,
 *               OptimizerCaps::starts_at_x0, is refused at startup: until
 *               that refusal the sweep ran
 *               NLopt's StoGO, DIRECT and AGS as the refinement too, and
 *               they recorded their end there);
 *   RESTART     PLAIN under an active RESTART (AFTER_STAGNATION 4, MAX 1):
 *               the backend's status, or RESTART's own once its restart is
 *               used.
 * Pins: exit status 0 and a status that is neither empty, nor COMPLETED,
 * nor "(no status recorded)".
 * Before the fix: TADAH:RANDOM, TADAH:GRID, the ANNEAL family, DLIB:GFS,
 * DLIB:BFGS and CERES:* ended COMPLETED (NLopt and ASA had their own
 * statuses). */
namespace {

/* The OPTIMIZER block of `c` under `strategy` ("PLAIN", "MULTISTART",
 * "PRESTAGE", "RESTART"). */
std::string registry_block(const tadah::hpo::OptimizerCaps& c,
                           const std::string& strategy)
{
    std::string backend = "    LIB " + c.lib + "\n    ALGO " + c.algo +
                          "\n    MAXEVAL 12\n";
    if (c.reads_seed) backend += "    SEED 3\n";
    if (c.lib == "TADAH" && c.algo == "GRID")
        backend += "    PARAM GRID_POINTS 2\n";
    if (c.lib == "DLIB" && c.algo == "GFS") backend += "    THREADS 1\n";
    /* MLSL and AUGLAG without a default local optimiser need an INNER
     * block (require_inner_backend). */
    if (c.accepts_inner && c.default_inner == nullptr)
        backend += "    INNER\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
                   "    ENDINNER\n";
    std::string body;
    std::string init;
    if (strategy == "PLAIN") {
        body = "    TYPE PLAIN\n" + backend;
    } else if (strategy == "MULTISTART") {
        init = "  INIT\n    METHOD RANDOM 2\n    SEED 4\n  ENDINIT\n";
        body = "    TYPE MULTISTART\n    N_STARTS 2\n" + backend;
    } else if (strategy == "PRESTAGE") {
        body = "    TYPE PRESTAGE\n    TOPK " +
               std::string(tadah::hpo::runs_from_each_start(c) ? "2" : "1") +
               "\n    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
               "      MAXEVAL 4\n      SEED 2\n    ENDEXPLORE\n" + backend;
    } else {
        body = "    TYPE PLAIN\n" + backend +
               "    RESTART\n      AFTER_STAGNATION 4\n      MAX 1\n"
               "      METHOD RANDOM 1\n      SEED 5\n    ENDRESTART\n";
    }
    return "OPTIMIZER\n" + init + "  STRATEGY\n" + body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

void check_every_backend_records_its_end(const std::string& strategy)
{
    const auto all = tadah::hpo::filter_optimizer_caps(
        [](const tadah::hpo::OptimizerCaps&) { return true; });
    REQUIRE(all.size() > 40u);
    for (const tadah::hpo::OptimizerCaps* c : all) {
#ifndef TADAH_USE_CERES
        if (c->lib == "CERES") continue;   // registered, not built
#endif
        if (strategy == "MULTISTART" && !tadah::hpo::runs_from_each_start(*c))
            continue;                      // refused at startup
        if (strategy == "PRESTAGE" && !c->starts_at_x0)
            continue;                      // refused at startup
        DYNAMIC_SECTION(c->lib << ":" << c->algo) {
            TaRun run(registry_block(*c, strategy));
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(
                                run, std::chrono::seconds(120),
                                c->lib + ":" + c->algo + " " + strategy));
            CHECK(status == 0);
            const std::string trailer = journal_status(read_file("hpo_run.log"));
            INFO("trailer: " << trailer);
            CHECK_FALSE(trailer.empty());
            CHECK(trailer != "COMPLETED");
            CHECK(trailer != "(no status recorded)");
            CHECK(trailer != "(no RESULT section)");
        }
    }
}

}  // namespace

TEST_CASE("End status: every registered backend records why it ended under "
          "STRATEGY TYPE PLAIN", "[HPO][termination][stop_status][registry]")
{
    check_every_backend_records_its_end("PLAIN");
}

TEST_CASE("End status: every registered backend records why it ended under "
          "STRATEGY TYPE MULTISTART",
          "[HPO][termination][stop_status][registry]")
{
    check_every_backend_records_its_end("MULTISTART");
}

TEST_CASE("End status: every registered backend records why it ended as the "
          "PRESTAGE refinement", "[HPO][termination][stop_status][registry]")
{
    check_every_backend_records_its_end("PRESTAGE");
}

TEST_CASE("End status: every registered backend records why it ended under "
          "RESTART", "[HPO][termination][stop_status][registry]")
{
    check_every_backend_records_its_end("RESTART");
}

/* ======================================================================
 * Runs in which no evaluation produced a usable loss, and DLIB:GFS with
 * failed evaluations.
 *
 * A usable loss is a feasible one (loss_is_feasible: finite and below the
 * failure band); only a usable loss can become the incumbent. A run none
 * of whose evaluations produced one writes no best_pot.tadah. It used to
 * end with status 0 and "COMPLETED" like a successful run, and its
 * journal's DIAGNOSTICS said "[ok] Every evaluation produced a usable
 * loss" (they counted NaN/Inf clampings and graded failures only). Such a
 * run now ends with status 1, an ERROR line and the trailer "STOPPED -- no
 * evaluation produced a usable loss", unless a signal (128 + N) or an
 * error (1, "FAILED -- ...") ended it first.
 *
 * DLIB:GFS used to hand dlib the failed evaluations' loss, FAILSCORE, by
 * default the largest double. find_min_global maximises the negated loss,
 * and its trust-region step takes as best sample the first whose value
 * exceeds -1e300: with every sample at or above 1e300 there is none, and
 * it indexed an empty vector (SIGSEGV). That happens at the first
 * trust-region step, after dims + 2 evaluations that all failed, whether
 * or not later evaluations would have produced a usable loss.
 * ==================================================================== */

namespace {

/* The fixture's validation ERMSE is about 9.75-10 meV/atom everywhere in
 * the box, so a skip cap of 1e-30 fails every evaluation (its loss is
 * FAILSCORE). No FAILSCORE line: the default, the largest double. */
const std::string kAllFailRoot =
    "LOSS L2\n"
    "QFLUSH 1000\n"
    "DIGITS 15\n"
    "PC_ERMSE 0 1 --skip-above 1e-30\n"
    "OPTIM CGRID2B (1) 0.1 0.9\n"
    "OPTIM CGRID2B (2) 1.0 2.0\n"
    "OPTIM SGRID2B (2) 4.0 6.0\n";

/* The default FAILSCORE as loss.tadah prints it (DIGITS 15). */
const std::string kDefaultFailScoreRow = "1.797693134862316e+308";

const std::string kNoUsableLossStatus =
    "STOPPED -- no evaluation produced a usable loss";
const std::string kNoUsableLossError =
    "no evaluation produced a usable loss; no best_pot.tadah was written";

/* The journal's note that DLIB:GFS gives dlib a stand-in for a failed
 * evaluation: once per DLIB:GFS search that has a failed evaluation. */
const std::string kGfsStandInNote =
    "DLIB:GFS: dlib is given a finite stand-in for each failed evaluation";

/* The "Unusable losses" count of the journal's DIAGNOSTICS, or -1. */
long unusable_losses(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(Unusable losses\s*: (\d+))")))
        return std::stol(m[1]);
    return -1;
}

/* What a run of `evals` evaluations that all failed leaves behind: one
 * loss.tadah row per evaluation, numbered 1, 2, ..., each `fail_row`; no
 * best_pot.tadah and no improvement row; DIAGNOSTICS that count every
 * evaluation as unusable. The trailer is `status`: the no-usable-loss
 * one, or the one of a signal or error that ended the run first. */
void check_no_usable_loss(std::size_t evals, const std::string& fail_row,
                          const std::string& status)
{
    const auto loss = data_rows("loss.tadah");
    CHECK(loss.size() == evals);
    CHECK(loss.size() == Host::g_iter_value());
    for (std::size_t i = 0; i < loss.size(); ++i) {
        INFO("loss.tadah row " << i);
        CHECK(std::stoull(loss[i].at(0)) == i + 1);
        CHECK(loss[i].back() == fail_row);
    }
    CHECK_FALSE(fs::exists("best_pot.tadah"));
    CHECK(data_rows("best_params.tadah").empty());

    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == status);
    CHECK(std::regex_search(
        journal, std::regex(R"(Best potential saved\s*: \(none written\))")));
    /* The ERROR line belongs to the no-usable-loss ending only. */
    CHECK((journal.find("ERROR     " + kNoUsableLossError) !=
           std::string::npos) == (status == kNoUsableLossStatus));
    CHECK(journal.find("[ok] Every evaluation produced a usable loss") ==
          std::string::npos);
    CHECK(journal.find("[!] No evaluation produced a usable loss.") !=
          std::string::npos);
    CHECK(unusable_losses(journal) == static_cast<long>(evals));
}

const std::string kRandomAllFail = plain_block(
    "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 20\n"
    "    SEED 3\n");

}  // namespace

TEST_CASE("No usable loss: a TADAH:RANDOM run whose every evaluation fails "
          "exits with status 1 and says so",
          "[HPO][termination][no_usable_loss]")
{
    TaRun run(kRandomAllFail, kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "an all-failed TADAH:RANDOM run"));
    CHECK(status == 1);
    check_no_usable_loss(20, kDefaultFailScoreRow, kNoUsableLossStatus);
    /* The optimiser's own ending is still in the progress section. */
    CHECK_THAT(read_file("hpo_run.log"),
               Catch::Contains("TADAH:RANDOM stopped: MAXEVAL reached after "
                               "20 evaluation(s)"));
}

TEST_CASE("No usable loss: evaluations that throw count as unusable losses",
          "[HPO][termination][no_usable_loss]")
{
    /* OALGO 2 (QR) with LAMBDA 0 refuses a design matrix with fewer
     * non-zero data rows than weights: the one training structure, energy
     * only, gives one row for the two weights of D2_BP 2, so every training
     * throws, and the host records each evaluation as FAILSCORE. (The
     * startup checks read no descriptors and do not refuse it; OALGO 3
     * with BIAS true and LAMBDA != 0, which every fit refuses too, is
     * refused when the run starts.) The journal used to follow "100.0% of
     * evaluations produced a non-finite loss" with "[ok] Every evaluation
     * produced a usable loss".
     * Pins also: DIAGNOSTICS counts the six evaluations under
     * "Evaluations that threw" and none as a NaN/Inf clamping (no loss
     * was non-finite), and the LOSS FUNCTION section's Fail score line
     * names every failure that scores it.
     * Before the fix: the journal read "NaN/Inf clampings : 6 (100.0%)"
     * and "100.0% of evaluations produced a non-finite loss that was
     * clamped", had no "Evaluations that threw" line, and its Fail score
     * line read "(returned when an evaluation produces NaN/Inf)"; this
     * case asserted the non-finite wording (it pinned the defect). */
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                          "    MAXEVAL 6\n    SEED 3\n"));
    run.ctx.storage_update("OALGO", {"2"});
    run.ctx.storage_update("LAMBDA", {"0"});
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "a TADAH:RANDOM run whose "
                                        "evaluations throw"));
    CHECK(status == 1);
    check_no_usable_loss(6, "1.000000000000000e+09", kNoUsableLossStatus);
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains("eval-thread caught throw"));
    CHECK_THAT(journal,
               Catch::Contains("Evaluations that threw  : 6  (100.0%)"));
    CHECK_THAT(journal, Catch::Contains("[!] 6 evaluation(s) raised an "
                                        "exception and scored FAIL_SCORE"));
    CHECK_THAT(journal, Catch::Contains("NaN/Inf clampings       : 0  (0.0%)"));
    CHECK_THAT(journal, !Catch::Contains("non-finite loss"));
    /* The LOSS FUNCTION section says what scores FAILSCORE in this run,
     * which has no PC_LAMMPS or PC_OBSERVABLE term (it said "returned when
     * an evaluation produces NaN/Inf", then "... a failed script ...", then
     * "... a failed PC_LAMMPS simulation ..." in every run;
     * FAILSCORE_GRADING OFF here). */
    CHECK_THAT(journal, Catch::Contains(
                            "1.0000e+09  (the loss of a failed evaluation: a "
                            "NaN/Inf loss, an exception, a skip cap or a gate "
                            "reject; the ceiling of every loss)"));
}

TEST_CASE("DLIB:GFS: a run whose every evaluation fails with the default "
          "FAILSCORE reaches MAXEVAL",
          "[HPO][termination][no_usable_loss][gfs]")
{
    /* It used to end in SIGSEGV at the first trust-region step, after
     * five failed evaluations of this three-dimensional search. */
    for (const char* threads : {"1", "4"}) {
        DYNAMIC_SECTION("THREADS " << threads) {
            TaRun run(plain_block(std::string(
                          "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
                          "    MAXEVAL 24\n    THREADS ") + threads + "\n"),
                      kAllFailRoot);
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(
                                run, std::chrono::seconds(120),
                                "an all-failed DLIB:GFS run"));
            CHECK(status == 1);
            check_no_usable_loss(24, kDefaultFailScoreRow,
                                 kNoUsableLossStatus);
            const std::string journal = read_file("hpo_run.log");
            CHECK_THAT(journal, Catch::Contains("DLIB:GFS stopped: MAXEVAL "
                                                "reached after 24 "
                                                "evaluation(s)"));
            CHECK(count_of(journal, kGfsStandInNote) == 1u);
        }
    }
}

TEST_CASE("DLIB:GFS: each search notes its stand-in for failed evaluations "
          "once and a RESTART round is a new search",
          "[HPO][termination][no_usable_loss][gfs][restart]")
{
    /* Every evaluation fails, so no evaluation is a new best and the
     * stagnation window closes after 5 evaluations of each round: round 0
     * and RESTART 1/1 are two DLIB:GFS searches, each with its own
     * stand-ins, and then the RESTART budget ends a run that has no usable
     * loss. */
    TaRun run(plain_block(
                  "    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
                  "    MAXEVAL 400\n    THREADS 1\n"
                  "    RESTART\n      AFTER_STAGNATION 5\n      MAX 1\n"
                  "      METHOD RANDOM 1\n      SEED 7\n    ENDRESTART\n"),
              kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "an all-failed DLIB:GFS run with "
                                        "RESTART"));
    CHECK(status == 1);
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains("RESTART 1/1"));
    CHECK_THAT(journal, Catch::Contains("RESTART: stagnation reached, max "
                                        "restarts (1) used; stopping"));
    CHECK(journal_status(journal) == kNoUsableLossStatus);
    CHECK(count_of(journal, kGfsStandInNote) == 2u);
    check_no_usable_loss(data_rows("loss.tadah").size(), kDefaultFailScoreRow,
                         kNoUsableLossStatus);
}

TEST_CASE("DLIB:GFS: a run without failed evaluations hands dlib its losses "
          "and notes no stand-in", "[HPO][termination][gfs]")
{
    TaRun run(plain_block("    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
                          "    MAXEVAL 12\n    THREADS 1\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "a DLIB:GFS run without failures"));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    REQUIRE(unusable_losses(journal) == 0);
    CHECK(count_of(journal, kGfsStandInNote) == 0u);
    /* Pins: "DLIB:GFS: MAXEVAL reached". Before the fix: COMPLETED. */
    CHECK(journal_status(journal) == "DLIB:GFS: MAXEVAL reached");
    check_run_files();
}

TEST_CASE("DLIB:GFS: a run whose first evaluations all fail goes on to "
          "usable losses and ends as usual",
          "[HPO][termination][gfs]")
{
    /* One dimension, CGRID2B[2] in [1, 2]. On this fixture the validation
     * ERMSE falls from 10 meV/atom at 1.0 to 9.76 at 2.0, but only in the
     * sixth decimal up to 1.4: 9.9999994 at 1.25, 9.9999974 at 1.375,
     * 9.9999944 at 1.4, 9.99989 at 1.5. The skip cap 9.99999 fails the box
     * below a point between 1.4 and 1.5 and leaves the rest usable. The
     * three starting points -- Sobol points 2 and 3 (SEED 1 selects block 1
     * of METHOD SOBOL 2; the chain hands its first method the SEED itself),
     * 1.25 and 1.375, and the configuration's 1.4 -- all fail: dims + 2
     * failed evaluations reach dlib's first trust-region step, which used
     * to end the run in SIGSEGV. GFS picks some points at random, and which
     * ones depends on whether an evaluation has finished when it asks for
     * the next point, so the usable half of the box is wide enough for a
     * run of 40 evaluations to reach it whatever GFS picks. */
    TaRun run(
        "OPTIMIZER\n"
        "  INIT\n    METHOD SOBOL 2\n    METHOD CONFIG\n    SEED 1\n"
        "  ENDINIT\n"
        "  STRATEGY\n    TYPE PLAIN\n    LIB DLIB\n    ALGO GFS\n"
        "    MAXEVAL 40\n    THREADS 1\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\n",
        "LOSS L2\n"
        "QFLUSH 1000\n"
        "DIGITS 15\n"
        "PC_ERMSE 0 1 --skip-above 9.99999\n"
        "OPTIM CGRID2B (2) 1.0 2.0\n");
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "a DLIB:GFS run that starts with "
                                        "failures"));
    CHECK(status == 0);
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains("DLIB:GFS seeding with 3 starting "
                                        "points"));
    /* Pins: "DLIB:GFS: MAXEVAL reached". Before the fix: COMPLETED. */
    CHECK(journal_status(journal) == "DLIB:GFS: MAXEVAL reached");

    /* Every row is a usable loss (an ERMSE of at most 9.99999, so a loss of
     * at most 99.9998) or the FAILSCORE the evaluation returned: whatever
     * dlib is told about a failure, the files record FAILSCORE. */
    const auto loss = data_rows("loss.tadah");
    REQUIRE(loss.size() == 40u);
    std::size_t failed = 0;
    for (std::size_t i = 0; i < loss.size(); ++i) {
        INFO("loss.tadah row " << i + 1 << ": " << loss[i].back());
        if (loss[i].back() == kDefaultFailScoreRow) {
            ++failed;
        } else {
            CHECK(loss_value(loss[i].back()) <= 99.9998);
        }
    }
    for (std::size_t i = 0; i < 3; ++i) {
        INFO("starting point " << i + 1 << " must fail");
        CHECK(loss[i].back() == kDefaultFailScoreRow);
    }
    CHECK(failed < loss.size());
    CHECK(unusable_losses(journal) == static_cast<long>(failed));
    CHECK(journal.find("[ok] Every evaluation produced a usable loss") ==
          std::string::npos);
    CHECK(count_of(journal, kGfsStandInNote) == 1u);
    check_run_files();
}

/* The body records its status, as every optimiser does; without a usable
 * loss the run still ends as a failed one. Before the fix: passes as well
 * (a pin; the body recorded no status). */
TEST_CASE("No usable loss: drive_run ends a normal end without a usable "
          "loss with status 1", "[HPO][termination][no_usable_loss]"
          "[run_driver]")
{
    HostFixture fx(kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            fx.log().set_status("TEST: done");
        }, fx.host, fx.log()));
    CHECK(status == 1);
    check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                         kNoUsableLossStatus);
}

TEST_CASE("No usable loss: drive_run ends a stagnation stop without a "
          "usable loss with status 1", "[HPO][termination][no_usable_loss]"
          "[run_driver]")
{
    HostFixture fx(kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw Host::StagnationStop{};
        }, fx.host, fx.log()));
    CHECK(status == 1);
    check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                         kNoUsableLossStatus);
}

TEST_CASE("No usable loss: an error still ends the run with its own "
          "trailer", "[HPO][termination][no_usable_loss][run_driver]")
{
    HostFixture fx(kAllFailRoot);
    CHECK_THROWS_WITH(tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw std::runtime_error("optimiser broke");
        }, fx.host, fx.log()), "optimiser broke");
    check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                         "FAILED -- optimiser broke");
}

TEST_CASE("No usable loss: a shutdown still ends the run with the SHUTDOWN "
          "trailer", "[HPO][termination][no_usable_loss][run_driver]")
{
    HostFixture fx(kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            throw tadah::core::ShutdownRequested{};
        }, fx.host, fx.log()));
    /* No real signal in this process (the SIGINT case is
     * [.shutdown_no_usable_loss]). */
    CHECK(status == tadah::hpo::shutdown_exit_status(
                        tadah::core::SignalHandler::signal_seen()));
    check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                         "SHUTDOWN (" + tadah::core::SignalHandler::reason() +
                             ")");
}

TEST_CASE("No usable loss: one usable loss among failures ends the run as "
          "before", "[HPO][termination][no_usable_loss][run_driver]")
{
    /* The fixture's points run CGRID2B[2] from 1.875 down to 1.125: the
     * first ones pass the 9.9 meV/atom skip cap, the last ones fail it. */
    HostFixture fx(
        "LOSS L2\n"
        "QFLUSH 1000\n"
        "DIGITS 15\n"
        "PC_ERMSE 0 1 --skip-above 9.9\n"
        "OPTIM CGRID2B (1) 0.1 0.9\n"
        "OPTIM CGRID2B (2) 1.0 2.0\n"
        "OPTIM SGRID2B (2) 4.0 6.0\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
        [&] {
            fx.evaluate(kDriverEvals);
            fx.log().set_status("TEST: done");
        }, fx.host, fx.log()));
    CHECK(status == 0);
    const auto loss = data_rows("loss.tadah");
    REQUIRE(loss.size() == kDriverEvals);
    std::size_t failed = 0;
    for (const auto& row : loss)
        if (row.back() == kDefaultFailScoreRow) ++failed;
    INFO("failed evaluations: " << failed);
    REQUIRE(failed > 0);
    REQUIRE(failed < kDriverEvals);
    const std::string journal = read_file("hpo_run.log");
    /* The body's status (it used to record none: COMPLETED). Before the
     * fix: passes as well (a pin). */
    CHECK(journal_status(journal) == "TEST: done");
    CHECK(journal.find(kNoUsableLossError) == std::string::npos);
    CHECK(unusable_losses(journal) == static_cast<long>(failed));
    CHECK(journal.find("[ok] Every evaluation produced a usable loss") ==
          std::string::npos);
    check_run_files();
}

/* Every optimiser records why its search ended (HPOLog::set_status,
 * run_end.h). A normal end without a recorded status is an internal error:
 * a driver that forgot to say why it ended. It used to end the run with
 * status 0 and the trailer COMPLETED, a status nothing recorded.
 * Pins: drive_run throws std::logic_error "internal error: the optimiser
 * ended without recording why it ended", the trailer starts "FAILED -- "
 * and names it, and the queued results are still flushed (every row, the
 * incumbent in best_pot.tadah). The internal error outranks a run without
 * a usable loss: that run ends FAILED too, not "STOPPED -- no evaluation
 * produced a usable loss".
 * Before the fix: status 0 (1 without a usable loss) and the trailer
 * COMPLETED (the no-usable-loss trailer without one). */
TEST_CASE("Termination: drive_run refuses a normal end that recorded no "
          "status", "[HPO][termination][run_driver][stop_status]")
{
    const std::string kWhat =
        "internal error: the optimiser ended without recording why it ended";
    SECTION("with a usable loss") {
        HostFixture fx;
        CHECK_THROWS_MATCHES(
            tadah::hpo::drive_run([&] { fx.evaluate(kDriverEvals); },
                                  fx.host, fx.log()),
            std::logic_error, Catch::Matchers::Message(kWhat));
        CHECK_THAT(journal_status(read_file("hpo_run.log")),
                   Catch::StartsWith("FAILED -- " + kWhat));
        CHECK(data_rows("loss.tadah").size() == kDriverEvals);
        check_run_files();
    }
    SECTION("without a usable loss") {
        HostFixture fx(kAllFailRoot);
        CHECK_THROWS_MATCHES(
            tadah::hpo::drive_run([&] { fx.evaluate(kDriverEvals); },
                                  fx.host, fx.log()),
            std::logic_error, Catch::Matchers::Message(kWhat));
        check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                             "FAILED -- " + kWhat);
    }
}

namespace {

/* A host that records when drive_run() asks it whether it produced a
 * usable loss -- the question of the run's one cross-rank collective
 * (detail::run_has_usable_loss) -- and when it drains its results. */
struct AgreementProbeHost {
    struct StagnationStop {};
    bool usable = true;
    bool flush_fails = false;
    mutable std::vector<std::string> calls;
    void flush_remaining(bool /*retry_io*/ = true) const
    {
        calls.emplace_back("flush");
        if (flush_fails) throw std::runtime_error("disk full");
    }
    bool has_usable_loss() const
    {
        calls.emplace_back("usable");
        return usable;
    }
};

}  // namespace

/* Pins: every ending of drive_run asks the host once whether it produced a
 * usable loss -- the run's one cross-rank collective -- before anything
 * else it asks, and flushes the results at least once; an error ending
 * rethrows. A normal end without a recorded status, drive_run's internal
 * error (std::logic_error, thrown before the collective on the normal
 * path), reaches the collective through the error handler like any other
 * error and is rethrown, also when its final flush fails.
 * Before the fix: the endings with a status, the stops and the errors pass
 * as well (a pin; the normal ends' bodies then recorded none, which
 * drive_run accepted), and so does the end without a status whose flush
 * fails (rethrown for the flush's error); a normal end without a status
 * ended normally, not rethrown (fails). */
TEST_CASE("No usable loss: every ending of drive_run reaches the cross-rank "
          "agreement once and before the final flush",
          "[HPO][termination][no_usable_loss][run_driver]")
{
    /* On several MPI ranks the question is a collective every rank must
     * reach, and exactly once. A rank that skipped it -- its final flush
     * failed, or an error or a signal ended its run after the drivers'
     * last collective -- left the other ranks waiting in it for ever. So
     * every ending asks, and before the rank-local flush that can fail. */
    struct Ending {
        const char* what;
        std::function<void()> body;
        bool flush_fails;
        bool rethrows;
    };
    /* A normal end records its status, as every optimiser does; the body
     * of a normal end without one is an internal error. */
    const auto normal_end = [] {
        tadah::hpo::HPOLog::instance().set_status("TEST: done");
    };
    const auto no_status = [] {};
    const Ending endings[] = {
        {"a normal end", normal_end, false, false},
        {"a normal end whose final flush fails", normal_end, true, true},
        {"a normal end without a status", no_status, false, true},
        {"a normal end without a status whose final flush fails", no_status,
         true, true},
        {"a stagnation stop",
         [] { throw AgreementProbeHost::StagnationStop{}; }, false, false},
        {"a shutdown", [] { throw tadah::core::ShutdownRequested{}; }, false,
         false},
        {"an error", [] { throw std::runtime_error("optimiser broke"); },
         false, true},
        {"a non-standard exception", [] { throw 42; }, false, true},
    };
    for (const auto& e : endings) {
        DYNAMIC_SECTION(e.what) {
            ScratchDir sd;
            auto& log = tadah::hpo::HPOLog::instance();
            log.reset_for_test();
            log.open((sd.dir / "hpo_run.log").string(), "test",
                     sd.dir.string(), 1);
            AgreementProbeHost host;
            host.flush_fails = e.flush_fails;
            bool rethrown = false;
            try {
                (void)tadah::hpo::drive_run(e.body, host, log);
            } catch (...) {
                rethrown = true;
            }
            CHECK(rethrown == e.rethrows);
            INFO("calls: " << [&] {
                std::string s;
                for (const auto& c : host.calls) s += c + " ";
                return s;
            }());
            CHECK(std::count(host.calls.begin(), host.calls.end(),
                             "usable") == 1);
            REQUIRE_FALSE(host.calls.empty());
            CHECK(host.calls.front() == "usable");
            CHECK(std::count(host.calls.begin(), host.calls.end(),
                             "flush") >= 1);
            log.reset_for_test();
        }
    }
}

TEST_CASE("No usable loss: SIGINT ends a run without a usable loss with 130",
          "[.shutdown_no_usable_loss]")
{
    /* The signal decides the status, not the missing usable loss. */
    TaRun run(plain_block("    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n"
                          "    MAXEVAL 100000000\n    SEED 3\n"),
              kAllFailRoot);
    const SignalOutcome o = run_and_signal(run.ctx, SIGINT);
    REQUIRE(o.raised);
    CHECK_FALSE(o.stalled);
    INFO("hpo_run exception: " << describe(o.error));
    CHECK_FALSE(o.error);
    CHECK(o.status == 130);
    CHECK(o.evals_final <= o.evals_at_raise + 1);
    check_no_usable_loss(o.evals_final, kDefaultFailScoreRow,
                         "SHUTDOWN (SIGINT)");
}

/* ======================================================================
 * A finite-difference gradient that is not finite because an evaluation of
 * its stencil failed.
 *
 * One dimension, CGRID2B[2] in [1, 2], PARAM fd_step 0.2: the stencil of
 * the start point 1.6 is 1.8 and 1.4 (the unit box: h = 0.2 of the width).
 * The validation ERMSE of this fixture falls from 10 meV/atom at 1.0 to
 * 9.76 at 2.0: 9.9999945 at 1.4, 9.9999768 at 1.45, 9.9999074 at 1.5,
 * 9.9987 at 1.6, 9.914 at 1.8. The skip cap 9.99995 fails the box below
 * about 1.47, so 1.4 fails and scores the default FAILSCORE, the largest
 * double, and the central difference overflows.
 *
 * Every gradient-based backend then stopped after four evaluations (1.6
 * for the value; 1.6 again, 1.8 and 1.4 for the gradient) and reported a
 * success: DLIB:BFGS "MAXEVAL=60 reached" and COMPLETED, NLopt LD_LBFGS,
 * LD_TNEWTON and LD_VAR2 "SUCCESS -- generic success", LD_SLSQP
 * "roundoff-limited", every CERES driver COMPLETED. LD_MMA and LD_CCSAQ
 * did not stop, but did not go on either: they asked for the same point
 * again and again (the three points twenty times over, then the start
 * point alone), 120 evaluations for MAXEVAL 60. The search now ends at
 * that gradient with its own status, and the journal names the backend
 * and the stencil point that failed. The run has a usable loss, so it ends
 * with status 0. The gradient takes the loss at 1.6 its caller has just
 * evaluated as its centre (gradient_at): three evaluations, 1.6, 1.8 and
 * 1.4; it evaluated 1.6 a second time until 2026-10-07.
 *
 * A failed evaluation fails the gradient whatever FAILSCORE is (2026-10-07,
 * "No usable gradient: ..."): at a usable point the first failed stencil
 * evaluation, at a failed point the loss there (FAILSCORE_GRADING OFF), so
 * a start inside the failure region (below 1.47) is no longer differenced
 * into an exact 0 between two failures, a "converged" start, nor a failed
 * stencil point under a finite FAILSCORE into a steep, finite slope. The
 * status says "no usable gradient" (it said "gradient not finite", which
 * that slope was not).
 * ==================================================================== */

namespace {

const std::string kStencilCause =
    "no usable gradient: a failed evaluation in the finite-difference "
    "stencil";
const std::string kStencilStatus = "STOPPED -- " + kStencilCause;

/* The objective of the straddle cases: `target` is the ERMSE target of
 * PC_ERMSE (0: the loss falls towards 2.0, away from the failures; 10.5:
 * it falls towards 1.0, into them). */
std::string straddle_root(const std::string& target = "0")
{
    return "LOSS L2\n"
           "QFLUSH 1000\n"
           "DIGITS 15\n"
           "PC_ERMSE " + target + " 1 --skip-above 9.99995\n"
           "OPTIM CGRID2B (2) 1.0 2.0\n";
}

std::string straddle_block(const std::string& lib, const std::string& algo,
                           const std::string& extra = "")
{
    return plain_block("    TYPE PLAIN\n    LIB " + lib + "\n    ALGO " +
                       algo + "\n    MAXEVAL 60\n    PARAM fd_step 0.2\n" +
                       extra);
}

/* The CGRID2B[2] of each evaluation, in order (params.tadah). */
std::vector<double> evaluated_cgrid2()
{
    std::vector<double> x;
    for (const auto& row : data_rows("params.tadah"))
        x.push_back(std::stod(row.back()));
    return x;
}

/* Runs `optimizer` from CGRID2B[2] = `start`. */
int straddle_run(TaRun& run, const std::string& start, const std::string& what)
{
    run.ctx.storage_update("CGRID2B", {"0.5", start});
    return run_within(run, std::chrono::seconds(120), what);
}

struct GradientBackend { std::string lib, algo; };

std::vector<GradientBackend> fd_gradient_backends()
{
    std::vector<GradientBackend> b = {
        {"DLIB", "BFGS"},
        {"NLOPT", "LD_LBFGS"}, {"NLOPT", "LD_TNEWTON"},
        {"NLOPT", "LD_TNEWTON_PRECOND_RESTART"}, {"NLOPT", "LD_VAR2"},
        {"NLOPT", "LD_SLSQP"}, {"NLOPT", "LD_MMA"}, {"NLOPT", "LD_CCSAQ"}};
#ifdef TADAH_USE_CERES
    /* CERES:STEEPEST_DESCENT, the Ceres solver HPO runs (CERES:LBFGS, WOLFE
     * and NLCG, rows here before the fix, are refused: their Wolfe line
     * search stalls at a point outside the OPTIM box). */
    b.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    return b;
}

}  // namespace

namespace {

/* The status each finite-difference backend ends with when, from 1.6, the
 * gradient there is one-sided into the usable side and the search reaches
 * the bound 2.0; and how many evaluations it makes. */
struct SideEnding { std::string status; std::size_t evals; };
SideEnding side_ending(const std::string& name)
{
    const std::string nlopt_ok = "SUCCESS -- generic success";
    const std::string maxeval = "MAXEVAL_REACHED -- maximum evaluations exhausted";
    if (name == "DLIB:BFGS") return {"DLIB:BFGS: FTOL_ABS reached", 6u};
    if (name == "NLOPT:LD_TNEWTON") return {nlopt_ok, 9u};
    if (name == "NLOPT:LD_SLSQP")
        return {"STOPPED -- nlopt roundoff-limited (algorithm converged; "
                "consider adding ftol_rel / xtol_rel / maxeval to stop "
                "cleanly next time)", 9u};
    if (name == "NLOPT:LD_MMA") return {maxeval, 9u};
    if (name == "NLOPT:LD_CCSAQ") return {maxeval, 6u};
    if (name == "CERES:STEEPEST_DESCENT")
        return {"CERES:STEEPEST_DESCENT: FTOL_REL reached", 35u};
    return {nlopt_ok, 6u};   /* LD_LBFGS, LD_TNEWTON_PRECOND_RESTART, LD_VAR2 */
}

}  // namespace

TEST_CASE("Finite differences: a failed stencil point at the start point: "
          "every finite-difference backend takes the usable side and reaches "
          "the bound", "[HPO][termination][fd_gradient]")
{
    /* A difference never mixes a usable loss with a failure: from 1.6
     * the central stencil is 1.8 and
     * 1.4, 1.4 fails (below 1.47), and the gradient is the one-sided
     * difference into the usable side, 1.8 and 2.0, with the loss at 1.6.
     * Pins: the first four evaluations 1.6, 1.8, 1.4 (failed) and 2.0; the
     * best is 2.0, on the bound (loss 95.25); no stop line; per backend the
     * status and evaluations of side_ending (DLIB:BFGS "FTOL_ABS reached"
     * after 6, NLopt LD_LBFGS 6 "SUCCESS -- generic success", ...,
     * CERES:STEEPEST_DESCENT "FTOL_REL reached" after 35, 4 of its points
     * outside the box and not evaluated); status 0.
     * Before the fix: every backend stopped at that gradient after three
     * evaluations, 1.6, 1.8 and 1.4, with the status "STOPPED -- no usable
     * gradient: a failed evaluation in the finite-difference stencil" and
     * the stop line "<backend> stopped: ... (at CGRID2B[2] = 1.6, the
     * stencil point CGRID2B[2] = 1.4 failed: loss 1.797693135e+308)". */
    for (const auto& b : fd_gradient_backends()) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(b.lib, b.algo), straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.6", name));
            CHECK(status == 0);
            const SideEnding e = side_ending(name);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == e.evals);
            CHECK(x[0] == Approx(1.6));
            CHECK(x[1] == Approx(1.8));
            CHECK(x[2] == Approx(1.4));
            CHECK(x[3] == Approx(2.0));
            const auto loss = data_rows("loss.tadah");
            REQUIRE(loss.size() == e.evals);
            CHECK(loss[2].back() == kDefaultFailScoreRow);
            CHECK(loss_value(loss[3].back()) == Approx(95.2545).epsilon(1e-5));
            const auto best = data_rows("best_params.tadah");
            REQUIRE_FALSE(best.empty());
            CHECK(std::stod(best.back().back()) == Approx(2.0));
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal_status(journal) == e.status);
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            check_run_files();
        }
    }
}

TEST_CASE("Finite differences: with Ceres' step rule a failed stencil point "
          "is not differenced either", "[HPO][termination][fd_gradient]")
{
    /* PARAM fd_kernel ceres: Ceres' step rule, fd_step * |u| = 0.1 at
     * u = 0.5 (CGRID2B[2] = 1.5): the central stencil 1.6 and 1.4, and 1.4
     * fails; the gradient is the one-sided difference into the usable
     * side, 1.6 and 1.7 (a difference never mixes a usable loss with a
     * failure).
     * Pins: the first four evaluations 1.5, 1.6, 1.4 (failed), 1.7; no stop
     * line at 1.5; DLIB:BFGS reaches the bound 2.0 after 9 evaluations
     * ("FTOL_ABS reached"); NLOPT:LD_LBFGS, misled next to the failure
     * region by the one-sided estimate over 0.2 of a strongly curved loss,
     * ends "STOPPED -- nlopt failure (...)" after 33; CERES:STEEPEST_DESCENT
     * "FAILURE (Numerical failure in line search)" after 49.
     * Before the fix: three evaluations, the last 1.4, and the stop line
     * "<backend> stopped: no usable gradient: ... (at CGRID2B[2] = 1.5, the
     * stencil point CGRID2B[2] = 1.4 failed ...". */
    struct Case { GradientBackend b; std::size_t evals; std::string status; };
    std::vector<Case> cases = {
        {{"DLIB", "BFGS"}, 9u, "DLIB:BFGS: FTOL_ABS reached"},
        {{"NLOPT", "LD_LBFGS"}, 33u,
         "STOPPED -- nlopt failure (algorithm converged; consider adding "
         "ftol_rel / xtol_rel / maxeval to stop cleanly next time)"}};
#ifdef TADAH_USE_CERES
    cases.push_back({{"CERES", "STEEPEST_DESCENT"}, 49u,
                     "STOPPED -- CERES:STEEPEST_DESCENT: FAILURE (Numerical "
                     "failure in line search)"});
#endif
    for (const auto& c : cases) {
        const std::string name = c.b.lib + ":" + c.b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(c.b.lib, c.b.algo,
                                     "    PARAM fd_kernel ceres\n"),
                      straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.5", name));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == c.evals);
            CHECK(x[0] == Approx(1.5));
            CHECK(x[1] == Approx(1.6));
            CHECK(x[2] == Approx(1.4));
            CHECK(x[3] == Approx(1.7));
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal_status(journal) == c.status);
            CHECK(journal.find("(at CGRID2B[2] = 1.5, the stencil point") ==
                  std::string::npos);
            check_run_files();
        }
    }
}

TEST_CASE("Finite differences: DLIB:BFGS goes on past a later point with a "
          "failed stencil point, to the edge of the failure region",
          "[HPO][termination][fd_gradient]")
{
    /* With the ERMSE target 10.5 the loss falls towards the failures. From
     * 1.8, BFGS's line search tries 1.059 (a failed evaluation: dlib steps
     * back), accepts 1.726; the stencil of its next point 1.6676 reaches
     * 1.4676, which fails, and the gradient there is the one-sided
     * difference into the usable side.
     * Pins: the search goes on to the edge of the failure region, 1.4813
     * (loss 0.250056, the best), and ends "DLIB:BFGS: FTOL_ABS reached"
     * after 54 evaluations; no stop line.
     * Before the fix: the gradient at 1.6676 failed, and the search ended
     * there after 10 evaluations, the last a failed one, with the stop line
     * "DLIB:BFGS stopped: no usable gradient: ... (at CGRID2B[2] = 1.6675"
     * and the status "STOPPED -- no usable gradient: ...". */
    TaRun run(straddle_block("DLIB", "BFGS"), straddle_root("10.5"));
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.8", "DLIB:BFGS"));
    CHECK(status == 0);
    const auto loss = data_rows("loss.tadah");
    CHECK(loss.size() == 54u);
    const auto best = data_rows("best_params.tadah");
    REQUIRE_FALSE(best.empty());
    CHECK(std::stod(best.back().back()) == Approx(1.481330854).epsilon(1e-8));
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == "DLIB:BFGS: FTOL_ABS reached");
    CHECK(journal.find("stopped: no usable gradient") == std::string::npos);
    CHECK(journal.find("MAXEVAL=60 reached") == std::string::npos);
    check_run_files();
}

namespace {

struct TrialCase {
    GradientBackend b;
    std::size_t     evals;   /* the evaluations of the search */
    std::string     ending;  /* the trailer of the search, which goes on */
};

/* Whether a failed evaluation of loss.tadah is followed by a usable one:
 * the search went on past it. */
bool went_on_past_a_failure(const std::vector<std::vector<std::string>>& loss)
{
    bool failed = false;
    for (const auto& row : loss) {
        if (row.back() == kDefaultFailScoreRow) failed = true;
        else if (failed) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("Gradient not finite: NLopt and Ceres reject a later point whose "
          "gradient is not finite and go on, as before",
          "[HPO][termination][fd_gradient]")
{
    /* NLopt sees +inf at a point whose gradient is not finite, Ceres a
     * failed evaluation: both reject the point and step back, and the
     * search goes on. A point next to the failure region whose stencil
     * fails on one side takes the other side. LD_MMA (its trust region
     * shrinks; only at a start point
     * without a finite gradient did it stall) and LD_LBFGS spend MAXEVAL
     * and end MAXEVAL_REACHED; CERES:STEEPEST_DESCENT ends "FAILURE
     * (Numerical failure in line search)" after 54, at the edge of the
     * failure region (it spent MAXEVAL, "CERES:STEEPEST_DESCENT: MAXEVAL
     * reached", before the fix: its gradients failed next to the failures).
     *
     * Pins: 60 evaluations for MAXEVAL 60 in LD_MMA and LD_LBFGS, 54 in
     * CERES:STEEPEST_DESCENT, with the endings above;
     * every NLopt evaluation, the loss and each finite-difference point,
     * counts against MAXEVAL (test_nlopt_maxeval.cpp).
     * Before the fix: LD_MMA made 120 evaluations and LD_LBFGS 101, ending
     * "STOPPED -- nlopt failure (algorithm converged ...)": the
     * GradientProvider counted the stencil evaluations against MAXEVAL,
     * NLopt its objective calls; once the provider's budget was spent, each
     * gradient request failed (GradientFailure::Budget) and the objective
     * still evaluated the point and returned +inf, until NLopt's own count
     * reached MAXEVAL (LD_MMA) or the search failed (LD_LBFGS).
     * The stop line it checks is absent names "no usable gradient" (it
     * named "gradient not finite" until 2026-10-07); with that, the case
     * passes before the failure-region fix as well (a pin: the counts and
     * endings of these searches are the same with it). */
    std::vector<TrialCase> cases = {
        {{"NLOPT", "LD_MMA"}, 60u,
         "MAXEVAL_REACHED -- maximum evaluations exhausted"},
        {{"NLOPT", "LD_LBFGS"}, 60u,
         "MAXEVAL_REACHED -- maximum evaluations exhausted"}};
#ifdef TADAH_USE_CERES
    cases.push_back({{"CERES", "STEEPEST_DESCENT"}, 54u,
                     "STOPPED -- CERES:STEEPEST_DESCENT: FAILURE (Numerical "
                     "failure in line search)"});
#endif
    for (const auto& c : cases) {
        const std::string name = c.b.lib + ":" + c.b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(c.b.lib, c.b.algo), straddle_root("10.5"));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.8", name));
            CHECK(status == 0);
            const auto loss = data_rows("loss.tadah");
            CHECK(loss.size() == c.evals);
            CHECK(went_on_past_a_failure(loss));
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find(name + " stopped: no usable gradient") ==
                  std::string::npos);
            CHECK(journal_status(journal) != kStencilStatus);
            if (!c.ending.empty())
                CHECK(journal_status(journal) == c.ending);
            check_run_files();
        }
    }
}

namespace {

/* An INNER block for the NLopt algorithm `algo`, with the lines `extra`. */
std::string inner_block(const std::string& algo, const std::string& extra = "")
{
    return "    INNER\n      LIB NLOPT\n      ALGO " + algo + "\n" + extra +
           "    ENDINNER\n";
}

struct AuglagCase {
    std::string algo;    /* the AUGLAG variant */
    std::string inner;   /* its INNER block ("": NLopt's default) */
    std::string label;
};

}  // namespace

TEST_CASE("Gradient not finite: AUGLAG's local search stops at a starting "
          "point without a usable gradient too, and goes on from one with a "
          "usable side", "[HPO][termination][fd_gradient][auglag]")
{
    /* HPO sets AUGLAG no constraint, so AUGLAG runs its local search once,
     * from the starting point (auglag.c), and that search is one local
     * gradient search: it is stopped where a plain LD_* search is. It was
     * not: LD_AUGLAG's default local search LD_MMA, and an INNER LD_MMA,
     * asked for the same point until MAXEVAL, and an INNER LD_LBFGS or
     * LD_TNEWTON ended at once, AUGLAG evaluated the point it returned
     * once more and reported FTOL_REACHED, a converged search.
     * Pins: from 1.25, inside the failure region, one evaluation, the stop
     * line "(the loss at CGRID2B[2] = 1.25 is 1.797693135e+308); a local
     * gradient search has no step to take from a starting point without a
     * usable gradient", no usable loss (status 1; a pin: the loss at the
     * point decides there, as before the fix). From 1.6 the stencil point
     * 1.4 fails and the gradient is the one-sided difference of 1.8 and
     * 2.0 (a difference never mixes a usable loss with a failure): the
     * search goes on to the bound 2.0, the first
     * four evaluations 1.6, 1.8, 1.4, 2.0; LD_AUGLAG with its default or an
     * INNER LD_MMA ends MAXEVAL_REACHED after 9, AUGLAG with an INNER LD_LBFGS
     * FTOL_REACHED after 6, with LD_TNEWTON after 9, LN_AUGLAG with LD_SLSQP
     * roundoff-limited after 9.
     * Before the fix: from 1.6 three evaluations, 1.6, 1.8 and 1.4, and the
     * stop line "(at CGRID2B[2] = 1.6, the stencil point CGRID2B[2] = 1.4
     * failed: loss 1.797693135e+308); a local gradient search has no step
     * ...", the status "STOPPED -- no usable gradient: ...". */
    struct Case { AuglagCase a; std::size_t evals; std::string status; };
    const std::string maxeval = "MAXEVAL_REACHED -- maximum evaluations exhausted";
    const std::string ftol = "FTOL_REACHED -- function tolerance converged";
    const std::vector<Case> cases = {
        {{"LD_AUGLAG", "", "NLopt's default LD_MMA"}, 9u, maxeval},
        {{"LD_AUGLAG", inner_block("LD_MMA"), "INNER LD_MMA"}, 9u, maxeval},
        {{"AUGLAG", inner_block("LD_LBFGS"), "INNER LD_LBFGS"}, 6u, ftol},
        {{"AUGLAG", inner_block("LD_TNEWTON"), "INNER LD_TNEWTON"}, 9u, ftol},
        {{"LN_AUGLAG", inner_block("LD_SLSQP"), "INNER LD_SLSQP"}, 9u,
         "STOPPED -- nlopt roundoff-limited (algorithm converged; consider "
         "adding ftol_rel / xtol_rel / maxeval to stop cleanly next time)"}};
    for (const auto& c : cases) {
        const std::string name = "NLOPT:" + c.a.algo;
        DYNAMIC_SECTION(name << " (" << c.a.label << ") from 1.25") {
            TaRun run(straddle_block("NLOPT", c.a.algo, c.a.inner),
                      straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.25", name));
            CHECK(status == 1);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == 1u);
            CHECK(x[0] == Approx(1.25));
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal,
                           name + " stopped: " + kStencilCause +
                           " (the loss at CGRID2B[2] = 1.25 is "
                           "1.797693135e+308); a local gradient search has no "
                           "step to take from a starting point without a "
                           "usable gradient") == 1u);
            check_no_usable_loss(1u, kDefaultFailScoreRow, kNoUsableLossStatus);
        }
        DYNAMIC_SECTION(name << " (" << c.a.label << ") from 1.6") {
            TaRun run(straddle_block("NLOPT", c.a.algo, c.a.inner),
                      straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.6", name));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == c.evals);
            CHECK(x[0] == Approx(1.6));
            CHECK(x[1] == Approx(1.8));
            CHECK(x[2] == Approx(1.4));
            CHECK(x[3] == Approx(2.0));
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal_status(journal) == c.status);
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            check_run_files();
        }
    }
}

TEST_CASE("No usable gradient: the straddle searches that ended right after "
          "a rejection go on to MAXEVAL", "[HPO][termination][fd_gradient]"
          "[auglag]")
{
    /* With the ERMSE target 10.5 these searches ended right after they
     * rejected a point whose stencil reached into the failure region: LD_SLSQP
     * from 1.72 rejected 1.6554 (its stencil point 1.4554 failed) and ended
     * "roundoff-limited", with XTOL_REL and FTOL_REL XTOL_REACHED, and as the
     * INNER search of AUGLAG FTOL_REACHED; the CERES drivers from 1.8 rejected
     * 1.429 (1.229 failed) and ended "Numerical failure in line search"
     * (the cases "a search that ends right after it rejected such a point
     * says so" and "a search that ends with a tolerance, or AUGLAG's ending,
     * right after it rejected such a point says so"). Each had been handed
     * the loss FAILSCORE with a zero gradient at a point inside the failure
     * region, 1.074 or 1.0586, whose stencil failed as well: the difference
     * of two failures. That point is rejected now, and the searches go on.
     * The endings right after a rejection are covered on an outage stub in
     * test_fd_bounds.cpp ("No usable gradient: an NLopt search that ends
     * right after it rejected a point says so, on an outage", and the Ceres
     * case).
     * A point next to the failure region whose stencil fails on one side
     * takes the other side, so with the
     * tolerances the searches converge before MAXEVAL.
     * Pins: every evaluation in the box [1, 2], a usable one after a failed
     * one, no stop line; LD_SLSQP 60 evaluations, MAXEVAL_REACHED; with
     * XTOL_REL and FTOL_REL 50, XTOL_REACHED; AUGLAG with that INNER block 50,
     * FTOL_REACHED; CERES:STEEPEST_DESCENT (the Ceres solver HPO runs) from
     * 1.8 54 (its own ending is not pinned here).
     * Before the fix: the tolerance cases spent MAXEVAL 60 and ended
     * MAXEVAL_REACHED, and the case ran CERES:LBFGS, WOLFE and NLCG,
     * refused since, to 60. Before the failed-evaluation rule (2026-10-07):
     * LD_SLSQP 10 evaluations, the last the failed stencil
     * point 1.4554, ending with the stop line "... (at CGRID2B[2] =
     * 1.65540558, the stencil point CGRID2B[2] = 1.45540558 failed: ...);
     * NLopt ended right after it (nlopt roundoff-limited)" (XTOL_REACHED
     * with the tolerances), AUGLAG 11 ("(FTOL_REACHED -- function tolerance
     * converged)"), the CERES drivers 9 ("... (at CGRID2B[2] = 1.429...;
     * Ceres ended right after it (FAILURE: Numerical failure in line
     * search ..."), and the trailer "STOPPED -- gradient not finite: a
     * failed evaluation in the finite-difference stencil". */
    const std::string tolerances = "    XTOL_REL 1e-3\n    FTOL_REL 1e-6\n";
    struct Case {
        std::string lib, algo, extra, label, start;
        std::size_t evals;
        std::string ending;   /* "": not pinned */
    };
    std::vector<Case> cases = {
        {"NLOPT", "LD_SLSQP", "", "LD_SLSQP", "1.72", 60u,
         "MAXEVAL_REACHED -- maximum evaluations exhausted"},
        {"NLOPT", "LD_SLSQP", tolerances, "LD_SLSQP, XTOL_REL and FTOL_REL",
         "1.72", 50u, "XTOL_REACHED -- parameter tolerance converged"},
        {"NLOPT", "AUGLAG",
         inner_block("LD_SLSQP", "      XTOL_REL 1e-3\n      FTOL_REL 1e-6\n"),
         "AUGLAG, INNER LD_SLSQP", "1.72", 50u,
         "FTOL_REACHED -- function tolerance converged"}};
#ifdef TADAH_USE_CERES
    cases.push_back({"CERES", "STEEPEST_DESCENT", "", "CERES:STEEPEST_DESCENT",
                     "1.8", 54u, ""});
#endif
    for (const auto& c : cases) {
        const std::string name = c.lib + ":" + c.algo;
        DYNAMIC_SECTION(c.label) {
            TaRun run(straddle_block(c.lib, c.algo, c.extra),
                      straddle_root("10.5"));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, c.start, name));
            CHECK(status == 0);
            const auto loss = data_rows("loss.tadah");
            CHECK(loss.size() == c.evals);
            CHECK(went_on_past_a_failure(loss));
            for (double xi : evaluated_cgrid2()) {
                CAPTURE(xi);
                CHECK((xi >= 1.0 && xi <= 2.0));
            }
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            CHECK(journal.find("ended right after it") == std::string::npos);
            CHECK(journal_status(journal) != kStencilStatus);
            if (!c.ending.empty())
                CHECK(journal_status(journal) == c.ending);
            check_run_files();
        }
    }
}

TEST_CASE("Gradient not finite: AUGLAG's local search that spends its own "
          "MAXEVAL at such a point ends on that criterion",
          "[HPO][termination][fd_gradient][auglag]")
{
    /* AUGLAG with the INNER search LD_SLSQP of the case above, from 1.72
     * with the ERMSE target 10.5, and an INNER MAXEVAL 4: its fourth
     * objective call is the point 1.6393, whose stencil point 1.4393 fails,
     * and its gradient is the forward difference with 1.8393 (a difference
     * never mixes a usable loss with a failure); the search ends there on
     * its MAXEVAL, which AUGLAG reports
     * as FTOL_REACHED, like any ending of its local search: the run keeps
     * AUGLAG's own status. An INNER MAXEVAL counts the local search's NLopt
     * calls, not its evaluations (the journal's
     * "Inner stop criteria" say so): its 4 calls are 8 evaluations here (the
     * rejected points 1.074 and 1.397 cost one each). AUGLAG's evaluation of
     * the point the search returned, 1.6393, eval 6, is answered with its
     * loss (every evaluation of this run is a function of its point).
     * With INNER MAXEVAL 5 the rejection used to end the search, as without
     * one; the search goes on past it now (the case above), and the split
     * between an INNER MAXEVAL that a rejection reaches and one it does not
     * is covered on an outage stub (test_fd_bounds.cpp, "No usable
     * gradient: an NLopt search that ends right after it rejected a point
     * says so, on an outage").
     * Pins: 8 evaluations, the 8th the failed stencil point 1.4393, and
     * the journal line "NLOPT:AUGLAG asked again for the point of eval 6
     * (CGRID2B[2] = 1.639256975): the loss of eval 6 is reused"; FTOL_REACHED,
     * no stop line.
     * Before the fix: the fourth call was the rejection of 1.6393 (its
     * gradient failed at the stencil point 1.4393), the search returned the
     * start, and the line said "the point of eval 1 (CGRID2B[2] = 1.72): the
     * loss of eval 1 is reused". Before the failed-evaluation rule
     * (2026-10-07): 11 evaluations, the rejection of 1.6554 the fourth
     * call (1.074 had been handed FAILSCORE with a zero gradient), and with
     * INNER MAXEVAL 5 the stop line and the trailer "STOPPED -- gradient not
     * finite: ...". */
    const std::string tolerances = "      XTOL_REL 1e-3\n      FTOL_REL 1e-6\n";
    TaRun run(straddle_block("NLOPT", "AUGLAG",
                             inner_block("LD_SLSQP",
                                         "      MAXEVAL 4\n" + tolerances)),
              straddle_root("10.5"));
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.72", "NLOPT:AUGLAG"));
    CHECK(status == 0);
    const auto x = evaluated_cgrid2();
    const auto loss = data_rows("loss.tadah");
    REQUIRE(x.size() == 8u);
    REQUIRE(loss.size() == 8u);
    CHECK(x[7] == Approx(1.439256975).epsilon(1e-9));
    CHECK(loss[7].back() == kDefaultFailScoreRow);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("NLOPT:AUGLAG asked again for the point of eval 6 "
                       "(CGRID2B[2] = 1.639256975): the loss of eval 6 is "
                       "reused") != std::string::npos);
    CHECK(journal_status(journal) ==
          "FTOL_REACHED -- function tolerance converged");
    CHECK(journal.find("stopped: no usable gradient") == std::string::npos);
    check_run_files();
}

TEST_CASE("Gradient not finite: ended_on_rejection counts the calls after "
          "the rejection and reads the MAXEVAL and MAXTIME of the local "
          "search", "[HPO][termination][fd_gradient][auglag]")
{
    using tadah::hpo::OptimizerSpec;
    using tadah::hpo::detail::RejectionRecord;
    using tadah::hpo::detail::auglag_reevaluations;
    using tadah::hpo::detail::ended_on_rejection;
    const auto nlopt = [](const std::string& algo) {
        OptimizerSpec s;
        s.lib = "NLOPT";
        s.algo = algo;
        return s;
    };

    /* AUGLAG evaluates the point its local search returned once more, per
     * AUGLAG level; nothing else does. */
    CHECK(auglag_reevaluations(nlopt("LD_LBFGS")) == 0u);
    CHECK(auglag_reevaluations(nlopt("GD_MLSL_LDS")) == 0u);
    CHECK(auglag_reevaluations(nlopt("AUGLAG")) == 0u);   // no local search
    CHECK(auglag_reevaluations(nlopt("LD_AUGLAG")) == 1u);
    auto nested = nlopt("AUGLAG");
    nested.inner = std::make_unique<OptimizerSpec>(nlopt("LD_AUGLAG"));
    CHECK(auglag_reevaluations(nested) == 2u);

    const std::optional<OptimizerSpec> none;
    RejectionRecord r;
    r.calls = 7;
    r.calls_at_rejection = 7;
    r.rejected_after = 2.0;
    CHECK_FALSE(ended_on_rejection(r, 0, none));  // nothing rejected
    r.last_rejected = true;
    CHECK(ended_on_rejection(r, 0, none));
    /* A call after the rejection: the search went on, unless it is
     * AUGLAG's re-evaluation. */
    r.calls = 8;
    CHECK_FALSE(ended_on_rejection(r, 0, none));
    CHECK(ended_on_rejection(r, 1, none));
    r.calls = 9;
    CHECK_FALSE(ended_on_rejection(r, 1, none));
    r.calls = 8;

    /* The local search's MAXEVAL and MAXTIME, reached by the rejection. */
    std::optional<OptimizerSpec> local = nlopt("LD_SLSQP");
    CHECK(ended_on_rejection(r, 1, local));
    local->stop.maxeval = 8;
    CHECK(ended_on_rejection(r, 1, local));
    local->stop.maxeval = 7;
    CHECK_FALSE(ended_on_rejection(r, 1, local));
    local->stop.maxeval.reset();
    local->stop.maxtime = 2.5;
    CHECK(ended_on_rejection(r, 1, local));
    local->stop.maxtime = 2.0;
    CHECK_FALSE(ended_on_rejection(r, 1, local));
}

TEST_CASE("Gradient not finite: MLSL and the global GD_* searches go on past "
          "a local start or a point without a finite gradient",
          "[HPO][termination][fd_gradient]")
{
    /* A global search is not one local search: a rejected local start of
     * MLSL, or a rejected point of GD_STOGO, leaves it the rest, and it is
     * not stopped there. GD_STOGO asks for its first gradient at the
     * centre 1.5, whose stencil point 1.3 fails. G_MLSL_LDS (POPULATION 3,
     * the ERMSE target 10.5) samples 1.6 (the starting point), 1.6875,
     * 1.9375 and 1.4375 and starts its first local search, INNER LD_LBFGS,
     * at the best of them, 1.6, whose stencil point 1.4 fails. Both go on
     * and end on their own.
     *
     * Pins: the evaluations up to the rejection (`first`: each gradient
     * takes the loss its caller has just evaluated at its point as its
     * centre, so the rejected point is evaluated once), a usable one after
     * it, and the ending: MAXEVAL_REACHED after exactly 60 evaluations for
     * MAXEVAL 60, the loss and each finite-difference point counted
     * (test_nlopt_maxeval.cpp).
     * Before the fix: GD_STOGO made 120 evaluations, ending "SUCCESS
     * -- generic success", and G_MLSL_LDS 104, ending "STOPPED -- nlopt
     * failure (algorithm converged ...)" (the NLopt MAXEVAL accounting of
     * the case "NLopt and Ceres reject a later point ..."); before the
     * centre fix their gradients evaluated the rejected point twice, 1.5
     * and 1.6 (`first` {1.5, 1.5, 1.7, 1.3} and {..., 1.6, 1.6, 1.8,
     * 1.4}). MLSL's local search asks for the gradient at its start, the
     * sample 1.6, which it evaluated without one: the sample's recorded
     * loss is the loss at the point that NUMERICAL's finite differences
     * take (repeat_points.h), so 1.6 is not evaluated again; until the
     * repeat-point review (2026-10-12) it was ({..., 1.4375, 1.6, 1.8,
     * 1.4}).
     * The stop line it checks is absent names "no usable gradient" (it
     * named "gradient not finite" until 2026-10-07); with that, the case
     * passes before the failure-region fix as well (a pin: the rule does not
     * stop the global searches). */
    struct Case {
        std::string algo, extra, target;
        std::vector<double> first;   /* evaluations up to the rejection */
    };
    const std::vector<Case> cases = {
        {"GD_STOGO", "", "0", {1.5, 1.7, 1.3}},
        {"G_MLSL_LDS", "    POPULATION 3\n" + inner_block("LD_LBFGS"), "10.5",
         {1.6, 1.6875, 1.9375, 1.4375, 1.8, 1.4}}};
    for (const auto& c : cases) {
        const std::string name = "NLOPT:" + c.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block("NLOPT", c.algo, c.extra),
                      straddle_root(c.target));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.6", name));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            const auto loss = data_rows("loss.tadah");
            REQUIRE(x.size() > c.first.size());
            for (std::size_t i = 0; i < c.first.size(); ++i)
                CHECK(x[i] == Approx(c.first[i]));
            CHECK(loss[c.first.size() - 1].back() == kDefaultFailScoreRow);
            CHECK(went_on_past_a_failure(loss));
            CHECK(x.size() == 60u);
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            CHECK(journal_status(journal) ==
                  "MAXEVAL_REACHED -- maximum evaluations exhausted");
            check_run_files();
        }
    }
}

TEST_CASE("Gradient not finite: under MULTISTART the other starts run, and "
          "the trailer is the last start's", "[HPO][termination][fd_gradient]"
          "[multistart]")
{
    /* DLIB:BFGS from the CONFIG point 1.25, inside the failure region,
     * stops at once: its loss failed, and it has no usable gradient there;
     * from the RANDOM point, 1.9546 after CONFIG with SEED 6 and 1.9036
     * before it with SEED 2, it runs to its own end: the loss changes by
     * less than FTOL_ABS (1e-7 by default) between iterations, "DLIB:BFGS:
     * FTOL_ABS reached" (it recorded no status, and the trailer said
     * COMPLETED, until 2026-10-07). Each start records its status; the last
     * start's is the trailer, and the run has a usable loss (status 0)
     * either way.
     * Pins: the stopped start makes one evaluation, 1.25, with the stop
     * line "(the loss at CGRID2B[2] = 1.25 is 1.797693135e+308)" and
     * "MULTISTART start ... done: 1 eval(s)": the RANDOM start begins at the
     * second (x[1]), and the CONFIG start, last, is the last evaluation.
     * Before the fix: passes as well (a pin). The case started at 1.6, whose
     * gradient failed then at the stencil point 1.4 and is the one-sided
     * difference into the usable side now (the case "a failed stencil
     * point at the start point" above). */
    const auto multistart = [](const std::string& chain,
                               const std::string& seed) {
        return "OPTIMIZER\n  INIT\n" + chain + "    SEED " + seed +
               "\n  ENDINIT\n"
               "  STRATEGY\n    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
               "    N_STARTS 2\n    BUDGET_PER_START 30\n"
               "    PARAM fd_step 0.2\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    const std::string stop_line =
        "DLIB:BFGS stopped: " + kStencilCause +
        " (the loss at CGRID2B[2] = 1.25 is 1.797693135e+308)";
    SECTION("the first start stops, the last one runs to its end") {
        TaRun run(multistart("    METHOD CONFIG\n    METHOD RANDOM 1\n", "6"),
                  straddle_root());
        int status = -1;
        REQUIRE_NOTHROW(status = straddle_run(run, "1.25", "MULTISTART"));
        CHECK(status == 0);
        const auto x = evaluated_cgrid2();
        REQUIRE(x.size() > 1u);
        CHECK(x[0] == Approx(1.25));
        CHECK(x[1] == Approx(1.9546).margin(1e-4));
        const std::string journal = read_file("hpo_run.log");
        CHECK(count_of(journal, stop_line) == 1u);
        CHECK(count_of(journal, "MULTISTART start 1/2 done: 1 eval(s)") == 1u);
        CHECK(count_of(journal, "MULTISTART start 2/2 done") == 1u);
        CHECK(journal_status(journal) == "DLIB:BFGS: FTOL_ABS reached");
        check_run_files();
    }
    SECTION("the first start runs to its end, the last one stops") {
        TaRun run(multistart("    METHOD RANDOM 1\n    METHOD CONFIG\n", "2"),
                  straddle_root());
        int status = -1;
        REQUIRE_NOTHROW(status = straddle_run(run, "1.25", "MULTISTART"));
        CHECK(status == 0);
        const auto x = evaluated_cgrid2();
        REQUIRE(x.size() > 2u);
        CHECK(x[0] == Approx(1.9036).margin(1e-4));
        CHECK(x.back() == Approx(1.25));
        const std::string journal = read_file("hpo_run.log");
        CHECK(count_of(journal, stop_line) == 1u);
        CHECK(count_of(journal, "stopped: no usable gradient") == 1u);
        CHECK(count_of(journal, "MULTISTART start 2/2 done: 1 eval(s)") == 1u);
        CHECK(journal_status(journal) == kStencilStatus);
        check_run_files();
    }
}

TEST_CASE("Gradient not finite: the ending keeps the run driver's order",
          "[HPO][termination][fd_gradient][run_driver]")
{
    /* A search that a gradient that is not finite ended records its status
     * like any optimiser (HPOLog::set_status). A run with a usable loss
     * ends with it and 0; one without a usable loss ends as a failed run
     * (1, the no-usable-loss trailer); a signal and an error still decide
     * the ending first. */
    SECTION("a usable loss: status 0 and the gradient trailer") {
        HostFixture fx;
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
            [&] {
                fx.evaluate(kDriverEvals);
                fx.log().set_status(kStencilStatus);
            }, fx.host, fx.log()));
        CHECK(status == 0);
        check_driver_files(kStencilStatus);
    }
    SECTION("no usable loss: status 1 and the no-usable-loss trailer") {
        HostFixture fx(kAllFailRoot);
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
            [&] {
                fx.evaluate(kDriverEvals);
                fx.log().set_status(kStencilStatus);
            }, fx.host, fx.log()));
        CHECK(status == 1);
        check_no_usable_loss(kDriverEvals, kDefaultFailScoreRow,
                             kNoUsableLossStatus);
    }
    SECTION("a shutdown: the SHUTDOWN trailer") {
        HostFixture fx;
        int status = -1;
        REQUIRE_NOTHROW(status = tadah::hpo::drive_run(
            [&] {
                fx.evaluate(kDriverEvals);
                fx.log().set_status(kStencilStatus);
                throw tadah::core::ShutdownRequested{};
            }, fx.host, fx.log()));
        CHECK(status == tadah::hpo::shutdown_exit_status(
                            tadah::core::SignalHandler::signal_seen()));
        CHECK(journal_status(read_file("hpo_run.log")) ==
              "SHUTDOWN (" + tadah::core::SignalHandler::reason() + ")");
    }
    SECTION("an error: its own trailer") {
        HostFixture fx;
        CHECK_THROWS_WITH(tadah::hpo::drive_run(
            [&] {
                fx.evaluate(kDriverEvals);
                fx.log().set_status(kStencilStatus);
                throw std::runtime_error("optimiser broke");
            }, fx.host, fx.log()), "optimiser broke");
        CHECK(journal_status(read_file("hpo_run.log")) ==
              "FAILED -- optimiser broke");
    }
}

namespace {

/* straddle_root() with FAILSCORE `failscore` and the root lines `extra`. */
std::string straddle_root_fs(const std::string& failscore,
                             const std::string& extra = "")
{
    return "FAILSCORE " + failscore + "\n" + extra + straddle_root();
}

/* Sets an environment variable for the scope (restores it after). */
struct ScopedEnv {
    std::string name;
    std::optional<std::string> old;
    ScopedEnv(const std::string& n, const std::string& v) : name(n)
    {
        if (const char* e = std::getenv(n.c_str())) old = e;
        ::setenv(n.c_str(), v.c_str(), 1);
    }
    ~ScopedEnv()
    {
        if (old) ::setenv(name.c_str(), old->c_str(), 1);
        else ::unsetenv(name.c_str());
    }
};

}  // namespace

TEST_CASE("No usable gradient: a start inside the failure region ends every "
          "finite-difference backend at its first gradient",
          "[HPO][termination][fd_gradient]")
{
    /* From 1.25 the stencil is 1.45 and 1.05: the start and both stencil
     * points fail (below 1.47), and the difference of two failures,
     * (FS - FS) / 2h, was an exact 0, a gradient every backend accepted.
     * Pins: one evaluation, the start; its failed loss fails the gradient
     * before the stencil is evaluated, and the journal names it: "<backend>
     * stopped: no usable gradient: a failed evaluation in the
     * finite-difference stencil (the loss at CGRID2B[2] = 1.25 is
     * 1.797693135e+308)". No evaluation is usable: the run ends as one
     * without a usable loss (status 1).
     * Before the fix: DLIB:BFGS made 6 evaluations (1.25, 1.45, 1.05 twice)
     * and ended as converged, NLopt LD_LBFGS, LD_TNEWTON* and LD_VAR2 3
     * ("SUCCESS -- generic success fmin=1.79769e+308"), LD_SLSQP 3
     * (roundoff-limited), LD_MMA and LD_CCSAQ asked for the same three
     * points until MAXEVAL (60), the CERES drivers 3 (CONVERGENCE at
     * iteration 0); no stop line. */
    for (const auto& b : fd_gradient_backends()) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(b.lib, b.algo), straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.25", name));
            CHECK(status == 1);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == 1u);
            CHECK(x[0] == Approx(1.25));
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal, name + " stopped: " + kStencilCause +
                                        " (the loss at CGRID2B[2] = 1.25 is "
                                        "1.797693135e+308)") == 1u);
            check_no_usable_loss(1u, kDefaultFailScoreRow, kNoUsableLossStatus);
        }
    }
}

TEST_CASE("No usable gradient: a MULTISTART start inside the failure region "
          "stops at its first gradient", "[HPO][termination][fd_gradient]"
          "[multistart]")
{
    /* DLIB:BFGS from the CONFIG point 1.9 runs to its end; the RANDOM
     * start drawn with SEED 3, 1.206248063 (a central stencil, 1.406 and
     * 1.006), or with SEED 17, 1.050065097 (one-sided, 1.250 and 1.450),
     * lies in the failure region.
     * Pins: the second start makes one evaluation, its start point, and
     * stops with the stop line naming its loss; as the last start, its
     * status is the trailer (the MULTISTART rule of "under MULTISTART the
     * other starts run, and the trailer is the last start's"); the first
     * start's usable losses give status 0.
     * Before the fix: the second start made 6 evaluations (the start and
     * its stencil, twice) of FAILSCORE and ended as converged, without a
     * word, and the trailer read COMPLETED. */
    const auto multistart = [](const std::string& seed) {
        return "OPTIMIZER\n  INIT\n    METHOD CONFIG\n    METHOD RANDOM 1\n"
               "    SEED " + seed + "\n  ENDINIT\n"
               "  STRATEGY\n    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
               "    N_STARTS 2\n    BUDGET_PER_START 30\n"
               "    PARAM fd_step 0.2\n  ENDSTRATEGY\nENDOPTIMIZER\n";
    };
    for (const auto& c : std::vector<std::pair<std::string, std::string>>{
             {"3", "1.206248063"}, {"17", "1.050065097"}}) {
        DYNAMIC_SECTION("SEED " << c.first) {
            TaRun run(multistart(c.first), straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.9", "MULTISTART"));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() >= 2u);
            CHECK(x[0] == Approx(1.9));
            CHECK(x.back() == Approx(std::stod(c.second)).epsilon(1e-9));
            CHECK(x[x.size() - 2] != Approx(x.back()));
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal, "MULTISTART start 2/2 done: 1 eval(s)") ==
                  1u);
            CHECK(count_of(journal, "DLIB:BFGS stopped: " + kStencilCause +
                                        " (the loss at CGRID2B[2] = " +
                                        c.second + " is 1.797693135e+308)") ==
                  1u);
            CHECK(journal_status(journal) == kStencilStatus);
            check_run_files();
        }
    }
}

TEST_CASE("Finite differences: a failed stencil point is not differenced "
          "under a finite FAILSCORE either", "[HPO][termination][fd_gradient]")
{
    /* FAILSCORE 1e9: from 1.6 the stencil point 1.4 fails, and the
     * difference (98.29 - 1e9) / 0.4 was finite, a slope as steep as the
     * jump to FAILSCORE, which the backends followed until a difference
     * that reached a failure failed the gradient. A difference never mixes
     * a usable loss with a failure: the
     * gradient is the one-sided difference of 1.8 and 2.0, as under the
     * default FAILSCORE.
     * Pins: every backend evaluates 1.6, 1.8, 1.4 (loss 1e9) and 2.0 first,
     * makes no stop line, and ends as under the default FAILSCORE
     * (side_ending: the same evaluations and status), but
     * CERES:STEEPEST_DESCENT, whose first step leaves the box (2.6, scored
     * a failure graded by its distance below 1e9) and whose line search then
     * takes a step of 1e-3 of it: "FTOL_REL reached" after 11 evaluations;
     * the best is 2.0; status 0.
     * Before the fix: three evaluations, 1.6, 1.8 and 1.4, the stop line
     * "<backend> stopped: no usable gradient: ... (at CGRID2B[2] = 1.6, the
     * stencil point CGRID2B[2] = 1.4 failed: loss 1000000000)" and the
     * trailer "STOPPED -- no usable gradient: ...". */
    for (const auto& b : fd_gradient_backends()) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(b.lib, b.algo), straddle_root_fs("1e9"));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.6", name));
            CHECK(status == 0);
            SideEnding e = side_ending(name);
            if (b.lib == "CERES")
                e = {"CERES:STEEPEST_DESCENT: FTOL_REL reached", 11u};
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == e.evals);
            CHECK(x[0] == Approx(1.6));
            CHECK(x[1] == Approx(1.8));
            CHECK(x[2] == Approx(1.4));
            CHECK(x[3] == Approx(2.0));
            const auto loss = data_rows("loss.tadah");
            REQUIRE(loss.size() == e.evals);
            CHECK(loss[2].back() == "1.000000000000000e+09");
            const auto best = data_rows("best_params.tadah");
            REQUIRE_FALSE(best.empty());
            CHECK(std::stod(best.back().back()) == Approx(2.0));
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal_status(journal) == e.status);
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            check_run_files();
        }
    }
}

TEST_CASE("No usable gradient: under FAILSCORE_GRADING a start inside the "
          "failure region follows the graded slope out",
          "[HPO][termination][fd_gradient][failscore_grading]")
{
    /* FAILSCORE 1e9: the skip cap's failures are graded band-D losses
     * (FAILSCORE_GRADING ON), graded by the cap's excess: 9.9975000125e8
     * at 1.25, 9.9975000067e8 at 1.45, 9.9975000125e8 at 1.05, a slope
     * towards the usable side. A graded failure is no usable loss, and its
     * differences are its way out of the failure region.
     * Pins (the graded searches pass before the fix as well: pins of the
     * graded slope, which a rule that failed every failed evaluation would
     * break): from 1.25, DLIB:BFGS reaches the bound 2 (95.2545212249) at
     * its 4th evaluation and ends after 6 (after 9 until 2026-10-12, when
     * its line search from the bound evaluated it again and BFGS took its
     * gradient there again), NLopt LD_LBFGS after 6,
     * CERES:STEEPEST_DESCENT after 7 (its first step leaves the box, 2.25,
     * a failure graded by its distance and scored above the graded
     * failures; its line search steps back to 1.361, whose stencil point
     * 1.561 is usable); the first loss is graded and no stop line or
     * stencil trailer follows. A graded difference takes graded failures
     * on both sides (never a usable loss and a failure):
     * 1.45 and 1.05 at 1.25. TADAH_FAILSCORE_GRADING=ON without
     * the key does the same (the override reaches the gradient; DLIB:BFGS
     * here). Without grading the start's loss 1e9 fails its gradient: one
     * evaluation, no usable loss, status 1.
     * Before the fix: the case ran CERES:LBFGS, refused since. Before the
     * failed-evaluation rule (2026-10-07): without grading DLIB:BFGS made 6
     * evaluations of 1e9 (1.25, 1.45, 1.05 twice), a zero gradient, and
     * ended as converged. */
    const std::string graded = "FAILSCORE_GRADING ON\n";
    struct Case {
        std::string lib, algo, label, root;
        bool env;
        std::size_t evals;   /* 0: not pinned */
    };
    std::vector<Case> cases = {
        {"DLIB", "BFGS", "DLIB:BFGS", graded, false, 6u},
        {"NLOPT", "LD_LBFGS", "NLOPT:LD_LBFGS", graded, false, 6u},
        {"DLIB", "BFGS", "DLIB:BFGS, TADAH_FAILSCORE_GRADING=ON", "", true, 6u}};
#ifdef TADAH_USE_CERES
    cases.push_back({"CERES", "STEEPEST_DESCENT", "CERES:STEEPEST_DESCENT",
                     graded, false, 7u});
#endif
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.label) {
            std::optional<ScopedEnv> env;
            if (c.env) env.emplace("TADAH_FAILSCORE_GRADING", "ON");
            TaRun run(straddle_block(c.lib, c.algo),
                      straddle_root_fs("1e9", c.root));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.25", c.label));
            CHECK(status == 0);
            const auto loss = data_rows("loss.tadah");
            REQUIRE(loss.size() >= 2u);
            if (c.evals) CHECK(loss.size() == c.evals);
            const double first = loss_value(loss[0].back());
            CHECK(first >= 0.999e9);
            CHECK(first < 1e9);
            bool usable = false;
            for (const auto& row : loss)
                if (loss_value(row.back()) < 0.999e9) usable = true;
            CHECK(usable);
            if (c.lib == "DLIB") {
                REQUIRE(loss.size() >= 4u);
                CHECK(loss_value(loss[3].back()) ==
                      Approx(95.2545212249).epsilon(1e-10));
            }
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            CHECK(journal_status(journal) != kStencilStatus);
            check_run_files();
        }
    }
    SECTION("FAILSCORE_GRADING OFF") {
        TaRun run(straddle_block("DLIB", "BFGS"), straddle_root_fs("1e9"));
        int status = -1;
        REQUIRE_NOTHROW(status = straddle_run(run, "1.25", "DLIB:BFGS"));
        CHECK(status == 1);
        const std::string journal = read_file("hpo_run.log");
        CHECK(count_of(journal, "DLIB:BFGS stopped: " + kStencilCause +
                                    " (the loss at CGRID2B[2] = 1.25 is "
                                    "1000000000)") == 1u);
        check_no_usable_loss(1u, "1.000000000000000e+09", kNoUsableLossStatus);
    }
}

TEST_CASE("Finite differences: under FAILSCORE_GRADING a graded start next "
          "to the usable side is differenced on its graded side, and no "
          "point outside the OPTIM box is evaluated",
          "[HPO][termination][fd_gradient][failscore_grading][out_of_box]")
{
    /* FAILSCORE 1e9, FAILSCORE_GRADING ON, from CGRID2B[2] = 1.4: a graded
     * band-D failure (the skip cap, 9.9975000111e8) whose central stencil
     * point 1.6 is usable (99.97) and 1.2 graded (9.9975000125e8). A
     * difference never mixes the kinds:
     * the gradient at 1.4 is the backward difference with 1.2, the graded
     * slope, not the jump to 1.6; a point an optimiser proposes outside the
     * box is scored out_of_box_score(FAILSCORE, d), not evaluated.
     * Pins: every backend evaluates 1.4, 1.6 (usable) and 1.2 (graded)
     * first; DLIB:BFGS then reaches the bound 2.0 (95.2545212249, the best)
     * and ends "FTOL_ABS reached" after 6 evaluations, NLOPT:LD_SLSQP the
     * same 6 and ends roundoff-limited, with no "bug: more than iter SQP
     * iterations"; CERES:STEEPEST_DESCENT proposes 3 points outside [1, 2]
     * -- each row carries out_of_box_score(1e9, d) of its own coordinate, d
     * its distance to the box (the box is 1 wide) -- and ends "FTOL_REL
     * reached" after 30, its best 1.985421581 in the box; every other row
     * lies in the box; DIAGNOSTICS counts the out-of-box points ("Out-of-box
     * points : 3", no such line for the bounded two) and the mixed-kind
     * stencils (1, and 2 for CERES:STEEPEST_DESCENT); no stop line; status 0.
     * Before the fix: passes as well (a pin of the fix of
     * this branch; nothing pinned the reproducer through hpo_run). A mutant
     * that differences a graded CENTRAL stencil as it is, v1-dev's rule,
     * fails it: LD_SLSQP 3 evaluations, CERES:STEEPEST_DESCENT 43,
     * DLIB:BFGS without the mixed-kind line. On v1-dev
     * before it (19187dd; the same runs through tadah hpo): LD_SLSQP
     * stopped after 3 evaluations,
     * "STOPPED -- bug: more than iter SQP iterations (...)";
     * CERES:STEEPEST_DESCENT evaluated 26 of its 39 points outside the box
     * and ended "FAILURE (Numerical failure in line search)", its best 2.4,
     * outside the box; DLIB:BFGS made the same six evaluations, its first
     * gradient the jump from 1.2 to 1.6 (about -2.5e9 per unit), and no
     * DIAGNOSTICS line counted a mixed-kind stencil. */
    struct Case {
        std::string lib, algo, status;
        std::size_t evals, mixed, outside;
        double best;
    };
    std::vector<Case> cases = {
        {"DLIB", "BFGS", "DLIB:BFGS: FTOL_ABS reached", 6u, 1u, 0u, 2.0},
        {"NLOPT", "LD_SLSQP",
         "STOPPED -- nlopt roundoff-limited (algorithm converged; consider "
         "adding ftol_rel / xtol_rel / maxeval to stop cleanly next time)",
         6u, 1u, 0u, 2.0}};
#ifdef TADAH_USE_CERES
    cases.push_back({"CERES", "STEEPEST_DESCENT",
                     "CERES:STEEPEST_DESCENT: FTOL_REL reached", 30u, 2u, 3u,
                     1.985421581250845});
#endif
    for (const auto& c : cases) {
        const std::string name = c.lib + ":" + c.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(straddle_block(c.lib, c.algo),
                      straddle_root_fs("1e9", "FAILSCORE_GRADING ON\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.4", name));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            const auto loss = data_rows("loss.tadah");
            REQUIRE(x.size() == c.evals);
            REQUIRE(loss.size() == c.evals);
            CHECK(x[0] == Approx(1.4));
            CHECK(x[1] == Approx(1.6));
            CHECK(x[2] == Approx(1.2));
            CHECK(loss_value(loss[0].back()) ==
                  Approx(9.9975000111e8).epsilon(1e-10));
            CHECK(loss_value(loss[1].back()) < 0.999e9);
            CHECK(loss_value(loss[2].back()) ==
                  Approx(9.9975000125e8).epsilon(1e-10));
            if (c.lib != "CERES") {
                CHECK(x[3] == 2.0);
                CHECK(loss_value(loss[3].back()) ==
                      Approx(95.2545212249).epsilon(1e-10));
            }
            std::size_t outside = 0;
            for (std::size_t k = 0; k < x.size(); ++k) {
                if (x[k] >= 1.0 && x[k] <= 2.0) continue;
                CAPTURE(k, x[k]);
                ++outside;
                const double d = x[k] > 2.0 ? x[k] - 2.0 : 1.0 - x[k];
                CHECK(loss_value(loss[k].back()) ==
                      Approx(tadah::hpo::out_of_box_score(1e9, d))
                          .epsilon(1e-12));
            }
            CHECK(outside == c.outside);
            const std::string journal = read_file("hpo_run.log");
            if (c.outside)
                CHECK_THAT(journal,
                           Catch::Contains("Out-of-box points       : " +
                                           std::to_string(c.outside) + " ("));
            else
                CHECK(journal.find("Out-of-box points") == std::string::npos);
            const std::string mixed = std::to_string(c.mixed);
            CHECK_THAT(journal,
                       Catch::Contains("Mixed-kind stencils     : " + mixed +
                                       " (") &&
                           Catch::Contains("without it): CGRID2B[2] " + mixed +
                                           "\n"));
            CHECK(journal.find("bug: more than iter SQP iterations") ==
                  std::string::npos);
            CHECK(journal.find("stopped: no usable gradient") ==
                  std::string::npos);
            CHECK(journal_status(journal) == c.status);
            const auto best = data_rows("best_params.tadah");
            REQUIRE_FALSE(best.empty());
            const double xb = std::stod(best.back().back());
            CHECK(xb >= 1.0);
            CHECK(xb <= 2.0);
            CHECK(xb == Approx(c.best).epsilon(1e-9));
            const auto best_loss = data_rows("best_loss.tadah");
            REQUIRE_FALSE(best_loss.empty());
            CHECK(std::stod(best_loss.back().back()) < 0.999e9);
            check_run_files();
        }
    }
}

TEST_CASE("No usable gradient: under GRAD_STRATEGY ANALYTICAL or HYBRID a "
          "start inside the failure region stops at its first gradient",
          "[HPO][termination][fd_gradient][analytical]")
{
    /* OALGO 4: the analytical gradient covers CGRID2B[2], so HYBRID
     * differences nothing ("analytical=1 | FD-fallback=0"), as under
     * PC_OBSERVABLE, whose default is ANALYTICAL. At a skip-capped point
     * the analytical gradient is that of the loss before the cap, small but
     * not 0, while the optimiser sees FAILSCORE.
     * Pins: from 1.25 DLIB:BFGS, NLopt LD_LBFGS and CERES:STEEPEST_DESCENT
     * under ANALYTICAL, and DLIB:BFGS and LD_LBFGS under HYBRID, make one
     * evaluation, the start: its failed loss fails the gradient before the
     * analytical pass, the journal says "<backend> stopped: no usable
     * gradient: a failed evaluation at the point (the loss at CGRID2B[2] =
     * 1.25 is 1.797693135e+308)", and the run ends without a usable loss
     * (status 1). A MULTISTART DLIB:BFGS start drawn inside the region
     * (SEED 3: 1.206248063) makes one evaluation and, as the last start,
     * its stop is the trailer, "STOPPED -- no usable gradient: a failed
     * evaluation at the point"; status 0.
     * Before the fix: the gradient was the analytical one under both
     * strategies. DLIB:BFGS made 2 evaluations of FAILSCORE (1.25 and
     * 1.250035746: its line search accepted a step along that small
     * gradient, the loss still FAILSCORE) and ended as converged without a
     * word; LD_LBFGS made 12, its step halved each time, and ended "nlopt
     * failure -- algorithm converged"; CERES:LBFGS (refused since
     * 2026-10-15; the case runs CERES:STEEPEST_DESCENT) made 5 and ended
     * CONVERGENCE at a cost of 1.79769e+308; the MULTISTART start made 2
     * (1.206248063 and 1.206256877) without a word, and the trailer read
     * COMPLETED. CERES:STEEPEST_DESCENT before the fix: passes as well (a
     * pin). */
    const std::string kPointCause =
        "no usable gradient: a failed evaluation at the point";
    const auto block = [](const std::string& lib, const std::string& algo,
                          const std::string& strategy) {
        return plain_block("    TYPE PLAIN\n    LIB " + lib + "\n    ALGO " +
                           algo + "\n    MAXEVAL 60\n    GRAD_STRATEGY " +
                           strategy + "\n" +
                           (strategy == "HYBRID" ? "    PARAM fd_step 0.2\n"
                                                 : ""));
    };
    struct Case { std::string lib, algo, strategy; };
    std::vector<Case> cases = {{"DLIB", "BFGS", "ANALYTICAL"},
                               {"NLOPT", "LD_LBFGS", "ANALYTICAL"},
                               {"DLIB", "BFGS", "HYBRID"},
                               {"NLOPT", "LD_LBFGS", "HYBRID"}};
#ifdef TADAH_USE_CERES
    cases.push_back({"CERES", "STEEPEST_DESCENT", "ANALYTICAL"});
#endif
    for (const auto& c : cases) {
        const std::string name = c.lib + ":" + c.algo;
        DYNAMIC_SECTION(name << ", " << c.strategy) {
            TaRun run(block(c.lib, c.algo, c.strategy), straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.25", name));
            CHECK(status == 1);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() == 1u);
            CHECK(x[0] == Approx(1.25));
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal, name + " stopped: " + kPointCause +
                                        " (the loss at CGRID2B[2] = 1.25 is "
                                        "1.797693135e+308)") == 1u);
            check_no_usable_loss(1u, kDefaultFailScoreRow, kNoUsableLossStatus);
        }
    }
    for (const std::string strategy : {"ANALYTICAL", "HYBRID"}) {
        DYNAMIC_SECTION("MULTISTART, SEED 3, " << strategy) {
            TaRun run(
                "OPTIMIZER\n  INIT\n    METHOD CONFIG\n    METHOD RANDOM 1\n"
                "    SEED 3\n  ENDINIT\n"
                "  STRATEGY\n    TYPE MULTISTART\n    LIB DLIB\n    ALGO BFGS\n"
                "    N_STARTS 2\n    BUDGET_PER_START 30\n"
                "    GRAD_STRATEGY " + strategy + "\n" +
                    (strategy == "HYBRID" ? "    PARAM fd_step 0.2\n" : "") +
                    "  ENDSTRATEGY\nENDOPTIMIZER\n",
                straddle_root());
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "1.9", "MULTISTART"));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() >= 2u);
            CHECK(x[0] == Approx(1.9));
            CHECK(x.back() == Approx(1.206248063).epsilon(1e-9));
            CHECK(x[x.size() - 2] != Approx(x.back()));
            const std::string journal = read_file("hpo_run.log");
            CHECK(count_of(journal, "MULTISTART start 2/2 done: 1 eval(s)") ==
                  1u);
            CHECK(count_of(journal, "DLIB:BFGS stopped: " + kPointCause +
                                        " (the loss at CGRID2B[2] = "
                                        "1.206248063 is 1.797693135e+308)") ==
                  1u);
            CHECK(journal_status(journal) == "STOPPED -- " + kPointCause);
            check_run_files();
        }
    }
}

/* ======================================================================
 * Finite differences and the OPTIM box, end to end (fd_stencil.h).
 * The straddle objective with the ERMSE
 * target 0 falls towards CGRID2B[2] = 2, the upper bound of [1, 2], and
 * the searches start on it (PARAM fd_step 0.2: h = 0.2 of the width).
 * ==================================================================== */

namespace {

/* The DIAGNOSTICS count of one-sided differences of CGRID2B[2], 0 without
 * the line. */
std::size_t one_sided_count(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(One-sided differences\s*: (\d+) )")))
        return static_cast<std::size_t>(std::stoul(m[1]));
    return 0;
}

const std::string kOneSidedAt2 =
    "finite differences: CGRID2B[2] = 2 lies within a step of its upper "
    "bound (OPTIM box [1, 2]): one-sided 3-point difference, at CGRID2B[2] "
    "= 1.8 and CGRID2B[2] = 1.6 with the loss at the point itself";

}  // namespace

TEST_CASE("Finite differences: a search on the upper bound evaluates nothing "
          "outside the box, with every gradient backend and step rule",
          "[HPO][termination][fd_gradient][fd_bounds]")
{
    /* The gradient at the start, CGRID2B[2] = 2, is the one-sided
     * difference at 1.8 and 1.6. Ceres' step rule (PARAM fd_kernel ceres,
     * the default under OPTIM_SCALE NONE) evaluated 2.2, outside the box;
     * HYBRID, with no analytical derivative (OALGO 1), took none and ended
     * at its start. The CERES driver ignores the bounds: its line search
     * proposes points outside the box, and none of them is evaluated: each
     * is scored as a failure graded by its
     * distance to the box, 1.797243...e+308 to 1.797468...e+308 here, with
     * that score's slope as its gradient and no stencil.
     * Pins (CERES:STEEPEST_DESCENT): 21 rows, the start and its stencil, then
     * 18 out-of-box points from 3 down towards 2, none evaluated: no row in
     * the box after the first three, each out-of-box row scored in
     * [0.99975, 1) x FAILSCORE and no higher than the one before; one
     * startup line "ignores the OPTIM box" and one WARNING "lies outside its
     * OPTIM box [1, 2]: it is not evaluated"; DIAGNOSTICS "Out-of-box points
     * : 18"; the best the start 2; the ending "FAILURE (Numerical failure in
     * line search)".
     * Before the fix: the case ran CERES:LBFGS, refused since; the CERES
     * drivers evaluated their points outside the box (the line "lies
     * outside its OPTIM box [1, 2] and is evaluated there", DIAGNOSTICS
     * "Out-of-box evaluations" and "Out-of-box gradients"), each with a
     * one-sided stencil stepping towards the box.
     *
     * Every gradient takes the loss its caller has just evaluated at its
     * point as its centre, NUMERICAL as HYBRID: the start 2, then its
     * stencil 1.8 and 1.6, and no point evaluated twice in a row.
     * Before the fix: NUMERICAL evaluated each gradient's point a second
     * time (2, 2, 1.8, 1.6 at the start, and every gradient's point twice
     * in a row); only HYBRID took the caller's loss. */
    struct Case {
        std::string lib, algo, extra, label;
        bool hybrid = false;
        bool start_gradient = true;   /* its first request is the start's */
    };
    const std::vector<Case> cases = {
        {"DLIB", "BFGS", "", "DLIB:BFGS"},
        {"NLOPT", "LD_LBFGS", "", "NLOPT:LD_LBFGS"},
        {"NLOPT", "LD_MMA", "", "NLOPT:LD_MMA"},
        {"NLOPT", "LD_SLSQP", "", "NLOPT:LD_SLSQP"},
        /* MLSL samples first; a local search reaches the bound later */
        {"NLOPT", "G_MLSL_LDS", inner_block("LD_LBFGS", "      MAXEVAL 10\n"),
         "NLOPT:G_MLSL_LDS, INNER LD_LBFGS", false, false},
        {"DLIB", "BFGS", "    PARAM fd_kernel ceres\n",
         "DLIB:BFGS, fd_kernel ceres"},
        {"DLIB", "BFGS", "    GRAD_STRATEGY HYBRID\n", "DLIB:BFGS, HYBRID", true},
        {"NLOPT", "LD_LBFGS", "    GRAD_STRATEGY HYBRID\n",
         "NLOPT:LD_LBFGS, HYBRID", true},
#ifdef TADAH_USE_CERES
        {"CERES", "STEEPEST_DESCENT", "", "CERES:STEEPEST_DESCENT"},
        {"CERES", "STEEPEST_DESCENT", "    PARAM fd_kernel ceres\n",
         "CERES:STEEPEST_DESCENT, fd_kernel ceres"},
#endif
    };
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.label) {
            TaRun run(straddle_block(c.lib, c.algo, c.extra), straddle_root("0"));
            if (c.hybrid) run.ctx.storage_update("OALGO", {"1"});
            int status = -1;
            REQUIRE_NOTHROW(status = straddle_run(run, "2.0", c.label));
            CHECK(status == 0);
            const auto x = evaluated_cgrid2();
            REQUIRE(x.size() >= 3u);
            const std::string journal = read_file("hpo_run.log");
            /* the start's gradient, which takes the optimiser's own loss at
             * the start (no second evaluation of it) */
            CHECK(x[0] == 2.0);
            if (c.start_gradient) {
                CHECK(x[1] == Approx(1.8).epsilon(1e-12));
                CHECK(x[2] == Approx(1.6).epsilon(1e-12));
            }
            /* one INFO line for the dimension: the first one-sided
             * difference */
            CHECK(count_of(journal, "finite differences: CGRID2B[2] = ") == 1u);
            if (c.start_gradient) CHECK(count_of(journal, kOneSidedAt2) == 1u);
            CHECK(one_sided_count(journal) >= 1u);
            CHECK(journal.find("HYBRID centre losses") == std::string::npos);
            if (c.lib != "CERES") {
                for (double xi : x) {
                    CAPTURE(xi);
                    CHECK((xi >= 1.0 && xi <= 2.0));
                }
                CHECK(journal.find("outside the OPTIM box") == std::string::npos);
            } else {
                /* once, at startup, and once at the first point outside the
                 * box */
                CHECK(count_of(journal, "the STRATEGY-level optimiser: "
                                        "CERES:STEEPEST_DESCENT ignores "
                                        "the OPTIM box: Ceres' "
                                        "line-search solver has no "
                                        "bounds") == 1u);
                CHECK(count_of(journal, "lies outside its OPTIM box [1, 2]: "
                                        "it is not evaluated") == 1u);
                CHECK(journal.find("is evaluated there") == std::string::npos);
                REQUIRE(x.size() == 21u);
                const auto loss = data_rows("loss.tadah");
                REQUIRE(loss.size() == x.size());
                const double fs = std::numeric_limits<double>::max();
                std::size_t outside = 0;
                double previous = fs;
                for (std::size_t k = 3; k < x.size(); ++k) {
                    CAPTURE(k, x[k]);
                    CHECK(x[k] > 2.0);   /* nothing evaluated in the box */
                    const double l = loss_value(loss[k].back());
                    CHECK(l >= 0.99975 * fs);
                    CHECK(l < fs);
                    CHECK(l <= previous);
                    previous = l;
                    ++outside;
                }
                CHECK(outside == 18u);
                CHECK_THAT(journal,
                           Catch::Contains("Out-of-box points       : 18 "));
                CHECK(journal.find("Out-of-box evaluations") ==
                      std::string::npos);
                CHECK(journal_status(journal) ==
                      "STOPPED -- CERES:STEEPEST_DESCENT: FAILURE (Numerical "
                      "failure in line search)");
                const auto best = data_rows("best_params.tadah");
                REQUIRE_FALSE(best.empty());
                CHECK(std::stod(best.back().back()) == 2.0);
            }
            /* the gradient takes the loss the optimiser evaluated at its
             * point: no point twice in a row */
            for (std::size_t k = 1; k < x.size(); ++k) {
                CAPTURE(k);
                CHECK(x[k] != x[k - 1]);
            }
            check_run_files();
        }
    }
}

#ifdef TADAH_USE_CERES
TEST_CASE("Box: CERES:STEEPEST_DESCENT in two dimensions: no point outside "
          "the OPTIM box is evaluated or can be the best",
          "[HPO][termination][fd_gradient][out_of_box]")
{
    /* The straddle objective in two dimensions, CGRID2B[2] in [1, 2] and
     * SGRID2B[2] in [4, 6], from CGRID2B[2] = 1.9: the loss falls towards
     * the bound CGRID2B[2] = 2, and Ceres' GradientProblemSolver, which has
     * no bounds, proposes points beyond it. None of them is evaluated:
     * each is scored
     * out_of_box_score(FAILSCORE, d) of its distance d to the box in the
     * search's coordinates (OPTIM_SCALE AUTO: fractions of each box's
     * width; the default FAILSCORE, the largest double).
     * Pins: every params.tadah row outside the box carries that score (to
     * 1e-12, from the row's own coordinates), every row inside it a loss of
     * the model trained there (usable, or FAILSCORE); DIAGNOSTICS
     * "Out-of-box points" counts the rows outside, at least one;
     * best_params.tadah lies in the box, with a usable loss; status 0.
     * Before the fix: the points outside the box were evaluated there, with
     * losses of the model trained at them (the line "... lies outside its
     * OPTIM box ... and is evaluated there"), and one could be the best.
     * A build without Ceres (TADAH_ENABLE_CERES OFF) does not compile the
     * case, as every other Ceres run of this file: it failed there, "CERES
     * support is not built; reconfigure with -DTADAH_ENABLE_CERES=ON" at
     * startup. */
    const double fs = std::numeric_limits<double>::max();
    TaRun run(plain_block("    TYPE PLAIN\n    LIB CERES\n"
                          "    ALGO STEEPEST_DESCENT\n    MAXEVAL 40\n"
                          "    PARAM fd_step 0.2\n"),
              "LOSS L2\nQFLUSH 1000\nDIGITS 15\n"
              "PC_ERMSE 0 1 --skip-above 9.99995\n"
              "OPTIM CGRID2B (2) 1.0 2.0\nOPTIM SGRID2B (2) 4.0 6.0\n");
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.9",
                                          "CERES:STEEPEST_DESCENT 2-D"));
    CHECK(status == 0);
    const std::vector<std::pair<double, double>> box = {{1.0, 2.0},
                                                        {4.0, 6.0}};
    const auto params = data_rows("params.tadah");
    const auto loss = data_rows("loss.tadah");
    REQUIRE(params.size() == loss.size());
    REQUIRE_FALSE(params.empty());
    std::size_t outside = 0;
    for (std::size_t r = 0; r < params.size(); ++r) {
        CAPTURE(r);
        REQUIRE(params[r].size() >= box.size());
        const std::size_t off = params[r].size() - box.size();
        double s2 = 0.0;
        bool out = false;
        for (std::size_t k = 0; k < box.size(); ++k) {
            const double x = std::stod(params[r][off + k]);
            const double w = box[k].second - box[k].first;
            double e = 0.0;
            if (x > box[k].second) e = (x - box[k].second) / w;
            if (x < box[k].first)  e = (x - box[k].first) / w;
            if (e != 0.0) out = true;
            s2 += e * e;
        }
        const double l = loss_value(loss[r].back());
        CAPTURE(l);
        if (out) {
            ++outside;
            CHECK(l == Approx(tadah::hpo::out_of_box_score(fs, std::sqrt(s2)))
                           .epsilon(1e-12));
            CHECK(l >= 0.99975 * fs);
            CHECK(l < fs);
        } else {
            CHECK((l < 0.999 * fs || loss[r].back() == kDefaultFailScoreRow));
        }
    }
    CHECK(outside > 0u);
    const std::string journal = read_file("hpo_run.log");
    CHECK_THAT(journal, Catch::Contains("Out-of-box points       : " +
                                        std::to_string(outside) + " ("));
    CHECK(journal.find("is evaluated there") == std::string::npos);
    const auto best = data_rows("best_params.tadah");
    REQUIRE_FALSE(best.empty());
    const std::size_t off = best.back().size() - box.size();
    for (std::size_t k = 0; k < box.size(); ++k) {
        CAPTURE(k);
        const double x = std::stod(best.back()[off + k]);
        CHECK(x >= box[k].first);
        CHECK(x <= box[k].second);
    }
    const auto best_loss = data_rows("best_loss.tadah");
    REQUIRE_FALSE(best_loss.empty());
    CHECK(std::stod(best_loss.back().back()) < 0.999 * fs);
    check_run_files();
}
#endif  // TADAH_USE_CERES

TEST_CASE("Finite differences: HYBRID differences every parameter when no "
          "analytical derivative can be formed", "[HPO][termination]"
          "[fd_gradient][hybrid]")
{
    /* OALGO 1: the analytical gradient does not cover the setup, and the
     * journal says every parameter uses finite differences. HYBRID
     * returned g = 0 instead: DLIB:BFGS ended COMPLETED at its start after
     * two evaluations. From
     * 1.9 the search now reaches the bound 2, where the loss is lowest. */
    TaRun run(straddle_block("DLIB", "BFGS", "    GRAD_STRATEGY HYBRID\n"),
              straddle_root("0"));
    run.ctx.storage_update("OALGO", {"1"});
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.9", "DLIB:BFGS HYBRID"));
    CHECK(status == 0);
    const auto x = evaluated_cgrid2();
    REQUIRE(x.size() > 2u);
    CHECK(x[1] == Approx(1.7).epsilon(1e-12));   /* one-sided: 2.1 is outside */
    CHECK(x[2] == Approx(1.5).epsilon(1e-12));
    CHECK(std::find(x.begin(), x.end(), 2.0) != x.end());
    const std::string journal = read_file("hpo_run.log");
    CHECK(count_of(journal, "No analytical gradient for this setup: OALGO is 1") ==
          1u);
    const auto best = data_rows("best_params.tadah");
    REQUIRE_FALSE(best.empty());
    CHECK(std::stod(best.back().back()) == 2.0);
    check_run_files();
}

TEST_CASE("Finite differences: LD_SLSQP from 1.8 spends exactly MAXEVAL",
          "[HPO][termination][fd_gradient]")
{
    /* With the ERMSE target 10.5 LD_SLSQP from 1.8 rejects points in the
     * failure region and goes on, to MAXEVAL: every evaluation in the box
     * [1, 2], a usable one after a failed one, MAXEVAL_REACHED.
     *
     * Pins: exactly 60 evaluations (params.tadah rows) for MAXEVAL 60, the
     * last at CGRID2B[2] = 1.278038548 (to 1e-9: it pins the path; each
     * gradient takes the loss its caller has just evaluated as its centre,
     * a point whose loss failed, 1.0586 and 1.4293 here, is rejected, a
     * point whose stencil fails on one side, 1.4775 the first, takes the
     * one-sided difference on the other,
     * and the call that asks again for 1.7073, evaluated with its gradient,
     * is answered from it without an evaluation); the last is the failed
     * central stencil point of 1.4780 and is not asked for twice in a row.
     * Before the fix: the last at 1.8659506446; the gradient of a point
     * whose stencil failed on one side failed, the point was rejected, and
     * 5 calls asked again for a point evaluated with its gradient.
     * Before the repeat-points fix (2026-10-12) the last was at
     * 1.8678643276: the 5 repeats were evaluated again, MAXEVAL counting
     * them.
     * Before the failure-region fix (2026-10-07) the last was at
     * 1.798850979: 1.0586 and its one-sided stencil 1.2586 and 1.4586 all
     * failed, and SLSQP was handed FAILSCORE there with a zero gradient.
     * Before the centre fix (each gradient evaluated its point a second
     * time) the last was at 1.8540926799.
     * Before the budget fix: 120 evaluations. The gradient provider
     * counted its stencil evaluations against MAXEVAL and NLopt its
     * objective calls; once the provider's budget was spent, SLSQP
     * converged on 1.677689403 and asked for that point again and again --
     * the last 25 evaluations -- until NLopt's own count ended it.
     * Until the bounded stencil (2026-09-29) the central stencil of 1.143
     * took a zero difference of two failures at 1.343 and 0.943, outside
     * the box, and the search ended after 22 evaluations. */
    TaRun run(straddle_block("NLOPT", "LD_SLSQP"), straddle_root("10.5"));
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.8", "NLOPT:LD_SLSQP"));
    CHECK(status == 0);
    const auto rows = data_rows("params.tadah");
    CHECK(rows.size() == 60u);
    REQUIRE_FALSE(rows.empty());
    std::size_t repeats = 0;
    for (auto it = rows.rbegin(); it != rows.rend() &&
                                  it->back() == rows.back().back(); ++it)
        ++repeats;
    CHECK(repeats == 1u);
    CHECK(std::stod(rows.back().back()) == Approx(1.278038548).epsilon(1e-9));
    for (double xi : evaluated_cgrid2()) {
        CAPTURE(xi);
        CHECK((xi >= 1.0 && xi <= 2.0));
    }
    CHECK(went_on_past_a_failure(data_rows("loss.tadah")));
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    check_run_files();
}

TEST_CASE("Snap onto the box: NLopt LN_COBYLA under OPTIM_SCALE NONE rounds a "
          "point on a bound past it, and it is evaluated on the bound",
          "[HPO][termination][fd_bounds][snap]")
{
    /* LN_COBYLA pressed against CGRID2B[2] <= 1.7 asks for 1.7000000000000002,
     * one ulp above the box, from eval 4 on (NLopt's own arithmetic on the
     * bound; OPTIM_SCALE NONE hands it the real box). It was evaluated
     * there -- 29 of the 60 evaluations -- and warned about as a point
     * outside the box; it is evaluated on the bound now and counted as
     * snapped. COBYLA asks for one of its points again
     * among its 60 calls, answered without an evaluation: 59 evaluations
     * (60 until 2026-10-12). */
    const std::string root =
        "OPTIM_SCALE NONE\nLOSS L2\nQFLUSH 1000\nDIGITS 15\n"
        "PC_ERMSE 0 1 --skip-above 9.99995\n"
        "OPTIM CGRID2B (1) 0.3 0.6\nOPTIM CGRID2B (2) 1.5 1.7\n";
    TaRun run(plain_block("    TYPE PLAIN\n    LIB NLOPT\n    ALGO LN_COBYLA\n"
                          "    MAXEVAL 60\n"),
              root);
    run.ctx.storage_update("CGRID2B", {"0.5", "1.6"});
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "NLOPT:LN_COBYLA"));
    CHECK(status == 0);
    const auto x = evaluated_cgrid2();
    CHECK(x.size() == 59u);
    for (double xi : x) {
        CAPTURE(xi);
        CHECK((xi >= 1.5 && xi <= 1.7));
    }
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("lies outside its OPTIM box") == std::string::npos);
    CHECK(journal.find("Out-of-box evaluations") == std::string::npos);
    std::smatch m;
    REQUIRE(std::regex_search(journal, m,
                              std::regex(R"(Snapped onto the box\s*: (\d+) )")));
    CHECK(std::stoul(m[1]) >= 1u);
    check_run_files();
}

/* ======================================================================
 * PRESTAGE refines only usable losses. The top-K buffer took every
 * evaluation's loss, a failed one included (FAILSCORE, a graded failure),
 * and kept it while fewer than TOPK distinct usable losses existed, so a
 * refinement seed could start at a failed point, and an exploration whose
 * every evaluation failed was refined from failed points only. The
 * candidates are the usable losses now: fewer than TOPK are refined as
 * they are and the journal says so; with none the refinement runs from
 * the OPTIMIZER-level INIT points, as it did for an exploration without a
 * candidate (every evaluation a gate reject, say), and the journal says
 * why.
 * ==================================================================== */

namespace {

/* A PRESTAGE of a TADAH:GRID exploration (`grid_points` per OPTIM
 * dimension, no MAXEVAL: the whole grid) refined by NLOPT:LN_SBPLX. */
std::string prestage_grid_block(const std::string& topk,
                                const std::string& grid_points)
{
    return plain_block(
        "    TYPE PRESTAGE\n    TOPK " + topk + "\n"
        "    EXPLORE\n      LIB TADAH\n      ALGO GRID\n"
        "      PARAM GRID_POINTS " + grid_points + "\n    ENDEXPLORE\n"
        "    REFINE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
        "      MAXEVAL 12\n    ENDREFINE\n");
}

/* The journal line that lists the PRESTAGE candidates' losses. */
std::string prestage_losses_line(const std::string& journal)
{
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(PRESTAGE: top-\d+ losses:[^\n]*)")))
        return m[0];
    return {};
}

}  // namespace

TEST_CASE("PRESTAGE: the refinement starts only from usable losses, fewer "
          "than TOPK when fewer exist", "[HPO][termination][prestage]")
{
    /* The straddle fixture (CGRID2B[2] in [1.0, 2.0] fails below about
     * 1.47) from 1.9: the GRID exploration evaluates 1.9, then 1.0, 1.25,
     * 1.5, 1.75 and 2.0; 1.0 and 1.25 fail (the default FAILSCORE).
     * Pins: the candidates are the four usable losses -- "PRESTAGE: top-4
     * losses:" without a failed one --, the journal says that there are
     * fewer than TOPK 6 and why ("PRESTAGE: 4 usable candidate(s) for TOPK
     * 6 (the exploration made 6 evaluation(s), 2 without a usable loss);
     * refining 4"), the refinement runs four starts ("REFINEMENT: refining
     * top-4 seed(s)", "MULTISTART start 4/4", no fifth), status 0.
     * Before the fix: "PRESTAGE: top-6 losses: 95.2545 96.0496 99.2483
     * 99.9981 1.79769e+308 ...", "refining top-6 seed(s)": two of the six
     * refinement starts began at the failed points 1.0 and 1.25. */
    TaRun run(prestage_grid_block("6", "5"), straddle_root());
    int status = -1;
    REQUIRE_NOTHROW(status = straddle_run(run, "1.9", "PRESTAGE"));
    CHECK(status == 0);
    const auto x = evaluated_cgrid2();
    REQUIRE(x.size() >= 6u);
    const std::vector<double> grid = {1.9, 1.0, 1.25, 1.5, 1.75, 2.0};
    for (std::size_t i = 0; i < grid.size(); ++i) {
        INFO("exploration evaluation " << i + 1);
        CHECK(x[i] == Approx(grid[i]));
    }
    const std::string journal = read_file("hpo_run.log");
    const std::string losses = prestage_losses_line(journal);
    INFO(losses);
    CHECK_THAT(losses, Catch::StartsWith("PRESTAGE: top-4 losses: "));
    CHECK(losses.find("e+308") == std::string::npos);
    CHECK(count_of(journal,
                   "PRESTAGE: 4 usable candidate(s) for TOPK 6 (the "
                   "exploration made 6 evaluation(s), 2 without a usable "
                   "loss); refining 4") == 1u);
    CHECK(count_of(journal, "REFINEMENT: refining top-4 seed(s)") == 1u);
    CHECK(count_of(journal, "MULTISTART start 4/4 done") == 1u);
    CHECK(journal.find("MULTISTART start 5/") == std::string::npos);
    check_run_files();
}

TEST_CASE("PRESTAGE: an exploration without a usable loss leaves the "
          "refinement to the OPTIMIZER-level INIT points",
          "[HPO][termination][prestage][no_usable_loss]")
{
    /* Every evaluation fails (a skip cap below every loss); the GRID
     * exploration evaluates the INIT point and the 2^3 nodes.
     * Pins: the journal says that the exploration's 9 evaluations produced
     * no usable loss and where the refinement starts instead ("PRESTAGE: no
     * evaluation of the exploration produced a usable loss (9
     * evaluation(s)); refining from the OPTIMIZER-level INIT point(s)
     * instead", a WARNING), there is no candidate line ("PRESTAGE: top-")
     * and no "REFINEMENT: refining"; the refinement's first evaluation (eval
     * 10) is the INIT point (eval 1's parameters) and it spends its MAXEVAL
     * 12 (21 rows); the run ends without a usable loss (status 1, "STOPPED
     * -- no evaluation produced a usable loss").
     * Before the fix: "PRESTAGE: no evaluation of the exploration produced
     * a usable loss (9 evaluation(s)); nothing to refine", the run ended
     * after the exploration's 9 rows. (Before the top-K took usable losses
     * only, the three failed points were the candidates: "PRESTAGE: top-3
     * losses: 1.79769e+308 1.79769e+308 1.79769e+308", "REFINEMENT:
     * refining top-3 seed(s)", 21 rows.) */
    TaRun run(prestage_grid_block("3", "2"), kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "an all-failed PRESTAGE"));
    CHECK(status == 1);
    check_no_usable_loss(21, kDefaultFailScoreRow, kNoUsableLossStatus);
    const std::string journal = read_file("hpo_run.log");
    CHECK(count_of(journal,
                   "WARNING   PRESTAGE: no evaluation of the exploration "
                   "produced a usable loss (9 evaluation(s)); refining from "
                   "the OPTIMIZER-level INIT point(s) instead") == 1u);
    CHECK(journal.find("nothing to refine") == std::string::npos);
    CHECK(journal.find("REFINEMENT: refining") == std::string::npos);
    CHECK(prestage_losses_line(journal).empty());
    const auto params = data_rows("params.tadah");
    REQUIRE(params.size() == 21u);
    /* The columns after ITERATION and SLOT: the parameters. */
    CHECK(std::vector<std::string>(params[9].begin() + 2, params[9].end()) ==
          std::vector<std::string>(params[0].begin() + 2, params[0].end()));
}

/* An exploration that asked for no point -- no evaluation, and no call
 * answered from the run's record -- has no candidate, and no failed
 * evaluation either: every exploration a parsed configuration sets asks
 * for at least one point (MAXEVAL >= 1, NLopt counting a call the record
 * answers too; every driver makes its first call, also when MAXTIME has
 * passed before it -- the next case; a stagnation stop is raised at an
 * evaluation or at a call answered from the record), so it is a driver
 * that returned without asking -- or a spec built outside the parser that
 * asks for nothing --, which a refinement from the OPTIMIZER-level INIT
 * points would hide. An exploration whose every call the record answers
 * makes no evaluation and hands on its candidates ("Run record: a PRESTAGE
 * exploration answered from the record ...", and the NOISE CALIBRATION
 * case below).
 * Pins: StagedPolicy over the parsed PRESTAGE spec whose exploration
 * (TADAH:RANDOM) is given MAXEVAL 0 and no INIT point throws
 * std::logic_error "PRESTAGE: the exploration (EXPLORE block) TADAH:RANDOM
 * returned without evaluating a point, so it has no candidate to refine
 * (it asked for no point: no call was answered from the run's record
 * either, and every exploration a configuration sets asks for at least
 * one point; its driver, or a spec built outside the parser, asked for
 * none)", no evaluation is made (the journal's census is unchanged) and
 * the journal has no "running outer optimiser as-is".
 * Before the fix: a WARNING "PRESTAGE: produced no candidates -- running
 * outer optimiser as-is" (journal only), and the refinement ran from the
 * OPTIMIZER-level INIT point: 3 evaluations (its MAXEVAL), no error.
 * Before the message was corrected: its parenthesis
 * read "every exploration a configuration sets evaluates at least one
 * point: ...", which the run's record made false (an exploration can make
 * no evaluation and go on): the Contains checks fail. */
TEST_CASE("PRESTAGE: an exploration that made no evaluation is an error, not "
          "a refinement from the INIT points",
          "[HPO][termination][prestage][no_evaluation]")
{
    HostFixture fx(kRootLines,
                   plain_block("    TYPE PRESTAGE\n    LIB TADAH\n"
                               "    ALGO RANDOM\n    MAXEVAL 3\n    SEED 1\n"
                               "    EXPLORE\n      LIB TADAH\n"
                               "      ALGO RANDOM\n      MAXEVAL 3\n"
                               "      SEED 2\n    ENDEXPLORE\n"));
    tadah::hpo::OptimizerSpec spec = fx.trg.spec;
    REQUIRE(spec.prestage);
    REQUIRE_FALSE(spec.optim_init.empty());
    spec.prestage->stop.maxeval = 0;
    spec.prestage->optim_init.clear();
    tadah::hpo::StagedPolicy<ParamVec, ::LammpsRunner> policy(1);
    const std::size_t evals_before = tadah::hpo::HPOLog::instance().n_eval();
    std::string what;
    bool logic = false;
    try {
        policy.run(spec, fx.host);
    } catch (const std::logic_error& e) {
        logic = true;
        what = e.what();
    } catch (const std::exception& e) {
        what = e.what();
    }
    CHECK(logic);
    CHECK_THAT(what, Catch::StartsWith(
                         "PRESTAGE: the exploration (EXPLORE block) "
                         "TADAH:RANDOM returned without evaluating a point"));
    CHECK_THAT(what, Catch::Contains(
                         "so it has no candidate to refine (it asked for no "
                         "point: no call was answered from the run's record "
                         "either, and every exploration a configuration sets "
                         "asks for at least one point; its driver, or a spec "
                         "built outside the parser, asked for none)"));
    CHECK_THAT(what, !Catch::Contains("evaluates at least one point"));
    CHECK(tadah::hpo::HPOLog::instance().n_eval() == evals_before);
    tadah::hpo::HPOLog::instance().finish("test-end");
    const std::string journal = read_file(fx.run.sd.dir / "hpo_run.log");
    INFO(journal);
    CHECK(journal.find("running outer optimiser as-is") == std::string::npos);
}

/* Pins: an exploration whose every call the run's record answers makes no
 * evaluation and is no error: under NOISE CALIBRATION 2 the calibration
 * evaluates the seed point, the configuration's (rows 1-2), and a PRESTAGE
 * exploration NLOPT:LN_SBPLX with MAXEVAL 1 asks for that point alone,
 * which the record answers ("NLopt finished: MAXEVAL_REACHED -- maximum
 * evaluations exhausted after 0 of MAXEVAL 1 evaluation(s); NLopt counts
 * its calls, and 1 of them asked for points the run had evaluated and were
 * answered without an evaluation"); the seed is its candidate ("PRESTAGE:
 * top-1 losses: ..."), the refinement LN_SBPLX (MAXEVAL 4, TOPK 1) runs
 * from it (rows 3-5: NLopt charges its first call, the seed again, answered
 * from the record), and the run ends with status 0, its trailer the
 * refinement's MAXEVAL_REACHED, with no "returned without evaluating a
 * point".
 * Before the fix: passes as well (a pin of the rule that the docs and the
 * changelogs now state: they said that every exploration evaluates at
 * least one point, so one that made none ends FAILED). */
TEST_CASE("PRESTAGE: an exploration whose every call the run's record "
          "answers makes no evaluation and hands on its candidate",
          "[HPO][termination][prestage][no_evaluation][repeat_points]")
{
    TaRun run("OPTIMIZER\n  NOISE\n    CALIBRATION 2\n  ENDNOISE\n"
              "  STRATEGY\n    TYPE PRESTAGE\n    LIB NLOPT\n"
              "    ALGO LN_SBPLX\n    MAXEVAL 4\n    TOPK 1\n"
              "    EXPLORE\n      LIB NLOPT\n      ALGO LN_SBPLX\n"
              "      MAXEVAL 1\n    ENDEXPLORE\n  ENDSTRATEGY\n"
              "ENDOPTIMIZER\n");
    int status = -1;
    REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
    CHECK(status == 0);
    check_run_files();
    const std::string journal = read_file("hpo_run.log");
    INFO("journal:\n" << journal);
    CHECK(journal.find("NLopt finished: MAXEVAL_REACHED -- maximum "
                       "evaluations exhausted after 0 of MAXEVAL 1 "
                       "evaluation(s); NLopt counts its calls, and 1 of them "
                       "asked for points the run had evaluated and were "
                       "answered without an evaluation") != std::string::npos);
    CHECK(journal.find("PRESTAGE: top-1 losses:") != std::string::npos);
    CHECK(journal.find("REFINEMENT: refining top-1") != std::string::npos);
    CHECK(journal.find("returned without evaluating a point") ==
          std::string::npos);
    CHECK(journal_status(journal).rfind("MAXEVAL_REACHED", 0) == 0);
    const auto params = data_rows("params.tadah");
    REQUIRE(params.size() == 5u);
    /* rows 1-2: the calibration's seed point, twice */
    CHECK(std::vector<std::string>(params[0].begin() + 2, params[0].end()) ==
          std::vector<std::string>(params[1].begin() + 2, params[1].end()));
}

/* MAXTIME keeps an evaluation from starting once it has passed, except a
 * search's first: every search makes at least one evaluation, as the TADAH
 * drivers (they read MAXTIME after an evaluation), NLopt and DLIB:GFS's
 * INIT points do. MAXTIME 1e-9 has passed before any search starts.
 * Pins: with MAXTIME 1e-9, a PLAIN DLIB:GFS without an INIT point (an
 * INIT block whose WARM seed WARM_ON_MISMATCH skip drops, its value
 * outside the OPTIM box; the case used METHOD RANDOM 0 until a K of 0 was
 * refused) and a PLAIN ASA each make one evaluation and end with
 * status 0 ("DLIB:GFS stopped: MAXTIME exceeded after 1 evaluation(s)",
 * "ASA stopped: MAXTIME exceeded after 1 host eval(s)", the run's status
 * their MAXTIME stop); a PRESTAGE (TADAH:RANDOM, MAXEVAL 2) whose
 * exploration is one of them makes that one exploration evaluation and
 * then the refinement's two, status 0, and the journal names no error.
 * Before the fix: the search made no evaluation; the PLAIN run ended with
 * status 1 ("STOPPED -- no evaluation produced a usable loss"), the
 * PRESTAGE run with status 1 and FAILED ("PRESTAGE: the exploration
 * (EXPLORE block) DLIB:GFS returned without evaluating a point ..."; ASA:ASA
 * likewise). */
TEST_CASE("MAXTIME: a search makes its first evaluation when MAXTIME has "
          "passed before it", "[HPO][termination][stop_criteria][maxtime]")
{
    struct Case {
        const char* name;
        std::string search;     // the search's lines, without its TYPE
        std::string init;       // its INIT block, if any
        std::string stopped;    // its journal line
        std::string status;     // the run's status under PLAIN
    };
    /* A WARM seed whose CGRID2B (1), 5, lies outside the box [0.1, 0.9]:
     * WARM_ON_MISMATCH skip drops it, and the chain gives no point. */
    ScratchDir seeds;
    const fs::path outside = seeds.dir / "outside.tadah";
    {
        std::ofstream out(outside);
        out << "CGRID2B 5.0 1.5\nSGRID2B 1.0 5.0\n";
    }
    const std::vector<Case> cases = {
        {"DLIB:GFS without an INIT point",
         "LIB DLIB\nALGO GFS\nMAXTIME 1e-9\n",
         "INIT\nMETHOD WARM " + outside.string() +
             "\nWARM_ON_MISMATCH skip\nENDINIT\n",
         "DLIB:GFS stopped: MAXTIME exceeded after 1 evaluation(s)",
         "DLIB:GFS: MAXTIME exceeded"},
        {"ASA", "LIB ASA\nALGO ASA\nMAXTIME 1e-9\nSEED 3\n", "",
         "ASA stopped: MAXTIME exceeded after 1 host eval(s)",
         "ASA: MAXTIME exceeded"},
    };
    /* `lines` indented by `indent` spaces, line by line. */
    const auto indented = [](const std::string& lines, std::size_t indent) {
        std::istringstream in(lines);
        std::string out;
        for (std::string l; std::getline(in, l);)
            out += std::string(indent, ' ') + l + "\n";
        return out;
    };
    for (const auto& c : cases) {
        INFO(c.name);
        SECTION(std::string("PLAIN ") + c.name) {
            TaRun run("OPTIMIZER\n" + indented(c.init, 2) + "  STRATEGY\n" +
                      "    TYPE PLAIN\n" + indented(c.search, 4) +
                      "  ENDSTRATEGY\nENDOPTIMIZER\n");
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK(status == 0);
            CHECK(data_rows("loss.tadah").size() == 1u);
            CHECK_THAT(journal, Catch::Contains(c.stopped));
            CHECK(journal_status(journal) == c.status);
        }
        SECTION(std::string("the PRESTAGE exploration ") + c.name) {
            TaRun run(plain_block(
                "    TYPE PRESTAGE\n    LIB TADAH\n    ALGO RANDOM\n"
                "    MAXEVAL 2\n    SEED 1\n    EXPLORE\n" +
                indented(c.init + c.search, 6) + "    ENDEXPLORE\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = tadah::hpo::hpo_run(run.ctx));
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK(status == 0);
            CHECK(data_rows("loss.tadah").size() == 3u);
            CHECK_THAT(journal, Catch::Contains(c.stopped));
            CHECK_THAT(journal, Catch::Contains(
                                    "REFINEMENT: refining top-1 seed(s) via "
                                    "TADAH:RANDOM"));
            CHECK(journal.find("returned without evaluating") ==
                  std::string::npos);
            CHECK(journal_status(journal) == "TADAH:RANDOM: MAXEVAL reached");
        }
    }
}

TEST_CASE("PRESTAGE under RESTART: an exploration that stagnates without a "
          "usable loss hands over to a refinement from the INIT points",
          "[HPO][termination][prestage][restart][no_usable_loss]")
{
    /* Every evaluation fails, so no evaluation is a new best and each
     * phase stagnates after 3 evaluations.
     * Pins: as without RESTART, a stagnated exploration hands over to the
     * refinement, here without a candidate: each round's exploration says
     * so ("PRESTAGE: no evaluation of the exploration produced a usable
     * loss (3 evaluation(s)); refining from the OPTIMIZER-level INIT
     * point(s) instead", twice) and is not refined ("REFINEMENT: refining"
     * never); the refinement from the INIT point stagnates and reports it
     * ("REFINEMENT: 1 of 1 seed(s) stopped on stagnation; reporting
     * stagnation to RESTART", twice), RESTART restarts once and its budget
     * then ends the run, which has no usable loss (status 1, "STOPPED -- no
     * evaluation produced a usable loss"), after 12 evaluations.
     * Before the fix: each exploration said "...; nothing to refine;
     * reporting stagnation to RESTART" and nothing was refined: 6
     * evaluations. (Before the top-K took usable losses only, each
     * exploration's failed points were refined, "REFINEMENT: refining top-2
     * seed(s)": 18 evaluations.) */
    TaRun run(plain_block(
                  "    TYPE PRESTAGE\n    TOPK 2\n"
                  "    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 20\n"
                  "    EXPLORE\n      LIB TADAH\n      ALGO RANDOM\n"
                  "      MAXEVAL 50\n      SEED 3\n    ENDEXPLORE\n"
                  "    RESTART\n      AFTER_STAGNATION 3\n      MAX 1\n"
                  "      METHOD RANDOM 1\n      SEED 7\n    ENDRESTART\n"),
              kAllFailRoot);
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "an all-failed PRESTAGE under "
                                        "RESTART"));
    CHECK(status == 1);
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal.find("REFINEMENT: refining") == std::string::npos);
    CHECK(journal.find("nothing to refine") == std::string::npos);
    CHECK(count_of(journal,
                   "PRESTAGE: no evaluation of the exploration produced a "
                   "usable loss (3 evaluation(s)); refining from the "
                   "OPTIMIZER-level INIT point(s) instead") == 2u);
    CHECK(count_of(journal,
                   "REFINEMENT: 1 of 1 seed(s) stopped on stagnation; "
                   "reporting stagnation to RESTART") == 2u);
    CHECK_THAT(journal, Catch::Contains("RESTART 1/1"));
    CHECK_THAT(journal, Catch::Contains("RESTART: stagnation reached, max "
                                        "restarts (1) used; stopping"));
    check_no_usable_loss(12, kDefaultFailScoreRow, kNoUsableLossStatus);
}

/* ======================================================================
 * GRAD_STRATEGY ANALYTICAL at a candidate the bump-support gate rejects
 * before training. Under FAILSCORE_GRADING ON such a candidate scores a
 * graded failure, and the analytical pass ran there: HPO_Host's trainer
 * slot holds no fit of the point, so it handed every parameter to finite
 * differences, which ANALYTICAL took for a parameter without an analytical
 * derivative -- an error that ended the run FAILED. The point has no
 * usable gradient instead (gradient_provider.h), which NLopt and Ceres
 * reject as any such point.
 * ==================================================================== */

namespace {

/* A D2_Blip channel (w = 2) under a hard Cut_Dummy cutoff at RCUT2B 5: a
 * bump whose support c + 2/eta passes 5 is rejected before training
 * (BUMPGATE CANDIDATE, the default), graded band D. The search starts at
 * CGRID2B[1] = 3.5, SGRID2B[1] = 2.0 (support 4.5); the loss falls
 * towards SGRID2B[1] = 1, where every CGRID2B[1] above 3 is rejected.
 * Energies -1 (train and validation), LAMBDA 1. */
void make_bump_gate_run(TaRun& run)
{
    write_ta_db(run.sd.dir / "train.db", 0.0, -1.0);
    write_ta_db(run.sd.dir / "val.db", 0.07, -1.0);
    run.ctx.storage_update("TYPE2B", {"D2_Blip", "2", "2", "Ta", "Ta"});
    run.ctx.storage_update("CGRID2B", {"3.5", "3.0"});
    run.ctx.storage_update("SGRID2B", {"2.0", "1.0"});
    run.ctx.storage_update("RCTYPE2B", {"Cut_Dummy"});
    run.ctx.storage_update("LAMBDA", {"1"});
}

const std::string kBumpGateRoot =
    "LOSS L2\n"
    "PC_EMAE 0 1\n"
    "FAILSCORE 1e9\n"
    "FAILSCORE_GRADING ON\n"
    "QFLUSH 1000\n"
    "DIGITS 15\n"
    "OPTIM CGRID2B (1) 3.0 4.8\n"
    "OPTIM SGRID2B (1) 1.0 3.0\n";

}  // namespace

TEST_CASE("GRAD_STRATEGY ANALYTICAL: a candidate the bump gate rejects "
          "before training has no usable gradient, and the run goes on",
          "[HPO][termination][analytical][bumpgate]")
{
    /* Pins, for every NLopt LD_* search (MAXEVAL 20): the run
     * evaluates a rejected candidate (a graded loss in [0.999e9, 1e9)),
     * does not throw, steps back from it and evaluates further points (the
     * first rejected candidate is not the last evaluation), and ends with
     * status 0, a usable best and its search's MAXEVAL status
     * ("MAXEVAL_REACHED -- maximum evaluations exhausted"), or, for LD_LBFGS
     * and LD_TNEWTON_PRECOND_RESTART,
     * which NLopt ends after 15 evaluations right after a rejected
     * candidate, the stop at it ("STOPPED -- no usable gradient: a failed
     * evaluation at the point", the stop line naming the graded failure and
     * "NLopt ended right after it (nlopt failure)"); the journal never
     * says "no analytical gradient for".
     * Before the fix: hpo_run threw "GRAD_STRATEGY=ANALYTICAL: no
     * analytical gradient for 2 parameters at this point (indices: [0, 1])
     * ..." at the first rejected candidate, after 5-9 evaluations, and the
     * run ended "FAILED -- GRAD_STRATEGY=ANALYTICAL: ...". Until 2026-10-12
     * ANALYTICAL ran its analytical pass at a graded failure, one counted
     * evaluation, so every rejected candidate counted twice and LD_LBFGS
     * and LD_TNEWTON_PRECOND_RESTART reached MAXEVAL 20 first. DLIB:BFGS
     * and CERES:STEEPEST_DESCENT ask no gradient at a rejected candidate
     * of this search, so they are not among the cases; CERES:LBFGS, WOLFE
     * and NLCG were, until they were refused. Before the fix: the NLopt
     * cases pass as well (a pin). */
    const std::vector<GradientBackend> backends = {
        {"NLOPT", "LD_LBFGS"}, {"NLOPT", "LD_MMA"}, {"NLOPT", "LD_SLSQP"},
        {"NLOPT", "LD_CCSAQ"}, {"NLOPT", "LD_TNEWTON"},
        {"NLOPT", "LD_TNEWTON_PRECOND_RESTART"}, {"NLOPT", "LD_VAR2"}};
    for (const auto& b : backends) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TaRun run(plain_block("    TYPE PLAIN\n    LIB " + b.lib +
                                  "\n    ALGO " + b.algo +
                                  "\n    MAXEVAL 20\n"
                                  "    GRAD_STRATEGY ANALYTICAL\n"),
                      kBumpGateRoot);
            make_bump_gate_run(run);
            int status = -1;
            REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                                name));
            CHECK(status == 0);
            const auto rows = data_rows("loss.tadah");
            std::size_t first_rejected = rows.size();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const double l = loss_value(rows[i].back());
                if (l >= 0.999e9 && l < 1e9) {
                    first_rejected = i;
                    break;
                }
            }
            REQUIRE(first_rejected < rows.size());
            CHECK(first_rejected + 1 < rows.size());
            const std::string journal = read_file("hpo_run.log");
            CHECK(journal.find("no analytical gradient for") ==
                  std::string::npos);
            if (b.algo == "LD_LBFGS" || b.algo == "LD_TNEWTON_PRECOND_RESTART") {
                CHECK(rows.size() == 15u);
                CHECK(journal_status(journal) ==
                      "STOPPED -- no usable gradient: a failed evaluation at "
                      "the point");
                CHECK_THAT(journal,
                           Catch::Contains(name + " stopped: no usable "
                                           "gradient: a failed evaluation at "
                                           "the point (the loss at "));
                CHECK_THAT(journal,
                           Catch::Contains(": a graded failure, whose grading "
                                           "the analytical gradient does not "
                                           "see; GRAD_STRATEGY HYBRID takes "
                                           "finite differences there); NLopt "
                                           "ended right after it (nlopt "
                                           "failure)"));
            } else {
                CHECK(journal_status(journal) ==
                      (b.lib == "NLOPT"
                           ? std::string("MAXEVAL_REACHED -- maximum "
                                         "evaluations exhausted")
                           : name + ": MAXEVAL reached"));
            }
            check_run_files();
        }
    }
}

TEST_CASE("PRESTAGE: an exploration whose every candidate the bump gate "
          "rejects is refined from the OPTIMIZER-level INIT point",
          "[HPO][termination][prestage][bumpgate]")
{
    /* The bump-gate fixture above: the exploration (its own INIT RANDOM 2,
     * TADAH:RANDOM MAXEVAL 2, SEED 2) evaluates points whose bump support
     * passes RCUT2B 5, rejected before training; the OPTIMIZER-level INIT
     * point (3.5, 2.0; support 4.5) is usable.
     * Pins: both exploration evaluations are gate rejects (a graded loss in
     * [0.999e9, 1e9), "eval 1: bump-support gate reject" and "eval 2: ..."
     * in the journal); the journal says that they produced no
     * usable loss and where the refinement starts instead ("PRESTAGE: no
     * evaluation of the exploration produced a usable loss (2
     * evaluation(s)); refining from the OPTIMIZER-level INIT point(s)
     * instead", a WARNING); the refinement's first evaluation (eval 3) is
     * the INIT point, it spends its MAXEVAL 20 of NLopt's calls and finds
     * usable losses: 18 rows (4 of its calls ask again for a point of its
     * own search with a usable loss, answered without an evaluation; 22
     * rows until 2026-10-12), status 0, "MAXEVAL_REACHED -- maximum
     * evaluations exhausted", a usable best.
     * Before the fix: "PRESTAGE: no evaluation of the exploration produced
     * a usable loss (2 evaluation(s)); nothing to refine", and the run
     * ended after the exploration's 2 rows without a usable loss (status 1,
     * "STOPPED -- no evaluation produced a usable loss"). (Before the top-K
     * took usable losses only, gate rejects were never candidates either,
     * and the refinement ran from the INIT point after "PRESTAGE: produced
     * no candidates -- running outer optimiser as-is".) */
    TaRun run(plain_block(
                  "    TYPE PRESTAGE\n    TOPK 2\n"
                  "    EXPLORE\n      INIT\n        METHOD RANDOM 2\n"
                  "        SEED 2\n      ENDINIT\n"
                  "      LIB TADAH\n      ALGO RANDOM\n      MAXEVAL 2\n"
                  "      SEED 2\n    ENDEXPLORE\n"
                  "    LIB NLOPT\n    ALGO LN_SBPLX\n    MAXEVAL 20\n"),
              kBumpGateRoot);
    make_bump_gate_run(run);
    int status = -1;
    REQUIRE_NOTHROW(status = run_within(run, std::chrono::seconds(120),
                                        "a gate-rejected PRESTAGE"));
    CHECK(status == 0);
    const auto rows = data_rows("loss.tadah");
    REQUIRE(rows.size() == 18u);
    for (std::size_t i = 0; i < 2; ++i) {
        INFO("exploration evaluation " << i + 1);
        const double l = loss_value(rows[i].back());
        CHECK((l >= 0.999e9 && l < 1e9));
    }
    const std::string journal = read_file("hpo_run.log");
    CHECK(count_of(journal, "eval 1: bump-support gate reject") == 1u);
    CHECK(count_of(journal, "eval 2: bump-support gate reject") == 1u);
    CHECK(count_of(journal,
                   "WARNING   PRESTAGE: no evaluation of the exploration "
                   "produced a usable loss (2 evaluation(s)); refining from "
                   "the OPTIMIZER-level INIT point(s) instead") == 1u);
    CHECK(journal.find("nothing to refine") == std::string::npos);
    CHECK(journal.find("REFINEMENT: refining") == std::string::npos);
    const auto params = data_rows("params.tadah");
    REQUIRE(params.size() == 18u);
    CHECK(std::stod(params[2].at(2)) == Approx(3.5));
    CHECK(std::stod(params[2].at(3)) == Approx(2.0));
    CHECK(journal_status(journal) ==
          "MAXEVAL_REACHED -- maximum evaluations exhausted");
    check_run_files();
}

/* Pins: hpo_run of a configuration without HPOTARGET is refused at startup
 * with a message that names where the run's own goes, by where its keys
 * came from (the KeyProvenance hpo_run is given), which the journal records
 * ("startup: ..."), and nothing is evaluated (no loss.tadah):
 *   - no provenance (a context a program built) and a -c run of a
 *     configuration file (the provenance of `tadah hpo -c config.tadah`):
 *     "tadah hpo: no HPOTARGET: name the hpotarget file with --hpotarget,
 *     or with an HPOTARGET line in a configuration file without WEIGHTS.";
 *   - the -c run of old.tadah, a potential whose HPOTARGET line the parse
 *     did not read (a best_pot.tadah of an earlier version has one; the CLI
 *     requires --hpotarget there, a program can build the run): the first
 *     message and "old.tadah is a potential (it has WEIGHTS): its HPOTARGET
 *     config.hpo line was a setting of the run that wrote it and is not
 *     read.".
 * Before the fix: every run was refused with one message, whatever its
 * input, which named no configuration file without WEIGHTS and no potential
 * whose line was not read: the message checks fail. Before the HPOTARGET
 * check: Context::get threw "Key not found in the Context object:
 * HPOTARGET". (A pin: these legs pass before the removal of the task
 * files as well.) */
TEST_CASE("hpo_run without HPOTARGET is refused, naming where the run's own "
          "goes",
          "[hpo][startup][run_settings]")
{
    using tadah::mlip::KeyProvenance;
    const std::string none_given =
        "tadah hpo: no HPOTARGET: name the hpotarget file with --hpotarget, "
        "or with an HPOTARGET line in a configuration file without WEIGHTS.";
    const std::string not_read =
        " old.tadah is a potential (it has WEIGHTS): its HPOTARGET config.hpo "
        "line was a setting of the run that wrote it and is not read.";
    const tadah::core::SkippedKeyLine old_line{"old.tadah", 12, "HPOTARGET",
                                               {"config.hpo"}};
    struct Leg {
        const char* name;
        bool with_provenance;
        const char* file;
        std::vector<tadah::core::SkippedKeyLine> skipped;
        std::string message;
    };
    const std::vector<Leg> legs = {
        {"no provenance", false, "", {}, none_given},
        {"-c config.tadah", true, "config.tadah", {}, none_given},
        {"-c old.tadah", true, "old.tadah", {old_line}, none_given + not_read},
    };
    for (const Leg& leg : legs) {
        INFO(leg.name);
        TaRun run(plain_block(
            "    TYPE PLAIN\n    LIB TADAH\n    ALGO RANDOM\n    MAXEVAL 1\n"));
        run.ctx.remove("HPOTARGET");
        const KeyProvenance provenance(tadah::core::Context(), run.ctx,
                                       leg.file, {}, leg.skipped);
        std::string what;
        try {
            (void)tadah::hpo::hpo_run(run.ctx, leg.with_provenance
                                                   ? &provenance
                                                   : nullptr);
        } catch (const std::exception& e) {
            what = e.what();
        }
        CHECK(what == leg.message);
        CHECK(read_file("hpo_run.log").find("startup: " + leg.message) !=
              std::string::npos);
        CHECK_FALSE(fs::exists("loss.tadah"));
    }
}

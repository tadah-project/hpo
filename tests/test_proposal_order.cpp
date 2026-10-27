/* The numbers of the evaluations that a parallel optimiser runs at once,
 * and the order in which their bests are committed, on the Ta test data
 * (tests/data/ta/common).
 *
 * TADAH:ANNEAL_PT, TADAH:ANNEAL_PA and ASA:ASA with a block of states
 * (Gener_Block > 1) evaluate several points at once on THREADS workers.
 * Each case drives the optimiser on an HPO_Host of the shared Ta model
 * (RCUT2B in [4, 6], SGRIDMB in [0.1, 0.4]; an evaluation takes a few
 * milliseconds) through a host that holds each evaluation back by a delay
 * of its slot, the replica, walker or block state (warmstart::
 * current_chain()), before it reaches HPO_Host::operator(). The delays make
 * the evaluations of a batch start and finish in an order of their own:
 * reverse slot order, or slot order, whatever the timing of the machine.
 * A seeded run then has to write the same files at any THREADS and in
 * either order: the evaluated points, the evaluation numbers (#ITERATION),
 * the slots (SLOT), the bests (best_* files, best_pot.tadah).
 *
 * The files are compared whole, except the times they record -- the
 * IT-TIME/ms column of outvar.tadah and best_outvar.tadah and the date line
 * of the header of best_pot.tadah -- and the run's scratch directory, which
 * the training record of best_pot.tadah names. */

#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/run_stop.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/warmstart_cache.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using tadah::hpo::HPOLog;

const fs::path& ta_common_dir()
{
    static const fs::path dir = fs::path(TADAH_HPO_TA_DATA_DIR) / "common";
    return dir;
}

/* A scratch working directory: the host writes its files into the current
 * directory. */
struct ScratchDir {
    fs::path dir;
    fs::path original_cwd;
    ScratchDir()
    {
        static unsigned counter = 0;
        dir = fs::temp_directory_path() /
              ("tadah_proposal_order_" + std::to_string(counter++) + "_" +
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

/* The host the optimiser drives: each call waits `delay_ms[slot]`
 * milliseconds, then `before(slot, n)` may throw (n: the call of the slot,
 * from 1), then `map(slot, n, x)` may replace the point, the point goes to
 * HPO_Host::operator(), and `after(slot, n)` follows its return. */
struct PacedHost {
    Host& host;
    std::vector<int> delay_ms;
    std::function<void(std::size_t, std::size_t)> before;
    std::function<ParamVec(std::size_t, std::size_t, const ParamVec&)> map;
    std::function<void(std::size_t, std::size_t)> after;
    std::mutex m;
    std::map<std::size_t, std::size_t> calls;

    double operator()(const ParamVec& x)
    {
        const std::size_t slot = tadah::hpo::warmstart::current_chain();
        std::size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(m);
            n = ++calls[slot];
        }
        if (slot < delay_ms.size() && delay_ms[slot] > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms[slot]));
        if (before) before(slot, n);
        const double loss = map ? host(map(slot, n, x)) : host(x);
        if (after) after(slot, n);
        return loss;
    }
};

enum class Driver { PT, PA, ASA };

/* What a run left: its files (time columns removed), its journal and how
 * it ended. */
struct RunFiles {
    std::map<std::string, std::string> files;
    std::string journal;
    bool stagnated = false;
    bool shut_down = false;
};

std::string read_file(const fs::path& path)
{
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/* `text` without its column `col` (0-based, whitespace-separated; the
 * header too). */
std::string drop_column(const std::string& text, std::size_t col)
{
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) tok.push_back(t);
        for (std::size_t i = 0, w = 0; i < tok.size(); ++i) {
            if (i == col) continue;
            out << (w++ ? " " : "") << tok[i];
        }
        out << "\n";
    }
    return out.str();
}

/* best_pot.tadah without the date line of its header. */
std::string drop_date_line(const std::string& text)
{
    static const std::regex date(R"(^#\s+\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s*$)");
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line))
        if (!std::regex_match(line, date)) out << line << "\n";
    return out.str();
}

struct RunOptions {
    std::vector<int> delay_ms;
    std::function<void(std::size_t, std::size_t)> before;
    std::function<ParamVec(std::size_t, std::size_t, const ParamVec&)> map;
    std::size_t stagnation_threshold = 0;
    std::function<void(std::size_t, std::size_t)> after;
};

/* One run of `target` with `driver` on the Ta model. */
RunFiles run_on_ta(const std::string& target, Driver driver,
                   const RunOptions& opt)
{
    ScratchDir sd;
    tadah::core::PeriodicTable::initialize();
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("reopen-for-test");
    log.reset_for_test();
    Host::reset_globals_for_test();

    const fs::path& common = ta_common_dir();
    const fs::path hpo = sd.dir / "config.hpo";
    const fs::path val = sd.dir / "config.val";
    std::ofstream(hpo) << target;
    std::ofstream(val) << "DBFILE "
                       << (common / "data" / "elastic.tadahdb").string() << "\n";
    tadah::core::Context ctx(tadah::core::CoreUserConfigParser().parse(
        (common / "config.tadah").string()));
    ctx.remove("DBFILE");
    ctx.add("DBFILE", (common / "data" / "elastic.tadahdb").string());
    ctx.add("DBFILE", (common / "data" / "eos_bcc.tadahdb").string());
    ctx.add("HPOTARGET", hpo.string());
    ctx.add("VALIDATION", val.string());

    log.open((sd.dir / "hpo_run.log").string(), "test", sd.dir.string(), 0);
    RunFiles out;
    {
        tadah::hpo::HPOTargets trg(ctx, hpo.string());
        Host host(ctx, trg, val.string());
        host.prepare_best_publication();
        Host::set_stagnation_threshold(opt.stagnation_threshold);
        PacedHost paced{host, opt.delay_ms, opt.before, opt.map, opt.after,
                        {}, {}};
        const auto anchored = tadah::hpo::anchor_seed_points(trg.spec);
        const tadah::hpo::OptimizerSpec& sp = anchored ? *anchored : trg.spec;
        try {
            switch (driver) {
                case Driver::PT:
                    tadah::hpo::run_anneal_pt_impl<ParamVec>(sp, paced);
                    break;
                case Driver::PA:
                    tadah::hpo::run_anneal_pa_impl<ParamVec>(sp, paced);
                    break;
                case Driver::ASA:
                    tadah::hpo::run_asa_impl<ParamVec>(sp, paced);
                    break;
            }
        } catch (const tadah::hpo::StagnationStop&) {
            out.stagnated = true;
        } catch (const tadah::core::ShutdownRequested&) {
            out.shut_down = true;
        }
        Host::set_stagnation_threshold(0);
        host.flush_remaining();
        host.finish_best_publication();
    }
    log.finish("test-end");
    log.reset_for_test();
    out.journal = read_file(sd.dir / "hpo_run.log");
    for (const char* f : {"loss.tadah", "params.tadah", "stats.tadah",
                          "best_loss.tadah", "best_params.tadah",
                          "best_stats.tadah"})
        out.files[f] = read_file(f);
    out.files["outvar.tadah"] = drop_column(read_file("outvar.tadah"), 1);
    out.files["best_outvar.tadah"] =
        drop_column(read_file("best_outvar.tadah"), 2);
    out.files["best_pot.tadah"] = drop_date_line(read_file("best_pot.tadah"));
    /* The training record names the run's files in the scratch directory,
     * which differs from run to run. */
    for (auto& kv : out.files) {
        std::string& text = kv.second;
        const std::string dir = sd.dir.string();
        for (auto p = text.find(dir); p != std::string::npos;
             p = text.find(dir, p + 9))
            text.replace(p, dir.size(), "<scratch>");
    }
    Host::reset_globals_for_test();
    return out;
}

/* The two runs wrote the same files. */
void check_same_files(const RunFiles& a, const RunFiles& b)
{
    REQUIRE(a.files.size() == b.files.size());
    for (const auto& [name, text] : a.files) {
        INFO(name << ":\n--- first run ---\n" << text
                  << "--- second run ---\n" << b.files.at(name));
        CHECK_FALSE(text.empty());
        CHECK(text == b.files.at(name));
    }
}

/* The data rows (not '#', not blank) of `text`, split on whitespace. */
std::vector<std::vector<std::string>> data_rows(const std::string& text)
{
    std::vector<std::vector<std::string>> rows;
    std::istringstream in(text);
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

/* The lines of the target files after the OPTIMIZER block: FAILSCORE
 * `failscore`, the loss term `pc_line`. */
std::string target_tail(const std::string& failscore = "1e16",
                        const std::string& pc_line = "PC_ERMSE 0 0.0001")
{
    return "LOSS L2\nFAILSCORE " + failscore + "\n" + pc_line +
           "\nOPTIM RCUT2B (1) 4.0 6.0\n"
           "OPTIM SGRIDMB (1) 0.1 0.4\nOUTPUT 1\nBOUTPUT 1\nDIGITS 15\n";
}

std::string pt_target(int threads, const std::string& extra = "",
                      int maxeval = 40, const std::string& tail = target_tail())
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
           "    ALGO ANNEAL_PT\n    THREADS " + std::to_string(threads) +
           "\n    SEED 1\n    MAXEVAL " + std::to_string(maxeval) +
           "\n    PARAM REPLICAS 4\n"
           "    PARAM T_HOT 0.10\n    PARAM T_COLD 0.001\n" + extra +
           "  ENDSTRATEGY\nENDOPTIMIZER\n" + tail;
}

std::string pa_target(int threads, const std::string& extra = "",
                      int maxeval = 40)
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
           "    ALGO ANNEAL_PA\n    THREADS " + std::to_string(threads) +
           "\n    POPULATION 8\n    SEED 1\n    MAXEVAL " +
           std::to_string(maxeval) + "\n"
           "    PARAM T_HOT 0.10\n    PARAM T_MIN 0.001\n    PARAM RT 0.85\n" +
           extra + "  ENDSTRATEGY\nENDOPTIMIZER\n" + target_tail();
}

std::string asa_target(int threads, const std::string& failscore = "1e16")
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB ASA\n"
           "    ALGO ASA\n    THREADS " + std::to_string(threads) +
           "\n    SEED 1\n    MAXEVAL 40\n    PARAM Limit_Generated 40\n"
           "    PARAM Temperature_Ratio_Scale 1e-4\n    PARAM Curvature_0 1\n"
           "    PARAM Gener_Block 4\n    PARAM Gener_Block_Max 4\n"
           "    PARAM Gener_Mov_Avr 0\n"
           "  ENDSTRATEGY\nENDOPTIMIZER\n" + target_tail(failscore);
}

/* The #ITERATION column of the data rows of `text`. */
std::vector<std::string> iterations(const std::string& text)
{
    std::vector<std::string> v;
    for (const auto& row : data_rows(text)) v.push_back(row.at(0));
    return v;
}

/* "1", "2", ..., `last` without the numbers in `missing`. */
std::vector<std::string> numbers_to(
    std::size_t last, const std::vector<std::size_t>& missing = {})
{
    std::vector<std::string> v;
    for (std::size_t n = 1; n <= last; ++n)
        if (std::find(missing.begin(), missing.end(), n) == missing.end())
            v.push_back(std::to_string(n));
    return v;
}

/* Slot k of n waits (n - 1 - k) * step ms: the batch starts and ends in
 * reverse slot order. */
std::vector<int> reverse_delays(std::size_t n, int step = 40)
{
    std::vector<int> d(n);
    for (std::size_t k = 0; k < n; ++k)
        d[k] = static_cast<int>(n - 1 - k) * step;
    return d;
}

/* Slot k waits k * step ms: the batch starts and ends in slot order. */
std::vector<int> forward_delays(std::size_t n, int step = 40)
{
    std::vector<int> d(n);
    for (std::size_t k = 0; k < n; ++k) d[k] = static_cast<int>(k) * step;
    return d;
}

std::size_t count_of(const std::string& text, const std::string& what)
{
    std::size_t n = 0;
    for (auto p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
}

}  // namespace

/* Pins: TADAH:ANNEAL_PT (SEED 1, REPLICAS 4, MAXEVAL 40, T_HOT and T_COLD
 * set, the default CLASSIC steps with the coordinate warm-up of their
 * calibration -- at MAXEVAL 40 the calibration's budget, 10 evaluations,
 * ends during the warm-up and the pilot makes none (the pilot case below
 * has one that runs) --, and PARAM CORANA ON, which runs no calibration)
 * writes the same files -- loss.tadah, params.tadah, stats.tadah,
 * outvar.tadah, the four best_* files and best_pot.tadah, whose header
 * names HPO_SOURCE_EVAL, HPO_SOURCE_SLOT and "Evaluation N, improvement K"
 * -- in two runs at THREADS 2 whose evaluations start and end in reverse
 * slot order in one and in slot order in the other, and at THREADS 1 and
 * at THREADS 4 in reverse slot order. In each, the rows of loss.tadah are
 * numbered 1, 2, ... without a gap, and the journal has no line saying
 * that an evaluation of a batch did not run.
 * Before the fix: an evaluation took its number when it started and a new
 * best was committed when it ended, so the runs numbered the evaluations
 * of an epoch in the order the delays gave them, and the ITERATION and
 * SLOT columns of params.tadah, the EVAL column of the best_* files and
 * HPO_SOURCE_EVAL differed. */
TEST_CASE("Proposal order: a seeded ANNEAL_PT run writes the same files at "
          "any THREADS and in any order of its evaluations",
          "[HPO][proposal_order][anneal_pt]")
{
    for (const std::string corana : {std::string(""),
                                     std::string("    PARAM CORANA ON\n")}) {
        INFO("CORANA: " << (corana.empty() ? "OFF" : "ON"));
        const RunFiles a = run_on_ta(pt_target(2, corana), Driver::PT,
                                     {reverse_delays(4), {}, {}, 0});
        const RunFiles b = run_on_ta(pt_target(2, corana), Driver::PT,
                                     {forward_delays(4), {}, {}, 0});
        check_same_files(a, b);
        const RunFiles c = run_on_ta(pt_target(1, corana), Driver::PT, {});
        const RunFiles d = run_on_ta(pt_target(4, corana), Driver::PT,
                                     {reverse_delays(4), {}, {}, 0});
        check_same_files(c, d);
        check_same_files(a, c);
        for (const RunFiles* r : {&a, &b, &c, &d}) {
            const auto rows = data_rows(r->files.at("loss.tadah"));
            REQUIRE(rows.size() == 40);
            for (std::size_t i = 0; i < rows.size(); ++i)
                CHECK(rows[i].at(0) == std::to_string(i + 1));
            CHECK(count_of(r->journal, "did not run") == 0);
        }
    }
}

/* Pins: TADAH:ANNEAL_PA (SEED 1, POPULATION 8, MAXEVAL 40) writes the same
 * files in two runs at THREADS 2 whose evaluations start and end in reverse
 * slot order in one and in slot order in the other, and at THREADS 1 and
 * at THREADS 4 in reverse slot order; its rows are numbered 1 to 40.
 * Before the fix: the runs numbered the evaluations of a T-step in the
 * order the delays gave them (ITERATION, SLOT, EVAL, HPO_SOURCE_EVAL
 * differed). */
TEST_CASE("Proposal order: a seeded ANNEAL_PA run writes the same files at "
          "any THREADS and in any order of its evaluations",
          "[HPO][proposal_order][anneal_pa]")
{
    const RunFiles a = run_on_ta(pa_target(2), Driver::PA,
                                 {reverse_delays(8, 20), {}, {}, 0});
    const RunFiles b = run_on_ta(pa_target(2), Driver::PA,
                                 {forward_delays(8, 20), {}, {}, 0});
    check_same_files(a, b);
    const RunFiles c = run_on_ta(pa_target(1), Driver::PA, {});
    const RunFiles d = run_on_ta(pa_target(4), Driver::PA,
                                 {reverse_delays(8, 20), {}, {}, 0});
    check_same_files(c, d);
    check_same_files(a, c);
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        const auto rows = data_rows(r->files.at("loss.tadah"));
        REQUIRE(rows.size() == 40);
        for (std::size_t i = 0; i < rows.size(); ++i)
            CHECK(rows[i].at(0) == std::to_string(i + 1));
    }
}

/* Pins: ASA:ASA with a block of 4 states (Gener_Block 4, Gener_Block_Max
 * 4, Gener_Mov_Avr 0; SEED 1, MAXEVAL 40) writes the same files in two
 * runs at THREADS 2 whose evaluations start and end in reverse block order
 * in one and in block order in the other, and at THREADS 1 and at THREADS
 * 4 in reverse block order; the rows of loss.tadah are numbered from 1
 * without a gap.
 * Before the fix: the runs numbered the states of a block in the order the
 * delays gave them (ITERATION, SLOT, EVAL, HPO_SOURCE_EVAL differed). */
TEST_CASE("Proposal order: a seeded ASA run with a block of states writes "
          "the same files at any THREADS and in any order of its "
          "evaluations", "[HPO][proposal_order][asa]")
{
    const RunFiles a = run_on_ta(asa_target(2), Driver::ASA,
                                 {reverse_delays(4), {}, {}, 0});
    const RunFiles b = run_on_ta(asa_target(2), Driver::ASA,
                                 {forward_delays(4), {}, {}, 0});
    check_same_files(a, b);
    const RunFiles c = run_on_ta(asa_target(1), Driver::ASA, {});
    const RunFiles d = run_on_ta(asa_target(4), Driver::ASA,
                                 {reverse_delays(4), {}, {}, 0});
    check_same_files(c, d);
    check_same_files(a, c);
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        const auto rows = data_rows(r->files.at("loss.tadah"));
        REQUIRE_FALSE(rows.empty());
        for (std::size_t i = 0; i < rows.size(); ++i)
            CHECK(rows[i].at(0) == std::to_string(i + 1));
    }
}

/* Pins: the equal-loss tie-break and the stagnation stop follow the
 * evaluation numbers. ANNEAL_PT (REPLICAS 4, STEP_MODE FIXED) through a
 * host that evaluates point B (RCUT2B 5.362132, SGRIDMB 0.1233477) for the
 * slots 2 and 4 (1-based) and point A (RCUT2B 5.6, SGRIDMB 0.2, the
 * training configuration's values, a higher loss) for the others, with a
 * RESTART window of 6 evaluations (HPO_Host::set_stagnation_threshold):
 * evaluation 1 (the seed, slot 1) is A, the population's first evaluation
 * of B, evaluation 2 of slot 2, is the only new best after it -- the B of
 * slot 4, evaluation 4, has the same loss and comes after it -- so
 * best_params.tadah has 2 rows (EVAL 1 and 2); the window closes at
 * evaluation 8, the last of epoch 1, the journal says "StagnationStop
 * fired at eval 8", and the run makes 8 evaluations: the same at THREADS
 * 1 and at THREADS 4 with the evaluations of each batch in reverse slot
 * order.
 * Before the fix: at THREADS 4 the B of slot 4 ended first and became the
 * new best, then the B of slot 2, with the same loss, won the tie on its
 * slot: best_params.tadah had 3 rows, the window counted from the later
 * one, and the stop came at another evaluation. */
TEST_CASE("Proposal order: the equal-loss tie-break and the stagnation stop "
          "give the same result at THREADS 1 and 4",
          "[HPO][proposal_order][anneal_pt][stagnation]")
{
    const auto point = [](double rcut, double sgrid) {
        ParamVec p(2);
        p(0) = rcut;
        p(1) = sgrid;
        return p;
    };
    const ParamVec A = point(5.6, 0.2), B = point(5.362132, 0.1233477);
    RunOptions opt;
    opt.map = [A, B](std::size_t slot, std::size_t, const ParamVec&) {
        return slot == 1 || slot == 3 ? B : A;
    };
    opt.stagnation_threshold = 6;
    const std::string target = pt_target(1, "    PARAM STEP_MODE FIXED\n");
    const RunFiles one = run_on_ta(target, Driver::PT, opt);
    RunOptions opt4 = opt;
    opt4.delay_ms = reverse_delays(4);
    const RunFiles four = run_on_ta(
        pt_target(4, "    PARAM STEP_MODE FIXED\n"), Driver::PT, opt4);
    for (const RunFiles* r : {&one, &four}) {
        INFO("journal:\n" << r->journal);
        CHECK(r->stagnated);
        const auto best = data_rows(r->files.at("best_params.tadah"));
        REQUIRE(best.size() == 2);
        CHECK(best[0].at(1) == "1");
        CHECK(best[1].at(1) == "2");
        CHECK(best[1].at(2) == "2");
        CHECK(count_of(r->journal, "StagnationStop fired at eval 8;") == 1);
        CHECK(data_rows(r->files.at("loss.tadah")).size() == 8);
    }
    check_same_files(one, four);
}

/* Pins: a stop during a batch keeps the evaluations that finished, at
 * their numbers. ANNEAL_PT (REPLICAS 4, THREADS 2, STEP_MODE FIXED: one
 * evaluation per slot and epoch) whose third evaluation of slot 3 (1-based)
 * -- in epoch 2 -- throws core::ShutdownRequested before it reaches the
 * host, as an evaluation that starts after a signal does, once the
 * evaluation of slot 4 in epoch 2 has ended (slot 1 waits 150 ms). The run
 * ends with the stop; loss.tadah and params.tadah have the rows of
 * evaluations 1 to 10 and 12 -- the seed 1, the population 2 to 4, epoch 1
 * 5 to 8, and of epoch 2 (9 to 12) slots 1, 2 and 4 -- with SLOT 1, 2 and
 * 4 for 9, 10 and 12: evaluation 11, slot 3's, did not run and has no
 * row; the journal says "epoch 2 of TADAH:ANNEAL_PT: 1 of its 4
 * evaluations did not run; evaluation 11 has no row", without a cause,
 * which the batch does not know (the line of the run's stop gives it).
 * Before the fix: the evaluations of epoch 2 took their numbers in the
 * order they started (9 slot 2, 10 slot 4, 11 slot 1), and the journal
 * had no such line; then the line said "did not run (the search stopped
 * during it)", also when the slot had ended its own sweep at STOPVAL. */
TEST_CASE("Proposal order: a stop during a batch keeps the evaluations that "
          "finished at their numbers", "[HPO][proposal_order][anneal_pt]")
{
    RunOptions opt;
    opt.delay_ms = {150, 0, 0, 0};
    /* Slot 4's third evaluation (epoch 2) has ended: slots 2 and 4 run on
     * the other worker, so slot 3 waiting for it holds up nothing it
     * needs. */
    const auto slot4_done = std::make_shared<std::atomic<bool>>(false);
    opt.after = [slot4_done](std::size_t slot, std::size_t n) {
        if (slot == 3 && n == 3) slot4_done->store(true);
    };
    opt.before = [slot4_done](std::size_t slot, std::size_t n) {
        if (slot != 2 || n != 3) return;
        for (int i = 0; i < 60000 && !slot4_done->load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        throw tadah::core::ShutdownRequested{};
    };
    const RunFiles r = run_on_ta(pt_target(2, "    PARAM STEP_MODE FIXED\n"),
                                 Driver::PT, opt);
    INFO("journal:\n" << r.journal);
    CHECK(r.shut_down);
    const auto rows = data_rows(r.files.at("params.tadah"));
    std::vector<std::string> iters, slots;
    for (const auto& row : rows) {
        iters.push_back(row.at(0));
        slots.push_back(row.at(1));
    }
    CHECK(iters == std::vector<std::string>{"1", "2", "3", "4", "5", "6", "7",
                                            "8", "9", "10", "12"});
    REQUIRE(slots.size() == 11);
    CHECK(std::vector<std::string>(slots.end() - 3, slots.end()) ==
          std::vector<std::string>{"1", "2", "4"});
    CHECK(data_rows(r.files.at("loss.tadah")).size() == 11);
    CHECK(count_of(r.journal,
                   "epoch 2 of TADAH:ANNEAL_PT: 1 of its 4 evaluations did "
                   "not run; evaluation 11 has no row\n") == 1);
    CHECK(count_of(r.journal, "stopped during it") == 0);
}

/* Pins: a STOPVAL reached in a round of the step calibration of
 * TADAH:ANNEAL_PT ends the search after that round. ANNEAL_PT (SEED 1,
 * REPLICAS 4, MAXEVAL 1200, T_HOT and T_COLD set, the default CLASSIC
 * steps, STOPVAL 1.7e-3): evaluation 13, slot 1's in the third round of
 * the coordinate warm-up (evaluations 13 to 16, slots 1 to 4), is the
 * first whose loss is below STOPVAL. The other slots make their
 * evaluations of that round and the search ends: loss.tadah has the rows
 * 1 to 16, the journal says "ANNEAL_PT stopped: STOPVAL reached after 16
 * host eval(s) (0 epoch(s)" and no evaluation "did not run" -- the same
 * files at THREADS 1, at THREADS 2 with the evaluations of each round in
 * slot order and at THREADS 4 in reverse slot order.
 * Before the fix: the replica that reached STOPVAL ended its own
 * calibration, and the other replicas went on through their whole
 * coordinate warm-up before the search ended: 79 rows ("STOPVAL reached
 * after 79 host eval(s)"). */
TEST_CASE("Proposal order: a STOPVAL reached in a round of the ANNEAL_PT "
          "step calibration ends the search after that round",
          "[HPO][proposal_order][anneal_pt][stopval]")
{
    const std::string stop = "    STOPVAL 1.7e-3\n";
    const RunFiles one = run_on_ta(pt_target(1, stop, 1200), Driver::PT, {});
    const RunFiles two = run_on_ta(pt_target(2, stop, 1200), Driver::PT,
                                   {forward_delays(4, 20), {}, {}, 0});
    const RunFiles four = run_on_ta(pt_target(4, stop, 1200), Driver::PT,
                                    {reverse_delays(4, 20), {}, {}, 0});
    for (const RunFiles* r : {&one, &two, &four}) {
        INFO("journal:\n" << r->journal);
        const auto rows = data_rows(r->files.at("loss.tadah"));
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(16));
        REQUIRE(rows.size() >= 16);
        std::size_t first_below = 0;
        for (std::size_t i = 0; i < rows.size() && first_below == 0; ++i)
            if (std::stod(rows[i].back()) < 1.7e-3) first_below = i + 1;
        CHECK(first_below == 13);
        for (std::size_t i = 12; i < 16; ++i)
            CHECK(rows[i].at(1) == std::to_string(i - 11));
        CHECK(count_of(r->journal, "ANNEAL_PT stopped: STOPVAL reached after "
                                   "16 host eval(s) (0 epoch(s)") == 1);
        CHECK(count_of(r->journal, "did not run") == 0);
    }
    check_same_files(one, two);
    check_same_files(one, four);
}

/* Pins: a replica that reaches STOPVAL during an epoch of TADAH:ANNEAL_PT
 * ends its own sweep, and the search ends after the epoch. ANNEAL_PT
 * (SEED 1, REPLICAS 4, MAXEVAL 400, PARAM CORANA ON, STOPVAL 1.2e-3):
 * epoch 22 has the evaluations 173 to 180, two for each slot; the first of
 * slot 1, 173, has a loss below STOPVAL, its second, 174, does not run,
 * and the other slots make theirs. loss.tadah and params.tadah have the
 * rows 1 to 180 without 174, and the journal says "epoch 22 of
 * TADAH:ANNEAL_PT: 1 of its 8 evaluations did not run; evaluation 174 has
 * no row" (no "the search stopped during it": the search ends after the
 * epoch, and the evaluation did not run because its slot ended its own
 * sweep) and "ANNEAL_PT stopped: STOPVAL
 * reached after 179 host eval(s) (22 epoch(s)" -- the same files at THREADS
 * 1, at THREADS 2 in reverse and in forward slot order and at THREADS 4 in
 * reverse slot order.
 * Before the fix: a replica that reached STOPVAL stopped the others at
 * their next proposal, and the evaluations of an epoch took their numbers
 * in the order they started: the runs stopped after 173 (THREADS 1), 175
 * and 174 (THREADS 2 in reverse and in forward order) and 179 (THREADS 4)
 * evaluations, without the journal line, and their files differed. */
TEST_CASE("Proposal order: a STOPVAL reached during an ANNEAL_PT epoch ends "
          "the search after the epoch at any THREADS",
          "[HPO][proposal_order][anneal_pt][stopval]")
{
    const std::string extra = "    PARAM CORANA ON\n    STOPVAL 1.2e-3\n";
    const RunFiles a = run_on_ta(pt_target(1, extra, 400), Driver::PT, {});
    const RunFiles b = run_on_ta(pt_target(2, extra, 400), Driver::PT,
                                 {reverse_delays(4, 10), {}, {}, 0});
    const RunFiles c = run_on_ta(pt_target(2, extra, 400), Driver::PT,
                                 {forward_delays(4, 10), {}, {}, 0});
    const RunFiles d = run_on_ta(pt_target(4, extra, 400), Driver::PT,
                                 {reverse_delays(4, 10), {}, {}, 0});
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        INFO("journal:\n" << r->journal);
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(180, {174}));
        CHECK(iterations(r->files.at("params.tadah")) ==
              numbers_to(180, {174}));
        CHECK(count_of(r->journal,
                       "epoch 22 of TADAH:ANNEAL_PT: 1 of its 8 evaluations "
                       "did not run; evaluation 174 has no row\n") == 1);
        CHECK(count_of(r->journal, "stopped during it") == 0);
        CHECK(count_of(r->journal, "ANNEAL_PT stopped: STOPVAL reached after "
                                   "179 host eval(s) (22 epoch(s)") == 1);
    }
    check_same_files(a, b);
    check_same_files(a, c);
    check_same_files(a, d);
}

/* Pins: a walker that reaches STOPVAL during a temperature step of
 * TADAH:ANNEAL_PA ends its own sweeps, and the search ends after the step.
 * ANNEAL_PA (SEED 1, POPULATION 8, MAXEVAL 300, PARAM CHAIN_LENGTH 5: a
 * step of 40 evaluations, five for each walker; STOPVAL 1.1e-3):
 * temperature step 2 has the evaluations 89 to 128; the first of walker 1,
 * 89, has a loss below STOPVAL, its other four, 90 to 93, do not run, and
 * the other walkers make theirs. loss.tadah and params.tadah have the
 * rows 1 to 128 without 90 to 93, and the journal says "temperature step 2
 * of TADAH:ANNEAL_PA: 4 of its 40 evaluations did not run; evaluations
 * 90-93 have no rows" (without "the search stopped during it", as the
 * search ends after the step) and "ANNEAL_PA
 * stopped: STOPVAL reached after 124 host eval(s)" -- the same files at
 * THREADS 1, at THREADS 2 in reverse and in forward walker order and at
 * THREADS 4 in reverse walker order.
 * Before the fix: a walker that reached STOPVAL stopped the others at
 * their next proposal, and the evaluations of a step took their numbers in
 * the order they started: the runs stopped after 89 (THREADS 1), 91 and 90
 * (THREADS 2 in reverse and in forward order) and 95 (THREADS 4)
 * evaluations, without the journal line, and their files differed. */
TEST_CASE("Proposal order: a STOPVAL reached during an ANNEAL_PA temperature "
          "step ends the search after the step at any THREADS",
          "[HPO][proposal_order][anneal_pa][stopval]")
{
    const std::string extra = "    PARAM CHAIN_LENGTH 5\n    STOPVAL 1.1e-3\n";
    const RunFiles a = run_on_ta(pa_target(1, extra, 300), Driver::PA, {});
    const RunFiles b = run_on_ta(pa_target(2, extra, 300), Driver::PA,
                                 {reverse_delays(8, 10), {}, {}, 0});
    const RunFiles c = run_on_ta(pa_target(2, extra, 300), Driver::PA,
                                 {forward_delays(8, 10), {}, {}, 0});
    const RunFiles d = run_on_ta(pa_target(4, extra, 300), Driver::PA,
                                 {reverse_delays(8, 10), {}, {}, 0});
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        INFO("journal:\n" << r->journal);
        const std::vector<std::size_t> gap{90, 91, 92, 93};
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(128, gap));
        CHECK(iterations(r->files.at("params.tadah")) == numbers_to(128, gap));
        CHECK(count_of(r->journal,
                       "temperature step 2 of TADAH:ANNEAL_PA: 4 of its 40 "
                       "evaluations did not run; evaluations 90-93 have no "
                       "rows\n") == 1);
        CHECK(count_of(r->journal, "stopped during it") == 0);
        CHECK(count_of(r->journal, "ANNEAL_PA stopped: STOPVAL reached after "
                                   "124 host eval(s)") == 1);
    }
    check_same_files(a, b);
    check_same_files(a, c);
    check_same_files(a, d);
}

/* Pins: the lowest values of a --skip-rel filter that an evaluation reads
 * are those of the evaluations recorded before its batch. ANNEAL_PT (SEED
 * 1, REPLICAS 4, MAXEVAL 40, T_HOT and T_COLD set) with PC_ERMSE 0 0.0001
 * --skip-rel 1.3: an evaluation whose ERMSE exceeds 1.3 times the lowest
 * ERMSE so far scores FAILSCORE (1e16), which 18 of the evaluations do.
 * The run writes the same files at THREADS 1, at
 * THREADS 2 and at THREADS 4 with the evaluations of each batch in reverse
 * slot order, where a later evaluation of a batch ends before an earlier
 * one.
 * Before the fix: the lowest values were updated as the evaluations
 * ended, so an evaluation of a batch read those of a later one that had
 * ended first, and the evaluations took their numbers in the order they
 * started: 23, 22 and 17 evaluations scored FAILSCORE at THREADS 1, 2 and
 * 4, and the files differed. With only the deferral of the lowest values
 * undone, 22 and 19 did at THREADS 1 and 2. */
TEST_CASE("Proposal order: a --skip-rel filter reads the lowest values "
          "recorded before the batch at any THREADS",
          "[HPO][proposal_order][anneal_pt][skip_rel]")
{
    const std::string tail =
        target_tail("1e16", "PC_ERMSE 0 0.0001 --skip-rel 1.3");
    const RunFiles a = run_on_ta(pt_target(1, "", 40, tail), Driver::PT, {});
    const RunFiles b = run_on_ta(pt_target(2, "", 40, tail), Driver::PT,
                                 {reverse_delays(4), {}, {}, 0});
    const RunFiles d = run_on_ta(pt_target(4, "", 40, tail), Driver::PT,
                                 {reverse_delays(4), {}, {}, 0});
    for (const RunFiles* r : {&a, &b, &d}) {
        INFO("journal:\n" << r->journal);
        std::size_t trips = 0;
        for (const auto& row : data_rows(r->files.at("loss.tadah")))
            if (std::stod(row.back()) == 1e16) ++trips;
        CHECK(trips == 18);
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(40));
    }
    check_same_files(a, b);
    check_same_files(a, d);
}

/* Pins: ASA:ASA with a block of 4 states (Gener_Block 4, Gener_Block_Max
 * 4, Gener_Mov_Avr 0; SEED 1, MAXEVAL 40) and FAILSCORE 6e-3, so that
 * states of the blocks have a loss at or above FAILSCORE and are invalid
 * for ASA (9 of them in the slots 2 to 4 of the blocks), writes the same
 * files at THREADS 1, at THREADS 2 in reverse and in forward block order
 * and at THREADS 4 in reverse block order; the rows of loss.tadah are
 * numbered 1 to 40.
 * Before the fix: the states of a block took their numbers in the order
 * they started, so loss.tadah, params.tadah, stats.tadah and outvar.tadah
 * of the runs at THREADS 2 and 4 in reverse block order differed from
 * those at THREADS 1. The cost that ASA reads for an invalid state, the
 * last valid cost before its block, does not reach the files (ASA sorts
 * the block by cost and skips an invalid state), so this case passes with
 * that part of the fix undone. */
TEST_CASE("Proposal order: an ASA run whose blocks have invalid states writes "
          "the same files at any THREADS",
          "[HPO][proposal_order][asa]")
{
    const RunFiles a = run_on_ta(asa_target(1, "6e-3"), Driver::ASA, {});
    const RunFiles b = run_on_ta(asa_target(2, "6e-3"), Driver::ASA,
                                 {reverse_delays(4), {}, {}, 0});
    const RunFiles c = run_on_ta(asa_target(2, "6e-3"), Driver::ASA,
                                 {forward_delays(4), {}, {}, 0});
    const RunFiles d = run_on_ta(asa_target(4, "6e-3"), Driver::ASA,
                                 {reverse_delays(4), {}, {}, 0});
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        INFO("journal:\n" << r->journal);
        std::size_t invalid_in_blocks = 0;
        for (const auto& row : data_rows(r->files.at("loss.tadah")))
            if (row.at(1) != "1" && std::stod(row.back()) >= 6e-3)
                ++invalid_in_blocks;
        CHECK(invalid_in_blocks == 9);
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(40));
    }
    check_same_files(a, b);
    check_same_files(a, c);
    check_same_files(a, d);
}

/* Pins: the rounds of the whole-vector pilot of ANNEAL_PT's step
 * calibration are numbered and recorded in slot order. ANNEAL_PT (SEED 1,
 * REPLICAS 4, MAXEVAL 320, T_HOT and T_COLD set, the default CLASSIC steps
 * with PARAM STEP_WARMUP_SWEEPS 2): the calibration's budget, 80
 * evaluations, leaves 48 for the pilot, which the journal
 * says ("and the pilot made 48 evaluations"). The run writes the
 * same files at THREADS 1, at THREADS 2 in reverse and in forward slot
 * order and at THREADS 4 in reverse slot order, with the rows 1 to 320.
 * The journal calls the phase a whole-vector pilot, as the documentation
 * does ("... 3 coordinate(s) missed for no known reason and 0 whole-vector
 * pilot(s) did not converge; ..."), never "full-vector".
 * Before the fix: each slot ran its whole pilot in one task, whose
 * evaluations took their numbers as they started, so the runs at THREADS
 * 2 and 4 numbered the pilot's evaluations, and the epochs after it, in
 * the order of the delays: their params.tadah, loss.tadah, the best_*
 * files and best_pot.tadah differed from those at THREADS 1. Then the
 * journal said "0 full-vector pilot(s) did not converge". */
TEST_CASE("Proposal order: the rounds of the ANNEAL_PT whole-vector pilot are "
          "numbered and recorded in slot order",
          "[HPO][proposal_order][anneal_pt]")
{
    const std::string extra = "    PARAM STEP_WARMUP_SWEEPS 2\n";
    const RunFiles a = run_on_ta(pt_target(1, extra, 320), Driver::PT, {});
    const RunFiles b = run_on_ta(pt_target(2, extra, 320), Driver::PT,
                                 {reverse_delays(4, 10), {}, {}, 0});
    const RunFiles c = run_on_ta(pt_target(2, extra, 320), Driver::PT,
                                 {forward_delays(4, 10), {}, {}, 0});
    const RunFiles d = run_on_ta(pt_target(4, extra, 320), Driver::PT,
                                 {reverse_delays(4, 10), {}, {}, 0});
    for (const RunFiles* r : {&a, &b, &c, &d}) {
        INFO("journal:\n" << r->journal);
        CHECK(count_of(r->journal, "and the pilot made 48 "
                                   "evaluations") == 1);
        CHECK(count_of(r->journal,
                       "3 coordinate(s) missed for no known reason and 0 "
                       "whole-vector pilot(s) did not converge; ") == 1);
        CHECK(count_of(r->journal, "full-vector") == 0);
        CHECK(iterations(r->files.at("loss.tadah")) == numbers_to(320));
    }
    check_same_files(a, b);
    check_same_files(a, c);
    check_same_files(a, d);
}

/* Pins: the journal line of a round of the whole-vector pilot that a stop
 * cuts short names the round by the documentation's name of the phase.
 * The run of the case above at THREADS 1 whose evaluation of slot 3
 * (1-based) in the first round of the pilot -- its tenth: its starting
 * point, the 8 evaluations of its coordinate warm-up, then the pilot --
 * throws core::ShutdownRequested before it reaches the host, as an
 * evaluation that starts after a signal does. The run ends with the stop,
 * after evaluation 38, and the journal says "round 9 of the whole-vector
 * pilot of TADAH:ANNEAL_PT: 2 of its 4 evaluations did not run": slot 4's
 * evaluation, after the stop, does not run either, and their numbers, 39
 * and 40, the last of the round, go back to the run.
 * Before the fix: "round 9 of the full-vector pilot of TADAH:ANNEAL_PT: 2
 * of its 4 evaluations did not run (the search stopped during it)". */
TEST_CASE("Proposal order: a stop during a round of the ANNEAL_PT "
          "whole-vector pilot names the round in the journal",
          "[HPO][proposal_order][anneal_pt]")
{
    RunOptions opt;
    opt.before = [](std::size_t slot, std::size_t n) {
        if (slot == 2 && n == 10) throw tadah::core::ShutdownRequested{};
    };
    const RunFiles r = run_on_ta(
        pt_target(1, "    PARAM STEP_WARMUP_SWEEPS 2\n", 320), Driver::PT,
        opt);
    INFO("journal:\n" << r.journal);
    CHECK(r.shut_down);
    CHECK(count_of(r.journal,
                   "round 9 of the whole-vector pilot of TADAH:ANNEAL_PT: 2 "
                   "of its 4 evaluations did not run\n") == 1);
    CHECK(count_of(r.journal, "full-vector") == 0);
    CHECK(iterations(r.files.at("loss.tadah")) == numbers_to(38));
}

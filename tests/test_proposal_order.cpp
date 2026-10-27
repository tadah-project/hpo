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

std::string pt_target(int threads, const std::string& extra = "")
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
           "    ALGO ANNEAL_PT\n    THREADS " + std::to_string(threads) +
           "\n    SEED 1\n    MAXEVAL 40\n    PARAM REPLICAS 4\n"
           "    PARAM T_HOT 0.10\n    PARAM T_COLD 0.001\n" + extra +
           "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\n"
           "PC_ERMSE 0 0.0001\nOPTIM RCUT2B (1) 4.0 6.0\n"
           "OPTIM SGRIDMB (1) 0.1 0.4\nOUTPUT 1\nBOUTPUT 1\nDIGITS 15\n";
}

std::string pa_target(int threads)
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
           "    ALGO ANNEAL_PA\n    THREADS " + std::to_string(threads) +
           "\n    POPULATION 8\n    SEED 1\n    MAXEVAL 40\n"
           "    PARAM T_HOT 0.10\n    PARAM T_MIN 0.001\n    PARAM RT 0.85\n"
           "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\n"
           "PC_ERMSE 0 0.0001\nOPTIM RCUT2B (1) 4.0 6.0\n"
           "OPTIM SGRIDMB (1) 0.1 0.4\nOUTPUT 1\nBOUTPUT 1\nDIGITS 15\n";
}

std::string asa_target(int threads)
{
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB ASA\n"
           "    ALGO ASA\n    THREADS " + std::to_string(threads) +
           "\n    SEED 1\n    MAXEVAL 40\n    PARAM Limit_Generated 40\n"
           "    PARAM Temperature_Ratio_Scale 1e-4\n    PARAM Curvature_0 1\n"
           "    PARAM Gener_Block 4\n    PARAM Gener_Block_Max 4\n"
           "    PARAM Gener_Mov_Avr 0\n"
           "  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\n"
           "PC_ERMSE 0 0.0001\nOPTIM RCUT2B (1) 4.0 6.0\n"
           "OPTIM SGRIDMB (1) 0.1 0.4\nOUTPUT 1\nBOUTPUT 1\nDIGITS 15\n";
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
 * set, the default CLASSIC steps with their calibration, and PARAM CORANA
 * ON) writes the same files -- loss.tadah, params.tadah, stats.tadah,
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
 * evaluations did not run (the search stopped during it); evaluation 11
 * has no row".
 * Before the fix: the evaluations of epoch 2 took their numbers in the
 * order they started (9 slot 2, 10 slot 4, 11 slot 1), and the journal
 * had no such line. */
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
                   "not run (the search stopped during it); evaluation 11 "
                   "has no row") == 1);
}

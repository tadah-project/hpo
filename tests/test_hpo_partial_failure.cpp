// test_hpo_partial_failure.cpp
//
// An evaluation with a failed loss term is a failed evaluation. HPO_Host's
// loss is the mean of its terms; a PC_OBSERVABLE term that cannot be
// computed (a relaxation whose lattice parameter leaves its --hotvar
// window, a value or loss that is not finite, an evaluation that throws)
// is FAILSCORE, and so the loss of an evaluation with one failed term out
// of n was about FAILSCORE / n: below 0.999 * FAILSCORE, a usable loss to
// every part of HPO that classifies losses (the census, the best
// potential, STOPVAL, the gradient provider). Under the default FAILSCORE,
// the largest double, the usable terms vanish in it: the loss is
// FAILSCORE / 2 all over the region where the term fails, every
// finite-difference gradient there is 0, and a gradient search that starts
// there converged on it. Such an evaluation now scores FAILSCORE under
// FAILSCORE_GRADING OFF, and under ON a graded band-B failure measured by
// the failed fraction of the PC_OBSERVABLE terms (pc_fail_grading.h).
// (PC_LAMMPS and PC_OBSERVABLE are mutually exclusive -- HPOTargets refuses
// a configuration with both -- and an evaluation with a failed PC_LAMMPS
// simulation already scored FAILSCORE, test_hpo_pc_readback_e2e.cpp.)
//
// The fixture: the minimal 4-atom bcc Ta structure (train and validation),
// a fixed potential (D2_BP, 5 Gaussians, Cut_Cos, every weight pinned:
// FIXINDEX 1-5), PC_ERMSE 0 1, and a cohesive_energy observable whose
// lattice-parameter window is [3.0, 3.35]. The relaxed lattice parameter
// of this potential grows with RCUT2B and leaves the window above about
// RCUT2B = 4.94, so in the OPTIM box [4.0, 5.5] the observable fails from
// there up (its value NaN, its term FAILSCORE) while L(ERMSE) is usable
// everywhere (7.7e4 at 4.0, 5.7e5 at 5.5). The same observable with the
// window [2.0, 5.0] never fails.
//
//   - "[HPO][partial_failure][host]": HPO_Host evaluations, in-process;
//   - "[HPO][partial_failure][run]": hpo_run(), a gradient search that
//     starts where the observable fails (under FAILSCORE_GRADING OFF and
//     ON), a TADAH:RANDOM run inside that region, and STOPVAL.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/observables/observable.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/runners.h>

#include "finish_within.h"

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using tadah::core::Context;
using tadah::hpo::FailBand;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using tadah::hpo::graded_failscore;

constexpr double kMax = std::numeric_limits<double>::max();

std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives (HPO_Host
 * and hpo_run() write their files to the cwd). */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() / unique_name("tadah_partial_");
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

/* hpo_run.log of a host-level case in a temporary file, read back after
 * finish() (which writes the DIAGNOSTICS census). */
struct Journal {
    fs::path path;
    Journal() : path(fs::temp_directory_path() /
                     (unique_name("tadah_partial_log_") + ".log")) {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
    }
    ~Journal() {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish() {
        HPOLog::instance().finish("TEST");
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p);
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

std::vector<std::string> tokens(const std::string& line) {
    std::istringstream is(line);
    std::vector<std::string> out;
    for (std::string t; is >> t;) out.push_back(t);
    return out;
}

/* A row file: the header's tokens and each row's. */
struct Table {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};
Table read_table(const fs::path& p) {
    Table t;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (t.header.empty() && line[0] == '#') { t.header = tokens(line); continue; }
        if (line[0] == '#') continue;
        t.rows.push_back(tokens(line));
    }
    return t;
}

/* The column of @p t headed @p name (header.size() when none is). */
std::size_t column(const Table& t, const std::string& name) {
    std::size_t c = 0;
    while (c < t.header.size() && t.header[c] != name) ++c;
    return c;
}

/* The cell of row @p row under @p name ("" when there is none). */
std::string cell(const Table& t, std::size_t row, const std::string& name) {
    const std::size_t c = column(t, name);
    if (row >= t.rows.size() || c >= t.rows[row].size()) return {};
    return t.rows[row][c];
}

/* The count of a DIAGNOSTICS line ("Unusable losses : 3  (50.0%)"), -1
 * without the line. */
long census(const std::string& journal, const std::string& what) {
    std::smatch m;
    if (std::regex_search(journal, m, std::regex(what + R"(\s*: (\d+))")))
        return std::stol(m[1]);
    return -1;
}

/* The "Optimiser status" value of the journal's RESULT section. */
std::string journal_status(const std::string& journal) {
    std::smatch m;
    if (std::regex_search(journal, m,
                          std::regex(R"(Optimiser status\s*: ([^\n]*))")))
        return m[1];
    return "(no RESULT section)";
}

std::size_t count_of(const std::string& text, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t p = text.find(needle); p != std::string::npos;
         p = text.find(needle, p + needle.size()))
        ++n;
    return n;
}

/* The minimal 4-atom bcc Ta structure of test_hpo_host_e2e.cpp. */
void write_ta_db(const fs::path& path, double offset_x, double energy) {
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

/* The fixed potential (test_nn_exact_e2e.cpp's): D2_BP with 5 Gaussians
 * and DM_Dummy, a linear model with every weight pinned, so it is not
 * trained; RCUT2B @p rcut2b, the start of the searches. */
Context pinned_context(const fs::path& train, const fs::path& hpo,
                       const fs::path& val, double rcut2b) {
    Context c;
    c.add("DBFILE",   train.string());
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   5);
    c.add("TYPE2B",   5);
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "2.5 3.2 3.9 4.6 5.3");
    c.add("SGRID2B",  "1.0 1.0 1.0 1.0 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   rcut2b);
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   4.0);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   0);
    c.add("OALGO",    1);
    c.add("FIXINDEX", "1-5");
    for (double w : {0.8, -0.35, 0.2, -0.12, 0.07}) c.add("FIXWEIGHT", w);
    c.add("HPOTARGET",  hpo.string());
    c.add("VALIDATION", val.string());
    return c;
}

/* Writes the train/validation data and config.val into @p dir. */
void write_data(const fs::path& dir) {
    write_ta_db(dir / "train.db", 0.0, -0.001);
    write_ta_db(dir / "val.db", 0.07, -0.04);
    std::ofstream(dir / "config.val") << "DBFILE " << (dir / "val.db").string()
                                      << "\n";
}

/* The cohesive energy whose relaxed lattice parameter leaves its window
 * [3.0, 3.35] above RCUT2B of about 4.94 (it fails there), and the same
 * observable with a window it never leaves. */
const std::string kFailingObservable =
    "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
    "--supercell 2 2 2 --hotvar a0 3.0 3.35 3.3\n";
const std::string kUsableObservable =
    "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
    "--supercell 2 2 2 --hotvar a0 2.0 5.0 3.3\n";
const std::string kOptimRcut = "OPTIM RCUT2B (1) 4.0 5.5\n";

/* One HPO_Host in the cwd with the hpotarget @p target (the objective
 * lines; QFLUSH 1 and DIGITS 15 are added); evaluates each candidate
 * RCUT2B in @p xs, flushes the rows and returns the losses. @p adjust, if
 * given, changes the parsed targets before the host is built (it replaces
 * an observable with one no PC_OBSERVABLE line gives). */
std::vector<double> evaluate(
    const fs::path& dir, const std::string& target,
    const std::vector<double>& xs,
    const std::function<void(HPOTargets&)>& adjust = {}) {
    write_data(dir);
    const fs::path hpo = dir / "config.hpo";
    std::ofstream(hpo) << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
                          "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\n"
                          "LOSS L2\nQFLUSH 1\nDIGITS 15\n"
                       << target << kOptimRcut;
    Context ctx = pinned_context(dir / "train.db", hpo, dir / "config.val",
                                 4.5);
    HPOTargets trg(ctx, hpo.string());
    if (adjust) adjust(trg);
    std::vector<double> losses;
    Host host(ctx, trg, (dir / "config.val").string());
    host.prepare_best_publication();
    for (const double x : xs) {
        ParamVec p(1);
        p(0) = x;
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = host(p));
        losses.push_back(loss);
    }
    host.flush_remaining();
    return losses;
}

/* A FAILSCORE as loss.tadah prints it under DIGITS 15. */
std::string row_text(double fs) {
    return fs == kMax ? "1.797693134862316e+308" : "1.000000000000000e+16";
}

}  // namespace

/* Pins: under FAILSCORE_GRADING OFF an evaluation whose PC_OBSERVABLE term
 * fails scores FAILSCORE, under the default (the largest double) and
 * FAILSCORE 1e16: at RCUT2B = 5.2 the loss is FAILSCORE exactly, its
 * loss.tadah row has the FAILSCORE term L(Ecoh)[1] and TOTAL_LOSS/N
 * FAILSCORE, Ecoh[1] is NaN in outvar.tadah, and the census counts it
 * ("Unusable losses : 1"); at RCUT2B = 4.5, where the observable is
 * computed, the loss is the mean of its two terms L(ERMSE)[avg] and
 * L(Ecoh)[1] and usable.
 * Before the fix: the loss at 5.2 was the mean of L(ERMSE) and FAILSCORE,
 * 8.988465674311579e+307 (the usable term vanishes in it) and
 * 5.0000000004e+15, below 0.999 * FAILSCORE, so it counted as usable
 * ("Unusable losses : 0", "[ok] Every evaluation produced a usable
 * loss."). */
TEST_CASE("a failed PC_OBSERVABLE term fails its evaluation under "
          "FAILSCORE_GRADING OFF", "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    for (const double fs : {kMax, 1e16}) {
        INFO("FAILSCORE " << fs);
        Host::reset_globals_for_test();
        ScratchDir sd;
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            (fs == kMax ? std::string() : std::string("FAILSCORE 1e16\n")) +
                "PC_ERMSE 0 1\n" + kFailingObservable,
            {4.5, 5.2});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 2);
        CHECK(losses[1] == fs);

        const Table loss = read_table("loss.tadah");
        REQUIRE(loss.rows.size() == 2);
        CHECK(cell(loss, 1, "L(Ecoh)[1]") == row_text(fs));
        CHECK(cell(loss, 1, "TOTAL_LOSS/N") == row_text(fs));
        const double l_ermse = std::stod(cell(loss, 0, "L(ERMSE)[avg]"));
        const double l_ecoh = std::stod(cell(loss, 0, "L(Ecoh)[1]"));
        CHECK(losses[0] == Approx(0.5 * (l_ermse + l_ecoh)).epsilon(1e-13));
        CHECK(tadah::hpo::loss_is_feasible(losses[0], fs));

        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.rows.size() == 2);
        CHECK(cell(outv, 0, "Ecoh[1]") != "NaN");
        CHECK(cell(outv, 1, "Ecoh[1]") == "NaN");
        CHECK(census(journal, "Unusable losses") == 1);
        Host::reset_globals_for_test();
    }
}

/* Pins: under FAILSCORE_GRADING ON (FAILSCORE 1e16) an evaluation whose
 * PC_OBSERVABLE term fails is a graded band-B failure measured by the
 * failed fraction of its PC_OBSERVABLE terms (pc_fail_grading.h): with the
 * failing observable alone, graded_failscore(1e16, B, 1, 1) at RCUT2B =
 * 5.2; with the failing one and one that is computed, graded_failscore(1e16,
 * B, 0.5, 1); its term stays FAILSCORE in loss.tadah; the census counts a
 * graded failure ("Graded failures : 1", "B=1 (script or observable
 * failed)"); at RCUT2B = 4.5 the loss is the mean of its terms.
 * Before the fix: an observable failure recorded no band, so the loss at
 * 5.2 was the mean of the terms, 5.0000000004e+15 and about 3.3e15, usable,
 * and "Graded failures : 0" (the band-B count was labelled "script
 * failed"). */
TEST_CASE("a failed PC_OBSERVABLE term is a graded band-B failure under "
          "FAILSCORE_GRADING ON", "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    struct Case { std::string observables; double fraction; };
    for (const Case& c : {Case{kFailingObservable, 1.0},
                          Case{kFailingObservable + kUsableObservable, 0.5}}) {
        INFO(c.observables);
        Host::reset_globals_for_test();
        ScratchDir sd;
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "FAILSCORE 1e16\nFAILSCORE_GRADING ON\nPC_ERMSE 0 1\n" +
                c.observables,
            {4.5, 5.2});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 2);
        CHECK(losses[1] ==
              graded_failscore(1e16, FailBand::B, c.fraction, 1.0));
        CHECK_FALSE(tadah::hpo::loss_is_feasible(losses[1], 1e16));
        CHECK(losses[1] < 1e16);

        const Table loss = read_table("loss.tadah");
        REQUIRE(loss.rows.size() == 2);
        CHECK(cell(loss, 1, "L(Ecoh)[1]") == "1.000000000000000e+16");
        double terms = std::stod(cell(loss, 0, "L(ERMSE)[avg]")) +
                       std::stod(cell(loss, 0, "L(Ecoh)[1]"));
        double n = 2.0;
        if (c.fraction < 1.0) {
            terms += std::stod(cell(loss, 0, "L(Ecoh)[2]"));
            n = 3.0;
            CHECK(cell(loss, 1, "L(Ecoh)[2]") != "1.000000000000000e+16");
        }
        CHECK(losses[0] == Approx(terms / n).epsilon(1e-13));
        CHECK(census(journal, "Graded failures") == 1);
        CHECK(journal.find("B=1 (script or observable failed)") !=
              std::string::npos);
        Host::reset_globals_for_test();
    }
}

/* Pins: a failed PC_OBSERVABLE term is journalled with its observable,
 * the evaluation and why it failed, as a failed PC_LAMMPS task is: a
 * WARNING "PC_OBSERVABLE term failed (cohesive_energy[1], eval <n>,
 * occurrence <k>): the relaxed lattice parameter a0 = <a> is outside its
 * --hotvar window [3, 3.35]" at the first, second and fourth failure of
 * the run (rate-limited on powers of two): evals 2, 3 and 5 of RCUT2B 4.5
 * (computed: no line), 5.2, 5.3, 5.4 and 5.5; the third failure (eval 4)
 * has no line; a0 is above 3.35.
 * Before the fix: no journal line named a failed term or its reason; the
 * evaluation's FAILSCORE (loss.tadah) and NaN value (outvar.tadah) were
 * all it left. */
TEST_CASE("a failed PC_OBSERVABLE term is journalled with its observable and "
          "why it failed", "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd;
    Journal j;
    const auto losses = evaluate(sd.dir, "PC_ERMSE 0 1\n" + kFailingObservable,
                                 {4.5, 5.2, 5.3, 5.4, 5.5});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(losses.size() == 5);
    for (std::size_t i = 1; i < losses.size(); ++i) CHECK(losses[i] == kMax);
    const std::regex line(
        R"(WARNING +PC_OBSERVABLE term failed \(cohesive_energy\[1\], eval )"
        R"((\d+), occurrence (\d+)\): the relaxed lattice parameter a0 = )"
        R"(([0-9.]+) is outside its --hotvar window \[3, 3\.35\])");
    std::vector<std::pair<int, int>> seen;
    for (auto it = std::sregex_iterator(journal.begin(), journal.end(), line);
         it != std::sregex_iterator(); ++it) {
        seen.emplace_back(std::stoi((*it)[1]), std::stoi((*it)[2]));
        CHECK(std::stod((*it)[3]) > 3.35);
    }
    CHECK(seen == std::vector<std::pair<int, int>>{{2, 1}, {3, 2}, {5, 4}});
    CHECK(count_of(journal, "PC_OBSERVABLE term failed") == 3u);
    Host::reset_globals_for_test();
}

/* An observable that fails as told: its evaluation throws, or it reports a
 * value that is not a number (without a reason of its own), or one whose
 * L2 loss overflows (1e300 against the target 0). */
class FailingStub final : public tadah::hpo::Observable {
public:
    enum class Mode { Throws, NotANumber, LossOverflows };
    explicit FailingStub(Mode m) : mode_(m) {}
    std::string kind() const override { return "stub"; }
    std::string column_tag() const override { return "Stub"; }
    std::string analytical_gradient_gap() const override { return {}; }
    tadah::hpo::ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base&, tadah::mlip::DC_Base&,
        const tadah::core::Context&) override {
        tadah::hpo::ObservableEvalResult res;
        switch (mode_) {
            case Mode::Throws:
                throw std::runtime_error("stub: the relaxation diverged");
            case Mode::NotANumber:
                res.value = std::numeric_limits<double>::quiet_NaN();
                break;
            case Mode::LossOverflows:
                res.value = 1e300;
                break;
        }
        return res;
    }
    double target() const override { return 0.0; }
    double weight() const override { return 1.0; }
    const tadah::LossFunction& lossfn() const override { return lossfn_; }

private:
    Mode mode_;
    tadah::LossFunction lossfn_{tadah::LossFunction::Type::L2};
};

/* The journal's "PC_OBSERVABLE term failed (...): <why>" reasons, one per
 * line, in journal order. */
std::vector<std::string> observable_failure_reasons(const std::string& journal,
                                                    const std::string& term) {
    std::vector<std::string> out;
    const std::string head = "PC_OBSERVABLE term failed (" + term + ", eval ";
    for (auto p = journal.find(head); p != std::string::npos;
         p = journal.find(head, p + head.size())) {
        const auto colon = journal.find("): ", p);
        const auto eol = journal.find('\n', p);
        if (colon == std::string::npos || colon > eol) continue;
        out.push_back(journal.substr(colon + 3, eol - colon - 3));
    }
    return out;
}

/* Pins: the journal line of a failed PC_OBSERVABLE term gives each of the
 * host's own reasons (HPO_Host::obs_loss): an evaluation that threw ("its
 * evaluation threw: stub: the relaxation diverged"), a value that is not a
 * number without a reason of the observable's ("its value is not finite
 * (nan)"), and a value whose loss is not finite ("its loss is not finite
 * (value 1e+300, --loss L2)"); each at eval 1, occurrence 1, and the
 * evaluation's loss is FAILSCORE.
 * Before the fix: passes as well (a pin of the reasons of the fix that
 * journals a failed term, which the test of that fix did not cover; before
 * that fix no journal line named a failed term). */
TEST_CASE("a failed PC_OBSERVABLE term's journal line says why: an evaluation "
          "that threw, a value or a loss that is not finite",
          "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    struct Case {
        FailingStub::Mode mode;
        const char* why;
    };
    for (const Case& c :
         {Case{FailingStub::Mode::Throws,
               "its evaluation threw: stub: the relaxation diverged"},
          Case{FailingStub::Mode::NotANumber, "its value is not finite (nan)"},
          Case{FailingStub::Mode::LossOverflows,
               "its loss is not finite (value 1e+300, --loss L2)"}}) {
        INFO(c.why);
        Host::reset_globals_for_test();
        ScratchDir sd;
        Journal j;
        const auto losses = evaluate(
            sd.dir, "PC_ERMSE 0 1\n" + kUsableObservable, {4.5},
            [&](HPOTargets& trg) {
                REQUIRE(trg.observables.size() == 1);
                trg.observables[0] = std::make_unique<FailingStub>(c.mode);
            });
        const std::string journal = j.finish();
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == kMax);
        CHECK(observable_failure_reasons(journal, "stub[1]") ==
              std::vector<std::string>{c.why});
        CHECK(count_of(journal, "PC_OBSERVABLE term failed (stub[1], eval 1, "
                                "occurrence 1): ") == 1u);
    }
    Host::reset_globals_for_test();
}

/* A cohesive_energy relaxation that does not converge (--maxeval 1, from a
 * compressed cell, a0 = 2.2 in the window [2, 5]) reports its --unconverged
 * value: the constant, or target + slope x fmax. When
 * that value is not finite the term fails, and the reason names the
 * strategy and its numbers. No PC_OBSERVABLE line gives such a value (the
 * parser reads only finite numbers, and the relaxed bcc cell's atom forces
 * vanish, so a finite slope does not overflow): a force-scaled value is not
 * finite when fmax is not, or slope x fmax overflows. The observable is
 * built here with a slope of +inf, and a constant of NaN.
 * Pins: the reason is "the relaxation did not converge (maxeval reached)
 * and its --unconverged force-scaled value, target + slope x fmax = 8.1 +
 * inf x <fmax> = <v>, is not finite", and "... and its --unconverged const
 * value nan is not finite"; no reason names --unconverged-penalty, a flag
 * the parser refuses.
 * Before the fix: both ended "... and its --unconverged-penalty value is
 * <v>". */
TEST_CASE("an unconverged cohesive_energy relaxation whose --unconverged value "
          "is not finite names its strategy", "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    using Cfg = tadah::hpo::CohesiveEnergyObservable::Config;
    struct Case {
        Cfg::UnconvergedStrategy strategy;
        double value;
        std::regex why;
    };
    const std::vector<Case> cases = {
        {Cfg::UnconvergedStrategy::ForceScaled,
         std::numeric_limits<double>::infinity(),
         std::regex(R"(the relaxation did not converge \(maxeval reached\) )"
                    R"(and its --unconverged force-scaled value, target \+ )"
                    R"(slope x fmax = 8\.1 \+ inf x [-+.0-9e]+ = -?(nan|inf), )"
                    R"(is not finite)")},
        {Cfg::UnconvergedStrategy::Const,
         std::numeric_limits<double>::quiet_NaN(),
         std::regex(R"(the relaxation did not converge \(maxeval reached\) )"
                    R"(and its --unconverged const value nan is not finite)")}};
    for (const Case& c : cases) {
        Host::reset_globals_for_test();
        ScratchDir sd;
        Journal j;
        const auto losses = evaluate(
            sd.dir, "PC_ERMSE 0 1\n" + kUsableObservable, {4.5},
            [&](HPOTargets& trg) {
                Cfg cfg;
                cfg.species = "Ta";
                cfg.nx = cfg.ny = cfg.nz = 2;
                cfg.target = 8.1;
                cfg.weight = 1.0;
                cfg.a0_min = 2.0;
                cfg.a0_max = 5.0;
                cfg.a0_initial = 2.2;  // compressed: not converged at once
                cfg.relax_maxeval = 1;
                cfg.unconverged_strategy = c.strategy;
                cfg.unconverged_value = c.value;
                REQUIRE(trg.observables.size() == 1);
                trg.observables[0] =
                    std::make_unique<tadah::hpo::CohesiveEnergyObservable>(cfg);
            });
        const std::string journal = j.finish();
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == kMax);
        const auto reasons =
            observable_failure_reasons(journal, "cohesive_energy[1]");
        REQUIRE(reasons.size() == 1);
        INFO(reasons[0]);
        CHECK(std::regex_match(reasons[0], c.why));
        CHECK(journal.find("--unconverged-penalty") == std::string::npos);
    }
    Host::reset_globals_for_test();
}

/* Pins: two failed PC_OBSERVABLE terms at the default FAILSCORE (the
 * largest double; RCUT2B = 5.2, both windows [3.0, 3.35]) fail their
 * evaluation as a failure, not as a NaN/Inf clamping: the loss is FAILSCORE,
 * the census counts it under "Unusable losses" and not under "NaN/Inf
 * clampings", and there is no "evaluation produced NaN/Inf" WARNING; under
 * FAILSCORE_GRADING ON with FAILSCORE 1e308, whose double overflows as well,
 * the graded band-B loss, counted under "Graded failures" only.
 * Before the fix: FAILSCORE + FAILSCORE overflowed the mean and
 * assemble_loss set was_nan_clamped before the failed terms made the
 * evaluation FAILSCORE: "evaluation produced NaN/Inf -- loss clamped to
 * FAIL_SCORE" and "NaN/Inf clampings : 1", under ON in both counters (one
 * failed term, whose mean does not overflow, was an ordinary failure). */
TEST_CASE("two failed PC_OBSERVABLE terms at the default FAILSCORE are a "
          "failed evaluation, not a NaN/Inf clamping",
          "[HPO][partial_failure][host]") {
    tadah::core::PeriodicTable::initialize();
    for (const bool graded : {false, true}) {
        INFO((graded ? "FAILSCORE_GRADING ON, FAILSCORE 1e308"
                     : "FAILSCORE_GRADING OFF, the default FAILSCORE"));
        Host::reset_globals_for_test();
        ScratchDir sd;
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            std::string(graded ? "FAILSCORE 1e308\nFAILSCORE_GRADING ON\n"
                               : "") +
                "PC_ERMSE 0 1\n" + kFailingObservable + kFailingObservable,
            {5.2});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == (graded ? graded_failscore(1e308, FailBand::B,
                                                       1.0, 1.0)
                                   : kMax));
        CHECK(count_of(journal, "PC_OBSERVABLE term failed (") == 2u);
        CHECK(journal.find("evaluation produced NaN/Inf") == std::string::npos);
        CHECK(census(journal, "NaN/Inf clampings") == 0);
        CHECK(census(journal, "Unusable losses") == 1);
        CHECK(census(journal, "Graded failures") == (graded ? 1 : 0));
    }
    Host::reset_globals_for_test();
}

/* ======================================================================
 * hpo_run(): the searches.
 * ==================================================================== */
namespace {

/* One hpo_run() setup in a scratch directory: the data, config.val and
 * config.hpo (@p optimizer, the OPTIMIZER block, then @p root), the fixed
 * potential starting at RCUT2B = 5.2, where the observable fails. */
struct PartialRun {
    ScratchDir sd;
    Context ctx;
    PartialRun(const std::string& optimizer, const std::string& root) {
        tadah::core::PeriodicTable::initialize();
        HPOLog::instance().reset_for_test();
        Host::reset_globals_for_test();
        write_data(sd.dir);
        const fs::path hpo = sd.dir / "config.hpo";
        std::ofstream(hpo) << optimizer << root;
        ctx = pinned_context(sd.dir / "train.db", hpo, sd.dir / "config.val",
                             5.2);
    }
    int run(const std::string& what) {
        int status = -1;
        hpo_test::finish_within(std::chrono::seconds(120), what, [&] {
            status = tadah::hpo::hpo_run(ctx);
        });
        return status;
    }
};

/* The objective and the box: L(ERMSE) and the failing observable, RCUT2B in
 * [@p lo, 5.5]; @p failscore_line is empty for the default FAILSCORE. */
std::string partial_root(const std::string& failscore_line,
                         const std::string& lo = "4.0") {
    return "LOSS L2\nQFLUSH 1000\nDIGITS 15\n" + failscore_line +
           "PC_ERMSE 0 1\n" + kFailingObservable + "OPTIM RCUT2B (1) " + lo +
           " 5.5\n";
}

std::string plain_block(const std::string& body) {
    return "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n" + body +
           "  ENDSTRATEGY\nENDOPTIMIZER\n";
}

const std::string kNoUsableLossStatus =
    "STOPPED -- no evaluation produced a usable loss";

/* What a run of @p evals evaluations that all failed leaves behind (as
 * test_hpo_termination.cpp's check_no_usable_loss): one loss.tadah row each,
 * @p fail_row; no best potential; the no-usable-loss trailer and verdict. */
void check_no_usable_loss(std::size_t evals, const std::string& fail_row) {
    const Table loss = read_table("loss.tadah");
    CHECK(loss.rows.size() == evals);
    for (std::size_t i = 0; i < loss.rows.size(); ++i) {
        INFO("loss.tadah row " << i);
        CHECK(loss.rows[i].back() == fail_row);
    }
    CHECK_FALSE(fs::exists("best_pot.tadah"));
    const std::string journal = read_file("hpo_run.log");
    CHECK(journal_status(journal) == kNoUsableLossStatus);
    CHECK(journal.find("[ok] Every evaluation produced a usable loss") ==
          std::string::npos);
    CHECK(journal.find("[!] No evaluation produced a usable loss.") !=
          std::string::npos);
    CHECK(census(journal, "Unusable losses") == static_cast<long>(evals));
}

}  // namespace

/* Pins: a gradient search (GRAD_STRATEGY HYBRID; the pinned weights leave
 * the analytical gradient nothing, so every parameter is differenced; PARAM
 * fd_step 0.05) that starts at RCUT2B = 5.2, where the observable fails,
 * stops at its first gradient under every backend: one evaluation, the
 * start, whose loss is FAILSCORE; the journal says "<backend> stopped: no
 * usable gradient: a failed evaluation at the point (the loss at RCUT2B =
 * 5.2 is 1.797693135e+308)"; the run ends without a usable loss (status 1,
 * no best_pot.tadah). Under FAILSCORE 1e16 DLIB:BFGS likewise ("... is
 * 1e+16"). The Ceres case runs CERES:STEEPEST_DESCENT (CERES:LBFGS, the
 * case until 2026-10-15, is refused since; before that change
 * CERES:STEEPEST_DESCENT passes as well, a pin).
 * Before the fix: the loss there was the mean of L(ERMSE) and FAILSCORE,
 * usable. Under the default FAILSCORE it was 8.988465674e+307 at every
 * point near the start, the gradient 0, and every backend converged on it
 * with status 0 and that loss in best_pot.tadah: DLIB:BFGS after 6
 * evaluations ("DLIB:BFGS: FTOL_ABS reached"), LD_LBFGS after 3 ("SUCCESS
 * -- generic success"), LD_SLSQP after 3 (roundoff-limited), CERES:LBFGS
 * after 3 ("tol_gradinf reached"), and LD_MMA spent 45 evaluations at it
 * (MAXEVAL). Under 1e16 the usable term was not absorbed, and DLIB:BFGS
 * followed its gradient out of the region, after publishing the start
 * (5.0000000004e+15) as the best. */
TEST_CASE("a gradient search that starts where a PC_OBSERVABLE term fails "
          "stops at its first gradient", "[HPO][partial_failure][run]") {
    struct Backend { std::string lib, algo, failscore_line, loss_text, row; };
    std::vector<Backend> backends = {
        {"DLIB", "BFGS", "", "1.797693135e+308", row_text(kMax)},
        {"NLOPT", "LD_LBFGS", "", "1.797693135e+308", row_text(kMax)},
        {"NLOPT", "LD_MMA", "", "1.797693135e+308", row_text(kMax)},
        {"NLOPT", "LD_SLSQP", "", "1.797693135e+308", row_text(kMax)},
        {"DLIB", "BFGS", "FAILSCORE 1e16\n", "1e+16", row_text(1e16)}};
#ifdef TADAH_USE_CERES
    backends.push_back({"CERES", "STEEPEST_DESCENT", "", "1.797693135e+308",
                        row_text(kMax)});
#endif
    for (const Backend& b : backends) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name << " " << b.failscore_line) {
            PartialRun run(plain_block("    LIB " + b.lib + "\n    ALGO " +
                                       b.algo + "\n    MAXEVAL 60\n"
                                       "    GRAD_STRATEGY HYBRID\n"
                                       "    PARAM fd_step 0.05\n"),
                           partial_root(b.failscore_line));
            int status = -1;
            REQUIRE_NOTHROW(status = run.run(name));
            CHECK(status == 1);
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK(count_of(journal,
                           name + " stopped: no usable gradient: a failed "
                                  "evaluation at the point (the loss at "
                                  "RCUT2B = 5.2 is " + b.loss_text + ")") ==
                  1u);
            check_no_usable_loss(1, b.row);
        }
    }
}

/* Pins: under FAILSCORE_GRADING ON (FAILSCORE 1e16) a gradient search --
 * DLIB:BFGS, LD_LBFGS, CERES:STEEPEST_DESCENT when built (CERES:LBFGS
 * until it was refused, 2026-10-15; CERES:STEEPEST_DESCENT passes before
 * that change as well, a pin); GRAD_STRATEGY HYBRID, every
 * parameter differenced, PARAM fd_step 0.05 -- that starts at RCUT2B =
 * 5.2, where the observable fails, stops at its first gradient: the start
 * and its stencil points 5.125 and 5.275 (the step is 0.05 of the box
 * width 1.5) are graded band-B failures with the same loss,
 * graded_failscore(1e16, B, 1, 1) = 9.99375e+15 (the failed fraction of
 * the one observable is 1 at each), so every difference is 0, and their
 * params.tadah rows are those three points; the journal says "<backend>
 * stopped: no usable gradient: a failed evaluation in the
 * finite-difference stencil (the loss at RCUT2B = 5.2 is 9.99375e+15, and
 * every difference at it is 0: a flat failure region)"; loss.tadah has
 * those 3 rows, each the graded loss, the census counts 3 graded failures
 * and 3 unusable losses, and the run ends without a usable loss (status 1,
 * no best_pot.tadah).
 * Before the fix: a failed observable recorded no band, so the loss there
 * was the mean of L(ERMSE) and FAILSCORE, 5.0000000004e+15, usable, as
 * under FAILSCORE_GRADING OFF, and every search followed the gradient of
 * L(ERMSE) out of the region, with status 0, a best_pot.tadah, no
 * flat-failure-region line and "Graded failures : 0": DLIB:BFGS made 9
 * evaluations and ended "DLIB:BFGS: FTOL_ABS reached" with the best
 * 7.7476e+04 at evaluation 4 (RCUT2B = 4.0, the bound), LD_LBFGS 6
 * ("SUCCESS -- generic success", fmin 77475.6), and CERES:LBFGS, which
 * ignores the box, 36 ("CERES:LBFGS: XTOL_REL reached", best 8.9619e+01
 * outside the box). */
TEST_CASE("a graded gradient search that starts where a PC_OBSERVABLE term "
          "fails stops at its first gradient", "[HPO][partial_failure][run]") {
    const double graded = graded_failscore(1e16, FailBand::B, 1.0, 1.0);
    std::ostringstream row;
    row << std::scientific << std::setprecision(15) << graded;
    struct Backend { std::string lib, algo; };
    std::vector<Backend> backends = {{"DLIB", "BFGS"},
                                     {"NLOPT", "LD_LBFGS"}};
#ifdef TADAH_USE_CERES
    backends.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    for (const Backend& b : backends) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            PartialRun run(plain_block("    LIB " + b.lib + "\n    ALGO " +
                                       b.algo + "\n    MAXEVAL 60\n"
                                       "    GRAD_STRATEGY HYBRID\n"
                                       "    PARAM fd_step 0.05\n"),
                           partial_root("FAILSCORE 1e16\n"
                                        "FAILSCORE_GRADING ON\n"));
            int status = -1;
            REQUIRE_NOTHROW(status = run.run(name + " graded"));
            CHECK(status == 1);
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK(count_of(journal,
                           name + " stopped: no usable gradient: a failed "
                                  "evaluation in the finite-difference "
                                  "stencil (the loss at RCUT2B = 5.2 is "
                                  "9.99375e+15, and every difference at it "
                                  "is 0: a flat failure region)") == 1u);
            CHECK(census(journal, "Graded failures") == 3);
            check_no_usable_loss(3, row.str());
            const Table params = read_table("params.tadah");
            REQUIRE(params.rows.size() == 3);
            std::vector<double> xs;
            for (const auto& r : params.rows) xs.push_back(std::stod(r.back()));
            std::sort(xs.begin(), xs.end());
            CHECK(xs[0] == Approx(5.125).epsilon(1e-12));
            CHECK(xs[1] == Approx(5.2).epsilon(1e-12));
            CHECK(xs[2] == Approx(5.275).epsilon(1e-12));
            /* Pins: the LOSS FUNCTION section's Fail grading line names the
             * failed PC_OBSERVABLE term, whose band-B losses these are.
             * Before the fix: "ON -- a skip cap, a failed PC_LAMMPS
             * simulation or a gate reject scores a graded loss ...". */
            CHECK_THAT(journal,
                       Catch::Contains("ON -- a skip cap, a failed "
                                       "PC_OBSERVABLE term or a gate reject "
                                       "scores a graded loss in [0.999, 1) x "
                                       "Fail score"));
        }
    }
}

namespace {

/* The trained variant of the fixture: the same D2_BP model fitted (OALGO 4,
 * LAMBDA 1e-6, no FIXINDEX), so the analytical gradient can be formed, to
 * training and validation energies of -40 and -39 eV. Its relaxed lattice
 * parameter is below 3.29 all over the OPTIM box (3.28682 at RCUT2B = 5.2),
 * so the observable with the window [3.29, 3.31] fails at every point. */
const std::string kTrainedFailingObservable =
    "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 --weight 1 "
    "--supercell 2 2 2 --hotvar a0 3.29 3.31 3.3\n";

struct TrainedRun {
    ScratchDir sd;
    Context ctx;
    TrainedRun(const std::string& optimizer, const std::string& root) {
        tadah::core::PeriodicTable::initialize();
        HPOLog::instance().reset_for_test();
        Host::reset_globals_for_test();
        write_ta_db(sd.dir / "train.db", 0.0, -40.0);
        write_ta_db(sd.dir / "val.db", 0.07, -39.0);
        std::ofstream(sd.dir / "config.val")
            << "DBFILE " << (sd.dir / "val.db").string() << "\n";
        const fs::path hpo = sd.dir / "config.hpo";
        std::ofstream(hpo) << optimizer << root;
        ctx = pinned_context(sd.dir / "train.db", hpo, sd.dir / "config.val",
                             5.2);
        ctx.remove("FIXINDEX");
        ctx.remove("FIXWEIGHT");
        ctx.remove("OALGO");
        ctx.remove("LAMBDA");
        ctx.add("OALGO", 4);
        ctx.add("LAMBDA", 1e-6);
    }
    int run(const std::string& what) {
        int status = -1;
        hpo_test::finish_within(std::chrono::seconds(120), what, [&] {
            status = tadah::hpo::hpo_run(ctx);
        });
        return status;
    }
};

std::string trained_root() {
    return "LOSS L2\nQFLUSH 1000\nDIGITS 15\nFAILSCORE 1e16\n"
           "FAILSCORE_GRADING ON\nPC_ERMSE 0 1\n" +
           kTrainedFailingObservable + kOptimRcut;
}

}  // namespace

/* Pins: under FAILSCORE_GRADING ON (FAILSCORE 1e16) a gradient search on a
 * trained model -- DLIB:BFGS, LD_LBFGS, CERES:STEEPEST_DESCENT when built
 * (CERES:LBFGS until it was refused, 2026-10-15; CERES:STEEPEST_DESCENT
 * passes before that change as well, a pin); no
 * GRAD_STRATEGY, which resolves to ANALYTICAL for PC_OBSERVABLE -- that
 * starts at RCUT2B = 5.2, where the observable fails (a graded band-B
 * failure, graded_failscore(1e16, B, 1, 1) = 9.99375e+15), stops at its
 * first gradient: the analytical gradient there is that of L(ERMSE) alone
 * (the failed term is left out of the chain), not of the graded loss the
 * optimiser sees, so the point has no usable gradient. The journal says
 * "<backend> stopped: no usable gradient: a failed evaluation at the point
 * (the loss at RCUT2B = 5.2 is 9.99375e+15: a graded failure, whose
 * grading the analytical gradient does not see; GRAD_STRATEGY HYBRID takes
 * finite differences there)"; loss.tadah has the one row, the run ends
 * without a usable loss (status 1).
 * Before the fix: the analytical gradient was kept at the graded failure,
 * every line search found no decrease and the backends reported
 * convergence inside the failure region: DLIB:BFGS "FTOL_ABS reached after
 * 4 evaluation(s)" (2 loss.tadah rows), LD_LBFGS 12 evaluations ending
 * "nlopt failure -- algorithm converged", CERES:LBFGS left the box (RCUT2B
 * = -4.5) and ended FAILED ("no analytical gradient for 1 parameter"). */
TEST_CASE("an analytical gradient search that starts at a graded "
          "PC_OBSERVABLE failure stops at its first gradient",
          "[HPO][partial_failure][run]") {
    const double graded = graded_failscore(1e16, FailBand::B, 1.0, 1.0);
    std::ostringstream row;
    row << std::scientific << std::setprecision(15) << graded;
    struct Backend { std::string lib, algo; };
    std::vector<Backend> backends = {{"DLIB", "BFGS"},
                                     {"NLOPT", "LD_LBFGS"}};
#ifdef TADAH_USE_CERES
    backends.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    for (const Backend& b : backends) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TrainedRun run(plain_block("    LIB " + b.lib + "\n    ALGO " +
                                       b.algo + "\n    MAXEVAL 60\n"),
                           trained_root());
            int status = -1;
            REQUIRE_NOTHROW(status = run.run(name + " analytical graded"));
            CHECK(status == 1);
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK_THAT(journal,
                       Catch::Contains("ANALYTICAL (resolved for "
                                       "PC_OBSERVABLE)"));
            CHECK(count_of(journal,
                           name + " stopped: no usable gradient: a failed "
                                  "evaluation at the point (the loss at "
                                  "RCUT2B = 5.2 is 9.99375e+15: a graded "
                                  "failure, whose grading the analytical "
                                  "gradient does not see; GRAD_STRATEGY "
                                  "HYBRID takes finite differences "
                                  "there)") == 1u);
            CHECK(census(journal, "Graded failures") == 1);
            check_no_usable_loss(1, row.str());
        }
    }
}

/* Pins: the same start under GRAD_STRATEGY HYBRID (PARAM fd_step 0.05)
 * differences every parameter at the graded failure, though the
 * analytical gradient can be formed there: the start and its stencil
 * points 5.125 and 5.275 are graded failures with the same loss, so every
 * difference is 0 and the search stops at its first gradient with "...
 * and every difference at it is 0: a flat failure region"; 3 loss.tadah
 * rows, 3 graded failures, status 1.
 * Before the fix: HYBRID took the analytical gradient for RCUT2B, which
 * has one, and differenced nothing: DLIB:BFGS ended "FTOL_ABS reached
 * after 4 evaluation(s)" with 2 rows, LD_LBFGS made 12 evaluations,
 * CERES:LBFGS left the box and ended "tol_gradinf reached" with status
 * 0. The Ceres case runs CERES:STEEPEST_DESCENT (CERES:LBFGS until it was
 * refused, 2026-10-15); it passes before that change as well (a pin). */
TEST_CASE("a HYBRID gradient search that starts at a graded PC_OBSERVABLE "
          "failure differences every parameter there",
          "[HPO][partial_failure][run]") {
    const double graded = graded_failscore(1e16, FailBand::B, 1.0, 1.0);
    std::ostringstream row;
    row << std::scientific << std::setprecision(15) << graded;
    struct Backend { std::string lib, algo; };
    std::vector<Backend> backends = {{"DLIB", "BFGS"},
                                     {"NLOPT", "LD_LBFGS"}};
#ifdef TADAH_USE_CERES
    backends.push_back({"CERES", "STEEPEST_DESCENT"});
#endif
    for (const Backend& b : backends) {
        const std::string name = b.lib + ":" + b.algo;
        DYNAMIC_SECTION(name) {
            TrainedRun run(plain_block("    LIB " + b.lib + "\n    ALGO " +
                                       b.algo + "\n    MAXEVAL 60\n"
                                       "    GRAD_STRATEGY HYBRID\n"
                                       "    PARAM fd_step 0.05\n"),
                           trained_root());
            int status = -1;
            REQUIRE_NOTHROW(status = run.run(name + " hybrid graded"));
            CHECK(status == 1);
            const std::string journal = read_file("hpo_run.log");
            INFO(journal);
            CHECK(count_of(journal,
                           name + " stopped: no usable gradient: a failed "
                                  "evaluation in the finite-difference "
                                  "stencil (the loss at RCUT2B = 5.2 is "
                                  "9.99375e+15, and every difference at it "
                                  "is 0: a flat failure region)") == 1u);
            CHECK(census(journal, "Graded failures") == 3);
            check_no_usable_loss(3, row.str());
            const Table params = read_table("params.tadah");
            REQUIRE(params.rows.size() == 3);
            std::vector<double> xs;
            for (const auto& r : params.rows) xs.push_back(std::stod(r.back()));
            std::sort(xs.begin(), xs.end());
            CHECK(xs[0] == Approx(5.125).epsilon(1e-12));
            CHECK(xs[1] == Approx(5.2).epsilon(1e-12));
            CHECK(xs[2] == Approx(5.275).epsilon(1e-12));
        }
    }
}

/* Pins: a TADAH:RANDOM run (SEED 3, MAXEVAL 8, FAILSCORE 1e16) whose every
 * evaluation lies where the observable fails (RCUT2B in [5.0, 5.5]) ends
 * without a usable loss: status 1, every loss.tadah row 1e+16, no
 * best_pot.tadah, "[!] No evaluation produced a usable loss.".
 * Before the fix: every loss was the mean 5.0000000004e+15 or
 * 5.0000000005e+15, usable: status 0, "[ok] Every evaluation produced a
 * usable loss.", and best_pot.tadah written with HPO_BEST_LOSS
 * 5.00000000041091500e+15. */
TEST_CASE("a run whose every evaluation has a failed PC_OBSERVABLE term has "
          "no usable loss", "[HPO][partial_failure][run]") {
    PartialRun run(plain_block("    LIB TADAH\n    ALGO RANDOM\n    SEED 3\n"
                               "    MAXEVAL 8\n"),
                   partial_root("FAILSCORE 1e16\n", "5.0"));
    int status = -1;
    REQUIRE_NOTHROW(status = run.run("TADAH:RANDOM in the failure region"));
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK(status == 1);
    check_no_usable_loss(8, row_text(1e16));
    /* Pins: the LOSS FUNCTION section names the failed PC_OBSERVABLE term
     * among what scores FAILSCORE, and no PC_LAMMPS simulation, which this
     * run cannot have.
     * Before the fix: "(the loss of a failed evaluation: a NaN/Inf loss,
     * an exception, a skip cap, a failed PC_LAMMPS simulation or a gate
     * reject; ...)", while every loss was FAILSCORE through the observable. */
    CHECK_THAT(journal, Catch::Contains("1.0000e+16  (the loss of a failed "
                                        "evaluation: a NaN/Inf loss, an "
                                        "exception, a skip cap, a failed "
                                        "PC_OBSERVABLE term or a gate reject; "
                                        "the ceiling of every loss)"));
    CHECK(journal.find("PC_LAMMPS simulation") == std::string::npos);
}

/* Pins: STOPVAL is reached by a usable loss only. TADAH:RANDOM (SEED 3,
 * FAILSCORE 1e16, STOPVAL 6e15) evaluates its configuration point first,
 * RCUT2B = 5.2, and then its first draw, RCUT2B = 5.2808121911, where the
 * observable fails too: both losses are FAILSCORE, and the run stops at
 * the next evaluation, RCUT2B = 4.1202553115, whose loss is usable:
 * "TADAH:RANDOM stopped: STOPVAL reached after 3 evaluation(s)", status
 * 0, best_pot.tadah that evaluation's. (TADAH:RANDOM draws a stream of
 * its own for its SEED, seed_stream.h; on the stream of the starting-point
 * method RANDOM, which it drew before, SEED 3's first draw was the usable
 * RCUT2B = 4.838148984 and the run stopped after 2 evaluations.)
 * Before the fix: the first evaluation's loss, the mean 5.0000000004e+15,
 * was usable and below STOPVAL: the run stopped after 1 evaluation with
 * that loss as its best. */
TEST_CASE("STOPVAL is not reached by an evaluation with a failed "
          "PC_OBSERVABLE term", "[HPO][partial_failure][run]") {
    PartialRun run(plain_block("    LIB TADAH\n    ALGO RANDOM\n    SEED 3\n"
                               "    MAXEVAL 20\n    STOPVAL 6e15\n"),
                   partial_root("FAILSCORE 1e16\n"));
    int status = -1;
    REQUIRE_NOTHROW(status = run.run("TADAH:RANDOM with STOPVAL"));
    const std::string journal = read_file("hpo_run.log");
    INFO(journal);
    CHECK(status == 0);
    CHECK_THAT(journal, Catch::Contains("TADAH:RANDOM stopped: STOPVAL "
                                        "reached after 3 evaluation(s)"));
    CHECK(journal_status(journal) == "TADAH:RANDOM: STOPVAL reached");
    const Table loss = read_table("loss.tadah");
    REQUIRE(loss.rows.size() == 3);
    CHECK(loss.rows[0].back() == row_text(1e16));
    CHECK(loss.rows[1].back() == row_text(1e16));
    const Table params = read_table("params.tadah");
    REQUIRE(params.rows.size() == 3);
    CHECK(std::stod(params.rows[1].back()) ==
          Approx(5.2808121911).epsilon(1e-9));
    CHECK(std::stod(params.rows[2].back()) ==
          Approx(4.1202553115).epsilon(1e-9));
    const double best = std::stod(loss.rows[2].back());
    CHECK(best < 6e15);
    REQUIRE(fs::exists("best_pot.tadah"));
    const std::string pot = read_file("best_pot.tadah");
    std::smatch m;
    REQUIRE(std::regex_search(pot, m,
                              std::regex(R"(HPO_BEST_LOSS\s+(\S+))")));
    CHECK(std::stod(m[1]) == Approx(best).epsilon(1e-12));
}

// test_hpo_pc_readback_e2e.cpp
//
// PC_LAMMPS variables read back after the script, end to end. HPO reads
// every --varloss/--outvar/--hotvar/--failvar by evaluating its formula in
// LAMMPS after the script has run, and a LAMMPS error can be raised there:
// a thermo keyword the script's thermo_style does not compute (`variable
// Pend equal press` under `thermo_style custom step pe`), or a divide by
// zero for one candidate. Covered here:
//   - [pc_readback_e2e] (hidden; CTest entry hpo_pc_readback_e2e, a process
//     of its own, label e2e): HPO_Host in-process, which is the
//     PC_ISOLATION THREAD path -- the only one of the MPI build, and the
//     fallback where socketpair is blocked. QFLUSH 1 writes each
//     evaluation's rows at once. An unreadable --varloss or --hotvar fails
//     the evaluation and an unreadable diagnostic --outvar is NaN, each
//     with a journal line naming the variable and the LAMMPS message (a
//     variable that reads an unreadable one carries that one's message);
//     and every row of loss.tadah, outvar.tadah and stats.tadah has its
//     header's columns, the cells an evaluation did not compute written as
//     NaN (`nan` in loss/stats.tadah, `NaN` in outvar.tadah). Hidden
//     because, before the fix, these cases ended their process with
//     SIGSEGV, and because the HPO_Host statics are shared by the cases of
//     one process.
//   - [.tadah_cli_pc_readback] (CTest entry hpo_tadah_cli_pc_readback): the
//     built tadah under PC_ISOLATION THREAD and PROCESS ends the same way,
//     with the same rows and, under FAILSCORE_GRADING, the same graded
//     losses (a --varloss failure graded by its failvar, an unreadable
//     PC_FAILPROBE S). It needs TADAH_BINARY (the entry sets it in a
//     build with the tadah target) and is Skipped without it; its PROCESS
//     half needs socketpair, like the pc pool tests.

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/runners.h>

#include <dlib/matrix.h>

#include <unistd.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives (HPO_Host
 * writes its row files to the cwd). */
struct ScratchDir {
    fs::path dir, original_cwd;
    explicit ScratchDir(const std::string& stem) {
        dir = fs::temp_directory_path() / unique_name(stem);
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
};

/* hpo_run.log in a temporary file, read back after finish(). */
struct Journal {
    fs::path path;
    Journal() : path(fs::temp_directory_path() /
                     (unique_name("tadah_pc_readback_log_") + ".log")) {
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

std::string read_text(const fs::path& p) {
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

/* Every row of @p t has its header's token count. */
void check_header_shaped(const Table& t, const std::string& file) {
    INFO(file << " header: " << t.header.size() << " columns");
    REQUIRE_FALSE(t.header.empty());
    for (std::size_t r = 0; r < t.rows.size(); ++r) {
        INFO("row " << r);
        CHECK(t.rows[r].size() == t.header.size());
    }
}

/* The minimal 4-atom bcc Ta structure of test_hpo_host_e2e.cpp. */
void write_minimal_ta_dbfile(const fs::path& path, double offset_x,
                             double energy = -0.001) {
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

/* Ta D2_BP (2 functions) + BF_Linear, trained by least squares: the
 * test_hpo_host_e2e.cpp master context, with CGRID2B as the OPTIM key.
 * The keys of the -c configuration file of the CLI case, which takes
 * DBFILE from the file and HPOTARGET/VALIDATION from the command line. */
constexpr const char* kTaModelKeys =
    "ATOM Ta\nWATOM 1\nINIT2B true\nINITMB false\n"
    "TYPE2B D2_BP 2 2 Ta Ta\nCGRID2B 0.5 1.4\nSGRID2B 3.0 5.0\n"
    "RCTYPE2B Cut_Cos\nRCUT2B 5.0\nFORCE false\nSTRESS false\n"
    "BIAS false\nNORM false\nVERBOSE 0\nMODEL M_BLR BF_Linear\n"
    "LAMBDA 1e-3\nOALGO 4\n";

Context make_ta_master_context(const fs::path& dbfile, const fs::path& hpo_cfg,
                               const fs::path& val_cfg) {
    Context c;
    c.add("DBFILE", dbfile.string());
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", 2);
    c.add("TYPE2B", 2);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "0.5 1.4");
    c.add("SGRID2B", "3.0 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", 5.0);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    c.add("EWEIGHT", 1.0);
    c.add("FWEIGHT", 1.0);
    c.add("SWEIGHT", 1.0);
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    c.add("HPOTARGET", hpo_cfg.string());
    c.add("VALIDATION", val_cfg.string());
    return c;
}

/* A bcc Ta cell run once with the candidate potential under a thermo_style
 * that computes pe but not press, then the variables @p tail (one
 * `variable` line each). */
std::string ta_script(const std::string& lattice, const std::string& tail) {
    return "clear\nunits metal\nboundary p p p\natom_style atomic\n"
           "lattice bcc " + lattice + "\n"
           "region box block 0 2 0 2 0 2 units lattice\n"
           "create_box 1 box\ncreate_atoms 1 region box\nmass 1 180.95\n"
           "pair_style tadah\npair_coeff * * EXTERN Ta\n"
           "thermo_style custom step pe\nrun 0 post no\n" + tail;
}

/* The two LAMMPS errors a read-back can raise, with the text each one's
 * message carries. */
struct ReadbackError {
    const char* formula;
    const char* lammps_text;
};
const ReadbackError kPressError{"press", "Thermo keyword press in variable "
                                         "requires thermo to use/init press"};
const ReadbackError kDivideError{"1.0/(pe-pe)",
                                 "Divide by 0 in variable formula"};

const std::string kHpoHead =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
    "LOSS L2\nQFLUSH 1\nFAILSCORE 1e16\nOPTIM CGRID2B (1) 0.0 4.0\n";

/* One HPO_Host in the cwd on the minimal Ta data, with the hpotarget
 * kHpoHead + @p target_tail; evaluates each candidate CGRID2B[1] in @p xs,
 * flushes the rows and returns the losses. */
std::vector<double> evaluate(const fs::path& dir, const std::string& target_tail,
                             const std::vector<double>& xs,
                             std::size_t n_val_datasets = 1) {
    const fs::path train = dir / "train.db", hpo = dir / "config.hpo",
                   val = dir / "config.val";
    write_minimal_ta_dbfile(train, 0.0);
    {
        std::ofstream os(val);
        for (std::size_t k = 0; k < n_val_datasets; ++k) {
            const fs::path db = dir / ("val" + std::to_string(k + 1) + ".db");
            write_minimal_ta_dbfile(db, 0.07 + 0.03 * k, -0.04 - 0.02 * k);
            os << "DBFILE " << db.string() << "\n";
        }
    }
    std::ofstream(hpo) << kHpoHead << target_tail;
    Context ctx = make_ta_master_context(train, hpo, val);
    HPOTargets trg(ctx, hpo.string());
    std::vector<double> losses;
    Host host(ctx, trg, val.string());
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

/* The journal line that contains @p needle ("" when none does). */
std::string line_with(const std::string& text, const std::string& needle) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(needle) != std::string::npos) return line;
    return {};
}

}  // namespace

/* Pins: a --varloss variable whose formula raises a LAMMPS error when HPO
 * reads it after the script (`variable E equal press` under a thermo_style
 * without press; `1.0/(pe-pe)`) fails that evaluation, not the run, under
 * PC_ISOLATION THREAD: each evaluation returns FAILSCORE, its rows are
 * written with the header's columns (E failscored in outvar.tadah, its loss
 * in loss.tadah), and the journal says, once per occurrence on the
 * powers-of-two schedule, "PC_LAMMPS task failed (readback.in, occurrence
 * n): reading --varloss E after the script: " and the LAMMPS message; the
 * evaluation is a failure, not a NaN clamp, and not the evaluation-wide
 * catch ("eval-thread caught throw").
 * Before the fix: SIGSEGV at the first row write. The read's LAMMPS
 * exception left run_lammps before the losses were sized, the evaluation's
 * catch submitted the partial Result (counted as a NaN clamp), and
 * print_loss_line read its empty v_loss out of bounds. */
TEST_CASE("an unreadable --varloss variable fails the evaluation, not the run",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    for (const ReadbackError& err : {kPressError, kDivideError}) {
        INFO("E = " << err.formula);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_readback_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", std::string("variable E equal ") +
                                      err.formula + "\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() + " --varloss E 0 1\n",
            {0.5, 0.6});
        const std::string journal = j.finish();
        INFO(journal);
        for (const double l : losses) CHECK(l == 1e16);

        const Table loss = read_table("loss.tadah");
        CHECK(loss.header == std::vector<std::string>{
                  "#ITERATION", "SLOT", "L(ERMSE)[1]", "L(ERMSE)[avg]",
                  "L(E)", "TOTAL_LOSS/N"});
        REQUIRE(loss.rows.size() == 2);
        check_header_shaped(loss, "loss.tadah");
        for (const auto& row : loss.rows) {
            CHECK(row.at(4) == "1.000000e+32");   // L2 of E = 1e16
            CHECK(row.at(5) == "1.000000e+16");
        }
        const Table outv = read_table("outvar.tadah");
        CHECK(outv.header == std::vector<std::string>{
                  "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E"});
        REQUIRE(outv.rows.size() == 2);
        check_header_shaped(outv, "outvar.tadah");
        for (const auto& row : outv.rows) CHECK(row.at(4) == "1.000000e+16");
        check_header_shaped(read_table("stats.tadah"), "stats.tadah");

        for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
            const std::string l = line_with(
                journal, std::string("readback.in, ") + n +
                             ": reading --varloss E after the script: ");
            INFO(n);
            CHECK_THAT(l, Catch::Contains("PC_LAMMPS task failed (") &&
                              Catch::Contains(err.lammps_text));
        }
        CHECK(journal.find("eval-thread caught throw") == std::string::npos);
        CHECK(journal.find("[ok] No NaN/Inf clampings were recorded.") !=
              std::string::npos);
        CHECK(journal.find("[!] No evaluation produced a usable loss.") !=
              std::string::npos);
        Host::reset_globals_for_test();
    }
}

/* Pins: a variable declared only with --outvar -- a diagnostic, not in the
 * loss -- whose read-back raises a LAMMPS error is NaN in its outvar.tadah
 * column, and the evaluation keeps its loss: bit for bit the loss of the
 * same candidates when the diagnostic reads fine, E as read in that run,
 * every row header-shaped with no other NaN. The journal says, once per
 * occurrence on the powers-of-two schedule, "PC_LAMMPS --outvar is NaN
 * (readback.in, occurrence n): reading --outvar Pend after the script: " and
 * the LAMMPS message; nothing fails and nothing is clamped.
 * Before the fix: SIGSEGV at the first row write (as for --varloss: the
 * read's exception reached the evaluation's catch, whose partial Result the
 * row writer read out of bounds). PROCESS isolation failed the whole
 * script, E included, so every such evaluation was lost. */
TEST_CASE("an unreadable diagnostic --outvar is NaN and the evaluation keeps "
          "its loss", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    const std::vector<double> xs = {0.5, 0.6};
    const std::string target_tail = "PC_ERMSE 0 1\nPC_LAMMPS --script ";
    const std::string lammps_tail = " --varloss E 0 1 --outvar Pend\n";

    /* The reference: the same candidates with a readable Pend. */
    std::vector<double> want;
    Table want_outv;
    {
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_readback_ref_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable Pend equal pe/atoms\n");
        want = evaluate(sd.dir,
                        target_tail + (sd.dir / "readback.in").string() +
                            lammps_tail, xs);
        want_outv = read_table("outvar.tadah");
        REQUIRE(want_outv.rows.size() == 2);
        for (const double l : want)
            REQUIRE(tadah::hpo::loss_is_feasible(l, 1e16));
        Host::reset_globals_for_test();
    }

    for (const ReadbackError& err : {kPressError, kDivideError}) {
        INFO("Pend = " << err.formula);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_readback_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", std::string("variable E equal pe\n"
                                              "variable Pend equal ") +
                                      err.formula + "\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir, target_tail + (sd.dir / "readback.in").string() +
                        lammps_tail, xs);
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == want.size());
        for (std::size_t k = 0; k < losses.size(); ++k) {
            INFO("candidate " << k);
            CHECK(losses[k] == want[k]);
        }

        const Table outv = read_table("outvar.tadah");
        CHECK(outv.header == std::vector<std::string>{
                  "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E", "Pend"});
        REQUIRE(outv.rows.size() == 2);
        check_header_shaped(outv, "outvar.tadah");
        for (std::size_t r = 0; r < outv.rows.size(); ++r) {
            CHECK(outv.rows[r].at(3) == want_outv.rows[r].at(3));
            CHECK(outv.rows[r].at(4) == want_outv.rows[r].at(4));
            CHECK(outv.rows[r].at(5) == "NaN");
        }
        const Table loss = read_table("loss.tadah");
        REQUIRE(loss.rows.size() == 2);
        check_header_shaped(loss, "loss.tadah");
        CHECK(read_text("loss.tadah").find("nan") == std::string::npos);

        for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
            const std::string l = line_with(
                journal, std::string("readback.in, ") + n +
                             ": reading --outvar Pend after the script: ");
            INFO(n);
            CHECK_THAT(l, Catch::Contains("PC_LAMMPS --outvar is NaN (") &&
                              Catch::Contains(err.lammps_text));
        }
        CHECK(journal.find("PC_LAMMPS task failed") == std::string::npos);
        CHECK(journal.find("eval-thread caught throw") == std::string::npos);
        CHECK(journal.find("[ok] No NaN/Inf clampings were recorded.") !=
              std::string::npos);
        CHECK(journal.find("[ok] Every evaluation produced a usable loss.") !=
              std::string::npos);
        Host::reset_globals_for_test();
    }
}

/* Pins: a --hotvar whose read-back raises a LAMMPS error fails the hot
 * script and with it the evaluation: FAILSCORE, the hot script's E and the
 * cold script's E2 failscored, the journal line "PC_LAMMPS task failed
 * (hot.in, occurrence 1): reading --hotvar hot_a0 after the script: " and
 * the LAMMPS message, and the cold script, which needs the hot value, not
 * run (no line about it).
 * Before the fix: the read's error was caught and the value taken as
 * absent, so the evaluation went on to the cold script without a hot value
 * (here it failed on the undefined ${hot_a0}, silently; a cold script
 * that had seen the value before would have run on the previous
 * evaluation's). The journal had no line naming hot_a0: that check fails. */
TEST_CASE("an unreadable --hotvar fails the evaluation and the cold scripts "
          "do not run", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_readback_hot_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal press\n");
    std::ofstream(sd.dir / "cold.in")
        << ta_script("${hot_a0}", "variable E2 equal pe\n");
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
            " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301\n"
        "PC_LAMMPS --script " + (sd.dir / "cold.in").string() +
            " --varloss E2 0 1\n",
        {0.5});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(losses.size() == 1);
    CHECK(losses[0] == 1e16);
    const Table outv = read_table("outvar.tadah");
    CHECK(outv.header == std::vector<std::string>{
              "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E", "E2"});
    REQUIRE(outv.rows.size() == 1);
    CHECK(outv.rows[0].at(4) == "1.000000e+16");
    CHECK(outv.rows[0].at(5) == "1.000000e+16");
    CHECK_THAT(line_with(journal, "hot.in, occurrence 1): reading --hotvar "
                                  "hot_a0 after the script: "),
               Catch::Contains("PC_LAMMPS task failed (") &&
                   Catch::Contains(kPressError.lammps_text));
    CHECK(journal.find("cold.in, occurrence") == std::string::npos);
    CHECK(journal.find("eval-thread caught throw") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: a variable that reads an unreadable one is journalled with the
 * LAMMPS error of that one, as reading it raised it, never with "has a
 * circular dependency":
 *   - `--outvar Pend --varloss Pdiff 0 1` with Pdiff = v_Pend-1.0 (Pend
 *     read first): the evaluation fails, "PC_LAMMPS task failed
 *     (readback.in, occurrence n): reading --varloss Pdiff after the
 *     script: " and Pend's LAMMPS message, Pend and Pdiff failscored in
 *     outvar.tadah, and no "PC_LAMMPS --outvar is NaN" line (Pend is not a
 *     NaN: its task failed);
 *   - the derived diagnostic `--varloss E 0 1 --outvar P --outvar Pgpa`
 *     with Pgpa = v_P/10000: the evaluation keeps its loss, P and Pgpa are
 *     NaN, and every journalled line carries P's LAMMPS message: "PC_LAMMPS
 *     --outvar is NaN (readback.in, occurrence n): reading --outvar P after
 *     the script: " for occurrence 1 and "... reading --outvar Pgpa after
 *     the script: " for occurrences 2 and 4 (Pgpa is read second in each
 *     evaluation).
 * Each for `press` under a thermo_style without it and for `1.0/(pe-pe)`.
 * Before the fix: the first read's error left Pend (P) marked as being
 * evaluated in LAMMPS, so the read of Pdiff (Pgpa) raised "Variable Pend:
 * has a circular dependency", which the journal reported instead of the
 * real cause; the message checks fail. */
TEST_CASE("a variable that reads an unreadable one is journalled with its "
          "LAMMPS error", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    for (const ReadbackError& err : {kPressError, kDivideError}) {
        INFO("formula " << err.formula);
        {
            INFO("--outvar Pend --varloss Pdiff");
            Host::reset_globals_for_test();
            ScratchDir sd("tadah_pc_readback_dep_");
            std::ofstream(sd.dir / "readback.in")
                << ta_script("3.301", std::string("variable Pend equal ") +
                                          err.formula + "\n"
                                          "variable Pdiff equal v_Pend-1.0\n");
            Journal j;
            const auto losses = evaluate(
                sd.dir,
                "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                    (sd.dir / "readback.in").string() +
                    " --outvar Pend --varloss Pdiff 0 1\n",
                {0.5, 0.6});
            const std::string journal = j.finish();
            INFO(journal);
            for (const double l : losses) CHECK(l == 1e16);
            const Table outv = read_table("outvar.tadah");
            CHECK(outv.header == std::vector<std::string>{
                      "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "Pend",
                      "Pdiff"});
            REQUIRE(outv.rows.size() == 2);
            check_header_shaped(outv, "outvar.tadah");
            for (const auto& row : outv.rows)
                if (row.size() == 6) {
                    CHECK(row[4] == "1.000000e+16");
                    CHECK(row[5] == "1.000000e+16");
                }
            for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
                const std::string l = line_with(
                    journal, std::string("readback.in, ") + n +
                                 ": reading --varloss Pdiff after the "
                                 "script: ");
                INFO(n);
                CHECK_THAT(l, Catch::Contains("PC_LAMMPS task failed (") &&
                                  Catch::Contains(err.lammps_text));
            }
            CHECK(journal.find("circular dependency") == std::string::npos);
            CHECK(journal.find("PC_LAMMPS --outvar is NaN") ==
                  std::string::npos);
            Host::reset_globals_for_test();
        }
        {
            INFO("--outvar P --outvar Pgpa");
            Host::reset_globals_for_test();
            ScratchDir sd("tadah_pc_readback_derived_");
            std::ofstream(sd.dir / "readback.in")
                << ta_script("3.301", std::string("variable E equal pe\n"
                                                  "variable P equal ") +
                                          err.formula + "\n"
                                          "variable Pgpa equal v_P/10000\n");
            Journal j;
            const auto losses = evaluate(
                sd.dir,
                "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                    (sd.dir / "readback.in").string() +
                    " --varloss E 0 1 --outvar P --outvar Pgpa\n",
                {0.5, 0.6});
            const std::string journal = j.finish();
            INFO(journal);
            for (const double l : losses)
                CHECK(tadah::hpo::loss_is_feasible(l, 1e16));
            const Table outv = read_table("outvar.tadah");
            REQUIRE(outv.header.size() == 7);
            CHECK(outv.header[5] == "P");
            CHECK(outv.header[6] == "Pgpa");
            REQUIRE(outv.rows.size() == 2);
            check_header_shaped(outv, "outvar.tadah");
            for (const auto& row : outv.rows)
                if (row.size() == 7) {
                    CHECK(row[5] == "NaN");
                    CHECK(row[6] == "NaN");
                }
            CHECK_THAT(line_with(journal, "readback.in, occurrence 1): "
                                          "reading --outvar P after the "
                                          "script: "),
                       Catch::Contains("PC_LAMMPS --outvar is NaN (") &&
                           Catch::Contains(err.lammps_text));
            for (const char* n : {"occurrence 2)", "occurrence 4)"}) {
                const std::string l = line_with(
                    journal, std::string("readback.in, ") + n +
                                 ": reading --outvar Pgpa after the script: ");
                INFO(n);
                CHECK_THAT(l, Catch::Contains("PC_LAMMPS --outvar is NaN (") &&
                                  Catch::Contains(err.lammps_text));
            }
            CHECK(journal.find("circular dependency") == std::string::npos);
            CHECK(journal.find("PC_LAMMPS task failed") == std::string::npos);
            Host::reset_globals_for_test();
        }
    }
}

/* Pins: a row of an evaluation that did not compute every column has its
 * header's columns, each cell it did not compute written as NaN -- `nan` in
 * loss.tadah and stats.tadah (the spelling those files use for a dataset
 * that is not measured), `NaN` in outvar.tadah (the spelling it uses for a
 * value that cannot be read) -- and TOTAL_LOSS/N last:
 *   - PC_ERMSE --skip-above stops the evaluation before PC_LAMMPS: L(E) is
 *     nan, E and Pend are NaN, the ERMSE cells are numbers;
 *   - and before PC_OBSERVABLE: L(Ecoh)[1] is nan, Ecoh[1] NaN.
 * Before the fix: each row wrote only the columns the evaluation had, so
 * loss.tadah's TOTAL_LOSS/N landed under L(E) (5 tokens under a 6-token
 * header) and outvar.tadah lacked E and Pend (4 under 6); the same for the
 * observable. The column-count checks fail. */
TEST_CASE("rows of an evaluation stopped before its simulations have the "
          "header's columns", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_readback_rows_");

    SECTION("--skip-above before PC_LAMMPS") {
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable Pend equal pe/atoms\n");
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1 --skip-above 1e-9\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() +
                " --varloss E 0 1 --outvar Pend\n",
            {0.5});
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        const Table loss = read_table("loss.tadah");
        CHECK(loss.header == std::vector<std::string>{
                  "#ITERATION", "SLOT", "L(ERMSE)[1]", "L(ERMSE)[avg]",
                  "L(E)", "TOTAL_LOSS/N"});
        REQUIRE(loss.rows.size() == 1);
        check_header_shaped(loss, "loss.tadah");
        const auto& lr = loss.rows[0];
        if (lr.size() == 6) {
            CHECK(lr[2] != "nan");
            CHECK(lr[3] != "nan");
            CHECK(lr[4] == "nan");
            CHECK(lr[5] == "1.000000e+16");
        }

        const Table outv = read_table("outvar.tadah");
        CHECK(outv.header == std::vector<std::string>{
                  "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E", "Pend"});
        REQUIRE(outv.rows.size() == 1);
        check_header_shaped(outv, "outvar.tadah");
        const auto& orow = outv.rows[0];
        if (orow.size() == 6) {
            CHECK(orow[3] != "NaN");
            CHECK(orow[4] == "NaN");
            CHECK(orow[5] == "NaN");
        }

        const Table stats = read_table("stats.tadah");
        REQUIRE(stats.rows.size() == 1);
        check_header_shaped(stats, "stats.tadah");
        CHECK(read_text("stats.tadah").find("nan") == std::string::npos);
    }
    SECTION("--skip-above before PC_OBSERVABLE") {
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1 --skip-above 1e-9\n"
            "PC_OBSERVABLE cohesive_energy --species Ta --target 8.10 "
            "--weight 1 --hotvar a0 2.5 4.5 3.302\n",
            {0.5});
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        const Table loss = read_table("loss.tadah");
        REQUIRE(loss.header.size() == 6);
        CHECK(loss.header[4] == "L(Ecoh)[1]");
        REQUIRE(loss.rows.size() == 1);
        check_header_shaped(loss, "loss.tadah");
        if (loss.rows[0].size() == 6) {
            CHECK(loss.rows[0][4] == "nan");
            CHECK(loss.rows[0][5] == "1.000000e+16");
        }
        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.header.size() == 5);
        CHECK(outv.header[4] == "Ecoh[1]");
        REQUIRE(outv.rows.size() == 1);
        check_header_shaped(outv, "outvar.tadah");
        if (outv.rows[0].size() == 5) CHECK(outv.rows[0][4] == "NaN");
    }
    Host::reset_globals_for_test();
}

/* Pins: a candidate the bump-support gate (BUMPGATE CANDIDATE, the
 * default) rejects before training -- so before validation, LAMMPS and the
 * observables -- writes header-shaped rows of not-computed cells: every
 * L(EMAE) and EMAE cell, per dataset (two validation datasets here) and
 * average, is nan in loss.tadah and stats.tadah and NaN in outvar.tadah
 * (not a computed-looking 0), L(E) is nan and E and Pend NaN; TOTAL_LOSS/N
 * is the gate's FAILSCORE.
 * Before the fix: the rows held what the default Result had -- loss.tadah
 * "0.000000e+00" for L(EMAE)[avg] and nothing else (4 tokens under 7),
 * outvar.tadah EMAE "0.000000e+00" and no E or Pend (4 under 6),
 * stats.tadah a 0 average (3 under 5). */
TEST_CASE("rows of a candidate rejected before training have the header's "
          "columns", "[pc_readback_e2e][.]") {
    using tadah::hpo::loss_is_feasible;
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_readback_gate_");

    const fs::path train = sd.dir / "train.db", hpo = sd.dir / "config.hpo",
                   val = sd.dir / "config.val";
    write_minimal_ta_dbfile(train, 0.0);
    {
        std::ofstream os(val);
        for (int k = 1; k <= 2; ++k) {
            const fs::path db = sd.dir / ("val" + std::to_string(k) + ".db");
            write_minimal_ta_dbfile(db, 0.04 * k, -0.04 * k);
            os << "DBFILE " << db.string() << "\n";
        }
    }
    std::ofstream(sd.dir / "readback.in")
        << ta_script("3.301", "variable E equal pe\n"
                              "variable Pend equal pe/atoms\n");
    /* The test_bump_support_gate.cpp model: two D2_Blip bumps (w = 2)
     * under Cut_Dummy whose seed supports end exactly at RCUT2B = 5. */
    std::ofstream(hpo)
        << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
           "LOSS L2\nQFLUSH 1\nFAILSCORE 1e9\nPC_EMAE 0 1\n"
           "OPTIM CGRID2B (1) 3.0 4.8\nOPTIM SGRID2B (1) 1.0 3.0\n"
           "PC_LAMMPS --script " << (sd.dir / "readback.in").string()
        << " --varloss E 0 1 --outvar Pend\n";
    Context ctx = make_ta_master_context(train, hpo, val);
    ctx.remove("TYPE2B");
    ctx.add("TYPE2B", "D2_Blip");
    ctx.add("TYPE2B", 2);
    ctx.add("TYPE2B", 2);
    ctx.add("TYPE2B", "Ta");
    ctx.add("TYPE2B", "Ta");
    ctx.remove("CGRID2B");
    ctx.add("CGRID2B", "4.0 3.0");
    ctx.remove("SGRID2B");
    ctx.add("SGRID2B", "2.0 1.0");
    ctx.remove("RCTYPE2B");
    ctx.add("RCTYPE2B", "Cut_Dummy");
    HPOTargets trg(ctx, hpo.string());
    REQUIRE_FALSE(trg.BUMPGATE_STRICT);
    {
        Host host(ctx, trg, val.string());
        host.prepare_best_publication();
        ParamVec p(2);
        p(0) = 4.8;   // CGRID2B[1]
        p(1) = 1.0;   // SGRID2B[1]: the bump reaches 6.8 > RCUT2B
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = host(p));
        CHECK(loss == 1e9);
        CHECK_FALSE(loss_is_feasible(loss, 1e9));
        host.flush_remaining();
    }

    const Table l = read_table("loss.tadah");
    CHECK(l.header == std::vector<std::string>{
              "#ITERATION", "SLOT", "L(EMAE)[1]", "L(EMAE)[2]",
              "L(EMAE)[avg]", "L(E)", "TOTAL_LOSS/N"});
    REQUIRE(l.rows.size() == 1);
    check_header_shaped(l, "loss.tadah");
    if (l.rows[0].size() == 7) {
        for (std::size_t c = 2; c <= 5; ++c) {
            INFO("loss.tadah column " << l.header.at(c));
            CHECK(l.rows[0][c] == "nan");
        }
        CHECK(l.rows[0][6] == "1.000000e+09");
    }

    const Table o = read_table("outvar.tadah");
    CHECK(o.header == std::vector<std::string>{
              "#ITERATION", "IT-TIME/ms", "SLOT", "EMAE", "E", "Pend"});
    REQUIRE(o.rows.size() == 1);
    check_header_shaped(o, "outvar.tadah");
    if (o.rows[0].size() == 6) {
        for (std::size_t c = 3; c <= 5; ++c) {
            INFO("outvar.tadah column " << o.header.at(c));
            CHECK(o.rows[0][c] == "NaN");
        }
    }

    const Table s = read_table("stats.tadah");
    CHECK(s.header == std::vector<std::string>{
              "#ITERATION", "SLOT", "EMAE[1]", "EMAE[2]", "EMAE[avg]"});
    REQUIRE(s.rows.size() == 1);
    check_header_shaped(s, "stats.tadah");
    if (s.rows[0].size() == 5) {
        for (std::size_t c = 2; c <= 4; ++c) {
            INFO("stats.tadah column " << s.header.at(c));
            CHECK(s.rows[0][c] == "nan");
        }
    }
    Host::reset_globals_for_test();
}

namespace {

/* `tadah hpo` in its own directory @p dir under PC_ISOLATION @p mode, with
 * the hpotarget @p target (PC_ISOLATION appended), the script readback.in
 * = @p script and each of @p more_scripts (name, text); returns the wait
 * status. tadah runs without TADAH_PC_ISOLATION, which would override the
 * hpotarget's PC_ISOLATION (src/hpo.cpp) -- a developer where socketpair
 * is blocked is told to set it to THREAD -- so @p mode is the mode it runs
 * in, as each case checks in its journal. */
int run_tadah_hpo(
        const char* bin, const fs::path& dir, const std::string& mode,
        const std::string& target, const std::string& script,
        const std::vector<std::pair<std::string, std::string>>& more_scripts) {
    fs::create_directories(dir);
    write_minimal_ta_dbfile(dir / "train.db", 0.0);
    write_minimal_ta_dbfile(dir / "val.db", 0.07, -0.04);
    std::ofstream(dir / "config.val") << "DBFILE " << (dir / "val.db").string()
                                      << "\n";
    std::ofstream(dir / "config") << "DBFILE " << (dir / "train.db").string()
                                  << "\n" << kTaModelKeys;
    std::ofstream(dir / "readback.in") << script;
    for (const auto& s : more_scripts) std::ofstream(dir / s.first) << s.second;
    std::ofstream(dir / "config.hpo") << target << "PC_ISOLATION " << mode
                                      << "\n";
    const std::string cmd =
        "cd '" + dir.string() + "' && env -u TADAH_PC_ISOLATION '" + bin +
        "' hpo -c config --hpotarget config.hpo --validation config.val"
        " > tadah.stdout 2> tadah.stderr";
    return std::system(cmd.c_str());
}

/* A row file's lines with the token at @p drop (0-based) left out, every
 * line; @p drop < 0 keeps every token. */
std::vector<std::vector<std::string>> lines_without(const fs::path& p,
                                                    int drop) {
    std::vector<std::vector<std::string>> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        auto t = tokens(line);
        if (drop >= 0 && static_cast<std::size_t>(drop) < t.size())
            t.erase(t.begin() + drop);
        out.push_back(std::move(t));
    }
    return out;
}

/* @p v as loss.tadah and outvar.tadah print it (the default DIGITS). */
std::string sci6(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6e", v);
    return buf;
}

/* The column of @p t headed @p name (header.size() when none is). */
std::size_t column(const Table& t, const std::string& name) {
    std::size_t c = 0;
    while (c < t.header.size() && t.header[c] != name) ++c;
    return c;
}

}  // namespace

/* Pins, through the built tadah: a LAMMPS error while a PC_LAMMPS variable
 * is read back, and one inside the script, end each evaluation the same way
 * under PC_ISOLATION THREAD and PROCESS (RANDOM, MAXEVAL 3; each run's
 * journal names the mode it ran in): tadah exits normally -- 0 when the
 * evaluations keep a loss, 1 when none does ("no evaluation produced a
 * usable loss") -- with the same journal line in both modes and the same
 * rows: loss.tadah, params.tadah, stats.tadah and their best_* twins byte
 * for byte, outvar.tadah and best_outvar.tadah without their IT-TIME/ms
 * column (wall time).
 *   - a diagnostic --outvar Pend reading `press`: exit 0, "PC_LAMMPS
 *     --outvar is NaN (", "reading --outvar Pend after the script: ERROR:
 *     Thermo keyword press ...", Pend NaN in outvar.tadah;
 *   - --varloss E reading `press`: exit 1, "PC_LAMMPS task failed (",
 *     "reading --varloss E after the script: ERROR: Thermo keyword press";
 *   - `bogus_command 1 2` in the script: exit 1, "PC_LAMMPS task failed
 *     (readback.in, occurrence 1): ERROR: Unknown command: bogus_command 1 2";
 *   - FAILSCORE_GRADING ON, the PC_FAILPROBE S reading `press`: the probe
 *     fails as a failed run does, so the evaluation is aborted (S and the
 *     PC_LAMMPS line's E failscored) and graded band D with S unavailable
 *     (the band top, TOTAL_LOSS/N 1.000000e+16): exit 1, "PC_LAMMPS task
 *     failed (probe.in, occurrence 1): reading PC_FAILPROBE --outvar S after
 *     the script: ERROR: Thermo keyword press", and no "--outvar is NaN"
 *     line (S is not a diagnostic);
 *   - FAILSCORE_GRADING ON, --varloss E reading `press` with --failvar viol
 *     1.0 (viol = 0.25): the failvar grades the failure, band B by
 *     sigma(0.25), TOTAL_LOSS/N 9.993000e+15, exit 1;
 *   - the same with viol = 1.0/(pe-pe): "PC_LAMMPS task failed
 *     (readback.in, occurrence 1): reading --failvar viol after the
 *     script: ERROR: ... Divide by 0", band B by the failed fraction (one of
 *     one script), TOTAL_LOSS/N 9.993750e+15, exit 1.
 * Before the fix: under THREAD tadah died of SIGSEGV in the read-back
 * sections (the shell's exit status 139); under PROCESS it exited 1 in the
 * --outvar section (the whole script failed, E included) and journalled
 * "pc-worker task failed (...): ERROR: Thermo keyword press ..." without
 * naming the variable, failed the graded --varloss without its failvar
 * (TOTAL_LOSS/N 9.993750e+15, the failed fraction); and neither mode
 * journalled the unknown command. Before the PC_FAILPROBE fix the probe's S
 * was taken for a diagnostic: "PC_LAMMPS --outvar is NaN (probe.in, ...)",
 * no "task failed" line, S NaN and E computed (the evaluation not
 * aborted). Before the harness fix a TADAH_PC_ISOLATION in the test's
 * environment overrode the config's PC_ISOLATION, so with
 * TADAH_PC_ISOLATION=THREAD the PROCESS runs ran THREAD (the journal check
 * of the mode fails). */
TEST_CASE("tadah hpo: a PC_LAMMPS read-back or script error ends the same "
          "way under PC_ISOLATION THREAD and PROCESS",
          "[pc_readback][tadah_cli][.tadah_cli_pc_readback]") {
    using tadah::hpo::FailBand;
    using tadah::hpo::graded_failscore;
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin)
        TADAH_SKIP("TADAH_BINARY is not set: it names the built tadah "
                   "executable");
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd("tadah_cli_pc_readback_");

    const std::string head =
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
        "    ALGO RANDOM\n    MAXEVAL 3\n    SEED 1234\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.0 4.0\n";
    const std::string press_text =
        "ERROR: Thermo keyword press in variable requires thermo to use/init "
        "press";
    struct Case {
        std::string script;
        std::string lammps_line;
        int exit_status;
        std::vector<std::string> journal_has;
        std::vector<std::string> journal_lacks;
        /* Every row's TOTAL_LOSS/N ("" unchecked). */
        std::string total_loss;
        /* outvar.tadah: every row's cell of each column. */
        std::vector<std::pair<std::string, std::string>> outvar_cells;
        std::vector<std::pair<std::string, std::string>> more_scripts;
    };
    Case c;
    std::string name;
    SECTION("a diagnostic --outvar that cannot be read") {
        name = "diagnostic";
        c = {ta_script("3.301", "variable E equal pe\n"
                                "variable Pend equal press\n"),
             "PC_LAMMPS --script readback.in --varloss E 0 1 --outvar Pend\n",
             0,
             {"PC_LAMMPS --outvar is NaN (readback.in, occurrence 1): "
              "reading --outvar Pend after the script: " + press_text},
             {"PC_LAMMPS task failed"},
             "",
             {{"Pend", "NaN"}},
             {}};
    }
    SECTION("a --varloss variable that cannot be read") {
        name = "varloss";
        c = {ta_script("3.301", "variable E equal press\n"),
             "PC_LAMMPS --script readback.in --varloss E 0 1\n", 1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): "
              "reading --varloss E after the script: " + press_text,
              "no evaluation produced a usable loss"},
             {},
             "1.000000e+16",
             {{"E", "1.000000e+16"}},
             {}};
    }
    SECTION("a LAMMPS error inside the script") {
        name = "script";
        c = {"clear\nunits metal\nbogus_command 1 2\nvariable E equal 1.0\n",
             "PC_LAMMPS --script readback.in --varloss E 0 1\n", 1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): "
              "ERROR: Unknown command: bogus_command 1 2",
              "no evaluation produced a usable loss"},
             {},
             "1.000000e+16",
             {{"E", "1.000000e+16"}},
             {}};
    }
    SECTION("a PC_FAILPROBE S that cannot be read") {
        name = "failprobe";
        c = {ta_script("3.301", "variable E equal pe\n"),
             "FAILSCORE_GRADING ON\n"
             "PC_FAILPROBE --script probe.in --outvar S --s0 0.5\n"
             "PC_LAMMPS --script readback.in --varloss E 0 1\n",
             1,
             {"PC_LAMMPS task failed (probe.in, occurrence 1): reading "
              "PC_FAILPROBE --outvar S after the script: " + press_text,
              "no evaluation produced a usable loss"},
             {"--outvar is NaN"},
             sci6(graded_failscore(1e16, FailBand::D,
                                   std::numeric_limits<double>::infinity(),
                                   0.5)),
             {{"S", "1.000000e+16"}, {"E", "1.000000e+16"}},
             {{"probe.in", ta_script("3.301", "variable S equal press\n")}}};
    }
    SECTION("a graded --varloss failure with a readable --failvar") {
        name = "graded_failvar";
        c = {ta_script("3.301", "variable viol equal 0.25\n"
                                "variable E equal press\n"),
             "FAILSCORE_GRADING ON\n"
             "PC_LAMMPS --script readback.in --varloss E 0 1 "
             "--failvar viol 1.0\n",
             1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): "
              "reading --varloss E after the script: " + press_text},
             {},
             sci6(graded_failscore(1e16, FailBand::B, 0.25, 1.0)),
             {{"E", "1.000000e+16"}},
             {}};
    }
    SECTION("a graded --varloss failure whose --failvar cannot be read") {
        name = "graded_failvar_unreadable";
        c = {ta_script("3.301", "variable viol equal 1.0/(pe-pe)\n"
                                "variable E equal press\n"),
             "FAILSCORE_GRADING ON\n"
             "PC_LAMMPS --script readback.in --varloss E 0 1 "
             "--failvar viol 1.0\n",
             1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): "
              "reading --failvar viol after the script: ",
              "Divide by 0 in variable formula"},
             {},
             sci6(graded_failscore(1e16, FailBand::B, 1.0, 1.0)),
             {{"E", "1.000000e+16"}},
             {}};
    }
    std::vector<fs::path> dirs;
    for (const std::string mode : {"THREAD", "PROCESS"}) {
        INFO(name << ", PC_ISOLATION " << mode);
        const fs::path dir = sd.dir / (name + "_" + mode);
        const int rc = run_tadah_hpo(bin, dir, mode, head + c.lammps_line,
                                     c.script, c.more_scripts);
        INFO(read_text(dir / "tadah.stderr"));
        const std::string journal = read_text(dir / "hpo_run.log");
        INFO(journal);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == c.exit_status);
        CHECK_THAT(journal,
                   Catch::Contains("PC_ISOLATION: " + mode +
                                   (mode == "THREAD"
                                        ? " (deprecated fallback)"
                                        : " -- pc_lammps tasks execute in")));
        for (const auto& needle : c.journal_has)
            CHECK_THAT(journal, Catch::Contains(needle));
        for (const auto& needle : c.journal_lacks)
            CHECK_THAT(journal, !Catch::Contains(needle));
        const Table outv = read_table(dir / "outvar.tadah");
        CHECK(outv.rows.size() == 3);
        check_header_shaped(outv, "outvar.tadah");
        for (const auto& cell : c.outvar_cells) {
            INFO("outvar.tadah column " << cell.first);
            const std::size_t col = column(outv, cell.first);
            REQUIRE(col < outv.header.size());
            for (const auto& row : outv.rows)
                if (row.size() == outv.header.size())
                    CHECK(row[col] == cell.second);
        }
        if (!c.total_loss.empty()) {
            const Table loss = read_table(dir / "loss.tadah");
            REQUIRE_FALSE(loss.header.empty());
            CHECK(loss.header.back() == "TOTAL_LOSS/N");
            CHECK(loss.rows.size() == 3);
            check_header_shaped(loss, "loss.tadah");
            for (const auto& row : loss.rows)
                if (!row.empty()) CHECK(row.back() == c.total_loss);
        }
        dirs.push_back(dir);
    }
    for (const char* f : {"loss.tadah", "params.tadah", "stats.tadah",
                          "best_loss.tadah", "best_params.tadah",
                          "best_stats.tadah"}) {
        INFO(name << ": " << f);
        CHECK_FALSE(read_text(dirs[0] / f).empty());
        CHECK(read_text(dirs[0] / f) == read_text(dirs[1] / f));
    }
    for (const char* f : {"outvar.tadah", "best_outvar.tadah"}) {
        INFO(name << ": " << f << " without IT-TIME/ms");
        CHECK(lines_without(dirs[0] / f, 1) == lines_without(dirs[1] / f, 1));
    }
}

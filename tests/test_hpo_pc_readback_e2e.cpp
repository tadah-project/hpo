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
//     NaN (`nan` in loss/stats.tadah, `NaN` in outvar.tadah). A hot value
//     LAMMPS refuses to inject before a script fails that script, with a
//     journal line too (a cold script is injected the hot values it reads,
//     and is not run when one was not produced), and so does a --varloss, --hotvar or --failvar or
//     the PC_FAILPROBE S with no (finite, for a failvar) number for its
//     value after the script, text after a number included (a diagnostic
//     is NaN with a journal line); a script that cannot be
//     rendered fails with a journal line; and a failed simulation scores
//     its evaluation FAILSCORE whatever loss its variables are measured
//     with (L1, TUKEY). Hidden
//     because, before the fix, these cases ended their process with
//     SIGSEGV, and because the HPO_Host statics are shared by the cases of
//     one process.
//   - [hotvar_e2e] (hidden; CTest entry hpo_hotvar_e2e, label e2e): two
//     runs in one process, as a program's two hpo_run() calls, each an
//     HPO_Host started with prepare_best_publication(): the second run
//     starts each --hotvar from its declared initial value, not from the
//     value the first run's last evaluation handed on.
//   - [kernel_line_e2e] (hidden; CTest entry hpo_kernel_line_e2e, label
//     e2e): two runs in one process, as a program's two hpo_run() calls on
//     the context TaskFactory makes of `tadah hpo -c ... -v 1`, under
//     PC_ISOLATION THREAD and PROCESS: each prints the settings line of its
//     own -c model's kernel once.
//   - [.tadah_cli_pc_readback] (CTest entry hpo_tadah_cli_pc_readback): the
//     built tadah under PC_ISOLATION THREAD and PROCESS ends the same way,
//     with the same rows and, under FAILSCORE_GRADING, the same graded
//     losses (a --varloss failure graded by its failvar, an unreadable
//     PC_FAILPROBE S). It needs TADAH_BINARY (the entry sets it in a
//     build with the tadah target) and is Skipped without it; its PROCESS
//     half needs socketpair, like the pc pool tests.
//   - [.tadah_cli_pc_lammps_local_search] (CTest entry
//     hpo_tadah_cli_pc_lammps_local_search, the same harness): a PC_LAMMPS
//     run with an NLopt MLSL or AUGLAG whose INNER search is derivative-free
//     starts and ends normally.
//   - [.tadah_cli_pc_lammps_verbose] (CTest entry
//     hpo_tadah_cli_pc_lammps_verbose, the same harness): the potential a
//     PC_LAMMPS script runs follows the run's VERBOSE (-v 0: a kernel
//     model prints no line when it is built for the script; the THREAD
//     leg shows it, a PROCESS worker printing no kernel line at all).
//   - [.tadah_cli_kernel_line_once] (CTest entry
//     hpo_tadah_cli_kernel_line_once, the same harness): at -v 1 a kernel's
//     settings line is printed once per run, not once per model the run
//     builds, under PC_ISOLATION THREAD and PROCESS.
//   - [.tadah_cli_pc_deadline_line] (CTest entry
//     hpo_tadah_cli_pc_deadline_line, the same harness): the startup line
//     on the per-script limit names the hard per-worker-task deadline of
//     PC_ISOLATION PROCESS (PC_WORKER_TIMEOUT, derived from
//     PC_SCRIPT_TIMEOUT, or none) and of THREAD (n/a).

#include "pc_process_workers.h"  // before catch2/catch.hpp
#include "tadah_test_skip.hpp"  // before catch2/catch.hpp

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <dlib/matrix.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
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

/* The column of @p t headed @p name (header.size() when none is). */
std::size_t column(const Table& t, const std::string& name) {
    std::size_t c = 0;
    while (c < t.header.size() && t.header[c] != name) ++c;
    return c;
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
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);
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
                             std::size_t n_val_datasets = 1,
                             const std::string& head = kHpoHead) {
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
    std::ofstream(hpo) << head << target_tail;
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

/* Pins: a failed PC_LAMMPS task (an unreadable --varloss, E = press) at the
 * default FAILSCORE, the largest double, under LOSS L2 fails its evaluation
 * as a failure, not as a NaN/Inf clamping: FAILSCORE (OFF), each evaluation
 * counted under "Unusable losses" and none under "NaN/Inf clampings", no
 * "evaluation produced NaN/Inf" WARNING; under FAILSCORE_GRADING ON
 * (FAILSCORE 1e200, whose square overflows as well) the graded band-B loss,
 * counted under "Graded failures" only.
 * Before the fix: the L2 of the --failscore value, DBL_MAX^2 = inf, made
 * the mean non-finite and assemble_loss set was_nan_clamped before the
 * failed task made the evaluation FAILSCORE; the census counted it as a
 * NaN/Inf clamping ("NaN/Inf clampings : 2 (100.0%)", "[!] 100.0% of
 * evaluations produced a non-finite loss that was clamped.") with an
 * "evaluation produced NaN/Inf -- loss clamped to FAIL_SCORE" WARNING and
 * bound hints for each, and under ON in both counters, the NaN one and
 * "Graded failures : 2", which DIAGNOSTICS calls disjoint. (The pins above
 * run under FAILSCORE 1e16, whose square is finite.) */
TEST_CASE("a failed PC_LAMMPS task at the default FAILSCORE is a failed "
          "evaluation, not a NaN/Inf clamping", "[pc_readback_e2e][.]") {
    using tadah::hpo::FailBand;
    using tadah::hpo::graded_failscore;
    tadah::core::PeriodicTable::initialize();
    for (const bool graded : {false, true}) {
        INFO((graded ? "FAILSCORE_GRADING ON, FAILSCORE 1e200"
                     : "FAILSCORE_GRADING OFF, the default FAILSCORE"));
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_default_fs_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal press\n");
        Journal j;
        const std::string head =
            std::string("OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
                        "ENDOPTIMIZER\nLOSS L2\nQFLUSH 1\n") +
            (graded ? "FAILSCORE 1e200\nFAILSCORE_GRADING ON\n" : "") +
            "OPTIM CGRID2B (1) 0.0 4.0\n";
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() + " --varloss E 0 1\n",
            {0.5, 0.6}, 1, head);
        const std::string journal = j.finish();
        INFO(journal);
        const double expected =
            graded ? graded_failscore(1e200, FailBand::B, 1.0, 1.0)
                   : std::numeric_limits<double>::max();
        for (const double l : losses) CHECK(l == expected);
        CHECK(journal.find("PC_LAMMPS task failed (") != std::string::npos);
        CHECK(journal.find("evaluation produced NaN/Inf") == std::string::npos);
        CHECK(journal.find("[ok] No NaN/Inf clampings were recorded.") !=
              std::string::npos);
        CHECK_THAT(journal, Catch::Contains("Unusable losses         : 2"));
        CHECK_THAT(journal,
                   Catch::Contains(std::string("Graded failures         : ") +
                                   (graded ? "2" : "0")));
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

/* Pins: a hot value LAMMPS refuses to inject into a cold script fails that
 * script and with it the evaluation: the cold line's `--invar hot_a0 3.3`
 * makes hot_a0 an index-style variable when its instance is created, so
 * the injected `variable hot_a0 equal 3.25` (the hot script's value) is
 * refused. Each evaluation returns FAILSCORE, the hot script's E is
 * computed, the cold script's E2 failscored, and the journal says, for the
 * occurrences 1 and 2, "PC_LAMMPS task failed (cold.in, occurrence n):
 * injecting `variable hot_a0 equal 3.25` before the script: ERROR: Cannot
 * redefine variable hot_a0 with a different style".
 * Before the fix: the injection's error was swallowed and the cold script
 * ran on its own value: E2 = 3.3 in every row, a feasible loss, no journal
 * line. */
TEST_CASE("a hot value LAMMPS refuses to inject fails the evaluation",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_inject_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal 3.25\n");
    std::ofstream(sd.dir / "cold.in")
        << "clear\nunits metal\nvariable E2 equal ${hot_a0}\n";
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
            " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301\n"
        "PC_LAMMPS --script " + (sd.dir / "cold.in").string() +
            " --varloss E2 0 1 --invar hot_a0 3.3\n",
        {0.5, 0.6});
    const std::string journal = j.finish();
    INFO(journal);
    for (const double l : losses) CHECK(l == 1e16);
    const Table outv = read_table("outvar.tadah");
    CHECK(outv.header == std::vector<std::string>{
              "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E", "E2"});
    REQUIRE(outv.rows.size() == 2);
    check_header_shaped(outv, "outvar.tadah");
    for (const auto& row : outv.rows)
        if (row.size() == 6) {
            CHECK(row[4] != "1.000000e+16");
            CHECK(row[5] == "1.000000e+16");
        }
    for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
        INFO(n);
        CHECK_THAT(line_with(journal, std::string("cold.in, ") + n +
                                          ": injecting `variable hot_a0 "
                                          "equal 3.25` before the script: "),
                   Catch::Contains("PC_LAMMPS task failed (") &&
                       Catch::Contains("ERROR: Cannot redefine variable "
                                       "hot_a0 with a different style"));
    }
    CHECK(journal.find("eval-thread caught throw") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: a cold script is injected the hot values it reads (`${name}` or
 * `v_name` in its text), each as `variable <name> equal <value>`, and no
 * other: with two hot scripts (hot_a0 = 3.25 from hot.in, hot_b = 1.0 from
 * hotb.in), colda.in, which reads only ${hot_a0} and has `--invar hot_b 7`
 * on its line (hot_b index style in its instance), runs: E2a = 3.25; and
 * coldab.in, which reads both, gets both: E2b = 4.25. Two evaluations,
 * feasible losses, no "PC_LAMMPS task failed" line.
 * Before the fix: every hot value went to every cold script that read any,
 * so colda.in got `variable hot_b equal 1` as well, which LAMMPS refused:
 * "PC_LAMMPS task failed (colda.in, occurrence 1): injecting `variable
 * hot_b equal 1` before the script: ERROR: Cannot redefine variable hot_b
 * with a different style", every evaluation FAILSCORE. */
TEST_CASE("a cold script is injected the hot values it reads",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_inject_reads_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal 3.25\n");
    std::ofstream(sd.dir / "hotb.in")
        << ta_script("3.301", "variable Eb equal pe\n"
                              "variable hot_b equal 1.0\n");
    std::ofstream(sd.dir / "colda.in")
        << "clear\nunits metal\nvariable E2a equal ${hot_a0}\n";
    std::ofstream(sd.dir / "coldab.in")
        << "clear\nunits metal\nvariable E2b equal v_hot_a0+${hot_b}\n";
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
            " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301\n"
        "PC_LAMMPS --script " + (sd.dir / "hotb.in").string() +
            " --varloss Eb 0 1 --hotvar hot_b 0.5 1.5 1.0\n"
        "PC_LAMMPS --script " + (sd.dir / "colda.in").string() +
            " --varloss E2a 0 1 --invar hot_b 7\n"
        "PC_LAMMPS --script " + (sd.dir / "coldab.in").string() +
            " --varloss E2b 0 1\n",
        {0.5, 0.6});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(losses.size() == 2);
    for (const double l : losses)
        CHECK(tadah::hpo::loss_is_feasible(l, 1e16));
    const Table outv = read_table("outvar.tadah");
    REQUIRE(outv.rows.size() == 2);
    check_header_shaped(outv, "outvar.tadah");
    const std::size_t ca = column(outv, "E2a"), cab = column(outv, "E2b");
    REQUIRE(ca < outv.header.size());
    REQUIRE(cab < outv.header.size());
    for (const auto& row : outv.rows)
        if (row.size() == outv.header.size()) {
            CHECK(row[ca] == "3.250000e+00");
            CHECK(row[cab] == "4.250000e+00");
        }
    CHECK(journal.find("PC_LAMMPS task failed") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: a cold script whose hot value was not produced -- its hot script
 * could not start: the hot line's `-log <dir>/hot.log` names, as the
 * worker's hot.log.t<slot>, a directory, so LAMMPS cannot create the hot
 * script's instance -- is not run: the evaluation scores FAILSCORE, E and
 * E2 are failscored, the journal has "PC_LAMMPS task failed (hot.in,
 * occurrence n): creating the LAMMPS instance: " for the occurrences 1 and
 * 2 and no line about cold.in.
 * Before the fix: the cold script ran without the value (here, in a fresh
 * instance, it failed on the undefined ${hot_a0}: "PC_LAMMPS task failed
 * (cold.in, occurrence 2): ERROR on proc 0: Substitution for illegal
 * variable hot_a0", and occurrence 4; in an instance an earlier evaluation
 * had injected, it would have run on that evaluation's value), so the hot
 * script's second failure was the run's occurrence 3, not journalled. */
TEST_CASE("a cold script whose hot value was not produced is not run",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_inject_missing_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal 3.25\n");
    std::ofstream(sd.dir / "cold.in")
        << "clear\nunits metal\nvariable E2 equal ${hot_a0}\n";
    /* The startup check opens hot.log itself; the workers append
     * .t<slot>. */
    for (int t = 0; t < 256; ++t)
        fs::create_directories(sd.dir / ("hot.log.t" + std::to_string(t)));
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
            " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301 -log " +
            (sd.dir / "hot.log").string() + "\n"
        "PC_LAMMPS --script " + (sd.dir / "cold.in").string() +
            " --varloss E2 0 1\n",
        {0.5, 0.6});
    const std::string journal = j.finish();
    INFO(journal);
    for (const double l : losses) CHECK(l == 1e16);
    const Table outv = read_table("outvar.tadah");
    REQUIRE(outv.rows.size() == 2);
    for (const auto& row : outv.rows) {
        REQUIRE(row.size() == 6);
        CHECK(row[4] == "1.000000e+16");
        CHECK(row[5] == "1.000000e+16");
    }
    for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
        INFO(n);
        CHECK(line_with(journal, std::string("hot.in, ") + n +
                                     ": creating the LAMMPS instance: ")
                  .find("PC_LAMMPS task failed (") != std::string::npos);
    }
    CHECK(journal.find("cold.in, occurrence") == std::string::npos);
    Host::reset_globals_for_test();
}

namespace {

/* The tail that leaves the variable @p name without a number after the
 * script: deleted (@p as_string false) or redefined as the string 'abc',
 * under a condition the startup check of the script cannot evaluate
 * (`$(step)` needs the run), so that it takes the variable as defined. */
std::string spoil(const std::string& name, bool as_string) {
    return "if \"$(step) == 0\" then \"variable " + name + " delete\"" +
           (as_string ? " \"variable " + name + " string abc\"" : "") + "\n";
}

/* What the journal says of such a variable after "after the script: ". */
std::string spoiled_reason(bool as_string) {
    return as_string ? "its value 'abc' is not a number"
                     : "no value (the variable is not defined, or has no "
                       "single value)";
}

}  // namespace

/* Pins: a --varloss variable with no number for its value after the script
 * -- deleted at run time, or redefined as the string 'abc' -- fails the
 * evaluation as one that cannot be read does: FAILSCORE, E and F failscored
 * in outvar.tadah, and the journal line "PC_LAMMPS task failed
 * (readback.in, occurrence n): reading --varloss F after the script: no
 * value (the variable is not defined, or has no single value)" (or "...:
 * its value 'abc' is not a number"). The deleted F reads the same in the
 * second evaluation (occurrence 2); the string F stays in the cached LAMMPS
 * instance (`clear` keeps variables), so the second run of the script fails
 * at its `variable F equal pe+1` ("ERROR: Cannot redefine variable F with
 * a different style", journalled as a script error).
 * Before the fix: F's FAIL_SCORE term was averaged with the evaluation's
 * other terms, L(ERMSE) and L(E): TOTAL_LOSS/N 3.333333e+15, below 0.999 *
 * FAILSCORE, so the evaluation counted as feasible; E was kept, F NaN, and
 * nothing was journalled. */
TEST_CASE("a --varloss variable with no number for its value fails the "
          "evaluation", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    for (const bool as_string : {false, true}) {
        INFO("F " << (as_string ? "redefined as a string" : "deleted"));
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_novalue_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable F equal pe+1\n" +
                                      spoil("F", as_string));
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() +
                " --varloss E 0 1 --varloss F 0 1\n",
            {0.5, 0.6});
        const std::string journal = j.finish();
        INFO(journal);
        for (const double l : losses) CHECK(l == 1e16);
        const Table outv = read_table("outvar.tadah");
        CHECK(outv.header == std::vector<std::string>{
                  "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "E", "F"});
        REQUIRE(outv.rows.size() == 2);
        check_header_shaped(outv, "outvar.tadah");
        for (const auto& row : outv.rows)
            if (row.size() == 6) {
                CHECK(row[4] == "1.000000e+16");
                CHECK(row[5] == "1.000000e+16");
            }
        CHECK(line_with(journal, "readback.in, occurrence 1): reading "
                                 "--varloss F after the script: " +
                                     spoiled_reason(as_string))
                  .find("PC_LAMMPS task failed (") != std::string::npos);
        /* The deleted F is defined afresh by the next run of the script;
         * the string F stays in the cached instance (`clear` keeps
         * variables), whose next run fails at `variable F equal pe+1`. */
        CHECK(line_with(journal,
                        as_string
                            ? "readback.in, occurrence 2): ERROR: Cannot "
                              "redefine variable F with a different style"
                            : "readback.in, occurrence 2): reading --varloss "
                              "F after the script: " +
                                  spoiled_reason(false))
                  .find("PC_LAMMPS task failed (") != std::string::npos);
        CHECK(journal.find("[!] No evaluation produced a usable loss.") !=
              std::string::npos);
        Host::reset_globals_for_test();
    }
}

/* Pins: a --hotvar with no number for its value after the hot script (its
 * hot_a0 redefined as the string 'abc') fails the hot script and with it
 * the evaluation: FAILSCORE, E and E2 failscored, the journal line
 * "PC_LAMMPS task failed (hot.in, occurrence 1): reading --hotvar hot_a0
 * after the script: its value 'abc' is not a number", and the cold script,
 * which needs the hot value, not run (no line about it).
 * Before the fix: the value was skipped -- no range check, no warm start,
 * no hot value for the cold script, which ran without it (here it failed on
 * the undefined ${hot_a0}, journalled as its own script error; a cold
 * script that had seen the value before would have run on the previous
 * evaluation's) -- and the journal had no line naming hot_a0. */
TEST_CASE("a --hotvar with no number for its value fails the evaluation",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_novalue_hot_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal 3.25\n" +
                                      spoil("hot_a0", true));
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
    REQUIRE(outv.rows.size() == 1);
    REQUIRE(outv.rows[0].size() == 6);
    CHECK(outv.rows[0][4] == "1.000000e+16");
    CHECK(outv.rows[0][5] == "1.000000e+16");
    CHECK_THAT(line_with(journal, "hot.in, occurrence 1): reading --hotvar "
                                  "hot_a0 after the script: "),
               Catch::Contains("PC_LAMMPS task failed (") &&
                   Catch::Contains(spoiled_reason(true)));
    CHECK(journal.find("cold.in, occurrence") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: a --failvar with no finite number for its value after the script
 * -- deleted at run time, or exp(1000), which LAMMPS evaluates to inf --
 * fails the evaluation, under FAILSCORE_GRADING OFF and ON alike: the
 * journal line "PC_LAMMPS task failed (readback.in, occurrence 1): reading
 * --failvar viol after the script: no value (the variable is not defined,
 * or has no single value)" (or "...: its value 'inf' is not a finite
 * number"), E failscored, and the loss FAILSCORE (OFF) or, graded by the
 * failed fraction of the one script since the failvar cannot grade, band
 * B's value for 1 (ON).
 * Before the fix: under OFF the failvar was not looked at, under ON its
 * empty or non-finite value counted as no violation: the evaluation was
 * feasible, E kept, and nothing was journalled. */
TEST_CASE("a --failvar with no finite number for its value fails the "
          "evaluation", "[pc_readback_e2e][.]") {
    using tadah::hpo::FailBand;
    using tadah::hpo::graded_failscore;
    tadah::core::PeriodicTable::initialize();
    struct Case { std::string tail, reason; };
    for (const bool graded : {false, true})
    for (const Case& c : {Case{"variable viol equal 0.0\n" +
                                   spoil("viol", false),
                               spoiled_reason(false)},
                          Case{"variable viol equal exp(1000)\n",
                               "its value 'inf' is not a finite number"}}) {
        INFO((graded ? "FAILSCORE_GRADING ON" : "FAILSCORE_GRADING OFF")
             << ", " << c.reason);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_novalue_fail_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n" + c.tail);
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            std::string(graded ? "FAILSCORE_GRADING ON\n" : "") +
                "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() +
                " --varloss E 0 1 --failvar viol 1.0\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == (graded ? graded_failscore(1e16, FailBand::B,
                                                       1.0, 1.0)
                                   : 1e16));
        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.rows.size() == 1);
        REQUIRE(outv.rows[0].size() == 5);
        CHECK(outv.rows[0][4] == "1.000000e+16");
        CHECK_THAT(line_with(journal, "readback.in, occurrence 1): reading "
                                      "--failvar viol after the script: "),
                   Catch::Contains("PC_LAMMPS task failed (") &&
                       Catch::Contains(c.reason));
        Host::reset_globals_for_test();
    }
}

/* Pins: a diagnostic (--outvar only) with no number for its value after
 * the script (deleted at run time) is NaN in outvar.tadah and the
 * evaluation keeps its loss, as for one that cannot be read, and the
 * journal says so for the occurrences 1 and 2: "PC_LAMMPS --outvar is NaN
 * (readback.in, occurrence n): reading --outvar P after the script: no
 * value (the variable is not defined, or has no single value)".
 * Before the fix: NaN with nothing in the journal (the line checks
 * fail). */
TEST_CASE("a diagnostic --outvar with no number for its value is NaN with a "
          "journal line", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_novalue_diag_");
    std::ofstream(sd.dir / "readback.in")
        << ta_script("3.301", "variable E equal pe\nvariable P equal pe\n" +
                                  spoil("P", false));
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\nPC_LAMMPS --script " +
            (sd.dir / "readback.in").string() +
            " --varloss E 0 1 --outvar P\n",
        {0.5, 0.6});
    const std::string journal = j.finish();
    INFO(journal);
    for (const double l : losses)
        CHECK(tadah::hpo::loss_is_feasible(l, 1e16));
    const Table outv = read_table("outvar.tadah");
    REQUIRE(outv.rows.size() == 2);
    for (const auto& row : outv.rows) {
        REQUIRE(row.size() == 6);
        CHECK(row[4] != "1.000000e+16");
        CHECK(row[5] == "NaN");
    }
    for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
        INFO(n);
        CHECK(line_with(journal, std::string("readback.in, ") + n +
                                     ": reading --outvar P after the "
                                     "script: " + spoiled_reason(false))
                  .find("PC_LAMMPS --outvar is NaN (") != std::string::npos);
    }
    CHECK(journal.find("PC_LAMMPS task failed") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: the S of the PC_FAILPROBE line (its first --outvar) with no number
 * for its value after the probe's script -- deleted at run time, or
 * redefined as the string 'abc' -- is a failed probe, as one that cannot be
 * read is, under FAILSCORE_GRADING OFF and ON alike: the journal line
 * "PC_LAMMPS task failed (probe.in, occurrence 1): reading PC_FAILPROBE
 * --outvar S after the script: no value (the variable is not defined, or
 * has no single value)" (or "...: its value 'abc' is not a number"), no
 * "--outvar is NaN" line (S is not a diagnostic), the evaluation aborted
 * (S and the PC_LAMMPS line's E failscored in outvar.tadah) and its loss
 * FAILSCORE (OFF) or band D's top with S unavailable (ON, --s0 0.5).
 * Before the fix: S "" or 'abc' was taken for a number nobody checked: the
 * probe's task counted as completed, E kept its value and nothing was
 * journalled; under OFF the evaluation was feasible, under ON band D was
 * recorded for the NaN S (the same loss) without aborting the
 * evaluation. */
TEST_CASE("a PC_FAILPROBE S with no number for its value fails the probe",
          "[pc_readback_e2e][.]") {
    using tadah::hpo::FailBand;
    using tadah::hpo::graded_failscore;
    tadah::core::PeriodicTable::initialize();
    for (const bool graded : {false, true})
    for (const bool as_string : {false, true}) {
        INFO((graded ? "FAILSCORE_GRADING ON" : "FAILSCORE_GRADING OFF")
             << ", S " << (as_string ? "redefined as a string" : "deleted"));
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_novalue_probe_");
        std::ofstream(sd.dir / "probe.in")
            << ta_script("3.301", "variable S equal 0.1\n" +
                                      spoil("S", as_string));
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            std::string(graded ? "FAILSCORE_GRADING ON\n" : "") +
                "PC_ERMSE 0 1\nPC_FAILPROBE --script " +
                (sd.dir / "probe.in").string() +
                " --outvar S --s0 0.5\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() + " --varloss E 0 1\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] ==
              (graded ? graded_failscore(
                            1e16, FailBand::D,
                            std::numeric_limits<double>::infinity(), 0.5)
                      : 1e16));
        const Table outv = read_table("outvar.tadah");
        CHECK(outv.header == std::vector<std::string>{
                  "#ITERATION", "IT-TIME/ms", "SLOT", "ERMSE", "S", "E"});
        REQUIRE(outv.rows.size() == 1);
        REQUIRE(outv.rows[0].size() == 6);
        CHECK(outv.rows[0][4] == "1.000000e+16");
        CHECK(outv.rows[0][5] == "1.000000e+16");
        CHECK_THAT(line_with(journal, "probe.in, occurrence 1): reading "
                                      "PC_FAILPROBE --outvar S after the "
                                      "script: "),
                   Catch::Contains("PC_LAMMPS task failed (") &&
                       Catch::Contains(spoiled_reason(as_string)));
        CHECK(journal.find("--outvar is NaN") == std::string::npos);
        Host::reset_globals_for_test();
    }
}

namespace {

/* The tail that redefines the variable @p name as the string @p text, under
 * the condition of spoil(). */
std::string respell(const std::string& name, const std::string& text) {
    return "if \"$(step) == 0\" then \"variable " + name + " delete\" "
           "\"variable " + name + " string " + text + "\"\n";
}

}  // namespace

/* Pins: a --hotvar or --failvar value is a number only when the whole value
 * is one -- the parse of every value read back --, so one with text after
 * its number fails the evaluation as text does:
 *   - a hot script whose hot_a0 is redefined as the string '3.25abc':
 *     FAILSCORE, E and E2 failscored, "PC_LAMMPS task failed (hot.in,
 *     occurrence 1): reading --hotvar hot_a0 after the script: its value
 *     '3.25abc' is not a number", and the cold script not run (no line
 *     about it);
 *   - a --failvar viol redefined as the string '0.5abc', under
 *     FAILSCORE_GRADING OFF and ON: "... reading --failvar viol after the
 *     script: its value '0.5abc' is not a number", E failscored, the loss
 *     FAILSCORE (OFF) or, graded by the failed fraction of the one script
 *     since the failvar cannot grade, band B's value for 1 (ON).
 * Before the fix: hot and fail values were parsed by their leading number
 * (core::to_double): hot_a0 = 3.25, in its window, warm-started the next
 * evaluation and ran the cold script (a feasible loss, E2 computed); under
 * OFF viol was not looked at (a feasible loss, E kept); under ON viol = 0.5
 * graded band B by 0.5, not by the failed fraction; nothing was
 * journalled. */
TEST_CASE("a --hotvar or --failvar with text after its number fails the "
          "evaluation", "[pc_readback_e2e][.]") {
    using tadah::hpo::FailBand;
    using tadah::hpo::graded_failscore;
    tadah::core::PeriodicTable::initialize();
    {
        INFO("--hotvar hot_a0 '3.25abc'");
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_trailing_hot_");
        std::ofstream(sd.dir / "hot.in")
            << ta_script("${hot_a0}", "variable E equal pe\n"
                                      "variable hot_a0 equal 3.25\n" +
                                          respell("hot_a0", "3.25abc"));
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
        REQUIRE(outv.rows.size() == 1);
        REQUIRE(outv.rows[0].size() == 6);
        CHECK(outv.rows[0][4] == "1.000000e+16");
        CHECK(outv.rows[0][5] == "1.000000e+16");
        CHECK_THAT(line_with(journal, "hot.in, occurrence 1): reading "
                                      "--hotvar hot_a0 after the script: "),
                   Catch::Contains("PC_LAMMPS task failed (") &&
                       Catch::Contains("its value '3.25abc' is not a "
                                       "number"));
        CHECK(journal.find("cold.in, occurrence") == std::string::npos);
        Host::reset_globals_for_test();
    }
    for (const bool graded : {false, true}) {
        INFO("--failvar viol '0.5abc', "
             << (graded ? "FAILSCORE_GRADING ON" : "FAILSCORE_GRADING OFF"));
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_trailing_fail_");
        std::ofstream(sd.dir / "readback.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable viol equal 0.0\n" +
                                      respell("viol", "0.5abc"));
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            std::string(graded ? "FAILSCORE_GRADING ON\n" : "") +
                "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                (sd.dir / "readback.in").string() +
                " --varloss E 0 1 --failvar viol 1.0\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == (graded ? graded_failscore(1e16, FailBand::B,
                                                       1.0, 1.0)
                                   : 1e16));
        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.rows.size() == 1);
        REQUIRE(outv.rows[0].size() == 5);
        CHECK(outv.rows[0][4] == "1.000000e+16");
        CHECK_THAT(line_with(journal, "readback.in, occurrence 1): reading "
                                      "--failvar viol after the script: "),
                   Catch::Contains("PC_LAMMPS task failed (") &&
                       Catch::Contains("its value '0.5abc' is not a "
                                       "number"));
        Host::reset_globals_for_test();
    }
}

/* Pins: the journal line of a task with several hot and fail values
 * without a number names every one, in the order the host judges them --
 * the hot values, then the fail values --, the first first, so it still
 * names the failure: a hot script whose hot_a0 is '3.25abc', hot_b 'xyz'
 * and --failvar viol '0.5abc' fails the evaluation (FAILSCORE, E and E2
 * failscored, the cold script not run) with "PC_LAMMPS task failed
 * (<dir>/hot.in, occurrence 1): reading --hotvar hot_a0 after the script:
 * its value '3.25abc' is not a number; reading --hotvar hot_b after the
 * script: its value 'xyz' is not a number; reading --failvar viol after
 * the script: its value '0.5abc' is not a number".
 * Before the fix: the line ended after hot_a0's clause (the judgement
 * stopped at the first variable at fault), so hot_b's and viol's clauses
 * are not found. */
TEST_CASE("every hot and fail value without a number is journalled, the "
          "first first", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_every_bad_var_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}", "variable E equal pe\n"
                                  "variable hot_a0 equal 3.25\n"
                                  "variable hot_b equal 1.5\n"
                                  "variable viol equal 0.0\n" +
                                      respell("hot_a0", "3.25abc") +
                                      respell("hot_b", "xyz") +
                                      respell("viol", "0.5abc"));
    std::ofstream(sd.dir / "cold.in")
        << ta_script("${hot_a0}", "variable E2 equal pe\n");
    Journal j;
    const auto losses = evaluate(
        sd.dir,
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
            " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301"
            " --hotvar hot_b 0 10 1.5 --failvar viol 1.0\n"
        "PC_LAMMPS --script " + (sd.dir / "cold.in").string() +
            " --varloss E2 0 1\n",
        {0.5});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(losses.size() == 1);
    CHECK(losses[0] == 1e16);
    const Table outv = read_table("outvar.tadah");
    REQUIRE(outv.rows.size() == 1);
    REQUIRE(outv.rows[0].size() == 6);
    CHECK(outv.rows[0][4] == "1.000000e+16");
    CHECK(outv.rows[0][5] == "1.000000e+16");
    const std::string line = line_with(journal, "hot.in, occurrence 1): ");
    CHECK_THAT(line,
               Catch::Contains("PC_LAMMPS task failed (") &&
               Catch::Contains("hot.in, occurrence 1): "
                               "reading --hotvar hot_a0 after the script: its "
                               "value '3.25abc' is not a number; reading "
                               "--hotvar hot_b after the script: its value "
                               "'xyz' is not a number; reading --failvar viol "
                               "after the script: its value '0.5abc' is not a "
                               "number"));
    CHECK(journal.find("cold.in, occurrence") == std::string::npos);
    Host::reset_globals_for_test();
}

/* Pins: the journal line of a task names every variable at fault whichever
 * side found it -- the LAMMPS error raised by reading one back (found as it
 * is read) or a value without a number (found as it is judged) --, in one
 * order: the line's variables (--varloss), then the --hotvar, then the
 * --failvar variables, each group in its line's order, joined with "; ":
 *   - --varloss E 'abc', --failvar F1 = 1.0/(pe-pe) (a divide by zero),
 *     --failvar F2 'xyz': "PC_LAMMPS task failed
 *     (<dir>/rb.in, occurrence 1): reading --varloss E after the script:
 *     its value 'abc' is not a number; reading --failvar F1 after the
 *     script: <LAMMPS divide by 0>; reading --failvar F2 after the script:
 *     its value 'xyz' is not a number";
 *   - a hot script whose --varloss E and --hotvar hot_a0 read `press` and
 *     whose --hotvar hot_b is 'xyz': "reading
 *     --varloss E after the script: <press error>; reading --hotvar hot_a0
 *     after the script: <press error>; reading --hotvar hot_b after the
 *     script: its value 'xyz' is not a number", the cold script not run;
 *   - --varloss E 'abc' then --varloss G reading `press`: E's clause first,
 *     then G's.
 * Each evaluation fails (FAILSCORE, its variables failscored).
 * Before the fix: a --hotvar or --failvar whose read raised a LAMMPS error
 * failed the task in the worker, which handed on no other value, and the
 * parent judged nothing else: the first two lines named F1 alone and
 * hot_a0 alone (E, F2 and hot_b missing); and the loss variables' read
 * errors came before their values without a number, so the third named G
 * first. The clause and order checks fail. */
TEST_CASE("every variable at fault in a task is journalled, the line's, then "
          "the hot, then the fail ones, whichever side found it",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    /* The journal entry that contains @p needle: from its "PC_LAMMPS task
     * failed (" to the next timestamped entry. A LAMMPS message carries a
     * line of its own ("Last input line: ..."), which the journal prints as
     * a continuation line of the entry. */
    auto entry = [](const std::string& journal, const std::string& needle) {
        const std::size_t at = journal.find(needle);
        if (at == std::string::npos) return std::string();
        const std::size_t head = journal.rfind("PC_LAMMPS task failed (", at);
        const std::size_t from = head == std::string::npos ? at : head;
        const std::size_t to = journal.find("\n  [", at);
        return journal.substr(from, to == std::string::npos ? std::string::npos
                                                           : to - from);
    };
    /* The clauses @p clauses, in this order, in the entry @p text. */
    auto check_clauses = [](const std::string& text,
                            const std::vector<std::string>& clauses) {
        INFO(text);
        CHECK_THAT(text, Catch::StartsWith("PC_LAMMPS task failed ("));
        std::size_t at = 0;
        for (const std::string& c : clauses) {
            INFO(c);
            const std::size_t p = text.find(c, at);
            CHECK(p != std::string::npos);
            if (p != std::string::npos) at = p + c.size();
        }
    };
    SECTION("a --varloss without a number and a --failvar read error") {
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_mixed_fail_");
        std::ofstream(sd.dir / "rb.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable F1 equal 1.0/(pe-pe)\n"
                                  "variable F2 equal 0.0\n" +
                                      respell("E", "abc") +
                                      respell("F2", "xyz"));
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\n"
            "PC_LAMMPS --script " + (sd.dir / "rb.in").string() +
                " --varloss E 0 1 --failvar F1 1.0 --failvar F2 1.0\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        check_clauses(entry(journal, "rb.in, occurrence 1): "),
                      {"rb.in, occurrence 1): reading --varloss E after the "
                       "script: its value 'abc' is not a number; ",
                       "reading --failvar F1 after the script: ",
                       kDivideError.lammps_text,
                       "; reading --failvar F2 after the script: its value "
                       "'xyz' is not a number"});
        Host::reset_globals_for_test();
    }
    SECTION("a --varloss and a --hotvar read error and a --hotvar without a "
            "number") {
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_mixed_hot_");
        std::ofstream(sd.dir / "hot.in")
            << ta_script("${hot_a0}", "variable E equal press\n"
                                      "variable hot_a0 equal press\n"
                                      "variable hot_b equal 1.5\n" +
                                          respell("hot_b", "xyz"));
        std::ofstream(sd.dir / "cold.in")
            << ta_script("${hot_a0}", "variable E2 equal pe\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\n"
            "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
                " --varloss E 0 1 --hotvar hot_a0 2.5 4.5 3.301"
                " --hotvar hot_b 0 10 1.5\n"
            "PC_LAMMPS --script " + (sd.dir / "cold.in").string() +
                " --varloss E2 0 1\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.rows.size() == 1);
        REQUIRE(outv.rows[0].size() == 6);
        CHECK(outv.rows[0][4] == "1.000000e+16");
        CHECK(outv.rows[0][5] == "1.000000e+16");
        check_clauses(entry(journal, "hot.in, occurrence 1): "),
                      {"hot.in, occurrence 1): reading --varloss E after the "
                       "script: ",
                       kPressError.lammps_text,
                       "; reading --hotvar hot_a0 after the script: ",
                       kPressError.lammps_text,
                       "; reading --hotvar hot_b after the script: its value "
                       "'xyz' is not a number"});
        CHECK(journal.find("cold.in, occurrence") == std::string::npos);
        Host::reset_globals_for_test();
    }
    SECTION("the line's variables in the line's order") {
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_mixed_loss_");
        std::ofstream(sd.dir / "rb.in")
            << ta_script("3.301", "variable E equal pe\n"
                                  "variable G equal press\n" +
                                      respell("E", "abc"));
        Journal j;
        const auto losses = evaluate(
            sd.dir,
            "PC_ERMSE 0 1\n"
            "PC_LAMMPS --script " + (sd.dir / "rb.in").string() +
                " --varloss E 0 1 --varloss G 0 1\n",
            {0.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        check_clauses(entry(journal, "rb.in, occurrence 1): "),
                      {"rb.in, occurrence 1): reading --varloss E after the "
                       "script: its value 'abc' is not a number; reading "
                       "--varloss G after the script: ",
                       kPressError.lammps_text});
        Host::reset_globals_for_test();
    }
}

namespace {

/* TMPDIR for the life of the object, restored (or unset) after. */
struct ScopedTmpdir {
    std::optional<std::string> old;
    explicit ScopedTmpdir(const std::string& value) {
        if (const char* v = std::getenv("TMPDIR")) old = std::string(v);
        ::setenv("TMPDIR", value.c_str(), 1);
    }
    ~ScopedTmpdir() {
        if (old) ::setenv("TMPDIR", old->c_str(), 1);
        else     ::unsetenv("TMPDIR");
    }
};

}  // namespace

/* Pins: a PC_LAMMPS script that cannot be rendered to its temporary file
 * (TMPDIR names a regular file during the evaluations) fails its task as
 * an infrastructure failure -- the evaluation's loss FAILSCORE, E
 * failscored -- and the journal says why, for the occurrences 1 and 2:
 * "PC_LAMMPS task failed (readback.in, occurrence n): rendering the script
 * to a temporary file: " and the reason.
 * Before the fix: the task carried the reason, but the parent's
 * CREATE_FAIL path returned before the journal line, so every evaluation
 * failed with nothing in the journal (the line checks fail). */
TEST_CASE("a PC_LAMMPS script that cannot be rendered is journalled",
          "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_unrendered_");
    /* A text no live rendering has: identical renderings are shared. */
    std::ofstream(sd.dir / "readback.in")
        << ta_script("3.301", "variable E equal pe\nvariable unrendered_" +
                                  std::to_string(::getpid()) + " equal 1\n");
    std::ofstream(sd.dir / "notadir") << "a file, not a directory\n";
    Journal j;
    std::vector<double> losses;
    {
        ScopedTmpdir tmpdir((sd.dir / "notadir").string());
        losses = evaluate(sd.dir,
                          "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                              (sd.dir / "readback.in").string() +
                              " --varloss E 0 1\n",
                          {0.5, 0.6});
    }
    const std::string journal = j.finish();
    INFO(journal);
    for (const double l : losses) CHECK(l == 1e16);
    const Table outv = read_table("outvar.tadah");
    REQUIRE(outv.rows.size() == 2);
    for (const auto& row : outv.rows) {
        REQUIRE(row.size() == 5);
        CHECK(row[4] == "1.000000e+16");
    }
    for (const char* n : {"occurrence 1)", "occurrence 2)"}) {
        INFO(n);
        CHECK(line_with(journal, std::string("readback.in, ") + n +
                                     ": rendering the script to a temporary "
                                     "file: ")
                  .find("PC_LAMMPS task failed (") != std::string::npos);
    }
    Host::reset_globals_for_test();
}

/* Pins: under FAILSCORE_GRADING OFF an evaluation with a failed PC_LAMMPS
 * simulation scores FAILSCORE whatever the loss its variables are measured
 * with: under L1 and TUKEY (--varloss overrides; L(ERMSE) stays L2), for
 *   - a --varloss F deleted at run time (no value after the script),
 *   - a LAMMPS error inside the script (`bogus_command 1 2`),
 *   - a script that cannot be rendered (TMPDIR names a regular file: an
 *     infrastructure failure, which does not abort the evaluation),
 *   - a hot value outside its window (hot_a0 = 5.0 outside [2.5, 4.5]),
 * every evaluation returns 1e16 and the journal says "[!] No evaluation
 * produced a usable loss."; the failed variables are --failscore (1e16) in
 * outvar.tadah, and under L1 their loss.tadah term is L1(1e16) =
 * 1.000000e+16 (the --failscore value goes through the line's loss, as
 * before). Under L2 the same cases scored 1e16 before the fix as well: the
 * squared --failscore value is past FAILSCORE (the cases above with
 * `LOSS L2`).
 * Before the fix: the evaluation's loss was the mean of its terms, the
 * failed line's loss of its --failscore value with L(ERMSE) (and L(E)): under
 * L1 6666666666666700 (F deleted) or 5000000000000050, under TUKEY, which is
 * bounded (a term of about 1 each), 33.99996 (F deleted) or 50.49994 --
 * below 0.999 * FAILSCORE, so each failed evaluation counted as feasible
 * ("[ok] Every evaluation produced a usable loss.") and could become the
 * best. */
TEST_CASE("a failed PC_LAMMPS simulation fails the evaluation whatever its "
          "variables' loss", "[pc_readback_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    struct Case {
        std::string what, script, options;
        bool unrendered;
    };
    const std::vector<Case> cases{
        {"a --varloss F deleted at run time",
         ta_script("3.301", "variable E equal pe\nvariable F equal pe+1\n" +
                                spoil("F", false)),
         " --varloss E 0 1 LOSS --varloss F 0 1 LOSS", false},
        {"a LAMMPS error inside the script",
         ta_script("3.301", "variable E equal pe\nbogus_command 1 2\n"),
         " --varloss E 0 1 LOSS", false},
        {"a script that cannot be rendered",
         ta_script("3.301", "variable E equal pe\nvariable unrendered_" +
                                std::to_string(::getpid()) + " equal 2\n"),
         " --varloss E 0 1 LOSS", true},
        {"a hot value outside its window",
         ta_script("3.301", "variable E equal pe\n"
                            "variable hot_a0 equal 5.0\n"),
         " --varloss E 0 1 LOSS --hotvar hot_a0 2.5 4.5 3.301", false}};
    for (const std::string loss : {"L1", "TUKEY"})
    for (const Case& c : cases) {
        INFO("--varloss ... " << loss << ": " << c.what);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_failed_loss_");
        std::ofstream(sd.dir / "readback.in") << c.script;
        std::ofstream(sd.dir / "notadir") << "a file, not a directory\n";
        std::string options = c.options;
        for (std::size_t p; (p = options.find("LOSS")) != std::string::npos;)
            options.replace(p, 4, loss);
        Journal j;
        std::vector<double> losses;
        {
            std::optional<ScopedTmpdir> tmpdir;
            if (c.unrendered) tmpdir.emplace((sd.dir / "notadir").string());
            losses = evaluate(sd.dir,
                              "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                                  (sd.dir / "readback.in").string() +
                                  options + "\n",
                              {0.5});
        }
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 1);
        CHECK(losses[0] == 1e16);
        CHECK(journal.find("[!] No evaluation produced a usable loss.") !=
              std::string::npos);
        const Table outv = read_table("outvar.tadah");
        REQUIRE(outv.rows.size() == 1);
        check_header_shaped(outv, "outvar.tadah");
        for (std::size_t col = 4; col < outv.rows[0].size(); ++col) {
            INFO("outvar.tadah column " << outv.header.at(col));
            CHECK(outv.rows[0][col] == "1.000000e+16");
        }
        if (loss == "L1") {
            const Table l = read_table("loss.tadah");
            const std::size_t col = column(l, "L(E)");
            REQUIRE(col < l.header.size());
            REQUIRE(l.rows.size() == 1);
            REQUIRE(l.rows[0].size() == l.header.size());
            CHECK(l.rows[0][col] == "1.000000e+16");
        }
        Host::reset_globals_for_test();
    }
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
 * = @p script and each of @p more_scripts (name, text), the environment
 * assignments @p env ("TMPDIR=notadir"), the -c configuration's model keys
 * @p model_keys (after its DBFILE line) and the further command-line
 * arguments @p cli_args ("-v 0"); returns the wait status. tadah runs without TADAH_PC_ISOLATION, which would override the
 * hpotarget's PC_ISOLATION (src/hpo.cpp) -- a developer where socketpair
 * is blocked is told to set it to THREAD -- so @p mode is the mode it runs
 * in, as each case checks in its journal. */
int run_tadah_hpo(
        const char* bin, const fs::path& dir, const std::string& mode,
        const std::string& target, const std::string& script,
        const std::vector<std::pair<std::string, std::string>>& more_scripts,
        const std::string& env = "",
        const std::string& model_keys = kTaModelKeys,
        const std::string& cli_args = "") {
    fs::create_directories(dir);
    write_minimal_ta_dbfile(dir / "train.db", 0.0);
    write_minimal_ta_dbfile(dir / "val.db", 0.07, -0.04);
    std::ofstream(dir / "config.val") << "DBFILE " << (dir / "val.db").string()
                                      << "\n";
    std::ofstream(dir / "config") << "DBFILE " << (dir / "train.db").string()
                                  << "\n" << model_keys;
    std::ofstream(dir / "readback.in") << script;
    for (const auto& s : more_scripts) std::ofstream(dir / s.first) << s.second;
    std::ofstream(dir / "config.hpo") << target << "PC_ISOLATION " << mode
                                      << "\n";
    /* An MPI build's tadah runs under the launcher of one rank
     * (TADAH_MPIEXEC, set by its CTest entry): started without it, Open
     * MPI makes its session directory under TMPDIR, and with the
     * unrendered section's TMPDIR -- a regular file -- Open MPI 5 says so
     * yet MPI_Init_thread returns MPI_SUCCESS, and tadah crashes at its
     * first communicator (SIGSEGV, exit status 139) before reading its
     * configuration. Under the launcher the session directory is the
     * launcher's, and TMPDIR is tadah's alone, as in the serial build. */
    const char* launcher = std::getenv("TADAH_MPIEXEC");
    const std::string cmd =
        "cd '" + dir.string() + "' && " +
        (launcher && *launcher ? std::string(launcher) + " " : std::string()) +
        "env -u TADAH_PC_ISOLATION " + env + " '" + bin +
        "' hpo -c config --hpotarget config.hpo --validation config.val" +
        (cli_args.empty() ? std::string() : " " + cli_args) +
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

}  // namespace

/* Pins, through the built tadah: a LAMMPS error while a PC_LAMMPS variable
 * is read back, one inside the script, and one injecting a hot value before
 * it, end each evaluation the same way
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
 *     (readback.in, occurrence 1): reading --varloss E after the script:
 *     ERROR: Thermo keyword press ...; reading --failvar viol after the
 *     script: ERROR: ... Divide by 0", band B by the failed fraction (one of
 *     one script), TOTAL_LOSS/N 9.993750e+15, exit 1;
 *   - a hot value LAMMPS refuses to inject into the cold script (its line's
 *     --invar hot_a0 3.3 makes the name index style): exit 1, "PC_LAMMPS
 *     task failed (cold.in, occurrence 1): injecting `variable hot_a0 equal
 *     3.25` before the script: ERROR: Cannot redefine variable hot_a0 with a
 *     different style", E2 failscored;
 *   - a --varloss F deleted at run time: exit 1, "PC_LAMMPS task failed
 *     (readback.in, occurrence 1): reading --varloss F after the script: no
 *     value (the variable is not defined, or has no single value)", E and F
 *     failscored, TOTAL_LOSS/N 1.000000e+16;
 *   - --varloss F 'abc', --hotvar hot_a0 reading `press` and --failvar viol
 *     = 1.0/(pe-pe), with a cold script that reads hot_a0: exit 1, one
 *     line naming the three in the line's order, "PC_LAMMPS task failed
 *     (readback.in, occurrence 1): reading --varloss F after the script:
 *     its value 'abc' is not a number; reading --hotvar hot_a0 after the
 *     script: ERROR: Thermo keyword press ...; reading --failvar viol after
 *     the script: ... Divide by 0 ...", E, F and E2 failscored, the cold
 *     script not run, TOTAL_LOSS/N 1.000000e+16;
 *   - the same LAMMPS error inside the script with the line's --varloss E
 *     measured by L1: a failed evaluation all the same, exit 1,
 *     TOTAL_LOSS/N 1.000000e+16, no "[ok] Every evaluation produced a
 *     usable loss";
 *   - FAILSCORE_GRADING OFF, a PC_FAILPROBE S deleted at run time: the
 *     startup warning says that S neither grades nor gates an evaluation
 *     and that a probe that fails fails its evaluation; exit 1, "PC_LAMMPS
 *     task failed (probe.in, occurrence 1): reading PC_FAILPROBE --outvar S
 *     after the script: no value (...)", S and E failscored, TOTAL_LOSS/N
 *     1.000000e+16;
 *   - a script that cannot be rendered (TMPDIR names a regular file in
 *     tadah's environment, and a PC_ISOLATION worker inherits it): exit 1,
 *     "PC_LAMMPS task failed (readback.in, occurrence 1): rendering the
 *     script to a temporary file: ", E failscored.
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
 * of the mode fails). Before the injection fix the refused injection was
 * dropped in both modes: exit 0, E2 = 3.3 (the --invar value) in every row,
 * no journal line. Before the no-value fix F's FAIL_SCORE term was averaged
 * with the others (TOTAL_LOSS/N 3.333333e+15, a feasible loss): exit 0, E
 * kept, F NaN, no journal line. Before the CREATE_FAIL fix the unrendered
 * script failed every evaluation with nothing in the journal in both
 * modes. Before the no-value fix a PC_FAILPROBE S with no value was not
 * checked: exit 0, feasible losses, S NaN, E kept, no line; and the startup
 * warning said "the probe runs as a plain diagnostic sim; no grading, no
 * gate." (a probe whose script failed aborted its evaluation then too).
 * Before the failed-evaluation fix the L1 section's evaluations scored the
 * mean of L2(ERMSE) and L1(1e16): TOTAL_LOSS/N 5.000000e+15, a feasible
 * loss, exit 0. Before the every-variable fix the mixed section's line
 * named hot_a0 and viol alone, in both modes, and the unreadable-failvar
 * section's viol alone, "reading --failvar viol after the script: ..."
 * (the worker failed the task on their read errors and handed on no other
 * value): their first needles are not found. */
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
        /* Environment assignments for tadah ("" none). */
        std::string env = "";
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
              "reading --varloss E after the script: " + press_text,
              "; reading --failvar viol after the script: ",
              "Divide by 0 in variable formula"},
             {},
             sci6(graded_failscore(1e16, FailBand::B, 1.0, 1.0)),
             {{"E", "1.000000e+16"}},
             {}};
    }
    SECTION("a hot value LAMMPS refuses to inject") {
        name = "inject";
        c = {ta_script("${hot_a0}", "variable E equal pe\n"
                                    "variable hot_a0 equal 3.25\n"),
             "PC_LAMMPS --script readback.in --varloss E 0 1 "
             "--hotvar hot_a0 2.5 4.5 3.301\n"
             "PC_LAMMPS --script cold.in --varloss E2 0 1 "
             "--invar hot_a0 3.3\n",
             1,
             {"PC_LAMMPS task failed (cold.in, occurrence 1): injecting "
              "`variable hot_a0 equal 3.25` before the script: ERROR: Cannot "
              "redefine variable hot_a0 with a different style",
              "no evaluation produced a usable loss"},
             {},
             "1.000000e+16",
             {{"E2", "1.000000e+16"}},
             {{"cold.in", "clear\nunits metal\nvariable E2 equal ${hot_a0}\n"}}};
    }
    SECTION("a --varloss variable with no value") {
        name = "novalue";
        c = {ta_script("3.301", "variable E equal pe\nvariable F equal pe+1\n" +
                                    spoil("F", false)),
             "PC_LAMMPS --script readback.in --varloss E 0 1 --varloss F 0 1\n",
             1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): reading "
              "--varloss F after the script: " + spoiled_reason(false),
              "no evaluation produced a usable loss"},
             {},
             "1.000000e+16",
             {{"E", "1.000000e+16"}, {"F", "1.000000e+16"}},
             {}};
    }
    SECTION("a --varloss, a --hotvar and a --failvar at fault together") {
        name = "mixed";
        c = {ta_script("3.301", "variable E equal pe\nvariable F equal pe+1\n"
                                "variable hot_a0 equal press\n"
                                "variable viol equal 1.0/(pe-pe)\n" +
                                    respell("F", "abc")),
             "PC_LAMMPS --script readback.in --varloss E 0 1 --varloss F 0 1 "
             "--hotvar hot_a0 2.5 4.5 3.301 --failvar viol 1.0\n"
             "PC_LAMMPS --script cold.in --varloss E2 0 1\n",
             1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): reading "
              "--varloss F after the script: its value 'abc' is not a "
              "number; reading --hotvar hot_a0 after the script: " +
                  press_text,
              "; reading --failvar viol after the script: ",
              "Divide by 0 in variable formula",
              "no evaluation produced a usable loss"},
             {"cold.in, occurrence"},
             "1.000000e+16",
             {{"E", "1.000000e+16"}, {"F", "1.000000e+16"},
              {"E2", "1.000000e+16"}},
             {{"cold.in", "clear\nunits metal\nvariable E2 equal ${hot_a0}\n"}}};
    }
    SECTION("a LAMMPS error inside the script under --varloss ... L1") {
        name = "script_l1";
        c = {"clear\nunits metal\nbogus_command 1 2\nvariable E equal 1.0\n",
             "PC_LAMMPS --script readback.in --varloss E 0 1 L1\n", 1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): "
              "ERROR: Unknown command: bogus_command 1 2",
              "no evaluation produced a usable loss"},
             {"[ok] Every evaluation produced a usable loss"},
             "1.000000e+16",
             {{"E", "1.000000e+16"}},
             {}};
    }
    SECTION("a PC_FAILPROBE S with no value under FAILSCORE_GRADING OFF") {
        name = "failprobe_novalue_off";
        c = {ta_script("3.301", "variable E equal pe\n"),
             "PC_FAILPROBE --script probe.in --outvar S\n"
             "PC_LAMMPS --script readback.in --varloss E 0 1\n",
             1,
             {"PC_FAILPROBE is defined but FAILSCORE_GRADING is OFF -- its S "
              "neither grades nor gates an evaluation; a probe that fails "
              "(its script fails, or its S cannot be read or has no number) "
              "fails its evaluation, as any PC_LAMMPS simulation does.",
              "PC_LAMMPS task failed (probe.in, occurrence 1): reading "
              "PC_FAILPROBE --outvar S after the script: " +
                  spoiled_reason(false),
              "no evaluation produced a usable loss"},
             {"plain diagnostic sim", "--outvar is NaN"},
             "1.000000e+16",
             {{"S", "1.000000e+16"}, {"E", "1.000000e+16"}},
             {{"probe.in", ta_script("3.301", "variable S equal 0.1\n" +
                                                  spoil("S", false))}}};
    }
    SECTION("a script that cannot be rendered") {
        name = "unrendered";
        c = {ta_script("3.301", "variable E equal pe\n"),
             "PC_LAMMPS --script readback.in --varloss E 0 1\n",
             1,
             {"PC_LAMMPS task failed (readback.in, occurrence 1): rendering "
              "the script to a temporary file: ",
              "no evaluation produced a usable loss"},
             {},
             "1.000000e+16",
             {{"E", "1.000000e+16"}},
             {{"notadir", "a file, not a directory\n"}},
             "TMPDIR=notadir"};
    }
    /* The MPI build (TADAH_HPO_MPI) has PC_ISOLATION THREAD only: there
     * the THREAD run is checked alone, its journal names the mode
     * ("PC_ISOLATION: THREAD (requested; the only mode of the MPI build)"),
     * and a PROCESS request is the startup error, exit 1; tadah runs under
     * the launcher of one rank (run_tadah_hpo). Before the fix the MPI
     * build's THREAD journal had no PC_ISOLATION line (the "deprecated
     * fallback" warning is the serial build's), this case ran PROCESS there
     * too, which the MPI build refuses, and, started without the launcher,
     * tadah crashed in the unrendered section (exit status 139). */
#ifdef TADAH_HPO_MPI
    const std::vector<std::string> modes{"THREAD"};
    const std::string thread_line =
        "PC_ISOLATION: THREAD (requested; the only mode of the MPI build)";
    {
        INFO(name << ", PC_ISOLATION PROCESS in the MPI build");
        const fs::path dir = sd.dir / (name + "_PROCESS");
        const int rc = run_tadah_hpo(bin, dir, "PROCESS", head + c.lammps_line,
                                     c.script, c.more_scripts, c.env);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == 1);
        CHECK_THAT(read_text(dir / "tadah.stderr"),
                   Catch::Contains("PC_ISOLATION PROCESS is not supported in "
                                   "the MPI build (TADAH_HPO_MPI)"));
    }
#else
    const std::vector<std::string> modes{"THREAD", "PROCESS"};
    const std::string thread_line = "PC_ISOLATION: THREAD (deprecated fallback)";
#endif
    std::vector<fs::path> dirs;
    for (const std::string& mode : modes) {
        INFO(name << ", PC_ISOLATION " << mode);
        const fs::path dir = sd.dir / (name + "_" + mode);
        const int rc = run_tadah_hpo(bin, dir, mode, head + c.lammps_line,
                                     c.script, c.more_scripts, c.env);
        INFO(read_text(dir / "tadah.stderr"));
        const std::string journal = read_text(dir / "hpo_run.log");
        INFO(journal);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == c.exit_status);
        CHECK_THAT(journal,
                   Catch::Contains(mode == "THREAD"
                                       ? thread_line
                                       : "PC_ISOLATION: PROCESS -- pc_lammps "
                                         "tasks execute in"));
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
        if (dirs.size() > 1)
            CHECK(read_text(dirs[0] / f) == read_text(dirs[1] / f));
    }
    if (dirs.size() > 1)
        for (const char* f : {"outvar.tadah", "best_outvar.tadah"}) {
            INFO(name << ": " << f << " without IT-TIME/ms");
            CHECK(lines_without(dirs[0] / f, 1) ==
                  lines_without(dirs[1] / f, 1));
        }
}

/* Pins: two runs in one process, as a program's two hpo_run() calls
 * (each an HPO_Host started with prepare_best_publication(), no reset in
 * between), with one PC_LAMMPS script whose --hotvar hot_a0 (2.5 4.5 3.30)
 * is the lattice constant of a bcc Ta cell, whose --varloss A is that
 * lattice constant, and which hands the next evaluation hot_a0 + 0.05: the
 * first run evaluates CGRID2B[1] = 2.8 twice, at lattices 3.30 and 3.35, and
 * the two losses differ by more than 1e-3 relative; the second run's
 * evaluation of 2.8 starts from the declared 3.30 again and its loss is the
 * first run's first loss, bit for bit.
 * Before the fix: the second run started from the value the first run's
 * last evaluation handed on, lattice 3.40, so its loss was another (the
 * bit comparison fails); with a relaxing script, the second run in one
 * process did not reproduce the same run made alone. */
TEST_CASE("a second run in one process starts each --hotvar from its "
          "declared initial value", "[hotvar_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    tadah::hpo::warmstart::clear();   // as a new process starts
    ScratchDir sd("tadah_hotvar_runs_");
    std::ofstream(sd.dir / "hot.in")
        << ta_script("${hot_a0}",
                     "variable A_i internal ${hot_a0}\n"
                     "variable A equal v_A_i\n"
                     "variable a_next internal $(v_hot_a0+0.05)\n"
                     "variable hot_a0 equal v_a_next\n");
    const std::string target =
        "PC_ERMSE 0 1\n"
        "PC_LAMMPS --script " + (sd.dir / "hot.in").string() +
        " --varloss A 0 1 --hotvar hot_a0 2.5 4.5 3.30\n";
    auto bits = [](double v) {
        std::uint64_t u;
        std::memcpy(&u, &v, sizeof u);
        return u;
    };
    Journal j;
    const auto first = evaluate(sd.dir, target, {2.8, 2.8});
    const auto second = evaluate(sd.dir, target, {2.8});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(first.size() == 2);
    REQUIRE(second.size() == 1);
    INFO(std::setprecision(17) << "first run " << first[0] << " "
                               << first[1] << ", second run " << second[0]);
    CHECK(std::fabs(first[1] - first[0]) > 1e-3 * std::fabs(first[0]));
    CHECK(bits(second[0]) == bits(first[0]));
    tadah::hpo::warmstart::clear();
    Host::reset_globals_for_test();
}

/* Pins, through the built tadah: a PC_LAMMPS run with an NLopt AUGLAG or
 * MLSL whose INNER search is derivative-free -- LD_AUGLAG with LN_COBYLA,
 * GD_MLSL_LDS with LN_SBPLX (MAXEVAL 6, PC_ISOLATION THREAD) -- starts and
 * ends normally (exit 0): its rows hold between 1 and 6 evaluations, every
 * one a feasible loss, best_loss.tadah has the best, and the journal names
 * no GRAD_STRATEGY (it does for a block that obtains gradients). MLSL and
 * AUGLAG evaluate their own points without a gradient; the local search
 * decides, and these take none.
 * Before the fix: refused at startup (exit 1, "--hpotarget defines
 * PC_LAMMPS but OPTIMIZER's optimiser NLOPT:LD_AUGLAG is
 * gradient-based"). */
TEST_CASE("tadah hpo: PC_LAMMPS with an MLSL or AUGLAG whose local search "
          "takes no gradient runs",
          "[pc_lammps_grad][tadah_cli][.tadah_cli_pc_lammps_local_search]") {
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin)
        TADAH_SKIP("TADAH_BINARY is not set: it names the built tadah "
                   "executable");
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd("tadah_cli_pc_local_search_");
    std::string algo, inner;
    SECTION("LD_AUGLAG with an INNER LN_COBYLA") {
        algo = "LD_AUGLAG";
        inner = "LN_COBYLA";
    }
    SECTION("GD_MLSL_LDS with an INNER LN_SBPLX") {
        algo = "GD_MLSL_LDS";
        inner = "LN_SBPLX";
    }
    INFO(algo << " with an INNER " << inner);
    const fs::path dir = sd.dir / algo;
    const std::string target =
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB NLOPT\n    ALGO " +
        algo + "\n    MAXEVAL 6\n    INNER\n      LIB NLOPT\n      ALGO " +
        inner + "\n    ENDINNER\n  ENDSTRATEGY\nENDOPTIMIZER\nLOSS L2\n"
        "FAILSCORE 1e16\nPC_ERMSE 0 1\nOPTIM CGRID2B (1) 0.0 4.0\n"
        "PC_LAMMPS --script readback.in --varloss E 0 1\n";
    const int rc = run_tadah_hpo(bin, dir, "THREAD", target,
                                 ta_script("3.301", "variable E equal pe\n"),
                                 {});
    INFO(read_text(dir / "tadah.stderr"));
    const std::string journal = read_text(dir / "hpo_run.log");
    INFO(journal);
    REQUIRE(WIFEXITED(rc));
    CHECK(WEXITSTATUS(rc) == 0);
    CHECK_THAT(journal, !Catch::Contains("GRAD_STRATEGY"));
    const Table loss = read_table(dir / "loss.tadah");
    CHECK(loss.rows.size() >= 1);
    CHECK(loss.rows.size() <= 6);
    for (const auto& row : loss.rows) {
        REQUIRE_FALSE(row.empty());
        double v = 0.0;
        REQUIRE(tadah::hpo::try_parse_double(row.back(), v));
        CHECK(tadah::hpo::loss_is_feasible(v, 1e16));
    }
    CHECK(read_table(dir / "best_loss.tadah").rows.size() >= 1);
}

namespace {

/* The lines of the file @p p that contain @p needle. */
std::size_t count_lines_with(const fs::path& p, const std::string& needle) {
    std::ifstream in(p);
    std::size_t n = 0;
    for (std::string line; std::getline(in, line);)
        if (line.find(needle) != std::string::npos) ++n;
    return n;
}

/* M_KRR Kern_RBF 1 on the minimal Ta data of kTaModelKeys (the four atoms
 * of the training structure give its SBASIS 2 basis vectors), and no
 * VERBOSE line: the run's verbosity is the command line's -v. */
constexpr const char* kTaKernelModelKeys =
    "ATOM Ta\nWATOM 1\nINIT2B true\nINITMB false\n"
    "TYPE2B D2_BP 2 2 Ta Ta\nCGRID2B 0.5 1.4\nSGRID2B 3.0 5.0\n"
    "RCTYPE2B Cut_Cos\nRCUT2B 5.0\nFORCE false\nSTRESS false\n"
    "BIAS false\nNORM false\nMODEL M_KRR Kern_RBF 1\nMPARAMS 1.0\n"
    "SBASIS 2\nLAMBDA 1e-3\nOALGO 1\n";

}  // namespace

/* Pins, through the built tadah: the potential a PC_LAMMPS script runs
 * (pair_coeff * * EXTERN Ta) is built from the evaluation's context, which
 * keeps the run's VERBOSE, as the trainer's does: under PC_ISOLATION
 * THREAD from that context, under PROCESS from its storage snapshot
 * (pc_pot_map), which each worker process rebuilds. A Kern_RBF prints its
 * line, "Kern_RBF | sigma: ... | gamma: ...", when it is built at VERBOSE
 * 1 or more: at most once per run (hpo_run's KernelSettingsLineScope), and
 * never in a PC_ISOLATION PROCESS worker (its Lines::None scope, for the
 * worker's life). M_KRR Kern_RBF 1, RANDOM MAXEVAL 3, a configuration
 * without VERBOSE (kTaKernelModelKeys):
 *   - -v 0 under PC_ISOLATION THREAD: exit 0, the journal names the mode,
 *     loss.tadah has the three evaluations, and tadah's stdout has no
 *     "Kern_RBF |" line. This leg pins the context both modes take: built
 *     from one without VERBOSE, at the blueprint default 1, the first
 *     PC_LAMMPS potential would print its line;
 *   - -v 0 under PROCESS (not in the MPI build): the same. It shows that no
 *     kernel line reaches tadah's stdout from the workers, but it cannot
 *     tell whether the context a worker rebuilds has VERBOSE: a worker
 *     prints no kernel line at any VERBOSE;
 *   - -v 1 under THREAD: exit 0, and tadah's stdout has "Kern_RBF |"
 *     lines (the line the -v 0 legs look for reaches tadah's stdout).
 * Before the fix (HB2 review F1): under -v 0 tadah's stdout had one
 * "Kern_RBF |" line per evaluation, 3, in both modes: the host took the
 * run's inputs, VERBOSE among them, out of the context the PC_LAMMPS
 * potential is built from, and a context without VERBOSE builds the kernel
 * at the blueprint default 1. The -v 1 leg passes as well (a pin).
 * Since the run's kernel-line scope and the workers' (fix/build-hygiene),
 * a context without VERBOSE again prints 1 line under THREAD (the first
 * VERBOSE-less build prints, the later ones are deduplicated) and none
 * under PROCESS: the THREAD check fails and the PROCESS one passes
 * (VERBOSE stripped from the evaluation's context). */
TEST_CASE("tadah hpo: the PC_LAMMPS potential is built at the run's VERBOSE",
          "[pc_lammps_verbose][tadah_cli][.tadah_cli_pc_lammps_verbose]") {
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin)
        TADAH_SKIP("TADAH_BINARY is not set: it names the built tadah "
                   "executable");
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd("tadah_cli_pc_verbose_");
    const std::string target =
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
        "    ALGO RANDOM\n    MAXEVAL 3\n    SEED 1234\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.0 4.0\n"
        "PC_LAMMPS --script readback.in --varloss E 0 1\n";
    const std::string script = ta_script("3.301", "variable E equal pe\n");
#ifdef TADAH_HPO_MPI
    const std::vector<std::string> modes{"THREAD"};
#else
    const std::vector<std::string> modes{"THREAD", "PROCESS"};
#endif
    /* THREAD first: it pins the context; PROCESS shows the workers' output
     * only (the comment above). */
    for (const std::string& mode : modes) {
        INFO("-v 0, PC_ISOLATION " << mode);
        const fs::path dir = sd.dir / ("v0_" + mode);
        const int rc = run_tadah_hpo(bin, dir, mode, target, script, {}, "",
                                     kTaKernelModelKeys, "-v 0");
        INFO(read_text(dir / "tadah.stderr"));
        const std::string journal = read_text(dir / "hpo_run.log");
        INFO(journal);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == 0);
        CHECK_THAT(journal, Catch::Contains("PC_ISOLATION: " + mode));
        CHECK(read_table(dir / "loss.tadah").rows.size() == 3);
        INFO(read_text(dir / "tadah.stdout"));
        CHECK(count_lines_with(dir / "tadah.stdout", "Kern_RBF |") == 0);
    }
    {
        INFO("-v 1, PC_ISOLATION THREAD");
        const fs::path dir = sd.dir / "v1_THREAD";
        const int rc = run_tadah_hpo(bin, dir, "THREAD", target, script, {},
                                     "", kTaKernelModelKeys, "-v 1");
        INFO(read_text(dir / "tadah.stderr"));
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == 0);
        CHECK(count_lines_with(dir / "tadah.stdout", "Kern_RBF |") > 0);
    }
}

namespace {

/* N of the journal line "PC_ISOLATION: PROCESS -- pc_lammps tasks execute
 * in N persistent worker processes ...", or -1 when @p journal has none. */
long process_workers(const std::string& journal) {
    const std::string head = "pc_lammps tasks execute in ";
    const auto p = journal.find(head);
    if (p == std::string::npos) return -1;
    return std::strtol(journal.c_str() + p + head.size(), nullptr, 10);
}

}  // namespace

/* Pins, through the built tadah: at VERBOSE 1 a tadah hpo run prints the
 * line a kernel prints when it is built, "Kern_RBF | sigma: 1 | gamma: 1",
 * once, however many models it builds (the -c model at startup, one per
 * evaluation, and the potential each PC_LAMMPS script runs). M_KRR Kern_RBF
 * 1 (kTaKernelModelKeys), RANDOM MAXEVAL 4, a PC_LAMMPS script, -v 1:
 *   - PC_ISOLATION THREAD: exit 0, four evaluations in loss.tadah, and
 *     tadah's stdout has exactly one "Kern_RBF |" line, the -c model's
 *     settings;
 *   - PC_ISOLATION PROCESS (not in the MPI build, which has no workers):
 *     exit 0, four evaluations, the journal's "execute in N persistent
 *     worker processes" with N >= 1, and again exactly one line: the worker
 *     processes, which build the potentials of the scripts, print none.
 * Before the fix: under PROCESS one line from tadah plus one from each
 * worker process, 2 with one worker (a worker printed the line of the
 * first potential it built); in the released code 9 lines in both modes,
 * one at startup and two per evaluation (its model and its PC_LAMMPS
 * potential). */
TEST_CASE("tadah hpo: a kernel's settings line is printed once per run",
          "[kernel_line_once][tadah_cli][.tadah_cli_kernel_line_once]") {
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin)
        TADAH_SKIP("TADAH_BINARY is not set: it names the built tadah "
                   "executable");
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd("tadah_cli_kernel_line_");
    const std::string target =
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
        "    ALGO RANDOM\n    MAXEVAL 4\n    SEED 1234\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.0 4.0\n"
        "PC_LAMMPS --script readback.in --varloss E 0 1\n";
    const std::string script = ta_script("3.301", "variable E equal pe\n");
    const std::string line = "Kern_RBF | sigma: 1 | gamma: 1";
#ifdef TADAH_HPO_MPI
    const std::vector<std::string> modes{"THREAD"};
#else
    const std::vector<std::string> modes{"THREAD", "PROCESS"};
#endif
    for (const std::string& mode : modes) {
        INFO("-v 1, PC_ISOLATION " << mode);
        const fs::path dir = sd.dir / ("v1_" + mode);
        const int rc = run_tadah_hpo(bin, dir, mode, target, script, {}, "",
                                     kTaKernelModelKeys, "-v 1");
        INFO(read_text(dir / "tadah.stderr"));
        const std::string journal = read_text(dir / "hpo_run.log");
        INFO(journal);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == 0);
        CHECK(read_table(dir / "loss.tadah").rows.size() == 4);
        INFO(read_text(dir / "tadah.stdout"));
        if (mode == "PROCESS") CHECK(process_workers(journal) >= 1);
        CHECK(count_lines_with(dir / "tadah.stdout", "Kern_RBF |") == 1);
        CHECK(count_lines_with(dir / "tadah.stdout", line) == 1);
    }
}

namespace {

/* What `tadah hpo -c <config> --hpotarget config.hpo --validation
 * config.val -v 1` hands TaskFactory. */
struct HpoCommandInput : tadah::mlip::ITaskInput {
    std::string config;
    explicit HpoCommandInput(std::string c) : config(std::move(c)) {}
    std::string getTaskName() const override { return "HPO"; }
    std::unordered_map<std::string, std::vector<std::string>>
    getKeyValueData() const override {
        return {{"config", {config}},
                {"hpotarget", {"config.hpo"}},
                {"validation", {"config.val"}},
                {"verbose", {"1"}}};
    }
};

/* TADAH_PC_ISOLATION unset while this lives (it overrides the
 * configuration's PC_ISOLATION), as the case found it afterwards. */
struct PcIsolationEnvUnset {
    std::optional<std::string> saved;
    PcIsolationEnvUnset() {
        if (const char* e = std::getenv("TADAH_PC_ISOLATION")) saved = e;
        ::unsetenv("TADAH_PC_ISOLATION");
    }
    ~PcIsolationEnvUnset() {
        if (saved) ::setenv("TADAH_PC_ISOLATION", saved->c_str(), 1);
        else       ::unsetenv("TADAH_PC_ISOLATION");
    }
};

/* File descriptor 1 of this process into the file @p path while the object
 * lives: the kernels' lines (std::cout), and whatever a PC_ISOLATION
 * PROCESS worker, which inherits the descriptor, writes there. */
struct StdoutToFile {
    int saved = -1;
    explicit StdoutToFile(const fs::path& path) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                              0644);
        REQUIRE(fd >= 0);
        std::cout.flush();
        std::fflush(stdout);
        saved = ::dup(1);
        ::dup2(fd, 1);
        ::close(fd);
    }
    ~StdoutToFile() {
        std::cout.flush();
        std::fflush(stdout);
        ::dup2(saved, 1);
        ::close(saved);
    }
    StdoutToFile(const StdoutToFile&) = delete;
    StdoutToFile& operator=(const StdoutToFile&) = delete;
};

/* Two runs in one process under PC_ISOLATION @p mode, as a program that
 * calls hpo_run() twice runs them, each with the context and provenance
 * TaskFactory makes of `tadah hpo -c config.a` (then config.b)
 * `--hpotarget config.hpo --validation config.val -v 1`, and the checks of
 * "hpo_run: a second run in the process prints its own kernel's settings
 * line once". */
void check_kernel_line_of_each_run(const std::string& mode) {
    tadah::core::PeriodicTable::initialize();
    /* As a new process starts; the second run starts without a reset. */
    HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    tadah::hpo::warmstart::clear();
    PcIsolationEnvUnset env;
    ScratchDir sd("tadah_kernel_line_runs_");
    std::string keys_b = kTaKernelModelKeys;
    const std::string mparams_a = "MPARAMS 1.0\n";
    const auto at = keys_b.find(mparams_a);
    REQUIRE(at != std::string::npos);
    keys_b.replace(at, mparams_a.size(), "MPARAMS 2.0\n");
    write_minimal_ta_dbfile(sd.dir / "train.db", 0.0);
    write_minimal_ta_dbfile(sd.dir / "val.db", 0.07, -0.04);
    std::ofstream(sd.dir / "config.val") << "DBFILE val.db\n";
    std::ofstream(sd.dir / "config.a") << "DBFILE train.db\n"
                                       << kTaKernelModelKeys;
    std::ofstream(sd.dir / "config.b") << "DBFILE train.db\n" << keys_b;
    std::ofstream(sd.dir / "readback.in")
        << ta_script("3.301", "variable E equal pe\n");
    std::ofstream(sd.dir / "config.hpo")
        << "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
           "    ALGO RANDOM\n    MAXEVAL 4\n    SEED 1234\n  ENDSTRATEGY\n"
           "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\nPC_ERMSE 0 1\n"
           "OPTIM CGRID2B (1) 0.0 4.0\n"
           "PC_LAMMPS --script readback.in --varloss E 0 1\n"
        << "PC_ISOLATION " << mode << "\n";

    struct Run {
        std::string config;
        int status = -1;
        std::string error, journal;
    };
    std::vector<Run> runs(2);
    runs[0].config = "config.a";
    runs[1].config = "config.b";
    {
        StdoutToFile capture(sd.dir / "runs.stdout");
        for (Run& run : runs) {
            try {
                std::vector<tadah::mlip::Task> tasks =
                    tadah::mlip::TaskFactory().createTasksFromInput(
                        HpoCommandInput(run.config));
                if (tasks.size() != 1)
                    throw std::runtime_error(
                        std::to_string(tasks.size()) + " tasks, not 1");
                Context ctx = tasks.front().getContext();
                run.status = tadah::hpo::hpo_run(
                    ctx, tasks.front().getKeyProvenance());
            } catch (const std::exception& e) {
                run.error = e.what();
            }
            run.journal = read_text(sd.dir / "hpo_run.log");
        }
    }
    for (const Run& run : runs) {
        INFO("-c " << run.config << ", PC_ISOLATION " << mode);
        INFO(run.journal);
        CHECK(run.error == "");
        CHECK(run.status == 0);
        CHECK_THAT(run.journal, Catch::Contains("PC_ISOLATION: " + mode));
        if (mode == "PROCESS") CHECK(process_workers(run.journal) >= 1);
    }
    const fs::path out = sd.dir / "runs.stdout";
    INFO(read_text(out));
    CHECK(count_lines_with(out, "Kern_RBF | sigma: 1 | gamma: 1") == 1);
    CHECK(count_lines_with(out, "Kern_RBF | sigma: 2 | gamma: 0.25") == 1);
    CHECK(count_lines_with(out, "Kern_RBF |") == 2);
}

}  // namespace

/* Pins: two runs in one process, as a program that calls hpo_run() twice
 * runs them, each print the settings line of their own -c model once. The
 * context of each run is the one TaskFactory makes of `tadah hpo -c config.a`
 * (then config.b) `--hpotarget config.hpo --validation config.val -v 1`:
 * run A trains M_KRR Kern_RBF 1 from MPARAMS 1.0, run B from MPARAMS 2.0
 * (kTaKernelModelKeys otherwise), each RANDOM MAXEVAL 4 with a PC_LAMMPS
 * script, and nothing is reset between them. Under PC_ISOLATION THREAD (this
 * case) and PROCESS (the next; Skipped in the MPI build, which has no
 * workers): each run returns 0, its journal names the mode (under PROCESS
 * with N >= 1 workers), and the standard output of the process, workers
 * included, has "Kern_RBF | sigma: 1 | gamma: 1" exactly once, "Kern_RBF |
 * sigma: 2 | gamma: 0.25" exactly once, and no other "Kern_RBF |" line.
 * Before the fix: the line was printed by the first kernel of its kind in
 * each process and by no later one: under THREAD run B printed none
 * (sigma: 2 zero times, one line in all, run A's); under PROCESS a worker
 * process of each run printed the line of its first potential once more
 * (sigma: 1 twice, sigma: 2 once, 3 lines). In the released code each
 * model built printed it, 9 lines per run. With hpo_run()'s scope made
 * static, one scope for the life of the process, run B prints none again
 * under both modes (sigma: 2 zero times). */
TEST_CASE("hpo_run: a second run in the process prints its own kernel's "
          "settings line once (PC_ISOLATION THREAD)",
          "[kernel_line_e2e][.]") {
    check_kernel_line_of_each_run("THREAD");
}

/* Pins: the case above under PC_ISOLATION PROCESS. */
TEST_CASE("hpo_run: a second run in the process prints its own kernel's "
          "settings line once (PC_ISOLATION PROCESS)",
          "[kernel_line_e2e][.]") {
    SKIP_WITHOUT_PC_PROCESS_WORKERS();
    check_kernel_line_of_each_run("PROCESS");
}

namespace {

/* The lines of @p journal that contain @p needle. */
std::vector<std::string> journal_lines_with(const std::string& journal,
                                            const std::string& needle) {
    std::vector<std::string> out;
    std::istringstream in(journal);
    for (std::string line; std::getline(in, line);)
        if (line.find(needle) != std::string::npos) out.push_back(line);
    return out;
}

}  // namespace

/* Checks, through the built tadah: the startup line on the per-script
 * limit, "PC_SCRIPT_TIMEOUT: <limit> (from <key>) -- soft LAMMPS `timer
 * timeout`, re-armed each run-section; hard per-worker-task deadline:
 * <deadline>", ends with the deadline the run's PC_ISOLATION mode gives a
 * worker task. A PC_LAMMPS line under TADAH:RANDOM MAXEVAL 1 (exit 0):
 *   - PROCESS, PC_WORKER_TIMEOUT 600: "600 s (PC_WORKER_TIMEOUT)", the
 *     limit "off (no per-script limit)" from "default";
 *   - PROCESS, PC_SCRIPT_TIMEOUT 10 and no PC_WORKER_TIMEOUT: "130 s,
 *     derived as max(3t, t + 120 s) from the per-script limit t (a line's
 *     --maxtime derives its own)", the limit "10 s" from
 *     "PC_SCRIPT_TIMEOUT";
 *   - PROCESS, neither key: "none";
 *   - THREAD, PC_WORKER_TIMEOUT 600: "n/a (THREAD mode)", the worker
 *     timeout taking no effect.
 * The MPI build, which has no worker processes, runs the THREAD case
 * alone. tadah runs without TADAH_PC_WORKER_TIMEOUT, which would override
 * the target file's PC_WORKER_TIMEOUT, as it runs without
 * TADAH_PC_ISOLATION (run_tadah_hpo).
 * With the text of the PC_WORKER_TIMEOUT branch in hpo_run changed ("s
 * (PC_WORKER_TIMEOUT)" to "s (worker timeout)"), the first case fails:
 * no test checked this line before. */
TEST_CASE("tadah hpo: the PC_SCRIPT_TIMEOUT startup line names the hard "
          "per-worker-task deadline of the run's PC_ISOLATION mode",
          "[pc_deadline_line][tadah_cli][.tadah_cli_pc_deadline_line]") {
    const char* bin = std::getenv("TADAH_BINARY");
    if (!bin || !*bin)
        TADAH_SKIP("TADAH_BINARY is not set: it names the built tadah "
                   "executable");
    REQUIRE(fs::exists(bin));
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd("tadah_cli_pc_deadline_");
    const std::string head =
        "OPTIMIZER\n  STRATEGY\n    TYPE PLAIN\n    LIB TADAH\n"
        "    ALGO RANDOM\n    MAXEVAL 1\n    SEED 1234\n  ENDSTRATEGY\n"
        "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e16\nPC_ERMSE 0 1\n"
        "OPTIM CGRID2B (1) 0.0 4.0\n"
        "PC_LAMMPS --script readback.in --varloss E 0 1\n";
    const std::string script = ta_script("3.301", "variable E equal pe\n");
    struct Case {
        std::string name, mode, keys, limit, deadline;
    };
    const std::string no_limit = "off (no per-script limit) (from default)";
    std::vector<Case> cases;
#ifndef TADAH_HPO_MPI
    cases.push_back({"worker_timeout", "PROCESS", "PC_WORKER_TIMEOUT 600\n",
                     no_limit, "600 s (PC_WORKER_TIMEOUT)"});
    cases.push_back({"derived", "PROCESS", "PC_SCRIPT_TIMEOUT 10\n",
                     "10 s (from PC_SCRIPT_TIMEOUT)",
                     "130 s, derived as max(3t, t + 120 s) from the "
                     "per-script limit t (a line's --maxtime derives its "
                     "own)"});
    cases.push_back({"none", "PROCESS", "", no_limit, "none"});
#endif
    cases.push_back({"thread", "THREAD", "PC_WORKER_TIMEOUT 600\n", no_limit,
                     "n/a (THREAD mode)"});
    for (const Case& c : cases) {
        INFO(c.name << ": PC_ISOLATION " << c.mode << ", " << c.keys);
        const fs::path dir = sd.dir / c.name;
        const int rc = run_tadah_hpo(bin, dir, c.mode, head + c.keys, script,
                                     {}, "-u TADAH_PC_WORKER_TIMEOUT");
        INFO(read_text(dir / "tadah.stderr"));
        const std::string journal = read_text(dir / "hpo_run.log");
        INFO(journal);
        REQUIRE(WIFEXITED(rc));
        CHECK(WEXITSTATUS(rc) == 0);
        const std::vector<std::string> lines =
            journal_lines_with(journal, "hard per-worker-task deadline: ");
        REQUIRE(lines.size() == 1);
        const std::string tail =
            "PC_SCRIPT_TIMEOUT: " + c.limit + " -- soft LAMMPS `timer "
            "timeout`, re-armed each run-section; hard per-worker-task "
            "deadline: " + c.deadline;
        INFO("the line: " << lines[0]);
        REQUIRE(lines[0].size() >= tail.size());
        CHECK(lines[0].compare(lines[0].size() - tail.size(), tail.size(),
                               tail) == 0);
    }
}

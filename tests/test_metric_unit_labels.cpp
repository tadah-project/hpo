// test_metric_unit_labels.cpp
//
// The units HPO prints for the relative energy RMSE (PC_ErRMSE) and the
// stress RMSE (PC_SRMSE) (known-issues workstream 1, branch
// fix/pc-frmse-force-off; tmp/known-issues/ws1_review.json, "New
// metric-unit table").
//
//   * ErRMSE is sqrt(mean(((E_ref - E_pred) / E_ref)^2)) x 1000
//     (Analytics::calc_e_rrmse, scaled by HPO_Host::fill_validation): a
//     dimensionless ratio in per mille, not an energy. It is labelled
//     "per mille".
//   * SRMSE is the raw stress difference (Analytics::calc_s_rmse, scaled by
//     1): it is in the dataset's unit, eV/A^3. It is labelled "eV/A^3".
//
// Every place that prints the two metrics is covered:
//   1. hpo_run.log LOSS FUNCTION rows   (HPOLog::write_loss_section);
//   2. hpo_run.log BEST progress line   (HPOLog::log_new_best);
//   3. hpo_run.log RESULT rows          (HPOLog::finish, "Best ErRMSE",
//                                        "Best SRMSE");
//   4. the best_pot.tadah header: its LOSS FUNCTION rows
//      (PotHeaderWriter::write_loss_section, pot_header_writer.cpp) and its
//      RESULT table (write_result_table, pot_header_writer_impl.hpp), through
//      the public write_best_header; the RESULT table of the best_pot_N
//      archive header through write_best_archive_header;
//   5. the --skip-above rejection message of HPO_Host::operator() (a
//      WARNING in hpo_run.log), which needs a real host: hidden case in the
//      [valmetric_e2e] group.
//
// Only the ErRMSE / SRMSE rows are asserted on, and only by substrings: an
// ErRMSE row contains "per mille" and not "meV"; an SRMSE row contains
// "eV/A^3" and not "GPa". Column spacing is not pinned. Each place must
// print at least one such row, so a renamed or dropped row fails instead of
// passing vacuously.
//
// Before the fix every one of these places printed ErRMSE as meV/atom
// (meV/at) and SRMSE as GPa, so every case below fails on both metrics.
//
// The fast cases are [metric_units]; the host case is hidden
// [valmetric_e2e][.] and runs in the CTest entry hpo_valmetric_e2e:
//   ./test_hpo "[metric_units]"
//   ./test_hpo "[valmetric_e2e]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

/* ---------------------------------------------------------------------------
 * Scratch files and the journal.
 * ------------------------------------------------------------------------- */

/* @p stem, this process id and a steady-clock count: distinct across the
 * parallel ctest processes. */
std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/* A temporary file holding @p body; removed with the object. */
struct ScopedTextFile {
    fs::path path;
    ScopedTextFile(const std::string& ext, const std::string& body)
        : path(fs::temp_directory_path() /
               (unique_name("tadah_metric_units_") + ext)) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
        REQUIRE(os.good());
    }
    ~ScopedTextFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* hpo_run.log in a temporary file, opened for one test case and read back
 * after finish(). open() cannot reopen a journal another test left open,
 * so one is closed first. */
struct JournalFile {
    fs::path path;
    JournalFile()
        : path(fs::temp_directory_path() /
               (unique_name("tadah_metric_units_log_") + ".log")) {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
        REQUIRE(HPOLog::instance().is_open());
    }
    ~JournalFile() {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish() {
        HPOLog::instance().finish("TEST");
        return read_text(path);
    }
};

/* ---------------------------------------------------------------------------
 * Reading the rows back.
 * ------------------------------------------------------------------------- */

/* The part of @p text from @p head up to (not including) the next
 * occurrence of @p next after it, or to the end; "" when @p head is
 * absent. */
std::string block(const std::string& text, const std::string& head,
                  const std::string& next) {
    const auto at = text.find(head);
    if (at == std::string::npos) return {};
    const auto end = text.find(next, at + head.size());
    return text.substr(at, end == std::string::npos ? std::string::npos
                                                    : end - at);
}

/* Every line of @p text that contains @p needle. */
std::vector<std::string> lines_with(const std::string& text,
                                    const std::string& needle) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);)
        if (line.find(needle) != std::string::npos) out.push_back(line);
    return out;
}

/* Rows that print an ErRMSE value or target: in per mille, never meV.
 * @p where names the place for the failure message; @p context is the text
 * the rows were taken from. */
void check_errmse_rows(const std::vector<std::string>& rows,
                       const std::string& where, const std::string& context) {
    INFO(where << ":\n" << context);
    REQUIRE_FALSE(rows.empty());
    for (const std::string& row : rows) {
        INFO("ErRMSE row: " << row);
        CHECK_THAT(row, Contains("per mille"));
        CHECK_THAT(row, !Contains("meV"));
    }
}

/* Rows that print an SRMSE value or target: in eV/A^3, never GPa. */
void check_srmse_rows(const std::vector<std::string>& rows,
                      const std::string& where, const std::string& context) {
    INFO(where << ":\n" << context);
    REQUIRE_FALSE(rows.empty());
    for (const std::string& row : rows) {
        INFO("SRMSE row: " << row);
        CHECK_THAT(row, Contains("eV/A^3"));
        CHECK_THAT(row, !Contains("GPa"));
    }
}

/* The field of a BEST progress line that starts at @p name ("ErRMSE=",
 * "SRMSE="): from @p name up to the next metric field of the line, or to
 * the end of the line, whatever order the fields come in. "" when the line
 * has no such field. The tail after the last metric (the wall time, a
 * slot tag) carries no unit text. */
std::string metric_field(const std::string& line, const std::string& name) {
    const auto at = line.find(name);
    if (at == std::string::npos) return {};
    std::size_t end = std::string::npos;
    for (const char* other :
         {"EMAE=", "ERMSE=", "ErRMSE=", "FRMSE=", "SRMSE="}) {
        const auto p = line.find(other, at + name.size());
        if (p != std::string::npos) end = std::min(end, p);
    }
    return line.substr(at, end == std::string::npos ? std::string::npos
                                                    : end - at);
}

/* ---------------------------------------------------------------------------
 * An hpotarget with every validation metric, PC_ErRMSE and PC_SRMSE among
 * them, on a context with one OPTIM key (the setup of
 * test_hpo_pot_digits.cpp).
 * ------------------------------------------------------------------------- */

Context watom_ctx() {
    Context ctx;
    ctx.add("ATOM", "C");
    ctx.add("ATOM", "H");
    ctx.add("WATOM", "C");
    ctx.add("WATOM", "H");
    ctx.add("WATOM", 1.0);
    return ctx;
}

const char* kAllMetricsTarget =
    "OPTIM WATOM (1) 0.1 5.0\n"
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n"
    "PC_EMAE 0 1\n"
    "PC_ERMSE 0 1\n"
    "PC_ErRMSE 0.5 1\n"
    "PC_FRMSE 0 1\n"
    "PC_SRMSE 0.01 2\n";

/* An evaluation result with every metric set, two validation datasets
 * (so the per-dataset RESULT rows are written too). */
tadah::hpo::EvalResult<ParamVec> all_metrics_result() {
    tadah::hpo::EvalResult<ParamVec> r;
    r.iter = 7;
    r.best_iter = 3;
    r.loss = 1.25;
    r.emae = 1.5;    r.emaeLoss = 2.25;
    r.ermse = 2.0;   r.ermseLoss = 4.0;
    r.errmse = 0.06; r.errmseLoss = 0.0036;
    r.frmse = 0.1;   r.frmseLoss = 0.01;
    r.srmse = 0.019; r.srmseLoss = 0.000361;
    r.v_emae   = {1.4, 1.6};     r.v_emaeLoss   = {1.96, 2.56};
    r.v_ermse  = {1.9, 2.1};     r.v_ermseLoss  = {3.61, 4.41};
    r.v_errmse = {0.05, 0.07};   r.v_errmseLoss = {0.0025, 0.0049};
    r.v_frmse  = {0.09, 0.11};   r.v_frmseLoss  = {0.0081, 0.0121};
    r.v_srmse  = {0.018, 0.020}; r.v_srmseLoss  = {0.000324, 0.0004};
    r.params.set_size(1);
    r.params(0) = 1.0;
    return r;
}

/* ---------------------------------------------------------------------------
 * The --skip-above host (hidden case): a fixed potential (every weight
 * pinned by FIXINDEX/FIXWEIGHT, OPTIM FIXWEIGHT (1), no training DBFILE;
 * the setup of test_validation_follows_metric_e2e.cpp case 2) validated on
 * two rattled bcc Ta cells with made-up energies and stresses.
 * ------------------------------------------------------------------------- */

/* A temporary working directory, the cwd while the object lives (the host
 * writes loss.tadah, stats.tadah, best_pot.tadah ... there); removed with
 * everything in it. */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() / unique_name("tadah_metric_units_e2e_");
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

/* A trainer built for this host: the thread-local trainer cache may still
 * hold another host's model. */
struct ForceRebuildGuard {
    bool restore;
    ForceRebuildGuard() {
        restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1);
    }
    ~ForceRebuildGuard() {
        if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD");
    }
};

/* A 2x2x2 conventional bcc Ta cell (16 atoms, edge 2a >= 6.5 A, above the
 * 4.5 A cutoff), rattled deterministically, in the DBFILE text format
 * Structure::read reads: label, weights, energy, cell, stress, atoms, blank
 * line. Energy @p energy (eV, nonzero: ErRMSE divides by it); a symmetric
 * stress whose six independent components are all nonzero; zero forces
 * (nothing here fits or measures forces). */
void write_cell(std::ostream& os, int id, double a, double energy) {
    const double L = 2.0 * a;
    os << std::setprecision(17);
    os << "metric units validation cell " << id << "\n"
       << "1.0 1.0 1.0\n"
       << energy << "\n"
       << L << " 0 0\n"
       << "0 " << L << " 0\n"
       << "0 0 " << L << "\n";
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y)
            os << (y ? " " : "")
               << 0.004 * static_cast<double>(1 + x + y + id) +
                      (x == y ? 0.003 * static_cast<double>(x + 1) : 0.0);
        os << "\n";
    }
    std::size_t i = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                for (double h : {0.0, 0.25}) {
                    const int n[3] = {x, y, z};
                    os << "Ta";
                    for (int k = 0; k < 3; ++k) {
                        const double f =
                            0.5 * n[k] + h + 0.02 +
                            0.006 * std::sin(2.3 * static_cast<double>(i) +
                                             1.1 * static_cast<double>(k) +
                                             0.7 * static_cast<double>(id));
                        os << " " << f * L;
                    }
                    os << " 0 0 0\n";
                    ++i;
                }
    os << "\n";
}

Context pinned_context(const fs::path& val_cfg, const fs::path& hpo_cfg) {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(3));
    c.add("TYPE2B",   static_cast<int>(3));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  "2.6 3.3 4.0");
    c.add("SGRID2B",  "1.0 1.0 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   4.5);
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
    c.add("LAMBDA",   0);
    c.add("OALGO",    1);
    c.add("FIXINDEX", "1-3");
    c.add("FIXWEIGHT", 0.8);
    c.add("FIXWEIGHT", -0.35);
    c.add("FIXWEIGHT", 0.2);
    c.add("VALIDATION", val_cfg.string());
    c.add("HPOTARGET",  hpo_cfg.string());
    return c;
}

/* One evaluation of the fixed potential with the hpotarget metric line
 * @p metric_line (a cap far below any value the potential can reach, so
 * the candidate is rejected); returns hpo_run.log. */
std::string skip_above_journal(const std::string& metric_line) {
    tadah::core::PeriodicTable::initialize();
    ScratchDir sd;
    const fs::path val_db  = sd.dir / "val.db";
    const fs::path val_cfg = sd.dir / "config.val";
    const fs::path hpo_cfg = sd.dir / "config.hpo";
    {
        std::ofstream os(val_db);
        REQUIRE(os.good());
        write_cell(os, 1, 3.25, -80.3);
        write_cell(os, 2, 3.35, -79.6);
        REQUIRE(os.good());
    }
    std::ofstream(val_cfg) << "DBFILE " << val_db.string() << "\n";
    std::ofstream(hpo_cfg)
        << "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
        << "LOSS L2\nFAILSCORE 1e9\n"
        << metric_line << "\n"
        << "OPTIM FIXWEIGHT (1) -1.0 1.0\n";

    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    Context ctx = pinned_context(val_cfg, hpo_cfg);
    JournalFile j;
    HPOTargets trg(ctx, hpo_cfg.string());
    {
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.val_size() == 2u);
        host.prepare_best_publication();
        ParamVec p(1);
        p(0) = 0.8;   // FIXWEIGHT (1) at its pinned value
        REQUIRE_NOTHROW(host(p));
        host.flush_remaining();
    }
    const std::string journal = j.finish();
    Host::reset_globals_for_test();
    return journal;
}

/* Lower-case copy, to find the metric tag of a rejection line whatever its
 * capitalisation ("eRRMSE", "srmse"). */
std::string lower(std::string s) {
    for (char& ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

}  // namespace

/* Place 1. HPOLog::write_loss_section with PC_ErRMSE and PC_SRMSE (and the
 * other metrics) set: the "ErRMSE" row of LOSS FUNCTION is in per mille,
 * the "SRMSE" row in eV/A^3.
 * Before the fix: "ErRMSE (meV/atom)" and "SRMSE  (GPa)" (hpo_log.cpp
 * write_loss_section), so every CHECK on both rows fails. */
TEST_CASE("metric units: the journal LOSS FUNCTION labels ErRMSE per mille "
          "and SRMSE eV per cubic Angstrom", "[metric_units]")
{
    JournalFile j;
    ScopedTextFile target(".hpo", kAllMetricsTarget);
    Context ctx = watom_ctx();
    HPOTargets trg(ctx, target.path.string());
    REQUIRE(trg.use_errmse);
    REQUIRE(trg.use_srmse);
    HPOLog::instance().write_loss_section(trg);
    const std::string journal = j.finish();

    const std::string loss = block(journal, "-- LOSS FUNCTION", "\n-- ");
    INFO(journal);
    REQUIRE_FALSE(loss.empty());
    check_errmse_rows(lines_with(loss, "ErRMSE"), "journal LOSS FUNCTION", loss);
    check_srmse_rows(lines_with(loss, "SRMSE"), "journal LOSS FUNCTION", loss);
}

/* Place 2. The BEST progress line HPOLog::log_new_best writes with every
 * metric: its "ErRMSE=" field is in per mille, its "SRMSE=" field in
 * eV/A^3 (fields compared, not the whole line, which also holds
 * EMAE/ERMSE in meV/at).
 * Before the fix: "ErRMSE=0.06 meV/at" and "SRMSE=0.019 GPa" (hpo_log.cpp
 * log_new_best). */
TEST_CASE("metric units: the journal BEST line labels ErRMSE per mille and "
          "SRMSE eV per cubic Angstrom", "[metric_units]")
{
    JournalFile j;
    HPOLog::instance().log_new_best(1, 2.0,
                                    1.5,   true,    // EMAE   (meV/atom)
                                    2.0,   true,    // ERMSE  (meV/atom)
                                    0.06,  true,    // ErRMSE (per mille)
                                    0.1,   true,    // FRMSE  (eV/A)
                                    0.019, true,    // SRMSE  (eV/A^3)
                                    12.0);
    const std::string journal = j.finish();
    INFO(journal);

    /* The BEST event line (tag "BEST"; the RESULT rows say "Best"). */
    std::vector<std::string> best;
    for (const auto& l : lines_with(journal, "ErRMSE="))
        if (l.find("BEST") != std::string::npos) best.push_back(l);
    REQUIRE(best.size() == 1u);
    const std::string& line = best.front();
    INFO("BEST line: " << line);
    const std::string errmse = metric_field(line, "ErRMSE=");
    const std::string srmse = metric_field(line, "SRMSE=");
    REQUIRE_FALSE(errmse.empty());
    REQUIRE_FALSE(srmse.empty());
    check_errmse_rows({errmse}, "journal BEST line, ErRMSE field", line);
    check_srmse_rows({srmse}, "journal BEST line, SRMSE field", line);
}

/* Place 3. The RESULT section HPOLog::finish writes after a best with every
 * metric: "Best ErRMSE" is in per mille, "Best SRMSE" in eV/A^3.
 * Before the fix: "Best ErRMSE ... meV/atom" and "Best SRMSE ... GPa"
 * (hpo_log.cpp finish). */
TEST_CASE("metric units: the journal RESULT labels Best ErRMSE per mille and "
          "Best SRMSE eV per cubic Angstrom", "[metric_units]")
{
    JournalFile j;
    HPOLog::instance().log_new_best(1, 2.0, 1.5, true, 2.0, true, 0.06, true,
                                    0.1, true, 0.019, true, 12.0);
    const std::string journal = j.finish();

    const std::string result = block(journal, "-- RESULT", "\n-- ");
    INFO(journal);
    REQUIRE_FALSE(result.empty());
    const auto errmse = lines_with(result, "ErRMSE");
    const auto srmse = lines_with(result, "SRMSE");
    check_errmse_rows(errmse, "journal RESULT", result);
    check_srmse_rows(srmse, "journal RESULT", result);
    /* The rows are the "Best ..." ones. */
    for (const auto& row : errmse) CHECK_THAT(row, Contains("Best ErRMSE"));
    for (const auto& row : srmse) CHECK_THAT(row, Contains("Best SRMSE"));
}

/* Place 4a. The best_pot.tadah header (PotHeaderWriter::write_best_header):
 * the LOSS FUNCTION rows (pot_header_writer.cpp write_loss_section) and the
 * RESULT table rows -- per dataset and average -- (pot_header_writer_impl.hpp
 * write_result_table) of ErRMSE are in per mille, those of SRMSE in eV/A^3.
 * Before the fix: "ErRMSE (meV/at)" and "SRMSE  (GPa)" in both sections. */
TEST_CASE("metric units: the best_pot.tadah header labels ErRMSE per mille "
          "and SRMSE eV per cubic Angstrom", "[metric_units]")
{
    ScopedTextFile target(".hpo", kAllMetricsTarget);
    Context ctx = watom_ctx();
    HPOTargets trg(ctx, target.path.string());
    REQUIRE(trg.use_errmse);
    REQUIRE(trg.use_srmse);
    const auto r = all_metrics_result();
    const Context val_ctx{};
    const tadah::mlip::StructureDB train_db{}, val_db{};
    const tadah::mlip::LoadTransformReport report{};
    std::ostringstream os;
    tadah::hpo::PotHeaderWriter::write_best_header(
        os, r, trg, ctx, val_ctx, train_db, val_db, {"WATOM[1]"}, report,
        /*cw=*/16, /*prec=*/6);
    const std::string header = os.str();
    INFO(header);

    const std::string loss = block(header, "# -- LOSS FUNCTION", "\n# -- ");
    REQUIRE_FALSE(loss.empty());
    check_errmse_rows(lines_with(loss, "ErRMSE"),
                      "best_pot.tadah LOSS FUNCTION", loss);
    check_srmse_rows(lines_with(loss, "SRMSE"),
                     "best_pot.tadah LOSS FUNCTION", loss);

    const std::string result = block(header, "# -- RESULT", "\n# -- ");
    REQUIRE_FALSE(result.empty());
    const auto errmse = lines_with(result, "ErRMSE");
    const auto srmse = lines_with(result, "SRMSE");
    /* Two datasets and the average. */
    CHECK(errmse.size() == 3u);
    CHECK(srmse.size() == 3u);
    check_errmse_rows(errmse, "best_pot.tadah RESULT", result);
    check_srmse_rows(srmse, "best_pot.tadah RESULT", result);

    /* Nowhere else in the header is either metric given another unit. */
    check_errmse_rows(lines_with(header, "ErRMSE"), "best_pot.tadah header",
                      header);
    check_srmse_rows(lines_with(header, "SRMSE"), "best_pot.tadah header",
                     header);
}

/* Place 4b. The best_pot_N.tadah archive header
 * (PotHeaderWriter::write_best_archive_header), whose RESULT table is the
 * same write_result_table: ErRMSE rows in per mille, SRMSE rows in eV/A^3.
 * Before the fix: "ErRMSE (meV/at)[...]" and "SRMSE  (GPa)[...]". */
TEST_CASE("metric units: the best_pot archive header labels ErRMSE per "
          "mille and SRMSE eV per cubic Angstrom", "[metric_units]")
{
    ScopedTextFile target(".hpo", kAllMetricsTarget);
    Context ctx = watom_ctx();
    HPOTargets trg(ctx, target.path.string());
    const auto r = all_metrics_result();
    std::ostringstream os;
    tadah::hpo::PotHeaderWriter::write_best_archive_header(os, r, trg,
                                                           /*cw=*/16,
                                                           /*prec=*/6);
    const std::string header = os.str();
    INFO(header);
    const auto errmse = lines_with(header, "ErRMSE");
    const auto srmse = lines_with(header, "SRMSE");
    CHECK(errmse.size() == 3u);
    CHECK(srmse.size() == 3u);
    check_errmse_rows(errmse, "best_pot archive RESULT", header);
    check_srmse_rows(srmse, "best_pot archive RESULT", header);
}

/* Place 5. The --skip-above rejection HPO_Host::operator() writes to
 * hpo_run.log (a WARNING "eval N: skipping downstream (obs+LAMMPS) --
 * <metric>=<value> <unit> > skip-above=<cap>, ..."). A fixed potential is
 * evaluated once with a cap (1e-12) no candidate can meet, on one metric at
 * a time (the checks stop at the first metric that rejects, so a second
 * capped metric would never be reported): the ErRMSE rejection is in per
 * mille, the SRMSE rejection in eV/A^3. The line must name the metric
 * (case-insensitively), so the rejection read is the one asked for.
 * Before the fix: "eRRMSE=<v> meV/at" and "srmse=<v> GPa"
 * (hpo_host_impl.hpp operator(), check_metric). */
TEST_CASE("valmetric e2e: the --skip-above rejection gives ErRMSE in per "
          "mille and SRMSE in eV per cubic Angstrom", "[valmetric_e2e][.]")
{
    SECTION("PC_ErRMSE --skip-above") {
        const std::string journal =
            skip_above_journal("PC_ErRMSE 0 1 --skip-above 1e-12");
        INFO(journal);
        const auto rows = lines_with(journal, "skipping downstream");
        for (const auto& row : rows)
            CHECK_THAT(lower(row), Contains("errmse="));
        check_errmse_rows(rows, "--skip-above rejection (PC_ErRMSE)", journal);
    }
    SECTION("PC_SRMSE --skip-above") {
        const std::string journal =
            skip_above_journal("PC_SRMSE 0 1 --skip-above 1e-12");
        INFO(journal);
        const auto rows = lines_with(journal, "skipping downstream");
        for (const auto& row : rows)
            CHECK_THAT(lower(row), Contains("srmse="));
        check_srmse_rows(rows, "--skip-above rejection (PC_SRMSE)", journal);
    }
}

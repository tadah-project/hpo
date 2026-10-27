// test_pot_header_result.cpp
//
// The LOSS FUNCTION and RESULT sections of the best_pot.tadah header
// (PotHeaderWriter::write_best_header) and the RESULT table of the
// best_pot_N archive header (write_best_archive_header):
//
//   1. a PC_OBSERVABLE term is listed like every other loss term: its
//      target, weight and loss function in LOSS FUNCTION, and a RESULT row
//      with its value and its loss ("Total loss / N" counts it);
//   2. the value and the loss of every RESULT row are separate fields at
//      any precision (DIGITS, 1 to 15; 17 for max_digits10).
//
// The fast cases are [pot_header]:
//   ./test_hpo "[pot_header]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using ParamVec = dlib::matrix<double, 0, 1>;
using Result   = tadah::hpo::EvalResult<ParamVec>;

/* Unique-across-processes temporary names (ctest runs each TEST_CASE as
 * its own process, and in parallel). */
fs::path temp_path(const std::string& stem) {
    static int counter = 0;
    return fs::temp_directory_path() /
           ("test_pot_header_result_" + stem + "_" +
            std::to_string(::getpid()) + "_" + std::to_string(++counter));
}

/* Every file this process writes, removed when it exits. */
struct WrittenFiles {
    std::vector<fs::path> paths;
    ~WrittenFiles() {
        for (const auto& p : paths) {
            std::error_code ec;
            fs::remove(p, ec);
        }
    }
};

fs::path write_file(const std::string& stem, const std::string& body) {
    static WrittenFiles written;
    auto p = temp_path(stem);
    written.paths.push_back(p);
    std::ofstream f(p);
    f << body;
    return p;
}

Context make_context_for_targets() {
    Context c;
    c.add("ATOM",   "Ta");
    c.add("WATOM",  1);
    c.add("INIT2B", "false");
    c.add("INITMB", "false");
    c.add("BIAS",   "false");
    c.add("NORM",   "false");
    c.add("MODEL",  "M_BLR");
    c.add("MODEL",  "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO",  4);
    c.add("FORCE",  "false");
    c.add("STRESS", "false");
    c.add("VERBOSE", 0);
    return c;
}

const char* kHead =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
    "LOSS L2\n";

/* A target file with PC_ERMSE and one PC_OBSERVABLE line. */
std::string observable_target() {
    return std::string(kHead) +
           "PC_ERMSE 0 0.25\n"
           "PC_OBSERVABLE cohesive_energy --species Ta --target 8.11398 "
           "--weight 2.5e5 --hotvar a0 3.1 3.5 3.32011\n";
}

/* A target file with PC_ERMSE and one PC_LAMMPS line (its script defines
 * the variables the line names). */
std::string lammps_target() {
    const fs::path script = write_file(
        "script.in",
        "clear\nrun 0\nvariable a0 equal 3.3\nvariable Ecoh equal 8.1\n");
    return std::string(kHead) + "PC_ERMSE 0 0.25\n" +
           "PC_LAMMPS --script " + script.string() +
           " --outvar a0 --varloss Ecoh 8.11398 2.5e5\n";
}

/* The ERMSE terms of an evaluation (one validation dataset). */
Result ermse_result() {
    Result r;
    r.iter = 7;
    r.best_iter = 3;
    r.ermse = 10.97730325762796;
    r.ermseLoss = 30.12530048346911;
    r.v_ermse = {10.97730325762796};
    r.v_ermseLoss = {30.12530048346911};
    return r;
}

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

/* The whitespace-separated fields of @p line. */
std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream in(line);
    for (std::string f; in >> f;) out.push_back(f);
    return out;
}

/* @p v in scientific notation with @p prec digits after the point. */
std::string sci(double v, int prec) {
    std::ostringstream s;
    s << std::scientific << std::setprecision(prec) << v;
    return s.str();
}

/* The best_pot.tadah header of @p r at precision @p prec. */
std::string best_header(const Result& r, const HPOTargets& trg, int prec) {
    const Context val_ctx{};
    const tadah::mlip::StructureDB train_db{}, val_db{};
    const tadah::mlip::LoadTransformReport report{};
    std::ostringstream os;
    tadah::hpo::PotHeaderWriter::write_best_header(
        os, r, trg, make_context_for_targets(), val_ctx, train_db, val_db, {},
        report, /*cw=*/16, prec);
    return os.str();
}

/* The best_pot_N archive header of @p r at precision @p prec. */
std::string archive_header(const Result& r, const HPOTargets& trg, int prec) {
    std::ostringstream os;
    tadah::hpo::PotHeaderWriter::write_best_archive_header(os, r, trg,
                                                           /*cw=*/16, prec);
    return os.str();
}

/* The one row of @p table that contains @p name ends with the value @p v
 * and the loss @p loss, as two fields written at precision @p prec. */
void check_row(const std::string& table, const std::string& name, double v,
               double loss, int prec) {
    const auto rows = lines_with(table, name);
    INFO("rows with '" << name << "' at precision " << prec << ":\n"
                       << table);
    REQUIRE(rows.size() == 1u);
    const auto f = fields(rows[0]);
    INFO("row: " << rows[0]);
    REQUIRE(f.size() >= 3u);
    CHECK(f[f.size() - 2] == sci(v, prec));
    CHECK(f.back() == sci(loss, prec));
}

}  // namespace

/* Pins: the best_pot.tadah header lists a PC_OBSERVABLE term as it lists
 * the others. LOSS FUNCTION has one row for it, named as the journal names
 * the term, cohesive_energy[1], with its target in eV/atom, its weight and
 * its loss function: "target=8.11398 eV/atom weight=250000 loss=L2".
 * RESULT has one row for it, "Ecoh (eV/atom)[1]", with its value and its
 * loss (8.126325e+00 and 3.809720e+01 at DIGITS 6), next to the ERMSE
 * rows; the RESULT table of the best_pot_N archive has the same row.
 * Before the fix: neither section named the term (LOSS FUNCTION and
 * RESULT listed the ERMSE rows alone), while "Total loss / N" counted
 * it. */
TEST_CASE("best_pot.tadah header: LOSS FUNCTION and RESULT list the "
          "PC_OBSERVABLE term", "[pot_header]")
{
    tadah::core::PeriodicTable::initialize();
    const fs::path target = write_file("observable.hpo", observable_target());
    Context ctx = make_context_for_targets();
    HPOTargets trg(ctx, target.string());
    REQUIRE(trg.observables.size() == 1u);

    Result r = ermse_result();
    r.v_obs_value = {8.126324585067213};
    r.v_obs_loss = {38.09719512041475};
    r.loss = (r.ermseLoss + r.v_obs_loss[0]) / 2.0;

    const std::string header = best_header(r, trg, 6);
    INFO(header);
    const std::string loss = block(header, "# -- LOSS FUNCTION", "\n# -- ");
    REQUIRE_FALSE(loss.empty());
    const auto term = lines_with(loss, "cohesive_energy[1]");
    REQUIRE(term.size() == 1u);
    CHECK_THAT(term[0], Catch::Contains(
        "target=8.11398 eV/atom weight=250000 loss=L2"));

    const std::string result = block(header, "# -- RESULT", "\n# -- ");
    REQUIRE_FALSE(result.empty());
    check_row(result, "Ecoh (eV/atom)[1]", 8.126324585067213,
              38.09719512041475, 6);
    check_row(result, "ERMSE  (meV/at)[avg]", r.ermse, r.ermseLoss, 6);

    const std::string archive = archive_header(r, trg, 6);
    check_row(archive, "Ecoh (eV/atom)[1]", 8.126324585067213,
              38.09719512041475, 6);
}

/* Pins: the value and the loss of every RESULT row are two fields at any
 * precision: at DIGITS 1, 6 and 15 (its range is 1 to 15) and at 17 (the
 * digits of max_digits10), the last two whitespace-separated fields of the
 * ERMSE rows, of a PC_OBSERVABLE row and of a PC_LAMMPS variable row are
 * the value and the loss, each written in scientific notation at that
 * precision, with a negative value and a three-digit exponent among them
 * (a loss of 1.7976931348623157e+308, the largest double). Both the
 * best_pot.tadah header and the best_pot_N archive header are checked.
 * Before the fix: the value and loss columns were 16 characters wide
 * whatever the precision, and at DIGITS 15 and 17 a number filled its
 * column: the row read "1.097730325762796e+013.012530048346911e+01", one
 * field (DIGITS 1 and 6 passed). */
TEST_CASE("best_pot.tadah header: RESULT rows keep the value and the loss "
          "apart at any DIGITS", "[pot_header]")
{
    tadah::core::PeriodicTable::initialize();
    const double huge = std::numeric_limits<double>::max();

    SECTION("PC_ERMSE and PC_OBSERVABLE") {
        const fs::path target =
            write_file("observable.hpo", observable_target());
        Context ctx = make_context_for_targets();
        HPOTargets trg(ctx, target.string());
        Result r = ermse_result();
        r.v_obs_value = {-0.8126324585067213};
        r.v_obs_loss = {huge};
        r.loss = huge;
        for (int prec : {1, 6, 15, 17}) {
            for (const std::string& header :
                 {best_header(r, trg, prec), archive_header(r, trg, prec)}) {
                const std::string result =
                    block(header, "# -- RESULT", "\n# -- ");
                REQUIRE_FALSE(result.empty());
                check_row(result, "ERMSE  (meV/at)[1]", r.v_ermse[0],
                          r.v_ermseLoss[0], prec);
                check_row(result, "ERMSE  (meV/at)[avg]", r.ermse,
                          r.ermseLoss, prec);
                check_row(result, "Ecoh (eV/atom)[1]", r.v_obs_value[0],
                          r.v_obs_loss[0], prec);
            }
        }
    }
    SECTION("PC_ERMSE and PC_LAMMPS") {
        const fs::path target = write_file("lammps.hpo", lammps_target());
        Context ctx = make_context_for_targets();
        HPOTargets trg(ctx, target.string());
        REQUIRE(trg.LS.size() == 1u);
        REQUIRE(trg.LS[0].n_outvar.size() == 2u);
        Result r = ermse_result();
        /* The outvars as the script returns them, in the line's order
         * (a0, then Ecoh); the loss of Ecoh, the --varloss one. */
        const std::size_t ecoh = trg.LS[0].n_outvar[0] == "Ecoh" ? 0 : 1;
        r.v_outvar = {{"", ""}};
        r.v_outvar[0][ecoh] = "-8.1263245850672128";
        r.v_outvar[0][1 - ecoh] = "3.3056775255784179";
        r.v_loss = {{0.0, 0.0}};
        r.v_loss[0][ecoh] = huge;
        r.loss = huge;
        for (int prec : {1, 6, 15, 17}) {
            for (const std::string& header :
                 {best_header(r, trg, prec), archive_header(r, trg, prec)}) {
                const std::string result =
                    block(header, "# -- RESULT", "\n# -- ");
                REQUIRE_FALSE(result.empty());
                check_row(result, "ERMSE  (meV/at)[avg]", r.ermse,
                          r.ermseLoss, prec);
                check_row(result, "Ecoh ", -8.1263245850672128, huge, prec);
                /* a0 has no loss: its row ends with its value and '-'. */
                const auto a0 = lines_with(result, "a0 ");
                REQUIRE(a0.size() == 1u);
                const auto f = fields(a0[0]);
                INFO("row: " << a0[0]);
                REQUIRE(f.size() >= 3u);
                CHECK(f[f.size() - 2] == sci(3.3056775255784179, prec));
                CHECK(f.back() == "-");
            }
        }
    }
}

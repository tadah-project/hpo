// The OPTIMISER section of the best_pot.tadah header
// (PotHeaderWriter::write_best_header) and of the journal
// (HPOLog::write_optimizer_section):
//  - the header's search space gives each OPTIM dimension's real bounds,
//    the box of its OPTIM line, as the journal's "real bounds" column does,
//    and not the optimiser-internal box (the unit box [0, 1] under
//    OPTIM_SCALE AUTO, the default);
//  - a numeric PARAM value is written as the shortest text that reads back
//    as the value ("GRID_POINTS=11", "STEP_FRAC=0.15"), not in the
//    four-decimal scientific form of the losses ("1.1000e+01").
// Before the fix: "WATOM[1]      [0.000e+00, 1.000e+00]" and
// "Params: GRID_POINTS=1.1000e+01" in the header, "GRID_POINTS :
// 1.1000e+01" in the journal.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;
using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using ParamVec = dlib::matrix<double, 0, 1>;

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

struct ScopedTextFile {
    fs::path path;
    explicit ScopedTextFile(const std::string& body)
        : path(fs::temp_directory_path() /
               (unique_name("tadah_pot_header_space_") + ".hpo")) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~ScopedTextFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

Context watom_ctx() {
    Context ctx;
    ctx.add("ATOM", "C");
    ctx.add("ATOM", "H");
    ctx.add("WATOM", "C");
    ctx.add("WATOM", "H");
    ctx.add("WATOM", 1.0);
    ctx.add("WATOM", 2.0);
    return ctx;
}

/* The best_pot.tadah header of a run of `target` (one evaluation result). */
std::string best_header(const std::string& target) {
    ScopedTextFile f(target);
    Context ctx = watom_ctx();
    HPOTargets trg(ctx, f.path.string());
    tadah::hpo::EvalResult<ParamVec> r;
    r.iter = 2;
    r.best_iter = 1;
    r.loss = 1.25;
    r.ermse = 2.0;
    r.ermseLoss = 4.0;
    r.params.set_size(static_cast<long>(trg.spec.low.size()));
    for (long i = 0; i < r.params.size(); ++i) r.params(i) = 0.5;
    const Context val_ctx{};
    const tadah::mlip::StructureDB train_db{}, val_db{};
    const tadah::mlip::LoadTransformReport report{};
    std::ostringstream os;
    tadah::hpo::PotHeaderWriter::write_best_header(
        os, r, trg, ctx, val_ctx, train_db, val_db, trg.spec.parameter_labels,
        report, /*cw=*/16, /*prec=*/6);
    return os.str();
}

/* The journal's OPTIMISER and SEARCH SPACE sections of `target`. */
std::string journal_optimizer(const std::string& target) {
    ScopedTextFile f(target);
    Context ctx = watom_ctx();
    HPOTargets trg(ctx, f.path.string());
    const fs::path log = fs::temp_directory_path() /
                         (unique_name("tadah_pot_header_space_log_") + ".log");
    if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
    HPOLog::instance().reset_for_test();
    HPOLog::instance().open(log.string(), "test", "test_cwd", 0);
    HPOLog::instance().write_optimizer_section(trg.spec,
                                               trg.spec.parameter_labels);
    HPOLog::instance().finish("TEST");
    HPOLog::instance().reset_for_test();
    const std::string text = read_text(log);
    std::error_code ec;
    fs::remove(log, ec);
    return text;
}

const std::string kGrid =
    "OPTIM WATOM (1) 0.1 5.0\n"
    "OPTIM WATOM (2) 0.2 4.0 LOG\n"
    "OPTIMIZER\nLIB TADAH\nALGO GRID\nPARAM GRID_POINTS 11\nENDOPTIMIZER\n"
    "PC_ERMSE 0 1\n";

}  // namespace

TEST_CASE("best_pot.tadah header: the search space in real bounds",
          "[HPO][pot_header]")
{
    const std::string header = best_header(kGrid);
    INFO(header);
    CHECK_THAT(header, Contains("WATOM[1]      [1.000e-01, 5.000e+00]"));
    CHECK_THAT(header, Contains("WATOM[2]      [2.000e-01, 4.000e+00]"));
    CHECK(header.find("[0.000e+00, 1.000e+00]") == std::string::npos);
}

TEST_CASE("best_pot.tadah header and journal: a numeric PARAM value as "
          "written", "[HPO][pot_header]")
{
    const std::string header = best_header(kGrid);
    INFO(header);
    CHECK_THAT(header, Contains("Params: GRID_POINTS=11\n"));
    CHECK(header.find("1.1000e+01") == std::string::npos);

    const std::string anneal = best_header(
        "OPTIM WATOM (1) 0.1 5.0\n"
        "OPTIMIZER\nLIB TADAH\nALGO ANNEAL\nMAXEVAL 10\n"
        "PARAM STEP_FRAC 0.15\nPARAM T0_PROBES 12\nENDOPTIMIZER\n"
        "PC_ERMSE 0 1\n");
    INFO(anneal);
    CHECK_THAT(anneal, Contains("STEP_FRAC=0.15"));
    CHECK_THAT(anneal, Contains("T0_PROBES=12"));

    const std::string journal = journal_optimizer(kGrid);
    INFO(journal);
    CHECK_THAT(journal, Contains("GRID_POINTS         : 11\n"));
    CHECK(journal.find("1.1000e+01") == std::string::npos);
}

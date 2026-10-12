// test_startup_validation_dbfile.cpp
//
// validate_hpo_setup() refuses at startup a training configuration whose
// DBFILE lists one dataset twice: StructureDB::add loads every entry as a
// dataset, so each structure of that file would be fitted twice by every
// evaluation. (An M_BLR potential written before the potential-content fix
// records the DBFILE it was trained on; used as the -c configuration with
// its training data added, it listed the dataset twice.)

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
namespace fs = std::filesystem;

/* A scratch directory with two dataset files, a.db and b.db (validation
 * never reads them), and a symbolic link to a.db, link.db. */
struct ScratchDatasets {
  fs::path dir;
  ScratchDatasets()
      : dir(fs::temp_directory_path() /
            ("tadah_sv_dbfile_" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()))) {
    fs::create_directories(dir);
    std::ofstream(dir / "a.db") << "a\n";
    std::ofstream(dir / "b.db") << "b\n";
    fs::create_symlink(dir / "a.db", dir / "link.db");
  }
  ~ScratchDatasets() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string operator()(const std::string &name) const {
    return (dir / name).string();
  }
};

/* The HPO target file: one OPTIM and a validation-set loss. */
struct ScopedTarget {
  fs::path path;
  ScopedTarget()
      : path(fs::temp_directory_path() /
             ("tadah_sv_dbfile_" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()) +
              ".hpo")) {
    std::ofstream(path) << "OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n"
                           "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                           "ENDOPTIMIZER\n";
  }
  ~ScopedTarget() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

/* A D2_BP + BF_Linear training configuration with the given DBFILE
 * values. */
Context model_ctx(const std::vector<std::string> &dbfiles) {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1.0);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "0.5 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  for (const std::string &f : dbfiles) c.add("DBFILE", f);
  return c;
}

}  // namespace

/* Pins: validate_hpo_setup refuses a training DBFILE that lists one
 * dataset twice, naming both entries ("DBFILE lists one dataset twice:
 * entry 1 (<a.db>) and entry 2 (<...>)"): the same path twice, a.db and
 * <dir>/./a.db, a.db and a symbolic link to it, and a path that does not
 * exist given twice; a.db and b.db pass, as does a.db alone.
 * Before the fix: every one passed, and every evaluation trained on the
 * structures of a.db twice. (The two files and a.db alone passed then as
 * well: a pin.) */
TEST_CASE("startup validation: a training DBFILE that lists one dataset "
          "twice is refused",
          "[startup_validation][dbfile][potential_content]") {
  PeriodicTable::initialize();
  ScratchDatasets ds;
  ScopedTarget tf;
  const std::string a = ds("a.db");
  const struct {
    const char *name;
    std::vector<std::string> dbfiles;
  } twice[] = {
      {"the same path twice", {a, a}},
      {"two spellings of one path", {a, (ds.dir / "." / "a.db").string()}},
      {"a file and a symbolic link to it", {a, ds("link.db")}},
      {"a missing file twice", {ds("missing.db"), ds("missing.db")}},
  };
  for (const auto &c : twice) DYNAMIC_SECTION(c.name) {
    Context ctx = model_ctx(c.dbfiles);
    HPOTargets trg(ctx, tf.path.string());
    CHECK_THROWS_WITH(
        validate_hpo_setup(ctx, trg, true, true),
        Catch::Contains("DBFILE lists one dataset twice: entry 1 (" +
                        c.dbfiles[0] + ") and entry 2 (" + c.dbfiles[1] +
                        ")"));
  }
  SECTION("two datasets, and one") {
    for (const std::vector<std::string> &files :
         {std::vector<std::string>{a, ds("b.db")}, std::vector<std::string>{a}}) {
      Context ctx = model_ctx(files);
      HPOTargets trg(ctx, tf.path.string());
      CHECK_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    }
  }
}

// Potential-file precision of tadah hpo.
//
// HPOTargets resolves NUMERIC once, WITHOUT default insertion, into
// pot_digits: the significant digits of the non-integer numbers that
// best_pot.tadah and the BDUMP/DUMP archives record (Context::write_potential),
// except the numbers of the keys the blueprint marks exact_in_potential, the
// cutoffs and radial grids (RCUT2B, RCUTMB, CGRID2B, CGRIDMB, SGRID2B,
// SGRIDMB), which are written exactly whatever pot_digits is, since a load
// checks them exactly. Absent -> 18, an exact round trip. An explicit
// NUMERIC below 17 -- notably the 'NUMERIC 12' line every tadah 1.5.0 hpo
// best_pot.tadah carries, which reaches the next run when such a pot is
// used as its configuration -- is honoured but warned about in
// hpo_run.log, naming the line to delete.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

namespace fs = std::filesystem;

std::string unique_stem(const char* prefix) {
  return std::string(prefix) +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count());
}

struct ScopedTempFile {
  fs::path path;
  explicit ScopedTempFile(const std::string& body)
  : path(fs::temp_directory_path() / (unique_stem("tadah_pot_digits_") + ".in"))
  {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~ScopedTempFile() { std::error_code ec; fs::remove(path, ec); }
};

/* A journal opened for one test case; returns its body at close. */
struct ScopedJournal {
  fs::path path;
  ScopedJournal()
  : path(fs::temp_directory_path() / (unique_stem("tadah_pot_digits_") + ".log"))
  {
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());
  }
  std::string close_and_read() {
    tadah::hpo::HPOLog::instance().finish("TEST_DONE");
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  }
  ~ScopedJournal() {
    tadah::hpo::HPOLog::instance().reset_for_test();
    std::error_code ec;
    fs::remove(path, ec);
  }
};

tadah::core::Context watom_ctx() {
  tadah::core::Context ctx;
  ctx.add("ATOM", "C"); ctx.add("ATOM", "H");
  ctx.add("WATOM", "C"); ctx.add("WATOM", "H"); ctx.add("WATOM", 1.0);
  return ctx;
}

const char* kTarget =
    "OPTIM WATOM (1) 0.1 5.0\n"
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

}  // namespace

TEST_CASE("HPOTargets: NUMERIC absent -> 18 significant digits, nothing "
          "inserted, no warning", "[HPO][numeric][lossless]")
{
  ScopedJournal journal;
  ScopedTempFile in(kTarget);
  auto ctx = watom_ctx();
  tadah::hpo::HPOTargets trg(ctx, in.path.string());
  CHECK(trg.pot_digits == 18);
  CHECK(trg.pot_digits >= 17);          // must round-trip every double
  CHECK_FALSE(trg.pot_digits_explicit);
  CHECK_FALSE(ctx.has_stored("NUMERIC"));
  const std::string body = journal.close_and_read();
  CHECK(body.find("limits every non-integer number") == std::string::npos);
}

/* Pins: NUMERIC 12 gives pot_digits 12, and hpo_run.log has a WARNING that
 * NUMERIC 12 limits every non-integer number of the potential's key/value
 * lines, that integers are written exactly and so are the numbers of
 * CGRID2B, CGRIDMB, RCUT2B, RCUTMB, SGRID2B and SGRIDMB, and that a 1.5.0
 * best_pot.tadah carries the line to delete.
 * Before the fix: the WARNING said only that integers are written exactly
 * ('(integers are written exactly): values do not round-trip ...'), naming
 * no key; the other checks pass as well (a pin). */
TEST_CASE("HPOTargets: the legacy 'NUMERIC 12' is honoured with a WARNING "
          "naming the fix", "[HPO][numeric][lossless]")
{
  ScopedJournal journal;
  ScopedTempFile in(kTarget);
  auto ctx = watom_ctx();
  ctx.add("NUMERIC", 12);
  tadah::hpo::HPOTargets trg(ctx, in.path.string());
  CHECK(trg.pot_digits == 12);
  CHECK(trg.pot_digits_explicit);
  const std::string body = journal.close_and_read();
  CHECK(body.find("WARNING") != std::string::npos);
  CHECK(body.find("NUMERIC 12 limits every non-integer number of the "
                  "potential's key/value lines") != std::string::npos);
  CHECK(body.find("integers are written exactly") != std::string::npos);
  CHECK(body.find("and so are the numbers of CGRID2B, CGRIDMB, RCUT2B, "
                  "RCUTMB, SGRID2B and SGRIDMB, which a load checks "
                  "exactly") != std::string::npos);
  CHECK(body.find("best_pot.tadah") != std::string::npos);
  CHECK(body.find("tadah 1.5.0 hpo") != std::string::npos);
  CHECK(body.find("delete that line") != std::string::npos);
}

TEST_CASE("HPOTargets: an explicit NUMERIC >= 17 is exact and silent",
          "[HPO][numeric][lossless]")
{
  for (int n : {17, 18, 25}) {
    INFO("NUMERIC " << n);
    ScopedJournal journal;
    ScopedTempFile in(kTarget);
    auto ctx = watom_ctx();
    ctx.add("NUMERIC", n);
    tadah::hpo::HPOTargets trg(ctx, in.path.string());
    CHECK(trg.pot_digits == n);
    CHECK(trg.pot_digits_explicit);
    const std::string body = journal.close_and_read();
    CHECK(body.find("limits every non-integer number") == std::string::npos);
  }
}

// `tadah refit` compact mode in every RHOFORM, end to end through the HPO
// refit engine (the compact fitter's ladders, FUSE and the EAMRefit
// handoff), on a synthesised single-element setfl with small optimiser
// budgets. The density is fitted in the declared form: SIGNED (the fit
// before RHOFORM), NONNEG (every amplitude >= 0) and MONOTONE (integrated-bump steps,
// every amplitude >= 0).
#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/engines/hpo_refit_engine.h>
#include <tadah/mlip/task.h>
#include <tadah/models/descriptors/dm/dm_ream.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Context;
using tadah::hpo::engines::HPORefitEngine;
using tadah::mlip::Task;
using tadah::models::RhoForm;
namespace fs = std::filesystem;

namespace {
struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

// The MLIP refit tests' synthetic fcc setfl: rho(r) = (1 - r/rcut)^2 (or
// rho_fn), phi(r) = 3 (1 - r/rcut)^3, F(rho) = -2 sqrt(rho) + 0.05 rho, rcut 5.
void write_synth_eam(const std::string &path,
                     const std::function<double(double)> &rho_fn = nullptr) {
  const int nr = 400, nrho = 400;
  const double rcut = 5.0, dr = rcut / (nr - 1);
  const double rhomax = 60.0, drho = rhomax / (nrho - 1);
  std::ofstream os(path);
  os << "# synthetic single-element EAM (compact refit RHOFORM test)\n"
     << "# comment 2\n# comment 3\n"
     << "1 Cu\n"
     << nrho << " " << drho << " " << nr << " " << dr << " " << rcut << "\n"
     << "29 63.55 3.6 fcc\n";
  os << std::setprecision(12);
  for (int i = 0; i < nrho; ++i) {
    const double rho = i * drho;
    os << (-2.0 * std::sqrt(rho) + 0.05 * rho) << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
  for (int i = 0; i < nr; ++i) {
    const double x = 1.0 - i * dr / rcut;
    os << (rho_fn ? rho_fn(i * dr) : x > 0 ? x * x : 0.0)
       << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
  for (int i = 0; i < nr; ++i) {
    const double r = i * dr, x = 1.0 - r / rcut;
    os << ((x > 0 ? 3.0 * x * x * x : 0.0) * r) << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
}

struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_compact_rhoform_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScopedTempDir() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

// A stage block as CORE's config parser deposits it (one STRINGS token each).
void add_block(Context &c, const std::string &key, const std::string &body) {
  std::istringstream iss(body);
  std::string t;
  while (iss >> t) c.add(key, t);
}

// Compact Blip5 refit of synth.eam.alloy with small budgets (seconds).
Context compact_context(const std::string &form) {
  Context c;
  c.add("EAM", std::string("synth.eam.alloy"));
  c.add("RHOFORM", form);
  c.add("REFIT_MODE", std::string("compact"));
  c.add("REFIT_BASIS", std::string("Blip5"));
  c.add("REFIT_COMPACT_NMAX", 8);
  c.add("REFIT_COMPACT_JITTERS", 0);
  c.add("REFIT_COMPACT_STALL", 3);
  c.add("REFIT_COMPACT_SAMPLES", 200);
  c.add("REFIT_TOL_AMBIENT", 5.0);
  c.add("REFIT_TOL_COMPRESSED", 20.0);
  add_block(c, "EXPLORE",
            "OPTIMIZER LIB NLOPT ALGO GN_MLSL_LDS MAXEVAL 4000 "
            "INNER LIB NLOPT ALGO LN_SBPLX MAXEVAL 1000 ENDINNER ENDOPTIMIZER");
  add_block(c, "REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 100 ENDOPTIMIZER");
  add_block(c, "FUSE",
            "OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 200 ENDOPTIMIZER");
  return c;
}

std::string slurp(const std::string &f) {
  std::ifstream in(f);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The crystal self-test's max error (meV/atom) from refit_opt.log's
// "    <label>    RMSE=... max=..." line (label ambient | compressed).
double logged_max(const std::string &log, const std::string &label) {
  const auto at = log.find("\n    " + label + " ");
  REQUIRE(at != std::string::npos);
  const auto m = log.find("max=", at);
  REQUIRE(m != std::string::npos);
  return std::stod(log.substr(m + 4));
}

// Runs the HPO refit engine on `c` with INFO logging captured; returns the
// exception text ("" when it succeeded) and the captured log.
std::pair<std::string, std::string> run_compact(const Context &c) {
  Task task("refit", c);
  auto &logger = tadah::core::Logger::instance();
  const auto old_level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  int rc = EXIT_FAILURE;
  try {
    HPORefitEngine engine;
    rc = engine.run(task);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  logger.setLogLevel(old_level);
  if (what.empty()) REQUIRE(rc == EXIT_SUCCESS);
  return {what, captured.str()};
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact fits, self-tests and writes every "
                 "RHOFORM", "[compact][refit][rhoform]") {
  for (const char *form : {"SIGNED", "NONNEG", "MONOTONE"}) {
    INFO("RHOFORM " << form);
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    const auto [what, logged] = run_compact(compact_context(form));
    INFO(logged);
    REQUIRE(what.empty());
    const RhoForm f = tadah::models::parse_rhoform(form);

    // the refit log records the form, and the compact path ran
    const std::string olog = slurp("refit_opt.log");
    CHECK(olog.find("compact mode") != std::string::npos);
    CHECK(olog.find(std::string("density: RHOFORM ") + form + " — ") !=
          std::string::npos);
    // SIGNED names the release that introduced the forms
    if (f == RhoForm::Signed)
      CHECK(olog.find("with signed amplitudes (the density before tadah "
                      "1.6.0)") != std::string::npos);
    CHECK(olog.find("self-test (crystal E-V") != std::string::npos);

    // The joint refinement optimises the radial bumps in their outer support
    // edges, bounded at rcut, so the post-FUSE bump-support projection has
    // nothing to move and the result is the joint optimum in every form: no
    // linear re-solve, no held re-run, and the crystal self-test within its
    // tolerances (5 / 20 meV/atom). With the old post-solve projection on
    // this table SIGNED failed the self-test, and the positive forms needed
    // the held re-run, which traded the density away.
    CHECK(olog.find("back inside rcut") == std::string::npos);
    CHECK(olog.find("moved bumps held") == std::string::npos);
    CHECK(logged_max(olog, "ambient") <= 5.0);
    CHECK(logged_max(olog, "compressed") <= 20.0);

    // the written potential declares the form and loads in it
    Context pot = tadah::core::CoreUserConfigParser().parse("pot.tadah");
    REQUIRE(pot.has_stored("RHOFORM"));
    CHECK(tadah::models::stored_rhoform(pot) == f);
    std::vector<double> amps(pot.size("AMPGRIDMB"));
    pot.get("AMPGRIDMB", amps);
    REQUIRE_FALSE(amps.empty());
    if (tadah::models::rhoform_is_positive(f))
      for (double a : amps) REQUIRE(a >= 0.0);
    pot.add("VERBOSE", 0);
    tadah::models::DM_REAM dm(pot);
    REQUIRE(dm.density().form == f);

    // the density the potential evaluates (Cut_Dummy: f_c = 1 below rcut)
    // follows the table (1 - r/5)^2 on the sampled shells, and carries the
    // form's guarantees on [0, rcut)
    double err = 0.0, prev = dm.density().value(0.0, 1.0), term_sum = 0.0;
    for (double a : amps) term_sum += std::fabs(a);
    int rises = 0;
    for (int i = 1; i < 5000; ++i) {
      const double r = 5.0 * i / 5000.0;
      const double v = dm.density().value(r, 1.0);
      if (r >= 2.2) {
        const double x = 1.0 - r / 5.0;
        err = std::max(err, std::fabs(v - x * x));
      }
      if (f != RhoForm::Signed) REQUIRE(v >= 0.0);
      if (v - prev > 8.0 * std::numeric_limits<double>::epsilon() * term_sum)
        ++rises;
      prev = v;
    }
    INFO("max |rho - rho_tab| on [2.2, 5) = " << err << ", rises = " << rises);
    // A coarse fit (small budgets; |rho| <= 0.31 there), in every form.
    CHECK(err < 1e-2);
    if (f == RhoForm::Monotone) CHECK(rises == 0);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: the RHOFORM table pre-checks and the "
                 "MONOTONE INIT width rule", "[compact][refit][rhoform]") {
  SECTION("NONNEG refuses a negative table before any fitting") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy", [](double r) {
      const double x = 1.0 - r / 5.0;
      return x * x - 0.02;
    });
    const auto [what, logged] = run_compact(compact_context("NONNEG"));
    CHECK(what.find("RHOFORM NONNEG cannot fit this table") !=
          std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
    CHECK(slurp("refit_opt.log").find("---- phi ladder ----") ==
          std::string::npos);
  }
  SECTION("MONOTONE refuses a rising table before any fitting") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy", [](double r) {
      const double x = 1.0 - r / 5.0, g = (r - 3.5) / 0.2;
      return x * x + 0.05 * std::exp(-g * g);
    });
    const auto [what, logged] = run_compact(compact_context("MONOTONE"));
    CHECK(what.find("RHOFORM MONOTONE cannot fit this table: the tabulated "
                    "density rises") != std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
  SECTION("MONOTONE: a density INIT anchor with eta <= 0 is an error") {
    ScopedTempDir tmp;
    write_synth_eam("synth.eam.alloy");
    for (const char *eta : {"-0.5", "0"}) {
      INFO("eta " << eta);
      Context c = compact_context("MONOTONE");
      add_block(c, "INIT", std::string("RHO MANUAL 3.0 ") + eta + " PIN");
      const auto [what, logged] = run_compact(c);
      CHECK(what.find("tadah refit INIT [rho]: the basis at mu 3") !=
            std::string::npos);
      CHECK(what.find("RHOFORM MONOTONE fits the density in integrated-bump "
                      "steps, which need eta > 0") != std::string::npos);
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit compact: a radial INIT anchor with eta <= 0 is "
                 "an error in every RHOFORM",
                 "[compact][refit][rhoform][bump_width]") {
  // Owner decision (2026-09-29, C8): a radial bump needs a width > 0 in
  // every form. INIT PHI MANUAL 5.0 -1.0 PIN was fitted as its mirror (the
  // bump is even in eta) and the bump-support checks skipped it.
  ScopedTempDir tmp;
  write_synth_eam("synth.eam.alloy");
  for (const char *form : {"SIGNED", "NONNEG"}) {
    for (const char *ch : {"PHI", "RHO"}) {
      // the value as written, and as the message prints it (shortest
      // round trip)
      for (const auto &[eta, shown] :
           {std::pair<const char *, const char *>{"-1.0", "-1"},
            std::pair<const char *, const char *>{"0", "0"}}) {
        INFO(form << " INIT " << ch << " eta " << eta);
        Context c = compact_context(form);
        add_block(c, "INIT", std::string(ch) + " MANUAL 3.0 2.0 " + ch +
                                 " MANUAL 3.5 " + eta + " PIN");
        const auto [what, logged] = run_compact(c);
        INFO("error: " << what);
        const std::string lower = std::string(ch) == "PHI" ? "phi" : "rho";
        CHECK(what.find("tadah refit INIT [" + lower +
                        "]: the basis at mu 3.5 (INIT " + ch +
                        " anchor 2) has eta " + shown + " <= 0") !=
              std::string::npos);
        CHECK(what.find("a radial bump needs a width eta > 0") !=
              std::string::npos);
        CHECK_FALSE(fs::exists("pot.tadah"));
      }
    }
  }
}

#endif  // TADAH_USE_CERES

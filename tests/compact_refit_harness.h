#pragma once
/* The `tadah refit` compact-mode harness of the end-to-end refit suites
 * (test_compact_refit_rhoform.cpp, test_compact_refit_stages.cpp): a
 * synthesised single-element setfl, a scratch working directory, the
 * Blip5 refit configuration with small optimiser budgets and a run of the
 * HPO refit engine with the INFO log captured. Header-only; include it
 * after catch2/catch.hpp, in a TADAH_USE_CERES build. */

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/engines/hpo_refit_engine.h>
#include <tadah/mlip/task.h>

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

namespace compact_refit_test {

namespace fs = std::filesystem;
using tadah::core::Context;

struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

// The MLIP refit tests' synthetic fcc setfl: rho(r) = (1 - r/rcut)^2 (or
// rho_fn), phi(r) = 3 (1 - r/rcut)^3, F(rho) = -2 sqrt(rho) + 0.05 rho, rcut 5.
inline void write_synth_eam(const std::string &path,
                            const std::function<double(double)> &rho_fn =
                                nullptr) {
  const int nr = 400, nrho = 400;
  const double rcut = 5.0, dr = rcut / (nr - 1);
  const double rhomax = 60.0, drho = rhomax / (nrho - 1);
  std::ofstream os(path);
  os << "# synthetic single-element EAM (compact refit test)\n"
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

// A fresh working directory for the life of the object (the refit writes
// its outputs into the current directory).
struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_compact_refit_" + std::to_string(::getpid()) + "_" +
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
inline void add_block(Context &c, const std::string &key,
                      const std::string &body) {
  std::istringstream iss(body);
  std::string t;
  while (iss >> t) c.add(key, t);
}

constexpr const char *kDefaultRefine =
    "OPTIMIZER LIB CERES ALGO LM MAXEVAL 100 ENDOPTIMIZER";
constexpr const char *kDefaultFuse =
    "OPTIMIZER LIB CERES ALGO DOGLEG MAXEVAL 200 ENDOPTIMIZER";

// Compact Blip5 refit of synth.eam.alloy with small budgets (seconds). The
// REFINE and FUSE blocks are `refine` and `fuse`; an empty one leaves the
// block out (the stage runs with its defaults).
inline Context compact_context(const std::string &form,
                               const std::string &refine = kDefaultRefine,
                               const std::string &fuse = kDefaultFuse) {
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
  if (!refine.empty()) add_block(c, "REFINE", refine);
  if (!fuse.empty()) add_block(c, "FUSE", fuse);
  return c;
}

inline std::string slurp(const std::string &f) {
  std::ifstream in(f);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Runs the HPO refit engine on `c` with INFO logging captured; returns the
// exception text ("" when it succeeded) and the captured log.
inline std::pair<std::string, std::string> run_compact(const Context &c) {
  tadah::mlip::Task task("refit", c);
  auto &logger = tadah::core::Logger::instance();
  const auto old_level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  int rc = EXIT_FAILURE;
  try {
    tadah::hpo::engines::HPORefitEngine engine;
    rc = engine.run(task);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  logger.setLogLevel(old_level);
  if (what.empty()) REQUIRE(rc == EXIT_SUCCESS);
  return {what, captured.str()};
}

}  // namespace compact_refit_test

#pragma once
/* Shared fixtures of the DM_REAM density suites (test_hpo_rhoform.cpp,
 * test_rho_gates.cpp): temporary files and directories, hpo_run.log capture,
 * the D2_Blip5 + DM_REAM Blip5 model with every weight pinned, a WARM seed
 * potential and a four-atom bcc Ta database. Header-only; include it after
 * catch2/catch.hpp. */

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace rhoform_test {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

/* @p stem, this process id and a steady-clock count: distinct across the
 * parallel ctest processes. */
inline std::string unique_name(const std::string& stem) {
  return stem + std::to_string(::getpid()) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary file holding @p body, removed with the object. */
struct TempFile {
  fs::path path;
  explicit TempFile(const std::string& body,
                    const std::string& stem = "tadah_rhoform_")
  : path(fs::temp_directory_path() / (unique_name(stem) + ".in")) {
    std::ofstream os(path);
    REQUIRE(os.good());
    os << body;
  }
  ~TempFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

/* hpo_run.log in a temporary file, read back after finish(). */
struct Journal {
  fs::path path;
  explicit Journal(const std::string& stem = "tadah_rhoform_log_")
  : path(fs::temp_directory_path() / (unique_name(stem) + ".log")) {
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

/* A temporary working directory, the cwd while the object lives. */
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

constexpr const char* kOptimizer =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

/* D2_Blip5 (4) + DM_REAM Blip5 (3 density bumps c 2.8 3.6 4.4, s 3; 5
 * embedding bumps), Cut_Dummy at 5.5, all 9 weights pinned (the
 * test_hpo_embf0_e2e model), in the density form `form` (nullptr: no
 * RHOFORM) with the density amplitudes `amps`. Valid under every form for
 * non-negative amplitudes. */
inline Context make_ream_ctx(const char* form,
                             const char* amps = "0.5 0.3 0.1") {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip5");
  c.add("TYPE2B", 4);
  c.add("TYPE2B", 4);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.5 3.2 3.9 4.5");
  c.add("SGRID2B", "3.0 3.0 3.0 3.0");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 3);
  c.add("TYPEMB", 3); c.add("TYPEMB", 3); c.add("TYPEMB", 3);
  c.add("TYPEMB", 5); c.add("TYPEMB", 5);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.8 3.6 4.4");
  c.add("SGRIDMB", "3.0 3.0 3.0");
  c.add("AMPGRIDMB", amps);
  if (form) c.add("RHOFORM", form);
  c.add("CEMBFUNC", "-0.2 0.3 0.8 1.3 1.8");
  c.add("SEMBFUNC", "2.0 2.0 2.0 2.0 2.0");
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("RCTYPEMB", "Cut_Dummy");
  c.add("RCUT2B", 5.5);
  c.add("RCUTMB", 5.5);
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
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  c.add("FIXINDEX", "1-9");
  for (double w : {0.8, -0.3, 0.1, -0.05, -2.0, 1.5, -0.7, 0.4, 0.2})
    c.add("FIXWEIGHT", w);
  return c;
}

/* The error HPOTargets(ctx, <target>) throws ("" when it does not). */
inline std::string targets_error(Context ctx, const std::string& target) {
  TempFile tf(target);
  try {
    HPOTargets trg(ctx, tf.path.string());
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

/* A DM_REAM seed potential for WARM: the density keys of make_ream_ctx
 * with the amplitudes `amps`, and `rhoform_line` verbatim ("" for none). */
inline std::string ream_seed(const std::string& amps,
                             const std::string& rhoform_line = "RHOFORM SIGNED\n") {
  return "INITMB true\nTYPEMB DM_REAM 3 3 3 3 5 5 0 0 Ta Ta\n"
         "CGRIDMB 2.8 3.6 4.4\nSGRIDMB 3.0 3.0 3.0\n"
         "AMPGRIDMB " + amps + "\n"
         "CEMBFUNC -0.2 0.3 0.8 1.3 1.8\n" + rhoform_line;
}

/* One bcc Ta structure of four atoms in the tadah DB format. */
inline void write_ta_db(const fs::path& path) {
  std::ofstream os(path);
  os << "bcc_ta\n1.0 1.0 1.0\n-8.1 0.0\n"
     << "3.304 0.0 0.0\n0.0 3.304 0.0\n0.0 0.0 6.608\n"
     << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
     << "Ta 0.0 0.0 0.0  0.0 0.0 0.0\n"
     << "Ta 1.652 1.652 1.652  0.0 0.0 0.0\n"
     << "Ta 0.0 0.0 3.304  0.0 0.0 0.0\n"
     << "Ta 1.652 1.652 4.956  0.0 0.0 0.0\n\n";
}

}  // namespace rhoform_test

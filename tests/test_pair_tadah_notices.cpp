// test_pair_tadah_notices.cpp
//
// The notices of the keys a potential carries that have no effect, through
// the real pair_style tadah (LammpsRunner): a deprecated key (SIGMA, the
// weight covariance of Tadah! 1.0.0 to 1.5.0 and the 1.6.0 development
// builds; MBLOCK, a key of the removed MPI training build) and EMBF0 on a
// potential without a DM_REAM block.
// pair_coeff writes each as a LAMMPS WARNING (error->warning on rank 0:
// the screen and log.lammps), not to standard error, where LAMMPS' log
// never saw it and every MPI rank wrote it once. It does so each time it
// loads such a potential, so every LAMMPS instance's log.lammps has it,
// also when the process loaded it before (core::IgnoredKeyNoticeCapture
// receives a key once per capture, whatever the process reported). And a
// potential with the covariance of an OALGO 3 fit (WCOV, WNOISE) loads.
//
// The potential is the D2_LJ pair potential of Tadah.LAMMPS'
// tests_data/pots/ta_lj_cutcos.tadah.

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

const char *kPairPotential =
    "ATOM     Ta\n"
    "BIAS     false\n"
    "DIMER     false     0     false\n"
    "EWEIGHT     1.0\n"
    "FWEIGHT     1.0\n"
    "INIT2B     true\n"
    "INITMB     false\n"
    "LAMBDA     0\n"
    "MODEL     M_KRR     Kern_Linear     1\n"
    "NORM     false\n"
    "OALGO     3\n"
    "RCTYPE2B     Cut_Cos\n"
    "RCUT2B     6.5\n"
    "SWEIGHT     1.0\n"
    "TYPE2B     D2_LJ     Ta     Ta\n"
    "WATOM     73\n"
    "WEIGHTS     14.204213349318765225     2604.396965191186154698\n";

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/* The lines of @p text that contain every one of @p needles. */
std::vector<std::string> lines_with(const std::string &text,
                                    const std::vector<std::string> &needles) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) {
    bool all = true;
    for (const auto &n : needles) all = all && line.find(n) != std::string::npos;
    if (all) out.push_back(line);
  }
  return out;
}

struct Run {
  bool ok = false;
  std::string log;  // log.lammps
  std::string err;  // standard error during the run
  double pe = 0.0;
};

/* Two Ta atoms 2.9 A apart; pair_coeff * * <the potential file> Ta (the
 * plain LAMMPS user path), log.lammps written to a file, no screen. */
Run run_with_potential(const std::string &extra, const std::string &tag) {
  tadah::core::PeriodicTable::initialize();
  const std::string stem =
      "pair_notice_" + std::to_string(::getpid()) + "_" + tag;
  const std::string pot = stem + ".tadah", script = stem + ".in",
                    log = stem + ".log";
  std::ofstream(pot) << kPairPotential << extra;
  {
    std::ofstream ofs(script);
    for (const char *line :
         {"units metal", "boundary p p p", "atom_style atomic",
          "atom_modify map array",
          "region box block 0 30 0 30 0 30 units box", "create_box 1 box",
          "create_atoms 1 single 10 10 10 units box",
          "create_atoms 1 single 12.9 10 10 units box",
          "mass 1 180.94788", "pair_style tadah"})
      ofs << line << "\n";
    ofs << "pair_coeff * * " << pot << " Ta\n";
    ofs << "neighbor 2.0 nsq\nvariable e equal pe\nrun 0 post no\n";
  }
  Run r;
  std::ostringstream err;
  std::streambuf *old = std::cerr.rdbuf(err.rdbuf());
  {
    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", log, "-screen", "none"});
    if (lmp) {
      r.ok = runner.run_script(lmp, script);
      if (r.ok) r.pe = std::stod(runner.get_variable(lmp, "e"));
    }
  }  // LAMMPS destroyed: log.lammps flushed and closed
  std::cerr.rdbuf(old);
  r.log = slurp(log);
  r.err = err.str();
  std::remove(pot.c_str());
  std::remove(script.c_str());
  std::remove(log.c_str());
  return r;
}

}  // namespace

/* Pins: pair_coeff with a potential that carries SIGMA and MBLOCK
 * (deprecated keys) writes each key's notice to log.lammps once, as a
 * LAMMPS WARNING naming the potential ("WARNING: pair tadah: potential
 * '<file>': Deprecated key SIGMA is ignored: ...", likewise MBLOCK), and
 * nothing of it to standard error; the run succeeds.
 * Before the fix: the notices went to standard error only ("WARNING:
 * Deprecated key MBLOCK is ignored: ..."), once per process (on every MPI
 * rank), and log.lammps had none; and SIGMA was stored, with no notice.
 * Before the review fix (once per capture): log.lammps had each notice
 * only when the process had not reported the key before (the count was
 * 1, or 0 after such a report). */
TEST_CASE("pair_coeff writes a deprecated key's notice to log.lammps",
          "[pair_tadah][notices]") {
  const std::size_t n_sigma = 1;
  const std::size_t n_mblock = 1;
  const Run r = run_with_potential("SIGMA     2     1 0 0 1\nMBLOCK     64\n",
                                   "deprecated");
  INFO("log.lammps:\n" << r.log << "\nstderr:\n" << r.err);
  REQUIRE(r.ok);
  CHECK(lines_with(r.log, {"WARNING: pair tadah: potential '",
                           "Deprecated key SIGMA is ignored"})
            .size() == n_sigma);
  CHECK(lines_with(r.log, {"WARNING: pair tadah: potential '",
                           "Deprecated key MBLOCK is ignored"})
            .size() == n_mblock);
  CHECK(lines_with(r.err, {"Deprecated key"}).empty());
}

/* Pins: pair_coeff with a potential without a DM_REAM block that carries
 * EMBF0 (a key only DM_REAM reads) writes its notice to log.lammps once as
 * a LAMMPS WARNING ("WARNING: pair tadah: potential '<file>': Key EMBF0 is
 * ignored: ..."), and nothing of it to standard error.
 * Before the fix: the notice went to standard error only. Before the
 * review fix (once per capture): only when the process had not reported
 * EMBF0 before. */
TEST_CASE("pair_coeff writes the notice of a key without effect to "
          "log.lammps",
          "[pair_tadah][notices]") {
  const std::size_t n = 1;
  const Run r = run_with_potential("EMBF0     FREE\n", "embf0");
  INFO("log.lammps:\n" << r.log << "\nstderr:\n" << r.err);
  REQUIRE(r.ok);
  CHECK(lines_with(r.log, {"WARNING: pair tadah: potential '",
                           "Key EMBF0 is ignored"})
            .size() == n);
  CHECK(lines_with(r.err, {"Key EMBF0 is ignored"}).empty());
}

/* Pins: two LAMMPS instances of one process, one after the other, each
 * with its own log.lammps and pair_coeff of a potential with SIGMA: each
 * log.lammps has the deprecation notice of SIGMA once, as a LAMMPS
 * WARNING naming the potential, and nothing of it goes to standard error
 * in either run (a driver such as the LAMMPS Python module, or tadah hpo's
 * PC_LAMMPS evaluations, runs several instances in one process).
 * Before the fix: the first run's log.lammps had it and the second's had
 * none (report_ignored_key_once() gave SIGMA once per process, so the
 * second load's capture received nothing). */
TEST_CASE("pair_coeff writes the notice to the log.lammps of every LAMMPS "
          "instance of a process",
          "[pair_tadah][notices]") {
  for (int run = 0; run < 2; ++run) {
    INFO("run " << run + 1);
    const Run r = run_with_potential("SIGMA     2     1 0 0 1\n",
                                     "twice" + std::to_string(run));
    INFO("log.lammps:\n" << r.log << "\nstderr:\n" << r.err);
    REQUIRE(r.ok);
    CHECK(lines_with(r.log, {"WARNING: pair tadah: potential '",
                             "Deprecated key SIGMA is ignored"})
              .size() == 1);
    CHECK(lines_with(r.err, {"Deprecated key"}).empty());
  }
}

/* Pins: pair_coeff loads a potential with the weight covariance of an
 * OALGO 3 fit (WCOV, WNOISE) and computes with it the energy of the
 * potential without them (bit for bit); log.lammps has no WARNING about
 * either key.
 * Before the fix: pair_coeff failed ("Invalid key: WCOV"). */
TEST_CASE("pair_coeff loads a potential with WCOV and WNOISE",
          "[pair_tadah][weight_covariance]") {
  const Run plain = run_with_potential("", "plain");
  REQUIRE(plain.ok);
  const Run r = run_with_potential(
      "WCOV     2     0.5 0.1 0.1 0.25\nWNOISE     E 0.04     F 0.0025\n",
      "wcov");
  INFO("log.lammps:\n" << r.log);
  REQUIRE(r.ok);
  CHECK(r.pe == plain.pe);
  CHECK(lines_with(r.log, {"WARNING", "WCOV"}).empty());
  CHECK(lines_with(r.log, {"WARNING", "WNOISE"}).empty());
}

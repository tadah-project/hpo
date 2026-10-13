#pragma once
/* SKIP_WITHOUT_PC_PROCESS_WORKERS(): the one skip of a test case that starts
 * PC_ISOLATION PROCESS workers -- a PcWorkerPool it builds, or an hpo_run
 * whose PC_LAMMPS lines run under PC_ISOLATION PROCESS -- or that asks for
 * PC_ISOLATION PROCESS at all.
 *
 * The MPI build (TADAH_HPO_MPI) has no such workers: hpo_run refuses an
 * explicit PROCESS at startup ("PC_ISOLATION PROCESS is not supported in the
 * MPI build (TADAH_HPO_MPI) ..."), whatever lines the configuration has, and
 * runs THREAD by default, and the worker entry refuses to start
 * (pc_worker_main exits 2: a fork+exec'd child of an MPI parent must not
 * touch MPI; "pc pool: in the MPI build the worker refuses to start" pins
 * it). There such a case is reported as Skipped (TADAH_SKIP,
 * tadah_test_skip.hpp), naming the MPI build. The pool's code is the same in
 * both builds, so the serial build's run of these cases is their coverage.
 *
 * In the serial build the macro allows the PROCESS workers of the running
 * test case to start: it sets TADAH_TEST_PC_PROCESS_WORKERS=1, which the
 * workers inherit (the pool forks and execs /proc/self/exe), and the
 * listener below unsets it when each test case starts and ends. test_main.cpp
 * makes a worker of test_hpo started without it refuse to start, as every
 * worker of the MPI build does, so a case that starts such workers without
 * the call fails in the serial build too ("pc-worker 1: handshake failed:
 * ..."), whatever code built the pool. "pc pool: in the serial build the
 * worker of a case that did not call SKIP_WITHOUT_PC_PROCESS_WORKERS()
 * refuses to start" pins it.
 *
 * Call it as the first statement of the TEST_CASE body (TADAH_SKIP's rules:
 * before any check, outside every SECTION). hpo_mpi_build_tests_lint
 * (check_mpi_build_tests.py) fails a case that builds a pool or requests
 * PROCESS in the MPI build's view of its file without it. Include this
 * header before catch2/catch.hpp, as tadah_test_skip.hpp requires. */

#include "tadah_test_skip.hpp"

#include <cstdlib>
#include <cstring>

namespace hpo_test {

/** The environment variable through which the running test case allows
 *  the PC_ISOLATION PROCESS workers it starts (serial build). */
inline constexpr const char* kPcProcessWorkersVariable =
    "TADAH_TEST_PC_PROCESS_WORKERS";

/** Whether this process -- a worker of test_hpo, which inherits the
 *  environment of the case that started it -- may run as a worker. */
inline bool pc_process_workers_allowed() {
  const char* v = std::getenv(kPcProcessWorkersVariable);
  return v != nullptr && std::strcmp(v, "1") == 0;
}

/** The serial build's SKIP_WITHOUT_PC_PROCESS_WORKERS(): the workers the
 *  running test case starts from now on may start. */
inline void allow_pc_process_workers() {
  if (::setenv(kPcProcessWorkersVariable, "1", 1) != 0)
    FAIL("SKIP_WITHOUT_PC_PROCESS_WORKERS(): setenv("
         << kPcProcessWorkersVariable << ") failed");
}

/** Unsets the variable when a test case starts and when it ends, so that
 *  the call allows the workers of its own test case only, in a run of the
 *  whole binary as in CTest's one-case runs (and a value set in the
 *  environment of test_hpo allows none). */
struct PcProcessWorkersListener : Catch::TestEventListenerBase {
  using TestEventListenerBase::TestEventListenerBase;

  void testCaseStarting(Catch::TestCaseInfo const& info) override {
    ::unsetenv(kPcProcessWorkersVariable);
    TestEventListenerBase::testCaseStarting(info);
  }

  void testCaseEnded(Catch::TestCaseStats const& stats) override {
    ::unsetenv(kPcProcessWorkersVariable);
    TestEventListenerBase::testCaseEnded(stats);
  }
};

/** One registration per test binary (an inline variable has a single
 *  instance), as tadah_test_skip.hpp registers its listener. */
inline const Catch::ListenerRegistrar<PcProcessWorkersListener>
    pc_process_workers_listener_registrar;

}  // namespace hpo_test

#ifdef TADAH_HPO_MPI
#define SKIP_WITHOUT_PC_PROCESS_WORKERS()                                     \
  TADAH_SKIP("PC_ISOLATION PROCESS is serial-build only: the MPI build "      \
             "(TADAH_HPO_MPI) refuses it and its pc_lammps worker refuses "   \
             "to start")
#else
#define SKIP_WITHOUT_PC_PROCESS_WORKERS() ::hpo_test::allow_pc_process_workers()
#endif

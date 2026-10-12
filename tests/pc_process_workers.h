#pragma once
/* SKIP_WITHOUT_PC_PROCESS_WORKERS(): the one skip of a test case that starts
 * PC_ISOLATION PROCESS workers -- a PcWorkerPool it builds, or an hpo_run
 * whose PC_LAMMPS lines run under PC_ISOLATION PROCESS.
 *
 * The MPI build (TADAH_HPO_MPI) has no such workers: hpo_run refuses an
 * explicit PROCESS at startup ("PC_ISOLATION PROCESS is not supported in the
 * MPI build (TADAH_HPO_MPI) ...") and runs THREAD by default, and the worker
 * entry refuses to start (pc_worker_main exits 2: a fork+exec'd child of an
 * MPI parent must not touch MPI; "pc pool: in the MPI build the worker
 * refuses to start" pins it). There such a case is reported as Skipped
 * (TADAH_SKIP, tadah_test_skip.hpp), naming the MPI build; in the serial
 * build the macro does nothing. The pool's code is the same in both builds,
 * so the serial build's run of these cases is their coverage.
 *
 * Call it as the first statement of the TEST_CASE body (TADAH_SKIP's rules:
 * before any check, outside every SECTION). hpo_mpi_build_tests_lint
 * (check_mpi_build_tests.py) fails a case that starts PROCESS workers in the
 * MPI build's view of its file without it. Include this header before
 * catch2/catch.hpp, as tadah_test_skip.hpp requires. */

#include "tadah_test_skip.hpp"

#ifdef TADAH_HPO_MPI
#define SKIP_WITHOUT_PC_PROCESS_WORKERS()                                     \
  TADAH_SKIP("PC_ISOLATION PROCESS is serial-build only: the MPI build "      \
             "(TADAH_HPO_MPI) refuses it and its pc_lammps worker refuses "   \
             "to start")
#else
#define SKIP_WITHOUT_PC_PROCESS_WORKERS() (void)0
#endif

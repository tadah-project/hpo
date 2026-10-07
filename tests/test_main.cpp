#define CATCH_CONFIG_RUNNER  // custom main: the PC_ISOLATION pool re-execs
                             // /proc/self/exe, so when the pool runs inside
                             // THIS test binary the child is test_hpo itself
                             // and must take the worker branch before Catch2.
#include "catch2/catch.hpp"

#include <tadah/core/signal_handler.h>

#include <csignal>
#include <cstdio>
#include <cstring>

#ifdef TADAH_HPO_MPI
#include <mpi.h>
#endif

namespace tadah { namespace hpo { int pc_worker_main(int argc, char** argv); } }

namespace {

/* `test_hpo --tadah-signal-handler-probe` prints, one per line, the signals
 * core::SignalHandler::install() handles: those whose disposition it
 * changes in a fresh process, before Catch2 installs handlers of its own.
 * test_pc_worker_pool.cpp checks that the PC_ISOLATION workers ignore
 * exactly these. */
int signal_handler_probe() {
  struct sigaction before[NSIG] = {};
  bool known[NSIG] = {};
  for (int s = 1; s < NSIG; ++s)
    known[s] = ::sigaction(s, nullptr, &before[s]) == 0;
  tadah::core::SignalHandler::install();
  for (int s = 1; s < NSIG; ++s) {
    struct sigaction now = {};
    if (!known[s] || ::sigaction(s, nullptr, &now) != 0) continue;
    if (now.sa_handler != before[s].sa_handler) std::printf("%d\n", s);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--tadah-pc-worker") == 0)
    return tadah::hpo::pc_worker_main(argc, argv);
  if (argc >= 2 && std::strcmp(argv[1], "--tadah-signal-handler-probe") == 0)
    return signal_handler_probe();
#ifdef TADAH_HPO_MPI
  /* The MPI build creates every LAMMPS instance on MPI_COMM_SELF
   * (hpo_lammps_comm()), which needs an initialised MPI: tadah's main()
   * initialises it, and so does this runner, as tadah does -- a one-rank
   * job (hpo_size() 1, the single-process paths) with
   * MPI_THREAD_MULTIPLE, and the topology framework touched once before
   * any thread creates a LAMMPS instance. Without it every case that
   * creates one aborts ("MPI_Comm_rank() function was called before
   * MPI_INIT"). The two branches above never touch MPI. */
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
    std::fprintf(stderr, "test_hpo: the MPI library provides thread level "
                         "%d < MPI_THREAD_MULTIPLE\n", provided);
    MPI_Finalize();
    return 1;
  }
  {
    int dims[1] = {1}, periods[1] = {0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_SELF, 1, dims, periods, 0, &cart);
    if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
  }
  const int status = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return status;
#else
  return Catch::Session().run(argc, argv);
#endif
}

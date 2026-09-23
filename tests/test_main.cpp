#define CATCH_CONFIG_RUNNER  // custom main: the PC_ISOLATION pool re-execs
                             // /proc/self/exe, so when the pool runs inside
                             // THIS test binary the child is test_hpo itself
                             // and must take the worker branch before Catch2.
#include "catch2/catch.hpp"

#include <tadah/core/signal_handler.h>

#include <csignal>
#include <cstdio>
#include <cstring>

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
  return Catch::Session().run(argc, argv);
}

#define CATCH_CONFIG_RUNNER  // custom main: the PC_ISOLATION pool re-execs
                             // /proc/self/exe, so when the pool runs inside
                             // THIS test binary the child is test_hpo itself
                             // and must take the worker branch before Catch2.
#include "catch2/catch.hpp"

#include <cstring>

namespace tadah { namespace hpo { int pc_worker_main(int argc, char** argv); } }

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--tadah-pc-worker") == 0)
    return tadah::hpo::pc_worker_main(argc, argv);
  return Catch::Session().run(argc, argv);
}

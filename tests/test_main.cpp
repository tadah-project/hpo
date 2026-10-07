#define CATCH_CONFIG_RUNNER  // custom main: the PC_ISOLATION pool re-execs
                             // /proc/self/exe, so when the pool runs inside
                             // THIS test binary the child is test_hpo itself
                             // and must take the worker branch before Catch2.
#include "catch2/catch.hpp"

#include <tadah/core/signal_handler.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/pc_worker_wire.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include <unistd.h>

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

/* A stand-in PC_ISOLATION worker, run instead of pc_worker_main when the
 * pool spawns a worker with TADAH_TEST_PC_WORKER_WRONG_ACK_PID in its
 * environment (test_pc_worker_pool.cpp sets it around one pool): it writes
 * its pid, the one the pool forked, to the file the variable names, answers
 * HELLO with a HELLO_ACK that names its pid + 1, then reads frames until
 * the pool closes the channel or kills it. */
int wrong_ack_pid_worker(const char* pid_file) {
  namespace pcwire = tadah::hpo::pcwire;
  constexpr int kFd = 3;   /* the spawn recipe's channel */
  try {
    pcwire::FrameType t;
    std::string payload;
    if (!pcwire::recv_frame(kFd, t, payload, -1)
        || t != pcwire::FrameType::HELLO)
      return 3;
    std::FILE* f = std::fopen(pid_file, "w");
    if (!f) return 4;
    const bool written =
        std::fprintf(f, "%ld\n", static_cast<long>(::getpid())) > 0;
    if (std::fclose(f) != 0 || !written) return 4;
    pcwire::Writer ack;
    ack.u32(tadah::hpo::PcWorkerPool::PROTOCOL_VERSION);
    ack.u32(static_cast<std::uint32_t>(::getpid()) + 1u);
    pcwire::send_frame(kFd, pcwire::FrameType::HELLO_ACK, ack.buf);
    while (pcwire::recv_frame(kFd, t, payload, -1)) {}
  } catch (const std::exception&) {
    /* EOF: the pool closed the channel */
  }
  return 0;
}

/* A stand-in PC_ISOLATION worker, run instead of pc_worker_main when the
 * pool spawns a worker with TADAH_TEST_PC_WORKER_DIE_IN_TASK in its
 * environment (test_pc_worker_pool.cpp): it answers HELLO as a worker does,
 * reads the first TASK frame whole, as pc_worker_main does before it runs
 * the script, and then kills itself with SIGKILL, as the kernel's
 * out-of-memory killer or a crash inside LAMMPS ends a worker in the middle
 * of its script. */
int die_in_task_worker() {
  namespace pcwire = tadah::hpo::pcwire;
  constexpr int kFd = 3;   /* the spawn recipe's channel */
  try {
    pcwire::FrameType t;
    std::string payload;
    if (!pcwire::recv_frame(kFd, t, payload, -1)
        || t != pcwire::FrameType::HELLO)
      return 3;
    pcwire::Writer ack;
    ack.u32(tadah::hpo::PcWorkerPool::PROTOCOL_VERSION);
    ack.u32(static_cast<std::uint32_t>(::getpid()));
    pcwire::send_frame(kFd, pcwire::FrameType::HELLO_ACK, ack.buf);
    if (!pcwire::recv_frame(kFd, t, payload, -1)
        || t != pcwire::FrameType::TASK)
      return 3;
  } catch (const std::exception&) {
    return 3;
  }
  std::raise(SIGKILL);
  return 5;   /* not reached */
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::strcmp(argv[1], "--tadah-pc-worker") == 0) {
    if (const char* pid_file = std::getenv("TADAH_TEST_PC_WORKER_WRONG_ACK_PID"))
      return wrong_ack_pid_worker(pid_file);
    if (std::getenv("TADAH_TEST_PC_WORKER_DIE_IN_TASK"))
      return die_in_task_worker();
    return tadah::hpo::pc_worker_main(argc, argv);
  }
  if (argc >= 2 && std::strcmp(argv[1], "--tadah-signal-handler-probe") == 0)
    return signal_handler_probe();
  return Catch::Session().run(argc, argv);
}

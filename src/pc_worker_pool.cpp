/* PC_ISOLATION worker-process pool -- parent side.
 * Design: INVESTIGATIONS/PC_ISOLATION_process_pool.md; protocol overview in
 * pc_worker_pool.h; wire codec in pc_worker_wire.h. */

#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/pc_worker_wire.h>
#include <tadah/hpo/hpo_log.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tadah {
namespace hpo {

using pcwire::FrameType;

namespace {

/* Encode one per-slot ScriptSpec table for the HELLO frame. */
void encode_specs(pcwire::Writer& w, const std::vector<PcScriptSpec>& specs) {
  w.u64(specs.size());
  for (const auto& s : specs) {
    w.str(s.script);
    w.vstr(s.script_lines);
    w.vstr(s.invar_final);
    w.vstr(s.outvar_names);
    w.vstr(s.hotvar_names);
  }
}

std::string decode_wait_status(int st) {
  if (WIFEXITED(st))
    return "exited " + std::to_string(WEXITSTATUS(st));
  if (WIFSIGNALED(st))
    return std::string("signal ") + std::to_string(WTERMSIG(st))
         + " (" + strsignal(WTERMSIG(st)) + ")";
  return "wait status " + std::to_string(st);
}

}  // namespace

PcWorkerPool::PcWorkerPool(int width,
                           std::vector<std::vector<PcScriptSpec>> per_slot_specs,
                           double worker_timeout_s)
    : width_(width < 1 ? 1 : width),
      worker_timeout_s_(worker_timeout_s),
      specs_(std::move(per_slot_specs)),
      /* exec /proc/self/exe LITERALLY: the old inode stays valid even if the
       * on-disk binary is replaced mid-run, so respawns are build-consistent. */
      self_exe_("/proc/self/exe") {
  if (specs_.size() != static_cast<std::size_t>(width_))
    throw std::runtime_error("PcWorkerPool: per_slot_specs size != width");
  channels_.reserve(width_);
  for (int s = 0; s < width_; ++s)
    channels_.push_back(std::make_unique<Channel>());
  /* Eager initial spawn (design hygiene: before the ScriptExecutor threads
   * have ever run a task). Respawns are safe by the fork+exec recipe alone. */
  for (int s = 0; s < width_; ++s) {
    std::lock_guard<std::mutex> lk(channels_[s]->mtx);
    spawn_locked(*channels_[s], s);
  }
}

PcWorkerPool::~PcWorkerPool() { shutdown(); }

long PcWorkerPool::deadline_ms_for(const std::string& maxtime) const {
  if (worker_timeout_s_ > 0.0)
    return static_cast<long>(worker_timeout_s_ * 1000.0);
  if (maxtime == "off" || maxtime.empty()) return -1;   /* thread-mode parity */
  char* endp = nullptr;
  const double mt = std::strtod(maxtime.c_str(), &endp);
  if (endp == maxtime.c_str() || mt <= 0.0) return -1;
  /* Covers multi-section scripts (timer timeout re-arms after every `clear`)
   * plus the non-run time timer timeout cannot bound (create, pot build,
   * neighbor lists). */
  const double s = (3.0 * mt > mt + 120.0) ? 3.0 * mt : mt + 120.0;
  return static_cast<long>(s * 1000.0);
}

void PcWorkerPool::spawn_locked(Channel& ch, int slot) {
  if (ch.consec_deaths >= PC_MAX_CONSEC_DEATHS)
    throw std::runtime_error(
        "pc-worker slot " + std::to_string(slot) + ": "
        + std::to_string(ch.consec_deaths)
        + " consecutive worker deaths -- giving up (see journal)");

  int sv[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
    throw std::runtime_error("pc-worker: socketpair failed");

  /* Parent end must not leak into OTHER children spawned later. */
  ::fcntl(sv[0], F_SETFD, FD_CLOEXEC);

  const pid_t parent_pid = ::getpid();
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(sv[0]);
    ::close(sv[1]);
    throw std::runtime_error("pc-worker: fork failed");
  }

  if (pid == 0) {
    /* CHILD: async-signal-safe calls ONLY between fork and exec. */
    if (::dup2(sv[1], 3) < 0) ::_exit(97);
    /* Close every fd >= 4: drops the parent's non-CLOEXEC ofstreams, the
     * spare socketpair copies, and anything else inherited. Plain close()
     * loop for portability (close_range needs kernel >= 5.9). */
    for (int f = 4; f < 65536; ++f) ::close(f);
    ::prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (::getppid() != parent_pid) ::_exit(98);   /* parent died pre-prctl */
    const char* argv[] = {"tadah-pc-worker", "--tadah-pc-worker", nullptr};
    ::execv(self_exe_.c_str(), const_cast<char* const*>(argv));
    ::_exit(99);
  }

  /* PARENT */
  ::close(sv[1]);
  ch.pid = pid;
  ch.fd = sv[0];
  ch.task_seq = 0;
  ch.last_pot_iter = -1;

  try {
    pcwire::Writer w;
    w.u32(PROTOCOL_VERSION);
    w.u32(static_cast<std::uint32_t>(slot));
    encode_specs(w, specs_[slot]);
    pcwire::send_frame(ch.fd, FrameType::HELLO, w.buf);

    FrameType t;
    std::string payload;
    if (!pcwire::recv_frame(ch.fd, t, payload, 60'000))
      throw std::runtime_error("HELLO_ACK timeout");
    if (t != FrameType::HELLO_ACK)
      throw std::runtime_error("expected HELLO_ACK");
    pcwire::Reader r(payload);
    const auto ver = r.u32();
    (void)r.u32();   /* worker pid (informational) */
    if (ver != PROTOCOL_VERSION)
      throw std::runtime_error("protocol version mismatch");
  } catch (const std::exception& e) {
    kill_reap_locked(ch, slot, e.what());
    throw std::runtime_error(std::string("pc-worker slot ")
                             + std::to_string(slot)
                             + ": handshake failed: " + e.what());
  }
}

void PcWorkerPool::kill_reap_locked(Channel& ch, int slot, const char* why) {
  int st = 0;
  std::string status = "(not reaped)";
  if (ch.pid > 0) {
    ::kill(ch.pid, SIGKILL);
    pid_t r;
    do { r = ::waitpid(ch.pid, &st, 0); } while (r < 0 && errno == EINTR);
    if (r == ch.pid) status = decode_wait_status(st);
  }
  if (ch.fd >= 0) ::close(ch.fd);
  ch.fd = -1;
  ch.pid = -1;
  ch.last_pot_iter = -1;
  ch.last_pot_map.clear();   /* fresh worker knows nothing: next pot is FULL */
  ++ch.consec_deaths;
  /* ALWAYS journaled: silent respawns would mask crash-loops and sanitizer
   * aborts inside workers. */
  HPOLog::instance().warning(
      std::string("pc-worker slot ") + std::to_string(slot)
      + " killed/died: " + why + "; wait: " + status
      + "; consecutive deaths: " + std::to_string(ch.consec_deaths));
}

PcTaskResult PcWorkerPool::execute(int slot, std::uint32_t script_idx,
                                   std::uint64_t iter,
                                   const PcPotMap& pot_map,
                                   const std::vector<std::string>& inject_lines,
                                   const std::string& maxtime) {
  if (slot < 0 || slot >= width_)
    throw std::runtime_error("PcWorkerPool::execute: bad slot");
  Channel& ch = *channels_[slot];
  /* Per-channel mutex: normally uncontended (slot == the ScriptExecutor
   * worker id driving it); it exists for the executor's nested-inline path,
   * which reports worker_id 0 and could otherwise interleave two frames on
   * channel 0. */
  std::lock_guard<std::mutex> lk(ch.mtx);

  auto synthetic_fail = [&](const std::string& why) {
    PcTaskResult res;
    res.status = PcTaskStatus::RUN_FAIL;   /* failscore + eval abort (parity
                                              with a crashed script) */
    res.error_msg = why;
    return res;
  };

  if (ch.fd < 0) spawn_locked(ch, slot);   /* throws after repeated deaths */

  const bool has_pot = (ch.last_pot_iter != static_cast<long long>(iter));
  const std::uint64_t seq = ++ch.task_seq;

  try {
    pcwire::Writer w;
    w.u64(seq);
    w.u32(script_idx);
    w.u64(iter);
    if (!has_pot) {
      w.u8(0);                                     /* pot_kind: none */
    } else if (ch.last_pot_map.empty()) {
      /* FULL: first contact / post-respawn -- ship every key as a set-op. */
      w.u8(1);
      w.u64(pot_map.size());
      for (const auto& kv : pot_map) { w.str(kv.first); w.vstr(kv.second); }
    } else {
      /* DELTA vs the exact map this channel last shipped: set-ops for
       * new/changed keys, erase-ops (empty values) for vanished keys. */
      std::uint64_t n_ops = 0;
      pcwire::Writer ops;
      for (const auto& kv : pot_map) {
        auto it = ch.last_pot_map.find(kv.first);
        if (it == ch.last_pot_map.end() || it->second != kv.second) {
          ops.str(kv.first); ops.vstr(kv.second); ++n_ops;
        }
      }
      for (const auto& kv : ch.last_pot_map) {
        if (pot_map.find(kv.first) == pot_map.end()) {
          ops.str(kv.first); ops.vstr({}); ++n_ops;   /* erase */
        }
      }
      w.u8(2);
      w.u64(n_ops);
      w.buf += ops.buf;
    }
    w.vstr(inject_lines);
    pcwire::send_frame(ch.fd, FrameType::TASK, w.buf);
  } catch (const std::exception& e) {
    kill_reap_locked(ch, slot, (std::string("TASK send failed: ") + e.what()).c_str());
    return synthetic_fail(e.what());
  }
  if (has_pot) {
    ch.last_pot_iter = static_cast<long long>(iter);
    ch.last_pot_map = pot_map;
  }

  FrameType t;
  std::string payload;
  try {
    const long dl = deadline_ms_for(maxtime);
    if (!pcwire::recv_frame(ch.fd, t, payload, dl)) {
      kill_reap_locked(ch, slot, ("task deadline expired ("
                                  + std::to_string(dl) + " ms)").c_str());
      return synthetic_fail("worker task deadline expired");
    }
  } catch (const std::exception& e) {
    kill_reap_locked(ch, slot, (std::string("RESULT recv failed: ") + e.what()).c_str());
    return synthetic_fail(e.what());
  }

  try {
    if (t != FrameType::RESULT) throw std::runtime_error("expected RESULT frame");
    pcwire::Reader r(payload);
    const auto rseq = r.u64();
    const auto ridx = r.u32();
    if (rseq != seq || ridx != script_idx)
      throw std::runtime_error("task_seq/script_idx echo mismatch");
    PcTaskResult res;
    res.status = static_cast<PcTaskStatus>(r.u32());
    res.error_msg = r.str();
    res.outvar_values = r.vstr();
    res.hotvar_values = r.vstr();
    res.create_ms = r.f64();
    res.run_ms = r.f64();
    ch.consec_deaths = 0;   /* a full round-trip proves the worker healthy */
    return res;
  } catch (const std::exception& e) {
    kill_reap_locked(ch, slot, (std::string("RESULT decode failed: ") + e.what()).c_str());
    return synthetic_fail(e.what());
  }
}

void PcWorkerPool::shutdown() {
  if (shut_) return;
  shut_ = true;
  for (int s = 0; s < width_; ++s) {
    Channel& ch = *channels_[s];
    std::lock_guard<std::mutex> lk(ch.mtx);
    if (ch.fd >= 0) {
      try { pcwire::send_frame(ch.fd, FrameType::SHUTDOWN, std::string()); }
      catch (...) { /* worker already gone */ }
      ::close(ch.fd);
      ch.fd = -1;
    }
    if (ch.pid > 0) {
      /* Grace period, then SIGKILL. */
      int st = 0;
      pid_t r = 0;
      for (int i = 0; i < 200; ++i) {   /* ~2 s at 10 ms */
        r = ::waitpid(ch.pid, &st, WNOHANG);
        if (r == ch.pid || r < 0) break;
        struct timespec ts{0, 10'000'000};
        ::nanosleep(&ts, nullptr);
      }
      if (r == 0) {
        ::kill(ch.pid, SIGKILL);
        do { r = ::waitpid(ch.pid, &st, 0); } while (r < 0 && errno == EINTR);
      }
      ch.pid = -1;
    }
  }
}

/* ---- process-wide singleton ------------------------------------------- */

namespace {
std::unique_ptr<PcWorkerPool>& pool_slot() {
  static std::unique_ptr<PcWorkerPool> p;
  return p;
}
}  // namespace

void pc_worker_pool_init(int width,
                         std::vector<std::vector<PcScriptSpec>> per_slot_specs,
                         double worker_timeout_s) {
  if (pool_slot())
    throw std::runtime_error("pc_worker_pool_init: pool already initialized");
  pool_slot() = std::make_unique<PcWorkerPool>(
      width, std::move(per_slot_specs), worker_timeout_s);
}

bool pc_worker_pool_enabled() { return static_cast<bool>(pool_slot()); }

PcWorkerPool& pc_worker_pool() {
  if (!pool_slot())
    throw std::runtime_error("pc_worker_pool: not initialized");
  return *pool_slot();
}

void pc_worker_pool_shutdown() { pool_slot().reset(); }

}  // namespace hpo
}  // namespace tadah

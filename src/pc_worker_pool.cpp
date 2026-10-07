/* PC_ISOLATION worker-process pool -- parent side.
 * Design: INVESTIGATIONS/PC_ISOLATION_process_pool.md; protocol overview in
 * pc_worker_pool.h; wire codec in pc_worker_wire.h. */

#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/pc_worker_wire.h>
#include <tadah/hpo/hpo_log.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
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
    w.vstr(s.failvar_names);   /* protocol v2 */
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
    : width_(width),
      worker_timeout_s_(worker_timeout_s),
      specs_(std::move(per_slot_specs)),
      /* exec /proc/self/exe LITERALLY: the old inode stays valid even if the
       * on-disk binary is replaced mid-run, so respawns are build-consistent. */
      self_exe_("/proc/self/exe") {
  /* THREADS is at least 1 at its line; a width below 1 used to be raised
   * to one worker without a word. */
  if (width_ < 1)
    throw std::invalid_argument(
        "PcWorkerPool: the width must be at least 1 worker, got " +
        std::to_string(width_));
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

namespace {

/* The deadline in seconds; nullopt for none (thread-mode parity: no soft
 * limit, no hard deadline). */
std::optional<double> deadline_seconds(double worker_timeout_s,
                                       std::optional<double> script_timeout_s) {
  if (worker_timeout_s > 0.0) return worker_timeout_s;
  if (!script_timeout_s) return std::nullopt;
  const double t = *script_timeout_s;
  return (3.0 * t > t + 120.0) ? 3.0 * t : t + 120.0;
}

/* s seconds in milliseconds fit a long: below 2^63, the first double a
 * long cannot hold. */
bool fits_long_ms(double s) {
  static_assert(sizeof(long) == 8, "pc_worker_deadline_ms: 64-bit long");
  return s * 1000.0 < 0x1p63;
}

}  // namespace

bool pc_worker_deadline_fits(double worker_timeout_s,
                             std::optional<double> script_timeout_s) {
  const std::optional<double> s =
      deadline_seconds(worker_timeout_s, script_timeout_s);
  return !s || fits_long_ms(*s);
}

long pc_worker_deadline_ms(double worker_timeout_s,
                           std::optional<double> script_timeout_s) {
  /* parse_script_timeout rejects a limit <= 0 at its line; one reaching
   * this point is a caller's error, not "no deadline". */
  if (script_timeout_s && !(*script_timeout_s > 0.0)) {
    std::ostringstream msg;
    msg << "pc_worker_deadline_ms: the script timeout must be positive, got "
        << *script_timeout_s << " s ('off' and 'unlimited' are no limit)";
    throw std::invalid_argument(msg.str());
  }
  const std::optional<double> s =
      deadline_seconds(worker_timeout_s, script_timeout_s);
  if (!s) return -1;
  /* The parsers reject such a timeout at its line; converting it would be
   * undefined behaviour. */
  if (!fits_long_ms(*s)) {
    std::ostringstream msg;
    msg << "pc_worker_deadline_ms: the deadline does not fit in 64-bit "
           "milliseconds (" << *s << " s)";
    throw std::invalid_argument(msg.str());
  }
  return static_cast<long>(*s * 1000.0);
}

void PcWorkerPool::spawn_locked(Channel& ch, int slot) {
  /* Repeated deaths back the channel off; they must NOT retire it.
   *
   * The previous behaviour threw here, before the fork. Because execute()
   * only calls spawn_locked when ch.fd < 0 -- exactly the state
   * kill_reap_locked leaves -- and consec_deaths is reset in only one place
   * (after a complete RESULT round trip, which needs a live channel), the
   * throw was permanent for that slot. It was then swallowed by the
   * eval-thread catch into a FAIL_SCORE, so the run continued while one of
   * N worker slots was dead forever.
   *
   * That degrades far worse than "lose 1/N of throughput": a channel that
   * throws before doing any work returns to the shared task queue in
   * microseconds, so it drains tasks orders of magnitude faster than the
   * healthy workers and captures most of them. The observable result is a
   * near-100% FAILSCORE rate at an inflated evaluation rate, with one
   * WARNING per eval and no fatal error.
   *
   * Backing off instead is strictly better: three deaths with no
   * intervening successful round trip is pathological regardless of elapsed
   * time (a healthy channel completes hundreds per epoch), so an
   * unconditional sleep is both simpler and better targeted than a
   * time-decay -- the deaths that trip the latch are by construction
   * closely spaced. Sleeping rather than returning a synthetic failure is
   * deliberate: it makes one evaluation slow instead of feeding the
   * optimiser a fabricated FAILSCORE, and it removes the fast-drain
   * amplification above.
   *
   * Safe to sleep under ch.mtx: kill_matching only try_locks, and
   * current_iter is still -1 here (ClearPub is constructed after this
   * call), so the gate correctly treats the channel as idle. */
  if (ch.consec_deaths >= PC_MAX_CONSEC_DEATHS) {
    HPOLog::instance().warning(
        "pc-worker " + logidx::uis(static_cast<std::size_t>(slot)) + ": "
        + std::to_string(ch.consec_deaths)
        + " consecutive worker deaths with no successful task -- backing off "
        + std::to_string(PC_RESPAWN_BACKOFF_MS) + " ms, then respawning");
    struct timespec ts{PC_RESPAWN_BACKOFF_MS / 1000,
                       (PC_RESPAWN_BACKOFF_MS % 1000) * 1'000'000L};
    ::nanosleep(&ts, nullptr);
    ch.consec_deaths = 0;
  }

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
    throw std::runtime_error(
        std::string("pc-worker ")
        + logidx::uis(static_cast<std::size_t>(slot))
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
  /* A kill we asked for (the FAILPROBE gate) says nothing about this
   * channel's health, so it does not count toward the respawn backoff. */
  const bool deliberate = ch.gate_killed.exchange(false,
                                                  std::memory_order_acq_rel);
  if (!deliberate) ++ch.consec_deaths;
  /* ALWAYS journaled: silent respawns would mask crash-loops and sanitizer
   * aborts inside workers. */
  HPOLog::instance().warning(
      std::string("pc-worker ") + logidx::uis(static_cast<std::size_t>(slot))
      + " killed/died: " + why + "; wait: " + status
      + (deliberate
             ? "; deliberate (FAILPROBE gate) -- not counted as a slot fault"
             : "; consecutive deaths: " + std::to_string(ch.consec_deaths)));
}

PcTaskResult PcWorkerPool::execute(int slot, std::uint32_t script_idx,
                                   std::uint64_t iter,
                                   const PcPotMap& pot_map,
                                   const std::vector<std::string>& inject_lines,
                                   std::optional<double> script_timeout_s) {
  if (slot < 0 || slot >= width_)
    throw std::runtime_error("PcWorkerPool::execute: bad slot");
  Channel& ch = *channels_[slot];
  /* Per-channel mutex: normally uncontended (slot == the ScriptExecutor
   * worker id driving it); it exists for the executor's nested-inline path,
   * which reports worker_id 0 and could otherwise interleave two frames on
   * channel 0. */
  std::lock_guard<std::mutex> lk(ch.mtx);

  /* D10: a killed task still consumed wall time -- usually the MOST of any
   * task in the eval, since it typically burned the whole deadline. It used
   * to report create_ms = run_ms = 0 (the struct defaults), and the host
   * folds those into the per-script EWMA that drives the longest-processing-
   * time dispatch order and into the fan-out min/max. So the one script that
   * hung was recorded as the cheapest, its EWMA fell ~30%, and the LPT sort
   * pushed it LATER -- the opposite of the intent -- while the journal
   * reported "fastest 0 ms". Measuring from the top of execute() covers
   * everything the task actually occupied the channel for. */
  const auto t_task0 = std::chrono::steady_clock::now();
  auto synthetic_fail = [&](const std::string& why) {
    PcTaskResult res;
    res.status = PcTaskStatus::KILLED;   /* parent-synthesized: worker died or
                                            was SIGKILLed (deadline / gate).
                                            Failscore + eval abort; graded as
                                            band C (did not complete). */
    res.error_msg = why;
    res.run_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_task0).count();
    return res;
  };

  if (ch.fd < 0) spawn_locked(ch, slot);   /* throws after repeated deaths */

  /* Publish (iter, pid) for kill_matching -- inside the locked region, so a
   * failed try_lock over there guarantees these describe the in-flight task.
   * Cleared on every exit path below. */
  ch.current_pid.store(ch.pid, std::memory_order_relaxed);
  ch.current_iter.store(static_cast<long long>(iter), std::memory_order_release);
  struct ClearPub {
    Channel& c;
    ~ClearPub() {
      /* Retire the pid as well as the iter. Correctness against
       * kill_matching already rests on ch.mtx (it only inspects a channel
       * whose try_lock failed, i.e. one inside execute(), which republishes
       * both fields before doing any work). But leaving current_pid holding
       * a reaped pid across idle periods is a loaded gun: it is safe only
       * for as long as the iter check short-circuits first, and pids are
       * reused by the OS. Clear pid before iter so the pair is never
       * observed as (idle, live-looking pid). */
      c.current_pid.store(-1, std::memory_order_relaxed);
      c.current_iter.store(-1, std::memory_order_release);
    }
  } clear_pub{ch};

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
    const long dl = pc_worker_deadline_ms(worker_timeout_s_, script_timeout_s);
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
    res.outvar_errors = r.vstr();    /* protocol v3 */
    res.hotvar_values = r.vstr();
    res.failvar_values = r.vstr();   /* protocol v2 */
    res.create_ms = r.f64();
    res.run_ms = r.f64();
    ch.consec_deaths = 0;   /* a full round-trip proves the worker healthy */
    return res;
  } catch (const std::exception& e) {
    kill_reap_locked(ch, slot, (std::string("RESULT decode failed: ") + e.what()).c_str());
    return synthetic_fail(e.what());
  }
}

void PcWorkerPool::kill_matching(std::uint64_t iter) {
  const long long target = static_cast<long long>(iter);
  for (int s = 0; s < width_; ++s) {
    Channel& ch = *channels_[s];
    /* try_lock SUCCESS => channel idle (no execute in flight) => nothing to
     * kill; release and move on. try_lock FAILURE => exactly one execute()
     * holds the lock for its whole task, and it published (pid, iter) inside
     * that region -- so the atomics reliably describe the in-flight task and
     * a mis-kill of another eval's task is impossible. */
    if (ch.mtx.try_lock()) {
      ch.mtx.unlock();
      continue;
    }
    if (ch.current_iter.load(std::memory_order_acquire) == target) {
      const pid_t p = ch.current_pid.load(std::memory_order_relaxed);
      /* Announce the kill BEFORE sending it. This death is deliberate --
       * the FAILPROBE gate decided the whole evaluation is not worth
       * finishing -- so it is not evidence that this channel is faulty and
       * must not count toward the consecutive-death backoff. Without this
       * flag, three gate firings on one slot in three consecutive tasks
       * would throttle a perfectly healthy worker. */
      ch.gate_killed.store(true, std::memory_order_release);
      if (p > 0) ::kill(p, SIGKILL);
      /* The blocked execute() on this channel sees EOF and runs its normal
       * kill/reap/journal/respawn path; its caller gets a synthetic KILLED
       * result which is clamped by the gated eval anyway. */
    }
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

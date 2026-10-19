#pragma once

/*
 * Bounded retry of transient filesystem errors.
 *
 * Network filesystems (SMB/CIFS, NFS) and busy local disks report short-lived
 * errors -- EBUSY on a lock file another client briefly holds, ESTALE after a
 * server failover, EIO on a dropped session -- that succeed when simply
 * repeated a few milliseconds later. The durable best potential and the
 * per-evaluation logs repeat such operations with exponential backoff before
 * giving up; permanent errors (ENOSPC, EROFS, ENOENT, ...) are not retried.
 *
 * Success-path contract: retry() costs one try block and a few integers --
 * no clock read, no allocation, no syscall. The clock is read, the jitter
 * generator touched, the abort predicate polled and (for EACCES) the
 * filesystem type queried only after a failure. consume_fault() is one
 * relaxed atomic load while no fault table is installed.
 *
 * Nothing here depends on the rest of HPO, so it can move to CORE unchanged:
 * the caller injects its shutdown flag as BackoffPolicy::should_abort.
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace tadah::hpo::io {

/* A stop request, polled on the failure path only (before every retry and
 * between the <= 100 ms slices of a backoff sleep). Once it reports true
 * the retry loop keeps going for at most BackoffPolicy::stop_grace (2 s)
 * from the moment it first saw the request -- delays are shortened to fit,
 * and the existing budget still applies -- and then gives up with the last
 * error (GiveUp::abandoned). So SIGINT/SIGTERM/SIGUSR2 is not held up by a
 * 30 s backoff, yet one transient blip right after the signal (a SLURM
 * time-limit warning, say) does not lose the in-flight best potential or
 * the final drain's rows. HPO passes core::SignalHandler::stop_requested. A
 * plain function pointer: copying a policy stays trivial. */
using AbortFn = bool (*)();

/* Exponential backoff: the n-th retry sleeps initial * factor^(n-1), capped at
 * `cap`, then scaled by a uniform factor in [1-jitter, 1+jitter], and never
 * less than 1 ms. A retry is attempted only while both the total sleep and
 * the wall time since the first failure, each plus the next delay, stay
 * within `budget`; budget 0 means "no retry after a transient error". A
 * failing call that blocks by itself (a soft-mount timeout) can still add its
 * own duration once at each end of that window; no budget bounds a call that
 * hangs indefinitely (a hard mount). */
struct BackoffPolicy {
    std::chrono::milliseconds initial{10};
    std::chrono::milliseconds cap{2000};
    std::chrono::milliseconds budget{0};
    double  factor = 2.0;
    double  jitter = 0.2;
    AbortFn should_abort = nullptr;
    /* How long a retry loop goes on after it first sees should_abort report
     * a stop (see AbortFn). */
    std::chrono::milliseconds stop_grace{2000};

    /** Durable best potential: 30 s budget (then the run fails). */
    static BackoffPolicy durable();
    /** Per-evaluation logs and archives: 10 s budget (then rows are lost
     *  and reported, the run continues). */
    static BackoffPolicy writer();

    /* Both factories honour the test-only environment variable
     * TADAH_HPO_TEST_IO_BACKOFF_MS="initial,cap,budget[,jitter]" (ms; read
     * once per process), which replaces the timings of BOTH policies so an
     * end-to-end test does not wait tens of seconds. Neither sets
     * should_abort. */

    /** Delay before retry number `retry` (1-based), without jitter. */
    std::chrono::milliseconds nominal_delay(unsigned retry) const noexcept;
    /** nominal_delay() with jitter applied, at least 1 ms, so every retry
     *  spends budget (failure path only). */
    std::chrono::milliseconds next_delay(unsigned retry) const;
    /** should_abort is set and reports a stop request. */
    bool abort_requested() const { return should_abort && should_abort(); }
};

/* Injectable sleep so tests record the schedule instead of waiting. An empty
 * SleepFn means std::this_thread::sleep_for, in slices of at most 100 ms when
 * `should_abort` is set. Returns false when should_abort reported a stop,
 * cutting the sleep short (an injected SleepFn gets the whole delay and the
 * predicate is asked once, afterwards). */
using SleepFn = std::function<void(std::chrono::milliseconds)>;
bool sleep_for(const SleepFn& sleep, std::chrono::milliseconds d,
               AbortFn should_abort = nullptr);

/* The shutdown grace of one retry loop (failure path only). Before each
 * retry, admit() starts the grace the first time policy.should_abort
 * reports a stop, shortens the delay to what is left of it, and returns
 * false once it is spent. "Spent" counts the larger of the wall time and
 * the sleeps since the grace started, so an injected SleepFn (tests) bounds
 * it too. A sleep that the stop request cut short starts the grace at once
 * (start()); the loop then retries without waiting further. */
class StopGrace {
public:
    bool admit(const BackoffPolicy& policy,
               std::chrono::steady_clock::time_point now,
               std::chrono::milliseconds& delay)
    {
        if (!active_) {
            if (!policy.abort_requested()) return true;
            start(now);
        }
        const auto wall =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - since_);
        const auto used = wall > slept_ ? wall : slept_;
        if (used >= policy.stop_grace) return false;
        const auto left = policy.stop_grace - used;
        if (delay > left) delay = left;
        return true;
    }
    void start(std::chrono::steady_clock::time_point now) noexcept
    {
        if (active_) return;
        active_ = true;
        since_  = now;
    }
    void slept(std::chrono::milliseconds d) noexcept { if (active_) slept_ += d; }
    /* While active, sleeps need no abort polling: the delay already fits. */
    bool active() const noexcept { return active_; }
private:
    bool active_ = false;
    std::chrono::steady_clock::time_point since_{};
    std::chrono::milliseconds slept_{0};
};

/** EBUSY, EAGAIN/EWOULDBLOCK, EACCES, ESTALE, ETXTBSY, EIO, ENOLCK, ETIMEDOUT,
 *  EDEADLK: errnos that CAN be transient somewhere. EACCES is here for SMB
 *  sharing violations / delete-pending files. */
bool is_transient_errno(int err) noexcept;

/** is_transient_errno() for an error on `path`: EACCES counts as transient
 *  only when the directory holding `path` is on a network filesystem (SMB/
 *  CIFS, NFS, AFS, Ceph, Lustre, GPFS, 9P, Coda, FUSE -- one statfs(2), on the
 *  failure path only; a directory that cannot be queried counts as network).
 *  On a local filesystem EACCES is a real permission error and fails at once. */
bool is_transient_errno(int err, const std::string& path) noexcept;

/** "EBUSY", "ENOSPC", ... ; the decimal value for anything unlisted. */
std::string errno_name(int err);

/** One failed attempt about to be retried. */
struct RetryEvent {
    std::string op;          // operation/fault site, e.g. "durable.lock_open"
    std::string path;        // file the operation touched
    int         err = 0;     // errno
    unsigned    attempt = 0; // 1-based number of the attempt that failed
    std::chrono::milliseconds next_delay{0};
    /* false: a single immediate re-attempt without backoff (the log
     * writers' after a permanent errno, a latched durable publication's
     * after a rename's ENOENT) -- not a transient retry, so not counted as
     * one. */
    bool        transient = true;
};
using OnRetryFn = std::function<void(const RetryEvent&)>;

/** Why a retry loop stopped before success. */
enum class GiveUp {
    budget,     // the backoff budget is spent
    abandoned,  // the stop_grace after a stop request (should_abort) is spent
};

/** " (gave up after N attempts over T s)", or for GiveUp::abandoned
 *  " (retry abandoned: shutdown requested) after N attempts over T s". */
std::string gave_up_suffix(unsigned attempts,
                           std::chrono::steady_clock::duration elapsed,
                           GiveUp why = GiveUp::budget);
std::string gave_up_suffix(unsigned attempts, double seconds,
                           GiveUp why = GiveUp::budget);

/* Default exhaustion hook: rethrow the last error unchanged. */
struct RethrowOnExhaustion {
    void operator()(const std::exception&, unsigned,
                    std::chrono::steady_clock::duration, GiveUp) const noexcept {}
};

/** Run op() until it returns, it throws something classify() rejects, the
 *  policy's budget is spent, or the policy's stop_grace after a stop request
 *  (should_abort) is spent.
 *
 *  classify(const std::exception&, RetryEvent&) -> bool: true = transient;
 *  it fills op/path/err of the event. on_retry(const RetryEvent&) runs
 *  before each sleep. exhausted(const std::exception& last, attempts,
 *  elapsed, GiveUp why) runs when the retry stops without success and may
 *  throw a replacement; if it returns, the last error is rethrown. op() must
 *  release every lock it takes before it throws (RAII scoped inside op),
 *  since the sleep happens outside it. Non-std exceptions propagate
 *  immediately. */
template<class Op, class Classify, class OnRetry,
         class Exhausted = RethrowOnExhaustion>
auto retry(const BackoffPolicy& policy, Op&& op, Classify&& classify,
           OnRetry&& on_retry, const SleepFn& sleep,
           Exhausted&& exhausted = Exhausted{}) -> decltype(op())
{
    unsigned attempt = 1;
    std::chrono::milliseconds slept{0};
    std::chrono::steady_clock::time_point first_failure{};
    StopGrace grace;
    for (;;) {
        try {
            return op();
        } catch (const std::exception& e) {
            RetryEvent ev;
            if (!classify(e, ev)) throw;
            const auto now = std::chrono::steady_clock::now();
            if (attempt == 1) first_failure = now;
            auto delay = policy.next_delay(attempt);
            if (policy.budget.count() <= 0 || slept + delay > policy.budget ||
                (now - first_failure) + delay > policy.budget) {
                exhausted(e, attempt, now - first_failure, GiveUp::budget);
                throw;
            }
            if (!grace.admit(policy, now, delay)) {
                exhausted(e, attempt, now - first_failure, GiveUp::abandoned);
                throw;
            }
            ev.attempt    = attempt;
            ev.next_delay = delay;
            on_retry(static_cast<const RetryEvent&>(ev));
            if (!sleep_for(sleep, delay,
                           grace.active() ? nullptr : policy.should_abort))
                grace.start(std::chrono::steady_clock::now());   // retry now
            else
                grace.slept(delay);
            slept += delay;
            ++attempt;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Test-only fault injection.
 *
 * I/O call sites ask consume_fault("<site>") before the real syscall; a
 * non-zero return is the errno to fail with instead. Rules come from the
 * environment variable
 *
 *   TADAH_HPO_TEST_IO_FAULT="site,ERRNO[,skip=K][,count=N|always][,partial=B]
 *                            [,land][,latency=MS][;...]"
 *
 * parsed once at load time, or from a testing::ScopedFaults. ERRNO is a
 * symbolic name (EBUSY) or a number. A rule lets its first K matching calls
 * through, then fails the next N (default 1; `always` = every later call).
 * partial=B (writer.append / writer.archive only) first writes B real bytes,
 * modelling a short write followed by an error. land (durable.rename only)
 * performs the real rename and then reports the error, modelling a rename
 * that landed although the call failed (an NFS retransmission); combine it
 * only with ENOENT or a transient errno -- any other errno tells
 * durable_best that the rename did not happen. latency=MS
 * makes each injected failure take MS milliseconds, modelling a call that
 * fails slowly. Sites:
 *   durable.{lock_open,lock_acquire,read_open,read,temp_create,write,fsync,
 *            close,rename,dir_open,dir_fsync,unlink,lock_unlink}
 *   writer.{open,append,truncate,archive}
 * ------------------------------------------------------------------------- */
struct FaultTable;

namespace detail {
/* Constant-initialised to nullptr; one relaxed load on every I/O call. */
extern std::atomic<const FaultTable*> g_fault_table;
int consume_fault_slow(const FaultTable& t, const char* site,
                       std::size_t* partial, bool* land) noexcept;
} // namespace detail

/** errno to inject at `site` (0 = none). When the matching rule carries
 *  partial=B and `partial` is non-null, *partial is set to B; when it
 *  carries `land` and `land` is non-null, *land is set to true. */
inline int consume_fault(const char* site, std::size_t* partial = nullptr,
                         bool* land = nullptr) noexcept
{
    const FaultTable* t = detail::g_fault_table.load(std::memory_order_relaxed);
    return t ? detail::consume_fault_slow(*t, site, partial, land) : 0;
}

namespace testing {
/** Install a fault table (same grammar as the environment variable; an empty
 *  spec installs "no faults", masking the environment) for this object's
 *  lifetime, restoring the previous table afterwards. Install and remove
 *  only while no other thread performs HPO file I/O. Throws
 *  std::invalid_argument on a malformed spec. */
class ScopedFaults {
public:
    explicit ScopedFaults(const std::string& spec);
    ~ScopedFaults();
    ScopedFaults(const ScopedFaults&) = delete;
    ScopedFaults& operator=(const ScopedFaults&) = delete;

    /** consume_fault() calls that matched `site` / failed at `site`. */
    unsigned long calls(const std::string& site) const;
    unsigned long injected(const std::string& site) const;

private:
    std::unique_ptr<FaultTable> table_;
    const FaultTable*           prev_ = nullptr;
};

/** For this object's lifetime, is_transient_errno(err, path) treats every
 *  path as on a network (true) or local (false) filesystem instead of asking
 *  statfs(2), so EACCES handling is testable on any disk. Same install
 *  contract as ScopedFaults. */
class ScopedNetworkFs {
public:
    explicit ScopedNetworkFs(bool network);
    ~ScopedNetworkFs();
    ScopedNetworkFs(const ScopedNetworkFs&) = delete;
    ScopedNetworkFs& operator=(const ScopedNetworkFs&) = delete;
private:
    int prev_;
};
} // namespace testing

} // namespace tadah::hpo::io

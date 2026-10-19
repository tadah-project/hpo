#pragma once

/*
 * Crash-safe best-potential publication.
 *
 * A candidate is compared while holding an advisory process lock, written to
 * a unique file in the destination directory, fsync'd, and atomically renamed
 * over the destination.  The directory is fsync'd before the lock is released
 * so a successful return means the rename is durable across a power loss.
 * Transient errors (EBUSY/ESTALE/EIO/... on network filesystems) retry the
 * whole transaction with bounded backoff (io_retry.h); after a shutdown
 * request (SIGINT/SIGTERM/SIGUSR2) retrying goes on for at most 2 s
 * (BackoffPolicy::stop_grace), then is abandoned.
 */

#include <tadah/hpo/io_retry.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tadah::hpo::durable_best {

struct Key {
    double      loss = 0.0;
    int         rank = 0;
    std::size_t slot = 0;
    std::size_t eval = 0;
};

/** Exact identity of the complete (loss, rank, slot, eval) tuple. */
inline bool operator==(const Key& a, const Key& b) noexcept
{
    return a.loss == b.loss && a.rank == b.rank &&
           a.slot == b.slot && a.eval == b.eval;
}
inline bool operator!=(const Key& a, const Key& b) noexcept { return !(a == b); }

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/** A failed system call: errno plus the operation (its fault-injection site,
 *  e.g. "durable.rename") and the path it touched. Transient errnos
 *  (io::is_transient_errno) are retried; the rest propagate at once.
 *  may_have_landed() (publish_if_better only): in some attempt the rename of
 *  the candidate over the destination may have happened, so the destination
 *  may hold the candidate although its durability was never confirmed --
 *  the rename returned 0 and the directory fsync failed, or it reported an
 *  errno a rename that landed can report (ENOENT: a retransmitted NFS
 *  rename; a transient errno on a network filesystem). A rename that failed
 *  with an errno meaning it did not happen (EROFS, EXDEV, ENOSPC, EDQUOT,
 *  EISDIR, ENOTDIR, EPERM, a local EACCES, ...) does not set it. */
class SysError : public Error {
public:
    SysError(const std::string& what, int err, std::string op, std::string path,
             bool may_have_landed = false)
        : Error(what), err_(err), op_(std::move(op)), path_(std::move(path)),
          may_have_landed_(may_have_landed) {}
    int                err()  const noexcept { return err_; }
    const std::string& op()   const noexcept { return op_; }
    const std::string& path() const noexcept { return path_; }
    bool may_have_landed()    const noexcept { return may_have_landed_; }
private:
    int         err_;
    std::string op_;
    std::string path_;
    bool        may_have_landed_ = false;
};

/** io::BackoffPolicy::durable() whose should_abort is
 *  core::SignalHandler::stop_requested: the default Options' policy. */
io::BackoffPolicy default_backoff();

/** Retry configuration. Each call retries its WHOLE transaction (lock, read,
 *  compare, replace) so every attempt re-evaluates the incumbent; locks are
 *  released between attempts. When the backoff budget is spent the call
 *  throws the last SysError with " (gave up after N attempts over T s)"
 *  appended. After backoff.should_abort reports a stop request (by default:
 *  a shutdown signal) retrying goes on for at most backoff.stop_grace (2 s,
 *  so a single transient blip still lands the candidate); when that is
 *  spent the call appends " (retry abandoned: shutdown requested) after N
 *  attempts over T s" -- the candidate was not persisted, so this is still
 *  an error. on_retry (optional) observes every retried failure; sleep
 *  (optional) replaces std::this_thread::sleep_for.
 *
 *  no_backoff makes a single attempt, never more persistent than the
 *  normal mode: a transient errno ends the call at once (" (gave up after
 *  1 attempt over 0 s)"), and a permanent one fails at once, as it does
 *  with a backoff. Only a rename reporting ENOENT (a rename that landed can
 *  report it) gets one immediate re-attempt of the whole transaction,
 *  reported to on_retry with RetryEvent::transient false. `backoff` and
 *  `sleep` are then unused; nothing sleeps. */
struct Options {
    io::BackoffPolicy backoff = default_backoff();
    io::OnRetryFn     on_retry;
    io::SleepFn       sleep;
    bool              no_backoff = false;
};

/** Strict total ordering used by every process. Lower loss wins; exact-loss
 * ties choose the lexicographically smaller (rank, slot, eval) tuple. */
bool precedes(const Key& candidate, const Key& incumbent) noexcept;

/** Parse the machine-readable metadata emitted by PotHeaderWriter.
 * Returns nullopt for legacy/unrecognised payloads. */
std::optional<Key> parse_key(const std::string& payload);

/** Read and parse the current file while holding the same process/thread
 * locks used by publication. Returns nullopt when the file is absent or uses
 * the legacy format. */
std::optional<Key> read_key_file(const std::string& path);
std::optional<Key> read_key_file(const std::string& path, const Options& opt);

/** Remove a prior run's artifact under the publication lock and fsync the
 * containing directory. Called at the start of every run: HPO runs are not
 * resumable, so a previous run's incumbent never carries over. */
void reset_for_fresh_run(const std::string& path);
void reset_for_fresh_run(const std::string& path, const Options& opt);

/** Remove the lock file of `path` (`path` + ".lock"), which every call above
 * creates and locks, at the end of a run. The lock is taken first, so a
 * publication that still holds it completes before the file goes; call it
 * only once no process publishes to `path` any more (after the run's last
 * collective), since a process that opened the old file would lock another
 * file than one that creates a new one. An absent lock file is not an
 * error. Throws SysError ("durable.lock_unlink") when the file cannot be
 * removed. */
void remove_lock_file(const std::string& path);

/** Publish payload at path only when key beats the current file's key.
 * Legacy files without machine metadata are replaced. Returns true when a
 * rename was committed, false when an equal-or-better file already existed.
 * Transient failures are retried (Options); a retry that finds its own
 * earlier rename already in place (a later step such as the directory fsync
 * had failed, or the rename itself reported ENOENT after landing) completes
 * that commit and returns true. Throws Error on a permanent, exhausted or
 * abandoned locking, I/O, fsync, or rename failure -- a SysError with
 * may_have_landed() set once the candidate's rename may have happened. The
 * 3-arg forms use default Options (default_backoff(), no callback). */
bool publish_if_better(const std::string& path,
                       const Key& key,
                       const std::string& payload);
bool publish_if_better(const std::string& path,
                       const Key& key,
                       const std::string& payload,
                       const Options& opt);

/** Failure latch for one run's best-potential publications. A publication
 *  that failed for good -- a permanent error, a spent retry budget, a
 *  retry abandoned after a shutdown request -- ends the run FAILED, and no
 *  later candidate can change that. Once the latch is tripped the owner
 *  publishes every later candidate with select()'s `latched` Options (no
 *  backoff, Options::no_backoff) instead of letting each candidate queued
 *  behind the failure spend a whole retry budget of its own (a DLIB:GFS
 *  THREADS 8 run with persistently failing storage took 8 x 30 s to end):
 *  failing storage ends the run after about one budget -- plus one call's
 *  duration per queued candidate when failing calls are slow -- and
 *  storage that has recovered still saves a later, lower best. The first failure stays
 *  the run's reported cause, and a later success does not reset the latch.
 *  Not synchronised: the owner serialises every call (HPO_Host holds its
 *  best mutex) and resets the latch for every run. */
class FailLatch {
public:
    /** The publication of eval `eval`'s candidate failed for good. The
     *  first trip is kept. */
    void trip(std::size_t eval) noexcept
    {
        if (!failed_eval_) failed_eval_ = eval;
    }
    void reset() noexcept { failed_eval_.reset(); }
    /** The eval whose failure tripped the latch; nullopt while open. */
    std::optional<std::size_t> failed_eval() const noexcept { return failed_eval_; }
    /** `normal` while open, `latched` once tripped. */
    const Options& select(const Options& normal,
                          const Options& latched) const noexcept
    {
        return failed_eval_ ? latched : normal;
    }
private:
    std::optional<std::size_t> failed_eval_;
};

} // namespace tadah::hpo::durable_best

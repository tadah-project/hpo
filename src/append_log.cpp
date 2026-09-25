#include <tadah/hpo/append_log.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace tadah::hpo::io {
namespace {

int open_append(const std::string& path)
{
    if (const int e = consume_fault("writer.open")) { errno = e; return -1; }
    int fd;
    do { fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0666); }
    while (fd < 0 && errno == EINTR);
    return fd;
}

int truncate_to(int fd, std::uint64_t size)
{
    if (const int e = consume_fault("writer.truncate")) { errno = e; return -1; }
    int rc;
    do { rc = ::ftruncate(fd, static_cast<off_t>(size)); }
    while (rc < 0 && errno == EINTR);
    return rc;
}

/* write(2) until done; short writes continue, EINTR repeats. An injected
 * fault with partial=B first writes B real bytes (a short write that then
 * fails). Returns 0 or errno; on failure *written (if given) is the number of
 * bytes this call did write before failing. */
int write_fully(int fd, const char* p, std::size_t left, const char* site,
                std::size_t* written = nullptr)
{
    std::size_t done = 0;
    auto fail = [&](int e) {
        if (written) *written = done;
        return e;
    };
    while (left > 0) {
        std::size_t partial = 0;
        if (const int e = consume_fault(site, &partial)) {
            if (partial > 0) {
                const ssize_t n = ::write(fd, p, std::min(partial, left));
                if (n > 0) done += static_cast<std::size_t>(n);
            }
            return fail(e);
        }
        const ssize_t n = ::write(fd, p, left);
        if (n < 0) {
            const int e = errno;
            if (e == EINTR) continue;
            return fail(e);
        }
        if (n == 0) return fail(EIO);   // no progress: treat as an I/O error
        p    += n;
        left -= static_cast<std::size_t>(n);
        done += static_cast<std::size_t>(n);
    }
    return 0;
}

double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/* Shared failure-path loop of both writers: `again` re-runs one complete
 * attempt and returns 0 or errno (updating `op`). Transient errnos
 * (is_transient_errno(err, path): EACCES only on a network filesystem) back
 * off while the sleeps, and the wall time since the first failure, each
 * plus the next delay stay within the policy budget (budget 0: no backoff);
 * a permanent errno gets one immediate re-attempt. After a stop request
 * (policy.should_abort) the loop goes on for at most policy.stop_grace
 * (StopGrace), then ends and sets `abandoned`. `single` (degraded mode)
 * allows no retry. */
template<class Again>
int retry_attempts(int err, const char*& op, const std::string& path,
                   const BackoffPolicy& policy, const SleepFn& sleep,
                   const OnRetryFn& on_retry, bool single,
                   unsigned& attempts, unsigned& retries, double& seconds,
                   bool& abandoned, Again&& again)
{
    if (single) return err;
    const auto t0 = std::chrono::steady_clock::now();
    std::chrono::milliseconds slept{0};
    int eacces_transient = -1;   // statfs at most once per call
    StopGrace grace;
    while (err != 0) {
        bool transient;
        if (err != EACCES) {
            transient = is_transient_errno(err);
        } else {
            if (eacces_transient < 0)
                eacces_transient = is_transient_errno(err, path) ? 1 : 0;
            transient = eacces_transient == 1;
        }
        const auto now = std::chrono::steady_clock::now();
        std::chrono::milliseconds delay{0};
        if (transient) {
            delay = policy.next_delay(attempts);
            if (policy.budget.count() <= 0 || slept + delay > policy.budget ||
                (now - t0) + delay > policy.budget)
                break;
        } else if (attempts >= 2) {
            break;
        }
        if (!grace.admit(policy, now, delay)) {
            abandoned = true;
            break;
        }
        if (on_retry) on_retry(RetryEvent{op, path, err, attempts, delay, transient});
        if (delay.count() > 0) {
            if (!sleep_for(sleep, delay,
                           grace.active() ? nullptr : policy.should_abort))
                grace.start(std::chrono::steady_clock::now());   // retry now
            else
                grace.slept(delay);
        }
        slept += delay;
        ++attempts;
        ++retries;
        err = again();
    }
    seconds = seconds_since(t0);
    return err;
}

} // namespace

/* ---------------------------------------------------------------------------
 * StageBuf
 * ------------------------------------------------------------------------- */
RetryingAppendFile::StageBuf::int_type
RetryingAppendFile::StageBuf::overflow(int_type ch)
{
    if (!traits_type::eq_int_type(ch, traits_type::eof()))
        s_.push_back(traits_type::to_char_type(ch));
    return traits_type::not_eof(ch);
}

std::streamsize RetryingAppendFile::StageBuf::xsputn(const char* p, std::streamsize n)
{
    s_.append(p, static_cast<std::size_t>(n));
    return n;
}

/* ---------------------------------------------------------------------------
 * RetryingAppendFile
 * ------------------------------------------------------------------------- */
RetryingAppendFile::RetryingAppendFile(std::string path)
    : path_(std::move(path)), os_(&buf_)
{
    fd_ = open_append(path_);
    if (fd_ < 0) { open_err_ = errno; return; }
    struct stat st {};
    if (::fstat(fd_, &st) < 0) {
        open_err_ = errno;
        close_fd();
        return;
    }
    offset_     = static_cast<std::uint64_t>(st.st_size);
    has_offset_ = true;
    dev_        = static_cast<std::uint64_t>(st.st_dev);
    ino_        = static_cast<std::uint64_t>(st.st_ino);
    has_id_     = true;
}

RetryingAppendFile::~RetryingAppendFile() { close_fd(); }

void RetryingAppendFile::close_fd() noexcept
{
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

/* Failure path: a failed write left `written` bytes of the batch behind.
 * Record where they start in the file that received them (an fstat of that
 * descriptor, before it is closed): with O_APPEND they landed at the file's
 * end, which is not offset_ when the file was truncated underneath us. */
void RetryingAppendFile::note_failed_write(std::size_t written) noexcept
{
    struct stat st {};
    if (fd_ >= 0 && ::fstat(fd_, &st) == 0) {
        const auto size  = static_cast<std::uint64_t>(st.st_size);
        const auto start = size >= written ? size - written : std::uint64_t{0};
        const auto dev   = static_cast<std::uint64_t>(st.st_dev);
        const auto ino   = static_cast<std::uint64_t>(st.st_ino);
        if (pend_ && pend_dev_ == dev && pend_ino_ == ino) {
            pend_start_ = std::min(pend_start_, start);   // an earlier rewind failed
        } else {
            pend_start_ = start;
            pend_dev_   = dev;
            pend_ino_   = ino;
        }
        pend_ = true;
    } else if (!pend_ && has_offset_ && has_id_) {
        /* The handle is gone (ESTALE): assume they followed offset_. */
        pend_start_ = offset_;
        pend_dev_   = dev_;
        pend_ino_   = ino_;
        pend_       = true;
    }
}

/* Where the file must end before a batch is (re)written: where a failed
 * write's own bytes start if they are in this file; else offset_, or the
 * file's end when it shrank, or when it is another file (replaced, or never
 * seen by the constructor). */
std::uint64_t RetryingAppendFile::rewind_target(std::uint64_t size, std::uint64_t dev,
                                                std::uint64_t ino) const noexcept
{
    if (pend_ && pend_dev_ == dev && pend_ino_ == ino)
        return has_offset_ ? std::min(offset_, pend_start_) : pend_start_;
    if (!has_offset_ || (has_id_ && (dev_ != dev || ino_ != ino)))
        return size;
    return std::min(offset_, size);
}

void RetryingAppendFile::adopt(std::uint64_t target, std::uint64_t dev,
                               std::uint64_t ino) noexcept
{
    offset_     = target;
    has_offset_ = true;
    dev_        = dev;
    ino_        = ino;
    has_id_     = true;
    pend_       = false;
}

/* Fresh descriptor for a retry: a stale handle (ESTALE, a dropped SMB
 * session) is not reused. Bytes a failed attempt left behind are truncated
 * away; a file that shrank underneath us (truncated or replaced externally)
 * is appended to where it now ends. */
int RetryingAppendFile::reopen_and_rewind(const char*& op)
{
    close_fd();
    op  = "writer.open";
    fd_ = open_append(path_);
    if (fd_ < 0) return open_err_ = errno;
    open_err_ = 0;
    struct stat st {};
    if (::fstat(fd_, &st) < 0) {
        const int e = errno;
        close_fd();
        return open_err_ = e;
    }
    const auto size   = static_cast<std::uint64_t>(st.st_size);
    const auto dev    = static_cast<std::uint64_t>(st.st_dev);
    const auto ino    = static_cast<std::uint64_t>(st.st_ino);
    const auto target = rewind_target(size, dev, ino);
    if (size > target) {
        op = "writer.truncate";
        if (truncate_to(fd_, target) < 0) return errno;   // pend_ kept: retried
    }
    adopt(target, dev, ino);
    op = "writer.append";
    return 0;
}

void RetryingAppendFile::rewind_best_effort() noexcept
{
    if (!has_offset_ && !pend_) return;
    if (fd_ < 0) {
        fd_ = open_append(path_);
        if (fd_ < 0) return;
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0) return;
    const auto size   = static_cast<std::uint64_t>(st.st_size);
    const auto dev    = static_cast<std::uint64_t>(st.st_dev);
    const auto ino    = static_cast<std::uint64_t>(st.st_ino);
    const auto target = rewind_target(size, dev, ino);
    /* A failed truncate keeps pend_: the next commit's reopen retries it. */
    if (size > target && truncate_to(fd_, target) < 0) return;
    adopt(target, dev, ino);
}

int RetryingAppendFile::write_batch(const std::string& batch)
{
    std::size_t written = 0;
    const int err = write_fully(fd_, batch.data(), batch.size(), "writer.append",
                                &written);
    if (err != 0 && written > 0) note_failed_write(written);
    return err;
}

RetryingAppendFile::Commit
RetryingAppendFile::commit(const BackoffPolicy& policy, const SleepFn& sleep,
                           const OnRetryFn& on_retry)
{
    Commit c;
    std::string& batch = buf_.data();
    os_.clear();   // a failed insertion must not silence later batches
    if (batch.empty()) return c;

    c.attempts = 1;
    c.degraded = degraded_;
    const char* op = "writer.append";
    int err;
    if (fd_ >= 0 && !degraded_) {
        err = write_batch(batch);
        if (err == 0) {                          // the success path
            offset_ += batch.size();
            batch.clear();
            return c;
        }
    } else {
        err = reopen_and_rewind(op);
        if (err == 0) err = write_batch(batch);
    }

    err = retry_attempts(err, op, path_, policy, sleep, on_retry, degraded_,
                         c.attempts, c.retries, c.seconds, c.abandoned, [&] {
                             int e = reopen_and_rewind(op);
                             return e != 0 ? e : write_batch(batch);
                         });

    if (err == 0) {
        offset_  += batch.size();
        degraded_ = false;
    } else {
        rewind_best_effort();
        c.ok        = false;
        c.err       = err;
        c.op        = op;
        c.rows_lost = static_cast<std::size_t>(
            std::count(batch.begin(), batch.end(), '\n'));
        degraded_   = true;
    }
    batch.clear();
    return c;
}

/* ---------------------------------------------------------------------------
 * write_whole_file
 * ------------------------------------------------------------------------- */
WriteFileResult write_whole_file(const std::string& path, std::string_view bytes,
                                 const BackoffPolicy& policy, const SleepFn& sleep,
                                 const OnRetryFn& on_retry)
{
    WriteFileResult r;
    const char* op = "writer.archive";
    /* Set once an attempt of THIS call has created or truncated the file:
     * only then is a failure's leftover ours to remove. */
    bool touched = false;
    auto attempt = [&]() -> int {
        int fd;
        do { fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666); }
        while (fd < 0 && errno == EINTR);
        if (fd < 0) return errno;
        touched = true;
        int err = write_fully(fd, bytes.data(), bytes.size(), "writer.archive");
        /* close() reports deferred write-back errors on NFS/SMB. */
        if (::close(fd) < 0 && err == 0) err = errno;
        return err;
    };

    r.attempts = 1;
    int err = attempt();
    if (err != 0)
        err = retry_attempts(err, op, path, policy, sleep, on_retry, false,
                             r.attempts, r.retries, r.seconds, r.abandoned,
                             attempt);
    if (err != 0) {
        /* No partial archive -- but never delete a file this call could not
         * even open (a read-only archive kept from an earlier run). */
        if (touched) ::unlink(path.c_str());
        r.ok  = false;
        r.err = err;
        r.op  = op;
    }
    return r;
}

} // namespace tadah::hpo::io

#pragma once

/*
 * Retrying writers for the per-evaluation logs (loss.tadah, outvar.tadah,
 * params.tadah, stats.tadah and their best_* twins) and the DUMP/BDUMP
 * potential archives.
 *
 * RetryingAppendFile collects one writer drain's rows in memory (stage())
 * and appends them with one write(2) per batch (commit()). It tracks the
 * byte offset where the batch starts, so after a failure -- including a
 * partial write -- it reopens the file, truncates it back to where the batch
 * started and rewrites the batch: every committed batch appears exactly once
 * and whole. A failed write's own bytes are located with an fstat of the
 * descriptor that received them, so this holds even when the file was
 * truncated in place (": > loss.tadah") before the failure; a file replaced
 * by another inode is appended to where it ends. Transient errnos
 * (io::is_transient_errno(err, path)) are retried with backoff; permanent
 * ones get one immediate re-attempt. After a stop request (the policy's
 * should_abort) retrying goes on for at most the policy's stop_grace. When
 * the budget or that grace is spent, the batch is dropped
 * (rows_lost), the file is truncated back to the batch start, and the writer
 * enters degraded mode: later failures get a single attempt and no sleep
 * until a commit succeeds again, so a dead mount cannot stall every drain.
 *
 * "Committed" means write(2) accepted the bytes. There is deliberately no
 * fsync per batch (it would cost every drain a round trip to the server), so
 * a write-back error that NFS/SMB reports only later, at fsync or close, is
 * not seen and the batch counts as committed. The durable best potential
 * (durable_best.h) is fsync'd; these logs are not.
 *
 * Success path: the formatting goes into a retained buffer (no allocation
 * once it has grown to a drain's size) and one write(2) per non-empty batch;
 * no clock read, no fstat.
 */

#include <tadah/hpo/io_retry.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <streambuf>
#include <string>
#include <string_view>

namespace tadah::hpo::io {

class RetryingAppendFile {
public:
    /** Opens path with O_WRONLY|O_APPEND|O_CREAT and records its size as the
     *  committed offset. An open failure is not thrown: the first commit
     *  retries the open under its policy. */
    explicit RetryingAppendFile(std::string path);
    ~RetryingAppendFile();
    RetryingAppendFile(const RetryingAppendFile&) = delete;
    RetryingAppendFile& operator=(const RetryingAppendFile&) = delete;

    /** The pending batch. Format rows into it; its formatting state persists
     *  across batches like a long-lived std::ofstream. A row is a line: a
     *  dropped batch reports its complete lines as rows_lost. */
    std::ostream& stage() noexcept { return os_; }

    struct Commit {
        bool        ok        = true;
        unsigned    attempts  = 0;   // write attempts made (0: nothing to write)
        unsigned    retries   = 0;   // attempts after the first
        std::size_t rows_lost = 0;   // complete lines of the dropped batch
        int         err       = 0;   // last errno on failure
        std::string op;              // failing step: writer.open/append/truncate
        double      seconds   = 0.0; // time spent in the failure path
        bool        degraded  = false; // single attempt: an earlier batch failed
        bool        abandoned = false; // the stop_grace after a stop request ran out
    };
    /** Append the staged batch (no-op when empty) and clear it. */
    Commit commit(const BackoffPolicy& policy, const SleepFn& sleep = {},
                  const OnRetryFn& on_retry = {});

    const std::string& path()   const noexcept { return path_; }
    bool          degraded()    const noexcept { return degraded_; }
    /** Bytes of the file this writer considers committed. */
    std::uint64_t offset()      const noexcept { return offset_; }
    std::size_t   pending_bytes() const noexcept { return buf_.data().size(); }

private:
    /* std::streambuf appending to a std::string that keeps its capacity
     * across batches (std::ostringstream::str("") would free it). */
    class StageBuf : public std::streambuf {
    public:
        std::string&       data()       noexcept { return s_; }
        const std::string& data() const noexcept { return s_; }
    protected:
        int_type overflow(int_type ch) override;
        std::streamsize xsputn(const char* p, std::streamsize n) override;
    private:
        std::string s_;
    };

    int  write_batch(const std::string& batch);          // 0 or errno
    int  reopen_and_rewind(const char*& op);             // 0 or errno
    void rewind_best_effort() noexcept;
    void close_fd() noexcept;
    void note_failed_write(std::size_t written) noexcept;
    std::uint64_t rewind_target(std::uint64_t size, std::uint64_t dev,
                                std::uint64_t ino) const noexcept;
    void adopt(std::uint64_t target, std::uint64_t dev, std::uint64_t ino) noexcept;

    std::string   path_;
    int           fd_       = -1;
    int           open_err_ = 0;
    std::uint64_t offset_   = 0;
    bool          has_offset_ = false;  // offset_ came from an fstat
    bool          degraded_ = false;
    /* Failure path only. dev_/ino_: the file offset_ refers to (from the
     * same fstat). pend_*: where the bytes a failed write left behind start,
     * and in which file, until a rewind removes them. */
    std::uint64_t dev_ = 0, ino_ = 0;
    bool          has_id_ = false;
    std::uint64_t pend_start_ = 0, pend_dev_ = 0, pend_ino_ = 0;
    bool          pend_ = false;
    StageBuf      buf_;
    std::ostream  os_;
};

/* Whole-file writer for potential archives: O_TRUNC write + close, retried
 * under `policy`. On exhaustion a file this call opened (created or
 * truncated) is unlinked, so an archive either exists complete or not at
 * all; a file it never managed to open (a read-only archive a user kept) is
 * left untouched. */
struct WriteFileResult {
    bool        ok        = true;
    unsigned    attempts  = 0;
    unsigned    retries   = 0;
    int         err       = 0;
    std::string op;
    double      seconds   = 0.0;
    bool        abandoned = false;   // the stop_grace after a stop request ran out
};
WriteFileResult write_whole_file(const std::string& path, std::string_view bytes,
                                 const BackoffPolicy& policy,
                                 const SleepFn& sleep = {},
                                 const OnRetryFn& on_retry = {});

} // namespace tadah::hpo::io

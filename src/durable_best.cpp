#include <tadah/hpo/durable_best.h>

#include <tadah/core/signal_handler.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include <utility>

namespace tadah::hpo::durable_best {
namespace {

namespace io = tadah::hpo::io;

std::mutex& process_mutex()
{
    /* POSIX record locks are owned by a process, not a thread.  This mutex is
     * therefore required in addition to fcntl() for concurrent HPO workers in
     * one rank. */
    static std::mutex m;
    return m;
}

std::atomic<unsigned long long>& temp_counter()
{
    static std::atomic<unsigned long long> n{0};
    return n;
}

std::string sys_message(const std::string& action,
                        const std::string& path,
                        int err)
{
    return action + " '" + path + "': " + std::strerror(err);
}

[[noreturn]] void throw_sys(const std::string& action, const char* site,
                            const std::string& path, int err)
{
    throw SysError(sys_message(action, path, err), err, site, path);
}

/* Every system call goes through one of these: first the test fault site
 * (one relaxed atomic load when no fault table is installed), then the real
 * call. An injected fault sets errno and fails like the real call would. */
int sys_open(const char* site, const char* path, int flags, mode_t mode = 0)
{
    if (const int e = io::consume_fault(site)) { errno = e; return -1; }
    return ::open(path, flags, mode);
}

int sys_setlkw(int fd, struct flock* fl)
{
    if (const int e = io::consume_fault("durable.lock_acquire")) { errno = e; return -1; }
    return ::fcntl(fd, F_SETLKW, fl);
}

ssize_t sys_read(int fd, void* buf, std::size_t n)
{
    if (const int e = io::consume_fault("durable.read")) { errno = e; return -1; }
    return ::read(fd, buf, n);
}

ssize_t sys_write(int fd, const void* buf, std::size_t n)
{
    if (const int e = io::consume_fault("durable.write")) { errno = e; return -1; }
    return ::write(fd, buf, n);
}

int sys_fsync(const char* site, int fd)
{
    if (const int e = io::consume_fault(site)) { errno = e; return -1; }
    return ::fsync(fd);
}

int sys_close(int fd)
{
    /* The descriptor is released either way; only the result is faked. */
    const int rc = ::close(fd);
    if (const int e = io::consume_fault("durable.close")) { errno = e; return -1; }
    return rc;
}

int sys_rename(const char* from, const char* to)
{
    bool land = false;
    if (const int e = io::consume_fault("durable.rename", nullptr, &land)) {
        /* `land`: the rename happens, the caller still sees the error. */
        if (land) (void)::rename(from, to);
        errno = e;
        return -1;
    }
    return ::rename(from, to);
}

int sys_unlink(const char* path)
{
    if (const int e = io::consume_fault("durable.unlink")) { errno = e; return -1; }
    return ::unlink(path);
}

class Fd {
public:
    explicit Fd(int fd = -1) noexcept : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    Fd& operator=(Fd&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
    int get() const noexcept { return fd_; }
    int release() noexcept { const int out = fd_; fd_ = -1; return out; }
private:
    int fd_;
};

class RecordLock {
public:
    explicit RecordLock(const std::string& path)
        : path_(path),
          fd_(sys_open("durable.lock_open", path.c_str(), O_CREAT | O_RDWR, 0666))
    {
        if (fd_.get() < 0)
            throw_sys("cannot open durable-best lock", "durable.lock_open", path_, errno);

        struct flock fl {};
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        int rc;
        do { rc = sys_setlkw(fd_.get(), &fl); }
        while (rc < 0 && errno == EINTR);
        if (rc < 0)
            throw_sys("cannot acquire durable-best lock", "durable.lock_acquire", path_, errno);
    }

    RecordLock(const RecordLock&) = delete;
    RecordLock& operator=(const RecordLock&) = delete;

private:
    std::string path_;
    Fd fd_; // close releases this process's lock on the lock file
};

std::optional<std::string> read_existing(const std::string& path)
{
    Fd fd(sys_open("durable.read_open", path.c_str(), O_RDONLY));
    if (fd.get() < 0) {
        if (errno == ENOENT) return std::nullopt;
        throw_sys("cannot read incumbent durable best", "durable.read_open", path, errno);
    }

    /* Metadata is the first five short lines. Keep the synchronous lock hold
     * independent of the size of the full potential that follows. */
    constexpr std::size_t max_header = 16 * 1024;
    std::string out;
    out.resize(max_header);
    std::size_t used = 0;
    while (used < max_header) {
        const ssize_t n = sys_read(fd.get(), out.data() + used, max_header - used);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw_sys("cannot read incumbent durable best", "durable.read", path, errno);
        }
        if (n == 0) break;
        used += static_cast<std::size_t>(n);
    }
    out.resize(used);
    return out;
}

void write_all(int fd, const std::string& payload, const std::string& path)
{
    std::size_t off = 0;
    while (off < payload.size()) {
        const ssize_t n = sys_write(fd, payload.data() + off, payload.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw_sys("cannot write durable-best temporary", "durable.write", path, errno);
        }
        if (n == 0)
            throw Error("short write to durable-best temporary '" + path + "'");
        off += static_cast<std::size_t>(n);
    }
}

void fsync_checked(int fd, const char* site, const std::string& what,
                   const std::string& path)
{
    int rc;
    do { rc = sys_fsync(site, fd); } while (rc < 0 && errno == EINTR);
    if (rc < 0) throw_sys(what, site, path, errno);
}

std::filesystem::path parent_dir(const std::string& path)
{
    const std::filesystem::path target(path);
    return target.has_parent_path() ? target.parent_path()
                                    : std::filesystem::path(".");
}

/* Make a completed rename/unlink in `dir` durable. */
void fsync_dir(const std::filesystem::path& dir)
{
    Fd dir_fd(sys_open("durable.dir_open", dir.c_str(), O_RDONLY | O_DIRECTORY));
    if (dir_fd.get() < 0)
        throw_sys("cannot open durable-best directory", "durable.dir_open",
                  dir.string(), errno);
    fsync_checked(dir_fd.get(), "durable.dir_fsync",
                  "cannot fsync durable-best directory", dir.string());
}

/* may_have_landed is set once the candidate may be at `path`: the rename
 * returned 0 (a later directory fsync can still fail), or it failed with an
 * errno that a rename which landed can report -- ENOENT (a retransmitted
 * NFS rename finds its source gone) or a transient one (network
 * filesystems; EACCES only there). Any other errno (EROFS, EXDEV, ENOSPC,
 * EDQUOT, EISDIR, ENOTDIR, EPERM, ...) means the destination is untouched.
 * The caller never resets it between attempts: a retry must recognise its
 * own payload as the incumbent, and the RESULT section reports whether the
 * file may hold the candidate. */
void atomic_replace(const std::string& path, const std::string& payload,
                    bool& may_have_landed)
{
    namespace fs = std::filesystem;
    const fs::path target(path);
    const fs::path dir = parent_dir(path);
    const std::string base = target.filename().string();

    fs::path tmp;
    Fd tmp_fd;
    for (unsigned attempt = 0; attempt < 128; ++attempt) {
        const auto serial = temp_counter().fetch_add(1, std::memory_order_relaxed);
        tmp = dir / ("." + base + ".tmp." + std::to_string(::getpid()) +
                     "." + std::to_string(serial));
        const int fd = sys_open("durable.temp_create", tmp.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd >= 0) {
            tmp_fd = Fd(fd);
            break;
        }
        if (errno != EEXIST)
            throw_sys("cannot create durable-best temporary", "durable.temp_create",
                      tmp.string(), errno);
    }
    if (tmp_fd.get() < 0)
        throw Error("cannot allocate a unique durable-best temporary for '" + path + "'");

    bool renamed = false;
    try {
        write_all(tmp_fd.get(), payload, tmp.string());
        fsync_checked(tmp_fd.get(), "durable.fsync",
                      "cannot fsync durable-best temporary", tmp.string());
        const int raw_fd = tmp_fd.release();
        const int close_rc = sys_close(raw_fd);
        if (close_rc < 0)
            throw_sys("cannot close durable-best temporary", "durable.close",
                      tmp.string(), errno);

        /* Deterministic crash-boundary regression hook. It is intentionally
         * private (not a supported runtime knob): the test child sets it
         * immediately after fork, stops here, and is SIGKILLed by its parent.
         * The normal environment never contains this test-only name. */
        if (const char* hook =
                std::getenv("TADAH_HPO_TEST_STOP_DURABLE_BEST_BEFORE_RENAME");
            hook && std::string(hook) == "1") {
            if (::raise(SIGSTOP) != 0)
                throw Error(sys_message("cannot stop durable-best test child",
                                        tmp.string(), errno));
        }

        if (sys_rename(tmp.c_str(), target.c_str()) < 0) {
            const int err = errno;   // before is_transient_errno's statfs
            if (err == ENOENT || io::is_transient_errno(err, path))
                may_have_landed = true;
            throw_sys("cannot rename durable-best temporary", "durable.rename",
                      path, err);
        }
        may_have_landed = true;
        renamed = true;

        fsync_dir(dir);
    } catch (...) {
        if (!renamed) ::unlink(tmp.c_str());
        throw;
    }
}

/* Transient SysErrors are retried; everything else propagates at once.
 * EACCES is transient only on a network filesystem (io::is_transient_errno
 * with the path). A rename reporting ENOENT is retried too: on NFS a
 * retransmitted rename that already landed fails that way (its source is
 * gone), and the next attempt re-reads the incumbent under the lock -- its
 * own key there proves the rename landed (idempotent path in
 * publish_if_better); otherwise the transaction simply runs again, and a
 * directory that really vanished fails the next attempt's lock open with a
 * permanent ENOENT. */
bool classify(const std::exception& e, io::RetryEvent& ev)
{
    const auto* se = dynamic_cast<const SysError*>(&e);
    if (!se) return false;
    const bool rename_enoent = se->err() == ENOENT && se->op() == "durable.rename";
    if (!rename_enoent && !io::is_transient_errno(se->err(), se->path()))
        return false;
    ev.op   = rename_enoent ? se->op() + " (possible landed rename)" : se->op();
    ev.path = se->path();
    ev.err  = se->err();
    return true;
}

/* Options::no_backoff: never more persistent than the normal mode. A
 * transient errno ends the call at once (one attempt, no backoff), as does
 * every permanent errno. The one non-transient failure classify() retries
 * -- a rename reporting ENOENT, possibly landed -- gets one immediate
 * re-attempt, whose idempotent path completes a rename that did land. */
template<class Op>
auto transact_no_backoff(const Options& opt, Op& op) -> decltype(op())
{
    std::chrono::steady_clock::time_point first_failure{};
    for (unsigned attempt = 1;; ++attempt) {
        try {
            return op();
        } catch (const SysError& e) {
            const auto now = std::chrono::steady_clock::now();
            if (attempt == 1) first_failure = now;   // as io::retry counts T
            if (io::is_transient_errno(e.err(), e.path()))
                throw SysError(e.what() + io::gave_up_suffix(attempt, now - first_failure),
                               e.err(), e.op(), e.path(), e.may_have_landed());
            const bool rename_enoent = e.err() == ENOENT && e.op() == "durable.rename";
            if (!rename_enoent || attempt >= 2) throw;
            if (opt.on_retry) {
                io::RetryEvent ev;
                ev.op         = e.op() + " (possible landed rename)";
                ev.path       = e.path();
                ev.err        = e.err();
                ev.attempt    = attempt;
                ev.transient  = false;
                opt.on_retry(static_cast<const io::RetryEvent&>(ev));
            }
        }
    }
}

/* Run one complete locked transaction under the retry policy. */
template<class Op>
auto transact(const Options& opt, Op&& op) -> decltype(op())
{
    if (opt.no_backoff) return transact_no_backoff(opt, op);
    return io::retry(
        opt.backoff, std::forward<Op>(op), classify,
        [&opt](const io::RetryEvent& ev) { if (opt.on_retry) opt.on_retry(ev); },
        opt.sleep,
        [](const std::exception& last, unsigned attempts,
           std::chrono::steady_clock::duration elapsed, io::GiveUp why) {
            const auto& se = static_cast<const SysError&>(last);
            throw SysError(se.what() + io::gave_up_suffix(attempts, elapsed, why),
                           se.err(), se.op(), se.path());
        });
}

const Options& default_options()
{
    static const Options opt;
    return opt;
}

} // namespace

io::BackoffPolicy default_backoff()
{
    io::BackoffPolicy p = io::BackoffPolicy::durable();
    p.should_abort = &core::SignalHandler::stop_requested;
    return p;
}

bool precedes(const Key& candidate, const Key& incumbent) noexcept
{
    if (!std::isfinite(candidate.loss)) return false;
    if (!std::isfinite(incumbent.loss)) return true;
    if (candidate.loss < incumbent.loss) return true;
    if (candidate.loss > incumbent.loss) return false;
    return std::tie(candidate.rank, candidate.slot, candidate.eval) <
           std::tie(incumbent.rank, incumbent.slot, incumbent.eval);
}

std::optional<Key> read_key_file(const std::string& path)
{
    return read_key_file(path, default_options());
}

std::optional<Key> read_key_file(const std::string& path, const Options& opt)
{
    if (path.empty()) throw Error("durable-best path is empty");

    return transact(opt, [&]() -> std::optional<Key> {
        std::lock_guard<std::mutex> thread_lock(process_mutex());
        RecordLock process_lock(path + ".lock");
        const auto payload = read_existing(path);
        return payload ? parse_key(*payload) : std::nullopt;
    });
}

void reset_for_fresh_run(const std::string& path)
{
    reset_for_fresh_run(path, default_options());
}

void reset_for_fresh_run(const std::string& path, const Options& opt)
{
    if (path.empty()) throw Error("durable-best path is empty");

    /* After a successful unlink a retry sees ENOENT; it must still make that
     * removal durable instead of returning early. */
    bool unlinked = false;
    transact(opt, [&] {
        std::lock_guard<std::mutex> thread_lock(process_mutex());
        RecordLock process_lock(path + ".lock");

        if (sys_unlink(path.c_str()) < 0) {
            if (errno != ENOENT)
                throw_sys("cannot remove prior durable best", "durable.unlink",
                          path, errno);
            if (!unlinked) return;
        }
        unlinked = true;
        fsync_dir(parent_dir(path));
    });
}

std::optional<Key> parse_key(const std::string& payload)
{
    Key key;
    int format_version = 0;
    bool have_format = false, have_loss = false, have_rank = false,
         have_slot = false, have_eval = false;
    std::istringstream input(payload);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string hash, name;
        if (!(fields >> hash >> name) || hash != "#") continue;
        if (name == "HPO_BEST_FORMAT") {
            have_format = static_cast<bool>(fields >> format_version) &&
                          (format_version == 1 || format_version == 2);
        } else if (name == "HPO_BEST_LOSS") {
            have_loss = static_cast<bool>(fields >> key.loss);
        } else if (name == "HPO_SOURCE_RANK") {
            have_rank = static_cast<bool>(fields >> key.rank);
        } else if (name == "HPO_SOURCE_SLOT") {
            have_slot = static_cast<bool>(fields >> key.slot);
        } else if (name == "HPO_SOURCE_EVAL") {
            have_eval = static_cast<bool>(fields >> key.eval);
        }
        if (have_format && have_loss && have_rank && have_slot && have_eval) break;
    }
    if (!have_format || !have_loss || !have_rank || !have_slot || !have_eval ||
        !std::isfinite(key.loss) || key.rank < 0)
        return std::nullopt;
    /* FORMAT 1 wrote the slot 0-based; FORMAT 2 writes it 1-based. Normalise
     * to the internal 0-based value either way, so a Key parsed from an old
     * best_pot.tadah still orders correctly against a freshly built one. */
    if (format_version >= 2) {
        if (key.slot == 0) return std::nullopt;   /* 1-based: 0 is malformed */
        key.slot -= 1;
    }
    return key;
}

bool publish_if_better(const std::string& path,
                       const Key& key,
                       const std::string& payload)
{
    return publish_if_better(path, key, payload, default_options());
}

bool publish_if_better(const std::string& path,
                       const Key& key,
                       const std::string& payload,
                       const Options& opt)
{
    if (path.empty()) throw Error("durable-best path is empty");
    if (!std::isfinite(key.loss)) throw Error("durable-best loss is not finite");
    if (key.rank < 0) throw Error("durable-best rank is negative");
    if (payload.empty()) throw Error("durable-best payload is empty");
    const auto payload_key = parse_key(payload);
    if (!payload_key || payload_key->loss != key.loss ||
        payload_key->rank != key.rank || payload_key->slot != key.slot ||
        payload_key->eval != key.eval)
        throw Error("durable-best payload metadata does not match its comparison key");

    /* Each attempt is the complete transaction: its locks are released
     * before the backoff sleep and the incumbent is re-read, so a better key
     * published meanwhile by another thread/process still wins. */
    /* Sticky across attempts (atomic_replace). */
    bool may_have_landed = false;
    try {
        return transact(opt, [&] {
            std::lock_guard<std::mutex> thread_lock(process_mutex());
            RecordLock process_lock(path + ".lock");

            if (const auto incumbent_payload = read_existing(path)) {
                if (const auto incumbent = parse_key(*incumbent_payload)) {
                    /* Idempotent retry: an earlier attempt's rename landed and
                     * a later step (directory fsync) failed, or the rename
                     * reported an error after landing. Finish that commit. */
                    if (may_have_landed && *incumbent == key) {
                        fsync_dir(parent_dir(path));
                        return true;
                    }
                    if (!precedes(key, *incumbent)) return false;
                }
            }
            atomic_replace(path, payload, may_have_landed);
            return true;
        });
    } catch (const SysError& e) {
        /* The caller reports whether the file may hold this candidate. */
        if (!may_have_landed || e.may_have_landed()) throw;
        throw SysError(e.what(), e.err(), e.op(), e.path(), true);
    }
}

} // namespace tadah::hpo::durable_best

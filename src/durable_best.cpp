#include <tadah/hpo/durable_best.h>

#include <atomic>
#include <cerrno>
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

namespace tadah::hpo::durable_best {
namespace {

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
        : path_(path), fd_(::open(path.c_str(), O_CREAT | O_RDWR, 0666))
    {
        if (fd_.get() < 0)
            throw Error(sys_message("cannot open durable-best lock", path_, errno));

        struct flock fl {};
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0;
        int rc;
        do { rc = ::fcntl(fd_.get(), F_SETLKW, &fl); }
        while (rc < 0 && errno == EINTR);
        if (rc < 0)
            throw Error(sys_message("cannot acquire durable-best lock", path_, errno));
    }

    RecordLock(const RecordLock&) = delete;
    RecordLock& operator=(const RecordLock&) = delete;

private:
    std::string path_;
    Fd fd_; // close releases this process's lock on the lock file
};

std::optional<std::string> read_existing(const std::string& path)
{
    Fd fd(::open(path.c_str(), O_RDONLY));
    if (fd.get() < 0) {
        if (errno == ENOENT) return std::nullopt;
        throw Error(sys_message("cannot read incumbent durable best", path, errno));
    }

    /* Metadata is the first five short lines. Keep the synchronous lock hold
     * independent of the size of the full potential that follows. */
    constexpr std::size_t max_header = 16 * 1024;
    std::string out;
    out.resize(max_header);
    std::size_t used = 0;
    while (used < max_header) {
        const ssize_t n = ::read(fd.get(), out.data() + used, max_header - used);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw Error(sys_message("cannot read incumbent durable best", path, errno));
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
        const ssize_t n = ::write(fd, payload.data() + off, payload.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw Error(sys_message("cannot write durable-best temporary", path, errno));
        }
        if (n == 0)
            throw Error("short write to durable-best temporary '" + path + "'");
        off += static_cast<std::size_t>(n);
    }
}

void fsync_checked(int fd, const std::string& what, const std::string& path)
{
    int rc;
    do { rc = ::fsync(fd); } while (rc < 0 && errno == EINTR);
    if (rc < 0) throw Error(sys_message(what, path, errno));
}

void atomic_replace(const std::string& path, const std::string& payload)
{
    namespace fs = std::filesystem;
    const fs::path target(path);
    const fs::path dir = target.has_parent_path() ? target.parent_path() : fs::path(".");
    const std::string base = target.filename().string();

    fs::path tmp;
    Fd tmp_fd;
    for (unsigned attempt = 0; attempt < 128; ++attempt) {
        const auto serial = temp_counter().fetch_add(1, std::memory_order_relaxed);
        tmp = dir / ("." + base + ".tmp." + std::to_string(::getpid()) +
                     "." + std::to_string(serial));
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd >= 0) {
            tmp_fd = Fd(fd);
            break;
        }
        if (errno != EEXIST)
            throw Error(sys_message("cannot create durable-best temporary",
                                    tmp.string(), errno));
    }
    if (tmp_fd.get() < 0)
        throw Error("cannot allocate a unique durable-best temporary for '" + path + "'");

    bool renamed = false;
    try {
        write_all(tmp_fd.get(), payload, tmp.string());
        fsync_checked(tmp_fd.get(), "cannot fsync durable-best temporary", tmp.string());
        const int raw_fd = tmp_fd.release();
        const int close_rc = ::close(raw_fd);
        if (close_rc < 0)
            throw Error(sys_message("cannot close durable-best temporary", tmp.string(), errno));

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

        if (::rename(tmp.c_str(), target.c_str()) < 0)
            throw Error(sys_message("cannot rename durable-best temporary", path, errno));
        renamed = true;

        Fd dir_fd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY));
        if (dir_fd.get() < 0)
            throw Error(sys_message("cannot open durable-best directory", dir.string(), errno));
        fsync_checked(dir_fd.get(), "cannot fsync durable-best directory", dir.string());
    } catch (...) {
        if (!renamed) ::unlink(tmp.c_str());
        throw;
    }
}

} // namespace

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
    if (path.empty()) throw Error("durable-best path is empty");

    std::lock_guard<std::mutex> thread_lock(process_mutex());
    RecordLock process_lock(path + ".lock");
    const auto payload = read_existing(path);
    return payload ? parse_key(*payload) : std::nullopt;
}

void reset_for_fresh_run(const std::string& path)
{
    namespace fs = std::filesystem;
    if (path.empty()) throw Error("durable-best path is empty");

    std::lock_guard<std::mutex> thread_lock(process_mutex());
    RecordLock process_lock(path + ".lock");

    if (::unlink(path.c_str()) < 0) {
        if (errno == ENOENT) return;
        throw Error(sys_message("cannot remove prior durable best", path, errno));
    }

    const fs::path target(path);
    const fs::path dir = target.has_parent_path() ? target.parent_path() : fs::path(".");
    Fd dir_fd(::open(dir.c_str(), O_RDONLY | O_DIRECTORY));
    if (dir_fd.get() < 0)
        throw Error(sys_message("cannot open durable-best directory", dir.string(), errno));
    fsync_checked(dir_fd.get(), "cannot fsync durable-best directory", dir.string());
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
    if (path.empty()) throw Error("durable-best path is empty");
    if (!std::isfinite(key.loss)) throw Error("durable-best loss is not finite");
    if (key.rank < 0) throw Error("durable-best rank is negative");
    if (payload.empty()) throw Error("durable-best payload is empty");
    const auto payload_key = parse_key(payload);
    if (!payload_key || payload_key->loss != key.loss ||
        payload_key->rank != key.rank || payload_key->slot != key.slot ||
        payload_key->eval != key.eval)
        throw Error("durable-best payload metadata does not match its comparison key");

    std::lock_guard<std::mutex> thread_lock(process_mutex());
    RecordLock process_lock(path + ".lock");

    if (const auto incumbent_payload = read_existing(path)) {
        if (const auto incumbent = parse_key(*incumbent_payload)) {
            if (!precedes(key, *incumbent)) return false;
        }
    }
    atomic_replace(path, payload);
    return true;
}

} // namespace tadah::hpo::durable_best

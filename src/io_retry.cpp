#include <tadah/hpo/io_retry.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#if defined(__linux__)
#include <sys/statfs.h>
#endif
#include <thread>
#include <vector>

namespace tadah::hpo::io {

/* ---------------------------------------------------------------------------
 * Fault table
 * ------------------------------------------------------------------------- */
struct FaultRule {
    std::string   site;
    int           err     = 0;
    unsigned long skip    = 0;
    unsigned long count   = 1;
    bool          always  = false;
    std::size_t   partial = 0;
    bool          has_partial = false;
    bool          land    = false;
    unsigned long latency_ms = 0;
    mutable std::atomic<unsigned long> calls{0};
    mutable std::atomic<unsigned long> injected{0};
};

struct FaultTable {
    std::deque<FaultRule> rules;   // deque: FaultRule holds atomics (immovable)
};

namespace detail {
std::atomic<const FaultTable*> g_fault_table{nullptr};

int consume_fault_slow(const FaultTable& t, const char* site,
                       std::size_t* partial, bool* land) noexcept
{
    int out = 0;
    unsigned long latency_ms = 0;
    for (const auto& r : t.rules) {
        if (std::strcmp(r.site.c_str(), site) != 0) continue;
        const unsigned long n = r.calls.fetch_add(1, std::memory_order_relaxed);
        if (out != 0 || n < r.skip) continue;
        if (!r.always && n - r.skip >= r.count) continue;
        r.injected.fetch_add(1, std::memory_order_relaxed);
        out = r.err;
        if (partial && r.has_partial) *partial = r.partial;
        if (land && r.land) *land = true;
        latency_ms = r.latency_ms;
    }
    if (latency_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(latency_ms));
    return out;
}
} // namespace detail

namespace {

struct ErrnoName { int value; const char* name; };
constexpr ErrnoName kErrnoNames[] = {
    {EBUSY, "EBUSY"},     {EAGAIN, "EAGAIN"},       {EACCES, "EACCES"},
    {ESTALE, "ESTALE"},   {ETXTBSY, "ETXTBSY"},     {EIO, "EIO"},
    {ENOLCK, "ENOLCK"},   {ETIMEDOUT, "ETIMEDOUT"}, {EDEADLK, "EDEADLK"},
    {ENOSPC, "ENOSPC"},   {EDQUOT, "EDQUOT"},       {EROFS, "EROFS"},
    {ENOENT, "ENOENT"},   {ENOTDIR, "ENOTDIR"},     {EISDIR, "EISDIR"},
    {EPERM, "EPERM"},     {EINVAL, "EINVAL"},       {EMFILE, "EMFILE"},
    {EBADF, "EBADF"},     {EEXIST, "EEXIST"},       {EXDEV, "EXDEV"},
    {EINTR, "EINTR"},     {EFBIG, "EFBIG"},         {ENFILE, "ENFILE"},
};

int parse_errno(const std::string& tok)
{
    if (tok == "EWOULDBLOCK") return EWOULDBLOCK;
    for (const auto& e : kErrnoNames)
        if (tok == e.name) return e.value;
    char* end = nullptr;
    const long v = std::strtol(tok.c_str(), &end, 10);
    if (tok.empty() || *end != '\0' || v <= 0 || v > 4095)
        throw std::invalid_argument("unknown errno '" + tok + "'");
    return static_cast<int>(v);
}

unsigned long parse_ulong(const std::string& key, const std::string& val)
{
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(val.c_str(), &end, 10);
    if (val.empty() || val[0] == '-' || *end != '\0' || errno == ERANGE)
        throw std::invalid_argument("bad value '" + val + "' for " + key);
    return static_cast<unsigned long>(v);
}

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::string item;
    std::istringstream in(s);
    while (std::getline(in, item, sep)) out.push_back(trim(item));
    return out;
}

/* "site,ERRNO[,skip=K][,count=N|always][,partial=B][,land][,latency=MS][;...]" */
std::unique_ptr<FaultTable> parse_fault_spec(const std::string& spec)
{
    auto table = std::make_unique<FaultTable>();
    for (const auto& rule_text : split(spec, ';')) {
        if (rule_text.empty()) continue;
        const auto fields = split(rule_text, ',');
        if (fields.size() < 2 || fields[0].empty())
            throw std::invalid_argument(
                "fault rule '" + rule_text + "' needs at least 'site,ERRNO'");
        FaultRule& r = table->rules.emplace_back();
        r.site = fields[0];
        r.err  = parse_errno(fields[1]);
        for (std::size_t i = 2; i < fields.size(); ++i) {
            const auto& f = fields[i];
            const auto eq = f.find('=');
            const std::string key = f.substr(0, eq);
            const std::string val = eq == std::string::npos ? "" : f.substr(eq + 1);
            if (key == "skip") {
                r.skip = parse_ulong(key, val);
            } else if (key == "count") {
                if (val == "always") r.always = true;
                else                 r.count  = parse_ulong(key, val);
            } else if (key == "always" && eq == std::string::npos) {
                r.always = true;
            } else if (key == "partial") {
                r.partial     = parse_ulong(key, val);
                r.has_partial = true;
            } else if (key == "land" && eq == std::string::npos) {
                r.land = true;
            } else if (key == "latency") {
                r.latency_ms = parse_ulong(key, val);
            } else {
                throw std::invalid_argument(
                    "unknown field '" + f + "' in fault rule '" + rule_text + "'");
            }
        }
    }
    return table;
}

/* The environment table is installed by a namespace-scope initializer, i.e.
 * once, before main(), and never freed: an I/O call during static
 * destruction must not see a dangling table. g_fault_table is constant-
 * initialised, so any I/O before this runs simply sees "no faults". */
bool install_env_faults() noexcept
{
    const char* spec = std::getenv("TADAH_HPO_TEST_IO_FAULT");
    if (!spec || !*spec) return false;
    try {
        auto table = parse_fault_spec(spec);
        detail::g_fault_table.store(table.release(), std::memory_order_release);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HPO] WARNING: ignoring malformed TADAH_HPO_TEST_IO_FAULT='"
                  << spec << "': " << e.what() << "\n";
        return false;
    }
}
[[maybe_unused]] const bool g_env_faults_installed = install_env_faults();

struct BackoffOverride {
    bool   set = false;
    long   initial = 0, cap = 0, budget = 0;
    double jitter = -1.0;   // < 0: keep the policy's jitter
};

const BackoffOverride& backoff_override()
{
    static const BackoffOverride o = [] {
        BackoffOverride out;
        const char* v = std::getenv("TADAH_HPO_TEST_IO_BACKOFF_MS");
        if (!v || !*v) return out;
        const auto f = split(v, ',');
        try {
            if (f.size() < 3 || f.size() > 4)
                throw std::invalid_argument("expected initial,cap,budget[,jitter]");
            out.initial = static_cast<long>(parse_ulong("initial", f[0]));
            out.cap     = static_cast<long>(parse_ulong("cap", f[1]));
            out.budget  = static_cast<long>(parse_ulong("budget", f[2]));
            if (f.size() == 4) {
                char* end = nullptr;
                out.jitter = std::strtod(f[3].c_str(), &end);
                if (f[3].empty() || *end != '\0' ||
                    !(out.jitter >= 0.0 && out.jitter < 1.0))
                    throw std::invalid_argument("jitter must be in [0,1)");
            }
            /* next_delay() floors every delay at 1 ms anyway; keep the
             * override's own schedule meaningful too. */
            out.initial = std::max(out.initial, 1L);
            out.cap     = std::max(out.cap, out.initial);
            out.set     = true;
        } catch (const std::exception& e) {
            std::cerr << "[HPO] WARNING: ignoring malformed "
                         "TADAH_HPO_TEST_IO_BACKOFF_MS='" << v << "': "
                      << e.what() << "\n";
            out = BackoffOverride{};
        }
        return out;
    }();
    return o;
}

BackoffPolicy with_override(BackoffPolicy p)
{
    const auto& o = backoff_override();
    if (!o.set) return p;
    p.initial = std::chrono::milliseconds(o.initial);
    p.cap     = std::chrono::milliseconds(o.cap);
    p.budget  = std::chrono::milliseconds(o.budget);
    if (o.jitter >= 0.0) p.jitter = o.jitter;
    return p;
}

} // namespace

/* ---------------------------------------------------------------------------
 * Backoff
 * ------------------------------------------------------------------------- */
BackoffPolicy BackoffPolicy::durable()
{
    BackoffPolicy p;
    p.budget = std::chrono::milliseconds(30000);
    return with_override(p);
}

BackoffPolicy BackoffPolicy::writer()
{
    BackoffPolicy p;
    p.budget = std::chrono::milliseconds(10000);
    return with_override(p);
}

std::chrono::milliseconds BackoffPolicy::nominal_delay(unsigned retry) const noexcept
{
    const double cap_ms = static_cast<double>(cap.count());
    double d = static_cast<double>(initial.count());
    /* Only a growing, positive delay can reach the cap: a factor <= 1 or a
     * zero initial delay would otherwise loop retry-1 times per call. */
    if (factor > 1.0)
        for (unsigned i = 1; i < retry && d > 0.0 && d < cap_ms; ++i) d *= factor;
    d = std::min(d, cap_ms);
    return std::chrono::milliseconds(static_cast<long long>(std::max(d, 0.0)));
}

std::chrono::milliseconds BackoffPolicy::next_delay(unsigned retry) const
{
    /* At least 1 ms: every retry spends budget, so any budget bounds the
     * number of retries whatever initial/cap/factor/jitter say. */
    constexpr std::chrono::milliseconds kMin{1};
    const auto nominal = nominal_delay(retry);
    if (!(jitter > 0.0) || nominal.count() == 0) return std::max(nominal, kMin);
    /* Failure path only. Seeded per thread so concurrent writers that failed
     * together do not retry in lockstep. */
    thread_local std::minstd_rand rng(static_cast<std::uint_fast32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count() ^
        std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::uniform_real_distribution<double> u(-jitter, jitter);
    const double d = static_cast<double>(nominal.count()) * (1.0 + u(rng));
    return std::max(std::chrono::milliseconds(static_cast<long long>(std::llround(d))),
                    kMin);
}

bool sleep_for(const SleepFn& sleep, std::chrono::milliseconds d,
               AbortFn should_abort)
{
    if (sleep) {
        sleep(d);
        return !(should_abort && should_abort());
    }
    if (!should_abort) {
        if (d.count() > 0) std::this_thread::sleep_for(d);
        return true;
    }
    /* Slices of <= 100 ms: a stop request ends the wait within one slice. */
    constexpr std::chrono::milliseconds kSlice{100};
    while (d.count() > 0) {
        if (should_abort()) return false;
        const auto s = std::min(d, kSlice);
        std::this_thread::sleep_for(s);
        d -= s;
    }
    return !should_abort();
}

namespace {

/* testing::ScopedNetworkFs: -1 = ask statfs, 0 = local, 1 = network. */
std::atomic<int> g_network_fs_override{-1};

/* The directory holding `path` is on a network (or FUSE) filesystem. One
 * statfs(2); called only after an EACCES. A directory that cannot be
 * queried counts as network: EACCES then keeps its retry. */
bool on_network_filesystem(const std::string& path) noexcept
{
    const int forced = g_network_fs_override.load(std::memory_order_relaxed);
    if (forced >= 0) return forced != 0;
#if !defined(__linux__)
    (void)path;
    return true;   // no filesystem-type query here: keep the retry
#else
    try {
        const auto slash = path.find_last_of('/');
        const std::string dir = slash == std::string::npos ? std::string(".")
                              : slash == 0                 ? std::string("/")
                                                           : path.substr(0, slash);
        struct statfs st {};
        int rc;
        do { rc = ::statfs(dir.c_str(), &st); } while (rc < 0 && errno == EINTR);
        if (rc != 0) return true;
        switch (static_cast<unsigned long>(st.f_type) & 0xFFFFFFFFul) {
        case 0xFF534D42ul:   // CIFS_MAGIC_NUMBER
        case 0xFE534D42ul:   // SMB2_MAGIC_NUMBER
        case 0x0000517Bul:   // SMB_SUPER_MAGIC
        case 0x00006969ul:   // NFS_SUPER_MAGIC
        case 0x5346414Ful:   // AFS_SUPER_MAGIC
        case 0x6B414653ul:   // AFS_FS_MAGIC (kAFS)
        case 0x73757245ul:   // CODA_SUPER_MAGIC
        case 0x00C36400ul:   // CEPH_SUPER_MAGIC
        case 0x0BD00BD0ul:   // LUSTRE_SUPER_MAGIC
        case 0x47504653ul:   // GPFS_SUPER_MAGIC
        case 0x01021997ul:   // V9FS_MAGIC (9P)
        case 0x65735546ul:   // FUSE_SUPER_MAGIC (sshfs, rclone, ...)
            return true;
        default:
            return false;
        }
    } catch (...) {
        return true;
    }
#endif
}

} // namespace

bool is_transient_errno(int err, const std::string& path) noexcept
{
    if (!is_transient_errno(err)) return false;
    return err != EACCES || on_network_filesystem(path);
}

bool is_transient_errno(int err) noexcept
{
    switch (err) {
    case EBUSY:
    case EAGAIN:
#if EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK:
#endif
    case EACCES:
    case ESTALE:
    case ETXTBSY:
    case EIO:
    case ENOLCK:
    case ETIMEDOUT:
    case EDEADLK:
        return true;
    default:
        return false;
    }
}

std::string errno_name(int err)
{
    for (const auto& e : kErrnoNames)
        if (e.value == err) return e.name;
    return std::to_string(err);
}

std::string gave_up_suffix(unsigned attempts,
                           std::chrono::steady_clock::duration elapsed,
                           GiveUp why)
{
    return gave_up_suffix(attempts, std::chrono::duration<double>(elapsed).count(),
                          why);
}

std::string gave_up_suffix(unsigned attempts, double s, GiveUp why)
{
    char buf[128];
    std::snprintf(buf, sizeof buf,
                  why == GiveUp::abandoned
                      ? " (retry abandoned: shutdown requested) after %u attempt%s over %.3g s"
                      : " (gave up after %u attempt%s over %.3g s)",
                  attempts, attempts == 1 ? "" : "s", s);
    return buf;
}

/* ---------------------------------------------------------------------------
 * ScopedFaults
 * ------------------------------------------------------------------------- */
namespace testing {

ScopedFaults::ScopedFaults(const std::string& spec)
    : table_(parse_fault_spec(spec)),
      prev_(detail::g_fault_table.exchange(nullptr, std::memory_order_acq_rel))
{
    detail::g_fault_table.store(table_.get(), std::memory_order_release);
}

ScopedFaults::~ScopedFaults()
{
    detail::g_fault_table.store(prev_, std::memory_order_release);
}

unsigned long ScopedFaults::calls(const std::string& site) const
{
    for (const auto& r : table_->rules)
        if (r.site == site) return r.calls.load(std::memory_order_relaxed);
    return 0;
}

unsigned long ScopedFaults::injected(const std::string& site) const
{
    unsigned long n = 0;
    for (const auto& r : table_->rules)
        if (r.site == site) n += r.injected.load(std::memory_order_relaxed);
    return n;
}

ScopedNetworkFs::ScopedNetworkFs(bool network)
    : prev_(g_network_fs_override.exchange(network ? 1 : 0,
                                           std::memory_order_relaxed))
{}

ScopedNetworkFs::~ScopedNetworkFs()
{
    g_network_fs_override.store(prev_, std::memory_order_relaxed);
}

} // namespace testing

} // namespace tadah::hpo::io

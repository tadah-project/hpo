#include "catch2/catch.hpp"

#include <tadah/hpo/append_log.h>
#include <tadah/hpo/io_retry.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <atomic>
#include <sys/stat.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>

/* Transient I/O retry for the per-evaluation logs and archives
 * (RetryingAppendFile / write_whole_file) and the shared io_retry core
 * (backoff schedule, errno classification, fault-injection grammar). */

namespace {

namespace fs = std::filesystem;
namespace io = tadah::hpo::io;
using io::testing::ScopedFaults;

std::string slurp(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void spit(const fs::path& path, const std::string& bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
}

struct Scratch {
    fs::path path;
    Scratch()
        : path(fs::temp_directory_path() /
               ("tadah_append_log_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this))))
    {
        fs::create_directories(path);
    }
    ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
};

io::BackoffPolicy exact_policy(long budget_ms)
{
    io::BackoffPolicy p;
    p.initial = std::chrono::milliseconds(10);
    p.cap     = std::chrono::milliseconds(2000);
    p.budget  = std::chrono::milliseconds(budget_ms);
    p.jitter  = 0.0;
    return p;
}

struct Recorder {
    std::vector<long>           sleeps;
    std::vector<io::RetryEvent> events;
    io::SleepFn sleep()
    {
        return [this](std::chrono::milliseconds d) { sleeps.push_back(d.count()); };
    }
    io::OnRetryFn on_retry()
    {
        return [this](const io::RetryEvent& ev) { events.push_back(ev); };
    }
    long slept() const
    {
        long s = 0;
        for (long v : sleeps) s += v;
        return s;
    }
};

/* A stop flag for BackoffPolicy::should_abort (a plain function pointer). */
std::atomic<bool> g_stop{false};
bool stop_flag() { return g_stop.load(); }
struct StopReset {
    StopReset()  { g_stop = false; }
    ~StopReset() { g_stop = false; }
};

/* Stage `rows` numbered rows the way the HPO writer does. */
std::string stage_rows(io::RetryingAppendFile& f, int first, int rows)
{
    std::ostringstream expect;
    for (int i = first; i < first + rows; ++i) {
        f.stage() << std::left << std::setw(12) << i << std::scientific
                  << std::setprecision(6) << 0.5 * i << std::endl;
        expect << std::left << std::setw(12) << i << std::scientific
               << std::setprecision(6) << 0.5 * i << "\n";
    }
    return expect.str();
}

} // namespace

/* ---------------------------------------------------------------------------
 * io_retry core
 * ------------------------------------------------------------------------- */
TEST_CASE("io retry classifies transient and permanent errnos", "[io_retry]")
{
    for (int e : {EBUSY, EAGAIN, EWOULDBLOCK, EACCES, ESTALE, ETXTBSY, EIO,
                  ENOLCK, ETIMEDOUT, EDEADLK})
        CHECK(io::is_transient_errno(e));
    for (int e : {0, ENOSPC, EDQUOT, EROFS, ENOENT, ENOTDIR, EISDIR, EPERM,
                  EINVAL, EMFILE, EBADF, EEXIST})
        CHECK_FALSE(io::is_transient_errno(e));
    CHECK(io::errno_name(EBUSY) == "EBUSY");
    CHECK(io::errno_name(ESTALE) == "ESTALE");
    CHECK(io::errno_name(4000) == "4000");
}

TEST_CASE("io retry backoff doubles to the cap and jitters within bounds",
          "[io_retry]")
{
    io::BackoffPolicy p = exact_policy(30000);
    const long expect[] = {10, 20, 40, 80, 160, 320, 640, 1280, 2000, 2000, 2000};
    for (unsigned i = 0; i < std::size(expect); ++i) {
        CHECK(p.nominal_delay(i + 1).count() == expect[i]);
        CHECK(p.next_delay(i + 1).count() == expect[i]);   // jitter 0
    }
    CHECK(p.nominal_delay(1000).count() == 2000);

    p.jitter = 0.2;
    for (int i = 0; i < 1000; ++i) {
        const unsigned r = 1 + static_cast<unsigned>(i % 12);
        const double nominal = static_cast<double>(p.nominal_delay(r).count());
        const auto d = static_cast<double>(p.next_delay(r).count());
        CHECK(d >= std::floor(0.8 * nominal));
        CHECK(d <= std::ceil(1.2 * nominal));
    }

    CHECK(io::BackoffPolicy::durable().budget.count() == 30000);
    CHECK(io::BackoffPolicy::writer().budget.count() == 10000);
    CHECK(io::BackoffPolicy::writer().initial.count() == 10);
    CHECK(io::BackoffPolicy::writer().cap.count() == 2000);
    CHECK(io::BackoffPolicy::writer().jitter == Approx(0.2));
}

TEST_CASE("io retry calls nothing on success and stops on non-transient errors",
          "[io_retry]")
{
    Recorder rec;
    const auto policy = exact_policy(1000);
    auto classify = [](const std::exception& e, io::RetryEvent& ev) {
        const auto* se = dynamic_cast<const std::system_error*>(&e);
        if (!se || !io::is_transient_errno(se->code().value())) return false;
        ev.op = "test.op";
        ev.err = se->code().value();
        return true;
    };
    auto on_retry = [&](const io::RetryEvent& ev) { rec.events.push_back(ev); };

    int calls = 0;
    CHECK(io::retry(policy, [&] { ++calls; return 42; }, classify, on_retry,
                    rec.sleep()) == 42);
    CHECK(calls == 1);
    CHECK(rec.sleeps.empty());
    CHECK(rec.events.empty());

    calls = 0;
    CHECK(io::retry(policy, [&] {
              if (++calls < 3)
                  throw std::system_error(EIO, std::generic_category());
              return 7;
          }, classify, on_retry, rec.sleep()) == 7);
    CHECK(calls == 3);
    CHECK(rec.sleeps == std::vector<long>{10, 20});
    REQUIRE(rec.events.size() == 2);
    CHECK(rec.events[1].attempt == 2);
    CHECK(rec.events[1].err == EIO);

    calls = 0;
    CHECK_THROWS_AS(io::retry(policy, [&]() -> int {
                        ++calls;
                        throw std::system_error(ENOSPC, std::generic_category());
                    }, classify, on_retry, rec.sleep()),
                    std::system_error);
    CHECK(calls == 1);

    calls = 0;
    CHECK_THROWS_AS(io::retry(policy, [&]() -> int { ++calls; throw 5; },
                              classify, on_retry, rec.sleep()),
                    int);
    CHECK(calls == 1);

    /* Exhaustion: the hook may replace the exception. */
    calls = 0;
    try {
        io::retry(exact_policy(35), [&]() -> int {
                      ++calls;
                      throw std::system_error(EBUSY, std::generic_category());
                  }, classify, on_retry, rec.sleep(),
                  [](const std::exception&, unsigned attempts,
                     std::chrono::steady_clock::duration elapsed, io::GiveUp why) {
                      CHECK(why == io::GiveUp::budget);
                      throw std::runtime_error(
                          "replaced" + io::gave_up_suffix(attempts, elapsed, why));
                  });
        FAIL("retry did not throw");
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).rfind("replaced (gave up after 3 attempts over ", 0) == 0);
    }
    CHECK(calls == 3);   // sleeps 10 + 20; 40 more would exceed 35
}

TEST_CASE("io fault rules skip, count, repeat and nest", "[io_retry]")
{
    REQUIRE(io::consume_fault("t.site") == 0);
    {
        ScopedFaults f("t.site,EBUSY,skip=2,count=2; t.other,5 ;t.site,EIO,skip=5,always");
        std::vector<int> seen;
        for (int i = 0; i < 8; ++i) seen.push_back(io::consume_fault("t.site"));
        CHECK(seen == std::vector<int>{0, 0, EBUSY, EBUSY, 0, EIO, EIO, EIO});
        CHECK(io::consume_fault("t.other") == 5);
        CHECK(io::consume_fault("t.other") == 0);
        CHECK(io::consume_fault("t.none") == 0);
        CHECK(f.calls("t.site") == 8);
        CHECK(f.injected("t.site") == 5);
        CHECK(f.injected("t.none") == 0);
        {
            ScopedFaults inner("t.site,EWOULDBLOCK,partial=7");
            std::size_t partial = 0;
            CHECK(io::consume_fault("t.site", &partial) == EWOULDBLOCK);
            CHECK(partial == 7);
            CHECK(io::consume_fault("t.site") == 0);
        }
        CHECK(io::consume_fault("t.site") == EIO);   // outer table restored
        ScopedFaults masked("");
        CHECK(io::consume_fault("t.site") == 0);
    }
    CHECK(io::consume_fault("t.site") == 0);

    for (const char* bad : {"t.site", ",EIO", "t.site,EFOO", "t.site,EIO,skip=x",
                            "t.site,EIO,count=-1", "t.site,EIO,bogus=1", "t.site,0"})
        CHECK_THROWS_AS(ScopedFaults(bad), std::invalid_argument);
}

/* The environment knobs are read once per process, so they are checked in a
 * re-executed copy of this binary running the hidden child case below. */
TEST_CASE("io fault and backoff environment knobs are read at startup",
          "[io_retry]")
{
    Scratch scratch;
    const auto out = scratch.path / "child.out";
    std::cout.flush();
    std::fflush(nullptr);   // or the child re-emits our buffered output
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        ::setenv("TADAH_HPO_TEST_IO_FAULT",
                 "durable.lock_open,EBUSY,skip=1,count=2;writer.append,EIO,always", 1);
        ::setenv("TADAH_HPO_TEST_IO_BACKOFF_MS", "1,4,50,0", 1);
        std::FILE* f = std::freopen(out.c_str(), "w", stdout);
        (void)f;
        ::dup2(::fileno(stdout), 2);
        ::execl("/proc/self/exe", "test_hpo", "[.io_retry_env_child]", (char*)nullptr);
        ::_exit(127);
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    INFO(slurp(out));
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    REQUIRE(slurp(out).find("All tests passed") != std::string::npos);
}

TEST_CASE("io retry environment child", "[.io_retry_env_child]")
{
    REQUIRE(io::consume_fault("durable.lock_open") == 0);
    REQUIRE(io::consume_fault("durable.lock_open") == EBUSY);
    REQUIRE(io::consume_fault("durable.lock_open") == EBUSY);
    REQUIRE(io::consume_fault("durable.lock_open") == 0);
    REQUIRE(io::consume_fault("writer.append") == EIO);
    REQUIRE(io::consume_fault("writer.append") == EIO);
    for (const auto& p : {io::BackoffPolicy::durable(), io::BackoffPolicy::writer()}) {
        CHECK(p.initial.count() == 1);
        CHECK(p.cap.count() == 4);
        CHECK(p.budget.count() == 50);
        CHECK(p.jitter == 0.0);
    }
}

/* ---------------------------------------------------------------------------
 * RetryingAppendFile
 * ------------------------------------------------------------------------- */
TEST_CASE("append log concatenates clean batches exactly with one write each",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "loss.tadah";
    spit(path, "#ITERATION  HEADER\n");
    std::string expect = "#ITERATION  HEADER\n";

    Recorder rec;
    io::RetryingAppendFile f(path.string());
    CHECK(f.offset() == expect.size());
    /* One write per batch: an injected failure on the 2nd write of the run
     * would fire in batch 2 only if batch 1 used exactly one write. */
    ScopedFaults faults("writer.append,EIO,skip=1000000;writer.open,EIO,skip=1000000");
    for (int b = 0; b < 3; ++b) {
        expect += stage_rows(f, 1 + 10 * b, 10);
        CHECK(f.pending_bytes() > 0);
        const auto c = f.commit(exact_policy(1000), rec.sleep(), rec.on_retry());
        CHECK(c.ok);
        CHECK(c.attempts == 1);
        CHECK(c.retries == 0);
        CHECK(f.pending_bytes() == 0);
        CHECK(faults.calls("writer.append") == static_cast<unsigned long>(b + 1));
    }
    /* Empty batch: no write at all. */
    const auto c = f.commit(exact_policy(1000), rec.sleep(), rec.on_retry());
    CHECK(c.ok);
    CHECK(c.attempts == 0);
    CHECK(faults.calls("writer.append") == 3);
    CHECK(faults.calls("writer.open") == 0);
    CHECK(rec.sleeps.empty());
    CHECK(rec.events.empty());
    CHECK(slurp(path) == expect);
    CHECK(f.offset() == expect.size());
}

TEST_CASE("append log rewrites a batch whose write failed part-way",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "loss.tadah";
    spit(path, "HDR\n");
    io::RetryingAppendFile f(path.string());
    std::string expect = "HDR\n" + stage_rows(f, 1, 3);
    REQUIRE(f.commit(exact_policy(1000)).ok);

    /* 7 real bytes land, then EIO: the retry truncates them away and writes
     * the whole batch once. */
    const std::string batch = stage_rows(f, 4, 5);
    expect += batch;
    ScopedFaults faults("writer.append,EIO,count=1,partial=7");
    Recorder rec;
    const auto c = f.commit(exact_policy(1000), rec.sleep(), rec.on_retry());
    CHECK(c.ok);
    CHECK(c.attempts == 2);
    CHECK(c.retries == 1);
    CHECK(rec.sleeps == std::vector<long>{10});
    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].op == "writer.append");
    CHECK(rec.events[0].err == EIO);
    CHECK(rec.events[0].path == path.string());
    CHECK(slurp(path) == expect);
    CHECK(f.offset() == expect.size());
    CHECK_FALSE(f.degraded());

    /* The same with a failing truncate on the first recovery attempt. */
    const std::string batch2 = stage_rows(f, 9, 2);
    expect += batch2;
    ScopedFaults faults2("writer.append,ESTALE,count=1,partial=3;"
                         "writer.truncate,EIO,count=1");
    Recorder rec2;
    const auto c2 = f.commit(exact_policy(1000), rec2.sleep(), rec2.on_retry());
    CHECK(c2.ok);
    CHECK(c2.retries == 2);
    REQUIRE(rec2.events.size() == 2);
    CHECK(rec2.events[1].op == "writer.truncate");
    CHECK(slurp(path) == expect);
}

TEST_CASE("append log drops a batch after its budget, truncates, then recovers",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "outvar.tadah";
    spit(path, "HDR\n");
    io::RetryingAppendFile f(path.string());
    std::string expect = "HDR\n" + stage_rows(f, 1, 2);
    REQUIRE(f.commit(exact_policy(1000)).ok);

    {
        ScopedFaults faults("writer.append,EIO,always,partial=5");
        (void)stage_rows(f, 3, 4);
        Recorder rec;
        const auto c = f.commit(exact_policy(100), rec.sleep(), rec.on_retry());
        CHECK_FALSE(c.ok);
        CHECK(c.rows_lost == 4);
        CHECK(c.err == EIO);
        CHECK(c.op == "writer.append");
        CHECK(c.attempts == 4);                        // sleeps 10+20+40; 80 > budget
        CHECK_FALSE(c.degraded);
        CHECK(rec.sleeps == std::vector<long>{10, 20, 40});
        CHECK(rec.slept() <= 100);
        CHECK(f.degraded());
        CHECK(f.pending_bytes() == 0);
        CHECK(slurp(path) == expect);                  // partial bytes truncated away

        /* Degraded: one attempt, no sleep, no retry event. */
        (void)stage_rows(f, 7, 1);
        Recorder rec2;
        const auto c2 = f.commit(exact_policy(100), rec2.sleep(), rec2.on_retry());
        CHECK_FALSE(c2.ok);
        CHECK(c2.rows_lost == 1);
        CHECK(c2.attempts == 1);
        CHECK(c2.retries == 0);
        CHECK(c2.degraded);
        CHECK(rec2.sleeps.empty());
        CHECK(rec2.events.empty());
        CHECK(slurp(path) == expect);
    }

    /* The mount is back: the next batch appends right after the last good
     * byte and clears degraded mode. */
    expect += stage_rows(f, 8, 3);
    Recorder rec3;
    const auto c3 = f.commit(exact_policy(100), rec3.sleep(), rec3.on_retry());
    CHECK(c3.ok);
    CHECK_FALSE(f.degraded());
    CHECK(rec3.sleeps.empty());
    CHECK(slurp(path) == expect);

    /* Permanent errno: one immediate retry, no sleep. */
    {
        ScopedFaults faults("writer.append,ENOSPC,count=1");
        expect += stage_rows(f, 11, 2);
        Recorder rec4;
        const auto c4 = f.commit(exact_policy(100), rec4.sleep(), rec4.on_retry());
        CHECK(c4.ok);
        CHECK(c4.retries == 1);
        CHECK(rec4.sleeps.empty());
        REQUIRE(rec4.events.size() == 1);
        CHECK(rec4.events[0].next_delay.count() == 0);
        CHECK(slurp(path) == expect);
    }
    {
        ScopedFaults faults("writer.append,ENOSPC,always");
        (void)stage_rows(f, 13, 2);
        Recorder rec5;
        const auto c5 = f.commit(exact_policy(100), rec5.sleep(), rec5.on_retry());
        CHECK_FALSE(c5.ok);
        CHECK(c5.attempts == 2);
        CHECK(c5.rows_lost == 2);
        CHECK(rec5.sleeps.empty());
        CHECK(slurp(path) == expect);
    }
}

TEST_CASE("append log follows a file truncated or replaced underneath it",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "params.tadah";
    spit(path, "HDR\n");
    io::RetryingAppendFile f(path.string());
    (void)stage_rows(f, 1, 3);
    REQUIRE(f.commit(exact_policy(1000)).ok);

    /* A new header replaces the file; the failed write's recovery must not
     * truncate to the stale (larger) offset but append after what exists. */
    spit(path, "NEWHDR\n");
    const std::string batch = stage_rows(f, 4, 2);
    ScopedFaults faults("writer.append,EIO,count=1");
    const auto c = f.commit(exact_policy(1000), [](std::chrono::milliseconds) {});
    CHECK(c.ok);
    CHECK(slurp(path) == "NEWHDR\n" + batch);
    CHECK(f.offset() == 7 + batch.size());
}

TEST_CASE("append log opens late without losing the header",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "stats.tadah";
    spit(path, "HEADER LINE\n");
    std::unique_ptr<io::RetryingAppendFile> f;
    {
        ScopedFaults faults("writer.open,EACCES,count=1");
        f = std::make_unique<io::RetryingAppendFile>(path.string());
    }
    const std::string batch = stage_rows(*f, 1, 2);
    Recorder rec;
    const auto c = f->commit(exact_policy(1000), rec.sleep(), rec.on_retry());
    CHECK(c.ok);
    CHECK(c.retries == 0);
    CHECK(slurp(path) == "HEADER LINE\n" + batch);

    /* A directory that does not exist is permanent: dropped, no sleep. */
    io::RetryingAppendFile missing((scratch.path / "no_such_dir" / "x.tadah").string());
    (void)stage_rows(missing, 1, 3);
    Recorder rec2;
    const auto c2 = missing.commit(exact_policy(1000), rec2.sleep(), rec2.on_retry());
    CHECK_FALSE(c2.ok);
    CHECK(c2.err == ENOENT);
    CHECK(c2.op == "writer.open");
    CHECK(c2.rows_lost == 3);
    CHECK(rec2.sleeps.empty());
}

TEST_CASE("archive writer retries and never leaves a partial file",
          "[append_log][io_retry]")
{
    Scratch scratch;
    const auto path = scratch.path / "pot_12.tadah";
    const std::string bytes(20000, 'x');

    auto r = io::write_whole_file(path.string(), bytes, exact_policy(1000));
    CHECK(r.ok);
    CHECK(r.attempts == 1);
    CHECK(slurp(path) == bytes);

    {
        ScopedFaults faults("writer.archive,EIO,count=1,partial=3");
        Recorder rec;
        const std::string other(5000, 'y');
        r = io::write_whole_file(path.string(), other, exact_policy(1000),
                                 rec.sleep(), rec.on_retry());
        CHECK(r.ok);
        CHECK(r.retries == 1);
        CHECK(rec.sleeps == std::vector<long>{10});
        REQUIRE(rec.events.size() == 1);
        CHECK(rec.events[0].op == "writer.archive");
        CHECK(slurp(path) == other);                   // O_TRUNC: no stale tail
    }
    {
        ScopedFaults faults("writer.archive,EIO,always,partial=3");
        Recorder rec;
        r = io::write_whole_file(path.string(), bytes, exact_policy(50),
                                 rec.sleep(), rec.on_retry());
        CHECK_FALSE(r.ok);
        CHECK(r.err == EIO);
        CHECK(rec.slept() <= 50);
        CHECK_FALSE(fs::exists(path));                 // partial file unlinked
    }
    Recorder rec;
    r = io::write_whole_file((scratch.path / "missing" / "a.tadah").string(), bytes,
                             exact_policy(1000), rec.sleep(), rec.on_retry());
    CHECK_FALSE(r.ok);
    CHECK(r.err == ENOENT);
    CHECK(r.attempts == 2);
    CHECK(rec.sleeps.empty());
}

/* ---------------------------------------------------------------------------
 * Retry edge cases: backoff schedule, slow failures, shutdown grace,
 * permission errors, archives, external truncation
 * ------------------------------------------------------------------------- */
namespace {
auto transient_classify = [](const std::exception& e, io::RetryEvent& ev) {
    const auto* se = dynamic_cast<const std::system_error*>(&e);
    if (!se || !io::is_transient_errno(se->code().value())) return false;
    ev.op  = "test.op";
    ev.err = se->code().value();
    return true;
};
} // namespace

TEST_CASE("io retry: every delay spends at least 1 ms, so any budget bounds it",
          "[io_retry]")
{
    /* Initial/cap/factor at or near 0 used to give 0 ms delays
     * that never spent the budget (an endless spin). */
    const std::vector<std::tuple<long, long, double>> shapes = {
        {0, 2000, 2.0}, {10, 0, 2.0}, {10, 2000, 0.0}, {10, 2000, 0.5}, {0, 0, 1.0}};
    for (const auto& [initial, cap, factor] : shapes) {
        io::BackoffPolicy p = exact_policy(100);
        p.initial = std::chrono::milliseconds(initial);
        p.cap     = std::chrono::milliseconds(cap);
        p.factor  = factor;
        INFO("initial " << initial << " cap " << cap << " factor " << factor);
        for (unsigned r : {1u, 2u, 50u, 100000u})
            CHECK(p.next_delay(r).count() >= 1);
        Recorder rec;
        int calls = 0;
        CHECK_THROWS_AS(io::retry(p, [&]() -> int {
                            ++calls;
                            throw std::system_error(EBUSY, std::generic_category());
                        }, transient_classify, [](const io::RetryEvent&) {},
                        rec.sleep()),
                        std::system_error);
        CHECK(calls <= 101);
        CHECK(rec.slept() <= 100);
    }

    /* Budget 0 never retries a transient error, whatever the jitter rounds
     * the delay to. */
    io::BackoffPolicy zero = exact_policy(0);
    zero.initial = std::chrono::milliseconds(1);
    zero.jitter  = 0.99;
    for (int i = 0; i < 200; ++i) {
        int calls = 0;
        CHECK_THROWS_AS(io::retry(zero, [&]() -> int {
                            ++calls;
                            throw std::system_error(EBUSY, std::generic_category());
                        }, transient_classify, [](const io::RetryEvent&) {},
                        [](std::chrono::milliseconds) {}),
                        std::system_error);
        REQUIRE(calls == 1);
    }
    Scratch scratch;
    io::RetryingAppendFile f((scratch.path / "l.tadah").string());
    ScopedFaults faults("writer.append,EIO,always");
    (void)stage_rows(f, 1, 2);
    Recorder rec;
    const auto c = f.commit(zero, rec.sleep(), rec.on_retry());
    CHECK_FALSE(c.ok);
    CHECK(c.attempts == 1);
    CHECK(rec.sleeps.empty());
}

TEST_CASE("io retry: the budget counts the time failing attempts take",
          "[io_retry]")
{
    /* Only the sleeps used to count, so slow failures (a soft
     * mount's timeout) multiplied the budget. Each attempt below takes
     * 20 ms, sleeps are no-ops of 1 ms nominal: a 50 ms budget allows ~3
     * attempts, not ~50. */
    io::BackoffPolicy p = exact_policy(50);
    p.initial = p.cap = std::chrono::milliseconds(1);
    int calls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_THROWS_AS(io::retry(p, [&]() -> int {
                        ++calls;
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        throw std::system_error(EIO, std::generic_category());
                    }, transient_classify, [](const io::RetryEvent&) {},
                    [](std::chrono::milliseconds) {}),
                    std::system_error);
    CHECK(calls >= 2);
    CHECK(calls <= 4);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(400));

    /* The log writer's loop, via the latency= fault field. */
    Scratch scratch;
    io::RetryingAppendFile f((scratch.path / "l.tadah").string());
    ScopedFaults faults("writer.append,EIO,always,latency=20");
    (void)stage_rows(f, 1, 2);
    const auto c = f.commit(p, [](std::chrono::milliseconds) {});
    CHECK_FALSE(c.ok);
    CHECK(c.attempts >= 2);
    CHECK(c.attempts <= 4);
    CHECK(c.seconds < 0.4);
}

TEST_CASE("io retry: after a stop request, retrying goes on for the grace only",
          "[io_retry]")
{
    /* A shutdown signal must neither hold the exit for the whole backoff
     * nor end retrying at once (one blip after the signal would lose the
     * in-flight best / the final drain's rows): the loop goes on for at
     * most stop_grace (2 s) from when it first sees the stop. */
    StopReset reset;
    io::BackoffPolicy p = exact_policy(30000);
    p.should_abort = &stop_flag;
    auto ebusy = [](int& calls) {
        return [&calls]() -> int {
            ++calls;
            throw std::system_error(EBUSY, std::generic_category());
        };
    };

    /* A stop already requested: one transient fault is still retried. */
    {
        g_stop = true;
        Recorder rec;
        int calls = 0;
        const int v = io::retry(p, [&]() -> int {
                                    if (++calls == 1)
                                        throw std::system_error(EIO, std::generic_category());
                                    return 7;
                                }, transient_classify, rec.on_retry(), rec.sleep());
        CHECK(v == 7);
        CHECK(calls == 2);
        CHECK(rec.sleeps == std::vector<long>{10});
        g_stop = false;
    }

    /* The injected sleep raises the stop during the 2nd backoff: the loop
     * retries at once, then backs off within the 2 s grace (delays
     * shortened to fit), then gives up and the exhaustion hook learns
     * why. */
    {
        Recorder rec;
        int calls = 0;
        io::GiveUp seen = io::GiveUp::budget;
        try {
            io::retry(p, ebusy(calls), transient_classify, rec.on_retry(),
                      [&](std::chrono::milliseconds d) {
                          rec.sleeps.push_back(d.count());
                          if (rec.sleeps.size() == 2) g_stop = true;
                      },
                      [&](const std::exception&, unsigned attempts,
                          std::chrono::steady_clock::duration elapsed, io::GiveUp why) {
                          seen = why;
                          throw std::runtime_error(io::gave_up_suffix(attempts, elapsed, why));
                      });
            FAIL("retry did not throw");
        } catch (const std::runtime_error& e) {
            CHECK(std::string(e.what()).rfind(" (retry abandoned: shutdown requested) "
                                              "after 9 attempts over ", 0) == 0);
        }
        CHECK(seen == io::GiveUp::abandoned);
        CHECK(calls == 9);
        CHECK(rec.sleeps == std::vector<long>{10, 20, 40, 80, 160, 320, 640, 760});
        g_stop = false;
    }

    /* stop_grace 0 restores "abandon at once": not even one retry. */
    {
        g_stop = true;
        io::BackoffPolicy none = p;
        none.stop_grace = std::chrono::milliseconds(0);
        Recorder rec;
        int calls = 0;
        CHECK_THROWS_AS(io::retry(none, ebusy(calls), transient_classify,
                                  [](const io::RetryEvent&) {}, rec.sleep()),
                        std::system_error);
        CHECK(calls == 1);
        CHECK(rec.sleeps.empty());
        g_stop = false;
    }

    /* Real sleeps: a stop 150 ms into a 2 s backoff cuts that sleep short;
     * the rest takes at most the grace (300 ms here) plus a slice. */
    io::BackoffPolicy slow = exact_policy(30000);
    slow.initial = slow.cap = std::chrono::milliseconds(2000);
    slow.should_abort = &stop_flag;
    slow.stop_grace = std::chrono::milliseconds(300);
    std::thread raiser([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        g_stop = true;
    });
    int calls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK_THROWS_AS(io::retry(slow, ebusy(calls), transient_classify,
                              [](const io::RetryEvent&) {}, {}),
                    std::system_error);
    raiser.join();
    const auto took = std::chrono::steady_clock::now() - t0;
    CHECK(calls == 3);   // 1st; at once after the stop; after the grace
    CHECK(took >= std::chrono::milliseconds(400));
    CHECK(took < std::chrono::milliseconds(1500));
}

TEST_CASE("append log: after a stop request a blip keeps its rows, an outage drops them",
          "[append_log][io_retry]")
{
    StopReset reset;
    Scratch scratch;
    const auto path = scratch.path / "loss.tadah";
    spit(path, "HDR\n");
    io::RetryingAppendFile f(path.string());
    io::BackoffPolicy p = exact_policy(10000);
    p.should_abort = &stop_flag;

    /* The final drain of a SIGTERM'd run: stop flag up, one EIO. */
    {
        g_stop = true;
        ScopedFaults faults("writer.append,EIO,count=1,partial=4");
        const std::string rows = stage_rows(f, 1, 3);
        Recorder rec;
        const auto c = f.commit(p, rec.sleep(), rec.on_retry());
        CHECK(c.ok);
        CHECK_FALSE(c.abandoned);
        CHECK(c.attempts == 2);
        CHECK(rec.sleeps == std::vector<long>{10});
        CHECK(slurp(path) == "HDR\n" + rows);
        g_stop = false;
    }

    /* A persistent failure: the grace, then the batch is dropped whole. */
    {
        ScopedFaults faults("writer.append,EIO,always,partial=4");
        (void)stage_rows(f, 4, 3);
        const std::string before = slurp(path);
        Recorder rec;
        const auto c = f.commit(p, [&](std::chrono::milliseconds d) {
            rec.sleeps.push_back(d.count());
            g_stop = true;
        }, rec.on_retry());
        CHECK_FALSE(c.ok);
        CHECK(c.abandoned);
        CHECK(c.rows_lost == 3);
        CHECK(rec.sleeps.size() > 2);
        long after_stop = 0;
        for (std::size_t i = 1; i < rec.sleeps.size(); ++i) after_stop += rec.sleeps[i];
        CHECK(after_stop == 2000);   // the grace (the 1st sleep was cut short)
        CHECK(f.degraded());
        CHECK(slurp(path) == before);   // the partial bytes were rewound
        g_stop = false;
    }

    /* A permanent error keeps its one immediate re-attempt after a stop. */
    {
        g_stop = true;
        io::RetryingAppendFile g((scratch.path / "no_dir" / "x.tadah").string());
        (void)stage_rows(g, 1, 1);
        Recorder rec;
        const auto c = g.commit(p, rec.sleep(), rec.on_retry());
        CHECK_FALSE(c.ok);
        CHECK_FALSE(c.abandoned);
        CHECK(c.attempts == 2);
        CHECK(rec.sleeps.empty());
        g_stop = false;
    }

    /* write_whole_file likewise: a blip after the stop is retried ... */
    {
        g_stop = true;
        ScopedFaults afaults("writer.archive,EIO,count=1");
        Recorder rec;
        const auto r = io::write_whole_file((scratch.path / "pot_1.tadah").string(),
                                            "x", p, rec.sleep());
        CHECK(r.ok);
        CHECK(rec.sleeps == std::vector<long>{10});
        CHECK(slurp(scratch.path / "pot_1.tadah") == "x");
        g_stop = false;
    }
    /* ... an outage gives up after the grace. */
    {
        ScopedFaults afaults("writer.archive,EIO,always");
        Recorder rec;
        const auto r = io::write_whole_file((scratch.path / "pot_2.tadah").string(),
                                            "x", p, [&](std::chrono::milliseconds d) {
                                                rec.sleeps.push_back(d.count());
                                                g_stop = true;
                                            });
        CHECK_FALSE(r.ok);
        CHECK(r.abandoned);
        CHECK(rec.sleeps.size() > 2);
        CHECK_FALSE(fs::exists(scratch.path / "pot_2.tadah"));
        g_stop = false;
    }
}

TEST_CASE("append log: a permanent error's re-attempt is not a transient retry",
          "[append_log][io_retry]")
{
    /* The DIAGNOSTICS counter used to count it as one. */
    Scratch scratch;
    io::RetryingAppendFile f((scratch.path / "l.tadah").string());
    ScopedFaults faults("writer.append,ENOSPC,count=1;writer.append,EIO,skip=2,count=1");
    (void)stage_rows(f, 1, 1);
    Recorder rec;
    REQUIRE(f.commit(exact_policy(1000), rec.sleep(), rec.on_retry()).ok);
    REQUIRE(rec.events.size() == 1);
    CHECK_FALSE(rec.events[0].transient);
    (void)stage_rows(f, 2, 1);
    REQUIRE(f.commit(exact_policy(1000), rec.sleep(), rec.on_retry()).ok);
    REQUIRE(rec.events.size() == 2);
    CHECK(rec.events[1].transient);
}

TEST_CASE("io retry: EACCES is transient only on a network filesystem",
          "[io_retry][append_log]")
{
    /* A genuine permission error on a local disk used to cost
     * the whole retry budget. */
    Scratch scratch;   // a local temporary directory
    const std::string local = (scratch.path / "f").string();
    CHECK(io::is_transient_errno(EACCES));            // errno-only: "can be"
    CHECK(io::is_transient_errno(EBUSY, local));
    CHECK_FALSE(io::is_transient_errno(ENOSPC, local));
    {
        io::testing::ScopedNetworkFs net(true);
        CHECK(io::is_transient_errno(EACCES, local));
        io::testing::ScopedNetworkFs loc(false);
        CHECK_FALSE(io::is_transient_errno(EACCES, local));
    }

    auto run = [&](Recorder& rec) {
        io::RetryingAppendFile f(local);
        ScopedFaults faults("writer.append,EACCES,count=3");
        (void)stage_rows(f, 1, 1);
        return f.commit(exact_policy(1000), rec.sleep(), rec.on_retry());
    };
    {
        io::testing::ScopedNetworkFs loc(false);
        Recorder rec;
        const auto c = run(rec);
        CHECK_FALSE(c.ok);
        CHECK(c.attempts == 2);          // one immediate re-attempt, no sleep
        CHECK(rec.sleeps.empty());
    }
    {
        io::testing::ScopedNetworkFs net(true);
        Recorder rec;
        const auto c = run(rec);
        CHECK(c.ok);
        CHECK(rec.sleeps == std::vector<long>{10, 20, 40});
    }
}

TEST_CASE("archive writer never deletes a file it could not open",
          "[append_log][io_retry]")
{
    /* A read-only archive kept from an earlier run in a reused
     * DUMP directory was retried, then unlinked. */
    if (::geteuid() == 0) {
        SUCCEED("Skipped as root: permissions are not enforced.");
        return;
    }
    /* EACCES is a permission error only on a local filesystem: pin that
     * classification, TMPDIR may be on NFS/Lustre/FUSE. */
    io::testing::ScopedNetworkFs local(false);
    Scratch scratch;
    const auto path = scratch.path / "pot_3.tadah";
    spit(path, "PRECIOUS USER FILE\n");
    fs::permissions(path, fs::perms::owner_read | fs::perms::group_read |
                              fs::perms::others_read);
    Recorder rec;
    const auto r = io::write_whole_file(path.string(), "NEW", exact_policy(1000),
                                        rec.sleep(), rec.on_retry());
    CHECK_FALSE(r.ok);
    CHECK(r.err == EACCES);
    CHECK(r.op == "writer.archive");
    CHECK(rec.sleeps.empty());            // local disk: permanent
    REQUIRE(fs::exists(path));
    CHECK(slurp(path) == "PRECIOUS USER FILE\n");

    /* A file this call DID truncate is removed even when a later attempt
     * could not open it: attempt 1 writes 3 bytes and fails, the file then
     * becomes read-only, the re-attempts fail at open(). */
    const auto mine = scratch.path / "pot_4.tadah";
    ScopedFaults faults("writer.archive,EIO,count=1,partial=3");
    Recorder rec2;
    const auto r2 = io::write_whole_file(
        mine.string(), "ABCDEFGH", exact_policy(1000),
        [&](std::chrono::milliseconds d) {
            rec2.sleeps.push_back(d.count());
            fs::permissions(mine, fs::perms::owner_read);
        },
        rec2.on_retry());
    CHECK_FALSE(r2.ok);
    CHECK(r2.err == EACCES);
    CHECK_FALSE(fs::exists(mine));
}

TEST_CASE("append log rewinds only its own bytes when the file was truncated "
          "underneath it", "[append_log][io_retry]")
{
    /* After an external truncation, a failed write's partial
     * bytes used to be adopted as committed (a torn row before the batch),
     * or a stale offset left a fragment behind. */
    Scratch scratch;
    fs::path path;
    std::unique_ptr<io::RetryingAppendFile> f;
    auto setup = [&](const std::string& name) {
        path = scratch.path / name;
        spit(path, "HDR\n");
        f = std::make_unique<io::RetryingAppendFile>(path.string());
        (void)stage_rows(*f, 1, 3);
        REQUIRE(f->commit(exact_policy(1000)).ok);
    };
    const auto nosleep = [](std::chrono::milliseconds) {};

    SECTION("emptied, then a partial write fails once") {
        setup("b.tadah");
        spit(path, "");
        const std::string batch = stage_rows(*f, 4, 2);
        ScopedFaults faults("writer.append,EIO,count=1,partial=7");
        REQUIRE(f->commit(exact_policy(1000), nosleep).ok);
        CHECK(slurp(path) == batch);
        CHECK(f->offset() == batch.size());
    }
    SECTION("shortened by fewer bytes than the partial write") {
        setup("c.tadah");
        std::string text = slurp(path);
        text.resize(text.size() - 3);
        spit(path, text);
        const std::string batch = stage_rows(*f, 4, 2);
        ScopedFaults faults("writer.append,EIO,count=1,partial=7");
        REQUIRE(f->commit(exact_policy(1000), nosleep).ok);
        CHECK(slurp(path) == text + batch);
        CHECK(f->offset() == text.size() + batch.size());
    }
    SECTION("emptied, a clean batch, then a batch dropped after partial writes") {
        setup("f.tadah");
        spit(path, "");
        const std::string clean = stage_rows(*f, 10, 2);
        REQUIRE(f->commit(exact_policy(1000)).ok);   // offset now stale
        {
            ScopedFaults faults("writer.append,EIO,always,partial=5");
            (void)stage_rows(*f, 20, 2);
            const auto c = f->commit(exact_policy(50), nosleep);
            CHECK_FALSE(c.ok);
        }
        CHECK(slurp(path) == clean);                 // no fragment left
        CHECK(f->offset() == clean.size());
        const std::string next = stage_rows(*f, 30, 1);
        REQUIRE(f->commit(exact_policy(1000)).ok);
        CHECK(slurp(path) == clean + next);          // no torn row
    }
    SECTION("replaced by another file after the partial write") {
        setup("r.tadah");
        const std::string batch = stage_rows(*f, 4, 2);
        ScopedFaults faults("writer.append,EIO,count=1,partial=7");
        const auto c = f->commit(exact_policy(1000), [&](std::chrono::milliseconds) {
            /* The partial bytes went to the old inode; a new file with its
             * own header takes the name. */
            const auto tmp = scratch.path / "r.new";
            spit(tmp, "NEWHDR\n");
            fs::rename(tmp, path);
        });
        REQUIRE(c.ok);
        CHECK(slurp(path) == "NEWHDR\n" + batch);
        CHECK(f->offset() == 7 + batch.size());
    }
}

TEST_CASE("io fault rules parse land and latency", "[io_retry]")
{
    ScopedFaults f("t.site,ENOENT,land,latency=15");
    bool land = false;
    std::size_t partial = 0;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(io::consume_fault("t.site", &partial, &land) == ENOENT);
    CHECK(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(15));
    CHECK(land);
    CHECK(partial == 0);
    CHECK(io::consume_fault("t.site", nullptr, &land) == 0);
    CHECK_THROWS_AS(ScopedFaults("t.site,EIO,land=1"), std::invalid_argument);
    CHECK_THROWS_AS(ScopedFaults("t.site,EIO,latency=x"), std::invalid_argument);
}

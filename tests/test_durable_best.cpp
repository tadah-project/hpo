#include "catch2/catch.hpp"

#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/warmstart_cache.h>

#include <tadah/hpo/io_retry.h>

#include <tadah/core/signal_handler.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <dlib/matrix.h>

/* The runner type of the HPO_Host instantiation the production binary
 * uses (global namespace, tadah/hpo/runners.h); only named here. */
class LammpsRunner;

namespace {

namespace fs = std::filesystem;
using tadah::hpo::durable_best::Key;

std::string payload(const Key& key, const std::string& marker)
{
    std::ostringstream out;
    out << "# HPO_BEST_FORMAT 1\n"
        << "# HPO_BEST_LOSS " << key.loss << "\n"
        << "# HPO_SOURCE_RANK " << key.rank << "\n"
        << "# HPO_SOURCE_SLOT " << key.slot << "\n"
        << "# HPO_SOURCE_EVAL " << key.eval << "\n"
        << marker << "\n";
    return out.str();
}

std::string slurp(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

struct Scratch {
    fs::path path;
    Scratch()
        : path(fs::temp_directory_path() /
               ("tadah_durable_best_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this))))
    {
        fs::create_directories(path);
    }
    ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
};

} // namespace

TEST_CASE("durable best key has deterministic loss/rank/slot/eval ordering",
          "[durable_best]")
{
    using tadah::hpo::durable_best::precedes;
    REQUIRE(precedes(Key{1.0, 9, 9, 9}, Key{2.0, 0, 0, 0}));
    REQUIRE(precedes(Key{1.0, 0, 8, 8}, Key{1.0, 1, 0, 0}));
    REQUIRE(precedes(Key{1.0, 0, 2, 8}, Key{1.0, 0, 3, 0}));
    REQUIRE(precedes(Key{1.0, 0, 2, 7}, Key{1.0, 0, 2, 8}));
    REQUIRE_FALSE(precedes(Key{1.0, 0, 2, 8}, Key{1.0, 0, 2, 8}));
    REQUIRE_FALSE(precedes(Key{NAN, 0, 0, 0}, Key{1.0, 0, 0, 0}));
}

TEST_CASE("durable best publish rejects worse files and atomically replaces better",
          "[durable_best]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";

    const Key first{4.0, 3, 7, 20};
    REQUIRE(publish_if_better(target.string(), first, payload(first, "FIRST")));
    REQUIRE_FALSE(publish_if_better(
        target.string(), Key{5.0, 0, 0, 1}, payload(Key{5.0, 0, 0, 1}, "WORSE")));
    REQUIRE(slurp(target).find("FIRST") != std::string::npos);

    const Key tie_winner{4.0, 2, 99, 99};
    REQUIRE(publish_if_better(target.string(), tie_winner,
                              payload(tie_winner, "TIE_WINNER")));
    const auto bytes = slurp(target);
    REQUIRE(bytes.find("TIE_WINNER") != std::string::npos);
    REQUIRE(parse_key(bytes).has_value());
    REQUIRE(parse_key(bytes)->rank == 2);

    for (const auto& entry : fs::directory_iterator(scratch.path))
        REQUIRE(entry.path().filename().string().find(".tmp.") == std::string::npos);
}

TEST_CASE("durable best serializes competing processes and keeps the true minimum",
          "[durable_best]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    constexpr int workers = 12;
    pid_t children[workers] {};

    for (int i = 0; i < workers; ++i) {
        children[i] = ::fork();
        REQUIRE(children[i] >= 0);
        if (children[i] == 0) {
            const Key key{static_cast<double>(workers - i), i,
                          static_cast<std::size_t>(i % 3),
                          static_cast<std::size_t>(100 + i)};
            try {
                publish_if_better(target.string(), key,
                                  payload(key, "CHILD_" + std::to_string(i)));
                ::_exit(0);
            } catch (...) {
                ::_exit(70);
            }
        }
    }

    for (pid_t child : children) {
        int status = 0;
        REQUIRE(::waitpid(child, &status, 0) == child);
        REQUIRE(WIFEXITED(status));
        REQUIRE(WEXITSTATUS(status) == 0);
    }

    const auto bytes = slurp(target);
    const auto key = parse_key(bytes);
    REQUIRE(key.has_value());
    REQUIRE(key->loss == Approx(1.0));
    REQUIRE(key->rank == workers - 1);
    REQUIRE(bytes.find("CHILD_11") != std::string::npos);
}

TEST_CASE("best HOT batches cannot be rolled back by an older improvement",
          "[durable_best][warmstart]")
{
    namespace ws = tadah::hpo::warmstart;
    using Hot = std::tuple<std::string, std::string, double>;
    ws::clear();

    const std::vector<Hot> newer{{"relax_a", "a0", 3.25},
                                 {"relax_b", "c0", 4.75}};
    const std::vector<Hot> older{{"relax_a", "a0", 9.0},
                                 {"relax_b", "c0", 8.0}};
    REQUIRE(ws::promote_batch(Key{1.0, 1, 4, 20}, newer));
    REQUIRE_FALSE(ws::promote_batch(Key{2.0, 0, 0, 1}, older));
    REQUIRE(ws::best_get("relax_a", "a0") == Approx(3.25));
    REQUIRE(ws::best_get("relax_b", "c0") == Approx(4.75));

    /* Exact-loss ties use the same rank/slot/eval order as the file, and the
     * winning candidate replaces (rather than patches) the complete map. */
    const std::vector<Hot> tie_winner{{"relax_a", "a0", 2.5}};
    REQUIRE(ws::promote_batch(Key{1.0, 0, 9, 99}, tie_winner));
    REQUIRE(ws::best_get("relax_a", "a0") == Approx(2.5));
    REQUIRE_FALSE(ws::best_get("relax_b", "c0").has_value());
    REQUIRE_FALSE(ws::promote_batch(Key{1.0, 1, 0, 0}, older));

    ws::clear();
}

/* remove_lock_file removes the lock file every publication creates, keeps
 * the best potential, accepts an absent lock file, and a later publication
 * creates the lock file again and publishes. A failed unlink is a SysError
 * naming the lock file ("durable.lock_unlink"). */
TEST_CASE("durable best: the lock file goes at the end of a run",
          "[durable_best]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const auto lock = scratch.path / "best_pot.tadah.lock";
    const Key first{2.0, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), first, payload(first, "FIRST")));
    REQUIRE(fs::exists(lock));

    remove_lock_file(target.string());
    CHECK_FALSE(fs::exists(lock));
    CHECK(slurp(target).find("FIRST") != std::string::npos);
    CHECK_NOTHROW(remove_lock_file(target.string()));   /* already gone */
    CHECK_FALSE(fs::exists(lock));

    const Key better{1.0, 0, 0, 2};
    REQUIRE(publish_if_better(target.string(), better, payload(better, "BETTER")));
    CHECK(fs::exists(lock));
    CHECK(slurp(target).find("BETTER") != std::string::npos);

    CHECK_THROWS_AS(remove_lock_file(""), Error);
}

TEST_CASE("fresh-run reset discards an unrelated incumbent",
          "[durable_best]")
{
    /* HPO runs are not resumable: every run starts by removing the best
     * potential a previous run left in the working directory. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key prior{0.01, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), prior, payload(prior, "PRIOR_RUN")));
    REQUIRE(read_key_file(target.string()).has_value());
    REQUIRE(slurp(target).find("PRIOR_RUN") != std::string::npos);

    reset_for_fresh_run(target.string());
    REQUIRE_FALSE(fs::exists(target));
    REQUIRE_FALSE(read_key_file(target.string()).has_value());

    /* The new objective may have a numerically larger loss; it must still own
     * this run's output instead of being suppressed by the stale 0.01. */
    const Key current{9.0, 0, 0, 1};
    REQUIRE(publish_if_better(
        target.string(), current, payload(current, "CURRENT_RUN")));
    REQUIRE(slurp(target).find("CURRENT_RUN") != std::string::npos);
}

TEST_CASE("fresh-run HOT reset preserves WARM's first-evaluation fallback",
          "[durable_best][warmstart]")
{
    namespace ws = tadah::hpo::warmstart;
    ws::clear();
    ws::prime("relax", "a0", 3.75);
    REQUIRE(ws::get(17, "relax", "a0") == Approx(3.75));
    REQUIRE(ws::best_get("relax", "a0") == Approx(3.75));

    ws::reset_best_snapshot_for_fresh_run();
    REQUIRE(ws::get(17, "relax", "a0") == Approx(3.75));
    REQUIRE_FALSE(ws::best_get("relax", "a0").has_value());
    ws::clear();
}

TEST_CASE("same-process exact-loss publishers converge on the compound minimum",
          "[durable_best][threaded]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    constexpr int workers = 12;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int i = workers - 1; i >= 0; --i) {
        threads.emplace_back([&, i] {
            const Key key{1.0, 0, static_cast<std::size_t>(i),
                          static_cast<std::size_t>(100 + i)};
            (void)publish_if_better(
                target.string(), key, payload(key, "THREAD_" + std::to_string(i)));
        });
    }
    for (auto& thread : threads) thread.join();

    const auto bytes = slurp(target);
    const auto key = parse_key(bytes);
    REQUIRE(key.has_value());
    REQUIRE(key->slot == 0u);
    REQUIRE(key->eval == 100u);
    REQUIRE(bytes.find("THREAD_0") != std::string::npos);
}

TEST_CASE("canonical and MPI rank-local best histories remain independent",
          "[durable_best][mpi]")
{
    using namespace tadah::hpo::durable_best;
    namespace ws = tadah::hpo::warmstart;
    ws::clear();
    Scratch scratch;
    const auto canonical = scratch.path / "best_pot.tadah";
    const auto local = scratch.path / "best_pot.tadah.r3";

    const Key global_old{2.0, 0, 4, 20};
    const Key local_better{1.0, 3, 1, 9};
    REQUIRE(publish_if_better(
        canonical.string(), global_old, payload(global_old, "GLOBAL_OLD")));
    REQUIRE(publish_if_better(
        local.string(), local_better, payload(local_better, "LOCAL_BETTER")));

    /* The global comparator is the canonical file, not this rank's
     * legitimately different local history. A candidate can improve global
     * while the local file correctly declines it. */
    const Key global_new{1.5, 3, 0, 30};
    REQUIRE(publish_if_better(
        canonical.string(), global_new, payload(global_new, "GLOBAL_NEW")));
    REQUIRE_FALSE(publish_if_better(
        local.string(), global_new, payload(global_new, "GLOBAL_NEW")));
    REQUIRE(read_key_file(canonical.string())->loss == Approx(1.5));
    REQUIRE(read_key_file(local.string())->loss == Approx(1.0));
    REQUIRE(slurp(canonical).find("GLOBAL_NEW") != std::string::npos);
    REQUIRE(slurp(local).find("LOCAL_BETTER") != std::string::npos);

    /* Another rank can already own a stronger canonical key. The losing
     * canonical publication must not prevent this rank from committing a
     * genuine local improvement and its candidate-local HOT snapshot. */
    const Key remote_champion{0.5, 0, 0, 40};
    REQUIRE(publish_if_better(
        canonical.string(), remote_champion,
        payload(remote_champion, "REMOTE_CHAMPION")));
    const Key rank_local_new{0.8, 3, 2, 41};
    REQUIRE_FALSE(publish_if_better(
        canonical.string(), rank_local_new,
        payload(rank_local_new, "RANK_LOCAL_NEW")));
    REQUIRE(publish_if_better(
        local.string(), rank_local_new,
        payload(rank_local_new, "RANK_LOCAL_NEW")));
    REQUIRE(ws::promote_batch(
        rank_local_new, {{"relax.in", "volume", 12.5}}));
    REQUIRE(ws::best_get("relax.in", "volume") == Approx(12.5));
    REQUIRE(read_key_file(canonical.string())->loss == Approx(0.5));
    REQUIRE(read_key_file(local.string())->loss == Approx(0.8));
    ws::clear();
}

TEST_CASE("multi-rank HPO dispatch is limited to plain PT and PA",
          "[durable_best][mpi][dispatch]")
{
    using namespace tadah::hpo;
    OptimizerSpec spec;

    /* MPI-enabled binaries at size one keep all serial algorithms. */
    spec.lib = "NLOPT";
    spec.algo = "LN_BOBYQA";
    REQUIRE(hpo_mpi_dispatch_supported(1, spec));

    spec.lib = "TADAH";
    spec.algo = "ANNEAL_PT";
    spec.strategy_type = StrategyType::PLAIN;
    REQUIRE(hpo_mpi_dispatch_supported(2, spec));
    spec.algo = "ANNEAL_PA";
    REQUIRE(hpo_mpi_dispatch_supported(8, spec));

    spec.algo = "ANNEAL";
    REQUIRE_FALSE(hpo_mpi_dispatch_supported(2, spec));
    spec.algo = "ANNEAL_PT";
    spec.strategy_type = StrategyType::PRESTAGE;
    REQUIRE_FALSE(hpo_mpi_dispatch_supported(2, spec));
    spec.strategy_type = StrategyType::MULTISTART;
    REQUIRE_FALSE(hpo_mpi_dispatch_supported(2, spec));
}

TEST_CASE("multi-rank HPO dispatch refuses an active RESTART",
          "[durable_best][mpi][dispatch][restart]")
{
    /* The stagnation watchdog is per rank: a rank that stagnated would
     * restart and wait in the driver's next collective while its peers
     * fail. So a launch on more than one rank refuses RESTART, before any
     * evaluation; one rank keeps it. This build has no multi-rank launch,
     * so the rule is checked on the spec, for each world size. */
    using namespace tadah::hpo;
    OptimizerSpec spec;
    spec.strategy_type = StrategyType::PLAIN;
    spec.lib = "TADAH";
    spec.restart.after_stagnation = 200;
    spec.restart.max = 3;

    for (const char* algo : {"ANNEAL_PT", "ANNEAL_PA"}) {
        INFO(algo << " with RESTART AFTER_STAGNATION 200, MAX 3");
        spec.algo = algo;
        CHECK(hpo_mpi_dispatch_supported(1, spec));
        CHECK_FALSE(hpo_mpi_dispatch_supported(2, spec));
        CHECK_FALSE(hpo_mpi_dispatch_supported(8, spec));
    }

    /* RESTART is active only when both AFTER_STAGNATION and MAX are
     * positive; an inactive block is no reason to refuse. */
    spec.algo = "ANNEAL_PT";
    spec.restart.max = 0;
    CHECK(hpo_mpi_dispatch_supported(2, spec));
    spec.restart.max = 3;
    spec.restart.after_stagnation = 0;
    CHECK(hpo_mpi_dispatch_supported(2, spec));
}

TEST_CASE("multi-rank HPO dispatch states why it refuses a launch",
          "[durable_best][mpi][dispatch][restart]")
{
    /* hpo_run() raises this reason as its startup error. */
    using namespace tadah::hpo;
    using Catch::Matchers::Contains;
    OptimizerSpec spec;
    spec.strategy_type = StrategyType::PLAIN;
    spec.lib = "TADAH";
    spec.algo = "ANNEAL_PT";
    CHECK(hpo_mpi_dispatch_refusal(4, spec).empty());

    spec.restart.after_stagnation = 200;
    spec.restart.max = 3;
    CHECK(hpo_mpi_dispatch_refusal(1, spec).empty());
    CHECK(hpo_mpi_dispatch_refusal(2, spec) ==
          "multi-rank HPO does not support RESTART (AFTER_STAGNATION 200, "
          "MAX 3 on 2 ranks): stagnation is tracked per rank; run on one "
          "rank or remove RESTART");

    /* An unsupported driver is named first, RESTART or not. */
    spec.lib = "NLOPT";
    spec.algo = "LN_SBPLX";
    CHECK_THAT(hpo_mpi_dispatch_refusal(2, spec),
               Contains("multi-rank HPO supports only STRATEGY TYPE PLAIN "
                        "with LIB TADAH and ALGO ANNEAL_PT or ANNEAL_PA; "
                        "got NLOPT:LN_SBPLX on 2 ranks"));
    CHECK(hpo_mpi_dispatch_refusal(1, spec).empty());
}

TEST_CASE("RESTART is active only with AFTER_STAGNATION and MAX both "
          "positive", "[restart]")
{
    tadah::hpo::RestartSpec rs;
    CHECK_FALSE(rs.active());
    rs.after_stagnation = 200;
    CHECK_FALSE(rs.active());
    rs.max = 3;
    CHECK(rs.active());
    rs.after_stagnation = 0;
    CHECK_FALSE(rs.active());
}

TEST_CASE("publication rejects payload/key metadata disagreement",
          "[durable_best]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key comparison{1.0, 0, 0, 1};
    const Key rendered{2.0, 0, 0, 1};
    REQUIRE_THROWS_AS(publish_if_better(
        target.string(), comparison, payload(rendered, "MISMATCH")), Error);
    REQUIRE_FALSE(fs::exists(target));
}

TEST_CASE("SIGKILL before rename preserves a complete incumbent",
          "[durable_best]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key incumbent{10.0, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), incumbent,
                              payload(incumbent, "INCUMBENT_COMPLETE")));

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        ::setenv("TADAH_HPO_TEST_STOP_DURABLE_BEST_BEFORE_RENAME", "1", 1);
        const Key candidate{1.0, 1, 4, 2};
        try {
            publish_if_better(target.string(), candidate,
                              payload(candidate, "CANDIDATE_COMPLETE"));
            ::_exit(0); // reached only if the parent lets it continue
        } catch (...) {
            ::_exit(70);
        }
    }

    int status = 0;
    REQUIRE(::waitpid(child, &status, WUNTRACED) == child);
    REQUIRE(WIFSTOPPED(status));
    REQUIRE(WSTOPSIG(status) == SIGSTOP);

    /* The child has fsync'd and closed its unique temp, but has not renamed.
     * Readers still see every byte of the prior canonical candidate. */
    auto bytes = slurp(target);
    REQUIRE(bytes.find("INCUMBENT_COMPLETE") != std::string::npos);
    REQUIRE(parse_key(bytes).has_value());
    REQUIRE(parse_key(bytes)->loss == Approx(10.0));

    REQUIRE(::kill(child, SIGKILL) == 0);
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGKILL);

    bytes = slurp(target);
    REQUIRE(bytes.find("INCUMBENT_COMPLETE") != std::string::npos);
    REQUIRE(parse_key(bytes).has_value());
    REQUIRE(parse_key(bytes)->loss == Approx(10.0));

    bool found_stale_temp = false;
    for (const auto& entry : fs::directory_iterator(scratch.path)) {
        if (entry.path().filename().string().find(".tmp.") != std::string::npos)
            found_stale_temp = true;
    }
    REQUIRE(found_stale_temp);

    /* Stale unique temps are harmless: comparison reads only the canonical
     * path and the next commit uses a different pid/counter name. */
    const Key recovery{0.5, 0, 0, 3};
    REQUIRE(publish_if_better(target.string(), recovery,
                              payload(recovery, "RECOVERY_COMPLETE")));
    bytes = slurp(target);
    REQUIRE(bytes.find("RECOVERY_COMPLETE") != std::string::npos);
    REQUIRE(parse_key(bytes)->loss == Approx(0.5));
}

/* ---------------------------------------------------------------------------
 * Transient I/O retry. Faults are injected at the durable_best
 * syscall sites (io::testing::ScopedFaults); a recording SleepFn replaces the
 * real backoff sleep, and jitter is 0 so the schedule is exact.
 * ------------------------------------------------------------------------- */
namespace {

namespace io = tadah::hpo::io;
using tadah::hpo::io::testing::ScopedFaults;

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
    std::vector<long>            sleeps;
    std::vector<io::RetryEvent>  events;

    tadah::hpo::durable_best::Options options(long budget_ms = 30000)
    {
        tadah::hpo::durable_best::Options opt;
        opt.backoff  = exact_policy(budget_ms);
        opt.sleep    = [this](std::chrono::milliseconds d) { sleeps.push_back(d.count()); };
        opt.on_retry = [this](const io::RetryEvent& ev) { events.push_back(ev); };
        return opt;
    }
    long slept() const
    {
        long s = 0;
        for (long v : sleeps) s += v;
        return s;
    }
};

bool has_temp(const fs::path& dir)
{
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.path().filename().string().find(".tmp.") != std::string::npos)
            return true;
    return false;
}

} // namespace

TEST_CASE("durable best key equality is the exact compound tuple",
          "[durable_best][io_retry]")
{
    using tadah::hpo::durable_best::Key;
    REQUIRE(Key{1.0, 2, 3, 4} == Key{1.0, 2, 3, 4});
    REQUIRE(Key{1.0, 2, 3, 4} != Key{1.0, 2, 3, 5});
    REQUIRE(Key{1.0, 2, 3, 4} != Key{1.0, 2, 4, 4});
    REQUIRE(Key{1.0, 2, 3, 4} != Key{1.0, 1, 3, 4});
    REQUIRE(Key{1.0, 2, 3, 4} != Key{std::nextafter(1.0, 2.0), 2, 3, 4});
}

TEST_CASE("durable best retries a transient lock-open EBUSY with exponential backoff",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{2.0, 0, 1, 7};
    const auto bytes = payload(key, "RETRIED");

    ScopedFaults faults("durable.lock_open,EBUSY,count=3");
    Recorder rec;
    REQUIRE(publish_if_better(target.string(), key, bytes, rec.options()));

    REQUIRE(rec.sleeps == std::vector<long>{10, 20, 40});
    REQUIRE(rec.events.size() == 3);
    for (unsigned i = 0; i < 3; ++i) {
        CHECK(rec.events[i].op == "durable.lock_open");
        CHECK(rec.events[i].err == EBUSY);
        CHECK(rec.events[i].attempt == i + 1);
        CHECK(rec.events[i].path == target.string() + ".lock");
        CHECK(rec.events[i].next_delay.count() == rec.sleeps[i]);
    }
    CHECK(faults.injected("durable.lock_open") == 3);
    CHECK(faults.calls("durable.lock_open") == 4);
    REQUIRE(slurp(target) == bytes);
    REQUIRE_FALSE(has_temp(scratch.path));
}

TEST_CASE("durable best gives up on a persistent transient error within the budget",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{2.0, 0, 1, 7};

    ScopedFaults faults("durable.lock_open,EBUSY,count=always");
    Recorder rec;
    /* 10 + 20 + 40 = 70 ms; the next 80 ms sleep would exceed 100 ms. */
    try {
        publish_if_better(target.string(), key, payload(key, "NEVER"),
                          rec.options(100));
        FAIL("publish_if_better did not throw");
    } catch (const SysError& e) {
        const std::string what = e.what();
        INFO(what);
        CHECK(e.err() == EBUSY);
        CHECK(e.op() == "durable.lock_open");
        CHECK(what.find("cannot open durable-best lock") != std::string::npos);
        CHECK(what.find(std::strerror(EBUSY)) != std::string::npos);
        CHECK(what.find("(gave up after 4 attempts over ") != std::string::npos);
    }
    CHECK(rec.sleeps == std::vector<long>{10, 20, 40});
    CHECK(rec.slept() <= 100);
    CHECK(rec.events.size() == 3);
    REQUIRE_FALSE(fs::exists(target));

    /* The shipped durable policy: 10 ms doubling to a 2 s cap within 30 s. */
    io::BackoffPolicy shipped = io::BackoffPolicy::durable();
    REQUIRE(shipped.budget.count() == 30000);
    for (const double jitter : {0.0, 0.2}) {
        Recorder full;
        Options opt = full.options();
        opt.backoff = shipped;
        opt.backoff.jitter = jitter;
        REQUIRE_THROWS_AS(publish_if_better(target.string(), key,
                                            payload(key, "NEVER"), opt),
                          Error);
        CHECK(full.slept() <= 30000);
        CHECK(full.slept() > 30000 - 2 * 2400);   // at most one capped sleep short
        if (jitter == 0.0) {
            CHECK(full.sleeps.size() == 21);   // 8 doubling + 13 capped
            CHECK(full.sleeps.back() == 2000);
        }
    }
}

TEST_CASE("durable best does not retry a permanent error",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key incumbent{5.0, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), incumbent,
                              payload(incumbent, "INCUMBENT")));

    const Key better{1.0, 0, 0, 2};
    ScopedFaults faults("durable.write,ENOSPC,count=always");
    Recorder rec;
    try {
        publish_if_better(target.string(), better, payload(better, "BETTER"),
                          rec.options());
        FAIL("publish_if_better did not throw");
    } catch (const SysError& e) {
        CHECK(e.err() == ENOSPC);
        CHECK(e.op() == "durable.write");
        CHECK(std::string(e.what()).find("gave up") == std::string::npos);
    }
    CHECK(rec.sleeps.empty());
    CHECK(rec.events.empty());
    CHECK(faults.calls("durable.write") == 1);
    REQUIRE(slurp(target).find("INCUMBENT") != std::string::npos);
    REQUIRE_FALSE(has_temp(scratch.path));
}

TEST_CASE("durable best completes a landed rename when the directory fsync failed",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{3.0, 1, 2, 9};
    const auto bytes = payload(key, "LANDED");

    /* skip=huge: never fires, only counts temp creations. */
    ScopedFaults faults("durable.dir_fsync,EIO,count=1;"
                        "durable.temp_create,EIO,skip=1000000");
    Recorder rec;
    REQUIRE(publish_if_better(target.string(), key, bytes, rec.options()));

    REQUIRE(rec.events.size() == 1);
    CHECK(rec.events[0].op == "durable.dir_fsync");
    CHECK(rec.events[0].err == EIO);
    CHECK(rec.sleeps == std::vector<long>{10});
    /* Attempt 2 found its own key as the incumbent: it fsync'd the directory
     * again and did not write a second temporary. */
    CHECK(faults.calls("durable.dir_fsync") == 2);
    CHECK(faults.calls("durable.temp_create") == 1);
    REQUIRE(slurp(target) == bytes);
    REQUIRE_FALSE(has_temp(scratch.path));

    /* Outside a retry an equal key is still "not better". */
    REQUIRE_FALSE(publish_if_better(target.string(), key, bytes));
}

TEST_CASE("durable best retries a failed rename with a fresh temporary",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key incumbent{5.0, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), incumbent,
                              payload(incumbent, "INCUMBENT")));

    const Key key{1.0, 0, 0, 2};
    const auto bytes = payload(key, "RENAMED");
    ScopedFaults faults("durable.rename,EBUSY,count=1;"
                        "durable.temp_create,EIO,skip=1000000");
    Recorder rec;
    REQUIRE(publish_if_better(target.string(), key, bytes, rec.options()));
    CHECK(rec.events.size() == 1);
    CHECK(rec.events[0].op == "durable.rename");
    CHECK(faults.calls("durable.temp_create") == 2);
    REQUIRE(slurp(target) == bytes);
    REQUIRE_FALSE(has_temp(scratch.path));
}

TEST_CASE("durable best reset and read retry transient errors",
          "[durable_best][io_retry][resume]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{0.5, 0, 0, 3};
    REQUIRE(publish_if_better(target.string(), key, payload(key, "PRIOR")));

    {
        /* EACCES is transient on a network filesystem only. */
        io::testing::ScopedNetworkFs network(true);
        ScopedFaults faults("durable.read_open,EACCES,count=1;"
                            "durable.read,EIO,count=1");
        Recorder rec;
        const auto k = read_key_file(target.string(), rec.options());
        REQUIRE(k.has_value());
        CHECK(*k == key);
        REQUIRE(rec.events.size() == 2);
        CHECK(rec.events[0].op == "durable.read_open");
        CHECK(rec.events[1].op == "durable.read");
    }
    {
        ScopedFaults faults("durable.unlink,EBUSY,count=1");
        Recorder rec;
        reset_for_fresh_run(target.string(), rec.options());
        CHECK(rec.events.size() == 1);
        REQUIRE_FALSE(fs::exists(target));
    }

    /* The unlink landed but the directory fsync failed: the retry sees
     * ENOENT and must still make the removal durable. */
    REQUIRE(publish_if_better(target.string(), key, payload(key, "PRIOR")));
    {
        ScopedFaults faults("durable.dir_fsync,EIO,count=1");
        Recorder rec;
        reset_for_fresh_run(target.string(), rec.options());
        CHECK(rec.events.size() == 1);
        CHECK(faults.calls("durable.dir_fsync") == 2);
        REQUIRE_FALSE(fs::exists(target));
    }

    /* Absent file: still a no-op success, no retry. */
    Recorder rec;
    reset_for_fresh_run(target.string(), rec.options());
    CHECK(rec.events.empty());
}

TEST_CASE("durable best releases its locks between attempts",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key ours{2.0, 0, 0, 5};
    const Key thread_better{1.0, 0, 0, 6};
    const Key process_better{0.5, 1, 0, 7};

    /* Attempt 1 fails on the lock open. While "sleeping", another thread of
     * this process (thread mutex) and then another process (fcntl record
     * lock) each publish a better key. Either would block forever if our
     * locks were still held; attempt 2 re-reads and declines. */
    ScopedFaults faults("durable.lock_open,EBUSY,count=1");
    Recorder rec;
    Options opt = rec.options();
    int child_status = -1;
    bool thread_published = false;
    /* Joins the publisher thread before `faults` and `scratch` are destroyed
     * (declared after them, so destroyed first -- also while a failed
     * REQUIRE unwinds). A publisher still wedged after 30 s means a lock
     * leaked across the retry sleep: no recovery is possible, so end the
     * process rather than let the thread touch a freed fault table or a
     * removed directory. */
    struct Joiner {
        std::thread             t;
        std::shared_future<bool> f;
        ~Joiner()
        {
            if (!t.joinable()) return;
            if (f.valid() &&
                f.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
                std::fputs("publisher thread wedged (lock leaked across retry sleep)\n",
                           stderr);
                std::_Exit(99);
            }
            t.join();
        }
    } joiner;
    opt.sleep = [&](std::chrono::milliseconds d) {
        rec.sleeps.push_back(d.count());
        if (joiner.t.joinable()) return;   // one publisher (REQUIREd below)
        /* Timed so a regression fails instead of hanging. */
        auto done = std::make_shared<std::promise<bool>>();
        joiner.f = done->get_future().share();
        joiner.t = std::thread([done, path = target.string(), k = thread_better] {
            try {
                done->set_value(publish_if_better(path, k, payload(k, "THREAD_BETTER")));
            } catch (...) {
                done->set_value(false);
            }
        });
        thread_published =
            joiner.f.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
            joiner.f.get();
        if (!thread_published) return;
        const pid_t child = ::fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            try {
                ::_exit(publish_if_better(target.string(), process_better,
                                          payload(process_better, "PROCESS_BETTER"))
                        ? 0 : 71);
            } catch (...) {
                ::_exit(70);
            }
        }
        for (int i = 0; i < 1000; ++i) {   // <= 10 s, then fail rather than hang
            const pid_t w = ::waitpid(child, &child_status, WNOHANG);
            if (w == child) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ::kill(child, SIGKILL);
        ::waitpid(child, &child_status, 0);
        child_status = -2;
    };

    REQUIRE_FALSE(publish_if_better(target.string(), ours,
                                    payload(ours, "OURS"), opt));
    REQUIRE(thread_published);
    REQUIRE(child_status >= 0);
    REQUIRE(WIFEXITED(child_status));
    REQUIRE(WEXITSTATUS(child_status) == 0);
    REQUIRE(rec.sleeps.size() == 1);
    const auto bytes = slurp(target);
    REQUIRE(bytes.find("PROCESS_BETTER") != std::string::npos);
    REQUIRE(*parse_key(bytes) == process_better);
}

TEST_CASE("durable best success path never sleeps or reports a retry",
          "[durable_best][io_retry]")
{
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    Recorder rec;
    const Options opt = rec.options();
    for (int i = 0; i < 5; ++i) {
        const Key k{10.0 - i, 0, 0, static_cast<std::size_t>(i + 1)};
        REQUIRE(publish_if_better(target.string(), k,
                                  payload(k, "P" + std::to_string(i)), opt));
    }
    const Key worse{100.0, 0, 0, 99};
    REQUIRE_FALSE(publish_if_better(target.string(), worse,
                                    payload(worse, "WORSE"), opt));
    REQUIRE(read_key_file(target.string(), opt)->loss == Approx(6.0));
    reset_for_fresh_run(target.string(), opt);
    CHECK(rec.sleeps.empty());
    CHECK(rec.events.empty());
}

/* ---------------------------------------------------------------------------
 * Retry edge cases: landed renames, shutdown grace, permission errors
 * ------------------------------------------------------------------------- */
namespace {
std::atomic<bool> g_db_stop{false};
bool db_stop_flag() { return g_db_stop.load(); }
} // namespace

TEST_CASE("durable best recognises a rename that landed but reported ENOENT",
          "[durable_best][io_retry]")
{
    /* An NFS rename retransmitted after it landed fails with
     * ENOENT (the source is gone). That errno is permanent elsewhere, but
     * here the retry re-reads the incumbent and finds its own key. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key incumbent{5.0, 0, 0, 1};
    REQUIRE(publish_if_better(target.string(), incumbent,
                              payload(incumbent, "INCUMBENT")));

    const Key key{1.0, 0, 0, 2};
    const auto bytes = payload(key, "LANDED");
    {
        ScopedFaults faults("durable.rename,ENOENT,count=1,land;"
                            "durable.temp_create,EIO,skip=1000000");
        Recorder rec;
        REQUIRE(publish_if_better(target.string(), key, bytes, rec.options()));
        REQUIRE(rec.events.size() == 1);
        CHECK(rec.events[0].op == "durable.rename (possible landed rename)");
        CHECK(rec.events[0].err == ENOENT);
        CHECK(faults.calls("durable.temp_create") == 1);   // not written twice
        REQUIRE(slurp(target) == bytes);
        REQUIRE_FALSE(has_temp(scratch.path));
    }

    /* Not landed (the temporary vanished): the transaction runs again with
     * a fresh temporary. */
    const Key better{0.5, 0, 0, 3};
    const auto better_bytes = payload(better, "REDONE");
    {
        ScopedFaults faults("durable.rename,ENOENT,count=1;"
                            "durable.temp_create,EIO,skip=1000000");
        Recorder rec;
        REQUIRE(publish_if_better(target.string(), better, better_bytes,
                                  rec.options()));
        CHECK(rec.events.size() == 1);
        CHECK(faults.calls("durable.temp_create") == 2);
        REQUIRE(slurp(target) == better_bytes);
        REQUIRE_FALSE(has_temp(scratch.path));
    }

    /* The directory really vanished: the next attempt's lock open fails
     * with a permanent ENOENT after one short backoff, not 30 s. */
    const auto gone = scratch.path / "gone";
    fs::create_directories(gone);
    const auto gone_target = gone / "best_pot.tadah";
    const Key k3{0.1, 0, 0, 4};
    ScopedFaults faults("durable.rename,ENOENT,count=1");
    Recorder rec;
    Options opt = rec.options();
    opt.sleep = [&](std::chrono::milliseconds d) {
        rec.sleeps.push_back(d.count());
        std::error_code ec;
        fs::remove_all(gone, ec);
    };
    try {
        publish_if_better(gone_target.string(), k3, payload(k3, "NOPE"), opt);
        FAIL("publish_if_better did not throw");
    } catch (const SysError& e) {
        CHECK(e.err() == ENOENT);
        CHECK(e.op() == "durable.lock_open");
    }
    CHECK(rec.sleeps.size() == 1);
}

TEST_CASE("durable best keeps a short grace after a stop request, then abandons",
          "[durable_best][io_retry]")
{
    /* A shutdown signal must neither hold the exit for the whole 30 s
     * budget nor end retrying at once (one blip right after a SLURM
     * time-limit signal would lose the in-flight best and fail the run).
     * After a stop request the retry goes on for at most stop_grace (2 s);
     * a candidate still not persisted then is an error. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{2.0, 0, 1, 7};

    SECTION("stop already requested: one transient fault is retried") {
        g_db_stop = true;
        ScopedFaults faults("durable.lock_open,EBUSY,count=1");
        Recorder rec;
        Options opt = rec.options();
        opt.backoff.should_abort = &db_stop_flag;
        REQUIRE(publish_if_better(target.string(), key, payload(key, "KEPT"), opt));
        g_db_stop = false;
        CHECK(rec.sleeps == std::vector<long>{10});
        CHECK(slurp(target) == payload(key, "KEPT"));
    }
    SECTION("a persistent fault: the grace, then abandoned") {
        g_db_stop = false;
        ScopedFaults faults("durable.lock_open,EBUSY,count=always");
        Recorder rec;
        Options opt = rec.options();
        opt.backoff.should_abort = &db_stop_flag;
        opt.sleep = [&](std::chrono::milliseconds d) {
            rec.sleeps.push_back(d.count());
            if (rec.sleeps.size() == 2) g_db_stop = true;
        };
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"), opt);
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            const std::string what = e.what();
            INFO(what);
            CHECK(e.err() == EBUSY);
            CHECK(what.find("(retry abandoned: shutdown requested) after 9 attempts over ")
                  != std::string::npos);
            CHECK(what.find("gave up") == std::string::npos);
            CHECK_FALSE(e.may_have_landed());
        }
        g_db_stop = false;
        /* 10, 20 (the stop arrives), at once, then 2 s of grace. */
        CHECK(rec.sleeps == std::vector<long>{10, 20, 40, 80, 160, 320, 640, 760});
        CHECK(faults.calls("durable.lock_open") == 9);
        REQUIRE_FALSE(fs::exists(target));
    }

    /* The default options use the shutdown flag and the 2 s grace. */
    CHECK(default_backoff().should_abort == &tadah::core::SignalHandler::stop_requested);
    CHECK(Options{}.backoff.should_abort == &tadah::core::SignalHandler::stop_requested);
    CHECK(default_backoff().budget == io::BackoffPolicy::durable().budget);
    CHECK(default_backoff().stop_grace == std::chrono::milliseconds(2000));
}

TEST_CASE("durable best reports a failed rename as maybe on disk only when it "
          "may have landed", "[durable_best][io_retry]")
{
    /* Every rename failure used to count as "may hold eval N (write
     * not synced)", EROFS/EXDEV included, where the destination is provably
     * untouched. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{2.0, 0, 1, 7};
    auto landed = [&](const std::string& spec) {
        ScopedFaults faults(spec);
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "X"),
                              rec.options(100));
        } catch (const SysError& e) {
            INFO(spec << ": " << e.what());
            CHECK(e.op() == "durable.rename");
            CHECK_FALSE(has_temp(scratch.path));
            return e.may_have_landed();
        }
        FAIL(spec << ": publish_if_better did not throw");
        return false;
    };
    CHECK_FALSE(landed("durable.rename,EROFS,always"));
    CHECK_FALSE(landed("durable.rename,EXDEV,always"));
    CHECK_FALSE(landed("durable.rename,ENOSPC,always"));
    CHECK_FALSE(landed("durable.rename,EPERM,always"));
    CHECK(landed("durable.rename,EIO,always"));      // transient: can land
    CHECK(landed("durable.rename,ENOENT,always"));   // a retransmitted rename
    /* Sticky: attempt 1 may have landed (EIO), attempt 2 says EROFS. */
    CHECK(landed("durable.rename,EIO,count=1;durable.rename,EROFS,skip=1,always"));
    {
        io::testing::ScopedNetworkFs local(false);
        CHECK_FALSE(landed("durable.rename,EACCES,always"));
    }
    {
        io::testing::ScopedNetworkFs network(true);
        CHECK(landed("durable.rename,EACCES,always"));
    }
    REQUIRE_FALSE(fs::exists(target));
}

TEST_CASE("durable best fails at once on EACCES on a local filesystem",
          "[durable_best][io_retry]")
{
    /* A genuine permission error cost the whole 30 s budget. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    const Key key{2.0, 0, 1, 7};
    {
        io::testing::ScopedNetworkFs local(false);
        ScopedFaults faults("durable.lock_open,EACCES,count=always");
        Recorder rec;
        CHECK_THROWS_AS(publish_if_better(target.string(), key,
                                          payload(key, "NEVER"), rec.options()),
                        SysError);
        CHECK(rec.sleeps.empty());
        CHECK(faults.calls("durable.lock_open") == 1);
    }
    {
        io::testing::ScopedNetworkFs network(true);
        ScopedFaults faults("durable.lock_open,EACCES,count=2");
        Recorder rec;
        REQUIRE(publish_if_better(target.string(), key, payload(key, "SMB"),
                                  rec.options()));
        CHECK(rec.sleeps == std::vector<long>{10, 20});
    }
    /* A real read-only lock file: no override, the real statfs decides.
     * TMPDIR can be on NFS/Lustre/GPFS/FUSE (HPC nodes, CI), and non-Linux
     * builds always count as network: probe the classification
     * with the same call and expect what it says. */
    if (::geteuid() != 0) {
        const auto lock = scratch.path / "ro" / "best_pot.tadah.lock";
        fs::create_directories(lock.parent_path());
        { std::ofstream(lock) << ""; }
        fs::permissions(lock, fs::perms::owner_read);
        const bool network = io::is_transient_errno(EACCES, lock.string());
        Recorder rec;
        CHECK_THROWS_AS(publish_if_better((scratch.path / "ro" / "best_pot.tadah").string(),
                                          key, payload(key, "RO"), rec.options()),
                        SysError);
        if (network) {
            WARN("TMPDIR is on a network/FUSE filesystem (or this is not Linux): "
                 "EACCES was retried as transient");
            CHECK_FALSE(rec.sleeps.empty());
        } else {
            CHECK(rec.sleeps.empty());
        }
    }
}

TEST_CASE("durable best no_backoff: a single attempt; only a possibly "
          "landed rename is re-attempted once",
          "[durable_best][io_retry][failure_handling]")
{
    /* The Options HPO_Host publishes with once a publication failed for
     * good (durable_best::FailLatch). Never more persistent than the
     * normal mode: every errno the normal mode does not retry fails at
     * once, and a transient one gets no retry. Nothing sleeps in any case. */
    using namespace tadah::hpo::durable_best;
    Scratch scratch;
    const auto target = scratch.path / "best_pot.tadah";
    auto no_backoff = [](Recorder& rec) {
        Options opt = rec.options();
        opt.no_backoff = true;
        return opt;
    };

    SECTION("a transient errno: one attempt, then gave up") {
        const Key key{2.0, 0, 1, 7};
        ScopedFaults faults("durable.lock_open,EBUSY,count=1");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            const std::string what = e.what();
            INFO(what);
            CHECK(e.err() == EBUSY);
            CHECK(what.find("(gave up after 1 attempt over ") != std::string::npos);
        }
        CHECK(faults.calls("durable.lock_open") == 1);
        CHECK(rec.sleeps.empty());
        CHECK(rec.events.empty());
        REQUIRE_FALSE(fs::exists(target));
        /* the next call (storage recovered) publishes */
        REQUIRE(publish_if_better(target.string(), key, payload(key, "LATER"),
                                  no_backoff(rec)));
        REQUIRE(slurp(target).find("LATER") != std::string::npos);
    }
    SECTION("a permanent errno once: fails at once, as without no_backoff") {
        const Key key{2.0, 0, 1, 7};
        ScopedFaults faults("durable.write,ENOSPC,count=1");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            CHECK(e.err() == ENOSPC);
            CHECK(std::string(e.what()).find("gave up") == std::string::npos);
        }
        CHECK(faults.calls("durable.write") == 1);
        CHECK(rec.events.empty());
        CHECK(rec.sleeps.empty());
        REQUIRE_FALSE(fs::exists(target));
        REQUIRE_FALSE(has_temp(scratch.path));
        /* the next call (storage recovered) publishes */
        const auto bytes = payload(key, "LATER");
        REQUIRE(publish_if_better(target.string(), key, bytes, no_backoff(rec)));
        REQUIRE(slurp(target) == bytes);
    }
    SECTION("a persistent permanent errno: one attempt, the error unchanged") {
        const Key key{2.0, 0, 1, 7};
        ScopedFaults faults("durable.write,ENOSPC,count=always");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            CHECK(e.err() == ENOSPC);
            CHECK(std::string(e.what()).find("gave up") == std::string::npos);
        }
        CHECK(faults.calls("durable.write") == 1);
        CHECK(rec.events.empty());
        CHECK(rec.sleeps.empty());
        REQUIRE_FALSE(fs::exists(target));
        REQUIRE_FALSE(has_temp(scratch.path));
    }
    SECTION("a rename ENOENT that did not land, then a transient errno: "
            "gave up after 2 attempts") {
        const Key key{2.0, 0, 1, 7};
        ScopedFaults faults("durable.rename,ENOENT,count=1;"
                            "durable.lock_open,EBUSY,skip=1,count=1");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            CHECK(e.err() == EBUSY);
            CHECK(std::string(e.what()).find("(gave up after 2 attempts over ") !=
                  std::string::npos);
        }
        REQUIRE(rec.events.size() == 1);
        CHECK(rec.events[0].op == "durable.rename (possible landed rename)");
        CHECK(rec.events[0].err == ENOENT);
        CHECK(rec.events[0].attempt == 1u);
        CHECK_FALSE(rec.events[0].transient);
        CHECK(rec.events[0].next_delay.count() == 0);
        CHECK(rec.sleeps.empty());
        REQUIRE_FALSE(fs::exists(target));
        REQUIRE_FALSE(has_temp(scratch.path));
    }
    SECTION("a persistent rename ENOENT: two attempts, the error unchanged") {
        const Key key{2.0, 0, 1, 7};
        ScopedFaults faults("durable.rename,ENOENT,count=always");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "NEVER"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            CHECK(e.err() == ENOENT);
            CHECK(std::string(e.what()).find("gave up") == std::string::npos);
        }
        CHECK(faults.calls("durable.rename") == 2);
        CHECK(rec.events.size() == 1);
        CHECK(rec.sleeps.empty());
        REQUIRE_FALSE(fs::exists(target));
        REQUIRE_FALSE(has_temp(scratch.path));
    }
    SECTION("a rename that landed but reported ENOENT completes") {
        const Key key{1.0, 0, 0, 2};
        const auto bytes = payload(key, "LANDED");
        ScopedFaults faults("durable.rename,ENOENT,count=1,land;"
                            "durable.temp_create,EIO,skip=1000000");
        Recorder rec;
        REQUIRE(publish_if_better(target.string(), key, bytes, no_backoff(rec)));
        REQUIRE(rec.events.size() == 1);
        CHECK_FALSE(rec.events[0].transient);
        CHECK(faults.calls("durable.temp_create") == 1);   // not written twice
        REQUIRE(slurp(target) == bytes);
        REQUIRE_FALSE(has_temp(scratch.path));
    }
    SECTION("a directory fsync that failed after the rename: maybe on disk") {
        const Key key{3.0, 1, 2, 9};
        ScopedFaults faults("durable.dir_fsync,EIO,count=1");
        Recorder rec;
        try {
            publish_if_better(target.string(), key, payload(key, "UNSYNCED"),
                              no_backoff(rec));
            FAIL("publish_if_better did not throw");
        } catch (const SysError& e) {
            CHECK(e.err() == EIO);
            CHECK(e.may_have_landed());
        }
        CHECK(rec.sleeps.empty());
        CHECK(rec.events.empty());
    }
}

/* Evaluations that cannot become the new best skip HPO_Host's best mutex
 * (they used to queue behind every publication, including a transient-I/O
 * backoff of up to 30 s). The pre-check must never reject a candidate the
 * locked decision could accept: lower_loss and tied_compound_winner both
 * need loss <= current best. The host instantiation the production binary
 * uses; may_become_best is a static constexpr member, so nothing is built. */
TEST_CASE("HPO_Host new-best pre-check rejects only losses that cannot win",
          "[hpo_host][durable_best][io_retry]")
{
    using Host = tadah::hpo::HPO_Host<dlib::matrix<double, 0, 1>, LammpsRunner>;
    constexpr double kMax = std::numeric_limits<double>::max();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    static_assert(Host::may_become_best(1.0, 2.0), "lower loss may win");
    static_assert(Host::may_become_best(2.0, 2.0), "an exact tie may win (compound key)");
    static_assert(!Host::may_become_best(2.5, 2.0), "a higher loss cannot win");
    CHECK(Host::may_become_best(1e9, kMax));      // first evaluation: reaches the gate
    CHECK(Host::may_become_best(kMax, kMax));     // a tie at the initial best
    CHECK_FALSE(Host::may_become_best(nan, 1.0));
    CHECK_FALSE(Host::may_become_best(1.0, nan));
    CHECK_FALSE(Host::may_become_best(inf, kMax));
    CHECK(Host::may_become_best(-inf, 0.0));
    CHECK_FALSE(Host::may_become_best(std::nextafter(2.0, 3.0), 2.0));
}

#include "catch2/catch.hpp"

#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/warmstart_cache.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

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

/*
 * Strategy-level unit tests for InitStrategy implementations.
 *
 * These tests build a minimal HPOTargets fixture by writing a temp
 * config file (mirroring the ScopedTempFile pattern in test_hpo.cpp)
 * and then exercise build_init_strategy + generate() directly.
 */

#include "catch2/catch.hpp"

#include "finish_within.h"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/warmstart_cache.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <vector>

namespace {

struct StrategyTempFile {
    std::filesystem::path path;
    explicit StrategyTempFile(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_init_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".in"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~StrategyTempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/* The journal's OPTIMISER section (and the SEARCH SPACE section that
 * follows it) for `sp`, as HPOLog::write_optimizer_section writes it. */
std::string optimizer_section(const tadah::hpo::OptimizerSpec& sp)
{
    auto& log = tadah::hpo::HPOLog::instance();
    const auto path = std::filesystem::temp_directory_path() /
        ("hpo_optsec_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(path.string(), "test", "/tmp", 1);
    log.write_optimizer_section(sp, {});
    log.finish("TEST");
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return ss.str();
}

/* The lines of `text` from the one that holds `header` to the next line
 * at its indentation or less ("" if no line holds it): the block the
 * journal prints under that header. */
std::string journal_block(const std::string& text, const std::string& header)
{
    std::istringstream in(text);
    std::string line, out;
    std::size_t indent = std::string::npos;
    while (std::getline(in, line)) {
        const std::size_t ind = line.find_first_not_of(' ');
        if (indent == std::string::npos) {
            if (line.find(header) != std::string::npos) {
                indent = ind;
                out = line + "\n";
            }
            continue;
        }
        if (ind == std::string::npos || ind <= indent) break;
        out += line + "\n";
    }
    return out;
}

/* Build a synthetic OptimizerSpec with explicit bounds (no optimap, so
 * generators see the bounds we set without depending on a full
 * descriptor configuration). */
tadah::hpo::OptimizerSpec
make_synthetic_spec(std::vector<double> low,
                    std::vector<double> high,
                    std::vector<bool>   log_scaled = {})
{
    tadah::hpo::OptimizerSpec sp;
    sp.low = std::move(low);
    sp.high = std::move(high);
    if (log_scaled.empty()) log_scaled.assign(sp.low.size(), false);
    sp.log_scaled = std::move(log_scaled);
    sp.is_int.assign(sp.low.size(), false);

    /* Mimic analyse_logscale's effect for log-flagged dims. The caller
     * is expected to pass already-untransformed bounds; we transform
     * the log-flagged ones in place. */
    for (std::size_t d = 0; d < sp.low.size(); ++d) {
        if (sp.log_scaled[d]) {
            sp.low[d]  = std::log(sp.low[d]);
            sp.high[d] = std::log(sp.high[d]);
        }
    }
    return sp;
}

/* Generators don't read targets.optimap when bounds are provided
 * directly via the spec, except CONFIG which does. We need a stand-in
 * HPOTargets; provide an empty one via a no-OPTIM tempfile. */
struct StubTargets {
    StrategyTempFile file;
    tadah::core::Context ctx;
    std::unique_ptr<tadah::hpo::HPOTargets> trg;

    StubTargets()
    : file("OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n"),
      trg(std::make_unique<tadah::hpo::HPOTargets>(ctx, file.path.string()))
    {}
};

}  // namespace

TEST_CASE("RandomInitStrategy: deterministic with fixed seed",
          "[HPO][init_strategy][random]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({-1.0, -2.0, 0.0}, {1.0, 2.0, 5.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"6"}});
    ispec.seed = 42;
    auto strat = build_init_strategy(ispec);

    InitContext ic{sp, *st.trg, *ispec.seed};
    auto a = strat->generate(ic, 6);
    auto b = strat->generate(ic, 6);
    REQUIRE(a.size() == 6);
    REQUIRE(a == b);
}

TEST_CASE("RandomInitStrategy: outputs respect bounds",
          "[HPO][init_strategy][random]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({-1.0, -2.0, 0.0}, {1.0, 2.0, 5.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"100"}});
    ispec.seed = 7;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, *ispec.seed};
    auto pts = strat->generate(ic, 100);
    REQUIRE(pts.size() == 100);
    for (const auto& p : pts) {
        REQUIRE(p.size() == 3);
        for (std::size_t d = 0; d < 3; ++d) {
            CHECK(p[d] >= sp.low[d] - 1e-12);
            CHECK(p[d] <= sp.high[d] + 1e-12);
        }
    }
}

TEST_CASE("RandomInitStrategy: log-scaled dims sampled in log-space",
          "[HPO][init_strategy][random][logscale]")
{
    using namespace tadah::hpo;
    /* dim 0: linear [0,1]; dim 1: log [1e-3, 1e3]. */
    auto sp = make_synthetic_spec({0.0, 1e-3}, {1.0, 1e3}, {false, true});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"2000"}});
    ispec.seed = 3;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, *ispec.seed};
    auto pts = strat->generate(ic, 2000);

    /* For a uniform-on-log sample, mean(log(x)) ≈ (log_low+log_high)/2 = 0;
     * naive linear sample would have mean ≈ 500 → log-mean ≈ 6.2. */
    double sum_log = 0.0;
    for (const auto& p : pts) {
        REQUIRE(p[1] > 0.0);
        sum_log += std::log(p[1]);
    }
    const double mean_log = sum_log / static_cast<double>(pts.size());
    CHECK(std::abs(mean_log) < 0.3);
}

TEST_CASE("LhsInitStrategy: stratification — one sample per stratum per dim",
          "[HPO][init_strategy][lhs]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;

    const std::size_t k = 8;
    InitStrategySpec ispec;
    ispec.chain.push_back({"LHS", {std::to_string(k)}});
    ispec.seed = 11;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, *ispec.seed};
    auto pts = strat->generate(ic, k);
    REQUIRE(pts.size() == k);

    /* For each dim, every stratum [i/k, (i+1)/k] must have exactly one
     * sample. */
    for (std::size_t d = 0; d < 2; ++d) {
        std::set<std::size_t> strata_seen;
        for (const auto& p : pts) {
            const std::size_t stratum =
                std::min(k - 1, static_cast<std::size_t>(p[d] * k));
            REQUIRE(strata_seen.insert(stratum).second);
        }
        CHECK(strata_seen.size() == k);
    }
}

TEST_CASE("LhsInitStrategy: deterministic with fixed seed",
          "[HPO][init_strategy][lhs]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({-3.0, 0.0}, {3.0, 1.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"LHS", {"8"}});
    ispec.seed = 99;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, *ispec.seed};
    auto a = strat->generate(ic, 8);
    auto b = strat->generate(ic, 8);
    REQUIRE(a == b);
}

TEST_CASE("SobolInitStrategy: deterministic and bounds-compliant",
          "[HPO][init_strategy][sobol]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({-1.0, 0.0, 5.0}, {1.0, 1.0, 10.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"SOBOL", {"32"}});
    ispec.seed = 0;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    auto a = strat->generate(ic, 32);
    auto b = strat->generate(ic, 32);
    REQUIRE(a.size() == 32);
    REQUIRE(a == b);
    for (const auto& p : a)
        for (std::size_t d = 0; d < 3; ++d) {
            CHECK(p[d] >= sp.low[d] - 1e-12);
            CHECK(p[d] <= sp.high[d] + 1e-12);
        }
}

TEST_CASE("SobolInitStrategy: cell coverage on unit square",
          "[HPO][init_strategy][sobol]")
{
    /* For K=64 points on [0,1]^2, splitting each axis into 8 cells gives
     * 64 cells. A Sobol sequence is constructed so that early samples
     * fill the cube uniformly — every cell of an 8×8 grid should
     * receive at least one point. Random uniform samples typically
     * leave a few empty cells. This is a robust space-filling test. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"SOBOL", {"64"}});
    ispec.seed = 0;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    auto pts = strat->generate(ic, 64);

    std::set<std::pair<int,int>> cells;
    for (const auto& p : pts) {
        const int cx = std::min(7, static_cast<int>(p[0] * 8));
        const int cy = std::min(7, static_cast<int>(p[1] * 8));
        cells.insert({cx, cy});
    }
    /* Sobol fills nearly all 64 cells with K=64; require ≥90% coverage. */
    CHECK(cells.size() >= 58);
}

namespace {

/* The first `n` points of the Sobol sequence in this spec's box: SEED 0
 * skips nothing. */
std::vector<std::vector<double>>
sobol_sequence(const tadah::hpo::OptimizerSpec& sp,
               const tadah::hpo::HPOTargets& trg, std::size_t n)
{
    using namespace tadah::hpo;
    InitStrategySpec ispec;
    ispec.chain.push_back({"SOBOL", {std::to_string(n)}});
    InitContext ic{sp, trg, 0};
    return build_init_strategy(ispec)->generate(ic, n);
}

/* The points METHOD SOBOL <k> draws with `seed` (K_TOTAL `k_total`, 0 for
 * none). */
std::vector<std::vector<double>>
sobol_draw(const tadah::hpo::OptimizerSpec& sp,
           const tadah::hpo::HPOTargets& trg, std::size_t k,
           std::uint64_t seed, std::size_t k_total = 0)
{
    using namespace tadah::hpo;
    InitStrategySpec ispec;
    ispec.chain.push_back({"SOBOL", {std::to_string(k)}});
    InitContext ic{sp, trg, seed};
    return build_init_strategy(ispec)->generate(ic, k_total);
}

}  // namespace

TEST_CASE("SobolInitStrategy: the first points are the Sobol sequence's",
          "[HPO][init_strategy][sobol]")
{
    /* Pins the Sobol points (Boost's boost::random::sobol, Joe-Kuo
     * direction numbers, the zero point skipped) in the unit cube, SEED 0:
     * exact binary fractions, so equality is exact. They must not change
     * with the Boost the build uses (every SOBOL start depends on them). */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    StubTargets st;
    const std::vector<std::vector<double>> want = {
        {0.5, 0.5, 0.5},       {0.75, 0.25, 0.25},     {0.25, 0.75, 0.75},
        {0.375, 0.375, 0.625}, {0.875, 0.875, 0.125},  {0.625, 0.125, 0.875},
        {0.125, 0.625, 0.375}, {0.1875, 0.3125, 0.9375}};
    const auto pts = sobol_sequence(sp, *st.trg, 8);
    REQUIRE(pts.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("point " << i);
        CHECK(pts[i] == want[i]);
    }
}

TEST_CASE("SobolInitStrategy: a seed selects a whole block of K points",
          "[HPO][init_strategy][sobol][seed]")
{
    /* A seed used to skip `seed` points, so the seeds s and s + 1 -- restart
     * rounds r and r + 1 under a RESTART SEED, or an INIT SEED and restart
     * 1 with the same SEED -- shared K - 1 of their K points. Seed s now
     * draws block s of K points: points s*K ... s*K + K - 1. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, -1.0}, {1.0, 1.0});
    StubTargets st;
    const auto seq = sobol_sequence(sp, *st.trg, 200);
    REQUIRE(seq.size() == 200u);

    const std::size_t K = 3;
    for (const std::uint64_t seed : {1u, 42u, 43u, 44u, 60u}) {
        INFO("SOBOL 3, seed " << seed);
        const auto pts = sobol_draw(sp, *st.trg, K, seed);
        REQUIRE(pts.size() == K);
        for (std::size_t i = 0; i < K; ++i)
            CHECK(pts[i] == seq[seed * K + i]);
    }

    /* Consecutive seeds share no point. */
    const auto a = sobol_draw(sp, *st.trg, K, 43);
    const auto b = sobol_draw(sp, *st.trg, K, 44);
    for (const auto& p : a)
        CHECK(std::find(b.begin(), b.end(), p) == b.end());

    /* K_TOTAL caps a lone SOBOL: K_TOTAL 2 of SOBOL 5 with seed 7 keeps
     * the first 2 points of block 7 of 5. (K_TOTAL replaced the K of a lone
     * method: SOBOL 2 with K_TOTAL 5 drew block 7 of 5, 5 points.) */
    const auto t = sobol_draw(sp, *st.trg, 5, 7, /*k_total=*/2);
    REQUIRE(t.size() == 2u);
    for (std::size_t i = 0; i < 2; ++i) CHECK(t[i] == seq[7 * 5 + i]);
    CHECK(sobol_draw(sp, *st.trg, 2, 7, /*k_total=*/5).size() == 2u);
}

TEST_CASE("SobolInitStrategy: SOBOL 1 still draws the point its seed "
          "names", "[HPO][init_strategy][sobol][seed]")
{
    /* A block of one point is the point itself, so METHOD SOBOL 1 is
     * unchanged: seed s draws point s. The seed's low 16 bits select the
     * block (seed 65536 + s draws what seed s draws), as they always did. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, -1.0, 2.0}, {1.0, 1.0, 3.0});
    StubTargets st;
    const auto seq = sobol_sequence(sp, *st.trg, 100);
    for (const std::uint64_t seed : {1u, 5u, 42u, 99u}) {
        INFO("SOBOL 1, seed " << seed);
        const auto pts = sobol_draw(sp, *st.trg, 1, seed);
        REQUIRE(pts.size() == 1u);
        CHECK(pts.front() == seq[seed]);
        CHECK(sobol_draw(sp, *st.trg, 1, seed + 65536u) == pts);
    }
}

/* Pins: METHOD SOBOL K with K a power of two (2, 4, 8, 16), in 1 to 5
 * dimensions of the unit cube and with the seeds 0, 1, 2 and 7: each
 * coordinate of the K points has exactly one point in each slice [j/K,
 * (j+1)/K) of its range, and no point is the low corner of the box (every
 * coordinate 0) -- a block of 2^m Sobol points that starts at a multiple of
 * 2^m is such a net.
 * Before the fix: every block started one point late (Sobol points bK + 1
 * to bK + K for block b), and each of the 80 blocks had a slice with two
 * points and one with none: K = 4 in 2-D with seed 0 drew (0.5, 0.5),
 * (0.75, 0.25), (0.25, 0.75) and (0.375, 0.375), 0, 2, 1 and 1 points in the
 * quarters of each coordinate. */
TEST_CASE("SobolInitStrategy: a block of K = 2^m points puts one point in "
          "each 1/K of every range", "[HPO][init_strategy][sobol][seed]")
{
    using namespace tadah::hpo;
    StubTargets st;
    for (std::size_t dim = 1; dim <= 5; ++dim) {
        const auto sp = make_synthetic_spec(std::vector<double>(dim, 0.0),
                                            std::vector<double>(dim, 1.0));
        for (const std::size_t K : {2u, 4u, 8u, 16u}) {
            for (const std::uint64_t seed : {0u, 1u, 2u, 7u}) {
                INFO("dimensions " << dim << ", SOBOL " << K << ", seed "
                     << seed);
                const auto pts = sobol_draw(sp, *st.trg, K, seed);
                REQUIRE(pts.size() == K);
                for (std::size_t d = 0; d < dim; ++d) {
                    INFO("coordinate " << d + 1);
                    std::vector<std::size_t> per_slice(K, 0);
                    for (const auto& p : pts) {
                        REQUIRE(p[d] >= 0.0);
                        REQUIRE(p[d] < 1.0);
                        ++per_slice[static_cast<std::size_t>(
                            p[d] * static_cast<double>(K))];
                    }
                    CHECK(per_slice == std::vector<std::size_t>(K, 1));
                }
                for (const auto& p : pts)
                    CHECK(p != std::vector<double>(dim, 0.0));
            }
        }
    }
}

TEST_CASE("SobolInitStrategy: a K above 1000000 is refused before a block "
          "is drawn", "[HPO][init_strategy][sobol][seed]")
{
    /* Block s of K points starts at point s*K; that product, and the end
     * of the block, must stay within the 64-bit sequence. A METHOD line's
     * K is at most 1000000, so every block a seed selects (s < 65536)
     * lies within it; before the fix K = 2^62 was read and refused only
     * when the block was drawn. K_TOTAL caps the points of a block and
     * draws none past it, so a request for 2^62 points draws the K of the
     * block: the legs in which a caller asked SOBOL 1 for 2^62 points
     * (k_total), a block of 2^62 points past the end of the sequence,
     * left this case (K_TOTAL replaced the K of a lone METHOD line). */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;
    const std::size_t huge = std::size_t{1} << 62;
    CHECK_THROWS_WITH(sobol_draw(sp, *st.trg, huge, 3),
                      "INIT METHOD SOBOL K must be an integer from 1 to "
                      "1000000, got '" + std::to_string(huge) + "'");
    for (const std::uint64_t seed : {3u, 4u, 65535u}) {
        INFO("seed " << seed);
        CHECK(sobol_draw(sp, *st.trg, 1, seed, huge).size() == 1u);
    }
}

TEST_CASE("ChainStrategy: concatenates and de-dups across producers",
          "[HPO][init_strategy][chain]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({-1.0, -1.0}, {1.0, 1.0});
    StubTargets st;

    /* CONFIG (1 zero-dim point because optimap is empty) +
     * RANDOM 4 non-empty points → CONFIG yields a zero-dim point but
     * we have a 2-dim spec, so config_point_from_targets returns {}.
     * The chain de-dups across producers in transformed space. */
    InitStrategySpec ispec;
    ispec.chain.push_back({"CONFIG", {}});
    ispec.chain.push_back({"RANDOM", {"4"}});
    ispec.seed = 2026;

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 2026};
    auto pts = strat->generate(ic, 0 /* unbounded */);

    /* Empty optimap → ConfigInitStrategy contributes a zero-dim vector
     * which the chain keeps as-is (different size from RANDOM's
     * 2-dim points, so de-dup keeps both classes). */
    REQUIRE(pts.size() >= 4);
    /* The 2-dim entries must respect the bounds. */
    for (const auto& p : pts) if (p.size() == 2) {
        CHECK(p[0] >= sp.low[0] - 1e-12);
        CHECK(p[0] <= sp.high[0] + 1e-12);
        CHECK(p[1] >= sp.low[1] - 1e-12);
        CHECK(p[1] <= sp.high[1] + 1e-12);
    }
}

TEST_CASE("ChainStrategy: INIT_K caps the total output",
          "[HPO][init_strategy][chain]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"5"}});
    ispec.chain.push_back({"LHS",    {"5"}});
    ispec.seed = 1234;

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 1234};
    /* k_total = 4 → chain stops at 4 even though 5+5=10 are available */
    auto pts = strat->generate(ic, 4);
    CHECK(pts.size() == 4);
}

/* Pins: K_TOTAL caps the points of a chain, a lone METHOD line included:
 * a lone RANDOM, LHS or SOBOL of K points asked for k draws its K points
 * and returns the first min(K, k): RANDOM 3 asked for 7 returns 3; RANDOM
 * 5 asked for 2 the first 2 of RANDOM 5; LHS 8 asked for 4 the first 4 of
 * the LHS of 8 (not an LHS of 4); SOBOL 2 asked for 5 returns 2. k = 0
 * sets no cap.
 * Before the fix: k replaced K (RANDOM 3 asked for 7 drew 7 points, LHS
 * 8 asked for 4 drew an LHS of 4, SOBOL 2 asked for 5 drew 5). */
TEST_CASE("InitStrategy: K_TOTAL caps a lone RANDOM, LHS or SOBOL at its K",
          "[HPO][init_strategy][k_total]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;
    InitContext ic{sp, *st.trg, 99};
    const auto lone = [&](const std::string& method, const std::string& k) {
        InitStrategySpec ispec;
        ispec.chain.push_back({method, {k}});
        return build_init_strategy(ispec);
    };
    CHECK(lone("RANDOM", "3")->generate(ic, 7).size() == 3u);
    CHECK(lone("SOBOL", "2")->generate(ic, 5).size() == 2u);
    for (const std::string method : {"RANDOM", "LHS", "SOBOL"}) {
        INFO("METHOD " << method << " 8");
        const auto all = lone(method, "8")->generate(ic, 0);
        REQUIRE(all.size() == 8u);
        const auto capped = lone(method, "8")->generate(ic, 4);
        REQUIRE(capped.size() == 4u);
        for (std::size_t i = 0; i < 4; ++i) CHECK(capped[i] == all[i]);
    }
}

TEST_CASE("WarmInitStrategy: missing file throws regardless of policy",
          "[HPO][init_strategy][warm]")
{
    /* WARM_ON_MISMATCH governs seed-CONTENT mismatches only; a missing
     * seed file is a broken launch and is always fatal (falling through
     * would silently start the optimiser from the box centre). The policy
     * is the one of the block whose chain holds the WARM entry. */
    using namespace tadah::hpo;
    for (const char* policy : {"warn", "skip"}) {
        auto sp = make_synthetic_spec({0.0}, {1.0});
        StubTargets st;

        InitStrategySpec ispec;
        ispec.chain.push_back({"WARM", {"/nonexistent_path/never_here.tadah"}});
        ispec.warm_on_mismatch = policy;

        auto strat = build_init_strategy(ispec);
        InitContext ic{sp, *st.trg, 0};
        REQUIRE_THROWS_WITH(strat->generate(ic, 1),
                            Catch::Contains("cannot open seed file"));
    }
}

TEST_CASE("WarmInitStrategy: missing file with policy=error throws",
          "[HPO][init_strategy][warm]")
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0}, {1.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"WARM", {"/nonexistent_path/never_here.tadah"}});
    ispec.warm_on_mismatch = "error";

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    CHECK_THROWS(strat->generate(ic, 1));
}

TEST_CASE("WarmInitStrategy: parser keyword acceptance",
          "[HPO][init_strategy][warm]")
{
    /* Round-trip the parser path so a typo'd file path doesn't
     * silently break the WARM keyword pipeline. */
    StrategyTempFile cfg(R"(OPTIMIZER
INIT
  STRATEGY WARM /tmp/no_such.tadah
  WARM_ON_MISMATCH skip
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    /* A missing seed file is unconditionally fatal (policy governs
     * content mismatches only). The SPECIFIC error message proves the
     * parser accepted both keywords and the path reached the WARM
     * strategy -- which is what this round-trip is for. */
    REQUIRE_THROWS_WITH(tadah::hpo::HPOTargets(ctx, cfg.path.string()),
                        Catch::Contains("cannot open seed file"));
}

TEST_CASE("WarmInitStrategy: HOT round-trip primes warmstart cache",
          "[HPO][init_strategy][warm][hot]")
{
    /* Writer dumps `# HOT <script> <name> <value>` lines into
     * best_pot.tadah; WARM reader must scan those and prime
     * warmstart::cache so the restarted run reproduces the original
     * loss instead of starting each --hotvar relaxation from its
     * declared `initial`. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0}, {1.0});
    StubTargets st;

    const std::string script = "test_hot_round_trip.in";
    const std::string name   = "hot_a0";
    const double      v_in   = 3.4567;

    StrategyTempFile pot(
        "# HPO_BEST_FORMAT 1\n"
        "# HPO_BEST_LOSS 1.25e-3\n"
        "# HPO_SOURCE_RANK 0\n"
        "# HPO_SOURCE_SLOT 2\n"
        "# HPO_SOURCE_EVAL 17\n"
        "# HOT " + script + " " + name + " " + std::to_string(v_in) + "\n"
        "# unrelated comment\n"
    );

    /* Sanity: cache is empty for our key before the WARM read. */
    warmstart::clear();
    REQUIRE(!warmstart::best_get(script, name).has_value());

    InitStrategySpec ispec;
    ispec.chain.push_back({"WARM", {pot.path.string()}});

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    (void) strat->generate(ic, 1);

    /* Cache must now hold the value the file declared, to within the
     * 17-digit precision the writer uses. */
    auto cached = warmstart::best_get(script, name);
    REQUIRE(cached.has_value());
    CHECK(*cached == Approx(v_in).margin(1e-12));

    /* Cleanup so subsequent tests don't see this stale entry. */
    warmstart::clear();
}

TEST_CASE("WarmInitStrategy / CONFIG: a subnormal OPTIM value is a number "
          "(no ordinal shift)", "[HPO][init_strategy][warm][lossless]")
{
    /* The lossless writers keep a subnormal value ("1e-310"; the old
     * fixed-decimal Context text flushed it to 0). core::is_number and
     * std::stod used to reject it (ERANGE underflow): WARM skipped the token
     * when counting numeric ordinals, so every later OPTIM'd dimension of
     * the key silently took its neighbour's value, and CONFIG threw. */
    using namespace tadah::hpo;
    StrategyTempFile target(
        "OPTIM CGRID2B (1-3) 0 10\n"
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n");
    tadah::core::Context ctx;
    ctx.add("CGRID2B", std::vector<double>{1.0, 1e-310, 3.0});
    HPOTargets trg(ctx, target.path.string());
    REQUIRE(trg.spec.low.size() == 3);

    SECTION("CONFIG seed") {
        InitStrategySpec ispec;
        ispec.chain.push_back({"CONFIG", {}});
        auto strat = build_init_strategy(ispec);
        InitContext ic{trg.spec, trg, 0};
        const auto pts = strat->generate(ic, 1);
        REQUIRE(pts.size() == 1);
        REQUIRE(pts[0].size() == 3);
        CHECK(pts[0][0] == 1.0);
        CHECK(pts[0][1] == 1e-310);
        CHECK(pts[0][2] == 3.0);
    }
    SECTION("WARM seed with a subnormal middle token") {
        StrategyTempFile pot("CGRID2B 1.5 9.99999999999996945e-311 2.5\n");
        InitStrategySpec ispec;
        ispec.chain.push_back({"WARM", {pot.path.string()}});
        auto strat = build_init_strategy(ispec);
        InitContext ic{trg.spec, trg, 0};
        const auto pts = strat->generate(ic, 1);
        REQUIRE(pts.size() == 1);
        REQUIRE(pts[0].size() == 3);
        CHECK(pts[0][0] == 1.5);
        CHECK(pts[0][1] == 1e-310);
        CHECK(pts[0][2] == 2.5);    /* its own value, not the fallback 3.0 */
    }
}

TEST_CASE("WARM_ON_MISMATCH applies to the WARM entries of its own block: "
          "a RESTART block's to the RESTART chain, an INIT block's to the "
          "INIT chain", "[HPO][init_strategy][warm][restart]")
{
    /* Pins: a RESTART block's WARM_ON_MISMATCH governs the WARM seed of
     * the RESTART chain, which HPOTargets resolves at startup: with
     * `error` a seed outside the OPTIM box is a startup error naming the
     * clamping, with `skip` the seed is dropped (no RESTART seed point).
     * The INIT block's policy does not reach the RESTART chain: under an
     * INIT block with `WARM_ON_MISMATCH error`, a RESTART block without
     * the key keeps the default `warn`, so its seed is clamped into the
     * box and kept. An INIT block's WARM follows the INIT block's policy.
     * Before the fix: every WARM entry took the OUTER INIT block's policy
     * (ctx.spec.init_strategy_spec.warm_on_mismatch), whichever block held
     * it, so the RESTART block's `error` and `skip` were ignored (the seed
     * was clamped with a warning: no throw, one RESTART seed point) and the
     * INIT block's `error` made the RESTART seed a startup error. The last
     * SECTION passes as well (a pin). */
    StrategyTempFile pot("CGRID2B 1.5 20 2.5\n");   // 20 lies above the box's 10
    tadah::core::Context ctx;
    ctx.add("CGRID2B", std::vector<double>{1.0, 2.0, 3.0});
    const std::string warm = "METHOD WARM " + pot.path.string() + "\n";
    const auto target = [](const std::string& init,
                           const std::string& restart) {
        return "OPTIM CGRID2B (1-3) 0 10\nOPTIMIZER\n" + init +
               "STRATEGY\n  TYPE PLAIN\n  LIB DLIB\n  ALGO BFGS\n"
               "  MAXEVAL 5\n  RESTART\n    AFTER_STAGNATION 2\n    MAX 1\n" +
               restart + "  ENDRESTART\nENDSTRATEGY\nENDOPTIMIZER\n";
    };
    std::unique_ptr<tadah::hpo::HPOTargets> trg;
    SECTION("RESTART WARM_ON_MISMATCH error: the clamped seed is a startup error") {
        StrategyTempFile cfg(target("", "    " + warm +
                                            "    WARM_ON_MISMATCH error\n"));
        REQUIRE_THROWS_WITH(
            trg = std::make_unique<tadah::hpo::HPOTargets>(ctx, cfg.path.string()),
            Catch::Contains("clamped into OPTIM bounds"));
    }
    SECTION("RESTART WARM_ON_MISMATCH skip: the clamped seed is dropped") {
        StrategyTempFile cfg(target("", "    " + warm +
                                            "    WARM_ON_MISMATCH skip\n"));
        REQUIRE_NOTHROW(trg = std::make_unique<tadah::hpo::HPOTargets>(
                            ctx, cfg.path.string()));
        CHECK(trg->spec.restart.seed_points.empty());
    }
    SECTION("an INIT block's WARM_ON_MISMATCH error does not reach the "
            "RESTART chain") {
        StrategyTempFile cfg(target(
            "INIT\n  METHOD CONFIG\n  WARM_ON_MISMATCH error\nENDINIT\n",
            "    " + warm));
        REQUIRE_NOTHROW(trg = std::make_unique<tadah::hpo::HPOTargets>(
                            ctx, cfg.path.string()));
        REQUIRE(trg->spec.restart.seed_points.size() == 1u);
        REQUIRE(trg->spec.restart.seed_points[0].size() == 3u);
        CHECK(trg->spec.restart.seed_points[0][1] == 10.0);   // clamped (warn)
    }
    SECTION("an INIT block's WARM follows the INIT block's WARM_ON_MISMATCH") {
        StrategyTempFile cfg(target(
            "INIT\n  " + warm + "  WARM_ON_MISMATCH error\nENDINIT\n", ""));
        REQUIRE_THROWS_WITH(
            trg = std::make_unique<tadah::hpo::HPOTargets>(ctx, cfg.path.string()),
            Catch::Contains("clamped into OPTIM bounds"));
    }
}

TEST_CASE("WarmInitStrategy: legacy HOT_A0 line still primes cache",
          "[HPO][init_strategy][warm][hot]")
{
    /* Old best_pot.tadah files written before the rename emit
     * `# HOT_A0 <script> <a0>`. The WARM parser keeps a back-compat
     * arm that maps these to (script, "hot_a0") in the new cache. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0}, {1.0});
    StubTargets st;

    const std::string script = "test_hot_a0_legacy.in";
    const double      a0_in  = 3.30;

    StrategyTempFile pot(
        "# HOT_A0 " + script + " " + std::to_string(a0_in) + "\n"
    );

    InitStrategySpec ispec;
    ispec.chain.push_back({"WARM", {pot.path.string()}});

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    (void) strat->generate(ic, 1);

    auto cached = warmstart::best_get(script, "hot_a0");
    REQUIRE(cached.has_value());
    CHECK(*cached == Approx(a0_in).margin(1e-12));

    warmstart::clear();
}

namespace {

/* The journal line holding `needle` ("" when absent). */
std::string journal_line(const std::string& txt, const std::string& needle)
{
    const auto at = txt.find(needle);
    if (at == std::string::npos) return "";
    const auto b = txt.rfind('\n', at);
    const auto e = txt.find('\n', at);
    return txt.substr(b == std::string::npos ? 0 : b + 1,
                      e == std::string::npos ? std::string::npos
                                             : e - (b == std::string::npos ? 0 : b + 1));
}

/* Run WARM on a seed file against a run context declaring `run_embf0`
 * (nullptr: key absent) and return the journal. */
std::string warm_embf0_journal(const std::string& seed_body,
                               const char* run_embf0,
                               bool pass_run_context = true,
                               bool run_has_ream = true)
{
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0}, {1.0});
    StubTargets st;   // no OPTIM line: the WARM point is zero-dimensional
    StrategyTempFile pot(seed_body);
    tadah::core::Context run;
    if (run_has_ream) {
        run.add("INITMB", "true");
        for (const char* t : {"DM_REAM", "0", "2", "2", "2", "3", "3", "0",
                              "0", "Ta", "Ta"})
            run.add("TYPEMB", t);
        run.add("RHOFORM", "SIGNED");
    }
    if (run_embf0) run.add("EMBF0", run_embf0);

    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("closed");
    const auto tmp_log = std::filesystem::temp_directory_path() /
        ("hpo_warm_embf0_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(tmp_log.string(), "test", "/tmp", 1);

    InitStrategySpec ispec;
    ispec.chain.push_back({"WARM", {pot.path.string()}});
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0, pass_run_context ? &run : nullptr};
    (void) strat->generate(ic, 1);
    log.finish("TEST");

    std::ifstream lf(tmp_log);
    std::stringstream ss; ss << lf.rdbuf();
    std::error_code ec; std::filesystem::remove(tmp_log, ec);
    /* WARM reads only the OPTIM dims: nothing inserted into the run context. */
    CHECK(run.has_stored("EMBF0") == (run_embf0 != nullptr));
    return ss.str();
}

const std::string kReamSeed =
    "INITMB true\nTYPEMB DM_REAM 0 2 2 2 3 3 0 0 Ta Ta\nRHOFORM SIGNED\n";

}  // namespace

TEST_CASE("WarmInitStrategy: EMBF0 of the seed vs the run's -c",
          "[HPO][init_strategy][warm][embf0]")
{
    SECTION("seed FREE, run ZERO (key absent): WARNING") {
        const std::string txt = warm_embf0_journal(kReamSeed + "EMBF0 FREE\n",
                                                   nullptr);
        const std::string ln = journal_line(txt, "declares EMBF0 FREE");
        CHECK(ln.find("WARNING") != std::string::npos);
        CHECK(ln.find("; this run evaluates EMBF0 ZERO (from -c). WARM takes "
                      "only the OPTIM values from the seed and evaluates them "
                      "unchanged in the run's reference (no conversion)")
              != std::string::npos);
    }
    SECTION("seed ZERO, run FREE: WARNING") {
        const std::string txt = warm_embf0_journal(kReamSeed + "EMBF0 ZERO\n",
                                                   "FREE");
        const std::string ln = journal_line(txt, "declares EMBF0 ZERO");
        CHECK(ln.find("WARNING") != std::string::npos);
        CHECK(ln.find("this run evaluates EMBF0 FREE (from -c)")
              != std::string::npos);
    }
    SECTION("seed without the key, run ZERO: INFO, read as ZERO") {
        const std::string txt = warm_embf0_journal(kReamSeed, "ZERO");
        const std::string ln = journal_line(txt, "does not declare EMBF0");
        CHECK(ln.find("INFO") != std::string::npos);
        CHECK(ln.find("read as EMBF0 ZERO") != std::string::npos);
        CHECK(txt.find("no conversion") == std::string::npos);
    }
    SECTION("seed without the key, run FREE: WARNING") {
        const std::string txt = warm_embf0_journal(kReamSeed, "FREE");
        const std::string ln = journal_line(txt, "does not declare EMBF0");
        CHECK(ln.find("WARNING") != std::string::npos);
        CHECK(ln.find("(read as EMBF0 ZERO); this run evaluates EMBF0 FREE")
              != std::string::npos);
    }
    SECTION("matching modes, no DM_REAM seed, or no run context: silent") {
        CHECK(warm_embf0_journal(kReamSeed + "EMBF0 FREE\n", "FREE")
                  .find("EMBF0") == std::string::npos);
        CHECK(warm_embf0_journal("INITMB false\nEMBF0 FREE\n", nullptr)
                  .find("EMBF0") == std::string::npos);
        CHECK(warm_embf0_journal(kReamSeed + "EMBF0 FREE\n", nullptr,
                                 /*pass_run_context=*/false)
                  .find("EMBF0") == std::string::npos);
        /* the run itself has no DM_REAM block: EMBF0 does not apply */
        CHECK(warm_embf0_journal(kReamSeed + "EMBF0 FREE\n", nullptr,
                                 /*pass_run_context=*/true,
                                 /*run_has_ream=*/false)
                  .find("EMBF0") == std::string::npos);
    }
}

TEST_CASE("HPO parses MULTISTART knobs",
          "[HPO][run_policy][multistart]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE MULTISTART
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
  N_STARTS 4
  BUDGET_PER_START 12
ENDSTRATEGY
ENDOPTIMIZER
)");
    /* PARALLEL is refused ("startup validation: STRATEGY PARALLEL is
     * refused at its line, naming THREADS"); this case read PARALLEL 2
     * as multistart.parallel == 2. */
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    const auto& cfg_run = trg.spec;
    CHECK(cfg_run.multistart.n_starts == 4);
    REQUIRE(cfg_run.multistart.budget_per_start.has_value());
    CHECK(*cfg_run.multistart.budget_per_start == 12);
}

TEST_CASE("HPO MULTISTART rejects zero",
          "[HPO][run_policy][multistart]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE MULTISTART
  LIB DLIB
  ALGO BFGS
  N_STARTS 0
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, cfg.path.string()));
}

TEST_CASE("MultiStartPolicy: validate rejects the NLopt global GN_DIRECT",
          "[HPO][run_policy][multistart]")
{
    using namespace tadah::hpo;
    OptimizerSpec sp;
    sp.lib = "NLOPT";
    sp.algo = "GN_DIRECT";
    CHECK_THROWS(MultiStartPolicy<dlib::matrix<double,0,1>, int>::validate(sp));
}

TEST_CASE("MultiStartPolicy: validate accepts local NLopt and DLIB",
          "[HPO][run_policy][multistart]")
{
    using namespace tadah::hpo;
    using V = dlib::matrix<double,0,1>;
    for (auto algo : {"LN_BOBYQA", "LD_LBFGS", "LN_SBPLX"}) {
        OptimizerSpec sp;
        sp.lib = "NLOPT";
        sp.algo = algo;
        CHECK_NOTHROW(MultiStartPolicy<V, int>::validate(sp));
    }
    OptimizerSpec dlib_bfgs;
    dlib_bfgs.lib = "DLIB";
    dlib_bfgs.algo = "BFGS";
    CHECK_NOTHROW(MultiStartPolicy<V, int>::validate(dlib_bfgs));
}

TEST_CASE("HPO parses PRESTAGE block and TOPK",
          "[HPO][run_policy][prestage]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
  TOPK 3
  EXPLORE
    LIB DLIB
    ALGO GFS
    MAXEVAL 80
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    const auto& cfg_run = trg.spec;
    CHECK(cfg_run.stage_topk == 3);
    REQUIRE(trg.spec.prestage);
    CHECK(trg.spec.prestage->algo == "GFS");
    /* Outer remains the refinement and is unchanged. */
    CHECK(trg.spec.algo == "BFGS");
    /* INNER is independent — only for NLopt MLSL/AUGLAG. */
    CHECK_FALSE(trg.spec.inner);
}

TEST_CASE("HPO TOPK rejects zero with PRESTAGE",
          "[HPO][run_policy][prestage]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  TOPK 0
  EXPLORE
    LIB DLIB
    ALGO GFS
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, cfg.path.string()));
}

TEST_CASE("HPO STAGE_GLOBAL keyword now errors with helpful message",
          "[HPO][run_policy][prestage]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  STAGE_GLOBAL true
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    bool caught = false;
    std::string msg;
    try {
        tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    } catch (const std::exception& e) {
        caught = true;
        msg = e.what();
    }
    CHECK(caught);
    CHECK(msg.find("PRESTAGE") != std::string::npos);
}

TEST_CASE("PRESTAGE supports a nested INNER sub-algo (MLSL pattern)",
          "[HPO][run_policy][prestage]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 80
  TOPK 4
  EXPLORE
    LIB NLOPT
    ALGO G_MLSL_LDS
    MAXEVAL 600
    INNER
      LIB NLOPT
      ALGO LN_BOBYQA
    ENDINNER
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.spec.prestage);
    REQUIRE(trg.spec.prestage->inner);
    CHECK(trg.spec.prestage->lib  == "NLOPT");
    CHECK(trg.spec.prestage->algo == "G_MLSL_LDS");
    CHECK(trg.spec.prestage->inner->lib  == "NLOPT");
    CHECK(trg.spec.prestage->inner->algo == "LN_BOBYQA");
    /* outer spec has no INNER of its own */
    CHECK_FALSE(trg.spec.inner);
    /* Pins: an INNER block takes no INIT block (the grammar refuses one;
     * MLSL picks the INNER search's starting points), so its spec holds no
     * starting points and the journal prints no INIT strategy for it,
     * while the exploration shares the OUTER's CONFIG point.
     * Before the fix: the INNER spec held a copy of the OUTER's CONFIG
     * point and its origin (share_init_with_nested_phases), which nothing
     * used, and the run policy printed "PRESTAGE.EXPLORE.INNER init
     * strategy: (default -- CONFIG only)". */
    CHECK(trg.spec.prestage->optim_init.size() == 1);
    CHECK(trg.spec.prestage->inner->optim_init.empty());
    CHECK(trg.spec.prestage->inner->optim_init_origin.empty());
    const std::string journal = optimizer_section(trg.spec);
    INFO(journal);
    CHECK(journal.find("PRESTAGE.EXPLORE init strategy:") != std::string::npos);
    CHECK(journal.find("PRESTAGE.EXPLORE.INNER") == std::string::npos);
}

TEST_CASE("PRESTAGE EXPLORE block parses INIT block and bounds independently",
          "[HPO][run_policy][prestage]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 80
  TOPK 4
  EXPLORE
    LIB DLIB
    ALGO GFS
    MAXEVAL 600
    INIT
      STRATEGY SOBOL 32
      SEED 1717
    ENDINIT
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.spec.prestage);
    const auto& ps = *trg.spec.prestage;
    CHECK(ps.lib == "DLIB");
    CHECK(ps.algo == "GFS");
    CHECK(ps.stop.maxeval.value_or(0) == 600);
    REQUIRE(!ps.init_strategy_spec.chain.empty());
    CHECK(ps.init_strategy_spec.chain.front().method == "SOBOL");
    CHECK(ps.init_strategy_spec.seed.value_or(0) == 1717ULL);
    /* Resolution regression: the parsed PRESTAGE chain must be resolved
     * into PRESTAGE's own optim_init by the constructor — not silently
     * left empty (and certainly not overridden by an inherited copy of
     * OUTER's optim_init). Pre-fix, prestage.optim_init had size 1 (the
     * inherited OUTER CONFIG point); post-fix it carries the SOBOL 32
     * points the user actually requested. */
    CHECK(ps.optim_init.size() == 32);
}

TEST_CASE("PRESTAGE INIT_STRATEGY RANDOM is resolved on the prestage spec",
          "[HPO][run_policy][prestage][init_strategy]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 80
  TOPK 4
  EXPLORE
    LIB NLOPT
    ALGO GN_CRS2_LM
    MAXEVAL 100
    INIT
      STRATEGY RANDOM 4
      SEED 1718
    ENDINIT
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.spec.prestage);
    /* Without the fix, PRESTAGE's optim_init was inherited from OUTER
     * (size 1, the CONFIG point). With the fix it carries 4 RANDOM
     * points produced by the prestage's own INIT chain. */
    CHECK(trg.spec.prestage->optim_init.size() == 4);
}

TEST_CASE("HPO parses NOISE.CALIBRATION",
          "[HPO][run_policy][noise_calibration]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION 5
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.calibration_n == 5);
}

TEST_CASE("HPO NOISE.CALIBRATION default is 0 (disabled)",
          "[HPO][run_policy][noise_calibration]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.calibration_n == 0);
}

TEST_CASE("HPO NOISE.CALIBRATION rejects N=1 (no spread possible)",
          "[HPO][run_policy][noise_calibration]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION 1
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, cfg.path.string()));
}

TEST_CASE("HPO NOISE.CALIBRATION rejects non-integer arg",
          "[HPO][run_policy][noise_calibration]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION abc
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, cfg.path.string()));
}

TEST_CASE("HPO NOISE.CALIBRATION accepts N=0 (explicit off)",
          "[HPO][run_policy][noise_calibration]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  CALIBRATION 0
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.calibration_n == 0);
}

TEST_CASE("PRESTAGE EXPLORE without explicit INIT inherits OUTER's optim_init",
          "[HPO][run_policy][prestage][init_strategy]")
{
    /* Legacy behaviour: a PRESTAGE EXPLORE block with no INIT chain of
     * its own should inherit OUTER's resolved optim_init — the implicit
     * CONFIG point flows in. The fix must not regress this. */
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 80
  TOPK 4
  EXPLORE
    LIB DLIB
    ALGO GFS
    MAXEVAL 600
  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.spec.prestage);
    /* Outer's CONFIG produces 1 zero-dim point; PRESTAGE inherits it. */
    REQUIRE(trg.spec.optim_init.size() == 1);
    CHECK(trg.spec.prestage->optim_init.size() == 1);
}

TEST_CASE("HPO parses RESTART knobs",
          "[HPO][run_policy][restart]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 200
  RESTART
    AFTER_STAGNATION 25
    MAX 4
    STRATEGY RANDOM 1
  ENDRESTART
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    const auto& cfg_run = trg.spec;
    CHECK(cfg_run.restart.after_stagnation == 25);
    CHECK(cfg_run.restart.max == 4);
    REQUIRE(cfg_run.restart.init_strategy.chain.size() == 1);
    CHECK(cfg_run.restart.init_strategy.chain.front().method == "RANDOM");
}

TEST_CASE("HPO RESTART defaults: zero window means no restart",
          "[HPO][run_policy][restart]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    const auto& cfg_run = trg.spec;
    CHECK(cfg_run.restart.after_stagnation == 0);
    CHECK(cfg_run.restart.max == 0);
    CHECK(cfg_run.restart.init_strategy.chain.empty());
}

TEST_CASE("HPO parses NOISE.AUTO boolean",
          "[HPO][run_policy][noise_auto]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  AUTO true
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.autodetect == true);
}

TEST_CASE("NOISE.AUTO bareword form parses to true",
          "[HPO][run_policy][noise_auto]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  AUTO
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.autodetect == true);
}

TEST_CASE("NOISE.AUTO false leaves fd_step untouched",
          "[HPO][run_policy][noise_auto]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
NOISE
  AUTO false
ENDNOISE
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK(trg.spec.noise.autodetect == false);
    CHECK(trg.spec.param.count("fd_step") == 0);
}

TEST_CASE("HPO BUDGET_PER_START rejects zero",
          "[HPO][run_policy][multistart]")
{
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE MULTISTART
  LIB DLIB
  ALGO BFGS
  BUDGET_PER_START 0
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    CHECK_THROWS(tadah::hpo::HPOTargets(ctx, cfg.path.string()));
}

TEST_CASE("InitStrategy: chain produces exactly the listed strategies",
          "[HPO][init_strategy][chain]")
{
    /* There is no implicit CONFIG. A chain of `RANDOM 3` produces
     * exactly 3 points -- the in-config point participates only when the
     * user lists `STRATEGY CONFIG` explicitly. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;

    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"3"}});
    ispec.seed = 7;
    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, *ispec.seed};
    auto pts = strat->generate(ic, 0);
    CHECK(pts.size() == 3);
}

TEST_CASE("InitStrategy: chain order is preserved (WARM not reordered)",
          "[HPO][init_strategy][warm][chain]")
{
    /* The user wrote SOBOL first, WARM second. build_init_strategy()
     * must keep that order -- WARM is no longer promoted to the chain
     * front. SOBOL contributes 4 two-dim points, then WARM (comment-only
     * file, empty optimap) contributes one zero-dim point at the end.
     * If WARM were still reordered, pts.front() would be the zero-dim
     * WARM point instead of a SOBOL point. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;

    StrategyTempFile pot("# comment-only warm file\n");

    InitStrategySpec ispec;
    ispec.chain.push_back({"SOBOL", {"4"}});
    ispec.chain.push_back({"WARM",  {pot.path.string()}});
    ispec.warm_on_mismatch = "warn";

    auto strat = build_init_strategy(ispec);
    InitContext ic{sp, *st.trg, 0};
    auto pts = strat->generate(ic, 0);
    REQUIRE(pts.size() == 5);     /* 4 SOBOL + 1 WARM */
    CHECK(pts.front().size() == 2);   /* SOBOL stayed first */
    CHECK(pts.back().empty());        /* WARM stayed last  */
}

TEST_CASE("HPO log file records init/run-policy summary",
          "[HPO][hpo_log][run_policy]")
{
    /* Smoke-test the log writer: it must at least open the file, emit
     * an OPTIMISER section, and include the new INIT_STRATEGY/run-policy
     * sub-sections introduced by the diagnostics patch. We do not run
     * a full optimisation here — only check the writer output. */
    using namespace tadah::hpo;
    /* K_TOTAL 3 caps the 4 SOBOL points (K_TOTAL 4, which this case gave
     * until a K_TOTAL that caps nothing was refused, is refused). */
    StrategyTempFile cfg(R"(OPTIMIZER
INIT
  STRATEGY SOBOL 4
  K_TOTAL 3
ENDINIT
NOISE
  AUTO
ENDNOISE
STRATEGY
  TYPE MULTISTART
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
  N_STARTS 4
  BUDGET_PER_START 12
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    HPOTargets trg(ctx, cfg.path.string());

    auto& log = HPOLog::instance();
    auto tmp_log = std::filesystem::temp_directory_path() /
        ("hpo_log_test_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
    log.open(tmp_log.string(), "test", "/tmp", 1);
    log.write_optimizer_section(trg.spec, {"p0","p1","p2","p3"});
    log.finish("TEST");

    std::ifstream lf(tmp_log);
    REQUIRE(lf.good());
    std::stringstream ss; ss << lf.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("Init strategy:") != std::string::npos);
    CHECK(body.find("SOBOL 4") != std::string::npos);
    /* The INIT block's K_TOTAL under its name (it read "INIT_K", the
     * retired key the parser refuses). */
    CHECK(journal_block(body, "Init strategy:").find("K_TOTAL           : 3") !=
          std::string::npos);
    CHECK(body.find("INIT_K") == std::string::npos);
    CHECK(body.find("Run policy:") != std::string::npos);
    CHECK(body.find("MULTISTART") != std::string::npos);
    CHECK(body.find("BUDGET_PER_START") != std::string::npos);
    CHECK(body.find("NOISE.AUTO") != std::string::npos);

    std::error_code ec; std::filesystem::remove(tmp_log, ec);
}

TEST_CASE("The journal's run policy lists the RESTART block's METHOD chain, "
          "K_TOTAL, SEED and WARM_ON_MISMATCH",
          "[HPO][hpo_log][run_policy]")
{
    /* Pins: under "RESTART init strategy:" the run policy prints the
     * RESTART block as the INIT block is printed: its METHOD chain, or
     * "(default -- CONFIG only)" without one (a restart then starts from
     * the CONFIG point), its K_TOTAL, SEED and WARM_ON_MISMATCH; and an
     * INIT block's K_TOTAL and SEED appear under those names.
     * Before the fix: the RESTART block printed only its METHOD chain, and
     * no "RESTART init strategy:" line at all without one, so SEED 77,
     * K_TOTAL 2 and WARM_ON_MISMATCH error appeared nowhere; the INIT
     * block's SEED read "INIT_SEED : 9" (the retired key name).
     * A block without a METHOD line gives one point, which a K_TOTAL caps
     * nothing of: that section has no K_TOTAL line (it had K_TOTAL 2 until
     * such a K_TOTAL was refused) and the journal none. */
    std::string method, k_total;
    SECTION("with a METHOD chain") {
        method = "    METHOD LHS 3\n";
        k_total = "    K_TOTAL 2\n";
    }
    SECTION("without a METHOD line") {}
    StrategyTempFile cfg(R"(OPTIMIZER
INIT
  METHOD RANDOM 1
  SEED 9
ENDINIT
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
  RESTART
    AFTER_STAGNATION 3
    MAX 1
)" + method + k_total + R"(    SEED 77
    WARM_ON_MISMATCH error
  ENDRESTART
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    const std::string journal = optimizer_section(trg.spec);
    INFO(journal);
    const std::string restart = journal_block(journal, "RESTART init strategy:");
    REQUIRE_FALSE(restart.empty());
    if (method.empty())
        CHECK(restart.find("(default -- CONFIG only)") != std::string::npos);
    else
        CHECK(restart.find("[1] LHS 3") != std::string::npos);
    if (method.empty())
        CHECK(restart.find("K_TOTAL") == std::string::npos);
    else
        CHECK(restart.find("K_TOTAL           : 2") != std::string::npos);
    CHECK(restart.find("SEED              : 77") != std::string::npos);
    CHECK(restart.find("WARM_ON_MISMATCH  : error") != std::string::npos);
    const std::string init = journal_block(journal, "Init strategy:");
    CHECK(init.find("SEED              : 9") != std::string::npos);
    CHECK(journal.find("INIT_SEED") == std::string::npos);
    CHECK(journal.find("INIT_K") == std::string::npos);
}

TEST_CASE("The journal says where the starting points come from and that "
          "an EXPLORE without INIT shares the OUTER's",
          "[HPO][hpo_log][run_policy][prestage]")
{
    /* Pins: "Starting point:" counts the points of the INIT chain by
     * origin, and an EXPLORE block without an INIT block of its own is
     * shown starting from the OUTER INIT points
     * (share_init_with_nested_phases), while one with its own chain shows
     * that chain.
     * Before the fix: "Starting point: user-supplied (3 seeds)" for three
     * LHS points, and "PRESTAGE.EXPLORE init strategy: (default -- CONFIG
     * only)" for an exploration that starts from them. The second
     * SECTION's check of the EXPLORE's own chain passes as well (a pin). */
    std::string explore_init;
    bool own = false;
    SECTION("an EXPLORE without INIT") {}
    SECTION("an EXPLORE with its own INIT") {
        own = true;
        explore_init = "    INIT\n      METHOD SOBOL 4\n      SEED 3\n    ENDINIT\n";
    }
    StrategyTempFile cfg(R"(OPTIMIZER
INIT
  METHOD LHS 3
  SEED 9
ENDINIT
STRATEGY
  TYPE PRESTAGE
  LIB DLIB
  ALGO BFGS
  MAXEVAL 80
  TOPK 2
  EXPLORE
    LIB TADAH
    ALGO RANDOM
    MAXEVAL 10
    SEED 5
)" + explore_init + R"(  ENDEXPLORE
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.spec.optim_init.size() == 3);
    const std::string journal = optimizer_section(trg.spec);
    INFO(journal);
    CHECK(journal.find("Starting point: 3 points from the INIT chain (3 LHS)") !=
          std::string::npos);
    CHECK(journal.find("user-supplied") == std::string::npos);
    const std::string explore =
        journal_block(journal, "PRESTAGE.EXPLORE init strategy:");
    REQUIRE_FALSE(explore.empty());
    if (own) {
        CHECK(explore.find("[1] SOBOL 4") != std::string::npos);
        CHECK(explore.find("OUTER") == std::string::npos);
    } else {
        CHECK(explore.find("(the OUTER INIT points, shared)") !=
              std::string::npos);
        CHECK(explore.find("(default -- CONFIG only)") == std::string::npos);
    }
}

TEST_CASE("The journal says when there is no starting point and counts "
          "points without a recorded origin", "[HPO][hpo_log][run_policy]")
{
    /* Pins: a spec without starting points (an INIT chain that produced
     * none, e.g. WARM_ON_MISMATCH skip) reads "Starting point: none", the
     * optimiser starting from its own default; points without recorded
     * origins (a spec built outside HPOTargets) are only counted.
     * Before the fix: "Starting point: random (uniform within bounds)",
     * which not every optimiser does (the ANNEAL family starts at the box
     * centre), and "user-supplied (2 seeds)". */
    StrategyTempFile cfg(R"(OPTIMIZER
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
ENDOPTIMIZER
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    tadah::hpo::OptimizerSpec sp(trg.spec);
    sp.optim_init_origin.clear();
    SECTION("no starting point") {
        sp.optim_init.clear();
        const std::string journal = optimizer_section(sp);
        INFO(journal);
        CHECK(journal.find("Starting point: none (the INIT chain produced no "
                           "point; the optimiser starts from its own "
                           "default)") != std::string::npos);
        CHECK(journal.find("uniform within bounds") == std::string::npos);
    }
    SECTION("points without a recorded origin") {
        sp.optim_init.assign(2, std::vector<double>{});
        const std::string journal = optimizer_section(sp);
        INFO(journal);
        CHECK(journal.find("Starting point: 2 points from the INIT chain\n") !=
              std::string::npos);
    }
}

TEST_CASE("HPOLog::set_status feeds finish() when status is empty",
          "[HPO][hpo_log][set_status]")
{
    /* Regression test for the MULTISTART + NLopt journal bug: each per-
     * start NLopt run records its termination reason via set_status(),
     * and the outer hpo_run() calls finish() once at the very end. The
     * journal trailer must reflect the most recent set_status value when
     * finish() is called with an empty (or default) argument. */
    using namespace tadah::hpo;
    auto& log = HPOLog::instance();
    auto tmp_log = std::filesystem::temp_directory_path() /
        ("hpo_setstatus_test_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");

    log.open(tmp_log.string(), "test", "/tmp", 1);
    log.set_status("PENULTIMATE");
    log.set_status("LAST_NLOPT_STATUS — most recent wins");
    log.finish();   /* default empty arg → falls back to last set_status */

    std::ifstream lf(tmp_log);
    REQUIRE(lf.good());
    std::stringstream ss; ss << lf.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("LAST_NLOPT_STATUS") != std::string::npos);
    CHECK(body.find("PENULTIMATE")       == std::string::npos);
    CHECK(body.find("COMPLETED")         == std::string::npos);

    std::error_code ec; std::filesystem::remove(tmp_log, ec);
}

TEST_CASE("HPOLog::finish explicit arg overrides set_status",
          "[HPO][hpo_log][set_status]")
{
    /* If hpo_run()'s catch block passes "FAILED -- ..." it must
     * override any status the inner driver recorded. */
    using namespace tadah::hpo;
    auto& log = HPOLog::instance();
    auto tmp_log = std::filesystem::temp_directory_path() /
        ("hpo_setstatus_test_b_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");

    log.open(tmp_log.string(), "test", "/tmp", 1);
    log.set_status("MAXEVAL_REACHED");
    log.finish("FAILED -- kaboom");

    std::ifstream lf(tmp_log);
    REQUIRE(lf.good());
    std::stringstream ss; ss << lf.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("FAILED -- kaboom") != std::string::npos);
    CHECK(body.find("MAXEVAL_REACHED")            == std::string::npos);

    std::error_code ec; std::filesystem::remove(tmp_log, ec);
}

/* A journal finished with no status at all says so; it does not invent
 * one. The run driver never gets here (a normal end without a status is an
 * internal error there, test_hpo_termination.cpp); a direct caller of
 * HPOLog can.
 * Pins: the trailer "Optimiser status : (no status recorded)", and no
 * COMPLETED.
 * Before the fix: HPOLog::finish wrote COMPLETED. */
TEST_CASE("HPOLog::finish writes '(no status recorded)' when no status was "
          "set", "[HPO][hpo_log][set_status]")
{
    using namespace tadah::hpo;
    auto& log = HPOLog::instance();
    log.set_status("");   /* clear any leftover state from prior tests */
    auto tmp_log = std::filesystem::temp_directory_path() /
        ("hpo_setstatus_test_c_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");

    log.open(tmp_log.string(), "test", "/tmp", 1);
    log.finish();   /* no set_status, no explicit status */

    std::ifstream lf(tmp_log);
    REQUIRE(lf.good());
    std::stringstream ss; ss << lf.rdbuf();
    const std::string body = ss.str();
    CHECK(body.find("Optimiser status        : (no status recorded)\n") !=
          std::string::npos);
    CHECK(body.find("COMPLETED") == std::string::npos);

    std::error_code ec; std::filesystem::remove(tmp_log, ec);
}

/* RunPolicy template instantiation tests need a real Runner satisfying
 * HPO_Host's API; we cover RunPolicy semantics end-to-end via the
 * integration matrix (Increment 8) instead. The parser tests above
 * already verify the knobs that drive build_run_policy(). */

namespace {

/* The error HPOTargets gives a target file of one search parameter
 * (CGRID2B, 1 value) whose OPTIMIZER block holds `block`, or "" when it
 * reads the file, with the number of starting points it resolved. */
struct CountRead {
    std::string error;
    std::size_t points = 0;
    std::size_t topk = 0;
    std::size_t calibration = 0;
};

CountRead read_counts(const std::string& block)
{
    StrategyTempFile cfg("OPTIM CGRID2B (1) 0 10\nOPTIMIZER\n" + block +
                         "ENDOPTIMIZER\n");
    tadah::core::Context ctx;
    ctx.add("CGRID2B", std::vector<double>{1.0});
    CountRead r;
    try {
        tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
        r.points = trg.spec.optim_init.size();
        r.topk = trg.spec.stage_topk;
        r.calibration = trg.spec.noise.calibration_n;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

const std::string kSbplx =
    "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nENDSTRATEGY\n";

std::string init_block(const std::string& lines)
{
    return "INIT\n" + lines + "ENDINIT\n" + kSbplx;
}

std::string prestage_block(const std::string& topk)
{
    return "STRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
           "TOPK " + topk + "\nEXPLORE\nLIB TADAH\nALGO RANDOM\nMAXEVAL 5\n"
           "ENDEXPLORE\nENDSTRATEGY\n";
}

std::string restart_block(const std::string& lines)
{
    return "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nRESTART\n"
           "AFTER_STAGNATION 2\nMAX 1\n" + lines + "ENDRESTART\nENDSTRATEGY\n";
}

}  // namespace

TEST_CASE("The starting points of a METHOD line, K_TOTAL and TOPK are at "
          "most 1000000, and NOISE CALIBRATION at most 2147483647 "
          "evaluations: a larger value is refused at its line",
          "[HPO][init_strategy][limits]")
{
    /* The points of an INIT or RESTART block are drawn when the target
     * file is read and held in memory, and each is a start of a search;
     * a PRESTAGE refinement starts from TOPK candidates. Before the fix
     * every count was read up to 18446744073709551615, and a large one
     * stopped tadah hpo with "vector::reserve" or "cannot create
     * std::vector larger than max_size()", and NOISE CALIBRATION when the
     * run started; the values below were read, and the K ones drew
     * 1000001 points. */
    struct Refused { std::string block, error; };
    const std::vector<Refused> refused = {
        {init_block("METHOD RANDOM 1000001\n"),
         "Error on line 4: INIT METHOD RANDOM K must be an integer from 1 "
         "to 1000000, got '1000001'\nLine: METHOD RANDOM 1000001"},
        {init_block("METHOD LHS 1000001\n"),
         "Error on line 4: INIT METHOD LHS K must be an integer from 1 to "
         "1000000, got '1000001'\nLine: METHOD LHS 1000001"},
        {init_block("METHOD SOBOL 1000001\n"),
         "Error on line 4: INIT METHOD SOBOL K must be an integer from 1 to "
         "1000000, got '1000001'\nLine: METHOD SOBOL 1000001"},
        /* K_TOTAL caps the points the METHOD lines give, so a K_TOTAL as
         * large as their sum caps nothing (it was refused above 1000000,
         * and replaced the K of a lone METHOD line). */
        {init_block("METHOD RANDOM 2\nK_TOTAL 1000001\n"),
         "Error on line 5: K_TOTAL 1000001 caps nothing: the METHOD lines of "
         "the INIT blocks inside OPTIMIZER, STRATEGY and REFINE give at most "
         "2 starting points (RANDOM 2), and K_TOTAL keeps the first K_TOTAL "
         "starting points\nLine: K_TOTAL 1000001"},
        {restart_block("METHOD RANDOM 1000001\n"),
         "Error on line 10: INIT METHOD RANDOM K must be an integer from 1 "
         "to 1000000, got '1000001'\nLine: METHOD RANDOM 1000001"},
        {restart_block("K_TOTAL 1000001\n"),
         "Error on line 10: K_TOTAL 1000001 caps nothing: a block without a "
         "METHOD line gives one starting point (the CONFIG point), and "
         "K_TOTAL keeps the first K_TOTAL starting points\nLine: K_TOTAL "
         "1000001"},
        {prestage_block("1000001"),
         "Error on line 8: TOPK must be an integer from 1 to 1000000, got "
         "'1000001'\nLine: TOPK 1000001"},
        {"NOISE\nCALIBRATION 2147483648\nENDNOISE\n" + kSbplx,
         "Error on line 4: CALIBRATION must be an integer from 0 to "
         "2147483647, got '2147483648'\nLine: CALIBRATION 2147483648"},
    };
    for (const auto& r : refused) {
        INFO(r.block);
        CHECK(read_counts(r.block).error == r.error);
    }
}

TEST_CASE("A count of starting points at its limit, 1000000, draws that many "
          "points, and the last SOBOL block lies within the sequence",
          "[HPO][init_strategy][limits]")
{
    /* SOBOL with seed 65535 draws block 65535, the last a seed selects:
     * points 65535 * 1000000 ... 65536 * 1000000 - 1 of the sequence.
     * K_TOTAL caps the points (a leg drew 1000000 points from METHOD RANDOM
     * 2 with K_TOTAL 1000000, which replaced its K): K_TOTAL 999999 keeps
     * all but one of RANDOM 1000000. */
    for (const std::string& lines :
         {std::string("METHOD RANDOM 1000000\n"),
          std::string("METHOD LHS 1000000\n"),
          std::string("METHOD SOBOL 1000000\nSEED 65535\n")}) {
        INFO(lines);
        const CountRead r = read_counts(init_block(lines));
        CHECK(r.error.empty());
        CHECK(r.points == 1000000u);
    }
    const CountRead capped =
        read_counts(init_block("METHOD RANDOM 1000000\nK_TOTAL 999999\n"));
    CHECK(capped.error.empty());
    CHECK(capped.points == 999999u);
    const CountRead topk = read_counts(prestage_block("1000000"));
    CHECK(topk.error.empty());
    CHECK(topk.topk == 1000000u);
    const CountRead cal =
        read_counts("NOISE\nCALIBRATION 2147483647\nENDNOISE\n" + kSbplx);
    CHECK(cal.error.empty());
    CHECK(cal.calibration == 2147483647u);
}

/* Pins: K_TOTAL caps the starting points the METHOD lines of its chain
 * give -- the INIT blocks inside OPTIMIZER, STRATEGY and REFINE make one
 * chain, the INIT blocks inside EXPLORE another, the RESTART blocks a third
 * -- so a K_TOTAL at or above their number caps nothing and is refused when
 * the file is read, at the K_TOTAL line, with the blocks, the count and the
 * METHOD lines (CONFIG and WARM give one point each): "K_TOTAL 3 caps
 * nothing: the METHOD lines of the INIT blocks inside OPTIMIZER, STRATEGY
 * and REFINE give at most 3 starting points (CONFIG, RANDOM 2), and K_TOTAL
 * keeps the first K_TOTAL starting points", "... the METHOD lines of the
 * RESTART blocks give ...", "... the METHOD lines of the INIT blocks inside
 * EXPLORE give ..."; a K_TOTAL in a block without a METHOD line counts the
 * METHOD lines of the other blocks of its chain. A chain without a METHOD
 * line gives one point, the CONFIG point. An INIT block inside EXPLORE
 * without a METHOD line gives the exploration the starting points of the
 * INIT blocks inside OPTIMIZER, STRATEGY and REFINE (METHOD RANDOM 5 here),
 * which its K_TOTAL does not cap: "K_TOTAL 2 caps nothing: an INIT block
 * inside EXPLORE without a METHOD line gives the exploration the starting
 * points of the INIT blocks inside OPTIMIZER, STRATEGY and REFINE (the
 * CONFIG point without one), and K_TOTAL caps only the points of the
 * block's own METHOD lines". The INIT blocks inside OPTIMIZER and
 * STRATEGY make one chain, counted together, and so do two RESTART blocks.
 * K_TOTAL 0 and METHOD RANDOM 0, which gave no cap and no point, are
 * refused at their line, as LHS 0 and SOBOL 0 are. K_TOTAL 3 of CONFIG and
 * RANDOM 4 keeps 3 points.
 * Before the fix: each of these files was read; K_TOTAL replaced the K of
 * a lone METHOD line (METHOD RANDOM 3 with K_TOTAL 7 drew 7 points), K_TOTAL
 * 0 set no cap, and RANDOM 0 drew no point. Before the fix of the EXPLORE
 * case, its refusal gave the reason of a block that starts from the CONFIG
 * point ("... a block without a METHOD line gives one starting point (the
 * CONFIG point) ..."; the text fails). Before the blocks were named: the
 * refusal said "the METHOD lines of the block give at most ...", also at a
 * K_TOTAL in a block that has no METHOD line (the text fails). */
TEST_CASE("K_TOTAL that caps nothing, K_TOTAL 0 and METHOD RANDOM 0 are "
          "refused at their line", "[HPO][init_strategy][k_total]")
{
    const std::string keeps =
        ", and K_TOTAL keeps the first K_TOTAL starting points";
    struct Refused { std::string block, error; };
    const std::vector<Refused> refused = {
        {init_block("METHOD RANDOM 3\nK_TOTAL 7\n"),
         "Error on line 5: K_TOTAL 7 caps nothing: the METHOD lines of the "
         "INIT blocks inside OPTIMIZER, STRATEGY and REFINE give at most 3 "
         "starting points (RANDOM 3)" + keeps +
             "\nLine: K_TOTAL 7"},
        {init_block("METHOD CONFIG\nMETHOD RANDOM 2\nK_TOTAL 3\n"),
         "Error on line 6: K_TOTAL 3 caps nothing: the METHOD lines of the "
         "INIT blocks inside OPTIMIZER, STRATEGY and REFINE give at most 3 "
         "starting points (CONFIG, RANDOM 2)" + keeps +
             "\nLine: K_TOTAL 3"},
        {init_block("METHOD WARM no_such_seed.tadah\nK_TOTAL 1\n"),
         "Error on line 5: K_TOTAL 1 caps nothing: the METHOD lines of the "
         "INIT blocks inside OPTIMIZER, STRATEGY and REFINE give at most 1 "
         "starting point (WARM no_such_seed.tadah)" +
             keeps + "\nLine: K_TOTAL 1"},
        {init_block("K_TOTAL 2\n"),
         "Error on line 4: K_TOTAL 2 caps nothing: a block without a METHOD "
         "line gives one starting point (the CONFIG point)" + keeps +
             "\nLine: K_TOTAL 2"},
        {"INIT\nMETHOD RANDOM 5\nSEED 2\nENDINIT\nSTRATEGY\nTYPE PRESTAGE\n"
         "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nTOPK 2\nEXPLORE\nLIB TADAH\n"
         "ALGO RANDOM\nMAXEVAL 5\nINIT\nK_TOTAL 2\nENDINIT\nENDEXPLORE\n"
         "ENDSTRATEGY\n",
         "Error on line 18: K_TOTAL 2 caps nothing: an INIT block inside "
         "EXPLORE without a METHOD line gives the exploration the starting "
         "points of the INIT blocks inside OPTIMIZER, STRATEGY and REFINE "
         "(the CONFIG point without one), and K_TOTAL caps only the points "
         "of the block's own METHOD lines\nLine: K_TOTAL 2"},
        {restart_block("METHOD LHS 2\nK_TOTAL 2\n"),
         "Error on line 11: K_TOTAL 2 caps nothing: the METHOD lines of the "
         "RESTART blocks give at most 2 starting points (LHS 2)" + keeps +
             "\nLine: K_TOTAL 2"},
        /* K_TOTAL in the STRATEGY-level INIT block (line 10) caps the chain
         * of the OPTIMIZER-level block's METHOD lines too. */
        {"INIT\nMETHOD LHS 3\nENDINIT\nSTRATEGY\nLIB NLOPT\nALGO LN_SBPLX\n"
         "MAXEVAL 10\nINIT\nK_TOTAL 5\nENDINIT\nENDSTRATEGY\n",
         "Error on line 11: K_TOTAL 5 caps nothing: the METHOD lines of the "
         "INIT blocks inside OPTIMIZER, STRATEGY and REFINE give at most 3 "
         "starting points (LHS 3)" + keeps + "\nLine: K_TOTAL 5"},
        /* The exploration's chain: its own METHOD lines. */
        {"STRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
         "EXPLORE\nLIB TADAH\nALGO RANDOM\nMAXEVAL 5\nINIT\nMETHOD LHS 2\n"
         "K_TOTAL 2\nENDINIT\nENDEXPLORE\nENDSTRATEGY\n",
         "Error on line 14: K_TOTAL 2 caps nothing: the METHOD lines of the "
         "INIT blocks inside EXPLORE give at most 2 starting points (LHS 2)" +
             keeps + "\nLine: K_TOTAL 2"},
        {init_block("METHOD RANDOM 2\nK_TOTAL 0\n"),
         "Error on line 5: K_TOTAL must be an integer >= 1, got '0'"
         "\nLine: K_TOTAL 0"},
        {init_block("METHOD RANDOM 0\n"),
         "Error on line 4: INIT METHOD RANDOM K must be an integer from 1 to "
         "1000000, got '0'\nLine: METHOD RANDOM 0"},
        {restart_block("METHOD RANDOM 0\n"),
         "Error on line 10: INIT METHOD RANDOM K must be an integer from 1 to "
         "1000000, got '0'\nLine: METHOD RANDOM 0"},
    };
    for (const auto& r : refused) {
        INFO(r.block);
        CHECK(read_counts(r.block).error == r.error);
    }
    /* Lines 3-6 and 8-10: the OPTIMIZER-level and STRATEGY-level INIT
     * blocks make one chain of 2 + 3 points, which K_TOTAL 4 caps. */
    const CountRead two = read_counts(
        "INIT\nMETHOD RANDOM 2\nK_TOTAL 4\nENDINIT\nSTRATEGY\nLIB NLOPT\n"
        "ALGO LN_SBPLX\nMAXEVAL 10\nINIT\nMETHOD LHS 3\nENDINIT\n"
        "ENDSTRATEGY\n");
    CHECK(two.error.empty());
    CHECK(two.points == 4u);
    const CountRead three =
        read_counts(init_block("METHOD CONFIG\nMETHOD RANDOM 4\nK_TOTAL 3\n"));
    CHECK(three.error.empty());
    CHECK(three.points == 3u);
    /* Two RESTART blocks make one: K_TOTAL in the first, the METHOD lines
     * in the second. */
    CHECK(read_counts("STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
                      "RESTART\nAFTER_STAGNATION 2\nMAX 1\nK_TOTAL 3\n"
                      "ENDRESTART\nRESTART\nMETHOD RANDOM 2\n"
                      "METHOD LHS 2\nENDRESTART\nENDSTRATEGY\n")
              .error.empty());
}

TEST_CASE("ChainStrategy: a chain of the largest RANDOM and LHS draws drops "
          "repeated points without comparing every pair",
          "[HPO][init_strategy][chain]")
{
    /* A chain drops a point equal to an earlier one (within 1e-12 in the
     * coordinates of the search). Before the fix each point was compared
     * with every point kept, each comparison mapping both to those
     * coordinates: two METHOD lines of 1000000 points, which the parser
     * takes, did not finish in hours. */
    using namespace tadah::hpo;
    auto sp = make_synthetic_spec({0.0, 0.0}, {1.0, 1.0});
    StubTargets st;
    InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {"1000000"}});
    ispec.chain.push_back({"LHS", {"1000000"}});
    std::size_t drawn = 0;
    hpo_test::finish_within(std::chrono::seconds(120),
                            "a chain of RANDOM 1000000 and LHS 1000000",
                            [&] {
        auto strat = build_init_strategy(ispec);
        InitContext ic{sp, *st.trg, 17};
        drawn = strat->generate(ic, 0).size();
    });
    CHECK(drawn == 2000000u);

    /* The same file named twice gives one point; a RANDOM line after it
     * adds its points. */
    StrategyTempFile pot("# comment-only warm file\n");
    InitStrategySpec twice;
    twice.chain.push_back({"WARM", {pot.path.string()}});
    twice.chain.push_back({"WARM", {pot.path.string()}});
    twice.chain.push_back({"RANDOM", {"3"}});
    twice.warm_on_mismatch = "warn";
    auto strat = build_init_strategy(twice);
    InitContext ic{sp, *st.trg, 5};
    const auto pts = strat->generate(ic, 0);
    REQUIRE(pts.size() == 4u);
    CHECK(pts.front().empty());
}

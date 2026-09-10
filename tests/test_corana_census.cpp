// test_corana_census.cpp
//
// Stage 11: reporting the per-dimension acceptance Corana already measures.
//
// The user-visible question was "does Corana track acceptance per dimension,
// or one number per temperature rung?". The ADAPTATION was always per
// dimension -- adjust_corana_step reads nA[i]/nP[i] for that coordinate alone
// -- so the defect was purely one of visibility, and these tests pin the two
// halves of the fix:
//
//   * the counters survive the NS reset that consumes the adaptation window
//     (without this the census silently reports the tail of one window, which
//     LOOKS like a plausible answer and is not one);
//   * the summary kernel's percentile, tie-breaking and exclusion rules.
//
// The first is the one that matters: a census built on corana_nA/corana_nP
// directly would pass every formatting test in this file and still be wrong.

#include "catch2/catch.hpp"

#include "anneal_test_support.h"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/corana_census.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace tadah::hpo;
using anneal_test::Vec;
using corana_census::DimAcceptance;

namespace {

/* A Corana chain on `dim` unit-box dimensions, stepped by `host`. */
struct CoranaFixture {
    OptimizerSpec sp;
    anneal::Knobs kb;
    anneal::ChainState<Vec> c;
    anneal::SharedBudget budget{std::numeric_limits<std::uint64_t>::max()};
    std::atomic<bool> stop{false};
    std::function<bool()> never = []() { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};

    CoranaFixture(std::size_t dim, std::size_t ns, std::size_t nt)
        : sp(anneal_test::serial_spec(std::vector<double>(dim, 0.0),
                                      std::vector<double>(dim, 1.0)))
    {
        sp.stop.maxeval = 1;
        sp.param_raw["CORANA"] = "ON";
        sp.param["NS"] = static_cast<double>(ns);
        sp.param["NT"] = static_cast<double>(nt);
        kb = anneal::Knobs::parse(sp, anneal::AlgoKind::Serial);
        REQUIRE(kb.corana);
        REQUIRE(kb.NS == ns);
        c.rng = anneal::make_stream(4242, anneal::SALT_CHAIN, 0);
        c.V.assign(dim, 0.1);
        c.x = Vec(static_cast<long>(dim));
        c.x = 0.5;
    }
};

}  // namespace

TEST_CASE("corana census: counters survive the NS adaptation reset",
          "[HPO][anneal][corana][census]")
{
    /* corana_sweep zeroes corana_nA/corana_nP every NS completed sweeps --
     * that is the adaptation's evidence window. The census must not be that
     * window. Run exactly NS sweeps so the reset fires on the last one, then
     * check that the adaptation counters are back at zero while the census
     * still holds every proposal. */
    const std::size_t dim = 4;
    CoranaFixture fx(dim, /*NS=*/2, /*NT=*/1);

    /* Accept everything: a strictly decreasing objective makes every move
     * downhill, so nA == nP and the arithmetic is unambiguous. */
    std::size_t evals = 0;
    auto host = [&](const Vec&) { return 1.0 - 1e-3 * (++evals); };
    fx.c.f = 1.0;

    for (std::size_t s = 0; s < fx.kb.NS; ++s)
        anneal::corana_sweep(fx.c, 1.0, fx.sp, fx.kb, host, fx.ctx);

    REQUIRE(fx.c.corana_sweeps == fx.kb.NS);
    for (std::size_t i = 0; i < dim; ++i) {
        INFO("dimension " << i);
        /* the adaptation window was consumed and zeroed ... */
        CHECK(fx.c.corana_nP[i] == 0);
        CHECK(fx.c.corana_nA[i] == 0);
        /* ... while the census kept every event */
        CHECK(fx.c.corana_census_nP[i] == fx.kb.NS);
        CHECK(fx.c.corana_census_nA[i] == fx.kb.NS);
    }
}

TEST_CASE("corana census: the serial rung kernel counts too",
          "[HPO][anneal][corana][census]")
{
    const std::size_t dim = 3;
    CoranaFixture fx(dim, /*NS=*/2, /*NT=*/2);

    std::size_t evals = 0;
    auto host = [&](const Vec&) { return 1.0 - 1e-3 * (++evals); };
    fx.c.f = 1.0;
    const auto rs =
        anneal::corana_rung(fx.c, 1.0, fx.sp, fx.kb, host, fx.ctx);

    REQUIRE(rs.proposed == fx.kb.NT * fx.kb.NS * dim);
    std::size_t total = 0;
    for (std::size_t i = 0; i < dim; ++i) total += fx.c.corana_census_nP[i];
    CHECK(total == rs.proposed);
    /* Per dimension, not one number for the rung: every coordinate was
     * offered exactly NT*NS moves. */
    for (std::size_t i = 0; i < dim; ++i)
        CHECK(fx.c.corana_census_nP[i] == fx.kb.NT * fx.kb.NS);
}

TEST_CASE("corana census: rejected moves count as proposals, not accepts",
          "[HPO][anneal][corana][census]")
{
    /* A census that incremented only on acceptance would read 100%
     * everywhere -- the exact blind spot the whole diagnostic exists to
     * remove. At T -> 0 with a strictly increasing objective, nothing is
     * accepted. */
    const std::size_t dim = 3;
    CoranaFixture fx(dim, /*NS=*/1, /*NT=*/1);

    std::size_t evals = 0;
    auto host = [&](const Vec&) { return 1.0 + 1e-3 * (++evals); };
    fx.c.f = 0.0;
    anneal::corana_sweep(fx.c, 1e-30, fx.sp, fx.kb, host, fx.ctx);

    for (std::size_t i = 0; i < dim; ++i) {
        CHECK(fx.c.corana_census_nP[i] == 1);
        CHECK(fx.c.corana_census_nA[i] == 0);
    }
}

TEST_CASE("corana census: unvisited dims are excluded, not scored 0%",
          "[HPO][anneal][corana][census]")
{
    /* A sweep cut short by the budget leaves its tail with proposed == 0.
     * Scoring those as "0% acceptance" would turn a bookkeeping artefact
     * into a diagnosis of frozen coordinates -- and, worse, would drag the
     * percentiles down in exactly the runs (budget-limited ones) where the
     * reader is most likely to act on them. */
    std::vector<DimAcceptance> d(4);
    d[0].accepted = 5;  d[0].proposed = 10;   /* 50% */
    d[1].accepted = 5;  d[1].proposed = 10;   /* 50% */
    d[2].accepted = 0;  d[2].proposed = 0;    /* never offered a move */
    d[3].accepted = 0;  d[3].proposed = 0;

    const auto c = corana_census::summarize(d);
    CHECK(c.dims == 4);
    CHECK(c.visited == 2);
    CHECK(c.unvisited == 2);
    CHECK(c.frozen == 0);          /* NOT 2 */
    CHECK(c.pct[corana_census::kP50] == Approx(0.5));
    CHECK(c.pct[corana_census::kP0] == Approx(0.5));
    /* Totals still cover every dimension. */
    CHECK(c.proposed == 20);
    CHECK(c.accepted == 10);
}

TEST_CASE("corana census: frozen and saturated are counted separately",
          "[HPO][anneal][corana][census]")
{
    std::vector<DimAcceptance> d(3);
    d[0].accepted = 0;  d[0].proposed = 10;   /* frozen    */
    d[1].accepted = 10; d[1].proposed = 10;   /* saturated */
    d[2].accepted = 4;  d[2].proposed = 10;

    const auto c = corana_census::summarize(d);
    CHECK(c.visited == 3);
    CHECK(c.frozen == 1);
    CHECK(c.saturated == 1);
    CHECK(c.pct[corana_census::kP0] == Approx(0.0));
    CHECK(c.pct[corana_census::kP100] == Approx(1.0));
}

TEST_CASE("corana census: percentiles are attained values, min/max exact",
          "[HPO][anneal][corana][census]")
{
    /* Nearest-rank, deliberately: these are ratios of small integer counts,
     * so an interpolating definition would print rates no dimension had. */
    std::vector<DimAcceptance> d(4);
    for (std::size_t i = 0; i < 4; ++i) {
        d[i].proposed = 4;
        d[i].accepted = i;         /* 0, 0.25, 0.5, 0.75 */
    }
    const auto c = corana_census::summarize(d);
    CHECK(c.pct[corana_census::kP0]   == Approx(0.00));
    CHECK(c.pct[corana_census::kP25]  == Approx(0.00));
    CHECK(c.pct[corana_census::kP50]  == Approx(0.25));
    CHECK(c.pct[corana_census::kP75]  == Approx(0.50));
    CHECK(c.pct[corana_census::kP100] == Approx(0.75));
    for (std::size_t q = 0; q < corana_census::kNumPct; ++q) {
        bool attained = false;
        for (const auto& e : d)
            if (std::abs(e.rate() - c.pct[q]) < 1e-12) attained = true;
        INFO("percentile index " << q << " = " << c.pct[q]);
        CHECK(attained);
    }
}

TEST_CASE("corana census: ties break on dimension index, deterministically",
          "[HPO][anneal][corana][census]")
{
    /* Hundreds of dims at exactly 0/n is the NORMAL case at the cold end.
     * Without an explicit secondary key the named extremes would depend on
     * the sort's internal ordering, so two runs that behaved identically
     * could name different dimensions -- and a reader chasing "why is
     * SGRIDMB[7] frozen" would be chasing an artefact of std::sort. */
    /* SIZE MATTERS HERE. libstdc++'s std::sort falls back to insertion sort
     * below 16 elements, which IS stable, so a six-dimension version of this
     * test passes with the tie-break DELETED -- verified, it does. Above the
     * threshold introsort partitions and the tied block comes out permuted
     * (at n=17 the first five became 1,15,14,13,12). 64 is comfortably past
     * it and is the order of a real search space, so do not shrink this. */
    const std::size_t n = 64;
    std::vector<DimAcceptance> d(n);
    for (auto& e : d) { e.proposed = 10; e.accepted = 0; }
    d[n - 1].accepted = 10;                   /* the one outlier */

    const auto c = corana_census::summarize(d);
    REQUIRE(c.lowest.size() == corana_census::kExtremes);
    /* All 63 lowest tie at 0%; the tie-break picks the lowest indices. */
    CHECK(c.lowest == std::vector<std::size_t>{0, 1, 2, 3, 4});
    /* highest runs high-to-low, so the outlier leads, and the rest of that
     * list is the TOP of the tied block -- again by index, descending. */
    REQUIRE(c.highest.size() == corana_census::kExtremes);
    CHECK(c.highest[0] == n - 1);
    CHECK(c.highest == std::vector<std::size_t>{n - 1, n - 2, n - 3, n - 4,
                                                n - 5});
}

TEST_CASE("corana census: extremes degrade gracefully below kExtremes dims",
          "[HPO][anneal][corana][census]")
{
    std::vector<DimAcceptance> d(2);
    d[0].proposed = 3; d[0].accepted = 1;
    d[1].proposed = 3; d[1].accepted = 2;
    const auto c = corana_census::summarize(d);
    /* Two dimensions: `lowest` already names both, so `highest` would be
     * the same set printed backwards. It is empty instead -- the driver
     * then prints ONE ordered list, and the observed
     * "lowest 5 = SGRIDMB 40.6%, RCUT2B 53.1% | highest 5 = RCUT2B 53.1%,
     * SGRIDMB 40.6%" cannot recur. */
    CHECK(c.lowest == std::vector<std::size_t>{0, 1});
    CHECK(c.highest.empty());

    /* And on nothing at all: no percentiles, no crash. */
    const auto empty = corana_census::summarize({});
    CHECK(empty.dims == 0);
    CHECK(empty.visited == 0);
    CHECK(empty.lowest.empty());
    CHECK(empty.pct[corana_census::kP50] == Approx(0.0));
}

TEST_CASE("corana census: the two extreme lists never name the same dim",
          "[HPO][anneal][corana][census]")
{
    /* The awkward band is kExtremes < visited < 2*kExtremes, where a naive
     * "first five, last five" overlaps in the middle and reports the same
     * coordinate as both the worst and the best. Sweep the whole band. */
    for (std::size_t n = 1; n <= 3 * corana_census::kExtremes; ++n) {
        std::vector<DimAcceptance> d(n);
        for (std::size_t i = 0; i < n; ++i) {
            d[i].proposed = n;
            d[i].accepted = i;          /* strictly increasing, no ties */
        }
        const auto c = corana_census::summarize(d);
        INFO("n = " << n);
        for (std::size_t a : c.lowest)
            for (std::size_t b : c.highest) CHECK(a != b);
        /* Together they cover everything, up to the 2*kExtremes cap. */
        CHECK(c.lowest.size() + c.highest.size() ==
              std::min(n, 2 * corana_census::kExtremes));
    }
}

TEST_CASE("corana census: rendered rows and lists are 1-based",
          "[HPO][anneal][corana][census][radix]")
{
    std::vector<DimAcceptance> d(3);
    for (std::size_t i = 0; i < 3; ++i) { d[i].proposed = 4; d[i].accepted = i; }
    const auto c = corana_census::summarize(d);

    const std::string row = corana_census::format_slot_row(0, 1.5e-3, c);
    /* Slot 0 internally renders as slot 1 for the reader. */
    CHECK(row.find(" 1  ") != std::string::npos);

    const auto label = [](std::size_t i) { return logidx::dim_label(i); };
    const std::string lo =
        corana_census::format_dim_list(c.lowest, d, label);
    CHECK(lo.find("dim[1]") != std::string::npos);
    CHECK(lo.find("dim[0]") == std::string::npos);
    CHECK(lo.find("0.0%") != std::string::npos);

    const std::string head = corana_census::slot_table_header();
    CHECK(head.find("frozen") != std::string::npos);
    CHECK(head.find("p50%") != std::string::npos);
}

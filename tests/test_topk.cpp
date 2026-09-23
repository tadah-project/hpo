// test_topk.cpp — near-clone dedup of the top-K buffer (topk.h).
//
// The buffer feeds the PRESTAGE refinement multistart: it must not fill up
// with near-clones from a low-temperature explorer hovering around its best
// (loss AND parameters nearly equal), while distinct points that happen to
// share a loss value must all be kept.

#include "catch2/catch.hpp"

#include <tadah/hpo/topk.h>

#include <dlib/matrix.h>

#include <utility>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;
using Buf = std::vector<std::pair<double, Vec>>;

namespace {

Vec vec2(double a, double b) {
    Vec v(2);
    v(0) = a; v(1) = b;
    return v;
}

const std::vector<double> WIDTHS{1.0, 1.0};   // unit boxes for readability

}  // namespace

TEST_CASE("topk: clone flooding keeps a single (best) entry",
          "[HPO][topk]") {
    Buf buf;
    /* 100 near-identical visits around one point: loss jitter ~1e-9,
     * param jitter ~1e-6 of the box width. */
    for (int i = 0; i < 100; ++i) {
        const double jitter = 1e-9 * i;
        tadah::hpo::topk_consider(buf, 3, 5.0 + jitter,
                                  vec2(0.5 + 1e-6 * i, 0.5), WIDTHS);
    }
    REQUIRE(buf.size() == 1);
    CHECK(buf[0].first == Approx(5.0));       // kept the best of the clones
}

TEST_CASE("topk: better clone replaces its near-clone entry in place",
          "[HPO][topk]") {
    Buf buf;
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.5, 0.5), WIDTHS);
    tadah::hpo::topk_consider(buf, 3, 5.0 - 1e-8, vec2(0.5, 0.5), WIDTHS);
    REQUIRE(buf.size() == 1);
    CHECK(buf[0].first == Approx(5.0 - 1e-8));
}

TEST_CASE("topk: distinct points with (near-)equal losses are all kept",
          "[HPO][topk]") {
    Buf buf;
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.1, 0.1), WIDTHS);
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.9, 0.9), WIDTHS);  // far away
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.1, 0.9), WIDTHS);  // far away
    REQUIRE(buf.size() == 3);
}

TEST_CASE("topk: capacity respected, worst evicted by a better outsider",
          "[HPO][topk]") {
    Buf buf;
    tadah::hpo::topk_consider(buf, 2, 3.0, vec2(0.1, 0.1), WIDTHS);
    tadah::hpo::topk_consider(buf, 2, 7.0, vec2(0.9, 0.9), WIDTHS);
    /* Worse than both and not a clone: rejected at capacity. */
    tadah::hpo::topk_consider(buf, 2, 9.0, vec2(0.5, 0.5), WIDTHS);
    REQUIRE(buf.size() == 2);
    /* Better than the worst and not a clone: evicts the 7.0 entry. */
    tadah::hpo::topk_consider(buf, 2, 5.0, vec2(0.5, 0.5), WIDTHS);
    REQUIRE(buf.size() == 2);
    double lo = std::min(buf[0].first, buf[1].first);
    double hi = std::max(buf[0].first, buf[1].first);
    CHECK(lo == Approx(3.0));
    CHECK(hi == Approx(5.0));
}

TEST_CASE("topk: empty widths fall back to loss-only dedup",
          "[HPO][topk]") {
    Buf buf;
    const std::vector<double> none;
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.1, 0.1), none);
    /* Same loss, far params: without widths this counts as a clone
     * (conservative pre-existing behaviour). */
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.9, 0.9), none);
    REQUIRE(buf.size() == 1);
}

TEST_CASE("topk: same params but genuinely different loss are two entries",
          "[HPO][topk]") {
    /* Different loss at the same point can happen across noisy re-evals of
     * a NON-deterministic objective; they are not clones. */
    Buf buf;
    tadah::hpo::topk_consider(buf, 3, 5.0, vec2(0.5, 0.5), WIDTHS);
    tadah::hpo::topk_consider(buf, 3, 6.0, vec2(0.5, 0.5), WIDTHS);
    REQUIRE(buf.size() == 2);
}

#include "catch2/catch.hpp"

#include <tadah/hpo/progress_tracker.h>

using tadah::hpo::ProgressTracker;

namespace {

ProgressTracker make_tracker(double tol, std::size_t min_evals,
                             std::size_t window)
{
    ProgressTracker t;
    ProgressTracker::StageConfig cfg;
    cfg.tol         = tol;
    cfg.min_evals   = min_evals;
    cfg.window_size = window;
    cfg.rate_factor = 0.5;
    cfg.max_evals   = 0;
    t.start_stage(cfg);
    return t;
}

}  // namespace

TEST_CASE("ProgressTracker: not stuck before min_evals", "[progress_tracker]")
{
    auto t = make_tracker(1e-3, 30, 5);
    /* Slow improvement: each eval improves by 1e-9. Tiny rate, but
     * we haven't reached min_evals yet — must not be stuck. */
    for (int i = 0; i < 10; ++i) {
        t.observe(1.0 - i * 1e-9, 1.0);
    }
    /* Optimiser convergence already signalled to make conditions 3
     * true; condition 1 should still block. */
    t.mark_optimizer_converged();
    REQUIRE_FALSE(t.is_stuck());
}

TEST_CASE("ProgressTracker: stuck only when all conditions hold",
          "[progress_tracker]")
{
    auto t = make_tracker(1e-3, 5, 5);
    /* Observe 10 evals with effectively zero improvement. */
    for (int i = 0; i < 10; ++i) t.observe(1.0, 1.0);

    /* Rate is zero (< 0.5*1e-3), n_obs >= window, but optimiser
     * hasn't signalled — not stuck. */
    REQUIRE_FALSE(t.is_stuck());

    t.mark_optimizer_converged();
    REQUIRE(t.is_stuck());
}

TEST_CASE("ProgressTracker: not stuck while still improving",
          "[progress_tracker]")
{
    auto t = make_tracker(1e-3, 5, 5);
    /* Sustained ~1% relative improvement per eval — far above the
     * 0.5*1e-3 threshold. */
    double L = 1.0;
    for (int i = 0; i < 20; ++i) {
        L *= 0.99;
        t.observe(L, 1.0);
    }
    t.mark_optimizer_converged();   // even with this set
    REQUIRE_FALSE(t.is_stuck());
    /* The tracker should report a positive rate. */
    REQUIRE(t.rate_per_eval() > 0.0);
}

TEST_CASE("ProgressTracker: max_evals trips the optimizer flag",
          "[progress_tracker]")
{
    ProgressTracker t;
    ProgressTracker::StageConfig cfg;
    cfg.tol = 1e-3; cfg.min_evals = 1; cfg.window_size = 5;
    cfg.rate_factor = 0.5; cfg.max_evals = 7;
    t.start_stage(cfg);

    for (int i = 0; i < 7; ++i) t.observe(1.0, 1.0);
    /* Rate is zero, n_obs=window=5..7, max_evals (7) hit — stuck. */
    REQUIRE(t.is_stuck());
}

TEST_CASE("ProgressTracker: rate is non-negative, infinite for trivial windows",
          "[progress_tracker]")
{
    auto t = make_tracker(1e-3, 5, 5);
    /* Single observation: rate is undefined; we surface infinity so
     * is_stuck() can never trip on c2 alone. */
    t.observe(1.0, 1.0);
    REQUIRE(t.rate_per_eval() == std::numeric_limits<double>::infinity());
    /* After enough observations, rate is finite. */
    for (int i = 0; i < 5; ++i) t.observe(1.0, 1.0);
    REQUIRE(t.rate_per_eval() == 0.0);
    /* Loss never goes up — drop is clamped at zero. */
    t.observe(2.0, 1.0);
    REQUIRE(t.rate_per_eval() == 0.0);
}

TEST_CASE("ProgressTracker: reset_stage clears all counters",
          "[progress_tracker]")
{
    auto t = make_tracker(1e-3, 1, 5);
    for (int i = 0; i < 5; ++i) t.observe(1.0 - i*0.01, 1.0);
    t.mark_optimizer_converged();
    REQUIRE(t.n_obs() == 5);
    t.reset_stage();
    REQUIRE(t.n_obs() == 0);
    REQUIRE_FALSE(t.is_stuck());
    REQUIRE_FALSE(t.optimizer_done());
}

#ifndef TADAH_HPO_PROGRESS_TRACKER_H
#define TADAH_HPO_PROGRESS_TRACKER_H

/*
 *  ProgressTracker -- per-stage stuck detector for the BASINS strategy.
 *
 *  The user's safety contract:
 *    do not kill a candidate that is still progressing slowly relative
 *    to the stage's expected rate.
 *
 *  is_stuck() returns true iff *all* of:
 *    1. We have observed at least `min_evals` evals (no premature kill).
 *    2. The recent rate of best-loss improvement falls below
 *       `rate_threshold` (tolerance-derived; tighter tol => lower bar).
 *    3. The optimiser itself has signaled either ftol_rel/xtol_rel
 *       convergence or per-stage maxeval -- recorded via the
 *       mark_optimizer_*() setters.
 *
 *  Conditions are AND'd. A single failing condition keeps the stage
 *  alive so a slow-but-genuinely-improving basin gets the room it
 *  needs.
 *
 *  Window semantics: "rate" is computed over the last `window_size`
 *  evals. Smaller windows react faster but are noisier; we default to
 *  W = max(50, 50 * round) so later rounds are more patient with
 *  slow progress (since the inner objective is harder).
 */

#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <string>

namespace tadah {
namespace hpo {

class ProgressTracker {
public:
    ProgressTracker() { reset_stage(); }

    /* Begin a new stage. */
    struct StageConfig {
        double      tol           = 1.0e-3;   // ftol_rel for this stage
        double      rate_factor   = 0.5;      // is_stuck if rate < factor*tol
        std::size_t min_evals     = 30;       // floor on observations
        std::size_t window_size   = 50;       // rate-of-progress window
        std::size_t max_evals     = 0;        // 0 = unlimited safety cap
    };
    void start_stage(const StageConfig& cfg);

    /* Per-eval update. `loss` is the best-so-far value the caller
     * wants to track (monotone non-increasing). `wall_ms` is purely
     * informational (rate_per_second). */
    void observe(double loss, double wall_ms);

    /* Caller-driven flags. The optimiser adapter sets one of these
     * when the inner library reports convergence. */
    void mark_optimizer_converged();
    void mark_optimizer_maxeval_hit();

    /* Composite verdict. */
    bool is_stuck() const;

    /* Inspectors (used by HPOLog). */
    double      rate_per_eval()    const;
    double      rate_per_second()  const;
    std::size_t window_size()      const { return cfg_.window_size; }
    std::size_t n_obs()            const { return n_obs_; }
    double      best_loss()        const { return best_; }
    double      stage_tol()        const { return cfg_.tol; }
    bool        optimizer_done()   const { return opt_converged_ || opt_maxeval_; }
    /* Reason string for hpolog ("rate=... < threshold=...", "optimizer
     * convergence", "max evals", ...). Best-effort, may be empty. */
    std::string stuck_reason()     const;

    /* Reset for a brand-new stage. */
    void reset_stage();

private:
    StageConfig                                   cfg_{};
    std::deque<std::pair<double, double>>         hist_;   // (loss, wall_ms)
    double                                        best_       = std::numeric_limits<double>::infinity();
    double                                        best_window_start_ = std::numeric_limits<double>::infinity();
    std::size_t                                   n_obs_      = 0;
    double                                        sum_wall_ms_ = 0.0;
    bool                                          opt_converged_ = false;
    bool                                          opt_maxeval_   = false;
};

}   // namespace hpo
}   // namespace tadah

#endif

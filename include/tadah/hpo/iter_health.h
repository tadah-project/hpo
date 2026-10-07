#ifndef TADAH_HPO_ITER_HEALTH_H
#define TADAH_HPO_ITER_HEALTH_H

// Phase 6 -- per-iteration HPO health monitor.
//
// The monitor records cheap per-block aggregates of the weights the
// model of every HPO evaluation holds (fitted, or provisioned from
// FIXWEIGHT) and emits rate-limited WARNINGs (core::Logger) when an
// aggregate trips a threshold. Total cost per iter: one linear scan of
// the weights, partitioned by the column layout of the model's design
// matrix (`ColumnLayout`, column_block.h; HPO_Host passes
// fb->phi_layout(dc.layout())).
//
// Aggregates:
//   - weights_have_nan_inf: any non-finite weight, whatever the layout.
//   - layout_matches   : the layout's blocks partition the weights
//                        (contiguous from column 0, as many columns as
//                        weights); when not, the per-block aggregates are
//                        skipped and record() says so.
//   per ColumnBlock (when the layout matches):
//   - block_max_abs    : max(|w|) within the block.
//   - block_median_abs : median(|w|) within the block.
//   - block_nz_fraction: fraction of |w| > 1e-15.
//
// Modes (HEALTH_LOG context key):
//   - off     : monitor is inactive; record() is a no-op.
//   - summary : (default) at most one warning per ~window evals per
//               (block, kind) pair.
//   - full    : every offending iter produces a WARNING.
//
// One monitor serves the process (global_iter_health_monitor()), and
// HPO_Host records into it from every evaluation thread at once (the
// annealers' replicas, DLIB:GFS batches): record(), warning_count() and
// clear_state() are thread-safe.

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/mlip/design_matrix/column_block.h>

namespace tadah {
namespace hpo {

enum class HealthLogMode { Off, Summary, Full };

HealthLogMode parse_health_log_mode(const std::string& s);
HealthLogMode health_log_mode_from_context(const tadah::core::Context& ctx);

/** \brief One per-iter snapshot: one slot per ColumnBlock when the layout
 *  describes the weights (layout_matches), none otherwise. */
struct IterHealthRecord {
  std::size_t              iter{0};
  std::vector<double>      block_max_abs{};
  std::vector<double>      block_median_abs{};
  std::vector<double>      block_nz_fraction{};
  /** A weight is NaN or infinite (any weight, inside a block or not). */
  bool                     weights_have_nan_inf{false};
  /** The layout's blocks partition the weights: contiguous from column 0,
   *  as many columns as there are weights (ColumnLayout::validate). */
  bool                     layout_matches{true};
  std::size_t              n_weights{0};    ///< weights.size()
  std::size_t              layout_cols{0};  ///< layout.total_cols()
};

/** \brief Compute an IterHealthRecord from a weight vector + layout.
 *  Pure function: no logging, no state. Tests can call this directly.
 *
 *  Every weight is checked for NaN/Inf. The per-block aggregates are
 *  computed only when the layout's blocks partition the weights
 *  (layout_matches); a layout that does not describe the weights leaves
 *  them empty -- it is not taken for non-finite weights.
 *
 *  Median-abs uses partial sort (`std::nth_element`) -- O(N) per block.
 *  Non-zero floor is the IEEE denormal floor 1e-15 -- explicitly the
 *  only "constant" allowed in the audit code per the no-magic-numbers
 *  contract (see plan sec."No-magic-numbers contract").
 */
IterHealthRecord compute_iter_health(
    std::size_t                            iter,
    const tadah::core::t_type&             weights,
    const tadah::mlip::ColumnLayout&       layout);

class IterHealthMonitor;

/** \brief Process-singleton accessor -- wired in `hpo_run()` from the
 *  HEALTH_LOG context key. Default mode: Off (no overhead until
 *  configured). Tests can construct local instances and call record()
 *  directly without going through the singleton. */
IterHealthMonitor& global_iter_health_monitor();

/** \brief Stateful monitor that emits rate-limited warnings. Thread-safe:
 *  HPO_Host's concurrent evaluations record into one monitor (a mutex
 *  guards the cooldown and count maps; the mode is atomic). The
 *  `record(...)` aggregation itself is small enough to live inside
 *  the eval loop.
 */
class IterHealthMonitor {
 public:
  IterHealthMonitor() = default;

  /** Construct with mode + summary-window size. Default window is 100
   *  evals -- emits at most one warning per (block, kind) per 100
   *  evals in summary mode. */
  IterHealthMonitor(HealthLogMode mode,
                    std::size_t   summary_window = 100)
      : mode_(mode), window_(summary_window) {}

  void set_mode(HealthLogMode m) { mode_.store(m, std::memory_order_relaxed); }
  HealthLogMode mode() const noexcept {
    return mode_.load(std::memory_order_relaxed);
  }

  /** Record one iter and emit any warnings. Safe to call with empty
   *  records -- no-ops then. A record whose layout does not describe the
   *  weights (layout_matches false) warns so, under the "*:layout" key. */
  void record(const IterHealthRecord&         rec,
              const tadah::mlip::ColumnLayout& layout);

  /** Test-only: pull the warning count for a given (block_idx, kind).
   *  kind in {"max_abs", "nz_fraction"}. */
  std::size_t warning_count(std::size_t block_idx,
                            const std::string& kind) const;

  /** Reset all cooldown counters. */
  void clear_state();

 private:
  std::atomic<HealthLogMode> mode_{HealthLogMode::Summary};
  std::size_t   window_{100};

  // Cooldown counters: key = "blockidx:kind" ("*:nan_inf", "*:layout" for
  // the warnings of no block), value = last-warning iter. Guarded by mtx_.
  mutable std::mutex mtx_;
  std::unordered_map<std::string, std::size_t> last_warned_iter_{};
  std::unordered_map<std::string, std::size_t> warning_counts_{};
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_ITER_HEALTH_H

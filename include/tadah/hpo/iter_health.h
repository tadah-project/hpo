#ifndef TADAH_HPO_ITER_HEALTH_H
#define TADAH_HPO_ITER_HEALTH_H

// Phase 6 -- per-iteration HPO health monitor.
//
// The monitor records cheap per-block aggregates of the trained
// weights produced by every HPO evaluation and emits rate-limited
// WARNINGs when an aggregate trips a per-block threshold. Total cost
// per iter: one linear scan of the weights, partitioned by the
// `ColumnLayout` from Phase 2.
//
// Aggregates (per ColumnBlock):
//   - block_max_abs    : max(|w|) within the block.
//   - block_median_abs : median(|w|) within the block.
//   - block_nz_fraction: fraction of |w| > 1e-15.
//   - weights_have_nan_inf: any non-finite w in the block.
//
// Modes (HEALTH_LOG context key):
//   - off     : monitor is inactive; record() is a no-op.
//   - summary : (default) at most one warning per ~window evals per
//               (block, kind) pair.
//   - full    : every offending iter produces a WARNING.

#include <atomic>
#include <cstddef>
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

/** \brief One per-iter snapshot, one slot per ColumnBlock. */
struct IterHealthRecord {
  std::size_t              iter{0};
  std::vector<double>      block_max_abs{};
  std::vector<double>      block_median_abs{};
  std::vector<double>      block_nz_fraction{};
  bool                     weights_have_nan_inf{false};
};

/** \brief Compute an IterHealthRecord from a weight vector + layout.
 *  Pure function: no logging, no state. Tests can call this directly.
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

/** \brief Stateful monitor that emits rate-limited warnings. The
 *  monitor is intentionally not thread-safe at the warning side --
 *  callers should funnel records through a single log thread. The
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

  void set_mode(HealthLogMode m) { mode_ = m; }
  HealthLogMode mode() const noexcept { return mode_; }

  /** Record one iter and emit any warnings. Safe to call with empty
   *  records -- no-ops then. */
  void record(const IterHealthRecord&         rec,
              const tadah::mlip::ColumnLayout& layout);

  /** Test-only: pull the warning count for a given (block_idx, kind).
   *  kind in {"max_abs", "nz_fraction", "nan_inf"}. */
  std::size_t warning_count(std::size_t block_idx,
                            const std::string& kind) const;

  /** Reset all cooldown counters. */
  void clear_state();

 private:
  HealthLogMode mode_{HealthLogMode::Summary};
  std::size_t   window_{100};

  // Cooldown counters: key = "blockidx:kind", value = last-warning iter.
  std::unordered_map<std::string, std::size_t> last_warned_iter_{};
  std::unordered_map<std::string, std::size_t> warning_counts_{};
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_ITER_HEALTH_H

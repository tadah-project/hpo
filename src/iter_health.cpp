#include <tadah/hpo/iter_health.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/logger.h>

namespace tadah {
namespace hpo {

namespace {

// IEEE denormal floor -- the only literal threshold permitted by the
// no-magic-numbers contract (plan sec."No-magic-numbers contract").
constexpr double kZeroFloor = 1.0e-15;

std::string lower(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

}  // namespace

IterHealthMonitor& global_iter_health_monitor() {
  // Default state: Off -- no overhead until hpo_run() flips it on
  // based on the HEALTH_LOG context key. Process-lifetime singleton.
  static IterHealthMonitor mon{HealthLogMode::Off};
  return mon;
}

HealthLogMode parse_health_log_mode(const std::string& s) {
  const std::string v = lower(s);
  if (v == "off")     return HealthLogMode::Off;
  if (v == "summary") return HealthLogMode::Summary;
  if (v == "full")    return HealthLogMode::Full;
  throw std::runtime_error(
      "Invalid HEALTH_LOG value '" + s + "'. Expected: off|summary|full.");
}

HealthLogMode health_log_mode_from_context(const tadah::core::Context& ctx) {
  if (!ctx.contains("HEALTH_LOG")) return HealthLogMode::Summary;
  return parse_health_log_mode(ctx.get<std::string>("HEALTH_LOG"));
}

IterHealthRecord compute_iter_health(
    std::size_t                            iter,
    const tadah::core::t_type&             weights,
    const tadah::mlip::ColumnLayout&       layout) {
  IterHealthRecord rec;
  rec.iter = iter;
  rec.n_weights = weights.size();
  rec.layout_cols = layout.total_cols();

  // Every weight, inside a block or not, whatever the layout.
  rec.weights_have_nan_inf = false;
  for (std::size_t c = 0; c < weights.size(); ++c)
    if (!std::isfinite(weights[c])) {
      rec.weights_have_nan_inf = true;
      break;
    }

  // The blocks partition the weights only when the layout has as many
  // columns as there are weights (validate(): contiguous from 0). A layout
  // that does not describe them is reported (record()), not read: a block
  // past the end of the weights used to be taken for NaN/Inf weights.
  rec.layout_matches = layout.validate(weights.size()).empty();
  if (!rec.layout_matches) return rec;

  const std::size_t nblocks = layout.blocks.size();
  rec.block_max_abs   .assign(nblocks, 0.0);
  rec.block_median_abs.assign(nblocks, 0.0);
  rec.block_nz_fraction.assign(nblocks, 0.0);
  rec.block_has_nan_inf.assign(nblocks, false);

  // Reusable scratch -- declared outside the loop to avoid reallocation.
  std::vector<double> scratch;

  for (std::size_t b = 0; b < nblocks; ++b) {
    const auto& blk = layout.blocks[b];
    const std::size_t lo = blk.col_offset;
    const std::size_t hi = lo + blk.col_count;
    if (blk.col_count == 0) continue;

    // A block that holds a NaN or infinite weight gets no aggregates: the
    // comparisons below are false for NaN, which would count it as an
    // exact zero (a block of NaN weights read as a block of zeros).
    bool finite = true;
    for (std::size_t c = lo; c < hi && finite; ++c)
      finite = std::isfinite(weights[c]);
    if (!finite) {
      rec.block_has_nan_inf[b] = true;
      continue;
    }

    scratch.clear();
    scratch.reserve(blk.col_count);
    double mx = 0.0;
    std::size_t nz = 0;
    for (std::size_t c = lo; c < hi; ++c) {
      const double w = weights[c];
      const double aw = std::fabs(w);
      scratch.push_back(aw);
      if (aw > mx) mx = aw;
      if (aw > kZeroFloor) ++nz;
    }
    rec.block_max_abs[b]    = mx;
    rec.block_nz_fraction[b] =
        static_cast<double>(nz) / static_cast<double>(blk.col_count);

    // Median via nth_element -- O(N) per block.
    const std::size_t mid = scratch.size() / 2;
    std::nth_element(scratch.begin(),
                     scratch.begin() + static_cast<std::ptrdiff_t>(mid),
                     scratch.end());
    rec.block_median_abs[b] = scratch[mid];
  }
  return rec;
}

void IterHealthMonitor::record(
    const IterHealthRecord&         rec,
    const tadah::mlip::ColumnLayout& layout) {
  const HealthLogMode mode = this->mode();
  if (mode == HealthLogMode::Off) return;

  // Evaluations record concurrently: the maps, and the order of the
  // warnings a key's cooldown decides, are under one lock.
  std::lock_guard<std::mutex> lk(mtx_);

  // Whether the cooldown of `key` lets this record warn; counts it if so.
  auto take = [&](const std::string& key) {
    auto& count = warning_counts_[key];
    auto& last  = last_warned_iter_[key];
    if (mode == HealthLogMode::Summary && count > 0 &&
        rec.iter < last + window_) {
      return false;
    }
    ++count;
    last = rec.iter;
    return true;
  };

  auto maybe_warn = [&](std::size_t block_idx,
                        const std::string& kind,
                        const std::string& text) {
    if (!take(std::to_string(block_idx) + ":" + kind)) return;
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        "iter_health: eval " + std::to_string(rec.iter) +
        " block[" + std::to_string(block_idx) + "/" +
        layout.blocks[block_idx].descriptor_label + "] " +
        kind + " -- " + text);
  };

  // NaN/Inf and a layout that does not describe the weights are the
  // globally-emitted warnings (no per-block index).
  if (rec.weights_have_nan_inf && take("*:nan_inf")) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        "iter_health: eval " + std::to_string(rec.iter) +
        " -- weights contain NaN/Inf entries (training likely diverged).");
  }
  if (!rec.layout_matches) {
    if (take("*:layout")) {
      const std::string what =
          rec.layout_cols != rec.n_weights
              ? "the model has " + std::to_string(rec.n_weights) +
                    " weights but its column layout has " +
                    std::to_string(rec.layout_cols) + " columns"
              : "the column layout of the model's " +
                    std::to_string(rec.n_weights) +
                    " weights is not contiguous from column 0";
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::WARNING,
          "iter_health: eval " + std::to_string(rec.iter) + " -- " + what +
          ": the per-block checks are skipped.");
    }
    return;
  }

  // Per-block exact-zero blocks. block_max_abs == 0 means the regressor
  // pruned the entire block -- the signature of a descriptor block that
  // collapsed (e.g. a fixed CGRIDMB outside the data's support). A block
  // that holds a NaN or infinite weight has no aggregates and is not
  // checked: the NaN/Inf warning above is the one that describes it.
  for (std::size_t b = 0; b < rec.block_max_abs.size(); ++b) {
    if (b >= layout.blocks.size()) break;
    if (b < rec.block_has_nan_inf.size() && rec.block_has_nan_inf[b])
      continue;
    if (rec.block_max_abs[b] == 0.0) {
      maybe_warn(b, "max_abs",
                 "all weights in this block are exactly zero -- the "
                 "descriptor block contributed nothing to the fit.");
    }
    // Very low non-zero fraction -- almost-empty block.
    if (rec.block_nz_fraction[b] < 0.05 &&
        rec.block_max_abs[b] > 0.0) {
      maybe_warn(b, "nz_fraction",
                 "fewer than 5% of weights in this block exceed 1e-15.");
    }
  }
}

std::size_t IterHealthMonitor::warning_count(
    std::size_t block_idx, const std::string& kind) const {
  const std::string key =
      std::to_string(block_idx) + ":" + kind;
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = warning_counts_.find(key);
  if (it == warning_counts_.end()) return 0;
  return it->second;
}

void IterHealthMonitor::clear_state() {
  std::lock_guard<std::mutex> lk(mtx_);
  warning_counts_.clear();
  last_warned_iter_.clear();
}

}  // namespace hpo
}  // namespace tadah

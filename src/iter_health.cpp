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
  rec.weights_have_nan_inf = false;

  const std::size_t nblocks = layout.blocks.size();
  rec.block_max_abs   .assign(nblocks, 0.0);
  rec.block_median_abs.assign(nblocks, 0.0);
  rec.block_nz_fraction.assign(nblocks, 0.0);

  // Reusable scratch -- declared outside the loop to avoid reallocation.
  std::vector<double> scratch;

  for (std::size_t b = 0; b < nblocks; ++b) {
    const auto& blk = layout.blocks[b];
    const std::size_t lo = blk.col_offset;
    const std::size_t hi = lo + blk.col_count;
    if (hi > weights.size()) {
      // Layout claims more columns than weights has -- record a NaN
      // sentinel and continue. The pre-flight audit handles the
      // structural check; here we just don't crash.
      rec.weights_have_nan_inf = true;
      continue;
    }
    if (blk.col_count == 0) continue;

    scratch.clear();
    scratch.reserve(blk.col_count);
    double mx = 0.0;
    std::size_t nz = 0;
    for (std::size_t c = lo; c < hi; ++c) {
      const double w = weights[c];
      if (!std::isfinite(w)) rec.weights_have_nan_inf = true;
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
  if (mode_ == HealthLogMode::Off) return;

  auto maybe_warn = [&](std::size_t block_idx,
                        const std::string& kind,
                        const std::string& text) {
    const std::string key =
        std::to_string(block_idx) + ":" + kind;
    auto& count = warning_counts_[key];
    auto& last  = last_warned_iter_[key];
    if (mode_ == HealthLogMode::Summary && count > 0 &&
        rec.iter < last + window_) {
      return;
    }
    ++count;
    last = rec.iter;
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::WARNING,
        "iter_health: eval " + std::to_string(rec.iter) +
        " block[" + std::to_string(block_idx) + "/" +
        layout.blocks[block_idx].descriptor_label + "] " +
        kind + " -- " + text);
  };

  // NaN/Inf is the only globally-emitted warning (no per-block index).
  if (rec.weights_have_nan_inf) {
    const std::string key = "*:nan_inf";
    auto& count = warning_counts_[key];
    auto& last  = last_warned_iter_[key];
    if (!(mode_ == HealthLogMode::Summary && count > 0 &&
          rec.iter < last + window_)) {
      ++count;
      last = rec.iter;
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::WARNING,
          "iter_health: eval " + std::to_string(rec.iter) +
          " -- weights contain NaN/Inf entries (training likely diverged).");
    }
  }

  // Per-block exact-zero blocks. block_max_abs == 0 means the regressor
  // pruned the entire block -- the signature of a descriptor block that
  // collapsed (e.g. a fixed CGRIDMB outside the data's support).
  for (std::size_t b = 0; b < rec.block_max_abs.size(); ++b) {
    if (b >= layout.blocks.size()) break;
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
  auto it = warning_counts_.find(key);
  if (it == warning_counts_.end()) return 0;
  return it->second;
}

void IterHealthMonitor::clear_state() {
  warning_counts_.clear();
  last_warned_iter_.clear();
}

}  // namespace hpo
}  // namespace tadah

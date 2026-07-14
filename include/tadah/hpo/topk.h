#pragma once
#ifndef TADAH_HPO_TOPK_H
#define TADAH_HPO_TOPK_H

/* Top-K candidate buffer insert with basin dedup.
 *
 * A low-temperature explorer (e.g. ANNEAL after reset-to-best) revisits the
 * neighbourhood of its best many times; without dedup the near-identical
 * clones evict all diversity and a staged refinement multistart receives K
 * copies of one seed. Two entries are treated as the SAME basin only when
 * BOTH hold:
 *   - losses agree to rel 1e-6, AND
 *   - parameters agree: max_d |dx_d| / width_d <= 1e-3 (normalised by the
 *     OPTIM box widths). With no widths available the parameter test is
 *     skipped (loss-only dedup -- conservative fallback).
 * Distinct basins that happen to share a loss value are both kept.
 *
 * Kept as a free function so it is unit-testable without an HPO_Host. */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace tadah {
namespace hpo {

template <class P>
inline bool topk_same_basin(const double loss_a, const P& params_a,
                            const double loss_b, const P& params_b,
                            const std::vector<double>& widths,
                            const double loss_rel = 1e-6,
                            const double param_rel = 1e-3) {
  const double eps = loss_rel * std::max(1.0, std::fabs(loss_b));
  if (std::fabs(loss_a - loss_b) > eps) return false;
  if (widths.empty()) return true;   // no scale info: loss-only dedup
  const std::size_t n = std::min<std::size_t>(
      static_cast<std::size_t>(params_a.size()), widths.size());
  for (std::size_t d = 0; d < n; ++d) {
    const double w = widths[d] > 0.0 ? widths[d] : 1.0;
    if (std::fabs(params_a(static_cast<long>(d)) -
                  params_b(static_cast<long>(d))) > param_rel * w)
      return false;
  }
  return true;
}

/* Consider (loss, params) for the buffer. Mutates buf in place. */
template <class P>
inline void topk_consider(std::vector<std::pair<double, P>>& buf,
                          const std::size_t capacity,
                          const double loss, const P& params,
                          const std::vector<double>& widths) {
  if (capacity == 0) return;
  auto similar = std::find_if(buf.begin(), buf.end(), [&](const auto& e) {
    return topk_same_basin(loss, params, e.first, e.second, widths);
  });
  if (similar != buf.end()) {
    if (loss < similar->first) *similar = std::make_pair(loss, params);
    return;
  }
  if (buf.size() < capacity) {
    buf.emplace_back(loss, params);
    return;
  }
  auto worst = std::max_element(buf.begin(), buf.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });
  if (loss < worst->first) *worst = std::make_pair(loss, params);
}

}  // namespace hpo
}  // namespace tadah
#endif

#pragma once
/*
 * corana_census -- reporting the PER-DIMENSION acceptance that the Corana
 * kernel already measures.
 *
 * Corana's step adaptation has always been per dimension: adjust_corana_step
 * divides or multiplies V[i] from THAT coordinate's own nA[i]/nP[i], never
 * from a rung-wide scalar.  What was missing was visibility.  The counters
 * feeding the adaptation are an evidence WINDOW -- corana_sweep/corana_rung
 * consume them every NS sweeps and zero them in place -- so anything reading
 * them afterwards sees at most the tail of one window, and the journal only
 * ever carried the rung-wide accept% next to a V_floor/V_cap tally.  A run
 * could therefore hold its ladder-wide acceptance at a healthy 30% while
 * three hundred coordinates never moved at all, and nothing said so.
 *
 * ChainState::corana_census_nA/nP are a parallel, never-reset copy of the
 * same events; this header turns them into something a person can read.
 *
 * PER-SLOT, NOT PER-WALKER.  An ANNEAL_PT exchange swaps x and f between two
 * slots and leaves V -- and these counters -- where they are.  So a row here
 * is "how did dimension i behave at THIS TEMPERATURE", aggregated over every
 * walker that visited the slot.  It is not a property of any one walker's
 * trajectory, and the journal says so wherever it prints one.
 *
 * The kernel below is pure (counters in, numbers and strings out) so the
 * percentile and tie-breaking rules can be tested without a driver, an MPI
 * world or an objective function.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <tadah/hpo/log_index.h>

namespace tadah {
namespace hpo {
namespace corana_census {

/* One (slot, dimension) cell. */
struct DimAcceptance {
  std::size_t accepted = 0;
  std::size_t proposed = 0;
  double sigma = 0.0;   /* the slot's adapted Corana half-width V[i] */

  double rate() const {
    return proposed > 0 ? static_cast<double>(accepted) /
                          static_cast<double>(proposed)
                        : 0.0;
  }
};

/* The seven order statistics printed per slot. */
enum : std::size_t { kP0, kP10, kP25, kP50, kP75, kP90, kP100, kNumPct };
inline const int* percentile_points()
{
  static const int pts[kNumPct] = {0, 10, 25, 50, 75, 90, 100};
  return pts;
}

constexpr std::size_t kExtremes = 5;   /* how many dims to name at each end */

struct SlotCensus {
  std::size_t dims       = 0;
  std::size_t visited    = 0;  /* proposed > 0                              */
  std::size_t unvisited  = 0;  /* proposed == 0: never offered a move       */
  std::size_t frozen     = 0;  /* visited but accepted == 0                 */
  std::size_t saturated  = 0;  /* visited and accepted == proposed          */
  std::size_t accepted   = 0;
  std::size_t proposed   = 0;
  double pct[kNumPct] = {0, 0, 0, 0, 0, 0, 0};   /* over VISITED dims only */
  /* Zero-based dimension indices, ascending rate then ascending index.
   * `lowest` is truncated to kExtremes from the front, `highest` from the
   * back (so `highest` reads high-to-low).  The two are DISJOINT: with
   * fewer than 2*kExtremes visited dimensions the naive pair is the same
   * set printed twice in opposite orders, which at dim = 2 rendered as
   * "lowest 5 = SGRIDMB 40.6%, RCUT2B 53.1% | highest 5 = RCUT2B 53.1%,
   * SGRIDMB 40.6%" -- two wrong counts and no new information. Anything
   * already named in `lowest` is dropped from `highest`, so `highest` may
   * be shorter than `lowest` or empty, and each list's size is the number
   * of dimensions it actually names. */
  std::vector<std::size_t> lowest, highest;

  double rate() const {
    return proposed > 0 ? static_cast<double>(accepted) /
                          static_cast<double>(proposed)
                        : 0.0;
  }
};

/* Nearest-rank percentile on the sorted sample: the p-th percentile is the
 * element at ceil(p/100 * n), 1-based, clamped into range.  Chosen over an
 * interpolating definition because these ARE integer ratios of small counts
 * (a dimension offered 20 moves can only report 0/20, 1/20, ...): every
 * value printed is then a value some dimension actually had, and p0/p100
 * are exactly the min and max. */
inline double nearest_rank(const std::vector<double>& sorted, int p)
{
  if (sorted.empty()) return 0.0;
  const double n = static_cast<double>(sorted.size());
  long idx = static_cast<long>(std::ceil(p / 100.0 * n)) - 1;
  if (idx < 0) idx = 0;
  if (idx >= static_cast<long>(sorted.size()))
    idx = static_cast<long>(sorted.size()) - 1;
  return sorted[static_cast<std::size_t>(idx)];
}

/* Summarise one slot.  Dimensions that were never proposed on are excluded
 * from the percentiles -- their rate() is 0 by convention, and letting a
 * budget-truncated sweep's unvisited tail masquerade as "0% acceptance"
 * would turn a bookkeeping artefact into a diagnosis.  They are counted
 * separately instead. */
inline SlotCensus summarize(const std::vector<DimAcceptance>& dims)
{
  SlotCensus c;
  c.dims = dims.size();
  std::vector<std::pair<double, std::size_t>> ranked;
  ranked.reserve(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    const auto& d = dims[i];
    c.accepted += d.accepted;
    c.proposed += d.proposed;
    if (d.proposed == 0) { ++c.unvisited; continue; }
    ++c.visited;
    if (d.accepted == 0) ++c.frozen;
    if (d.accepted == d.proposed) ++c.saturated;
    ranked.emplace_back(d.rate(), i);
  }
  /* Ties are the normal case (hundreds of dims at exactly 0/n), so the
   * secondary key matters: without it the named extremes would depend on
   * the sort's internal ordering and change between runs that behaved
   * identically. */
  std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<double, std::size_t>& a,
               const std::pair<double, std::size_t>& b) {
              if (a.first != b.first) return a.first < b.first;
              return a.second < b.second;
            });
  std::vector<double> rates;
  rates.reserve(ranked.size());
  for (const auto& r : ranked) rates.push_back(r.first);
  const int* pts = percentile_points();
  for (std::size_t q = 0; q < kNumPct; ++q)
    c.pct[q] = nearest_rank(rates, pts[q]);
  const std::size_t take = std::min(kExtremes, ranked.size());
  for (std::size_t j = 0; j < take; ++j) c.lowest.push_back(ranked[j].second);
  for (std::size_t j = 0; j < take && j + take < ranked.size(); ++j)
    c.highest.push_back(ranked[ranked.size() - 1 - j].second);
  return c;
}

/* "RCUT2B[1] 0.0%" -- a named dimension next to its rate. */
template <class LabelFn>
std::string format_dim_list(const std::vector<std::size_t>& idx,
                            const std::vector<DimAcceptance>& dims,
                            LabelFn&& label)
{
  std::ostringstream o;
  for (std::size_t j = 0; j < idx.size(); ++j) {
    if (j) o << ", ";
    o << label(idx[j]) << ' ' << std::fixed << std::setprecision(1)
      << 100.0 * dims[idx[j]].rate() << '%';
  }
  return o.str();
}

/* One fixed-width row of the end-of-run per-slot table. */
inline std::string format_slot_row(std::size_t slot0, double T,
                                   const SlotCensus& c)
{
  std::ostringstream o;
  o << std::setw(5) << logidx::uis(slot0) << "  " << std::scientific
    << std::setprecision(3) << T << std::setw(8) << c.dims
    << std::setw(9) << c.visited << std::setw(8) << c.frozen
    << std::setw(8) << c.saturated;
  o << std::fixed << std::setprecision(1);
  for (std::size_t q = 0; q < kNumPct; ++q)
    o << std::setw(8) << (100.0 * c.pct[q]);
  return o.str();
}

inline std::string slot_table_header()
{
  std::ostringstream o;
  o << std::setw(5) << "slot" << "  " << std::setw(9) << "T"
    << std::setw(8) << "dims" << std::setw(9) << "visited"
    << std::setw(8) << "frozen" << std::setw(8) << "satur"
    << std::setw(8) << "min%" << std::setw(8) << "p10%"
    << std::setw(8) << "p25%" << std::setw(8) << "p50%"
    << std::setw(8) << "p75%" << std::setw(8) << "p90%"
    << std::setw(8) << "max%";
  return o.str();
}

}  // namespace corana_census
}  // namespace hpo
}  // namespace tadah

#ifndef TADAH_HPO_BUMP_SUPPORT_GATE_H
#define TADAH_HPO_BUMP_SUPPORT_GATE_H

// Per-candidate bump-support gate (BUMPGATE CANDIDATE, the default).
//
// On a hard (Cut_Dummy) cutoff channel every compact bump must satisfy
//   c + w/eta <= RCUT
// or the truncated overhang is an energy/force STEP at the cutoff. The
// legacy startup gate (BUMPGATE STRICT) enforces this on the OPTIM box
// CORNER (CGRID high bound, SGRID low bound) — exact per index, since
// reach(c, eta) = c + w/eta is monotone in both — but rejects the whole
// box when only the corner region is infeasible, forcing users to shrink
// boxes until near-cutoff bumps degenerate into spikes.
//
// CANDIDATE mode instead allows the full boxes: startup errors only when a
// box contains NO feasible point, and this header supplies the per-
// evaluation test HPO_Host::operator() runs BEFORE any training work.
// Violating candidates return a graded band-D failure (rankable — the
// annealer keeps a slope back toward feasibility — but never publishable
// as best). The prebuilt NN list needs no padding: RCUT stays
// authoritative on bump channels and nn_safe_rcutmax covers every OPTIM
// RCUT high bound, so any candidate this gate passes is inside the list.
//
// The table is built ONCE (HPO_Host ctor) and arms only bumps whose box
// can actually violate — a config that passes the STRICT corner check
// arms nothing and evaluates byte-identically.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/models/descriptors/basis_kind.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace hpo {
namespace detail {

/** \brief Effective truncation cutoff of one channel (RCUT{sfx}MAX
 *  semantics: the cutoff object truncates at the max over the per-pair
 *  RCUT list). Non-OPT'd entries contribute their config value to
 *  fixed_max; OPT'd entries contribute their sampled value at eval time
 *  (flat_idx) and their bounds to the reachable range. */
struct BindingRcut {
  /* max over non-OPT'd RCUT entries; -inf when EVERY entry is OPT'd */
  double fixed_max = -std::numeric_limits<double>::infinity();
  /* flat theta indices of OPT'd RCUT{sfx} entries */
  std::vector<long> flat_idx;
  /* worst / best cutoff any candidate can carry:
   *   min_reachable = max(fixed_max, max_j lowbound[j])
   *   max_reachable = max(fixed_max, max_j highbound[j]) */
  double min_reachable = -std::numeric_limits<double>::infinity();
  double max_reachable = -std::numeric_limits<double>::infinity();
};

inline BindingRcut derive_binding_rcut(const core::Context &ctx,
                                       const HPOTargets &trg,
                                       const std::string &sfx) {
  BindingRcut rc;
  const std::string rkey = std::string("RCUT") + sfx;
  if (!ctx.contains(rkey) || ctx.size(rkey) == 0) return rc;

  core::v_type rvals;
  try {
    models::D_Base::get_grid(ctx, rkey, rvals);
  } catch (const std::exception &) {
    rvals.clear();
  }
  if (rvals.empty()) rvals.push_back(ctx.get<double>(rkey));

  /* the OPTIM entry of rkey; keys compare case-insensitively, as the
   * Context compares them (the density gates' reach shares this range) */
  auto itr = trg.optimap.find(rkey);
  if (itr == trg.optimap.end())
    itr = std::find_if(trg.optimap.begin(), trg.optimap.end(),
                       [&rkey](const auto &kv) {
                         return core::to_upper(kv.first) == rkey;
                       });
  const bool has_opt = itr != trg.optimap.end();

  /* flat theta indices come from the one shared flattening contract */
  const auto prefs = make_param_ref_table(trg);

  for (std::size_t i = 0; i < rvals.size(); ++i) {
    const bool opt = has_opt && i < itr->second.opt.size() &&
                     itr->second.opt[i];
    if (opt) {
      for (const auto &r : prefs)
        if (r.theta_name == itr->first && r.arg_pos == static_cast<int>(i + 1))
          rc.flat_idx.push_back(static_cast<long>(r.flat_idx));
      rc.min_reachable = std::max(rc.min_reachable, itr->second.lowbound[i]);
      rc.max_reachable = std::max(rc.max_reachable, itr->second.highbound[i]);
    } else {
      rc.fixed_max = std::max(rc.fixed_max, rvals[i]);
    }
  }
  rc.min_reachable = std::max(rc.min_reachable, rc.fixed_max);
  rc.max_reachable = std::max(rc.max_reachable, rc.fixed_max);
  return rc;
}

/** \brief One hard-cutoff bump channel with the bumps that can violate
 *  c + w/eta <= rcut for SOME candidate in the OPTIM boxes. */
struct BumpGateChannel {
  std::string ckey, skey, rkey;  /* "CGRID2B"/"SGRID2B"/"RCUT2B" or MB */
  double w = 0.0;                /* half-width units: Blip 2, Blip5 3 */
  BindingRcut rcut;
  struct Bump {
    long c_flat = -1;            /* flat theta idx of the centre, or -1 */
    double c_fix = 0.0;          /*   fixed (seed) centre when c_flat < 0 */
    long s_flat = -1;            /* flat theta idx of eta, or -1 */
    double s_fix = 0.0;          /*   fixed (seed) eta when s_flat < 0 */
    std::size_t index = 0;       /* 0-based grid index (diagnostics) */
  };
  std::vector<Bump> bumps;
};

/** \brief Build the per-candidate gate table. Empty result = nothing can
 *  ever violate (smooth cutoffs, non-bump bases, or fully-feasible
 *  boxes) — the caller skips the per-eval check entirely. Bumps that are
 *  fully static (centre, eta AND every RCUT entry non-OPT'd) are the seed
 *  check's job and are never armed. */
inline std::vector<BumpGateChannel>
build_bump_gate_table(const core::Context &ctx, const HPOTargets &trg) {
  std::vector<BumpGateChannel> tbl;
  const auto prefs = make_param_ref_table(trg);
  auto flat_of = [&prefs](const std::string &key, std::size_t i) -> long {
    for (const auto &r : prefs)
      if (r.theta_name == key && r.arg_pos == static_cast<int>(i + 1))
        return static_cast<long>(r.flat_idx);
    return -1;
  };

  for (const char *sfx : {"2B", "MB"}) {
    const auto wopt = models::potedit::bump_channel_halfwidth(ctx, sfx);
    if (!wopt) continue;
    if (!models::potedit::channel_cutoff_is_discontinuous(ctx, sfx))
      continue;  /* smooth cutoff: overhang is damped to zero — no gate */
    BumpGateChannel ch;
    ch.w = *wopt;
    ch.ckey = std::string("CGRID") + sfx;
    ch.skey = std::string("SGRID") + sfx;
    ch.rkey = std::string("RCUT") + sfx;
    if (!ctx.contains(ch.ckey) || !ctx.contains(ch.skey) ||
        !ctx.contains(ch.rkey))
      continue;
    core::v_type mu, eta;
    try {
      models::D_Base::get_grid(ctx, ch.ckey, mu);
      models::D_Base::get_grid(ctx, ch.skey, eta);
    } catch (const std::exception &) { continue; }
    if (mu.empty() || mu.size() != eta.size()) continue;

    ch.rcut = derive_binding_rcut(ctx, trg, sfx);
    if (!std::isfinite(ch.rcut.min_reachable)) continue;
    const double tol = models::bump_rcut_tol(ch.rcut.min_reachable);

    const auto itc = trg.optimap.find(ch.ckey);
    const auto its = trg.optimap.find(ch.skey);
    for (std::size_t i = 0; i < mu.size(); ++i) {
      BumpGateChannel::Bump b;
      b.index = i;
      b.c_fix = mu[i];
      b.s_fix = eta[i];
      const bool c_opt = itc != trg.optimap.end() &&
                         i < itc->second.opt.size() && itc->second.opt[i];
      const bool s_opt = its != trg.optimap.end() &&
                         i < its->second.opt.size() && its->second.opt[i];
      if (c_opt) b.c_flat = flat_of(ch.ckey, i);
      if (s_opt) b.s_flat = flat_of(ch.skey, i);
      if (b.c_flat < 0 && b.s_flat < 0 && ch.rcut.flat_idx.empty())
        continue;  /* fully static: the startup seed check owns this bump */

      /* Arm only when the WORST corner (max centre, min eta, min rcut)
       * violates — otherwise no candidate in the boxes can spill and the
       * per-eval check would be dead weight. */
      const double c_hi = b.c_flat >= 0 ? itc->second.highbound[i] : b.c_fix;
      double eta_lo = b.s_flat >= 0 && its->second.lowbound[i] > 0.0
                          ? its->second.lowbound[i]
                          : b.s_fix;
      if (!(eta_lo > 0.0)) continue;  /* startup parity: eta <= 0 skipped */
      if (c_hi + ch.w / eta_lo <= ch.rcut.min_reachable + tol) continue;
      ch.bumps.push_back(b);
    }
    if (!ch.bumps.empty()) tbl.push_back(std::move(ch));
  }
  return tbl;
}

/** \brief The per-candidate test. Returns 0 when every armed bump is
 *  feasible at p; otherwise the summed scale-free overhang
 *  sum_violating (c + w/eta - rcut) / (w/eta) — additive across bumps and
 *  strictly monotone in every violating coordinate, so within the graded
 *  band any move that shrinks any violation is strictly downhill.
 *  p is the REAL-space theta vector (optimizers call host(sp.out(x))). */
template <class ParamVec>
inline double bump_gate_violation(const std::vector<BumpGateChannel> &tbl,
                                  const ParamVec &p) {
  double v = 0.0;
  for (const auto &ch : tbl) {
    double rcut = ch.rcut.fixed_max;
    for (long j : ch.rcut.flat_idx) rcut = std::max(rcut, p(j));
    const double tol = models::bump_rcut_tol(rcut);
    for (const auto &b : ch.bumps) {
      const double c = b.c_flat >= 0 ? p(b.c_flat) : b.c_fix;
      const double eta = b.s_flat >= 0 ? p(b.s_flat) : b.s_fix;
      if (!(eta > 0.0)) continue;
      const double reach = c + ch.w / eta;
      if (reach > rcut + tol) v += (reach - rcut) * eta / ch.w;
    }
  }
  return v;
}

} // namespace detail
} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_BUMP_SUPPORT_GATE_H

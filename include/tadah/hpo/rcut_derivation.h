#ifndef TADAH_HPO_RCUT_DERIVATION_H
#define TADAH_HPO_RCUT_DERIVATION_H

// F2 support for knot-family refit potentials inside the HPO loop.
//
// A knot channel's basis vanishes C^4-smoothly at its largest knot, so its
// effective cutoff IS max(knot centre) and the RCUT2B/RCUTMB keys are
// derived quantities:
//  * per evaluation (update_context hook) the keys are re-derived from the
//    CURRENT grid values, so OPTIM'd knot positions move the cutoff with
//    them (the evaluation's Trainer is built from that context);
//  * the ONE-TIME neighbour build must be sized for the largest cutoff any
//    evaluation can reach — i.e. also the OPTIM upper bounds of the knot
//    grids, which the legacy RCUT-only scan missed (an OPTIM'd knot pushed
//    beyond the stated RCUT would silently outrun the prebuilt NN list).
// Bump (blip) channels are never touched: RCUT is authoritative there and
// never auto-extended. Their counterpart protection is the bump-support
// GATE — under a hard Cut_Dummy cutoff a truncated overhang (c + w/eta
// past RCUT) is an energy/force step at the cutoff. A SEED whose bumps
// spill is a hard startup error in every mode (validate_hpo_setup). OPTIM
// boxes: BUMPGATE STRICT hard-errors on a violating box corner (legacy);
// BUMPGATE CANDIDATE (default) errors only on a box with NO feasible
// point and rejects violating trials per evaluation BEFORE training
// (bump_support_gate.h, graded band-D). Either way no spilling trial is
// ever trained, and the prebuilt NN list needs no bump-side padding:
// nn_safe_rcutmax below covers every OPTIM RCUT high bound, so any in-box
// candidate the gate passes is inside the list (a candidate outside the
// box is not evaluated: box_penalty.h).

#include <cmath>
#include <string>

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace hpo {
namespace detail {

/** \brief Re-derive RCUT2B/RCUTMB from the CURRENT knot grids. Called from
 *  HPO_Host::update_context on every evaluation, so it is deliberately
 *  silent (the one-time user-facing explanation is logged by
 *  validate_hpo_setup at startup) and cheap (a max over the grid; no-op
 *  add/remove churn is avoided). The comparison is EXACT: Context text is
 *  lossless, so an unchanged cutoff compares equal, while any tolerance
 *  would make the cutoff depend on the evaluation history of the
 *  (persistent, per-thread) trainer context rather than on theta alone.
 *  Returns true when a key changed — Context::add refreshes
 *  RCUT2BMAX/RCUTMBMAX/RCUTMAX automatically. */
inline bool apply_knot_rcut_derivation(core::Context &ctx, bool knot2b,
                                       bool knotmb) {
  bool changed = false;
  auto apply = [&](const char *sfx) {
    const auto d = models::potedit::derived_knot_rcut(ctx, sfx);
    if (!d) return;
    const std::string key = std::string("RCUT") + sfx;
    if (ctx.contains(key) && ctx.size(key) > 0 &&
        ctx.get<double>(key) == *d)
      return;
    ctx.remove(key);
    ctx.add(key, *d);
    changed = true;
  };
  if (knot2b) apply("2B");
  if (knotmb) apply("MB");
  return changed;
}

/** \brief Cutoff for the ONE-TIME neighbour-list build: the stated RCUTMAX,
 *  any OPTIM RCUT2B/RCUTMB upper bounds (legacy behaviour), and — for knot
 *  channels — the OPTIM CGRID2B/CGRIDMB upper bounds (a knot moved to its
 *  bound IS the new effective cutoff and the prebuilt list must cover it). */
inline double nn_safe_rcutmax(const core::Context &master,
                              const HPOTargets &trg) {
  double rcutmax = master.get<double>("RCUTMAX");
  auto scan = [&](const std::string &key) {
    const auto it = trg.optimap.find(key);
    if (it == trg.optimap.end()) return;
    for (std::size_t i = 0; i < it->second.opt.size(); ++i)
      if (it->second.opt[i] && it->second.highbound[i] > rcutmax)
        rcutmax = it->second.highbound[i];
  };
  scan("RCUT2B");
  scan("RCUTMB");
  if (models::potedit::derived_knot_rcut(master, "2B")) scan("CGRID2B");
  if (models::potedit::derived_knot_rcut(master, "MB")) scan("CGRIDMB");
  return rcutmax;
}

} // namespace detail
} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_RCUT_DERIVATION_H

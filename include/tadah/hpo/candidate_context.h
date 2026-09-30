#ifndef TADAH_HPO_CANDIDATE_CONTEXT_H
#define TADAH_HPO_CANDIDATE_CONTEXT_H

// The context a candidate parameter vector is evaluated in.
//
// HPO_Host::update_context writes every candidate into the model context
// through update_candidate_context below. Startup code that must see a seed
// exactly as the host will evaluate it (for example a check on the density
// of each CONFIG/WARM seed) builds its context with the same function, from
// a copy of the training configuration, so the two cannot drift.

#include <cstddef>

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>

namespace tadah {
namespace hpo {
namespace detail {

/** \brief Write the candidate @p p into @p ctx.
 *
 *  Every OPTIM key of @p trg is removed and re-added token by token: the
 *  OPTIM'd tokens from @p p, in the optimap traversal order the search space
 *  and params.tadah use, the others from the -c configuration
 *  (Opti::config). Context text is lossless, so the model evaluates the
 *  candidate's exact doubles. Then the knot channels' RCUT2B/RCUTMB are
 *  re-derived from the (possibly OPTIM'd) largest knot (@p knot2b,
 *  @p knotmb; rcut_derivation.h), which refreshes RCUT*MAX.
 *
 *  @p P is a vector indexed p(i) (the host's dlib column vector) holding
 *  one value per OPTIM dimension. */
template <class P>
void update_candidate_context(core::Context &ctx, const HPOTargets &trg,
                              const P &p, bool knot2b, bool knotmb) {
  for (const auto &kv : trg.optimap) ctx.remove(kv.first);

  std::size_t idx = 0;
  for (const auto &kv : trg.optimap) {
    for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
      if (kv.second.opt[i]) ctx.add(kv.first, p(idx++));
      else                  ctx.add(kv.first, kv.second.config[i]);
    }
  }

  /* Knot channels: the effective cutoff follows the (possibly OPTIM'd)
   * largest knot -- re-derive RCUT2B/RCUTMB so Trainer::reinit applies it
   * via fcut->set_rcut(RCUT*MAX). Overrides a user OPTIM RCUT* dimension
   * deliberately (warned once at startup). The prebuilt NN list covers
   * every cutoff inside the OPTIM box (detail::nn_safe_rcutmax); the host
   * rebuilds the lists for a candidate beyond it (HPO_Host::rebuilt_lists). */
  apply_knot_rcut_derivation(ctx, knot2b, knotmb);
}

} // namespace detail
} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_CANDIDATE_CONTEXT_H

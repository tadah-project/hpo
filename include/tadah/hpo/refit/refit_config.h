#ifndef TADAH_HPO_REFIT_REFIT_CONFIG_H
#define TADAH_HPO_REFIT_REFIT_CONFIG_H

/** \file refit_config.h
 *  \brief Parsed representation of the `tadah refit` stage pipeline
 *  (INIT -> EXPLORE -> REFINE -> FUSE), produced by RefitParser and consumed by
 *  HPORefitEngine::compact_fit (which flattens it into CompactConfig + per-
 *  channel INIT seeds). Each stage's optimiser is an OptimizerSpec, parsed with
 *  the shared optimizer-block reader so the syntax matches `tadah hpo`.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <array>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {
namespace refit {

/** How an INIT basis position is placed. Manual = user gives mu (+eta). The
 *  others are resolved at fit time against the channel's fit window / target:
 *  XMin/XMax = window edges, YMin/YMax = target extremum location, SMax/CMax =
 *  max |slope| / |curvature| location. (XMax is a no-op for the LEFT-active
 *  phi/rho channels — they vanish at the cutoff — and is skipped there.) */
enum class PinMode { Manual, XMin, XMax, YMin, YMax, SMax, CMax };

/** One user-declared INIT basis on one channel. */
struct InitBasis {
  enum Channel { PHI, RHO, F } channel = PHI;
  PinMode placement = PinMode::Manual;
  double mu = 0.0;          ///< used when placement==Manual; else resolved
  double eta = 0.0;         ///< blip width (Manual only; knots ignore eta)
  bool has_eta = false;     ///< distinguishes "MANUAL mu" from "MANUAL mu eta"
  bool pinned = false;      ///< freeze the position in EXPLORE/REFINE/FUSE
};

/** EXPLORE stage: structure search + growth (absorbs the old GLOBAL stage at
 *  rung 0). opt.algo/opt.inner select the global outer + per-rung local.
 *  `enabled` is the explicit ON/OFF toggle (ENABLED true|false); EXPLORE cannot
 *  be disabled (it builds the basis ladder) — the parser rejects ENABLED false. */
struct ExploreStage {
  /** Max basis ADDED per function, indexed [phi, rho, F]; -1 => derive from
   *  NMAX (no add-cap). A single MAX_ADD value fills all three; three values
   *  set them per-function. Only EXPLORE adds basis (REFINE/FUSE just refine
   *  the existing basis), so this lives on the EXPLORE stage only. */
  std::array<int, 3> max_add = {-1, -1, -1};
  OptimizerSpec opt;        ///< OPTIMIZER block (outer + INNER)
  bool has_opt = false;
  bool enabled = true;      ///< ENABLED true|false (must be true for EXPLORE)
};

/** REFINE stage: per-rung Ceres LM/DOGLEG (or NLOPT/DLIB local) refine.
 *  `enabled` false (ENABLED false) skips the stage (engine maps it to ALGO
 *  NONE); combining ENABLED false with a non-NONE optimiser is a contradiction
 *  the parser rejects. */
struct RefineStage {
  OptimizerSpec opt;
  bool has_opt = false;
  bool enabled = true;      ///< ENABLED true|false
};

/** FUSE stage: final per-function curve refine (curve mode) / joint Ceres
 *  least-squares (energy mode), or NONE. `enabled` as for RefineStage. */
struct FuseStage {
  OptimizerSpec opt;
  bool has_opt = false;
  bool enabled = true;      ///< ENABLED true|false
};

/** The whole parsed pipeline. Any block omitted from the config leaves its
 *  member default-constructed (has_opt=false / empty init), and the engine
 *  applies the built-in CompactConfig defaults. */
struct RefitConfig {
  std::vector<InitBasis> init;
  ExploreStage explore;
  RefineStage refine;
  FuseStage fuse;
};

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_REFIT_REFIT_CONFIG_H

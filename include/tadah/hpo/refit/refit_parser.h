#ifndef TADAH_HPO_REFIT_REFIT_PARSER_H
#define TADAH_HPO_REFIT_REFIT_PARSER_H

/** \file refit_parser.h
 *  \brief Parser for the `tadah refit` stage pipeline. CORE's config parser
 *  captures each stage block (INIT/EXPLORE/REFINE/FUSE) verbatim into a Context
 *  STRINGS key; RefitParser turns those flat token streams into a RefitConfig.
 *  The OPTIMIZER sub-blocks reuse the shared optimizer-block reader (so the
 *  syntax matches `tadah hpo --hpotarget`); the legal ALGO set per stage is
 *  validated/mapped by the refit-local helpers below.
 */

#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/refit/refit_config.h>

#include <string>

namespace tadah {
namespace core {
class Context;
}
namespace hpo {
namespace refit {

class RefitParser {
public:
  /** Parse the INIT/EXPLORE/REFINE/FUSE Context keys into a RefitConfig.
   *  Throws std::runtime_error on the old grammar (a top-level OPTIMIZER
   *  block: MLIP's top_level_optimizer_refusal(), the text RefitEngine's
   *  mode rule refuses it with in either REFIT_MODE) and on invalid tokens. Missing blocks => default-constructed stages. A
   *  stage's OPTIMIZER block (and EXPLORE's INNER block) accepts only the keys
   *  that stage reads -- any other optimiser attribute is an error naming the
   *  keys it does read -- and must set ALGO. A stop criterion (MAXEVAL,
   *  MAXITER, FTOL_REL, FTOL_ABS, XTOL_REL, GTOL) the block's ALGO does not
   *  read is an error too, naming the keys it reads: "REFINE: GTOL has no
   *  effect with ALGO LD_LBFGS (it reads MAXEVAL, FTOL_REL); remove GTOL";
   *  with ALGO NONE (or UNIFORM, EXPLORE's), which runs no optimiser, "REFINE:
   *  MAXITER has no effect with ALGO NONE (the stage is skipped); remove
   *  MAXITER". MAXITER (Ceres iterations) is read by LM and DOGLEG only;
   *  dlib's LBFGS reads FTOL_ABS, not FTOL_REL. EXPLORE INNER's MAXEVAL,
   *  FTOL_REL and XTOL_REL stay accepted under the EXPLORE optimiser
   *  GN_MLSL_LDS, whose local LN_SBPLX searches read them. ENABLED false
   *  with an OPTIMIZER block in REFINE or FUSE is an error. Every tolerance
   *  is a finite number >= 0. */
  static RefitConfig parse(const tadah::core::Context &c);
};

/** REFIT_OBJECTIVE, in any case: true for 'curve', false for 'energy' (the
 *  default without the key; no blueprint default is inserted). Throws
 *  std::runtime_error for anything else. MLIP's
 *  read_refit_curve_objective (refit_keys.h), which the mode rule reads it
 *  with in either REFIT_MODE. The engine sets CompactConfig::objective from
 *  it. */
bool refit_curve_objective(const tadah::core::Context &c);

/** Map a parsed OPTIMIZER spec to the compact fitter's flat algorithm string
 *  for the given stage, validating the (LIB, ALGO) pair against that stage's
 *  legal set. Throw std::runtime_error on an illegal pair. NLOPT long names
 *  (e.g. GN_MLSL_LDS, LN_SBPLX, LD_LBFGS) and the short fitter names (MLSL,
 *  SBPLX, NLOPT_LBFGS) are both accepted. As in `tadah hpo`, a block that
 *  runs an optimiser names the library of its ALGO in LIB (NLOPT, DLIB or
 *  CERES; a missing or different LIB is an error naming the right one, and
 *  the stage's algorithms of the LIB given when it has any), and a block
 *  with ALGO NONE runs none, so LIB there is an error (it would have no
 *  effect). The ALGO names are the refit's own, not those of `tadah hpo`:
 *  its dlib global search is GLOBAL (`tadah hpo`'s DLIB GFS), and LBFGS is
 *  dlib's L-BFGS (a CERES algorithm in `tadah hpo`; NLopt's is LD_LBFGS).
 *  UNIFORM (EXPLORE: refit's own uniform start, no global search) runs no
 *  optimiser either: a LIB with it is an error, as with NONE. */
/** EXPLORE's optimiser also refuses ALGO GN_MLSL, GN_DIRECT and GN_CRS2,
 *  whatever the LIB, naming the algorithm the refit runs in their place
 *  ("EXPLORE: ALGO GN_MLSL is not run by tadah refit (NLopt's MLSL with
 *  pseudo-random starts); its MLSL is GN_MLSL_LDS: set ALGO GN_MLSL_LDS");
 *  before 1.6.0 they ran GN_MLSL_LDS, GN_DIRECT_L and GN_CRS2_LM. */
std::string refit_explore_outer_algo(const OptimizerSpec &s);  ///< -> seed_algo
std::string refit_explore_inner_algo(const OptimizerSpec &s);  ///< -> explore_algo
std::string refit_refine_algo(const OptimizerSpec &s);         ///< -> refine_algo
std::string refit_fuse_algo(const OptimizerSpec &s);           ///< -> joint_algo

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_REFIT_REFIT_PARSER_H

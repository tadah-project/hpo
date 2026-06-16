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
   *  Throws std::runtime_error on the old grammar (a top-level OPTIMIZER block)
   *  and on invalid tokens. Missing blocks => default-constructed stages. */
  static RefitConfig parse(const tadah::core::Context &c);
};

/** Map a parsed OPTIMIZER spec to the compact fitter's flat algorithm string
 *  for the given stage, validating the (LIB, ALGO) pair against that stage's
 *  legal set. Throw std::runtime_error on an illegal pair. NLOPT long names
 *  (e.g. GN_MLSL_LDS, LN_SBPLX, LD_LBFGS) and the short fitter names (MLSL,
 *  SBPLX, NLOPT_LBFGS) are both accepted. */
std::string refit_explore_outer_algo(const OptimizerSpec &s);  ///< -> seed_algo
std::string refit_explore_inner_algo(const OptimizerSpec &s);  ///< -> explore_algo
std::string refit_refine_algo(const OptimizerSpec &s);         ///< -> refine_algo
std::string refit_fuse_algo(const OptimizerSpec &s);           ///< -> joint_algo

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_REFIT_REFIT_PARSER_H

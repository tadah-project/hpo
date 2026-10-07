#ifndef TADAH_HPO_ENGINES_HPO_REFIT_ENGINE_H
#define TADAH_HPO_ENGINES_HPO_REFIT_ENGINE_H

#include <tadah/mlip/engines/refit_engine.h>

namespace tadah {
namespace hpo {
namespace engines {

/** \brief `tadah refit` engine with the compact fitting mode.
 *
 *  Registered over the MLIP "REFIT" entry by registerAllHPOEngines(). Dense
 *  mode delegates to the base class; REFIT_MODE=compact runs the per-function
 *  ladder (NLopt SBPLX/MLSL structure search + analytic-Jacobian Ceres LM
 *  polish) with energy-metric rung selection, followed by a joint refinement
 *  of all three functions against the crystal E-V residuals. Requires the
 *  Ceres-enabled HPO build: without Ceres has_compact_fitter() is false and
 *  the base refuses REFIT_MODE compact before the EAM file is read (it fell
 *  back to dense fitting).
 */
class HPORefitEngine : public tadah::mlip::engines::RefitEngine {
protected:
  bool has_compact_fitter() const override;
  bool compact_fit(const tadah::core::Context &c,
                   const tadah::models::EAMSetflIO &eam,
                   tadah::models::EAMRefitConfig &cfg,
                   std::unique_ptr<tadah::models::EAMRefit> &refit,
                   SelfTest &st) override;
};

}  // namespace engines
}  // namespace hpo
}  // namespace tadah
#endif  // TADAH_HPO_ENGINES_HPO_REFIT_ENGINE_H

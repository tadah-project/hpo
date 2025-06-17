#ifndef TADAH_MLIP_ENGINES_HPO_ENGINE_H
#define TADAH_MLIP_ENGINES_HPO_ENGINE_H

#include <tadah/core/context.h>
#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace hpo {
namespace engines {

/**
 * @brief Provides hyperparameter optimization functionality.
 */
class HPOEngine: public mlip::engines::IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace hpo
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPO_ENGINE_H


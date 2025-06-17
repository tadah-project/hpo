#include <tadah/mlip/engines/engine_factory.h>

#include <tadah/hpo/engines/all_engines.h>
#include <tadah/hpo/engines/hpo_engine.h>

#include <stdexcept>
#include <iostream>

namespace tadah {
namespace hpo {
namespace engines {

void registerAllHPOEngines() {
  using tadah::mlip::engines::EngineFactory;
  EngineFactory::registerEngine("HPO", [](){ return std::make_unique<HPOEngine>(); });
}

} // namespace engines
} // namespace hpo
} // namespace tadah

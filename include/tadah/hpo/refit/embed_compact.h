#pragma once
#include <tadah/models/eam/eam_refit.h>
namespace tadah {
namespace hpo {
std::unique_ptr<models::EAMRefit> compact_embed_fit(const core::Context &,
                                                    const models::EAMSetflIO &,
                                                    models::EAMRefitConfig &);
}
} // namespace tadah

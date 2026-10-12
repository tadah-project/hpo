#pragma once
/* What LIB CERES gives in the build of the tests. TADAH_ENABLE_CERES (CMake,
 * ON by default) builds the Ceres optimisers and defines TADAH_USE_CERES;
 * without it every LIB CERES is refused, before any setting of its block is
 * checked, with kCeresNotBuilt (backend_rules.cpp validate_backend, which
 * the startup checks run for each phase; run_policy_base.h run_inner, when
 * a driver starts). A test that checks a Ceres backend's own validation
 * expects that refusal in a build without Ceres instead: the case then
 * checks what such a build does. */

#include <string>

namespace hpo_test {

inline const std::string kCeresNotBuilt =
    "CERES support is not built; reconfigure with -DTADAH_ENABLE_CERES=ON";

/* `with_ceres` in a build with Ceres; without it the refusal of LIB CERES,
 * after `prefix` (a phase's name and ": ", as validate_phase_backends
 * names the phase it refuses; empty for validate_backend itself). */
inline std::string ceres_expect(const std::string& with_ceres,
                                const std::string& prefix = "")
{
#ifdef TADAH_USE_CERES
    (void)prefix;
    return with_ceres;
#else
    (void)with_ceres;
    return prefix + kCeresNotBuilt;
#endif
}

}  // namespace hpo_test

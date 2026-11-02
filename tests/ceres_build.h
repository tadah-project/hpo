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

/* The tail of the refusal of CERES:<ALGO>, which names the Installation
 * page as the refusal of tadah hpo without the Nested Fitting Procedure
 * does, "(see Installation)" (it said "CERES support is not built;
 * reconfigure with -DTADAH_ENABLE_CERES=ON", then "(Installation) has
 * it"). */
inline const std::string kCeresNotBuilt =
    " is not in this build of tadah; a build with -DTADAH_ENABLE_CERES=ON "
    "(see Installation) has it";

/* `with_ceres` in a build with Ceres; without it the refusal of LIB CERES
 * with ALGO `algo`, after `prefix` (a phase's name and ": ", as
 * validate_phase_backends names the phase it refuses; empty for
 * validate_backend itself). */
inline std::string ceres_expect(const std::string& with_ceres,
                                const std::string& prefix = "",
                                const std::string& algo = "STEEPEST_DESCENT")
{
#ifdef TADAH_USE_CERES
    (void)prefix;
    (void)algo;
    return with_ceres;
#else
    (void)with_ceres;
    return prefix + "CERES:" + algo + kCeresNotBuilt;
#endif
}

}  // namespace hpo_test

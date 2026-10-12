/* NLopt-version tripwire. HPO's NLopt refusals (the NLOPT entries of the
 * refusal table, backend_refusal(), optimizer_caps.cpp) and the NLopt
 * behaviour its registry and tests pin (optimizer_caps.h, optimizer_spec.h,
 * search_space.h, test_optimizer_caps_step.cpp,
 * test_startup_validation_strategy.cpp) were measured on one NLopt version,
 * which nlopt_tested_version.h records (kNloptTestedVersion). HPO fetches
 * that version (CMakeLists.txt GIT_TAG; offline builds use deps/nlopt), but
 * FETCHCONTENT_SOURCE_DIR_NLOPT, or an upgrade of the tag, builds it against
 * another one, which would keep every refusal without testing it again. */
#include "catch2/catch.hpp"
#include <tadah/hpo/nlopt_tested_version.h>

#include <nlopt.hpp>

#include <sstream>
#include <string>

namespace {

std::string dotted(int major, int minor, int bugfix) {
    std::ostringstream os;
    os << major << '.' << minor << '.' << bugfix;
    return os.str();
}

}  // namespace

/* Pins: the NLopt HPO is linked against reports (nlopt::version) the
 * version kNloptTestedVersion records, 2.10.0; on another version the case
 * fails, its message naming both versions, the documents whose repros to
 * run again (docs/issues/nlopt_newuoa_bound_stall.md, the other
 * docs/issues/nlopt_*.md, the NLOPT entries of kRefused and their tests)
 * and that kNloptTestedVersion changes only after them.
 * Before the fix: does not compile (there is no nlopt_tested_version.h:
 * nothing recorded the version; an NLopt upgrade kept the LN_NEWUOA_BOUND
 * refusal and every pinned behaviour without re-testing them). With the
 * recorded version set to 2.10.1 against the fetched 2.10.0 the case fails
 * with its message. */
TEST_CASE("the NLopt HPO is built against is the version its NLopt "
          "refusals were measured on",
          "[nlopt][caps]") {
    int major = -1, minor = -1, bugfix = -1;
    nlopt::version(major, minor, bugfix);
    const auto& tested = tadah::hpo::kNloptTestedVersion;
    const std::string linked = dotted(major, minor, bugfix);
    const std::string recorded =
        dotted(tested.major, tested.minor, tested.bugfix);
    INFO("HPO is linked against NLopt " << linked << ", but its NLopt "
         "refusals and the NLopt behaviour it pins were measured on NLopt "
         << recorded << " (tadah::hpo::kNloptTestedVersion, "
         "include/tadah/hpo/nlopt_tested_version.h). On NLopt " << linked
         << " run the repros again: docs/issues/nlopt_newuoa_bound_stall.md "
         "(LN_NEWUOA_BOUND, cases A, B and F), the other "
         "docs/issues/nlopt_*.md, and the NLOPT entries of kRefused "
         "(src/optimizer_caps.cpp) with their tests "
         "(tests/test_optimizer_caps_step.cpp, "
         "tests/test_startup_validation_strategy.cpp); then lift or keep "
         "each refusal and set kNloptTestedVersion to " << linked << ".");
    CHECK(major == tested.major);
    CHECK(minor == tested.minor);
    CHECK(bugfix == tested.bugfix);
}

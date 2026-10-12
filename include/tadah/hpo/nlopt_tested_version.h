#pragma once
/*
 * The NLopt version HPO's NLopt refusals and pinned NLopt behaviour were
 * measured on.
 *
 * The NLOPT entries of the refusal table (backend_refusal(),
 * src/optimizer_caps.cpp) and the NLopt semantics HPO's registry and tests
 * pin (optimizer_caps.h, optimizer_spec.h, search_space.h,
 * test_optimizer_caps_step.cpp, test_startup_validation_strategy.cpp; the
 * analyses in docs/issues/nlopt_*.md) hold for this version. HPO fetches it
 * (CMakeLists.txt, GIT_TAG v<major>.<minor>.<bugfix>; offline builds use
 * deps/nlopt), but FETCHCONTENT_SOURCE_DIR_NLOPT or a new tag can build HPO
 * against another one: the test case "the NLopt HPO is built against is the
 * version its NLopt refusals were measured on" (tests/test_nlopt_version.cpp)
 * then fails, naming the repros to run again. Change this constant only
 * after them (docs/issues/nlopt_newuoa_bound_stall.md, "Repro").
 */

namespace tadah::hpo {

/** \brief An NLopt version, as nlopt::version() reports it. */
struct NloptVersion {
    int major;   ///< NLopt's major version
    int minor;   ///< NLopt's minor version
    int bugfix;  ///< NLopt's bugfix version
};

/** \brief The NLopt version HPO's NLopt refusals and pinned NLopt
 *  behaviour were measured on (see the comment at the top of this file). */
inline constexpr NloptVersion kNloptTestedVersion{2, 10, 0};

}  // namespace tadah::hpo

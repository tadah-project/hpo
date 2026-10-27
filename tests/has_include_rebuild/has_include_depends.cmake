# hpo_depend_on_has_include(<source> <header>...): <source>, which tests
# for <header>... with __has_include, is compiled again by the build when
# one of them appears, changes or goes. The compiler's dependency file of a
# source lists the headers it included, not a header __has_include looked
# for and did not find; so the headers are a glob that each build checks
# (CONFIGURE_DEPENDS: a change of its result configures again), and the
# source depends on those of them that exist (OBJECT_DEPENDS). The source
# file property holds in the directory that calls the function, where the
# target that compiles <source> is defined.
# Checked by check_has_include_rebuild.cmake (hpo_has_include_rebuild).
function(hpo_depend_on_has_include source)
  file(GLOB present CONFIGURE_DEPENDS ${ARGN})
  set_property(SOURCE "${source}" APPEND PROPERTY OBJECT_DEPENDS ${present})
endfunction()

# hpo_write_no_ceres_preset(<file>): writes the initial cache (-C) with
# which hpo_configure_without_ceres configures HPO once more, without Ceres
# and without tests (check_configure_without_ceres.cmake), called in the
# configure step of the build that runs it. A configure from it sees what
# this one has seen so far:
#   - every cache entry that is not INTERNAL or STATIC, with its value and
#     type (an entry given without a type, such as -DCMAKE_PREFIX_PATH=...,
#     as STRING): the user's hints and options (CMAKE_PREFIX_PATH,
#     CMAKE_TOOLCHAIN_FILE, BLA_VENDOR, ...), the compilers and build tool,
#     and what this configure found (LAPACK, curl, ... libraries), but not an
#     entry that names a path in this build tree (FETCHCONTENT_BASE_DIR,
#     Eigen3_DIR's package redirects), which belongs to this configuration:
#     the scratch configure makes its own;
#   - the source directory of every dependency, given as
#     FETCHCONTENT_SOURCE_DIR_<NAME> (a cache entry, as above, kept even in
#     this build tree) or populated under FETCHCONTENT_BASE_DIR
#     (<name>-src), so with FETCHCONTENT_FULLY_DISCONNECTED nothing is
#     downloaded;
#   - FETCHCONTENT_FULLY_DISCONNECTED ON, TADAH_ENABLE_CERES OFF and
#     BUILD_TESTING OFF, whatever this build has.
# The environment is that of the process that runs the scratch configure.
# Checked by check_no_ceres_preset.cmake (hpo_configure_without_ceres_preset).
include(FetchContent)

# <out>: <text> as a bracket argument that no "]=*]" inside it closes.
function(_hpo_bracket out text)
  set(eq "=")
  while(text MATCHES "\\]${eq}\\]")
    string(APPEND eq "=")
  endwhile()
  set(${out} "[${eq}[${text}]${eq}]" PARENT_SCOPE)
endfunction()

function(hpo_write_no_ceres_preset file)
  set(overrides FETCHCONTENT_FULLY_DISCONNECTED TADAH_ENABLE_CERES BUILD_TESTING)
  string(CONCAT text
    "# Written by hpo_write_no_ceres_preset() (HPO/tests/configure_without_ceres/no_ceres_preset.cmake)\n"
    "# for hpo_configure_without_ceres, from the cache of the build that runs it.\n")
  set(given_sources "")
  get_cmake_property(entries CACHE_VARIABLES)
  list(SORT entries)
  foreach(name IN LISTS entries)
    get_property(type CACHE "${name}" PROPERTY TYPE)
    if(type STREQUAL "INTERNAL" OR type STREQUAL "STATIC" OR name IN_LIST overrides)
      continue()
    endif()
    get_property(value CACHE "${name}" PROPERTY VALUE)
    if(name MATCHES "^FETCHCONTENT_SOURCE_DIR_")
      if(value STREQUAL "")
        continue()
      endif()
      list(APPEND given_sources "${name}")
    else()
      string(FIND "${value}" "${CMAKE_BINARY_DIR}/" in_tree)
      if(in_tree GREATER_EQUAL 0 OR value STREQUAL CMAKE_BINARY_DIR)
        continue()
      endif()
    endif()
    if(type STREQUAL "UNINITIALIZED")
      set(type STRING)
    endif()
    _hpo_bracket(qname "${name}")
    _hpo_bracket(qvalue "${value}")
    string(APPEND text "set(${qname} ${qvalue} CACHE ${type} \"\")\n")
  endforeach()

  if(NOT "${FETCHCONTENT_BASE_DIR}" STREQUAL "")
    file(GLOB populated LIST_DIRECTORIES true "${FETCHCONTENT_BASE_DIR}/*-src")
    list(SORT populated)
    foreach(dir IN LISTS populated)
      if(NOT IS_DIRECTORY "${dir}")
        continue()
      endif()
      get_filename_component(dep "${dir}" NAME)
      string(REGEX REPLACE "-src$" "" dep "${dep}")
      string(TOUPPER "${dep}" dep)
      if("FETCHCONTENT_SOURCE_DIR_${dep}" IN_LIST given_sources)
        continue()
      endif()
      _hpo_bracket(qdir "${dir}")
      string(APPEND text
        "set(FETCHCONTENT_SOURCE_DIR_${dep} ${qdir} CACHE PATH \"\")\n")
    endforeach()
  endif()

  string(APPEND text
    "set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL \"\" FORCE)\n"
    "set(TADAH_ENABLE_CERES OFF CACHE BOOL \"\" FORCE)\n"
    "set(BUILD_TESTING OFF CACHE BOOL \"\" FORCE)\n")
  file(WRITE "${file}" "${text}")
endfunction()

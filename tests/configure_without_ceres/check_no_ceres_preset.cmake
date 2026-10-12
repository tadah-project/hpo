# Checks hpo_write_no_ceres_preset() (no_ceres_preset.cmake): a configure
# from the initial cache it writes sees what the configure that wrote it
# saw, so hpo_configure_without_ceres configures HPO in the context of the
# build that runs it, whatever that context is made of. On a throw-away
# pair of projects (nothing is downloaded; no compiler is used):
#   - outer: configured with a package found only through
#     -DCMAKE_PREFIX_PATH=<prefix> (a cache entry without a type, as a user
#     gives it), an option of its own (-DTADAH_PROBE_FLAG:BOOL=ON), a
#     dependency source given as FETCHCONTENT_SOURCE_DIR_PROBE_GIVEN, a
#     dependency populated under its FETCHCONTENT_BASE_DIR
#     (_deps/probe_unlisted-src) that no FETCHCONTENT_SOURCE_DIR_* names, and
#     TADAH_ENABLE_CERES ON, BUILD_TESTING ON,
#     FETCHCONTENT_FULLY_DISCONNECTED OFF; it finds the package and calls
#     hpo_write_no_ceres_preset(preset.cmake);
#   - inner: configured with -C preset.cmake, it finds the same package
#     (find_package REQUIRED) and prints what it was given.
# The check passes when the inner configure succeeds and has
# CMAKE_PREFIX_PATH <prefix>, TADAH_PROBE_FLAG ON, FETCHCONTENT_SOURCE_DIR_PROBE_GIVEN = <given>,
# FETCHCONTENT_SOURCE_DIR_PROBE_UNLISTED = <outer build>/_deps/probe_unlisted-src,
# a FETCHCONTENT_BASE_DIR of its own (in its build directory), and
# FETCHCONTENT_FULLY_DISCONNECTED ON, TADAH_ENABLE_CERES OFF, BUILD_TESTING
# OFF; and when no line of the preset names a path in the outer build
# directory but a FETCHCONTENT_SOURCE_DIR_* line.
#
#   cmake -DMODULE=<no_ceres_preset.cmake> -DGENERATOR=<CMake generator>
#         -DMAKE_PROGRAM=<build tool> -DWORK_DIR=<scratch>
#         -P check_no_ceres_preset.cmake
cmake_minimum_required(VERSION 3.24)

foreach(v MODULE GENERATOR MAKE_PROGRAM WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "check_no_ceres_preset.cmake: -D${v}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
set(given "${WORK_DIR}/given")
set(outer "${WORK_DIR}/outer")
set(outer_build "${WORK_DIR}/outer-build")
set(inner "${WORK_DIR}/inner")
set(inner_build "${WORK_DIR}/inner-build")
file(WRITE "${prefix}/lib/cmake/TadahProbeDep/TadahProbeDepConfig.cmake"
     "set(TadahProbeDep_FOUND TRUE)\n")
file(MAKE_DIRECTORY "${given}" "${outer_build}/_deps/probe_unlisted-src")
file(WRITE "${outer}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(hpo_no_ceres_preset_outer NONE)
find_package(TadahProbeDep REQUIRED)
include("${PRESET_MODULE}")
hpo_write_no_ceres_preset("${CMAKE_BINARY_DIR}/preset.cmake")
]=])
file(WRITE "${inner}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(hpo_no_ceres_preset_inner NONE)
find_package(TadahProbeDep REQUIRED)
include(FetchContent)
foreach(v CMAKE_PREFIX_PATH TADAH_PROBE_FLAG FETCHCONTENT_SOURCE_DIR_PROBE_GIVEN
          FETCHCONTENT_SOURCE_DIR_PROBE_UNLISTED FETCHCONTENT_BASE_DIR
          FETCHCONTENT_FULLY_DISCONNECTED TADAH_ENABLE_CERES BUILD_TESTING)
  message(STATUS "${v}=[${${v}}]")
endforeach()
]=])

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${outer}" -B "${outer_build}" -G "${GENERATOR}"
          "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
          "-DPRESET_MODULE=${MODULE}"
          "-DCMAKE_PREFIX_PATH=${prefix}"
          -DTADAH_PROBE_FLAG:BOOL=ON
          "-DFETCHCONTENT_SOURCE_DIR_PROBE_GIVEN:PATH=${given}"
          -DTADAH_ENABLE_CERES:BOOL=ON -DBUILD_TESTING:BOOL=ON
          -DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=OFF
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "the outer probe did not configure (exit ${rc}):\n${out}")
endif()

# fail(<text>...): the arguments are concatenated exactly.
set_property(GLOBAL PROPERTY NO_CERES_PRESET_NFAIL 0)
function(fail)
  set(msg "")
  math(EXPR last "${ARGC} - 1")
  foreach(i RANGE 0 ${last})
    string(APPEND msg "${ARGV${i}}")
  endforeach()
  get_property(n GLOBAL PROPERTY NO_CERES_PRESET_NFAIL)
  math(EXPR n "${n} + 1")
  set_property(GLOBAL PROPERTY NO_CERES_PRESET_NFAIL "${n}")
  message(NOTICE "FAIL: ${msg}")
endfunction()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${inner}" -B "${inner_build}" -G "${GENERATOR}"
          -C "${outer_build}/preset.cmake"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
if(NOT rc EQUAL 0)
  fail("a configure from the preset did not find what the build found "
       "(exit ${rc}):\n${out}")
else()
  foreach(pair
      "CMAKE_PREFIX_PATH|${prefix}"
      "TADAH_PROBE_FLAG|ON"
      "FETCHCONTENT_SOURCE_DIR_PROBE_GIVEN|${given}"
      "FETCHCONTENT_SOURCE_DIR_PROBE_UNLISTED|${outer_build}/_deps/probe_unlisted-src"
      "FETCHCONTENT_BASE_DIR|${inner_build}/_deps"
      "FETCHCONTENT_FULLY_DISCONNECTED|ON"
      "TADAH_ENABLE_CERES|OFF"
      "BUILD_TESTING|OFF")
    string(FIND "${pair}" "|" bar)
    string(SUBSTRING "${pair}" 0 ${bar} name)
    math(EXPR bar "${bar} + 1")
    string(SUBSTRING "${pair}" ${bar} -1 want)
    string(FIND "${out}" "-- ${name}=[${want}]\n" at)
    if(at LESS 0)
      fail("from the preset, ${name} is not [${want}]:\n${out}")
    endif()
  endforeach()
endif()

# The preset's lines, a ';' in them kept (file(STRINGS) would split there).
file(READ "${outer_build}/preset.cmake" text)
string(REPLACE ";" "<SEMICOLON>" text "${text}")
string(REPLACE "\n" ";" lines "${text}")
foreach(line IN LISTS lines)
  string(FIND "${line}" "${outer_build}" at)
  if(at GREATER_EQUAL 0 AND NOT line MATCHES "FETCHCONTENT_SOURCE_DIR_")
    fail("the preset names a path in the build directory: ${line}")
  endif()
endforeach()

get_property(nfail GLOBAL PROPERTY NO_CERES_PRESET_NFAIL)
if(nfail GREATER 0)
  file(READ "${outer_build}/preset.cmake" preset)
  message(NOTICE "the preset:\n${preset}")
  message(FATAL_ERROR "${nfail} check(s) failed")
endif()
message(STATUS "hpo_write_no_ceres_preset: a configure from its preset sees "
               "the build's cache, dependencies and overrides")

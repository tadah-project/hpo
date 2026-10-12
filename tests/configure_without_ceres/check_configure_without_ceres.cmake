# Check that HPO configures without Ceres (TADAH_ENABLE_CERES OFF) and gives
# tadah.hpo Eigen all the same: the public headers adjoint_solver.h,
# lammps_hessian.h, observable_registry.h and sims_grad.h include
# <Eigen/Dense> in every build. The build that runs this check configures
# HPO once more, in WORK_DIR, from its own sources and dependencies: PRESET
# is the initial cache hpo_write_no_ceres_preset() writes in its configure
# step (no_ceres_preset.cmake: its cache, but the entries that name a path
# in its build tree, the source directory of each dependency,
# FETCHCONTENT_FULLY_DISCONNECTED, so nothing is downloaded,
# TADAH_ENABLE_CERES OFF, no tests). It reads the
# configuration through CMake's file API (codemodel-v2) and checks that
#   - the configure step succeeds;
#   - it is the configuration without Ceres: no target named ceres, and no
#     TADAH_USE_CERES among tadah.hpo's definitions;
#   - one of tadah.hpo's include directories has Eigen/Dense.
#
#   cmake -DHPO_DIR=<HPO source> -DPRESET=<initial cache> -DGENERATOR=<CMake generator>
#         -DWORK_DIR=<scratch> -P check_configure_without_ceres.cmake
cmake_minimum_required(VERSION 3.24)

foreach(v HPO_DIR PRESET GENERATOR WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "check_configure_without_ceres.cmake: -D${v}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(build "${WORK_DIR}/build")
# The file API query: CMake writes the codemodel of the configuration.
file(WRITE "${build}/.cmake/api/v1/query/codemodel-v2" "")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${HPO_DIR}" -B "${build}" -G "${GENERATOR}"
          -C "${PRESET}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "HPO did not configure with TADAH_ENABLE_CERES OFF "
                      "(exit ${rc}):\n${out}")
endif()

set(reply "${build}/.cmake/api/v1/reply")
file(GLOB index "${reply}/index-*.json")
list(LENGTH index nindex)
if(NOT nindex EQUAL 1)
  message(FATAL_ERROR "expected one file API index in ${reply}, found "
                      "${nindex}: '${index}'")
endif()
file(READ "${index}" json)
string(JSON codemodel_file GET "${json}" reply codemodel-v2 jsonFile)
file(READ "${reply}/${codemodel_file}" codemodel)

set(hpo_file "")
set(ceres_targets "")
string(JSON ntargets LENGTH "${codemodel}" configurations 0 targets)
math(EXPR last "${ntargets} - 1")
foreach(i RANGE ${last})
  string(JSON name GET "${codemodel}" configurations 0 targets ${i} name)
  if(name STREQUAL "tadah.hpo")
    string(JSON hpo_file GET "${codemodel}" configurations 0 targets ${i} jsonFile)
  elseif(name STREQUAL "ceres")
    list(APPEND ceres_targets "${name}")
  endif()
endforeach()
if(hpo_file STREQUAL "")
  message(FATAL_ERROR "the configuration has no target tadah.hpo")
endif()

set(nfail 0)
macro(fail)
  string(CONCAT what ${ARGV})
  message(NOTICE "FAIL: ${what}")
  math(EXPR nfail "${nfail} + 1")
endmacro()

if(ceres_targets)
  fail("the configuration has a target ceres: it is not the one without Ceres")
endif()

file(READ "${reply}/${hpo_file}" target)
set(includes "")
set(defines "")
string(JSON ngroups ERROR_VARIABLE no_groups LENGTH "${target}" compileGroups)
if(no_groups)
  set(ngroups 0)
endif()
if(ngroups EQUAL 0)
  fail("tadah.hpo has no compile group (no sources)")
else()
  math(EXPR last_group "${ngroups} - 1")
  foreach(g RANGE ${last_group})
    string(JSON n ERROR_VARIABLE none LENGTH "${target}" compileGroups ${g} includes)
    if(NOT none AND n GREATER 0)
      math(EXPR last_n "${n} - 1")
      foreach(k RANGE ${last_n})
        string(JSON p GET "${target}" compileGroups ${g} includes ${k} path)
        list(APPEND includes "${p}")
      endforeach()
    endif()
    string(JSON n ERROR_VARIABLE none LENGTH "${target}" compileGroups ${g} defines)
    if(NOT none AND n GREATER 0)
      math(EXPR last_n "${n} - 1")
      foreach(k RANGE ${last_n})
        string(JSON d GET "${target}" compileGroups ${g} defines ${k} define)
        list(APPEND defines "${d}")
      endforeach()
    endif()
  endforeach()
endif()

if("TADAH_USE_CERES" IN_LIST defines)
  fail("tadah.hpo is compiled with TADAH_USE_CERES: it is not the "
       "configuration without Ceres")
endif()

set(eigen "")
foreach(p IN LISTS includes)
  if(EXISTS "${p}/Eigen/Dense")
    set(eigen "${p}")
    break()
  endif()
endforeach()
if(eigen STREQUAL "")
  string(REPLACE ";" "\n  " shown "${includes}")
  fail("no include directory of tadah.hpo has Eigen/Dense, which its public "
       "headers adjoint_solver.h, lammps_hessian.h, observable_registry.h "
       "and sims_grad.h include. Its include directories:\n  ${shown}")
endif()

if(nfail GREATER 0)
  message(FATAL_ERROR "${nfail} check(s) failed")
endif()
message(STATUS "HPO without Ceres: tadah.hpo has Eigen from ${eigen}")

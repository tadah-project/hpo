# Checks hpo_depend_on_has_include() (has_include_depends.cmake): a source
# that tests for headers with __has_include is compiled again when they
# appear and when they go, by the build alone (no configure by hand). The
# compiler's dependency file of a source lists the headers it included,
# not a header __has_include looked for and did not find, so without the
# function an incremental build keeps the object compiled without the
# headers after they appear. On a throw-away project whose probe.cpp
# prints "absent" without its two headers and the texts they define with
# them:
#   1. configured and built without the headers: "absent";
#   2. both headers written, built again: "a1 b1";
#   3. one header changed, built again: "a2 b1";
#   4. one header removed, built again: "absent".
# A second passes before each change, so a file system that keeps whole
# seconds still sees each header newer than the object.
#
#   cmake -DMODULE=<has_include_depends.cmake> -DGENERATOR=<CMake generator>
#         -DMAKE_PROGRAM=<build tool> -DCXX=<C++ compiler> -DWORK_DIR=<scratch>
#         -P check_has_include_rebuild.cmake
cmake_minimum_required(VERSION 3.24)

foreach(v MODULE GENERATOR MAKE_PROGRAM CXX WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "check_has_include_rebuild.cmake: -D${v}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(src "${WORK_DIR}/src")
set(build "${WORK_DIR}/build")
set(inc "${src}/inc/probe")
file(MAKE_DIRECTORY "${inc}")
file(WRITE "${src}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(hpo_has_include_probe CXX)
include("${HAS_INCLUDE_MODULE}")
add_executable(probe probe.cpp)
target_include_directories(probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
# The program in the build directory itself, for every generator ($<0:>
# keeps a multi-config generator from adding a directory of the config).
set_target_properties(probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}$<0:>")
hpo_depend_on_has_include("${CMAKE_CURRENT_SOURCE_DIR}/probe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/probe/gen_a.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/probe/gen_b.h")
]=])
file(WRITE "${src}/probe.cpp" [=[
#include <cstdio>
#if __has_include(<probe/gen_a.h>) && __has_include(<probe/gen_b.h>)
#include <probe/gen_a.h>
#include <probe/gen_b.h>
int main() { std::puts(PROBE_A " " PROBE_B); }
#else
int main() { std::puts("absent"); }
#endif
]=])

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${src}" -B "${build}" -G "${GENERATOR}"
          "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
          "-DCMAKE_CXX_COMPILER=${CXX}"
          "-DHAS_INCLUDE_MODULE=${MODULE}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "the probe did not configure (exit ${rc}):\n${out}")
endif()

set(nfail 0)
# build_and_run(<step> <expected output>): builds the probe and runs it.
macro(build_and_run step want)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${build}" --config Release
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${step}: the probe did not build (exit ${rc}):\n${out}")
  endif()
  execute_process(COMMAND "${build}/probe"
    RESULT_VARIABLE rc OUTPUT_VARIABLE got ERROR_VARIABLE err)
  string(STRIP "${got}" got)
  if(NOT rc EQUAL 0 OR NOT got STREQUAL "${want}")
    math(EXPR nfail "${nfail} + 1")
    message(NOTICE "FAIL: ${step}: the probe prints [${got}] (exit ${rc}), "
                   "not [${want}]; the build said:\n${out}")
  else()
    message(STATUS "${step}: [${got}]")
  endif()
endmacro()

macro(after_a_second)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1.1)
endmacro()

build_and_run("without the headers" "absent")
after_a_second()
file(WRITE "${inc}/gen_a.h" "#define PROBE_A \"a1\"\n")
file(WRITE "${inc}/gen_b.h" "#define PROBE_B \"b1\"\n")
build_and_run("both headers written" "a1 b1")
after_a_second()
file(WRITE "${inc}/gen_a.h" "#define PROBE_A \"a2\"\n")
build_and_run("gen_a.h changed" "a2 b1")
after_a_second()
file(REMOVE "${inc}/gen_b.h")
build_and_run("gen_b.h removed" "absent")

if(nfail GREATER 0)
  message(FATAL_ERROR "${nfail} step(s) failed: a source that tests for "
                      "headers with __has_include is not compiled again when "
                      "they appear or go")
endif()
message(STATUS "hpo_depend_on_has_include: the probe is compiled again when "
               "its headers appear, change and go")

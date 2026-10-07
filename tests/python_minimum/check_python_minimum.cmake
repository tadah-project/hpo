# Check of the Python interpreter HPO's tests accept (cmake/hpo_test_python.cmake),
# on the throw-away project in project/:
#
#   CASE=old      an interpreter that reports Python 3.10 (the given
#                 interpreter, run through a wrapper whose sitecustomize.py
#                 sets sys.version_info, which FindPython reads) -> the
#                 configure step fails: 'Could NOT find Python3: Found
#                 unsuitable version "3.10.0", but required is at least
#                 "3.11"'
#   CASE=current  the given interpreter -> configures, and finds it
#
#   cmake -DCASE=<case> -DPYTHON=<interpreter of the build> -DMODULE=<HPO>/cmake/hpo_test_python.cmake
#         -DWORK_DIR=<scratch> -P check_python_minimum.cmake
cmake_minimum_required(VERSION 3.20)

foreach(v CASE PYTHON MODULE WORK_DIR)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "check_python_minimum.cmake: -D${v}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(nfail 0)
macro(fail what)
  message(NOTICE "FAIL: ${what}")
  math(EXPR nfail "${nfail} + 1")
endmacro()

if(CASE STREQUAL "old")
  set(fake "${WORK_DIR}/fake")
  file(MAKE_DIRECTORY "${fake}")
  file(WRITE "${fake}/sitecustomize.py"
    "import sys\nsys.version_info = (3, 10, 0, 'final', 0)\n")
  file(WRITE "${fake}/python3"
    "#!/bin/sh\n"
    "PYTHONPATH=\"${fake}\${PYTHONPATH:+:\$PYTHONPATH}\" exec \"${PYTHON}\" \"\$@\"\n")
  file(CHMOD "${fake}/python3" PERMISSIONS
       OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
  set(interpreter "${fake}/python3")
  # The stand-in must report 3.10, or the case checks nothing.
  execute_process(
    COMMAND "${interpreter}" -c
      "import sys; print('.'.join(str(x) for x in sys.version_info[:3]))"
    RESULT_VARIABLE rc OUTPUT_VARIABLE reported ERROR_VARIABLE err
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT rc EQUAL 0 OR NOT reported STREQUAL "3.10.0")
    message(FATAL_ERROR "the stand-in interpreter reports '${reported}' "
                        "(exit ${rc}, ${err}), not 3.10.0")
  endif()
  set(expect_ok FALSE)
elseif(CASE STREQUAL "current")
  set(interpreter "${PYTHON}")
  set(expect_ok TRUE)
else()
  message(FATAL_ERROR "check_python_minimum.cmake: unknown CASE '${CASE}'")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --fresh
          -S "${CMAKE_CURRENT_LIST_DIR}/project" -B "${WORK_DIR}/build"
          "-DHPO_TEST_PYTHON=${MODULE}"
          "-DPython3_EXECUTABLE=${interpreter}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
string(REGEX REPLACE "[ \t\r\n]+" " " flat "${out}")
string(FIND "${out}" "HPO_TEST_PYTHON_FOUND" found)

if(expect_ok)
  if(NOT rc EQUAL 0)
    fail("configure failed (exit ${rc})")
  endif()
  if(found EQUAL -1)
    fail("the project did not finish configuring")
  elseif(NOT out MATCHES "HPO_TEST_PYTHON_FOUND ([^ ]+) ([0-9.]+)")
    fail("no interpreter reported")
  else()
    file(REAL_PATH "${CMAKE_MATCH_1}" got)
    file(REAL_PATH "${interpreter}" want)
    if(NOT got STREQUAL want)
      fail("found ${CMAKE_MATCH_1}, not ${interpreter}")
    endif()
  endif()
else()
  if(rc EQUAL 0)
    fail("configure succeeded with an interpreter that reports 3.10.0")
  endif()
  if(NOT found EQUAL -1)
    fail("the project finished configuring")
  endif()
  set(text "Could NOT find Python3: Found unsuitable version \"3.10.0\", but required is at least \"3.11\"")
  string(FIND "${flat}" "${text}" at)
  if(at EQUAL -1)
    fail("the message lacks '${text}'")
  endif()
endif()

if(nfail GREATER 0)
  message(FATAL_ERROR "${nfail} check(s) failed for CASE=${CASE}; "
                      "configure output:\n${out}")
endif()
message(STATUS "CASE=${CASE}: ok")

# Throw-away project driven by ../check_python_minimum.cmake: it finds the
# Python interpreter exactly as HPO's tests do, with HPO's own
# cmake/hpo_test_python.cmake, and reports what it found.
cmake_minimum_required(VERSION 3.15)
project(hpo_test_python_minimum_case NONE)

include("${HPO_TEST_PYTHON}")
message(STATUS "HPO_TEST_PYTHON_FOUND ${Python3_EXECUTABLE} ${Python3_VERSION}")

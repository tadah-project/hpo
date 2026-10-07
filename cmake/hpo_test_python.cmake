# The Python interpreter of HPO's tests (tests/CMakeLists.txt): the registry
# drift gate, the example fingerprint gates and the vendored-Boost gate run
# Python scripts. Python 3 is required to build the tests (as in the
# umbrella build, whose tests need it too): without it the gates would
# silently not exist.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

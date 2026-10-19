# The Python interpreter of HPO's tests (tests/CMakeLists.txt): the registry
# drift gate, the vendored-Boost gate and other checks run Python
# scripts. Python 3 is required to build the tests (as in the
# umbrella build, whose tests need it too): without it the gates would
# silently not exist.
#
# Python 3.11 or later: the registry drift gate hpo_anneal_registry_sync
# runs blueprint/gen_anneal_keys.py, which reads blueprint/anneal_keys.toml
# with tomllib (in the standard library from 3.11). An older interpreter is
# refused here, at the configure step, instead of failing that test with
# "No module named 'tomllib'"; FindPython passes over an older one on PATH
# for a newer one it can find.
find_package(Python3 3.11 COMPONENTS Interpreter REQUIRED)

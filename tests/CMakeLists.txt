# Catch2 and its test discovery come from cmake-tools (tadah_setup_tests()).
tadah_setup_tests()

file(GLOB tests_src CONFIGURE_DEPENDS "*.cpp")
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_hpo ${tests_src})
target_link_libraries(test_hpo PRIVATE
    Catch2 tadah.core tadah.mlip tadah.hpo)
# Pristine all-gates-off ASA reference build for the golden-neutrality
# gate (test_asa_golden.cpp); the featured tadah_asa arrives through
# tadah.hpo's link interface.
target_link_libraries(test_hpo PRIVATE tadah_asa_pristine)

# Phase 2A.5 microbench gate. Enable Catch2's BENCHMARK macros and point at
# a real, fitted Ta potential checked into the worktree so the bench runs
# out-of-the-box. Override the pot at run time with the TADAH_BENCH_POT env
# var; override the cell with TADAH_BENCH_CELL_DATA (LAMMPS read_data file).
target_compile_definitions(test_hpo PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
if(DEFINED ENV{TADAH_PATH})
    set(_tadah_bench_default_pot
        "$ENV{TADAH_PATH}/HPO_test/pot.tadah")
    target_compile_definitions(test_hpo PRIVATE
        "TADAH_BENCH_DEFAULT_POT=\"${_tadah_bench_default_pot}\"")
endif()

# Worked examples (HPO/examples). test_examples_parse.cpp gives every
# examples/*/config.hpo a test case of its own: the list is generated here,
# and CONFIGURE_DEPENDS re-globs the directory on every build, so a new
# example is picked up (and fails until examples/tools/parse_guard.manifest
# names what it needs). The PC_OBSERVABLE gradient tests
# (test_hpo_host_e2e.cpp) train on the examples' shared Ta data.
cmake_path(SET _hpo_examples_dir NORMALIZE
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples")
file(GLOB _hpo_example_configs CONFIGURE_DEPENDS
    "${_hpo_examples_dir}/*/config.hpo")
list(SORT _hpo_example_configs)
set(_hpo_example_cases "")
foreach(_cfg IN LISTS _hpo_example_configs)
    cmake_path(GET _cfg PARENT_PATH _dir)
    cmake_path(GET _dir FILENAME _name)
    string(APPEND _hpo_example_cases "TADAH_EXAMPLE_PARSE_CASE(\"${_name}\")\n")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/examples_parse_cases.inc"
     CONTENT "${_hpo_example_cases}" @ONLY)
target_include_directories(test_hpo PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(test_hpo PRIVATE
    "TADAH_HPO_EXAMPLES_DIR=\"${_hpo_examples_dir}\"")

include(Catch)
catch_discover_tests(test_hpo)

# Statistical validation suite (equilibrium sampling, swap acceptance,
# transport): hidden Catch2 tag [.statistical], excluded from the default
# run; exposed as one labelled ctest entry for `ctest -L statistical`.
# --warn NoTests: a tag that matches no test case fails instead of passing.
add_test(NAME hpo_anneal_statistical
    COMMAND test_hpo "[.statistical]" --warn NoTests)
set_tests_properties(hpo_anneal_statistical PROPERTIES
    LABELS "anneal;statistical"
    TIMEOUT 900)

# Corana et al. (1987) protocol-replication oracle: hidden tag
# [.corana1987] (one test case), armed by the transcribed reference table
# tests/data/corana1987_protocol.tsv. While the table is empty the case
# calls TADAH_SKIP and the entry is reported as Skipped (cmake-tools'
# TADAH-SKIP marker; include(Catch) sets the regular expression).
add_test(NAME hpo_corana1987_protocol
    COMMAND test_hpo "[.corana1987]" --warn NoTests)
set_tests_properties(hpo_corana1987_protocol PROPERTIES
    LABELS "anneal;statistical"
    TIMEOUT 900
    SKIP_REGULAR_EXPRESSION "${TADAH_TEST_SKIP_REGULAR_EXPRESSION}")

# Real-signal termination tests (test_hpo_termination.cpp): each case raises
# SIGINT/SIGTERM/SIGUSR2 against a live hpo_run(). CORE's stop flag can only
# be raised by a real signal and is never lowered, and the HPOLog / HPO_Host
# statics persist, so every case is a hidden tag with an entry -- a process --
# of its own. `ctest -L shutdown` runs them. NB: no ';' in add_test()
# arguments (MLIP's add_test() override splits them).
set(_hpo_shutdown_cases
    dlib sigterm sigusr2 nlopt multistart prestage gfs anneal_pt anneal_pa
    anneal_pa_threads asa asa_block asa_periodic asa_sampling noise
    no_usable_loss)
if(TADAH_ENABLE_CERES)
    list(APPEND _hpo_shutdown_cases ceres)
endif()
foreach(_case IN LISTS _hpo_shutdown_cases)
    add_test(NAME hpo_shutdown_${_case}
        COMMAND test_hpo "[.shutdown_${_case}]" --warn NoTests)
    set_tests_properties(hpo_shutdown_${_case} PROPERTIES
        LABELS "hpo;shutdown"
        TIMEOUT 300)
endforeach()

# Host end-to-end groups: hidden tags, like the other HPO_Host end-to-end
# cases, because they drive real hpo_run()/HPO_Host evaluations whose
# process-wide state (HPOLog, the HPO_Host statics, the thread-local
# trainer cache) the cases of a group share; each group is one entry -- a
# process -- of its own. `ctest -L e2e` runs them.
#   fixweight_e2e  test_hpo_fixweight_e2e.cpp: pinned/OPTIM'd weights, exact
#                  potential files, seed anchors, LAMMPS read-backs
#   durable_e2e    test_hpo_fixweight_e2e.cpp: best-potential persistence and
#                  run endings; spawns test_hpo children
#                  ([.durable_e2e_child], [.durable_latch_child])
#   embf0_e2e      test_hpo_embf0_e2e.cpp: DM_REAM EMBF0 per candidate
#   rcut_e2e       test_rcut_autoderive.cpp: knot RCUT re-derivation
#   rhoform_e2e    test_hpo_rhoform.cpp: RHOFORM through hpo_run (an invalid
#                  density form stops the run; the NONNEG/MONOTONE
#                  projection, [rhoform_projection_e2e], is one of its cases)
#   rhogate_e2e    test_rho_gates.cpp: RHOGATE rejects before training
#   rhobargate_e2e test_rho_gates.cpp: RHOBARGATE rejects before training
#   bumpgate_e2e   test_bump_support_gate.cpp: BUMPGATE rejects before
#                  training (also tagged [!benchmark])
#   nn_e2e         test_nn_exact_e2e.cpp: exact neighbour lists -- thin,
#                  triclinic, sheared and unwrapped validation cells give
#                  the loss of their supercells; a candidate cutoff beyond
#                  the one-time lists is evaluated with rebuilt lists
#   valmetric_e2e  test_validation_follows_metric_e2e.cpp: validation
#                  predicts forces/stresses when PC_FRMSE/PC_SRMSE ask,
#                  whatever FORCE/STRESS training used, and leaves out the
#                  structures stored with force/stress weight 0; and
#                  test_metric_unit_labels.cpp: the --skip-above message
#                  gives ErRMSE and SRMSE in their units
#   trainer_e2e    test_hpo_fresh_trainer_e2e.cpp: every evaluation trains
#                  a model built from its own candidate (row weights,
#                  MPARAMS, SIGMA, a second host on a thread, an M_KRR -c
#                  potential, gradients of another host's model)
set(_hpo_e2e_groups fixweight_e2e durable_e2e embf0_e2e rcut_e2e
                    rhoform_e2e rhogate_e2e rhobargate_e2e bumpgate_e2e
                    nn_e2e valmetric_e2e trainer_e2e)
foreach(_group IN LISTS _hpo_e2e_groups)
    add_test(NAME hpo_${_group}
        COMMAND test_hpo "[${_group}]" --warn NoTests)
    set_tests_properties(hpo_${_group} PROPERTIES
        LABELS "hpo;e2e"
        TIMEOUT 600)
endforeach()

# End-to-end cases that run the built `tadah` executable, named by the
# TADAH_BINARY environment variable (test_hpo_fixweight_e2e.cpp, tag
# [tadah_cli]). Each is a hidden tag with an entry of its own, so that the
# TADAH-SKIP marker applies to exactly one case: without TADAH_BINARY a
# case calls TADAH_SKIP and CTest reports it as Skipped, not Passed. In a
# build that has the `tadah` executable target (Tadah.MLIP, which adds this
# directory) the entry sets TADAH_BINARY to it, so the case runs; in one
# without it (a standalone HPO build) TADAH_BINARY comes from the
# environment of ctest, if at all.
set(_hpo_tadah_cli_cases seed_anchor_warm seed_anchor_nlopt durable_exit)
foreach(_case IN LISTS _hpo_tadah_cli_cases)
    add_test(NAME hpo_tadah_cli_${_case}
        COMMAND test_hpo "[.tadah_cli_${_case}]" --warn NoTests)
    set_tests_properties(hpo_tadah_cli_${_case} PROPERTIES
        LABELS "hpo;e2e"
        TIMEOUT 600
        SKIP_REGULAR_EXPRESSION "${TADAH_TEST_SKIP_REGULAR_EXPRESSION}"
        ENVIRONMENT
            "$<$<TARGET_EXISTS:tadah>:TADAH_BINARY=$<TARGET_FILE:tadah>>")
endforeach()

# Registry drift gate: the checked-in anneal_keys_gen.h must match what the
# generator produces from blueprint/anneal_keys.toml. Regenerate with
#   python3 HPO/blueprint/gen_anneal_keys.py --header
# Python 3 is required to build the tests (as in the umbrella build, whose
# tests need it too): without it the gate would silently not exist.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME hpo_anneal_registry_sync
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../blueprint/gen_anneal_keys.py
        --check)
set_tests_properties(hpo_anneal_registry_sync PROPERTIES
    LABELS "anneal"
    TIMEOUT 30)

# Fingerprint drift gate: the fields of each example's
# reference_fingerprint.json that come from tools/tolerance_overrides.json
# (tolerances, accepted statuses, journal checks, the rationale) must equal
# that file, so an overrides edit cannot leave a fingerprint behind.
# Regenerate with tools/generate_fingerprint.py <example> --overrides ...
add_test(NAME hpo_example_fingerprint_overrides_sync
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/generate_fingerprint.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples
        --overrides ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/tolerance_overrides.json
        --check)
set_tests_properties(hpo_example_fingerprint_overrides_sync PROPERTIES
    LABELS "examples"
    TIMEOUT 30)
# The generator's and the gate's handling of the overrides, on a synthetic
# run (examples/tools/test_generate_fingerprint.py).
add_test(NAME hpo_example_fingerprint_tool
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/test_generate_fingerprint.py)
set_tests_properties(hpo_example_fingerprint_tool PROPERTIES
    LABELS "examples"
    TIMEOUT 30)

# Boost drift gate: every Boost header HPO includes must be in the Boost
# Tadah vendors (MLIP's external/boost, on HPO's include path through
# tadah.mlip), so the build never depends on a system Boost. The include
# directories are joined with ':' (no ';' in add_test() arguments).
add_test(NAME hpo_vendored_boost_includes
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_vendored_boost.py
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        "$<JOIN:$<TARGET_PROPERTY:boost_iostream,INTERFACE_INCLUDE_DIRECTORIES>,:>"
        --cxx ${CMAKE_CXX_COMPILER})
set_tests_properties(hpo_vendored_boost_includes PROPERTIES
    TIMEOUT 60)
# The gate itself, on fake trees (tests/test_check_vendored_boost.py).
add_test(NAME hpo_vendored_boost_gate_tool
    COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_check_vendored_boost.py)
set_tests_properties(hpo_vendored_boost_gate_tool PROPERTIES
    TIMEOUT 60)

if(TADAH_HPO_MPI)
    add_executable(test_hpo_anneal_mpi mpi/test_anneal_mpi.cpp)
    target_link_libraries(test_hpo_anneal_mpi PRIVATE
        tadah.core tadah.mlip tadah.hpo MPI::MPI_CXX)
    configure_file(
        mpi/run_anneal_mpi_matrix.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/run_anneal_mpi_matrix.cmake
        @ONLY)
    add_test(NAME hpo_anneal_mpi_matrix
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:test_hpo_anneal_mpi>
            -P ${CMAKE_CURRENT_BINARY_DIR}/run_anneal_mpi_matrix.cmake)
    set_tests_properties(hpo_anneal_mpi_matrix PROPERTIES
        LABELS "anneal;mpi"
        TIMEOUT 180)
endif()

# Integration matrix — opt-in via `ctest -L slow`. Requires a bundle
# directory; the env var TADAH_HPO_INTEGRATION_BUNDLE selects which one.
# Without it the wrapper exits with 77, which CTest reports as Skipped.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh")
    add_test(NAME hpo_strategy_matrix
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh)
    set_tests_properties(hpo_strategy_matrix PROPERTIES
        LABELS "slow"
        SKIP_RETURN_CODE 77)
endif()

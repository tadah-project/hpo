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
# [.corana1987], armed by the transcribed reference table
# tests/data/corana1987_protocol.tsv (reports SKIPPED while empty).
add_test(NAME hpo_corana1987_protocol
    COMMAND test_hpo "[.corana1987]" --warn NoTests)
set_tests_properties(hpo_corana1987_protocol PROPERTIES
    LABELS "anneal;statistical"
    TIMEOUT 900)

# Real-signal termination tests (test_hpo_termination.cpp): each case raises
# SIGINT/SIGTERM/SIGUSR2 against a live hpo_run(). CORE's stop flag can only
# be raised by a real signal and is never lowered, and the HPOLog / HPO_Host
# statics persist, so every case is a hidden tag with an entry -- a process --
# of its own. `ctest -L shutdown` runs them. NB: no ';' in add_test()
# arguments (MLIP's add_test() override splits them).
set(_hpo_shutdown_cases
    dlib sigterm sigusr2 nlopt multistart prestage gfs anneal_pt anneal_pa
    anneal_pa_threads asa asa_block asa_periodic asa_sampling noise)
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

# Registry drift gate: the checked-in anneal_keys_gen.h must match what the
# generator produces from blueprint/anneal_keys.toml. Regenerate with
#   python3 HPO/blueprint/gen_anneal_keys.py --header
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME hpo_anneal_registry_sync
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../blueprint/gen_anneal_keys.py
            --check)
    set_tests_properties(hpo_anneal_registry_sync PROPERTIES
        LABELS "anneal"
        TIMEOUT 30)
endif()

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
# Skipped (gracefully) when the env var is not set.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh")
    add_test(NAME hpo_strategy_matrix
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh)
    set_tests_properties(hpo_strategy_matrix PROPERTIES LABELS "slow")
endif()

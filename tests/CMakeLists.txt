
file(GLOB tests_src "*.cpp")
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_hpo ${tests_src})
target_link_libraries(test_hpo PRIVATE
    Catch2 tadah.core tadah.mlip tadah.hpo)

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

include(Catch.cmake)
catch_discover_tests(test_hpo)

# Integration matrix — opt-in via `ctest -L slow`. Requires a bundle
# directory; the env var TADAH_HPO_INTEGRATION_BUNDLE selects which one.
# Skipped (gracefully) when the env var is not set.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh")
    add_test(NAME hpo_strategy_matrix
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh)
    set_tests_properties(hpo_strategy_matrix PROPERTIES LABELS "slow")
endif()

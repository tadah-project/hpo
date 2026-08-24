# Opt-in project injection for annealing_benchmark.cpp.  Pass this file as
# CMAKE_PROJECT_INCLUDE while configuring either the baseline or current TADAH
# snapshot; no production CMakeLists.txt needs to be edited.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND
   NOT TARGET hpo_annealing_benchmark)
  if(NOT ANNEAL_BENCHMARK_SOURCE)
    set(ANNEAL_BENCHMARK_SOURCE
        "${CMAKE_CURRENT_LIST_DIR}/annealing_benchmark.cpp"
        CACHE FILEPATH "Cross-revision annealing benchmark source")
  endif()
  if(NOT EXISTS "${ANNEAL_BENCHMARK_SOURCE}")
    message(FATAL_ERROR
      "ANNEAL_BENCHMARK_SOURCE does not exist: ${ANNEAL_BENCHMARK_SOURCE}")
  endif()

  add_executable(hpo_annealing_benchmark "${ANNEAL_BENCHMARK_SOURCE}")
  set_target_properties(hpo_annealing_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  target_link_libraries(hpo_annealing_benchmark PRIVATE tadah.hpo)
endif()

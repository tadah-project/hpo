# Vendored ASA-30.53 (Adaptive Simulated Annealing), Lester Ingber,
# BSD-3-Clause -- see ASA-LICENSE.  Compiled as C through the wrapper TU
# asa_build.c, whose gate set lives in tadah_asa_config.h.  Deliberately
# NOT linked against tadah::compile_options (third-party code is exempt
# from the project warning set; precedent: MLIP external/boost).
add_library(tadah_asa STATIC asa_build.c tadah_asa_hook.c)
target_include_directories(tadah_asa SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tadah_asa PROPERTIES
  C_STANDARD 11
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(tadah_asa PRIVATE -w)

# Pristine all-gates-off reference build (tadah_asaref_-prefixed
# symbols); linked only by the golden-neutrality unit test.
if(BUILD_TESTING)
  add_library(tadah_asa_pristine STATIC asa_pristine.c)
  target_include_directories(tadah_asa_pristine SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(tadah_asa_pristine PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
    EXCLUDE_FROM_ALL ON)
  target_compile_options(tadah_asa_pristine PRIVATE -w)
endif()

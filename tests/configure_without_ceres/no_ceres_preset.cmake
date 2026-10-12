# hpo_write_no_ceres_preset(<file>): writes the initial cache (-C) with
# which hpo_configure_without_ceres configures HPO once more, without Ceres
# and without tests (check_configure_without_ceres.cmake), in the configure
# step of the build that runs it: the source directory of each dependency
# this build made available (FETCHCONTENT_FULLY_DISCONNECTED, so nothing is
# downloaded), the compilers and build tool, the build type, the LAMMPS
# paths and TADAH_GIT_BRANCH; TADAH_ENABLE_CERES OFF, BUILD_TESTING OFF.
include(FetchContent)

function(hpo_write_no_ceres_preset file)
  set(_hpo_no_ceres_cache
      "# Written by HPO/tests/CMakeLists.txt for hpo_configure_without_ceres.\n")
  foreach(_dep tadah_cmake Tadah.CORE Tadah.MODELS Tadah.MLIP Tadah.MD
               tadah.lammps gemmi json dlib Eigen3 nlopt)
      FetchContent_GetProperties(${_dep} SOURCE_DIR _dep_src)
      if(_dep_src)
          string(TOUPPER "${_dep}" _dep_up)
          string(APPEND _hpo_no_ceres_cache
              "set(FETCHCONTENT_SOURCE_DIR_${_dep_up} [==[${_dep_src}]==] CACHE PATH \"\")\n")
      endif()
  endforeach()
  foreach(_var CMAKE_MAKE_PROGRAM CMAKE_C_COMPILER CMAKE_CXX_COMPILER
               CMAKE_BUILD_TYPE BUILD_SHARED_LIBS TADAH_LAMMPS_DIR
               TADAH_LAMMPS_LIB TADAH_LAMMPS_LIBS TADAH_GIT_BRANCH)
      if(DEFINED ${_var})
          string(APPEND _hpo_no_ceres_cache
              "set(${_var} [==[${${_var}}]==] CACHE STRING \"\")\n")
      endif()
  endforeach()
  string(APPEND _hpo_no_ceres_cache
      "set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL \"\")\n"
      "set(TADAH_ENABLE_CERES OFF CACHE BOOL \"\")\n"
      "set(BUILD_TESTING OFF CACHE BOOL \"\")\n")
  file(WRITE "${file}" "${_hpo_no_ceres_cache}")
endfunction()

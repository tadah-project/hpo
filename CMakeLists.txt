cmake_minimum_required(VERSION 3.12)
project(Tadah.HPO LANGUAGES CXX)

option(TADAH_LAMMPS_DIR "Path to lammps root directory")
option(TADAH_LAMMPS_LIB "Path to the lammps library")
get_filename_component(_lammps_libname "${TADAH_LAMMPS_LIB}" NAME)
message(STATUS "Using LAMMPS library: ${_lammps_libname}")

# 0. Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "Create a separate build directory and run CMake from there.")
endif()

# 1. Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type for single-config generators (Ninja/Unix Makefiles).
set(TADAH_DEFAULT_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Default build type for single-config generators when CMAKE_BUILD_TYPE is not set.")

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to ${TADAH_DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${TADAH_DEFAULT_BUILD_TYPE}" CACHE STRING
      "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel).")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

if(BUILD_SHARED_LIBS)
  # Always full RPATH
  # use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# 2. Bring in the shared compile/link options
include(FetchContent)
if(NOT TARGET tadah::compile_options)
  FetchContent_Declare (
    tadah_cmake
    GIT_REPOSITORY  https://git.ecdf.ed.ac.uk/tadah/cmake-tools.git
    GIT_TAG         main
  )
  FetchContent_MakeAvailable(tadah_cmake)   # defines tadah::compile_options
endif()

# (option) Developer builds -DTADAH_DEVELOP=ON
tadah_use_local_module(Tadah.MLIP MLIP)  # FetchContent name + folder name

# 3. Git branch information, for informational purposes 
tadah_get_git_branch(TADAH_GIT_BRANCH)
message(STATUS "${PROJECT_NAME} module is using TADAH_GIT_BRANCH ${TADAH_GIT_BRANCH}")

# 4. Fetch Tadah! modules
FetchContent_Declare(
    Tadah.MLIP
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/mlip.git
    GIT_TAG ${TADAH_GIT_BRANCH}
)
FetchContent_MakeAvailable(Tadah.MLIP)

FetchContent_Declare(
    Tadah.MD
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/md.git
    GIT_TAG ${TADAH_GIT_BRANCH}
)
FetchContent_MakeAvailable(Tadah.MD)

# 5. External dependencies
FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24.6
)
FetchContent_MakeAvailable(dlib)
add_compile_definitions(TADAH_USE_DLIB)

FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(nlopt)

# User must specify the root directory for lammps
if(NOT TADAH_LAMMPS_DIR)
  message( FATAL_ERROR 
    "
    TADAH_LAMMPS_DIR is not specified.
    example:
    -DTADAH_LAMMPS_DIR=/path/to/lammps
    CMake will exit." )
endif()
if(NOT EXISTS ${TADAH_LAMMPS_DIR})
  message(FATAL_ERROR "The LAMMPS directory does not exist: ${TADAH_LAMMPS_DIR}")
endif()

# we need to know which LAMMPS to use
# note that LAMMPS lib must by compiles with tadah.lammps
if(NOT TADAH_LAMMPS_LIB)
  message( FATAL_ERROR 
    "
    TADAH_LAMMPS_LIB is not specified.
    example:
    DTADAH_LAMMPS_LIB=liblammps.a
    CMake will exit." )
endif()
if(NOT EXISTS ${TADAH_LAMMPS_LIB})
  message(FATAL_ERROR "The ${TADAH_LAMMPS_LIB} does not exist.")
endif()

#if(TADAH_BUILD_MPI)
#  string(FIND "${TADAH_LAMMPS_LIB}" "mpi" FOUND_INDEX)
#  if(${FOUND_INDEX} EQUAL -1)
#    message(FATAL_ERROR "Error: You are linking the serial LAMMPS library against an MPI-enabled build. Please link the MPI version of #the LAMMPS library.")
#  endif()
#else()
#  string(FIND "${TADAH_LAMMPS_LIB}" "serial" FOUND_INDEX)
#  if(${FOUND_INDEX} EQUAL -1)
#    message(FATAL_ERROR "Error: You are linking the MPI-enabled LAMMPS library against a serial build. Please link the serial version #of the LAMMPS library.")
#
#  endif()
#endif()

# add_compile_definitions(TADAH_LAMMPS_LIB=\"${TADAH_LAMMPS_LIB}\")

file(GLOB HPO_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(tadah.hpo ${HPO_SRC})

target_include_directories(tadah.hpo
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
)
target_link_libraries(tadah.hpo PUBLIC tadah::compile_options )
target_link_libraries(tadah.hpo PRIVATE nlopt)

# add_library(tadah.lammps STATIC IMPORTED GLOBAL)
# set_target_properties(tadah.lammps PROPERTIES
#   IMPORTED_LOCATION
#   ${TADAH_LAMMPS_LIB}
#   #${TADAH_LAMMPS_DIR}/lib/tadah.lammps/build/libtadah.lammps.a
#   INTERFACE_INCLUDE_DIRECTORIES
#   ${TADAH_LAMMPS_DIR}/lib/tadah.lammps/include
#   )

FetchContent_Declare(
    tadah.lammps
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/tadah.lammps.git
    GIT_TAG ${TADAH_GIT_BRANCH}
)
FetchContent_MakeAvailable(tadah.lammps)


# target_link_libraries(tadah.lammps
#     INTERFACE tadah.core tadah.models tadah.md)

if (NOT TADAH_BUILD_DOCS_ONLY)
  find_package(LAPACK REQUIRED)
  target_link_libraries(tadah.hpo PUBLIC ${LAPACK_LIBRARIES})
endif()

if(TADAH_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(tadah.hpo PUBLIC OpenMP::OpenMP_CXX)
endif()

# static LAMMPS lib is compiled withput fPIC so we cannot link it to shared lib
string(FIND "${_lammps_libname}" ".so" POSITION)
if (BUILD_SHARED_LIBS AND POSITION EQUAL -1)
  message(FATAL_ERROR "
    The static LAMMPS library is not compatible with a shared Tadah! build.
    Please either link a shared LAMMPS library or rebuild Tadah! with 

    -DBUILD_SHARED_LIBS=OFF.")
endif()

# Find and link ALL lammps external libraries, this step is necessary for static builds
file(GLOB_RECURSE LAMMPS_ALL_LIBS "${TADAH_LAMMPS_DIR}/lib/*.a" "${TADAH_LAMMPS_DIR}/lib/*.so")

# Create a list to store filtered libraries
set(FILTERED_LAMMPS_LIBS)

# Iterate over found libraries and filter out ones containing "tadah"
foreach(lib ${LAMMPS_ALL_LIBS})
  if(NOT lib MATCHES "tadah")
    list(APPEND FILTERED_LAMMPS_LIBS ${lib})
  endif()
endforeach()

# Replace original list with filtered list
set(LAMMPS_ALL_LIBS ${FILTERED_LAMMPS_LIBS})
add_library(lammps STATIC IMPORTED GLOBAL)
set_target_properties(lammps PROPERTIES
    IMPORTED_LOCATION ${TADAH_LAMMPS_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${TADAH_LAMMPS_DIR}/src 
  )
  target_include_directories(lammps INTERFACE ${tadah.lammps_SOURCE_DIR}/ML-TADAH)
# target_include_directories(lammps INTERFACE ${TADAH_LAMMPS_DIR}/lib/tadah.lammps/ML-TADAH)

# LAMMPS itself needs the plugin
target_link_libraries(lammps INTERFACE tadah.lammps dl ${TADAH_LAMMPS_LIBS} tadah.lammps)
target_link_libraries(tadah.hpo
  PUBLIC
    dlib::dlib
    lammps
    tadah.mlip
    tadah.md            # <- new, provides <tadah/md/...> headers
)

# must be last as we redefine TADAH_LAMMPS_LIB
add_compile_definitions(TADAH_LAMMPS_LIB=\"${_lammps_libname}\")

if (TADAH_BUILD_MPI)
  find_package(MPI REQUIRED)
  set_target_properties(tadah.hpo PROPERTIES OUTPUT_NAME "tadah_mpi.hpo")
  target_link_libraries(tadah.hpo PUBLIC MPI::MPI_CXX)
  add_compile_definitions(TADAH_BUILD_MPI)
else()
  target_include_directories(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS)
  # target_link_libraries(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS/libmpi_stubs.a)
endif()

set(BUILD_TESTING OFF CACHE BOOL "Build tests")
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Installation rules
install(TARGETS tadah.hpo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
    DESTINATION include
)

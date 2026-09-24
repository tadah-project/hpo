cmake_minimum_required(VERSION 3.24)   # OVERRIDE_FIND_PACKAGE for fetched Eigen3
project(Tadah.HPO LANGUAGES CXX C)     # C: vendored ASA library (external/asa)

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
# Developer build (-DTADAH_DEVELOP=ON): use a local cmake-tools checkout
# rather than fetching it. Skipped when the source dir is already pinned
# (e.g. -DFETCHCONTENT_SOURCE_DIR_TADAH_CMAKE=...).
if(TADAH_DEVELOP AND NOT FETCHCONTENT_SOURCE_DIR_TADAH_CMAKE
   AND DEFINED ENV{TADAH_PATH})
  foreach(_tadah_ct "$ENV{TADAH_PATH}/cmake-tools"
                    "$ENV{TADAH_PATH}/../cmake-tools")
    if(IS_DIRECTORY "${_tadah_ct}")
      get_filename_component(FETCHCONTENT_SOURCE_DIR_TADAH_CMAKE
                             "${_tadah_ct}" ABSOLUTE)
      message(STATUS
        "TADAH_DEVELOP: cmake-tools taken from ${FETCHCONTENT_SOURCE_DIR_TADAH_CMAKE}")
      break()
    endif()
  endforeach()
  unset(_tadah_ct)
endif()

if(NOT TARGET tadah::compile_options)
  FetchContent_Declare (
    tadah_cmake
    GIT_REPOSITORY  https://git.ecdf.ed.ac.uk/tadah/cmake-tools.git
    GIT_TAG         v1-dev
  )
  FetchContent_MakeAvailable(tadah_cmake)   # defines tadah::compile_options
endif()

# Declare the BUILD_TESTING switch (default OFF) before any sub-module
# is fetched, so a parent project's value propagates down. See
# cmake-tools/TadahTesting.cmake.
tadah_declare_testing()

option(TADAH_ENABLE_ASAN  "Enable AddressSanitizer (-fsanitize=address -fno-omit-frame-pointer)" OFF)
option(TADAH_ENABLE_CERES "Enable Ceres Solver optimizer family (LBFGS/WOLFE/NLCG/STEEPEST_DESCENT)" ON)

# (option) Developer builds -DTADAH_DEVELOP=ON
tadah_use_local_module(Tadah.MLIP MLIP)          # FetchContent name + folder name
tadah_use_local_module(Tadah.MD   MD)            # FetchContent name + folder name
tadah_use_local_module(tadah.lammps Tadah.LAMMPS) # FetchContent name + folder name

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

# Single-source-of-truth version (from CORE's blueprint TOML). Read after
# the Tadah dependencies are available so CORE's blueprint is found in
# every layout (sibling checkout or FetchContent); then re-state project().
if(NOT COMMAND tadah_read_version_from_blueprint)
  message(FATAL_ERROR "cmake-tools is too old: tadah_read_version_from_blueprint() is missing")
endif()
tadah_read_version_from_blueprint(TADAH_VERSION REQUIRED)
project(Tadah.HPO VERSION ${TADAH_VERSION} LANGUAGES CXX C)
message(STATUS "Tadah.HPO version: ${TADAH_VERSION}")

# 5. External dependencies
#
# Shield every external library from BUILD_TESTING for the whole fetch
# region so none of them build their own test suites (Ceres in
# particular keys its tests off BUILD_TESTING). The cache value is
# force-set OFF here and restored by tadah_end_external_deps() below.
tadah_begin_external_deps()

FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24.6
)
FetchContent_MakeAvailable(dlib)
add_compile_definitions(TADAH_USE_DLIB)

# ---------------------------------------------------------------------------
# Ceres Solver — used by the CERES:LBFGS/WOLFE/NLCG/STEEPEST_DESCENT optimizer
# family in HPO and by the unified GradientProvider (Ceres NumericDiff).
# Block goes BEFORE nlopt so Ceres' fetched Eigen3 is in scope before nlopt's
# own Eigen probe.
# ---------------------------------------------------------------------------
if(NOT TADAH_BUILD_DOCS_ONLY AND TADAH_ENABLE_CERES)
  message(STATUS "Fetching Eigen3 (Ceres dep)...")
  set(EIGEN_BUILD_DOC          OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING      OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG    OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(Eigen3)

  message(STATUS "Fetching Ceres Solver...")
  set(MINIGLOG                 ON  CACHE BOOL "" FORCE)
  set(GFLAGS                   OFF CACHE BOOL "" FORCE)
  set(SUITESPARSE              OFF CACHE BOOL "" FORCE)
  set(EIGENSPARSE              OFF CACHE BOOL "" FORCE)
  set(ACCELERATESPARSE         OFF CACHE BOOL "" FORCE)
  set(LAPACK                   OFF CACHE BOOL "" FORCE)   # HPO links its own
  set(USE_CUDA                 OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES           OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARKS         OFF CACHE BOOL "" FORCE)
  set(BUILD_DOCUMENTATION      OFF CACHE BOOL "" FORCE)
  set(SCHUR_SPECIALIZATIONS    OFF CACHE BOOL "" FORCE)
  set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ceres
    GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
    GIT_TAG        2.2.0
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(ceres)
endif()

set(NLOPT_TESTS OFF CACHE BOOL "Build nlopt's own unit tests" FORCE)
FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(nlopt)

# End of the external-dependency fetch region: restore BUILD_TESTING.
tadah_end_external_deps()

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

# add_compile_definitions(TADAH_LAMMPS_LIB=\"${TADAH_LAMMPS_LIB}\")

# Vendored third-party ASA-30.53 (BSD-3, Lester Ingber) -- LIB ASA optimizer
add_subdirectory(external/asa)

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
target_link_libraries(tadah.hpo PRIVATE tadah_asa)

# Phase 3 LammpsHessianAssembler uses Eigen3 directly (already fetched
# via the Ceres dependency block above). PUBLIC because the public
# header lammps_hessian.h returns Eigen::MatrixXd.
if(TARGET Eigen3::Eigen)
    target_link_libraries(tadah.hpo PUBLIC Eigen3::Eigen)
endif()

if(TADAH_ENABLE_CERES)
  # PUBLIC: gradient_provider.h is in tadah.hpo's public include, and
  # downstream consumers (e.g. test_hpo, future drivers) need both the
  # TADAH_USE_CERES define and the Ceres include directories to compile
  # the templated factory's numerical branch.  When OFF, the header
  # compiles to the autodiff-stub-only variant.
  target_link_libraries     (tadah.hpo PUBLIC Ceres::ceres)
  target_compile_definitions(tadah.hpo PUBLIC TADAH_USE_CERES)

  # Mark Ceres' include dirs as SYSTEM so the inevitable miniglog CHECK
  # macro redefinition (vs. Catch2 / downstream libraries) does not fire
  # `-Werror`-promoted "macro redefined" diagnostics in consumer TUs.
  # Apply to both the local target and as an INTERFACE on the imported
  # Ceres target (so transitive consumers also see the dirs as -isystem).
  get_target_property(_ceres_inc Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
  if(_ceres_inc)
    target_include_directories(tadah.hpo SYSTEM PUBLIC ${_ceres_inc})
  endif()
endif()

if(TADAH_ENABLE_ASAN)
  target_compile_options(tadah.hpo PUBLIC -fsanitize=address -fno-omit-frame-pointer -g)
  target_link_libraries(tadah.hpo PUBLIC -fsanitize=address)
endif()

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
  target_compile_definitions(tadah.hpo PUBLIC TADAH_ENABLE_OPENMP)
endif()

# static LAMMPS lib is compiled without fPIC so we cannot link it to shared lib
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

# Opt-in MPI for the HPO parallel-annealing drivers (TADAH:ANNEAL_PT /
# TADAH:ANNEAL_PA): replicas/walkers distribute over ranks, every rank
# trains locally (plain Trainer) and runs its LAMMPS instances on
# MPI_COMM_SELF. This is the only MPI build of Tadah: the legacy
# ScaLAPACK training build (TADAH_BUILD_MPI) was removed in 1.5.1, and
# cmake-tools rejects that option at configure time.
option(TADAH_HPO_MPI
  "Distribute TADAH:ANNEAL_PT / ANNEAL_PA replicas over MPI ranks (requires an MPI LAMMPS library)"
  OFF)

if (TADAH_HPO_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  # PUBLIC on tadah.hpo: the `tadah` executable inherits the link and the
  # compile definition transitively -- no Tadah.MLIP CMake change, no
  # binary rename (an MPI-linked tadah still runs standalone).
  target_link_libraries(tadah.hpo PUBLIC MPI::MPI_CXX)
  target_compile_definitions(tadah.hpo PUBLIC TADAH_HPO_MPI)
  # The serial LAMMPS archive contains the MPI STUB symbol DEFINITIONS
  # (src/STUBS/mpi.cpp) -- linking it together with a real libmpi is an
  # ODR/symbol-collision hazard. Require an MPI-flavoured LAMMPS lib.
  if (NOT _lammps_libname MATCHES "mpi" AND NOT TADAH_HPO_MPI_SKIP_LIB_CHECK)
    message(FATAL_ERROR
      "TADAH_HPO_MPI=ON requires an MPI LAMMPS library (got "
      "'${_lammps_libname}'): the serial archive carries MPI stub symbol "
      "definitions that collide with the real MPI library at link time. "
      "Point TADAH_LAMMPS_LIB at the MPI build (e.g. "
      "build_lammps_mpi/liblammps_develop_mpi.a) or set "
      "-DTADAH_HPO_MPI_SKIP_LIB_CHECK=ON if your lib name simply lacks "
      "the 'mpi' marker.")
  endif()
  message(STATUS "TADAH_HPO_MPI: ANNEAL_PT/ANNEAL_PA replicas distribute over MPI ranks (LAMMPS on MPI_COMM_SELF)")
else()
  target_include_directories(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS)
  # target_link_libraries(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS/libmpi_stubs.a)
endif()

tadah_add_tests()

# Installation rules
install(TARGETS tadah.hpo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
# tadah.hpo is a static archive and does not embed the vendored ASA
# objects; install the archive so downstream links resolve.
install(TARGETS tadah_asa ARCHIVE DESTINATION lib)
install(DIRECTORY include/
    DESTINATION include
)

# Note: 
# For non-MPI version link serial version of LAMMPS
# Tadah MPI version requires MPI version of LAMMPS

cmake_minimum_required(VERSION 3.12)
project(Tadah.HPO)

option(TADAH_LAMMPS_DIR "Path to lammps root directory")
option(TADAH_LAMMPS_LIB "Name of the lammps library")

# Guard against in-source builds
# Also, make sure user does not accidently invoke cmake .. from the source dir
# of the module in case module is added as a submodule.
get_filename_component(MODULE_TEST ${CMAKE_BINARY_DIR} NAME)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR}
        OR ${MODULE_TEST} STREQUAL "HPO"
)
  message(FATAL_ERROR "
    In-source builds not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt and CMakeFiles dir. ")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_SHARED_LIBS)
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

  # the RPATH to be used when installing, but only if it's not a system directory
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
endif()


include(FetchContent)

if(DEFINED ENV{CI_COMMIT_REF_NAME})
  set(GIT_BRANCH "$ENV{CI_COMMIT_REF_NAME}")
else()
  execute_process(
    COMMAND bash -c "git status | grep origin | sed -e 's/.*origin\\///' -e 's/[^a-zA-Z].*//'" 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "${TADAH}.HPO Using GIT_BRANCH:  ${GIT_BRANCH}.")

FetchContent_Declare(
    Tadah.MLIP
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/mlip.git
    GIT_TAG origin/${GIT_BRANCH}
)
FetchContent_MakeAvailable(Tadah.MLIP)

include(FetchContent)
FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24.6
)
FetchContent_MakeAvailable(dlib)

# User must specify the root directory for lammps
if(NOT DEFINED TADAH_LAMMPS_DIR)
  message( FATAL_ERROR 
    "
    TADAH_LAMMPS_DIR is not specified.
    example:
    -DTADAH_LAMMPS_DIR=/path/to/lammps
    CMake will exit." )
endif()
if(NOT EXISTS ${TADAH_LAMMPS_DIR})
  message(FATAL_ERROR "The ${TADAH_LAMMPS_DIR} does not exist.")
endif()

# we need to know which LAMMPS to use
# note that LAMMPS lib must by compiles with tadah.lammps
if(NOT DEFINED TADAH_LAMMPS_LIB)
  message( FATAL_ERROR 
    "
    TADAH_LAMMPS_LIB is not specified.
    example:
    DTADAH_LAMMPS_LIB=liblammps.a
    CMake will exit." )
endif()
if(NOT EXISTS ${TADAH_LAMMPS_DIR}/src/${TADAH_LAMMPS_LIB})
  message(FATAL_ERROR "The ${TADAH_LAMMPS_DIR}/src/${TADAH_LAMMPS_LIB} does not exist.")
endif()

if(TADAH_BUILD_MPI)
  string(FIND "${TADAH_LAMMPS_LIB}" "mpi" FOUND_INDEX)
  if(${FOUND_INDEX} EQUAL -1)
    message(FATAL_ERROR "Error: You are linking the serial LAMMPS library against an MPI-enabled build. Please link the MPI version of the LAMMPS library.")
  endif()
else()
  string(FIND "${TADAH_LAMMPS_LIB}" "serial" FOUND_INDEX)
  if(${FOUND_INDEX} EQUAL -1)
    message(FATAL_ERROR "Error: You are linking the MPI-enabled LAMMPS library against a serial build. Please link the serial version of the LAMMPS library.")

  endif()
endif()


add_compile_definitions(TADAH_LAMMPS_LIB=\"${TADAH_LAMMPS_LIB}\")

file(GLOB HPO_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(tadah.hpo ${HPO_SRC})
target_include_directories(tadah.hpo PUBLIC include)
target_include_directories(tadah.hpo PUBLIC ${Tadah.MLIP_SOURCE_DIR}/include)
target_include_directories(tadah.hpo PUBLIC ${Tadah.MODELS_SOURCE_DIR}/include)
target_include_directories(tadah.hpo PUBLIC ${Tadah.CORE_SOURCE_DIR}/include)
target_include_directories(tadah.hpo PUBLIC ${Tadah.CORE_SOURCE_DIR}/external/toml11)
target_include_directories(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src)

if(TADAH_BUILD_MPI)
  set_target_properties(tadah.hpo PROPERTIES OUTPUT_NAME "tadah_mpi.hpo")
endif()

find_package(LAPACK)
target_link_libraries(tadah.hpo PUBLIC ${LAPACK_LIBRARIES})

if(TADAH_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(tadah.hpo PUBLIC OpenMP::OpenMP_CXX)
endif()

if (TADAH_BUILD_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(tadah.hpo PUBLIC MPI::MPI_CXX)
  add_compile_definitions(TADAH_BUILD_MPI)
else()
  target_include_directories(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS)
endif()

# static LAMMPS lib is compiled withput fPIC so we cannot link it to shared lib
string(FIND "${TADAH_LAMMPS_LIB}" ".so" POSITION)
if (BUILD_SHARED_LIBS AND POSITION EQUAL -1)
  message(FATAL_ERROR "
    The static LAMMPS library is not compatible with a shared Tadah! build.
    Please either link a shared LAMMPS library or rebuild Tadah! with 

    -DBUILD_SHARED_LIBS=OFF.")
endif()

add_library(liblammps INTERFACE)
target_link_libraries(liblammps INTERFACE ${TADAH_LAMMPS_DIR}/src/${TADAH_LAMMPS_LIB})
target_link_libraries(liblammps INTERFACE ${TADAH_LAMMPS_DIR}/lib/tadah.lammps/build/libtadah.lammps.a)
target_link_libraries(liblammps INTERFACE dl)  # required by lammps

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

target_link_libraries(liblammps INTERFACE ${LAMMPS_ALL_LIBS})

target_link_libraries(tadah.hpo PUBLIC dlib::dlib)
target_link_libraries(tadah.hpo PUBLIC liblammps)
target_link_libraries(tadah.hpo PUBLIC config::rc)

if (TADAH_BUILD_MPI)
  target_link_libraries(tadah.hpo PUBLIC MPI::MPI_CXX)
else()
  target_link_libraries(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS/libmpi_stubs.a)
endif()

if(TADAH_BUILD_TESTS)
  include(CTest) 
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

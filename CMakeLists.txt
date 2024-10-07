cmake_minimum_required(VERSION 3.12)
project(Tadah.HPO)

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

#if(BUILD_SHARED_LIBS)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#endif()

#if (BUILD_SHARED_LIBS)
#    # Always full RPATH
#    # use, i.e. don't skip the full RPATH for the build tree
#    set(CMAKE_SKIP_BUILD_RPATH FALSE)
#
#    # when building, don't use the install RPATH already
#    # (but later on when installing)
#    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#
#    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#
#    # add the automatically determined parts of the RPATH
#    # which point to directories outside the build tree to the install RPATH
#    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#
#    # the RPATH to be used when installing, but only if it's not a system directory
#    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#    if("${isSystemDir}" STREQUAL "-1")
#        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#    endif("${isSystemDir}" STREQUAL "-1")
#endif()


include(FetchContent)
get_filename_component(BUILD_DIR_NAME ${CMAKE_BINARY_DIR} NAME)
FetchContent_Declare(
    Tadah.MLIP
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/mlip.git
    GIT_TAG origin/main
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MLIP
    PATCH_COMMAND  mkdir -p ${BUILD_DIR_NAME}
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MLIP/${BUILD_DIR_NAME}
    )

FetchContent_MakeAvailable(Tadah.MLIP)

include(FetchContent)
FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.24
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
add_compile_definitions(TADAH_LAMMPS_LIB=\"${TADAH_LAMMPS_LIB}\")

add_compile_definitions(TADAH_LMPA=\"${TADAH_LAMMPS_DIR}/src/lammps.h\")
add_compile_definitions(TADAH_LMPB=\"${TADAH_LAMMPS_DIR}/src/input.h\")
add_compile_definitions(TADAH_LMPC=\"${TADAH_LAMMPS_DIR}/src/variable.h\")

# Add resources...
cmrc_add_resource_library(
  hpo-resources 
  ALIAS hpo::rc
  NAMESPACE hpo 
  WHENCE lammps
  PREFIX hpo
  ${PROJECT_SOURCE_DIR}/lammps/tadah.lata
  ${PROJECT_SOURCE_DIR}/lammps/tadah.envol
  ${PROJECT_SOURCE_DIR}/lammps/tadah.elastic
  ${PROJECT_SOURCE_DIR}/lammps/tadah.surface
  ${PROJECT_SOURCE_DIR}/lammps/tadah.usurface
  ${PROJECT_SOURCE_DIR}/lammps/tadah.FR_vacancy_at_p
  ${PROJECT_SOURCE_DIR}/lammps/tadah.unrelaxed_vacancy_at_p
  )

file(GLOB HPO_SRC CONFIGURE_DEPENDS "*.cpp" "*.h" "*.hpp")

add_library(tadah.hpo.obj OBJECT
    ${HPO_SRC}
    )
target_link_libraries(tadah.hpo.obj PUBLIC config::rc)
target_link_libraries(tadah.hpo.obj PUBLIC dlib::dlib)
#target_link_libraries(tadah.hpo.obj PUBLIC hpo::rc)

find_package(LAPACK)
target_link_libraries(tadah.hpo.obj PUBLIC ${LAPACK_LIBRARIES})
#target_link_directories(tadah.hpo.obj PUBLIC ${TADAH_LAMMPS_DIR}/src)
#target_link_libraries(tadah.hpo.obj PUBLIC ${TADAH_LAMMPS_LIB})


if(TADAH_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tadah.hpo.obj PUBLIC OpenMP::OpenMP_CXX)
endif()

if (TADAH_BUILD_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(tadah.hpo.obj PUBLIC MPI::MPI_CXX)
    add_compile_definitions(TADAH_BUILD_MPI)
else()
    target_include_directories(tadah.hpo.obj PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS)
endif()

add_library(liblammps STATIC IMPORTED)
set_target_properties(liblammps PROPERTIES IMPORTED_LOCATION ${TADAH_LAMMPS_DIR}/src/${TADAH_LAMMPS_LIB})
#target_link_libraries(liblammps INTERFACE ${TADAH_LAMMPS_DIR}/lib/tadah.lammps/build/libtadah.lammps.so)
target_link_libraries(liblammps INTERFACE ${TADAH_LAMMPS_DIR}/lib/tadah.lammps/build/libtadah.lammps.a)
target_link_libraries(liblammps INTERFACE dl)  # required by lammps

add_library(tadah.hpo STATIC $<TARGET_OBJECTS:tadah.hpo.obj>)
target_link_libraries(tadah.hpo PUBLIC dlib::dlib)
target_link_libraries(tadah.hpo PUBLIC liblammps)

#target_link_directories(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src)
#target_link_libraries(tadah.hpo PUBLIC ${TADAH_LAMMPS_LIB})
target_link_libraries(tadah.hpo PUBLIC liblammps)
if (NOT TADAH_BUILD_MPI)
    target_link_libraries(tadah.hpo PUBLIC ${TADAH_LAMMPS_DIR}/src/STUBS/libmpi_stubs.a)
endif()
target_link_libraries(tadah.hpo PUBLIC hpo::rc)

if(TADAH_BUILD_TESTS)
    include(CTest) 
    add_subdirectory(tests)
endif()

set(TADAH_HPO_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include/tadah/HPO)
file(GLOB HPO_H CONFIGURE_DEPENDS "*.h" "*.hpp")
install(FILES ${HPO_H} DESTINATION ${TADAH_HPO_INSTALL_DIR})

# Checks that the handlers of the registry-driven parser follow the keys of
# the registry (blueprint/README.md, "The handlers"): src/hpotarget_handlers.cpp
# defines one handler for each key, which hpotarget_handlers_gen.h declares
# and whose address the parser takes (kKeyHandlers). On copies of the
# registry, each regenerated into the include directory the probe targets
# put first (tests/CMakeLists.txt, hpo_hpotarget_handlers_probe and
# hpo_hpotarget_handlers_probe_objects):
#   1. ROOT.QFLUSH removed: hpotarget_handlers.cpp does not compile, and the
#      compiler names the handler ROOT_QFLUSH, which no key declares;
#   2. ROOT.QFLUSHX added (a copy of ROOT.QFLUSH): hpotarget_handlers.cpp
#      compiles, and the probe, which takes the address of every handler,
#      does not link: nothing defines ROOT_QFLUSHX.
# The include directory is emptied at the start and at the end, so the
# probe targets compile with the headers of the repository otherwise.
#
#   cmake -DPYTHON=<python3> -DHPO_SOURCE_DIR=<HPO> -DBUILD_DIR=<build tree>
#         -DPROBE_INCLUDE_DIR=<directory> -DWORK_DIR=<scratch>
#         -DOBJECTS_TARGET=<target> -DPROBE_TARGET=<target>
#         -P check_handlers_removal.cmake
cmake_minimum_required(VERSION 3.24)

foreach(v PYTHON HPO_SOURCE_DIR BUILD_DIR PROBE_INCLUDE_DIR WORK_DIR
          OBJECTS_TARGET PROBE_TARGET)
  if(NOT DEFINED ${v} OR "${${v}}" STREQUAL "")
    message(FATAL_ERROR "check_handlers_removal.cmake: -D${v}=... is required")
  endif()
endforeach()

set(handlers "${HPO_SOURCE_DIR}/src/hpotarget_handlers.cpp")
if(NOT EXISTS "${handlers}")
  message(FATAL_ERROR "${handlers} does not exist: the parser has no "
                      "handlers to hold to the keys of the registry")
endif()

set(blueprint "${HPO_SOURCE_DIR}/blueprint")
set(headers "${PROBE_INCLUDE_DIR}/tadah/hpo")

# The [[key]] table of ROOT.QFLUSH in hpotarget.toml, its examples and
# invalid lines included: from its "[[key]]" line to the next one.
file(READ "${blueprint}/hpotarget.toml" main)
string(FIND "${main}" "[[key]]\nid = \"ROOT.QFLUSH\"\n" begin)
if(begin EQUAL -1)
  message(FATAL_ERROR "hpotarget.toml has no [[key]] table of ROOT.QFLUSH, "
                      "the key the check removes and copies")
endif()
string(SUBSTRING "${main}" ${begin} -1 rest)
string(LENGTH "[[key]]" head)
string(SUBSTRING "${rest}" ${head} -1 after)
string(FIND "${after}" "\n[[key]]\n" next)
if(next EQUAL -1)
  message(FATAL_ERROR "the [[key]] table of ROOT.QFLUSH is the last of "
                      "hpotarget.toml; the check cuts it before the next one")
endif()
math(EXPR length "${head} + ${next} + 1")
string(SUBSTRING "${rest}" 0 ${length} qflush)
string(REPLACE "${qflush}" "" removed "${main}")
string(REPLACE "QFLUSH" "QFLUSHX" copy "${qflush}")
string(REPLACE "${qflush}" "${qflush}${copy}" added "${main}")

function(clear_headers)
  file(REMOVE_RECURSE "${PROBE_INCLUDE_DIR}")
endfunction()

# write_registry(<name> <text of hpotarget.toml>): a copy of the registry
# with that hpotarget.toml, regenerated into the probe's include directory.
function(write_registry name text)
  set(dir "${WORK_DIR}/${name}")
  file(REMOVE_RECURSE "${dir}")
  file(MAKE_DIRECTORY "${dir}")
  file(GLOB parts "${blueprint}/*.toml" "${blueprint}/*.py")
  file(COPY ${parts} DESTINATION "${dir}")
  file(WRITE "${dir}/hpotarget.toml" "${text}")
  clear_headers()
  execute_process(
    COMMAND "${PYTHON}" "${blueprint}/gen_hpotarget.py" --header
            --registry "${dir}" -o "${headers}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  if(NOT rc EQUAL 0)
    clear_headers()
    message(FATAL_ERROR "${name}: the generator refused the registry "
                        "(exit ${rc}):\n${out}")
  endif()
endfunction()

# build(<target> <result variable> <output variable>)
function(build target rc_var out_var)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${target}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE out)
  set(${rc_var} "${rc}" PARENT_SCOPE)
  set(${out_var} "${out}" PARENT_SCOPE)
endfunction()

set(failures "")

# 1. A key removed: the handlers do not compile.
write_registry(removed "${removed}")
build("${OBJECTS_TARGET}" rc out)
if(rc EQUAL 0)
  string(APPEND failures "with ROOT.QFLUSH removed, hpotarget_handlers.cpp "
                         "compiled: its handler ROOT_QFLUSH is defined for "
                         "no key\n")
elseif(NOT out MATCHES "hpotarget_handlers\\.cpp" OR
       NOT out MATCHES "ROOT_QFLUSH")
  string(APPEND failures "with ROOT.QFLUSH removed, the build failed "
                         "without naming the handler ROOT_QFLUSH of "
                         "hpotarget_handlers.cpp:\n${out}\n")
endif()

# 2. A key added: the handlers compile, the probe does not link.
write_registry(added "${added}")
build("${OBJECTS_TARGET}" rc out)
if(NOT rc EQUAL 0)
  string(APPEND failures "with ROOT.QFLUSHX added, hpotarget_handlers.cpp "
                         "did not compile:\n${out}\n")
else()
  build("${PROBE_TARGET}" rc out)
  if(rc EQUAL 0)
    string(APPEND failures "with ROOT.QFLUSHX added, the probe linked: "
                           "a handler of a key is defined nowhere and "
                           "nothing failed\n")
  elseif(NOT out MATCHES "undefined reference" OR
         NOT out MATCHES "ROOT_QFLUSHX")
    string(APPEND failures "with ROOT.QFLUSHX added, the probe did not "
                           "build, but not for the missing handler "
                           "ROOT_QFLUSHX:\n${out}\n")
  endif()
endif()

clear_headers()
if(failures)
  message(FATAL_ERROR "${failures}")
endif()
message(STATUS "a key removed from the registry fails to compile the "
               "handlers; a key added fails to link the parser")

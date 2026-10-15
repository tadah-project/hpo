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

# The LAMMPS-driven cases (bench_pot.h) run on the Ta potential shipped
# with the tests, data/bench_ta.tadah (its header gives its provenance);
# TADAH_BENCH_POT names another at run time, and TADAH_BENCH_CELL_DATA (a
# LAMMPS read_data file) the cell of the pair_tadah microbench. Catch2's
# BENCHMARK macros are enabled for the microbenches.
target_compile_definitions(test_hpo PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
target_compile_definitions(test_hpo PRIVATE
    "TADAH_BENCH_DEFAULT_POT=\"${CMAKE_CURRENT_SOURCE_DIR}/data/bench_ta.tadah\"")

# Worked examples (HPO/examples). test_examples_parse.cpp gives every
# examples/*/config.hpo a test case of its own: the list is generated here,
# and CONFIGURE_DEPENDS re-globs the directory on every build, so a new
# example is picked up (and fails until examples/tools/parse_guard.manifest
# names what it needs). The PC_OBSERVABLE gradient tests
# (test_hpo_host_e2e.cpp) train on the examples' shared Ta data.
cmake_path(SET _hpo_examples_dir NORMALIZE
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples")
file(GLOB _hpo_example_configs CONFIGURE_DEPENDS
    "${_hpo_examples_dir}/*/config.hpo")
list(SORT _hpo_example_configs)
set(_hpo_example_cases "")
foreach(_cfg IN LISTS _hpo_example_configs)
    cmake_path(GET _cfg PARENT_PATH _dir)
    cmake_path(GET _dir FILENAME _name)
    string(APPEND _hpo_example_cases "TADAH_EXAMPLE_PARSE_CASE(\"${_name}\")\n")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/examples_parse_cases.inc"
     CONTENT "${_hpo_example_cases}" @ONLY)
target_include_directories(test_hpo PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(test_hpo PRIVATE
    "TADAH_HPO_EXAMPLES_DIR=\"${_hpo_examples_dir}\"")

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
# [.corana1987] (one test case), armed by the transcribed reference table
# tests/data/corana1987_protocol.tsv. While the table is empty the case
# calls TADAH_SKIP and the entry is reported as Skipped (cmake-tools'
# TADAH-SKIP marker; include(Catch) sets the regular expression).
add_test(NAME hpo_corana1987_protocol
    COMMAND test_hpo "[.corana1987]" --warn NoTests)
set_tests_properties(hpo_corana1987_protocol PROPERTIES
    LABELS "anneal;statistical"
    TIMEOUT 900
    SKIP_REGULAR_EXPRESSION "${TADAH_TEST_SKIP_REGULAR_EXPRESSION}")

# Real-signal termination tests (test_hpo_termination.cpp; bfgs_not_run in
# test_repeat_points.cpp): each case raises SIGINT/SIGTERM/SIGUSR2 against a
# live hpo_run() or optimiser search. CORE's stop flag can only
# be raised by a real signal and is never lowered, and the HPOLog / HPO_Host
# statics persist, so every case is a hidden tag with an entry -- a process --
# of its own. `ctest -L shutdown` runs them. NB: no ';' in add_test()
# arguments (MLIP's add_test() override splits them).
set(_hpo_shutdown_cases
    dlib sigterm sigusr2 nlopt multistart prestage gfs anneal_pt anneal_pa
    anneal_pa_threads asa asa_block asa_periodic asa_sampling noise
    no_usable_loss bfgs_not_run)
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

# Host end-to-end groups: hidden tags, like the other HPO_Host end-to-end
# cases, because they drive real hpo_run()/HPO_Host evaluations whose
# process-wide state (HPOLog, the HPO_Host statics, the thread-local
# trainer cache) the cases of a group share; each group is one entry -- a
# process -- of its own. `ctest -L e2e` runs them.
#   fixweight_e2e  test_hpo_fixweight_e2e.cpp: pinned/OPTIM'd weights, exact
#                  potential files, seed anchors, LAMMPS read-backs, the
#                  weights' health monitor (HEALTH_LOG)
#   durable_e2e    test_hpo_fixweight_e2e.cpp: best-potential persistence and
#                  run endings; spawns test_hpo children
#                  ([.durable_e2e_child], [.durable_latch_child])
#   embf0_e2e      test_hpo_embf0_e2e.cpp: DM_REAM EMBF0 per candidate
#   rcut_e2e       test_rcut_autoderive.cpp: knot RCUT re-derivation
#   rhoform_e2e    test_hpo_rhoform.cpp: RHOFORM through hpo_run (an invalid
#                  density form stops the run; a NONNEG/MONOTONE amplitude
#                  below 0, outside its box, is not evaluated,
#                  [rhoform_projection_e2e], one of its cases)
#   rhogate_e2e    test_rho_gates.cpp: RHOGATE rejects before training
#   rhobargate_e2e test_rho_gates.cpp: RHOBARGATE rejects before training
#   bumpgate_e2e   test_bump_support_gate.cpp: BUMPGATE rejects before
#                  training (also tagged [!benchmark])
#   nn_e2e         test_nn_exact_e2e.cpp: exact neighbour lists -- thin,
#                  triclinic, sheared and unwrapped validation cells give
#                  the loss of their supercells; the one-time lists cover
#                  every cutoff of the OPTIM box, and a candidate cutoff
#                  beyond the box is not evaluated
#   valmetric_e2e  test_validation_follows_metric_e2e.cpp: validation
#                  predicts forces/stresses when PC_FRMSE/PC_SRMSE ask,
#                  whatever FORCE/STRESS training used, and leaves out the
#                  structures stored with force/stress weight 0; and
#                  test_metric_unit_labels.cpp: the --skip-above message
#                  gives ErRMSE and SRMSE in their units
#   pc_readback_e2e test_hpo_pc_readback_e2e.cpp: a LAMMPS error while a
#                  PC_LAMMPS variable is read back fails the evaluation
#                  (--varloss, --hotvar) or is NaN (a diagnostic --outvar)
#                  under PC_ISOLATION THREAD, and every row has its
#                  header's columns
#   trainer_e2e    test_hpo_fresh_trainer_e2e.cpp: every evaluation trains
#                  a model built from its own candidate (row weights,
#                  MPARAMS, SIGMA, a second host on a thread, an M_KRR -c
#                  potential, gradients of another host's model)
#   hotvar_e2e     test_hpo_pc_readback_e2e.cpp: a second run in one process
#                  starts each PC_LAMMPS --hotvar from its declared initial
#                  value
set(_hpo_e2e_groups fixweight_e2e durable_e2e embf0_e2e rcut_e2e
                    rhoform_e2e rhogate_e2e rhobargate_e2e bumpgate_e2e
                    nn_e2e valmetric_e2e pc_readback_e2e trainer_e2e
                    hotvar_e2e)
foreach(_group IN LISTS _hpo_e2e_groups)
    add_test(NAME hpo_${_group}
        COMMAND test_hpo "[${_group}]" --warn NoTests)
    set_tests_properties(hpo_${_group} PROPERTIES
        LABELS "hpo;e2e"
        TIMEOUT 600)
endforeach()

# LAMMPS-driven cases: hidden tags (`[.][!benchmark]`), each group one
# entry -- a process -- of its own; the first three on the shipped
# potential (data/bench_ta.tadah, bench_pot.h). `ctest -L bench` runs them;
# they take a few seconds each and run in the full suite.
#   hpo_lammps_hessian    test_lammps_hessian.cpp: the 4-atom bcc Ta
#                         Hessian (symmetry, acoustic modes, an H.v probe,
#                         the iso and Voigt-6 box blocks, Tikhonov, the
#                         adjoint snapshot) and the vflag=1 microbench; the
#                         two LAMMPS-free cases of the file are discovered
#   hpo_phase3_pilot      test_hpo_phase3_pilot.cpp: the PC_LAMMPS adjoint
#                         chain on real relaxations (closed-form cohesive
#                         energy, dvirial_dW against finite differences,
#                         dlx/dW against re-relaxed finite differences, the
#                         SimsGradProvider body with and without a hot
#                         variable)
#   hpo_bench_pair_tadah  bench_pair_tadah.cpp: pair_tadah::compute on 256
#                         atoms (Catch2 BENCHMARK)
#   hpo_host_e2e          test_hpo_host_e2e.cpp: HPO_Host's analytical
#                         gradient with no PC_LAMMPS line, with a
#                         PC_OBSERVABLE and with a PC_LAMMPS script run in
#                         process (its own 4-atom training structure; no
#                         potential file)
# With the bump-gate case of bumpgate_e2e above, every hidden [!benchmark]
# case has an entry.
add_test(NAME hpo_lammps_hessian
    COMMAND test_hpo "[lammps_hessian][.]" --warn NoTests)
add_test(NAME hpo_phase3_pilot
    COMMAND test_hpo "[phase3_pilot],[phase3_orchestrator]" --warn NoTests)
add_test(NAME hpo_bench_pair_tadah
    COMMAND test_hpo "[bench][pair_tadah]" --warn NoTests)
add_test(NAME hpo_host_e2e
    COMMAND test_hpo "[hpo_host_e2e]" --warn NoTests)
set_tests_properties(hpo_lammps_hessian hpo_phase3_pilot hpo_bench_pair_tadah
                     hpo_host_e2e
    PROPERTIES
        LABELS "hpo;bench"
        TIMEOUT 300)

# End-to-end cases that run the built `tadah` executable, named by the
# TADAH_BINARY environment variable (test_hpo_fixweight_e2e.cpp and
# test_hpo_pc_readback_e2e.cpp, tag [tadah_cli]). Each is a hidden tag
# with an entry of its own, so that the
# TADAH-SKIP marker applies to exactly one case: without TADAH_BINARY a
# case calls TADAH_SKIP and CTest reports it as Skipped, not Passed. In a
# build that has the `tadah` executable target (Tadah.MLIP, which adds this
# directory) the entry sets TADAH_BINARY to it, so the case runs; in one
# without it (a standalone HPO build) TADAH_BINARY comes from the
# environment of ctest, if at all. In an MPI build (TADAH_HPO_MPI)
# TADAH_MPIEXEC is the launcher of one rank ("<MPIEXEC_EXECUTABLE> -n 1"),
# through which a case that changes the environment Open MPI starts in
# (TMPDIR) runs `tadah` (test_hpo_pc_readback_e2e.cpp).
set(_hpo_tadah_cli_env
    "$<$<TARGET_EXISTS:tadah>:TADAH_BINARY=$<TARGET_FILE:tadah>>")
if(TADAH_HPO_MPI)
    list(APPEND _hpo_tadah_cli_env
        "TADAH_MPIEXEC=${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1")
endif()
set(_hpo_tadah_cli_cases seed_anchor_warm seed_anchor_nlopt durable_exit
                         pc_readback pc_lammps_local_search pc_lammps_verbose
                         kernel_line_once kernel_line_two_runs)
foreach(_case IN LISTS _hpo_tadah_cli_cases)
    add_test(NAME hpo_tadah_cli_${_case}
        COMMAND test_hpo "[.tadah_cli_${_case}]" --warn NoTests)
    set_tests_properties(hpo_tadah_cli_${_case} PROPERTIES
        LABELS "hpo;e2e"
        TIMEOUT 600
        SKIP_REGULAR_EXPRESSION "${TADAH_TEST_SKIP_REGULAR_EXPRESSION}"
        ENVIRONMENT "${_hpo_tadah_cli_env}")
endforeach()

# Registry drift gate: the checked-in anneal_keys_gen.h must match what the
# generator produces from blueprint/anneal_keys.toml. Regenerate with
#   python3 HPO/blueprint/gen_anneal_keys.py --header
# The interpreter of these scripts: cmake/hpo_test_python.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hpo_test_python.cmake)
# hpo_test_python_minimum_{old,current}: an interpreter that reports Python
# 3.10 is refused at the configure step, the build's own is found
# (python_minimum/check_python_minimum.cmake). Before the fix: the 'old'
# case fails, a 3.10 interpreter was accepted (and the registry gate failed
# at test time, 'No module named tomllib'); 'current' passes as well (a pin).
foreach(_case old current)
    add_test(NAME hpo_test_python_minimum_${_case}
        COMMAND ${CMAKE_COMMAND}
            -DCASE=${_case}
            -DPYTHON=${Python3_EXECUTABLE}
            -DMODULE=${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hpo_test_python.cmake
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/python_minimum/${_case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/python_minimum/check_python_minimum.cmake)
    set_tests_properties(hpo_test_python_minimum_${_case} PROPERTIES
        LABELS cmake
        TIMEOUT 120)
endforeach()
# hpo_configure_without_ceres: HPO configured without Ceres
# (TADAH_ENABLE_CERES OFF) gives tadah.hpo Eigen all the same, which its
# public headers adjoint_solver.h, lammps_hessian.h, observable_registry.h
# and sims_grad.h include in every build
# (configure_without_ceres/check_configure_without_ceres.cmake). HPO is
# configured once more, without Ceres and without tests, from this build's
# cache and the source directories of its dependencies (the initial cache
# hpo_write_no_ceres_preset() writes here, configure_without_ceres/
# no_ceres_preset.cmake; FETCHCONTENT_FULLY_DISCONNECTED, nothing is
# downloaded; the environment is CTest's), and the check
# reads its codemodel (CMake's file API): no ceres target, no
# TADAH_USE_CERES, and an include directory of tadah.hpo that has
# Eigen/Dense. Before the fix: Eigen was fetched only with Ceres and linked
# only when its target existed, so the check fails ("no include directory
# of tadah.hpo has Eigen/Dense ..."), and a build with TADAH_ENABLE_CERES OFF
# stopped compiling tadah.hpo ("adjoint_solver.h:4:10: fatal error:
# Eigen/Dense: No such file or directory").
include(${CMAKE_CURRENT_SOURCE_DIR}/configure_without_ceres/no_ceres_preset.cmake)
set(_hpo_no_ceres_dir ${CMAKE_CURRENT_BINARY_DIR}/configure_without_ceres)
hpo_write_no_ceres_preset(${_hpo_no_ceres_dir}/preset.cmake)
cmake_path(SET _hpo_source_dir NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME hpo_configure_without_ceres
    COMMAND ${CMAKE_COMMAND}
        -DHPO_DIR=${_hpo_source_dir}
        -DPRESET=${_hpo_no_ceres_dir}/preset.cmake
        "-DGENERATOR=${CMAKE_GENERATOR}"
        -DWORK_DIR=${_hpo_no_ceres_dir}/work
        -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_without_ceres/check_configure_without_ceres.cmake)
set_tests_properties(hpo_configure_without_ceres PROPERTIES
    LABELS cmake
    TIMEOUT 300)
# hpo_configure_without_ceres_preset: a configure from the preset that
# hpo_write_no_ceres_preset() writes sees what the configure that wrote it
# saw (configure_without_ceres/check_no_ceres_preset.cmake, on a throw-away
# pair of projects): a package found only through -DCMAKE_PREFIX_PATH, an
# option of the build, a dependency given as FETCHCONTENT_SOURCE_DIR_<NAME>
# and one populated under FETCHCONTENT_BASE_DIR that none names; a
# FETCHCONTENT_BASE_DIR of its own; TADAH_ENABLE_CERES OFF, BUILD_TESTING
# OFF and FETCHCONTENT_FULLY_DISCONNECTED ON whatever the build has. Before
# the fix: the preset held a fixed list (11 dependency names, compilers,
# build tool and type, LAMMPS paths, TADAH_GIT_BRANCH), so the configure
# from it did not find the package ("Could not find a package configuration
# file provided by \"TadahProbeDep\""), and hpo_configure_without_ceres
# failed in a valid build that finds a dependency through such a hint.
add_test(NAME hpo_configure_without_ceres_preset
    COMMAND ${CMAKE_COMMAND}
        -DMODULE=${CMAKE_CURRENT_SOURCE_DIR}/configure_without_ceres/no_ceres_preset.cmake
        "-DGENERATOR=${CMAKE_GENERATOR}"
        -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/configure_without_ceres/preset_check
        -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_without_ceres/check_no_ceres_preset.cmake)
set_tests_properties(hpo_configure_without_ceres_preset PROPERTIES
    LABELS cmake
    TIMEOUT 120)
add_test(NAME hpo_anneal_registry_sync
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../blueprint/gen_anneal_keys.py
        --check)
set_tests_properties(hpo_anneal_registry_sync PROPERTIES
    LABELS "anneal"
    TIMEOUT 30)

# Fingerprint drift gate: the fields of each example's
# reference_fingerprint.json that come from tools/tolerance_overrides.json
# (tolerances, accepted statuses, journal checks, the rationale) must equal
# that file, so an overrides edit cannot leave a fingerprint behind.
# Regenerate with tools/generate_fingerprint.py <example> --overrides ...
add_test(NAME hpo_example_fingerprint_overrides_sync
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/generate_fingerprint.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples
        --overrides ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/tolerance_overrides.json
        --check)
set_tests_properties(hpo_example_fingerprint_overrides_sync PROPERTIES
    LABELS "examples"
    TIMEOUT 30)
# The generator's and the gate's handling of the overrides, on a synthetic
# run (examples/tools/test_generate_fingerprint.py).
add_test(NAME hpo_example_fingerprint_tool
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/test_generate_fingerprint.py)
set_tests_properties(hpo_example_fingerprint_tool PROPERTIES
    LABELS "examples"
    TIMEOUT 30)
# The comparator's failure-count guards (NaN/Inf clampings, evaluations
# that threw) on a synthetic run (examples/tools/test_compare_fingerprint.py).
add_test(NAME hpo_example_fingerprint_compare
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/test_compare_fingerprint.py)
set_tests_properties(hpo_example_fingerprint_compare PROPERTIES
    LABELS "examples"
    TIMEOUT 30)
# The status the example scripts print (examples/common/journal_status.sh,
# sourced by common/run.sh, run_all.sh and the run.sh of 16, 17 and 27):
# the whole "Optimiser status" value, colons included
# (examples/tools/test_journal_status.py; it needs bash).
add_test(NAME hpo_example_journal_status
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/tools/test_journal_status.py)
set_tests_properties(hpo_example_journal_status PROPERTIES
    LABELS "examples"
    TIMEOUT 30)

# Boost drift gate: every Boost header HPO includes must be in the Boost
# Tadah vendors (MLIP's external/boost, on HPO's include path through
# tadah.mlip), so the build never depends on a system Boost. The include
# directories are joined with ':' (no ';' in add_test() arguments).
add_test(NAME hpo_vendored_boost_includes
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_vendored_boost.py
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        "$<JOIN:$<TARGET_PROPERTY:boost_iostream,INTERFACE_INCLUDE_DIRECTORIES>,:>"
        --cxx ${CMAKE_CXX_COMPILER})
set_tests_properties(hpo_vendored_boost_includes PROPERTIES
    TIMEOUT 60)
# The gate itself, on fake trees (tests/test_check_vendored_boost.py).
add_test(NAME hpo_vendored_boost_gate_tool
    COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_check_vendored_boost.py)
set_tests_properties(hpo_vendored_boost_gate_tool PROPERTIES
    TIMEOUT 60)

# The vendored ASA's upstream files keep their bytes, CRLF line endings
# included: the TADAH-PATCH hunks of asa.c are written in them
# (external/asa/TADAH-PATCHES.md; check_vendored_asa.py).
# hpo_vendored_asa_line_endings: every line of asa.c, asa.h, asa_usr_asa.h,
# ASA-LICENSE, ASA-NOTES and ASA-README.txt ends in CRLF. Before the fix:
# passes as well (a pin of the bytes; hpo_vendored_asa_gate_tool shows it
# fails on an LF or mixed file).
add_test(NAME hpo_vendored_asa_line_endings
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_vendored_asa.py
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_tests_properties(hpo_vendored_asa_line_endings PROPERTIES
    TIMEOUT 60)
# hpo_vendored_asa_gitattributes: with HPO's .gitattributes, git stores those
# files' bytes unchanged under core.autocrlf=input, core.autocrlf=true and a
# global attributes file '* text eol=lf' (git add --renormalize, and git add
# of an edit, in a scratch repository). Skipped when CMake finds no git; one
# --git=<path> token: MLIP's add_test() drops an empty argument.
# Before the fix: fails, every file under each setting ("attributes read:
# none"): nothing kept git from converting the CRLF endings to LF, which would
# have rewritten every line of asa.c.
find_package(Git QUIET)
add_test(NAME hpo_vendored_asa_gitattributes
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_vendored_asa.py
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        --git=${GIT_EXECUTABLE})
set_tests_properties(hpo_vendored_asa_gitattributes PROPERTIES
    TIMEOUT 120
    SKIP_RETURN_CODE 77)
# The gate itself, on fake trees (tests/test_check_vendored_asa.py); its git
# cases are skipped without git (GIT), and the entry is then Skipped.
# Before the fix: passes as well (the gate's own unit test).
add_test(NAME hpo_vendored_asa_gate_tool
    COMMAND ${CMAKE_COMMAND} -E env GIT=${GIT_EXECUTABLE}
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_check_vendored_asa.py)
set_tests_properties(hpo_vendored_asa_gate_tool PROPERTIES
    TIMEOUT 120
    SKIP_REGULAR_EXPRESSION "\nOK \\(skipped=[0-9]+"
    FAIL_REGULAR_EXPRESSION "\nRan 0 tests ")

# The MPI build's suite (TADAH_HPO_MPI) is not the serial build's, and the
# serial checks never run it: hpo_mpi_build_tests_lint reads every test
# source here (*.cpp, *.h, *.hpp, subdirectories included) as the MPI build
# compiles it and fails a TEST_CASE that builds a PcWorkerPool or requests
# PC_ISOLATION PROCESS there (which that build does not have, and refuses)
# without SKIP_WITHOUT_PC_PROCESS_WORKERS() as its first statement
# (pc_process_workers.h), such a pool or request outside every TEST_CASE
# body, and, in every build's view, any use of the LAMMPS class other than
# as a pointer, a reference or a smart-pointer holder: an instance comes
# from LammpsRunner::create_instance, which runs it on the build's
# communicator (check_mpi_build_tests.py; its docstring lists what it does
# not see). In every build. The serial build's test_hpo backs the pool part
# at run time (pc_process_workers.h).
# Before the fix: fails, 15 cases -- the 12 pc pool cases that build a pool
# (7 skipped through a macro of their file, 5 not at all; one of them the
# width check, whose pools are temporaries), the 2 'Two runs in one process'
# cases under PC_ISOLATION PROCESS, and the startup-strategy case whose
# section asks for PC_ISOLATION PROCESS without a PC_LAMMPS line and said
# nothing of the MPI build's refusal of it (it passed there: the strategy
# is refused first, which it now checks in that build).
add_test(NAME hpo_mpi_build_tests_lint
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/check_mpi_build_tests.py
        ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(hpo_mpi_build_tests_lint PROPERTIES
    TIMEOUT 60)
# The check itself, on synthetic sources (tests/test_check_mpi_build_tests.py).
# Before the fix: passes as well (the check's own unit test).
add_test(NAME hpo_mpi_build_tests_lint_tool
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_check_mpi_build_tests.py)
set_tests_properties(hpo_mpi_build_tests_lint_tool PROPERTIES
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "\nRan 0 tests ")

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

    # A signal that one rank receives ends every rank as a signal stop
    # (mpi/test_signal_mpi.cpp): one entry per scenario, each run at 1, 2
    # and 4 ranks (mpi/run_signal_mpi.cmake.in); CORE's stop flag is never
    # lowered, so a scenario is a process of its own.
    add_executable(test_hpo_signal_mpi mpi/test_signal_mpi.cpp)
    target_link_libraries(test_hpo_signal_mpi PRIVATE
        tadah.core tadah.mlip tadah.hpo MPI::MPI_CXX)
    configure_file(
        mpi/run_signal_mpi.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/run_signal_mpi.cmake
        @ONLY)
    foreach(_scenario pt_seed pt_t0 pt_pass2 pt_jitter pt_init pt_rung
                      pa_seed pa_t0 pa_init pa_sweep)
        add_test(NAME hpo_mpi_signal_${_scenario}
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:test_hpo_signal_mpi>
                -DSCENARIO=${_scenario}
                -P ${CMAKE_CURRENT_BINARY_DIR}/run_signal_mpi.cmake)
        set_tests_properties(hpo_mpi_signal_${_scenario} PROPERTIES
            LABELS "mpi;shutdown"
            TIMEOUT 400)
    endforeach()

    # Multi-rank `tadah hpo` end to end (mpi/hpo_mpi_e2e.py): the RESTART
    # and gradient-search refusals, the LAMMPS fast kernels, the density
    # gates and the density report on 1 and 2 ranks. The cases run the
    # built `tadah` (Skipped, 77, in a build without it). Every argument is
    # one --key=value token: MLIP's add_test() drops an empty argument.
    # The launcher's flags as one shell-quoted string each (no ';').
    string(REPLACE ";" " " _mpi_preflags "${MPIEXEC_PREFLAGS}")
    string(REPLACE ";" " " _mpi_postflags "${MPIEXEC_POSTFLAGS}")
    foreach(_case restart_refusal gradient_refusal eamfast rhogates)
        add_test(NAME hpo_mpi_e2e_${_case}
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/mpi/hpo_mpi_e2e.py
                --case=${_case}
                --mpiexec=${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
                "--preflags=${_mpi_preflags}"
                "--postflags=${_mpi_postflags}"
                "--tadah=$<$<TARGET_EXISTS:tadah>:$<TARGET_FILE:tadah>>"
                --examples=${CMAKE_CURRENT_SOURCE_DIR}/../examples
                --workdir=${CMAKE_CURRENT_BINARY_DIR}/hpo_mpi_e2e)
        set_tests_properties(hpo_mpi_e2e_${_case} PROPERTIES
            LABELS "mpi;e2e"
            TIMEOUT 600
            SKIP_RETURN_CODE 77)
    endforeach()
endif()

# Integration matrix — opt-in via `ctest -L slow`. Requires a bundle
# directory; the env var TADAH_HPO_INTEGRATION_BUNDLE selects which one.
# Without it the wrapper exits with 77, which CTest reports as Skipped.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh")
    add_test(NAME hpo_strategy_matrix
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/integration/run_matrix_ctest.sh)
    set_tests_properties(hpo_strategy_matrix PROPERTIES
        LABELS "slow"
        SKIP_RETURN_CODE 77)
endif()

# tadah.hpo_headers (ctest -L headers): every header of include/
# compiles on its own and twice in a row (cmake-tools'
# tadah_add_header_check). Before the fix: fails on hpo_host_impl.hpp,
# which does not include hpo_host.h ('expected initializer before <').
tadah_add_header_check(tadah.hpo)

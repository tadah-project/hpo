/***********************************************************************
 * tadah_asa_pristine_config.h -- configuration for the PRISTINE
 * (all-algorithmic-gates-off) reference build of the vendored ASA-30.53
 * library, target `tadah_asa_pristine` (asa_pristine.c).
 *
 * Purpose: the golden-neutrality unit test runs the same problem
 * through this canonical build and through the featured build
 * (tadah_asa_config.h) at default option values and asserts bitwise
 * identical trajectories.  Only embedding gates that cannot influence
 * the annealing trajectory are shared with the featured build:
 * output-file redirection (USER_ASA_OUT), the passive user-data channel
 * (OPTIONAL_DATA_PTR), and library etiquette (no stdout, no popen).
 *
 * Every external symbol defined by asa.c is renamed with the
 * tadah_asaref_ prefix so both builds can be linked into one test
 * binary.  The USER_DEFINES layouts of the two builds differ; the type
 * never crosses a TU boundary in the tests.
 ***********************************************************************/
#ifndef TADAH_ASA_PRISTINE_CONFIG_H
#define TADAH_ASA_PRISTINE_CONFIG_H

/* --- embedding gates (trajectory-neutral), as in tadah_asa_config.h - */
#define OPTIONS_FILE 0
#define OPTIONS_FILE_DATA 0
#define HAVE_ANSI 1
#define ASA_PRINT 1
#define ASA_PRINT_MORE 0
#define USER_ASA_OUT 1
#define INCL_STDOUT 0
#define SYSTEM_CALL 0
#define TIME_CALC 0
#define MY_TEMPLATE 0
#define OPTIONAL_DATA_PTR 1
#define OPTIONAL_PTR_TYPE void

/* --- ALL algorithmic gates off: canonical ASA ------------------------ */
#define QUENCH_COST 0
#define QUENCH_PARAMETERS 0
#define RATIO_TEMPERATURE_SCALES 0
#define DELTA_PARAMETERS 0
#define USER_INITIAL_PARAMETERS_TEMPS 0
#define MULTI_MIN 0
#define ASA_QUEUE 0
#define ASA_PARALLEL 0
#define ASA_RESOLUTION 0
#define USER_INITIAL_COST_TEMP 0
#define USER_ACCEPTANCE_TEST 0
#define USER_ACCEPT_ASYMP_EXP 0
#define USER_ACCEPT_THRESHOLD 0
#define USER_COST_SCHEDULE 0
#define USER_GENERATING_FUNCTION 0
#define USER_REANNEAL_COST 0
#define USER_REANNEAL_PARAMETERS 0
#define ASA_FUZZY 0
#define FITLOC 0
#define ASA_SAVE 0
#define ASA_SAVE_OPT 0
#define ASA_SAVE_BACKUP 0
#define SELF_OPTIMIZE 0
#define ASA_SAMPLE 0
#define ASA_PIPE 0
#define ASA_PIPE_FILE 0
#define ADAPTIVE_OPTIONS 0
#define ASA_EXIT_ANYTIME 0

/* --- rename every external symbol asa.c can define ------------------ */
#define exit_msg tadah_asaref_exit_msg
#define asa tadah_asaref_asa
#define asa_exit tadah_asaref_asa_exit
#define generate_new_state tadah_asaref_generate_new_state
#define generate_new_state_par tadah_asaref_generate_new_state_par
#define generate_asa_state tadah_asaref_generate_asa_state
#define accept_new_state tadah_asaref_accept_new_state
#define reanneal tadah_asaref_reanneal
#define cost_derivatives tadah_asaref_cost_derivatives
#define asa_test_asa_options tadah_asaref_asa_test_asa_options
#define cost_function_test tadah_asaref_cost_function_test
#define print_string tadah_asaref_print_string
#define print_string_index tadah_asaref_print_string_index
#define print_state tadah_asaref_print_state
#define print_asa_options tadah_asaref_print_asa_options
#define print_time tadah_asaref_print_time
#define aux_print_time tadah_asaref_aux_print_time
#define Exit_ASA tadah_asaref_Exit_ASA
#define gener_block_state_qsort tadah_asaref_gener_block_state_qsort
#define multi_cost_qsort tadah_asaref_multi_cost_qsort

#endif /* TADAH_ASA_PRISTINE_CONFIG_H */

/***********************************************************************
 * tadah_asa_config.h -- TADAH build configuration for the vendored
 * ASA-30.53 library (Lester Ingber, BSD-3-Clause; see ASA-LICENSE).
 *
 * This header is the single source of truth for every ASA compile-time
 * gate used by the featured build (target `tadah_asa`).  It is included
 * by asa_build.c BEFORE asa.c, so these definitions win over the
 * #ifndef-guarded defaults in asa_usr_asa.h / asa.h.  Values are the
 * literal 1/0 because TRUE/FALSE are not defined until asa_usr_asa.h.
 *
 * RULES (see TADAH-PATCHES.md and the HPO ASA documentation):
 *  - An algorithmic gate may be enabled here ONLY if its default option
 *    values bitwise-reproduce the canonical (all-gates-off) trajectory.
 *    This is enforced by the golden-neutrality unit test, which compares
 *    this build against `tadah_asa_pristine` (asa_pristine.c).
 *  - ASA_PRINT must stay 1: asa_exit() calls fclose(ptr_asa_out)
 *    unconditionally, and ptr_asa_out is only opened under ASA_PRINT.
 *  - Never route ASA output to the literal filename "STDOUT": the same
 *    unguarded fclose would close the process's stdout.
 ***********************************************************************/
#ifndef TADAH_ASA_CONFIG_H
#define TADAH_ASA_CONFIG_H

/* --- embedding: no option file, no user module, library etiquette --- */
#define OPTIONS_FILE 0        /* options are supplied programmatically  */
#define OPTIONS_FILE_DATA 0
#define HAVE_ANSI 1
#define ASA_PRINT 1           /* MUST stay 1 (unguarded fclose)         */
#define ASA_PRINT_MORE 0
#define USER_ASA_OUT 1        /* per-run OPTIONS->Asa_Out_File          */
#define INCL_STDOUT 0         /* no printf from library code            */
#define SYSTEM_CALL 0         /* no popen("/bin/cp ...") shell-outs     */
#define TIME_CALC 0
#define MY_TEMPLATE 0

/* --- user-data channel for the C++ trampoline ----------------------- */
#define OPTIONAL_DATA_PTR 1
#define OPTIONAL_PTR_TYPE void /* OPTIONS->Asa_Data_Ptr is a void*      */

/* --- algorithmic gates, neutral at their documented defaults -------- */
#define QUENCH_COST 1              /* User_Quench_Cost_Scale  = 1.0     */
#define QUENCH_PARAMETERS 1        /* User_Quench_Param_Scale = 1.0     */
#define RATIO_TEMPERATURE_SCALES 1 /* User_Temperature_Ratio  = 1.0     */
#define DELTA_PARAMETERS 1         /* User_Delta_Parameter mirrors Delta_X */
#define USER_INITIAL_PARAMETERS_TEMPS 1 /* mirrors Initial_Parameter_Temperature */
#define MULTI_MIN 1                /* Multi_Number=1, Multi_Specify=1, Multi_Grid=0 */
#define ASA_QUEUE 1                /* Queue_Size=0 disables the cache   */
#define ASA_PARALLEL 1             /* Gener_Block=1 == canonical serial */

/* --- explicitly excluded gates (kept 0; see TADAH-PATCHES.md) -------
 * ASA_RESOLUTION            grid snap has no neutral default
 * USER_INITIAL_COST_TEMP    compiles out the Number_Cost_Samples loop
 * USER_ACCEPTANCE_TEST      does not compile upstream (Acceptance_Fit)
 * USER_ACCEPT_ASYMP_EXP / USER_ACCEPT_THRESHOLD  change acceptance law
 * USER_COST_SCHEDULE / USER_GENERATING_FUNCTION /
 * USER_REANNEAL_COST / USER_REANNEAL_PARAMETERS  function-pointer opts
 * ASA_FUZZY / FITLOC        implementation lives in un-vendored asa_usr.c
 * ASA_SAVE*                 popen, #include "asa_opt", persisted statics
 * SELF_OPTIMIZE / ASA_SAMPLE / ASA_PIPE* / ADAPTIVE_OPTIONS /
 * ASA_EXIT_ANYTIME          not applicable to library embedding
 * ------------------------------------------------------------------- */
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

#endif /* TADAH_ASA_CONFIG_H */

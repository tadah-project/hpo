/***********************************************************************
 * tadah_asa_hook.h -- TADAH parallel-evaluation hook for the vendored
 * ASA-30.53 library (see TADAH-PATCHES.md).
 *
 * ASA's ASA_PARALLEL main loop generates a block of Gener_Block trial
 * states serially and marks the region evaluating their costs as the
 * user's parallel-spawn insertion point ("ENTER CODE TO SPAWN OFF
 * PARALLEL GENERATED STATES", asa.c).  The TADAH patch inserts a call
 * through this function pointer there: when non-NULL it must evaluate
 * the cost of every NON-cached slot of the block; when NULL the
 * original upstream serial loop runs unchanged.
 *
 * Hook contract (mirrors the upstream loop):
 *  - evaluate only slots with queue_new_par[i] == 1; for cached slots
 *    (queue_new_par[i] == 0) copy queue_par_cost[i] into block[i].cost
 *    and leave valid_state_generated_flag_par[i] untouched;
 *  - write the returned cost into block[i].cost and the validity flag
 *    into valid_state_generated_flag_par[i] (TRUE/FALSE);
 *  - never write options->parallel_id (upstream's racy write is
 *    deliberately absent from the hook path; slot identity travels
 *    through the TADAH-side context instead);
 *  - the hook runs on asa()'s own (serial) thread, so it may safely
 *    set options->Immediate_Exit after its internal barrier;
 *  - no C++ exception may propagate out of the hook.
 *
 * This header assumes the TADAH gate set (tadah_asa_config.h), which it
 * includes to pin the USER_DEFINES layout.
 ***********************************************************************/
#ifndef TADAH_ASA_HOOK_H
#define TADAH_ASA_HOOK_H

#include "tadah_asa_config.h"
#include "asa_usr_asa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors ASA's internal STATE struct at the TADAH gate set (asa.h:
 * USER_ACCEPTANCE_TEST off => {cost, parameter}).  Layout is checked
 * with a _Static_assert at the patch site inside asa.c. */
typedef struct {
  double cost;
  double *parameter;
} tadah_asa_state_view;

typedef void (*tadah_asa_parallel_hook_fn) (
    tadah_asa_state_view * block, LONG_INT gener_block,
    double *parameter_minimum, double *parameter_maximum,
    double *tangents, double *curvature,
    ALLOC_INT * number_parameters, int *parameter_type,
    int *valid_state_generated_flag_par, int *exit_status,
    USER_DEFINES * options, int *queue_new_par, double *queue_par_cost);

/* Installed (under the TADAH single-instance mutex) by the HPO ASA
 * driver before calling asa(); reset to NULL after the call. */
extern tadah_asa_parallel_hook_fn tadah_asa_parallel_hook;

/* Live mirror of asa()'s *current_cost_temperature (the acceptance
 * temperature), refreshed by the patched spawn region before every
 * block-evaluation hook call. The TADAH driver resets it to NaN before
 * each run and stamps it on block evaluations as the journal's
 * "Tcost=" tag; it stays NaN until the first block (serial contexts
 * never refresh it). Written on asa()'s thread; read by the hook on
 * the same thread (race-free under the single-instance mutex). */
extern double tadah_asa_cost_temp;

/* Index of the dimension whose temperature underflowed when asa()
 * exits with P_TEMP_TOO_SMALL (-1 otherwise). Upstream records it only
 * in an internal variable (*index_exit_v) and its own journal; the
 * TADAH patch mirrors it here so the driver can name the dimension in
 * the stop-reason line. Written on asa()'s thread at the exit site;
 * the driver resets it to -1 before each call (single-instance mutex
 * makes this race-free). */
extern LONG_INT tadah_asa_exit_index;

#ifdef __cplusplus
}
#endif

#endif /* TADAH_ASA_HOOK_H */

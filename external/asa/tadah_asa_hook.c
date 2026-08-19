/* Definitions of the TADAH hook pointer (NULL keeps the original
 * upstream serial loop) and the P_TEMP_TOO_SMALL dimension mirror.
 * See tadah_asa_hook.h. */
#include "tadah_asa_hook.h"

tadah_asa_parallel_hook_fn tadah_asa_parallel_hook = NULL;
LONG_INT tadah_asa_exit_index = -1;
double tadah_asa_cost_temp = 0.0;  /* driver resets to NaN per run */

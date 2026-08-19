/* Definition of the TADAH parallel-evaluation hook pointer; NULL keeps
 * the original upstream serial loop (see tadah_asa_hook.h). */
#include "tadah_asa_hook.h"

tadah_asa_parallel_hook_fn tadah_asa_parallel_hook = NULL;

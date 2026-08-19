/* Featured build of the vendored ASA-30.53 library.  The compile-time
 * gate set lives in tadah_asa_config.h so it never leaks into CMake or
 * consumer translation units.  asa.c is compiled ONLY through wrapper
 * TUs like this one (see also asa_pristine.c). */
#include "tadah_asa_config.h"
#include "asa.c"

// The probe of check_handlers_removal.cmake (CTest
// hpo_hpotarget_handlers_removal): it takes the address of each table of
// handlers of hpotarget_handlers_gen.h, so the program links only when a
// handler is defined for each key, line option and form the header
// declares.
#include <tadah/hpo/hpotarget_handlers_gen.h>

int main()
{
    namespace R = tadah::hpo::hpotarget_registry;
    const void* volatile keys = &R::kKeyHandlers;
    const void* volatile options = &R::kOptionHandlers;
    const void* volatile forms = &R::kFormHandlers;
    return keys != nullptr && options != nullptr && forms != nullptr ? 0 : 1;
}

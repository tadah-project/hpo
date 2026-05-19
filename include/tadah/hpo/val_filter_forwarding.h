#ifndef TADAH_HPO_VAL_FILTER_FORWARDING_H
#define TADAH_HPO_VAL_FILTER_FORWARDING_H

// Helper for the HPO host: forward EFILTER / FFILTER from a parsed
// --validation context onto the stripped transform context that will be
// applied to val_stdb_.
//
// Filters declared in the validation file are intentionally honoured;
// filters set in the training (master) context are NOT forwarded -- they
// would change what the validation metric is computed on without the
// user opting in, breaking apples-to-apples comparison across HPO trials.
//
// The helper lives in its own header so it can be unit-tested without
// pulling in the full HPO host (which transitively requires moodycamel,
// dlib, mlip workspace managers, etc.).

#include <tadah/core/context.h>

namespace tadah::hpo::detail {

inline void forward_val_filters(core::Context& vt, const core::Context& val_ctx)
{
    if (val_ctx.contains("EFILTER") && val_ctx.size("EFILTER") == 2) {
        vt.add("EFILTER", val_ctx.get<std::string>("EFILTER", 0));
        vt.add("EFILTER", val_ctx.get<std::string>("EFILTER", 1));
    }
    if (val_ctx.contains("FFILTER") && val_ctx.size("FFILTER") > 0) {
        vt.add("FFILTER", val_ctx.get<std::string>("FFILTER"));
    }
}

}  // namespace tadah::hpo::detail

#endif  // TADAH_HPO_VAL_FILTER_FORWARDING_H

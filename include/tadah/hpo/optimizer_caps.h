#pragma once
/*
 * OptimizerCaps -- capability descriptor for an optimiser backend.
 *
 * Single source of truth for "what does (LIB, ALGO) support". Every
 * caller that previously hard-coded `spec.lib == "DLIB" && spec.algo
 * == "BFGS"`-style branches now queries this descriptor instead, so
 * adding a new optimiser is two changes: implement a templated
 * `run_<lib>_<algo>` runner, then register a CapsEntry below. No
 * caller code changes.
 *
 * BasinsPolicy uses `ladder_ftol` to decide whether the per-rung
 * tolerance is written to ftol_rel (NLopt -- relative test) or ftol_abs
 * (DLIB:BFGS -- absolute objective-delta test). Without this, the
 * BasinsPolicy ladder unconditionally set ftol_rel and DLIB:BFGS's
 * REFINE step threw "DLIB:BFGS does not support FTOL_REL".
 */

#include <cstddef>
#include <string>
#include <vector>

namespace tadah::hpo {

struct OptimizerSpec;

struct OptimizerCaps {
    std::string lib;
    std::string algo;

    /* Algorithmic topology -------------------------------------------- */
    bool is_global          = false;  // GLOBAL search (vs. LOCAL refinement)
    bool is_population      = false;  // accepts POPULATION
    bool is_gradient_based  = false;  // uses derivatives (NLopt LD_*, DLIB:BFGS)
    bool accepts_inner      = false;  // INNER/ENDINNER sub-optimiser block
    bool accepts_multi_seed = false;  // a single run can consume >1 starting point

    /* Stop-criteria support: which fields of StopCriteria the backend
     * actually honours. Used by BasinsPolicy and validators to refuse
     * silent drops and to translate generic "objective tolerance"
     * into the right field. */
    bool supports_maxeval   = true;
    bool supports_maxtime   = false;
    bool supports_stopval   = false;
    bool supports_ftol_rel  = false;
    bool supports_ftol_abs  = false;
    bool supports_xtol_rel  = false;
    bool supports_xtol_abs  = false;

    /* Which f-tolerance field BASINS' FTOL_LADDER should write to.
     * Rel = relative objective change (NLopt's stop_ftol_rel, DLIB:GFS
     *       solver_eps which is also a relative-noise-style criterion).
     * Abs = absolute objective change (DLIB:BFGS objective_delta).
     * None = no useful ftol-style stop (e.g. TADAH:GRID, TADAH:ANNEAL). */
    enum class FtolKind { None, Rel, Abs };
    FtolKind ladder_ftol = FtolKind::None;
};

/* Lookup capabilities for (lib, algo). Throws std::runtime_error if
 * the (lib, algo) pair is not registered. */
const OptimizerCaps& optimizer_caps(const std::string& lib,
                                    const std::string& algo);

/* Convenience: look up by spec.lib / spec.algo. */
const OptimizerCaps& optimizer_caps(const OptimizerSpec& sp);

/* Apply a BASINS-style ladder rung to the spec's stop criteria.
 * Picks ftol_rel or ftol_abs depending on the backend's ladder_ftol;
 * a no-op for backends that have no ftol-style stop. */
void apply_ladder_ftol(OptimizerSpec& sp, double tol);

/* Return all registered backends matching a predicate. Used to compose
 * "compatible alternatives" diagnostics from the live registry instead
 * of hard-coding LIB:ALGO names in error strings -- the registry is the
 * single source of truth for who supports what. */
using CapsPredicate = bool (*)(const OptimizerCaps&);
std::vector<const OptimizerCaps*> filter_optimizer_caps(CapsPredicate pred);

/* Format LIB:ALGO entries from filter_optimizer_caps as a comma-
 * separated list (truncated, with "..." suffix when the list is long).
 * NLopt enumerates ~40 algorithms, so an unfiltered list is unhelpful
 * to a human reader; max_show keeps the message scannable. */
std::string format_caps_list(const std::vector<const OptimizerCaps*>& caps,
                             std::size_t max_show = 6);

}  // namespace tadah::hpo

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
 */

#include <cstddef>
#include <string>
#include <vector>

namespace tadah::hpo {

struct OptimizerSpec;

/* StepUse -- how a backend consumes the OPTIMIZER-block `STEP` value.
 * Single source of truth for the STEP acceptance policy: the startup
 * validator (scale_validation.h) and run_inner reject STEP on backends
 * that would silently drop it, and the journal prints the consumption
 * mode next to the effective per-dimension step. NLopt's own semantics
 * (verified against nlopt-src/src/api/optimize.c, 2.10):
 *   PerDimVector     -- raw per-dim initial steps (LN_NELDERMEAD, LN_SBPLX)
 *   PerDimRescale    -- the step vector becomes a per-dim coordinate
 *                       rescaling s_i = dx_i/dx_0 (LN_BOBYQA, LN_COBYLA;
 *                       util/rescale.c); rhobeg = |dx_0|, rhoend =
 *                       XTOL_REL*rhobeg
 *   MinCollapsed     -- the vector collapses to min_i |dx_i|
 *                       (LN_NEWUOA, LN_NEWUOA_BOUND, LN_PRAXIS)
 *   TrustHalfWidth   -- per-dim initial trust-region half-width
 *                       (LD_MMA, LD_CCSAQ)
 *   ForwardedToInner -- copied onto the local sub-optimiser; an OUTER
 *                       STEP overrides an INNER STEP (MLSL, AUGLAG)
 *   Ignored          -- the algorithm never reads an initial step
 *                       (CRS, ISRES, ESCH, DIRECT*, STOGO, LD_LBFGS,
 *                       LD_VAR*, LD_TNEWTON*, LD_SLSQP): STEP is rejected
 *   GridCount        -- TADAH:GRID historic overload (STEP as an integer
 *                       point count); replaced by PARAM GRID_POINTS, STEP
 *                       is rejected
 *   WidthFraction    -- the family has its own width-relative step knob
 *                       (ANNEAL*: PARAM STEP_FRAC); STEP is rejected
 *   Rejected         -- no initial-step concept at all (RANDOM, GFS,
 *                       BFGS, CERES, ASA): STEP is rejected */
enum class StepUse {
    Rejected, Ignored, PerDimVector, PerDimRescale, MinCollapsed,
    TrustHalfWidth, ForwardedToInner, GridCount, WidthFraction
};

/* Human-readable consumption mode for journal/error text. */
const char* step_use_name(StepUse u);

/* True when the backend honours an OPTIMIZER-block STEP (NLopt consumers). */
bool step_use_consumes(StepUse u);

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
     * actually honours. Used by validators to refuse silent drops. */
    bool supports_maxeval   = true;
    bool supports_maxtime   = false;
    bool supports_stopval   = false;
    bool supports_ftol_rel  = false;
    bool supports_ftol_abs  = false;
    bool supports_xtol_rel  = false;
    bool supports_xtol_abs  = false;

    /* STEP policy (see StepUse). step_note is the actionable suffix of the
     * rejection message ("set PARAM STEP_FRAC ...") or the one-line
     * consumption description printed in hpo_run.log for consumers. */
    StepUse     step_use  = StepUse::Rejected;
    const char* step_note = "";

    /* The backend evaluates outside the OPTIM box: LN_NEWUOA is called with
     * lb=ub=NULL (nlopt optimize.c), CERES' GradientProblemSolver has no box
     * constraints. Surfaced as a startup WARNING with bounds_note. */
    bool        ignores_bounds = false;
    const char* bounds_note    = "";
};

/* Lookup capabilities for (lib, algo). Throws std::runtime_error if
 * the (lib, algo) pair is not registered. */
const OptimizerCaps& optimizer_caps(const std::string& lib,
                                    const std::string& algo);

/* Convenience: look up by spec.lib / spec.algo. */
const OptimizerCaps& optimizer_caps(const OptimizerSpec& sp);

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

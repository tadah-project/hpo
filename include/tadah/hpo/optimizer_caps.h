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
#include <string_view>
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
 *                       (LN_PRAXIS; LN_NEWUOA and LN_NEWUOA_BOUND too,
 *                       which are refused: backend_refusal())
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
    bool is_gradient_based  = false;  // uses derivatives (NLopt LD_*, DLIB:BFGS)
    bool accepts_inner      = false;  // INNER/ENDINNER sub-optimiser block
    bool accepts_multi_seed = false;  // a single run can consume >1 starting point

    /* The attributes a run reads besides its stops. reject_unread_attributes()
     * (backend_rules.cpp: TADAH:RANDOM, TADAH:GRID, DLIB:GFS, NLopt) refuses
     * one the backend never reads; the annealers, ASA, DLIB:BFGS and CERES
     * refuse them in their own validators, and their flags say the same.
     *   reads_population     POPULATION: TADAH:ANNEAL_PA's walkers, and
     *                        NLopt's stochastic population (POP() in nlopt
     *                        2.10's optimize.c): GN_CRS2_LM, GN_ISRES,
     *                        GN_ESCH, the six MLSL forms, GD_STOGO_RAND
     *   reads_vector_storage VECTOR_ARR: the Luksan algorithms of NLopt
     *                        (optimize.c): LD_LBFGS, LD_VAR1/2, LD_TNEWTON*
     *   reads_seed           SEED: TADAH:RANDOM, the ANNEAL family, ASA and
     *                        NLopt (every NLopt algorithm: which of them draw
     *                        random numbers is not recorded)
     *   reads_ftol_abs_as_setting  FTOL_ABS as a setting, not a stop (no
     *                        supports_ftol_abs): DLIB:GFS's solver epsilon
     *   reads_threads        the THREADS of its own block, beyond the
     *                        STRATEGY-level THREADS that sizes the run's
     *                        workers: DLIB:GFS (its evaluation pool), ASA:ASA
     *                        (a block of generated states), TADAH:ANNEAL_PT
     *                        (its replicas and the REPLICAS default) and
     *                        TADAH:ANNEAL_PA (its walkers). validate_phase_
     *                        backends() refuses THREADS in an EXPLORE block of
     *                        another optimiser, and in an INNER block
     *   warns_unread_seed    SEED is not read, but configurations that set it
     *                        predate the check, so run_dlib_global() warns
     *                        instead of refusing it (DLIB:GFS) */
    bool reads_population          = false;
    bool reads_vector_storage      = false;
    bool reads_seed                = false;
    bool reads_ftol_abs_as_setting = false;
    bool reads_threads             = false;
    bool warns_unread_seed         = false;

    /* Stop criteria: which fields of StopCriteria end the backend's run
     * when they are met. require_stop() (backend_rules.h) counts these as
     * the stops of a backend that needs one, a tolerance only above 0 (0
     * or below switches it off), and reject_unread_attributes()
     * (backend_rules.cpp) refuses a tolerance none of them marks, as the
     * annealers', ASA's, DLIB:BFGS's and CERES' own validators do. A key the
     * backend reads for another purpose is not a stop: DLIB:GFS reads
     * FTOL_ABS as dlib's solver epsilon (the precision of its local model
     * fits), so its supports_ftol_abs is false (reads_ftol_abs_as_setting).
     * NLopt: the global algorithms and LD_MMA/LD_CCSAQ/LD_SLSQP list the
     * tolerances their stop tests read (nlopt 2.10 sources, checked by
     * probe runs); the other NLopt algorithms keep NLopt's generic
     * set. */
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

    /* The backend has no convergence test and no finite schedule: without
     * a stop criterion it runs until it is interrupted (each measured by
     * probe runs). require_stop() (backend_rules.h) refuses such
     * a spec at startup unless it sets MAXEVAL or another criterion the
     * supports_* flags mark; stop_note says why, for the message. Every
     * other backend ends on its own when no criterion is set: a
     * convergence test, a finite schedule or limits of its own. */
    bool        needs_stop = false;
    const char* stop_note  = "";

    /* The backend ignores NLopt's forced stop (NLopt's StoGO, GD_STOGO and
     * GD_STOGO_RAND: it tests only NLopt's evaluation count and clock), by
     * which HPO ends a search -- on a signal, an error, the RESTART
     * stagnation stop, a spent budget or a withheld loss at STOPVAL: the
     * objective then refuses every call, and only NLopt's own MAXEVAL ends
     * the run promptly (MAXTIME only once the rest of it has passed).
     * require_stop() counts only MAXEVAL as its stop. */
    bool        ignores_forced_stop = false;

    /* The backend's first call is at the point HPO hands it (x0: the
     * phase's first INIT point, a MULTISTART start, a PRESTAGE candidate).
     * False for the NLopt algorithms that take no starting point and
     * return x as their result only (nlopt 2.10 sources): StoGO, GD_STOGO
     * and GD_STOGO_RAND (stogo.cc never reads x0; its first trial is the
     * centre of its box, global.cc FillRegular, or with GD_STOGO_RAND's
     * POPULATION 2n + 1 or more a random point), DIRECT (cdirect.c: the
     * centre of the box; DIRect.c for GN_ORIG_DIRECT*) and AGS (ags.cc).
     * Every other NLopt algorithm evaluates x0 first -- the local ones,
     * MLSL (mlsl.c), CRS2, ISRES and ESCH (x0 is the first member of their
     * population). Classified for NLopt and DLIB:BFGS, the backends that
     * answer a call from the run's record (repeat_points.h): the startup
     * gradient-budget warning (gradient_budget_warnings, backend_rules.h)
     * takes the start of a search on one of them for a point the run has
     * evaluated only when its first call is there. The other backends
     * evaluate every point they ask for and keep the default. A search on
     * one of these never uses the point it is given, so a PRESTAGE
     * refinement on one is refused at startup (require_prestage_
     * refinement(), strategy_rules.h); each is global, so it is refused as
     * an INNER search too (require_inner_backend(), whose refusal names
     * start_note). */
    bool        starts_at_x0 = true;

    /* Where the first call of a backend whose starts_at_x0 is false is
     * instead, what it then evaluates, and that it never reads the point it
     * is given: the clause of those refusals ("DIRECT evaluates the centre
     * of the OPTIM box first and then divides the whole box; NLopt never
     * reads that point"); "" for every other backend. */
    const char* start_note = "";

    /* Whether a run of a backend whose starts_at_x0 is false draws random
     * numbers, so that a run with other numbers asks for other points
     * (nlopt 2.10 sources):
     *   Never         GD_STOGO (stogo.cc: no random trial point), GN_DIRECT,
     *                 GN_DIRECT_L and the three _NOSCAL forms (optimize.c
     *                 runs GN_DIRECT_L_RAND_NOSCAL with GN_DIRECT_L_NOSCAL's
     *                 cdirect settings), GN_ORIG_DIRECT and GN_ORIG_DIRECT_L
     *                 (DIRect.c), GN_AGS (ags);
     *   BetweenSides  GN_DIRECT_L_RAND: cdirect.c draws only to choose one of
     *                 several equally long sides of a box, which a box of
     *                 one dimension never has;
     *   TrialPoints   GD_STOGO_RAND: POPULATION random trial points per box
     *                 (optimize.c POP(2n); stogo.cc det_pnts = 2n + 1 -
     *                 POPULATION). Below POPULATION 2n + 1, NLopt's default
     *                 2n included, the box centre is a trial point too and
     *                 the first taken (global.cc NewtonTest adds the random
     *                 points, then FillRegular the centre last; TBox::
     *                 RemoveTrial takes the last), and the local search from
     *                 it draws nothing: a run asks for the points of that
     *                 search before any random one. At 2n + 1 or more every
     *                 trial point is random;
     *   NotRecorded   every backend whose first call is at its point.
     * A search that draws none (before its first random trial point) asks
     * for the same points in every run: the RESTART warnings of
     * start_point_warnings() (backend_rules.h). */
    enum class RandomDraws { NotRecorded, Never, BetweenSides, TrialPoints };
    RandomDraws random_draws = RandomDraws::NotRecorded;

    /* How the tolerances end the run when one of them is costly, the last
     * clause of require_stop()'s message and a line of the ALGO help (""
     * for none): DIRECT tests XTOL on every rectangle it divides, the
     * largest included, so an XTOL ends it only once the whole box is
     * divided that finely. */
    const char* tolerance_note = "";

    /* Why a tolerance the run does not test has no effect, the reason
     * clause of reject_unread_attributes()'s refusal ("uniform random
     * search has no convergence test to apply it to"): a tolerance no
     * supports_* flag marks, and every tolerance of MLSL and AUGLAG with an
     * INNER block, which NLopt hands only to the default local optimiser it
     * creates without one (default_inner). "" for a backend that tests
     * every tolerance it accepts. */
    const char* untested_tolerance_note = "";

    /* NLopt's MLSL and AUGLAG run a local optimiser inside their run
     * (accepts_inner); with no INNER block NLopt creates this one, handing
     * it the block's tolerances (nullptr: NLopt has none, so the block needs
     * an INNER block -- AUGLAG, AUGLAG_EQ, G_MLSL, G_MLSL_LDS --, which
     * require_inner_backend(), strategy_rules.h, enforces at startup).
     * inner_run_spec() (phase_specs.h) derives the spec that local search
     * runs with. */
    const char* default_inner = nullptr;

    /* The backend proposes points outside the OPTIM box: CERES'
     * GradientProblemSolver has no box constraints (CERES:STEEPEST_DESCENT,
     * the CERES driver HPO registers, is the only such backend: NLopt's
     * LN_NEWUOA, called with lb = ub = NULL, and Ceres' Wolfe drivers are
     * refused). Such a point is scored, not evaluated
     * (box_penalty.h). Surfaced as a startup WARNING with bounds_note. */
    bool        ignores_bounds = false;
    const char* bounds_note    = "";
};

/* Backends HPO refuses although their library offers them: a defect of
 * the library version HPO builds makes them unsafe to run, or they do not
 * solve the problem HPO poses, one scalar loss inside the OPTIM box (the
 * refusal table in optimizer_caps.cpp, each entry with its repro:
 * NLOPT:LN_NEWUOA_BOUND, LN_NEWUOA, NLOPT_LD_LBFGS_NOCEDAL, Ceres' Wolfe
 * line-search drivers CERES:LBFGS, WOLFE and NLCG, which stall at a point
 * outside the box scored by box_penalty.h, and Ceres' least-squares solvers
 * CERES:LEVENBERG_MARQUARDT, DOGLEG and SUBSPACE_DOGLEG). A refused backend is not registered, so
 * filter_optimizer_caps() and every list of alternatives leave it out, and
 * find_optimizer_caps() returns nullptr.
 *
 * backend_refusal(): why (lib, algo) is refused -- the reason, ending with
 * the backends to use instead -- or nullptr when it is not refused.
 *
 * reject_refused_backend(): throws std::runtime_error
 * "LIB:ALGO is refused: <reason>" for a refused (lib, algo), prefixed with
 * "<where>: " when `where` is not empty (the phase: "the EXPLORE block");
 * a no-op otherwise. optimizer_caps() and the driver rules
 * (require_driver(), require_inner_driver(), require_inner_backend(),
 * strategy_rules.h) throw this text, so it is the same everywhere:
 * "the STRATEGY-level optimiser: NLOPT:LN_NEWUOA_BOUND is refused: NLopt
 * 2.10 plans each LN_NEWUOA_BOUND step with a nested LD_MMA run ...". */
const char* backend_refusal(std::string_view lib, std::string_view algo);
/* Every refused backend, "LIB:ALGO", sorted (the ALGO help's table). */
std::vector<std::string> refused_backend_names();
void reject_refused_backend(std::string_view lib, std::string_view algo,
                            const std::string& where = "");

/* The libraries of the registered backends, sorted: ASA, CERES, DLIB,
 * NLOPT, TADAH. */
std::vector<std::string> optimizer_libraries();

/* Throws std::runtime_error unless (lib, algo) is registered: the
 * refusal of a refused one (reject_refused_backend()'s text), and for one
 * neither registered nor refused "unknown optimiser LIB:ALGO: <why>;
 * `tadah explain ALGO` lists the optimisers", <why> being "LIB NLOPT has
 * no ALGO LD_FOO" for a library of optimizer_libraries() and "there is no
 * LIB FOO (LIB is ASA, CERES, DLIB, NLOPT or TADAH)" otherwise (LIB and
 * ALGO are read as written). Prefixed with "<where>: " when `where` is not
 * empty. The parser's checks of a block's optimiser (require_driver(),
 * require_inner_driver(), strategy_rules.h) throw it with the phase, at the
 * line that closes the block; optimizer_caps() throws it with none. */
void require_registered_backend(std::string_view lib, std::string_view algo,
                                const std::string& where = "");

/* Lookup capabilities for (lib, algo). Throws require_registered_backend()'s
 * error, with no phase, if the (lib, algo) pair is refused or not
 * registered. The result refers to the static
 * registry, never to the arguments, and stays valid for the whole run. The
 * names are views: with `const std::string&` parameters every call with a
 * literal bound a temporary std::string, and GCC 13's -Wdangling-reference
 * reported a reference bound to the result as possibly dangling. */
const OptimizerCaps& optimizer_caps(std::string_view lib,
                                    std::string_view algo);

/* Convenience: look up by spec.lib / spec.algo. */
const OptimizerCaps& optimizer_caps(const OptimizerSpec& sp);

/* The same lookup for a rule that does not own the "unknown optimiser"
 * error: nullptr when (lib, algo) is not registered (a refused pair is
 * not). require_inner_backend() (strategy_rules.h) runs before
 * validate_phase_backends(), which reports a refused or unregistered
 * LIB:ALGO with the phase (reject_refused_backends()). */
const OptimizerCaps* find_optimizer_caps(std::string_view lib,
                                         std::string_view algo);

/* Return all registered backends matching a predicate. Used to compose
 * "compatible alternatives" diagnostics from the live registry instead
 * of hard-coding LIB:ALGO names in error strings -- the registry is the
 * single source of truth for who supports what. */
using CapsPredicate = bool (*)(const OptimizerCaps&);
std::vector<const OptimizerCaps*> filter_optimizer_caps(CapsPredicate pred);

/* Format LIB:ALGO entries from filter_optimizer_caps as a comma-
 * separated list, sorted by LIB:ALGO whatever the order of `caps` (the
 * registry order is a hash order), truncated after max_show entries with
 * a ", ... (N more)" suffix. NLopt enumerates ~40 algorithms, so an
 * unfiltered list is unhelpful to a human reader; max_show keeps the
 * message scannable. */
std::string format_caps_list(const std::vector<const OptimizerCaps*>& caps,
                             std::size_t max_show = 6);

}  // namespace tadah::hpo

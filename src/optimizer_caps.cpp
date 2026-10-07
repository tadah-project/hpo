#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>

#include <nlopt.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tadah::hpo {

namespace {

struct CapsKey {
    std::string lib;
    std::string algo;
    bool operator==(const CapsKey& o) const noexcept {
        return lib == o.lib && algo == o.algo;
    }
};
struct CapsKeyHash {
    std::size_t operator()(const CapsKey& k) const noexcept {
        return std::hash<std::string>{}(k.lib) * 1315423911u
             ^ std::hash<std::string>{}(k.algo);
    }
};

using Registry = std::unordered_map<CapsKey, OptimizerCaps, CapsKeyHash>;

/* Backends HPO refuses although their library offers them
 * (backend_refusal(), optimizer_caps.h). Each entry is a defect measured in
 * the library version HPO builds: before an upgrade of that library lifts
 * an entry, the entry's repro is run again.
 *
 * NLOPT:LN_NEWUOA_BOUND -- NLopt v2.10.0 (HPO/CMakeLists.txt fetches
 * GIT_TAG v2.10.0; offline builds use deps/nlopt). NLopt's bound-
 * constrained NEWUOA, its own variant and not Powell's code, solves each
 * trust-region and BIGLAG subproblem with a nested LD_MMA run (newuoa.c:
 * up to 1000 evaluations, each MMA iteration a dual solve of up to 100000,
 * optimize.c) whose stop struct has no MAXTIME and no forced-stop flag;
 * newuob_ tests those only just before an objective call (newuoa.c L310).
 * Once the model gradient holds inf or NaN, MMA's acceptance tests
 * (mma.c: `dd.gval >= fcur`, `fcur > dd.gval`) are false for ever and each
 * subproblem runs to its caps, about 1e8 model evaluations and 10-60 s
 * without an objective call. A non-finite value reaches the model
 *   (1) from any HUGE_VAL objective value: NLopt's C++ wrapper (nlopt.hpp)
 *       and HPO's (nlopt_adapter.h) return HUGE_VAL after force_stop when
 *       the objective throws, NLopt's convention for a forced stop. The
 *       first 2n+1 evaluations of a run are NEWUOA's starting points: after
 *       each of the first 2n newuob_ goes straight to the next point and
 *       L310, so a stop there returns at once; after the (2n+1)-th it
 *       builds its model and plans its first step. From the (2n+1)-th
 *       evaluation on, every signal, RESTART stagnation stop and run-ending
 *       error therefore costs one such subproblem, on any loss (a failed
 *       evaluation scored FAILSCORE is finite and is no such value);
 *   (2) without an XTOL floor (rhoend = 0) on a flat loss -- a box where
 *       every evaluation fails, scored FAILSCORE, is one -- where the trust
 *       radius shrinks without limit and NEWUOA's ZMAT overflows;
 *   (3) near convergence, where MMA's steps are ~1e-8 of the trust radius
 *       (its rho is dimensionful, mma.c MMA_RHOMIN) and BMAT overflows,
 *       with XTOL_REL 1e-12 as well.
 * LN_BOBYQA and LN_NEWUOA, which use no MMA, return at once after the same
 * forced stop. HPO's HUGE_VAL stays: it is NLopt's own convention. The
 * analysis and the repro (cases A, B and F) are in
 * docs/issues/nlopt_newuoa_bound_stall.md.
 *
 * NLOPT:NLOPT_LD_LBFGS_NOCEDAL -- NLopt v2.10.0 names it (nlopt.h,
 * nlopt_algorithm_to_string: the name the registry below registers each
 * NLopt algorithm under) but does not implement it: nlopt_optimize()'s
 * algorithm switch has no case for it and returns NLOPT_INVALID_ARGS
 * (optimize.c), so every run of it ended "FAILED -- nlopt invalid
 * argument" once its setup was done. The repro is the unit test that
 * calls nlopt_optimize() with it (test_optimizer_caps_step.cpp). */
struct Refusal {
    const char* lib;
    const char* algo;
    const char* reason;
};
constexpr Refusal kRefused[] = {
    {"NLOPT", "LN_NEWUOA_BOUND",
     "NLopt 2.10 plans each LN_NEWUOA_BOUND step with a nested LD_MMA run "
     "that has no time limit and sees no stop request, and once a "
     "non-finite value reaches its model each step takes about 10-60 s "
     "without evaluating the loss: after every stop HPO makes through the "
     "objective (a signal, a RESTART stagnation stop, an error that ends "
     "the run) from a run's (2n+1)-th evaluation on, the last of NEWUOA's "
     "starting points for n OPTIM dimensions, as the objective then "
     "returns HUGE_VAL, NLopt's convention for a forced stop, and when the "
     "loss barely changes over the trust region; Ctrl-C, SIGTERM, MAXTIME "
     "and RESTART wait that long. Use NLOPT:LN_BOBYQA, Powell's "
     "bound-constrained successor of NEWUOA (its STEP rescales each "
     "dimension instead of setting one radius), or NLOPT:LN_NEWUOA where "
     "the search may leave the OPTIM box"},
    {"NLOPT", "NLOPT_LD_LBFGS_NOCEDAL",
     "NLopt 2.10 does not implement it (NLopt names it, but its optimiser "
     "rejects it as an invalid argument, so every run of it failed once "
     "its setup was done). Use NLOPT:LD_LBFGS, NLopt's L-BFGS"},
};

void add(Registry& r, OptimizerCaps c)
{
    CapsKey k{c.lib, c.algo};
    r.emplace(k, std::move(c));
}

Registry build_registry()
{
    Registry r;

    /* TADAH built-ins ------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "RANDOM";
        c.is_global       = true;
        c.reads_seed      = true;
        /* run_random draws until MAXEVAL is spent, MAXTIME has passed or a
         * loss reaches STOPVAL: nothing else ends it. */
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        c.needs_stop = true;
        c.stop_note  = "uniform random search has no convergence test and "
                       "runs until it is interrupted";
        c.untested_tolerance_note =
            "uniform random search has no convergence test to apply it to";
        c.step_use  = StepUse::Rejected;
        c.step_note = "uniform sampling in the OPTIM box has no initial step; remove STEP";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "GRID";
        c.is_global       = true;
        /* Evaluates the whole (finite) grid, or stops earlier: after
         * MAXEVAL points in the order it visits them, once MAXTIME has
         * passed, or at a loss at or below STOPVAL. */
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* It draws no random numbers: SEED is refused. */
        c.untested_tolerance_note =
            "the grid search has no convergence test to apply it to";
        c.step_use  = StepUse::GridCount;
        c.step_note = "STEP is not the grid resolution; use PARAM GRID_POINTS <n> "
                      "(integer >= 2, default 20 points per dimension)";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL";
        c.is_global       = true;
        c.reads_seed      = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* eps-NEPS plateau rule (both SA variants): NEPS consecutive
         * rungs whose best improves by less than the tolerance stop the
         * run. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL_PT";
        c.is_global       = true;
        c.reads_seed      = true;   /* its replica ladder is PARAM REPLICAS;
                                       POPULATION is refused */
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* Corana sec.-4 eps-NEPS plateau evaluated on the coldest slot
         * + the global best. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        /* The replica ladder keeps its temperatures: no end of schedule
         * (the phase 4b and 4c probes ran it until it was killed). */
        c.needs_stop = true;
        c.stop_note  = "parallel tempering keeps its temperature ladder, so "
                       "no schedule ends it, and it runs until it is "
                       "interrupted";
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL_PA";
        c.is_global       = true;
        c.reads_population = true;  /* the walkers */
        c.reads_seed       = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* Corana sec.-4 eps-NEPS plateau evaluated on the population
         * best + the global best. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }

    /* ASA (vendored Ingber ASA-30.53, external/asa) ------------------- */
    {
        OptimizerCaps c;
        c.lib = "ASA";  c.algo = "ASA";
        c.is_global       = true;
        c.reads_seed      = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* No ftol/xtol: ASA's own plateau stops are PARAM Cost_Precision
         * + Maximum_Cost_Repeat and Accepted_To_Generated_Ratio. */
        c.step_use  = StepUse::Rejected;
        c.step_note = "the generating distribution width is driven by the per-dim "
                      "temperatures (Initial_Parameter_Temperature, Temperature_Ratio_Scale)";
        add(r, c);
    }

    /* DLIB ----------------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "GFS";
        c.is_global          = true;
        c.accepts_multi_seed = true;
        /* find_min_global ends when its call budget is spent (MAXEVAL),
         * its run time (MAXTIME) is up or its stop condition (STOPVAL)
         * holds. FTOL_ABS is its solver epsilon, which ends nothing. */
        c.supports_maxeval   = true;
        c.supports_maxtime   = true;
        c.supports_stopval   = true;
        c.needs_stop = true;
        c.stop_note  = "dlib's global search has no convergence test (FTOL_ABS "
                       "sets its solver's precision, which stops nothing) and "
                       "runs until it is interrupted";
        c.reads_ftol_abs_as_setting = true;
        c.untested_tolerance_note =
            "dlib's global search has no convergence test to apply it to";
        /* find_min_global (MaxLIPO+TR) is deterministic and does not read
         * SEED; configurations that set it predate the check, so
         * run_dlib_global() warns instead of refusing it. */
        c.warns_unread_seed = true;
        c.step_use  = StepUse::Rejected;
        c.step_note = "MaxLIPO+TR sizes its own trust region from the samples; remove STEP";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "BFGS";
        c.is_gradient_based  = true;
        c.supports_maxeval   = true;
        c.supports_ftol_abs  = true;     // objective_delta_stop_strategy
        c.step_use  = StepUse::Rejected;
        c.step_note = "the line search starts from -grad; the finite-difference step is PARAM fd_step";
        add(r, c);
    }

    /* CERES -- GradientProblemSolver family (LBFGS / WOLFE / NLCG /
     * STEEPEST_DESCENT). All four are local, gradient-based, accept
     * function/parameter tolerance (relative) and a gradient-norm
     * tolerance via PARAM tol_gradinf. The Solver-family (LM, DOGLEG,
     * SUBSPACE_DOGLEG) is intentionally NOT registered: those algorithms
     * are degenerate over a scalar loss and require residual-mode loss
     * exposure plus autodiff to be useful -- deferred to a later phase.
     * The dispatch in run_policy_base.h short-circuits those names with
     * a tailored message before this caps lookup runs. */
    {
        const char* ceres_grad_algos[] = {
            "LBFGS", "WOLFE", "NLCG", "STEEPEST_DESCENT"
        };
        for (const char* a : ceres_grad_algos) {
            OptimizerCaps c;
            c.lib                = "CERES";
            c.algo               = a;
            c.is_gradient_based  = true;
            c.supports_maxeval   = true;
            c.supports_ftol_rel  = true;
            c.supports_xtol_rel  = true;
            c.step_use           = StepUse::Rejected;
            c.step_note          = "Ceres chooses the initial line-search step itself; remove "
                                   "STEP (PARAM max_ls_iter caps the line-search iterations)";
            c.ignores_bounds     = true;
            c.bounds_note        = "ignores the OPTIM box: Ceres' GradientProblemSolver has no "
                                   "bounds, so its iterates can leave the box (a LOG dimension "
                                   "stays positive, a linear one is unbounded) and are evaluated "
                                   "there, and such a point can become the best; the first is "
                                   "reported with a WARNING and DIAGNOSTICS counts them all "
                                   "(finite differences at such a point step towards the box). "
                                   "DLIB:BFGS and the NLopt LD_* algorithms keep to the box";
            add(r, c);
        }
    }

    /* NLopt -- enumerate the catalogue. Every NLopt algorithm takes the
     * same StopCriteria via nlopt::opt::set_*; algorithm-class flags are
     * inferred from the canonical name prefix.
     *
     * Which stops end a run (nlopt 2.10 sources; the phase 4c probes ran
     * each algorithm without any, on a synthetic objective and through
     * hpo_run, and with each criterion alone):
     *   - MAXEVAL, MAXTIME and STOPVAL end every algorithm.
     *   - No convergence test and no finite schedule, so a stop is needed:
     *     DIRECT (cdirect.c: an FTOL test at each improvement, an XTOL test
     *     on every rectangle it divides, the largest included), CRS2 and
     *     ISRES (an FTOL and an XTOL test at each improvement), ESCH and
     *     AGS (no tolerance test; AGS's own eps is 0), StoGO (none: its
     *     local searches run StoGO's own BFGS, and the NLopt-local path that
     *     would read the tolerances is compiled out, stogo/local.cc `#if
     *     0`), MLSL (none: the tolerances reach only the local searches of
     *     NLopt's default local optimiser, optimize.c, never an INNER
     *     block), and LD_MMA, LD_CCSAQ and LD_SLSQP, whose only convergence
     *     test is the tolerances.
     *   - The original DIRECT (GN_ORIG_DIRECT, _L) stops itself after
     *     about 100000 evaluations (DIRect.c's cap, checked once per
     *     iteration: the probes ended after 100011 to 100071), and the
     *     other local algorithms
     *     converge on their own (a simplex, trust region or line search
     *     that shrinks to roundoff): they keep NLopt's generic set.
     *   - AUGLAG (HPO sets no constraint) runs its local search once and
     *     ends when that ends: an INNER block, or NLopt's default local
     *     optimiser (LN_COBYLA for LN_AUGLAG, LD_MMA for LD_AUGLAG), which
     *     takes the block's tolerances; with an INNER block auglag.c tests
     *     them only to name the ending (ICM is 0 after the first search).
     *     MLSL has the same defaults; G_MLSL, G_MLSL_LDS, AUGLAG and
     *     AUGLAG_EQ have none and need an INNER block.
     * Which attributes a run reads (optimize.c): the stochastic population
     * (POP()) CRS2, ISRES, ESCH, the MLSL forms and GD_STOGO_RAND; the
     * vector storage the Luksan algorithms LD_LBFGS, LD_VAR1/2 and
     * LD_TNEWTON*; the random seed (nlopt::srand) is set for every one. */
    for (int i = 0; i < NLOPT_NUM_ALGORITHMS; ++i) {
        const char* name = nlopt_algorithm_to_string(static_cast<nlopt_algorithm>(i));
        if (!name || !name[0]) continue;
        /* A refused algorithm is not registered: no list offers it, and
         * the lookup throws its refusal (kRefused). */
        if (backend_refusal("NLOPT", name)) continue;
        OptimizerCaps c;
        c.lib = "NLOPT";  c.algo = name;
        c.supports_maxeval  = true;
        c.supports_maxtime  = true;
        c.supports_stopval  = true;
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.supports_xtol_rel = true;
        c.supports_xtol_abs = true;

        const std::string s = name;
        if (!s.empty() && s[0] == 'G')                    c.is_global = true;
        if (s.size() >= 2 && s[1] == 'D')                 c.is_gradient_based = true;
        if (s.find("MLSL")   != std::string::npos ||
            s.find("AUGLAG") != std::string::npos)        c.accepts_inner = true;

        const auto starts = [&](const char* p) { return s.rfind(p, 0) == 0; };
        const bool mlsl = s.find("MLSL") != std::string::npos;
        const bool auglag = s.find("AUGLAG") != std::string::npos;
        const bool no_tolerance_test =
            starts("GN_ESCH") || starts("GN_AGS") || starts("GD_STOGO") ||
            mlsl;
        if (starts("GN_DIRECT") || starts("GN_CRS2") || starts("GN_ISRES") ||
            no_tolerance_test) {
            c.needs_stop = true;
            c.stop_note = s.find("MLSL") != std::string::npos
                ? "MLSL's multistart loop has no convergence test (the "
                  "tolerances reach only its local searches) and runs until "
                  "it is interrupted"
                : "NLopt's global search has no convergence test of its own "
                  "and runs until it is interrupted";
            /* NLopt's StoGO (algs/stogo) tests NLopt's evaluation count and
             * clock (nlopt_stop_evalstime), never its forced stop: once HPO
             * stops a StoGO search, StoGO calls on and the objective refuses
             * each call (a few microseconds) until NLopt's MAXEVAL ends it.
             * Without MAXEVAL it never ended (a host exception or a signal,
             * with STOPVAL as its only stop), or, with MAXTIME, only once the
             * rest of MAXTIME had passed, at 100% CPU. */
            if (starts("GD_STOGO")) {
                c.ignores_forced_stop = true;
                c.stop_note =
                    "NLopt's StoGO has no convergence test of its own and "
                    "ignores NLopt's forced stop, by which HPO ends a search "
                    "(on a signal, an error, the RESTART stagnation stop, or "
                    "a loss at or below STOPVAL that NLopt was not given): it "
                    "then calls on, each call refused, and only NLopt's own "
                    "evaluation count ends it promptly (MAXTIME only once the "
                    "rest of its time has passed)";
            }
        } else if (s == "LD_MMA" || s == "LD_CCSAQ" || s == "LD_SLSQP") {
            c.needs_stop = true;
            c.stop_note  = "without a tolerance it has no convergence test "
                           "and runs until it is interrupted";
        }
        /* cdirect.c tests XTOL on the rectangles it divides in a sweep, and
         * divides the largest one in every sweep: XTOL is met only once
         * every rectangle is that small, about (1/XTOL_REL)^n rectangles,
         * one evaluation each (the probes: XTOL_REL 1e-4 in three
         * dimensions ran past 300000 evaluations). It does end the run, a
         * finite schedule, so it stays a stop. */
        if (starts("GN_DIRECT"))
            c.tolerance_note =
                "DIRECT's XTOL_REL and XTOL_ABS end it only once every "
                "rectangle is that small, after roughly (1/XTOL_REL)^n "
                "evaluations in n dimensions";
        if (no_tolerance_test) {
            c.supports_ftol_rel = c.supports_ftol_abs = false;
            c.supports_xtol_rel = c.supports_xtol_abs = false;
        } else if (starts("GN_ORIG_DIRECT")) {
            /* direct_optimize() takes XTOL_REL as its volume tolerance and
             * no other tolerance. */
            c.supports_ftol_rel = c.supports_ftol_abs = false;
            c.supports_xtol_abs = false;
            c.untested_tolerance_note =
                "the original DIRECT tests only XTOL_REL, its volume "
                "tolerance";
        }
        if (mlsl)
            c.untested_tolerance_note =
                "MLSL's multistart loop tests no tolerance, and NLopt passes "
                "the block's tolerances to no INNER search";
        else if (auglag)
            c.untested_tolerance_note =
                "AUGLAG ends when its INNER search ends (HPO sets no "
                "constraint) and tests the block's tolerances only to name "
                "that ending, and NLopt passes them to no INNER search";
        else if (no_tolerance_test)
            c.untested_tolerance_note =
                "NLopt's global search tests no tolerance";
        if (s == "GN_MLSL" || s == "GN_MLSL_LDS" || s == "LN_AUGLAG" ||
            s == "LN_AUGLAG_EQ")
            c.default_inner = "LN_COBYLA";
        else if (s == "GD_MLSL" || s == "GD_MLSL_LDS" || s == "LD_AUGLAG" ||
                 s == "LD_AUGLAG_EQ")
            c.default_inner = "LD_MMA";
        c.reads_population = starts("GN_CRS2") || starts("GN_ISRES") ||
                             starts("GN_ESCH") || mlsl ||
                             s == "GD_STOGO_RAND";
        c.reads_vector_storage = s == "LD_LBFGS" || starts("LD_VAR") ||
                                 starts("LD_TNEWTON");
        c.reads_seed = true;

        /* Initial-step consumption, verified against nlopt 2.10
         * src/api/optimize.c (see StepUse doc in optimizer_caps.h). */
        auto has = [&](const char* sub) { return s.find(sub) != std::string::npos; };
        if (has("SBPLX") || has("NELDERMEAD")) {
            c.step_use  = StepUse::PerDimVector;
            c.step_note = "per-dim first simplex edges; no coordinate rescaling; later "
                          "step scale is one L1 progress ratio in [0.1,10] applied to "
                          "every dim; the per-dim x-stop guard is |x|-relative (use XTOL_ABS)";
        } else if (has("BOBYQA") || has("COBYLA")) {
            c.step_use  = StepUse::PerDimRescale;
            c.step_note = "coordinates rescaled by the step vector (rhobeg = step in every "
                          "dim after rescaling); rhoend = XTOL_REL*rhobeg, floored by XTOL_ABS";
        } else if (has("NEWUOA") || has("PRAXIS")) {
            c.step_use  = StepUse::MinCollapsed;
            c.step_note = "uses one radius = min over dims of |step|";
        } else if (has("MMA") || has("CCSAQ")) {
            c.step_use  = StepUse::TrustHalfWidth;
            c.step_note = "per-dim initial trust-region half-width (default 0.5*width)";
        } else if (has("MLSL") || has("AUGLAG")) {
            c.step_use  = StepUse::ForwardedToInner;
            c.step_note = "forwarded to the INNER local optimiser (an OUTER STEP overrides "
                          "an INNER STEP)";
        } else {
            c.step_use  = StepUse::Ignored;
            c.step_note = "this algorithm never reads an initial step (population, "
                          "branch-and-bound, or line-search method); remove STEP";
        }
        if (s == "LN_NEWUOA") {
            c.ignores_bounds = true;
            c.bounds_note    = "ignores the OPTIM box: NLopt calls LN_NEWUOA without bounds "
                               "(optimize.c passes lb = ub = NULL), so its evaluations can "
                               "leave the box, and such a point can become the best; the "
                               "first is reported with a WARNING and DIAGNOSTICS counts them "
                               "all. LN_BOBYQA keeps to the box";
        }

        add(r, c);
    }

    return r;
}

const Registry& registry()
{
    static const Registry r = build_registry();
    return r;
}

}  // namespace

const char* step_use_name(StepUse u)
{
    switch (u) {
        case StepUse::Rejected:         return "rejected (no initial-step concept)";
        case StepUse::Ignored:          return "ignored by the algorithm (rejected)";
        case StepUse::PerDimVector:     return "per-dim vector, raw simplex edges";
        case StepUse::PerDimRescale:    return "per-dim vector -> coordinate rescaling";
        case StepUse::MinCollapsed:     return "collapsed to min over dims of |step|";
        case StepUse::TrustHalfWidth:   return "per-dim trust-region half-width";
        case StepUse::ForwardedToInner: return "forwarded to the INNER optimiser";
        case StepUse::GridCount:        return "grid point count (use PARAM GRID_POINTS)";
        case StepUse::WidthFraction:    return "family-owned width fraction (PARAM STEP_FRAC)";
    }
    return "?";
}

bool step_use_consumes(StepUse u)
{
    switch (u) {
        case StepUse::PerDimVector:
        case StepUse::PerDimRescale:
        case StepUse::MinCollapsed:
        case StepUse::TrustHalfWidth:
        case StepUse::ForwardedToInner:
            return true;
        default:
            return false;
    }
}

const char* backend_refusal(std::string_view lib, std::string_view algo)
{
    for (const Refusal& r : kRefused)
        if (lib == r.lib && algo == r.algo) return r.reason;
    return nullptr;
}

void reject_refused_backend(std::string_view lib, std::string_view algo,
                            const std::string& where)
{
    const char* reason = backend_refusal(lib, algo);
    if (!reason) return;
    throw std::runtime_error((where.empty() ? "" : where + ": ") +
                             std::string(lib) + ":" + std::string(algo) +
                             " is refused: " + reason);
}

const OptimizerCaps* find_optimizer_caps(std::string_view lib,
                                         std::string_view algo)
{
    const auto& reg = registry();
    auto it = reg.find({std::string(lib), std::string(algo)});
    return it == reg.end() ? nullptr : &it->second;
}

const OptimizerCaps& optimizer_caps(std::string_view lib,
                                    std::string_view algo)
{
    const OptimizerCaps* caps = find_optimizer_caps(lib, algo);
    if (!caps) {
        reject_refused_backend(lib, algo);
        throw std::runtime_error("unknown optimizer " + std::string(lib) +
                                 ":" + std::string(algo) +
                                 " (not registered in optimizer_caps)");
    }
    return *caps;
}

const OptimizerCaps& optimizer_caps(const OptimizerSpec& sp)
{
    return optimizer_caps(sp.lib, sp.algo);
}

std::vector<const OptimizerCaps*> filter_optimizer_caps(CapsPredicate pred)
{
    std::vector<const OptimizerCaps*> out;
    for (const auto& kv : registry()) {
        if (pred(kv.second)) out.push_back(&kv.second);
    }
    return out;
}

std::string format_caps_list(const std::vector<const OptimizerCaps*>& caps,
                             std::size_t max_show)
{
    if (caps.empty()) return "<none registered>";
    /* Sorted by LIB:ALGO: the registry is a hash map, so the order of
     * filter_optimizer_caps() carries no meaning, and a message must read
     * the same on every run. */
    std::vector<std::string> names;
    names.reserve(caps.size());
    for (const auto* c : caps) names.push_back(c->lib + ":" + c->algo);
    std::sort(names.begin(), names.end());
    std::string s;
    const std::size_t n_show = std::min(max_show, names.size());
    for (std::size_t i = 0; i < n_show; ++i) {
        if (i) s += ", ";
        s += names[i];
    }
    if (n_show < names.size()) {
        s += ", ... (";
        s += std::to_string(names.size() - n_show);
        s += " more)";
    }
    return s;
}

}  // namespace tadah::hpo

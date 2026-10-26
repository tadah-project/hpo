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
 * (backend_refusal(), optimizer_caps.h): a defect measured in the library
 * version HPO builds, or a backend that does not solve the problem HPO
 * poses -- one scalar loss, minimised inside the OPTIM box. Before an
 * upgrade of the library lifts an entry, the entry's repro is run again.
 *
 * NLOPT:LN_NEWUOA -- NLopt v2.10.0 calls Powell's NEWUOA without bounds
 * (optimize.c: newuoa(..., lb = ub = NULL, ...)), so the search leaves the
 * OPTIM box: a 2-D run of MAXEVAL 25 (RCUT2B [4, 6], CGRID2B [0.1, 2])
 * made 20 of its evaluations outside the box, and the best one, RCUT2B 8.11,
 * became best_pot.tadah; in 10 of 720 survey cases without XTOL it passed
 * NaN coordinates to the objective. NEWUOA needs two dimensions or more,
 * and NLopt refused one only once the run had started ("dimension 1 must
 * be >= 2", 0 evaluations). The box is the configuration's contract, and
 * LN_BOBYQA is Powell's bound-constrained successor of NEWUOA, so HPO
 * refuses LN_NEWUOA as it refuses LN_NEWUOA_BOUND. The analysis:
 * docs/issues/nlopt_newuoa_ignores_bounds.md.
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
 * docs/issues/nlopt_newuoa_bound_stall.md. The NLopt version the NLOPT
 * entries were measured on is kNloptTestedVersion (nlopt_tested_version.h):
 * a build against another fails test_nlopt_version.cpp.
 *
 * NLOPT:NLOPT_LD_LBFGS_NOCEDAL -- NLopt v2.10.0 names it (nlopt.h,
 * nlopt_algorithm_to_string: the name the registry below registers each
 * NLopt algorithm under) but does not implement it: nlopt_optimize()'s
 * algorithm switch has no case for it and returns NLOPT_INVALID_ARGS
 * (optimize.c), so every run of it ended "FAILED -- nlopt invalid
 * argument" once its setup was done. The repro is the unit test that
 * calls nlopt_optimize() with it (test_optimizer_caps_step.cpp).
 *
 * CERES:LBFGS, WOLFE, NLCG -- Ceres' GradientProblemSolver has no bounds, and
 * HPO does not evaluate a point outside the OPTIM box: it scores it as a
 * failure graded by its distance to the box (box_penalty.h), a loss in [0.99975, 1) x FAILSCORE, far above every
 * usable loss. These three search with Ceres' Wolfe line search, whose zoom
 * phase interpolates that value with a cubic and shrinks the step to a
 * vanishing fraction of the trial (a fraction of order |slope x step| /
 * (FAILSCORE - loss)): measured with CERES:LBFGS, WOLFE and NLCG alike,
 *   - OPTIM RCUT2B [4, 6], CGRID2B (2) [0.1, 2] on the shared Ta model of
 *   the tests (tests/data/ta/common), FAILSCORE 1e16, MAXEVAL 25: the first
 *   trial left the box, the search took a step of 0 and ended "Parameter
 *   tolerance reached" at its start after 16 evaluations (loss 13.84); it
 *   had evaluated 20 of its 25 points outside the box before, ending at
 *   MAXEVAL with RCUT2B 11.2;
 *   - the straddle fixture of test_hpo_termination.cpp, FAILSCORE 1e9,
 *   FAILSCORE_GRADING ON, from CGRID2B[2] = 1.4: after two trials outside
 *   the box it moved by 2.5e-7 per iteration and ran to MAXEVAL 60 (best
 *   95.39 on a stencil point, 1.968), where it had converged before
 *   (XTOL_REL, 15 evaluations, at 2.4 outside the box).
 * CERES:STEEPEST_DESCENT, whose Armijo line search contracts each step by
 * 1e-3 at most, converges on the straddle fixture (FTOL_REL, 30 evaluations,
 * 1.985) and stays registered. The box is the configuration's contract, so HPO refuses the
 * three, naming the bound-constrained alternatives.
 *
 * CERES:LEVENBERG_MARQUARDT, DOGLEG, SUBSPACE_DOGLEG -- Ceres' least-squares
 * Solver minimises a sum of squared residuals, with their Jacobian; HPO
 * gives an optimiser one scalar loss, over which these methods degenerate
 * (a single residual, a rank-one model). Running them needs the loss as
 * residuals and their derivatives, which HPO does not expose. They had a
 * tailored refusal of their own (reject_deferred_ceres_nlls), which the
 * caps lookup of HPOTargets' constructor pre-empted with "unknown
 * optimizer"; as entries here every lookup gives their reason. */
struct Refusal {
    const char* lib;
    const char* algo;
    const char* reason;
};
constexpr const char* kCeresLeastSquares =
    "tadah hpo does not run Ceres' least-squares solver (LEVENBERG_MARQUARDT, "
    "DOGLEG, SUBSPACE_DOGLEG): it minimises a sum of squared residuals with "
    "their derivatives, and tadah hpo gives an optimiser one scalar loss, "
    "over which these methods degenerate. The recommended replacements are "
    "CERES:STEEPEST_DESCENT, Ceres' line-search solver of a scalar loss "
    "that tadah hpo runs, and the bound-constrained gradient searches "
    "NLOPT:LD_LBFGS and DLIB:BFGS";
constexpr const char* kCeresWolfe =
    "Ceres' line-search solver has no bounds, and a point it proposes "
    "outside the OPTIM box is not evaluated: it is scored as a failure "
    "graded by its distance to the box, in [0.99975, 1) x FAILSCORE, far "
    "above every usable loss. Ceres' Wolfe line search interpolates that "
    "value and shrinks its step to almost nothing (a 2-D run stopped at its "
    "start after 16 evaluations, 'Parameter tolerance reached' with a step "
    "of 0; a 1-D run moved by 2.5e-7 per iteration until MAXEVAL). The "
    "recommended replacements are NLOPT:LD_LBFGS (L-BFGS within the box), "
    "DLIB:BFGS and CERES:STEEPEST_DESCENT, whose Armijo line search steps "
    "back from such a point";
constexpr Refusal kRefused[] = {
    {"NLOPT", "LN_NEWUOA_BOUND",
     "NLopt 2.10 plans each LN_NEWUOA_BOUND step with a nested LD_MMA run "
     "that has no time limit and sees no stop request. Once a non-finite "
     "value reaches its model, each step takes about 10-60 s without "
     "evaluating the loss. Such a value reaches it after every stop tadah "
     "hpo makes through the objective (a signal, a RESTART stagnation "
     "stop, an error that ends the run) from a run's (2n+1)-th evaluation "
     "on, the last of NEWUOA's starting points for n OPTIM dimensions, as "
     "the objective then returns HUGE_VAL, NLopt's convention for a forced "
     "stop. It reaches it too when the loss barely changes over the trust "
     "region. Ctrl-C, SIGTERM, MAXTIME and RESTART wait that long. The "
     "recommended replacement is NLOPT:LN_BOBYQA, Powell's "
     "bound-constrained successor of NEWUOA (its STEP rescales each "
     "dimension instead of setting one radius)"},
    {"NLOPT", "LN_NEWUOA",
     "NLopt 2.10 runs LN_NEWUOA without the OPTIM box: it passes NEWUOA no "
     "bounds, so its evaluations leave the box and the best potential can "
     "lie outside the bounds the configuration sets (a 2-D run made 20 of "
     "its 25 evaluations outside the box, and its best was one of them). It "
     "also needs two OPTIM dimensions or more. The recommended replacement "
     "is NLOPT:LN_BOBYQA, Powell's bound-constrained successor of NEWUOA "
     "(its STEP rescales each dimension instead of setting one radius)"},
    {"NLOPT", "NLOPT_LD_LBFGS_NOCEDAL",
     "NLopt 2.10 does not implement it: NLopt names it, but its optimiser "
     "rejects it as an invalid argument, so a run of it would fail once its "
     "setup was done. The recommended replacement is NLOPT:LD_LBFGS, "
     "NLopt's L-BFGS"},
    {"CERES", "LBFGS", kCeresWolfe},
    {"CERES", "WOLFE", kCeresWolfe},
    {"CERES", "NLCG", kCeresWolfe},
    {"CERES", "LEVENBERG_MARQUARDT", kCeresLeastSquares},
    {"CERES", "DOGLEG", kCeresLeastSquares},
    {"CERES", "SUBSPACE_DOGLEG", kCeresLeastSquares},
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
        c.reads_threads   = true;   /* its replicas, and the REPLICAS default */
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
         * (probe runs ran it until it was killed). */
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
        c.reads_threads    = true;  /* the walkers run concurrently */
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
        c.reads_threads   = true;   /* a block of generated states (Gener_Block) */
        c.reads_seed      = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* No ftol/xtol: ASA's own plateau stops are PARAM Cost_Precision
         * + Maximum_Cost_Repeat and Accepted_To_Generated_Ratio. */
        c.step_use  = StepUse::Rejected;
        c.step_note = "the width of the generating distribution follows the parameter "
                      "temperature of each dimension (Initial_Parameter_Temperature, "
                      "Temperature_Ratio_Scale)";
        add(r, c);
    }

    /* DLIB ----------------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "GFS";
        c.is_global          = true;
        c.reads_threads      = true;  /* find_min_global's evaluation pool */
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
        c.step_note = "the line search starts along the negative gradient; PARAM fd_step "
                      "sets the finite-difference step";
        add(r, c);
    }

    /* CERES -- GradientProblemSolver: STEEPEST_DESCENT (Armijo line
     * search), local, gradient-based, with the function/parameter tolerances
     * (relative) and a gradient-norm tolerance via PARAM tol_gradinf. The
     * Wolfe-line-search directions (LBFGS, WOLFE, NLCG) and the Solver
     * family (LEVENBERG_MARQUARDT, DOGLEG, SUBSPACE_DOGLEG) are refused
     * (kRefused above). */
    {
        const char* ceres_grad_algos[] = {"STEEPEST_DESCENT"};
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
            c.bounds_note        = "ignores the OPTIM box: Ceres' line-search solver has no "
                                   "bounds, so its line searches can propose points outside "
                                   "the box (a LOG dimension stays positive, a linear one is "
                                   "unbounded). Such a point is not evaluated: it is scored as "
                                   "a failure graded by its distance to the box, whose slope "
                                   "points away from the box, so a descent step leads back "
                                   "into it; it counts against MAXEVAL and it is never the "
                                   "best; the first is reported with a WARNING and the "
                                   "DIAGNOSTICS section of hpo_run.log counts them all. "
                                   "DLIB:BFGS and the NLopt LD_* algorithms keep to the box";
            add(r, c);
        }
    }

    /* NLopt -- enumerate the catalogue. Every NLopt algorithm takes the
     * same StopCriteria via nlopt::opt::set_*; algorithm-class flags are
     * inferred from the canonical name prefix.
     *
     * Which stops end a run (nlopt 2.10 sources; probe runs ran
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
                    "ignores NLopt's forced stop, by which tadah hpo ends a "
                    "search "
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
                "AUGLAG ends when its INNER search ends (tadah hpo sets no "
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
        /* Where the first call is (OptimizerCaps::starts_at_x0): StoGO,
         * DIRECT and AGS take x0 as their output only. */
        c.starts_at_x0 = !(starts("GD_STOGO") || starts("GN_DIRECT") ||
                           starts("GN_ORIG_DIRECT") || starts("GN_AGS"));
        /* Where they start instead (stogo/global.cc FillRegular and
         * FillRandom, cdirect.c, ags.cc; measured as the PRESTAGE refinement
         * of a 2-D run in [4, 6] x [0.1, 0.4]: GN_DIRECT_L's first points
         * (5, 0.25), (4.33, 0.25), (5.67, 0.25), GD_STOGO's (5, 0.25) and
         * its stencil, GN_AGS's (5.0002, 0.2500), (4.0002, 0.25),
         * (5.9998, 0.25); none at the candidate). */
        if (starts("GD_STOGO"))
            c.start_note =
                s == "GD_STOGO_RAND"
                    ? "StoGO evaluates the centre of the OPTIM box first (with "
                      "POPULATION 2n + 1 or more, n the OPTIM dimensions, a "
                      "random point of the box) and then samples the whole "
                      "box; NLopt never reads that point"
                    : "StoGO evaluates the centre of the OPTIM box first and "
                      "then samples the whole box; NLopt never reads that "
                      "point";
        else if (starts("GN_DIRECT") || starts("GN_ORIG_DIRECT"))
            c.start_note = "DIRECT evaluates the centre of the OPTIM box first "
                           "and then divides the whole box; NLopt never reads "
                           "that point";
        else if (starts("GN_AGS"))
            c.start_note = "AGS evaluates points of a space-filling curve "
                           "through the whole OPTIM box, the first near its "
                           "centre; NLopt never reads that point";
        /* Which of them draw random numbers (OptimizerCaps::random_draws):
         * GD_STOGO_RAND its POPULATION trial points (stogo.cc, POP(2n)),
         * the box centre first below POPULATION 2n + 1; GN_DIRECT_L_RAND
         * only between equally long sides (cdirect.c, which_div 2); the
         * other StoGO and DIRECT forms and AGS none (optimize.c hands every
         * _NOSCAL form which_alg 13, GN_DIRECT_L_NOSCAL's). */
        if (!c.starts_at_x0)
            c.random_draws =
                s == "GD_STOGO_RAND" ? OptimizerCaps::RandomDraws::TrialPoints
                : s == "GN_DIRECT_L_RAND"
                    ? OptimizerCaps::RandomDraws::BetweenSides
                    : OptimizerCaps::RandomDraws::Never;

        /* Initial-step consumption, verified against nlopt 2.10
         * src/api/optimize.c (see StepUse doc in optimizer_caps.h). */
        auto has = [&](const char* sub) { return s.find(sub) != std::string::npos; };
        if (has("SBPLX") || has("NELDERMEAD")) {
            c.step_use  = StepUse::PerDimVector;
            c.step_note = "the first simplex has an edge of STEP in each dimension, and "
                          "the coordinates are not rescaled; the later steps of every "
                          "dimension change by one progress ratio between 0.1 and 10. "
                          "The stop on a small step is relative to |x| in each "
                          "dimension; XTOL_ABS adds an absolute one";
        } else if (has("BOBYQA") || has("COBYLA")) {
            c.step_use  = StepUse::PerDimRescale;
            c.step_note = "STEP rescales the coordinates, so that the first trust-region "
                          "radius is STEP in each dimension; the last radius is XTOL_REL "
                          "times the first, and not below XTOL_ABS";
        } else if (has("NEWUOA") || has("PRAXIS")) {
            c.step_use  = StepUse::MinCollapsed;
            c.step_note = "one radius for every dimension, the smallest STEP of the "
                          "dimensions";
        } else if (has("MMA") || has("CCSAQ")) {
            c.step_use  = StepUse::TrustHalfWidth;
            c.step_note = "the first trust-region half-width in each dimension; without "
                          "STEP it is half the width of the OPTIM box";
        } else if (has("MLSL") || has("AUGLAG")) {
            c.step_use  = StepUse::ForwardedToInner;
            c.step_note = "passed to the local optimiser of the INNER block; a STEP of the "
                          "enclosing block replaces the STEP of the INNER block";
        } else {
            c.step_use  = StepUse::Ignored;
            c.step_note = "this algorithm never reads an initial step (population, "
                          "branch-and-bound, or line-search method); remove STEP";
        }
        /* Every registered NLopt algorithm keeps to the box: LN_NEWUOA,
         * which NLopt runs without it, is refused (kRefused). */

        /* The PARAM keys NLopt reads for the algorithm (OptimizerCaps::
         * algorithm_params): the names optimize.c passes to
         * nlopt_get_param in its case of the algorithm. */
        if (starts("GN_DIRECT"))
            c.algorithm_params = {"magic_eps"};
        else if (starts("GN_ORIG_DIRECT"))
            c.algorithm_params = {"magic_eps", "magic_eps_abs", "sigma_reltol",
                                  "fglobal_reltol"};
        else if (s == "LN_PRAXIS")
            c.algorithm_params = {"t0_tol"};
        else if (c.reads_vector_storage || starts("LD_TNEWTON"))
            c.algorithm_params = {"tolg"};
        else if (s == "LD_MMA" || s == "LD_CCSAQ")
            c.algorithm_params = {"inner_maxeval", "verbosity", "rho_init",
                                  "dual_algorithm", "dual_ftol_rel",
                                  "dual_ftol_abs", "dual_xtol_rel",
                                  "dual_xtol_abs", "dual_maxeval"};

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
        case StepUse::PerDimVector:     return "one per dimension: the edges of the first simplex";
        case StepUse::PerDimRescale:    return "one per dimension: rescaling of the coordinates";
        case StepUse::MinCollapsed:     return "one radius: the smallest STEP of the dimensions";
        case StepUse::TrustHalfWidth:   return "one per dimension: the first trust-region half-width";
        case StepUse::ForwardedToInner: return "passed to the optimiser of the INNER block";
        case StepUse::GridCount:        return "grid point count (use PARAM GRID_POINTS)";
        case StepUse::WidthFraction:    return "a fraction of the width (PARAM STEP_FRAC)";
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

std::vector<std::string> refused_backend_names()
{
    std::vector<std::string> names;
    for (const Refusal& r : kRefused)
        names.push_back(std::string(r.lib) + ":" + r.algo);
    std::sort(names.begin(), names.end());
    return names;
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

std::vector<std::string> optimizer_libraries()
{
    std::vector<std::string> libs;
    for (const auto& kv : registry())
        if (std::find(libs.begin(), libs.end(), kv.first.lib) == libs.end())
            libs.push_back(kv.first.lib);
    std::sort(libs.begin(), libs.end());
    return libs;
}

void require_registered_backend(std::string_view lib, std::string_view algo,
                                const std::string& where)
{
    reject_refused_backend(lib, algo, where);
    if (find_optimizer_caps(lib, algo)) return;
    const std::vector<std::string> libs = optimizer_libraries();
    std::string why;
    if (std::find(libs.begin(), libs.end(), std::string(lib)) != libs.end()) {
        why = "LIB " + std::string(lib) + " has no ALGO " + std::string(algo);
    } else {
        why = "there is no LIB " + std::string(lib) + " (LIB is ";
        for (std::size_t i = 0; i < libs.size(); ++i)
            why += (i == 0 ? "" : i + 1 == libs.size() ? " or " : ", ") +
                   libs[i];
        why += ")";
    }
    throw std::runtime_error((where.empty() ? "" : where + ": ") +
                             "unknown optimiser " + std::string(lib) + ":" +
                             std::string(algo) + ": " + why +
                             "; `tadah explain ALGO` lists the optimisers");
}

const OptimizerCaps& optimizer_caps(std::string_view lib,
                                    std::string_view algo)
{
    const OptimizerCaps* caps = find_optimizer_caps(lib, algo);
    /* Throws for a refused or unknown (lib, algo): caps is set below. */
    if (!caps) require_registered_backend(lib, algo);
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

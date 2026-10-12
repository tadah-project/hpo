#pragma once
/*
 * backend_rules -- the checks each optimiser backend makes of its spec
 * (attributes, PARAM keys, budget) before it runs, and the startup
 * application of them to every phase of a STRATEGY.
 *
 * Each backend keeps its own validator next to its driver (DLIB:BFGS and
 * CERES in their adapters, the annealers' Knobs::parse, ASA's
 * parse_asa_options) or here (TADAH:RANDOM, TADAH:GRID, DLIB:GFS,
 * NLopt), and its driver calls it when it starts: a safety net for specs
 * built at run time or outside HPOTargets. validate_backend() calls the
 * same validator for any spec; validate_phase_backends() applies it, at
 * startup, to the spec each phase of the STRATEGY runs with
 * (strategy_phase_specs(), phase_specs.h), so a setting a backend cannot
 * honour stops the run before any data is loaded or any evaluation runs.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tadah::hpo {

/* Whether the RESTART stagnation stop bounds a run. RestartPolicy
 * (run_policy.h), built for an active RESTART (AFTER_STAGNATION and MAX
 * both positive, RestartSpec::active()), arms HPO_Host's stagnation
 * watchdog for the whole run; once its MAX restarts are used, the next
 * AFTER_STAGNATION evaluations without a new best end the run. The watchdog
 * reaches every phase it wraps -- each MULTISTART start, the PRESTAGE
 * exploration (which ends at its first stagnation) and refinement, and the
 * local search of NLopt's MLSL and AUGLAG -- so it is a stop of each of
 * them. */
enum class RestartStop { Off, On };

/* At startup: whether the STRATEGY of `root` has an active RESTART. */
RestartStop restart_stop_of(const OptimizerSpec& root);

/* At run time: whether RestartPolicy has armed the stagnation watchdog of
 * `Host` (HPO_Host<Vec,Runner>) -- the run a driver starts is one it
 * wraps. */
template <class Host>
RestartStop restart_stop_armed()
{
    return Host::get_stagnation_threshold() > 0 ? RestartStop::On
                                                : RestartStop::Off;
}

/* A backend with no convergence test and no finite schedule
 * (OptimizerCaps::needs_stop: NLopt's global searches, MLSL, LD_MMA,
 * LD_CCSAQ and LD_SLSQP, TADAH:ANNEAL_PT, TADAH:RANDOM, DLIB:GFS) cannot run
 * without a stop: it would run until it is interrupted. Its stops are
 * MAXEVAL, MAXTIME and STOPVAL, a tolerance above 0 its stop tests read (the
 * caps' supports_* flags; NLopt switches a tolerance of 0 or below off, and
 * ANNEAL_PT's best-stall rule arms only above 0), and the RESTART
 * stagnation stop (`restart`) -- only MAXEVAL for a backend that ignores
 * NLopt's forced stop (OptimizerCaps::ignores_forced_stop: GD_STOGO,
 * GD_STOGO_RAND), which only NLopt's own count ends promptly once HPO has
 * to stop it (its clock, only after the rest of MAXTIME). Throws
 * std::runtime_error naming the
 * backend, why, and the stops it honours when `spec` sets none of them,
 * e.g. "NLOPT:GN_ESCH needs a stop: ...; set MAXEVAL, MAXTIME or STOPVAL
 * (it honours no tolerance)", followed by the caps' tolerance_note, if any.
 * Every other backend ends on its own when no stop is set. */
void require_stop(const OptimizerSpec& spec, RestartStop restart);

/* Whether `spec` sets a stop that ends a run of its backend (as
 * require_stop() counts them: a tolerance only above 0), the RESTART
 * stagnation stop aside. The journal says what ends a run that has none. */
bool sets_own_stop(const OptimizerSpec& spec);

/* The local search NLopt's MLSL or AUGLAG runs inside a run of `parent`
 * (inner_run_spec(), phase_specs.h) must end: it takes the enclosing
 * block's MAXEVAL, MAXTIME and STOPVAL, and its own MAXEVAL, MAXTIME and
 * the tolerances it honours (an INNER block's STOPVAL is replaced by the
 * enclosing block's), and the RESTART stagnation stop ends it too. AUGLAG
 * ends when that search ends. Throws naming the search and where to set a
 * stop; a no-op when `parent` runs none. */
void require_inner_run_stop(const OptimizerSpec& parent, RestartStop restart);

/* What a spec validate_backend() checks configures: a phase's own
 * optimiser, or the search an INNER block runs (strategy_phase_specs():
 * PhaseSpec::inner), whose STOPVAL is the enclosing block's. It changes only
 * where a refusal of an untested tolerance says to set a stop. */
enum class SpecRole { Phase, Inner };

/* Each validator below refuses an attribute its backend never reads, with
 * the same rule (reject_unread_attributes, backend_rules.cpp, from the
 * caps: optimizer_caps.h):
 *   - a tolerance its run does not test: no supports_* flag marks it (but
 *     DLIB:GFS's FTOL_ABS, its solver epsilon), or it is set on NLopt's
 *     MLSL or AUGLAG with an INNER block, which NLopt hands only to the
 *     default local optimiser it creates without one. "TADAH:RANDOM does
 *     not support FTOL_REL: uniform random search has no convergence test
 *     to apply it to; stop it with MAXEVAL, MAXTIME or STOPVAL". The
 *     refusal on an MLSL or AUGLAG says to set it in the INNER block when
 *     that block's search tests it, and otherwise names the stops that
 *     search honours ("...; the INNER block's NLOPT:GN_ESCH does not test
 *     it either: stop that search with MAXEVAL or MAXTIME in the INNER
 *     block, or MAXEVAL, MAXTIME or STOPVAL in the NLOPT:AUGLAG block, which
 *     NLopt hands on to it");
 *   - POPULATION, VECTOR_ARR or SEED it does not read (reads_population,
 *     reads_vector_storage, reads_seed): "NLOPT:LN_BOBYQA does not use
 *     POPULATION; remove it (it is read by ...)". DLIB:GFS's SEED is
 *     reported with a warning when it runs (warns_unread_seed). */

/* TADAH:RANDOM: no PARAM, no attribute it never reads (the tolerances,
 * POPULATION, VECTOR_ARR), and require_stop() (MAXEVAL, MAXTIME or
 * STOPVAL). */
void validate_random_spec(const OptimizerSpec& spec, RestartStop restart);

/* DLIB:GFS: no PARAM; FTOL_REL, XTOL_REL, XTOL_ABS, POPULATION and
 * VECTOR_ARR are rejected (find_min_global has no convergence test to apply
 * the tolerances to, and no population or vector storage; FTOL_ABS is its
 * solver epsilon); require_stop() (MAXEVAL, MAXTIME or STOPVAL). */
void validate_dlib_gfs_spec(const OptimizerSpec& spec, RestartStop restart);

/* TADAH:GRID: STEP is rejected (it is not the grid resolution), PARAM
 * GRID_POINTS is an integer >= 2 (default 20) and the only PARAM, and the
 * tolerances, POPULATION, VECTOR_ARR and SEED are rejected (the grid draws
 * no random numbers). Returns the number of grid points per dimension. */
unsigned validate_grid_spec(const OptimizerSpec& spec);

/* NLopt: MAXEVAL fits the int NLopt counts evaluations in, for the spec and
 * its INNER block; the INNER block has no seed (NLopt draws the INNER
 * search from the enclosing run's random stream), no GRAD_STRATEGY and no
 * STOPVAL (mlsl.c and auglag.c replace it with the enclosing run's); no
 * attribute the algorithm never reads (`role`: SpecRole); require_stop()
 * and require_inner_run_stop(). Throws std::invalid_argument for the INNER
 * spec and MAXEVAL, std::runtime_error for the rest; make_nlopt() applies it
 * before it builds the nlopt::opt. */
void validate_nlopt_spec(const OptimizerSpec& spec, RestartStop restart,
                         SpecRole role = SpecRole::Phase);

/* ASA: at least one OPTIM dimension, and the ASA options
 * (parse_asa_options: PARAM keys, their ranges and cross-checks, and the
 * generic attributes ASA does not implement). */
void validate_asa_spec(const OptimizerSpec& spec);

/* Every check the backend of `spec` (its LIB:ALGO) makes before it runs:
 * a refused LIB:ALGO (Ceres' least-squares solvers among them) and an
 * unknown one are rejected (require_registered_backend(),
 * optimizer_caps.h), then the backend's own validator runs (the annealers'
 * Knobs::parse and ANNEAL_PT's replica count, ASA, DLIB:BFGS, DLIB:GFS,
 * the CERES drivers, NLopt with `role`, TADAH:RANDOM and TADAH:GRID), then
 * require_stop() (`restart`: whether the RESTART stagnation stop bounds
 * the run). A spec that names no optimiser ("NOT SET") is left to
 * require_driver(). Throws with the backend's own message, e.g.
 * "DLIB:BFGS does not support MAXTIME". */
void validate_backend(const OptimizerSpec& spec, RestartStop restart,
                      SpecRole role = SpecRole::Phase);

/* validate_backend() on the spec of every phase the STRATEGY of `root`
 * runs (strategy_phase_specs(): the STRATEGY-level optimiser, each
 * MULTISTART start, the PRESTAGE exploration and refinement, and every
 * INNER block, as SpecRole::Inner), with the RESTART of `root`
 * (restart_stop_of()), after the rule that NLOPT, DLIB and CERES need a
 * search space of at least one dimension ("... NLOPT:GN_DIRECT cannot
 * optimise a zero-dimensional search space (there is no OPTIM line; ...")),
 * each message prefixed with the phase and named at the line its block
 * closes at (with_block_line, phase_specs.h):
 * "the PRESTAGE refinement: DLIB:BFGS does not support MAXTIME". A refused
 * backend (reject_refused_backend(), optimizer_caps.h) is reported first,
 * for its own phase, with the text validate_run_policy() throws for it
 * (reject_refused_backends(), strategy_rules.h), in every STRATEGY shape --
 * a MULTISTART from INIT points and a PRESTAGE refinement of TOPK > 1
 * included, whose phases cannot be derived without the caps of their
 * backend: "the INNER block of the STRATEGY-level optimiser:
 * NLOPT:LN_NEWUOA_BOUND is refused: ...", "the EXPLORE block: ...".
 * validate_hpo_setup() calls it after validate_run_policy(). */
void validate_phase_backends(const OptimizerSpec& root);

/* The evaluations one gradient of a search with the settings of `spec`
 * costs, as GradientProvider counts them (gradient_provider.h) when the
 * caller has the loss at the point: GRAD_STRATEGY NUMERICAL 2n for central
 * differences in n OPTIM dimensions (one-sided ones within a step of a
 * bound cost the same), n for forward ones (PARAM diff_method 1);
 * ANALYTICAL 1, the analytical gradient; HYBRID at least 1 (`at_least`):
 * the analytical gradient and the stencil of each parameter it does not
 * cover, known only at a point. `text` says so for a message: "4
 * evaluations: central differences in 2 dimensions, GRAD_STRATEGY
 * NUMERICAL". The finite-difference settings must be valid (fd_settings_of,
 * fd_stencil.h). */
struct GradientCost {
    std::uint64_t evaluations = 0;
    bool          at_least = false;
    std::string   text;
};
GradientCost gradient_cost(const OptimizerSpec& spec);

/* The startup WARNINGs of the phases of `root` whose budget cannot pay for
 * one step of their gradient search, in a run that answers a point a
 * search asks for again from the search's record when `reuse` (the run's
 * permission, RepeatPolicy::reuse: HPOTargets::repeat_reuse_blocker() is
 * "", which HPO_Host::repeat_reuse_blocker() forwards) and evaluates it
 * again otherwise (repeat_points.h). A search evaluates the loss at its
 * starting point (1 evaluation) and the gradient there (gradient_cost())
 * before any point it steps to, so a MAXEVAL below gradient_cost() + 2
 * ends it where it started (MAXEVAL_REACHED). Over strategy_phase_specs():
 * every phase whose optimiser obtains gradients (obtains_gradients(): an
 * MLSL or AUGLAG through its local search) and has a MAXEVAL -- the
 * STRATEGY-level optimiser, each MULTISTART start (BUDGET_PER_START or its
 * share of MAXEVAL), the PRESTAGE exploration and refinement (with TOPK >
 * 1 its share) -- "the STRATEGY-level optimiser NLOPT:LD_LBFGS: MAXEVAL 4
 * takes no step: its gradient search evaluates the loss at a starting
 * point (1 evaluation) and the gradient there (4 evaluations: central
 * differences in 2 dimensions, GRAD_STRATEGY NUMERICAL) before it evaluates
 * a point it steps to, which needs MAXEVAL 6 or more". An MLSL first
 * evaluates its starting point and its first N samples without a gradient
 * (N its POPULATION, NLopt's default 4), and its first local search starts
 * at one of them, a point the search has evaluated: the loss there is
 * counted as the run spends it (repeat_use, the rule of RepeatPoints::use):
 * none under `reuse` and GRAD_STRATEGY NUMERICAL, whose finite differences
 * take the recorded loss as their centre, so it needs N more: "the
 * STRATEGY-level optimiser NLOPT:GD_MLSL: MAXEVAL 9 takes no step of a
 * local search: MLSL evaluates its starting point and its first 4 samples
 * without a gradient (5 evaluations; POPULATION 4, NLopt's default), then
 * its first local search, from one of them, takes the loss there from the
 * search's record (no evaluation) and evaluates the gradient there (4
 * evaluations: ...) before it evaluates a point it steps to, which needs
 * MAXEVAL 10 or more"; one evaluation otherwise (without `reuse`, or under
 * ANALYTICAL or HYBRID), N + 1 more: "... then its first local search,
 * from one of them, evaluates the loss there again (1 evaluation) and the
 * gradient there (...) ..., which needs MAXEVAL 11 or more" under NUMERICAL
 * without `reuse`, "... (1 evaluation: one analytical gradient,
 * GRAD_STRATEGY ANALYTICAL) ..., which needs MAXEVAL 8 or more" under
 * ANALYTICAL. AUGLAG goes straight into its local search. And an INNER block whose local search
 * obtains gradients and whose own MAXEVAL, in NLopt's calls (each the loss
 * and the gradient at a point), is 1; an MLSL's local searches start at
 * points it evaluated, so a call costs the gradient alone where the run
 * answers the loss from the record. Not a refusal: a budget-limited probe
 * of the start can be deliberate. validate_hpo_setup() journals each one
 * after the phase checks (the finite-difference settings are valid by
 * then). */
std::vector<std::string> gradient_budget_warnings(const OptimizerSpec& root,
                                                  bool reuse);

}  // namespace tadah::hpo

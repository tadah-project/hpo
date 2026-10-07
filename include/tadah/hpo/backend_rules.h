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
 * stagnation stop (`restart`) -- only MAXEVAL and MAXTIME for a backend
 * that ignores NLopt's forced stop (OptimizerCaps::ignores_forced_stop:
 * GD_STOGO, GD_STOGO_RAND), which HPO cannot end otherwise once it has to
 * stop it. Throws std::runtime_error naming the
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

/* TADAH:RANDOM: no PARAM, and require_stop() (MAXEVAL, MAXTIME or
 * STOPVAL). */
void validate_random_spec(const OptimizerSpec& spec, RestartStop restart);

/* DLIB:GFS: no PARAM; FTOL_REL, XTOL_REL and XTOL_ABS are rejected
 * (find_min_global has no convergence test to apply them to; FTOL_ABS is
 * its solver epsilon); require_stop() (MAXEVAL, MAXTIME or STOPVAL). */
void validate_dlib_gfs_spec(const OptimizerSpec& spec, RestartStop restart);

/* TADAH:GRID: STEP is rejected (it is not the grid resolution), PARAM
 * GRID_POINTS is an integer >= 2 (default 20) and the only PARAM. Returns
 * the number of grid points per dimension. */
unsigned validate_grid_spec(const OptimizerSpec& spec);

/* NLopt: MAXEVAL fits the int NLopt counts evaluations in, for the spec and
 * its INNER block, and the INNER block has no seed (NLopt draws the INNER
 * search from the enclosing run's random stream); require_stop() and
 * require_inner_run_stop(). Throws std::invalid_argument for the first two,
 * std::runtime_error for the stops; make_nlopt() applies it before it
 * builds the nlopt::opt. */
void validate_nlopt_spec(const OptimizerSpec& spec, RestartStop restart);

/* ASA: at least one OPTIM dimension, and the ASA options
 * (parse_asa_options: PARAM keys, their ranges and cross-checks, and the
 * generic attributes ASA does not implement). */
void validate_asa_spec(const OptimizerSpec& spec);

/* Every check the backend of `spec` (its LIB:ALGO) makes before it runs:
 * the deferred CERES least-squares algorithms and an unknown LIB:ALGO are
 * rejected, then the backend's own validator runs (the annealers'
 * Knobs::parse and ANNEAL_PT's replica count, ASA, DLIB:BFGS, DLIB:GFS,
 * the CERES drivers, NLopt, TADAH:RANDOM and TADAH:GRID), then
 * require_stop() (`restart`: whether the RESTART stagnation stop bounds
 * the run). A spec that names no optimiser ("NOT SET") is left to
 * require_driver(). Throws with the backend's own message, e.g.
 * "DLIB:BFGS does not support MAXTIME". */
void validate_backend(const OptimizerSpec& spec, RestartStop restart);

/* validate_backend() on the spec of every phase the STRATEGY of `root`
 * runs (strategy_phase_specs(): the STRATEGY-level optimiser, each
 * MULTISTART start, the PRESTAGE exploration and refinement, and every
 * INNER block), with the RESTART of `root` (restart_stop_of()), each
 * message prefixed with the phase:
 * "the PRESTAGE refinement: DLIB:BFGS does not support MAXTIME".
 * validate_hpo_setup() calls it after validate_run_policy(). */
void validate_phase_backends(const OptimizerSpec& root);

}  // namespace tadah::hpo

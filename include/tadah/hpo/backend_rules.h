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

/* A backend with no stopping rule of its own (OptimizerCaps::needs_maxeval:
 * TADAH:RANDOM, DLIB:GFS) cannot run without MAXEVAL: it would run until it
 * is interrupted. Throws std::runtime_error naming the backend and why when
 * `spec` has none. Every other backend runs without an evaluation limit
 * when MAXEVAL is left out. */
void require_maxeval_budget(const OptimizerSpec& spec);

/* TADAH:RANDOM: require_maxeval_budget(). */
void validate_random_spec(const OptimizerSpec& spec);

/* DLIB:GFS: require_maxeval_budget(). */
void validate_dlib_gfs_spec(const OptimizerSpec& spec);

/* TADAH:GRID: STEP is rejected (it is not the grid resolution), PARAM
 * GRID_POINTS is an integer >= 2 (default 20) and the only PARAM. Returns
 * the number of grid points per dimension. */
unsigned validate_grid_spec(const OptimizerSpec& spec);

/* NLopt: MAXEVAL fits the int NLopt counts evaluations in, for the spec and
 * its INNER block, and the INNER block has no seed (NLopt draws the INNER
 * search from the enclosing run's random stream). Throws
 * std::invalid_argument; make_nlopt() applies it before it builds the
 * nlopt::opt. */
void validate_nlopt_spec(const OptimizerSpec& spec);

/* ASA: at least one OPTIM dimension, and the ASA options
 * (parse_asa_options: PARAM keys, their ranges and cross-checks, and the
 * generic attributes ASA does not implement). */
void validate_asa_spec(const OptimizerSpec& spec);

/* Every check the backend of `spec` (its LIB:ALGO) makes before it runs:
 * the deferred CERES least-squares algorithms and an unknown LIB:ALGO are
 * rejected, then the backend's own validator runs (the annealers'
 * Knobs::parse and ANNEAL_PT's replica count, ASA, DLIB:BFGS, DLIB:GFS,
 * the CERES drivers, NLopt, TADAH:RANDOM and TADAH:GRID). A spec that
 * names no optimiser ("NOT SET") is left to require_driver(). Throws with
 * the backend's own message, e.g. "DLIB:BFGS does not support MAXTIME". */
void validate_backend(const OptimizerSpec& spec);

/* validate_backend() on the spec of every phase the STRATEGY of `root`
 * runs (strategy_phase_specs(): the STRATEGY-level optimiser, each
 * MULTISTART start, the PRESTAGE exploration and refinement, and every
 * INNER block), each message prefixed with the phase:
 * "the PRESTAGE refinement: DLIB:BFGS does not support MAXTIME".
 * validate_hpo_setup() calls it after validate_run_policy(). */
void validate_phase_backends(const OptimizerSpec& root);

}  // namespace tadah::hpo

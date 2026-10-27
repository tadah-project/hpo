#pragma once
/*
 * scale_validation -- fail-fast checks of the search-space / initial-step
 * setup, run once after HPOTargets parsing (all specs) and again from
 * run_inner (defence in depth for specs assembled at run time, e.g.
 * MULTISTART and PRESTAGE clones).
 *
 * The STEP acceptance policy comes from the OptimizerCaps registry
 * (OptimizerCaps::step_use / step_note), so a backend that would silently
 * drop STEP is a hard error with an actionable hint instead of a silent
 * no-op (DLIB:GFS, TADAH:RANDOM and every NLopt algorithm that never reads
 * an initial step used to ignore it; TADAH:GRID overloaded it as a point
 * count). Backends that keep their own per-adapter throws (ANNEAL family,
 * ASA, DLIB:BFGS, CERES) are reached first by this validator with the same
 * hint text.
 *
 * Start points: every resolved optim_init point must have the search
 * dimension, be finite, be > 0 on log-scaled dims (in() would take
 * std::log silently) and lie inside the OPTIM box. Before this check the
 * behaviour was backend-dependent: the ANNEAL family and ASA rejected at
 * run time, NLopt failed at optimize time ("bounds %d fail"), DLIB/CERES
 * asserted or silently clamped.
 *
 * Diagnostics follow the journal severity rule: exact facts -> INFO,
 * provable problems -> WARNING, empirical advice -> HINT.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <string>

namespace tadah::hpo {

/** Validate one spec. `where` is its place, "OUTER", "INNER", "PRESTAGE"
 *  or "PRESTAGE.INNER" (or "run" and "run.INNER" for a spec built at run
 *  time); the messages name the optimiser in words ("the STRATEGY-level
 *  optimiser", "the INNER search", ...). Throws std::runtime_error on the
 *  first violated rule; journals warnings/hints. `check_start_points`
 *  is false on the run_inner path (seeds were validated at startup, and
 *  run-time stage seeds come from the INIT chain or the top-K, already
 *  inside the box). */
void validate_optimizer_scaling(const OptimizerSpec& sp,
                                const std::string& where,
                                bool check_start_points = true);

/** Walk the whole spec tree of the parsed OPTIMIZER block (OUTER, INNER,
 *  PRESTAGE EXPLORE and its INNER) and validate each. Called twice: from
 *  the HPOTargets constructor with
 *  check_start_points=false (config-level STEP policy, so a bad config
 *  fails at parse time and unit tests can exercise it without a
 *  training context), and from validate_hpo_setup() with
 *  check_start_points=true (seeds inside the box + the width audit,
 *  alongside the other fail-fast startup rules, after the bump-support
 *  gate has had its say). */
void run_scale_validation(const OptimizerSpec& root,
                          bool check_start_points = true);

/** Journal the real-space width audit of the search box: min/max width,
 *  the max/min ratio (INFO) and any dim whose width is below 1e-12 of
 *  its magnitude (WARNING: effectively pinned). Exposed for tests. */
void audit_search_space_widths(const OptimizerSpec& sp);

}  // namespace tadah::hpo

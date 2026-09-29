#pragma once
/*
 * phase_specs -- the spec each phase of a STRATEGY runs with, derived from
 * the parsed spec. The single source of truth for:
 *   - MultiStartPolicy and StagedPolicy (run_policy.h), which run the
 *     phases: plan_multistart(), multistart_start_spec() and
 *     prestage_refinement_spec();
 *   - validate_hpo_setup(), which checks each phase's backend settings at
 *     startup (validate_phase_backends(), backend_rules.h), through
 *     strategy_phase_specs(), built on the same three functions.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tadah::hpo {

/* How MULTISTART runs `spec` from its INIT points. */
struct MultiStartPlan {
    /* One run of `spec` itself: a multi-seed backend (DLIB:GFS) consumes
     * every INIT point in that run, and a spec without INIT points has no
     * point to start each run from. */
    bool single_run = false;
    /* The starts run: N_STARTS, or the INIT points when there are fewer. */
    std::size_t starts = 0;
    /* Each start's MAXEVAL: BUDGET_PER_START, or the outer MAXEVAL split
     * over the starts, rounded up (so the starts together may spend up to
     * starts - 1 more than MAXEVAL); none when neither is set, and the
     * starts run with the spec's own stop criteria. */
    std::optional<std::uint64_t> per_start_maxeval;
    bool from_budget_per_start = false;
};

/* The plan of MULTISTART with `n_starts` starts and `budget_per_start`
 * (STRATEGY BUDGET_PER_START) over spec.optim_init. */
MultiStartPlan plan_multistart(const OptimizerSpec& spec, std::size_t n_starts,
                               const std::optional<std::size_t>& budget_per_start);

/* Start `i` (0-based, below plan.starts) of `plan`: `spec` with the single
 * INIT point i, the plan's per-start MAXEVAL, and SEED + i when `spec` has
 * a SEED (the starts draw different random numbers). */
OptimizerSpec multistart_start_spec(const OptimizerSpec& spec,
                                    const MultiStartPlan& plan, std::size_t i);

/* The PRESTAGE refinement: the outer `spec` without its exploration,
 * starting from the exploration's `candidates`. StagedPolicy runs it once
 * from a single candidate, and over several as a MULTISTART
 * (plan_multistart(refinement, candidates.size(), no BUDGET_PER_START)). */
OptimizerSpec prestage_refinement_spec(
    const OptimizerSpec& spec, std::vector<std::vector<double>> candidates);

/* The local search NLopt's MLSL or AUGLAG (OptimizerCaps::accepts_inner)
 * runs inside a run of `parent`, as NLopt runs it: the INNER block, or
 * without one NLopt's default local optimiser (OptimizerCaps::
 * default_inner) with the tolerances of `parent`; its MAXEVAL and MAXTIME
 * the tighter of its own and those of `parent` (NLopt hands each local run
 * what is left of the enclosing budget, nlopt_optimize_limited), and its
 * STOPVAL that of `parent`, if any (mlsl.c and auglag.c replace the local
 * optimiser's own). nullopt when `parent` runs no local search: an
 * algorithm without one, or one that needs an INNER block and has none
 * (NLopt refuses the run). */
std::optional<OptimizerSpec> inner_run_spec(const OptimizerSpec& parent);

/* True when a run of `spec` asks the host for gradients: a gradient-based
 * backend (OptimizerCaps::is_gradient_based), or an MLSL/AUGLAG whose local
 * search (inner_run_spec) is one -- NLopt hands that local optimiser the
 * gradient the objective of `spec` computes, so the GRAD_STRATEGY of
 * `spec` decides how it is obtained. */
bool obtains_gradients(const OptimizerSpec& spec);

/* The local search a run of `spec` is, when the run is one local search
 * from its starting point: `spec` itself for a local backend without a
 * local search of its own, or for NLopt's AUGLAG -- a local algorithm
 * (not OptimizerCaps::is_global) that runs a local search (accepts_inner)
 * -- that search (inner_run_spec), which is one in turn. HPO sets AUGLAG
 * no constraint, so auglag.c runs its local search once, from the starting
 * point, and ends when it ends. nullopt for a global backend -- MLSL runs a
 * local search from each of many points, GD_STOGO and the other global
 * searches sample the box -- and for an AUGLAG without a local search
 * (NLopt refuses the run). run_nlopt() stops a gradient-based one at a
 * starting point without a finite gradient, and reads its MAXEVAL and
 * MAXTIME, which AUGLAG does not report. */
std::optional<OptimizerSpec> single_local_search(const OptimizerSpec& spec);

/* One phase of a run: its name, for messages, and the spec it runs.
 * `inner` marks the search an INNER block runs: it takes the gradient of
 * the phase it runs in (obtains_gradients), which that phase's
 * GRAD_STRATEGY and PARAM fd_step / fd_kernel decide, not its own. */
struct PhaseSpec {
    std::string   name;
    OptimizerSpec spec;
    bool          inner = false;
};

/* The phases the policy of `spec` (STRATEGY TYPE) runs, derived as the
 * policies derive them:
 *   PLAIN       "the STRATEGY-level optimiser": `spec`;
 *   MULTISTART  "each MULTISTART start": the first start of
 *               plan_multistart() -- the starts differ only in their INIT
 *               point and SEED + i -- or "the MULTISTART run" when the
 *               plan is a single run;
 *   PRESTAGE    "the PRESTAGE exploration (EXPLORE block)": spec.prestage,
 *               and "the PRESTAGE refinement": prestage_refinement_spec()
 *               over TOPK candidates, as a MULTISTART start when TOPK > 1.
 *               The exploration finds the candidates at run time; here TOPK
 *               stand-ins take their place (the spec's first INIT point, or
 *               the centre of the box): what a backend checks is how many
 *               start points there are, not where they lie.
 * followed by "the INNER block of <phase>" for each phase with an INNER
 * block, as inner_run_spec() derives the search it runs (PhaseSpec::inner). RESTART changes
 * no phase's settings, only its start points and seeds. A PRESTAGE without an EXPLORE block has only its refinement here
 * (validate_run_policy() rejects it). */
std::vector<PhaseSpec> strategy_phase_specs(const OptimizerSpec& spec);

}  // namespace tadah::hpo

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

/* One phase of a run: its name, for messages, and the spec it runs. */
struct PhaseSpec {
    std::string   name;
    OptimizerSpec spec;
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
 * block. RESTART changes no phase's settings, only its start points and
 * seeds. A PRESTAGE without an EXPLORE block has only its refinement here
 * (validate_run_policy() rejects it). */
std::vector<PhaseSpec> strategy_phase_specs(const OptimizerSpec& spec);

}  // namespace tadah::hpo

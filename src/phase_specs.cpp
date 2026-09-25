#include <tadah/hpo/phase_specs.h>

#include <tadah/hpo/optimizer_caps.h>

#include <algorithm>
#include <type_traits>
#include <utility>

namespace tadah::hpo {

MultiStartPlan plan_multistart(const OptimizerSpec& spec, std::size_t n_starts,
                               const std::optional<std::size_t>& budget_per_start)
{
    MultiStartPlan plan;
    const std::size_t n_init = spec.optim_init.size();
    /* A multi-seed backend consumes every INIT point as an initial
     * evaluation of one run (keyed by the caps flag, so a future backend
     * registers the capability without touching this). */
    if (n_init == 0 || optimizer_caps(spec).accepts_multi_seed) {
        plan.single_run = true;
        plan.starts = 1;
        return plan;
    }
    plan.starts = std::min(n_starts, n_init);
    if (budget_per_start) {
        plan.per_start_maxeval = static_cast<std::uint64_t>(*budget_per_start);
        plan.from_budget_per_start = true;
    } else if (spec.stop.maxeval && plan.starts > 0) {
        const std::uint64_t total = *spec.stop.maxeval;
        const std::uint64_t k = static_cast<std::uint64_t>(plan.starts);
        plan.per_start_maxeval = total / k + (total % k != 0 ? 1 : 0);
    }
    return plan;
}

OptimizerSpec multistart_start_spec(const OptimizerSpec& spec,
                                    const MultiStartPlan& plan, std::size_t i)
{
    OptimizerSpec sub = spec;                       // deep copy
    if (i < spec.optim_init.size())
        sub.optim_init = {spec.optim_init[i]};      // single start
    if (plan.per_start_maxeval) sub.stop.maxeval = *plan.per_start_maxeval;
    /* Decorrelate the starts: reusing one seed hands every start the same
     * proposal stream, so stochastic optimisers (ANNEAL) produce
     * near-duplicate chains that differ only by their start point. */
    if (spec.seed) sub.seed = *spec.seed + static_cast<std::uint64_t>(i);
    return sub;
}

OptimizerSpec prestage_refinement_spec(
    const OptimizerSpec& spec, std::vector<std::vector<double>> candidates)
{
    OptimizerSpec refine = spec;
    refine.prestage.reset();          // the refinement does not explore again
    refine.optim_init = std::move(candidates);
    return refine;
}

std::optional<OptimizerSpec> inner_run_spec(const OptimizerSpec& parent)
{
    const OptimizerCaps& caps = optimizer_caps(parent);
    if (!caps.accepts_inner) return std::nullopt;
    OptimizerSpec run;
    if (parent.inner) {
        run = *parent.inner;
    } else if (caps.default_inner) {
        /* NLopt creates the default local optimiser with the block's
         * tolerances (optimize.c) and no evaluation limit of its own. */
        run.lib  = "NLOPT";
        run.algo = caps.default_inner;
        run.stop.ftol_rel = parent.stop.ftol_rel;
        run.stop.ftol_abs = parent.stop.ftol_abs;
        run.stop.xtol_rel = parent.stop.xtol_rel;
        run.stop.xtol_abs = parent.stop.xtol_abs;
    } else {
        return std::nullopt;
    }
    const auto tighter = [](const auto& own, const auto& enclosing) {
        using T = std::decay_t<decltype(own)>;
        if (own && enclosing) return T(std::min(*own, *enclosing));
        return own ? own : enclosing;
    };
    run.stop.maxeval = tighter(run.stop.maxeval, parent.stop.maxeval);
    run.stop.maxtime = tighter(run.stop.maxtime, parent.stop.maxtime);
    run.stop.stopval = parent.stop.stopval;
    return run;
}

namespace {

/* A point the refinement's candidates stand in for at startup: the spec's
 * first INIT point, or the centre of the box. */
std::vector<double> stand_in_point(const OptimizerSpec& spec)
{
    if (!spec.optim_init.empty()) return spec.optim_init.front();
    std::vector<double> centre(spec.low.size());
    for (std::size_t d = 0; d < centre.size() && d < spec.high.size(); ++d)
        centre[d] = 0.5 * (spec.low[d] + spec.high[d]);
    return spec.out(centre);
}

/* The phase that runs `spec` under MULTISTART `plan`. */
PhaseSpec multistart_phase(const OptimizerSpec& spec,
                           const MultiStartPlan& plan,
                           const std::string& each_start,
                           const std::string& single_run)
{
    if (plan.single_run) return {single_run, spec};
    return {each_start, multistart_start_spec(spec, plan, 0)};
}

}  // namespace

std::vector<PhaseSpec> strategy_phase_specs(const OptimizerSpec& spec)
{
    std::vector<PhaseSpec> phases;
    switch (spec.strategy_type) {
        case StrategyType::PLAIN:
            phases.push_back({"the STRATEGY-level optimiser", spec});
            break;
        case StrategyType::MULTISTART: {
            const MultiStartPlan plan = plan_multistart(
                spec, spec.multistart.n_starts, spec.multistart.budget_per_start);
            phases.push_back(multistart_phase(spec, plan,
                                              "each MULTISTART start",
                                              "the MULTISTART run"));
            break;
        }
        case StrategyType::PRESTAGE: {
            if (spec.prestage)
                phases.push_back({"the PRESTAGE exploration (EXPLORE block)",
                                  *spec.prestage});
            const std::size_t topk = std::max<std::size_t>(1, spec.stage_topk);
            OptimizerSpec refine = prestage_refinement_spec(
                spec, std::vector<std::vector<double>>(topk,
                                                       stand_in_point(spec)));
            const std::string name = "the PRESTAGE refinement";
            if (topk > 1)
                phases.push_back(multistart_phase(
                    refine, plan_multistart(refine, topk, std::nullopt), name,
                    name));
            else
                phases.push_back({name, std::move(refine)});
            break;
        }
    }
    const std::size_t n = phases.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (!phases[i].spec.inner) continue;
        /* An INNER block under a backend that runs none is the business of
         * require_inner_backend() (strategy_rules.h); it is checked as it
         * stands. */
        std::optional<OptimizerSpec> run = inner_run_spec(phases[i].spec);
        phases.push_back({"the INNER block of " + phases[i].name,
                          run ? std::move(*run) : *phases[i].spec.inner});
    }
    return phases;
}

}  // namespace tadah::hpo

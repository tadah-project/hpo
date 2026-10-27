#include <tadah/hpo/phase_specs.h>

#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/strict_parse.h>

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
    if (i < spec.optim_init.size()) {
        sub.optim_init = {spec.optim_init[i]};      // single start
        /* Its origin (CONFIG, WARM, a sampled point), when the spec keeps
         * them: an NLopt start that has to move warns for a user's seed. */
        if (spec.optim_init_origin.size() == spec.optim_init.size())
            sub.optim_init_origin = {spec.optim_init_origin[i]};
        else
            sub.optim_init_origin.clear();
    }
    if (plan.per_start_maxeval) sub.stop.maxeval = *plan.per_start_maxeval;
    /* Decorrelate the starts: reusing one seed hands every start the same
     * proposal stream, so stochastic optimisers (ANNEAL) produce
     * near-duplicate chains that differ only by their start point. NLopt
     * without a SEED takes the INIT block's (nlopt_seed_from_init()), which
     * is offset the same way. */
    if (spec.seed)
        sub.seed = *spec.seed + static_cast<std::uint64_t>(i);
    else if (spec.nlopt_seed_from_init())
        sub.init_strategy_spec.seed =
            *spec.init_strategy_spec.seed + static_cast<std::uint64_t>(i);
    return sub;
}

OptimizerSpec prestage_refinement_spec(
    const OptimizerSpec& spec, std::vector<std::vector<double>> candidates)
{
    OptimizerSpec refine = spec;
    refine.seed_seeds_exploration =
        spec.prestage && spec.prestage->seed_from_strategy;
    refine.prestage.reset();          // the refinement does not explore again
    refine.optim_init = std::move(candidates);
    /* Exploration candidates have no INIT origin: a search whose start
     * moves gives an INFO line, not the WARNING of a user's own point. */
    refine.optim_init_origin.clear();
    /* Streams of its own: with the exploration's optimiser it drew the
     * exploration's numbers for an equal seed (seed_stream.h). */
    refine.stream_phase = StreamPhase::Refinement;
    return refine;
}

std::string with_block_line(const OptimizerSpec& spec, const std::string& what)
{
    if (spec.block_line == 0) return what;
    return config_line_error(spec.block_line, what, spec.block_line_text);
}

std::string with_line(const OptimizerSpec& spec, const KeyLine& line,
                      const std::string& what)
{
    if (line.number == 0) return with_block_line(spec, what);
    return config_line_error(line.number, what, line.text);
}

std::string with_key_line(const OptimizerSpec& spec, const std::string& key,
                          const std::string& what)
{
    const auto it = spec.key_lines.find(key);
    return with_line(spec, it != spec.key_lines.end() ? it->second : KeyLine{},
                     what);
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
         * tolerances (optimize.c) and no evaluation limit of its own; the
         * block configures it, and its refusals name the block's line. */
        run.lib  = "NLOPT";
        run.algo = caps.default_inner;
        run.block_line = parent.block_line;
        run.block_line_text = parent.block_line_text;
        run.key_lines = parent.key_lines;
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
/* GN_DIRECT_L_RAND in one OPTIM dimension: cdirect.c draws only to choose
 * between equally long sides of a box (which_div 2), and a box of one
 * dimension has one side. */
bool direct_rand_in_one_dimension(const OptimizerSpec& spec)
{
    const OptimizerCaps* caps = find_optimizer_caps(spec.lib, spec.algo);
    return caps && seed_use(*caps) == SeedUse::ReadInTwoDimensions &&
           spec.low.size() == 1;
}
}  // namespace

bool reads_seed(const OptimizerSpec& spec)
{
    const OptimizerCaps* caps = find_optimizer_caps(spec.lib, spec.algo);
    if (!caps) return false;
    switch (seed_use(*caps)) {
        case SeedUse::Read:
            return true;
        case SeedUse::ReadInTwoDimensions:
            return !direct_rand_in_one_dimension(spec);
        case SeedUse::ThroughLocalSearch: {
            const std::optional<OptimizerSpec> local = inner_run_spec(spec);
            return local && reads_seed(*local);
        }
        case SeedUse::Refused:
            return false;
    }
    return false;
}

std::string seed_unread_reason(const OptimizerSpec& spec)
{
    if (direct_rand_in_one_dimension(spec))
        return "in one OPTIM dimension it draws no random numbers (it draws "
               "them only to choose between equally long sides of a box)";
    return "it draws no random numbers";
}

bool obtains_gradients(const OptimizerSpec& spec)
{
    /* MLSL and AUGLAG evaluate their own points without a gradient (mlsl.c,
     * auglag.c: grad NULL) and pass the objective the gradient their local
     * search asks for: the D or N of their name only picks NLopt's default
     * local optimiser, which an INNER block replaces. */
    if (!optimizer_caps(spec).accepts_inner)
        return optimizer_caps(spec).is_gradient_based;
    const std::optional<OptimizerSpec> local = inner_run_spec(spec);
    return local && obtains_gradients(*local);
}

std::optional<OptimizerSpec> single_local_search(const OptimizerSpec& spec)
{
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (caps.is_global) return std::nullopt;
    if (!caps.accepts_inner) return spec;
    const std::optional<OptimizerSpec> local = inner_run_spec(spec);
    if (!local) return std::nullopt;
    return single_local_search(*local);
}

std::optional<OptimizerSpec> search_ignoring_start(const OptimizerSpec& spec)
{
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (!caps.starts_at_x0) return spec;
    if (caps.accepts_inner && !caps.is_global) {
        const std::optional<OptimizerSpec> local = inner_run_spec(spec);
        if (local) return search_ignoring_start(*local);
    }
    return std::nullopt;
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
                                  *spec.prestage, /*inner=*/false,
                                  /*exploration=*/true});
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
                          run ? std::move(*run) : *phases[i].spec.inner,
                          /*inner=*/true});
    }
    return phases;
}

bool takes_fd_gradients(const PhaseSpec& phase)
{
    if (phase.inner || phase.spec.lib == "NOT SET" ||
        phase.spec.algo == "NOT SET")
        return false;
    if (phase.spec.grad_strategy.value_or(GradStrategy::Numerical) ==
        GradStrategy::Analytical)
        return false;
    return obtains_gradients(phase.spec);
}

}  // namespace tadah::hpo

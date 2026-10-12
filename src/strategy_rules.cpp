#include <tadah/hpo/strategy_rules.h>

#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah::hpo {

namespace {

constexpr const char* kNotSet = "NOT SET";

/* "LIB and ALGO are not set", "LIB is not set", "ALGO is not set"; "" when
 * both are set. */
std::string missing_driver(const OptimizerSpec& s)
{
    const bool no_lib  = s.lib  == kNotSet;
    const bool no_algo = s.algo == kNotSet;
    if (no_lib && no_algo) return "LIB and ALGO are not set";
    if (no_lib)            return "LIB is not set";
    if (no_algo)           return "ALGO is not set";
    return "";
}

std::string backend_name(const OptimizerSpec& s)
{
    return s.lib + ":" + s.algo;
}

/* The phases as validate_run_policy() names them: the EXPLORE block, and
 * the optimiser of `s` itself -- under PRESTAGE the refinement.
 * require_driver()'s refusal, validate_run_policy() and
 * reject_refused_backends() take them from here, so all three name a phase
 * alike. */
constexpr const char* kExploreBlock = "the EXPLORE block";

std::string run_phase_name(const OptimizerSpec& s)
{
    return s.strategy_type == StrategyType::PRESTAGE
               ? "the PRESTAGE refinement"
               : "the STRATEGY-level optimiser";
}

/* The backends runs_from_each_start() accepts, sorted (format_caps_list),
 * from the live registry rather than a hard-coded list. */
std::string multi_start_alternatives()
{
    return format_caps_list(filter_optimizer_caps(&runs_from_each_start));
}

/* The backends that run a local search without an INNER block
 * (OptimizerCaps::default_inner), every one, sorted. */
std::string with_default_inner()
{
    const auto caps = filter_optimizer_caps([](const OptimizerCaps& c) {
        return c.accepts_inner && c.default_inner != nullptr;
    });
    return format_caps_list(caps, caps.size());
}

/* NLopt's default local optimisers, "LD_MMA or LN_COBYLA", from the
 * registry. */
std::string default_inner_algorithms()
{
    std::vector<std::string> algos;
    for (const auto* c : filter_optimizer_caps([](const OptimizerCaps& k) {
             return k.default_inner != nullptr;
         }))
        if (std::find(algos.begin(), algos.end(), c->default_inner) ==
            algos.end())
            algos.push_back(c->default_inner);
    std::sort(algos.begin(), algos.end());
    std::string out;
    for (std::size_t i = 0; i < algos.size(); ++i) {
        if (i) out += (i + 1 == algos.size()) ? " or " : ", ";
        out += algos[i];
    }
    return out;
}

struct StrategyTypeKey {
    const char*  key;
    StrategyType home;
    const char*  effect;   // what the key does under its TYPE
};

/* The STRATEGY keys of one TYPE. */
constexpr StrategyTypeKey kStrategyTypeKeys[] = {
    {"N_STARTS",         StrategyType::MULTISTART,
     "only TYPE MULTISTART runs N_STARTS starts"},
    {"PARALLEL",         StrategyType::MULTISTART,
     "only TYPE MULTISTART has starts to run in parallel"},
    {"BUDGET_PER_START", StrategyType::MULTISTART,
     "only TYPE MULTISTART has starts to budget"},
    {"TOPK",             StrategyType::PRESTAGE,
     "only TYPE PRESTAGE forwards TOPK candidates to its refinement"},
};

const StrategyTypeKey* find_strategy_type_key(const std::string& key)
{
    for (const auto& k : kStrategyTypeKeys)
        if (key == k.key) return &k;
    return nullptr;
}

}  // namespace

const char* strategy_type_name(StrategyType type)
{
    switch (type) {
        case StrategyType::PLAIN:      return "PLAIN";
        case StrategyType::MULTISTART: return "MULTISTART";
        case StrategyType::PRESTAGE:   return "PRESTAGE";
    }
    return "?";
}

void require_driver(const OptimizerSpec& s, DriverPhase phase)
{
    const std::string missing = missing_driver(s);
    if (missing.empty()) {
        require_registered_backend(s.lib, s.algo,
                                   phase == DriverPhase::Explore
                                       ? std::string(kExploreBlock)
                                       : run_phase_name(s));
        return;
    }
    if (phase == DriverPhase::Explore)
        throw std::runtime_error(
            "the EXPLORE block names no optimiser: " + missing +
            "; set LIB and ALGO inside the EXPLORE block (the exploration "
            "phase of STRATEGY TYPE PRESTAGE)");
    switch (s.strategy_type) {
        case StrategyType::MULTISTART:
            throw std::runtime_error(
                "STRATEGY TYPE MULTISTART names no optimiser to run from each "
                "start: " + missing + "; set LIB and ALGO in the STRATEGY "
                "block");
        case StrategyType::PRESTAGE:
            throw std::runtime_error(
                "STRATEGY TYPE PRESTAGE names no refinement optimiser: " +
                missing + "; set LIB and ALGO in the STRATEGY block or in a "
                "REFINE block");
        case StrategyType::PLAIN:
            break;
    }
    throw std::runtime_error(
        "the OPTIMIZER block names no optimiser: " + missing + "; set LIB and "
        "ALGO in its STRATEGY block (or directly inside OPTIMIZER)");
}

void require_inner_driver(const OptimizerSpec& s, const std::string& enclosing)
{
    const std::string missing = missing_driver(s);
    if (missing.empty()) {
        require_registered_backend(s.lib, s.algo,
                                   "the INNER block inside " + enclosing);
        return;
    }
    throw std::runtime_error(
        "the INNER block inside " + enclosing + " names no optimiser: " +
        missing + "; set LIB and ALGO inside the INNER block (the local "
        "optimiser that NLopt's MLSL and AUGLAG run)");
}

void require_inner_backend(const OptimizerSpec& parent,
                           const std::string& phase)
{
    const std::string name = backend_name(parent);
    /* NLopt refuses MLSL or AUGLAG without a local optimiser when it has no
     * default one ("local optimizer must be specified for AUGLAG",
     * optimize.c). An unregistered LIB:ALGO is validate_backend()'s to
     * report (find_optimizer_caps). */
    const OptimizerCaps* caps = find_optimizer_caps(parent.lib, parent.algo);
    if (caps && caps->accepts_inner && !caps->default_inner && !parent.inner)
        throw std::runtime_error(
            name + " needs an INNER block" +
            (phase.empty() ? "" : " (" + phase + ")") +
            ": NLopt has no default local optimiser for it; add INNER ... "
            "ENDINNER with LIB NLOPT and a local ALGO such as LN_BOBYQA, or "
            "use one of " + with_default_inner() + ", which run " +
            default_inner_algorithms() + " without one");
    if (!parent.inner) return;
    const OptimizerSpec& inner = *parent.inner;
    /* A refused INNER optimiser, with the INNER block's phase: NLopt would
     * run it inside the enclosing run (build_nlopt). */
    reject_refused_backend(inner.lib, inner.algo,
                           "the INNER block of " +
                               (phase.empty() ? name : phase));
    if (!optimizer_caps(parent).accepts_inner) {
        const std::string where =
            phase.empty() ? "" : " (the INNER block of " + phase + ")";
        throw std::runtime_error(
            name + " does not accept an INNER sub-optimiser" + where +
            ": only NLopt's MLSL and AUGLAG algorithms run a local search "
            "inside their run (" +
            format_caps_list(filter_optimizer_caps(
                [](const OptimizerCaps& c) { return c.accepts_inner; })) +
            "); remove the INNER block or pick one of them");
    }
    if (inner.lib != "NLOPT")
        throw std::runtime_error(
            "the INNER block of " + (phase.empty() ? name : phase) +
            " names " + backend_name(inner) + ", but " + name + " runs its "
            "INNER search with NLopt: set LIB NLOPT and an NLopt ALGO (such "
            "as LN_BOBYQA) inside the INNER block");
    /* An INNER search that itself needs an INNER block: the grammar has no
     * INNER inside INNER, and NLopt refused it inside the enclosing run
     * with only "nlopt invalid argument". */
    const OptimizerCaps* inner_caps = find_optimizer_caps(inner.lib, inner.algo);
    if (inner_caps && inner_caps->accepts_inner && !inner_caps->default_inner)
        throw std::runtime_error(
            "the INNER block of " + (phase.empty() ? name : phase) +
            " names " + backend_name(inner) + ", which needs an INNER block "
            "of its own; INNER blocks do not nest: pick a local algorithm "
            "such as LN_BOBYQA");
}

bool runs_from_each_start(const OptimizerCaps& caps)
{
    return !caps.is_global || caps.accepts_multi_seed;
}

void require_multistart_backend(const OptimizerSpec& spec)
{
    require_driver(spec, DriverPhase::Run);
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (caps.accepts_multi_seed && spec.multistart.budget_per_start)
        throw std::runtime_error(
            "BUDGET_PER_START has no effect with " + backend_name(spec) +
            ", which runs every start in one optimiser; use MAXEVAL");
    if (runs_from_each_start(caps)) return;
    const std::string name = backend_name(spec);
    throw std::runtime_error(
        "STRATEGY TYPE MULTISTART cannot run " + name + ": it is a global "
        "optimiser that consumes a single starting point, so restarting it "
        "from each INIT point is rejected for any N_STARTS. Use a local or "
        "multi-seed optimiser (" + multi_start_alternatives() + "), or "
        "STRATEGY TYPE PLAIN to run " + name + " once.");
}

void require_prestage_explore(const OptimizerSpec& spec)
{
    if (!spec.prestage)
        throw std::runtime_error(
            "STRATEGY TYPE PRESTAGE requires an EXPLORE ... ENDEXPLORE block "
            "inside the STRATEGY block (it defines the exploration phase); "
            "the refinement optimiser is the STRATEGY-level LIB/ALGO or an "
            "optional REFINE ... ENDREFINE block.");
    require_driver(*spec.prestage, DriverPhase::Explore);
}

void require_prestage_refinement(const OptimizerSpec& spec)
{
    require_driver(spec, DriverPhase::Run);
    if (spec.stage_topk <= 1) return;
    if (runs_from_each_start(optimizer_caps(spec))) return;
    const std::string name = backend_name(spec);
    const std::string topk = std::to_string(spec.stage_topk);
    throw std::runtime_error(
        "STRATEGY TYPE PRESTAGE with TOPK " + topk + " cannot refine with " +
        name + ": the refinement runs its optimiser once from each of the "
        "TOPK candidates the exploration forwards, and " + name + " is a "
        "global optimiser that consumes a single starting point. Use a local "
        "or multi-seed refinement optimiser (" + multi_start_alternatives() +
        "), or TOPK 1 to refine the best candidate once.");
}

void validate_run_policy(const OptimizerSpec& spec)
{
    /* Each refusal names the line of the block it is about, as the
     * parser's refusals do (with_block_line, phase_specs.h): the
     * STRATEGY-level optimiser's, an INNER block's -- the enclosing
     * block's when it has none. A spec built outside the parser has no
     * line. */
    const auto at = [](const OptimizerSpec& block, const auto& check) {
        try {
            check();
        } catch (const std::exception& e) {
            throw std::runtime_error(with_block_line(block, e.what()));
        }
    };
    at(spec, [&] {
        switch (spec.strategy_type) {
            case StrategyType::MULTISTART:
                require_multistart_backend(spec);
                break;
            case StrategyType::PRESTAGE:
                require_prestage_explore(spec);
                require_prestage_refinement(spec);
                break;
            case StrategyType::PLAIN:
                require_driver(spec, DriverPhase::Run);
                break;
        }
    });
    at(spec.inner ? *spec.inner : spec,
       [&] { require_inner_backend(spec, run_phase_name(spec)); });
    if (spec.strategy_type == StrategyType::PRESTAGE) {
        const OptimizerSpec& explore = *spec.prestage;
        at(explore.inner ? *explore.inner : explore,
           [&] { require_inner_backend(explore, kExploreBlock); });
    }
}

void reject_refused_backends(const OptimizerSpec& spec)
{
    /* validate_run_policy()'s order: the exploration, the STRATEGY-level
     * optimiser, its INNER block, the exploration's INNER block. A block
     * without LIB or ALGO ("NOT SET") is require_driver()'s to report. */
    const auto check = [](const OptimizerSpec& s, const std::string& where) {
        if (!missing_driver(s).empty()) return;
        require_registered_backend(s.lib, s.algo, where);
    };
    const OptimizerSpec* explore =
        spec.strategy_type == StrategyType::PRESTAGE ? spec.prestage.get()
                                                     : nullptr;
    const std::string run = run_phase_name(spec);
    if (explore) check(*explore, kExploreBlock);
    check(spec, run);
    if (spec.inner) check(*spec.inner, "the INNER block of " + run);
    if (explore && explore->inner)
        check(*explore->inner,
              std::string("the INNER block of ") + kExploreBlock);
}

bool is_strategy_type_key(const std::string& key)
{
    return find_strategy_type_key(key) != nullptr;
}

void require_strategy_key(const std::string& key, StrategyType type,
                          bool type_given)
{
    const StrategyTypeKey* k = find_strategy_type_key(key);
    if (!k || k->home == type) return;
    const std::string home = strategy_type_name(k->home);
    throw std::runtime_error(
        key + " is not accepted with STRATEGY TYPE " +
        strategy_type_name(type) +
        (type_given ? "" : " (the default when the block has no TYPE line)") +
        ": " + k->effect + ", so it would have no effect; set TYPE " + home +
        " or remove " + key);
}

void require_prestage_block(const std::string& block, StrategyType type)
{
    if (type == StrategyType::PRESTAGE) return;
    throw std::runtime_error(
        block + " block is only valid with TYPE PRESTAGE, but the STRATEGY "
        "block ends with TYPE " + strategy_type_name(type) + " (the last TYPE "
        "line decides); set TYPE PRESTAGE or remove the " + block + " block");
}

}  // namespace tadah::hpo

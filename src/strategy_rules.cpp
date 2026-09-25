#include <tadah/hpo/strategy_rules.h>

#include <tadah/hpo/optimizer_caps.h>

#include <stdexcept>
#include <string>

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

/* The backends runs_from_each_start() accepts, sorted (format_caps_list),
 * from the live registry rather than a hard-coded list. */
std::string multi_start_alternatives()
{
    return format_caps_list(filter_optimizer_caps(&runs_from_each_start));
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
    if (missing.empty()) return;
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
    if (missing.empty()) return;
    throw std::runtime_error(
        "the INNER block inside " + enclosing + " names no optimiser: " +
        missing + "; set LIB and ALGO inside the INNER block (the local "
        "optimiser that NLopt's MLSL and AUGLAG run)");
}

void require_inner_backend(const OptimizerSpec& parent,
                           const std::string& phase)
{
    if (!parent.inner) return;
    const OptimizerSpec& inner = *parent.inner;
    const std::string name = backend_name(parent);
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
    const bool prestage = spec.strategy_type == StrategyType::PRESTAGE;
    require_inner_backend(spec, prestage ? "the PRESTAGE refinement"
                                         : "the STRATEGY-level optimiser");
    if (prestage) require_inner_backend(*spec.prestage, "the EXPLORE block");
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

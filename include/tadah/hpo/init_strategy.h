#pragma once
/*
 * InitStrategy -- produces a list of starting points (in real, untransformed
 * space) for the optimiser, given bounds, the OPTIM map and a seed.
 *
 * Concrete strategies live in init_strategies.cpp. A free factory
 * build_init_strategy() walks the parsed InitStrategySpec and returns
 * one polymorphic InitStrategy that may be a Chain over several
 * producers.
 *
 * Numerical contract:
 *   - Sampling happens in the transformed (post analyse_logscale) space
 *     using spec.low, spec.high, spec.log_scaled.
 *   - Output points are returned in real space.
 *   - log_scaled[d] == false => dim d is sampled linearly between low[d]
 *     and high[d] (already real-valued).
 *   - log_scaled[d] == true  => dim d is sampled in log-space (low[d] and
 *     high[d] are already log-transformed by analyse_logscale()) and
 *     exponentiated before being returned.
 */

#include <tadah/hpo/optimizer_spec.h>   // InitStrategySpec::Entry

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tadah::core { class Context; }

namespace tadah::hpo {

class HPOTargets;        // forward-declared -- full type pulled in by .cpp
struct OptimizerSpec;

struct InitContext {
    const OptimizerSpec& spec;            // bounds, log_scaled (post-transform)
    const HPOTargets&    targets;         // optimap, ui_offset
    std::uint64_t        seed;            // resolved INIT_SEED or random_device
    /* The run's -c configuration (HPOTargets' context), for WARM's EMBF0
     * check; nullptr skips the check (e.g. RESTART re-seeding). */
    const core::Context* run_context = nullptr;
};

class InitStrategy {
public:
    virtual ~InitStrategy() = default;

    /** Produce up to k starting points. A strategy may return fewer
     *  (e.g. CONFIG always returns 1). The caller is responsible for
     *  truncating / padding to its budget. */
    virtual std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k) const = 0;

    /** Human-readable name for logs (matches the parser keyword). */
    virtual std::string name() const = 0;

    /** Natural number of points this strategy produces when no explicit
     *  K is given. Used by build_init_strategy() to size INIT_K. */
    virtual std::size_t default_k() const { return 1; }
};

/* Build the polymorphic InitStrategy from the parsed spec.
 *
 *   - Walks ispec.chain in order, instantiating each entry. The chain is
 *     never reordered -- entries run exactly as the user wrote them.
 *   - The in-config OPTIM point is part of the chain only when the user
 *     listed `STRATEGY CONFIG` explicitly. There is no implicit CONFIG.
 *   - If the chain is empty (no INIT block, or one with no STRATEGY
 *     lines) a single CONFIG point is used as a safe default.
 */
std::unique_ptr<InitStrategy> build_init_strategy(const struct InitStrategySpec& ispec);

/* Checks one chain entry the way build_init_strategy() instantiates it:
 * a known METHOD, its arity, and K a whole integer, >= 0 for RANDOM and
 * >= 1 for LHS/SOBOL. HPOTargets::read() calls it at the METHOD line, so
 * the error names that line. Throws std::runtime_error. */
void validate_init_entry(const InitStrategySpec::Entry& e);

}  // namespace tadah::hpo

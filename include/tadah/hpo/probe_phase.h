#pragma once
/*
 * probe_phase -- an OFF-BUDGET probe phase.
 *
 * Some work has to evaluate the real objective without being part of the
 * search: the INIT_MODE JITTER AUTO calibration measures a per-dimension
 * displacement before the optimiser starts. Those evaluations are real (they
 * cost LAMMPS time, they can find a better point) but they are not steps of
 * the optimisation, and treating them as such breaks five different things at
 * once. This flag is the single switch the host consults, so the list below
 * is enforced in one place instead of five.
 *
 * While a probe phase is active:
 *
 *   1. STAGNATION IS NOT COUNTED. HPO_Host::operator() throws StagnationStop
 *      when g_iter - iter_at_last_best exceeds the threshold, and the
 *      annealers run INSIDE RestartPolicy, which sets that threshold. A
 *      calibration is hundreds of deliberately non-improving evaluations in a
 *      row, so without this it would trip the threshold and fire a restart
 *      partway through. (NOISE_CALIBRATION does not need this: it runs before
 *      build_run_policy(), outside the window by construction. It establishes
 *      MAXEVAL exemption only -- do not cite it as precedent here.)
 *
 *   2. CHECKPOINT TICKS ARE SUPPRESSED. Every eval calls
 *      CheckpointManager::tick_eval, so mid-calibration checkpoints would be
 *      written with g_iter advanced and zero annealer state behind it.
 *
 *   3. WARM STARTS ARE NAMESPACED. Probes must not pollute a chain's
 *      warm-start lineage -- warm-start selection bias was the root cause of
 *      the run15/16/17 artefact. Note this is done WITHOUT touching
 *      warmstart::current_chain(): that value becomes EvalResult::chain_id and
 *      from there the SLOT column (fixed width), the durable-best comparison
 *      key, `# HPO_SOURCE_SLOT` and the resume key. Offsetting it by a large
 *      constant -- the obvious trick -- corrupts all four.
 *
 * The budget itself is NOT handled here: the drivers give the phase its own
 * anneal::SharedBudget, so "does not consume MAXEVAL" holds by construction
 * rather than by a flag anyone can forget to set.
 */

#include <atomic>
#include <cstddef>

namespace tadah {
namespace hpo {
namespace probe {

/* Process-wide on purpose: a probe phase is a driver-level bracket around a
 * fan-out over many pool threads, and no production eval overlaps it.
 *
 * Atomic because the driver writes it while pool workers read it. The pool's
 * own dispatch/join already orders those against the phase boundaries, so
 * relaxed is enough -- the atomic is here to make the concurrent access
 * well-defined rather than to establish ordering. */
inline std::atomic<bool>& active_flag()
{
    static std::atomic<bool> active{false};
    return active;
}

inline bool active() { return active_flag().load(std::memory_order_relaxed); }

/* Warm-start keys for probe evaluations live above every real chain index, so
 * a probe can never be mistaken for a chain's cached hot variable. Used only
 * as a cache key -- it never reaches chain_id (see the header note). */
inline constexpr std::size_t kWarmstartBase = std::size_t{1} << 40;

/* RAII bracket. Nested scopes are tolerated (the inner one is a no-op) so a
 * phase can call a helper that also brackets itself. */
class Scope {
public:
    Scope() : owned_(!active()) {
        if (owned_) active_flag().store(true, std::memory_order_relaxed);
    }
    ~Scope() {
        if (owned_) active_flag().store(false, std::memory_order_relaxed);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    bool owned_;
};

}  // namespace probe
}  // namespace hpo
}  // namespace tadah

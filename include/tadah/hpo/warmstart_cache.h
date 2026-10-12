#pragma once
/*
 * Per-(chain, script, hot-var) warm-start cache.
 *
 * Heavy LAMMPS scripts (e.g. computing elastic constants from a relaxed
 * BCC bulk) start with a box/relax minimisation from some lattice guess.
 * After one HPO evaluation the relaxed lattice constant is known --
 * passing it back as the next eval's input lets the relax start near
 * the converged state, reducing the minimise from ~340 iter to ~3-5.
 *
 * The user opts in by adding `--hotvar <name> <min> <max> [<initial>]`
 * to a PC_LAMMPS line. The framework then:
 *   - injects `variable <name> equal <cached-or-initial>` before the
 *     hot script runs (so it can use ${<name>} as the seed);
 *   - reads <name> back after the script runs;
 *   - validates <name> against [min, max] (an out-of-range value is
 *     treated as an unphysical-potential signal -- the eval is failscored
 *     and cold scripts are skipped, via a run-local abort flag);
 *   - caches the in-range value here for the next eval AND propagates
 *     it as `${<name>}` to the cold scripts that read it in the *same*
 *     eval.
 *
 * WHY PER-CHAIN (the determinism fix, 2026-07-15)
 * -----------------------------------------------
 * ANNEAL_PT / ANNEAL_PA run K chains (replicas / walkers) concurrently.
 * If the warm-start slot were keyed by (script, name) only, all K chains
 * would share ONE box/relax seed, and last-writer-wins across concurrently
 * running chains is execution-order dependent -- so any change to fan-out
 * order (the shared ScriptExecutor, or LPT scheduling) would perturb the
 * seed and thus the (chaotic, ppm-sensitive) relaxation basin, silently
 * changing results. Keying the runtime cache by the chain's GLOBAL slot
 * index makes each chain warm-start from its OWN previous converged value,
 * so results are independent of which worker runs which script when.
 *
 * Three maps:
 *   - `chain_cache`    : (chain, script, name) -> value.  The live per-chain
 *     warm seed, written by run_lammps after each successful hot script.
 *   - `warm_prime_map` : (script, name) -> value.  The FALLBACK for a chain's
 *     first eval (no chain entry yet). Written ONLY by WARM priming, before
 *     the run -- never during it -- so the fallback is a DETERMINISTIC value,
 *     never a value another chain is concurrently promoting.
 *   - `best_hot_map`   : (script, name) -> value.  The persistence layer:
 *     primed by WARM, updated at the serialized new-global-best point, dumped
 *     into best_pot's `# HOT` lines. Read ONLY at dump time / by the gradient
 *     path -- NEVER as an eval fallback -- so its updates can't make a running
 *     eval order-dependent. Keeping fallback (static) and dump (dynamic)
 *     separate is what closes the init-epoch promote-vs-fallback race.
 *
 * The current chain is bridged via a thread_local: the annealing drivers
 * set `current_chain()` to the global slot before each chain's rung;
 * operator() copies it into EvalResult::chain_id; run_lammps keys the
 * cache by r.chain_id. Serial and single-sequence optimisers leave it at
 * the default 0 (one implicit chain), so their behaviour is unchanged.
 *
 * Why per-(script, name) within a chain and not per-name globally: two hot
 * scripts may each declare `--hotvar a0 ...` and converge to different a0
 * values depending on their own minimiser tolerances / fix box/relax flags.
 * Each script gets its own warm-start slot so a different basin in one
 * doesn't poison the seed of the other.
 *
 * The WARM init strategy (init_strategies.cpp) primes both (script, name) maps
 * from `# HOT <script> <name> <value>` lines saved into best_pot.tadah, so a
 * cold restart picks up exactly where the previous run's best converged to.
 *
 * No environment variables -- all knobs (ranges, initial, what's hot)
 * are declared per PC_LAMMPS line in config.hpo.
 */

#include <atomic>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/probe_phase.h>

namespace tadah::hpo::warmstart {

/* Global slot of the chain whose eval is running on THIS thread. Set by the
 * parallel annealing drivers before each chain's rung (PT/PA); read by
 * HPO_Host::operator() into EvalResult::chain_id. Default 0 = the single
 * implicit chain of serial anneal / NLOPT / dlib / gradient optimisers. */
inline std::size_t& current_chain() {
    static thread_local std::size_t c = 0;
    return c;
}

/* (chain, script, varname) -> cached value. */
struct Key {
    std::size_t chain;
    std::string script;
    std::string name;
    bool operator==(const Key& o) const noexcept {
        return chain == o.chain && script == o.script && name == o.name;
    }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        std::size_t h = std::hash<std::size_t>{}(k.chain) * 1099511628211ull;
        h ^= std::hash<std::string>{}(k.script) * 1315423911u;
        h ^= std::hash<std::string>{}(k.name);
        return h;
    }
};
/* (script, varname) -> value, for the persistence/fallback seed map. */
struct SeedKey {
    std::string script;
    std::string name;
    bool operator==(const SeedKey& o) const noexcept {
        return script == o.script && name == o.name;
    }
};
struct SeedKeyHash {
    std::size_t operator()(const SeedKey& k) const noexcept {
        return std::hash<std::string>{}(k.script) * 1315423911u
             ^ std::hash<std::string>{}(k.name);
    }
};

inline std::mutex& mtx() { static std::mutex m; return m; }
inline std::unordered_map<Key, double, KeyHash>& chain_cache() {
    static std::unordered_map<Key, double, KeyHash> m;
    return m;
}
/* Fallback seed for a chain's FIRST eval (no chain entry yet). Written ONLY
 * by WARM priming, before the run -- never during it -- so the fallback is
 * static and a chain's first eval reads a DETERMINISTIC value, never a value
 * another chain is concurrently promoting. This is the determinism-critical
 * separation from best_hot below. */
inline std::unordered_map<SeedKey, double, SeedKeyHash>& warm_prime_map() {
    static std::unordered_map<SeedKey, double, SeedKeyHash> m;
    return m;
}
/* Best-so-far hot values, dumped into best_pot.tadah's `# HOT` lines. Written
 * by WARM priming AND at the serialized new-global-best point; read ONLY at
 * dump time and by the (single-threaded) gradient path -- NEVER as an eval
 * fallback -- so its updates cannot make a running eval order-dependent. */
inline std::unordered_map<SeedKey, double, SeedKeyHash>& best_hot_map() {
    static std::unordered_map<SeedKey, double, SeedKeyHash> m;
    return m;
}
/* Compound identity of the batch applied to best_hot_map. It uses exactly the
 * same loss/rank/slot/eval ordering as durable best-potential publication, so
 * an exact-loss tie cannot leave the HOT snapshot and potential file pointing
 * at different candidates. Protected by mtx(). */
inline std::optional<durable_best::Key>& best_hot_key() {
    static std::optional<durable_best::Key> key;
    return key;
}

/* Lookup for the given chain: the chain's own value if present, else the
 * STATIC warm-prime fallback, else nullopt (caller then uses the decl's
 * `initial`). Never consults best_hot, so a concurrent new-best promotion
 * cannot perturb a first eval. */
/* Cache key for the calling context. During an off-budget probe phase the
 * probes get their own key space above every real chain index, so a probe's
 * converged hot variable can never be handed to a chain as if the chain had
 * produced it (warm-start selection bias was the run15/16/17 artefact).
 *
 * This offsets the CACHE KEY only. current_chain() itself is untouched: it
 * becomes EvalResult::chain_id and from there the fixed-width SLOT column,
 * the durable-best comparison key and `# HPO_SOURCE_SLOT`. */
inline std::size_t cache_chain(std::size_t chain) {
    return probe::active() ? probe::kWarmstartBase + chain : chain;
}

inline std::optional<double> get(std::size_t chain,
                                 const std::string& script,
                                 const std::string& name) {
    chain = cache_chain(chain);
    std::lock_guard<std::mutex> lg(mtx());
    auto it = chain_cache().find({chain, script, name});
    if (it != chain_cache().end()) return it->second;
    auto st = warm_prime_map().find({script, name});
    if (st != warm_prime_map().end()) return st->second;
    return std::nullopt;
}

/* Store this chain's freshly converged value. Range-check is the caller's
 * responsibility (the framework holds the per-decl [min, max]). */
inline void set(std::size_t chain,
                const std::string& script,
                const std::string& name,
                double value) {
    chain = cache_chain(chain);
    std::lock_guard<std::mutex> lg(mtx());
    chain_cache()[{chain, script, name}] = value;
}

/* Single-entry helper for setup/tests. Concurrent new-best publication must
 * use promote_batch() so a delayed older worker cannot roll the map back.
 * Does NOT touch warm_prime_map (the eval fallback stays static). */
inline void promote(const std::string& script,
                    const std::string& name,
                    double value) {
    std::lock_guard<std::mutex> lg(mtx());
    best_hot_map()[{script, name}] = value;
}

/* Atomically replace the HOT snapshot with every value produced by one
 * completed compound-best evaluation. Clearing first is important: a
 * candidate that legitimately omits a HOT value must not inherit that value
 * from an older potential. Returns false for a stale batch. */
inline bool promote_batch(
    const durable_best::Key& key,
    const std::vector<std::tuple<std::string, std::string, double>>& values) {
    std::lock_guard<std::mutex> lg(mtx());
    if (!std::isfinite(key.loss) || key.rank < 0) return false;
    if (best_hot_key() && !durable_best::precedes(key, *best_hot_key()))
        return false;
    best_hot_map().clear();
    for (const auto& value : values) {
        best_hot_map()[{std::get<0>(value), std::get<1>(value)}] =
            std::get<2>(value);
    }
    best_hot_key() = key;
    return true;
}

/* Best hot value for (script, name) -- used by the single-threaded analytical
 * gradient path (no chain context). */
inline std::optional<double> best_get(const std::string& script,
                                      const std::string& name) {
    std::lock_guard<std::mutex> lg(mtx());
    auto st = best_hot_map().find({script, name});
    if (st != best_hot_map().end()) return st->second;
    return std::nullopt;
}

/* WARM priming (from best_pot.tadah's `# HOT` lines). Seeds BOTH the static
 * fallback (so restarted chains start where the best converged) AND the best
 * map (so a subsequent dump preserves the values if nothing beats them). */
inline void prime(const std::string& script,
                  const std::string& name,
                  double value) {
    std::lock_guard<std::mutex> lg(mtx());
    warm_prime_map()[{script, name}] = value;
    best_hot_map()[{script, name}]   = value;
    best_hot_key().reset();
}

/* Reset ALL warm-start state (per-chain cache + both seed maps). Primarily a
 * test-isolation helper; also safe to call at a hard run boundary. */
inline void clear() {
    std::lock_guard<std::mutex> lg(mtx());
    chain_cache().clear();
    warm_prime_map().clear();
    best_hot_map().clear();
    best_hot_key().reset();
}

/* Start a run of hpo_run(), before its WARM init strategy primes the seed
 * maps (hpo_run calls it before HPOTargets is built): forget everything an
 * earlier run in this process left -- every chain's value, WARM's static
 * fallback, the best snapshot. warm_prime_map is otherwise written only by
 * WARM priming and read by a chain's first evaluation, so a run without
 * WARM after one with a WARM seed of HOT lines started its --hotvar values
 * from that earlier run's seed, not from their initial values. */
inline void begin_run() {
    clear();
}

/* Start a new run: forget every chain's value an earlier run in this process
 * converged to (chain_cache) and the best snapshot, keep WARM's static
 * fallback. A chain's first evaluation then reads warm_prime_map (WARM) or
 * nothing (its --hotvar initial value), as in a run of its own process: the
 * cache is keyed by (chain, script, name), so a second run of a task file
 * with the same script read the first run's last relaxed value instead.
 * HPO_Host::prepare_best_publication() calls it at the start of every run;
 * WARM primes warm_prime_map before (and begin_run() cleared it before
 * that). */
inline void reset_for_fresh_run() {
    std::lock_guard<std::mutex> lg(mtx());
    chain_cache().clear();
    best_hot_map().clear();
    best_hot_key().reset();
}

/* Start a new output generation without discarding WARM's static fallback.
 * HPOTargets parses/loads WARM before the run binds its best-potential
 * publication; the fresh run must forget the prior best snapshot/key while
 * keeping warm_prime_map available to seed the first evaluation. */
inline void reset_best_snapshot_for_fresh_run() {
    std::lock_guard<std::mutex> lg(mtx());
    best_hot_map().clear();
    best_hot_key().reset();
}

/* Persisted cache format used inside best_pot.tadah:
 *   # HOT <script> <name> <value>
 * Dumps the SEED map (the run's best hot values). The deserializer strips
 * '#'-prefixed lines, so this is invisible to everything that isn't
 * WarmInitStrategy. Full numerical precision so the round-trip is bit-exact. */
inline void dump_to_stream(std::ostream& os) {
    std::lock_guard<std::mutex> lg(mtx());
    const auto& m = best_hot_map();
    if (m.empty()) return;
    const auto old_prec = os.precision();
    const auto old_flags = os.flags();
    os << std::scientific << std::setprecision(17);
    for (const auto& kv : m)
        os << "# HOT " << kv.first.script << " " << kv.first.name
           << " " << kv.second << "\n";
    os.flags(old_flags);
    os.precision(old_prec);
}

/* Parse one line that may carry a HOT record into the SEED map. Returns true
 * only when the line is a well-formed `# HOT <script> <name> <value>`.
 * Validation against the per-decl [min, max] happens later when the script's
 * run actually fires; here we just round-trip the seed.
 *
 * Also accepts the legacy `# HOT_A0 <script> <a0>` format for back-compat
 * with best_pot.tadah files written by older versions, mapping it to
 * (script, "hot_a0") since that was the only name in use then. */
inline bool try_parse_line(const std::string& line) {
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] != '#') return false;
    std::istringstream iss(line.substr(first + 1));
    std::string tag;
    if (!(iss >> tag)) return false;
    if (tag == "HOT") {
        std::string script, name;
        double v;
        if (!(iss >> script >> name >> v)) return false;
        if (!std::isfinite(v)) return false;
        prime(script, name, v);
        return true;
    }
    if (tag == "HOT_A0") {
        /* Legacy: HOT_A0 <script> <a0>. Old configs only ever varied a0
         * under the fixed name "hot_a0". */
        std::string script;
        double v;
        if (!(iss >> script >> v)) return false;
        if (!std::isfinite(v)) return false;
        prime(script, "hot_a0", v);
        return true;
    }
    return false;
}

} // namespace tadah::hpo::warmstart

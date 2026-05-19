#pragma once
/*
 * Per-(script, hot-var) warm-start cache shared across HPO evaluations.
 *
 * Heavy LAMMPS scripts (e.g. computing elastic constants from a relaxed
 * BCC bulk) start with a box/relax minimisation from some lattice guess.
 * After one HPO evaluation the relaxed lattice constant is known --
 * passing it back as the next eval's input lets the relax start near
 * the converged state, reducing the minimise from ~340 iter to ~3-5.
 *
 * The user opts in by adding `--hotvar <name> <min> <max> [<initial>]`
 * to a PC_LAMMPS line. The framework then:
 *   - injects `variable <name> string <cached-or-initial>` before the
 *     hot script runs (so it can use ${<name>} as the seed);
 *   - reads <name> back after the script runs;
 *   - validates <name> against [min, max] (an out-of-range value is
 *     treated as an unphysical-potential signal -- the iter is failscored
 *     and cold scripts are skipped);
 *   - caches the in-range value here for the next eval AND propagates
 *     it as `${<name>}` to every cold script in the *same* eval.
 *
 * Why per-(script, name) and not per-name globally: two hot scripts may
 * each declare `--hotvar a0 ...` and converge to different a0 values
 * depending on their own minimiser tolerances / fix box/relax flags.
 * Each script gets its own warm-start slot so a different basin in one
 * doesn't poison the seed of the other.
 *
 * The WARM init strategy (init_strategies.cpp) primes this cache from
 * `# HOT <script> <name> <value>` lines saved into best_pot.tadah,
 * so a cold restart picks up exactly where the previous run's best
 * converged to. Without that priming, the loss recorded in a previous
 * run's best_pot is not reproducible on cold restart: every script
 * starts from `initial` and may land box/relax in a different basin.
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
#include <unordered_map>

namespace tadah::hpo::warmstart {

/* (script, varname) -> cached value. */
struct Key {
    std::string script;
    std::string name;
    bool operator==(const Key& o) const noexcept {
        return script == o.script && name == o.name;
    }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        return std::hash<std::string>{}(k.script) * 1315423911u
             ^ std::hash<std::string>{}(k.name);
    }
};

inline std::mutex& mtx() { static std::mutex m; return m; }
inline std::unordered_map<Key, double, KeyHash>& cache() {
    static std::unordered_map<Key, double, KeyHash> m;
    return m;
}

/* Lookup; returns nullopt if (script, name) has never been cached. */
inline std::optional<double> get(const std::string& script,
                                 const std::string& name) {
    std::lock_guard<std::mutex> lg(mtx());
    auto it = cache().find({script, name});
    if (it == cache().end()) return std::nullopt;
    return it->second;
}

/* Store; range-check is the caller's responsibility (the framework holds
 * the per-decl [min, max] in LammpsSim::HotVarDecl). */
inline void set(const std::string& script,
                const std::string& name,
                double value) {
    std::lock_guard<std::mutex> lg(mtx());
    cache()[{script, name}] = value;
}

/* Counters used by the abort path in run_lammps. `aborting_iter` is set
 * by the first hot-script thread to detect an out-of-range value; all
 * other in-flight threads check it and short-circuit. */
inline std::atomic<std::size_t>& reject_count() {
    static std::atomic<std::size_t> c{0};
    return c;
}
inline std::atomic<std::size_t>& last_rejected_iter() {
    static std::atomic<std::size_t> i{0};
    return i;
}
inline std::atomic<std::size_t>& aborting_iter() {
    static std::atomic<std::size_t> i{0};
    return i;
}

/* Persisted cache format used inside best_pot.tadah:
 *   # HOT <script> <name> <value>
 * The deserializer strips '#'-prefixed lines, so this is invisible to
 * everything that isn't WarmInitStrategy. One line per (script, var),
 * full numerical precision so the round-trip is bit-exact. */
inline void dump_to_stream(std::ostream& os) {
    std::lock_guard<std::mutex> lg(mtx());
    const auto& m = cache();
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

/* Parse one line that may carry a HOT record. Returns true only when the
 * line is a well-formed `# HOT <script> <name> <value>`. Validation
 * against the per-decl [min, max] happens later when the script's run
 * actually fires; here we just round-trip the cache.
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
        set(script, name, v);
        return true;
    }
    if (tag == "HOT_A0") {
        /* Legacy: HOT_A0 <script> <a0>. Old configs only ever varied a0
         * under the fixed name "hot_a0". */
        std::string script;
        double v;
        if (!(iss >> script >> v)) return false;
        if (!std::isfinite(v)) return false;
        set(script, "hot_a0", v);
        return true;
    }
    return false;
}

} // namespace tadah::hpo::warmstart

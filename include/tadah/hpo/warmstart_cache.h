#pragma once
/*
 * Per-script lattice warm-start cache shared across HPO evaluations.
 *
 * Heavy LAMMPS scripts (e.g. computing elastic constants) start with a
 * box/relax minimisation from a hard-coded lattice guess. After one HPO
 * evaluation we know the relaxed lattice constant; passing that back as
 * `${hot_a0}` for the next eval lets the script start near the converged
 * state, reducing the minimise from ~340 iter to ~3–5 iter.
 *
 * Configurable via env vars (in Angstroms):
 *   TADAH_HPO_HOT_A0_DEFAULT  initial guess when cache is empty (default 3.2)
 *   TADAH_HPO_A0_MIN          physical lower bound (default 1.0)
 *   TADAH_HPO_A0_MAX          physical upper bound (default 10.0)
 * A relaxed a0 outside [MIN, MAX] is treated as a sign of an unphysical
 * potential: the corresponding script's outputs are clamped to failscore
 * and the cache is not updated. Tightening the range (e.g. 2.5/4.5 for Ta)
 * is an effective extra "bad-potential" filter.
 *
 * This namespace lives in its own header so the WARM init strategy can
 * prime the cache from `# HOT_A0 <script> <a0>` lines saved into
 * best_pot.tadah. Without that priming, the loss recorded in a previous
 * run's best_pot is not reproducible on cold restart: every script
 * starts from TADAH_HPO_HOT_A0_DEFAULT instead of the lattice constant
 * the original best converged to, which can land box/relax in a
 * different basin and produce a wildly different loss for the same
 * parameter vector.
 */

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tadah::hpo::warmstart {

inline std::mutex& mtx() { static std::mutex m; return m; }
inline std::unordered_map<std::string,double>& a0_map() {
    static std::unordered_map<std::string,double> m; return m;
}
inline double a0_default() {
    static const double v = []{
        const char* s = std::getenv("TADAH_HPO_HOT_A0_DEFAULT");
        return s && *s ? std::atof(s) : 3.2;
    }();
    return v;
}
inline double a0_min() {
    static const double v = []{
        const char* s = std::getenv("TADAH_HPO_A0_MIN");
        return s && *s ? std::atof(s) : 1.0;
    }();
    return v;
}
inline double a0_max() {
    static const double v = []{
        const char* s = std::getenv("TADAH_HPO_A0_MAX");
        return s && *s ? std::atof(s) : 10.0;
    }();
    return v;
}
inline bool in_range(double v) {
    return std::isfinite(v) && v >= a0_min() && v <= a0_max();
}
inline double get(const std::string& script) {
    std::lock_guard<std::mutex> lg(mtx());
    auto it = a0_map().find(script);
    return it != a0_map().end() ? it->second : a0_default();
}
inline bool set(const std::string& script, double v) {
    if (!in_range(v)) return false;
    std::lock_guard<std::mutex> lg(mtx());
    a0_map()[script] = v;
    return true;
}
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

/* Persisted-cache format used in best_pot.tadah. The deserializer
 * strips '#'-prefixed lines, so this is invisible to anything that
 * isn't explicitly looking for it (e.g. WarmInitStrategy). One line
 * per script, full numerical precision so a round-trip is bit-exact. */
inline void dump_to_stream(std::ostream& os) {
    std::lock_guard<std::mutex> lg(mtx());
    const auto& m = a0_map();
    if (m.empty()) return;
    const auto old_prec = os.precision();
    const auto old_flags = os.flags();
    os << std::scientific << std::setprecision(17);
    for (const auto& kv : m)
        os << "# HOT_A0 " << kv.first << " " << kv.second << "\n";
    os.flags(old_flags);
    os.precision(old_prec);
}

/* Parse one line that may carry a HOT_A0 record. Returns true only
 * when the line is a well-formed `# HOT_A0 <script> <a0>` AND the
 * value passes set()'s in-range check. Any other line — non-HOT_A0
 * comment, data body, malformed HOT_A0, or out-of-range value —
 * returns false. The bool is the caller's per-line "did this prime
 * the cache?" signal, used for the WARM summary log. */
inline bool try_parse_line(const std::string& line) {
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos || line[first] != '#') return false;
    std::istringstream iss(line.substr(first + 1));
    std::string tag;
    if (!(iss >> tag) || tag != "HOT_A0") return false;
    std::string script;
    double a0;
    if (!(iss >> script >> a0)) return false;
    return set(script, a0);
}

} // namespace tadah::hpo::warmstart

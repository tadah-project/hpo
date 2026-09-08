/*
 * Concrete InitStrategy implementations and the build_init_strategy
 * factory that resolves a parsed InitStrategySpec into a single
 * polymorphic strategy (possibly a Chain).
 *
 * Increment 2 ships CONFIG and RANDOM. LHS, SOBOL, WARM are added
 * by later increments; the factory throws "not yet implemented" for them
 * here so the parser surface can be exercised in advance.
 */

#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/core/context.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace tadah::hpo {

namespace {

/* Map a unit-cube point [0,1]^d to a point in real space, respecting
 * the per-dim log_scaled mask. low[d]/high[d] are already in transformed
 * space (post analyse_logscale). */
std::vector<double>
unit_to_real(const OptimizerSpec& sp,
             const std::vector<double>& u)
{
    std::vector<double> t(u.size());
    for (std::size_t d = 0; d < u.size(); ++d)
        t[d] = sp.low[d] + u[d] * (sp.high[d] - sp.low[d]);
    return sp.out(t);   // INTERNAL -> REAL through the search space
}

/* Walk targets.optimap in deterministic (lex-sorted) order and pull the
 * scalar config value for every OPTIM-flagged dim. The returned vector
 * is in the same dim order as spec.low / spec.high / spec.optim_init. */
std::vector<double>
config_point_from_targets(const HPOTargets& trg)
{
    std::vector<double> point;
    for (const auto& kv : trg.optimap) {
        const auto& o = kv.second;
        for (std::size_t i = 0; i < o.opt.size(); ++i) {
            if (!o.opt[i]) continue;
            point.push_back(std::stod(o.config[i]));
        }
    }
    return point;
}

/* ---------------- CONFIG ----------------------------------------- */
class ConfigInitStrategy : public InitStrategy {
public:
    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t /*k*/) const override
    {
        return { config_point_from_targets(ctx.targets) };
    }
    std::string name() const override { return "CONFIG"; }
};

/* ---------------- RANDOM ----------------------------------------- */
class RandomInitStrategy : public InitStrategy {
public:
    explicit RandomInitStrategy(std::size_t k) : k_(k) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_override) const override
    {
        const std::size_t k = k_override > 0 ? k_override : k_;
        const std::size_t dim = ctx.spec.low.size();

        std::mt19937_64 rng(ctx.seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        std::vector<std::vector<double>> out;
        out.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            std::vector<double> u(dim);
            for (std::size_t d = 0; d < dim; ++d) u[d] = uniform(rng);
            out.push_back(unit_to_real(ctx.spec, u));
        }
        return out;
    }
    std::string name() const override { return "RANDOM"; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- LHS (Latin Hypercube) -------------------------- */
class LhsInitStrategy : public InitStrategy {
public:
    explicit LhsInitStrategy(std::size_t k) : k_(k) {
        if (k_ == 0) throw std::runtime_error("INIT_STRATEGY LHS K must be > 0");
    }

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_override) const override
    {
        const std::size_t k = k_override > 0 ? k_override : k_;
        const std::size_t dim = ctx.spec.low.size();

        std::mt19937_64 rng(ctx.seed);
        std::uniform_real_distribution<double> jitter(0.0, 1.0);

        /* For each dim independently: random permutation of [0..k-1] +
         * per-stratum jitter. Each sample then gets one stratum per dim. */
        std::vector<std::vector<std::size_t>> perms(dim, std::vector<std::size_t>(k));
        for (std::size_t d = 0; d < dim; ++d) {
            std::iota(perms[d].begin(), perms[d].end(), std::size_t{0});
            std::shuffle(perms[d].begin(), perms[d].end(), rng);
        }

        std::vector<std::vector<double>> out;
        out.reserve(k);
        const double inv_k = 1.0 / static_cast<double>(k);
        for (std::size_t i = 0; i < k; ++i) {
            std::vector<double> u(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                u[d] = (static_cast<double>(perms[d][i]) + jitter(rng)) * inv_k;
            }
            out.push_back(unit_to_real(ctx.spec, u));
        }
        return out;
    }
    std::string name() const override { return "LHS"; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- SOBOL (boost quasi-random) --------------------- */
class SobolInitStrategy : public InitStrategy {
public:
    explicit SobolInitStrategy(std::size_t k) : k_(k) {
        if (k_ == 0) throw std::runtime_error("INIT_STRATEGY SOBOL K must be > 0");
    }

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_override) const override
    {
        const std::size_t k = k_override > 0 ? k_override : k_;
        const std::size_t dim = ctx.spec.low.size();

        if (dim == 0) {
            /* Sobol with dim=0 is meaningless; emulate the strategy by
             * returning k zero-dimensional points (the chain de-dup
             * step in ChainStrategy will collapse them). */
            return std::vector<std::vector<double>>(k);
        }

        boost::random::sobol engine(static_cast<unsigned>(dim));
        if (ctx.seed != 0) {
            /* Sobol is deterministic; "seeding" amounts to discarding
             * the first N points so different runs sample different
             * sub-sequences. We use the lower 16 bits as a discard
             * count (0..65535) to keep the wait time bounded. */
            const std::uint64_t skip = ctx.seed & 0xFFFFu;
            engine.discard(skip * dim);
        }
        const auto sobol_max = static_cast<double>(engine.max());
        const auto sobol_min = static_cast<double>(engine.min());
        const double range = sobol_max - sobol_min + 1.0;

        std::vector<std::vector<double>> out;
        out.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            std::vector<double> u(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                /* Sobol returns integer-typed quasi-random output spanning
                 * [min, max]; map to [0,1) via the documented formula. */
                const auto raw = static_cast<double>(engine());
                u[d] = (raw - sobol_min) / range;
            }
            out.push_back(unit_to_real(ctx.spec, u));
        }
        return out;
    }
    std::string name() const override { return "SOBOL"; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- WARM (read previous pot.tadah) ----------------- */
class WarmInitStrategy : public InitStrategy {
public:
    explicit WarmInitStrategy(std::string filename) : file_(std::move(filename)) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t /*k*/) const override
    {
        const std::string& policy = ctx.spec.init_strategy_spec.warm_on_mismatch;

        std::ifstream in(file_);
        if (!in) {
            /* UNCONDITIONALLY fatal, independent of WARM_ON_MISMATCH:
             * the policy governs seed-CONTENT mismatches (missing
             * parameters, out-of-box clamping). A missing file is a
             * broken launch -- falling through would silently start the
             * optimiser from its default point (ANNEAL: the OPTIM box
             * centre) instead of the seed the user asked for. */
            throw std::runtime_error(
                "WARM: cannot open seed file '" + file_ + "'. Provide the "
                "file (production submit scripts copy it into the run dir) "
                "or drop the WARM line to start from the in-config values.");
        }

        /* Read the file into a buffer, stripping comment + blank lines.
         * Comment lines beginning with `# HOT <script> <name> <value>`
         * (or the legacy `# HOT_A0 <script> <a0>`) carry the warmstart
         * cache snapshot that the writer dumped alongside the best-pot.
         * Priming `warmstart::cache` from them makes the WARM eval
         * start each --hotvar relaxation from the same value the
         * original best converged to -- without that, a cold restart
         * begins each hot script from its declared `initial` and may
         * land box/relax in a different basin, producing a loss that
         * does not match what this file's header claims. */
        std::vector<char> buf;
        std::string line;
        std::size_t hot_loaded = 0;
        while (std::getline(in, line)) {
            const auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue;
            if (line[first] == '#') {
                if (warmstart::try_parse_line(line)) ++hot_loaded;
                continue;
            }
            buf.insert(buf.end(), line.begin(), line.end());
            buf.push_back('\n');
        }
        if (hot_loaded > 0) {
            std::ostringstream oss;
            oss << "WARM: primed warmstart cache with " << hot_loaded
                << " HOT entry(ies) from " << file_;
            std::cout << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().info(oss.str());
        }

        core::Context loaded;
        try {
            loaded.deserialize(buf);
        } catch (const std::exception& e) {
            /* Same class as a missing file: an unreadable seed cannot be
             * warm-started from, and falling through would silently
             * launch from the box centre. Always fatal. */
            throw std::runtime_error(
                std::string("WARM: parse failure in seed file '") + file_ +
                "': " + e.what());
        }

        std::vector<double> point;
        std::vector<std::string> missing;

        for (const auto& kv : ctx.targets.optimap) {
            const auto& opti = kv.second;

            std::vector<std::string> loaded_vec;
            bool key_present = false;
            try {
                std::size_t n = loaded.size(kv.first);
                loaded_vec.resize(n);
                loaded.get<std::vector<std::string>>(kv.first, loaded_vec);
                key_present = true;
            } catch (...) {
                key_present = false;
            }

            for (std::size_t i = 0; i < opti.opt.size(); ++i) {
                if (!opti.opt[i]) continue;

                /* Numeric ordinal of dim i within opti.config. */
                std::size_t ord = 0;
                for (std::size_t j = 0; j <= i; ++j) {
                    if (core::is_number(opti.config[j])) ord++;
                }

                bool resolved = false;
                if (key_present) {
                    std::size_t found = 0;
                    for (const auto& tok : loaded_vec) {
                        if (!core::is_number(tok)) continue;
                        if (++found == ord) {
                            try {
                                point.push_back(std::stod(tok));
                                resolved = true;
                            } catch (...) { /* fall through to fallback */ }
                            break;
                        }
                    }
                }
                if (!resolved) {
                    /* Name the dimension the way every other message does:
                     * the search-space label (KEY[n], n = OPTIM index);
                     * point.size() is this dimension's position because
                     * the traversal order is the optimiser order. */
                    const auto& lbls = ctx.targets.spec.parameter_labels;
                    missing.push_back(point.size() < lbls.size()
                                          ? lbls[point.size()]
                                          : kv.first + "[" + std::to_string(ord) + "]");
                    /* Fallback to the in-config value so a partial WARM
                     * is still a valid point. The chain semantics let
                     * the user follow WARM with CONFIG to override
                     * the per-dim fallback if desired. */
                    point.push_back(std::stod(opti.config[i]));
                }
            }
        }

        if (!missing.empty()) {
            std::ostringstream oss;
            oss << "WARM: " << missing.size() << " parameter(s) not found in "
                << file_ << ": ";
            for (std::size_t i = 0; i < missing.size() && i < 5; ++i) {
                if (i) oss << ", ";
                oss << missing[i];
            }
            if (missing.size() > 5) oss << ", ...";
            handle_mismatch(policy, oss.str());
            if (policy == "skip") return {};
        } else {
            std::ostringstream oss;
            oss << "WARM: loaded " << point.size()
                << " parameter(s) from " << file_;
            std::cout << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().info(oss.str());
        }

        /* Clamp into bounds (in real space) -- bounds may have moved
         * between runs. A clamped seed is a DIFFERENT starting point than
         * the user asked for (it silently destroyed a whole run once:
         * seed weights far outside narrowed OPTIM boxes were pinned to the
         * box corner and the seed was never evaluated), so clamping goes
         * through the same WARM_ON_MISMATCH policy as missing parameters:
         * error -> throw, skip -> drop the point, warn (default) -> log. */
        std::size_t clamped = 0;
        std::ostringstream coss;
        for (std::size_t d = 0; d < point.size(); ++d) {
            double lo, hi;
            if (ctx.spec.space && d < ctx.spec.space->dim()) {
                lo = ctx.spec.space->at(d).lo;
                hi = ctx.spec.space->at(d).hi;
            } else {
                const bool log_d = ctx.spec.log_scaled[d];
                lo = log_d ? std::exp(ctx.spec.low[d])  : ctx.spec.low[d];
                hi = log_d ? std::exp(ctx.spec.high[d]) : ctx.spec.high[d];
            }
            const double orig = point[d];
            if (point[d] < lo) point[d] = lo;
            if (point[d] > hi) point[d] = hi;
            if (point[d] != orig) {
                if (clamped < 5) {
                    coss << (clamped ? "; " : "") << "dim[" << d
                         << "] " << orig << " -> " << point[d]
                         << " (bounds [" << lo << ", " << hi << "])";
                }
                ++clamped;
            }
        }
        if (clamped > 0) {
            std::ostringstream oss;
            oss << "WARM: " << clamped << " parameter(s) from " << file_
                << " clamped into OPTIM bounds -- the seed point is NOT the "
                   "one on disk: " << coss.str();
            if (clamped > 5) oss << "; ... (further entries suppressed)";
            HPOLog::instance().warning(oss.str());
            handle_mismatch(policy, oss.str());
            if (policy == "skip") return {};
        }
        return { std::move(point) };
    }
    std::string name() const override { return "WARM"; }

private:
    static void handle_mismatch(const std::string& policy, const std::string& msg) {
        if (policy == "error") throw std::runtime_error(msg);
        std::cerr << "[HPO] " << msg << "\n";
        /* stderr alone is easy to lose under SLURM -- a missing/short
         * seed silently changes the STARTING POINT of the run, so the
         * journal must carry it too. */
        HPOLog::instance().warning(msg);
    }
    std::string file_;
};

/* ---------------- Chain (concatenation with de-dup) -------------- */
class ChainStrategy : public InitStrategy {
public:
    explicit ChainStrategy(std::vector<std::unique_ptr<InitStrategy>> children)
        : children_(std::move(children)) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_total) const override
    {
        std::vector<std::vector<double>> out;
        std::uint64_t base_seed = ctx.seed;
        for (const auto& s : children_) {
            InitContext sub{ctx.spec, ctx.targets, base_seed};
            auto pts = s->generate(sub, s->default_k());
            for (auto& p : pts) {
                if (!duplicate(out, p, ctx.spec))
                    out.push_back(std::move(p));
                if (k_total > 0 && out.size() >= k_total) return out;
            }
            base_seed = base_seed * 6364136223846793005ull + 1442695040888963407ull;
        }
        return out;
    }
    std::string name() const override { return "CHAIN"; }
    std::size_t default_k() const override {
        std::size_t s = 0;
        for (const auto& c : children_) s += c->default_k();
        return s;
    }
private:
    static bool duplicate(const std::vector<std::vector<double>>& haystack,
                          const std::vector<double>& p,
                          const OptimizerSpec& sp)
    {
        const double tol = 1.0e-12;
        /* compare in INTERNAL space so log/lin dims share a meaningful
         * tolerance. */
        const std::vector<double> pi = sp.in(p);
        for (const auto& q : haystack) {
            if (q.size() != p.size()) continue;
            const std::vector<double> qi = sp.in(q);
            bool eq = true;
            for (std::size_t d = 0; d < p.size() && eq; ++d)
                if (std::abs(pi[d] - qi[d]) > tol) eq = false;
            if (eq) return true;
        }
        return false;
    }
    std::vector<std::unique_ptr<InitStrategy>> children_;
};

/* ---------------- Factory ---------------------------------------- */
std::unique_ptr<InitStrategy>
make_one(const InitStrategySpec::Entry& e)
{
    auto parse_uint = [](const std::string& s) -> std::size_t {
        try { return static_cast<std::size_t>(std::stoul(s)); }
        catch (...) {
            throw std::runtime_error("INIT_STRATEGY " + s +
                                     ": expected a non-negative integer");
        }
    };

    if (e.method == "CONFIG") {
        if (!e.args.empty())
            throw std::runtime_error("INIT_STRATEGY CONFIG takes no arguments");
        return std::make_unique<ConfigInitStrategy>();
    }
    if (e.method == "RANDOM") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT_STRATEGY RANDOM expects exactly one arg: K");
        return std::make_unique<RandomInitStrategy>(parse_uint(e.args[0]));
    }
    if (e.method == "LHS") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT_STRATEGY LHS expects exactly one arg: K");
        return std::make_unique<LhsInitStrategy>(parse_uint(e.args[0]));
    }
    if (e.method == "SOBOL") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT_STRATEGY SOBOL expects exactly one arg: K");
        return std::make_unique<SobolInitStrategy>(parse_uint(e.args[0]));
    }
    if (e.method == "WARM") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT_STRATEGY WARM expects exactly one arg: <pot_file>");
        return std::make_unique<WarmInitStrategy>(e.args[0]);
    }
    /* NOISE_AUTO is rejected up-front in the parser so the user gets
     * an immediate, clear error rather than discovering at resolve
     * time. The branch here would only fire if a caller bypassed the
     * parser; treat it the same as an unknown method for safety. */
    throw std::runtime_error("Unknown INIT_STRATEGY method: " + e.method);
}

}  // namespace

std::unique_ptr<InitStrategy>
build_init_strategy(const InitStrategySpec& ispec)
{
    /* The chain is instantiated in exactly the order the user wrote it
     * -- no entry is reordered. iter 0 is therefore the first STRATEGY
     * line in the INIT block. The in-config OPTIM point participates
     * only when the user listed `STRATEGY CONFIG` explicitly; there is
     * no implicit CONFIG prepend. */
    std::vector<std::unique_ptr<InitStrategy>> kids;
    for (const auto& e : ispec.chain) kids.push_back(make_one(e));

    /* An empty chain (no INIT block, or one with no STRATEGY lines)
     * falls back to a single CONFIG point -- the in-config OPTIM values
     * are always a valid starting point. */
    if (kids.empty())
        kids.push_back(std::make_unique<ConfigInitStrategy>());

    if (kids.size() == 1) return std::move(kids.front());
    return std::make_unique<ChainStrategy>(std::move(kids));
}

}  // namespace tadah::hpo

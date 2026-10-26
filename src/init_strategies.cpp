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
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/seed_stream.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/retired_keys.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/utils.h>
#include <tadah/core/number_format.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace tadah::hpo {

std::vector<double>
config_point_from_targets(const HPOTargets& trg)
{
    std::vector<double> point;
    for (const auto& kv : trg.optimap) {
        const auto& o = kv.second;
        for (std::size_t i = 0; i < o.opt.size(); ++i) {
            if (!o.opt[i]) continue;
            point.push_back(core::to_double(o.config[i]));
        }
    }
    return point;
}

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

/* The generator of a random starting-point method (RANDOM, LHS): an INIT
 * block's chain draws std::mt19937_64(seed), as it always did; a RESTART
 * round's draws a stream of its own (InitStream::Restart,
 * seed_streams::kRestartPoints), so a round never draws an INIT block's
 * points, whatever the two SEEDs: an INIT block's SEED s and a round whose
 * RESTART SEED + r was s drew the same points. */
std::mt19937_64 method_generator(const InitContext& ctx)
{
    if (ctx.stream == InitStream::Restart)
        return seed_stream(ctx.seed, seed_streams::kRestartPoints);
    return std::mt19937_64(ctx.seed);
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
    InitOrigin origin() const override { return {InitOrigin::Kind::Config, {}}; }
};

/* ---------------- RANDOM ----------------------------------------- */
class RandomInitStrategy : public InitStrategy {
public:
    explicit RandomInitStrategy(std::size_t k) : k_(k) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_cap) const override
    {
        /* K_TOTAL caps the K points (up to k_cap of them: the first ones
         * of the same draw). */
        const std::size_t k = k_cap > 0 ? std::min(k_cap, k_) : k_;
        const std::size_t dim = ctx.spec.low.size();

        std::mt19937_64 rng = method_generator(ctx);
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
    bool draws_from_seed() const override { return true; }
    InitOrigin origin() const override { return {InitOrigin::Kind::Random, {}}; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- LHS (Latin Hypercube) -------------------------- */
class LhsInitStrategy : public InitStrategy {
public:
    explicit LhsInitStrategy(std::size_t k) : k_(k) {
        if (k_ == 0) throw std::runtime_error("INIT METHOD LHS K must be > 0");
    }

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_cap) const override
    {
        /* The Latin hypercube of K points, of which K_TOTAL keeps the first
         * (up to k_cap of them). */
        const std::size_t k = k_;
        const std::size_t n = k_cap > 0 ? std::min(k_cap, k_) : k_;
        const std::size_t dim = ctx.spec.low.size();

        std::mt19937_64 rng = method_generator(ctx);
        std::uniform_real_distribution<double> jitter(0.0, 1.0);

        /* For each dim independently: random permutation of [0..k-1] +
         * per-stratum jitter. Each sample then gets one stratum per dim. */
        std::vector<std::vector<std::size_t>> perms(dim, std::vector<std::size_t>(k));
        for (std::size_t d = 0; d < dim; ++d) {
            std::iota(perms[d].begin(), perms[d].end(), std::size_t{0});
            std::shuffle(perms[d].begin(), perms[d].end(), rng);
        }

        std::vector<std::vector<double>> out;
        out.reserve(n);
        const double inv_k = 1.0 / static_cast<double>(k);
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> u(dim);
            for (std::size_t d = 0; d < dim; ++d) {
                u[d] = (static_cast<double>(perms[d][i]) + jitter(rng)) * inv_k;
            }
            out.push_back(unit_to_real(ctx.spec, u));
        }
        return out;
    }
    std::string name() const override { return "LHS"; }
    bool draws_from_seed() const override { return true; }
    InitOrigin origin() const override { return {InitOrigin::Kind::Lhs, {}}; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- SOBOL (boost quasi-random) --------------------- */
class SobolInitStrategy : public InitStrategy {
public:
    explicit SobolInitStrategy(std::size_t k) : k_(k) {
        if (k_ == 0) throw std::runtime_error("INIT METHOD SOBOL K must be > 0");
    }

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_cap) const override
    {
        /* The block of K points, of which K_TOTAL keeps the first (up to
         * k_cap of them). */
        const std::size_t k = k_;
        const std::size_t n = k_cap > 0 ? std::min(k_cap, k_) : k_;
        const std::size_t dim = ctx.spec.low.size();

        if (dim == 0) {
            /* Sobol with dim=0 is meaningless; emulate the strategy by
             * returning n zero-dimensional points (the chain de-dup
             * step in ChainStrategy will collapse them). */
            return std::vector<std::vector<double>>(n);
        }

        boost::random::sobol engine(static_cast<unsigned>(dim));
        /* Sobol is deterministic; a seed selects which points to draw. Seed
         * s draws block b = s mod 65536 of k points: points b*k ... b*k +
         * k - 1 of the sequence. Distinct blocks share no point, so the
         * seeds SEED + r of consecutive RESTART rounds, or an INIT block
         * and a RESTART block with the same SEED and K, draw disjoint
         * points; with k = 1 seed s draws point s, as it always did. (A
         * seed used to skip s mod 65536 points, so the seeds s and s + 1
         * shared k - 1 of their k points.) The block index keeps the 16
         * low bits of the seed, as the skip always did: a chain's later
         * methods and a run without SEED receive a pseudo-random 64-bit
         * seed. */
        const std::uint64_t block = ctx.seed & 0xFFFFu;
        const std::uint64_t kk = k;   // >= 1
        constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
        /* The block, points block*k ... block*k + k - 1, lies within the
         * sequence: K is at most kMaxStartingPoints (make_one), and a block
         * holds K points whatever k_cap asks for, so neither the product
         * nor the end wraps. */
        static_assert(kMaxStartingPoints <= kMax / 0x10000u,
                      "block 65535 of kMaxStartingPoints points must end "
                      "within the Sobol sequence");
        /* Position the engine at point `first` (the Gray-code state of that
         * index): the same point discard(first * dim) reaches, without the
         * product with the dimension. */
        const std::uint64_t first = block * kk;
        if (first != 0) engine.seed(first);
        const auto sobol_max = static_cast<double>(engine.max());
        const auto sobol_min = static_cast<double>(engine.min());
        const double range = sobol_max - sobol_min + 1.0;

        std::vector<std::vector<double>> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
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
    bool draws_from_seed() const override { return true; }
    InitOrigin origin() const override { return {InitOrigin::Kind::Sobol, {}}; }
    std::size_t default_k() const override { return k_; }
private:
    std::size_t k_;
};

/* ---------------- WARM (read previous pot.tadah) ----------------- */
/* `policy` is the WARM_ON_MISMATCH of the block whose chain holds this
 * entry (an INIT or a RESTART block; build_init_strategy passes it), not
 * the INIT block of the spec in the InitContext: a RESTART chain is drawn
 * with the OUTER spec, whose INIT block is not the RESTART block. */
class WarmInitStrategy : public InitStrategy {
public:
    WarmInitStrategy(std::string filename, std::string policy)
        : file_(std::move(filename)), policy_(std::move(policy)) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t /*k*/) const override
    {
        const std::string& policy = policy_;

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
        bool structured=false;
        std::string line;
        std::size_t hot_loaded = 0;
        /* The lines of retired keys (a potential of an earlier version has
         * EWEIGHT, FWEIGHT, SWEIGHT, WDBFILE_AUTO): not read, and named in
         * one notice for the seed, as every load of a potential names
         * them (core::retired_keys_notice). */
        std::vector<core::RetiredKeyLine> retired;
        std::size_t line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            const auto first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue;
            if (line[first] == '#') {
                if (warmstart::try_parse_line(line)) ++hot_loaded;
                continue;
            }
            std::istringstream header(line);std::string key;header>>key;
            const std::string upper_key = core::to_upper(key);
            const auto& keys = core::BlueprintParser::getKeysMapRef();
            if (const auto k = keys.find(upper_key);
                k != keys.end() && k->second.isRetired()) {
                std::vector<std::string> values;
                for (std::string v; header >> v;) values.push_back(v);
                const auto last = line.find_last_not_of(" \t\r");
                retired.push_back(core::RetiredKeyLine{
                    upper_key, file_, line_no, std::move(values),
                    line.substr(first, last - first + 1)});
                continue;
            }
            structured=structured || upper_key=="DESCRIPTOR";
            buf.insert(buf.end(), line.begin(), line.end());
            buf.push_back('\n');
        }
        if (!retired.empty())
            core::report_key_notice(core::retired_keys_notice(retired));
        if (hot_loaded > 0) {
            std::ostringstream oss;
            oss << "WARM: primed warmstart cache with " << hot_loaded
                << " HOT entry(ies) from " << file_;
            std::cout << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().info(oss.str());
        }

        core::Context loaded;
        try {
            if(structured) {
              std::istringstream input(std::string(buf.begin(),buf.end()));
              loaded=core::CoreUserConfigParser().parse(input);
            } else loaded.deserialize(buf);
        } catch (const std::exception& e) {
            /* Same class as a missing file: an unreadable seed cannot be
             * warm-started from, and falling through would silently
             * launch from the box centre. Always fatal. */
            throw std::runtime_error(
                std::string("WARM: parse failure in seed file '") + file_ +
                "': " + e.what());
        }

        /* EMBF0: WARM reads only the OPTIM dimensions, so the run evaluates
         * the embedding reference of its -c configuration. Stored values
         * only (no default inserted into either context). */
        if (ctx.run_context && models::context_has_ream(*ctx.run_context) &&
            models::context_has_ream(loaded)) {
            using models::EmbF0Mode;
            std::optional<EmbF0Mode> seed_mode;
            bool seed_valid = true;
            try { seed_mode = models::stored_embf0(loaded); }
            catch (const std::exception&) { seed_valid = false; }
            const EmbF0Mode run_mode =
                models::stored_embf0(*ctx.run_context)
                    .value_or(EmbF0Mode::Zero);
            if (!seed_valid) {
                const std::string msg =
                    "WARM seed " + file_ + " declares an invalid EMBF0 value "
                    "(ZERO|FREE); ignored: this run evaluates EMBF0 " +
                    models::embf0_name(run_mode) + " (from -c).";
                std::cerr << "[HPO] " << msg << "\n";
                HPOLog::instance().warning(msg);
            } else if (seed_mode.value_or(EmbF0Mode::Zero) != run_mode) {
                const std::string seed_decl =
                    seed_mode ? std::string("declares EMBF0 ") +
                                    models::embf0_name(*seed_mode)
                              : std::string("does not declare EMBF0 (read as "
                                            "EMBF0 ZERO)");
                const std::string msg =
                    "WARM seed " + file_ + " " + seed_decl + "; this run "
                    "evaluates EMBF0 " + models::embf0_name(run_mode) +
                    " (from -c). WARM takes only the OPTIM values from the "
                    "seed and evaluates them unchanged in the run's reference "
                    "(no conversion): pinned embedding weights give energies "
                    "shifted by a constant per atom (linear models: forces "
                    "unchanged), weights re-fitted by least squares can "
                    "differ, so the seed's recorded loss need not be "
                    "reproduced";
                std::cerr << "[HPO] " << msg << "\n";
                HPOLog::instance().warning(msg);
            } else if (!seed_mode) {
                const std::string msg =
                    "WARM seed " + file_ + " does not declare EMBF0: it is "
                    "read as EMBF0 ZERO (the default), as this run evaluates.";
                std::cout << "[HPO] " << msg << "\n";
                HPOLog::instance().info(msg);
            }
        }

        /* RHOFORM: density amplitudes mean something only in the form they
         * were fitted in, so a seed whose form is absent or differs from
         * the run's is an ERROR when WARM would read density values from
         * it (an OPTIM'd AMPGRIDMB, CGRIDMB or SGRIDMB), else a WARNING.
         * Independent of WARM_ON_MISMATCH (that policy is about missing and
         * clamped values). Stored values only. */
        if (ctx.run_context && models::context_has_ream(*ctx.run_context) &&
            models::context_has_ream(loaded))
            check_seed_rhoform(ctx, loaded);

        std::vector<double> point;
        std::vector<std::string> missing;

        for (const auto& kv : ctx.targets.optimap) {
            const auto& opti = kv.second;

            std::vector<std::string> loaded_vec;
            bool key_present = false;
            try {
                if(kv.first.find('.')!=std::string::npos) {
                  auto binding=models::embed_binding(loaded,kv.first);
                  for(double v:binding.values) loaded_vec.push_back(core::format_roundtrip(v));
                } else {
                  /* A key the seed has no line of is missing: reading it
                   * would give its blueprint default (Context::size
                   * inserts it), not a value of the seed -- e.g. the error
                   * scales of a potential of an earlier version. */
                  if (!loaded.has_stored(kv.first))
                    throw std::out_of_range(kv.first);
                  std::size_t n = loaded.size(kv.first);
                  loaded_vec.resize(n);
                  loaded.get<std::vector<std::string>>(kv.first, loaded_vec);
                }
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
                                point.push_back(core::to_double(tok));
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
                    point.push_back(core::to_double(opti.config[i]));
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
                    coss << (clamped ? "; " : "") << logidx::dim_label(d)
                         << " " << orig << " -> " << point[d]
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
    InitOrigin origin() const override { return {InitOrigin::Kind::Warm, file_}; }

private:
    void check_seed_rhoform(const InitContext& ctx,
                            const core::Context& loaded) const {
        const models::RhoForm run_form =
            models::require_rhoform(*ctx.run_context, "tadah hpo");
        std::optional<models::RhoForm> seed_form;
        bool seed_valid = true;
        try { seed_form = models::stored_rhoform(loaded); }
        catch (const std::exception&) { seed_valid = false; }
        if (seed_valid && seed_form && *seed_form == run_form) return;

        std::string density_dims;   // the OPTIM'd density keys, if any
        for (const auto& kv : ctx.targets.optimap) {
            const std::string& key = kv.first;   // upper-case (OPTIM)
            if (key != "AMPGRIDMB" && key != "CGRIDMB" && key != "SGRIDMB")
                continue;
            const auto& opt = kv.second.opt;
            if (std::find(opt.begin(), opt.end(), true) == opt.end()) continue;
            if (!density_dims.empty()) density_dims += ", ";
            density_dims += key;
        }
        const std::string run_name = models::rhoform_name(run_form);
        const std::string seed_decl =
            !seed_valid ? "declares an invalid RHOFORM value '" +
                              loaded.stored_value("RHOFORM").value_or("") + "'"
            : !seed_form ? std::string("does not declare RHOFORM")
                         : std::string("declares RHOFORM ") +
                               models::rhoform_name(*seed_form);
        /* A seed written before tadah 1.6.0 (no RHOFORM) is SIGNED:
         * declaring that repairs it for a SIGNED run only; a positive form
         * needs a seed fitted in that form. */
        const std::string remedy =
            run_form == models::RhoForm::Signed
                ? "Use a seed written in RHOFORM SIGNED (a potential written "
                  "before tadah 1.6.0 is SIGNED: add 'RHOFORM SIGNED' to it), "
                  "or drop the WARM line."
                : "Use a seed written in RHOFORM " + run_name + " (a potential "
                  "written before tadah 1.6.0 is SIGNED and cannot seed a " +
                  run_name + " density; refit it by tadah refit with "
                  "RHOFORM " + run_name + "), or drop the WARM line.";
        if (!density_dims.empty())
            throw std::runtime_error(
                "WARM seed " + file_ + " " + seed_decl + ", but this run "
                "evaluates RHOFORM " + run_name + " (from -c) and takes the "
                "OPTIM density parameters (" + density_dims + ") from the "
                "seed: density parameters are meaningful only in the form "
                "they were fitted in. " + remedy);
        const std::string msg =
            "WARM seed " + file_ + " " + seed_decl + "; this run evaluates "
            "RHOFORM " + run_name + " (from -c). No OPTIM dimension is a "
            "density key (AMPGRIDMB, CGRIDMB, SGRIDMB), so the seed's density "
            "is not read: every candidate has the -c density.";
        std::cerr << "[HPO] " << msg << "\n";
        HPOLog::instance().warning(msg);
    }

    static void handle_mismatch(const std::string& policy, const std::string& msg) {
        if (policy == "error") throw std::runtime_error(msg);
        std::cerr << "[HPO] " << msg << "\n";
        /* stderr alone is easy to lose under SLURM -- a missing/short
         * seed silently changes the STARTING POINT of the run, so the
         * journal must carry it too. */
        HPOLog::instance().warning(msg);
    }
    std::string file_;
    std::string policy_;   // warn | error | skip: WARM_ON_MISMATCH of the block
};

/* ---------------- Chain (concatenation with de-dup) -------------- */
class ChainStrategy : public InitStrategy {
public:
    explicit ChainStrategy(std::vector<std::unique_ptr<InitStrategy>> children)
        : children_(std::move(children)) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k_total) const override
    {
        std::vector<InitOrigin> origins;
        return generate_tagged(ctx, k_total, origins);
    }
    /* Each point carries the origin of the child that produced it; a
     * de-duplicated point keeps the first (the later duplicate is dropped). */
    std::vector<std::vector<double>>
    generate_tagged(const InitContext& ctx, std::size_t k_total,
                    std::vector<InitOrigin>& origins) const override
    {
        std::vector<std::vector<double>> out;
        origins.clear();
        KeptPoints kept(ctx.spec);
        std::uint64_t base_seed = ctx.seed;
        for (const auto& s : children_) {
            InitContext sub{ctx.spec, ctx.targets, base_seed, ctx.run_context,
                            ctx.stream};
            std::vector<InitOrigin> sub_origins;
            auto pts = s->generate_tagged(sub, s->default_k(), sub_origins);
            for (std::size_t i = 0; i < pts.size(); ++i) {
                if (kept.keep(pts[i])) {
                    out.push_back(std::move(pts[i]));
                    origins.push_back(sub_origins[i]);
                }
                if (k_total > 0 && out.size() >= k_total) return out;
            }
            base_seed = base_seed * 6364136223846793005ull + 1442695040888963407ull;
        }
        return out;
    }
    std::string name() const override { return "CHAIN"; }
    bool draws_from_seed() const override {
        for (const auto& c : children_)
            if (c->draws_from_seed()) return true;
        return false;
    }
    std::size_t default_k() const override {
        std::size_t s = 0;
        for (const auto& c : children_) s += c->default_k();
        return s;
    }
private:
    /* The points a chain keeps, in the coordinates of the search (INTERNAL
     * space, so log and linear dimensions share one tolerance). A point
     * duplicates a kept one of its size when every coordinate differs by
     * at most kTol; the kept points are ordered by their first coordinate,
     * so a point is compared only with those whose first coordinate lies
     * near its own, not with every point kept (a chain of two METHOD lines
     * of kMaxStartingPoints points each compared about 10^12 pairs). */
    class KeptPoints {
    public:
        explicit KeptPoints(const OptimizerSpec& sp) : sp_(sp) {}

        /* True, and `p` is kept, when no kept point duplicates it. */
        bool keep(const std::vector<double>& p)
        {
            std::vector<double> pi = sp_.in(p);
            if (pi.empty()) {           // a point of no search parameter
                if (kept_empty_) return false;
                kept_empty_ = true;
                return true;
            }
            /* The window is twice the tolerance wide on each side, so that
             * rounding of x -+ kTol leaves out no point the test below
             * calls a duplicate. */
            const double x = pi.front();
            for (auto it = by_first_.lower_bound(x - 2.0 * kTol);
                 it != by_first_.end() && it->first <= x + 2.0 * kTol; ++it) {
                const std::vector<double>& q = internal_[it->second];
                if (q.size() != pi.size()) continue;
                bool eq = true;
                for (std::size_t d = 0; d < pi.size() && eq; ++d)
                    if (std::abs(pi[d] - q[d]) > kTol) eq = false;
                if (eq) return false;
            }
            by_first_.emplace(x, internal_.size());
            internal_.push_back(std::move(pi));
            return true;
        }

    private:
        static constexpr double kTol = 1.0e-12;
        const OptimizerSpec& sp_;
        std::multimap<double, std::size_t> by_first_;  // first coord -> index
        std::vector<std::vector<double>> internal_;
        bool kept_empty_ = false;
    };

    std::vector<std::unique_ptr<InitStrategy>> children_;
};

/* ---------------- An entry with its METHOD line ------------------- */
/* An entry the parser read: a refusal of its points, which a chain draws
 * after the whole file has been read (a WARM file that is missing or
 * unreadable, a WARM seed the run refuses, a WARM_ON_MISMATCH error, a
 * SOBOL block beyond the sequence), names the METHOD line in the parser's
 * form (config_line_error), as the parser's own refusals of the line do. */
class LinedEntryStrategy : public InitStrategy {
public:
    LinedEntryStrategy(std::unique_ptr<InitStrategy> entry, std::size_t line,
                       std::string text)
        : entry_(std::move(entry)), line_(line), text_(std::move(text)) {}

    std::vector<std::vector<double>>
    generate(const InitContext& ctx, std::size_t k) const override
    {
        try {
            return entry_->generate(ctx, k);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(config_line_error(line_, e.what(), text_));
        }
    }
    std::vector<std::vector<double>>
    generate_tagged(const InitContext& ctx, std::size_t k,
                    std::vector<InitOrigin>& origins) const override
    {
        try {
            return entry_->generate_tagged(ctx, k, origins);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(config_line_error(line_, e.what(), text_));
        }
    }
    InitOrigin origin() const override { return entry_->origin(); }
    std::string name() const override { return entry_->name(); }
    std::size_t default_k() const override { return entry_->default_k(); }
    bool draws_from_seed() const override { return entry_->draws_from_seed(); }

private:
    std::unique_ptr<InitStrategy> entry_;
    std::size_t line_;
    std::string text_;
};

/* ---------------- Factory ---------------------------------------- */
/* `warm_policy`: the WARM_ON_MISMATCH of the block that holds the entry,
 * for a WARM entry. */
std::unique_ptr<InitStrategy>
make_one(const InitStrategySpec::Entry& e, const std::string& warm_policy)
{
    /* K: a whole integer (strict_parse.h) from 1 -- each method draws at
     * least one point (LHS and SOBOL check it in their constructors too;
     * RANDOM 0 drew none) -- to kMaxStartingPoints, the points a block can
     * hold; stoul read "3x" as 3 and "-1" as 2^64-1. */
    auto parse_k = [&](const std::string& s,
                       std::uint64_t min) -> std::size_t {
        return parse_uint(s, "INIT METHOD " + e.method + " K", min,
                          kMaxStartingPoints);
    };

    if (e.method == "CONFIG") {
        if (!e.args.empty())
            throw std::runtime_error("INIT METHOD CONFIG takes no arguments");
        return std::make_unique<ConfigInitStrategy>();
    }
    if (e.method == "RANDOM") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT METHOD RANDOM expects exactly one arg: K");
        return std::make_unique<RandomInitStrategy>(parse_k(e.args[0], 1));
    }
    if (e.method == "LHS") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT METHOD LHS expects exactly one arg: K");
        return std::make_unique<LhsInitStrategy>(parse_k(e.args[0], 1));
    }
    if (e.method == "SOBOL") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT METHOD SOBOL expects exactly one arg: K");
        return std::make_unique<SobolInitStrategy>(parse_k(e.args[0], 1));
    }
    if (e.method == "WARM") {
        if (e.args.size() != 1)
            throw std::runtime_error("INIT METHOD WARM expects exactly one arg: <pot_file>");
        return std::make_unique<WarmInitStrategy>(e.args[0], warm_policy);
    }
    /* NOISE_AUTO is rejected up-front in the parser so the user gets
     * an immediate, clear error rather than discovering at resolve
     * time. The branch here would only fire if a caller bypassed the
     * parser; treat it the same as an unknown method for safety. */
    throw std::runtime_error("Unknown INIT_STRATEGY method: " + e.method);
}

}  // namespace

void validate_init_entry(const InitStrategySpec::Entry& e)
{
    /* The check builds the entry and draws no point, so the WARM policy
     * plays no part in it: the block's default stands in. */
    (void)make_one(e, InitStrategySpec{}.warm_on_mismatch);
}

std::unique_ptr<InitStrategy>
build_init_strategy(const InitStrategySpec& ispec)
{
    /* The chain is instantiated in exactly the order the user wrote it
     * -- no entry is reordered. iter 0 is therefore the first STRATEGY
     * line in the INIT block. The in-config OPTIM point participates
     * only when the user listed `STRATEGY CONFIG` explicitly; there is
     * no implicit CONFIG prepend. */
    std::vector<std::unique_ptr<InitStrategy>> kids;
    for (const auto& e : ispec.chain) {
        auto kid = make_one(e, ispec.warm_on_mismatch);
        if (e.line != 0)
            kid = std::make_unique<LinedEntryStrategy>(std::move(kid), e.line,
                                                       e.line_text);
        kids.push_back(std::move(kid));
    }

    /* An empty chain (no INIT block, or one with no STRATEGY lines)
     * falls back to a single CONFIG point -- the in-config OPTIM values
     * are always a valid starting point. */
    if (kids.empty())
        kids.push_back(std::make_unique<ConfigInitStrategy>());

    if (kids.size() == 1) return std::move(kids.front());
    return std::make_unique<ChainStrategy>(std::move(kids));
}

}  // namespace tadah::hpo

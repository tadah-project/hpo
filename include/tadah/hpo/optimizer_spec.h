#pragma once
#include <algorithm>
#include <unordered_map>
#include <tadah/hpo/log_index.h>
#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <memory>
#include <limits>
#include <cmath>
#include <unordered_set>
#include <tadah/hpo/search_space.h>

namespace tadah::hpo {



/* ---------------- basic containers -------------------------------- */
struct StopCriteria {
    std::optional<double> stopval, ftol_rel, ftol_abs,
                           xtol_rel, xtol_abs, maxtime;
    std::optional<uint64_t> maxeval;
};

/* GradStrategy -- selector for the unified gradient provider used by
 * gradient-based HPO optimizers (DLIB:BFGS, CERES:LBFGS/WOLFE/NLCG/
 * STEEPEST_DESCENT, future NLOPT:LD_*).
 *
 *   NUMERICAL  -- finite differences on full forward passes through the
 *                host, kept inside the OPTIM box (one-sided within a step
 *                of a bound), with the in-house step rule or Ceres' (PARAM
 *                fd_kernel), CENTRAL or FORWARD (PARAM diff_method;
 *                fd_stencil.h).
 *   ANALYTICAL -- closed-form analytical gradient. The HPO host injects
 *                a per-theta derivative callback (descriptor + cutoff +
 *                kernel closed forms chained through the IFT-back-
 *                substitution on the cached Cholesky factor of the
 *                regression solve). Strict mode: a theta index the host
 *                cannot differentiate throws a diagnostic naming it; a
 *                setup it can never differentiate is refused at startup
 *                (analytical_gate.h). Use HYBRID to fall back.
 *   HYBRID     -- analytical-where-registered + finite-difference
 *                fallback (the stencil of NUMERICAL, PARAM fd_kernel and
 *                diff_method honoured) for unregistered indices -- every
 *                index when no analytical derivative can be formed --
 *                evaluated in one
 *                gradient(x, g) call. Used when LAMMPS-derived loss
 *                components (`sims_loss`) or tabulated descriptors
 *                (EAM SETFL) participate in theta.
 *
 * Configured at OPTIMIZER block level via `GRAD_STRATEGY {NUMERICAL|
 * ANALYTICAL|HYBRID}`. Default (unset) = NUMERICAL; with PC_OBSERVABLE
 * an unset value resolves to ANALYTICAL in the blocks that run an
 * optimiser and obtain gradients (HPOTargets::read). An INNER block
 * takes none (a parse error at its line): MLSL/AUGLAG hand their local
 * search the gradient the enclosing block obtains. The Phase-1 keyword
 * `AUTODIFF` is no longer accepted; the parser emits a tailored error
 * pointing at ANALYTICAL. */
enum class GradStrategy { Numerical, Analytical, Hybrid };

/* The GRAD_STRATEGY keyword of `g`. */
inline const char* grad_strategy_name(GradStrategy g)
{
    switch (g) {
        case GradStrategy::Numerical:  return "NUMERICAL";
        case GradStrategy::Analytical: return "ANALYTICAL";
        case GradStrategy::Hybrid:     return "HYBRID";
    }
    return "NUMERICAL";
}

using ParamMap = std::unordered_map<std::string,double>;

/* PARAM values that did not parse as numbers are stored verbatim here
 * (name -> raw token). Families with enum-valued PARAM keys consume the
 * token form; families whose keys are all numeric reject any entry with
 * a hard error. Before this map existed a non-numeric PARAM value was
 * silently stored as 0.0. */
using RawParamMap = std::unordered_map<std::string,std::string>;

/* Full value-token vector of a PARAM line (name -> all tokens after the
 * key), populated ONLY when a PARAM carries more than one value token --
 * e.g. `PARAM INIT_MODE JITTER AUTO 0.05` -> {"JITTER","AUTO","0.05"}.
 * `param`/`param_raw` are still filled from token 0, so every consumer
 * that predates multi-token PARAM is unaffected.
 *
 * Arity is per key, not global: `STEP KEY(1-4) 0.1` has always been
 * multi-token, while a stray token on a single-value key used to be
 * silently discarded by `iss >> p >> raw`. Keys that accept extras
 * declare them in their family registry; validate_optimizer_scaling()
 * rejects an entry here whose key declares none. */
using ParamTokenMap = std::unordered_map<std::string,std::vector<std::string>>;

/* Defined in anneal_core.h (test/tooling observation seam). */
namespace anneal { struct ChainObserver; }

/* InitStrategySpec -- parsed `INIT/ENDINIT` block.
 * Resolved into a polymorphic InitStrategy by build_init_strategy()
 * after bounds + log-scale are settled. */
struct InitStrategySpec {
    struct Entry {
        std::string method;                    // CONFIG, RANDOM, LHS, SOBOL, WARM, NOISE_AUTO
        std::vector<std::string> args;         // method-specific tokens
    };
    std::vector<Entry> chain;                  // STRATEGY lines (in order, inside INIT block)
    std::optional<std::size_t> k_total;        // K_TOTAL cap (default = sum of strategy default_k)
    std::optional<std::uint64_t> seed;         // SEED for stochastic strategies
    std::string warm_on_mismatch = "warn";     // warn | error | skip (used by WarmInitStrategy)
};

/* InitOrigin -- where one resolved INIT start point came from
 * (OptimizerSpec::optim_init_origin, parallel to optim_init): the in-config
 * OPTIM values (CONFIG, also the empty-chain default), a WARM seed file, or
 * a sampled point (RANDOM, LHS, SOBOL). Lets startup code tell the user's
 * seeds (CONFIG, WARM) from sampled candidates. */
struct InitOrigin {
    enum class Kind { Config, Warm, Random, Lhs, Sobol };
    Kind        kind = Kind::Config;
    std::string file;                          // WARM: the seed file as written in INIT

    /* "CONFIG" | "WARM <file>" | "RANDOM" | "LHS" | "SOBOL" */
    std::string name() const {
        switch (kind) {
            case Kind::Config: return "CONFIG";
            case Kind::Warm:   return "WARM " + file;
            case Kind::Random: return "RANDOM";
            case Kind::Lhs:    return "LHS";
            case Kind::Sobol:  return "SOBOL";
        }
        return "CONFIG";
    }
    /* A user seed (CONFIG or WARM), as opposed to a sampled point. */
    bool is_seed() const { return kind == Kind::Config || kind == Kind::Warm; }
};

/* StrategyType -- selected by `STRATEGY { TYPE <X> }` in the parser.
 * Drives build_run_policy() in run_policy.h to assemble the right
 * RunPolicy decorator stack. */
enum class StrategyType {
    PLAIN,        // single-shot run of spec.lib/algo (no decorators)
    MULTISTART,   // MultiStartPolicy (N independent starts of spec.lib/algo)
    PRESTAGE,     // StagedPolicy (one EXPLORE pass -> top-K refined by spec.lib/algo)
};

/* MultiStartSpec -- knobs for `STRATEGY { TYPE MULTISTART }`. */
struct MultiStartSpec {
    std::size_t n_starts = 1;                      // N_STARTS -- count
    std::size_t parallel = 1;                      // PARALLEL -- concurrent runs
    std::optional<std::size_t> budget_per_start;   // BUDGET_PER_START override
};

/* RestartSpec -- `STRATEGY { RESTART/ENDRESTART }` decorator.
 * build_run_policy wraps every STRATEGY TYPE (PLAIN/MULTISTART/PRESTAGE)
 * in RestartPolicy; on HPO_Host::StagnationStop it re-runs the whole inner
 * policy from a fresh init (see RestartPolicy in run_policy.h). */
struct RestartSpec {
    std::size_t after_stagnation = 0;              // 0 disables
    std::size_t max              = 0;              // cap on restart count
    InitStrategySpec init_strategy;                // INIT chain to refresh on restart;
                                                   // its SEED draws round r with SEED + r
    /* The chain's seeds -- its CONFIG and WARM points, the user's starting
     * points as opposed to sampled ones -- resolved once at startup
     * (HPOTargets) when the block is active: a WARM file is read, and its
     * EMBF0/RHOFORM rules applied, when the target is read, and the
     * density gates check these points as seeds. seed_origins is parallel
     * to seed_points. Every round draws the chain afresh (RestartPolicy). */
    std::vector<std::vector<double>> seed_points;
    std::vector<InitOrigin>          seed_origins;

    /* RESTART takes effect only when both AFTER_STAGNATION and MAX are
     * positive. The one rule for build_run_policy() and for the
     * multi-rank dispatch, which refuses an active RESTART. */
    bool active() const noexcept { return after_stagnation > 0 && max > 0; }
};

/* NoiseSpec -- `OPTIMIZER { NOISE/ENDNOISE }` block. */
struct NoiseSpec {
    bool        autodetect    = false;             // AUTO TRUE -- derive PARAM fd_step
                                                   // from box diagonal in transformed
                                                   // space, for the finite differences
                                                   // of the STRATEGY-level phase
                                                   // (HPOTargets; refused at startup
                                                   // where it takes none)
    std::size_t calibration_n = 0;                 // CALIBRATION N -- repeat seed point
                                                   // N times to surface objective noise
                                                   // and suggest FTOL_REL. 0 = off.
};

/* StepOverride -- `STEP <KEY>[(indices)] <frac>`: per-key / per-index
 * override of the OPTIMIZER-block STEP fraction. Parsed by
 * dispatch_opt_attr into OptimizerSpec::step_overrides; resolved against
 * the search space (unknown key / non-OPTIM'd index are hard errors)
 * when the per-dimension initial-step plan is built. An empty `indices`
 * means every OPTIM'd index of `key`. */
struct StepOverride {
    std::string              key;
    std::vector<std::size_t> indices;   // 1-based OPTIM indices; empty = all
    double                   frac = 0.0;
};

/* ---------------- Optimizer description --------------------------- */
struct OptimizerSpec {
    OptimizerSpec() = default;

    /* Deep-copy. Required by RunPolicy decorators (MULTISTART, PRESTAGE)
     * that create per-start spec snapshots with a single replaced init
     * point. The std::unique_ptr members block the implicit copy ctor;
     * clone them explicitly. */
    OptimizerSpec(const OptimizerSpec& o)
        : strategy_type(o.strategy_type),
          lib(o.lib), algo(o.algo),
          inner(o.inner ? std::make_unique<OptimizerSpec>(*o.inner) : nullptr),
          prestage(o.prestage ? std::make_unique<OptimizerSpec>(*o.prestage) : nullptr),
          vector_storage(o.vector_storage), pop(o.pop),
          num_threads(o.num_threads), seed(o.seed), step(o.step),
          step_overrides(o.step_overrides),
          grad_strategy(o.grad_strategy),
          grad_strategy_resolved(o.grad_strategy_resolved), stop(o.stop),
          param(o.param), param_raw(o.param_raw),
          param_tokens(o.param_tokens),
          fd_step_from_noise_auto(o.fd_step_from_noise_auto),
          anneal_observer(o.anneal_observer), low(o.low), high(o.high),
          parameter_labels(o.parameter_labels), is_int(o.is_int),
          log_scaled(o.log_scaled), log_threshold(o.log_threshold),
          space(o.space), scale(o.scale),
          optim_init(o.optim_init),
          optim_init_origin(o.optim_init_origin),
          failscore(o.failscore),
          failscore_grading(o.failscore_grading),
          init_strategy_spec(o.init_strategy_spec),
          noise(o.noise),
          multistart(o.multistart),
          stage_topk(o.stage_topk),
          restart(o.restart),
          block_line(o.block_line), block_line_text(o.block_line_text) {}
    OptimizerSpec& operator=(const OptimizerSpec& o) {
        if (this != &o) {
            OptimizerSpec tmp(o);
            *this = std::move(tmp);
        }
        return *this;
    }
    OptimizerSpec(OptimizerSpec&&) = default;
    OptimizerSpec& operator=(OptimizerSpec&&) = default;

    /* --------- strategy selection -------------------------------- */
    StrategyType strategy_type = StrategyType::PLAIN;

    /* --------- driver (the optimiser run by this spec) ------------ */
    std::string  lib   = "NOT SET";   // DLIB | NLOPT | TADAH
    std::string  algo  = "NOT SET";

    /* INNER/ENDINNER sub-block -- NLopt's local sub-algorithm
     * (used only with MLSL/G_MLSL_LDS/AUGLAG, which delegate inner
     * work to a local optimiser via set_local_optimizer). It has no
     * seed: NLopt draws the INNER search from the enclosing run's random
     * stream (make_nlopt), so HPOTargets rejects SEED inside INNER. It
     * has no GRAD_STRATEGY either: MLSL/AUGLAG hand the local search the
     * gradient the enclosing block's GRAD_STRATEGY obtains, so HPOTargets
     * rejects GRAD_STRATEGY inside INNER too. validate_nlopt_spec (and so
     * make_nlopt) refuses an INNER spec that carries either.
     * The v1.0.0-v1.1.1 keyword `LOCAL/ENDLOCAL` was renamed to INNER. */
    std::unique_ptr<OptimizerSpec> inner;

    /* When strategy_type == PRESTAGE: the EXPLORE block content (one-pass
     * global exploration). build_run_policy() runs `*prestage` first,
     * picks the top-K (loss, params) pairs, and refines them via the
     * outer spec's lib/algo. */
    std::unique_ptr<OptimizerSpec> prestage;

    /* --------- algorithm-specific parameters ---------------------- */
    std::optional<unsigned> vector_storage, pop, num_threads;
    /* 64-bit so SEED values at or above 2^32 are preserved exactly;
     * every consumer already widens to std::uint64_t. */
    std::optional<std::uint64_t> seed;
    /* STEP: initial step of derivative-free NLopt algorithms. Consumed
     * only by backends whose OptimizerCaps::step_use is a consumer
     * (scale_validation.h rejects it everywhere else). */
    std::optional<double> step;
    /* STEP <KEY>[(indices)] <frac> overrides (see StepOverride). */
    std::vector<StepOverride> step_overrides;

    /* Gradient backend used by gradient-based optimizers (DLIB:BFGS,
     * CERES:LBFGS/...). std::nullopt means default (NUMERICAL). The
     * field lives at OptimizerSpec scope (not PARAM) because it is
     * shared across optimizer families. See GradStrategy doc above. */
    std::optional<GradStrategy> grad_strategy;
    /* grad_strategy was not written in the block but set by the parser:
     * with PC_OBSERVABLE an unset GRAD_STRATEGY resolves to ANALYTICAL in
     * a block that runs an optimiser and obtains gradients (the
     * STRATEGY-level block, EXPLORE; HPOTargets::read). The journal
     * says so. */
    bool grad_strategy_resolved = false;

    StopCriteria stop;
    ParamMap     param;             // algorithm-specific knobs
    RawParamMap  param_raw;         // PARAM values that were not numeric
    ParamTokenMap param_tokens;     // full token vector, multi-token PARAMs only
    /* PARAM fd_step was derived by the NOISE block's AUTO (HPOTargets), not
     * written: the startup refusal of a box too narrow for the step, and
     * the journal, name it as the step's source (fd_stencil.h). */
    bool          fd_step_from_noise_auto = false;

    /* Test/tooling observation seam for the ANNEAL family (see
     * anneal::ChainObserver in anneal_core.h). Never config-parseable;
     * not owned. */
    anneal::ChainObserver* anneal_observer = nullptr;

    /* -------- bounds, integer mask, transform ---------------------- */
    /* low/high are INTERNAL (optimiser-space) bounds -- derived caches
     * filled by bind_space() from the SearchSpace (or, for hand-built
     * specs, by the analyse_logscale() shim). Every backend iterates on
     * [low, high]; in()/out() map REAL <-> INTERNAL. */
    std::vector<double> low, high;
    std::vector<std::string> parameter_labels; // canonical flattened OPTIM names
    std::vector<bool>   is_int;
    /* log_scaled[i] == true iff dim i's shape is LOG (ln). Kept as a
     * derived cache for display code and hand-built test specs; ALL
     * coordinate arithmetic must go through in()/out()/space, which also
     * cover the ASINH shape (log_scaled is false for it). */
    std::vector<bool>   log_scaled;
    std::optional<double> log_threshold;    // LOG_HP ratio; OFF = std::nullopt
    /* The search space this spec iterates in (shared with every sub-spec
     * and stage-spec clone). Null only for hand-built specs that never
     * called bind_space()/analyse_logscale() -- in()/out() then fall back
     * to the log_scaled mask. */
    SearchSpacePtr        space;
    ScalePolicy           scale;            // OPTIM_SCALE + LOG_HP as parsed
    std::vector<std::vector<double>> optim_init;
    /* Provenance of each optim_init point, as resolved at startup by
     * HPOTargets (the INIT chains of OUTER, PRESTAGE and PRESTAGE.INNER; a
     * phase that shares the OUTER's points, share_init_with_nested_phases,
     * shares their origins) and by a RESTART refresh. MULTISTART and the
     * PRESTAGE refinement, which run from single points or exploration
     * candidates, do not maintain it: a consumer uses it only when its
     * size equals optim_init's. Empty for hand-built specs. */
    std::vector<InitOrigin> optim_init_origin;

    /* FAILSCORE from the enclosing HPOTargets, copied in so optimisers
     * can tell a real loss from a failed-evaluation sentinel (the
     * ANNEAL family excludes FAILSCORE losses from its T0 statistics).
     * Unset only in bare specs built outside HPOTargets (unit tests). */
    std::optional<double> failscore;

    /* FAILSCORE_GRADING of the enclosing HPOTargets, after the
     * TADAH_FAILSCORE_GRADING override (HPOTargets::set_failscore_grading):
     * the finite differences take the differences at a graded failure as
     * they are, a direction out of the failure region, where without
     * grading a failed loss at the point fails the gradient
     * (gradient_provider.h). false in bare specs built outside HPOTargets. */
    bool failscore_grading = false;

    /* -------- top-level OPTIMIZER blocks (parsed children) -------- */
    InitStrategySpec init_strategy_spec;    // INIT/ENDINIT block
    NoiseSpec        noise;                 // NOISE/ENDNOISE block

    /* -------- strategy-specific sub-spec / decorator knobs -------- */
    MultiStartSpec   multistart;            // STRATEGY { TYPE MULTISTART }
    std::size_t      stage_topk = 1;        // STRATEGY { TYPE PRESTAGE TOPK ... }
    RestartSpec      restart;                // STRATEGY { RESTART/ENDRESTART }

    /* -------- where the configuration completes this optimiser -------
     * The line of the hpotarget file at which HPOTargets::read() checks
     * the block that configures this optimiser, and its text: ENDOPTIMIZER
     * for the STRATEGY-level spec (whose copies are the MULTISTART starts
     * and the PRESTAGE refinement, a REFINE block's keys included),
     * ENDEXPLORE for the PRESTAGE exploration, ENDINNER for an INNER block
     * (inner_run_spec gives NLopt's default local optimiser the line of
     * the block it runs in). 0 for a spec built outside the parser. The
     * startup refusals of a phase's settings name it (with_block_line,
     * phase_specs.h), as the parser's refusals of the block do. */
    std::size_t      block_line = 0;
    std::string      block_line_text;

    /* Bind the search space: low/high become the INTERNAL bounds, is_int,
     * log_scaled and parameter_labels are derived from it. Called by
     * HPOTargets for the outer spec and every sub-spec. */
    void bind_space(SearchSpacePtr s)
    {
        space = std::move(s);
        const std::size_t n = space ? space->dim() : 0;
        low.resize(n); high.resize(n);
        is_int.assign(n, false);
        log_scaled.assign(n, false);
        for (std::size_t i = 0; i < n; ++i) {
            low[i]  = space->internal_lo(i);
            high[i] = space->internal_hi(i);
            is_int[i] = space->at(i).is_int;
            log_scaled[i] = (space->at(i).tf.kind == ScaleKind::Log);
        }
        parameter_labels = space ? space->labels() : std::vector<std::string>{};
    }

    /* The nested phases without an INIT chain of their own -- the
     * PRESTAGE exploration and its INNER -- start from this spec's INIT
     * points (optim_init). HPOTargets applies this once the OUTER chain is
     * resolved, and RestartPolicy again after drawing the fresh points of a
     * restart, so a restarted PRESTAGE explores from them. A phase with
     * its own INIT chain keeps its own points. */
    void share_init_with_nested_phases()
    {
        if (!prestage) return;
        if (prestage->init_strategy_spec.chain.empty()) {
            prestage->optim_init = optim_init;
            prestage->optim_init_origin = optim_init_origin;
        }
        if (prestage->inner &&
            prestage->inner->init_strategy_spec.chain.empty()) {
            prestage->inner->optim_init = optim_init;
            prestage->inner->optim_init_origin = optim_init_origin;
        }
    }

    /* The number of starts the strategy runs from this spec's SEED, start i
     * with SEED + i: the N_STARTS starts of MULTISTART, the TOPK
     * candidates a PRESTAGE refinement multistarts (StagedPolicy), and the
     * one run of PLAIN. N_STARTS counts only under MULTISTART (the parser
     * rejects it under another TYPE; a spec built by hand may carry one). */
    std::uint64_t seeded_starts() const noexcept
    {
        switch (strategy_type) {
            case StrategyType::MULTISTART:
                return std::max<std::uint64_t>(1, multistart.n_starts);
            case StrategyType::PRESTAGE:
                return std::max<std::uint64_t>(1, stage_topk);
            case StrategyType::PLAIN:
            default:
                return 1;
        }
    }

    /* The seeds of RESTART round `round` (1..MAX), derived from `round0`:
     * the spec as parsed, whose SEEDs round 0 runs with. Every phase with
     * a SEED gets a new one, so a restarted seeded stochastic optimiser
     * draws a new random sequence instead of replaying round 0's from its
     * new start point:
     *  - this (outer) spec: SEED + round*seeded_starts(). Start i of the
     *    strategy runs with SEED + i, so round r's starts use
     *    SEED + r*seeded_starts() + i, which no other round uses (SEED 7,
     *    TOPK 3: {7,8,9}, {10,11,12}, {13,14,15});
     *  - the EXPLORE spec: SEED + round.
     * An INNER spec has no SEED: NLopt draws the INNER search from the
     * random stream of the enclosing block's run, which that block's
     * derived SEED seeds (make_nlopt). A phase without SEED stays without
     * one. The sums wrap modulo 2^64, like MULTISTART's. RestartPolicy
     * applies this to its copy of the spec on every restart, together with
     * share_init_with_nested_phases(); round 0 keeps every SEED. */
    void derive_restart_seeds(const OptimizerSpec& round0, std::size_t round)
    {
        const std::uint64_t r = static_cast<std::uint64_t>(round);
        const auto derive = [r](const std::optional<std::uint64_t>& s,
                                std::uint64_t stride) {
            return s ? std::optional<std::uint64_t>(*s + r * stride)
                     : std::nullopt;
        };
        seed = derive(round0.seed, round0.seeded_starts());
        if (prestage && round0.prestage)
            prestage->seed = derive(round0.prestage->seed, 1);
    }

    /* Shim for hand-built specs (tests): low/high hold REAL bounds on
     * entry; builds an UNNORMALISED search space (OPTIM_SCALE NONE) with
     * the LOG_HP rule from log_threshold and binds it, so low/high end
     * up ln-transformed for qualifying dims exactly as before.
     * Production code builds the space in HPOTargets; do not call this
     * on a spec that already has one. */
    void analyse_logscale()
    {
        std::vector<DimSpec> dims(low.size());
        for (std::size_t i = 0; i < low.size(); ++i) {
            dims[i].label = (i < parameter_labels.size() && !parameter_labels[i].empty())
                            ? parameter_labels[i] : logidx::dim_label(i);
            dims[i].key = dims[i].label;
            dims[i].lo = low[i];
            dims[i].hi = high[i];
            dims[i].is_int = (i < is_int.size()) && is_int[i];
        }
        ScalePolicy pol;
        pol.log_ratio = log_threshold;
        pol.normalise = false;
        scale = pol;
        bind_space(std::make_shared<const SearchSpace>(
            SearchSpace::build(std::move(dims), pol)));
    }

    /* REAL -> INTERNAL and INTERNAL -> REAL for std::vector<double> or a
     * dlib column matrix. Through the search space when bound; the
     * legacy ln/exp mask otherwise (specs built by hand in tests). */
    template<typename Vec>
    Vec in(const Vec& x) const
    {
        if (space) return space->to_internal(x);
        Vec y = x;
        for (std::size_t i = 0; i < ss_detail::len(y); ++i)
            if (i < log_scaled.size() && log_scaled[i])
                ss_detail::elem(y, i) = std::log(ss_detail::elem(y, i));
        return y;
    }
    template<typename Vec>
    Vec out(const Vec& x) const
    {
        if (space) return space->to_real(x);
        Vec y = x;
        for (std::size_t i = 0; i < ss_detail::len(y); ++i)
            if (i < log_scaled.size() && log_scaled[i])
                ss_detail::elem(y, i) = std::exp(ss_detail::elem(y, i));
        return y;
    }

    /* ------------- capability print-outs -------------------------- */
    static void print_caps(const std::string& lib,
                           const std::string& algo);
};

/* Seed anchoring for one optimiser run (SearchSpace::with_seed_anchors):
 * a copy of `sp` whose space maps every start point in sp.optim_init back
 * to exactly its loaded doubles, or std::nullopt when there is no space or
 * no in-box seed coordinate. run_inner() applies it to every run, so
 * CONFIG and WARM seeds, MULTISTART starts, RESTART refreshes and PRESTAGE
 * top-K points are evaluated exactly where they came from.
 * An optimiser that perturbs its start by rounding before evaluating it
 * (NLopt BOBYQA/COBYLA rescale unequal initial steps, u -> (u/s)*s, by at
 * most 1 ulp(u0); PRAXIS maps the start through atanh/tanh, a few ulp of
 * the internal box) lands within the anchor's window
 * (SearchSpace::seed_anchor_tolerance) and is anchored too -- except,
 * possibly, PRAXIS on an OPTIM_SCALE NONE dimension whose box reaches
 * 2^24, for a seed of smaller magnitude than the box: PRAXIS rounds its
 * start by a few ulp of the box's midpoint and half-width, while the
 * window there is capped (kSeedAnchorMaxTol) or 2 ulp of the seed
 * (measured with NLopt 2.10: seeds up to ~0.3 of max(|lo|, |hi|) on a
 * [0, W] box, up to ~0.6, rarely 0.8, on a box straddling 0; only on boxes
 * of 2^25 or more). A start NLopt moves on purpose (PRAXIS clamps a start
 * on a bound 1e-12 inside, BOBYQA shifts a start within its initial step
 * of a bound) keeps the plain mapping. */
std::optional<OptimizerSpec> anchor_seed_points(const OptimizerSpec& sp);

}   // namespace

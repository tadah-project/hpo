#pragma once
#include <unordered_map>
#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <memory>
#include <limits>
#include <cmath>
#include <unordered_set>

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
 *   NUMERICAL  -- Ceres DynamicNumericDiffCostFunction; finite differences
 *                on full forward passes through the host.
 *   ANALYTICAL -- closed-form analytical gradient. The HPO host injects
 *                a per-theta derivative callback (descriptor + cutoff +
 *                kernel closed forms chained through the IFT-back-
 *                substitution on the cached Cholesky factor of the
 *                regression solve). Strict mode: any theta index without a
 *                registered analytical derivative throws a diagnostic
 *                naming the missing entry. Use HYBRID to fall back.
 *   HYBRID     -- analytical-where-registered + central-difference
 *                fallback for unregistered indices, evaluated in one
 *                gradient(x, g) call. Used when LAMMPS-derived loss
 *                components (`sims_loss`) or tabulated descriptors
 *                (EAM SETFL) participate in theta.
 *
 * Configured at OPTIMIZER block level via `GRAD_STRATEGY {NUMERICAL|
 * ANALYTICAL|HYBRID}`. Default (unset) = NUMERICAL. The Phase-1 keyword
 * `AUTODIFF` is no longer accepted; the parser emits a tailored error
 * pointing at ANALYTICAL. */
enum class GradStrategy { Numerical, Analytical, Hybrid };

using ParamMap = std::unordered_map<std::string,double>;

/* InitStrategySpec -- parsed `INIT/ENDINIT` block.
 * Resolved into a polymorphic InitStrategy by build_init_strategy()
 * after bounds + log-scale are settled. */
struct InitStrategySpec {
    struct Entry {
        std::string method;                    // CONFIG, RANDOM, LHS, SOBOL, WARM, PRIOR, NOISE_AUTO
        std::vector<std::string> args;         // method-specific tokens
    };
    std::vector<Entry> chain;                  // STRATEGY lines (in order, inside INIT block)
    std::optional<std::size_t> k_total;        // K_TOTAL cap (default = sum of strategy default_k)
    std::optional<std::uint64_t> seed;         // SEED for stochastic strategies
    std::string warm_on_mismatch = "warn";     // warn | error | skip (used by WarmInitStrategy)
};

/* StrategyType -- selected by `STRATEGY { TYPE <X> }` in the parser.
 * Drives build_run_policy() in run_policy.h to assemble the right
 * RunPolicy decorator stack. */
enum class StrategyType {
    PLAIN,        // single-shot run of spec.lib/algo (no decorators)
    BASINS,       // BasinsPolicy (anytime/interruptible, basin pool, ladders)
    MULTISTART,   // MultiStartPolicy (N independent starts of spec.lib/algo)
    PRESTAGE,     // StagedPolicy (one EXPLORE pass -> top-K refined by spec.lib/algo)
    BASINSV2,     // BasinsV2Policy (priority-driven scheduler, per-basin lifecycle,
                  // adaptive REFINE termination, explicit duplicate merging,
                  // Sobol+filtered EXPLORE, coarse path memory, smart checkpoint).
                  // Additive -- existing BASINS code paths untouched.
};

/* Forward decl needed because BasinsSpec holds unique_ptr<OptimizerSpec>. */
struct OptimizerSpec;

/* BasinsSpec -- knobs for `STRATEGY { TYPE BASINS }`.
 * Carries the EXPLORE/REFINE stage sub-specs, per-stage tolerance
 * ladders + budget caps, and BASINS-wide pool/checkpoint knobs.
 * Resolved at hpo_run() time into a BasinsPolicy<Vec, Runner> by
 * build_run_policy().
 *
 * Defaults give a functional config with just `STRATEGY { TYPE BASINS }`:
 *   EXPLORE = NLOPT GN_CRS2_LM, REFINE = NLOPT LN_BOBYQA,
 *   default 6-rung ladders, 32-basin pool. */
struct BasinsSpec {
    BasinsSpec();
    BasinsSpec(const BasinsSpec& o);
    BasinsSpec& operator=(const BasinsSpec& o);
    BasinsSpec(BasinsSpec&&) = default;
    BasinsSpec& operator=(BasinsSpec&&) = default;
    ~BasinsSpec() = default;

    /* EXPLORE / REFINE stage sub-specs.
     * - explore: global stage. Default NLOPT GN_CRS2_LM
     *   (population-based, stochastic, noise-tolerant). May carry an
     *   INNER block for MLSL/AUGLAG inner local sub-optimizer.
     * - refine: per-basin local stage. Default NLOPT LN_BOBYQA
     *   (derivative-free quadratic trust-region, bound-aware).
     *   LN_BOBYQA replaces an earlier LN_SBPLX default -- SBPLX
     *   collapses ("simplex too small in dimension X" -> masked
     *   XTOL_REACHED) when a parameter approaches a bound, which
     *   silently ends a basin's refinement without progress.
     * unique_ptr because OptimizerSpec is not yet fully defined here
     * and because BasinsSpec is itself a member of OptimizerSpec. */
    std::unique_ptr<OptimizerSpec> explore;
    std::unique_ptr<OptimizerSpec> refine;

    /* Per-stage FTOL ladders. Empty -> auto-derive from NOISE_CALIBRATION
     * (or 1e-9 floor). Populated by `EXPLORE { FTOL_LADDER ... }` and
     * `REFINE { FTOL_LADDER ... }` in the parser. */
    std::vector<double> explore_ftol_ladder;
    std::vector<double> refine_ftol_ladder;

    /* Per-stage budget caps & shape. Each stage has its own:
     *   - BUDGET_MAX_EVALS  upper cap on evals consumed by one stage
     *   - BUDGET_MIN_EVALS  lower cap (used by stagnation cap math)
     *   - BUDGET_RATE_FACTOR  shape coefficient for the per-basin
     *     stagnation budget (cap = max(min_evals, max_evals * rate)). */
    std::size_t explore_budget_min_evals    = 30;
    std::size_t explore_budget_max_evals    = 5000;
    double      explore_budget_rate_factor  = 0.5;
    std::size_t refine_budget_min_evals     = 30;
    std::size_t refine_budget_max_evals     = 5000;
    double      refine_budget_rate_factor   = 0.5;

    /* BASINS-wide knobs (cross-stage / pool-level / process-level). */
    double      basin_dedup_dist            = 0.05;
    std::size_t basin_pool_max              = 32;
    std::size_t stop_no_improve_rounds      = 5;

    /* Checkpoint cadence (one or both can be set). 0 disables. */
    double      checkpoint_seconds          = 0.0;
    std::size_t checkpoint_evals            = 0;

    /* EXPLORE re-seeding: number of top-loss basins to pre-evaluate at the
     * start of each round k>1 EXPLORE stage. Default 1 = old behaviour
     * (only the best basin's params seed the optimiser). With N>1, the
     * top-N basins by loss are evaluated via host() (which feeds the
     * top-K capture and BasinPool), and the lowest-loss of them is used
     * as the optimiser start. */
    std::size_t seed_top_n = 1;

    /* Noise-floor ladder clamp (opt-in). When true and noise_floor is
     * known (populated post-NOISE_CALIBRATION by hpo_run), strictly
     * trim explore_ftol_ladder/refine_ftol_ladder rungs below noise_floor.
     * Default OFF because NOISE_CALIBRATION at the seed point may
     * report a misleading rel_spread when the seed is a poor potential
     * (BAD_POT clamps inflate spread, or seed is below the BAD_POT
     * threshold giving 0 spread). User opts in once they trust the
     * calibration (e.g. from a converged WARM seed). */
    bool                  clamp_to_noise_floor = false;
    std::optional<double> noise_floor;
};

/* BasinsV2Spec -- knobs for `STRATEGY { TYPE BASINSV2 }`.
 *
 * BASINSv2 is an additive strategy that replaces v1's per-round
 * GLOBAL->LOCAL sweep with an event-driven scheduler picking ONE work
 * item per iteration: REFINE (priority-selected), EXPLORE (diversity),
 * VERIFY_KILL (cheap tickle on inactive basins), or MERGE_PASS
 * (cadence-triggered, no LAMMPS evals). Each basin carries a lifecycle
 * state, an EMA of recent improvement, its own tol cursor, and a capped
 * coarse trajectory.
 *
 * v1's BasinsSpec is kept intact; v2 lives in this parallel struct so a
 * minimal `STRATEGY { TYPE BASINSV2 }` config is functional.
 *
 * Defaults give a working config with no other keys:
 *   EXPLORE = NLOPT GN_CRS2_LM, REFINE = NLOPT LN_BOBYQA,
 *   default 6-rung ladders, 24-basin pool. */
struct BasinsV2Spec {
    BasinsV2Spec();
    BasinsV2Spec(const BasinsV2Spec& o);
    BasinsV2Spec& operator=(const BasinsV2Spec& o);
    BasinsV2Spec(BasinsV2Spec&&) = default;
    BasinsV2Spec& operator=(BasinsV2Spec&&) = default;
    ~BasinsV2Spec() = default;

    /* EXPLORE / REFINE stage sub-specs. Same conventions as BasinsSpec. */
    std::unique_ptr<OptimizerSpec> explore;
    std::unique_ptr<OptimizerSpec> refine;

    /* Per-stage FTOL ladders. Empty -> auto-derive from NOISE_CALIBRATION. */
    std::vector<double> explore_ftol_ladder;
    std::vector<double> refine_ftol_ladder;

    /* Per-stage budget caps & shape. Same names/semantics as BasinsSpec. */
    std::size_t explore_budget_min_evals    = 30;
    std::size_t explore_budget_max_evals    = 5000;
    double      explore_budget_rate_factor  = 0.5;
    std::size_t refine_budget_min_evals     = 30;
    std::size_t refine_budget_max_evals     = 5000;
    double      refine_budget_rate_factor   = 0.5;

    /* Pool / dedup / merge knobs. */
    double      basin_dedup_dist            = 0.05;
    double      basin_merge_dist            = 0.10;   // periodic deep-merge threshold
    double      basin_merge_loss_threshold  = 0.05;   // relative loss similarity for merge
    std::size_t basin_pool_max              = 24;
    std::size_t archive_max                 = 48;     // cap on MERGED+KILLED basins

    /* Lifecycle / scheduler knobs. */
    std::size_t inactive_after_evals        = 0;      // 0 = derive from refine_budget_max_evals (ratio 0.75)
    std::size_t verify_cadence              = 30;     // iters between verify-kill checks per INACTIVE basin
    std::size_t verify_max_attempts         = 2;
    std::size_t verify_evals_cap            = 0;      // 0 = derive from dim (10 * dim)
    std::size_t merge_cadence               = 25;     // iters between deep-merge passes
    std::size_t explore_kick_cadence        = 50;     // iters between forced fresh-region EXPLOREs
    std::size_t summary_cadence             = 25;
    std::size_t starvation_cap              = 200;    // iters since basin's last_iteration -> force-pick
    std::size_t stop_no_improve_iterations  = 200;    // global stop after N iters without best improvement

    /* Adaptive REFINE termination. plateau_eps_factor * tol = relative
     * improvement threshold for stagnation detection over the per-basin
     * window. The per-basin stagnation cap also clamps to
     * max(refine_budget_min_evals, 100 * dim) to protect slow descent. */
    double      plateau_eps_factor          = 0.5;

    /* EXPLORE distance floor: drop Sobol candidates within this normalised
     * L2 of any known basin's coarse path. */
    double      explore_dist_floor          = 0.05;

    /* Path recording (Phase 4). 0 path_max_points disables recording. */
    std::size_t path_max_points             = 16;
    double      path_loss_eps               = 0.01;
    double      path_dist_eps               = 0.01;

    /* Priority weights -- six-tuple in fixed order: improve, neglect,
     * distance, tol_gap, state, age. */
    double      w_improve   = 2.0;
    double      w_neglect   = 0.5;
    double      w_distance  = 0.3;
    double      w_tol_gap   = 0.5;
    double      w_state     = 1.0;
    double      w_age       = 0.05;

    /* Diagnostics. */
    bool        verbose_diagnostics = false;

    /* Checkpoint cadence. Same conventions as BasinsSpec. */
    double      checkpoint_seconds          = 0.0;
    std::size_t checkpoint_evals            = 0;

    /* Noise-floor ladder clamp (opt-in, same as BasinsSpec). */
    bool                  clamp_to_noise_floor = false;
    std::optional<double> noise_floor;
};

/* MultiStartSpec -- knobs for `STRATEGY { TYPE MULTISTART }`. */
struct MultiStartSpec {
    std::size_t n_starts = 1;                      // N_STARTS -- count
    std::size_t parallel = 1;                      // PARALLEL -- concurrent runs
    std::optional<std::size_t> budget_per_start;   // BUDGET_PER_START override
};

/* RestartSpec -- `STRATEGY { RESTART/ENDRESTART }` decorator.
 * Composes with TYPE PLAIN/MULTISTART/PRESTAGE; not with BASINS. */
struct RestartSpec {
    std::size_t after_stagnation = 0;              // 0 disables
    std::size_t max              = 0;              // cap on restart count
    InitStrategySpec init_strategy;                // INIT chain to refresh on restart
};

/* NoiseSpec -- `OPTIMIZER { NOISE/ENDNOISE }` block. */
struct NoiseSpec {
    bool        autodetect    = false;             // AUTO TRUE -- derive PARAM fd_step
                                                   // from box diagonal in transformed
                                                   // space (DLIB:BFGS only)
    std::size_t calibration_n = 0;                 // CALIBRATION N -- repeat seed point
                                                   // N times to surface objective noise
                                                   // and suggest FTOL_REL. 0 = off.
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
          vector_storage(o.vector_storage), pop(o.pop), seed(o.seed),
          num_threads(o.num_threads), step(o.step),
          grad_strategy(o.grad_strategy), stop(o.stop),
          param(o.param), low(o.low), high(o.high), is_int(o.is_int),
          log_scaled(o.log_scaled), log_threshold(o.log_threshold),
          optim_init(o.optim_init),
          init_strategy_spec(o.init_strategy_spec),
          noise(o.noise),
          multistart(o.multistart),
          stage_topk(o.stage_topk),
          restart(o.restart),
          basins(o.basins),
          basins_v2(o.basins_v2) {}
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
     * work to a local optimiser via set_local_optimizer).
     * The previous keyword `LOCAL/ENDLOCAL` was renamed to INNER to
     * avoid collision with BASINS' REFINE stage (the artist formerly
     * known as LOCAL_LIB/LOCAL_ALGO). */
    std::unique_ptr<OptimizerSpec> inner;

    /* When strategy_type == PRESTAGE: the EXPLORE block content (one-pass
     * global exploration). build_run_policy() runs `*prestage` first,
     * picks the top-K (loss, params) pairs, and refines them via the
     * outer spec's lib/algo. */
    std::unique_ptr<OptimizerSpec> prestage;

    /* --------- algorithm-specific parameters ---------------------- */
    std::optional<unsigned> vector_storage, pop, seed, num_threads;
    std::optional<double> step;

    /* Gradient backend used by gradient-based optimizers (DLIB:BFGS,
     * CERES:LBFGS/...). std::nullopt means default (NUMERICAL). The
     * field lives at OptimizerSpec scope (not PARAM) because it is
     * shared across optimizer families. See GradStrategy doc above. */
    std::optional<GradStrategy> grad_strategy;

    StopCriteria stop;
    ParamMap     param;             // algorithm-specific knobs

    /* -------- bounds, integer mask, log-scale --------------------- */
    std::vector<double> low, high;          // filled by HPOTargets
    std::vector<bool>   is_int;
    std::vector<bool>   log_scaled;         // same size as low/high
    std::optional<double> log_threshold;    // OFF = std::nullopt
    std::vector<std::vector<double>> optim_init;

    /* -------- top-level OPTIMIZER blocks (parsed children) -------- */
    InitStrategySpec init_strategy_spec;    // INIT/ENDINIT block
    NoiseSpec        noise;                 // NOISE/ENDNOISE block

    /* -------- strategy-specific sub-spec / decorator knobs -------- */
    MultiStartSpec   multistart;            // STRATEGY { TYPE MULTISTART }
    std::size_t      stage_topk = 1;        // STRATEGY { TYPE PRESTAGE TOPK ... }
    RestartSpec      restart;                // STRATEGY { RESTART/ENDRESTART }
    BasinsSpec       basins;                 // STRATEGY { TYPE BASINS }
    BasinsV2Spec     basins_v2;              // STRATEGY { TYPE BASINSV2 }

    /* call once after low/high are known */
    void analyse_logscale()
    {
        const double thr = log_threshold.value_or(std::numeric_limits<double>::infinity());
        log_scaled.resize(low.size(),false);
        for (size_t i=0;i<low.size();++i)
        {
            if ( low[i]>0 && high[i]/low[i] >= thr )
            {
                log_scaled[i]=true;
                low [i] = std::log(low [i]);
                high[i] = std::log(high[i]);
            }
        }
    }
    /* Helper to go in/out of transformed space */
    template<typename Vec>
    Vec in(const Vec& x)  const
    {   Vec y=x;  for(size_t i=0;i<y.size();++i) if(log_scaled[i]) y[i]=std::log(y[i]); return y; }
    template<typename Vec>
    Vec out(const Vec& x) const
    {   Vec y=x;  for(long int i=0;i<y.size();++i) if(log_scaled[i]) y(i)=std::exp(y(i)); return y; }

    /* ------------- capability print-outs -------------------------- */
    static void print_caps(const std::string& lib,
                           const std::string& algo);
};

/* BasinsSpec ctor / copy implementations need OptimizerSpec to be a
 * complete type -- defined here, after OptimizerSpec.
 *
 * The default ctor leaves explore/refine *null*. OptimizerSpec contains
 * BasinsSpec by value, and BasinsSpec contains OptimizerSpec via these
 * unique_ptrs -- eagerly allocating in the default ctor would cause
 * unbounded recursive default-construction. The parser allocates them
 * (via ensure_defaults) when TYPE BASINS is selected. */
inline BasinsSpec::BasinsSpec() = default;

/* Allocate default explore/refine sub-specs (NLOPT GN_CRS2_LM + LN_BOBYQA).
 * Called by the parser when STRATEGY TYPE BASINS is seen. Idempotent. */
inline void basins_spec_ensure_defaults(BasinsSpec& bs)
{
    if (!bs.explore) {
        bs.explore = std::make_unique<OptimizerSpec>();
        bs.explore->lib  = "NLOPT";
        bs.explore->algo = "GN_CRS2_LM";
    }
    if (!bs.refine) {
        bs.refine = std::make_unique<OptimizerSpec>();
        bs.refine->lib   = "NLOPT";
        bs.refine->algo  = "LN_BOBYQA";
    }
}

inline BasinsSpec::BasinsSpec(const BasinsSpec& o)
    : explore(o.explore ? std::make_unique<OptimizerSpec>(*o.explore) : nullptr),
      refine (o.refine  ? std::make_unique<OptimizerSpec>(*o.refine)  : nullptr),
      explore_ftol_ladder(o.explore_ftol_ladder),
      refine_ftol_ladder (o.refine_ftol_ladder),
      explore_budget_min_evals  (o.explore_budget_min_evals),
      explore_budget_max_evals  (o.explore_budget_max_evals),
      explore_budget_rate_factor(o.explore_budget_rate_factor),
      refine_budget_min_evals   (o.refine_budget_min_evals),
      refine_budget_max_evals   (o.refine_budget_max_evals),
      refine_budget_rate_factor (o.refine_budget_rate_factor),
      basin_dedup_dist     (o.basin_dedup_dist),
      basin_pool_max       (o.basin_pool_max),
      stop_no_improve_rounds(o.stop_no_improve_rounds),
      checkpoint_seconds   (o.checkpoint_seconds),
      checkpoint_evals     (o.checkpoint_evals),
      seed_top_n           (o.seed_top_n),
      clamp_to_noise_floor (o.clamp_to_noise_floor),
      noise_floor          (o.noise_floor)
{}

inline BasinsSpec& BasinsSpec::operator=(const BasinsSpec& o)
{
    if (this != &o) {
        BasinsSpec tmp(o);
        *this = std::move(tmp);
    }
    return *this;
}

/* BasinsV2Spec -- same forward-decl trick as BasinsSpec: definitions live
 * after OptimizerSpec is complete. */
inline BasinsV2Spec::BasinsV2Spec() = default;

/* Allocate default explore/refine sub-specs (NLOPT GN_CRS2_LM + LN_BOBYQA).
 * Called by the parser when STRATEGY TYPE BASINSV2 is seen. Idempotent. */
inline void basins_v2_spec_ensure_defaults(BasinsV2Spec& bs)
{
    if (!bs.explore) {
        bs.explore = std::make_unique<OptimizerSpec>();
        bs.explore->lib  = "NLOPT";
        bs.explore->algo = "GN_CRS2_LM";
    }
    if (!bs.refine) {
        bs.refine = std::make_unique<OptimizerSpec>();
        bs.refine->lib   = "NLOPT";
        bs.refine->algo  = "LN_BOBYQA";
    }
}

inline BasinsV2Spec::BasinsV2Spec(const BasinsV2Spec& o)
    : explore(o.explore ? std::make_unique<OptimizerSpec>(*o.explore) : nullptr),
      refine (o.refine  ? std::make_unique<OptimizerSpec>(*o.refine)  : nullptr),
      explore_ftol_ladder(o.explore_ftol_ladder),
      refine_ftol_ladder (o.refine_ftol_ladder),
      explore_budget_min_evals  (o.explore_budget_min_evals),
      explore_budget_max_evals  (o.explore_budget_max_evals),
      explore_budget_rate_factor(o.explore_budget_rate_factor),
      refine_budget_min_evals   (o.refine_budget_min_evals),
      refine_budget_max_evals   (o.refine_budget_max_evals),
      refine_budget_rate_factor (o.refine_budget_rate_factor),
      basin_dedup_dist          (o.basin_dedup_dist),
      basin_merge_dist          (o.basin_merge_dist),
      basin_merge_loss_threshold(o.basin_merge_loss_threshold),
      basin_pool_max            (o.basin_pool_max),
      archive_max               (o.archive_max),
      inactive_after_evals      (o.inactive_after_evals),
      verify_cadence            (o.verify_cadence),
      verify_max_attempts       (o.verify_max_attempts),
      verify_evals_cap          (o.verify_evals_cap),
      merge_cadence             (o.merge_cadence),
      explore_kick_cadence      (o.explore_kick_cadence),
      summary_cadence           (o.summary_cadence),
      starvation_cap            (o.starvation_cap),
      stop_no_improve_iterations(o.stop_no_improve_iterations),
      plateau_eps_factor        (o.plateau_eps_factor),
      explore_dist_floor        (o.explore_dist_floor),
      path_max_points           (o.path_max_points),
      path_loss_eps             (o.path_loss_eps),
      path_dist_eps             (o.path_dist_eps),
      w_improve(o.w_improve), w_neglect(o.w_neglect),
      w_distance(o.w_distance), w_tol_gap(o.w_tol_gap),
      w_state(o.w_state), w_age(o.w_age),
      verbose_diagnostics(o.verbose_diagnostics),
      checkpoint_seconds(o.checkpoint_seconds),
      checkpoint_evals(o.checkpoint_evals),
      clamp_to_noise_floor(o.clamp_to_noise_floor),
      noise_floor(o.noise_floor)
{}

inline BasinsV2Spec& BasinsV2Spec::operator=(const BasinsV2Spec& o)
{
    if (this != &o) {
        BasinsV2Spec tmp(o);
        *this = std::move(tmp);
    }
    return *this;
}

}   // namespace

#ifndef TADAH_HPO_HOST_H_
#define TADAH_HPO_HOST_H_

/*
 *  Thread-safe, "stateless" HPO_Host
 *  ---------------------------------
 *  -  All data produced during one evaluation live in EvalResult, not in the
 *     functor object; operator() is therefore const and thread-safe.
 *  -  Ordinary result logs are forwarded to a single writer through a
 *     blocking moodycamel queue. New-best potentials are committed
 *     synchronously and atomically before operator() returns.
 *  -  MPI code is protected by one global mutex so that threaded builds can
 *     still work.  Remove the mutex if you choose to forbid that mix.
 */

#include <atomic>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>

#include <tadah/core/checkpoint.h>          // for ShutdownRequested
#include <tadah/core/context.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/append_log.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/bound_audit.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/run_stop.h>          // StagnationStop, ExceptionStash
#include <tadah/hpo/stagnation_stop.h>    // BestPublications, throw_if_stagnated
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#if defined(TADAH_ENABLE_OPENMP)
#   include <omp.h>
#endif

#include <moodycamel/blockingconcurrentqueue.h>     // moodycamel
#include <tadah/hpo/hpo_log.h>

namespace tadah::hpo {

namespace detail {
/* The next HPO_Host generation of the process: 1, 2, ... (thread-safe; one
 * counter for every HPO_Host instantiation). It stamps the per-thread
 * trainer slot, so a host never reads a trainer another host built. */
inline std::uint64_t next_host_generation() {
    static std::atomic<std::uint64_t> last{0};
    return last.fetch_add(1, std::memory_order_relaxed) + 1;
}
}  // namespace detail

/* =========================================================================
 *  EvalResult - everything produced by *one* evaluation
 * ========================================================================= */
template<class ParamVec>
struct EvalResult
{
    std::size_t                            iter      = 0;
    std::size_t                            best_iter = 0;
    /* Global slot of the chain (PT replica / PA walker) that produced this
     * eval; set from warmstart::current_chain() at the top of operator().
     * Keys the per-chain warm-start cache so PT/PA results are independent
     * of fan-out execution order. 0 for serial / single-sequence optimisers. */
    std::size_t                            chain_id  = 0;
    /* Rank that evaluated this exact candidate. Persisted with chain_id and
     * iter for deterministic cross-process best-file tie breaking. */
    int                                    source_rank = 0;
    double                                 loss      = 0.0;
    double                                 loss_sum  = 0.0;
    std::size_t                            loss_count = 0;

    /* validation metrics */
    double emae=0, ermse=0, errmse=0, ersq=0, frmse=0, srmse=0;
    double emaeLoss=0, ermseLoss=0, errmseLoss=0, ersqLoss=0, frmseLoss=0, srmseLoss=0;
    /* per-dataset validation metrics (one entry per validation dataset) */
    std::vector<double> v_emae, v_ermse, v_errmse, v_ersq, v_frmse, v_srmse;
    std::vector<double> v_emaeLoss, v_ermseLoss, v_errmseLoss, v_ersqLoss, v_frmseLoss, v_srmseLoss;
    std::vector<std::vector<double>>       v_loss;    // [sim][loss_i]
    std::vector<std::vector<std::string>>  v_outvar;  // [sim][outvar_i]

    /* In-range (script, name, value) hot-var triples this eval produced.
     * Promoted into the warmstart seed map when this eval becomes the new
     * global best, so best_pot.tadah's `# HOT` lines reflect the best
     * trajectory (deterministic: the new-best point is serialized). */
    std::vector<std::tuple<std::string, std::string, double>> hot_produced;

    /* Graded-FAILSCORE classification of this eval (band == 0 => feasible).
     * Accumulated by run_lammps/sims_loss; when FAILSCORE_GRADING is ON and
     * band != 0, the final r.loss is graded_eval_loss(fail, FS, s0) instead
     * of the constant FAILSCORE. See pc_fail_grading.h. */
    PcFailInfo fail;

    /* A PC_LAMMPS simulation of this eval failed: run_lammps sets it on both
     * of its exits when the eval was aborted (a failed task, a hot value
     * outside its window, the probe gate) or a task could not start
     * (CREATE_FAIL, which does not abort). sims_loss then scores the eval
     * FAIL_SCORE (the graded loss under FAILSCORE_GRADING), whatever the
     * mean of its terms. */
    bool sims_failed = false;

    /* A loss term of this eval could not be computed: a PC_OBSERVABLE
     * term (obs_loss: no trained model, an observable that threw, a value
     * or a loss that is not finite) or a --varloss term without a value
     * (sims_loss). Its term is FAIL_SCORE in v_obs_loss / v_loss, and the
     * eval scores FAIL_SCORE (under FAILSCORE_GRADING, which classifies
     * it, the graded loss) whatever the mean of its terms, as for
     * sims_failed: that mean, one FAIL_SCORE term among usable ones, was a
     * usable-looking loss below 0.999 * FAIL_SCORE. */
    bool terms_failed = false;

    /* PC_OBSERVABLE (Tier 1/2) results, one entry per
     * trg_.observables[k]. v_obs_value[k] is the computed observable
     * (e.g. cohesive energy in eV/atom); v_obs_loss[k] is the loss
     * contribution evaluated through the observable's LossFunction
     * against its target/weight. Both empty when trg_.observables is
     * empty. */
    std::vector<double>                    v_obs_value;
    std::vector<double>                    v_obs_loss;

    std::chrono::milliseconds              wall_ms{};

    /* pc_lammps fan-out busy-time stats for THIS eval (Stage 1a
     * diagnostics). Filled by run_lammps from per-script timings;
     * consumed once by HPOLog::record_fanout at the journal step (pool
     * utilisation is derived there from the shared executor width and
     * elapsed wall, not per-eval). All zero when the eval ran no
     * pc_lammps scripts. */
    double                                 fanout_busy_ms  = 0.0; // Σ (create+run) ms
    std::size_t                            fanout_nscripts = 0;
    double                                 fanout_min_ms   = 0.0;
    double                                 fanout_max_ms   = 0.0;
    std::string                            fanout_slowest;        // slowest script name

    /* EMBF0 journal (DM_REAM + linear model only): the raw embedding value
     * at zero density, sum_m w_m g_m(0), of this eval's weights. Filled only
     * for eval 1 and for an eval that becomes the new best; NaN otherwise.
     * Diagnostic only: never feeds a loss, row or potential. */
    double                                 embf0_raw = std::numeric_limits<double>::quiet_NaN();

    ParamVec                               params;

    bool                                   is_best         = false;
    bool                                   was_nan_clamped = false;
    // bool                                   run_lammps = false;   // option for the future?
    core::Context                          pot_ctx;
};

/* =========================================================================
 *  HPO_Host  -  stateless evaluation functor
 * ========================================================================= */
template<class ParamVec, class LammpsRunner>
class HPO_Host
{
    using Result = EvalResult<ParamVec>;
    using Queue  = moodycamel::BlockingConcurrentQueue<Result>;

public:
    HPO_Host(core::Context master,
        HPOTargets&   trg,
        const std::string& val_cfg);

    double operator()(const ParamVec& p) const;      // thread-safe

    /** Analytical-gradient callback for GRAD_STRATEGY=ANALYTICAL/HYBRID.
     *
     *  Signature matches `AnalyticalGradFn<ParamVec>`:
     *    bool(const ParamVec& x, ParamVec& g, std::vector<int>& needs_fd)
     *
     *  Phase 2A.4: does not run a forward eval itself; it builds a
     *  param-ref table and consults the orchestrator
     *  (compute_dloss_dtheta). Until the per-theta analytical evaluation
     *  is wired (sub-task), every index is reported in needs_fd, which
     *  HYBRID then handles via masked central differences.
     *
     *  ANALYTICAL strict mode is therefore a no-op for production
     *  configs (the GradientProvider raises a tailored diagnostic on
     *  the first call), and callers should prefer HYBRID until the
     *  registered descriptors widen.
     *
     *  Thread-safety: same guarantees as operator(); the returned
     *  std::function captures `this` by pointer. Lifetime is bounded
     *  by the surrounding adapter call. */
    bool analytical_gradient(const ParamVec& x, ParamVec& g,
                             std::vector<int>& needs_fd) const;

    /** One-shot log on the first analytical_gradient call: prints
     *  GRAD_STRATEGY, total theta count, analytical count, and
     *  FD-fallback count. Idempotent across calls (atomic_flag gate).
     *  Surfaces hints for the three different "all-FD" routes:
     *    - cache_miss:  no trainer of this host's evaluation at x in
     *                   the thread's slot (before its first evaluation,
     *                   after a rejected candidate; transient)
     *    - ift_failed:  analytical helper returned false (IFT Cholesky
     *                   non-PD); rare numerical edge case
     *    - default:     orchestrator gate (STRESS/NORM/OALGO/LAMBDA/
     *                   MODEL[1]/use_srmse/PC_LAMMPS) rejected the run */
    void emit_grad_diagnostic(const std::vector<ParamRef>& prefs,
                              const std::vector<int>&      needs_fd,
                              bool                         cache_miss,
                              bool                         ift_failed = false) const;

    /* Test-only: reset the static comparator atomics, the evaluation
     * counter, the result writer and the stagnation watchdog to their
     * freshly-constructed values. Used by regression tests to simulate
     * process restart without spawning a new process. The per-eval writer
     * starts over too: rows an earlier host left queued or buffered are
     * dropped and the row cursor rewinds to eval 1, so the next host's rows
     * are written whatever ran before it in this process. */
    static void reset_globals_for_test();
    /** Reset only run-scoped best-comparator state while preserving the
     * monotonic evaluation counter/writer cursor. Used by fresh-run startup
     * and exposed for the embedded two-runs-in-one-process regression. */
    static void reset_best_state_for_fresh_run();
    /* Test-only accessors of the best comparator and the process-wide
     * evaluation counter (test_hpo_host_e2e pins the fresh-run reset). */
    static double      current_best()       { return g_best_.load(); }
    static std::size_t current_best_iter()  { return g_best_iter_.load(); }
    static std::size_t current_last_best()  { return iter_at_last_best_.load(); }
    static std::size_t g_iter_value();

    /** Establish this run's output generation before the first evaluation.
     * HPO runs are not resumable: every run removes stale best-potential
     * incumbents and starts its comparator from scratch, and its PC_LAMMPS
     * --hotvar values from their declared initial values or WARM's seeds
     * (warmstart::reset_for_fresh_run). The evaluation counter continues
     * (reset_best_state_for_fresh_run). MPI ranks coordinate before any
     * evaluation can publish. */
    void prepare_best_publication() const;

    std::size_t train_size() const;   // number of training structures
    std::size_t val_size()   const;   // number of validation structures

    /** Read-only view of the training StructureDB. The audit (Phase 5)
     *  passes this to `run_preflight_from_db` for cheap dataset stats. */
    const mlip::StructureDB& train_stdb() const { return train_stdb_; }
    /** Mutable view of the training StructureDB. compute_data_stats()
     *  may run NN-finder on it; the host has already done so but the
     *  call is idempotent. */
    mlip::StructureDB&       train_stdb_mut()   { return train_stdb_; }

    /** Report of load-time transforms applied to the training set during
     *  construction. */
    const mlip::LoadTransformReport& load_transform_report() const { return report_; }

    /** Report of load-time transforms applied to the validation set during
     *  construction. Only LSCALE / resolved ESHIFT, plus EFILTER / FFILTER
     *  taken from the --validation file itself, contribute here; training-side
     *  filters and weighting knobs are deliberately not forwarded. */
    const mlip::LoadTransformReport& val_load_transform_report() const { return val_report_; }

    /** True when the EMBF0 journal records per-evaluation values: the model
     *  has a DM_REAM block and is linear in the raw descriptor. */
    bool embf0_diagnostic() const { return embf0_diag_; }

    /** Write every result still queued, in iteration order -- including
     *  results stranded behind an iteration that never arrived, with a
     *  journal warning (result_drain.h). Call once the optimiser has
     *  stopped; drive_run() does, however the run ends.
     *  retry_io=false: no backoff, only a permanent errno's one immediate
     *  re-attempt (after a best-potential persistence failure the storage is
     *  known to be failing). After a shutdown request the drain's retry
     *  budget is the 2 s stop grace either way. */
    void flush_remaining(bool retry_io = true) const;

    /** Whether an evaluation of this run on this rank produced a usable
     *  (feasible) loss. The first one always becomes the incumbent, so this
     *  is whether this rank published a best potential since
     *  prepare_best_publication(). drive_run() ends a run without one with
     *  status 1. */
    bool has_usable_loss() const
    {
        return g_best_iter_.load(std::memory_order_acquire) > 0;
    }

    /* ---------- top-K tracking (used by StagedPolicy) ------------------ */
    /** Enable a buffer that records the top-K (lowest loss) (loss, params)
     *  pairs across operator() calls. Capacity = k; older worse-loss
     *  entries are evicted, near-clones are deduped (see topk.h -- a
     *  clone agrees in loss AND normalised parameter distance). `widths` are the
     *  real-space OPTIM box widths used to normalise the parameter test;
     *  empty means loss-only dedup. Disabled by default (zero overhead). */
    static void set_track_topk(std::size_t k,
                               std::vector<double> widths = {});
    /** Snapshot the current top-K, sorted ascending by loss. */
    static std::vector<std::pair<double, ParamVec>> get_topk();
    /** Reset the top-K buffer (called at start of each staged-global run). */
    static void clear_topk();

    /* ---------- stagnation tracking (used by RestartPolicy) ------------ */
    /** Throw StagnationStop from operator() when (g_iter_ - iter_at_last_best)
     *  exceeds n, where iter_at_last_best resets on every new best.
     *  n = 0 disables (default). */
    static void set_stagnation_threshold(std::size_t n);
    static std::size_t get_stagnation_threshold();
    /** Reset the stagnation counter without changing the threshold. */
    static void reset_stagnation();
    /** Current stagnation count: g_iter_ - iter_at_last_best_. */
    static std::size_t stagnation_count();

    /** Exception thrown by operator() to signal a stagnation-driven
     *  abort. The RestartPolicy catches and re-enters with a fresh init;
     *  every optimiser adapter lets it through, and MULTISTART / the
     *  PRESTAGE refinement end the start that stagnated. One type for every
     *  host (run_stop.h), so the evaluation pools can rank it against the
     *  other run-level stops. */
    using StagnationStop = ::tadah::hpo::StagnationStop;


private:

    /* ---------- immutable after construction --------------------------- */
    core::Context                master_ctx_;
    HPOTargets&                  trg_;
    /* Stage 1b.1: per-script fan-out cost estimate (EWMA of create+run ms),
     * indexed by script index into trg_.LS. Drives longest-processing-time-
     * first (LPT) fan-out ordering so the slowest scripts dispatch first and
     * short ones backfill the freed threads. Steers dispatch ORDER only, never
     * a result value, so lock-free relaxed atomics with benign torn reads are
     * fine (same rationale as dloss_dtheta.h's theta_cost_cache). mutable: it
     * is updated from the const run_lammps. */
    mutable std::vector<std::atomic<double>> script_ewma_ms_;
    std::size_t                  n_val_datasets_ = 0;
    const OptimizerSpec &spec_;
    mlip::StructureDB            val_stdb_;
    mlip::StructureDB            train_stdb_;
    mlip::LoadTransformReport    report_;
    mlip::LoadTransformReport    val_report_;
    bool                         use_lammps_;
    /* No-training-DBFILE mode: weights are provisioned from
     * FIXINDEX/FIXWEIGHT instead of the per-eval LS fit. */
    bool                         has_train_data_ = true;
    /* FIXINDEX covers every model weight (the refit coefficient-search
     * workload): the LS solve is vacuous, so the per-eval train stage
     * provisions the weights directly and skips the design-matrix
     * build. Computed once in the ctor. */
    bool                         all_weights_fixed_ = false;
    /* EMBF0 journal: a DM_REAM model linear in the raw descriptor
     * (BF_Linear / Kern_Linear; NORM irrelevant), so sum_m w_m g_m(0) of the
     * trained weights is the raw F(0). Computed once in the ctor; false ->
     * no per-evaluation work at all. */
    bool                         embf0_diag_ = false;
    /* --validation is optional when the loss has other sources
     * (PC_OBSERVABLE / PC_LAMMPS); validated in validate_hpo_setup(). */
    bool                         has_validation_ = true;
    /* The validation datasets PC_SRMSE measures (a dataset without a stress
     * tensor is not; PC_FRMSE keeps its own in trg_.frmse_targets). Empty:
     * every dataset. */
    std::vector<bool>            val_measured_srmse_;
    std::shared_ptr<MetricSkipTracker> metric_skip_tracker_ = std::make_shared<MetricSkipTracker>();
    /* Knot-family channels (D2_Knot5 / DM_REAM knot bcode): RCUT2B/RCUTMB
     * are DERIVED from the largest knot — re-derived on every evaluation in
     * update_context (which is const, hence flags precomputed here) and
     * folded into the one-time NN cutoff via detail::nn_safe_rcutmax. */
    bool                         knot2b_ = false;
    bool                         knotmb_ = false;
    /* Per-candidate bump-support gate (BUMPGATE CANDIDATE): the bumps
     * whose OPTIM box can violate c + w/eta <= RCUT on a hard Cut_Dummy
     * channel. Empty -- the common case, and always under BUMPGATE
     * STRICT -- means no per-eval check. Built once in the ctor. */
    std::vector<detail::BumpGateChannel> bump_gate_;
    /* RHOFORM NONNEG / MONOTONE: the flat indices of the OPTIM'd AMPGRIDMB
     * coordinates, projected onto [0, inf) before every evaluation
     * (project_positive_form). Empty -- SIGNED, no DM_REAM block, or no
     * OPTIM AMPGRIDMB -- means no per-eval work. Built once in the ctor. */
    std::vector<std::size_t>     form_proj_idx_;
    std::string                  form_proj_name_;   // "NONNEG" | "MONOTONE"
    /* DM_REAM density (a bare DM_REAM many-body term; rho_gates.h): the
     * RHOGATE / RHOBARGATE tables, checked right after update_context and
     * before training, and the settings of the density summary recorded at
     * each new best. nullopt -- no bare DM_REAM -- means no per-eval work.
     * Built once in the ctor; operator() only reads it. */
    std::optional<detail::DensityGates> density_;
    mlip::NNFinder               nnf_;
    /* Cutoff of the one-time neighbour lists of train_stdb_ / val_stdb_
     * (detail::nn_safe_rcutmax: the stated RCUTMAX and the OPTIM box tops).
     * A candidate whose model reads pairs further out -- a point outside
     * the OPTIM box, from an optimiser that ignores the bounds -- is
     * evaluated with lists rebuilt at its cutoff (rebuilt_lists). */
    double                       nn_list_cutoff_ = 0.0;

    /* ---------- data for pot header writer ----------------------------- */
    core::Context                val_ctx_;
    std::vector<std::string>     param_labels_;

    /* ---------- helpers (pure) ----------------------------------------- */
    void update_context(core::Context&, const ParamVec&) const;
    /** @p p with its AMPGRIDMB coordinates projected onto [0, inf)
     *  (form_proj_idx_): @p p itself when none is negative, else @p buf
     *  holding the projection. Counts the projection in HPOLog and
     *  journals the first one (eval @p iter). Never throws. */
    const ParamVec& project_positive_form(const ParamVec& p, ParamVec& buf,
                                          std::size_t iter) const;
    /** @p p with each coordinate proposed outside its OPTIM box within
     *  rounding (SearchSpace::snapped_onto_box) moved onto the bound: @p p
     *  itself when there is none, else @p buf. Counts such an evaluation
     *  in HPOLog (record_snap), with no warning: NLopt's own arithmetic
     *  rounds a point on a bound past it under OPTIM_SCALE NONE (owner
     *  decision D10). Before the out-of-box count and any projection.
     *  Never throws. */
    const ParamVec& snap_onto_box(const ParamVec& p, ParamVec& buf) const;
    /** Counts evaluation @p iter in HPOLog when a coordinate of the REAL
     *  point @p p lies outside its OPTIM box (record_out_of_box_eval), and
     *  warns at the first: finite differences keep to the box
     *  (fd_stencil.h), and so do ASA's probes (asa_probe_box_error,
     *  TADAH-PATCH 8), and a point within rounding of a bound was snapped
     *  onto it, so an optimiser that ignores the bounds
     *  (OptimizerCaps::ignores_bounds), or a start it handed on, put it
     *  there. After snap_onto_box, before any projection. Never throws. */
    void note_out_of_box(const ParamVec& p, std::size_t iter) const;
    void fill_validation(Result&, const mlip::Analytics&)  const;
    void run_lammps     (Result&) const;

    /** Format the current parameter vector as "KEY=val KEY=val ..."
     *  Used in NaN-clamping log messages. */
    std::string format_params_str(const ParamVec& p) const;

    /** Legacy one-line bound hint for the NaN-clamp log path (compares in
     *  INTERNAL space). The generic audit is bound_occ_ / bound_audit.h. */
    std::string bound_hints(const ParamVec& p) const;
    void fit_loss(Result&) const;
    void sims_loss(Result&) const;
    /* The eval's loss from its terms so far; fit_loss, obs_loss and
     * sims_loss each end with it, so one that averages again after
     * another cannot turn the other's failure into a usable loss: the mean
     * of r.loss_sum over r.loss_count, a non-finite mean (a NaN clamp,
     * r.was_nan_clamped) or one above FAIL_SCORE clipped to FAIL_SCORE; an
     * eval with a failed simulation (r.sims_failed) or a failed term
     * (r.terms_failed) scores FAIL_SCORE whatever that mean; under
     * FAILSCORE_GRADING a classified failure (r.fail.band != 0) scores
     * the graded loss. */
    void assemble_loss(Result&) const;
    /** Render and synchronously publish this completed candidate to the
     * canonical cross-rank best file and this rank's local best file. On
     * failure it trips durable_latch_, records the candidate as unsaved,
     * journals the failure, sets `journalled` and throws
     * durable_best::Error. */
    void persist_durable_best(const Result&, bool& journalled) const;
    static void note_failed_publication(const Result&, const durable_best::Error&,
                                        std::optional<std::size_t> failed_before);
    /* Retry policy of every durable_best call: durable_best::default_backoff()
     * (30 s, abandoned on a shutdown request), each retried failure logged
     * as a WARNING and counted (record_io_retry_*). Built in the ctor. */
    durable_best::Options        dbopts_;
    /* The canonical publication once durable_latch_ is tripped: dbopts_
     * with no_backoff (a single attempt; only a rename reporting ENOENT
     * gets one immediate re-attempt). */
    durable_best::Options        dbopts_latched_;
    /* The non-fatal rank-local mirror (MPI multi-rank runs): a 1 s budget,
     * and none at all (single attempt) once a mirror publication has failed
     * until one succeeds again (mirror_degraded_, guarded by best_mtx_), so a
     * failing mirror cannot stall every new best by the full 30 s. */
    durable_best::Options        dbopts_mirror_;
    durable_best::Options        dbopts_mirror_single_;
    mutable bool                 mirror_degraded_ = false;
    /* PC_OBSERVABLE (D4c) loss aggregation: evaluates every entry of
     * trg_.observables at the current trained model, populates
     * r.v_obs_value / r.v_obs_loss, and folds the per-observable loss
     * into r.loss_sum / r.loss_count. A term that cannot be computed (a
     * diverging relax, a value or loss that is not finite) is FAIL_SCORE
     * and fails the eval (r.terms_failed; under FAILSCORE_GRADING band B,
     * measured by the failed fraction of the observables), so it does not
     * crash the eval nor average into a usable loss. Ends with
     * assemble_loss. No-op when trg_.observables is empty. */
    void obs_loss(Result&) const;

    /* ---------- global mutable state ----------------------------------- */
    static std::atomic_size_t   g_iter_;
    static std::atomic_size_t   g_best_iter_;
    static std::atomic<double>  g_best_;
    /* Serializes the complete loss/rank/slot/eval comparison and synchronous
     * publication. The atomics remain for cheap observers; this mutex makes
     * exact-loss ties deterministic within one process. */
    static std::mutex                       best_mtx_;
    /* New-best decisions started / done (the would-fire stagnation check
     * waits for those in progress; stagnation_stop.h). */
    static detail::BestPublications         best_pubs_;
    /* Tripped by the first canonical best-potential publication that fails
     * for good; every later candidate of the run then gets a single attempt
     * without backoff (dbopts_latched_, durable_best::FailLatch). The
     * non-fatal rank-local mirror never trips it. Guarded by best_mtx_;
     * reset by prepare_best_publication() (every run) and
     * reset_globals_for_test(). */
    static durable_best::FailLatch          durable_latch_;
public:
    /** The lock-free pre-check of the new-best decision: false when `loss`
     *  cannot become the best against a best-so-far snapshot `cur_best`
     *  (loss above it, or NaN), so the evaluation skips best_mtx_. Exact
     *  because g_best_ only decreases while evaluations run. */
    static constexpr bool may_become_best(double loss, double cur_best) noexcept
    {
        return loss <= cur_best;
    }
private:
    /* Bound-occupancy audit of the running best (bound_audit.h); armed
     * from trg.space() at construction, updated on the new-best path
     * under its own mutex, reported by HPOLog. */
    static BoundOccupancy                   bound_occ_;
    static std::mutex                       bound_occ_mtx_;
    static std::optional<durable_best::Key> g_best_key_;

    /* Top-K tracking -- gated. Pays mutex cost only when track_topk_capacity_ > 0. */
    static std::mutex                                            topk_mtx_;
    static std::size_t                                           topk_capacity_;
    static std::vector<std::pair<double, ParamVec>>              topk_buf_;
    static std::vector<double>                                   topk_widths_;

    /* Stagnation tracking -- gated. iter at which the most recent best was
     * recorded; stagnation = g_iter_ - iter_at_last_best_. */
    static std::atomic_size_t   iter_at_last_best_;
    static std::atomic_size_t   stagnation_threshold_;

    /* Once-per-event gate for the StagnationStop diagnostic. With OMP /
     * concurrent eval threads, every thread that finishes its LAMMPS
     * work after the threshold is crossed but before the policy resets
     * iter_at_last_best_ throws StagnationStop in turn. Set on the first
     * throw within an event window, cleared by reset_stagnation() (which
     * the surrounding policy calls when starting a new stage). */
    static std::atomic_bool     stagnation_warned_;

    /* opportunistic single-writer --------------------------------------- */
    static Queue                                   g_q_;          // MPMC queue
    static std::atomic_flag                        writer_token_; // cleared = free
    static std::mutex                              io_mtx_;       // protects FILEs
    static std::map<std::size_t,Result>            io_buf_;       // re-order
    static std::atomic<std::size_t>                next_flush_;   // next iter
    /* Bug 6: time (steady_clock ms since epoch) at which `next_flush_`
     * first became "stuck" -- i.e., the writer found io_buf_ has higher
     * keys than next_flush_ but the next_flush_ key itself is missing.
     * Reset to 0 on every successful drain. When > GAP_SKIP_MS old, the
     * writer skips next_flush_ forward to the smallest available key
     * (recovering from a hung eval thread that consumed the slot but
     * never returned). */
    static std::atomic<std::int64_t>               next_flush_stuck_since_ms_;
    /* The eight per-eval logs, opened lazily by the first drain after the
     * ctor wrote their headers (reset there and in reset_globals_for_test so
     * the offsets follow the current run's files). Both drains stage rows
     * and commit once per drain with io::BackoffPolicy::writer(), one budget
     * shared by the whole drain; a batch that still fails is dropped whole
     * and reported. Guarded by io_mtx_. */
    struct WriterFiles;
    static std::unique_ptr<WriterFiles>            writer_files_;
    static WriterFiles& writer_files(const HPOTargets& trg);     // io_mtx_ held
    static void reset_writer_files();                            // takes io_mtx_
    static void commit_writer_files(WriterFiles& wf);            // io_mtx_ held
    /* Archive (DUMP/BDUMP) writer: whole file or nothing, retried. */
    static void write_archive(WriterFiles& wf, const std::string& path,
                              const std::string& bytes);             // io_mtx_ held
    static void submit_result(Result&&, const HPOTargets& trg_,
                              const core::Context& master_ctx,
                              const core::Context& val_ctx,
                              const mlip::StructureDB& train_stdb,
                              const mlip::StructureDB& val_stdb,
                              const std::vector<std::string>& param_labels,
                              const mlip::LoadTransformReport& report,
                              std::size_t n_val, std::size_t cw, int prec);

    /* per thread MLIP workspace ----------------------------------------- */
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws_;

    /* The bump-gate reject path returns BEFORE training, breaking the
     * "operator()(x) ran immediately before us at the same x" contract
     * analytical_gradient() reads the trainer slot under. Set on reject,
     * cleared after every successful train; while set, gradients fall
     * back to all-FD instead of reading a stale cache. */
    static thread_local bool tls_trainer_theta_stale_;

    /* per-thread trainer slot -------------------------------------------
     * The Trainer of this thread's latest evaluation: every evaluation
     * builds a Trainer from its own candidate context (master_ctx_ plus the
     * candidate) and trains it, replacing the slot's. The slot is kept for
     * what reads the evaluation afterwards on the same thread
     * (analytical_gradient, obs_loss, tls_trainer_for_test). `host` is the
     * generation_ of the HPO_Host whose evaluation built it: a slot of
     * another host (an earlier run on this thread, e.g. the first TASK hpo
     * of a task file) is never read as this host's. */
    struct TrainerDeleter {
        void operator()(mlip::Trainer* p) const noexcept { if (p) p->~Trainer(); }
    };
    struct TrainerCache {
        alignas(mlip::Trainer) unsigned char             buf[sizeof(mlip::Trainer)];
        std::unique_ptr<mlip::Trainer, TrainerDeleter>   trainer;
        std::uint64_t                                    host = 0;
    };
    static thread_local TrainerCache tls_trainer_cache_;
    /* A number no other HPO_Host of the process has (from 1, increasing):
     * it stamps the trainer slot of this host's evaluations. */
    const std::uint64_t generation_ = detail::next_host_generation();
    /* True when this thread's trainer slot holds a trainer of this host. */
    bool slot_is_mine() const {
        return tls_trainer_cache_.trainer && tls_trainer_cache_.host == generation_;
    }
    /* See last_validation_switches_for_test(). */
    static thread_local std::optional<detail::FitSwitches>
        tls_last_validation_switches_;

    /* Neighbour lists rebuilt for one evaluation whose model cutoff exceeds
     * nn_list_cutoff_: copies of the datasets the evaluation reads (the
     * training set when it is fitted, the validation set when there is
     * one) with lists at `cutoff`; a set it does not read stays empty. */
    struct RebuiltLists {
        double            cutoff = 0.0;
        mlip::StructureDB train;
        mlip::StructureDB val;
    };
    /* The rebuilt lists of this thread's latest evaluation, or nullptr when
     * the one-time lists covered it (or it reads no dataset):
     * analytical_gradient differentiates the evaluation the trainer slot
     * holds, on the same lists. Cleared, with tls_trainer_theta_stale_ set,
     * before an evaluation touches the trainer, so a failed evaluation
     * leaves no state for the gradient to use. */
    static thread_local std::shared_ptr<const RebuiltLists> tls_rebuilt_lists_;
    /* Build RebuiltLists at @p cutoff for evaluation @p iter, counted in
     * DIAGNOSTICS (the first one also warned); nullptr, and not counted,
     * when the evaluation reads no dataset. */
    std::shared_ptr<const RebuiltLists> rebuilt_lists(double cutoff,
                                                      std::size_t iter) const;

public:
    /** Test hook: expose the Trainer of this thread's latest evaluation (of
     *  any host) so e2e tests can inspect the trained/provisioned model
     *  after an operator() call. nullptr before the first evaluation on
     *  this thread. */
    static const mlip::Trainer* tls_trainer_for_test() {
        return tls_trainer_cache_.trainer.get();
    }
    /* The FORCE/STRESS the most recent validation prediction on this thread
     * used (detail::validation_switches()), or nullopt before any. For
     * tests: no output shows what validation predicted when no metric
     * reads it. */
    static std::optional<detail::FitSwitches> last_validation_switches_for_test() {
        return tls_last_validation_switches_;
    }
private:

    /* formatting -- fixed geometry */
    static constexpr std::size_t iter_cw_   = 12;  // #ITERATION(10) + 2; holds up to 10^6
    static constexpr std::size_t ittime_cw_ = 12;  // IT-TIME/ms(10)  + 2
    // D19: replica/slot column (Result::chain_id). "SLOT"(4) + 2, wide
    // enough for any REPLICAS a shared-memory pool can drive.
    static constexpr std::size_t slot_cw_   = 6;

    /* formatting -- derived from outputDigits (set in constructor) */
    int          prec_;           // digits after decimal in scientific notation
    std::size_t  cw_;             // column_width(prec_) (column_titles.h)
    std::size_t  max_title_len_;  // column_title_budget(prec_) = cw_-4: L(VAR) always >=1 space gap; same truncation as VAR

    /* line printers ------------------------------------ */
    void init_loss   (const std::string& fn) const;
    void init_outvar (const std::string& fn) const;
    void init_params (const std::string& fn) const;
    void init_stats  (const std::string& fn) const;
    /* D18: false when no per-structure metric is active (a run with no
     * training/validation DB), in which case stats.tadah has no columns
     * beyond #ITERATION and its rows are suppressed. */
    bool stats_have_columns() const;

    /* Rows with their header's columns (n_val: validation datasets, the
     * per-dataset columns of each metric); a cell the evaluation did not
     * compute is NaN (see the definitions). */
    static void print_loss_line  (std::ostream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t n_val,
                                  std::size_t cw, int prec);
    static void print_outvar_line(std::ostream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);
    static void print_params_line(std::ostream&, size_t it, const Result&,
                                  std::size_t cw, int prec);
    static void print_stats_line (std::ostream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t n_val,
                                  std::size_t cw, int prec);

};

/* =======================================================================*/
}   // namespace tadah::hpo

#include "hpo_host_impl.hpp"
#endif

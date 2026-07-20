#ifndef TADAH_HPO_HOST_H_
#define TADAH_HPO_HOST_H_

/*
 *  Thread-safe, "stateless" HPO_Host
 *  ---------------------------------
 *  -  All data produced during one evaluation live in EvalResult, not in the
 *     functor object; operator() is therefore const and thread-safe.
 *  -  Results are forwarded to a single writer thread through a blocking
 *     moodycamel queue which guarantees ordered, race-free file output.
 *  -  MPI code is protected by one global mutex so that threaded builds can
 *     still work.  Remove the mutex if you choose to forbid that mix.
 */

#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include <tadah/core/checkpoint.h>          // for ShutdownRequested
#include <tadah/core/checkpoint_manager.h>  // for tick_eval cadence
#include <tadah/core/context.h>
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#if defined(TADAH_ENABLE_OPENMP)
#   include <omp.h>
#endif
#if defined(TADAH_BUILD_MPI)
#   include <mpi.h>
#endif

#include <moodycamel/blockingconcurrentqueue.h>     // moodycamel
#include <tadah/hpo/hpo_log.h>

namespace tadah::hpo {

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
class HPO_Host : public core::ICheckpointable
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
     *    - cache_miss:  trainer cache empty on first call (transient)
     *    - ift_failed:  analytical helper returned false (IFT Cholesky
     *                   non-PD); rare numerical edge case
     *    - default:     orchestrator gate (STRESS/NORM/OALGO/LAMBDA/
     *                   MODEL[1]/use_srmse/PC_LAMMPS) rejected the run */
    void emit_grad_diagnostic(const std::vector<ParamRef>& prefs,
                              const std::vector<int>&      needs_fd,
                              bool                         cache_miss,
                              bool                         ift_failed = false) const;

    /* ICheckpointable -- persists the static best-so-far comparator state
     * (g_best_, g_best_iter_, iter_at_last_best_, g_iter_) so that a
     * resumed run uses the prior best as the comparator for the very
     * first post-resume eval. Without this, g_best_ defaults to
     * +infinity and the first post-resume eval always reports
     * spuriously as a new best, corrupting best_pot.tadah and the
     * journal's running best. The actual journal counters live in
     * HPOLog and are persisted there. */
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;
    std::string tag() const override { return "HPO_HOST"; }

    /* Static save/load helpers -- the ICheckpointable overrides delegate
     * here so tests can exercise the round-trip without constructing a
     * full HPO_Host (which requires datasets, a trainer, etc.). The
     * checkpoint section tag is the same as tag(), i.e. HPO_HOST. */
    static void save_globals(core::Checkpoint& cp);
    static void load_globals(const core::Checkpoint& cp);

    /* Test-only: reset the static comparator atomics to their freshly-
     * constructed values. Used by regression tests to simulate process
     * restart without spawning a new process. */
    static void reset_globals_for_test();
    /* Test-only accessors. */
    static double      current_best()       { return g_best_.load(); }
    static std::size_t current_best_iter()  { return g_best_iter_.load(); }
    static std::size_t current_last_best()  { return iter_at_last_best_.load(); }

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

    /** Flush any results still in the queue (call after optimizer finishes). */
    void flush_remaining() const;

    /* ---------- top-K tracking (used by StagedPolicy) ------------------ */
    /** Enable a buffer that records the top-K (lowest loss) (loss, params)
     *  pairs across operator() calls. Capacity = k; older worse-loss
     *  entries are evicted, near-clones are deduped (see topk.h -- basin
     *  dedup by loss AND normalised parameter distance). `widths` are the
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

    /* ---------- restart support ----------------------------------------- */
    /** Bring the eval counter forward to at least @p n after a checkpoint
     *  load -- eval numbering then continues monotonically across the
     *  resume boundary. Idempotent: never lowers g_iter_. */
    static void set_g_iter_floor(std::size_t n);
    static std::size_t g_iter_value();

    /** Exception thrown by operator() to signal a stagnation-driven
     *  abort. The RestartPolicy catches and re-enters with a fresh init. */
    struct StagnationStop : public std::exception {
      const char* what() const noexcept override { return "HPO stagnation threshold reached"; }
    };


    /* -------- adapter for NLopt C++ wrapper ----------------------------- */
    static double nlopt_adapter_cpp(const std::vector<double>& x,
        std::vector<double>&       grad,
        void*                      data)
    {
      auto* self = static_cast<const HPO_Host*>(data);
      if (!grad.empty()) {};               // no analytic gradient
      return (*self)(make_param_vec(static_cast<unsigned>(x.size()),
            x.data()));
    }

private:


    // helper that converts the raw array coming from NLopt -----------------
    static ParamVec make_param_vec(unsigned n, const double* x)
    {
      ParamVec v;                   // v is a dlib::matrix<double,0,1>
      v.set_size(n);
      for (unsigned i = 0; i < n; ++i)
        v(i) = x[i];
      return v;
    }


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
#ifndef TADAH_BUILD_MPI
    mlip::StructureDB            train_stdb_;
#endif
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
    /* --validation is optional when the loss has other sources
     * (PC_OBSERVABLE / PC_LAMMPS); validated in validate_hpo_setup(). */
    bool                         has_validation_ = true;
    /* Knot-family channels (D2_Knot5 / DM_REAM knot bcode): RCUT2B/RCUTMB
     * are DERIVED from the largest knot — re-derived on every evaluation in
     * update_context (which is const, hence flags precomputed here) and
     * folded into the one-time NN cutoff via detail::nn_safe_rcutmax. */
    bool                         knot2b_ = false;
    bool                         knotmb_ = false;
    mlip::NNFinder               nnf_;

    /* ---------- data for pot header writer ----------------------------- */
    core::Context                val_ctx_;
    std::vector<std::string>     param_labels_;

    /* ---------- helpers (pure) ----------------------------------------- */
    void update_context(core::Context&, const ParamVec&) const;
    void fill_validation(Result&, const mlip::Analytics&)  const;
    void run_lammps     (Result&) const;

    /** Format the current parameter vector as "KEY=val KEY=val ..."
     *  Used in NaN-clamping log messages. */
    std::string format_params_str(const ParamVec& p) const;

    /** Check if any value in p is suspiciously close to a bound.
     *  Returns a hint string (empty if everything looks fine). */
    std::string bound_hints(const ParamVec& p) const;
#if defined(TADAH_BUILD_MPI)
    void run_lammps_mpi (Result&) const;
#endif
    void fit_loss(Result&) const;
    void sims_loss(Result&) const;
    /* PC_OBSERVABLE (D4c) loss aggregation: evaluates every entry of
     * trg_.observables at the current trained model, populates
     * r.v_obs_value / r.v_obs_loss, and folds the per-observable loss
     * into r.loss_sum / r.loss_count. Recomputes r.loss = mean.
     * Mirrors sims_loss's FAIL_SCORE / NaN clamp behaviour so a
     * diverging relax doesn't crash the eval. No-op when
     * trg_.observables is empty. */
    void obs_loss(Result&) const;

    /* ---------- global mutable state ----------------------------------- */
    static std::atomic_size_t   g_iter_;
    static std::atomic_size_t   g_best_iter_;
    static std::atomic<double>  g_best_;

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
    static void submit_result(Result&&, const HPOTargets& trg_,
                              const core::Context& master_ctx,
                              const core::Context& val_ctx,
                              const mlip::StructureDB& train_stdb,
                              const mlip::StructureDB& val_stdb,
                              const std::vector<std::string>& param_labels,
                              const mlip::LoadTransformReport& report,
                              std::size_t cw, int prec);

#if defined(TADAH_BUILD_MPI)
    static std::mutex           mpi_mutex_;
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws2_;
#endif

    /* per thread MLIP workspace ----------------------------------------- */
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws_;

    /* per-thread trainer cache ------------------------------------------ */
#ifndef TADAH_BUILD_MPI
    struct TrainerDeleter {
        void operator()(mlip::Trainer* p) const noexcept { if (p) p->~Trainer(); }
    };
    struct TrainerCache {
        alignas(mlip::Trainer) unsigned char             buf[sizeof(mlip::Trainer)];
        std::unique_ptr<mlip::Trainer, TrainerDeleter>   trainer;
    };
    static thread_local TrainerCache tls_trainer_cache_;

public:
    /** Test hook: expose this thread's cached trainer so e2e tests can
     *  inspect the trained/provisioned model after an operator() call.
     *  nullptr before the first evaluation on this thread. */
    static const mlip::Trainer* tls_trainer_for_test() {
        return tls_trainer_cache_.trainer.get();
    }
private:
#endif
#if defined(TADAH_BUILD_MPI)
    struct TrainerHostDeleter {
        void operator()(mlip::TrainerHost* p) const noexcept { if (p) p->~TrainerHost(); }
    };
    struct TrainerHostCache {
        alignas(mlip::TrainerHost) unsigned char                   buf[sizeof(mlip::TrainerHost)];
        std::unique_ptr<mlip::TrainerHost, TrainerHostDeleter>     trainer;
    };
    static thread_local TrainerHostCache tls_trainer_host_cache_;
#endif



    /* formatting -- fixed geometry */
    static constexpr std::size_t iter_cw_   = 12;  // #ITERATION(10) + 2; holds up to 10^6
    static constexpr std::size_t ittime_cw_ = 12;  // IT-TIME/ms(10)  + 2
    static constexpr std::size_t min_cw_    = 15;  // floor: fits L(ErRMSE)[avg](14) + 1 space

    /* formatting -- derived from outputDigits (set in constructor) */
    int          prec_;           // digits after decimal in scientific notation
    std::size_t  cw_;             // max(prec_+9, min_cw_)
    std::size_t  max_title_len_;  // cw_-4: L(VAR) always >=1 space gap; same truncation as VAR

    /* line printers ------------------------------------ */
    void init_loss   (const std::string& fn) const;
    void init_outvar (const std::string& fn) const;
    void init_params (const std::string& fn) const;
    void init_stats  (const std::string& fn) const;

    static void print_loss_line  (std::ofstream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);
    static void print_outvar_line(std::ofstream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);
    static void print_params_line(std::ofstream&, size_t it, const Result&,
                                  std::size_t cw, int prec);
    static void print_stats_line (std::ofstream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);

};

/* =======================================================================*/
}   // namespace tadah::hpo

#include "hpo_host_impl.hpp"
#endif

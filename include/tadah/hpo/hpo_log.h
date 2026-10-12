#pragma once
/*
 * HPOLog -- human-readable run journal for Tadah HPO jobs.
 *
 * Thread-safe singleton.  Call open() once from the main thread before
 * the optimiser starts; all other methods are safe from any thread.
 *
 * Output file default: hpo_run.log  (written to the working directory)
 *
 * File layout
 * -----------
 *   == banner (version, cwd, PID, timestamp) ==
 *   -- FILES         --  -- config / output files
 *   -- OPTIMISER     --  -- algorithm, stop criteria, params
 *   -- SEARCH SPACE  --  -- per-dimension bounds (optimiser-internal space)
 *   -- POTENTIAL      --  -- 2B/MB, cutoffs, EMBF0 (DM_REAM), element list,
 *                            DM_REAM density report
 *   -- LOSS FUNCTION --  -- targets, weights, FAIL_SCORE
 *   -- PROGRESS      --  -- timestamped events: bests, NaN warnings, milestones
 *   -- DIAGNOSTICS   --  -- NaN rate, bound-proximity analysis
 *   -- PERFORMANCE   --  -- timing statistics
 *   -- RESULT        --  -- final status, best loss, best metrics, EMBF0 F(0),
 *                            the best's DM_REAM density
 */

#include <atomic>
#include <tadah/hpo/bound_audit.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Forward declarations of HPO types used in section writers
namespace tadah::hpo { struct OptimizerSpec; class HPOTargets; struct RawEmbeddingF0; }
namespace tadah::hpo::detail { struct DensitySummary; struct DensityReportEntry; }
namespace tadah::core { class Context; class Element; }
namespace tadah::mlip { struct LoadTransformReport; }

namespace tadah::hpo {

/* What one evaluation's loss was, for the journal's DIAGNOSTICS census.
 * Only a usable loss can become the incumbent. */
enum class EvalLoss {
    Usable,      /* a usable (feasible) loss */
    Failed,      /* FAILSCORE or a graded failure: finite, not usable */
    NanClamped,  /* a NaN/Inf loss: recorded as FAILSCORE and counted as a
                    NaN/Inf clamping */
    Threw        /* the evaluation raised an exception: recorded as
                    FAILSCORE and counted as an evaluation that threw */
};

/* The pre-training feasibility gates of HPO_Host::operator(), in the order
 * they are checked: BUMPGATE (c + w/eta past the cutoff), RHOGATE (the
 * DM_REAM rho(r) sign) and RHOBARGATE (the lattice rhobar(a)). A reject is
 * a graded band-D failure, counted for the first failing gate. */
enum class GateKind : std::size_t { Bump = 0, RhoSign = 1, RhoBar = 2 };
inline constexpr std::size_t kGateKinds = 3;

/* One near-bound observation reported by an annealing driver: the
 * chain sits within one proposal step of an OPTIM bound (under CORANA
 * the step has additionally bottomed out at its floor, so no further
 * shrinkage can work it free). Inward proposals remain possible; this
 * mark is a heuristic diagnostic and makes no claim about the optimum. */
struct PinnedDim {
    std::size_t dim;   /* search-space dimension index */
    bool        hi;    /* true = nearest the HIGH bound, false = LOW */
    double      gap;   /* |x - bound|, optimiser-internal units */
    double      tol;   /* audit tolerance = the dim's proposal scale
                        * (CORANA: the V floor; CLASSIC: sigma[i]) */
};

/* =========================================================================
 * HPOLog -- the logger.
 * ========================================================================= */
class HPOLog
{
public:
    static HPOLog& instance()
    {
        static HPOLog inst;
        return inst;
    }

    /* ------------------------------------------------------------------
     * Lifecycle
     * ------------------------------------------------------------------ */

    /** Open the log file and write the top banner.
     *  @param path        Path to the log file (default: "hpo_run.log")
     *  @param version     Tadah version string from BlueprintParser
     *  @param cwd         Working directory (std::filesystem::current_path())
     *  @param pid         Process ID (getpid())
     */
    void open(const std::string& path,
              const std::string& version,
              const std::string& cwd,
              int pid);

    bool is_open() const { return f_.is_open(); }

    /** The MPI rank whose journal this is, and the number of ranks of the
     *  run. Each rank of a multi-rank run keeps its own journal, which
     *  counts that rank's evaluations only; with `world` > 1 the
     *  DIAGNOSTICS verdict on usable losses names the rank ("No evaluation
     *  on MPI rank 1 (of 2) produced a usable loss."), since whether the
     *  run as a whole produced one is decided over every rank (drive_run).
     *  Default rank 0 of 1; hpo_run() sets it after open(). */
    void set_rank_scope(int rank, int world);

    /** The evaluations an earlier run in this process made: this run's are
     *  numbered on from them (HPO_Host's counter), and the journal's
     *  milestones ("--- N evals done") count this run's alone. Default 0;
     *  hpo_run() sets it once the host is built, before any evaluation. */
    void set_eval_base(std::size_t evaluations_before);

    /* ------------------------------------------------------------------
     * Primitive writers (all thread-safe, append to file)
     * ------------------------------------------------------------------ */

    void writeln(const std::string& text = "");
    void section(const std::string& title);                 // -- TITLE --
    void kv     (const std::string& key,
                 const std::string& val,
                 int key_width = 22);                       // "  key : val"
    void indent (const std::string& text);                  // "  text"
    void separator();                                       // thin rule

    /** Timestamped event line: "[HH:MM:SS]  tag  msg" */
    void event(const std::string& tag, const std::string& msg);
    void info   (const std::string& msg) { event("INFO", msg); }
    void warning(const std::string& msg) { event("WARNING", msg); }
    void diag   (const std::string& msg) { event("DIAG   ", msg); }
    /** Heuristic/empirical advice -- weaker than WARNING by design: use
     *  for literature-derived or rule-of-thumb thresholds, never for
     *  structural certainties (those are warnings). */
    void hint   (const std::string& msg) { event("HINT", msg); }

    /* ------------------------------------------------------------------
     * Structured section writers (implemented in hpo_log.cpp)
     * ------------------------------------------------------------------ */

    void write_files_section(
        const std::string& target_file,
        const std::string& val_file,
        const HPOTargets&  trg);

    /** Writes the OPTIMISER section, then the SEARCH SPACE section (the
     *  per-dimension bounds dump, in optimiser-internal space). */
    void write_optimizer_section(
        const OptimizerSpec&             sp,
        const std::vector<std::string>&  param_labels);

    /** POTENTIAL SETUP section. For a context with a DM_REAM block it adds
     *  the "Embedding F(0)" row (the EMBF0 mode, read without inserting the
     *  default) and, when @p embf0_seed is given, the raw F(0) of the -c
     *  potential (derive_raw_embedding_f0). */
    void write_potential_section(
        const core::Context&  ctx,
        std::size_t           n_train,
        std::size_t           n_val,
        const std::string&    elements_str,
        const RawEmbeddingF0* embf0_seed = nullptr);

    /** What training fits (FORCE, STRESS: their values and where each came
     *  from) and what validation predicts, for POTENTIAL SETUP. */
    struct FitSwitchReport {
        bool        force = false;
        std::string force_source;
        bool        stress = false;
        std::string stress_source;
        bool        has_validation = false;
        bool        val_forces = false;    // PC_FRMSE
        bool        val_stresses = false;  // PC_SRMSE
    };

    /** The rows "Fit forces (FORCE)", "Fit stresses (STRESS)" and
     *  "Validation predicts", in the current section (POTENTIAL SETUP:
     *  hpo_run() calls it after write_potential_section). */
    void write_fit_switches(const FitSwitchReport& report);

    /** LOAD TRANSFORMS section -- emitted once at host construction summarising
     *  what load-time transforms (LSCALE, ESHIFT, EFILTER, FFILTER, WDBFILE,
     *  EWEIGHT_TEMP, ZERO_COM_FORCE, equilibrium-volume) were applied to the
     *  training set. The validation report is summarised on a single line
     *  (validation receives only LSCALE / resolved ESHIFT, plus EFILTER /
     *  FFILTER if declared in the --validation file) so the validation-set
     *  size after filtering is always visible.
     *
     *  Filter lines (training / validation) report `loaded -> remain` so
     *  the two sides are directly comparable.
     *
     *  No-op if @p train_report is informational-only (applied=false) AND
     *  the validation report did not drop any configs. */
    void write_load_transforms_section(const mlip::LoadTransformReport& train_report,
                                       std::size_t                      n_train_loaded,
                                       const mlip::LoadTransformReport& val_report,
                                       std::size_t                      n_val_loaded);

    void write_loss_section(const HPOTargets& trg);

    /** NOISE CALIBRATION section -- written when the engine repeatedly
     *  evaluates the seed point to measure objective noise. Surfaces
     *  per-run loss, summary stats, and a suggested FTOL_REL.
     *  @param losses           Total-loss values from each calibration eval
     *  @param suggested_ftol   The recommended FTOL_REL (caller-computed)
     */
    void write_noise_calibration_section(
        const std::vector<double>& losses,
        double                     suggested_ftol);

    void start_progress();          // writes the PROGRESS section header

    /* ------------------------------------------------------------------
     * Per-eval recording (called from worker threads, no file I/O here)
     * ------------------------------------------------------------------ */

    /** Record timing and the kind of loss of one completed evaluation.
     *  @param eval_num   Global evaluation counter (1-based)
     *  @param wall_ms    Wall time for this evaluation in milliseconds
     *  @param loss       Usable, failed, NaN/Inf clamped or threw. Every
     *                    kind but Usable counts as an unusable loss in
     *                    DIAGNOSTICS; NanClamped also as a NaN/Inf
     *                    clamping, Threw also as an evaluation that threw.
     */
    void record_eval(std::size_t eval_num, double wall_ms, EvalLoss loss);

    /** Record the graded-FAILSCORE band of one completed evaluation.
     *
     *  "NaN rate" is not a failure rate. A pc_lammps task that was killed or
     *  whose script failed comes back with a parseable failscore string, so
     *  the assembled loss is a large but FINITE number and nothing sets
     *  was_nan_clamped -- a long run can report NaN rate=0.0% across tens of millions of evals
     *  while six workers were SIGKILLed. Nothing anywhere counted failed
     *  evaluations, so a run could degrade to a near-total failure rate and
     *  still look healthy in the journal.
     *
     *  @param band  PcFailInfo::band -- 0 none, 1..4 = A..D (worst wins).
     *               Bands 1..4 are tallied; 0 is ignored. */
    void record_fail_band(int band);

    /** Tally one pre-training gate reject (a band-D subset: the candidate
     *  was rejected before training by @p kind, the first gate it failed).
     *  Surfaced in the SUMMARY failure census (" bumpgate=N rhogate=N
     *  rhobargate=N", each when > 0) and the DIAGNOSTICS trailer so a high
     *  reject rate is visible. Call strictly after record_fail_band: the
     *  SUMMARY that record_eval may emit must never show a gate count above
     *  its band-D count. */
    void record_gate_reject(GateKind kind);
    /** record_gate_reject(GateKind::Bump): a bump-support per-candidate
     *  gate reject (the trial's c + w/eta reached past RCUT). */
    void record_bump_gate_reject();
    /** Rejects counted so far for @p kind (record_gate_reject). */
    std::size_t gate_rejects(GateKind kind) const {
        return n_gate_[static_cast<std::size_t>(kind)].load(std::memory_order_relaxed);
    }
    /** The rate limit of the per-reject journal notes, one counter per
     *  gate: returns the running count of @p kind's rejects including this
     *  one; the host writes the note when it is <= 5 or a power of two
     *  ("total rejects=N"). Reset by open() and reset_for_test(). */
    std::size_t next_gate_note(GateKind kind);

    /** DM_REAM density report (a model whose many-body term is a bare
     *  DM_REAM; hpo_run() calls it after write_potential_section): one
     *  "Density ..." block per entry -- the -c potential and each WARM
     *  seed, built by detail::startup_density_report -- and arms the RESULT
     *  line for the best (plus its rhobar line when @p rhobar, i.e.
     *  RHOBARGATE is configured). A summary that could not be computed
     *  prints "not derivable: <reason>". Never throws. */
    void write_density_report(const std::vector<detail::DensityReportEntry>& entries,
                              bool rhobar);
    /** The density of a new best (HPO_Host, after the best decision, the
     *  EMBF0 pattern): kept only when @p loss is below the loss recorded
     *  with the previous summary, so out-of-order journal calls cannot
     *  leave a superseded best. finish() prints it under RESULT, or "n/a
     *  (not derivable)" for a best without one (the host records one for
     *  every new best unless it could not build the density at startup).
     *  Never throws. */
    void record_density_best(std::size_t eval_num, double loss,
                             const detail::DensitySummary& summary);

    /** Tally one evaluation whose AMPGRIDMB coordinates were projected onto
     *  [0, inf) under RHOFORM NONNEG / MONOTONE (HPO_Host::operator()): a
     *  point outside the OPTIM box, which only an optimiser that ignores
     *  the bounds evaluates (finite differences keep to the box,
     *  fd_stencil.h), evaluated and recorded at its projection. Returns the
     *  running count (the host journals the first projection). DIAGNOSTICS
     *  prints the total when it is > 0, so other runs are unchanged. */
    std::size_t record_form_projection();
    /** Evaluations projected so far (record_form_projection). */
    std::size_t form_projections() const {
        return n_form_proj_.load(std::memory_order_relaxed);
    }

    /** Finite differences and the OPTIM box (fd_stencil.h,
     *  gradient_provider.h, HPO_Host::operator()). Relaxed counters, no file
     *  I/O; DIAGNOSTICS prints each only when it is > 0, so the DIAGNOSTICS
     *  of a run that never came near a bound are unchanged. Reset by open()
     *  and reset_for_test(). Each returns the running count.
     *   - record_out_of_box_eval: an evaluated REAL point outside its
     *     OPTIM box, counted before any projection (the host warns at the
     *     first);
     *   - record_out_of_box_gradient: a finite-difference gradient at an
     *     internal point outside the box (differenced towards it);
     *   - record_one_sided: a finite-difference component of dimension
     *     `label` that a bound turned (one-sided into the box, backward
     *     instead of forward, or towards the box from outside it); the
     *     count is per dimension, and the caller journals the first;
     *   - record_hybrid_centre: the loss at a HYBRID gradient's point,
     *     evaluated for its one-sided or forward differences;
     *   - record_snap: an evaluation with a coordinate proposed outside its
     *     box within rounding (SearchSpace::snapped_onto_box), evaluated on
     *     the bound (the host; owner decision D10). */
    std::size_t record_out_of_box_eval();
    std::size_t record_snap();
    std::size_t snaps() const {
        return n_snap_.load(std::memory_order_relaxed);
    }
    std::size_t record_out_of_box_gradient();
    std::size_t record_one_sided(const std::string& label);
    std::size_t record_hybrid_centre();
    std::size_t out_of_box_evals() const {
        return n_oob_eval_.load(std::memory_order_relaxed);
    }
    std::size_t out_of_box_gradients() const {
        return n_oob_grad_.load(std::memory_order_relaxed);
    }
    std::size_t hybrid_centres() const {
        return n_hybrid_centre_.load(std::memory_order_relaxed);
    }
    /** An evaluation whose model cutoff exceeds the one-time neighbour
     *  lists (HPO_Host builds them at detail::nn_safe_rcutmax), evaluated
     *  with lists rebuilt for it; DIAGNOSTICS prints the count when it is
     *  > 0. Reset with the counters above. Returns the running count. */
    std::size_t record_nn_rebuild();
    std::size_t nn_rebuilds() const {
        return n_nn_rebuild_.load(std::memory_order_relaxed);
    }
    /** The one-sided components counted so far for `label` (0 if none). */
    std::size_t one_sided(const std::string& label) const;

    /** Transient-I/O accounting: retries of the durable best
     *  potential and of the per-evaluation logs/archives, rows dropped after
     *  a log writer exhausted its retries, and archives not written. Relaxed
     *  counters, no file I/O; finish() prints one DIAGNOSTICS line only when
     *  any is non-zero, so a clean run's journal is unchanged. HPOLog itself
     *  is deliberately not retried: it is the reporting channel, and a retry
     *  would hold mtx_ and serialise the evaluation threads. */
    void record_io_retry_durable(std::size_t n = 1);
    void record_io_retry_writer (std::size_t n = 1);
    void record_rows_lost       (std::size_t n);
    void record_archive_lost    ();
    /** The run is ending because the best potential could not be persisted:
     *  finish() must not describe it as complete. */
    void record_best_persistence_failure();

    /** A completed evaluation that was to become the new best could not be
     *  persisted (HPO_Host, just before the durable_best::Error leaves the
     *  evaluation). `may_be_on_disk`: the rename of that candidate over the
     *  best potential may have happened, so the file may hold it,
     *  unconfirmed (a failed directory fsync, or a rename that reported an
     *  errno it can report after landing; durable_best::SysError::
     *  may_have_landed). Of several such candidates (parallel chains) the
     *  record keeps the lowest loss (its eval) and the FIRST error text
     *  (the run's status), and lists separately every candidate that may
     *  be on disk (maybe_on_disk, in record order). finish() reports the
     *  candidate as NOT saved instead of claiming that no evaluation
     *  produced a usable loss or that best_pot.tadah holds the reported
     *  best -- unless a lower best was persisted after it, which then is
     *  the result (the candidate is listed as superseded) -- and names the
     *  candidates that may be in best_pot.tadah; hpo_run() and
     *  RestartPolicy fail the run on the record either way, even if the
     *  error was swallowed. */
    struct UnsavedBest {
        std::size_t eval = 0;
        double      loss = 0.0;
        std::string what;
        /* (eval, loss) of each candidate whose rename may have landed. */
        std::vector<std::pair<std::size_t, double>> maybe_on_disk;
        bool may_be_on_disk() const noexcept { return !maybe_on_disk.empty(); }
    };
    void record_unsaved_best(std::size_t eval, double loss,
                             const std::string& what, bool may_be_on_disk);
    std::optional<UnsavedBest> unsaved_best() const;

    /** MULTISTART starts (and PRESTAGE refinement seeds) over the whole run:
     *  detail::run_starts adds every batch, so the status note "(N of M
     *  MULTISTART starts failed ...)" survives RESTART relaunches and the
     *  RESTART exhaustion status. Reset by open() and reset_for_test(). */
    struct StartTally {
        std::size_t ran     = 0;   // starts run
        std::size_t failed  = 0;   // of which failed
        std::size_t batches = 0;   // run_starts batches (launches)
    };
    void record_start_batch(std::size_t ran, std::size_t failed);
    StartTally start_tally() const;
    std::size_t io_retries_durable() const { return n_io_retry_durable_.load(std::memory_order_relaxed); }
    std::size_t io_retries_writer () const { return n_io_retry_writer_ .load(std::memory_order_relaxed); }
    std::size_t rows_lost         () const { return n_rows_lost_       .load(std::memory_order_relaxed); }
    std::size_t archives_lost     () const { return n_archives_lost_   .load(std::memory_order_relaxed); }

    /** Register the shared script-executor pool width (Option A: THREADS).
     *  Enables the honest pool-utilisation figure in the periodic SUMMARY:
     *  util = Σ script busy / (pool width x window elapsed). Without it the
     *  per-script stats still print but no utilisation is claimed.
     *  (The pre-shared-pool per-eval width x wall denominator double-counted
     *  overlapped wall time under concurrent replicas -- efficiencies >100%.) */
    void set_fanout_width(int width);

    /** Record pc_lammps fan-out busy time for one completed evaluation.
     *  Folded into the periodic SUMMARY (pool utilisation + per-script
     *  avg/fastest/slowest). Cheap: a few adds under mtx_, no file I/O.
     *  Skipped when n_scripts==0.
     *  @param busy_ms    sum of per-script busy time across the fan-out
     *  @param n_scripts  number of pc_lammps scripts that ran this eval
     *  @param min_ms     fastest single script this eval
     *  @param max_ms     slowest single script this eval
     *  @param slowest    name of the slowest script (worst offender) */
    void record_fanout(double busy_ms,
                       std::size_t n_scripts, double min_ms, double max_ms,
                       const std::string& slowest);

    /** Bound-occupancy audit (every backend): HPO_Host reports the marks
     *  of the latest new best (REAL best within EDGE of a bound in
     *  internal units). The milestone SUMMARY prints them as an INFO
     *  table with one HINT; finish() prints the final best's table in
     *  DIAGNOSTICS. Empty vector = no dim near a bound. */
    void record_bound_occupancy(const std::vector<BoundOccupancy::Mark>& marks,
                                std::uint64_t eval, double edge);

    /** Arm the pinned-dim audit (CORANA annealers only -- CLASSIC never
     *  shrinks its step, so "step at the V floor" is undefined there).
     *  Stores the per-dim bounds snapshot the milestone table prints
     *  (bound value, v_floor = v_floor_frac*(high-low), (ln) tags) plus
     *  the chain count for the slots column (1 = serial -> column is
     *  dropped) and the work-unit display name ("rung"/"epoch"/"T-step").
     *  Re-arming clears any previous marks (fresh audit per driver run).
     *  Dim labels come from write_optimizer_section when available. */
    void arm_pin_audit(const OptimizerSpec& sp, double v_floor_frac,
                       int n_chains, const std::string& unit_name);

    /** Replace chain @p chain's pinned-dim set as of work unit @p unit
     *  (the driver's rung/epoch/T-step counter -- only monotonicity is
     *  used, for the table's "observed for" column). A dim marked before
     *  but absent from @p pins is unmarked; a dim still marked on the
     *  same side keeps its original since-unit. Cheap map update under
     *  mtx_, no file I/O: the table prints with the next milestone
     *  SUMMARY (single WARNING block, only while marks exist). */
    void record_pin_state(int chain, std::uint64_t unit,
                          const std::vector<PinnedDim>& pins);

    /** Register the optimiser's fixed work-unit cadence (rung / epoch /
     *  T-step) for the periodic milestone SUMMARY: each milestone then
     *  appends "<unit> = N evals ~ <wall> at current rate [; ~U <unit>(s)
     *  / ~<eta> to MAXEVAL]". The estimate is re-derived from the CURRENT
     *  milestone-window wall rate every time (never frozen), so it tracks
     *  throughput drift.
     *  @param evals_per_unit  EXACT evals per unit (arithmetic, not an
     *                         estimate) -- 0 disables the line
     *  @param unit            display name ("epoch", "rung", "T-step")
     *  @param maxeval_total   total eval budget; 0 = none -> no budget ETA.
     *  Single-process optimisers only: the throughput window is local. */
    void set_cadence(std::size_t evals_per_unit, const std::string& unit,
                     std::uint64_t maxeval_total);

    /** Log a new-best event (timestamped, written to file).
     *  @param top_breakdown  Optional human-readable summary of the top
     *                        loss components driving this best (e.g.
     *                        "Ediff=509.8(15%) C44=364.3(11%) ...").
     *                        Built by the caller from r.v_loss + r.*Loss.
     *                        Empty string => no breakdown shown. */
    void log_new_best(std::size_t eval_num,
                      double      loss,
                      double      emae,   bool has_emae,
                      double      ermse,  bool has_ermse,
                      double      errmse, bool has_errmse,
                      double      frmse,  bool has_frmse,
                      double      srmse,  bool has_srmse,
                      double      wall_ms,
                      const std::string& top_breakdown = "");

    /** EMBF0 journal line (DM_REAM runs only: hpo_run() arms it once the
     *  host is built; never armed, finish() prints nothing). finish() then
     *  prints one RESULT line with the raw embedding value at zero density,
     *  sum_m w_m g_m(0), of eval 1 and of the final best: the per-atom
     *  constant EMBF0 ZERO removes, or the isolated-atom embedding energy
     *  under FREE. HPO_Host records values only when @p derivable (a model
     *  linear in the raw descriptor); a value not recorded prints n/a.
     *  Diagnostic only: never feeds a loss, row or potential. HPO runs
     *  are not resumed: the values are this run's. */
    void arm_embf0(bool zero_mode, bool derivable);
    void record_embf0_start(std::size_t eval_num, double f0);
    /** Same guard as log_new_best(): kept only when @p loss is below the
     *  loss recorded with the previous value, so out-of-order journal calls
     *  from concurrent evaluations cannot leave a superseded best. */
    void record_embf0_best(std::size_t eval_num, double loss, double f0);

    /* ------------------------------------------------------------------
     * Annealer eval context (thread-local): which ladder slot / absolute
     * temperature the CURRENT thread's evaluations belong to. Stamped
     * per task by the parallel annealing drivers (next to the warm-start
     * key) and snapshotted by log_new_best(), so the journal can say
     * WHERE each best was found. Threads that never stamp a context
     * produce untagged bests -- serial drivers are unaffected.
     * ------------------------------------------------------------------ */

    static constexpr int kSlotNone  = -1;   /* no context stamped        */
    static constexpr int kSlotProbe = -2;   /* T0-calibration probe eval */

    /** Stamp the calling thread's evals as ladder slot @p slot of
     *  @p n_slots (0-based; the coldest slot, journalled as 1) at absolute
     *  temperature @p T. Stamp with
     *  the T at eval time so a cooling ladder (RT < 1) stays correct.
     *  @p t_label names the temperature in the journal tag (default
     *  "T"; ASA block slots stamp the COST temperature as "Tcost").
     *  Must point at a string with static storage duration. A NaN @p T
     *  omits the temperature from the tag entirely (slot-only tag). */
    static void set_eval_slot(int slot, std::size_t n_slots, double T,
                              const char* t_label = "T");
    /** Stamp the calling thread's evals as T0-calibration probes. */
    static void set_eval_probe();
    /** Clear the calling thread's eval context (bests become untagged). */
    static void clear_eval_context();

    /** Per-slot new-best counts accumulated since the last call (the
     *  drivers' per-epoch histogram), sorted by slot; probe bests
     *  (kSlotProbe) are not counted. Single-process semantics: under
     *  MPI the counts only cover THIS rank's slots. */
    std::vector<std::pair<int, std::size_t>> take_best_slot_window();
    /** Same histogram accumulated over the whole session. */
    std::vector<std::pair<int, std::size_t>> best_slot_totals() const;
    /** D29: eval number of the LAST new best recorded per slot, sorted by
     *  slot.  Separates "this slot still contributes" from "this slot's
     *  cumulative count is a fossil of the initial transient" (one observed long run's
     *  hot-slot counts all dated from the first ~500 of 20,457 bests).
     *  Single-process semantics, same as the histograms. */
    std::vector<std::pair<int, std::size_t>> best_slot_last_evals() const;

    /** Log a NaN/Inf clamping event (timestamped, with parameter dump).
     *  @param eval_num     Global evaluation counter
     *  @param params_str   Human-readable string of parameter name=value pairs
     *  @param bound_hints  Optional string with bound-proximity hints
     */
    void log_nan_clamped(std::size_t        eval_num,
                         const std::string& params_str,
                         const std::string& bound_hints = "");

    /* ------------------------------------------------------------------
     * Final summary sections (called once after optimisation)
     * ------------------------------------------------------------------ */

    /** Record an interim status string: why an optimiser run ended (every
     *  driver records one: "<LIB>:<ALGO>: <reason>", run_end.h; NLopt's
     *  result string; ASA's). Repeated calls overwrite -- the last status
     *  before finish() wins; open() and clear_status() clear it. Used by
     *  per-start drivers in MULTISTART so they don't have to call finish()
     *  themselves and prematurely close the journal.
     */
    void set_status(const std::string& status_str);
    /** The status last recorded by set_status() ("" if none). */
    std::string status() const;

    /** Forget the recorded status. Every optimiser run starts with none
     *  (run_inner), so the trailer is the reason of the run that ended the
     *  whole run, never one left behind by an earlier PRESTAGE phase,
     *  MULTISTART start or RESTART round. */
    void clear_status();

    /** Write DIAGNOSTICS + PERFORMANCE + RESULT sections and close the file.
     *  Best metrics are taken from what was recorded by log_new_best() calls.
     *  Idempotent: subsequent calls are no-ops (file is already closed).
     *  @param status_str  Human-readable stop reason. If empty, falls back to
     *                     whatever was most recently recorded via set_status(),
     *                     and without that to "(no status recorded)" (it used
     *                     to invent COMPLETED); drive_run() never finishes a
     *                     normal end without a status (run_driver.h).
     */
    void finish(const std::string& status_str = "");

    /* ------------------------------------------------------------------
     * Accumulated statistics (read by finish())
     * ------------------------------------------------------------------ */
    std::size_t n_eval()  const { return n_eval_.load(std::memory_order_relaxed); }
    std::size_t n_nan()   const { return n_nan_ .load(std::memory_order_relaxed); }
    /** Evaluations that raised an exception (EvalLoss::Threw). */
    std::size_t n_threw() const { return n_threw_.load(std::memory_order_relaxed); }
    /** Evaluations whose loss was not usable (EvalLoss other than Usable). */
    std::size_t n_unusable() const { return n_unusable_.load(std::memory_order_relaxed); }
    /** Current best total loss (max double if no BEST yet). */
    double      best_loss() const {
        std::lock_guard<std::mutex> lg(mtx_);
        return cur_best_loss_;
    }
    /** Eval at which current best was recorded (0 if no BEST yet). */
    std::size_t best_eval() const {
        std::lock_guard<std::mutex> lg(mtx_);
        return cur_best_eval_;
    }

    /* The notes a run prints once, and the counts of the notes it
     * rate-limits, are run state: a new journal (open(), every run) and
     * reset_for_test() start them afresh, so the second run of a process
     * prints its own and counts from 1. first_in_run(key) is true the
     * first time @p key is asked in this run; count_in_run(key) returns
     * the occurrences of @p key in this run, this one included. Thread-
     * safe. */
    bool        first_in_run(const std::string& key);
    std::size_t count_in_run(const std::string& key);

    /* Test-only: reset the singleton's running counters and best-so-far
     * to their freshly-constructed values (including the per-gate reject
     * and note counters, the timing statistics, the fan-out tallies, the
     * bound-occupancy snapshot and the per-run notes): the reset open()
     * makes for every run (reset_run_locked), without a new file.
     * Used by regression tests so one Catch2 test case's state does not
     * leak into the next. Does not close the underlying file -- call after
     * open() if you also want a fresh log file on disk. */
    void reset_for_test();

    /** "12.3 s" / "4.5 min" / "2.1 h" -- shared duration formatter (also
     *  used by the annealers for measured epoch/step wall times). */
    static std::string format_duration(double seconds);

private:
    /* The run state every journal starts from (open(), reset_for_test());
     * mtx_ held. */
    void reset_run_locked();

    HPOLog() = default;
    HPOLog(const HPOLog&) = delete;
    HPOLog& operator=(const HPOLog&) = delete;

    std::string now_hms()  const;   // "HH:MM:SS"
    std::string now_ymdhms() const; // "YYYY-MM-DD HH:MM:SS"

    /* event() body without the lock/flush -- lets the milestone writer
     * (which already holds mtx_) emit correctly starred+tagged blocks. */
    void event_locked(const std::string& tag, const std::string& msg);

    std::ofstream  f_;
    mutable std::mutex     mtx_;
    std::chrono::system_clock::time_point t_start_;
    /* set_rank_scope(): this journal's MPI rank and the run's rank count. */
    int scope_rank_  = 0;
    int scope_world_ = 1;

    /* thread-safe counters */
    std::atomic<std::size_t> n_eval_     {0};
    std::atomic<std::size_t> n_nan_      {0};
    std::atomic<std::size_t> n_nan_logged_{0};  // how many NaN events written
    std::atomic<std::size_t> n_threw_    {0};  // evaluations that threw
    /* Evaluations without a usable loss, whatever the reason: NaN/Inf
     * clampings, evaluations that threw, FAILSCORE (a skip cap, a failed
     * script) and graded failures. The DIAGNOSTICS verdict reads this
     * count, not the partial ones above and below. */
    std::atomic<std::size_t> n_unusable_ {0};
    /* Graded-FAILSCORE band tallies, index 1..4 = A..D (0 unused). Cheap
     * relaxed counters; surfaced in the milestone SUMMARY and the RESULT
     * trailer so a failing run is visibly failing. */
    std::atomic<std::size_t> n_fail_band_[5] = {};
    /* Pre-training gate rejects per GateKind (a band-D subset), and the
     * per-gate rate-limit counters of the reject notes (next_gate_note). */
    std::atomic<std::size_t> n_gate_[kGateKinds] = {};
    std::atomic<std::size_t> n_gate_note_[kGateKinds] = {};
    /* Positive-form projections (record_form_projection). */
    std::atomic<std::size_t> n_form_proj_ {0};
    /* Finite differences and the OPTIM box (record_out_of_box_eval, ...). */
    std::atomic<std::size_t> n_oob_eval_      {0};
    std::atomic<std::size_t> n_snap_          {0};
    std::atomic<std::size_t> n_oob_grad_      {0};
    std::atomic<std::size_t> n_hybrid_centre_ {0};
    /* Evaluations with rebuilt neighbour lists (record_nn_rebuild). */
    std::atomic<std::size_t> n_nn_rebuild_    {0};
    /* One-sided components per dimension, in the order first seen --
     * protected by mtx_. */
    std::vector<std::pair<std::string, std::size_t>> one_sided_;
    void reset_fd_counters_locked();
    /* Transient-I/O retries / losses (record_io_retry_* etc.). */
    std::atomic<std::size_t> n_io_retry_durable_{0};
    std::atomic<std::size_t> n_io_retry_writer_ {0};
    std::atomic<std::size_t> n_rows_lost_       {0};
    std::atomic<std::size_t> n_archives_lost_   {0};
    std::atomic<bool>        best_persist_failed_{false};

    /* timing stats -- protected by mtx_ */
    double      sum_ms_    = 0.0;
    double      min_ms_    = std::numeric_limits<double>::max();
    double      max_ms_    = 0.0;
    std::size_t min_eval_  = 0;
    std::size_t max_eval_  = 0;

    /* pc_lammps fan-out utilisation -- protected by mtx_. Two horizons:
     * *_win_ resets each milestone (drives the periodic SUMMARY line);
     * *_tot_ is cumulative (drives the final PERFORMANCE line). Pool
     * utilisation = Σ busy / (pool width x elapsed wall): busy time is
     * attributed to the window in which the eval FINISHES, so a single
     * window's figure can smear slightly across a boundary; the whole-run
     * figure in finish() has no boundary effect. Width comes from
     * set_fanout_width() (the shared executor); 0 = unknown -> the
     * utilisation figure is omitted rather than guessed. */
    double      fo_busy_win_      = 0.0;   // Σ per-script ms this window (= avg numerator)
    double      fo_smin_win_      = std::numeric_limits<double>::max();
    double      fo_smax_win_      = 0.0;
    std::size_t fo_scount_win_    = 0;
    std::string fo_slowest_win_;           // name of the window's worst script
    double      fo_busy_tot_      = 0.0;
    std::size_t fo_evals_         = 0;      // # evals that ran a fan-out
    int         fanout_pool_width_ = 0;     // shared executor width (0 = unset)
    double      fo_util_note_     = 0.60;   // neutral low-utilisation note below this

    /* optimiser work-unit cadence -- protected by mtx_. Set once by the
     * annealers via set_cadence(); drives the per-milestone "<unit> ~ T
     * at current rate" SUMMARY line. evals==0 -> line disabled. */
    std::size_t   cadence_evals_   = 0;
    std::string   cadence_unit_;
    std::uint64_t cadence_maxeval_ = 0;

    /* pinned-dim audit (armed by the CORANA drivers) -- protected by
     * mtx_. Key = (chain, dim, side 0=LO/1=HI); value = first unit the
     * mark was set + latest gap. Bounds/log flags are snapshotted at
     * arm time (labels at write_optimizer_section time) so the printer
     * never reaches back into optimiser state. pin_chains_ == 0 <=>
     * audit off. */
    struct PinInfo { std::uint64_t since; double gap; double tol; };
    std::map<std::tuple<int, std::size_t, int>, PinInfo> pin_state_;
    std::vector<std::string> pin_labels_;
    /* Bound-occupancy audit snapshot (latest new best of this run) --
     * reset by open() and reset_for_test(). */
    std::vector<BoundOccupancy::Mark> occ_marks_;
    std::uint64_t                     occ_eval_ = 0;
    double                            occ_edge_ = 0.02;
    bool                              occ_seen_ = false;   // any new best reported
    void write_bound_occupancy_locked(const char* indent_tag, bool final_table);
    void reset_occupancy_locked();
    /* first_in_run / count_in_run -- protected by mtx_; reset by open()
     * and reset_for_test(). */
    std::map<std::string, std::size_t> run_notes_;
    std::vector<double>      pin_low_, pin_high_;
    std::vector<char>        pin_log_;             // 1 = log-scaled dim
    double                   pin_floor_frac_ = 0.0;
    int                      pin_chains_     = 0;
    std::uint64_t            pin_unit_       = 0;  // latest unit seen
    std::string              pin_unit_name_;

    /* next milestone for periodic summary + the window over which the
     * WALL throughput is measured (latency alone misleads under the
     * parallel drivers: K concurrent workers inflate per-eval wall_ms
     * by ~K vs what the run actually delivers per unit time). */
    std::size_t next_milestone_ = 100;
    std::atomic<std::size_t> eval_base_{0};   // set_eval_base()
    std::chrono::steady_clock::time_point last_milestone_tp_{};
    std::size_t last_milestone_evals_ = 0;

    /* best result tracking -- set by log_new_best(), used by finish() */
    double      cur_best_loss_  = std::numeric_limits<double>::max();
    std::size_t cur_best_eval_  = 0;

    /* EMBF0 journal line (arm_embf0 / record_embf0_*) -- protected by mtx_;
     * reset by open() and reset_for_test(). */
    bool        embf0_armed_      = false;
    bool        embf0_zero_       = true;
    bool        embf0_derivable_  = false;
    std::size_t embf0_start_eval_ = 0;
    double      embf0_start_      = 0.0;
    std::size_t embf0_best_eval_  = 0;
    double      embf0_best_       = 0.0;
    double      embf0_best_loss_  = std::numeric_limits<double>::max();
    void reset_embf0_locked();

    /* Density RESULT line (write_density_report arms it; record_density_best
     * keeps the best's formatted rows) -- protected by mtx_; reset by
     * open() and reset_for_test(). */
    bool        density_armed_     = false;
    bool        density_rhobar_    = false;
    std::size_t density_best_eval_ = 0;
    double      density_best_loss_ = std::numeric_limits<double>::max();
    std::string density_best_rho_;
    std::string density_best_rhobar_;
    void reset_density_locked();

    /* Where the current best was found: snapshot of the recording
     * thread's eval context at log_new_best() time, plus per-slot
     * new-best histograms (window = since last take_best_slot_window(),
     * tot = whole session). Protected by mtx_. */
    int         cur_best_slot_   = kSlotNone;
    std::size_t cur_best_nslots_ = 0;
    double      cur_best_T_      = 0.0;
    const char* cur_best_T_label_ = "T";
    std::map<int, std::size_t> best_slot_win_;
    std::map<int, std::size_t> best_slot_tot_;
    std::map<int, std::size_t> best_slot_last_;   /* slot -> last best eval */

    /* ", slot 5/32 T=3.712e+00" | ", T0 probe" | "" -- mtx_ held. */
    std::string best_origin_str() const;

    /* "slot 5/32 T=3.712e+00" from a ZERO-BASED slot. The single place the
     * slot tag is rendered: the BEST line and best_origin_str() read the
     * slot from different members (tl_slot vs cur_best_slot_), so formatting
     * them separately is how the same best ends up reported under two
     * different numbers in one file. Callers keep their 0-based value --
     * it is also the key of the best_slot_* maps below. */
    static std::string slot_tag(int slot, std::size_t n_slots, double T,
                                const char* t_label);

    double      best_emae_      = 0; bool has_emae_   = false;
    double      best_ermse_     = 0; bool has_ermse_  = false;
    double      best_errmse_    = 0; bool has_errmse_ = false;
    double      best_frmse_     = 0; bool has_frmse_  = false;
    double      best_srmse_     = 0; bool has_srmse_  = false;

    /* last interim status set via set_status(); used as fallback in finish()
     * ("(no status recorded)" without one) */
    std::string last_status_;
    /* new best(s) that could not be persisted (record_unsaved_best) */
    std::optional<UnsavedBest> unsaved_best_;
    StartTally                 start_tally_;

    /* Lazy-section bookkeeping. event() emits a "-- SETUP DIAGNOSTICS --"
     * header on its very first call when sections_started_ is still false,
     * so that init-time diagnostics (NOISE_AUTO, InitStrategy, ...) emitted
     * during HPOTargets construction don't appear naked between the banner
     * and the first proper section. write_*_section() and start_progress()
     * set sections_started_ = true so subsequent events flow under PROGRESS
     * (the existing behaviour). */
    bool sections_started_     = false;
    bool wrote_setup_section_  = false;
};

/* =========================================================================
 * nlopt result -> human-readable string
 * ========================================================================= */
std::string nlopt_result_str(int code);

} // namespace tadah::hpo

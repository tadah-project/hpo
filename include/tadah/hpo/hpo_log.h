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
 *   -- POTENTIAL      --  -- 2B/MB, cutoffs, element list
 *   -- LOSS FUNCTION --  -- targets, weights, FAIL_SCORE
 *   -- PROGRESS      --  -- timestamped events: bests, NaN warnings, milestones
 *   -- DIAGNOSTICS   --  -- NaN rate, bound-proximity analysis
 *   -- PERFORMANCE   --  -- timing statistics
 *   -- RESULT        --  -- final status, best loss, best metrics
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <tadah/core/checkpoint.h>

// Forward declarations of HPO types used in section writers
namespace tadah::hpo { struct OptimizerSpec; class HPOTargets; }
namespace tadah::core { class Context; class Element; }
namespace tadah::mlip { struct LoadTransformReport; }

namespace tadah::hpo {

/* =========================================================================
 * HPOLog -- the logger.
 *
 * HPOLog also implements ICheckpointable so that the journal's running
 * counters (n_eval, best loss, best eval) survive across restarts. The
 * actual log file is opened in append mode by open() -- the
 * checkpointed counters let the journal stay coherent (eval numbers
 * keep monotonically increasing, "best so far" doesn't reset).
 * ========================================================================= */
class HPOLog : public core::ICheckpointable
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

    void write_potential_section(
        const core::Context& ctx,
        std::size_t          n_train,
        std::size_t          n_val,
        const std::string&   elements_str);

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

    /** Record timing and NaN flag for one completed evaluation.
     *  @param eval_num   Global evaluation counter (1-based)
     *  @param wall_ms    Wall time for this evaluation in milliseconds
     *  @param was_nan    True if NaN/Inf was detected and loss was clamped
     */
    void record_eval(std::size_t eval_num, double wall_ms, bool was_nan);

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

    /** Record an interim status string (e.g. NLopt's per-run termination
     *  reason). Repeated calls overwrite -- the last status before finish()
     *  wins. Used by per-start drivers in MULTISTART so they don't have to
     *  call finish() themselves and prematurely close the journal.
     */
    void set_status(const std::string& status_str);

    /** Write DIAGNOSTICS + PERFORMANCE + RESULT sections and close the file.
     *  Best metrics are taken from what was recorded by log_new_best() calls.
     *  Idempotent: subsequent calls are no-ops (file is already closed).
     *  @param status_str  Human-readable stop reason. If empty, falls back to
     *                     whatever was most recently recorded via set_status().
     */
    void finish(const std::string& status_str = "");

    /* ------------------------------------------------------------------
     * Accumulated statistics (read by finish())
     * ------------------------------------------------------------------ */
    std::size_t n_eval()  const { return n_eval_.load(std::memory_order_relaxed); }
    std::size_t n_nan()   const { return n_nan_ .load(std::memory_order_relaxed); }
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

    /** suggested FTOL_REL from NOISE_CALIBRATION (10 x rel_spread, floored
     *  at 1e-6). Returns nullopt if NOISE_CALIBRATION did not run.
     *  Persisted in the [HPOLOG] checkpoint section so a resumed BASINS
     *  run (which gates NOISE_CALIBRATION on `!resumed`) can still apply
     *  C3's `clamp_ladder_to_noise_floor`. */
    std::optional<double> noise_suggested_ftol() const {
        std::lock_guard<std::mutex> lg(mtx_);
        return noise_suggested_ftol_;
    }
    /** Set by hpo_run() right after NOISE_CALIBRATION completes. */
    void set_noise_suggested_ftol(double v) {
        std::lock_guard<std::mutex> lg(mtx_);
        noise_suggested_ftol_ = v;
    }

    /* ICheckpointable -- persists eval counters and best-so-far so an
     * interrupted run resumes with consistent journal state. */
    std::string tag() const override { return "HPOLOG"; }
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;

    /* Test-only: reset the singleton's running counters and best-so-far
     * to their freshly-constructed values. Used by regression tests so
     * one Catch2 test case's state does not leak into the next. Does
     * not close the underlying file -- call after open() if you also
     * want a fresh log file on disk. */
    void reset_for_test();

    /** "12.3 s" / "4.5 min" / "2.1 h" -- shared duration formatter (also
     *  used by the annealers for measured epoch/step wall times). */
    static std::string format_duration(double seconds);

private:
    HPOLog() = default;
    HPOLog(const HPOLog&) = delete;
    HPOLog& operator=(const HPOLog&) = delete;

    std::string now_hms()  const;   // "HH:MM:SS"
    std::string now_ymdhms() const; // "YYYY-MM-DD HH:MM:SS"

    std::ofstream  f_;
    mutable std::mutex     mtx_;
    std::chrono::system_clock::time_point t_start_;

    /* thread-safe counters */
    std::atomic<std::size_t> n_eval_     {0};
    std::atomic<std::size_t> n_nan_      {0};
    std::atomic<std::size_t> n_nan_logged_{0};  // how many NaN events written

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

    /* next milestone for periodic summary + the window over which the
     * WALL throughput is measured (latency alone misleads under the
     * parallel drivers: K concurrent workers inflate per-eval wall_ms
     * by ~K vs what the run actually delivers per unit time). */
    std::size_t next_milestone_ = 100;
    std::chrono::steady_clock::time_point last_milestone_tp_{};
    std::size_t last_milestone_evals_ = 0;

    /* best result tracking -- set by log_new_best(), used by finish() */
    double      cur_best_loss_  = std::numeric_limits<double>::max();
    std::size_t cur_best_eval_  = 0;

    /* C3: suggested FTOL_REL from NOISE_CALIBRATION. Persisted so a
     * resumed BASINS run (which skips re-running NOISE_CALIBRATION)
     * can still apply the ladder clamp on its first round. */
    std::optional<double> noise_suggested_ftol_;
    double      best_emae_      = 0; bool has_emae_   = false;
    double      best_ermse_     = 0; bool has_ermse_  = false;
    double      best_errmse_    = 0; bool has_errmse_ = false;
    double      best_frmse_     = 0; bool has_frmse_  = false;
    double      best_srmse_     = 0; bool has_srmse_  = false;

    /* last interim status set via set_status(); used as fallback in finish() */
    std::string last_status_;

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

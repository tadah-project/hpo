#pragma once
/*
 * HPOLog — human-readable run journal for Tadah HPO jobs.
 *
 * Thread-safe singleton.  Call open() once from the main thread before
 * the optimiser starts; all other methods are safe from any thread.
 *
 * Output file default: hpo_run.log  (written to the working directory)
 *
 * File layout
 * -----------
 *   ══ banner (version, cwd, PID, timestamp) ══
 *   ┄┄ FILES         ┄┄  — config / output files
 *   ┄┄ OPTIMISER     ┄┄  — algorithm, stop criteria, search space
 *   ┄┄ POTENTIAL      ┄┄  — 2B/MB, cutoffs, element list
 *   ┄┄ LOSS FUNCTION ┄┄  — targets, weights, FAIL_SCORE
 *   ┄┄ PROGRESS      ┄┄  — timestamped events: bests, NaN warnings, milestones
 *   ┄┄ DIAGNOSTICS   ┄┄  — NaN rate, bound-proximity analysis
 *   ┄┄ PERFORMANCE   ┄┄  — timing statistics
 *   ┄┄ RESULT        ┄┄  — final status, best loss, best metrics
 */

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations of HPO types used in section writers
namespace tadah::hpo { struct OptimizerSpec; class HPOTargets; }
namespace tadah::core { class Context; class Element; }

namespace tadah::hpo {

/* =========================================================================
 * HPOLog — the logger
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

    /* ------------------------------------------------------------------
     * Primitive writers (all thread-safe, append to file)
     * ------------------------------------------------------------------ */

    void writeln(const std::string& text = "");
    void section(const std::string& title);                 // ┄┄ TITLE ┄┄
    void kv     (const std::string& key,
                 const std::string& val,
                 int key_width = 22);                       // "  key : val"
    void indent (const std::string& text);                  // "  text"
    void separator();                                       // thin rule

    /** Timestamped event line: "[HH:MM:SS]  tag  msg" */
    void event(const std::string& tag, const std::string& msg);
    void info   (const std::string& msg) { event("      ", msg); }
    void warning(const std::string& msg) { event("WARNING", msg); }
    void diag   (const std::string& msg) { event("DIAG   ", msg); }

    /* ------------------------------------------------------------------
     * Structured section writers (implemented in hpo_log.cpp)
     * ------------------------------------------------------------------ */

    void write_files_section(
        const std::string& target_file,
        const std::string& val_file,
        const HPOTargets&  trg);

    void write_optimizer_section(
        const OptimizerSpec&             sp,
        const std::vector<std::string>&  param_labels);

    void write_potential_section(
        const core::Context& ctx,
        std::size_t          n_train,
        std::size_t          n_val,
        const std::string&   elements_str);

    void write_loss_section(const HPOTargets& trg);

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

    /** Log a new-best event (timestamped, written to file). */
    void log_new_best(std::size_t eval_num,
                      double      loss,
                      double      emae,   bool has_emae,
                      double      ermse,  bool has_ermse,
                      double      errmse, bool has_errmse,
                      double      frmse,  bool has_frmse,
                      double      srmse,  bool has_srmse,
                      double      wall_ms);

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

    /** Write DIAGNOSTICS + PERFORMANCE + RESULT sections and close the file.
     *  Best metrics are taken from what was recorded by log_new_best() calls.
     *  @param status_str  Human-readable stop reason (e.g. "ftol_rel reached")
     */
    void finish(const std::string& status_str);

    /* ------------------------------------------------------------------
     * Accumulated statistics (read by finish())
     * ------------------------------------------------------------------ */
    std::size_t n_eval()  const { return n_eval_.load(std::memory_order_relaxed); }
    std::size_t n_nan()   const { return n_nan_ .load(std::memory_order_relaxed); }

private:
    HPOLog() = default;
    HPOLog(const HPOLog&) = delete;
    HPOLog& operator=(const HPOLog&) = delete;

    std::string now_hms()  const;   // "HH:MM:SS"
    std::string now_ymdhms() const; // "YYYY-MM-DD HH:MM:SS"
    static std::string format_duration(double seconds);

    std::ofstream  f_;
    std::mutex     mtx_;
    std::chrono::system_clock::time_point t_start_;

    /* thread-safe counters */
    std::atomic<std::size_t> n_eval_     {0};
    std::atomic<std::size_t> n_nan_      {0};
    std::atomic<std::size_t> n_nan_logged_{0};  // how many NaN events written

    /* timing stats — protected by mtx_ */
    double      sum_ms_    = 0.0;
    double      min_ms_    = std::numeric_limits<double>::max();
    double      max_ms_    = 0.0;
    std::size_t min_eval_  = 0;
    std::size_t max_eval_  = 0;

    /* next milestone for periodic summary */
    std::size_t next_milestone_ = 100;

    /* best result tracking — set by log_new_best(), used by finish() */
    double      cur_best_loss_  = std::numeric_limits<double>::max();
    std::size_t cur_best_eval_  = 0;
    double      best_emae_      = 0; bool has_emae_   = false;
    double      best_ermse_     = 0; bool has_ermse_  = false;
    double      best_errmse_    = 0; bool has_errmse_ = false;
    double      best_frmse_     = 0; bool has_frmse_  = false;
    double      best_srmse_     = 0; bool has_srmse_  = false;
};

/* =========================================================================
 * nlopt result → human-readable string
 * ========================================================================= */
std::string nlopt_result_str(int code);

} // namespace tadah::hpo

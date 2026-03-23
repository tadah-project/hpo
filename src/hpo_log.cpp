#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace tadah::hpo {

/* ─────────────────────────────────────────────────────────────────────────
 * Internal helpers
 * ───────────────────────────────────────────────────────────────────────── */

static constexpr int  KW       = 24;   // key column width in kv()

static const std::string HEAVY(80, '=');
static const std::string LIGHT(80, '-');

static std::string fmt_double(double v, int prec = 4)
{
    std::ostringstream s;
    s << std::scientific << std::setprecision(prec) << v;
    return s.str();
}

static std::string fmt_millis(double ms)
{
    if (ms < 1000.0)  return std::to_string(static_cast<int>(ms)) + " ms";
    if (ms < 60000.0) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << ms / 1000.0 << " s";
        return s.str();
    }
    int m = static_cast<int>(ms / 60000.0);
    double s = (ms - m * 60000.0) / 1000.0;
    std::ostringstream oss;
    oss << m << "m " << std::fixed << std::setprecision(0) << s << "s";
    return oss.str();
}

/* ─────────────────────────────────────────────────────────────────────────
 * HPOLog — lifecycle
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::open(const std::string& path,
                  const std::string& version,
                  const std::string& cwd,
                  int pid)
{
    std::lock_guard<std::mutex> lg(mtx_);
    t_start_ = std::chrono::system_clock::now();

    f_.open(path, std::ios::out | std::ios::trunc);
    if (!f_.is_open()) {
        std::cerr << "[HPOLog] Cannot open log file: " << path << "\n";
        return;
    }

    auto tt = std::chrono::system_clock::to_time_t(t_start_);

    f_ << "\n" << HEAVY << "\n";
    f_ << "  TADAH Hyperparameter Optimisation — Run Journal\n";
    f_ << HEAVY << "\n\n";
    f_ << "  Tadah version  : " << version << "\n";
    f_ << "  Started        : " << std::put_time(std::localtime(&tt),
                                                  "%Y-%m-%d %H:%M:%S") << "\n";
    f_ << "  Working dir    : " << cwd << "\n";
    f_ << "  Log file       : " << path << "\n";
    f_ << "  Process ID     : " << pid << "\n";
    f_ << "\n";
    f_.flush();
}

/* ─────────────────────────────────────────────────────────────────────────
 * HPOLog — primitive writers
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::writeln(const std::string& text)
{
    std::lock_guard<std::mutex> lg(mtx_);
    f_ << text << "\n";
    f_.flush();
}

void HPOLog::section(const std::string& title)
{
    std::lock_guard<std::mutex> lg(mtx_);
    std::string line = "-- " + title + " ";
    while (line.size() < 72) line += '-';
    f_ << "\n" << line << "\n";
    f_.flush();
}

void HPOLog::kv(const std::string& key, const std::string& val, int key_width)
{
    std::lock_guard<std::mutex> lg(mtx_);
    f_ << "  " << std::left << std::setw(key_width) << key << ": " << val << "\n";
    f_.flush();
}

void HPOLog::indent(const std::string& text)
{
    std::lock_guard<std::mutex> lg(mtx_);
    f_ << "    " << text << "\n";
    f_.flush();
}

void HPOLog::separator()
{
    std::lock_guard<std::mutex> lg(mtx_);
    f_ << "  " << std::string(68, '-') << "\n";
    f_.flush();
}

void HPOLog::event(const std::string& tag, const std::string& msg)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;
    f_ << "[" << now_hms() << "]  " << std::left << std::setw(8) << tag
       << "  " << msg << "\n";
    f_.flush();
}

/* ─────────────────────────────────────────────────────────────────────────
 * HPOLog — internal time helpers
 * ───────────────────────────────────────────────────────────────────────── */

std::string HPOLog::now_hms() const
{
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::ostringstream s;
    s << std::put_time(std::localtime(&tt), "%H:%M:%S");
    return s.str();
}

std::string HPOLog::now_ymdhms() const
{
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::ostringstream s;
    s << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S");
    return s.str();
}

/* static */ std::string HPOLog::format_duration(double seconds)
{
    if (seconds < 60.0) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << seconds << "s";
        return s.str();
    }
    int h = static_cast<int>(seconds / 3600);
    int m = static_cast<int>((seconds - h * 3600) / 60);
    int s = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    if (h > 0) oss << h << "h ";
    oss << m << "m " << s << "s";
    return oss.str();
}

/* ─────────────────────────────────────────────────────────────────────────
 * Structured section writers
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::write_files_section(const std::string& target_file,
                                 const std::string& val_file,
                                 const HPOTargets&  trg)
{
    section("FILES");
    kv("HPO targets",    target_file);
    kv("Validation set", val_file);
    kv("Loss log",       trg.lossFile);
    kv("Outvar log",     trg.outvarFile);
    kv("Params log",     trg.paramsFile);
    kv("Stats log",      trg.statsFile);
    kv("Best loss log",  trg.bestLossFile);
    kv("Best potential", "best_pot.tadah");
    if (trg.dumpPotEvery != std::numeric_limits<std::size_t>::max())
        kv("Pot archive dir",  trg.dumpPotDir
           + "  (every " + std::to_string(trg.dumpPotEvery) + " evals)");
    if (trg.dumpBestPotEvery != std::numeric_limits<std::size_t>::max())
        kv("Best pot archive", trg.dumpBestPotDir
           + "  (every " + std::to_string(trg.dumpBestPotEvery) + " new bests)");
}

void HPOLog::write_optimizer_section(const OptimizerSpec&            sp,
                                     const std::vector<std::string>& param_labels)
{
    section("OPTIMISER");
    kv("Library / algo", sp.lib + " / " + sp.algo);
    kv("Dimensions",     std::to_string(sp.low.size()) + " parameter"
                         + (sp.low.size() != 1 ? "s" : "") + " being optimised");

    // ---- local optimizer
    if (sp.local)
        kv("Local optimizer", sp.local->lib + " / " + sp.local->algo);

    // ---- stop criteria
    {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Stop criteria:\n";
        if (sp.stop.maxeval)
            f_ << "    " << std::left << std::setw(KW-4) << "Max evaluations"
               << ": " << *sp.stop.maxeval << "\n";
        if (sp.stop.maxtime)
            f_ << "    " << std::left << std::setw(KW-4) << "Max time (s)"
               << ": " << *sp.stop.maxtime << "\n";
        if (sp.stop.ftol_rel)
            f_ << "    " << std::left << std::setw(KW-4) << "ftol_rel"
               << ": " << fmt_double(*sp.stop.ftol_rel) << "\n";
        if (sp.stop.ftol_abs)
            f_ << "    " << std::left << std::setw(KW-4) << "ftol_abs"
               << ": " << fmt_double(*sp.stop.ftol_abs) << "\n";
        if (sp.stop.xtol_rel)
            f_ << "    " << std::left << std::setw(KW-4) << "xtol_rel"
               << ": " << fmt_double(*sp.stop.xtol_rel) << "\n";
        if (sp.stop.xtol_abs)
            f_ << "    " << std::left << std::setw(KW-4) << "xtol_abs"
               << ": " << fmt_double(*sp.stop.xtol_abs) << "\n";
        if (sp.stop.stopval)
            f_ << "    " << std::left << std::setw(KW-4) << "stopval"
               << ": " << fmt_double(*sp.stop.stopval) << "\n";
        if (!sp.stop.maxeval && !sp.stop.maxtime && !sp.stop.ftol_rel
            && !sp.stop.ftol_abs && !sp.stop.xtol_rel && !sp.stop.stopval)
            f_ << "    (none set — optimiser may run indefinitely)\n";
        f_.flush();
    }

    // ---- algorithm-specific params
    if (sp.pop)            kv("Population size", std::to_string(*sp.pop));
    if (sp.vector_storage) kv("Vector storage",  std::to_string(*sp.vector_storage));
    if (sp.seed)           kv("Random seed",     std::to_string(*sp.seed));

    // ---- search space
    {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Search space (bounds in optimiser-internal space):\n";
        const std::size_t n = sp.low.size();
        for (std::size_t i = 0; i < n; ++i) {
            const std::string& lbl = (i < param_labels.size())
                                     ? param_labels[i] : "param[" + std::to_string(i) + "]";
            std::string row = "    " + lbl;
            // pad label
            if (row.size() < 28) row.resize(28, ' ');
            row += ": [" + fmt_double(sp.low[i], 3) + ",  " + fmt_double(sp.high[i], 3) + "]";
            if (i < sp.log_scaled.size() && sp.log_scaled[i])
                row += "  (log-scale — bounds are ln(value))";
            f_ << row << "\n";
        }
        f_.flush();
    }

    // ---- initial point
    if (!sp.optim_init.empty()) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Starting point: user-supplied\n";
        f_.flush();
    } else {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Starting point: random (uniform within bounds)\n";
        f_.flush();
    }
}

void HPOLog::write_potential_section(const core::Context& ctx,
                                     std::size_t          n_train,
                                     std::size_t          n_val,
                                     const std::string&   elements_str)
{
    section("POTENTIAL SETUP");

    auto bool_str = [](bool b) { return b ? "yes" : "no"; };

    bool init2b = false, initmb = false;
    if (ctx.contains("INIT2B")) init2b = ctx.get<bool>("INIT2B");
    if (ctx.contains("INITMB")) initmb = ctx.get<bool>("INITMB");

    kv("2-body terms",    std::string(bool_str(init2b))
                          + (init2b ? "" : "  (WARNING: neither 2B nor MB active!)"));
    kv("Many-body terms", bool_str(initmb));

    auto rcut_str = [&](const std::string& key) -> std::string {
        std::string s;
        std::size_t n = ctx.size(key);
        for (std::size_t i = 0; i < n; ++i) {
            if (i) s += ", ";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << ctx.get<double>(key, i);
            s += oss.str();
        }
        return s;
    };
    if (init2b && ctx.contains("RCUT2B"))
        kv("Cutoff 2B (Ang)", rcut_str("RCUT2B"));
    if (initmb && ctx.contains("RCUTMB"))
        kv("Cutoff MB (Ang)", rcut_str("RCUTMB"));

    kv("Training structures",   std::to_string(n_train));
    kv("Validation structures", std::to_string(n_val));
    kv("Elements",              elements_str);

    if (!init2b && !initmb) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "\n  !! Neither INIT2B nor INITMB is true — the potential has no terms.\n"
              "     Check your configuration file.\n";
        f_.flush();
    }
}

void HPOLog::write_loss_section(const HPOTargets& trg)
{
    section("LOSS FUNCTION");

    auto metric_line = [&](const std::string& name, bool used,
                           double target, double weight) {
        if (!used) return;
        std::ostringstream s;
        s << "weight=" << std::fixed << std::setprecision(2) << weight
          << "  target=" << std::scientific << std::setprecision(3) << target;
        kv(name, s.str());
    };

    metric_line("EMAE   (meV/atom)",  trg.use_emae,   trg.emae   * 1000.0, trg.w_emae);
    metric_line("ERMSE  (meV/atom)",  trg.use_ermse,  trg.ermse  * 1000.0, trg.w_ermse);
    metric_line("ErRMSE (meV/atom)",  trg.use_errmse, trg.errmse * 1000.0, trg.w_errmse);
    metric_line("ERSQ",               trg.use_ersq,   trg.ersq,            trg.w_ersq);
    metric_line("FRMSE  (eV/Å)",      trg.use_frmse,  trg.frmse,           trg.w_frmse);
    metric_line("SRMSE  (GPa)",       trg.use_srmse,  trg.srmse,           trg.w_srmse);

    for (const auto& lsim : trg.LS) {
        for (std::size_t i = 0; i < lsim.n_varloss.size(); ++i) {
            std::ostringstream s;
            s << "weight=" << std::fixed << std::setprecision(2) << lsim.w_varloss[i]
              << "  target=" << std::scientific << std::setprecision(3) << lsim.t_varloss[i];
            kv(lsim.n_varloss[i], s.str());
        }
    }

    kv("Aggregate",   "mean over active components");
    kv("Fail score",  fmt_double(trg.FAIL_SCORE)
                      + "  (returned when training produces NaN/Inf)");
    kv("Output every", std::to_string(trg.outputEvery) + " eval(s)");
}

/* ─────────────────────────────────────────────────────────────────────────
 * Progress section
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::start_progress()
{
    section("PROGRESS");
    {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  [HH:MM:SS]  tag       message\n";
        f_ << "  " << std::string(68, '-') << "\n";
        f_.flush();
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Per-eval recording
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::record_eval(std::size_t eval_num, double wall_ms, bool was_nan)
{
    n_eval_.fetch_add(1, std::memory_order_relaxed);
    if (was_nan) n_nan_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lg(mtx_);
        sum_ms_ += wall_ms;
        if (wall_ms < min_ms_) { min_ms_ = wall_ms; min_eval_ = eval_num; }
        if (wall_ms > max_ms_) { max_ms_ = wall_ms; max_eval_ = eval_num; }
    }

    // Periodic milestone summary
    if (eval_num >= next_milestone_) {
        std::lock_guard<std::mutex> lg(mtx_);
        if (eval_num >= next_milestone_) { // re-check under lock
            std::size_t done    = n_eval_.load(std::memory_order_relaxed);
            std::size_t nan_cnt = n_nan_.load(std::memory_order_relaxed);
            double avg = (done > 0) ? sum_ms_ / done : 0.0;
            double nan_pct = (done > 0) ? 100.0 * nan_cnt / done : 0.0;

            std::ostringstream msg;
            msg << "--- " << std::setw(6) << eval_num << " evals done"
                << "   avg=" << fmt_millis(avg) << "/eval"
                << "   best=" << fmt_double(cur_best_loss_)
                << " (eval " << cur_best_eval_ << ")"
                << "   NaN rate=" << std::fixed << std::setprecision(1)
                << nan_pct << "%";
            f_ << "[" << now_hms() << "]  SUMMARY  "
               << msg.str() << "\n";
            f_.flush();

            // schedule next milestone: 100, 200, 500, 1000, 2000, 5000, ...
            if      (next_milestone_ <  500)  next_milestone_ += 100;
            else if (next_milestone_ < 2000)  next_milestone_ += 500;
            else if (next_milestone_ < 10000) next_milestone_ += 1000;
            else                              next_milestone_ += 5000;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * Progress events
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::log_new_best(std::size_t eval_num,
                          double      loss,
                          double      emae,   bool has_emae,
                          double      ermse,  bool has_ermse,
                          double      errmse, bool has_errmse,
                          double      frmse,  bool has_frmse,
                          double      srmse,  bool has_srmse,
                          double      wall_ms)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    cur_best_loss_ = loss;
    cur_best_eval_ = eval_num;
    if (has_emae)   { best_emae_   = emae;   has_emae_   = true; }
    if (has_ermse)  { best_ermse_  = ermse;  has_ermse_  = true; }
    if (has_errmse) { best_errmse_ = errmse; has_errmse_ = true; }
    if (has_frmse)  { best_frmse_  = frmse;  has_frmse_  = true; }
    if (has_srmse)  { best_srmse_  = srmse;  has_srmse_  = true; }

    std::ostringstream msg;
    msg << "★ NEW BEST   eval=" << std::setw(6) << eval_num
        << "  loss=" << fmt_double(loss);
    if (has_emae)   msg << "  EMAE="   << std::fixed << std::setprecision(1)
                        << emae   << " meV/at";
    if (has_ermse)  msg << "  ERMSE="  << std::fixed << std::setprecision(1)
                        << ermse  << " meV/at";
    if (has_errmse) msg << "  ErRMSE=" << std::fixed << std::setprecision(1)
                        << errmse << " meV/at";
    if (has_frmse)  msg << "  FRMSE="  << std::fixed << std::setprecision(3)
                        << frmse  << " eV/Å";
    if (has_srmse)  msg << "  SRMSE="  << std::fixed << std::setprecision(3)
                        << srmse  << " GPa";
    msg << "  (" << fmt_millis(wall_ms) << ")";

    f_ << "[" << now_hms() << "]  BEST     " << msg.str() << "\n";
    f_.flush();
}

void HPOLog::log_nan_clamped(std::size_t        eval_num,
                             const std::string& params_str,
                             const std::string& bound_hints)
{
    const std::size_t logged = n_nan_logged_.fetch_add(1, std::memory_order_relaxed);
    // Log every NaN for the first 10, then every 10th to avoid log flooding
    if (logged >= 10 && (logged % 10 != 0)) return;

    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    f_ << "[" << now_hms() << "]  WARNING  "
       << "Eval " << std::setw(6) << eval_num
       << ": training produced NaN/Inf — loss clamped to FAIL_SCORE\n";
    if (!params_str.empty())
        f_ << "           Parameters: " << params_str << "\n";
    if (!bound_hints.empty())
        f_ << "           " << bound_hints << "\n";
    if (logged == 9)
        f_ << "           (further NaN events will be summarised every 10 occurrences)\n";
    f_.flush();
}

/* ─────────────────────────────────────────────────────────────────────────
 * Final summary
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::finish(const std::string& status_str)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    const std::size_t total  = n_eval_.load(std::memory_order_relaxed);
    const std::size_t nnan   = n_nan_ .load(std::memory_order_relaxed);
    const double nan_pct     = total > 0 ? 100.0 * nnan / total : 0.0;
    const double avg_ms      = total > 0 ? sum_ms_ / total : 0.0;
    const double wall_sec    = std::chrono::duration<double>(
                                   std::chrono::system_clock::now() - t_start_).count();

    /* ── DIAGNOSTICS ─────────────────────────────────────────────────── */
    f_ << "\n"; { std::string s = "-- DIAGNOSTICS "; while(s.size()<72) s+='-'; f_ << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Total evaluations"
       << ": " << total << "\n";
    f_ << "  " << std::left << std::setw(KW) << "NaN/Inf clampings"
       << ": " << nnan;
    if (total > 0)
        f_ << "  (" << std::fixed << std::setprecision(1) << nan_pct << "%)";
    f_ << "\n";

    // NaN rate interpretation
    if (nnan == 0) {
        f_ << "  ✓ No NaN clampings — search space looks well-configured.\n";
    } else if (nan_pct < 5.0) {
        f_ << "  ✓ NaN rate < 5% — occasional numerical instability, normal.\n";
    } else if (nan_pct < 20.0) {
        f_ << "  ⚠ NaN rate " << std::fixed << std::setprecision(1) << nan_pct
           << "% — the optimiser is frequently hitting bad regions.\n"
              "    Consider narrowing the search bounds or adjusting FAIL_SCORE.\n";
    } else {
        f_ << "  ✗ NaN rate " << std::fixed << std::setprecision(1) << nan_pct
           << "% — most evaluations are failing numerically.\n"
              "    The search space likely includes parameter values where the\n"
              "    linear system is severely ill-conditioned.  Common causes:\n"
              "      • Regularisation (LAMBDA) lower bound is too small\n"
              "      • Descriptor parameters (SIGMA/ETA) allow too-peaked functions\n"
              "      • Cutoff radii (RCUT2B/RCUTMB) too small for the training data\n"
              "    Action: check the NaN warnings in the PROGRESS section above\n"
              "    and tighten the bounds for the offending parameters.\n";
    }

    /* ── PERFORMANCE ─────────────────────────────────────────────────── */
    { std::string s = "-- PERFORMANCE "; while(s.size()<72) s+='-'; f_ << "\n" << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Total wall time"
       << ": " << format_duration(wall_sec) << "\n";
    f_ << "  " << std::left << std::setw(KW) << "Mean per eval"
       << ": " << fmt_millis(avg_ms) << "\n";
    if (min_ms_ < std::numeric_limits<double>::max())
        f_ << "  " << std::left << std::setw(KW) << "Fastest eval"
           << ": " << fmt_millis(min_ms_) << "  (eval " << min_eval_ << ")\n";
    if (max_ms_ > 0.0)
        f_ << "  " << std::left << std::setw(KW) << "Slowest eval"
           << ": " << fmt_millis(max_ms_) << "  (eval " << max_eval_ << ")\n";
    if (wall_sec > 0 && total > 0) {
        std::ostringstream thr;
        thr << std::fixed << std::setprecision(1) << total / wall_sec * 60.0;
        f_ << "  " << std::left << std::setw(KW) << "Throughput"
           << ": " << thr.str() << " evals/min\n";
    }

    /* ── RESULT ──────────────────────────────────────────────────────── */
    { std::string s = "-- RESULT "; while(s.size()<72) s+='-'; f_ << "\n" << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Optimiser status"
       << ": " << status_str << "\n";
    f_ << "  " << std::left << std::setw(KW) << "Best total loss"
       << ": " << fmt_double(cur_best_loss_)
       << "  (eval " << cur_best_eval_ << ")\n";
    if (has_emae_)
        f_ << "  " << std::left << std::setw(KW) << "Best EMAE"
           << ": " << std::fixed << std::setprecision(2) << best_emae_ << " meV/atom\n";
    if (has_ermse_)
        f_ << "  " << std::left << std::setw(KW) << "Best ERMSE"
           << ": " << std::fixed << std::setprecision(2) << best_ermse_ << " meV/atom\n";
    if (has_errmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best ErRMSE"
           << ": " << std::fixed << std::setprecision(2) << best_errmse_ << " meV/atom\n";
    if (has_frmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best FRMSE"
           << ": " << std::fixed << std::setprecision(4) << best_frmse_ << " eV/Å\n";
    if (has_srmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best SRMSE"
           << ": " << std::fixed << std::setprecision(4) << best_srmse_ << " GPa\n";
    f_ << "  " << std::left << std::setw(KW) << "Best potential saved"
       << ": best_pot.tadah\n";

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    f_ << "  " << std::left << std::setw(KW) << "Finished"
       << ": " << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << "\n";

    f_ << "\n" << HEAVY << "\n\n";
    f_.flush();
    f_.close();
}

/* ─────────────────────────────────────────────────────────────────────────
 * nlopt result code → human-readable string
 * ───────────────────────────────────────────────────────────────────────── */
std::string nlopt_result_str(int code)
{
    switch (code) {
        case  1: return "SUCCESS — generic success";
        case  2: return "STOPVAL_REACHED — objective below stopval threshold";
        case  3: return "FTOL_REACHED — function tolerance converged";
        case  4: return "XTOL_REACHED — parameter tolerance converged";
        case  5: return "MAXEVAL_REACHED — maximum evaluations exhausted";
        case  6: return "MAXTIME_REACHED — maximum time exhausted";
        case -1: return "FAILURE — generic failure (often caused by NaN objective values)";
        case -2: return "INVALID_ARGS — invalid algorithm arguments (check bounds/dimensions)";
        case -3: return "OUT_OF_MEMORY — insufficient memory for algorithm";
        case -4: return "ROUNDOFF_LIMITED — progress blocked by floating-point roundoff";
        case -5: return "FORCED_STOP — optimisation was interrupted";
        default: return "UNKNOWN (" + std::to_string(code) + ")";
    }
}

} // namespace tadah::hpo

#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/column_titles.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/search_space.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/models/descriptors/dm/embf0.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace tadah::hpo {

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

static constexpr int  KW       = 24;   // key column width in kv()

static const std::string HEAVY(80, '=');
static const std::string LIGHT(80, '-');

static std::string fmt_double(double v, int prec = 4)
{
    std::ostringstream s;
    s << std::scientific << std::setprecision(prec) << v;
    return s.str();
}

/* The GRAD_STRATEGY a block obtains its gradients with, for the journal:
 * the keyword, "(resolved for PC_OBSERVABLE)" when the parser set it, or
 * the NUMERICAL default. */
static std::string grad_strategy_text(const OptimizerSpec& sp)
{
    if (!sp.grad_strategy)
        return "NUMERICAL (default: finite differences)";
    std::string s = grad_strategy_name(*sp.grad_strategy);
    if (sp.grad_strategy_resolved)
        s += " (resolved for PC_OBSERVABLE)";
    return s;
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

/* -------------------------------------------------------------------------
 * HPOLog -- annealer eval context (thread-local)
 * ------------------------------------------------------------------------- */

namespace {
thread_local int         tl_slot   = HPOLog::kSlotNone;
thread_local std::size_t tl_nslots = 0;
thread_local double      tl_T      = 0.0;
thread_local const char* tl_T_label = "T";
}

void HPOLog::set_eval_slot(int slot, std::size_t n_slots, double T,
                           const char* t_label)
{
    tl_slot   = slot;
    tl_nslots = n_slots;
    tl_T      = T;
    tl_T_label = t_label;
}

void HPOLog::set_eval_probe()
{
    tl_slot   = kSlotProbe;
    tl_nslots = 0;
    tl_T      = 0.0;
}

void HPOLog::clear_eval_context()
{
    tl_slot   = kSlotNone;
    tl_nslots = 0;
    tl_T      = 0.0;
}

std::vector<std::pair<int, std::size_t>> HPOLog::take_best_slot_window()
{
    std::lock_guard<std::mutex> lg(mtx_);
    std::vector<std::pair<int, std::size_t>> out(best_slot_win_.begin(),
                                                 best_slot_win_.end());
    best_slot_win_.clear();
    return out;
}

std::vector<std::pair<int, std::size_t>> HPOLog::best_slot_totals() const
{
    std::lock_guard<std::mutex> lg(mtx_);
    return {best_slot_tot_.begin(), best_slot_tot_.end()};
}

std::vector<std::pair<int, std::size_t>> HPOLog::best_slot_last_evals() const
{
    std::lock_guard<std::mutex> lg(mtx_);
    return {best_slot_last_.begin(), best_slot_last_.end()};
}

std::string HPOLog::slot_tag(int slot, std::size_t n_slots, double T,
                             const char* t_label)
{
    std::ostringstream s;
    s << "slot " << logidx::uis(static_cast<std::size_t>(slot)) << "/"
      << n_slots;
    if (!std::isnan(T))
        s << " " << t_label << "=" << std::scientific << std::setprecision(3)
          << T;
    return s.str();
}

std::string HPOLog::best_origin_str() const
{
    if (cur_best_slot_ == kSlotProbe) return ", T0 probe";
    if (cur_best_slot_ < 0) return "";
    return ", " + slot_tag(cur_best_slot_, cur_best_nslots_, cur_best_T_,
                           cur_best_T_label_);
}

/* -------------------------------------------------------------------------
 * HPOLog -- lifecycle
 * ------------------------------------------------------------------------- */

void HPOLog::reset_for_test()
{
    std::lock_guard<std::mutex> lg(mtx_);
    n_eval_.store(0, std::memory_order_relaxed);
    n_nan_ .store(0, std::memory_order_relaxed);
    n_nan_logged_.store(0, std::memory_order_relaxed);
    n_unusable_.store(0, std::memory_order_relaxed);
    for (auto& b : n_fail_band_) b.store(0, std::memory_order_relaxed);
    for (auto& g : n_gate_) g.store(0, std::memory_order_relaxed);
    for (auto& g : n_gate_note_) g.store(0, std::memory_order_relaxed);
    n_form_proj_.store(0, std::memory_order_relaxed);
    reset_fd_counters_locked();
    n_io_retry_durable_.store(0, std::memory_order_relaxed);
    n_io_retry_writer_ .store(0, std::memory_order_relaxed);
    n_rows_lost_       .store(0, std::memory_order_relaxed);
    n_archives_lost_   .store(0, std::memory_order_relaxed);
    best_persist_failed_.store(false, std::memory_order_relaxed);
    unsaved_best_.reset();
    start_tally_ = StartTally{};
    last_status_.clear();
    cur_best_loss_ = std::numeric_limits<double>::max();
    cur_best_eval_ = 0;
    best_emae_ = best_ermse_ = best_errmse_ = best_frmse_ = best_srmse_ = 0.0;
    has_emae_ = has_ermse_ = has_errmse_ = has_frmse_ = has_srmse_ = false;
    cadence_evals_   = 0;
    cadence_unit_.clear();
    cadence_maxeval_ = 0;
    cur_best_slot_   = kSlotNone;
    cur_best_nslots_ = 0;
    cur_best_T_      = 0.0;
    cur_best_T_label_ = "T";
    best_slot_win_.clear();
    best_slot_tot_.clear();
    best_slot_last_.clear();
    pin_state_.clear();
    pin_chains_ = 0;
    pin_unit_   = 0;
    next_milestone_       = 100;
    last_milestone_evals_ = 0;
    scope_rank_  = 0;
    scope_world_ = 1;
    /* timing and fan-out statistics (PERFORMANCE, SUMMARY) */
    sum_ms_   = 0.0;
    min_ms_   = std::numeric_limits<double>::max();
    max_ms_   = 0.0;
    min_eval_ = 0;
    max_eval_ = 0;
    fo_busy_win_       = 0.0;
    fo_smin_win_       = std::numeric_limits<double>::max();
    fo_smax_win_       = 0.0;
    fo_scount_win_     = 0;
    fo_slowest_win_.clear();
    fo_busy_tot_       = 0.0;
    fo_evals_          = 0;
    fanout_pool_width_ = 0;
    reset_embf0_locked();
    reset_density_locked();
    reset_occupancy_locked();
    run_notes_.clear();
    clear_eval_context();
}

void HPOLog::reset_occupancy_locked()
{
    occ_marks_.clear();
    occ_eval_ = 0;
    occ_edge_ = 0.02;
    occ_seen_ = false;
}

bool HPOLog::first_in_run(const std::string& key)
{
    std::lock_guard<std::mutex> lg(mtx_);
    return ++run_notes_[key] == 1;
}

std::size_t HPOLog::count_in_run(const std::string& key)
{
    std::lock_guard<std::mutex> lg(mtx_);
    return ++run_notes_[key];
}

void HPOLog::set_rank_scope(int rank, int world)
{
    std::lock_guard<std::mutex> lg(mtx_);
    scope_rank_  = rank;
    scope_world_ = world;
}

void HPOLog::reset_embf0_locked()
{
    embf0_armed_      = false;
    embf0_zero_       = true;
    embf0_derivable_  = false;
    embf0_start_eval_ = 0;
    embf0_start_      = 0.0;
    embf0_best_eval_  = 0;
    embf0_best_       = 0.0;
    embf0_best_loss_  = std::numeric_limits<double>::max();
}

void HPOLog::reset_density_locked()
{
    density_armed_     = false;
    density_rhobar_    = false;
    density_best_eval_ = 0;
    density_best_loss_ = std::numeric_limits<double>::max();
    density_best_rho_.clear();
    density_best_rhobar_.clear();
}

void HPOLog::open(const std::string& path,
                  const std::string& version,
                  const std::string& cwd,
                  int pid)
{
    std::lock_guard<std::mutex> lg(mtx_);
    t_start_ = std::chrono::system_clock::now();
    /* A new journal starts without a termination status: a status recorded
     * for an earlier run in this process (HPOLog is a singleton) must not
     * become this run's trailer. */
    last_status_.clear();
    /* A journal an earlier run in this process left open -- one that ended
     * before its trailer, with an error during its setup -- is closed
     * first: opening an open stream fails, and this run's journal would
     * silently go nowhere. */
    if (f_.is_open()) f_.close();
    f_.clear();
    start_tally_ = StartTally{};
    reset_embf0_locked();
    reset_density_locked();
    /* The bound-occupancy snapshot and the per-run notes belong to the
     * run: an earlier run's best must not become this journal's final
     * table, nor its notes silence or number this run's. */
    reset_occupancy_locked();
    run_notes_.clear();
    /* The finite-difference counters of DIAGNOSTICS are this run's. */
    reset_fd_counters_locked();

    f_.open(path, std::ios::out | std::ios::trunc);
    if (!f_.is_open()) {
        std::cerr << "[HPOLog] Cannot open log file: " << path << "\n";
        return;
    }

    auto tt = std::chrono::system_clock::to_time_t(t_start_);

    f_ << "\n" << HEAVY << "\n";
    f_ << "  TADAH Hyperparameter Optimisation -- Run Journal\n";
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

/* -------------------------------------------------------------------------
 * HPOLog -- primitive writers
 * ------------------------------------------------------------------------- */

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
    /* Any "real" section header marks the end of the lazy SETUP DIAGNOSTICS
     * region so subsequent events flow inline as PROGRESS-style entries. */
    sections_started_ = true;
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
    /* Lazy SETUP DIAGNOSTICS header: emit before the first event when no
     * other section has been started yet (init-time diagnostics that fire
     * during HPOTargets construction land here). */
    if (!sections_started_ && !wrote_setup_section_) {
        std::string s = "-- SETUP DIAGNOSTICS ";
        while (s.size() < 72) s += '-';
        f_ << s << "\n";
        wrote_setup_section_ = true;
    }
    event_locked(tag, msg);
    f_.flush();
}

void HPOLog::event_locked(const std::string& tag, const std::string& msg)
{
    /* Multi-line entries: the block's FIRST line carries the (single)
     * timestamp and a star marking where the block starts; continuation
     * lines blank the timestamp column but repeat the tag, so any grep
     * hit stays attributed to its severity. Single-line entries stay
     * unstarred. */
    const bool multi = msg.find('\n') != std::string::npos;
    std::size_t start = 0;
    bool        first = true;
    for (;;) {
        const std::size_t nl = msg.find('\n', start);
        if (first) f_ << "[" << now_hms() << "]  ";
        else       f_ << std::string(12, ' ');
        f_ << std::left << std::setw(8) << tag << "  ";
        if (multi) f_ << (first ? "* " : "  ");
        f_ << msg.substr(start,
                         nl == std::string::npos ? std::string::npos
                                                 : nl - start) << "\n";
        first = false;
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
}

/* -------------------------------------------------------------------------
 * HPOLog -- internal time helpers
 * ------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------
 * Structured section writers
 * ------------------------------------------------------------------------- */

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
    /* Strategy-aware library/algo summary. PRESTAGE shows OUTER as the
     * refinement driver and reports EXPLORE separately under Run policy
     * below. */
    kv("Strategy", std::string(strategy_type_name(sp.strategy_type)));
    kv("Library / algo", sp.lib + " / " + sp.algo);
    kv("Dimensions",     std::to_string(sp.low.size()) + " parameter"
                         + (sp.low.size() != 1 ? "s" : "") + " being optimised");

    // ---- inner optimizer (NLopt MLSL/AUGLAG sub-algo)
    if (sp.inner)
        kv("Inner optimizer", sp.inner->lib + " / " + sp.inner->algo);

    // ---- how the block obtains gradients (for its INNER search too; an
    //      INNER block takes no GRAD_STRATEGY of its own)
    if (obtains_gradients(sp))
        kv("GRAD_STRATEGY", grad_strategy_text(sp));

    // ---- how each phase that takes finite-difference gradients takes them
    //      (fd_stencil.h): the step, its rule and source, and what happens
    //      next to a bound
    {
        std::vector<std::pair<std::string, std::string>> fd_lines;
        try {
            for (const PhaseSpec& phase : strategy_phase_specs(sp)) {
                if (!takes_fd_gradients(phase)) continue;
                std::string d;
                try {
                    const bool hybrid =
                        phase.spec.grad_strategy.value_or(
                            GradStrategy::Numerical) == GradStrategy::Hybrid;
                    d = fd_describe(fd_settings_of(phase.spec),
                                    phase.spec.space &&
                                        phase.spec.space->normalised(),
                                    hybrid);
                } catch (const std::exception& e) {
                    d = std::string("invalid: ") + e.what();
                }
                fd_lines.emplace_back(phase.name, d);
            }
        } catch (const std::exception&) {
            /* an optimiser the registry does not know: validate_hpo_setup
             * reports it */
        }
        for (const auto& l : fd_lines)
            kv("Finite differences",
               fd_lines.size() == 1 ? l.second : l.first + ": " + l.second);
    }

    auto write_stop_criteria = [&](const StopCriteria& s, const char* indent) {
        if (s.maxeval)
            f_ << indent << std::left << std::setw(KW-4) << "Max evaluations"
               << ": " << *s.maxeval << "\n";
        if (s.maxtime)
            f_ << indent << std::left << std::setw(KW-4) << "Max time (s)"
               << ": " << *s.maxtime << "\n";
        if (s.ftol_rel)
            f_ << indent << std::left << std::setw(KW-4) << "ftol_rel"
               << ": " << fmt_double(*s.ftol_rel) << "\n";
        if (s.ftol_abs)
            f_ << indent << std::left << std::setw(KW-4) << "ftol_abs"
               << ": " << fmt_double(*s.ftol_abs) << "\n";
        if (s.xtol_rel)
            f_ << indent << std::left << std::setw(KW-4) << "xtol_rel"
               << ": " << fmt_double(*s.xtol_rel) << "\n";
        if (s.xtol_abs)
            f_ << indent << std::left << std::setw(KW-4) << "xtol_abs"
               << ": " << fmt_double(*s.xtol_abs) << "\n";
        if (s.stopval)
            f_ << indent << std::left << std::setw(KW-4) << "stopval"
               << ": " << fmt_double(*s.stopval) << "\n";
    };
    auto stop_is_empty = [](const StopCriteria& s) {
        return !s.maxeval && !s.maxtime && !s.ftol_rel && !s.ftol_abs
            && !s.xtol_rel && !s.xtol_abs && !s.stopval;
    };

    // ---- stop criteria (outer)
    {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Stop criteria:\n";
        write_stop_criteria(sp.stop, "    ");
        /* A backend that cannot end without a stop was refused at startup
         * (require_stop, backend_rules.h), unless an active RESTART's
         * stagnation stop ends it. */
        const OptimizerCaps* caps = nullptr;
        try { caps = &optimizer_caps(sp); } catch (const std::exception&) {}
        const bool budget_per_start =
            sp.strategy_type == StrategyType::MULTISTART &&
            sp.multistart.budget_per_start.has_value();
        if (stop_is_empty(sp.stop) && budget_per_start) {
            f_ << "    (none set here -- each MULTISTART start runs with "
                  "BUDGET_PER_START " << *sp.multistart.budget_per_start
               << ")\n";
        } else if (caps && caps->needs_stop && sp.restart.active() &&
                   !sets_own_stop(sp)) {
            f_ << "    (" << (stop_is_empty(sp.stop) ? "none set" :
                              "none that ends it")
               << " -- the RESTART stagnation stop ends " << sp.lib << ":"
               << sp.algo << ": once its MAX " << sp.restart.max
               << " restart(s) are used, AFTER_STAGNATION "
               << sp.restart.after_stagnation
               << " evaluations without a new best)\n";
        } else if (stop_is_empty(sp.stop)) {
            f_ << "    (none set -- " << sp.lib << ":" << sp.algo
               << " ends on its own: a convergence test, a finite "
                  "schedule or limits of its own)\n";
        }
        f_.flush();
    }

    // ---- stop criteria (INNER sub-spec).
    // Critical for NLopt MLSL/G_MLSL_LDS: with no INNER stop criteria
    // set, NLopt defaults to ftol_rel=1e-15 and xtol_rel=1e-7 for the
    // inner local searches -- far tighter than a noisy LAMMPS objective
    // can ever satisfy, so each descent runs until something else
    // intervenes. Surface the resolved values (or the silent default)
    // so the user can see what is actually being asked of the inner.
    if (sp.inner) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Inner stop criteria:\n";
        write_stop_criteria(sp.inner->stop, "    ");
        /* NLopt hands each local search an evaluation limit in its own
         * count, objective calls, and the objective cannot tell the local
         * searches apart: the INNER MAXEVAL keeps that count (owner
         * decision, 2026-10-07), while run_nlopt counts the enclosing
         * block's MAXEVAL in evaluations. The two differ for a gradient
         * search. */
        bool inner_gradient = false;
        try {
            inner_gradient = optimizer_caps(*sp.inner).is_gradient_based;
        } catch (const std::exception&) {}
        if (sp.inner->stop.maxeval && inner_gradient)
            f_ << "    (MAXEVAL here counts the NLopt calls of each local "
                  "search, not its evaluations: a call evaluates the loss "
                  "and its gradient; the MAXEVAL of the enclosing block "
                  "counts every evaluation of the run, each loss, each "
                  "finite-difference point and each analytical gradient)\n";
        if (stop_is_empty(sp.inner->stop)) {
            const bool is_mlsl = (sp.lib == "NLOPT" &&
                (sp.algo == "G_MLSL" || sp.algo == "G_MLSL_LDS"));
            if (is_mlsl) {
                f_ << "    (none set -- NLopt MLSL will use defaults "
                      "ftol_rel=1e-15 xtol_rel=1e-7 for inner local searches; "
                      "loosen these if inner descents stall on a noisy objective)\n";
            } else {
                f_ << "    (none set -- the INNER search runs with the "
                      "enclosing block's MAXEVAL, MAXTIME and STOPVAL, or ends "
                      "on its own)\n";
            }
        }
        f_.flush();
    }

    // ---- initial step (config level; the effective per-dim vector is
    //      journaled by the NLopt adapter at run time, where the start
    //      point is known). The parser requires LIB/ALGO and a registered
    //      backend, so the caps lookup cannot fail on a parsed spec.
    {
        std::string line;
        if (sp.step || !sp.step_overrides.empty()) {
            std::ostringstream oss;
            if (sp.step) oss << "STEP " << *sp.step << " (fraction of transformed OPTIM width per dim)";
            else         oss << "STEP overrides only";
            for (const auto& o : sp.step_overrides) {
                oss << "; " << o.key;
                if (!o.indices.empty()) {
                    oss << "(";
                    for (std::size_t k = 0; k < o.indices.size(); ++k)
                        oss << (k ? "," : "") << o.indices[k];
                    oss << ")";
                }
                oss << " " << o.frac;
            }
            line = oss.str();
        } else if (sp.lib == "NLOPT") {
            line = "(STEP unset -- NLopt default per dim: 0.25*width, or 0.75*distance to "
                   "the nearer bound; read back at run time)";
        } else if (sp.lib == "TADAH" && sp.algo.rfind("ANNEAL", 0) == 0) {
            auto it = sp.param.find("STEP_FRAC");
            line = "PARAM STEP_FRAC " + (it != sp.param.end() ? fmt_double(it->second) : std::string("(default)"))
                 + " (fraction of transformed width per dim)";
        } else if (sp.lib == "TADAH" && sp.algo == "GRID") {
            auto it = sp.param.find("GRID_POINTS");
            line = "PARAM GRID_POINTS " + (it != sp.param.end() ? std::to_string(static_cast<int>(it->second)) : std::string("20"))
                 + " per dim";
        } else {
            line = "n/a (no initial-step concept)";
        }
        kv("Initial step", line);
        const auto& c = optimizer_caps(sp);
        if (step_use_consumes(c.step_use) || sp.step)
            kv("Step consumption", std::string(step_use_name(c.step_use)) + " -- " + c.step_note);
    }

    // ---- algorithm-specific params
    if (sp.pop)            kv("Population size", std::to_string(*sp.pop));
    if (sp.vector_storage) kv("Vector storage",  std::to_string(*sp.vector_storage));
    if (sp.seed)           kv("Random seed",     std::to_string(*sp.seed));
    /* Every PARAM, not just the numeric ones. This block used to walk
     * sp.param alone, so a token-valued key was recorded NOWHERE -- a run
     * with `PARAM INIT_MODE UNIFORM` produced a journal indistinguishable
     * from the default, and the same hole existed in the potential header.
     * Multi-token values (`INIT_MODE JITTER AUTO 0.05`) print in full. */
    if (!sp.param.empty() || !sp.param_raw.empty()) {
        std::vector<std::pair<std::string, std::string>> params;
        params.reserve(sp.param.size() + sp.param_raw.size());
        const auto rendered = [&sp](const std::string& name,
                                    const std::string& fallback) {
            const auto it = sp.param_tokens.find(name);
            if (it == sp.param_tokens.end()) return fallback;
            std::string joined;
            for (const auto& t : it->second) {
                if (!joined.empty()) joined += ' ';
                joined += t;
            }
            return joined;
        };
        for (const auto& kvp : sp.param)
            params.emplace_back(kvp.first,
                                rendered(kvp.first, fmt_double(kvp.second)));
        for (const auto& kvp : sp.param_raw)
            params.emplace_back(kvp.first, rendered(kvp.first, kvp.second));
        std::sort(params.begin(), params.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Params:\n";
        for (const auto& kvp : params) {
            f_ << "    " << std::left << std::setw(KW-4) << kvp.first
               << ": " << kvp.second << "\n";
        }
        f_.flush();
    }

    // ---- initial point
    if (!sp.optim_init.empty()) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Starting point: user-supplied"
           << " (" << sp.optim_init.size() << " seed"
           << (sp.optim_init.size()==1 ? "" : "s") << ")\n";
        f_.flush();
    } else {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Starting point: random (uniform within bounds)\n";
        f_.flush();
    }

    // ---- INIT chain (printed in the exact order the user wrote it).
    // Helper used by both OUTER (here) and PRESTAGE (below in run-policy).
    auto write_init_strategy = [&](const InitStrategySpec& isp,
                                   const char* header,
                                   const char* indent) {
        f_ << indent << header << "\n";
        if (isp.chain.empty()) {
            f_ << indent << "  (default -- CONFIG only)\n";
        } else {
            auto pretty_entry = [](const auto& e) {
                std::string s = e.method;
                for (const auto& a : e.args) s += " " + a;
                return s;
            };
            for (std::size_t i = 0; i < isp.chain.size(); ++i)
                f_ << indent << "  [" << (i+1) << "] "
                   << pretty_entry(isp.chain[i]) << "\n";
        }
        if (isp.k_total)
            f_ << indent << "  INIT_K            : " << *isp.k_total << "\n";
        if (isp.seed)
            f_ << indent << "  INIT_SEED         : " << *isp.seed << "\n";
        f_ << indent << "  WARM_ON_MISMATCH  : " << isp.warm_on_mismatch << "\n";
    };
    {
        std::lock_guard<std::mutex> lg(mtx_);
        write_init_strategy(sp.init_strategy_spec, "Init strategy:", "  ");
        f_.flush();
    }

    // ---- run-policy (STRATEGY / RESTART / NOISE block)
    {
        std::lock_guard<std::mutex> lg(mtx_);
        const auto& ms = sp.multistart;
        const auto& rs = sp.restart;
        const auto& ns = sp.noise;
        const bool has_prestage = (sp.strategy_type == StrategyType::PRESTAGE && sp.prestage);
        const bool is_multistart= (sp.strategy_type == StrategyType::MULTISTART && ms.n_starts > 1);
        const bool any_extra =
            is_multistart || has_prestage ||
            rs.active() ||
            ns.autodetect || (ns.calibration_n >= 2);
        f_ << "  Run policy:\n";
        if (!any_extra) {
            f_ << "    (TYPE PLAIN -- single-shot dispatch)\n";
        }
        if (is_multistart) {
            f_ << "    MULTISTART        : N_STARTS=" << ms.n_starts
               << "  (PARALLEL=" << ms.parallel << ")\n";
            if (ms.budget_per_start)
                f_ << "    BUDGET_PER_START  : " << *ms.budget_per_start << "\n";
        }
        if (has_prestage) {
            /* A block without MAXEVAL has no evaluation limit (MAXEVAL 0
             * is an error at its line, so it is not a budget to print). */
            const auto maxeval = [](const StopCriteria& s) {
                return s.maxeval ? std::to_string(*s.maxeval)
                                 : std::string("none");
            };
            const auto& ps = *sp.prestage;
            f_ << "    PRESTAGE EXPLORE  : " << ps.lib << ":" << ps.algo
               << "  MAXEVAL=" << maxeval(ps.stop)
               << "  TOPK=" << sp.stage_topk << "\n";
            f_ << "    PRESTAGE REFINE   : " << sp.lib << ":" << sp.algo
               << "  MAXEVAL=" << maxeval(sp.stop) << "\n";
            if (obtains_gradients(ps))
                f_ << "    EXPLORE GRAD_STRATEGY: " << grad_strategy_text(ps)
                   << "\n";
            write_init_strategy(ps.init_strategy_spec,
                                "PRESTAGE.EXPLORE init strategy:", "    ");
            if (ps.inner) {
                write_init_strategy(ps.inner->init_strategy_spec,
                                    "PRESTAGE.EXPLORE.INNER init strategy:", "      ");
            }
        }
        if (rs.active()) {
            f_ << "    RESTART AFTER_STAGNATION: " << rs.after_stagnation
               << "\n    RESTART MAX             : " << rs.max << "\n";
            if (!rs.init_strategy.chain.empty()) {
                f_ << "    RESTART init strategy:\n";
                for (std::size_t i = 0; i < rs.init_strategy.chain.size(); ++i) {
                    const auto& e = rs.init_strategy.chain[i];
                    std::string s = e.method;
                    for (const auto& a : e.args) s += " " + a;
                    f_ << "      [" << (i+1) << "] " << s << "\n";
                }
            }
        }
        if (ns.autodetect) f_ << "    NOISE.AUTO        : true\n";
        if (ns.calibration_n >= 2)
            f_ << "    NOISE.CALIBRATION : " << ns.calibration_n
               << "  (extra evals at the seed before optimisation)\n";
        f_.flush();
    }

    /* ---- search space -- own section: the per-dim bounds dump grows
     * with dim (hundreds of lines) and would drown the OPTIMISER prose. */
    section("SEARCH SPACE");
    {
        std::lock_guard<std::mutex> lg(mtx_);
        const std::size_t n = sp.low.size();
        /* Resolved labels double as the pinned-dim audit's dim names. */
        pin_labels_.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            pin_labels_[i] = (i < param_labels.size())
                             ? param_labels[i] : logidx::param_label(i);

        if (sp.space && sp.space->dim() == n && n > 0) {
            const SearchSpace& ss = *sp.space;
            /* -- policy summary ------------------------------------------ */
            if (ss.normalised())
                f_ << "  Scaling               : OPTIM_SCALE AUTO -- every OPTIM dimension "
                      "is mapped to the unit box [0,1] (optimisers iterate on u_i = "
                      "(t_i(x_i)-t_i(lo_i))/(t_i(hi_i)-t_i(lo_i)))\n"
                   << "  Optimiser units       : STEP, XTOL_ABS and PARAM fd_step are "
                      "fractions of each box width\n";
            else
                f_ << "  Scaling               : OPTIM_SCALE NONE -- optimisers iterate "
                      "in raw OPTIM units (ln for LOG dims, asinh(x/c) for ASINH dims)\n";
            std::size_t n_auto = 0, n_log = 0, n_asinh = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const auto& d = ss.at(i);
                if (d.tf.kind == ScaleKind::Log)   { ++n_log;   if (d.auto_log) ++n_auto; }
                if (d.tf.kind == ScaleKind::Asinh) ++n_asinh;
            }
            {
                std::ostringstream pol;
                if (sp.log_threshold)
                    pol << "LOG_HP " << *sp.log_threshold << " (" << n_auto
                        << " dim(s) auto-LOG)";
                else
                    pol << "LOG_HP off";
                pol << "; explicit tags: " << (n_log - n_auto) << " LOG, "
                    << n_asinh << " ASINH";
                f_ << "  Scale policy          : " << pol.str() << "\n";
            }
            {
                double wmin = std::numeric_limits<double>::infinity(), wmax = 0.0;
                std::size_t imin = 0, imax = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double w = ss.real_width(i);
                    if (w < wmin) { wmin = w; imin = i; }
                    if (w > wmax) { wmax = w; imax = i; }
                }
                std::ostringstream wr;
                wr << fmt_double(wmin > 0 ? wmax / wmin : 0.0, 2)
                   << " (min " << pin_labels_[imin] << " " << fmt_double(wmin, 3)
                   << ", max " << pin_labels_[imax] << " " << fmt_double(wmax, 3) << ")";
                if (ss.normalised()) wr << " -> 1.0 for every dim after scaling";
                f_ << "  Real width ratio      : " << wr.str() << "\n";
            }
            f_ << "  Space version         : " << ss.version() << "\n";

            /* -- per-dim table --------------------------------------------- */
            const std::vector<double>* start =
                sp.optim_init.empty() ? nullptr : &sp.optim_init.front();
            auto pad = [](std::string s, std::size_t w) {
                if (s.size() < w) s.resize(w, ' ');
                return s;
            };
            /* origin column: 11 wide, wider only when a longer origin name
             * ("AUTO FRAC>=0") occurs */
            std::size_t ow = 11;
            for (std::size_t i = 0; i < n; ++i)
                ow = std::max(ow, std::strlen(box_origin_name(ss.at(i).origin)) + 1);
            f_ << "   " << pad("#", 4) << pad("parameter", 24) << pad("scale", 13)
               << pad("origin", ow) << pad("real bounds", 30)
               << pad("internal bounds", 26) << pad("start (real)", 14)
               << "u0\n";
            for (std::size_t i = 0; i < n; ++i) {
                const auto& d = ss.at(i);
                std::string row = "   " + pad(std::to_string(i + 1), 4)
                                + pad(pin_labels_[i], 24)
                                + pad(ss.describe_dim(i), 13)
                                + pad(box_origin_name(d.origin), ow)
                                + pad("[" + fmt_double(d.lo, 3) + ", " + fmt_double(d.hi, 3) + "]", 30)
                                + pad("[" + fmt_double(sp.low[i], 3) + ", " + fmt_double(sp.high[i], 3) + "]", 26);
                if (start && i < start->size()) {
                    const double x0 = (*start)[i];
                    row += pad(fmt_double(x0, 4), 14);
                    row += fmt_double(ss.to_internal(i, x0), 4);
                } else {
                    row += pad("-", 14) + "-";
                }
                f_ << row << "\n";
            }
        } else {
            f_ << "  (bounds in optimiser-internal space)\n";
            for (std::size_t i = 0; i < n; ++i) {
                std::string row = "    " + pin_labels_[i];
                if (row.size() < 28) row.resize(28, ' ');
                row += ": [" + fmt_double(sp.low[i], 3) + ",  " + fmt_double(sp.high[i], 3) + "]";
                if (i < sp.log_scaled.size() && sp.log_scaled[i])
                    row += "  (log-scale -- bounds are ln(value))";
                f_ << row << "\n";
            }
        }
        f_.flush();
    }
}

void HPOLog::write_potential_section(const core::Context&  ctx,
                                     std::size_t           n_train,
                                     std::size_t           n_val,
                                     const std::string&    elements_str,
                                     const RawEmbeddingF0* embf0_seed)
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

    /* DM_REAM embedding reference. stored_embf0: never Context::get or
     * contains on EMBF0 here (that would insert the default). */
    if (models::context_has_ream(ctx)) {
        const auto stored = models::stored_embf0(ctx);
        const bool zero =
            stored.value_or(models::EmbF0Mode::Zero) == models::EmbF0Mode::Zero;
        kv("Embedding F(0)",
           zero ? std::string("EMBF0 ZERO") +
                      (stored ? "" : " (key absent -> default)") +
                      ": DM_REAM embedding zero-referenced, isolated-atom "
                      "embedding energy 0 by construction"
                : std::string("EMBF0 FREE: raw embedding (tadah <= 1.5.0), "
                              "F(0) = sum w g(0)"));
        if (embf0_seed) {
            std::string v;
            switch (embf0_seed->kind) {
                case RawEmbeddingF0::Kind::Value:
                    v = fmt_double(embf0_seed->value, 6) + " eV = sum w g(0), " +
                        embf0_seed->source + " of the -c potential (" +
                        (zero ? "removed per atom by EMBF0 ZERO)"
                              : "isolated-atom embedding energy)");
                    break;
                case RawEmbeddingF0::Kind::PerEvaluation:
                    v = "determined per evaluation (the weights are fitted on "
                        "every evaluation; start/best under RESULT)";
                    break;
                case RawEmbeddingF0::Kind::NotApplicable:
                    v = "n/a (" + embf0_seed->reason + ")";
                    break;
            }
            kv("Raw F(0) of -c pot", v);
        }
    }

    kv("Training structures",   std::to_string(n_train));
    kv("Validation structures", std::to_string(n_val));
    kv("Elements",              elements_str);

    if (!init2b && !initmb) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "\n  !! Neither INIT2B nor INITMB is true -- the potential has no terms.\n"
              "     Check your configuration file.\n";
        f_.flush();
    }
}

void HPOLog::write_fit_switches(const FitSwitchReport& r)
{
    auto row = [](bool on, const std::string& source) {
        return std::string(on ? "yes" : "no") + "  (" + source + ")";
    };
    kv("Fit forces (FORCE)",    row(r.force, r.force_source));
    kv("Fit stresses (STRESS)", row(r.stress, r.stress_source));
    std::string v;
    if (!r.has_validation) {
        v = "no validation set";
    } else {
        v = "energies";
        if (r.val_forces)   v += ", forces (PC_FRMSE)";
        if (r.val_stresses) v += ", stresses (PC_SRMSE)";
    }
    kv("Validation predicts", v);
}

void HPOLog::write_loss_section(const HPOTargets& trg)
{
    section("LOSS FUNCTION");

    auto metric_line = [&](const std::string& name, bool used,
                           double target, double weight) {
        if (!used) return;
        std::ostringstream s;
        s << "weight=" << std::fixed << std::setprecision(2) << weight
          << "  target=" << std::scientific << std::setprecision(3) << target
          << "  (per-dataset + avg)";
        kv(name, s.str());
    };

    metric_line("EMAE   (meV/atom)",  trg.use_emae,   trg.emae, trg.w_emae);
    metric_line("ErRMSE (per mille)",  trg.use_errmse, trg.errmse, trg.w_errmse);
    metric_line("ERSQ",               trg.use_ersq,   trg.ersq,            trg.w_ersq);
    metric_line("SRMSE  (eV/A^3)",    trg.use_srmse,  trg.srmse,           trg.w_srmse);

    auto dataset_metric = [&](const std::string& name, const ValidationMetricTargets& targets) {
        for (const auto& description : targets.descriptions()) kv(name, description);
    };
    dataset_metric("ERMSE (meV/at)", trg.ermse_targets);
    dataset_metric("FRMSE (eV/A)", trg.frmse_targets);

    for (const auto& lsim : trg.LS) {
        for (std::size_t i = 0; i < lsim.n_varloss.size(); ++i) {
            std::ostringstream s;
            s << "weight=" << std::fixed << std::setprecision(2) << lsim.w_varloss[i]
              << "  target=" << std::scientific << std::setprecision(3) << lsim.t_varloss[i];
            kv(lsim.n_varloss[i], s.str());
        }
    }

    /* PC_OBSERVABLE typed-observable components -- surface them in the
     * header so users see their cohesive_energy / etc. contribution
     * alongside ERMSE and PC_LAMMPS varloss entries, named <kind>[k] for
     * the k-th PC_OBSERVABLE line (1-based; its columns are L(<tag>)[k]
     * and <tag>[k]). Without this loop the section silently omits
     * cohesive_energy and the user wonders whether their PC_OBSERVABLE
     * line took effect. */
    /* The column titles as loss.tadah and outvar.tadah write them: past
     * k = 99 a narrow DIGITS shortens the tag and keeps the index. */
    const std::size_t title_budget = column_title_budget(trg.outputDigits);
    for (std::size_t k = 0; k < trg.observables.size(); ++k) {
        const auto& obs = trg.observables[k];
        if (!obs) continue;
        std::ostringstream s;
        s << "weight=" << std::fixed << std::setprecision(2) << obs->weight()
          << "  target=" << std::scientific << std::setprecision(3) << obs->target()
          << "  columns "
          << header_detail::fit_keep_index(observable_loss_label(*obs, k + 1),
                                           title_budget)
          << ", "
          << header_detail::fit_keep_index(observable_value_label(*obs, k + 1),
                                           title_budget);
        kv(observable_journal_label(*obs, k + 1), s.str());
    }

    kv("Aggregate",   "mean over active components");
    kv("Fail score",  fmt_double(trg.FAIL_SCORE)
                      + "  (returned when an evaluation produces NaN/Inf)");
    kv("Output every",  std::to_string(trg.outputEvery) + " eval(s)");
    kv("Output digits", std::to_string(trg.outputDigits));
}

/* -------------------------------------------------------------------------
 * Load-transforms section
 * ------------------------------------------------------------------------- */

void HPOLog::write_load_transforms_section(const mlip::LoadTransformReport& r,
                                            std::size_t                       n_train_loaded,
                                            const mlip::LoadTransformReport& vr,
                                            std::size_t                       n_val_loaded)
{
    const bool val_filter_active =
        vr.n_dropped_energy != 0 || vr.n_dropped_force != 0;
    if (!r.applied && !val_filter_active) return;
    section("LOAD TRANSFORMS");

    if (r.lscale != 1.0) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(6) << r.lscale;
        kv("Length rescale (LSCALE)", s.str());
    }

    if (!r.eshift.empty()) {
        const char* src = "user";
        switch (r.eshift_source) {
            case mlip::LoadTransformReport::EshiftSource::User:   src = "user";   break;
            case mlip::LoadTransformReport::EshiftSource::Atom:   src = "atom";   break;
            case mlip::LoadTransformReport::EshiftSource::DBAtom: src = "dbatom"; break;
            case mlip::LoadTransformReport::EshiftSource::None:   src = "none";   break;
        }
        std::ostringstream s;
        s << "(source=" << src << ")";
        for (const auto& kv_pair : r.eshift) {
            const auto& el = core::PeriodicTable::find_by_Z(kv_pair.first);
            s << "  " << el.symbol << "=" << std::fixed
              << std::setprecision(4) << kv_pair.second;
        }
        kv("Energy shift (ESHIFT)", s.str());
    }

    auto fmt_filter = [](const mlip::LoadTransformReport& rep,
                         std::size_t n_loaded) {
        std::ostringstream s;
        s << "dropped " << rep.n_dropped_energy << " by E, "
          << rep.n_dropped_force << " by F  ("
          << n_loaded << " loaded, "
          << (n_loaded - rep.n_dropped_energy - rep.n_dropped_force)
          << " remain)";
        return s.str();
    };

    if (r.n_dropped_energy || r.n_dropped_force)
        kv("Outlier filter (training)", fmt_filter(r, n_train_loaded));

    /* Validation outlier filter: only emitted when the user actually
     * declared EFILTER / FFILTER on the --validation file (otherwise both
     * counters are zero by construction -- training-side filters are not
     * forwarded to validation). */
    if (val_filter_active)
        kv("Outlier filter (validation)", fmt_filter(vr, n_val_loaded));

    if (!r.wdbfile_applied.empty()) {
        std::ostringstream s;
        for (std::size_t i = 0; i < r.wdbfile_applied.size(); ++i) {
            if (i) s << " ";
            s << std::scientific << std::setprecision(3) << r.wdbfile_applied[i];
        }
        kv("Per-dataset weights", s.str());
    }

    if (r.eweight_temp > 0.0) {
        std::ostringstream s;
        s << r.eweight_temp << " K";
        kv("Boltzmann eweight (T)", s.str());
    }

    if (r.zero_com_force)
        kv("Zero COM force", "applied");

    if (!r.eq_label.empty()) {
        std::ostringstream s;
        s << "label='" << r.eq_label << "'  E/N="
          << std::fixed << std::setprecision(4) << r.eq_energy_per_atom
          << " eV  V/N=" << r.eq_volume_per_atom << " A^3";
        kv("Reference state (eq vol)", s.str());
    }
}

/* -------------------------------------------------------------------------
 * Noise calibration section
 * ------------------------------------------------------------------------- */

void HPOLog::write_noise_calibration_section(
    const std::vector<double>& losses,
    double                     suggested_ftol)
{
    section("NOISE CALIBRATION");

    /* Compute summary stats outside any lock -- purely arithmetic. */
    double mn = 0.0, mx = 0.0, mean = 0.0, stddev = 0.0;
    double abs_spread = 0.0, rel_spread = 0.0;
    if (!losses.empty()) {
        mn = losses.front(); mx = losses.front();
        double sum = 0.0;
        for (double v : losses) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        const double n = static_cast<double>(losses.size());
        mean = sum / n;
        if (losses.size() >= 2) {
            double sq = 0.0;
            for (double v : losses) {
                const double d = v - mean;
                sq += d * d;
            }
            stddev = std::sqrt(sq / (n - 1.0));
        }
        abs_spread = mx - mn;
        rel_spread = (mean != 0.0) ? abs_spread / std::fabs(mean) : 0.0;
    }

    if (losses.size() < 2) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  (calibration not run -- need >= 2 evals)\n";
        f_.flush();
        return;
    }

    /* Single critical section for all of this section's body -- holding
     * the mutex once and writing directly to f_ avoids the kv() /
     * lock_guard re-entrance trap (mtx_ is std::mutex, not recursive). */
    auto fmt_sci = [](double v) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << v;
        return oss.str();
    };
    constexpr int KW = 22;
    std::lock_guard<std::mutex> lg(mtx_);
    f_ << "  " << std::left << std::setw(KW-2) << "Repeats"
       << ": " << losses.size() << "\n";
    f_ << "  Per-run total loss:\n";
    for (std::size_t i = 0; i < losses.size(); ++i)
        f_ << "    [" << (i+1) << "] " << fmt_double(losses[i]) << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Mean"
       << ": " << fmt_double(mean)   << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Stddev"
       << ": " << fmt_double(stddev) << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Min"
       << ": " << fmt_double(mn)     << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Max"
       << ": " << fmt_double(mx)     << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Abs spread"
       << ": " << fmt_double(abs_spread) << "\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Rel spread"
       << ": " << fmt_sci(rel_spread)
       << "  ((max-min)/|mean|)\n";
    f_ << "  " << std::left << std::setw(KW-2) << "Suggested FTOL_REL"
       << ": " << fmt_sci(suggested_ftol)
       << "  (10x rel_spread, floored at 1e-6)\n";
    if (rel_spread < 1e-12) {
        f_ << "  Note: objective looks effectively deterministic at the seed point;\n"
              "        any FTOL_REL >= 1e-6 will not be the bottleneck.\n";
    } else if (rel_spread > 1e-2) {
        f_ << "  Note: large noise floor (>1%). Local-search FTOL_REL tighter\n"
              "        than the suggested value will not be reachable; consider\n"
              "        averaging multiple LAMMPS runs or tightening MD sampling.\n";
    }
    f_.flush();
}

/* -------------------------------------------------------------------------
 * Progress section
 * ------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------
 * Per-eval recording
 * ------------------------------------------------------------------------- */

void HPOLog::record_fail_band(int band)
{
    if (band >= 1 && band <= 4)
        n_fail_band_[band].fetch_add(1, std::memory_order_relaxed);
}

void HPOLog::record_gate_reject(GateKind kind)
{
    n_gate_[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

void HPOLog::record_bump_gate_reject()
{
    record_gate_reject(GateKind::Bump);
}

std::size_t HPOLog::next_gate_note(GateKind kind)
{
    return n_gate_note_[static_cast<std::size_t>(kind)].fetch_add(
               1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_form_projection()
{
    return n_form_proj_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_snap()
{
    return n_snap_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_out_of_box_eval()
{
    return n_oob_eval_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_out_of_box_gradient()
{
    return n_oob_grad_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_hybrid_centre()
{
    return n_hybrid_centre_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_nn_rebuild()
{
    return n_nn_rebuild_.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::size_t HPOLog::record_one_sided(const std::string& label)
{
    std::lock_guard<std::mutex> lg(mtx_);
    for (auto& e : one_sided_)
        if (e.first == label) return ++e.second;
    one_sided_.emplace_back(label, 1);
    return 1;
}

std::size_t HPOLog::one_sided(const std::string& label) const
{
    std::lock_guard<std::mutex> lg(mtx_);
    for (const auto& e : one_sided_)
        if (e.first == label) return e.second;
    return 0;
}

void HPOLog::reset_fd_counters_locked()
{
    n_oob_eval_.store(0, std::memory_order_relaxed);
    n_snap_.store(0, std::memory_order_relaxed);
    n_oob_grad_.store(0, std::memory_order_relaxed);
    n_hybrid_centre_.store(0, std::memory_order_relaxed);
    n_nn_rebuild_.store(0, std::memory_order_relaxed);
    one_sided_.clear();
}

void HPOLog::record_io_retry_durable(std::size_t n)
{
    n_io_retry_durable_.fetch_add(n, std::memory_order_relaxed);
}

void HPOLog::record_io_retry_writer(std::size_t n)
{
    n_io_retry_writer_.fetch_add(n, std::memory_order_relaxed);
}

void HPOLog::record_rows_lost(std::size_t n)
{
    n_rows_lost_.fetch_add(n, std::memory_order_relaxed);
}

void HPOLog::record_best_persistence_failure()
{
    best_persist_failed_.store(true, std::memory_order_relaxed);
}

void HPOLog::record_archive_lost()
{
    n_archives_lost_.fetch_add(1, std::memory_order_relaxed);
}

void HPOLog::record_unsaved_best(std::size_t eval, double loss,
                                 const std::string& what, bool may_be_on_disk)
{
    std::lock_guard<std::mutex> lg(mtx_);
    /* Several chains can fail to persist before the run stops: report the
     * lowest loss, keep the first error (the run's status), and list every
     * candidate that may have reached the file. */
    if (!unsaved_best_) {
        unsaved_best_ = UnsavedBest{eval, loss, what, {}};
    } else if (loss < unsaved_best_->loss) {
        unsaved_best_->eval = eval;
        unsaved_best_->loss = loss;
    }
    if (may_be_on_disk) unsaved_best_->maybe_on_disk.emplace_back(eval, loss);
}

void HPOLog::record_start_batch(std::size_t ran, std::size_t failed)
{
    std::lock_guard<std::mutex> lg(mtx_);
    start_tally_.ran    += ran;
    start_tally_.failed += failed;
    ++start_tally_.batches;
}

HPOLog::StartTally HPOLog::start_tally() const
{
    std::lock_guard<std::mutex> lg(mtx_);
    return start_tally_;
}

std::optional<HPOLog::UnsavedBest> HPOLog::unsaved_best() const
{
    std::lock_guard<std::mutex> lg(mtx_);
    return unsaved_best_;
}

void HPOLog::record_eval(std::size_t eval_num, double wall_ms, EvalLoss loss)
{
    n_eval_.fetch_add(1, std::memory_order_relaxed);
    if (loss == EvalLoss::NanClamped) n_nan_.fetch_add(1, std::memory_order_relaxed);
    if (loss != EvalLoss::Usable) n_unusable_.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lg(mtx_);
        sum_ms_ += wall_ms;
        if (wall_ms < min_ms_) { min_ms_ = wall_ms; min_eval_ = eval_num; }
        if (wall_ms > max_ms_) { max_ms_ = wall_ms; max_eval_ = eval_num; }
        if (last_milestone_evals_ == 0) {
            /* anchor the first wall-throughput window at the first eval */
            last_milestone_tp_    = std::chrono::steady_clock::now();
            last_milestone_evals_ = 1;
        }
    }

    // Periodic milestone summary
    if (eval_num >= next_milestone_) {
        std::lock_guard<std::mutex> lg(mtx_);
        if (eval_num >= next_milestone_) { // re-check under lock
            std::size_t done    = n_eval_.load(std::memory_order_relaxed);
            std::size_t nan_cnt = n_nan_.load(std::memory_order_relaxed);
            double avg = (done > 0) ? sum_ms_ / done : 0.0;
            double nan_pct = (done > 0) ? 100.0 * nan_cnt / done : 0.0;

            /* WALL throughput over the window since the previous
             * milestone: what the run delivers per unit time under the
             * CURRENT parallelisation. `avg` above is the mean PER-EVAL
             * LATENCY on its worker thread -- when K evaluations are fully concurrent,
             * it can approach K x the wall cost and must
             * not be read as throughput. */
            const auto now_tp = std::chrono::steady_clock::now();
            double wall_ms_per_eval = 0.0;
            double win_ms           = 0.0;   /* milestone-window elapsed */
            if (last_milestone_evals_ > 0 && done > last_milestone_evals_) {
                win_ms =
                    std::chrono::duration<double, std::milli>(
                        now_tp - last_milestone_tp_).count();
                wall_ms_per_eval =
                    win_ms / static_cast<double>(done - last_milestone_evals_);
            }
            last_milestone_tp_    = now_tp;
            last_milestone_evals_ = done;

            const std::size_t evals_since_best =
                (cur_best_eval_ > 0 && eval_num >= cur_best_eval_)
                    ? (eval_num - cur_best_eval_) : 0;

            std::ostringstream msg;
            msg << "--- " << std::setw(6) << eval_num << " evals done";
            if (wall_ms_per_eval > 0.0) {
                std::ostringstream rate;
                rate << std::fixed << std::setprecision(1)
                     << 1000.0 / wall_ms_per_eval;
                msg << "   wall=" << fmt_millis(wall_ms_per_eval)
                    << "/eval (" << rate.str() << " ev/s)";
            }
            msg << "   latency=" << fmt_millis(avg) << "/eval"
                << "   best=" << fmt_double(cur_best_loss_)
                << " (eval " << cur_best_eval_ << ", "
                << evals_since_best << " evals ago"
                << best_origin_str() << ")"
                << "   NaN rate=" << std::fixed << std::setprecision(1)
                << nan_pct << "%";
            /* Graded-failure census. NaN rate above counts only non-finite
             * losses; a killed worker or a failed script yields a large but
             * FINITE failscore, so it is invisible there. Printed only when
             * something actually failed, so healthy runs keep the line
             * short. */
            {
                const std::size_t fa = n_fail_band_[1].load(std::memory_order_relaxed);
                const std::size_t fb = n_fail_band_[2].load(std::memory_order_relaxed);
                const std::size_t fc = n_fail_band_[3].load(std::memory_order_relaxed);
                const std::size_t fd = n_fail_band_[4].load(std::memory_order_relaxed);
                const std::size_t ft = fa + fb + fc + fd;
                if (ft > 0) {
                    const double fpct = eval_num ? 100.0 * static_cast<double>(ft) /
                                                       static_cast<double>(eval_num)
                                                 : 0.0;
                    msg << "   failed=" << ft << " (" << std::fixed
                        << std::setprecision(1) << fpct << "%; A=" << fa
                        << " B=" << fb << " C=" << fc << " D=" << fd << ")";
                    /* per-gate subsets of D (pre-training rejects) */
                    static const char* const gate_tag[kGateKinds] = {
                        " bumpgate=", " rhogate=", " rhobargate="};
                    for (std::size_t g = 0; g < kGateKinds; ++g) {
                        const std::size_t n =
                            n_gate_[g].load(std::memory_order_relaxed);
                        if (n > 0) msg << gate_tag[g] << n;
                    }
                }
            }
            /* The milestone SUMMARY is one logical block: the head line
             * carries the (single) timestamp, the pc_lammps and cadence
             * sub-lines below blank the timestamp column. */
            f_ << "[" << now_hms() << "]  SUMMARY  "
               << msg.str() << "\n";
            /* pc_lammps shared-pool utilisation over this milestone
             * window: Σ script busy / (pool width x window elapsed).
             * Honest under concurrent replicas (per-eval width x wall
             * double-counted overlapped time -- the old ">100% eff"
             * artefact). Printed only when the pool width is known;
             * per-script stats print regardless. NOTE: this is
             * utilisation of the WHOLE window, so time spent in
             * train/predict/sims_loss also reads as pool idle -- a low
             * value is not, by itself, evidence of script imbalance. */
            if (fo_scount_win_ > 0) {
                const double avg =
                    fo_busy_win_ / static_cast<double>(fo_scount_win_);
                std::ostringstream fo;
                fo << "    pc_lammps:";
                if (fanout_pool_width_ > 0 && win_ms > 0.0) {
                    const double util = fo_busy_win_ /
                        (static_cast<double>(fanout_pool_width_) * win_ms);
                    fo << " pool util " << std::fixed << std::setprecision(0)
                       << 100.0 * util << "% of width "
                       << fanout_pool_width_ << ";";
                }
                fo << " per-script avg " << fmt_millis(avg)
                   << ", fastest " << fmt_millis(fo_smin_win_)
                   << ", slowest " << fmt_millis(fo_smax_win_);
                if (!fo_slowest_win_.empty()) fo << " [" << fo_slowest_win_ << "]";
                f_ << std::string(12, ' ') << "SUMMARY  " << fo.str() << "\n";
                /* reset the window accumulators for the next milestone */
                fo_busy_win_ = 0.0; fo_scount_win_ = 0;
                fo_smin_win_ = std::numeric_limits<double>::max();
                fo_smax_win_ = 0.0; fo_slowest_win_.clear();
            }
            /* Optimiser work-unit cadence: translate the registered
             * rung/epoch/T-step eval cost into wall time at the CURRENT
             * window rate. Re-derived every milestone (never frozen) so
             * it tracks throughput drift; the unit eval count itself is
             * exact arithmetic supplied by the optimiser. */
            if (cadence_evals_ > 0 && wall_ms_per_eval > 0.0) {
                std::ostringstream cs;
                cs << "    " << cadence_unit_ << " = " << cadence_evals_
                   << " evals ~ "
                   << format_duration(
                          static_cast<double>(cadence_evals_) *
                          wall_ms_per_eval / 1000.0)
                   << " at current rate";
                if (cadence_maxeval_ > done) {
                    const std::uint64_t rem  = cadence_maxeval_ - done;
                    const std::uint64_t left =
                        (rem + cadence_evals_ - 1) / cadence_evals_;
                    cs << "; ~" << left << " " << cadence_unit_ << "(s) / ~"
                       << format_duration(static_cast<double>(rem) *
                                          wall_ms_per_eval / 1000.0)
                       << " to MAXEVAL";
                }
                f_ << std::string(12, ' ') << "SUMMARY  " << cs.str() << "\n";
            }
            /* Bound-occupancy audit (every backend): where the latest best
             * sits relative to its OPTIM box. */
            write_bound_occupancy_locked("SUMMARY", /*final_table=*/false);

            /* Near-bound floor-step audit: a CORANA driver marks a dim
             * whose chain is within one floor-step of a bound while V is
             * already at that floor. Inward proposals remain possible, so
             * this is a heuristic signal, not proof that the optimum lies
             * outside the box. One WARNING block per milestone aggregates
             * marks by (dim, side) across chains. */
            if (pin_chains_ > 0 && !pin_state_.empty()) {
                struct Row { std::size_t dim; int side; int n;
                             double gap; double tol; std::uint64_t since; };
                std::map<std::pair<std::size_t, int>, Row> agg;
                for (const auto& [k, v] : pin_state_) {
                    auto& r = agg.try_emplace(
                        std::pair<std::size_t, int>{std::get<1>(k),
                                                    std::get<2>(k)},
                        Row{std::get<1>(k), std::get<2>(k), 0, 0.0, 0.0,
                            std::numeric_limits<std::uint64_t>::max()})
                        .first->second;
                    ++r.n;
                    if (v.gap   > r.gap)   r.gap   = v.gap;
                    if (v.tol   > r.tol)   r.tol   = v.tol;
                    if (v.since < r.since) r.since = v.since;
                }
                std::vector<Row> rows;
                rows.reserve(agg.size());
                for (const auto& [k, r] : agg) rows.push_back(r);
                std::sort(rows.begin(), rows.end(),
                          [](const Row& a, const Row& b) {
                              if (a.n   != b.n)   return a.n   > b.n;
                              if (a.dim != b.dim) return a.dim < b.dim;
                              return a.side < b.side;
                          });
                auto label_of = [&](const Row& r) {
                    std::string l = r.dim < pin_labels_.size()
                        ? pin_labels_[r.dim]
                        : logidx::param_label(r.dim);
                    if (r.dim < pin_log_.size() && pin_log_[r.dim] == 1)
                        l += " (ln)";
                    else if (r.dim < pin_log_.size() && pin_log_[r.dim] == 2)
                        l += " (asinh)";
                    return l;
                };
                std::size_t lw = 24;
                for (const auto& r : rows)
                    lw = std::max(lw, label_of(r).size());
                auto pad = [](std::string s, std::size_t w) {
                    if (s.size() < w) s.resize(w, ' ');
                    return s;
                };
                auto sci1 = [](double v) {
                    if (v == 0.0) return std::string("0");
                    std::ostringstream o;
                    o << std::scientific << std::setprecision(1) << v;
                    return o.str();
                };
                const bool multi = pin_chains_ > 1;
                std::ostringstream t;
                /* Variant-neutral wording: this audit now runs for CLASSIC
                 * as well as CORANA. Under CORANA a row additionally means
                 * the step has bottomed out; under CLASSIC only the
                 * near-bound occupancy is asserted, because V_FLOOR_FRAC is
                 * a CORANA knob and has no meaning for the calibrated
                 * CLASSIC step. Claiming "step at the V floor" for both
                 * would be false half the time. */
                t << rows.size() << " OPTIM dim(s) near a bound -- x sits "
                     "within one proposal step (tol) of an OPTIM limit "
                     "(under CORANA the step has also reached its floor, "
                     "so no further shrinkage is available):\n"
                  << "  " << pad("dim", lw + 2) << pad("side", 6)
                  << pad("bound", 13) << pad("gap", 11) << pad("tol", 11);
                if (multi) t << pad("slots", 8);
                t << "observed for";
                for (const auto& r : rows) {
                    const bool   ok = r.dim < pin_low_.size();
                    const double b  = ok ? (r.side ? pin_high_[r.dim]
                                                   : pin_low_[r.dim]) : 0.0;
                    const std::uint64_t u = pin_unit_ >= r.since
                                            ? pin_unit_ - r.since + 1 : 1;
                    t << "\n  " << pad(label_of(r), lw + 2)
                      << pad(r.side ? "HI" : "LO", 6)
                      << pad(fmt_double(b, 3), 13)
                      << pad(sci1(r.gap), 11)
                      << pad(sci1(r.tol), 11);
                    if (multi)
                        t << pad(std::to_string(r.n) + "/"
                                 + std::to_string(pin_chains_), 8);
                    t << u << " " << pin_unit_name_ << (u == 1 ? "" : "s");
                }
                event_locked("WARNING", t.str());
                event_locked("HINT",
                    "persistent near-bound occupancy may indicate "
                    "OPTIM bound pressure -- inspect the trace; widen or "
                    "re-centre only when the admissible domain permits it.");
            }
            /* Stagnation hint: if we've spent more than half the recent
             * milestone window with no new best, point the user at the
             * common causes (inner stop tolerance too tight for noise,
             * the best point's basin of attraction already exhausted,
             * penalty cliff sampling). Single line, only once per
             * milestone. */
            if (cur_best_eval_ > 0 && evals_since_best >= 100 &&
                evals_since_best * 2 >= eval_num) {
                f_ << "[" << now_hms() << "]  HINT     "
                   << "no new best for " << evals_since_best
                   << " evals (>50% of run). Possible causes: "
                   << "(a) inner stop tolerance too tight for objective noise; "
                   << "(b) the best basin is genuinely exhausted; "
                   << "(c) penalty cliff -- most samples land on "
                   << "infeasible side. Inspect loss.tadah spread.\n";
            }
            f_.flush();

            // schedule next milestone: 100, 200, 500, 1000, 2000, 5000, ...
            if      (next_milestone_ <  500)  next_milestone_ += 100;
            else if (next_milestone_ < 2000)  next_milestone_ += 500;
            else if (next_milestone_ < 10000) next_milestone_ += 1000;
            else                              next_milestone_ += 5000;
        }
    }
}

void HPOLog::record_bound_occupancy(const std::vector<BoundOccupancy::Mark>& marks,
                                    std::uint64_t eval, double edge)
{
    std::lock_guard<std::mutex> lg(mtx_);
    occ_marks_ = marks;
    occ_eval_  = eval;
    occ_edge_  = edge;
    occ_seen_  = true;
}

/* Caller holds mtx_. Milestone form: multi-line INFO block + one HINT.
 * Final form (DIAGNOSTICS): plain table. Nothing is printed when the
 * latest best sits away from every bound (final form says so). */
void HPOLog::write_bound_occupancy_locked(const char* /*indent_tag*/, bool final_table)
{
    if (!occ_seen_) return;
    if (occ_marks_.empty()) {
        if (final_table)
            f_ << "  [ok] Final best sits away from every OPTIM bound (edge "
               << std::fixed << std::setprecision(0) << occ_edge_ * 100.0 << "% of width).\n";
        return;
    }
    std::ostringstream tbl;
    tbl << (final_table ? "  Bound occupancy of the final best (eval " : "best point (eval ")
        << occ_eval_ << ") sits within " << std::fixed << std::setprecision(0)
        << occ_edge_ * 100.0 << "% of an OPTIM limit on " << occ_marks_.size()
        << " dim(s) (gap = fraction of width):\n";
    tbl << (final_table ? "    " : "  ")
        << std::left << std::setw(24) << "dim" << std::setw(6) << "side"
        << std::setw(14) << "value" << std::setw(14) << "bound"
        << std::setw(11) << "gap/width" << "since eval\n";
    bool any_hard = false, any_soft = false;
    for (const auto& m : occ_marks_) {
        const std::string lbl = m.dim < pin_labels_.size() ? pin_labels_[m.dim]
                                                           : logidx::param_label(m.dim);
        if (m.hard) any_hard = true;
        else        any_soft = true;
        tbl << (final_table ? "    " : "  ")
            << std::left << std::setw(24) << lbl
            << std::setw(6) << (m.hi ? "HI" : m.hard ? "LO*" : "LO")
            << std::setw(14) << fmt_double(m.real_value, 4)
            << std::setw(14) << fmt_double(m.real_bound, 4)
            << std::setw(11) << fmt_double(m.gap_u, 2)
            << m.since_eval << "\n";
    }
    /* A hard lower bound (BoundOccupancy::set_hard_lo) cannot be widened:
     * a best there is an admissible optimum, so it gets no remedy. */
    if (any_hard)
        tbl << (final_table ? "    " : "  ")
            << "(* a hard lower bound: RHOFORM NONNEG/MONOTONE require AMPGRIDMB >= 0,"
               " so an amplitude at 0 is an admissible optimum, not bound pressure)\n";
    std::string text = tbl.str();
    if (!text.empty() && text.back() == '\n') text.pop_back();
    if (final_table) {
        f_ << text << "\n";
        if (any_soft)
            f_ << "    (advisory: a best that rides a bound suggests bound pressure -- widen that side\n"
                  "     and rerun with INIT METHOD WARM; the run never changes OPTIM bounds by itself)\n";
    } else {
        event_locked("INFO", text);
        if (any_soft)
            event_locked("HINT", "a best that keeps riding a bound suggests bound pressure -- widen that "
                                 "side (where the physics permits) and rerun with INIT METHOD WARM");
    }
    f_.flush();
}

void HPOLog::arm_pin_audit(const OptimizerSpec& sp, double v_floor_frac,
                           int n_chains, const std::string& unit_name)
{
    std::lock_guard<std::mutex> lg(mtx_);
    pin_low_  = sp.low;
    pin_high_ = sp.high;
    /* 1 = ln-scaled, 2 = asinh-scaled (tag suffix in the milestone table). */
    pin_log_.assign(sp.low.size(), 0);
    for (std::size_t i = 0; i < pin_log_.size(); ++i) {
        if (sp.space && i < sp.space->dim()) {
            const auto k = sp.space->at(i).tf.kind;
            pin_log_[i] = (k == ScaleKind::Log) ? 1 : (k == ScaleKind::Asinh ? 2 : 0);
        } else if (i < sp.log_scaled.size() && sp.log_scaled[i]) {
            pin_log_[i] = 1;
        }
    }
    pin_floor_frac_ = v_floor_frac;
    pin_chains_     = n_chains > 0 ? n_chains : 1;
    pin_unit_name_  = unit_name;
    pin_unit_       = 0;
    pin_state_.clear();
}

void HPOLog::record_pin_state(int chain, std::uint64_t unit,
                              const std::vector<PinnedDim>& pins)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (pin_chains_ == 0) return;   /* audit not armed */
    if (unit > pin_unit_) pin_unit_ = unit;
    /* Replace this chain's set: a mark still held on the same (dim, side)
     * keeps its original since-unit; everything else for the chain is
     * unmarked. */
    const auto lo = pin_state_.lower_bound({chain, 0, 0});
    const auto hi = pin_state_.lower_bound({chain + 1, 0, 0});
    std::map<std::pair<std::size_t, int>, std::uint64_t> old_since;
    for (auto it = lo; it != hi; ++it)
        old_since[{std::get<1>(it->first), std::get<2>(it->first)}] =
            it->second.since;
    pin_state_.erase(lo, hi);
    for (const auto& p : pins) {
        const int  side = p.hi ? 1 : 0;
        const auto f    = old_since.find({p.dim, side});
        pin_state_[{chain, p.dim, side}] =
            PinInfo{f != old_since.end() ? f->second : unit, p.gap, p.tol};
    }
}

void HPOLog::set_cadence(std::size_t evals_per_unit, const std::string& unit,
                         std::uint64_t maxeval_total)
{
    std::lock_guard<std::mutex> lg(mtx_);
    cadence_evals_   = evals_per_unit;
    cadence_unit_    = unit;
    cadence_maxeval_ = maxeval_total;
}

void HPOLog::set_fanout_width(int width)
{
    std::lock_guard<std::mutex> lg(mtx_);
    fanout_pool_width_ = width > 0 ? width : 0;
}

void HPOLog::record_fanout(double busy_ms,
                           std::size_t n_scripts, double min_ms, double max_ms,
                           const std::string& slowest)
{
    if (n_scripts == 0) return;
    std::lock_guard<std::mutex> lg(mtx_);
    fo_busy_win_      += busy_ms;
    fo_scount_win_    += n_scripts;
    if (min_ms < fo_smin_win_) fo_smin_win_ = min_ms;
    if (max_ms > fo_smax_win_) { fo_smax_win_ = max_ms; fo_slowest_win_ = slowest; }
    fo_busy_tot_      += busy_ms;
    ++fo_evals_;
}

void HPOLog::arm_embf0(bool zero_mode, bool derivable)
{
    std::lock_guard<std::mutex> lg(mtx_);
    embf0_armed_     = true;
    embf0_zero_      = zero_mode;
    embf0_derivable_ = derivable;
}

void HPOLog::record_embf0_start(std::size_t eval_num, double f0)
{
    std::lock_guard<std::mutex> lg(mtx_);
    embf0_start_eval_ = eval_num;
    embf0_start_      = f0;
}

void HPOLog::record_embf0_best(std::size_t eval_num, double loss, double f0)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!(loss < embf0_best_loss_)) return;   /* log_new_best's guard */
    embf0_best_loss_ = loss;
    embf0_best_eval_ = eval_num;
    embf0_best_      = f0;
}

void HPOLog::write_density_report(
    const std::vector<detail::DensityReportEntry>& entries, bool rhobar)
{
    try {
        if (!entries.empty()) {
            const detail::DensitySummary& c = entries.front().summary;
            kv("Density form", c.derivable ? detail::describe_density_form(c.form)
                                           : "not derivable: " + c.reason);
        }
        for (const auto& e : entries) {
            kv("Density rho(r), " + e.label,
               detail::format_density_rho(e.summary) + e.note);
            if (rhobar)
                kv("Density rhobar, " + e.label,
                   detail::format_density_rhobar(e.summary));
        }
        std::lock_guard<std::mutex> lg(mtx_);
        density_armed_  = true;
        density_rhobar_ = rhobar;
    } catch (...) { /* the report must never fail a run */ }
}

void HPOLog::record_density_best(std::size_t eval_num, double loss,
                                 const detail::DensitySummary& summary)
{
    try {
        std::string rho = detail::format_density_rho(summary);
        std::string rb = detail::format_density_rhobar(summary);
        std::lock_guard<std::mutex> lg(mtx_);
        if (!(loss < density_best_loss_)) return;   /* log_new_best's guard */
        density_best_loss_   = loss;
        density_best_eval_   = eval_num;
        density_best_rho_    = std::move(rho);
        density_best_rhobar_ = std::move(rb);
    } catch (...) { /* diagnostic only */ }
}

/* -------------------------------------------------------------------------
 * Progress events
 * ------------------------------------------------------------------------- */

void HPOLog::log_new_best(std::size_t eval_num,
                          double      loss,
                          double      emae,   bool has_emae,
                          double      ermse,  bool has_ermse,
                          double      errmse, bool has_errmse,
                          double      frmse,  bool has_frmse,
                          double      srmse,  bool has_srmse,
                          double      wall_ms,
                          const std::string& top_breakdown)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    /* Defensive guard: refuse to record a "new best" that is not actually
     * better than the current best. This catches upstream bugs where the
     * is_best comparator (e.g. HPO_Host::g_best_) drifted out of sync with
     * the journal -- without this guard, a stale comparator would silently
     * overwrite cur_best_loss_ with a worse value and downstream summaries
     * would report regression as progress. cur_best_loss_ defaults to
     * numeric_limits<double>::max(), so the very first best at any finite
     * loss is still accepted. */
    if (loss >= cur_best_loss_) return;

    cur_best_loss_ = loss;
    cur_best_eval_ = eval_num;
    /* Snapshot the recording thread's annealer eval context (log_new_best
     * runs on the thread that evaluated the point). Always overwritten --
     * an untagged best clears a previous tag rather than inheriting it. */
    cur_best_slot_   = tl_slot;
    cur_best_nslots_ = tl_nslots;
    cur_best_T_      = tl_T;
    cur_best_T_label_ = tl_T_label;
    if (tl_slot >= 0) {
        ++best_slot_win_[tl_slot];
        ++best_slot_tot_[tl_slot];
        best_slot_last_[tl_slot] = eval_num;
    }
    if (has_emae)   { best_emae_   = emae;   has_emae_   = true; }
    if (has_ermse)  { best_ermse_  = ermse;  has_ermse_  = true; }
    if (has_errmse) { best_errmse_ = errmse; has_errmse_ = true; }
    if (has_frmse)  { best_frmse_  = frmse;  has_frmse_  = true; }
    if (has_srmse)  { best_srmse_  = srmse;  has_srmse_  = true; }

    /* The "BEST" tag column already announces the event, so the message
     * itself carries only the data (the old "* NEW BEST" prefix was
     * redundant). */
    std::ostringstream msg;
    msg << "* eval=" << std::setw(6) << eval_num
        << "  loss=" << fmt_double(loss);
    if (has_emae)   msg << "  EMAE="   << std::fixed << std::setprecision(2)
                        << emae   << " meV/at";
    if (has_ermse)  msg << "  ERMSE="  << std::fixed << std::setprecision(2)
                        << ermse  << " meV/at";
    if (has_errmse) msg << "  ErRMSE=" << std::fixed << std::setprecision(2)
                        << errmse << " per mille";
    if (has_frmse)  msg << "  FRMSE="  << std::fixed << std::setprecision(3)
                        << frmse  << " eV/Ang";
    if (has_srmse)  msg << "  SRMSE="  << std::fixed << std::setprecision(3)
                        << srmse  << " eV/A^3";
    msg << "  (" << fmt_millis(wall_ms) << ")";
    if (tl_slot == kSlotProbe) {
        msg << "  [T0 probe]";
    } else if (tl_slot >= 0) {
        msg << "  [" << slot_tag(tl_slot, tl_nslots, tl_T, tl_T_label) << "]";
    }

    f_ << "[" << now_hms() << "]  BEST     " << msg.str() << "\n";
    if (!top_breakdown.empty()) {
        /* Continuation line: blank timestamp column + tag, like
         * multi-line INFO entries (the head line above holds the entry
         * star); grep for "top:" to extract these lines later. */
        f_ << std::string(12, ' ') << "BEST       top: "
           << top_breakdown << "\n";
    }
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
       << ": evaluation produced NaN/Inf -- loss clamped to FAIL_SCORE\n";
    if (!params_str.empty())
        f_ << "           Parameters: " << params_str << "\n";
    if (!bound_hints.empty())
        f_ << "           " << bound_hints << "\n";
    if (logged == 9)
        f_ << "           (further NaN events will be summarised every 10 occurrences)\n";
    f_.flush();
}

/* -------------------------------------------------------------------------
 * Final summary
 * ------------------------------------------------------------------------- */

void HPOLog::set_status(const std::string& status_str)
{
    std::lock_guard<std::mutex> lg(mtx_);
    last_status_ = status_str;
}

void HPOLog::clear_status()
{
    std::lock_guard<std::mutex> lg(mtx_);
    last_status_.clear();
}

std::string HPOLog::status() const
{
    std::lock_guard<std::mutex> lg(mtx_);
    return last_status_;
}

void HPOLog::finish(const std::string& status_str)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    /* If caller passed an explicit status, use it (overrides any set_status).
     * If empty, fall back to whatever the inner driver recorded most recently
     * via set_status(). If neither, use a generic placeholder. */
    std::string effective_status = !status_str.empty()
        ? status_str
        : (last_status_.empty() ? std::string("COMPLETED") : last_status_);
    /* Rebind the local arg name so the rest of the body keeps working. */
    const std::string& status_str_local = effective_status;

    const std::size_t total  = n_eval_.load(std::memory_order_relaxed);
    const std::size_t nnan   = n_nan_ .load(std::memory_order_relaxed);
    const double nan_pct     = total > 0 ? 100.0 * nnan / total : 0.0;
    const double avg_ms      = total > 0 ? sum_ms_ / total : 0.0;
    const double wall_sec    = std::chrono::duration<double>(
                                   std::chrono::system_clock::now() - t_start_).count();

    /* DX: cur_best_loss_ starts at DBL_MAX and log_new_best only lowers it,
     * so a run that recorded no best printed the sentinel verbatim --
     * "Best total loss : 1.7977e+308 (eval 0)" -- immediately followed by
     * "Best potential saved : best_pot.tadah", a file that was never
     * written. Both lines have to know that "no best" is a state. */
    const bool have_best =
        cur_best_eval_ > 0 &&
        cur_best_loss_ < std::numeric_limits<double>::max();
    /* A candidate that could not be persisted beat every best saved before
     * it (the host compared it with the persisted best and did not advance
     * that). So a saved best with a LOWER loss was persisted after it -- by
     * another chain's single attempt once the storage recovered
     * (durable_best::FailLatch) -- and replaced any rename of the candidate
     * that did land. Of the candidates that may be on disk, only those at
     * or below the saved best's loss can still be in best_pot.tadah
     * (`on_disk`, lowest loss first; an equal loss is ambiguous). The
     * reported candidate is superseded by a lower saved best, or by an
     * equal one when no candidate can still be on disk: then
     * best_pot.tadah holds the run's best. The run still ends FAILED (the
     * status is the caller's); only what is reported as the best changes. */
    std::vector<std::pair<std::size_t, double>> on_disk;
    if (unsaved_best_) {
        for (const auto& c : unsaved_best_->maybe_on_disk)
            if (!have_best || c.second <= cur_best_loss_) on_disk.push_back(c);
        std::stable_sort(on_disk.begin(), on_disk.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
    }
    const bool unsaved_superseded =
        unsaved_best_ && have_best &&
        (cur_best_loss_ < unsaved_best_->loss ||
         (cur_best_loss_ == unsaved_best_->loss && on_disk.empty()));
    /* The best potential is not (or not confirmed) on disk. */
    const bool best_unsaved =
        unsaved_best_ ? !unsaved_superseded
                      : best_persist_failed_.load(std::memory_order_relaxed);

    /* -- DIAGNOSTICS --------------------------------------------------- */
    f_ << "\n"; { std::string s = "-- DIAGNOSTICS "; while(s.size()<72) s+='-'; f_ << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Total evaluations"
       << ": " << total << "\n";
    f_ << "  " << std::left << std::setw(KW) << "NaN/Inf clampings"
       << ": " << nnan;
    if (total > 0)
        f_ << "  (" << std::fixed << std::setprecision(1) << nan_pct << "%)";
    f_ << "\n";

    // Report the observation without inferring a cause from its rate alone.
    if (total == 0) {
        f_ << "  [info] No evaluations recorded; NaN/Inf rate is unavailable.\n";
    } else if (nnan == 0) {
        f_ << "  [ok] No NaN/Inf clampings were recorded.\n";
    } else {
        f_ << "  [!] " << std::fixed << std::setprecision(1) << nan_pct
           << "% of evaluations produced a non-finite loss that was clamped.\n"
              "    This counter does not identify the cause. Inspect the matching\n"
              "    WARNING entries before changing bounds, the model, or FAIL_SCORE.\n";
    }

    /* Graded failures. Disjoint from the NaN counter above: a killed worker
     * or a failed script produces a large but FINITE failscore, so it never
     * reaches was_nan_clamped. Without this line a run can fail almost every
     * evaluation and still report "No NaN/Inf clampings were recorded". */
    {
        const std::size_t fa = n_fail_band_[1].load(std::memory_order_relaxed);
        const std::size_t fb = n_fail_band_[2].load(std::memory_order_relaxed);
        const std::size_t fc = n_fail_band_[3].load(std::memory_order_relaxed);
        const std::size_t fd = n_fail_band_[4].load(std::memory_order_relaxed);
        const std::size_t ft = fa + fb + fc + fd;
        f_ << "  " << std::left << std::setw(KW) << "Graded failures"
           << ": " << ft;
        if (total > 0)
            f_ << "  (" << std::fixed << std::setprecision(1)
               << 100.0 * static_cast<double>(ft) / static_cast<double>(total)
               << "%)";
        f_ << "\n";
        if (ft > 0) {
            f_ << "    A=" << fa << " (hot-var out of range)"
               << "  B=" << fb << " (script failed)"
               << "  C=" << fc << " (did not complete)"
               << "  D=" << fd << " (pot build / gated)\n";
            static const char* const gate_line[kGateKinds] = {
                "    of which bump-support gate rejects: ",
                "    of which rho-sign gate rejects: ",
                "    of which lattice-rhobar gate rejects: "};
            static const char* const gate_why[kGateKinds] = {
                " (c + w/eta past RCUT, rejected before training)\n",
                " (rho_min < -tol*rho_max on the RHOGATE window, rejected "
                "before training)\n",
                " (rhobar(a) not strictly decreasing, rejected before "
                "training)\n"};
            for (std::size_t g = 0; g < kGateKinds; ++g) {
                const std::size_t n = n_gate_[g].load(std::memory_order_relaxed);
                if (n > 0) f_ << gate_line[g] << n << gate_why[g];
            }
        }
    }

    /* Unusable losses: every evaluation whose loss cannot become the
     * incumbent -- NaN/Inf clamped, FAILSCORE (a skip cap, a failed script,
     * an evaluation that threw) or a graded failure. The two counts above
     * see only part of them, so the verdict reads this one: the journal
     * used to follow "100.0% of evaluations produced a non-finite loss"
     * with "[ok] Every evaluation produced a usable loss", which it printed
     * whenever no failure was graded. */
    {
        const std::size_t nu = n_unusable_.load(std::memory_order_relaxed);
        f_ << "  " << std::left << std::setw(KW) << "Unusable losses"
           << ": " << nu;
        if (total > 0)
            f_ << "  (" << std::fixed << std::setprecision(1)
               << 100.0 * static_cast<double>(nu) / static_cast<double>(total)
               << "%)";
        f_ << "\n";
        /* A multi-rank run keeps one journal per rank, and these counts are
         * this rank's: the verdict names the rank, since whether the run
         * produced a usable loss is decided over every rank. */
        const std::string on_rank = scope_world_ > 1
            ? " on MPI rank " + std::to_string(scope_rank_) + " (of " +
                  std::to_string(scope_world_) + ")"
            : std::string();
        if (total == 0) {
            /* nothing to judge; the NaN/Inf line says so */
        } else if (nu == 0) {
            f_ << "  [ok] Every evaluation" << on_rank
               << " produced a usable loss.\n";
        } else if (nu == total) {
            f_ << "  [!] No evaluation" << on_rank
               << " produced a usable loss.\n";
        } else if (2 * nu >= total) {
            if (on_rank.empty())
                f_ << "  [!] The majority of evaluations failed. The reported best\n"
                      "    may predate the onset -- check when the failures started\n"
                      "    before trusting it.\n";
            else
                f_ << "  [!] The majority of evaluations" << on_rank
                   << " failed.\n"
                      "    The reported best may predate the onset -- check when\n"
                      "    the failures started before trusting it.\n";
        }
    }

    /* RHOFORM NONNEG / MONOTONE: evaluations whose AMPGRIDMB coordinates
     * were projected onto [0, inf). Printed only when one was. */
    {
        const std::size_t np = n_form_proj_.load(std::memory_order_relaxed);
        if (np > 0)
            f_ << "  " << std::left << std::setw(KW) << "Positive-form projection"
               << ": " << np << " evaluation(s) with an AMPGRIDMB amplitude < 0,\n"
                  "    evaluated and recorded at 0 (points outside the OPTIM box: an\n"
                  "    optimiser that ignores the bounds)\n";
    }

    /* Finite differences and the OPTIM box (fd_stencil.h): each line only
     * when its count is > 0, so the DIAGNOSTICS of a run that never came
     * within a step of a bound stay byte-identical. */
    {
        const std::size_t oe = n_oob_eval_.load(std::memory_order_relaxed);
        const std::size_t sn = n_snap_.load(std::memory_order_relaxed);
        const std::size_t og = n_oob_grad_.load(std::memory_order_relaxed);
        const std::size_t hc = n_hybrid_centre_.load(std::memory_order_relaxed);
        if (sn > 0)
            f_ << "  " << std::left << std::setw(KW) << "Snapped onto the box"
               << ": " << sn << " (points proposed outside the OPTIM box "
                  "within\n    rounding, " << SearchSpace::kSnapUlps
               << " units in the last place, and evaluated on the bound)\n";
        if (oe > 0)
            f_ << "  " << std::left << std::setw(KW) << "Out-of-box evaluations"
               << ": " << oe << " (points outside the OPTIM box, evaluated "
                  "there: an\n    optimiser that ignores the bounds; see the "
                  "first such WARNING)\n";
        if (og > 0)
            f_ << "  " << std::left << std::setw(KW) << "Out-of-box gradients"
               << ": " << og << " (at points outside the OPTIM box, each "
                  "finite\n    difference stepped towards the box)\n";
        const std::size_t nr = n_nn_rebuild_.load(std::memory_order_relaxed);
        if (nr > 0)
            f_ << "  " << std::left << std::setw(KW) << "Neighbour lists rebuilt"
               << ": " << nr << " evaluation(s) with a cutoff above the "
                  "one-time neighbour\n    lists (sized for the OPTIM box), "
                  "evaluated with lists rebuilt for them;\n    see the first "
                  "such WARNING\n";
        if (!one_sided_.empty()) {
            std::size_t total_os = 0;
            std::string per_dim;
            for (const auto& e : one_sided_) {
                total_os += e.second;
                per_dim += (per_dim.empty() ? "" : ", ") + e.first + " " +
                           std::to_string(e.second);
            }
            f_ << "  " << std::left << std::setw(KW) << "One-sided differences"
               << ": " << total_os << " (a coordinate within a step of its "
                  "bound, or outside\n"
                  "    the box, differenced into it): " << per_dim << "\n";
        }
        if (hc > 0)
            f_ << "  " << std::left << std::setw(KW) << "HYBRID centre losses"
               << ": " << hc << " (evaluations of a gradient's point for its\n"
                  "    one-sided or forward differences, where the optimiser's "
                  "own loss could not\n    be reused)\n";
    }

    /* Transient I/O. Printed only when something happened, so the
     * DIAGNOSTICS of a clean run stay byte-identical. */
    {
        const std::size_t rd = n_io_retry_durable_.load(std::memory_order_relaxed);
        const std::size_t rw = n_io_retry_writer_ .load(std::memory_order_relaxed);
        const std::size_t rl = n_rows_lost_       .load(std::memory_order_relaxed);
        const std::size_t al = n_archives_lost_   .load(std::memory_order_relaxed);
        if (rd + rw + rl + al > 0) {
            f_ << "  " << std::left << std::setw(KW) << "Transient I/O retries"
               << ": " << rd + rw << " (best-potential " << rd
               << ", per-eval logs/archives " << rw << "); rows lost " << rl
               << "; archives lost " << al << "\n";
            if (rl + al > 0) {
                f_ << "  [!] Some per-evaluation log rows or potential archives were\n"
                      "    not written after retrying; see the WARNING entries.";
                /* The best potential has its own, fatal path: only vouch for
                 * it when it was persisted -- a run ending on a persistence
                 * failure may have no best_pot.tadah, or a stale one. */
                if (best_unsaved)
                    f_ << "\n    The best potential could not be persisted either;\n"
                          "    see Optimiser status under RESULT.\n";
                else if (have_best)
                    f_ << " The\n    best potential is published separately and is complete.\n";
                else
                    f_ << "\n";
            }
        }
    }

    /* -- PERFORMANCE --------------------------------------------------- */
    if (occ_seen_) write_bound_occupancy_locked(nullptr, /*final_table=*/true);

    { std::string s = "-- PERFORMANCE "; while(s.size()<72) s+='-'; f_ << "\n" << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Total wall time"
       << ": " << format_duration(wall_sec) << "\n";
    f_ << "  " << std::left << std::setw(KW) << "Mean eval latency"
       << ": " << fmt_millis(avg_ms)
       << "  (per worker thread; can approach ~K x wall cost when K evaluations\n"
          "   are fully concurrent)\n";
    if (wall_sec > 0 && total > 0)
        f_ << "  " << std::left << std::setw(KW) << "Wall per eval"
           << ": " << fmt_millis(1000.0 * wall_sec / total) << "\n";
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
    if (fo_evals_ > 0 && fanout_pool_width_ > 0 && wall_sec > 0.0) {
        const double util = fo_busy_tot_ /
            (static_cast<double>(fanout_pool_width_) * wall_sec * 1000.0);
        std::ostringstream fo;
        fo << std::fixed << std::setprecision(0) << 100.0 * util << "%";
        f_ << "  " << std::left << std::setw(KW) << "pc_lammps pool util"
           << ": " << fo.str()
           << "  (Sum script busy / (pool width " << fanout_pool_width_
           << " x total wall))\n";
        if (util < fo_util_note_)
            f_ << "  Note: low pool utilisation. Common causes (not mutually "
               << "exclusive): the\n"
               << "      train/predict phase dominates the eval, THREADS is "
               << "wider than the\n"
               << "      script workload, or scripts are imbalanced. Per-script "
               << "breakdown:\n"
               << "      TADAH_HPO_BENCH=1 -> hpo_script_times.csv.\n";
    }

    /* -- RESULT -------------------------------------------------------- */
    { std::string s = "-- RESULT "; while(s.size()<72) s+='-'; f_ << "\n" << s << "\n"; }
    f_ << "  " << std::left << std::setw(KW) << "Optimiser status"
       << ": " << status_str_local << "\n";
    /* A new best that could not be persisted is the run's best loss, but it
     * is NOT in best_pot.tadah (or not confirmed there); the last persisted
     * best, if any, follows as "Last saved best". A superseded candidate
     * (see unsaved_superseded) is listed after the saved best instead. */
    if (unsaved_best_ && !unsaved_superseded) {
        f_ << "  " << std::left << std::setw(KW) << "Best total loss"
           << ": " << fmt_double(unsaved_best_->loss)
           << "  (eval " << unsaved_best_->eval
           << ") -- NOT saved (best-potential persistence failed)\n";
        if (have_best)
            f_ << "  " << std::left << std::setw(KW) << "Last saved best"
               << ": " << fmt_double(cur_best_loss_)
               << "  (eval " << cur_best_eval_ << best_origin_str() << ")\n";
    } else if (have_best) {
        f_ << "  " << std::left << std::setw(KW) << "Best total loss"
           << ": " << fmt_double(cur_best_loss_)
           << "  (eval " << cur_best_eval_ << best_origin_str() << ")\n";
        if (unsaved_superseded)
            f_ << "  " << std::left << std::setw(KW) << "Unsaved candidate"
               << ": " << fmt_double(unsaved_best_->loss)
               << "  (eval " << unsaved_best_->eval
               << ") -- could not be persisted; "
               << (cur_best_loss_ < unsaved_best_->loss ? "superseded by eval "
                                                        : "same loss as eval ")
               << cur_best_eval_ << "\n";
    } else {
        f_ << "  " << std::left << std::setw(KW) << "Best total loss"
           << ": (none -- no evaluation produced a usable loss)\n";
    }
    if (has_emae_)
        f_ << "  " << std::left << std::setw(KW) << "Best EMAE"
           << ": " << std::fixed << std::setprecision(2) << best_emae_ << " meV/atom\n";
    if (has_ermse_)
        f_ << "  " << std::left << std::setw(KW) << "Best ERMSE"
           << ": " << std::fixed << std::setprecision(2) << best_ermse_ << " meV/atom\n";
    if (has_errmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best ErRMSE"
           << ": " << std::fixed << std::setprecision(2) << best_errmse_ << " per mille\n";
    if (has_frmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best FRMSE"
           << ": " << std::fixed << std::setprecision(4) << best_frmse_ << " eV/Ang\n";
    if (has_srmse_)
        f_ << "  " << std::left << std::setw(KW) << "Best SRMSE"
           << ": " << std::fixed << std::setprecision(4) << best_srmse_ << " eV/A^3\n";
    if (embf0_armed_) {
        std::ostringstream s;
        s << (embf0_zero_ ? "EMBF0 ZERO (removed per atom): "
                          : "EMBF0 FREE (isolated-atom embedding energy): ");
        if (!embf0_derivable_) {
            s << "n/a (the model is not linear in the descriptor)";
        } else {
            s << "start ";
            if (embf0_start_eval_ > 0)
                s << "(eval " << embf0_start_eval_ << ") "
                  << fmt_double(embf0_start_, 6) << " eV";
            else
                s << "n/a";
            s << ", best ";
            /* The recorded best must be the reported one: a value is
             * recorded for every new best of this run (runs are not
             * resumed), so n/a means the model was not derivable. */
            if (embf0_best_eval_ > 0 && have_best &&
                embf0_best_eval_ == cur_best_eval_)
                s << "(eval " << embf0_best_eval_ << ") "
                  << fmt_double(embf0_best_, 6) << " eV";
            else
                s << "n/a";
        }
        f_ << "  " << std::left << std::setw(KW) << "Embedding F(0)"
           << ": " << s.str() << "\n";
    }
    /* DM_REAM density of the best (write_density_report armed it). The
     * recorded summary must be the reported best's. The host records one
     * for every new best unless it could not build the density at startup
     * (the -c pot row then reads "not derivable: <reason>"). */
    if (density_armed_) {
        const bool recorded = density_best_eval_ > 0 && have_best &&
                              density_best_eval_ == cur_best_eval_;
        const std::string none = have_best
            ? "n/a (not derivable)" : "n/a (no best)";
        const std::string at = "best (eval " + std::to_string(density_best_eval_) + ") ";
        f_ << "  " << std::left << std::setw(KW) << "Density rho(r)"
           << ": " << (recorded ? at + density_best_rho_ : none) << "\n";
        if (density_rhobar_)
            f_ << "  " << std::left << std::setw(KW) << "Density rhobar"
               << ": " << (recorded ? at + density_best_rhobar_ : none) << "\n";
    }
    f_ << "  " << std::left << std::setw(KW) << "Best potential saved"
       << ": ";
    if (!unsaved_best_) {
        f_ << (have_best ? "best_pot.tadah" : "(none written)");
    } else if (unsaved_superseded) {
        f_ << "best_pot.tadah (eval " << cur_best_eval_ << ")";
    } else if (!on_disk.empty()) {
        f_ << "not confirmed -- best_pot.tadah may hold eval ";
        for (std::size_t k = 0; k < on_disk.size(); ++k)
            f_ << (k ? " or eval " : "") << on_disk[k].first;
        f_ << " (write not synced) or ";
        if (have_best) f_ << "eval " << cur_best_eval_;
        else           f_ << "be absent";
        f_ << "; check its HPO_SOURCE_EVAL header";
    } else if (have_best) {
        f_ << "best_pot.tadah (eval " << cur_best_eval_ << ")";
    } else {
        f_ << "(none written)";
    }
    f_ << "\n";

    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    f_ << "  " << std::left << std::setw(KW) << "Finished"
       << ": " << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << "\n";

    f_ << "\n" << HEAVY << "\n\n";
    f_.flush();
    f_.close();
}

/* -------------------------------------------------------------------------
 * nlopt result code -> human-readable string
 * ------------------------------------------------------------------------- */
std::string nlopt_result_str(int code)
{
    switch (code) {
        case  1: return "SUCCESS -- generic success";
        case  2: return "STOPVAL_REACHED -- objective below stopval threshold";
        case  3: return "FTOL_REACHED -- function tolerance converged";
        case  4: return "XTOL_REACHED -- parameter tolerance converged";
        case  5: return "MAXEVAL_REACHED -- maximum evaluations exhausted";
        case  6: return "MAXTIME_REACHED -- maximum time exhausted";
        case -1: return "FAILURE -- generic failure";
        case -2: return "INVALID_ARGS -- invalid algorithm arguments (check bounds/dimensions)";
        case -3: return "OUT_OF_MEMORY -- insufficient memory for algorithm";
        case -4: return "ROUNDOFF_LIMITED -- progress blocked by floating-point roundoff";
        case -5: return "FORCED_STOP -- optimisation was interrupted";
        default: return "UNKNOWN (" + std::to_string(code) + ")";
    }
}

} // namespace tadah::hpo

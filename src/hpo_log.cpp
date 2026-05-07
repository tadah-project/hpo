#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/load_transforms.h>

#include <algorithm>
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

void HPOLog::reset_for_test()
{
    std::lock_guard<std::mutex> lg(mtx_);
    n_eval_.store(0, std::memory_order_relaxed);
    n_nan_ .store(0, std::memory_order_relaxed);
    cur_best_loss_ = std::numeric_limits<double>::max();
    cur_best_eval_ = 0;
    best_emae_ = best_ermse_ = best_errmse_ = best_frmse_ = best_srmse_ = 0.0;
    has_emae_ = has_ermse_ = has_errmse_ = has_frmse_ = has_srmse_ = false;
}

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
    /* Strategy-aware library/algo summary. With TYPE BASINS the OUTER
     * lib/algo are unset (the policy uses spec.basins.explore/refine);
     * surface those instead. PRESTAGE shows OUTER as the refinement
     * driver and reports EXPLORE separately under Run policy below. */
    auto strategy_name = [&](StrategyType t) -> const char* {
        switch (t) {
            case StrategyType::PLAIN:      return "PLAIN";
            case StrategyType::BASINS:     return "BASINS";
            case StrategyType::MULTISTART: return "MULTISTART";
            case StrategyType::PRESTAGE:   return "PRESTAGE";
        }
        return "?";
    };
    kv("Strategy", std::string(strategy_name(sp.strategy_type)));
    if (sp.strategy_type == StrategyType::BASINS) {
        if (sp.basins.explore)
            kv("EXPLORE library / algo",
               sp.basins.explore->lib + " / " + sp.basins.explore->algo);
        if (sp.basins.refine)
            kv("REFINE library / algo",
               sp.basins.refine->lib + " / " + sp.basins.refine->algo);
    } else {
        kv("Library / algo", sp.lib + " / " + sp.algo);
    }
    kv("Dimensions",     std::to_string(sp.low.size()) + " parameter"
                         + (sp.low.size() != 1 ? "s" : "") + " being optimised");

    // ---- inner optimizer (NLopt MLSL/AUGLAG sub-algo)
    if (sp.inner)
        kv("Inner optimizer", sp.inner->lib + " / " + sp.inner->algo);
    if (sp.strategy_type == StrategyType::BASINS) {
        if (sp.basins.explore && sp.basins.explore->inner)
            kv("EXPLORE inner",
               sp.basins.explore->inner->lib + " / " + sp.basins.explore->inner->algo);
        if (sp.basins.refine && sp.basins.refine->inner)
            kv("REFINE inner",
               sp.basins.refine->inner->lib + " / " + sp.basins.refine->inner->algo);
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
        if (stop_is_empty(sp.stop))
            f_ << "    (none set — optimiser may run indefinitely)\n";
        f_.flush();
    }

    // ---- stop criteria (INNER sub-spec).
    // Critical for NLopt MLSL/G_MLSL_LDS: with no INNER stop criteria
    // set, NLopt defaults to ftol_rel=1e-15 and xtol_rel=1e-7 for the
    // inner local searches — far tighter than a noisy LAMMPS objective
    // can ever satisfy, so each descent runs until something else
    // intervenes. Surface the resolved values (or the silent default)
    // so the user can see what is actually being asked of the inner.
    if (sp.inner) {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Inner stop criteria:\n";
        write_stop_criteria(sp.inner->stop, "    ");
        if (stop_is_empty(sp.inner->stop)) {
            const bool is_mlsl = (sp.lib == "NLOPT" &&
                (sp.algo == "G_MLSL" || sp.algo == "G_MLSL_LDS"));
            if (is_mlsl) {
                f_ << "    (none set — NLopt MLSL will use defaults "
                      "ftol_rel=1e-15 xtol_rel=1e-7 for inner local searches; "
                      "loosen these if inner descents stall on a noisy objective)\n";
            } else {
                f_ << "    (none set — inner sub-algo may run indefinitely)\n";
            }
        }
        f_.flush();
    }

    // ---- algorithm-specific params
    if (sp.pop)            kv("Population size", std::to_string(*sp.pop));
    if (sp.vector_storage) kv("Vector storage",  std::to_string(*sp.vector_storage));
    if (sp.seed)           kv("Random seed",     std::to_string(*sp.seed));
    if (!sp.param.empty()) {
        std::vector<std::pair<std::string, double>> params(sp.param.begin(), sp.param.end());
        std::sort(params.begin(), params.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Params:\n";
        for (const auto& kvp : params) {
            f_ << "    " << std::left << std::setw(KW-4) << kvp.first
               << ": " << fmt_double(kvp.second) << "\n";
        }
        f_.flush();
    }

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
        f_ << "  Starting point: user-supplied"
           << " (" << sp.optim_init.size() << " seed"
           << (sp.optim_init.size()==1 ? "" : "s") << ")\n";
        f_.flush();
    } else {
        std::lock_guard<std::mutex> lg(mtx_);
        f_ << "  Starting point: random (uniform within bounds)\n";
        f_.flush();
    }

    // ---- INIT_STRATEGY chain (resolved order, post WARM-reorder).
    // Helper used by both OUTER (here) and PRESTAGE (below in run-policy).
    auto write_init_strategy = [&](const InitStrategySpec& isp,
                                   const char* header,
                                   const char* indent) {
        f_ << indent << header << "\n";
        if (isp.chain.empty()) {
            f_ << indent << "  (default — implicit CONFIG only)\n";
        } else {
            auto pretty_entry = [](const auto& e) {
                std::string s = e.method;
                for (const auto& a : e.args) s += " " + a;
                return s;
            };
            auto chain = isp.chain;
            auto wit = std::find_if(chain.begin(), chain.end(),
                [](const auto& e){ return e.method == "WARM"; });
            if (wit != chain.end() && wit != chain.begin()) {
                auto w = std::move(*wit);
                chain.erase(wit);
                chain.insert(chain.begin(), std::move(w));
            }
            for (std::size_t i = 0; i < chain.size(); ++i)
                f_ << indent << "  [" << (i+1) << "] " << pretty_entry(chain[i]) << "\n";
        }
        if (isp.k_total)
            f_ << indent << "  INIT_K            : " << *isp.k_total << "\n";
        if (isp.seed)
            f_ << indent << "  INIT_SEED         : " << *isp.seed << "\n";
        f_ << indent << "  INIT_INCLUDE_CONFIG: "
           << (isp.include_config ? "true" : "false");
        if (isp.include_config && !isp.chain.empty())
            f_ << "  (suppressed — explicit chain present)";
        f_ << "\n";
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
        const bool is_basins    = (sp.strategy_type == StrategyType::BASINS);
        const bool is_multistart= (sp.strategy_type == StrategyType::MULTISTART && ms.n_starts > 1);
        const bool any_extra =
            is_multistart || has_prestage || is_basins ||
            (rs.after_stagnation > 0 && rs.max > 0) ||
            ns.autodetect || (ns.calibration_n >= 2);
        f_ << "  Run policy:\n";
        if (!any_extra) {
            f_ << "    (TYPE PLAIN — single-shot dispatch)\n";
        }
        if (is_multistart) {
            f_ << "    MULTISTART        : N_STARTS=" << ms.n_starts
               << "  (PARALLEL=" << ms.parallel << ")\n";
            if (ms.budget_per_start)
                f_ << "    BUDGET_PER_START  : " << *ms.budget_per_start << "\n";
        }
        if (has_prestage) {
            const auto& ps = *sp.prestage;
            f_ << "    PRESTAGE EXPLORE  : " << ps.lib << ":" << ps.algo
               << "  MAXEVAL=" << ps.stop.maxeval.value_or(0)
               << "  TOPK=" << sp.stage_topk << "\n";
            f_ << "    PRESTAGE REFINE   : " << sp.lib << ":" << sp.algo
               << "  MAXEVAL=" << sp.stop.maxeval.value_or(0) << "\n";
            write_init_strategy(ps.init_strategy_spec,
                                "PRESTAGE.EXPLORE init strategy:", "    ");
            if (ps.inner) {
                write_init_strategy(ps.inner->init_strategy_spec,
                                    "PRESTAGE.EXPLORE.INNER init strategy:", "      ");
            }
        }
        if (rs.after_stagnation > 0 && rs.max > 0) {
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
          << "  target=" << std::scientific << std::setprecision(3) << target
          << "  (per-dataset + avg)";
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
    kv("Output every",  std::to_string(trg.outputEvery) + " eval(s)");
    kv("Output digits", std::to_string(trg.outputDigits));
}

/* ─────────────────────────────────────────────────────────────────────────
 * Load-transforms section
 * ───────────────────────────────────────────────────────────────────────── */

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
     * counters are zero by construction — training-side filters are not
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

/* ─────────────────────────────────────────────────────────────────────────
 * Noise calibration section
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::write_noise_calibration_section(
    const std::vector<double>& losses,
    double                     suggested_ftol)
{
    section("NOISE CALIBRATION");

    /* Compute summary stats outside any lock — purely arithmetic. */
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
        f_ << "  (calibration not run — need >= 2 evals)\n";
        f_.flush();
        return;
    }

    /* Single critical section for all of this section's body — holding
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

            const std::size_t evals_since_best =
                (cur_best_eval_ > 0 && eval_num >= cur_best_eval_)
                    ? (eval_num - cur_best_eval_) : 0;

            std::ostringstream msg;
            msg << "--- " << std::setw(6) << eval_num << " evals done"
                << "   avg=" << fmt_millis(avg) << "/eval"
                << "   best=" << fmt_double(cur_best_loss_)
                << " (eval " << cur_best_eval_ << ", "
                << evals_since_best << " evals ago)"
                << "   NaN rate=" << std::fixed << std::setprecision(1)
                << nan_pct << "%";
            f_ << "[" << now_hms() << "]  SUMMARY  "
               << msg.str() << "\n";
            /* Stagnation hint: if we've spent more than half the recent
             * milestone window with no new best, point the user at the
             * common causes (LOCAL too tight for noise, basin already
             * exhausted, penalty cliff sampling). Single line, only
             * once per milestone. */
            if (cur_best_eval_ > 0 && evals_since_best >= 100 &&
                evals_since_best * 2 >= eval_num) {
                f_ << "[" << now_hms() << "]  HINT     "
                   << "no new best for " << evals_since_best
                   << " evals (>50% of run). Possible causes: "
                   << "(a) inner stop tolerance too tight for objective noise; "
                   << "(b) the best basin is genuinely exhausted; "
                   << "(c) penalty cliff — most samples land on "
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
                          double      wall_ms,
                          const std::string& top_breakdown)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    /* Defensive guard: refuse to record a "new best" that is not actually
     * better than the current best. This catches upstream bugs where the
     * is_best comparator (e.g. HPO_Host::g_best_) drifted out of sync with
     * the persisted journal state across a resume — without this guard, a
     * stale comparator would silently overwrite cur_best_loss_ with a
     * worse value and downstream summaries would report regression as
     * progress. cur_best_loss_ defaults to numeric_limits<double>::max(),
     * so the very first best at any finite loss is still accepted. */
    if (loss >= cur_best_loss_) return;

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
    if (has_emae)   msg << "  EMAE="   << std::fixed << std::setprecision(2)
                        << emae   << " meV/at";
    if (has_ermse)  msg << "  ERMSE="  << std::fixed << std::setprecision(2)
                        << ermse  << " meV/at";
    if (has_errmse) msg << "  ErRMSE=" << std::fixed << std::setprecision(2)
                        << errmse << " meV/at";
    if (has_frmse)  msg << "  FRMSE="  << std::fixed << std::setprecision(3)
                        << frmse  << " eV/Å";
    if (has_srmse)  msg << "  SRMSE="  << std::fixed << std::setprecision(3)
                        << srmse  << " GPa";
    msg << "  (" << fmt_millis(wall_ms) << ")";

    f_ << "[" << now_hms() << "]  BEST     " << msg.str() << "\n";
    if (!top_breakdown.empty()) {
        /* Indent under the BEST line so the breakdown is visually
         * attached. Cap line length defensively — if the caller
         * produced an over-long string, write it whole; the user can
         * grep for "top:" to extract these lines later. */
        f_ << "                       top: " << top_breakdown << "\n";
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

void HPOLog::set_status(const std::string& status_str)
{
    std::lock_guard<std::mutex> lg(mtx_);
    last_status_ = status_str;
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
       << ": " << status_str_local << "\n";
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

/* ─────────────────────────────────────────────────────────────────────────
 * HPOLog — checkpoint participant
 *
 * Persists the journal-running counters and best-so-far so an
 * interrupted run resumes with consistent state. The actual log file
 * is reopened in append mode by open(); these counters keep the
 * monotonic eval numbering and the "best so far" indicator coherent
 * across the boundary.
 * ───────────────────────────────────────────────────────────────────────── */

void HPOLog::save(core::Checkpoint& cp) const
{
    std::lock_guard<std::mutex> lg(mtx_);
    cp.section(tag());
    cp.put_size("n_eval",        n_eval_.load(std::memory_order_relaxed));
    cp.put_size("n_nan",         n_nan_ .load(std::memory_order_relaxed));
    cp.put_dbl ("cur_best_loss", cur_best_loss_);
    cp.put_size("cur_best_eval", cur_best_eval_);
    cp.put_dbl ("best_emae",     best_emae_);
    cp.put_dbl ("best_ermse",    best_ermse_);
    cp.put_dbl ("best_errmse",   best_errmse_);
    cp.put_dbl ("best_frmse",    best_frmse_);
    cp.put_dbl ("best_srmse",    best_srmse_);
    cp.put_bool("has_emae",      has_emae_);
    cp.put_bool("has_ermse",     has_ermse_);
    cp.put_bool("has_errmse",    has_errmse_);
    cp.put_bool("has_frmse",     has_frmse_);
    cp.put_bool("has_srmse",     has_srmse_);
    /* C3: persist NOISE_CALIBRATION's suggested_ftol so a resumed BASINS
     * run (NOISE_CALIBRATION is gated on `!resumed`) can still clamp the
     * ladder against the noise floor on its first round. */
    if (noise_suggested_ftol_)
        cp.put_dbl("noise_suggested_ftol", *noise_suggested_ftol_);
}

void HPOLog::load(const core::Checkpoint& cp)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (auto v = cp.get_size(tag(), "n_eval"))         n_eval_.store(*v);
    if (auto v = cp.get_size(tag(), "n_nan"))          n_nan_ .store(*v);
    if (auto v = cp.get_dbl (tag(), "cur_best_loss"))  cur_best_loss_ = *v;
    if (auto v = cp.get_size(tag(), "cur_best_eval"))  cur_best_eval_ = *v;
    if (auto v = cp.get_dbl (tag(), "best_emae"))      best_emae_   = *v;
    if (auto v = cp.get_dbl (tag(), "best_ermse"))     best_ermse_  = *v;
    if (auto v = cp.get_dbl (tag(), "best_errmse"))    best_errmse_ = *v;
    if (auto v = cp.get_dbl (tag(), "best_frmse"))     best_frmse_  = *v;
    if (auto v = cp.get_dbl (tag(), "best_srmse"))     best_srmse_  = *v;
    if (auto v = cp.get_bool(tag(), "has_emae"))       has_emae_   = *v;
    if (auto v = cp.get_bool(tag(), "has_ermse"))      has_ermse_  = *v;
    if (auto v = cp.get_bool(tag(), "has_errmse"))     has_errmse_ = *v;
    if (auto v = cp.get_bool(tag(), "has_frmse"))      has_frmse_  = *v;
    if (auto v = cp.get_bool(tag(), "has_srmse"))      has_srmse_  = *v;
    if (auto v = cp.get_dbl (tag(), "noise_suggested_ftol")) noise_suggested_ftol_ = *v;
}

} // namespace tadah::hpo

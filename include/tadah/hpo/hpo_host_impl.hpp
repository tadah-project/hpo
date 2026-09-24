#pragma once
#include <tadah/core/core_user_config_parser.h>
#include <tadah/hpo/probe_phase.h>
#include <tadah/hpo/log_index.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/core/logger.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/sims_grad_lammps.h>
#include <tadah/hpo/val_filter_forwarding.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/pot_header_writer_impl.hpp>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/hpo/script_executor.h>
#include <tadah/hpo/pc_task_exec.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/iter_health.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/result_drain.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/topk.h>
#include <tadah/tadah.lammps/lammps_tadah.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace {
/* ---------- utility ---------------------------------------------------- */
inline constexpr double LOG_2 = 0.6931471805599453;
inline double stable_log_cosh(double r)
{
    double a = std::abs(r);
    return a + std::log1p(std::exp(-2.0 * a)) - LOG_2;
}
  
} // empty namespace

namespace tadah::hpo {

/* ----------  per-phase timing harness (TADAH_HPO_BENCH=1) ------------ */
namespace bench_detail {
  inline bool enabled() {
    static const bool e = []{
      const char* v = std::getenv("TADAH_HPO_BENCH");
      return v && *v && std::string(v) != "0";
    }();
    return e;
  }
  inline std::mutex& mtx() { static std::mutex m; return m; }
  inline std::ofstream& csv() {
    static std::ofstream f;
    if (!f.is_open()) {
      f.open("hpo_phase_times.csv" + hpo_rank_suffix(),
             std::ios::out | std::ios::trunc);
      f << "iter,total_ms,train_ms,predict_ms,lammps_ms,sims_loss_ms,submit_ms,nthreads\n";
    }
    return f;
  }
  inline double elapsed_ms(std::chrono::high_resolution_clock::time_point a,
                           std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }
} // namespace bench_detail

/* ----------  per-stage watchdog --------------------------------------
 *
 * Each worker thread that enters operator() registers itself in a
 * shared table on entry to a pipeline stage (train / predict / lammps
 * / etc.) and clears the entry on exit. Two ways the table is read:
 *
 *   1. End-of-stage post-hoc check. If the just-finished stage took
 *      longer than `TADAH_HPO_STAGE_WARN_MS` (default 30 000 ms),
 *      a structured journal warning names the stage + duration. This
 *      catches stages that complete but were slow.
 *
 *   2. Snapshot from the writer's gap-skip block (Bug 6). When the
 *      writer detects a hung iter and is about to skip it, it asks
 *      the watchdog for any thread whose stage has been running longer
 *      than the gap-skip threshold. The names of those stages are
 *      appended to the gap-skip warning so the journal alone identifies
 *      *which* part of the eval pipeline is hanging.
 *
 * Lock contention is low: each stage transition takes the table mutex
 * briefly (a vector lookup + assignment). With 22 worker threads and
 * a few stage transitions per eval, this adds <1% overhead.
 *
 * Disable by setting `TADAH_HPO_STAGE_WARN_MS=0` (no post-hoc warnings;
 * the writer-side snapshot still runs as it is essentially free). */
namespace stage_watchdog {

struct StageState {
    std::thread::id                       tid;
    std::size_t                           iter      = 0;
    const char*                           stage     = nullptr;  // nullptr = idle
    std::chrono::steady_clock::time_point started{};
};

inline std::mutex& mtx() { static std::mutex m; return m; }
inline std::vector<StageState>& table() { static std::vector<StageState> v; return v; }

inline std::int64_t threshold_ms() {
    static const std::int64_t v = []{
        const char* s = std::getenv("TADAH_HPO_STAGE_WARN_MS");
        return s && *s ? std::atoll(s) : 30'000;   // 30 s default
    }();
    return v;
}

/* Register the current thread as entering `stage` for `iter`. */
inline void enter(std::size_t iter, const char* stage) {
    const auto tid = std::this_thread::get_id();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mtx());
    auto& v = table();
    for (auto& s : v) {
        if (s.tid == tid) {
            s.iter = iter; s.stage = stage; s.started = now;
            return;
        }
    }
    v.push_back({tid, iter, stage, now});
}

/* Mark the current thread idle and emit a post-hoc warning if the
 * just-finished stage took longer than the threshold. Returns the
 * elapsed milliseconds (or 0 if the thread was not registered). */
template<class LogFn>
inline std::int64_t leave(std::size_t iter, const char* stage, LogFn&& log) {
    std::int64_t took_ms = 0;
    {
        const auto tid = std::this_thread::get_id();
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(mtx());
        auto& v = table();
        for (auto& s : v) {
            if (s.tid == tid) {
                took_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - s.started).count();
                s.stage = nullptr;
                break;
            }
        }
    }
    const auto thr = threshold_ms();
    if (thr > 0 && took_ms > thr) {
        log(std::string("eval ") + std::to_string(iter) +
            " stage '" + stage + "' took " +
            std::to_string(took_ms) + " ms (threshold " +
            std::to_string(thr) + " ms) -- possible hang surface");
    }
    return took_ms;
}

/* Snapshot of all thread-stages that have been running longer than
 * `min_age_ms`. Returned as a single human-readable string suitable
 * for embedding in a journal warning. Empty string if none. */
inline std::string snapshot_stuck(std::int64_t min_age_ms) {
    std::ostringstream oss;
    bool first = true;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mtx());
    for (const auto& s : table()) {
        if (!s.stage) continue;
        const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - s.started).count();
        if (age_ms < min_age_ms) continue;
        if (!first) oss << "; ";
        oss << "iter=" << s.iter << " stage='" << s.stage
            << "' age_ms=" << age_ms;
        first = false;
    }
    return oss.str();
}

/* RAII helper: enter on construction, leave on destruction. */
template<class LogFn>
struct Scope {
    std::size_t iter; const char* stage; LogFn log;
    Scope(std::size_t i, const char* s, LogFn l)
        : iter(i), stage(s), log(std::move(l)) { enter(iter, stage); }
    ~Scope() { leave(iter, stage, log); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

} // namespace stage_watchdog

/* ----------  bad-potential early-exit heuristic ----------------------
 * Skip the (expensive) LAMMPS stage when validation metrics show the
 * trained potential is unphysical. Per-metric thresholds are declared
 * in config.hpo on the PC_* line:
 *
 *   PC_ERMSE  <target> <weight>  --skip-above 200      # meV/atom
 *   PC_FRMSE  <target> <weight>  --skip-above 0.5  --skip-rel 5
 *
 * --skip-above:  skip when |metric| > cap (cap is in the metric's units).
 * --skip-rel:    skip when metric > rel * best_seen_so_far for that metric.
 *
 * Either trigger sets the eval's loss to FAIL_SCORE and bypasses LAMMPS.
 * Histories live in the run-owned MetricSkipTracker, keyed by metric and
 * (for indexed declarations) dataset identity. */
namespace header_detail {
  /* D18: truncated column labels must stay DISTINCT.
   *
   * Column titles are cut to max_title_len_ with a trailing '*', which is
   * fine until two different names share a prefix. run22's best_loss.tadah
   * carried FIVE colliding labels out of 245 columns -- L(rat_dH_P50*) was
   * both rat_dH_P50_4157 and rat_dH_P500_4157, and likewise for four
   * pen_fcc_P<n> pairs -- so any parser keyed by column name silently read
   * the wrong series. run23's config has eight such prefix clashes.
   *
   * Collisions are resolved by replacing the tail with ~2, ~3, ... in first-
   * appearance order, which keeps the label inside the width budget and
   * stable for a given config. `seen` must live for one file's header. */
  inline std::string unique_label(std::string name, std::size_t maxlen,
                                  std::set<std::string>& seen)
  {
    if (maxlen < 4) return name;                 /* nothing sensible to do */
    if (name.size() > maxlen)
      name = name.substr(0, maxlen - 1) + "*";
    if (seen.insert(name).second) return name;
    for (int n = 2; n < 1000; ++n) {
      const std::string sfx = "~" + std::to_string(n);
      const std::string base =
          name.size() + sfx.size() > maxlen
              ? name.substr(0, maxlen - sfx.size())
              : name;
      const std::string cand = base + sfx;
      if (seen.insert(cand).second) return cand;
    }
    return name;   /* unreachable in practice */
  }
}  // namespace header_detail



/* ----------  per-script lattice cache (warm start) -------------------
 *
 * The warmstart namespace lives in warmstart_cache.h so the WARM init
 * strategy (init_strategies.cpp) can prime the cache from `# HOT_A0`
 * lines saved into best_pot.tadah without having to include this whole
 * template-heavy header.
 *
 * The cache is now keyed by (chain, script, name) so PT/PA replicas do not
 * share a warm-start seed; a separate (script, name) seed map handles WARM
 * priming / best_pot persistence. `reject_count` (rate-limited warning) also
 * lives there. The out-of-range abort is a run-LOCAL flag in run_lammps, not
 * a global -- see the eval_aborted note there. */


/* -----------------  header writers  ---------------------------------- */
template<class P,class R>
void HPO_Host<P,R>::init_loss(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;
  /* D19: which replica produced this row. Under ANNEAL_PT/PA every
   * evaluation belongs to a slot, and without this column a row cannot
   * be attributed to one -- run22 wrote 2.5 GB of rows in which a
   * TOTAL_LOSS/N of 4.1e5 (a hot replica's rejected proposal) was
   * indistinguishable from a cold-chain result. Costs nothing to carry:
   * the driver already sets warmstart::current_chain() to the global
   * slot and every eval picks it up as Result::chain_id. 0 for serial,
   * NLOPT and dlib runs. */
  f << std::setw(slot_cw_) << "SLOT";
  if (trg_.use_emae) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(EMAE)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(EMAE)[avg]";
  }
  if (trg_.use_ermse) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(ERMSE)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(ERMSE)[avg]";
  }
  if (trg_.use_errmse) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(ErRMSE)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(ErRMSE)[avg]";
  }
  if (trg_.use_ersq) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(ERSQ)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(ERSQ)[avg]";
  }
  if (trg_.use_frmse) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(FRMSE)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(FRMSE)[avg]";
  }
  if (trg_.use_srmse) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_) << ("L(SRMSE)[" + std::to_string(i) + "]");
    f << std::setw(cw_) << "L(SRMSE)[avg]";
  }

  std::set<std::string> seen_loss;   /* D18: keep truncated titles distinct */
  for (const auto &lsim : trg_.LS) {
    for (const auto &varname : lsim.n_varloss)
      f << std::setw(cw_)
        << "L(" + header_detail::unique_label(varname, max_title_len_,
                                              seen_loss) + ")";
  }

  /* PC_OBSERVABLE columns: one L(<kind>[k]) per observable. The k
   * index disambiguates multiple entries of the same kind (e.g. two
   * cohesive_energy observables for different species). */
  for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
    const std::string label = "L(" + trg_.observables[k]->kind() + "["
                            + std::to_string(k) + "])";
    f << std::setw(cw_)
      << header_detail::unique_label(label, max_title_len_, seen_loss);
  }

  f << std::setw(cw_) << "TOTAL_LOSS/N" << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_outvar(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(iter_cw_) << "#ITERATION";
  f << std::setw(ittime_cw_) << "IT-TIME/ms" << std::right;
  f << std::setw(slot_cw_) << "SLOT";
  if (trg_.use_emae)
    f << std::setw(cw_) << "EMAE";
  if (trg_.use_ermse)
    f << std::setw(cw_) << "ERMSE";
  if (trg_.use_errmse)
    f << std::setw(cw_) << "ErRMSE";
  if (trg_.use_ersq)
    f << std::setw(cw_) << "ERSQ";
  if (trg_.use_frmse)
    f << std::setw(cw_) << "FRMSE";
  if (trg_.use_srmse)
    f << std::setw(cw_) << "SRMSE";

  std::set<std::string> seen_outvar;   /* D18 */
  for (const auto &lsim : trg_.LS) {
    for (const auto &n_outvar : lsim.n_outvar)
      f << std::setw(cw_)
        << header_detail::unique_label(n_outvar, max_title_len_, seen_outvar);
  }
  /* PC_OBSERVABLE values: <kind>[k] for each observable. */
  for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
    const std::string label = trg_.observables[k]->kind() + "["
                            + std::to_string(k) + "]";
    f << std::setw(cw_)
      << header_detail::unique_label(label, max_title_len_, seen_outvar);
  }
  f << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_params(const std::string& fn) const
{

  std::ofstream f(fn);
  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;
  f << std::setw(slot_cw_) << "SLOT";

  /* D18: label every column with its INDEX, not just its family name.
   * This used to print the bare key, so a 16-element family produced 16
   * identical "AMPGRIDMB" headings and a 174-column params.tadah could not
   * be read by name at all. param_labels_ already carries the indexed form
   * (built above from the same optimap traversal, so the order matches
   * exactly) and is what print_params_line's values correspond to. */
  std::set<std::string> seen_params;
  for (const auto &lbl : param_labels_) {
    /* Keep the INDEX when the label does not fit -- it is the part that
     * distinguishes the columns. Plain truncation would cut "AMPGRIDMB[12]"
     * down to "AMPGRIDMB[*" for every member of the family, leaving the
     * collision resolver to number them ~2, ~3, ... which is unique but
     * says nothing. Compressing the family name instead gives
     * "AMPGRID*[12]": still unambiguous, and still readable as index 12. */
    std::string shown = lbl;
    if (shown.size() > max_title_len_) {
      const std::size_t br = shown.rfind('[');
      if (br != std::string::npos && br > 0) {
        const std::string idx = shown.substr(br);          /* "[12]" */
        if (idx.size() + 2 <= max_title_len_)
          shown = shown.substr(0, max_title_len_ - idx.size() - 1) + "*" + idx;
      }
    }
    f << std::setw(cw_)
      << header_detail::unique_label(shown, max_title_len_, seen_params);
  }
  f << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_stats(const std::string& fn) const
{
  std::ofstream f(fn);
  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;
  f << std::setw(slot_cw_) << "SLOT";

  std::set<std::string> seen_stats;   /* D18 */
  auto write_header = [&](const std::string& label) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i)
      f << std::setw(cw_)
        << header_detail::unique_label(label + "[" + std::to_string(i) + "]",
                                       max_title_len_, seen_stats);
    f << std::setw(cw_)
      << header_detail::unique_label(label + "[avg]", max_title_len_,
                                     seen_stats);
  };

  if (trg_.use_emae)   write_header("EMAE");
  if (trg_.use_ermse)  write_header("ERMSE");
  if (trg_.use_errmse) write_header("ErRMSE");
  if (trg_.use_ersq)   write_header("ERSQ");
  if (trg_.use_frmse)  write_header("FRMSE");
  if (trg_.use_srmse)  write_header("SRMSE");
  f << std::endl;
  /* D18: say so when there is nothing to say. With no training/validation
   * database no per-structure metric is active, so this file used to be a
   * lone "#ITERATION" heading followed by one bare integer per evaluation
   * -- 2.9 MB of stats.tadah and 1.7 MB of best_stats.tadah in run22,
   * carrying no statistics at all and looking like a truncated file. The
   * rows are suppressed in print_stats_line; this records why. */
  if (!stats_have_columns())
    f << "# (no per-structure statistics: this run has no training or "
         "validation database, so no metric columns exist)\n";
}

/* True when at least one per-structure metric column exists. */
template<class P,class R>
bool HPO_Host<P,R>::stats_have_columns() const
{
  return trg_.use_emae || trg_.use_ermse || trg_.use_errmse ||
         trg_.use_ersq || trg_.use_frmse || trg_.use_srmse;
}

/* ----------  line printers (append) ---------------------------------- */
/* ======= line writers ================================================ */
template<class P,class R>
void HPO_Host<P,R>::print_loss_line(std::ofstream& f, size_t it,const Result& e,
                                    const HPOTargets& trg,
                                    std::size_t cw,int prec)
{
    f << std::left << std::setw(iter_cw_) << it;
    /* 1-based like every other slot a person reads (1.5.0); e.chain_id
     * itself stays 0-based -- it is the warm-start cache key and the
     * durable-best comparison key. */
    f << std::right << std::setw(slot_cw_) << logidx::uis(e.chain_id);
    f << std::scientific << std::setprecision(prec);

    auto write_loss_metric = [&](bool used, const std::vector<double>& per_ds, double avg) {
      if (!used) return;
      for (double v : per_ds) f << std::setw(cw) << v;
      f << std::setw(cw) << avg;
    };

    write_loss_metric(trg.use_emae,   e.v_emaeLoss,   e.emaeLoss);
    write_loss_metric(trg.use_ermse,  e.v_ermseLoss,  e.ermseLoss);
    write_loss_metric(trg.use_errmse, e.v_errmseLoss, e.errmseLoss);
    write_loss_metric(trg.use_ersq,   e.v_ersqLoss,   e.ersqLoss);
    write_loss_metric(trg.use_frmse,  e.v_frmseLoss,  e.frmseLoss);
    write_loss_metric(trg.use_srmse,  e.v_srmseLoss,  e.srmseLoss);
      
    for (size_t i = 0; i < e.v_outvar.size(); ++i) {
      auto &indices = trg.LS[i].i_varloss;
      for (const auto &j : indices) {
        f << std::setw(cw) << e.v_loss[i][j];
      }
    }

    /* PC_OBSERVABLE per-observable loss columns. e.v_obs_loss is empty
     * when trg.observables is empty (header is also empty then). */
    for (std::size_t k = 0; k < e.v_obs_loss.size(); ++k)
      f << std::setw(cw) << e.v_obs_loss[k];

    f << std::setw(cw) << e.loss << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::print_outvar_line(std::ofstream& f,size_t it,const Result& e,
                                      const HPOTargets& trg,
                                      std::size_t cw,int prec)
{
    f << std::left << std::setw(iter_cw_) << it;
    f << std::left << std::setw(ittime_cw_) << e.wall_ms.count();
    /* 1-based like every other slot a person reads (1.5.0); e.chain_id
     * itself stays 0-based -- it is the warm-start cache key and the
     * durable-best comparison key. */
    f << std::right << std::setw(slot_cw_) << logidx::uis(e.chain_id);

    if(trg.use_emae)   f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.emae;
    if(trg.use_ermse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.ermse;
    if(trg.use_errmse) f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.errmse;
    if(trg.use_ersq)   f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.ersq;
    if(trg.use_frmse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.frmse;
    if(trg.use_srmse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.srmse;

    for (size_t i = 0; i < e.v_outvar.size(); ++i) {
      for (size_t j = 0; j < e.v_outvar[i].size(); ++j) {
        double v;
        if (try_parse_double(e.v_outvar[i][j], v)) {
          f<<std::right << std::setw(cw) << std::scientific
            << std::setprecision(prec) << v;
        } else {
          f<<std::right << std::setw(cw) << "NaN";
        }
      }
    }
    /* PC_OBSERVABLE per-observable values. NaN is printed for failed
     * evaluations (relax divergence / model NaN). */
    for (std::size_t k = 0; k < e.v_obs_value.size(); ++k) {
      if (std::isfinite(e.v_obs_value[k]))
        f << std::right << std::setw(cw) << std::scientific
          << std::setprecision(prec) << e.v_obs_value[k];
      else
        f << std::right << std::setw(cw) << "NaN";
    }
    f<<std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::print_params_line(std::ofstream& f,size_t it,const Result& e,
                                      std::size_t cw,int prec)
{
  f << std::left << std::setw(iter_cw_) << it;
  /* 1-based like every other slot a person reads (1.5.0); e.chain_id
     * itself stays 0-based -- it is the warm-start cache key and the
     * durable-best comparison key. */
    f << std::right << std::setw(slot_cw_) << logidx::uis(e.chain_id);

  for(long i=0;i<e.params.size();++i)
  {
    f << std::right << std::setw(cw) << std::scientific
      << std::setprecision(prec) << e.params(i);
  }
  f << std::endl;

}


template<class P,class R>
void HPO_Host<P,R>::print_stats_line(std::ofstream& f, size_t it, const Result& e,
                                     const HPOTargets& trg,
                                     std::size_t cw, int prec)
{
  /* D18: with no metric enabled this row would be a bare iteration number
   * and nothing else -- run22 accumulated 4.6 MB of exactly that across
   * stats.tadah and best_stats.tadah. The header explains the emptiness
   * once; repeating an empty row per evaluation says nothing. */
  if (!(trg.use_emae || trg.use_ermse || trg.use_errmse ||
        trg.use_ersq || trg.use_frmse || trg.use_srmse))
    return;

  f << std::left << std::setw(iter_cw_) << it;
  /* 1-based like every other slot a person reads (1.5.0); e.chain_id
     * itself stays 0-based -- it is the warm-start cache key and the
     * durable-best comparison key. */
    f << std::right << std::setw(slot_cw_) << logidx::uis(e.chain_id);
  f << std::scientific << std::setprecision(prec);

  auto write_metric = [&](const std::vector<double>& per_ds, double avg) {
    for (double v : per_ds) f << std::setw(cw) << v;
    f << std::setw(cw) << avg;
  };

  if (trg.use_emae)   write_metric(e.v_emae,   e.emae);
  if (trg.use_ermse)  write_metric(e.v_ermse,  e.ermse);
  if (trg.use_errmse) write_metric(e.v_errmse, e.errmse);
  if (trg.use_ersq)   write_metric(e.v_ersq,   e.ersq);
  if (trg.use_frmse)  write_metric(e.v_frmse,  e.frmse);
  if (trg.use_srmse)  write_metric(e.v_srmse,  e.srmse);
  f << std::endl;
}

/* ----------  static data definitions ---------------------------------- */
template<class P, class R>
std::atomic_size_t  HPO_Host<P,R>::g_iter_{0};
template<class P, class R>
std::atomic_size_t  HPO_Host<P,R>::g_best_iter_{0};
template<class P, class R>
std::atomic<double> HPO_Host<P,R>::g_best_{std::numeric_limits<double>::max()};
template<class P, class R>
std::mutex HPO_Host<P,R>::best_mtx_;
template<class P, class R>
BoundOccupancy HPO_Host<P,R>::bound_occ_;
template<class P, class R>
std::mutex HPO_Host<P,R>::bound_occ_mtx_;
template<class P, class R>
std::optional<durable_best::Key> HPO_Host<P,R>::g_best_key_;

template<class P,class R> typename HPO_Host<P,R>::Queue  HPO_Host<P,R>::g_q_;
template<class P,class R> std::atomic_flag               HPO_Host<P,R>::writer_token_;
template<class P,class R> std::mutex                     HPO_Host<P,R>::io_mtx_;
template<class P,class R> std::map<std::size_t,
  typename HPO_Host<P,R>::Result> HPO_Host<P,R>::io_buf_;
template<class P,class R> std::atomic<std::size_t>       HPO_Host<P,R>::next_flush_{1};
template<class P,class R> std::atomic<std::int64_t>      HPO_Host<P,R>::next_flush_stuck_since_ms_{0};
template<class P,class R> std::atomic_bool               HPO_Host<P,R>::stagnation_warned_{false};

/* Top-K + stagnation static state. Capacity 0 means top-K tracking is
 * disabled; threshold 0 means stagnation throws are disabled. */
template<class P,class R> std::mutex
  HPO_Host<P,R>::topk_mtx_;
template<class P,class R> std::size_t
  HPO_Host<P,R>::topk_capacity_{0};
template<class P,class R>
  std::vector<std::pair<double, P>> HPO_Host<P,R>::topk_buf_;
template<class P,class R>
  std::vector<double> HPO_Host<P,R>::topk_widths_;
template<class P,class R> std::atomic_size_t
  HPO_Host<P,R>::iter_at_last_best_{0};
template<class P,class R> std::atomic_size_t
  HPO_Host<P,R>::stagnation_threshold_{0};

/* Top-K accessors. Cheap when disabled. */
template<class P,class R>
void HPO_Host<P,R>::set_track_topk(std::size_t k, std::vector<double> widths) {
  std::lock_guard<std::mutex> lk(topk_mtx_);
  topk_capacity_ = k;
  topk_buf_.clear();
  topk_buf_.reserve(k);
  topk_widths_ = std::move(widths);
}
template<class P,class R>
std::vector<std::pair<double, P>> HPO_Host<P,R>::get_topk() {
  std::lock_guard<std::mutex> lk(topk_mtx_);
  auto copy = topk_buf_;
  std::sort(copy.begin(), copy.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; });
  return copy;
}
template<class P,class R>
void HPO_Host<P,R>::clear_topk() {
  std::lock_guard<std::mutex> lk(topk_mtx_);
  topk_buf_.clear();
}

/* Stagnation accessors. */
template<class P,class R>
void HPO_Host<P,R>::set_stagnation_threshold(std::size_t n) {
  stagnation_threshold_.store(n, std::memory_order_relaxed);
}
template<class P,class R>
std::size_t HPO_Host<P,R>::get_stagnation_threshold() {
  return stagnation_threshold_.load(std::memory_order_relaxed);
}
template<class P,class R>
void HPO_Host<P,R>::reset_stagnation() {
  iter_at_last_best_.store(g_iter_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
  /* Re-arm the once-per-event diagnostic gate so the next stagnation
   * event in this run produces a fresh "begin" message. */
  stagnation_warned_.store(false, std::memory_order_relaxed);
}
template<class P,class R>
std::size_t HPO_Host<P,R>::stagnation_count() {
  const std::size_t cur  = g_iter_.load(std::memory_order_relaxed);
  const std::size_t last = iter_at_last_best_.load(std::memory_order_relaxed);
  return cur > last ? cur - last : 0;
}

template<class P,class R>
std::size_t HPO_Host<P,R>::g_iter_value() {
  return g_iter_.load(std::memory_order_relaxed);
}

template<class P,class R>
void HPO_Host<P,R>::reset_globals_for_test() {
    {
        std::lock_guard<std::mutex> lock(best_mtx_);
        g_best_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
        g_best_iter_.store(0, std::memory_order_relaxed);
        iter_at_last_best_.store(0, std::memory_order_relaxed);
        g_iter_.store(0, std::memory_order_relaxed);
        g_best_key_.reset();
    }
    {
        /* A new process also starts with an empty result writer. Results a
         * test evaluated but never flushed belong to the simulated earlier
         * process; the final drain would otherwise write them into the next
         * test's files (and iteration numbers restart at 1 here). */
        std::lock_guard<std::mutex> lock(io_mtx_);
        Result item;
        while (g_q_.try_dequeue(item)) {}
        io_buf_.clear();
        next_flush_.store(1, std::memory_order_relaxed);
        next_flush_stuck_since_ms_.store(0, std::memory_order_relaxed);
    }
    stagnation_threshold_.store(0, std::memory_order_relaxed);
    stagnation_warned_.store(false, std::memory_order_relaxed);
    warmstart::reset_best_snapshot_for_fresh_run();
}

template<class P,class R>
void HPO_Host<P,R>::reset_best_state_for_fresh_run() {
    std::lock_guard<std::mutex> lock(best_mtx_);
    g_best_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
    g_best_iter_.store(0, std::memory_order_relaxed);
    /* Keep eval numbering/writer order monotonic when an embedding invokes
     * two HPO runs in one process, but start stagnation from this boundary. */
    iter_at_last_best_.store(g_iter_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
    g_best_key_.reset();
}

template<class P,class R>
void HPO_Host<P,R>::prepare_best_publication() const {
    std::string local_error;
    try {
        if (hpo_rank() == 0)
            durable_best::reset_for_fresh_run(trg_.canonicalBestPotFile);
        if (trg_.localBestPotFile != trg_.canonicalBestPotFile)
            durable_best::reset_for_fresh_run(trg_.localBestPotFile);
    } catch (const std::exception& e) {
        local_error = e.what();
    } catch (...) {
        local_error = "unknown best-potential preparation failure";
    }

    /* Every rank reaches both collectives even when its local filesystem
     * operation failed, so startup errors cannot strand peers in a barrier. */
    const int any_failed = hpo_allreduce_max(local_error.empty() ? 0 : 1);
    hpo_barrier();
    if (any_failed) {
        if (local_error.empty())
            local_error = "best-potential preparation failed on another MPI rank";
        throw durable_best::Error(local_error);
    }

    reset_best_state_for_fresh_run();
    warmstart::reset_best_snapshot_for_fresh_run();
}

template<class P, class R>
thread_local tadah::mlip::memory::MLIPWorkspaceManager
HPO_Host<P,R>::tls_ws_{};

template<class P, class R>
thread_local bool HPO_Host<P,R>::tls_trainer_theta_stale_ = false;

template<class P, class R>
thread_local typename HPO_Host<P,R>::TrainerCache
HPO_Host<P,R>::tls_trainer_cache_{};

/* ----------  constructor ---------------------------------------------- */
template<class P, class R>
HPO_Host<P,R>::HPO_Host(core::Context master_ctx,
                        HPOTargets&   trg,
                        const std::string& val_cfg)
: master_ctx_(std::move(master_ctx))
, trg_(trg), script_ewma_ms_(trg.LS.size()), spec_(trg.spec)
, use_lammps_( !trg.LS.empty() )
, has_train_data_( master_ctx_.contains("DBFILE")
                   && master_ctx_.size("DBFILE") > 0 )
, has_validation_( !val_cfg.empty() )
, prec_(trg.outputDigits)
, cw_(static_cast<std::size_t>(std::max(prec_ + 9, static_cast<int>(min_cw_))))
, max_title_len_(cw_ - 4)
{    /* Arm the bound-occupancy audit on this run's search space (a pure
     * function of the config, so re-arming on every host construction is
     * idempotent across policies). */
    if (!has_validation_ && (trg_.use_ermse || trg_.use_frmse))
        trg_.resolve_validation_targets({});
    {
        std::lock_guard<std::mutex> lg(bound_occ_mtx_);
        bound_occ_ = BoundOccupancy(trg.space());
    }

    /* Stage 1b.1: zero the fan-out cost estimates. All-zero on the first eval
     * means the LPT stable_sort keeps today's script order, so the first-eval
     * trajectory is byte-identical; the estimates converge from eval 2 on. */
    for (auto& a : script_ewma_ms_) a.store(0.0, std::memory_order_relaxed);

    /* knot channels: RCUT keys are derived from the largest knot (see
     * rcut_derivation.h). Flags are computed once here because the per-eval
     * hook lives in the const update_context(). */
    knot2b_ = models::potedit::derived_knot_rcut(master_ctx_, "2B").has_value();
    knotmb_ = models::potedit::derived_knot_rcut(master_ctx_, "MB").has_value();

    /* Per-candidate bump-support gate (bump_support_gate.h). Under STRICT
     * validate_hpo_setup already hard-errored on any violable box, so the
     * table is only built -- and only ever non-empty -- under CANDIDATE
     * with a partially infeasible OPTIM box. */
    if (!trg_.BUMPGATE_STRICT)
        bump_gate_ = detail::build_bump_gate_table(master_ctx_, trg_);
    if (!bump_gate_.empty()) {
        std::string dims;
        for (const auto& ch : bump_gate_)
            for (const auto& b : ch.bumps) {
                if (!dims.empty()) dims += ", ";
                dims += ch.ckey + "[" + std::to_string(b.index + 1) + "]";
            }
        HPOLog::instance().info(
            "bump-support per-candidate gate armed on " + dims +
            ": infeasible candidates (c + w/eta > RCUT) are rejected "
            "before training as graded band-D failures");
    }

    /* neighbours -------------------------------------------------------- */
    /* One-time build: sized for the largest cutoff any evaluation can
     * reach — stated RCUTMAX, OPTIM RCUT2B/RCUTMB upper bounds, and (knot
     * channels) OPTIM CGRID2B/CGRIDMB upper bounds. */
    const double rcutmax = detail::nn_safe_rcutmax(master_ctx_, trg_);

    /* validation set (optional: PC_OBSERVABLE/PC_LAMMPS-only runs) ------- */
    core::Context val_ctx;
    if (has_validation_) {
        val_ctx = core::Context(core::CoreUserConfigParser().parse(val_cfg));
        val_ctx_ = val_ctx;
        std::vector<std::string> files;
        if (val_ctx.contains("DBFILE")) files = val_ctx("DBFILE");
        trg_.resolve_validation_targets(files);
        val_stdb_.add(val_ctx);
        n_val_datasets_ = val_stdb_.dbidx.size() > 1 ? val_stdb_.dbidx.size() - 1 : 1;
    }
    if (has_train_data_) {
        train_stdb_.add(master_ctx_);

        /* Apply load-time transforms once on host (LSCALE, ESHIFT,
         * EFILTER/FFILTER, WDBFILE/WDBFILE_AUTO, EWEIGHT_TEMP,
         * ZERO_COM_FORCE) before any neighbour finding so descriptors see
         * the transformed dataset. The report is stored for later:
         * pot.tadah header generation and stamping LSCALE/ESHIFT into
         * each best_pot.tadah. */
        report_ = mlip::apply_load_transforms(train_stdb_, master_ctx_);
    }

    /* Apply LSCALE/ESHIFT to validation, and also forward EFILTER/FFILTER if
     * the user declared them in the --validation file itself. Training-only
     * knobs (filters from master_ctx_, WDBFILE, EWEIGHT_TEMP, ZERO_COM_FORCE)
     * must NOT touch validation references. Use the resolved values from
     * report_ so ESHIFT_ATOM / ESHIFT_DBATOM derivations on the training set
     * propagate consistently. Without training data report_ is empty, so
     * explicit ESHIFT values from the potential config are forwarded
     * instead. */
    if (has_validation_) {
        core::Context vt;
        if (master_ctx_.contains("LSCALE"))
            vt.add("LSCALE", master_ctx_.get<std::string>("LSCALE"));
        if (!report_.eshift.empty()) {
            std::vector<int> Z_sorted;
            Z_sorted.reserve(report_.eshift.size());
            for (const auto &p : report_.eshift) Z_sorted.push_back(p.first);
            std::sort(Z_sorted.begin(), Z_sorted.end());
            for (int Z : Z_sorted)
                vt.add("ESHIFT", core::to_string(report_.eshift.at(Z)));
        } else if (!has_train_data_ && master_ctx_.contains("ESHIFT")
                   && master_ctx_.size("ESHIFT") > 0) {
            std::vector<double> es(master_ctx_.size("ESHIFT"));
            master_ctx_.get<>("ESHIFT", es);
            for (double e : es) vt.add("ESHIFT", core::to_string(e));
        }
        detail::forward_val_filters(vt, val_ctx);
        const std::size_t n_val_before = val_stdb_.size();
        val_report_ = mlip::apply_load_transforms(val_stdb_, vt);
        core::Logger::instance().log(
            tadah::core::LogLevel::INFO,
            "Validation set: ", std::to_string(n_val_before),
            " configs loaded; dropped ",
            std::to_string(val_report_.n_dropped_energy), " by EFILTER, ",
            std::to_string(val_report_.n_dropped_force), " by FFILTER; ",
            std::to_string(val_stdb_.size()), " remain.");

        // Filters preserve dataset identity; an empty segment has no RMSE
        // and must never disappear or shift subsequent target assignments.
        if (stats_have_columns()) {
            for (std::size_t i = 0; i + 1 < val_stdb_.dbidx.size(); ++i)
                if (val_stdb_.dbidx[i] == val_stdb_.dbidx[i + 1])
                    throw std::runtime_error("Empty validation dataset " + std::to_string(i + 1)
                        + " (" + val_ctx.get<std::string>("DBFILE", i) + ") after loading/filtering");
        }
        nnf_.calc(val_stdb_, rcutmax);
    }
    if (has_train_data_) nnf_.calc(train_stdb_, rcutmax);

    /* elements check ---------------------------------------------------- */
    if (has_train_data_ && has_validation_) {
        auto tr_el = mlip::StructureDB::find_unique_elements(master_ctx_);
        for(auto e: mlip::StructureDB::find_unique_elements(val_ctx))
            if(!tr_el.count(e))
                throw std::runtime_error("Validation element missing in training");
    }

    /* all-fixed fast path ------------------------------------------------
     * When FIXINDEX pins every model weight the per-eval LS solve is
     * vacuous — the train stage can scatter the FIXWEIGHT values
     * directly and skip the design-matrix build. NORM must be off
     * (validate_hpo_setup rejects NORM+FIXINDEX; re-checked here since
     * tests construct HPO_Host directly). Classification failures fall
     * back to the ordinary train() path, which surfaces the error with
     * full context. */
    if (has_train_data_ && master_ctx_.contains("FIXINDEX")
        && !(master_ctx_.contains("NORM") && master_ctx_.get<bool>("NORM"))) {
        try {
            const std::size_t K = derive_weight_count(master_ctx_);
            all_weights_fixed_ =
                K > 0 && mlip::classify_weight_source(master_ctx_, K)
                             == mlip::WeightSource::AllFixedData;
        } catch (const std::exception&) {
            all_weights_fixed_ = false;
        }
        if (all_weights_fixed_) {
            core::Logger::instance().log(
                tadah::core::LogLevel::INFO,
                "FIXINDEX covers every model weight: per-eval LS solve and "
                "design-matrix build are skipped (weights provisioned from "
                "FIXWEIGHT).");
        }
    }

    /* param labels: ONE builder (HPOTargets::build_search_space), the same
     * optimap traversal that orders the optimiser vector, so params.tadah
     * columns, the journal and hpo_search_space.tsv agree by construction.
     * KEY[n] with n the OPTIM index (1-based; the pinned model weight for
     * FIXWEIGHT); single-value keys carry no index. */
    param_labels_ = trg_.spec.parameter_labels;
    {
        std::size_t n_opt = 0;
        for (const auto& kv : trg_.optimap)
            for (bool b : kv.second.opt) if (b) ++n_opt;
        if (param_labels_.size() != n_opt)
            throw std::runtime_error(
                "internal: " + std::to_string(param_labels_.size()) +
                " parameter label(s) for " + std::to_string(n_opt) +
                " OPTIM dimension(s)");
    }

    init_loss   (trg_.lossFile);
    init_outvar (trg_.outvarFile);
    init_params (trg_.paramsFile);
    init_stats  (trg_.statsFile);
    init_loss   (trg_.bestLossFile);
    init_outvar (trg_.bestOutvarFile);
    init_params (trg_.bestParamsFile);
    init_stats  (trg_.bestStatsFile);

}

/* ----------  helper: update context ----------------------------------- */
template<class P, class R>
void HPO_Host<P,R>::update_context(core::Context& ctx,
                                   const P& p) const
{
    for(auto& kv : trg_.optimap) ctx.remove(kv.first);

    std::size_t idx=0;
    for(const auto& kv : trg_.optimap){
        for(std::size_t i=0;i<kv.second.opt.size();++i){
            if(kv.second.opt[i]) ctx.add(kv.first, p(idx++));
            else                 ctx.add(kv.first, kv.second.config[i]);
        }
    }

    /* Knot channels: the effective cutoff follows the (possibly OPTIM'd)
     * largest knot — re-derive RCUT2B/RCUTMB so Trainer::reinit applies it
     * via fcut->set_rcut(RCUT*MAX). Overrides a user OPTIM RCUT* dimension
     * deliberately (warned once at startup). The prebuilt NN list already
     * covers every reachable cutoff (detail::nn_safe_rcutmax). */
    detail::apply_knot_rcut_derivation(ctx, knot2b_, knotmb_);
}

/* ----------  helper: train/val structure counts ----------------------- */
template<class P, class R>
std::size_t HPO_Host<P,R>::train_size() const
{
    return train_stdb_.size();
}

template<class P, class R>
std::size_t HPO_Host<P,R>::val_size() const { return val_stdb_.size(); }

/* ----------  helper: format params for log ---------------------------- */
template<class P, class R>
std::string HPO_Host<P,R>::format_params_str(const P& p) const
{
    std::ostringstream s;
    std::size_t idx = 0;
    bool first = true;
    /* One label convention everywhere: param_labels_ (KEY[n], n = OPTIM
     * index) is built from this same optimap traversal. */
    for (const auto& kv : trg_.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            if (!first) s << "  ";
            first = false;
            s << (idx < param_labels_.size() ? param_labels_[idx] : kv.first);
            s << "=" << std::scientific << std::setprecision(3) << p(idx);
            ++idx;
        }
    }
    return s.str();
}

/* ----------  helper: bound proximity hints for log -------------------- */
template<class P, class R>
std::string HPO_Host<P,R>::bound_hints(const P& p) const
{
    const auto& sp = spec_;
    if (sp.low.empty()) return {};
    constexpr double EDGE_FRAC = 0.02; // flag if within 2% of bound
    std::ostringstream hints;
    std::size_t idx = 0;
    /* p is REAL-space; the bounds are INTERNAL -- compare in one space. */
    const P u = sp.in(p);
    for (const auto& kv : trg_.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            if (idx >= sp.low.size()) break;
            const double v   = u(static_cast<long>(idx));
            const double lo  = sp.low[idx];
            const double hi  = sp.high[idx];
            const double rng = hi - lo;
            const std::string name =
                idx < param_labels_.size() ? param_labels_[idx] : kv.first;
            if (rng > 0) {
                if ((v - lo) / rng < EDGE_FRAC)
                    hints << "Tip: " << name << " is near its LOWER bound -- "
                          << "consider tightening or expanding the range. ";
                else if ((hi - v) / rng < EDGE_FRAC)
                    hints << "Tip: " << name << " is near its UPPER bound -- "
                          << "consider expanding the range. ";
            }
            ++idx;
        }
    }
    return hints.str();
}

/* ----------  helper: fill validation metrics -------------------------- */
template<class P, class R>
void HPO_Host<P,R>::fill_validation(Result& r,
                                    const mlip::Analytics& a) const
{
    auto fill = [](double& avg, std::vector<double>& per_ds,
                   const core::t_type& v, double scale)
    {
        avg = v.mean() * scale;
        per_ds.resize(v.size());
        for (std::size_t i = 0; i < v.size(); ++i)
            per_ds[i] = v[i] * scale;
    };
    if(trg_.use_emae)   fill(r.emae,   r.v_emae,   a.calc_e_mae(),   1000.0);
    if(trg_.use_ermse)  fill(r.ermse,  r.v_ermse,  a.calc_e_rmse(),  1000.0);
    if(trg_.use_errmse) fill(r.errmse, r.v_errmse, a.calc_e_rrmse(), 1000.0);
    if(trg_.use_ersq)   fill(r.ersq,   r.v_ersq,   a.calc_e_r_sq(),     1.0);
    if(trg_.use_frmse)  fill(r.frmse,  r.v_frmse,  a.calc_f_rmse(),     1.0);
    if(trg_.use_srmse)  fill(r.srmse,  r.v_srmse,  a.calc_s_rmse(),     1.0);
}

/* ----------  helper: run LAMMPS (serial + OpenMP) ---------------------
 *
 * Two-wave execution when any PC_LAMMPS line carries `--hotvar` (Stage 2.4):
 *
 *   WAVE 1    Run the hot scripts AND every cold script that references no
 *             produced hot-var, in ONE parallel batch. Each hot script reads
 *             its warmstart-cached value (or the user-declared `initial`,
 *             midpoint of [min, max] on first eval) as `${<name>}`, relaxes,
 *             exposes a fresh `<name>` value; the framework validates against
 *             [min, max], updates the cache, and collects (name -> fresh value)
 *             into `hot_values_for_iter`. An out-of-range value sets the
 *             eval-wide abort flag. The hot-INDEPENDENT cold scripts run
 *             alongside (no injection needed), filling cores that would
 *             otherwise idle behind the hot box/relax.
 *
 *   WAVE 2    Run the hot-DEPENDENT cold scripts. If wave 1 aborted, ALL cold
 *             outputs (independent ones included) are clamped to failscore.
 *             Otherwise every value collected in wave 1 is injected as
 *             `variable <name> string <value>` before each dependent cold
 *             script, so it reads `${<name>}` to get the freshly computed hot
 *             value from THIS eval.
 *
 * No --hotvar lines means wave 1 is all scripts (all classified independent)
 * and wave 2 is empty -- identical to the pre-hotvar single parallel wave. */
template<class P, class R>
void HPO_Host<P,R>::run_lammps(Result& r) const
{
  if(!use_lammps_) return;
  const auto& LS = trg_.LS;
  r.v_outvar.resize(LS.size());

  /* Stage 1a: per-script busy time (create+run ms) for the fan-out
   * utilisation diagnostic. Index i is written by exactly one fan-out
   * thread (the one that runs script i), so no locking is needed.
   * -1.0 = the script did not run (aborted/skipped). */
  std::vector<double> per_script_busy(LS.size(), -1.0);

  /* Stage 1b.1: stable longest-processing-time-first sort of a phase's script
   * indices by their EWMA cost estimate (descending). stable_sort + all-equal
   * costs (first eval, all zero) preserves the config order, so the first eval
   * is byte-identical to the pre-change behaviour. */
  auto lpt_sort = [&](std::vector<std::size_t>& idx) {
    /* Snapshot the EWMA costs BEFORE sorting. script_ewma_ms_ is a shared
     * HPO_Host member whose entries are updated concurrently by OTHER
     * replicas' fan-out worker threads (script_ewma_ms_[i].store(...) after
     * each script run). Reading the live atomics inside the comparator lets a
     * cost change mid-sort, so the comparator ceases to be the consistent
     * strict weak ordering std::stable_sort requires; its unguarded insertion
     * sort then runs off the FRONT of idx -- a heap-buffer-overflow (a silent
     * OOB write that detonates later as free(): invalid pointer). Sorting
     * against a fixed local snapshot keeps the ordering consistent for the
     * whole sort. (All-zero costs on the first eval still preserve config
     * order via stable_sort, so first-eval output stays byte-identical.) */
    std::vector<double> cost(LS.size());
    for (std::size_t i = 0; i < LS.size(); ++i)
      cost[i] = script_ewma_ms_[i].load(std::memory_order_relaxed);
    std::stable_sort(idx.begin(), idx.end(),
                     [&cost](std::size_t a, std::size_t b) {
                       return cost[a] > cost[b];
                     });
  };

  /* Partition scripts into hot (any --hotvar) and cold (none). The hot
   * set runs first so its produced values can flow into the cold set. */
  std::vector<std::size_t> hot_idx, cold_idx;
  hot_idx.reserve(LS.size());
  cold_idx.reserve(LS.size());
  for (std::size_t i = 0; i < LS.size(); ++i) {
    if (LS[i].is_failprobe) continue;   /* dispatched separately (front of
                                           wave 1, LPT-exempt) */
    if (LS[i].hotvars.empty()) cold_idx.push_back(i);
    else                       hot_idx.push_back(i);
  }

  /* Per-iter map of fresh hot values (varname -> value). Populated in
   * phase A under hot_mtx_, then read-only during phase B. */
  std::unordered_map<std::string, double> hot_values_for_iter;
  std::mutex hot_mtx;

  /* Run-LOCAL abort flag (replaces the global warmstart::aborting_iter). An
   * out-of-range hot var or a crashed hot script raises it; phase B (and any
   * still-dispatching phase-A tasks) see it and clamp to failscore. Local to
   * this run_lammps call => per-eval, so a concurrent eval's abort can neither
   * leak into nor race this one: the fan-out order cannot change the result. */
  std::atomic<bool> eval_aborted{false};
  /* Separate run-local latch so the rate-limited "rejecting" warning fires at
   * most once per eval, decoupled from eval_aborted (a crashed script raises
   * eval_aborted without being a hot-var rejection). Replaces the global
   * warmstart::last_rejected_iter, which worked only because r.iter is unique. */
  std::atomic<bool> reject_logged{false};

  /* PC_ISOLATION: in PROCESS mode every pc_lammps task ships to a persistent
   * worker process (pc_worker_pool) instead of running in-process; the trial
   * potential's raw storage map is snapshotted ONCE per eval here and each
   * channel forwards it as a DELTA vs its last shipped map (FULL on first
   * contact/respawn). THREAD mode: pool never constructed, map unused. */
  const bool pc_process_mode = pc_worker_pool_enabled();
  PcPotMap pc_pot_map;
  if (pc_process_mode) pc_pot_map = r.pot_ctx.storage_snapshot();

  /* Hot-var / fail-var NAME tables (one vector<string> per sim, declaration
   * order) for the extracted execution core, which reads raw strings by name
   * and leaves all parsing/validation here. Built once per eval. */
  std::vector<std::vector<std::string>> hotvar_names(LS.size());
  std::vector<std::vector<std::string>> failvar_names(LS.size());
  for (std::size_t i = 0; i < LS.size(); ++i) {
    hotvar_names[i].reserve(LS[i].hotvars.size());
    for (const auto& hv : LS[i].hotvars) hotvar_names[i].push_back(hv.name);
    failvar_names[i].reserve(LS[i].failvars.size());
    for (const auto& fv : LS[i].failvars) failvar_names[i].push_back(fv.name);
  }

  /* Graded-FAILSCORE (FAILSCORE_GRADING ON): per-eval failure classification.
   * Fan-out threads record failures here (worst band wins); the compact
   * snapshot lands in r.fail at the end of run_lammps and sims_loss turns it
   * into the graded loss. OFF => nothing is recorded and every site below
   * behaves byte-identically to the legacy constant-FAILSCORE path. */
  const bool fg_on = trg_.FAILSCORE_GRADING;
  PcFailState fail_state;

  /* Parent-side post-processing of one executed script task: failscore
   * formatting, abort policy, hot-var validation + warmstart, and the
   * busy/EWMA/CSV bookkeeping. Everything that is NOT LAMMPS execution.
   * (PC_ISOLATION split: execute_script_task runs identically in-process or
   * inside a worker process; this function is always parent-side.) */
  auto apply_task_result = [&](std::size_t i, int tid,
                               const PcTaskResult& tres) -> bool {
    const auto& lsim = LS[i];

    if (tres.status == PcTaskStatus::CREATE_FAIL) {
      /* Pre-extraction semantics preserved exactly: an instance-create
       * failure failscores THIS script only -- it does NOT raise
       * eval_aborted and records no busy/EWMA/CSV entry (per_script_busy[i]
       * stays -1 = "did not run"); the create is retried on the next task.
       * Grading: INFRASTRUCTURE failure -- deliberately ungraded (constant
       * top of band C); grading it would chase cluster noise, not theta. */
      if (fg_on) fail_state.record_infra();
      for (const auto& n : lsim.n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      }
      return false;
    }

    bool fail = (tres.status != PcTaskStatus::OK);

    /* D5: never index a wire-supplied vector on trust.
     *
     * The outvar and hot-var loops below are bounded by the HOST's spec
     * (lsim.n_outvar.size(), lsim.hotvars.size()) but read from vectors the
     * WORKER produced. Under PC_ISOLATION PROCESS -- the default -- those
     * arrive decoded from a socket with no arity check on either side, so a
     * short vector is an out-of-bounds operator[]: plain undefined
     * behaviour, and specifically NOT caught by the try/catch around the
     * hot-var block, which only guards std::stod. The arities match by
     * construction today (both sides build from the same spec table), so
     * this can only fire on frame corruption or a version-skewed worker
     * binary -- exactly the cases where UB is least acceptable and, because
     * error_msg was discarded at the socket boundary, least visible.
     *
     * One check covers both loops: marking the task failed sends the outvar
     * loop down its failscore branch and skips the hot-var block entirely
     * (it is already inside `if (!fail)`). Deliberately NOT done by
     * shortening the loops to the received length -- that would silently
     * skip hot-var range validation and let the cold phase run against a
     * stale warm-start value, turning a detectable protocol fault into a
     * wrong-but-finite loss. record_infra() is the same band-C grading the
     * CREATE_FAIL path uses: infrastructure, not physics. */
    if (!fail &&
        (tres.outvar_values.size() < lsim.n_outvar.size() ||
         tres.hotvar_values.size() < lsim.hotvars.size())) {
      fail = true;
      if (fg_on) fail_state.record_infra();
      HPOLog::instance().warning(
          "pc-worker returned a short result for " + lsim.script + ": "
          + std::to_string(tres.outvar_values.size()) + "/"
          + std::to_string(lsim.n_outvar.size()) + " outvars, "
          + std::to_string(tres.hotvar_values.size()) + "/"
          + std::to_string(lsim.hotvars.size())
          + " hotvars -- treating the task as failed (protocol desync)");
    }

    /* Graded-FAILSCORE Tier 2: a declared failvar > 0 means the sim
     * completed but vouches AGAINST itself -- classify band B, grade by
     * scale*value, and treat the sim as failed (outvars excluded from the
     * fit via the failscore fill below; a failed HOT sim also skips the
     * warmstart caching and aborts the cold phase, same as a rejection). */
    if (fg_on && !fail && !lsim.failvars.empty()) {
      for (std::size_t j = 0; j < lsim.failvars.size() &&
                              j < tres.failvar_values.size(); ++j) {
        const std::string& s = tres.failvar_values[j];
        if (s.empty()) continue;
        double v = 0.0;
        try { v = std::stod(s); } catch (...) { continue; }
        if (std::isfinite(v) && v > 0.0) {
          fail_state.record(FailBand::B, lsim.failvars[j].scale * v);
          fail = true;
        }
      }
    }

    /* Graded-FAILSCORE Tier 1: classify hard task failures. */
    if (fg_on && tres.status != PcTaskStatus::OK) {
      switch (tres.status) {
        case PcTaskStatus::POT_BUILD_FAIL:
          fail_state.record(FailBand::D);           /* graded by S at assembly */
          break;
        case PcTaskStatus::KILLED:
          fail_state.record(FailBand::C);           /* graded by S at assembly */
          break;
        case PcTaskStatus::RUN_FAIL: {
          /* Instance alive: failvars (defined EARLY by contract) may still
           * carry the violation measure -> band B graded; else band B by
           * failed fraction. */
          bool graded = false;
          for (std::size_t j = 0; j < lsim.failvars.size() &&
                                  j < tres.failvar_values.size(); ++j) {
            const std::string& s = tres.failvar_values[j];
            if (s.empty()) continue;
            double v = 0.0;
            try { v = std::stod(s); } catch (...) { continue; }
            if (std::isfinite(v) && v > 0.0) {
              fail_state.record(FailBand::B, lsim.failvars[j].scale * v);
              graded = true;
            }
          }
          if (!graded)
            fail_state.record(FailBand::B,
                              1.0 / static_cast<double>(LS.size()));
          break;
        }
        default: break;
      }
    }

    /* D17: say WHY the task failed.
     *
     * error_msg is filled on every worker-side failure path, marshalled
     * across the socket, and was then read by nobody -- so a POT_BUILD_FAIL
     * or an in-worker exception produced a graded loss with its cause
     * discarded at the process boundary, leaving only a number. Only the
     * statuses whose text originates in the WORKER are surfaced here:
     * KILLED carries the parent's own reason, which kill_reap_locked has
     * already journaled, and repeating it would double every death.
     *
     * Rate-limited on powers of two: a systematically broken potential
     * fails identically on every script of every eval, and the first few
     * occurrences carry all the information. */
    if ((tres.status == PcTaskStatus::POT_BUILD_FAIL ||
         tres.status == PcTaskStatus::RUN_FAIL) &&
        !tres.error_msg.empty()) {
      static std::atomic<std::size_t> n_worker_err{0};
      const std::size_t n = n_worker_err.fetch_add(1,
                                                   std::memory_order_relaxed) + 1;
      if ((n & (n - 1)) == 0)
        HPOLog::instance().warning(
            "pc-worker task failed (" + lsim.script + ", occurrence "
            + std::to_string(n) + "): " + tres.error_msg);
    }

    for (std::size_t j = 0; j < lsim.n_outvar.size(); ++j) {
      if (fail) r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      else      r.v_outvar[i].emplace_back(tres.outvar_values[j]);
    }

    if (fail) {
      eval_aborted.store(true, std::memory_order_relaxed);
    }

    /* Hot-var validation + cache update. Only meaningful for hot
     * scripts (cold scripts have empty hotvars). */
    if (!fail) {
      for (std::size_t j = 0; j < lsim.hotvars.size(); ++j) {
        const auto& hv = lsim.hotvars[j];
        try {
          const std::string& s = tres.hotvar_values[j];
          if (s.empty()) continue;
          double v = 0.0;
          try { v = std::stod(s); }
          catch (...) { continue; }
          if (!std::isfinite(v) || v < hv.min || v > hv.max) {
            /* Out of range -- clamp this script's outputs and raise the
             * run-local abort so phase B is skipped.
             * Grading Tier 1 (band A): the normalized distance to the box is
             * the violation measure, so a proposal that moves the hot value
             * TOWARD [min,max] scores strictly better than one that moves it
             * away -- instead of both landing on the same constant. */
            if (fg_on) {
              double dist;
              if (!std::isfinite(v)) {
                dist = std::numeric_limits<double>::infinity();
              } else {
                const double w = hv.max - hv.min;
                dist = (v < hv.min ? (hv.min - v) : (v - hv.max)) / w;
              }
              fail_state.record(FailBand::A, dist);
            }
            for (auto& vs : r.v_outvar[i]) vs = std::to_string(lsim.failscore);
            eval_aborted.store(true, std::memory_order_relaxed);
            /* Log at most once per eval (run-local latch), rate-limited
             * across evals by the global reject_count. */
            const bool first_in_eval =
                !reject_logged.exchange(true, std::memory_order_relaxed);
            if (first_in_eval) {
              const auto rejs = warmstart::reject_count().fetch_add(
                  1, std::memory_order_relaxed) + 1;
              if (rejs <= 5 || (rejs & (rejs - 1)) == 0) {
                HPOLog::instance().warning(
                    std::string("eval ") + std::to_string(r.iter)
                    + " script " + lsim.script
                    + ": rejecting (" + hv.name + "=" + std::to_string(v)
                    + " outside [" + std::to_string(hv.min) + ", "
                    + std::to_string(hv.max) + "], total rejects="
                    + std::to_string(rejs) + ")");
              }
            }
            break; /* one out-of-range hot var is enough */
          }
          /* In-range -- cache for THIS CHAIN for next eval, record for phase B,
           * and stash for possible promotion into the seed map if this eval
           * turns out to be the new global best. */
          warmstart::set(r.chain_id, lsim.script, hv.name, v);
          {
            std::lock_guard<std::mutex> lg(hot_mtx);
            hot_values_for_iter[hv.name] = v;
            r.hot_produced.emplace_back(lsim.script, hv.name, v);
          }
        } catch (...) { /* variable not defined or unparseable -- ignore */ }
      }
    }

    /* Always record per-script busy time (a couple of clock reads, ~30 ns,
     * invisible vs ms-scale LAMMPS) so the fan-out diagnostic works without
     * TADAH_HPO_BENCH. The detailed CSV stays gated below.
     *
     * D10: a FAILED task that reports no time at all did not cost zero -- it
     * failed before anything could be measured (the worker-exception path
     * returns a default-constructed result). Feeding that 0 into the EWMA
     * would halve this script's estimated cost and, via the
     * longest-processing-time sort, dispatch the problem script LATER; it
     * would also drag fanout_min_ms to 0 and put "fastest 0 ms" in the
     * journal. Leave per_script_busy at -1 -- "did not run" -- exactly as
     * the CREATE_FAIL path above already does. Deadline kills DO carry a
     * measured elapsed time now (see PcWorkerPool::execute) and are
     * therefore recorded normally, which is the point: the script that hung
     * must sort FIRST. */
    if (fail && tres.create_ms + tres.run_ms <= 0.0) return true;
    per_script_busy[i] = tres.create_ms + tres.run_ms;
    /* Stage 1b.1: fold this script's cost into its EWMA (alpha=0.3) so the
     * next eval's LPT sort dispatches it earliest if it is slow. First
     * observation seeds directly. Relaxed RMW: a torn read across concurrent
     * PA/PT evals only nudges dispatch order, never a result value. */
    {
      constexpr double alpha = 0.3;
      const double prev = script_ewma_ms_[i].load(std::memory_order_relaxed);
      const double nextv = (prev <= 0.0)
          ? per_script_busy[i]
          : (1.0 - alpha) * prev + alpha * per_script_busy[i];
      script_ewma_ms_[i].store(nextv, std::memory_order_relaxed);
    }
    if (bench_detail::enabled()) {
      std::lock_guard<std::mutex> lg(bench_detail::mtx());
      static std::ofstream f("hpo_script_times.csv" + hpo_rank_suffix(),
                             std::ios::out | std::ios::trunc);
      static bool header_written = (f << "iter,thread,script,create_ms,run_ms\n", true);
      (void)header_written;
      f << r.iter << ',' << tid << ',' << lsim.script << ','
        << tres.create_ms << ',' << tres.run_ms << '\n';
      f.flush();
    }
    return true;
  };

  /* Generic per-script runner -- used by both phases. Returns true on a
   * successful LAMMPS run; false on script failure (in which case v_outvar is
   * filled with failscore). `inject_lines` are pre-formatted LAMMPS commands
   * (format_hot_inject) executed once after the instance is ready and before
   * run_script. The LAMMPS-touching core is execute_script_task
   * (pc_task_exec.h) -- shared verbatim with the PC_ISOLATION worker-process
   * path so THREAD and PROCESS modes cannot diverge. */
  auto run_one = [&](std::size_t i, int tid,
                     const std::vector<std::string>& inject_lines) -> bool {
    const auto& lsim = LS[i];

    /* PROCESS mode: proxy the task to worker-process channel `tid` (1:1 with
     * the ScriptExecutor worker id). The worker runs the SAME
     * execute_script_task below; all post-processing stays here. */
    if (pc_process_mode) {
      const PcTaskResult tres = pc_worker_pool().execute(
          tid, static_cast<std::uint32_t>(i), r.iter, pc_pot_map,
          inject_lines, lsim.maxtime);
      return apply_task_result(i, tid, tres);
    }

    /* -log/-screen get a .t<tid> suffix so concurrent fan-out workers don't
     * clobber the same file; tid is the ScriptExecutor worker id, stable for
     * the whole run, so the per-slot caches warm once per worker. */
    const std::vector<std::string> invar =
        make_worker_invar(lsim.invar, hpo_rank_suffix(), tid);

    /* Per-slot persistent LAMMPS/potential caches. Same lifetime as the
     * pre-extraction tls_lmp_cache/tls_lt/tls_lt_iter thread-locals: one per
     * (worker thread, HPO_Host instantiation), warm across evals. */
    static thread_local PcSlotState<R> slot_state;

    const PcTaskResult tres = execute_script_task<R>(
        lsim.script, lsim.script_lines, invar, lsim.n_outvar, hotvar_names[i],
        failvar_names[i], r.pot_ctx, r.iter, inject_lines, slot_state);
    return apply_task_result(i, tid, tres);
  };

  /* --- Stage 2.4: classify cold scripts by hot-var dependency ------ */
  /* A cold script that references NO produced hot-var (`${name}` or `v_name`
   * for any declared name) needs no phase-A output, so it can run CONCURRENTLY
   * with the hot scripts instead of idling behind the coarse phase barrier.
   * This fills the cores that would otherwise sit idle while a lone hot
   * box/relax runs (the small-hot-set case, e.g. 1 hot of 31). Cold scripts
   * that DO reference a hot-var stay in wave 2, after the values are known.
   *
   * Determinism: an "independent" cold script does not reference the hot var,
   * so the value it would have been injected is DEAD -- running it in wave 1
   * with no injection produces exactly the same r.v_outvar[i] as running it in
   * wave 2 with the (unused) injection. Classification is conservative: any
   * textual reference marks the script dependent, so a mis-classification can
   * only demote an independent script to wave 2 (a missed speed-up), never
   * feed a cold script a stale value. */
  std::vector<std::size_t> cold_indep_idx, cold_dep_idx;
  cold_indep_idx.reserve(cold_idx.size());
  cold_dep_idx.reserve(cold_idx.size());
  {
    std::vector<std::string> ref_brace, ref_v;   /* ${name} and v_name per hot var */
    for (auto hi : hot_idx)
      for (const auto& hv : LS[hi].hotvars) {
        ref_brace.push_back("${" + hv.name + "}");
        ref_v.push_back("v_" + hv.name);
      }
    auto references_hot = [&](std::size_t i) -> bool {
      for (const auto& line : LS[i].script_lines)
        for (std::size_t p = 0; p < ref_brace.size(); ++p)
          if (line.find(ref_brace[p]) != std::string::npos ||
              line.find(ref_v[p])     != std::string::npos)
            return true;
      return false;
    };
    for (auto i : cold_idx) {
      if (ref_brace.empty() || !references_hot(i)) cold_indep_idx.push_back(i);
      else                                          cold_dep_idx.push_back(i);
    }
  }

  /* Uniform clamp of one cold script to failscore, overwriting any value it may
   * have already produced in wave 1 (clear first) so the recorded result on an
   * abort is exactly "all cold failscored", identical to the pre-merge
   * behaviour and independent of execution order. */
  auto clamp_cold = [&](std::size_t i) {
    r.v_outvar[i].clear();
    for (const auto& n : LS[i].n_outvar) {
      (void)n;
      r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
    }
  };

  /* --- WAVE 1: hot scripts + hot-INDEPENDENT cold scripts ---------- */
  /* Submitted as ONE shared-pool batch, so idle cores during a long box/relax
   * are filled by independent cold work. Hot scripts are dispatched first (LPT
   * within each group) so the dependent cold wave that waits on them starts as
   * early as possible. Order steers dispatch only; each script writes its own
   * r.v_outvar[i], so results are unchanged (verified vs the pre-merge binary).
   * The pool serves EVERY replica (no OMP nesting; train/predict share it too). */
  lpt_sort(hot_idx);
  lpt_sort(cold_indep_idx);
  std::vector<std::size_t> wave1;
  wave1.reserve(hot_idx.size() + cold_indep_idx.size() + 1);
  /* PC_FAILPROBE runs at the FRONT of wave 1 (LPT-exempt): it is the
   * cheapest task, so it returns first and its S(theta) is in hand before
   * (almost) anything else can fail; good potentials pay zero latency
   * because the probe just absorbs into fan-out slack. */
  const int probe_idx = trg_.failprobe_idx;
  if (probe_idx >= 0) wave1.push_back(static_cast<std::size_t>(probe_idx));
  wave1.insert(wave1.end(), hot_idx.begin(), hot_idx.end());
  wave1.insert(wave1.end(), cold_indep_idx.begin(), cold_indep_idx.end());

  script_executor().run_batch(wave1.size(), [&](std::size_t k, int wid) {
    const std::size_t i = wave1[k];
    if (probe_idx >= 0 && i == static_cast<std::size_t>(probe_idx)) {
      /* The probe runs even if another wave-1 task already aborted the eval:
       * its S(theta) grades bands C/D of whatever failure occurred. */
      run_one(i, wid, {});
      double S = std::numeric_limits<double>::quiet_NaN();
      if (!r.v_outvar[i].empty()) {
        try { S = std::stod(r.v_outvar[i].front()); } catch (...) {}
      }
      /* S >= failscore means the probe task itself failed (its outvars
       * were clamped to failscore strings) -- that is "probe garbage",
       * not a huge-but-real S: fall through to the band-D branch below
       * so a broken probe consistently reads as "potential sick". */
      if (std::isfinite(S) && S >= 0.0 && S < LS[i].failscore) {
        fail_state.set_S(S);
        if (fg_on && S > trg_.failprobe_gate) {
          /* GATE: potential too sick to be worth the fan-out. Classify the
           * WHOLE eval band D (graded by S), cancel everything still queued
           * (the eval_aborted check below), and -- PROCESS mode -- SIGKILL
           * this eval's in-flight sims (slot-targeted; other replicas'
           * tasks on the shared pool are untouched). Results of tasks that
           * already completed are clamped at assembly regardless, so the
           * outcome does not depend on the probe-vs-sims race. */
          fail_state.set_gated();
          eval_aborted.store(true, std::memory_order_relaxed);
          if (pc_process_mode) pc_worker_pool().kill_matching(r.iter);
          static std::atomic<std::size_t> gate_count{0};
          const auto n = gate_count.fetch_add(1, std::memory_order_relaxed) + 1;
          if (n <= 5 || (n & (n - 1)) == 0) {
            HPOLog::instance().warning(
                "eval " + std::to_string(r.iter) + " FAILPROBE gate: S="
                + std::to_string(S) + " > " + std::to_string(trg_.failprobe_gate)
                + " -- eval short-circuited (total gated=" + std::to_string(n)
                + ")");
          }
        }
      } else if (fg_on) {
        /* The probe itself failed or produced garbage -- by design it cannot,
         * so treat the potential as sick (band D, S unavailable = band top). */
        fail_state.record(FailBand::D);
      }
      return;
    }
    if (eval_aborted.load(std::memory_order_relaxed)) {
      for (const auto& n : LS[i].n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
      }
      return;
    }
    if (!LS[i].hotvars.empty()) {
      /* hot script: seed each declared hot var with its cached/initial value.
       * Seeds are resolved HERE (parent-side, warmstart state) and shipped as
       * pre-formatted inject lines -- identical commands to the old injector. */
      std::vector<std::string> inject;
      inject.reserve(LS[i].hotvars.size());
      for (const auto& hv : LS[i].hotvars) {
        double seed;
        if (auto cached = warmstart::get(r.chain_id, LS[i].script, hv.name)) {
          seed = *cached;
        } else if (hv.initial) {
          seed = *hv.initial;
        } else {
          seed = 0.5 * (hv.min + hv.max);
        }
        inject.push_back(format_hot_inject(hv.name, seed));
      }
      run_one(i, wid, inject);
    } else {
      /* hot-independent cold script: references no hot var => no injection. */
      run_one(i, wid, {});
    }
  });

  /* Graded-FAILSCORE: persist this eval's classification (compact, copyable)
   * for sims_loss to turn into the graded loss. Called on BOTH exits of
   * run_lammps -- the abort gate below and the normal end. An aborted eval
   * is precisely the one whose classification matters most: an OOR hot-var
   * reject (band A) or a failed wave script (bands B/C/D) raises
   * eval_aborted, so a merge only at the normal exit would drop every
   * graded band. band stays 0 on feasible evals and always when
   * FAILSCORE_GRADING is OFF. */
  auto persist_fail_info = [&]() {
    if (!fg_on) return;
    const PcFailInfo snap = fail_state.snapshot();
    if (snap.band > r.fail.band ||
        (snap.band == r.fail.band && snap.band != 0)) {
      const double keep_vD = r.fail.vD;   /* set by the skip-caps path */
      r.fail = snap;
      if (r.fail.vD < 0.0) r.fail.vD = keep_vD;
    } else if (snap.S >= 0.0 && r.fail.S < 0.0) {
      r.fail.S = snap.S;                  /* S is useful even band-unchanged */
    }
  };

  /* --- Between waves: abort gate + snapshot ------------------------ */
  const bool aborted = eval_aborted.load(std::memory_order_relaxed);

  if (aborted) {
    /* An out-of-range hot var or a crashed wave-1 script => unphysical
     * potential. Clamp EVERY cold script to failscore (incl. independents that
     * already ran in wave 1) and skip wave 2. Recorded result is identical to
     * the pre-merge behaviour: all cold failscored, deterministically. */
    for (auto i : cold_indep_idx) clamp_cold(i);
    for (auto i : cold_dep_idx)   clamp_cold(i);
    persist_fail_info();
    return;
  }

  /* Snapshot the per-iter hot-values map before the dependent cold wave --
   * wave 1 is done, no more writes happen, so cold tasks read it lock-free. */
  std::unordered_map<std::string, double> hot_snapshot;
  {
    std::lock_guard<std::mutex> lg(hot_mtx);
    hot_snapshot = hot_values_for_iter;
  }

  /* --- WAVE 2: hot-DEPENDENT cold scripts -------------------------- */
  /* Inject every fresh hot value from wave 1; cold scripts read them as
   * `${<name>}`. The command list is identical for every wave-2 task, so it
   * is built once here (same commands the old per-task injector emitted). */
  std::vector<std::string> wave2_inject;
  wave2_inject.reserve(hot_snapshot.size());
  for (const auto& kv : hot_snapshot)
    wave2_inject.push_back(format_hot_inject(kv.first, kv.second));

  lpt_sort(cold_dep_idx);   /* longest-first + dynamic (see wave 1) */
  script_executor().run_batch(cold_dep_idx.size(), [&](std::size_t k, int wid) {
    const std::size_t i = cold_dep_idx[k];
    if (eval_aborted.load(std::memory_order_relaxed)) {
      for (const auto& n : LS[i].n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
      }
      return;
    }
    run_one(i, wid, wave2_inject);
  });

  /* Stage 1a: aggregate per-script busy time into the eval's fan-out
   * utilisation fields (consumed once by HPOLog::record_fanout). Runs
   * single-threaded here -- both phases have joined. */
  {
    double busy = 0.0, smin = std::numeric_limits<double>::max(), smax = 0.0;
    std::size_t nsc = 0, worst = 0;
    for (std::size_t i = 0; i < per_script_busy.size(); ++i) {
      const double b = per_script_busy[i];
      if (b < 0.0) continue;                 /* did not run */
      busy += b; ++nsc;
      if (b < smin) smin = b;
      if (b > smax) { smax = b; worst = i; }
    }
    r.fanout_busy_ms  = busy;
    r.fanout_nscripts = nsc;
    r.fanout_min_ms   = nsc ? smin : 0.0;
    r.fanout_max_ms   = smax;
    r.fanout_slowest  = nsc ? LS[worst].script : std::string();
  }

  /* Graded-FAILSCORE: normal-exit persist (see persist_fail_info above). */
  persist_fail_info();
}

template<class P, class R>
void HPO_Host<P,R>::fit_loss(Result& r) const
{
  auto add_global = [&](double& sink, double y, double ref, double w,
                        std::vector<double>& v_sink, const std::vector<double>& v_y)
  {
    sink = trg_.globalLoss.loss(y, ref, w);
    r.loss_sum += sink;
    ++r.loss_count;
    v_sink.resize(v_y.size());
    for (std::size_t i = 0; i < v_y.size(); ++i)
      v_sink[i] = trg_.globalLoss.loss(v_y[i], ref, w);
  };
  if (trg_.use_emae)   add_global(r.emaeLoss,   r.emae,   trg_.emae,   trg_.w_emae,   r.v_emaeLoss,   r.v_emae);
  if (trg_.use_ermse) {
    r.ermseLoss = trg_.ermse_targets.loss(trg_.globalLoss, r.ermse, r.v_ermse, r.v_ermseLoss);
    r.loss_sum += r.ermseLoss;
    ++r.loss_count;
  }
  if (trg_.use_errmse) add_global(r.errmseLoss, r.errmse, trg_.errmse, trg_.w_errmse, r.v_errmseLoss, r.v_errmse);
  if (trg_.use_ersq)   add_global(r.ersqLoss,   r.ersq,   trg_.ersq,   trg_.w_ersq,   r.v_ersqLoss,   r.v_ersq);
  if (trg_.use_frmse) {
    r.frmseLoss = trg_.frmse_targets.loss(trg_.globalLoss, r.frmse, r.v_frmse, r.v_frmseLoss);
    r.loss_sum += r.frmseLoss;
    ++r.loss_count;
  }
  if (trg_.use_srmse)  add_global(r.srmseLoss,  r.srmse,  trg_.srmse,  trg_.w_srmse,  r.v_srmseLoss,  r.v_srmse);

  r.loss = r.loss_count ? r.loss_sum / static_cast<double>(r.loss_count) : 0.0;
  if (!std::isfinite(r.loss) || r.loss > trg_.FAIL_SCORE) {
    if (!std::isfinite(r.loss)) r.was_nan_clamped = true;
    r.loss = trg_.FAIL_SCORE;
  }
}
template<class P, class R>
void HPO_Host<P,R>::sims_loss(Result& r) const
{
  r.v_loss.resize(trg_.LS.size());
  for (std::size_t i = 0; i < r.v_outvar.size(); ++i) {
    const auto& lsim = trg_.LS[i];
    r.v_loss[i].resize(lsim.n_outvar.size());

    for (std::size_t k = 0; k < lsim.i_varloss.size(); ++k) {
      auto j = lsim.i_varloss[k];
      double v;
      double w = lsim.w_varloss[k];
      double l;
      // LAMMPS variable retrievals can be empty when the script bailed
      // before computing them (NaN potential, undefined variable, ...).
      // std::stod("") throws std::invalid_argument; if that escapes the
      // user objective, NLopt's C++ wrapper translates it to
      // NLOPT_INVALID_ARGS and aborts the WHOLE optimisation. Treat
      // unparseable as a failed component contributing FAIL_SCORE.
      // A short vector (a task whose result was never applied) is the
      // same failure class -- guard it here rather than reading past
      // the end of v_outvar[i].
      if (j < r.v_outvar[i].size() && try_parse_double(r.v_outvar[i][j], v)) {
        l = lsim.lossFn[k].loss(v, lsim.t_varloss[k], w);
      } else {
        l = trg_.FAIL_SCORE;
        /* Graded-FAILSCORE: an unparseable/empty component is a completed-
         * but-failed sim output -> band B, graded by the failed fraction. */
        if (trg_.FAILSCORE_GRADING) {
          if (r.fail.band < static_cast<int>(FailBand::B))
            r.fail.band = static_cast<int>(FailBand::B);
          r.fail.vB += 1.0 / static_cast<double>(trg_.LS.size());
        }
      }
      r.v_loss[i][j] = l;
      r.loss_sum += l;
      ++r.loss_count;
    }
  }

  r.loss = r.loss_count ? r.loss_sum / static_cast<double>(r.loss_count) : 0.0;
  if (!std::isfinite(r.loss) || r.loss > trg_.FAIL_SCORE) {
    if (!std::isfinite(r.loss)) r.was_nan_clamped = true;
    r.loss = trg_.FAIL_SCORE;
  }

  /* Graded-FAILSCORE assembly: a classified failure replaces the constant
   * (or the launder-prone partial mean) with the banded, monotone L_fail in
   * [0.999*FS, FS) -- every existing failure filter still sees "failure"
   * (>= 0.999*FS), any feasible loss still wins, and moves toward
   * feasibility are strictly downhill. Feasible evals (band 0) and
   * FAILSCORE_GRADING OFF are byte-identical to the legacy path above. */
  if (trg_.FAILSCORE_GRADING && r.fail.band != 0) {
    r.loss = graded_eval_loss(r.fail, trg_.FAIL_SCORE, trg_.failprobe_s0);
  }
}

namespace detail {

/* RAII guard that temporarily sets FORCE=true on a Context and
 * restores the prior value on scope exit. Used to bracket observable
 * evaluations: NativeRelaxer + IFT adjoint paths consume fd_ij
 * arrays that dc.calc only fills when FORCE=true, even when the
 * training fit itself doesn't use forces (force-free dataset). The
 * trainer's context is per-thread (TLS cache), so mutating it during
 * the host's eval/gradient call is safe. */
struct ObsForceFlagGuard {
    tadah::core::Context& ctx;
    bool                  had_key;
    bool                  prev_val;
    ObsForceFlagGuard(tadah::core::Context& c) : ctx(c), had_key(false), prev_val(false) {
        try {
            prev_val = ctx.template get<bool>("FORCE");
            had_key = true;
        } catch (...) {
            had_key = false;
        }
        if (!had_key || !prev_val) {
            ctx.remove("FORCE");
            ctx.add("FORCE", "true");
        }
    }
    ~ObsForceFlagGuard() {
        if (had_key && !prev_val) {
            ctx.remove("FORCE");
            ctx.add("FORCE", "false");
        } else if (!had_key) {
            ctx.remove("FORCE");
        }
        /* If had_key && prev_val: FORCE was already true; nothing to
         * restore. */
    }
};

}  // namespace detail

template<class P, class R>
void HPO_Host<P,R>::obs_loss(Result& r) const
{
  if (trg_.observables.empty()) return;

  r.v_obs_value.resize(trg_.observables.size());
  r.v_obs_loss.resize(trg_.observables.size());

  auto& cache = tls_trainer_cache_;

  const bool have_trainer = cache.trainer && cache.trainer->model;

  /* Flip FORCE=true for the duration of all observable evaluations.
   * dc.calc reads FORCE from the trainer's context (which dc holds
   * by reference); NativeRelaxer's atom-force path requires fd_ij to
   * be populated. Restored on scope exit. */
  std::unique_ptr<detail::ObsForceFlagGuard> force_guard;
  if (have_trainer) {
    force_guard.reset(new detail::ObsForceFlagGuard(cache.trainer->context));
  }

  for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
    double v = std::numeric_limits<double>::quiet_NaN();
    double l = trg_.FAIL_SCORE;

    if (have_trainer && trg_.observables[k]) {
      auto& obs = *trg_.observables[k];
      try {
        ObservableEvalResult res = obs.evaluate(
            *cache.trainer->model,
            cache.trainer->dc,
            cache.trainer->context);
        v = res.value;
        if (std::isfinite(v)) {
          l = obs.lossfn().loss(v, obs.target(), obs.weight());
          if (!std::isfinite(l)) l = trg_.FAIL_SCORE;
        }
      } catch (const std::exception&) {
        /* Relaxer divergence / NaN / model mismatch -- record as a
         * failed component contributing FAIL_SCORE so the optimiser
         * walks away from this theta. */
        v = std::numeric_limits<double>::quiet_NaN();
        l = trg_.FAIL_SCORE;
      }
    }

    r.v_obs_value[k] = v;
    r.v_obs_loss[k]  = l;
    r.loss_sum      += l;
    ++r.loss_count;
  }

  r.loss = r.loss_count ? r.loss_sum / static_cast<double>(r.loss_count) : 0.0;
  if (!std::isfinite(r.loss) || r.loss > trg_.FAIL_SCORE) {
    if (!std::isfinite(r.loss)) r.was_nan_clamped = true;
    r.loss = trg_.FAIL_SCORE;
  }
}

/* ---------- synchronous crash-safe best potential ------------------- */
template<class P, class R>
void HPO_Host<P,R>::persist_durable_best(const Result& r) const
{
    try {
        std::ostringstream payload;
        const mlip::StructureDB& train_ref = train_stdb_;
        PotHeaderWriter::write_best_header(
            payload, r, trg_, master_ctx_, val_ctx_, train_ref, val_stdb_,
            param_labels_, report_, cw_, prec_);
        PotHeaderWriter::write_candidate_hot(payload, r);
        payload << std::setprecision(
            r.pot_ctx.template get<std::size_t>("NUMERIC")) << r.pot_ctx;
        if (!payload.good())
            throw durable_best::Error("failed to render best-potential payload");

        const durable_best::Key key{r.loss, r.source_rank, r.chain_id, r.iter};
        const std::string bytes = payload.str();

        /* Canonical first: if SIGKILL lands between the two commits, the
         * globally useful artifact still contains the improvement. */
        (void)durable_best::publish_if_better(
            trg_.canonicalBestPotFile, key, bytes);
        if (trg_.localBestPotFile != trg_.canonicalBestPotFile) {
            /* The canonical file is the correctness boundary. A rank-local
             * mirror is useful diagnostics/history, but a pre-existing better
             * local key (or a local I/O failure) must not undo or invalidate an
             * already durable canonical commit. */
            try {
                (void)durable_best::publish_if_better(
                    trg_.localBestPotFile, key, bytes);
            } catch (const std::exception& e) {
                try {
                    HPOLog::instance().warning(
                        std::string("rank-local best-potential mirror failed: ") +
                        e.what());
                } catch (...) {}
            } catch (...) {
                try {
                    HPOLog::instance().warning(
                        "rank-local best-potential mirror failed: unknown exception");
                } catch (...) {}
            }
        }
    } catch (const durable_best::Error&) {
        throw;
    } catch (const std::exception& e) {
        throw durable_best::Error(
            std::string("cannot persist completed best candidate: ") + e.what());
    } catch (...) {
        throw durable_best::Error(
            "cannot persist completed best candidate: unknown exception");
    }
}

/* ----------  operator()  (thread-safe) ------------------------------- */
template<class P, class R>
double HPO_Host<P,R>::operator()(const P& p) const
{
  /* Cooperative shutdown: SIGINT/SIGTERM/SIGUSR2 raised the stop flag.
   * Throw before consuming an iter slot. The exception unwinds every
   * optimiser and strategy to drive_run() (run_driver.h), which flushes
   * the result files, writes the SHUTDOWN trailer and returns 128 + the
   * signal number. SignalHandler::install() is wired in hpo_run(). */
  if (core::SignalHandler::stop_requested()) {
    throw core::ShutdownRequested{};
  }

  Result r;
  r.iter   = ++g_iter_;
    /* Which chain (PT replica / PA walker) is this? Bridged from the driver
     * via a thread_local; keys the per-chain warm-start cache so results are
     * independent of fan-out execution order. 0 for serial / NLOPT / dlib. */
    r.chain_id = warmstart::current_chain();
    r.source_rank = hpo_rank();
    r.params = p;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_train_start = t0, t_train_end = t0, t_predict_end = t0,
         t_lammps_end = t0, t_sims_loss_end = t0, t_submit_end = t0;

    /* Per-stage watchdog logger -- closes over r.iter so the post-hoc
     * leave() warning includes the iter that was running. */
    auto _stage_log = [](const std::string& msg) {
        HPOLog::instance().warning(msg);
    };

    /* D3: every evaluation that consumed an iter slot must reach the
     * journal's census, including the ones that leave through a catch.
     *
     * record_eval() used to be called only on the success path, from inside
     * the try below. n_eval_ is what drives the milestone SUMMARY schedule,
     * so a run whose evaluations all took an exception path stopped
     * producing SUMMARY lines entirely -- it read as FROZEN rather than as
     * FAILING, which is the worst possible presentation of a failure. Worse,
     * "NaN rate" stayed pinned at whatever it was, because n_nan_ is
     * incremented in the same place.
     *
     * An RAII census closes every exit at once, including the three catches
     * that rethrow, without duplicating the call five times. `census_done`
     * is set by the normal path so the success case still records exactly
     * once, with its own precise wall time. */
    struct EvalCensus {
        std::size_t iter;
        std::chrono::high_resolution_clock::time_point t0;
        bool  nan  = false;
        bool  done = false;
        ~EvalCensus() {
            if (done) return;
            const double ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count());
            HPOLog::instance().record_eval(iter, ms, nan);
        }
    } _census{r.iter, t0, false, false};

    /* Bug 5: wrap the entire eval pipeline in try/catch.
     *
     * Without this guard, a throw between `r.iter = ++g_iter_` above
     * and the eventual `submit_result(std::move(r), ...)` below leaves
     * the writer's `next_flush_` permanently stuck at the missing
     * iter -- every subsequent eval enqueues fine but the writer never
     * advances past the gap, freezing loss.tadah / params.tadah / etc
     * while hpo_run.log keeps progressing.
     *
     * Recovery: catch any std::exception (or unknown throw), submit a
     * synthetic FAIL_SCORE Result for this iter so the writer queue
     * stays contiguous, and return FAIL_SCORE to the optimiser. The
     * synthetic Result is flagged with was_nan_clamped = true so
     * downstream consumers can identify it as recovery output rather
     * than a genuine high-loss point.
     *
     * Excluded from the catch:
     *  - core::ShutdownRequested -- must propagate to hpo_run() so the
     *    run shuts down cleanly (result files flushed, journal closed).
     *  - StagnationStop -- must propagate to RestartPolicy. */
    try {

    /* Bump-support per-candidate gate (BUMPGATE CANDIDATE): an infeasible
     * (c, eta[, rcut]) combination would be a hard-truncated bump -- an
     * energy/force STEP at the cutoff -- so it is rejected BEFORE any
     * training work with a graded band-D loss (the pre-LAMMPS skip-caps
     * idiom below). The table is empty unless startup armed it, so
     * feasible-only searches pay a single branch. */
    if (!bump_gate_.empty()) {
      const double v = detail::bump_gate_violation(bump_gate_, p);
      if (v > 0.0) {
        r.fail.band = std::max(r.fail.band, static_cast<int>(FailBand::D));
        r.fail.vD = v;
        r.loss = trg_.FAILSCORE_GRADING
                     ? graded_eval_loss(r.fail, trg_.FAIL_SCORE,
                                        trg_.failprobe_s0)
                     : trg_.FAIL_SCORE;
        /* Rate-limited journal note: first 5, then powers of two --
         * parity with the bad-pot skip counter. */
        static std::atomic<std::size_t> n_rejects{0};
        const auto k = n_rejects.fetch_add(1, std::memory_order_relaxed) + 1;
        if (k <= 5 || (k & (k - 1)) == 0) {
            HPOLog::instance().info(
                "eval " + std::to_string(r.iter) + ": bump-support gate "
                "reject (c + w/eta past the cutoff, violation=" +
                std::to_string(v) + ", graded band D); total rejects=" +
                std::to_string(k));
        }
        /* The trainer cache still holds the previous feasible theta; the
         * next analytical_gradient() must not read it as "current x". */
        tls_trainer_theta_stale_ = true;
        /* Stagnation check (parity with post_lammps_finalize): an
         * all-infeasible run must still trip StagnationStop -- its catch
         * submits r with the real graded loss. */
        {
          const std::size_t threshold =
              stagnation_threshold_.load(std::memory_order_relaxed);
          /* A probe phase is deliberately non-improving; counting it would
           * trip the threshold and fire a restart mid-calibration. */
          if (threshold > 0 && !probe::active()) {
            const std::size_t last =
                iter_at_last_best_.load(std::memory_order_relaxed);
            if (r.iter > last && r.iter - last >= threshold) {
              throw StagnationStop{};
            }
          }
        }
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        const std::size_t save_iter = r.iter;
        const double save_loss = r.loss;
        const double save_wall_ms = static_cast<double>(r.wall_ms.count());
        const int save_fail_band = r.fail.band;
        submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                      train_stdb_, val_stdb_, param_labels_, report_,
                      cw_, prec_);
        auto& log = HPOLog::instance();
        log.record_eval(save_iter, save_wall_ms, /*nan=*/false);
        _census.done = true;   /* counted here, with the measured wall */
        log.record_fail_band(save_fail_band);
        log.record_bump_gate_reject();
        return save_loss;
      }
    }

    /* build context ---------------------------------------------------- */
    core::Context ctx = master_ctx_;
    update_context(ctx, p);

    t_train_start = std::chrono::high_resolution_clock::now();
    /* train ------------------------------------------------------------ */
    stage_watchdog::enter(r.iter, "train");
    {
        auto& cache = tls_trainer_cache_;
        const bool force_rebuild = std::getenv("TADAH_HPO_FORCE_REBUILD") != nullptr;
        if (!cache.trainer || force_rebuild) {
            if (cache.trainer) cache.trainer.reset();
            cache.trainer.reset(new (cache.buf) mlip::Trainer(ctx, tls_ws_));
        } else {
            bool needs_rebuild = false;
            try {
                update_context(cache.trainer->context, p);
                needs_rebuild = cache.trainer->reinit();
            } catch (const std::exception& e) {
                // reinit threw -- log and fall back to a clean rebuild
                HPOLog::instance().warning(
                    std::string("reinit() threw (eval ") + std::to_string(r.iter)
                    + "): " + e.what() + " -- rebuilding Trainer");
                needs_rebuild = true;
            }
            if (needs_rebuild) {
                cache.trainer.reset();
                cache.trainer.reset(new (cache.buf) mlip::Trainer(ctx, tls_ws_));
            }
        }
        if (has_train_data_ && !all_weights_fixed_) {
            cache.trainer->train(train_stdb_);
        } else {
            /* No training DBFILE, or FIXINDEX pins every weight: the LS
             * solve is absent/vacuous, so scatter the FIXWEIGHT values
             * directly (update_context above already wrote the
             * optimiser's live values into the trainer context) and
             * skip the design-matrix build. */
            cache.trainer->provision_fixed_weights();
        }
    }
    mlip::Trainer& host = *tls_trainer_cache_.trainer;
    /* Training succeeded: the trainer cache reflects theta = p again. */
    tls_trainer_theta_stale_ = false;
    t_train_end = std::chrono::high_resolution_clock::now();
    stage_watchdog::leave(r.iter, "train", _stage_log);

    /* -- Per-iter health monitor (Phase 6) ---------------------------
     * Cheap O(P) scan of the trained weights, partitioned by the
     * design-matrix ColumnLayout. Off by default (singleton starts in
     * HealthLogMode::Off; hpo_run() flips it to summary unless
     * HEALTH_LOG=off). The aggregation cost is negligible against the
     * train+predict path that brackets it. */
    {
        auto& mon = global_iter_health_monitor();
        if (mon.mode() != HealthLogMode::Off) {
            const auto& w = host.model->get_weights();
            const auto& layout = host.dm.layout();
            if (!layout.blocks.empty()) {
                auto rec = compute_iter_health(r.iter, w, layout);
                mon.record(rec, layout);
            }
        }
    }

    /* validation ------------------------------------------------------- */
    stage_watchdog::enter(r.iter, "predict_validate");
    core::Context pot = host.model->get_param_file();
    /* Stamp LSCALE/ESHIFT into the per-eval pot context so any best_pot.tadah
     * derived from this eval round-trips through `tadah predict`. The trainer
     * itself doesn't know about load-time transforms, so the host injects them. */
    mlip::stamp_into_param_file(pot, report_);
    /* Strip runtime-only audit knobs before the pot context is handed to
     * LAMMPS (or written to disk). The LAMMPS-side PairTadah parser does
     * not know about AUDIT / HEALTH_LOG and refuses pair_coeff with an
     * "Invalid key" error; the partial init then segfaults inside
     * calc_rho. Removing them here is the single choke-point -- every
     * downstream consumer (best_pot.tadah, per-eval pots, LAMMPS scripts)
     * uses the same `pot` Context. */
    pot.remove("AUDIT");
    pot.remove("HEALTH_LOG");
    if (has_validation_) {
        auto pred = host.model->predict(pot, val_stdb_, host.dc);
        mlip::Analytics   a(val_stdb_, pred); // we do not need to recalc NN as each descriptor checks cutoffs internally
        fill_validation(r, a);
    }
    r.pot_ctx  = std::move(pot);
    if (has_validation_) fit_loss(r);

    // Pre-filters use raw metrics independently of the loss weights. Global
    // declarations gate the average; indexed declarations gate each dataset.
    bool skip_bad_pot = false;
    double skip_excess = -1.0;
    {
      auto check_metric = [&](const std::string& tag, bool used, double value,
                              const HPOTargets::SkipCaps& caps, const char* unit) {
        if (!used) return false;
        const auto hit = metric_skip_tracker_->check(tag, value, caps);
        if (!hit) return false;
        skip_excess = hit->excess;
        if (hit->count <= 5 || (hit->count & (hit->count - 1)) == 0) {
          std::ostringstream os;
          os << "eval " << r.iter << ": skipping downstream (obs+LAMMPS) -- "
             << tag << "=" << value << " " << unit;
          if (hit->absolute) os << " > skip-above=" << *caps.abs_above;
          if (hit->relative) os << " > skip-rel=" << *caps.rel_above
                               << " * best(" << hit->best << ")";
          os << ", total skips=" << hit->count;
          HPOLog::instance().warning(os.str());
        }
        return true;
      };
      auto check_targets = [&](const char* tag, bool used, double average,
                               const std::vector<double>& values,
                               const ValidationMetricTargets& targets, const char* unit) {
        if (!used) return false;
        if (!targets.indexed()) return check_metric(tag, true, average, targets.global().caps, unit);
        const auto& assignments = targets.datasets();
        if (values.size() != assignments.size())
          throw std::logic_error("Validation skip-filter dataset count mismatch");
        for (std::size_t i = 0; i < values.size(); ++i) {
          const std::string label = std::string(tag) + "[" + std::to_string(i + 1)
                                  + "] " + targets.files().at(i);
          if (check_metric(label, true, values[i], assignments[i].caps, unit)) return true;
        }
        return false;
      };
      skip_bad_pot =
          check_targets("ermse", trg_.use_ermse, r.ermse, r.v_ermse, trg_.ermse_targets, "meV/at") ||
          check_metric("emae", trg_.use_emae, r.emae, trg_.skip_emae, "meV/at") ||
          check_metric("eRRMSE", trg_.use_errmse, r.errmse, trg_.skip_errmse, "meV/at") ||
          check_metric("ersq", trg_.use_ersq, r.ersq, trg_.skip_ersq, "") ||
          check_targets("frmse", trg_.use_frmse, r.frmse, r.v_frmse, trg_.frmse_targets, "eV/A") ||
          check_metric("srmse", trg_.use_srmse, r.srmse, trg_.skip_srmse, "GPa");
      if (skip_bad_pot) {
        if (trg_.FAILSCORE_GRADING) {
          /* Band D graded by how far past the cap the metric landed --
           * a pre-LAMMPS sick-potential verdict with its own natural
           * violation measure. */
          r.fail.band = std::max(r.fail.band,
                                 static_cast<int>(FailBand::D));
          r.fail.vD = std::max(skip_excess, 0.0);
          r.loss = graded_eval_loss(r.fail, trg_.FAIL_SCORE,
                                    trg_.failprobe_s0);
        } else {
          r.loss = trg_.FAIL_SCORE;
        }
      }
    }

    /* PC_OBSERVABLE: typed observables fold into the same loss_sum /
     * loss_count accumulators as fit + sims. Skipped on bad-pot --
     * unlike the original D4c assumption ("observables are cheap"),
     * cohesive_energy with maxeval=5000 and --max-wall-time can still
     * be ~1 s per bad-pot iter, worth gating. If an observable
     * diverges its loss component is clamped to FAIL_SCORE per
     * obs_loss's contract; r.loss recomputes inside obs_loss. */
    if (!skip_bad_pot) {
      obs_loss(r);
    }
    t_predict_end = std::chrono::high_resolution_clock::now();
    t_lammps_end = t_predict_end;
    t_sims_loss_end = t_predict_end;
    stage_watchdog::leave(r.iter, "predict_validate", _stage_log);

    if (use_lammps_ && r.loss < trg_.FAIL_SCORE && !skip_bad_pot) {
      stage_watchdog::enter(r.iter, "lammps");
      run_lammps(r);
      t_lammps_end = std::chrono::high_resolution_clock::now();
      sims_loss(r);
      t_sims_loss_end = std::chrono::high_resolution_clock::now();
      stage_watchdog::leave(r.iter, "lammps", _stage_log);
    }

    /* Bug 7: post-LAMMPS finalize watchdog. Covers best tracking +
     * top-K + stagnation in one stage. RAII ensures leave() runs even
     * if stagnation_check throws StagnationStop. */
    {
    stage_watchdog::Scope<decltype(_stage_log)> _sg_finalize(r.iter, "post_lammps_finalize", _stage_log);
    /* best tracking ----------------------------------------------------
     * Serialize the full compound decision through durable publication. A
     * loss-only CAS cannot resolve exact-loss ties deterministically: a later
     * finishing (rank,slot,eval)-smaller candidate would never reach the file
     * comparator. New-best commits are rare, so holding this mutex across the
     * two fsync+rename operations is preferable to split-brain metadata/HOT. */
    {
      std::lock_guard<std::mutex> best_lock(best_mtx_);
      const durable_best::Key key{
          r.loss, r.source_rank, r.chain_id, r.iter};
      const double cur = g_best_.load(std::memory_order_relaxed);
      /* D2: a failure is never an achievement.
       *
       * Every comparison below is against g_best_, which starts at DBL_MAX.
       * A NaN loses all of them and so could never win, but a FAILURE is
       * finite by construction -- with FAILSCORE_GRADING the reserved band
       * is [0.999*FS, FS), e.g. ~9.9975e+08 for FAILSCORE 1e9 -- and beats
       * DBL_MAX comfortably. So a run whose first completed evaluation
       * failed (a bad WARM seed, a cold worker tripping its deadline) used
       * to publish that failure: best_pot.tadah written with
       * HPO_BEST_LOSS 9.997e+08, its empty HOT set promoted as the warm-start
       * baseline, and "BEST * eval=1" in the journal. If no feasible
       * evaluation ever arrived, that artifact is what the user collected.
       *
       * Neither layer below catches it: durable_best::publish_if_better and
       * HPOLog::log_new_best both test only isfinite(). Gate here, at the
       * decision, using the band predicate that owns the contract. */
      const bool feasible = loss_is_feasible(r.loss, trg_.FAIL_SCORE);
      const bool lower_loss = feasible && r.loss < cur;
      const bool tied_compound_winner =
          feasible && r.loss == cur && g_best_key_ &&
          durable_best::precedes(key, *g_best_key_);

      if (lower_loss || tied_compound_winner) {
        r.best_iter = g_best_iter_.load(std::memory_order_relaxed) + 1;

        /* On return the canonical artifact contains this candidate or an
         * equal/better compound incumbent from another MPI rank. The latter
         * does not reject this rank's improvement: g_best_, r.is_best, and
         * best_hot_map are deliberately rank-local live-search state. The
         * early multi-rank dispatch guard limits this split ownership to plain
         * PT/PA; the canonical file remains the sole global result. */
        persist_durable_best(r);

        if (!warmstart::promote_batch(key, r.hot_produced))
          throw durable_best::Error(
              "HOT snapshot comparator contradicts the durable compound best key");

        g_best_.store(r.loss, std::memory_order_release);
        g_best_key_ = key;
        g_best_iter_.store(r.best_iter, std::memory_order_relaxed);
        iter_at_last_best_.store(r.iter, std::memory_order_relaxed);
        r.is_best = true;
      }
    }

    /* Top-K tracking -- cheap fast-path when disabled (capacity == 0).
     * Near-clone dedup (loss + normalised parameter distance) lives in topk.h. */
    if (topk_capacity_ > 0) {
      std::lock_guard<std::mutex> lk(topk_mtx_);
      if (topk_capacity_ > 0) {  /* re-check inside lock */
        topk_consider(topk_buf_, topk_capacity_, r.loss, r.params,
                      topk_widths_);
      }
    }

    /* Stagnation check -- throw to short-circuit the optimiser loop. The
     * surrounding RestartPolicy catches StagnationStop and decides
     * whether to restart from a fresh init. */
    {
      const std::size_t threshold =
          stagnation_threshold_.load(std::memory_order_relaxed);
      /* See the BUMPGATE path above: probe evals do not count. */
      if (threshold > 0 && !probe::active()) {
        const std::size_t last = iter_at_last_best_.load(std::memory_order_relaxed);
        if (r.iter > last && r.iter - last >= threshold) {
          throw StagnationStop{};
        }
      }
    }
    } /* end post_lammps_finalize watchdog */

    r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0);

    /* save fields before r is moved into the queue ---------------------- */
    const std::size_t save_iter    = r.iter;
    const double      save_loss    = r.loss;
    const double      save_wall_ms = static_cast<double>(r.wall_ms.count());
    const bool        save_best    = r.is_best;
    const bool        save_nan     = r.was_nan_clamped;
    /* D4: the graded-failure band of this eval. Disjoint from save_nan --
     * a killed worker or a failed script yields a finite failscore, so it
     * never trips the NaN path. Worst band wins within the eval. */
    const int         save_fail_band = r.fail.band;
    const double      save_emae    = r.emae,   save_ermse  = r.ermse;
    const double      save_errmse  = r.errmse, save_frmse  = r.frmse;
    const double      save_srmse   = r.srmse;
    /* Stage 1a: fan-out utilisation snapshot (r is moved into the queue below). */
    const double      save_fo_busy  = r.fanout_busy_ms;
    const std::size_t save_fo_nsc   = r.fanout_nscripts;
    const double      save_fo_min   = r.fanout_min_ms;
    const double      save_fo_max   = r.fanout_max_ms;
    const std::string save_fo_slow  = r.fanout_slowest;

    /* Build a one-line "top loss components" breakdown when this eval
     * is a new best. Defensive: any failure here must not crash the
     * eval -- catch and continue with an empty breakdown. */
    std::string save_top_breakdown;
    if (save_best && save_loss > 0.0) {
        try {
            struct Comp { std::string label; double value; };
            std::vector<Comp> comps;
            if (trg_.use_emae   && r.emaeLoss   > 0) comps.push_back({std::string("EMAE_global"),   r.emaeLoss});
            if (trg_.use_ermse  && r.ermseLoss  > 0) comps.push_back({std::string("ERMSE_global"),  r.ermseLoss});
            if (trg_.use_errmse && r.errmseLoss > 0) comps.push_back({std::string("ErRMSE_global"), r.errmseLoss});
            if (trg_.use_ersq   && r.ersqLoss   > 0) comps.push_back({std::string("ErSQ_global"),   r.ersqLoss});
            if (trg_.use_frmse  && r.frmseLoss  > 0) comps.push_back({std::string("FRMSE_global"),  r.frmseLoss});
            if (trg_.use_srmse  && r.srmseLoss  > 0) comps.push_back({std::string("SRMSE_global"),  r.srmseLoss});
            for (std::size_t i = 0; i < trg_.LS.size() && i < r.v_loss.size(); ++i) {
                const auto& lsim = trg_.LS[i];
                for (std::size_t k = 0; k < lsim.i_varloss.size(); ++k) {
                    const std::size_t j = lsim.i_varloss[k];
                    if (j >= r.v_loss[i].size()) continue;
                    const double v = r.v_loss[i][j];
                    if (!(v > 0.0)) continue;
                    std::string lbl;
                    if (k < lsim.n_varloss.size() && !lsim.n_varloss[k].empty())
                        lbl = lsim.n_varloss[k];
                    else
                        lbl = std::string("sim") + std::to_string(i)
                            + "[" + std::to_string(k) + "]";
                    comps.push_back({std::move(lbl), v});
                }
            }
            /* PC_OBSERVABLE (D4c) -- typed observable contributions to
             * the top-breakdown. Symmetric with the PC_LAMMPS loop
             * above: pull r.v_obs_loss[k] keyed by trg_.observables[k]'s
             * kind() label. Without this, the journal's "top:" line
             * silently omits cohesive_energy and similar observables,
             * leaving users wondering whether their PC_OBSERVABLE
             * weight is even being applied. */
            for (std::size_t k = 0;
                 k < trg_.observables.size() && k < r.v_obs_loss.size();
                 ++k) {
                const double v = r.v_obs_loss[k];
                if (!(v > 0.0)) continue;
                const auto& obs_ptr = trg_.observables[k];
                std::string lbl = obs_ptr ? obs_ptr->kind()
                                          : std::string("obs[")
                                            + std::to_string(k) + "]";
                comps.push_back({std::move(lbl), v});
            }
            std::sort(comps.begin(), comps.end(),
                      [](const Comp& a, const Comp& b) { return a.value > b.value; });
            double sum_total = 0.0;
            for (const auto& c : comps) sum_total += c.value;
            /* Display values are scaled to per-component contribution to the
             * mean (comp / N), so they line up arithmetically with the
             * displayed `loss=` (which IS the mean): pct x loss ~= value.
             * sum_total is in raw (un-averaged) units, so the percentages
             * are unaffected by the per-display rescaling. */
            const double n_active = r.loss_count > 0
                                  ? static_cast<double>(r.loss_count) : 1.0;
            const std::size_t TOPN = std::min<std::size_t>(5, comps.size());
            std::ostringstream oss;
            for (std::size_t i = 0; i < TOPN; ++i) {
                if (i) oss << "  ";
                const double pct = sum_total > 0.0
                                 ? 100.0 * comps[i].value / sum_total : 0.0;
                const double display_val = comps[i].value / n_active;
                std::ostringstream val;
                val << std::scientific << std::setprecision(2) << display_val;
                oss << comps[i].label << "=" << val.str()
                    << "(" << std::fixed << std::setprecision(1) << pct << "%)";
            }
            save_top_breakdown = oss.str();
        } catch (const std::exception& e) {
            save_top_breakdown.clear();
            HPOLog::instance().warning(
                std::string("loss-breakdown failed: ") + e.what());
        }
    }

    {
    stage_watchdog::Scope<decltype(_stage_log)> _sg_submit(save_iter, "submit_result", _stage_log);
    submit_result(std::move(r), trg_, master_ctx_, val_ctx_, train_stdb_, val_stdb_, param_labels_, report_, cw_, prec_);
    }
    t_submit_end = std::chrono::high_resolution_clock::now();

    if (bench_detail::enabled()) {
      const double total_ms     = bench_detail::elapsed_ms(t0,              t_submit_end);
      const double train_ms     = bench_detail::elapsed_ms(t_train_start,   t_train_end);
      const double predict_ms   = bench_detail::elapsed_ms(t_train_end,     t_predict_end);
      const double lammps_ms    = bench_detail::elapsed_ms(t_predict_end,   t_lammps_end);
      const double sims_loss_ms = bench_detail::elapsed_ms(t_lammps_end,    t_sims_loss_end);
      const double submit_ms    = bench_detail::elapsed_ms(t_sims_loss_end, t_submit_end);
      int nthreads = 1;
#ifdef TADAH_ENABLE_OPENMP
      nthreads = omp_get_max_threads();
#endif
      std::lock_guard<std::mutex> lg(bench_detail::mtx());
      bench_detail::csv()
        << save_iter << ','
        << total_ms  << ','
        << train_ms  << ','
        << predict_ms << ','
        << lammps_ms << ','
        << sims_loss_ms << ','
        << submit_ms << ','
        << nthreads << '\n';
      bench_detail::csv().flush();
    }

    /* log to run journal ------------------------------------------------ */
    {
    stage_watchdog::Scope<decltype(_stage_log)> _sg_journal(save_iter, "journal_log", _stage_log);
    auto& log = HPOLog::instance();
    log.record_eval(save_iter, save_wall_ms, save_nan);
    _census.done = true;   /* counted here, with the exact measured wall */
    log.record_fail_band(save_fail_band);
    log.record_fanout(save_fo_busy,
                      save_fo_nsc, save_fo_min, save_fo_max, save_fo_slow);

    if (save_best) {
        log.log_new_best(save_iter, save_loss,
                         save_emae,   trg_.use_emae,
                         save_ermse,  trg_.use_ermse,
                         save_errmse, trg_.use_errmse,
                         save_frmse,  trg_.use_frmse,
                         save_srmse,  trg_.use_srmse,
                         save_wall_ms,
                         save_top_breakdown);
        /* Bound-occupancy audit on the new best (REAL coordinates). */
        std::lock_guard<std::mutex> lg(bound_occ_mtx_);
        if (bound_occ_.armed()) {
            std::vector<double> xr(static_cast<std::size_t>(p.size()));
            for (long i = 0; i < p.size(); ++i) xr[static_cast<std::size_t>(i)] = p(i);
            bound_occ_.on_new_best(save_iter, xr);
            log.record_bound_occupancy(bound_occ_.marks(), save_iter, bound_occ_.edge());
        }
    }

    if (save_nan)
        log.log_nan_clamped(save_iter,
                            format_params_str(p),
                            bound_hints(p));

    }

    return save_loss;

    } catch (const core::ShutdownRequested&) {
        /* Cooperative shutdown -- pass through to hpo_run()'s outer
         * catch, which drains the result queue and closes the journal.
         *
         * Bug 7: before rethrowing, submit a synthetic FAIL_SCORE
         * Result so the iter slot we consumed at `++g_iter_` does not
         * leak from the writer queue. Without this, the writer's
         * next_flush_ would be permanently stuck at our iter (the
         * gap-skip recovery would eventually advance it after 60 s,
         * but by then per-eval logs are lost). */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        _census.nan = true;      /* D3: this eval still counts, as a NaN */
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
        } catch (...) { /* swallow -- main shutdown path takes over */ }
        throw;
    } catch (const StagnationStop&) {
        /* StagnationStop is thrown from post_lammps_finalize AFTER r.loss
         * is computed and counted in best/top-K tracking. This eval is
         * therefore a real, valid result -- submit it as-is. The earlier
         * recovery path (Bug 7) overwrote r.loss with FAIL_SCORE and
         * flagged was_nan_clamped, which corrupted loss.tadah and
         * fabricated a NaN-clamp event for every concurrent thread that
         * happened to cross the threshold while the surrounding policy
         * was still resetting it.
         *
         * We still need to submit so the writer's next_flush_ does not
         * stall on the iter slot consumed at ++g_iter_ -- but with the
         * real loss, not a synthetic FAIL_SCORE. */
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        /* Diagnostic at most once per stagnation event. Concurrent
         * eval threads commonly race to this catch when the threshold
         * is crossed; without this gate the journal gets one warning
         * per concurrent eval. Cleared by reset_stagnation() at the
         * next stage boundary. */
        if (!stagnation_warned_.exchange(true, std::memory_order_relaxed)) {
            try {
                HPOLog::instance().info(
                    std::string("StagnationStop fired at eval ") +
                    std::to_string(r.iter) +
                    "; short-circuiting optimiser, real loss preserved "
                    "(threshold=" +
                    std::to_string(stagnation_threshold_.load(std::memory_order_relaxed)) +
                    " evals without improvement)");
            } catch (...) {}
        }
        try {
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
        } catch (...) {}
        throw;
    } catch (const durable_best::Error& e) {
        /* Persistence is part of committing a completed best evaluation.
         * Converting this failure to FAIL_SCORE would let optimization carry
         * on while silently losing the global best, so it must escape. The
         * evaluation itself is nevertheless complete and consumed an ordered
         * iter slot; enqueue its real loss before rethrowing so final cleanup
         * does not discard every later result behind a permanent gap. */
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            HPOLog::instance().warning(
                std::string("fatal best-potential persistence failure at eval ") +
                std::to_string(r.iter) + ": " + e.what());
        } catch (...) {}
        try {
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
        } catch (...) {
            /* Preserve the primary durability exception. The writer's
             * time-based gap recovery remains the last resort if enqueueing
             * itself fails. */
        }
        throw;
    } catch (const std::exception& e) {
        /* Any other throw between ++g_iter_ and submit_result must
         * not leak the iter slot. Build a synthetic FAIL_SCORE Result
         * carrying r.iter and submit it so the writer's next_flush_
         * cursor can advance past this iter. */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        _census.nan = true;      /* D3: this eval still counts, as a NaN */
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            HPOLog::instance().warning(
                std::string("eval ") + std::to_string(r.iter) +
                ": eval-thread caught throw, submitting FAIL_SCORE to "
                "keep writer queue contiguous (type=" + typeid(e).name() +
                "): " + e.what());
        } catch (...) { /* logging must never abort the recovery path */ }
        try {
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
        } catch (...) {
            /* If the synthetic submit also throws (extremely
             * unlikely), the writer fix from Bug 4 will at worst drop
             * this single iter from the per-iter logs. The eval
             * counter still advances; we don't make the freeze
             * recurrence worse. */
        }
        return trg_.FAIL_SCORE;
    } catch (...) {
        /* Same path for non-std::exception throws. */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        _census.nan = true;      /* D3: this eval still counts, as a NaN */
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            HPOLog::instance().warning(
                std::string("eval ") + std::to_string(r.iter) +
                ": eval-thread caught unknown throw, submitting "
                "FAIL_SCORE to keep writer queue contiguous");
        } catch (...) {}
        try {
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
        } catch (...) {}
        return trg_.FAIL_SCORE;
    }
}

/* ---------- opportunistic writer helper ------------------------------ */
  template<class P,class R>
void HPO_Host<P,R>::submit_result(Result&& res, const HPOTargets& trg,
                                  const core::Context& /*master_ctx*/,
                                  const core::Context& /*val_ctx*/,
                                  const mlip::StructureDB& /*train_stdb*/,
                                  const mlip::StructureDB& /*val_stdb*/,
                                  const std::vector<std::string>& /*param_labels*/,
                                  const mlip::LoadTransformReport& /*report*/,
                                  std::size_t cw, int prec)
{
    /* 1. enqueue ------------------------------------------------------- */
    g_q_.enqueue(std::move(res));

    /* 2. quick exit if queue still small ------------------------------- */
    if (g_q_.size_approx() < trg.QFLUSH_THRESHOLD)
        return;

    /* 3. try to become the writer ------------------------------------- */
    if(writer_token_.test_and_set(std::memory_order_acquire))
        return;                                  // somebody else is writing


    /* 4. exclusive writer section ------------------------------------- */
    /* Wrapped in try/catch so a throw inside any per-iter write cannot
     * leave writer_token_ stuck set -- that would silently freeze every
     * subsequent per-evaluation/result log while hpo_run.log keeps moving.
     * Canonical best persistence no longer depends on this queue. */
    /* Debug instrumentation: gated by TADAH_HPO_DEBUG_WRITER=1.
     * Writes a short log line to writer_debug.log on entry/exit of
     * the exclusive section + heartbeat every N iters. Used to track
     * down a freeze surface that bypasses the catch (Bug 6). */
    static const bool _writer_debug_enabled = []{
        const char* v = std::getenv("TADAH_HPO_DEBUG_WRITER");
        return v && *v && std::string(v) != "0";
    }();
    static std::mutex   _writer_debug_mtx;
    static std::ofstream _writer_debug_f;
    static std::atomic<std::size_t> _writer_call_id{0};
    auto _writer_debug_log = [](const std::string& msg) {
        if (!_writer_debug_enabled) return;
        std::lock_guard<std::mutex> lk(_writer_debug_mtx);
        if (!_writer_debug_f.is_open())
            _writer_debug_f.open("writer_debug.log",
                                 std::ios::out | std::ios::trunc);
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()) % 1000000;
        std::tm lt = *std::localtime(&t);
        _writer_debug_f << std::put_time(&lt, "%H:%M:%S") << "."
                        << std::setw(6) << std::setfill('0') << us.count()
                        << " " << msg << "\n";
        _writer_debug_f.flush();
    };
    const std::size_t _wid = _writer_call_id.fetch_add(1) + 1;
    _writer_debug_log("ENTRY wid=" + std::to_string(_wid)
                      + " q_size=" + std::to_string(g_q_.size_approx())
                      + " next_flush=" + std::to_string(next_flush_.load()));
    try {
        std::lock_guard<std::mutex> lg(io_mtx_);
        _writer_debug_log("LOCK_ACQUIRED wid=" + std::to_string(_wid));

        Result item;
        std::size_t _drained_from_q = 0;
        while(g_q_.try_dequeue(item)) {
            io_buf_.emplace(item.iter,std::move(item));
            ++_drained_from_q;
        }
        _writer_debug_log("Q_DRAINED wid=" + std::to_string(_wid)
                          + " items=" + std::to_string(_drained_from_q)
                          + " io_buf_size=" + std::to_string(io_buf_.size())
                          + " next_flush=" + std::to_string(next_flush_.load()));

        // Streams belong to this flushed batch. Retaining process-static
        // streams kept writing to the first run's files when a later host
        // used a different working directory (or replaced its output files).
        std::ofstream loss_f  (trg.lossFile, std::ios::app),
                             outv_f  (trg.outvarFile, std::ios::app),
                             par_f   (trg.paramsFile, std::ios::app),
                             stats_f (trg.statsFile, std::ios::app),
                             best_f  (trg.bestLossFile, std::ios::app),
                             best_outv_f(trg.bestOutvarFile, std::ios::app),
                             best_par_f (trg.bestParamsFile, std::ios::app),
                             best_stats_f(trg.bestStatsFile, std::ios::app);

        std::size_t _written = 0;
        std::size_t _last_iter_processed = 0;
        while(io_buf_.count(next_flush_))
        {
            _last_iter_processed = next_flush_.load();
            if (_writer_debug_enabled && (_written % 100) == 0) {
                _writer_debug_log("ITER_BEGIN wid=" + std::to_string(_wid)
                                  + " iter=" + std::to_string(_last_iter_processed)
                                  + " io_buf_size=" + std::to_string(io_buf_.size()));
            }
            try {
                const auto& e = io_buf_[next_flush_];

        /* ---- print standard logs ------------------------------------ */
        if(next_flush_ % trg.outputEvery == 0){
            print_loss_line  (loss_f , e.iter, e, trg, cw, prec);
            print_outvar_line(outv_f , e.iter, e, trg, cw, prec);
            print_params_line(par_f  , e.iter, e,      cw, prec);
            print_stats_line (stats_f, e.iter, e, trg, cw, prec);
        }

        /* ---- new best logs ------------------------------------------ */
        if(e.is_best) {
          if(e.best_iter % trg.bestOutputEvery == 0) {
            print_loss_line  (best_f      , e.best_iter, e, trg, cw, prec);
            print_outvar_line(best_outv_f , e.best_iter, e, trg, cw, prec);
            print_params_line(best_par_f  , e.best_iter, e,      cw, prec);
            print_stats_line (best_stats_f, e.best_iter, e, trg, cw, prec);
          }

          /* archive best_pot_N.tadah ---------------------------------- */
          if(e.best_iter % trg.dumpBestPotEvery == 0){
            std::ofstream bpa(std::string(trg.dumpBestPotDir)
                +"/best_pot_" + std::to_string(e.best_iter) + ".tadah"
                + trg.potArchiveSuffix);

            PotHeaderWriter::write_best_archive_header(
                bpa, e, trg, cw, prec);

            PotHeaderWriter::write_candidate_hot(bpa, e);

            bpa << std::setprecision(
                e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
          }
        }

        /* ---- archive ANY pot every ---------------------------------- */
        if(e.iter % trg.dumpPotEvery == 0){
            std::ofstream pa(std::string(trg.dumpPotDir)
                +"/pot_"+std::to_string(e.iter)+".tadah"
                + trg.potArchiveSuffix);

            PotHeaderWriter::write_archive_header(pa, e, cw, prec);

            PotHeaderWriter::write_candidate_hot(pa, e);

            pa << std::setprecision(
                e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
        }

                io_buf_.erase(next_flush_++);
                ++_written;
            } catch (const std::exception& ex) {
                /* Drop the offending iter's logs and advance so the queue
                 * keeps draining. One iter's logs are lost; the journal
                 * records the failure for follow-up. The Result fields
                 * (is_best, has pot_ctx NUMERIC?) are included so the
                 * journal entry alone is enough to differentiate the
                 * pot_ctx.get / archive-header / per-iter print
                 * candidate throw sites without needing a second
                 * reproduction. */
                bool is_best = false, has_numeric = false;
                std::size_t pot_ctx_size = 0;
                if (auto it = io_buf_.find(next_flush_); it != io_buf_.end()) {
                    is_best = it->second.is_best;
                    has_numeric = it->second.pot_ctx.contains("NUMERIC");
                    pot_ctx_size = it->second.pot_ctx.size();
                }
                HPOLog::instance().warning(
                    std::string("HPO writer: dropping iter=") +
                    std::to_string(next_flush_.load()) +
                    " (is_best=" + (is_best ? "true" : "false") +
                    ", pot_ctx.size=" + std::to_string(pot_ctx_size) +
                    ", has NUMERIC=" + (has_numeric ? "true" : "false") +
                    ", exception type=" + typeid(ex).name() +
                    "): " + ex.what());
                io_buf_.erase(next_flush_++);
                ++_written;
            } catch (...) {
                HPOLog::instance().warning(
                    std::string("HPO writer: dropping iter=") +
                    std::to_string(next_flush_.load()) +
                    " after unknown (non-std::exception) throw");
                io_buf_.erase(next_flush_++);
                ++_written;
            }
        }
        _writer_debug_log("LOOP_DONE wid=" + std::to_string(_wid)
                          + " written=" + std::to_string(_written)
                          + " last_iter=" + std::to_string(_last_iter_processed)
                          + " next_flush=" + std::to_string(next_flush_.load())
                          + " io_buf_remaining=" + std::to_string(io_buf_.size()));

        /* Bug 6: gap-skip recovery for hung eval threads.
         *
         * A worker thread can consume an iter slot via `r.iter = ++g_iter_`
         * inside operator() and then hang inside train/predict/LAMMPS
         * before reaching submit_result. The slot N is then never
         * enqueued, but iters N+1, N+2, ... continue to flow into io_buf_.
         * The drain loop terminates at next_flush_=N because count(N)==0,
         * and stays there indefinitely. operator()'s try/catch (Bug 5)
         * does not help -- there is no exception, the eval thread is
         * simply stuck.
         *
         * Recovery: if next_flush_ is missing AND io_buf_ has higher
         * keys AND we have been stuck for more than GAP_SKIP_MS, advance
         * next_flush_ to the smallest available key, recording the skip
         * in the journal. The hung iter's per-eval logs are lost; if the
         * eval thread eventually unsticks and submits the result, that
         * Result lands in io_buf_ with key < next_flush_ and is silently
         * abandoned (memory cost is bounded). */
        constexpr std::int64_t GAP_SKIP_MS = 60'000;  // 60 s
        if (_written > 0) {
            /* Healthy progress this call -- clear the stuck timer. */
            next_flush_stuck_since_ms_.store(0, std::memory_order_relaxed);
        } else if (!io_buf_.empty()) {
            const auto smallest = io_buf_.begin()->first;
            const auto cur = next_flush_.load(std::memory_order_relaxed);
            if (smallest > cur) {
                const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                std::int64_t since = next_flush_stuck_since_ms_.load(std::memory_order_relaxed);
                if (since == 0) {
                    /* Start the stuck timer. */
                    next_flush_stuck_since_ms_.store(now_ms, std::memory_order_relaxed);
                    _writer_debug_log("GAP_DETECTED wid=" + std::to_string(_wid)
                                      + " missing=" + std::to_string(cur)
                                      + " smallest_avail=" + std::to_string(smallest)
                                      + " io_buf_size=" + std::to_string(io_buf_.size()));
                } else if (now_ms - since > GAP_SKIP_MS) {
                    const std::size_t skipped = smallest - cur;
                    /* Snapshot the watchdog: list any thread that has been
                     * inside an eval pipeline stage for longer than the
                     * gap-skip threshold. If the hang is in train/predict/
                     * lammps, the offending stage(s) are named here so
                     * follow-up does not need a fresh reproduction. */
                    const std::string stuck_stages =
                        stage_watchdog::snapshot_stuck(GAP_SKIP_MS);
                    HPOLog::instance().warning(
                        std::string("HPO writer: gap-skip -- advancing next_flush_ from ") +
                        std::to_string(cur) + " to " +
                        std::to_string(smallest) + " (" +
                        std::to_string(skipped) + " hung iter slot" +
                        (skipped == 1 ? "" : "s") +
                        " whose eval thread did not return within " +
                        std::to_string(GAP_SKIP_MS / 1000) + " s; per-eval logs lost)" +
                        (stuck_stages.empty() ? std::string{}
                            : std::string(" | active stuck stages: [") + stuck_stages + "]"));
                    _writer_debug_log("GAP_SKIP wid=" + std::to_string(_wid)
                                      + " from=" + std::to_string(cur)
                                      + " to=" + std::to_string(smallest)
                                      + " skipped=" + std::to_string(skipped));
                    next_flush_.store(smallest, std::memory_order_relaxed);
                    next_flush_stuck_since_ms_.store(0, std::memory_order_relaxed);
                    /* Re-enter the drain loop to process the now-available iters. */
                    while (io_buf_.count(next_flush_)) {
                        try {
                            const auto& e = io_buf_[next_flush_];
                            if (next_flush_ % trg.outputEvery == 0) {
                                print_loss_line  (loss_f , e.iter, e, trg, cw, prec);
                                print_outvar_line(outv_f , e.iter, e, trg, cw, prec);
                                print_params_line(par_f  , e.iter, e,      cw, prec);
                                print_stats_line (stats_f, e.iter, e, trg, cw, prec);
                            }
                            if (e.is_best) {
                                if (e.best_iter % trg.bestOutputEvery == 0) {
                                    print_loss_line  (best_f      , e.best_iter, e, trg, cw, prec);
                                    print_outvar_line(best_outv_f , e.best_iter, e, trg, cw, prec);
                                    print_params_line(best_par_f  , e.best_iter, e,      cw, prec);
                                    print_stats_line (best_stats_f, e.best_iter, e, trg, cw, prec);
                                }
                                if (e.best_iter % trg.dumpBestPotEvery == 0) {
                                    std::ofstream bpa(std::string(trg.dumpBestPotDir)
                                        +"/best_pot_" + std::to_string(e.best_iter) + ".tadah"
                                        + trg.potArchiveSuffix);
                                    PotHeaderWriter::write_best_archive_header(bpa, e, trg, cw, prec);
                                    PotHeaderWriter::write_candidate_hot(bpa, e);
                                    bpa << std::setprecision(
                                        e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                                }
                            }
                            if (e.iter % trg.dumpPotEvery == 0) {
                                std::ofstream pa(std::string(trg.dumpPotDir)
                                    +"/pot_"+std::to_string(e.iter)+".tadah"
                                    + trg.potArchiveSuffix);
                                PotHeaderWriter::write_archive_header(pa, e, cw, prec);
                                PotHeaderWriter::write_candidate_hot(pa, e);
                                pa << std::setprecision(
                                    e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                            }
                            io_buf_.erase(next_flush_++);
                            ++_written;
                        } catch (const std::exception& ex) {
                            HPOLog::instance().warning(
                                std::string("HPO writer (post-skip): dropping iter=") +
                                std::to_string(next_flush_.load()) +
                                " exception: " + ex.what());
                            io_buf_.erase(next_flush_++);
                            ++_written;
                        }
                    }
                    _writer_debug_log("POST_SKIP_DRAIN wid=" + std::to_string(_wid)
                                      + " written_total=" + std::to_string(_written)
                                      + " next_flush=" + std::to_string(next_flush_.load())
                                      + " io_buf_remaining=" + std::to_string(io_buf_.size()));
                }
            }
        }

        _writer_debug_log("FLUSH_BEGIN wid=" + std::to_string(_wid));
        loss_f.flush();
        outv_f.flush();
        par_f.flush();
        stats_f.flush();
        best_f.flush();
        best_outv_f.flush();
        best_par_f.flush();
        best_stats_f.flush();
        _writer_debug_log("FLUSH_DONE wid=" + std::to_string(_wid)
                          + " loss_good=" + (loss_f.good() ? "Y" : "N")
                          + " loss_fail=" + (loss_f.fail() ? "Y" : "N")
                          + " loss_bad=" + (loss_f.bad() ? "Y" : "N")
                          + " loss_eof=" + (loss_f.eof() ? "Y" : "N"));
    } catch (const std::exception& ex) {
        /* Belt-and-suspenders: covers throws outside the per-iter block
         * (e.g. dequeue, ofstream construction). Token is cleared below. */
        HPOLog::instance().warning(
            std::string("HPO writer: outer catch (next_flush=") +
            std::to_string(next_flush_.load()) +
            ", type=" + typeid(ex).name() + "): " + ex.what());
        _writer_debug_log("OUTER_CATCH wid=" + std::to_string(_wid)
                          + " what=" + ex.what());
    } catch (...) {
        HPOLog::instance().warning(
            std::string("HPO writer: outer catch (next_flush=") +
            std::to_string(next_flush_.load()) + ") unknown exception");
        _writer_debug_log("OUTER_CATCH_UNKNOWN wid=" + std::to_string(_wid));
    }

    /* 5. release token ------------------------------------------------- */
    writer_token_.clear(std::memory_order_release);
    _writer_debug_log("EXIT wid=" + std::to_string(_wid));
}

/* ---------- analytical gradient callback (Phase 2A.4) ---------------- *
 *
 * Wires the IFT chain rule end-to-end against the live trainer cache.
 * The optimizer's cost-then-gradient call sequence guarantees that
 * operator()(x) ran immediately before us at the same x, leaving:
 *   - tls_trainer_cache_.trainer with `dm.getPhi()`/`dm.getT()` reflecting
 *     the augmented training matrix at theta = x (data rows scaled by
 *     sqrt(eweight*eweightglob); augmentation block zero -- train works
 *     on a copy and writes sqrt(lambda)*I_no_bias into that copy).
 *   - model->get_weights() holding W*(x).
 *
 * Strategy: on-demand factor reconstruction (locked decision, see
 * project memory). Read live Phi/T/W, reassemble the
 * sqrt(lambda)*I_no_bias augmentation block, build the Cholesky factor
 * via dloss_dtheta::detail helpers, then call assemble_dphi_train_energy_rows
 * twice per theta (once each over train_stdb_ and val_stdb_) to populate
 * `per_theta`. compute_dloss_dtheta runs the metric chain rule. Indices
 * the descriptor walker can't supply analytically (registry returns
 * false) end up in `needs_fd` for HYBRID's masked-FD pass.
 *
 * Phase 2A first-cut restrictions -- fall back to all-FD if violated:
 *   - FORCE=false, STRESS=false  (energy-only chain rule)
 *   - NORM=false                 (raw-feature W; chain rule lives in raw space)
 *   - BF_Linear                  (the only basis the walker handles)
 *   - OALGO=4 with LAMBDA != -1  (NormalEquations + fixed lambda)
 * Each restriction lifts in a follow-up sub-task.
 *
 * The body delegates to compute_analytical_gradient_from_trainer (see
 * dloss_dtheta.h), a free helper that takes a Trainer + train/val DBs +
 * HPOTargets and returns the gradient. Keeping the plumbing as a free
 * function makes it testable without HPO_Host's full data scaffolding
 * (DBFILE / config.hpo / val.tadah). */
template<class P, class R>
bool HPO_Host<P,R>::analytical_gradient(const P& x, P& g,
                                        std::vector<int>& needs_fd) const
{
    const auto prefs = make_param_ref_table(trg_);
    g.set_size(static_cast<long>(prefs.size()));
    for (long i = 0; i < g.size(); ++i) g(i) = 0.0;
    needs_fd.clear();
    needs_fd.reserve(prefs.size());
    (void)x;  /* unused: compute_dloss_dtheta reads everything from the plan. */

    auto fall_back_all_fd = [&]() {
        for (std::size_t i = 0; i < prefs.size(); ++i)
            needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
        return true;
    };

    /* No training data (weights provisioned from FIXWEIGHT), or FIXINDEX
     * present: the IFT orchestrator assumes the unconstrained ridge
     * solution map, which partial fixing invalidates (the free weights
     * solve a reduced system against T_r = T - Phi_f * w_f). Route every
     * theta onto the masked-FD path. Closed-form dL/dFIXWEIGHT and a
     * fixed-aware IFT are follow-ups. */
    if (!has_train_data_ || master_ctx_.contains("FIXINDEX")) {
        static std::once_flag warn_once;
        std::call_once(warn_once, [this]{
            HPOLog::instance().warning(
                has_train_data_
                ? "FIXINDEX present: analytical/IFT gradients assume the "
                  "unconstrained LS solution map, so all parameters use "
                  "finite differences for this run."
                : "No training DBFILE: analytical/IFT gradients need live "
                  "Phi/T, so all parameters use finite differences for "
                  "this run.");
        });
        return fall_back_all_fd();
    }

    auto& cache = tls_trainer_cache_;
    /* tls_trainer_theta_stale_: the bump-gate reject path returned before
     * training, so the cache holds the PREVIOUS feasible theta, not x --
     * fall back to FD, which re-evaluates operator() and passes through
     * the gate (non-smooth across the constraint boundary, same class as
     * the existing skip caps). */
    if (!cache.trainer || tls_trainer_theta_stale_) {
        fall_back_all_fd();
        emit_grad_diagnostic(prefs, needs_fd, /*cache_miss=*/true);
        return true;
    }
    std::vector<double> g_flat;
    bool helper_ok = false;
    std::string helper_throw_msg;

    /* Phase 3.7B-orchestrator -- SimsGradProvider construction.
     *
     * Lambda spins up a fresh LAMMPS instance per script, runs the
     * relaxed-state script with the current iteration's pot, then folds
     * the LAMMPS-observable gradients into g_out via
     * `populate_sims_grad_input_from_lammps`. The instances are owned
     * by `lmp_holders` (declared OUTSIDE the lambda so they outlive the
     * `compute_sims_grad_dWchain` call which aliases pointers into
     * `lt->aeds` / `lt->adjoint_snapshot`).
     *
     * Phase 3.7B-warmstart: hot-var scripts are now supported. A
     * name-keyed snapshot is built from the warmstart cache (with
     * `hv.initial` / midpoint fallback when the cache is unprimed) at
     * the start of each lambda invocation; every script -- hot or cold
     * -- gets `variable <name> string <value>` injected before
     * `run_script` runs. The gradient lambda is strictly read-only on
     * the cache (no `warmstart::set`) -- only operator() owns cache
     * mutation. Out-of-range produced hot values aren't validated; if
     * the relaxation under the seeded value gives an unphysical
     * observable, `populate_sims_grad_input_from_lammps` returns false
     * via the FAIL_SCORE-clamp path and the orchestrator falls all
     * theta back to FD per the all-or-nothing contract. */
    std::vector<typename R::LMPPtr> lmp_holders;
    SimsGradProvider sims_grad_provider = nullptr;

    /* PC_OBSERVABLE (D4c) -- typed-observable provider lambda. Parallel
     * to SimsGradProvider but much simpler: the observable's
     * evaluate() is pure Tadah (no LAMMPS instance, no adjoint
     * snapshot for `needs_adjoint=false` observables like
     * cohesive_energy). Captures the thread-local trainer cache to
     * route model + dc + context references through.
     *
     * Hot-var warmstart: lives inside Observable state (a0_current_
     * for cohesive_energy), updated in-place on every evaluate call.
     * No external WarmStart cache plumbing needed -- the Observable
     * owns its hot vars per the D3 design contract. */
    ObservableGradProvider obs_grad_provider = nullptr;
    if (!trg_.observables.empty()) {
        obs_grad_provider =
            [this, &cache]
            (std::vector<ObservableGradInput>& inputs) -> bool {
                inputs.clear();
                inputs.resize(trg_.observables.size());
                if (!cache.trainer || !cache.trainer->model)
                    return false;
                /* FORCE=true required by NativeRelaxer's atom-force
                 * path inside each observable's evaluate. See
                 * obs_loss() for the same guard reasoning. */
                detail::ObsForceFlagGuard _force_guard(cache.trainer->context);
                for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
                    auto& obs_ptr = trg_.observables[k];
                    if (!obs_ptr) {
                        /* Structural error -- trg_ is malformed. Hard
                         * fail; the helper falls back to all-FD. */
                        return false;
                    }
                    ObservableEvalResult res;
                    try {
                        res = obs_ptr->evaluate(
                            *cache.trainer->model,
                            cache.trainer->dc,
                            cache.trainer->context);
                    } catch (const std::exception&) {
                        /* Phase A masking: relaxer divergence, NaN
                         * energy, model mismatch. Mark this observable
                         * as failed and continue. The loss obs_loss()
                         * returns FAIL_SCORE for the same observable,
                         * so dFAIL_SCORE/dtheta = 0 and the missing chain
                         * contribution is exact in the continuous
                         * region. */
                        inputs[k].failed = true;
                        continue;
                    }
                    if (!std::isfinite(res.value)) {
                        inputs[k].failed = true;
                        continue;
                    }

                    inputs[k].value         = res.value;
                    inputs[k].target        = obs_ptr->target();
                    inputs[k].weight        = obs_ptr->weight();
                    inputs[k].loss_fn       = obs_ptr->lossfn();
                    inputs[k].needs_adjoint = res.needs_adjoint;
                    inputs[k].dO_dW.resize(static_cast<long>(res.dO_dW.size()));
                    for (std::size_t c = 0; c < res.dO_dW.size(); ++c)
                        inputs[k].dO_dW(static_cast<long>(c)) = res.dO_dW[c];
                }
                return true;
            };
    }

    if (!trg_.LS.empty()) {
        lmp_holders.reserve(trg_.LS.size());
        sims_grad_provider =
            [this, &lmp_holders, &cache]
            (std::vector<SimsGradScriptInput>& inputs) -> bool {
                inputs.clear();
                inputs.resize(trg_.LS.size());
                lmp_holders.clear();
                lmp_holders.reserve(trg_.LS.size());

                /* Build a name-keyed snapshot of hot-var seeds from
                 * the warmstart cache. Mirrors operator()'s phase A
                 * seed logic + phase B cross-script propagation:
                 * a cold script that reads ${<name>} sees the same
                 * value the hot script produced on the most recent
                 * successful eval. Collapsed by name (not by (script,
                 * name)) because cold scripts inject the same way. */
                std::unordered_map<std::string, double> hot_snapshot;
                for (const auto& lsim : trg_.LS) {
                    for (const auto& hv : lsim.hotvars) {
                        if (hot_snapshot.count(hv.name)) continue;
                        /* Gradient path has no chain context -- read the
                         * chain-agnostic best hot value. */
                        if (auto cached = warmstart::best_get(lsim.script, hv.name)) {
                            hot_snapshot[hv.name] = *cached;
                        } else if (hv.initial) {
                            hot_snapshot[hv.name] = *hv.initial;
                        } else {
                            hot_snapshot[hv.name] = 0.5 * (hv.min + hv.max);
                        }
                    }
                }

                auto inject_hot_value =
                    [](typename R::LMPPtr& lmp,
                       const std::string& name, double value) {
                        /* equal-style for same-style overwrite -- see the
                         * eval-pipeline variant above for rationale. */
                        char cmd[256];
                        std::snprintf(cmd, sizeof(cmd),
                                      "variable %s equal %.17g",
                                      name.c_str(), value);
                        try { lmp->input->one(cmd); }
                        catch (...) { /* keep going -- same as operator() */ }
                    };

                for (std::size_t k = 0; k < trg_.LS.size(); ++k) {
                    const auto& lsim = trg_.LS[k];

                    /* Build pot fresh from the current trainer (mirror
                     * eval pipeline at hpo_host_impl.hpp:1323-1336). */
                    core::Context pot = cache.trainer->model->get_param_file();
                    mlip::stamp_into_param_file(pot, report_);
                    pot.remove("AUDIT");
                    pot.remove("HEALTH_LOG");

                    R runner;
                    auto lmp = runner.create_instance(hpo_lammps_comm(),
                                                      lsim.invar);
                    if (!lmp) {
                        /* Phase A masking: one LAMMPS instance creation
                         * failure shouldn't poison the whole gradient.
                         * Mark this entry as failed and continue. */
                        inputs[k].failed = true;
                        continue;
                    }

                    /* Inject every known hot-var seed BEFORE running the
                     * script body. Scripts may overwrite via their own
                     * `variable` lines (last-wins for same-style). */
                    for (const auto& kv : hot_snapshot)
                        inject_hot_value(lmp, kv.first, kv.second);

                    bool grad_run_ok = false;
                    try {
                        /* Rendered once per sim (lazy), executed through
                         * Input::file() -- same executor as the eval
                         * pipeline. */
                        grad_run_ok =
                            runner.run_script(lmp, lsim.script_file(), pot);
                    } catch (const std::exception&) {
                        grad_run_ok = false;  /* ScriptFile render failed */
                    }
                    if (!grad_run_ok) {
                        /* Phase A masking: LAMMPS minimisation diverged
                         * or hit NaN under the current pot. Mark and
                         * continue -- the failing script's loss is
                         * FAIL_SCORE, a constant, so its gradient
                         * contribution is exactly zero in the
                         * continuous region. */
                        inputs[k].failed = true;
                        continue;
                    }

                    std::vector<std::string> v_outvar;
                    v_outvar.reserve(lsim.n_outvar.size());
                    for (const auto& name : lsim.n_outvar)
                        v_outvar.push_back(runner.get_variable(lmp, name));

                    const int box_dofs =
                        detect_box_dofs_from_script(lsim.script_lines);
                    const auto exprs = extract_varloss_expressions(lsim);

                    if (!populate_sims_grad_input_from_lammps(
                            lmp.get(), lsim, v_outvar, exprs,
                            box_dofs, lsim.failscore, inputs[k])) {
                        /* Phase A masking: populate_sims_grad_input
                         * returns false on FAIL_SCORE clamps (unphysical
                         * energies, non-finite varloss values). The
                         * resulting loss is FAIL_SCORE -> zero
                         * gradient contribution. Reset inputs[k] to a
                         * clean failed state (populate may have
                         * partially mutated it). */
                        inputs[k] = SimsGradScriptInput{};
                        inputs[k].failed = true;
                        continue;
                    }

                    lmp_holders.push_back(std::move(lmp));
                }
                return true;
            };
    }

    try {
        helper_ok = compute_analytical_gradient_from_trainer(
            *cache.trainer, train_stdb_, val_stdb_, trg_, prefs,
            g_flat, needs_fd, sims_grad_provider, obs_grad_provider);
    } catch (const std::exception& e) {
        /* Any throw from the analytical helper (out-of-range, walker
         * mismatch, internal Cholesky, ...) is gracefully degraded
         * here: route all theta to FD, surface the message in the
         * diagnostic and the journal so the user can debug. Without
         * this catch the exception would propagate through gp.gradient
         * into NLopt, which translates to a generic NLOPT_FAILURE with
         * no useful information. */
        helper_throw_msg = e.what();
        helper_ok = false;
    }
    if (!helper_ok) {
        if (!helper_throw_msg.empty()) {
            std::ostringstream oss;
            oss << "analytical_gradient: helper threw '"
                << helper_throw_msg << "' -- routing every theta to FD";
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
        }
        /* Analytical helper signalled a hard failure (today the only
         * path: IFT-side Cholesky non-PD even though the trainer's
         * Cholesky succeeded -- rare but happens when LAMBDA is tiny
         * relative to Phi^TPhi's smallest non-trivial eigenvalue, or when
         * a per-theta dPhi row is itself rank-deficient). Returning
         * false propagates to gp.gradient and the NLopt wrapper, where
         * LD_LBFGS sees +inf gradient and bails out as NLOPT_FAILURE.
         *
         * Instead, route every theta onto needs_fd so HYBRID's masked
         * FD pass produces a valid (slower) gradient and the
         * optimiser keeps making progress. The diagnostic surfaces
         * `ift_failed` so the user can tell this from a clean
         * fall_back_all_fd inside the helper. */
        needs_fd.clear();
        fall_back_all_fd();
        emit_grad_diagnostic(prefs, needs_fd, /*cache_miss=*/false,
                             /*ift_failed=*/true);
        return true;
    }
    for (std::size_t i = 0; i < g_flat.size(); ++i)
        g(static_cast<long>(i)) = g_flat[i];
    emit_grad_diagnostic(prefs, needs_fd, /*cache_miss=*/false);
    return true;
}

/* ---------- one-shot diagnostic for the analytical-gradient pipeline ---
 *
 * Logs the analytical-vs-FD split on the first analytical_gradient call
 * (per process). Users running GRAD_STRATEGY=HYBRID / ANALYTICAL with
 * a gradient-based optimiser (NLOPT:LD_LBFGS, DLIB:BFGS, CERES:LBFGS,
 * ...) have no other way to verify whether the closed-form path is
 * actually engaging -- the wall-clock difference vs Numerical depends on
 * the descriptor stack and is easy to misattribute. The log emits:
 *
 *   - GRAD_STRATEGY in use
 *   - total theta count
 *   - count covered by closed-form (analytical)
 *   - count routed to masked FD (HYBRID) or that would throw (ANALYTICAL)
 *
 * If the orchestrator's gate fell back entirely (all theta on
 * needs_fd), surface that as a separate line so the user can find the
 * specific restriction (FORCE/STRESS/NORM/OALGO/LAMBDA/MODEL[1]/
 * use_srmse/LAMMPS -- see compute_analytical_gradient_from_trainer's
 * gate stack). When the trainer cache is empty (first eval of the
 * BLR-warmup window), log that explicitly so a transient all-FD line
 * isn't mistaken for a gate failure. */
template<class P, class R>
void HPO_Host<P,R>::emit_grad_diagnostic(
    const std::vector<ParamRef>& prefs,
    const std::vector<int>&      needs_fd,
    bool                         cache_miss,
    bool                         ift_failed) const
{
    static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (emitted.test_and_set(std::memory_order_relaxed)) return;

    const std::size_t n_total = prefs.size();
    const std::size_t n_fd    = needs_fd.size();
    const std::size_t n_anal  = n_total - n_fd;

    const char* strat_name = "NUMERICAL";
    if (spec_.grad_strategy) {
        switch (*spec_.grad_strategy) {
          case GradStrategy::Numerical:  strat_name = "NUMERICAL";  break;
          case GradStrategy::Analytical: strat_name = "ANALYTICAL"; break;
          case GradStrategy::Hybrid:     strat_name = "HYBRID";     break;
        }
    }

    std::ostringstream oss;
    oss << "Gradient-pipeline diagnostic (first eval): "
        << "GRAD_STRATEGY=" << strat_name
        << " | total_theta=" << n_total
        << " | analytical="  << n_anal
        << " | FD-fallback=" << n_fd;
    if (cache_miss) {
        oss << "  [trainer cache empty on first call -- usually the BLR-"
               "warmup eval; subsequent calls populate the cache and "
               "engage the analytical path]";
    } else if (ift_failed) {
        oss << "  [IFT-side Cholesky non-PD even though the trainer "
               "fitted OK. Likely LAMBDA is small relative to Phi^TPhi's "
               "smallest eigenvalue, or a per-theta dPhi row is "
               "rank-deficient. Routed every theta to FD so the "
               "optimiser keeps making progress. Raise LAMBDA or "
               "switch to GRAD_STRATEGY=NUMERICAL to silence this]";
    } else if (n_total > 0 && n_fd == n_total) {
        oss << "  [gate fall-back: every theta routed to FD. Check "
               "STRESS/NORM/OALGO/LAMBDA/MODEL[1]/use_srmse/PC_LAMMPS "
               "restrictions in compute_analytical_gradient_from_trainer]";
    }
    if (n_fd > 0 && n_fd < n_total) {
        oss << "\n  FD-fallback theta:";
        for (int flat : needs_fd) {
            for (const auto& p : prefs) {
                if (static_cast<int>(p.flat_idx) == flat) {
                    oss << " " << p.theta_name;
                    if (p.arg_pos > 0) oss << "[" << p.arg_pos << "]";
                    break;
                }
            }
        }
    }
    std::cout << "[HPO] " << oss.str() << "\n";
    HPOLog::instance().info(oss.str());
}

/* ---------- final drain (call after optimizer returns) --------------- */
template<class P,class R>
void HPO_Host<P,R>::flush_remaining() const
{
    // Drain g_q_ into the ordered buffer, then write everything pending.
    std::lock_guard<std::mutex> lg(io_mtx_);

    Result item;
    while(g_q_.try_dequeue(item))
        io_buf_.emplace(item.iter, std::move(item));

    // Re-open (append) the same files that submit_result uses.
    // Using named ofstream objects so they close + flush on scope exit.
    std::ofstream loss_f  (trg_.lossFile,       std::ios::app),
                  outv_f  (trg_.outvarFile,      std::ios::app),
                  par_f   (trg_.paramsFile,      std::ios::app),
                  stats_f (trg_.statsFile,       std::ios::app),
                  best_f  (trg_.bestLossFile,    std::ios::app),
                  best_outv_f(trg_.bestOutvarFile, std::ios::app),
                  best_par_f (trg_.bestParamsFile, std::ios::app),
                  best_stats_f(trg_.bestStatsFile, std::ios::app);

    auto write_row = [&](const Result& e) {
        if(e.iter % trg_.outputEvery == 0){
            print_loss_line  (loss_f , e.iter, e, trg_, cw_, prec_);
            print_outvar_line(outv_f , e.iter, e, trg_, cw_, prec_);
            print_params_line(par_f  , e.iter, e,       cw_, prec_);
            print_stats_line (stats_f, e.iter, e, trg_, cw_, prec_);
        }
        if(e.is_best && e.best_iter % trg_.bestOutputEvery == 0) {
            print_loss_line  (best_f      , e.best_iter, e, trg_, cw_, prec_);
            print_outvar_line(best_outv_f , e.best_iter, e, trg_, cw_, prec_);
            print_params_line(best_par_f  , e.best_iter, e,       cw_, prec_);
            print_stats_line (best_stats_f, e.best_iter, e, trg_, cw_, prec_);
        }

        if(e.is_best) {
            if(e.best_iter % trg_.dumpBestPotEvery == 0){
                std::ofstream bpa(std::string(trg_.dumpBestPotDir)
                    + "/best_pot_" + std::to_string(e.best_iter) + ".tadah"
                    + trg_.potArchiveSuffix);

                PotHeaderWriter::write_best_archive_header(
                    bpa, e, trg_, cw_, prec_);

                PotHeaderWriter::write_candidate_hot(bpa, e);

                bpa << std::setprecision(
                    e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
            }
        }

        if(e.iter % trg_.dumpPotEvery == 0){
            std::ofstream pa(std::string(trg_.dumpPotDir)
                + "/pot_" + std::to_string(e.iter) + ".tadah"
                + trg_.potArchiveSuffix);

            PotHeaderWriter::write_archive_header(pa, e, cw_, prec_);

            PotHeaderWriter::write_candidate_hot(pa, e);

            pa << std::setprecision(
                e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
        }
    };

    /* In iteration order; rows stranded behind a missing iteration are
     * written too, with a journal warning (result_drain.h). The run is
     * over, so no evaluation can fill the gap any more. */
    drain_remaining_results(io_buf_, next_flush_, write_row,
        [](const std::string& msg) { HPOLog::instance().warning(msg); });
}

}   // namespace tadah::hpo

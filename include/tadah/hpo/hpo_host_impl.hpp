#pragma once
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/core/logger.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/sims_grad_lammps.h>
#include <tadah/hpo/val_filter_forwarding.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/pot_header_writer_impl.hpp>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/hpo/iter_health.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <fstream>
#include <mutex>
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
      f.open("hpo_phase_times.csv", std::ios::out | std::ios::trunc);
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
 * The running-best tracker is per-metric; we keep them as static atomics
 * because the trigger is a thread-shared pre-LAMMPS gate, not a per-host
 * field. */
namespace bad_pot_detail {
  /* One running best per metric. infinity = nothing seen yet. */
  inline std::atomic<double>& best_for(const char* tag) {
    static std::atomic<double> b_emae   {std::numeric_limits<double>::infinity()};
    static std::atomic<double> b_ermse  {std::numeric_limits<double>::infinity()};
    static std::atomic<double> b_errmse {std::numeric_limits<double>::infinity()};
    static std::atomic<double> b_ersq   {std::numeric_limits<double>::infinity()};
    static std::atomic<double> b_frmse  {std::numeric_limits<double>::infinity()};
    static std::atomic<double> b_srmse  {std::numeric_limits<double>::infinity()};
    /* Hand-rolled dispatch -- avoids string compare in the hot path's
     * fallback (the call site already has a compile-time tag). */
    if (tag[0]=='e' && tag[1]=='m')           return b_emae;
    if (tag[0]=='e' && tag[1]=='r' && tag[2]=='m') return b_ermse;
    if (tag[0]=='e' && tag[1]=='R')           return b_errmse;
    if (tag[0]=='e' && tag[1]=='r' && tag[2]=='s') return b_ersq;
    if (tag[0]=='f')                          return b_frmse;
    return b_srmse;
  }
  inline std::atomic<std::size_t>& skip_count() {
    static std::atomic<std::size_t> c{0};
    return c;
  }
} // namespace bad_pot_detail

/* ----------  per-script lattice cache (warm start) -------------------
 *
 * The warmstart namespace lives in warmstart_cache.h so the WARM init
 * strategy (init_strategies.cpp) can prime the cache from `# HOT_A0`
 * lines saved into best_pot.tadah without having to include this whole
 * template-heavy header.
 *
 * `aborting_iter`, `last_rejected_iter`, `reject_count` are also in
 * warmstart_cache.h since they participate in the same per-eval
 * rejection bookkeeping. */


/* -----------------  header writers  ---------------------------------- */
template<class P,class R>
void HPO_Host<P,R>::init_loss(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;
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

  for (const auto &lsim : trg_.LS) {
    for (auto varname : lsim.n_varloss) {
      if (varname.size() > max_title_len_)
        varname = varname.substr(0, max_title_len_ - 1) + "*";
      f << std::setw(cw_) << "L("+varname+")";
    }
  }

  /* PC_OBSERVABLE columns: one L(<kind>[k]) per observable. The k
   * index disambiguates multiple entries of the same kind (e.g. two
   * cohesive_energy observables for different species). */
  for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
    std::string label = "L(" + trg_.observables[k]->kind() + "["
                      + std::to_string(k) + "])";
    if (label.size() > max_title_len_)
      label = label.substr(0, max_title_len_ - 1) + "*";
    f << std::setw(cw_) << label;
  }

  f << std::setw(cw_) << "TOTAL_LOSS/N" << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_outvar(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(iter_cw_) << "#ITERATION";
  f << std::setw(ittime_cw_) << "IT-TIME/ms" << std::right;
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

  for (const auto &lsim : trg_.LS) {
    for (auto n_outvar : lsim.n_outvar) {
      if (n_outvar.size() > max_title_len_) {
        n_outvar = n_outvar.substr(0, max_title_len_ - 1) + "*";
      }
      f << std::setw(cw_) << n_outvar;
    }
  }
  /* PC_OBSERVABLE values: <kind>[k] for each observable. */
  for (std::size_t k = 0; k < trg_.observables.size(); ++k) {
    std::string label = trg_.observables[k]->kind() + "["
                      + std::to_string(k) + "]";
    if (label.size() > max_title_len_)
      label = label.substr(0, max_title_len_ - 1) + "*";
    f << std::setw(cw_) << label;
  }
  f << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_params(const std::string& fn) const
{

  std::ofstream f(fn);
  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;

  for (const auto &p : trg_.optimap) {
    for (size_t i = 0; i < p.second.opt.size(); ++i) {
      if (p.second.opt[i]) {
        std::string name = p.first;
        if (name.size() > max_title_len_)
          name = name.substr(0, max_title_len_ - 1) + "*";
        f << std::setw(cw_) << name;
      }
    }
  }
  f << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_stats(const std::string& fn) const
{
  std::ofstream f(fn);
  f << std::left << std::setw(iter_cw_) << "#ITERATION" << std::right;

  auto write_header = [&](const std::string& label) {
    for (std::size_t i = 1; i <= n_val_datasets_; ++i) {
      std::string col = label + "[" + std::to_string(i) + "]";
      if (col.size() > max_title_len_)
        col = col.substr(0, max_title_len_ - 1) + "*";
      f << std::setw(cw_) << col;
    }
    std::string avg_col = label + "[avg]";
    if (avg_col.size() > max_title_len_)
      avg_col = avg_col.substr(0, max_title_len_ - 1) + "*";
    f << std::setw(cw_) << avg_col;
  };

  if (trg_.use_emae)   write_header("EMAE");
  if (trg_.use_ermse)  write_header("ERMSE");
  if (trg_.use_errmse) write_header("ErRMSE");
  if (trg_.use_ersq)   write_header("ERSQ");
  if (trg_.use_frmse)  write_header("FRMSE");
  if (trg_.use_srmse)  write_header("SRMSE");
  f << std::endl;
}

/* ----------  line printers (append) ---------------------------------- */
/* ======= line writers ================================================ */
template<class P,class R>
void HPO_Host<P,R>::print_loss_line(std::ofstream& f, size_t it,const Result& e,
                                    const HPOTargets& trg,
                                    std::size_t cw,int prec)
{
    f << std::left << std::setw(iter_cw_) << it;
    f << std::right << std::scientific << std::setprecision(prec);

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
  f << std::left << std::setw(iter_cw_) << it;
  f << std::right << std::scientific << std::setprecision(prec);

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
#if defined(TADAH_BUILD_MPI)
template<class P, class R>
std::mutex          HPO_Host<P,R>::mpi_mutex_;
#endif

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
template<class P,class R> std::atomic_size_t
  HPO_Host<P,R>::iter_at_last_best_{0};
template<class P,class R> std::atomic_size_t
  HPO_Host<P,R>::stagnation_threshold_{0};

/* Top-K accessors. Cheap when disabled. */
template<class P,class R>
void HPO_Host<P,R>::set_track_topk(std::size_t k) {
  std::lock_guard<std::mutex> lk(topk_mtx_);
  topk_capacity_ = k;
  topk_buf_.clear();
  topk_buf_.reserve(k);
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
void HPO_Host<P,R>::set_g_iter_floor(std::size_t n) {
  std::size_t cur = g_iter_.load(std::memory_order_relaxed);
  while (n > cur) {
    if (g_iter_.compare_exchange_weak(cur, n,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) break;
  }
  /* Bug-7: writer's `next_flush_` is a static atomic default-init to 1.
   * On resume it would otherwise stay at 1 -- the new io_buf_ starts empty
   * and the resumed run's first eval submits at iter > n, so the writer's
   * gap-skip timer fires for 60s before advancing. The "8499 hung iter
   * slots" warning is misleading because the per-eval logs from the prior
   * run are intact on disk. Push next_flush_ past the resumed iter so
   * the writer starts where the prior run left off. */
  std::size_t cur_nf = next_flush_.load(std::memory_order_relaxed);
  while (n + 1 > cur_nf) {
    if (next_flush_.compare_exchange_weak(cur_nf, n + 1,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed)) break;
  }
  /* Reset gap-skip timer too so it doesn't carry stale state. */
  next_flush_stuck_since_ms_.store(0, std::memory_order_relaxed);
}

template<class P,class R>
std::size_t HPO_Host<P,R>::g_iter_value() {
  return g_iter_.load(std::memory_order_relaxed);
}

/* ---------- ICheckpointable: persist best-so-far comparator state ----- */
template<class P,class R>
void HPO_Host<P,R>::save_globals(core::Checkpoint& cp) {
    cp.section("HPO_HOST");
    cp.put_dbl ("g_best",            g_best_.load(std::memory_order_relaxed));
    cp.put_size("g_best_iter",       g_best_iter_.load(std::memory_order_relaxed));
    cp.put_size("iter_at_last_best", iter_at_last_best_.load(std::memory_order_relaxed));
    cp.put_size("g_iter",            g_iter_.load(std::memory_order_relaxed));
}

template<class P,class R>
void HPO_Host<P,R>::load_globals(const core::Checkpoint& cp) {
    if (auto v = cp.get_dbl ("HPO_HOST", "g_best"))
        g_best_.store(*v, std::memory_order_relaxed);
    if (auto v = cp.get_size("HPO_HOST", "g_best_iter"))
        g_best_iter_.store(*v, std::memory_order_relaxed);
    if (auto v = cp.get_size("HPO_HOST", "iter_at_last_best"))
        iter_at_last_best_.store(*v, std::memory_order_relaxed);
    if (auto v = cp.get_size("HPO_HOST", "g_iter"))
        g_iter_.store(*v, std::memory_order_relaxed);
}

template<class P,class R>
void HPO_Host<P,R>::save(core::Checkpoint& cp) const { save_globals(cp); }

template<class P,class R>
void HPO_Host<P,R>::load(const core::Checkpoint& cp) { load_globals(cp); }

template<class P,class R>
void HPO_Host<P,R>::reset_globals_for_test() {
    g_best_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
    g_best_iter_.store(0, std::memory_order_relaxed);
    iter_at_last_best_.store(0, std::memory_order_relaxed);
    g_iter_.store(0, std::memory_order_relaxed);
}

template<class P, class R>
thread_local tadah::mlip::memory::MLIPWorkspaceManager
HPO_Host<P,R>::tls_ws_{};
#if defined(TADAH_BUILD_MPI)
template<class P, class R>
thread_local tadah::mlip::memory::MLIPWorkspaceManager
HPO_Host<P,R>::tls_ws2_{};
#endif

#ifndef TADAH_BUILD_MPI
template<class P, class R>
thread_local typename HPO_Host<P,R>::TrainerCache
HPO_Host<P,R>::tls_trainer_cache_{};
#endif
#if defined(TADAH_BUILD_MPI)
template<class P, class R>
thread_local typename HPO_Host<P,R>::TrainerHostCache
HPO_Host<P,R>::tls_trainer_host_cache_{};
#endif

/* ----------  constructor ---------------------------------------------- */
template<class P, class R>
HPO_Host<P,R>::HPO_Host(core::Context master_ctx,
                        HPOTargets&   trg,
                        const std::string& val_cfg)
: master_ctx_(std::move(master_ctx))
, trg_(trg), spec_(trg.spec)
, use_lammps_( !trg.LS.empty() )
, has_train_data_( master_ctx_.contains("DBFILE")
                   && master_ctx_.size("DBFILE") > 0 )
, has_validation_( !val_cfg.empty() )
, prec_(trg.outputDigits)
, cw_(static_cast<std::size_t>(std::max(prec_ + 9, static_cast<int>(min_cw_))))
, max_title_len_(cw_ - 4)
{

    /* knot channels: RCUT keys are derived from the largest knot (see
     * rcut_derivation.h). Flags are computed once here because the per-eval
     * hook lives in the const update_context(). */
    knot2b_ = models::potedit::derived_knot_rcut(master_ctx_, "2B").has_value();
    knotmb_ = models::potedit::derived_knot_rcut(master_ctx_, "MB").has_value();

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
        val_stdb_.add(val_ctx);
        n_val_datasets_ = val_stdb_.dbidx.size() > 1 ? val_stdb_.dbidx.size() - 1 : 1;
    }
#ifndef TADAH_BUILD_MPI
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
     * instead (same policy as the MPI host below). */
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

        nnf_.calc(val_stdb_, rcutmax);
    }
    if (has_train_data_) nnf_.calc(train_stdb_, rcutmax);
#else
    /* MPI host: training stdb lives on workers; validation stdb lives here.
     * Global-view transforms (ESHIFT_ATOM, ESHIFT_DBATOM, EWEIGHT_TEMP,
     * WDBFILE_AUTO) are not yet wired through TrainerHost; we rely on the
     * fail-fast guard in TrainEngine for those. Explicit ESHIFT/LSCALE on
     * validation can still be applied. EFILTER/FFILTER from the --validation
     * file itself are also forwarded; master_ctx_ filters are not. */
    if (has_validation_) {
        core::Context vt;
        if (master_ctx_.contains("LSCALE"))
            vt.add("LSCALE", master_ctx_.get<std::string>("LSCALE"));
        if (master_ctx_.contains("ESHIFT") && master_ctx_.size("ESHIFT") > 0) {
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

        nnf_.calc(val_stdb_, rcutmax);
    }
#endif

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

    /* build param labels ------------------------------------------------ */
    for (const auto& kv : trg_.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            std::string lbl = kv.first;
            if (kv.second.opt.size() > 1) lbl += "[" + std::to_string(i) + "]";
            param_labels_.push_back(lbl);
        }
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
#ifndef TADAH_BUILD_MPI
    return train_stdb_.size();
#else
    return 0; // not available on MPI host rank
#endif
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
    for (const auto& kv : trg_.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            if (!first) s << "  ";
            first = false;
            s << kv.first;
            if (kv.second.opt.size() > 1) s << "[" << i << "]";
            s << "=" << std::scientific << std::setprecision(3) << p(idx++);
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
    for (const auto& kv : trg_.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            if (idx >= sp.low.size()) break;
            const double v   = p(idx);
            const double lo  = sp.low[idx];
            const double hi  = sp.high[idx];
            const double rng = hi - lo;
            std::string name = kv.first;
            if (kv.second.opt.size() > 1) name += "[" + std::to_string(i) + "]";
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
 * Two-phase execution when any PC_LAMMPS line carries `--hotvar`:
 *
 *   PHASE A   Run hot scripts in parallel. Each hot script reads its
 *             warmstart-cached value (or the user-declared `initial`,
 *             midpoint of [min, max] on first eval) as `${<name>}`,
 *             relaxes, exposes a fresh `<name>` value. The framework
 *             validates against [min, max], updates the cache, and
 *             collects (name -> fresh value) into `hot_values_for_iter`.
 *             An out-of-range value sets the eval-wide abort flag.
 *
 *   PHASE B   Run cold scripts in parallel. If phase A aborted, all
 *             cold outputs are clamped to failscore. Otherwise every
 *             value collected in phase A is injected as
 *             `variable <name> string <value>` before each cold script,
 *             so cold scripts read `${<name>}` to get the freshly
 *             computed hot value from THIS eval.
 *
 * No --hotvar lines means phase A is empty; phase B runs all scripts in
 * parallel as before. */
template<class P, class R>
void HPO_Host<P,R>::run_lammps(Result& r) const
{
  if(!use_lammps_) return;
  const auto& LS = trg_.LS;
  r.v_outvar.resize(LS.size());

  /* Partition scripts into hot (any --hotvar) and cold (none). The hot
   * set runs first so its produced values can flow into the cold set. */
  std::vector<std::size_t> hot_idx, cold_idx;
  hot_idx.reserve(LS.size());
  cold_idx.reserve(LS.size());
  for (std::size_t i = 0; i < LS.size(); ++i) {
    if (LS[i].hotvars.empty()) cold_idx.push_back(i);
    else                       hot_idx.push_back(i);
  }

  /* Per-iter map of fresh hot values (varname -> value). Populated in
   * phase A under hot_mtx_, then read-only during phase B. */
  std::unordered_map<std::string, double> hot_values_for_iter;
  std::mutex hot_mtx;

  /* Generic per-script runner -- used by both phases. Returns true on
   * a successful LAMMPS run; false on script failure (in which case
   * v_outvar is filled with failscore). The `extra_inject` callable
   * is invoked once after the LAMMPS instance is ready and before
   * `run_script`, with the raw lmp pointer; used by both phases to
   * inject `variable <name> string <value>` lines. */
  auto run_one = [&](std::size_t i,
                     auto&& extra_inject) -> bool {
    const auto& lsim = LS[i];
    /* Per-iteration copy of LAMMPS argv. -log / -screen with a real
     * filename get a .t<tid> suffix so concurrent OMP threads don't
     * clobber the same file. */
    std::vector<std::string> invar = lsim.invar;
#ifdef TADAH_ENABLE_OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto rewrite = [&](const char* flag) {
      for (std::size_t k = 0; k + 1 < invar.size(); ++k) {
        if (invar[k] == flag && invar[k+1] != "none") {
          invar[k+1] += ".t" + std::to_string(tid);
          break;
        }
      }
    };
    rewrite("-log");
    rewrite("-screen");

    /* Per-thread LAMMPS instance cache, keyed by (script + invar).
     * The key MUST include invar: the same script with different
     * --invar values needs distinct LMP instances (observed
     * deterministically on 2026-05-05; reusing one LMP returned ""
     * for varloss vars on the second-and-later call). */
    std::string cache_key = lsim.script;
    for (const auto& s : invar) { cache_key += '\x01'; cache_key += s; }
    thread_local std::unordered_map<std::string, typename R::LMPPtr> tls_lmp_cache;
    R runner;
    auto t_create0 = std::chrono::high_resolution_clock::now();
    auto cache_it = tls_lmp_cache.find(cache_key);
    if (cache_it == tls_lmp_cache.end()) {
      auto fresh = runner.create_instance(0, invar);
      if (!fresh) {
        for (const auto& n : lsim.n_outvar) {
          (void)n;
          r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
        }
        return false;
      }
      cache_it = tls_lmp_cache.emplace(cache_key, std::move(fresh)).first;
    }
    auto& lmp = cache_it->second;
    auto t_create1 = std::chrono::high_resolution_clock::now();

    /* Phase-specific injections happen here. */
    extra_inject(lmp);

    bool fail = !runner.run_script(lmp, lsim.script_lines, r.pot_ctx);
    auto t_run1 = std::chrono::high_resolution_clock::now();

    for (const auto& n : lsim.n_outvar) {
      if (fail) r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      else      r.v_outvar[i].emplace_back(runner.get_variable(lmp, n));
    }

    if (fail) {
      warmstart::aborting_iter().store(r.iter, std::memory_order_relaxed);
    }

    /* Hot-var validation + cache update. Only meaningful for hot
     * scripts (cold scripts have empty hotvars). */
    if (!fail) {
      for (const auto& hv : lsim.hotvars) {
        try {
          const std::string s = runner.get_variable(lmp, hv.name);
          if (s.empty()) continue;
          double v = 0.0;
          try { v = std::stod(s); }
          catch (...) { continue; }
          if (!std::isfinite(v) || v < hv.min || v > hv.max) {
            /* Out of range -- clamp this script's outputs and signal
             * the iter-wide abort so phase B is skipped. */
            for (auto& vs : r.v_outvar[i]) vs = std::to_string(lsim.failscore);
            warmstart::aborting_iter().store(r.iter, std::memory_order_relaxed);
            auto& last = warmstart::last_rejected_iter();
            std::size_t prev = last.load(std::memory_order_relaxed);
            bool first_in_eval = false;
            while (prev < r.iter) {
              if (last.compare_exchange_weak(prev, r.iter,
                                             std::memory_order_relaxed)) {
                first_in_eval = true;
                break;
              }
            }
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
          /* In-range -- cache for next eval AND record for phase B. */
          warmstart::set(lsim.script, hv.name, v);
          {
            std::lock_guard<std::mutex> lg(hot_mtx);
            hot_values_for_iter[hv.name] = v;
          }
        } catch (...) { /* variable not defined or unparseable -- ignore */ }
      }
    }

    if (bench_detail::enabled()) {
      const double t_create_ms = bench_detail::elapsed_ms(t_create0, t_create1);
      const double t_run_ms    = bench_detail::elapsed_ms(t_create1, t_run1);
      std::lock_guard<std::mutex> lg(bench_detail::mtx());
      static std::ofstream f("hpo_script_times.csv", std::ios::out | std::ios::trunc);
      static bool header_written = (f << "iter,thread,script,create_ms,run_ms\n", true);
      (void)header_written;
      f << r.iter << ',' << tid << ',' << lsim.script << ','
        << t_create_ms << ',' << t_run_ms << '\n';
      f.flush();
    }
    return true;
  };

  auto inject_hot_value = [](typename R::LMPPtr& lmp,
                             const std::string& name, double value) {
    /* Inject as `equal`-style (a constant expression) rather than
     * `string` so that a hot script ending in
     *   variable <name> equal v_lx/<divisor>
     * is a *same-style* redefinition -- LAMMPS allows that without a
     * prior `variable delete`. HPO caches one LAMMPS instance per
     * (script, invar) tuple across iters (see line 970), and LAMMPS
     * would reject a style change inside that instance, so keeping
     * both ends `equal`-style makes the warmstart-cache cycle work
     * without forcing scripts to insert delete boilerplate. */
    char cmd[256];
    std::snprintf(cmd, sizeof(cmd),
                  "variable %s equal %.17g", name.c_str(), value);
    try { lmp->input->one(cmd); } catch (...) { /* keep going */ }
  };

  /* --- PHASE A: hot scripts --------------------------------------- */
#ifdef TADAH_ENABLE_OPENMP
#pragma omp parallel for shared(r,LS,hot_idx,hot_values_for_iter,hot_mtx)
#endif
  for (std::size_t k = 0; k < hot_idx.size(); ++k) {
    const std::size_t i = hot_idx[k];
    if (warmstart::aborting_iter().load(std::memory_order_relaxed) == r.iter) {
      for (const auto& n : LS[i].n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
      }
      continue;
    }
    run_one(i, [&](typename R::LMPPtr& lmp) {
      /* Seed each declared hot var with its cached or initial value. */
      for (const auto& hv : LS[i].hotvars) {
        double seed;
        if (auto cached = warmstart::get(LS[i].script, hv.name)) {
          seed = *cached;
        } else if (hv.initial) {
          seed = *hv.initial;
        } else {
          seed = 0.5 * (hv.min + hv.max);
        }
        inject_hot_value(lmp, hv.name, seed);
      }
    });
  }

  /* --- PHASE B: cold scripts -------------------------------------- */
  const bool aborted =
      (warmstart::aborting_iter().load(std::memory_order_relaxed) == r.iter);

  if (aborted) {
    /* Phase A produced an out-of-range hot var (or a script crashed) --
     * the potential is unphysical. Skip every cold script and clamp. */
    for (auto i : cold_idx) {
      for (const auto& n : LS[i].n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
      }
    }
    return;
  }

  /* Snapshot the per-iter hot-values map for use under #pragma omp
   * parallel -- once phase A is done, no more writes happen. */
  std::unordered_map<std::string, double> hot_snapshot;
  {
    std::lock_guard<std::mutex> lg(hot_mtx);
    hot_snapshot = hot_values_for_iter;
  }

#ifdef TADAH_ENABLE_OPENMP
#pragma omp parallel for shared(r,LS,cold_idx,hot_snapshot)
#endif
  for (std::size_t k = 0; k < cold_idx.size(); ++k) {
    const std::size_t i = cold_idx[k];
    if (warmstart::aborting_iter().load(std::memory_order_relaxed) == r.iter) {
      for (const auto& n : LS[i].n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(LS[i].failscore));
      }
      continue;
    }
    run_one(i, [&](typename R::LMPPtr& lmp) {
      /* Inject every fresh hot value from phase A. Cold scripts read
       * them as `${<name>}`. */
      for (const auto& kv : hot_snapshot) {
        inject_hot_value(lmp, kv.first, kv.second);
      }
    });
  }
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
  if (trg_.use_ermse)  add_global(r.ermseLoss,  r.ermse,  trg_.ermse,  trg_.w_ermse,  r.v_ermseLoss,  r.v_ermse);
  if (trg_.use_errmse) add_global(r.errmseLoss, r.errmse, trg_.errmse, trg_.w_errmse, r.v_errmseLoss, r.v_errmse);
  if (trg_.use_ersq)   add_global(r.ersqLoss,   r.ersq,   trg_.ersq,   trg_.w_ersq,   r.v_ersqLoss,   r.v_ersq);
  if (trg_.use_frmse)  add_global(r.frmseLoss,  r.frmse,  trg_.frmse,  trg_.w_frmse,  r.v_frmseLoss,  r.v_frmse);
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
      if (try_parse_double(r.v_outvar[i][j], v)) {
        l = lsim.lossFn[k].loss(v, lsim.t_varloss[k], w);
      } else {
        l = trg_.FAIL_SCORE;
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

#if defined(TADAH_BUILD_MPI)
  auto& cache = tls_trainer_host_cache_;
#else
  auto& cache = tls_trainer_cache_;
#endif

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

/* ----------  operator()  (thread-safe) ------------------------------- */
template<class P, class R>
double HPO_Host<P,R>::operator()(const P& p) const
{
  /* Cooperative shutdown: SIGINT/SIGTERM/SIGUSR2 raised the stop flag.
   * Throw before consuming an iter slot so the caller's run loop
   * (RestartPolicy / BasinsPolicy / hpo_run) can flush a checkpoint
   * and exit cleanly. SignalHandler::install() is wired in hpo_run(). */
  if (core::SignalHandler::stop_requested()) {
    throw core::ShutdownRequested{};
  }

  Result r;
  r.iter   = ++g_iter_;
    r.params = p;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_train_start = t0, t_train_end = t0, t_predict_end = t0,
         t_lammps_end = t0, t_sims_loss_end = t0, t_submit_end = t0;

    /* Per-stage watchdog logger -- closes over r.iter so the post-hoc
     * leave() warning includes the iter that was running. */
    auto _stage_log = [](const std::string& msg) {
        HPOLog::instance().warning(msg);
    };

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
     *    run shuts down cleanly with a final checkpoint.
     *  - StagnationStop -- must propagate to RestartPolicy. */
    try {

    /* build context ---------------------------------------------------- */
    core::Context ctx = master_ctx_;
    update_context(ctx, p);

    t_train_start = std::chrono::high_resolution_clock::now();
    /* train ------------------------------------------------------------ */
    stage_watchdog::enter(r.iter, "train");
#if defined(TADAH_BUILD_MPI)
    int rank;
    int ncpu;

    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // New context file must be communicated to all processes
    std::vector<char> cs = ctx.serialize();
    int cs_size = cs.size();
    MPI_Bcast(&cs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cs.data(), cs_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::lock_guard<std::mutex> lock(mpi_mutex_);     // serialise MPI
    {
        auto& cache = tls_trainer_host_cache_;
        if (!cache.trainer) {
            cache.trainer.reset(new (cache.buf) mlip::TrainerHost(ctx, tls_ws_, tls_ws2_));
        } else {
            bool needs_rebuild = false;
            try {
                update_context(cache.trainer->context, p);
                needs_rebuild = cache.trainer->reinit();
            } catch (const std::exception& e) {
                HPOLog::instance().warning(
                    std::string("reinit() threw (eval ") + std::to_string(r.iter)
                    + "): " + e.what() + " -- rebuilding Trainer");
                needs_rebuild = true;
            }
            if (needs_rebuild) {
                cache.trainer.reset();
                cache.trainer.reset(new (cache.buf) mlip::TrainerHost(ctx, tls_ws_, tls_ws2_));
            }
        }
        cache.trainer->run();
        cache.trainer->solve();
    }
    mlip::TrainerHost& host = *tls_trainer_host_cache_.trainer;
#else
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
#endif
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

    /* Bad-potential pre-filter. Each per-metric --skip-above / --skip-rel
     * cap (declared on PC_* lines in config.hpo) is tested independently;
     * the first trigger wins. Caps are optional -- a metric without flags
     * never causes a skip. The metric -> tag mapping passed to
     * bad_pot_detail::best_for() is a 2-3 char prefix that disambiguates
     * the running-best slot for that metric.
     *
     * Runs before BOTH obs_loss and run_lammps so the gate gates every
     * downstream phase, not just LAMMPS. Was previously wrapped in
     * `if (use_lammps_)` -- silently no-op on PC_OBSERVABLE-only configs
     * which is a usability footgun (the parser accepts --skip-above on
     * a PC_OBSERVABLE-only run but nothing acts on it). */
    bool skip_bad_pot = false;
    {
      auto check_metric = [&](const char* tag,
                              bool used,
                              double value,
                              const HPOTargets::SkipCaps& caps,
                              const char* unit) -> bool {
        if (!used || !std::isfinite(value)) return false;
        const double best = bad_pot_detail::best_for(tag).load(std::memory_order_relaxed);
        const bool bad_abs = caps.abs_above && (value > *caps.abs_above);
        const bool bad_rel = caps.rel_above && std::isfinite(best)
                             && (value > *caps.rel_above * best);
        if (bad_abs || bad_rel) {
          const auto skips = bad_pot_detail::skip_count().fetch_add(
              1, std::memory_order_relaxed) + 1;
          if (skips <= 5 || (skips & (skips - 1)) == 0) {
            std::ostringstream os;
            os << "eval " << r.iter << ": skipping downstream (obs+LAMMPS) -- "
               << tag << "=" << value << " " << unit;
            if (bad_abs) os << " > skip-above=" << *caps.abs_above;
            if (bad_rel) os << " > skip-rel=" << *caps.rel_above
                            << " * best(" << best << ")";
            os << ", total skips=" << skips;
            HPOLog::instance().warning(os.str());
          }
          return true;
        }
        /* Update per-metric running best -- relaxed CAS, no ordering needed. */
        double cur = bad_pot_detail::best_for(tag).load(std::memory_order_relaxed);
        while (value < cur &&
               !bad_pot_detail::best_for(tag).compare_exchange_weak(
                   cur, value, std::memory_order_relaxed)) {}
        return false;
      };
      skip_bad_pot =
          check_metric("ermse",   trg_.use_ermse,   r.ermse,   trg_.skip_ermse,   "meV/at") ||
          check_metric("emae",    trg_.use_emae,    r.emae,    trg_.skip_emae,    "meV/at") ||
          check_metric("eRRMSE",  trg_.use_errmse,  r.errmse,  trg_.skip_errmse,  "meV/at") ||
          check_metric("ersq",    trg_.use_ersq,    r.ersq,    trg_.skip_ersq,    "")       ||
          check_metric("frmse",   trg_.use_frmse,   r.frmse,   trg_.skip_frmse,   "eV/A")   ||
          check_metric("srmse",   trg_.use_srmse,   r.srmse,   trg_.skip_srmse,   "GPa");
      if (skip_bad_pot) r.loss = trg_.FAIL_SCORE;
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
#ifdef TADAH_BUILD_MPI
      run_lammps_mpi(r);
#else
      run_lammps(r);
#endif
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
    /* best tracking ---------------------------------------------------- */
    double cur=g_best_.load(std::memory_order_relaxed);
    bool i_am_best = false;
    while(r.loss < cur &&
        !(i_am_best =
          g_best_.compare_exchange_weak(cur, r.loss,
            std::memory_order_acq_rel)));
    if(i_am_best){
      r.is_best   = true;
      r.best_iter = g_best_iter_.fetch_add(1, std::memory_order_relaxed) + 1;
      iter_at_last_best_.store(r.iter, std::memory_order_relaxed);
    }

    /* Top-K tracking -- cheap fast-path when disabled (capacity == 0). */
    if (topk_capacity_ > 0) {
      std::lock_guard<std::mutex> lk(topk_mtx_);
      if (topk_capacity_ > 0) {  /* re-check inside lock */
        if (topk_buf_.size() < topk_capacity_) {
          topk_buf_.emplace_back(r.loss, r.params);
        } else {
          /* Replace the worst entry if this one is better. */
          auto worst = std::max_element(topk_buf_.begin(), topk_buf_.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
          if (r.loss < worst->first) {
            *worst = std::make_pair(r.loss, r.params);
          }
        }
      }
    }

    /* Stagnation check -- throw to short-circuit the optimiser loop. The
     * surrounding RestartPolicy catches StagnationStop and decides
     * whether to restart from a fresh init. */
    {
      const std::size_t threshold =
          stagnation_threshold_.load(std::memory_order_relaxed);
      if (threshold > 0) {
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
    const double      save_emae    = r.emae,   save_ermse  = r.ermse;
    const double      save_errmse  = r.errmse, save_frmse  = r.frmse;
    const double      save_srmse   = r.srmse;

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
#ifndef TADAH_BUILD_MPI
    submit_result(std::move(r), trg_, master_ctx_, val_ctx_, train_stdb_, val_stdb_, param_labels_, report_, cw_, prec_);
#else
    {
      static const mlip::StructureDB empty_stdb;
      submit_result(std::move(r), trg_, master_ctx_, val_ctx_, empty_stdb, val_stdb_, param_labels_, report_, cw_, prec_);
    }
#endif
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

    if (save_best)
        log.log_new_best(save_iter, save_loss,
                         save_emae,   trg_.use_emae,
                         save_ermse,  trg_.use_ermse,
                         save_errmse, trg_.use_errmse,
                         save_frmse,  trg_.use_frmse,
                         save_srmse,  trg_.use_srmse,
                         save_wall_ms,
                         save_top_breakdown);

    if (save_nan)
        log.log_nan_clamped(save_iter,
                            format_params_str(p),
                            bound_hints(p));

    /* Drive the time-/eval-based checkpoint cadence. Stage-boundary
     * commits in BasinsPolicy are still authoritative; this fills the
     * gap inside long stages so CHECKPOINT_EVERY actually has effect.
     * Non-BASINS runs configure no period, making this a no-op. */
    core::CheckpointManager::instance().tick_eval(save_iter);
    }

    return save_loss;

    } catch (const core::ShutdownRequested&) {
        /* Cooperative shutdown -- pass through to hpo_run()'s outer
         * catch which performs the final flush + checkpoint commit.
         *
         * Bug 7: before rethrowing, submit a synthetic FAIL_SCORE
         * Result so the iter slot we consumed at `++g_iter_` does not
         * leak from the writer queue. Without this, the writer's
         * next_flush_ would be permanently stuck at our iter (the
         * gap-skip recovery would eventually advance it after 60 s,
         * but by then per-eval logs are lost). */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
#ifndef TADAH_BUILD_MPI
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#else
            static const mlip::StructureDB empty_stdb;
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          empty_stdb, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#endif
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
#ifndef TADAH_BUILD_MPI
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#else
            static const mlip::StructureDB empty_stdb;
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          empty_stdb, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#endif
        } catch (...) {}
        throw;
    } catch (const std::exception& e) {
        /* Any other throw between ++g_iter_ and submit_result must
         * not leak the iter slot. Build a synthetic FAIL_SCORE Result
         * carrying r.iter and submit it so the writer's next_flush_
         * cursor can advance past this iter. */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
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
#ifndef TADAH_BUILD_MPI
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#else
            static const mlip::StructureDB empty_stdb;
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          empty_stdb, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#endif
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
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            HPOLog::instance().warning(
                std::string("eval ") + std::to_string(r.iter) +
                ": eval-thread caught unknown throw, submitting "
                "FAIL_SCORE to keep writer queue contiguous");
        } catch (...) {}
        try {
#ifndef TADAH_BUILD_MPI
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          train_stdb_, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#else
            static const mlip::StructureDB empty_stdb;
            submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                          empty_stdb, val_stdb_, param_labels_, report_,
                          cw_, prec_);
#endif
        } catch (...) {}
        return trg_.FAIL_SCORE;
    }
}

/* ---------- opportunistic writer helper ------------------------------ */
  template<class P,class R>
void HPO_Host<P,R>::submit_result(Result&& res, const HPOTargets& trg,
                                  const core::Context& master_ctx,
                                  const core::Context& val_ctx,
                                  const mlip::StructureDB& train_stdb,
                                  const mlip::StructureDB& val_stdb,
                                  const std::vector<std::string>& param_labels,
                                  const mlip::LoadTransformReport& report,
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
     * subsequent submit_result and corrupt best_pot.tadah while
     * hpo_run.log keeps reporting better losses. See WRITER_THREAD_BUG.md. */
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

        static std::ofstream loss_f  (trg.lossFile, std::ios::app),
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
          if(trg.bestOutputEvery > 0) {
            print_loss_line  (best_f      , e.best_iter, e, trg, cw, prec);
            print_outvar_line(best_outv_f , e.best_iter, e, trg, cw, prec);
            print_params_line(best_par_f  , e.best_iter, e,      cw, prec);
            print_stats_line (best_stats_f, e.best_iter, e, trg, cw, prec);
          }

          /* overwrite best_pot.tadah ---------------------------------- */
          {
            std::ofstream bp("best_pot.tadah");

            PotHeaderWriter::write_best_header(
                bp, e, trg, master_ctx,
                val_ctx,
                train_stdb, val_stdb,
                param_labels, report, cw, prec);

            warmstart::dump_to_stream(bp);

            bp << std::setprecision(
                e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
          }

          /* archive best_pot_N.tadah ---------------------------------- */
          if(e.best_iter % trg.dumpBestPotEvery == 0){
            std::ofstream bpa(std::string(trg.dumpBestPotDir)
                +"/best_pot_" + std::to_string(e.best_iter) + ".tadah");

            PotHeaderWriter::write_best_archive_header(
                bpa, e, trg, cw, prec);

            warmstart::dump_to_stream(bpa);

            bpa << std::setprecision(
                e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
          }
        }

        /* ---- archive ANY pot every ---------------------------------- */
        if(e.iter % trg.dumpPotEvery == 0){
            std::ofstream pa(std::string(trg.dumpPotDir)
                +"/pot_"+std::to_string(e.iter)+".tadah");

            PotHeaderWriter::write_archive_header(pa, e, cw, prec);

            warmstart::dump_to_stream(pa);

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
                 * pot_ctx.get / write_best_header / per-iter print
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
                                if (trg.bestOutputEvery > 0) {
                                    print_loss_line  (best_f      , e.best_iter, e, trg, cw, prec);
                                    print_outvar_line(best_outv_f , e.best_iter, e, trg, cw, prec);
                                    print_params_line(best_par_f  , e.best_iter, e,      cw, prec);
                                    print_stats_line (best_stats_f, e.best_iter, e, trg, cw, prec);
                                }
                                {
                                    std::ofstream bp("best_pot.tadah");
                                    PotHeaderWriter::write_best_header(
                                        bp, e, trg, master_ctx, val_ctx,
                                        train_stdb, val_stdb, param_labels, report, cw, prec);
                                    warmstart::dump_to_stream(bp);
                                    bp << std::setprecision(
                                        e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                                }
                                if (e.best_iter % trg.dumpBestPotEvery == 0) {
                                    std::ofstream bpa(std::string(trg.dumpBestPotDir)
                                        +"/best_pot_" + std::to_string(e.best_iter) + ".tadah");
                                    PotHeaderWriter::write_best_archive_header(bpa, e, trg, cw, prec);
                                    warmstart::dump_to_stream(bpa);
                                    bpa << std::setprecision(
                                        e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                                }
                            }
                            if (e.iter % trg.dumpPotEvery == 0) {
                                std::ofstream pa(std::string(trg.dumpPotDir)
                                    +"/pot_"+std::to_string(e.iter)+".tadah");
                                PotHeaderWriter::write_archive_header(pa, e, cw, prec);
                                warmstart::dump_to_stream(pa);
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

#if defined(TADAH_BUILD_MPI)
    /* MPI build: trainer cache layout differs (TrainerHost). Phase 2A
     * first cut targets the serial / OpenMP path only. */
    return fall_back_all_fd();
#else
    auto& cache = tls_trainer_cache_;
    if (!cache.trainer) {
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
                        if (auto cached = warmstart::get(lsim.script, hv.name)) {
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
                    auto lmp = runner.create_instance(0, lsim.invar);
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

                    if (!runner.run_script(lmp, lsim.script_lines, pot)) {
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
#endif
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

#if defined(TADAH_BUILD_MPI)
    static const mlip::StructureDB empty_stdb;
    const mlip::StructureDB& train_ref = empty_stdb;
#else
    const mlip::StructureDB& train_ref = train_stdb_;
#endif
    const mlip::StructureDB& val_ref = val_stdb_;

    while(io_buf_.count(next_flush_))
    {
        try {
            const auto& e = io_buf_[next_flush_];

            if(next_flush_ % trg_.outputEvery == 0){
                print_loss_line  (loss_f , e.iter, e, trg_, cw_, prec_);
                print_outvar_line(outv_f , e.iter, e, trg_, cw_, prec_);
                print_params_line(par_f  , e.iter, e,       cw_, prec_);
                print_stats_line (stats_f, e.iter, e, trg_, cw_, prec_);
            }
            if(e.is_best && trg_.bestOutputEvery > 0) {
                print_loss_line  (best_f      , e.best_iter, e, trg_, cw_, prec_);
                print_outvar_line(best_outv_f , e.best_iter, e, trg_, cw_, prec_);
                print_params_line(best_par_f  , e.best_iter, e,       cw_, prec_);
                print_stats_line (best_stats_f, e.best_iter, e, trg_, cw_, prec_);
            }

            if(e.is_best) {
                {
                    std::ofstream bp("best_pot.tadah");

                    PotHeaderWriter::write_best_header(
                        bp, e, trg_, master_ctx_,
                        val_ctx_,
                        train_ref, val_ref,
                        param_labels_, report_, cw_, prec_);

                    warmstart::dump_to_stream(bp);

                    bp << std::setprecision(
                        e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                }

                if(e.best_iter % trg_.dumpBestPotEvery == 0){
                    std::ofstream bpa(std::string(trg_.dumpBestPotDir)
                        + "/best_pot_" + std::to_string(e.best_iter) + ".tadah");

                    PotHeaderWriter::write_best_archive_header(
                        bpa, e, trg_, cw_, prec_);

                    warmstart::dump_to_stream(bpa);

                    bpa << std::setprecision(
                        e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
                }
            }

            if(e.iter % trg_.dumpPotEvery == 0){
                std::ofstream pa(std::string(trg_.dumpPotDir)
                    + "/pot_" + std::to_string(e.iter) + ".tadah");

                PotHeaderWriter::write_archive_header(pa, e, cw_, prec_);

                warmstart::dump_to_stream(pa);

                pa << std::setprecision(
                    e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
            }

            io_buf_.erase(next_flush_++);
        } catch (const std::exception& ex) {
            /* Same erase-on-throw as submit_result so the final drain
             * cannot get stuck on a single bad iter. */
            HPOLog::instance().warning(
                std::string("HPO flush_remaining: dropping iter=") +
                std::to_string(next_flush_.load()) +
                " after exception: " + ex.what());
            io_buf_.erase(next_flush_++);
        } catch (...) {
            HPOLog::instance().warning(
                std::string("HPO flush_remaining: dropping iter=") +
                std::to_string(next_flush_.load()) +
                " after unknown exception");
            io_buf_.erase(next_flush_++);
        }
    }
}

#ifdef TADAH_BUILD_MPI
template<class P, class R>
void HPO_Host<P,R>::run_lammps_mpi(Result& r) const {

    if(!use_lammps_) return;

    int rank;
    int ncpu;

    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    const static int CONFIG_TAG = 4;
    const static int WAIT_TAG = 3;
    const static int RELEASE_TAG = 2;
    const static int DATA_TAG = 1;
    const static int WORK_TAG = 0;

    MPI_Status status;

    const auto &wpkgs = trg_.LS;
    // LS is sorted
    std::vector<bool> is_wpkgs_avail(wpkgs.size(), true); // zero index is root

    r.v_outvar.resize(wpkgs.size());

    if (wpkgs[0].ncpu > ncpu - 1) {
      throw std::runtime_error("Requested LAMMPS simulation size is bigger\n\
than the total number of workers.\n");
    }
    std::vector<int> available_workers;
    std::vector<int> distribution(
        ncpu); // This array tracks the jobs assigned to each worker.
    // zero index is root,

    while (true) {
      bool has_wpkgs = std::any_of(is_wpkgs_avail.begin(), is_wpkgs_avail.end(),
                                   [](bool value) { return value; });

      if (!has_wpkgs) {
        break;
      }

      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;
      int worker = status.MPI_SOURCE;

      if (tag == WORK_TAG) {
        MPI_Recv(NULL, 0, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);

        auto g_it = std::find(available_workers.begin(), available_workers.end(),
                            worker);
        if (g_it == available_workers.end()) { // new worker is available
          available_workers.push_back(worker);
          MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);
        } else {
          // release an eager worker as we will select a group of workers here
          MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);

          // There are no more new workers available at the moment.
          // Let's find the biggest job we can fit to them
          for (int j = 0; j < static_cast<int>(wpkgs.size()); ++j) {

            if (!is_wpkgs_avail[j])
              continue;

            int job_size = wpkgs[j].ncpu;
            if (job_size <= static_cast<int>(available_workers.size())) {
              std::vector<int> granks(available_workers.end() - job_size,
                                      available_workers.end());
              for (int w : granks) {
                distribution[w] = j;
                MPI_Recv(NULL, 0, MPI_INT, w, WORK_TAG, MPI_COMM_WORLD,
                         &status);
                MPI_Send(granks.data(), granks.size(), MPI_INT, w,
                         WORK_TAG, MPI_COMM_WORLD);
                available_workers.pop_back();
                // send wpkgs[j].script
                MPI_Send(wpkgs[j].script.c_str(), wpkgs[j].script.size() + 1,
                         MPI_CHAR, w, WORK_TAG, MPI_COMM_WORLD);

                // send wpkgs[j].invar
                // First, send the number of strings
                int num_strings = static_cast<int>(wpkgs[j].invar.size());
                MPI_Send(&num_strings, 1, MPI_INT, w, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].invar) {
                  int length =
                      static_cast<int>(message.size()) + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, w, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, w, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send wpkgs[j].n_outvar
                // First, send the number of strings
                num_strings = static_cast<int>(wpkgs[j].n_outvar.size());
                MPI_Send(&num_strings, 1, MPI_INT, w, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].n_outvar) {
                  int length = static_cast<int>(message.size()) + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, w, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, w, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send wpkgs[j].n_varloss
                // First, send the number of strings
                num_strings = static_cast<int>(wpkgs[j].n_varloss.size());
                MPI_Send(&num_strings, 1, MPI_INT, w, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].n_varloss) {
                  int length = static_cast<int>(message.size()) + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, w, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, w, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send failscore
                double fs = wpkgs[j].failscore;
                MPI_Send(&fs, 1, MPI_DOUBLE, w, WORK_TAG,
                         MPI_COMM_WORLD);
              }
              is_wpkgs_avail[j] = false;
              break; // break outer for loop
            }
          }
        }
      } else if (tag == DATA_TAG) {
        int j = distribution[worker];
        std::vector<double> v_varloss(wpkgs[j].n_varloss.size());
        MPI_Recv(v_varloss.data(), static_cast<int>(v_varloss.size()),
                 MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        r.v_outvar[j].resize(wpkgs[j].n_outvar.size());
        for (std::size_t i = 0; i < wpkgs[j].n_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          char *buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          r.v_outvar[j][i] = std::string(buffer);
          delete[] buffer;
        }
      }
    }
    int count2 = 1;
    while (true) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;
      int worker = status.MPI_SOURCE;

      if (tag == WORK_TAG) {
        MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(NULL, 0, MPI_INT, worker, RELEASE_TAG, MPI_COMM_WORLD);
        count2++;
      } else if (tag == DATA_TAG) {
        int j = distribution[worker];
        std::vector<double> v_varloss(wpkgs[j].n_varloss.size());
        MPI_Recv(v_varloss.data(), static_cast<int>(v_varloss.size()),
                 MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        r.v_outvar[j].resize(wpkgs[j].n_outvar.size());
        for (std::size_t i = 0; i < wpkgs[j].n_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          char *buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          r.v_outvar[j][i] = std::string(buffer);
          delete[] buffer;
        }
      }

      if (count2 == ncpu)
        break;
    }
}
#endif
}   // namespace tadah::hpo

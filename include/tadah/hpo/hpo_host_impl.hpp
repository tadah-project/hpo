#pragma once
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/val_filter_forwarding.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/pot_header_writer_impl.hpp>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/hpo/iter_health.h>
#include <fstream>
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
            std::to_string(thr) + " ms) — possible hang surface");
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
 * trained potential is unphysical. Two configurable knobs:
 *   TADAH_HPO_BAD_POT_ERMSE_MEV   absolute energy-RMSE cap (default 200)
 *   TADAH_HPO_BAD_POT_ERMSE_REL   relative cap vs best seen so far
 *                                 (default 0 = disabled). If best seen
 *                                 is finite, current eval is rejected when
 *                                 ermse > REL * best_seen_ermse.
 * Either trigger sets the eval's loss to FAIL_SCORE and bypasses LAMMPS. */
namespace bad_pot_detail {
  inline double abs_threshold_mev() {
    static const double v = []{
      const char* s = std::getenv("TADAH_HPO_BAD_POT_ERMSE_MEV");
      return s && *s ? std::atof(s) : 200.0;
    }();
    return v;
  }
  inline double rel_threshold() {
    static const double v = []{
      const char* s = std::getenv("TADAH_HPO_BAD_POT_ERMSE_REL");
      return s && *s ? std::atof(s) : 0.0;
    }();
    return v;
  }
  inline std::atomic<double>& best_ermse() {
    static std::atomic<double> b{std::numeric_limits<double>::infinity()};
    return b;
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
   * On resume it would otherwise stay at 1 — the new io_buf_ starts empty
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
, prec_(trg.outputDigits)
, cw_(static_cast<std::size_t>(std::max(prec_ + 9, static_cast<int>(min_cw_))))
, max_title_len_(cw_ - 4)
{

    /* neighbours -------------------------------------------------------- */
    double rcutmax = master_ctx_.get<double>("RCUTMAX");
    // Loop over all optimap entries to find either RCUT2B or RCUTMB
    // Overwrite rcutmax if higher value is found in optimap.highbound
    for (const auto &opti: trg_.optimap) {
      if (opti.first == "RCUT2B" || opti.first == "RCUTMB") {
        for (double rcut : opti.second.highbound) {
          if (rcut > rcutmax) {
            rcutmax = rcut;
          }
        }
      }
    }

    /* validation set ---------------------------------------------------- */
    core::Context val_ctx(core::CoreUserConfigParser().parse(val_cfg));
    val_ctx_ = val_ctx;
    val_stdb_.add(val_ctx);
    n_val_datasets_ = val_stdb_.dbidx.size() > 1 ? val_stdb_.dbidx.size() - 1 : 1;
#ifndef TADAH_BUILD_MPI
    train_stdb_.add(master_ctx_);

    /* Apply load-time transforms once on host (LSCALE, ESHIFT, EFILTER/FFILTER,
     * WDBFILE/WDBFILE_AUTO, EWEIGHT_TEMP, ZERO_COM_FORCE) before any neighbour
     * finding so descriptors see the transformed dataset. The report is stored
     * for later: pot.tadah header generation and stamping LSCALE/ESHIFT into
     * each best_pot.tadah. */
    report_ = mlip::apply_load_transforms(train_stdb_, master_ctx_);

    /* Apply LSCALE/ESHIFT to validation, and also forward EFILTER/FFILTER if
     * the user declared them in the --validation file itself. Training-only
     * knobs (filters from master_ctx_, WDBFILE, EWEIGHT_TEMP, ZERO_COM_FORCE)
     * must NOT touch validation references. Use the resolved values from
     * report_ so ESHIFT_ATOM / ESHIFT_DBATOM derivations on the training set
     * propagate consistently. */
    {
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
    }

    nnf_.calc(val_stdb_, rcutmax);
    nnf_.calc(train_stdb_, rcutmax);
#else
    /* MPI host: training stdb lives on workers; validation stdb lives here.
     * Global-view transforms (ESHIFT_ATOM, ESHIFT_DBATOM, EWEIGHT_TEMP,
     * WDBFILE_AUTO) are not yet wired through TrainerHost; we rely on the
     * fail-fast guard in TrainEngine for those. Explicit ESHIFT/LSCALE on
     * validation can still be applied. EFILTER/FFILTER from the --validation
     * file itself are also forwarded; master_ctx_ filters are not. */
    {
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
    }
    nnf_.calc(val_stdb_, rcutmax);
#endif

    /* elements check ---------------------------------------------------- */
    auto tr_el = mlip::StructureDB::find_unique_elements(master_ctx_);
    for(auto e: mlip::StructureDB::find_unique_elements(val_ctx))
        if(!tr_el.count(e))
            throw std::runtime_error("Validation element missing in training");

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
                    hints << "Tip: " << name << " is near its LOWER bound — "
                          << "consider tightening or expanding the range. ";
                else if ((hi - v) / rng < EDGE_FRAC)
                    hints << "Tip: " << name << " is near its UPPER bound — "
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

/* ----------  helper: run LAMMPS (serial + OpenMP) --------------------- */
template<class P, class R>
void HPO_Host<P,R>::run_lammps(Result& r) const
{
  if(!use_lammps_) return;
  const auto& LS = trg_.LS;
  r.v_outvar.resize(LS.size());

#ifdef TADAH_ENABLE_OPENMP
#pragma omp parallel for shared(r,LS)
#endif
  for(std::size_t i=0;i<LS.size();++i)
  {
    const auto& lsim = LS[i];

    // Early abort: another script in this eval already rejected, so skip
    // LAMMPS for the remaining scripts and clamp their outputs to failscore.
    if (warmstart::aborting_iter().load(std::memory_order_relaxed) == r.iter) {
      for(const auto& n : lsim.n_outvar) {
        (void)n;
        r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      }
      continue;
    }

    // Per-iteration copy of LAMMPS argv. If the user supplied -log/-screen
    // with a real filename, append .t<thread_id> so concurrent OMP threads
    // do not clobber the same file. "none" is left untouched.
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

    // Per-thread LAMMPS instance cache, keyed by (script + invar). Saves
    // the constructor and package-list initialisation cost across HPO
    // iterations. The script's own `clear` resets the simulation between
    // calls. The key MUST include the invar list: when the same script is
    // listed in multiple PC_LAMMPS lines with different --invar values
    // (e.g. surfaces_fast.in --invar surface 100 vs --invar surface 110,
    // or phase_bcc_x_P_fast.in with 17 different P values), reusing one
    // LMP across them caused the second-and-later calls to return ""
    // for their varloss variables — observed deterministically on
    // 2026-05-05. Distinct (script, invar) → distinct LMP fixes that.
    std::string cache_key = lsim.script;
    for (const auto& s : invar) { cache_key += '\x01'; cache_key += s; }
    thread_local std::unordered_map<std::string, typename R::LMPPtr> tls_lmp_cache;
    R runner;
    auto t_create0 = std::chrono::high_resolution_clock::now();
    auto cache_it = tls_lmp_cache.find(cache_key);
    if (cache_it == tls_lmp_cache.end()) {
      auto fresh = runner.create_instance(0, invar);
      if (!fresh) continue;
      cache_it = tls_lmp_cache.emplace(cache_key, std::move(fresh)).first;
    }
    auto& lmp = cache_it->second;
    auto t_create1 = std::chrono::high_resolution_clock::now();

    // Inject (or update) ${hot_a0} via a string-style variable. LAMMPS
    // allows re-issuing `variable name string value` to update an existing
    // string variable, so this works on both first and reused calls.
    {
      const double hot_a0 = warmstart::get(lsim.script);
      char cmd[96];
      std::snprintf(cmd, sizeof(cmd),
                    "variable hot_a0 string %.10g", hot_a0);
      try { lmp->input->one(cmd); } catch (...) { /* keep going */ }
    }

    bool fail = !runner.run_script(lmp, lsim.script_lines, r.pot_ctx);
    auto t_run1 = std::chrono::high_resolution_clock::now();

    for(const auto& n : lsim.n_outvar) {
      if (fail) r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      else  r.v_outvar[i].emplace_back(runner.get_variable(lmp, n));
    }

    // A failed LAMMPS run is itself a rejection signal — short-circuit the
    // remaining scripts in this eval rather than running them on a bad pot.
    if (fail) {
      warmstart::aborting_iter().store(r.iter, std::memory_order_relaxed);
    }

    // Capture the script's relaxed lattice constant if exposed as ${a0},
    // and reject the result when a0 is outside the physical range — an
    // unphysical lattice means the potential is broken even if the
    // validation metrics happened to look fine.
    if (!fail) {
      try {
        const std::string s = runner.get_variable(lmp, "a0");
        if (!s.empty()) {
          double a0_val = std::stod(s);
          if (!warmstart::set(lsim.script, a0_val)) {
            // out of [min, max] — clamp this script's outputs to failscore
            for (auto& v : r.v_outvar[i]) v = std::to_string(lsim.failscore);
            // Signal remaining scripts in this eval to short-circuit. Any
            // peer thread already past the early-abort check will still run
            // (we don't yank LAMMPS mid-script), but threads that have not
            // yet entered the next script will see this and skip.
            warmstart::aborting_iter().store(
                r.iter, std::memory_order_relaxed);
            // Count rejections per *eval*, not per script: only the first
            // script in this eval to reject bumps the counter and emits a
            // warning (subsequent rejections in the same eval are silent).
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
                    + ": rejecting (a0=" + std::to_string(a0_val)
                    + " outside [" + std::to_string(warmstart::a0_min())
                    + ", " + std::to_string(warmstart::a0_max())
                    + "], total rejects=" + std::to_string(rejs) + ")");
              }
            }
          }
        }
      } catch (...) { /* variable not defined or unparseable — ignore */ }
    }

    if (bench_detail::enabled()) {
      const double t_create_ms = bench_detail::elapsed_ms(t_create0, t_create1);
      const double t_run_ms    = bench_detail::elapsed_ms(t_create1, t_run1);
      std::lock_guard<std::mutex> lg(bench_detail::mtx());
      static std::ofstream f("hpo_script_times.csv", std::ios::out | std::ios::trunc);
      static bool header_written = (f << "iter,thread,script,create_ms,run_ms\n", true);
      (void)header_written;
      int tid = 0;
#ifdef TADAH_ENABLE_OPENMP
      tid = omp_get_thread_num();
#endif
      f << r.iter << ',' << tid << ',' << lsim.script << ','
        << t_create_ms << ',' << t_run_ms << '\n';
      f.flush();
    }
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

    /* Per-stage watchdog logger — closes over r.iter so the post-hoc
     * leave() warning includes the iter that was running. */
    auto _stage_log = [](const std::string& msg) {
        HPOLog::instance().warning(msg);
    };

    /* Bug 5: wrap the entire eval pipeline in try/catch.
     *
     * Without this guard, a throw between `r.iter = ++g_iter_` above
     * and the eventual `submit_result(std::move(r), ...)` below leaves
     * the writer's `next_flush_` permanently stuck at the missing
     * iter — every subsequent eval enqueues fine but the writer never
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
     *  - core::ShutdownRequested — must propagate to hpo_run() so the
     *    run shuts down cleanly with a final checkpoint.
     *  - StagnationStop — must propagate to RestartPolicy. */
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
                    + "): " + e.what() + " — rebuilding Trainer");
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
                // reinit threw — log and fall back to a clean rebuild
                HPOLog::instance().warning(
                    std::string("reinit() threw (eval ") + std::to_string(r.iter)
                    + "): " + e.what() + " — rebuilding Trainer");
                needs_rebuild = true;
            }
            if (needs_rebuild) {
                cache.trainer.reset();
                cache.trainer.reset(new (cache.buf) mlip::Trainer(ctx, tls_ws_));
            }
        }
        cache.trainer->train(train_stdb_);
    }
    mlip::Trainer& host = *tls_trainer_cache_.trainer;
#endif
    t_train_end = std::chrono::high_resolution_clock::now();
    stage_watchdog::leave(r.iter, "train", _stage_log);

    /* ── Per-iter health monitor (Phase 6) ───────────────────────────
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
     * calc_rho. Removing them here is the single choke-point — every
     * downstream consumer (best_pot.tadah, per-eval pots, LAMMPS scripts)
     * uses the same `pot` Context. */
    pot.remove("AUDIT");
    pot.remove("HEALTH_LOG");
    auto pred = host.model->predict(pot, val_stdb_, host.dc);
    mlip::Analytics   a(val_stdb_, pred); // we do not need to recalc NN as each descriptor checks cutoffs internally
    fill_validation(r, a);
    r.pot_ctx  = std::move(pot);
    fit_loss(r);
    t_predict_end = std::chrono::high_resolution_clock::now();
    t_lammps_end = t_predict_end;
    t_sims_loss_end = t_predict_end;
    stage_watchdog::leave(r.iter, "predict_validate", _stage_log);

    /* optional LAMMPS -------------------------------------------------- */
    /* bad-potential heuristic: cheap pre-filter before launching LAMMPS */
    bool skip_lammps_bad_pot = false;
    if (use_lammps_ && trg_.use_ermse && std::isfinite(r.ermse)) {
      const double abs_thr = bad_pot_detail::abs_threshold_mev();
      const double rel     = bad_pot_detail::rel_threshold();
      const double best    = bad_pot_detail::best_ermse().load(std::memory_order_relaxed);
      const bool bad_abs   = (abs_thr > 0.0) && (r.ermse > abs_thr);
      const bool bad_rel   = (rel > 0.0) && std::isfinite(best) && (r.ermse > rel * best);
      if (bad_abs || bad_rel) {
        skip_lammps_bad_pot = true;
        r.loss = trg_.FAIL_SCORE;
        const auto skips = bad_pot_detail::skip_count().fetch_add(1, std::memory_order_relaxed) + 1;
        if (skips <= 5 || (skips & (skips - 1)) == 0) {
          HPOLog::instance().warning(
              std::string("eval ") + std::to_string(r.iter)
              + ": skipping LAMMPS (ERMSE=" + std::to_string(r.ermse)
              + " meV/at, abs_thr=" + std::to_string(abs_thr)
              + (bad_rel ? ", rel_thr breached vs best=" + std::to_string(best) : "")
              + ", total skips=" + std::to_string(skips) + ")");
        }
      } else {
        // Update running best; relaxed CAS is fine — strict ordering not needed.
        double cur = bad_pot_detail::best_ermse().load(std::memory_order_relaxed);
        while (r.ermse < cur &&
               !bad_pot_detail::best_ermse().compare_exchange_weak(
                   cur, r.ermse, std::memory_order_relaxed)) {}
      }
    }

    if (use_lammps_ && r.loss < trg_.FAIL_SCORE && !skip_lammps_bad_pot) {
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

    /* Top-K tracking — cheap fast-path when disabled (capacity == 0). */
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

    /* Stagnation check — throw to short-circuit the optimiser loop. The
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
     * eval — catch and continue with an empty breakdown. */
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
            std::sort(comps.begin(), comps.end(),
                      [](const Comp& a, const Comp& b) { return a.value > b.value; });
            double sum_total = 0.0;
            for (const auto& c : comps) sum_total += c.value;
            /* Display values are scaled to per-component contribution to the
             * mean (compᵢ / N), so they line up arithmetically with the
             * displayed `loss=` (which IS the mean): pctᵢ × loss ≈ valueᵢ.
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
        /* Cooperative shutdown — pass through to hpo_run()'s outer
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
        } catch (...) { /* swallow — main shutdown path takes over */ }
        throw;
    } catch (const StagnationStop&) {
        /* RestartPolicy expects this. Bug 7: same iter-slot recovery
         * as ShutdownRequested — submit FAIL_SCORE before rethrowing
         * so the writer's next_flush_ does not stall on our iter while
         * the RestartPolicy spins up the next stage. This was the
         * root cause of the 3459 freeze observed at every BASINS
         * GLOBAL→LOCAL boundary: stagnation was already past
         * threshold by ~eval 3459, the throw escaped without
         * submitting, and the writer waited 60 s before gap-skipping. */
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        try {
            HPOLog::instance().warning(
                std::string("eval ") + std::to_string(r.iter) +
                ": StagnationStop — submitting FAIL_SCORE to keep "
                "writer queue contiguous before rethrow");
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
     * leave writer_token_ stuck set — that would silently freeze every
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
         * does not help — there is no exception, the eval thread is
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
            /* Healthy progress this call — clear the stuck timer. */
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
                        std::string("HPO writer: gap-skip — advancing next_flush_ from ") +
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

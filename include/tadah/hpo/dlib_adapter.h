#pragma once
#include <dlib/global_optimization.h>
#include <tadah/hpo/log_index.h>
#include <dlib/optimization.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/run_stop.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tadah::hpo {

namespace detail {

/* DLIB's stop strategies count BFGS *iterations*, not function evals.
 * At dim=D with central differences a single iteration consumes
 * 2*D + line_search + step ~= 2*D + 2..5 evals, so passing the user's
 * MAXEVAL straight into objective_delta_stop_strategy(min_delta, max_iter)
 * silently lets BFGS overshoot the user's budget by O(D)x. We therefore
 * cap evaluations from outside DLIB by counting calls to the objective
 * lambda and throwing this on overflow. The catch site treats it as a
 * graceful termination. */
struct MaxEvalReached : public std::exception {
  const char* what() const noexcept override {
    return "DLIB:BFGS reached MAXEVAL function evaluations (eval-counted, not iter-counted)";
  }
};

/* The gradient BFGS asked for is not finite (GradientProvider::failure
 * says why): dlib cannot reject a point, so the search ends there, with
 * its own status (report_gradient_stop). */
struct GradientNotFinite : public std::exception {
  GradientFailureInfo info;
  explicit GradientNotFinite(GradientFailureInfo i) : info(std::move(i)) {}
  const char* what() const noexcept override {
    return "DLIB:BFGS: gradient not finite";
  }
};

inline std::optional<double> dlib_param(const OptimizerSpec& sp,
                                        const std::string& key)
{
  const auto it = sp.param.find(key);
  if (it == sp.param.end())
    return std::nullopt;
  return it->second;
}

inline void validate_dlib_bfgs_spec(const OptimizerSpec& sp)
{
  if (sp.lib != "DLIB" || sp.algo != "BFGS")
    throw std::runtime_error(
        "validate_dlib_bfgs_spec expects LIB=DLIB and ALGO=BFGS");

  if (sp.inner)
    throw std::runtime_error(
        "DLIB:BFGS is a standalone optimiser and does not support INNER blocks");

  if (sp.low.empty() || sp.low.size() != sp.high.size())
    throw std::runtime_error(
        "DLIB:BFGS requires non-empty, same-sized low/high bounds");

  if (!sp.is_int.empty() &&
      std::any_of(sp.is_int.begin(), sp.is_int.end(), [](bool v) { return v; })) {
    throw std::runtime_error(
        "DLIB:BFGS does not support integer-valued optimisation variables");
  }

  if (sp.vector_storage)
    throw std::runtime_error(
        "DLIB:BFGS does not support VECTOR_ARR; use ALGO GFS or an NLopt L-BFGS variant instead");
  if (sp.pop)
    throw std::runtime_error("DLIB:BFGS does not support POPULATION");
  if (sp.seed)
    throw std::runtime_error("DLIB:BFGS does not support SEED");
  /* Option A: THREADS sizes the shared fan-out pool (pc_lammps + MLIP
   * train/predict), not any DLIB-internal parallelism -- BFGS itself runs ONE
   * serial trajectory, but its per-eval fan-out uses the pool. Accept
   * THREADS >= 1 (reject only 0), consistent with ANNEAL and NLOPT GN_/LN_. */
  if (sp.num_threads && *sp.num_threads == 0)
    throw std::runtime_error("DLIB:BFGS: THREADS must be >= 1");
  if (sp.step)
    throw std::runtime_error("DLIB:BFGS does not support STEP");

  if (sp.stop.maxtime)
    throw std::runtime_error("DLIB:BFGS does not support MAXTIME");
  if (sp.stop.stopval)
    throw std::runtime_error("DLIB:BFGS does not support STOPVAL");
  if (sp.stop.ftol_rel)
    throw std::runtime_error("DLIB:BFGS does not support FTOL_REL; use FTOL_ABS");
  if (sp.stop.xtol_rel)
    throw std::runtime_error("DLIB:BFGS does not support XTOL_REL");
  if (sp.stop.xtol_abs)
    throw std::runtime_error("DLIB:BFGS does not support XTOL_ABS");

  if (sp.stop.maxeval && *sp.stop.maxeval == 0)
    throw std::runtime_error("DLIB:BFGS requires MAXEVAL > 0 when provided");

  /* PARAM fd_step, fd_kernel and diff_method are the gradient's finite
   * differences: validate_hpo_setup checks them on every phase that takes
   * them, and the gradient before its first evaluation (fd_settings_of,
   * fd_stencil.h), with the same message on every backend -- a number for
   * fd_kernel too, which this list refused as unsupported. */
  for (const auto& kv : sp.param) {
    if (is_fd_param(kv.first) || kv.first == "tol_gradinf") continue;
    throw std::runtime_error(
        "DLIB:BFGS does not support PARAM " + kv.first +
        "; supported PARAM keys are fd_step, fd_kernel, diff_method and "
        "tol_gradinf");
  }


  const auto tol_gradinf = dlib_param(sp, "tol_gradinf");
  if (tol_gradinf && !(*tol_gradinf > 0.0))
    throw std::runtime_error("DLIB:BFGS requires PARAM tol_gradinf > 0");

  if (tol_gradinf && sp.stop.ftol_abs)
    throw std::runtime_error(
        "DLIB:BFGS accepts either FTOL_ABS or PARAM tol_gradinf, but not both");
}

/* The value DLIB:GFS is given for an evaluation's loss (run_dlib_global).
 *
 * find_min_global maximises the negated loss, and the trust-region step of
 * its search (dlib::qopt_impl::pick_next_sample_using_trust_region) takes as
 * best sample the first one whose negated value exceeds -1e300. A failed
 * evaluation reports FAILSCORE, by default the largest double: once the
 * dims + 2 samples dlib had when it first took that step all sat at or above
 * 1e300 it found no best sample and, its assertions compiled out, indexed an
 * empty vector (SIGSEGV) -- whether or not later evaluations would have
 * produced usable losses. Values that large also overflow the squares of
 * dlib's upper-bound model, which normalises the samples by their standard
 * deviation. So dlib is given a finite stand-in for a failed evaluation;
 * loss.tadah, the journal and the incumbent keep the loss the evaluation
 * reported.
 *
 * A failed evaluation is one without a usable loss (loss_is_feasible):
 * non-finite, FAILSCORE, or a graded failure in [0.999 FS, FS). Its grade
 * g in [0, 1] places it among the failures -- (loss - 0.999 FS) /
 * (0.001 FS) for a graded failure, 1 for any other:
 *
 *   once usable losses exist, with W the worst and B the best so far:
 *     W + m (1 + g),  m = max(W - B, |W|), or 1 when both are 0
 *   -- above every usable loss seen, by one to two times the larger of their
 *   spread and magnitude: a step of the size of the landscape dlib has seen,
 *   not a cliff that dwarfs it;
 *
 *   before any usable loss:  K (0.999 + 0.001 g),  K = min(FS, kCeiling)
 *   (kCeiling when FS is not positive) -- a positive FAILSCORE up to
 *   kCeiling gives exactly the loss the evaluation reported, the default
 *   largest double gives kCeiling.
 *
 * A stand-in is fixed when dlib is given it and never revised, and "so
 * far" is this object's: one per DLIB:GFS search (run_dlib_global), so a
 * RESTART round starts afresh with its fresh dlib model. Graded failures
 * therefore keep their order (FAILSCORE_GRADING's way out of a failing
 * region), below the ungraded ones, only among the failures handed over
 * in the same state: all before the first usable loss, or all while the
 * same W and B hold. Across states the order is not kept: a failure
 * before the first usable loss keeps its value near K, far above the W +
 * 2m of a later ungraded failure, and an ungraded failure handed over
 * while W is small can rank below a graded one handed over after W has
 * grown.
 *
 * No value handed to dlib exceeds kCeiling in magnitude: a usable loss
 * beyond it is clamped as well, and a stand-in capped at it, so a stand-in
 * can rank below a usable loss only when that loss is above 1e100 itself.
 * kCeiling = 1e100 lies 200 orders of magnitude below dlib's 1e300
 * sentinel; its square, and a sum of such squares over any feasible number
 * of evaluations, stays finite in the standard deviation dlib normalises by,
 * and so do its products with the quadratic model's interpolation
 * coefficients, which grow like the inverse square of the trust-region
 * radius, for any radius above 1e-100. Yet it lies far above any loss a
 * potential produces.
 *
 * Thread-safe: find_min_global evaluates on a pool. */
class GfsLossForDlib {
public:
  static constexpr double kCeiling = 1e100;

  explicit GfsLossForDlib(std::optional<double> failscore)
      : failscore_(failscore)
  {}

  /* Is `loss` a failed evaluation's, one dlib is given a stand-in for? */
  bool failed(double loss) const
  {
    return failscore_ ? !loss_is_feasible(loss, *failscore_)
                      : !std::isfinite(loss);
  }

  /* The value dlib is given for an evaluation that reported `loss`; a
   * usable loss also joins the worst and best usable losses so far. */
  double operator()(double loss)
  {
    std::lock_guard<std::mutex> lk(m_);
    if (failed(loss))
      return stand_in_locked(grade(loss));
    const double v = std::clamp(loss, -kCeiling, kCeiling);
    if (!have_usable_) {
      best_ = worst_ = v;
      have_usable_ = true;
    } else {
      best_  = std::min(best_, v);
      worst_ = std::max(worst_, v);
    }
    return v;
  }

  /* The value dlib is given for an evaluation that did not run (queued
   * after a stop): an ungraded failure's. */
  double not_evaluated() const
  {
    std::lock_guard<std::mutex> lk(m_);
    return stand_in_locked(1.0);
  }

private:
  double grade(double loss) const
  {
    if (!failscore_ || !std::isfinite(loss)) return 1.0;
    const double fs = *failscore_;
    if (!(fs > 0.0) || !std::isfinite(fs)) return 1.0;
    const double floor = kFailBandFloor * fs;
    if (!(loss >= floor && loss < fs)) return 1.0;
    return std::clamp((loss - floor) / (fs - floor), 0.0, 1.0);
  }

  double stand_in_locked(double g) const
  {
    if (have_usable_) {
      double m = std::max(worst_ - best_, std::fabs(worst_));
      if (!(m > 0.0)) m = 1.0;
      return std::min(worst_ + m * (1.0 + g), kCeiling);
    }
    double k = kCeiling;
    if (failscore_ && *failscore_ > 0.0)
      k = std::min(*failscore_, kCeiling);
    return k * (kFailBandFloor + (1.0 - kFailBandFloor) * g);
  }

  std::optional<double> failscore_;
  mutable std::mutex m_;
  bool   have_usable_ = false;
  double best_  = 0.0;
  double worst_ = 0.0;
};

}  // namespace detail

/* DLIB:GFS on any host with `double operator()(const dlib::matrix<double,
 * 0,1>&)` (real-space parameters); run_dlib_global binds HPO_Host and
 * passes whether RestartPolicy has armed the stagnation stop (`restart`),
 * the unit tests a stub host. */
template<class HostT>
  void run_dlib_global_impl(const OptimizerSpec& sp, HostT& host,
                            RestartStop restart = RestartStop::Off)
  {
    using namespace dlib;

    /* find_min_global ends when its call budget (MAXEVAL) is spent, its
     * run time (MAXTIME) is up or its stop condition (STOPVAL) holds, and
     * on nothing else: one of them, or an armed RESTART's stagnation stop,
     * is required, and the tolerances it cannot apply and any PARAM are
     * refused (the startup check, again for a spec built at run time). */
    validate_dlib_gfs_spec(sp, restart);
    const auto t0 = std::chrono::steady_clock::now();

    /* find_min_global (MaxLIPO+TR) is a deterministic algorithm; a
     * SEED on this spec is never consulted. DLIB:BFGS rejects SEED
     * outright, but GFS configs with SEED already exist in the wild,
     * so warn instead of throwing (OptimizerCaps::warns_unread_seed, which
     * keeps validate_dlib_gfs_spec from refusing it). (INIT-block SEED is
     * unaffected -- it feeds the INIT chain, whose points are consumed
     * below.) */
    if (sp.seed) {
      const std::string msg =
          "DLIB:GFS (find_min_global, MaxLIPO+TR) is deterministic; "
          "SEED has no effect on the search. Randomise via the INIT "
          "block (e.g. METHOD RANDOM K with SEED) instead.";
      std::cerr << "[HPO] " << msg << "\n";
      HPOLog::instance().warning(msg);
    }

    /* dlib::find_min_global applies its OWN ln transform to any
     * non-integer dim with lower > 0 and upper/lower >= 1000 in the
     * coordinates it is given, and does not transform the seed vectors
     * (initial_function_evals) accordingly -- a second, invisible layer
     * on top of HPO's search space. It can only fire under
     * OPTIM_SCALE NONE (the unit box has lower = 0); name the dims. */
    for (std::size_t i = 0; i < sp.low.size(); ++i) {
      const bool is_int = i < sp.is_int.size() && sp.is_int[i];
      if (!is_int && sp.low[i] > 0.0 && sp.high[i] / sp.low[i] >= 1000.0) {
        std::ostringstream oss;
        oss << "DLIB:GFS: dlib applies its own ln transform to "
            << (i < sp.parameter_labels.size() ? sp.parameter_labels[i]
                                               : logidx::dim_label(i))
            << " (internal bounds [" << sp.low[i] << ", " << sp.high[i]
            << "], ratio >= 1000) and does not transform the INIT seeds "
               "handed to it; tag the dim LOG on its OPTIM line (or set "
               "OPTIM_SCALE AUTO) so HPO owns the transform";
        std::cerr << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }

    /* ------- bounds in optimiser-internal space ------------------- */
    const matrix<double,0,1> lo = mat(sp.low);
    const matrix<double,0,1> hi = mat(sp.high);

    /* ------- thread pool ---------------------------------------- */
    const unsigned nthreads = sp.num_threads.value_or(1);  // single thread as default
    dlib::thread_pool tp(nthreads);

    double solver_eps = sp.stop.ftol_abs.value_or(1e-3);
    /* Without MAXEVAL, dlib's own default: no limit on the calls. */
    std::size_t maxeval = sp.stop.maxeval
        ? static_cast<std::size_t>(*sp.stop.maxeval)
        : std::numeric_limits<std::size_t>::max();

    /* STOPVAL: the first loss at or below it ends the search
     * (anneal::stopval_reached: a feasible loss). MAXTIME: no evaluation
     * starts once it has passed since the search began. Evaluations
     * already running on the pool when a stop comes finish. */
    std::atomic<bool> stopval_hit{false};
    std::atomic<std::size_t> evaluations{0};
    const auto elapsed = [&t0]() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    };
    const auto time_up = [&]() {
        return sp.stop.maxtime && elapsed() >= *sp.stop.maxtime;
    };
    const auto note_loss = [&](double y) {
        evaluations.fetch_add(1, std::memory_order_relaxed);
        if (anneal::stopval_reached(y, sp.stop.stopval, sp.failscore))
            stopval_hit.store(true, std::memory_order_relaxed);
    };

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

    /* dlib is given a finite stand-in for a failed evaluation's loss
     * (detail::GfsLossForDlib): FAILSCORE, by default the largest double,
     * crashed its trust-region step. The journal says so once per search,
     * at its first failed evaluation. */
    detail::GfsLossForDlib to_dlib(sp.failscore);
    std::atomic<bool> stand_in_noted{false};
    const auto dlib_value = [&](double loss) {
        if (to_dlib.failed(loss) && !stand_in_noted.exchange(true)) {
            const std::string msg =
                "DLIB:GFS: dlib is given a finite stand-in for each failed "
                "evaluation of this search (the worst usable loss of the "
                "search so far plus a margin; before any, FAILSCORE, at "
                "most 1e100); loss.tadah records the loss the evaluation "
                "reported";
            HPOLog::instance().info(msg);
        }
        return to_dlib(loss);
    };

    /* find_min_global runs its evaluations on `tp`, and no exception may
     * leave it while evaluations are still running. dlib rethrows a task's
     * exception from its next add_task call, so find_min_global unwinds
     * while the other tasks still use its stack frame, and ~thread_pool
     * aborts the process when a second task threw as well -- which a
     * signal (every evaluation started after it throws) and a RESTART
     * stagnation stop (every evaluation ending after the window closed
     * throws) make the rule with THREADS > 1. So a pool evaluation that
     * throws stashes the exception (run_stop.h keeps the one that decides
     * how the run ends), returns a placeholder loss and makes
     * `stop_on_error` ask find_min_global to stop; the evaluations queued
     * after it return the placeholder without running; find_min_global
     * waits for the running ones and returns normally; and the stash is
     * rethrown below -- the NLopt and ASA adapters' pattern. The
     * placeholder is what dlib is given for a failed evaluation; it never
     * reaches the result files. */
    ExceptionStash stopped;
    /* A STOPVAL stop is taken the same way: the evaluations queued after
     * the one that reached it return the placeholder without running. */
    auto pool_feval = [&](const matrix<double,0,1>& x)->double
    {
        if (stopped.has() || stopval_hit.load(std::memory_order_relaxed))
            return to_dlib.not_evaluated();
        try {
            const double y = feval(x);
            note_loss(y);
            return dlib_value(y);
        } catch (...) {
            stopped.offer(std::current_exception());
            return to_dlib.not_evaluated();
        }
    };
    const stop_condition stop_on_error_or_stopval = [&](double) {
        return stopped.has() || stopval_hit.load(std::memory_order_relaxed);
    };

    /* ------- seed dlib with every point from sp.optim_init.
     * find_min_global() treats `initial_function_evals` as already-known
     * (x,y) pairs, so we evaluate each seed once here and hand the
     * pair(s) over.  This guarantees the user's seeds (CONFIG, WARM,
     * RANDOM/LHS/SOBOL chain entries) are part of the explored
     * set and tightens dlib's UCB-style stop criterion from the very
     * first iteration. With INIT_STRATEGY chains plus DLIB:GFS the
     * adapter consumes ALL K points (matches the documented behaviour
     * of MULTISTART when the inner ALGO is GFS). */
    std::vector<function_evaluation> init;
    if (!sp.optim_init.empty()) {
      std::ostringstream oss;
      oss << "DLIB:GFS seeding with " << sp.optim_init.size()
          << " starting point" << (sp.optim_init.size()==1 ? "" : "s");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
      for (const auto& seed_real : sp.optim_init) {
        const auto start_vec = sp.in(seed_real);
        matrix<double,0,1> x_seed;
        x_seed.set_size(static_cast<long>(start_vec.size()));
        for (long i = 0; i < x_seed.size(); ++i)
          x_seed(i) = start_vec[static_cast<std::size_t>(i)];
        /* Serial, before any pool task: an exception unwinds directly. The
         * seeds are evaluations of the search: MAXEVAL counts them, and
         * STOPVAL and MAXTIME end the search among them too. */
        const double y_seed = feval(x_seed);
        note_loss(y_seed);
        init.emplace_back(x_seed, dlib_value(y_seed));
        if (maxeval != std::numeric_limits<std::size_t>::max()) --maxeval;
        if (maxeval == 0 || stopval_hit.load() || time_up()) break;
      }
    }

    /* dlib's run time: what MAXTIME leaves after the seeds (FOREVER
     * without MAXTIME). */
    std::chrono::nanoseconds runtime = FOREVER;
    if (sp.stop.maxtime) {
      const double left = std::max(0.0, *sp.stop.maxtime - elapsed());
      runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(left));
    }
    if (maxeval > 0 && !stopval_hit.load() && !time_up())
      find_min_global(tp, pool_feval,
          lo, hi, sp.is_int,
          max_function_calls(maxeval),
          runtime,
          solver_eps,
          init,
          stop_on_error_or_stopval
          );
    stopped.rethrow_if_any();

    /* Which stop ended the search: the journal line, and the run's status
     * for a STOPVAL or MAXTIME stop (a spent MAXEVAL ends it COMPLETED). */
    const std::size_t n_evals = evaluations.load();
    const bool budget_spent = sp.stop.maxeval && n_evals >= *sp.stop.maxeval;
    const std::string done = std::to_string(n_evals) + " evaluation(s)";
    std::string msg;
    if (stopval_hit.load()) {
      msg = "DLIB:GFS stopped: STOPVAL reached after " + done;
      HPOLog::instance().set_status("DLIB:GFS: STOPVAL reached");
    } else if (!budget_spent && time_up()) {
      msg = "DLIB:GFS stopped: MAXTIME exceeded after " + done;
      HPOLog::instance().set_status("DLIB:GFS: MAXTIME exceeded");
    } else {
      msg = "DLIB:GFS stopped: MAXEVAL reached after " + done;
    }
    std::cout << "[HPO] " << msg << "\n";
    HPOLog::instance().info(msg);
  }

template<class Vec,class Runner>
  void run_dlib_global(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_dlib_global_impl(sp, host,
                         restart_stop_armed<HPO_Host<Vec,Runner>>());
  }

template<class Vec,class Runner>
  void run_dlib_bfgs(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    using namespace dlib;
    detail::validate_dlib_bfgs_spec(sp);

    const matrix<double,0,1> lo = mat(sp.low);
    const matrix<double,0,1> hi = mat(sp.high);
    matrix<double,0,1> x;

    if (!sp.optim_init.empty()) {
      const auto start = sp.in(sp.optim_init.front());
      if (start.size() != sp.low.size()) {
        throw std::runtime_error(
            "DLIB:BFGS initial point dimension does not match the search space");
      }
      x.set_size(static_cast<long>(start.size()));
      for (long i = 0; i < x.size(); ++i)
        x(i) = start[static_cast<std::size_t>(i)];
    } else {
      throw std::runtime_error(
          "DLIB:BFGS requires a starting point; HPO should provide one from the current config");
    }

    const auto maxeval = sp.stop.maxeval.has_value()
        ? static_cast<unsigned long>(*sp.stop.maxeval)
        : 0ul;

    /* Wrap the objective so each call increments a shared atomic
     * counter. The same counter is handed to the GradientProvider so
     * that gradient-side evaluations also count toward MAXEVAL -- without
     * this, dlib::derivative used to count 2*D extra cost calls per
     * iteration via the feval lambda; now the finite-difference
     * evaluations go through gp.gradient and bump the same counter. When
     * the user's MAXEVAL is reached, throw detail::MaxEvalReached; the catch site
     * below treats it as graceful termination so the best-so-far tracked
     * by HPO_Host is preserved. */
    std::atomic<unsigned long> evals_used{0};
    /* The last evaluation feval made, and the count after it.
     * find_min_box_constrained asks for a gradient at its start, right
     * after the loss there, and after each line search at the point the
     * search ended on -- the last point it evaluated (backtracking_line_search
     * evaluates the loss alone and returns the step it evaluated last) --
     * and NUMERICAL and HYBRID then take that loss as the loss at the
     * point (gradient_at), without evaluating the point again, when it is
     * the same doubles and nothing was evaluated since. */
    struct LastEval {
        matrix<double,0,1> x;
        double             y = 0.0;
        unsigned long      count = 0;
        bool               valid = false;
    } last;
    auto feval = [&, maxeval](const matrix<double,0,1>& xin)->double {
      if (maxeval > 0 && evals_used.load() >= maxeval)
          throw detail::MaxEvalReached{};
      evals_used.fetch_add(1);
      const double y = host( sp.out(xin) );
      last.x = xin;
      last.y = y;
      last.count = evals_used.load();
      last.valid = true;
      return y;
    };

    /* Gradient via the unified provider. Default GRAD_STRATEGY is
     * NUMERICAL -- finite differences (fd_stencil.h: PARAM fd_step,
     * fd_kernel, diff_method). ANALYTICAL/HYBRID consume an
     * analytical_grad_fn that delegates to HPO_Host::analytical_gradient
     * (Phase 2A.4 wiring; see hpo_host_impl.hpp). */
    AnalyticalGradFn<Vec> agf = nullptr;
    if (sp.grad_strategy &&
        *sp.grad_strategy != GradStrategy::Numerical) {
        agf = [&host](const Vec& x_in, Vec& g_out,
                      std::vector<int>& nfd) -> bool {
            return host.analytical_gradient(x_in, g_out, nfd);
        };
    }
    auto gp = make_gradient_provider<Vec, Runner>(
        sp, host, &evals_used, maxeval, agf);
    /* A gradient that is not finite ends the search with its own status;
     * dlib has no way to reject the point. Before 1.6.0 it was reported as
     * "MAXEVAL=<n> reached". */
    const auto grad = [&gp, &last, &evals_used](const matrix<double,0,1>& x) {
        matrix<double,0,1> g;
        const bool just_evaluated = last.valid &&
                                    last.count == evals_used.load() &&
                                    detail::same_point(last.x, x);
        if (!(just_evaluated ? gp.gradient_at(x, last.y, g)
                             : gp.gradient(x, g))) {
            switch (gp.failure->kind) {
                case GradientFailure::Budget:
                    throw detail::MaxEvalReached{};
                case GradientFailure::Stencil:
                case GradientFailure::NotFinite:
                    throw detail::GradientNotFinite{*gp.failure};
                case GradientFailure::None:
                    break;
            }
            throw std::logic_error(
                "DLIB:BFGS: the gradient provider failed without a reason");
        }
        return g;
    };

    auto run_with_eval_cap = [&](auto&& search, auto&& stop) {
      try {
        find_min_box_constrained(std::forward<decltype(search)>(search),
                                 std::forward<decltype(stop)>(stop),
                                 feval, grad, x, lo, hi);
      } catch (const detail::MaxEvalReached&) {
        /* MAXEVAL is enforced externally because DLIB's stop strategies
         * count iterations (not evals). Hitting it is a normal termination
         * once BFGS has consumed its eval budget; the preamble explaining
         * this design fires once per HPO run (HPOLog::first_in_run) so the
         * run's later searches (MULTISTART starts, PRESTAGE refinements,
         * RESTART rounds) don't repeat it. */
        if (HPOLog::instance().first_in_run("DLIB:BFGS MAXEVAL preamble")) {
            const std::string preamble =
                "DLIB:BFGS -- MAXEVAL is enforced externally by HPO "
                "(DLIB stop strategies count iterations, not evals). "
                "Subsequent MAXEVAL hits in this run will be summarised tersely.";
            std::cout << "[HPO] " << preamble << "\n";
            HPOLog::instance().info(preamble);
        }
        std::ostringstream oss;
        oss << "DLIB:BFGS -- MAXEVAL=" << maxeval
            << " reached; preserving best-so-far";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().info(oss.str());
      } catch (const detail::GradientNotFinite& e) {
        report_gradient_stop("DLIB:BFGS", e.info);
      }
    };

    if (const auto tol_gradinf = detail::dlib_param(sp, "tol_gradinf")) {
      run_with_eval_cap(bfgs_search_strategy(),
                        gradient_norm_stop_strategy(*tol_gradinf));
      return;
    }

    const double ftol_abs = sp.stop.ftol_abs.value_or(1.0e-7);
    run_with_eval_cap(bfgs_search_strategy(),
                      objective_delta_stop_strategy(ftol_abs));
  }

}   // namespace tadah::hpo

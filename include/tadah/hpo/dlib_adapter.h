#pragma once
#include <dlib/global_optimization.h>
#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/log_index.h>
#include <dlib/optimization.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/failed_loss_stand_in.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/repeat_points.h>
#include <tadah/hpo/run_end.h>
#include <tadah/hpo/run_stop.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tadah::hpo {

namespace detail {

/** DLIB's stop strategies count BFGS *iterations*, not function evals.
 *  At dim=D with central differences a single iteration consumes
 *  2*D + line_search + step ~= 2*D + 2..5 evals, so passing the user's
 *  MAXEVAL straight into objective_delta_stop_strategy(min_delta, max_iter)
 *  silently lets BFGS overshoot the user's budget by O(D)x. We therefore
 *  cap evaluations from outside DLIB by counting calls to the objective
 *  lambda and throwing this on overflow. The catch site treats it as a
 *  graceful termination. */
struct MaxEvalReached : public std::exception {
  /** A fixed message naming the eval-counted MAXEVAL. */
  const char* what() const noexcept override {
    return "DLIB:BFGS reached MAXEVAL function evaluations (eval-counted, not iter-counted)";
  }
};

/** The point BFGS asked for the gradient of has no usable gradient
 *  (GradientProvider::failure says why: a failed evaluation, or a gradient
 *  that is not finite): dlib cannot reject a point, so the search ends
 *  there, with its own status (report_gradient_stop). */
struct GradientNotFinite : public std::exception {
  GradientFailureInfo info;  ///< why the gradient is not usable
  /** Carries the provider's failure `i`. */
  explicit GradientNotFinite(GradientFailureInfo i) : info(std::move(i)) {}
  /** "DLIB:BFGS: no usable gradient"; `info` has the cause. */
  const char* what() const noexcept override {
    return "DLIB:BFGS: no usable gradient";
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
  /* A PRESTAGE refinement's SEED that the exploration took seeds the
   * exploration (OptimizerSpec::seed_seeds_exploration). */
  if (sp.seed && !sp.seed_seeds_exploration)
    throw std::runtime_error("DLIB:BFGS does not support SEED");
  /* THREADS sizes the shared fan-out pool (pc_lammps + MLIP
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
  const auto unsupported = [](const std::string& key) {
    return std::runtime_error(
        "DLIB:BFGS does not support PARAM " + key +
        "; supported PARAM keys are fd_step, fd_kernel, diff_method and "
        "tol_gradinf");
  };
  for (const auto& kv : sp.param) {
    if (is_fd_param(kv.first) || kv.first == "tol_gradinf") continue;
    throw unsupported(kv.first);
  }
  /* A word (OptimizerSpec::param_raw): refused under a name DLIB:BFGS does
   * not take as a number is, and under tol_gradinf, which takes a number;
   * the finite-difference keys judge theirs (fd_kernel takes a word). */
  for (const auto& kv : sp.param_raw) {
    if (is_fd_param(kv.first)) continue;
    if (kv.first != "tol_gradinf") throw unsupported(kv.first);
    throw std::runtime_error("DLIB:BFGS: PARAM " + kv.first +
                             " takes a number (got '" + kv.second + "')");
  }


  const auto tol_gradinf = dlib_param(sp, "tol_gradinf");
  if (tol_gradinf && !(*tol_gradinf > 0.0))
    throw std::runtime_error("DLIB:BFGS requires PARAM tol_gradinf > 0");

  if (tol_gradinf && sp.stop.ftol_abs)
    throw std::runtime_error(
        "DLIB:BFGS accepts either FTOL_ABS or PARAM tol_gradinf, but not both");
}

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

    /* find_min_global (MaxLIPO+TR) is a deterministic algorithm; a SEED
     * on this spec is refused (validate_dlib_gfs_spec), unless it is a
     * PRESTAGE refinement's SEED that the exploration took, which seeds
     * the exploration: the journal names it as the exploration's (EXPLORE
     * SEED). (INIT-block SEED is unaffected -- it feeds the INIT chain,
     * whose points are consumed below.) */
    if (sp.seed && sp.seed_seeds_exploration) {
      const std::string msg =
          "DLIB:GFS (find_min_global, MaxLIPO+TR) is deterministic; SEED has "
          "no effect on the refinement: it seeds the exploration (EXPLORE "
          "SEED " + std::to_string(*sp.seed) + ").";
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
     * but the search's first starts once it has passed since the search
     * began. Evaluations already running on the pool when a stop comes
     * finish. */
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
     * (detail::FailedLossStandIn): FAILSCORE, by default the largest
     * double, crashed its trust-region step. The journal says so once per
     * search, at its first failed evaluation. */
    detail::FailedLossStandIn to_dlib(sp.failscore);
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

    /* dlib's run time: what MAXTIME leaves after the seeds, and at most
     * dlib's own FOREVER (200 years; also without MAXTIME). dlib counts it
     * in nanoseconds, which hold about 292 years: a longer MAXTIME, which
     * the parser accepts, does not convert (it came out negative and
     * ended dlib at once). */
    std::chrono::nanoseconds runtime = FOREVER;
    if (sp.stop.maxtime) {
      const double left = std::max(0.0, *sp.stop.maxtime - elapsed());
      if (left < std::chrono::duration<double>(FOREVER).count())
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
    /* A search makes its first evaluation whatever MAXTIME, as the INIT
     * points' loop above and the other drivers do: MAXTIME keeps the later
     * ones from starting. Without an INIT point, a MAXTIME that had passed
     * left find_min_global out, or dlib's own run-time check came before its
     * first call, and the search made none (the run had no result, and a
     * PRESTAGE exploration nothing to refine). One call with no run-time
     * limit evaluates the point dlib starts every search with. */
    if (evaluations.load() == 0 && !stopped.has())
      find_min_global(tp, pool_feval,
          lo, hi, sp.is_int,
          max_function_calls(1),
          FOREVER,
          solver_eps,
          init,
          stop_on_error_or_stopval
          );
    stopped.rethrow_if_any();

    /* Which stop ended the search: the journal line and the run's status
     * (run_end.h). find_min_global returns once its call budget (MAXEVAL)
     * is spent, its run time (what MAXTIME left) is up or its stop
     * condition holds; the seed loop above stops on the same three. A
     * search that spends its MAXEVAL ends on that, even when MAXTIME has
     * passed by then (no evaluation was left to start). With calls left and
     * no STOPVAL, the run time ended it, which only MAXTIME sets. */
    const std::size_t n_evals = evaluations.load();
    const bool budget_spent = sp.stop.maxeval && n_evals >= *sp.stop.maxeval;
    EndReason end = EndReason::MaxEval;
    if (stopval_hit.load())    end = EndReason::StopVal;
    else if (budget_spent)     end = EndReason::MaxEval;
    else if (sp.stop.maxtime)  end = EndReason::MaxTime;
    else
      throw std::logic_error(
          "internal error: DLIB:GFS ended with calls left, no STOPVAL "
          "reached and no MAXTIME set");
    report_end("DLIB:GFS", end, std::to_string(n_evals) + " evaluation(s)");
  }

template<class Vec,class Runner>
  void run_dlib_global(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_dlib_global_impl(sp, host,
                         restart_stop_armed<HPO_Host<Vec,Runner>>());
  }

namespace detail {

/** How detail::bfgs_box_search ended. */
enum class BfgsBoxEnd {
  Stop,    ///< the stop strategy ended the search
  NoStep,  ///< no later iteration can move the point (see bfgs_box_search)
  Revisit  ///< an iteration moved the point without an evaluation (see
           ///< bfgs_box_search (e))
};

/** What detail::bfgs_box_search returns. */
struct BfgsBoxResult {
  BfgsBoxEnd               end = BfgsBoxEnd::Stop;  ///< how the search ended
  dlib::matrix<double,0,1> gradient;                ///< the gradient at the final point
  std::size_t              line_searches_not_run = 0;  ///< iterations that could not move
  /** NoStep: the farthest first step of a line search from the final
   *  point (stalled_reach), a multiple of the steepest-descent step. */
  double                   reach = 0.0;
  /** NoStep: the line search from `reach` ran and came back to the point
   *  (bfgs_box_search (d)); false when no step up to `reach` moves it. */
  bool                     came_back = false;
  /** Revisit: the point the iteration that made no evaluation started
   *  from (the search ends at the point it moved to). */
  dlib::matrix<double,0,1> revisit_from;
};

/** The farthest first step a line search of dlib's loop takes from a point
 *  where the last iteration did not move (bfgs_box_search (c)): x, its loss
 *  f0 and gradient g and BFGS's identity H stand, and so do the step s and
 *  d0 = dot(g, s) <= 0, so a later iteration differs only in its first
 *  step, a0 now. The loss at every trial that cannot move the point is f0,
 *  and dlib's line search accepts its first trial t there exactly when its
 *  Armijo test, f0 <= f0 + t rho d0, holds in floating point -- when the
 *  decrease it asks for is below the rounding of f0, for every t up to
 *  some T. Only an accepted first step grows, tenfold and to at most 1;
 *  any other shrinks. So no later first step exceeds
 *  max(|a0|, min(10 T, 1)). T is found by bisection over the doubles in
 *  [0, 0.1] with dlib's own line search on the constant f0 (two of its
 *  iterations: the first trial is accepted, or it shrinks); with T >= 0.1
 *  (d0 = 0 among them) the bound is max(|a0|, 1). */
inline double stalled_reach(double f0, double d0, double a0, double rho)
{
  const auto accepted = [&](double t) {
    const double alpha = dlib::backtracking_line_search(
        [f0](double) { return f0; }, f0, d0, t, rho, 2);
    return std::abs(alpha) == t;
  };
  const double a = std::abs(a0);
  const double tenth = 0.1;
  if (accepted(tenth)) return std::max(a, 1.0);
  /* The largest accepted t below 0.1: positive doubles are ordered as their
   * bit patterns, and accepted() is monotone in t (0 is accepted). */
  std::uint64_t lo = 0, hi = 0;
  std::memcpy(&hi, &tenth, sizeof hi);
  while (hi - lo > 1) {
    const std::uint64_t mid = lo + (hi - lo) / 2;
    double t = 0.0;
    std::memcpy(&t, &mid, sizeof t);
    (accepted(t) ? lo : hi) = mid;
  }
  double T = 0.0;
  std::memcpy(&T, &lo, sizeof T);
  return std::max(a, std::min(T * 10, 1.0));
}

/** dlib's find_min_box_constrained (dlib/optimization/optimization.h, the
 *  projected-BFGS loop of Schmidt et al.), step for step and with dlib's own
 *  parts -- the search and stop strategies, zero_bounded_variables,
 *  gap_step_assign_bounded_variables, backtracking_line_search,
 *  make_line_search_function, clamp_function and clamp -- so it asks for the
 *  same losses and gradients at the same points, bit for bit, wherever every
 *  iteration moves the point and evaluates. Four changes for an iteration
 *  that cannot move, (a) to (d), and one, (e), for an iteration that moves
 *  without an evaluation:
 *
 *  (a) The first trial of the line search, clamp(x + a0 s) with a0 its
 *      first step (dlib's sign rule), is x itself bit for bit: every trial is
 *      x -- the step only shrinks, its sign kept, and rounding and the clamp
 *      are monotone -- so the line search can only re-evaluate the point and
 *      keep it. It is not run: its step length follows dlib's line search on
 *      the unchanged loss (computed, no evaluation), and the point, its loss
 *      and its gradient stand. dlib evaluated the point again, up to the
 *      line search's 100 iterations when the loss came back above f0 by
 *      rounding (a warm-started PC_LAMMPS relaxation), took its gradient
 *      again and kept the point. `on_not_run(x)` is called.
 *  (b) A line search that ended on its own start (its step shrank until the
 *      trial was x) leaves the point where it was, and its gradient is not
 *      formed again.
 *  (c) After an iteration that did not move -- the point is stalled -- dlib's
 *      BFGS update has no step (dot(delta, gamma) = 0) and resets H to the
 *      identity: the step is the projected steepest descent, the same at
 *      every later iteration from the point. When the first trial of a
 *      stalled iteration cannot move the point, no later first step exceeds
 *      the reach R (stalled_reach), and when the step R cannot move the
 *      point either, no later iteration can, and the search ends:
 *      BfgsBoxEnd::NoStep. dlib went on re-evaluating the point (its loss
 *      unchanged, so its step length cycled below R) until MAXEVAL, or
 *      without end when the stop test does not end a search at an
 *      unchanged point (PARAM tol_gradinf, whose gradient norm at a bound
 *      optimum is not small; FTOL_ABS 0). Under FTOL_ABS > 0 the stop test
 *      ends a stalled search first: the loss did not change.
 *  (d) When the step R does move the point, the stalled iteration runs its
 *      line search from R, the farthest first step dlib could reach from
 *      the point, and every trial of which is evaluated as dlib's are: dlib
 *      got there, if at all, only through iterations that each evaluated the
 *      point again. When that line search comes back to the point, the next
 *      stalled iteration would run it again from the same point, step and R
 *      -- the same trials -- and the search ends NoStep instead
 *      (`came_back`).
 *  So no two iterations in a row ask for no loss, and a stalled point runs
 *  at most two line searches (one from a first step that moves it, one from
 *  R) before the search moves or ends.
 *  (e) `evaluations()` is the number of evaluations made so far (the loss
 *      and the gradient answer some calls from the run's record of the
 *      points evaluated: BfgsObjective). An iteration that moved the point
 *      and made no evaluation had every trial of its line search, and the
 *      gradient at its new point, answered from the record. The gradients
 *      there are only those this search formed where it stood (its start,
 *      the end of an earlier iteration), so it is back at a point where it
 *      stood, with the loss and gradient it had there: the iterations since
 *      left the loss where it was, and the last asked for no new point. On
 *      a plateau of the loss's rounding, where every step is accepted (its
 *      Armijo decrease is lost in the rounding), under a stop test that
 *      does not end a search whose loss stays (FTOL_ABS 0; PARAM
 *      tol_gradinf with a coordinate held at a bound) BFGS goes round such
 *      points for ever, without an evaluation, so MAXEVAL never ends it.
 *      The search ends at the next stop test that does not end it itself
 *      (an FTOL_ABS > 0 test does, the loss unchanged): BfgsBoxEnd::Revisit
 *      (`revisit_from` the point the iteration started from, x the point it
 *      came back to). It is a stop rule: a search that a changed BFGS
 *      matrix would have taken elsewhere later is ended there too. With
 *      (a)-(d), which bound the iterations that do not move, no search
 *      makes more than a few iterations in a row without an evaluation,
 *      and MAXEVAL bounds the rest. Every iteration that evaluates is
 *      dlib's.
 *
 *  dlib's checks for a non-finite loss or gradient are kept (dlib::error). */
template <class Search, class Stop, class F, class G, class OnNotRun,
          class Evaluations>
BfgsBoxResult bfgs_box_iterate(Search search, Stop stop, const F& f,
                               const G& der, dlib::matrix<double,0,1>& x,
                               const dlib::matrix<double,0,1>& x_lower,
                               const dlib::matrix<double,0,1>& x_upper,
                               double& f_value,
                               const dlib::matrix<double,0,1>& g_start,
                               OnNotRun&& on_not_run,
                               Evaluations&& evaluations);

template <class Search, class Stop, class F, class G, class OnNotRun,
          class Evaluations>
BfgsBoxResult bfgs_box_search(Search search, Stop stop, const F& f,
                              const G& der, dlib::matrix<double,0,1>& x,
                              const dlib::matrix<double,0,1>& x_lower,
                              const dlib::matrix<double,0,1>& x_upper,
                              double& f_value, OnNotRun&& on_not_run,
                              Evaluations&& evaluations)
{
  using T = dlib::matrix<double,0,1>;
  f_value = f(x);
  const T g = der(x);

  if (!dlib::is_finite(f_value))
    throw dlib::error("The objective function generated non-finite outputs");
  if (!dlib::is_finite(g))
    throw dlib::error("The objective function generated non-finite outputs");
  return bfgs_box_iterate(std::move(search), std::move(stop), f, der, x,
                          x_lower, x_upper, f_value, g,
                          std::forward<OnNotRun>(on_not_run),
                          std::forward<Evaluations>(evaluations));
}

/** The scale DLIB:BFGS divides its loss by (bfgs_box_search_scaled): the
 *  length of the gradient `g` at the start `x`, whose loss is `f`, over the
 *  coordinates its first step can move -- the gradient dlib's first step
 *  takes, without the components of coordinates held at a bound of the box
 *  (`x_lower`, `x_upper`) that point out of it (zero_bounded_variables). A
 *  component the box holds takes no part in the step: counted, a start on
 *  a bound with a steep loss beyond it divided the free coordinates'
 *  gradient by it, and their steps shrank to that fraction. None (the loss
 *  is not divided) when the length is 0 or not finite, or the loss divided
 *  by it is not finite. */
inline std::optional<double> bfgs_loss_scale(
    double f, const dlib::matrix<double,0,1>& g,
    const dlib::matrix<double,0,1>& x,
    const dlib::matrix<double,0,1>& x_lower,
    const dlib::matrix<double,0,1>& x_upper)
{
  const double gap_eps = 1e-8;   // dlib's, as in bfgs_box_iterate
  const double c = dlib::length(
      dlib::zero_bounded_variables(gap_eps, g, x, g, x_lower, x_upper));
  if (!(c > 0.0) || !std::isfinite(c) || !std::isfinite(f / c))
    return std::nullopt;
  return c;
}

/** bfgs_box_search on the loss divided by c = bfgs_loss_scale of the start
 *  (returned in `scale`): dlib's loop takes its steps in multiples of the
 *  gradient, and its first trial is the start minus the gradient, so on
 *  the loss itself its path followed the unit of the loss -- a loss in
 *  meV^2 crept, the same loss times 1e6 jumped to a corner of the box and
 *  stalled. Divided by the length of its first gradient (over the
 *  coordinates the box lets it move), every loss starts with a step of
 *  length 1, and the same problem in any unit asks for the same points
 *  (exactly so for a factor that is a power of 2). The
 *  gradient is divided by c with the loss, and the stop strategy is built
 *  by `make_stop(length)` for the divided loss (its tolerances divided by
 *  c; `length` is bfgs_loss_scale's, none when c is 1);
 *  `f_value` and the result's gradient are those of the divided loss; the
 *  calls of `f` and `der` -- the evaluations, their records -- are in the
 *  loss's own units. */
template <class Search, class MakeStop, class F, class G, class OnNotRun,
          class Evaluations>
BfgsBoxResult bfgs_box_search_scaled(Search search, MakeStop&& make_stop,
                                     const F& f, const G& der,
                                     dlib::matrix<double,0,1>& x,
                                     const dlib::matrix<double,0,1>& x_lower,
                                     const dlib::matrix<double,0,1>& x_upper,
                                     double& f_value, OnNotRun&& on_not_run,
                                     Evaluations&& evaluations, double& scale)
{
  using T = dlib::matrix<double,0,1>;
  const double f0 = f(x);
  const T g0 = der(x);
  if (!dlib::is_finite(f0))
    throw dlib::error("The objective function generated non-finite outputs");
  if (!dlib::is_finite(g0))
    throw dlib::error("The objective function generated non-finite outputs");
  const std::optional<double> length =
      bfgs_loss_scale(f0, g0, x, x_lower, x_upper);
  const double c = length.value_or(1.0);
  scale = c;
  const auto fc = [&f, c](const T& v) { return f(v) / c; };
  const auto gc = [&der, c](const T& v) -> T { return der(v) / c; };
  f_value = f0 / c;
  const T g = g0 / c;
  return bfgs_box_iterate(std::move(search), make_stop(length), fc, gc, x,
                          x_lower, x_upper, f_value, g,
                          std::forward<OnNotRun>(on_not_run),
                          std::forward<Evaluations>(evaluations));
}

/** The iterations of bfgs_box_search from `x`, whose loss `f_value` and
 *  gradient `g_start` are known. */
template <class Search, class Stop, class F, class G, class OnNotRun,
          class Evaluations>
BfgsBoxResult bfgs_box_iterate(Search search, Stop stop, const F& f,
                               const G& der, dlib::matrix<double,0,1>& x,
                               const dlib::matrix<double,0,1>& x_lower,
                               const dlib::matrix<double,0,1>& x_upper,
                               double& f_value,
                               const dlib::matrix<double,0,1>& g_start,
                               OnNotRun&& on_not_run,
                               Evaluations&& evaluations)
{
  using T = dlib::matrix<double,0,1>;
  BfgsBoxResult r;
  T& g = r.gradient;
  T s;
  g = g_start;

  const double gap_eps = 1e-8;   // dlib's
  double last_alpha = 1;
  bool stalled = false;          // the last iteration did not move the point
  bool from_reach = false;       // this stall's line search from R ran (d)
  bool revisit = false;          // the last iteration moved without an evaluation (e)
  T revisit_from;
  while (stop.should_continue_search(x, f_value, g)) {
    if (revisit) {
      /* (e) */
      r.end = BfgsBoxEnd::Revisit;
      r.revisit_from = revisit_from;
      return r;
    }
    const auto evaluations_before = evaluations();
    s = search.get_next_direction(
        x, f_value,
        dlib::zero_bounded_variables(gap_eps, g, x, g, x_lower, x_upper));
    s = dlib::gap_step_assign_bounded_variables(gap_eps, s, x, g, x_lower,
                                                x_upper);
    const double d0 = dlib::dot(g, s);
    double a0 = last_alpha;   // backtracking_line_search's first step
    if ((d0 > 0 && a0 > 0) || (d0 < 0 && a0 < 0)) a0 = -a0;

    double alpha;
    bool moved;
    const T first = dlib::clamp(x + a0 * s, x_lower, x_upper);
    if (same_point(first, x) && !stalled) {
      /* (a) */
      const double f0 = f_value;
      alpha = dlib::backtracking_line_search(
          [f0](double) { return f0; }, f_value, d0, last_alpha,
          search.get_wolfe_rho(), search.get_max_line_search_iterations());
      ++r.line_searches_not_run;
      on_not_run(x);
      moved = false;
    } else {
      if (same_point(first, x)) {
        /* (c), (d): stalled, and the first trial cannot move. */
        const double reach =
            stalled_reach(f_value, d0, a0, search.get_wolfe_rho());
        const double start = a0 < 0 ? -reach : reach;
        const T farthest = dlib::clamp(x + start * s, x_lower, x_upper);
        const bool cannot = same_point(farthest, x);
        if (cannot || from_reach) {
          r.end = BfgsBoxEnd::NoStep;
          r.reach = reach;
          r.came_back = !cannot;
          return r;
        }
        from_reach = true;
        last_alpha = start;
      }
      alpha = dlib::backtracking_line_search(
          dlib::make_line_search_function(
              dlib::clamp_function(f, x_lower, x_upper), x, s, f_value),
          f_value, d0, last_alpha, search.get_wolfe_rho(),
          search.get_max_line_search_iterations());
      const T next = dlib::clamp(x + alpha * s, x_lower, x_upper);
      moved = !same_point(next, x);
      if (moved) {
        revisit_from = x;
        x = next;
      }
    }

    /* dlib's trust-region rule for the next first step. */
    if (alpha == last_alpha)
      last_alpha = std::min(last_alpha * 10, 1.0);
    else
      last_alpha = alpha;

    if (moved) {
      g = der(x);
      from_reach = false;
      revisit = evaluations() == evaluations_before;
    }
    stalled = !moved;

    if (!dlib::is_finite(f_value))
      throw dlib::error("The objective function generated non-finite outputs");
    if (!dlib::is_finite(g))
      throw dlib::error("The objective function generated non-finite outputs");
  }
  return r;
}

/** bfgs_box_search on plain functions: every call of `f` and `der` is an
 *  evaluation, so (e) never ends the search. */
template <class Search, class Stop, class F, class G, class OnNotRun>
BfgsBoxResult bfgs_box_search(Search search, Stop stop, const F& f,
                              const G& der, dlib::matrix<double,0,1>& x,
                              const dlib::matrix<double,0,1>& x_lower,
                              const dlib::matrix<double,0,1>& x_upper,
                              double& f_value, OnNotRun&& on_not_run)
{
  using T = dlib::matrix<double,0,1>;
  unsigned long calls = 0;
  const auto counted_f = [&](const T& v) {
    ++calls;
    return f(v);
  };
  const auto counted_der = [&](const T& v) {
    ++calls;
    return der(v);
  };
  return bfgs_box_search(std::move(search), std::move(stop), counted_f,
                         counted_der, x, x_lower, x_upper, f_value,
                         std::forward<OnNotRun>(on_not_run),
                         [&calls] { return calls; });
}

/** DLIB:BFGS's loss and gradient over the internal coordinates of `sp`,
 *  with its MAXEVAL (every evaluation, the gradient's included, in one
 *  counter, as the GradientProvider counts them), the run's record of the
 *  points evaluated and the gradients this search formed (repeat_points.h).
 *  A Centre call (below) evaluates the stencil and so counts in MAXEVAL,
 *  which it checks first.
 *
 *  find_min_box_constrained asks for a gradient at its start, right after
 *  the loss there, and after each line search at the point the search ended
 *  on -- the last point it evaluated (backtracking_line_search evaluates the
 *  loss alone and returns the step it evaluated last) -- and NUMERICAL and
 *  HYBRID then take that loss as the loss at the point (gradient_at),
 *  without evaluating the point again, when it is the same doubles and
 *  nothing was evaluated since. dlib asked for no gradient of another
 *  point; a loss answered from the record can be followed by one. Under
 *  NUMERICAL its recorded loss is the loss at the point (gradient_at), and
 *  only the stencil is evaluated; under ANALYTICAL and HYBRID the point is
 *  evaluated first: they differentiate the host's last evaluation
 *  (HPO_Host::analytical_gradient does not check its point), and the
 *  provider's gradient(x) formed such a gradient without evaluating x.
 *
 *  A loss answered from the record stands for an evaluation saved -- one
 *  reuse in the RESTART window and in DIAGNOSTICS (RepeatPoints::note_reuse)
 *  -- except when the gradient asked next at its point evaluates the point
 *  (ANALYTICAL, HYBRID: RepeatUse::Evaluate), which answers the loss call
 *  too: the two calls are one evaluation, as dlib's start and the end of
 *  each line search ask for them, and a PRESTAGE refinement's start, an
 *  exploration candidate, is evaluated once and counted once. So a loss
 *  call whose gradient would be evaluated is counted at the next call
 *  (pending_): not at all when that call is the gradient there, made
 *  within MAXEVAL; as a reuse otherwise -- another call, a gradient that
 *  MAXEVAL refuses, or the end of the search (settle(), which
 *  run_dlib_bfgs_impl calls when the loop returns; bfgs_box_search asks
 *  for a call after each such loss: the line search's next trial, or the
 *  gradient of the point it ended on). */
template <class Vec, class Runner, class HostT>
class BfgsObjective {
public:
  /** `maxeval` 0: no limit. */
  BfgsObjective(const OptimizerSpec& sp, HostT& host, unsigned long maxeval)
      : sp_(sp), host_(host), maxeval_(maxeval),
        gp_(make_gradient_provider<Vec, Runner>(
            sp, host, &evals_, maxeval,
            host_analytical_grad_fn<Vec>(sp, host))),
        repeats_("DLIB:BFGS", sp, repeat_policy<Vec>(host),
                 "a reuse is no evaluation, and MAXEVAL counts evaluations",
                 latest_evaluation(host))
  {}
  BfgsObjective(const BfgsObjective&) = delete;
  BfgsObjective& operator=(const BfgsObjective&) = delete;

  /** The loss at `x`: the record's, or an evaluation (MaxEvalReached once
   *  MAXEVAL is spent). */
  double loss(const Vec& x)
  {
    settle();
    if (const auto e = repeats_.find(x)) {
      if (repeats_.use(*e, false) == RepeatUse::Record) {
        /* A gradient asked at x next would evaluate x (ANALYTICAL,
         * HYBRID): this call is counted when the next one shows whether
         * it saved an evaluation (see the class comment). */
        if (repeats_.use(*e, true) == RepeatUse::Evaluate)
          pending_ = Pending{x, *e};
        else
          repeats_.note_reuse(x, *e, RepeatUse::Record, false);
        /* The loss just asked for, as an evaluation's: a gradient asked
         * at x next takes it (gradient). */
        last_.x = x;
        last_.y = e->loss;
        last_.count = evals_.load();
        last_.valid = true;
        last_.reused = true;
        return e->loss;
      }
      if (spent()) throw MaxEvalReached{};
      repeats_.note_evaluated_again(x, *e, false);
    }
    return evaluate(x);
  }

  /** The gradient at `x`: the record's, or formed by the provider after the
   *  loss at `x` -- the last evaluation, the record's under NUMERICAL, or
   *  an evaluation of `x` (MaxEvalReached, GradientNotFinite). */
  Vec gradient(const Vec& x)
  {
    const auto e = repeats_.find(x);
    /* The loss at x was the last one asked for, nothing evaluated since:
     * evaluated, or answered from the record. dlib asks for the gradient
     * of the point its line search ended on, its last loss. */
    const bool just_asked = last_.valid &&
                            last_.count == evals_.load() &&
                            same_point(last_.x, x);
    /* The pending loss at x is answered by the evaluation of x that this
     * gradient makes, when MAXEVAL allows it (class comment). */
    if (pending_ && just_asked && same_point(pending_->x, x) && e &&
        repeats_.use(*e, true) == RepeatUse::Evaluate && !spent())
      pending_.reset();
    else
      settle();
    if (e && e->has_gradient && repeats_.use(*e, true) == RepeatUse::Record) {
      repeats_.note_reuse(x, *e, RepeatUse::Record, true);
      return e->gradient;
    }
    const bool just_evaluated = just_asked && !last_.reused;
    double y = last_.y;
    if (just_asked && last_.reused && e &&
        repeats_.use(*e, true) == RepeatUse::Centre) {
      /* NUMERICAL takes that loss as the loss at x, as it takes an
       * evaluation's: the call that answered it was counted (in the RESTART
       * window too), and this one evaluates only the stencil, as after an
       * evaluation of x. */
      if (spent()) throw MaxEvalReached{};
    } else if (!just_evaluated) {
      const RepeatUse how = e ? repeats_.use(*e, true) : RepeatUse::Evaluate;
      if (how == RepeatUse::Centre) {
        /* NUMERICAL takes the loss at x from the caller and evaluates the
         * stencil alone: the recorded loss is that loss. */
        if (spent()) throw MaxEvalReached{};
        repeats_.note_reuse(x, *e, how, true);
        y = e->loss;
      } else {
        if (spent()) throw MaxEvalReached{};
        if (e) repeats_.note_evaluated_again(x, *e, true);
        y = evaluate(x);
      }
    }
    Vec g;
    if (!gp_.gradient_at(x, y, g)) {
      switch (gp_.failure->kind) {
        case GradientFailure::Budget:
          throw MaxEvalReached{};
        case GradientFailure::Stencil:
        case GradientFailure::NotFinite:
          throw GradientNotFinite{*gp_.failure};
        case GradientFailure::None:
          break;
      }
      throw std::logic_error(
          "DLIB:BFGS: the gradient provider failed without a reason");
    }
    repeats_.record_gradient(x, g);
    return g;
  }

  /** The evaluations made: the losses, the finite-difference points and
   *  one per analytical gradient. */
  unsigned long evaluations() const { return evals_.load(); }

  /** Counts a loss answered from the record that no call has followed
   *  since (the class comment): a reuse, in the RESTART window (which may
   *  throw StagnationStop) and in DIAGNOSTICS. Called when the search
   *  ends; nothing when no such loss is pending. */
  void settle()
  {
    if (!pending_) return;
    const Pending p = std::move(*pending_);
    pending_.reset();
    repeats_.note_reuse(p.x, p.e, RepeatUse::Record, false);
  }

  /** The run's number of the recorded evaluation of `x`, if it has one. */
  std::optional<std::size_t> recorded_eval(const Vec& x) const
  {
    const auto e = repeats_.find(x);
    return e ? e->eval : std::nullopt;
  }

private:
  /* MAXEVAL is spent: an evaluation would throw MaxEvalReached. */
  bool spent() const { return maxeval_ > 0 && evals_.load() >= maxeval_; }

  double evaluate(const Vec& x)
  {
    if (spent()) throw MaxEvalReached{};
    evals_.fetch_add(1);
    const double y = host_(sp_.out(x));
    last_.x = x;
    last_.y = y;
    last_.count = evals_.load();
    last_.valid = true;
    last_.reused = false;
    repeats_.record_loss(x, y, latest_evaluation(host_));
    return y;
  }

  /* The last loss asked for -- an evaluation, or (`reused`) one answered
   * from the record -- and the count of evaluations after it. */
  struct LastEval {
    Vec           x;
    double        y = 0.0;
    unsigned long count = 0;
    bool          valid = false;
    bool          reused = false;
  };

  /* A loss answered from the record whose count waits for the next call:
   * its point and its entry. */
  struct Pending {
    Vec                               x;
    typename RepeatPoints<Vec>::Entry e;
  };

  const OptimizerSpec&         sp_;
  HostT&                       host_;
  const unsigned long          maxeval_;
  std::atomic<unsigned long>   evals_{0};   // before gp_, which holds its address
  GradientProvider<Vec,Runner> gp_;
  RepeatPoints<Vec>            repeats_;
  LastEval                     last_;
  std::optional<Pending>       pending_;
};

/* "A at its upper bound 1, B at its upper bound 1, each with a gradient that
 * points out of the box": why no step moves the point `x` (internal; `g`
 * its gradient), the rule of zero_bounded_variables -- each coordinate at a
 * bound its gradient points out of, with a gradient of zero, or with a step
 * (BFGS's identity: minus its gradient) below the rounding of its value. */
template <class Vec>
std::string no_step_detail(const OptimizerSpec& sp, const Vec& x, const Vec& g,
                           const Vec& lo, const Vec& hi)
{
  const double gap_eps = 1e-8;   // dlib's
  const auto real = sp.out(x);
  std::vector<std::string> held, flat, small;
  for (long i = 0; i < x.size(); ++i) {
    const auto k = static_cast<std::size_t>(i);
    const std::string label = k < sp.parameter_labels.size()
                                  ? sp.parameter_labels[k]
                                  : logidx::dim_label(k);
    std::ostringstream v;
    v.precision(10);
    v << real(i);
    const double tol = gap_eps * std::abs(x(i));
    if (lo(i) + tol >= x(i) && g(i) > 0)
      held.push_back(label + " at its lower bound " + v.str());
    else if (hi(i) - tol <= x(i) && g(i) < 0)
      held.push_back(label + " at its upper bound " + v.str());
    else if (g(i) == 0.0)
      flat.push_back(label + " = " + v.str());
    else
      small.push_back(label + " = " + v.str());
  }
  std::string out;
  for (std::size_t k = 0; k < held.size(); ++k)
    out += (k ? ", " : "") + held[k];
  if (!held.empty())
    out += held.size() == 1 ? ", with a gradient that points out of the box"
                            : ", each with a gradient that points out of the "
                              "box";
  const auto add = [&out](const std::vector<std::string>& part,
                          const char* one, const char* many) {
    if (part.empty()) return;
    out += out.empty() ? "" : "; ";
    for (std::size_t k = 0; k < part.size(); ++k)
      out += (k ? ", " : "") + part[k];
    out += part.size() == 1 ? one : many;
  };
  add(flat, ", with a gradient of zero", ", each with a gradient of zero");
  add(small, ", whose step is below the rounding of its value",
      ", whose steps are below the rounding of their values");
  return out;
}

/* "from A = 0.5, B = 0.25 the line search from the farthest step BFGS
 * takes there (0.125 of the steepest-descent step) came back to the point":
 * a NoStep of bfgs_box_search (d), at the internal point `x`. */
template <class Vec>
std::string came_back_detail(const OptimizerSpec& sp, const Vec& x,
                             double reach)
{
  std::ostringstream r;
  r.precision(10);
  r << reach;
  return "from " + describe_point(sp, x) +
         " the line search from the farthest step BFGS takes there (" +
         r.str() + " of the steepest-descent step) found no lower loss and "
         "came back to the point";
}

/* "the iteration from A = 0.25 (the point of eval 9) made no evaluation:
 * it asked only for points the run had evaluated, and took the search back
 * to A = 0.5 (the point of eval 7), where it had stood with the same loss
 * and gradient": a Revisit of bfgs_box_search (e), from the internal point
 * `from` to `x`, whose recorded evaluations are `from_eval` and `eval`. The
 * points are given to 17 digits: they can differ beyond the tenth. */
template <class Vec>
std::string revisit_detail(const OptimizerSpec& sp, const Vec& from,
                           std::optional<std::size_t> from_eval, const Vec& x,
                           std::optional<std::size_t> eval)
{
  const auto of_eval = [](std::optional<std::size_t> e) {
    return e ? " (the point of eval " + std::to_string(*e) + ")"
             : std::string();
  };
  return "the iteration from " + describe_point(sp, from, 17) +
         of_eval(from_eval) +
         " made no evaluation: it asked only for points the run had "
         "evaluated, and took the search back to " +
         describe_point(sp, x, 17) + of_eval(eval) +
         ", where it had stood with the same loss and gradient";
}

/** The stop test of PARAM tol_gradinf: the search ends once the largest
 *  component of the gradient, in absolute value, is below `tol` -- the
 *  max-norm CERES's gradient_tolerance tests. dlib's
 *  gradient_norm_stop_strategy tests the L2 norm, length(g), which for n
 *  components can be up to sqrt(n) times the largest: the same
 *  tol_gradinf ended DLIB:BFGS later than CERES. The stop strategy
 *  interface of dlib's optimisers (should_continue_search). */
class GradientMaxStop {
public:
  explicit GradientMaxStop(double tol) : tol_(tol) {}
  template <class T>
  bool should_continue_search(const T&, double, const T& g) {
    return !(dlib::max(dlib::abs(g)) < tol_);
  }
private:
  double tol_;
};

}  // namespace detail

/* DLIB:BFGS on any host with `double operator()(const Vec&)` (real-space
 * parameters; an analytical_gradient member and repeat_reuse_blocker() are
 * optional, see repeat_points.h); run_dlib_bfgs binds HPO_Host, the unit
 * tests a stub host. */
template<class Vec,class Runner,class HostT>
  void run_dlib_bfgs_impl(const OptimizerSpec& sp, HostT& host)
  {
    using namespace dlib;
    static_assert(std::is_same<Vec, matrix<double,0,1>>::value,
                  "DLIB:BFGS runs on dlib::matrix<double,0,1>");
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

    /* The loss and the gradient count every evaluation in one counter, the
     * GradientProvider's (detail::BfgsObjective): DLIB's stop strategies
     * count iterations, so MAXEVAL is enforced here -- the objective throws
     * detail::MaxEvalReached once it is spent, and the catch site below
     * treats it as a graceful termination, the best-so-far tracked by the
     * host preserved. GRAD_STRATEGY NUMERICAL (default) takes finite
     * differences (fd_stencil.h: PARAM fd_step, fd_kernel, diff_method);
     * ANALYTICAL and HYBRID the host's analytical gradient. */
    detail::BfgsObjective<Vec, Runner, HostT> objective(sp, host, maxeval);
    const auto feval = [&objective](const matrix<double,0,1>& xin) {
      return objective.loss(xin);
    };
    const auto grad = [&objective](const matrix<double,0,1>& xin) {
      return objective.gradient(xin);
    };
    /* An iteration whose step cannot move the point (bfgs_box_search (a)):
     * counted, the first journalled. It asks for no evaluation, so it
     * checks the stop flag as an evaluation does. */
    bool not_run_noted = false;
    const auto on_not_run = [&](const matrix<double,0,1>& xp) {
      if (core::SignalHandler::stop_requested())
        throw core::ShutdownRequested{};
      HPOLog::instance().record_line_search_not_run();
      if (not_run_noted) return;
      not_run_noted = true;
      HPOLog::instance().info(
          "DLIB:BFGS: the step from " + describe_point(sp, xp) +
          " cannot move the point within the OPTIM box: the line search is "
          "not run, and the point, its loss and its gradient stand "
          "(DIAGNOSTICS counts such steps)");
    };

    /* dlib's loop runs on the loss divided by the length of its first
     * gradient (detail::bfgs_box_search_scaled): the journal gives that
     * length when the search has it, and the stop strategy `make` builds
     * for it. */
    const auto noted_stop = [&](auto make) {
      return [make](std::optional<double> length) {
        std::ostringstream oss;
        if (length) {
          oss << "DLIB:BFGS: the search runs on the loss divided by the "
                 "length of its first gradient, " << *length
              << " (over the coordinates its first step can move), and takes "
                 "its steps in units of that gradient; FTOL_ABS and PARAM "
                 "tol_gradinf are divided by it as well, and the records keep "
                 "the loss";
        } else {
          oss << "DLIB:BFGS: the gradient at the start of the search has a "
                 "length of 0 over the coordinates its first step can move, "
                 "or one the loss cannot be divided by: the search runs on "
                 "the loss itself";
        }
        HPOLog::instance().info(oss.str());
        return make(length.value_or(1.0));
      };
    };

    /* How the search ends (run_end.h): MAXEVAL (MaxEvalReached, from the
     * loss or the gradient), a gradient that is not finite, no step within
     * the box (bfgs_box_search (c)), an iteration that moved without an
     * evaluation (bfgs_box_search (e)), or -- when the loop returns on its
     * stop strategy -- that strategy's test: `converged` (FTOL_ABS or
     * tol_gradinf) with the stop line's `test` text. `make_stop` builds the
     * stop strategy for the loss divided by the scale it is given. */
    auto run_with_eval_cap = [&](auto&& search, auto&& make_stop,
                                 EndReason converged,
                                 const std::string& test) {
      try {
        double f_value = 0.0;
        double scale = 1.0;
        const detail::BfgsBoxResult r = detail::bfgs_box_search_scaled(
            std::forward<decltype(search)>(search),
            noted_stop(std::forward<decltype(make_stop)>(make_stop)), feval,
            grad, x, lo, hi, f_value, on_not_run,
            [&objective] { return objective.evaluations(); }, scale);
        /* A loss answered from the record that no call followed is counted
         * now (it can end the run's window: StagnationStop). */
        objective.settle();
        if (r.end == detail::BfgsBoxEnd::Revisit)
          report_end("DLIB:BFGS", EndReason::Revisit,
                     std::to_string(objective.evaluations()) +
                         " evaluation(s): " +
                         detail::revisit_detail(
                             sp, r.revisit_from,
                             objective.recorded_eval(r.revisit_from), x,
                             objective.recorded_eval(x)));
        else if (r.end == detail::BfgsBoxEnd::NoStep)
          report_end("DLIB:BFGS", EndReason::NoStep,
                     std::to_string(objective.evaluations()) +
                         " evaluation(s): " +
                         (r.came_back
                              ? detail::came_back_detail(sp, x, r.reach)
                              : detail::no_step_detail(sp, x, r.gradient, lo,
                                                       hi)));
        else
          report_end("DLIB:BFGS", converged,
                     std::to_string(objective.evaluations()) +
                         " evaluation(s): " + test);
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
        /* The MAXEVAL stop line (the only one: no "stopped:" line), and
         * the run's status. */
        std::ostringstream oss;
        oss << "DLIB:BFGS -- MAXEVAL=" << maxeval
            << " reached; preserving best-so-far";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().info(oss.str());
        record_end("DLIB:BFGS", EndReason::MaxEval);
      } catch (const detail::GradientNotFinite& e) {
        report_gradient_stop("DLIB:BFGS", e.info);
      }
    };

    if (const auto tol_gradinf = detail::dlib_param(sp, "tol_gradinf")) {
      std::ostringstream test;
      test << "the largest component of the gradient fell below PARAM "
              "tol_gradinf = " << *tol_gradinf;
      const double tol = *tol_gradinf;
      run_with_eval_cap(bfgs_search_strategy(),
                        [tol](double c) {
                          return detail::GradientMaxStop(tol / c);
                        },
                        EndReason::GradTol, test.str());
      return;
    }

    const double ftol_abs = sp.stop.ftol_abs.value_or(1.0e-7);
    std::ostringstream test;
    test << "the loss changed by less than FTOL_ABS = " << ftol_abs
         << (sp.stop.ftol_abs ? "" : " (default)")
         << " from one iteration to the next";
    run_with_eval_cap(bfgs_search_strategy(),
                      [ftol_abs](double c) {
                        return objective_delta_stop_strategy(ftol_abs / c);
                      },
                      EndReason::FtolAbs, test.str());
  }

/* Production entry point used by run_inner(). */
template<class Vec,class Runner>
  void run_dlib_bfgs(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_dlib_bfgs_impl<Vec, Runner>(sp, host);
  }

}   // namespace tadah::hpo

#pragma once
#include <dlib/global_optimization.h>
#include <dlib/optimization.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>

#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah::hpo {

namespace detail {

/* DLIB's stop strategies count BFGS *iterations*, not function evals.
 * At dim=D with central differences a single iteration consumes
 * 2*D + line_search + step ≈ 2*D + 2..5 evals, so passing the user's
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
  if (sp.num_threads)
    throw std::runtime_error("DLIB:BFGS does not support THREADS");
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

  for (const auto& kv : sp.param) {
    if (kv.first != "fd_step" && kv.first != "tol_gradinf") {
      throw std::runtime_error(
          "DLIB:BFGS does not support PARAM " + kv.first +
          "; supported PARAM keys are fd_step and tol_gradinf");
    }
  }

  const double fd_step = dlib_param(sp, "fd_step").value_or(1.0e-6);
  if (!(fd_step > 0.0))
    throw std::runtime_error("DLIB:BFGS requires PARAM fd_step > 0");

  const auto tol_gradinf = dlib_param(sp, "tol_gradinf");
  if (tol_gradinf && !(*tol_gradinf > 0.0))
    throw std::runtime_error("DLIB:BFGS requires PARAM tol_gradinf > 0");

  if (tol_gradinf && sp.stop.ftol_abs)
    throw std::runtime_error(
        "DLIB:BFGS accepts either FTOL_ABS or PARAM tol_gradinf, but not both");
}

}  // namespace detail

template<class Vec,class Runner>
  void run_dlib_global(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    using namespace dlib;

    /* ------- transform bounds into (possibly) log space ---------- */
    const matrix<double,0,1> lo = mat(sp.low);
    const matrix<double,0,1> hi = mat(sp.high);

    /* ------- thread pool ---------------------------------------- */
    const unsigned nthreads = sp.num_threads.value_or(1);  // single thread as default
    dlib::thread_pool tp(nthreads);

    double solver_eps = sp.stop.ftol_abs.value_or(1e-3);
    constexpr uint64_t HUGE_UNSIGNED = std::numeric_limits<uint64_t>::max();
    uint64_t maxeval = sp.stop.maxeval.value_or(HUGE_UNSIGNED);

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

    /* ------- seed dlib with every point from sp.optim_init.
     * find_min_global() treats `initial_function_evals` as already-known
     * (x,y) pairs, so we evaluate each seed once here and hand the
     * pair(s) over.  This guarantees the user's priors (CONFIG, WARM,
     * PRIOR, RANDOM/LHS/SOBOL chain entries) are part of the explored
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
        const double y_seed = feval(x_seed);
        init.emplace_back(x_seed, y_seed);
        if (maxeval != HUGE_UNSIGNED && maxeval > 0) --maxeval;
        if (maxeval == 0) break;
      }
    }

    find_min_global(tp, feval,
        lo, hi, sp.is_int,
        max_function_calls(maxeval),
        FOREVER,
        solver_eps,
        init
        );
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

    /* Wrap the objective so each call increments an external counter.
     * When the user's MAXEVAL is reached, throw detail::MaxEvalReached;
     * the catch site below treats it as graceful termination so the
     * best-so-far tracked by HPO_Host is preserved. */
    unsigned long evals_used = 0;
    auto feval = [&, maxeval](const matrix<double,0,1>& xin)->double {
      if (maxeval > 0 && evals_used >= maxeval) throw detail::MaxEvalReached{};
      ++evals_used;
      return host( sp.out(xin) );
    };
    const auto grad = derivative(feval,
        detail::dlib_param(sp, "fd_step").value_or(1.0e-6));

    auto run_with_eval_cap = [&](auto&& search, auto&& stop) {
      try {
        find_min_box_constrained(std::forward<decltype(search)>(search),
                                 std::forward<decltype(stop)>(stop),
                                 feval, grad, x, lo, hi);
      } catch (const detail::MaxEvalReached&) {
        /* MAXEVAL is enforced externally because DLIB's stop strategies
         * count iterations (not evals). Hitting it is a normal termination
         * once BFGS has consumed its eval budget; the preamble explaining
         * this design fires once per process so subsequent stage runs in
         * BASINS don't repeat it. */
        static std::atomic_flag preamble_emitted = ATOMIC_FLAG_INIT;
        if (!preamble_emitted.test_and_set(std::memory_order_relaxed)) {
            const std::string preamble =
                "DLIB:BFGS — MAXEVAL is enforced externally by HPO "
                "(DLIB stop strategies count iterations, not evals). "
                "Subsequent MAXEVAL hits in this run will be summarised tersely.";
            std::cout << "[HPO] " << preamble << "\n";
            HPOLog::instance().info(preamble);
        }
        std::ostringstream oss;
        oss << "DLIB:BFGS — MAXEVAL=" << maxeval
            << " reached; preserving best-so-far";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().info(oss.str());
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

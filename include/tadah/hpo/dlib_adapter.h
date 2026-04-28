#pragma once
#include <dlib/global_optimization.h>
#include <dlib/optimization.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tadah::hpo {

namespace detail {

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

  if (sp.local)
    throw std::runtime_error(
        "DLIB:BFGS is a standalone optimiser and does not support LOCAL blocks");

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

    /* ------- optional user initial guesses ----------------------- */
    std::vector<function_evaluation> init;
    // for (const auto& v : sp.optim_init)
    //     init.emplace_back( mat(sp.in(v)), 0.0);          // y filled by dlib later

    /* ------- thread pool ---------------------------------------- */
    const unsigned nthreads = sp.num_threads.value_or(1);  // single thread as default
    dlib::thread_pool tp(nthreads);

    double solver_eps = sp.stop.ftol_abs.value_or(1e-3);
    constexpr uint64_t HUGE_UNSIGNED = std::numeric_limits<uint64_t>::max();
    uint64_t maxeval = sp.stop.maxeval.value_or(HUGE_UNSIGNED);

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

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

    auto feval = [&](const matrix<double,0,1>& xin)->double
    {   return host( sp.out(xin) ); };
    const auto grad = derivative(feval,
        detail::dlib_param(sp, "fd_step").value_or(1.0e-6));

    const auto maxeval = sp.stop.maxeval.has_value()
        ? static_cast<unsigned long>(*sp.stop.maxeval)
        : 0ul;

    if (const auto tol_gradinf = detail::dlib_param(sp, "tol_gradinf")) {
      if (maxeval > 0) {
        find_min_box_constrained(
            bfgs_search_strategy(),
            gradient_norm_stop_strategy(*tol_gradinf, maxeval),
            feval, grad, x, lo, hi);
      } else {
        find_min_box_constrained(
            bfgs_search_strategy(),
            gradient_norm_stop_strategy(*tol_gradinf),
            feval, grad, x, lo, hi);
      }
      return;
    }

    const double ftol_abs = sp.stop.ftol_abs.value_or(1.0e-7);
    if (maxeval > 0) {
      find_min_box_constrained(
          bfgs_search_strategy(),
          objective_delta_stop_strategy(ftol_abs, maxeval),
          feval, grad, x, lo, hi);
    } else {
      find_min_box_constrained(
          bfgs_search_strategy(),
          objective_delta_stop_strategy(ftol_abs),
          feval, grad, x, lo, hi);
    }
  }

}   // namespace tadah::hpo

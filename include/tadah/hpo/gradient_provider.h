#pragma once
/*
 * gradient_provider.h — unified gradient backend for gradient-based HPO
 * optimizers (DLIB:BFGS, CERES:LBFGS/WOLFE/NLCG/STEEPEST_DESCENT, future
 * NLOPT:LD_*).
 *
 * Two strategies, selected by OptimizerSpec::grad_strategy (or default):
 *
 *   GradStrategy::Numerical  — Ceres DynamicNumericDiffCostFunction
 *                              (CENTRAL by default; FORWARD or RIDDERS
 *                              selectable via PARAM diff_method).
 *   GradStrategy::Autodiff   — stub that throws "not yet implemented".
 *                              Reserved for a later phase that templates
 *                              the train/predict pipeline through Context/
 *                              MODELS plus an implicit-function-theorem
 *                              rule for the LAPACK solve.
 *
 * MAXEVAL is enforced at this layer by counting host calls in the shared
 * eval_counter. Optimizers receive `gradient(x,g) -> false` (and likewise
 * `cost(x) -> +inf`) once the counter overflows or the host returns a
 * non-finite value, so they can terminate cleanly without exceptions
 * propagating across Ceres' Solve() frames (Ceres is not exception-safe).
 *
 * Lifetime: capture by reference is safe because the provider is built
 * inside a `run_<lib>_<algo>` adapter and consumed within the same call.
 */

#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/optimizer_spec.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TADAH_USE_CERES
/* Ceres' bundled miniglog defines a `CHECK(...)` macro that may
 * collide with Catch2 or other libraries.  We push/pop the caller's
 * `CHECK` around the includes so the caller's definition is restored
 * after this header.  The Ceres include directory is added with
 * `-isystem` (see HPO/CMakeLists.txt) so the redefinition warning is
 * suppressed when the conflict is unavoidable. */
#pragma push_macro("CHECK")
#include <ceres/ceres.h>                              // pulls miniglog setup
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/numeric_diff_options.h>
#include <ceres/types.h>
#pragma pop_macro("CHECK")
#endif

namespace tadah::hpo {

template<class Vec, class Runner>
struct GradientProvider {
  /** Return host(sp.out(x)). +inf on bad eval or MAXEVAL overflow. */
  std::function<double(const Vec& x)>            cost;

  /** Fill g with d(host(sp.out(x)))/dx. Return false on bad eval or
   *  MAXEVAL overflow; caller should treat as graceful termination. */
  std::function<bool(const Vec& x, Vec& g)>      gradient;
};

/* Build a GradientProvider for `host`. Honours sp.grad_strategy
 * (default: Numerical). The eval_counter is shared with the optimizer
 * driver, so external MAXEVAL accounting works uniformly.
 *
 * `HostT` is templated so tests can substitute a stub callable in
 * place of HPO_Host<Vec,Runner>. Production call sites pass an
 * HPO_Host& and the type is deduced. The only requirement on HostT
 * is `double operator()(const Vec&)` returning the loss in real space
 * (sp.out is applied internally before the call). */
template<class Vec, class Runner, class HostT>
GradientProvider<Vec, Runner>
make_gradient_provider(const OptimizerSpec&     sp,
                       HostT&                   host,
                       std::atomic<unsigned long>* eval_counter,
                       unsigned long             maxeval /*0 = unbounded*/)
{
  const GradStrategy strat = sp.grad_strategy.value_or(GradStrategy::Numerical);
  GradientProvider<Vec, Runner> gp;

  // ---- Cost: direct host call, shared counter, return +inf on failure.
  gp.cost = [&host, &sp, eval_counter, maxeval](const Vec& x)->double {
    if (maxeval > 0 && eval_counter->load() >= maxeval)
      return std::numeric_limits<double>::infinity();
    eval_counter->fetch_add(1);
    const double y = host(sp.out(x));
    if (!std::isfinite(y)) return std::numeric_limits<double>::infinity();
    return y;
  };

  // ---- Autodiff: stubbed. Phase-2 work.
  if (strat == GradStrategy::Autodiff) {
    gp.gradient = [](const Vec&, Vec&)->bool {
      throw std::runtime_error(
          "GRAD_STRATEGY=AUTODIFF is not yet implemented. It requires "
          "templating Context and the train/predict pipeline through "
          "MODELS, plus an implicit-function-theorem rule for the LAPACK "
          "solve. Use GRAD_STRATEGY=NUMERICAL for now.");
    };
    return gp;
  }

#ifndef TADAH_USE_CERES
  gp.gradient = [](const Vec&, Vec&)->bool {
    throw std::runtime_error(
        "Numerical gradient via Ceres requires the build to be configured "
        "with -DTADAH_ENABLE_CERES=ON.");
  };
  return gp;
#else

  // ---- Numerical: Ceres DynamicNumericDiffCostFunction over a host-
  // wrapping CostFunctor that bumps the same shared counter.
  struct CostFunctor {
    HostT*                       host;
    const OptimizerSpec*         sp;
    std::atomic<unsigned long>*  counter;
    unsigned long                maxeval;
    int                          dim;

    bool operator()(double const* const* params, double* residuals) const {
      if (maxeval > 0 && counter->load() >= maxeval) return false;
      counter->fetch_add(1);
      Vec vlog;
      vlog.set_size(dim);
      for (int i = 0; i < dim; ++i) vlog(i) = params[0][i];
      const double y = (*host)(sp->out(vlog));
      if (!std::isfinite(y)) return false;
      residuals[0] = y;
      return true;
    }
  };

  ceres::NumericDiffOptions ndo;
  // PARAM fd_step: relative finite-diff step (alias of DLIB:BFGS PARAM).
  if (auto it = sp.param.find("fd_step"); it != sp.param.end())
    ndo.relative_step_size = it->second;

  // PARAM diff_method: 0 = CENTRAL (default), 1 = FORWARD, 2 = RIDDERS.
  // RIDDERS extrapolates assuming smooth Taylor expansion; not advisable
  // near HPO's NaN-clamp plateau or for noisy LAMMPS-loss components.
  int diff_kind = 0;
  if (auto it = sp.param.find("diff_method"); it != sp.param.end())
    diff_kind = static_cast<int>(it->second);

  const int dim = static_cast<int>(sp.low.size());

  // Build the cost function with the chosen NumericDiff method and
  // share it via shared_ptr (the gradient lambda needs to outlive the
  // factory call).
  std::shared_ptr<ceres::CostFunction> cf;
  auto* functor = new CostFunctor{&host, &sp, eval_counter, maxeval, dim};

  using NDFC_Central = ceres::DynamicNumericDiffCostFunction<CostFunctor, ceres::CENTRAL>;
  using NDFC_Forward = ceres::DynamicNumericDiffCostFunction<CostFunctor, ceres::FORWARD>;
  using NDFC_Ridders = ceres::DynamicNumericDiffCostFunction<CostFunctor, ceres::RIDDERS>;

  if (diff_kind == 1) {
    auto* p = new NDFC_Forward(functor, ceres::TAKE_OWNERSHIP, ndo);
    p->AddParameterBlock(dim);
    p->SetNumResiduals(1);
    cf.reset(p);
  } else if (diff_kind == 2) {
    auto* p = new NDFC_Ridders(functor, ceres::TAKE_OWNERSHIP, ndo);
    p->AddParameterBlock(dim);
    p->SetNumResiduals(1);
    cf.reset(p);
  } else {
    auto* p = new NDFC_Central(functor, ceres::TAKE_OWNERSHIP, ndo);
    p->AddParameterBlock(dim);
    p->SetNumResiduals(1);
    cf.reset(p);
  }

  gp.gradient = [cf, dim](const Vec& x, Vec& g)->bool {
    std::vector<double> px(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) px[static_cast<std::size_t>(i)] = x(i);
    const double* params[1] = { px.data() };
    double residual = 0.0;
    std::vector<double> jac(static_cast<std::size_t>(dim), 0.0);
    double* jacs[1] = { jac.data() };
    if (!cf->Evaluate(params, &residual, jacs)) return false;
    if (!std::isfinite(residual)) return false;
    g.set_size(dim);
    for (int i = 0; i < dim; ++i) {
      const double v = jac[static_cast<std::size_t>(i)];
      if (!std::isfinite(v)) return false;
      g(i) = v;
    }
    return true;
  };

  return gp;
#endif
}

}  // namespace tadah::hpo

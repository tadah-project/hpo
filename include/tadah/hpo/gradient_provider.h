#pragma once
/*
 * gradient_provider.h -- unified gradient backend for gradient-based HPO
 * optimizers (DLIB:BFGS, CERES:LBFGS/WOLFE/NLCG/STEEPEST_DESCENT, future
 * NLOPT:LD_*).
 *
 * Three strategies, selected by OptimizerSpec::grad_strategy (or default):
 *
 *   GradStrategy::Numerical  -- Ceres DynamicNumericDiffCostFunction
 *                              (CENTRAL by default; FORWARD or RIDDERS
 *                              selectable via PARAM diff_method).
 *   GradStrategy::Analytical -- strict closed-form. The caller provides
 *                              an `analytical_grad_fn` callback that
 *                              fills g[] and reports the theta indices
 *                              it could not differentiate via a
 *                              `needs_fd` out-list. Throws if any index
 *                              is left -- use HYBRID for graceful
 *                              fallback.
 *   GradStrategy::Hybrid     -- analytical for registered theta + masked
 *                              central-difference for unregistered
 *                              indices, in one gradient(x, g) call.
 *                              Same `analytical_grad_fn` signature.
 *                              Degenerate "no theta registered" case:
 *                              skip the FD pass and return g=0 so a
 *                              gradient-based inner terminates cleanly
 *                              and an outer derivative-free sampler
 *                              keeps progressing. Use NUMERICAL to opt
 *                              into full FD on the whole vector.
 *
 * The factory's `analytical_grad_fn` parameter has the signature
 *
 *     bool(const Vec& x, Vec& g, std::vector<int>& needs_fd)
 *
 * Returning false signals a bad eval (e.g. r.was_nan_clamped or
 * non-finite W*). Returning true means "g[i] is filled for every i not
 * in needs_fd; for i in needs_fd, the caller must source g[i]
 * elsewhere". The Hybrid strategy then runs central differences on
 * exactly those indices, sharing the same eval_counter so MAXEVAL
 * accounting works uniformly. Numerical ignores the callback.
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
#include <sstream>
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

/* Caller-injected analytical gradient lambda. Fills g with the
 * closed-form gradient where it can; pushes any theta indices for which no
 * analytical derivative is registered onto `needs_fd`. Returns false on
 * bad eval (clamp, NaN). The HOST captures whatever state it needs
 * (Trainer factor, Analytics, registry...) before binding this. */
template<class Vec>
using AnalyticalGradFn =
    std::function<bool(const Vec& x, Vec& g, std::vector<int>& needs_fd)>;

/* Build a GradientProvider for `host`. Honours sp.grad_strategy
 * (default: Numerical). The eval_counter is shared with the optimizer
 * driver, so external MAXEVAL accounting works uniformly.
 *
 * `HostT` is templated so tests can substitute a stub callable in
 * place of HPO_Host<Vec,Runner>. Production call sites pass an
 * HPO_Host& and the type is deduced. The only requirement on HostT
 * is `double operator()(const Vec&)` returning the loss in real space
 * (sp.out is applied internally before the call).
 *
 * `analytical_grad_fn` is required for ANALYTICAL/HYBRID strategies and
 * ignored for NUMERICAL. Defaulted to nullptr so existing Phase-1
 * callers (which only used NUMERICAL/AUTODIFF) keep working without
 * code change. */
template<class Vec, class Runner, class HostT>
GradientProvider<Vec, Runner>
make_gradient_provider(const OptimizerSpec&     sp,
                       HostT&                   host,
                       std::atomic<unsigned long>* eval_counter,
                       unsigned long             maxeval /*0 = unbounded*/,
                       AnalyticalGradFn<Vec>    analytical_grad_fn = nullptr)
{
  const GradStrategy strat = sp.grad_strategy.value_or(GradStrategy::Numerical);
  GradientProvider<Vec, Runner> gp;

  /* Chain rule for the analytical path: the host differentiates the loss
   * with respect to the REAL parameters (dL/dtheta), while the optimiser
   * iterates in INTERNAL coordinates u (ln / asinh / unit-box affine), so
   * g_u[i] = dL/dtheta_i * dtheta_i/du_i. The FD paths perturb u directly
   * and need nothing. Without a bound space the legacy ln mask applies
   * (dtheta/du = theta = exp(u) on log dims). */
  auto pull_to_internal = [&sp](const Vec& x, Vec& g,
                                const std::vector<int>& skip) {
    for (long i = 0; i < g.size(); ++i) {
      bool fd = false;
      for (int k : skip) if (k == i) { fd = true; break; }
      if (fd) continue;
      double jac = 1.0;
      if (sp.space) {
        jac = sp.space->dreal_dinternal(static_cast<std::size_t>(i), x(i));
      } else if (static_cast<std::size_t>(i) < sp.log_scaled.size() &&
                 sp.log_scaled[static_cast<std::size_t>(i)]) {
        jac = std::exp(x(i));
      }
      g(i) *= jac;
    }
  };

  // ---- Cost: direct host call, shared counter, return +inf on failure.
  gp.cost = [&host, &sp, eval_counter, maxeval](const Vec& x)->double {
    if (maxeval > 0 && eval_counter->load() >= maxeval)
      return std::numeric_limits<double>::infinity();
    eval_counter->fetch_add(1);
    const double y = host(sp.out(x));
    if (!std::isfinite(y)) return std::numeric_limits<double>::infinity();
    return y;
  };

  // ---- ANALYTICAL: strict mode. Defer to the caller-supplied callback;
  // any index left in `needs_fd` is a hard error pointing the user at
  // HYBRID.
  if (strat == GradStrategy::Analytical) {
    if (!analytical_grad_fn) {
      gp.gradient = [](const Vec&, Vec&)->bool {
        throw std::runtime_error(
            "GRAD_STRATEGY=ANALYTICAL requires an analytical gradient "
            "callback, but none was provided. This is an internal wiring "
            "error -- the HPO host must register an analytical gradient "
            "function before constructing the GradientProvider.");
      };
      return gp;
    }
    gp.gradient = [analytical_grad_fn, eval_counter, maxeval, pull_to_internal]
                  (const Vec& x, Vec& g)->bool {
      if (maxeval > 0 && eval_counter->load() >= maxeval) return false;
      eval_counter->fetch_add(1);
      std::vector<int> needs_fd;
      if (!analytical_grad_fn(x, g, needs_fd)) return false;
      pull_to_internal(x, g, needs_fd);
      if (!needs_fd.empty()) {
        /* The host does not say which cause applies; name them all. The
         * settings-level ones (analytical_gate.h) were refused at startup,
         * so what is left is per parameter or per point. */
        std::ostringstream oss;
        oss << "GRAD_STRATEGY=ANALYTICAL: no analytical gradient for "
            << needs_fd.size() << " parameter"
            << (needs_fd.size() == 1 ? "" : "s")
            << " at this point (indices: [";
        for (std::size_t i = 0; i < needs_fd.size(); ++i) {
          if (i) oss << ", ";
          oss << needs_fd[i];
        }
        oss << "]): a parameter without an analytical derivative, a "
               "candidate whose fit cannot be differentiated (rejected "
               "before training, or an ill-conditioned fit), or a loss term "
               "without an analytical derivative at this point. Use "
               "GRAD_STRATEGY=HYBRID to fill these with finite differences.";
        throw std::runtime_error(oss.str());
      }
      for (long i = 0; i < g.size(); ++i)
        if (!std::isfinite(g(i))) return false;
      return true;
    };
    return gp;
  }

  // ---- HYBRID: analytical fills what it can; remaining indices get a
  // masked central-difference pass that varies only the unregistered
  // indices. fd_step honours PARAM fd_step (default 1e-6). The shared
  // eval_counter accounts for FD calls so MAXEVAL stays correct.
  if (strat == GradStrategy::Hybrid) {
    if (!analytical_grad_fn) {
      gp.gradient = [](const Vec&, Vec&)->bool {
        throw std::runtime_error(
            "GRAD_STRATEGY=HYBRID requires an analytical gradient "
            "callback, but none was provided. This is an internal wiring "
            "error.");
      };
      return gp;
    }
    double fd_step = 1.0e-6;
    if (auto it = sp.param.find("fd_step"); it != sp.param.end())
      fd_step = it->second;

    gp.gradient = [&host, &sp, eval_counter, maxeval,
                   analytical_grad_fn, fd_step, pull_to_internal]
                  (const Vec& x, Vec& g)->bool {
      const auto over = [&]() {
        return maxeval > 0 && eval_counter->load() >= maxeval;
      };
      if (over()) return false;
      eval_counter->fetch_add(1);

      std::vector<int> needs_fd;
      if (!analytical_grad_fn(x, g, needs_fd)) return false;
      pull_to_internal(x, g, needs_fd);

      /* All-FD gate: when every theta falls back to FD the masked-FD pass
       * degenerates into pure central differences on the whole vector.
       * On LAMMPS-driven loss components (`PC_LAMMPS`, see
       * `compute_analytical_gradient_from_trainer`'s sims-grad chain
       * failure path) this is both very expensive (2.dim host calls per
       * gradient eval, each running LAMMPS) AND noisy -- the relaxation
       * cliff at `--hotvar` boundaries makes one perturbed point land on
       * `FAIL_SCORE`, blowing up that gradient component. Gradient-based
       * inner optimisers (`LD_LBFGS`, DLIB:BFGS, Ceres) then monopolise
       * the eval budget on hopeless local descents and the outer global
       * sampler (e.g. `GN_MLSL_LDS`) never gets to explore. Skip the FD
       * pass instead and surface g = analytical_pass_zeros. The
       * gradient-based local search receives g=0 and terminates as
       * "converged"; an outer global driver simply samples another
       * point. NUMERICAL strategy remains the explicit opt-in for users
       * who genuinely want full FD. */
      if (needs_fd.size() == static_cast<std::size_t>(g.size())) {
        for (long i = 0; i < g.size(); ++i)
          if (!std::isfinite(g(i))) g(i) = 0.0;
        return true;
      }

      // Central differences over the masked indices.
      for (int idx : needs_fd) {
        const double xi = x(idx);
        const double h  = fd_step * std::max(std::abs(xi), 1.0);

        Vec xp = x; xp(idx) = xi + h;
        Vec xm = x; xm(idx) = xi - h;

        if (over()) return false;
        eval_counter->fetch_add(1);
        const double yp = host(sp.out(xp));
        if (!std::isfinite(yp)) return false;

        if (over()) return false;
        eval_counter->fetch_add(1);
        const double ym = host(sp.out(xm));
        if (!std::isfinite(ym)) return false;

        const double gi = (yp - ym) / (2.0 * h);
        if (!std::isfinite(gi)) return false;
        g(idx) = gi;
      }

      // Sanity check: every index should now be finite (the analytical
      // pass may have left bad entries even outside needs_fd).
      for (long i = 0; i < g.size(); ++i)
        if (!std::isfinite(g(i))) return false;
      return true;
    };
    return gp;
  }

  // ---- NUMERICAL: two kernels.
  //   inhouse -- central differences in INTERNAL coordinates with
  //              h_i = fd_step * max(|u_i|, 1): in the unit box
  //              (OPTIM_SCALE AUTO) exactly fd_step * width on every
  //              dimension, which is what PARAM fd_step is documented to
  //              mean there; the same kernel serves HYBRID's masked pass.
  //   ceres   -- Ceres DynamicNumericDiffCostFunction with per-parameter
  //              delta max(sqrt(eps), fd_step*|u|): the pre-existing
  //              kernel, kept as the default under OPTIM_SCALE NONE so
  //              legacy runs stay byte-identical; unsuitable in the unit
  //              box because the delta collapses to 1.5e-8 next to u = 0
  //              (every lower bound).
  // PARAM fd_kernel inhouse|ceres overrides the default either way.
  bool use_ceres_kernel = !(sp.space && sp.space->normalised());
  if (auto it = sp.param_raw.find("fd_kernel"); it != sp.param_raw.end()) {
    if (it->second == "ceres" || it->second == "CERES") use_ceres_kernel = true;
    else if (it->second != "inhouse" && it->second != "INHOUSE")
      throw std::runtime_error("PARAM fd_kernel must be inhouse (default) or ceres (got '" +
                               it->second + "')");
  }
  if (!use_ceres_kernel) {
    double fd_step = 1.0e-6;
    if (auto it = sp.param.find("fd_step"); it != sp.param.end())
      fd_step = it->second;
    if (!(fd_step > 0.0) || !std::isfinite(fd_step))
      throw std::runtime_error("PARAM fd_step must be a positive finite fraction of the box width");
    gp.gradient = [&host, &sp, eval_counter, maxeval, fd_step]
                  (const Vec& x, Vec& g)->bool {
      const auto over = [&]() {
        return maxeval > 0 && eval_counter->load() >= maxeval;
      };
      /* Centre evaluation first: a non-finite loss AT x means the
       * gradient is meaningless (bad-eval contract shared with the
       * Ceres kernel, which evaluates the residual at x as well). */
      if (over()) return false;
      eval_counter->fetch_add(1);
      if (!std::isfinite(host(sp.out(x)))) return false;
      const long n = x.size();
      g.set_size(n);
      for (long idx = 0; idx < n; ++idx) {
        const double xi = x(idx);
        const double h  = fd_step * std::max(std::abs(xi), 1.0);
        Vec xp = x; xp(idx) = xi + h;
        Vec xm = x; xm(idx) = xi - h;
        if (over()) return false;
        eval_counter->fetch_add(1);
        const double yp = host(sp.out(xp));
        if (!std::isfinite(yp)) return false;
        if (over()) return false;
        eval_counter->fetch_add(1);
        const double ym = host(sp.out(xm));
        if (!std::isfinite(ym)) return false;
        const double gi = (yp - ym) / (2.0 * h);
        if (!std::isfinite(gi)) return false;
        g(idx) = gi;
      }
      return true;
    };
    return gp;
  }

#ifndef TADAH_USE_CERES
  gp.gradient = [](const Vec&, Vec&)->bool {
    throw std::runtime_error(
        "PARAM fd_kernel ceres requires the build to be configured "
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

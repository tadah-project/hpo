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
 * Why a gradient() call returned false is recorded in
 * GradientProvider::failure (GradientFailure): the MAXEVAL budget
 * (Budget), a finite-difference gradient that is not finite because an
 * evaluation of its stencil failed (Stencil: a non-finite loss, or a
 * failed one -- FAILSCORE or a graded failure, loss_is_feasible() -- whose
 * difference quotient overflows, as the default FAILSCORE, the largest
 * double, does against a usable loss for any step h below 1/2), or a
 * gradient not finite for another reason (NotFinite). The adapters turn Stencil and NotFinite into the search's
 * own ending, "STOPPED -- gradient not finite: <cause>"
 * (gradient_stop_status), where they used to report MAXEVAL or success.
 * A failed stencil evaluation whose difference stays finite (a finite
 * FAILSCORE such as 1e9 with a step that is not tiny) gives a finite,
 * steep gradient, which the optimisers use.
 *
 * Lifetime: capture by reference is safe because the provider is built
 * inside a `run_<lib>_<algo>` adapter and consumed within the same call.
 */

#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pc_fail_grading.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

/* Why GradientProvider::gradient() returned false. */
enum class GradientFailure {
  None,      /* the last call returned a finite gradient */
  Budget,    /* MAXEVAL was spent before the gradient was complete */
  Stencil,   /* a finite-difference gradient that is not finite because an
                evaluation of its stencil failed */
  NotFinite  /* a gradient that is not finite for another reason */
};

/* The failure of the last gradient() call: `cause` is the fixed reason
 * the status and the journal give, `detail` what failed, for the journal
 * (the point, the stencil point and its loss). */
struct GradientFailureInfo {
  GradientFailure kind = GradientFailure::None;
  std::string     cause;
  std::string     detail;
};

/* The cause of a Stencil failure. */
inline const std::string kGradientStencilCause =
    "a failed evaluation in the finite-difference stencil";

/* The run's status when a gradient that is not finite ended the search
 * (HPOLog::set_status; the journal trailer of a run it ends). */
inline std::string gradient_stop_status(const GradientFailureInfo& f)
{
  return "STOPPED -- gradient not finite: " + f.cause;
}

/* Records that a gradient that is not finite ended the search of
 * `backend` ("DLIB:BFGS"): a journal warning, the same line on stdout, and
 * the run's status. `ended` says how the optimiser ended right after it
 * rejected the point, or what HPO did ("" for neither). */
inline void report_gradient_stop(const std::string& backend,
                                 const GradientFailureInfo& f,
                                 const std::string& ended = "")
{
  std::string line = backend + " stopped: gradient not finite: " + f.cause;
  if (!f.detail.empty()) line += " (" + f.detail + ")";
  if (!ended.empty()) line += "; " + ended;
  line += "; best-so-far preserved";
  std::cout << "[HPO] " << line << "\n";
  HPOLog::instance().warning(line);
  HPOLog::instance().set_status(gradient_stop_status(f));
}

template<class Vec, class Runner>
struct GradientProvider {
  /** Return host(sp.out(x)). +inf on bad eval or MAXEVAL overflow. */
  std::function<double(const Vec& x)>            cost;

  /** Fill g with d(host(sp.out(x)))/dx. Return false on bad eval or
   *  MAXEVAL overflow; `failure` then says which. */
  std::function<bool(const Vec& x, Vec& g)>      gradient;

  /** Why the last gradient() call returned false (kind None after a call
   *  that returned true). Shared with the gradient callable, which fills
   *  it. */
  std::shared_ptr<GradientFailureInfo> failure =
      std::make_shared<GradientFailureInfo>();
};

namespace detail {

/* Fills GradientFailureInfo for the finite-difference paths: what a
 * stencil evaluation or quotient says about the gradient at x. */
template<class Vec>
struct StencilReport {
  const OptimizerSpec*  sp;
  GradientFailureInfo*  out;

  /* Whether `y` is a failed evaluation: not a usable loss. */
  bool failed(double y) const {
    const double fs = sp->failscore.value_or(
        std::numeric_limits<double>::infinity());
    return !loss_is_feasible(y, fs);
  }
  std::string label(long i) const {
    const auto k = static_cast<std::size_t>(i);
    return (k < sp->parameter_labels.size() && !sp->parameter_labels[k].empty())
               ? sp->parameter_labels[k] : logidx::dim_label(k);
  }
  /* "CGRID2B[2] = 1.4": dimension i of the internal point u, in real
   * units. */
  std::string at(const Vec& u, long i) const {
    const Vec r = sp->out(u);
    std::ostringstream os;
    os << label(i) << " = " << std::setprecision(10) << r(i);
    return os.str();
  }
  static std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
  }

  void budget() const { *out = {GradientFailure::Budget, "MAXEVAL spent", ""}; }

  /* The loss at x itself is not finite. */
  void centre(const Vec& x, double y) const {
    std::ostringstream os;
    os << "the loss at ";
    const long n = x.size();
    for (long i = 0; i < n; ++i) os << (i ? ", " : "") << at(x, i);
    os << " is " << num(y);
    *out = {GradientFailure::Stencil, kGradientStencilCause, os.str()};
  }

  /* The derivative along dimension i, from the stencil evaluations
   * `pts` (point, loss), is not finite: Stencil when one of them failed,
   * NotFinite (an overflow between usable losses) otherwise. */
  void stencil(const Vec& x, long i,
               const std::vector<std::pair<Vec, double>>& pts) const {
    std::ostringstream os;
    os << "at " << at(x, i) << ", ";
    std::vector<const std::pair<Vec, double>*> bad;
    for (const auto& p : pts)
      if (failed(p.second)) bad.push_back(&p);
    if (!bad.empty()) {
      os << (bad.size() == 1 ? "the stencil point " : "the stencil points ");
      for (std::size_t k = 0; k < bad.size(); ++k)
        os << (k ? " and " : "") << at(bad[k]->first, i);
      os << " failed: loss ";
      for (std::size_t k = 0; k < bad.size(); ++k)
        os << (k ? " and " : "") << num(bad[k]->second);
      *out = {GradientFailure::Stencil, kGradientStencilCause, os.str()};
      return;
    }
    os << "the losses ";
    for (std::size_t k = 0; k < pts.size(); ++k)
      os << (k ? " and " : "") << num(pts[k].second) << " at "
         << at(pts[k].first, i);
    os << " overflow their difference";
    *out = {GradientFailure::NotFinite,
            "the finite-difference quotient overflowed", os.str()};
  }

  /* The central difference of dimension i, from yp at xp and ym at xm,
   * is not finite. */
  void quotient(const Vec& x, long i, const Vec& xp, double yp,
                const Vec& xm, double ym) const {
    stencil(x, i, {{xp, yp}, {xm, ym}});
  }

  /* A stencil evaluation (xs, ys) of dimension i is not finite; the other
   * point of the pair was not evaluated. */
  void point(const Vec& x, long i, const Vec& xs, double ys) const {
    std::ostringstream os;
    os << "at " << at(x, i) << ", the stencil point " << at(xs, i)
       << " failed: loss " << num(ys);
    *out = {GradientFailure::Stencil, kGradientStencilCause, os.str()};
  }

  void analytical(const std::string& what) const {
    *out = {GradientFailure::NotFinite,
            "the analytical gradient " + what, ""};
  }
};

}  // namespace detail

/* Caller-injected analytical gradient lambda. Fills g with the
 * closed-form gradient where it can; pushes any theta indices for which no
 * analytical derivative is registered onto `needs_fd`. Returns false on
 * bad eval (clamp, NaN). The HOST captures whatever state it needs
 * (Trainer factor, Analytics, registry...) before binding this. */
template<class Vec>
using AnalyticalGradFn =
    std::function<bool(const Vec& x, Vec& g, std::vector<int>& needs_fd)>;

namespace detail {
template<class HostT, class Vec, class = void>
struct has_analytical_gradient : std::false_type {};
template<class HostT, class Vec>
struct has_analytical_gradient<HostT, Vec, std::void_t<decltype(
    std::declval<HostT&>().analytical_gradient(
        std::declval<const Vec&>(), std::declval<Vec&>(),
        std::declval<std::vector<int>&>()))>> : std::true_type {};
}  // namespace detail

/* The host's analytical_gradient member as an AnalyticalGradFn when sp asks
 * for ANALYTICAL/HYBRID; nullptr for NUMERICAL. A stub host without the
 * member (unit tests of the optimiser adapters) also gets nullptr, and the
 * provider then reports the missing callback if an analytical strategy is
 * requested. */
template<class Vec, class HostT>
AnalyticalGradFn<Vec> host_analytical_grad_fn(const OptimizerSpec& sp,
                                              HostT& host)
{
  if constexpr (detail::has_analytical_gradient<HostT, Vec>::value) {
    if (sp.grad_strategy && *sp.grad_strategy != GradStrategy::Numerical)
      return [&host](const Vec& x_in, Vec& g_out,
                     std::vector<int>& nfd) -> bool {
        return host.analytical_gradient(x_in, g_out, nfd);
      };
  }
  (void)sp; (void)host;
  return nullptr;
}

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
  /* Each gradient callable owns a share of gp.failure and fills it. */
  const std::shared_ptr<GradientFailureInfo> failure = gp.failure;
  const detail::StencilReport<Vec> report{&sp, failure.get()};

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
    gp.gradient = [analytical_grad_fn, eval_counter, maxeval, pull_to_internal,
                   failure, report]
                  (const Vec& x, Vec& g)->bool {
      *failure = {};
      if (maxeval > 0 && eval_counter->load() >= maxeval) {
        report.budget();
        return false;
      }
      eval_counter->fetch_add(1);
      std::vector<int> needs_fd;
      if (!analytical_grad_fn(x, g, needs_fd)) {
        report.analytical("could not be formed at this point");
        return false;
      }
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
        if (!std::isfinite(g(i))) {
          report.analytical("is not finite at this point");
          return false;
        }
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
                   analytical_grad_fn, fd_step, pull_to_internal,
                   failure, report]
                  (const Vec& x, Vec& g)->bool {
      *failure = {};
      const auto over = [&]() {
        if (maxeval > 0 && eval_counter->load() >= maxeval) {
          report.budget();
          return true;
        }
        return false;
      };
      if (over()) return false;
      eval_counter->fetch_add(1);

      std::vector<int> needs_fd;
      if (!analytical_grad_fn(x, g, needs_fd)) {
        report.analytical("could not be formed at this point");
        return false;
      }
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
        if (!std::isfinite(yp)) {
          report.point(x, idx, xp, yp);
          return false;
        }

        if (over()) return false;
        eval_counter->fetch_add(1);
        const double ym = host(sp.out(xm));
        if (!std::isfinite(ym)) {
          report.point(x, idx, xm, ym);
          return false;
        }

        const double gi = (yp - ym) / (2.0 * h);
        if (!std::isfinite(gi)) {
          report.quotient(x, idx, xp, yp, xm, ym);
          return false;
        }
        g(idx) = gi;
      }

      // Sanity check: every index should now be finite (the analytical
      // pass may have left bad entries even outside needs_fd).
      for (long i = 0; i < g.size(); ++i)
        if (!std::isfinite(g(i))) {
          report.analytical("is not finite at this point");
          return false;
        }
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
  //              legacy runs keep their pre-1.5 finite-difference kernel;
  //              unsuitable in the unit box because the delta collapses to
  //              1.5e-8 next to u = 0 (every lower bound).
  // PARAM fd_kernel inhouse|ceres overrides the default either way.
  bool use_ceres_kernel = !(sp.space && sp.space->normalised());
  if (auto it = sp.param_raw.find("fd_kernel"); it != sp.param_raw.end()) {
    if (it->second == "ceres" || it->second == "CERES") use_ceres_kernel = true;
    else if (it->second == "inhouse" || it->second == "INHOUSE")
      use_ceres_kernel = false;
    else
      throw std::runtime_error("PARAM fd_kernel must be inhouse (default) or ceres (got '" +
                               it->second + "')");
  }
  if (!use_ceres_kernel) {
    double fd_step = 1.0e-6;
    if (auto it = sp.param.find("fd_step"); it != sp.param.end())
      fd_step = it->second;
    if (!(fd_step > 0.0) || !std::isfinite(fd_step))
      throw std::runtime_error("PARAM fd_step must be a positive finite fraction of the box width");
    gp.gradient = [&host, &sp, eval_counter, maxeval, fd_step, failure,
                   report]
                  (const Vec& x, Vec& g)->bool {
      *failure = {};
      const auto over = [&]() {
        if (maxeval > 0 && eval_counter->load() >= maxeval) {
          report.budget();
          return true;
        }
        return false;
      };
      /* Centre evaluation first: a non-finite loss AT x means the
       * gradient is meaningless (bad-eval contract shared with the
       * Ceres kernel, which evaluates the residual at x as well). */
      if (over()) return false;
      eval_counter->fetch_add(1);
      if (const double y0 = host(sp.out(x)); !std::isfinite(y0)) {
        report.centre(x, y0);
        return false;
      }
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
        if (!std::isfinite(yp)) {
          report.point(x, idx, xp, yp);
          return false;
        }
        if (over()) return false;
        eval_counter->fetch_add(1);
        const double ym = host(sp.out(xm));
        if (!std::isfinite(ym)) {
          report.point(x, idx, xm, ym);
          return false;
        }
        const double gi = (yp - ym) / (2.0 * h);
        if (!std::isfinite(gi)) {
          report.quotient(x, idx, xp, yp, xm, ym);
          return false;
        }
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
  /* What the functor evaluated during one gradient() call, to tell why
   * Ceres' gradient is not finite. */
  struct StencilLog {
    bool                                 budget = false;
    std::vector<std::pair<Vec, double>>  evals;   /* (internal point, loss) */
  };
  struct CostFunctor {
    HostT*                       host;
    const OptimizerSpec*         sp;
    std::atomic<unsigned long>*  counter;
    unsigned long                maxeval;
    int                          dim;
    StencilLog*                  log;   /* owned by the gradient callable */

    bool operator()(double const* const* params, double* residuals) const {
      if (maxeval > 0 && counter->load() >= maxeval) {
        log->budget = true;
        return false;
      }
      counter->fetch_add(1);
      Vec vlog;
      vlog.set_size(dim);
      for (int i = 0; i < dim; ++i) vlog(i) = params[0][i];
      const double y = (*host)(sp->out(vlog));
      log->evals.emplace_back(vlog, y);
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
  const auto stencil_log = std::make_shared<StencilLog>();
  auto* functor = new CostFunctor{&host, &sp, eval_counter, maxeval, dim,
                                  stencil_log.get()};

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

  /* The evaluations Ceres made along dimension i: the points that differ
   * from x there, and x itself for FORWARD differences (a quotient over
   * the centre). */
  const bool forward = (diff_kind == 1);
  const auto points_of = [stencil_log, forward](const Vec& x, long i) {
    std::vector<std::pair<Vec, double>> pts;
    for (const auto& e : stencil_log->evals) {
      bool centre = true, along_i = true;
      for (long k = 0; k < x.size(); ++k) {
        if (e.first(k) == x(k)) continue;
        centre = false;
        if (k != i) along_i = false;
      }
      if ((!centre && along_i) || (centre && forward)) pts.push_back(e);
    }
    return pts;
  };

  gp.gradient = [cf, dim, stencil_log, points_of, failure, report]
                (const Vec& x, Vec& g)->bool {
    *failure = {};
    *stencil_log = {};
    std::vector<double> px(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) px[static_cast<std::size_t>(i)] = x(i);
    const double* params[1] = { px.data() };
    double residual = 0.0;
    std::vector<double> jac(static_cast<std::size_t>(dim), 0.0);
    double* jacs[1] = { jac.data() };
    const bool ok = cf->Evaluate(params, &residual, jacs);
    if (stencil_log->budget) {
      report.budget();
      return false;
    }
    /* A non-finite loss stops the functor: at x itself, or at a stencil
     * point of the dimension it perturbs. */
    for (const auto& e : stencil_log->evals) {
      if (std::isfinite(e.second)) continue;
      long along = -1;
      for (long k = 0; k < x.size(); ++k)
        if (e.first(k) != x(k)) { along = k; break; }
      if (along < 0) report.centre(x, e.second);
      else           report.point(x, along, e.first, e.second);
      return false;
    }
    if (!ok || !std::isfinite(residual)) {
      /* Every evaluation was finite, and Ceres still failed. */
      *failure = {GradientFailure::NotFinite,
                  "Ceres' numeric differentiation failed", ""};
      return false;
    }
    g.set_size(dim);
    for (int i = 0; i < dim; ++i) {
      const double v = jac[static_cast<std::size_t>(i)];
      if (!std::isfinite(v)) {
        report.stencil(x, i, points_of(x, i));
        return false;
      }
      g(i) = v;
    }
    return true;
  };

  return gp;
#endif
}

}  // namespace tadah::hpo

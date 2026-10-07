#pragma once
/*
 * ceres_adapter.h -- drivers for the Ceres GradientProblemSolver family.
 *
 * Algos exposed in this rollout:
 *   CERES:LBFGS              -- line_search_direction_type = LBFGS, Wolfe LS
 *   CERES:WOLFE              -- direction = BFGS,  Wolfe LS (full-Hessian BFGS)
 *   CERES:NLCG               -- direction = NONLINEAR_CONJUGATE_GRADIENT
 *   CERES:STEEPEST_DESCENT   -- direction = STEEPEST_DESCENT, Armijo LS
 *
 * Algos explicitly deferred (not exposed; rejected at dispatch in
 * run_policy_base.h):
 *   CERES:LEVENBERG_MARQUARDT, CERES:DOGLEG, CERES:SUBSPACE_DOGLEG
 * -- degenerate over a scalar loss; require residual-mode loss exposure
 * (objective semantics change Sigmar^2 vs. mean(r)) and benefit most from
 * autodiff. Both belong to a later phase.
 *
 * Bounds caveat: GradientProblemSolver does not enforce box constraints.
 * Users wanting bounds-enforced gradient optimization should choose
 * NLOPT:LD_LBFGS or DLIB:BFGS instead.
 *
 * Design echoes dlib_adapter.h:
 *   - validators throw std::runtime_error with a message naming the
 *     unsupported feature and an alternative algo when relevant;
 *   - MAXEVAL is enforced externally via a shared atomic counter (Ceres
 *     iterations != host evaluations);
 *   - bad host evals (non-finite loss, MAXEVAL overflow) are reported via
 *     FirstOrderFunction::Evaluate -> false, which Ceres turns into a
 *     termination type, rather than as exceptions. A point without a
 *     usable gradient (GradientProvider::failure: Stencil or NotFinite) is
 *     such a failed evaluation: Ceres' line search steps back from it. When the
 *     solve then ends with FAILURE right after it -- at the starting point,
 *     "Initial cost and jacobian evaluation failed", or a line search that
 *     finds no valid step -- the gradient ended the search, which reports
 *     it with its own status (report_gradient_stop);
 *   - no exception crosses ceres::Solve (Ceres is not exception-safe): an
 *     exception from the host -- the run-level stops core::ShutdownRequested
 *     and the RESTART StagnationStop, durable_best::Error, any other error --
 *     is captured in Evaluate, the solve is aborted (HostExceptionAbort),
 *     and the original is rethrown with its own type after Solve returns;
 *     test_hpo_termination.cpp's [.shutdown_ceres] case runs that path
 *     under a real SIGINT.
 */

#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TADAH_USE_CERES
/* See gradient_provider.h for the macro-collision rationale.  Push/pop
 * `CHECK` around Ceres' includes so the caller's CHECK is restored. */
#pragma push_macro("CHECK")
#include <ceres/ceres.h>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>
#pragma pop_macro("CHECK")
#endif

namespace tadah::hpo {

/* Throws std::runtime_error with a tailored "use CERES:LBFGS" message
 * when spec asks for a deferred Ceres NLLS algorithm
 * (LEVENBERG_MARQUARDT / DOGLEG / SUBSPACE_DOGLEG). Called from
 * run_inner() before the caps lookup so users get this specific
 * diagnostic rather than a generic "unknown optimizer" error. No-op
 * for any non-deferred (lib, algo). */
inline void reject_deferred_ceres_nlls(const OptimizerSpec& spec) {
    if (spec.lib != "CERES") return;
    if (spec.algo == "LEVENBERG_MARQUARDT" ||
        spec.algo == "DOGLEG"             ||
        spec.algo == "SUBSPACE_DOGLEG") {
        throw std::runtime_error(
            "CERES:" + spec.algo + " is deferred to a later phase "
            "(requires residual-mode loss exposure and autodiff support). "
            "Use CERES:LBFGS, CERES:WOLFE, CERES:NLCG, or "
            "CERES:STEEPEST_DESCENT.");
    }
}

namespace detail {

/* Common checks shared by all Ceres GradientProblemSolver validators. */
inline void validate_ceres_common_(const OptimizerSpec& sp,
                                   const std::string& algo)
{
    if (sp.lib != "CERES" || sp.algo != algo)
        throw std::runtime_error(
            "validate_ceres_" + algo + "_spec expects LIB=CERES and ALGO=" + algo);

    if (sp.inner)
        throw std::runtime_error(
            "CERES:" + algo + " is a standalone optimiser and does not "
            "support INNER blocks");

    if (sp.low.empty() || sp.low.size() != sp.high.size())
        throw std::runtime_error(
            "CERES:" + algo + " requires non-empty, same-sized low/high bounds");

    if (!sp.is_int.empty() &&
        std::any_of(sp.is_int.begin(), sp.is_int.end(),
                    [](bool v) { return v; })) {
        throw std::runtime_error(
            "CERES:" + algo + " does not support integer-valued optimisation "
            "variables; use DLIB:GFS, NLOPT global algorithms, or TADAH:GRID");
    }

    if (sp.vector_storage)
        throw std::runtime_error(
            "CERES:" + algo + " does not support VECTOR_ARR; use DLIB:GFS or "
            "an NLopt global algorithm");
    if (sp.pop)
        throw std::runtime_error("CERES:" + algo + " does not support POPULATION");
    /* GradientProblemSolver draws no random numbers; DLIB:BFGS refuses SEED
     * likewise. */
    if (sp.seed)
        throw std::runtime_error("CERES:" + algo + " does not support SEED");
    if (sp.num_threads)
        throw std::runtime_error("CERES:" + algo + " does not support THREADS "
            "(host evaluation is single-threaded inside one optimizer step)");
    if (sp.step)
        throw std::runtime_error("CERES:" + algo + " does not support STEP "
            "(Ceres chooses the initial line-search step itself; PARAM "
            "max_ls_iter caps the line-search iterations)");

    if (sp.stop.maxtime)
        throw std::runtime_error("CERES:" + algo + " does not support MAXTIME");
    if (sp.stop.stopval)
        throw std::runtime_error("CERES:" + algo + " does not support STOPVAL");
    if (sp.stop.ftol_abs)
        throw std::runtime_error("CERES:" + algo + " uses relative tolerance; "
            "use FTOL_REL (mapped to function_tolerance) instead of FTOL_ABS");
    if (sp.stop.xtol_abs)
        throw std::runtime_error("CERES:" + algo + " uses relative tolerance; "
            "use XTOL_REL (mapped to parameter_tolerance) instead of XTOL_ABS");

    if (sp.stop.maxeval && *sp.stop.maxeval == 0)
        throw std::runtime_error("CERES:" + algo + " requires MAXEVAL > 0 when provided");

    /* PARAM keys recognised across all four CERES gradient algos:
     *   fd_step           -- finite-difference step (fd_stencil.h)
     *   diff_method       -- 0 CENTRAL (default), 1 FORWARD (2, RIDDERS,
     *                        is refused)
     *   (fd_kernel        -- the step rule, a word: param_raw)
     *   tol_gradinf       -- gradient_tolerance (infinity norm)
     *   max_lbfgs_rank    -- LBFGS only
     *   nlcg_type         -- NLCG only: 0 FR, 1 PR, 2 HS
     *   max_iter          -- max_num_iterations cap (defaults to MAXEVAL or INT_MAX)
     *   max_ls_iter       -- max line search step size iterations
     */
    /* PARAM fd_step, fd_kernel and diff_method are the gradient's finite
     * differences: validate_hpo_setup checks them on every phase that takes
     * them, and the gradient before its first evaluation (fd_settings_of,
     * fd_stencil.h), with the same message on every backend -- a number for
     * fd_kernel too, which this list refused as unsupported. */
    static const std::vector<std::string> common_params = {
        "tol_gradinf", "max_iter", "max_ls_iter"
    };
    for (const auto& kv : sp.param) {
        if (is_fd_param(kv.first)) continue;
        if (std::find(common_params.begin(), common_params.end(), kv.first)
                == common_params.end()
            && kv.first != "max_lbfgs_rank"   // LBFGS-only, accepted everywhere
                                              // for forward-compat; ignored elsewhere
            && kv.first != "nlcg_type") {     // NLCG-only, idem
            throw std::runtime_error(
                "CERES:" + algo + " does not support PARAM " + kv.first +
                "; supported: fd_step, fd_kernel, diff_method, tol_gradinf, "
                "max_iter, max_ls_iter, max_lbfgs_rank, nlcg_type");
        }
    }
    if (auto it = sp.param.find("tol_gradinf"); it != sp.param.end())
        if (!(it->second > 0.0))
            throw std::runtime_error("CERES:" + algo + " requires PARAM tol_gradinf > 0");
}

inline void validate_ceres_lbfgs_spec(const OptimizerSpec& sp) {
    validate_ceres_common_(sp, "LBFGS");
    if (auto it = sp.param.find("max_lbfgs_rank"); it != sp.param.end())
        if (!(it->second >= 1.0))
            throw std::runtime_error("CERES:LBFGS requires PARAM max_lbfgs_rank >= 1");
}
inline void validate_ceres_wolfe_spec(const OptimizerSpec& sp) {
    validate_ceres_common_(sp, "WOLFE");
}
inline void validate_ceres_nlcg_spec(const OptimizerSpec& sp) {
    validate_ceres_common_(sp, "NLCG");
    if (auto it = sp.param.find("nlcg_type"); it != sp.param.end()) {
        const int v = static_cast<int>(it->second);
        if (v < 0 || v > 2)
            throw std::runtime_error(
                "CERES:NLCG PARAM nlcg_type must be 0 (FLETCHER_REEVES), "
                "1 (POLAK_RIBIERE), or 2 (HESTENES_STIEFEL)");
    }
}
inline void validate_ceres_steepest_spec(const OptimizerSpec& sp) {
    validate_ceres_common_(sp, "STEEPEST_DESCENT");
}

#ifdef TADAH_USE_CERES

/* FirstOrderFunction adapter: bridges Ceres' const-double-pointer API to
 * the GradientProvider's Vec-typed callbacks. Returns false on bad-eval
 * propagated from the host or finite-diff functor; Ceres translates
 * that into LINE_SEARCH_FAILED / NUMERICAL_FAILURE termination.
 *
 * An exception from the host or the gradient provider (durable_best::Error,
 * StagnationStop, core::ShutdownRequested, ...) is captured in *host_exc --
 * the first one wins -- and reported to Ceres as a failed evaluation. Once
 * captured, every later call fails at once without touching the host,
 * HostExceptionAbort ends the solve, and run_ceres_gradient_problem
 * rethrows the original after Solve returns. */
template<class Vec, class Runner>
class HPO_FirstOrderFunction : public ceres::FirstOrderFunction {
public:
    HPO_FirstOrderFunction(GradientProvider<Vec,Runner>* gp, int dim,
                           std::exception_ptr* host_exc)
        : gp_(gp), dim_(dim), host_exc_(host_exc) {}

    bool Evaluate(const double* parameters,
                  double*       cost,
                  double*       gradient) const override {
        last_rejected_ = false;
        if (*host_exc_) return false;
        try {
            Vec x;
            x.set_size(dim_);
            for (int i = 0; i < dim_; ++i) x(i) = parameters[i];

            const double y = gp_->cost(x);
            if (!std::isfinite(y)) return false;
            *cost = y;

            if (gradient) {
                /* y is the loss of exactly x, just evaluated: NUMERICAL
                 * and HYBRID take it as the loss at the point
                 * (gradient_at) and do not evaluate x again. */
                Vec g;
                if (!gp_->gradient_at(x, y, g)) {
                    const GradientFailure k = gp_->failure->kind;
                    if (k == GradientFailure::Stencil ||
                        k == GradientFailure::NotFinite) {
                        last_rejected_ = true;
                        rejected_ = *gp_->failure;
                    }
                    return false;
                }
                for (int i = 0; i < dim_; ++i) gradient[i] = g(i);
            }
            return true;
        } catch (...) {
            *host_exc_ = std::current_exception();
            return false;
        }
    }

    int NumParameters() const override { return dim_; }

    /* Whether the last Evaluate() failed on a gradient that is not
     * finite, and why. */
    bool last_rejected() const { return last_rejected_; }
    const GradientFailureInfo& rejected() const { return rejected_; }

private:
    GradientProvider<Vec,Runner>* gp_;
    int dim_;
    mutable bool                  last_rejected_ = false;
    mutable GradientFailureInfo   rejected_;
    std::exception_ptr* host_exc_;
};

/* Ends the solve (SOLVER_ABORT -> USER_FAILURE) at the first iteration
 * boundary after a host exception was captured. A failed evaluation alone
 * normally ends it through a line-search failure; this makes the stop
 * independent of how Ceres treats the failed trial step. */
class HostExceptionAbort : public ceres::IterationCallback {
public:
    explicit HostExceptionAbort(const std::exception_ptr* host_exc)
        : host_exc_(host_exc) {}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) override {
        return *host_exc_ ? ceres::SOLVER_ABORT : ceres::SOLVER_CONTINUE;
    }
private:
    const std::exception_ptr* host_exc_;
};

inline std::optional<double> ceres_param(const OptimizerSpec& sp,
                                         const std::string& key)
{
    const auto it = sp.param.find(key);
    if (it == sp.param.end()) return std::nullopt;
    return it->second;
}

/* Map TADAH StopCriteria + PARAM keys onto Ceres' tolerance fields and
 * (for NLCG) the conjugate-gradient subtype. Returns options ready to
 * pass to ceres::Solve. */
inline ceres::GradientProblemSolver::Options
make_ceres_options(const OptimizerSpec& sp,
                   ceres::LineSearchDirectionType dir,
                   ceres::LineSearchType ls_type)
{
    ceres::GradientProblemSolver::Options opt;
    opt.line_search_direction_type   = dir;
    opt.line_search_type             = ls_type;
    opt.minimizer_progress_to_stdout = false;
    opt.logging_type                 = ceres::SILENT;

    /* Tolerance mapping. */
    opt.function_tolerance  = sp.stop.ftol_rel.value_or(1e-6);
    opt.parameter_tolerance = sp.stop.xtol_rel.value_or(1e-8);
    opt.gradient_tolerance  = ceres_param(sp, "tol_gradinf").value_or(1e-10);

    /* Iteration cap: PARAM max_iter overrides; otherwise we use a high
     * sentinel and rely on the external eval-counter MAXEVAL gate. */
    if (auto v = ceres_param(sp, "max_iter"))
        opt.max_num_iterations = static_cast<int>(*v);
    else
        opt.max_num_iterations = std::numeric_limits<int>::max() / 2;

    if (auto v = ceres_param(sp, "max_ls_iter"))
        opt.max_num_line_search_step_size_iterations = static_cast<int>(*v);

    /* LBFGS-only: max_lbfgs_rank (default 20). Ignored for other dirs. */
    if (dir == ceres::LBFGS) {
        opt.max_lbfgs_rank = static_cast<int>(
            ceres_param(sp, "max_lbfgs_rank").value_or(20.0));
    }

    /* NLCG subtype. */
    if (dir == ceres::NONLINEAR_CONJUGATE_GRADIENT) {
        const int t = static_cast<int>(ceres_param(sp, "nlcg_type").value_or(0.0));
        switch (t) {
            case 1: opt.nonlinear_conjugate_gradient_type = ceres::POLAK_RIBIERE;   break;
            case 2: opt.nonlinear_conjugate_gradient_type = ceres::HESTENES_STIEFEL; break;
            default: opt.nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;
        }
    }

    return opt;
}

/* Common driver: build provider, run Ceres, report. Algo-specific
 * adapters (run_ceres_*) wrap this with the right direction + line
 * search type. Generic over the host (`double operator()(const Vec&)`;
 * analytical_gradient optional) so unit tests can drive it with a stub. */
template<class Vec, class Runner, class HostT>
inline void run_ceres_gradient_problem(
    const OptimizerSpec&         sp,
    HostT&                       host,
    ceres::LineSearchDirectionType dir,
    ceres::LineSearchType        ls_type,
    const std::string&           algo_label)
{
    if (sp.optim_init.empty())
        throw std::runtime_error(
            "CERES:" + algo_label + " requires a starting point; HPO should "
            "provide one from the current config");

    const auto start = sp.in(sp.optim_init.front());
    if (start.size() != sp.low.size())
        throw std::runtime_error(
            "CERES:" + algo_label + " initial point dimension does not match "
            "the search space");

    const int dim = static_cast<int>(start.size());
    const unsigned long maxeval =
        sp.stop.maxeval.has_value()
            ? static_cast<unsigned long>(*sp.stop.maxeval)
            : 0ul;
    std::atomic<unsigned long> evals_used{0};

    /* ANALYTICAL/HYBRID: route through the host's analytical_gradient
     * member function. NUMERICAL leaves the callback null so the
     * provider takes finite differences (fd_stencil.h). */
    AnalyticalGradFn<Vec> agf = host_analytical_grad_fn<Vec>(sp, host);
    auto gp = make_gradient_provider<Vec, Runner>(
        sp, host, &evals_used, maxeval, agf);

    /* GradientProblem takes ownership of the FirstOrderFunction; `fof`
     * stays valid while `problem` lives. */
    std::exception_ptr host_exc;
    auto owned = std::make_unique<HPO_FirstOrderFunction<Vec, Runner>>(
        &gp, dim, &host_exc);
    const HPO_FirstOrderFunction<Vec, Runner>* fof = owned.get();
    ceres::GradientProblem problem(owned.release());

    auto opts = make_ceres_options(sp, dir, ls_type);
    HostExceptionAbort abort_on_host_exc(&host_exc);
    opts.callbacks.push_back(&abort_on_host_exc);

    std::vector<double> x(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) x[i] = start[static_cast<std::size_t>(i)];

    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(opts, problem, x.data(), &summary);

    /* The host's exception is the outcome (fatal persistence failure,
     * stagnation restart, shutdown); Ceres' own termination is moot. */
    if (host_exc) std::rethrow_exception(host_exc);

    std::ostringstream oss;
    oss << "CERES:" << algo_label
        << " -- termination: " << ceres::TerminationTypeToString(summary.termination_type)
        << " | iters=" << summary.iterations.size()
        << " | host_evals=" << evals_used.load()
        << " | initial_cost=" << summary.initial_cost
        << " | final_cost="   << summary.final_cost;
    HPOLog::instance().info(oss.str());

    /* Surface a degraded-termination diagnostic so the journal records
     * cause when the run did not converge. */
    if (summary.termination_type != ceres::CONVERGENCE) {
        std::ostringstream oss2;
        oss2 << "CERES:" << algo_label
             << " -- non-convergent: " << summary.message;
        HPOLog::instance().warning(oss2.str());
    }

    /* A FAILURE right after an evaluation without a usable gradient: that
     * gradient ended the search. */
    if (summary.termination_type == ceres::FAILURE && fof->last_rejected())
        report_gradient_stop(
            "CERES:" + algo_label, fof->rejected(),
            "Ceres ended right after it (" +
                std::string(ceres::TerminationTypeToString(
                    summary.termination_type)) +
                ": " + summary.message + ")");
}

#else  // !TADAH_USE_CERES

template<class Vec, class Runner, class HostT>
inline void run_ceres_gradient_problem(const OptimizerSpec& sp,
                                       HostT&,
                                       int /*dir*/, int /*ls_type*/,
                                       const std::string& algo_label)
{
    (void)sp;
    throw std::runtime_error(
        "CERES:" + algo_label + " is unavailable: build was not configured "
        "with -DTADAH_ENABLE_CERES=ON");
}

#endif  // TADAH_USE_CERES

}  // namespace detail

/* ---- Public adapters ------------------------------------------------ */

template<class Vec, class Runner>
inline void run_ceres_lbfgs(const OptimizerSpec& sp,
                            HPO_Host<Vec, Runner>& host)
{
    detail::validate_ceres_lbfgs_spec(sp);
#ifdef TADAH_USE_CERES
    detail::run_ceres_gradient_problem<Vec, Runner>(
        sp, host, ceres::LBFGS, ceres::WOLFE, "LBFGS");
#else
    detail::run_ceres_gradient_problem<Vec, Runner>(sp, host, 0, 0, "LBFGS");
#endif
}

template<class Vec, class Runner>
inline void run_ceres_wolfe(const OptimizerSpec& sp,
                            HPO_Host<Vec, Runner>& host)
{
    detail::validate_ceres_wolfe_spec(sp);
#ifdef TADAH_USE_CERES
    detail::run_ceres_gradient_problem<Vec, Runner>(
        sp, host, ceres::BFGS, ceres::WOLFE, "WOLFE");
#else
    detail::run_ceres_gradient_problem<Vec, Runner>(sp, host, 0, 0, "WOLFE");
#endif
}

template<class Vec, class Runner>
inline void run_ceres_nlcg(const OptimizerSpec& sp,
                           HPO_Host<Vec, Runner>& host)
{
    detail::validate_ceres_nlcg_spec(sp);
#ifdef TADAH_USE_CERES
    detail::run_ceres_gradient_problem<Vec, Runner>(
        sp, host, ceres::NONLINEAR_CONJUGATE_GRADIENT, ceres::WOLFE, "NLCG");
#else
    detail::run_ceres_gradient_problem<Vec, Runner>(sp, host, 0, 0, "NLCG");
#endif
}

template<class Vec, class Runner>
inline void run_ceres_steepest_descent(const OptimizerSpec& sp,
                                       HPO_Host<Vec, Runner>& host)
{
    detail::validate_ceres_steepest_spec(sp);
#ifdef TADAH_USE_CERES
    detail::run_ceres_gradient_problem<Vec, Runner>(
        sp, host, ceres::STEEPEST_DESCENT, ceres::ARMIJO, "STEEPEST_DESCENT");
#else
    detail::run_ceres_gradient_problem<Vec, Runner>(sp, host, 0, 0, "STEEPEST_DESCENT");
#endif
}

}  // namespace tadah::hpo

#pragma once
/*
 * ceres_adapter.h -- drivers for the Ceres GradientProblemSolver family.
 *
 * The algorithm HPO runs:
 *   CERES:STEEPEST_DESCENT   -- direction = STEEPEST_DESCENT, Armijo LS
 *
 * Refused by the caps registry (optimizer_caps.cpp, kRefused):
 *   CERES:LBFGS, CERES:WOLFE, CERES:NLCG -- Ceres' Wolfe line search
 *   stalls at the score of a point outside the OPTIM box (below);
 *   CERES:LEVENBERG_MARQUARDT, CERES:DOGLEG, CERES:SUBSPACE_DOGLEG
 * -- degenerate over a scalar loss; require residual-mode loss exposure
 * (objective semantics change Sigmar^2 vs. mean(r)) and benefit most from
 * autodiff.
 *
 * Bounds: GradientProblemSolver does not enforce box constraints, so its
 * line searches propose points outside the OPTIM box. HPO_Host does not
 * evaluate such a point: it scores it as a failure graded by its distance
 * to the box (box_penalty.h), and the GradientProvider gives it the slope
 * of that score, which points away from the box, so a descent step leads
 * back into it and the line search steps back from the point; it counts
 * against MAXEVAL and is never the best. NLOPT:LD_LBFGS and DLIB:BFGS keep to the box.
 *
 * Design echoes dlib_adapter.h:
 *   - validators throw std::runtime_error with a message naming the
 *     unsupported feature and an alternative algo when relevant;
 *   - MAXEVAL is enforced externally via a shared atomic counter (Ceres
 *     iterations != host evaluations);
 *   - bad host evals (non-finite loss, MAXEVAL overflow) are reported via
 *     FirstOrderFunction::Evaluate -> false, which Ceres turns into a
 *     termination type, rather than as exceptions. An evaluation refused
 *     because MAXEVAL is spent ends the solve with FAILURE, which the
 *     adapter reports as MAXEVAL (below). A point without a usable
 *     gradient (GradientProvider::failure: Stencil or NotFinite) is such a
 *     failed evaluation: Ceres' line search steps back from it. When the
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
 *     under a real SIGINT;
 *   - the run's status says why the solve ended (run_end.h,
 *     run_ceres_gradient_problem): Ceres' own reason when it converged
 *     ("Function tolerance reached" FTOL_REL, "Parameter tolerance
 *     reached" XTOL_REL, "Gradient tolerance reached" tol_gradinf) or ran
 *     out of iterations (PARAM max_iter); MAXEVAL when an evaluation
 *     refused for the budget made it fail; "STOPPED -- CERES:<ALGO>:
 *     FAILURE (<head>)" for any other failure, where <head> is the head of
 *     Ceres' message, a fixed text without the numbers Ceres puts after it
 *     (ceres_message_head): "FAILURE (Numerical failure in line search)".
 *     The stop line and the WARNING give the whole message.
 */

#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/run_end.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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

/* Ceres' least-squares solvers (LEVENBERG_MARQUARDT, DOGLEG,
 * SUBSPACE_DOGLEG) are refused by the caps registry (optimizer_caps.cpp:
 * kRefused), so every lookup -- the parser's at the block's closing line,
 * validate_hpo_setup, run_inner -- gives their reason and the four solvers
 * below. */

namespace detail {

/* The head of a Ceres termination message: its text before the first
 * ". ", ", " or ": ", or before its closing "." -- the words that name the
 * ending, without the numbers Ceres puts after them ("Numerical failure in
 * line search" of "Numerical failure in line search, failed to find a
 * valid step size, ... using initial_step_size: 1.25000e-02, initial_cost:
 * ..."). A text without such a stop is its own head. A status that names
 * an ending of Ceres' own quotes the head, so it is the same for every run
 * that ends that way (run_ceres_gradient_problem); the stop line gives the
 * whole message. */
inline std::string ceres_message_head(const std::string& message)
{
    for (std::size_t i = 0; i < message.size(); ++i) {
        const char c = message[i];
        if ((c == '.' || c == ',' || c == ':') &&
            (i + 1 == message.size() || message[i + 1] == ' '))
            return message.substr(0, i);
    }
    return message;
}

/* `v` as a PARAM value in a message: "2.5", "3e+09". */
inline std::string ceres_value_text(double v)
{
    std::ostringstream o;
    o << v;
    return o.str();
}

/* Throws unless PARAM `key` of `sp`, when set, is an integer from `lo` to
 * 2147483647: Ceres takes it as an int (a fraction would be cut, and a
 * value past an int does not convert). `what` names Ceres' option. */
inline void validate_ceres_int_param_(const OptimizerSpec& sp,
                                      const std::string& algo,
                                      const std::string& key, int lo,
                                      const std::string& what)
{
    const auto it = sp.param.find(key);
    if (it == sp.param.end()) return;
    const double v = it->second;
    if (!(v >= lo && v <= std::numeric_limits<int>::max() &&
          std::floor(v) == v))
        throw std::runtime_error(
            "CERES:" + algo + " requires PARAM " + key +
            " to be an integer from " + std::to_string(lo) +
            " to 2147483647 (" + what + "), got " + ceres_value_text(v));
}

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
    /* Ceres refuses a negative tolerance before it evaluates anything
     * (Solver::Options::IsValid); 0 switches the test off in effect. */
    if (sp.stop.ftol_rel && !(*sp.stop.ftol_rel >= 0.0))
        throw std::runtime_error("CERES:" + algo + " requires FTOL_REL >= 0 "
            "(Ceres' function_tolerance), got " +
            ceres_value_text(*sp.stop.ftol_rel));
    if (sp.stop.xtol_rel && !(*sp.stop.xtol_rel >= 0.0))
        throw std::runtime_error("CERES:" + algo + " requires XTOL_REL >= 0 "
            "(Ceres' parameter_tolerance), got " +
            ceres_value_text(*sp.stop.xtol_rel));

    if (sp.stop.maxeval && *sp.stop.maxeval == 0)
        throw std::runtime_error("CERES:" + algo + " requires MAXEVAL > 0 when provided");

    /* PARAM keys of CERES:STEEPEST_DESCENT:
     *   fd_step           -- finite-difference step (fd_stencil.h)
     *   diff_method       -- 0 CENTRAL (default), 1 FORWARD (2, RIDDERS,
     *                        is refused)
     *   (fd_kernel        -- the step rule, a word: param_raw)
     *   tol_gradinf       -- gradient_tolerance (max-norm of the gradient)
     *   max_iter          -- max_num_iterations cap (defaults to MAXEVAL or INT_MAX)
     *   max_ls_iter       -- max line search step size iterations
     * (max_lbfgs_rank and nlcg_type, the PARAMs of the refused CERES:LBFGS
     * and CERES:NLCG, are refused as any PARAM the solver does not read.)
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
                == common_params.end()) {
            throw std::runtime_error(
                "CERES:" + algo + " does not support PARAM " + kv.first +
                "; supported: fd_step, fd_kernel, diff_method, tol_gradinf, "
                "max_iter, max_ls_iter");
        }
    }
    if (auto it = sp.param.find("tol_gradinf"); it != sp.param.end())
        if (!(it->second > 0.0))
            throw std::runtime_error("CERES:" + algo + " requires PARAM tol_gradinf > 0");
    /* Ceres' iteration caps, as Ceres checks them: max_num_iterations may
     * be 0 (the start is evaluated and the solve ends), a line search
     * takes at least one step size. */
    validate_ceres_int_param_(sp, algo, "max_iter", 0,
                              "Ceres' max_num_iterations");
    validate_ceres_int_param_(sp, algo, "max_ls_iter", 1,
                              "Ceres' max_num_line_search_step_size_"
                              "iterations");
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
 * rethrows the original after Solve returns.
 *
 * An evaluation refused because MAXEVAL is spent -- the loss
 * (GradientProvider::exhausted, before cost() is asked) or the gradient
 * (GradientFailure::Budget) -- is a failed one too, and budget_refused()
 * says so from then on: the budget is spent for the rest of the solve. */
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

            if (gp_->exhausted()) {
                budget_refused_ = true;
                return false;
            }
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
                    } else if (k == GradientFailure::Budget) {
                        budget_refused_ = true;
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
    /* Whether an Evaluate() was refused because MAXEVAL is spent. */
    bool budget_refused() const { return budget_refused_; }

private:
    GradientProvider<Vec,Runner>* gp_;
    int dim_;
    mutable bool                  last_rejected_ = false;
    mutable bool                  budget_refused_ = false;
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

/* Map TADAH StopCriteria + PARAM keys onto Ceres' tolerance fields.
 * Returns options ready to pass to ceres::Solve. */
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

    /* Ceres checks its options before it evaluates anything and, when it
     * refuses them, ends with FAILURE "Invalid configuration. ...". The
     * validators refuse every setting Ceres would (validate_ceres_common_),
     * so options Ceres refuses here are HPO's error, not an ending of the
     * search. */
    if (std::string refused; !opts.IsValid(&refused))
        throw std::logic_error(
            "internal error: CERES:" + algo_label + ": Ceres refuses the "
            "options built from a validated spec: " + refused);

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

    /* Why the solve ended, as the run's status (run_end.h); see the file
     * comment. */
    const std::string backend = "CERES:" + algo_label;
    const std::string spent = std::to_string(evals_used.load()) +
                              " evaluation(s)";
    const auto starts_with = [&](const char* prefix) {
        return summary.message.rfind(prefix, 0) == 0;
    };
    /* "FTOL_REL = 1e-06 (default)": the setting a Ceres test applied. */
    const auto setting = [](const std::string& key, double value,
                            bool user_set) {
        std::ostringstream o;
        o << key << " = " << value << (user_set ? "" : " (default)");
        return o.str();
    };
    /* A converged or capped solve: the reason, its setting, Ceres'
     * message. */
    const auto report_reason = [&](EndReason r, const std::string& set) {
        report_end(backend, r, spent + ": " + set + "; Ceres: " +
                                   summary.message);
    };
    /* An ending the vocabulary has no reason for: Ceres' own termination
     * type, with its whole message in the stop line and the head of the
     * message (ceres_message_head) in the status, "STOPPED -- " before it
     * when the search stopped (a FAILURE). */
    const auto report_native = [&](const char* type, bool stopped) {
        const std::string msg = backend + " stopped: " + type + " after " +
                                spent + "; Ceres: " + summary.message;
        std::cout << "[HPO] " << msg << "\n";
        HPOLog::instance().info(msg);
        HPOLog::instance().set_status(
            std::string(stopped ? "STOPPED -- " : "") + backend + ": " +
            type + " (" + ceres_message_head(summary.message) + ")");
    };
    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            if (starts_with("Function tolerance reached"))
                report_reason(EndReason::FtolRel,
                              setting("FTOL_REL", opts.function_tolerance,
                                      sp.stop.ftol_rel.has_value()));
            else if (starts_with("Parameter tolerance reached"))
                report_reason(EndReason::XtolRel,
                              setting("XTOL_REL", opts.parameter_tolerance,
                                      sp.stop.xtol_rel.has_value()));
            else if (starts_with("Gradient tolerance reached"))
                report_reason(EndReason::GradTol,
                              setting("PARAM tol_gradinf",
                                      opts.gradient_tolerance,
                                      ceres_param(sp, "tol_gradinf")
                                          .has_value()));
            else
                report_native("CONVERGENCE", false);
            return;
        case ceres::NO_CONVERGENCE:
            if (summary.message == "Maximum number of iterations reached.")
                report_reason(EndReason::MaxIter,
                              setting("PARAM max_iter",
                                      opts.max_num_iterations,
                                      ceres_param(sp, "max_iter")
                                          .has_value()));
            else
                report_native("NO_CONVERGENCE", false);
            return;
        case ceres::FAILURE:
            /* A FAILURE right after an evaluation without a usable
             * gradient: that gradient ended the search. */
            if (fof->last_rejected()) {
                report_gradient_stop(
                    backend, fof->rejected(),
                    "Ceres ended right after it (" +
                        std::string(ceres::TerminationTypeToString(
                            summary.termination_type)) +
                        ": " + summary.message + ")");
                return;
            }
            /* The budget refused an evaluation: Ceres cannot go on
             * without one. */
            if (fof->budget_refused()) {
                report_end(backend, EndReason::MaxEval,
                           spent + ": MAXEVAL = " + std::to_string(maxeval) +
                               "; Ceres: " + summary.message);
                return;
            }
            /* A failure of Ceres' own (a line search that finds no valid
             * step, a loss that is not finite): the search stopped, the
             * best stands -- the class of NLopt's "STOPPED -- nlopt
             * roundoff-limited". */
            report_native("FAILURE", true);
            return;
        case ceres::USER_SUCCESS:
        case ceres::USER_FAILURE:
            /* The only callback, HostExceptionAbort, aborts once a host
             * exception was captured, which is rethrown above. */
            break;
    }
    throw std::logic_error(
        "internal error: " + backend + " ended with " +
        ceres::TerminationTypeToString(summary.termination_type) +
        " without a host exception (" + summary.message + ")");
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

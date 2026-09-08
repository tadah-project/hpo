#pragma once
#include <nlopt.hpp>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/initial_step.h>
#include <tadah/hpo/optimizer_caps.h>
#include <dlib/rand.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <sstream>

namespace tadah::hpo {

/* small helper ---------------------------------------------------- */
inline nlopt_algorithm nlopt_algo(const std::string& s)
{
    auto a = nlopt_algorithm_from_string(s.c_str());
    if (a==NLOPT_NUM_ALGORITHMS)
        throw std::invalid_argument("unknown NLopt algo "+s);
    return a;
}

/* create an nlopt::opt recursively from OptimizerSpec -------------
 * `plan` is the per-dimension initial-step plan for THIS spec (nullptr
 * = STEP unset: NLopt's own per-dim heuristic stays in force). INNER
 * specs get a plan without a start point (MLSL picks the local starts). */
inline nlopt::opt make_nlopt(const OptimizerSpec& sp, unsigned dim,
                             const StepPlan* plan = nullptr)
{
    nlopt::opt o(sp.algo.c_str(), dim);

    /* stopping criteria */
    if (sp.stop.maxeval)   o.set_maxeval(*sp.stop.maxeval);
    if (sp.stop.maxtime)   o.set_maxtime(*sp.stop.maxtime);
    if (sp.stop.ftol_rel)  o.set_ftol_rel(*sp.stop.ftol_rel);
    if (sp.stop.ftol_abs)  o.set_ftol_abs(*sp.stop.ftol_abs);
    if (sp.stop.xtol_rel)  o.set_xtol_rel(*sp.stop.xtol_rel);
    /* XTOL_ABS as a per-dimension vector (NLopt's width-relative x-stop
     * lever; the |x|-relative XTOL_REL test is vacuous next to u = 0).
     * One value per dim today -- in the unit box that is a fraction of
     * every width. */
    if (sp.stop.xtol_abs)
        o.set_xtol_abs(std::vector<double>(dim, *sp.stop.xtol_abs));
    if (sp.stop.stopval)   o.set_stopval(*sp.stop.stopval);

    /* algorithm-specific parameters */
    if (sp.vector_storage) o.set_vector_storage(*sp.vector_storage);
    if (sp.pop) o.set_population(*sp.pop);
    /* NLopt's stochastic globals (CRS, MLSL, ISRES, ESCH, ...) draw from
     * an internal RNG seeded only by `nlopt::srand`. The C library
     * `srand()` does NOT propagate. Prefer INIT_SEED (the modern,
     * init-strategy-scoped seed) over the legacy SEED keyword. */
    if (sp.init_strategy_spec.seed)
        nlopt::srand(static_cast<unsigned long>(*sp.init_strategy_spec.seed));
    else if (sp.seed)
        nlopt::srand(static_cast<unsigned long>(*sp.seed));
    /* STEP is a fraction of each dimension's internal width; the plan
     * carries the per-dim vector (see initial_step.h). Never the old
     * uniform scalar: identical dx on every dim made BOBYQA/COBYLA's
     * internal rescaling a no-op and starved narrow dims. */
    if (plan && plan->user_set) o.set_initial_step(plan->dx);

    /* generic param lines */
    for (auto& kv : sp.param) {
        o.set_param(kv.first.c_str(), kv.second);
      }

    /* inner sub-optimizer (one level), for MLSL/AUGLAG-family globals.
     * Its own STEP (if any) becomes a plan without a start point; when
     * the OUTER block also sets STEP, NLopt overwrites the local dx with
     * the outer vector at optimize time (the validator warns). */
    if (sp.inner) {
        StepPlan inner_plan;
        const StepPlan* ip = nullptr;
        if (sp.inner->step || !sp.inner->step_overrides.empty()) {
            inner_plan = make_step_plan(*sp.inner, {}, optimizer_caps(*sp.inner));
            ip = &inner_plan;
        }
        o.set_local_optimizer( make_nlopt(*sp.inner, dim, ip) );
    }

    return o;
}

/* main driver ----------------------------------------------------- */
template<class Vec,class Runner>
void run_nlopt(const OptimizerSpec& sp,
               HPO_Host<Vec,Runner>& host)
{
    const unsigned dim = static_cast<unsigned>(sp.low.size());
    const OptimizerCaps& caps = optimizer_caps(sp);

    /* starting point - either user supplied or random inside bounds */
    std::vector<double> start(dim);
    if (!sp.optim_init.empty())
    {
        auto v = sp.in( sp.optim_init.front() );
        for(size_t i=0;i<dim;++i) start[i]=v[i];
    } else {
        dlib::rand r;                     // no seed: time-based
        for(size_t i=0;i<dim;++i)
            start[i]=r.get_double_in_range(sp.low[i],sp.high[i]);
    }

    /* Per-dimension initial-step plan (fraction of width, overrides,
     * per-dim near-bound rules) -- see initial_step.h. */
    const StepPlan plan = make_step_plan(sp, start, caps);
    nlopt::opt opt = make_nlopt(sp, dim, &plan);

    /* The *_MLSL_LDS variants explore with a deterministic Sobol
     * low-discrepancy sequence -- nlopt's RNG (seeded in make_nlopt)
     * is never consulted for the sampled points (mlsl.c uses
     * nlopt_sobol_next when lds=1, nlopt_urand only when lds=0), and
     * the sequence's skip count is a fixed function of dim and
     * POPULATION. Every eval after the INIT-chain starting point is
     * therefore identical across runs, with or without SEED. Surface
     * that so neither "why does every run sample the same points?"
     * nor "why doesn't SEED change the sampling?" needs a source
     * dive; the non-LDS twin (e.g. G_MLSL) gives seed-controlled
     * random sampling. */
    if (sp.algo.find("MLSL_LDS") != std::string::npos) {
        const bool seeded = sp.init_strategy_spec.seed.has_value() ||
                            sp.seed.has_value();
        std::ostringstream oss;
        oss << sp.algo << ": exploration points come from a deterministic "
               "Sobol (LDS) sequence and are identical across runs";
        if (seeded)
            oss << "; SEED affects at most the INIT-chain starting point "
                   "(and only when the chain has a RANDOM/LHS/SOBOL entry), "
                   "not the sampled points. Use " <<
                   sp.algo.substr(0, sp.algo.size() - 4)
                << " for seed-controlled random sampling";
        oss << ".";
        if (seeded) {
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
        } else {
            HPOLog::instance().info(oss.str());
        }
    }

    opt.set_lower_bounds(sp.low);
    opt.set_upper_bounds(sp.high);

    /* GradientProvider for gradient-based NLopt algorithms (LD_*).
     * Mirrors DLIB / CERES adapters: NUMERICAL routes through Ceres
     * DynamicNumericDiffCostFunction; ANALYTICAL / HYBRID consume the
     * host's analytical_gradient member with masked FD on the
     * unregistered indices. Derivative-free algos (LN_*, GN_*) leave
     * grad empty on each call -- the wrapper skips the gradient path.
     *
     * MAXEVAL is enforced by NLopt itself (set_maxeval inside
     * make_nlopt). The eval_counter here is used only by the
     * GradientProvider's internal masked-FD bookkeeping; it is
     * separate from NLopt's nfev counter. */
    std::atomic<unsigned long> evals_used{0};
    const unsigned long maxeval =
        sp.stop.maxeval.has_value()
            ? static_cast<unsigned long>(*sp.stop.maxeval)
            : 0ul;

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

    /* Userdata for the NLopt callback. The GradientProvider must
     * outlive opt.optimize(); it does, as a stack local in this
     * function. */
    struct UData {
        const OptimizerSpec*               sp;
        HPO_Host<Vec,Runner>*              host;
        GradientProvider<Vec,Runner>*      gp;
    };
    UData udata{&sp, &host, &gp};

    /* objective adapter (handles log-scale automatically). For
     * gradient-based algorithms NLopt passes a writable `grad` vector
     * of size dim; for derivative-free algorithms `grad` is empty. We
     * always compute f via host(); when grad is non-empty we ALSO
     * compute the gradient via gp.gradient(). gp.gradient handles the
     * Numerical / Analytical / Hybrid dispatch internally. */
    auto wrapper = [](const std::vector<double>& x,
                      std::vector<double>& grad, void* userdata)->double
    {
        auto* u = static_cast<UData*>(userdata);
        Vec v;
        v.set_size(static_cast<long>(x.size()));
        for(size_t i=0;i<x.size();++i) v(static_cast<long>(i))=x[i];

        /* Compute f first so the trainer cache is populated at this x;
         * host.analytical_gradient (Phase 2A.4 wiring) consumes the
         * cached fit state. */
        const double f = (*u->host)( u->sp->out(v) );

        if (!grad.empty()) {
            Vec g;
            g.set_size(static_cast<long>(x.size()));
            if (!u->gp->gradient(v, g)) {
                /* gp.gradient returns false on bad eval (NaN clamp,
                 * non-finite W*, or MAXEVAL overflow during masked FD).
                 * Signal a bad step to NLopt's line search: zero the
                 * gradient and return +inf so the optimizer rejects
                 * this point and backs off. Matches how Phase 1's
                 * Numerical path handles non-finite host returns. */
                std::fill(grad.begin(), grad.end(), 0.0);
                return std::numeric_limits<double>::infinity();
            }
            for (size_t i=0;i<x.size();++i)
                grad[i] = g(static_cast<long>(i));
        }
        return f;
    };
    opt.set_min_objective(wrapper,&udata);

    /* SBPLX/NELDERMEAD's per-dim x-stop guard compares |xstep_j| against
     * xtol_rel*|x_j| (+ xtol_abs_j): relative to the coordinate's
     * MAGNITUDE, so a dim whose optimum sits near its lower bound (u ~ 0
     * in the unit box, ln(value) ~ 0 on a log dim) can only satisfy it
     * through XTOL_ABS. Say so once when only XTOL_REL is set. */
    if (caps.step_use == StepUse::PerDimVector && sp.stop.xtol_rel &&
        !sp.stop.xtol_abs) {
        HPOLog::instance().hint(
            sp.algo + ": XTOL_REL is relative to each coordinate's magnitude; "
            "dims whose optimum sits near their lower bound" +
            std::string(sp.space && sp.space->normalised()
                        ? " (u ~ 0 in the unit box)" : "") +
            " cannot satisfy it -- set XTOL_ABS (a fraction of the OPTIM "
            "width under OPTIM_SCALE AUTO) as well");
    }

    /* Journal the EFFECTIVE per-dimension initial step: the plan's
     * vector when STEP is set, otherwise NLopt's x-dependent default read
     * back through get_initial_step(x, dx) so the printed number is the
     * number used. Only for algorithms that read an initial step. */
    if (step_use_consumes(caps.step_use)) {
        StepPlan effective = plan;
        if (!plan.user_set) {
            const bool inner_sets = sp.inner &&
                (sp.inner->step || !sp.inner->step_overrides.empty());
            if (caps.step_use == StepUse::ForwardedToInner && inner_sets) {
                /* The INNER block's own vector is what the local optimiser
                 * runs with (no OUTER STEP to override it). */
                effective = make_step_plan(*sp.inner, {}, optimizer_caps(*sp.inner));
                effective.source = "INNER " + effective.source +
                                   " (forwarded to " + sp.inner->algo + ")";
            } else if (caps.step_use == StepUse::TrustHalfWidth) {
                /* MMA/CCSAQ never consult the derivative-free heuristic:
                 * their default is sigma = 0.5*width per dim. */
                for (unsigned i = 0; i < dim; ++i) {
                    effective.dx[i]   = 0.5 * (sp.high[i] - sp.low[i]);
                    effective.frac[i] = 0.5;
                }
                effective.source = "STEP unset -- " + sp.algo +
                                   " default trust half-width 0.5*width per dim";
            } else {
                std::vector<double> dx(dim, 0.0);
                opt.get_initial_step(start, dx);
                effective.dx = dx;
                for (unsigned i = 0; i < dim; ++i) {
                    const double w = sp.high[i] - sp.low[i];
                    effective.frac[i] = std::fabs(dx[i]) / (w > 0.0 ? w : 1.0);
                }
                if (caps.step_use == StepUse::ForwardedToInner)
                    effective.source += "; shown for the OUTER start point -- each "
                                        "MLSL/AUGLAG local start recomputes it from "
                                        "its own start";
            }
        }
        const std::string report = format_step_report(effective, sp, caps, start);
        HPOLog::instance().info(report);
        std::cout << "[HPO] Initial step (" << sp.lib << ":" << sp.algo << "): "
                  << effective.source << " -- per-dim table in hpo_run.log\n";
        for (unsigned i = 0; i < dim; ++i) {
            const std::string& nt = effective.note[i];
            if (nt.find("near-bound") != std::string::npos ||
                nt.find("inward")     != std::string::npos ||
                nt.find("on a bound") != std::string::npos ||
                nt.find("capped")     != std::string::npos) {
                std::ostringstream oss;
                oss << sp.algo << ": "
                    << (i < sp.parameter_labels.size() ? sp.parameter_labels[i]
                                                       : "dim[" + std::to_string(i) + "]")
                    << ": " << effective.note[i];
                HPOLog::instance().warning(oss.str());
            }
        }
    }

    double fmin=0;
    int result_code = 1;   // success by default
    std::string status_str;
    try {
        nlopt::result res = opt.optimize(start, fmin);
        result_code  = static_cast<int>(res);
        status_str   = nlopt_result_str(result_code);

        /* NLopt sets an internal stop_msg only on failure paths inside
         * an algorithm. A handful of algorithms catch that internal
         * NLOPT_FAILURE and translate it to a "nice-looking" return
         * code at the upper layer:
         *   - LN_SBPLX  (sbplx.c:162,183): swallows nldrmd's
         *     "starting step size led to a simplex that was too small
         *     in dimension N" as NLOPT_XTOL_REACHED. This fires when
         *     the WARM seed sits near optimum and SBPLX's per-
         *     iteration xstep scaling shrinks the step below
         *     ~1e-13*|x|, so the next simplex vertex coincides with
         *     the start under nldrmd's `close()` test. The user then
         *     sees a "successful" XTOL_REACHED that actually did no
         *     useful exploration -- the same symptom as BOBYQA's silent
         *     start-point snap, but post-hoc rather than pre-flight.
         *   - LN_NELDERMEAD: identical close-test path as SBPLX.
         *   - Several others (cobyla/mma/slsqp/isres/...) surface the
         *     same channel for "invalid scaling / population too small".
         * Surface that diagnostic so the failure is not silent. */
        const char* nlopt_msg = opt.get_errmsg();
        if (nlopt_msg && nlopt_msg[0] != '\0') {
            const int nfev = opt.get_numevals();
            std::ostringstream oss;
            oss << sp.algo << ": NLopt internal diagnostic was set after "
                << nfev << " eval(s) -- '" << nlopt_msg
                << "'. Algorithm reported " << status_str
                << " but the underlying issue was masked. For "
                   "LN_SBPLX/LN_NELDERMEAD this typically means the "
                   "initial step shrank below precision near a near-"
                   "optimum WARM seed; consider raising STEP (a fraction of "
                   "each OPTIM width), setting XTOL_ABS, loosening the "
                   "tightest bound on the seed coord, or switching to "
                   "LD_LBFGS / DLIB:BFGS / LN_BOBYQA.";
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
            status_str = "STOPPED -- masked " + status_str + ": " + nlopt_msg;
        } else {
            std::ostringstream nfo;
            nfo << "NLopt finished: " << status_str << "  fmin=" << fmin;
            std::cout << "[HPO] " << nfo.str() << "\n";
            HPOLog::instance().info(nfo.str());
        }
    } catch (const std::invalid_argument& e) {
        /* NLOPT_INVALID_ARGS: a rejected SETUP (e.g. BOBYQA 'insufficient
         * space between the bounds', 'invalid scaling', 'zero step size').
         * NLopt attaches its stop message to this exception too, so it
         * must be classified BEFORE the message-based graceful test below:
         * nothing was optimised, the run is FAILED. */
        const std::string status = std::string("FAILED -- ") + e.what();
        std::cout << "[HPO] NLopt rejected the setup: " << e.what() << "\n";
        HPOLog::instance().warning(std::string("NLopt invalid argument: ") + e.what());
        HPOLog::instance().set_status(status);
        throw;
    } catch (const std::exception& e) {
        const std::string what = e.what();
        /* NLOPT_FAILURE (-1): population converged, local mutation simplex
         * degenerated -- algorithmic self-termination, not a bug.
         * NLOPT_ROUNDOFF_LIMITED (-4): blocked by floating-point roundoff.
         * Both mean the algorithm has nothing left to explore; the best
         * result found so far is valid and already saved.
         *
         * When NLopt has set an internal stop message (nlopt_stop_msg), the
         * C++ wrapper throws runtime_error(<that message>) instead of the
         * bare "nlopt failure" -- e.g. LN_NELDERMEAD's "starting step size
         * led to simplex that was too small in dimension N". That is the
         * same algorithmic self-termination and must not be reported as a
         * FAILED run. Compare against the opt's own errmsg so host
         * exceptions (which propagate through the wrapper verbatim) are
         * never misclassified. */
        const char* nlopt_msg = opt.get_errmsg();
        const bool graceful = (what == "nlopt failure" ||
                               what == "roundoff-limited" ||
                               (nlopt_msg && nlopt_msg[0] != '\0' &&
                                what == nlopt_msg));
        if (graceful) {
            status_str = "STOPPED -- " + what +
                " (algorithm converged; consider adding ftol_rel / xtol_rel"
                " / maxeval to stop cleanly next time)";
            std::cout << "[HPO] NLopt self-terminated (" << what
                      << ") -- best result preserved.\n";
            HPOLog::instance().warning(what +
                " -- algorithm converged; best result preserved");
        } else {
            status_str = std::string("FAILED -- ") + what;
            std::cout << "[HPO] NLopt threw: " << what << "\n";
            HPOLog::instance().warning(std::string("NLopt exception: ") + what);
        }
        /* Record this run's status; the outer hpo_run() driver will call
         * finish() once after the whole policy (including MULTISTART) ends. */
        HPOLog::instance().set_status(status_str);
        if (!graceful) throw;
        return;
    }

    /* Record this NLopt run's termination reason. Under MULTISTART each
     * start calls set_status() once; the last status wins, and the outer
     * hpo_run() driver writes the journal trailer after all starts have
     * finished. Calling finish() here would close the journal after the
     * first start and silence subsequent ones. */
    HPOLog::instance().set_status(status_str);
}

}   // namespace tadah::hpo

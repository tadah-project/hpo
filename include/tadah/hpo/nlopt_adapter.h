#pragma once
#include <nlopt.hpp>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <dlib/rand.h>

#include <algorithm>
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

/* create an nlopt::opt recursively from OptimizerSpec -------------*/
inline nlopt::opt make_nlopt(const OptimizerSpec& sp, unsigned dim)
{
    nlopt::opt o(sp.algo.c_str(), dim);

    /* stopping criteria */
    if (sp.stop.maxeval)   o.set_maxeval(*sp.stop.maxeval);
    if (sp.stop.maxtime)   o.set_maxtime(*sp.stop.maxtime);
    if (sp.stop.ftol_rel)  o.set_ftol_rel(*sp.stop.ftol_rel);
    if (sp.stop.ftol_abs)  o.set_ftol_abs(*sp.stop.ftol_abs);
    if (sp.stop.xtol_rel)  o.set_xtol_rel(*sp.stop.xtol_rel);
    if (sp.stop.xtol_abs)  o.set_xtol_abs(*sp.stop.xtol_abs);
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
    if (sp.step) o.set_initial_step(*sp.step);

    /* generic param lines */
    for (auto& kv : sp.param) {
        o.set_param(kv.first.c_str(), kv.second);
      }

    /* inner sub-optimizer (one level), for MLSL/AUGLAG-family globals */
    if (sp.inner)
        o.set_local_optimizer( make_nlopt(*sp.inner,dim) );

    return o;
}

/* main driver ----------------------------------------------------- */
template<class Vec,class Runner>
void run_nlopt(const OptimizerSpec& sp,
               HPO_Host<Vec,Runner>& host)
{
    const unsigned dim = static_cast<unsigned>(sp.low.size());
    nlopt::opt opt = make_nlopt(sp, dim);

    opt.set_lower_bounds(sp.low);
    opt.set_upper_bounds(sp.high);

    /* objective adapter (handles log-scale automatically) */
    auto wrapper = [](const std::vector<double>& x,
                      std::vector<double>&, void* userdata)->double
    {
        auto* pack = static_cast<std::pair<OptimizerSpec const*, HPO_Host<Vec,Runner>*>*>(userdata);
        const auto& spec = *pack->first;
        auto* hostp      =  pack->second;
        Vec v(x.size());
        for(size_t i=0;i<x.size();++i) v(i)=x[i];
        return (*hostp)( spec.out(v) );
    };
    std::pair<const OptimizerSpec*, HPO_Host<Vec,Runner>*> udata{&sp,&host};
    opt.set_min_objective(wrapper,&udata);

    /* starting point – either user supplied or random inside bounds */
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

    /* BOBYQA / NEWUOA_BOUND silently relocate the start point onto an
     * interior offset whenever any coordinate sits within rhobeg of a
     * bound (see nlopt/src/algs/bobyqa/bobyqa.c around line 3219). With
     * a uniform user-supplied STEP this destroys WARM/PRIOR seeds whose
     * previous best lies near the edge of the box. Detect that case and
     * either auto-shrink the initial step so the seed is preserved, or
     * (when the seed is on/outside a bound) warn that the snap is
     * unavoidable. NLopt's per-dim default heuristic already protects
     * against this when the user did NOT set STEP, so we only intervene
     * when sp.step is set. */
    const bool dfo_snaps =
        (sp.algo == "LN_BOBYQA" || sp.algo == "LN_NEWUOA_BOUND");
    if (dfo_snaps && !sp.optim_init.empty() && sp.step) {
        double min_slack = std::numeric_limits<double>::infinity();
        std::vector<unsigned> tight_dims;
        for (unsigned i = 0; i < dim; ++i) {
            const double slack = std::min(start[i] - sp.low[i],
                                          sp.high[i] - start[i]);
            if (slack < min_slack) min_slack = slack;
            if (slack <= *sp.step) tight_dims.push_back(i);
        }
        if (!tight_dims.empty()) {
            std::ostringstream oss;
            oss << sp.algo << ": initial point sits within STEP="
                << *sp.step << " of a bound on dim(s)";
            for (auto d : tight_dims) oss << ' ' << d;
            if (min_slack > 0) {
                const double safe_step = 0.5 * min_slack;
                oss << "; auto-shrinking initial step to " << safe_step
                    << " (was " << *sp.step
                    << ") so the warm/prior start is preserved.";
                opt.set_initial_step(safe_step);
            } else {
                oss << "; seed is on or outside a bound and BOBYQA will "
                       "snap it to the interior. Widen the bounds, or use "
                       "DLIB:BFGS / NLOPT:LN_SBPLX which honour edge starts.";
            }
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
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
         *     the WARM/PRIOR seed sits near optimum and SBPLX's per-
         *     iteration xstep scaling shrinks the step below
         *     ~1e-13*|x|, so the next simplex vertex coincides with
         *     the start under nldrmd's `close()` test. The user then
         *     sees a "successful" XTOL_REACHED that actually did no
         *     useful exploration — the same symptom as BOBYQA's silent
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
                << nfev << " eval(s) — '" << nlopt_msg
                << "'. Algorithm reported " << status_str
                << " but the underlying issue was masked. For "
                   "LN_SBPLX/LN_NELDERMEAD this typically means the "
                   "initial step shrank below precision near a near-"
                   "optimum WARM/PRIOR seed; consider raising STEP, "
                   "loosening the tightest bound on the seed coord, or "
                   "switching to LD_LBFGS / DLIB:BFGS / LN_BOBYQA.";
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
            status_str = "STOPPED — masked " + status_str + ": " + nlopt_msg;
        } else {
            std::ostringstream nfo;
            nfo << "NLopt finished: " << status_str << "  fmin=" << fmin;
            std::cout << "[HPO] " << nfo.str() << "\n";
            HPOLog::instance().info(nfo.str());
        }
    } catch (const std::exception& e) {
        const std::string what = e.what();
        /* NLOPT_FAILURE (-1): population converged, local mutation simplex
         * degenerated — algorithmic self-termination, not a bug.
         * NLOPT_ROUNDOFF_LIMITED (-4): blocked by floating-point roundoff.
         * Both mean the algorithm has nothing left to explore; the best
         * result found so far is valid and already saved. */
        const bool graceful = (what == "nlopt failure" ||
                               what == "roundoff-limited");
        if (graceful) {
            status_str = "STOPPED — " + what +
                " (algorithm converged; consider adding ftol_rel / xtol_rel"
                " / maxeval to stop cleanly next time)";
            std::cout << "[HPO] NLopt self-terminated (" << what
                      << ") — best result preserved.\n";
            HPOLog::instance().warning(what +
                " — algorithm converged; best result preserved");
        } else {
            status_str = std::string("FAILED — ") + what;
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

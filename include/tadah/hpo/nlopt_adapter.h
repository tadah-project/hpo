#pragma once
#include <nlopt.hpp>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <dlib/rand.h>

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
    if (sp.seed) srand(*sp.seed);
    if (sp.step) o.set_initial_step(*sp.step);

    /* generic param lines */
    for (auto& kv : sp.param) {
        o.set_param(kv.first.c_str(), kv.second);
      }

    /* local optimizer (one level) */
    if (sp.local)
        o.set_local_optimizer( make_nlopt(*sp.local,dim) );

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

    double fmin=0;
    int result_code = 1;   // success by default
    std::string status_str;
    try {
        nlopt::result res = opt.optimize(start, fmin);
        result_code  = static_cast<int>(res);
        status_str   = nlopt_result_str(result_code);
        std::cout << "[HPO] NLopt finished: " << status_str
                  << "  fmin=" << fmin << "\n";
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
        /* Flush remaining results before reporting finish */
        host.flush_remaining();
        HPOLog::instance().finish(status_str);
        if (!graceful) throw;
    }

    /* Flush any results that didn't reach QFLUSH_THRESHOLD (e.g. short runs). */
    host.flush_remaining();

    /* Write diagnostics + performance + result into the run journal.
     * Best metrics are tracked internally by HPOLog::log_new_best(). */
    HPOLog::instance().finish(status_str);
}

}   // namespace tadah::hpo

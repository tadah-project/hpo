#pragma once
#include <nlopt.hpp>
#include <tadah/hpo/optimiser_spec.h>
#include <tadah/hpo/hpo_host.h>
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

/* create an nlopt::opt recursively from OptimiserSpec -------------*/
inline nlopt::opt make_nlopt(const OptimiserSpec& sp, unsigned dim)
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

    /* generic PARAM lines */
    for (auto& kv : sp.param)          // skip POPULATION; handled next
        if (kv.first!="POPULATION")
            o.set_param(kv.first.c_str(), kv.second);

    if (auto it=sp.param.find("POPULATION"); it!=sp.param.end())
        o.set_population(static_cast<unsigned>(it->second));

    /* local optimiser (one level) */
    if (sp.local)
        o.set_local_optimizer( make_nlopt(*sp.local,dim) );

    return o;
}

/* main driver ----------------------------------------------------- */
template<class Vec,class Runner>
void run_nlopt(const OptimiserSpec& sp,
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
        auto* pack = static_cast<std::pair<OptimiserSpec const*, HPO_Host<Vec,Runner>*>*>(userdata);
        const auto& spec = *pack->first;
        auto* hostp      =  pack->second;
        Vec v(x.size());
        for(size_t i=0;i<x.size();++i) v(i)=x[i];
        return (*hostp)( spec.out(v) );
    };
    std::pair<const OptimiserSpec*, HPO_Host<Vec,Runner>*> udata{&sp,&host};
    opt.set_min_objective(wrapper,&udata);

    /* starting point – either user supplied or random inside bounds */
    std::vector<double> start(dim);
    if (!host.targets().optim_init.empty())
    {
        auto v = sp.in( host.targets().optim_init.front() );
        for(size_t i=0;i<dim;++i) start[i]=v(i);
    } else {
        dlib::rand r;                     // no seed: time-based
        for(size_t i=0;i<dim;++i)
            start[i]=r.get_double_in_range(sp.low[i],sp.high[i]);
    }

    double fmin=0;
    nlopt::result res = opt.optimize(start,fmin);
    std::cout<<"[HPO] NLopt finished with code "<<res
             <<"  fmin="<<fmin<<"\n";
}

}   // namespace tadah::hpo

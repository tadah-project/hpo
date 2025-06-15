#pragma once
#include <tadah/hpo/hpo_host.h>
#include <random>

namespace tadah::hpo {

/* ----- RANDOM ---------------------------------------------------- */
template<typename Vec, typename Runner>
void run_random(const OptimiserSpec& sp,
                HPO_Host<Vec,Runner>& host)
{
    std::mt19937_64 rng( sp.param.count("SEED")?
                         static_cast<uint64_t>(sp.param.at("SEED"))
                       : std::random_device{}() );
    const auto& lo = sp.low, &hi = sp.high;

    uint64_t max = sp.stop.maxeval.value_or(10000);
    for(uint64_t c=0;c<max;++c)
    {
        Vec x(lo.size());
        for(size_t i=0;i<lo.size();++i)
            x(i)=std::uniform_real_distribution<>(lo[i],hi[i])(rng);
        host( sp.out(x) );           // evaluate in original scale
    }
}

/* ----- GRID  (naïve Cartesian product) --------------------------- */
template<typename Vec, typename Runner>
void run_grid(const OptimiserSpec& sp,
              HPO_Host<Vec,Runner>& host)
{
    const unsigned steps =
        static_cast<unsigned>( sp.param.count("SPACING") ?
                               sp.param.at("SPACING") : 20.0 );
    const size_t dim = sp.low.size();
    std::vector<unsigned> idx(dim,0);

    auto total=[&](){
        double t=1; for(size_t d=0;d<dim;++d) t*=steps; return static_cast<uint64_t>(t); }();
    uint64_t done=0;
    Vec x(dim);
    while (true)
    {
        for(size_t d=0;d<dim;++d)
        {
            double t= static_cast<double>(idx[d])/(steps-1);
            x(d) = sp.low[d]*(1-t) + sp.high[d]*t;
        }
        host( sp.out(x) );
        if (++done==total) break;

        /* increment mixed-radix counter */
        for(size_t d=0;d<dim;++d)
            if(++idx[d] < steps) break; else idx[d]=0;
    }
}
}   // namespace

#pragma once
#include <tadah/hpo/hpo_host.h>
#include <random>
#include <sstream>
#include <unordered_set>

namespace tadah::hpo {

/* ----- RANDOM ---------------------------------------------------- */
template<typename Vec, typename Runner>
  void run_random(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    uint64_t seed = sp.seed.value_or(std::random_device{}());  // default seed
    {
      std::ostringstream oss;
      oss << "RANDOM search: seed=" << seed;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    std::mt19937_64 rng(seed);
    const auto& lo = sp.low, &hi = sp.high;

    constexpr uint64_t HUGE_UNSIGNED = std::numeric_limits<uint64_t>::max();
    uint64_t max = sp.stop.maxeval.value_or(HUGE_UNSIGNED);

    /* evaluate the user's config point first so the prior is always part
     * of the explored set. */
    if (!sp.optim_init.empty() && max > 0)
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      host(sp.out(x_init));
      if (max != HUGE_UNSIGNED) --max;
    }

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
  void run_grid(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {

    unsigned steps = sp.step.value_or(20);  // default steps in each dimension

    const size_t dim = sp.low.size();
    std::vector<unsigned> idx(dim,0);
    {
      std::ostringstream oss;
      oss << "GRID search: " << steps << " steps in each dimension, "
          << dim << " dimensions.";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    auto total=[&](){
      double t=1; for(size_t d=0;d<dim;++d) t*=steps; return static_cast<uint64_t>(t); }();

    /* evaluate the user's config point first so the prior is always part
     * of the explored set (in addition to the regular grid sweep). */
    if (!sp.optim_init.empty())
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      host(sp.out(x_init));
    }

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

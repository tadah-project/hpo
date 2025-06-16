#pragma once
#include <dlib/global_optimization.h>
#include <tadah/hpo/optimiser_spec.h>
#include <tadah/hpo/hpo_host.h>

namespace tadah::hpo {

template<class Vec,class Runner>
  void run_dlib_global(const OptimiserSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    using namespace dlib;

    /* ------- transform bounds into (possibly) log space ---------- */
    const matrix<double,0,1> lo = mat(sp.low);
    const matrix<double,0,1> hi = mat(sp.high);

    /* ------- optional user initial guesses ----------------------- */
    std::vector<function_evaluation> init;
    // for (const auto& v : sp.optim_init)
    //     init.emplace_back( mat(sp.in(v)), 0.0);          // y filled by dlib later

    /* ------- thread pool ---------------------------------------- */
    // const unsigned nthreads = sp.num_threads(dlib::default_thread_pool().num_threads_in_pool());
    const unsigned nthreads = sp.num_threads(1);  // signle thread for now
    dlib::thread_pool tp(nthreads);

    std::unordered_set<std::string> used;          // <- track names

    double solver_eps = 1e-7;
    if (auto it = sp.param.find("SOLVER_EPS"); it != sp.param.end()) {
      solver_eps = it->second;
      used.insert("SOLVER_EPS");
    }

    double rel_noise = 1e-12;
    if (auto it = sp.param.find("REL_NOISE"); it != sp.param.end()) {
      rel_noise = it->second;
      used.insert("REL_NOISE");
    }

    warn_unused_params(sp, used, "DLIB::GFS");

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

    find_min_global(tp, feval,
        lo, hi, sp.is_int,
        max_function_calls(sp.stop.maxeval.value_or(100000)),
        FOREVER,
        solver_eps,
        init
        );
  }

}   // namespace tadah::hpo

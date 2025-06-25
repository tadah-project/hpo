#pragma once
#include <dlib/global_optimization.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>

namespace tadah::hpo {

template<class Vec,class Runner>
  void run_dlib_global(const OptimizerSpec& sp,
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
    const unsigned nthreads = sp.num_threads.value_or(1);  // single thread as default
    dlib::thread_pool tp(nthreads);

    double solver_eps = sp.stop.ftol_abs.value_or(1e-3);
    constexpr uint64_t HUGE_UNSIGNED = std::numeric_limits<uint64_t>::max();
    uint64_t maxeval = sp.stop.maxeval.value_or(HUGE_UNSIGNED);

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

    find_min_global(tp, feval,
        lo, hi, sp.is_int,
        max_function_calls(maxeval),
        FOREVER,
        solver_eps,
        init
        );
  }

}   // namespace tadah::hpo

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
    for (const Vec& v : host.targets().optim_init)
        init.emplace_back( sp.in(mat(v)), 0.0);          // y filled by dlib later

    /* ------- thread pool ---------------------------------------- */
    int nthreads = host.targets().THREADS;
    if (nthreads<1) nthreads = default_thread_pool().num_threads_in_pool();
    thread_pool tp(nthreads);

    /* ------- solver parameters  --------------------------------- */
    const double solver_eps =
        sp.param.count("SOLVER_EPS") ? sp.param.at("SOLVER_EPS") : 1e-7;
    const double rel_noise =
        sp.param.count("REL_NOISE")  ? sp.param.at("REL_NOISE")  : 1e-12;

    /* ------- main call ------------------------------------------ */
    auto feval = [&](const matrix<double,0,1>& x)->double
    {   return host( sp.out(x) ); };            // host expects real scale

    find_min_global(tp, feval,
                    lo, hi, sp.is_int,
                    max_function_calls(sp.stop.maxeval.value_or(100000)),
                    FOREVER,
                    solver_eps,
                    init,
                    rel_noise );
}

}   // namespace tadah::hpo

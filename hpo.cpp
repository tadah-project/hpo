#include "hyper_param_opti.h"
#include "hpo_targets.h"
#include "runners.h"
#include "hpo.h"
#include "glf.h"
#include "../CORE/config/config.h"
#include <dlib/global_optimization/find_max_global.h>

using R = LammpsRunner;
using D = GLF;
using T = dlib::matrix<double,0,1>;
int hpo_run(Config &config, std::string &trg_file) {

    HPOTargets trg(trg_file);
    HyperParamOpti<T, D, R> hpo(config,trg);

    const dlib::matrix<double,0,1> lowbound = dlib::mat(trg.configmin);
    const dlib::matrix<double,0,1> highbound = dlib::mat(trg.configmax);

    auto result1 = dlib::find_min_global(hpo,lowbound,highbound,
            dlib::max_function_calls(trg.maxcalls),
            dlib::FOREVER,
            trg.eps
            );
    std::cout << result1.x << std::endl;
    std::cout << result1.y << std::endl;
    return 0;
}


// TODO Attempt on serialisation of the global optimisator...
// It does not work!
int hpo_run2(Config &config, std::string &trg_file) {

    HPOTargets trg(trg_file);
    HyperParamOpti<T, D, R> hpo(config,trg);


    bool cont_run=false;
    std::string backup_file;
    backup_file="hpo.tadah_obj";
    size_t backup_interval;


    const dlib::matrix<double,0,1> lowbound = dlib::mat(trg.configmin);
    const dlib::matrix<double,0,1> highbound = dlib::mat(trg.configmax);

    dlib::function_spec spec_glf(lowbound,highbound);
    dlib::global_function_search optimiser(spec_glf);

    if (cont_run) {

        double eps,b,c;
        size_t d,e;
        dlib::deserialize(backup_file)
            >> backup_interval
            >> eps
            >> b
            >> c
            >> d >> e;
        optimiser.set_solver_epsilon(eps);
        optimiser.set_relative_noise_magnitude(b);
        optimiser.set_pure_random_search_probability(c);
        optimiser.set_monte_carlo_upper_bound_sample_num(d);
    }
    else {
        backup_interval=10;
        optimiser.set_solver_epsilon(trg.eps);
    }

    for (size_t i = 0; i < trg.maxcalls; ++i)
    {
        dlib::function_evaluation_request next = optimiser.get_next_x();
        auto x = next.x();
        next.set(-hpo(x));  // minimise, not maximise

        // backup TODO
        if (i%backup_interval==0) {
            dlib::serialize(backup_file)
                << backup_interval
                << optimiser.get_solver_epsilon()
                << optimiser.get_relative_noise_magnitude()
                << optimiser.get_pure_random_search_probability()
                << optimiser.get_monte_carlo_upper_bound_sample_num();

        }
    }

    dlib::matrix<double,0,1> x;
    double y;
    size_t function_idx;
    optimiser.get_best_function_eval(x,y,function_idx);

    std::cout << "function_idx: "<< function_idx << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "x: " << x << std::endl;

    return 0;
}

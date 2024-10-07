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

#ifdef TADAH_BUILD_MPI
    // shut all workers
    int cont=0;
    MPI_Bcast(&cont, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    std::cout << result1.x << std::endl;
    std::cout << result1.y << std::endl;
    return 0;
}

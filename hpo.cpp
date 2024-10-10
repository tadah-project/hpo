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

#ifdef TADAH_ENABLE_MPI
int hpo_run(Config &config, std::string &trg_file,
    std::string &val_file, MPI_Comm &lmp_comm) {

  HPOTargets trg(trg_file);
  HyperParamOpti<T, D, R> hpo(config,trg, val_file, lmp_comm);

  const dlib::matrix<double,0,1> lowbound = dlib::mat(trg.configmin);
  const dlib::matrix<double,0,1> highbound = dlib::mat(trg.configmax);

  auto result1 = dlib::find_min_global(hpo,lowbound,highbound,
      dlib::max_function_calls(trg.maxcalls),
      dlib::FOREVER,
      trg.eps
      );
  return 0;
}
#else
int hpo_run(Config &config, std::string &trg_file,
    std::string &val_file) {

  HPOTargets trg(trg_file);
  HyperParamOpti<T, D, R> hpo(config,trg, val_file);

  const dlib::matrix<double,0,1> lowbound = dlib::mat(trg.configmin);
  const dlib::matrix<double,0,1> highbound = dlib::mat(trg.configmax);

  auto result1 = dlib::find_min_global(hpo,lowbound,highbound,
      dlib::max_function_calls(trg.maxcalls),
      dlib::FOREVER,
      trg.eps
      );
  return 0;
}
#endif

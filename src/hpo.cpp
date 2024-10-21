#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/hpo.h>
#include <tadah/core/config.h>
#include <dlib/global_optimization/find_max_global.h>

using R = LammpsRunner;
//using D = GLF;
using T = dlib::matrix<double,0,1>;

int hpo_run(Config &config, std::string &trg_file,
    std::string &val_file) {

  HPOTargets trg(config, trg_file);
  HPO_Host<T, R> hpo(config,trg, val_file);

  const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
  const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);

  // TODO add option for initial_function_evals 
  // TODO add option for is_integer_variable

  auto result1 = dlib::find_min_global(hpo,low,high,
      dlib::max_function_calls(trg.MAX_CALLS),
      dlib::FOREVER,
      trg.EPS
      );
  return 0;
}

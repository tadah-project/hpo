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

  std::vector<dlib::function_evaluation> init_values(1);
  const dlib::matrix<double,0,1> init = dlib::mat(trg.optim_init);
  init_values[0]=dlib::function_evaluation(init,0);


  auto result1 = dlib::find_min_global(hpo,low,high,
      dlib::max_function_calls(trg.MAX_CALLS),
      dlib::FOREVER,
      trg.EPS, init_values
      );
  return 0;
}

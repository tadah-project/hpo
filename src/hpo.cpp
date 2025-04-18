#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/hpo.h>
#include <tadah/core/context.h>
#include <dlib/global_optimization/find_max_global.h>

#include <random>

namespace tadah {
namespace hpo {

using R = LammpsRunner;
using T = dlib::matrix<double,0,1>;

void random_search(core::Context &context, HPOTargets &trg, std::string &val_file) {
  //auto &low = trg.optim_low;
  //auto &high = trg.optim_high;
  const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
  const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
  std::random_device rd;
  std::mt19937 gen(rd());

  HPO_Host<T, R> hpo(context,trg, val_file);

  for (size_t i=0; i<trg.MAX_CALLS; ++i) {
    T hp(low.size());
    for (long int j=0; j<hp.size(); ++j) {
      std::uniform_real_distribution<> dis(low(j), high(j));
      hp(j) = dis(gen);
    }
    hpo(hp);
  }
}

void grid_search(core::Context &context, HPOTargets &trg, std::string &val_file) {
}

void find_min_global(core::Context &context, HPOTargets &trg, std::string &val_file) {
  HPO_Host<T, R> hpo(context,trg, val_file);
  const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
  const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);

  // std::vector<dlib::function_evaluation> init_values(1);
  // const dlib::matrix<double,0,1> init = dlib::mat(trg.optim_init);
  // init_values[0]=dlib::function_evaluation(init,0);

  auto result1 = dlib::find_min_global(hpo,low,high,
      dlib::max_function_calls(trg.MAX_CALLS),
      dlib::FOREVER,
      trg.EPS//, init_values
      );
}

int hpo_run(core::Context &context) {
  std::string trg_file = context.get<std::string>("HPOTARGET");
  std::string val_file = context.get<std::string>("VALIDATION");

  HPOTargets trg(context, trg_file);

  if (trg.ALGO == "0") {
    random_search(context, trg, val_file);
  }
  else if(trg.ALGO == "1") {
    grid_search(context, trg, val_file);
  }
  else if(trg.ALGO == "2") {
    find_min_global(context, trg, val_file);
  }
  else {
    throw std::runtime_error("Wrong HPO algorithm.");
  }

  return 0;
}

} // namespace hpo
} // namespace tadah

#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/hpo.h>
#include <tadah/core/context.h>
#include <dlib/global_optimization/find_max_global.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <tadah/hpo/gfs.h>

#include <random>

namespace tadah {
namespace experimental {


        // The point of normalize() is to handle some of the overloaded argument types in
        // find_max_global() instances below and turn them into the argument types expected by
        // find_max_global() above.
        template <typename T>
        const T& tadah_normalize(const T& item) 
        {
            return item;
        }

        inline std::vector<std::vector<function_evaluation>> tadah_normalize(
            const std::vector<function_evaluation>& initial_function_evals
        )
        {
            return {initial_function_evals};
        }

  using namespace dlib;
  template <
    typename funct
    >
    std::pair<size_t,function_evaluation> tadah_find_max_global (
        double ymult,
        thread_pool& tp,
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        const std::vector<double>& params = {0}, // solver_epsilon
        std::vector<std::vector<function_evaluation>> initial_function_evals = {},
        stop_condition should_stop = never_stop_early
        ) 
    {
      // Decide which parameters should be searched on a log scale.  Basically, it's
      // common for machine learning models to have parameters that should be searched on
      // a log scale (e.g. SVM C).  These parameters are usually identifiable because
      // they have bounds like [1e-5 1e10], that is, they span a very large range of
      // magnitudes from really small to really big.  So there we are going to check for
      // that and if we find parameters with that kind of bound constraints we will
      // transform them to a log scale automatically.
      double yshift = 0;
      std::vector<std::vector<bool>> log_scale(specs.size());
      for (size_t i = 0; i < specs.size(); ++i)
      {
        for (long j = 0; j < specs[i].lower.size(); ++j)
        {
          if (!specs[i].is_integer_variable[j] && specs[i].lower(j) > 0 && specs[i].upper(j)/specs[i].lower(j) >= 1000)
          {
            log_scale[i].push_back(true);
            specs[i].lower(j) = std::log(specs[i].lower(j));
            specs[i].upper(j) = std::log(specs[i].upper(j));
          }
          else
          {
            log_scale[i].push_back(false);
          }
        }
      }

      if (initial_function_evals.empty()) 
      {
        initial_function_evals.resize(specs.size());
      }
      for (auto& evals : initial_function_evals) {
        for (auto& eval : evals) {
          eval.y *= ymult;
          eval.y += yshift; // shift the y values so that they are all positive
        }
      }

      gfs opt(specs, {initial_function_evals});
      opt.init_hypercube_separators(20);
      double solver_epsilon = params[0];
      double explore_epsilon = params[1];
      opt.set_solver_epsilon(solver_epsilon);
      opt.set_explore_epsilon(explore_epsilon);
      // opt.set_seed(12341234123);
      // opt.set_relative_noise_magnitude( 1e-16);  // TODO guess
      // opt.set_monte_carlo_upper_bound_sample_num(50000); // TODO guess
      opt.set_pure_random_search_probability(1.0);

      running_stats_decayed<double> objective_funct_eval_time(functions.size()*5);
      std::mutex eval_time_mutex;
      using namespace std::chrono;

      const auto time_to_stop = steady_clock::now() + max_runtime;
      //atomic<bool> doesn't support .fetch_or, use std::atomic<int> instead
      std::atomic<int> this_should_stop{false};

      double max_solver_overhead_time = 0;

      // Now run the main solver loop.
      std::cout << "Starting global optimization with " << num.max_calls << " calls." << std::endl;
      for (size_t i = 0; i < num.max_calls && steady_clock::now() < time_to_stop && !this_should_stop.load(); ++i)
      {

        const auto get_next_x_start_time = steady_clock::now();
        auto next = std::make_shared<fer>(opt.get_next_x());
        const auto get_next_x_runtime = steady_clock::now() - get_next_x_start_time;

        auto execute_call = [&opt, &specs, &functions,&ymult,&yshift,&log_scale,&eval_time_mutex,&objective_funct_eval_time,next,&should_stop,&this_should_stop]() {
          matrix<double,0,1> x = next->x();
          // Undo any log-scaling that was applied to the variables before we pass them
          // to the functions being optimized.
          for (long j = 0; j < x.size(); ++j)
          {
            if (log_scale[next->function_idx()][j])
              x(j) = std::exp(x(j));
          }
          const auto funct_eval_start = steady_clock::now();

          double y = call_function_and_expand_args(functions[next->function_idx()], x);
          const double funct_eval_runtime = duration_cast<nanoseconds>(steady_clock::now() - funct_eval_start).count();
          this_should_stop.fetch_or(should_stop(y));
          next->set(y*ymult + yshift); // shift the y values back to their original scale
          std::lock_guard<std::mutex> lock(eval_time_mutex);
          objective_funct_eval_time.add(funct_eval_runtime);
        };

        tp.add_task_by_value(execute_call);

        // std::lock_guard<std::mutex> lock(eval_time_mutex);
        // const double obj_funct_time = objective_funct_eval_time.mean()/std::max(1ul,tp.num_threads_in_pool());
        // const double solver_overhead_time = duration_cast<nanoseconds>(get_next_x_runtime).count();
        // max_solver_overhead_time = std::max(max_solver_overhead_time, solver_overhead_time);
        // if (objective_funct_eval_time.current_n() > functions.size()*5) 
        // {
        //   // If calling opt.get_next_x() is taking a long time relative to how long it takes
        //   // to evaluate the objective function then we should spend less time grinding on the
        //   // internal details of the optimizer and more time running the actual objective
        //   // function.  E.g. if we could just run 2x more objective function calls in the same
        //   // amount of time then we should just do that.  The main slowness in the solver is
        //   // from the Monte Carlo sampling, which we can turn down if the objective function
        //   // is really fast to evaluate.  This is because the point of the Monte Carlo part is
        //   // to try really hard to avoid calls to really expensive objective functions.  But
        //   // if the objective function is not expensive then we should just call it.
        //   if (obj_funct_time < solver_overhead_time) 
        //   {
        //     // Reduce the amount of Monte Carlo sampling we do.  If it goes low enough
        //     // we will disable it altogether.
        //     const size_t new_val = static_cast<size_t>(std::floor(opt.get_monte_carlo_upper_bound_sample_num()*0.8));
        //     opt.set_monte_carlo_upper_bound_sample_num(std::max<size_t>(1, new_val));
        //     // At this point just disable the upper bounding Monte Carlo search stuff and
        //     // use only pure random search since the objective function is super cheap to
        //     // evaluate, making this more fancy search a waste of time.
        //     if (opt.get_monte_carlo_upper_bound_sample_num() == 1) 
        //     {
        //       opt.set_pure_random_search_probability(1);
        //     }
        //   } else if (obj_funct_time > 1.5*max_solver_overhead_time) // Consider reenabling
        //   {
        //     // The Monte Carlo overhead grows over time as the solver accumulates more
        //     // information about the objective function.  So we only want to reenable it
        //     // or make it bigger if the objective function really is more expensive.  So
        //     // we compare to the max solver runtime we have seen so far. If the
        //     // objective function has suddenly gotten more expensive then we start to
        //     // turn the Monte Carlo modeling back on.
        //     const size_t new_val = static_cast<size_t>(std::ceil(opt.get_monte_carlo_upper_bound_sample_num()*1.28));
        //     opt.set_monte_carlo_upper_bound_sample_num(std::min<size_t>(5000, new_val));
        //     // Set this back to its default value.
        //     opt.set_pure_random_search_probability(0.02);
        //   }
        // }
      }
      tp.wait_for_all_tasks();


      matrix<double,0,1> x;
      double y;
      size_t function_idx;
      opt.get_best_function_eval(x,y,function_idx);
      // Undo any log-scaling that was applied to the variables before we output them. 
      for (long j = 0; j < x.size(); ++j)
      {
        if (log_scale[function_idx][j])
          x(j) = std::exp(x(j));
      }
      return std::make_pair(function_idx, function_evaluation(x,y));
    }

    template <
        typename funct,
        typename ...Args
        >
    std::pair<size_t,function_evaluation> tadah_find_min_global (
        int algo,
        thread_pool& tp,
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        Args&& ...args
    ) 
    {
      if (algo == 0) {
        return tadah_find_max_global(-1, tp, functions, std::move(specs), std::forward<Args>(args)...);
      } else {
        throw std::runtime_error("Unknown algorithm for tadah_find_min_global");
      }
    }

    template <
        typename funct,
        typename ...Args
        >
    function_evaluation tadah_find_min_global (
        int algo,
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        Args&& ...args
    ) 
    {
      std::vector<funct> functions(1,std::move(f));
      std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
      return tadah_find_min_global(algo, tp, functions, std::move(specs), tadah_normalize(args)...).second;
    }
} // namespace experimental
} // namespace dlib


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

void grid_search(core::Context &, HPOTargets &, std::string &) {
}

void tadah_find_min_global(core::Context &context, HPOTargets &trg, std::string &val_file, int algo) {
  HPO_Host<T, R> hpo(context,trg, val_file);
  const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
  const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);

  std::vector<dlib::function_evaluation> init_values(trg.optim_init.size());
  for (size_t i = 0; i < trg.optim_init.size(); ++i) {
    init_values[i]=dlib::function_evaluation(dlib::mat(trg.optim_init[i]),i);
  }

  dlib::thread_pool tp(1);  // Use a single thread for now as HPO_Host::operator() is not thread-safe.
  // dlib::thread_pool &tp = dlib::default_thread_pool();

  //auto result1 = dlib::find_min_global(
  auto result1 = tadah::experimental::tadah_find_min_global(
      algo, // 99 dlib, 0 for max
      tp,
      hpo,
      low,
      high,
      trg.optim_is_int,
      dlib::max_function_calls(trg.MAX_CALLS),
      dlib::FOREVER,
      trg.PARAMS,
      init_values
      );
}

void find_min_global(core::Context &context, HPOTargets &trg, std::string &val_file) {
  HPO_Host<T, R> hpo(context,trg, val_file);
  const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
  const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);

  std::vector<dlib::function_evaluation> init_values(trg.optim_init.size());
  for (size_t i = 0; i < trg.optim_init.size(); ++i) {
    init_values[i]=dlib::function_evaluation(dlib::mat(trg.optim_init[i]),i);
  }

  dlib::thread_pool tp(1);  // Use a single thread for now as HPO_Host::operator() is not thread-safe.
  // dlib::thread_pool &tp = dlib::default_thread_pool();

  auto result1 = dlib::find_min_global(
      tp,
      hpo,
      low,
      high,
      trg.optim_is_int,
      dlib::max_function_calls(trg.MAX_CALLS),
      dlib::FOREVER,
      trg.PARAMS[0],
      init_values
      );
}
int hpo_run(core::Context &context) {
  std::string trg_file = context.get<std::string>("HPOTARGET");
  std::string val_file = context.get<std::string>("VALIDATION");

  HPOTargets trg(context, trg_file);

  if (trg.MINIMIZER == "0") {
    random_search(context, trg, val_file);
  }
  else if(trg.MINIMIZER == "1") {
    grid_search(context, trg, val_file);
  }
  else if(trg.MINIMIZER == "2") { // dlib::find_min_global
    find_min_global(context, trg, val_file);
  }
  else if(trg.MINIMIZER == "3") { // tadah::experimental::tadah_find_min_global
    tadah_find_min_global(context, trg, val_file, 0);
  }
  else {
    throw std::runtime_error("Wrong HPO algorithm.");
  }

  return 0;
}

} // namespace hpo
} // namespace tadah

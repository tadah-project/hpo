#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo.h>
#include <tadah/core/context.h>
// #include <dlib/global_optimization/find_max_global.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <tadah/hpo/gfs.h>

#include <nlopt.hpp>

#include <random>
#include <variant>

// namespace tadah {
// namespace experimental {
//
//         // The point of normalize() is to handle some of the overloaded argument types in
//         // find_max_global() instances below and turn them into the argument types expected by
//         // find_max_global() above.
//         template <typename T>
//         const T& tadah_normalize(const T& item) 
//         {
//             return item;
//         }
//
//         inline std::vector<std::vector<function_evaluation>> tadah_normalize(
//             const std::vector<function_evaluation>& initial_function_evals
//         )
//         {
//             return {initial_function_evals};
//         }
//
//   using namespace dlib;
//   template <
//     typename funct
//     >
//     std::pair<size_t,function_evaluation> tadah_find_max_global (
//         double ymult,
//         thread_pool& tp,
//         std::vector<funct>& functions,
//         std::vector<function_spec> specs,
//         const max_function_calls num,
//         const std::chrono::nanoseconds max_runtime = FOREVER,
//         const std::vector<double>& params = {0}, // solver_epsilon
//         std::vector<std::vector<function_evaluation>> initial_function_evals = {},
//         stop_condition should_stop = never_stop_early
//         ) 
//     {
//       // Decide which parameters should be searched on a log scale.  Basically, it's
//       // common for machine learning models to have parameters that should be searched on
//       // a log scale (e.g. SVM C).  These parameters are usually identifiable because
//       // they have bounds like [1e-5 1e10], that is, they span a very large range of
//       // magnitudes from really small to really big.  So there we are going to check for
//       // that and if we find parameters with that kind of bound constraints we will
//       // transform them to a log scale automatically.
//       double yshift = 0;
//       std::vector<std::vector<bool>> log_scale(specs.size());
//       for (size_t i = 0; i < specs.size(); ++i)
//       {
//         for (long j = 0; j < specs[i].lower.size(); ++j)
//         {
//           if (!specs[i].is_integer_variable[j] && specs[i].lower(j) > 0 && specs[i].upper(j)/specs[i].lower(j) >= 1000)
//           {
//             log_scale[i].push_back(true);
//             specs[i].lower(j) = std::log(specs[i].lower(j));
//             specs[i].upper(j) = std::log(specs[i].upper(j));
//           }
//           else
//           {
//             log_scale[i].push_back(false);
//           }
//         }
//       }
//
//       if (initial_function_evals.empty()) 
//       {
//         initial_function_evals.resize(specs.size());
//       }
//       for (auto& evals : initial_function_evals) {
//         for (auto& eval : evals) {
//           eval.y *= ymult;
//           eval.y += yshift; // shift the y values so that they are all positive
//         }
//       }
//
//       gfs opt(specs, {initial_function_evals});
//       opt.init_chunks(16);
//       // opt.init_hypercube_separators(20);
//       double solver_epsilon = params[0];
//       double explore_epsilon = params[1];
//       opt.set_solver_epsilon(solver_epsilon);
//       opt.set_explore_epsilon(explore_epsilon);
//       // opt.set_seed(12341234123);
//       // opt.set_relative_noise_magnitude( 1e-16);  // TODO guess
//       // opt.set_monte_carlo_upper_bound_sample_num(50000); // TODO guess
//       opt.set_pure_random_search_probability(1.0);
//
//       running_stats_decayed<double> objective_funct_eval_time(functions.size()*5);
//       std::mutex eval_time_mutex;
//       using namespace std::chrono;
//
//       const auto time_to_stop = steady_clock::now() + max_runtime;
//       //atomic<bool> doesn't support .fetch_or, use std::atomic<int> instead
//       std::atomic<int> this_should_stop{false};
//
//       double max_solver_overhead_time = 0;
//
//       // Now run the main solver loop.
//       std::cout << "Starting optimization with " << num.max_calls << " calls." << std::endl;
//       std::cout << "Using " << tp.num_threads_in_pool() << " threads." << std::endl;
//       for (size_t i = 0; i < num.max_calls && steady_clock::now() < time_to_stop && !this_should_stop.load(); ++i)
//       {
//
//         const auto get_next_x_start_time = steady_clock::now();
//         auto next = std::make_shared<fer>(opt.get_next_x());
//         const auto get_next_x_runtime = steady_clock::now() - get_next_x_start_time;
//
//         auto execute_call = [&opt, &specs, &functions,&ymult,&yshift,&log_scale,&eval_time_mutex,&objective_funct_eval_time,next,&should_stop,&this_should_stop]() {
//           matrix<double,0,1> x = next->x();
//           // Undo any log-scaling that was applied to the variables before we pass them
//           // to the functions being optimized.
//           for (long j = 0; j < x.size(); ++j)
//           {
//             if (log_scale[next->function_idx()][j])
//               x(j) = std::exp(x(j));
//           }
//           const auto funct_eval_start = steady_clock::now();
//
//           double y = call_function_and_expand_args(functions[next->function_idx()], x);
//           const double funct_eval_runtime = duration_cast<nanoseconds>(steady_clock::now() - funct_eval_start).count();
//           // std::cout << "Dlib Thread id1: " << std::this_thread::get_id() 
//           //           << std::endl;
//           this_should_stop.fetch_or(should_stop(y));
//           next->set(y*ymult + yshift); // shift the y values back to their original scale
//           std::lock_guard<std::mutex> lock(eval_time_mutex);
//           objective_funct_eval_time.add(funct_eval_runtime);
//         };
//
//         tp.add_task_by_value(execute_call);
//
//         // std::lock_guard<std::mutex> lock(eval_time_mutex);
//         // const double obj_funct_time = objective_funct_eval_time.mean()/std::max(1ul,tp.num_threads_in_pool());
//         // const double solver_overhead_time = duration_cast<nanoseconds>(get_next_x_runtime).count();
//         // max_solver_overhead_time = std::max(max_solver_overhead_time, solver_overhead_time);
//         // if (objective_funct_eval_time.current_n() > functions.size()*5) 
//         // {
//         //   // If calling opt.get_next_x() is taking a long time relative to how long it takes
//         //   // to evaluate the objective function then we should spend less time grinding on the
//         //   // internal details of the optimizer and more time running the actual objective
//         //   // function.  E.g. if we could just run 2x more objective function calls in the same
//         //   // amount of time then we should just do that.  The main slowness in the solver is
//         //   // from the Monte Carlo sampling, which we can turn down if the objective function
//         //   // is really fast to evaluate.  This is because the point of the Monte Carlo part is
//         //   // to try really hard to avoid calls to really expensive objective functions.  But
//         //   // if the objective function is not expensive then we should just call it.
//         //   if (obj_funct_time < solver_overhead_time) 
//         //   {
//         //     // Reduce the amount of Monte Carlo sampling we do.  If it goes low enough
//         //     // we will disable it altogether.
//         //     const size_t new_val = static_cast<size_t>(std::floor(opt.get_monte_carlo_upper_bound_sample_num()*0.8));
//         //     opt.set_monte_carlo_upper_bound_sample_num(std::max<size_t>(1, new_val));
//         //     // At this point just disable the upper bounding Monte Carlo search stuff and
//         //     // use only pure random search since the objective function is super cheap to
//         //     // evaluate, making this more fancy search a waste of time.
//         //     if (opt.get_monte_carlo_upper_bound_sample_num() == 1) 
//         //     {
//         //       opt.set_pure_random_search_probability(1);
//         //     }
//         //   } else if (obj_funct_time > 1.5*max_solver_overhead_time) // Consider reenabling
//         //   {
//         //     // The Monte Carlo overhead grows over time as the solver accumulates more
//         //     // information about the objective function.  So we only want to reenable it
//         //     // or make it bigger if the objective function really is more expensive.  So
//         //     // we compare to the max solver runtime we have seen so far. If the
//         //     // objective function has suddenly gotten more expensive then we start to
//         //     // turn the Monte Carlo modeling back on.
//         //     const size_t new_val = static_cast<size_t>(std::ceil(opt.get_monte_carlo_upper_bound_sample_num()*1.28));
//         //     opt.set_monte_carlo_upper_bound_sample_num(std::min<size_t>(5000, new_val));
//         //     // Set this back to its default value.
//         //     opt.set_pure_random_search_probability(0.02);
//         //   }
//         // }
//       }
//       tp.wait_for_all_tasks();
//
//
//       matrix<double,0,1> x;
//       double y;
//       size_t function_idx;
//       opt.get_best_function_eval(x,y,function_idx);
//       // Undo any log-scaling that was applied to the variables before we output them. 
//       for (long j = 0; j < x.size(); ++j)
//       {
//         if (log_scale[function_idx][j])
//           x(j) = std::exp(x(j));
//       }
//       return std::make_pair(function_idx, function_evaluation(x,y));
//     }
//
//     template <
//         typename funct,
//         typename ...Args
//         >
//     std::pair<size_t,function_evaluation> tadah_find_min_global (
//         int algo,
//         thread_pool& tp,
//         std::vector<funct>& functions,
//         std::vector<function_spec> specs,
//         Args&& ...args
//     ) 
//     {
//       if (algo == 0) {
//         return tadah_find_max_global(-1, tp, functions, std::move(specs), std::forward<Args>(args)...);
//       } else {
//         throw std::runtime_error("Unknown algorithm for tadah_find_min_global");
//       }
//     }
//
//     template <
//         typename funct,
//         typename ...Args
//         >
//     function_evaluation tadah_find_min_global (
//         int algo,
//         thread_pool& tp,
//         funct f,
//         const matrix<double,0,1>& bound1,
//         const matrix<double,0,1>& bound2,
//         const std::vector<bool>& is_integer_variable,
//         Args&& ...args
//     ) 
//     {
//       std::vector<funct> functions(1,std::move(f));
//       std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
//       return tadah_find_min_global(algo, tp, functions, std::move(specs), tadah_normalize(args)...).second;
//     }
// } // namespace experimental
// } // namespace dlib


// namespace tadah {
// namespace hpo {

// using R = LammpsRunner;
// using T = dlib::matrix<double,0,1>;
//
// void random_search(core::Context &context, HPOTargets &trg, std::string &val_file) {
//   //auto &low = trg.optim_low;
//   //auto &high = trg.optim_high;
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//   std::random_device rd;
//   std::mt19937 gen(rd());
//
//   HPO_Host<T, R> hpo(context,trg, val_file);
//
//   for (size_t i=0; i<trg.spec.maxeval; ++i) {
//     T hp(low.size());
//     for (long int j=0; j<hp.size(); ++j) {
//       std::uniform_real_distribution<> dis(low(j), high(j));
//       hp(j) = dis(gen);
//     }
//     hpo(hp);
//   }
// }
//
// void grid_search(core::Context &, HPOTargets &, std::string &) {
// }
//
// void tadah_find_min_global(core::Context &context, HPOTargets &trg, std::string &val_file, int algo) {
//   HPO_Host<T, R> hpo(context,trg, val_file);
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//
//   std::vector<dlib::function_evaluation> init_values(trg.optim_init.size());
//   for (size_t i = 0; i < trg.optim_init.size(); ++i) {
//     init_values[i]=dlib::function_evaluation(dlib::mat(trg.optim_init[i]),i);
//   }
//
//   // dlib::thread_pool tp(2);  // Use a single thread for now as HPO_Host::operator() is not thread-safe.
//   // dlib::thread_pool &tp = dlib::default_thread_pool();
//   int threads = trg.THREADS;
//   if (threads < 1) {
//     threads = dlib::default_thread_pool().num_threads_in_pool();
//   }
//   dlib::thread_pool tp(threads);
//
//   //auto result1 = dlib::find_min_global(
//   auto result1 = tadah::experimental::tadah_find_min_global(
//       algo, // 99 dlib, 0 for max
//       tp,
//       hpo,
//       low,
//       high,
//       trg.optim_is_int,
//       dlib::max_function_calls(trg.spec.maxeval),
//       dlib::FOREVER,
//       trg.params,
//       init_values
//       );
// }

// void find_min_global(core::Context &context, HPOTargets &trg, std::string &val_file) {
//   HPO_Host<T, R> hpo(context,trg, val_file);
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//
//   std::vector<dlib::function_evaluation> init_values(trg.optim_init.size());
//   for (size_t i = 0; i < trg.optim_init.size(); ++i) {
//     init_values[i]=dlib::function_evaluation(dlib::mat(trg.optim_init[i]),i);
//   }
//
//   // dlib::thread_pool tp(1);  // Use a single thread for now as HPO_Host::operator() is not thread-safe.
//   int threads = trg.THREADS;
//   if (threads < 1) {
//     threads = dlib::default_thread_pool().num_threads_in_pool();
//   }
//   dlib::thread_pool tp(threads);
//
//   auto result1 = dlib::find_min_global(
//       tp,
//       hpo,
//       low,
//       high,
//       trg.optim_is_int,
//       dlib::max_function_calls(trg.NSTEPS),
//       dlib::FOREVER,
//       trg.params[0],
//       init_values
//       );
// }
// void find_min_bobyqa(core::Context &context, HPOTargets &trg, std::string &val_file) {
//   HPO_Host<T, R> hpo(context,trg, val_file);
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//
//   dlib::matrix<double,0,1> starting_point(low.size());
//   time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//   dlib::rand rnd(dlib::cast_to_string(start_time)); // Seed the random number generator with the current time
//   for (long int j = 0; j < starting_point.size(); ++j) {
//     starting_point(j) = rnd.get_double_in_range(low(j), high(j));
//   }
//
//   std::vector<dlib::function_evaluation> init_values(trg.optim_init.size());
//   for (size_t i = 0; i < trg.optim_init.size(); ++i) {
//     init_values[i]=dlib::function_evaluation(dlib::mat(trg.optim_init[i]),i);
//   }
//
//   long npt = 2*starting_point.size() + 1; // Number of interpolation points for the BOBYQA algorithm.
//
//   double rho_begin = trg.params[1]; // Initial trust region radius for the BOBYQA algorithm.
//   double rho_end = trg.params[0];   // Final trust region radius for the BOBYQA algorithm.
//
//   auto result1 = dlib::find_min_bobyqa(
//       hpo,
//       starting_point,
//       npt,
//       low, high,
//       rho_begin, rho_end,
//       trg.NSTEPS
//       );
// }
// void find_min_using_approximate_derivatives(core::Context &context, HPOTargets &trg, std::string &val_file) {
//   HPO_Host<T, R> hpo(context,trg, val_file);
//
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//
//   dlib::matrix<double,0,1> starting_point(low.size());
//   time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//   dlib::rand rnd(dlib::cast_to_string(start_time)); // Seed the random number generator with the current time
//   for (long int j = 0; j < starting_point.size(); ++j) {
//     starting_point(j) = rnd.get_double_in_range(low(j), high(j));
//   }
//
//   using Any = std::variant<dlib::cg_search_strategy,
//         dlib::bfgs_search_strategy,
//         dlib::lbfgs_search_strategy>;
//
//   // params[i] is used as follows:
//   // params[0] = min_f (stop when the loss function < min_f)
//   // params[1] = stop_strategy.epsilon (stop when the change in the loss function is less than this value)
//   // params[2] = derivative_eps (step size for numerical derivative approximation)
//   // params[3] = search_strategyID (1 for CG, 2 for BFGS, 3 for LBFGS)
//   // params[4] = lbfgs_max_size (optional, default 10, used only for LBFGS search strategy)
//   auto search_strategy = dlib::bfgs_search_strategy();
//   double min_f = trg.params[0]; // stop when the loss function < min_f
//   auto stop_strategy = dlib::objective_delta_stop_strategy(trg.params[1]);
//   double derivative_eps = trg.params[2]; // step size for numerical derivative approximation
//   int search_strategyID = trg.params[3];
//   long lbfgs_max_size = trg.params.size()==5? trg.params[4] : 0; // default value for lbfgs_max_size, can be changed in params
//   Any value = [search_strategyID, lbfgs_max_size]() -> Any         // immediately-invoked λ
//   {
//     switch (search_strategyID)
//     {
//       case 1: return dlib::cg_search_strategy{};
//       case 2: return dlib::bfgs_search_strategy{};
//       case 3: return dlib::lbfgs_search_strategy{lbfgs_max_size};
//       default: throw std::invalid_argument("Invalid search strategy ID. Must be 1, 2, or 3.");
//     }
//   }();
//
//   auto result = std::visit(
//       [&stop_strategy,&hpo,&starting_point,min_f,derivative_eps](const auto& search_strategy) -> double  // generic visitor ✓
//       {
//       return dlib::find_min_using_approximate_derivatives
//       (search_strategy, stop_strategy,hpo, starting_point, min_f, derivative_eps);     // correct instantiation chosen
//       },
//       value
//       );
// }
// inline nlopt_algorithm algo_from_string(const std::string& s)
// {
//     nlopt_algorithm a = nlopt_algorithm_from_string(s.c_str());
//     if (a == NLOPT_NUM_ALGORITHMS)           // not recognised
//         throw std::invalid_argument("unknown NLopt algorithm: " + s);
//     return a;
// }
// void find_nlopt(core::Context &context, HPOTargets &trg, std::string &val_file) {
//
//   // size(trg.optimizer ==2) throw otherwise
//   if (trg.optimizer.size() < 2) {
//     throw std::invalid_argument("nlopt optimizer must have at least two elements: [6, algorithm_name, [optional]");
//   }
//
//   // params[0] = ftol_rel (required)
//   // params[1] = xtol_rel (optional)
//   // params[2] = ftol_rel for local optimizer (optional)
//   // params[3] = xtol_rel (for local optimizer, optional)
//
//   // using T = std::vector<double>;
//   HPO_Host<T, R> host(context,trg, val_file);
//
//   const dlib::matrix<double,0,1> low = dlib::mat(trg.optim_low);
//   const dlib::matrix<double,0,1> high = dlib::mat(trg.optim_high);
//
//   std::vector<double> starting_point;
//
//   // if (trg.optim_init.empty())
//   // {
//     starting_point.resize(low.size());
//     time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
//     dlib::rand rnd(dlib::cast_to_string(start_time)); // Seed the random number generator with the current time
//     for (long int j = 0; j < starting_point.size(); ++j)
//     {
//       starting_point[j] = rnd.get_double_in_range(low(j), high(j));
//     }
//   // } else 
//   // {
//   //   starting_point = trg.optim_init[0]; // Use the first initial point if available
//   // }
//   nlopt::opt opt(trg.optimizer[1].c_str(), trg.optim_low.size());
//   auto gnum_params = opt.num_params();
//   std::cout << "Using NLopt algorithm: " << trg.optimizer[1] << std::endl;
//   std::cout << "Number of parameters: " << gnum_params << std::endl;
//   if (trg.optimizer.size()==3) {
//     nlopt::opt local_opt(trg.optimizer[2].c_str(), trg.optim_low.size());
//     // check params size before accessing
//     if (trg.params.size() < 3) {
//       throw std::invalid_argument("this nlopt optimizer requires at least 3 parameters: ftol_rel, xtol_rel, ftol_rel_local, [xtol_rel_local]");
//     } 
//     if (trg.params.size() > 3) {
//       local_opt.set_ftol_rel(trg.params[2]); // Set the relative tolerance for the local optimizer
//       local_opt.set_xtol_rel(trg.params[3]); // Set the relative tolerance for the local optimizer
//     } else {
//       local_opt.set_ftol_rel(trg.params[2]); // Set the relative tolerance for the local optimizer
//     }
//     opt.set_local_optimizer(local_opt); // Set local optimizer if specified
//     auto lnum_params = local_opt.num_params();
//     std::cout << "Using local optimizer: " << trg.optimizer[2] << std::endl;
//     std::cout << "Number of local parameters: " << lnum_params << std::endl;
//     std::cout << "Stopping criteria for local optimizer: "
//               << "ftol_rel = " << local_opt.get_ftol_rel() << ", xtol_rel = " << local_opt.get_xtol_rel() << std::endl;
//   }
//   opt.set_lower_bounds(trg.optim_low);
//   opt.set_upper_bounds(trg.optim_high);
//   opt.set_min_objective(&HPO_Host<T,R>::nlopt_adapter_cpp, &host);
//
//     if (trg.params.size() < 1) {
//       throw std::invalid_argument("this nlopt optimizer requires at least 1 parameter: ftol_rel, [xtol_rel]");
//     } 
//     if (trg.params.size() > 1) {
//       opt.set_ftol_rel(trg.params[0]); // Set the relative tolerance for the local optimizer
//       opt.set_xtol_rel(trg.params[1]); // Set the relative tolerance for the local optimizer
//     } else {
//       opt.set_ftol_rel(trg.params[0]); // Set the relative tolerance for the local optimizer
//     }
//     std::cout << "Stopping criteria for global optimizer: "
//               << "ftol_rel = " << opt.get_ftol_rel() << ", xtol_rel = " << opt.get_xtol_rel() << std::endl;
//
//
//   double fmin;
//   nlopt::result result = opt.optimize(starting_point, fmin);
//
//   // params[i] is used as follows:
//   // params[0] = min_f (stop when the loss function < min_f)
//   // params[1] = stop_strategy.epsilon (stop when the change in the loss function is less than this value)
//   // params[2] = derivative_eps (step size for numerical derivative approximation)
//   // params[3] = search_strategyID (1 for CG, 2 for BFGS, 3 for LBFGS)
//   // params[4] = lbfgs_max_size (optional, default 10, used only for LBFGS search strategy)
//
// }
// int hpo_run(core::Context &context) {
//   std::string trg_file = context.get<std::string>("HPOTARGET");
//   std::string val_file = context.get<std::string>("VALIDATION");
//
//   HPOTargets trg(context, trg_file);
//
//   // We are beginning the optimization process. Let's print some information about the optimization process.
//   std::cout << "Starting optimization with " << trg.NSTEPS << " steps." << std::endl;
//   std::cout << "Dimension of the optimization problem: " << trg.optim_low.size() << std::endl;
//
//   std::cout << "Using optimizer: ";
//   for (const auto& o : trg.optimizer) {
//     std::cout << o << " ";
//   }
//   std::cout << std::endl;
//
//   std::cout << "Optimizer parameters: ";
//   for (const auto& p : trg.params) {
//     std::cout << p << " ";
//   }
//   std::cout << std::endl;
//
//   if (trg.optimizer[0] == "0") {
//     random_search(context, trg, val_file);
//   }
//   else if(trg.optimizer[0] == "1") {
//     grid_search(context, trg, val_file);
//   }
//   else if(trg.optimizer[0] == "2") { // dlib::find_min_global
//     find_min_global(context, trg, val_file);
//   }
//   else if(trg.optimizer[0] == "3") { // tadah::experimental::tadah_find_min_global
//     tadah_find_min_global(context, trg, val_file, 0);
//   }
//   else if(trg.optimizer[0] == "4") { // dlib::find_min_bobyqa
//     find_min_bobyqa(context, trg, val_file);
//   }
//   else if(trg.optimizer[0] == "5") { // dlib::find_min_using_approximate_derivatives
//     find_min_using_approximate_derivatives(context, trg, val_file);
//   }
//   else if(trg.optimizer[0] == "6") { // nlopt::optimize
//     find_nlopt(context, trg, val_file);
//   }
//   else {
//     throw std::runtime_error("Wrong HPO algorithm.");
//   }
//
//   return 0;
// }

// } // namespace hpo
// } // namespace tadah

namespace tadah::hpo {
using R = LammpsRunner;
using V = dlib::matrix<double,0,1>;
int hpo_run(core::Context& ctx)
{
    HPOTargets trg(ctx, ctx.get<std::string>("HPOTARGET"));
    std::string val = ctx.get<std::string>("VALIDATION");

    HPO_Host<V,R> host(ctx,trg,val);
    const OptimiserSpec& sp = trg.spec;          // short alias

    /* quick info */
    std::cout<<"[HPO] "<<sp.lib<<":"<<sp.algo<<"   dim="<<sp.low.size()<<"\n";
    // if (ctx.exists("LIST_OPT")) {                // CLI: tadah hpo --list
    //     OptimiserSpec::print_caps(sp.lib, sp.algo);
    //     return 0;
    // }

    if (sp.lib=="TADAH" && sp.algo=="RANDOM")
        run_random<V,R>(sp,host);
    else if(sp.lib=="TADAH" && sp.algo=="GRID")
        run_grid<V,R>  (sp,host);
    else if(sp.lib=="DLIB")
        run_dlib_global(sp,host);        // helper in new dlib_adapter.h
    else if(sp.lib=="NLOPT")
        run_nlopt<V,R>(sp,host);              // helper in new nlopt_adapter.h
    else
        throw std::runtime_error("unknown optimiser "+sp.lib+":"+sp.algo);
    return 0;
}
} // namespace tadah::hpo

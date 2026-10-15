#include <tadah/core/logger.h>
#include <tadah/hpo/refit/embed_compact.h>
#include <tadah/mlip/engines/refit_keys.h>
#ifdef TADAH_USE_CERES
#include <algorithm>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {
namespace {
class Objective final : public ceres::FirstOrderFunction {
  struct Variable {
    bool density, scale, bounded;
    std::size_t atom;
  };
  const models::EAMSetflIO &eam_;
  models::EAMRefitConfig cfg_;
  std::vector<Variable> variables_;
  // What Evaluate caught (Ceres's solve must not be unwound through):
  // the first internal error, rethrown after the solve, and the last
  // trial point the model refused.
  mutable std::exception_ptr internal_;
  mutable std::string refused_;

public:
  Objective(const models::EAMSetflIO &eam, const models::EAMRefitConfig &cfg,
            std::vector<double> &x)
      : eam_(eam), cfg_(cfg) {
    const auto &s = *cfg.component;
    for (bool density : {true, false}) {
      const auto &c = density ? s.density : s.embedding;
      const bool bounded = density && s.cutoff == "NONE";
      for (std::size_t i = 0; i < c.centres.size(); ++i) {
        double u = c.centres[i];
        if (bounded)
          u = std::sqrt(std::max(0.0, s.radius -
                                          models::embed_support(c.basis) /
                                              models::embed_eta(c, i) -
                                          u));
        variables_.push_back({density, false, bounded, i});
        x.push_back(u);
        if (c.basis != models::EmbedBasis::Knot5) {
          variables_.push_back({density, true, bounded, i});
          x.push_back(std::log(c.scales[i]));
        }
      }
    }
  }
  int NumParameters() const override {
    return static_cast<int>(variables_.size());
  }
  models::EAMRefitConfig decode(const double *x) const {
    auto cfg = cfg_;
    auto &s = *cfg.component;
    for (std::size_t j = 0; j < variables_.size(); ++j) {
      const auto &v = variables_[j];
      auto &c = v.density ? s.density : s.embedding;
      if (v.scale)
        c.scales[v.atom] = std::exp(x[j]);
    }
    for (std::size_t j = 0; j < variables_.size(); ++j) {
      const auto &v = variables_[j];
      if (v.scale)
        continue;
      auto &c = v.density ? s.density : s.embedding;
      c.centres[v.atom] = v.bounded ? s.radius -
                                          models::embed_support(c.basis) /
                                              models::embed_eta(c, v.atom) -
                                          x[j] * x[j]
                                    : x[j];
    }
    return cfg;
  }
  /** The cost (and gradient) at x. Throws what the fit throws: a
   *  precondition refusal of the table (EAMRefit), a component the trial
   *  point makes invalid (std::runtime_error), an internal error
   *  (std::logic_error). */
  bool cost_at(const double *x, double *cost, double *gradient) const {
    {
      auto cfg = decode(x);
      models::EAMRefit fit(eam_, cfg);
      fit.fit();
      auto pot = fit.build_pot_context();
      auto spec = models::embed_specs(pot).at(0);
      models::EmbedEvaluator evaluator(spec);
      const auto offset = pot.size("CGRID2B");
      *cost = 0;
      if (gradient)
        std::fill_n(gradient, variables_.size(), 0);
      for (bool density : {true, false}) {
        const auto &c = density ? spec.density : spec.embedding;
        const std::size_t n =
            std::max<std::size_t>(2048, c.centres.size() * cfg.oversample);
        double lo = density ? fit.fit_window_r_lo() : 0,
               hi = density ? spec.radius : eam_.rhomax();
        double norm = 0;
        for (std::size_t i = 0; i < n; ++i) {
          double q = lo + (hi - lo) * i / (n - 1),
                 target = density ? eam_.rho_tab(q) : fit.F_ref(q);
          norm = std::max(norm, std::abs(target));
        }
        const double weight = 1 / (n * std::max(norm * norm, 1e-24));
        if (cfg.lambda > 0)
          for (std::size_t j = 0; j < c.centres.size(); ++j) {
            double coefficient = density
                                     ? spec.density.amplitudes[j]
                                     : pot.get<double>("WEIGHTS", offset + j);
            *cost += 0.5 * weight * cfg.lambda * coefficient * coefficient;
          }
        for (std::size_t i = 0; i < n; ++i) {
          double q = lo + (hi - lo) * i / (n - 1),
                 target = density ? eam_.rho_tab(q) : fit.F_ref(q);
          double residual = (density ? fit.rho_fit(q) : fit.F_fit(q)) - target;
          *cost += 0.5 * weight * residual * residual;
          if (!gradient)
            continue;
          for (std::size_t j = 0; j < variables_.size(); ++j) {
            const auto &v = variables_[j];
            if (v.density != density)
              continue;
            double dc = v.scale ? 0 : v.bounded ? -2 * x[j] : 1;
            double ds = v.scale ? c.scales[v.atom] : 0;
            if (v.scale && v.bounded) {
              double reach =
                  models::embed_support(c.basis) / models::embed_eta(c, v.atom);
              dc = c.coordinates == models::EmbedCoordinates::Eta ? reach
                                                                  : -reach;
            }
            double derivative = 0;
            if (density)
              derivative = dc * evaluator.density(q, 'c', v.atom).value +
                           ds * evaluator.density(q, 's', v.atom).value;
            else {
              double w = pot.get<double>("WEIGHTS", offset + v.atom);
              derivative =
                  w * (dc * evaluator.embedding(v.atom, q, 'c', v.atom).value +
                       ds * evaluator.embedding(v.atom, q, 's', v.atom).value);
            }
            gradient[j] += weight * residual * derivative;
          }
        }
      }
      return std::isfinite(*cost);
    }
  }
  /** Ceres's callback: a trial point the model refuses (std::runtime_error:
   *  a centre or scale it cannot evaluate) is infeasible, which Ceres
   *  handles by stepping back; anything else (std::logic_error, an internal
   *  error) stops the solve and is rethrown by rethrow_internal(). The
   *  starting point is evaluated before the solve with cost_at, outside
   *  any catch, so a precondition refusal reports itself. */
  bool Evaluate(const double *x, double *cost,
                double *gradient) const override {
    if (internal_)
      return false;
    try {
      return cost_at(x, cost, gradient);
    } catch (const std::runtime_error &e) {
      refused_ = e.what();
      return false;
    } catch (...) {
      internal_ = std::current_exception();
      return false;
    }
  }
  void rethrow_internal() const {
    if (internal_)
      std::rethrow_exception(internal_);
  }
  const std::string &last_refusal() const { return refused_; }
};
} // namespace
std::unique_ptr<models::EAMRefit>
compact_embed_fit(const core::Context &context, const models::EAMSetflIO &eam,
                  models::EAMRefitConfig &cfg) {
  for (const auto &key : context.storage_snapshot())
    if (key.first.rfind("REFIT_COMPACT_", 0) == 0 || key.first == "INIT" ||
        key.first == "EXPLORE" || key.first == "REFINE" ||
        key.first == "FUSE" || key.first.rfind("REFIT_CURVE_", 0) == 0 ||
        key.first == "REFIT_OBJECTIVE")
      throw std::runtime_error("tadah refit: " + key.first +
                               " belongs to the legacy ladder; component grids "
                               "are the compact seeds");
  const int iterations =
      mlip::engines::read_refit_numbers(context).component_iterations;
  std::vector<double> x;
  auto *function = new Objective(eam, cfg, x);
  ceres::GradientProblem problem(function);
  // The starting point (the configured grids) outside the solve: a table
  // the declared FORM cannot follow, or a component that cannot be
  // evaluated, is refused here with its own message.
  {
    double cost = 0;
    std::vector<double> gradient(x.size());
    if (!function->cost_at(x.data(), &cost, gradient.data()))
      throw std::runtime_error(
          "DM_Embed compact refit: the declared components give a non-finite "
          "fit cost at their configured grids");
  }
  ceres::GradientProblemSolver::Options options;
  options.max_num_iterations = iterations;
  options.line_search_direction_type = ceres::LBFGS;
  // no progress lines (glog, stderr) whatever VERBOSE is; the summary is
  // logged below
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = false;
  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(options, problem, x.data(), &summary);
  function->rethrow_internal();
  if (!summary.IsSolutionUsable())
    throw std::runtime_error(
        "DM_Embed compact refit: the centre/scale optimisation failed: " +
        summary.message +
        (function->last_refusal().empty()
             ? std::string()
             : " (last trial point refused: " + function->last_refusal() +
                   ")"));
  cfg = function->decode(x.data());
  auto fit = std::make_unique<models::EAMRefit>(eam, cfg);
  fit->fit();
  core::Logger::instance().log(
      core::LogLevel::INFO, "DM_Embed compact refit: ", summary.BriefReport());
  if (summary.termination_type == ceres::NO_CONVERGENCE)
    core::Logger::instance().log(
        core::LogLevel::WARNING,
        "DM_Embed compact refit: the centre/scale optimisation stopped after " +
            std::to_string(iterations) +
            " iterations without converging (" + summary.message +
            "); the potential is written at the point it reached. Raise "
            "REFIT_COMPONENT_ITERATIONS (" + std::to_string(iterations) +
            (context.has_stored("REFIT_COMPONENT_ITERATIONS") ? ""
                                                              : ", the default") +
            ") to let it converge.");
  return fit;
}
} // namespace hpo
} // namespace tadah
#endif

#include <tadah/core/logger.h>
#include <tadah/hpo/refit/embed_compact.h>
#ifdef TADAH_USE_CERES
#include <algorithm>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>
#include <cmath>

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
  bool Evaluate(const double *x, double *cost,
                double *gradient) const override {
    try {
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
    } catch (const std::exception &) {
      return false;
    } // Ceres rejects infeasible trial points.
  }
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
  std::vector<double> x;
  auto *function = new Objective(eam, cfg, x);
  ceres::GradientProblem problem(function);
  ceres::GradientProblemSolver::Options options;
  options.max_num_iterations = 60;
  options.line_search_direction_type = ceres::LBFGS;
  ceres::GradientProblemSolver::Summary summary;
  ceres::Solve(options, problem, x.data(), &summary);
  if (!summary.IsSolutionUsable())
    throw std::runtime_error("DM_Embed compact refit: " +
                             summary.BriefReport());
  cfg = function->decode(x.data());
  auto fit = std::make_unique<models::EAMRefit>(eam, cfg);
  fit->fit();
  core::Logger::instance().log(
      core::LogLevel::INFO, "DM_Embed compact refit: ", summary.BriefReport());
  return fit;
}
} // namespace hpo
} // namespace tadah
#endif

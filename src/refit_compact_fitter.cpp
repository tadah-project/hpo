#include <tadah/hpo/refit/compact_fitter.h>

#ifdef TADAH_USE_CERES

#include <tadah/models/descriptors/d_basis_functions.h>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace tadah {
namespace hpo {
namespace refit {

using tadah::models::B;
using tadah::models::dB;

// ---- optimiser termination → short tag (for the per-stage convergence log)
namespace {
const char *nlopt_term(nlopt::result r) {
  switch (r) {
    case nlopt::SUCCESS:          return "SUCCESS";
    case nlopt::STOPVAL_REACHED:  return "STOPVAL_REACHED";
    case nlopt::FTOL_REACHED:     return "FTOL_REACHED";
    case nlopt::XTOL_REACHED:     return "XTOL_REACHED";
    case nlopt::MAXEVAL_REACHED:  return "MAXEVAL_REACHED";
    case nlopt::MAXTIME_REACHED:  return "MAXTIME_REACHED";
    case nlopt::FAILURE:          return "FAILURE";
    case nlopt::INVALID_ARGS:     return "INVALID_ARGS";
    case nlopt::OUT_OF_MEMORY:    return "OUT_OF_MEMORY";
    case nlopt::ROUNDOFF_LIMITED: return "ROUNDOFF_LIMITED";
    case nlopt::FORCED_STOP:      return "FORCED_STOP";
    default:                      return "UNKNOWN";
  }
}
const char *ceres_term(ceres::TerminationType t) {
  switch (t) {
    case ceres::CONVERGENCE:    return "CONVERGENCE";
    case ceres::NO_CONVERGENCE: return "NO_CONVERGENCE";
    case ceres::FAILURE:        return "FAILURE";
    case ceres::USER_SUCCESS:   return "USER_SUCCESS";
    case ceres::USER_FAILURE:   return "USER_FAILURE";
    default:                    return "UNKNOWN";
  }
}
}  // namespace

// ---- basis dispatch: value g(x_c) and r-derivative eta*g'(x_c) per family.
// A 3-way switch on a hot member is branch-predicted perfectly; the blip
// path calls the exact same functions as before (bit-for-bit regression
// gated against a fixed-seed golden refit).
namespace {
inline double bval(BasisKind k, double xc) {
  switch (k) {
    case BasisKind::Blip:   return B(xc, 1.0);
    case BasisKind::Knot5L: return tadah::models::K5L(xc);
    case BasisKind::Knot5R: return tadah::models::K5R(xc);
    case BasisKind::Blip5:  return tadah::models::B5(xc);
  }
  return 0.0;
}
inline double bder(BasisKind k, double xc, double eta) {
  switch (k) {
    case BasisKind::Blip:   return dB(xc, eta);
    case BasisKind::Knot5L: return tadah::models::dK5L(xc, eta);
    case BasisKind::Knot5R: return tadah::models::dK5R(xc, eta);
    case BasisKind::Blip5:  return tadah::models::dB5(xc, eta);
  }
  return 0.0;
}
// degree-aware dispatch for the knot kinds (d == 3: cubic atom — an exact
// f''' corner; d == 5 or non-knot kind: defer to the family default above)
inline double bval_d(BasisKind k, int d, double xc) {
  if (d == 3) {
    if (k == BasisKind::Knot5L) return tadah::models::K3L(xc);
    if (k == BasisKind::Knot5R) return tadah::models::K3R(xc);
  }
  return bval(k, xc);
}
inline double bder_d(BasisKind k, int d, double xc, double eta) {
  if (d == 3) {
    if (k == BasisKind::Knot5L) return tadah::models::dK3L(xc, eta);
    if (k == BasisKind::Knot5R) return tadah::models::dK3R(xc, eta);
  }
  return bder(k, xc, eta);
}
// bare second derivative d2g/dx_c^2 (d2/dr2 = eta^2 * this); every family
// is at least C^2, so this is continuous — used by the d2 inspection
// criterion (KnotMix double-win) and BlipFit::deriv2
inline double bval2_d(BasisKind k, int d, double xc) {
  switch (k) {
    case BasisKind::Blip:   return tadah::models::d2B(xc);
    case BasisKind::Blip5:  return tadah::models::d2B5(xc);
    case BasisKind::Knot5L:
      return d == 3 ? tadah::models::d2K3L(xc) : tadah::models::d2K5L(xc);
    case BasisKind::Knot5R:
      return d == 3 ? tadah::models::d2K3R(xc) : tadah::models::d2K5R(xc);
  }
  return 0.0;
}
}  // namespace

// ---------------------------------------------------------------- BlipFit
double BlipFit::eval(double x) const {
  double acc = 0.0;
  for (std::size_t k = 0; k < mu.size(); ++k)
    acc += c[k] * bval_d(kind, degree(static_cast<int>(k)),
                         eta[k] * (x - mu[k]));
  return acc;
}

double BlipFit::deriv(double x) const {
  double acc = 0.0;
  for (std::size_t k = 0; k < mu.size(); ++k)
    acc += c[k] * bder_d(kind, degree(static_cast<int>(k)),
                         eta[k] * (x - mu[k]), eta[k]);
  return acc;
}

double BlipFit::deriv2(double x) const {
  double acc = 0.0;
  for (std::size_t k = 0; k < mu.size(); ++k)
    acc += c[k] * eta[k] * eta[k] *
           bval2_d(kind, degree(static_cast<int>(k)), eta[k] * (x - mu[k]));
  return acc;
}

// ------------------------------------------------------------- Ceres cost
// Residuals r_i = sqrt(w_i) (sum_k c_k B(eta_k(x_i-mu_k)) - y_i), parameter
// blocks {c, mu, eta}. Analytic Jacobian uses only B and B' = dB/eta — the
// B'' knot discontinuities never enter, which is why LM stays healthy where
// quadratic-model DFO (BOBYQA) failed on blips.
class CurveCost : public ceres::CostFunction {
public:
  CurveCost(std::vector<double> xs, std::vector<double> ys,
            std::vector<double> wts, int n, BasisKind kind = BasisKind::Blip,
            std::vector<int> deg = {})
      : xs_(std::move(xs)), ys_(std::move(ys)), wts_(std::move(wts)), n_(n),
        kind_(kind), deg_(std::move(deg)) {
    set_num_residuals(static_cast<int>(xs_.size()));
    mutable_parameter_block_sizes()->assign(3, n_);  // c, mu, eta
  }

  bool Evaluate(double const *const *p, double *residuals,
                double **jacobians) const override {
    const double *c = p[0], *mu = p[1], *eta = p[2];
    const std::size_t N = xs_.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double x = xs_[i];
      const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
      double m = 0.0;
      for (int k = 0; k < n_; ++k) {
        const int d = (k < static_cast<int>(deg_.size())) ? deg_[k] : 5;
        const double xc = eta[k] * (x - mu[k]);
        const double b = bval_d(kind_, d, xc);
        m += c[k] * b;
        if (jacobians) {
          const double db = bder_d(kind_, d, xc, eta[k]);  // = eta * g'(xc)
          if (jacobians[0]) jacobians[0][i * n_ + k] = sw * b;
          if (jacobians[1]) jacobians[1][i * n_ + k] = -sw * c[k] * db;
          if (jacobians[2])
            jacobians[2][i * n_ + k] =
                sw * c[k] * (x - mu[k]) * (db / eta[k]);  // (x-mu) g'(xc)
        }
      }
      residuals[i] = sw * (m - ys_[i]);
    }
    return true;
  }

private:
  // by-value copies: the cost may outlive the caller's buffers (joint refine)
  std::vector<double> xs_, ys_, wts_;
  int n_;
  BasisKind kind_;
  std::vector<int> deg_;  // per-atom degree (knot kinds; empty = all 5)
};

// ---------------------------------------------------------- CompactBlipFitter
CompactBlipFitter::CompactBlipFitter(std::vector<double> xs,
                                     std::vector<double> ys,
                                     std::vector<double> wts, double lo,
                                     double hi, const CompactConfig &cfg,
                                     BasisKind kind)
    : xs_(std::move(xs)), ys_(std::move(ys)), wts_(std::move(wts)), lo_(lo),
      hi_(hi), span_(hi - lo), cfg_(cfg), kind_(kind) {
  eta0_ = (cfg_.n_min > 1) ? (cfg_.n_min - 1) / span_ : 1.0 / span_;
}

double CompactBlipFitter::solve_coeffs(BlipFit &f) const {
  const int n = f.n();
  const std::size_t N = xs_.size();
  Eigen::MatrixXd A(N, n);
  Eigen::VectorXd b(N);
  for (std::size_t i = 0; i < N; ++i) {
    const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
    for (int k = 0; k < n; ++k)
      A(i, k) = sw * bval_d(f.kind, f.degree(k),
                            f.eta[k] * (xs_[i] - f.mu[k]));
    b(i) = sw * ys_[i];
  }
  // rank-revealing: a blip pushed outside the sampled range yields a ~zero
  // column, which plain QR mishandles
  Eigen::VectorXd c =
      A.completeOrthogonalDecomposition().solve(b);
  f.c.assign(c.data(), c.data() + n);
  double se = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double d = f.eval(xs_[i]) - ys_[i];
    se += d * d;
  }
  f.curve_rmse = std::sqrt(se / N);
  return f.curve_rmse;
}

namespace {
struct VarproCtx {
  const CompactBlipFitter *self;
  BlipFit *work;
  double span;
};
// Soft anti-coalescence factor for the knot families (>= 1): a coalesced
// +-c knot pair is a derivative impostor whose VALUE rmse often beats any
// separated configuration locally, so an unpenalised derivative-free search
// walks straight into it (measured: the ladder pinned at n=1, all knots on
// the boundary). Penalise adjacent gaps inside 2% of the span — comfortably
// below genuine EAM knot spacings (Fe's tightest: 0.1 A ~ 3% of its span).
// Applied to the DERIVATIVE-FREE objective only; the Ceres polish is kept
// exact and its coalesced outcomes are repaired by merge_close_knots.
double knot_gap_factor(const BlipFit &f, double span) {
  if (!one_sided(f.kind) || f.n() < 2) return 1.0;
  std::vector<double> t(f.mu.begin(), f.mu.end());
  std::sort(t.begin(), t.end());
  const double g2 = 0.02 * span;
  double pen = 0.0;
  for (std::size_t i = 1; i < t.size(); ++i) {
    const double d = t[i] - t[i - 1];
    if (d < g2) {
      const double v = (g2 - d) / g2;
      pen += v * v;
    }
  }
  return 1.0 + pen;
}
double varpro_obj(const std::vector<double> &v, std::vector<double> &grad,
                  void *data) {
  (void)grad;  // SBPLX is derivative-free
  auto *ctx = static_cast<VarproCtx *>(data);
  const int n = ctx->work->n();
  for (int k = 0; k < n; ++k) ctx->work->mu[k] = v[k];
  if (!one_sided(ctx->work->kind))  // knot families: eta pinned (gauge)
    for (int k = 0; k < n; ++k) ctx->work->eta[k] = v[n + k];
  return ctx->self->solve_coeffs(*ctx->work) *
         knot_gap_factor(*ctx->work, ctx->span);
}
}  // namespace

double CompactBlipFitter::d2_rmse(const BlipFit &f) const {
  const std::size_t N = xs_.size();
  if (N < 5) return 0.0;
  double se = 0.0;
  std::size_t cnt = 0;
  for (std::size_t i = 1; i + 1 < N; ++i) {
    const double hl = xs_[i] - xs_[i - 1], hr = xs_[i + 1] - xs_[i];
    if (hl <= 0.0 || hr <= 0.0) continue;
    // 3-point second derivative (nonuniform-safe) of the SAMPLE data
    const double d2_data =
        2.0 * (hl * ys_[i + 1] - (hl + hr) * ys_[i] + hr * ys_[i - 1]) /
        (hl * hr * (hl + hr));
    const double d = f.deriv2(xs_[i]) - d2_data;
    se += d * d;
    ++cnt;
  }
  return cnt ? std::sqrt(se / cnt) : 0.0;
}

double CompactBlipFitter::reduced_mse(BlipFit &f, std::vector<double> *grad) const {
  solve_coeffs(f);  // inner closed-form c at the current (mu, eta)
  const int n = f.n();
  const bool os = one_sided(f.kind);  // knot families: mu-only theta space
  const std::size_t N = xs_.size();
  double sse = 0.0;
  if (grad) grad->assign(os ? n : 2 * n, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
    const double r = sw * (f.eval(xs_[i]) - ys_[i]);
    sse += r * r;
    if (grad) {
      // Golub-Pereyra: A^T r = 0 at the inner optimum, so the c-variation
      // term vanishes and d(SSE)/dtheta = 2 r^T (dA/dtheta) c exactly.
      for (int k = 0; k < n; ++k) {
        const double xc = f.eta[k] * (xs_[i] - f.mu[k]);
        const double db = bder_d(f.kind, f.degree(k), xc, f.eta[k]);
        (*grad)[k] += 2.0 * r * sw * f.c[k] * (-db);
        if (!os)
          (*grad)[n + k] +=
              2.0 * r * sw * f.c[k] * (xs_[i] - f.mu[k]) * (db / f.eta[k]);
      }
    }
  }
  if (grad)
    for (auto &g : *grad) g /= N;
  return sse / N;
}

namespace {
// shared bounds/start vector helpers for the (mu..., eta...) parameterisation
struct ThetaBox {
  std::vector<double> lb, ub, v;
};
// knot-position box per family orientation. Left-active: mu in [lo, hi] —
// mu > hi breaks the smooth-vanish-at-rcut construction (a (t-rcut)^k step
// would survive the hard cutoff), mu < lo is a dead column. Right-active:
// [lo - span, hi] — mu < lo supplies the polynomial span the embedding
// needs near rho = 0; mu > hi is a dead column.
inline double knot_lb(BasisKind k, double lo, double span) {
  return (k == BasisKind::Knot5R) ? lo - span : lo;
}
ThetaBox theta_box(const BlipFit &f, double lo, double hi, double span) {
  const int n = f.n();
  ThetaBox b;
  if (one_sided(f.kind)) {  // single shape param per atom (eta pinned at 1)
    b.lb.resize(n); b.ub.resize(n); b.v.resize(n);
    const double lb = knot_lb(f.kind, lo, span);
    for (int k = 0; k < n; ++k) {
      b.lb[k] = lb;
      b.ub[k] = hi;
      b.v[k] = std::clamp(f.mu[k], b.lb[k], b.ub[k]);
    }
    return b;
  }
  const double eta0 = std::max(1, n - 1) / span;
  b.lb.resize(2 * n);
  b.ub.resize(2 * n);
  b.v.resize(2 * n);
  for (int k = 0; k < n; ++k) {
    b.lb[k] = lo - span;       b.ub[k] = hi + span;
    b.lb[n + k] = 0.03 * eta0; b.ub[n + k] = 80.0 * eta0;
    b.v[k] = std::clamp(f.mu[k], b.lb[k], b.ub[k]);
    b.v[n + k] = std::clamp(f.eta[k], b.lb[n + k], b.ub[n + k]);
  }
  return b;
}
void theta_unpack(const std::vector<double> &v, BlipFit &f) {
  const int n = f.n();
  for (int k = 0; k < n; ++k) f.mu[k] = v[k];
  if (!one_sided(f.kind))
    for (int k = 0; k < n; ++k) f.eta[k] = v[n + k];
}
struct GradCtx {
  const CompactBlipFitter *self;
  BlipFit *work;
};
double nlopt_grad_obj(const std::vector<double> &v, std::vector<double> &grad,
                      void *data) {
  auto *ctx = static_cast<GradCtx *>(data);
  theta_unpack(v, *ctx->work);
  if (!grad.empty()) {
    std::vector<double> g;
    const double f = ctx->self->reduced_mse(*ctx->work, &g);
    grad = g;
    return f;
  }
  return ctx->self->reduced_mse(*ctx->work, nullptr);
}
}  // namespace

double CompactBlipFitter::sbplx_explore(BlipFit &f, int maxeval) const {
  const int n = f.n();
  const double eta0 = std::max(1, n - 1) / span_;

  // NLopt local algorithms (derivative-free + VARPRO-gradient LBFGS)
  const std::string &a = cfg_.explore_algo;
  nlopt::algorithm algo = nlopt::LN_SBPLX;
  bool use_grad = false;
  if (a == "SBPLX") algo = nlopt::LN_SBPLX;
  else if (a == "NELDERMEAD") algo = nlopt::LN_NELDERMEAD;
  else if (a == "BOBYQA") algo = nlopt::LN_BOBYQA;
  else if (a == "PRAXIS") algo = nlopt::LN_PRAXIS;
  else if (a == "NLOPT_LBFGS") { algo = nlopt::LD_LBFGS; use_grad = true; }
  else if (a == "DLIB_LBFGS") {
    // dlib box-constrained LBFGS on the VARPRO reduced objective
    ThetaBox b = theta_box(f, lo_, hi_, span_);
    const int dim = static_cast<int>(b.v.size());
    using dvec = dlib::matrix<double, 0, 1>;
    dvec x(dim), xl(dim), xu(dim);
    for (int i = 0; i < dim; ++i) { x(i) = b.v[i]; xl(i) = b.lb[i]; xu(i) = b.ub[i]; }
    BlipFit work = f;
    auto obj = [&](const dvec &p) {
      std::vector<double> v(p.begin(), p.end());
      theta_unpack(v, work);
      return reduced_mse(work, nullptr);
    };
    auto der = [&](const dvec &p) {
      std::vector<double> v(p.begin(), p.end()), g;
      theta_unpack(v, work);
      reduced_mse(work, &g);
      dvec out(dim);
      for (int i = 0; i < dim; ++i) out(i) = g[i];
      return out;
    };
    last_explore_term_ = "DLIB_LBFGS(delta/maxeval)";
    try {
      dlib::find_min_box_constrained(
          dlib::lbfgs_search_strategy(20),
          dlib::objective_delta_stop_strategy(cfg_.sbplx_ftol, maxeval),
          obj, der, x, xl, xu);
    } catch (const std::exception &) {
      last_explore_term_ = "DLIB_LBFGS(threw)";
    }
    std::vector<double> v(x.begin(), x.end());
    theta_unpack(v, f);
    return solve_coeffs(f);
  } else if (a == "NONE") {
    last_explore_term_ = "NONE(linear-solve)";
    return solve_coeffs(f);
  } else {
    throw std::runtime_error("compact_fitter: unknown explore algo '" + a +
                             "' (SBPLX|NELDERMEAD|BOBYQA|PRAXIS|NLOPT_LBFGS|"
                             "DLIB_LBFGS|NONE)");
  }

  ThetaBox b = theta_box(f, lo_, hi_, span_);
  const unsigned dim = static_cast<unsigned>(b.v.size());
  nlopt::opt opt(algo, dim);
  std::vector<double> step(dim);
  for (int k = 0; k < n; ++k) step[k] = 0.12 * span_;
  if (!one_sided(f.kind))
    for (int k = 0; k < n; ++k) step[n + k] = 0.4 * eta0;
  opt.set_lower_bounds(b.lb);
  opt.set_upper_bounds(b.ub);
  opt.set_initial_step(step);
  opt.set_xtol_rel(cfg_.sbplx_xtol);
  opt.set_ftol_rel(cfg_.sbplx_ftol);
  opt.set_maxeval(maxeval);
  BlipFit work = f;
  double fval = 1e300;
  if (use_grad) {
    GradCtx ctx{this, &work};
    opt.set_min_objective(nlopt_grad_obj, &ctx);
  } else {
    VarproCtx ctx_df{this, &work, span_};
    opt.set_min_objective(varpro_obj, &ctx_df);
    try {
      last_explore_term_ = nlopt_term(opt.optimize(b.v, fval));
    } catch (const std::exception &) {
      last_explore_term_ = nlopt_term(opt.last_optimize_result());
    }
    theta_unpack(b.v, f);
    return solve_coeffs(f);
  }
  try {
    last_explore_term_ = nlopt_term(opt.optimize(b.v, fval));
  } catch (const std::exception &) {
    last_explore_term_ = nlopt_term(opt.last_optimize_result());
  }
  theta_unpack(b.v, f);
  return solve_coeffs(f);
}

double CompactBlipFitter::explore(BlipFit &f) const {
  return sbplx_explore(f, cfg_.sbplx_maxeval);
}

double CompactBlipFitter::polish_dispatch(BlipFit &f) const {
  const std::string &a = cfg_.polish_algo;
  if (a == "CERES_LM" || a == "CERES_DOGLEG") return polish(f);
  if (a == "NONE") { last_polish_term_ = "NONE(linear-solve)"; return solve_coeffs(f); }
  if (a == "NLOPT_LBFGS" || a == "DLIB_LBFGS") {
    // VARPRO reduced-gradient local, tight tolerances
    ThetaBox b = theta_box(f, lo_, hi_, span_);
    const int dim = static_cast<int>(b.v.size());
    if (a == "NLOPT_LBFGS") {
      nlopt::opt opt(nlopt::LD_LBFGS, static_cast<unsigned>(dim));
      opt.set_lower_bounds(b.lb);
      opt.set_upper_bounds(b.ub);
      opt.set_ftol_rel(cfg_.ceres_ftol);
      opt.set_xtol_rel(1e-14);
      opt.set_maxeval(50 * cfg_.ceres_max_iter);
      BlipFit work = f;
      GradCtx ctx{this, &work};
      opt.set_min_objective(nlopt_grad_obj, &ctx);
      double fval = 1e300;
      try {
        last_polish_term_ = nlopt_term(opt.optimize(b.v, fval));
      } catch (const std::exception &) {
        last_polish_term_ = nlopt_term(opt.last_optimize_result());
      }
      theta_unpack(b.v, f);
      return solve_coeffs(f);
    }
    using dvec = dlib::matrix<double, 0, 1>;
    dvec x(dim), xl(dim), xu(dim);
    for (int i = 0; i < dim; ++i) { x(i) = b.v[i]; xl(i) = b.lb[i]; xu(i) = b.ub[i]; }
    BlipFit work = f;
    auto obj = [&](const dvec &p) {
      std::vector<double> v(p.begin(), p.end());
      theta_unpack(v, work);
      return reduced_mse(work, nullptr);
    };
    auto der = [&](const dvec &p) {
      std::vector<double> v(p.begin(), p.end()), g;
      theta_unpack(v, work);
      reduced_mse(work, &g);
      dvec out(dim);
      for (int i = 0; i < dim; ++i) out(i) = g[i];
      return out;
    };
    last_polish_term_ = "DLIB_LBFGS(delta/maxeval)";
    try {
      dlib::find_min_box_constrained(
          dlib::lbfgs_search_strategy(30),
          dlib::objective_delta_stop_strategy(cfg_.ceres_ftol,
                                              50 * cfg_.ceres_max_iter),
          obj, der, x, xl, xu);
    } catch (const std::exception &) {
      last_polish_term_ = "DLIB_LBFGS(threw)";
    }
    std::vector<double> v(x.begin(), x.end());
    theta_unpack(v, f);
    return solve_coeffs(f);
  }
  throw std::runtime_error("compact_fitter: unknown polish algo '" + a +
                           "' (CERES_LM|CERES_DOGLEG|NLOPT_LBFGS|DLIB_LBFGS|"
                           "NONE)");
}

BlipFit CompactBlipFitter::seed_mlsl() const {
  const int n = cfg_.n_min;
  const double eta0 = std::max(1, n - 1) / span_;
  BlipFit f;
  f.kind = kind_;
  f.mu.resize(n);
  f.eta.assign(n, one_sided(kind_) ? 1.0 : eta0);  // knot families: eta gauge
  if (one_sided(kind_) && cfg_.knot_degree != 5)
    f.deg.assign(n, cfg_.knot_degree);  // pure-cubic seed (Knot3)
  for (int k = 0; k < n; ++k)
    f.mu[k] = lo_ + span_ * (n == 1 ? 0.5 : double(k) / (n - 1));
  f.c.assign(n, 0.0);

  const std::string &a = cfg_.seed_algo;
  if (a == "UNIFORM") {  // no global step: uniform start, polish handles it
    last_seed_term_ = "UNIFORM(no-global-step)";
    solve_coeffs(f);
    return f;
  }
  ThetaBox b = theta_box(f, lo_, hi_, span_);

  if (a == "DLIB_GLOBAL") {
    // dlib's MaxLIPO+TR model-based global search: needs FAR fewer evals
    const int gdim = static_cast<int>(b.v.size());
    using dvec = dlib::matrix<double, 0, 1>;
    dvec xl(gdim), xu(gdim);
    for (int i = 0; i < gdim; ++i) { xl(i) = b.lb[i]; xu(i) = b.ub[i]; }
    BlipFit work = f;
    auto obj = [&](const dvec &p) {
      std::vector<double> v(p.begin(), p.end());
      theta_unpack(v, work);
      return reduced_mse(work, nullptr);
    };
    const int calls = std::max(500, cfg_.mlsl_maxeval / 20);
    last_seed_term_ = "DLIB_GLOBAL(maxcalls=" + std::to_string(calls) + ")";
    auto res = dlib::find_min_global(
        obj, xl, xu, dlib::max_function_calls(static_cast<std::size_t>(calls)));
    std::vector<double> v(res.x.begin(), res.x.end());
    theta_unpack(v, f);
    solve_coeffs(f);
    return f;
  }

  nlopt::algorithm galgo = nlopt::GN_MLSL_LDS;
  if (a == "MLSL") galgo = nlopt::GN_MLSL_LDS;
  else if (a == "DIRECT") galgo = nlopt::GN_DIRECT_L;
  else if (a == "CRS2") galgo = nlopt::GN_CRS2_LM;
  else
    throw std::runtime_error("compact_fitter: unknown seed algo '" + a +
                             "' (MLSL|DIRECT|CRS2|DLIB_GLOBAL|UNIFORM)");

  const unsigned dim = static_cast<unsigned>(b.v.size());
  nlopt::opt local(nlopt::LN_SBPLX, dim);
  local.set_xtol_rel(cfg_.sbplx_xtol);
  local.set_ftol_rel(cfg_.sbplx_ftol);
  local.set_maxeval(cfg_.sbplx_maxeval);

  nlopt::opt opt(galgo, dim);
  opt.set_lower_bounds(b.lb);
  opt.set_upper_bounds(b.ub);
  if (a == "MLSL") opt.set_local_optimizer(local);
  opt.set_maxeval(cfg_.mlsl_maxeval);
  opt.set_population(8 * n);
  BlipFit work = f;
  VarproCtx ctx{this, &work, span_};
  opt.set_min_objective(varpro_obj, &ctx);
  double fval = 1e300;
  try {
    last_seed_term_ = std::string(a) + ":" + nlopt_term(opt.optimize(b.v, fval));
  } catch (const std::exception &) {
    last_seed_term_ = std::string(a) + ":" + nlopt_term(opt.last_optimize_result());
  }
  theta_unpack(b.v, f);
  solve_coeffs(f);
  return f;
}

double CompactBlipFitter::polish(BlipFit &f) const {
  const int n = f.n();
  ceres::Problem prob;
  std::vector<double> &c = f.c, &mu = f.mu, &eta = f.eta;
  prob.AddResidualBlock(new CurveCost(xs_, ys_, wts_, n, f.kind, f.deg),
                        nullptr, c.data(), mu.data(), eta.data());
  if (one_sided(f.kind)) {
    // knot families: eta is an exact gauge — hold it; knots stay in their box
    prob.SetParameterBlockConstant(eta.data());
    const double lb = knot_lb(f.kind, lo_, span_);
    for (int k = 0; k < n; ++k) {
      prob.SetParameterLowerBound(mu.data(), k, lb);
      prob.SetParameterUpperBound(mu.data(), k, hi_);
    }
  } else {
    const double eta0 = std::max(1, n - 1) / span_;
    for (int k = 0; k < n; ++k) {
      prob.SetParameterLowerBound(mu.data(), k, lo_ - span_);
      prob.SetParameterUpperBound(mu.data(), k, hi_ + span_);
      prob.SetParameterLowerBound(eta.data(), k, 0.03 * eta0);
      prob.SetParameterUpperBound(eta.data(), k, 80.0 * eta0);
    }
  }
  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::DENSE_QR;
  if (cfg_.polish_algo == "CERES_DOGLEG")
    opt.trust_region_strategy_type = ceres::DOGLEG;
  opt.max_num_iterations = cfg_.ceres_max_iter;
  opt.function_tolerance = cfg_.ceres_ftol;
  opt.gradient_tolerance = cfg_.ceres_gtol;
  opt.parameter_tolerance = 1e-16;
  opt.logging_type = ceres::SILENT;
  opt.check_gradients = cfg_.check_gradients;  // FD audit of the analytic J
  // Compact-support bases have numerically-zero Jacobian entries at support
  // edges; Ceres' check has only a RELATIVE knob, so FD noise on those ~0
  // entries (incl. B''' kinks crossed by the probe — the blip is C2, only
  // B''' jumps at its knots) reads as large relative error there.
  // Measured: 4/1920 components, worst 1.2e-3, all near-zero entries; real
  // Jacobian bugs show O(1) errors across whole blocks.
  opt.gradient_check_relative_precision = 5e-3;
  ceres::Solver::Summary sum;
  ceres::Solve(opt, &prob, &sum);
  if (cfg_.check_gradients && sum.termination_type == ceres::FAILURE)
    throw std::runtime_error("compact_fitter: analytic Jacobian failed the "
                             "Ceres finite-difference gradient check: " +
                             sum.message);
  last_polish_term_ = std::string(ceres_term(sum.termination_type)) + "(" +
                      std::to_string(sum.iterations.size()) + "it)";
  return solve_coeffs(f);  // final closed-form coeffs at the polished (mu,eta)
}

std::vector<double> CompactBlipFitter::residual_peaks(const BlipFit &f,
                                                      int k) const {
  // greedy: repeated worst-residual scans, excluding samples within 2% span
  // of an already-chosen peak. k=1 reproduces the legacy single-worst scan
  // exactly (ascending scan, strict >).
  std::vector<double> res(xs_.size());
  for (std::size_t i = 0; i < xs_.size(); ++i) {
    const double w = wts_.empty() ? 1.0 : wts_[i];
    res[i] = std::fabs(f.eval(xs_[i]) - ys_[i]) * std::sqrt(w);
  }
  const double sep = 0.02 * span_;
  std::vector<double> peaks;
  for (int p = 0; p < k; ++p) {
    double worst = -1.0, xw = 0.5 * (lo_ + hi_);
    bool found = false;
    for (std::size_t i = 0; i < xs_.size(); ++i) {
      bool excluded = false;
      for (double q : peaks)
        if (std::fabs(xs_[i] - q) < sep) { excluded = true; break; }
      if (excluded) continue;
      if (res[i] > worst) { worst = res[i]; xw = xs_[i]; found = true; }
    }
    if (!found) break;
    peaks.push_back(xw);
  }
  return peaks;
}

void CompactBlipFitter::insert_blip_at_worst(BlipFit &f) const {
  const std::vector<double> p = residual_peaks(f, 1);
  insert_at(f, p.empty() ? 0.5 * (lo_ + hi_) : p[0]);
}

void CompactBlipFitter::insert_at(BlipFit &f, double xw) const {
  insert_at_deg(f, xw, cfg_.knot_degree);
}

void CompactBlipFitter::insert_at_deg(BlipFit &f, double xw, int d) const {
  if (one_sided(f.kind)) {
    // A one-sided atom is identically ZERO at (and beyond) its own knot, so
    // a knot AT the residual peak has no leverage there — insert it 30% of
    // the gap toward the next knot on the atom's ACTIVE side (measured in
    // the pilot: ~800x value-RMSE difference vs inserting at the peak).
    double t;
    if (f.kind == BasisKind::Knot5L) {       // active below the knot
      double next = hi_;
      for (double m : f.mu)
        if (m > xw && m < next) next = m;
      t = std::min(hi_, xw + 0.3 * (next - xw));
    } else {                                  // right-active: below the peak
      double next = knot_lb(f.kind, lo_, span_);
      for (double m : f.mu)
        if (m < xw && m > next) next = m;
      t = std::max(knot_lb(f.kind, lo_, span_), xw - 0.3 * (xw - next));
    }
    f.mu.push_back(t);
    f.eta.push_back(1.0);
    f.c.push_back(0.0);
    if (f.deg.empty() && d != 5)
      f.deg.assign(f.mu.size() - 1, 5);  // materialise: prior atoms quintic
    if (!f.deg.empty() || d != 5) f.deg.push_back(d);
    return;
  }
  const double eta0 = std::max(1, f.n()) / span_;  // (n+1)-blip uniform scale
  f.mu.push_back(xw);
  f.eta.push_back(eta0);
  f.c.push_back(0.0);
}

void CompactBlipFitter::jitter(BlipFit &f, std::mt19937 &rng) const {
  std::normal_distribution<double> g(0.0, 1.0);
  if (one_sided(f.kind)) {  // knot jitter only; eta is pinned (gauge)
    const double lb = knot_lb(f.kind, lo_, span_);
    for (int k = 0; k < f.n(); ++k)
      f.mu[k] = std::clamp(f.mu[k] + 0.02 * span_ * g(rng), lb, hi_);
    return;
  }
  const double eta0 = std::max(1, f.n() - 1) / span_;
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] = std::clamp(f.mu[k] + 0.02 * span_ * g(rng), lo_ - span_,
                         hi_ + span_);
    f.eta[k] = std::clamp(f.eta[k] * std::exp(0.2 * g(rng)), 0.03 * eta0,
                          80.0 * eta0);
  }
}

void CompactBlipFitter::merge_close_knots(BlipFit &f) const {
  // Coalescence guard for the knot families: near-coincident knot pairs with
  // large opposite-sign coefficients are divided-difference impostors of
  // higher-order terms — value RMSE stays pristine while f'' acquires local
  // spikes and the design rank collapses (measured in the pilot). Merge any
  // pair closer than 1e-3 * span to its midpoint, then RE-INSERT a fresh
  // knot at the worst residual for every knot removed ("merge + re-insert
  // elsewhere", the pilot's recipe). Without the re-insert the ladder LOSES
  // capacity every rung the local optimiser walks into a coalesced basin
  // (measured here: the ladder pinned at n=1 forever) — with it, a coalesced
  // candidate degrades gracefully into an insertion-only step.
  if (!one_sided(f.kind) || f.n() < 2) return;
  const int n_orig = f.n();
  const double gap = 1e-3 * span_;
  std::vector<int> idx(f.mu.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return f.mu[a] < f.mu[b]; });
  // merge only SAME-degree neighbours: a cubic and a quintic atom at the
  // same knot are linearly independent, not a coalesced impostor pair
  std::vector<double> mu_new;
  std::vector<int> deg_new;
  bool merged = false;
  for (std::size_t s = 0; s < idx.size(); ++s) {
    const double m = f.mu[idx[s]];
    const int d = f.degree(idx[s]);
    if (!mu_new.empty() && m - mu_new.back() < gap && d == deg_new.back()) {
      mu_new.back() = 0.5 * (mu_new.back() + m);  // midpoint merge
      merged = true;
    } else {
      mu_new.push_back(m);
      deg_new.push_back(d);
    }
  }
  if (!merged) return;
  if (cfg_.log_fn)
    cfg_.log_fn("coalescence guard: merged " +
                std::to_string(n_orig - static_cast<int>(mu_new.size())) +
                " knot(s), re-inserting at the residual peak(s)");
  const bool any_cubic =
      std::any_of(deg_new.begin(), deg_new.end(), [](int d) { return d != 5; });
  f.mu = std::move(mu_new);
  f.eta.assign(f.mu.size(), 1.0);
  f.c.assign(f.mu.size(), 0.0);
  f.deg = any_cubic ? deg_new : std::vector<int>{};
  solve_coeffs(f);
  while (f.n() < n_orig) {  // restore capacity where the residual lives
    insert_blip_at_worst(f);
    solve_coeffs(f);
  }
}

BlipFit CompactBlipFitter::fit(const Metric &metric,
                               std::vector<LadderRow> &trace) {
  using clock = std::chrono::steady_clock;
  std::mt19937 rng(cfg_.seed);
  nlopt::srand(cfg_.seed);  // MLSL determinism (global NLopt RNG)

  auto log = [&](const std::string &m) {
    if (cfg_.log_fn) cfg_.log_fn(m);
  };
  auto fmt = [](double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.4g", v);
    return std::string(b);
  };
  auto describe = [&](const BlipFit &f) {
    int n3 = 0;
    for (int k = 0; k < f.n(); ++k)
      if (f.degree(k) == 3) ++n3;
    std::string d = "n=" + std::to_string(f.n());
    if (n3 > 0) d += "(" + std::to_string(n3) + " cubic)";
    d += " metric=" + fmt(f.metric) + " curve_rmse=" + fmt(f.curve_rmse) +
         " d2_rmse=" + fmt(d2_rmse(f));
    return d;
  };

  auto stamp = [&](BlipFit &f) {
    polish_dispatch(f);
    merge_close_knots(f);  // knot families only; no-op for blips
    f.metric = metric(f);
  };

  // Compact histogram of stage-termination tags across a rung's candidates,
  // rendered as e.g. "{FTOL_REACHED:2,MAXEVAL_REACHED:1}".
  auto render_hist = [](const std::map<std::string, int> &h) {
    std::string s = "{";
    bool first = true;
    for (const auto &kv : h) {
      s += (first ? "" : ",") + kv.first + ":" + std::to_string(kv.second);
      first = false;
    }
    return s + "}";
  };

  BlipFit best = seed_mlsl();
  stamp(best);
  {
    LadderRow r;
    r.n = best.n(); r.curve_rmse = best.curve_rmse; r.metric = best.metric;
    r.kept = true;
    trace.push_back(r);
  }
  // global stage: ran once; report which convergence criterion it hit so the
  // user can tweak the GLOBAL block (e.g. MAXEVAL_REACHED => raise MAXEVAL).
  log("global(" + cfg_.seed_algo + "): " + describe(best) +
      "  term=" + last_seed_term_ + " polish=" + last_polish_term_ +
      "  [metric <= " + fmt(cfg_.target) + " stops the ladder]");

  // energy-aware insertion (knot kinds with insert_topk > 1): build one
  // candidate per separated residual peak, keep the METRIC-best after a
  // linear solve. The curve-residual peak is not always where the energy
  // metric hurts — measured: phi/F ladders stalled on curve-peak insertion.
  auto insert_best = [&](const BlipFit &from) {
    const int k = (one_sided(kind_)) ? std::max(1, cfg_.insert_topk) : 1;
    const std::vector<double> peaks = residual_peaks(from, k);
    BlipFit out;
    for (double xw : peaks) {
      BlipFit site;  // best candidate AT this site
      if (one_sided(kind_) && cfg_.knot_mix) {
        // KnotMix DOUBLE-WIN rule: accept the cubic atom only if it beats
        // the quintic on the energy metric AND is no worse on the d2
        // (curvature) residual. A genuine f''' break wins both — the C^4
        // quintic cannot make a corner; a SPURIOUS cubic wins value/energy
        // slightly but loses d2. (The first, metric-only detector deployed
        // cubics on 40-60% of atoms everywhere and degraded Cij broadly —
        // measured, harness/out_mix v1.)
        BlipFit q = from, cb = from;
        insert_at_deg(q, xw, 5);
        solve_coeffs(q);
        q.metric = metric(q);
        insert_at_deg(cb, xw, 3);
        solve_coeffs(cb);
        cb.metric = metric(cb);
        site = (cb.metric < q.metric && d2_rmse(cb) <= d2_rmse(q)) ? cb : q;
      } else {
        site = from;
        insert_at_deg(site, xw, cfg_.knot_degree);
        solve_coeffs(site);
        site.metric = metric(site);
      }
      if (site.metric < out.metric) out = site;
    }
    if (out.n() == 0) {  // no peaks (degenerate): legacy single insertion
      out = from;
      insert_blip_at_worst(out);
      solve_coeffs(out);
      out.metric = metric(out);
    }
    return out;
  };

  BlipFit warm = best;  // ladder continues from the latest rung winner, the
                        // ENVELOPE keeps the metric-best across all rungs
  int stall = 0;
  int rung_idx = 0;
  const int opt_every =
      one_sided(kind_) ? std::max(1, cfg_.opt_every) : 1;  // blip: legacy
  while (warm.n() < cfg_.n_max && best.metric > cfg_.target &&
         stall < cfg_.stall_rungs) {
    const auto t0 = clock::now();
    ++rung_idx;
    const bool opt_rung = (opt_every == 1) || (rung_idx % opt_every == 0);

    std::map<std::string, int> expl_hist, pol_hist;  // per-rung term tally
    BlipFit rung_best;
    if (one_sided(kind_)) {
      BlipFit base = insert_best(warm);
      if (!opt_rung) {
        // cheap growth rung: insertion + linear solve only (pilot-measured
        // strong baseline, ~300x cheaper than per-rung optimisation)
        merge_close_knots(base);
        base.metric = metric(base);
        rung_best = base;
      } else {
        rung_best = base;  // the un-optimised insertion is a candidate too:
                           // the optimiser sometimes lands in a worse basin
        for (int j = 0; j <= cfg_.jitters; ++j) {
          BlipFit cand = base;
          if (j > 0) jitter(cand, rng);
          explore(cand);
          stamp(cand);
          ++expl_hist[last_explore_term_];
          ++pol_hist[last_polish_term_];
          if (cand.metric < rung_best.metric) rung_best = cand;
        }
      }
    } else {
      // ---- legacy blip rung (bit-for-bit regression-gated) ----
      BlipFit base = warm;
      insert_blip_at_worst(base);
      for (int j = 0; j <= cfg_.jitters; ++j) {
        BlipFit cand = base;
        if (j > 0) jitter(cand, rng);
        explore(cand);
        stamp(cand);
        ++expl_hist[last_explore_term_];
        ++pol_hist[last_polish_term_];
        if (cand.metric < rung_best.metric) rung_best = cand;
      }
    }

    LadderRow row;
    row.n = rung_best.n();
    row.curve_rmse = rung_best.curve_rmse;
    row.metric = rung_best.metric;
    row.seconds =
        std::chrono::duration<double>(clock::now() - t0).count();
    if (rung_best.metric < best.metric) {
      best = rung_best;
      row.kept = true;
      stall = 0;
    } else {
      ++stall;
    }
    trace.push_back(row);
    // per-stage convergence reasons: aggregated across this rung's candidates
    // (explore + polish run once per candidate). A run of MAXEVAL_REACHED here
    // means the EXPLORE budget is the binding limit — raise MAXEVAL or loosen
    // FTOL_REL; CONVERGENCE means the tolerance bound first.
    const std::string term = opt_rung
        ? "  explore=" + render_hist(expl_hist) + " polish=" +
              render_hist(pol_hist)
        : "  (grow: insertion+solve, no optimiser)";
    log("rung " + std::to_string(rung_idx) + (opt_rung ? " (opt): " : " (grow): ") +
        describe(rung_best) + (row.kept ? "  KEPT" : "  stall=" +
        std::to_string(stall) + "/" + std::to_string(cfg_.stall_rungs)) +
        "  (" + fmt(row.seconds) + "s)" + term);
    warm = rung_best;  // continue the ladder from this rung regardless
  }

  // final full optimisation of the envelope best when intermediate rungs
  // were insertion-only ("grow fast, optimise at the end" — pilot recipe)
  if (one_sided(kind_) && opt_every > 1) {
    const auto t0 = clock::now();
    BlipFit fin = best;
    explore(fin);
    stamp(fin);
    if (fin.metric < best.metric) {
      best = fin;
      LadderRow row;
      row.n = best.n();
      row.curve_rmse = best.curve_rmse;
      row.metric = best.metric;
      row.seconds = std::chrono::duration<double>(clock::now() - t0).count();
      row.kept = true;
      trace.push_back(row);
      log("final optimisation: " + describe(best) + "  KEPT");
    } else {
      log("final optimisation: no improvement (envelope best stands)");
    }
  }
  log("ladder DONE: " + describe(best));
  return best;
}

// ------------------------------------------------------------- joint refine
namespace {
// E-V residuals over the scan: r_i = w_i (E_fit(i) - E_orig(i)), with
// E(i) = sum_d 0.5 phi(r_d) + F(rho_i), rho_i = sum_d rho(r_d).
// Parameter blocks: {c,mu,eta} x {phi,rho,F} (9 blocks). All derivatives are
// analytic: the only nontrivial chain is d(F(rho_i))/d(theta_rho) =
// F'(rho_i) * sum_d d(rho(r_d))/d(theta_rho).
class EnergyCost : public ceres::CostFunction {
public:
  EnergyCost(const std::vector<std::vector<double>> &dists,
             const std::vector<double> &e0, const std::vector<double> &w,
             int nphi, int nrho, int nF, BasisKind kphi = BasisKind::Blip,
             BasisKind krho = BasisKind::Blip, BasisKind kF = BasisKind::Blip,
             std::vector<int> dphi = {}, std::vector<int> drho = {},
             std::vector<int> dF = {})
      : dists_(dists), e0_(e0), w_(w), nphi_(nphi), nrho_(nrho), nF_(nF),
        kphi_(kphi), krho_(krho), kF_(kF), dphi_(std::move(dphi)),
        drho_(std::move(drho)), dF_(std::move(dF)) {
    set_num_residuals(static_cast<int>(dists_.size()));
    auto *s = mutable_parameter_block_sizes();
    s->assign({nphi_, nphi_, nphi_, nrho_, nrho_, nrho_, nF_, nF_, nF_});
  }
  static int deg_of(const std::vector<int> &d, int k) {
    return (k < static_cast<int>(d.size())) ? d[k] : 5;
  }

  bool Evaluate(double const *const *p, double *res,
                double **jac) const override {
    const double *cP = p[0], *mP = p[1], *eP = p[2];
    const double *cR = p[3], *mR = p[4], *eR = p[5];
    const double *cF = p[6], *mF = p[7], *eF = p[8];
    const std::size_t M = dists_.size();
    for (std::size_t i = 0; i < M; ++i) {
      const auto &dd = dists_[i];
      const double w = w_[i];
      // zero this row of every requested jacobian
      if (jac)
        for (int b = 0; b < 9; ++b)
          if (jac[b]) {
            const int nb = parameter_block_sizes()[b];
            std::fill(jac[b] + i * nb, jac[b] + (i + 1) * nb, 0.0);
          }
      double half_pair = 0.0, rho_i = 0.0;
      for (double r : dd) {
        for (int k = 0; k < nphi_; ++k) {
          const double xc = eP[k] * (r - mP[k]);
          const double b = bval_d(kphi_, deg_of(dphi_, k), xc);
          half_pair += 0.5 * cP[k] * b;
          if (jac) {
            const double db = bder_d(kphi_, deg_of(dphi_, k), xc, eP[k]);
            if (jac[0]) jac[0][i * nphi_ + k] += w * 0.5 * b;
            if (jac[1]) jac[1][i * nphi_ + k] += -w * 0.5 * cP[k] * db;
            if (jac[2])
              jac[2][i * nphi_ + k] += w * 0.5 * cP[k] * (r - mP[k]) *
                                       (db / eP[k]);
          }
        }
        for (int k = 0; k < nrho_; ++k)
          rho_i += cR[k] * bval_d(krho_, deg_of(drho_, k),
                                  eR[k] * (r - mR[k]));
      }
      // embedding value + F'(rho_i)
      double Fv = 0.0, Fp = 0.0;
      for (int m = 0; m < nF_; ++m) {
        const double xc = eF[m] * (rho_i - mF[m]);
        const double b = bval_d(kF_, deg_of(dF_, m), xc);
        const double db = bder_d(kF_, deg_of(dF_, m), xc, eF[m]);
        Fv += cF[m] * b;
        Fp += cF[m] * db;
        if (jac) {
          if (jac[6]) jac[6][i * nF_ + m] = w * b;
          if (jac[7]) jac[7][i * nF_ + m] = -w * cF[m] * db;
          if (jac[8])
            jac[8][i * nF_ + m] = w * cF[m] * (rho_i - mF[m]) * (db / eF[m]);
        }
      }
      // chain rule into the density parameters
      if (jac && (jac[3] || jac[4] || jac[5])) {
        for (double r : dd)
          for (int k = 0; k < nrho_; ++k) {
            const double xc = eR[k] * (r - mR[k]);
            const double b = bval_d(krho_, deg_of(drho_, k), xc);
            const double db = bder_d(krho_, deg_of(drho_, k), xc, eR[k]);
            if (jac[3]) jac[3][i * nrho_ + k] += w * Fp * b;
            if (jac[4]) jac[4][i * nrho_ + k] += -w * Fp * cR[k] * db;
            if (jac[5])
              jac[5][i * nrho_ + k] += w * Fp * cR[k] * (r - mR[k]) *
                                       (db / eR[k]);
          }
      }
      res[i] = w * (half_pair + Fv - e0_[i]);
    }
    return true;
  }

private:
  const std::vector<std::vector<double>> &dists_;
  const std::vector<double> &e0_, &w_;
  int nphi_, nrho_, nF_;
  BasisKind kphi_, krho_, kF_;
  std::vector<int> dphi_, drho_, dF_;  // per-atom degrees (empty = all 5)
};
}  // namespace

double joint_refine(BlipFit &phi, BlipFit &rho, BlipFit &F,
                    const JointCurveData &dphi, const JointCurveData &drho,
                    const JointCurveData &dF,
                    const std::vector<std::vector<double>> &dists,
                    const std::vector<double> &e_orig,
                    const std::vector<double> &e_weight, double curve_weight,
                    int max_iter, double ftol, const std::string &algo,
                    std::string *term_out) {
  if (algo != "NONE" && algo != "CERES_LM" && algo != "CERES_DOGLEG")
    throw std::runtime_error("joint_refine: unknown algo '" + algo +
                             "' (CERES_LM|CERES_DOGLEG|NONE)");
  if (term_out) *term_out = "NONE(skipped)";
  ceres::Problem prob;
  auto add_curve = [&](BlipFit &f, const JointCurveData &d) {
    // scale curve rows by curve_weight (fold into per-sample weights)
    std::vector<double> wts =
        d.wts.empty() ? std::vector<double>(d.xs.size(), curve_weight) : d.wts;
    if (!d.wts.empty())
      for (auto &w : wts) w *= curve_weight;
    prob.AddResidualBlock(
        new CurveCost(d.xs, d.ys, std::move(wts), f.n(), f.kind, f.deg),
        nullptr, f.c.data(), f.mu.data(), f.eta.data());
  };
  add_curve(phi, dphi);
  add_curve(rho, drho);
  add_curve(F, dF);
  prob.AddResidualBlock(
      new EnergyCost(dists, e_orig, e_weight, phi.n(), rho.n(), F.n(),
                     phi.kind, rho.kind, F.kind, phi.deg, rho.deg, F.deg),
      nullptr, phi.c.data(), phi.mu.data(), phi.eta.data(), rho.c.data(),
      rho.mu.data(), rho.eta.data(), F.c.data(), F.mu.data(), F.eta.data());

  // Knot families: eta is a gauge (hold it) and the knots must stay inside
  // their box THROUGH the joint refinement too — a knot pushed past the last
  // curve sample (rcut for phi/rho) would break the smooth-vanish-at-cutoff
  // construction. Bounds derive from each function's own sample range.
  auto pin_one_sided = [&](BlipFit &f, const JointCurveData &d) {
    if (!one_sided(f.kind) || d.xs.empty()) return;
    prob.SetParameterBlockConstant(f.eta.data());
    const auto [mn, mx] = std::minmax_element(d.xs.begin(), d.xs.end());
    const double lb = knot_lb(f.kind, *mn, *mx - *mn);
    for (int k = 0; k < f.n(); ++k) {
      prob.SetParameterLowerBound(f.mu.data(), k, lb);
      prob.SetParameterUpperBound(f.mu.data(), k, *mx);
    }
  };
  pin_one_sided(phi, dphi);
  pin_one_sided(rho, drho);
  pin_one_sided(F, dF);

  if (algo != "NONE") {
    ceres::Solver::Options opt;
    opt.linear_solver_type = ceres::DENSE_QR;
    if (algo == "CERES_DOGLEG")
      opt.trust_region_strategy_type = ceres::DOGLEG;
    opt.max_num_iterations = max_iter;
    opt.function_tolerance = ftol;
    opt.gradient_tolerance = 1e-14;
    opt.parameter_tolerance = 1e-16;
    opt.logging_type = ceres::SILENT;
    ceres::Solver::Summary sum;
    ceres::Solve(opt, &prob, &sum);
    if (term_out)
      *term_out = std::string(ceres_term(sum.termination_type)) + "(" +
                  std::to_string(sum.iterations.size()) + "it, cost " + [&] {
                    char b[32];
                    std::snprintf(b, sizeof(b), "%.3g", sum.final_cost);
                    return std::string(b);
                  }() + ")";
  }

  // final max weighted |E residual|
  double mx = 0.0;
  for (std::size_t i = 0; i < dists.size(); ++i) {
    double hp = 0.0, ri = 0.0;
    for (double r : dists[i]) {
      hp += 0.5 * phi.eval(r);
      ri += rho.eval(r);
    }
    mx = std::max(mx, std::fabs(e_weight[i] * (hp + F.eval(ri) - e_orig[i])));
  }
  return mx;
}

// ---- test hooks (see header) ----
ceres::CostFunction *make_curve_cost(std::vector<double> xs,
                                     std::vector<double> ys,
                                     std::vector<double> wts, int n,
                                     BasisKind kind) {
  return new CurveCost(std::move(xs), std::move(ys), std::move(wts), n, kind);
}
ceres::CostFunction *make_energy_cost(
    const std::vector<std::vector<double>> &dists,
    const std::vector<double> &e_orig, const std::vector<double> &e_weight,
    int nphi, int nrho, int nF, BasisKind kphi, BasisKind krho,
    BasisKind kF) {
  return new EnergyCost(dists, e_orig, e_weight, nphi, nrho, nF, kphi, krho,
                        kF);
}

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_USE_CERES

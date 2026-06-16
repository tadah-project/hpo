#include <tadah/hpo/refit/compact_fitter.h>

#ifdef TADAH_USE_CERES

#include <tadah/models/descriptors/d_basis_functions.h>

#include <ceres/ceres.h>
#include <ceres/manifold.h>
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
// Ceres reports only the coarse termination_type in the enum; WHICH tolerance
// actually ended the solve (function / gradient / parameter / iteration cap) is
// only in summary.message. Distil it to a short tag so the refit log says e.g.
// CONVERGENCE(11it, ftol) instead of a bare CONVERGENCE.
std::string ceres_reason(const ceres::Solver::Summary &s) {
  const std::string &m = s.message;
  auto has = [&](const char *k) { return m.find(k) != std::string::npos; };
  if (has("Function tolerance"))           return "ftol";
  if (has("Gradient tolerance"))           return "gtol";
  if (has("Parameter tolerance"))          return "ptol";
  if (has("Maximum number of iterations")) return "maxiter";
  if (has("Minimum trust region"))         return "trust-region-collapsed";
  if (has("step is numerically zero") || has("numeric") || has("NaN") ||
      has("Inf"))                          return "numeric";
  if (m.empty())                           return "?";
  const std::size_t dot = m.find('.');  // fallback: first sentence of the message
  return dot == std::string::npos ? m : m.substr(0, dot);
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
                                     BasisKind kind, std::vector<double> dys,
                                     std::vector<double> d2ys)
    : xs_(std::move(xs)), ys_(std::move(ys)), wts_(std::move(wts)), lo_(lo),
      hi_(hi), span_(hi - lo), cfg_(cfg), kind_(kind) {
  eta0_ = (cfg_.n_min > 1) ? (cfg_.n_min - 1) / span_ : 1.0 / span_;

  // Target 1st/2nd derivatives for the curve objective + the d1/d2 diagnostics.
  // PREFERRED: caller-supplied ANALYTIC derivatives (the EAM cubic spline's
  // exact f' / f'' — no finite-difference noise), valid over the FULL range.
  // FALLBACK (no analytic derivatives, e.g. standalone callers / tests):
  // 3-point nonuniform finite differences of `ys` over the interior. Either
  // way also compute the per-block RMS normalisers that make value / d1 / d2
  // comparable; a numerically-flat block (RMS ~ 0) is excluded so its
  // normalised residual cannot explode.
  const std::size_t N = xs_.size();
  dy_.assign(N, 0.0);
  d2y_.assign(N, 0.0);
  fd_ok_.assign(N, 0);
  // Per-sample weight accessor (empty wts_ => 1 => byte-identical to the
  // unweighted path). The per-block RMS normalisers are WEIGHTED RMS so the
  // dimensionless combined score weights numerator and denominator the same
  // (both emphasise the ambient region under the pressure-relaxed profile).
  auto wt = [&](std::size_t i) { return wts_.empty() ? 1.0 : wts_[i]; };
  double sy = 0.0, sd1 = 0.0, sd2 = 0.0, wy = 0.0, wd = 0.0;
  for (std::size_t i = 0; i < N; ++i) { sy += wt(i) * ys_[i] * ys_[i]; wy += wt(i); }
  if (dys.size() == N && d2ys.size() == N) {  // analytic derivatives supplied
    dy_ = std::move(dys);
    d2y_ = std::move(d2ys);
    for (std::size_t i = 0; i < N; ++i) {
      fd_ok_[i] = 1;  // exact everywhere, including the endpoints
      sd1 += wt(i) * dy_[i] * dy_[i];
      sd2 += wt(i) * d2y_[i] * d2y_[i];
      wd += wt(i);
    }
  } else {  // finite-difference fallback (interior only)
    for (std::size_t i = 1; i + 1 < N; ++i) {
      const double hl = xs_[i] - xs_[i - 1], hr = xs_[i + 1] - xs_[i];
      if (hl <= 0.0 || hr <= 0.0) continue;
      dy_[i] = (hl * hl * ys_[i + 1] - hr * hr * ys_[i - 1] -
                (hl * hl - hr * hr) * ys_[i]) / (hl * hr * (hl + hr));
      d2y_[i] = 2.0 * (hl * ys_[i + 1] - (hl + hr) * ys_[i] + hr * ys_[i - 1]) /
                (hl * hr * (hl + hr));
      fd_ok_[i] = 1;
      sd1 += wt(i) * dy_[i] * dy_[i];
      sd2 += wt(i) * d2y_[i] * d2y_[i];
      wd += wt(i);
    }
  }
  ny_ = wy > 0.0 ? std::sqrt(sy / wy) : 1.0;
  nd1_ = wd > 0.0 ? std::sqrt(sd1 / wd) : 0.0;
  nd2_ = wd > 0.0 ? std::sqrt(sd2 / wd) : 0.0;
  // a block counts only if its target is not numerically flat relative to the
  // value scale (avoids dividing a ~0 residual by a ~0 normaliser)
  const double floor = 1e-12 * (1.0 + ny_);
  use_d1_ = nd1_ > floor;
  use_d2_ = nd2_ > floor;
  if (ny_ <= 0.0) ny_ = 1.0;
  if (!use_d1_) nd1_ = 1.0;
  if (!use_d2_) nd2_ = 1.0;
}

double CompactBlipFitter::solve_coeffs(BlipFit &f) const {
  const int n = f.n();
  const std::size_t N = xs_.size();

  if (cfg_.objective == RefitObjective::Curve) {
    // STACKED value + 1st-deriv + 2nd-deriv least squares. value, f', f'' are
    // ALL linear in c, so this is still a closed-form linear solve (VARPRO
    // holds). Each block is divided by the target's own RMS so the three are
    // dimensionless; the derivative blocks additionally carry sqrt(deriv weight)
    // so VALUE stays dominant (crystal energy depends on the curve VALUES — an
    // equal weighting lets the solver trade value accuracy away). A
    // numerically-flat block (use_dX_ false) is dropped. Basis columns: value =
    // bval_d, f' = bder_d (= eta*g'), f'' = eta^2 * bval2_d.
    const double dwr = std::sqrt(std::max(0.0, cfg_.curve_deriv_weight));
    std::size_t Nd = 0;
    for (std::size_t i = 0; i < N; ++i) Nd += fd_ok_[i];
    const bool d1on = use_d1_ && dwr > 0.0, d2on = use_d2_ && dwr > 0.0;
    const std::size_t rows = N + (d1on ? Nd : 0) + (d2on ? Nd : 0);
    Eigen::MatrixXd A(rows, n);
    Eigen::VectorXd b(rows);
    std::size_t r = 0;
    for (std::size_t i = 0; i < N; ++i) {  // value block
      const double sw = (wts_.empty() ? 1.0 : std::sqrt(wts_[i])) / ny_;
      for (int k = 0; k < n; ++k)
        A(r, k) = sw * bval_d(f.kind, f.degree(k), f.eta[k] * (xs_[i] - f.mu[k]));
      b(r) = sw * ys_[i];
      ++r;
    }
    if (d1on)
      for (std::size_t i = 0; i < N; ++i)
        if (fd_ok_[i]) {  // first-derivative block (value-dominant: x dwr)
          const double sw = dwr * (wts_.empty() ? 1.0 : std::sqrt(wts_[i])) / nd1_;
          for (int k = 0; k < n; ++k)
            A(r, k) = sw * bder_d(f.kind, f.degree(k),
                                  f.eta[k] * (xs_[i] - f.mu[k]), f.eta[k]);
          b(r) = sw * dy_[i];
          ++r;
        }
    if (d2on)
      for (std::size_t i = 0; i < N; ++i)
        if (fd_ok_[i]) {  // second-derivative block (value-dominant: x dwr)
          const double sw = dwr * (wts_.empty() ? 1.0 : std::sqrt(wts_[i])) / nd2_;
          for (int k = 0; k < n; ++k) {
            const double xc = f.eta[k] * (xs_[i] - f.mu[k]);
            A(r, k) = sw * f.eta[k] * f.eta[k] * bval2_d(f.kind, f.degree(k), xc);
          }
          b(r) = sw * d2y_[i];
          ++r;
        }
    Eigen::VectorXd c = A.completeOrthogonalDecomposition().solve(b);
    f.c.assign(c.data(), c.data() + n);
    // f.curve_rmse is the WEIGHTED VALUE rmse (sum w_i d_i^2 / sum w_i) so the
    // reported/selection value error matches the weighted fit (empty wts_ => the
    // plain mean, unchanged).
    double se = 0.0, sw = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const double w = wts_.empty() ? 1.0 : wts_[i];
      const double d = f.eval(xs_[i]) - ys_[i];
      se += w * d * d;
      sw += w;
    }
    f.curve_rmse = std::sqrt(se / (sw > 0.0 ? sw : 1.0));
    return combined_curve_score(f);  // the active objective in Curve mode
  }

  // ---- energy mode: value-only fit (unchanged, regression-gated) ----
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
  bool guard = true;  ///< apply the soft anti-coalescence penalty (knot guard)
};
// Soft anti-coalescence factor for the knot families (>= 1): a coalesced
// +-c knot pair is a derivative impostor whose VALUE rmse often beats any
// separated configuration locally, so an unpenalised derivative-free search
// walks straight into it (measured: the ladder pinned at n=1, all knots on
// the boundary). Penalise adjacent gaps inside 2% of the span — comfortably
// below genuine EAM knot spacings (Fe's tightest: 0.1 A ~ 3% of its span).
// Applied to the DERIVATIVE-FREE objective only; the Ceres refine is kept
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
         (ctx->guard ? knot_gap_factor(*ctx->work, ctx->span) : 1.0);
}
}  // namespace

// RMSE of the fit's analytic f' / f'' against the TARGET derivatives dy_ / d2y_
// (the EAM cubic spline's exact derivatives when supplied, else finite
// differences of ys_ — both precomputed in the ctor, valid where fd_ok_).
double CompactBlipFitter::d1_rmse(const BlipFit &f) const {
  double se = 0.0, sw = 0.0;  // WEIGHTED RMS (empty wts_ => plain mean)
  for (std::size_t i = 0; i < xs_.size(); ++i)
    if (fd_ok_[i]) {
      const double w = wts_.empty() ? 1.0 : wts_[i];
      const double d = f.deriv(xs_[i]) - dy_[i];
      se += w * d * d;
      sw += w;
    }
  return sw > 0.0 ? std::sqrt(se / sw) : 0.0;
}

double CompactBlipFitter::d2_rmse(const BlipFit &f) const {
  double se = 0.0, sw = 0.0;  // WEIGHTED RMS (empty wts_ => plain mean)
  for (std::size_t i = 0; i < xs_.size(); ++i)
    if (fd_ok_[i]) {
      const double w = wts_.empty() ? 1.0 : wts_[i];
      const double d = f.deriv2(xs_[i]) - d2y_[i];
      se += w * d * d;
      sw += w;
    }
  return sw > 0.0 ? std::sqrt(se / sw) : 0.0;
}

double CompactBlipFitter::combined_curve_score(const BlipFit &f) const {
  // each block's RMSE / the target's own RMS, RMS-combined over the included
  // blocks (flat blocks excluded in the ctor). The derivative blocks carry
  // weight cfg_.curve_deriv_weight relative to value so VALUE stays dominant
  // (crystal energy depends on the curve VALUES; see CompactConfig).
  const double dw = std::max(0.0, cfg_.curve_deriv_weight);
  const double cv = f.curve_rmse / ny_;
  double sumsq = cv * cv, wsum = 1.0;
  if (use_d1_) { const double c1 = d1_rmse(f) / nd1_; sumsq += dw * c1 * c1; wsum += dw; }
  if (use_d2_) { const double c2 = d2_rmse(f) / nd2_; sumsq += dw * c2 * c2; wsum += dw; }
  return std::sqrt(sumsq / wsum);
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
      // INIT-pinned knot: collapse the box to a point (NLopt/dlib honour an
      // equal lower==upper bound as a fixed variable). theta_unpack also
      // leaves pinned positions untouched, so the anchor never drifts.
      if (f.is_pinned(k)) { b.lb[k] = b.ub[k] = b.v[k] = f.mu[k]; }
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
    if (f.is_pinned(k)) {  // bump family: pin BOTH centre and width
      b.lb[k] = b.ub[k] = b.v[k] = f.mu[k];
      b.lb[n + k] = b.ub[n + k] = b.v[n + k] = f.eta[k];
    }
  }
  return b;
}
void theta_unpack(const std::vector<double> &v, BlipFit &f) {
  const int n = f.n();
  for (int k = 0; k < n; ++k)
    if (!f.is_pinned(k)) f.mu[k] = v[k];
  if (!one_sided(f.kind))
    for (int k = 0; k < n; ++k)
      if (!f.is_pinned(k)) f.eta[k] = v[n + k];
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

  // NLopt local algorithms (derivative-free + VARPRO-gradient LBFGS). Both
  // objectives HONOUR the configured inner (cfg_.explore_algo); the legal set
  // is enforced at parse time. In curve mode the pressure-relaxed weighting
  // (engine-supplied wts_) is what keeps an aggressive inner from over-fitting
  // the unprobed high-magnitude regions.
  std::string a = cfg_.explore_algo;
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
    VarproCtx ctx_df{this, &work, span_, cfg_.merge_guard};
    opt.set_min_objective(varpro_obj, &ctx_df);
    try {
      last_explore_term_ = nlopt_term(opt.optimize(b.v, fval));
    } catch (const std::exception &) {
      last_explore_term_ = nlopt_term(opt.last_optimize_result());
    }
    last_explore_term_ += "(" + std::to_string(opt.get_numevals()) + "ev)";
    theta_unpack(b.v, f);
    return solve_coeffs(f);
  }
  try {
    last_explore_term_ = nlopt_term(opt.optimize(b.v, fval));
  } catch (const std::exception &) {
    last_explore_term_ = nlopt_term(opt.last_optimize_result());
  }
  last_explore_term_ += "(" + std::to_string(opt.get_numevals()) + "ev)";
  theta_unpack(b.v, f);
  return solve_coeffs(f);
}

double CompactBlipFitter::explore(BlipFit &f) const {
  return sbplx_explore(f, cfg_.sbplx_maxeval);
}

double CompactBlipFitter::refine_dispatch(BlipFit &f) const {
  // REFINE runs the configured optimiser (cfg_.refine_algo) in BOTH objectives;
  // the legal set is enforced at parse time, and REFINE ALGO NONE (or ENABLED
  // false, mapped to NONE by the engine) is honoured as "no refine". In curve
  // mode the pressure-relaxed weighting (wts_) keeps an aggressive least-squares
  // refine from over-fitting the unprobed curve regions.
  const std::string &a = cfg_.refine_algo;
  if (a == "CERES_LM" || a == "CERES_DOGLEG") return refine(f);
  if (a == "NONE") { last_refine_term_ = "NONE(linear-solve)"; return solve_coeffs(f); }
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
        last_refine_term_ = nlopt_term(opt.optimize(b.v, fval));
      } catch (const std::exception &) {
        last_refine_term_ = nlopt_term(opt.last_optimize_result());
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
    last_refine_term_ = "DLIB_LBFGS(delta/maxeval)";
    try {
      dlib::find_min_box_constrained(
          dlib::lbfgs_search_strategy(30),
          dlib::objective_delta_stop_strategy(cfg_.ceres_ftol,
                                              50 * cfg_.ceres_max_iter),
          obj, der, x, xl, xu);
    } catch (const std::exception &) {
      last_refine_term_ = "DLIB_LBFGS(threw)";
    }
    std::vector<double> v(x.begin(), x.end());
    theta_unpack(v, f);
    return solve_coeffs(f);
  }
  throw std::runtime_error("compact_fitter: unknown refine algo '" + a +
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
  return seed_global(std::move(f));  // uniform start -> global structure search
}

// Global structure search on a GIVEN starting fit: the uniform seed above, or
// an INIT-stage seed carrying pinned anchors (fit()'s rung 0, which absorbs the
// old GLOBAL stage). The collapsed theta box holds pinned positions fixed; n is
// taken from the incoming fit.
BlipFit CompactBlipFitter::seed_global(BlipFit f) const {
  const int n = f.n();
  const std::string &a = cfg_.seed_algo;
  if (a == "UNIFORM") {  // no global step: uniform start, refine handles it
    last_seed_term_ = "UNIFORM(no-global-step)";
    solve_coeffs(f);
    return f;
  }
  ThetaBox b = theta_box(f, lo_, hi_, span_);

  // An all-pinned INIT seed has zero free dimensions (every box entry is
  // collapsed to lb==ub). A global search then just burns its whole MAXEVAL
  // budget evaluating the same point — skip it and solve the coefficients.
  bool any_free = false;
  for (std::size_t i = 0; i < b.lb.size(); ++i)
    if (b.ub[i] > b.lb[i]) { any_free = true; break; }
  if (!any_free) {
    last_seed_term_ = "ALL-PINNED(no-op)";
    solve_coeffs(f);
    return f;
  }

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
  VarproCtx ctx{this, &work, span_, cfg_.merge_guard};
  opt.set_min_objective(varpro_obj, &ctx);
  double fval = 1e300;
  try {
    last_seed_term_ = std::string(a) + ":" + nlopt_term(opt.optimize(b.v, fval));
  } catch (const std::exception &) {
    last_seed_term_ = std::string(a) + ":" + nlopt_term(opt.last_optimize_result());
  }
  last_seed_term_ += "(" + std::to_string(opt.get_numevals()) + "ev)";
  theta_unpack(b.v, f);
  solve_coeffs(f);
  return f;
}

double CompactBlipFitter::refine(BlipFit &f) const {
  const int n = f.n();
  ceres::Problem prob;
  std::vector<double> &c = f.c, &mu = f.mu, &eta = f.eta;
  prob.AddResidualBlock(new CurveCost(xs_, ys_, wts_, n, f.kind, f.deg),
                        nullptr, c.data(), mu.data(), eta.data());
  std::vector<int> pin;
  for (int k = 0; k < n; ++k)
    if (f.is_pinned(k)) pin.push_back(k);
  if (one_sided(f.kind)) {
    // knot families: eta is an exact gauge — hold it; knots stay in their box
    prob.SetParameterBlockConstant(eta.data());
    const double lb = knot_lb(f.kind, lo_, span_);
    for (int k = 0; k < n; ++k) {
      if (f.is_pinned(k)) continue;  // held constant by the manifold below
      prob.SetParameterLowerBound(mu.data(), k, lb);
      prob.SetParameterUpperBound(mu.data(), k, hi_);
    }
  } else {
    const double eta0 = std::max(1, n - 1) / span_;
    for (int k = 0; k < n; ++k) {
      if (f.is_pinned(k)) continue;  // held constant by the manifold below
      prob.SetParameterLowerBound(mu.data(), k, lo_ - span_);
      prob.SetParameterUpperBound(mu.data(), k, hi_ + span_);
      prob.SetParameterLowerBound(eta.data(), k, 0.03 * eta0);
      prob.SetParameterUpperBound(eta.data(), k, 80.0 * eta0);
    }
  }
  // INIT-pinned positions: hold them constant with a SubsetManifold (the
  // documented Ceres mechanism for fixing a subset of a parameter block).
  // An equal lower==upper bound does NOT work: Ceres' bounded trust region
  // cannot project a feasible step and returns FAILURE at iteration 0, which
  // silently turns the whole REFINE stage into a no-op (measured on every
  // pinned run). With no pins `pin` is empty and this is a no-op (the blip/knot
  // default path is byte-identical).
  if (!pin.empty()) {
    if (static_cast<int>(pin.size()) == n)
      prob.SetParameterBlockConstant(mu.data());
    else
      prob.SetManifold(mu.data(), new ceres::SubsetManifold(n, pin));
    if (!one_sided(f.kind)) {  // bump families: hold the pinned widths too
      if (static_cast<int>(pin.size()) == n)
        prob.SetParameterBlockConstant(eta.data());
      else
        prob.SetManifold(eta.data(), new ceres::SubsetManifold(n, pin));
    }
  }
  ceres::Solver::Options opt;
  opt.linear_solver_type = ceres::DENSE_QR;
  if (cfg_.refine_algo == "CERES_DOGLEG")
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
  last_refine_term_ = std::string(ceres_term(sum.termination_type)) + "(" +
                      std::to_string(sum.iterations.size()) + "it, " +
                      ceres_reason(sum) + ")";
  return solve_coeffs(f);  // final closed-form coeffs at the refined (mu,eta)
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
    if (!f.pinned.empty()) f.pinned.push_back(false);  // inserted knots are free
    return;
  }
  const double eta0 = std::max(1, f.n()) / span_;  // (n+1)-blip uniform scale
  f.mu.push_back(xw);
  f.eta.push_back(eta0);
  f.c.push_back(0.0);
  if (!f.pinned.empty()) f.pinned.push_back(false);  // inserted blips are free
}

void CompactBlipFitter::jitter(BlipFit &f, std::mt19937 &rng) const {
  std::normal_distribution<double> g(0.0, 1.0);
  if (one_sided(f.kind)) {  // knot jitter only; eta is pinned (gauge)
    const double lb = knot_lb(f.kind, lo_, span_);
    for (int k = 0; k < f.n(); ++k) {
      if (f.is_pinned(k)) continue;  // INIT-pinned anchors stay put
      f.mu[k] = std::clamp(f.mu[k] + 0.02 * span_ * g(rng), lb, hi_);
    }
    return;
  }
  const double eta0 = std::max(1, f.n() - 1) / span_;
  for (int k = 0; k < f.n(); ++k) {
    if (f.is_pinned(k)) continue;  // INIT-pinned anchors stay put
    f.mu[k] = std::clamp(f.mu[k] + 0.02 * span_ * g(rng), lo_ - span_,
                         hi_ + span_);
    f.eta[k] = std::clamp(f.eta[k] * std::exp(0.2 * g(rng)), 0.03 * eta0,
                          80.0 * eta0);
  }
}

int CompactBlipFitter::merge_close_knots(BlipFit &f) const {
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
  // Returns the number of knots removed by merging (0 = nothing merged). The
  // caller (run_cand) collects this so the per-rung log can report the winning
  // candidate's merge count ONCE, instead of this routine logging a line every
  // time it fires — it is invoked once per jittered candidate (jitters+1 times
  // per rung), which produced the puzzling run of identical "merged N knot(s)"
  // lines. REFIT_COMPACT_MERGE=false disables the guard entirely.
  if (!cfg_.merge_guard || !one_sided(f.kind) || f.n() < 2) return 0;
  const int n_orig = f.n();
  const bool had_pins = !f.pinned.empty();
  const double gap = 1e-3 * span_;
  std::vector<int> idx(f.mu.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return f.mu[a] < f.mu[b]; });
  // merge only SAME-degree neighbours: a cubic and a quintic atom at the
  // same knot are linearly independent, not a coalesced impostor pair. A
  // PINNED knot (INIT anchor) is never merged or moved.
  std::vector<double> mu_new;
  std::vector<int> deg_new;
  std::vector<bool> pin_new;
  bool merged = false;
  for (std::size_t s = 0; s < idx.size(); ++s) {
    const double m = f.mu[idx[s]];
    const int d = f.degree(idx[s]);
    const bool pinned = f.is_pinned(idx[s]);
    if (!mu_new.empty() && m - mu_new.back() < gap && d == deg_new.back() &&
        !pinned && !pin_new.back()) {
      mu_new.back() = 0.5 * (mu_new.back() + m);  // midpoint merge
      merged = true;
    } else {
      mu_new.push_back(m);
      deg_new.push_back(d);
      pin_new.push_back(pinned);
    }
  }
  if (!merged) return 0;
  const int n_merged = n_orig - static_cast<int>(mu_new.size());
  const bool any_cubic =
      std::any_of(deg_new.begin(), deg_new.end(), [](int d) { return d != 5; });
  f.mu = std::move(mu_new);
  f.eta.assign(f.mu.size(), 1.0);
  f.c.assign(f.mu.size(), 0.0);
  f.deg = any_cubic ? deg_new : std::vector<int>{};
  f.pinned = had_pins ? pin_new : std::vector<bool>{};
  solve_coeffs(f);
  while (f.n() < n_orig) {  // restore capacity where the residual lives
    insert_blip_at_worst(f);
    solve_coeffs(f);
  }
  return n_merged;
}

BlipFit CompactBlipFitter::fit(const Metric &metric,
                               std::vector<LadderRow> &trace,
                               const BlipFit *init) {
  using clock = std::chrono::steady_clock;
  std::mt19937 rng(cfg_.seed);
  nlopt::srand(cfg_.seed);  // MLSL determinism (global NLopt RNG)

  // The DECISION objective (candidate selection, stall, stop, insertion peak):
  // the caller's energy metric, or — in RefitObjective::Curve — the normalised
  // combined value+d1+d2 curve score (the energy metric is then unused). All
  // `metric(...)` decision sites below route through `score(...)`.
  const bool curve_obj = (cfg_.objective == RefitObjective::Curve);
  auto score = [&](const BlipFit &f) {
    return curve_obj ? combined_curve_score(f) : metric(f);
  };
  // Stop target: the energy metric is normalised so target=1 == within
  // tolerance; the curve score is an open-ended relative error with no such
  // threshold, so in Curve mode the ladder grows until stall / n_max (target 0,
  // never satisfied as the score is > 0).
  const double target = curve_obj ? 0.0 : cfg_.target;

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
    refine_dispatch(f);
    merge_close_knots(f);  // knot families only; no-op for blips
    f.metric = score(f);
  };

  // Per-rung stage attribution: the winning candidate's energy metric and curve
  // RMSE at each checkpoint — after insertion, after EXPLORE, after REFINE (the
  // Ceres curve refine), and after the coalescence-guard MERGE (knot families
  // only). REFINE and MERGE are reported SEPARATELY: REFINE is a least-squares
  // descent so it cannot raise the curve RMSE, whereas MERGE deliberately
  // breaks up coalesced "divided-difference impostor" knot pairs (low value-RMSE
  // but spiky f'' / rank-collapsing) and re-inserts, which CAN raise the
  // value-RMSE — so a large positive "merge" delta is the guard at work, not the
  // optimiser failing.
  // m_=energy metric, r_=value-curve RMSE, d1_=first-derivative (slope) RMSE,
  // d2_=second-derivative (curvature) RMSE, each captured at four checkpoints:
  // _ins (after insertion+linear solve), _exp (after EXPLORE), _pol (after
  // REFINE), _ref (after the coalescence-guard MERGE). merges = knots the guard
  // removed on the winning candidate (0 if it did not fire / is disabled).
  struct StageTrace {
    double m_ins = 1e300, r_ins = 1e300, m_exp = 1e300, r_exp = 1e300,
           m_pol = 1e300, r_pol = 1e300, m_ref = 1e300, r_ref = 1e300;
    double d1_ins = 0, d1_exp = 0, d1_pol = 0, d1_ref = 0;
    double d2_ins = 0, d2_exp = 0, d2_pol = 0, d2_ref = 0;
    int merges = 0;
    std::string explore_term = "-", refine_term = "-";
  };
  // Run one EXPLORE -> REFINE -> MERGE candidate, capturing its per-stage trace.
  // This inlines stamp()'s refine + merge + metric so the merge can be measured
  // separately; the numerics (and the post-merge selection metric) are identical
  // to stamp(). The extra metric() calls are read-only (logging only; they do
  // not perturb the RNG or the regression-gated result).
  auto run_cand = [&](BlipFit cand, double m_ins, double r_ins, double d1_ins,
                      double d2_ins, StageTrace &st) {
    st.m_ins = m_ins;
    st.r_ins = r_ins;
    st.d1_ins = d1_ins;
    st.d2_ins = d2_ins;
    explore(cand);
    st.r_exp = cand.curve_rmse;  // EXPLORE re-solved the linear coefficients
    st.m_exp = score(cand);
    st.d1_exp = d1_rmse(cand);
    st.d2_exp = d2_rmse(cand);
    st.explore_term = last_explore_term_;
    refine_dispatch(cand);  // REFINE: curve refine
    st.r_pol = cand.curve_rmse;
    st.m_pol = score(cand);
    st.d1_pol = d1_rmse(cand);
    st.d2_pol = d2_rmse(cand);
    st.refine_term = last_refine_term_;
    // coalescence guard (one-sided knot families only); merge count drives the
    // per-rung "merged=N" report so this no longer logs a line per candidate.
    st.merges = merge_close_knots(cand);
    cand.metric = score(cand);
    st.r_ref = cand.curve_rmse;
    st.m_ref = cand.metric;
    st.d1_ref = d1_rmse(cand);
    st.d2_ref = d2_rmse(cand);
    return cand;
  };
  // Signed relative change a->b as a percent (negative = improvement).
  auto pct = [](double from, double to) {
    if (!(std::fabs(from) > 0) || !std::isfinite(from) || !std::isfinite(to))
      return std::string("n/a");
    char b[16];
    std::snprintf(b, sizeof(b), "%+.0f%%", 100.0 * (to - from) / from);
    return std::string(b);
  };
  // Checkpoint progressions A->B->C->D (knot families) / A->B->C (bump families).
  auto prog4 = [&](double a, double b, double c, double d) {
    return fmt(a) + "->" + fmt(b) + "->" + fmt(c) + "->" + fmt(d);
  };
  auto prog3 = [&](double a, double b, double c) {
    return fmt(a) + "->" + fmt(b) + "->" + fmt(c);
  };

  // Rung 0: optimise the INIT seed (if any) with EXPLORE's global optimiser —
  // this absorbs the old GLOBAL stage. No INIT => the legacy uniform seed.
  BlipFit best = init ? seed_global(*init) : seed_mlsl();
  stamp(best);
  {
    LadderRow r;
    r.n = best.n(); r.curve_rmse = best.curve_rmse; r.metric = best.metric;
    r.kept = true;
    trace.push_back(r);
  }
  // EXPLORE rung 0 = the global seed; report which convergence criterion it hit
  // so the user can tweak the EXPLORE.seed optimiser (e.g. MAXEVAL_REACHED =>
  // raise its MAXEVAL).
  log("EXPLORE rung 0 seed(" + cfg_.seed_algo + "): " + describe(best) +
      "  term=" + last_seed_term_ + " refine=" + last_refine_term_ +
      (curve_obj
           ? "  [objective=curve (value+d1+d2); grows to stall/n_max]"
           : "  [metric <= " + fmt(target) + " stops the ladder]"));

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
        q.metric = score(q);
        insert_at_deg(cb, xw, 3);
        solve_coeffs(cb);
        cb.metric = score(cb);
        site = (cb.metric < q.metric && d2_rmse(cb) <= d2_rmse(q)) ? cb : q;
      } else {
        site = from;
        insert_at_deg(site, xw, cfg_.knot_degree);
        solve_coeffs(site);
        site.metric = score(site);
      }
      if (site.metric < out.metric) out = site;
    }
    if (out.n() == 0) {  // no peaks (degenerate): legacy single insertion
      out = from;
      insert_blip_at_worst(out);
      solve_coeffs(out);
      out.metric = score(out);
    }
    return out;
  };

  BlipFit warm = best;  // ladder continues from the latest rung winner, the
                        // ENVELOPE keeps the metric-best across all rungs
  int stall = 0;
  int rung_idx = 0;
  const int opt_every =
      one_sided(kind_) ? std::max(1, cfg_.opt_every) : 1;  // blip: legacy
  while (warm.n() < cfg_.n_max && best.metric > target &&
         stall < cfg_.stall_rungs) {
    const auto t0 = clock::now();
    ++rung_idx;
    const bool opt_rung = (opt_every == 1) || (rung_idx % opt_every == 0);

    BlipFit rung_best;
    StageTrace won;
    bool grow_only = false;
    // Per-rung coalescence-guard activity across ALL candidates (not just the
    // winner): how many of the rung's candidates the guard fired on and the
    // most knots it merged on any one of them. This is what produced the run of
    // identical "merged N knot(s)" lines in the old log — the guard runs once
    // per candidate (jitters+1 times) and several losing candidates can merge
    // while the winner does not. Reported once, here, instead of per candidate.
    int rung_merge_cands = 0, rung_merge_max = 0;
    if (one_sided(kind_)) {
      BlipFit base = insert_best(warm);
      if (!opt_rung) {
        // cheap growth rung: insertion + linear solve only (pilot-measured
        // strong baseline, ~300x cheaper than per-rung optimisation)
        merge_close_knots(base);
        base.metric = score(base);
        rung_best = base;
        grow_only = true;
      } else {
        rung_best = base;  // the un-optimised insertion is a candidate too:
                           // the optimiser sometimes lands in a worse basin
        const double d1_base = d1_rmse(base), d2_base = d2_rmse(base);
        won.m_ins = won.m_exp = won.m_pol = won.m_ref = base.metric;
        won.r_ins = won.r_exp = won.r_pol = won.r_ref = base.curve_rmse;
        won.d1_ins = won.d1_exp = won.d1_pol = won.d1_ref = d1_base;
        won.d2_ins = won.d2_exp = won.d2_pol = won.d2_ref = d2_base;
        won.explore_term = won.refine_term = "(insert only)";
        for (int j = 0; j <= cfg_.jitters; ++j) {
          BlipFit cand = base;
          if (j > 0) jitter(cand, rng);
          StageTrace st;
          cand = run_cand(cand, base.metric, base.curve_rmse, d1_base, d2_base,
                          st);
          if (st.merges > 0) {
            ++rung_merge_cands;
            rung_merge_max = std::max(rung_merge_max, st.merges);
          }
          if (cand.metric < rung_best.metric) { rung_best = cand; won = st; }
        }
      }
    } else {
      // ---- legacy blip rung (bit-for-bit regression-gated) ----
      BlipFit base = warm;
      insert_blip_at_worst(base);
      // insert reference for stage attribution (logging only; on a copy so the
      // blip path's numerics and RNG stream are untouched)
      BlipFit ref = base;
      solve_coeffs(ref);
      const double m_ins = score(ref), r_ins = ref.curve_rmse;
      const double d1_ins = d1_rmse(ref), d2_ins = d2_rmse(ref);
      for (int j = 0; j <= cfg_.jitters; ++j) {
        BlipFit cand = base;
        if (j > 0) jitter(cand, rng);
        StageTrace st;
        cand = run_cand(cand, m_ins, r_ins, d1_ins, d2_ins, st);
        if (cand.metric < rung_best.metric) { rung_best = cand; won = st; }
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
    // Stage attribution: the winning candidate's energy metric and curve RMSE
    // after each stage (insert -> EXPLORE -> REFINE) with the per-stage change,
    // so it is obvious which stage improved the fit (negative % = improvement).
    // The bracketed terms are that candidate's EXPLORE / REFINE stop reasons.
    // rmse = value-curve RMSE; d1 = first-derivative (slope) RMSE; d2 =
    // second-derivative (curvature) RMSE — all vs the tabulated EAM. metric is
    // the energy gate that drives selection; rmse/d1/d2 are diagnostics only.
    std::string detail;
    if (grow_only) {
      detail = "  (grow: insertion+solve, no optimiser)  rmse=" +
               fmt(rung_best.curve_rmse) + " d1=" + fmt(d1_rmse(rung_best)) +
               " d2=" + fmt(d2_rmse(rung_best));
    } else if (one_sided(kind_)) {
      // knot families: report the MERGE coalescence-guard stage separately from
      // REFINE so a merge-induced RMSE rise is not blamed on the Ceres refine.
      detail =
          "  metric " + prog4(won.m_ins, won.m_exp, won.m_pol, won.m_ref) +
          " (explore " + pct(won.m_ins, won.m_exp) + ", refine " +
          pct(won.m_exp, won.m_pol) + ", merge " + pct(won.m_pol, won.m_ref) +
          ")  rmse " + prog4(won.r_ins, won.r_exp, won.r_pol, won.r_ref) +
          " (explore " + pct(won.r_ins, won.r_exp) + ", refine " +
          pct(won.r_exp, won.r_pol) + ", merge " + pct(won.r_pol, won.r_ref) +
          ")  d1 " + prog4(won.d1_ins, won.d1_exp, won.d1_pol, won.d1_ref) +
          "  d2 " + prog4(won.d2_ins, won.d2_exp, won.d2_pol, won.d2_ref) +
          (rung_merge_cands > 0
               ? "  guard merged " + std::to_string(rung_merge_cands) + "/" +
                     std::to_string(cfg_.jitters + 1) + " cands (<=" +
                     std::to_string(rung_merge_max) + " knot(s))"
               : "") +
          "  [explore=" + won.explore_term + " refine=" + won.refine_term + "]";
    } else {
      // bump families: MERGE is a no-op, so REFINE is the full post-explore step
      detail = "  metric " + prog3(won.m_ins, won.m_exp, won.m_ref) +
               " (explore " + pct(won.m_ins, won.m_exp) + ", refine " +
               pct(won.m_exp, won.m_ref) + ")  rmse " +
               prog3(won.r_ins, won.r_exp, won.r_ref) + " (explore " +
               pct(won.r_ins, won.r_exp) + ", refine " +
               pct(won.r_exp, won.r_ref) + ")  d1 " +
               prog3(won.d1_ins, won.d1_exp, won.d1_ref) + "  d2 " +
               prog3(won.d2_ins, won.d2_exp, won.d2_ref) + "  [explore=" +
               won.explore_term + " refine=" + won.refine_term + "]";
    }
    log("rung " + std::to_string(rung_idx) +
        (opt_rung ? " (opt): " : " (grow): ") + "n=" +
        std::to_string(rung_best.n()) +
        (row.kept ? "  KEPT"
                  : "  stall=" + std::to_string(stall) + "/" +
                        std::to_string(cfg_.stall_rungs)) +
        "  (" + fmt(row.seconds) + "s)" + detail);
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
      if (f.is_pinned(k)) continue;  // held constant by the manifold below
      prob.SetParameterLowerBound(f.mu.data(), k, lb);
      prob.SetParameterUpperBound(f.mu.data(), k, *mx);
    }
  };
  pin_one_sided(phi, dphi);
  pin_one_sided(rho, drho);
  pin_one_sided(F, dF);

  // INIT-pinned positions: hold them constant with a SubsetManifold. (An equal
  // lower==upper bound makes Ceres' bounded trust region return FAILURE at
  // iteration 0 — the FUSE step would silently do nothing.) No pins => no-op.
  auto hold_pins = [&](BlipFit &f) {
    const int n = f.n();
    std::vector<int> pin;
    for (int k = 0; k < n; ++k)
      if (f.is_pinned(k)) pin.push_back(k);
    if (pin.empty()) return;
    if (static_cast<int>(pin.size()) == n)
      prob.SetParameterBlockConstant(f.mu.data());
    else
      prob.SetManifold(f.mu.data(), new ceres::SubsetManifold(n, pin));
    if (!one_sided(f.kind)) {  // bump families: hold the pinned widths too
      if (static_cast<int>(pin.size()) == n)
        prob.SetParameterBlockConstant(f.eta.data());
      else
        prob.SetManifold(f.eta.data(), new ceres::SubsetManifold(n, pin));
    }
  };
  hold_pins(phi);
  hold_pins(rho);
  hold_pins(F);

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
                  std::to_string(sum.iterations.size()) + "it, " +
                  ceres_reason(sum) + ", cost " + [&] {
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

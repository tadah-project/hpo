#include <tadah/hpo/refit/compact_fitter.h>

#ifdef TADAH_USE_CERES

#include <tadah/hpo/refit/refit_parser.h>

#include <tadah/models/descriptors/basis_kind.h>
#include <tadah/models/descriptors/d_basis_functions.h>
#include <tadah/core/logger.h>
#include <tadah/models/nnls.h>

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {
namespace refit {

using tadah::models::B;
using tadah::models::dB;
using tadah::models::RhoForm;
using tadah::models::rhoform_is_positive;
using tadah::models::rhoform_name;

void apply_stage_optimizers(const RefitConfig &rc, CompactConfig &cfg) {
  // EXPLORE outer = the global seed at rung 0 (absorbs the old GLOBAL stage);
  // EXPLORE INNER = the per-rung local search. MAXEVAL is the seed's budget
  // whichever algorithm runs it.
  if (rc.explore.has_opt) {
    const OptimizerSpec &o = rc.explore.opt;
    cfg.seed_algo = refit_explore_outer_algo(o);
    if (o.stop.maxeval) {
      cfg.mlsl_maxeval = static_cast<int>(*o.stop.maxeval);
      cfg.dlib_global_maxeval = static_cast<int>(*o.stop.maxeval);
    }
    if (o.inner) {
      const OptimizerSpec &in = *o.inner;
      cfg.explore_algo = refit_explore_inner_algo(in);
      if (in.stop.maxeval) cfg.sbplx_maxeval = static_cast<int>(*in.stop.maxeval);
      if (in.stop.ftol_rel) cfg.sbplx_ftol = *in.stop.ftol_rel;
      if (in.stop.xtol_rel) cfg.sbplx_xtol = *in.stop.xtol_rel;
      if (in.stop.ftol_abs) cfg.dlib_inner_ftol_abs = *in.stop.ftol_abs;
    }
  }
  // ceres_ftol / joint_ftol feed Ceres' function_tolerance, which is RELATIVE
  // ((new_cost - old_cost) < ftol * old_cost) — Ceres has no absolute function
  // tolerance, so FTOL_REL is the only accepted key (FTOL / FTOL_ABS are
  // rejected by RefitParser). REFINE's MAXEVAL/MAXITER is the budget of the
  // algorithm that runs: the L-BFGS algorithms' evaluations, or Ceres'
  // iterations. FUSE's settings are the joint solve's and, under
  // REFIT_OBJECTIVE curve, each function's FUSE refine's (fuse_curve_config).
  if (rc.refine.has_opt) {
    const OptimizerSpec &o = rc.refine.opt;
    cfg.refine_algo = refit_refine_algo(o);
    if (o.stop.maxeval) {
      if (cfg.refine_algo == "NLOPT_LBFGS" || cfg.refine_algo == "DLIB_LBFGS")
        cfg.lbfgs_refine_maxeval = static_cast<int>(*o.stop.maxeval);
      else
        cfg.ceres_max_iter = static_cast<int>(*o.stop.maxeval);
    }
    if (o.stop.ftol_rel) cfg.ceres_ftol = *o.stop.ftol_rel;
    if (o.stop.ftol_abs) cfg.dlib_refine_ftol_abs = *o.stop.ftol_abs;
    auto it = o.param.find("GTOL");
    if (it != o.param.end()) cfg.ceres_gtol = it->second;
  }
  if (rc.fuse.has_opt) {
    const OptimizerSpec &o = rc.fuse.opt;
    cfg.joint_algo = refit_fuse_algo(o);
    if (o.stop.maxeval) cfg.joint_max_iter = static_cast<int>(*o.stop.maxeval);
    if (o.stop.ftol_rel) cfg.joint_ftol = *o.stop.ftol_rel;
    auto it = o.param.find("GTOL");
    if (it != o.param.end()) cfg.joint_gtol = it->second;
  }
  // ENABLED false switches the stage off (RefitParser refuses it beside an
  // OPTIMIZER block).
  if (!rc.refine.enabled) cfg.refine_algo = "NONE";
  if (!rc.fuse.enabled) cfg.joint_algo = "NONE";
}

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

// Ceres checks its options in Solve() and, for one it refuses (a negative
// function or gradient tolerance), returns FAILURE without iterating: taken
// for a stop term, the stage silently did nothing. RefitParser refuses such
// settings (every tolerance >= 0), so they come only from a caller that set
// CompactConfig itself: an error, raised before the solve.
void require_valid_options(const ceres::Solver::Options &opt,
                           const std::string &where) {
  std::string why;
  if (!opt.IsValid(&why))
    throw std::invalid_argument(where + ": Ceres refuses the solver options (" +
                                why + ")");
}
}  // namespace

// ---- basis dispatch: value g(x_c) and r-derivative eta*g'(x_c) per family.
// A 3-way switch on a hot member is branch-predicted perfectly; the blip
// path calls the exact same functions as before (bit-for-bit regression
// gated against a fixed-seed golden refit — with cfg.hard_rcut == 0, the
// standalone default; the engine's radial ladders set the bump-support gate,
// which caps boxes and projects candidates).
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
// Support half-width of the bump families (in units of 1/eta) from the shared
// MODELS trait — the fitter's private BasisKind enum maps onto it here. Only
// the symmetric bumps reach this (one-sided kinds are gated out by callers).
inline double bump_w(BasisKind k) {
  return tadah::models::bump_halfwidth(k == BasisKind::Blip5
                                           ? tadah::models::BasisKind::Blip5
                                           : tadah::models::BasisKind::Blip);
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
// ---- RHOFORM MONOTONE step kinds: the integrated bump S of the bump
// families, from the MODELS step primitives (the ones DM_REAM and
// EAMRefit::rho_fit evaluate). With x_c = eta (x - mu) and A the bump area in
// x_c (1.5 Blip, 20/11 Blip5):
//   value S(x_c);  dS/dr = -eta B(x_c)/A;  d2S/dr2 = -eta^2 B'(x_c)/A
// and, since S depends on (mu, eta) only through x_c, the Jacobians follow
// from dS/dr exactly as for the bumps: dS/dmu = -dS/dr = eta B(x_c)/A and
// dS/deta = (x - mu) (dS/dr)/eta = -(x - mu) B(x_c)/A. The one-sided knot
// families have no step form (the fitter refuses them under MONOTONE).
[[noreturn]] void no_step_kind(BasisKind k) {
  throw std::runtime_error(
      "compact_fitter: RHOFORM MONOTONE has no step form for the one-sided "
      "knot kind " + std::to_string(static_cast<int>(k)) +
      " (bump kinds Blip and Blip5 only)");
}
inline double sval(BasisKind k, double xc) {
  switch (k) {
    case BasisKind::Blip:   return tadah::models::SB(xc);
    case BasisKind::Blip5:  return tadah::models::SB5(xc);
    case BasisKind::Knot5L:
    case BasisKind::Knot5R: break;
  }
  no_step_kind(k);
}
inline double sder(BasisKind k, double xc, double eta) {  // dS/dr
  switch (k) {
    case BasisKind::Blip:   return tadah::models::dSB(xc, eta);
    case BasisKind::Blip5:  return tadah::models::dSB5(xc, eta);
    case BasisKind::Knot5L:
    case BasisKind::Knot5R: break;
  }
  no_step_kind(k);
}
inline double sder2(BasisKind k, double xc, double eta) {  // d2S/dr2
  switch (k) {
    case BasisKind::Blip:   return tadah::models::d2SB(xc, eta);
    case BasisKind::Blip5:  return tadah::models::d2SB5(xc, eta);
    case BasisKind::Knot5L:
    case BasisKind::Knot5R: break;
  }
  no_step_kind(k);
}
// Form-aware column: the step kinds under MONOTONE, the family itself
// otherwise (SIGNED, NONNEG and every phi / F fit — the same call, so those
// stay bit-for-bit the values before RHOFORM). cval = value, cder =
// r-derivative.
inline double cval(BasisKind k, RhoForm form, int d, double xc) {
  return form == RhoForm::Monotone ? sval(k, xc) : bval_d(k, d, xc);
}
inline double cder(BasisKind k, RhoForm form, int d, double xc, double eta) {
  return form == RhoForm::Monotone ? sder(k, xc, eta) : bder_d(k, d, xc, eta);
}
// RHOFORM NONNEG / MONOTONE coefficients: min ||A c - b|| over c >= 0 by
// Lawson-Hanson (models::nnls, Householder QR on the passive columns — never
// the normal equations: step columns reach cond ~1e9). Eigen stores A
// column-major, the layout nnls() takes; the row weights are already folded
// into A and b. Numerically zero columns (a basis pushed off the samples)
// stay at 0, which is what the rank-revealing solve does for SIGNED. A
// non-finite system (which the SIGNED solve turns into NaN coefficients)
// gives NaN coefficients here too, so the candidate loses every comparison.
// A solve that stopped before a Kuhn-Tucker point (nnls' iteration cap, or
// every column with a dual above the tolerance refused) is feasible (every
// c_k >= 0) but not proven optimal: it returns the nnls result, whose
// `converged` the caller counts and reports (CompactBlipFitter::fit).
tadah::models::NnlsResult solve_nonneg(const Eigen::MatrixXd &A,
                                       const Eigen::VectorXd &b,
                                       std::vector<double> &c) {
  c.assign(static_cast<std::size_t>(A.cols()), 0.0);
  try {
    return tadah::models::nnls(A.data(), static_cast<std::size_t>(A.rows()),
                               static_cast<std::size_t>(A.cols()), b.data(),
                               c.data());
  } catch (const std::invalid_argument &) {
    // a non-finite system: NaN coefficients (as the SIGNED solve gives), a
    // failed candidate rather than a non-converged solve
    c.assign(c.size(), std::numeric_limits<double>::quiet_NaN());
    return tadah::models::NnlsResult{};
  }
}
// EMBF0 ZERO reference terms of an n-atom expansion at x = 0: b0[k] =
// g(x_c0) and db0[k] = eta_k g'(x_c0) with x_c0 = eta_k (0 - mu_k) — the same
// primitives (and the same x_c expression with x = 0.0) as the x-dependent
// terms, so the zero-referenced column g(x_c) - b0 is exactly 0 at x = 0.
// Parameter-only constants: callers compute them once per evaluation, and
// ONLY under zero_ref (the FREE paths never evaluate the basis at 0).
// Knot5R with mu >= 0 gives exactly 0; only bumps covering 0 and negative
// knots contribute.
inline void ref_terms(BasisKind kind, int n, const double *mu,
                      const double *eta, const std::vector<int> &deg,
                      std::vector<double> &b0, std::vector<double> &db0) {
  b0.resize(n);
  db0.resize(n);
  for (int k = 0; k < n; ++k) {
    const int d = (k < static_cast<int>(deg.size())) ? deg[k] : 5;
    const double xc0 = eta[k] * (0.0 - mu[k]);
    b0[k] = bval_d(kind, d, xc0);
    db0[k] = bder_d(kind, d, xc0, eta[k]);
  }
}
inline void ref_terms(const BlipFit &f, std::vector<double> &b0,
                      std::vector<double> &db0) {
  ref_terms(f.kind, f.n(), f.mu.data(), f.eta.data(), f.deg, b0, db0);
}
}  // namespace

// ---------------------------------------------------------------- BlipFit
double BlipFit::eval(double x) const {
  if (zero_ref) {
    // EMBF0 ZERO: f(x) = sum_k c_k (g(x_c) - g(x_c0)). F(0) = 0 exactly by
    // construction — returned explicitly, independent of FP contraction.
    if (x == 0.0) return 0.0;
    double acc = 0.0;
    for (std::size_t k = 0; k < mu.size(); ++k) {
      const int d = degree(static_cast<int>(k));
      acc += c[k] * (bval_d(kind, d, eta[k] * (x - mu[k])) -
                     bval_d(kind, d, eta[k] * (0.0 - mu[k])));
    }
    return acc;
  }
  double acc = 0.0;
  for (std::size_t k = 0; k < mu.size(); ++k)
    acc += c[k] * cval(kind, form, degree(static_cast<int>(k)),
                       eta[k] * (x - mu[k]));
  return acc;
}

double BlipFit::deriv(double x) const {
  double acc = 0.0;
  for (std::size_t k = 0; k < mu.size(); ++k)
    acc += c[k] * cder(kind, form, degree(static_cast<int>(k)),
                       eta[k] * (x - mu[k]), eta[k]);
  return acc;
}

double BlipFit::deriv2(double x) const {
  double acc = 0.0;
  if (form == RhoForm::Monotone) {  // step kinds: sder2 carries the eta^2
    for (std::size_t k = 0; k < mu.size(); ++k)
      acc += c[k] * sder2(kind, eta[k] * (x - mu[k]), eta[k]);
    return acc;
  }
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
// zero_ref (EMBF0 ZERO, F channel): the model is the zero-referenced
// sum_k c_k (B(x_c) - B(x_c0)), x_c0 = eta_k (0 - mu_k) — evaluate_zero_ref.
// form (RHOFORM, rho channel): MONOTONE evaluates the step kinds S (cval /
// cder), with the same Jacobian expressions in S and dS/dr; SIGNED and
// NONNEG share the bump columns (NONNEG's c >= 0 is a Ceres bound).
class CurveCost : public ceres::CostFunction {
public:
  CurveCost(std::vector<double> xs, std::vector<double> ys,
            std::vector<double> wts, int n, BasisKind kind = BasisKind::Blip,
            std::vector<int> deg = {}, bool zero_ref = false,
            RhoForm form = RhoForm::Signed)
      : xs_(std::move(xs)), ys_(std::move(ys)), wts_(std::move(wts)), n_(n),
        kind_(kind), deg_(std::move(deg)), zero_ref_(zero_ref), form_(form) {
    if (zero_ref_ && form_ != RhoForm::Signed)
      throw std::logic_error("compact_fitter: a density form (rho) and the "
                             "EMBF0 zero reference (F) cannot share a fit");
    if (form_ == RhoForm::Monotone && one_sided(kind_))
      no_step_kind(kind_);  // refused here, never inside a Ceres Evaluate
    set_num_residuals(static_cast<int>(xs_.size()));
    mutable_parameter_block_sizes()->assign(3, n_);  // c, mu, eta
  }

  bool Evaluate(double const *const *p, double *residuals,
                double **jacobians) const override {
    if (zero_ref_) return evaluate_zero_ref(p, residuals, jacobians);
    const double *c = p[0], *mu = p[1], *eta = p[2];
    const std::size_t N = xs_.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double x = xs_[i];
      const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
      double m = 0.0;
      for (int k = 0; k < n_; ++k) {
        const int d = (k < static_cast<int>(deg_.size())) ? deg_[k] : 5;
        const double xc = eta[k] * (x - mu[k]);
        const double b = cval(kind_, form_, d, xc);
        m += c[k] * b;
        if (jacobians) {
          // = eta * g'(xc); for the steps dS/dr = -eta B(xc)/A
          const double db = cder(kind_, form_, d, xc, eta[k]);
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
  // EMBF0 ZERO twin of Evaluate. With b0 = B(x_c0), db0 = eta B'(x_c0) (the
  // parameter-only reference terms, once per call):
  //   d/dc   = sw (b - b0)
  //   d/dmu  = -sw c (db - db0)                      (dx_c/dmu = -eta)
  //   d/deta = sw c ((x - mu) db - (0 - mu) db0) / eta  (dx_c/deta = x - mu)
  // At x == 0 the column is exactly 0 (b == b0 bitwise), matching
  // BlipFit::eval's explicit F(0) = 0.
  bool evaluate_zero_ref(double const *const *p, double *residuals,
                         double **jacobians) const {
    const double *c = p[0], *mu = p[1], *eta = p[2];
    std::vector<double> b0, db0;
    ref_terms(kind_, n_, mu, eta, deg_, b0, db0);
    const std::size_t N = xs_.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double x = xs_[i];
      const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
      double m = 0.0;
      for (int k = 0; k < n_; ++k) {
        const int d = (k < static_cast<int>(deg_.size())) ? deg_[k] : 5;
        const double xc = eta[k] * (x - mu[k]);
        const double b = bval_d(kind_, d, xc) - b0[k];
        m += c[k] * b;
        if (jacobians) {
          const double db = bder_d(kind_, d, xc, eta[k]);  // = eta * g'(xc)
          if (jacobians[0]) jacobians[0][i * n_ + k] = sw * b;
          if (jacobians[1])
            jacobians[1][i * n_ + k] = -sw * c[k] * (db - db0[k]);
          if (jacobians[2])
            jacobians[2][i * n_ + k] =
                sw * c[k] *
                ((x - mu[k]) * (db / eta[k]) - (0.0 - mu[k]) * (db0[k] / eta[k]));
        }
      }
      residuals[i] = sw * (m - ys_[i]);
    }
    return true;
  }

  // by-value copies: the cost may outlive the caller's buffers (joint refine)
  std::vector<double> xs_, ys_, wts_;
  int n_;
  BasisKind kind_;
  std::vector<int> deg_;  // per-atom degree (knot kinds; empty = all 5)
  bool zero_ref_ = false;  // EMBF0 ZERO zero-referenced basis (F channel)
  RhoForm form_ = RhoForm::Signed;  // RHOFORM (rho channel)
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
  // RHOFORM is a density-channel property: the step kinds exist for the bump
  // families only, and the EMBF0 zero reference belongs to the F channel.
  if (cfg_.rho_form == RhoForm::Monotone && one_sided(kind_))
    throw std::runtime_error(
        std::string("compact_fitter: RHOFORM MONOTONE is not defined for the "
                    "one-sided knot families (use a bump basis, Blip or "
                    "Blip5, or RHOFORM NONNEG: ") +
        tadah::models::monotone_knot_refusal_text() + ")");
  if (cfg_.rho_form != RhoForm::Signed && cfg_.zero_ref)
    throw std::logic_error("compact_fitter: a density form (rho) and the "
                           "EMBF0 zero reference (F) cannot share a fit");
  // The ladder counts (REFIT_COMPACT_OPT_EVERY, INSERT_TOPK, JITTERS, read
  // with these ranges by read_refit_numbers): opt_every and insert_topk below
  // 1 used to become 1 silently, and jitters + 1 candidates per rung must
  // fit the int that counts them.
  const auto count = [](const char *field, int v, int lo, int hi) {
    if (v < lo || v > hi)
      throw std::invalid_argument(std::string("compact_fitter: ") + field +
                                  " must be from " + std::to_string(lo) +
                                  " to " + std::to_string(hi) + ", got " +
                                  std::to_string(v));
  };
  constexpr int kIntMax = std::numeric_limits<int>::max();
  count("opt_every", cfg_.opt_every, 1, kIntMax);
  count("insert_topk", cfg_.insert_topk, 1, kIntMax);
  count("jitters", cfg_.jitters, 0, kIntMax - 1);
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

void CompactBlipFitter::note_nonneg(
    const tadah::models::NnlsResult &res) const {
  ++n_nonneg_solves_;
  if (res.converged) return;
  ++n_nonneg_unproven_;
  last_nonneg_stop_ = res.stop;
}

double CompactBlipFitter::solve_coeffs(BlipFit &f) const {
  // Every candidate passes through here before any eval / metric, so this is
  // the one place the channel's EMBF0 reference and RHOFORM are stamped onto
  // the fit. NONNEG / MONOTONE: every coefficient >= 0 (solve_nonneg);
  // MONOTONE also takes the step columns (cval / cder / sder2).
  f.zero_ref = cfg_.zero_ref;
  f.form = cfg_.rho_form;
  const bool positive = rhoform_is_positive(f.form);
  const int n = f.n();
  const std::size_t N = xs_.size();
  // EMBF0 ZERO: the VALUE columns are g(x_c) - g(x_c0) (derivative columns
  // unchanged); the reference terms exist only under zero_ref.
  std::vector<double> b0, db0;
  if (f.zero_ref) ref_terms(f, b0, db0);

  if (cfg_.objective == RefitObjective::Curve) {
    // STACKED value + 1st-deriv + 2nd-deriv least squares. value, f', f'' are
    // ALL linear in c, so this is still a closed-form linear solve (VARPRO
    // holds). Each block is divided by the target's own RMS so the three are
    // dimensionless; the derivative blocks additionally carry sqrt(deriv weight)
    // so VALUE stays dominant (crystal energy depends on the curve VALUES — an
    // equal weighting lets the solver trade value accuracy away). A
    // numerically-flat block (use_dX_ false) is dropped. Basis columns: value =
    // bval_d, f' = bder_d (= eta*g'), f'' = eta^2 * bval2_d (RHOFORM
    // MONOTONE: the step kinds S, dS/dr, d2S/dr2).
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
      if (f.zero_ref) {
        for (int k = 0; k < n; ++k)
          A(r, k) = sw * (bval_d(f.kind, f.degree(k),
                                 f.eta[k] * (xs_[i] - f.mu[k])) - b0[k]);
      } else {
        for (int k = 0; k < n; ++k)
          A(r, k) = sw * cval(f.kind, f.form, f.degree(k), f.eta[k] * (xs_[i] - f.mu[k]));
      }
      b(r) = sw * ys_[i];
      ++r;
    }
    if (d1on)
      for (std::size_t i = 0; i < N; ++i)
        if (fd_ok_[i]) {  // first-derivative block (value-dominant: x dwr)
          const double sw = dwr * (wts_.empty() ? 1.0 : std::sqrt(wts_[i])) / nd1_;
          for (int k = 0; k < n; ++k)
            A(r, k) = sw * cder(f.kind, f.form, f.degree(k),
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
            A(r, k) = f.form == RhoForm::Monotone
                          ? sw * sder2(f.kind, xc, f.eta[k])
                          : sw * f.eta[k] * f.eta[k] *
                                bval2_d(f.kind, f.degree(k), xc);
          }
          b(r) = sw * d2y_[i];
          ++r;
        }
    if (positive) {
      note_nonneg(solve_nonneg(A, b, f.c));
    } else {
      Eigen::VectorXd c = A.completeOrthogonalDecomposition().solve(b);
      f.c.assign(c.data(), c.data() + n);
    }
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
    if (f.zero_ref) {
      for (int k = 0; k < n; ++k)
        A(i, k) = sw * (bval_d(f.kind, f.degree(k),
                               f.eta[k] * (xs_[i] - f.mu[k])) - b0[k]);
    } else {
      for (int k = 0; k < n; ++k)
        A(i, k) = sw * cval(f.kind, f.form, f.degree(k),
                            f.eta[k] * (xs_[i] - f.mu[k]));
    }
    b(i) = sw * ys_[i];
  }
  if (positive) {
    note_nonneg(solve_nonneg(A, b, f.c));
  } else {
    // rank-revealing: a blip pushed outside the sampled range yields a ~zero
    // column, which plain QR mishandles
    Eigen::VectorXd c =
        A.completeOrthogonalDecomposition().solve(b);
    f.c.assign(c.data(), c.data() + n);
  }
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
  long long *evals;   ///< the fitter's objective-evaluation count
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
  ++*ctx->evals;
  const int n = ctx->work->n();
  for (int k = 0; k < n; ++k) ctx->work->mu[k] = v[k];
  if (!one_sided(ctx->work->kind))  // knot families: eta pinned (gauge)
    for (int k = 0; k < n; ++k) ctx->work->eta[k] = v[n + k];
  return ctx->self->solve_coeffs(*ctx->work) *
         (ctx->guard ? knot_gap_factor(*ctx->work, ctx->span) : 1.0) *
         ctx->self->bump_spill_guard(*ctx->work);  // 1 unless hard_rcut gate
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
  const bool positive = rhoform_is_positive(f.form);
  const int n = f.n();
  const bool os = one_sided(f.kind);  // knot families: mu-only theta space
  const std::size_t N = xs_.size();
  double sse = 0.0;
  if (grad) grad->assign(os ? n : 2 * n, 0.0);
  // EMBF0 ZERO: the columns are g(x_c) - g(x_c0), so dA/dtheta carries the
  // reference terms too (same mu/eta Jacobian as CurveCost::evaluate_zero_ref)
  std::vector<double> b0, db0;
  if (grad && f.zero_ref) ref_terms(f, b0, db0);
  for (std::size_t i = 0; i < N; ++i) {
    const double sw = wts_.empty() ? 1.0 : std::sqrt(wts_[i]);
    const double r = sw * (f.eval(xs_[i]) - ys_[i]);
    sse += r * r;
    if (grad && f.zero_ref) {
      for (int k = 0; k < n; ++k) {
        const double xc = f.eta[k] * (xs_[i] - f.mu[k]);
        const double db = bder_d(f.kind, f.degree(k), xc, f.eta[k]);
        (*grad)[k] += 2.0 * r * sw * f.c[k] * (-(db - db0[k]));
        if (!os)
          (*grad)[n + k] += 2.0 * r * sw * f.c[k] *
                            ((xs_[i] - f.mu[k]) * (db / f.eta[k]) -
                             (0.0 - f.mu[k]) * (db0[k] / f.eta[k]));
      }
    } else if (grad) {
      // Golub-Pereyra: A^T r = 0 at the inner optimum, so the c-variation
      // term vanishes and d(SSE)/dtheta = 2 r^T (dA/dtheta) c exactly.
      // RHOFORM NONNEG / MONOTONE (non-negative least squares): the same on
      // the passive set P = {k : c_k > 0}, whose columns satisfy
      // A_P^T r = 0; a column held at c_k = 0 stays there under a small
      // change of theta and contributes nothing, so it is skipped. The step
      // kinds enter through dS/dr (cder), as the bump columns through g'.
      for (int k = 0; k < n; ++k) {
        if (positive && f.c[k] == 0.0) continue;
        const double xc = f.eta[k] * (xs_[i] - f.mu[k]);
        const double db = cder(f.kind, f.form, f.degree(k), xc, f.eta[k]);
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
ThetaBox theta_box(const BlipFit &f, double lo, double hi, double span,
                   double hard_rcut = 0.0) {
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
  // Bump-support gate (cfg.hard_rcut): no centre may sit past the hard
  // cutoff — the width-driven joint constraint mu + w/eta <= rcut is handled
  // by the spill penalty + clamp_bump_support.
  const double mu_ub = hard_rcut > 0.0 ? std::min(hi + span, hard_rcut)
                                       : hi + span;
  b.lb.resize(2 * n);
  b.ub.resize(2 * n);
  b.v.resize(2 * n);
  for (int k = 0; k < n; ++k) {
    b.lb[k] = lo - span;       b.ub[k] = mu_ub;
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
  long long *evals;  ///< the fitter's objective-evaluation count
};
double nlopt_grad_obj(const std::vector<double> &v, std::vector<double> &grad,
                      void *data) {
  auto *ctx = static_cast<GradCtx *>(data);
  ++*ctx->evals;
  theta_unpack(v, *ctx->work);
  if (!grad.empty()) {
    std::vector<double> g;
    const double f = ctx->self->reduced_mse(*ctx->work, &g);
    grad = g;
    return f;
  }
  return ctx->self->reduced_mse(*ctx->work, nullptr);
}

// Thrown by the objective wrapper of dlib_lbfgs() at the call past MAXEVAL.
struct EvalBudgetSpent {};
}  // namespace

std::string CompactBlipFitter::dlib_lbfgs(BlipFit &f, int maxeval,
                                          double min_delta,
                                          unsigned long history) const {
  // dlib's stop strategies count L-BFGS iterations (each a line search of
  // one or more evaluations), not evaluations: MAXEVAL is counted here, in
  // the objective wrapper, as NLopt counts it. dlib asks for the value and
  // the gradient of the VARPRO objective in separate calls, each of which
  // solves the linear coefficients: each is one evaluation, and the call
  // past the budget ends the search at the last iterate.
  ThetaBox b = theta_box(f, lo_, hi_, span_, cfg_.hard_rcut);
  const int dim = static_cast<int>(b.v.size());
  using dvec = dlib::matrix<double, 0, 1>;
  dvec x(dim), xl(dim), xu(dim);
  for (int i = 0; i < dim; ++i) { x(i) = b.v[i]; xl(i) = b.lb[i]; xu(i) = b.ub[i]; }
  BlipFit work = f;
  long long used = 0;
  const auto spend = [&]() {
    if (used >= maxeval) throw EvalBudgetSpent{};
    ++used;
    ++n_obj_evals_;
  };
  auto obj = [&](const dvec &p) {
    spend();
    std::vector<double> v(p.begin(), p.end());
    theta_unpack(v, work);
    return reduced_mse(work, nullptr);
  };
  auto der = [&](const dvec &p) {
    spend();
    std::vector<double> v(p.begin(), p.end()), g;
    theta_unpack(v, work);
    reduced_mse(work, &g);
    dvec out(dim);
    for (int i = 0; i < dim; ++i) out(i) = g[i];
    return out;
  };
  std::string term;
  try {
    dlib::find_min_box_constrained(dlib::lbfgs_search_strategy(history),
                                   dlib::objective_delta_stop_strategy(min_delta),
                                   obj, der, x, xl, xu);
    term = "FTOL_REACHED";
  } catch (const EvalBudgetSpent &) {
    term = "MAXEVAL_REACHED";
  } catch (const std::exception &) {
    term = "threw";
  }
  std::vector<double> v(x.begin(), x.end());
  theta_unpack(v, f);
  return "DLIB_LBFGS:" + term + "(" + std::to_string(used) + "ev)";
}

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
    // dlib box-constrained LBFGS on the VARPRO reduced objective, MAXEVAL
    // objective evaluations, ending at FTOL_ABS (an absolute change)
    last_explore_term_ = dlib_lbfgs(f, maxeval, cfg_.dlib_inner_ftol_abs, 20);
    return solve_coeffs(f);
  } else if (a == "NONE") {
    last_explore_term_ = "NONE(linear-solve)";
    return solve_coeffs(f);
  } else {
    throw std::runtime_error("compact_fitter: unknown explore algo '" + a +
                             "' (SBPLX|NELDERMEAD|BOBYQA|PRAXIS|NLOPT_LBFGS|"
                             "DLIB_LBFGS|NONE)");
  }

  ThetaBox b = theta_box(f, lo_, hi_, span_, cfg_.hard_rcut);
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
  // Both objectives outlive optimize(): the gradient one used to be a local
  // of the if-branch that set it, so NLOPT_LBFGS optimised through a
  // dangling pointer.
  GradCtx ctx{this, &work, &n_obj_evals_};
  VarproCtx ctx_df{this, &work, span_, cfg_.merge_guard, &n_obj_evals_};
  if (use_grad)
    opt.set_min_objective(nlopt_grad_obj, &ctx);
  else
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
    // VARPRO reduced-gradient local, tight tolerances, at most
    // cfg_.lbfgs_refine_maxeval objective evaluations: NLopt counts them,
    // and dlib_lbfgs() counts them in its objective wrapper.
    if (a == "DLIB_LBFGS") {
      last_refine_term_ = dlib_lbfgs(f, cfg_.lbfgs_refine_maxeval,
                                     cfg_.dlib_refine_ftol_abs, 30);
      return solve_coeffs(f);
    }
    ThetaBox b = theta_box(f, lo_, hi_, span_, cfg_.hard_rcut);
    const int dim = static_cast<int>(b.v.size());
    nlopt::opt opt(nlopt::LD_LBFGS, static_cast<unsigned>(dim));
    opt.set_lower_bounds(b.lb);
    opt.set_upper_bounds(b.ub);
    opt.set_ftol_rel(cfg_.ceres_ftol);
    opt.set_xtol_rel(1e-14);
    opt.set_maxeval(cfg_.lbfgs_refine_maxeval);
    BlipFit work = f;
    GradCtx ctx{this, &work, &n_obj_evals_};
    opt.set_min_objective(nlopt_grad_obj, &ctx);
    double fval = 1e300;
    try {
      last_refine_term_ = nlopt_term(opt.optimize(b.v, fval));
    } catch (const std::exception &) {
      last_refine_term_ = nlopt_term(opt.last_optimize_result());
    }
    last_refine_term_ = std::string("NLOPT_LBFGS:") + last_refine_term_ +
                        "(" + std::to_string(opt.get_numevals()) + "ev)";
    theta_unpack(b.v, f);
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
  f.zero_ref = cfg_.zero_ref;  // EMBF0 ZERO (F channel); re-stamped per solve
  f.form = cfg_.rho_form;      // RHOFORM (rho channel); likewise
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
  ThetaBox b = theta_box(f, lo_, hi_, span_, cfg_.hard_rcut);

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
    // dlib calls the objective exactly max_function_calls times (MAXEVAL
    // objective evaluations); it used to take max(500, MAXEVAL/20) calls.
    long long used = 0;
    auto obj = [&](const dvec &p) {
      ++used;
      ++n_obj_evals_;
      std::vector<double> v(p.begin(), p.end());
      theta_unpack(v, work);
      return reduced_mse(work, nullptr);
    };
    auto res = dlib::find_min_global(
        obj, xl, xu,
        dlib::max_function_calls(
            static_cast<std::size_t>(cfg_.dlib_global_maxeval)));
    last_seed_term_ =
        "DLIB_GLOBAL:MAXEVAL_REACHED(" + std::to_string(used) + "ev)";
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
  VarproCtx ctx{this, &work, span_, cfg_.merge_guard, &n_obj_evals_};
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
  f.zero_ref = cfg_.zero_ref;  // the channel's EMBF0 reference (cf. solve_coeffs)
  f.form = cfg_.rho_form;      // and its RHOFORM
  ceres::Problem prob;
  std::vector<double> &c = f.c, &mu = f.mu, &eta = f.eta;
  prob.AddResidualBlock(
      new CurveCost(xs_, ys_, wts_, n, f.kind, f.deg, f.zero_ref, f.form),
      nullptr, c.data(), mu.data(), eta.data());
  // RHOFORM NONNEG / MONOTONE: the coefficients stay >= 0 through the refine
  // too (solve_coeffs' non-negative least squares does not bind Ceres, which
  // moves c freely). Ceres needs a feasible start: a negative c_k (a
  // caller's arbitrary start) begins at the bound.
  if (rhoform_is_positive(f.form))
    for (int k = 0; k < n; ++k) {
      if (c[k] < 0.0) c[k] = 0.0;
      prob.SetParameterLowerBound(c.data(), k, 0.0);
    }
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
    // bump-support gate: centres stay at/below the hard cutoff (the joint
    // mu + w/eta constraint is projected by clamp_bump_support afterwards)
    const double mu_ub = cfg_.hard_rcut > 0.0
                             ? std::min(hi_ + span_, cfg_.hard_rcut)
                             : hi_ + span_;
    for (int k = 0; k < n; ++k) {
      if (f.is_pinned(k)) continue;  // held constant by the manifold below
      prob.SetParameterLowerBound(mu.data(), k, lo_ - span_);
      prob.SetParameterUpperBound(mu.data(), k, mu_ub);
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
  require_valid_options(opt, "compact_fitter refine");
  ceres::Solver::Summary sum;
  ceres::Solve(opt, &prob, &sum);
  if (cfg_.check_gradients && sum.termination_type == ceres::FAILURE)
    throw std::runtime_error("compact_fitter: analytic Jacobian failed the "
                             "Ceres finite-difference gradient check: " +
                             sum.message);
  last_refine_term_ = std::string(ceres_term(sum.termination_type)) + "(" +
                      std::to_string(sum.iterations.size()) + "it, " +
                      ceres_reason(sum) + ")";
  // final closed-form coeffs at the refined (mu,eta); NONNEG / MONOTONE: by
  // non-negative least squares, so c >= 0 whatever Ceres returned
  return solve_coeffs(f);
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
  double mu_ins = xw, eta_ins = eta0;
  if (cfg_.hard_rcut > 0.0) {
    // bump-support gate: start the new bump feasible (mu + w/eta <= rcut) —
    // a residual peak AT the cutoff gets the narrowest legal bump there.
    const double w = bump_w(f.kind);
    const double eta_ub = 80.0 * eta0;
    mu_ins = std::min(mu_ins, cfg_.hard_rcut - w / eta_ub);
    eta_ins = std::max(eta_ins, w / (cfg_.hard_rcut - mu_ins));
  }
  f.mu.push_back(mu_ins);
  f.eta.push_back(eta_ins);
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
  const double mu_ub = cfg_.hard_rcut > 0.0
                           ? std::min(hi_ + span_, cfg_.hard_rcut)
                           : hi_ + span_;  // bump-support gate on centres
  for (int k = 0; k < f.n(); ++k) {
    if (f.is_pinned(k)) continue;  // INIT-pinned anchors stay put
    f.mu[k] = std::clamp(f.mu[k] + 0.02 * span_ * g(rng), lo_ - span_,
                         mu_ub);
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

namespace {
// Every radial bump width is > 0: INIT refuses an anchor with eta <= 0
// (HPORefitEngine), the ladder searches widths in [0.03 eta0, 80 eta0] and
// FUSE keeps them at or above its width floor. A width <= 0 here is a
// broken invariant, not a bump to pass over: its c + w/eta is no support
// edge (owner decision 2026-09-29).
void require_positive_bump_widths(const BlipFit &f, const char *who) {
  for (int k = 0; k < f.n(); ++k)
    if (!(f.eta[k] > 0.0))
      throw std::logic_error(
          std::string("compact_fitter ") + who + ": radial bump " +
          std::to_string(k + 1) + " (mu " + std::to_string(f.mu[k]) +
          ") has width eta " + std::to_string(f.eta[k]) +
          " <= 0; every radial bump width must be > 0");
}
}  // namespace

int clamp_bump_support_params(BlipFit &f, double hard_rcut, double span) {
  // Bump-support projection: the optimisers' boxes cap the CENTRE at
  // hard_rcut but cannot express the JOINT constraint mu + w/eta <= rcut
  // (a wide bump centred inside rcut still spills), so every scored
  // candidate is projected back here. Centre first — only when even the
  // narrowest legal bump (the eta box ceiling) cannot fit at mu — then eta
  // is raised minimally so the support ends exactly at rcut. A violating
  // PINNED bump is an INIT anchor that cannot be honoured: throw.
  if (hard_rcut <= 0.0 || one_sided(f.kind)) return 0;
  const double w = bump_w(f.kind);
  const double tol = tadah::models::bump_rcut_tol(hard_rcut);
  const double eta_ub = 80.0 * (std::max(1, f.n() - 1) / span);
  require_positive_bump_widths(f, "clamp_bump_support_params");
  int moved = 0;
  for (int k = 0; k < f.n(); ++k) {
    if (f.mu[k] + w / f.eta[k] <= hard_rcut + tol) continue;
    if (f.is_pinned(k))
      throw std::runtime_error(
          "compact_fitter: pinned INIT anchor " + std::to_string(k + 1) +
          " (mu " + std::to_string(f.mu[k]) + ", eta " +
          std::to_string(f.eta[k]) + ") has support past the hard cutoff " +
          std::to_string(hard_rcut) + " — raise its --eta to at least " +
          std::to_string(w / (hard_rcut - f.mu[k])) +
          " or move the anchor inward.");
    if (f.mu[k] > hard_rcut - w / eta_ub) f.mu[k] = hard_rcut - w / eta_ub;
    f.eta[k] = std::max(f.eta[k], w / (hard_rcut - f.mu[k]));
    ++moved;
  }
  return moved;
}

int CompactBlipFitter::clamp_bump_support(BlipFit &f) const {
  const int moved = clamp_bump_support_params(f, cfg_.hard_rcut, span_);
  if (moved > 0) solve_coeffs(f);
  return moved;
}

double CompactBlipFitter::bump_spill_guard(const BlipFit &f) const {
  // Soft penalty (>= 1) so the derivative-free search is steered away from
  // spilling configurations instead of relying on the clamp to undo them
  // (precedent: knot_gap_factor). Scale: overhang normalised by 2% of the
  // span, squared — a bump reaching 2% past rcut roughly doubles the
  // objective.
  if (cfg_.hard_rcut <= 0.0 || one_sided(f.kind)) return 1.0;
  const double w = bump_w(f.kind);
  const double s = 0.02 * span_;
  require_positive_bump_widths(f, "bump_spill_guard");
  double pen = 0.0;
  for (int k = 0; k < f.n(); ++k) {
    const double over = f.mu[k] + w / f.eta[k] - cfg_.hard_rcut;
    if (over > 0.0) {
      const double v = over / s;
      pen += v * v;
    }
  }
  return 1.0 + pen;
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
    merge_close_knots(f);   // knot families only; no-op for blips
    clamp_bump_support(f);  // bump families under the hard_rcut gate only
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
    // bump-support gate (bump families under hard_rcut only): every SCORED
    // candidate is projected back to mu + w/eta <= rcut, so selection is fair
    clamp_bump_support(cand);
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
    const int k = (one_sided(kind_)) ? cfg_.insert_topk : 1;
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
  const int opt_every = one_sided(kind_) ? cfg_.opt_every : 1;  // blip: legacy
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
      // ---- legacy blip rung (bit-for-bit regression-gated at
      //      cfg.hard_rcut == 0; the gate only adds box caps + the
      //      post-merge clamp) ----
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
  // Positive forms: say how many coefficient solves were not proven
  // optimal (feasible, every c_k >= 0, but stopped before a Kuhn-Tucker
  // point), in the refit log and as a WARNING.
  if (n_nonneg_unproven_ > 0) {
    const std::string msg =
        std::string("WARNING: RHOFORM ") + rhoform_name(cfg_.rho_form) +
        ": " + std::to_string(n_nonneg_unproven_) + " of " +
        std::to_string(n_nonneg_solves_) +
        " non-negative least-squares coefficient solves stopped before a "
        "Kuhn-Tucker point (the last: " +
        tadah::models::nnls_stop_text(last_nonneg_stop_) +
        "): those candidates' coefficients are >= 0 but not proven "
        "optimal, so their scores can be worse than their basis allows.";
    log(msg);
    tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING,
                                        msg.substr(9));
  }
  return best;
}

// ------------------------------------------------------------- joint refine
namespace {
// E-V residuals over the scan: r_i = w_i (E_fit(i) - E_orig(i)), with
// E(i) = sum_d 0.5 phi(r_d) + F(rho_i), rho_i = sum_d rho(r_d).
// Parameter blocks: {c,mu,eta} x {phi,rho,F} (9 blocks). All derivatives are
// analytic: the only nontrivial chain is d(F(rho_i))/d(theta_rho) =
// F'(rho_i) * sum_d d(rho(r_d))/d(theta_rho).
// F_zero_ref (EMBF0 ZERO): F(rho) = sum_m c_m (g(x_c) - g(x_c0)) — the F
// block's value and F-parameter Jacobians carry the reference terms (as in
// CurveCost::evaluate_zero_ref); F'(rho) and the chain into rho are unchanged.
// rho_form (RHOFORM): MONOTONE evaluates the density block in the step kinds
// (rho_i = sum_d sum_k c_k S(x_c)); the chain into its parameters takes S and
// dS/dr where the bumps take g and eta g' (cval / cder).
class EnergyCost : public ceres::CostFunction {
public:
  EnergyCost(const std::vector<std::vector<double>> &dists,
             const std::vector<double> &e0, const std::vector<double> &w,
             int nphi, int nrho, int nF, BasisKind kphi = BasisKind::Blip,
             BasisKind krho = BasisKind::Blip, BasisKind kF = BasisKind::Blip,
             std::vector<int> dphi = {}, std::vector<int> drho = {},
             std::vector<int> dF = {}, bool F_zero_ref = false,
             RhoForm rho_form = RhoForm::Signed)
      : dists_(dists), e0_(e0), w_(w), nphi_(nphi), nrho_(nrho), nF_(nF),
        kphi_(kphi), krho_(krho), kF_(kF), dphi_(std::move(dphi)),
        drho_(std::move(drho)), dF_(std::move(dF)), F_zero_ref_(F_zero_ref),
        rho_form_(rho_form) {
    if (rho_form_ == RhoForm::Monotone && one_sided(krho_))
      no_step_kind(krho_);  // refused here, never inside a Ceres Evaluate
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
    // EMBF0 ZERO reference terms of the F block (parameter-only; ZERO only)
    std::vector<double> bF0, dbF0;
    if (F_zero_ref_) ref_terms(kF_, nF_, mF, eF, dF_, bF0, dbF0);
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
          rho_i += cR[k] * cval(krho_, rho_form_, deg_of(drho_, k),
                                eR[k] * (r - mR[k]));
      }
      // embedding value + F'(rho_i)
      double Fv = 0.0, Fp = 0.0;
      if (F_zero_ref_) {
        for (int m = 0; m < nF_; ++m) {
          const double xc = eF[m] * (rho_i - mF[m]);
          const double b = bval_d(kF_, deg_of(dF_, m), xc) - bF0[m];
          const double db = bder_d(kF_, deg_of(dF_, m), xc, eF[m]);
          Fv += cF[m] * b;
          Fp += cF[m] * db;
          if (jac) {
            if (jac[6]) jac[6][i * nF_ + m] = w * b;
            if (jac[7]) jac[7][i * nF_ + m] = -w * cF[m] * (db - dbF0[m]);
            if (jac[8])
              jac[8][i * nF_ + m] =
                  w * cF[m] *
                  ((rho_i - mF[m]) * (db / eF[m]) -
                   (0.0 - mF[m]) * (dbF0[m] / eF[m]));
          }
        }
      } else {
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
      }
      // chain rule into the density parameters
      if (jac && (jac[3] || jac[4] || jac[5])) {
        for (double r : dd)
          for (int k = 0; k < nrho_; ++k) {
            const double xc = eR[k] * (r - mR[k]);
            const double b = cval(krho_, rho_form_, deg_of(drho_, k), xc);
            const double db =
                cder(krho_, rho_form_, deg_of(drho_, k), xc, eR[k]);
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
  bool F_zero_ref_ = false;  // EMBF0 ZERO zero-referenced embedding block
  RhoForm rho_form_ = RhoForm::Signed;  // RHOFORM of the density block
};

// A bump channel parametrised by its outer support edges e = mu + w/eta
// instead of its centres mu (joint_refine's bump-support gate: e <= rcut is
// a Ceres box, mu + w/eta <= rcut is not). Wraps a cost function whose
// parameter block mu_block holds a channel's centres and eta_block its
// widths, evaluating it at mu = e - w/eta; chain rule d/de = d/dmu,
// d/deta|e = d/deta|mu + (w/eta^2) d/dmu. Owns the wrapped cost function.
class EdgeParamCost : public ceres::CostFunction {
public:
  EdgeParamCost(ceres::CostFunction *inner, std::vector<EdgeParamMap> maps)
      : inner_(inner), maps_(std::move(maps)) {
    set_num_residuals(inner_->num_residuals());
    *mutable_parameter_block_sizes() = inner_->parameter_block_sizes();
  }
  ~EdgeParamCost() override { delete inner_; }

  bool Evaluate(double const *const *p, double *residuals,
                double **jacobians) const override {
    const auto &sizes = parameter_block_sizes();
    std::vector<const double *> q(p, p + sizes.size());
    std::vector<std::vector<double>> mu(maps_.size());
    for (std::size_t m = 0; m < maps_.size(); ++m) {
      const EdgeParamMap &g = maps_[m];
      const double *e = p[g.mu_block], *eta = p[g.eta_block];
      mu[m].resize(sizes[g.mu_block]);
      for (int k = 0; k < sizes[g.mu_block]; ++k)
        mu[m][k] = e[k] - g.w / eta[k];
      q[g.mu_block] = mu[m].data();
    }
    if (!jacobians) return inner_->Evaluate(q.data(), residuals, nullptr);
    // d/de = d/dmu, so the mu Jacobian goes straight into the edge block's;
    // the eta chain needs it even when the edge block's is not requested.
    const int R = num_residuals();
    std::vector<double *> jac(jacobians, jacobians + sizes.size());
    std::vector<std::vector<double>> jmu(maps_.size());
    for (std::size_t m = 0; m < maps_.size(); ++m) {
      const EdgeParamMap &g = maps_[m];
      if (!jac[g.mu_block] && jac[g.eta_block]) {
        jmu[m].resize(static_cast<std::size_t>(R) * sizes[g.mu_block]);
        jac[g.mu_block] = jmu[m].data();
      }
    }
    if (!inner_->Evaluate(q.data(), residuals, jac.data())) return false;
    for (const EdgeParamMap &g : maps_) {
      if (!jacobians[g.eta_block]) continue;
      const int n = sizes[g.mu_block];
      const double *eta = p[g.eta_block];
      const double *jm = jac[g.mu_block];
      double *je = jacobians[g.eta_block];
      for (int i = 0; i < R; ++i)
        for (int k = 0; k < n; ++k)
          je[i * n + k] += jm[i * n + k] * (g.w / (eta[k] * eta[k]));
    }
    return true;
  }

private:
  ceres::CostFunction *inner_;
  std::vector<EdgeParamMap> maps_;
};
}  // namespace

double CompactBlipFitter::refine_edges(BlipFit &f) const {
  if (one_sided(f.kind) || !(cfg_.hard_rcut > 0.0))
    throw std::logic_error("compact_fitter refine_edges: a bump family under "
                           "a hard cutoff only");
  const int n = f.n();
  f.zero_ref = cfg_.zero_ref;
  f.form = cfg_.rho_form;
  const double w = bump_w(f.kind), rc = cfg_.hard_rcut;
  // the ladder's width floor (joint_refine's): 0.03 eta0
  const double floor = 0.03 * std::max(1, n - 1) / span_;
  require_positive_bump_widths(f, "refine_edges");
  std::vector<double> &c = f.c, &eta = f.eta;
  std::vector<double> e(n);
  for (int k = 0; k < n; ++k) {
    if (!f.is_pinned(k) && eta[k] < floor) eta[k] = floor;
    e[k] = f.mu[k] + w / eta[k];
    if (!f.is_pinned(k) && e[k] > rc) e[k] = rc;  // a spilling start: on the bound
  }
  ceres::Problem prob;
  prob.AddResidualBlock(
      new EdgeParamCost(new CurveCost(xs_, ys_, wts_, n, f.kind, f.deg,
                                      f.zero_ref, f.form),
                        {{1, 2, w}}),
      nullptr, c.data(), e.data(), eta.data());
  if (rhoform_is_positive(f.form))
    for (int k = 0; k < n; ++k) {
      if (c[k] < 0.0) c[k] = 0.0;
      prob.SetParameterLowerBound(c.data(), k, 0.0);
    }
  std::vector<int> pin;
  for (int k = 0; k < n; ++k) {
    if (f.is_pinned(k)) {
      pin.push_back(k);
      continue;
    }
    prob.SetParameterUpperBound(e.data(), k, rc);
    prob.SetParameterLowerBound(eta.data(), k, floor);
  }
  if (!pin.empty()) {  // hold pinned bumps (see refine())
    if (static_cast<int>(pin.size()) == n) {
      prob.SetParameterBlockConstant(e.data());
      prob.SetParameterBlockConstant(eta.data());
    } else {
      prob.SetManifold(e.data(), new ceres::SubsetManifold(n, pin));
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
  require_valid_options(opt, "compact_fitter refine_edges");
  ceres::Solver::Summary sum;
  ceres::Solve(opt, &prob, &sum);
  for (int k = 0; k < n; ++k)  // back to centres (a pinned bump keeps its)
    if (!f.is_pinned(k)) f.mu[k] = e[k] - w / eta[k];
  last_refine_term_ = std::string(ceres_term(sum.termination_type)) + "(" +
                      std::to_string(sum.iterations.size()) + "it, " +
                      ceres_reason(sum) + ", edges)";
  return solve_coeffs(f);
}

CompactConfig fuse_curve_config(const CompactConfig &cfg) {
  CompactConfig f = cfg;
  f.refine_algo = cfg.joint_algo;      // FUSE's optimiser
  f.ceres_max_iter = cfg.joint_max_iter;  // per function
  f.ceres_ftol = cfg.joint_ftol;
  f.ceres_gtol = cfg.joint_gtol;
  return f;
}

int fuse_curve_refine(const CompactBlipFitter &ff, BlipFit &fit) {
  const CompactConfig &cfg = ff.config();
  const bool ceres =
      cfg.refine_algo == "CERES_LM" || cfg.refine_algo == "CERES_DOGLEG";
  if (ceres && !one_sided(fit.kind) && cfg.hard_rcut > 0.0)
    ff.refine_edges(fit);
  else
    ff.refine_with_cfg(fit);
  return ff.clamp_bump_support(fit);
}

double joint_refine(BlipFit &phi, BlipFit &rho, BlipFit &F,
                    const JointCurveData &dphi, const JointCurveData &drho,
                    const JointCurveData &dF,
                    const std::vector<std::vector<double>> &dists,
                    const std::vector<double> &e_orig,
                    const std::vector<double> &e_weight, double curve_weight,
                    int max_iter, double ftol, const std::string &algo,
                    std::string *term_out, double phi_hard_rcut,
                    double rho_hard_rcut, int *iterations_out, double gtol) {
  if (iterations_out) *iterations_out = 0;
  if (algo != "NONE" && algo != "CERES_LM" && algo != "CERES_DOGLEG")
    throw std::runtime_error("joint_refine: unknown algo '" + algo +
                             "' (CERES_LM|CERES_DOGLEG|NONE)");
  // The EMBF0 zero reference is an EMBEDDING property: the E-V cost
  // evaluates phi/rho raw, so a zero-referenced radial fit would be
  // silently fitted as a different function than it evaluates.
  if (phi.zero_ref || rho.zero_ref)
    throw std::runtime_error("joint_refine: the EMBF0 zero reference applies "
                             "to the embedding F only, not to phi/rho");
  // Likewise the density form is a DENSITY property: the E-V cost evaluates
  // phi and F in their raw bases.
  if (phi.form != RhoForm::Signed || F.form != RhoForm::Signed)
    throw std::runtime_error("joint_refine: RHOFORM applies to the density "
                             "rho only, not to phi/F");
  if (rho.form == RhoForm::Monotone && one_sided(rho.kind))
    no_step_kind(rho.kind);
  if (term_out) *term_out = "NONE(skipped)";

  // Feasible starting values for the bounds below: the density amplitudes
  // of the positive forms at >= 0 (a caller's arbitrary rho.c begins at the
  // bound) and every free radial bump width at or above the ladder's floor.
  if (rhoform_is_positive(rho.form))
    for (int k = 0; k < rho.n(); ++k)
      if (rho.c[k] < 0.0) rho.c[k] = 0.0;
  auto eta_floor = [](const BlipFit &f, const JointCurveData &d) {
    if (one_sided(f.kind) || d.xs.size() < 2) return 0.0;
    const auto [mn, mx] = std::minmax_element(d.xs.begin(), d.xs.end());
    return 0.03 * std::max(1, f.n() - 1) / (*mx - *mn);
  };
  const double phi_eta_lb = eta_floor(phi, dphi);
  const double rho_eta_lb = eta_floor(rho, drho);
  for (auto [f, lb] : {std::pair<BlipFit *, double>{&phi, phi_eta_lb},
                       std::pair<BlipFit *, double>{&rho, rho_eta_lb}})
    if (lb > 0.0)
      for (int k = 0; k < f->n(); ++k)
        if (!f->is_pinned(k) && f->eta[k] < lb) f->eta[k] = lb;

  // Bump-support gate on the radial channels, inside the solve: a bump
  // channel with a hard cutoff is optimised in its outer support edges
  // e = mu + w/eta (EdgeParamCost), bounded at the cutoff, so every free bump
  // ends inside it and the caller's clamp_bump_support_params projection has
  // nothing to move. A spilling start begins on the bound (its centre moves
  // inward). Pinned bumps are held, knots and a fully pinned channel keep
  // their centres, and algo NONE only lifts the centres (no solve).
  struct Edge {
    bool on = false;
    double w = 0.0;
    std::vector<double> e;
  };
  auto make_edge = [&](const BlipFit &f, double hard_rcut) {
    Edge g;
    if (algo == "NONE" || hard_rcut <= 0.0 || one_sided(f.kind)) return g;
    int npin = 0;
    for (int k = 0; k < f.n(); ++k)
      if (f.is_pinned(k)) ++npin;
    if (npin == f.n()) return g;
    g.on = true;
    g.w = bump_w(f.kind);
    g.e.resize(f.n());
    for (int k = 0; k < f.n(); ++k) {
      g.e[k] = f.mu[k] + g.w / f.eta[k];
      if (!f.is_pinned(k) && g.e[k] > hard_rcut) g.e[k] = hard_rcut;
    }
    return g;
  };
  Edge ephi = make_edge(phi, phi_hard_rcut);
  Edge erho = make_edge(rho, rho_hard_rcut);
  // the parameter block holding a channel's positions (edges or centres)
  auto pos_block = [&](BlipFit &f) {
    if (&f == &phi && ephi.on) return ephi.e.data();
    if (&f == &rho && erho.on) return erho.e.data();
    return f.mu.data();
  };

  ceres::Problem prob;
  auto add_curve = [&](BlipFit &f, const JointCurveData &d, const Edge *g) {
    // scale curve rows by curve_weight (fold into per-sample weights)
    std::vector<double> wts =
        d.wts.empty() ? std::vector<double>(d.xs.size(), curve_weight) : d.wts;
    if (!d.wts.empty())
      for (auto &w : wts) w *= curve_weight;
    ceres::CostFunction *cost =
        new CurveCost(d.xs, d.ys, std::move(wts), f.n(), f.kind, f.deg,
                      f.zero_ref,  // EMBF0 ZERO: F only (d.ys = F_ref)
                      f.form);     // RHOFORM: rho only
    if (g && g->on) cost = new EdgeParamCost(cost, {{1, 2, g->w}});
    prob.AddResidualBlock(cost, nullptr, f.c.data(), pos_block(f),
                          f.eta.data());
  };
  add_curve(phi, dphi, &ephi);
  add_curve(rho, drho, &erho);
  add_curve(F, dF, nullptr);
  {
    ceres::CostFunction *cost =
        new EnergyCost(dists, e_orig, e_weight, phi.n(), rho.n(), F.n(),
                       phi.kind, rho.kind, F.kind, phi.deg, rho.deg, F.deg,
                       F.zero_ref, rho.form);
    std::vector<EdgeParamMap> maps;
    if (ephi.on) maps.push_back({1, 2, ephi.w});
    if (erho.on) maps.push_back({4, 5, erho.w});
    if (!maps.empty()) cost = new EdgeParamCost(cost, std::move(maps));
    prob.AddResidualBlock(cost, nullptr, phi.c.data(), pos_block(phi),
                          phi.eta.data(), rho.c.data(), pos_block(rho),
                          rho.eta.data(), F.c.data(), F.mu.data(),
                          F.eta.data());
  }

  // RHOFORM NONNEG / MONOTONE: the density amplitudes stay >= 0 through the
  // joint refinement (the ladder's non-negative solve does not bind Ceres).
  if (rhoform_is_positive(rho.form))
    for (int k = 0; k < rho.n(); ++k)
      prob.SetParameterLowerBound(rho.c.data(), k, 0.0);
  // The radial bump widths stay > 0 in every form. Unlike the ladder's
  // refine, the joint solve has no width box, and a bump is even in eta, so
  // it can walk a width through 0 to its mirror: harmless for the bump's
  // values, but eta <= 0 has no support edge (the bump-support clamp
  // refuses it, and tadah hpo refuses such a width in a potential), and a
  // MONOTONE step with eta < 0 RISES (DM_REAM refuses SGRIDMB <= 0). phi and rho keep the
  // ladder's own width floor 0.03 eta0 (eta0 = max(1, n - 1)/span over the
  // channel's samples); a start below it is lifted. Pinned widths are held
  // constant (the starts were lifted above).
  auto bound_bump_eta = [&](BlipFit &f, double lb) {
    if (lb <= 0.0) return;
    for (int k = 0; k < f.n(); ++k)
      if (!f.is_pinned(k)) prob.SetParameterLowerBound(f.eta.data(), k, lb);
  };
  bound_bump_eta(phi, phi_eta_lb);
  bound_bump_eta(rho, rho_eta_lb);

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

  // Bump-support gate on the radial channels: the outer support edges of
  // an edge-parametrised channel are bounded at the hard cutoff (see
  // make_edge). Otherwise (algo NONE) the centres are bounded at it.
  auto bound_bump = [&](BlipFit &f, const Edge &g, double hard_rcut) {
    if (hard_rcut <= 0.0 || one_sided(f.kind)) return;
    int npin = 0;
    for (int k = 0; k < f.n(); ++k)
      if (f.is_pinned(k)) ++npin;
    if (npin == f.n()) return;  // whole block held constant below
    for (int k = 0; k < f.n(); ++k) {
      if (f.is_pinned(k)) continue;  // held constant by the manifold below
      if (g.on) {
        prob.SetParameterUpperBound(pos_block(f), k, hard_rcut);
      } else {
        prob.SetParameterUpperBound(f.mu.data(), k, hard_rcut);
        if (f.mu[k] > hard_rcut) f.mu[k] = hard_rcut;  // feasible start
      }
    }
  };
  bound_bump(phi, ephi, phi_hard_rcut);
  bound_bump(rho, erho, rho_hard_rcut);

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
      prob.SetParameterBlockConstant(pos_block(f));
    else
      prob.SetManifold(pos_block(f), new ceres::SubsetManifold(n, pin));
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
    opt.gradient_tolerance = gtol;
    opt.parameter_tolerance = 1e-16;
    opt.logging_type = ceres::SILENT;
    require_valid_options(opt, "joint_refine");
    ceres::Solver::Summary sum;
    ceres::Solve(opt, &prob, &sum);
    // the iterations max_num_iterations bounds: sum.iterations holds the
    // initial point (iteration 0) too
    if (iterations_out)
      *iterations_out = sum.iterations.empty()
                            ? 0
                            : static_cast<int>(sum.iterations.size()) - 1;
    // back to centres: mu = e - w/eta for every free bump of an
    // edge-parametrised channel (a pinned bump keeps its exact centre)
    for (auto [f, g] : {std::pair<BlipFit *, const Edge *>{&phi, &ephi},
                        std::pair<BlipFit *, const Edge *>{&rho, &erho}})
      if (g->on)
        for (int k = 0; k < f->n(); ++k)
          if (!f->is_pinned(k)) f->mu[k] = g->e[k] - g->w / f->eta[k];
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

HeldPasses run_held_passes(int max_passes, FuseBudget &budget,
                           const std::function<int()> &hold,
                           const std::function<int(int)> &solve,
                           const std::function<void(const std::string &)> &log) {
  HeldPasses r;
  if (budget.remaining() == 0) {
    r.skipped = true;
    log("FUSE held re-run skipped: MAXEVAL " + std::to_string(budget.maxeval) +
        " spent");
    return r;
  }
  for (int pass = 1; pass <= max_passes; ++pass) {
    if (budget.remaining() == 0) {
      r.budget_spent = true;
      log("FUSE held re-run stopped after pass " + std::to_string(r.passes) +
          " of " + std::to_string(max_passes) + ": MAXEVAL " +
          std::to_string(budget.maxeval) + " spent");
      break;
    }
    if (hold() == 0) break;
    budget.spend(solve(budget.remaining()));
    ++r.passes;
  }
  return r;
}

// ---- test hooks (see header) ----
ceres::CostFunction *make_edge_param_cost(ceres::CostFunction *inner,
                                          std::vector<EdgeParamMap> maps) {
  return new EdgeParamCost(inner, std::move(maps));
}
ceres::CostFunction *make_curve_cost(std::vector<double> xs,
                                     std::vector<double> ys,
                                     std::vector<double> wts, int n,
                                     BasisKind kind, bool zero_ref,
                                     std::vector<int> deg, RhoForm form) {
  return new CurveCost(std::move(xs), std::move(ys), std::move(wts), n, kind,
                       std::move(deg), zero_ref, form);
}
ceres::CostFunction *make_energy_cost(
    const std::vector<std::vector<double>> &dists,
    const std::vector<double> &e_orig, const std::vector<double> &e_weight,
    int nphi, int nrho, int nF, BasisKind kphi, BasisKind krho,
    BasisKind kF, bool F_zero_ref, std::vector<int> dF, RhoForm rho_form) {
  return new EnergyCost(dists, e_orig, e_weight, nphi, nrho, nF, kphi, krho,
                        kF, {}, {}, std::move(dF), F_zero_ref, rho_form);
}

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_USE_CERES

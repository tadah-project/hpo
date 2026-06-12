#ifndef TADAH_HPO_REFIT_COMPACT_FITTER_H
#define TADAH_HPO_REFIT_COMPACT_FITTER_H

/** \file compact_fitter.h
 *  \brief Compact optimised-blip curve fitter for `tadah refit` (REFIT_MODE
 *  compact): finds a SMALL set of blips (free centres, widths AND linear
 *  coefficients) representing a tabulated 1-D function, so the resulting
 *  pot.tadah exposes an HPO search space of ~10^2 instead of ~10^3 parameters.
 *
 *  Algorithm (ladder over the blip count n):
 *    1. n = n_min seed: NLopt GN_MLSL_LDS (low-discrepancy multistart) over
 *       (mu, eta) with the linear coefficients solved in closed form (VARPRO).
 *    2. each rung n -> n+1: insert one blip at the maximum-|residual| point;
 *       candidates = warm start + `jitters` randomised perturbations
 *       (exploration is essential: pure warm starts lock into bad basins).
 *       Every candidate: LN_SBPLX explore (derivative-free basin search —
 *       the measured binding constraint here is landscape multimodality, not
 *       smoothness: the blip is C2, only B''' jumps at its knots) then a
 *       Ceres Levenberg-Marquardt polish over (c, mu, eta) JOINTLY with
 *       ANALYTIC Jacobians (LM consumes only B and B') down to ~numerical
 *       noise (ftol 1e-15).
 *    3. rung selection + stopping use the caller-supplied ENERGY metric (the
 *       compression-aware crystal E-V error, normalised to the user
 *       tolerances), not the curve RMSE: curve-optimal != energy-optimal.
 *  A separate joint_refine() then polishes ALL THREE functions' parameters
 *  against curve residuals PLUS the E-V residuals themselves (analytic
 *  d(E)/d(parameter) through the lattice sums).
 *
 *  Built only when the HPO library is configured with Ceres
 *  (TADAH_USE_CERES); the engine falls back to dense mode otherwise.
 */

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {
namespace refit {

/** Basis family for one fitted expansion f(x) = sum_k c_k g(eta_k(x - mu_k)).
 *  - Blip:   g = cubic B-spline bump (two shape params mu, eta per function)
 *  - Knot5L: g = max(0,-u)^5 left-active quintic knot function (Mendelev/
 *            Ackland form; eta is a gauge, pinned to 1 — mu is the knot)
 *  - Knot5R: right-active mirror max(0,u)^5 (embedding F(rho) side)
 *  - Blip5:  g = quintic B-spline bump (M6, support [-3,3], C^4) — the
 *            symmetric C^4 twin of Blip; full bump semantics (free mu, eta) */
enum class BasisKind { Blip = 0, Knot5L = 1, Knot5R = 2, Blip5 = 3 };

/** True for the one-sided knot families (eta pinned, single shape param).
 *  The symmetric bumps (Blip, Blip5) keep the full (mu, eta) ladder. */
inline bool one_sided(BasisKind k) {
  return k == BasisKind::Knot5L || k == BasisKind::Knot5R;
}

/** One fitted expansion: f(x) = sum_k c_k g(eta_k (x - mu_k)). */
struct BlipFit {
  std::vector<double> mu, eta, c;
  BasisKind kind = BasisKind::Blip;
  /** Per-atom degree for the KNOT kinds: 5 (quintic, C^4) or 3 (cubic, C^2 —
   *  an exact f''' corner, the Mendelev/Ackland atom). Empty = all 5.
   *  Ignored by the bump kinds. */
  std::vector<int> deg;
  double curve_rmse = 1e300;   ///< unweighted RMSE over the fit samples
  double metric = 1e300;       ///< caller's energy metric (<=1 == within tol)
  int n() const { return static_cast<int>(mu.size()); }
  int degree(int k) const {
    return (k < static_cast<int>(deg.size())) ? deg[k] : 5;
  }
  double eval(double x) const;
  double deriv(double x) const;
  double deriv2(double x) const;  ///< analytic f'' (all families are >= C^2)
};

/** Per-rung record for the ladder trace file. */
struct LadderRow {
  int n = 0;
  double curve_rmse = 0, metric = 0, seconds = 0;
  bool kept = false;           ///< became the running-best (monotone envelope)
};

struct CompactConfig {
  int n_min = 2;
  int n_max = 40;              ///< per-function blip ceiling
  int jitters = 2;             ///< randomised warm restarts per rung
  int sbplx_maxeval = 20000;   ///< explore budget per candidate
  double sbplx_xtol = 1e-11;
  double sbplx_ftol = 1e-13;
  int mlsl_maxeval = 60000;    ///< n_min seed only
  int ceres_max_iter = 1000;
  double ceres_ftol = 1e-15;   ///< "close to numerical noise" polish
  double ceres_gtol = 1e-14;
  double target = 1.0;         ///< stop when metric <= target
  int stall_rungs = 6;         ///< stop after this many rungs w/o improvement
  unsigned seed = 12345;       ///< deterministic jitter
  bool check_gradients = false;  ///< Ceres FD-vs-analytic Jacobian audit (tests)

  // ---- knot-family ladder schedule (one-sided kinds ONLY; the blip path is
  //      untouched — bit-for-bit regression-gated) ----
  // Full explore+polish every K-th rung; intermediate rungs grow by
  // energy-aware insertion + linear solve only (the pilot's measured-strong
  // insertion-only baseline), with one final full optimisation of the
  // envelope best. Rationale: the per-rung optimiser falls into coalescence
  // basins whose repair leaves the ENERGY metric flat — 6 such rungs fired
  // the stall counter at n~9-14 and under-resolved curves broke Fe C11 /
  // Ta C44 in the LAMMPS gate.
  int opt_every = 1;    ///< 1 = optimise every rung (legacy / blip behaviour)
  int insert_topk = 1;  ///< try the top-k separated residual peaks, keep the
                        ///< METRIC-best insertion (curve peak != energy peak)
  int knot_degree = 5;  ///< uniform atom degree for the knot kinds: 5 | 3
  bool knot_mix = false;  ///< KnotMix auto-mode: insertion tries BOTH degrees
                          ///< per candidate site; the cubic atom is accepted
                          ///< only on a DOUBLE WIN — better energy metric AND
                          ///< no worse curvature (d2) residual. A genuine
                          ///< f''' break wins both (the quintic cannot make a
                          ///< corner); a spurious cubic wins value/energy but
                          ///< loses d2 — the measured failure mode of the
                          ///< first (metric-only) detector, which deployed
                          ///< cubics on 40-60% of atoms everywhere and
                          ///< degraded elastic constants broadly.

  /** Optional live progress sink (the refit optimisation log): called with
   *  one formatted line per ladder event (seed, rung, merge, final polish).
   *  The engine prefixes lines with the function name and writes them to
   *  REFIT_LOG (flushed per line, so the file is inspectable DURING the
   *  fit). Null = silent. */
  std::function<void(const std::string &)> log_fn;

  // ---- optimiser strategy per stage (Cycle-6 study; all user-selectable) ----
  // seed (global structure search at n_min, over mu/eta with closed-form c):
  //   MLSL | DIRECT | CRS2 | DLIB_GLOBAL | UNIFORM(no global step)
  std::string seed_algo = "MLSL";
  // explore (per rung candidate, derivative-free or VARPRO-gradient local):
  //   SBPLX | NELDERMEAD | BOBYQA | PRAXIS | NLOPT_LBFGS | DLIB_LBFGS | NONE
  std::string explore_algo = "SBPLX";
  // polish (high-precision local):
  //   CERES_LM (joint c,mu,eta analytic-Jacobian trust region)
  //   CERES_DOGLEG | NLOPT_LBFGS | DLIB_LBFGS (VARPRO reduced gradient) | NONE
  std::string polish_algo = "CERES_LM";
};

/** Fits one tabulated function on [lo,hi] given dense samples (xs, ys) and
 *  optional per-sample weights (empty = 1). */
class CompactBlipFitter {
public:
  using Metric = std::function<double(const BlipFit &)>;

  CompactBlipFitter(std::vector<double> xs, std::vector<double> ys,
                    std::vector<double> wts, double lo, double hi,
                    const CompactConfig &cfg,
                    BasisKind kind = BasisKind::Blip);

  /** Run the ladder. `metric` maps a candidate fit to the normalised energy
   *  error (the fitter treats it as a black box). Returns the monotone-envelope
   *  best; `trace` receives one row per rung. */
  BlipFit fit(const Metric &metric, std::vector<LadderRow> &trace);

  /** Closed-form linear coefficients for fixed (mu, eta); returns curve rmse.
   *  Exposed for tests. */
  double solve_coeffs(BlipFit &f) const;

  /** Ceres trust-region polish of (c, mu, eta) jointly, analytic Jacobian
   *  (LM, or DOGLEG when cfg.polish_algo == CERES_DOGLEG); returns final
   *  curve rmse (also refreshes f.curve_rmse). Exposed for tests. */
  double polish(BlipFit &f) const;

  /** Polish with the backend selected by cfg.polish_algo (CERES_* | the
   *  VARPRO reduced-gradient NLOPT_LBFGS / DLIB_LBFGS | NONE). */
  double polish_with_cfg(BlipFit &f) const { return polish_dispatch(f); }

  /** VARPRO reduced objective (mean squared error over samples) and, when
   *  grad is non-null, its EXACT gradient — Golub-Pereyra: at the inner
   *  least-squares optimum d(SSE)/dtheta = 2 r^T (dPhi/dtheta) c. Gradient
   *  layout: (mu..., eta...) for Blip; (mu...) only for the one-sided knot
   *  families (eta is a gauge, pinned). Exposed for tests. */
  double reduced_mse(BlipFit &f, std::vector<double> *grad) const;

  /** Coalescence guard for the knot families: merge knots closer than
   *  1e-3*span to their midpoint and re-solve coefficients (no-op for
   *  blips). Called after every polish in the ladder. Exposed for tests. */
  void merge_close_knots(BlipFit &f) const;

  /** RMSE of the fit's analytic f'' against the second derivative of the
   *  SAMPLE DATA (3-point finite differences on the sample grid), over the
   *  interior. The KnotMix double-win criterion and the progress log use
   *  it; cheap (O(N n)). Exposed for tests. */
  double d2_rmse(const BlipFit &f) const;

private:
  double sbplx_explore(BlipFit &f, int maxeval) const;  ///< VARPRO objective
  double explore(BlipFit &f) const;        ///< dispatch on cfg_.explore_algo
  double polish_dispatch(BlipFit &f) const;
  BlipFit seed_mlsl() const;               ///< dispatch on cfg_.seed_algo
  void insert_blip_at_worst(BlipFit &f) const;
  /** Up to k weighted-residual peak locations, pairwise >= 2% span apart,
   *  in descending residual order (k=1 == the insert_blip_at_worst peak). */
  std::vector<double> residual_peaks(const BlipFit &f, int k) const;
  /** Insert one basis function targeting residual location xw (one-sided
   *  kinds: offset to the atom's active side; blips: at xw). Degree comes
   *  from cfg.knot_degree; insert_at_deg overrides it per insertion (the
   *  KnotMix auto-mode tries both 5 and 3 and keeps the metric-best). */
  void insert_at(BlipFit &f, double xw) const;
  void insert_at_deg(BlipFit &f, double xw, int d) const;
  void jitter(BlipFit &f, std::mt19937 &rng) const;

  std::vector<double> xs_, ys_, wts_;
  double lo_, hi_, span_, eta0_;
  CompactConfig cfg_;
  BasisKind kind_ = BasisKind::Blip;
};

/** Joint Ceres refinement of all three functions against their curve residuals
 *  PLUS the crystal E-V residuals (analytic d(E)/d(parameter) through the
 *  neighbour sums). `dists[i]` = neighbour distances of scan point i (within
 *  rcut), `e_orig[i]` = original crystal energy/atom there, `e_weight[i]` =
 *  residual weight (encode the ambient/compressed tolerances here).
 *  phi/rho/F: fits to refine in place. Curve data as in the fitter.
 *  Returns the final max weighted |E residual|. */
struct JointCurveData {
  std::vector<double> xs, ys, wts;  // wts empty = 1
};
double joint_refine(BlipFit &phi, BlipFit &rho, BlipFit &F,
                    const JointCurveData &dphi, const JointCurveData &drho,
                    const JointCurveData &dF,
                    const std::vector<std::vector<double>> &dists,
                    const std::vector<double> &e_orig,
                    const std::vector<double> &e_weight,
                    double curve_weight, int max_iter, double ftol,
                    const std::string &algo = "CERES_LM");
                    // algo: CERES_LM | CERES_DOGLEG | NONE
                    // (basis families are carried by each BlipFit's `kind`)

}  // namespace refit
}  // namespace hpo
}  // namespace tadah

// ---- test hooks: raw Ceres cost factories so the analytic Jacobians can be
//      audited against finite differences with a proper significance floor
//      (Ceres' built-in check has only a RELATIVE knob, which false-positives
//      on the numerically-zero support-edge entries of a compact basis).
namespace ceres { class CostFunction; }
namespace tadah {
namespace hpo {
namespace refit {
ceres::CostFunction *make_curve_cost(std::vector<double> xs,
                                     std::vector<double> ys,
                                     std::vector<double> wts, int n,
                                     BasisKind kind = BasisKind::Blip);
ceres::CostFunction *make_energy_cost(
    const std::vector<std::vector<double>> &dists,
    const std::vector<double> &e_orig, const std::vector<double> &e_weight,
    int nphi, int nrho, int nF,
    BasisKind kphi = BasisKind::Blip, BasisKind krho = BasisKind::Blip,
    BasisKind kF = BasisKind::Blip);

}  // namespace refit
}  // namespace hpo
}  // namespace tadah
#endif  // TADAH_HPO_REFIT_COMPACT_FITTER_H

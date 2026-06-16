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
 *       Ceres Levenberg-Marquardt refine over (c, mu, eta) JOINTLY with
 *       ANALYTIC Jacobians (LM consumes only B and B') down to ~numerical
 *       noise (ftol 1e-15).
 *    3. rung selection + stopping use the caller-supplied ENERGY metric (the
 *       compression-aware crystal E-V error, normalised to the user
 *       tolerances), not the curve RMSE: curve-optimal != energy-optimal.
 *  A separate joint_refine() then refines ALL THREE functions' parameters
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

/** What the whole fitting chain (every stage: seed, EXPLORE, REFINE, FUSE, and
 *  the rung selection / stall / stop decisions) is steered toward.
 *  - Energy: the caller's crystal energy metric (default; the metric is a
 *    black-box passed to fit()). Inner optimisers do value-curve least squares.
 *  - Curve: a normalised combined RMSE of the fitted VALUE + 1st-derivative +
 *    2nd-derivative against the tabulated function (each term divided by the
 *    table's own RMS over the window, then RMS-combined into one dimensionless
 *    score). The linear coefficients are solved from a STACKED value+d1+d2
 *    least-squares; the nonlinear positions are searched derivative-free
 *    (SBPLX) on the same combined objective; the energy metric is not used.
 *    REFIT_OBJECTIVE. */
enum class RefitObjective { Energy = 0, Curve = 1 };

/** One fitted expansion: f(x) = sum_k c_k g(eta_k (x - mu_k)). */
struct BlipFit {
  std::vector<double> mu, eta, c;
  BasisKind kind = BasisKind::Blip;
  /** Per-atom degree for the KNOT kinds: 5 (quintic, C^4) or 3 (cubic, C^2 —
   *  an exact f''' corner, the Mendelev/Ackland atom). Empty = all 5.
   *  Ignored by the bump kinds. */
  std::vector<int> deg;
  /** Per-basis pin mask (parallel to mu/eta/c when non-empty). pinned[k]==true
   *  freezes the position mu[k] (and eta[k] for the bump families) in
   *  EXPLORE/REFINE/FUSE; the linear coefficient c[k] is still solved by VARPRO.
   *  Empty = nothing pinned (the regression-gated default — the blip path is
   *  byte-identical when this is empty). Maintained index-aligned through
   *  insert/merge/jitter; pinned knots are never moved or merged away. */
  std::vector<bool> pinned;
  bool is_pinned(int k) const {
    return k >= 0 && k < static_cast<int>(pinned.size()) && pinned[k];
  }
  double curve_rmse = 1e300;   ///< weighted value RMSE over the fit samples
                               ///< (pressure-relaxed weights; plain RMSE if none)
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
  double ceres_ftol = 1e-15;   ///< "close to numerical noise" refine
  double ceres_gtol = 1e-14;
  double target = 1.0;         ///< stop when metric <= target
  int stall_rungs = 6;         ///< stop after this many rungs w/o improvement
  unsigned seed = 12345;       ///< deterministic jitter
  bool check_gradients = false;  ///< Ceres FD-vs-analytic Jacobian audit (tests)

  /** Objective the whole chain optimises toward (REFIT_OBJECTIVE). Energy =
   *  the caller's energy metric (default, regression-gated). Curve = the
   *  normalised value+1st+2nd-derivative combined RMSE (see RefitObjective).
   *  In Curve mode the energy metric passed to fit() is ignored; solve_coeffs
   *  fits the stacked value+d1+d2 system and EXPLORE/REFINE/FUSE run
   *  derivative-free on the combined objective. */
  RefitObjective objective = RefitObjective::Energy;

  /** Curve-objective weight of EACH derivative block (d1, d2) RELATIVE to the
   *  value block, in both the stacked coefficient solve and the combined score
   *  (REFIT_CURVE_DERIV_WEIGHT). 0 (DEFAULT) = pure VALUE fit — crystal-faithful:
   *  measured (Ti) to reproduce the crystal as well as energy mode (cohesive
   *  +3.8 meV/atom, E-V 5-6 meV) because LAMMPS computes energy from the curve
   *  VALUES, and a value-only fit drives them to ~value-LS accuracy. ANY
   *  nonzero weight degrades the crystal sharply (measured: 0.01 -> -15 meV,
   *  0.1 -> -69 meV, 1.0 -> +37 meV cohesive) because the relative
   *  normalisation lets even a 1% derivative term trade away the large absolute
   *  VALUE error of a high-magnitude function like F. So raise it ONLY when you
   *  want the 1st/2nd-derivative SHAPE at the EXPENSE of crystal energy. 1.0 =
   *  value/d1/d2 equal (the original, crystal-wrecking curve mode). Ignored
   *  unless objective == Curve. */
  double curve_deriv_weight = 0.0;

  // ---- knot-family ladder schedule (one-sided kinds ONLY; the blip path is
  //      untouched — bit-for-bit regression-gated) ----
  // Full explore+refine every K-th rung; intermediate rungs grow by
  // energy-aware insertion + linear solve only (the pilot's measured-strong
  // insertion-only baseline), with one final full optimisation of the
  // envelope best. Rationale: the per-rung optimiser falls into coalescence
  // basins whose repair leaves the ENERGY metric flat — 6 such rungs fired
  // the stall counter at n~9-14 and under-resolved curves broke Fe C11 /
  // Ta C44 in the LAMMPS gate.
  int opt_every = 1;    ///< 1 = optimise every rung (legacy / blip behaviour)
  int insert_topk = 1;  ///< try the top-k separated residual peaks, keep the
                        ///< METRIC-best insertion (curve peak != energy peak)
  bool merge_guard = true;  ///< coalescence guard (one-sided knot families):
                            ///< the soft anti-coalescence penalty during EXPLORE
                            ///< AND the post-REFINE merge_close_knots repair.
                            ///< false => knots may coalesce freely (no penalty,
                            ///< no merge). REFIT_COMPACT_MERGE. Default on
                            ///< (blip path is unaffected — it has no knots).
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
   *  one formatted line per ladder event (seed, rung, merge, final refine).
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
  // refine (high-precision local):
  //   CERES_LM (joint c,mu,eta analytic-Jacobian trust region)
  //   CERES_DOGLEG | NLOPT_LBFGS | DLIB_LBFGS (VARPRO reduced gradient) | NONE
  std::string refine_algo = "CERES_LM";
  // joint (post-ladder all-function refinement vs curve + E-V residuals):
  //   CERES_DOGLEG | CERES_LM | NONE
  std::string joint_algo = "CERES_DOGLEG";
  int joint_max_iter = 2000;   ///< joint Ceres max_num_iterations
  double joint_ftol = 1e-15;   ///< joint Ceres function_tolerance
};

/** Fits one tabulated function on [lo,hi] given dense samples (xs, ys) and
 *  optional per-sample weights (empty = 1). */
class CompactBlipFitter {
public:
  using Metric = std::function<double(const BlipFit &)>;

  /** `dys` / `d2ys` (optional): the TARGET function's ANALYTIC 1st / 2nd
   *  derivatives at each xs (e.g. from the EAM cubic spline). When supplied
   *  (size == xs), the curve objective and the d1/d2 diagnostics use these
   *  exact derivatives over the FULL range. When omitted, they fall back to
   *  3-point finite differences of `ys` over the interior (noisier; used by
   *  standalone callers / tests that have no spline). */
  CompactBlipFitter(std::vector<double> xs, std::vector<double> ys,
                    std::vector<double> wts, double lo, double hi,
                    const CompactConfig &cfg,
                    BasisKind kind = BasisKind::Blip,
                    std::vector<double> dys = {},
                    std::vector<double> d2ys = {});

  /** Run the ladder. `metric` maps a candidate fit to the normalised energy
   *  error (the fitter treats it as a black box). Returns the monotone-envelope
   *  best; `trace` receives one row per rung.
   *  `init` (optional): an INIT-stage seed (resolved knot/blip positions with
   *  per-basis pin flags). When given, rung 0 optimises THAT basis (the EXPLORE
   *  stage's global optimiser, absorbing the old GLOBAL stage) instead of the
   *  uniform n_min seed, and the ladder grows from it. Pinned positions are
   *  held fixed throughout. nullptr => the legacy uniform seed (bit-for-bit). */
  BlipFit fit(const Metric &metric, std::vector<LadderRow> &trace,
              const BlipFit *init = nullptr);

  /** Closed-form linear coefficients for fixed (mu, eta); returns curve rmse.
   *  Exposed for tests. */
  double solve_coeffs(BlipFit &f) const;

  /** Ceres trust-region refine of (c, mu, eta) jointly, analytic Jacobian
   *  (LM, or DOGLEG when cfg.refine_algo == CERES_DOGLEG); returns final
   *  curve rmse (also refreshes f.curve_rmse). Exposed for tests. */
  double refine(BlipFit &f) const;

  /** Polish with the backend selected by cfg.refine_algo (CERES_* | the
   *  VARPRO reduced-gradient NLOPT_LBFGS / DLIB_LBFGS | NONE). */
  double refine_with_cfg(BlipFit &f) const { return refine_dispatch(f); }

  /** VARPRO reduced objective (mean squared error over samples) and, when
   *  grad is non-null, its EXACT gradient — Golub-Pereyra: at the inner
   *  least-squares optimum d(SSE)/dtheta = 2 r^T (dPhi/dtheta) c. Gradient
   *  layout: (mu..., eta...) for Blip; (mu...) only for the one-sided knot
   *  families (eta is a gauge, pinned). Exposed for tests. */
  double reduced_mse(BlipFit &f, std::vector<double> *grad) const;

  /** Coalescence guard for the knot families: merge knots closer than
   *  1e-3*span to their midpoint and re-solve coefficients (no-op for
   *  blips, or when cfg.merge_guard is false). Called after every refine in
   *  the ladder. Returns the number of knots removed by merging (0 if none).
   *  Exposed for tests. */
  int merge_close_knots(BlipFit &f) const;

  /** RMSE of the fit's analytic f' / f'' against the TARGET derivatives — the
   *  EAM cubic spline's exact f' / f'' when the ctor was given them, else a
   *  finite-difference fallback on the sample grid. d1 = slope fidelity, d2 =
   *  curvature fidelity (the KnotMix double-win criterion and the progress log
   *  use d2). Both diagnostics, cheap (O(N n)); neither drives rung selection.
   *  Exposed for tests. */
  double d1_rmse(const BlipFit &f) const;
  double d2_rmse(const BlipFit &f) const;

  /** Normalised combined curve score used as the DECISION objective in
   *  RefitObjective::Curve mode: sqrt(mean of the squared, per-block-normalised
   *  value / d1 / d2 RMSEs) — each of f.curve_rmse, d1_rmse(f), d2_rmse(f)
   *  divided by the tabulated function's own RMS of value / d1 / d2 over the
   *  window (a flat block, RMS ~ 0, is dropped). Dimensionless, lower = better.
   *  Exposed for tests. */
  double combined_curve_score(const BlipFit &f) const;

private:
  double sbplx_explore(BlipFit &f, int maxeval) const;  ///< VARPRO objective
  double explore(BlipFit &f) const;        ///< dispatch on cfg_.explore_algo
  double refine_dispatch(BlipFit &f) const;
  BlipFit seed_mlsl() const;               ///< uniform n_min seed -> seed_global
  /** Run the global structure search (cfg_.seed_algo) on a given starting fit,
   *  honouring its pin mask via the collapsed theta box. Used by seed_mlsl()
   *  (uniform start) and by fit()'s INIT-seeded rung 0. */
  BlipFit seed_global(BlipFit f) const;
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

  // ---- RefitObjective::Curve support (computed once in the ctor) ----
  // Finite-difference value derivatives of the TARGET data on the sample grid
  // (same 3-point nonuniform formulas as d1_rmse/d2_rmse), the per-block
  // normalisers (RMS of value / d1 / d2 over the window), and per-block
  // include flags (a numerically-flat block, RMS ~ 0, is excluded from both
  // the stacked solve and the combined score so it cannot blow up).
  std::vector<double> dy_, d2y_;   ///< FD 1st/2nd deriv of ys_ at interior pts
  std::vector<char> fd_ok_;        ///< 1 where dy_/d2y_ are valid (interior)
  double ny_ = 1.0, nd1_ = 1.0, nd2_ = 1.0;
  bool use_d1_ = false, use_d2_ = false;  ///< block has non-flat target

  // Last termination tag for each stage (e.g. "FTOL_REACHED",
  // "MAXEVAL_REACHED", "CONVERGENCE(12it)"). Set by the stage functions,
  // read by fit() for the per-stage convergence log. mutable: the stage
  // functions are const.
  mutable std::string last_seed_term_;
  mutable std::string last_explore_term_;
  mutable std::string last_refine_term_;
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
  std::vector<double> dys, d2ys;    // analytic target 1st/2nd derivs (optional)
};
double joint_refine(BlipFit &phi, BlipFit &rho, BlipFit &F,
                    const JointCurveData &dphi, const JointCurveData &drho,
                    const JointCurveData &dF,
                    const std::vector<std::vector<double>> &dists,
                    const std::vector<double> &e_orig,
                    const std::vector<double> &e_weight,
                    double curve_weight, int max_iter, double ftol,
                    const std::string &algo = "CERES_LM",
                    std::string *term_out = nullptr);
                    // algo: CERES_LM | CERES_DOGLEG | NONE
                    // (basis families are carried by each BlipFit's `kind`)
                    // term_out (optional): receives the Ceres termination tag
                    // (e.g. "CONVERGENCE(143it, cost 2.1e-09)")

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

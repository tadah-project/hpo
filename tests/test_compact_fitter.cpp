#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include <tadah/hpo/refit/compact_fitter.h>
#include <tadah/models/descriptors/d_basis_functions.h>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tadah::hpo::refit;

namespace {
// ground-truth blip set
BlipFit make_truth() {
  BlipFit t;
  t.mu = {1.5, 2.4, 3.6, 4.8};
  t.eta = {1.2, 0.9, 0.8, 1.1};
  t.c = {2.0, -1.3, 0.7, -0.4};
  return t;
}

void sample(const BlipFit &f, double lo, double hi, int N,
            std::vector<double> &xs, std::vector<double> &ys) {
  xs.resize(N);
  ys.resize(N);
  for (int i = 0; i < N; ++i) {
    xs[i] = lo + (hi - lo) * i / (N - 1);
    ys[i] = f.eval(xs[i]);
  }
}
}  // namespace

TEST_CASE("compact fitter: the LBFGS REFINE budget is 50 per MAXEVAL "
          "iteration, computed without overflow", "[refit][compact][budget]")
{
  /* (Before <ceres/ceres.h>, whose miniglog CHECK would replace Catch's.)
   * The REFINE L-BFGS budget is 50 per REFINE MAXEVAL/MAXITER iteration:
   * NLopt's evaluation budget for NLOPT_LBFGS, dlib's L-BFGS iteration cap
   * for DLIB_LBFGS. The product was taken in int: REFINE MAXEVAL 42949673
   * (and anything up to the parser's 2147483647) overflowed, which is
   * undefined behaviour (in practice a negative budget, which NLopt reads
   * as no limit). It is taken in 64 bits and capped at INT_MAX, the most
   * NLopt counts. */
  constexpr int kIntMax = std::numeric_limits<int>::max();
  CHECK(lbfgs_refine_budget(1) == 50);
  CHECK(lbfgs_refine_budget(1000) == 50000);
  CHECK(lbfgs_refine_budget(42949672) == 2147483600);
  CHECK(lbfgs_refine_budget(42949673) == kIntMax);
  CHECK(lbfgs_refine_budget(1000000000) == kIntMax);
  CHECK(lbfgs_refine_budget(kIntMax) == kIntMax);
  /* A cap below 1, which RefitParser does not accept, has no budget: a 0
   * budget is no limit to NLopt and breaks dlib's precondition (its stop
   * strategy asserts max_iter > 0), so it is an error, not a 0. */
  CHECK_THROWS_WITH(lbfgs_refine_budget(0),
                    "lbfgs_refine_budget: the REFINE iteration cap "
                    "(CompactConfig::ceres_max_iter) must be at least 1, got 0");
  CHECK_THROWS_WITH(lbfgs_refine_budget(-5),
                    "lbfgs_refine_budget: the REFINE iteration cap "
                    "(CompactConfig::ceres_max_iter) must be at least 1, got -5");
  CHECK_THROWS_AS(lbfgs_refine_budget(0), std::invalid_argument);
}

#include <ceres/ceres.h>

namespace {
// FD audit with a significance floor: components are compared in ABSOLUTE
// terms against the Jacobian's own scale. (A pure relative criterion
// false-positives on the numerically-zero support-edge entries of a compact
// basis, where FD straddles the blip support boundary.)
void audit_jacobian(ceres::CostFunction *cost,
                    std::vector<std::vector<double>> params) {
  const int M = cost->num_residuals();
  const auto &sizes = cost->parameter_block_sizes();
  const int nb = static_cast<int>(sizes.size());
  std::vector<double *> p(nb);
  for (int b = 0; b < nb; ++b) p[b] = params[b].data();

  std::vector<std::vector<double>> jac(nb);
  std::vector<double *> jp(nb);
  for (int b = 0; b < nb; ++b) {
    jac[b].assign(static_cast<std::size_t>(M) * sizes[b], 0.0);
    jp[b] = jac[b].data();
  }
  std::vector<double> r0(M);
  REQUIRE(cost->Evaluate(p.data(), r0.data(), jp.data()));

  double scale = 0.0;
  for (int b = 0; b < nb; ++b)
    for (double v : jac[b]) scale = std::max(scale, std::fabs(v));
  REQUIRE(scale > 0.0);

  std::vector<double> rp(M), rm(M);
  double worst = 0.0;
  for (int b = 0; b < nb; ++b)
    for (int k = 0; k < sizes[b]; ++k) {
      const double h = 1e-6 * std::max(1.0, std::fabs(params[b][k]));
      const double save = params[b][k];
      params[b][k] = save + h;
      REQUIRE(cost->Evaluate(p.data(), rp.data(), nullptr));
      params[b][k] = save - h;
      REQUIRE(cost->Evaluate(p.data(), rm.data(), nullptr));
      params[b][k] = save;
      for (int i = 0; i < M; ++i) {
        const double num = (rp[i] - rm[i]) / (2 * h);
        const double diff = std::fabs(jac[b][i * sizes[b] + k] - num);
        worst = std::max(worst, diff / scale);
      }
    }
  INFO("worst |analytic - FD| / scale = " << worst);
  REQUIRE(worst < 1e-6);
}
}  // namespace

TEST_CASE("compact fitter: curve Jacobian matches finite differences",
          "[compact]") {
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 160, xs, ys);

  BlipFit f = truth;
  std::mt19937 rng(7);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.05 * g(rng);
    f.eta[k] *= std::exp(0.05 * g(rng));
  }
  std::unique_ptr<ceres::CostFunction> cost(
      make_curve_cost(xs, ys, {}, f.n()));
  audit_jacobian(cost.get(), {f.c, f.mu, f.eta});
}

TEST_CASE("compact fitter: energy Jacobian matches finite differences",
          "[compact]") {
  BlipFit phi = make_truth();
  BlipFit rho;
  rho.mu = {2.0, 3.2, 4.4};
  rho.eta = {0.9, 0.8, 0.9};
  rho.c = {0.6, 0.35, 0.15};
  BlipFit F;
  F.mu = {1.0, 4.0, 7.0};
  F.eta = {0.45, 0.4, 0.45};
  F.c = {-2.0, -0.8, 0.5};

  std::vector<std::vector<double>> dists = {
      {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
      {2.1, 2.1, 2.45, 2.45, 3.0}};
  std::vector<double> e0 = {0.0, 0.0, 0.0};
  std::vector<double> ew = {1.0, 2.0, 0.5};

  std::unique_ptr<ceres::CostFunction> cost(make_energy_cost(
      dists, e0, ew, phi.n(), rho.n(), F.n()));
  audit_jacobian(cost.get(), {phi.c, phi.mu, phi.eta, rho.c, rho.mu, rho.eta,
                              F.c, F.mu, F.eta});
}

TEST_CASE("compact fitter: VARPRO reduced gradient matches finite differences",
          "[compact]") {
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 220, xs, ys);
  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);

  BlipFit f = truth;
  std::mt19937 rng(13);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.07 * g(rng);
    f.eta[k] *= std::exp(0.07 * g(rng));
  }
  std::vector<double> grad;
  fit.reduced_mse(f, &grad);
  double scale = 0.0;
  for (double v : grad) scale = std::max(scale, std::fabs(v));
  REQUIRE(scale > 0.0);

  const int n = f.n();
  double worst = 0.0;
  for (int i = 0; i < 2 * n; ++i) {
    auto &p = (i < n) ? f.mu[i] : f.eta[i - n];
    const double h = 1e-6 * std::max(1.0, std::fabs(p));
    const double save = p;
    p = save + h;
    const double fp = fit.reduced_mse(f, nullptr);
    p = save - h;
    const double fm = fit.reduced_mse(f, nullptr);
    p = save;
    worst = std::max(worst, std::fabs(grad[i] - (fp - fm) / (2 * h)) / scale);
  }
  INFO("worst |grad - FD| / scale = " << worst);
  // Golub-Pereyra is exact at the inner optimum; the FD probe re-solves c at
  // each step so tiny rank-related noise enters — floor accordingly
  REQUIRE(worst < 1e-5);
}

TEST_CASE("compact fitter: alternative refine backends recover blips",
          "[compact]") {
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);

  for (const char *algo :
       {"CERES_LM", "CERES_DOGLEG", "NLOPT_LBFGS", "DLIB_LBFGS"}) {
    CompactConfig cfg;
    cfg.refine_algo = algo;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    BlipFit f = truth;
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int k = 0; k < f.n(); ++k) {
      f.mu[k] += 0.05 * g(rng);
      f.eta[k] *= std::exp(0.05 * g(rng));
    }
    fit.solve_coeffs(f);
    const double before = f.curve_rmse;
    const double after = fit.refine_with_cfg(f);
    INFO("refine algo = " << algo << " before = " << before
                          << " after = " << after);
    REQUIRE(after < before);
    REQUIRE(after < 1e-7);  // representable problem + exact gradients
  }
}

TEST_CASE("compact fitter: Ceres refine recovers exact blips to ~noise",
          "[compact]") {
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);

  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);

  BlipFit f = truth;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.08 * g(rng);
    f.eta[k] *= std::exp(0.08 * g(rng));
  }
  fit.solve_coeffs(f);
  const double before = f.curve_rmse;
  const double after = fit.refine(f);
  REQUIRE(after < before);
  REQUIRE(after < 1e-10);  // exact-recovery: representable target
}

TEST_CASE("compact fitter: ladder reaches target on a smooth function",
          "[compact]") {
  // Morse-like curve — not exactly representable, so the ladder must work
  std::vector<double> xs(500), ys(500);
  for (int i = 0; i < 500; ++i) {
    const double r = 1.0 + 4.5 * i / 499.0;
    xs[i] = r;
    const double e = std::exp(-1.4 * (r - 2.2));
    ys[i] = (e * e - 2.0 * e);
  }
  CompactConfig cfg;
  cfg.n_max = 16;
  cfg.sbplx_maxeval = 4000;   // keep the unit test quick
  cfg.mlsl_maxeval = 8000;
  cfg.target = 1.0;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);

  // metric: curve rmse in units of 1e-4 (a uniform-grid fit of this Morse at
  // n=14 manages only ~1e-2 — the ladder must beat it by two orders)
  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-4; }, trace);

  REQUIRE(best.metric <= 1.0);            // rmse <= 1e-4 reached
  REQUIRE(best.n() <= cfg.n_max);
  REQUIRE(!trace.empty());
  // monotone envelope: every kept row improves the metric
  double prev = 1e300;
  for (const auto &row : trace)
    if (row.kept) {
      REQUIRE(row.metric < prev);
      prev = row.metric;
    }
}

TEST_CASE("compact fitter: joint refine drives E-V residuals to ~noise",
          "[compact]") {
  // synthetic EAM from known blip sets; the scan energies are computed from
  // the truth, parameters are perturbed, and joint_refine must recover them
  BlipFit phi = make_truth();
  BlipFit rho;
  rho.mu = {2.0, 3.2, 4.4};
  rho.eta = {0.9, 0.8, 0.9};
  rho.c = {0.6, 0.35, 0.15};
  BlipFit F;
  F.mu = {1.0, 4.0, 7.0};
  F.eta = {0.45, 0.4, 0.45};
  F.c = {-2.0, -0.8, 0.5};

  // toy neighbour shells at three compressions
  std::vector<std::vector<double>> dists = {
      {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
      {2.1, 2.1, 2.45, 2.45, 3.0}};
  std::vector<double> e0, ew(dists.size(), 1.0);
  for (const auto &dd : dists) {
    double hp = 0.0, ri = 0.0;
    for (double r : dd) {
      hp += 0.5 * phi.eval(r);
      ri += rho.eval(r);
    }
    e0.push_back(hp + F.eval(ri));
  }

  // curve data from the truth
  JointCurveData dphi, drho, dF;
  sample(phi, 1.0, 5.5, 300, dphi.xs, dphi.ys);
  sample(rho, 1.0, 5.5, 300, drho.xs, drho.ys);
  sample(F, 0.0, 9.0, 300, dF.xs, dF.ys);

  std::mt19937 rng(23);
  std::normal_distribution<double> g(0.0, 1.0);
  auto perturb = [&](BlipFit &f) {
    for (int k = 0; k < f.n(); ++k) {
      f.mu[k] += 0.04 * g(rng);
      f.eta[k] *= std::exp(0.04 * g(rng));
      f.c[k] *= 1.0 + 0.04 * g(rng);
    }
  };
  perturb(phi);
  perturb(rho);
  perturb(F);

  const double final_max =
      joint_refine(phi, rho, F, dphi, drho, dF, dists, e0, ew,
                   /*curve_weight=*/1.0, /*max_iter=*/500, /*ftol=*/1e-15);
  REQUIRE(final_max < 1e-9);
}


// ===================== one-sided quintic knot family =====================

namespace {
// ground-truth knot set (left-active Mendelev/Ackland form):
// f(x) = sum_j b_j (t_j - x)_+^5, eta pinned at 1
BlipFit make_knot_truth() {
  BlipFit t;
  t.kind = BasisKind::Knot5L;
  t.mu = {2.2, 3.0, 4.1, 5.2};
  t.eta = {1.0, 1.0, 1.0, 1.0};
  t.c = {0.8, -0.45, 0.12, -0.03};
  return t;
}
}  // namespace

TEST_CASE("compact fitter: knot curve Jacobian matches finite differences",
          "[compact][knot5]") {
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 160, xs, ys);

  BlipFit f = truth;
  std::mt19937 rng(7);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) f.mu[k] += 0.05 * g(rng);
  std::unique_ptr<ceres::CostFunction> cost(
      make_curve_cost(xs, ys, {}, f.n(), BasisKind::Knot5L));
  audit_jacobian(cost.get(), {f.c, f.mu, f.eta});
}

TEST_CASE("compact fitter: knot energy Jacobian matches finite differences",
          "[compact][knot5]") {
  BlipFit phi = make_knot_truth();
  BlipFit rho;
  rho.kind = BasisKind::Knot5L;
  rho.mu = {2.8, 3.9, 5.0};
  rho.eta = {1.0, 1.0, 1.0};
  rho.c = {0.05, 0.03, 0.01};
  BlipFit F;
  F.kind = BasisKind::Knot5R;
  F.mu = {-1.0, 0.0, 0.4};   // below-window knots supply the polynomial span
  F.eta = {1.0, 1.0, 1.0};
  F.c = {-0.4, -0.2, 0.1};

  std::vector<std::vector<double>> dists = {
      {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
      {2.1, 2.1, 2.45, 2.45, 3.0}};
  std::vector<double> e0 = {0.0, 0.0, 0.0};
  std::vector<double> ew = {1.0, 2.0, 0.5};

  std::unique_ptr<ceres::CostFunction> cost(make_energy_cost(
      dists, e0, ew, phi.n(), rho.n(), F.n(), BasisKind::Knot5L,
      BasisKind::Knot5L, BasisKind::Knot5R));
  audit_jacobian(cost.get(), {phi.c, phi.mu, phi.eta, rho.c, rho.mu, rho.eta,
                              F.c, F.mu, F.eta});
}

TEST_CASE("compact fitter: knot refine recovers exact knots (eta pinned)",
          "[compact][knot5]") {
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);

  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  BlipFit f = truth;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) f.mu[k] += 0.05 * g(rng);
  fit.solve_coeffs(f);
  const double before = f.curve_rmse;
  const double after = fit.refine(f);
  REQUIRE(after < before);
  REQUIRE(after < 1e-10);  // representable target, exact gradients
  for (int k = 0; k < f.n(); ++k) {
    REQUIRE(f.eta[k] == 1.0);          // gauge stayed pinned
    REQUIRE(f.mu[k] <= 5.5 + 1e-12);   // knots never crossed the cutoff
  }
}

TEST_CASE("compact fitter: knot VARPRO gradient matches finite differences",
          "[compact][knot5]") {
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 220, xs, ys);
  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  BlipFit f = truth;
  std::mt19937 rng(13);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) f.mu[k] += 0.04 * g(rng);

  std::vector<double> grad;
  fit.reduced_mse(f, &grad);
  REQUIRE(grad.size() == static_cast<std::size_t>(f.n()));  // mu-only space

  double scale = 0.0;
  for (double v : grad) scale = std::max(scale, std::fabs(v));
  REQUIRE(scale > 0.0);
  double worst = 0.0;
  const double h = 1e-6;
  for (int k = 0; k < f.n(); ++k) {
    BlipFit fp = f, fm = f;
    fp.mu[k] += h;
    fm.mu[k] -= h;
    const double vp = fit.reduced_mse(fp, nullptr);
    const double vm = fit.reduced_mse(fm, nullptr);
    worst = std::max(worst,
                     std::fabs(grad[k] - (vp - vm) / (2 * h)) / scale);
  }
  INFO("worst |grad - FD| / scale = " << worst);
  REQUIRE(worst < 1e-5);
}

TEST_CASE("compact fitter: merge_close_knots collapses coalesced pairs",
          "[compact][knot5]") {
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 300, xs, ys);
  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  // a coalesced impostor pair: two knots 1e-5 apart with huge +-b
  BlipFit f = truth;
  f.mu.push_back(3.60);     f.eta.push_back(1.0); f.c.push_back(500.0);
  f.mu.push_back(3.600009); f.eta.push_back(1.0); f.c.push_back(-500.0);
  const int n_before = f.n();
  const int merged = fit.merge_close_knots(f);
  REQUIRE(merged == 1);                    // returns the merge count
  REQUIRE(f.n() == n_before);              // merged + re-inserted elsewhere
  std::vector<double> ts(f.mu.begin(), f.mu.end());
  std::sort(ts.begin(), ts.end());
  for (std::size_t i = 1; i < ts.size(); ++i)
    REQUIRE(ts[i] - ts[i - 1] >= 1e-3 * 4.5);  // no coalesced pair survives
  REQUIRE(f.curve_rmse < 1e-10);           // truth still representable

  // blips are untouched by the guard
  BlipFit b = make_truth();
  CompactBlipFitter bfit(xs, ys, {}, 1.0, 5.5, cfg);
  const int bn = b.n();
  REQUIRE(bfit.merge_close_knots(b) == 0);
  REQUIRE(b.n() == bn);

  // REFIT_COMPACT_MERGE false disables the guard entirely: the coalesced pair
  // survives untouched and the routine reports zero merges.
  CompactConfig cfg_off = cfg;
  cfg_off.merge_guard = false;
  CompactBlipFitter fit_off(xs, ys, {}, 1.0, 5.5, cfg_off, BasisKind::Knot5L);
  BlipFit g = truth;
  g.mu.push_back(3.60);     g.eta.push_back(1.0); g.c.push_back(500.0);
  g.mu.push_back(3.600009); g.eta.push_back(1.0); g.c.push_back(-500.0);
  const int gn = g.n();
  REQUIRE(fit_off.merge_close_knots(g) == 0);  // guard off => no-op
  REQUIRE(g.n() == gn);
  REQUIRE(g.mu.back() == 3.600009);            // the coalesced knot is kept
}

TEST_CASE("compact fitter: curve objective fits derivatives (stacked solve)",
          "[compact][curve]") {
  // A smooth, NOT-in-class target (a Gaussian), under-resolved by a fixed knot
  // set so neither solve is exact — then the value-only (Energy) solve and the
  // stacked value+d1+d2 (Curve) solve differ, and Curve trades a little value
  // RMSE for markedly lower curvature (d2) error.
  std::vector<double> xs, ys;
  const int N = 200;
  for (int i = 0; i < N; ++i) {
    const double x = 1.0 + (5.5 - 1.0) * i / (N - 1);
    xs.push_back(x);
    ys.push_back(std::exp(-(x - 3.0) * (x - 3.0)));
  }
  BlipFit knots;
  knots.kind = BasisKind::Knot5L;
  for (double m : {1.5, 2.5, 3.5, 4.5}) {
    knots.mu.push_back(m); knots.eta.push_back(1.0); knots.c.push_back(0.0);
  }

  CompactConfig ce;  // Energy = value-only solve (the default)
  CompactBlipFitter fe(xs, ys, {}, 1.0, 5.5, ce, BasisKind::Knot5L);
  BlipFit ev = knots; fe.solve_coeffs(ev);

  // exercise the STACKED value+d1+d2 solve: curve_deriv_weight > 0 (the default
  // is 0 = pure value, which is deliberately identical to the energy value-LS).
  CompactConfig cc; cc.objective = RefitObjective::Curve;
  cc.curve_deriv_weight = 1.0;
  CompactBlipFitter fc(xs, ys, {}, 1.0, 5.5, cc, BasisKind::Knot5L);
  BlipFit cv = knots; fc.solve_coeffs(cv);

  // the two objectives produce genuinely different coefficients
  double dmax = 0.0;
  for (int k = 0; k < knots.n(); ++k)
    dmax = std::max(dmax, std::fabs(cv.c[k] - ev.c[k]));
  REQUIRE(dmax > 1e-6);

  // and the DEFAULT curve weight (0) reproduces the value-only solve exactly
  // (this is the crystal-faithful default — pure value fit)
  CompactConfig cc0; cc0.objective = RefitObjective::Curve;  // deriv_weight=0
  CompactBlipFitter fc0(xs, ys, {}, 1.0, 5.5, cc0, BasisKind::Knot5L);
  BlipFit cv0 = knots; fc0.solve_coeffs(cv0);
  for (int k = 0; k < knots.n(); ++k)
    REQUIRE(cv0.c[k] == Approx(ev.c[k]).margin(1e-9));

  // value-only minimises VALUE rmse (so it is <= the curve solve's value rmse)
  REQUIRE(ev.curve_rmse <= cv.curve_rmse + 1e-10);
  // ...and the curve solve fits the 2nd derivative strictly better
  REQUIRE(fc.d2_rmse(cv) < fe.d2_rmse(ev));
  // combined score is a positive, finite dimensionless number
  const double s = fc.combined_curve_score(cv);
  REQUIRE(s > 0.0);
  REQUIRE(std::isfinite(s));
}

TEST_CASE("compact fitter: curve-objective ladder runs and lowers d2",
          "[compact][curve]") {
  // full ladder smoke test in curve mode: it must grow and reduce the curvature
  // error vs the rung-0 seed, with the energy metric (a dummy here) unused.
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 300, xs, ys);
  CompactConfig cfg;
  cfg.objective = RefitObjective::Curve;
  cfg.n_max = 8;
  cfg.stall_rungs = 6;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);
  std::vector<LadderRow> trace;
  // dummy energy metric: must be IGNORED in curve mode (return a constant)
  BlipFit best = fit.fit([](const BlipFit &) { return 42.0; }, trace);
  REQUIRE(best.n() >= 2);
  REQUIRE(best.metric != 42.0);              // decisions used the curve score
  REQUIRE(best.metric == Approx(fit.combined_curve_score(best)));
  REQUIRE(fit.combined_curve_score(best) > 0.0);
  REQUIRE(std::isfinite(best.curve_rmse));
}

TEST_CASE("compact fitter: knot ladder recovers an in-class knotted target",
          "[compact][knot5]") {
  // The left-active knot class is exactly the functions vanishing C^4-
  // smoothly at the last knot (<= hi) — the EAM phi/rho convention. A target
  // that does NOT vanish at hi (e.g. a raw Morse) is OUTSIDE the class, so
  // the decisive ladder test is exact recovery of an in-class knotted
  // quintic (the pilot's headline result, 4-knot scale).
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 500, xs, ys);

  CompactConfig cfg;
  cfg.n_max = 10;
  cfg.sbplx_maxeval = 4000;
  cfg.mlsl_maxeval = 8000;
  cfg.target = 1.0;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-8; }, trace);

  INFO("recovered n = " << best.n() << " curve_rmse = " << best.curve_rmse);
  REQUIRE(best.kind == BasisKind::Knot5L);
  REQUIRE(best.metric <= 1.0);            // rmse <= 1e-8: exact-class recovery
  REQUIRE(best.n() <= cfg.n_max);
  for (int k = 0; k < best.n(); ++k) {
    REQUIRE(best.eta[k] == 1.0);
    REQUIRE(best.mu[k] <= 5.5 + 1e-12);
  }
}

TEST_CASE("compact fitter: knot joint refine drives E-V residuals to ~noise",
          "[compact][knot5]") {
  BlipFit phi = make_knot_truth();
  BlipFit rho;
  rho.kind = BasisKind::Knot5L;
  rho.mu = {2.8, 3.9, 5.0};
  rho.eta = {1.0, 1.0, 1.0};
  rho.c = {0.05, 0.03, 0.01};
  BlipFit F;
  F.kind = BasisKind::Knot5R;
  // in-window right-active knots: below-window knots span polynomials but
  // are strongly collinear (the survey's truncated-power hazard), which
  // stalls LM short of exact 1e-9 recovery — production tolerances are
  // ~1e-4-relative, but this test asserts the exact-recovery limit
  F.mu = {0.0, 0.6, 1.2};
  F.eta = {1.0, 1.0, 1.0};
  F.c = {-0.05, -0.02, 0.01};

  std::vector<std::vector<double>> dists;
  std::vector<double> e0, ew;
  for (int i = 0; i < 12; ++i) {
    const double s = 0.8 + 0.3 * i / 11.0;
    std::vector<double> dd = {2.5 * s, 2.5 * s, 2.9 * s,
                              2.9 * s, 3.6 * s, 3.6 * s};
    dists.push_back(dd);
    double hp = 0.0, ri = 0.0;
    for (double r : dd) {
      hp += 0.5 * phi.eval(r);
      ri += rho.eval(r);
    }
    e0.push_back(hp + F.eval(ri));
    ew.push_back(1.0);
  }

  JointCurveData dphi, drho, dF;
  sample(phi, 1.0, 5.5, 300, dphi.xs, dphi.ys);
  sample(rho, 1.0, 5.5, 300, drho.xs, drho.ys);
  sample(F, 0.0, 2.0, 300, dF.xs, dF.ys);

  std::mt19937 rng(23);
  std::normal_distribution<double> g(0.0, 1.0);
  auto perturb = [&](BlipFit &f) {
    for (int k = 0; k < f.n(); ++k) {
      f.mu[k] += 0.02 * g(rng);
      f.c[k] *= 1.0 + 0.02 * g(rng);
    }
  };
  perturb(phi);
  perturb(rho);
  perturb(F);

  const double final_max =
      joint_refine(phi, rho, F, dphi, drho, dF, dists, e0, ew,
                   /*curve_weight=*/1.0, /*max_iter=*/2000, /*ftol=*/1e-15);
  REQUIRE(final_max < 1e-9);
  for (int k = 0; k < phi.n(); ++k) REQUIRE(phi.eta[k] == 1.0);
}

// ===================== quintic B-spline bump (Blip5) =====================

TEST_CASE("compact fitter: Blip5 refine recovers exact quintic bumps",
          "[compact][blip5]") {
  BlipFit truth;
  truth.kind = BasisKind::Blip5;
  truth.mu = {1.5, 2.4, 3.6, 4.8};
  truth.eta = {1.2, 0.9, 0.8, 1.1};
  truth.c = {2.0, -1.3, 0.7, -0.4};
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);

  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Blip5);

  BlipFit f = truth;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.05 * g(rng);
    f.eta[k] *= std::exp(0.05 * g(rng));
  }
  fit.solve_coeffs(f);
  const double before = f.curve_rmse;
  const double after = fit.refine(f);
  REQUIRE(after < before);
  REQUIRE(after < 1e-10);  // representable target, exact gradients (C^4!)
}

TEST_CASE("compact fitter: Blip5 ladder reaches target on a smooth function",
          "[compact][blip5]") {
  // same Morse target as the Blip ladder test — Blip5 keeps full bump
  // semantics (free mu AND eta), so the smooth-target test applies
  std::vector<double> xs(500), ys(500);
  for (int i = 0; i < 500; ++i) {
    const double r = 1.0 + 4.5 * i / 499.0;
    xs[i] = r;
    const double e = std::exp(-1.4 * (r - 2.2));
    ys[i] = (e * e - 2.0 * e);
  }
  CompactConfig cfg;
  cfg.n_max = 16;
  cfg.sbplx_maxeval = 4000;
  cfg.mlsl_maxeval = 8000;
  cfg.target = 1.0;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Blip5);

  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-4; }, trace);

  INFO("Blip5 ladder: n = " << best.n() << " rmse = " << best.curve_rmse);
  REQUIRE(best.kind == BasisKind::Blip5);
  REQUIRE(best.metric <= 1.0);
  REQUIRE(best.n() <= cfg.n_max);
}

// =================== mixed cubic/quintic knot atoms (KnotMix) ===================

TEST_CASE("compact fitter: KnotMix ladder recovers a mixed-degree truth",
          "[compact][knotmix]") {
  // truth: quintic background + genuine f-triple-prime corners (cubic atoms)
  // — the Mendelev/Ackland structure the auto-mode exists for
  BlipFit truth;
  truth.kind = BasisKind::Knot5L;
  truth.mu  = {2.4, 3.2, 4.0, 4.9, 5.4};
  truth.eta = {1.0, 1.0, 1.0, 1.0, 1.0};
  truth.deg = {5, 3, 5, 3, 5};
  truth.c   = {0.5, -0.9, 0.08, 0.6, -0.02};
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 600, xs, ys);

  CompactConfig cfg;
  cfg.n_max = 12;
  cfg.sbplx_maxeval = 4000;
  cfg.mlsl_maxeval = 8000;
  cfg.target = 1.0;
  cfg.knot_mix = true;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-7; }, trace);

  INFO("KnotMix: n = " << best.n() << " rmse = " << best.curve_rmse);
  REQUIRE(best.metric <= 1.0);   // rmse <= 1e-7: the mix found the corners
  int n_cubic = 0;
  for (int k = 0; k < best.n(); ++k)
    if (best.degree(k) == 3) ++n_cubic;
  INFO("cubic atoms used: " << n_cubic);
  REQUIRE(n_cubic >= 1);         // the auto-mode actually deployed cubics
}

TEST_CASE("compact fitter: pure Knot3 refine recovers exact cubic knots",
          "[compact][knotmix]") {
  BlipFit truth;
  truth.kind = BasisKind::Knot5L;
  truth.mu  = {2.2, 3.0, 4.1, 5.2};
  truth.eta = {1.0, 1.0, 1.0, 1.0};
  truth.deg = {3, 3, 3, 3};
  truth.c   = {0.8, -0.45, 0.12, -0.03};
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);

  CompactConfig cfg;
  cfg.knot_degree = 3;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  BlipFit f = truth;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) f.mu[k] += 0.04 * g(rng);
  fit.solve_coeffs(f);
  const double before = f.curve_rmse;
  const double after = fit.refine(f);
  REQUIRE(after < before);
  REQUIRE(after < 1e-10);
  for (int k = 0; k < f.n(); ++k) REQUIRE(f.degree(k) == 3);
}

TEST_CASE("compact fitter: double-win detector rejects spurious cubics",
          "[compact][knotmix]") {
  // pure-quintic truth: the metric-only v1 detector deployed cubics on
  // 40-60% of atoms here (measured) and degraded elastic constants; the
  // double-win rule (cubic only if better metric AND no worse d2) must
  // keep the mix essentially quintic on a smooth in-class target.
  const BlipFit truth = make_knot_truth();   // all degree 5
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 500, xs, ys);

  CompactConfig cfg;
  cfg.n_max = 10;
  cfg.sbplx_maxeval = 4000;
  cfg.mlsl_maxeval = 8000;
  cfg.target = 1.0;
  cfg.knot_mix = true;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-8; }, trace);

  int n_cubic = 0;
  for (int k = 0; k < best.n(); ++k)
    if (best.degree(k) == 3) ++n_cubic;
  INFO("n = " << best.n() << " cubic = " << n_cubic
              << " rmse = " << best.curve_rmse);
  REQUIRE(best.metric <= 1.0);              // still recovers the truth
  REQUIRE(n_cubic * 5 <= best.n());         // <= 20% spurious cubics
}

TEST_CASE("compact fitter: d2_rmse and deriv2 are consistent",
          "[compact][knotmix]") {
  const BlipFit truth = make_knot_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 400, xs, ys);
  CompactConfig cfg;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, BasisKind::Knot5L);

  // deriv2 vs FD of eval
  const double h = 1e-5;
  for (double x : {1.7, 2.9, 4.3, 5.1}) {
    const double fd2 =
        (truth.eval(x + h) - 2 * truth.eval(x) + truth.eval(x - h)) / (h * h);
    REQUIRE(truth.deriv2(x) == Approx(fd2).margin(1e-4));
  }
  // the truth's own d2_rmse vs its sampled data is ~the FD discretisation
  // noise — far below any meaningful fit error
  BlipFit f = truth;
  REQUIRE(fit.d2_rmse(f) < 1e-3);
}

TEST_CASE("compact fitter: bump-support clamp projects spilling bumps",
          "[compact][bumpgate]") {
  const double rcut = 5.5, span = 4.5;  // fit window [1.0, 5.5]
  const double tol = 1e-6 * rcut;

  SECTION("free-function core: projection + eta floor") {
    // make_truth() spills TWICE at rcut 5.5: bump 3 (3.6 + 2/0.8 = 6.1)
    // and bump 4 (4.8 + 2/1.1 = 6.62)
    BlipFit f = make_truth();
    REQUIRE(clamp_bump_support_params(f, rcut, span) == 2);
    for (int k = 0; k < f.n(); ++k)
      REQUIRE(f.mu[k] + 2.0 / f.eta[k] <= rcut + tol);
    // the centres stayed (narrowest legal bumps fit there): the etas rose
    REQUIRE(f.mu[2] == Approx(3.6));
    REQUIRE(f.eta[2] == Approx(2.0 / (rcut - 3.6)));
    REQUIRE(f.mu[3] == Approx(4.8));
    REQUIRE(f.eta[3] == Approx(2.0 / (rcut - 4.8)));
    // idempotent
    REQUIRE(clamp_bump_support_params(f, rcut, span) == 0);
  }
  SECTION("centre beyond what the narrowest legal bump allows moves inward") {
    BlipFit f = make_truth();
    f.mu[3] = 5.6;                     // past rcut entirely
    REQUIRE(clamp_bump_support_params(f, rcut, span) >= 1);
    REQUIRE(f.mu[3] < rcut);
    REQUIRE(f.mu[3] + 2.0 / f.eta[3] <= rcut + tol);
  }
  SECTION("gate off / knot family: no-op") {
    BlipFit f = make_truth();
    f.eta[3] = 0.5;
    REQUIRE(clamp_bump_support_params(f, 0.0, span) == 0);
    BlipFit k;
    k.kind = BasisKind::Knot5L;
    k.mu = {6.0};
    k.eta = {1.0};
    k.c = {1.0};
    REQUIRE(clamp_bump_support_params(k, rcut, span) == 0);
  }
  SECTION("a pinned violating anchor throws with the eta hint") {
    BlipFit f = make_truth();
    f.pinned.assign(f.mu.size(), false);
    f.pinned[3] = true;  // bump 4 spills at rcut 5.5 and cannot be moved
    REQUIRE_THROWS_WITH(clamp_bump_support_params(f, rcut, span),
                        Catch::Contains("pinned INIT anchor") &&
                            Catch::Contains("--eta"));
  }
  SECTION("member clamp re-solves the coefficients") {
    const BlipFit truth = make_truth();
    std::vector<double> xs, ys;
    sample(truth, 1.0, rcut, 200, xs, ys);
    CompactConfig cfg;
    cfg.hard_rcut = rcut;
    CompactBlipFitter fit(xs, ys, {}, 1.0, rcut, cfg);
    BlipFit f = truth;                 // bumps 3 and 4 spill (see above)
    REQUIRE(fit.clamp_bump_support(f) == 2);
    REQUIRE(f.curve_rmse < 1e300);     // solve_coeffs ran
    for (int k = 0; k < f.n(); ++k)
      REQUIRE(f.mu[k] + 2.0 / f.eta[k] <= rcut + tol);
    // spill guard: 1.0 exactly on a compliant fit, > 1 when spilling
    REQUIRE(fit.bump_spill_guard(f) == Approx(1.0));
    BlipFit g = truth;
    REQUIRE(fit.bump_spill_guard(g) > 1.0);
  }
}

TEST_CASE("compact fitter: ladder under hard_rcut keeps every bump inside",
          "[compact][bumpgate]") {
  // Morse-like curve on [1.0, 5.5] with a hard cutoff at 5.5: the ladder,
  // jitters, refine and insertion may propose spilling bumps, but every
  // scored candidate is clamped — the final best must be gate-compliant.
  const double rcut = 5.5;
  std::vector<double> xs(400), ys(400);
  for (int i = 0; i < 400; ++i) {
    const double r = 1.0 + (rcut - 1.0) * i / 399.0;
    xs[i] = r;
    const double e = std::exp(-1.4 * (r - 2.2));
    ys[i] = (e * e - 2.0 * e);
  }
  CompactConfig cfg;
  cfg.n_max = 10;
  cfg.sbplx_maxeval = 2000;  // keep the unit test quick
  cfg.mlsl_maxeval = 4000;
  cfg.target = 1.0;
  cfg.hard_rcut = rcut;
  CompactBlipFitter fit(xs, ys, {}, 1.0, rcut, cfg);
  std::vector<LadderRow> trace;
  BlipFit best = fit.fit(
      [&](const BlipFit &f) { return f.curve_rmse / 1e-4; }, trace);
  REQUIRE(best.n() >= 2);
  const double tol = 1e-6 * rcut;
  for (int k = 0; k < best.n(); ++k) {
    INFO("bump " << k << ": mu=" << best.mu[k] << " eta=" << best.eta[k]);
    REQUIRE(best.mu[k] + 2.0 / best.eta[k] <= rcut + tol);
  }
  // and the gate does not wreck the fit: still a good curve
  REQUIRE(best.curve_rmse < 1e-2);
}

#endif  // TADAH_USE_CERES

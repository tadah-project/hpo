#include "catch2/catch.hpp"

#ifdef TADAH_USE_CERES

#include <tadah/core/context.h>
#include <tadah/hpo/refit/compact_fitter.h>
#include <tadah/hpo/refit/refit_parser.h>
#include <tadah/models/descriptors/basis_dispatch.h>
#include <tadah/models/descriptors/d_basis_functions.h>
#include <tadah/models/nnls.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

/* (Before <ceres/ceres.h>, whose miniglog CHECK would replace Catch's.)
 * tadah refit's MAXEVAL counts objective evaluations exactly, for every
 * optimiser but Ceres (iterations), as NLopt counts them. dlib's GLOBAL seed
 * used to take max(500, MAXEVAL/20) calls, and dlib's L-BFGS (EXPLORE INNER
 * and REFINE) capped its iterations -- in REFINE 50 x MAXEVAL of them, as
 * NLopt's REFINE L-BFGS took 50 x MAXEVAL evaluations. */
namespace {

/* A perturbed copy of the truth: a refine start that is not converged. */
BlipFit perturbed_truth() {
  BlipFit f = make_truth();
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.05 * g(rng);
    f.eta[k] *= std::exp(0.05 * g(rng));
  }
  return f;
}

/* The log lines of a ladder run with `cfg`, and its objective evaluations:
 * the seed at n_min and, with n_max = n_min + 1, one rung of one candidate
 * (no jitter), whose metric never reaches the target. */
struct LadderRun {
  std::vector<std::string> log;
  long long evaluations = 0;
};

LadderRun run_ladder(CompactConfig cfg) {
  std::vector<double> xs, ys;
  sample(make_truth(), 1.0, 5.5, 160, xs, ys);
  LadderRun out;
  cfg.log_fn = [&out](const std::string &line) { out.log.push_back(line); };
  cfg.jitters = 0;
  cfg.target = 0.0;
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
  std::vector<LadderRow> trace;
  (void)fit.fit([](const BlipFit &f) { return f.curve_rmse; }, trace);
  out.evaluations = fit.objective_evaluations();
  return out;
}

bool logged(const LadderRun &r, const std::string &what) {
  for (const auto &line : r.log)
    if (line.find(what) != std::string::npos) return true;
  return false;
}

}  // namespace

TEST_CASE("compact fitter: REFINE MAXEVAL counts the L-BFGS objective "
          "evaluations", "[refit][compact][budget]")
{
  std::vector<double> xs, ys;
  sample(make_truth(), 1.0, 5.5, 400, xs, ys);
  for (const char *algo : {"NLOPT_LBFGS", "DLIB_LBFGS"}) {
    for (const int maxeval : {1, 7, 40}) {
      INFO(algo << " with MAXEVAL " << maxeval);
      CompactConfig cfg;
      cfg.refine_algo = algo;
      cfg.lbfgs_refine_maxeval = maxeval;
      cfg.ceres_ftol = 0.0;  // no convergence test: the budget ends it
      CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
      BlipFit f = perturbed_truth();
      fit.solve_coeffs(f);
      const double before = f.curve_rmse;
      REQUIRE(fit.objective_evaluations() == 0);
      const double after = fit.refine_with_cfg(f);
      CHECK(fit.objective_evaluations() == maxeval);
      CHECK(after <= before);
    }
  }
}

TEST_CASE("compact fitter: EXPLORE MAXEVAL counts the seed's objective "
          "evaluations for every seed algorithm", "[refit][compact][budget]")
{
  /* The seed alone: n_max = n_min, no local search or refine. */
  for (const char *seed : {"DLIB_GLOBAL", "MLSL", "DIRECT", "CRS2"}) {
    INFO("seed " << seed);
    CompactConfig cfg;
    cfg.n_max = cfg.n_min;
    cfg.seed_algo = seed;
    cfg.mlsl_maxeval = 40;
    cfg.dlib_global_maxeval = 40;
    cfg.explore_algo = "NONE";
    cfg.refine_algo = "NONE";
    const LadderRun r = run_ladder(cfg);
    CHECK(r.evaluations == 40);
    CHECK(logged(r, "(40ev)"));
  }
  /* dlib's line: exactly MAXEVAL, even below its old 500-call floor. */
  CompactConfig cfg;
  cfg.n_max = cfg.n_min;
  cfg.seed_algo = "DLIB_GLOBAL";
  cfg.dlib_global_maxeval = 25;
  cfg.explore_algo = "NONE";
  cfg.refine_algo = "NONE";
  const LadderRun r = run_ladder(cfg);
  CHECK(r.evaluations == 25);
  CHECK(logged(r, "term=DLIB_GLOBAL:MAXEVAL_REACHED(25ev)"));
}

TEST_CASE("compact fitter: EXPLORE INNER MAXEVAL counts the local search's "
          "objective evaluations", "[refit][compact][budget]")
{
  /* No seed search (UNIFORM) and no refine: one rung of one candidate,
   * whose local search stops at its budget (no tolerance ends it). */
  for (const char *algo : {"DLIB_LBFGS", "NLOPT_LBFGS", "SBPLX"}) {
    INFO("inner " << algo);
    CompactConfig cfg;
    cfg.n_max = cfg.n_min + 1;
    cfg.seed_algo = "UNIFORM";
    cfg.explore_algo = algo;
    cfg.sbplx_maxeval = 9;
    cfg.sbplx_ftol = 0.0;
    cfg.sbplx_xtol = 0.0;
    cfg.refine_algo = "NONE";
    const LadderRun r = run_ladder(cfg);
    CHECK(r.evaluations == 9);
    CHECK(logged(r, "(9ev)"));
    if (std::string(algo) == "DLIB_LBFGS")
      CHECK(logged(r, "explore=DLIB_LBFGS:MAXEVAL_REACHED(9ev)"));
  }
}

TEST_CASE("compact fitter: each stage's MAXEVAL sets the budget of the "
          "optimiser that runs it", "[refit][compact][budget]")
{
  const auto parsed = [](const std::vector<std::pair<std::string,
                                                     std::string>> &blocks) {
    tadah::core::Context c;
    for (const auto &[key, body] : blocks) {
      std::istringstream iss(body);
      for (std::string t; iss >> t;) c.add(key, t);
    }
    return RefitParser::parse(c);
  };
  CompactConfig cfg;
  apply_stage_optimizers(
      parsed({{"EXPLORE", "OPTIMIZER LIB DLIB ALGO GLOBAL MAXEVAL 40 INNER "
                          "LIB DLIB ALGO LBFGS MAXEVAL 9 ENDINNER "
                          "ENDOPTIMIZER"},
              {"REFINE", "OPTIMIZER LIB DLIB ALGO LBFGS MAXITER 7 "
                         "ENDOPTIMIZER"},
              {"FUSE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 11 "
                       "ENDOPTIMIZER"}}),
      cfg);
  CHECK(cfg.seed_algo == "DLIB_GLOBAL");
  CHECK(cfg.dlib_global_maxeval == 40);
  CHECK(cfg.mlsl_maxeval == 40);
  CHECK(cfg.explore_algo == "DLIB_LBFGS");
  CHECK(cfg.sbplx_maxeval == 9);
  CHECK(cfg.refine_algo == "DLIB_LBFGS");
  CHECK(cfg.lbfgs_refine_maxeval == 7);
  CHECK(cfg.ceres_max_iter == 7);  // what the curve-mode FUSE refine reads
  CHECK(cfg.joint_algo == "CERES_LM");
  CHECK(cfg.joint_max_iter == 11);

  /* Without MAXEVAL each keeps its default. */
  CompactConfig dflt;
  apply_stage_optimizers(
      parsed({{"EXPLORE", "OPTIMIZER LIB DLIB ALGO GLOBAL ENDOPTIMIZER"},
              {"REFINE", "OPTIMIZER LIB NLOPT ALGO LD_LBFGS ENDOPTIMIZER"}}),
      dflt);
  CHECK(dflt.dlib_global_maxeval == 3000);
  CHECK(dflt.mlsl_maxeval == 60000);
  CHECK(dflt.lbfgs_refine_maxeval == 50000);
  CHECK(dflt.ceres_max_iter == 1000);

  /* ENABLED false switches a stage off whatever its optimiser. */
  CompactConfig off;
  apply_stage_optimizers(
      parsed({{"REFINE", "ENABLED false OPTIMIZER LIB CERES ALGO LM "
                         "ENDOPTIMIZER"},
              {"FUSE", "ENABLED false"}}),
      off);
  CHECK(off.refine_algo == "NONE");
  CHECK(off.joint_algo == "NONE");
}

TEST_CASE("compact fitter: a Ceres solve with options Ceres refuses throws",
          "[refit][compact]")
{
  /* Pins: refine(), refine_edges() and joint_refine() throw
   * std::invalid_argument naming the solve and Ceres' reason when Ceres
   * refuses the solver options (here a negative gradient or function
   * tolerance), and leave the fit as it was; the same solves with
   * tolerance 0 run. RefitParser refuses a negative tolerance, so only a
   * caller that sets CompactConfig itself reaches this.
   * Before the fix: Ceres returned FAILURE without iterating, the solve
   * recorded it as its stop term (FAILURE(0it, Invalid configuration...))
   * and returned without an error: no exception is thrown. */
  std::vector<double> xs, ys;
  sample(make_truth(), 1.0, 5.5, 200, xs, ys);
  /* The message of the std::invalid_argument `run` throws ("" if none). */
  const auto refusal = [](const std::function<void()> &run) -> std::string {
    try {
      run();
    } catch (const std::invalid_argument &e) {
      return e.what();
    }
    return "";
  };
  const std::string refuses = ": Ceres refuses the solver options (Invalid "
                              "configuration. Solver::Options::";
  {
    CompactConfig cfg;
    cfg.ceres_gtol = -1.0;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    BlipFit f = perturbed_truth();
    fit.solve_coeffs(f);
    const BlipFit start = f;
    CHECK(refusal([&] { fit.refine(f); }) ==
          "compact_fitter refine" + refuses + "gradient_tolerance = -1. "
          "Violated constraint: Solver::Options::gradient_tolerance >= 0.0)");
    CHECK(f.mu == start.mu);
    CHECK(f.eta == start.eta);
  }
  {
    CompactConfig cfg;
    cfg.ceres_ftol = -1.0;
    cfg.hard_rcut = 5.5;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    BlipFit f = perturbed_truth();
    fit.solve_coeffs(f);
    CHECK(refusal([&] { fit.refine_edges(f); }) ==
          "compact_fitter refine_edges" + refuses + "function_tolerance = "
          "-1. Violated constraint: Solver::Options::function_tolerance >= "
          "0.0)");
  }
  {
    CompactConfig cfg;
    cfg.ceres_ftol = 0.0;
    cfg.ceres_gtol = 0.0;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    BlipFit f = perturbed_truth();
    fit.solve_coeffs(f);
    const double before = f.curve_rmse;
    CHECK(fit.refine(f) < before);
  }
  /* joint_refine, FUSE's solve in energy mode. */
  BlipFit phi = make_truth(), rho, F;
  rho.mu = {2.0, 3.2, 4.4};
  rho.eta = {0.9, 0.8, 0.9};
  rho.c = {0.6, 0.35, 0.15};
  F.mu = {1.0, 4.0, 7.0};
  F.eta = {0.45, 0.4, 0.45};
  F.c = {-2.0, -0.8, 0.5};
  const std::vector<std::vector<double>> dists = {{2.5, 2.5, 2.9, 3.6},
                                                  {2.2, 2.2, 2.6, 3.2}};
  const std::vector<double> e0 = {0.1, 0.2}, ew(dists.size(), 1.0);
  JointCurveData dphi, drho, dF;
  sample(phi, 1.0, 5.5, 100, dphi.xs, dphi.ys);
  sample(rho, 1.0, 5.5, 100, drho.xs, drho.ys);
  sample(F, 0.0, 9.0, 100, dF.xs, dF.ys);
  for (const char *algo : {"CERES_LM", "CERES_DOGLEG"}) {
    INFO(algo);
    BlipFit p = phi, r = rho, Fc = F;
    CHECK(refusal([&] {
            joint_refine(p, r, Fc, dphi, drho, dF, dists, e0, ew, 1.0, 10,
                         -1.0, algo);
          }) == "joint_refine" + refuses + "function_tolerance = -1. "
                "Violated constraint: Solver::Options::function_tolerance "
                ">= 0.0)");
  }
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

namespace {
// The outer support edges e = mu + w/eta of a bump channel (joint_refine's
// edge parametrisation).
std::vector<double> support_edges(const BlipFit &f, double w) {
  std::vector<double> e(f.n());
  for (int k = 0; k < f.n(); ++k) e[k] = f.mu[k] + w / f.eta[k];
  return e;
}
}  // namespace

TEST_CASE("compact fitter: edge-parametrised curve and energy Jacobians match "
          "finite differences", "[compact]") {
  // joint_refine's bump-support gate optimises a radial channel in its outer
  // support edges (EdgeParamCost): d/de = d/dmu, d/deta|e = d/deta|mu +
  // (w/eta^2) d/dmu.
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 160, xs, ys);
  BlipFit f = truth;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.05 * g(rng);
    f.eta[k] *= std::exp(0.05 * g(rng));
  }
  for (const BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
    INFO("kind " << static_cast<int>(kind));
    const double w = kind == BasisKind::Blip5 ? 3.0 : 2.0;
    std::unique_ptr<ceres::CostFunction> cost(make_edge_param_cost(
        make_curve_cost(xs, ys, {}, f.n(), kind), {{1, 2, w}}));
    audit_jacobian(cost.get(), {f.c, support_edges(f, w), f.eta});

    // the eta Jacobian alone (no edge Jacobian requested) carries the chain
    std::vector<double> c = f.c, e = support_edges(f, w), eta = f.eta;
    const double *p[3] = {c.data(), e.data(), eta.data()};
    const int M = cost->num_residuals(), n = f.n();
    std::vector<double> r(M), jc(M * n), je(M * n), jeta(M * n), jeta2(M * n);
    double *all[3] = {jc.data(), je.data(), jeta.data()};
    double *only_eta[3] = {nullptr, nullptr, jeta2.data()};
    REQUIRE(cost->Evaluate(p, r.data(), all));
    REQUIRE(cost->Evaluate(p, r.data(), only_eta));
    CHECK(jeta2 == jeta);
  }

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
  std::unique_ptr<ceres::CostFunction> cost(make_edge_param_cost(
      make_energy_cost(dists, e0, ew, phi.n(), rho.n(), F.n()),
      {{1, 2, 2.0}, {4, 5, 2.0}}));
  audit_jacobian(cost.get(),
                 {phi.c, support_edges(phi, 2.0), phi.eta, rho.c,
                  support_edges(rho, 2.0), rho.eta, F.c, F.mu, F.eta});
}

TEST_CASE("compact fitter: the joint refinement keeps every free bump inside "
          "the hard cutoff", "[compact]") {
  // Truths whose outer bumps reach past 5.5: without a hard cutoff the joint
  // refinement spills; with one, the edges are bounded, the post-solve
  // projection has nothing to move, and a pinned bump keeps its exact place.
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
  std::vector<double> e0, ew(dists.size(), 1.0);
  for (const auto &dd : dists) {
    double hp = 0.0, ri = 0.0;
    for (double r : dd) {
      hp += 0.5 * phi.eval(r);
      ri += rho.eval(r);
    }
    e0.push_back(hp + F.eval(ri));
  }
  JointCurveData dphi, drho, dF;
  sample(phi, 1.0, 5.5, 300, dphi.xs, dphi.ys);
  sample(rho, 1.0, 5.5, 300, drho.xs, drho.ys);
  sample(F, 0.0, 9.0, 300, dF.xs, dF.ys);
  const double rc = 5.5, w = 2.0;

  auto spills = [&](const BlipFit &f) {
    int n = 0;
    for (int k = 0; k < f.n(); ++k)
      if (f.mu[k] + w / f.eta[k] > rc + 1e-6 * rc) ++n;
    return n;
  };
  {
    BlipFit p = phi, r = rho, Fc = F;
    joint_refine(p, r, Fc, dphi, drho, dF, dists, e0, ew, 1.0, 300, 1e-15);
    REQUIRE(spills(p) + spills(r) > 0);  // the case exercises the gate
  }
  BlipFit p = phi, r = rho, Fc = F;
  p.pinned = {true, false, false, false};  // mu 1.5, eta 1.2: inside rc
  const double mu0 = p.mu[0], eta0 = p.eta[0];
  std::string term;
  const double mx = joint_refine(p, r, Fc, dphi, drho, dF, dists, e0, ew, 1.0,
                                 300, 1e-15, "CERES_LM", &term, rc, rc);
  INFO("term " << term << ", max |dE| " << mx);
  REQUIRE(std::isfinite(mx));
  CHECK(spills(p) == 0);
  CHECK(spills(r) == 0);
  CHECK(p.mu[0] == mu0);
  CHECK(p.eta[0] == eta0);
  BlipFit pc = p, rcp = r;
  CHECK(clamp_bump_support_params(pc, rc, 4.5) == 0);
  CHECK(clamp_bump_support_params(rcp, rc, 4.5) == 0);
}

TEST_CASE("compact FUSE: the held re-run passes share the stage's MAXEVAL",
          "[compact][fuse_budget]") {
  // Owner decision (2026-09-29): FUSE MAXEVAL (Ceres iterations) bounds the
  // whole FUSE stage. The held re-run passes -- the fallback when the
  // linear re-solve after the bump-support projection leaves the E-V
  // residuals above tolerance -- ran up to n_phi + n_rho more joint solves
  // with the full MAXEVAL each; they now use what the stage has left, and
  // none runs when nothing is left (the re-solve result is kept).
  // (REQUIRE: past the ceres include above, CHECK is glog's.)
  std::vector<std::string> lines;
  auto log = [&](const std::string &l) { lines.push_back(l); };
  auto logged = [&](const std::string &needle) {
    for (const auto &l : lines)
      if (l.find(needle) != std::string::npos) return true;
    return false;
  };

  SECTION("a spent budget skips the re-run") {
    FuseBudget b{100};
    b.spend(100);
    int holds = 0, solves = 0;
    const HeldPasses r = run_held_passes(
        5, b, [&] { ++holds; return 1; },
        [&](int) { ++solves; return 0; }, log);
    REQUIRE(r.skipped);
    REQUIRE(r.passes == 0);
    REQUIRE(holds == 0);
    REQUIRE(solves == 0);
    REQUIRE(logged("FUSE held re-run skipped: MAXEVAL 100 spent"));
  }
  SECTION("a partly spent budget limits the passes") {
    FuseBudget b{100};
    b.spend(70);
    std::vector<int> budgets;
    const HeldPasses r = run_held_passes(
        5, b, [] { return 1; },
        [&](int max_iter) {
          budgets.push_back(max_iter);
          return std::min(max_iter, 20);
        },
        log);
    // 30 left: a pass takes 20, the next the 10 that remain, then nothing
    REQUIRE(budgets == (std::vector<int>{30, 10}));
    REQUIRE(r.passes == 2);
    REQUIRE_FALSE(r.skipped);
    REQUIRE(r.budget_spent);
    REQUIRE(b.spent == 100);
    REQUIRE(logged("FUSE held re-run stopped after pass 2 of 5: MAXEVAL 100 "
                 "spent"));
  }
  SECTION("the stage never uses more than MAXEVAL") {
    for (int maxeval : {1, 7, 50, 2000}) {
      for (int first : {0, maxeval / 2, maxeval - 1}) {
        INFO("MAXEVAL " << maxeval << ", joint solve " << first);
        FuseBudget b{maxeval};
        b.spend(first);
        run_held_passes(
            8, b, [] { return 1; }, [](int max_iter) { return max_iter; },
            [](const std::string &) {});
        REQUIRE(b.spent <= maxeval);
        REQUIRE(b.remaining() == 0);
      }
    }
  }
  SECTION("nothing more to hold ends the passes") {
    FuseBudget b{100};
    int n = 0;
    const HeldPasses r = run_held_passes(
        5, b, [&] { return ++n <= 2 ? 1 : 0; }, [](int) { return 3; }, log);
    REQUIRE(r.passes == 2);
    REQUIRE_FALSE(r.budget_spent);
    REQUIRE(b.spent == 6);
  }
  SECTION("joint_refine reports the iterations it used, at most max_iter") {
    BlipFit phi = make_truth(), rho, F;
    rho.mu = {2.0, 3.2, 4.4};
    rho.eta = {0.9, 0.8, 0.9};
    rho.c = {0.6, 0.35, 0.15};
    F.mu = {1.0, 4.0, 7.0};
    F.eta = {0.45, 0.4, 0.45};
    F.c = {-2.0, -0.8, 0.5};
    const std::vector<std::vector<double>> dists = {{2.5, 2.5, 2.9, 3.6},
                                                    {2.2, 2.2, 2.6, 3.2}};
    std::vector<double> e0, ew(dists.size(), 1.0);
    for (const auto &dd : dists) {
      double hp = 0.0, ri = 0.0;
      for (double r : dd) {
        hp += 0.5 * phi.eval(r);
        ri += rho.eval(r);
      }
      e0.push_back(hp + F.eval(ri) + 0.01);
    }
    JointCurveData dphi, drho, dF;
    sample(phi, 1.0, 5.5, 200, dphi.xs, dphi.ys);
    sample(rho, 1.0, 5.5, 200, drho.xs, drho.ys);
    sample(F, 0.0, 9.0, 200, dF.xs, dF.ys);
    for (int max_iter : {1, 3, 300}) {
      BlipFit p = phi, r = rho, Fc = F;
      int iters = -1;
      joint_refine(p, r, Fc, dphi, drho, dF, dists, e0, ew, 1.0, max_iter,
                   1e-15, "CERES_LM", nullptr, 5.5, 5.5, &iters);
      INFO("max_iter " << max_iter);
      REQUIRE(iters >= 1);
      REQUIRE(iters <= max_iter);
    }
    BlipFit p = phi, r = rho, Fc = F;
    int iters = -1;
    joint_refine(p, r, Fc, dphi, drho, dF, dists, e0, ew, 1.0, 10, 1e-15,
                 "NONE", nullptr, 5.5, 5.5, &iters);
    REQUIRE(iters == 0);
  }
}

TEST_CASE("compact FUSE, curve mode: the result is never worse than FUSE's "
          "start on a spilling truth", "[compact][fuse_curve]") {
  // Curve-mode FUSE refines each function against its curve (the
  // configured Ceres solver), then the bump-support projection moves a bump
  // that ends past the hard cutoff back inside it and re-solves the
  // coefficients. The refine bounded only the centres, so on a target
  // whose best fit spills it pushed bumps out, and the projection undid
  // the fit: the result could be worse than the start. (REQUIRE: past the
  // ceres include above, CHECK is glog's.)
  const double rcut = 5.5, lo = 1.0;
  const BlipFit truth = make_truth();  // bumps 3 and 4 reach past 5.5
  std::vector<double> xs, ys;
  sample(truth, lo, rcut, 300, xs, ys);
  for (const char *algo : {"CERES_LM", "CERES_DOGLEG"}) {
    INFO(algo);
    CompactConfig cfg;
    cfg.objective = RefitObjective::Curve;
    cfg.refine_algo = algo;
    cfg.hard_rcut = rcut;
    CompactBlipFitter ff(xs, ys, {}, lo, rcut, cfg);
    // FUSE's start: a feasible fit near the truth (the ladder's result),
    // here the truth moved inside the cutoff with bump 2 shifted inward and
    // the coefficients re-solved. The old step refined to the spilling
    // truth, the projection moved bumps 3 and 4 back, and the curve score
    // went from 0.0948 to 0.1020.
    BlipFit fit = truth;
    fit.mu[1] -= 0.1;
    REQUIRE(clamp_bump_support_params(fit, rcut, rcut - lo) == 2);
    ff.solve_coeffs(fit);
    const double before = ff.combined_curve_score(fit);
    const int moved = fuse_curve_refine(ff, fit);
    const double after = ff.combined_curve_score(fit);
    INFO("curve score " << before << " -> " << after << ", moved " << moved);
    REQUIRE(moved == 0);
    REQUIRE(after <= before);
    for (int k = 0; k < fit.n(); ++k)
      REQUIRE(fit.mu[k] + 2.0 / fit.eta[k] <= rcut + 1e-6 * rcut);
  }
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

TEST_CASE("compact fitter: each EXPLORE INNER local search lowers the "
          "rung's objective", "[refit][compact]") {
  /* One optimised rung of one candidate (no jitter), from a uniform start
   * (no seed search) with no refine: the rung's curve RMSE after the local
   * search, against the insertion alone (ALGO NONE, a linear solve). The
   * NLopt L-BFGS search used to optimise through a dangling pointer: its
   * objective context was a local of the branch that set it, gone when
   * optimize() ran (a stack-use-after-scope under AddressSanitizer). */
  std::vector<double> xs, ys;
  sample(make_truth(), 1.0, 5.5, 160, xs, ys);
  const auto rung_rmse = [&](const char *inner) {
    CompactConfig cfg;
    cfg.n_max = cfg.n_min + 1;
    cfg.jitters = 0;
    cfg.target = 0.0;
    cfg.seed_algo = "UNIFORM";
    cfg.explore_algo = inner;
    cfg.sbplx_maxeval = 200;
    cfg.refine_algo = "NONE";
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    std::vector<LadderRow> trace;
    (void)fit.fit([](const BlipFit &f) { return f.curve_rmse; }, trace);
    REQUIRE(trace.size() == 2u);  // the seed, then the rung
    return trace.back().curve_rmse;
  };
  const double insertion_only = rung_rmse("NONE");
  for (const char *inner : {"NLOPT_LBFGS", "DLIB_LBFGS", "SBPLX"}) {
    INFO("EXPLORE INNER " << inner << ", insertion alone "
                          << insertion_only);
    CHECK(rung_rmse(inner) < insertion_only);
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
  SECTION("a radial bump width <= 0 is refused, not skipped") {
    // every radial width is > 0 (INIT refuses eta <= 0, the ladder and
    // FUSE keep their width floors): a width <= 0 here is a broken
    // invariant, and its support edge c + w/eta says nothing
    for (double eta : {0.0, -0.8}) {
      BlipFit f = make_truth();
      f.eta[1] = eta;
      REQUIRE_THROWS_AS(clamp_bump_support_params(f, rcut, span),
                        std::logic_error);
      REQUIRE_THROWS_WITH(clamp_bump_support_params(f, rcut, span),
                          Catch::Contains("bump 2") &&
                              Catch::Contains("width"));
    }
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

// ============ EMBF0 ZERO: zero-referenced embedding basis g - g(0) ============
// The FREE tests above are the regression for the raw basis; these cover the
// zero_ref paths (BlipFit::eval, CurveCost, solve_coeffs, the VARPRO gradient,
// the EnergyCost F block, joint_refine).

namespace {
// Independent zero-referenced model sum_k c_k (g_k(x) - g_k(0)) built straight
// from the MODELS primitives (not the fitter's dispatch).
double zero_ref_model(const BlipFit &f, double x) {
  namespace tm = tadah::models;
  double acc = 0.0;
  for (int k = 0; k < f.n(); ++k) {
    auto g = [&](double u) {
      const double xc = f.eta[k] * (u - f.mu[k]);
      switch (f.kind) {
        case BasisKind::Blip:   return tm::B(xc);
        case BasisKind::Blip5:  return tm::B5(xc);
        case BasisKind::Knot5L: return f.degree(k) == 3 ? tm::K3L(xc) : tm::K5L(xc);
        case BasisKind::Knot5R: return f.degree(k) == 3 ? tm::K3R(xc) : tm::K5R(xc);
      }
      return 0.0;
    };
    acc += f.c[k] * (g(x) - g(0.0));
  }
  return acc;
}

// Embedding cases whose atoms have g(0) != 0 — bumps covering rho = 0 and
// right-active knots below 0 — each mixed with atoms that have g(0) == 0.
struct ZeroCase {
  const char *name;
  BlipFit f;
  double lo, hi;
};
std::vector<ZeroCase> zero_ref_cases() {
  std::vector<ZeroCase> v;
  BlipFit b;
  b.kind = BasisKind::Blip;
  b.mu = {0.3, 1.5, 2.6, 4.0};  // x_c0 = -0.27, -1.65 (cover 0), -2.08, -4
  b.eta = {0.9, 1.1, 0.8, 1.0};
  b.c = {2.0, -1.3, 0.7, -0.4};
  v.push_back({"Blip", b, 0.0, 5.0});
  BlipFit b5 = b;
  b5.kind = BasisKind::Blip5;  // support 3: the first three cover 0
  v.push_back({"Blip5", b5, 0.0, 5.0});
  BlipFit k5;
  k5.kind = BasisKind::Knot5R;
  k5.mu = {-0.8, -0.3, 0.4, 1.1};  // two knots below 0 (g(0) != 0)
  k5.eta = {1.0, 1.0, 1.0, 1.0};
  k5.c = {0.3, -0.2, 0.1, -0.05};
  v.push_back({"Knot5R deg 5", k5, 0.0, 2.0});
  BlipFit k3 = k5;
  k3.deg = {3, 3, 3, 3};
  v.push_back({"Knot5R deg 3", k3, 0.0, 2.0});
  return v;
}
}  // namespace

TEST_CASE("compact fitter: EMBF0 ZERO curve cost — model, F(0) = 0, Jacobian",
          "[compact][embf0]") {
  for (const ZeroCase &zc : zero_ref_cases()) {
    INFO("case " << zc.name);
    BlipFit f = zc.f;
    f.zero_ref = true;
    BlipFit raw = f;
    raw.zero_ref = false;
    REQUIRE(std::fabs(raw.eval(0.0)) > 1e-3);  // g(0) != 0: the case bites
    REQUIRE(f.eval(0.0) == 0.0);               // explicit F(0) = 0

    const int N = 160;
    std::vector<double> xs, ys, wts;
    for (int i = 0; i < N; ++i) {
      const double x = zc.lo + (zc.hi - zc.lo) * i / (N - 1);
      xs.push_back(x);
      ys.push_back(0.3 * std::cos(1.7 * x));  // any target, nonzero at 0
      wts.push_back(0.5 + static_cast<double>(i) / N);
      REQUIRE(f.eval(x) == Approx(zero_ref_model(f, x)).margin(1e-12));
      // derivatives are reference-free
      REQUIRE(f.deriv(x) == raw.deriv(x));
      REQUIRE(f.deriv2(x) == raw.deriv2(x));
    }
    REQUIRE(xs[0] == 0.0);

    std::unique_ptr<ceres::CostFunction> cost(make_curve_cost(
        xs, ys, wts, f.n(), f.kind, /*zero_ref=*/true, f.deg));
    const int n = f.n();
    std::vector<double> r(N), j0(N * n), j1(N * n), j2(N * n);
    const double *pp[3] = {f.c.data(), f.mu.data(), f.eta.data()};
    double *jj[3] = {j0.data(), j1.data(), j2.data()};
    REQUIRE(cost->Evaluate(pp, r.data(), jj));
    for (int i = 0; i < N; ++i)
      REQUIRE(r[i] == Approx(std::sqrt(wts[i]) *
                             (zero_ref_model(f, xs[i]) - ys[i]))
                          .margin(1e-12));
    // the rho = 0 row: model exactly 0, and every Jacobian entry exactly 0
    // (F(0) = 0 identically in the parameters)
    REQUIRE(r[0] == -std::sqrt(wts[0]) * ys[0]);
    for (int k = 0; k < n; ++k) {
      REQUIRE(j0[k] == 0.0);
      REQUIRE(j1[k] == 0.0);
      REQUIRE(j2[k] == 0.0);
    }
    audit_jacobian(cost.get(), {f.c, f.mu, f.eta});
  }
}

TEST_CASE("compact fitter: EMBF0 ZERO equals FREE where every g(0) == 0",
          "[compact][embf0]") {
  // right-active knots at/above 0 vanish at rho = 0 (with their derivative),
  // so the zero reference is an exact no-op: bitwise-equal model, residuals
  // and least-squares coefficients
  BlipFit k;
  k.kind = BasisKind::Knot5R;
  k.mu = {0.0, 0.4, 0.9, 1.3};
  k.eta = {1.0, 1.0, 1.0, 1.0};
  k.c = {0.3, -0.2, 0.1, -0.05};
  std::vector<double> xs, ys;
  sample(k, 0.0, 2.0, 200, xs, ys);
  for (auto &y : ys) y += 0.01 * std::sin(3.0 * y);  // not exactly in class

  BlipFit z = k;
  z.zero_ref = true;
  for (double x : xs) REQUIRE(z.eval(x) == k.eval(x));

  std::unique_ptr<ceres::CostFunction> cf(
      make_curve_cost(xs, ys, {}, k.n(), k.kind));
  std::unique_ptr<ceres::CostFunction> cz(
      make_curve_cost(xs, ys, {}, k.n(), k.kind, /*zero_ref=*/true));
  std::vector<double> rf(xs.size()), rz(xs.size());
  const double *pp[3] = {k.c.data(), k.mu.data(), k.eta.data()};
  REQUIRE(cf->Evaluate(pp, rf.data(), nullptr));
  REQUIRE(cz->Evaluate(pp, rz.data(), nullptr));
  for (std::size_t i = 0; i < xs.size(); ++i) REQUIRE(rz[i] == rf[i]);

  CompactConfig cfree, czero;
  czero.zero_ref = true;
  CompactBlipFitter ffree(xs, ys, {}, 0.0, 2.0, cfree, BasisKind::Knot5R);
  CompactBlipFitter fzero(xs, ys, {}, 0.0, 2.0, czero, BasisKind::Knot5R);
  BlipFit a = k, b = k;
  ffree.solve_coeffs(a);
  fzero.solve_coeffs(b);
  REQUIRE(!a.zero_ref);
  REQUIRE(b.zero_ref);  // stamped from the fitter's config
  for (int j = 0; j < k.n(); ++j) REQUIRE(b.c[j] == a.c[j]);
  REQUIRE(b.curve_rmse == a.curve_rmse);
}

TEST_CASE("compact fitter: EMBF0 ZERO energy Jacobian (F block) matches FD",
          "[compact][embf0]") {
  const std::vector<std::vector<double>> dists = {
      {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
      {2.1, 2.1, 2.45, 2.45, 3.0}};
  const std::vector<double> e0 = {-1.0, 0.5, 2.0};
  const std::vector<double> ew = {1.0, 2.0, 0.5};

  auto check = [&](BlipFit phi, BlipFit rho, BlipFit F) {
    F.zero_ref = true;
    std::unique_ptr<ceres::CostFunction> cost(make_energy_cost(
        dists, e0, ew, phi.n(), rho.n(), F.n(), phi.kind, rho.kind, F.kind,
        /*F_zero_ref=*/true, F.deg));
    // the residual is w (sum 0.5 phi + F~(rho_i) - e0) with the zero-
    // referenced F~ (phi, rho raw)
    std::vector<double> r(dists.size());
    const double *pp[9] = {phi.c.data(), phi.mu.data(), phi.eta.data(),
                           rho.c.data(), rho.mu.data(), rho.eta.data(),
                           F.c.data(),   F.mu.data(),   F.eta.data()};
    REQUIRE(cost->Evaluate(pp, r.data(), nullptr));
    for (std::size_t i = 0; i < dists.size(); ++i) {
      double hp = 0.0, ri = 0.0;
      for (double d : dists[i]) {
        hp += 0.5 * phi.eval(d);
        ri += rho.eval(d);
      }
      const double Fz = zero_ref_model(F, ri);
      REQUIRE(std::fabs(Fz - F.eval(ri)) <= 1e-12 * (1.0 + std::fabs(Fz)));
      REQUIRE(r[i] == Approx(ew[i] * (hp + Fz - e0[i]))
                          .epsilon(1e-12)
                          .margin(1e-12));
    }
    audit_jacobian(cost.get(), {phi.c, phi.mu, phi.eta, rho.c, rho.mu,
                                rho.eta, F.c, F.mu, F.eta});
  };

  SECTION("bump embeddings covering rho = 0 (Blip, Blip5)") {
    BlipFit rho;
    rho.mu = {2.0, 3.2, 4.4};
    rho.eta = {0.9, 0.8, 0.9};
    rho.c = {0.6, 0.35, 0.15};
    for (BasisKind kF : {BasisKind::Blip, BasisKind::Blip5}) {
      INFO("F kind " << static_cast<int>(kF));
      BlipFit F;
      F.kind = kF;
      F.mu = {1.0, 4.0, 7.0};  // x_c0 = -0.45, -1.6 (cover 0), -3.15
      F.eta = {0.45, 0.4, 0.45};
      F.c = {-2.0, -0.8, 0.5};
      check(make_truth(), rho, F);
    }
  }
  SECTION("right-active knots below 0 (quintic, cubic)") {
    BlipFit rho;
    rho.kind = BasisKind::Knot5L;
    rho.mu = {2.8, 3.9, 5.0};
    rho.eta = {1.0, 1.0, 1.0};
    rho.c = {0.05, 0.03, 0.01};
    for (int d : {5, 3}) {
      INFO("F knot degree " << d);
      BlipFit F;
      F.kind = BasisKind::Knot5R;
      F.mu = {-1.0, -0.3, 0.4};
      F.eta = {1.0, 1.0, 1.0};
      F.c = {-0.4, -0.2, 0.1};
      if (d == 3) F.deg = {3, 3, 3};
      check(make_knot_truth(), rho, F);
    }
  }
}

TEST_CASE("compact fitter: EMBF0 ZERO VARPRO gradient matches finite differences",
          "[compact][embf0]") {
  for (const ZeroCase &zc : zero_ref_cases()) {
    INFO("case " << zc.name);
    BlipFit truth = zc.f;
    truth.zero_ref = true;
    std::vector<double> xs, ys;
    sample(truth, zc.lo, zc.hi, 220, xs, ys);  // zero-referenced samples
    CompactConfig cfg;
    cfg.zero_ref = true;
    CompactBlipFitter fit(xs, ys, {}, zc.lo, zc.hi, cfg, truth.kind);

    const bool os = one_sided(truth.kind);
    BlipFit f = truth;
    f.zero_ref = false;  // the fitter stamps it
    std::mt19937 rng(13);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int k = 0; k < f.n(); ++k) {
      f.mu[k] += 0.05 * g(rng);
      if (!os) f.eta[k] *= std::exp(0.05 * g(rng));
    }
    std::vector<double> grad;
    fit.reduced_mse(f, &grad);
    REQUIRE(f.zero_ref);
    const int n = f.n();
    REQUIRE(grad.size() == static_cast<std::size_t>(os ? n : 2 * n));
    double scale = 0.0;
    for (double v : grad) scale = std::max(scale, std::fabs(v));
    REQUIRE(scale > 0.0);

    double worst = 0.0;
    for (int i = 0; i < static_cast<int>(grad.size()); ++i) {
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
    REQUIRE(worst < 1e-5);
  }
}

TEST_CASE("compact fitter: EMBF0 ZERO refine recovers exact zero-referenced "
          "blips", "[compact][embf0]") {
  BlipFit truth = zero_ref_cases()[0].f;  // Blip, two bumps cover rho = 0
  truth.zero_ref = true;
  std::vector<double> xs, ys;
  sample(truth, 0.0, 5.0, 400, xs, ys);
  CompactConfig cfg;
  cfg.zero_ref = true;
  CompactBlipFitter fit(xs, ys, {}, 0.0, 5.0, cfg);

  BlipFit f = truth;
  f.zero_ref = false;
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < f.n(); ++k) {
    f.mu[k] += 0.03 * g(rng);
    f.eta[k] *= std::exp(0.03 * g(rng));
  }
  fit.solve_coeffs(f);
  REQUIRE(f.zero_ref);
  const double before = f.curve_rmse;
  const double after = fit.refine(f);
  INFO("before = " << before << " after = " << after);
  REQUIRE(after < before);
  REQUIRE(after < 1e-9);  // representable target, exact zero-ref Jacobians
  REQUIRE(f.eval(0.0) == 0.0);
}

TEST_CASE("compact fitter: EMBF0 ZERO joint refine drives E-V residuals to "
          "~noise", "[compact][embf0]") {
  // the FREE joint test with a zero-referenced embedding truth (two bumps
  // cover rho = 0): scan energies and the F curve are zero-referenced
  BlipFit phi = make_truth();
  BlipFit rho;
  rho.mu = {2.0, 3.2, 4.4};
  rho.eta = {0.9, 0.8, 0.9};
  rho.c = {0.6, 0.35, 0.15};
  BlipFit F;
  F.mu = {1.0, 4.0, 7.0};
  F.eta = {0.45, 0.4, 0.45};
  F.c = {-2.0, -0.8, 0.5};
  F.zero_ref = true;

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
  JointCurveData dphi, drho, dF;
  sample(phi, 1.0, 5.5, 300, dphi.xs, dphi.ys);
  sample(rho, 1.0, 5.5, 300, drho.xs, drho.ys);
  sample(F, 0.0, 9.0, 300, dF.xs, dF.ys);
  REQUIRE(dF.ys[0] == 0.0);

  // the zero reference is an embedding property: radial fits reject it
  {
    BlipFit bad = phi;
    bad.zero_ref = true;
    BlipFit r2 = rho, F2 = F;
    REQUIRE_THROWS_WITH(
        joint_refine(bad, r2, F2, dphi, drho, dF, dists, e0, ew, 1.0, 10,
                     1e-15),
        Catch::Contains("embedding F only"));
  }

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
  REQUIRE(final_max < 1e-8);
  REQUIRE(F.zero_ref);
  REQUIRE(F.eval(0.0) == 0.0);
}

TEST_CASE("compact fitter: EMBF0 ZERO ladder fits F_ref with F(0) == 0",
          "[compact][embf0]") {
  // An embedding-like table with F_tab(0) != 0 (the isolated-atom constant).
  // The ZERO chain fits F_ref = F_tab - F_tab(0) (the engine's target) in the
  // zero-referenced basis; FREE fits the same shifted target in the raw basis
  // as the quality reference.
  auto F_tab = [](double x) {
    return 0.37 + std::exp(-(x - 2.0)) - 2.0 * std::exp(-0.5 * (x - 2.0));
  };
  const int N = 300;
  std::vector<double> xs(N), ys(N);
  for (int i = 0; i < N; ++i) {
    xs[i] = 5.0 * i / (N - 1);
    ys[i] = F_tab(xs[i]) - F_tab(0.0);
  }
  REQUIRE(ys[0] == 0.0);
  auto run = [&](bool zero) {
    CompactConfig cfg;  // the Morse ladder test's budget
    cfg.n_max = 16;
    cfg.sbplx_maxeval = 4000;  // keep the unit test quick
    cfg.mlsl_maxeval = 8000;
    cfg.target = 1.0;
    cfg.zero_ref = zero;
    CompactBlipFitter fit(xs, ys, {}, 0.0, 5.0, cfg);
    std::vector<LadderRow> trace;
    return fit.fit([](const BlipFit &f) { return f.curve_rmse / 1e-4; },
                   trace);
  };
  const BlipFit bz = run(true);
  const BlipFit bf = run(false);
  INFO("ZERO: n = " << bz.n() << " rmse = " << bz.curve_rmse
                    << "   FREE: n = " << bf.n() << " rmse = "
                    << bf.curve_rmse);
  REQUIRE(bz.zero_ref);
  REQUIRE(!bf.zero_ref);
  REQUIRE(bz.eval(0.0) == 0.0);                  // exact, by construction
  REQUIRE(std::fabs(bz.eval(1e-12)) < 1e-6);     // and continuous there
  REQUIRE(bz.curve_rmse < 1e-2);                 // |F_ref| ~ 3: a real fit
  // comparable to FREE on the same target (both normally reach 1e-4)
  REQUIRE(bz.curve_rmse <= 5.0 * bf.curve_rmse + 1e-4);
}

// ================ RHOFORM: the density forms of the rho channel ================
// SIGNED is the raw bump expansion (every test above); NONNEG the same with
// c >= 0 (non-negative least squares, Ceres lower bounds); MONOTONE the
// integrated-bump step kinds S with c >= 0. These cover the step kinds'
// values and Jacobians, the non-negative solve against brute force, the
// passive-set VARPRO gradient, the Ceres bounds and small ladders per form.

namespace {
using tadah::models::RhoForm;
namespace tm = tadah::models;

// The bump area in x_c units: dS/dr = -eta B(x_c)/A.
double step_area(BasisKind k) {
  return k == BasisKind::Blip5 ? 20.0 / 11.0 : 1.5;
}
double bump(BasisKind k, double xc) {
  return k == BasisKind::Blip5 ? tm::B5(xc) : tm::B(xc);
}
double step(BasisKind k, double xc) {
  return k == BasisKind::Blip5 ? tm::SB5(xc) : tm::SB(xc);
}
// The column of basis k under a form, built straight from the MODELS
// primitives (not the fitter's dispatch).
double form_column(RhoForm form, BasisKind kind, double eta, double mu,
                   double x) {
  const double xc = eta * (x - mu);
  return form == RhoForm::Monotone ? step(kind, xc) : bump(kind, xc);
}

// A positive steps truth (MONOTONE) / bumps truth (NONNEG) on [1.0, 5.5].
BlipFit make_positive_truth(BasisKind kind, RhoForm form) {
  BlipFit t;
  t.kind = kind;
  t.form = form;
  t.mu = {1.8, 2.9, 3.8, 4.6};
  t.eta = {1.1, 0.9, 1.3, 1.6};
  t.c = {0.9, 0.5, 0.3, 0.1};
  return t;
}

// min ||A c - y|| over c >= 0 by enumerating every support set: the
// unconstrained least squares on the set, kept only when strictly positive
// (the zero vector is the empty set). Small n only.
std::vector<double> brute_force_nnls(const Eigen::MatrixXd &A,
                                     const Eigen::VectorXd &y,
                                     double *best_sse) {
  const int n = static_cast<int>(A.cols());
  std::vector<double> best(n, 0.0);
  double bsse = y.squaredNorm();
  for (int mask = 1; mask < (1 << n); ++mask) {
    std::vector<int> cols;
    for (int k = 0; k < n; ++k)
      if (mask & (1 << k)) cols.push_back(k);
    Eigen::MatrixXd As(A.rows(), static_cast<int>(cols.size()));
    for (std::size_t q = 0; q < cols.size(); ++q) As.col(q) = A.col(cols[q]);
    const Eigen::VectorXd cs = As.colPivHouseholderQr().solve(y);
    if ((cs.array() <= 0.0).any()) continue;
    const double sse = (As * cs - y).squaredNorm();
    if (sse < bsse) {
      bsse = sse;
      best.assign(n, 0.0);
      for (std::size_t q = 0; q < cols.size(); ++q) best[cols[q]] = cs(q);
    }
  }
  *best_sse = bsse;
  return best;
}
}  // namespace

TEST_CASE("compact fitter RHOFORM: defaults are SIGNED", "[compact][rhoform]") {
  REQUIRE(CompactConfig{}.rho_form == RhoForm::Signed);
  REQUIRE(BlipFit{}.form == RhoForm::Signed);
  // a SIGNED solve keeps its negative coefficients: the truth is signed
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 200, xs, ys);
  CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, CompactConfig{});
  BlipFit f = truth;
  f.c.assign(f.n(), 0.0);
  fit.solve_coeffs(f);
  REQUIRE(f.form == RhoForm::Signed);
  REQUIRE(f.curve_rmse < 1e-12);
  REQUIRE(*std::min_element(f.c.begin(), f.c.end()) < -1.0);
}

TEST_CASE("compact fitter RHOFORM: step kinds against the closed forms",
          "[compact][rhoform]") {
  for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
    INFO("kind " << static_cast<int>(kind));
    const double A = step_area(kind);
    BlipFit f = make_positive_truth(kind, RhoForm::Monotone);
    BlipFit bumps = f;
    bumps.form = RhoForm::NonNeg;
    for (double x = 0.3; x < 6.8; x += 0.0137) {
      double v = 0.0, d = 0.0, d2 = 0.0, vb = 0.0;
      for (int k = 0; k < f.n(); ++k) {
        const double xc = f.eta[k] * (x - f.mu[k]);
        // the MODELS step primitive the written DM_REAM evaluates (bitwise)
        v += f.c[k] * tm::radial_step_value_bare(
                          kind == BasisKind::Blip5 ? tm::BasisKind::Blip5
                                                   : tm::BasisKind::Blip,
                          x, f.eta[k], f.mu[k]);
        vb += f.c[k] * bump(kind, xc);
        // dS/dr = -eta B/A, d2S/dr2 = -eta^2 B'/A
        d += f.c[k] * (-f.eta[k] * bump(kind, xc) / A);
        const double bp = kind == BasisKind::Blip5 ? tm::dB5(xc, 1.0)
                                                   : tm::dB(xc, 1.0);
        d2 += f.c[k] * (-f.eta[k] * f.eta[k] * bp / A);
      }
      REQUIRE(f.eval(x) == v);
      REQUIRE(bumps.eval(x) == vb);  // NONNEG evaluates the SIGNED columns
      REQUIRE(f.deriv(x) == Approx(d).epsilon(1e-12).margin(1e-14));
      REQUIRE(f.deriv2(x) == Approx(d2).epsilon(1e-12).margin(1e-13));
      // and against finite differences of eval / deriv
      const double h = 1e-6;
      REQUIRE(f.deriv(x) ==
              Approx((f.eval(x + h) - f.eval(x - h)) / (2 * h)).margin(1e-8));
      REQUIRE(f.deriv2(x) ==
              Approx((f.deriv(x + h) - f.deriv(x - h)) / (2 * h)).margin(1e-7));
      // a positive steps expansion is >= 0 and non-increasing
      REQUIRE(f.eval(x) >= 0.0);
      REQUIRE(f.deriv(x) <= 0.0);
    }
  }
}

TEST_CASE("compact fitter RHOFORM: step curve Jacobian — closed forms and FD",
          "[compact][rhoform]") {
  for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
    INFO("kind " << static_cast<int>(kind));
    const double A = step_area(kind);
    // one step, c = 1, unit weights, target 0: residual = S, and the
    // mu / eta columns are exactly dS/dmu = eta B/A, dS/deta = -(x-mu) B/A
    const double mu = 3.1, eta = 0.8, c = 1.0;
    std::vector<double> xs;
    for (int i = 0; i < 120; ++i) xs.push_back(0.5 + 5.5 * i / 119.0);
    std::unique_ptr<ceres::CostFunction> one(make_curve_cost(
        xs, std::vector<double>(xs.size(), 0.0), {}, 1, kind, false, {},
        RhoForm::Monotone));
    const std::size_t N = xs.size();
    std::vector<double> r(N), j0(N), j1(N), j2(N);
    const double *pp[3] = {&c, &mu, &eta};
    double *jj[3] = {j0.data(), j1.data(), j2.data()};
    REQUIRE(one->Evaluate(pp, r.data(), jj));
    for (std::size_t i = 0; i < N; ++i) {
      const double xc = eta * (xs[i] - mu);
      REQUIRE(r[i] == step(kind, xc));
      REQUIRE(j0[i] == step(kind, xc));
      REQUIRE(j1[i] ==
              Approx(eta * bump(kind, xc) / A).epsilon(1e-12).margin(1e-15));
      REQUIRE(j2[i] == Approx(-(xs[i] - mu) * bump(kind, xc) / A)
                           .epsilon(1e-12)
                           .margin(1e-15));
    }

    // FD audit on a weighted multi-step fit (the signed audit's recipe)
    const BlipFit truth = make_positive_truth(kind, RhoForm::Monotone);
    std::vector<double> sx, sy, wts;
    sample(truth, 1.0, 5.5, 160, sx, sy);
    for (std::size_t i = 0; i < sx.size(); ++i)
      wts.push_back(0.5 + static_cast<double>(i) / sx.size());
    BlipFit f = truth;
    std::mt19937 rng(7);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int k = 0; k < f.n(); ++k) {
      f.mu[k] += 0.05 * g(rng);
      f.eta[k] *= std::exp(0.05 * g(rng));
    }
    std::unique_ptr<ceres::CostFunction> cost(make_curve_cost(
        sx, sy, wts, f.n(), kind, false, {}, RhoForm::Monotone));
    audit_jacobian(cost.get(), {f.c, f.mu, f.eta});
  }
}

TEST_CASE("compact fitter RHOFORM: NONNEG curve cost is the SIGNED cost",
          "[compact][rhoform]") {
  // same columns: NONNEG differs only by the Ceres bounds and the solve
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 160, xs, ys);
  for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
    std::unique_ptr<ceres::CostFunction> cs(
        make_curve_cost(xs, ys, {}, truth.n(), kind));
    std::unique_ptr<ceres::CostFunction> cn(make_curve_cost(
        xs, ys, {}, truth.n(), kind, false, {}, RhoForm::NonNeg));
    const int n = truth.n();
    const std::size_t N = xs.size();
    std::vector<double> rs(N), rn(N), s0(N * n), s1(N * n), s2(N * n),
        n0(N * n), n1(N * n), n2(N * n);
    const double *pp[3] = {truth.c.data(), truth.mu.data(), truth.eta.data()};
    double *js[3] = {s0.data(), s1.data(), s2.data()};
    double *jn[3] = {n0.data(), n1.data(), n2.data()};
    REQUIRE(cs->Evaluate(pp, rs.data(), js));
    REQUIRE(cn->Evaluate(pp, rn.data(), jn));
    REQUIRE(rs == rn);
    REQUIRE(s0 == n0);
    REQUIRE(s1 == n1);
    REQUIRE(s2 == n2);
  }
}

TEST_CASE("compact fitter RHOFORM: step energy Jacobian (rho block) matches FD",
          "[compact][rhoform]") {
  const std::vector<std::vector<double>> dists = {
      {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
      {2.1, 2.1, 2.45, 2.45, 3.0}};
  const std::vector<double> e0 = {-1.0, 0.5, 2.0};
  const std::vector<double> ew = {1.0, 2.0, 0.5};
  for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
    INFO("rho kind " << static_cast<int>(kind));
    BlipFit phi = make_truth();
    BlipFit rho;
    rho.kind = kind;
    rho.form = RhoForm::Monotone;
    rho.mu = {2.4, 3.1, 3.7};
    rho.eta = {1.2, 0.9, 1.4};
    rho.c = {0.4, 0.25, 0.1};
    BlipFit F;
    F.mu = {0.5, 1.5, 2.5};
    F.eta = {0.9, 0.8, 0.9};
    F.c = {-2.0, -0.8, 0.5};
    std::unique_ptr<ceres::CostFunction> cost(make_energy_cost(
        dists, e0, ew, phi.n(), rho.n(), F.n(), phi.kind, rho.kind, F.kind,
        false, {}, RhoForm::Monotone));
    // the residual evaluates the step density sum_d sum_k c_k S(x_c)
    std::vector<double> r(dists.size());
    const double *pp[9] = {phi.c.data(), phi.mu.data(), phi.eta.data(),
                           rho.c.data(), rho.mu.data(), rho.eta.data(),
                           F.c.data(),   F.mu.data(),   F.eta.data()};
    REQUIRE(cost->Evaluate(pp, r.data(), nullptr));
    for (std::size_t i = 0; i < dists.size(); ++i) {
      double hp = 0.0, ri = 0.0;
      for (double d : dists[i]) {
        hp += 0.5 * phi.eval(d);
        ri += rho.eval(d);
      }
      REQUIRE(ri > 0.0);
      REQUIRE(r[i] == Approx(ew[i] * (hp + F.eval(ri) - e0[i]))
                          .epsilon(1e-12)
                          .margin(1e-12));
    }
    audit_jacobian(cost.get(), {phi.c, phi.mu, phi.eta, rho.c, rho.mu,
                                rho.eta, F.c, F.mu, F.eta});
  }
}

TEST_CASE("compact fitter RHOFORM: non-negative solve_coeffs equals brute force",
          "[compact][rhoform]") {
  // Targets the form cannot follow exactly, so the constraints bind: NONNEG
  // on a curve with a negative lobe, MONOTONE on a curve with a rise.
  struct Case {
    const char *name;
    BasisKind kind;
    RhoForm form;
  };
  for (const Case &cs : {Case{"Blip NONNEG", BasisKind::Blip, RhoForm::NonNeg},
                         Case{"Blip5 NONNEG", BasisKind::Blip5, RhoForm::NonNeg},
                         Case{"Knot5L NONNEG", BasisKind::Knot5L, RhoForm::NonNeg},
                         Case{"Blip MONOTONE", BasisKind::Blip, RhoForm::Monotone},
                         Case{"Blip5 MONOTONE", BasisKind::Blip5,
                              RhoForm::Monotone}}) {
    INFO(cs.name);
    const int N = 180;
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
      const double x = 1.0 + 4.5 * i / (N - 1);
      xs[i] = x;
      const double s = 1.0 - x / 5.5;
      ys[i] = s * s * (1.0 + 0.6 * std::exp(-8.0 * (x - 3.6) * (x - 3.6))) -
              0.4 * std::exp(-10.0 * (x - 2.4) * (x - 2.4));
    }
    BlipFit f;
    f.kind = cs.kind;
    if (one_sided(cs.kind)) {
      f.mu = {2.2, 3.0, 3.7, 4.4, 5.5};
      f.eta.assign(5, 1.0);
    } else {
      f.mu = {1.4, 2.3, 3.0, 3.6, 4.3};
      f.eta = {1.5, 2.0, 1.2, 2.2, 1.4};
    }
    f.c.assign(f.n(), 0.0);

    CompactConfig cfg;
    cfg.rho_form = cs.form;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, cs.kind);
    fit.solve_coeffs(f);
    REQUIRE(f.form == cs.form);

    // the same design, from the MODELS primitives (knots: the fitter's own
    // SIGNED columns, read back through a unit-coefficient eval)
    Eigen::MatrixXd A(N, f.n());
    Eigen::VectorXd y(N);
    for (int i = 0; i < N; ++i) {
      y(i) = ys[i];
      for (int k = 0; k < f.n(); ++k) {
        if (one_sided(cs.kind)) {
          BlipFit unit;
          unit.kind = cs.kind;
          unit.mu = {f.mu[k]};
          unit.eta = {1.0};
          unit.c = {1.0};
          A(i, k) = unit.eval(xs[i]);
        } else {
          A(i, k) = form_column(cs.form, cs.kind, f.eta[k], f.mu[k], xs[i]);
        }
      }
    }
    // the constraints bind: the unconstrained solution has a negative c
    const Eigen::VectorXd cu = A.completeOrthogonalDecomposition().solve(y);
    REQUIRE(cu.minCoeff() < 0.0);

    double sse_bf = 0.0;
    const std::vector<double> cbf = brute_force_nnls(A, y, &sse_bf);
    double sse = 0.0;
    for (int i = 0; i < N; ++i) {
      const double d = f.eval(xs[i]) - ys[i];
      sse += d * d;
    }
    INFO("sse = " << sse << "  brute force = " << sse_bf);
    REQUIRE(sse == Approx(sse_bf).epsilon(1e-9));
    REQUIRE(f.curve_rmse == Approx(std::sqrt(sse_bf / N)).epsilon(1e-9));
    int n_zero = 0;
    for (int k = 0; k < f.n(); ++k) {
      REQUIRE(f.c[k] >= 0.0);
      REQUIRE(f.c[k] == Approx(cbf[k]).margin(1e-8));
      if (f.c[k] == 0.0) ++n_zero;
    }
    REQUIRE(n_zero >= 1);  // an active bound, held at exactly 0
  }
}

TEST_CASE("compact fitter RHOFORM: passive-set VARPRO gradient matches FD",
          "[compact][rhoform]") {
  // truth: positive terms, plus a small term of the WRONG sign at x = 2.3
  // whose fit column (the 5th) is then held at c = 0 by the constraint —
  // its (mu, eta) gradient is 0 and FD agrees
  for (RhoForm form : {RhoForm::NonNeg, RhoForm::Monotone}) {
    for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
      INFO("form " << tm::rhoform_name(form) << " kind "
                   << static_cast<int>(kind));
      BlipFit truth = make_positive_truth(kind, form);
      BlipFit wrong = truth;
      wrong.mu = {2.3};
      wrong.eta = {2.5};
      wrong.c = {form == RhoForm::Monotone ? -0.04 : -0.08};
      std::vector<double> xs, ys, yw, tmp;
      sample(truth, 1.0, 5.5, 220, xs, ys);
      sample(wrong, 1.0, 5.5, 220, tmp, yw);
      for (std::size_t i = 0; i < ys.size(); ++i) ys[i] += yw[i];

      CompactConfig cfg;
      cfg.rho_form = form;
      CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, kind);
      BlipFit f = truth;
      f.form = RhoForm::Signed;  // the fitter stamps it
      f.mu.push_back(2.3);
      f.eta.push_back(2.5);
      f.c.push_back(0.0);
      std::mt19937 rng(13);
      std::normal_distribution<double> g(0.0, 1.0);
      for (int k = 0; k < f.n(); ++k) {
        f.mu[k] += 0.03 * g(rng);
        f.eta[k] *= std::exp(0.03 * g(rng));
      }
      std::vector<double> grad;
      fit.reduced_mse(f, &grad);
      REQUIRE(f.form == form);
      const int n = f.n();
      REQUIRE(grad.size() == static_cast<std::size_t>(2 * n));
      for (int k = 0; k < n; ++k) REQUIRE(f.c[k] >= 0.0);
      REQUIRE(f.c[n - 1] == 0.0);           // held at the bound
      REQUIRE(grad[n - 1] == 0.0);          // ... so no mu gradient
      REQUIRE(grad[2 * n - 1] == 0.0);      // ... and no eta gradient
      double scale = 0.0;
      for (double v : grad) scale = std::max(scale, std::fabs(v));
      REQUIRE(scale > 0.0);
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
      REQUIRE(worst < 1e-5);
    }
  }
}

TEST_CASE("compact fitter RHOFORM: refine keeps c >= 0 and recovers "
          "in-form truths", "[compact][rhoform]") {
  for (RhoForm form : {RhoForm::NonNeg, RhoForm::Monotone}) {
    for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
      INFO("form " << tm::rhoform_name(form) << " kind "
                   << static_cast<int>(kind));
      const BlipFit truth = make_positive_truth(kind, form);
      std::vector<double> xs, ys;
      sample(truth, 1.0, 5.5, 400, xs, ys);
      CompactConfig cfg;
      cfg.rho_form = form;
      CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg, kind);

      BlipFit f = truth;
      f.form = RhoForm::Signed;  // the fitter stamps it
      std::mt19937 rng(11);
      std::normal_distribution<double> g(0.0, 1.0);
      for (int k = 0; k < f.n(); ++k) {
        f.mu[k] += 0.04 * g(rng);
        f.eta[k] *= std::exp(0.04 * g(rng));
      }
      fit.solve_coeffs(f);
      const double before = f.curve_rmse;
      f.c[1] = -0.2;  // an infeasible start is lifted to the bound
      const double after = fit.refine(f);
      INFO("before = " << before << " after = " << after);
      REQUIRE(f.form == form);
      REQUIRE(after < before);
      REQUIRE(after < 1e-8);  // representable target, exact Jacobians
      for (double c : f.c) REQUIRE(c >= 0.0);
    }
  }
}

TEST_CASE("compact fitter RHOFORM: refine on a table outside the form ends "
          "at the bound", "[compact][rhoform]") {
  // the signed truth needs negative coefficients: under NONNEG the refine
  // must stay feasible (Ceres lower bounds + the final non-negative solve)
  const BlipFit truth = make_truth();
  std::vector<double> xs, ys;
  sample(truth, 1.0, 5.5, 300, xs, ys);
  for (const char *algo : {"CERES_LM", "CERES_DOGLEG", "NLOPT_LBFGS"}) {
    INFO(algo);
    CompactConfig cfg;
    cfg.rho_form = RhoForm::NonNeg;
    cfg.refine_algo = algo;
    CompactBlipFitter fit(xs, ys, {}, 1.0, 5.5, cfg);
    BlipFit f = truth;
    fit.solve_coeffs(f);
    const double before = f.curve_rmse;
    const double after = fit.refine_with_cfg(f);
    INFO("before = " << before << " after = " << after);
    REQUIRE(f.form == RhoForm::NonNeg);
    REQUIRE(after <= before * (1.0 + 1e-12));
    for (double c : f.c) REQUIRE(c >= 0.0);
    for (double x = 0.0; x <= 7.0; x += 0.01) REQUIRE(f.eval(x) >= 0.0);
  }
}

TEST_CASE("compact fitter RHOFORM: joint refine bounds the density "
          "amplitudes", "[compact][rhoform]") {
  for (RhoForm form : {RhoForm::NonNeg, RhoForm::Monotone}) {
    for (BasisKind kind : {BasisKind::Blip, BasisKind::Blip5}) {
      INFO("form " << tm::rhoform_name(form) << " kind "
                   << static_cast<int>(kind));
      BlipFit phi = make_truth();
      BlipFit rho;
      rho.kind = kind;
      rho.form = form;
      rho.mu = {2.4, 3.2, 4.2};
      rho.eta = {1.2, 0.9, 1.1};
      rho.c = {0.4, 0.25, 0.1};
      BlipFit F;
      F.mu = {1.0, 4.0, 7.0};
      F.eta = {0.45, 0.4, 0.45};
      F.c = {-2.0, -0.8, 0.5};

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
      JointCurveData dphi, drho, dF;
      sample(phi, 1.0, 5.5, 300, dphi.xs, dphi.ys);
      sample(rho, 1.0, 5.5, 300, drho.xs, drho.ys);
      sample(F, 0.0, 9.0, 300, dF.xs, dF.ys);

      // the form is a density property: phi / F must stay SIGNED
      {
        BlipFit bad = phi, r2 = rho, F2 = F;
        bad.form = form;
        REQUIRE_THROWS_WITH(
            joint_refine(bad, r2, F2, dphi, drho, dF, dists, e0, ew, 1.0, 10,
                         1e-15),
            Catch::Contains("density rho only"));
      }

      std::mt19937 rng(23);
      std::normal_distribution<double> g(0.0, 1.0);
      auto perturb = [&](BlipFit &f) {
        for (int k = 0; k < f.n(); ++k) {
          f.mu[k] += 0.03 * g(rng);
          f.eta[k] *= std::exp(0.03 * g(rng));
          f.c[k] *= 1.0 + 0.03 * g(rng);
        }
      };
      perturb(phi);
      perturb(rho);
      perturb(F);
      const double eta_floor = 0.03 * (rho.n() - 1) / 4.5;
      {
        // infeasible starts are lifted to the bounds, c >= 0 and
        // eta >= 0.03 eta0 (here a mirrored width), before any solve
        BlipFit p2 = phi, r2 = rho, F2 = F;
        r2.c[2] = -0.05;
        r2.eta[1] = -0.9;
        joint_refine(p2, r2, F2, dphi, drho, dF, dists, e0, ew, 1.0, 10,
                     1e-15, "NONE");
        REQUIRE(r2.c[2] == 0.0);
        REQUIRE(r2.eta[1] == Approx(eta_floor).epsilon(1e-12));
      }
      rho.c[2] = -0.05;  // an infeasible start is lifted to the bound
      const double final_max =
          joint_refine(phi, rho, F, dphi, drho, dF, dists, e0, ew,
                       /*curve_weight=*/1.0, /*max_iter=*/500, /*ftol=*/1e-15);
      INFO("final max |dE| = " << final_max);
      REQUIRE(final_max < 1e-8);
      REQUIRE(rho.form == form);
      for (int k = 0; k < rho.n(); ++k) {
        REQUIRE(rho.c[k] >= 0.0);
        REQUIRE(rho.eta[k] >= eta_floor * (1.0 - 1e-12));
      }
    }
  }
}

TEST_CASE("compact fitter RHOFORM: refusals", "[compact][rhoform]") {
  std::vector<double> xs, ys;
  sample(make_knot_truth(), 1.0, 5.5, 100, xs, ys);
  // MONOTONE has no step form for the knot kinds
  CompactConfig mono;
  mono.rho_form = RhoForm::Monotone;
  REQUIRE_THROWS_WITH(
      CompactBlipFitter(xs, ys, {}, 1.0, 5.5, mono, BasisKind::Knot5L),
      Catch::Contains("RHOFORM MONOTONE is not defined for the one-sided"));
  REQUIRE_THROWS_WITH(make_curve_cost(xs, ys, {}, 2, BasisKind::Knot5L, false,
                                      {}, RhoForm::Monotone),
                      Catch::Contains("no step form"));
  REQUIRE_THROWS_WITH(
      make_energy_cost({{2.5}}, {0.0}, {1.0}, 1, 1, 1, BasisKind::Blip,
                       BasisKind::Knot5L, BasisKind::Blip, false, {},
                       RhoForm::Monotone),
      Catch::Contains("no step form"));
  // NONNEG with the knot kinds is fine (non-negative knot amplitudes)
  CompactConfig nn;
  nn.rho_form = RhoForm::NonNeg;
  REQUIRE_NOTHROW(
      CompactBlipFitter(xs, ys, {}, 1.0, 5.5, nn, BasisKind::Knot5L));
  // a density form and the EMBF0 zero reference never share a fit
  CompactConfig both = nn;
  both.zero_ref = true;
  REQUIRE_THROWS_WITH(CompactBlipFitter(xs, ys, {}, 1.0, 5.5, both),
                      Catch::Contains("cannot share a fit"));
  REQUIRE_THROWS_WITH(make_curve_cost(xs, ys, {}, 2, BasisKind::Blip, true,
                                      {}, RhoForm::NonNeg),
                      Catch::Contains("cannot share a fit"));
  // joint_refine: a MONOTONE knot density
  BlipFit phi = make_knot_truth(), rho = make_knot_truth(), F;
  rho.form = RhoForm::Monotone;
  F.kind = BasisKind::Knot5R;
  F.mu = {0.0};
  F.eta = {1.0};
  F.c = {1.0};
  JointCurveData d;
  d.xs = xs;
  d.ys = ys;
  REQUIRE_THROWS_WITH(joint_refine(phi, rho, F, d, d, d, {{2.5}}, {0.0},
                                   {1.0}, 1.0, 10, 1e-15),
                      Catch::Contains("no step form"));
}

TEST_CASE("compact fitter RHOFORM: ladders per form on a density-like table",
          "[compact][rhoform]") {
  // rho-like: positive, decreasing, a shoulder near the first shells; vanishes
  // at the cutoff 5.5 (hard_rcut: the bump-support gate of the engine)
  const double rcut = 5.5;
  const int N = 300;
  std::vector<double> xs(N), ys(N);
  for (int i = 0; i < N; ++i) {
    const double x = 1.5 + (rcut - 1.5) * i / (N - 1);
    xs[i] = x;
    const double s = 1.0 - x / rcut;
    ys[i] = s * s * s * (1.0 + 0.5 * std::exp(-4.0 * (x - 2.8) * (x - 2.8)));
  }
  for (RhoForm form : {RhoForm::Signed, RhoForm::NonNeg, RhoForm::Monotone}) {
    INFO("form " << tm::rhoform_name(form));
    CompactConfig cfg;
    cfg.rho_form = form;
    cfg.n_max = 8;
    cfg.jitters = 1;
    cfg.sbplx_maxeval = 1500;  // keep the unit test quick
    cfg.mlsl_maxeval = 3000;
    cfg.ceres_max_iter = 200;
    cfg.target = 1.0;
    cfg.hard_rcut = rcut;
    CompactBlipFitter fit(xs, ys, {}, 1.5, rcut, cfg, BasisKind::Blip5);
    std::vector<LadderRow> trace;
    const BlipFit best = fit.fit(
        [&](const BlipFit &f) { return f.curve_rmse / 1e-4; }, trace);
    INFO("n = " << best.n() << " rmse = " << best.curve_rmse);
    REQUIRE(best.form == form);
    REQUIRE(best.curve_rmse < 5e-4);  // |rho| ~ 0.4: a real fit
    for (int k = 0; k < best.n(); ++k) {
      REQUIRE(best.mu[k] + 3.0 / best.eta[k] <= rcut * (1.0 + 1e-6));
      if (form != RhoForm::Signed) REQUIRE(best.c[k] >= 0.0);
    }
    if (form == RhoForm::Signed) continue;
    // the guarantees on a fine grid: rho >= 0 everywhere, and non-increasing
    // to rounding under MONOTONE
    double prev = best.eval(0.0), scale = 0.0;
    for (double c : best.c) scale += c;
    for (int i = 1; i <= 20000; ++i) {
      const double x = 7.0 * i / 20000.0;
      const double v = best.eval(x);
      REQUIRE(v >= 0.0);
      if (form == RhoForm::Monotone)
        REQUIRE(v - prev <= 8.0 * std::numeric_limits<double>::epsilon() * scale);
      prev = v;
    }
  }
}

// ====== RHOFORM SIGNED: the numbers of the fitter before RHOFORM, bit for bit ======

#include "compact_fitter_signed_goldens.h"

#include <cstdint>
#include <cstring>
#include <sstream>

namespace {
// The goldens were produced by the fitter before RHOFORM (HPO 3ce1001) in the
// HPO library's own build: -O3 without -march=native or -ffast-math, where
// every value below is reproducible to the bit (no libm call on these paths
// except inside Ceres), so SIGNED must match them bit for bit (DESIGN 9).
// A build with -ffast-math or a wider vector ISA (FMA / AVX Eigen kernels)
// rounds differently (measured: 1e-14 relative on the closed-form values);
// there every value is compared to 1e-6 relative, and the ladder, whose
// derivative-free path follows every rounding, is not compared.
#if defined(__x86_64__) && !defined(__FAST_MATH__) && !defined(__FMA__) && \
    !defined(__AVX__)
constexpr bool golden_bits = true;
#else
constexpr bool golden_bits = false;
#endif

// `got` is `want`: the same bits in the golden build, else within 1e-6
// relative. Reports the first differing entry.
void check_golden(const std::vector<double> &got,
                  const std::vector<double> &want, const char *name) {
  INFO("golden " << name);
  REQUIRE(got.size() == want.size());
  std::size_t at = 0;
  for (; at < got.size(); ++at) {
    std::uint64_t a, b;
    std::memcpy(&a, &got[at], sizeof a);
    std::memcpy(&b, &want[at], sizeof b);
    const bool same =
        golden_bits ? a == b
                    : std::fabs(got[at] - want[at]) <=
                          1e-6 * std::max(1.0, std::fabs(want[at]));
    if (!same) break;
  }
  std::ostringstream first;
  if (at < got.size())
    first << "entry " << at << ": " << std::hexfloat << got[at]
          << " vs golden " << want[at];
  INFO(first.str());
  CHECK(at == got.size());
}

// The golden table: a rational, libm-free function that changes sign on
// [1, 5.5] and lies outside every basis; row weights 1, 1.5, 2, ...
void golden_table(int N, std::vector<double> &xs, std::vector<double> &ys,
                  std::vector<double> &wts) {
  xs.resize(N);
  ys.resize(N);
  wts.resize(N);
  for (int i = 0; i < N; ++i) {
    const double x = 1.0 + 4.5 * i / (N - 1);
    xs[i] = x;
    ys[i] = 1.0 / (1.0 + x * x) - 0.3;
    wts[i] = 1.0 + 0.5 * (i % 3);
  }
}

BlipFit golden_basis(BasisKind kind, std::vector<double> mu,
                     std::vector<double> eta) {
  BlipFit f;
  f.kind = kind;
  f.mu = std::move(mu);
  f.eta = std::move(eta);
  f.c.assign(f.mu.size(), 0.0);
  return f;
}

// residuals, then every Jacobian block (row-major), in block order
std::vector<double> golden_cost_values(
    ceres::CostFunction *cost, std::vector<std::vector<double>> params) {
  const int M = cost->num_residuals();
  const auto &sizes = cost->parameter_block_sizes();
  std::vector<double *> p;
  for (auto &b : params) p.push_back(b.data());
  std::vector<std::vector<double>> jac(sizes.size());
  std::vector<double *> jp;
  for (std::size_t b = 0; b < sizes.size(); ++b) {
    jac[b].assign(static_cast<std::size_t>(M) * sizes[b], 0.0);
    jp.push_back(jac[b].data());
  }
  std::vector<double> out(M);
  REQUIRE(cost->Evaluate(p.data(), out.data(), jp.data()));
  for (const auto &j : jac) out.insert(out.end(), j.begin(), j.end());
  return out;
}

std::vector<double> golden_params(std::vector<double> head,
                                  std::initializer_list<const BlipFit *> fs) {
  for (const BlipFit *f : fs) {
    head.insert(head.end(), f->c.begin(), f->c.end());
    head.insert(head.end(), f->mu.begin(), f->mu.end());
    head.insert(head.end(), f->eta.begin(), f->eta.end());
  }
  return head;
}
}  // namespace

TEST_CASE("compact fitter RHOFORM: SIGNED reproduces the numbers of the "
          "fitter before RHOFORM",
          "[compact][rhoform]") {
  namespace golden = compact_signed_golden;
  INFO("golden comparison: " << (golden_bits
                                     ? "bit for bit"
                                     : "1e-6 relative (not the golden build)"));
  std::vector<double> xs, ys, wts;
  golden_table(120, xs, ys, wts);
  const CompactConfig cfg{};  // SIGNED, value objective

  SECTION("closed-form solve, evaluation and VARPRO gradient, per kind") {
    struct Case {
      const char *name;
      BlipFit basis;
      const std::vector<double> *want;
    };
    const std::vector<Case> cases = {
        {"blip", golden_basis(BasisKind::Blip, {1.6, 2.5, 3.4, 4.3, 5.0},
                              {1.1, 0.9, 1.0, 1.2, 1.4}), &golden::blip},
        {"blip5", golden_basis(BasisKind::Blip5, {1.5, 2.6, 3.7, 4.6},
                               {0.8, 0.9, 1.0, 1.1}), &golden::blip5},
        {"knot5", golden_basis(BasisKind::Knot5L, {2.0, 3.0, 4.0, 5.5},
                               {1.0, 1.0, 1.0, 1.0}), &golden::knot5}};
    for (const auto &[name, basis, want] : cases) {
      CompactBlipFitter fit(xs, ys, wts, 1.0, 5.5, cfg, basis.kind);
      BlipFit f = basis;
      const double rmse = fit.solve_coeffs(f);
      REQUIRE(f.form == RhoForm::Signed);
      std::vector<double> v = f.c;
      v.push_back(rmse);
      for (double x : {1.3, 2.75, 4.1}) {
        v.push_back(f.eval(x));
        v.push_back(f.deriv(x));
        v.push_back(f.deriv2(x));
      }
      std::vector<double> grad;
      BlipFit g = basis;
      v.push_back(fit.reduced_mse(g, &grad));
      v.insert(v.end(), grad.begin(), grad.end());
      check_golden(v, *want, name);
    }
  }

  SECTION("stacked value + d1 + d2 solve (curve objective)") {
    CompactConfig cc;
    cc.objective = RefitObjective::Curve;
    cc.curve_deriv_weight = 1.0;
    CompactBlipFitter fit(xs, ys, wts, 1.0, 5.5, cc);
    BlipFit f = golden_basis(BasisKind::Blip, {1.6, 2.5, 3.4, 4.3, 5.0},
                             {1.1, 0.9, 1.0, 1.2, 1.4});
    std::vector<double> v = {fit.solve_coeffs(f)};
    v.insert(v.end(), f.c.begin(), f.c.end());
    v.push_back(fit.d2_rmse(f));
    v.push_back(fit.combined_curve_score(f));
    check_golden(v, golden::stacked, "stacked");
  }

  SECTION("curve and energy costs: residuals and Jacobians") {
    std::vector<double> x6, y6, w6;
    golden_table(6, x6, y6, w6);
    std::unique_ptr<ceres::CostFunction> curve(
        make_curve_cost(x6, y6, w6, 3, BasisKind::Blip5));
    check_golden(golden_cost_values(curve.get(), {{0.3, -0.7, 0.2},
                                                  {1.8, 3.1, 4.4},
                                                  {0.7, 0.9, 1.2}}),
                 golden::curve_cost, "curve_cost");
    const std::vector<std::vector<double>> dists = {
        {2.5, 2.5, 2.9, 2.9, 3.6}, {2.1, 2.1, 2.45, 2.45, 3.0}};
    const std::vector<double> e0 = {-1.0, -0.5}, ew = {1.0, 2.0};
    std::unique_ptr<ceres::CostFunction> energy(
        make_energy_cost(dists, e0, ew, 2, 2, 2));  // holds the references
    check_golden(golden_cost_values(energy.get(),
                                    {{1.5, -0.6}, {2.0, 3.4}, {0.9, 1.1},
                                     {0.6, -0.2}, {2.2, 3.5}, {0.8, 0.9},
                                     {-2.0, 0.5}, {1.0, 4.0}, {0.45, 0.4}}),
                 golden::energy_cost, "energy_cost");
  }

  SECTION("Ceres refine from a displaced start") {
    // a signed truth plus a small out-of-basis term: the optimum is near the
    // truth, not on it
    BlipFit truth = golden_basis(BasisKind::Blip, {1.5, 2.4, 3.6, 4.8},
                                 {1.2, 0.9, 0.8, 1.1});
    truth.c = {2.0, -1.3, 0.7, -0.4};
    std::vector<double> yr(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
      yr[i] = truth.eval(xs[i]) + 0.02 / (1.0 + xs[i] * xs[i]);
    CompactBlipFitter fit(xs, yr, wts, 1.0, 5.5, cfg);
    BlipFit f = golden_basis(BasisKind::Blip, {1.55, 2.35, 3.65, 4.75},
                             {1.15, 0.95, 0.85, 1.05});
    const double rmse = fit.refine(f);
    check_golden(golden_params({rmse}, {&f}), golden::refine, "refine");
  }

  SECTION("joint refine with a negative density amplitude") {
    BlipFit phi = golden_basis(BasisKind::Blip, {1.5, 2.4, 3.6, 4.8},
                               {1.2, 0.9, 0.8, 1.1});
    phi.c = {2.0, -1.3, 0.7, -0.4};
    BlipFit rho =
        golden_basis(BasisKind::Blip, {2.0, 3.2, 4.4}, {0.9, 0.8, 0.9});
    rho.c = {0.6, -0.2, 0.15};
    BlipFit F =
        golden_basis(BasisKind::Blip, {1.0, 4.0, 7.0}, {0.45, 0.4, 0.45});
    F.c = {-2.0, -0.8, 0.5};
    const std::vector<std::vector<double>> dists = {
        {2.5, 2.5, 2.9, 2.9, 3.6}, {2.3, 2.3, 2.7, 2.7, 3.3},
        {2.1, 2.1, 2.45, 2.45, 3.0}};
    std::vector<double> e0, ew = {1.0, 2.0, 0.5};
    for (const auto &dd : dists) {
      double hp = 0.0, ri = 0.0;
      for (double r : dd) {
        hp += 0.5 * phi.eval(r);
        ri += rho.eval(r);
      }
      e0.push_back(hp + F.eval(ri));
    }
    JointCurveData dphi, drho, dF;
    auto curve = [](const BlipFit &f, double lo, double hi, JointCurveData &d) {
      for (int i = 0; i < 150; ++i) {
        d.xs.push_back(lo + (hi - lo) * i / 149);
        d.ys.push_back(f.eval(d.xs.back()));
      }
    };
    curve(phi, 1.0, 5.5, dphi);
    curve(rho, 1.0, 5.5, drho);
    curve(F, 0.0, 9.0, dF);
    auto displace = [](BlipFit &f, double s) {
      for (int k = 0; k < f.n(); ++k) {
        f.mu[k] += s * (k % 2 ? -1.0 : 1.0);
        f.eta[k] *= 1.0 + s;
        f.c[k] *= 1.0 - s;
      }
    };
    displace(phi, 0.03);
    displace(rho, -0.02);
    displace(F, 0.04);
    const double mx = joint_refine(phi, rho, F, dphi, drho, dF, dists, e0, ew,
                                   1.0, 500, 1e-15);
    check_golden(golden_params({mx}, {&phi, &rho, &F}), golden::joint, "joint");
  }

  SECTION("a short ladder (bit for bit in the golden build only)") {
    if (!golden_bits) return;
    CompactConfig lc;
    lc.n_max = 5;
    lc.jitters = 0;
    lc.mlsl_maxeval = 2000;
    lc.sbplx_maxeval = 400;
    lc.ceres_max_iter = 100;
    lc.stall_rungs = 2;
    CompactBlipFitter fit(xs, ys, wts, 1.0, 5.5, lc);
    std::vector<LadderRow> trace;
    const BlipFit f = fit.fit(
        [](const BlipFit &b) { return b.curve_rmse / 1e-5; }, trace);
    check_golden(
        golden_params({static_cast<double>(f.n()), f.curve_rmse}, {&f}),
        golden::ladder, "ladder");
  }
}

TEST_CASE("compact fitter RHOFORM NONNEG: a coefficient solve that stops "
          "before a Kuhn-Tucker point is counted and reported",
          "[compact][rhoform][nnls]") {
  // Two Blip columns that differ only by a few ulps of eta: the second is
  // numerically dependent on the first (models::nnls refuses it), yet with
  // y = a1 + e (da/deta) its dual is above the tolerance once the first has
  // entered. nnls then stops by refusal: the coefficients are feasible but
  // not a proven Kuhn-Tucker point. The fitter must count such a solve
  // (nonneg_unproven_solves) and fit() report it, not accept it silently.
  // The eta offset is searched (deterministically) for a refusal stop,
  // confirmed on the same system with models::nnls. (REQUIRE only: after
  // <ceres/ceres.h> above, CHECK is miniglog's, not Catch's.)
  const std::vector<double> xs = {0.35, 0.6, 0.85, 1.1, 1.35, 1.6};
  const double mu = 1.75, eta0 = 1.0;
  auto column = [&xs](double m, double e) {
    std::vector<double> a(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
      BlipFit unit;
      unit.kind = BasisKind::Blip;
      unit.mu = {m};
      unit.eta = {e};
      unit.c = {1.0};
      a[i] = unit.eval(xs[i]);
    }
    return a;
  };
  const std::vector<double> a1 = column(mu, eta0);
  // d a / d eta by central differences (the direction the offset moves a2)
  const double h = 1e-6;
  const std::vector<double> ap = column(mu, eta0 + h), am = column(mu, eta0 - h);
  std::vector<double> u(xs.size());
  double ua = 0.0, uu = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) {
    u[i] = (ap[i] - am[i]) / (2.0 * h);
    ua += u[i] * a1[i];
    uu += u[i] * u[i];
  }
  REQUIRE(ua < 0.0);  // a narrower bump: every value down
  const double e = -0.2 * ua / uu;  // a1 still enters first
  std::vector<double> ys(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) ys[i] = a1[i] + e * u[i];

  bool found = false;
  for (int k = 4; k <= 400 && !found; ++k) {
    const double eta1 = eta0 + k * std::numeric_limits<double>::epsilon();
    const std::vector<double> a2 = column(mu, eta1);
    std::vector<double> A(a1);
    A.insert(A.end(), a2.begin(), a2.end());
    std::vector<double> x(2);
    const tadah::models::NnlsResult ref = tadah::models::nnls(
        A.data(), xs.size(), 2, ys.data(), x.data());
    if (ref.stop != tadah::models::NnlsStop::Refused) continue;
    found = true;
    INFO("eta offset " << k << " eps");
    REQUIRE_FALSE(ref.converged);

    std::vector<std::string> log;
    CompactConfig cfg;
    cfg.rho_form = RhoForm::NonNeg;
    cfg.log_fn = [&log](const std::string &line) { log.push_back(line); };
    CompactBlipFitter fit(xs, ys, {}, 0.3, 1.7, cfg, BasisKind::Blip);
    BlipFit f;
    f.kind = BasisKind::Blip;
    f.mu = {mu, mu};
    f.eta = {eta0, eta1};
    f.c = {0.0, 0.0};
    const long long before = fit.nonneg_solves();
    fit.solve_coeffs(f);
    REQUIRE(fit.nonneg_solves() == before + 1);
    REQUIRE(fit.nonneg_unproven_solves() == 1);
    REQUIRE(f.c[0] == x[0]);  // the solution nnls returned, feasible
    REQUIRE(f.c[1] == x[1]);
    REQUIRE(f.c[0] >= 0.0);
    REQUIRE(f.c[1] >= 0.0);
  }
  REQUIRE(found);
}

#endif  // TADAH_USE_CERES

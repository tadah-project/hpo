#include <tadah/hpo/engines/hpo_refit_engine.h>
#include <tadah/mlip/engines/refit_log.h>

#include <tadah/core/logger.h>
#include <tadah/models/eam/eam_lattice.h>

#ifdef TADAH_USE_CERES
#include <tadah/hpo/refit/compact_fitter.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {
namespace engines {

#ifndef TADAH_USE_CERES

bool HPORefitEngine::compact_fit(const tadah::core::Context &,
                                 const tadah::models::EAMSetflIO &,
                                 tadah::models::EAMRefitConfig &,
                                 std::unique_ptr<tadah::models::EAMRefit> &,
                                 SelfTest &) {
  return false;  // no Ceres in this build -> dense fallback (base warns)
}

#else  // TADAH_USE_CERES

namespace {
using tadah::core::Context;
using tadah::core::Logger;
using tadah::core::LogLevel;
using tadah::models::EAMRefit;
using tadah::models::EAMRefitConfig;
using tadah::models::EAMSetflIO;
namespace lat = tadah::models::eam_lattice;
using namespace tadah::hpo::refit;

double cget(const Context &c, const std::string &k, double def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<double>(k) : def;
}
int iget(const Context &c, const std::string &k, int def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<int>(k) : def;
}

std::string up_(std::string s) {
  for (char &ch : s) ch = static_cast<char>(std::toupper((unsigned char)ch));
  return s;
}
void check_algo(const std::string &stage, const std::string &v,
                std::initializer_list<const char *> vocab) {
  for (const char *w : vocab)
    if (v == w) return;
  std::string list;
  for (const char *w : vocab) list += (list.empty() ? "" : "|") + std::string(w);
  throw std::runtime_error("OPTIMIZER " + stage + ": unknown ALGO '" + v +
                           "' (" + list + ")");
}

// Parse the inline OPTIMIZER block (token stream captured verbatim by the
// config parser between OPTIMIZER and ENDOPTIMIZER) into CompactConfig.
// Grammar (subblocks and attributes all optional; keywords case-insensitive):
//   GLOBAL  ALGO <a> [MAXEVAL <n>]                              ENDGLOBAL
//   EXPLORE ALGO <a> [MAXEVAL <n>] [FTOL_REL <v>] [XTOL_REL <v>] ENDEXPLORE
//   POLISH  ALGO <a> [MAXITER <n>] [FTOL <v>] [GTOL <v>]         ENDPOLISH
//   JOINT   ALGO <a> [MAXITER <n>] [FTOL <v>]                    ENDJOINT
// The GLOBAL stage's inner local refinement (MLSL) reuses the EXPLORE
// tolerances. (GLOBAL maps to the CompactConfig seed_* fields internally.)
void parse_optimizer_block(const std::vector<std::string> &tok,
                           CompactConfig &cfg) {
  auto as_int = [](const std::string &s, const char *attr) {
    try {
      size_t p = 0;
      int v = std::stoi(s, &p);
      if (p != s.size()) throw std::invalid_argument(s);
      return v;
    } catch (...) {
      throw std::runtime_error(std::string("OPTIMIZER: ") + attr +
                               " expects an integer, got '" + s + "'");
    }
  };
  auto as_dbl = [](const std::string &s, const char *attr) {
    try {
      size_t p = 0;
      double v = std::stod(s, &p);
      if (p != s.size()) throw std::invalid_argument(s);
      return v;
    } catch (...) {
      throw std::runtime_error(std::string("OPTIMIZER: ") + attr +
                               " expects a number, got '" + s + "'");
    }
  };
  size_t i = 0;
  while (i < tok.size()) {
    const std::string blk = up_(tok[i]);
    if (blk != "GLOBAL" && blk != "EXPLORE" && blk != "POLISH" &&
        blk != "JOINT")
      throw std::runtime_error("OPTIMIZER: unknown subblock '" + tok[i] +
                               "' (GLOBAL|EXPLORE|POLISH|JOINT)");
    const std::string end = "END" + blk;
    ++i;  // consume subblock opener
    bool closed = false;
    while (i < tok.size()) {
      const std::string a = up_(tok[i]);
      if (a == end) {
        ++i;
        closed = true;
        break;
      }
      if (i + 1 >= tok.size())
        throw std::runtime_error("OPTIMIZER " + blk + ": missing value after '" +
                                 tok[i] + "'");
      const std::string &val = tok[i + 1];
      if (a == "ALGO") {
        const std::string A = up_(val);
        if (blk == "GLOBAL") {
          check_algo("GLOBAL", A, {"MLSL", "DIRECT", "CRS2", "DLIB_GLOBAL",
                                   "UNIFORM"});
          cfg.seed_algo = A;
        } else if (blk == "EXPLORE") {
          check_algo("EXPLORE", A, {"SBPLX", "NELDERMEAD", "BOBYQA", "PRAXIS",
                                    "NLOPT_LBFGS", "DLIB_LBFGS", "NONE"});
          cfg.explore_algo = A;
        } else if (blk == "POLISH") {
          check_algo("POLISH", A, {"CERES_LM", "CERES_DOGLEG", "NLOPT_LBFGS",
                                   "DLIB_LBFGS", "NONE"});
          cfg.polish_algo = A;
        } else {
          check_algo("JOINT", A, {"CERES_DOGLEG", "CERES_LM", "NONE"});
          cfg.joint_algo = A;
        }
      } else if (a == "MAXEVAL" && (blk == "GLOBAL" || blk == "EXPLORE")) {
        const int n = as_int(val, "MAXEVAL");
        if (blk == "GLOBAL") cfg.mlsl_maxeval = n;
        else cfg.sbplx_maxeval = n;
      } else if (a == "FTOL_REL" && blk == "EXPLORE") {
        cfg.sbplx_ftol = as_dbl(val, "FTOL_REL");
      } else if (a == "XTOL_REL" && blk == "EXPLORE") {
        cfg.sbplx_xtol = as_dbl(val, "XTOL_REL");
      } else if (a == "MAXITER" && (blk == "POLISH" || blk == "JOINT")) {
        const int n = as_int(val, "MAXITER");
        if (blk == "POLISH") cfg.ceres_max_iter = n;
        else cfg.joint_max_iter = n;
      } else if (a == "FTOL" && (blk == "POLISH" || blk == "JOINT")) {
        const double v = as_dbl(val, "FTOL");
        if (blk == "POLISH") cfg.ceres_ftol = v;
        else cfg.joint_ftol = v;
      } else if (a == "GTOL" && blk == "POLISH") {
        cfg.ceres_gtol = as_dbl(val, "GTOL");
      } else {
        throw std::runtime_error("OPTIMIZER " + blk + ": unknown keyword '" +
                                 tok[i] + "'");
      }
      i += 2;
    }
    if (!closed)
      throw std::runtime_error("OPTIMIZER: " + blk + " not closed with " + end);
  }
}
}  // namespace

bool HPORefitEngine::compact_fit(const tadah::core::Context &c,
                                 const tadah::models::EAMSetflIO &eam,
                                 tadah::models::EAMRefitConfig &cfg,
                                 std::unique_ptr<tadah::models::EAMRefit> &refit,
                                 SelfTest &st) {
  // ---- reachability first: compact mode NEEDS the energy metric ----
  EAMRefit probe(eam, cfg);
  probe.analyze_reachability();
  if (!probe.lattice_known() || eam.lattice_param() <= 0.0 ||
      probe.r_reach() <= 0.0) {
    Logger::instance().log(LogLevel::WARNING,
        "tadah refit[compact]: header lattice '" + eam.lattice() +
        "' / a0 unusable for the crystal energy metric — compact mode needs "
        "it for rung selection.");
    return false;
  }
  const double rcut = eam.rcut();
  const double a0 = eam.lattice_param();
  const std::string lattice = eam.lattice();
  const double r_reach = probe.r_reach();
  const double rho_reach = probe.rho_reach();
  const double smin = cfg.reach_smin;

  // fit windows: same margins as the dense path
  const double rwin_lo =
      std::max({0.8 * r_reach, cfg.r_lo_frac * rcut, eam.dr()});
  // the final EAMRefit joins the setfl continuations exactly at the window
  cfg.r_lo_frac = rwin_lo / rcut;

  // ---- basis family (REFIT_BASIS, validated by the base engine) ----
  // Knot5: one-sided quintic knot functions — phi/rho LEFT-active (vanish
  // C^4-smoothly at their knots, all knots <= rcut so the cutoff truncation
  // is a no-op), embedding F RIGHT-active (free at high density). eta is a
  // gauge for this family; the fitter pins it at 1.
  const bool knot5 = (cfg.basis == "Knot5" || cfg.basis == "KNOT5" ||
                      cfg.basis == "knot5");
  const bool knot3 = (cfg.basis == "Knot3" || cfg.basis == "KNOT3" ||
                      cfg.basis == "knot3");
  const bool knotmix = (cfg.basis == "KnotMix" || cfg.basis == "KNOTMIX" ||
                        cfg.basis == "knotmix" || cfg.basis == "Mixed" ||
                        cfg.basis == "MIXED");
  const bool knotfam = knot5 || knot3 || knotmix;
  const bool blip5 = (cfg.basis == "Blip5" || cfg.basis == "BLIP5" ||
                      cfg.basis == "blip5" || cfg.basis == "M6");
  const BasisKind k_rad = knotfam ? BasisKind::Knot5L
                        : blip5 ? BasisKind::Blip5
                                : BasisKind::Blip;
  const BasisKind k_emb = knotfam ? BasisKind::Knot5R
                        : blip5 ? BasisKind::Blip5
                                : BasisKind::Blip;

  // ---- ladder configuration (config-file keys) ----
  const double tol_amb = cget(c, "REFIT_TOL_AMBIENT", 0.1) * 1e-3;   // eV
  const double tol_cmp = cget(c, "REFIT_TOL_COMPRESSED", 1.0) * 1e-3;
  const double share = cget(c, "REFIT_COMPACT_SHARE", 0.5);
  CompactConfig lcfg;
  lcfg.n_max = iget(c, "REFIT_COMPACT_NMAX", 40);
  lcfg.seed = static_cast<unsigned>(iget(c, "REFIT_COMPACT_SEED", 12345));
  // knot-family ladder schedule (blip defaults = legacy, regression-gated).
  // Matrix-measured (Fe+Ta LAMMPS batteries, 4 schedules): per-rung
  // optimisation + energy-aware top-3 insertion + stall 12 ("D") is the most
  // consistent — worst-case |dCij| 1.6 GPa, EV_cmp <= 0.24 meV; the
  // insertion-only deep-growth schedule (REFIT_COMPACT_OPT_EVERY 999) is
  // spectacular when no ladder stalls (Ta: Cij <= 0.09 GPa, forces 7e-5)
  // but stalled Fe's F ladder — documented alternative, not the default.
  lcfg.opt_every = iget(c, "REFIT_COMPACT_OPT_EVERY", 1);
  lcfg.stall_rungs = iget(c, "REFIT_COMPACT_STALL", knotfam ? 12 : 6);
  lcfg.insert_topk = iget(c, "REFIT_COMPACT_INSERT_TOPK", knotfam ? 3 : 1);
  lcfg.knot_degree = knot3 ? 3 : 5;
  lcfg.knot_mix = knotmix;
  // Per-stage optimiser algorithm + budget + tolerances: defaults live in
  // CompactConfig; the optional inline OPTIMIZER block overrides them.
  if (c.contains("OPTIMIZER") && c.size("OPTIMIZER") > 0) {
    std::vector<std::string> tok(c.size("OPTIMIZER"));
    c.get("OPTIMIZER", tok);
    parse_optimizer_block(tok, lcfg);
  }
  const std::string &joint_algo = lcfg.joint_algo;
  auto sget = [&](const std::string &k, const std::string &def) {
    return c.contains(k) && c.size(k) > 0 ? c.get<std::string>(k) : def;
  };

  // ---- the refit optimisation log (REFIT_LOG; the user inspects this
  //      DURING the fit — every line is flushed) ----
  tadah::mlip::engines::RefitLog rlog;
  rlog.open(sget("REFIT_LOG", "refit_opt.log"), /*append=*/false);
  rlog.raw("# tadah refit optimisation log (compact mode)");
  rlog.raw("# one line per ladder event; metric <= 1 means within "
           "REFIT_COMPACT_SHARE x tolerance; d2_rmse tracks curvature");
  rlog.line("input element=" + eam.lattice() + " a0=" +
            core::to_string(a0, 4) + " rcut=" + core::to_string(rcut, 4));
  rlog.line("basis=" + cfg.basis + "  strategy global/explore/polish/joint = " +
            lcfg.seed_algo + "/" + lcfg.explore_algo + "/" + lcfg.polish_algo +
            "/" + joint_algo);
  // effective per-stage optimiser settings (defaults, or your OPTIMIZER block)
  auto g = [](double v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%.2g", v);
    return std::string(b);
  };
  rlog.line(
      "optimiser: global{algo=" + lcfg.seed_algo + " maxeval=" +
      std::to_string(lcfg.mlsl_maxeval) + "}  explore{algo=" +
      lcfg.explore_algo + " maxeval=" + std::to_string(lcfg.sbplx_maxeval) +
      " ftol_rel=" + g(lcfg.sbplx_ftol) + " xtol_rel=" + g(lcfg.sbplx_xtol) +
      "}  polish{algo=" + lcfg.polish_algo + " maxiter=" +
      std::to_string(lcfg.ceres_max_iter) + " ftol=" + g(lcfg.ceres_ftol) +
      " gtol=" + g(lcfg.ceres_gtol) + "}  joint{algo=" + joint_algo +
      " maxiter=" + std::to_string(lcfg.joint_max_iter) + " ftol=" +
      g(lcfg.joint_ftol) + "}");
  rlog.line("schedule: opt_every=" + std::to_string(lcfg.opt_every) +
            " stall=" + std::to_string(lcfg.stall_rungs) + " insert_topk=" +
            std::to_string(lcfg.insert_topk) +
            (lcfg.knot_mix ? "  KnotMix double-win detector ON" : "") +
            "  n_max=" + std::to_string(lcfg.n_max));
  rlog.line("windows: r in [" + core::to_string(rwin_lo, 4) + ", " +
            core::to_string(rcut, 4) + "]  rho in [0, " +
            core::to_string(rho_reach, 4) + "]  tolerances amb/cmp = " +
            core::to_string(tol_amb * 1e3, 3) + "/" +
            core::to_string(tol_cmp * 1e3, 3) + " meV/atom");

  // ---- the E-V scan backbone shared by metric + joint refinement ----
  // Rows 0..M_nat-1: the native lattice across the pressure regimes (these
  // drive the per-function ladder metric). The joint refinement additionally
  // sees the OTHER two lattices (phase enthalpies sample different r / rho
  // combinations — without these rows the compact fit nails the native scan
  // but drifts ~0.5 meV on fcc/hcp enthalpies; measured on Ta).
  const int M_nat = 57, M_alt = 16;  // 29 native rows left 1.25%-step gaps: Ti wiggled 1.5 meV between them (steep F-prime amplification); 0.6% steps close it
  std::vector<std::vector<double>> dists;
  std::vector<double> e_orig, e_w;
  std::vector<double> sum_phi, rho_i;  // original per-scan-point sums
  auto add_scan = [&](const std::string &l, double a_eq, int npts) {
    for (int i = 0; i < npts; ++i) {
      const double s = smin + (1.10 - smin) * i / (npts - 1);
      dists.push_back(lat::neighbour_distances(l, s * a_eq, rcut));
      double hp = 0.0, ri = 0.0;
      for (double r : dists.back()) {
        hp += 0.5 * eam.phi_tab(r);
        ri += eam.rho_tab(r);
      }
      sum_phi.push_back(hp);
      rho_i.push_back(ri);
      e_orig.push_back(hp + eam.F_tab(ri));
      e_w.push_back(1.0 / (s >= 0.95 ? tol_amb : tol_cmp));  // 1 == at tol
    }
  };
  add_scan(lattice, a0, M_nat);
  {
    // equal-atomic-volume lattice constants for the two non-native lattices
    const double vat =
        lattice == "fcc" ? a0 * a0 * a0 / 4.0
        : lattice == "hcp" ? std::sqrt(3.0) / 2.0 * a0 * a0 *
                                 (std::sqrt(8.0 / 3.0) * a0) / 2.0
                           : a0 * a0 * a0 / 2.0;
    for (const char *l : {"bcc", "fcc", "hcp"}) {
      if (l == lattice) continue;
      const double a_eq = std::string(l) == "bcc" ? std::cbrt(2.0 * vat)
                          : std::string(l) == "fcc" ? std::cbrt(4.0 * vat)
                                                    : std::cbrt(vat / 0.7071);
      add_scan(l, a_eq, M_alt);
    }
  }
  const int M = M_nat;  // ladder metrics use the native rows only

  // ---- dense curve samples on the windows ----
  const int N = 800;
  auto linspace_eval = [&](double lo, double hi,
                           const std::function<double(double)> &f,
                           std::vector<double> &xs, std::vector<double> &ys) {
    xs.resize(N);
    ys.resize(N);
    for (int i = 0; i < N; ++i) {
      xs[i] = lo + (hi - lo) * i / (N - 1);
      ys[i] = f(xs[i]);
    }
  };
  JointCurveData dphi, drho, dF;
  linspace_eval(rwin_lo, rcut, [&](double r) { return eam.phi_tab(r); },
                dphi.xs, dphi.ys);
  linspace_eval(rwin_lo, rcut, [&](double r) { return eam.rho_tab(r); },
                drho.xs, drho.ys);
  linspace_eval(0.0, rho_reach, [&](double x) { return eam.F_tab(x); },
                dF.xs, dF.ys);

  // ---- per-function ladders with energy-metric rung selection ----
  // metric: candidate function + ORIGINAL other two, normalised so that 1.0
  // == `share` of the user tolerance in every band (the joint refinement
  // afterwards handles the cross-function coupling).
  auto run_ladder = [&](const JointCurveData &d, double lo, double hi,
                        const CompactBlipFitter::Metric &metric,
                        const char *name, std::vector<LadderRow> &trace,
                        BasisKind kind) {
    CompactConfig fcfg = lcfg;
    fcfg.log_fn = [&rlog, name](const std::string &m) {
      rlog.line(std::string("[") + name + "] " + m);
    };
    rlog.line(std::string("---- ") + name + " ladder ----");
    CompactBlipFitter fitter(d.xs, d.ys, d.wts, lo, hi, fcfg, kind);
    BlipFit best = fitter.fit(metric, trace);
    Logger::instance().log(LogLevel::WARNING,
        std::string("tadah refit[compact]: ") + name + " ladder: n=" +
        std::to_string(best.n()) + "  curve_rmse=" +
        core::to_string(best.curve_rmse, 4) + "  energy_metric=" +
        core::to_string(best.metric, 4) + " (<=1 means within " +
        core::to_string(share, 2) + "x tolerance)");
    return best;
  };

  std::vector<LadderRow> tr_phi, tr_rho, tr_F;
  BlipFit fphi = run_ladder(
      dphi, rwin_lo, rcut,
      [&](const BlipFit &f) {
        double worst = 0.0;
        for (int i = 0; i < M; ++i) {
          double hp = 0.0;
          for (double r : dists[i]) hp += 0.5 * f.eval(r);
          worst = std::max(worst,
                           std::fabs(hp - sum_phi[i]) * e_w[i] / share);
        }
        return worst;
      },
      "phi", tr_phi, k_rad);

  BlipFit frho = run_ladder(
      drho, rwin_lo, rcut,
      [&](const BlipFit &f) {
        double worst = 0.0;
        for (int i = 0; i < M; ++i) {
          double ri = 0.0;
          for (double r : dists[i]) ri += f.eval(r);
          worst = std::max(worst, std::fabs(eam.F_tab(ri) - eam.F_tab(rho_i[i])) *
                                      e_w[i] / share);
        }
        return worst;
      },
      "rho", tr_rho, k_rad);

  BlipFit fF = run_ladder(
      dF, 0.0, rho_reach,
      [&](const BlipFit &f) {
        double worst = 0.0;
        for (int i = 0; i < M; ++i)
          worst = std::max(worst, std::fabs(f.eval(rho_i[i]) -
                                            eam.F_tab(rho_i[i])) *
                                      e_w[i] / share);
        return worst;
      },
      "F", tr_F, k_emb);

  // ---- joint refinement: curves + E-V residuals, analytic gradients ----
  rlog.line("---- joint refinement (curves + E-V residuals, " + joint_algo +
            ") ----");
  std::string joint_term;
  const double e_resid_max =
      joint_refine(fphi, frho, fF, dphi, drho, dF, dists, e_orig, e_w,
                   /*curve_weight=*/1.0, lcfg.joint_max_iter, lcfg.joint_ftol,
                   joint_algo, &joint_term);
  rlog.line("joint: max |dE|/tol = " + core::to_string(e_resid_max, 4) +
            "  params n2b/nrho/nF = " + std::to_string(fphi.n()) + "/" +
            std::to_string(frho.n()) + "/" + std::to_string(fF.n()) +
            "  term=" + joint_term);
  Logger::instance().log(LogLevel::WARNING,
      "tadah refit[compact]: joint refinement (curves + E-V residuals): "
      "max |dE|/tol = " + core::to_string(e_resid_max, 4) +
      "  params n2b/nrho/nF = " + std::to_string(fphi.n()) + "/" +
      std::to_string(frho.n()) + "/" + std::to_string(fF.n()) +
      " (basis " +
      std::string(knotmix ? "KnotMix" : knot3 ? "Knot3" : knot5 ? "Knot5"
                  : blip5 ? "Blip5" : "Blip") +
      "; strategy seed/explore/polish/joint = " + lcfg.seed_algo + "/" +
      lcfg.explore_algo + "/" + lcfg.polish_algo + "/" + joint_algo +
      "; control: REFIT_BASIS + REFIT_COMPACT_* keys)");

  // ---- ladder trace file ----
  {
    std::ofstream os("refit_ladder_trace.dat");
    if (os) {
      os << "# tadah refit compact-mode ladder trace\n"
         << "# function  n  curve_rmse  energy_metric(<=1==within "
         << share << "x tol)  seconds  kept\n";
      auto dump = [&](const char *nm, const std::vector<LadderRow> &tr) {
        for (const auto &r : tr)
          os << nm << " " << r.n << " " << r.curve_rmse << " " << r.metric
             << " " << r.seconds << " " << (r.kept ? 1 : 0) << "\n";
      };
      dump("phi", tr_phi);
      dump("rho", tr_rho);
      dump("F", tr_F);
    }
  }

  // ---- assemble the EAMRefit (injection) + self-test ----
  cfg.n2b = fphi.n();
  cfg.nrho = frho.n();
  cfg.nF = fF.n();
  refit = std::make_unique<EAMRefit>(eam, cfg);
  refit->analyze_reachability();
  refit->set_blip_fit(EAMRefit::Fn::PHI, fphi.mu, fphi.eta, fphi.c,
                      fphi.deg);
  refit->set_blip_fit(EAMRefit::Fn::RHO, frho.mu, frho.eta, frho.c,
                      frho.deg);
  refit->set_blip_fit(EAMRefit::Fn::F, fF.mu, fF.eta, fF.c, fF.deg);
  refit->finalize_injected(rwin_lo, rho_reach);
  refit->set_rho_join(rho_reach);  // setfl: original+offset beyond the window
  st = run_self_test(eam, *refit, smin);
  if (st.valid) {
    rlog.line("self-test (crystal E-V, a/a0 in [" +
              core::to_string(smin, 2) + ", 1.10]):");
    rlog.raw("    ambient    RMSE=" + core::to_string(1000.0 * st.rmse_a, 4) +
             " meV/atom  max=" + core::to_string(1000.0 * st.mx_a, 4));
    rlog.raw("    compressed RMSE=" + core::to_string(1000.0 * st.rmse_c, 4) +
             " meV/atom  max=" + core::to_string(1000.0 * st.mx_c, 4) +
             "  (worst at a/a0=" + core::to_string(st.worst_s, 3) + ")");
    rlog.raw("    cohesive: original=" + core::to_string(st.e_orig, 6) +
             " refit=" + core::to_string(st.e_fit, 6) + " delta=" +
             core::to_string(1000.0 * (st.e_fit - st.e_orig), 4) +
             " meV/atom");
  }
  rlog.line("outputs: pot.tadah (native), refit.eam (round-trip setfl), "
            "refit_curve_{phi,rho,F}.dat (value/d1/d2 inspection), "
            "refit_ladder_trace.dat, hpotarget");
  return true;
}

#endif  // TADAH_USE_CERES

}  // namespace engines
}  // namespace hpo
}  // namespace tadah

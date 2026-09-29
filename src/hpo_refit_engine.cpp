#include <tadah/hpo/engines/hpo_refit_engine.h>
#include <tadah/mlip/engines/refit_log.h>

#include <tadah/core/logger.h>
#include <tadah/models/eam/eam_lattice.h>

#ifdef TADAH_USE_CERES
#include <tadah/hpo/refit/compact_fitter.h>
#include <tadah/hpo/refit/refit_config.h>
#include <tadah/hpo/refit/refit_parser.h>
#include <tadah/models/descriptors/basis_kind.h>
#include <tadah/models/descriptors/dm/ream_density.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
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
using tadah::models::RhoForm;
using tadah::models::rhoform_is_positive;
using tadah::models::rhoform_name;
namespace lat = tadah::models::eam_lattice;
using namespace tadah::hpo::refit;

double cget(const Context &c, const std::string &k, double def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<double>(k) : def;
}
int iget(const Context &c, const std::string &k, int def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<int>(k) : def;
}
bool bget(const Context &c, const std::string &k, bool def) {
  return c.contains(k) && c.size(k) > 0 ? c.get<bool>(k) : def;
}

// The legacy inline OPTIMIZER-block parser (GLOBAL/EXPLORE/POLISH/JOINT
// subblocks) was replaced by the stage pipeline parser RefitParser (see
// refit_parser.{h,cpp}). The new INIT/EXPLORE/REFINE/FUSE blocks are parsed
// there into a RefitConfig, which compact_fit() flattens into CompactConfig
// (apply_stage_optimizers) and resolves into per-channel INIT seeds.

// Flatten the parsed EXPLORE/REFINE/FUSE OptimizerSpecs onto CompactConfig's
// scalar fields. EXPLORE outer = the global seed at rung 0 (absorbs the old
// GLOBAL stage); EXPLORE INNER = the per-rung local search.
void apply_stage_optimizers(const RefitConfig &rc, CompactConfig &lcfg) {
  if (rc.explore.has_opt) {
    const OptimizerSpec &o = rc.explore.opt;
    lcfg.seed_algo = refit_explore_outer_algo(o);
    if (o.stop.maxeval) lcfg.mlsl_maxeval = static_cast<int>(*o.stop.maxeval);
    if (o.inner) {
      const OptimizerSpec &in = *o.inner;
      lcfg.explore_algo = refit_explore_inner_algo(in);
      if (in.stop.maxeval) lcfg.sbplx_maxeval = static_cast<int>(*in.stop.maxeval);
      if (in.stop.ftol_rel) lcfg.sbplx_ftol = *in.stop.ftol_rel;
      if (in.stop.xtol_rel) lcfg.sbplx_xtol = *in.stop.xtol_rel;
    }
  }
  // ceres_ftol / joint_ftol feed Ceres' function_tolerance, which is RELATIVE
  // ((new_cost - old_cost) < ftol * old_cost) — Ceres has no absolute function
  // tolerance, so FTOL_REL is the only accepted key (FTOL / FTOL_ABS are
  // rejected by RefitParser).
  if (rc.refine.has_opt) {
    const OptimizerSpec &o = rc.refine.opt;
    lcfg.refine_algo = refit_refine_algo(o);
    if (o.stop.maxeval) lcfg.ceres_max_iter = static_cast<int>(*o.stop.maxeval);
    if (o.stop.ftol_rel) lcfg.ceres_ftol = *o.stop.ftol_rel;
    auto it = o.param.find("GTOL");
    if (it != o.param.end()) lcfg.ceres_gtol = it->second;
  }
  if (rc.fuse.has_opt) {
    const OptimizerSpec &o = rc.fuse.opt;
    lcfg.joint_algo = refit_fuse_algo(o);
    if (o.stop.maxeval) lcfg.joint_max_iter = static_cast<int>(*o.stop.maxeval);
    if (o.stop.ftol_rel) lcfg.joint_ftol = *o.stop.ftol_rel;
  }
  // Explicit per-stage ON/OFF (ENABLED false) is a master kill-switch: it WINS
  // over any configured optimiser (sets the stage's algo to NONE). If an
  // optimiser was also configured, log that it is being ignored so the override
  // is never silent. (EXPLORE cannot be disabled — enforced by the parser.)
  if (!rc.refine.enabled) {
    if (rc.refine.has_opt && refit_refine_algo(rc.refine.opt) != "NONE")
      Logger::instance().log(LogLevel::WARNING,
          "tadah refit: REFINE ENABLED false — stage DISABLED; the configured "
          "REFINE optimiser (" + refit_refine_algo(rc.refine.opt) +
          ") is ignored.");
    else
      Logger::instance().log(LogLevel::INFO,
          "tadah refit: REFINE disabled (ENABLED false).");
    lcfg.refine_algo = "NONE";
  }
  if (!rc.fuse.enabled) {
    if (rc.fuse.has_opt && refit_fuse_algo(rc.fuse.opt) != "NONE")
      Logger::instance().log(LogLevel::WARNING,
          "tadah refit: FUSE ENABLED false — stage DISABLED; the configured "
          "FUSE optimiser (" + refit_fuse_algo(rc.fuse.opt) + ") is ignored.");
    else
      Logger::instance().log(LogLevel::INFO,
          "tadah refit: FUSE disabled (ENABLED false).");
    lcfg.joint_algo = "NONE";
  }
}

// Resolve one INIT anchor's centre on a channel's window [lo,hi] / target d.
// Returns false if it should be skipped (XMAX on the left-active phi/rho).
bool resolve_init_mu(const InitBasis &b, const JointCurveData &d, double lo,
                     double hi, double &mu_out) {
  auto arg_extreme = [&](bool want_max) {
    if (d.ys.empty()) return 0.5 * (lo + hi);
    std::size_t bi = 0;
    for (std::size_t i = 1; i < d.ys.size(); ++i)
      if (want_max ? (d.ys[i] > d.ys[bi]) : (d.ys[i] < d.ys[bi])) bi = i;
    return d.xs[bi];
  };
  auto arg_deriv = [&](int order) {  // location of max |1st|/|2nd| derivative
    const std::size_t n = d.xs.size();
    if (n < 3) return 0.5 * (lo + hi);
    std::size_t bi = 1;
    double best = -1.0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
      const double hl = d.xs[i] - d.xs[i - 1], hr = d.xs[i + 1] - d.xs[i];
      if (hl <= 0.0 || hr <= 0.0) continue;
      const double val =
          (order == 1)
              ? (d.ys[i + 1] - d.ys[i - 1]) / (hl + hr)
              : 2.0 *
                    (hl * d.ys[i + 1] - (hl + hr) * d.ys[i] + hr * d.ys[i - 1]) /
                    (hl * hr * (hl + hr));
      if (std::fabs(val) > best) { best = std::fabs(val); bi = i; }
    }
    return d.xs[bi];
  };
  switch (b.placement) {
    case PinMode::Manual: mu_out = b.mu; return true;
    case PinMode::XMin: mu_out = lo; return true;
    case PinMode::XMax:
      if (b.channel != InitBasis::F) {
        Logger::instance().log(LogLevel::WARNING,
            "tadah refit INIT: XMAX skipped for PHI/RHO (left-active, vanish "
            "at the cutoff); only F (embedding) honours XMAX.");
        return false;
      }
      mu_out = hi;
      return true;
    case PinMode::YMin: mu_out = arg_extreme(false); return true;
    case PinMode::YMax: mu_out = arg_extreme(true); return true;
    case PinMode::SMax: mu_out = arg_deriv(1); return true;
    case PinMode::CMax: mu_out = arg_deriv(2); return true;
  }
  return false;
}

// Build the per-channel INIT seed (empty BlipFit if no INIT bases on `ch`).
// zero_ref: the EMBF0 ZERO zero-referenced basis (the F channel only).
BlipFit build_init_seed(const RefitConfig &rc, InitBasis::Channel ch,
                        BasisKind kind, const JointCurveData &d, double lo,
                        double hi, int knot_degree, bool zero_ref) {
  BlipFit f;
  f.kind = kind;
  f.zero_ref = zero_ref;
  const bool os = one_sided(kind);
  const double eta_def = os ? 1.0 : 4.0 / std::max(1e-12, hi - lo);
  bool any_nondefault_deg = false;
  for (const InitBasis &b : rc.init) {
    if (b.channel != ch) continue;
    double mu;
    if (!resolve_init_mu(b, d, lo, hi, mu)) continue;
    f.mu.push_back(mu);
    f.eta.push_back(os ? 1.0 : (b.has_eta ? b.eta : eta_def));
    f.c.push_back(0.0);
    f.pinned.push_back(b.pinned);
    const int deg = (os && knot_degree != 5) ? knot_degree : 5;
    f.deg.push_back(deg);
    if (deg != 5) any_nondefault_deg = true;
  }
  if (!any_nondefault_deg) f.deg.clear();  // empty = all quintic (default)
  return f;
}
}  // namespace

bool HPORefitEngine::compact_fit(const tadah::core::Context &c,
                                 const tadah::models::EAMSetflIO &eam,
                                 tadah::models::EAMRefitConfig &cfg,
                                 std::unique_ptr<tadah::models::EAMRefit> &refit,
                                 SelfTest &st) {
  // ---- reachability first: compact mode NEEDS the energy metric ----
  // The lattice/a0 are already guaranteed recognised by the crystal gate in
  // RefitEngine::run(); a non-positive reachable window would still be unusable
  // for rung selection, so fall back to dense in that (defensive) case.
  EAMRefit probe(eam, cfg);
  probe.analyze_reachability();
  if (probe.r_reach() <= 0.0) {
    Logger::instance().log(LogLevel::WARNING,
        "tadah refit[compact]: reachable pair-distance window could not be "
        "computed (r_reach<=0) — falling back to dense fitting.");
    return false;
  }
  const double rcut = eam.rcut();
  const double a0 = eam.lattice_param();
  const std::string lattice = eam.lattice();
  const double r_reach = probe.r_reach();
  const double rho_reach = probe.rho_reach();
  const double smin = cfg.reach_smin;

  // ---- EMBF0: the embedding reference the written pot declares ----
  // "Fit what you evaluate": under EMBF0 ZERO the pot's DM_REAM (and
  // EAMRefit::F_fit) evaluate the zero-referenced embedding with F(0) = 0, so
  // every ABSOLUTE use of the table's F below — the E-V scan energies, the F
  // curve target, the F ladder metric — uses F_ref = F_tab - F_tab(0), and
  // the F ladder / FUSE fit the zero-referenced basis (CompactConfig /
  // BlipFit zero_ref). FREE: the raw table verbatim (tadah 1.5.0). F' / F''
  // and differences of F are reference-free and stay on the table. The table
  // spline itself is never modified (DM_EAM reads it).
  const bool zero = cfg.embf0_zero;
  const double F0_tab = zero ? eam.F_tab(0.0) : 0.0;
  auto Fref = [&](double x) {
    return zero ? eam.F_tab(x) - F0_tab : eam.F_tab(x);
  };

  // fit windows: same margins as the dense path
  const double rwin_lo =
      std::max({0.8 * r_reach, cfg.r_lo_frac * rcut, eam.dr()});
  // the final EAMRefit joins the setfl continuations exactly at the window
  cfg.r_lo_frac = rwin_lo / rcut;

  // ---- RHOFORM: the density form the written pot declares ----
  // "Fit what you evaluate" again: the rho ladder, FUSE and the handoff fit
  // the density in cfg.rho_form (CompactConfig / BlipFit form): NONNEG =
  // the bump expansion with every amplitude >= 0, MONOTONE = the integrated-
  // bump steps with every amplitude >= 0 (the knot families were refused by
  // RefitEngine). A table the form cannot follow is refused before any
  // fitting, with the dense path's pre-checks on this path's fit window.
  const RhoForm rho_form = cfg.rho_form;
  probe.check_rho_table(rwin_lo);

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
  // REFIT_COMPACT_NMAX: per-function basis ceiling, indexed [phi, rho, F]. One
  // value applies to all three; three values set them per-function.
  int nmax_fn[3] = {40, 40, 40};
  {
    std::vector<std::string> t;
    if (c.contains("REFIT_COMPACT_NMAX") && c.size("REFIT_COMPACT_NMAX") > 0) {
      t.resize(c.size("REFIT_COMPACT_NMAX"));
      c.get("REFIT_COMPACT_NMAX", t);
    }
    auto toi = [](const std::string &s) {
      try { size_t p = 0; int v = std::stoi(s, &p);
            if (p != s.size()) throw 0; return v; }
      catch (...) { throw std::runtime_error(
          "tadah refit: REFIT_COMPACT_NMAX expects integer(s); got '" + s + "'"); }
    };
    if (t.size() == 1) { nmax_fn[0] = nmax_fn[1] = nmax_fn[2] = toi(t[0]); }
    else if (t.size() == 3) { nmax_fn[0] = toi(t[0]); nmax_fn[1] = toi(t[1]); nmax_fn[2] = toi(t[2]); }
    else if (!t.empty())
      throw std::runtime_error(
          "tadah refit: REFIT_COMPACT_NMAX takes 1 value (all functions) or 3 "
          "(phi rho F); got " + std::to_string(t.size()));
    for (int v : nmax_fn)
      if (v < 1) throw std::runtime_error(
          "tadah refit: REFIT_COMPACT_NMAX values must be >= 1");
  }
  // lcfg.n_max = the max across functions — used for the sample-grid clamp and
  // as the CompactConfig default; per-function ceilings are passed to run_ladder.
  lcfg.n_max = std::max({nmax_fn[0], nmax_fn[1], nmax_fn[2]});
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
  // Randomised warm restarts per rung (each one runs the full EXPLORE->REFINE
  // candidate, then the coalescence guard): jitters+1 candidates per rung, the
  // metric-best kept. REFIT_COMPACT_JITTERS 0 = warm start only (cheaper, less
  // exploration). This is WHY the guard's repair fired several times per rung.
  lcfg.jitters = iget(c, "REFIT_COMPACT_JITTERS", 2);
  lcfg.knot_degree = knot3 ? 3 : 5;
  lcfg.knot_mix = knotmix;
  // REFIT_OBJECTIVE: what the whole chain optimises toward. energy (default) =
  // the crystal energy metric; curve = the normalised value+1st+2nd-derivative
  // combined RMSE (energy is then unused end-to-end; FUSE drops its E-V term).
  {
    std::string obj = (c.contains("REFIT_OBJECTIVE") &&
                       c.size("REFIT_OBJECTIVE") > 0)
                          ? c.get<std::string>("REFIT_OBJECTIVE")
                          : std::string("energy");
    for (auto &ch : obj)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (obj == "curve")
      lcfg.objective = RefitObjective::Curve;
    else if (obj == "energy")
      lcfg.objective = RefitObjective::Energy;
    else
      throw std::runtime_error(
          "tadah refit: REFIT_OBJECTIVE must be 'energy' (default) or 'curve'; "
          "got '" + obj + "'");
  }
  const bool curve_obj = (lcfg.objective == RefitObjective::Curve);
  // Coalescence guard (Knot families): soft anti-coalescence penalty in EXPLORE
  // + post-REFINE merge/re-insert of near-coincident knot pairs.
  // REFIT_COMPACT_MERGE 'auto' (default) = OFF in curve mode (it over-constrains
  // the weighted-curve ladder) and ON in energy mode (where it was validated);
  // 'true'/'false' force it. No effect on the Blip families (no knots).
  {
    std::string m = (c.contains("REFIT_COMPACT_MERGE") &&
                     c.size("REFIT_COMPACT_MERGE") > 0)
                        ? c.get<std::string>("REFIT_COMPACT_MERGE")
                        : std::string("auto");
    for (auto &ch : m)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (m == "auto")       lcfg.merge_guard = !curve_obj;
    else if (m == "true")  lcfg.merge_guard = true;
    else if (m == "false") lcfg.merge_guard = false;
    else
      throw std::runtime_error(
          "tadah refit: REFIT_COMPACT_MERGE must be 'true', 'false', or 'auto' "
          "(default: off in curve mode, on in energy mode); got '" + m + "'");
  }
  // Curve mode: weight of each derivative block vs value. Default 0 (pure
  // value) = crystal-faithful (LAMMPS energy depends on the curve VALUES);
  // nonzero trades crystal energy for derivative SHAPE (measured: sharp
  // degradation even at 0.01).
  lcfg.curve_deriv_weight = cget(c, "REFIT_CURVE_DERIV_WEIGHT", 0.0);
  // Pressure-relaxed curve weighting (curve mode, ON by default). Tight fit
  // where the crystal actually probes (ambient: r in [r_amb,rcut], rho in
  // [0,rho_amb]); smooth relaxation toward the compressed/high-pressure extreme
  // down to a floor (REFIT_CURVE_RELAX). r_amb/rho_amb are derived self-
  // contained from the lattice + EAM tables (no E-V scan dependency); the
  // weights flow via JointCurveData.wts into BOTH the VARPRO solve and the
  // curve selection metrics. REFIT_CURVE_WEIGHTING off => uniform (legacy).
  const bool curve_weighting = bget(c, "REFIT_CURVE_WEIGHTING", true);
  const double curve_relax = cget(c, "REFIT_CURVE_RELAX", 0.05);
  if (curve_obj && (curve_relax <= 0.0 || curve_relax > 1.0))
    throw std::runtime_error(
        "tadah refit: REFIT_CURVE_RELAX must be in (0, 1]; got " +
        core::to_string(curve_relax, 6));
  // (curve-only knobs REFIT_CURVE_WEIGHTING/RELAX/DERIV_WEIGHT are read above but
  // only consumed when curve_obj is true; in energy mode they are ignored. No
  // contradiction error here: the blueprint injects their defaults on query, so
  // an energy run could not distinguish a user-set knob from the default.)
  // Per-stage optimiser algorithm + budget + tolerances: defaults live in
  // CompactConfig; the optional inline OPTIMIZER block overrides them.
  // ---- stage pipeline (INIT -> EXPLORE -> REFINE -> FUSE) ----
  // RefitParser reads the INIT/EXPLORE/REFINE/FUSE Context keys (throws on the
  // old top-level OPTIMIZER grammar); apply_stage_optimizers flattens the
  // EXPLORE/REFINE/FUSE optimisers onto the CompactConfig scalar fields.
  const RefitConfig rcfg = RefitParser::parse(c);
  apply_stage_optimizers(rcfg, lcfg);
  if (lcfg.seed_algo == "UNIFORM" && rcfg.init.empty())
    Logger::instance().log(LogLevel::WARNING,
        "tadah refit: EXPLORE rung 0 is UNIFORM (no global search) and INIT is "
        "empty — the global structure search is disabled. Set a global EXPLORE "
        "ALGO (e.g. GN_MLSL_LDS) or provide INIT guesses for robustness.");
  const std::string &joint_algo = lcfg.joint_algo;

  // ---- curve-objective optimiser policy ----
  // Curve mode HONOURS the user-configured optimiser at every stage (EXPLORE
  // inner, REFINE, FUSE) exactly as energy mode does — there is no silent
  // substitution. The per-stage legal-ALGO set is enforced (hard error) at
  // parse time by RefitParser; the actual stack used is logged in the header
  // line below so the run is fully transparent. NOTE: aggressive least-squares
  // (Ceres) at low basis can over-fit the unprobed high-magnitude regions of
  // the curve and degrade the crystal — the pressure-relaxed weighting
  // (REFIT_CURVE_WEIGHTING, on by default) is the mitigation that keeps the
  // crystal faithful while honouring the configured optimiser. In curve mode
  // FUSE is a per-function curve refine (no E-V coupling) using its configured
  // optimiser; ENABLED false / ALGO NONE skips a stage.

  auto sget = [&](const std::string &k, const std::string &def) {
    return c.contains(k) && c.size(k) > 0 ? c.get<std::string>(k) : def;
  };

  // ---- the refit optimisation log (REFIT_LOG; the user inspects this
  //      DURING the fit — every line is flushed) ----
  tadah::mlip::engines::RefitLog rlog;
  rlog.open(sget("REFIT_LOG", "refit_opt.log"), /*append=*/false);
  rlog.raw("# tadah refit optimisation log (compact mode)");
  rlog.raw("# one line per ladder rung. metric = energy gate (drives selection;"
           " <= 1 means within REFIT_COMPACT_SHARE x tolerance). The other three"
           " are curve diagnostics vs the tabulated EAM: rmse = value RMSE,"
           " d1 = 1st-derivative (slope) RMSE, d2 = 2nd-derivative (curvature)"
           " RMSE. Each is shown across the per-rung checkpoints");
  rlog.raw("#   insert -> explore -> refine -> merge   (knot families)");
  rlog.raw("#   insert -> explore -> refine            (blip families; merge is"
           " a no-op)");
  rlog.raw("# insert=add a basis fn + linear solve, explore=EXPLORE.inner local"
           " search, refine=REFINE Ceres refine, merge=coalescence guard"
           " (REFIT_COMPACT_MERGE). 'guard merged X/Y cands' = the guard broke up"
           " coalesced impostor knot pairs (and re-inserted at the residual"
           " peaks) on X of the rung's Y candidates; the guard runs once per"
           " candidate (Y = REFIT_COMPACT_JITTERS+1), which is why the old log"
           " showed several identical merge lines per rung. The bracketed"
           " [explore=.. refine=..] are the winning candidate's stop reasons"
           " (NLopt evals / Ceres iterations). refine shows the SPECIFIC Ceres"
           " cause: ftol|gtol|ptol|maxiter.");
  rlog.raw("# NOTE on EXPLORE=XTOL_REACHED: for the derivative-free simplex"
           " explorers (SBPLX/NELDERMEAD/BOBYQA/PRAXIS) this is the algorithm's"
           " OWN parameter-space convergence (its simplex collapsed) and fires"
           " regardless of XTOL_REL. XTOL_REL only gates an ADDITIONAL optional"
           " outer early-stop; setting XTOL_REL 0 disables that gate but cannot"
           " suppress the intrinsic simplex-collapse XTOL_REACHED. Treat it as"
           " benign convergence (like FTOL_REACHED); the metric is the real"
           " progress signal.");
  if (curve_obj)
    rlog.raw("# REFIT_OBJECTIVE=curve: the 'metric' column is the NORMALISED"
             " curve score (lower=better), NOT the energy metric; the energy"
             " metric is unused and FUSE drops the E-V term (curve mode is"
             " independent of energy mode); the ladder grows until stall/n_max."
             " deriv_weight sets value vs d1/d2 in the objective"
             " (0=pure value=crystal-faithful). EXPLORE inner, REFINE and FUSE"
             " each run the OPTIMISER YOU CONFIGURED (honoured, not overridden;"
             " the actual stack is in the 'optimiser:' line above). The"
             " pressure-relaxed weighting (REFIT_CURVE_WEIGHTING, on by default)"
             " is what keeps an aggressive least-squares optimiser (Ceres) from"
             " over-fitting the unprobed high-magnitude curve regions and"
             " wrecking the crystal. NOTE: Ceres REFINE cost grows steeply with"
             " the basis count n — cap its MAXEVAL / use REFIT_COMPACT_OPT_EVERY"
             " (or the gentle SBPLX-only arm) at high NMAX.");
  rlog.line("input element=" + eam.lattice() + " a0=" +
            core::to_string(a0, 4) + " rcut=" + core::to_string(rcut, 4));
  rlog.line("basis=" + cfg.basis +
            "  pipeline EXPLORE(seed/inner)/REFINE/FUSE = " + lcfg.seed_algo +
            "/" + lcfg.explore_algo + "/" + lcfg.refine_algo + "/" + joint_algo);
  // effective per-stage optimiser settings (defaults, or your OPTIMIZER block)
  auto g = [](double v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%.2g", v);
    return std::string(b);
  };
  rlog.line(
      "optimiser: EXPLORE.seed{algo=" + lcfg.seed_algo + " maxeval=" +
      std::to_string(lcfg.mlsl_maxeval) + "}  EXPLORE.inner{algo=" +
      lcfg.explore_algo + " maxeval=" + std::to_string(lcfg.sbplx_maxeval) +
      " ftol_rel=" + g(lcfg.sbplx_ftol) + " xtol_rel=" + g(lcfg.sbplx_xtol) +
      "}  REFINE{algo=" + lcfg.refine_algo + " maxiter=" +
      std::to_string(lcfg.ceres_max_iter) + " ftol_rel=" + g(lcfg.ceres_ftol) +
      " gtol=" + g(lcfg.ceres_gtol) + "}  FUSE{algo=" + joint_algo +
      " maxiter=" + std::to_string(lcfg.joint_max_iter) + " ftol_rel=" +
      g(lcfg.joint_ftol) + "}");
  rlog.line("  (maxeval = NLopt objective-function evaluations [each call to "
            "the loss], not optimiser iterations; for EXPLORE.seed=MLSL it is "
            "the TOTAL budget across all inner restarts, EXPLORE.inner is the "
            "per-restart budget; maxiter = Ceres solver iterations)");
  rlog.line("schedule: opt_every=" + std::to_string(lcfg.opt_every) +
            " stall=" + std::to_string(lcfg.stall_rungs) + " insert_topk=" +
            std::to_string(lcfg.insert_topk) + " jitters=" +
            std::to_string(lcfg.jitters) + " (=> " +
            std::to_string(lcfg.jitters + 1) + " candidates/rung)" +
            "  merge_guard=" + (lcfg.merge_guard ? "on" : "off") +
            "  objective=" +
            (curve_obj ? "curve(value+d1+d2; deriv_weight=" + g(lcfg.curve_deriv_weight) + ")"
                       : "energy") +
            (lcfg.knot_mix ? "  KnotMix double-win detector ON" : "") +
            "  n_max=" +
            ((nmax_fn[0] == nmax_fn[1] && nmax_fn[1] == nmax_fn[2])
                 ? std::to_string(nmax_fn[0])
                 : std::to_string(nmax_fn[0]) + "/" + std::to_string(nmax_fn[1]) +
                       "/" + std::to_string(nmax_fn[2]) + " (phi/rho/F)"));
  rlog.line("windows: r in [" + core::to_string(rwin_lo, 4) + ", " +
            core::to_string(rcut, 4) + "]  rho in [0, " +
            core::to_string(rho_reach, 4) +
            "]  self-test tolerances ambient(a/a0 0.95-1.10)/compressed(0.75-"
            "0.95) = " +
            core::to_string(tol_amb * 1e3, 3) + "/" +
            core::to_string(tol_cmp * 1e3, 3) + " meV/atom");
  {
    char f0[32];
    std::snprintf(f0, sizeof(f0), "%.10g", F0_tab);
    rlog.line(zero ? "embedding: EMBF0 ZERO — F is fitted in the zero-"
                     "referenced basis g - g(0) (F(0) = 0 exactly) against "
                     "F_tab - F_tab(0); table F(0) = " + std::string(f0) +
                     " eV removed (E/atom relative to the isolated atom; "
                     "forces unchanged)"
                   : "embedding: EMBF0 FREE — raw basis against the raw "
                     "table F (legacy, tadah 1.5.0)");
  }
  rlog.line(
      std::string("density: RHOFORM ") + rhoform_name(rho_form) +
      (rho_form == RhoForm::Signed
           ? " — rho = sum_k c_k g(eta_k (r - mu_k)) with signed amplitudes "
             "(tadah <= 1.5.1)"
       : rho_form == RhoForm::NonNeg
           ? " — rho = sum_k c_k g(eta_k (r - mu_k)) with every c_k >= 0 "
             "(non-negative least squares; Ceres lower bounds in REFINE / "
             "FUSE): rho >= 0"
           : " — rho = sum_k c_k S(eta_k (r - mu_k)), S the integrated bump "
             "(1 inside, 0 beyond mu + w/eta), every c_k >= 0 (non-negative "
             "least squares; Ceres lower bounds in REFINE / FUSE): rho >= 0 "
             "and non-increasing"));

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
      e_orig.push_back(hp + Fref(ri));  // EMBF0 ZERO: vs the isolated atom
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
  // Uniform grid per function across its fit window; the VALUE rmse and the
  // 1st/2nd-derivative RMSEs (and the curve objective) are all evaluated on
  // THIS shared grid (value = spline value, d1/d2 = analytic spline
  // derivatives at the same points). REFIT_COMPACT_SAMPLES tunes it: more =
  // a finer RMSE estimate + finer curve fit but slower (each VARPRO solve is
  // O(N*n)); fewer = faster, coarser. Distinct from the energy E-V scan and
  // the OUTPUT_GRID_SCALE setfl grid. Clamped to comfortably exceed the basis.
  int N = iget(c, "REFIT_COMPACT_SAMPLES", 800);
  const int N_min = std::max(50, 4 * lcfg.n_max);
  if (N < N_min) {
    Logger::instance().log(
        LogLevel::WARNING,
        "tadah refit[compact]: REFIT_COMPACT_SAMPLES=" + std::to_string(N) +
            " is too few for the basis ceiling; raising to " +
            std::to_string(N_min));
    N = N_min;
  }
  rlog.line("curve sample grid: " + std::to_string(N) +
            " points/function (value + d1 + d2 RMSE share this grid)");
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
  linspace_eval(0.0, rho_reach, [&](double x) { return Fref(x); },
                dF.xs, dF.ys);  // EMBF0 ZERO: the zero-referenced target
  // ANALYTIC target derivatives straight from the EAM cubic spline (no finite
  // differencing): used by the curve objective and the d1/d2 log diagnostics.
  for (double r : dphi.xs) { dphi.dys.push_back(eam.dphi_tab(r));
                             dphi.d2ys.push_back(eam.d2phi_tab(r)); }
  for (double r : drho.xs) { drho.dys.push_back(eam.drho_tab(r));
                             drho.d2ys.push_back(eam.d2rho_tab(r)); }
  for (double x : dF.xs)   { dF.dys.push_back(eam.dF_tab(x));
                             dF.d2ys.push_back(eam.d2F_tab(x)); }

  // ---- pressure-relaxed curve weights (curve mode, default ON) ----
  // Ambient reference, SELF-CONTAINED (no E-V scan): the equilibrium lattice at
  // a0. r_amb = nearest-neighbour distance there; rho_amb = per-atom density.
  // phi/rho: weight 1 on [r_amb, rcut] (ambient .. cutoff), smoothstep down to
  // the floor toward rwin_lo (compression = smaller r). F: weight 1 on
  // [0, rho_amb] (zero .. ambient pressure), smoothstep down to the floor
  // toward rho_reach (compression = higher rho). Filled into .wts so BOTH the
  // stacked VARPRO solve and the curve selection metrics honour the profile.
  if (curve_obj && curve_weighting) {
    const std::vector<double> nbr = lat::neighbour_distances(lattice, a0, rcut);
    double r_amb = rcut, rho_amb = 0.0;
    for (double r : nbr) { r_amb = std::min(r_amb, r); rho_amb += eam.rho_tab(r); }
    auto smoothstep = [](double t) {  // C1 ramp: S(0)=0, S(1)=1
      t = std::clamp(t, 0.0, 1.0);
      return t * t * (3.0 - 2.0 * t);
    };
    // tight (1) inside the ambient band; smoothstep down to `curve_relax` toward
    // the compressed edge. w_lo: tight for x>=amb (phi/rho). w_hi: tight for
    // x<=amb (F). Degenerate bands (amb outside the window) => uniform 1.
    auto w_lo = [&](double x, double edge_lo, double amb) {
      if (x >= amb || edge_lo >= amb) return 1.0;
      const double t = (x - edge_lo) / (amb - edge_lo);  // 0 at edge, 1 at amb
      return curve_relax + (1.0 - curve_relax) * smoothstep(t);
    };
    auto w_hi = [&](double x, double amb, double edge_hi) {
      if (x <= amb || edge_hi <= amb) return 1.0;
      const double t = (edge_hi - x) / (edge_hi - amb);  // 1 at amb, 0 at edge
      return curve_relax + (1.0 - curve_relax) * smoothstep(t);
    };
    dphi.wts.resize(N);
    drho.wts.resize(N);
    dF.wts.resize(N);
    for (int i = 0; i < N; ++i) {
      dphi.wts[i] = w_lo(dphi.xs[i], rwin_lo, r_amb);
      drho.wts[i] = w_lo(drho.xs[i], rwin_lo, r_amb);
      dF.wts[i] = w_hi(dF.xs[i], rho_amb, rho_reach);
    }
    rlog.line("curve weighting: ON (pressure-relaxed)  r_amb=" +
              core::to_string(r_amb, 4) + " rho_amb=" +
              core::to_string(rho_amb, 4) + " relax_floor=" +
              core::to_string(curve_relax, 3) +
              "  [phi/rho tight on r>=r_amb; F tight on rho<=rho_amb]");
  } else if (curve_obj) {
    rlog.line("curve weighting: OFF (uniform) — REFIT_CURVE_WEIGHTING off");
  }

  // ---- INIT seeds: resolved anchors + pins per channel (empty => uniform) ----
  const BlipFit init_phi =
      build_init_seed(rcfg, InitBasis::PHI, k_rad, dphi, rwin_lo, rcut,
                      lcfg.knot_degree, /*zero_ref=*/false);
  const BlipFit init_rho =
      build_init_seed(rcfg, InitBasis::RHO, k_rad, drho, rwin_lo, rcut,
                      lcfg.knot_degree, /*zero_ref=*/false);
  const BlipFit init_F = build_init_seed(rcfg, InitBasis::F, k_emb, dF, 0.0,
                                         rho_reach, lcfg.knot_degree, zero);

  // Bump-support gate on the INIT anchors (authoring surface, radial bump
  // channels): an anchor whose support reaches past rcut cannot be honoured
  // — the ladder's clamp would silently move it, so reject it up front with
  // the minimal --eta that fits.
  if (!knotfam) {
    const double w_rad = tadah::models::bump_halfwidth(
        blip5 ? tadah::models::BasisKind::Blip5 : tadah::models::BasisKind::Blip);
    const double tol = tadah::models::bump_rcut_tol(rcut);
    // `step`: the density channel under RHOFORM MONOTONE, whose written
    // SGRIDMB must be > 0 (an eta <= 0 step is not non-increasing), so an
    // anchor with --eta <= 0 is an error there instead of being skipped.
    auto check_init = [&](const BlipFit &f, const char *nm, bool step) {
      for (int k = 0; k < f.n(); ++k) {
        if (f.eta[k] <= 0.0 && step)
          throw std::runtime_error(
              std::string("tadah refit INIT [") + nm + "]: the basis at mu " +
              core::to_string(f.mu[k], 6) + " has eta " +
              core::to_string(f.eta[k], 6) + " <= 0; RHOFORM MONOTONE fits "
              "the density in integrated-bump steps, which need eta > 0 "
              "(SGRIDMB > 0). Set --eta > 0 on the anchor.");
        if (f.eta[k] <= 0.0) continue;
        const double reach = f.mu[k] + w_rad / f.eta[k];
        if (reach <= rcut + tol) continue;
        const std::string hint =
            f.mu[k] < rcut - tol
                ? "Set --eta >= " +
                      core::to_string(w_rad / (rcut - f.mu[k]), 6) +
                      " or move the anchor inward."
                : "Move the anchor below rcut (no width can fit a bump "
                  "centred at/past the cutoff).";
        throw std::runtime_error(
            std::string("tadah refit INIT [") + nm + "]: the basis at mu " +
            core::to_string(f.mu[k], 6) + " (eta " +
            core::to_string(f.eta[k], 6) + ") has support reaching " +
            core::to_string(reach, 6) + ", past rcut = " +
            core::to_string(rcut, 6) + " — the emitted pot is hard-truncated "
            "there (bump-support gate). " + hint);
      }
    };
    check_init(init_phi, "phi", /*step=*/false);
    check_init(init_rho, "rho", /*step=*/rho_form == RhoForm::Monotone);
  }

  // ---- per-function ladders with energy-metric rung selection ----
  // metric: candidate function + ORIGINAL other two, normalised so that 1.0
  // == `share` of the user tolerance in every band (the joint refinement
  // afterwards handles the cross-function coupling).
  auto run_ladder = [&](const JointCurveData &d, double lo, double hi,
                        const CompactBlipFitter::Metric &metric,
                        const char *name, std::vector<LadderRow> &trace,
                        BasisKind kind, const BlipFit &init, int nmax,
                        int max_add, double hard_rcut, bool zero_ref,
                        RhoForm form) {
    CompactConfig fcfg = lcfg;
    fcfg.n_max = nmax;  // this function's REFIT_COMPACT_NMAX ceiling
    // EMBF0 ZERO zero-referenced basis: the F ladder only (d.ys = F_ref)
    fcfg.zero_ref = zero_ref;
    // RHOFORM: the rho ladder only (phi and F pass Signed)
    fcfg.rho_form = form;
    // Bump-support gate (radial bump channels only; 0 = off): the emitted
    // pot is hard-truncated at rcut, so every bump must keep mu + w/eta <=
    // rcut — spilling tails are fitted against no data and `tadah hpo`
    // rejects the pot. The knot families need no gate (left-active knots
    // vanish at their own knot <= rcut) and the F channel has no r cutoff.
    fcfg.hard_rcut = hard_rcut;
    // EXPLORE MAX_ADD caps the basis this stage may add (per function): the
    // ladder grows from the INIT count (or n_min) up to base + MAX_ADD,
    // clamped to this function's NMAX.
    const int base_n = init.n() > 0 ? init.n() : fcfg.n_min;
    if (max_add >= 0)
      fcfg.n_max = std::min(nmax, base_n + max_add);
    if (max_add >= 0 && base_n + max_add < nmax)
      rlog.line(std::string("[") + name + "] NOTE: EXPLORE MAX_ADD=" +
                std::to_string(max_add) + " caps the ladder at n=" +
                std::to_string(base_n + max_add) + " (base " +
                std::to_string(base_n) + "+MAX_ADD) BELOW REFIT_COMPACT_NMAX=" +
                std::to_string(nmax) +
                " — raise MAX_ADD to let the ladder reach NMAX.");
    fcfg.log_fn = [&rlog, name](const std::string &m) {
      rlog.line(std::string("[") + name + "] " + m);
    };
    rlog.line(std::string("---- ") + name + " ladder ----");
    CompactBlipFitter fitter(d.xs, d.ys, d.wts, lo, hi, fcfg, kind, d.dys,
                             d.d2ys);
    BlipFit best = fitter.fit(metric, trace, init.n() > 0 ? &init : nullptr);
    if (curve_obj) {
      // Curve objective: best.metric is the normalised value+d1+d2 score (no
      // tolerance threshold). Report the curve score + value rmse; the ladder
      // grew until stall / n_max.
      Logger::instance().log(
          LogLevel::INFO,
          std::string("tadah refit[compact]: ") + name + " ladder: n=" +
              std::to_string(best.n()) + "  curve_rmse=" + g(best.curve_rmse) +
              "  curve_score(value+d1+d2)=" + core::to_string(best.metric, 4) +
              " (REFIT_OBJECTIVE=curve; grew to stall/n_max)");
      return best;
    }
    // energy_metric <= 1 means this function alone is within its share of the
    // tolerance budget => SUCCESS (log at INFO). > 1 means it fell short and is
    // the binding limit on faithfulness => WARNING with guidance. (Either way
    // the joint refinement below then couples the three functions.)
    const bool within = best.metric <= 1.0;
    Logger::instance().log(
        within ? LogLevel::INFO : LogLevel::WARNING,
        std::string("tadah refit[compact]: ") + name + " ladder: n=" +
            std::to_string(best.n()) + "  curve_rmse=" + g(best.curve_rmse) +
            "  energy_metric=" + core::to_string(best.metric, 4) + " (" +
            (within ? "within"
                    : "ABOVE — raise REFIT_COMPACT_NMAX / EXPLORE MAX_ADD or "
                      "loosen REFIT_TOL_*") +
            " " + core::to_string(share, 2) + "x tolerance)");
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
      "phi", tr_phi, k_rad, init_phi, nmax_fn[0], rcfg.explore.max_add[0],
      knotfam ? 0.0 : rcut, /*zero_ref=*/false, RhoForm::Signed);

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
      "rho", tr_rho, k_rad, init_rho, nmax_fn[1], rcfg.explore.max_add[1],
      knotfam ? 0.0 : rcut, /*zero_ref=*/false, rho_form);

  // F metric: absolute F values -> Fref (EMBF0 ZERO: f is zero-referenced)
  BlipFit fF = run_ladder(
      dF, 0.0, rho_reach,
      [&](const BlipFit &f) {
        double worst = 0.0;
        for (int i = 0; i < M; ++i)
          worst = std::max(worst, std::fabs(f.eval(rho_i[i]) -
                                            Fref(rho_i[i])) *
                                      e_w[i] / share);
        return worst;
      },
      "F", tr_F, k_emb, init_F, nmax_fn[2], rcfg.explore.max_add[2],
      /*hard_rcut=*/0.0, zero, RhoForm::Signed);  // embedding: no r cutoff

  // ---- FUSE stage ----
  std::string joint_term;
  double e_resid_max = 0.0;
  // Curve mode drops the E-V energy coupling; the energy residual is still
  // reported (report-only) so the user sees the energy cost of the curve fit.
  auto report_energy_residual = [&]() {
    double mx = 0.0;
    for (std::size_t i = 0; i < dists.size(); ++i) {
      double hp = 0.0, ri = 0.0;
      for (double r : dists[i]) { hp += 0.5 * fphi.eval(r); ri += frho.eval(r); }
      mx = std::max(mx, std::fabs(e_w[i] * (hp + fF.eval(ri) - e_orig[i])));
    }
    return mx;
  };
  auto report_fuse = [&](const std::string &what) {
    e_resid_max = report_energy_residual();
    rlog.line("joint: max |dE|/tol = " + core::to_string(e_resid_max, 4) +
              " (report-only; energy dropped in curve mode)  params n2b/nrho/nF = " +
              std::to_string(fphi.n()) + "/" + std::to_string(frho.n()) + "/" +
              std::to_string(fF.n()));
    Logger::instance().log(LogLevel::INFO,
        "tadah refit[compact]: FUSE (" + what + ", no E-V): energy max |dE|/tol = " +
        core::to_string(e_resid_max, 4) + " (report-only)  params n2b/nrho/nF = " +
        std::to_string(fphi.n()) + "/" + std::to_string(frho.n()) + "/" +
        std::to_string(fF.n()));
  };
  if (curve_obj) {
    // Curve mode: FUSE is a per-function curve refine (energy E-V coupling
    // dropped — curve mode stays independent of the crystal-energy metric). It
    // HONOURS the configured FUSE optimiser (joint_algo) per function, on the
    // SAME pressure-relaxed weighted curve as the ladder (d.wts). FUSE ALGO NONE
    // (or ENABLED false) skips it.
    joint_term = "curve (no E-V coupling)";
    if (joint_algo == "NONE") {
      rlog.line("---- FUSE (curve: skipped, ALGO NONE / disabled) ----");
    } else {
      rlog.line(std::string("---- FUSE (curve: per-function ") + joint_algo +
                " refine, weighted, no E-V) ----");
      auto fuse_one = [&](BlipFit &fit, const JointCurveData &d, double lo,
                          double hi, BasisKind kind, const char *nm,
                          double hard_rcut, bool zero_ref, RhoForm form) {
        CompactConfig ffcfg = lcfg;
        ffcfg.refine_algo = joint_algo;  // FUSE honours its configured optimiser
        ffcfg.hard_rcut = hard_rcut;     // bump-support gate (radial bumps)
        ffcfg.zero_ref = zero_ref;       // EMBF0 ZERO basis (F only)
        ffcfg.rho_form = form;           // RHOFORM (rho only)
        CompactBlipFitter ff(d.xs, d.ys, d.wts, lo, hi, ffcfg, kind, d.dys,
                             d.d2ys);
        const double before = ff.combined_curve_score(fit);
        ff.refine_with_cfg(fit);
        const int moved = ff.clamp_bump_support(fit);  // + coeff re-solve
        if (moved > 0)
          rlog.line(std::string("FUSE[") + nm + "] bump-support clamp moved " +
                    std::to_string(moved) + " bump(s) back inside rcut.");
        const double after = ff.combined_curve_score(fit);
        rlog.line(std::string("FUSE[") + nm + "] curve score " +
                  core::to_string(before, 4) + " -> " + core::to_string(after, 4));
      };
      fuse_one(fphi, dphi, rwin_lo, rcut, k_rad, "phi", knotfam ? 0.0 : rcut,
               /*zero_ref=*/false, RhoForm::Signed);
      fuse_one(frho, drho, rwin_lo, rcut, k_rad, "rho", knotfam ? 0.0 : rcut,
               /*zero_ref=*/false, rho_form);
      fuse_one(fF, dF, 0.0, rho_reach, k_emb, "F", 0.0, zero,
               RhoForm::Signed);
    }
    report_fuse("curve, weighted curve refine");
  } else {
    // ---- joint refinement: curves + E-V residuals, analytic gradients ----
    rlog.line("---- joint refinement (curves + E-V residuals, " + joint_algo +
              ") ----");
    // EMBF0: joint_refine takes the reference from fF.zero_ref (stamped by
    // the F ladder); dF.ys and e_orig are already on F_ref. RHOFORM: it
    // takes the density form from frho.form (stamped by the rho ladder);
    // under NONNEG / MONOTONE it bounds the amplitudes at 0 and the bump
    // widths at the ladder's floor.
    auto run_joint = [&]() {
      return joint_refine(fphi, frho, fF, dphi, drho, dF, dists, e_orig, e_w,
                          /*curve_weight=*/1.0, lcfg.joint_max_iter,
                          lcfg.joint_ftol, joint_algo, &joint_term,
                          knotfam ? 0.0 : rcut, knotfam ? 0.0 : rcut);
    };
    e_resid_max = run_joint();
    // Bump-support gate: joint_refine optimises the radial bumps in their
    // outer support edges bounded at rcut, so this projection (the clamp
    // core shared with the ladder's clamp_bump_support) normally moves
    // nothing; it still guards what the solve does not bound (a pinned INIT
    // anchor, FUSE ALGO NONE). A moved channel gets a linear re-solve in its
    // form: rho under NONNEG / MONOTONE through non-negative least squares.
    if (!knotfam) {
      const BlipFit phi_joint = fphi, rho_joint = frho;  // the joint solution
      bool moved_any = false;
      auto reclamp = [&](BlipFit &fit, const JointCurveData &d, double lo,
                         double hi, const char *nm) {
        const int moved = clamp_bump_support_params(fit, rcut, hi - lo);
        if (moved == 0) return;
        moved_any = true;
        CompactConfig lincfg = lcfg;
        lincfg.rho_form = fit.form;  // Signed for phi, the form for rho
        CompactBlipFitter lin(d.xs, d.ys, d.wts, lo, hi, lincfg, fit.kind,
                              d.dys, d.d2ys);
        lin.solve_coeffs(fit);
        rlog.line(std::string("joint[") + nm + "]: bump-support clamp moved " +
                  std::to_string(moved) +
                  " bump(s) back inside rcut (+ linear re-solve).");
      };
      reclamp(fphi, dphi, rwin_lo, rcut, "phi");
      reclamp(frho, drho, rwin_lo, rcut, "rho");
      if (moved_any) e_resid_max = report_energy_residual();

      // The linear re-solve refits a moved channel's coefficients to its
      // curve alone, which can drop the E-V coupling the joint solve
      // converged to (before the edge bound: Ravelo Ta setfl, NONNEG, seed
      // 12345, one phi bump moved after a cost-0.0098 joint solve, max
      // |dE|/tol went to 426.9 and the self-test failed at 42 / 88
      // meV/atom). When it leaves the E-V residuals above tolerance, the
      // joint refinement is re-run from the joint solution with the moved
      // bumps held: each pass projects both radial channels, holds every
      // moved bump's (mu, eta) through the INIT pin mask (restored
      // afterwards) and refits every coefficient and every other bump against
      // curves + E-V (under NONNEG / MONOTONE the rho amplitudes keep their
      // >= 0 bounds); a bump a re-run pushes out is held in the next pass.
      // Every pass holds at least one more bump, so the passes end, within
      // the radial bump count, when nothing spills any more; a bump that
      // still spills after that gets the linear re-solve. The result with the
      // smaller max |dE|/tol is kept. Within tolerance the re-solve stands (it
      // follows the curves more closely). Every density form.
      if (moved_any && e_resid_max > 1.0) {
        const BlipFit phi_lin = fphi, rho_lin = frho, F_lin = fF;
        const double e_lin = e_resid_max;
        const std::string term_lin = joint_term;
        rlog.line("joint: the linear re-solve leaves max |dE|/tol = " +
                  core::to_string(e_lin, 4) + " (RHOFORM " +
                  rhoform_name(rho_form) + "): joint refinement re-run from "
                  "the joint solution with the moved bumps held");
        fphi = phi_joint;
        frho = rho_joint;
        // each pass holds >= 1 more bump: the loop ends once nothing spills
        const int held_passes = fphi.n() + frho.n();
        auto hold_clamped = [&](BlipFit &fit) {
          const std::vector<double> mu0 = fit.mu, eta0 = fit.eta;
          const int moved =
              clamp_bump_support_params(fit, rcut, rcut - rwin_lo);
          if (moved == 0) return 0;
          fit.pinned.resize(fit.mu.size(), false);
          for (int k = 0; k < fit.n(); ++k)
            if (fit.mu[k] != mu0[k] || fit.eta[k] != eta0[k])
              fit.pinned[k] = true;
          return moved;
        };
        for (int pass = 1; pass <= held_passes; ++pass) {
          const int mphi = hold_clamped(fphi), mrho = hold_clamped(frho);
          if (mphi + mrho == 0) break;
          e_resid_max = run_joint();
          rlog.line("joint: pass " + std::to_string(pass) + "/" +
                    std::to_string(held_passes) + " held " +
                    std::to_string(mphi) + " phi / " + std::to_string(mrho) +
                    " rho bump(s) moved back inside rcut: max |dE|/tol = " +
                    core::to_string(e_resid_max, 4) + "  term=" + joint_term);
        }
        fphi.pinned = phi_joint.pinned;
        frho.pinned = rho_joint.pinned;
        moved_any = false;
        reclamp(fphi, dphi, rwin_lo, rcut, "phi");
        reclamp(frho, drho, rwin_lo, rcut, "rho");
        if (moved_any) e_resid_max = report_energy_residual();
        const bool held = e_resid_max < e_lin;
        rlog.line(std::string("joint: kept the ") +
                  (held ? "held re-run" : "linear re-solve") +
                  " (max |dE|/tol " + core::to_string(e_lin, 4) +
                  " linear re-solve, " + core::to_string(e_resid_max, 4) +
                  " held re-run)");
        if (!held) {
          fphi = phi_lin;
          frho = rho_lin;
          fF = F_lin;
          e_resid_max = e_lin;
          joint_term = term_lin;
        }
      }
    }
    rlog.line("joint: max |dE|/tol = " + core::to_string(e_resid_max, 4) +
              "  params n2b/nrho/nF = " + std::to_string(fphi.n()) + "/" +
              std::to_string(frho.n()) + "/" + std::to_string(fF.n()) +
              "  term=" + joint_term);
    // max |dE|/tol <= 1 => the coupled E-V residuals are within tolerance
    // (SUCCESS, INFO); > 1 => still above tolerance (WARNING). The crystal
    // self-test below is the final faithfulness judge.
    Logger::instance().log(
        e_resid_max <= 1.0 ? LogLevel::INFO : LogLevel::WARNING,
        "tadah refit[compact]: FUSE (joint curves + E-V residuals): "
        "max |dE|/tol = " + core::to_string(e_resid_max, 4) +
        (e_resid_max <= 1.0 ? " (within tolerance)" : " (ABOVE tolerance)") +
        "  params n2b/nrho/nF = " + std::to_string(fphi.n()) + "/" +
        std::to_string(frho.n()) + "/" + std::to_string(fF.n()) +
        " (basis " +
        std::string(knotmix ? "KnotMix" : knot3 ? "Knot3" : knot5 ? "Knot5"
                    : blip5 ? "Blip5" : "Blip") +
        "; pipeline EXPLORE(seed/inner)/REFINE/FUSE = " + lcfg.seed_algo + "/" +
        lcfg.explore_algo + "/" + lcfg.refine_algo + "/" + joint_algo +
        "; control: REFIT_BASIS + INIT/EXPLORE/REFINE/FUSE blocks)");
  }

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

  // ---- basis-info file: per-basis positions for the inspection plots ----
  // (mu/eta/coeff/degree + the INIT pin flag). The plotting script overlays
  // these on every function and derivative subplot.
  {
    std::ofstream os("refit_basis_info.dat");
    if (os) {
      os << "# tadah refit basis info (compact mode)\n"
         << "# function  index  mu  eta  coeff  degree  pinned\n";
      os << std::scientific << std::setprecision(10);
      auto dump = [&](const char *nm, const BlipFit &f) {
        for (int k = 0; k < f.n(); ++k)
          os << nm << " " << k << " " << f.mu[k] << " " << f.eta[k] << " "
             << f.c[k] << " " << f.degree(k) << " " << (f.is_pinned(k) ? 1 : 0)
             << "\n";
      };
      dump("phi", fphi);
      dump("rho", frho);
      dump("F", fF);
    }
  }

  // ---- assemble the EAMRefit (injection) + self-test ----
  // EMBF0 handoff: fF holds the RAW parameters (c, mu, eta) of the embedding
  // it was fitted as (zero-referenced iff fF.zero_ref). EAMRefit (built from
  // the same cfg) evaluates F_fit zero-referenced iff cfg.embf0_zero,
  // subtracting g(0) itself from the injected raw parameters, and writes the
  // matching EMBF0 line — so the two must agree, or the pot would evaluate a
  // different F than was fitted.
  if (fF.zero_ref != zero)
    throw std::logic_error(
        "tadah refit[compact]: the fitted embedding's zero reference does not "
        "match EMBF0 " + std::string(zero ? "ZERO" : "FREE") +
        " (internal error)");
  // RHOFORM handoff: frho holds the amplitudes of the terms it was fitted in
  // (g, or the steps S under MONOTONE). EAMRefit (built from the same cfg)
  // evaluates rho_fit and writes RHOFORM in cfg.rho_form, so the two must
  // agree, and a positive form's density must pass the checks DM_REAM
  // applies when the pot is loaded: ReamDensity::validate on the grids the
  // pot gets (every amplitude finite and >= 0, every centre and width
  // finite, under MONOTONE every width > 0 and the cutoff whitelisted). A
  // step with eta <= 0 rises; the self-test evaluates rho_fit and cannot
  // see that, and RefitEngine writes pot.tadah without loading it. The
  // ladder's non-negative solves, the Ceres bounds, the joint width floor
  // and the INIT width check guarantee all of it.
  if (frho.form != cfg.rho_form)
    throw std::logic_error(
        std::string("tadah refit[compact]: the fitted density's form ") +
        rhoform_name(frho.form) + " does not match RHOFORM " +
        rhoform_name(cfg.rho_form) + " (internal error)");
  if (rhoform_is_positive(cfg.rho_form)) {
    tadah::models::ReamDensity dens;
    dens.form = cfg.rho_form;
    dens.basis = knotfam ? tadah::models::BasisKind::Knot5
                 : blip5 ? tadah::models::BasisKind::Blip5
                         : tadah::models::BasisKind::Blip;
    dens.crho = frho.mu;
    if (!knotfam) dens.srho = frho.eta;   // the knot width is a gauge
    dens.arho = frho.c;
    dens.rho_deg.assign(frho.deg.begin(), frho.deg.end());
    try {
      dens.validate(cfg.rctype);
    } catch (const tadah::models::DensityFormError &e) {
      throw std::logic_error(std::string("tadah refit[compact]: ") + e.what() +
                             " (internal error)");
    }
  }
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
    rlog.raw(std::string(zero ? "    cohesive (relative to the isolated "
                                "atom): original="
                              : "    cohesive: original=") +
             core::to_string(st.e_orig, 6) +
             " refit=" + core::to_string(st.e_fit, 6) + " delta=" +
             core::to_string(1000.0 * (st.e_fit - st.e_orig), 4) +
             " meV/atom");
  }
  rlog.line("outputs: pot.tadah (native), refit.eam (round-trip setfl), "
            "refit_curve_{phi,rho,F}.dat (value/d1/d2 inspection), "
            "refit_basis_info.dat (knot/blip positions), "
            "refit_ladder_trace.dat, hpotarget");
  return true;
}

#endif  // TADAH_USE_CERES

}  // namespace engines
}  // namespace hpo
}  // namespace tadah

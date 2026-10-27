#include <tadah/hpo/rho_gates.h>

#include <tadah/core/number_format.h>
#include <tadah/core/registry.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/candidate_context.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/strict_parse.h>   // config_line_error
#include <tadah/models/cutoffs.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/dm/ream_density.h>
#include <tadah/models/descriptors/dm/embed_kernel.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The gate arithmetic (the verdicts and v) lives in this translation unit,
// which tadah.hpo compiles without -ffast-math, so the non-finite checks
// stay; the density values themselves come from models::ReamDensity.

namespace tadah {
namespace hpo {
namespace detail {

namespace {

struct CandidateDensity {
  models::ReamDensity legacy;
  std::unique_ptr<models::EmbedEvaluator> component;
  models::RhoForm form=models::RhoForm::Signed;
  models::DensityScan scan_grid(double lo,double hi,std::size_t n,models::Cut_Base &fc,
      std::vector<double> *values=nullptr,std::vector<double> *terms=nullptr) const {
    if(!component)return legacy.scan_grid(lo,hi,n,fc,values,terms);
    auto grid=models::ReamDensity::linspace(lo,hi,n);
    std::vector<double> rho(n),absolute(n);
    models::DensityScan result;
    for(std::size_t i=0;i<n;++i) {
      rho[i]=component->density(grid[i]).value;
      const auto &c=component->spec().density;
      for(std::size_t j=0;j<c.amplitudes.size();++j)
        absolute[i]+=std::abs(c.amplitudes[j]*component->density(grid[i],'a',j).value);
      if(!std::isfinite(rho[i]))result.finite=false;
      if(i==0 || rho[i]<result.rho_min){result.rho_min=rho[i];result.i_min=i;}
      if(i==0 || rho[i]>result.rho_max){result.rho_max=rho[i];result.i_max=i;}
    }
    result.r_at_min=grid[result.i_min];result.r_at_max=grid[result.i_max];
    if(values)*values=std::move(rho);if(terms)*terms=std::move(absolute);return result;
  }
  models::RhoBarScan rhobar_scan(const std::vector<models::eam_lattice::Shell> &shells,
      const std::vector<double> &a,models::Cut_Base &fc) const {
    if(!component)return legacy.rhobar_scan(shells,a,fc);
    models::RhoBarScan result;result.rhobar.assign(a.size(),0);
    for(std::size_t i=0;i<a.size();++i) {
      for(const auto &sh:shells) result.rhobar[i]+=sh.m*component->density(sh.d*a[i]).value;
      if(!std::isfinite(result.rhobar[i]))result.finite=false;
      if(i && result.rhobar[i]>=result.rhobar[i-1]){if(result.rises==0)result.first_rise=i-1;++result.rises;}
    }
    return result;
  }
};

/* A genuine violation measure is capped here; a saturated one is DBL_MAX,
 * which fail_sigma maps to the top of band D, above every genuine value. */
constexpr double kVCap = 1e8;
constexpr double kVSat = std::numeric_limits<double>::max();

std::string fmt(const char *f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  return b;
}
std::string e4(double v) { return fmt("%.4e", v); }
std::string f4(double v) { return fmt("%.4f", v); }
std::string gnum(double v) { return fmt("%.10g", v); }
std::string vtext(double v) { return v == kVSat ? std::string("saturated") : e4(v); }

/* The number a stored token spells (the lossless context text): the
 * decimal grammar of the files and of the model's readers
 * (core::parse_decimal_finite; D_Base::watom_value for WATOM). nullopt when
 * the key is absent; throws when the token is outside the grammar (strtod
 * read "0x1.6p2" as 5.5, where the model reads a different number). */
std::optional<double> stored_double(const core::Context &c, const char *key,
                                    std::size_t n = 0) {
  const auto tok = c.stored_value(key, n);
  if (!tok) return std::nullopt;
  double v = 0.0;
  if (!core::parse_decimal_finite(*tok, v))
    throw std::runtime_error(std::string(key) + " ('" + *tok +
                             "') is not a decimal number");
  return v;
}

/* The many-body cutoff of a candidate as DC_Selector builds it,
 * factory<Cut_Base, double&>(RCTYPEMB, RCUTMBMAX), from stored values; a
 * fresh object per call (no shared mutable instance). Throws when a key is
 * absent or not a number, or the class unknown. */
std::unique_ptr<models::Cut_Base> candidate_cutoff(const core::Context &c,
                                                   double &r_hi) {
  const auto specs=models::embed_specs(c);
  if(!specs.empty()) {
    if(specs.size()!=1)throw std::runtime_error("Density gates require one scalar component");
    r_hi=specs[0].radius;
    return std::make_unique<models::Cut_Dummy>(r_hi);
  }
  const auto label = c.stored_value("RCTYPEMB");
  const auto rc = stored_double(c, "RCUTMBMAX");
  if (!label || !rc)
    throw std::runtime_error("RCTYPEMB or RCUTMBMAX is not set");
  r_hi = *rc;
  std::unique_ptr<models::Cut_Base> fc(
      core::factory<models::Cut_Base, double &>(*label, r_hi));
  if (!fc) throw std::runtime_error("unknown many-body cutoff " + *label);
  return fc;
}

/* The candidate's density as DM_REAM builds and validates it. Throws. */
CandidateDensity candidate_density(const core::Context &c,
                                      models::Cut_Base *fc) {
  CandidateDensity d;
  auto specs=models::embed_specs(c);
  if(!specs.empty()) {
    if(specs.size()!=1)throw std::runtime_error("Density gates require one scalar component");
    d.component=std::make_unique<models::EmbedEvaluator>(specs[0]);d.form=specs[0].form;
  } else {d.legacy=models::ReamDensity::from_context(c);d.legacy.validate(fc);d.form=d.legacy.form;}
  return d;
}

/* "" when every density pair weight of @p c is finite and > 0, else what is
 * wrong (the saturation reason). */
std::string weight_problem(const core::Context &c) {
  std::vector<double> w;
  try {
    w = models::D_Base::density_pair_weights(c);
  } catch (const std::exception &e) {
    return e.what();
  }
  for (const double x : w)
    if (!(std::isfinite(x) && x > 0.0))
      return "a density pair weight W = " + gnum(x) +
             " (WATOM) is not finite and > 0";
  return "";
}

template <class R>
R saturate(R r, const std::string &why) {
  r.fail = true;
  r.saturated = true;
  r.v = kVSat;
  r.reason = why;
  return r;
}

/* RHOGATE's rule on a scan (r carries the window): the verdict boolean
 * decides, v only on FAIL. */
RhoGateResult rho_verdict(RhoGateResult r, const models::DensityScan &sc,
                          const std::vector<double> &rho, double tol) {
  r.rho_min = sc.rho_min;
  r.r_min = sc.r_at_min;
  r.rho_max = sc.rho_max;
  r.r_max = sc.r_at_max;
  if (!sc.finite) return saturate(r, "a non-finite rho value on the window");
  if (!(sc.rho_max > 0.0)) return saturate(r, "rho_max <= 0 on the window");
  const double thr = tol * sc.rho_max;
  if (!(sc.rho_min < -thr)) return r;
  r.fail = true;
  const double v_peak = std::max(0.0, (-sc.rho_min - thr) / sc.rho_max);
  const std::size_t n = rho.size();
  const double step = (r.r_hi - r.r_lo) / static_cast<double>(n - 1);
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double h = -rho[i] - thr;
    if (h > 0.0) s += h * (i == 0 || i + 1 == n ? 0.5 * step : step);
  }
  const double v_int = s / (sc.rho_max * (r.r_hi - r.r_lo));
  r.v = std::min(kVCap, v_peak + v_int);
  return r;
}

/* RHOBARGATE's rule on a lattice scan. */
RhoBarGateResult rhobar_verdict(RhoBarGateResult r, const models::RhoBarScan &rb,
                                const RhoBarGateTable &t) {
  if (!rb.finite) return saturate(r, "a non-finite rhobar value");
  double max_abs = 0.0;
  for (const double x : rb.rhobar) max_abs = std::max(max_abs, std::fabs(x));
  if (!(max_abs > 0.0))
    return saturate(r, "rhobar(a) = 0 on the whole a-grid");
  if (rb.rises == 0) return r;
  r.fail = true;
  r.rises = rb.rises;
  r.first_rise = t.f[rb.first_rise];
  double s = 0.0;
  for (std::size_t k = 0; k + 1 < rb.rhobar.size(); ++k)
    s += std::max(0.0, rb.rhobar[k + 1] - rb.rhobar[k]);
  r.v = std::min(kVCap, s / max_abs);
  return r;
}

/* The OPTIM entry of @p key (OPTIM keys are upper-cased at parse), or
 * nullptr. */
using Opti = decltype(HPOTargets::optimap)::mapped_type;
const Opti *find_optim(const HPOTargets &trg, const std::string &key) {
  const auto it = trg.optimap.find(key);
  return it == trg.optimap.end() ? nullptr : &it->second;
}

/* What keeps a density pair weight from being finite and > 0 on EVERY
 * candidate, "" when some point of the box has them all positive. A pair
 * weight is w_i, sqrt(w_i w_j) or a pair triple's value, so it can be
 * positive iff it is with every OPTIM'd WATOM token at its high bound: the
 * weights of the CONFIG point moved there decide. */
std::string fixed_weight_problem(const core::Context &train_ctx,
                                 const HPOTargets &trg, const DensityGates &g) {
  std::vector<double> p = config_point_from_targets(trg);
  for (const ParamRef &r : make_param_ref_table(trg))
    if (r.theta_name == "WATOM")
      p[r.flat_idx] = trg.optimap.at(r.theta_name).highbound[r.token];
  return weight_problem(seed_context(train_ctx, trg, p, g));
}

/* The labels of the OPTIM dimensions that arm the gates
 * (HPOTargets::density_gate_keys; the keys are upper-case). */
std::vector<std::string> arming_dims(const HPOTargets &trg) {
  std::vector<std::string> dims;
  const std::vector<std::string> &labels = trg.spec.parameter_labels;
  const auto &keys = HPOTargets::density_gate_keys();
  for (const ParamRef &r : make_param_ref_table(trg))
    if (std::find(keys.begin(), keys.end(), r.theta_name) != keys.end() || r.theta_name.find(".DENSITY.")!=std::string::npos)
      dims.push_back(r.flat_idx < labels.size()
                         ? labels[r.flat_idx]
                         : r.theta_name + "[" + std::to_string(r.arg_pos) + "]");
  return dims;
}

std::string join(const std::vector<std::string> &v, const std::string &sep) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? sep : "") + v[i];
  return s;
}

/* Where a gate's window / grid is described (state line, seed messages). */
std::string rho_window_text(const RhoGateResult &r) {
  return "[" + f4(r.r_lo) + ", " + f4(r.r_hi) + "] A (" + std::to_string(r.npts) +
         " points)";
}
std::string rhobar_grid_text(const RhoBarGateSpec &s) {
  return s.lattice + " (a0 " + gnum(s.a0) + " A, a/a0 [" + gnum(s.f_lo) + ", " +
         gnum(s.f_hi) + "], " + std::to_string(s.npts) + " points)";
}

/* The body of a failing RHOGATE seed / -c message. */
std::string rho_fail_detail(const RhoGateResult &r, const RhoGateSpec &spec) {
  if (r.saturated) return r.reason + " on " + rho_window_text(r);
  return "rho_min " + e4(r.rho_min) + " at r " + f4(r.r_min) + " A, rho_max " +
         e4(r.rho_max) + ", rel " + e4(r.rho_min / r.rho_max) + " < -tol = " +
         e4(-spec.tol) + " on " + rho_window_text(r) + ", v = " + vtext(r.v);
}
std::string rhobar_fail_detail(const RhoBarGateResult &r, const RhoBarGateSpec &spec) {
  if (r.saturated) return r.reason + " on " + rhobar_grid_text(spec);
  return "rhobar rises at " + std::to_string(r.rises) + " of " +
         std::to_string(spec.npts - 1) + " steps on " + rhobar_grid_text(spec) +
         ", first at a/a0 = " + f4(r.first_rise) + ", v = " + vtext(r.v);
}

}  // namespace

bool context_is_bare_ream(const core::Context &ctx) {
  if(!ctx.blocks("DESCRIPTOR").empty()) return models::embed_contexts(ctx).size()==1;
  const auto label = ctx.stored_value("TYPEMB", 0);
  return label && *label == "DM_REAM" && models::context_has_ream(ctx);
}

ReachableRcut knot_aware_reachable_rcut(const core::Context &ctx,
                                        const HPOTargets &trg) {
  // A copy: get_grid / get below may materialise defaults.
  core::Context c = ctx;
  ReachableRcut out;
  auto specs=models::embed_specs(c);
  if(specs.size()==1) {
    out.min_reachable=out.max_reachable=specs[0].radius;
    const auto *o=find_optim(trg,core::to_upper(specs[0].id+".density.cutoff.radius"));
    if(o && !o->opt.empty() && o->opt[0]) {out.min_reachable=o->lowbound[0];out.max_reachable=o->highbound[0];}
    return out;
  }
  if (!models::potedit::derived_knot_rcut(c, "MB")) {
    // a bump channel: the BUMPGATE's range (-inf without RCUTMB)
    const BindingRcut b = derive_binding_rcut(c, trg, "MB");
    out.min_reachable = b.min_reachable;
    out.max_reachable = b.max_reachable;
    return out;
  }
  out.knot = true;
  out.min_reachable = out.max_reachable =
      -std::numeric_limits<double>::infinity();
  core::v_type knots;
  models::D_Base::get_grid(c, "CGRIDMB", knots);
  const Opti *o = find_optim(trg, "CGRIDMB");
  for (std::size_t i = 0; i < knots.size(); ++i) {
    const bool opt = o && i < o->opt.size() && o->opt[i];
    out.min_reachable = std::max(out.min_reachable, opt ? o->lowbound[i] : knots[i]);
    out.max_reachable = std::max(out.max_reachable, opt ? o->highbound[i] : knots[i]);
  }
  return out;
}

DensityGates build_density_gates(const core::Context &master,
                                 const HPOTargets &trg) {
  DensityGates g;
  const bool gated = trg.RHOGATE.has_value() || trg.RHOBARGATE.has_value();
  if (gated && !context_is_bare_ream(master))
    throw std::runtime_error(
        "RHOGATE / RHOBARGATE need a bare DM_REAM many-body term (INITMB "
        "true, TYPEMB DM_REAM)");
  g.knot2b = has_knot_channel(master, "2B");
  g.knotmb = has_knot_channel(master, "MB");
  g.reach = knot_aware_reachable_rcut(master, trg);
  if (!gated) return g;

  /* HPOTargets refuses a gate that no OPTIM dimension arms (it cannot
   * act), at the gate's line. */
  const std::vector<std::string> armed_on = arming_dims(trg);
  if (armed_on.empty())
    throw std::logic_error(
        "RHOGATE / RHOBARGATE without an OPTIM dimension that arms them "
        "(HPOTargets refuses such a gate)");
  if (trg.RHOGATE) {
    RhoGateTable t;
    t.spec = *trg.RHOGATE;
    t.armed_on = armed_on;
    g.rho = std::move(t);
  }
  if (trg.RHOBARGATE) {
    RhoBarGateTable t;
    t.spec = *trg.RHOBARGATE;
    const double dmax =
        g.reach.max_reachable / (t.spec.f_lo * t.spec.a0) * (1.0 + 1e-12);
    if (!std::isfinite(dmax))
      throw std::runtime_error(
          "RHOBARGATE: the largest reachable RCUTMB is not finite");
    if (dmax > kRhoBarMaxShellRadius)
      throw std::runtime_error(
          "RHOBARGATE: f_lo = " + gnum(t.spec.f_lo) + " puts the largest "
          "reachable RCUTMB, " + f4(g.reach.max_reachable) + " A, at " +
          gnum(g.reach.max_reachable / (t.spec.f_lo * t.spec.a0)) +
          " lattice constants of the smallest lattice (a = f_lo a0 = " +
          gnum(t.spec.f_lo * t.spec.a0) + " A); the limit is " +
          gnum(kRhoBarMaxShellRadius) + ". f_lo and f_hi are fractions of "
          "a0 (0.75 scans down to 25 % compression), and a larger f_lo "
          "puts that RCUTMB at fewer lattice constants.");
    t.shells = models::eam_lattice::shells(t.spec.lattice, dmax);
    t.f = models::ReamDensity::linspace(t.spec.f_lo, t.spec.f_hi, t.spec.npts);
    t.a.reserve(t.f.size());
    for (const double x : t.f) t.a.push_back(t.spec.a0 * x);
    t.armed_on = armed_on;
    g.rhobar = std::move(t);
  }
  return g;
}

RhoGateResult eval_rho_gate(const RhoGateTable &t, const core::Context &cand,
                            bool check_weights) {
  RhoGateResult r;
  r.npts = t.spec.npts;
  std::unique_ptr<models::Cut_Base> fc;
  CandidateDensity d;
  try {
    fc = candidate_cutoff(cand, r.r_hi);
    d = candidate_density(cand, fc.get());
  } catch (const std::exception &e) {
    r.abstain = true;
    r.reason = e.what();
    return r;
  }
  r.r_lo = t.spec.r_lo ? *t.spec.r_lo : 0.15 * r.r_hi;
  if (check_weights) {
    const std::string w = weight_problem(cand);
    if (!w.empty()) return saturate(r, w);
  }
  if (!(r.r_lo < r.r_hi))
    return saturate(r, "the window [r_lo, RCUTMB] is empty");
  std::vector<double> rho;
  const models::DensityScan sc = d.scan_grid(r.r_lo, r.r_hi, t.spec.npts, *fc, &rho);
  return rho_verdict(r, sc, rho, t.spec.tol);
}

RhoBarGateResult eval_rhobar_gate(const RhoBarGateTable &t,
                                  const core::Context &cand,
                                  bool check_weights) {
  RhoBarGateResult r;
  std::unique_ptr<models::Cut_Base> fc;
  CandidateDensity d;
  try {
    fc = candidate_cutoff(cand, r.r_hi);
    d = candidate_density(cand, fc.get());
  } catch (const std::exception &e) {
    r.abstain = true;
    r.reason = e.what();
    return r;
  }
  if (check_weights) {
    const std::string w = weight_problem(cand);
    if (!w.empty()) return saturate(r, w);
  }
  return rhobar_verdict(r, d.rhobar_scan(t.shells, t.a, *fc), t);
}

DensityGateVerdict check_density_gates(const DensityGates &g,
                                       const core::Context &cand) {
  DensityGateVerdict out;
  if (g.rho) out.rho = eval_rho_gate(*g.rho, cand);
  if (g.rhobar) out.rhobar = eval_rhobar_gate(*g.rhobar, cand);
  return out;
}

std::optional<GateReject> combine_gate_verdicts(double v_bump,
                                                const std::string &bump_what,
                                                const DensityGates &g,
                                                const DensityGateVerdict &d) {
  struct Failing {
    GateKind    kind;
    double      v;
    std::string what;
    const char *name;
  };
  std::vector<Failing> fails;
  if (v_bump > 0.0)
    fails.push_back({GateKind::Bump, v_bump, bump_what, "bump-support gate"});
  if (g.rho && d.rho && d.rho->fail)
    fails.push_back({GateKind::RhoSign, d.rho->v,
                     describe_rho_reject(*d.rho, g.rho->spec), "rho-sign gate"});
  if (g.rhobar && d.rhobar && d.rhobar->fail)
    fails.push_back({GateKind::RhoBar, d.rhobar->v,
                     describe_rhobar_reject(*d.rhobar, g.rhobar->spec),
                     "lattice-rhobar gate"});
  if (fails.empty()) return std::nullopt;

  GateReject out{fails.front().kind, fails.front().v, fails.front().what};
  if (fails.size() > 1) {
    double sum = 0.0;
    std::vector<std::string> others;
    for (std::size_t i = 0; i < fails.size(); ++i) {
      sum += fails[i].v;   // DBL_MAX + DBL_MAX = inf: clamped below
      if (i) others.push_back(std::string(fails[i].name) + " (v = " +
                              vtext(fails[i].v) + ")");
    }
    out.v = std::min(kVSat, sum);
    out.what += "; also failing: " + join(others, ", ") +
                "; graded by the summed v = " + vtext(out.v);
  }
  return out;
}

std::string describe_rho_reject(const RhoGateResult &r, const RhoGateSpec &spec) {
  if (r.saturated)
    return "rho-sign gate reject (" + r.reason + ", v = saturated)";
  return "rho-sign gate reject (rho_min/rho_max = " + e4(r.rho_min / r.rho_max) +
         " at r = " + f4(r.r_min) + " A, tol " + e4(spec.tol) + ", v = " +
         e4(r.v) + ")";
}

std::string describe_rhobar_reject(const RhoBarGateResult &r,
                                   const RhoBarGateSpec &spec) {
  if (r.saturated)
    return "lattice-rhobar gate reject (" + r.reason + " on " + spec.lattice +
           ", v = saturated)";
  return "lattice-rhobar gate reject (" + std::to_string(r.rises) +
         " rises over a/a0 [" + gnum(spec.f_lo) + ", " + gnum(spec.f_hi) +
         "] on " + spec.lattice + ", first at a/a0 = " + f4(r.first_rise) +
         ", v = " + e4(r.v) + ")";
}

std::string GateSeed::label() const {
  std::string init = origins.empty() ? "" : "INIT " + join(origins, ", ");
  if (!restart_origins.empty())
    init += (init.empty() ? "" : "; ") + std::string("RESTART ") +
            join(restart_origins, ", ");
  if (!c_pot) {
    // A WARM seed is evaluated with its OPTIM dimensions over the -c
    // potential (seed_context), not as the file's own density.
    for (const auto *list : {&origins, &restart_origins})
      for (const auto &o : *list)
        if (o.rfind("WARM", 0) == 0)
          return init + " (its OPTIM dimensions over the -c potential)";
    return init;
  }
  return init.empty() ? "-c pot" : "-c pot (" + init + ")";
}

std::vector<GateSeed> collect_gate_seeds(const HPOTargets &trg) {
  std::vector<GateSeed> out(1);
  out[0].point = config_point_from_targets(trg);
  out[0].c_pot = true;
  auto add = [&](const std::vector<double> &point, const InitOrigin &o,
                 bool restart) {
    if (!o.is_seed()) return;
    const std::string name = o.name();
    auto it = std::find_if(out.begin(), out.end(), [&](const GateSeed &e) {
      return e.point == point;
    });
    if (it == out.end()) {
      out.push_back(GateSeed{point, {}, false, true, {}});
      it = std::prev(out.end());
    }
    it->in_seed_set = true;
    auto &names = restart ? it->restart_origins : it->origins;
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(name);
  };
  auto scan = [&](const OptimizerSpec *s) {
    if (!s || s->optim_init_origin.size() != s->optim_init.size()) return;
    for (std::size_t i = 0; i < s->optim_init.size(); ++i)
      add(s->optim_init[i], s->optim_init_origin[i], false);
  };
  const OptimizerSpec &sp = trg.spec;
  scan(&sp);
  if (sp.prestage) scan(sp.prestage.get());
  const RestartSpec &rs = sp.restart;
  if (rs.seed_origins.size() == rs.seed_points.size())
    for (std::size_t i = 0; i < rs.seed_points.size(); ++i)
      add(rs.seed_points[i], rs.seed_origins[i], true);
  return out;
}

core::Context seed_context(const core::Context &base, const HPOTargets &trg,
                           const std::vector<double> &point,
                           const DensityGates &g) {
  struct Point {
    const std::vector<double> &v;
    double operator()(std::size_t i) const { return v.at(i); }
  };
  core::Context c = base;
  update_candidate_context(c, trg, Point{point}, g.knot2b, g.knotmb);
  return c;
}

void validate_density_gates(const core::Context &train_ctx,
                            const HPOTargets &trg) {
  if (!trg.RHOGATE && !trg.RHOBARGATE) return;
  const std::string keys = trg.RHOGATE && trg.RHOBARGATE ? "RHOGATE and RHOBARGATE"
                           : trg.RHOGATE                 ? "RHOGATE"
                                                         : "RHOBARGATE";
  /* A refusal about the gate @p key (RHOGATE, RHOBARGATE, or "" for both:
   * the first of their lines), framed at its line. */
  auto at_gate = [&](const std::string &key, const std::string &what) {
    std::size_t line = 0;
    std::string text;
    auto take = [&](std::size_t l, const std::string &t) {
      if (l != 0 && (line == 0 || l < line)) { line = l; text = t; }
    };
    if (trg.RHOGATE && key != "RHOBARGATE")
      take(trg.RHOGATE->line, trg.RHOGATE->line_text);
    if (trg.RHOBARGATE && key != "RHOGATE")
      take(trg.RHOBARGATE->line, trg.RHOBARGATE->line_text);
    return std::runtime_error(line ? config_line_error(line, what, text)
                                   : what);
  };
  if (!context_is_bare_ream(train_ctx)) {
    const std::string head =
        keys + (trg.RHOGATE && trg.RHOBARGATE ? " evaluate" : " evaluates") +
        " the density of one DM_REAM many-body term or of one DM_Embed "
        "component, and ";
    const auto label = train_ctx.stored_value("TYPEMB", 0);
    const auto initmb = train_ctx.stored_value("INITMB");
    std::string have;
    if (!train_ctx.blocks("DESCRIPTOR").empty())
      have = "this configuration has " +
             std::to_string(models::embed_contexts(train_ctx).size()) +
             " DM_Embed components";
    else if (!initmb || core::to_upper(*initmb) != "TRUE" || !label)
      have = "this configuration has no many-body term";
    else
      have = "the many-body term of this configuration is " + *label +
             (*label == "DM_mJoin" ? " (DM_mJoin is not read by these checks)"
                                   : "");
    throw at_gate("", head + have);
  }
  const auto specs=models::embed_specs(train_ctx);
  const models::RhoForm form = specs.empty()?models::require_rhoform(train_ctx, "tadah hpo"):specs.at(0).form;
  /* The form of the density: RHOFORM of a DM_REAM term, the FORM of the
   * DENSITY block of the DM_Embed component. */
  const std::string form_label =
      specs.empty() ? std::string("RHOFORM ") + models::rhoform_name(form)
                    : std::string("FORM ") + models::rhoform_name(form) +
                          " of the DENSITY of DESCRIPTOR " + specs.at(0).id;
  if (trg.RHOGATE && form != models::RhoForm::Signed)
    throw at_gate("RHOGATE",
        "RHOGATE: under " + form_label + " every density rho(r) is >= 0, so "
        "the sign condition of RHOGATE holds for every candidate; RHOGATE is "
        "taken under RHOFORM SIGNED only (RHOBARGATE under every form)");

  const DensityGates g = build_density_gates(train_ctx, trg);
  const double rmin = g.reach.min_reachable;
  const std::string rmin_text =
      "RCUTMB = " + f4(rmin) + " A" +
      (g.reach.knot ? " (the largest CGRIDMB knot over its OPTIM box)" : "");
  if (!std::isfinite(rmin) || !(rmin > 0.0))
    throw at_gate("", keys + ": the many-body cutoff RCUTMB is not set.");
  if (trg.RHOGATE && trg.RHOGATE->r_lo && !(*trg.RHOGATE->r_lo < rmin))
    throw at_gate("RHOGATE",
        "RHOGATE R_LO " + gnum(*trg.RHOGATE->r_lo) + " A must lie below the "
        "smallest cutoff a candidate can have, " + rmin_text + ": the window "
        "[R_LO, RCUTMB] would be empty; a smaller R_LO gives the window a "
        "width.");
  if (trg.RHOBARGATE) {
    const RhoBarGateSpec &s = *trg.RHOBARGATE;
    const double d1 = models::eam_lattice::shells(s.lattice, 2.0).front().d;
    const double r1 = d1 * s.f_hi * s.a0;
    if (r1 > rmin)
      throw at_gate("RHOBARGATE",
          "RHOBARGATE: the first " + s.lattice + " shell at a = f_hi a0 = " +
          f4(s.f_hi * s.a0) + " A lies at r = " + f4(r1) + " A, beyond the "
          "smallest cutoff a candidate can have, " + rmin_text + ": rhobar(a) "
          "would lose its nearest neighbours; a smaller f_hi or a0, or a "
          "larger RCUTMB, keeps the first shell inside the cutoff.");
  }

  /* A density pair weight no candidate can make positive (WATOM fixed at
   * <= 0, or its OPTIM box not reaching above 0) saturates both gates on
   * every evaluation: an error under either SEED mode, which stands in for
   * the per-seed errors below. */
  std::vector<std::string> errors;
  const std::string fixed_w = fixed_weight_problem(train_ctx, trg, g);
  if (!fixed_w.empty())
    errors.push_back(
        keys + ": " + fixed_w + " on every candidate (WATOM has no OPTIM "
        "line, or its OPTIM box does not reach above 0), so every evaluation "
        "would be refused before training, under SEED ERROR and SEED WARN "
        "alike. The checks of " + keys + " need every density pair weight "
        "> 0, which a WATOM above 0 gives (under RHOFORM SIGNED the sign of "
        "rho is that of the AMPGRIDMB amplitudes); without the " + keys +
        " line(s) the run makes no such check.");

  /* The seeds, each evaluated as a candidate (seed_context) by the
   * per-candidate functions. */
  const std::vector<GateSeed> seeds = collect_gate_seeds(trg);
  auto warn = [](const std::string &m) {
    std::cerr << "[HPO] WARNING: " << m << "\n";
    HPOLog::instance().warning(m);
  };
  const std::string repair =
      " A seed that fails is refused under SEED ERROR, the default, and is "
      "evaluated as a rejected start under SEED WARN. tadah refit --edit "
      "changes the density of a potential file; a WARM seed takes the values "
      "of its search parameters from its file and the others from the "
      "training configuration.";
  /* Whether the optimiser starts from a given seed is not known here (a
   * single-start backend consumes only the first of several). */
  const std::string seed_warn_note =
      " SEED WARN: the run proceeds; if the optimiser starts from this seed, "
      "that evaluation is rejected before training (band D).";

  if (g.rho) {
    const RhoGateTable &t = *g.rho;
    const std::string arm_why =
        "armed per candidate on " + join(t.armed_on, ", ") +
        ": candidates with rho_min < -tol*rho_max are rejected before "
        "training as graded band-D failures";
    std::vector<std::string> verdicts;
    const bool all_rejected = !fixed_w.empty();
    for (const GateSeed &s : seeds) {
      const RhoGateResult r = eval_rho_gate(t, seed_context(train_ctx, trg, s.point, g));
      const std::string v = r.abstain ? "not evaluated" : r.fail ? "FAIL" : "PASS";
      verdicts.push_back(s.label() + " " + v);
      if (!r.fail || all_rejected) continue;
      const std::string m = std::string("RHOGATE: ") + (s.in_seed_set ? "seed " : "") +
                            s.label() + " fails the rho(r) sign gate: " +
                            rho_fail_detail(r, t.spec) + ".";
      if (!s.in_seed_set)
        warn(m + " It is not an INIT start: the run proceeds.");
      else if (t.spec.seed_warn)
        warn(m + seed_warn_note);
      else
        errors.push_back(m + repair);
    }
    const std::string window =
        t.spec.r_lo ? "[" + gnum(*t.spec.r_lo) + " A (R_LO), RCUTMB of each candidate]"
                    : std::string("[0.15 RCUTMB, RCUTMB] of each candidate");
    HPOLog::instance().info(
        "RHOGATE: rho(r) sign gate, tol " + e4(t.spec.tol) + ", window " +
        window + ", " + std::to_string(t.spec.npts) + "-point grid, SEED " +
        (t.spec.seed_warn ? "WARN" : "ERROR") + "; " + arm_why +
        "; seeds: " + join(verdicts, ", "));
  }
  if (g.rhobar) {
    const RhoBarGateTable &t = *g.rhobar;
    const std::string arm_why =
        "armed per candidate on " + join(t.armed_on, ", ") +
        ": candidates whose rhobar(a) is not strictly decreasing are "
        "rejected before training as graded band-D failures";
    std::vector<std::string> verdicts;
    const bool all_rejected = !fixed_w.empty();
    for (const GateSeed &s : seeds) {
      const RhoBarGateResult r =
          eval_rhobar_gate(t, seed_context(train_ctx, trg, s.point, g));
      const std::string v = r.abstain ? "not evaluated" : r.fail ? "FAIL" : "PASS";
      verdicts.push_back(s.label() + " " + v);
      if (!r.fail || all_rejected) continue;
      const std::string m = std::string("RHOBARGATE: ") + (s.in_seed_set ? "seed " : "") +
                            s.label() + " fails the lattice rhobar(a) gate: " +
                            rhobar_fail_detail(r, t.spec) + ".";
      if (!s.in_seed_set)
        warn(m + " It is not an INIT start: the run proceeds.");
      else if (t.spec.seed_warn)
        warn(m + seed_warn_note);
      else
        errors.push_back(m + repair);
    }
    HPOLog::instance().info(
        "RHOBARGATE: lattice rhobar(a) gate, " + rhobar_grid_text(t.spec) +
        ", " + std::to_string(t.shells.size()) + " shells within the largest "
        "reachable RCUTMB, SEED " + (t.spec.seed_warn ? "WARN" : "ERROR") +
        "; " + arm_why + "; seeds: " + join(verdicts, ", "));
  }
  /* A density pair weight no candidate makes positive refuses every
   * evaluation: the seeds are not judged, and the refusal names the gate
   * line. */
  if (!fixed_w.empty())
    throw at_gate("", errors.front());
  if (!errors.empty()) {
    std::string msg = errors.front();
    for (std::size_t i = 1; i < errors.size(); ++i) msg += "\n  " + errors[i];
    throw std::runtime_error(msg);
  }
}

DensitySummary summarise_density(const DensityGates &g,
                                 const core::Context &cand) {
  DensitySummary s;
  try {
    double r_hi = 0.0;
    const std::unique_ptr<models::Cut_Base> fc = candidate_cutoff(cand, r_hi);
    const CandidateDensity d = candidate_density(cand, fc.get());
    s.form = models::rhoform_name(d.form);
    {
      // tadah hpo always has WATOM here (it inserts WATOM = Z when the
      // configuration has none); "no WATOM" is for other callers' contexts.
      const std::vector<double> w = models::D_Base::density_pair_weights(cand);
      if (w.empty()) {
        s.weights = "W = 1 (no WATOM)";
      } else {
        const auto mm = std::minmax_element(w.begin(), w.end());
        s.weights = *mm.first == *mm.second
                        ? "W = " + gnum(*mm.first)
                        : "W in [" + gnum(*mm.first) + ", " + gnum(*mm.second) + "]";
      }
    }
    const RhoGateSpec *rs = g.rho ? &g.rho->spec : nullptr;
    s.npts = rs ? rs->npts : 20001;
    s.r_hi = r_hi;
    s.r_lo = rs && rs->r_lo ? *rs->r_lo : 0.15 * r_hi;
    if (!(s.r_lo < s.r_hi))
      throw std::runtime_error("the report window [r_lo, RCUTMB] is empty");
    std::vector<double> rho, terms;
    const models::DensityScan sc = d.scan_grid(s.r_lo, s.r_hi, s.npts, *fc, &rho, &terms);
    s.finite = sc.finite;
    s.rho_min = sc.rho_min;
    s.r_min = sc.r_at_min;
    s.rho_max = sc.rho_max;
    s.r_max = sc.r_at_max;
    if (sc.finite) {
      const std::vector<double> r = models::ReamDensity::linspace(s.r_lo, s.r_hi, s.npts);
      double rise = -std::numeric_limits<double>::infinity();
      double term_max = 0.0;
      s.max_drho = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i + 1 < rho.size(); ++i) {
        const double dr = rho[i + 1] - rho[i];
        rise = std::max(rise, dr);
        const double slope = dr / (r[i + 1] - r[i]);
        if (slope > s.max_drho) {
          s.max_drho = slope;
          s.r_max_drho = r[i];
        }
      }
      for (const double x : terms) term_max = std::max(term_max, x);
      s.non_increasing =
          rise <= 8.0 * std::numeric_limits<double>::epsilon() * term_max;
    }
    if (g.rho) {
      RhoGateResult r;
      r.npts = s.npts;
      r.r_lo = s.r_lo;
      r.r_hi = s.r_hi;
      const std::string w = weight_problem(cand);
      s.rhogate = w.empty() ? rho_verdict(r, sc, rho, g.rho->spec.tol) : saturate(r, w);
      s.tol = g.rho->spec.tol;
    }
    s.derivable = true;
    if (g.rhobar) {
      s.rhobar = eval_rhobar_gate(*g.rhobar, cand);
      s.rhobar_spec = g.rhobar->spec;
    }
  } catch (const std::exception &e) {
    s.derivable = false;
    s.reason = e.what();
  }
  return s;
}

std::vector<DensityReportEntry> startup_density_report(const core::Context &ctx,
                                                       const HPOTargets &trg) {
  std::vector<DensityReportEntry> out;
  try {
    const DensityGates g = build_density_gates(ctx, trg);
    out.push_back({"-c pot", "", summarise_density(g, ctx)});
    std::size_t n_warm = 0;
    for (const GateSeed &s : collect_gate_seeds(trg)) {
      if (s.c_pot) continue;
      bool warm = false;
      for (const std::string &o : s.origins) warm = warm || o.rfind("WARM ", 0) == 0;
      if (!warm) continue;
      const std::string file = s.origins.front().substr(std::string("WARM ").size());
      out.push_back({"WARM " + std::to_string(++n_warm),
                     " (file " + file + "; OPTIM dims from the seed, the rest "
                     "from -c)",
                     summarise_density(g, seed_context(ctx, trg, s.point, g))});
    }
  } catch (const std::exception &e) {
    DensityReportEntry e0{"-c pot", "", DensitySummary{}};
    e0.summary.reason = e.what();
    out.assign(1, e0);
  }
  return out;
}

std::string describe_density_form(const std::string &form) {
  if (form == "SIGNED")
    return "RHOFORM SIGNED -- rho = W f_c sum a_n psi_n, signed amplitudes "
           "(no sign guarantee)";
  if (form == "NONNEG")
    return "RHOFORM NONNEG -- rho = W max(f_c, 0) sum a_n psi_n, a_n >= 0 "
           "(rho >= 0 by construction)";
  if (form == "MONOTONE")
    return "RHOFORM MONOTONE -- rho = W max(f_c, 0) sum a_n S_n, a_n >= 0 "
           "(rho >= 0 and non-increasing by construction)";
  return "RHOFORM " + form;
}

std::string format_density_rho(const DensitySummary &s) {
  if (!s.derivable) return "not derivable: " + s.reason;
  std::string t = "[" + f4(s.r_lo) + ", " + f4(s.r_hi) + "] A, " +
                  std::to_string(s.npts) + " pts: ";
  if (!s.finite) {
    t += "non-finite values on the window";
  } else {
    t += "min " + e4(s.rho_min) + " (r " + f4(s.r_min) + ") max " + e4(s.rho_max) +
         " (r " + f4(s.r_max) + ") rel " +
         (s.rho_max > 0.0 ? e4(s.rho_min / s.rho_max) : std::string("n/a")) +
         "; max rho' " + fmt("%+.2e", s.max_drho) + " (r " + f4(s.r_max_drho) +
         "); non-increasing on window: " + (s.non_increasing ? "yes" : "no");
  }
  t += "; " + s.weights;
  if (s.rhogate) {
    const RhoGateResult &r = *s.rhogate;
    t += "; RHOGATE tol " + e4(s.tol) + ": ";
    t += !r.fail ? std::string("PASS")
         : r.saturated ? "FAIL (" + r.reason + ")"
                       : "FAIL (v = " + e4(r.v) + ")";
  }
  return t;
}

std::string format_density_rhobar(const DensitySummary &s) {
  if (!s.derivable) return "not derivable: " + s.reason;
  if (!s.rhobar) return "n/a (RHOBARGATE is not configured)";
  const RhoBarGateResult &r = *s.rhobar;
  const std::string grid = s.rhobar_spec.lattice + " a0 " + gnum(s.rhobar_spec.a0) +
                           " A, a/a0 [" + gnum(s.rhobar_spec.f_lo) + ", " +
                           gnum(s.rhobar_spec.f_hi) + "], " +
                           std::to_string(s.rhobar_spec.npts) + " pts: ";
  if (r.abstain) return grid + "not derivable: " + r.reason;
  if (r.saturated) return grid + r.reason + ": RHOBARGATE FAIL";
  if (!r.fail) return grid + "strictly decreasing (0 rising steps)";
  return grid + "rises at " + std::to_string(r.rises) + " of " +
         std::to_string(s.rhobar_spec.npts - 1) + " steps, first at a/a0 = " +
         f4(r.first_rise) + " (v = " + e4(r.v) + "): RHOBARGATE FAIL";
}

} // namespace detail
} // namespace hpo
} // namespace tadah

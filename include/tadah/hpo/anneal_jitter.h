#pragma once
/*
 * anneal_jitter -- measure a per-dimension initial-jitter magnitude.
 *
 * `INIT_MODE JITTER AUTO <p>` asks: how far can each coordinate be displaced
 * from the seed before the loss degrades by more than a relative tolerance p?
 * The answer is a per-dimension sigma, and replica initialisation then draws
 * uniformly on +/-sigma so p is a bound on every coordinate rather than an
 * average over them.
 *
 * This header is deliberately PURE: no host, no thread pool, no MPI, no
 * logging. It takes a callable that evaluates the objective with ONE
 * coordinate moved, and returns a plan. That is what makes it testable
 * against closed-form objectives (tests/test_anneal_jitter.cpp) instead of
 * only through a full annealing run, and it keeps the orchestration
 * decisions -- budget, fan-out, journal -- in the driver where they belong.
 *
 * WHAT sigma MEANS. sigma is always the largest displacement that was
 * ACTUALLY MEASURED in band, never an interpolated bracket midpoint. This is
 * the deliberate difference from pt_detail::coordinate_warmup, which targets
 * an acceptance RATE and may legitimately return an untested geometric mean:
 * here the deliverable is a BOUND, and an untested bound is not a bound.
 *
 * NO RNG. The ladder is a fixed geometric descent from the cap and the
 * bisection is deterministic, so the result is a pure function of
 * (x0, L0, box, p, options) and the objective -- identical for any thread
 * count, rank count or completion order. That is a stronger reproducibility
 * property than the step warm-up has, and it is why the anchor (x0, L0) is
 * frozen for the whole phase rather than tracking the best-so-far.
 */

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {
namespace anneal {

struct JitterOptions {
  double      tol       = 0.05;  /* p: allowed relative loss degradation   */
  double      max_frac  = 0.5;   /* cap on sigma / OPTIM width             */
  std::size_t max_probes = 6;    /* per dimension, both signs together     */

  /* Cost lever. 0 = measure every dimension (the default: no silent
   * capping). Otherwise measure this many, chosen round-robin across
   * parameter FAMILIES, and give each unmeasured dimension its family's
   * median sigma/width. Measured cost is ~4 probes per dimension, so on an
   * 800-dimension search space this is the difference between thousands of
   * objective evaluations before the run starts and a few hundred. An
   * inferred sigma is flagged JitterInferred and counted in the summary --
   * it must never be mistaken for a measurement. */
  std::size_t sample = 0;

  /* Honesty lever, cheap. sigma is the largest displacement measured in
   * band, which makes it a GRID certificate: nothing was tested between the
   * grid points. Re-probe this many dimensions at sigma/2; a failure there
   * means the loss is non-monotone in |displacement| along that axis and the
   * bound does not hold in its interior. Reported with the sample size, so
   * the claim stays proportionate to the evidence. 0 = off. */
  std::size_t audit_sample = 32;
};

struct DimJitter {
  double      sigma  = 0.0;      /* largest MEASURED in-band displacement  */
  double      lower  = 0.0;      /* same, per side, before the min         */
  double      upper  = std::numeric_limits<double>::infinity();
                                 /* smallest measured OUT-of-band          */
  unsigned    status = JitterOk;
  std::size_t probes = 0;
};

struct JitterPlan {
  std::vector<double>    sigma;      /* per dimension, ready for seed_replica */
  std::vector<DimJitter> dims;
  double                 L0        = 0.0;
  double                 threshold = 0.0;   /* L0 + p*|L0|                 */
  std::size_t            probes    = 0;
  std::size_t            audited      = 0;  /* dims re-probed at sigma/2   */
  std::size_t            non_monotone = 0;  /* of those, out of band       */
  bool                   abandoned = false; /* anchor unusable; see below  */
};

/* Is this loss inside the tolerance band?
 *
 * One-sided in the WORSE direction only: a probe that IMPROVES the loss
 * passes. Shrinking sigma because a direction helped would penalise exactly
 * the directions worth exploring. An infeasible loss (non-rankable, or in the
 * graded FAILSCORE band) never passes however small the numeric gap looks --
 * it is not a measurement of the objective. */
inline bool jitter_in_band(double f, double threshold,
                           const std::optional<double>& failscore)
{
  if (!loss_is_finite(f, failscore)) return false;
  return f <= threshold;
}

/* Measure one dimension.
 *
 * `probe(value) -> double` evaluates the objective with THIS coordinate set
 * to `value` and every other coordinate at the seed. */
template <class ProbeFn>
DimJitter calibrate_dim(double x0_i, double lo, double hi, double threshold,
                        const std::optional<double>& failscore,
                        const JitterOptions& opt, ProbeFn&& probe)
{
  DimJitter d;
  const double width = hi - lo;
  const double floor = resolvable_coordinate_scale(x0_i, lo, hi);
  const double cap   = std::min(opt.max_frac * width, width);

  /* Room is clamped PER SIDE. Using max(room_lo, room_hi) -- the obvious
   * form -- leaves the box on the tighter side: under OPTIM_SCALE AUTO every
   * width is 1, so a seed at 0.5 with s = 1.0 escapes in both directions. */
  const double room_hi = std::min(cap, hi - x0_i);
  const double room_lo = std::min(cap, x0_i - lo);

  const bool has_hi = room_hi > floor;
  const bool has_lo = room_lo > floor;
  if (!has_hi && !has_lo) {          /* pinned into a corner */
    d.sigma = floor;
    d.status |= JitterNumericalFloor;
    return d;
  }
  if (!has_hi || !has_lo) d.status |= JitterOneSided;

  /* Each side gets its own probe allowance. Sharing one counter lets the
   * first side spend everything and leaves the second unmeasured, which then
   * reads as sigma = 0 and collapses the result to the numerical floor. */
  const std::size_t sides = (has_hi && has_lo) ? 2u : 1u;
  const std::size_t per_side = std::max<std::size_t>(1, opt.max_probes / sides);

  /* One side: screen at the cap, then BISECT ON log s between the largest
   * displacement known to pass and the smallest known to fail.
   *
   * Bisecting the exponent rather than descending by a fixed ratio is what
   * makes a stiff coordinate affordable: the interesting scale can be ten or
   * more decades below the cap, which a geometric descent reaches only after
   * as many probes as there are decades, whereas log-bisection halves the
   * remaining exponent range each time. */
  /* Per-side result. `capped` travels with the side rather than being set on
   * the dimension, because the dimension's sigma is the MIN over sides: a
   * coordinate that is flat one way and steep the other is asymmetric, not
   * insensitive, and reporting it as capped would contradict its own sigma. */
  struct SideResult { double sigma = 0.0; bool capped = false; };

  auto measure_side = [&](double sign, double room) -> SideResult {
    std::size_t used = 0;
    double pass_s = 0.0;   /* 0 always passes: it IS the seed */
    double fail_s = std::numeric_limits<double>::infinity();

    const double f_cap = probe(x0_i + sign * room);
    ++d.probes; ++used;
    if (jitter_in_band(f_cap, threshold, failscore)) return {room, true};
    if (!loss_is_finite(f_cap, failscore)) d.status |= JitterCliff;
    fail_s = room;

    while (used < per_side) {
      const double lo_ref = pass_s > 0.0 ? pass_s : floor;
      if (!(fail_s / lo_ref > 1.25)) break;      /* converged */
      const double mid = std::sqrt(lo_ref * fail_s);
      if (!(mid > floor) || !(mid < fail_s)) break;
      const double f = probe(x0_i + sign * mid);
      ++d.probes; ++used;
      if (jitter_in_band(f, threshold, failscore)) {
        pass_s = mid;
      } else {
        if (!loss_is_finite(f, failscore)) d.status |= JitterCliff;
        fail_s = mid;
      }
    }
    d.upper = std::min(d.upper, fail_s);
    return {pass_s, false};
  };

  SideResult up_side, down_side;
  if (has_hi) up_side   = measure_side(+1.0, room_hi);
  if (has_lo) down_side = measure_side(-1.0, room_lo);

  double sigma;
  bool capped;
  if (has_hi && has_lo) {
    /* A bound must hold on BOTH sides, so the tighter one wins -- and the
     * "capped" verdict belongs to that side alone. */
    const bool up_wins = up_side.sigma <= down_side.sigma;
    sigma  = up_wins ? up_side.sigma : down_side.sigma;
    capped = up_wins ? up_side.capped : down_side.capped;
    const double a = std::max(up_side.sigma, down_side.sigma);
    const double b = std::min(up_side.sigma, down_side.sigma);
    if (b > 0.0 && a / b > 2.0) d.status |= JitterAsymmetric;
  } else {
    const SideResult& only = has_hi ? up_side : down_side;
    sigma  = only.sigma;
    capped = only.capped;
  }
  if (capped) d.status |= JitterSupportCapped | JitterInsensitive;

  if (!(sigma > floor)) {
    sigma = floor;
    d.status |= JitterNumericalFloor;
    d.status &= ~static_cast<unsigned>(JitterInsensitive);
  }
  d.sigma = sigma;
  d.lower = sigma;
  return d;
}

/* Measure every dimension.
 *
 * `probe(i, value)` evaluates the objective with coordinate `i` set to
 * `value`. `fallback` is the sigma vector to return if the anchor is
 * unusable -- see below.
 *
 * ANCHOR PATHOLOGY. A RELATIVE tolerance needs a loss to be relative TO. If
 * L0 is not rankable (a FAILSCORE seed) or |L0| underflows, p*|L0| is
 * meaningless and every dimension would "pass" at the cap, silently handing
 * back a half-box jitter. AUTO is abandoned instead: zero probes, the
 * fallback sigma, and every dimension flagged JitterNoAnchor for the caller
 * to report. */
template <class ProbeFn>
JitterPlan calibrate_jitter(const OptimizerSpec& sp,
                            const std::vector<double>& x0, double L0,
                            const JitterOptions& opt,
                            const std::vector<double>& fallback,
                            ProbeFn&& probe)
{
  const std::size_t dim = sp.low.size();
  JitterPlan plan;
  plan.L0 = L0;
  plan.dims.assign(dim, DimJitter{});
  plan.sigma.assign(dim, 0.0);

  const bool usable = loss_is_rankable(L0, sp.failscore) &&
                      std::isfinite(L0) &&
                      std::fabs(L0) > std::numeric_limits<double>::min();
  if (!usable) {
    plan.abandoned = true;
    plan.sigma = fallback;
    for (auto& d : plan.dims) d.status |= JitterNoAnchor;
    return plan;
  }

  plan.threshold = L0 + opt.tol * std::fabs(L0);

  /* Which dimensions to actually measure. With `sample` unset that is all of
   * them; otherwise walk the FAMILIES round-robin so a sample of 64 out of
   * 800 still touches every parameter group rather than the first 64
   * coordinates, which on a grouped search space would all belong to one. */
  std::vector<std::size_t> measure;
  std::vector<std::size_t> family_of(dim, 0);
  std::vector<std::string> families;
  {
    std::vector<std::vector<std::size_t>> members;
    for (std::size_t i = 0; i < dim; ++i) {
      std::string fam = sp.parameter_labels.size() == dim
          ? sp.parameter_labels[i] : std::string("dim");
      const auto br = fam.find('[');
      if (br != std::string::npos) fam.resize(br);   /* KEY[3] -> KEY */
      auto it = std::find(families.begin(), families.end(), fam);
      if (it == families.end()) {
        families.push_back(fam);
        members.emplace_back();
        it = families.end() - 1;
      }
      const std::size_t f =
          static_cast<std::size_t>(it - families.begin());
      family_of[i] = f;
      members[f].push_back(i);
    }
    if (opt.sample == 0 || opt.sample >= dim) {
      measure.resize(dim);
      for (std::size_t i = 0; i < dim; ++i) measure[i] = i;
    } else {
      std::vector<std::size_t> cursor(families.size(), 0);
      while (measure.size() < opt.sample) {
        bool progressed = false;
        for (std::size_t f = 0; f < families.size() &&
                                measure.size() < opt.sample; ++f) {
          if (cursor[f] < members[f].size()) {
            measure.push_back(members[f][cursor[f]++]);
            progressed = true;
          }
        }
        if (!progressed) break;
      }
      std::sort(measure.begin(), measure.end());
    }
  }

  for (std::size_t i : measure) {
    plan.dims[i] = calibrate_dim(
        x0[i], sp.low[i], sp.high[i], plan.threshold, sp.failscore, opt,
        [&](double value) { return probe(i, value); });
    plan.sigma[i] = plan.dims[i].sigma;
    plan.probes += plan.dims[i].probes;
  }

  /* Unmeasured dimensions inherit their family's MEDIAN sigma/width -- a
   * fraction, not an absolute length, so a family whose members have
   * different box widths still transfers sensibly. */
  if (measure.size() < dim) {
    std::vector<std::vector<double>> frac_by_family(families.size());
    std::vector<double> all_frac;
    for (std::size_t i : measure) {
      const double w = sp.high[i] - sp.low[i];
      if (!(w > 0.0)) continue;
      frac_by_family[family_of[i]].push_back(plan.sigma[i] / w);
      all_frac.push_back(plan.sigma[i] / w);
    }
    const auto median = [](std::vector<double> v) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double global = median(all_frac);
    std::vector<char> measured(dim, 0);
    for (std::size_t i : measure) measured[i] = 1;
    for (std::size_t i = 0; i < dim; ++i) {
      if (measured[i]) continue;
      const auto& fam = frac_by_family[family_of[i]];
      const double f = fam.empty() ? global : median(fam);
      plan.sigma[i] = f * (sp.high[i] - sp.low[i]);
      plan.dims[i].sigma = plan.sigma[i];
      plan.dims[i].status |= JitterInferred;
    }
  }

  /* Interior audit: is the bound still a bound halfway in? */
  plan.audited = 0;
  plan.non_monotone = 0;
  if (opt.audit_sample > 0 && !measure.empty()) {
    const std::size_t want = std::min(opt.audit_sample, measure.size());
    const std::size_t stride = std::max<std::size_t>(1, measure.size() / want);
    for (std::size_t k = 0; k < measure.size() && plan.audited < want;
         k += stride) {
      const std::size_t i = measure[k];
      if (!(plan.sigma[i] > 0.0)) continue;
      const double half = plan.sigma[i] * 0.5;
      const double f = probe(i, x0[i] + half);
      ++plan.probes;
      ++plan.audited;
      if (!jitter_in_band(f, plan.threshold, sp.failscore)) {
        plan.dims[i].status |= JitterNonMonotone;
        ++plan.non_monotone;
      }
    }
  }
  return plan;
}

}  // namespace anneal
}  // namespace hpo
}  // namespace tadah

/* =========================================================================
 * Driver orchestration
 *
 * Everything above is pure. Below is the part that talks to a host, a clock
 * and the journal, shared by ANNEAL_PT and ANNEAL_PA so the two cannot drift.
 * ========================================================================= */

#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/probe_phase.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace tadah {
namespace hpo {
namespace anneal {

/* Percentile of a sorted-in-place copy; for the summary distribution. */
inline double jitter_pct(std::vector<double> v, double q)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const std::size_t i = static_cast<std::size_t>(pos);
  const double frac = pos - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] * (1.0 - frac) + v[i + 1] * frac : v.back();
}

/* Run the AUTO calibration for one driver.
 *
 * BUDGET. The phase never touches the optimiser's SharedBudget, so "not
 * charged to MAXEVAL" holds by construction rather than by remembering to
 * skip a counter. What DOES bound it is a probe cap and a share of MAXTIME,
 * both journaled up front: with MAXEVAL exempt, time is the only axis left.
 *
 * The eval counter in HPOLog still ticks for every probe -- they are real
 * objective evaluations -- so the summary reconciles the two numbers
 * explicitly. A "Total evaluations" larger than MAXEVAL is otherwise a very
 * reasonable bug report. */
template <class Vec, class HostT>
JitterPlan run_jitter_calibration(const OptimizerSpec& sp, const Knobs& kb,
                                  const Vec& x0, double f0,
                                  const std::vector<double>& fallback,
                                  HostT& host, StopCtx& ctx,
                                  double max_time,
                                  const char* banner_ctx,
                                  int rank = 0, int world = 1)
{
  auto& log = HPOLog::instance();
  const std::size_t dim = sp.low.size();
  const std::string tag = banner_ctx;

  /* Contiguous per-rank dimension blocks, the same shape the drivers use to
   * split ladder slots. Every rank measures its own block and they are
   * gathered afterwards; the alternative (rank 0 measures everything) leaves
   * the other ranks idle through the most expensive part of startup. */
  const std::size_t base  = dim / static_cast<std::size_t>(world);
  const std::size_t extra = dim % static_cast<std::size_t>(world);
  const auto block_count = [&](int r) {
    return base + (static_cast<std::size_t>(r) < extra ? 1u : 0u);
  };
  std::size_t off_local = 0;
  for (int r = 0; r < rank; ++r) off_local += block_count(r);
  const std::size_t n_local = block_count(rank);

  JitterOptions opt;
  opt.tol        = kb.init_jitter_tol;
  opt.max_frac   = kb.init_jitter_max;
  opt.max_probes   = kb.init_jitter_probes;
  opt.sample       = kb.init_jitter_sample;
  opt.audit_sample = kb.init_jitter_audit;

  const std::size_t measured_dims =
      (opt.sample == 0 || opt.sample >= dim) ? dim : opt.sample;
  const std::size_t planned =
      measured_dims * opt.max_probes + std::min(opt.audit_sample, measured_dims);
  const std::size_t cap =
      kb.init_jitter_maxeval > 0 ? kb.init_jitter_maxeval : planned;
  const double time_budget = max_time > 0.0
      ? max_time * kb.init_jitter_maxtime_frac
      : 0.0;

  std::vector<double> x0v(dim);
  for (std::size_t i = 0; i < dim; ++i) x0v[i] = x0(static_cast<long>(i));

  /* ---- preflight, BEFORE the first probe and flushed ------------------
   * A calibration on a real search space is thousands of relaxations. The
   * user gets the bill, the cap and where the cap came from while there is
   * still time to kill the run. */
  {
    std::ostringstream o;
    o << tag << " init-jitter calibration (INIT_MODE JITTER AUTO "
      << opt.tol << "): " << dim << " dimension(s), " << measured_dims
      << " measured";
    if (measured_dims < dim)
      o << " (INIT_JITTER_SAMPLE; the other " << (dim - measured_dims)
        << " inherit their family's median)";
    o << " x <= " << opt.max_probes << " probe(s)";
    if (opt.audit_sample > 0)
      o << " + <= " << std::min(opt.audit_sample, measured_dims)
        << " interior audit probe(s)";
    o << " = <= " << planned << " objective evaluation(s), cap "
      << cap << (kb.init_jitter_maxeval > 0 ? " (INIT_JITTER_MAXEVAL)"
                                            : " (AUTO)")
      << ".  L0=" << std::scientific << std::setprecision(6) << f0
      << "  band <= " << (f0 + opt.tol * std::fabs(f0))
      << ".  NOT charged to MAXEVAL";
    if (sp.stop.maxeval) o << " (MAXEVAL=" << *sp.stop.maxeval << ")";
    if (time_budget > 0.0)
      o << "; time budget " << std::fixed << std::setprecision(1)
        << time_budget << " s (" << kb.init_jitter_maxtime_frac
        << " of MAXTIME)";
    if (rank == 0) {
      log.info(o.str());
      std::cout << "[HPO] " << o.str() << "\n" << std::flush;
    }
  }

  /* Provable arithmetic, so WARNING rather than HINT. */
  if (rank == 0 && sp.stop.maxeval && planned > *sp.stop.maxeval) {
    std::ostringstream w;
    w << tag << " init-jitter calibration may spend up to " << planned
      << " evaluation(s) before the search starts, more than MAXEVAL ("
      << *sp.stop.maxeval << "). It is not charged against MAXEVAL, but it "
         "is real wall time. Lower INIT_JITTER_PROBES or set "
         "INIT_JITTER_MAXEVAL.";
    log.warning(w.str());
  }

  const auto t_start = std::chrono::steady_clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t_start).count();
  };

  probe::Scope phase;          /* stagnation off, checkpoints off, warm-start
                                * namespaced -- see probe_phase.h */
  std::size_t spent = 0, done_dims = 0, reported = 0;
  bool truncated = false;
  const std::size_t report_every = std::max<std::size_t>(1, dim / 10);

  Vec xp = x0;
  auto probe_fn = [&](std::size_t i, double value) -> double {
    if (spent >= cap || truncated) { truncated = true; return f0; }
    if (time_budget > 0.0 && elapsed() > time_budget) {
      truncated = true; return f0;
    }
    if (ctx.stop.load(std::memory_order_relaxed)) { truncated = true; return f0; }
    const double keep = xp(static_cast<long>(i));
    xp(static_cast<long>(i)) = value;
    const double f = host(sp.out(xp));
    xp(static_cast<long>(i)) = keep;
    ++spent;
    if (done_dims / report_every > reported) {
      reported = done_dims / report_every;
      std::ostringstream p;
      p << tag << " init-jitter " << (100 * done_dims / (dim ? dim : 1))
        << "% (" << done_dims << "/" << dim << " dims, " << spent
        << " probes, " << std::fixed << std::setprecision(0) << elapsed()
        << " s)";
      log.info(p.str());
    }
    return f;
  };

  /* Measure only this rank's block. The kernel is dimension-agnostic, so a
   * block is just a narrower spec sharing the same anchor. */
  OptimizerSpec local_sp = sp;
  local_sp.low.assign(sp.low.begin() + off_local,
                      sp.low.begin() + off_local + n_local);
  local_sp.high.assign(sp.high.begin() + off_local,
                       sp.high.begin() + off_local + n_local);
  const std::vector<double> local_x0(x0v.begin() + off_local,
                                     x0v.begin() + off_local + n_local);
  const std::vector<double> local_fallback(
      fallback.begin() + off_local, fallback.begin() + off_local + n_local);

  auto local = calibrate_jitter(local_sp, local_x0, f0, opt, local_fallback,
                                [&](std::size_t i, double v) {
                                  const double f = probe_fn(off_local + i, v);
                                  if (i + 1 > done_dims) done_dims = i + 1;
                                  return f;
                                });
  if (truncated)
    for (auto& d : local.dims) if (d.probes == 0) d.status |= JitterInterrupted;

  /* ---- gather, then BROADCAST ----------------------------------------
   * The gather is for reporting; the broadcast is for CORRECTNESS. Every
   * rank seeds its own local replicas, so all of them must draw from the
   * SAME sigma vector -- a rank keeping only its own block would jitter its
   * slots with a vector nobody else agrees on, and the ensemble would no
   * longer be one ensemble. */
  JitterPlan plan;
  plan.L0           = local.L0;
  plan.threshold    = local.threshold;
  plan.abandoned    = local.abandoned;
  plan.probes       = local.probes;
  plan.audited      = local.audited;
  plan.non_monotone = local.non_monotone;
  plan.sigma.assign(dim, 0.0);
  plan.dims.assign(dim, DimJitter{});

  if (world == 1) {
    plan.sigma = local.sigma;
    plan.dims  = local.dims;
  } else {
    std::vector<int> counts(static_cast<std::size_t>(world));
    for (int r = 0; r < world; ++r)
      counts[static_cast<std::size_t>(r)] = static_cast<int>(block_count(r));

    /* Three parallel per-dim fields; status and probe count cross as doubles
     * exactly as the step-scale gather already does. */
    std::vector<double> l_sigma(n_local), l_status(n_local), l_probes(n_local),
                        l_upper(n_local);
    for (std::size_t i = 0; i < n_local; ++i) {
      l_sigma[i]  = local.sigma[i];
      l_status[i] = static_cast<double>(local.dims[i].status);
      l_probes[i] = static_cast<double>(local.dims[i].probes);
      l_upper[i]  = local.dims[i].upper;
    }
    std::vector<double> g_sigma(dim), g_status(dim), g_probes(dim),
                        g_upper(dim);
    hpo_gatherv(l_sigma,  g_sigma,  counts);
    hpo_gatherv(l_status, g_status, counts);
    hpo_gatherv(l_probes, g_probes, counts);
    hpo_gatherv(l_upper,  g_upper,  counts);

    hpo_bcast(g_sigma);                    /* <- the correctness-critical one */
    hpo_bcast(g_status);
    hpo_bcast(g_probes);
    hpo_bcast(g_upper);

    plan.sigma = g_sigma;
    for (std::size_t i = 0; i < dim; ++i) {
      plan.dims[i].sigma  = g_sigma[i];
      plan.dims[i].lower  = g_sigma[i];
      plan.dims[i].upper  = g_upper[i];
      plan.dims[i].status = static_cast<unsigned>(g_status[i]);
      plan.dims[i].probes = static_cast<std::size_t>(g_probes[i]);
    }
    plan.probes = static_cast<std::size_t>(hpo_allreduce_sum_sz(local.probes));
    /* The audit is per-rank too, so its verdict is only meaningful summed:
     * reporting one rank's slice would understate both the sample size and
     * the number of non-monotone axes. */
    plan.audited =
        static_cast<std::size_t>(hpo_allreduce_sum_sz(local.audited));
    plan.non_monotone =
        static_cast<std::size_t>(hpo_allreduce_sum_sz(local.non_monotone));
    /* An abandoned anchor is a property of the shared (x0, f0), so every
     * rank reaches the same verdict; assert it rather than assume it. */
    plan.abandoned = hpo_allreduce_max(local.abandoned ? 1 : 0) != 0;
    if (plan.abandoned) plan.sigma = fallback;
  }
  spent = plan.probes;

  /* ---- summary ------------------------------------------------------- */
  if (plan.abandoned) {
    if (rank != 0) return plan;
    std::ostringstream w;
    w << tag << " init-jitter calibration ABANDONED: the seed loss ("
      << std::scientific << std::setprecision(3) << f0
      << ") gives a relative tolerance nothing to be relative to, so every "
         "dimension would pass at the cap. Falling back to the STEP_FRAC "
         "jitter; fix the seed or use `INIT_MODE JITTER <fraction>`.";
    log.warning(w.str());
    return plan;
  }

  if (rank != 0) return plan;   /* summary + TSV belong to rank 0 */
  std::vector<double> frac(dim);
  std::size_t st_ok = 0, st_cap = 0, st_floor = 0, st_one = 0, st_asym = 0,
              st_cliff = 0, st_cut = 0, st_inf = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    const double w = sp.high[i] - sp.low[i];
    frac[i] = w > 0.0 ? plan.sigma[i] / w : 0.0;
    const unsigned st = plan.dims[i].status;
    if (st == JitterOk) ++st_ok;
    if (st & JitterSupportCapped)  ++st_cap;
    if (st & JitterNumericalFloor) ++st_floor;
    if (st & JitterOneSided)       ++st_one;
    if (st & JitterAsymmetric)     ++st_asym;
    if (st & JitterCliff)          ++st_cliff;
    if (st & JitterInterrupted)    ++st_cut;
    if (st & JitterInferred)       ++st_inf;
  }
  {
    std::ostringstream o;
    o << tag << " init-jitter summary: dims=" << dim << "  probes=" << spent
      << "  wall=" << std::fixed << std::setprecision(1) << elapsed() << " s"
      << "  ok=" << st_ok << "  support-capped=" << st_cap
      << "  numerical-floor=" << st_floor << "  one-sided=" << st_one
      << "  asymmetric=" << st_asym << "  cliff=" << st_cliff
      << "  interrupted=" << st_cut << "  inferred=" << st_inf;
    log.info(o.str());
  }
  {
    /* Distribution, not 816 lines: the per-dim table goes to the TSV. */
    std::ostringstream o;
    o << tag << " init-jitter sigma/width: min=" << std::scientific
      << std::setprecision(3) << jitter_pct(frac, 0.0)
      << "  p25=" << jitter_pct(frac, 0.25)
      << "  median=" << jitter_pct(frac, 0.5)
      << "  p75=" << jitter_pct(frac, 0.75)
      << "  max=" << jitter_pct(frac, 1.0);
    log.info(o.str());
  }
  /* The audit's verdict, always with its sample size: a claim about 32 of
   * 816 dimensions is not a claim about all of them, and stating it without
   * the denominator is how a spot check turns into a guarantee. */
  if (plan.audited > 0) {
    std::ostringstream o;
    o << tag << " init-jitter interior audit: " << plan.non_monotone
      << " of " << plan.audited << " re-probed dimension(s) were OUT of band "
      << "at sigma/2";
    if (plan.non_monotone == 0) {
      o << " -- sigma behaved as a bound on the sampled axes";
      log.info(o.str());
    } else {
      o << ". The loss is NOT monotone in displacement along those axes, so "
           "sigma bounds only the points that were tested, not the interval "
           "below it. Treat the jitter as a grid certificate there.";
      log.hint(o.str());
    }
  }
  if (st_inf > 0) {
    std::ostringstream h;
    h << tag << " init-jitter: " << st_inf << " of " << dim
      << " dimension(s) were NOT measured (INIT_JITTER_SAMPLE=" << opt.sample
      << "); each took its family's median sigma/width and is flagged "
         "`inferred` in hpo_init_jitter.tsv. That assumes members of a family "
         "share a sensitivity scale -- raise INIT_JITTER_SAMPLE if that is "
         "doubtful for this basis.";
    log.hint(h.str());
  }
  if (st_cut > 0) {
    std::ostringstream w;
    w << tag << " init-jitter calibration was cut short: " << st_cut
      << " dimension(s) never probed (cap " << cap
      << " or the MAXTIME share). They keep the STEP_FRAC sigma; the rest "
         "are measured. Raise INIT_JITTER_MAXEVAL to finish the job.";
    log.warning(w.str());
  }

  /* ---- per-dimension table -------------------------------------------- */
  {
    std::ofstream f("hpo_init_jitter.tsv", std::ios::trunc);
    if (!f) {
      log.warning(std::string(banner_ctx) +
                  ": cannot open hpo_init_jitter.tsv for writing");
    } else {
      f << "# INIT_MODE JITTER AUTO calibration. sigma is the largest\n"
           "# displacement MEASURED inside the band, never interpolated.\n"
           "#\n"
           "# Lengths are INTERNAL optimiser coordinates, the space the\n"
           "# jitter is actually drawn in -- under OPTIM_SCALE AUTO (the\n"
           "# default) every dimension spans [0,1], so sigma and sigma/width\n"
           "# coincide there. Use hpo_search_space.tsv to map back to real\n"
           "# units.\n"
           "#\n"
           "#   dim        1-based position in the parameter vector\n"
           "#   sigma      half-width of the uniform initial jitter\n"
           "#   upper      smallest displacement measured OUT of band\n"
           "#              (inf = never left the band, i.e. support-capped)\n"
           "#   probes     objective evaluations spent on this dimension\n";
      f << "dim\tparameter\tsigma\tsigma/width\tupper\tprobes\tstatus\n";
      f << std::scientific << std::setprecision(10);
      for (std::size_t i = 0; i < dim; ++i) {
        const std::string label = sp.parameter_labels.size() == dim
            ? sp.parameter_labels[i] : logidx::dim_label(i);
        f << logidx::uis(i) << '\t' << label << '\t' << plan.sigma[i] << '\t'
          << frac[i] << '\t' << plan.dims[i].upper << '\t'
          << plan.dims[i].probes << '\t'
          << jitter_status_text(plan.dims[i].status) << '\n';
      }
      log.info(std::string(banner_ctx) +
               " init-jitter per-dimension table: hpo_init_jitter.tsv");
    }
  }

  /* ---- reconcile the two eval counters --------------------------------
   * HPOLog counts every objective evaluation, probes included, while
   * MAXEVAL governs only the search. So the run's "Total evaluations" is
   * legitimately LARGER than MAXEVAL after a calibration -- which, unsaid,
   * is a very reasonable bug report. Say it. */
  {
    std::ostringstream o;
    o << tag << " init-jitter eval accounting: " << spent
      << " probe(s) spent, NOT charged against MAXEVAL";
    if (sp.stop.maxeval)
      o << " (" << *sp.stop.maxeval << ") -- the run's reported total is the "
           "search plus these " << spent << ", so it may exceed MAXEVAL by "
           "up to that much";
    log.info(o.str());
  }
  return plan;
}

}  // namespace anneal
}  // namespace hpo
}  // namespace tadah

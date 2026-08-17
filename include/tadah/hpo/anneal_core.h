#pragma once
/* anneal_core.h -- shared building blocks for the TADAH simulated-
 * annealing family.
 *
 * Used by the serial driver (TADAH:ANNEAL, algorithms.h) and by the
 * parallel drivers (TADAH:ANNEAL_PT / TADAH:ANNEAL_PA). Everything
 * here is host-agnostic: no HPO_Host dependency, so unit tests can
 * exercise each piece directly.
 *
 * Contents:
 *   - AlgoKind / algo_name       family-member selector
 *   - make_stream                decorrelated RNG streams (seed_seq salts)
 *   - reflect_into               exact mirror reflection into [lo, hi]
 *   - constrain_proposal         reflection / redraw boundary policy
 *   - loss_is_finite             FAILSCORE-aware feasibility test
 *   - loss_is_rankable           annealer ordering test (graded failures)
 *   - robust_t0                  cliff-robust initial-temperature estimate
 *   - Knobs / Knobs::parse       PARAM parsing, validation, whitelists
 *   - CoranaPlateau              Corana et al. 1987 sec. 4 eps-NEPS stop
 */
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pc_fail_grading.h>   /* kFailBandFloor, loss_is_feasible */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tadah::hpo {
namespace anneal {

enum class AlgoKind { Serial, PT, PA };

enum class BoundaryMode : int { Reflect = 0, Redraw = 1 };

inline const char* algo_name(AlgoKind k)
{
  switch (k) {
    case AlgoKind::PT: return "ANNEAL_PT";
    case AlgoKind::PA: return "ANNEAL_PA";
    default:           return "ANNEAL";
  }
}

/* ---- RNG stream discipline ----------------------------------------
 *
 * Every consumer of randomness gets its own mt19937_64 seeded from
 * (SEED, salt, index) through a seed_seq. Streams are therefore
 * decorrelated and independent: enabling/disabling the T0 probes no
 * longer shifts the proposal sequence, and parallel chains draw
 * reproducibly regardless of thread scheduling. */
constexpr std::uint64_t SALT_CHAIN       = 0xC4A1C;  /* chain i proposals+coins */
constexpr std::uint64_t SALT_PROBES      = 0x70B35;  /* T0 calibration probes   */
constexpr std::uint64_t SALT_PT_SWAP     = 0x5757;   /* PT swap coins (epoch)   */
constexpr std::uint64_t SALT_PA_RESAMPLE = 0x9E5A;   /* PA resampler (tstep)    */

inline std::mt19937_64 make_stream(std::uint64_t seed,
                                   std::uint64_t salt,
                                   std::uint64_t idx = 0)
{
  std::seed_seq seq{
      static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
      static_cast<std::uint32_t>(salt), static_cast<std::uint32_t>(salt >> 32),
      static_cast<std::uint32_t>(idx),  static_cast<std::uint32_t>(idx >> 32)};
  return std::mt19937_64(seq);
}

/* Mirror-reflect v into [lo, hi] exactly, however many box widths the
 * finite proposal crossed.  The old bounded loop folded at most 16
 * times and then clamped to an endpoint, creating artificial boundary
 * atoms for large STEP overrides.  Long-double modulo arithmetic keeps
 * the operation defined even when subtracting opposite-sign doubles
 * would overflow in double precision. */
inline bool reflect_into(long double& v, long double lo, long double hi)
{
  if (!std::isfinite(v) || !std::isfinite(lo) || !std::isfinite(hi) ||
      !(hi > lo))
    throw std::invalid_argument(
        "anneal::reflect_into requires finite v and finite lo < hi");
  if (v >= lo && v <= hi) return false;

  const long double llo = lo;
  const long double lhi = hi;
  const long double w   = lhi - llo;
  const long double p   = 2.0L * w;
  long double y = std::fmod(static_cast<long double>(v) - llo, p);
  if (y < 0.0L) y += p;
  const long double folded = y <= w ? llo + y : lhi - (y - w);
  v = folded;
  /* Round-off at an exactly representable endpoint must never leak a
   * proposal outside the public invariant. */
  v = std::min(hi, std::max(lo, v));
  return true;
}

inline bool reflect_into(double& v, double lo, double hi)
{
  long double wide = static_cast<long double>(v);
  const bool changed = reflect_into(
      wide, static_cast<long double>(lo), static_cast<long double>(hi));
  v = static_cast<double>(wide);
  v = std::min(hi, std::max(lo, v));
  return changed;
}

/* Apply the configured boundary kernel to an already-drawn coordinate.
 * `redraw` must produce a fresh draw from the same unconstrained proposal
 * distribution.  Returns true iff the first draw was outside; later
 * discarded draws do not consume host-evaluation budget and count as one
 * OOB proposal, keeping the diagnostic comparable with reflection. */
template<class Real, class Draw>
inline bool constrain_proposal(Real& v, double lo, double hi,
                               BoundaryMode mode, Draw&& redraw)
{
  if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo))
    throw std::invalid_argument(
        "anneal::constrain_proposal requires finite lo < hi");
  if (mode != BoundaryMode::Reflect && mode != BoundaryMode::Redraw)
    throw std::invalid_argument("anneal::constrain_proposal invalid boundary mode");
  if (std::isfinite(v) && v >= static_cast<Real>(lo) &&
      v <= static_cast<Real>(hi)) return false;
  if (mode == BoundaryMode::Reflect) {
    reflect_into(v, static_cast<Real>(lo), static_cast<Real>(hi));
    return true;
  }
  /* A malformed/out-of-box centre used to make this an unbounded loop.
   * Valid annealer centres have substantial in-box mass, so 4096 misses is
   * incontrovertibly a broken proposal kernel rather than bad luck. */
  for (std::size_t attempt = 0; attempt < 4096; ++attempt) {
    v = redraw();
    if (std::isfinite(v) && v >= static_cast<Real>(lo) &&
        v <= static_cast<Real>(hi)) return true;
  }
  throw std::runtime_error(
      "anneal redraw kernel failed to produce an in-bounds finite coordinate");
}

/* FAILSCORE-aware finiteness: a loss is usable for statistics when it
 * is a normal number comfortably below the configured FAILSCORE. The
 * reserved-band margin absorbs failscore values that went through lossy
 * text round-trips. When no failscore is known every finite double counts.
 * The band boundary itself is defined once, in pc_fail_grading.h, next to
 * the grading arithmetic that produces the values being excluded. */
inline bool loss_is_finite(double f, const std::optional<double>& failscore)
{
  if (!failscore) return std::isfinite(f);
  return loss_is_feasible(f, *failscore);
}

/* A loss may still carry useful ordering information without being feasible.
 * FAILSCORE_GRADING reserves [0.999*FS, FS) for directional failure losses:
 * T0 calibration and PA resampling must continue to filter that band through
 * loss_is_finite(), while Metropolis moves and chain personal bests must be
 * able to descend through it.  The exact legacy FAILSCORE remains unrankable. */
inline bool loss_is_rankable(double f,
                             const std::optional<double>& failscore)
{
  if (!std::isfinite(f)) return false;
  return !failscore || f < *failscore;
}

/* STOPVAL is a success criterion, so neither non-finite nor graded/legacy
 * failure losses may satisfy it even when a user supplied a high threshold. */
inline bool stopval_reached(double f,
                            const std::optional<double>& stopval,
                            const std::optional<double>& failscore)
{
  return stopval && loss_is_finite(f, failscore) && f <= *stopval;
}

/* Cliff-robust initial-temperature estimate from probes around a
 * FINITE anchor loss f0 (Metropolis target-acceptance solve on the
 * median uphill move):
 *
 *     U  = { f_j - f0 : f_j finite, f_j > f0 }
 *     T0 = median(U) / ln(1 / paccept)
 *
 * The median tolerates <50% FAILSCORE contamination even when the
 * failscore value is unknown. If no uphill move exists but finite
 * probes do (pure-descent neighbourhood), fall back to median|f_j-f0|.
 *
 * Returns: nullopt when NO finite probe exists (cliff-saturated
 * neighbourhood -- caller decides); 0.0 when finite probes are all
 * identical to f0 (constant landscape -- caller picks a fallback). */
inline std::optional<double> robust_t0(double f0,
                                       const std::vector<double>& probes,
                                       const std::optional<double>& failscore,
                                       double paccept)
{
  if (!std::isfinite(f0) || !std::isfinite(paccept) ||
      !(paccept > 0.0 && paccept < 1.0))
    throw std::invalid_argument(
        "anneal::robust_t0 requires finite f0 and paccept in (0,1)");
  std::vector<long double> up, absd;
  up.reserve(probes.size());
  absd.reserve(probes.size());
  for (double f : probes) {
    if (!loss_is_finite(f, failscore)) continue;
    long double d = static_cast<long double>(f) -
                    static_cast<long double>(f0);
    if (!std::isfinite(d))
      d = std::copysign(
          static_cast<long double>(std::numeric_limits<double>::max()), d);
    if (d > 0.0L) up.push_back(d);
    absd.push_back(std::fabs(d));
  }
  if (absd.empty()) return std::nullopt;
  std::vector<long double>& src = up.empty() ? absd : up;
  std::sort(src.begin(), src.end());
  const std::size_t n = src.size();
  const long double med = (n % 2)
      ? src[n / 2]
      : 0.5L * src[n / 2 - 1] + 0.5L * src[n / 2];
  const long double estimate =
      med / -std::log(static_cast<long double>(paccept));
  if (estimate >= static_cast<long double>(
                      std::numeric_limits<double>::max()))
    return std::numeric_limits<double>::max();
  return static_cast<double>(estimate);
}

/* ---- Knobs ---------------------------------------------------------
 *
 * One vocabulary across the whole family; a per-AlgoKind whitelist
 * decides which keys are accepted. Renamed keys and knobs that are
 * dead for the selected algorithm are hard errors (never silently
 * ignored) so typos and stale configs surface immediately. */
struct Knobs {
  /* stepper (shared across the family)
   *
   * A fixed-MAXEVAL grid on one 42-D Kr EAM workload retained the prior
   * NS=5 default and changed NT 2->5, RT 0.85->0.90, and T0_PACCEPT
   * 0.80->0.05.  On matched seeds 201-203, the measured median best losses
   * were 3.35e-3 for NS5/NT5/P0.05, 1.41 for NS5/NT2/P0.05, and 5.45
   * for NS5/NT2/P0.80.  These are workload-specific fixed-budget
   * measurements, not a general ranking of settings. */
  bool        corana        = false;
  std::size_t L             = 50;     /* classic chain length per rung   */
  std::size_t NS            = 5;      /* Corana step cycles              */
  std::size_t NT            = 5;      /* Corana step-adjust cycles per T */
  double      C             = 2.0;    /* Corana step-adjust gain         */
  double      step_frac     = 0.10;
  double      v_floor_frac  = 1e-3;
  BoundaryMode boundary_mode = BoundaryMode::Reflect;
  bool        reset_to_best = false;  /* default flips to true w/ CORANA */
  std::size_t neps          = 4;

  /* temperature family */
  std::optional<double> t_hot;        /* explicit T_HOT skips calibration */
  std::optional<double> t_min;        /* explicit floor; auto = T0 * 1e-6 */
  double      rt            = 0.90;
  std::size_t t0_probes     = 20;
  double      t0_paccept    = 0.05;

  /* serial warm re-anneal on T_MIN (run15 lever) */
  bool        reanneal       = true;
  double      reanneal_scale = 0.25;

  /* parallel tempering (ANNEAL_PT only) */
  std::size_t replicas    = 0;      /* 0 = resolve to max(THREADS, 2)   */
  std::optional<double> t_cold;     /* explicit coldest slot            */
  double      ladder_span = 1e3;    /* T_COLD = T_HOT / LADDER_SPAN     */
  double      ladder_power = 1.0;   /* initial log-ladder shape         */
  bool        ladder_adapt = false; /* adapt interior temperatures      */
  std::size_t ladder_adapt_every = 20; /* attempts/pair between updates */
  double      ladder_adapt_gain = 1.0; /* initial stochastic gain       */
  double      ladder_adapt_decay = 0.6;/* diminishing-gain exponent     */
  std::size_t ladder_adapt_updates = 0;/* 0 = no update-count cap (the
                                     * equalization freeze may still stop
                                     * adaptation; see ladder_freeze)   */
  double      ladder_min_gap = 0.1; /* min interior log-gap as a fraction
                                     * of the mean gap; 0 = no floor    */
  bool        ladder_freeze = true; /* pooled-evidence equalization
                                     * freeze; 0 = adapt forever        */
  std::size_t swap_every  = 1;      /* exchange attempt every N epochs  */
  std::size_t swap_window = 20;     /* windowed swap-acc table every N
                                     * epochs (0 = cumulative only)     */
  bool        step_auto = true;     /* PT CLASSIC coordinate warm-up    */
  std::size_t step_auto_sweeps = 12;/* minimum attempts per coordinate  */
  std::size_t step_adapt_every = 0; /* PT diagonal RAM apply cadence     */
  double      step_trust_factor = 64.0; /* RAM sigma bound as a factor of
                                     * the calibrated baseline; 0 = off */

  /* shared by both parallel drivers */
  int         init_mode   = 0;      /* 0 = seed + Gaussian jitter, 1 = box */

  /* population annealing (ANNEAL_PA only) */
  std::size_t sweeps      = 1;      /* rungs per walker per T-step      */
  double      ess_target  = 0.5;    /* adaptive dbeta target; 0 = RT    */

  static Knobs parse(const OptimizerSpec& sp, AlgoKind kind);
};

inline Knobs Knobs::parse(const OptimizerSpec& sp, AlgoKind kind)
{
  const std::string ctx = std::string("TADAH:") + algo_name(kind);

  /* -- renamed keys: hard error naming the replacement -------------- */
  static const std::unordered_map<std::string, std::string> renamed = {
      {"T_INIT", "T_HOT"},
  };
  for (const auto& [old_key, new_key] : renamed) {
    if (sp.param.count(old_key))
      throw std::runtime_error(
          ctx + ": PARAM " + old_key + " was renamed to " + new_key +
          " (same meaning: the starting/hottest temperature); update the config");
  }

  /* -- per-kind whitelist: unknown keys are typos or knobs for a
   *    different family member -------------------------------------- */
  static const std::unordered_set<std::string> shared_keys = {
      "CORANA", "NS", "NT", "C", "L", "STEP_FRAC", "V_FLOOR_FRAC",
      "BOUNDARY_MODE", "RESET_TO_BEST", "NEPS", "T0_PROBES", "T0_PACCEPT",
      "T_HOT", "T_MIN", "RT",
  };
  static const std::unordered_set<std::string> serial_keys = {
      "REANNEAL", "REANNEAL_SCALE",
  };
  static const std::unordered_set<std::string> pt_keys = {
      "REPLICAS", "T_COLD", "LADDER_SPAN", "SWAP_EVERY", "SWAP_WINDOW",
      "LADDER_POWER", "LADDER_ADAPT", "LADDER_ADAPT_EVERY",
      "LADDER_ADAPT_GAIN", "LADDER_ADAPT_DECAY", "LADDER_ADAPT_UPDATES",
      "LADDER_MIN_GAP", "LADDER_FREEZE",
      "STEP_AUTO", "STEP_AUTO_SWEEPS", "STEP_ADAPT_EVERY",
      "STEP_TRUST_FACTOR",
  };
  static const std::unordered_set<std::string> pa_keys = {
      "SWEEPS", "ESS_TARGET",
  };
  static const std::unordered_set<std::string> par_keys = {
      "INIT_MODE",   /* shared by both parallel drivers */
  };
  auto allowed = [&](const std::string& k) {
    if (shared_keys.count(k)) return true;
    if (kind == AlgoKind::Serial && serial_keys.count(k)) return true;
    if (kind == AlgoKind::PT && pt_keys.count(k)) return true;
    if (kind == AlgoKind::PA && pa_keys.count(k)) return true;
    if (kind != AlgoKind::Serial && par_keys.count(k)) return true;
    return false;
  };
  for (const auto& kv : sp.param) {
    if (!allowed(kv.first)) {
      /* Steer knobs of a sibling family member explicitly. */
      if (kind != AlgoKind::PT && pt_keys.count(kv.first))
        throw std::runtime_error(
            ctx + ": PARAM " + kv.first + " belongs to TADAH:ANNEAL_PT");
      if (kind != AlgoKind::PA && pa_keys.count(kv.first))
        throw std::runtime_error(
            ctx + ": PARAM " + kv.first + " belongs to TADAH:ANNEAL_PA");
      if (kind != AlgoKind::Serial && serial_keys.count(kv.first))
        throw std::runtime_error(
            ctx + ": PARAM " + kv.first + " belongs to the serial TADAH:ANNEAL");
      std::string keys;
      for (const auto& k : shared_keys) { keys += k; keys += " "; }
      if (kind == AlgoKind::Serial)
        for (const auto& k : serial_keys) { keys += k; keys += " "; }
      if (kind == AlgoKind::PT)
        for (const auto& k : pt_keys) { keys += k; keys += " "; }
      if (kind == AlgoKind::PA)
        for (const auto& k : pa_keys) { keys += k; keys += " "; }
      if (kind != AlgoKind::Serial)
        for (const auto& k : par_keys) { keys += k; keys += " "; }
      throw std::runtime_error(
          ctx + ": unknown PARAM '" + kv.first + "'; accepted keys: " + keys);
    }
  }

  /* -- spec-level knobs that are dead for this family --------------- */
  if (sp.stop.xtol_rel || sp.stop.xtol_abs)
    throw std::runtime_error(
        ctx + " does not implement XTOL_REL/XTOL_ABS (no x-space "
        "convergence test); remove them or use FTOL_REL/FTOL_ABS "
        "(eps-NEPS plateau stop)");
  if (sp.vector_storage)
    throw std::runtime_error(ctx + " does not use VECTOR_ARR; remove it");
  if (sp.grad_strategy)
    throw std::runtime_error(
        ctx + " is derivative-free and does not use GRAD_STRATEGY; remove it");
  /* THREADS is honoured by EVERY family under the shared-pool model
   * (Option A): it sizes the per-rank worker pool that runs the pc_lammps
   * fan-out + MLIP train/predict. Serial ANNEAL still drives ONE chain, but
   * its within-eval fan-out runs on that pool, so THREADS is meaningful here
   * too -- only THREADS 0 is rejected. PT/PA additionally use THREADS to cap
   * the number of concurrent replicas (see algorithms_p{t,a}.h). */
  if (sp.num_threads && *sp.num_threads == 0)
    throw std::runtime_error(ctx + ": THREADS must be >= 1");
  if (kind != AlgoKind::PA && sp.pop)
    throw std::runtime_error(
        ctx + " does not use POPULATION; use TADAH:ANNEAL_PA");

  auto require_finite_stop = [&](const char* name,
                                 const std::optional<double>& value,
                                 bool allow_negative) {
    if (!value) return;
    if (!std::isfinite(*value) || (!allow_negative && *value < 0.0))
      throw std::runtime_error(
          ctx + ": " + name + " must be finite" +
          std::string(allow_negative ? "" : " and non-negative"));
  };
  require_finite_stop("STOPVAL", sp.stop.stopval, true);
  require_finite_stop("FTOL_REL", sp.stop.ftol_rel, false);
  require_finite_stop("FTOL_ABS", sp.stop.ftol_abs, false);
  require_finite_stop("MAXTIME", sp.stop.maxtime, false);
  if (sp.stop.maxtime && !(*sp.stop.maxtime > 0.0))
    throw std::runtime_error(ctx + ": MAXTIME must be > 0");
  if (sp.failscore &&
      (!std::isfinite(*sp.failscore) || !(*sp.failscore > 0.0)))
    throw std::runtime_error(ctx + ": FAILSCORE must be finite and > 0");

  auto pget = [&](const char* k, double dflt) {
    auto it = sp.param.find(k);
    if (it == sp.param.end()) return dflt;
    if (!std::isfinite(it->second))
      throw std::runtime_error(ctx + ": PARAM " + k + " must be finite");
    return it->second;
  };
  auto pcount = [&](const char* k, double dflt) -> std::size_t {
    const double v = pget(k, dflt);
    const long double size_limit = std::ldexp(
        1.0L, std::numeric_limits<std::size_t>::digits);
    if (v < 0.0 || std::floor(v) != v ||
        static_cast<long double>(v) >= size_limit)
      throw std::runtime_error(
          ctx + ": PARAM " + k + " must be a non-negative integer");
    return static_cast<std::size_t>(v);
  };
  auto pbool = [&](const char* k, bool dflt) -> bool {
    const double v = pget(k, dflt ? 1.0 : 0.0);
    if (v != 0.0 && v != 1.0)
      throw std::runtime_error(ctx + ": PARAM " + k + " must be 0 or 1");
    return v == 1.0;
  };

  Knobs kb;
  kb.corana        = pbool("CORANA", false);

  /* Variant-level no-ops are errors.  They used to be admitted by the
   * family whitelist and then silently ignored by the selected kernel. */
  if (kb.corana && sp.param.count("L"))
    throw std::runtime_error(ctx + ": PARAM L belongs to the CLASSIC variant");
  if (!kb.corana) {
    for (const char* key : {"NS", "NT", "C", "V_FLOOR_FRAC"})
      if (sp.param.count(key))
        throw std::runtime_error(
            ctx + ": PARAM " + key + " requires PARAM CORANA 1");
  }
  if (kind == AlgoKind::PA && sp.param.count("RESET_TO_BEST"))
    throw std::runtime_error(
        ctx + ": PARAM RESET_TO_BEST is not defined for population annealing");

  kb.L             = pcount("L", 50.0);
  kb.NS            = pcount("NS", 5.0);
  kb.NT            = pcount("NT", 5.0);
  kb.C             = pget("C", 2.0);
  kb.step_frac     = pget("STEP_FRAC", 0.10);
  kb.v_floor_frac  = pget("V_FLOOR_FRAC", 1e-3);
  const std::size_t boundary = pcount("BOUNDARY_MODE", 0.0);
  if (boundary > 1)
    throw std::runtime_error(
        ctx + ": PARAM BOUNDARY_MODE must be 0 (REFLECT) or 1 (REDRAW)");
  kb.boundary_mode = static_cast<BoundaryMode>(boundary);
  /* D23: REDRAW is not a symmetric proposal, and the acceptance rule does
   * not correct for that.
   *
   * constrain_proposal's redraw branch resamples only the offending
   * coordinate until it lands in range, which is a truncated Gaussian whose
   * normalisation depends on how much of the kernel's mass sat inside the
   * box -- i.e. on the CURRENT x. metropolis_accept applies no
   * Metropolis-Hastings ratio q(x|x')/q(x'|x), so detailed balance does not
   * hold near a bound and the stationary distribution is skewed away from
   * the box edges. REFLECT (the default) has no such problem: it is an
   * exact period-2w fold, hence an involution, hence symmetric.
   *
   * Announced rather than corrected, deliberately. The correct ratio for a
   * SEQUENTIAL per-coordinate redraw depends on which coordinates were
   * resampled and on each one's in-box mass at both x and x'; a plausible
   * but wrong ratio would be worse than a documented, bounded bias. For a
   * minimiser the practical cost is modest -- the deliverable is the best
   * point visited, not the stationary measure -- but a user sampling near
   * bounds must know. HINT, not WARNING: the magnitude depends on the
   * objective and the geometry, so it is a caution, not a proof. */
  if (kb.boundary_mode == BoundaryMode::Redraw)
    HPOLog::instance().hint(
        ctx + ": BOUNDARY_MODE 1 (REDRAW) resamples out-of-box coordinates "
        "from a truncated kernel whose normalisation depends on the current "
        "point, and the Metropolis rule applies no Hastings correction for "
        "it. Sampling near an OPTIM bound is therefore biased away from the "
        "edge. This does not affect which best point is reported, but if you "
        "care about the distribution rather than the optimum, use "
        "BOUNDARY_MODE 0 (REFLECT), which is an exact symmetric fold.");
  /* Corana's published cooling step returns to the best-so-far point.
   * Keep that default for serial Corana; PT defaults off because snapping
   * hot replicas defeats exchange, while PA rejects the knob above. */
  kb.reset_to_best = pbool("RESET_TO_BEST",
                           kb.corana && kind == AlgoKind::Serial);
  kb.neps          = pcount("NEPS", 4.0);

  if (sp.param.count("T_HOT")) kb.t_hot = pget("T_HOT", 0.0);
  if (sp.param.count("T_MIN")) kb.t_min = pget("T_MIN", 0.0);
  /* RT: cooling ratio everywhere. PT default 1.0 = static ladder (pure
   * tempering); RT < 1 turns PT into a cooling-ladder hybrid. */
  kb.rt            = pget("RT", kind == AlgoKind::PT ? 1.0 : 0.90);
  kb.t0_probes     = pcount("T0_PROBES", 20.0);
  kb.t0_paccept    = pget("T0_PACCEPT", 0.05);

  kb.reanneal       = kind == AlgoKind::Serial && pbool("REANNEAL", true);
  kb.reanneal_scale = pget("REANNEAL_SCALE", 0.25);

  kb.replicas    = pcount("REPLICAS", 0.0);
  if (sp.param.count("T_COLD")) kb.t_cold = pget("T_COLD", 0.0);
  kb.ladder_span = pget("LADDER_SPAN", 1e3);
  kb.ladder_power = pget("LADDER_POWER", 1.0);
  kb.ladder_adapt = pbool("LADDER_ADAPT", false);
  kb.ladder_adapt_every = pcount("LADDER_ADAPT_EVERY", 20.0);
  kb.ladder_adapt_gain = pget("LADDER_ADAPT_GAIN", 1.0);
  kb.ladder_adapt_decay = pget("LADDER_ADAPT_DECAY", 0.6);
  kb.ladder_adapt_updates = pcount("LADDER_ADAPT_UPDATES", 0.0);
  kb.ladder_min_gap = pget("LADDER_MIN_GAP", 0.1);
  kb.ladder_freeze = pbool("LADDER_FREEZE", true);
  kb.swap_every  = pcount("SWAP_EVERY", 1.0);
  kb.swap_window = pcount("SWAP_WINDOW", 20.0);
  kb.step_auto = kind == AlgoKind::PT && pbool("STEP_AUTO", true);
  kb.step_auto_sweeps = pcount("STEP_AUTO_SWEEPS", 12.0);
  kb.step_adapt_every = pcount("STEP_ADAPT_EVERY", 0.0);
  kb.step_trust_factor = pget("STEP_TRUST_FACTOR", 64.0);
  const std::size_t init_mode = pcount("INIT_MODE", 0.0);
  if (kind != AlgoKind::Serial && init_mode > 1)
    throw std::runtime_error(
        ctx + ": PARAM INIT_MODE must be 0 (seed-jitter) or 1 (uniform box)");
  kb.init_mode   = static_cast<int>(init_mode);
  kb.sweeps      = pcount("SWEEPS", 1.0);
  kb.ess_target  = pget("ESS_TARGET", 0.5);

  /* Explicitly reject controls whose values would be ignored.  Silent
   * precedence is especially dangerous in long HPO runs because the banner
   * can show both knobs while only one affects the trajectory. */
  if (sp.step && sp.param.count("STEP_FRAC"))
    throw std::runtime_error(
        ctx + ": STEP and PARAM STEP_FRAC are mutually exclusive");
  if (sp.param.count("T_HOT")) {
    for (const char* key : {"T0_PROBES", "T0_PACCEPT"})
      if (sp.param.count(key))
        throw std::runtime_error(
            ctx + ": PARAM " + key + " is unused when T_HOT is set");
  }
  if (kind == AlgoKind::Serial && !kb.reanneal &&
      sp.param.count("REANNEAL_SCALE"))
    throw std::runtime_error(
        ctx + ": PARAM REANNEAL_SCALE is unused when REANNEAL is 0");
  if (kind == AlgoKind::PT && kb.t_cold &&
      sp.param.count("LADDER_SPAN"))
    throw std::runtime_error(
        ctx + ": PARAM LADDER_SPAN is unused when T_COLD is set");
  if (kind == AlgoKind::PT && kb.rt == 1.0 && kb.t_min)
    throw std::runtime_error(
        ctx + ": PARAM T_MIN is unused by a static PT ladder (RT=1)");
  if (kind == AlgoKind::PT && !kb.ladder_adapt) {
    for (const char* key : {"LADDER_ADAPT_EVERY", "LADDER_ADAPT_GAIN",
                            "LADDER_ADAPT_DECAY", "LADDER_ADAPT_UPDATES",
                            "LADDER_MIN_GAP", "LADDER_FREEZE"})
      if (sp.param.count(key))
        throw std::runtime_error(
            ctx + ": PARAM " + key +
            " is unused when LADDER_ADAPT is 0");
  }
  if (kind == AlgoKind::PT && kb.corana) {
    for (const char* key : {"STEP_AUTO", "STEP_AUTO_SWEEPS",
                            "STEP_ADAPT_EVERY", "STEP_TRUST_FACTOR"})
      if (sp.param.count(key))
        throw std::runtime_error(
            ctx + ": PARAM " + key + " belongs to the CLASSIC variant");
    kb.step_auto = false;
    kb.step_adapt_every = 0;
    kb.step_trust_factor = 0.0;
  }
  if (kind == AlgoKind::PT && sp.step) {
    if (sp.param.count("STEP_AUTO_SWEEPS") ||
        sp.param.count("STEP_ADAPT_EVERY") ||
        sp.param.count("STEP_TRUST_FACTOR"))
      throw std::runtime_error(
          ctx + ": STEP_AUTO_SWEEPS/STEP_ADAPT_EVERY/STEP_TRUST_FACTOR are "
                "unused when explicit STEP disables automatic step "
                "calibration");
    kb.step_auto = false;
    kb.step_adapt_every = 0;
    kb.step_trust_factor = 0.0;
  }
  if (kind == AlgoKind::PT && !kb.step_auto && !sp.step && !kb.corana) {
    if (sp.param.count("STEP_AUTO_SWEEPS") ||
        sp.param.count("STEP_ADAPT_EVERY") ||
        sp.param.count("STEP_TRUST_FACTOR"))
      throw std::runtime_error(
          ctx + ": STEP_AUTO_SWEEPS/STEP_ADAPT_EVERY/STEP_TRUST_FACTOR are "
                "unused when PARAM STEP_AUTO is 0");
    kb.step_adapt_every = 0;
    kb.step_trust_factor = 0.0;
  }
  /* The trust region binds inside the diagonal RAM apply; with the RAM
   * disabled the key would be a silently dead knob. */
  if (kind == AlgoKind::PT && sp.param.count("STEP_TRUST_FACTOR") &&
      kb.step_adapt_every == 0)
    throw std::runtime_error(
        ctx + ": PARAM STEP_TRUST_FACTOR is unused when STEP_ADAPT_EVERY "
              "is 0 (no adaptive diagonal to bound)");

  /* -- validation ---------------------------------------------------- */
  const bool rt_one_ok = (kind == AlgoKind::PT);
  if (!(kb.rt > 0.0 && (kb.rt < 1.0 || (rt_one_ok && kb.rt == 1.0))))
    throw std::runtime_error(ctx + ": PARAM RT must be in "
        + std::string(rt_one_ok ? "(0,1]" : "(0,1)") + " (got "
        + std::to_string(kb.rt) + ")");
  if (!kb.corana && kb.L == 0)
    throw std::runtime_error(ctx + ": PARAM L must be >= 1");
  if (kb.corana) {
    if (kb.NS == 0)
      throw std::runtime_error(ctx + ": PARAM NS must be >= 1 (Corana variant)");
    if (kb.NT == 0)
      throw std::runtime_error(ctx + ": PARAM NT must be >= 1 (Corana variant)");
    if (!(kb.C > 0.0))
      throw std::runtime_error(ctx + ": PARAM C must be > 0 (Corana step-adjust gain)");
  }
  if (!(kb.step_frac > 0.0 && kb.step_frac <= 1.0))
    throw std::runtime_error(ctx + ": PARAM STEP_FRAC must be in (0,1]");
  if (!(kb.v_floor_frac >= 0.0 && kb.v_floor_frac <= 1.0))
    throw std::runtime_error(ctx + ": PARAM V_FLOOR_FRAC must be in [0,1]");
  if (!(kb.t0_paccept > 0.0 && kb.t0_paccept < 1.0))
    throw std::runtime_error(ctx + ": PARAM T0_PACCEPT must be in (0,1) (got "
        + std::to_string(kb.t0_paccept) + ")");
  if (kb.t_hot && !(*kb.t_hot > 0.0))
    throw std::runtime_error(ctx + ": PARAM T_HOT must be > 0");
  if (kb.t_min && !(*kb.t_min > 0.0))
    throw std::runtime_error(ctx + ": PARAM T_MIN must be > 0");
  if (kb.t_hot && kb.t_min && *kb.t_min > *kb.t_hot)
    throw std::runtime_error(ctx + ": PARAM T_MIN must be <= T_HOT");
  if (!kb.t_hot && kb.t0_probes < 2)
    throw std::runtime_error(
        ctx + ": PARAM T0_PROBES must be >= 2 when T_HOT is not set "
        "(the calibration needs at least two probes)");
  if (!(kb.reanneal_scale > 0.0 && kb.reanneal_scale < 1.0))
    throw std::runtime_error(ctx + ": PARAM REANNEAL_SCALE must be in (0,1)");
  const bool plateau_armed =
      sp.stop.ftol_rel.has_value() || sp.stop.ftol_abs.has_value();
  if (plateau_armed && kb.neps == 0)
    throw std::runtime_error(ctx + ": PARAM NEPS must be >= 1");
  if (kind == AlgoKind::PT) {
    if (sp.param.count("REPLICAS") && kb.replicas < 2)
      throw std::runtime_error(ctx + ": PARAM REPLICAS must be >= 2");
    if (kb.t_cold) {
      if (!(*kb.t_cold > 0.0))
        throw std::runtime_error(ctx + ": PARAM T_COLD must be > 0");
      if (kb.t_hot && !(*kb.t_cold < *kb.t_hot))
        throw std::runtime_error(ctx + ": PARAM T_COLD must be < T_HOT");
    }
    if (!(kb.ladder_span > 1.0))
      throw std::runtime_error(ctx + ": PARAM LADDER_SPAN must be > 1");
    if (!(kb.ladder_power > 0.0))
      throw std::runtime_error(ctx + ": PARAM LADDER_POWER must be > 0");
    if (kb.ladder_adapt) {
      if (kb.ladder_adapt_every == 0)
        throw std::runtime_error(
            ctx + ": PARAM LADDER_ADAPT_EVERY must be >= 1");
      if (!(kb.ladder_adapt_gain > 0.0 && kb.ladder_adapt_gain <= 1.0))
        throw std::runtime_error(
            ctx + ": PARAM LADDER_ADAPT_GAIN must be in (0,1]");
      if (!(kb.ladder_adapt_decay > 0.5 && kb.ladder_adapt_decay <= 1.0))
        throw std::runtime_error(
            ctx + ": PARAM LADDER_ADAPT_DECAY must be in (0.5,1]");
      if (!std::isfinite(kb.ladder_min_gap) ||
          !(kb.ladder_min_gap >= 0.0 && kb.ladder_min_gap < 1.0))
        throw std::runtime_error(
            ctx + ": PARAM LADDER_MIN_GAP must be in [0,1) "
            "(0 disables the floor; 1 would leave no adaptable span)");
    }
    if (kb.swap_every == 0)
      throw std::runtime_error(ctx + ": PARAM SWAP_EVERY must be >= 1");
    if (kb.step_auto && kb.step_auto_sweeps == 0)
      throw std::runtime_error(
          ctx + ": PARAM STEP_AUTO_SWEEPS must be >= 1");
    if (!(kb.step_trust_factor == 0.0 ||
          (std::isfinite(kb.step_trust_factor) &&
           kb.step_trust_factor > 1.0)))
      throw std::runtime_error(
          ctx + ": PARAM STEP_TRUST_FACTOR must be 0 (off) or > 1");
    if (kb.rt < 1.0 && kb.t_min && !(*kb.t_min > 0.0))
      throw std::runtime_error(ctx + ": PARAM T_MIN must be > 0");
  }
  if (kind == AlgoKind::PA) {
    if (kb.sweeps == 0)
      throw std::runtime_error(ctx + ": PARAM SWEEPS must be >= 1");
    if (!(kb.ess_target >= 0.0 && kb.ess_target < 1.0))
      throw std::runtime_error(
          ctx + ": PARAM ESS_TARGET must be in [0,1) (0 = geometric RT schedule)");
    if (sp.pop && *sp.pop < 8)
      throw std::runtime_error(ctx + ": POPULATION must be >= 8");
  }

  if (sp.step && (!std::isfinite(*sp.step) || !(*sp.step > 0.0)))
    throw std::runtime_error(ctx + ": STEP must be finite and > 0");
  if (sp.low.size() != sp.high.size())
    throw std::runtime_error(ctx + ": mismatched lower/upper bound counts");
  if (!sp.log_scaled.empty() && sp.log_scaled.size() != sp.low.size())
    throw std::runtime_error(ctx + ": mismatched LOG_SCALE mask size");
  for (std::size_t i = 0; i < sp.low.size(); ++i) {
    const double lo = sp.low[i], hi = sp.high[i];
    if (!std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo))
      throw std::runtime_error(
          ctx + ": every optimisable dimension requires finite low < high");
    const double width = hi - lo;
    if (!std::isfinite(width))
      throw std::runtime_error(
          ctx + ": every optimisable dimension requires a finite bound width");
    if (kb.boundary_mode == BoundaryMode::Redraw && sp.step &&
        *sp.step > width)
      throw std::runtime_error(
          ctx + ": STEP must not exceed any dimension width under "
          "BOUNDARY_MODE 1 (REDRAW)");
  }

  return kb;
}

/* Resolve the first starting point into transformed space and enforce the
 * chain invariant that every coordinate begins inside its OPTIM interval.
 * Rejecting malformed CONFIG/WARM seeds is preferable to an unbounded redraw
 * loop or to reflecting only the one coordinate touched by a Corana move. */
inline std::vector<double> make_initial_point(const OptimizerSpec& sp,
                                              AlgoKind kind)
{
  const std::string ctx = std::string("TADAH:") + algo_name(kind);
  const std::size_t dim = sp.low.size();
  std::vector<double> x(dim);
  if (sp.optim_init.empty()) {
    for (std::size_t i = 0; i < dim; ++i)
      x[i] = sp.low[i] + 0.5 * (sp.high[i] - sp.low[i]);
  } else {
    const auto& raw = sp.optim_init.front();
    if (raw.size() != dim)
      throw std::runtime_error(
          ctx + ": starting point has " + std::to_string(raw.size()) +
          " coordinate(s), expected " + std::to_string(dim));
    for (std::size_t i = 0; i < dim; ++i) {
      double v = raw[i];
      if (!sp.log_scaled.empty() && sp.log_scaled[i]) {
        if (!(v > 0.0) || !std::isfinite(v))
          throw std::runtime_error(
              ctx + ": log-scaled starting coordinate must be finite and > 0");
        v = std::log(v);
      }
      x[i] = v;
    }
  }
  for (std::size_t i = 0; i < dim; ++i) {
    if (!std::isfinite(x[i]) || x[i] < sp.low[i] || x[i] > sp.high[i])
      throw std::runtime_error(
          ctx + ": starting coordinate " + std::to_string(i) +
          " lies outside its OPTIM bounds");
  }
  return x;
}

/* Build the per-dimension proposal scale once for every driver.  Classic
 * keeps an explicit STEP verbatim (large reflected Gaussian steps are a
 * supported exploration choice); Corana's adaptive state always starts
 * inside the same advertised [floor*width,width] interval it must obey
 * after adjustment. */
inline std::vector<double> make_initial_steps(const OptimizerSpec& sp,
                                              const Knobs& kb)
{
  std::vector<double> out(sp.low.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double width = sp.high[i] - sp.low[i];
    const double raw = sp.step ? *sp.step : kb.step_frac * width;
    if (!(raw > 0.0) || !std::isfinite(raw))
      throw std::runtime_error(
          "anneal: proposal scale underflowed or is non-finite in dimension " +
          std::to_string(i) + "; use an explicit positive STEP");
    out[i] = kb.corana
        ? std::min(width, std::max(kb.v_floor_frac * width, raw))
        : raw;
  }
  return out;
}

/* One coordinate of Corana's section-4 step adjustment.  Factored out so
 * the exact 0.4/0.6 table and both clamps have direct regression tests. */
inline double adjust_corana_step(double current, double width,
                                 std::size_t accepted,
                                 std::size_t proposed,
                                 double gain, double floor_frac)
{
  if (!(std::isfinite(current) && current >= 0.0) ||
      !(std::isfinite(width) && width > 0.0) || proposed == 0 ||
      accepted > proposed || !(std::isfinite(gain) && gain > 0.0) ||
      !(std::isfinite(floor_frac) && floor_frac >= 0.0 &&
        floor_frac <= 1.0))
    throw std::invalid_argument("invalid Corana step-adjustment inputs");
  const double rate = static_cast<double>(accepted) /
                      static_cast<double>(proposed);
  if (rate > 0.6)
    current *= 1.0 + gain * (rate - 0.6) / 0.4;
  else if (rate < 0.4)
    current /= 1.0 + gain * (0.4 - rate) / 0.4;
  return std::min(width, std::max(floor_frac * width, current));
}

struct StepSaturation {
  std::size_t floor = 0;
  std::size_t cap   = 0;
  std::size_t total = 0;
};

inline StepSaturation step_saturation(const std::vector<double>& V,
                                      const OptimizerSpec& sp,
                                      const Knobs& kb)
{
  StepSaturation s;
  s.total = V.size();
  for (std::size_t i = 0; i < V.size(); ++i) {
    const double width = sp.high[i] - sp.low[i];
    const double floor = kb.v_floor_frac * width;
    const double tol = 32.0 * std::numeric_limits<double>::epsilon() * width;
    if (std::fabs(V[i] - floor) <= tol) ++s.floor;
    if (std::fabs(V[i] - width) <= tol) ++s.cap;
  }
  return s;
}

/* ---- CoranaPlateau --------------------------------------------------
 *
 * Corana et al. 1987 sec. 4 termination, applied at the end of every
 * rung (temperature step): stop when the END-OF-RUNG CURRENT value
 * f_u differs from each of the previous NEPS end-of-rung values by at
 * most eps AND sits within eps of the running optimum,
 *
 *     eps = max(FTOL_ABS, FTOL_REL * |f_u|)      (NLOPT semantics,
 *                                                 either tolerance arms it)
 *
 * This is deliberately NOT a test on the monotone best-so-far: a chain
 * still exploring at high T has a fluctuating f_u and must not be
 * stopped just because the best stalled for NEPS rungs. Armed only
 * when FTOL_REL and/or FTOL_ABS is set on the strategy. */
class CoranaPlateau {
public:
  CoranaPlateau(const StopCriteria& stop, std::size_t neps,
                std::optional<double> failscore = std::nullopt)
      : ftol_rel_(stop.ftol_rel), ftol_abs_(stop.ftol_abs),
        failscore_(failscore),
        active_(stop.ftol_rel.has_value() || stop.ftol_abs.has_value()),
        neps_(neps)
  {
    if (active_ && neps_ == 0)
      throw std::invalid_argument(
          "anneal::CoranaPlateau requires NEPS >= 1 when FTOL is active");
    if (active_) ring_.reserve(neps_);
  }

  bool active() const { return active_; }

  /* Feed one end-of-rung current value + the running best; returns
   * true when the run should stop. */
  bool update(double f_rung_end, double f_best)
  {
    if (!active_) return false;
    if (!loss_is_finite(f_rung_end, failscore_) ||
        !loss_is_finite(f_best, failscore_)) {
      ring_.clear();
      next_ = 0;
      return false;
    }
    const double eps = std::max(
        ftol_abs_.value_or(0.0),
        ftol_rel_ ? *ftol_rel_ * std::fabs(f_rung_end) : 0.0);
    bool flat = ring_.size() == neps_;
    if (flat) {
      for (double fp : ring_) {
        if (std::fabs(f_rung_end - fp) > eps) { flat = false; break; }
      }
      if (flat && (f_rung_end - f_best) > eps) flat = false;
    }
    if (ring_.size() < neps_) {
      ring_.push_back(f_rung_end);
    } else {
      ring_[next_] = f_rung_end;
      next_ = (next_ + 1) % neps_;
    }
    return flat;
  }

private:
  std::optional<double> ftol_rel_, ftol_abs_, failscore_;
  bool        active_ = false;
  std::size_t neps_   = 4;
  std::vector<double> ring_;
  std::size_t next_   = 0;
};

/* ---- population-annealing math (pure, unit-testable) -----------------
 *
 * Reweighting a population from inverse temperature beta to
 * beta + dbeta multiplies each walker's weight by exp(-dbeta * f_i).
 * The effective sample size ESS = (sum w)^2 / sum w^2 measures how many
 * walkers still carry weight; the adaptive schedule picks the largest
 * dbeta keeping ESS/N at a target (Machta-style fixed-culling). */
inline double reweight_ess(const std::vector<double>& f, double dbeta)
{
  if (f.empty()) return 0.0;
  if (!std::isfinite(dbeta) || dbeta < 0.0)
    throw std::invalid_argument("anneal::reweight_ess requires finite dbeta >= 0");
  for (double fi : f)
    if (!std::isfinite(fi))
      throw std::invalid_argument("anneal::reweight_ess requires finite losses");
  if (dbeta == 0.0) return static_cast<double>(f.size());
  const double fmin = *std::min_element(f.begin(), f.end());
  long double s = 0.0L, s2 = 0.0L;
  for (double fi : f) {
    const long double delta = static_cast<long double>(fi) -
                              static_cast<long double>(fmin);
    const long double w = std::exp(-static_cast<long double>(dbeta) * delta);
    s += w;
    s2 += w * w;
  }
  return s2 > 0.0L ? static_cast<double>((s * s) / s2) : 0.0;
}

/* Largest dbeta in (0, cap] with ESS(dbeta)/N >= target (ESS is
 * monotone non-increasing in dbeta). Returns cap when even the cap
 * satisfies the target (cool at the maximum allowed rate). */
inline double choose_dbeta(const std::vector<double>& f, double target,
                           double dbeta_cap)
{
  const double n = static_cast<double>(f.size());
  if (!std::isfinite(target) || target < 0.0 || target > 1.0 ||
      !std::isfinite(dbeta_cap) || dbeta_cap < 0.0)
    throw std::invalid_argument("anneal::choose_dbeta invalid target or cap");
  if (f.empty() || dbeta_cap == 0.0) return 0.0;
  if (reweight_ess(f, dbeta_cap) / n >= target) return dbeta_cap;
  /* First find a factor-two bracket around the transition.  A linear search
   * over [0,cap] has only cap/2^N absolute resolution and returns zero for
   * perfectly representable roots near 1e-300. */
  double hi = dbeta_cap;
  double lo = 0.0;
  for (;;) {
    const double half = 0.5 * hi;
    if (half == 0.0 || half == hi) return 0.0;
    if (reweight_ess(f, half) / n >= target) {
      lo = half;
      break;
    }
    hi = half;
  }
  /* The bracket spans at most a factor of two, so 60 iterations now provide
   * relative rather than cap-relative resolution. */
  for (int it = 0; it < 60; ++it) {
    const double mid = lo + 0.5 * (hi - lo);
    if (mid == lo || mid == hi) break;
    if (reweight_ess(f, mid) / n >= target) lo = mid;
    else                                    hi = mid;
  }
  return lo;
}

/* PA needs a representable geometric endpoint, not cool_temperature's zero
 * sentinel (which is intended to terminate serial/PT schedules). */
inline double pa_geometric_temperature(double current, double rt,
                                       double floor)
{
  if (!std::isfinite(current) || !std::isfinite(rt) ||
      !std::isfinite(floor) || !(current > floor && floor > 0.0) ||
      !(rt > 0.0 && rt < 1.0))
    throw std::invalid_argument(
        "anneal::pa_geometric_temperature invalid schedule");
  double next = current * rt;
  if (!(next < current)) next = std::nextafter(current, 0.0);
  return std::max(floor, next);
}

struct PaTemperatureStep {
  double temperature = 0.0;
  long double dbeta = 0.0L;
};

/* Materialize an adaptive PA endpoint in double precision.  A positive
 * choose_dbeta result can round back to the current temperature; in that case
 * take one representable step without exceeding the geometric cooling cap.
 * The returned inverse-temperature increment is recomputed from the endpoint
 * and is the value the resampling weights must use. */
inline PaTemperatureStep realize_pa_temperature_step(
    double current, double geometric_floor, double proposed)
{
  if (!std::isfinite(current) || !std::isfinite(geometric_floor) ||
      !std::isfinite(proposed) || !(current > 0.0) ||
      !(geometric_floor > 0.0 && geometric_floor < current) ||
      proposed < 0.0)
    throw std::invalid_argument(
        "anneal::realize_pa_temperature_step invalid endpoint");
  double next = std::max(geometric_floor, std::min(current, proposed));
  if (!(next < current))
    next = std::max(geometric_floor, std::nextafter(current, 0.0));
  if (!(next > 0.0 && next < current))
    throw std::runtime_error(
        "anneal::realize_pa_temperature_step made no progress");
  const long double dbeta =
      1.0L / static_cast<long double>(next) -
      1.0L / static_cast<long double>(current);
  if (!(dbeta > 0.0L))
    throw std::runtime_error(
        "anneal::realize_pa_temperature_step has non-positive dbeta");
  return {next, dbeta};
}

/* Systematic resampling: N ordered positions (u + j)/N over the
 * normalised weight CDF -- the lowest-variance O(N) resampler, one
 * RNG draw (deterministic given u). `w` may contain zeros (FAILSCORE
 * walkers die); the caller guarantees sum(w) > 0. Returns the parent
 * index for each of the N output slots. */
inline std::vector<std::size_t> systematic_resample(
    const std::vector<double>& w, double u)
{
  const std::size_t n = w.size();
  std::vector<std::size_t> parent(n, 0);
  if (n == 0) return parent;
  if (!std::isfinite(u) || u < 0.0 || u >= 1.0)
    throw std::invalid_argument("anneal::systematic_resample requires u in [0,1)");
  long double wsum = 0.0L;
  std::size_t last_positive = n;
  for (std::size_t k = 0; k < n; ++k) {
    const double wi = w[k];
    if (!std::isfinite(wi) || wi < 0.0)
      throw std::invalid_argument(
          "anneal::systematic_resample requires finite non-negative weights");
    wsum += static_cast<long double>(wi);
    if (wi > 0.0) last_positive = k;
  }
  if (!(wsum > 0.0L) || !std::isfinite(wsum) || last_positive == n)
    throw std::invalid_argument(
        "anneal::systematic_resample requires a finite positive weight sum");
  std::size_t i = 0;         /* current parent */
  long double cdf = static_cast<long double>(w[0]);
  for (std::size_t j = 0; j < n; ++j) {
    const long double p =
        (static_cast<long double>(u) + static_cast<long double>(j)) /
        static_cast<long double>(n) * wsum;
    while (p >= cdf && i < last_positive) {
      ++i;
      cdf += static_cast<long double>(w[i]);
    }
    parent[j] = i;
  }
  return parent;
}

/* Planned rung count for one geometric schedule: rungs executed from
 * T_begin down to t_min at ratio rt, capped by the eval budget
 * (evals_left / rung_cost, rounded up so a partial final rung counts).
 * This is the "total" in the journal's rung=current/total display --
 * an upper estimate: plateau/STOPVAL stops end the run earlier. */
inline std::size_t planned_rungs(double T_begin, double t_min, double rt,
                                 std::uint64_t evals_left,
                                 std::uint64_t rung_cost)
{
  if (!std::isfinite(T_begin) || !std::isfinite(t_min) ||
      !std::isfinite(rt) || !(T_begin > 0.0) || !(t_min > 0.0) ||
      !(rt > 0.0 && rt <= 1.0))
    throw std::invalid_argument("anneal::planned_rungs invalid temperature input");
  std::size_t by_T = std::numeric_limits<std::size_t>::max();
  if (T_begin < t_min) return 1;   /* schedule would not run; display floor */
  if (rt < 1.0) {
    const long double k =
        (std::log(static_cast<long double>(t_min)) -
         std::log(static_cast<long double>(T_begin))) /
        std::log(static_cast<long double>(rt));
    const long double lim = static_cast<long double>(
        std::numeric_limits<std::size_t>::max() - 1);
    by_T = !std::isfinite(k) || k >= lim
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(std::floor(k)) + 1;
  }
  std::size_t by_budget = std::numeric_limits<std::size_t>::max();
  if (rung_cost > 0 &&
      evals_left != std::numeric_limits<std::uint64_t>::max()) {
    const std::uint64_t q = evals_left / rung_cost;
    const std::uint64_t r = evals_left % rung_cost;
    const std::uint64_t ceil = q + (r != 0 ? 1 : 0);
    by_budget = ceil > std::numeric_limits<std::size_t>::max()
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(ceil);
  }
  const std::size_t n = std::min(by_T, by_budget);
  return n == 0 ? 1 : n;
}

inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b)
{
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    return std::numeric_limits<std::uint64_t>::max();
  return a * b;
}

inline std::size_t saturating_add(std::size_t a, std::size_t b)
{
  const std::size_t max = std::numeric_limits<std::size_t>::max();
  return b > max - a ? max : a + b;
}

inline double automatic_t_min(double T0)
{
  if (!std::isfinite(T0) || !(T0 > 0.0))
    throw std::invalid_argument("anneal::automatic_t_min requires finite T0 > 0");
  const double scaled = T0 * 1e-6;
  return scaled > 0.0 ? scaled : std::numeric_limits<double>::denorm_min();
}

/* Return a strictly smaller representable temperature, or zero when
 * multiplication rounds back to the same value. Zero is below every valid
 * positive floor and therefore terminates the schedule deterministically. */
inline double cool_temperature(double T, double rt)
{
  const double next = T * rt;
  return next < T ? next : 0.0;
}

/* ---- shared budget / stop plumbing ----------------------------------
 *
 * One SharedBudget instance covers every host() consumer of a run
 * (seed eval, T0 probes, all chains): reserve-before-eval, so MAXEVAL
 * is a hard cap with no overshoot even under concurrent chains. */
struct SharedBudget {
  std::atomic<std::uint64_t> used{0};
  std::uint64_t cap = std::numeric_limits<std::uint64_t>::max();

  explicit SharedBudget(std::uint64_t c) : cap(c) {}
  bool try_reserve()
  {
    const std::uint64_t u = used.fetch_add(1, std::memory_order_relaxed);
    if (u < cap) return true;
    used.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
  void release() { used.fetch_sub(1, std::memory_order_relaxed); }
  std::uint64_t spent() const { return used.load(std::memory_order_relaxed); }
  std::uint64_t left()  const {
    const std::uint64_t u = spent();
    return cap > u ? cap - u : 0;
  }
};

/* Per-proposal stop context shared by all chains of a run. `stop` is
 * raised by whichever consumer first hits STOPVAL (or by the driver)
 * and makes every other chain wind down at its next proposal. */
struct StopCtx {
  SharedBudget&                budget;
  std::atomic<bool>&           stop;
  const std::function<bool()>& time_exceeded;
  std::optional<double>        stopval;
};

enum class RungStop { Completed, Budget, Time, StopVal, External };

struct RungStats {
  std::size_t proposed = 0, accepted = 0, oob = 0;
  std::size_t improved = 0;   /* moves that set a new chain personal best */
  RungStop    stop = RungStop::Completed;
};

inline long double metropolis_probability(
    double current, double proposed, double T,
    const std::optional<double>& failscore)
{
  if (!loss_is_rankable(proposed, failscore)) return 0.0L;
  if (!loss_is_rankable(current, failscore) || proposed <= current) return 1.0L;
  if (!(T > 0.0)) return 0.0L;
  const long double delta = static_cast<long double>(proposed) -
                            static_cast<long double>(current);
  return std::exp(-delta / static_cast<long double>(T));
}

/* FAILSCORE-aware Metropolis decision.  A rankable proposal always repairs an
 * unusable (NaN/Inf/exact-FAILSCORE) current state, while an unusable proposal
 * is never allowed to poison a healthy chain.  Graded failures below FS remain
 * rankable so the chain can follow their repair direction.  Forming the uphill
 * delta in long double avoids overflow for opposite-sign finite losses.
 * `coin` is invoked only for a genuine uphill move, preserving the documented
 * RNG stream. */
template<class DrawCoin>
inline bool metropolis_accept(double current, double proposed, double T,
                              const std::optional<double>& failscore,
                              DrawCoin&& coin)
{
  if (!loss_is_rankable(proposed, failscore)) return false;
  if (!loss_is_rankable(current, failscore) || proposed <= current) return true;
  const long double delta = static_cast<long double>(proposed) -
                            static_cast<long double>(current);
  const long double p = std::exp(-delta / static_cast<long double>(T));
  return static_cast<long double>(coin()) < p;
}

/* Smallest positive Gaussian scale that can move x by one representable
 * value while remaining inside its hard support. */
inline double resolvable_coordinate_scale(double x, double lo, double hi)
{
  double best = std::numeric_limits<double>::infinity();
  if (x < hi) {
    const double d = std::nextafter(x, hi) - x;
    if (d > 0.0 && std::isfinite(d)) best = std::min(best, d);
  }
  if (x > lo) {
    const double d = x - std::nextafter(x, lo);
    if (d > 0.0 && std::isfinite(d)) best = std::min(best, d);
  }
  return std::isfinite(best) ? best : std::numeric_limits<double>::denorm_min();
}

/* ---- ChainState ------------------------------------------------------
 *
 * One annealing chain (= one PT replica / PA walker / the single serial
 * chain). `V` stores the per-dimension proposal scale: a fixed Gaussian
 * sigma for CLASSIC, or the adaptive uniform half-width for Corana. The
 * RNG is the chain's private stream (make_stream(seed, SALT_CHAIN, chain_index)). */
template<class Vec>
struct ChainState {
  Vec    x;
  double f = std::numeric_limits<double>::infinity();
  std::vector<double> V;
  /* D27 trust-region anchor for the diagonal RAM: the calibrated per-dim
   * scale recorded once after warm-up.  Empty = no trust region (serial,
   * PA, Corana, and legacy callers). */
  std::vector<double> V_baseline;
  std::mt19937_64     rng;
  Vec    best_x;
  double best_f = std::numeric_limits<double>::infinity();
  std::vector<double> diag_ram_delta;
  double              diag_ram_scalar_delta = 0.0;
  std::uint64_t       diag_ram_proposals = 0;
  std::size_t         diag_ram_pending = 0;
};

/* Scalar-plus-diagonal robust-adaptive-Metropolis contribution. Every
 * proposal updates both accumulators; STEP_ADAPT_EVERY controls only when the
 * proposal kernel changes. The scalar term can repair a common scale error in
 * O(1) rather than O(D) adaptation time. The centred diagonal term changes
 * shape without also changing the geometric-mean scale. */
template<class Vec>
void accumulate_diagonal_ram(ChainState<Vec>& c,
                             const std::vector<double>& z,
                             long double alpha,
                             const OptimizerSpec& sp,
                             std::size_t adapt_every,
                             double trust_factor = 0.0)
{
  if (adapt_every == 0) return;
  const std::size_t dim = z.size();
  if (c.diag_ram_delta.size() != dim)
    c.diag_ram_delta.assign(dim, 0.0);
  long double norm2 = 0.0L;
  for (double zi : z) norm2 += static_cast<long double>(zi) * zi;
  const std::uint64_t j = ++c.diag_ram_proposals;
  ++c.diag_ram_pending;
  const long double eta = std::min<long double>(
      1.0L, std::pow(static_cast<long double>(j), -0.6L));
  c.diag_ram_scalar_delta +=
      static_cast<double>(eta * (alpha - 0.234L));
  if (norm2 > 0.0L && std::isfinite(norm2)) {
    for (std::size_t i = 0; i < dim; ++i) {
      const long double ratio =
          static_cast<long double>(z[i]) * z[i] / norm2;
      const long double term = eta * (alpha - 0.234L) * ratio;
      c.diag_ram_delta[i] += static_cast<double>(std::log1p(term));
    }
  }
  if (c.diag_ram_pending < adapt_every) return;

  const double log2 = std::log(2.0);
  const double shape_mean = dim > 0
      ? std::accumulate(c.diag_ram_delta.begin(), c.diag_ram_delta.end(), 0.0) /
            static_cast<double>(dim)
      : 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    const double log_sigma_factor = std::clamp(
        c.diag_ram_scalar_delta +
            0.5 * (c.diag_ram_delta[i] - shape_mean),
        -log2, log2);
    long double next = static_cast<long double>(c.V[i]) *
                       std::exp(static_cast<long double>(log_sigma_factor));
    /* D27: the centred shape term is a mean-zero random walk with no
     * restoring force, so per-dim sigma can drift arbitrarily far from
     * its calibrated scale (run24 held dims at sigma/width ~ 2e-10 for
     * 70k epochs).  Bound it to a factor of the recorded baseline.  The
     * support floor/cap is applied LAST: representability outranks the
     * trust region, so a coordinate can always still move. */
    if (trust_factor > 1.0 && c.V_baseline.size() == dim) {
      const long double base = static_cast<long double>(c.V_baseline[i]);
      next = std::max(base / trust_factor,
                      std::min(base * trust_factor, next));
    }
    const double floor = resolvable_coordinate_scale(
        c.x(i), sp.low[i], sp.high[i]);
    const double cap = sp.high[i] - sp.low[i];
    next = std::max(static_cast<long double>(floor),
                    std::min(static_cast<long double>(cap), next));
    c.V[i] = static_cast<double>(next);
  }
  std::fill(c.diag_ram_delta.begin(), c.diag_ram_delta.end(), 0.0);
  c.diag_ram_scalar_delta = 0.0;
  c.diag_ram_pending = 0;
}

/* ---- one rung (temperature step) on one chain ------------------------
 *
 * Corana variant: NT step-adjust cycles x NS step cycles x D dims of
 * one-dim-at-a-time U(-1,1)*V[i] proposals; V[i] adapted per NT-cycle
 * from the per-dim accept ratio (Corana 1987 0.4/0.6 deadband, gain C),
 * clamped to [V_FLOOR_FRAC*width, width].
 *
 * `allowance` additionally caps this rung's evals (deterministic
 * per-epoch budget slicing in the parallel drivers); pass UINT64_MAX
 * for the serial driver where SharedBudget alone is the guard.
 *
 * RNG discipline preserved from the serial implementation: the
 * Metropolis coin is drawn ONLY for uphill moves (short-circuit), so
 * per-seed sequences stay comparable across configurations. */
template<class Vec, class HostT>
RungStats corana_rung(ChainState<Vec>& c, double T,
                      const OptimizerSpec& sp, const Knobs& kb,
                      HostT& host, StopCtx& ctx,
                      std::uint64_t allowance =
                          std::numeric_limits<std::uint64_t>::max())
{
  const std::size_t dim = sp.low.size();
  RungStats rs;
  std::vector<std::size_t> nA(dim, 0), nP(dim, 0);
  std::uniform_real_distribution<double> u_pm(-1.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  bool inner_break = false;

  for (std::size_t nt = 0; nt < kb.NT && !inner_break; ++nt) {
    std::fill(nA.begin(), nA.end(), 0);
    std::fill(nP.begin(), nP.end(), 0);

    for (std::size_t ns = 0; ns < kb.NS && !inner_break; ++ns) {
      for (std::size_t i = 0; i < dim && !inner_break; ++i) {
        if (rs.proposed >= allowance) {
          rs.stop = RungStop::Budget; inner_break = true; break;
        }
        if (!ctx.budget.try_reserve()) {
          rs.stop = RungStop::Budget; inner_break = true; break;
        }
        if (ctx.stop.load(std::memory_order_relaxed)) {
          ctx.budget.release();
          rs.stop = RungStop::External; inner_break = true; break;
        }
        if (ctx.time_exceeded()) {
          ctx.budget.release();
          rs.stop = RungStop::Time; inner_break = true; break;
        }

        Vec x_prop = c.x;
        auto draw = [&]() -> long double {
          return static_cast<long double>(c.x(i)) +
                 static_cast<long double>(u_pm(c.rng)) *
                 static_cast<long double>(c.V[i]);
        };
        long double v = draw();
        if (constrain_proposal(v, sp.low[i], sp.high[i],
                               kb.boundary_mode, draw))
          ++rs.oob;
        x_prop(i) = static_cast<double>(v);

        const double f_prop = host(sp.out(x_prop));
        ++rs.proposed;
        ++nP[i];
        const bool accept = metropolis_accept(
            c.f, f_prop, T, sp.failscore, [&] { return udist(c.rng); });
        if (accept) {
          c.x = x_prop;
          c.f = f_prop;
          ++rs.accepted;
          ++nA[i];
          if (!loss_is_rankable(c.best_f, sp.failscore) || c.f < c.best_f) {
            c.best_f = c.f;
            c.best_x = c.x;
            ++rs.improved;
          }
        }
        if (stopval_reached(c.f, ctx.stopval, sp.failscore)) {
          ctx.stop.store(true, std::memory_order_relaxed);
          rs.stop = RungStop::StopVal; inner_break = true; break;
        }
      }
    }

    /* Corana adjusts only after a complete NS-by-D block.  Treating
     * unvisited dimensions as zero-acceptance when a budget/stop cut the
     * block short spuriously divided their steps by up to 1+C. */
    if (!inner_break)
      for (std::size_t i = 0; i < dim; ++i)
        c.V[i] = adjust_corana_step(
            c.V[i], sp.high[i] - sp.low[i], nA[i], nP[i],
            kb.C, kb.v_floor_frac);
  }
  return rs;
}

/* ---- near-bound floor-step audit ---------------------------------------
 *
 * Mark a dim when the chain lies within v_lo = V_FLOOR_FRAC*width of an
 * OPTIM bound and V[i] is already clamped at that floor. This records that
 * adaptive shrinkage is exhausted at the observed scale. It does not stop
 * inward proposals or prove bound pressure. Corana-only by construction
 * (CLASSIC never adapts V). Drivers feed the observations to HPOLog after
 * each rung/epoch; the journal prints the aggregate heuristic at milestones. */
template<class Vec>
std::vector<PinnedDim> pinned_dims(const ChainState<Vec>& c,
                                   const OptimizerSpec& sp,
                                   const Knobs& kb)
{
  std::vector<PinnedDim> out;
  const std::size_t dim = sp.low.size();
  for (std::size_t i = 0; i < dim; ++i) {
    const double w = sp.high[i] - sp.low[i];
    if (w <= 0.0) continue;
    const double v_lo = kb.v_floor_frac * w;
    /* Corana only: a dim is interesting when its step has ALSO bottomed
     * out, because then no further shrinkage is available to work it free.
     * That clause cannot be applied to CLASSIC -- V_FLOOR_FRAC is a
     * Corana-only knob (Knobs::parse rejects it otherwise), so under
     * CLASSIC it is just the 1e-3 default, while a calibrated CLASSIC
     * sigma/width sits several orders of magnitude below that. The test
     * would be vacuously true for every dim and the audit would degenerate
     * into an unconditional near-bound report while still claiming "step at
     * the V floor". Under CLASSIC the remaining clause -- x within
     * v_floor_frac*width of a bound -- is exact, checkable, and is what the
     * table actually reports. */
    if (kb.corana && c.V[i] > v_lo * (1.0 + 1e-9))
      continue;   /* step can still shrink */
    const double g_lo = c.x(i) - sp.low[i];
    const double g_hi = sp.high[i] - c.x(i);
    const bool   hi   = g_hi < g_lo;
    const double gap  = hi ? g_hi : g_lo;
    if (gap <= v_lo) out.push_back(PinnedDim{i, hi, gap});
  }
  return out;
}

/* Classic Kirkpatrick/Metropolis rung: L whole-vector Gaussian
 * proposals with fixed per-dim sigma (c.V), mirror-reflected. */
template<class Vec, class HostT>
RungStats classic_rung(ChainState<Vec>& c, double T,
                       const OptimizerSpec& sp, const Knobs& kb,
                       HostT& host, StopCtx& ctx,
                       std::uint64_t allowance =
                           std::numeric_limits<std::uint64_t>::max())
{
  const std::size_t dim = sp.low.size();
  RungStats rs;
  std::normal_distribution<double>       ndist(0.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  std::vector<double> z;
  if (kb.step_adapt_every > 0) z.resize(dim);

  for (std::size_t k = 0; k < kb.L; ++k) {
    if (rs.proposed >= allowance) { rs.stop = RungStop::Budget; break; }
    if (!ctx.budget.try_reserve()) { rs.stop = RungStop::Budget; break; }
    if (ctx.stop.load(std::memory_order_relaxed)) {
      ctx.budget.release();
      rs.stop = RungStop::External; break;
    }
    if (ctx.time_exceeded()) {
      ctx.budget.release();
      rs.stop = RungStop::Time; break;
    }

    /* Bounds are enforced per coordinate.  REDRAW resamples only the
     * offending coordinate, avoiding the exponential whole-vector retry
     * cost in high dimension. */
    Vec x_prop(dim);
    bool needed_bounds = false;
    for (std::size_t i = 0; i < dim; ++i) {
      double unit = 0.0;
      auto draw = [&]() -> long double {
        unit = ndist(c.rng);
        return static_cast<long double>(c.x(i)) +
               static_cast<long double>(c.V[i]) *
               static_cast<long double>(unit);
      };
      long double v = draw();
      if (constrain_proposal(v, sp.low[i], sp.high[i],
                             kb.boundary_mode, draw))
        needed_bounds = true;
      x_prop(i) = static_cast<double>(v);
      if (!z.empty()) {
        /* Reflection/redraw changes the realised direction. Feed RAM the
         * effective standardised displacement, not the discarded raw draw. */
        z[i] = c.V[i] > 0.0
            ? (x_prop(i) - c.x(i)) / c.V[i]
            : 0.0;
      }
    }
    if (needed_bounds) ++rs.oob;

    const double f_prop = host(sp.out(x_prop));
    ++rs.proposed;
    const long double alpha =
        metropolis_probability(c.f, f_prop, T, sp.failscore);
    const bool accept = metropolis_accept(
        c.f, f_prop, T, sp.failscore, [&] { return udist(c.rng); });
    if (accept) {
      c.x = x_prop;
      c.f = f_prop;
      ++rs.accepted;
      if (!loss_is_rankable(c.best_f, sp.failscore) || c.f < c.best_f) {
        c.best_f = c.f;
        c.best_x = c.x;
        ++rs.improved;
      }
    }
    if (!z.empty())
      accumulate_diagonal_ram(c, z, alpha, sp, kb.step_adapt_every,
                              kb.step_trust_factor);
    if (stopval_reached(c.f, ctx.stopval, sp.failscore)) {
      ctx.stop.store(true, std::memory_order_relaxed);
      rs.stop = RungStop::StopVal;
      break;
    }
  }
  return rs;
}

template<class Vec, class HostT>
RungStats run_rung(ChainState<Vec>& c, double T,
                   const OptimizerSpec& sp, const Knobs& kb,
                   HostT& host, StopCtx& ctx,
                   std::uint64_t allowance =
                       std::numeric_limits<std::uint64_t>::max())
{
  return kb.corana ? corana_rung(c, T, sp, kb, host, ctx, allowance)
                   : classic_rung(c, T, sp, kb, host, ctx, allowance);
}

/* ---- T0 calibration --------------------------------------------------
 *
 * Shared by the serial driver and the parallel drivers. Anchors on the
 * ALREADY-EVALUATED point (x, f) — typically the seed — and probes its
 * proposal-kernel neighbourhood; see robust_t0 for the statistics. On a
 * FAILSCORE anchor, spends one uniform-box batch to re-anchor (x, f)
 * on the best finite point, then probes locally again; throws when
 * nothing finite exists anywhere. Mutates (x, f) only on re-anchor and
 * keeps (best_x, best_f) consistent. Logs the same T0 line the serial
 * driver always emitted. */
struct T0Outcome {
  double T0 = 0.0;
  bool   re_anchored = false;
  bool   aborted     = false;   /* budget/STOPVAL/MAXTIME mid-probe */
  bool   budget_exhausted = false;
  bool   stopval_hit = false;
  bool   external_stop = false;
  std::size_t n_probes = 0;
};

template<class Vec, class HostT>
T0Outcome calibrate_t0(const OptimizerSpec& sp, const Knobs& kb,
                       const std::vector<double>& proposal_scale,
                       Vec& x, double& f_x, Vec& best_x, double& best_f,
                       HostT& host, std::mt19937_64& probe_rng,
                       StopCtx& ctx, const char* banner_ctx = "ANNEAL")
{
  const std::size_t dim = sp.low.size();
  T0Outcome out;

  if (kb.t_hot) {
    out.T0 = *kb.t_hot;
    HPOLog::instance().info(std::string(banner_ctx) +
      ": using user-specified T_HOT=" + std::to_string(out.T0));
    return out;
  }

  /* Tag this thread's evals as T0 probes for the journal's best-origin
   * tracking; cleared before returning so the serial driver's chain
   * evals (same thread) are not mislabelled. */
  struct ProbeContextGuard {
    ProbeContextGuard() { HPOLog::set_eval_probe(); }
    ~ProbeContextGuard() { HPOLog::clear_eval_context(); }
  } probe_context_guard;

  std::normal_distribution<double>       ndist(0.0, 1.0);
  std::uniform_real_distribution<double> u_pm(-1.0, 1.0);
  double bf_probe = std::numeric_limits<double>::infinity();
  Vec    bx_probe(dim);

  /* Local probes must mirror the PROPOSAL KERNEL, or the measured dE
   * scale is wrong: the Corana chain moves ONE dim at a time, and on a
   * narrow high-dimensional ridge a full-vector jitter (norm ~
   * sigma*sqrt(D)) lands on the FAILSCORE cliff every single time even
   * though every actual chain move is finite. Classic keeps the
   * full-vector Gaussian probe (that IS its kernel). */
  auto run_probes = [&](bool local, const Vec& centre,
                        std::vector<double>& probes) -> bool {
    std::uniform_int_distribution<std::size_t> pick_dim(0, dim - 1);
    for (std::size_t k = 0; k < kb.t0_probes; ++k) {
      if (!ctx.budget.try_reserve()) {
        out.aborted = true;
        out.budget_exhausted = true;
        return false;
      }
      if (ctx.stop.load(std::memory_order_relaxed)) {
        ctx.budget.release();
        out.aborted = true;
        out.external_stop = true;
        return false;
      }
      if (ctx.time_exceeded()) {
        ctx.budget.release();
        out.aborted = true;
        return false;
      }
      Vec xp(dim);
      if (local && kb.corana) {
        xp = centre;
        const std::size_t i = pick_dim(probe_rng);
        auto draw = [&]() -> long double {
          return static_cast<long double>(centre(i)) +
                 static_cast<long double>(u_pm(probe_rng)) *
                 static_cast<long double>(proposal_scale[i]);
        };
        long double v = draw();
        constrain_proposal(v, sp.low[i], sp.high[i],
                           kb.boundary_mode, draw);
        xp(i) = static_cast<double>(v);
      } else {
        for (std::size_t i = 0; i < dim; ++i) {
          auto draw = [&]() -> long double {
            return local
                ? static_cast<long double>(centre(i)) +
                      static_cast<long double>(proposal_scale[i]) *
                      static_cast<long double>(ndist(probe_rng))
                : static_cast<long double>(std::uniform_real_distribution<>(
                      sp.low[i], sp.high[i])(probe_rng));
          };
          long double v = draw();
          constrain_proposal(v, sp.low[i], sp.high[i],
                             kb.boundary_mode, draw);
          xp(i) = static_cast<double>(v);
        }
      }
      const double f = host(sp.out(xp));
      probes.push_back(f);
      if (loss_is_finite(f, sp.failscore) &&
          (!loss_is_finite(best_f, sp.failscore) || f < best_f)) {
        best_f = f;
        best_x = xp;
      }
      if (loss_is_finite(f, sp.failscore) && f < bf_probe) {
        bf_probe = f;
        bx_probe = xp;
      }
      if (stopval_reached(f, ctx.stopval, sp.failscore)) {
        ctx.stop.store(true, std::memory_order_relaxed);
        out.aborted     = true;
        out.stopval_hit = true;
        return false;
      }
    }
    return true;
  };

  std::vector<double> probes;
  const bool seed_finite = loss_is_finite(f_x, sp.failscore);

  if (seed_finite) {
    run_probes(/*local=*/true, x, probes);
  } else {
    HPOLog::instance().warning(std::string(banner_ctx) +
      ": the seed point evaluates to FAILSCORE; sampling "
      + std::to_string(kb.t0_probes) + " uniform box probe(s) to re-anchor");
    if (run_probes(/*local=*/false, x, probes) && !out.aborted) {
      if (std::isfinite(bf_probe)) {
        x   = bx_probe;
        f_x = bf_probe;
        if (!loss_is_finite(best_f, sp.failscore) || bf_probe < best_f) {
          best_f = bf_probe;
          best_x = bx_probe;
        }
        out.re_anchored = true;
        probes.clear();
        run_probes(/*local=*/true, x, probes);
      } else {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << banner_ctx << ": T0 calibration failed -- the seed point and all "
            << probes.size() << " uniform box probe(s) scored FAILSCORE";
        if (sp.failscore) oss << " (" << *sp.failscore << ")";
        oss << ". The optimiser has no finite loss to anchor on. "
               "Remedies: set PARAM T_HOT explicitly, provide a viable "
               "starting point (INIT block / config values), or shrink "
               "the OPTIM boxes.";
        throw std::runtime_error(oss.str());
      }
    }
  }
  out.n_probes = probes.size();

  /* Probe-loss spread stats use the same FAILSCORE-aware population as
   * robust_t0; sentinel/Inf values must not turn a diagnostic into NaN. */
  std::vector<double> diagnostic_probes;
  diagnostic_probes.reserve(probes.size());
  for (double v : probes)
    if (loss_is_finite(v, sp.failscore)) diagnostic_probes.push_back(v);
  double pmin = 0.0, pmax = 0.0, pstd = 0.0;
  if (!diagnostic_probes.empty()) {
    pmin = pmax = diagnostic_probes.front();
    long double mean = 0.0L, m2 = 0.0L;
    std::size_t count = 0;
    for (double v : diagnostic_probes) {
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
      ++count;
      const long double delta = static_cast<long double>(v) - mean;
      mean += delta / static_cast<long double>(count);
      const long double delta2 = static_cast<long double>(v) - mean;
      m2 += delta * delta2;
    }
    const long double dmax = static_cast<long double>(
        std::numeric_limits<double>::max());
    if (diagnostic_probes.size() >= 2) {
      const long double sd = std::sqrt(
          m2 / static_cast<long double>(diagnostic_probes.size() - 1));
      pstd = static_cast<double>(std::min(dmax, sd));
    }
  }

  if (!out.aborted) {
    const auto t0_est = robust_t0(f_x, probes, sp.failscore, kb.t0_paccept);
    double med_de = 0.0;
    if (!t0_est) {
      out.T0 = 1.0;
      HPOLog::instance().warning(std::string(banner_ctx) + ": all "
        + std::to_string(probes.size()) + " local probe(s) around the "
        "(finite) anchor scored FAILSCORE; falling back to T0=1.0. "
        "Consider a smaller STEP_FRAC or PARAM T_HOT.");
    } else if (*t0_est <= 0.0) {
      out.T0 = 1.0;
      HPOLog::instance().warning(std::string(banner_ctx) +
        ": T0 calibration saw zero loss variation over "
        + std::to_string(probes.size())
        + " probe(s); falling back to T0=1.0. Consider PARAM T_HOT "
        "or a larger STEP_FRAC.");
    } else {
      out.T0 = *t0_est;
      med_de = out.T0 * -std::log(kb.t0_paccept);
    }
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3)
        << banner_ctx << ": T0=" << out.T0
        << " from " << probes.size() << " probes,"
        << "  median|dE|=" << med_de
        << "  loss range=[" << pmin << ", " << pmax << "]"
        << "  std=" << pstd
        << "  target p_accept=" << kb.t0_paccept
        << "  anchor=" << (out.re_anchored ? "re-anchored" : "seed");
    std::cout << "[HPO] " << oss.str() << "\n";
    HPOLog::instance().info(oss.str());
  } else {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3)
        << banner_ctx << ": T0 probes truncated after " << probes.size()
        << " sample(s)  loss range=[" << pmin << ", " << pmax << "]";
    HPOLog::instance().info(oss.str());
  }
  return out;
}

}  // namespace anneal
}  // namespace tadah::hpo

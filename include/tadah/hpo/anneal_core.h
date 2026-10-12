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
#include <tadah/core/number_format.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pc_fail_grading.h>   /* kFailBandFloor, loss_is_feasible */
#include <tadah/hpo/seed_stream.h>       /* seed_stream, seed_streams         */
#include <tadah/hpo/anneal_keys_gen.h>   /* generated key/rule registry      */
#include <tadah/hpo/strict_parse.h>     /* parse_number                     */

#include <algorithm>
#include <atomic>
#include <cctype>
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
#include <string_view>
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
 * (SEED, salt, index) through a seed_seq: chains (SALT_CHAIN, per global
 * slot), T0 probes (SALT_PROBES; index 1 = the PT second pass), PT swap
 * coins (SALT_PT_SWAP, per epoch*K+pair), the PT step warm-up
 * (SALT_WARMUP, per global slot), and the PA resampler. Streams are
 * decorrelated and independent, so the calibration phases (T0 probes,
 * warm-up, pilot) never shift a chain's production proposal sequence:
 * with explicit temperatures and a fixed ladder, chain proposals are a
 * pure function of (SEED, slot) for every STEP_MODE, and parallel
 * chains draw reproducibly regardless of thread scheduling. (Chain
 * INITIALIZATION -- JITTER/UNIFORM draws for slots >= 1 -- does consume
 * the chain stream and legitimately differs between INIT_MODEs.) The salts
 * are registered with those of the other optimisers' streams in
 * seed_stream.h, so no two consumers of a SEED share one. */
constexpr std::uint64_t SALT_CHAIN       = seed_streams::kAnnealChain;
constexpr std::uint64_t SALT_PROBES      = seed_streams::kAnnealProbes;
constexpr std::uint64_t SALT_PT_SWAP     = seed_streams::kAnnealPtSwap;
constexpr std::uint64_t SALT_PA_RESAMPLE = seed_streams::kAnnealPaResample;
constexpr std::uint64_t SALT_WARMUP      = seed_streams::kAnnealWarmup;

/* ---- adaptation targets (literature constants) ----------------------
 *
 * kTargetAcceptVector: optimal acceptance for whole-vector random-walk
 *   Metropolis in moderate-to-high dimension (Roberts, Gelman & Gilks,
 *   Ann. Appl. Probab. 7, 110 (1997): 0.234).
 * kTargetAcceptCoord: optimal acceptance for one-coordinate-at-a-time
 *   proposals (the d = 1 scaling limit; Gelman, Roberts & Gilks 1996:
 *   0.44).
 * kRamDecay: Robbins-Monro gain exponent eta_j = j^-kRamDecay for the
 *   adaptive-Metropolis update; any value in (1/2, 1] satisfies the
 *   diminishing-adaptation conditions (Andrieu & Thoms, Stat. Comput.
 *   18, 343 (2008)). Distinct from LADDER_ADAPT_DECAY, which governs the
 *   temperature-ladder controller.  */
constexpr double kTargetAcceptVector = 0.234;
constexpr double kTargetAcceptCoord  = 0.44;
constexpr double kRamDecay           = 0.6;

/* AUTO's gamma. NOT a tuned number and deliberately NOT a function of
 * LADDER_SPAN: in a locally quadratic basin the equilibrium coordinate
 * spread scales as sqrt(T) by equipartition, whatever the ladder happens to
 * span. The span already enters through T_k/T_hot, so making gamma depend on
 * it as well would count the same quantity twice. On the default geometric
 * ladder (S = 1e3) gamma = 0.5 gives the coldest non-seed slot 3.2% of
 * sigma_d and the hottest the full value -- the intended shape, produced by
 * the span alone. */
constexpr double kLadderJitterGammaAuto = 0.5;

inline std::mt19937_64 make_stream(std::uint64_t seed,
                                   std::uint64_t salt,
                                   std::uint64_t idx = 0)
{
  return seed_stream(seed, salt, idx);
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
/* Test/tooling observation seam: the drivers report every chain's
 * end-of-rung (PT: end-of-sweep, PRE-swap) state and every exchange
 * attempt. Coordinates arrive in TRANSFORMED (post-LOG_HP) space. Not
 * config-parseable; wired via OptimizerSpec::anneal_observer. Zero cost
 * when null. Callbacks run on the driver thread that owns the chain
 * (PT worker threads for on_rung_end -- implementations must be
 * thread-safe there; on_swap_attempt fires on the driver's exchange
 * phase, single-threaded). */
struct ChainObserver {
  virtual void on_rung_end(int /*slot*/, double /*T*/,
                           const std::vector<double>& /*x*/, double /*f*/) {}
  virtual void on_swap_attempt(std::size_t /*epoch*/, std::size_t /*pair*/,
                               bool /*accepted*/, bool /*degenerate*/,
                               double /*t_lo*/, double /*t_hi*/,
                               double /*f_lo*/, double /*f_hi*/) {}
  virtual ~ChainObserver() = default;
};

struct Knobs {
  /* Every default below comes from the generated anneal_defaults
   * constants (source of truth: blueprint/anneal_keys.toml).  Do not
   * restate a literal here.  0 on chain_length/NT is the AUTO sentinel
   * (serial: resolved from the MAXEVAL budget after T0 calibration). */

  /* stepper (shared across the family) */
  bool        corana        = anneal_defaults::CORANA == 1.0;
  std::size_t chain_length  = 0;      /* CLASSIC proposals per rung;
                                       * 0 = AUTO (serial), PA default
                                       * applied in parse                */
  std::size_t NS            = static_cast<std::size_t>(anneal_defaults::NS);
                                      /* Corana step cycles (PT: sweeps
                                       * between per-slot V adjustments) */
  std::size_t NT            = 0;      /* Corana step-adjust cycles per
                                       * rung; 0 = AUTO (serial)         */
  double      corana_c      = anneal_defaults::CORANA_C;
                                      /* Corana step-adjust gain         */
  double      step_frac     = anneal_defaults::STEP_FRAC;
  double      v_floor_frac  = anneal_defaults::V_FLOOR_FRAC;
  BoundaryMode boundary_mode =
      static_cast<BoundaryMode>(static_cast<int>(anneal_defaults::BOUNDARY_MODE));
  bool        reset_to_best = false;  /* dynamic default: true under
                                       * serial CORANA (Knobs::parse)    */
  std::size_t neps          = static_cast<std::size_t>(anneal_defaults::NEPS);

  /* temperature family.  t_start holds the serial T_INIT or the PT/PA
   * T_HOT -- one field, kind-appropriate key name at parse time. */
  std::optional<double> t_start;      /* explicit value skips calibration */
  std::optional<double> t_min;        /* explicit floor; auto = T0 * 1e-6 */
  double      rt            = anneal_defaults::RT_SERIAL;  /* PA: RT_PA */
  std::size_t t0_probes     = static_cast<std::size_t>(anneal_defaults::T0_PROBES);
  double      t0_accept     = anneal_defaults::T0_ACCEPT;

  /* serial warm re-anneal on T_MIN */
  bool        reanneal       = anneal_defaults::REANNEAL == 1.0;
  double      reanneal_scale = anneal_defaults::REANNEAL_SCALE;

  /* parallel tempering (ANNEAL_PT only) */
  std::size_t replicas    = static_cast<std::size_t>(anneal_defaults::REPLICAS);
                                    /* 0 = resolve to max(THREADS, 2)   */
  std::optional<double> t_cold;     /* explicit coldest slot            */
  double      ladder_span = anneal_defaults::LADDER_SPAN;
                                    /* T_COLD = T_HOT / LADDER_SPAN     */
  double      ladder_power = anneal_defaults::LADDER_POWER;
                                    /* initial log-ladder shape         */
  bool        ladder_adapt = anneal_defaults::LADDER_ADAPT == 1.0;
                                    /* adapt interior temperatures      */
  std::size_t ladder_adapt_every =
      static_cast<std::size_t>(anneal_defaults::LADDER_ADAPT_EVERY);
                                    /* attempts/pair between updates    */
  double      ladder_adapt_gain = anneal_defaults::LADDER_ADAPT_GAIN;
  double      ladder_adapt_decay = anneal_defaults::LADDER_ADAPT_DECAY;
  double      ladder_min_gap = anneal_defaults::LADDER_MIN_GAP;
                                    /* min interior log-gap as a fraction
                                     * of the mean gap; 0 = no floor    */
  bool        ladder_freeze = anneal_defaults::LADDER_FREEZE == 1.0;
                                    /* pooled-evidence equalization
                                     * freeze; 0 = adapt forever        */
  std::size_t swap_every  = static_cast<std::size_t>(anneal_defaults::SWAP_EVERY);
                                    /* exchange attempt every N epochs  */
  std::size_t swap_window = static_cast<std::size_t>(anneal_defaults::SWAP_WINDOW);
                                    /* epochs per diagnostics window
                                     * (0 = cumulative reports only)    */
  int         step_mode   = anneal_enums::STEP_MODE::FIXED;
                                    /* PT CLASSIC step policy; parse sets
                                     * the FROZEN default for PT         */
  std::size_t step_warmup_sweeps =
      static_cast<std::size_t>(anneal_defaults::STEP_WARMUP_SWEEPS);
                                    /* min calibration attempts/coord    */
  double      step_trust_factor = 0.0;
                                    /* RAM sigma bound as a factor of the
                                     * calibrated baseline; parse sets
                                     * the registry default under
                                     * ADAPTIVE, 0 = off                 */

  /* shared by both parallel drivers */
  int         init_mode   = static_cast<int>(anneal_defaults::INIT_MODE);
                                    /* JITTER = seed + jitter, UNIFORM =
                                     * box draw (slot 0 always starts
                                     * exactly at the seed)              */

  /* How BIG the initial jitter is, decoupled from STEP_FRAC (1.5.0).
   *
   * STEP is the historical behaviour and stays the default: one vector,
   * STEP_FRAC * width, doing double duty as the Metropolis proposal scale
   * AND the initialisation sigma. That coupling is the defect -- a
   * STEP_FRAC chosen for the production kernel also decides how far the
   * warm start is thrown away, and at high dimension "a fraction of every
   * box at once" is a different point in the space, not a perturbation.
   *
   * FIXED takes an explicit fraction; AUTO measures one per dimension.
   * Both draw UNIFORMLY on +/-sigma rather than Gaussian, so the magnitude
   * is a bound on every coordinate instead of a one-sigma average. */
  enum class JitterMode { Step, Fixed, Auto };
  JitterMode  init_jitter_mode = JitterMode::Step;
  double      init_jitter_frac = 0.0;   /* FIXED: sigma / OPTIM width      */
  double      init_jitter_tol  = 0.0;   /* AUTO:  relative loss tolerance p */
  /* AUTO shape and cost. MAXEVAL is deliberately NOT charged (the phase runs
   * on its own probe budget), so TIME is the axis that still bounds it --
   * hence a MAXTIME fraction rather than an eval fraction. */
  double      init_jitter_max   = 0.5;  /* cap on sigma/width; screen point */
  std::size_t init_jitter_probes = 8;   /* per dimension, both sides        */
  std::size_t init_jitter_maxeval = 0;  /* 0 = AUTO: dim * probes           */
  double      init_jitter_maxtime_frac = 0.25;
  std::size_t init_jitter_sample = 0;   /* 0 = every dimension            */
  std::size_t init_jitter_audit  = 32;  /* dims re-probed at sigma/2      */
  /* PT ladder scaling of the INITIAL jitter: sigma_k = sigma_d *
   * (T_k/T_hot)^gamma.  0 = OFF (every slot draws from the same sigma,
   * the pre-1.5.0 shape). `_auto` only selects the journal wording; the
   * resolvable-scale floor applies whenever gamma > 0, because an
   * explicit gamma can collapse the cold slots just as easily. */
  double      init_jitter_ladder = 0.0;
  bool        init_jitter_ladder_auto = false;

  /* population annealing (ANNEAL_PA only) */
  std::size_t sweeps      = static_cast<std::size_t>(anneal_defaults::SWEEPS);
                                    /* rungs per walker per T-step      */
  double      ess_target  = anneal_defaults::ESS_TARGET;
                                    /* adaptive dbeta target; 0 = RT    */

  /* test/tooling seam, copied from OptimizerSpec::anneal_observer */
  ChainObserver* observer = nullptr;

  static Knobs parse(const OptimizerSpec& sp, AlgoKind kind);
};

/* Case-insensitive token compare for the 1.5.0 PARAM value vocabulary
 * (ON / OFF / AUTO). Config keywords elsewhere in Tadah are matched
 * case-insensitively, so the value vocabulary follows suit. */
inline std::string anneal_upper(std::string s)
{
  for (char& c : s)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

/* Render a knob value the way the user most likely typed it, for migration
 * messages: integral values without a decimal tail, so `PARAM NS 20`
 * echoes back as '20' rather than '20.000000'. */
inline std::string fmt_knob_number(double v)
{
  if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
    std::ostringstream os;
    os << static_cast<long long>(v);
    return os.str();
  }
  std::ostringstream os;
  os << v;
  return os.str();
}

inline Knobs Knobs::parse(const OptimizerSpec& sp, AlgoKind kind)
{
  namespace reg = anneal_registry;
  const std::string ctx = std::string("TADAH:") + algo_name(kind);
  const unsigned kind_mask = kind == AlgoKind::Serial ? reg::KIND_SERIAL
                           : kind == AlgoKind::PT     ? reg::KIND_PT
                                                      : reg::KIND_PA;

  /* -- retired/wrong-kind key names: hard error with the registry's
   *    migration hint (blueprint/anneal_keys.toml [[renamed]]) -------- */
  for (const auto& ren : reg::kRenamed) {
    if (!(ren.kinds & kind_mask)) continue;
    if (sp.param.count(ren.old_name) || sp.param_raw.count(ren.old_name))
      throw std::runtime_error(ctx + ": " + ren.msg);
  }

  /* -- membership: key names and family applicability come from the
   *    generated registry (blueprint/anneal_keys.toml). Unknown or
   *    wrong-family PARAMs are hard errors; keys of a sibling family
   *    member steer the user to the right ALGO. --------------------- */
  auto check_membership = [&](const std::string& name) {
    const auto* ki = reg::find_key(name);
    if (ki && (ki->kinds & kind_mask)) return;
    if (ki) {
      if (ki->kinds == reg::KIND_PT)
        throw std::runtime_error(
            ctx + ": PARAM " + name + " belongs to TADAH:ANNEAL_PT");
      if (ki->kinds == reg::KIND_PA)
        throw std::runtime_error(
            ctx + ": PARAM " + name + " belongs to TADAH:ANNEAL_PA");
      if (ki->kinds == reg::KIND_SERIAL)
        throw std::runtime_error(
            ctx + ": PARAM " + name + " belongs to the serial TADAH:ANNEAL");
      if (ki->kinds == (reg::KIND_PT | reg::KIND_PA))
        throw std::runtime_error(
            ctx + ": PARAM " + name + " belongs to the parallel drivers "
            "TADAH:ANNEAL_PT/TADAH:ANNEAL_PA");
      std::string members;
      if (ki->kinds & reg::KIND_SERIAL) members += " TADAH:ANNEAL";
      if (ki->kinds & reg::KIND_PT)     members += " TADAH:ANNEAL_PT";
      if (ki->kinds & reg::KIND_PA)     members += " TADAH:ANNEAL_PA";
      throw std::runtime_error(
          ctx + ": PARAM " + name + " belongs to" + members);
    }
    std::string keys;
    for (const auto& k : reg::kKeys)
      if (k.kinds & kind_mask) { keys += k.name; keys += ' '; }
    throw std::runtime_error(
        ctx + ": unknown PARAM '" + name + "'; accepted keys: " + keys);
  };
  for (const auto& kv : sp.param) check_membership(kv.first);
  for (const auto& kv : sp.param_raw) {
    check_membership(kv.first);
    /* Token values are legal for enum-typed keys, for booleans (ON/OFF)
     * and for numeric keys that declare AUTO and/or OFF in the registry.
     * Anything else is a typo or a misplaced word, never a silent 0.0. */
    const auto* ki = reg::find_key(kv.first);
    if (ki->type == reg::AnnealKeyType::Enum) continue;
    if (ki->type == reg::AnnealKeyType::Bool) continue;   // ON/OFF, see pbool
    const std::string t = anneal_upper(kv.second);
    if ((ki->accepts_auto && t == "AUTO") || (ki->accepts_off && t == "OFF"))
      continue;
    std::string accepted = "a number";
    if (ki->accepts_auto) accepted += ", AUTO";
    if (ki->accepts_off)  accepted += ", OFF";
    throw std::runtime_error(
        ctx + ": PARAM " + kv.first + " must be " + accepted + " (got '" +
        kv.second + "')");
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
  if (sp.step)
    throw std::runtime_error(
        ctx + " does not use STEP; set PARAM STEP_FRAC (and PARAM "
        "STEP_MODE for ANNEAL_PT) instead");
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

  /* -- typed getters: defaults and value form come from the registry
   *    entry of each key.  Every (key, value) pair that reaches the
   *    knobs -- explicit or defaulted -- is recorded for the
   *    registry-declared range checks that run after the cross-key
   *    rules (so e.g. "requires CORANA ON" outranks a range error). -- */
  std::vector<std::pair<const reg::AnnealKeyInfo*, double>> parsed_values;
  auto pinfo = [&](const char* k) -> const reg::AnnealKeyInfo& {
    const auto* ki = reg::find_key(k);
    if (!ki)
      throw std::logic_error(
          std::string("anneal_keys.toml lost key ") + k +
          "; regenerate anneal_keys_gen.h");
    return *ki;
  };
  auto default_of = [&](const reg::AnnealKeyInfo& ki, bool& has) -> double {
    switch (kind) {
      case AlgoKind::Serial: has = ki.has_def_serial; return ki.def_serial;
      case AlgoKind::PT:     has = ki.has_def_pt;     return ki.def_pt;
      default:               has = ki.has_def_pa;     return ki.def_pa;
    }
  };
  auto pget = [&](const char* k) -> double {
    const auto& ki = pinfo(k);
    double v;
    /* 1.5.0 vocabulary: AUTO / OFF resolve to the registry's internal
     * value and take no range check -- the sentinel is by construction
     * outside the key's legal numeric range (e.g. REPLICAS AUTO -> 0
     * while REPLICAS itself must be >= 2). */
    if (auto rit = sp.param_raw.find(k); rit != sp.param_raw.end()) {
      const std::string t = anneal_upper(rit->second);
      if (ki.accepts_auto && t == "AUTO") return ki.auto_value;
      if (ki.accepts_off  && t == "OFF")  return ki.off_value;
    }
    auto it = sp.param.find(k);
    if (it == sp.param.end()) {
      bool has = false;
      v = default_of(ki, has);
      /* No registry default for this kind = presence-tested optional or
       * a run-time AUTO key: the 0 sentinel takes no range check. */
      if (!has) return v;
    } else {
      if (!std::isfinite(it->second))
        throw std::runtime_error(ctx + ": PARAM " + k + " must be finite");
      v = it->second;
    }
    parsed_values.emplace_back(&ki, v);
    return v;
  };
  auto pcount = [&](const char* k) -> std::size_t {
    const double v = pget(k);
    const long double size_limit = std::ldexp(
        1.0L, std::numeric_limits<std::size_t>::digits);
    if (v < 0.0 || std::floor(v) != v ||
        static_cast<long double>(v) >= size_limit)
      throw std::runtime_error(
          ctx + ": PARAM " + k + " must be a non-negative integer");
    return static_cast<std::size_t>(v);
  };
  /* Booleans take the tokens ON / OFF (1.5.0). A numeric 0/1 is a hard
   * error carrying its own migration hint: accepting it silently would
   * make the old and new spellings indistinguishable in a config, which
   * is exactly the ambiguity this vocabulary removes. */
  auto pbool_token = [&](const char* k, bool& out) -> bool {
    auto rit = sp.param_raw.find(k);
    if (rit == sp.param_raw.end()) return false;
    const std::string t = anneal_upper(rit->second);
    if (t == "ON")  { out = true;  return true; }
    if (t == "OFF") { out = false; return true; }
    throw std::runtime_error(
        ctx + ": PARAM " + k + " takes ON or OFF (got '" + rit->second + "')");
  };
  auto pbool_reject_numeric = [&](const char* k) {
    if (sp.param.count(k))
      throw std::runtime_error(
          ctx + ": PARAM " + k + " takes ON or OFF (got '" +
          fmt_knob_number(sp.param.at(k)) + "'). 0/1 booleans were removed "
          "in 1.5.0 -- use `PARAM " + std::string(k) + " " +
          (sp.param.at(k) != 0.0 ? "ON" : "OFF") + "`.");
  };
  auto pbool = [&](const char* k) -> bool {
    pbool_reject_numeric(k);
    bool v = false;
    if (pbool_token(k, v)) return v;
    const auto& ki = pinfo(k);
    /* A key that does not apply to this kind has no per-kind default and
     * reads as false -- Knobs::parse walks every key unconditionally, so
     * e.g. serial ANNEAL still asks for the PT-only LADDER_ADAPT. This
     * mirrors what pget() did for numeric bools before 1.5.0. */
    bool has = false;
    const double def = default_of(ki, has);
    return has && def != 0.0;
  };
  /* For keys whose default depends on other knobs (no registry default). */
  auto pbool_dyn = [&](const char* k, bool dflt) -> bool {
    pbool_reject_numeric(k);
    bool v = false;
    if (pbool_token(k, v)) return v;
    return dflt;
  };
  /* Enum keys take token values (PARAM STEP_MODE FROZEN); the token list
   * and default come from the registry. */
  auto penum = [&](const char* k) -> int {
    const auto& ki = pinfo(k);
    auto tokens_of = [&]() {
      std::string tokens;
      for (std::size_t i = 0; i < ki.enum_count; ++i) {
        if (i) tokens += ' ';
        tokens += ki.enum_values[i];
      }
      return tokens;
    };
    auto raw = sp.param_raw.find(k);
    if (raw != sp.param_raw.end()) {
      for (std::size_t i = 0; i < ki.enum_count; ++i)
        if (raw->second == ki.enum_values[i]) return static_cast<int>(i);
      throw std::runtime_error(
          ctx + ": PARAM " + k + " must be one of: " + tokens_of() +
          " (got '" + raw->second + "')");
    }
    if (sp.param.count(k))
      throw std::runtime_error(
          ctx + ": PARAM " + k + " takes a token, one of: " + tokens_of());
    bool has = false;
    const double def = default_of(ki, has);
    if (!has)
      throw std::logic_error(
          std::string("anneal_keys.toml enum key ") + k +
          " lacks a default for this kind");
    return static_cast<int>(def);
  };

  Knobs kb;
  kb.corana        = pbool("CORANA");

  kb.chain_length  = pcount("CHAIN_LENGTH");   /* serial: 0 = AUTO */
  kb.NS            = pcount("NS");
  kb.NT            = pcount("NT");             /* serial: 0 = AUTO */
  kb.corana_c      = pget("CORANA_C");
  kb.step_frac     = pget("STEP_FRAC");
  kb.v_floor_frac  = pget("V_FLOOR_FRAC");
  const int boundary_raw = penum("BOUNDARY_MODE");
  /* Corana's published cooling step returns to the best-so-far point.
   * Keep that default for serial Corana; PT defaults off because snapping
   * hot replicas defeats exchange, while PA rejects the knob (registry
   * KindReject rule). */
  kb.reset_to_best = pbool_dyn("RESET_TO_BEST",
                               kb.corana && kind == AlgoKind::Serial);
  kb.neps          = pcount("NEPS");

  /* One start-temperature field, kind-appropriate key name: the serial
   * schedule starts at T_INIT; PT/PA anchor the hot end at T_HOT. */
  const char* t_start_key = kind == AlgoKind::Serial ? "T_INIT" : "T_HOT";
  if (sp.param.count(t_start_key)) kb.t_start = pget(t_start_key);
  if (sp.param.count("T_MIN")) kb.t_min = pget("T_MIN");
  kb.rt            = pget("RT");
  kb.t0_probes     = pcount("T0_PROBES");
  kb.t0_accept     = pget("T0_ACCEPT");

  kb.reanneal       = kind == AlgoKind::Serial && pbool("REANNEAL");
  kb.reanneal_scale = pget("REANNEAL_SCALE");

  kb.replicas    = pcount("REPLICAS");
  if (sp.param.count("T_COLD")) kb.t_cold = pget("T_COLD");
  kb.ladder_span = pget("LADDER_SPAN");
  kb.ladder_power = pget("LADDER_POWER");
  kb.ladder_adapt = pbool("LADDER_ADAPT");
  kb.ladder_adapt_every = pcount("LADDER_ADAPT_EVERY");
  kb.ladder_adapt_gain = pget("LADDER_ADAPT_GAIN");
  kb.ladder_adapt_decay = pget("LADDER_ADAPT_DECAY");
  kb.ladder_min_gap = pget("LADDER_MIN_GAP");
  kb.ladder_freeze = pbool("LADDER_FREEZE");
  kb.swap_every  = pcount("SWAP_EVERY");
  kb.swap_window = pcount("SWAP_WINDOW");
  kb.step_mode   = kind == AlgoKind::PT ? penum("STEP_MODE")
                                        : anneal_enums::STEP_MODE::FIXED;
  kb.step_warmup_sweeps = pcount("STEP_WARMUP_SWEEPS");
  kb.step_trust_factor = pget("STEP_TRUST_FACTOR");
  kb.init_mode   = kind != AlgoKind::Serial
                       ? penum("INIT_MODE")
                       : anneal_enums::INIT_MODE::JITTER;
  /* INIT_MODE's optional extras (multi-token PARAM, 1.5.0):
   *     INIT_MODE UNIFORM
   *     INIT_MODE JITTER                 sigma = STEP_FRAC * width
   *     INIT_MODE JITTER <frac>          sigma = <frac> * width
   *     INIT_MODE JITTER AUTO <p>        sigma measured per dimension
   * The registry owns the mode token; the extras are a small grammar the
   * table cannot express, so they are parsed here and every rejection is a
   * hard error rather than a silently ignored word. */
  if (auto tk = sp.param_tokens.find("INIT_MODE");
      tk != sp.param_tokens.end() && tk->second.size() > 1) {
    const auto& toks = tk->second;
    if (kind == AlgoKind::Serial)
      throw std::runtime_error(
          ctx + ": PARAM INIT_MODE takes no value beyond the mode for "
                "serial annealing (it runs a single chain)");
    if (kb.init_mode != anneal_enums::INIT_MODE::JITTER)
      throw std::runtime_error(
          ctx + ": PARAM INIT_MODE " + toks[0] + " takes no further value "
                "(got '" + toks[1] + "'); the jitter magnitude applies to "
                "JITTER only");
    /* The number grammar of every key (strict_parse.h, parse_number):
     * std::stod also read hexadecimal floats. */
    auto as_number = [&](const std::string& t, const char* what) {
      double v = 0.0;
      if (parse_number(t, v)) return v;
      throw std::runtime_error(ctx + ": PARAM INIT_MODE JITTER " +
                               std::string(what) + " must be a finite "
                               "number (got '" + t + "')" +
                               number_underflow_note(t));
    };
    if (anneal_upper(toks[1]) == "AUTO") {
      if (toks.size() < 3)
        throw std::runtime_error(
            ctx + ": PARAM INIT_MODE JITTER AUTO needs a relative loss "
                  "tolerance, e.g. `PARAM INIT_MODE JITTER AUTO 0.05`");
      kb.init_jitter_mode = Knobs::JitterMode::Auto;
      kb.init_jitter_tol  = as_number(toks[2], "AUTO tolerance");
      if (!(kb.init_jitter_tol > 0.0))
        throw std::runtime_error(
            ctx + ": PARAM INIT_MODE JITTER AUTO tolerance must be > 0");
    } else {
      kb.init_jitter_mode = Knobs::JitterMode::Fixed;
      kb.init_jitter_frac = as_number(toks[1], "fraction");
      if (!(kb.init_jitter_frac > 0.0 && kb.init_jitter_frac <= 1.0))
        throw std::runtime_error(
            ctx + ": PARAM INIT_MODE JITTER <fraction> must be in (0,1] "
                  "(a fraction of each OPTIM interval)");
    }
  }
  kb.init_jitter_max    = pget("INIT_JITTER_MAX");
  kb.init_jitter_probes = pcount("INIT_JITTER_PROBES");
  kb.init_jitter_maxeval = pcount("INIT_JITTER_MAXEVAL");
  kb.init_jitter_maxtime_frac = pget("INIT_JITTER_MAXTIME_FRAC");
  kb.init_jitter_sample = pcount("INIT_JITTER_SAMPLE");
  kb.init_jitter_audit  = pcount("INIT_JITTER_AUDIT");
  if (kind == AlgoKind::PT) {
    const double g = pget("INIT_JITTER_LADDER");
    kb.init_jitter_ladder_auto = (g == anneal_defaults::INIT_JITTER_LADDER);
    kb.init_jitter_ladder = kb.init_jitter_ladder_auto
                                ? kLadderJitterGammaAuto : g;
  }
  kb.sweeps      = pcount("SWEEPS");
  kb.ess_target  = pget("ESS_TARGET");

  /* -- registry cross-key rules (blueprint/anneal_keys.toml): explicitly
   *    set keys whose values would be ignored are hard errors, never
   *    silent.  Silent precedence is especially dangerous in long HPO
   *    runs because the banner can show both knobs while only one
   *    affects the trajectory. ------------------------------------------ */
  auto flag_parsed = [&](std::string_view flag) -> bool {
    if (flag == "CORANA")       return kb.corana;
    if (flag == "REANNEAL")     return kb.reanneal;
    if (flag == "LADDER_ADAPT") return kb.ladder_adapt;
    throw std::logic_error(
        "anneal_keys.toml rule references unmapped flag '" +
        std::string(flag) + "'");
  };
  auto enum_parsed = [&](std::string_view name) -> int {
    if (name == "STEP_MODE") return kb.step_mode;
    if (name == "INIT_MODE") return kb.init_mode;
    throw std::logic_error(
        "anneal_keys.toml rule references unmapped enum '" +
        std::string(name) + "'");
  };
  auto enum_index = [&](const char* name, const char* token) -> int {
    const auto* ki = reg::find_key(name);
    for (std::size_t i = 0; ki && i < ki->enum_count; ++i)
      if (std::string_view(token) == ki->enum_values[i])
        return static_cast<int>(i);
    throw std::logic_error(
        std::string("anneal_keys.toml rule token ") + token +
        " is not a value of " + name);
  };
  for (const auto& rule : reg::kRules) {
    if (!sp.param.count(rule.key) && !sp.param_raw.count(rule.key)) continue;
    bool fire = false;
    switch (rule.type) {
      case reg::AnnealRuleType::RequiresFlag:
        fire = !flag_parsed(rule.arg); break;
      case reg::AnnealRuleType::ConflictsFlag:
        fire = flag_parsed(rule.arg); break;
      case reg::AnnealRuleType::UnusedWhenSet:
        fire = sp.param.count(rule.arg) != 0 ||
               sp.param_raw.count(rule.arg) != 0;
        break;
      case reg::AnnealRuleType::UnusedWhenFlagOff:
        fire = !flag_parsed(rule.arg); break;
      case reg::AnnealRuleType::RequiresEnum:
        fire = enum_parsed(rule.arg) != enum_index(rule.arg, rule.arg2);
        break;
      case reg::AnnealRuleType::KindReject:
        fire = (rule.kind_mask & kind_mask) != 0; break;
    }
    if (fire) throw std::runtime_error(ctx + ": " + rule.msg);
  }

  /* -- cross-key semantics beyond the registry rule forms ------------ */
  if (kind == AlgoKind::PT && kb.corana) {
    /* Explicit STEP_MODE/STEP_WARMUP_SWEEPS/... alongside CORANA are
     * rejected by the registry conflicts rules above; the DEFAULT
     * STEP_MODE collapses to FIXED because CORANA adapts its own
     * per-dimension steps and must not run the CLASSIC warm-up. */
    kb.step_mode = anneal_enums::STEP_MODE::FIXED;
  }
  if (kind == AlgoKind::PT &&
      kb.step_mode == anneal_enums::STEP_MODE::FIXED && !kb.corana &&
      sp.param.count("STEP_WARMUP_SWEEPS"))
    throw std::runtime_error(
        ctx + ": PARAM STEP_WARMUP_SWEEPS is unused when STEP_MODE is "
              "FIXED (no calibration warm-up runs)");
  /* The trust region exists only under ADAPTIVE (the per-proposal RAM);
   * FIXED/FROZEN kernels carry the 0 = off sentinel internally. */
  if (kb.step_mode != anneal_enums::STEP_MODE::ADAPTIVE)
    kb.step_trust_factor = 0.0;

  /* The AUTO shape/cost keys are registry-gated on INIT_MODE JITTER, which is
   * the closest the rule table can express: AUTO is a multi-token EXTRA on
   * INIT_MODE, not an enum value of it. Finish the job here so setting them
   * on a run that never calibrates is an error rather than a no-op. */
  if (kb.init_jitter_mode != Knobs::JitterMode::Auto) {
    for (const char* k : {"INIT_JITTER_MAX", "INIT_JITTER_PROBES",
                          "INIT_JITTER_MAXEVAL", "INIT_JITTER_MAXTIME_FRAC",
                          "INIT_JITTER_SAMPLE", "INIT_JITTER_AUDIT"})
      if (sp.param.count(k) || sp.param_raw.count(k))
        throw std::runtime_error(
            ctx + ": PARAM " + std::string(k) + " is unused unless INIT_MODE "
                  "is `JITTER AUTO <tolerance>`");
  } else if (sp.param.count("INIT_JITTER_MAXEVAL") &&
             kb.init_jitter_maxeval == 0) {
    throw std::runtime_error(
        ctx + ": PARAM INIT_JITTER_MAXEVAL must be >= 1, or AUTO to let the "
              "protocol size its own cap");
  }

  /* INIT_JITTER_LADDER is deliberately NOT in the list above: it shapes the
   * initial population for EVERY INIT_MODE JITTER variant, not just AUTO.
   * What it needs is a jitter at all, so UNIFORM is the case to reject --
   * and that cannot be a registry rule, because a `requires_enum` against
   * INIT_MODE's own default value can never fire. */
  if (kind == AlgoKind::PT &&
      (sp.param.count("INIT_JITTER_LADDER") ||
       sp.param_raw.count("INIT_JITTER_LADDER"))) {
    if (kb.init_mode != anneal_enums::INIT_MODE::JITTER)
      throw std::runtime_error(
          ctx + ": PARAM INIT_JITTER_LADDER scales the per-slot INITIAL "
                "JITTER and is unused when INIT_MODE is UNIFORM");
    /* A bare 0 is the pre-1.5.0 spelling of OFF and is rejected with the
     * migration hint rather than silently meaning gamma = 0, which happens
     * to do the same thing -- accepting it would leave the old and new
     * spellings indistinguishable in a config file. */
    if (sp.param.count("INIT_JITTER_LADDER")) {
      const double g = sp.param.at("INIT_JITTER_LADDER");
      if (g == 0.0)
        throw std::runtime_error(
            ctx + ": PARAM INIT_JITTER_LADDER 0 is no longer accepted; "
                  "0/1 booleans and 0-means-disabled were removed in 1.5.0 -- "
                  "write PARAM INIT_JITTER_LADDER OFF to disable ladder "
                  "scaling, or AUTO for the sqrt(T) default");
      if (!(g > 0.0 && g <= 4.0))
        throw std::runtime_error(
            ctx + ": PARAM INIT_JITTER_LADDER must be in (0,4], or AUTO "
                  "(gamma=0.5, equipartition) or OFF; got " +
                  fmt_knob_number(g));
    }
  }

  kb.observer = sp.anneal_observer;

  /* -- registry-declared ranges (checked after the cross-key rules so
   *    a wrong-variant key errors as such, not with a range message;
   *    covers every explicit or defaulted value that was parsed) ----- */
  for (const auto& [ki, v] : parsed_values) {
    if (!ki->range_msg) continue;
    bool bad = false;
    if (ki->has_min) bad = ki->min_open ? !(v > ki->min) : !(v >= ki->min);
    if (!bad && ki->has_max)
      bad = ki->max_open ? !(v < ki->max) : !(v <= ki->max);
    if (bad) {
      std::string msg =
          ctx + ": PARAM " + ki->name + " " + ki->range_msg;
      if (ki->report_value) msg += " (got " + std::to_string(v) + ")";
      throw std::runtime_error(msg);
    }
  }
  kb.boundary_mode = static_cast<BoundaryMode>(boundary_raw);
  /* BOUNDARY_MODE REDRAW is announced by announce_knobs() when the
   * driver starts (D23), not here: Knobs::parse() is also the startup
   * check of the spec (backend_rules.h), which journals nothing. */

  /* -- validation beyond the registry ranges ------------------------- */
  if (kind != AlgoKind::PT && !(kb.rt > 0.0 && kb.rt < 1.0))
    throw std::runtime_error(ctx + ": PARAM RT must be in (0,1) (got "
        + std::to_string(kb.rt) + ")");
  if (kb.t_start && kb.t_min && *kb.t_min > *kb.t_start)
    throw std::runtime_error(
        ctx + ": PARAM T_MIN must be <= " + t_start_key);
  if (!kb.t_start && kb.t0_probes < 2)
    throw std::runtime_error(
        ctx + ": PARAM T0_PROBES must be >= 2 when " + t_start_key +
        " is not set (the calibration needs at least two probes)");
  const bool plateau_armed =
      (sp.stop.ftol_rel && *sp.stop.ftol_rel > 0.0) ||
      (sp.stop.ftol_abs && *sp.stop.ftol_abs > 0.0);
  if (plateau_armed && kb.neps == 0)
    throw std::runtime_error(ctx + ": PARAM NEPS must be >= 1");
  /* Serial AUTO lengths are resolved from the eval budget after T0
   * calibration; without MAXEVAL there is no principled value to derive,
   * so make the user choose instead of inventing one. */
  if (kind == AlgoKind::Serial && !sp.stop.maxeval) {
    if (!kb.corana && kb.chain_length == 0)
      throw std::runtime_error(
          ctx + ": CHAIN_LENGTH AUTO needs MAXEVAL to derive the "
          "per-rung budget; set PARAM CHAIN_LENGTH explicitly (e.g. for "
          "MAXTIME- or STOPVAL-driven runs)");
    if (kb.corana && kb.NT == 0)
      throw std::runtime_error(
          ctx + ": NT AUTO needs MAXEVAL to derive the per-rung budget; "
          "set PARAM NT explicitly (e.g. for MAXTIME- or STOPVAL-driven "
          "runs)");
  }
  if (kind == AlgoKind::PT) {
    if (sp.param.count("REPLICAS") && kb.replicas < 2)
      throw std::runtime_error(ctx + ": PARAM REPLICAS must be >= 2");
    if (kb.t_cold) {
      if (!(*kb.t_cold > 0.0))
        throw std::runtime_error(ctx + ": PARAM T_COLD must be > 0");
      if (kb.t_start && !(*kb.t_cold < *kb.t_start))
        throw std::runtime_error(ctx + ": PARAM T_COLD must be < T_HOT");
    }
    if (kb.step_mode != anneal_enums::STEP_MODE::FIXED &&
        kb.step_warmup_sweeps == 0)
      throw std::runtime_error(
          ctx + ": PARAM STEP_WARMUP_SWEEPS must be >= 1");
    /* 1.5.0: OFF is the "disable the clamp" spelling; a numeric 0 is not.
     * Validate what the USER wrote (sp.param), not kb.step_trust_factor --
     * the latter is deliberately zeroed above for non-ADAPTIVE step modes,
     * an internal sentinel that is none of the config's business. */
    if (auto tf = sp.param.find("STEP_TRUST_FACTOR"); tf != sp.param.end()) {
      if (!(std::isfinite(tf->second) && tf->second > 1.0))
        throw std::runtime_error(
            ctx + ": PARAM STEP_TRUST_FACTOR must be > 1, or OFF to disable "
                  "the clamp");
    }
  }
  if (kind == AlgoKind::PA) {
    if (sp.pop && *sp.pop < 8)
      throw std::runtime_error(ctx + ": POPULATION must be >= 8");
  }

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
  }

  return kb;
}

/* The journal notes a run's knobs call for, written by the driver when the
 * run starts (Knobs::parse() writes nothing, so the startup check of the
 * spec does not repeat them).
 *
 * D23: REDRAW is not a symmetric proposal, and the acceptance rule does
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
inline void announce_knobs(const Knobs& kb, AlgoKind kind)
{
  const std::string ctx = std::string("TADAH:") + algo_name(kind);
  if (kb.boundary_mode == BoundaryMode::Redraw)
    HPOLog::instance().hint(
        ctx + ": BOUNDARY_MODE REDRAW resamples out-of-box coordinates "
        "from a truncated kernel whose normalisation depends on the current "
        "point, and the Metropolis rule applies no Hastings correction for "
        "it. Sampling near an OPTIM bound is therefore biased away from the "
        "edge. This does not affect which best point is reported, but if you "
        "care about the distribution rather than the optimum, use "
        "BOUNDARY_MODE REFLECT, which is an exact symmetric fold.");
}

/* The ladder size K of an ANNEAL_PT run: PARAM REPLICAS, or else THREADS
 * (at least two). Throws the rules on K that Knobs::parse cannot check on
 * its own (K follows THREADS when REPLICAS is AUTO): K >= 2, and
 * LADDER_ADAPT needs K >= 3. The PT driver and the startup checks
 * (backend_rules.h) apply the same rule. */
inline std::size_t pt_replica_count(const OptimizerSpec& sp, const Knobs& kb)
{
  const std::size_t threads = sp.num_threads.value_or(1);
  const std::size_t K = kb.replicas ? kb.replicas
                                    : std::max<std::size_t>(threads, 2);
  if (K < 2)
    throw std::runtime_error("TADAH:ANNEAL_PT: needs REPLICAS >= 2");
  if (kb.ladder_adapt && K < 3)
    throw std::runtime_error(
        "TADAH:ANNEAL_PT: LADDER_ADAPT requires REPLICAS >= 3");
  return K;
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
      if (!sp.log_scaled.empty() && sp.log_scaled[i] &&
          (!(raw[i] > 0.0) || !std::isfinite(raw[i])))
        throw std::runtime_error(
            ctx + ": log-scaled starting coordinate must be finite and > 0");
    }
    /* REAL -> INTERNAL through the search space (ln / asinh / affine). */
    x = sp.in(raw);
  }
  for (std::size_t i = 0; i < dim; ++i) {
    if (!std::isfinite(x[i]) || x[i] < sp.low[i] || x[i] > sp.high[i])
      throw std::runtime_error(
          ctx + ": starting coordinate " + std::to_string(i) +
          " lies outside its OPTIM bounds");
  }
  return x;
}

/* Build the per-dimension proposal scale once for every driver.
 * Corana's adaptive state always starts inside the same advertised
 * [floor*width, width] interval it must obey after adjustment. */
inline std::vector<double> make_initial_steps(const OptimizerSpec& sp,
                                              const Knobs& kb)
{
  std::vector<double> out(sp.low.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double width = sp.high[i] - sp.low[i];
    const double raw = kb.step_frac * width;
    if (!(raw > 0.0) || !std::isfinite(raw))
      throw std::runtime_error(
          "anneal: proposal scale underflowed or is non-finite in dimension " +
          std::to_string(i) + "; widen the OPTIM interval or raise STEP_FRAC");
    out[i] = kb.corana
        ? std::min(width, std::max(kb.v_floor_frac * width, raw))
        : raw;
  }
  return out;
}

/* Per-dimension jitter sigma for replica initialisation.
 *
 * STEP reuses the proposal scale, which is exactly the historical coupling:
 * one number sets both the production step and how far the warm start is
 * thrown away. FIXED takes its own fraction of each OPTIM interval instead.
 * (AUTO measures a vector per dimension and supplies it directly, so it does
 * not come through here.) */
inline std::vector<double> make_jitter_sigma(
    const OptimizerSpec& sp, const Knobs& kb,
    const std::vector<double>& proposal_scale)
{
  if (kb.init_jitter_mode != Knobs::JitterMode::Fixed) return proposal_scale;
  std::vector<double> out(sp.low.size());
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = kb.init_jitter_frac * (sp.high[i] - sp.low[i]);
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
        /* 0 = disabled, not "bitwise equality over NEPS rungs": an exact
         * eps of zero can only ever fire on identical doubles, which no
         * user asking for a tolerance stop means. */
        active_((stop.ftol_rel.has_value() && *stop.ftol_rel > 0.0) ||
                (stop.ftol_abs.has_value() && *stop.ftol_abs > 0.0)),
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
 * value while remaining inside its hard support, floored at one machine
 * epsilon of the interval width. The ULP of x alone collapses to a
 * subnormal when x sits at 0 -- which, in the unit-box search space
 * (OPTIM_SCALE AUTO), is every dimension's lower bound -- so the width
 * term keeps the floor meaningful there; for |x| ~ width the two agree
 * within a factor of 2. */
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
  const double ulp = std::isfinite(best) ? best : std::numeric_limits<double>::denorm_min();
  const double width = hi - lo;
  const double width_floor = (std::isfinite(width) && width > 0.0)
      ? std::numeric_limits<double>::epsilon() * width : 0.0;
  return std::max(ulp, width_floor);
}

/* Factor applied to the initial jitter at a slot of temperature T_k.
 *
 * Returns 1 for the hottest slot, for gamma <= 0 (scaling OFF) and for any
 * degenerate ladder -- a non-finite or non-positive temperature means the
 * caller has bigger problems than the jitter shape, and silently returning
 * the unscaled sigma is the behaviour that cannot make things worse. */
inline double ladder_jitter_factor(double T_k, double T_max, double gamma)
{
  if (!(gamma > 0.0)) return 1.0;
  if (!std::isfinite(T_k) || !std::isfinite(T_max)) return 1.0;
  if (!(T_k > 0.0) || !(T_max > 0.0)) return 1.0;
  const double r = T_k / T_max;
  if (!std::isfinite(r) || !(r > 0.0) || r >= 1.0) return 1.0;
  const double f = std::pow(r, gamma);
  return std::isfinite(f) ? f : 1.0;
}

/* One slot's scaled jitter vector, plus how many coordinates the floor
 * caught.
 *
 * The floor is where a genuinely span-dependent problem lives: at
 * LADDER_SPAN = 1e6, gamma = 0.5 leaves slot 1 with 0.1% of sigma_d and the
 * coldest few slots become numerically indistinguishable from the seed --
 * K replicas collapsing to a handful of distinct starting points, which
 * defeats the ensemble. Clamping at resolvable_coordinate_scale (the floor
 * this codebase already uses everywhere else, so no new magic number) keeps
 * every slot a distinct point. Clamped coordinates are counted, not hidden:
 * a large count means the ladder is too wide for the search space and the
 * user should hear about it. */
struct SlotJitter {
  std::vector<double> sigma;
  std::size_t floored = 0;
};

inline SlotJitter scale_jitter_for_slot(const std::vector<double>& base,
                                        const std::vector<double>& x0,
                                        const std::vector<double>& lo,
                                        const std::vector<double>& hi,
                                        double T_k, double T_max, double gamma)
{
  SlotJitter out;
  out.sigma = base;
  const double f = ladder_jitter_factor(T_k, T_max, gamma);
  if (f >= 1.0) return out;                 /* nothing to scale or to floor */
  for (std::size_t i = 0; i < out.sigma.size(); ++i) {
    const double scaled = base[i] * f;
    const double floor_i = resolvable_coordinate_scale(x0[i], lo[i], hi[i]);
    /* Only a sigma the scaling actually pushed BELOW the floor counts: a
     * base sigma that was already sub-resolvable is the calibration's
     * verdict, not this feature's doing, and reporting it here would blame
     * ladder scaling for something it did not cause. */
    if (scaled < floor_i && base[i] >= floor_i) {
      out.sigma[i] = floor_i;
      ++out.floored;
    } else {
      out.sigma[i] = scaled;
    }
  }
  return out;
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
  std::uint64_t       diag_ram_proposals = 0;  /* RAM eta counter j */
  /* Sweep-mode Corana accumulators (ANNEAL_PT): per-dimension accept /
   * propose counts and the completed-sweep counter feeding the V
   * adjustment every NS sweeps. Slot-resident, like V itself. The
   * serial rung kernel uses rung-local counters instead. */
  std::vector<std::size_t> corana_nA, corana_nP;
  std::size_t              corana_sweeps = 0;
  /* Lifetime per-dimension acceptance census.  corana_nA/corana_nP above
   * are the ADAPTATION evidence window: adjust_corana_step consumes them
   * and they are then zeroed IN PLACE, so by the time any reporting code
   * looks at them they describe at most the tail of one window.  These
   * are never zeroed, so "which coordinates actually moved" survives the
   * reset.  Slot-resident like V: an exchange swaps x/f between slots and
   * leaves these behind, making the census per-SLOT (per-temperature),
   * not per-walker.  See corana_census.h. */
  std::vector<std::size_t> corana_census_nA, corana_census_nP;
};

/* Size the lifetime census to `dim` on first use.  Separate from the
 * corana_nA/nP sizing next to it because these must survive every reset
 * that one undergoes. */
template<class Vec>
void ensure_corana_census(ChainState<Vec>& c, std::size_t dim)
{
  if (c.corana_census_nA.size() != dim) c.corana_census_nA.assign(dim, 0);
  if (c.corana_census_nP.size() != dim) c.corana_census_nP.assign(dim, 0);
}

/* Scalar-plus-diagonal robust-adaptive-Metropolis update, applied ONCE
 * PER PROPOSAL (Robbins-Monro: ln V_i += clamp(eta_j * g_i, +/- ln 2),
 * eta_j = j^-kRamDecay).  The scalar part eta*(alpha - kTargetAcceptVector)
 * repairs a common scale error in O(1) adaptation time; the centred
 * shape part changes per-dimension shape without moving the
 * geometric-mean scale.
 *
 * Per-observation application matters: the previous implementation
 * accumulated a whole STEP_ADAPT_EVERY window of increments and applied
 * the sum in one multiplicative step, so every acceptance in the window
 * was measured at a stale kernel and the feedback-loop gain was
 * multiplied by the window length -- the +/- ln 2 clamp and the trust
 * region were containing that oscillation rather than any real drift.
 * Here each acceptance feeds back immediately at its own eta_j, which
 * is the form the convergence results are stated for (Andrieu & Thoms
 * 2008), and STEP_ADAPT_EVERY ceased to mean anything and was retired. */
template<class Vec>
void apply_diagonal_ram(ChainState<Vec>& c,
                        const std::vector<double>& z,
                        long double alpha,
                        const OptimizerSpec& sp,
                        double trust_factor = 0.0)
{
  const std::size_t dim = z.size();
  if (dim == 0) return;
  long double norm2 = 0.0L;
  for (double zi : z) norm2 += static_cast<long double>(zi) * zi;
  const std::uint64_t j = ++c.diag_ram_proposals;
  const long double eta = std::min<long double>(
      1.0L, std::pow(static_cast<long double>(j),
                     -static_cast<long double>(kRamDecay)));
  const long double err =
      alpha - static_cast<long double>(kTargetAcceptVector);
  const double scalar = static_cast<double>(eta * err);

  /* Centred shape term from THIS proposal's standardised displacement. */
  std::vector<double> delta(dim, 0.0);
  double shape_mean = 0.0;
  if (norm2 > 0.0L && std::isfinite(norm2)) {
    for (std::size_t i = 0; i < dim; ++i) {
      const long double ratio =
          static_cast<long double>(z[i]) * z[i] / norm2;
      delta[i] = static_cast<double>(std::log1p(eta * err * ratio));
      shape_mean += delta[i];
    }
    shape_mean /= static_cast<double>(dim);
  }

  const double log2 = std::log(2.0);
  for (std::size_t i = 0; i < dim; ++i) {
    const double log_sigma_factor = std::clamp(
        scalar + 0.5 * (delta[i] - shape_mean), -log2, log2);
    long double next = static_cast<long double>(c.V[i]) *
                       std::exp(static_cast<long double>(log_sigma_factor));
    /* D27: the centred shape term is a mean-zero random walk with no
     * restoring force, so per-dim sigma can still drift far from its
     * calibrated scale over long runs.  Bound it to a factor of the
     * recorded baseline.  The support floor/cap is applied LAST:
     * representability outranks the trust region, so a coordinate can
     * always still move. */
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
 * per-seed sequences stay comparable across configurations.
 *
 * One single-coordinate Corana move on dimension i (shared by the serial
 * NT-by-NS rung kernel and the ANNEAL_PT sweep kernel).  Returns false
 * when the caller must stop, with rs.stop recording why. */
template<class Vec, class HostT>
bool corana_move(ChainState<Vec>& c, std::size_t i, double T,
                 const OptimizerSpec& sp, const Knobs& kb,
                 HostT& host, StopCtx& ctx, RungStats& rs,
                 std::uint64_t allowance,
                 std::vector<std::size_t>& nA, std::vector<std::size_t>& nP,
                 std::uniform_real_distribution<double>& u_pm,
                 std::uniform_real_distribution<double>& udist)
{
  if (rs.proposed >= allowance) { rs.stop = RungStop::Budget; return false; }
  if (!ctx.budget.try_reserve()) { rs.stop = RungStop::Budget; return false; }
  if (ctx.stop.load(std::memory_order_relaxed)) {
    ctx.budget.release();
    rs.stop = RungStop::External; return false;
  }
  if (ctx.time_exceeded()) {
    ctx.budget.release();
    rs.stop = RungStop::Time; return false;
  }

  Vec x_prop = c.x;
  auto draw = [&]() -> long double {
    return static_cast<long double>(c.x(i)) +
           static_cast<long double>(u_pm(c.rng)) *
           static_cast<long double>(c.V[i]);
  };
  long double v = draw();
  if (constrain_proposal(v, sp.low[i], sp.high[i], kb.boundary_mode, draw))
    ++rs.oob;
  x_prop(i) = static_cast<double>(v);

  const double f_prop = host(sp.out(x_prop));
  ++rs.proposed;
  ++nP[i];
  /* Bounds-checked rather than assumed: the census is sized by the two
   * kernels below, and corana_move is also reachable directly from tests. */
  if (i < c.corana_census_nP.size()) ++c.corana_census_nP[i];
  const bool accept = metropolis_accept(
      c.f, f_prop, T, sp.failscore, [&] { return udist(c.rng); });
  if (accept) {
    c.x = x_prop;
    c.f = f_prop;
    ++rs.accepted;
    ++nA[i];
    if (i < c.corana_census_nA.size()) ++c.corana_census_nA[i];
    if (!loss_is_rankable(c.best_f, sp.failscore) || c.f < c.best_f) {
      c.best_f = c.f;
      c.best_x = c.x;
      ++rs.improved;
    }
  }
  if (stopval_reached(c.f, ctx.stopval, sp.failscore)) {
    ctx.stop.store(true, std::memory_order_relaxed);
    rs.stop = RungStop::StopVal; return false;
  }
  return true;
}

/* The Corana 1987 NT-by-NS rung kernel (serial ANNEAL / ANNEAL_PA). */
template<class Vec, class HostT>
RungStats corana_rung(ChainState<Vec>& c, double T,
                      const OptimizerSpec& sp, const Knobs& kb,
                      HostT& host, StopCtx& ctx,
                      std::uint64_t allowance =
                          std::numeric_limits<std::uint64_t>::max())
{
  const std::size_t dim = sp.low.size();
  RungStats rs;
  ensure_corana_census(c, dim);
  std::vector<std::size_t> nA(dim, 0), nP(dim, 0);
  std::uniform_real_distribution<double> u_pm(-1.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  bool inner_break = false;

  for (std::size_t nt = 0; nt < kb.NT && !inner_break; ++nt) {
    std::fill(nA.begin(), nA.end(), 0);
    std::fill(nP.begin(), nP.end(), 0);

    for (std::size_t ns = 0; ns < kb.NS && !inner_break; ++ns)
      for (std::size_t i = 0; i < dim && !inner_break; ++i)
        if (!corana_move(c, i, T, sp, kb, host, ctx, rs, allowance,
                         nA, nP, u_pm, udist))
          inner_break = true;

    /* Corana adjusts only after a complete NS-by-D block.  Treating
     * unvisited dimensions as zero-acceptance when a budget/stop cut the
     * block short spuriously divided their steps by up to 1+C. */
    if (!inner_break)
      for (std::size_t i = 0; i < dim; ++i)
        c.V[i] = adjust_corana_step(
            c.V[i], sp.high[i] - sp.low[i], nA[i], nP[i],
            kb.corana_c, kb.v_floor_frac);
  }
  return rs;
}

/* One Corana sweep (dim single-coordinate moves) for ANNEAL_PT, where a
 * replica runs exactly one sweep per epoch so exchange stays frequent.
 * The accept/propose counters live in the ChainState (slot-resident,
 * like V) and the V adjustment fires after every NS COMPLETED sweeps --
 * the same NS-by-D evidence window as the serial kernel, accumulated
 * across epochs instead of inside one rung.  A sweep cut short by
 * budget/stop leaves the counters untouched for the next epoch. */
template<class Vec, class HostT>
RungStats corana_sweep(ChainState<Vec>& c, double T,
                       const OptimizerSpec& sp, const Knobs& kb,
                       HostT& host, StopCtx& ctx,
                       std::uint64_t allowance =
                           std::numeric_limits<std::uint64_t>::max())
{
  const std::size_t dim = sp.low.size();
  RungStats rs;
  if (c.corana_nA.size() != dim) c.corana_nA.assign(dim, 0);
  if (c.corana_nP.size() != dim) c.corana_nP.assign(dim, 0);
  ensure_corana_census(c, dim);
  std::uniform_real_distribution<double> u_pm(-1.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);

  bool complete = true;
  for (std::size_t i = 0; i < dim; ++i) {
    if (!corana_move(c, i, T, sp, kb, host, ctx, rs, allowance,
                     c.corana_nA, c.corana_nP, u_pm, udist)) {
      complete = false;
      break;
    }
  }
  if (complete) {
    ++c.corana_sweeps;
    if (c.corana_sweeps % kb.NS == 0) {
      for (std::size_t i = 0; i < dim; ++i)
        c.V[i] = adjust_corana_step(
            c.V[i], sp.high[i] - sp.low[i], c.corana_nA[i], c.corana_nP[i],
            kb.corana_c, kb.v_floor_frac);
      std::fill(c.corana_nA.begin(), c.corana_nA.end(), 0);
      std::fill(c.corana_nP.begin(), c.corana_nP.end(), 0);
    }
  }
  return rs;
}

/* Outcome of measuring one dimension's jitter magnitude.
 *
 * Deliberately NOT pt_detail::StepStatus: that lives in algorithms_pt.h,
 * which ANNEAL_PA does not include, and the two mean different things -- the
 * step warm-up targets an acceptance RATE, this measures a BOUND.
 *
 * The bands are distinguished because they call for different actions and
 * because collapsing them is how a diagnostic stops being actionable:
 * "insensitive" says the objective does not respond to a coordinate over its
 * whole OPTIM range (consider dropping or fixing it), while "gate-reject"
 * says a perfectly sensitive coordinate is bounded by a FEASIBILITY wall
 * (BUMPGATE), not by the loss -- pre-training and therefore cheap. A cliff is
 * neither: the objective returned a non-rankable loss. */
enum JitterStatus : unsigned {
  JitterOk            = 0,
  JitterNumericalFloor = 1u << 0,  /* sigma hit the resolvable scale        */
  JitterSupportCapped  = 1u << 1,  /* sigma hit INIT_JITTER_MAX * width     */
  JitterInsensitive    = 1u << 2,  /* in band even at the cap               */
  JitterOneSided       = 1u << 3,  /* seed on a bound: only one side probed */
  JitterAsymmetric     = 1u << 4,  /* the two signs disagree by > 2x        */
  JitterCliff          = 1u << 5,  /* a probe returned a non-rankable loss  */
  JitterGateReject     = 1u << 6,  /* BUMPGATE band D: infeasible, not bad  */
  JitterNoAnchor       = 1u << 7,  /* L0 unusable; AUTO abandoned           */
  JitterInterrupted    = 1u << 8,  /* cap/time cut this dimension short     */
  JitterInferred       = 1u << 9,  /* not measured: inherited from its family */
  JitterNonMonotone    = 1u << 10, /* out of band INSIDE sigma (audit)        */
};

inline std::string jitter_status_text(unsigned st)
{
  if (st == JitterOk) return "ok";
  std::string out;
  auto add = [&](const char* n) { if (!out.empty()) out += '+'; out += n; };
  if (st & JitterNumericalFloor) add("numerical-floor");
  if (st & JitterSupportCapped)  add("support-capped");
  if (st & JitterInsensitive)    add("insensitive");
  if (st & JitterOneSided)       add("one-sided");
  if (st & JitterAsymmetric)     add("asymmetric");
  if (st & JitterCliff)          add("cliff");
  if (st & JitterGateReject)     add("gate-reject");
  if (st & JitterNoAnchor)       add("no-anchor");
  if (st & JitterInterrupted)    add("interrupted");
  if (st & JitterInferred)       add("inferred");
  if (st & JitterNonMonotone)    add("non-monotone");
  return out;
}

/* Initial state of ONE replica (ANNEAL_PT slot) / walker (ANNEAL_PA), shared
 * by both drivers so the two cannot drift apart.
 *
 * `k` is the GLOBAL index: slot/walker 0 always starts exactly at the seed
 * `x0`, whatever INIT_MODE says. JITTER spreads the rest around it at the
 * initial proposal scale (independent of STEP_MODE -- the warm-up calibrates
 * scale, not position); UNIFORM draws them from the whole box.
 *
 * RNG-CONSUMPTION CONTRACT. This function defines how much of the chain's
 * private stream initialisation eats, and PT/PA trajectories at a fixed SEED
 * are only reproducible while that is stable. Three things matter and are
 * easy to "tidy" away:
 *   - `nd` is constructed ONCE per replica, not per dimension. A
 *     std::normal_distribution generates variates in pairs and caches the
 *     second, so a per-dimension object would draw twice as often.
 *   - slot 0 constructs `nd` but draws NOTHING (the `k > 0` guard), so the
 *     branch shape has to stay as written rather than being hoisted.
 *   - under BOUNDARY_MODE REDRAW, constrain_proposal calls `draw` again for
 *     every rejected coordinate, so consumption is data-dependent.
 * tests/test_seed_replica.cpp pins all three against an inline transcription
 * of the pre-1.5.0 loop, comparing stream POSITION and not just coordinates:
 * the two can agree while the streams have diverged. */
template<class Vec>
void seed_replica(ChainState<Vec>& c, std::uint64_t seed, std::size_t k,
                  const Vec& x0, const std::vector<double>& proposal_scale,
                  const std::vector<double>& jitter_sigma,
                  const OptimizerSpec& sp, const Knobs& kb)
{
  const std::size_t dim = sp.low.size();
  c.rng = make_stream(seed, SALT_CHAIN, k);
  /* V is the PRODUCTION step and always comes from proposal_scale; only the
   * initial displacement uses jitter_sigma. Under INIT_JITTER STEP the two
   * are the same vector, which is the pre-1.5.0 behaviour. */
  c.V   = proposal_scale;
  c.x   = Vec(dim);
  if (k == 0 || kb.init_mode == anneal_enums::INIT_MODE::JITTER) {
    /* STEP keeps the Gaussian, so an unconfigured run is bit-identical to
     * pre-1.5.0. FIXED/AUTO draw uniformly on +/-sigma instead: their sigma
     * is a measured or declared BOUND, and a Gaussian would put ~32% of
     * coordinates outside the bound it was told to respect. */
    const bool gaussian = kb.init_jitter_mode == Knobs::JitterMode::Step;
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (std::size_t i = 0; i < dim; ++i) {
      long double v = static_cast<long double>(x0(i));
      if (k > 0) {
        auto draw = [&]() {
          return static_cast<long double>(x0(i)) +
                 static_cast<long double>(jitter_sigma[i]) *
                     (gaussian ? nd(c.rng) : ud(c.rng));
        };
        v = draw();
        (void)constrain_proposal(v, sp.low[i], sp.high[i], kb.boundary_mode,
                                 draw);
      }
      c.x(i) = static_cast<double>(v);
    }
  } else {
    for (std::size_t i = 0; i < dim; ++i)
      c.x(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(c.rng);
  }
  c.best_x = c.x;
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
    if (w <= 0.0 || c.V.size() <= i) continue;
    /* The audit tolerance is the dim's ACTUAL proposal scale, so the
     * report means the same thing under both kernels: "x sits within one
     * proposal step of an OPTIM limit".
     *   CORANA:  V[i] is the adaptive half-width; a dim is interesting
     *            only when V[i] has ALSO bottomed out at the
     *            V_FLOOR_FRAC*width floor -- no further shrinkage is
     *            available to work it free.
     *   CLASSIC: V[i] is the (calibrated or fixed) Gaussian sigma;
     *            there is no floor clause because sigma does not shrink
     *            in response to bound pressure. */
    const double v_lo = kb.v_floor_frac * w;
    if (kb.corana && c.V[i] > v_lo * (1.0 + 1e-9))
      continue;   /* step can still shrink */
    const double tol = kb.corana ? v_lo : c.V[i];
    const double g_lo = c.x(i) - sp.low[i];
    const double g_hi = sp.high[i] - c.x(i);
    const bool   hi   = g_hi < g_lo;
    const double gap  = hi ? g_hi : g_lo;
    if (gap <= tol) out.push_back(PinnedDim{i, hi, gap, tol});
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
  if (kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE) z.resize(dim);

  for (std::size_t k = 0; k < kb.chain_length; ++k) {
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
      apply_diagonal_ram(c, z, alpha, sp, kb.step_trust_factor);
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

/* One sweep for ANNEAL_PT: CLASSIC = one whole-vector proposal, CORANA =
 * dim single-coordinate moves with the slot-resident NS-sweep V cadence.
 * PT rejects CHAIN_LENGTH/NT, so the CLASSIC path runs the rung kernel
 * with an internal length of one. */
template<class Vec, class HostT>
RungStats run_sweep(ChainState<Vec>& c, double T,
                    const OptimizerSpec& sp, const Knobs& kb,
                    HostT& host, StopCtx& ctx,
                    std::uint64_t allowance =
                        std::numeric_limits<std::uint64_t>::max())
{
  if (kb.corana) return corana_sweep(c, T, sp, kb, host, ctx, allowance);
  Knobs one = kb;
  one.chain_length = 1;
  return classic_rung(c, T, sp, one, host, ctx, allowance);
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

  if (kb.t_start) {
    out.T0 = *kb.t_start;
    /* Key name matches the kind: the serial schedule's start is T_INIT,
     * the PT/PA hot end is T_HOT. */
    const bool serial = std::string_view(banner_ctx) == "ANNEAL";
    HPOLog::instance().info(std::string(banner_ctx) +
      ": using user-specified " + (serial ? "T_INIT=" : "T_HOT=") +
      std::to_string(out.T0));
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
    const auto t0_est = robust_t0(f_x, probes, sp.failscore, kb.t0_accept);
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
      med_de = out.T0 * -std::log(kb.t0_accept);
    }
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3)
        << banner_ctx << ": T0=" << out.T0
        << " from " << probes.size() << " probes,"
        << "  median|dE|=" << med_de
        << "  loss range=[" << pmin << ", " << pmax << "]"
        << "  std=" << pstd
        << "  target p_accept=" << kb.t0_accept
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

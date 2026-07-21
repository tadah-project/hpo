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
 *   - reflect_into               mirror reflection into [lo, hi]
 *   - loss_is_finite             FAILSCORE-aware finiteness test
 *   - robust_t0                  cliff-robust initial-temperature estimate
 *   - Knobs / Knobs::parse       PARAM parsing, validation, whitelists
 *   - CoranaPlateau              Corana et al. 1987 sec. 4 eps-NEPS stop
 */
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_log.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
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

/* Mirror-reflect v into [lo, hi]. Up to 16 fold-reflections handle
 * steps up to ~16 * width; pathological inputs fall through to a
 * final clamp. Returns true when any reflection/clamp was needed
 * (feeds the `oob` diagnostic counters). */
inline bool reflect_into(double& v, double lo, double hi)
{
  if (v >= lo && v <= hi) return false;
  for (int it = 0; it < 16 && (v < lo || v > hi); ++it) {
    if      (v < lo) v = 2.0 * lo - v;
    else if (v > hi) v = 2.0 * hi - v;
  }
  if (v < lo || v > hi) v = std::min(hi, std::max(lo, v));
  return true;
}

/* FAILSCORE-aware finiteness: a loss is usable for statistics when it
 * is a normal number comfortably below the configured FAILSCORE. The
 * 0.999 margin absorbs failscore values that went through lossy text
 * round-trips. When no failscore is known every finite double counts. */
inline bool loss_is_finite(double f, const std::optional<double>& failscore)
{
  if (!std::isfinite(f)) return false;
  if (failscore) return f < 0.999 * *failscore;
  return true;
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
  std::vector<double> up, absd;
  up.reserve(probes.size());
  absd.reserve(probes.size());
  for (double f : probes) {
    if (!loss_is_finite(f, failscore)) continue;
    const double d = f - f0;
    if (d > 0.0) up.push_back(d);
    absd.push_back(std::fabs(d));
  }
  if (absd.empty()) return std::nullopt;
  std::vector<double>& src = up.empty() ? absd : up;
  std::sort(src.begin(), src.end());
  const std::size_t n = src.size();
  const double med = (n % 2) ? src[n / 2]
                             : 0.5 * (src[n / 2 - 1] + src[n / 2]);
  return med / std::log(1.0 / paccept);
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
   * NS/NT/RT/T0_PACCEPT defaults were re-derived empirically on the
   * 42-D Kr EAM workload at fixed MAXEVAL (2026-07-13 bench,
   * playground_anneal_defaults/): NS5/NT5/RT0.90/PACCEPT0.05 beat the
   * old NS5/NT2/RT0.85/PACCEPT0.80 by ~3 orders of magnitude in median
   * best-at-budget, and beat a hand-tuned T_HOT as well. Deviations
   * from Corana 1987 (NS 20, NT max(100,5D)): a rung costs NS*NT*D
   * evals, so the paper's values are unaffordable at LAMMPS evaluation
   * costs -- NT 5 keeps part of the paper's V-equilibration intent;
   * PACCEPT 0.05 targets the small-dE tail, which is what actually
   * matters on FAILSCORE-cliff landscapes. */
  bool        corana        = false;
  std::size_t L             = 50;     /* classic chain length per rung   */
  std::size_t NS            = 5;      /* Corana step cycles              */
  std::size_t NT            = 5;      /* Corana step-adjust cycles per T */
  double      C             = 2.0;    /* Corana step-adjust gain         */
  double      step_frac     = 0.10;
  double      v_floor_frac  = 1e-3;
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
  std::size_t swap_every  = 1;      /* exchange attempt every N epochs  */
  std::size_t swap_window = 20;     /* windowed swap-acc table every N
                                     * epochs (0 = cumulative only)     */

  /* shared by both parallel drivers */
  int         init_mode   = 0;      /* 0 = seed + sigma-jitter, 1 = box */

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
      "RESET_TO_BEST", "NEPS", "T0_PROBES", "T0_PACCEPT",
      "T_HOT", "T_MIN", "RT",
  };
  static const std::unordered_set<std::string> serial_keys = {
      "REANNEAL", "REANNEAL_SCALE",
  };
  static const std::unordered_set<std::string> pt_keys = {
      "REPLICAS", "T_COLD", "LADDER_SPAN", "SWAP_EVERY", "SWAP_WINDOW",
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

  auto pget = [&](const char* k, double dflt) {
    auto it = sp.param.find(k);
    return it == sp.param.end() ? dflt : it->second;
  };

  Knobs kb;
  kb.corana        = pget("CORANA", 0.0) > 0.5;
  kb.L             = static_cast<std::size_t>(pget("L", 50.0));
  kb.NS            = static_cast<std::size_t>(pget("NS", 5.0));
  kb.NT            = static_cast<std::size_t>(pget("NT", 5.0));
  kb.C             = pget("C", 2.0);
  kb.step_frac     = pget("STEP_FRAC", 0.10);
  kb.v_floor_frac  = pget("V_FLOOR_FRAC", 1e-3);
  /* Brommer reset-to-best defaults ON under CORANA: on FAILSCORE-cliff
   * landscapes the adaptive chain otherwise drifts during the high-T
   * phase with no way back to the deepest basin seen. In PT it defaults
   * OFF and applies only to the coldest slot (hot replicas exist to
   * explore; snapping them to their best defeats tempering). */
  kb.reset_to_best = pget("RESET_TO_BEST",
                          (kb.corana && kind != AlgoKind::PT) ? 1.0 : 0.0) > 0.5;
  kb.neps          = static_cast<std::size_t>(pget("NEPS", 4.0));

  if (sp.param.count("T_HOT")) kb.t_hot = pget("T_HOT", 0.0);
  if (sp.param.count("T_MIN")) kb.t_min = pget("T_MIN", 0.0);
  /* RT: cooling ratio everywhere. PT default 1.0 = static ladder (pure
   * tempering); RT < 1 turns PT into a cooling-ladder hybrid. */
  kb.rt            = pget("RT", kind == AlgoKind::PT ? 1.0 : 0.90);
  kb.t0_probes     = static_cast<std::size_t>(pget("T0_PROBES", 20.0));
  kb.t0_paccept    = pget("T0_PACCEPT", 0.05);

  kb.reanneal       = kind == AlgoKind::Serial && pget("REANNEAL", 1.0) > 0.5;
  kb.reanneal_scale = pget("REANNEAL_SCALE", 0.25);

  kb.replicas    = static_cast<std::size_t>(pget("REPLICAS", 0.0));
  if (sp.param.count("T_COLD")) kb.t_cold = pget("T_COLD", 0.0);
  kb.ladder_span = pget("LADDER_SPAN", 1e3);
  kb.swap_every  = static_cast<std::size_t>(pget("SWAP_EVERY", 1.0));
  if (pget("SWAP_WINDOW", 20.0) < 0.0)
    throw std::runtime_error(ctx + ": PARAM SWAP_WINDOW must be >= 0 "
        "(epochs between windowed swap-acceptance tables; 0 disables them)");
  kb.swap_window = static_cast<std::size_t>(pget("SWAP_WINDOW", 20.0));
  kb.init_mode   = static_cast<int>(pget("INIT_MODE", 0.0));
  kb.sweeps      = static_cast<std::size_t>(pget("SWEEPS", 1.0));
  kb.ess_target  = pget("ESS_TARGET", 0.5);

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
  if (!(kb.step_frac > 0.0))
    throw std::runtime_error(ctx + ": PARAM STEP_FRAC must be > 0");
  if (!(kb.t0_paccept > 0.0 && kb.t0_paccept < 1.0))
    throw std::runtime_error(ctx + ": PARAM T0_PACCEPT must be in (0,1) (got "
        + std::to_string(kb.t0_paccept) + ")");
  if (kb.t_hot && !(*kb.t_hot > 0.0))
    throw std::runtime_error(ctx + ": PARAM T_HOT must be > 0");
  if (kb.t_min && !(*kb.t_min > 0.0))
    throw std::runtime_error(ctx + ": PARAM T_MIN must be > 0");
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
    if (kb.swap_every == 0)
      throw std::runtime_error(ctx + ": PARAM SWAP_EVERY must be >= 1");
    if (kb.rt < 1.0 && kb.t_min && !(*kb.t_min > 0.0))
      throw std::runtime_error(ctx + ": PARAM T_MIN must be > 0");
  }
  if (kind != AlgoKind::Serial && kb.init_mode != 0 && kb.init_mode != 1)
    throw std::runtime_error(
        ctx + ": PARAM INIT_MODE must be 0 (seed-jitter) or 1 (uniform box)");
  if (kind == AlgoKind::PA) {
    if (kb.sweeps == 0)
      throw std::runtime_error(ctx + ": PARAM SWEEPS must be >= 1");
    if (!(kb.ess_target >= 0.0 && kb.ess_target < 1.0))
      throw std::runtime_error(
          ctx + ": PARAM ESS_TARGET must be in [0,1) (0 = geometric RT schedule)");
    if (sp.pop && *sp.pop < 8)
      throw std::runtime_error(ctx + ": POPULATION must be >= 8");
  }

  return kb;
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
  CoranaPlateau(const StopCriteria& stop, std::size_t neps)
      : ftol_rel_(stop.ftol_rel), ftol_abs_(stop.ftol_abs),
        active_(stop.ftol_rel.has_value() || stop.ftol_abs.has_value()),
        neps_(neps)
  {
    if (active_) ring_.reserve(neps_);
  }

  bool active() const { return active_; }

  /* Feed one end-of-rung current value + the running best; returns
   * true when the run should stop. */
  bool update(double f_rung_end, double f_best)
  {
    if (!active_) return false;
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
  std::optional<double> ftol_rel_, ftol_abs_;
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
  const double fmin = *std::min_element(f.begin(), f.end());
  double s = 0.0, s2 = 0.0;
  for (double fi : f) {
    const double w = std::exp(-dbeta * (fi - fmin));
    s += w;
    s2 += w * w;
  }
  return s2 > 0.0 ? (s * s) / s2 : 0.0;
}

/* Largest dbeta in (0, cap] with ESS(dbeta)/N >= target (ESS is
 * monotone non-increasing in dbeta). Returns cap when even the cap
 * satisfies the target (cool at the maximum allowed rate). */
inline double choose_dbeta(const std::vector<double>& f, double target,
                           double dbeta_cap)
{
  const double n = static_cast<double>(f.size());
  if (f.empty() || dbeta_cap <= 0.0) return 0.0;
  if (reweight_ess(f, dbeta_cap) / n >= target) return dbeta_cap;
  double lo = 0.0, hi = dbeta_cap;
  for (int it = 0; it < 60; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (reweight_ess(f, mid) / n >= target) lo = mid;
    else                                    hi = mid;
  }
  return lo;
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
  double wsum = 0.0;
  for (double wi : w) wsum += wi;
  std::size_t i = 0;         /* current parent */
  double cdf = w.empty() ? 0.0 : w[0] / wsum;
  for (std::size_t j = 0; j < n; ++j) {
    const double p = (u + static_cast<double>(j)) / static_cast<double>(n);
    while (p > cdf && i + 1 < n) {
      ++i;
      cdf += w[i] / wsum;
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
  std::size_t by_T = std::numeric_limits<std::size_t>::max();
  if (T_begin < t_min) return 1;   /* schedule would not run; display floor */
  if (rt < 1.0) {
    const double k = std::log(t_min / T_begin) / std::log(rt);   /* >= 0 */
    by_T = static_cast<std::size_t>(std::floor(k)) + 1;
  }
  std::size_t by_budget = std::numeric_limits<std::size_t>::max();
  if (rung_cost > 0 &&
      evals_left != std::numeric_limits<std::uint64_t>::max()) {
    by_budget = static_cast<std::size_t>(
        (evals_left + rung_cost - 1) / rung_cost);
  }
  const std::size_t n = std::min(by_T, by_budget);
  return n == 0 ? 1 : n;
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
  double                       stopval;
};

enum class RungStop { Completed, Budget, Time, StopVal, External };

struct RungStats {
  std::size_t proposed = 0, accepted = 0, oob = 0;
  std::size_t improved = 0;   /* moves that set a new chain personal best */
  RungStop    stop = RungStop::Completed;
};

/* ---- ChainState ------------------------------------------------------
 *
 * One annealing chain (= one PT replica / PA walker / the single serial
 * chain). `V` is the per-dim step vector: adapted by the Corana rung,
 * constant sigma for the classic rung. The RNG is the chain's private
 * stream (make_stream(seed, SALT_CHAIN, chain_index)). */
template<class Vec>
struct ChainState {
  Vec    x;
  double f = std::numeric_limits<double>::infinity();
  std::vector<double> V;
  std::mt19937_64     rng;
  Vec    best_x;
  double best_f = std::numeric_limits<double>::infinity();
};

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
  std::vector<std::size_t> nA(dim, 0);
  std::uniform_real_distribution<double> u_pm(-1.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  bool inner_break = false;

  for (std::size_t nt = 0; nt < kb.NT && !inner_break; ++nt) {
    std::fill(nA.begin(), nA.end(), 0);

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
        double v = c.x(i) + u_pm(c.rng) * c.V[i];
        if (reflect_into(v, sp.low[i], sp.high[i])) ++rs.oob;
        x_prop(i) = v;

        const double f_prop = host(sp.out(x_prop));
        ++rs.proposed;
        const double df = f_prop - c.f;
        const bool accept = (df <= 0.0) || (udist(c.rng) < std::exp(-df / T));
        if (accept) {
          c.x = x_prop;
          c.f = f_prop;
          ++rs.accepted;
          ++nA[i];
          if (c.f < c.best_f) {
            c.best_f = c.f;
            c.best_x = c.x;
            ++rs.improved;
          }
        }
        if (c.f <= ctx.stopval) {
          rs.stop = RungStop::StopVal; inner_break = true; break;
        }
      }
    }

    for (std::size_t i = 0; i < dim; ++i) {
      const double rate_i = static_cast<double>(nA[i]) /
                            static_cast<double>(kb.NS);
      if (rate_i > 0.6) {
        c.V[i] *= 1.0 + kb.C * (rate_i - 0.6) / 0.4;
      } else if (rate_i < 0.4) {
        c.V[i] /= 1.0 + kb.C * (0.4 - rate_i) / 0.4;
      }
      const double w = sp.high[i] - sp.low[i];
      const double v_lo = kb.v_floor_frac * w;
      if (c.V[i] > w)    c.V[i] = w;
      if (c.V[i] < v_lo) c.V[i] = v_lo;
    }
  }
  return rs;
}

/* ---- pinned-dim audit -------------------------------------------------
 *
 * A dim is "pinned" when the chain hugs an OPTIM bound (within one
 * floor-step v_lo = V_FLOOR_FRAC*width of it -- closer than the smallest
 * possible move is indistinguishable from ON the bound) AND V[i] is
 * already clamped at that floor, so the Corana adaptation has no finer
 * step left that could pull the walk off the bound. While V[i] can
 * still shrink the walk may yet leave legitimately -- not pinned.
 * Corana-only by construction (CLASSIC never adapts V). The drivers
 * feed the result to HPOLog::record_pin_state after every rung/epoch;
 * the journal prints the aggregate table with each milestone SUMMARY. */
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
    if (c.V[i] > v_lo * (1.0 + 1e-9)) continue;   /* step can still shrink */
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

    /* Reflection instead of whole-proposal rejection: in high
     * dimensions the joint in-bounds probability collapses near
     * corners (see the serial driver's history). */
    Vec x_prop(dim);
    bool needed_reflect = false;
    for (std::size_t i = 0; i < dim; ++i) {
      double v = c.x(i) + c.V[i] * ndist(c.rng);
      if (reflect_into(v, sp.low[i], sp.high[i])) needed_reflect = true;
      x_prop(i) = v;
    }
    if (needed_reflect) ++rs.oob;

    const double f_prop = host(sp.out(x_prop));
    ++rs.proposed;
    const double df = f_prop - c.f;
    const bool accept = (df <= 0.0) || (udist(c.rng) < std::exp(-df / T));
    if (accept) {
      c.x = x_prop;
      c.f = f_prop;
      ++rs.accepted;
      if (c.f < c.best_f) {
        c.best_f = c.f;
        c.best_x = c.x;
        ++rs.improved;
      }
    }
    if (c.f <= ctx.stopval) { rs.stop = RungStop::StopVal; break; }
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
 * sigma-neighbourhood; see robust_t0 for the statistics. On a
 * FAILSCORE anchor, spends one uniform-box batch to re-anchor (x, f)
 * on the best finite point, then probes locally again; throws when
 * nothing finite exists anywhere. Mutates (x, f) only on re-anchor and
 * keeps (best_x, best_f) consistent. Logs the same T0 line the serial
 * driver always emitted. */
struct T0Outcome {
  double T0 = 0.0;
  bool   re_anchored = false;
  bool   aborted     = false;   /* STOPVAL/MAXTIME fired mid-probe */
  bool   stopval_hit = false;
  std::size_t n_probes = 0;
};

template<class Vec, class HostT>
T0Outcome calibrate_t0(const OptimizerSpec& sp, const Knobs& kb,
                       const std::vector<double>& sigma,
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
  HPOLog::set_eval_probe();

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
      if (!ctx.budget.try_reserve()) return true;
      if (ctx.time_exceeded()) {
        ctx.budget.release();
        out.aborted = true;
        return false;
      }
      Vec xp(dim);
      if (local && kb.corana) {
        xp = centre;
        const std::size_t i = pick_dim(probe_rng);
        double v = centre(i) + u_pm(probe_rng) * sigma[i];
        reflect_into(v, sp.low[i], sp.high[i]);
        xp(i) = v;
      } else {
        for (std::size_t i = 0; i < dim; ++i) {
          double v = local
              ? centre(i) + sigma[i] * ndist(probe_rng)
              : std::uniform_real_distribution<>(sp.low[i], sp.high[i])(probe_rng);
          reflect_into(v, sp.low[i], sp.high[i]);
          xp(i) = v;
        }
      }
      const double f = host(sp.out(xp));
      probes.push_back(f);
      if (f < best_f) { best_f = f; best_x = xp; }
      if (loss_is_finite(f, sp.failscore) && f < bf_probe) {
        bf_probe = f;
        bx_probe = xp;
      }
      if (f <= ctx.stopval) {
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

  /* Probe-loss spread stats for the log line. */
  double pmin = 0.0, pmax = 0.0, pmean = 0.0, pstd = 0.0;
  if (!probes.empty()) {
    pmin = pmax = probes.front();
    double sum = 0.0;
    for (double v : probes) {
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
      sum += v;
    }
    pmean = sum / static_cast<double>(probes.size());
    if (probes.size() >= 2) {
      double sq = 0.0;
      for (double v : probes) { const double d = v - pmean; sq += d*d; }
      pstd = std::sqrt(sq / static_cast<double>(probes.size() - 1));
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
      med_de = out.T0 * std::log(1.0 / kb.t0_paccept);
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
  HPOLog::clear_eval_context();
  return out;
}

}  // namespace anneal
}  // namespace tadah::hpo

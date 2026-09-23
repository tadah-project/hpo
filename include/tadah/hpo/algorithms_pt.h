#pragma once
/* ----- ANNEAL_PT  (parallel tempering / replica exchange) ------------
 *
 * Refs: Swendsen & Wang, PRL 57 (1986); Geyer (1991); Earl & Deem,
 *       PCCP 7 (2005) [review]; Corana et al., ACM TOMS 13:262 (1987)
 *       for the per-replica adaptive stepper; Roberts, Gelman & Gilks,
 *       AoAP 7 (1997), Garthwaite, Fan & Sisson (2010), and Vihola
 *       (2012) motivate the CLASSIC scaling targets/adaptation. The
 *       deterministic even/odd (DEO) pairing follows Okabe et al.,
 *       CPL 335 (2001). Temperature-gap adaptation follows Miasojedow,
 *       Moulines & Vihola (2013) and dynamic temperature selection by
 *       Vousden, Farr & Mandel (2016).
 *
 * K replicas run the shared annealing rung on a power-shaped log ladder
 * T_0 < T_1 < ... < T_{K-1} (journalled 1-based: slot 1 is coldest). It
 * is static by default;
 * optional adaptation moves interiors, and RT < 1 cools it each epoch. Every
 * SWAP_EVERY epochs, adjacent pairs attempt a state exchange with
 * Metropolis probability
 *
 *     p = min(1, exp((beta_i - beta_j) * (f_i - f_j)))     (i colder)
 *
 * A lower-f state found on the hot side moves cold-ward with probability
 * one; reverse exchanges remain possible to preserve detailed balance.
 * Swaps move STATES (x, f), not temperatures: the Corana step vector V is
 * calibrated to the acceptance rate AT its slot's temperature and
 * therefore stays with the slot.
 *
 * Execution: THREADS is the TOTAL width of the shared worker pool (the
 * per-rank CPU-core budget) that actually runs the pc_lammps fan-out and
 * MLIP train/predict; it no longer multiplies OMP_NUM_THREADS. A bounded
 * set of driver threads (min(THREADS, K)) steps the K replicas, but each
 * driver BLOCKS on the shared pool while its eval's scripts run, so the
 * real concurrency is THREADS and one replica's eval alone submits far
 * more tasks than THREADS -- enough to keep the pool saturated. One host,
 * many threads -- HPO_Host's operator() is thread-safe by contract.
 * With REPLICAS fixed, a deterministic objective, and no timing-dependent
 * stop, the same SEED gives the same evaluated set for any THREADS (per-replica
 * RNG streams, driver-side swap coins keyed by epoch, deterministic
 * per-epoch budget slicing). Documented exceptions: MAXTIME truncation and
 * runs cut short by STOPVAL/exceptions are timing-dependent.
 * (OMP_NUM_THREADS independently drives the residual OpenMP kernels --
 * neighbour-list build, data-stats -- not this pool.)
 *
 * Ladder construction: T_HOT from PARAM T_HOT or the shared T0
 * calibration (T0_PACCEPT is assigned to the calibration delta at the hot end);
 * T_COLD from PARAM T_COLD or T_HOT / LADDER_SPAN. LADDER_POWER shapes
 * normalized log-temperature positions (1 = geometric). Optional
 * LADDER_ADAPT redistributes positive interior log gaps to equalize
 * adjacent-pair swap rates while preserving both endpoints. The commonly
 * cited 20-40% per-pair acceptance range is a starting heuristic, not a
 * workload-independent target: fixed endpoints and replica count control
 * the absolute rate. RT (default 1.0) < 1 additionally cools the WHOLE
 * ladder each epoch (PT -> SA hybrid); the run then ends when the hot
 * end crosses T_MIN.
 *
 * PT-specific PARAMs (shared stepper knobs as in ANNEAL):
 *   REPLICAS     max(THREADS,2)  ladder size K (default: one replica per
 *                                core; set REPLICAS to decouple K from the
 *                                THREADS core budget)
 *   T_HOT        auto            hottest slot
 *   T_COLD       T_HOT/LADDER_SPAN  coldest slot
 *   LADDER_SPAN  1e3             T_HOT/T_COLD when T_COLD unset
 *   SWAP_EVERY   1               exchange attempt every N epochs
 *   SWAP_WINDOW  20              windowed per-pair swap-acceptance table
 *                                every N epochs (0 = cumulative only)
 *   LADDER_POWER 1.0             initial normalized log-ladder power
 *   LADDER_ADAPT 0               adapt interior temperatures in production
 *   LADDER_ADAPT_EVERY 20        attempts at every pair before an update
 *   LADDER_ADAPT_GAIN 1.0        initial stochastic gain in (0,1]
 *   LADDER_ADAPT_DECAY 0.6       diminishing-gain exponent in (0.5,1]
 *   LADDER_ADAPT_UPDATES 0       maximum updates (0 = ongoing)
 *   RT           1.0             ladder cooling per epoch ((0,1])
 *   STEP_AUTO    1               CLASSIC-only coordinate calibration;
 *                                explicit STEP disables it, 0 is legacy fixed
 *   STEP_AUTO_SWEEPS 12          minimum coordinate attempts (up to 2x)
 *   STEP_ADAPT_EVERY 0           production scalar+diagonal RAM apply cadence
 *                                per slot; 0 freezes after warm-up
 *   INIT_MODE    0               0: automatic CLASSIC starts every slot at
 *                                the valid warm seed; fixed/Corana use seed
 *                                plus Gaussian jitter in higher slots
 *                                1: chains>0 drawn uniformly in the box
 *   RESET_TO_BEST 0              when 1, snaps ONLY the coldest slot to
 *                                the global best after each epoch
 */
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_jitter.h>
#include <tadah/hpo/corana_census.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/warmstart_cache.h>   // current_chain() bridge

#ifdef TADAH_ENABLE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tadah::hpo {

namespace pt_detail {

#ifdef TADAH_HPO_MPI
/* PT broadcasts a calibration packet containing T_hot, f0, and the full
 * parameter vector (dim + 2 doubles), then exchanges dim doubles at ladder
 * block boundaries.  Reject an unrepresentable packet before allocating any
 * dimension-sized driver state or entering a collective. */
inline void validate_mpi_payload_dimension(std::size_t dim)
{
  const std::size_t mpi_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  constexpr std::size_t calibration_fields = 2;
  if (dim > mpi_limit - calibration_fields)
    throw std::length_error(
        "TADAH:ANNEAL_PT: parameter dimension " + std::to_string(dim) +
        " exceeds the MPI payload limit (dimension + 2 must fit in int)");
}
#endif

/* Reconstruct an endpoint-preserving temperature ladder from normalized
 * positions in log-temperature space.  nextafter bounds reserve at least
 * one representable double for every remaining slot; this is both the
 * numerical gap floor used by adaptation and the strict-monotonicity guard
 * for very narrow or extreme finite endpoint spans. */
inline std::vector<double> ladder_from_log_positions(
    double cold, double hot, const std::vector<long double>& position)
{
  if (!std::isfinite(cold) || !std::isfinite(hot) ||
      !(cold > 0.0) || !(hot > cold))
    throw std::invalid_argument(
        "ANNEAL_PT ladder requires finite 0 < T_COLD < T_HOT");
  if (position.size() < 2 || position.front() != 0.0L ||
      position.back() != 1.0L)
    throw std::invalid_argument(
        "ANNEAL_PT ladder positions require endpoints 0 and 1");
  for (std::size_t k = 1; k < position.size(); ++k) {
    if (!std::isfinite(position[k]) || position[k] < position[k - 1] ||
        position[k] > 1.0L)
      throw std::invalid_argument(
          "ANNEAL_PT ladder positions must be finite and nondecreasing");
  }

  const std::size_t K = position.size();
  std::vector<double> upper(K, hot);
  for (std::size_t k = K - 1; k > 1; --k)
    upper[k - 1] = std::nextafter(upper[k], cold);
  if (K > 2 && !(upper[1] > cold))
    throw std::runtime_error(
        "ANNEAL_PT ladder endpoints cannot represent the requested number "
        "of strictly increasing temperatures");

  const long double log_cold = std::log(static_cast<long double>(cold));
  const long double log_hot = std::log(static_cast<long double>(hot));
  const long double span = log_hot - log_cold;
  std::vector<double> temperature(K, 0.0);
  temperature.front() = cold;
  temperature.back() = hot;
  for (std::size_t k = 1; k + 1 < K; ++k) {
    const double lower = std::nextafter(temperature[k - 1], hot);
    if (lower > upper[k])
      throw std::runtime_error(
          "ANNEAL_PT ladder numerical gap floor exhausted the endpoint span");
    const double ideal = static_cast<double>(
        std::exp(log_cold + span * position[k]));
    temperature[k] = std::clamp(ideal, lower, upper[k]);
  }
  return temperature;
}

/* Power-shaped initial ladder.  power==1 follows the former geometric
 * arithmetic exactly, preserving legacy trajectories when no new control is
 * supplied. */
inline std::vector<double> make_temperature_ladder(
    double cold, double hot, std::size_t replicas, double power = 1.0)
{
  if (replicas < 2)
    throw std::invalid_argument("ANNEAL_PT ladder requires at least 2 replicas");
  if (!std::isfinite(power) || !(power > 0.0))
    throw std::invalid_argument("ANNEAL_PT LADDER_POWER must be finite and > 0");
  std::vector<long double> position(replicas, 0.0L);
  position.back() = 1.0L;
  for (std::size_t k = 1; k + 1 < replicas; ++k) {
    const long double fraction = static_cast<long double>(k) /
                                 static_cast<long double>(replicas - 1);
    position[k] = power == 1.0
        ? fraction
        : std::pow(fraction, static_cast<long double>(power));
  }
  return ladder_from_log_positions(cold, hot, position);
}

inline long double ladder_log_gap(double cold, double hot)
{
  return std::log(static_cast<long double>(hot)) -
         std::log(static_cast<long double>(cold));
}

struct LadderAdaptation {
  std::vector<double> temperature;
  std::vector<double> acceptance;
  std::vector<long double> old_log_gap;
  std::vector<long double> new_log_gap;
  std::vector<char> at_floor;   /* pair held at the LADDER_MIN_GAP floor */
  double gain = 0.0;
};

/* One diminishing stochastic-approximation update.  High-acceptance gaps
 * expand and low-acceptance gaps contract, after which all positive gaps are
 * renormalized to the fixed endpoint span.
 *
 * D26: min_gap_frac > 0 additionally floors every interior log-gap at
 * min_gap_frac * (span / (K-1)).  Without it the controller keeps
 * contracting any pair whose acceptance is persistently below the mean, and
 * a pair whose low acceptance is caused by a basin offset rather than by
 * temperature spacing (acceptance gap-INSENSITIVE) is driven towards the
 * 1-ULP representability guard: an adapted ladder can reach adjacent-rung ratios of
 * 1.006-1.010 this way, en route to the D21 degenerate-pair failure (equal
 * rungs exchange unconditionally, reporting 100% acceptance with zero
 * tempering).  The floor bounds that contraction; the pairs it holds are
 * reported in at_floor.  See docs/annealing_implementation_audit_2026-08-09.md. */
inline LadderAdaptation adapt_temperature_ladder(
    const std::vector<double>& temperature,
    const std::vector<std::size_t>& attempts,
    const std::vector<std::size_t>& accepted,
    std::size_t update, double initial_gain, double decay,
    double min_gap_frac = 0.0)
{
  const std::size_t K = temperature.size();
  if (K < 3 || attempts.size() + 1 != K || accepted.size() != attempts.size())
    throw std::invalid_argument(
        "ANNEAL_PT ladder adaptation requires K>=3 and one counter per pair");
  if (update == 0 || !std::isfinite(initial_gain) ||
      !(initial_gain > 0.0 && initial_gain <= 1.0) ||
      !std::isfinite(decay) || !(decay > 0.5 && decay <= 1.0))
    throw std::invalid_argument("ANNEAL_PT ladder adaptation controls are invalid");
  if (!std::isfinite(min_gap_frac) ||
      !(min_gap_frac >= 0.0 && min_gap_frac < 1.0))
    throw std::invalid_argument(
        "ANNEAL_PT ladder min-gap fraction must be in [0,1)");

  LadderAdaptation out;
  out.acceptance.resize(K - 1);
  out.old_log_gap.resize(K - 1);
  long double mean = 0.0L;
  for (std::size_t p = 0; p + 1 < K; ++p) {
    if (!std::isfinite(temperature[p]) || !(temperature[p] > 0.0) ||
        !std::isfinite(temperature[p + 1]) ||
        !(temperature[p + 1] > temperature[p]) || attempts[p] == 0 ||
        accepted[p] > attempts[p])
      throw std::invalid_argument(
          "ANNEAL_PT ladder adaptation requires a positive increasing "
          "ladder and valid nonzero pair counters");
    out.acceptance[p] = static_cast<double>(accepted[p]) /
                        static_cast<double>(attempts[p]);
    mean += static_cast<long double>(out.acceptance[p]);
    out.old_log_gap[p] = ladder_log_gap(
        temperature[p], temperature[p + 1]);
    if (!(out.old_log_gap[p] > 0.0L))
      throw std::runtime_error("ANNEAL_PT ladder log gap is not positive");
  }
  mean /= static_cast<long double>(K - 1);
  out.gain = initial_gain * std::pow(static_cast<double>(update), -decay);

  const long double max_change = std::log(2.0L);
  std::vector<long double> weight(K - 1, 0.0L);
  long double weight_sum = 0.0L;
  for (std::size_t p = 0; p + 1 < K; ++p) {
    const long double change = std::clamp(
        static_cast<long double>(out.gain) *
            (static_cast<long double>(out.acceptance[p]) - mean),
        -max_change, max_change);
    weight[p] = out.old_log_gap[p] * std::exp(change);
    weight_sum += weight[p];
  }
  out.at_floor.assign(K - 1, 0);
  std::vector<long double> position(K, 0.0L);
  if (min_gap_frac > 0.0) {
    /* Clamp-and-redistribute waterfilling in log-gap space.  Every clamped
     * pair takes exactly G; free pairs share the remaining span in
     * proportion to their weights.  Termination: if every free pair fell
     * below G at once, the whole span would be below (K-1)*G =
     * min_gap_frac*S < S -- impossible -- so each pass either finishes or
     * clamps at least one pair while at least one stays free; at most K-2
     * passes run. */
    const long double S =
        ladder_log_gap(temperature.front(), temperature.back());
    const long double G = static_cast<long double>(min_gap_frac) * S /
                          static_cast<long double>(K - 1);
    std::vector<long double> gap(K - 1, 0.0L);
    for (;;) {
      std::size_t n_clamped = 0;
      long double free_weight = 0.0L;
      for (std::size_t p = 0; p + 1 < K; ++p) {
        if (out.at_floor[p]) ++n_clamped;
        else free_weight += weight[p];
      }
      const long double remaining =
          S - static_cast<long double>(n_clamped) * G;
      bool clamped_more = false;
      for (std::size_t p = 0; p + 1 < K; ++p) {
        if (out.at_floor[p]) { gap[p] = G; continue; }
        gap[p] = weight[p] / free_weight * remaining;
        if (gap[p] < G) { out.at_floor[p] = 1; clamped_more = true; }
      }
      if (!clamped_more) break;
    }
    long double cumulative = 0.0L;
    for (std::size_t p = 0; p + 2 < K; ++p) {
      cumulative += gap[p];
      position[p + 1] = cumulative / S;
    }
  } else {
    long double cumulative = 0.0L;
    for (std::size_t p = 0; p + 2 < K; ++p) {
      cumulative += weight[p];
      position[p + 1] = cumulative / weight_sum;
    }
  }
  position.back() = 1.0L;
  out.temperature = ladder_from_log_positions(
      temperature.front(), temperature.back(), position);
  out.new_log_gap.resize(K - 1);
  for (std::size_t p = 0; p + 1 < K; ++p)
    out.new_log_gap[p] = ladder_log_gap(
        out.temperature[p], out.temperature[p + 1]);
  return out;
}

/* D26 (revised after review): pooled evidence for the equalization freeze.
 *
 * The original per-window streak test had two confirmed defects: a
 * zero-information window (every pair exactly 0% or 100% -- e.g. an
 * equal-loss plateau, where accept_exchange accepts unconditionally) has
 * sigma = 0 and dev = 0 and counted as "settled", so three transient
 * plateau windows latched the freeze permanently; and at the default
 * 20-attempt window the 2-sigma tolerance is ~+/-19 points, so a
 * genuinely unequal ladder could pass three noisy windows in a row.
 *
 * This version accumulates: windows carrying no acceptance signal are
 * HELD (not pooled, not reset), settled windows pool their counters so
 * the test gains power instead of re-rolling the same noise, and any
 * unsettled pooled test clears the evidence.  Freeze fires only when
 * every pair is settled on the POOLED statistics and every pair has at
 * least `min_pooled_attempts` pooled attempts. */
struct LadderFreezeEvidence {
  std::vector<std::size_t> attempt;
  std::vector<std::size_t> accept;
};

inline bool ladder_freeze_step(LadderFreezeEvidence& ev,
                               const std::vector<std::size_t>& attempts,
                               const std::vector<std::size_t>& accepted,
                               const std::vector<char>& at_floor,
                               std::size_t min_pooled_attempts)
{
  const std::size_t P = attempts.size();
  if (P == 0 || accepted.size() != P || at_floor.size() != P)
    throw std::invalid_argument(
        "ANNEAL_PT ladder freeze evidence shape mismatch");
  if (ev.attempt.size() != P) {
    ev.attempt.assign(P, 0);
    ev.accept.assign(P, 0);
  }
  std::size_t win_att = 0, win_acc = 0;
  for (std::size_t p = 0; p < P; ++p) {
    win_att += attempts[p];
    win_acc += accepted[p];
  }
  if (win_att == 0) return false;
  if (win_acc == 0 || win_acc == win_att) return false;  /* hold */

  for (std::size_t p = 0; p < P; ++p) {
    ev.attempt[p] += attempts[p];
    ev.accept[p] += accepted[p];
  }
  long double tot_att = 0.0L, tot_acc = 0.0L;
  std::size_t min_att = std::numeric_limits<std::size_t>::max();
  for (std::size_t p = 0; p < P; ++p) {
    tot_att += static_cast<long double>(ev.attempt[p]);
    tot_acc += static_cast<long double>(ev.accept[p]);
    min_att = std::min(min_att, ev.attempt[p]);
  }
  /* The hold above guarantees at least one pooled window with
   * 0 < acc < att, so 0 < mean < 1 and sigma > 0 for every pair. */
  const long double mean = tot_acc / tot_att;
  bool all_settled = true;
  for (std::size_t p = 0; p < P; ++p) {
    const long double n = static_cast<long double>(ev.attempt[p]);
    const long double a = static_cast<long double>(ev.accept[p]) / n;
    const long double sigma =
        std::sqrt(std::max<long double>(0.0L, mean * (1.0L - mean) / n));
    const bool converged = std::fabs(a - mean) <= 2.0L * sigma;
    const bool floor_limited = at_floor[p] && a < mean;
    if (!(converged || floor_limited)) {
      all_settled = false;
      break;
    }
  }
  if (!all_settled) {
    std::fill(ev.attempt.begin(), ev.attempt.end(), std::size_t{0});
    std::fill(ev.accept.begin(), ev.accept.end(), std::size_t{0});
    return false;
  }
  return min_att >= min_pooled_attempts;
}

/* Replica-exchange decision for adjacent slots, `cold_temperature <=
 * hot_temperature`. Resolve deterministic limits before forming inverse
 * temperatures: cooling may legitimately underflow ladder endpoints to zero,
 * while equal temperatures/losses make the formal beta-gap product
 * indeterminate even though the exchange probability is one.
 *
 * For a genuine uphill exchange, compare in the log domain:
 *
 *   coin < exp(-m),
 *   m = (f_hot - f_cold) * (T_hot - T_cold) / (T_cold * T_hot).
 *
 * This is algebraically the usual replica-exchange rule for finite positive
 * temperatures, but it cannot create `inf - inf` or `inf * 0`. Normalized
 * unusable losses are represented by infinities; their ordering has the
 * corresponding zero-temperature limit. A coin is drawn only when the
 * decision is genuinely probabilistic, matching the annealing core RNG
 * discipline.
 */
template<class DrawCoin>
inline bool accept_exchange(double cold_temperature, double hot_temperature,
                            double cold_loss, double hot_loss,
                            DrawCoin&& draw_coin)
{
  if (!std::isfinite(cold_temperature) ||
      !std::isfinite(hot_temperature) || cold_temperature < 0.0 ||
      hot_temperature < cold_temperature)
    throw std::invalid_argument(
        "ANNEAL_PT exchange requires finite 0 <= T_cold <= T_hot");
  if (std::isnan(cold_loss) || std::isnan(hot_loss))
    throw std::invalid_argument(
        "ANNEAL_PT exchange requires losses normalized away from NaN");

  /* Equal beta or equal energy gives exponent zero, including 0/0 and
   * infinity/infinity limiting cases. */
  if (cold_temperature == hot_temperature || cold_loss == hot_loss)
    return true;

  /* Moving a strictly lower loss into the colder slot is always accepted.
   * This comparison also handles a normalized +/-infinity without doing
   * arithmetic on it. */
  if (hot_loss < cold_loss) return true;

  /* The only remaining ordering is f_cold < f_hot. At zero cold
   * temperature, or with an infinite endpoint loss, the reverse exchange has
   * probability zero. */
  if (cold_temperature == 0.0 || !std::isfinite(cold_loss) ||
      !std::isfinite(hot_loss))
    return false;

  const double coin = draw_coin();
  if (!std::isfinite(coin) || coin < 0.0 || !(coin < 1.0))
    throw std::invalid_argument(
        "ANNEAL_PT exchange coin must be finite and in [0,1)");

  const long double energy_gap = static_cast<long double>(hot_loss) -
                                 static_cast<long double>(cold_loss);
  const long double temperature_gap =
      static_cast<long double>(hot_temperature) -
      static_cast<long double>(cold_temperature);
  const long double log_m = std::log(energy_gap) +
                            std::log(temperature_gap) -
                            std::log(static_cast<long double>(cold_temperature)) -
                            std::log(static_cast<long double>(hot_temperature));

  /* coin==0 is below every positive, finite mathematical probability. An
   * infinite m is the exact zero-probability limit and remains rejected. */
  if (coin == 0.0) return !std::isinf(log_m);
  const long double log_threshold =
      std::log(-std::log(static_cast<long double>(coin)));
  return log_m < log_threshold;
}

/* D21: hard diagnostic thresholds, named in one place and stated once in
 * the journal's startup diagnostics block. They shape WARNINGs only --
 * never the sampling or adaptation itself (with one exception: degenerate
 * equal-loss exchanges are excluded from ladder-adaptation evidence). */
constexpr double      kStepHealthAcceptLo   = 0.05;
constexpr double      kStepHealthAcceptHi   = 0.50;
constexpr std::size_t kStepHealthStreak     = 2;    /* windows */
constexpr std::size_t kTransportWindowFactor = 100; /* x (K-1) rounds */
constexpr std::size_t kLadderFreezeMinAttempts = 100; /* pooled/pair */
/* A window whose swap acceptance is mostly degenerate exchanges carries
 * (almost) no tempering information; warn above this fraction. */
constexpr double      kDegenerateWarnFrac   = 0.5;

/* D29: render a per-slot count grid on the standard [lo-hi] rows-of-8
 * layout (slots absent from `entries` print 0). */
inline std::string format_slot_count_table(
    const std::string& header, std::size_t K,
    const std::vector<std::pair<int, std::size_t>>& entries)
{
  std::map<int, std::size_t> cnt(entries.begin(), entries.end());
  /* `k` stays 0-based: it is the key of HPOLog's best_slot_* maps. */
  return logidx::grid(header, K, [&](std::ostringstream& o, std::size_t k) {
    const auto it = cnt.find(static_cast<int>(k));
    o << "  " << std::setw(9) << (it != cnt.end() ? it->second : 0);
  });
}

/* D29: render the att/acc per-pair percentage table ("-" = pair not yet
 * attempted), rows of 8, att/pair resolution stated in the header. */
inline std::string format_pair_table(
    std::size_t K, const std::string& label,
    const std::vector<std::size_t>& attv,
    const std::vector<std::size_t>& accv)
{
  std::size_t amin = std::numeric_limits<std::size_t>::max();
  std::size_t amax = 0;
  for (std::size_t p = 0; p + 1 < K; ++p) {
    amin = std::min(amin, attv[p]);
    amax = std::max(amax, attv[p]);
  }
  std::ostringstream head;
  head << "ANNEAL_PT swap acc% per pair (" << label << ", att/pair " << amin;
  if (amax != amin) head << "-" << amax;
  head << "; pair p = slots p<->p+1):";
  /* K-1 pairs; the pair family has its own label width. */
  return logidx::grid(head.str(), K - 1,
                      [&](std::ostringstream& o, std::size_t p) {
    if (attv[p] == 0)
      o << "  " << std::setw(9) << "-";
    else
      o << "  " << std::setw(9)
        << std::lround(100.0 * static_cast<double>(accv[p]) /
                       static_cast<double>(attv[p]));
  });
}

enum StepStatus : unsigned {
  StepOk            = 0,
  StepNumericalFloor = 1u << 0,
  StepSupportCapped  = 1u << 1,
  StepUnconverged    = 1u << 2,
  StepInterrupted    = 1u << 3,
  /* D28: appended bits only -- the values cross the MPI gather as doubles
   * and appear in hpo_step_scales.tsv, so existing bits stay stable.
   * BracketOk: the two-consecutive-batches rule never certified the dim,
   * but the bisection bracket had tightened enough that the in-band scale
   * provably lies within a small factor -- calibrated, not unexplained.
   * Insensitive/Hypersensitive: attempt-cap exhaustion with the best
   * tested batch still at alpha ~ 1 (objective does not respond to this
   * coordinate) or alpha ~ 0 (acceptance collapses at every tested
   * scale); both accompany StepUnconverged and name its cause. */
  StepBracketOk      = 1u << 4,
  StepInsensitive    = 1u << 5,
  StepHypersensitive = 1u << 6,
};

struct DimCalibration {
  std::size_t attempts = 0;
  std::size_t accepted = 0;
  std::size_t unusable = 0;
  std::size_t backoffs = 0;
  unsigned status = StepOk;

  /* The fixed-scale batch selected for production and diagnostics. */
  std::size_t validation_attempts = 0;
  std::size_t validation_accepted = 0;
  double validation_mean_alpha = 0.0;

  /* Persistent state permits deterministic continuation after a fair-share
   * budget round ends part-way through a batch. */
  std::size_t batch_attempts = 0;
  std::size_t batch_accepted = 0;
  std::size_t batch_unusable = 0;
  long double batch_alpha_sum = 0.0L;
  double batch_scale = 0.0;
  std::size_t target_batches = 0;
  bool have_lower_scale = false;
  bool have_upper_scale = false;
  double lower_scale = 0.0;
  double upper_scale = 0.0;
  bool have_best = false;
  double best_scale = 0.0;
  double best_score = std::numeric_limits<double>::infinity();
  bool finished = false;
  /* D9: the sigma the COORDINATE phase settled on, kept because the pilot
   * immediately overwrites c.V[i] with pilot_base*multiplier. Every other
   * column the diagnostics file reports for this dimension (acceptance,
   * status, mean_alpha, attempts, backoffs, unusable) was measured at THIS
   * scale, not at the production one, and the two differ by
   * sqrt(dim)/pilot_multiplier -- about 26x in one observed configuration. Reporting them on
   * one row against the production sigma made rows self-contradictory:
   * every "support-capped" row means the coordinate scale reached 1.0 x
   * width, yet printed sigma/width = 0.0379. */
  double coordinate_scale = 0.0;
};

struct SlotCalibration {
  /* Dedicated warm-up RNG stream (SALT_WARMUP, keyed by global slot):
   * calibration draws must not touch the chain's production stream, so
   * toggling STEP_MODE never shifts the production proposal sequence
   * for a given SEED. Seeded by the driver before the first phase. */
  std::mt19937_64 rng;
  std::vector<DimCalibration> dims;
  std::size_t coordinate_proposed = 0;
  std::size_t coordinate_accepted = 0;
  std::size_t pilot_proposed = 0;
  std::size_t pilot_accepted = 0;
  std::size_t pilot_unusable = 0;
  std::size_t pilot_backoffs = 0;
  double pilot_multiplier = 1.0;
  std::size_t pilot_validation_attempts = 0;
  std::size_t pilot_validation_accepted = 0;
  double pilot_validation_mean_alpha = 0.0;
  unsigned pilot_status = StepOk;

  std::vector<double> pilot_base;
  double pilot_multiplier_min = 0.0;
  double pilot_multiplier_max = std::numeric_limits<double>::infinity();
  std::size_t pilot_window_attempts = 0;
  std::size_t pilot_window_accepted = 0;
  std::size_t pilot_window_unusable = 0;
  long double pilot_window_alpha_sum = 0.0L;
  std::size_t pilot_target_windows = 0;
  bool pilot_have_lower = false;
  bool pilot_have_upper = false;
  double pilot_lower = 0.0;
  double pilot_upper = 0.0;
  bool pilot_have_best = false;
  double pilot_best_multiplier = 1.0;
  double pilot_best_score = std::numeric_limits<double>::infinity();
  /* D12: did the pilot ever actually move the multiplier off its 1.0
   * start? On a flat or noisy score surface the strict `score <
   * pilot_best_score` comparison lets window 1 win outright, so an
   * `unconverged` pilot restores 1.0 -- the initial value -- and the log
   * ("pilot_attempts=200 pilot_multiplier=1.000e+00 pilot_status=unconverged")
   * is indistinguishable from a pilot that genuinely converged there. Two
   * slots have been observed spending 200 host evaluations each to arrive back at the
   * default with nothing saying so. */
  bool pilot_multiplier_moved = false;
  bool pilot_finished = false;
  anneal::RungStop stop = anneal::RungStop::Completed;
};

inline bool coordinate_converged(const DimCalibration& d,
                                 std::size_t minimum_attempts)
{
  return d.finished && d.attempts >= minimum_attempts &&
         d.status == StepOk;
}

inline double clamp_coordinate_scale(double scale, double x,
                                     double lo, double hi,
                                     unsigned& status)
{
  const double floor = anneal::resolvable_coordinate_scale(x, lo, hi);
  const double cap = hi - lo;
  if (!std::isfinite(scale) || scale < floor) {
    status |= StepNumericalFloor;
    return floor;
  }
  if (scale > cap) {
    status |= StepSupportCapped;
    return cap;
  }
  return scale;
}

template<class Vec, class HostT>
void coordinate_warmup(
    anneal::ChainState<Vec>& c, double T, const OptimizerSpec& sp,
    const anneal::Knobs& kb, HostT& host, anneal::StopCtx& ctx,
    std::uint64_t allowance, SlotCalibration& out)
{
  constexpr double target = anneal::kTargetAcceptCoord;
  constexpr double target_lo = target - 0.15;
  constexpr double target_hi = target + 0.15;
  /* Gross-miss thresholds (also the D28 sensitivity-classification
   * boundaries) and the bracket-acceptance factor: a two-sided bracket
   * with upper <= 4x lower puts the in-band scale within 2x of either
   * edge, each of which measured on the wrong side of the band. */
  constexpr double gross_low_alpha = 0.01;
  constexpr double gross_high_alpha = 0.9;
  constexpr double bracket_ok_factor = 4.0;
  const std::size_t dim = sp.low.size();
  const std::size_t minimum = kb.step_warmup_sweeps;
  const std::size_t maximum = anneal::saturating_add(minimum, minimum);
  const std::size_t batch_size = std::min<std::size_t>(4, minimum);
  if (out.dims.empty()) out.dims.resize(dim);
  out.stop = anneal::RungStop::Completed;
  for (auto& dc : out.dims)
    dc.status &= ~StepInterrupted;
  std::normal_distribution<double> ndist(0.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  std::vector<std::size_t> active;
  active.reserve(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    const auto& dc = out.dims[i];
    if (!dc.finished && dc.attempts < maximum)
      active.push_back(i);
  }
  const std::size_t proposed_at_entry = out.coordinate_proposed;

  auto select_batch = [&](DimCalibration& dc, double scale,
                          std::size_t attempts, std::size_t accepted,
                          double mean_alpha) {
    dc.best_scale = scale;
    dc.validation_attempts = attempts;
    dc.validation_accepted = accepted;
    dc.validation_mean_alpha = mean_alpha;
  };

  auto close_batch = [&](std::size_t i) {
    DimCalibration& dc = out.dims[i];
    if (dc.batch_attempts == 0) return;
    const double scale = dc.batch_scale;
    const double mean_alpha = static_cast<double>(
        dc.batch_alpha_sum / static_cast<long double>(dc.batch_attempts));
    const double score = std::fabs(mean_alpha - target) +
                         (dc.batch_unusable > 0 ? 1.0 : 0.0);
    if (!dc.have_best || score < dc.best_score) {
      dc.have_best = true;
      dc.best_score = score;
      select_batch(dc, scale, dc.batch_attempts, dc.batch_accepted,
                   mean_alpha);
    }

    const bool usable = dc.batch_unusable == 0;
    const bool in_target = usable && mean_alpha >= target_lo &&
                           mean_alpha <= target_hi;
    bool change_scale = !in_target;
    const double floor = anneal::resolvable_coordinate_scale(
        c.x(i), sp.low[i], sp.high[i]);
    const double cap = sp.high[i] - sp.low[i];
    const bool at_floor = scale <= floor * (1.0 + 1e-12);
    const bool at_cap = scale >= cap * (1.0 - 1e-12);

    if (in_target) {
      ++dc.target_batches;  /* the scale is deliberately unchanged */
      if (dc.attempts >= minimum && dc.target_batches >= 2) {
        dc.finished = true;
        dc.status = StepOk;
        c.V[i] = scale;
        dc.coordinate_scale = scale;   /* D9 */
        select_batch(dc, scale, dc.batch_attempts, dc.batch_accepted,
                     mean_alpha);
      }
    } else {
      const bool had_target_batch = dc.target_batches > 0;
      dc.target_batches = 0;
      if ((!usable || mean_alpha < target_lo) && at_floor &&
          dc.attempts >= minimum) {
        dc.finished = true;
        dc.status = StepNumericalFloor;
        c.V[i] = scale;
        dc.coordinate_scale = scale;   /* D9 */
        select_batch(dc, scale, dc.batch_attempts, dc.batch_accepted,
                     mean_alpha);
      } else if (usable && mean_alpha > target_hi && at_cap &&
                 dc.attempts >= minimum) {
        dc.finished = true;
        dc.status = StepSupportCapped;
        c.V[i] = scale;
        dc.coordinate_scale = scale;   /* D9 */
        select_batch(dc, scale, dc.batch_attempts, dc.batch_accepted,
                     mean_alpha);
      } else {
        /* One ordinary miss after a valid window is treated as validation
         * noise and retried at the identical scale. Gross misses still move
         * immediately. This preserves the two-consecutive-window rule while
         * avoiding a bracket jump after one unlucky four-sample batch. */
        const bool gross_miss = !usable || mean_alpha < gross_low_alpha ||
                                mean_alpha > gross_high_alpha;
        change_scale = !had_target_batch || gross_miss;
        if (change_scale) {
          if (!usable || mean_alpha < target_lo) {
            dc.have_upper_scale = true;
            dc.upper_scale = scale;
          } else {
            dc.have_lower_scale = true;
            dc.lower_scale = scale;
          }
        }
      }
    }

    if (!dc.finished && dc.attempts >= maximum) {
      dc.finished = true;
      if (dc.have_lower_scale && dc.have_upper_scale &&
          dc.lower_scale < dc.upper_scale &&
          dc.upper_scale <= bracket_ok_factor * dc.lower_scale) {
        /* D28 bracket-ok: the noisy 4-sample in-band test never fired
         * twice in a row, but the bisection bracket is tight -- the
         * geometric mean is within sqrt(bracket_ok_factor) of both
         * edges, each of which measured on the wrong side of the band.
         * a long run once finished a third of its dims "unconverged/unexplained" under
         * the old rule. */
        unsigned clamp_status = StepOk;
        c.V[i] = clamp_coordinate_scale(
            std::sqrt(dc.lower_scale * dc.upper_scale),
            c.x(i), sp.low[i], sp.high[i], clamp_status);
        dc.status = StepBracketOk | clamp_status;
        /* D9 contract: the TSV's coord_sigma column is the scale the
         * row's acceptance/alpha measurements were taken at.  Those
         * came from the best tested batch, not from the (never
         * sampled) bracket midpoint that seeds production in c.V[i]. */
        dc.coordinate_scale = dc.have_best ? dc.best_scale : c.V[i];
      } else {
        dc.status = StepUnconverged;
        /* D28: name the cause when the best tested batch pins it. */
        if (dc.have_best) {
          if (dc.validation_mean_alpha > gross_high_alpha)
            dc.status |= StepInsensitive;
          else if (dc.validation_mean_alpha < gross_low_alpha)
            dc.status |= StepHypersensitive;
          c.V[i] = dc.best_scale;
        }
        dc.coordinate_scale = c.V[i];   /* D9: best tested, or the last */
      }
    } else if (!dc.finished && change_scale) {
      double next_scale = scale;
      if (dc.have_lower_scale && dc.have_upper_scale &&
          dc.lower_scale < dc.upper_scale) {
        next_scale = std::sqrt(dc.lower_scale * dc.upper_scale);
      } else if (!usable || mean_alpha < target_lo) {
        next_scale /= (!usable || mean_alpha < gross_low_alpha) ? 10.0 : 2.0;
      } else {
        next_scale *= mean_alpha > gross_high_alpha ? 10.0 : 2.0;
      }
      unsigned clamp_status = StepOk;
      const double clamped_scale = clamp_coordinate_scale(
          next_scale, c.x(i), sp.low[i], sp.high[i], clamp_status);
      if (clamped_scale < scale) ++dc.backoffs;
      /* D8: keep the reason the scale was clamped. It used to be computed
       * here and dropped, so an interim floor/cap saturation left no trace:
       * a dim that spent the whole search pressed against the resolvable
       * floor still finished labelled a bare "unconverged", and
       * unexplained_dimensions counted it as inexplicable. One run reported
       * numerical-floor for exactly 0 of 5,568 dims while 1,882 came back
       * unconverged -- the two facts were the same fact. */
      dc.status |= clamp_status;
      c.V[i] = clamped_scale;
    }

    dc.batch_attempts = 0;
    dc.batch_accepted = 0;
    dc.batch_unusable = 0;
    dc.batch_alpha_sum = 0.0L;
  };

  while (!active.empty()) {
    std::shuffle(active.begin(), active.end(), out.rng);
    std::vector<std::size_t> next;
    next.reserve(active.size());
    for (std::size_t i : active) {
      if (out.coordinate_proposed - proposed_at_entry >= allowance) {
        out.stop = anneal::RungStop::Budget;
        break;
      }
      if (!ctx.budget.try_reserve()) {
        out.stop = anneal::RungStop::Budget;
        break;
      }
      if (ctx.stop.load(std::memory_order_relaxed)) {
        ctx.budget.release();
        out.stop = anneal::RungStop::External;
        break;
      }
      if (ctx.time_exceeded()) {
        ctx.budget.release();
        out.stop = anneal::RungStop::Time;
        break;
      }

      Vec proposal = c.x;
      auto draw = [&]() -> long double {
        return static_cast<long double>(c.x(i)) +
               static_cast<long double>(c.V[i]) * ndist(out.rng);
      };
      long double value = draw();
      anneal::constrain_proposal(value, sp.low[i], sp.high[i],
                                 kb.boundary_mode, draw);
      proposal(i) = static_cast<double>(value);
      const double f_proposal = host(sp.out(proposal));
      ++out.coordinate_proposed;
      DimCalibration& dc = out.dims[i];
      if (dc.batch_attempts == 0) dc.batch_scale = c.V[i];
      ++dc.attempts;
      const long double alpha = anneal::metropolis_probability(
          c.f, f_proposal, T, sp.failscore);
      ++dc.batch_attempts;
      dc.batch_alpha_sum += alpha;
      if (!anneal::loss_is_finite(f_proposal, sp.failscore)) {
        ++dc.unusable;
        ++dc.batch_unusable;
      }
      const bool accepted = anneal::metropolis_accept(
          c.f, f_proposal, T, sp.failscore, [&] { return udist(out.rng); });
      if (accepted) {
        c.x = proposal;
        c.f = f_proposal;
        ++dc.accepted;
        ++dc.batch_accepted;
        ++out.coordinate_accepted;
        if (!anneal::loss_is_rankable(c.best_f, sp.failscore) ||
            c.f < c.best_f) {
          c.best_f = c.f;
          c.best_x = c.x;
        }
      }

      if (dc.batch_attempts >= batch_size || dc.attempts >= maximum)
        close_batch(i);

      if (anneal::stopval_reached(c.f, ctx.stopval, sp.failscore)) {
        ctx.stop.store(true, std::memory_order_relaxed);
        out.stop = anneal::RungStop::StopVal;
        break;
      }
      if (!dc.finished && dc.attempts < maximum)
        next.push_back(i);
    }
    if (out.stop != anneal::RungStop::Completed) break;
    active.swap(next);
  }

  for (std::size_t i = 0; i < out.dims.size(); ++i) {
    auto& dc = out.dims[i];
    if (out.stop != anneal::RungStop::Completed &&
        !dc.finished && dc.attempts < maximum) {
      dc.status |= StepInterrupted;
      /* D11: an interruption can land between batches, leaving V[i] at
       * the last PROBE scale -- possibly a /10 gross-miss backoff that
       * was never scored -- rather than anything tested. Continue at the
       * best tested scale instead, and keep the diagnostics column
       * consistent with the kernel that will actually run. */
      if (dc.have_best && dc.best_scale > 0.0) {
        c.V[i] = dc.best_scale;
        dc.coordinate_scale = dc.best_scale;
      }
    }
  }
}

template<class Vec, class HostT>
void full_vector_pilot(anneal::ChainState<Vec>& c, double T,
                       const OptimizerSpec& sp, const anneal::Knobs& kb,
                       HostT& host, anneal::StopCtx& ctx,
                       std::uint64_t allowance, SlotCalibration& out)
{
  constexpr std::size_t window_size = 20;
  constexpr std::size_t pilot_minimum = 100;
  constexpr std::size_t pilot_maximum = 200;
  constexpr double target = anneal::kTargetAcceptVector;
  constexpr double target_lo = 0.15;
  constexpr double target_hi = 0.35;
  const std::size_t dim = sp.low.size();
  if (out.pilot_base.empty()) {
    const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(dim));
    out.pilot_base.resize(dim);
    out.pilot_multiplier_min = 0.0;
    out.pilot_multiplier_max = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < dim; ++i) {
      const double floor = anneal::resolvable_coordinate_scale(
          c.x(i), sp.low[i], sp.high[i]);
      out.pilot_base[i] = c.V[i] * inv_sqrt_dim;
      /* D22: a coordinate whose base has been raised TO the floor must not
       * veto shrinkage for every other dimension.
       *
       * pilot_multiplier_min is a max over floor/base across all dims. When
       * base was clamped up to floor, that ratio is exactly 1.0, so the
       * global minimum multiplier becomes 1.0 and the clamp below turns
       * every shrink into a no-op -- for ALL dims. The pilot can then only
       * grow the step, will exhaust its 200 evaluations, and reports
       * `unconverged` with no indication that a single numerically pinned
       * coordinate disabled the search. Such a dim is already flagged
       * StepNumericalFloor and already cannot shrink; excluding it from the
       * global bound is what keeps the bound meaningful for the rest. */
      bool at_floor = false;
      if (!(out.pilot_base[i] >= floor)) {
        out.pilot_base[i] = floor;
        /* D9: this is a PILOT-phase fact, so it belongs to pilot_status.
         * It used to be ORed into dims[i].status, which is the COORDINATE
         * phase's status column -- so a dim that converged cleanly in the
         * coordinate phase was retroactively relabelled numerical-floor,
         * and the coordinate status_floor tally counted pilot events.
         * (Safe to move: coordinate_converged() reads dims[].status only
         * during the coordinate phase, which has already finished here.) */
        out.pilot_status |= StepNumericalFloor;
        at_floor = true;
      }
      const double floor_ratio = floor / out.pilot_base[i];
      const double cap_ratio =
          (sp.high[i] - sp.low[i]) / out.pilot_base[i];
      if (!at_floor)
        out.pilot_multiplier_min =
            std::max(out.pilot_multiplier_min, floor_ratio);
      out.pilot_multiplier_max =
          std::min(out.pilot_multiplier_max, cap_ratio);
    }
    out.pilot_multiplier = std::clamp(
        1.0, out.pilot_multiplier_min, out.pilot_multiplier_max);
  }

  out.stop = anneal::RungStop::Completed;
  out.pilot_status &= ~StepInterrupted;
  for (auto& dc : out.dims) dc.status &= ~StepInterrupted;
  const std::size_t proposed_at_entry = out.pilot_proposed;
  std::normal_distribution<double> ndist(0.0, 1.0);
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  while (!out.pilot_finished && out.pilot_proposed < pilot_maximum) {
    if (out.pilot_proposed - proposed_at_entry >= allowance) {
      out.stop = anneal::RungStop::Budget;
      break;
    }
    if (!ctx.budget.try_reserve()) {
      out.stop = anneal::RungStop::Budget;
      break;
    }
    if (ctx.stop.load(std::memory_order_relaxed)) {
      ctx.budget.release();
      out.stop = anneal::RungStop::External;
      break;
    }
    if (ctx.time_exceeded()) {
      ctx.budget.release();
      out.stop = anneal::RungStop::Time;
      break;
    }

    Vec proposal(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      auto draw = [&]() -> long double {
        return static_cast<long double>(c.x(i)) +
               static_cast<long double>(out.pilot_base[i] *
                                        out.pilot_multiplier) * ndist(out.rng);
      };
      long double value = draw();
      anneal::constrain_proposal(value, sp.low[i], sp.high[i],
                                 kb.boundary_mode, draw);
      proposal(i) = static_cast<double>(value);
    }
    const double f_proposal = host(sp.out(proposal));
    ++out.pilot_proposed;
    const long double alpha = anneal::metropolis_probability(
        c.f, f_proposal, T, sp.failscore);
    ++out.pilot_window_attempts;
    out.pilot_window_alpha_sum += alpha;
    if (!anneal::loss_is_finite(f_proposal, sp.failscore)) {
      ++out.pilot_unusable;
      ++out.pilot_window_unusable;
    }
    const bool accepted = anneal::metropolis_accept(
        c.f, f_proposal, T, sp.failscore, [&] { return udist(out.rng); });
    if (accepted) {
      c.x = proposal;
      c.f = f_proposal;
      ++out.pilot_accepted;
      ++out.pilot_window_accepted;
      if (!anneal::loss_is_rankable(c.best_f, sp.failscore) ||
          c.f < c.best_f) {
        c.best_f = c.f;
        c.best_x = c.x;
      }
    }

    if (out.pilot_window_attempts >= window_size ||
        out.pilot_proposed >= pilot_maximum) {
      const double mean_alpha = static_cast<double>(
          out.pilot_window_alpha_sum /
          static_cast<long double>(out.pilot_window_attempts));
      const double score = std::fabs(mean_alpha - target) +
                           (out.pilot_window_unusable > 0 ? 1.0 : 0.0);
      if (!out.pilot_have_best || score < out.pilot_best_score) {
        out.pilot_have_best = true;
        out.pilot_best_score = score;
        out.pilot_best_multiplier = out.pilot_multiplier;
        out.pilot_validation_attempts = out.pilot_window_attempts;
        out.pilot_validation_accepted = out.pilot_window_accepted;
        out.pilot_validation_mean_alpha = mean_alpha;
      }

      const bool in_target = out.pilot_window_unusable == 0 &&
                             mean_alpha >= target_lo &&
                             mean_alpha <= target_hi;
      bool change_multiplier = !in_target;
      if (in_target) {
        ++out.pilot_target_windows;
      } else {
        const bool had_target_window = out.pilot_target_windows > 0;
        out.pilot_target_windows = 0;
        const bool gross_miss = out.pilot_window_unusable > 0 ||
                                mean_alpha < 0.01 || mean_alpha > 0.9;
        change_multiplier = !had_target_window || gross_miss;
        if (change_multiplier) {
          if (mean_alpha < target_lo || out.pilot_window_unusable > 0) {
            out.pilot_have_upper = true;
            out.pilot_upper = out.pilot_multiplier;
          } else {
            out.pilot_have_lower = true;
            out.pilot_lower = out.pilot_multiplier;
          }
        }
      }

      if (out.pilot_proposed >= pilot_minimum &&
          out.pilot_target_windows >= 2) {
        out.pilot_finished = true;
        out.pilot_status = StepOk;
        out.pilot_validation_attempts = out.pilot_window_attempts;
        out.pilot_validation_accepted = out.pilot_window_accepted;
        out.pilot_validation_mean_alpha = mean_alpha;
      } else if (out.pilot_proposed >= pilot_maximum) {
        out.pilot_finished = true;
        out.pilot_status = StepUnconverged;
        if (out.pilot_have_best)
          out.pilot_multiplier = out.pilot_best_multiplier;
      } else if (change_multiplier) {
        const double old_multiplier = out.pilot_multiplier;
        double next_multiplier = old_multiplier;
        if (out.pilot_have_lower && out.pilot_have_upper &&
            out.pilot_lower < out.pilot_upper) {
          next_multiplier = std::sqrt(out.pilot_lower * out.pilot_upper);
        } else if (mean_alpha < target_lo ||
                   out.pilot_window_unusable > 0) {
          next_multiplier /= 2.0;
        } else {
          next_multiplier *= 2.0;
        }
        next_multiplier = std::clamp(
            next_multiplier, out.pilot_multiplier_min,
            out.pilot_multiplier_max);
        if (next_multiplier < old_multiplier) ++out.pilot_backoffs;
        if (next_multiplier != old_multiplier)
          out.pilot_multiplier_moved = true;   /* D12 */
        out.pilot_multiplier = next_multiplier;
      }

      out.pilot_window_attempts = 0;
      out.pilot_window_accepted = 0;
      out.pilot_window_unusable = 0;
      out.pilot_window_alpha_sum = 0.0L;
    }
    if (anneal::stopval_reached(c.f, ctx.stopval, sp.failscore)) {
      ctx.stop.store(true, std::memory_order_relaxed);
      out.stop = anneal::RungStop::StopVal;
      break;
    }
  }

  for (std::size_t i = 0; i < dim; ++i) {
    c.V[i] = out.pilot_base[i] * out.pilot_multiplier;
    if (!out.pilot_finished && out.stop != anneal::RungStop::Completed)
      out.dims[i].status |= StepInterrupted;
  }
  if (!out.pilot_finished && out.stop != anneal::RungStop::Completed)
    out.pilot_status |= StepInterrupted;
}

}  // namespace pt_detail

template<typename Vec, typename HostT>
  void run_anneal_pt_impl(const OptimizerSpec& sp, HostT& host)
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
#ifdef TADAH_HPO_MPI
    if (hpo_mpi_active()) pt_detail::validate_mpi_payload_dimension(dim);
#endif
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::PT);
    const std::vector<double> initial =
        anneal::make_initial_point(sp, AlgoKind::PT);

    const std::size_t threads = sp.num_threads.value_or(1);
    const std::size_t K = kb.replicas ? kb.replicas
                                      : std::max<std::size_t>(threads, 2);
    if (K < 2)
      throw std::runtime_error("TADAH:ANNEAL_PT: needs REPLICAS >= 2");
    if (kb.ladder_adapt && K < 3)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: LADDER_ADAPT requires REPLICAS >= 3");
    const bool ladder_controls_explicit =
        sp.param.count("LADDER_POWER") || sp.param.count("LADDER_ADAPT");

    /* --- MPI layout (no-op when built without TADAH_HPO_MPI or run on
     * one rank): contiguous ladder blocks per rank, slot 0 (coldest) on
     * rank 0. Chain RNG streams are keyed by GLOBAL slot index, budget
     * slices are computed from the GLOBAL remaining budget, and swap
     * coins come from per-pair streams -- so the evaluated set is
     * identical for any rank count (and any THREADS). */
    const int world = hpo_size();
    const int rank  = hpo_rank();
    if (static_cast<std::size_t>(world) > K)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: REPLICAS (" + std::to_string(K) +
          ") must be >= the MPI world size (" + std::to_string(world) + ")");
    auto slot_count = [&](int r) -> std::size_t {
      return K / world + (static_cast<std::size_t>(r) < K % world ? 1 : 0);
    };
    auto slot_offset = [&](int r) -> std::size_t {
      std::size_t off = 0;
      for (int q = 0; q < r; ++q) off += slot_count(q);
      return off;
    };
    const std::size_t n_local  = slot_count(rank);
    const std::size_t off_local = slot_offset(rank);

    if (!sp.num_threads) {
      const std::string hint =
          "ANNEAL_PT: THREADS not set -- running " + std::to_string(K) +
          " replicas on 1 worker thread. Set THREADS <n> in the STRATEGY "
          "block to evaluate replicas concurrently.";
      std::cout << "[HPO] " << hint << "\n";
      HPOLog::instance().warning(hint);
    }
#ifdef TADAH_ENABLE_OPENMP
    {
      // Under the shared-pool model THREADS is the actual concurrency: the
      // pc_lammps fan-out and the MLIP train/predict all run on the pool.
      // OMP_NUM_THREADS drives ONLY the residual NN-list / data-stats kernels,
      // which fire transiently during per-eval training and not at all for
      // pc_lammps-only configs -- so it is NOT multiplicative with THREADS.
      // Warn only when the pool itself oversubscribes the cores. Do NOT
      // re-introduce a "THREADS x OMP_NUM_THREADS > cores" warning: it is a
      // false positive for the common (pc_lammps) case, where OMP is inert.
      const unsigned hw = std::thread::hardware_concurrency();
      if (hw && threads > hw) {
        std::ostringstream oss;
        oss << "ANNEAL_PT: THREADS(" << threads
            << ") exceeds hardware_concurrency(" << hw
            << "); the shared worker pool is oversubscribed. Set THREADS <= cores.";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
#endif

    /* Per-dimension proposal scale in transformed space: CLASSIC Gaussian
     * sigma, or Corana uniform half-width V (already floor/cap-clamped). */
    const std::vector<double> proposal_scale = anneal::make_initial_steps(sp, kb);
    /* Initial displacement, decoupled from the production step (1.5.0).
     * Identical to proposal_scale unless INIT_MODE JITTER carries a
     * magnitude, so the default trajectory is unchanged. */
    const std::vector<double> jitter_sigma =
        anneal::make_jitter_sigma(sp, kb, proposal_scale);

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t max_eval_global = sp.stop.maxeval.value_or(HUGE_UNSIGNED);
    /* Rank 0 alone owns seed/T0 calibration.  Cap it at the GLOBAL limit;
     * subsequent distributed phases are admitted from deterministic global
     * slot allowances and allreduce their rank-local counters. */
    anneal::SharedBudget budget(
        world > 1 && rank != 0 ? HUGE_UNSIGNED : max_eval_global);

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    const std::function<bool()> time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const auto stopval = sp.stop.stopval;

    std::atomic<bool> stop_flag{false};
    anneal::StopCtx ctx{budget, stop_flag, time_exceeded, stopval};

    /* Best-stall stop (PT analogue of the serial eps-NEPS plateau): the
     * run stops when the GLOBAL best has not improved by more than the
     * FTOL tolerance over NEPS consecutive diagnostics windows. The
     * coldest slot's occupant changes at every exchange, so a
     * per-trajectory plateau has no clean meaning here; the run best is
     * the quantity a stopping user actually watches. Armed only by a
     * positive FTOL_REL/FTOL_ABS. */
    const double stall_ftol_rel = sp.stop.ftol_rel.value_or(0.0);
    const double stall_ftol_abs = sp.stop.ftol_abs.value_or(0.0);
    const bool   stall_armed = stall_ftol_rel > 0.0 || stall_ftol_abs > 0.0;
    const std::size_t stall_window =
        kb.swap_window > 0
            ? kb.swap_window
            : static_cast<std::size_t>(anneal_defaults::SWAP_WINDOW);
    double      stall_ref_best = std::numeric_limits<double>::infinity();
    std::size_t stall_windows  = 0;

    enum class StopReason {
      Done, MaxEval, MaxTime, StopVal, Plateau,
      StopValOnProbe, External, ExternalOnProbe, MaxTimeOnProbe,
      NoDims, NoBudget
    };
    StopReason stop_reason = StopReason::Done;
    /* Set when the epoch loop unwound on SIGINT/SIGTERM/SIGUSR2 (see the
     * catch below). The exception is re-thrown after the end-of-run
     * report so hpo_run() still writes its SHUTDOWN trailer. */
    bool shutdown_seen = false;

    double      T_hot = 0.0, T_cold = 0.0;
    std::size_t epoch_count    = 0;
    std::size_t exchange_rounds = 0;
    std::size_t total_proposed = 0, total_accepted = 0, total_oob = 0;
    std::size_t reanneal_count = 0;  (void)reanneal_count;
    /* g_best is the GLOBAL best value (allreduced for logging/stops);
     * (l_best, l_best_x) is the best THIS RANK evaluated -- ownership
     * questions (who broadcasts the winning x) are always resolved
     * from l_best, never from the reduced value (after a reduction
     * every rank holds the same number, so a MINLOC over it would tie
     * onto rank 0 which need not own the matching x). */
    double      g_best = std::numeric_limits<double>::infinity();
    double      l_best = std::numeric_limits<double>::infinity();
    Vec         g_best_x(dim);
    Vec         l_best_x(dim);
    /* NOT RANK-UNIFORM, and must never gate a collective.  Rank 0 sets it
     * as soon as the seed evaluation returns; every other rank sets it only
     * on the success path of the T0 broadcast below, so any run that stops
     * during seeding/probing (NoBudget, *OnProbe, ...) leaves rank 0 with
     * true and the rest with false.  That is fine for the rank-0 journal
     * blocks it guards -- and it deadlocked the Corana census the moment a
     * gatherv was placed inside one.  Gate collectives on epoch_count or a
     * reduced value instead. */
    bool        ran_sa = false;

    {
      std::ostringstream oss;
      const char* mode = kb.corana
          ? "CORANA"
          : (kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE
                 ? "CLASSIC[ADAPTIVE]"
                 : (kb.step_mode == anneal_enums::STEP_MODE::FROZEN
                        ? "CLASSIC[FROZEN]"
                        : "CLASSIC[FIXED]"));
      oss << "ANNEAL_PT[" << mode << "]: seed=" << seed
          << "  REPLICAS=" << K
          << "  THREADS=" << threads
          << "  SWAP_EVERY=" << kb.swap_every;
      if (kb.t_cold)      oss << "  T_COLD=" << *kb.t_cold;
      else                oss << "  LADDER_SPAN=" << kb.ladder_span;
      if (ladder_controls_explicit)
        oss << "  LADDER_POWER=" << kb.ladder_power
            << "  LADDER_ADAPT=" << (kb.ladder_adapt ? "ON" : "OFF");
      if (kb.ladder_adapt) {
        /* D15: LADDER_ADAPT_EVERY counts ATTEMPTS PER PAIR, while its
         * neighbours SWAP_EVERY and SWAP_WINDOW count EPOCHS -- and DEO
         * parity attempts each pair only every other exchange round, so the
         * two scales differ by 2x. Printed side by side with no units this
         * reads as a contradiction against hpo_temperature_ladder.tsv,
         * whose updates land at epoch 40, 80, ... for a value of 20. State
         * the derived epoch cadence rather than making the reader
         * rediscover the factor. */
        oss << "  LADDER_ADAPT_EVERY=" << kb.ladder_adapt_every
            << " attempts/pair (= every "
            << kb.ladder_adapt_every * 2 * kb.swap_every << " epochs at "
            << "SWAP_EVERY=" << kb.swap_every << " with DEO parity)"
            << "  LADDER_ADAPT_GAIN=" << kb.ladder_adapt_gain
            << "  LADDER_ADAPT_DECAY=" << kb.ladder_adapt_decay
            << "  LADDER_MIN_GAP=" << kb.ladder_min_gap
            << (kb.ladder_min_gap == 0.0 ? " (no floor)" : "")
            << "  LADDER_FREEZE=" << (kb.ladder_freeze ? "ON" : "OFF");
      }
      if (kb.corana)
        oss << "  NS=" << kb.NS << " (sweeps per V adjustment)"
            << "  CORANA_C=" << kb.corana_c
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
      if (kb.step_mode != anneal_enums::STEP_MODE::FIXED) {
        oss << "  STEP_WARMUP_SWEEPS=" << kb.step_warmup_sweeps;
        if (kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE) {
          if (kb.step_trust_factor == 0.0) oss << "  STEP_TRUST_FACTOR=OFF";
          else oss << "  STEP_TRUST_FACTOR=" << kb.step_trust_factor;
        }
      }
      oss << "  STEP_FRAC=" << kb.step_frac
          << (kb.corana ? " (Corana V half-width/width)"
                        : " (CLASSIC Gaussian sigma/width)");
      oss << "  BOUNDARY_MODE="
          << (kb.boundary_mode == anneal::BoundaryMode::Reflect
                  ? "REFLECT" : "REDRAW")
          << "  INIT_MODE=";
      /* State the distribution ONCE: the magnitude clause decides it, so a
       * fixed "others Gaussian around it" prefix would contradict the
       * uniform draw that FIXED/AUTO actually use. */
      if (kb.init_mode != anneal_enums::INIT_MODE::JITTER) {
        oss << "UNIFORM (slot 1 = seed; others drawn uniformly in the box)";
      } else {
        oss << "JITTER (slot 1 = seed; others ";
        switch (kb.init_jitter_mode) {
          case anneal::Knobs::JitterMode::Step:
            oss << "Gaussian, sigma=STEP_FRAC*width=" << kb.step_frac << ")";
            break;
          case anneal::Knobs::JitterMode::Fixed:
            oss << "uniform on +/-sigma, sigma=" << kb.init_jitter_frac
                << "*width)";
            break;
          case anneal::Knobs::JitterMode::Auto:
            oss << "uniform on +/-sigma, sigma measured per dim at tol="
                << kb.init_jitter_tol << ")";
            break;
        }
        /* The magnitude clause above is the BASE sigma; since 1.5.0 each
         * slot's is scaled by the ladder unless this says OFF, so leaving
         * it out would make the banner quietly wrong about what slot k
         * actually draws. */
        oss << " x (T_k/T_hot)^";
        if (kb.init_jitter_ladder > 0.0)
          oss << anneal::fmt_knob_number(kb.init_jitter_ladder)
              << (kb.init_jitter_ladder_auto ? " [INIT_JITTER_LADDER AUTO]"
                                             : " [INIT_JITTER_LADDER]");
        else
          oss << "0 [INIT_JITTER_LADDER OFF: every slot uses the same sigma]";
      }
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* D20: an adapting ladder with a frozen step.
     *
     * STEP_MODE FROZEN calibrates each slot's step once and freezes it.
     * That is the right default for a STATIC ladder -- the step was
     * calibrated at the slot's temperature and the temperature does not
     * move. With LADDER_ADAPT ON the interior temperatures do move, and
     * every slot then keeps a step tuned for a temperature it no longer
     * has. Combination is legal (a user may want a frozen kernel
     * deliberately) and the consequence is a modelling judgement rather
     * than a proof, so this is a HINT. Not applicable under CORANA,
     * whose own V adaptation is always live. */
    if (!kb.corana && kb.step_mode == anneal_enums::STEP_MODE::FROZEN &&
        kb.ladder_adapt) {
      std::ostringstream h;
      h << "ANNEAL_PT: interior temperatures move (LADDER_ADAPT=ON) but "
           "STEP_MODE FROZEN keeps each slot's step at the value "
           "calibrated for its ORIGINAL temperature. Acceptance will "
           "drift away from target as the ladder moves, and no later "
           "message reports that drift. STEP_MODE ADAPTIVE keeps the "
           "step adapting with the temperature.";
      HPOLog::instance().hint(h.str());
    }

    /* Layout arithmetic -- exact, printed up front. A replica runs ONE
     * sweep per epoch (CLASSIC: one whole-vector proposal; CORANA: dim
     * single-coordinate moves) so exchange rounds stay frequent; the
     * journal cadence is SWAP_WINDOW epochs. */
    const std::uint64_t rung_cost =
        kb.corana ? static_cast<std::uint64_t>(dim ? dim : 1) : 1u;
    const std::uint64_t epoch_cost =
        anneal::saturating_mul(static_cast<std::uint64_t>(K), rung_cost);
    {
      std::ostringstream oss;
      oss << "ANNEAL_PT layout: sweep = ";
      if (kb.corana)
        oss << "dim = " << rung_cost;
      else
        oss << "1 proposal";
      oss << " eval(s)/replica;  epoch = REPLICAS*sweep = "
          << K << "*" << rung_cost << " = " << epoch_cost
          << " evals;  exchange attempt every " << kb.swap_every
          << " epoch(s);  reports every "
          << (kb.swap_window > 0 ? std::to_string(kb.swap_window)
                                 : std::string("(cumulative only)"))
          << " epoch(s)";
      HPOLog::instance().info(oss.str());
      /* Cadence registration is single-process only: the milestone
       * throughput window is rank-local and would mis-scale a global
       * budget under MPI. */
      if (world == 1 && epoch_cost > 0)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(epoch_cost), "epoch",
            max_eval_global == HUGE_UNSIGNED ? 0 : max_eval_global);
    }
    /* D21: state the fixed diagnostic thresholds once, so every later
     * WARNING/HINT that fires from them can be read against its rule. */
    if (rank == 0) {
      std::ostringstream d;
      d << "ANNEAL_PT diagnostics: step-health acceptance band ["
        << std::fixed << std::setprecision(0)
        << 100.0 * pt_detail::kStepHealthAcceptLo << "%, "
        << 100.0 * pt_detail::kStepHealthAcceptHi << "%] per SWAP_WINDOW, "
        << "warning after " << pt_detail::kStepHealthStreak
        << " consecutive windows;  transport window = "
        << pt_detail::kTransportWindowFactor
        << "*(REPLICAS-1) exchange rounds;  ladder-freeze evidence >= "
        << pt_detail::kLadderFreezeMinAttempts
        << " pooled attempts/pair;  degenerate (equal-loss) exchanges "
           "are reported separately and excluded from ladder adaptation";
      HPOLog::instance().info(d.str());
    }
    /* Pinned-dim audit (single-process like the per-epoch report --
     * marks are rank-local, a rank-0 table would miss other ranks). */
    if (world == 1)
      HPOLog::instance().arm_pin_audit(
          sp, kb.v_floor_frac, static_cast<int>(K), "epoch");

    /* A throw from the seed evaluation or the T0 calibration (rank 0) is
     * kept whole and rethrown after the handshake below: its type matters
     * -- core::ShutdownRequested must reach hpo_run() as a signal stop,
     * HPO_Host::StagnationStop the RESTART policy -- so it is never
     * reduced to its message. */
    bool calib_threw = false;
    std::exception_ptr calib_error;
    if (dim == 0) {
      stop_reason = StopReason::NoDims;
      if (rank == 0 && !sp.optim_init.empty()) {
        if (!budget.try_reserve()) {
          stop_reason = StopReason::NoBudget;
        } else {
          try {
            Vec xv(dim);
            for (std::size_t i = 0; i < dim; ++i) xv(i) = initial[i];
            double f = host(sp.out(xv));
            if (!std::isfinite(f))
              f = std::numeric_limits<double>::infinity();
            if (anneal::stopval_reached(f, stopval, sp.failscore))
              stop_reason = StopReason::StopVal;
            else if (time_exceeded()) stop_reason = StopReason::MaxTime;
          } catch (...) {
            calib_threw = true;
            calib_error = std::current_exception();
          }
        }
      }
    }

    bool proceed = (stop_reason == StopReason::Done);

    /* --- seed evaluation + T0 calibration --------------------------------
     * Rank 0 evaluates the seed and runs the probe calibration; the
     * outcome (T_hot, f0, x0, or an abort code) is broadcast so every
     * rank proceeds from identical state without duplicate probe evals. */
    Vec x0(dim);
    double f0 = std::numeric_limits<double>::infinity();
    if (proceed && rank == 0) {
      for (std::size_t i = 0; i < dim; ++i) x0(i) = initial[i];
      if (!budget.try_reserve()) {
        stop_reason = StopReason::NoBudget;
        proceed = false;
      } else {
        try {
          f0 = host(sp.out(x0));
          if (!std::isfinite(f0))
            f0 = std::numeric_limits<double>::infinity();
          ran_sa = true;
          g_best = l_best = f0;
          g_best_x = l_best_x = x0;
          if (anneal::stopval_reached(f0, stopval, sp.failscore)) {
            stop_reason = StopReason::StopVal; proceed = false;
          } else if (time_exceeded()) {
            stop_reason = StopReason::MaxTime; proceed = false;
          }
        } catch (...) {
          calib_threw = true;
          calib_error = std::current_exception();
          proceed = false;
        }
      }
    }

    if (proceed && rank == 0) {
      std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);
      try {
        const auto t0out = anneal::calibrate_t0(
            sp, kb, proposal_scale, x0, f0, l_best_x, l_best,
            host, probe_rng, ctx, "ANNEAL_PT");
        g_best = l_best;
        T_hot = t0out.T0;
        if (t0out.aborted) {
          stop_reason = t0out.budget_exhausted
                            ? StopReason::MaxEval
                            : (t0out.stopval_hit
                                   ? StopReason::StopValOnProbe
                                   : (t0out.external_stop
                                          ? StopReason::ExternalOnProbe
                                          : StopReason::MaxTimeOnProbe));
          proceed = false;
        } else if (budget.left() == 0) {
          stop_reason = StopReason::MaxEval;
          proceed = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTimeOnProbe;
          proceed = false;
        }
      } catch (...) {
        calib_threw = true;
        calib_error = std::current_exception();
      }
    }
    if (world > 1) {
      /* Unconditional handshake: every rank participates, so a rank-0
       * abort can never leave the others hanging on the broadcast. */
      int code = 0;
      if (rank == 0)
        code = calib_threw ? -1
                           : (proceed ? 0 : 1 + static_cast<int>(stop_reason));
      hpo_bcast(code, 0);
      if (code == -1) {
        if (rank == 0) std::rethrow_exception(calib_error);
        throw std::runtime_error(
            "TADAH:ANNEAL_PT: rank 0 aborted during seed/T0 calibration");
      }
      std::vector<double> pkt(dim + 2, 0.0);
      if (rank == 0) {
        pkt[0] = T_hot; pkt[1] = f0;
        for (std::size_t i = 0; i < dim; ++i) pkt[2 + i] = x0(i);
      }
      hpo_bcast(pkt, 0);
      if (rank != 0) {
        T_hot = pkt[0];
        f0    = pkt[1];
        for (std::size_t i = 0; i < dim; ++i) x0(i) = pkt[2 + i];
        if (code != 0) {
          stop_reason = static_cast<StopReason>(code - 1);
          proceed = false;
        } else {
          proceed  = true;
          ran_sa   = true;
          g_best   = f0;
          g_best_x = x0;
        }
      }
    } else if (calib_threw) {
      std::rethrow_exception(calib_error);
    }
    T_cold = kb.t_cold ? *kb.t_cold : T_hot / kb.ladder_span;
    if (proceed && !(T_cold > 0.0 && T_cold < T_hot))
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved ladder is invalid (T_COLD=" +
          std::to_string(T_cold) + ", T_HOT=" + std::to_string(T_hot) +
          "); set PARAM T_COLD/T_HOT or LADDER_SPAN consistently");

    /* --- ladder + chains ----------------------------------------------
     * Every rank builds the FULL temperature ladder (identical, cheap)
     * but only its own contiguous slot block of chains. Chain RNG
     * streams are keyed by the GLOBAL slot index. */
    std::vector<double> T(K, 0.0);
    long double log_t_cold = 0.0L, log_t_hot = 0.0L;
    if (proceed) {
      log_t_cold = std::log(static_cast<long double>(T_cold));
      log_t_hot  = std::log(static_cast<long double>(T_hot));
      /* D21: one ladder constructor, always.
       *
       * This used to fork on the mere PRESENCE of LADDER_POWER/LADDER_ADAPT,
       * and the legacy branch was an open-coded geometric loop with neither
       * the nextafter gap floor nor the strict-increase check that
       * make_temperature_ladder enforces. With narrow endpoints and large K
       * it silently collapsed adjacent T[k] onto the same double -- and
       * accept_exchange returns true unconditionally when the two
       * temperatures are equal, so those pairs reported 100% swap
       * acceptance forever while performing no tempering at all. The
       * per-pair table looked perfect precisely where the method had
       * stopped working.
       *
       * Removing the branch is a no-op on every well-formed ladder rather
       * than a behaviour change: make_temperature_ladder with power == 1
       * (the default) sets position[k] = k/(K-1) with no pow() call, and
       * ladder_from_log_positions then evaluates the same
       * exp(log_cold + span*position[k]) in the same long double. The only
       * divergence is the clamp against the nextafter floor, which engages
       * exactly in the degenerate case above -- where it now throws with a
       * diagnostic instead of returning a mute, broken ladder. */
      T = pt_detail::make_temperature_ladder(
          T_cold, T_hot, K, kb.ladder_power);
    }

    /* Resolved-ladder report: T0 calibration fixes only the HOT end; the
     * cold end and the per-slot geometry are decided here, so this is
     * the first point the full ladder is known (also fires when a
     * user-set T_HOT skipped the calibration entirely). */
    if (rank == 0 && proceed) {
      {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "ANNEAL_PT ladder: T_HOT=" << T_hot
            << "  T_COLD=" << T_cold
            << (kb.t_cold ? " (PARAM T_COLD)" : " (T_HOT/LADDER_SPAN)")
            << "  ";
        if (kb.ladder_power == 1.0) {
          oss << "geometric over " << K << " slots, ratio="
              << std::fixed << std::setprecision(3)
              << std::exp((log_t_hot - log_t_cold) /
                          static_cast<long double>(K - 1))
              << " per rung";
        } else {
          oss << "power-shaped log ladder over " << K
              << " slots, LADDER_POWER=" << std::fixed
              << std::setprecision(3) << kb.ladder_power;
        }
        /* The ladder endpoints are always fixed; LADDER_ADAPT
         * redistributes the interior between them. */
        if (kb.ladder_adapt) {
          oss << "  [endpoints fixed; interior adapts]";
        } else {
          oss << "  [static ladder]";
        }
        HPOLog::instance().info(oss.str());
      }
      /* Per-slot table: [lo-hi] range rows of 8 (HPOLog::event tags and
       * indents continuation lines). The per-epoch report and the
       * end-of-run new-best histogram mirror this grid so their values
       * map onto temperatures by eye. */
      {
        std::ostringstream head;
        head << "ANNEAL_PT ladder T per slot (" << logidx::uis(0)
             << "=coldest .. " << logidx::uis(K - 1) << "=hottest):";
        HPOLog::instance().info(logidx::grid(
            head.str(), K, [&](std::ostringstream& o, std::size_t k) {
              o << std::scientific << std::setprecision(3) << "  " << T[k];
            }));
      }
    }

    std::ofstream ladder_file;
    if (rank == 0 && proceed && kb.ladder_adapt) {
      ladder_file.open("hpo_temperature_ladder.tsv", std::ios::trunc);
      if (!ladder_file) {
        HPOLog::instance().warning(
            "ANNEAL_PT: cannot open hpo_temperature_ladder.tsv for writing");
      } else {
        ladder_file << "update\tepoch\tpair\tattempts\taccepted\tacceptance"
                       "\told_T_low\told_T_high\tnew_T_low\tnew_T_high"
                       "\told_log_gap\tnew_log_gap\tat_floor\n";
        ladder_file << std::scientific << std::setprecision(17);
      }
    }

    /* ---- INIT_MODE JITTER AUTO -----------------------------------------
     * Measure the per-dimension jitter here: the anchor (x0, f0) is known
     * and frozen, the ladder is resolved, and no replica has been placed
     * yet. Single-rank for now (the MPI split is a later step), so a
     * multi-rank run falls back to the STEP_FRAC sigma with a note rather
     * than having each rank calibrate a different vector. */
    std::vector<double> auto_sigma = jitter_sigma;
    if (proceed && kb.init_jitter_mode == anneal::Knobs::JitterMode::Auto) {
      /* Every rank participates: the dimensions are split into contiguous
       * blocks and the resulting sigma vector is broadcast, so all ranks
       * seed their own replicas from the same measurement. */
      const auto plan = anneal::run_jitter_calibration<Vec>(
          sp, kb, x0, f0, jitter_sigma, host, ctx, max_time, "ANNEAL_PT",
          rank, world);
      auto_sigma = plan.sigma;
    }

    std::vector<anneal::ChainState<Vec>> chains(n_local);
    std::vector<pt_detail::SlotCalibration> step_calibration(n_local);
    std::vector<double> gathered_step_records;
    anneal::ReplicaPool pool(std::min(threads, n_local ? n_local : 1),
                             stop_flag);
    auto rethrow_pool_error = [&](const char* phase) {
      const bool local_error = pool.has_error();
      const int any_error = world > 1
          ? hpo_allreduce_max(local_error ? 1 : 0)
          : (local_error ? 1 : 0);
      if (!any_error) return;
      if (local_error) pool.rethrow_if_error();
      throw std::runtime_error(
          std::string("TADAH:ANNEAL_PT: worker failed on another MPI rank during ") +
          phase);
    };

    if (proceed) {
      /* --- ladder scaling of the INITIAL jitter (INIT_JITTER_LADDER) -----
       *
       * sigma_k = sigma_d * (T_k/T_hot)^gamma. Without it every slot is
       * placed with the SAME displacement, so the coldest replica -- the one
       * whose whole job is to refine the seed -- starts as far from it as
       * the hottest, and the ensemble has no cold end to speak of.
       *
       * The ladder read here is the one resolved above. Under the
       * DERIVED T_COLD (the default, T_hot/LADDER_SPAN) the shape T_k/T_hot
       * is invariant under the T0 pass-2 re-anchoring further down, because
       * that rescales both ends by the same factor; only a PINNED T_COLD
       * makes pass 2 change the shape, and the population must exist before
       * the warm-up that pass 2 depends on, so this is the ladder to use.
       *
       * Slot 0 is exempt: seed_replica places it exactly at the seed with
       * no draw at all, so it has no sigma to scale and must not appear in
       * the floor census. */
      std::size_t jitter_floored = 0, jitter_scaled_dims = 0;
      const double T_max = K > 0 ? T[K - 1] : 0.0;
      std::vector<double> x0v(dim), slot_factor(K, 1.0);
      for (std::size_t i = 0; i < dim; ++i) x0v[i] = x0(i);
      const bool ladder_jitter =
          kb.init_mode == anneal_enums::INIT_MODE::JITTER &&
          kb.init_jitter_ladder > 0.0 && dim > 0;
      if (ladder_jitter)
        for (std::size_t k = 1; k < K; ++k)
          slot_factor[k] = anneal::ladder_jitter_factor(
              T[k], T_max, kb.init_jitter_ladder);
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        if (ladder_jitter && k > 0) {
          const auto scaled = anneal::scale_jitter_for_slot(
              auto_sigma, x0v, sp.low, sp.high, T[k], T_max,
              kb.init_jitter_ladder);
          jitter_floored += scaled.floored;
          jitter_scaled_dims += dim;
          anneal::seed_replica(chains[li], seed, k, x0, proposal_scale,
                               scaled.sigma, sp, kb);
        } else {
          anneal::seed_replica(chains[li], seed, k, x0, proposal_scale,
                               auto_sigma, sp, kb);
        }
      }
      if (ladder_jitter && world > 1) {
        jitter_floored = hpo_allreduce_sum_sz(jitter_floored);
        jitter_scaled_dims = hpo_allreduce_sum_sz(jitter_scaled_dims);
      }
      if (ladder_jitter && rank == 0) {
        std::ostringstream oss;
        oss << "ANNEAL_PT initial-jitter ladder scaling: sigma_k = sigma_d * "
               "(T_k/T_hot)^gamma with gamma=" << std::fixed
            << std::setprecision(3) << kb.init_jitter_ladder
            << (kb.init_jitter_ladder_auto
                    ? " (AUTO: equipartition, spread ~ sqrt(T))"
                    : " (INIT_JITTER_LADDER)")
            << ".  Slot " << logidx::uis(0)
            << " is the seed exactly; slot " << logidx::uis(1) << " draws "
            << std::setprecision(2) << (100.0 * slot_factor[K > 1 ? 1 : 0])
            << "% of sigma_d and slot " << logidx::uis(K - 1) << " 100%.";
        HPOLog::instance().info(oss.str());
        HPOLog::instance().info(logidx::grid(
            "ANNEAL_PT initial-jitter factor per slot (% of sigma_d; the "
            "coldest slot is the seed itself and draws none):", K,
            [&](std::ostringstream& o, std::size_t k) {
              /* slot_factor[0] is 1.0 only because nothing scales it --
               * printing "100.000" there would read as "the coldest slot
               * draws the FULL sigma", the exact opposite of the truth. */
              o << "  " << std::setw(9);
              if (k == 0) o << "seed";
              else o << std::fixed << std::setprecision(3)
                     << (100.0 * slot_factor[k]);
            }));
        if (jitter_floored > 0) {
          std::ostringstream w;
          w << "ANNEAL_PT: ladder scaling drove " << jitter_floored << " of "
            << jitter_scaled_dims << " (slot, dimension) initial sigmas below "
               "the resolvable coordinate scale; they were clamped there so "
               "the cold slots stay distinct points rather than K copies of "
               "the seed. A large share means the ladder is wider than the "
               "search space can express -- reduce LADDER_SPAN, or set "
               "PARAM INIT_JITTER_LADDER to a smaller gamma or OFF.";
          HPOLog::instance().warning(w.str());
        }
      }
      if (rank == 0 && n_local > 0) {
        chains[0].f      = f0;
        chains[0].best_f = f0;
      }

      /* Evaluate a deterministic global prefix of non-seed starting points.
       * This makes partial initialization identical for every rank/thread
       * layout and prevents MPI ranks collectively overshooting MAXEVAL. */
      const std::uint64_t spent_before_init = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t init_remaining = max_eval_global > spent_before_init
          ? max_eval_global - spent_before_init : 0;
      const std::uint64_t init_admitted = std::min<std::uint64_t>(
          K > 0 ? static_cast<std::uint64_t>(K - 1) : 0, init_remaining);
      pool.run_epoch(n_local, [&](std::size_t li) {
        /* key this chain's warm-start by its GLOBAL slot (off_local+li) so
         * results are independent of which worker runs it. */
        const std::size_t gk = off_local + li;
        warmstart::current_chain() = gk;
        HPOLog::set_eval_slot(static_cast<int>(gk), K, T[gk]);
        if (rank == 0 && li == 0) return;   /* global slot 0 = seed */
        if (gk == 0 || gk > init_admitted) return;
        auto& c = chains[li];
        if (stop_flag.load(std::memory_order_relaxed)) return;
        if (!ctx.budget.try_reserve()) return;   /* f stays inf */
        c.f = host(sp.out(c.x));
        if (!std::isfinite(c.f))
          c.f = std::numeric_limits<double>::infinity();
        c.best_f = c.f;
        c.best_x = c.x;
        if (anneal::stopval_reached(c.f, stopval, sp.failscore))
          stop_flag.store(true, std::memory_order_relaxed);
      });
      rethrow_pool_error("population initialization");

      /* ---- what the jitter actually DID ----------------------------------
       * Costs no extra evaluations: every start above was already evaluated.
       *
       * This is the reality check on the calibration, and it is mandatory
       * for a reason. AUTO bounds each coordinate SEPARATELY, and per-axis
       * tolerances do not compose: at high dimension a replica can respect
       * every single one of them and still land far outside the band
       * jointly. The per-rung loss is the only number that says whether the
       * warm start actually survived, so it is reported whatever the jitter
       * mode -- including the STEP_FRAC default, where nothing reported it
       * before. */
      if (world == 1 && K > 1) {
        const double denom = std::fabs(f0) > 0.0 ? std::fabs(f0) : 1.0;
        HPOLog::instance().info(logidx::grid(
            "ANNEAL_PT initial loss per slot (slot 1 = seed):", K,
            [&](std::ostringstream& o, std::size_t k) {
              o << "  " << std::scientific << std::setprecision(3)
                << std::setw(9) << chains[k].f;
            }));
        HPOLog::instance().info(logidx::grid(
            "ANNEAL_PT initial (L-L0)/|L0| per slot:", K,
            [&](std::ostringstream& o, std::size_t k) {
              const double d = (chains[k].f - f0) / denom;
              o << " " << std::scientific << std::setprecision(3)
                << std::showpos << std::setw(10) << d << std::noshowpos;
            }));
        if (kb.init_jitter_mode == anneal::Knobs::JitterMode::Auto) {
          std::string over;
          for (std::size_t k = 1; k < K; ++k) {
            if (!std::isfinite(chains[k].f)) continue;
            if ((chains[k].f - f0) / denom > kb.init_jitter_tol) {
              if (!over.empty()) over += ", ";
              over += logidx::uis(k);
            }
          }
          if (!over.empty()) {
            std::ostringstream w;
            w << "ANNEAL_PT: slot(s) " << over << " start outside the "
              << kb.init_jitter_tol << " band the per-dimension jitter was "
                 "measured against. Each COORDINATE respects it; their sum "
                 "over " << dim << " dimensions need not. Lower the AUTO "
                 "tolerance if the warm start matters more than the spread.";
            HPOLog::instance().warning(w.str());
          }
        }
      }

      for (std::size_t li = 0; li < n_local; ++li) {
        if (chains[li].best_f < l_best) {
          l_best   = chains[li].best_f;
          l_best_x = chains[li].best_x;
        }
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        double gb = std::isfinite(l_best) ? l_best
                                          : std::numeric_limits<double>::max();
        int owner = 0;
        hpo_allreduce_minloc(gb, owner);
        if (gb < std::numeric_limits<double>::max())
          g_best = std::min(g_best, gb);
        int code = (stopval && stop_flag.load(std::memory_order_relaxed) &&
                    anneal::stopval_reached(g_best, stopval, sp.failscore)) ? 1 : 0;
        code = hpo_allreduce_max(code);
        if (code) { stop_reason = StopReason::StopVal; proceed = false; }
      } else if (stopval && stop_flag.load(std::memory_order_relaxed) &&
                 anneal::stopval_reached(g_best, stopval, sp.failscore)) {
        stop_reason = StopReason::StopVal;
        proceed = false;
      }
    }

    /* --- automatic diagonal calibration -------------------------------
     * This phase deliberately precedes every epoch/swap/flow counter. Host
     * calls still consume the shared hard budget and discovered states/bests
     * are retained, while production diagnostics start from a clean epoch 0. */
    if (proceed && kb.step_mode != anneal_enums::STEP_MODE::FIXED) {
      /* Dedicated warm-up streams (SALT_WARMUP, global slot key): the
       * calibration must not consume the chains' production streams, so
       * with explicit temperatures the production proposal sequence is
       * a pure function of (SEED, slot) for every STEP_MODE. */
      for (std::size_t li = 0; li < n_local; ++li)
        step_calibration[li].rng = anneal::make_stream(
            seed, anneal::SALT_WARMUP, off_local + li);
      const std::uint64_t warmup_spent_begin = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t max_coordinate_attempts = anneal::saturating_mul(
          anneal::saturating_mul(
              static_cast<std::uint64_t>(kb.step_warmup_sweeps), 2),
          static_cast<std::uint64_t>(dim));
      /* Calibration budget governor: the warm-up + pilot may spend at
       * most a quarter of MAXEVAL. Without a cap the default
       * calibration (up to 2*STEP_WARMUP_SWEEPS*dim + ~200 evals PER
       * SLOT) consumes the entire budget of a modest run and
       * production never starts. A capped, incomplete calibration is
       * safe: interrupted coordinates continue at their best TESTED
       * scale and are flagged in the step diagnostics. */
      const std::uint64_t warmup_budget =
          max_eval_global == HUGE_UNSIGNED ? HUGE_UNSIGNED
                                           : max_eval_global / 4;

      auto phase_allowance = [&](std::uint64_t desired_per_slot,
                                 std::size_t extra_offset = 0) {
        const std::uint64_t spent = world > 1
            ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t warmup_spent = spent > warmup_spent_begin
            ? spent - warmup_spent_begin : 0;
        const std::uint64_t cap_left = warmup_budget > warmup_spent
            ? warmup_budget - warmup_spent : 0;
        const std::uint64_t remaining = std::min<std::uint64_t>(
            max_eval_global > spent ? max_eval_global - spent : 0,
            cap_left);
        const std::uint64_t base = remaining / static_cast<std::uint64_t>(K);
        const std::uint64_t extra = remaining % static_cast<std::uint64_t>(K);
        std::vector<std::uint64_t> result(K, 0);
        for (std::size_t k = 0; k < K; ++k) {
          const std::size_t extra_position =
              (k + K - extra_offset % K) % K;
          result[k] = std::min(
              desired_per_slot,
              base + (static_cast<std::uint64_t>(extra_position) < extra ? 1 : 0));
        }
        return result;
      };

      auto phase_stop_code = [&]() {
        int code = 0;  /* 0 complete, 1 budget, 2 time, 3 STOPVAL */
        for (const auto& sc : step_calibration) {
          if (sc.stop == RungStop::StopVal ||
              (sc.stop == RungStop::External &&
               stop_flag.load(std::memory_order_relaxed)))
            code = std::max(code, 3);
          else if (sc.stop == RungStop::Time)
            code = std::max(code, 2);
          else if (sc.stop == RungStop::Budget)
            code = std::max(code, 1);
        }
        if (time_exceeded()) code = std::max(code, 2);
        if (world > 1) code = hpo_allreduce_max(code);
        return code;
      };

      int warmup_code = 0;
      std::uint64_t after_coordinates = warmup_spent_begin;
      std::size_t extra_offset = 0;
      std::size_t stalled_rounds = 0;
      for (;;) {
        const std::uint64_t before_coordinates = after_coordinates;
        const auto coordinate_allowance =
            phase_allowance(max_coordinate_attempts, extra_offset);
        pool.run_epoch(n_local, [&](std::size_t li) {
          const std::size_t k = off_local + li;
          warmstart::current_chain() = k;
          HPOLog::set_eval_slot(static_cast<int>(k), K, T[k]);
          pt_detail::coordinate_warmup(
              chains[li], T[k], sp, kb, host, ctx,
              coordinate_allowance[k], step_calibration[li]);
        });
        rethrow_pool_error("automatic coordinate warm-up");
        warmup_code = phase_stop_code();
        after_coordinates = world > 1
            ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        if (warmup_code != 1 ||
            (max_eval_global != HUGE_UNSIGNED &&
             (after_coordinates >= max_eval_global ||
              after_coordinates - warmup_spent_begin >= warmup_budget)))
          break;
        stalled_rounds = after_coordinates == before_coordinates
            ? stalled_rounds + 1 : 0;
        if (stalled_rounds >= K) break;
        extra_offset = (extra_offset + 1) % K;
        /* Some slots converged before using their fair share. Redistribute
         * the remaining global budget and rotate indivisible extras so a
         * lone unfinished hot slot cannot be starved by cold-slot priority. */
      }
      if (warmup_code == 0 && max_eval_global != HUGE_UNSIGNED &&
          after_coordinates >= max_eval_global)
        warmup_code = 1;
      /* Calibration-cap exhaustion is not a run stop: clear the phase
       * budget code so production proceeds on best tested scales. */
      if (warmup_code == 1 && max_eval_global != HUGE_UNSIGNED &&
          after_coordinates < max_eval_global &&
          after_coordinates - warmup_spent_begin >= warmup_budget)
        warmup_code = 0;

      if (warmup_code == 0) {
        std::uint64_t after_pilot = after_coordinates;
        extra_offset = 0;
        stalled_rounds = 0;
        for (;;) {
          const std::uint64_t before_pilot = after_pilot;
          const auto pilot_allowance = phase_allowance(200, extra_offset);
          pool.run_epoch(n_local, [&](std::size_t li) {
            const std::size_t k = off_local + li;
            warmstart::current_chain() = k;
            HPOLog::set_eval_slot(static_cast<int>(k), K, T[k]);
            pt_detail::full_vector_pilot(
                chains[li], T[k], sp, kb, host, ctx,
                pilot_allowance[k], step_calibration[li]);
          });
          rethrow_pool_error("automatic full-vector pilot");
          warmup_code = phase_stop_code();
          after_pilot = world > 1
              ? hpo_allreduce_sum(budget.spent()) : budget.spent();
          if (warmup_code != 1 ||
              (max_eval_global != HUGE_UNSIGNED &&
               (after_pilot >= max_eval_global ||
                after_pilot - warmup_spent_begin >= warmup_budget)))
            break;
          stalled_rounds = after_pilot == before_pilot
              ? stalled_rounds + 1 : 0;
          if (stalled_rounds >= K) break;
          extra_offset = (extra_offset + 1) % K;
        }
        if (warmup_code == 0 && max_eval_global != HUGE_UNSIGNED &&
            after_pilot >= max_eval_global)
          warmup_code = 1;
        /* As after the coordinate phase: exhausting only the
         * CALIBRATION budget is not a run stop. */
        if (warmup_code == 1 && max_eval_global != HUGE_UNSIGNED &&
            after_pilot < max_eval_global &&
            after_pilot - warmup_spent_begin >= warmup_budget)
          warmup_code = 0;
      }
      {
        const std::uint64_t spent_now = world > 1
            ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        if (rank == 0 && warmup_code == 0 &&
            max_eval_global != HUGE_UNSIGNED &&
            spent_now - warmup_spent_begin >= warmup_budget)
          HPOLog::instance().info(
              "ANNEAL_PT step warm-up reached the calibration budget "
              "(MAXEVAL/4 = " + std::to_string(warmup_budget) +
              " evals); unfinished coordinates continue at their best "
              "tested scales (see the calibration summary)");
      }

      for (std::size_t li = 0; li < n_local; ++li) {
        if (chains[li].best_f < l_best) {
          l_best = chains[li].best_f;
          l_best_x = chains[li].best_x;
        }
        chains[li].diag_ram_proposals = 0;
      }
      g_best = std::min(g_best, l_best);
      if (world > 1) {
        double gb = std::isfinite(l_best) ? l_best
                                          : std::numeric_limits<double>::max();
        int owner = 0;
        hpo_allreduce_minloc(gb, owner);
        if (gb < std::numeric_limits<double>::max()) g_best = gb;
      }

      /* Gather the slot-contiguous records so rank 0 can emit one canonical
       * file even when the ladder is split across MPI ranks. */
      /* 17 -> 19: + coordinate_scale (D9) and pilot_multiplier_moved (D12).
       *
       * The layout is named rather than indexed by literal. D9 inserted
       * `rc_coord_sigma` at offset 1, shifting everything after it; the
       * per-dim TSV reader below was updated and the per-slot journal
       * summary was not, so slot lines printed `pilot_attempts=0`
       * (coordinate `unusable`), `pilot_multiplier=0` (pilot `backoffs`)
       * and `pilot_status=ok` (pilot `unusable`) for a whole
       * calibration. Both readers now name the field they want, so the
       * next insertion cannot desynchronise them. */
      enum : std::size_t {
        rc_sigma = 0, rc_coord_sigma, rc_accepted, rc_attempts_val,
        rc_attempts, rc_backoffs, rc_status, rc_mean_alpha, rc_unusable,
        rc_pilot_proposed, rc_pilot_accepted, rc_pilot_backoffs,
        rc_pilot_multiplier, rc_pilot_val_accepted, rc_pilot_val_attempts,
        rc_pilot_val_mean_alpha, rc_pilot_unusable, rc_pilot_status,
        rc_pilot_moved, rc_field_count
      };
      constexpr std::size_t fields = rc_field_count;
      static_assert(fields == 19, "step-scale record layout changed");
      std::vector<double> local_records;
      local_records.reserve(n_local * dim * fields);
      for (std::size_t li = 0; li < n_local; ++li) {
        const auto& sc = step_calibration[li];
        for (std::size_t i = 0; i < dim; ++i) {
          const auto& dc = sc.dims[i];
          double rec[fields];
          rec[rc_sigma]            = chains[li].V[i];
          rec[rc_coord_sigma]      = dc.coordinate_scale;            /* D9 */
          rec[rc_accepted]         = static_cast<double>(dc.validation_accepted);
          rec[rc_attempts_val]     = static_cast<double>(dc.validation_attempts);
          rec[rc_attempts]         = static_cast<double>(dc.attempts);
          rec[rc_backoffs]         = static_cast<double>(dc.backoffs);
          rec[rc_status]           = static_cast<double>(dc.status);
          rec[rc_mean_alpha]       = dc.validation_mean_alpha;
          rec[rc_unusable]         = static_cast<double>(dc.unusable);
          rec[rc_pilot_proposed]   = static_cast<double>(sc.pilot_proposed);
          rec[rc_pilot_accepted]   = static_cast<double>(sc.pilot_accepted);
          rec[rc_pilot_backoffs]   = static_cast<double>(sc.pilot_backoffs);
          rec[rc_pilot_multiplier] = sc.pilot_multiplier;
          rec[rc_pilot_val_accepted] =
              static_cast<double>(sc.pilot_validation_accepted);
          rec[rc_pilot_val_attempts] =
              static_cast<double>(sc.pilot_validation_attempts);
          rec[rc_pilot_val_mean_alpha] = sc.pilot_validation_mean_alpha;
          rec[rc_pilot_unusable]   = static_cast<double>(sc.pilot_unusable);
          rec[rc_pilot_status]     = static_cast<double>(sc.pilot_status);
          rec[rc_pilot_moved]      = sc.pilot_multiplier_moved ? 1.0 : 0.0; /* D12 */
          local_records.insert(local_records.end(), rec, rec + fields);
        }
      }
      std::vector<int> record_counts(static_cast<std::size_t>(world), 0);
      std::size_t record_total = 0;
      for (int r = 0; r < world; ++r) {
        const std::size_t count = slot_count(r) * dim * fields;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::length_error(
              "TADAH:ANNEAL_PT: step-scale diagnostic exceeds MPI count limit");
        record_counts[static_cast<std::size_t>(r)] = static_cast<int>(count);
        record_total += count;
      }
      if (rank == 0) gathered_step_records.resize(record_total);
      hpo_gatherv(local_records, gathered_step_records, record_counts);
      const std::uint64_t warmup_spent_end = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();

      if (rank == 0) {
        auto status_text = [](unsigned status) {
          if (status == pt_detail::StepOk) return std::string("ok");
          std::string text;
          auto add = [&](const char* word) {
            if (!text.empty()) text += ',';
            text += word;
          };
          if (status & pt_detail::StepNumericalFloor) add("numerical-floor");
          if (status & pt_detail::StepSupportCapped) add("support-capped");
          if (status & pt_detail::StepUnconverged) add("unconverged");
          if (status & pt_detail::StepInterrupted) add("interrupted");
          if (status & pt_detail::StepBracketOk) add("bracket-ok");
          if (status & pt_detail::StepInsensitive) add("insensitive");
          if (status & pt_detail::StepHypersensitive) add("hypersensitive");
          return text;
        };
        std::ofstream scale_file("hpo_step_scales.tsv", std::ios::trunc);
        if (!scale_file) {
          HPOLog::instance().warning(
              "ANNEAL_PT: cannot open hpo_step_scales.tsv for writing");
        } else {
          /* D9: `sigma`/`sigma/width` are the PRODUCTION step actually
           * used by the run (post-pilot). Everything from `acceptance` to
           * `unusable` was measured during the COORDINATE phase, at
           * `coord_sigma` -- larger by sqrt(dim)/pilot_multiplier, ~26x for
           * such runs. The two used to share a row with only the production
           * sigma printed, so a `support-capped` row (meaning the
           * coordinate scale reached 1.0 x width) displayed
           * sigma/width = 0.038 and read as a contradiction. Both scales
           * are now columns; each measurement sits next to the sigma it was
           * taken at. */
          scale_file << "slot\ttemperature\tparameter\tsigma\tsigma/width"
                        "\tcoord_sigma\tcoord_sigma/width"
                        "\tacceptance\tstatus\tmean_alpha\tattempts"
                        "\tbackoffs\tunusable\tpilot_multiplier"
                        "\tpilot_multiplier_moved"
                        "\tpilot_acceptance\tpilot_mean_alpha"
                        "\tpilot_attempts\tpilot_backoffs\tpilot_unusable"
                        "\tpilot_status\n";
          scale_file << std::scientific << std::setprecision(17);
        }
        std::size_t status_ok = 0;
        std::size_t status_floor = 0;
        std::size_t status_capped = 0;
        std::size_t status_unconverged = 0;
        std::size_t status_interrupted = 0;
        std::size_t status_bracket_ok = 0;
        std::size_t status_insensitive = 0;
        std::size_t status_hypersensitive = 0;
        std::size_t unexplained_dimensions = 0;
        std::size_t unresolved_pilots = 0;
        for (std::size_t k = 0; k < K; ++k) {
          std::size_t coordinate_accepted = 0;
          std::size_t coordinate_validation_attempts = 0;
          std::size_t coordinate_total_attempts = 0;
          std::size_t coordinate_backoffs = 0;
          std::size_t coordinate_unusable = 0;
          long double coordinate_alpha_sum = 0.0L;
          double sigma_min = std::numeric_limits<double>::infinity();
          double sigma_max = 0.0;
          const std::size_t slot_base = k * dim * fields;
          for (std::size_t i = 0; i < dim; ++i) {
            const std::size_t p = slot_base + i * fields;
            const double sigma = gathered_step_records[p + rc_sigma];
            const double coord_sigma = gathered_step_records[p + rc_coord_sigma];
            const std::size_t accepted = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_accepted]));
            const std::size_t validation_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_attempts_val]));
            const std::size_t total_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_attempts]));
            const std::size_t backoffs = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_backoffs]));
            const unsigned status = static_cast<unsigned>(
                std::llround(gathered_step_records[p + rc_status]));
            const double mean_alpha = gathered_step_records[p + rc_mean_alpha];
            const std::size_t unusable = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_unusable]));
            const double acceptance = validation_attempts > 0
                ? static_cast<double>(accepted) / validation_attempts : 0.0;
            const std::size_t pilot_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_pilot_proposed]));
            const std::size_t pilot_backoffs = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_pilot_backoffs]));
            const double pilot_multiplier =
                gathered_step_records[p + rc_pilot_multiplier];
            const std::size_t pilot_validation_accepted =
                static_cast<std::size_t>(
                    std::llround(gathered_step_records[p + rc_pilot_val_accepted]));
            const std::size_t pilot_validation_attempts =
                static_cast<std::size_t>(
                    std::llround(gathered_step_records[p + rc_pilot_val_attempts]));
            const double pilot_mean_alpha =
                gathered_step_records[p + rc_pilot_val_mean_alpha];
            const std::size_t pilot_unusable = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + rc_pilot_unusable]));
            const unsigned pilot_status = static_cast<unsigned>(
                std::llround(gathered_step_records[p + rc_pilot_status]));
            const bool pilot_moved =
                gathered_step_records[p + rc_pilot_moved] != 0.0;
            const double pilot_acceptance = pilot_validation_attempts > 0
                ? static_cast<double>(pilot_validation_accepted) /
                      pilot_validation_attempts
                : 0.0;
            const std::string parameter = sp.parameter_labels.size() == dim
                ? sp.parameter_labels[i]
                : logidx::dim_label(i);
            if (scale_file) {
              const double width = sp.high[i] - sp.low[i];
              scale_file << logidx::uis(k) << '\t' << T[k] << '\t'
                         << parameter << '\t'
                         << sigma << '\t'
                         << sigma / width << '\t'
                         << coord_sigma << '\t'
                         << coord_sigma / width << '\t'
                         << acceptance << '\t' << status_text(status) << '\t'
                         << mean_alpha << '\t' << total_attempts << '\t'
                         << backoffs << '\t' << unusable << '\t'
                         << pilot_multiplier << '\t'
                         << (pilot_moved ? 1 : 0) << '\t'
                         << pilot_acceptance << '\t'
                         << pilot_mean_alpha << '\t' << pilot_attempts << '\t'
                         << pilot_backoffs << '\t' << pilot_unusable << '\t'
                         << status_text(pilot_status) << '\n';
            }
            coordinate_accepted += accepted;
            coordinate_validation_attempts += validation_attempts;
            coordinate_total_attempts += total_attempts;
            coordinate_backoffs += backoffs;
            coordinate_unusable += unusable;
            coordinate_alpha_sum +=
                static_cast<long double>(mean_alpha) * validation_attempts;
            sigma_min = std::min(sigma_min, sigma);
            sigma_max = std::max(sigma_max, sigma);
            if (status == pt_detail::StepOk) ++status_ok;
            if (status & pt_detail::StepNumericalFloor) ++status_floor;
            if (status & pt_detail::StepSupportCapped) ++status_capped;
            if (status & pt_detail::StepUnconverged) ++status_unconverged;
            if (status & pt_detail::StepInterrupted) ++status_interrupted;
            if (status & pt_detail::StepBracketOk) ++status_bracket_ok;
            if (status & pt_detail::StepInsensitive) ++status_insensitive;
            if (status & pt_detail::StepHypersensitive)
              ++status_hypersensitive;
            /* D28: "unexplained" = unconverged with nothing naming the
             * cause -- neither an interim floor/cap saturation nor a
             * sensitivity classification. */
            if ((status & pt_detail::StepUnconverged) &&
                !(status & (pt_detail::StepNumericalFloor |
                            pt_detail::StepSupportCapped |
                            pt_detail::StepInsensitive |
                            pt_detail::StepHypersensitive)))
              ++unexplained_dimensions;
          }
          /* Slot-wide pilot fields are replicated across every dim record,
           * so dim 0 carries them. Named offsets: see the layout enum. */
          const std::size_t pilot_attempts = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + rc_pilot_proposed])) : 0;
          const std::size_t pilot_backoffs = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + rc_pilot_backoffs])) : 0;
          const double pilot_multiplier = dim > 0
              ? gathered_step_records[slot_base + rc_pilot_multiplier] : 1.0;
          const bool pilot_moved = dim > 0 &&
              gathered_step_records[slot_base + rc_pilot_moved] != 0.0;
          const std::size_t pilot_accepted = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + rc_pilot_val_accepted])) : 0;
          const std::size_t pilot_validation_attempts = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + rc_pilot_val_attempts])) : 0;
          const double pilot_mean_alpha = dim > 0
              ? gathered_step_records[slot_base + rc_pilot_val_mean_alpha] : 0.0;
          const unsigned pilot_status = dim > 0
              ? static_cast<unsigned>(std::llround(
                    gathered_step_records[slot_base + rc_pilot_status]))
              : pt_detail::StepOk;
          if (pilot_status & pt_detail::StepUnconverged) ++unresolved_pilots;
          std::ostringstream diag;
          diag << std::scientific << std::setprecision(3)
               << "ANNEAL_PT step calibration slot=" << logidx::uis(k)
               << "  T=" << T[k]
               << "  coordinate_acc=" << std::fixed << std::setprecision(1)
               << (coordinate_validation_attempts > 0
                       ? 100.0 * coordinate_accepted /
                             coordinate_validation_attempts
                       : 0.0) << "%"
               << "  coordinate_alpha="
               << (coordinate_validation_attempts > 0
                       ? 100.0 * static_cast<double>(coordinate_alpha_sum) /
                             coordinate_validation_attempts
                       : 0.0) << "%"
               << "  coordinate_attempts=" << coordinate_total_attempts
               << "  coordinate_backoffs=" << coordinate_backoffs
               << "  coordinate_unusable=" << coordinate_unusable
               << "  pilot_acc="
               << (pilot_validation_attempts > 0
                       ? 100.0 * pilot_accepted / pilot_validation_attempts
                       : 0.0) << "%"
               << "  pilot_alpha=" << 100.0 * pilot_mean_alpha << "%"
               << "  pilot_attempts=" << pilot_attempts
               << "  pilot_backoffs=" << pilot_backoffs
               << "  pilot_multiplier=" << std::scientific
               << std::setprecision(3) << pilot_multiplier
               /* D12: 1.0 from "the controller converged there" reads the
                * same as 1.0 from "the controller never moved". Only the
                * TSV carried the distinction; the journal does now too. */
               << (pilot_moved ? " (moved)" : " (never moved)")
               << "  pilot_status=" << status_text(pilot_status)
               << "  sigma=[" << sigma_min << ", " << sigma_max << "]";
          HPOLog::instance().info(diag.str());
        }
        {
          std::ostringstream summary;
          summary << "ANNEAL_PT step calibration summary: dimensions="
                  << (K * dim) << "  ok=" << status_ok
                  << "  bracket-ok=" << status_bracket_ok
                  << "  numerical-floor=" << status_floor
                  << "  support-capped=" << status_capped
                  << "  unconverged=" << status_unconverged
                  << " (insensitive=" << status_insensitive
                  << " hypersensitive=" << status_hypersensitive
                  << " unexplained=" << unexplained_dimensions << ")"
                  << "  interrupted=" << status_interrupted
                  << "  unresolved-pilots=" << unresolved_pilots;
          HPOLog::instance().info(summary.str());
        }
        if (warmup_code == 0 &&
            (unexplained_dimensions > 0 || unresolved_pilots > 0)) {
          HPOLog::instance().warning(
              "ANNEAL_PT automatic step warm-up continuing with best tested "
              "scales: " + std::to_string(unexplained_dimensions) +
              " unexplained coordinate miss(es), " +
              std::to_string(unresolved_pilots) +
              " unresolved full-vector pilot(s); inspect the calibration-only "
              "hpo_step_scales.tsv file");
        }
        if (warmup_code == 0 &&
            (status_insensitive > 0 || status_hypersensitive > 0)) {
          HPOLog::instance().hint(
              "step warm-up sensitivity classification: 'insensitive' dims "
              "(best alpha ~ 1 at every tested scale) do not affect the "
              "objective over their OPTIM range -- consider dropping or "
              "fixing them; 'hypersensitive' dims (best alpha ~ 0) collapse "
              "acceptance at every tested scale -- consider narrowing or "
              "log-scaling their OPTIM range");
        }
        HPOLog::instance().info(
            "ANNEAL_PT automatic step warm-up spent " +
            std::to_string(warmup_spent_end - warmup_spent_begin) +
            " host eval(s); production epoch/swap/flow statistics reset");
      }

      if (warmup_code != 0) {
        stop_reason = warmup_code == 3 ? StopReason::StopVal
                    : warmup_code == 2 ? StopReason::MaxTime
                                       : StopReason::MaxEval;
        proceed = false;
      }
    }

    /* D27: record each slot's calibrated sigma as the diagonal-RAM trust
     * anchor.  The post-warm-up V encodes the legitimate per-dim
     * sensitivity profile (including deliberately tiny sigmas on
     * hypersensitive wide-box dims); production adaptation may move
     * around it but not drift without bound.  Guarded on the RAM
     * condition, not the auto condition, so it stays correct if the two
     * ever decouple. */
    if (proceed && !kb.corana &&
        kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE)
      for (auto& c : chains) c.V_baseline = c.V;

    /* D12: second calibration pass.
     *
     * Pass 1 estimated T0 with the generic STEP_FRAC-width kernel -- its
     * only job was to give the warm-up a ladder to calibrate against.
     * The kernel the run actually uses is the warm-up's frozen per-slot
     * sigma, which can sit an order of magnitude away from that generic
     * scale, and a temperature derived for a kernel the run never makes
     * moves with anchors the whole ladder wrong. Re-estimate T_HOT from
     * the same seed anchor under the HOTTEST slot's calibrated kernel
     * and rebuild the ladder. Skipped when T_HOT is explicit (no
     * calibration ran) or STEP_MODE is FIXED (pass 1's kernel IS the
     * production kernel). Pass-2 probes draw from their own stream
     * (SALT_PROBES, index 1) and count against MAXEVAL like any probe. */
    if (proceed && !kb.t_start && !kb.corana &&
        kb.step_mode != anneal_enums::STEP_MODE::FIXED && dim > 0) {
      std::vector<double> hot_V(dim, 0.0);
      const int hot_owner = world - 1;   /* global slot K-1 lives on the
                                          * last rank's block */
      if (rank == hot_owner && n_local > 0)
        for (std::size_t i = 0; i < dim; ++i)
          hot_V[i] = chains[n_local - 1].V[i];
      if (world > 1) hpo_bcast(hot_V, hot_owner);

      double t_hot_pass2 = 0.0;
      /* Kept whole, like the pass-1 calibration error above. */
      bool pass2_threw = false;
      std::exception_ptr pass2_error;
      if (rank == 0) {
        try {
          Vec xa = x0;
          double fa = f0;
          std::mt19937_64 probe2 =
              anneal::make_stream(seed, anneal::SALT_PROBES, 1);
          const auto t2 = anneal::calibrate_t0(
              sp, kb, hot_V, xa, fa, l_best_x, l_best, host, probe2, ctx,
              "ANNEAL_PT calibration pass 2");
          g_best = std::min(g_best, l_best);
          if (t2.aborted) {
            stop_reason = t2.budget_exhausted
                              ? StopReason::MaxEval
                              : (t2.stopval_hit
                                     ? StopReason::StopValOnProbe
                                     : (t2.external_stop
                                            ? StopReason::ExternalOnProbe
                                            : StopReason::MaxTimeOnProbe));
            proceed = false;
          } else {
            t_hot_pass2 = t2.T0;
          }
        } catch (...) {
          pass2_threw = true;
          pass2_error = std::current_exception();
        }
      }
      if (world > 1) {
        int code = 0;
        if (rank == 0)
          code = pass2_threw ? -1
                             : (proceed ? 0 : 1 + static_cast<int>(stop_reason));
        hpo_bcast(code, 0);
        if (code == -1) {
          if (rank == 0) std::rethrow_exception(pass2_error);
          throw std::runtime_error(
              "TADAH:ANNEAL_PT: rank 0 aborted during calibration pass 2");
        }
        std::vector<double> pkt{t_hot_pass2};
        hpo_bcast(pkt, 0);
        t_hot_pass2 = pkt[0];
        if (rank != 0 && code != 0) {
          stop_reason = static_cast<StopReason>(code - 1);
          proceed = false;
        }
      } else if (pass2_threw) {
        std::rethrow_exception(pass2_error);
      }

      if (proceed && t_hot_pass2 > 0.0) {
        const double ratio = T_hot > 0.0 ? t_hot_pass2 / T_hot : 0.0;
        const double t_cold_pass2 =
            kb.t_cold ? *kb.t_cold : t_hot_pass2 / kb.ladder_span;
        if (!(t_cold_pass2 > 0.0 && t_cold_pass2 < t_hot_pass2))
          throw std::runtime_error(
              "TADAH:ANNEAL_PT: calibration pass 2 produced an invalid "
              "ladder (T_COLD=" + std::to_string(t_cold_pass2) +
              ", T_HOT=" + std::to_string(t_hot_pass2) +
              "); set PARAM T_HOT/T_COLD explicitly");
        T_hot = t_hot_pass2;
        T_cold = t_cold_pass2;
        log_t_cold = std::log(static_cast<long double>(T_cold));
        log_t_hot  = std::log(static_cast<long double>(T_hot));
        T = pt_detail::make_temperature_ladder(
            T_cold, T_hot, K, kb.ladder_power);
        if (rank == 0) {
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PT calibration pass 2: T_HOT=" << T_hot
              << " from the calibrated production kernel (pass 1 used the "
                 "generic STEP_FRAC kernel; pass2/pass1 ratio="
              << std::fixed << std::setprecision(2) << ratio
              << "); ladder rebuilt";
          HPOLog::instance().info(oss.str());
          std::ostringstream head2;
          head2 << "ANNEAL_PT ladder T per slot after calibration pass 2 ("
                << logidx::uis(0) << "=coldest .. " << logidx::uis(K - 1)
                << "=hottest):";
          HPOLog::instance().info(logidx::grid(
              head2.str(), K, [&](std::ostringstream& o, std::size_t s) {
                o << std::scientific << std::setprecision(3) << "  " << T[s];
              }));
        }
      }
    }

    /* Per-pair swap-acceptance bookkeeping. Degenerate = equal-loss
     * exchanges: accepting them with probability 1 is exact Metropolis
     * (exp(0) = 1), but they carry NO tempering information, so they are
     * counted separately -- a pair whose acceptance is dominated by
     * degenerate exchanges (e.g. both replicas pinned at FAILSCORE)
     * must not read as healthy (D16). */
    std::vector<std::size_t> swap_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept (K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_degenerate(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_degenerate_win(K > 1 ? K - 1 : 0, 0);
    /* Independent adaptation windows.  Only the rank owning the colder
     * side of a pair increments its local entry; every exchange round
     * allreduces these counters before deciding whether to update. */
    std::vector<std::size_t> ladder_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_accept(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_attempt_global(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_accept_global(K > 1 ? K - 1 : 0, 0);
    std::size_t ladder_update_count = 0;
    bool ladder_adaptation_active = kb.ladder_adapt;
    /* D26 convergence freeze: pooled per-pair acceptance evidence (see
     * pt_detail::ladder_freeze_step).  100 pooled attempts/pair = 5
     * default windows; the pooled 2-sigma test then resolves ~+/-9
     * acceptance points. */
    pt_detail::LadderFreezeEvidence ladder_freeze_evidence;
    /* SWAP_WINDOW-epoch window of the same counters (reset after each
     * windowed table) -- shows acceptance DRIFT the cumulative table
     * averages away. */
    std::vector<std::size_t> swap_attempt_win(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept_win (K > 1 ? K - 1 : 0, 0);
    /* D29: last COMPLETED window, kept for the final summary. */
    std::vector<std::size_t> swap_attempt_last(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept_last (K > 1 ? K - 1 : 0, 0);
    std::size_t swap_last_end_epoch = 0;
    std::vector<std::size_t> step_attempt_win(K, 0);
    std::vector<std::size_t> step_accept_win(K, 0);
    std::vector<std::size_t> step_health_streak(K, 0);

    /* Ladder-flow bookkeeping (Katzgraber-style up/down tagging),
     * single-process runs only -- walker identities would have to ride
     * the MPI boundary-swap messages to stay exact across ranks.
     * walker_at[slot] tracks the STATE lineage (swaps move states, not
     * slots); each walker is tagged by the extreme it visited last.
     * A +1-tagged walker reaching the hot end = one genuine cold->hot
     * traversal (and vice versa), so the counts are exact -- no
     * initial-transient overcount. min(up, down) ~ completed round
     * trips, the literature's tempering-effectiveness currency. */
    std::vector<std::size_t> walker_at(K);
    std::vector<signed char> walker_tag(K, 0);  /* 0 none, +1 cold, -1 hot */
    std::size_t flow_up = 0, flow_down = 0;     /* cold->hot, hot->cold */
    for (std::size_t i = 0; i < K; ++i) walker_at[i] = i;
    /* Measured-transport window. The cumulative flow counters above answer
     * "has the ladder EVER communicated"; they cannot say "is it still
     * communicating", because a ladder that severs at hour 16 keeps
     * reporting the traversals it banked before that. So the same tagging
     * is also accumulated over a rolling window and reported separately.
     *
     * Window length: 100 * (K-1) exchange rounds. K-1 is the EXACT
     * structural minimum for one one-way traversal (a DEO round moves a
     * state at most one slot), so the window is 100 one-way minima -- long
     * enough that a healthy ladder completes several traversals inside it
     * at any plausible acceptance, short enough to localise when transport
     * stops. Derived from K, so no new PARAM key.
     *
     * slot_seen[s*K + w] marks "walker w occupied slot s". The per-slot
     * population count is what localises WHERE the ladder is severed:
     * under mixing every slot is visited by ~all K walkers, while a
     * severed ladder leaves the far side circulating among a handful. */
    const std::size_t transport_window =
        pt_detail::kTransportWindowFactor * (K > 1 ? K - 1 : 1);
    std::size_t flow_up_win = 0, flow_down_win = 0;
    std::vector<char> slot_seen(K * K, 0), slot_seen_win(K * K, 0);
    auto flow_touch_extremes = [&]() {
      if (world != 1 || K < 2) return;
      {
        const std::size_t w = walker_at[0];          /* coldest slot */
        if (walker_tag[w] == -1) { ++flow_down; ++flow_down_win; }
        walker_tag[w] = +1;
      }
      {
        const std::size_t w = walker_at[K - 1];      /* hottest slot */
        if (walker_tag[w] == +1) { ++flow_up; ++flow_up_win; }
        walker_tag[w] = -1;
      }
      for (std::size_t s = 0; s < K; ++s) {
        slot_seen[s * K + walker_at[s]] = 1;
        slot_seen_win[s * K + walker_at[s]] = 1;
      }
    };
    flow_touch_extremes();   /* initial tags only: all tags are still 0 */

    /* --- Corana per-dimension acceptance census -------------------------
     *
     * Corana's step adaptation has ALWAYS been per dimension:
     * adjust_corana_step divides or multiplies V[i] from that coordinate's
     * own nA[i]/nP[i], never from a rung-wide scalar. What was missing was
     * visibility -- the journal printed one accept% per epoch for the whole
     * ladder, so a run could sit at a healthy 30% while hundreds of
     * coordinates never moved at all and nothing said so. The counters that
     * feed the adaptation cannot be read afterwards either: corana_sweep
     * consumes and zeroes them every NS sweeps. ChainState carries a
     * parallel never-reset copy for this.
     *
     * Wire layout: cc_fields doubles per (slot, dim). The WHOLE table is
     * gathered rather than each rank shipping a packed per-slot summary,
     * because a packed summary needs two independent readers (the journal
     * formatter and the TSV writer) whose field offsets can silently
     * desynchronise -- precisely the failure the step-scale layout enum
     * above was introduced to end. One layout, one reader. The cost is
     * K*dim*cc_fields doubles per report (816 dims x 16 slots ~ 313 kB) once
     * per SWAP_WINDOW epochs, against K host evaluations per epoch: not
     * measurable.
     *
     * Everything here is a no-op unless PARAM CORANA is ON. */
    enum : std::size_t { cc_accepted = 0, cc_proposed, cc_sigma, cc_fields };
    const bool census_on = kb.corana && dim > 0;
    std::vector<int> census_counts(static_cast<std::size_t>(world), 0);
    std::size_t census_total = 0;
    if (census_on) {
      for (int r = 0; r < world; ++r) {
        const std::size_t count = slot_count(r) * dim * cc_fields;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::length_error(
              "TADAH:ANNEAL_PT: Corana acceptance census exceeds MPI count "
              "limit");
        census_counts[static_cast<std::size_t>(r)] = static_cast<int>(count);
        census_total += count;
      }
    }
    /* Rank-0 snapshots. `census_at_window` is the table as it stood at the
     * last report, `census_window` the delta of the last COMPLETED
     * SWAP_WINDOW window -- the same bookkeeping as the D29 swap tables,
     * for the same reason: a whole-run cumulative average hides late
     * drift, which for Corana is exactly the thing that goes wrong. */
    std::vector<double> census_at_window, census_window;
    std::size_t census_at_window_epoch = 0, census_window_end_epoch = 0;

    /* Collective: EVERY rank must call this, rank 0 receives. */
    auto census_gather = [&](std::vector<double>& all) {
      std::vector<double> local;
      local.reserve(n_local * dim * cc_fields);
      for (std::size_t li = 0; li < n_local; ++li) {
        const auto& c = chains[li];
        for (std::size_t i = 0; i < dim; ++i) {
          local.push_back(i < c.corana_census_nA.size()
                              ? static_cast<double>(c.corana_census_nA[i])
                              : 0.0);
          local.push_back(i < c.corana_census_nP.size()
                              ? static_cast<double>(c.corana_census_nP[i])
                              : 0.0);
          local.push_back(i < c.V.size() ? c.V[i] : 0.0);
        }
      }
      all.assign(rank == 0 ? census_total : std::size_t{0}, 0.0);
      hpo_gatherv(local, all, census_counts);
    };

    /* One slot's cells out of a gathered table. `base`, when given, turns
     * the cumulative counters into a window delta. */
    auto census_slot = [&](const std::vector<double>& all,
                           const std::vector<double>* base,
                           std::size_t k) {
      std::vector<corana_census::DimAcceptance> cells(dim);
      const std::size_t slot_base = k * dim * cc_fields;
      for (std::size_t i = 0; i < dim; ++i) {
        const std::size_t p = slot_base + i * cc_fields;
        double a = all[p + cc_accepted];
        double q = all[p + cc_proposed];
        if (base) {
          a -= (*base)[p + cc_accepted];
          q -= (*base)[p + cc_proposed];
        }
        cells[i].accepted =
            static_cast<std::size_t>(std::llround(std::max(0.0, a)));
        cells[i].proposed =
            static_cast<std::size_t>(std::llround(std::max(0.0, q)));
        cells[i].sigma = all[p + cc_sigma];
      }
      return cells;
    };

    auto census_label = [&](std::size_t i) {
      return sp.parameter_labels.size() == dim ? sp.parameter_labels[i]
                                               : logidx::dim_label(i);
    };

    /* --- epoch loop ------------------------------------------------------
     * Fixed per-epoch collective order (every rank, every epoch):
     * [allreduce spent] -> local rungs -> [allreduce stats+stopcode] ->
     * boundary sendrecvs -> [minloc best] -> [bcast rank-0 stop code]
     * (-> [bcast best-x] when RESET_TO_BEST). Ranks therefore always
     * agree on loop exits and no collective can be left unmatched. */
    if (proceed) {
      std::vector<anneal::RungStats> rstats(n_local);
      std::uniform_real_distribution<double> udist(0.0, 1.0);
      /* Planned total for the epoch=current/total display: min of the
       * budget bound (all replicas rung in lockstep) and, in hybrid
       * mode (RT < 1), the ladder's cooling bound on the hot end.
       * 0 = unbounded (pure PT with no MAXEVAL) -> plain counter. */
      std::size_t planned_epochs = 0;
      /* Cooperative shutdown must not cost the run its own diagnostics.
       * HPO_Host::operator() throws ShutdownRequested at the next eval,
       * and that used to unwind straight past the end-of-run block --
       * so a wall-clock-limited or scancel'd PT run produced NO whole-run
       * mixing profile, NO new-best-per-slot histogram and NO final swap
       * table, which for a transport study is the entire deliverable.
       * Absorb it here, report, then re-throw so hpo_run() still writes
       * its SHUTDOWN trailer and the exit status is unchanged.
       *
       * world > 1 re-throws immediately: the finalisation below runs MPI
       * collectives, and a rank that reached them while a peer unwound
       * would hang instead of exiting. Signals are delivered to the whole
       * job step in practice, but "in practice" is not a guarantee worth
       * a deadlock. Every diagnostic absorbed here is world == 1 only
       * anyway -- walker identities never cross the rank boundary. */
      try {
      for (;;) {
        const auto epoch_wall_t0 = std::chrono::steady_clock::now();
        const std::uint64_t spent_global =
            world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t remaining =
            max_eval_global > spent_global ? max_eval_global - spent_global : 0;
        if (remaining == 0) { stop_reason = StopReason::MaxEval; break; }
        if (planned_epochs == 0 && max_eval_global != HUGE_UNSIGNED) {
          planned_epochs = epoch_cost > 0
              ? static_cast<std::size_t>(remaining / epoch_cost +
                                         (remaining % epoch_cost ? 1 : 0))
              : 0;
          /* Plan line -- printed once, from the same arithmetic that
           * feeds the epoch=k/total display (an upper bound: STOPVAL /
           * best-stall / MAXTIME end the run earlier). */
          if (rank == 0 && planned_epochs > 0) {
            const std::size_t rounds = planned_epochs / kb.swap_every +
                (planned_epochs % kb.swap_every != 0 ? 1 : 0);
            std::ostringstream oss;
            oss << "ANNEAL_PT plan: ~" << planned_epochs
                << " epoch(s) (budget left " << remaining
                << " evals; upper bound), ~" << rounds
                << " exchange round(s)";
            HPOLog::instance().info(oss.str());
            /* Round-trip feasibility gates. The floor is STRUCTURAL: a
             * DEO exchange round moves a state at most one slot, so one
             * cold<->hot round trip needs >= 2*(K-1) rounds even at
             * 100% swap acceptance -- below that, a complete round trip is
             * structurally impossible (WARNING). Above it, the common 25%
             * planning approximation gives 2*(K-1)/0.25 rounds per nominal
             * trip; it is reported only as a heuristic HINT, while measured
             * ladder-flow counts remain authoritative. */
            if (K >= 2) {
              const std::size_t rt_floor = 2 * (K - 1);
              if (rounds < rt_floor) {
                std::ostringstream w;
                w << "ANNEAL_PT: planned exchange rounds " << rounds
                  << " < 2*(REPLICAS-1) = " << rt_floor
                  << " -- one cold<->hot ladder round trip is impossible "
                     "even at 100% swap acceptance (a DEO round moves a "
                     "state at most one slot). Exchanges can still couple "
                     "adjacent slots, but no lineage can complete a round trip. ";
                w << "Raise MAXEVAL or reduce REPLICAS/SWAP_EVERY.";
                HPOLog::instance().warning(w.str());
              } else if (rounds < 4 * rt_floor) {
                std::ostringstream h;
                h << "ANNEAL_PT: planned exchange rounds " << rounds
                  << " < 2*(REPLICAS-1)/0.25 = " << 4 * rt_floor
                  << " -- the 25%-acceptance planning heuristic yields fewer "
                     "than ~1 nominal ladder round trip (literature reference "
                     "values: Kone-Kofke 23%, Predescu 39% for optimization). "
                     "Measured ladder-flow counts are authoritative. ";
                h << "Consider a larger MAXEVAL or fewer REPLICAS.";
                HPOLog::instance().hint(h.str());
              }
            }
          }
        }
        int pre_time = time_exceeded() ? 1 : 0;
        if (world > 1) pre_time = hpo_allreduce_max(pre_time);
        if (pre_time) {
          stop_reason = StopReason::MaxTime; break;
        }

        /* Deterministic per-epoch budget slicing: full rungs while the
         * budget allows, else floor(remaining/K) with the remainder
         * given to the coldest slots. Computed identically on every
         * rank from the GLOBAL remaining count. */
        std::vector<std::uint64_t> allowance(K, rung_cost);
        if (remaining < epoch_cost) {
          const std::uint64_t base = remaining / K;
          const std::uint64_t rem  = remaining % K;
          for (std::size_t k = 0; k < K; ++k)
            allowance[k] = base + (k < rem ? 1 : 0);
        }

        /* f at epoch start (post previous exchange), for the per-epoch
         * replica-state report's df column. */
        std::vector<double> f_start(n_local);
        for (std::size_t li = 0; li < n_local; ++li)
          f_start[li] = chains[li].f;
        /* D29: ladder adaptation and RT cooling mutate T later in this
         * same iteration; the epoch line must report the T this epoch's
         * rungs actually RAN at. */
        const double epoch_T_cold = T[0];
        const double epoch_T_hot = T[K - 1];

        pool.run_epoch(n_local, [&](std::size_t li) {
          const std::size_t k = off_local + li;
          warmstart::current_chain() = k;   /* global slot for warm-start key */
          HPOLog::set_eval_slot(static_cast<int>(k), K, T[k]);
          rstats[li] = anneal::run_sweep(chains[li], T[k], sp, kb, host, ctx,
                                         allowance[k]);
          if (kb.observer) {
            /* PRE-swap state, on the worker thread that owns the chain. */
            std::vector<double> xv(dim);
            for (std::size_t d = 0; d < dim; ++d) xv[d] = chains[li].x(d);
            kb.observer->on_rung_end(static_cast<int>(k), T[k], xv,
                                     chains[li].f);
          }
        });
        rethrow_pool_error("replica rung");

        std::size_t e_prop = 0, e_acc = 0, e_oob = 0;
        anneal::StepSaturation e_vsat;
        bool any_stopval = false, any_time = false;
        for (std::size_t li = 0; li < n_local; ++li) {
          e_prop += rstats[li].proposed;
          e_acc  += rstats[li].accepted;
          e_oob  += rstats[li].oob;
          /* Accumulated only where the sigma/step-health block below can
           * consume them; under CORANA or a non-ADAPTIVE step mode they
           * would grow unread for the whole run. */
          if (world == 1 && !kb.corana &&
              kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE &&
              kb.swap_window > 0) {
            step_attempt_win[li] += rstats[li].proposed;
            step_accept_win[li] += rstats[li].accepted;
          }
          if (rstats[li].stop == RungStop::StopVal) any_stopval = true;
          if (rstats[li].stop == RungStop::Time)    any_time    = true;
          if (chains[li].best_f < l_best) {
            l_best   = chains[li].best_f;
            l_best_x = chains[li].best_x;
          }
          if (kb.corana) {
            const auto s = anneal::step_saturation(chains[li].V, sp, kb);
            e_vsat.floor += s.floor;
            e_vsat.cap   += s.cap;
            e_vsat.total += s.total;
          }
        }
        g_best = std::min(g_best, l_best);
        any_time = any_time || time_exceeded();
        if (world > 1) {
          e_prop = hpo_allreduce_sum_sz(e_prop);
          e_acc  = hpo_allreduce_sum_sz(e_acc);
          e_oob  = hpo_allreduce_sum_sz(e_oob);
          if (kb.corana) {
            e_vsat.floor = hpo_allreduce_sum_sz(e_vsat.floor);
            e_vsat.cap   = hpo_allreduce_sum_sz(e_vsat.cap);
            e_vsat.total = hpo_allreduce_sum_sz(e_vsat.total);
          }
          int code = any_stopval ? 2 : (any_time ? 1 : 0);
          code = hpo_allreduce_max(code);
          any_stopval = (code == 2);
          any_time    = (code == 1);
        }
        total_proposed += e_prop;
        total_accepted += e_acc;
        total_oob      += e_oob;
        ++epoch_count;
        const bool maxeval_hit = max_eval_global != HUGE_UNSIGNED &&
                                 e_prop >= remaining;
        if (any_stopval)       stop_reason = StopReason::StopVal;
        else if (any_time)     stop_reason = StopReason::MaxTime;
        else if (e_prop == 0 || maxeval_hit)
                               stop_reason = StopReason::MaxEval;
        const bool terminal_epoch = stop_reason != StopReason::Done;

        /* Pre-swap snapshot for the per-epoch replica-state report: the
         * exchange below permutes x/f, while the report describes what
         * each slot's rung actually did. (The walker row is deliberately
         * POST-swap -- walkers only move at exchanges, so the pre-swap
         * occupancy is last epoch's post-swap row.) Single-process only
         * (chains are rank-local), same as the report itself. */
        std::vector<double> f_pre, f_delta;
        if (world == 1) {
          f_pre.resize(n_local);
          f_delta.resize(n_local);
          for (std::size_t li = 0; li < n_local; ++li) {
            f_pre[li]   = chains[li].f;
            f_delta[li] = chains[li].f - f_start[li];
          }
          /* Pinned-dim audit refresh: PRE-swap (slot V + the x that
           * walked in the slot this epoch -- the exchange permutes x/f,
           * never V). A slot whose x left the bound is unmarked; the
           * table prints with the milestone SUMMARY. */
          for (std::size_t li = 0; li < n_local; ++li)
            HPOLog::instance().record_pin_state(
                static_cast<int>(li), epoch_count,
                anneal::pinned_dims(chains[li], sp, kb));
        }
        /* Per-pair outcome of THIS epoch's exchange round for the
         * events row: 'x' swapped, '.' rejected, ' ' not attempted
         * (DEO parity). */
        std::vector<char> pair_event(K > 1 ? K - 1 : 0, ' ');

        /* --- exchange phase (deterministic; per-pair coin streams so
         * the decision is identical no matter which rank(s) own the
         * pair or how many ranks there are) --------------------------- */
        double ep_swap_rate = -1.0;
        if (!terminal_epoch && epoch_count % kb.swap_every == 0) {
          const std::size_t parity =
              (epoch_count / kb.swap_every) % 2;  /* DEO alternation */
          std::size_t att = 0, acc = 0;
          for (std::size_t i = parity; i + 1 < K; i += 2) {
            const bool i_mine = (i >= off_local && i < off_local + n_local);
            const bool j_mine = (i + 1 >= off_local && i + 1 < off_local + n_local);
            if (!i_mine && !j_mine) continue;
            std::mt19937_64 pair_rng = anneal::make_stream(
                seed, anneal::SALT_PT_SWAP,
                static_cast<std::uint64_t>(epoch_count) * K + i);
            if (i_mine && j_mine) {
              auto& ci = chains[i - off_local];
              auto& cj = chains[i + 1 - off_local];
              const double f_lo = ci.f, f_hi = cj.f;
              const bool degenerate = f_lo == f_hi;
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], f_lo, f_hi,
                  [&]() { return udist(pair_rng); });
              ++swap_attempt[i];
              if (kb.swap_window > 0) ++swap_attempt_win[i];
              if (degenerate) {
                ++swap_degenerate[i];
                if (kb.swap_window > 0) ++swap_degenerate_win[i];
              }
              /* Degenerate exchanges say nothing about whether the pair
               * gap is right; feeding their automatic acceptance into
               * the equalization controller (or its freeze evidence)
               * would adapt the ladder toward a false 100% signal. */
              if (ladder_adaptation_active && !degenerate)
                ++ladder_attempt[i];
              ++att;
              pair_event[i] = '.';
              if (accepted) {
                std::swap(ci.x, cj.x);
                std::swap(ci.f, cj.f);
                if (world == 1) std::swap(walker_at[i], walker_at[i + 1]);
                ++swap_accept[i];
                if (kb.swap_window > 0) ++swap_accept_win[i];
                if (ladder_adaptation_active && !degenerate)
                  ++ladder_accept[i];
                ++acc;
                pair_event[i] = 'x';
              }
              if (kb.observer)
                kb.observer->on_swap_attempt(
                    epoch_count, i, accepted, degenerate,
                    T[i], T[i + 1], f_lo, f_hi);
            } else {
              /* Boundary pair: I own exactly one side; my peer owns the
               * other. Both sides use the SAME pair coin stream when the
               * decision is probabilistic and compute the SAME result -- only f (and x
               * on accept) travel. */
              const std::size_t mine_g = i_mine ? i : i + 1;
              auto& c = chains[mine_g - off_local];
              const int peer = i_mine ? rank + 1 : rank - 1;
              std::vector<double> sendf{c.f}, recvf(1, 0.0);
              hpo_sendrecv(sendf, recvf, peer, static_cast<int>(i % 1024) * 2);
              const double f_i = i_mine ? c.f : recvf[0];
              const double f_j = i_mine ? recvf[0] : c.f;
              const bool degenerate = f_i == f_j;
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], f_i, f_j,
                  [&]() { return udist(pair_rng); });
              if (i_mine) {
                ++swap_attempt[i];
                if (kb.swap_window > 0) ++swap_attempt_win[i];
                if (degenerate) {
                  ++swap_degenerate[i];
                  if (kb.swap_window > 0) ++swap_degenerate_win[i];
                }
                if (ladder_adaptation_active && !degenerate)
                  ++ladder_attempt[i];
                ++att;
                if (accepted) {
                  ++swap_accept[i];
                  if (kb.swap_window > 0) ++swap_accept_win[i];
                  if (ladder_adaptation_active && !degenerate)
                    ++ladder_accept[i];
                  ++acc;
                }
                if (kb.observer)
                  kb.observer->on_swap_attempt(
                      epoch_count, i, accepted, degenerate,
                      T[i], T[i + 1], f_i, f_j);
              }
              if (accepted) {
                std::vector<double> sx(dim), rx(dim, 0.0);
                for (std::size_t d = 0; d < dim; ++d) sx[d] = c.x(d);
                hpo_sendrecv(sx, rx, peer, static_cast<int>(i % 1024) * 2 + 1);
                for (std::size_t d = 0; d < dim; ++d) c.x(d) = rx[d];
                c.f = i_mine ? f_j : f_i;
              }
            }
          }
          if (world > 1) {
            att = hpo_allreduce_sum_sz(att);
            acc = hpo_allreduce_sum_sz(acc);
          }
          ++exchange_rounds;
          ep_swap_rate = att > 0 ? static_cast<double>(acc) /
                                   static_cast<double>(att) : 0.0;

          if (ladder_adaptation_active) {
            bool ready = true;
            for (std::size_t p = 0; p + 1 < K; ++p) {
              ladder_attempt_global[p] = world > 1
                  ? hpo_allreduce_sum_sz(ladder_attempt[p])
                  : ladder_attempt[p];
              ladder_accept_global[p] = world > 1
                  ? hpo_allreduce_sum_sz(ladder_accept[p])
                  : ladder_accept[p];
              ready = ready &&
                      ladder_attempt_global[p] >= kb.ladder_adapt_every;
            }
            if (ready) {
              bool representable = T.front() > 0.0;
              for (std::size_t k = 1; k < K; ++k)
                representable = representable && T[k] > T[k - 1];
              if (!representable) {
                ladder_adaptation_active = false;
                if (rank == 0)
                  HPOLog::instance().warning(
                      "ANNEAL_PT ladder adaptation frozen: an update "
                      "produced a non-positive or non-distinct temperature; "
                      "the current ladder is kept");
              } else {
                const std::vector<double> old_temperature = T;
                const auto update = pt_detail::adapt_temperature_ladder(
                    T, ladder_attempt_global, ladder_accept_global,
                    ladder_update_count + 1, kb.ladder_adapt_gain,
                    kb.ladder_adapt_decay, kb.ladder_min_gap);
                T = update.temperature;
                ++ladder_update_count;

                /* D26: pooled-evidence freeze decision.  A pair is settled
                 * when its POOLED acceptance is within 2 binomial sigma of
                 * the pooled mean, or when the LADDER_MIN_GAP floor holds
                 * it open against a below-mean acceptance (gap-insensitive
                 * -- contracting further cannot help).  Zero-information
                 * windows (all pairs exactly 0% or 100%, e.g. an equal-loss
                 * plateau) neither advance nor destroy the evidence.
                 * Derived from the allreduced counters, so identical on
                 * every rank. */
                const bool ladder_freeze_now =
                    kb.ladder_freeze &&
                    pt_detail::ladder_freeze_step(
                        ladder_freeze_evidence, ladder_attempt_global,
                        ladder_accept_global, update.at_floor,
                        pt_detail::kLadderFreezeMinAttempts);

                if (rank == 0) {
                  const auto [accept_min, accept_max] = std::minmax_element(
                      update.acceptance.begin(), update.acceptance.end());
                  const auto [old_gap_min, old_gap_max] = std::minmax_element(
                      update.old_log_gap.begin(), update.old_log_gap.end());
                  const auto [new_gap_min, new_gap_max] = std::minmax_element(
                      update.new_log_gap.begin(), update.new_log_gap.end());
                  std::ostringstream summary;
                  summary << std::fixed << std::setprecision(1)
                          << "ANNEAL_PT ladder update=" << ladder_update_count
                          << " epoch=" << epoch_count
                          << "  acceptance=[" << 100.0 * *accept_min << "%, "
                          << 100.0 * *accept_max << "%]"
                          << std::scientific << std::setprecision(3)
                          << "  gain=" << update.gain
                          << "  log-gap(old)= [" << *old_gap_min << ", "
                          << *old_gap_max << "]"
                          << "  log-gap(new)= [" << *new_gap_min << ", "
                          << *new_gap_max << "]";
                  if (kb.ladder_min_gap > 0.0) {
                    const std::size_t n_floor = static_cast<std::size_t>(
                        std::count(update.at_floor.begin(),
                                   update.at_floor.end(), char{1}));
                    summary << "  floor-clamped=" << n_floor << "/"
                            << (K - 1);
                  }
                  HPOLog::instance().info(summary.str());

                  std::ostringstream uhead;
                  uhead << "ANNEAL_PT ladder T per slot after update "
                        << ladder_update_count << ":";
                  HPOLog::instance().info(logidx::grid(
                      uhead.str(), K,
                      [&](std::ostringstream& o, std::size_t k) {
                        o << std::scientific << std::setprecision(3)
                          << "  " << T[k];
                      }));

                  if (ladder_file) {
                    for (std::size_t p = 0; p + 1 < K; ++p) {
                      ladder_file
                          << ladder_update_count << '\t' << epoch_count << '\t'
                          << logidx::uis(p) << '\t'
                          << ladder_attempt_global[p] << '\t'
                          << ladder_accept_global[p] << '\t'
                          << update.acceptance[p] << '\t'
                          << old_temperature[p] << '\t'
                          << old_temperature[p + 1] << '\t'
                          << T[p] << '\t' << T[p + 1] << '\t'
                          << update.old_log_gap[p] << '\t'
                          << update.new_log_gap[p] << '\t'
                          << static_cast<int>(update.at_floor[p]) << '\n';
                    }
                    ladder_file.flush();
                  }
                }
                std::fill(ladder_attempt.begin(), ladder_attempt.end(),
                          std::size_t{0});
                std::fill(ladder_accept.begin(), ladder_accept.end(),
                          std::size_t{0});
                std::fill(ladder_attempt_global.begin(),
                          ladder_attempt_global.end(), std::size_t{0});
                std::fill(ladder_accept_global.begin(),
                          ladder_accept_global.end(), std::size_t{0});
                if (ladder_adaptation_active && ladder_freeze_now) {
                  ladder_adaptation_active = false;
                  if (rank == 0)
                    HPOLog::instance().info(
                        "ANNEAL_PT ladder adaptation frozen: per-pair "
                        "acceptance indistinguishable from equalized (<=2 "
                        "sigma) or held at the LADDER_MIN_GAP floor, on >= " +
                        std::to_string(pt_detail::kLadderFreezeMinAttempts) +
                        " pooled attempts/pair; ladder is final "
                        "(LADDER_FREEZE 0 disables this)");
                }
              }
            }
          }
          flow_touch_extremes();   /* states moved -- update up/down flow */
        }

        /* --- global best + rank-0 stop decisions ----------------------
         * Ownership resolved from the rank-LOCAL best (see l_best). */
        int best_owner = 0;
        if (world > 1) {
          double gb = std::isfinite(l_best) ? l_best
                                            : std::numeric_limits<double>::max();
          hpo_allreduce_minloc(gb, best_owner);
          if (gb < std::numeric_limits<double>::max())
            g_best = std::min(g_best, gb);
        }

        int rank0_code = 0;   /* 0 none, 1 best-stall */
        if (rank == 0 && !terminal_epoch && stall_armed &&
            epoch_count % stall_window == 0) {
          if (!anneal::loss_is_finite(g_best, sp.failscore)) {
            stall_windows = 0;
            stall_ref_best = g_best;
          } else {
            const double eps = std::max(
                stall_ftol_abs, stall_ftol_rel * std::fabs(g_best));
            if (std::isfinite(stall_ref_best) &&
                stall_ref_best - g_best <= eps)
              ++stall_windows;
            else
              stall_windows = 0;
            stall_ref_best = g_best;
            if (stall_windows >= kb.neps) rank0_code = 1;
          }
        }
        if (world > 1) hpo_bcast(rank0_code, 0);
        const bool plateau_hit = rank0_code == 1;

        /* --- reset-to-best: coldest slot only ------------------------- */
        if (!terminal_epoch && !plateau_hit && kb.reset_to_best) {
          std::vector<double> bx(dim, 0.0);
          if (world > 1) {
            if (rank == best_owner)
              for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
            hpo_bcast(bx, best_owner);
          } else {
            for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
          }
          if (rank == 0 && n_local > 0 && chains[0].f > g_best) {
            for (std::size_t d = 0; d < dim; ++d) chains[0].x(d) = bx[d];
            chains[0].f = g_best;
            chains[0].best_x = chains[0].x;
            chains[0].best_f = g_best;
          }
        }

        /* --- journal reports, every SWAP_WINDOW epochs (an epoch is one
         * sweep per replica, so per-epoch lines would be one line per
         * eval-round; SWAP_WINDOW 0 = cumulative summaries only).
         * Terminal and stall epochs always report so the last state
         * before a stop is on record. Rank 0 carries the global numbers;
         * other ranks stay quiet. ------------------------------------- */
        const bool report_epoch =
            (kb.swap_window > 0 && epoch_count % kb.swap_window == 0) ||
            terminal_epoch || plateau_hit;
        /* Corana per-dimension acceptance drift, on the report cadence.
         * COLLECTIVE -- deliberately outside the rank==0 guard below, and
         * gated only on quantities that are already globally agreed
         * (epoch_count, terminal_epoch, plateau_hit), so every rank enters
         * the gather together. */
        if (census_on && report_epoch) {
          std::vector<double> census_now;
          census_gather(census_now);
          const bool window_closed =
              kb.swap_window > 0 && epoch_count % kb.swap_window == 0;
          if (rank == 0 && census_now.size() == census_total) {
            const bool have_base = census_at_window.size() == census_now.size();
            const std::vector<double>* base =
                have_base ? &census_at_window : nullptr;
            /* State the exact epoch span rather than "last SWAP_WINDOW":
             * terminal and plateau epochs report off-cadence, so the
             * interval since the previous report is often shorter. */
            const std::string span =
                "epochs " +
                std::to_string(have_base ? census_at_window_epoch + 1 : 1) +
                "-" + std::to_string(epoch_count);
            std::vector<corana_census::SlotCensus> cs(K);
            for (std::size_t k = 0; k < K; ++k)
              cs[k] = corana_census::summarize(census_slot(census_now, base, k));
            HPOLog::instance().info(logidx::grid(
                "ANNEAL_PT Corana median per-dim acceptance % per slot (" +
                    span + "; per-slot, not per-walker):",
                K, [&](std::ostringstream& o, std::size_t k) {
                  o << "  " << std::setw(9) << std::fixed
                    << std::setprecision(1)
                    << (100.0 * cs[k].pct[corana_census::kP50]);
                }));
            std::size_t frozen_total = 0;
            for (const auto& c : cs) frozen_total += c.frozen;
            if (frozen_total > 0)
              HPOLog::instance().info(logidx::grid(
                  "ANNEAL_PT Corana dims with ZERO accepted moves per slot (of "
                  + std::to_string(dim) + "; " + span + "):",
                  K, [&](std::ostringstream& o, std::size_t k) {
                    o << "  " << std::setw(9) << cs[k].frozen;
                  }));
            if (window_closed) {
              if (have_base) {
                census_window.resize(census_now.size());
                for (std::size_t j = 0; j < census_now.size(); ++j)
                  census_window[j] = census_now[j] - census_at_window[j];
              } else {
                census_window = census_now;   /* first window = epochs 1-n */
              }
              /* sigma is a level, not a count: a delta of it is meaningless.
               * Carry the current value through so the TSV's window columns
               * sit next to the step they were measured at. */
              for (std::size_t j = cc_sigma; j < census_window.size();
                   j += cc_fields)
                census_window[j] = census_now[j];
              census_window_end_epoch = epoch_count;
            }
            census_at_window = census_now;
            census_at_window_epoch = epoch_count;
          }
        }
        if (rank == 0 && report_epoch) {
          const double rate = e_prop > 0
              ? static_cast<double>(e_acc) / static_cast<double>(e_prop) : 0.0;
          const double oob_pct = e_prop > 0
              ? 100.0 * static_cast<double>(e_oob) /
                    static_cast<double>(e_prop)
              : 0.0;
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << "ANNEAL_PT epoch=" << epoch_count;
          if (planned_epochs > 0) oss << "/" << planned_epochs;
          oss << "  T=[" << epoch_T_cold << ", " << epoch_T_hot << "]"
              << "  proposed=" << e_prop
              << "  accepted=" << std::fixed << std::setprecision(1)
              << (rate * 100.0) << "%"
              << "  oob=" << e_oob << " (" << oob_pct << "%)";
          if (ep_swap_rate >= 0.0)
            oss << "  swap_acc=" << (ep_swap_rate * 100.0) << "%";
          oss << std::scientific << std::setprecision(3)
              << "  f_cold=" << chains[0].f
              << "  best=" << g_best;
          if (kb.corana)
            oss << "  V_floor=" << e_vsat.floor << "/" << e_vsat.total
                << "  V_cap=" << e_vsat.cap << "/" << e_vsat.total;
          oss
              << "  epoch_wall=" << HPOLog::format_duration(
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - epoch_wall_t0)
                         .count());
          HPOLog::instance().info(oss.str());
          /* Per-pair swap-acceptance tables. DEO parity gives each pair
           * at most one attempt per 2*SWAP_EVERY epochs, so per-epoch
           * per-pair rates would be 0/100% noise -- the CUMULATIVE table
           * (every exchange epoch) is the live ladder diagnostic, and
           * the SWAP_WINDOW table (then reset) shows drift the
           * cumulative average hides. att/pair in the header states the
           * resolution; "-" = pair not attempted yet. Single-process
           * only: the counters are rank-local. */
          if (world == 1 && K >= 2 && ep_swap_rate >= 0.0) {
            auto pair_table = [&](const std::string& label,
                                  const std::vector<std::size_t>& attv,
                                  const std::vector<std::size_t>& accv) {
              HPOLog::instance().info(
                  pt_detail::format_pair_table(K, label, attv, accv));
            };
            pair_table("cumulative", swap_attempt, swap_accept);
            {
              /* D16: equal-loss exchanges accept with probability 1 by
               * exact Metropolis but move no tempering information; a
               * pair whose traffic is dominated by them must not read
               * as healthy from the acceptance table alone. */
              std::size_t deg_tot = 0;
              for (auto d : swap_degenerate) deg_tot += d;
              if (deg_tot > 0)
                pair_table("DEGENERATE% (equal-loss; cumulative)",
                           swap_attempt, swap_degenerate);
            }
            if (kb.swap_window > 0 && epoch_count % kb.swap_window == 0) {
              std::size_t att_tot = 0, deg_tot = 0;
              for (auto a : swap_attempt_win) att_tot += a;
              for (auto d : swap_degenerate_win) deg_tot += d;
              if (att_tot > 0) {
                pair_table("last " + std::to_string(kb.swap_window) +
                           " epoch(s)", swap_attempt_win, swap_accept_win);
                if (deg_tot > 0)
                  pair_table("DEGENERATE% (equal-loss; last window)",
                             swap_attempt_win, swap_degenerate_win);
                if (static_cast<double>(deg_tot) /
                        static_cast<double>(att_tot) >
                    pt_detail::kDegenerateWarnFrac) {
                  std::ostringstream w;
                  w << "ANNEAL_PT: " << std::fixed << std::setprecision(0)
                    << 100.0 * static_cast<double>(deg_tot) /
                           static_cast<double>(att_tot)
                    << "% of the last window's exchange attempts were "
                       "DEGENERATE (equal losses on both sides -- typical "
                       "when replicas sit on a FAILSCORE plateau). Their "
                       "automatic acceptance carries no tempering "
                       "information, so the swap tables overstate ladder "
                       "health; they are excluded from ladder adaptation.";
                  HPOLog::instance().warning(w.str());
                }
                /* D29: keep the completed window for the final summary,
                 * where the cumulative table averages late drift away. */
                swap_attempt_last = swap_attempt_win;
                swap_accept_last = swap_accept_win;
                swap_last_end_epoch = epoch_count;
                std::fill(swap_attempt_win.begin(), swap_attempt_win.end(),
                          std::size_t{0});
                std::fill(swap_accept_win.begin(), swap_accept_win.end(),
                          std::size_t{0});
                std::fill(swap_degenerate_win.begin(),
                          swap_degenerate_win.end(), std::size_t{0});
              }
            }
            /* THIS epoch's exchange outcomes -- the walker motion itself,
             * readable without diffing walker rows across epochs. */
            {
              HPOLog::instance().info(logidx::grid(
                  "ANNEAL_PT exchange this epoch "
                  "(x=swapped, .=rejected, blank=other parity):",
                  K - 1, [&](std::ostringstream& o, std::size_t p) {
                    o << "  " << std::setw(9) << pair_event[p];
                  }));
            }
          }
          /* Sparse per-slot new-best histogram for THIS epoch (chain-init
           * bests land in epoch 1's window), as its own tagged line.
           * Single-process only: the window counts are rank-local, so
           * under MPI a rank-0 line would silently miss the other ranks'
           * slots. */
          if (world == 1) {
            const auto win = HPOLog::instance().take_best_slot_window();
            if (!win.empty()) {
              std::ostringstream ossw;
              ossw << "ANNEAL_PT new bests per slot:";
              for (const auto& [s, n] : win)
                ossw << "  [" << logidx::uis(static_cast<std::size_t>(s))
                     << "]:" << n;
              HPOLog::instance().info(ossw.str());
            }
          }
          /* --- per-epoch replica-state report (pre-swap snapshot) ------
           * One grid block per metric, on the same [lo-hi] rows and
           * column widths as the startup ladder table so slot columns
           * line up down the journal. f/df/imp/acc describe the rung
           * phase just run; best_f is slot-resident (never swapped);
           * walker is the state lineage that produced this slot's rung.
           * Single-process only, like the histogram above. */
          if (world == 1) {
            auto grid = [&](const std::string& head, auto&& cell) {
              HPOLog::instance().info(logidx::grid(head, K, cell));
            };
            grid("ANNEAL_PT f per slot (pre-swap):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::scientific << std::setprecision(3)
                     << std::setw(9) << f_pre[s];
                 });
            grid("ANNEAL_PT df per slot (epoch):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << " " << std::scientific << std::setprecision(3)
                     << std::showpos << std::setw(10) << f_delta[s]
                     << std::noshowpos;
                 });
            grid("ANNEAL_PT best_f per slot:",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::scientific << std::setprecision(3)
                     << std::setw(9) << chains[s].best_f;
                 });
            grid("ANNEAL_PT imp per slot (epoch, new personal bests):",
                 [&](std::ostringstream& o, std::size_t s) {
                   o << "  " << std::setw(9) << rstats[s].improved;
                 });
            grid("ANNEAL_PT acc% per slot (epoch):",
                 [&](std::ostringstream& o, std::size_t s) {
                   const std::size_t p = rstats[s].proposed;
                   const long pct = p ? std::lround(
                       100.0 * static_cast<double>(rstats[s].accepted) /
                       static_cast<double>(p)) : 0;
                   o << "  " << std::setw(9) << pct;
                 });
            /* POST-swap walker lineup (what the next epoch runs), each
             * tagged by the last ladder end it touched (Katzgraber
             * up/down labeling, from the flow bookkeeping): a 'v'
             * walker descending toward slot 0 is a hot-end discovery
             * being carried to the cold end. */
            grid("ANNEAL_PT walker per slot (post-swap = next epoch's "
                 "lineup; ^=up walker (last touched cold end), v=down "
                 "walker (last touched hot end), .=no extreme yet):",
                 [&](std::ostringstream& o, std::size_t s) {
                   const std::size_t w = walker_at[s];
                   const char t = walker_tag[w] > 0
                                      ? '^' : (walker_tag[w] < 0 ? 'v' : '.');
                   o << "  " << std::setw(8) << logidx::uis(w) << t;
                 });
            /* Running end-to-end communication scoreboard (the same
             * counters as the end-of-run ladder-flow line): 0 round
             * trips = the replicas are still effectively independent
             * annealers. */
            {
              std::ostringstream of;
              of << "ANNEAL_PT ladder flow: traversals cold->hot="
                 << flow_up << "  hot->cold=" << flow_down
                 << "  (~" << std::min(flow_up, flow_down)
                 << " round trip(s) so far)";
              HPOLog::instance().info(of.str());
            }
            /* --- measured-transport report (windowed) --------------------
             * The cumulative line above cannot distinguish "communicating"
             * from "communicated once, an hour ago". This block reports the
             * same tagging over the last `transport_window` exchange rounds
             * and escalates on it, because measured transport -- not the
             * per-pair acceptance table -- is what says whether tempering is
             * doing anything. An adaptive ladder equalises acceptance, and
             * acceptance can stay textbook-uniform while transport is zero.
             *
             * ep_swap_rate >= 0 restricts this to epochs that actually ran
             * an exchange round (the file's own idiom, as used by the pair
             * tables), so the window does not re-fire when SWAP_EVERY > 1. */
            if (K >= 2 && ep_swap_rate >= 0.0 && transport_window > 0 &&
                exchange_rounds > 0 &&
                exchange_rounds % transport_window == 0) {
              const std::size_t rt_win = std::min(flow_up_win, flow_down_win);
              {
                std::ostringstream o;
                o << "ANNEAL_PT ladder transport (last " << transport_window
                  << " exchange round(s)): traversals cold->hot=" << flow_up_win
                  << "  hot->cold=" << flow_down_win << "  (" << rt_win
                  << " round trip(s) in window; cumulative " << flow_up << "/"
                  << flow_down << " over " << exchange_rounds << " round(s))";
                HPOLog::instance().info(o.str());
              }
              grid("ANNEAL_PT distinct walkers per slot (last " +
                       std::to_string(transport_window) +
                       " exchange round(s); " + std::to_string(K) +
                       " = fully mixed):",
                   [&](std::ostringstream& o, std::size_t s) {
                     std::size_t n = 0;
                     for (std::size_t w = 0; w < K; ++w)
                       n += slot_seen_win[s * K + w] ? 1 : 0;
                     o << "  " << std::setw(9) << n;
                   });
              /* Severity. Both branches state only what was measured over a
               * stated window, so both are exact -- but they differ in what
               * they prove. Zero tags at BOTH extremes means no lineage
               * reached either end: the ladder demonstrably did not carry
               * anything end to end, which is a structural failure of the
               * method and therefore a WARNING. One-sided flow is weaker
               * evidence -- it can also be a long round trip straddling the
               * window boundary -- so it is a HINT. */
              if (flow_up_win == 0 && flow_down_win == 0) {
                std::ostringstream w;
                w << "ANNEAL_PT: no ladder traversal in the last "
                  << transport_window << " exchange round(s) -- no replica "
                     "lineage reached either temperature extreme, so the "
                     "ensemble is running as " << K
                  << " effectively independent annealers over this window. "
                     "Per-pair swap acceptance does NOT diagnose this: an "
                     "adaptive ladder equalises acceptance, and acceptance "
                     "can stay in the usual 20-40% band while transport is "
                     "zero. See the distinct-walkers-per-slot row above -- "
                     "slots stuck near 1 mark where the ladder is severed.";
                HPOLog::instance().warning(w.str());
              } else if (rt_win == 0) {
                std::ostringstream h;
                h << "ANNEAL_PT: no COMPLETE round trip in the last "
                  << transport_window << " exchange round(s) (cold->hot="
                  << flow_up_win << ", hot->cold=" << flow_down_win
                  << "). One-sided flow can be a round trip straddling the "
                     "window boundary, but sustained one-sidedness means the "
                     "hot replicas' work is not reaching the cold end.";
                HPOLog::instance().hint(h.str());
              }
              flow_up_win = flow_down_win = 0;
              std::fill(slot_seen_win.begin(), slot_seen_win.end(), char{0});
            }
            /* Corana step-vector state per slot, normalised to the box:
             * r_i = V[i]/width[i], so every slot starts at STEP_FRAC and
             * lives in [V_FLOOR_FRAC, 1]. V is slot-resident (exchanges
             * swap x/f, never V), so each slot's step scale equilibrates
             * to its own temperature. CLASSIC never adapts V -- the
             * blocks would be a constant STEP_FRAC row, so skip them.
             * Emitted through the shared grid emitter -- HPOLog::event
             * stars the header and tags every row. */
            if (kb.corana) {
              std::vector<double> v_gm(K), v_mn(K), v_mx(K);
              for (std::size_t s = 0; s < K; ++s) {
                double slog = 0.0;
                double mn = std::numeric_limits<double>::infinity();
                double mx = 0.0;
                for (std::size_t i = 0; i < dim; ++i) {
                  const double r = chains[s].V[i] / (sp.high[i] - sp.low[i]);
                  slog += std::log(r);
                  mn = std::min(mn, r);
                  mx = std::max(mx, r);
                }
                v_gm[s] = std::exp(slog / static_cast<double>(dim));
                v_mn[s] = mn;
                v_mx[s] = mx;
              }
              auto vcell = [&](const std::vector<double>& v) {
                return [&](std::ostringstream& o, std::size_t s) {
                  o << "  " << std::scientific << std::setprecision(3)
                    << std::setw(9) << v[s];
                };
              };
              grid("ANNEAL_PT V geo-mean per slot "
                   "(geomean_i V[i]/width[i]; start=STEP_FRAC, "
                   "floor=V_FLOOR_FRAC, cap=1):", vcell(v_gm));
              grid("ANNEAL_PT V min per slot (min_i V[i]/width[i]):",
                   vcell(v_mn));
              grid("ANNEAL_PT V max per slot (max_i V[i]/width[i]):",
                   vcell(v_mx));
            }
            if (!kb.corana &&
                kb.step_mode == anneal_enums::STEP_MODE::ADAPTIVE &&
                kb.swap_window > 0 &&
                epoch_count % kb.swap_window == 0) {
              std::vector<double> v_gm(K), v_mn(K), v_mx(K);
              for (std::size_t s = 0; s < K; ++s) {
                long double slog = 0.0L;
                double mn = std::numeric_limits<double>::infinity();
                double mx = 0.0;
                for (std::size_t i = 0; i < dim; ++i) {
                  const double ratio =
                      chains[s].V[i] / (sp.high[i] - sp.low[i]);
                  slog += std::log(static_cast<long double>(ratio));
                  mn = std::min(mn, ratio);
                  mx = std::max(mx, ratio);
                }
                v_gm[s] = static_cast<double>(
                    std::exp(slog / static_cast<long double>(dim)));
                v_mn[s] = mn;
                v_mx[s] = mx;
              }
              auto vcell = [&](const std::vector<double>& v) {
                return [&](std::ostringstream& o, std::size_t s) {
                  o << "  " << std::scientific << std::setprecision(3)
                    << std::setw(9) << v[s];
                };
              };
              /* The printed values are the live chains[s].V at print time,
               * not an aggregate over the window -- say so. */
              const std::string window =
                  " (instantaneous V; sigma[i]/OPTIM_width[i])";
              grid("ANNEAL_PT adaptive sigma/width geo-mean per slot" + window +
                       ":",
                   vcell(v_gm));
              grid("ANNEAL_PT adaptive sigma/width min per slot" + window +
                       ":",
                   vcell(v_mn));
              grid("ANNEAL_PT adaptive sigma/width max per slot" + window +
                       ":",
                   vcell(v_mx));
              if (kb.step_trust_factor > 1.0 &&
                  !chains.empty() && chains[0].V_baseline.size() == dim) {
                /* D27: dims pinned against the trust region, per slot. */
                std::vector<std::size_t> n_lo(K, 0), n_hi(K, 0);
                for (std::size_t s = 0; s < K; ++s) {
                  for (std::size_t i = 0; i < dim; ++i) {
                    const double base = chains[s].V_baseline[i];
                    if (chains[s].V[i] <=
                        base / kb.step_trust_factor * (1.0 + 1e-12))
                      ++n_lo[s];
                    else if (chains[s].V[i] >=
                             base * kb.step_trust_factor * (1.0 - 1e-12))
                      ++n_hi[s];
                  }
                }
                grid("ANNEAL_PT sigma dims at trust bound per slot "
                     "(lo|hi of " + std::to_string(dim) + "):",
                     [&](std::ostringstream& o, std::size_t s) {
                       o << "  " << std::setw(4) << n_lo[s] << '|'
                         << std::left << std::setw(4) << n_hi[s]
                         << std::right;
                     });
              }

              std::ostringstream unhealthy;
              std::size_t unhealthy_count = 0;
              for (std::size_t s = 0; s < K; ++s) {
                if (step_attempt_win[s] == 0) {
                  step_health_streak[s] = 0;
                  continue;
                }
                const double acceptance =
                    static_cast<double>(step_accept_win[s]) /
                    static_cast<double>(step_attempt_win[s]);
                if (acceptance < pt_detail::kStepHealthAcceptLo ||
                    acceptance > pt_detail::kStepHealthAcceptHi)
                  ++step_health_streak[s];
                else
                  step_health_streak[s] = 0;
                if (step_health_streak[s] >= pt_detail::kStepHealthStreak) {
                  if (unhealthy_count++ > 0) unhealthy << ", ";
                  unhealthy << "slot " << logidx::uis(s) << '=' << std::fixed
                            << std::setprecision(1) << 100.0 * acceptance
                            << '%';
                }
              }
              if (unhealthy_count > 0) {
                HPOLog::instance().warning(
                    "ANNEAL_PT adaptive step health: production acceptance "
                    "outside the diagnostics band for at least " +
                    std::to_string(pt_detail::kStepHealthStreak) +
                    " consecutive SWAP_WINDOW intervals: " + unhealthy.str());
              }
              std::fill(step_attempt_win.begin(), step_attempt_win.end(),
                        std::size_t{0});
              std::fill(step_accept_win.begin(), step_accept_win.end(),
                        std::size_t{0});
            }
          }
        }
        if (terminal_epoch) break;
        if (plateau_hit) { stop_reason = StopReason::Plateau; break; }
      }
      } catch (const core::ShutdownRequested&) {
        if (world > 1) throw;
        shutdown_seen = true;
        stop_reason = StopReason::External;
      }
    }

    /* --- multi-rank finalisation ------------------------------------------
     * Broadcast the winning coordinates for the driver's global result.
     * Do not re-evaluate on rank 0: the host publishes best_pot.tadah at
     * discovery time, and a synthetic final evaluation would violate the
     * hard global MAXEVAL contract. */
    anneal::StepSaturation final_vsat;
    if (kb.corana) {
      for (const auto& c : chains) {
        const auto s = anneal::step_saturation(c.V, sp, kb);
        final_vsat.floor += s.floor;
        final_vsat.cap   += s.cap;
        final_vsat.total += s.total;
      }
      if (world > 1) {
        final_vsat.floor = hpo_allreduce_sum_sz(final_vsat.floor);
        final_vsat.cap   = hpo_allreduce_sum_sz(final_vsat.cap);
        final_vsat.total = hpo_allreduce_sum_sz(final_vsat.total);
      }
    }
    std::uint64_t spent_report = budget.spent();
    if (world > 1 && dim > 0) {
      double gb = std::isfinite(l_best) ? l_best
                                        : std::numeric_limits<double>::max();
      int owner = 0;
      hpo_allreduce_minloc(gb, owner);
      if (gb < std::numeric_limits<double>::max()) {
        g_best = gb;
        std::vector<double> bx(dim, 0.0);
        if (rank == owner)
          for (std::size_t d = 0; d < dim; ++d) bx[d] = l_best_x(d);
        hpo_bcast(bx, owner);
        for (std::size_t d = 0; d < dim; ++d) g_best_x(d) = bx[d];
      }
    }
    if (world > 1) spent_report = hpo_allreduce_sum(budget.spent());

    std::vector<std::size_t> swap_attempt_report = swap_attempt;
    std::vector<std::size_t> swap_accept_report  = swap_accept;
    std::vector<std::size_t> swap_degenerate_report = swap_degenerate;
    if (world > 1) {
      for (std::size_t i = 0; i + 1 < K; ++i) {
        swap_attempt_report[i] = hpo_allreduce_sum_sz(swap_attempt[i]);
        swap_accept_report[i]  = hpo_allreduce_sum_sz(swap_accept[i]);
        swap_degenerate_report[i] =
            hpo_allreduce_sum_sz(swap_degenerate[i]);
      }
    }

    if (rank == 0 && ran_sa && kb.ladder_adapt) {
      std::size_t pending_min = 0, pending_max = 0;
      if (!ladder_attempt_global.empty()) {
        const auto pending = std::minmax_element(
            ladder_attempt_global.begin(), ladder_attempt_global.end());
        pending_min = *pending.first;
        pending_max = *pending.second;
      }
      std::ostringstream fhead;
      fhead << "ANNEAL_PT final ladder: updates=" << ladder_update_count
            << "  pending_attempts/pair=" << pending_min;
      if (pending_max != pending_min) fhead << '-' << pending_max;
      fhead << "  mode=" << (ladder_adaptation_active ? "ongoing" : "frozen")
            << "; T per slot:";
      std::ostringstream final_ladder;
      final_ladder << logidx::grid(
          fhead.str(), K, [&](std::ostringstream& o, std::size_t k) {
            o << std::scientific << std::setprecision(3) << "  " << T[k];
          });
      HPOLog::instance().info(final_ladder.str());
    }

    /* --- final summary --------------------------------------------------- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) /
            static_cast<double>(total_proposed) : 0.0;
      const double oob_pct = total_proposed > 0
          ? 100.0 * static_cast<double>(total_oob) /
                static_cast<double>(total_proposed)
          : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL_PT summary: epochs=" << epoch_count
          << "  replicas=" << K
          << "  proposed=" << total_proposed
          << "  accepted=" << std::fixed << std::setprecision(1)
          << (overall_rate * 100.0) << "%"
          << std::scientific << std::setprecision(3)
          << "  oob=" << total_oob << " (" << std::fixed
          << std::setprecision(1) << oob_pct << "%)"
          << std::scientific << std::setprecision(3)
          << "  T_hot=" << T_hot
          << "  T_cold=" << T_cold
          << "  best=" << g_best;
      if (kb.corana)
        oss << "  V_floor=" << final_vsat.floor << "/" << final_vsat.total
            << "  V_cap=" << final_vsat.cap << "/" << final_vsat.total;
      HPOLog::instance().info(oss.str());
      /* Per-pair swap-acceptance table -- THE diagnostic for tuning
       * LADDER_SPAN/REPLICAS (aim 20-40% per pair). */
      if (exchange_rounds > 0) {
        std::ostringstream ossp;
        ossp << "ANNEAL_PT swap acceptance per pair (cold->hot):";
        for (std::size_t i = 0; i + 1 < K; ++i) {
          const double r = swap_attempt_report[i] > 0
              ? 100.0 * static_cast<double>(swap_accept_report[i]) /
                        static_cast<double>(swap_attempt_report[i]) : 0.0;
          ossp << "  " << logidx::uis(i) << ":" << std::fixed << std::setprecision(0)
               << r << "%";
        }
        HPOLog::instance().info(ossp.str());
        /* D16: cumulative degenerate share next to the acceptance it
         * inflates, plus the dominance warning when most of the run's
         * exchange traffic carried no tempering information. */
        std::size_t att_tot = 0, deg_tot = 0;
        for (std::size_t i = 0; i + 1 < K; ++i) {
          att_tot += swap_attempt_report[i];
          deg_tot += swap_degenerate_report[i];
        }
        if (rank == 0 && deg_tot > 0) {
          HPOLog::instance().info(pt_detail::format_pair_table(
              K, "DEGENERATE% (equal-loss; cumulative)",
              swap_attempt_report, swap_degenerate_report));
          if (att_tot > 0 &&
              static_cast<double>(deg_tot) / static_cast<double>(att_tot) >
                  pt_detail::kDegenerateWarnFrac) {
            std::ostringstream w;
            w << "ANNEAL_PT: " << std::fixed << std::setprecision(0)
              << 100.0 * static_cast<double>(deg_tot) /
                     static_cast<double>(att_tot)
              << "% of ALL exchange attempts were DEGENERATE (equal "
                 "losses on both sides -- typical when replicas sit on a "
                 "FAILSCORE plateau). Their automatic acceptance carries "
                 "no tempering information, so the swap tables overstate "
                 "ladder health; they were excluded from ladder "
                 "adaptation.";
            HPOLog::instance().warning(w.str());
          }
        }
      }
      /* D29: the cumulative table alone averages late drift away (observed:
       * bottleneck pairs at 1-3% in the final windows while the
       * cumulative row still read 11-22%).  Repeat the last completed
       * SWAP_WINDOW table next to it; fall back to the labelled partial
       * window when no window ever completed.  Window counters are
       * rank-local, hence single-process only. */
      if (world == 1 && K >= 2 && exchange_rounds > 0 &&
          kb.swap_window > 0) {
        if (swap_last_end_epoch > 0) {
          HPOLog::instance().info(pt_detail::format_pair_table(
              K,
              "last completed SWAP_WINDOW=" +
                  std::to_string(kb.swap_window) + " window, ended epoch " +
                  std::to_string(swap_last_end_epoch) + " of " +
                  std::to_string(epoch_count),
              swap_attempt_last, swap_accept_last));
        } else {
          /* No window ever completed (short run, or window boundaries
           * never coincided with exchange epochs).  The counters then
           * cover every exchange since epoch 1 -- say so explicitly
           * rather than implying "less than one window". */
          std::size_t att_tot = 0;
          for (auto a : swap_attempt_win) att_tot += a;
          if (att_tot > 0)
            HPOLog::instance().info(pt_detail::format_pair_table(
                K,
                "no completed SWAP_WINDOW; covers epochs 1-" +
                    std::to_string(epoch_count),
                swap_attempt_win, swap_accept_win));
        }
      }
      /* Per-slot new-best histogram over the whole run, on the same
       * [lo-hi] grid (and column widths) as the startup ladder table so
       * counts map onto temperatures by eye -- the direct answer to
       * "did the hot rungs ever contribute?". Single-process only: the
       * counts come from this rank's journal.  D29: the counts are
       * cumulative since process start (init/warm-up/transient included),
       * so the last-new-best row distinguishes "still contributing" from
       * "fossil of the initial transient" -- a long run's hot-slot counts all
       * dated from the first ~500 of 20,457 bests. */
      if (world == 1) {
        const auto tot = HPOLog::instance().best_slot_totals();
        if (!tot.empty()) {
          HPOLog::instance().info(pt_detail::format_slot_count_table(
              "ANNEAL_PT new-best count per slot (" + logidx::uis(0) +
              "=coldest .. " +
                  logidx::uis(K - 1) + "=hottest):",
              K, tot));
          HPOLog::instance().info(pt_detail::format_slot_count_table(
              "ANNEAL_PT last new best per slot (eval #; 0 = never):",
              K, HPOLog::instance().best_slot_last_evals()));
        }
      }
      /* Measured ladder flow (single-process runs; see the bookkeeping
       * comment above). Exact counts of end-to-end traversals; 0 round
       * trips = the ladder never communicated end to end. */
      if (world == 1 && K >= 2 && exchange_rounds > 0) {
        const std::size_t trips = std::min(flow_up, flow_down);
        std::ostringstream ossf;
        ossf << "ANNEAL_PT ladder flow: traversals cold->hot=" << flow_up
             << "  hot->cold=" << flow_down;
        if (trips > 0)
          ossf << "  (~" << trips << " round trip(s) completed)";
        else
          ossf << "  (0 round trips -- the ladder never communicated "
                  "end to end)";
        HPOLog::instance().info(ossf.str());
        /* Whole-run mixing profile. Round trips are an end-to-end scalar;
         * this says which slots participated. A slot visited by 1 walker
         * never exchanged its state with anyone -- reading across the row
         * shows exactly where the ladder separated into non-communicating
         * populations. Cumulative, so it is a property of the whole run. */
        std::ostringstream whead;
        whead << "ANNEAL_PT distinct walkers per slot over the whole run ("
              << K << " = fully mixed):";
        std::size_t worst = K, worst_slot = 0;
        std::ostringstream osw;
        osw << logidx::grid(
            whead.str(), K, [&](std::ostringstream& o, std::size_t s) {
              std::size_t n = 0;
              for (std::size_t w = 0; w < K; ++w)
                n += slot_seen[s * K + w] ? 1 : 0;
              if (n < worst) { worst = n; worst_slot = s; }
              o << "  " << std::setw(9) << n;
            });
        HPOLog::instance().info(osw.str());
        /* Only meaningful once the run was long enough for mixing to be
         * possible at all. A DEO round moves a state at most one slot, so
         * below 2*(K-1) rounds a lone walker per slot is arithmetic, not a
         * defect -- and the planning gate has already warned about that
         * case. Warning here too would fire on every short run and teach
         * the reader to ignore the message. */
        if (worst <= 1 && exchange_rounds >= 2 * (K - 1)) {
          std::ostringstream w;
          w << "ANNEAL_PT: slot " << logidx::uis(worst_slot)
            << " was occupied by a single "
               "walker across all " << exchange_rounds << " exchange round(s) "
               "-- its state never exchanged with any other replica, so that "
               "slot contributed nothing the ensemble could use. The ladder "
               "is severed at or adjacent to it.";
          HPOLog::instance().warning(w.str());
        }
      }
    }

    /* --- whole-run Corana per-dimension acceptance census ---------------
     * COLLECTIVE first, rank-0 reporting second -- and deliberately OUTSIDE
     * `if (ran_sa)` above: that flag is not rank-uniform (see its
     * declaration), so a gatherv inside it hangs every run that stopped
     * during seeding. `epoch_count` is incremented in lockstep by the
     * collective epoch loop, so it is the right gate, and it says the
     * honest thing as well: with no epoch there were no Corana moves and
     * the census would be a table of zeros. */
    if (census_on && epoch_count > 0) {
      std::vector<double> census_final;
      census_gather(census_final);
      if (rank == 0 && census_final.size() == census_total) {
        std::vector<corana_census::SlotCensus> cs(K);
        std::vector<std::vector<corana_census::DimAcceptance>> cells(K);
        for (std::size_t k = 0; k < K; ++k) {
          cells[k] = census_slot(census_final, nullptr, k);
          cs[k] = corana_census::summarize(cells[k]);
        }
        std::ostringstream tbl;
        tbl << "ANNEAL_PT Corana per-dimension acceptance census "
               "(cumulative over the whole run).\n"
               "  Corana adapts every coordinate's step from THAT "
               "coordinate's own accept/propose\n"
               "  counts, so these are the per-dimension rates the kernel "
               "actually steered on --\n"
               "  not a rung-wide scalar. Rows are per SLOT (temperature), "
               "NOT per walker: an\n"
               "  exchange moves x/f between slots and leaves V and these "
               "counters behind, so a\n"
               "  row aggregates every walker that visited that "
               "temperature. Percentiles are\n"
               "  nearest-rank over the VISITED dimensions; 'frozen' = "
               "visited but never accepted.\n"
               "  Full per-(slot, dimension) table: "
               "hpo_corana_acceptance.tsv\n"
            << corana_census::slot_table_header();
        for (std::size_t k = 0; k < K; ++k)
          tbl << '\n' << corana_census::format_slot_row(k, T[k], cs[k]);
        HPOLog::instance().info(tbl.str());

        /* Named extremes for the two ends of the ladder only. The cold
         * slot is where a frozen coordinate is expected and the hot slot
         * is where it is not, so those two rows carry the diagnosis; the
         * K-2 interiors would add K-2 lines of the same shape and are in
         * the TSV. */
        auto extremes_line = [&](std::size_t k, const char* which) {
          if (cs[k].lowest.empty()) return;
          std::ostringstream o;
          o << "ANNEAL_PT Corana acceptance extremes, " << which
            << " slot " << logidx::uis(k) << " (T=" << std::scientific
            << std::setprecision(3) << T[k] << "): ";
          /* Sizes come from the lists, never from kExtremes: the two are
           * disjoint and either can be short, so a hard-coded "5" is
           * simply wrong on a small search space. When `highest` is empty
           * the lists together already name every visited dimension, and
           * one ordered list says more than a pair of mirrored ones. */
          if (cs[k].highest.empty()) {
            o << "all " << cs[k].lowest.size()
              << " visited dimension(s), lowest first: "
              << corana_census::format_dim_list(cs[k].lowest, cells[k],
                                                census_label);
          } else {
            o << "lowest " << cs[k].lowest.size() << " = "
              << corana_census::format_dim_list(cs[k].lowest, cells[k],
                                                census_label)
              << " | highest " << cs[k].highest.size() << " = "
              << corana_census::format_dim_list(cs[k].highest, cells[k],
                                                census_label);
          }
          HPOLog::instance().info(o.str());
        };
        extremes_line(0, "coldest");
        if (K > 1) extremes_line(K - 1, "hottest");

        /* A coordinate frozen at ONE slot is ordinary: the cold end is
         * meant to reject nearly everything. A coordinate frozen at
         * EVERY slot, hot end included, never moved anywhere in the run
         * -- either the objective does not respond to it over its OPTIM
         * range, or its V collapsed onto the floor. That is the
         * actionable number, so it is separated from the per-slot
         * counts, which cannot distinguish the two cases. */
        std::vector<std::size_t> dead;
        for (std::size_t i = 0; i < dim; ++i) {
          bool visited_somewhere = false, accepted_somewhere = false;
          for (std::size_t k = 0; k < K; ++k) {
            if (cells[k][i].proposed > 0) visited_somewhere = true;
            if (cells[k][i].accepted > 0) { accepted_somewhere = true; break; }
          }
          if (visited_somewhere && !accepted_somewhere) dead.push_back(i);
        }
        if (!dead.empty()) {
          std::ostringstream o;
          o << "Corana accepted no move on " << dead.size() << " of " << dim
            << " dimension(s) at ANY of the " << K
            << " temperature(s) over the whole run";
          const std::size_t show = std::min<std::size_t>(10, dead.size());
          o << " -- ";
          for (std::size_t j = 0; j < show; ++j) {
            if (j) o << ", ";
            o << census_label(dead[j]);
          }
          if (show < dead.size())
            o << ", ... (" << (dead.size() - show) << " more in "
                 "hpo_corana_acceptance.tsv)";
          o << ". Either the objective does not respond to them over their "
               "OPTIM range, or their step collapsed onto the V floor; "
               "compare their sigma column against V_FLOOR_FRAC x width.";
          HPOLog::instance().hint(o.str());
        }

        std::ofstream acc_file("hpo_corana_acceptance.tsv", std::ios::trunc);
        if (!acc_file) {
          HPOLog::instance().warning(
              "ANNEAL_PT: cannot open hpo_corana_acceptance.tsv for writing");
        } else {
          /* `window_*` cover the last COMPLETED SWAP_WINDOW window; they
           * are 0 when no window completed (the journal line below says
           * which). Cumulative and window sit side by side because a
           * whole-run average over a cooling slot conflates the two
           * regimes the reader is trying to tell apart. */
          acc_file << "slot\ttemperature\tparameter\tacceptance\taccepted"
                      "\tproposed\tsigma\tsigma/width\twindow_acceptance"
                      "\twindow_accepted\twindow_proposed\n";
          acc_file << std::scientific << std::setprecision(17);
          const bool have_window = census_window.size() == census_final.size();
          for (std::size_t k = 0; k < K; ++k) {
            const auto win = have_window
                ? census_slot(census_window, nullptr, k)
                : std::vector<corana_census::DimAcceptance>(dim);
            for (std::size_t i = 0; i < dim; ++i) {
              const double width = sp.high[i] - sp.low[i];
              acc_file << logidx::uis(k) << '\t' << T[k] << '\t'
                       << census_label(i) << '\t'
                       << cells[k][i].rate() << '\t'
                       << cells[k][i].accepted << '\t'
                       << cells[k][i].proposed << '\t'
                       << cells[k][i].sigma << '\t'
                       << (width > 0.0 ? cells[k][i].sigma / width : 0.0)
                       << '\t' << win[i].rate() << '\t'
                       << win[i].accepted << '\t' << win[i].proposed << '\n';
            }
          }
          std::ostringstream w;
          w << "ANNEAL_PT wrote hpo_corana_acceptance.tsv: " << (K * dim)
            << " row(s) (" << K << " slot(s) x " << dim << " dimension(s)); "
            << "window_* columns cover ";
          if (have_window && census_window_end_epoch > 0)
            w << "the SWAP_WINDOW=" << kb.swap_window
              << " window ending at epoch " << census_window_end_epoch
              << " of " << epoch_count;
          else
            w << "nothing -- no SWAP_WINDOW window completed in "
              << epoch_count << " epoch(s)";
          HPOLog::instance().info(w.str());
        }
      }
    }

    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::External:       reason = "external stop requested (SIGINT/SIGTERM/SIGUSR2)"; break;
        case StopReason::ExternalOnProbe: reason = "external stop requested during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::Plateau:        reason = "FTOL best-stall (global best improved by <= tol over NEPS diagnostics windows)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL_PT stopped: " << reason
          << " after " << spent_report << " host eval(s)"
          << " (" << epoch_count << " epoch(s), "
          << exchange_rounds << " exchange round(s))";
      HPOLog::instance().info(oss.str());
    }

    /* Report first, then unwind: hpo_run() writes the SHUTDOWN trailer and
     * the process exit status stays exactly what it was before. */
    if (shutdown_seen) throw core::ShutdownRequested{};
  }

template<typename Vec, typename Runner>
  void run_anneal_pt(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pt_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo

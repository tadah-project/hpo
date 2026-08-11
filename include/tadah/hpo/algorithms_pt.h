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
 * T_0 < T_1 < ... < T_{K-1} (slot 0 coldest). It is static by default;
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
  double gain = 0.0;
};

/* One diminishing stochastic-approximation update.  High-acceptance gaps
 * expand and low-acceptance gaps contract, after which all positive gaps are
 * renormalized to the fixed endpoint span. */
inline LadderAdaptation adapt_temperature_ladder(
    const std::vector<double>& temperature,
    const std::vector<std::size_t>& attempts,
    const std::vector<std::size_t>& accepted,
    std::size_t update, double initial_gain, double decay)
{
  const std::size_t K = temperature.size();
  if (K < 3 || attempts.size() + 1 != K || accepted.size() != attempts.size())
    throw std::invalid_argument(
        "ANNEAL_PT ladder adaptation requires K>=3 and one counter per pair");
  if (update == 0 || !std::isfinite(initial_gain) ||
      !(initial_gain > 0.0 && initial_gain <= 1.0) ||
      !std::isfinite(decay) || !(decay > 0.5 && decay <= 1.0))
    throw std::invalid_argument("ANNEAL_PT ladder adaptation controls are invalid");

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
  std::vector<long double> position(K, 0.0L);
  long double cumulative = 0.0L;
  for (std::size_t p = 0; p + 2 < K; ++p) {
    cumulative += weight[p];
    position[p + 1] = cumulative / weight_sum;
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

enum StepStatus : unsigned {
  StepOk            = 0,
  StepNumericalFloor = 1u << 0,
  StepSupportCapped  = 1u << 1,
  StepUnconverged    = 1u << 2,
  StepInterrupted    = 1u << 3,
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
   * sqrt(dim)/pilot_multiplier -- about 26x for run22. Reporting them on
   * one row against the production sigma made rows self-contradictory:
   * every "support-capped" row means the coordinate scale reached 1.0 x
   * width, yet printed sigma/width = 0.0379. */
  double coordinate_scale = 0.0;
};

struct SlotCalibration {
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
   * run22 slots spent 200 host evaluations each to arrive back at the
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
  constexpr double target = 0.44;
  constexpr double target_lo = 0.29;
  constexpr double target_hi = 0.59;
  const std::size_t dim = sp.low.size();
  const std::size_t minimum = kb.step_auto_sweeps;
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
        const bool gross_miss = !usable || mean_alpha < 0.01 ||
                                mean_alpha > 0.9;
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
      dc.status = StepUnconverged;
      if (dc.have_best) c.V[i] = dc.best_scale;
      dc.coordinate_scale = c.V[i];   /* D9: best tested, or the last one */
    } else if (!dc.finished && change_scale) {
      double next_scale = scale;
      if (dc.have_lower_scale && dc.have_upper_scale &&
          dc.lower_scale < dc.upper_scale) {
        next_scale = std::sqrt(dc.lower_scale * dc.upper_scale);
      } else if (!usable || mean_alpha < target_lo) {
        next_scale /= (!usable || mean_alpha < 0.01) ? 10.0 : 2.0;
      } else {
        next_scale *= mean_alpha > 0.9 ? 10.0 : 2.0;
      }
      unsigned clamp_status = StepOk;
      const double clamped_scale = clamp_coordinate_scale(
          next_scale, c.x(i), sp.low[i], sp.high[i], clamp_status);
      if (clamped_scale < scale) ++dc.backoffs;
      /* D8: keep the reason the scale was clamped. It used to be computed
       * here and dropped, so an interim floor/cap saturation left no trace:
       * a dim that spent the whole search pressed against the resolvable
       * floor still finished labelled a bare "unconverged", and
       * unexplained_dimensions counted it as inexplicable. run22 reported
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
    std::shuffle(active.begin(), active.end(), c.rng);
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
               static_cast<long double>(c.V[i]) * ndist(c.rng);
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
          c.f, f_proposal, T, sp.failscore, [&] { return udist(c.rng); });
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

  for (auto& dc : out.dims) {
    if (out.stop != anneal::RungStop::Completed &&
        !dc.finished && dc.attempts < maximum)
      dc.status |= StepInterrupted;
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
  constexpr double target = 0.234;
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
                                        out.pilot_multiplier) * ndist(c.rng);
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
        c.f, f_proposal, T, sp.failscore, [&] { return udist(c.rng); });
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

    anneal::CoranaPlateau plateau(sp.stop, kb.neps, sp.failscore);

    enum class StopReason {
      Done, MaxEval, MaxTime, StopVal, TMin, Plateau,
      StopValOnProbe, ExternalOnProbe, MaxTimeOnProbe, NoDims, NoBudget
    };
    StopReason stop_reason = StopReason::Done;

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
    bool        ran_sa = false;

    {
      std::ostringstream oss;
      const char* mode = kb.corana
          ? "CORANA"
          : (kb.step_auto
                 ? (kb.step_adapt_every > 0
                        ? "CLASSIC[ADAPTIVE-DIAG]"
                        : "CLASSIC[AUTO-FROZEN]")
                 : "CLASSIC[FIXED]");
      oss << "ANNEAL_PT[" << mode << "]: seed=" << seed
          << "  REPLICAS=" << K
          << "  THREADS=" << threads
          << "  SWAP_EVERY=" << kb.swap_every
          << "  RT=" << kb.rt;
      if (kb.t_cold)      oss << "  T_COLD=" << *kb.t_cold;
      else                oss << "  LADDER_SPAN=" << kb.ladder_span;
      if (ladder_controls_explicit)
        oss << "  LADDER_POWER=" << kb.ladder_power
            << "  LADDER_ADAPT=" << (kb.ladder_adapt ? 1 : 0);
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
            << "  LADDER_ADAPT_UPDATES=" << kb.ladder_adapt_updates;
      }
      if (kb.corana)
        oss << "  NS=" << kb.NS << "  NT=" << kb.NT << "  C=" << kb.C
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
      else
        oss << "  L=" << kb.L;
      if (kb.step_auto)
        oss << "  STEP_AUTO=1"
            << "  STEP_AUTO_SWEEPS=" << kb.step_auto_sweeps
            << "  STEP_ADAPT_EVERY=" << kb.step_adapt_every;
      else if (!kb.corana)
        oss << "  STEP_AUTO=0";
      if (sp.step.has_value()) {
        if (kb.corana)
          oss << "  STEP=" << *sp.step
              << " (Corana U(-V,V) half-width; floor/cap applied per dim)";
        else
          oss << "  sigma=" << *sp.step
              << " (CLASSIC Gaussian; explicit STEP)";
      } else {
        oss << "  STEP_FRAC=" << kb.step_frac
            << (kb.corana ? " (Corana V half-width/width)"
                          : " (CLASSIC Gaussian sigma/width)");
      }
      oss << "  BOUNDARY_MODE=" << static_cast<int>(kb.boundary_mode)
          << (kb.boundary_mode == anneal::BoundaryMode::Reflect
                  ? "(REFLECT)" : "(REDRAW)")
          << "  INIT_MODE=" << kb.init_mode
          << (kb.init_mode == 0
                  ? (kb.step_auto
                         ? "(shared warm seed for automatic calibration)"
                         : (kb.corana ? "(Gaussian jitter; sigma=initial V)"
                                      : "(Gaussian jitter; sigma=proposal sigma)"))
                  : "(uniform box)");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* D20: a moving ladder with a frozen step.
     *
     * STEP_ADAPT_EVERY defaults to 0, which freezes each slot's calibrated
     * step. That is the right default for a STATIC ladder -- the step was
     * calibrated at the slot's temperature and the temperature does not
     * move. But with LADDER_ADAPT=1 or RT<1 the temperatures do move, and
     * every slot then keeps a step tuned for a temperature it no longer
     * has. run22 ran 861 ladder updates that shifted the interior log-gaps
     * from [1.87e-1, 2.07] to [2.02e-1, 3.49] against steps calibrated once
     * at the start, and nothing said so.
     *
     * Combination is legal (a user may want a frozen kernel deliberately)
     * and the consequence is a modelling judgement rather than a proof, so
     * this is a HINT. The CLI help already recommends STEP_ADAPT_EVERY=L;
     * the point here is that the recommendation reaches the person who
     * actually tripped the condition. Not applicable under CORANA, whose
     * own V adaptation is always live. */
    if (!kb.corana && kb.step_auto && kb.step_adapt_every == 0 &&
        (kb.ladder_adapt || kb.rt < 1.0)) {
      std::ostringstream h;
      h << "ANNEAL_PT: temperatures move ("
        << (kb.ladder_adapt ? "LADDER_ADAPT=1" : "")
        << (kb.ladder_adapt && kb.rt < 1.0 ? ", " : "")
        << (kb.rt < 1.0 ? "RT<1" : "")
        << ") but STEP_ADAPT_EVERY=0 freezes each slot's step at the value "
           "calibrated for its ORIGINAL temperature. Acceptance will drift "
           "away from target as the ladder moves, and no later message "
           "reports that drift. STEP_ADAPT_EVERY=" << kb.L
        << " (= L) keeps the step adapting with the temperature.";
      HPOLog::instance().hint(h.str());
    }

    /* Layout arithmetic -- exact, printed up front: with a large
     * NT*NS*dim product the first per-epoch line can be hours away, and
     * the epoch size is what sets both the swap and journal cadence. */
    const std::uint64_t rung_cost =
        kb.corana ? anneal::saturating_mul(
                        anneal::saturating_mul(kb.NT, kb.NS), dim)
                  : static_cast<std::uint64_t>(kb.L);
    const std::uint64_t epoch_cost =
        anneal::saturating_mul(static_cast<std::uint64_t>(K), rung_cost);
    {
      std::ostringstream oss;
      oss << "ANNEAL_PT layout: rung = ";
      if (kb.corana)
        oss << "NT*NS*dim = " << kb.NT << "*" << kb.NS << "*" << dim << " = ";
      else
        oss << "L = ";
      oss << rung_cost << " evals/replica;  epoch = REPLICAS*rung = "
          << K << "*" << rung_cost << " = " << epoch_cost
          << " evals;  exchange attempt every " << kb.swap_every
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
    /* Pinned-dim audit (single-process like the per-epoch report --
     * marks are rank-local, a rank-0 table would miss other ranks). */
    if (world == 1)
      HPOLog::instance().arm_pin_audit(
          sp, kb.v_floor_frac, static_cast<int>(K), "epoch");

    bool calib_threw = false;
    std::string calib_what;
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
          } catch (const std::exception& e) {
            calib_threw = true;
            calib_what = e.what();
          } catch (...) {
            calib_threw = true;
            calib_what = "TADAH:ANNEAL_PT: non-standard exception during zero-dimensional evaluation";
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
        } catch (const std::exception& e) {
          calib_threw = true;
          calib_what = e.what();
          proceed = false;
        } catch (...) {
          calib_threw = true;
          calib_what =
              "TADAH:ANNEAL_PT: non-standard exception during seed evaluation";
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
      } catch (const std::exception& e) {
        calib_threw = true;
        calib_what  = e.what();
      } catch (...) {
        calib_threw = true;
        calib_what =
            "TADAH:ANNEAL_PT: non-standard exception during T0 calibration";
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
        if (rank == 0) throw std::runtime_error(calib_what);
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
      throw std::runtime_error(calib_what);
    }
    T_cold = kb.t_cold ? *kb.t_cold : T_hot / kb.ladder_span;
    if (proceed && !(T_cold > 0.0 && T_cold < T_hot))
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved ladder is invalid (T_COLD=" +
          std::to_string(T_cold) + ", T_HOT=" + std::to_string(T_hot) +
          "); set PARAM T_COLD/T_HOT or LADDER_SPAN consistently");

    /* Hybrid cooling floor (only meaningful when RT < 1). */
    const double t_min = kb.t_min ? *kb.t_min
                                  : (T_hot > 0.0
                                         ? anneal::automatic_t_min(T_hot)
                                         : 1e-12);
    if (proceed && t_min > T_hot)
      throw std::runtime_error(
          "TADAH:ANNEAL_PT: resolved T_MIN must be <= resolved T_HOT");

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
        /* D16: RT and LADDER_ADAPT are orthogonal, and printing them as two
         * independent tags read as a contradiction -- "[adaptive interior]
         * [static ladder (RT=1)]" on one line, which is what EVERY adaptive
         * run showed, because RT=1 is the PT default. RT cools the whole
         * ladder per epoch; LADDER_ADAPT redistributes the interior between
         * frozen endpoints. Say which of the two is happening, once. */
        if (kb.rt < 1.0) {
          oss << "  [endpoints cool x" << kb.rt
              << " per epoch, floor T_MIN="
              << std::scientific << std::setprecision(3) << t_min << ";"
              << (kb.ladder_adapt ? " interior adapts]" : " interior fixed]");
        } else if (kb.ladder_adapt) {
          oss << "  [endpoints fixed (RT=1); interior adapts]";
        } else {
          oss << "  [static ladder (RT=1, no interior adaptation)]";
        }
        HPOLog::instance().info(oss.str());
      }
      /* Per-slot table: [lo-hi] range rows of 8 (HPOLog::event tags and
       * indents continuation lines). The per-epoch report and the
       * end-of-run new-best histogram mirror this grid so their values
       * map onto temperatures by eye. */
      {
        const int iw = static_cast<int>(std::to_string(K - 1).size());
        std::ostringstream oss;
        oss << "ANNEAL_PT ladder T per slot (0=coldest .. " << (K - 1)
            << "=hottest):";
        for (std::size_t lo = 0; lo < K; lo += 8) {
          const std::size_t hi = std::min(lo + 8, K) - 1;
          oss << "\n"
              << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
              << "]" << std::scientific << std::setprecision(3);
          for (std::size_t k = lo; k <= hi; ++k) oss << "  " << T[k];
        }
        HPOLog::instance().info(oss.str());
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
                       "\told_log_gap\tnew_log_gap\n";
        ladder_file << std::scientific << std::setprecision(17);
      }
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
      for (std::size_t li = 0; li < n_local; ++li) {
        const std::size_t k = off_local + li;   /* global slot */
        auto& c = chains[li];
        c.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, k);
        c.V   = proposal_scale;
        c.x   = Vec(dim);
        if (k == 0 || kb.init_mode == 0) {
          /* Automatic CLASSIC calibration needs the same valid warm anchor at
           * every temperature. Fixed CLASSIC and Corana retain their legacy
           * Gaussian initialization jitter for higher slots. */
          std::normal_distribution<double> nd(0.0, 1.0);
          for (std::size_t i = 0; i < dim; ++i) {
            long double v = static_cast<long double>(x0(i));
            if (k > 0 && !kb.step_auto) {
              auto draw = [&]() {
                return static_cast<long double>(x0(i)) +
                       static_cast<long double>(proposal_scale[i]) * nd(c.rng);
              };
              v = draw();
              (void)anneal::constrain_proposal(
                  v, sp.low[i], sp.high[i], kb.boundary_mode, draw);
            }
            c.x(i) = static_cast<double>(v);
          }
        } else {
          for (std::size_t i = 0; i < dim; ++i)
            c.x(i) = std::uniform_real_distribution<>(sp.low[i], sp.high[i])(c.rng);
        }
        c.best_x = c.x;
      }
      if (kb.step_auto && kb.init_mode == 0) {
        for (auto& c : chains) {
          c.f = f0;
          c.best_f = f0;
        }
      } else if (rank == 0 && n_local > 0) {
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
        if (kb.step_auto && kb.init_mode == 0)
          return;  /* all slots share the already-evaluated warm seed */
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
    if (proceed && kb.step_auto) {
      const std::uint64_t warmup_spent_begin = world > 1
          ? hpo_allreduce_sum(budget.spent()) : budget.spent();
      const std::uint64_t max_coordinate_attempts = anneal::saturating_mul(
          anneal::saturating_mul(
              static_cast<std::uint64_t>(kb.step_auto_sweeps), 2),
          static_cast<std::uint64_t>(dim));

      auto phase_allowance = [&](std::uint64_t desired_per_slot,
                                 std::size_t extra_offset = 0) {
        const std::uint64_t spent = world > 1
            ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t remaining = max_eval_global > spent
            ? max_eval_global - spent : 0;
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
             after_coordinates >= max_eval_global))
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
               after_pilot >= max_eval_global))
            break;
          stalled_rounds = after_pilot == before_pilot
              ? stalled_rounds + 1 : 0;
          if (stalled_rounds >= K) break;
          extra_offset = (extra_offset + 1) % K;
        }
        if (warmup_code == 0 && max_eval_global != HUGE_UNSIGNED &&
            after_pilot >= max_eval_global)
          warmup_code = 1;
      }

      for (std::size_t li = 0; li < n_local; ++li) {
        if (chains[li].best_f < l_best) {
          l_best = chains[li].best_f;
          l_best_x = chains[li].best_x;
        }
        chains[li].diag_ram_delta.assign(dim, 0.0);
        chains[li].diag_ram_scalar_delta = 0.0;
        chains[li].diag_ram_proposals = 0;
        chains[li].diag_ram_pending = 0;
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
      /* 17 -> 19: + coordinate_scale (D9) and pilot_multiplier_moved (D12). */
      constexpr std::size_t fields = 19;
      std::vector<double> local_records;
      local_records.reserve(n_local * dim * fields);
      for (std::size_t li = 0; li < n_local; ++li) {
        const auto& sc = step_calibration[li];
        for (std::size_t i = 0; i < dim; ++i) {
          const auto& dc = sc.dims[i];
          local_records.push_back(chains[li].V[i]);
          local_records.push_back(dc.coordinate_scale);          /* D9 */
          local_records.push_back(static_cast<double>(dc.validation_accepted));
          local_records.push_back(static_cast<double>(dc.validation_attempts));
          local_records.push_back(static_cast<double>(dc.attempts));
          local_records.push_back(static_cast<double>(dc.backoffs));
          local_records.push_back(static_cast<double>(dc.status));
          local_records.push_back(dc.validation_mean_alpha);
          local_records.push_back(static_cast<double>(dc.unusable));
          local_records.push_back(static_cast<double>(sc.pilot_proposed));
          local_records.push_back(static_cast<double>(sc.pilot_accepted));
          local_records.push_back(static_cast<double>(sc.pilot_backoffs));
          local_records.push_back(sc.pilot_multiplier);
          local_records.push_back(
              static_cast<double>(sc.pilot_validation_accepted));
          local_records.push_back(
              static_cast<double>(sc.pilot_validation_attempts));
          local_records.push_back(sc.pilot_validation_mean_alpha);
          local_records.push_back(static_cast<double>(sc.pilot_unusable));
          local_records.push_back(static_cast<double>(sc.pilot_status));
          local_records.push_back(sc.pilot_multiplier_moved ? 1.0 : 0.0); /* D12 */
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
           * run22. The two used to share a row with only the production
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
            const double sigma = gathered_step_records[p];
            const double coord_sigma = gathered_step_records[p + 1];
            const std::size_t accepted = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 2]));
            const std::size_t validation_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 3]));
            const std::size_t total_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 4]));
            const std::size_t backoffs = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 5]));
            const unsigned status = static_cast<unsigned>(
                std::llround(gathered_step_records[p + 6]));
            const double mean_alpha = gathered_step_records[p + 7];
            const std::size_t unusable = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 8]));
            const double acceptance = validation_attempts > 0
                ? static_cast<double>(accepted) / validation_attempts : 0.0;
            const std::size_t pilot_attempts = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 9]));
            const std::size_t pilot_backoffs = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 11]));
            const double pilot_multiplier = gathered_step_records[p + 12];
            const std::size_t pilot_validation_accepted =
                static_cast<std::size_t>(
                    std::llround(gathered_step_records[p + 13]));
            const std::size_t pilot_validation_attempts =
                static_cast<std::size_t>(
                    std::llround(gathered_step_records[p + 14]));
            const double pilot_mean_alpha = gathered_step_records[p + 15];
            const std::size_t pilot_unusable = static_cast<std::size_t>(
                std::llround(gathered_step_records[p + 16]));
            const unsigned pilot_status = static_cast<unsigned>(
                std::llround(gathered_step_records[p + 17]));
            const bool pilot_moved =
                gathered_step_records[p + 18] != 0.0;
            const double pilot_acceptance = pilot_validation_attempts > 0
                ? static_cast<double>(pilot_validation_accepted) /
                      pilot_validation_attempts
                : 0.0;
            const std::string parameter = sp.parameter_labels.size() == dim
                ? sp.parameter_labels[i]
                : "dim[" + std::to_string(i) + "]";
            if (scale_file) {
              const double width = sp.high[i] - sp.low[i];
              scale_file << k << '\t' << T[k] << '\t' << parameter << '\t'
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
            if ((status & pt_detail::StepUnconverged) &&
                !(status & (pt_detail::StepNumericalFloor |
                            pt_detail::StepSupportCapped)))
              ++unexplained_dimensions;
          }
          const std::size_t pilot_attempts = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + 8])) : 0;
          const std::size_t pilot_backoffs = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + 10])) : 0;
          const double pilot_multiplier = dim > 0
              ? gathered_step_records[slot_base + 11] : 1.0;
          const std::size_t pilot_accepted = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + 12])) : 0;
          const std::size_t pilot_validation_attempts = dim > 0
              ? static_cast<std::size_t>(std::llround(
                    gathered_step_records[slot_base + 13])) : 0;
          const double pilot_mean_alpha = dim > 0
              ? gathered_step_records[slot_base + 14] : 0.0;
          const unsigned pilot_status = dim > 0
              ? static_cast<unsigned>(std::llround(
                    gathered_step_records[slot_base + 16]))
              : pt_detail::StepOk;
          if (pilot_status & pt_detail::StepUnconverged) ++unresolved_pilots;
          std::ostringstream diag;
          diag << std::scientific << std::setprecision(3)
               << "ANNEAL_PT step calibration slot=" << k
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
               << "  pilot_status=" << status_text(pilot_status)
               << "  sigma=[" << sigma_min << ", " << sigma_max << "]";
          HPOLog::instance().info(diag.str());
        }
        {
          std::ostringstream summary;
          summary << "ANNEAL_PT step calibration summary: dimensions="
                  << (K * dim) << "  ok=" << status_ok
                  << "  numerical-floor=" << status_floor
                  << "  support-capped=" << status_capped
                  << "  unconverged=" << status_unconverged
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

    /* Per-pair swap-acceptance bookkeeping. */
    std::vector<std::size_t> swap_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept (K > 1 ? K - 1 : 0, 0);
    /* Independent adaptation windows.  Only the rank owning the colder
     * side of a pair increments its local entry; every exchange round
     * allreduces these counters before deciding whether to update. */
    std::vector<std::size_t> ladder_attempt(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_accept(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_attempt_global(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> ladder_accept_global(K > 1 ? K - 1 : 0, 0);
    std::size_t ladder_update_count = 0;
    bool ladder_adaptation_active = kb.ladder_adapt;
    /* SWAP_WINDOW-epoch window of the same counters (reset after each
     * windowed table) -- shows acceptance DRIFT the cumulative table
     * averages away. */
    std::vector<std::size_t> swap_attempt_win(K > 1 ? K - 1 : 0, 0);
    std::vector<std::size_t> swap_accept_win (K > 1 ? K - 1 : 0, 0);
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
    const std::size_t transport_window = 100 * (K > 1 ? K - 1 : 1);
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
      for (;;) {
        const auto epoch_wall_t0 = std::chrono::steady_clock::now();
        const std::uint64_t spent_global =
            world > 1 ? hpo_allreduce_sum(budget.spent()) : budget.spent();
        const std::uint64_t remaining =
            max_eval_global > spent_global ? max_eval_global - spent_global : 0;
        if (remaining == 0) { stop_reason = StopReason::MaxEval; break; }
        if (planned_epochs == 0 &&
            (max_eval_global != HUGE_UNSIGNED || kb.rt < 1.0)) {
          planned_epochs = anneal::planned_rungs(
              T[K - 1], t_min, kb.rt,
              max_eval_global == HUGE_UNSIGNED
                  ? std::numeric_limits<std::uint64_t>::max()
                  : remaining,
              epoch_cost);
          /* Plan line -- printed once, from the same arithmetic that
           * feeds the epoch=k/total display (an upper bound: STOPVAL /
           * plateau / MAXTIME end the run earlier). */
          if (rank == 0 && planned_epochs > 0) {
            const std::size_t rounds = planned_epochs / kb.swap_every +
                (planned_epochs % kb.swap_every != 0 ? 1 : 0);
            std::ostringstream oss;
            oss << "ANNEAL_PT plan: ~" << planned_epochs << " epoch(s) (";
            if (max_eval_global != HUGE_UNSIGNED)
              oss << "budget left " << remaining << " evals";
            if (max_eval_global != HUGE_UNSIGNED && kb.rt < 1.0) oss << ", ";
            if (kb.rt < 1.0) oss << "hot-end cooling to T_MIN";
            oss << "; upper bound), ~" << rounds << " exchange round(s)";
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
                if (kb.corana)
                  w << "Reduce NS or NT (epoch = REPLICAS*NT*NS*dim evals) ";
                else
                  w << "Reduce L (epoch = REPLICAS*L evals) ";
                w << "or raise MAXEVAL.";
                HPOLog::instance().warning(w.str());
              } else if (rounds < 4 * rt_floor) {
                std::ostringstream h;
                h << "ANNEAL_PT: planned exchange rounds " << rounds
                  << " < 2*(REPLICAS-1)/0.25 = " << 4 * rt_floor
                  << " -- the 25%-acceptance planning heuristic yields fewer "
                     "than ~1 nominal ladder round trip (literature reference "
                     "values: Kone-Kofke 23%, Predescu 39% for optimization). "
                     "Measured ladder-flow counts are authoritative. ";
                if (kb.corana)
                  h << "Consider smaller NS/NT or larger MAXEVAL.";
                else
                  h << "Consider smaller L or larger MAXEVAL.";
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

        pool.run_epoch(n_local, [&](std::size_t li) {
          const std::size_t k = off_local + li;
          warmstart::current_chain() = k;   /* global slot for warm-start key */
          HPOLog::set_eval_slot(static_cast<int>(k), K, T[k]);
          rstats[li] = anneal::run_rung(chains[li], T[k], sp, kb, host, ctx,
                                        allowance[k]);
        });
        rethrow_pool_error("replica rung");

        std::size_t e_prop = 0, e_acc = 0, e_oob = 0;
        anneal::StepSaturation e_vsat;
        bool any_stopval = false, any_time = false;
        for (std::size_t li = 0; li < n_local; ++li) {
          e_prop += rstats[li].proposed;
          e_acc  += rstats[li].accepted;
          e_oob  += rstats[li].oob;
          if (world == 1) {
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
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], ci.f, cj.f,
                  [&]() { return udist(pair_rng); });
              ++swap_attempt[i];
              ++swap_attempt_win[i];
              if (ladder_adaptation_active) ++ladder_attempt[i];
              ++att;
              pair_event[i] = '.';
              if (accepted) {
                std::swap(ci.x, cj.x);
                std::swap(ci.f, cj.f);
                if (world == 1) std::swap(walker_at[i], walker_at[i + 1]);
                ++swap_accept[i];
                ++swap_accept_win[i];
                if (ladder_adaptation_active) ++ladder_accept[i];
                ++acc;
                pair_event[i] = 'x';
              }
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
              const bool accepted = pt_detail::accept_exchange(
                  T[i], T[i + 1], f_i, f_j,
                  [&]() { return udist(pair_rng); });
              if (i_mine) {
                ++swap_attempt[i];
                ++swap_attempt_win[i];
                if (ladder_adaptation_active) ++ladder_attempt[i];
                ++att;
                if (accepted) {
                  ++swap_accept[i];
                  ++swap_accept_win[i];
                  if (ladder_adaptation_active) ++ladder_accept[i];
                  ++acc;
                }
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
                      "ANNEAL_PT ladder adaptation frozen: cooling reached "
                      "a non-positive or non-distinct temperature; endpoint "
                      "cooling continues unchanged");
              } else {
                const std::vector<double> old_temperature = T;
                const auto update = pt_detail::adapt_temperature_ladder(
                    T, ladder_attempt_global, ladder_accept_global,
                    ladder_update_count + 1, kb.ladder_adapt_gain,
                    kb.ladder_adapt_decay);
                T = update.temperature;
                ++ladder_update_count;

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
                  HPOLog::instance().info(summary.str());

                  const int iw = static_cast<int>(
                      std::to_string(K - 1).size());
                  std::ostringstream table;
                  table << "ANNEAL_PT ladder T per slot after update "
                        << ladder_update_count << ":";
                  for (std::size_t lo = 0; lo < K; lo += 8) {
                    const std::size_t hi = std::min(lo + 8, K) - 1;
                    table << "\n[" << std::setw(iw) << lo << "-"
                          << std::setw(iw) << hi << "]"
                          << std::scientific << std::setprecision(3);
                    for (std::size_t k = lo; k <= hi; ++k)
                      table << "  " << T[k];
                  }
                  HPOLog::instance().info(table.str());

                  if (ladder_file) {
                    for (std::size_t p = 0; p + 1 < K; ++p) {
                      ladder_file
                          << ladder_update_count << '\t' << epoch_count << '\t'
                          << p << '\t' << ladder_attempt_global[p] << '\t'
                          << ladder_accept_global[p] << '\t'
                          << update.acceptance[p] << '\t'
                          << old_temperature[p] << '\t'
                          << old_temperature[p + 1] << '\t'
                          << T[p] << '\t' << T[p + 1] << '\t'
                          << update.old_log_gap[p] << '\t'
                          << update.new_log_gap[p] << '\n';
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
                if (kb.ladder_adapt_updates > 0 &&
                    ladder_update_count >= kb.ladder_adapt_updates) {
                  ladder_adaptation_active = false;
                  if (rank == 0)
                    HPOLog::instance().info(
                        "ANNEAL_PT ladder adaptation frozen after configured "
                        "LADDER_ADAPT_UPDATES=" +
                        std::to_string(kb.ladder_adapt_updates));
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

        int rank0_code = 0;   /* 0 none, 1 plateau */
        if (rank == 0 && !terminal_epoch) {
          if (plateau.update(chains[0].f, g_best)) rank0_code = 1;
        }
        if (world > 1) hpo_bcast(rank0_code, 0);
        const bool plateau_hit = rank0_code == 1;

        /* --- optional ladder cooling (PT -> SA hybrid) ---------------- */
        bool tmin_hit = false;
        if (!terminal_epoch && !plateau_hit && kb.rt < 1.0) {
          for (auto& t : T) t = anneal::cool_temperature(t, kb.rt);
          tmin_hit = T[K - 1] < t_min;
        }

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

        /* --- one journal line per epoch (rank 0 carries the global
         * numbers; other ranks stay quiet) ----------------------------- */
        if (rank == 0) {
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
          oss << "  T=[" << T[0] << ", " << T[K - 1] << "]"
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
            const int iwp = static_cast<int>(std::to_string(K - 2).size());
            auto pair_table = [&](const std::string& label,
                                  const std::vector<std::size_t>& attv,
                                  const std::vector<std::size_t>& accv) {
              std::size_t amin = std::numeric_limits<std::size_t>::max();
              std::size_t amax = 0;
              for (std::size_t p = 0; p + 1 < K; ++p) {
                amin = std::min(amin, attv[p]);
                amax = std::max(amax, attv[p]);
              }
              std::ostringstream o;
              o << "ANNEAL_PT swap acc% per pair (" << label
                << ", att/pair " << amin;
              if (amax != amin) o << "-" << amax;
              o << "; pair p = slots p<->p+1):";
              for (std::size_t lo = 0; lo + 1 < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K - 1) - 1;
                o << "\n[" << std::setw(iwp) << lo << "-" << std::setw(iwp)
                  << hi << "]";
                for (std::size_t p = lo; p <= hi; ++p) {
                  if (attv[p] == 0)
                    o << "  " << std::setw(9) << "-";
                  else
                    o << "  " << std::setw(9)
                      << std::lround(100.0 *
                             static_cast<double>(accv[p]) /
                             static_cast<double>(attv[p]));
                }
              }
              HPOLog::instance().info(o.str());
            };
            pair_table("cumulative", swap_attempt, swap_accept);
            if (kb.swap_window > 0 && epoch_count % kb.swap_window == 0) {
              std::size_t att_tot = 0;
              for (auto a : swap_attempt_win) att_tot += a;
              if (att_tot > 0) {
                pair_table("last " + std::to_string(kb.swap_window) +
                           " epoch(s)", swap_attempt_win, swap_accept_win);
                std::fill(swap_attempt_win.begin(), swap_attempt_win.end(),
                          std::size_t{0});
                std::fill(swap_accept_win.begin(), swap_accept_win.end(),
                          std::size_t{0});
              }
            }
            /* THIS epoch's exchange outcomes -- the walker motion itself,
             * readable without diffing walker rows across epochs. */
            {
              std::ostringstream o;
              o << "ANNEAL_PT exchange this epoch "
                   "(x=swapped, .=rejected, blank=other parity):";
              for (std::size_t lo = 0; lo + 1 < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K - 1) - 1;
                o << "\n[" << std::setw(iwp) << lo << "-" << std::setw(iwp)
                  << hi << "]";
                for (std::size_t p = lo; p <= hi; ++p)
                  o << "  " << std::setw(9) << pair_event[p];
              }
              HPOLog::instance().info(o.str());
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
              for (const auto& [s, n] : win) ossw << "  [" << s << "]:" << n;
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
            const int iw = static_cast<int>(std::to_string(K - 1).size());
            auto grid = [&](const std::string& head, auto&& cell) {
              std::ostringstream o;
              o << head;
              for (std::size_t lo = 0; lo < K; lo += 8) {
                const std::size_t hi = std::min(lo + 8, K) - 1;
                o << "\n"
                  << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
                  << "]";
                for (std::size_t s = lo; s <= hi; ++s) cell(o, s);
              }
              HPOLog::instance().info(o.str());
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
                   o << "  " << std::setw(8) << w << t;
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
            if (!kb.corana && kb.step_adapt_every > 0 &&
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
              const std::string window =
                  " (last " + std::to_string(kb.swap_window) +
                  " epochs; sigma[i]/OPTIM_width[i])";
              grid("ANNEAL_PT adaptive sigma/width geo-mean per slot" + window +
                       ":",
                   vcell(v_gm));
              grid("ANNEAL_PT adaptive sigma/width min per slot" + window +
                       ":",
                   vcell(v_mn));
              grid("ANNEAL_PT adaptive sigma/width max per slot" + window +
                       ":",
                   vcell(v_mx));

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
                if (acceptance < 0.05 || acceptance > 0.50)
                  ++step_health_streak[s];
                else
                  step_health_streak[s] = 0;
                if (step_health_streak[s] >= 2) {
                  if (unhealthy_count++ > 0) unhealthy << ", ";
                  unhealthy << "slot " << s << '=' << std::fixed
                            << std::setprecision(1) << 100.0 * acceptance
                            << '%';
                }
              }
              if (unhealthy_count > 0) {
                HPOLog::instance().warning(
                    "ANNEAL_PT adaptive step health: production acceptance "
                    "outside [5%, 50%] for at least two consecutive "
                    "SWAP_WINDOW intervals: " + unhealthy.str());
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
        if (tmin_hit)    { stop_reason = StopReason::TMin; break; }
      }
    }

    /* --- multi-rank finalisation ------------------------------------------
     * Broadcast the winning coordinates for the driver's global result.
     * Do not re-evaluate on rank 0: host-side best checkpoints are written
     * at discovery time, and a synthetic final evaluation would violate the
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
    if (world > 1) {
      for (std::size_t i = 0; i + 1 < K; ++i) {
        swap_attempt_report[i] = hpo_allreduce_sum_sz(swap_attempt[i]);
        swap_accept_report[i]  = hpo_allreduce_sum_sz(swap_accept[i]);
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
      const int iw = static_cast<int>(std::to_string(K - 1).size());
      std::ostringstream final_ladder;
      final_ladder << "ANNEAL_PT final ladder: updates="
                   << ladder_update_count
                   << "  pending_attempts/pair=" << pending_min;
      if (pending_max != pending_min) final_ladder << '-' << pending_max;
      final_ladder << "  mode="
                   << (ladder_adaptation_active ? "ongoing" : "frozen")
                   << "; T per slot:";
      for (std::size_t lo = 0; lo < K; lo += 8) {
        const std::size_t hi = std::min(lo + 8, K) - 1;
        final_ladder << "\n[" << std::setw(iw) << lo << '-'
                     << std::setw(iw) << hi << ']'
                     << std::scientific << std::setprecision(3);
        for (std::size_t k = lo; k <= hi; ++k)
          final_ladder << "  " << T[k];
      }
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
          ossp << "  " << i << ":" << std::fixed << std::setprecision(0)
               << r << "%";
        }
        HPOLog::instance().info(ossp.str());
      }
      /* Per-slot new-best histogram over the whole run, on the same
       * [lo-hi] grid (and column widths) as the startup ladder table so
       * counts map onto temperatures by eye -- the direct answer to
       * "did the hot rungs ever contribute?". Single-process only: the
       * counts come from this rank's journal. */
      if (world == 1) {
        const auto tot = HPOLog::instance().best_slot_totals();
        if (!tot.empty()) {
          std::map<int, std::size_t> cnt(tot.begin(), tot.end());
          const int iw = static_cast<int>(std::to_string(K - 1).size());
          std::ostringstream ossb;
          ossb << "ANNEAL_PT new-best count per slot (0=coldest .. "
               << (K - 1) << "=hottest):";
          for (std::size_t lo = 0; lo < K; lo += 8) {
            const std::size_t hi = std::min(lo + 8, K) - 1;
            ossb << "\n"
                 << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
                 << "]";
            for (std::size_t k = lo; k <= hi; ++k) {
              const auto it = cnt.find(static_cast<int>(k));
              ossb << "  " << std::setw(9)
                   << (it != cnt.end() ? it->second : 0);
            }
          }
          HPOLog::instance().info(ossb.str());
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
        const int iw = static_cast<int>(std::to_string(K - 1).size());
        std::ostringstream osw;
        osw << "ANNEAL_PT distinct walkers per slot over the whole run ("
            << K << " = fully mixed):";
        std::size_t worst = K, worst_slot = 0;
        for (std::size_t lo = 0; lo < K; lo += 8) {
          const std::size_t hi = std::min(lo + std::size_t{8}, K) - 1;
          osw << "\n"
              << "[" << std::setw(iw) << lo << "-" << std::setw(iw) << hi
              << "]";
          for (std::size_t s = lo; s <= hi; ++s) {
            std::size_t n = 0;
            for (std::size_t w = 0; w < K; ++w)
              n += slot_seen[s * K + w] ? 1 : 0;
            if (n < worst) { worst = n; worst_slot = s; }
            osw << "  " << std::setw(9) << n;
          }
        }
        HPOLog::instance().info(osw.str());
        /* Only meaningful once the run was long enough for mixing to be
         * possible at all. A DEO round moves a state at most one slot, so
         * below 2*(K-1) rounds a lone walker per slot is arithmetic, not a
         * defect -- and the planning gate has already warned about that
         * case. Warning here too would fire on every short run and teach
         * the reader to ignore the message. */
        if (worst <= 1 && exchange_rounds >= 2 * (K - 1)) {
          std::ostringstream w;
          w << "ANNEAL_PT: slot " << worst_slot << " was occupied by a single "
               "walker across all " << exchange_rounds << " exchange round(s) "
               "-- its state never exchanged with any other replica, so that "
               "slot contributed nothing the ensemble could use. The ladder "
               "is severed at or adjacent to it.";
          HPOLog::instance().warning(w.str());
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
        case StopReason::ExternalOnProbe: reason = "external stop requested during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T < T_MIN (hot end)"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule, coldest slot)"; break;
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
  }

template<typename Vec, typename Runner>
  void run_anneal_pt(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_pt_impl<Vec>(sp, host);
  }

}  // namespace tadah::hpo

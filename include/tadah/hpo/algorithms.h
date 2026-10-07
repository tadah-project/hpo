#pragma once
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/backend_rules.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tadah::hpo {

/* ----- stops of the sampling searches ----------------------------- *
 *
 * TADAH:RANDOM and TADAH:GRID (and DLIB:GFS, dlib_adapter.h) end on
 * MAXEVAL, on MAXTIME -- no evaluation starts once MAXTIME seconds have
 * passed since the search began -- or at the first evaluation whose loss is
 * at or below STOPVAL (a feasible loss: FAILSCORE never reaches it,
 * anneal::stopval_reached). RANDOM runs until one of them (or the RESTART
 * stagnation stop) ends it (require_stop), GRID at the latest after its
 * whole grid. The journal says which stop ended the search, and a MAXTIME
 * or STOPVAL stop is the run's status (HPOLog::set_status); a search that
 * spends its MAXEVAL or its grid ends COMPLETED, even when MAXTIME has
 * passed by then (no evaluation was left to start), as DLIB:GFS reports
 * it. */
namespace sampling_detail {

enum class Stop { None, StopVal, MaxTime };

class StopWatch {
public:
    explicit StopWatch(const OptimizerSpec& sp)
        : sp_(sp), t0_(std::chrono::steady_clock::now()) {}

    /* The stop `loss`, the loss of the evaluation just made, triggers:
     * STOPVAL first, then MAXTIME -- only when `more`, another evaluation
     * would start (MAXEVAL or the grid not yet spent): MAXTIME keeps
     * evaluations from starting, and after the last one none would. */
    Stop after(double loss, bool more) const
    {
        if (anneal::stopval_reached(loss, sp_.stop.stopval, sp_.failscore))
            return Stop::StopVal;
        if (more && sp_.stop.maxtime &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t0_).count() >= *sp_.stop.maxtime)
            return Stop::MaxTime;
        return Stop::None;
    }

private:
    const OptimizerSpec& sp_;
    std::chrono::steady_clock::time_point t0_;
};

inline const char* stop_reason(Stop s)
{
    return s == Stop::StopVal ? "STOPVAL reached" : "MAXTIME exceeded";
}

/* The journal line of a search that `stop` ended after `detail`
 * ("7 evaluation(s)"), and the run's status. */
inline void report_stop(const std::string& backend, Stop stop,
                        const std::string& detail)
{
    const std::string msg =
        backend + " stopped: " + stop_reason(stop) + " after " + detail;
    std::cout << "[HPO] " << msg << "\n";
    HPOLog::instance().info(msg);
    HPOLog::instance().set_status(backend + ": " + stop_reason(stop));
}

}  // namespace sampling_detail

/* ----- RANDOM ---------------------------------------------------- */
template<typename Vec, typename Runner>
  void run_random(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    /* RANDOM stops only on MAXEVAL, MAXTIME or STOPVAL, or under an armed
     * RESTART on its stagnation stop: without one the run would never end,
     * and it takes no PARAM (the startup check, again for a spec built at
     * run time). */
    validate_random_spec(sp, restart_stop_armed<HPO_Host<Vec,Runner>>());
    using sampling_detail::Stop;
    const sampling_detail::StopWatch watch(sp);
    uint64_t seed = sp.seed.value_or(std::random_device{}());  // default seed
    {
      std::ostringstream oss;
      oss << "RANDOM search: seed=" << seed;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    std::mt19937_64 rng(seed);
    const auto& lo = sp.low, &hi = sp.high;

    const std::optional<uint64_t> budget = sp.stop.maxeval;
    uint64_t done = 0;
    Stop stop = Stop::None;
    const auto more = [&] { return !budget || done < *budget; };

    /* evaluate the user's config point first so the prior is always part
     * of the explored set. */
    if (!sp.optim_init.empty())
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      const double loss = host(sp.out(x_init));
      ++done;
      stop = watch.after(loss, more());
    }

    while (stop == Stop::None && more())
    {
      Vec x(lo.size());
      for(size_t i=0;i<lo.size();++i)
        x(i)=std::uniform_real_distribution<>(lo[i],hi[i])(rng);
      const double loss = host( sp.out(x) );   // evaluate in original scale
      ++done;
      stop = watch.after(loss, more());
    }

    const std::string detail = std::to_string(done) + " evaluation(s)";
    if (stop != Stop::None) {
      sampling_detail::report_stop("TADAH:RANDOM", stop, detail);
    } else {
      const std::string msg =
          "TADAH:RANDOM stopped: MAXEVAL reached after " + detail;
      std::cout << "[HPO] " << msg << "\n";
      HPOLog::instance().info(msg);
    }
  }

/* ----- GRID  (naive Cartesian product) --------------------------- *
 *
 * Visits the INIT point, then every node of a GRID_POINTS^dim grid, the
 * first dimension varying fastest. MAXEVAL stops the sweep after that
 * many points in this order (the journal says so); without MAXEVAL, or
 * with one at least the number of points, the whole grid is evaluated.
 * MAXTIME and STOPVAL end it earlier, still in grid order
 * (sampling_detail). */
namespace grid_detail {
/* steps^dim, or UINT64_MAX when it does not fit. */
inline std::uint64_t node_count(unsigned steps, std::size_t dim)
{
    std::uint64_t n = 1;
    for (std::size_t d = 0; d < dim; ++d) {
        if (n > std::numeric_limits<std::uint64_t>::max() / steps)
            return std::numeric_limits<std::uint64_t>::max();
        n *= steps;
    }
    return n;
}
}  // namespace grid_detail

template<typename Vec, typename Runner>
  void run_grid(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    /* Grid resolution: PARAM GRID_POINTS <n> (integer >= 2, default 20);
     * STEP and any other PARAM are errors (validate_grid_spec). */
    const unsigned steps = validate_grid_spec(sp);
    using sampling_detail::Stop;
    const sampling_detail::StopWatch watch(sp);

    const size_t dim = sp.low.size();
    std::vector<unsigned> idx(dim,0);
    {
      std::ostringstream oss;
      oss << "GRID search: " << steps << " points in each dimension "
          << "(PARAM GRID_POINTS; spacing = width/" << (steps - 1) << "), "
          << dim << " dimensions.";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    const bool with_init = !sp.optim_init.empty();
    const std::uint64_t nodes = grid_detail::node_count(steps, dim);
    const std::uint64_t total =
        (with_init && nodes < kMax) ? nodes + 1 : nodes;
    std::uint64_t budget = total;
    if (sp.stop.maxeval && *sp.stop.maxeval < total) {
      budget = *sp.stop.maxeval;
      std::ostringstream oss;
      oss << "GRID: MAXEVAL " << budget << " < ";
      if (nodes == kMax) oss << "more than " << kMax;
      else               oss << total;
      oss << " grid points (" << (with_init ? "the INIT point, then " : "");
      if (nodes == kMax) oss << "more than " << kMax;
      else               oss << nodes;
      oss << " = " << steps << "^" << dim << " grid nodes); evaluating the "
          << "first " << budget << " in grid order";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    std::uint64_t done = 0;
    Stop stop = Stop::None;
    /* evaluate the user's config point first so the prior is always part
     * of the explored set (in addition to the regular grid sweep). */
    if (with_init && done < budget)
    {
      const auto v = sp.in(sp.optim_init.front());
      Vec x_init(v.size());
      for (size_t i = 0; i < v.size(); ++i) x_init(i) = v[i];
      const double loss = host(sp.out(x_init));
      ++done;
      stop = watch.after(loss, done < budget);
    }

    Vec x(dim);
    while (stop == Stop::None && done < budget)
    {
      for(size_t d=0;d<dim;++d)
      {
        double t= static_cast<double>(idx[d])/(steps-1);
        x(d) = sp.low[d]*(1-t) + sp.high[d]*t;
      }
      const double loss = host( sp.out(x) );
      ++done;
      stop = watch.after(loss, done < budget);

      /* increment mixed-radix counter */
      for(size_t d=0;d<dim;++d)
        if(++idx[d] < steps) break; else idx[d]=0;
    }

    std::ostringstream points;
    points << done << " of ";
    if (nodes == kMax) points << "more than " << kMax;
    else               points << total;
    points << " grid points";
    if (stop != Stop::None) {
      sampling_detail::report_stop("TADAH:GRID", stop, points.str());
    } else {
      const std::string msg =
          done < total ? "TADAH:GRID stopped: MAXEVAL reached after " +
                             points.str()
                       : "TADAH:GRID finished: all " + std::to_string(total) +
                             " grid points evaluated";
      std::cout << "[HPO] " << msg << "\n";
      HPOLog::instance().info(msg);
    }
  }

/* ----- ANNEAL  (classic Kirkpatrick/Metropolis SA) ---------------- *
 *
 * Refs: Kirkpatrick et al., Science 220 (1983); Metropolis et al.,
 *       J. Chem. Phys. 21 (1953); Corana et al., ACM TOMS 13:262
 *       (1987); Brommer, Univ. Stuttgart (2003).
 *
 * Serial driver over the shared chain core in anneal_core.h (the
 * parallel drivers ANNEAL_PT / ANNEAL_PA reuse the same rungs, T0
 * calibration, plateau rule, and budget plumbing). Operates entirely
 * in transformed (log-scaled) parameter space: `sp.low/sp.high` are
 * post-`analyse_logscale`; `host()` is called via `sp.out(...)` so the
 * LAMMPS side sees real-space values.
 *
 * Stops honoured: MAXEVAL (hard cap on host calls, including T0
 * probes), MAXTIME (wall clock), STOPVAL (any host eval `<=` stopval),
 * T < T_MIN (with optional warm re-anneal, see REANNEAL), and
 * FTOL_REL/FTOL_ABS via the Corana sec.-4 eps-NEPS plateau rule (see
 * anneal::CoranaPlateau).
 *
 * T0 calibration (when PARAM T_INIT is not set): seed-anchored local
 * probing with FAILSCORE-robust statistics; see anneal::calibrate_t0.
 *
 * PARAM keys, defaults, ranges, and cross-key rules are defined in
 * blueprint/anneal_keys.toml (generated into anneal_keys_gen.h); this
 * driver adds only the AUTO resolution of CHAIN_LENGTH/NT from the
 * MAXEVAL budget once T0 (and hence the schedule length) is known.
 *
 * Knobs that are dead for this algorithm are hard errors, not silent
 * no-ops: STEP, XTOL_REL/XTOL_ABS, VECTOR_ARR, GRAD_STRATEGY, and
 * POPULATION (use TADAH:ANNEAL_PA), unknown PARAM keys, and retired
 * key names (registry rename hints).
 * THREADS IS accepted here under the shared-pool model (Option A): it
 * sizes the per-rank worker pool that runs the within-eval pc_lammps
 * fan-out -- serial ANNEAL still drives ONE chain, only the fan-out is
 * parallel. See anneal::Knobs::parse.
 *
 * `run_anneal_impl` is parameterised on a generic callable host
 * (`double operator()(const Vec&) const`) so unit tests can drive the
 * SA loop with a stub instead of a full HPO_Host. The public
 * `run_anneal<Vec, Runner>(...)` wrapper below delegates here.
 *
 * `elapsed_seconds` is the clock MAXTIME is measured with: the seconds
 * since the search began. Empty (the default, and what run_anneal()
 * passes) reads std::chrono::steady_clock from the start of this call; a
 * unit test passes a scripted clock to place the MAXTIME deadline between
 * two of the driver's checks.
 */
using AnnealElapsedSeconds = std::function<double()>;

template<typename Vec, typename HostT>
  void run_anneal_impl(const OptimizerSpec& sp, HostT& host,
                       const AnnealElapsedSeconds& elapsed_seconds = {})
  {
    using anneal::Knobs;
    using anneal::AlgoKind;
    using anneal::RungStop;

    const std::size_t dim = sp.low.size();
    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());

    const Knobs kb = Knobs::parse(sp, AlgoKind::Serial);
    anneal::announce_knobs(kb, AlgoKind::Serial);
    const std::vector<double> initial =
        anneal::make_initial_point(sp, AlgoKind::Serial);

    /* Decorrelated streams: probing never shifts the proposal
     * sequence (see anneal_core.h). */
    std::mt19937_64 probe_rng = anneal::make_stream(seed, anneal::SALT_PROBES);

    /* Per-dim transformed-space proposal scale.  Corana initialization
     * goes through the same floor/cap invariant as later adjustments. */
    const std::vector<double> proposal_scale = anneal::make_initial_steps(sp, kb);

    constexpr std::uint64_t HUGE_UNSIGNED = std::numeric_limits<std::uint64_t>::max();
    anneal::SharedBudget budget(sp.stop.maxeval.value_or(HUGE_UNSIGNED));

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    const std::function<bool()> time_exceeded = [&]() {
      if (max_time <= 0.0) return false;
      if (elapsed_seconds) return elapsed_seconds() >= max_time;
      using Sec = std::chrono::duration<double>;
      return std::chrono::duration_cast<Sec>(
               std::chrono::steady_clock::now() - t0_wall).count() >= max_time;
    };

    const auto stopval = sp.stop.stopval;

    std::atomic<bool> stop_flag{false};
    anneal::StopCtx ctx{budget, stop_flag, time_exceeded, stopval};

    anneal::CoranaPlateau plateau(sp.stop, kb.neps, sp.failscore);

    /* Stop reason -- set at every exit path; emitted as a single
     * `ANNEAL stopped: ...` line at the end so the user can see why
     * the loop finished without grepping per-eval timestamps. */
    enum class StopReason {
      Done,            /* loop completed normally (rare) */
      MaxEval,
      MaxTime,
      StopVal,
      TMin,
      Plateau,         /* Corana sec.-4 eps-NEPS plateau */
      StopValOnProbe,  /* STOPVAL hit during T0 calibration probes */
      ExternalOnProbe,
      MaxTimeOnProbe,
      NoDims,
      NoBudget         /* MAXEVAL ran out before any SA proposal */
    };
    StopReason stop_reason = StopReason::Done;

    /* Aggregates referenced by the final summary; declared up front so
     * every early-exit path can fall through to one summary block. */
    double      T0              = 0.0;
    double      T_final         = 0.0;
    std::size_t total_proposed  = 0;
    std::size_t total_accepted  = 0;
    std::size_t total_oob       = 0;
    std::size_t rung_count      = 0;
    std::size_t reanneal_count  = 0;
    double      accept_rate_min = 1.0;
    double      accept_rate_max = 0.0;
    bool        any_rate        = false;
    bool        ran_sa          = false;   /* true once the seed evaluated */

    /* Entry banner. When Corana is enabled it shows NS/NT/CORANA_C in
     * place of CHAIN_LENGTH (which Corana doesn't use). AUTO lengths
     * print as such here; the resolved values follow in the layout line
     * once T0 (and hence the schedule length) is known. */
    {
      std::ostringstream oss;
      oss << "ANNEAL[" << (kb.corana ? "CORANA" : "CLASSIC") << "]: "
          << "seed=" << seed
          << "  RT=" << kb.rt
          << "  BOUNDARY_MODE="
          << (kb.boundary_mode == anneal::BoundaryMode::Reflect
                  ? "REFLECT" : "REDRAW");
      if (kb.t_start) oss << "  T_INIT=" << *kb.t_start;
      if (kb.t_min) oss << "  T_MIN=" << *kb.t_min;
      else          oss << "  T_MIN=auto(T0*1e-6)";
      if (kb.corana) {
        oss << "  NS=" << kb.NS;
        if (kb.NT > 0) oss << "  NT=" << kb.NT;
        else           oss << "  NT=auto";
        oss << "  CORANA_C=" << kb.corana_c
            << "  V_FLOOR_FRAC=" << kb.v_floor_frac;
        if (kb.reset_to_best) oss << "  RESET_TO_BEST=ON";
      } else {
        if (kb.chain_length > 0) oss << "  CHAIN_LENGTH=" << kb.chain_length;
        else                     oss << "  CHAIN_LENGTH=auto";
      }
      oss << "  STEP_FRAC=" << kb.step_frac
          << (kb.corana ? " (Corana V half-width/width)"
                        : " (CLASSIC Gaussian sigma/width)");
      oss << "  REANNEAL=" << (kb.reanneal ? "ON" : "OFF");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    HPOLog::instance().arm_pin_audit(sp, kb.v_floor_frac, 1, "rung");

    if (dim == 0) {
      if (!sp.optim_init.empty()) {
        if (!budget.try_reserve()) {
          stop_reason = StopReason::NoBudget;
        } else {
          Vec xv(dim);
          for (std::size_t i = 0; i < dim; ++i) xv(i) = initial[i];
          double f = host(sp.out(xv));
          if (!std::isfinite(f))
            f = std::numeric_limits<double>::infinity();
          if (anneal::stopval_reached(f, stopval, sp.failscore)) stop_reason = StopReason::StopVal;
          else if (time_exceeded()) stop_reason = StopReason::MaxTime;
          else                  stop_reason = StopReason::NoDims;
        }
      } else {
        stop_reason = StopReason::NoDims;
      }
    }

    bool proceed = (stop_reason == StopReason::Done);

    /* --- chain state + seed/anchor evaluation --------------------------
     * The seed is evaluated BEFORE T0 calibration: the estimator anchors
     * its local probes on it. */
    anneal::ChainState<Vec> chain;
    chain.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, 0);
    chain.V   = proposal_scale;
    chain.x   = Vec(dim);
    if (proceed) {
      for (std::size_t i = 0; i < dim; ++i) chain.x(i) = initial[i];

      if (!budget.try_reserve()) {
        stop_reason = StopReason::NoBudget;
        proceed     = false;
      } else {
        chain.f = host(sp.out(chain.x));
        ran_sa = true;
        /* A synthetic/non-HPO host can return NaN.  Treat it like +Inf so
         * calibration or the first finite proposal can repair the chain. */
        if (!std::isfinite(chain.f)) chain.f =
            std::numeric_limits<double>::infinity();
        chain.best_f = chain.f;
        chain.best_x = chain.x;

        if (anneal::stopval_reached(chain.f, stopval, sp.failscore)) {
          stop_reason = StopReason::StopVal;
          proceed     = false;
        } else if (time_exceeded()) {
          stop_reason = StopReason::MaxTime;
          proceed     = false;
        }
      }
    }
    if (chain.best_x.size() == 0) chain.best_x = chain.x;

    /* --- T0 calibration (skipped when PARAM T_HOT is set) --- */
    if (proceed) {
      const auto t0out = anneal::calibrate_t0(
          sp, kb, proposal_scale, chain.x, chain.f, chain.best_x, chain.best_f,
          host, probe_rng, ctx);
      T0      = t0out.T0;
      T_final = T0;
      if (t0out.aborted) {
        if (t0out.budget_exhausted) stop_reason = StopReason::MaxEval;
        else if (t0out.stopval_hit) stop_reason = StopReason::StopValOnProbe;
        else if (t0out.external_stop) stop_reason = StopReason::ExternalOnProbe;
        else stop_reason = StopReason::MaxTimeOnProbe;
        proceed = false;
      } else if (budget.left() == 0) {
        /* Completing the final requested probe exactly at MAXEVAL is just
         * as terminal as failing to reserve the next one. */
        stop_reason = StopReason::MaxEval;
        proceed = false;
      } else if (time_exceeded()) {
        stop_reason = StopReason::MaxTimeOnProbe;
        proceed = false;
      }
    }

    /* Auto cooling floor T0*1e-6: ~86 rungs at the RT=0.85 default.
     * Explicit PARAM T_MIN always wins. */
    const double t_min = kb.t_min ? *kb.t_min
                                  : (T0 > 0.0
                                         ? anneal::automatic_t_min(T0)
                                         : 1e-12);
    if (proceed && t_min > T0)
      throw std::runtime_error(
          "TADAH:ANNEAL: resolved T_MIN must be <= T_INIT/T0 (T_MIN=" +
          std::to_string(t_min) + ", resolved T0=" +
          std::to_string(T0) + ")");

    /* --- AUTO length resolution ----------------------------------------
     * CHAIN_LENGTH/NT default to AUTO: the remaining MAXEVAL budget
     * spread evenly over the cooling schedule (T0 -> T_MIN at RT), NT
     * additionally capped at the Corana 1987 structure max(100, 5n).
     * Knobs::parse guarantees MAXEVAL is set whenever AUTO is in play,
     * and surplus budget flows into REANNEAL cycles. */
    anneal::Knobs kbr = kb;
    if (proceed) {
      const std::size_t schedule_rungs = [&]() -> std::size_t {
        if (!(T0 > 0.0) || !(t_min > 0.0) || t_min >= T0) return 1;
        const double m = std::log(t_min / T0) / std::log(kb.rt);
        if (!std::isfinite(m) || m < 0.0) return 1;
        return static_cast<std::size_t>(std::floor(m)) + 1;
      }();
      if (!kb.corana && kb.chain_length == 0) {
        kbr.chain_length = static_cast<std::size_t>(std::max<std::uint64_t>(
            1, budget.left() / schedule_rungs));
        std::ostringstream oss;
        oss << "ANNEAL auto CHAIN_LENGTH: remaining budget " << budget.left()
            << " over " << schedule_rungs
            << " scheduled rung(s) (T0 -> T_MIN at RT=" << kb.rt
            << ") -> CHAIN_LENGTH=" << kbr.chain_length;
        HPOLog::instance().info(oss.str());
      }
      if (kb.corana && kb.NT == 0) {
        const std::uint64_t cap = std::max<std::uint64_t>(
            100, 5 * static_cast<std::uint64_t>(dim));
        const std::uint64_t per_nt = anneal::saturating_mul(
            anneal::saturating_mul(schedule_rungs, kb.NS),
            std::max<std::size_t>(dim, 1));
        const std::uint64_t nt_raw =
            per_nt > 0 ? budget.left() / per_nt : 1;
        kbr.NT = static_cast<std::size_t>(
            std::min<std::uint64_t>(cap, std::max<std::uint64_t>(1, nt_raw)));
        std::ostringstream oss;
        oss << "ANNEAL auto NT: remaining budget " << budget.left()
            << " over " << schedule_rungs << " scheduled rung(s) x NS="
            << kb.NS << " x dim=" << dim << " -> NT=" << kbr.NT
            << " (cap max(100, 5*dim)=" << cap << ")";
        HPOLog::instance().info(oss.str());
      }
    }

    /* --- one annealing schedule from T_begin ---------------------------
     * Shared by the first anneal and every REANNEAL cycle. */
    const std::uint64_t rung_cost =
        kbr.corana ? anneal::saturating_mul(
                         anneal::saturating_mul(kbr.NT, kbr.NS), dim)
                   : static_cast<std::uint64_t>(kbr.chain_length);
    /* Layout arithmetic -- exact, printed before the first per-rung line
     * so the rung cadence (and hence the journal cadence) is visible. */
    if (proceed) {
      std::ostringstream oss;
      oss << "ANNEAL layout: rung = ";
      if (kbr.corana)
        oss << "NT*NS*dim = " << kbr.NT << "*" << kbr.NS << "*" << dim
            << " = ";
      else
        oss << "CHAIN_LENGTH = ";
      oss << rung_cost << " evals";
      if (sp.stop.maxeval && rung_cost > 0) {
        const std::uint64_t maxeval =
            static_cast<std::uint64_t>(*sp.stop.maxeval);
        oss << ";  MAXEVAL=" << maxeval << " -> <= "
            << (maxeval / rung_cost + (maxeval % rung_cost != 0 ? 1 : 0))
            << " rung(s) by budget (T_MIN schedule / plateau may end "
               "earlier, REANNEAL restarts extend rung count)";
      }
      HPOLog::instance().info(oss.str());
      if (rung_cost > 0)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(rung_cost), "rung",
            sp.stop.maxeval
                ? static_cast<std::uint64_t>(*sp.stop.maxeval) : 0);
    }
    auto run_schedule = [&](double T_begin) {
      double T = T_begin;
      if (budget.left() == 0) {
        stop_reason = StopReason::MaxEval;
        return;
      }
      if (time_exceeded()) {
        stop_reason = StopReason::MaxTime;
        return;
      }
      if (T < t_min) {
        stop_reason = StopReason::TMin;
        return;
      }
      /* Planned total for the rung=current/total display: whichever of
       * the T_MIN schedule or the eval budget will end this schedule
       * first (plateau/STOPVAL may still end it earlier). Recomputed
       * per REANNEAL cycle on top of the rungs already completed. */
      const std::size_t schedule_total = anneal::saturating_add(
          rung_count, anneal::planned_rungs(T_begin, t_min, kb.rt,
                                            budget.left(), rung_cost));
      while (T >= t_min && budget.left() > 0 && !time_exceeded()) {
        const auto rs = anneal::run_rung(chain, T, sp, kbr, host, ctx);
        if (kbr.observer) {
          std::vector<double> xv(dim);
          for (std::size_t d = 0; d < dim; ++d) xv[d] = chain.x(d);
          kbr.observer->on_rung_end(0, T, xv, chain.f);
        }

        /* Pinned-dim audit refresh: marks follow the chain's post-rung
         * state -- a dim that left its bound is unmarked here; the table
         * prints with the milestone SUMMARY. Applies to both steppers:
         * pinned_dims() drops its Corana-only step-floor clause under
         * CLASSIC and reports pure near-bound occupancy. */
        HPOLog::instance().record_pin_state(
            0, rung_count + 1, anneal::pinned_dims(chain, sp, kbr));

        /* Per-rung summary line. Emitted whenever the rung consumed
         * any host evals, even if we broke out mid-rung. */
        if (rs.proposed > 0 || rs.oob > 0) {
          const double rate = rs.proposed > 0
              ? static_cast<double>(rs.accepted) / static_cast<double>(rs.proposed) : 0.0;
          const double oob_pct = rs.proposed > 0
              ? 100.0 * static_cast<double>(rs.oob) /
                    static_cast<double>(rs.proposed)
              : 0.0;
          if (rs.proposed > 0) {
            any_rate = true;
            if (rate < accept_rate_min) accept_rate_min = rate;
            if (rate > accept_rate_max) accept_rate_max = rate;
          }
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(3)
              << (kb.corana ? "ANNEAL[CORANA] rung=" : "ANNEAL rung=")
              << (rung_count + 1) << "/" << schedule_total
              << "  T=" << T
              << "  proposed=" << rs.proposed
              << "  accepted=" << rs.accepted
              << " (" << std::fixed << std::setprecision(1) << (rate * 100.0) << "%)"
              << "  oob=" << rs.oob
              << " (" << oob_pct << "%)"
              << std::scientific << std::setprecision(3)
              << "  f_x=" << chain.f
              << "  best=" << chain.best_f;
          if (kb.corana) {
            const double V_min = *std::min_element(chain.V.begin(), chain.V.end());
            const double V_max = *std::max_element(chain.V.begin(), chain.V.end());
            const auto sat = anneal::step_saturation(chain.V, sp, kbr);
            oss << "  V=[" << V_min << ", " << V_max << "]"
                << "  V_floor=" << sat.floor << "/" << sat.total
                << "  V_cap=" << sat.cap << "/" << sat.total;
          }
          HPOLog::instance().info(oss.str());
        }
        total_proposed += rs.proposed;
        total_accepted += rs.accepted;
        total_oob      += rs.oob;
        T_final         = T;
        ++rung_count;

        if (rs.stop != RungStop::Completed) {
          switch (rs.stop) {
            case RungStop::Budget:  stop_reason = StopReason::MaxEval; break;
            case RungStop::Time:    stop_reason = StopReason::MaxTime; break;
            case RungStop::StopVal: stop_reason = StopReason::StopVal; break;
            default: break;
          }
          break;
        }

        if (budget.left() == 0) { stop_reason = StopReason::MaxEval; break; }
        if (time_exceeded()) { stop_reason = StopReason::MaxTime; break; }
        if (plateau.update(chain.f, chain.best_f)) {
          stop_reason = StopReason::Plateau; break;
        }

        T = anneal::cool_temperature(T, kb.rt);
        if (T < t_min) { stop_reason = StopReason::TMin; break; }

        /* Corana reset-to-best after each cool. Cheap (no host
         * eval consumed); just snaps the chain back to the best-so-far
         * point. The next rung then refines around best_x with the
         * freshly-cooled T and the V[] adapted so far. */
        if (kb.reset_to_best && chain.f > chain.best_f) {
          chain.x = chain.best_x;
          chain.f = chain.best_f;
        }
      }
    };

    /* --- main anneal + warm re-anneal driver ---------------------------
     * When a schedule cools out (T < T_MIN) with budget left, REANNEAL
     * snaps the chain to the best-so-far and restarts the schedule at
     * REANNEAL_SCALE^n * T0, so leftover budget refines rather than
     * truncating progress mid-improvement. Stops re-annealing when the
     * next start temperature could not complete ~2 rungs above T_MIN. */
    if (proceed) {
      double T_start = T0;
      for (;;) {
        run_schedule(T_start);
        if (stop_reason != StopReason::TMin) break;
        if (!kb.reanneal) break;
        if (budget.left() == 0 || time_exceeded()) break;
        const double next_T =
            T0 * std::pow(kb.reanneal_scale,
                          static_cast<double>(reanneal_count + 1));
        if (next_T < t_min / kb.rt) break;
        ++reanneal_count;
        chain.x = chain.best_x;
        chain.f = chain.best_f;
        T_start = next_T;
        stop_reason = StopReason::Done;
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "ANNEAL re-anneal #" << reanneal_count
            << ": T_start=" << T_start
            << "  from best=" << chain.best_f;
        HPOLog::instance().info(oss.str());
      }
    }

    /* --- final SA summary block (only when SA actually ran past the
     * seed evaluation). --- */
    if (ran_sa) {
      const double overall_rate = total_proposed > 0
          ? static_cast<double>(total_accepted) / static_cast<double>(total_proposed) : 0.0;
      const double oob_pct = total_proposed > 0
          ? 100.0 * static_cast<double>(total_oob) /
                static_cast<double>(total_proposed)
          : 0.0;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(3)
          << "ANNEAL summary: rungs=" << rung_count
          << "  proposed=" << total_proposed
          << "  accepted=" << total_accepted
          << " (" << std::fixed << std::setprecision(1) << (overall_rate * 100.0) << "%)"
          << "  oob=" << total_oob
          << " (" << oob_pct << "%)"
          << std::scientific << std::setprecision(3)
          << "  T0=" << T0
          << "  T_final=" << T_final
          << "  reanneals=" << reanneal_count
          << "  best=" << chain.best_f;
      if (kb.corana) {
        const auto sat = anneal::step_saturation(chain.V, sp, kbr);
        oss << "  V_floor=" << sat.floor << "/" << sat.total
            << "  V_cap=" << sat.cap << "/" << sat.total;
      }
      HPOLog::instance().info(oss.str());
      if (any_rate) {
        std::ostringstream oss2;
        oss2 << std::fixed << std::setprecision(1)
             << "ANNEAL accept-rate trajectory: min="
             << (accept_rate_min * 100.0) << "%  max="
             << (accept_rate_max * 100.0) << "%";
        HPOLog::instance().info(oss2.str());
      }
    }

    /* --- explicit stop reason (single line, every exit path) --- */
    {
      const char* reason = nullptr;
      switch (stop_reason) {
        case StopReason::MaxEval:        reason = "MAXEVAL exhausted"; break;
        case StopReason::MaxTime:        reason = "MAXTIME exceeded"; break;
        case StopReason::StopVal:        reason = "STOPVAL reached"; break;
        case StopReason::StopValOnProbe: reason = "STOPVAL reached during T0 probes"; break;
        case StopReason::ExternalOnProbe: reason = "external stop during T0 probes"; break;
        case StopReason::MaxTimeOnProbe: reason = "MAXTIME exceeded during T0 probes"; break;
        case StopReason::TMin:           reason = "T < T_MIN"; break;
        case StopReason::Plateau:        reason = "FTOL plateau (Corana eps-NEPS rule)"; break;
        case StopReason::NoDims:         reason = "no optimisable dimensions"; break;
        case StopReason::NoBudget:       reason = "MAXEVAL exhausted before SA proposal"; break;
        case StopReason::Done:           reason = "loop completed normally"; break;
      }
      std::ostringstream oss;
      oss << "ANNEAL stopped: " << reason
          << " after " << budget.spent() << " host eval(s)"
          << " (" << rung_count << " rung(s) completed";
      if (reanneal_count > 0) oss << ", " << reanneal_count << " re-anneal(s)";
      oss << ")";
      HPOLog::instance().info(oss.str());
    }
  }

template<typename Vec, typename Runner>
  void run_anneal(const OptimizerSpec& sp,
      HPO_Host<Vec,Runner>& host)
  {
    run_anneal_impl<Vec>(sp, host);
  }
}   // namespace

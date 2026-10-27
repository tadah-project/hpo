#pragma once
/*
 * RunPolicy -- execution decorator around the per-optimiser dispatch.
 *
 * BasePolicy runs the optimiser exactly once via run_inner(). The other
 * policies (MultiStartPolicy, StagedPolicy, RestartPolicy) are wrappers
 * around the same run_inner().
 *
 * Splitting "what to run" (spec / algorithm dispatch) from "how to drive
 * it" (single shot, multi-start, restart, two-stage) keeps each concern
 * testable in isolation and lets the eventual init-strategy plumbing live
 * outside the optimiser adapters.
 */

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

/* RunPolicy<> + run_inner<> (the single-shot dispatch) live in
 * run_policy_base.h; this header builds the decorator stack on top. */
#include <tadah/core/checkpoint.h>        // core::ShutdownRequested
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/run_policy_base.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/stagnation_stop.h>
#include <tadah/hpo/strategy_rules.h>

namespace tadah::hpo {

namespace detail {

/* How one MULTISTART start (or PRESTAGE refinement seed) ended. */
enum class StartOutcome { Completed, Stagnated, Failed };

/* Run MULTISTART start i (0-based) of k.
 *   Completed  the inner optimiser returned;
 *   Stagnated  it stopped on StagnationStop: the RESTART window of this
 *              start ran out; the batch reports it after its last start
 *              (report_stagnation);
 *   Failed     a failure confined to that start -- a rejected setup, an
 *              optimiser error -- logged (journal WARNING and stderr) and,
 *              when `failure` is given and still empty, stored there.
 * Exceptions about the whole run propagate:
 *   durable_best::Error      the best potential can no longer be saved;
 *   core::ShutdownRequested  SIGINT/SIGTERM/SIGUSR2 -- stop now;
 * and so do non-standard exceptions. A Stagnated or Failed start whose run
 * has recorded a best potential that could not be saved (a concurrent
 * exception won over the durable_best::Error) throws that error instead
 * (rethrow_if_best_unsaved). */
template<class Fn>
StartOutcome run_start_guarded(std::size_t i, std::size_t k, Fn&& fn,
                               std::exception_ptr* failure = nullptr)
{
  try {
    fn();
    return StartOutcome::Completed;
  } catch (const durable_best::Error&) {
    throw;
  } catch (const core::ShutdownRequested&) {
    throw;
  } catch (const StagnationStop&) {
    /* A concurrent persistence failure may have lost to this stop. */
    rethrow_if_best_unsaved();
    return StartOutcome::Stagnated;
  } catch (const std::exception& e) {
    rethrow_if_best_unsaved();
    const std::string msg = "MULTISTART start " + std::to_string(i+1) + "/" +
                            std::to_string(k) + " failed: " + e.what();
    HPOLog::instance().warning(msg);
    std::cerr << "[HPO] WARNING: " << msg << "\n";
    if (failure && !*failure) *failure = std::current_exception();
    return StartOutcome::Failed;
  }
}

/* `status` without the failed-starts note (failed_starts_suffix) at its end. */
inline std::string strip_failed_starts_note(std::string status)
{
  static const std::string tail = "; see the WARNINGs)";
  if (status.size() >= tail.size() &&
      status.compare(status.size() - tail.size(), tail.size(), tail) == 0) {
    const auto open = status.rfind(" (");
    if (open != std::string::npos &&
        status.find(" MULTISTART starts failed", open) != std::string::npos)
      status.erase(open);
  }
  return status;
}

/* " (N of M MULTISTART starts failed[ across L launches]; see the
 * WARNINGs)" for the run's start tally (HPOLog::start_tally), or "" when no
 * start failed. */
inline std::string failed_starts_suffix(const HPOLog::StartTally& t)
{
  if (t.failed == 0) return {};
  std::string out = " (" + std::to_string(t.failed) + " of " +
                    std::to_string(t.ran) + " MULTISTART starts failed";
  if (t.batches > 1) out += " across " + std::to_string(t.batches) + " launches";
  return out + "; see the WARNINGs)";
}

/* Counts of one batch of starts (run_starts). */
struct StartsReport {
  std::size_t k           = 0;
  std::size_t n_failed    = 0;
  std::size_t n_stagnated = 0;
  bool        improved    = false;   // the global best improved during the batch
};

/* Run starts 0..k-1 in order. `reset_window()` runs before each start, so
 * every start gets its own stagnation window (a no-op without RESTART:
 * the threshold is 0); `run_one(i)` runs start i. A start that stagnates
 * or fails ends, and the next one runs.
 *
 * Status: a failed start's own status (NLopt records "FAILED -- ..."
 * before it rethrows) is replaced by the last good one, so it cannot become
 * the RESULT of an otherwise successful batch. Every batch adds to the
 * run's start tally (HPOLog::record_start_batch). After the batch:
 *   - every start failed  -> the first failure is rethrown: the run fails
 *     (FAILED, non-zero exit) as it would without MULTISTART;
 *   - a start of this or an earlier batch (a RESTART relaunch) failed ->
 *     the status gets "(N of M MULTISTART starts failed[ across L
 *     launches]; see the WARNINGs)" -- counted over the run, so a later
 *     clean batch neither drops nor misstates it, and a note carried in
 *     from an earlier batch is replaced, not stacked. The note goes onto
 *     a recorded status only: a batch whose last good start stagnated has
 *     none (its RESTART window ran out; the batch reports the stagnation,
 *     and RESTART's own status carries the note), and a start that ended
 *     without one is the run driver's internal error, not a COMPLETED
 *     invented here.
 * Stagnation is reported by the caller (report_stagnation). Generic over
 * the two callables so it runs without an HPO_Host in the unit tests. */
template<class ResetFn, class RunOne>
StartsReport run_starts(std::size_t k, ResetFn&& reset_window, RunOne&& run_one)
{
  HPOLog& log = HPOLog::instance();
  StartsReport rep;
  rep.k = k;
  std::exception_ptr first_failure;
  /* A note from an earlier batch is rewritten from the run's tally below. */
  std::string ok_status = strip_failed_starts_note(log.status());
  const double best_at_start = log.best_loss();

  for (std::size_t i = 0; i < k; ++i) {
    /* Snapshot global state at start so we can summarise this start's
     * contribution: how many evals it consumed and whether it produced
     * a new global best. Without this each start beyond the first
     * appears "silent" in the journal whenever its local fmin doesn't
     * beat the running global best. */
    const std::size_t evals_before = log.n_eval();
    const double      best_before  = log.best_loss();
    reset_window();
    const StartOutcome out =
        run_start_guarded(i, k, [&] { run_one(i); }, &first_failure);
    const std::size_t evals_after = log.n_eval();
    const double      best_after  = log.best_loss();

    if (out == StartOutcome::Failed) {
      ++rep.n_failed;
      log.set_status(ok_status);
    } else {
      if (out == StartOutcome::Stagnated) ++rep.n_stagnated;
      ok_status = strip_failed_starts_note(log.status());
    }

    std::ostringstream done;
    done << "MULTISTART start " << (i+1) << "/" << k
         << " done: " << (evals_after - evals_before) << " eval(s) consumed";
    {
      std::ostringstream tmp;
      tmp << std::scientific << std::setprecision(4);
      if (best_after < best_before)
        tmp << ", improved global best " << best_before << " -> " << best_after;
      else
        tmp << ", no global improvement (best stays at " << best_after << ")";
      done << tmp.str();
    }
    if (out == StartOutcome::Stagnated)
      done << "; stopped on stagnation (RESTART window)";
    else if (out == StartOutcome::Failed)
      done << "; FAILED (see the WARNING above)";
    std::cout << "[HPO] " << done.str() << "\n";
    log.info(done.str());
  }

  rep.improved = log.best_loss() < best_at_start;
  log.record_start_batch(k, rep.n_failed);
  if (k > 0 && rep.n_failed == k) std::rethrow_exception(first_failure);
  const HPOLog::StartTally tally = log.start_tally();
  if (tally.failed > 0 && !ok_status.empty())
    log.set_status(ok_status + failed_starts_suffix(tally));
  return rep;
}

/* Which batch of starts a MultiStartPolicy runs: MULTISTART's starts or
 * the PRESTAGE refinement's seeds. Only the journal wording differs; both
 * report stagnation by the same rule (report_stagnation). */
enum class StartBatch { MultiStart, Refinement };

/* After the last start of a batch: when any start (or refinement seed) ran
 * out of its stagnation window, the batch reports StagnationStop upwards,
 * to RestartPolicy, which draws fresh points and relaunches the whole
 * strategy -- whether or not the batch improved the global best, as under
 * PLAIN, where a run that improved and then stagnated restarts too. A run
 * under an active RESTART therefore never ends on a stagnation a batch
 * absorbed: it restarts, or, once its MAX restarts are used, ends with
 * RESTART's "STOPPED -- stagnation: ..." status. */
inline void report_stagnation(std::size_t n_stagnated)
{
  if (n_stagnated > 0) throw StagnationStop{};
}

}  // namespace detail

/* ----------------------------------------------------------------------
 * BasePolicy -- single-shot. Used as the default and as the innermost
 * layer of every decorator stack.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class BasePolicy : public RunPolicy<Vec,Runner> {
public:
  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    run_inner<Vec,Runner>(spec, host);
  }
};

/* ----------------------------------------------------------------------
 * MultiStartPolicy -- runs the inner optimiser K times from K starting
 * points, taking the K points from spec.optim_init (which the
 * InitStrategy chain has already populated). Local optimisers benefit
 * the most: each run can descend into a different local minimum; the
 * host's best-tracker collects the global best across the K runs.
 *
 * Compatibility (from the caps registry; strategy_rules.h):
 *   - backends the registry does not flag global (DLIB:BFGS, CERES:*,
 *     NLOPT:LD_*, NLOPT:LN_*, NLOPT:AUGLAG*)
 *                                         -> K independent runs (intended).
 *   - multi-seed globals (DLIB:GFS)       -> take every point in one
 *     run, so MULTISTART would run them once whatever N_STARTS:
 *     require_multistart_backend() rejects them (STRATEGY TYPE PLAIN runs
 *     them once from every point). A batch of the PRESTAGE refinement
 *     (StartBatch::Refinement) seeds the TOPK candidates into
 *     find_min_global's `initial_function_evals`, one run.
 *   - every other global (NLOPT:G*, TADAH:RANDOM/GRID/ANNEAL*, ASA)
 *                                         -> takes one starting point,
 *     so require_multistart_backend() rejects it for any N_STARTS; a
 *     local backend, or STRATEGY TYPE PLAIN, runs instead.
 *     validate_hpo_setup() applies the rule at startup; validate()
 *     applies the same function again when the policy runs.
 *
 * The starts run one after another (the parser refuses PARALLEL, which
 * asked for starts at the same time). Each start's evaluations run on the
 * shared worker pool (THREADS), which runs the pc_lammps fan-out and the
 * training and prediction of an evaluation.
 *
 * A start that fails is logged and the next one runs; a batch in which
 * every start failed fails the run (detail::run_starts). Two exceptions
 * end the whole multistart at once, because they are about the run, not
 * the start: core::ShutdownRequested (a signal: drive_run() ends the run)
 * and durable_best::Error (the incumbent can no longer be persisted).
 *
 * Under RESTART every start gets its own stagnation window; a start that
 * runs out of it ends and the next start runs. After the last start the
 * batch reports StagnationStop to RestartPolicy (which relaunches the
 * whole strategy: all K starts from K fresh points, or the whole PRESTAGE)
 * when any start stagnated (detail::report_stagnation). The PRESTAGE
 * refinement is a batch of this kind (`batch` Refinement), with the same
 * rule; only its journal lines differ.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class MultiStartPolicy : public RunPolicy<Vec,Runner> {
public:
  MultiStartPolicy(std::size_t k,
                   std::optional<std::size_t> budget_per_start = std::nullopt,
                   detail::StartBatch batch = detail::StartBatch::MultiStart)
      : k_(k), budget_per_start_(budget_per_start), batch_(batch) {}

  /* The startup rules (strategy_rules.h), applied again when the policy
   * runs: a safety net for specs built outside HPOTargets. */
  static void validate(const OptimizerSpec& spec) {
    require_multistart_backend(spec);
    require_inner_backend(spec, run_phase_name(spec));
  }

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    validate(spec);

    /* The plan and each start's spec come from phase_specs.h, which
     * validate_hpo_setup() derives the startup checks of each start from
     * too. A multi-seed backend (DLIB:GFS) consumes every INIT point in a
     * single run, and so does a spec without INIT points. */
    const MultiStartPlan plan =
        plan_multistart(spec, k_, budget_per_start_);
    if (plan.single_run) {
      run_single(spec, host);
      return;
    }

    const std::size_t actual_k = plan.starts;
    if (actual_k < k_) warn_fewer_starts(spec, actual_k);
    HPOLog::instance().info(
        "MULTISTART: running " + std::to_string(actual_k) +
        (actual_k == 1 ? " start" : " starts"));

    /* Each start's MAXEVAL: BUDGET_PER_START, or the outer MAXEVAL split
     * over the starts (plan_multistart); without either the starts run
     * with the spec's own stop criteria. */
    if (plan.per_start_maxeval) {
      std::ostringstream oss;
      oss << "MULTISTART per-start MAXEVAL = " << *plan.per_start_maxeval
          << (plan.from_budget_per_start ? " (from BUDGET_PER_START)"
                                         : " (split from outer MAXEVAL)");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* A failure in one start does not stop the batch; a run-level
     * exception (fatal persistence, shutdown) does. Each start runs in its
     * own stagnation window (reset_stagnation; a no-op without RESTART). */
    const detail::StartsReport rep = detail::run_starts(
        actual_k,
        [] { HPO_Host<Vec,Runner>::reset_stagnation(); },
        [&](std::size_t i) {
          /* Start i: its INIT point, its MAXEVAL and SEED + i
           * (multistart_start_spec, phase_specs.h). */
          const OptimizerSpec sub = multistart_start_spec(spec, plan, i);

          /* The start's seed: its SEED, or the INIT block's SEED that NLopt
           * takes without one, when it draws random numbers (reads_seed:
           * LN_SBPLX takes a seed it never reads). */
          const std::string start_msg =
              "MULTISTART start " + std::to_string(i+1) + "/" +
              std::to_string(actual_k) + " -- running inner optimiser" +
              (sub.seed ? " (seed=" + std::to_string(*sub.seed) + ")"
               : sub.nlopt_seed_from_init() && reads_seed(sub)
                   ? " (INIT seed=" +
                         std::to_string(*sub.init_strategy_spec.seed) + ")"
                   : "");
          std::cout << "[HPO] " << start_msg << "\n";
          HPOLog::instance().info(start_msg);
          run_inner<Vec,Runner>(sub, host);
        });
    report(rep.n_stagnated, rep.improved, actual_k);
  }

  /* One inner run (all of spec's points) in its own stagnation window; a
   * StagnationStop ends it and is reported by the batch rule
   * (detail::report_stagnation). A failure propagates (nothing else runs
   * to carry the result). */
  void run_one_windowed(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) {
    const double best_before = HPOLog::instance().best_loss();
    HPO_Host<Vec,Runner>::reset_stagnation();
    bool stagnated = false;
    try {
      run_inner<Vec,Runner>(spec, host);
    } catch (const StagnationStop&) {
      detail::rethrow_if_best_unsaved();
      stagnated = true;
    }
    report(stagnated ? 1 : 0, HPOLog::instance().best_loss() < best_before, 1);
  }

private:
  /* The WARNING of a batch of fewer starts than N_STARTS, naming the chain
   * that gave the points and why they are fewer:
   *  - the INIT chain: the startup check took its count
   *    (require_multistart_points), so it gave fewer when it was drawn: a
   *    WARM point dropped under WARM_ON_MISMATCH skip, a drawn point equal
   *    to another, which the chain keeps once. "MULTISTART N_STARTS 3: the
   *    INIT chain gave 2 starting points when it was drawn (WARM_ON_MISMATCH
   *    skip drops a WARM point, and a point equal to another counts once);
   *    running 2 starts, and N_STARTS 2 asks for the 2 that run." A lower
   *    N_STARTS is named for two starts or more (N_STARTS is 2 or more; one
   *    start is STRATEGY TYPE PLAIN);
   *  - a RESTART round (OptimizerSpec::init_from_restart), whose chain the
   *    startup check counts too for a parsed file (require_restart_points),
   *    so one of N_STARTS points or more gets the cause, as above; a chain
   *    of fewer points as written, which only a spec built outside the
   *    parser has, gets the count and what gives N_STARTS starts, as the
   *    startup refusal does ("... the RESTART round gave 1 starting point;
   *    running 1 start. The RESTART chain gives 1 (RANDOM 1); a chain of 3
   *    or more starting points (METHOD LHS 3, RANDOM 3 or SOBOL 3) runs 3
   *    starts."). */
  void warn_fewer_starts(const OptimizerSpec& spec, std::size_t actual_k) const {
    const bool restart = spec.init_from_restart;
    const ChainPoints c = chain_points(restart ? spec.restart.init_strategy
                                               : spec.init_strategy_spec);
    const std::string k = std::to_string(k_);
    const std::string got = std::to_string(actual_k);
    const std::string gave =
        got + (actual_k == 1 ? " starting point" : " starting points");
    std::string msg = "MULTISTART N_STARTS " + k + ": the " +
                      (restart ? "RESTART round" : "INIT chain") + " gave " +
                      gave;
    const bool written_fewer = restart && c.points < k_;
    if (!written_fewer)
      msg += std::string(" when it was drawn (") +
             (c.warm_skip ? "WARM_ON_MISMATCH skip drops a WARM point, and "
                          : "") +
             "a point equal to another counts once)";
    msg += "; running " + got + (actual_k == 1 ? " start" : " starts");
    if (written_fewer) {
      msg += ". The RESTART chain gives " +
             std::string(c.warm ? "at most " : "") +
             std::to_string(c.points) + " (" + c.from() + "); " +
             multistart_chain_advice(c, k_) + ".";
    } else if (restart) {
      msg += ".";
    } else if (actual_k == 1) {
      msg += ", which is STRATEGY TYPE PLAIN.";
    } else {
      msg += ", and N_STARTS " + got + " asks for the " + got + " that run.";
    }
    HPOLog::instance().warning(msg);
  }

  /* One inner run for all points (no per-start loop). MULTISTART: a
   * StagnationStop propagates as under PLAIN. PRESTAGE refinement: see
   * run_one_windowed. */
  void run_single(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) {
    if (batch_ == detail::StartBatch::MultiStart) {
      run_inner<Vec,Runner>(spec, host);
      return;
    }
    run_one_windowed(spec, host);
  }

  /* Journal the stagnation of a batch, then report it
   * (detail::report_stagnation). A refinement that improved the global
   * best says so: it restarts all the same. */
  void report(std::size_t n_stagnated, bool improved, std::size_t k) const {
    if (n_stagnated == 0) return;
    const bool ms = batch_ == detail::StartBatch::MultiStart;
    std::ostringstream oss;
    oss << (ms ? "MULTISTART: " : "REFINEMENT: ") << n_stagnated << " of "
        << k << (ms ? " start(s)" : " seed(s)") << " stopped on stagnation";
    if (!ms && improved) oss << " (the refinement improved the global best)";
    oss << "; reporting stagnation to RESTART";
    std::cout << "[HPO] " << oss.str() << "\n";
    HPOLog::instance().info(oss.str());
    detail::report_stagnation(n_stagnated);
  }

  std::size_t k_;
  std::optional<std::size_t> budget_per_start_;
  detail::StartBatch batch_;
};

/* ----------------------------------------------------------------------
 * StagedPolicy -- runs the PRESTAGE block as a global exploration phase,
 * collects the top-K best (loss, params) pairs, then refines each via
 * the outer optimiser.
 *
 * Selected by STRATEGY TYPE PRESTAGE; the exploration phase is the
 * OptimizerSpec::prestage sub-spec parsed from the EXPLORE ... ENDEXPLORE
 * block, the refinement is the outer spec (STRATEGY-level keys or an
 * optional REFINE ... ENDREFINE block). With a configured TOPK > 1 the
 * refinement is a multistart over the candidates, so its optimiser must be
 * one MULTISTART accepts: require_prestage_refinement() (strategy_rules.h)
 * says so at startup, whatever number of candidates the exploration will
 * harvest.
 *
 * The two phases:
 *   1. PRESTAGE (broad search):    spec.prestage          (e.g. DLIB:GFS)
 *   2. REFINEMENT (local sweep):   spec     (the outer)   (e.g. DLIB:BFGS)
 *
 * The candidates are the exploration's usable losses (HPO_Host's top-K
 * takes no failed evaluation): with fewer than TOPK the refinement runs
 * from those, and the journal says how many the exploration made and how
 * many failed. When every evaluation failed the refinement runs from the
 * outer spec's own points (the OPTIMIZER-level INIT points, or a RESTART
 * round's fresh ones), with a WARNING that says why: the exploration's
 * failed points are no seeds, and the refinement may still find a usable
 * loss from those points (a search under FAILSCORE_GRADING ON walks out of
 * a failure region). The candidates include the points of calls the
 * exploration answered from the run's record of an earlier evaluation (a
 * RESTART round's exploration asking for points of an earlier round:
 * HPO_Host::count_reused_repeat), as the evaluations they stand for would
 * be: an exploration whose every call the record answers makes no
 * evaluation and still hands on its candidates (a RESTART round retracing
 * an earlier one; under NOISE CALIBRATION an exploration whose one call,
 * MAXEVAL 1, is the seed point the calibration evaluated). An exploration
 * that asked for no point -- no evaluation, and no call answered from the
 * record -- is an error (std::logic_error): every exploration a parsed
 * configuration sets asks for at least one point (MAXEVAL >= 1, NLopt
 * counting a call the record answers too; every driver makes its first
 * call, also when MAXTIME has passed before it; a stagnation stop is
 * raised at an evaluation or at a call answered from the record), so its
 * driver returned without asking, or a spec built outside the parser asked
 * for nothing; it ran the refinement from the INIT points with a WARNING,
 * which hid that.
 *
 * Under RESTART a stagnating exploration hands its top-K over to the
 * refinement early (the exploration catch below) instead of restarting;
 * each refinement seed then gets its own stagnation window. When a
 * refinement seed stagnated, the staged run reports StagnationStop to
 * RestartPolicy after the refinement's last seed, whether or not the
 * refinement improved the global best (detail::report_stagnation), and
 * RestartPolicy re-runs both phases from a fresh init.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class StagedPolicy : public RunPolicy<Vec,Runner> {
public:
  explicit StagedPolicy(std::size_t top_k) : top_k_(top_k) {
    if (top_k_ == 0)
      throw std::runtime_error("Staged-global mode requires TOPK >= 1");
  }

  /* The startup rules (strategy_rules.h), applied again when the policy
   * runs: a safety net for specs built outside HPOTargets. */
  static void validate(const OptimizerSpec& spec) {
    require_prestage_explore(spec);
    require_prestage_refinement(spec);
    require_inner_backend(spec, run_phase_name(spec));
  }

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    validate(spec);

    /* Step 1: enable top-K tracking, run the PRESTAGE. Real-space box
     * widths normalise the parameter half of the near-clone dedup (topk.h). */
    std::vector<double> widths(spec.low.size());
    for (std::size_t d = 0; d < widths.size(); ++d) {
      if (spec.space) {
        widths[d] = spec.space->real_width(d);
      } else {
        const bool log_d = d < spec.log_scaled.size() && spec.log_scaled[d];
        widths[d] = log_d ? std::exp(spec.high[d]) - std::exp(spec.low[d])
                          : spec.high[d] - spec.low[d];
      }
    }
    HPO_Host<Vec,Runner>::set_track_topk(top_k_, std::move(widths));
    {
      std::ostringstream oss;
      oss << "PRESTAGE: exploration phase "
          << spec.prestage->lib << ":" << spec.prestage->algo
          << " (MAXEVAL=";
      if (spec.prestage->stop.maxeval)
        oss << *spec.prestage->stop.maxeval << ")";
      else
        oss << "none: no evaluation limit)";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    /* The exploration's evaluations and failed ones (the journal's census,
     * HPOLog::record_eval): the candidates are its usable losses only
     * (HPO_Host's top-K takes no failed one), and the journal says how
     * many failed when fewer than TOPK, or none, are left. */
    const std::size_t evals_before    = HPOLog::instance().n_eval();
    const std::size_t unusable_before = HPOLog::instance().n_unusable();
    const std::size_t reused_before   = HPOLog::instance().repeats_reused();
    try {
      run_inner<Vec,Runner>(*spec.prestage, host);
    } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
      /* Stagnation during the PRESTAGE is a valid stop signal --
       * proceed to refinement with whatever we collected -- unless a new
       * best could not be saved meanwhile (it may have lost to this stop
       * in a parallel explorer): then the run fails now. */
      detail::rethrow_if_best_unsaved();
    }
    const std::size_t explore_evals =
        HPOLog::instance().n_eval() - evals_before;
    const std::size_t explore_failed =
        HPOLog::instance().n_unusable() - unusable_before;
    /* Its calls answered from the run's record of an earlier evaluation
     * (repeat_points.h): a RESTART round's exploration can ask for points
     * of an earlier round. Each is a candidate (count_reused_repeat). */
    const std::size_t explore_reused =
        HPOLog::instance().repeats_reused() - reused_before;
    /* Step 2: snapshot top-K, disable tracking. Stage boundary: the
     * refinement starts a stagnation window of its own (every path; each
     * seed also gets its own). Without this it inherited the exploration's
     * count, so after a stagnating exploration its first evaluation
     * already crossed the threshold. No-op without RESTART (threshold 0). */
    auto topk = HPO_Host<Vec,Runner>::get_topk();
    HPO_Host<Vec,Runner>::set_track_topk(0);
    HPO_Host<Vec,Runner>::reset_stagnation();

    /* The refinement is a batch of seeds: each in its own window, a
     * stagnating seed reported after the last one. */
    MultiStartPolicy<Vec,Runner> refiner(
        std::max<std::size_t>(topk.size(), 1), std::nullopt,
        detail::StartBatch::Refinement);

    /* An exploration that asked for no point: no candidate and nothing
     * that failed. No configuration the parser accepts gets here (the class
     * comment), so its driver returned without asking; the refinement from
     * the INIT points hid that behind a WARNING. A point answered from the
     * run's record was asked for, and is a candidate: an exploration that
     * made no evaluation but asked for such points goes on. */
    if (explore_evals == 0 && explore_reused == 0) {
      std::ostringstream oss;
      oss << "PRESTAGE: the exploration (EXPLORE block) "
          << spec.prestage->lib << ":" << spec.prestage->algo
          << " returned without evaluating a point, so it has no candidate "
             "to refine (it asked for no point: no call was answered from "
             "the run's record either, and every exploration a "
             "configuration sets asks for at least one point; its driver, "
             "or a spec built outside the parser, asked for none)";
      throw std::logic_error(oss.str());
    }
    /* No candidate: every evaluation of the exploration failed (a failed
     * loss is no seed; the refinement used to start from such points while
     * the top-K took them). The refinement runs from the outer spec's own
     * points, as a run of it without the PRESTAGE would (after a stagnation
     * stop too: a stagnated exploration hands over to the refinement, here
     * without a candidate), and the journal says why. */
    if (topk.empty()) {
      std::ostringstream oss;
      oss << "PRESTAGE: no evaluation of the exploration produced a usable "
             "loss (" << explore_evals << " evaluation(s)); refining from "
             "the OPTIMIZER-level INIT point(s) instead";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().warning(oss.str());
      refiner.run_one_windowed(prestage_refinement_spec(spec, spec.optim_init),
                               host);
      return;
    }
    /* Fewer candidates than TOPK: the exploration made fewer distinct
     * points with a usable loss; the refinement runs from those. */
    if (topk.size() < top_k_) {
      std::ostringstream oss;
      oss << "PRESTAGE: " << topk.size() << " usable candidate(s) for TOPK "
          << top_k_ << " (the exploration made " << explore_evals
          << " evaluation(s), " << explore_failed
          << " without a usable loss";
      if (explore_reused > 0)
        oss << ", and answered " << explore_reused
            << " call(s) from the run's record of earlier evaluations";
      oss << "); refining " << topk.size();
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Surface the snapshot so the user can sanity-check the PRESTAGE
     * actually found low-loss seeds before paying for the refinement. */
    {
      auto sorted = topk;
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
      std::ostringstream oss;
      oss << "PRESTAGE: top-" << sorted.size() << " losses:";
      const std::size_t show = std::min<std::size_t>(sorted.size(), 5);
      for (std::size_t i = 0; i < show; ++i) oss << ' ' << sorted[i].first;
      if (sorted.size() > show) oss << " ...";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }

    /* Step 3: the refinement spec (prestage_refinement_spec, phase_specs.h:
     * the outer spec without the PRESTAGE, so we do not recurse, starting
     * from the top-K params), run via MULTISTART (or a single-shot if K=1),
     * which runs candidate i with SEED + i (OptimizerSpec::seeded_starts()
     * counts TOPK of them). */
    std::vector<std::vector<double>> candidates;
    candidates.reserve(topk.size());
    for (const auto& kv : topk) {
      const Vec& params = kv.second;
      std::vector<double> v(params.size());
      for (long i = 0; i < params.size(); ++i) v[i] = params(i);
      candidates.push_back(std::move(v));
    }
    const OptimizerSpec refine =
        prestage_refinement_spec(spec, std::move(candidates));

    {
      std::ostringstream oss;
      oss << "REFINEMENT: refining top-" << topk.size()
          << " seed(s) via " << spec.lib << ":" << spec.algo;
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    if (topk.size() > 1) {
      refiner.run(refine, host);
    } else {
      refiner.run_one_windowed(refine, host);
    }
  }

private:
  std::size_t top_k_;
};

/* ----------------------------------------------------------------------
 * RestartPolicy -- wraps an inner policy with stagnation-driven restart.
 *
 * On every operator() call HPO_Host checks whether
 *   (g_iter_ - iter_at_last_best_) >= window
 * and throws StagnationStop if so. RestartPolicy catches that, draws a
 * fresh init via the configured strategy, and re-enters the inner
 * policy with the new starting point. The cycle terminates when:
 *   - the inner policy returns naturally, or
 *   - max_restarts attempts have been used: the journal status then reads
 *     "STOPPED -- stagnation: no new best in W evals after R restart(s)".
 * MULTISTART gives each start its own window and, after its last
 * start, reports stagnation when any start stagnated: a restart relaunches
 * all K starts from K fresh points. The PRESTAGE refinement does the same
 * with its seeds, and a restart re-runs both phases (see StagedPolicy).
 * An improvement of the global best earlier in the launch does not prevent
 * the restart. MAXEVAL applies per launch, so a run can use up to
 * (max_restarts + 1) x MAXEVAL evaluations.
 *
 * The refresh strategy (built from the RESTART block, whose
 * WARM_ON_MISMATCH governs its WARM seeds), its SEED and the host's
 * HPOTargets are captured at build_run_policy() time. A round whose chain
 * produces no point starts from the previous round's points, with a
 * WARNING. Restart r draws its points with S + r (modulo 2^64): S the
 * RESTART block's SEED, so the restart points of a run are reproducible,
 * or, without one, a seed drawn from std::random_device once per run, at
 * its first restart, which the journal names with the SEED that draws the
 * same points again ("RESTART: the RESTART block has no SEED, ..."; before,
 * each round drew std::random_device() + r and the journal named none).
 * The RANDOM and LHS methods of the RESTART chain draw a stream of their
 * own (InitStream::Restart, seed_stream.h), not an INIT block's for the
 * same seed. Round 0 does not use the RESTART chain: it starts from the
 * INIT points.
 *
 * A restart needs StagnationStop to propagate out of the inner policy.
 * Every optimiser adapter lets it through (NLopt, Ceres, DLIB:GFS and ASA
 * capture the exception and rethrow it after the library returns; it
 * leaves the other drivers by itself), and a restart re-initialises the
 * whole inner policy: every MULTISTART start (as many as the fresh points
 * allow), or the whole PRESTAGE -- its exploration starts from the fresh
 * points unless it has an INIT chain of its own. MULTISTART and the
 * PRESTAGE refinement report it after their last start (above);
 * StagedPolicy's exploration phase absorbs it and hands its top-K over to
 * the refinement.
 *
 * Restart r also runs every phase that has a SEED with a seed derived
 * from it (OptimizerSpec::derive_restart_seeds: the outer spec
 * SEED + r*seeded_starts() -- N_STARTS under MULTISTART, TOPK under
 * PRESTAGE, 1 under PLAIN -- so no two starts of the run share a seed; the
 * EXPLORE spec SEED + r; an NLopt phase without SEED its INIT block's
 * SEED, which NLopt takes, the same way), so a seeded stochastic optimiser
 * draws new random numbers instead of replaying round 0's; round 0 keeps
 * every SEED. An INNER block has no SEED: its search draws from the
 * enclosing run's random stream. The journal's "RESTART r/MAX" line lists
 * the round's seeds: the restart points' ("RESTART SEED", drawn or the
 * block's), then the derived optimiser seeds ("INIT SEED" for an INIT
 * block's).
 *
 * When the MAX restarts are used up, the next stagnation ends the run
 * normally, with the trailer "STOPPED -- stagnation: no new best in W
 * evals after MAX restart(s)" (and the failed-starts note of a MULTISTART
 * run whose starts failed). The watchdog is armed only while this policy
 * runs: however run() ends, the threshold is reset to 0, so a later run
 * in the same process is not stopped by a stale window.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class RestartPolicy : public RunPolicy<Vec,Runner> {
public:
  RestartPolicy(std::unique_ptr<RunPolicy<Vec,Runner>> inner,
                std::size_t window,
                std::size_t max_restarts,
                std::unique_ptr<InitStrategy> refresh,
                const HPOTargets* trg,
                std::size_t refresh_k = 0,
                std::optional<std::uint64_t> refresh_seed = std::nullopt)
      : inner_(std::move(inner)),
        window_(window),
        max_restarts_(max_restarts),
        refresh_(std::move(refresh)),
        trg_(trg),
        refresh_k_(refresh_k),
        refresh_seed_(refresh_seed) {}

  void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) override {
    OptimizerSpec cur = spec;
    std::size_t restarts = 0;

    /* Disarm the process-wide threshold on EVERY exit, including a
     * durable-best failure or a shutdown: a later run in this process
     * would otherwise start with a stale window. */
    struct ThresholdGuard {
      ~ThresholdGuard() { HPO_Host<Vec,Runner>::set_stagnation_threshold(0); }
    } disarm;
    HPO_Host<Vec,Runner>::set_stagnation_threshold(window_);
    HPO_Host<Vec,Runner>::reset_stagnation();

    while (true) {
      try {
        inner_->run(cur, host);
        break;
      } catch (const typename HPO_Host<Vec,Runner>::StagnationStop&) {
        /* A new best that could not be saved ends the run even when a
         * concurrent StagnationStop got here first. */
        detail::rethrow_if_best_unsaved();
        if (restarts >= max_restarts_) {
          const std::string msg =
              "RESTART: stagnation reached, max restarts (" +
              std::to_string(max_restarts_) + ") used; stopping";
          std::cout << "[HPO] " << msg << "\n";
          HPOLog::instance().info(msg);
          /* Say why the run ended: whatever an inner run recorded last (a
           * per-start MAXEVAL_REACHED, say) is not the reason. */
          HPOLog::instance().set_status(
              "STOPPED -- stagnation: no new best in " +
              std::to_string(window_) + " evals after " +
              std::to_string(max_restarts_) + " restart(s)" +
              detail::failed_starts_suffix(HPOLog::instance().start_tally()));
          break;
        }
        ++restarts;

        /* Refresh init points. Honour the RESTART block's K_TOTAL when
         * set; otherwise fall back to each strategy's default_k. Round r
         * draws its points with S + r: S the RESTART block's SEED, or,
         * without one, a seed std::random_device draws once per run (the
         * journal names it, so the run's restart points can be drawn
         * again), and its RANDOM/LHS methods a stream of their own
         * (InitStream::Restart). */
        std::optional<std::uint64_t> point_seed;
        if (refresh_ && trg_) {
          /* A chain of CONFIG/WARM only takes no seed: none is drawn. */
          std::uint64_t seed = 0;
          if (refresh_seed_ || refresh_->draws_from_seed()) {
            seed = restart_base_seed() + static_cast<std::uint64_t>(restarts);
            point_seed = seed;
          }
          InitContext ic{cur, *trg_, seed, &trg_->run_context(),
                         InitStream::Restart};
          const std::size_t k_use = (refresh_k_ > 0) ? refresh_k_
                                                     : refresh_->default_k();
          std::vector<InitOrigin> origins;
          auto pts = refresh_->generate_tagged(ic, k_use, origins);
          if (!pts.empty()) {
            cur.optim_init = std::move(pts);
            cur.optim_init_origin = std::move(origins);
            cur.init_from_restart = true;
            /* A PRESTAGE exploration that starts from the OUTER INIT
             * points restarts from the fresh ones too. */
            cur.share_init_with_nested_phases();
          } else {
            /* A chain can come back empty (its only WARM seed dropped
             * under WARM_ON_MISMATCH skip, RANDOM 0): the round then
             * starts where the previous one did. */
            const std::string msg =
                "RESTART " + std::to_string(restarts) + "/" +
                std::to_string(max_restarts_) + ": the METHOD chain (" +
                refresh_->name() + ") produced no starting point; the round "
                "starts from the previous round's starting points";
            std::cerr << "[HPO] WARNING: " << msg << "\n";
            HPOLog::instance().warning(msg);
          }
        }
        /* Seeded phases draw new random numbers this round. */
        cur.derive_restart_seeds(spec, restarts);
        HPO_Host<Vec,Runner>::reset_stagnation();
        {
          std::ostringstream oss;
          oss << "RESTART " << restarts << "/" << max_restarts_
              << " -- fresh init from "
              << (refresh_ ? refresh_->name() : std::string("(none)"));
          const std::string seeds = describe_seeds(point_seed, cur);
          if (!seeds.empty()) oss << "; " << seeds;
          std::cout << "[HPO] " << oss.str() << "\n";
          HPOLog::instance().info(oss.str());
        }
      }
    }
  }

private:
  /* S of the restart points' seeds S + r: the RESTART block's SEED, or the
   * run's draw from std::random_device (64 bits, two draws), made at the
   * first restart and journalled with the SEED that draws the same points
   * again. */
  std::uint64_t restart_base_seed()
  {
    if (refresh_seed_) return *refresh_seed_;
    if (!drawn_seed_) {
      std::random_device rd;
      const std::uint64_t hi = rd(), lo = rd();
      drawn_seed_ = (hi << 32) ^ lo;
      const std::string msg =
          "RESTART: the RESTART block has no SEED, so its restart points "
          "are drawn from std::random_device: restart r draws with " +
          std::to_string(*drawn_seed_) + " + r (SEED " +
          std::to_string(*drawn_seed_) +
          " in the RESTART block draws the same points again)";
      std::cout << "[HPO] " << msg << "\n";
      HPOLog::instance().info(msg);
    }
    return *drawn_seed_;
  }

  /* "RESTART SEED 43, SEED 6, EXPLORE SEED 1718": the seed the round's
   * points were drawn with, then the seeds of the phases that have one (an
   * INNER block has none) -- for an NLopt phase that takes its INIT block's
   * SEED (OptimizerSpec::nlopt_seed_from_init()), that SEED as derived for
   * the round: "INIT SEED 8", "EXPLORE INIT SEED 31". */
  static std::string describe_seeds(
      const std::optional<std::uint64_t>& point_seed, const OptimizerSpec& s)
  {
    std::ostringstream oss;
    const char* sep = "";
    const auto add = [&](const char* phase,
                         const std::optional<std::uint64_t>& seed) {
      if (!seed) return;
      oss << sep << phase << "SEED " << *seed;
      sep = ", ";
    };
    /* The INIT block's SEED that an NLopt phase takes is named when the
     * phase draws random numbers (reads_seed): LD_MMA reads none. */
    const auto add_phase = [&](const char* phase, const char* init_phase,
                               const OptimizerSpec& p) {
      if (p.nlopt_seed_from_init()) {
        if (reads_seed(p)) add(init_phase, p.init_strategy_spec.seed);
      } else {
        add(phase, p.seed);
      }
    };
    add("RESTART ", point_seed);
    add_phase("", "INIT ", s);
    if (s.prestage) add_phase("EXPLORE ", "EXPLORE INIT ", *s.prestage);
    return oss.str();
  }

  std::unique_ptr<RunPolicy<Vec,Runner>> inner_;
  std::size_t window_, max_restarts_;
  std::unique_ptr<InitStrategy> refresh_;
  const HPOTargets* trg_;
  std::size_t refresh_k_ = 0;   // 0 => use refresh_->default_k()
  std::optional<std::uint64_t> refresh_seed_;   // RESTART SEED; none => random_device
  std::optional<std::uint64_t> drawn_seed_;     // the run's random_device draw
};

/* ----------------------------------------------------------------------
 * build_run_policy -- assembles the RunPolicy decorator stack from the
 * parsed OptimizerSpec.
 *
 * Layering (innermost -> outermost):
 *    BasePolicy / MultiStartPolicy / StagedPolicy   <- chosen by STRATEGY TYPE
 *      +-- wrapped by RestartPolicy if RESTART sets both AFTER_STAGNATION
 *          and MAX > 0
 *
 * Selection rules (STRATEGY TYPE):
 *   - PRESTAGE   -> StagedPolicy (EXPLORE block, then top-K refinement).
 *   - MULTISTART -> MultiStartPolicy (N_STARTS starts of LIB/ALGO).
 *   - PLAIN      -> BasePolicy (single shot).
 *
 * targets is captured for RestartPolicy's refresh_strategy.
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
std::unique_ptr<RunPolicy<Vec,Runner>>
build_run_policy(const OptimizerSpec& spec, const HPOTargets* targets = nullptr)
{
  /* Strategy block dispatch -- TYPE chooses the inner policy. */
  std::unique_ptr<RunPolicy<Vec,Runner>> inner;
  switch (spec.strategy_type) {
    case StrategyType::PRESTAGE:
      inner = std::make_unique<StagedPolicy<Vec,Runner>>(spec.stage_topk);
      break;

    case StrategyType::MULTISTART:
      inner = std::make_unique<MultiStartPolicy<Vec,Runner>>(
          spec.multistart.n_starts, spec.multistart.budget_per_start);
      break;

    case StrategyType::PLAIN:
    default:
      inner = std::make_unique<BasePolicy<Vec,Runner>>();
      break;
  }

  /* RESTART decorator: wraps every STRATEGY TYPE; a restart re-runs the
   * whole inner policy (see RestartPolicy). */
  const auto& rs = spec.restart;
  if (rs.active()) {
    auto refresh = build_init_strategy(rs.init_strategy);
    const std::size_t refresh_k = rs.init_strategy.k_total.value_or(0);
    return std::make_unique<RestartPolicy<Vec,Runner>>(
        std::move(inner),
        rs.after_stagnation,
        rs.max,
        std::move(refresh),
        targets,
        refresh_k,
        rs.init_strategy.seed);
  }
  return inner;
}

}  // namespace tadah::hpo

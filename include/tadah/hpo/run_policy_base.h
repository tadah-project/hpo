#pragma once
/*
 * run_policy_base.h -- the RunPolicy<> interface and run_inner<>(), the
 * single-shot dispatch from an OptimizerSpec onto its optimiser driver.
 *
 * run_policy.h builds the decorator stack (MULTISTART, PRESTAGE,
 * RESTART) on top of this layer; everything here depends only on the
 * optimiser adapters, not on any policy.
 */

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/optimizer_spec.h>

namespace tadah::hpo {

namespace detail {
/* Backstop for a new best that could not be persisted (HPO_Host records it
 * before the durable_best::Error leaves the evaluation): whatever swallowed
 * the error on its way up -- a concurrent exception that won a pool's
 * stash, say -- the run must still fail, not restart or complete. Every
 * catch that swallows a StagnationStop or a failed start calls it
 * (run_policy.h), so a lost best stops the run at once rather than after
 * the rest of a stage or batch. */
inline void rethrow_if_best_unsaved()
{
  if (const auto ub = HPOLog::instance().unsaved_best())
    throw durable_best::Error(ub->what);
}

/* The cause hpo_run() reports a best-potential persistence failure with:
 * the first recorded failure (HPOLog::unsaved_best()->what, the one that
 * tripped HPO_Host's durable_best::FailLatch), else e's own message. Of
 * several failed candidates (parallel chains) a later one's error -- a
 * latched single attempt that failed too -- can reach hpo_run() first:
 * the evaluation pools' ExceptionStash (DLIB:GFS, ASA, the annealers'
 * ReplicaPool) keeps the first exception of the highest rank to ARRIVE,
 * and the tripping candidate submits its result before its exception
 * does. */
inline std::string durable_failure_cause(const durable_best::Error& e)
{
  if (const auto ub = HPOLog::instance().unsaved_best()) return ub->what;
  return e.what();
}
}  // namespace detail

/* ----------------------------------------------------------------------
 * run_inner -- single-shot dispatch from an OptimizerSpec onto its
 * concrete optimiser driver. Every starting point comes from
 * spec.optim_init (today a single vector, later a chain-resolved set).
 *
 * Adding a new optimiser is two changes:
 *   1. Implement a templated `run_<lib>_<algo><Vec, Runner>(spec, host)`
 *      function (or one entry point per LIB).
 *   2. Register a CapsEntry in optimizer_caps.cpp.
 * Then add a single dispatch case below. No other call sites change --
 * MultiStartPolicy / StagedPolicy / hpo_targets all derive their
 * behaviour from the caps registry rather than ad-hoc lib/algo strings.
 * -------------------------------------------------------------------- */
namespace detail {
template<class Vec, class Runner>
void run_inner_dispatch(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host);
}  // namespace detail

template<class Vec, class Runner>
inline void run_inner(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host)
{
  /* Every start point this run takes from spec.optim_init is evaluated at
   * exactly its REAL doubles (anchor_seed_points, search_space.h): the
   * run iterates on an anchored copy of the space, alive for this run
   * only -- nested policies call run_inner with their own seeds. */
  if (auto anchored = anchor_seed_points(spec)) {
    detail::run_inner_dispatch<Vec,Runner>(*anchored, host);
    return;
  }
  detail::run_inner_dispatch<Vec,Runner>(spec, host);
}

template<class Vec, class Runner>
inline void detail::run_inner_dispatch(const OptimizerSpec& spec,
                                       HPO_Host<Vec,Runner>& host)
{
  /* Early-reject deferred CERES NLLS-family algos before the caps
   * lookup so users see the tailored "use CERES:LBFGS" message rather
   * than a generic "unknown optimizer" error. These algos are
   * intentionally not registered in optimizer_caps because they are
   * degenerate over a scalar HPO loss; making them useful requires
   * residual-mode loss exposure (objective semantics change) and
   * autodiff -- both belong to a later phase. */
  reject_deferred_ceres_nlls(spec);

  /* Caps lookup throws if (lib, algo) is not registered, giving a
   * single, uniform "unknown optimizer" error instead of a fall-through
   * at the end of the dispatch chain. */
  (void)optimizer_caps(spec);

  /* Registry-driven STEP policy, re-checked here for specs assembled at
   * run time (MULTISTART and PRESTAGE clones, RESTART refreshes). Seeds
   * were validated at startup; run-time seeds come from the INIT chain or
   * the top-K. */
  validate_optimizer_scaling(spec, "run", /*check_start_points=*/false);
  if (spec.inner)
    validate_optimizer_scaling(*spec.inner, "run.INNER", false);

  /* The INNER rules of validate_hpo_setup (strategy_rules.h), again for a
   * spec built at run time. */
  require_inner_backend(spec, "");

  /* This run starts without a termination status: the journal trailer
   * of a run that ends on its own is the reason the optimiser run that
   * ended it recorded (HPOLog::set_status), or COMPLETED -- never a status
   * left behind by an earlier PRESTAGE phase, MULTISTART start or RESTART
   * round. */
  HPOLog::instance().clear_status();

  if (spec.lib == "TADAH") {
    if      (spec.algo == "RANDOM")    run_random   <Vec,Runner>(spec, host);
    else if (spec.algo == "GRID")      run_grid     <Vec,Runner>(spec, host);
    else if (spec.algo == "ANNEAL")    run_anneal   <Vec,Runner>(spec, host);
    else if (spec.algo == "ANNEAL_PT") run_anneal_pt<Vec,Runner>(spec, host);
    else if (spec.algo == "ANNEAL_PA") run_anneal_pa<Vec,Runner>(spec, host);
    else throw std::runtime_error("unknown TADAH algo " + spec.algo);
  } else if (spec.lib == "DLIB") {
    if      (spec.algo == "GFS")  run_dlib_global<Vec,Runner>(spec, host);
    else if (spec.algo == "BFGS") run_dlib_bfgs  <Vec,Runner>(spec, host);
    else throw std::runtime_error("unknown DLIB algo " + spec.algo);
  } else if (spec.lib == "NLOPT") {
    run_nlopt<Vec,Runner>(spec, host);
  } else if (spec.lib == "ASA") {
    if (spec.algo == "ASA") run_asa<Vec,Runner>(spec, host);
    else throw std::runtime_error("unknown ASA algo " + spec.algo +
                                  " (the ASA library has one algorithm: "
                                  "ALGO ASA)");
  } else if (spec.lib == "CERES") {
#ifndef TADAH_USE_CERES
    throw std::runtime_error(
        "CERES support is not built; reconfigure with -DTADAH_ENABLE_CERES=ON");
#else
    if      (spec.algo == "LBFGS")            run_ceres_lbfgs           <Vec,Runner>(spec, host);
    else if (spec.algo == "WOLFE")            run_ceres_wolfe           <Vec,Runner>(spec, host);
    else if (spec.algo == "NLCG")             run_ceres_nlcg            <Vec,Runner>(spec, host);
    else if (spec.algo == "STEEPEST_DESCENT") run_ceres_steepest_descent<Vec,Runner>(spec, host);
    else throw std::runtime_error("unknown CERES algo " + spec.algo);
#endif
  } else {
    throw std::runtime_error("unknown optimizer LIB " + spec.lib);
  }
}

/* ----------------------------------------------------------------------
 * RunPolicy -- interface
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
class RunPolicy {
public:
  virtual ~RunPolicy() = default;
  virtual void run(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host) = 0;

  /** Reserved for adapters that want to finalise the journal themselves.
   *  Currently always false: both NLopt and DLIB adapters merely record
   *  their per-run termination reason via HPOLog::set_status(); the
   *  outer hpo_run() always calls HPOLog::finish() once after the whole
   *  policy (BasePolicy / MultiStart / StageGlobal / Restart ...) ends. */
  static bool driver_finishes_log(const OptimizerSpec& /*spec*/) {
    return false;
  }
};

}  // namespace tadah::hpo

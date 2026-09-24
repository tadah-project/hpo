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
template<class Vec, class Runner>
inline void run_inner(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host)
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

#pragma once
/*
 * run_policy_base.h -- minimum include surface for RunPolicy<> and
 * run_inner<>().
 *
 * Why this header exists separately from run_policy.h
 * ---------------------------------------------------
 * run_policy.h's build_run_policy() needs to instantiate
 * BasinsPolicy<Vec,Runner> (defined in basins_policy.h). BasinsPolicy
 * in turn inherits from RunPolicy<Vec,Runner> and dispatches via
 * run_inner<Vec,Runner>. If basins_policy.h were to #include
 * run_policy.h, the resulting cycle (run_policy.h includes
 * basins_policy.h to call its constructor) would force partial-
 * include behaviour: when test_basins_policy.cpp includes
 * basins_policy.h first, run_policy.h's body gets parsed *while*
 * BasinsPolicy is still mid-declaration, which the compiler rejects
 * at build_run_policy()'s use of BasinsPolicy.
 *
 * Splitting the abstract base + run_inner into this small header
 * lets basins_policy.h depend on only the bits it needs (RunPolicy
 * and run_inner), so the cycle is broken.
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
 * BasinsPolicy / MultiStartPolicy / hpo_targets all derive their
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
  const auto& caps = optimizer_caps(spec);

  /* Registry-driven STEP policy, re-checked here for specs assembled at
   * run time (BASINS stage specs, MULTISTART clones). Seeds were validated
   * at startup; run-time seeds come from the pool / top-K. */
  validate_optimizer_scaling(spec, "run", /*check_start_points=*/false);
  if (spec.inner)
    validate_optimizer_scaling(*spec.inner, "run.INNER", false);

  if (spec.inner && !caps.accepts_inner) {
    throw std::runtime_error(
        spec.lib + ":" + spec.algo +
        " does not accept an INNER sub-optimiser; remove the INNER block "
        "or pick an algorithm that delegates a local search (NLopt MLSL/"
        "AUGLAG family)");
  }

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

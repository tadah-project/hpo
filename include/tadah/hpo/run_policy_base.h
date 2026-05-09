#pragma once
/*
 * run_policy_base.h — minimum include surface for RunPolicy<> and
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
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>

namespace tadah::hpo {

/* ----------------------------------------------------------------------
 * run_inner — single-shot dispatch from an OptimizerSpec onto its
 * concrete optimiser driver. Every starting point comes from
 * spec.optim_init (today a single vector, later a chain-resolved set).
 * -------------------------------------------------------------------- */
template<class Vec, class Runner>
inline void run_inner(const OptimizerSpec& spec, HPO_Host<Vec,Runner>& host)
{
  if (spec.lib == "DLIB" && spec.inner) {
    throw std::runtime_error(
        "DLIB optimisers do not support INNER blocks; choose a standalone "
        "ALGO such as BFGS or GFS");
  }

  if (spec.lib == "TADAH" && spec.algo == "RANDOM") {
    run_random<Vec,Runner>(spec, host);
  } else if (spec.lib == "TADAH" && spec.algo == "GRID") {
    run_grid<Vec,Runner>(spec, host);
  } else if (spec.lib == "TADAH" && spec.algo == "ANNEAL") {
    run_anneal<Vec,Runner>(spec, host);
  } else if (spec.lib == "DLIB" && spec.algo == "GFS") {
    run_dlib_global<Vec,Runner>(spec, host);
  } else if (spec.lib == "DLIB" && spec.algo == "BFGS") {
    run_dlib_bfgs<Vec,Runner>(spec, host);
  } else if (spec.lib == "NLOPT") {
    run_nlopt<Vec,Runner>(spec, host);
  } else {
    throw std::runtime_error("unknown optimizer " + spec.lib + ":" + spec.algo);
  }
}

/* ----------------------------------------------------------------------
 * RunPolicy — interface
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
   *  policy (BasePolicy / MultiStart / StageGlobal / Restart …) ends. */
  static bool driver_finishes_log(const OptimizerSpec& /*spec*/) {
    return false;
  }
};

}  // namespace tadah::hpo

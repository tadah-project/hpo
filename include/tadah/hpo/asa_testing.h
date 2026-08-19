#pragma once
/* asa_testing.h -- test seam for the golden-neutrality gate of the
 * vendored ASA library (see external/asa/TADAH-PATCHES.md).
 *
 * run_raw_featured() drives the FEATURED asa() build directly (no HPO
 * host, no budget/stop machinery) at the TADAH default options, over a
 * plain-C cost/RNG callback pair. tests/test_asa_golden.cpp runs the
 * same callbacks through the pristine (all-gates-off) build and asserts
 * bitwise-identical trajectories. Not part of the public API. */

#include <cstddef>
#include <vector>

namespace tadah::hpo::asa_testing {

/* Reduced plain-C callback shapes shared with the pristine test TU
 * (tests/asa_golden_common.h). The cost callback must set *cost_flag
 * to 1 (valid) or 0 (invalid) and may record its own trajectory. */
using RawCost = double (*)(const double* x, long dim, int* cost_flag);
using RawRng  = double (*)();

struct RawResult {
    int    exit_status = 0;
    double final_cost  = 0.0;
    long   n_accepted  = 0;
    long   n_generated = 0;
    std::vector<double> best_x;
};

/* Run the featured asa() at canonical-default options (the AsaOptions
 * defaults: every compiled-in gate at its neutral value) on dim
 * variables in [lo, hi] starting from x0, stopping via
 * Limit_Generated = limit_generated. */
RawResult run_raw_featured(RawCost cost, RawRng rng, std::size_t dim,
                           const double* lo, const double* hi,
                           const double* x0, long limit_generated);

}  // namespace tadah::hpo::asa_testing

#pragma once
/* algorithms_asa.h -- LIB ASA / ALGO ASA driver entry points.
 *
 * Adaptive Simulated Annealing (Lester Ingber, ASA-30.53), vendored
 * under external/asa (BSD-3-Clause; see external/asa/TADAH-PATCHES.md).
 * This installed header is deliberately free of ASA includes: the only
 * translation unit that touches ASA headers is src/algorithms_asa.cpp,
 * reached through the type-erased entry point below.
 *
 * Concurrency contract: asa() is not re-entrant (upstream statics), so
 * run_asa_erased serializes every call behind a process-wide mutex.
 * Within one run, PARAM Gener_Block > 1 evaluates each generated block
 * concurrently on an anneal::ReplicaPool sized by min(THREADS,
 * Gener_Block_Max); each evaluation's pc_lammps fan-out lands on the
 * shared ScriptExecutor as usual. Results are THREADS-invariant (ASA
 * sorts each block by cost before consuming it sequentially), and so are
 * the evaluation numbers and the bests: a block is a batch
 * (eval_batch.h), numbered in block order and committed after it. */

#include <tadah/hpo/optimizer_spec.h>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace tadah::hpo {

template<class Vec, class Runner> class HPO_Host;

/* Transformed-space objective: x (size dim) in optimiser-internal
 * coordinates, returns the loss. Must be thread-safe when PARAM
 * Gener_Block > 1. */
using AsaHostFn = std::function<double(const std::vector<double>&)>;

/* Run ASA on a type-erased objective. Implemented in
 * src/algorithms_asa.cpp (the only ASA-including TU). */
void run_asa_erased(const OptimizerSpec& sp, std::size_t dim,
                    const AsaHostFn& eval);

/* "" when ASA's finite-difference probes (external/asa cost_derivatives:
 * the reannealing and exit tangents at (1 + Delta) x, or (1 - Delta) x
 * when that lies outside the box, and with Curvature_0 0 or -1 the exit
 * curvature at x +- Delta |x|, or x -+ Delta |x| and x -+ 2 Delta |x|
 * next to a bound) fit every OPTIM box of `sp` (its internal bounds) with
 * each dimension's Delta (PARAM Delta_X or User_Delta_Parameter), else the
 * error: each dimension where a probe would lie outside the box, with the
 * largest Delta that fits (asa_probe_fit, fd_stencil.h). A dimension ASA
 * does not probe is skipped: a Delta below ASA's SMALL_FLOAT (0 disables
 * the probes), an integer dimension without Include_Integer_Parameters, a
 * range below ASA's EPS_DOUBLE. validate_hpo_setup refuses such a box at
 * startup, and run_asa_erased before it starts. Implemented in
 * src/algorithms_asa.cpp (ASA's constants). Throws what parse_asa_options
 * throws. */
std::string asa_probe_box_error(const OptimizerSpec& sp);

/* Generic-host adapter (unit tests drive this with a stub host). */
template<typename Vec, typename HostT>
void run_asa_impl(const OptimizerSpec& sp, HostT& host)
{
    const std::size_t dim = sp.low.size();
    const AsaHostFn eval = [&sp, &host, dim](const std::vector<double>& xt) {
        Vec v(dim);
        for (std::size_t i = 0; i < dim; ++i) v(i) = xt[i];
        return host(sp.out(v));
    };
    run_asa_erased(sp, dim, eval);
}

/* Production entry point used by run_inner(). */
template<class Vec, class Runner>
void run_asa(const OptimizerSpec& sp, HPO_Host<Vec, Runner>& host)
{
    run_asa_impl<Vec>(sp, host);
}

}  // namespace tadah::hpo

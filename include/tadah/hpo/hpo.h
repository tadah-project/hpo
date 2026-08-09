#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include <tadah/core/context.h>

namespace tadah {
namespace hpo {

struct OptimizerSpec;

int hpo_run(core::Context&);

/* NOISE_CALIBRATION gate: only run on a fresh start with a non-trivial
 * calibration knob. On resume, the prior run already calibrated noise
 * and re-running would charge N extra LAMMPS evals against the resumed
 * counter and pollute the per-eval logs. Defined in hpo.cpp; declared
 * here so unit tests can exercise the predicate without invoking the
 * full optimiser. */
bool should_run_noise_calibration(bool resumed, std::size_t noise_n);

/* The TADAH_HPO_MPI communicator is intentionally owned only by the plain
 * parallel-annealing drivers. A size-one run (including an MPI-enabled binary
 * launched without mpirun) retains the complete serial optimiser surface. */
bool hpo_mpi_dispatch_supported(int world_size,
                                const OptimizerSpec& spec) noexcept;

} // namespace hpo
} // namespace tadah

#endif

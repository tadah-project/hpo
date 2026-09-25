#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include <tadah/core/context.h>

#include <string>

namespace tadah {
namespace hpo {

struct OptimizerSpec;

/* Runs one HPO job in the current directory. Returns the process exit
 * status: 0 when the optimiser ends on its own (stop criteria met) and an
 * evaluation, on any MPI rank, produced a usable loss; EXIT_FAILURE (1)
 * when it ends on its own without one (no best_pot.tadah; trailer
 * "STOPPED -- no evaluation produced a usable loss"); and 128 + the signal
 * number when SIGINT, SIGTERM or SIGUSR2 stopped it (130, 143, 140 on
 * Linux), with or without a usable loss. The result files are flushed and
 * the journal trailer written in every case. An error is thrown after the
 * trailer ("STOPPED with error: ..."). See run_driver.h. */
int hpo_run(core::Context&);

/* NOISE_CALIBRATION gate: run only for a non-trivial calibration knob
 * (N >= 2 evaluations). Defined in hpo.cpp; declared here so unit tests
 * can exercise the predicate without invoking the full optimiser. */
bool should_run_noise_calibration(std::size_t noise_n);

/* Whether a launch on `world_size` ranks can run `spec`: the reason it
 * cannot, or an empty string when it can. hpo_run() raises the reason as
 * its error before any evaluation.
 *
 * The TADAH_HPO_MPI communicator is intentionally owned only by the plain
 * parallel-annealing drivers (STRATEGY TYPE PLAIN, TADAH:ANNEAL_PT/PA),
 * and a multi-rank run refuses an active RESTART: the stagnation watchdog
 * is per rank, so a rank that stagnated would restart and wait in the
 * driver's next collective while its peers fail. A size-one run
 * (including an MPI-enabled binary launched without mpirun) retains the
 * complete serial optimiser surface, RESTART included. */
std::string hpo_mpi_dispatch_refusal(int world_size, const OptimizerSpec& spec);

/* hpo_mpi_dispatch_refusal(world_size, spec).empty(). */
bool hpo_mpi_dispatch_supported(int world_size, const OptimizerSpec& spec);

} // namespace hpo
} // namespace tadah

#endif

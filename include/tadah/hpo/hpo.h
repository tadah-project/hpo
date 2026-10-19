#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include <tadah/core/context.h>

#include <string>

namespace tadah {
namespace mlip {
class KeyProvenance;
}
namespace hpo {

struct OptimizerSpec;

/* Runs one HPO job in the current directory. Returns the process exit
 * status: 0 when the optimiser ends on its own (stop criteria met, or a
 * gradient-based search ended at a point without a usable gradient,
 * trailer "STOPPED -- no usable gradient: ...") and an evaluation, on any MPI
 * rank, produced a usable loss; EXIT_FAILURE (1)
 * when it ends on its own without one (no best_pot.tadah; trailer
 * "STOPPED -- no evaluation produced a usable loss"); and 128 + the signal
 * number when SIGINT, SIGTERM or SIGUSR2 stopped it (130, 143, 140 on
 * Linux), with or without a usable loss. The result files are flushed and
 * the journal trailer written in every case. An error is thrown after the
 * trailer: "FAILED -- best-potential persistence: <cause>" when a best
 * potential could not be saved (durable_best::Error, however the run then
 * ended; the fresh-run reset of best_pot.tadah included), "FAILED --
 * <what>" for another std::exception and "FAILED -- non-standard
 * exception" for any other; the caller exits with 1. A startup error
 * (the configuration, validate_hpo_setup()) is thrown before any trailer,
 * after an "ERROR startup: ..." journal line. See run_driver.h.
 *
 * With CHECK true in the context (`tadah hpo --check`, CHECK true in a
 * TASK hpo section) it makes the startup checks of the run alone, in the
 * run's order -- the working directory, which must take the journal, the
 * result files and best_pot.tadah, the training configuration, the target
 * file (its DUMP and BDUMP directories included) and its LAMMPS scripts,
 * the result files that exist, validate_hpo_setup(), PC_ISOLATION and
 * THREADS, the training and
 * validation datasets (loaded as the run loads them), HEALTH_LOG, the
 * pre-flight audit and the STRATEGY TYPE and RESTART settings -- then
 * writes the summary of write_check_summary() (check_summary.h) to
 * standard output, with the WARNING and HINT lines the run writes to its
 * journal at startup, and returns 0. It evaluates nothing, installs no
 * signal handler, starts no worker and writes no file: no journal (one an
 * earlier run in the process left open is closed:
 * HPOLog::begin_without_journal), no search-space sidecar, no result file,
 * no DUMP or BDUMP directory. A startup error is thrown as the run throws
 * it, after those WARNING and HINT lines on standard error. CHECK is taken
 * out of the context (the run's too, CHECK false); a potential has no
 * CHECK line (Tadah.MLIP's training_record_keys()) and no parse stores a
 * potential's (Tadah.CORE's run_setting_keys()). A check starts no
 * PC_ISOLATION PROCESS worker and takes no lock of best_pot.tadah: the run
 * does both after the check's last step, and stops if either fails.
 *
 * A run creates its DUMP and BDUMP directories, the one inside the other's
 * after it, once it has prepared best_pot.tadah (its lock taken), just
 * before its first evaluation: a run that stops before creates none. */
int hpo_run(core::Context&);
/* The same, with where the context's keys came from (the command line, the
 * configuration file, the defaults): the journal's POTENTIAL SETUP names
 * the source of FORCE and STRESS. hpo_run(ctx) records "default" for a key
 * the context does not hold and "source not recorded" for one it does. */
int hpo_run(core::Context&, const mlip::KeyProvenance* provenance);

/* NOISE_CALIBRATION gate: run only for a non-trivial calibration knob
 * (N >= 2 evaluations). Defined in hpo.cpp; declared here so unit tests
 * can exercise the predicate without invoking the full optimiser. */
bool should_run_noise_calibration(std::size_t noise_n);

/* Whether a launch on `world_size` ranks can run `spec`: the reason it
 * cannot, or an empty string when it can. hpo_run() raises the reason as
 * its error before any evaluation. For a parsed spec the reason names the
 * line its OPTIMIZER block closes at, "Error on line N: multi-rank HPO
 * ...\nLine: ENDOPTIMIZER" (with_block_line, phase_specs.h).
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

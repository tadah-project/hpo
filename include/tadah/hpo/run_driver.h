#pragma once
/*
 * run_driver.h -- how an HPO run ends.
 *
 * drive_run() runs the part of hpo_run() that evaluates candidates (the
 * NOISE_CALIBRATION repeats and the RunPolicy) and owns every way it can
 * end. Whatever the ending, the result queue is drained into the history
 * files (HPO_Host::flush_remaining) and the journal trailer is written
 * exactly once, with a status that says why the run ended:
 *
 *   normal end              flush; trailer = the status recorded by the
 *                           optimiser run that ended the run
 *                           (HPOLog::set_status; each run starts without
 *                           one: a stop criterion, "STOPPED -- gradient
 *                           not finite: ..." from a gradient-based search,
 *                           ...) or COMPLETED; returns 0
 *   core::ShutdownRequested flush; "SHUTDOWN (<signal>)"; returns
 *                           shutdown_exit_status(<signal>) = 128 + signal
 *   StagnationStop          flush; "STOPPED -- stagnation ..."; returns 0.
 *                           RestartPolicy is the only consumer of the
 *                           stagnation stop (the optimiser adapters let it
 *                           propagate); reaching the driver means no RESTART
 *                           was there to act on it, which is an ordinary end
 *   std::exception          flush; "STOPPED with error: <what>"; rethrows
 *   anything else           flush; "STOPPED with unknown error"; rethrows
 *
 * An ordinary end (a normal end or a StagnationStop) after which no
 * evaluation of the run, on any MPI rank, produced a usable loss left no
 * best_pot.tadah: it is a failed run, not a completed one. It is reported
 * with an ERROR line in the journal and on stderr ("no evaluation produced
 * a usable loss; no best_pot.tadah was written") and the trailer "STOPPED
 * -- no evaluation produced a usable loss", and returns EXIT_FAILURE (1).
 * A signal or an error decides the ending first, whether or not a usable
 * loss was produced.
 *
 * Whether any rank produced a usable loss is one collective
 * (detail::run_has_usable_loss), and every ending reaches it exactly once,
 * first -- before the final flush and anything else rank-local that can
 * fail. A rank whose flush failed, or whose run an error or a signal ended
 * after the drivers' last collective, would otherwise leave the others
 * waiting in it for ever (the codebase's rule for collectives, as in
 * HPO_Host::prepare_best_publication). Each rank still ends its own way.
 *
 * best_pot.tadah needs none of this: HPO_Host publishes every new best
 * synchronously and durably (fsync + rename) before the evaluation returns,
 * so it holds the incumbent even after SIGKILL.
 */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string>

namespace tadah::hpo {

/* Exit status of a run stopped by a watched signal: 128 + the signal number,
 * the shell convention for "ended by signal N" (130 SIGINT, 143 SIGTERM, 140
 * SIGUSR2 on Linux). `signo` 0 means a ShutdownRequested was raised without a
 * recorded signal -- only code can do that, never the handler -- and the run
 * did not complete, so it is EXIT_FAILURE rather than a made-up signal. */
inline int shutdown_exit_status(int signo) noexcept
{
    return signo > 0 ? 128 + signo : EXIT_FAILURE;
}

/* The journal trailer and the error of a run that ended without a usable
 * loss (see above). */
inline const std::string kNoUsableLossStatus =
    "STOPPED -- no evaluation produced a usable loss";
inline const std::string kNoUsableLossError =
    "no evaluation produced a usable loss; no best_pot.tadah was written";

namespace detail {

/* Whether an evaluation of the run, on any rank, produced a usable loss: a
 * collective over the ranks of a multi-rank run (plain ANNEAL_PT /
 * ANNEAL_PA, the only drivers allowed on several ranks). A rank without a
 * usable loss of its own may still share the global best_pot.tadah another
 * rank published. drive_run() calls it exactly once on every ending. */
template<class Host>
bool run_has_usable_loss(const Host& host)
{
    return hpo_allreduce_max(host.has_usable_loss() ? 1 : 0) != 0;
}

/* Ends a run that produced no usable loss: an ERROR line in the journal and
 * on stderr, the trailer, EXIT_FAILURE. */
inline int end_without_usable_loss(HPOLog& log)
{
    log.event("ERROR", kNoUsableLossError);
    std::cerr << "[HPO] ERROR: " << kNoUsableLossError << std::endl;
    log.finish(kNoUsableLossStatus);
    return EXIT_FAILURE;
}

}  // namespace detail

/* `Host` provides `void flush_remaining() const`, `bool has_usable_loss()
 * const` and the exception type `Host::StagnationStop` (HPO_Host<Vec,
 * Runner>). Returns the exit status of the run (see above); errors are
 * rethrown after the trailer. */
template<class Host>
int drive_run(const std::function<void()>& body, const Host& host,
              HPOLog& log)
{
    /* Drain in a failure path without replacing the exception in flight;
     * a failed drain is reported in the journal. */
    auto drain = [&]() {
        try {
            host.flush_remaining();
        } catch (const std::exception& e) {
            log.warning(std::string("final flush of the result files "
                                    "failed: ") + e.what());
        } catch (...) {
            log.warning("final flush of the result files failed: unknown "
                        "exception");
        }
    };

    /* The run's one collective (see above), reached exactly once whatever
     * the ending; a signal or an error decides its ending without the
     * answer. */
    bool agreed = false;
    bool usable_anywhere = false;
    auto agree_on_usable_loss = [&]() {
        if (!agreed) {
            agreed = true;
            usable_anywhere = detail::run_has_usable_loss(host);
        }
        return usable_anywhere;
    };

    try {
        body();
        agree_on_usable_loss();
        /* Inside the try: a failing drain on the normal path is an error
         * end, reported like any other. */
        host.flush_remaining();
    } catch (const core::ShutdownRequested&) {
        agree_on_usable_loss();
        const std::string why = core::SignalHandler::reason();
        log.warning("Shutdown requested: " + why);
        drain();
        log.finish("SHUTDOWN (" + why + ")");
        return shutdown_exit_status(core::SignalHandler::signal_seen());
    } catch (const typename Host::StagnationStop&) {
        const bool usable = agree_on_usable_loss();
        log.warning("stagnation stop reached the run driver: no RESTART "
                    "policy acted on it, so the run ends here");
        drain();
        if (!usable)
            return detail::end_without_usable_loss(log);
        log.finish("STOPPED -- stagnation (no new best within the "
                   "stagnation window)");
        return 0;
    } catch (const std::exception& e) {
        agree_on_usable_loss();
        log.warning(std::string("Optimiser threw: ") + e.what());
        drain();
        log.finish(std::string("STOPPED with error: ") + e.what());
        throw;
    } catch (...) {
        agree_on_usable_loss();
        log.warning("Optimiser threw a non-standard exception");
        drain();
        log.finish("STOPPED with unknown error");
        throw;
    }

    if (!usable_anywhere)
        return detail::end_without_usable_loss(log);

    /* Empty status: HPOLog::finish() uses the status recorded by the
     * optimiser run that ended the run (e.g. NLopt's MAXEVAL_REACHED; under
     * MULTISTART the last start's, under PRESTAGE the refinement's), the
     * RESTART one when its MAX restarts were used up, or COMPLETED. */
    log.finish();
    return 0;
}

}  // namespace tadah::hpo

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
 *                           one) or COMPLETED; returns 0
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
 * best_pot.tadah needs none of this: HPO_Host publishes every new best
 * synchronously and durably (fsync + rename) before the evaluation returns,
 * so it holds the incumbent even after SIGKILL.
 */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/hpo_log.h>

#include <cstdlib>
#include <exception>
#include <functional>
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

/* `Host` provides `void flush_remaining() const` and the exception type
 * `Host::StagnationStop` (HPO_Host<Vec,Runner>). Returns the exit status
 * of the run (see above); errors are rethrown after the trailer. */
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

    try {
        body();
        /* Inside the try: a failing drain on the normal path is an error
         * end, reported like any other. */
        host.flush_remaining();
    } catch (const core::ShutdownRequested&) {
        const std::string why = core::SignalHandler::reason();
        log.warning("Shutdown requested: " + why);
        drain();
        log.finish("SHUTDOWN (" + why + ")");
        return shutdown_exit_status(core::SignalHandler::signal_seen());
    } catch (const typename Host::StagnationStop&) {
        log.warning("stagnation stop reached the run driver: no RESTART "
                    "policy acted on it, so the run ends here");
        drain();
        log.finish("STOPPED -- stagnation (no new best within the "
                   "stagnation window)");
        return 0;
    } catch (const std::exception& e) {
        log.warning(std::string("Optimiser threw: ") + e.what());
        drain();
        log.finish(std::string("STOPPED with error: ") + e.what());
        throw;
    } catch (...) {
        log.warning("Optimiser threw a non-standard exception");
        drain();
        log.finish("STOPPED with unknown error");
        throw;
    }

    /* Empty status: HPOLog::finish() uses the status recorded by the
     * optimiser run that ended the run (e.g. NLopt's MAXEVAL_REACHED; under
     * MULTISTART the last start's, under PRESTAGE the refinement's), the
     * RESTART one when its MAX restarts were used up, or COMPLETED. */
    log.finish();
    return 0;
}

}  // namespace tadah::hpo

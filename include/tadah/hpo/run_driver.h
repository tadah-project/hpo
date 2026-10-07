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
 *                           one: why it ended, "<LIB>:<ALGO>: MAXEVAL
 *                           reached" (run_end.h), NLopt's MAXEVAL_REACHED,
 *                           "STOPPED -- no usable gradient: ..." from a
 *                           gradient-based search, RESTART's "STOPPED --
 *                           stagnation: ...", ...); returns 0. A normal
 *                           end without a recorded status is an internal
 *                           error (a driver that did not say why it
 *                           ended): std::logic_error, reported as below
 *   core::ShutdownRequested flush; "SHUTDOWN (<signal>)"; returns
 *                           shutdown_exit_status(<signal>) = 128 + signal.
 *                           A rank of a multi-rank run that did not receive
 *                           the signal itself and stops on another rank's
 *                           (PeerShutdown, run_stop.h): "SHUTDOWN (<signal>
 *                           on MPI rank <r>)", 128 + that signal
 *   StagnationStop          flush; "STOPPED -- stagnation ..."; returns 0.
 *                           RestartPolicy is the only consumer of the
 *                           stagnation stop (the optimiser adapters let it
 *                           propagate); reaching the driver means no RESTART
 *                           was there to act on it, which is an ordinary end
 *   durable_best::Error     flush without backoff; "FAILED -- best-potential
 *                           persistence: <cause>"; rethrows
 *   std::exception          flush; "FAILED -- <what>"; rethrows
 *   anything else           flush; "FAILED -- non-standard exception";
 *                           rethrows
 * A rethrown error reaches the caller (tadah hpo exits with status 1).
 *
 * Best-potential persistence is one rule. HPO_Host publishes every new best
 * synchronously and durably (fsync + rename) before the evaluation returns,
 * retrying transient I/O errors (durable_best.h, io_retry.h), so
 * best_pot.tadah holds the incumbent even after SIGKILL. A new best that
 * still cannot be saved is recorded (HPOLog::record_unsaved_best) and throws
 * durable_best::Error, which every policy lets through. The record is the
 * backstop: wherever the error was outranked or swallowed on its way up --
 * a concurrent stop that won an evaluation pool's ExceptionStash, a start
 * or stage that ended on stagnation -- the policy that swallowed it calls
 * detail::rethrow_if_best_unsaved(), and so does drive_run() after a normal
 * end, a signal stop or a stagnation stop. A run whose best could not be
 * saved therefore always ends FAILED, never with an optimiser's status,
 * SHUTDOWN or a restart, and its cause is the first recorded failure
 * (detail::durable_failure_cause).
 *
 * An ordinary end (a normal end or a StagnationStop) after which no
 * evaluation of the run, on any MPI rank, produced a usable loss left no
 * best_pot.tadah: it is a failed run, not a completed one. It is reported
 * with an ERROR line in the journal and on stderr ("no evaluation produced
 * a usable loss; no best_pot.tadah was written") and the trailer "STOPPED
 * -- no evaluation produced a usable loss", and returns EXIT_FAILURE (1).
 * A signal or an error decides the ending first, whether or not a usable
 * loss was produced. So does a normal end without a recorded status (an
 * internal error, FAILED); a best potential that could not be saved
 * outranks it (the test comes right after detail::rethrow_if_best_unsaved).
 *
 * Whether any rank produced a usable loss is one collective
 * (detail::run_has_usable_loss), and every ending reaches it exactly once,
 * first -- before the final flush and anything else rank-local that can
 * fail. A rank whose flush failed, or whose run an error or a signal ended
 * after the drivers' last collective, would otherwise leave the others
 * waiting in it for ever (the codebase's rule for collectives, as in
 * HPO_Host::prepare_best_publication). Each rank still ends its own way.
 */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/run_stop.h>          // PeerShutdown, signal_name

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
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

/* The error of a normal end without a recorded status (see above). */
inline const std::string kNoEndStatusError =
    "internal error: the optimiser ended without recording why it ended";

namespace detail {

/* The backstop of the persistence rule (see above): throws the recorded
 * durable_best::Error when a new best of this run could not be saved
 * (HPO_Host records it before the error leaves the evaluation), whatever
 * swallowed the error on its way up. Every catch that swallows a
 * StagnationStop or a failed start calls it (run_policy.h), so a lost best
 * stops the run at once rather than after the rest of a stage or batch. */
inline void rethrow_if_best_unsaved()
{
  if (const auto ub = HPOLog::instance().unsaved_best())
    throw durable_best::Error(ub->what);
}

/* The cause drive_run() reports a best-potential persistence failure
 * with: the first recorded failure (HPOLog::unsaved_best()->what, the one
 * that tripped HPO_Host's durable_best::FailLatch), else e's own message.
 * Of several failed candidates (parallel chains) a later one's error -- a
 * latched single attempt that failed too -- can reach drive_run() first:
 * the evaluation pools' ExceptionStash (DLIB:GFS, ASA, the annealers'
 * ReplicaPool) keeps the first exception of the highest rank to ARRIVE,
 * and the tripping candidate submits its result before its exception
 * does. */
inline std::string durable_failure_cause(const durable_best::Error& e)
{
  if (const auto ub = HPOLog::instance().unsaved_best()) return ub->what;
  return e.what();
}

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

/* `Host` provides `void flush_remaining(bool retry_io) const`, `bool
 * has_usable_loss() const` and the exception type `Host::StagnationStop`
 * (HPO_Host<Vec, Runner>). Returns the exit status of the run (see above);
 * errors are rethrown after the trailer. */
template<class Host>
int drive_run(const std::function<void()>& body, const Host& host,
              HPOLog& log)
{
    /* Drain in a failure path without replacing the exception in flight;
     * a failed drain is reported in the journal. retry_io=false after a
     * best-potential persistence failure: the storage has just failed for
     * up to the whole durable budget, so the queued rows get no backoff (a
     * permanent errno still gets its one immediate re-attempt). */
    auto drain = [&](bool retry_io) {
        try {
            host.flush_remaining(retry_io);
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

    /* The best potential could not be saved even after retrying transient
     * errors (durable_best.h): the run cannot promise its result, so it
     * fails loudly. */
    auto fail_persistence = [&](const std::string& cause) {
        log.warning("Best-potential persistence failed: " + cause);
        log.record_best_persistence_failure();
        drain(/*retry_io=*/false);
        log.finish("FAILED -- best-potential persistence: " + cause);
    };

    try {
        body();
        /* A new best that could not be saved fails the run even when the
         * durable_best::Error was swallowed on its way up. */
        detail::rethrow_if_best_unsaved();
        /* Every optimiser records why it ended (run_end.h): a normal end
         * without a status is a driver that did not, an internal error.
         * The std::exception handler below reaches the collective. */
        if (log.status().empty()) throw std::logic_error(kNoEndStatusError);
        agree_on_usable_loss();
        /* Inside the try: a failing drain on the normal path is an error
         * end, reported like any other. */
        host.flush_remaining(/*retry_io=*/true);
    } catch (const core::ShutdownRequested& stop) {
        agree_on_usable_loss();
        /* A new best that could not be saved outranks the shutdown: the
         * run did not keep its result. */
        if (const auto ub = log.unsaved_best()) {
            fail_persistence(ub->what);
            throw durable_best::Error(ub->what);
        }
        /* After the stop request the drain's retry budget is the 2 s stop
         * grace for all files together: a blip still keeps its rows, and a
         * dead mount cannot hold the exit past a scheduler's kill
         * timeout. */
        /* The rank's own signal when it received one; otherwise the one
         * the ranks' agreement carried from the rank that did
         * (PeerShutdown, run_stop.h). */
        const auto* peer = dynamic_cast<const PeerShutdown*>(&stop);
        const int own = core::SignalHandler::signal_seen();
        const bool from_peer = own == 0 && peer != nullptr;
        const int signo = from_peer ? peer->signo : own;
        const std::string why = from_peer
            ? signal_name(peer->signo) + " on MPI rank " +
                  std::to_string(peer->origin_rank)
            : core::SignalHandler::reason();
        log.warning("Shutdown requested: " + why);
        drain(/*retry_io=*/true);
        log.finish("SHUTDOWN (" + why + ")");
        return shutdown_exit_status(signo);
    } catch (const typename Host::StagnationStop&) {
        const bool usable = agree_on_usable_loss();
        if (const auto ub = log.unsaved_best()) {
            fail_persistence(ub->what);
            throw durable_best::Error(ub->what);
        }
        log.warning("stagnation stop reached the run driver: no RESTART "
                    "policy acted on it, so the run ends here");
        drain(/*retry_io=*/true);
        if (!usable)
            return detail::end_without_usable_loss(log);
        log.finish("STOPPED -- stagnation (no new best within the "
                   "stagnation window)");
        return 0;
    } catch (const durable_best::Error& e) {
        agree_on_usable_loss();
        /* Of several failed candidates (parallel chains) the one that
         * failed first is the cause, even when a later one's error reaches
         * this handler first (detail::durable_failure_cause). */
        const std::string cause = detail::durable_failure_cause(e);
        fail_persistence(cause);
        if (cause != e.what()) throw durable_best::Error(cause);
        throw;
    } catch (const std::exception& e) {
        agree_on_usable_loss();
        log.warning(std::string("Optimiser threw: ") + e.what());
        drain(/*retry_io=*/true);
        log.finish(std::string("FAILED -- ") + e.what());
        throw;
    } catch (...) {
        agree_on_usable_loss();
        log.warning("Optimiser threw a non-standard exception");
        drain(/*retry_io=*/true);
        log.finish("FAILED -- non-standard exception");
        throw;
    }

    if (!usable_anywhere)
        return detail::end_without_usable_loss(log);

    /* The status recorded by the optimiser run that ended the run (e.g.
     * NLopt's MAXEVAL_REACHED, "DLIB:BFGS: FTOL_ABS reached"; under
     * MULTISTART the last good start's, under PRESTAGE the refinement's),
     * or the RESTART one when its MAX restarts were used up. */
    log.finish(log.status());
    return 0;
}

}  // namespace tadah::hpo

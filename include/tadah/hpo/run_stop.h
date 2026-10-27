#pragma once
/*
 * run_stop.h -- the run-level stops, and which exception ends a run when
 * several evaluations throw at once.
 *
 * Two exceptions end a run on purpose rather than because something
 * failed:
 *   core::ShutdownRequested  a watched signal (SIGINT/SIGTERM/SIGUSR2);
 *                            drive_run() ends the run with 128 + signal
 *   StagnationStop           the RESTART watchdog (HPO_Host::operator());
 *                            RestartPolicy restarts the inner policy
 * Any other exception that leaves an evaluation is an error -- above all
 * a failure to save best_pot.tadah (durable_best::Error) -- and ends the
 * run with status 1.
 *
 * Where evaluations run concurrently (DLIB:GFS's dlib::thread_pool, the
 * annealing ReplicaPool, ASA's block pool), more than one of them can
 * throw before the optimiser stops: every evaluation that starts after a
 * signal throws ShutdownRequested, every evaluation that is committed
 * after the stagnation window closed throws StagnationStop (DLIB:GFS's
 * when it ends; those of an annealer's or ASA's batch after the batch, in
 * number order, eval_batch.h), and an evaluation that was already running
 * can fail on its own. ExceptionStash keeps the one
 * that decides how the run ends: an error before a signal stop before a
 * stagnation stop. A stop request never hides a failure, and a signal
 * ends the run even when a restart was due. Among exceptions of the same
 * rank the first one is kept.
 *
 * The ranks of a multi-rank run (TADAH_HPO_MPI: plain ANNEAL_PT /
 * ANNEAL_PA) apply the same rule to each other: after a phase in which an
 * evaluation can throw on some ranks only -- rank 0's seed and T0
 * calibrations, a ReplicaPool epoch, the INIT_MODE JITTER AUTO
 * calibration -- the drivers call agree_on_rank_stop(), and every rank
 * ends the way the most severe exception of any rank decides. A signal
 * that one rank received (signals reach the ranks of a job at different
 * moments, or one process only) therefore ends every rank as a signal
 * stop: a rank without one of its own throws PeerShutdown, which names
 * the signal and the rank that received it.
 */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/hpo_mpi.h>

#include <csignal>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tadah::hpo {

/** Thrown by HPO_Host::operator() once no new best was found within the
 *  RESTART stagnation window (HPO_Host<>::StagnationStop names this type).
 *  RestartPolicy catches it and restarts the inner policy; every optimiser
 *  adapter and MultiStartPolicy let it through. */
struct StagnationStop : public std::exception {
    /** "HPO stagnation threshold reached". */
    const char* what() const noexcept override
    {
        return "HPO stagnation threshold reached";
    }
};

/** How much an exception weighs when several end a run at once (see
 *  above): 0 StagnationStop, 1 core::ShutdownRequested, 2 any other
 *  exception (an error); -1 for no exception. */
inline int run_stop_rank(const std::exception_ptr& e) noexcept
{
    if (!e) return -1;
    try {
        std::rethrow_exception(e);
    } catch (const StagnationStop&) {
        return 0;
    } catch (const core::ShutdownRequested&) {
        return 1;
    } catch (...) {
        return 2;
    }
}

/** Keeps the exception that decides how the run ends out of those the
 *  concurrent evaluations of one optimiser raise (see above). Thread-safe. */
class ExceptionStash {
public:
    /** Keep `e` if it outranks the exception kept so far. */
    void offer(std::exception_ptr e)
    {
        const int rank = run_stop_rank(e);
        std::lock_guard<std::mutex> lk(m_);
        if (rank > rank_) {
            error_ = std::move(e);
            rank_  = rank;
        }
    }

    /** Whether an exception is kept. */
    bool has() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return static_cast<bool>(error_);
    }

    /** The kept exception (null if none), left in the stash: what a stop
     *  line names (run_stop_rank() tells which stop it is). */
    std::exception_ptr peek() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return error_;
    }

    /** Hand over the kept exception (null if none) and empty the stash. */
    std::exception_ptr take()
    {
        std::lock_guard<std::mutex> lk(m_);
        std::exception_ptr e = std::move(error_);
        error_ = nullptr;
        rank_  = -1;
        return e;
    }

    /** Rethrow the kept exception, if any, and empty the stash. */
    void rethrow_if_any()
    {
        if (std::exception_ptr e = take()) std::rethrow_exception(e);
    }

private:
    mutable std::mutex m_;
    std::exception_ptr error_;
    int                rank_ = -1;
};

/* The name core::SignalHandler::reason() gives a signal number: SIGINT,
 * SIGTERM, SIGUSR2, "signal N" for another, "no signal" for 0. */
inline std::string signal_name(int signo)
{
    switch (signo) {
        case 0:       return "no signal";
        case SIGINT:  return "SIGINT";
        case SIGTERM: return "SIGTERM";
        case SIGUSR2: return "SIGUSR2";
        default:      return "signal " + std::to_string(signo);
    }
}

/* A signal stop this MPI rank takes from another rank (agree_on_rank_stop):
 * `origin_rank` received `signo` (0 when its ShutdownRequested came from
 * code, not from a signal) and this rank had no stop of its own. A
 * core::ShutdownRequested, so every handler of a signal stop handles it;
 * drive_run() ends the rank with "SHUTDOWN (<signal> on MPI rank
 * <origin>)" and 128 + signo. */
struct PeerShutdown : public core::ShutdownRequested {
    PeerShutdown(int signo_in, int origin_rank_in) noexcept
        : signo(signo_in), origin_rank(origin_rank_in) {}
    const char* what() const noexcept override
    {
        return "shutdown requested on another MPI rank";
    }
    int signo = 0;
    int origin_rank = 0;
};

/* The ranks' agreement on how a phase ended (see the header comment). A
 * collective: every rank of a multi-rank run calls it at the same point,
 * with the exception the phase ended with on that rank (null if none); on
 * one rank, or without MPI, it only rethrows `local`, if any. When no rank
 * has an exception it returns. Otherwise it throws on every rank, and the
 * most severe exception of any rank (run_stop_rank: an error before a
 * signal stop before a stagnation stop) decides how:
 *   - a rank whose own exception is of that kind rethrows it, whole (its
 *     type matters: a ShutdownRequested to the run driver, a
 *     durable_best::Error, ...);
 *   - another rank throws std::runtime_error(peer_error()) for an error,
 *     PeerShutdown for a signal stop -- the signal of the lowest rank whose
 *     own exception is a ShutdownRequested, and that rank -- and
 *     StagnationStop for a stagnation stop.
 * One allreduce when no rank has an exception (the common path); two more
 * collectives name the signal of a signal stop. `peer_error` is called only
 * on a rank that needs the message. */
template<class PeerError>
void agree_on_rank_stop(std::exception_ptr local, PeerError&& peer_error)
{
    if (hpo_size() <= 1) {
        if (local) std::rethrow_exception(local);
        return;
    }
    const int own = run_stop_rank(local);
    const int agreed = hpo_allreduce_max(own);
    if (agreed < 0) return;
    if (agreed == 1) {
        /* The lowest rank that stopped on a signal, and its signal: every
         * rank takes part, also those that rethrow their own below. */
        const int rank = hpo_rank();
        const int origin =
            -hpo_allreduce_max(own == 1 ? -rank
                                        : std::numeric_limits<int>::min() + 1);
        int signo = rank == origin ? core::SignalHandler::signal_seen() : 0;
        hpo_bcast(signo, origin);
        if (own == 1) std::rethrow_exception(local);
        throw PeerShutdown(signo, origin);
    }
    if (own == agreed) std::rethrow_exception(local);
    if (agreed == 2) throw std::runtime_error(peer_error());
    throw StagnationStop{};
}

}  // namespace tadah::hpo

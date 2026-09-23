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
 * signal throws ShutdownRequested, every evaluation that ends after the
 * stagnation window closed throws StagnationStop, and an evaluation that
 * was already running can fail on its own. ExceptionStash keeps the one
 * that decides how the run ends: an error before a signal stop before a
 * stagnation stop. A stop request never hides a failure, and a signal
 * ends the run even when a restart was due. Among exceptions of the same
 * rank the first one is kept.
 */

#include <tadah/core/checkpoint.h>   // core::ShutdownRequested

#include <exception>
#include <mutex>

namespace tadah::hpo {

/* Thrown by HPO_Host::operator() once no new best was found within the
 * RESTART stagnation window (HPO_Host<>::StagnationStop names this type).
 * RestartPolicy catches it and restarts the inner policy; every optimiser
 * adapter and MultiStartPolicy let it through. */
struct StagnationStop : public std::exception {
    const char* what() const noexcept override
    {
        return "HPO stagnation threshold reached";
    }
};

/* How much an exception weighs when several end a run at once (see
 * above): 0 StagnationStop, 1 core::ShutdownRequested, 2 any other
 * exception (an error); -1 for no exception. */
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

/* Keeps the exception that decides how the run ends out of those the
 * concurrent evaluations of one optimiser raise (see above). Thread-safe. */
class ExceptionStash {
public:
    /* Keep `e` if it outranks the exception kept so far. */
    void offer(std::exception_ptr e)
    {
        const int rank = run_stop_rank(e);
        std::lock_guard<std::mutex> lk(m_);
        if (rank > rank_) {
            error_ = std::move(e);
            rank_  = rank;
        }
    }

    bool has() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return static_cast<bool>(error_);
    }

    /* Hand over the kept exception (null if none) and empty the stash. */
    std::exception_ptr take()
    {
        std::lock_guard<std::mutex> lk(m_);
        std::exception_ptr e = std::move(error_);
        error_ = nullptr;
        rank_  = -1;
        return e;
    }

    /* Rethrow the kept exception, if any, and empty the stash. */
    void rethrow_if_any()
    {
        if (std::exception_ptr e = take()) std::rethrow_exception(e);
    }

private:
    mutable std::mutex m_;
    std::exception_ptr error_;
    int                rank_ = -1;
};

}  // namespace tadah::hpo

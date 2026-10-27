#pragma once
/* anneal_parallel.h -- persistent worker pool for the parallel
 * annealing drivers (TADAH:ANNEAL_PT / TADAH:ANNEAL_PA).
 *
 * Design constraints (see the parallel-SA plan):
 *   - PERSISTENT workers: HPO_Host's heavy caches (trainer, LAMMPS
 *     instances) are `static thread_local`; fork-join per epoch would
 *     re-pay the ~cold-eval cost every epoch. One pool lives for the
 *     whole driver call.
 *   - STATIC task assignment (task t -> worker t % N): chain-to-thread
 *     mapping is stable, so per-worker thread_local caches see a fixed
 *     subset of chains and results are identical for ANY worker count
 *     (each chain draws only from its own RNG stream).
 *   - Exceptions (core::ShutdownRequested on SIGINT, anything else) are
 *     caught on the worker and stashed, an external stop flag is raised
 *     so sibling chains wind down at their next proposal, and the driver
 *     rethrows after the epoch barrier -- preserving the RestartPolicy /
 *     signal-handler semantics that rely on exceptions propagating out of
 *     host(). The commits of the epoch's evaluations (eval_batch.h) run
 *     on the workers as the evaluations end, and keep their exceptions --
 *     HPO_Host's stagnation stop, a failure to save best_pot.tadah --
 *     until the barrier, after which EvalBatch::commit hands them over
 *     and the driver adds them with offer_error. When several are stashed, the stash keeps the one that
 *     decides how the run ends (an error before a signal stop before a
 *     stagnation stop, run_stop.h).
 */
#include <tadah/hpo/run_stop.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace tadah::hpo {
namespace anneal {

class ReplicaPool {
public:
  /* `ext_stop` is raised when a worker catches an exception, so chains
   * running on other workers stop at their next proposal check. */
  ReplicaPool(std::size_t n_workers, std::atomic<bool>& ext_stop)
      : ext_stop_(ext_stop)
  {
    if (n_workers == 0) n_workers = 1;
    workers_.reserve(n_workers);
    for (std::size_t id = 0; id < n_workers; ++id)
      workers_.emplace_back([this, id] { worker_loop(id); });
  }

  ~ReplicaPool()
  {
    {
      std::lock_guard<std::mutex> lk(m_);
      shutdown_ = true;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ReplicaPool(const ReplicaPool&) = delete;
  ReplicaPool& operator=(const ReplicaPool&) = delete;

  std::size_t n_workers() const { return workers_.size(); }

  /* Run fn(t) for t in [0, n_tasks) with static assignment
   * t -> worker t % n_workers; blocks until every task returned. */
  void run_epoch(std::size_t n_tasks,
                 const std::function<void(std::size_t)>& fn)
  {
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_         = &fn;
      n_tasks_    = n_tasks;
      done_count_ = 0;
      ++epoch_;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return done_count_ == workers_.size(); });
    fn_ = nullptr;
  }

  bool has_error() const { return errors_.has(); }

  /* Rethrow the exception the workers' stash kept (see the header
   * comment), then clear it. Call after run_epoch from the driver
   * thread. */
  void rethrow_if_error() { errors_.rethrow_if_any(); }

  /* Hand over the exception the workers' stash kept (null if none) and
   * clear it, for the ranks' agreement on how the epoch ended
   * (agree_on_rank_stop, run_stop.h). Driver thread, after run_epoch. */
  std::exception_ptr take_error() { return errors_.take(); }

  /* Add an exception the driver met after run_epoch -- of the commits of
   * the epoch's evaluations (EvalBatch::commit, eval_batch.h) -- to the
   * workers' stash, which keeps the one that decides how the run ends;
   * null adds nothing. Driver thread, after run_epoch. */
  void offer_error(std::exception_ptr e)
  {
    if (e) errors_.offer(std::move(e));
  }

private:
  void worker_loop(std::size_t id)
  {
    std::size_t seen_epoch = 0;
    for (;;) {
      const std::function<void(std::size_t)>* fn = nullptr;
      std::size_t n_tasks = 0;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_start_.wait(lk, [&] { return shutdown_ || epoch_ != seen_epoch; });
        if (shutdown_) return;
        seen_epoch = epoch_;
        fn         = fn_;
        n_tasks    = n_tasks_;
      }
      for (std::size_t t = id; t < n_tasks; t += workers_.size()) {
        try {
          (*fn)(t);
        } catch (...) {
          errors_.offer(std::current_exception());
          ext_stop_.store(true, std::memory_order_relaxed);
        }
      }
      {
        std::lock_guard<std::mutex> lk(m_);
        ++done_count_;
      }
      cv_done_.notify_all();
    }
  }

  std::vector<std::thread> workers_;
  std::atomic<bool>&       ext_stop_;

  mutable std::mutex       m_;
  std::condition_variable  cv_start_, cv_done_;
  std::size_t              epoch_ = 0;
  std::size_t              n_tasks_ = 0;
  const std::function<void(std::size_t)>* fn_ = nullptr;
  std::size_t              done_count_ = 0;
  bool                     shutdown_ = false;
  ExceptionStash           errors_;
};

}  // namespace anneal
}  // namespace tadah::hpo

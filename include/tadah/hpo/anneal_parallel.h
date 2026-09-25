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
 *   - Exceptions (HPO_Host stagnation-stop, core::ShutdownRequested on
 *     SIGINT, anything else) are caught on the worker, the most severe one
 *     is kept (ExceptionSlot, exception_severity.h), an external stop flag
 *     is raised so sibling chains wind down at their next proposal, and
 *     the driver rethrows after the epoch barrier -- preserving the
 *     RestartPolicy / signal-handler semantics that rely on exceptions
 *     propagating out of host().
 */
#include <tadah/hpo/exception_severity.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
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

  bool has_error() const { return errors_.has_error(); }

  /* Rethrow the most severe exception any worker caught -- the first one
   * of that severity (then clears it). Call after run_epoch from the
   * driver thread. Chains of one epoch can fail concurrently, and the one
   * kept decides how the run ends (exception_severity.h). */
  void rethrow_if_error() { errors_.rethrow_if_error(); }

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
          errors_.keep_current();
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
  ExceptionSlot            errors_;
};

}  // namespace anneal
}  // namespace tadah::hpo

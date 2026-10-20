#pragma once
/* The stagnation check of the RESTART watchdog (HPO_Host::operator()):
 * when it fires it throws StagnationStop, the run-level stop defined in
 * run_stop.h. Kept apart from the host so code that only needs the check
 * (and the new-best bookkeeping it waits on) does not pull the host in. */

#include <tadah/hpo/run_stop.h>   // StagnationStop

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace tadah::hpo {

namespace detail {

/** New-best decisions in progress. An evaluation whose loss may become the
 *  new best counts itself in `started` when it makes the decision
 *  (BestDecision), BEFORE what it does ahead of the host's best mutex (the
 *  balance table of its fit) and before it waits for the mutex; it counts
 *  itself in `done` under the mutex when the decision ends, and wakes the
 *  waiters of throw_if_stagnated. Under the mutex the decision publishes
 *  the best potential (fsync + rename, or a transient-I/O backoff) and
 *  records the eval number of a new best. `done` counts decisions in the
 *  order they take the mutex, which is not the order they were counted in
 *  `started` (std::mutex is not FIFO). */
struct BestPublications {
  std::atomic<std::size_t> started{0};  ///< decisions counted
  std::atomic<std::size_t> done{0};     ///< decisions ended (under the mutex)
  std::condition_variable  cv;          ///< wakes throw_if_stagnated
};

/** One new-best decision, from its count to its end: counted in
 *  `started` on construction; lock() takes the best mutex; the destructor
 *  counts it in `done` under the mutex -- taking it if the decision ended
 *  before lock(), as when what it did ahead of the mutex threw -- and wakes
 *  the waiters, so a decision that ends by an exception never leaves a
 *  stagnation check waiting. */
class BestDecision {
public:
  /** Counts the decision in @p p's `started`; @p best_mtx is the host's
   *  best mutex, which lock() takes. */
  BestDecision(BestPublications& p, std::mutex& best_mtx)
      : p_(p), lock_(best_mtx, std::defer_lock)
  {
    p_.started.fetch_add(1);
  }
  /** Takes the best mutex, held until the decision ends. */
  void lock() { lock_.lock(); }
  ~BestDecision()
  {
    if (!lock_.owns_lock()) lock_.lock();
    p_.done.fetch_add(1);
    lock_.unlock();
    p_.cv.notify_all();   // without a waiter: no system call
  }
  BestDecision(const BestDecision&) = delete;
  BestDecision& operator=(const BestDecision&) = delete;
private:
  BestPublications& p_;
  std::unique_lock<std::mutex> lock_;
};

/** The stagnation check of evaluation `iter`: throws StagnationStop when
 *  `threshold` (> 0) evaluations have passed since `last_best` (the eval
 *  number of the last new best), each of the `reused` calls since then
 *  that a search answered from the run's record of an earlier evaluation
 *  (repeat_points.h) counted as one: such a call is no new best, and a
 *  search that only asks for points the run has makes no evaluation. A reused
 *  call checks with `iter` the latest evaluation (HPO_Host::
 *  count_reused_repeat).
 *
 *  An evaluation that cannot become the new best skips the best mutex, so
 *  it can check while an earlier evaluation's new best is still being
 *  decided -- publishing under the mutex, or queued for it -- and would
 *  stop the optimiser although the run is improving. So a check that would
 *  fire first waits until as many new-best decisions have finished as had
 *  started by then, and checks again. Decisions finish in the order they
 *  take the mutex, which is not FIFO: a decision overtaken at the mutex
 *  by one counted later (a barging arrival at an unlock, or a thread
 *  preempted between counting and locking) can still be missed. That miss
 *  is rare, and of the same kind as missing an evaluation that is still
 *  computing: an early stop, never a lost best. A check that does not fire
 *  costs two relaxed loads; only an evaluation about to stop the optimiser
 *  waits. The caller must not hold `best_mtx`. */
inline void throw_if_stagnated(std::size_t iter, std::size_t threshold,
                               const std::atomic<std::size_t>& last_best,
                               const std::atomic<std::size_t>& reused,
                               std::mutex& best_mtx, BestPublications& pubs)
{
  if (threshold == 0) return;
  auto fires = [&] {
    const std::size_t last = last_best.load(std::memory_order_relaxed);
    if (iter < last) return false;
    return (iter - last) + reused.load(std::memory_order_relaxed) >= threshold;
  };
  if (!fires()) return;
  const std::size_t started = pubs.started.load();
  {
    std::unique_lock<std::mutex> lk(best_mtx);
    pubs.cv.wait(lk, [&] { return pubs.done.load() >= started; });
  }
  if (fires()) throw StagnationStop{};
}

}  // namespace detail

}  // namespace tadah::hpo

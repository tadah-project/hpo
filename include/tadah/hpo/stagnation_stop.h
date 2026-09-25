#pragma once
/* StagnationStop -- the control-flow exception HPO_Host::operator() throws
 * when a RESTART (or BASINS stage) stagnation window runs out. Kept in its
 * own header so code that only needs to recognise it (the replica pool, the
 * run policies) does not pull in the host. */

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>

namespace tadah::hpo {

/** Exception thrown by HPO_Host::operator() to signal a stagnation-driven
 *  abort. The RestartPolicy catches and re-enters with a fresh init. One
 *  type for every host instantiation (HPO_Host<...>::StagnationStop is an
 *  alias), so code that is generic over the host -- optimiser adapters,
 *  MULTISTART -- can let it through without knowing the host type. */
struct StagnationStop : public std::exception {
  const char* what() const noexcept override { return "HPO stagnation threshold reached"; }
};

namespace detail {

/** New-best decisions in progress. An evaluation whose loss may become the
 *  new best counts itself in `started` BEFORE it waits for the host's best
 *  mutex; a PublicationDone declared after that lock -- so it runs while the
 *  mutex is still held -- counts it in `done` and wakes the waiters of
 *  throw_if_stagnated. Under the mutex the decision publishes the best
 *  potential (fsync + rename, or a transient-I/O backoff) and records the
 *  eval number of a new best. `done` counts decisions in the order they
 *  take the mutex, which is not the order they were counted in `started`
 *  (std::mutex is not FIFO). */
struct BestPublications {
  std::atomic<std::size_t> started{0};
  std::atomic<std::size_t> done{0};   // written under the best mutex only
  std::condition_variable  cv;
};

class PublicationDone {
public:
  explicit PublicationDone(BestPublications& p) noexcept : p_(p) {}
  ~PublicationDone()
  {
    p_.done.fetch_add(1);
    p_.cv.notify_all();   // without a waiter: no system call
  }
  PublicationDone(const PublicationDone&) = delete;
  PublicationDone& operator=(const PublicationDone&) = delete;
private:
  BestPublications& p_;
};

/** The stagnation check of evaluation `iter`: throws StagnationStop when
 *  `threshold` (> 0) evaluations have passed since `last_best` (the eval
 *  number of the last new best).
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
                               std::mutex& best_mtx, BestPublications& pubs)
{
  if (threshold == 0) return;
  auto fires = [&] {
    const std::size_t last = last_best.load(std::memory_order_relaxed);
    return iter > last && iter - last >= threshold;
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

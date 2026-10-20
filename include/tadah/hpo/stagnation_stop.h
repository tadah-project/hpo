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

class BestDecision;

/** New-best decisions in progress. An evaluation whose loss may become the
 *  new best counts itself when it makes the decision (BestDecision), BEFORE
 *  what it does ahead of the host's best mutex (the balance table of its
 *  fit) and before it waits for the mutex: it takes the next ticket,
 *  `started`, and joins the list of decisions in progress, in ticket order.
 *  It leaves the list and counts itself in `done` under the mutex when the
 *  decision ends, and wakes the waiters of throw_if_stagnated. Under the
 *  mutex the decision publishes the best potential (fsync + rename, or a
 *  transient-I/O backoff) and records the eval number of a new best.
 *  Decisions end in the order they take the mutex, which is not the order
 *  of their tickets (std::mutex is not FIFO, and the work ahead of the
 *  mutex takes longer for some), so a check waits for the tickets below
 *  its own snapshot of `started` (ended_below), not for a count. */
struct BestPublications {
  std::atomic<std::size_t> started{0};  ///< decisions counted: the next ticket
  std::atomic<std::size_t> done{0};     ///< decisions ended (under the mutex)
  std::condition_variable  cv;          ///< wakes throw_if_stagnated
  std::mutex               mtx;         ///< guards the tickets and the list
  BestDecision*            first = nullptr;  ///< the oldest in progress
  BestDecision*            last = nullptr;   ///< the newest in progress

  /** The ticket the next decision takes: every decision with a smaller one
   *  has been counted and is in the list until it ends. */
  std::size_t next_ticket()
  {
    std::lock_guard<std::mutex> g(mtx);
    return started.load();
  }
  /** Whether every decision with a ticket below @p ticket has ended. */
  inline bool ended_below(std::size_t ticket);
};

/** One new-best decision, from its count to its end: counted on
 *  construction (its ticket, and a place at the end of the list of
 *  decisions in progress); lock() takes the best mutex; the destructor
 *  takes it off the list and counts it in `done` under the mutex -- taking
 *  the mutex if the decision ended before lock(), as when what it did
 *  ahead of the mutex threw -- and wakes the waiters, so a decision that
 *  ends by an exception never leaves a stagnation check waiting. The lock
 *  order is the best mutex, then BestPublications::mtx. */
class BestDecision {
public:
  /** Counts the decision in @p p; @p best_mtx is the host's best mutex,
   *  which lock() takes. */
  BestDecision(BestPublications& p, std::mutex& best_mtx)
      : p_(p), lock_(best_mtx, std::defer_lock)
  {
    std::lock_guard<std::mutex> g(p_.mtx);
    ticket_ = p_.started.fetch_add(1);
    prev_ = p_.last;
    if (prev_) prev_->next_ = this;
    else p_.first = this;
    p_.last = this;
  }
  /** Takes the best mutex, held until the decision ends. */
  void lock() { lock_.lock(); }
  /** The decision's place in the order of counting, from 0. */
  std::size_t ticket() const { return ticket_; }
  ~BestDecision()
  {
    if (!lock_.owns_lock()) lock_.lock();
    {
      std::lock_guard<std::mutex> g(p_.mtx);
      if (prev_) prev_->next_ = next_;
      else p_.first = next_;
      if (next_) next_->prev_ = prev_;
      else p_.last = prev_;
    }
    p_.done.fetch_add(1);
    lock_.unlock();
    p_.cv.notify_all();   // without a waiter: no system call
  }
  BestDecision(const BestDecision&) = delete;
  BestDecision& operator=(const BestDecision&) = delete;
private:
  BestPublications& p_;
  std::unique_lock<std::mutex> lock_;
  std::size_t ticket_ = 0;
  BestDecision* prev_ = nullptr;   ///< the decision counted before, in progress
  BestDecision* next_ = nullptr;   ///< the decision counted after, in progress
};

inline bool BestPublications::ended_below(std::size_t ticket)
{
  std::lock_guard<std::mutex> g(mtx);
  return first == nullptr || first->ticket() >= ticket;
}

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
 *  decided -- computing its balance table, queued for the mutex, or
 *  publishing under it -- and would stop the optimiser although the run
 *  is improving. So a check that would fire first waits until every
 *  new-best decision counted before it began has ended (each has a
 *  ticket, BestPublications::ended_below), whatever the order in which
 *  they end, and checks again. A decision counted after the check began is
 *  not waited for; like an evaluation that is still computing, missing it
 *  can stop the optimiser early, but never loses a best. A check that does
 *  not fire costs two relaxed loads; only an evaluation about to stop the
 *  optimiser waits. The caller must not hold `best_mtx`, nor be in a
 *  decision of its own. */
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
  const std::size_t counted = pubs.next_ticket();
  {
    std::unique_lock<std::mutex> lk(best_mtx);
    pubs.cv.wait(lk, [&] { return pubs.ended_below(counted); });
  }
  if (fires()) throw StagnationStop{};
}

}  // namespace detail

}  // namespace tadah::hpo

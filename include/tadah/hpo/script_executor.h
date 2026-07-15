#pragma once
/* script_executor.h -- a persistent, per-rank pool of worker std::threads that
 * runs batches of independent tasks (Stage 2 of the pc_lammps fan-out work).
 *
 * Why this exists
 * ---------------
 * The pc_lammps fan-out used `#pragma omp parallel for`. In ANNEAL_PT/PA every
 * replica driver spawned its OWN OMP team, so K replicas nested K teams and the
 * cores could not be shared across replicas: replica A's fan-out imbalance sat
 * idle while replica B still had work. Confining the ambient OMP width to 1 to
 * curb the nesting broke bit-determinism (LAMMPS on nested OMP team threads at
 * OMP=1 perturbs the chaotic relaxations -- see the Stage 1 report).
 *
 * A single shared std::thread pool fixes both: ONE pool of C workers (C = the
 * per-rank core budget) serves EVERY replica, so a replica's fan-out gap is
 * backfilled by another replica's scripts, and there is no OMP nesting. Running
 * LAMMPS on a plain std::thread (not a nested OMP team thread) is the
 * determinism-safe regime -- it matches the 4:1 case that stayed bit-identical.
 *
 * Design (mirrors anneal::ReplicaPool's proven exception/latch handling)
 * ---------------------------------------------------------------------
 *   - PERSISTENT workers: HPO_Host's heavy caches (LAMMPS instances, LammpsTadah)
 *     are `static thread_local`, so a fixed worker set warms them ONCE per run
 *     rather than once per OMP team.
 *   - CONCURRENT batches: several replica drivers call run_batch() at once. Tasks
 *     from all batches share one MPMC queue (moodycamel), so the C workers pull
 *     the next ready script regardless of which replica it belongs to (dynamic
 *     scheduling + cross-replica backfill in one mechanism). Each task carries a
 *     pointer to its own Batch latch; a driver waits only on its own latch.
 *   - worker_id in [0, C) is handed to the task in place of omp_get_thread_num()
 *     for the per-thread `-log/-screen` filename suffix.
 *   - Determinism is a property of the CALLER (each script writes only its own
 *     r.v_outvar[i]; sims_loss sums in fixed index order): the executor only
 *     changes WHICH worker runs a script and WHEN, never a result value.
 *
 * Latch lifetime (UAF-safety): the Batch lives on run_batch's stack. A worker
 * decrements `remaining` and notifies WHILE HOLDING the batch mutex, and the
 * waiter can only return (and destroy the Batch) after re-acquiring that mutex,
 * so no worker touches a freed latch. Same invariant as ReplicaPool::done_count_.
 */
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <moodycamel/blockingconcurrentqueue.h>

namespace tadah::hpo {

class ScriptExecutor {
public:
  /* fn(task_index, worker_id) is invoked once per task in [0, n_tasks). */
  using TaskFn = std::function<void(std::size_t, int)>;

  explicit ScriptExecutor(int n_workers)
      : n_workers_(n_workers < 1 ? 1 : n_workers)
  {
    workers_.reserve(static_cast<std::size_t>(n_workers_));
    for (int w = 0; w < n_workers_; ++w)
      workers_.emplace_back([this, w] { worker_loop(w); });
  }

  ~ScriptExecutor()
  {
    /* One poison pill per worker; the shared queue hands each worker exactly
     * one, so every worker sees a stop and joins. No batches are in flight at
     * destruction (the executor outlives every run_batch caller). */
    for (int w = 0; w < n_workers_; ++w) {
      Task t; t.stop = true;
      q_.enqueue(t);
    }
    for (auto& w : workers_) w.join();
  }

  ScriptExecutor(const ScriptExecutor&) = delete;
  ScriptExecutor& operator=(const ScriptExecutor&) = delete;

  int width() const { return n_workers_; }

  /* Run fn(i, worker_id) for i in [0, n_tasks); block until every task has
   * returned. If any task threw, the FIRST exception is rethrown here after all
   * tasks complete (so no task outlives the Batch). Tasks are otherwise
   * independent -- one throwing does not cancel the rest. */
  void run_batch(std::size_t n_tasks, const TaskFn& fn)
  {
    if (n_tasks == 0) return;

    /* Reentrancy guard (strict-DAG / deadlock-free): a task already running on a
     * pool worker that calls run_batch again must NOT submit-then-wait -- its
     * own worker slot is occupied by itself, so it could wait on work only it
     * could run. Execute such a nested batch INLINE on the calling thread, still
     * honouring the "run all, rethrow the first exception" contract. Not reached
     * by the current call sites (fan-out and MLIP train/predict are driven from
     * ReplicaPool / driver threads, never from a pool worker), but it makes the
     * pool safe for any future nested use. worker_id 0 is fine: kernels ignore
     * it and the fan-out never nests. */
    if (on_worker_thread_()) {
      std::exception_ptr first;
      for (std::size_t i = 0; i < n_tasks; ++i) {
        try { fn(i, 0); }
        catch (...) { if (!first) first = std::current_exception(); }
      }
      if (first) std::rethrow_exception(first);
      return;
    }

    Batch b;
    b.remaining = n_tasks;

    /* Enqueue in the caller's order; moodycamel is FIFO-ish per producer, so a
     * caller that pre-sorts longest-first gets longest-first dispatch. */
    for (std::size_t i = 0; i < n_tasks; ++i) {
      Task t;
      t.batch = &b;
      t.idx   = i;
      t.fn    = &fn;
      q_.enqueue(t);
    }

    std::unique_lock<std::mutex> lk(b.m);
    b.cv.wait(lk, [&] { return b.remaining == 0; });
    if (b.err) std::rethrow_exception(b.err);
  }

private:
  struct Batch {
    std::size_t             remaining = 0;   /* guarded by m */
    std::mutex              m;
    std::condition_variable cv;
    std::exception_ptr      err;             /* first exception, guarded by m */
  };

  struct Task {
    Batch*        batch = nullptr;
    std::size_t   idx   = 0;
    const TaskFn* fn    = nullptr;
    bool          stop  = false;
  };

  /* True on this pool's own worker threads (read by the run_batch reentrancy
   * guard). thread_local + static: a driver thread reads false, a worker true. */
  static bool& on_worker_thread_() { static thread_local bool v = false; return v; }

  void worker_loop(int worker_id)
  {
    on_worker_thread_() = true;
    for (;;) {
      Task t;
      q_.wait_dequeue(t);
      if (t.stop) return;

      std::exception_ptr caught;
      try {
        (*t.fn)(t.idx, worker_id);
      } catch (...) {
        caught = std::current_exception();
      }

      /* Decrement + notify UNDER the batch lock so the waiter cannot free the
       * Batch while we are still touching it (see the header note). */
      std::lock_guard<std::mutex> lk(t.batch->m);
      if (caught && !t.batch->err) t.batch->err = caught;
      if (--t.batch->remaining == 0) t.batch->cv.notify_all();
    }
  }

  int                             n_workers_;
  std::vector<std::thread>        workers_;
  moodycamel::BlockingConcurrentQueue<Task> q_;
};

/* -- per-rank singleton --------------------------------------------------
 * The pool width (the per-rank core budget) is known only at startup, so
 * hpo_run() calls init_script_executor(C) once before the optimiser runs; every
 * later script_executor() returns that instance. Kept as a leak-free function
 * static so teardown order never races a late fan-out. */
namespace detail {
inline std::atomic<int>& script_executor_width() {
  static std::atomic<int> w{1};
  return w;
}
}  // namespace detail

inline void init_script_executor(int width) {
  detail::script_executor_width().store(width < 1 ? 1 : width,
                                        std::memory_order_relaxed);
}

inline ScriptExecutor& script_executor() {
  static ScriptExecutor inst(
      detail::script_executor_width().load(std::memory_order_relaxed));
  return inst;
}

}  // namespace tadah::hpo

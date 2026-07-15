#include "catch2/catch.hpp"

#include <tadah/hpo/script_executor.h>

#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using tadah::hpo::ScriptExecutor;

/* Stage 2.1 -- the shared std::thread pool that replaces the pc_lammps fan-out's
 * `#pragma omp parallel for`. The determinism contract it must uphold: each task
 * writes only its own slot, so the RESULT is invariant to worker count and
 * scheduling; the executor only changes which worker runs a task and when. */

TEST_CASE("ScriptExecutor runs every task exactly once", "[script_executor]") {
  for (int width : {1, 2, 4, 8}) {
    ScriptExecutor ex(width);
    REQUIRE(ex.width() == width);

    const std::size_t N = 200;
    std::vector<int> ran(N, 0);
    std::vector<int> seen_worker(N, -1);

    ex.run_batch(N, [&](std::size_t i, int wid) {
      ran[i] += 1;                 // each slot written by exactly one task
      seen_worker[i] = wid;
    });

    for (std::size_t i = 0; i < N; ++i) {
      REQUIRE(ran[i] == 1);                       // exactly once
      REQUIRE(seen_worker[i] >= 0);
      REQUIRE(seen_worker[i] < width);            // worker_id in [0, width)
    }
  }
}

TEST_CASE("ScriptExecutor result is invariant to worker count", "[script_executor]") {
  const std::size_t N = 500;
  // reference: serial computation
  std::vector<long> ref(N);
  for (std::size_t i = 0; i < N; ++i) ref[i] = static_cast<long>(i) * 7 - 3;

  for (int width : {1, 3, 8, 16}) {
    ScriptExecutor ex(width);
    std::vector<long> out(N, 0);
    ex.run_batch(N, [&](std::size_t i, int) {
      out[i] = static_cast<long>(i) * 7 - 3;      // pure per-index write
    });
    REQUIRE(out == ref);                          // bit-identical across widths
  }
}

TEST_CASE("ScriptExecutor rethrows the first exception, all tasks still run",
          "[script_executor]") {
  ScriptExecutor ex(4);
  const std::size_t N = 100;
  std::atomic<int> completed{0};

  auto body = [&](std::size_t i, int) {
    completed.fetch_add(1, std::memory_order_relaxed);
    if (i == 42) throw std::runtime_error("boom");
  };

  REQUIRE_THROWS_AS(ex.run_batch(N, body), std::runtime_error);
  // Even though task 42 threw, the batch drained fully before run_batch returned.
  REQUIRE(completed.load() == static_cast<int>(N));
}

TEST_CASE("ScriptExecutor handles concurrent batches (cross-driver sharing)",
          "[script_executor]") {
  // One shared pool, several driver threads each submitting its own batch --
  // this is the ANNEAL_PT/PA case (K replica drivers, one executor).
  ScriptExecutor ex(6);
  const int DRIVERS = 5;
  const std::size_t N = 300;

  std::vector<std::vector<long>> out(DRIVERS, std::vector<long>(N, -1));
  std::vector<std::thread> drivers;
  for (int d = 0; d < DRIVERS; ++d) {
    drivers.emplace_back([&, d] {
      ex.run_batch(N, [&, d](std::size_t i, int) {
        out[d][i] = static_cast<long>(d) * 1000 + static_cast<long>(i);
      });
    });
  }
  for (auto& t : drivers) t.join();

  for (int d = 0; d < DRIVERS; ++d)
    for (std::size_t i = 0; i < N; ++i)
      REQUIRE(out[d][i] == static_cast<long>(d) * 1000 + static_cast<long>(i));
}

TEST_CASE("ScriptExecutor empty batch is a no-op", "[script_executor]") {
  ScriptExecutor ex(4);
  int calls = 0;
  ex.run_batch(0, [&](std::size_t, int) { ++calls; });
  REQUIRE(calls == 0);
}

TEST_CASE("ScriptExecutor reused across many batches keeps workers warm",
          "[script_executor]") {
  ScriptExecutor ex(4);
  long total = 0;
  std::mutex mtx;
  for (int round = 0; round < 20; ++round) {
    ex.run_batch(50, [&](std::size_t i, int) {
      std::lock_guard<std::mutex> lk(mtx);
      total += static_cast<long>(i);
    });
  }
  REQUIRE(total == 20L * (49 * 50 / 2));   // 20 rounds * sum(0..49)
}

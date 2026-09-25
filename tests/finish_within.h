#pragma once
/* finish_within() -- a bounded wait for code that may never return.
 *
 * Some defects make a call loop forever without returning control (ASA's
 * generation loop spinning inside the C library, for one). No test can
 * interrupt or join such a thread, and waiting for the CTest timeout turns
 * a clear failure into a slow, silent one. finish_within() runs `body` on a
 * thread of its own and waits at most `limit` for it; if `body` has not
 * returned by then, the process reports `what` on stderr and exits with
 * EXIT_FAILURE, so the CTest entry of the case fails at once and says why
 * (every Catch case is a CTest entry of its own). An exception thrown by
 * `body` is rethrown on the calling thread.
 *
 * Catch2 assertions are not thread-safe: `body` only computes, and the
 * case checks its results after finish_within() returns. */

#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <thread>

namespace hpo_test {

inline void finish_within(std::chrono::seconds limit, const std::string& what,
                          std::function<void()> body)
{
    std::packaged_task<void()> task(std::move(body));
    std::future<void> done = task.get_future();
    std::thread runner(std::move(task));
    if (done.wait_for(limit) != std::future_status::ready) {
        std::cerr << "FAILED: " << what << " did not return within "
                  << limit.count() << " s; the thread running it cannot be "
                  << "joined, so the test process exits now\n"
                  << std::flush;
        std::_Exit(EXIT_FAILURE);
    }
    runner.join();
    done.get();
}

}  // namespace hpo_test

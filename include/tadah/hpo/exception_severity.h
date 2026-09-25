#pragma once
/* exception_severity.h -- which of several concurrently caught host
 * exceptions decides how a run ends.
 *
 * Parallel drivers (the anneal::ReplicaPool of ANNEAL_PT / ANNEAL_PA, ASA's
 * block evaluations, DLIB:GFS with THREADS > 1) can catch more than one
 * exception from HPO_Host in one step. The one they rethrow decides the
 * outcome, so a real error -- a best potential that could not be saved, any
 * other failure -- outranks a shutdown request (exit 0), which outranks a
 * stagnation stop (RESTART restarts). Of equal severity the first is kept.
 * One definition for every driver; light enough for anneal_parallel.h. */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/hpo/stagnation_stop.h>

#include <atomic>
#include <exception>
#include <mutex>

namespace tadah::hpo {

/** Severity of the exception being handled: StagnationStop 1,
 *  core::ShutdownRequested 2, anything else 3. Call only inside a catch
 *  handler; std::current_exception() still refers to that exception
 *  afterwards. */
inline int exception_severity_of_current() noexcept
{
  try {
    throw;
  } catch (const StagnationStop&) {
    return 1;
  } catch (const core::ShutdownRequested&) {
    return 2;
  } catch (...) {
    return 3;
  }
}

/** Severity of a stored exception (0 for none). */
inline int exception_severity(const std::exception_ptr& e) noexcept
{
  if (!e) return 0;
  try {
    std::rethrow_exception(e);
  } catch (...) {
    return exception_severity_of_current();
  }
}

/** Thread-safe slot for the exceptions of one parallel step: keeps the most
 *  severe, the first of equal severity. Failure path only, apart from
 *  has_error() (one atomic load). */
class ExceptionSlot {
public:
  /** Offer the exception being handled; call inside a catch handler. */
  void keep_current() noexcept
  {
    keep(std::current_exception(), exception_severity_of_current());
  }
  /** Offer a stored exception. */
  void keep(std::exception_ptr e) noexcept
  {
    const int severity = exception_severity(e);
    keep(std::move(e), severity);
  }

  bool has_error() const noexcept { return set_.load(std::memory_order_acquire); }

  /** Rethrow the kept exception, if any, and clear the slot. */
  void rethrow_if_error()
  {
    std::exception_ptr e;
    {
      std::lock_guard<std::mutex> lk(m_);
      e = error_;
      error_ = nullptr;
      severity_ = 0;
      set_.store(false, std::memory_order_release);
    }
    if (e) std::rethrow_exception(e);
  }

private:
  void keep(std::exception_ptr e, int severity) noexcept
  {
    if (!e) return;
    std::lock_guard<std::mutex> lk(m_);
    if (severity > severity_) {
      error_    = std::move(e);
      severity_ = severity;
      set_.store(true, std::memory_order_release);
    }
  }

  mutable std::mutex m_;
  std::exception_ptr error_;
  int                severity_ = 0;   // of error_; 0 = none
  std::atomic<bool>  set_{false};
};

}  // namespace tadah::hpo

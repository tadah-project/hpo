#pragma once
/* hpo_mpi.h -- narrow MPI shim for the parallel annealing drivers
 * (TADAH_HPO_MPI, the only MPI build of Tadah). Apart from MPI start-up
 * and shutdown in the `tadah` main(), this header and the ANNEAL_PT /
 * ANNEAL_PA drivers are the ONLY places allowed to call MPI: training
 * stays rank-local (plain Trainer) and LAMMPS instances run on
 * MPI_COMM_SELF.
 *
 * Non-MPI builds get inline no-ops (rank 0 / size 1), so the drivers
 * compile identically with zero MPI dependence and the single-rank
 * code path is the thread-parallel one.
 *
 * All wrappers speak plain doubles / vectors -- no MPI types leak into
 * the drivers. Collective ordering contract: every rank calls the same
 * shim functions in the same order per epoch (stop decisions are
 * allreduced, so ranks always agree on loop exits).
 */
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TADAH_HPO_MPI
#include <mpi.h>
#endif

namespace tadah::hpo {

#ifdef TADAH_HPO_MPI

inline constexpr bool hpo_mpi_enabled = true;

/* The MPI C interfaces used here take element counts and vector
 * displacements as `int`.  Never narrow a container size silently: a
 * wrapped negative count can turn a bounded message into undefined buffer
 * access inside MPI.  The displacement helper also checks the cumulative
 * one-past-end count, so no intermediate or final sum can overflow `int`. */
inline int hpo_mpi_checked_count(std::size_t count, const char* operation)
{
  const std::size_t limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (count > limit)
    throw std::length_error(
        std::string(operation) + ": element count " +
        std::to_string(count) + " exceeds the MPI int limit " +
        std::to_string(limit));
  return static_cast<int>(count);
}

inline std::vector<int> hpo_mpi_checked_displacements(
    const std::vector<int>& counts, const char* operation)
{
  const std::size_t limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::vector<int> displacements(counts.size(), 0);
  std::size_t total = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] < 0)
      throw std::invalid_argument(
          std::string(operation) + ": negative per-rank MPI count");
    displacements[i] = hpo_mpi_checked_count(total, operation);
    const std::size_t count = static_cast<std::size_t>(counts[i]);
    if (count > limit - total)
      throw std::length_error(
          std::string(operation) +
          ": aggregate count exceeds the MPI int displacement limit");
    total += count;
  }
  return displacements;
}

/* A process linked against the MPI-enabled HPO library need not
 * initialise MPI (a program embedding tadah.hpo).  Keep the shim a true
 * no-op until MPI is live, so such a process takes the single-process path
 * instead of making illegal pre-init MPI calls.  tadah's main() and the
 * test runners (test_hpo, test_tadah) do initialise it, a one-rank job
 * outside mpiexec: the LAMMPS instances of this build run on
 * MPI_COMM_SELF (hpo_lammps_comm()), which needs MPI. */
inline bool hpo_mpi_active()
{
  int init = 0, final = 0;
  MPI_Initialized(&init);
  if (init) MPI_Finalized(&final);
  return init && !final;
}
inline int hpo_rank()
{
  if (!hpo_mpi_active()) return 0;
  int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;
}
inline int hpo_size()
{
  if (!hpo_mpi_active()) return 1;
  int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s;
}
inline void hpo_barrier()
{
  if (hpo_mpi_active()) MPI_Barrier(MPI_COMM_WORLD);
}

/* Communicator handed to every LAMMPS instance: each rank runs its own
 * single-rank LAMMPS objects (the serial-build equivalent of comm 0).
 * LammpsRunner::create_instance passes it; no caller chooses one (the
 * serial build's 0 is a null communicator here). */
inline MPI_Comm hpo_lammps_comm() { return MPI_COMM_SELF; }

/* ".r<rank>" for every rank-local fixed-filename output when running
 * multi-rank.  Rank 0 is suffixed too: the unsuffixed namespace is reserved
 * for the canonical, cross-rank best potential (best_pot.tadah). */
inline std::string hpo_rank_suffix()
{
  /* Do not cache this result: callers may legitimately query it before
   * MPI_Init (the serial-looking answer is then empty) and again after MPI
   * becomes live.  A function-static value would freeze the pre-init answer
   * for the lifetime of the process. */
  return hpo_size() > 1
      ? ".r" + std::to_string(hpo_rank()) : std::string{};
}

inline void hpo_bcast(std::vector<double>& v, int root = 0)
{
  if (!hpo_mpi_active()) return;
  const int count = hpo_mpi_checked_count(v.size(), "MPI_Bcast<double>");
  MPI_Bcast(v.data(), count, MPI_DOUBLE, root, MPI_COMM_WORLD);
}
inline void hpo_bcast(int& v, int root = 0)
{
  if (!hpo_mpi_active()) return;
  MPI_Bcast(&v, 1, MPI_INT, root, MPI_COMM_WORLD);
}
inline void hpo_bcast(std::vector<int>& v, int root = 0)
{
  if (!hpo_mpi_active()) return;
  const int count = hpo_mpi_checked_count(v.size(), "MPI_Bcast<int>");
  MPI_Bcast(v.data(), count, MPI_INT, root, MPI_COMM_WORLD);
}

inline std::uint64_t hpo_allreduce_sum(std::uint64_t v)
{
  if (!hpo_mpi_active()) return v;
  std::uint64_t out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  return out;
}
inline int hpo_allreduce_max(int v)
{
  if (!hpo_mpi_active()) return v;
  int out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return out;
}
inline std::size_t hpo_allreduce_sum_sz(std::size_t v)
{
  return static_cast<std::size_t>(
      hpo_allreduce_sum(static_cast<std::uint64_t>(v)));
}

/* (value, owner-rank) minimum across ranks. */
inline void hpo_allreduce_minloc(double& val, int& owner)
{
  if (!hpo_mpi_active()) { owner = 0; return; }
  struct { double v; int r; } in{val, hpo_rank()}, out{};
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  val   = out.v;
  owner = out.r;
}

inline void hpo_sendrecv(const std::vector<double>& send,
                         std::vector<double>& recv, int peer, int tag)
{
  if (!hpo_mpi_active()) { recv = send; return; }
  const int send_count =
      hpo_mpi_checked_count(send.size(), "MPI_Sendrecv send");
  const int recv_count =
      hpo_mpi_checked_count(recv.size(), "MPI_Sendrecv receive");
  MPI_Sendrecv(send.data(), send_count, MPI_DOUBLE, peer, tag,
               recv.data(), recv_count, MPI_DOUBLE, peer, tag,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Variable-count gather/scatter of doubles (root = 0). `counts` are
 * per-rank element counts, identical on every rank. */
inline void hpo_gatherv(const std::vector<double>& local,
                        std::vector<double>& all,
                        const std::vector<int>& counts)
{
  if (!hpo_mpi_active()) { all = local; return; }
  if (counts.size() != static_cast<std::size_t>(hpo_size()))
    throw std::invalid_argument(
        "MPI_Gatherv: counts size must equal the MPI world size");
  const std::vector<int> displ =
      hpo_mpi_checked_displacements(counts, "MPI_Gatherv");
  const int local_count =
      hpo_mpi_checked_count(local.size(), "MPI_Gatherv local");
  MPI_Gatherv(local.data(), local_count, MPI_DOUBLE,
              all.data(), counts.data(), displ.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
}
inline void hpo_scatterv(const std::vector<double>& all,
                         std::vector<double>& local,
                         const std::vector<int>& counts)
{
  if (!hpo_mpi_active()) { local = all; return; }
  if (counts.size() != static_cast<std::size_t>(hpo_size()))
    throw std::invalid_argument(
        "MPI_Scatterv: counts size must equal the MPI world size");
  const std::vector<int> displ =
      hpo_mpi_checked_displacements(counts, "MPI_Scatterv");
  const int local_count =
      hpo_mpi_checked_count(local.size(), "MPI_Scatterv local");
  MPI_Scatterv(all.data(), counts.data(), displ.data(), MPI_DOUBLE,
               local.data(), local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

#else  /* ------------- non-MPI build: inline no-ops ------------------ */

inline constexpr bool hpo_mpi_enabled = false;
inline constexpr bool hpo_mpi_active() { return false; }

inline int hpo_rank() { return 0; }
inline int hpo_size() { return 1; }
inline void hpo_barrier() {}
inline int hpo_lammps_comm() { return 0; }
inline std::string hpo_rank_suffix()
{
  return {};
}

inline void hpo_bcast(std::vector<double>&, int = 0) {}
inline void hpo_bcast(int&, int = 0) {}
inline void hpo_bcast(std::vector<int>&, int = 0) {}
inline std::uint64_t hpo_allreduce_sum(std::uint64_t v) { return v; }
inline int hpo_allreduce_max(int v) { return v; }
inline std::size_t hpo_allreduce_sum_sz(std::size_t v) { return v; }
inline void hpo_allreduce_minloc(double&, int& owner) { owner = 0; }
inline void hpo_sendrecv(const std::vector<double>&, std::vector<double>&,
                         int, int) {}
inline void hpo_gatherv(const std::vector<double>& local,
                        std::vector<double>& all,
                        const std::vector<int>&) { all = local; }
inline void hpo_scatterv(const std::vector<double>& all,
                         std::vector<double>& local,
                         const std::vector<int>&) { local = all; }

#endif

/* The error each rank throws after a startup step that every rank makes
 * on its own and that can fail on one rank alone (a rank-suffixed file it
 * cannot write): the rank's own error where the step failed on it,
 * "<what> failed on another MPI rank" where it failed on another rank
 * alone (@p any_failed: the ranks' failure flags combined), "" (no error)
 * where it failed on no rank. */
inline std::string agreed_startup_error(const std::string& local_error,
                                        bool any_failed,
                                        const std::string& what)
{
    if (!any_failed) return {};
    return local_error.empty() ? what + " failed on another MPI rank"
                               : local_error;
}

/* Makes @p step on this rank and throws, on every rank, when it failed on
 * any rank (agreed_startup_error): one collective, which every rank reaches
 * whether or not its step failed, so no rank waits in a later collective
 * for a peer that stopped. A serial run is one rank. */
template <class Step>
void run_on_every_rank(const std::string& what, Step&& step)
{
    std::string local_error;
    try {
        step();
    } catch (const std::exception& e) {
        local_error = e.what();
    }
    const std::string error = agreed_startup_error(
        local_error, hpo_allreduce_max(local_error.empty() ? 0 : 1) != 0,
        what);
    if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace tadah::hpo

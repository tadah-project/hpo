#pragma once
/* hpo_mpi.h -- narrow MPI shim for the parallel annealing drivers
 * (TADAH_HPO_MPI builds). This header and the ANNEAL_PT / ANNEAL_PA
 * drivers are the ONLY places allowed to reference MPI: training stays
 * rank-local (plain Trainer), LAMMPS instances run on MPI_COMM_SELF,
 * and the stale ScaLAPACK TADAH_BUILD_MPI machinery is untouched.
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
#include <string>
#include <vector>

#ifdef TADAH_HPO_MPI
#include <mpi.h>
#endif

namespace tadah::hpo {

#ifdef TADAH_HPO_MPI

inline constexpr bool hpo_mpi_enabled = true;

inline int hpo_rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
inline int hpo_size() { int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }

/* Communicator handed to every LAMMPS instance: each rank runs its own
 * single-rank LAMMPS objects (the serial-build equivalent of comm 0). */
inline MPI_Comm hpo_lammps_comm() { return MPI_COMM_SELF; }

/* ".r<rank>" for every fixed-filename output when running multi-rank;
 * rank 0 keeps the unsuffixed names so single-rank tooling continues
 * to find them. */
inline const std::string& hpo_rank_suffix()
{
  static const std::string s = [] {
    int init = 0;
    MPI_Initialized(&init);
    if (!init) return std::string{};
    return hpo_size() > 1 && hpo_rank() > 0
        ? ".r" + std::to_string(hpo_rank()) : std::string{};
  }();
  return s;
}

inline void hpo_bcast(std::vector<double>& v, int root = 0)
{
  MPI_Bcast(v.data(), static_cast<int>(v.size()), MPI_DOUBLE, root,
            MPI_COMM_WORLD);
}
inline void hpo_bcast(int& v, int root = 0)
{
  MPI_Bcast(&v, 1, MPI_INT, root, MPI_COMM_WORLD);
}
inline void hpo_bcast(std::vector<int>& v, int root = 0)
{
  MPI_Bcast(v.data(), static_cast<int>(v.size()), MPI_INT, root,
            MPI_COMM_WORLD);
}

inline std::uint64_t hpo_allreduce_sum(std::uint64_t v)
{
  std::uint64_t out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  return out;
}
inline int hpo_allreduce_max(int v)
{
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
  struct { double v; int r; } in{val, hpo_rank()}, out{};
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  val   = out.v;
  owner = out.r;
}

inline void hpo_sendrecv(const std::vector<double>& send,
                         std::vector<double>& recv, int peer, int tag)
{
  MPI_Sendrecv(send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
               peer, tag,
               recv.data(), static_cast<int>(recv.size()), MPI_DOUBLE,
               peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Variable-count gather/scatter of doubles (root = 0). `counts` are
 * per-rank element counts, identical on every rank. */
inline void hpo_gatherv(const std::vector<double>& local,
                        std::vector<double>& all,
                        const std::vector<int>& counts)
{
  std::vector<int> displ(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i)
    displ[i] = displ[i - 1] + counts[i - 1];
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
              all.data(), counts.data(), displ.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
}
inline void hpo_scatterv(const std::vector<double>& all,
                         std::vector<double>& local,
                         const std::vector<int>& counts)
{
  std::vector<int> displ(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i)
    displ[i] = displ[i - 1] + counts[i - 1];
  MPI_Scatterv(all.data(), counts.data(), displ.data(), MPI_DOUBLE,
               local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
               0, MPI_COMM_WORLD);
}

#else  /* ------------- non-MPI build: inline no-ops ------------------ */

inline constexpr bool hpo_mpi_enabled = false;

inline int hpo_rank() { return 0; }
inline int hpo_size() { return 1; }
inline int hpo_lammps_comm() { return 0; }
inline const std::string& hpo_rank_suffix()
{
  static const std::string s;
  return s;
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

}  // namespace tadah::hpo

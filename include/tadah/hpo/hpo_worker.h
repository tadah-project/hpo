#ifndef HPO_WORKER_H
#define HPO_WORKER_H

#ifdef TADAH_BUILD_MPI
#include <mpi.h>

namespace tadah {
namespace hpo {

class HPO_Worker {
  const static int CONFIG_TAG = 4;
  const static int WAIT_TAG = 3;
  const static int RELEASE_TAG = 2;
  const static int DATA_TAG = 1;
  const static int WORK_TAG = 0;
  int rank;
  int ncpu;
  MPI_Group &world_group;

  public:
  HPO_Worker(int rank, int ncpu, MPI_Group &world_group);

  bool use_lammps;
  void runLammps();
};
} // namespace hpo
} // namespace tadah
#endif

#endif


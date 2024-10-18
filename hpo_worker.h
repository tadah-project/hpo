#ifndef HPO_WORKER_H
#define HPO_WORKER_H

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif
#include "lammps_sim.h"

class HPO_Worker {
  const static int CONFIG_TAG = 4;
  const static int WAIT_TAG = 3;
  const static int RELEASE_TAG = 2;
  const static int DATA_TAG = 1;
  const static int WORK_TAG = 0;
  int rank;
  int ncpu;
  MPI_Group &world_group;
  std::vector<LammpsSim> LS;
  public:
  HPO_Worker(int rank, int ncpu, MPI_Group &world_group);

  void run();
};
#endif


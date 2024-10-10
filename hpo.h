#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include "../CORE/config/config.h"

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#else 
#define MPI_Comm int
#endif

int hpo_run(Config&, std::string &trg_file,
    std::string &val_file, MPI_Comm &lmp_comm);

/** Run optimisation of the hyperparameters.
 *
 */

#endif

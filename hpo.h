#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include "../CORE/config/config.h"

#ifdef TADAH_ENABLE_MPI
#include <mpi.h>
#endif

#include TADAH_LMPD
/** Run optimisation of the hyperparameters.
 *
 */

int hpo_run(Config&, std::string &trg_file,
    std::string &val_file, MPI_Comm &lmp_comm);
#endif

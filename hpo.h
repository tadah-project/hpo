#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include "../CORE/config/config.h"

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
int hpo_run(Config&, std::string &trg_file,
    std::string &val_file, MPI_Comm &lmp_comm);
#else 
#include TADAH_LMPA
#include TADAH_LMPB
#include TADAH_LMPC
int hpo_run(Config&, std::string &trg_file,
    std::string &val_file);
#endif


/** Run optimisation of the hyperparameters.
 *
 */

#endif

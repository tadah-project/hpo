#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include "../CORE/config/config.h"

/** Run optimisation of the hyperparameters.
 *
 */
int hpo_run(Config&, std::string &trg_file,
    std::string &val_file, MPI_Comm &lmp_comm);
#endif

#ifndef TADAH_HPO_H
#define TADAH_HPO_H

#include "../CORE/config/config.h"

/** Run optimisation of the hyperparameters.
 *
 *  This is experimental...
 *
 *
 */
int hpo_run(Config&, std::string &);

int hpo_run2(Config&, std::string &);
int hpo_mpi_run(Config&, std::string &);
#endif

#pragma once
/*
 * check_summary.h -- what `tadah hpo --check` prints.
 *
 * hpo_run() with CHECK true (`tadah hpo --check`, CHECK true in a TASK hpo
 * section) makes every startup check of the run and, when they pass,
 * writes this summary to standard output:
 *
 *   tadah hpo --check: config.hpo passed the startup checks.
 *
 *   Search parameters: 2
 *     RCUT2B   [4, 6]  LIN
 *     SGRIDMB  [0.1, 0.4]  LIN
 *
 *   Optimiser: STRATEGY TYPE PRESTAGE
 *     EXPLORE   NLOPT:GN_CRS2_LM; stops: MAXEVAL 60
 *     STRATEGY  DLIB:BFGS; stops: MAXEVAL 30; TOPK 3
 *
 *   Loss terms: LOSS L2
 *     PC_ERMSE  target 0, weight 0.0001
 *     PC_LAMMPS ../common/lammps/Ecoh.in: Ecoh  target 8.1, weight 800
 *
 *   Data: 167 training structures, 29 validation structures
 *
 *   No candidate was evaluated and no file was written.
 */

#include <cstddef>
#include <ostream>
#include <string>

namespace tadah::hpo {

class HPOTargets;

/** Writes the summary above to @p out: the target file @p target_file;
 *  every search parameter (the OPTIM dimensions in the optimiser's order)
 *  with its box in real units, its scale and, for a box an AUTO rule
 *  derived, the rule; the STRATEGY TYPE and each block that runs a search
 *  -- EXPLORE (PRESTAGE), STRATEGY, the INNER block of either -- with its
 *  LIB:ALGO and stops (MAXEVAL, MAXTIME, STOPVAL, FTOL_REL, FTOL_ABS,
 *  XTOL_REL, XTOL_ABS as set, "none set" without one), N_STARTS and
 *  BUDGET_PER_START (MULTISTART), TOPK (PRESTAGE) and the RESTART block;
 *  the LOSS and every loss term with its target and weight (the validation
 *  metrics, an indexed PC_ERMSE or PC_FRMSE per dataset, each --varloss of
 *  a PC_LAMMPS line, each PC_OBSERVABLE); and the numbers of training
 *  (@p n_train) and validation (@p n_val) structures loaded. @p trg is the
 *  parsed target file after the run's evaluation host resolved its
 *  validation targets. */
void write_check_summary(std::ostream& out, const std::string& target_file,
                         const HPOTargets& trg, std::size_t n_train,
                         std::size_t n_val);

}  // namespace tadah::hpo

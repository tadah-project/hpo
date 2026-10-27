#pragma once
/*
 * check_summary.h -- what `tadah hpo --check` prints.
 *
 * hpo_run() with CHECK true (`tadah hpo --check`, CHECK true in a
 * configuration file without WEIGHTS) makes every startup check of the run
 * and, when they pass,
 * writes this summary to standard output:
 *
 *   tadah hpo --check: config.hpo passed the startup checks, with 1 warning.
 *
 *   Search parameters: 2
 *     RCUT2B   [4, 6]  LIN
 *     SGRIDMB  [0.1, 0.4]  LIN
 *
 *   Optimiser: STRATEGY TYPE PRESTAGE
 *     EXPLORE   NLOPT:GN_CRS2_LM; stops: MAXEVAL 60
 *     STRATEGY  NLOPT:LN_BOBYQA; stops: MAXEVAL 30; TOPK 3
 *
 *   Loss terms:
 *     PC_ERMSE  target 0, weight 0.0001, loss L2
 *     PC_LAMMPS ../common/lammps/Ecoh.in: Ecoh  target 8.1, weight 800, loss L2
 *
 *   Data: 167 training structures, 29 validation structures
 *
 *   Warnings and hints, which a run writes to its journal (hpo_run.log):
 *     WARNING  THREADS not set: the pc_lammps fan-out and MLIP ...
 *
 *   No candidate was evaluated and no file was written.
 */

#include <tadah/hpo/hpo_log.h>   // JournalNote

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace tadah::hpo {

class HPOTargets;

/** Writes the section of the summary that lists @p notes, the WARNING and
 *  HINT lines a run writes to its journal at startup
 *  (HPOLog::notes_without_journal()): "Warnings and hints, which a run
 *  writes to its journal (hpo_run.log):", then one entry per note, its tag
 *  in a column of its own ("  WARNING  ", "  HINT     ") and each further
 *  line of its text under the first. Nothing for no note. */
void write_check_notes(std::ostream& out,
                       const std::vector<JournalNote>& notes);

/** Writes the summary above to @p out: the target file @p target_file;
 *  every search parameter (the OPTIM dimensions in the optimiser's order)
 *  with its box in real units, its scale and, for a box an AUTO rule
 *  derived, the rule; the STRATEGY TYPE and each block that runs a search
 *  -- EXPLORE (PRESTAGE), STRATEGY, the INNER block of either -- with its
 *  LIB:ALGO and stops (MAXEVAL, MAXTIME, STOPVAL, FTOL_REL, FTOL_ABS,
 *  XTOL_REL, XTOL_ABS as set, "none set" without one), N_STARTS and
 *  BUDGET_PER_START (MULTISTART), TOPK (PRESTAGE) and the RESTART block;
 *  every loss term with its target, weight and loss function (the
 *  validation metrics, which take the last LOSS line's, an indexed
 *  PC_ERMSE or PC_FRMSE per dataset, each --varloss of a PC_LAMMPS line,
 *  each PC_OBSERVABLE, which take their own or the LOSS line's above
 *  them); the numbers of training
 *  (@p n_train) and validation (@p n_val) structures loaded; and the
 *  WARNING and HINT lines a run writes to its journal at startup
 *  (@p notes: write_check_notes), which the first line counts ("passed the
 *  startup checks, with 2 warnings and 1 hint."). @p trg is the parsed
 *  target file after the run's evaluation host resolved its validation
 *  targets. */
void write_check_summary(std::ostream& out, const std::string& target_file,
                         const HPOTargets& trg, std::size_t n_train,
                         std::size_t n_val,
                         const std::vector<JournalNote>& notes);

}  // namespace tadah::hpo

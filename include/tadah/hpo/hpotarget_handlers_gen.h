// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: HPO/blueprint/hpotarget.toml and the part files of its
//                  meta.parts, read by HPO/blueprint/hpotarget_registry.py
// Generator:       HPO/blueprint/gen_hpotarget.py --header
//                  (build target hpo_regenerate_hpotarget_registry)
// Drift gate:      ctest hpo_hpotarget_registry_sync (gen_hpotarget.py --check)
//
// The handlers of the registry-driven parser of the target file
// (blueprint/README.md, "The handlers"): one for each key, and for
// each line option and form of the lines HPO reads, each named as its
// enumerator. src/hpotarget_handlers.cpp defines them; the tables
// hold their addresses in the order of the enumerations, so a key the
// registry drops leaves a definition that names no member (the
// handlers do not compile), and a key it adds a member no file
// defines (the parser does not link). The options and forms of the
// OPTIM line, which MLIP's OPTIM reader reads, have none (nullptr).
#pragma once

#include <tadah/hpo/hpotarget_registry_gen.h>

namespace tadah {
namespace hpo {
namespace hpotarget_registry {

/* What a handler reads (src/hpotarget_reads.h defines them): the line of a
 * key, an option of a line, the line of a form. */
struct KeyRead;
struct OptionRead;
struct FormRead;

struct Handlers {
  static void ROOT_FAILSCORE(KeyRead& r);
  static void ROOT_PC_SCRIPT_TIMEOUT(KeyRead& r);
  static void ROOT_MAXTIME(KeyRead& r);
  static void ROOT_PC_ISOLATION(KeyRead& r);
  static void ROOT_PC_WORKER_TIMEOUT(KeyRead& r);
  static void ROOT_FAILSCORE_GRADING(KeyRead& r);
  static void ROOT_BUMPGATE(KeyRead& r);
  static void ROOT_RHOGATE(KeyRead& r);
  static void ROOT_RHOBARGATE(KeyRead& r);
  static void ROOT_QFLUSH(KeyRead& r);
  static void ROOT_OUTPUT(KeyRead& r);
  static void ROOT_BOUTPUT(KeyRead& r);
  static void ROOT_DIGITS(KeyRead& r);
  static void ROOT_DUMP(KeyRead& r);
  static void ROOT_BDUMP(KeyRead& r);
  static void ROOT_OPTIM_SCALE(KeyRead& r);
  static void OPTIMIZER_LIB(KeyRead& r);
  static void OPTIMIZER_ALGO(KeyRead& r);
  static void OPTIMIZER_MAXEVAL(KeyRead& r);
  static void OPTIMIZER_MAXTIME(KeyRead& r);
  static void OPTIMIZER_STOPVAL(KeyRead& r);
  static void OPTIMIZER_FTOL_REL(KeyRead& r);
  static void OPTIMIZER_FTOL_ABS(KeyRead& r);
  static void OPTIMIZER_XTOL_REL(KeyRead& r);
  static void OPTIMIZER_XTOL_ABS(KeyRead& r);
  static void OPTIMIZER_POPULATION(KeyRead& r);
  static void OPTIMIZER_STEP(KeyRead& r);
  static void OPTIMIZER_SEED(KeyRead& r);
  static void OPTIMIZER_VECTOR_ARR(KeyRead& r);
  static void OPTIMIZER_THREADS(KeyRead& r);
  static void OPTIMIZER_GRAD_STRATEGY(KeyRead& r);
  static void OPTIMIZER_PARAM(KeyRead& r);
  static void OPTIMIZER_LOG_HP(KeyRead& r);
  static void STRATEGY_TYPE(KeyRead& r);
  static void STRATEGY_N_STARTS(KeyRead& r);
  static void STRATEGY_BUDGET_PER_START(KeyRead& r);
  static void STRATEGY_TOPK(KeyRead& r);
  static void INIT_METHOD(KeyRead& r);
  static void INIT_K_TOTAL(KeyRead& r);
  static void INIT_SEED(KeyRead& r);
  static void INIT_WARM_ON_MISMATCH(KeyRead& r);
  static void NOISE_AUTO(KeyRead& r);
  static void NOISE_CALIBRATION(KeyRead& r);
  static void RESTART_AFTER_STAGNATION(KeyRead& r);
  static void RESTART_MAX(KeyRead& r);
  static void ROOT_PC_EMAE(KeyRead& r);
  static void ROOT_PC_ERMSE(KeyRead& r);
  static void ROOT_PC_ErRMSE(KeyRead& r);
  static void ROOT_PC_ERSQ(KeyRead& r);
  static void ROOT_PC_SRMSE_GPA(KeyRead& r);
  static void ROOT_PC_FRMSE(KeyRead& r);
  static void ROOT_PC_LAMMPS(KeyRead& r);
  static void ROOT_PC_FAILPROBE(KeyRead& r);
  static void ROOT_PC_OBSERVABLE(KeyRead& r);
  static void ROOT_LOSS(KeyRead& r);
  static void ROOT_OPTIM(KeyRead& r);
  static void RHOGATE_R_LO(OptionRead& r);
  static void RHOGATE_NPTS(OptionRead& r);
  static void RHOGATE_SEED(OptionRead& r);
  static void RHOBARGATE_NPTS(OptionRead& r);
  static void RHOBARGATE_SEED(OptionRead& r);
  static void PC_EMAE_skipabove(OptionRead& r);
  static void PC_EMAE_skiprel(OptionRead& r);
  static void PC_LAMMPS_script(OptionRead& r);
  static void PC_LAMMPS_varloss(OptionRead& r);
  static void PC_LAMMPS_invar(OptionRead& r);
  static void PC_LAMMPS_outvar(OptionRead& r);
  static void PC_LAMMPS_failscore(OptionRead& r);
  static void PC_LAMMPS_maxtime(OptionRead& r);
  static void PC_LAMMPS_hotvar(OptionRead& r);
  static void PC_LAMMPS_failvar(OptionRead& r);
  static void PC_LAMMPS_ncpu(OptionRead& r);
  static void PC_FAILPROBE_gate(OptionRead& r);
  static void PC_FAILPROBE_s0(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_species(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_target(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_weight(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_hotvar(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_lattice(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_supercell(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_loss(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_maxiter(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_maxeval(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_maxwalltime(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_etol(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_fmaxconverged(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_ptol(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy_unconverged(OptionRead& r);
  static void PC_OBSERVABLE_cohesive_energy(FormRead& r);
};

using KeyHandler = void (*)(KeyRead&);
using OptionHandler = void (*)(OptionRead&);
using FormHandler = void (*)(FormRead&);

/* kKeyHandlers[i] reads the line of kKeys[i], kOptionHandlers[i] the option
 * kLineOptions[i], kFormHandlers[i] the line of kForms[i]. */
inline constexpr KeyHandler kKeyHandlers[] = {
  &Handlers::ROOT_FAILSCORE,
  &Handlers::ROOT_PC_SCRIPT_TIMEOUT,
  &Handlers::ROOT_MAXTIME,
  &Handlers::ROOT_PC_ISOLATION,
  &Handlers::ROOT_PC_WORKER_TIMEOUT,
  &Handlers::ROOT_FAILSCORE_GRADING,
  &Handlers::ROOT_BUMPGATE,
  &Handlers::ROOT_RHOGATE,
  &Handlers::ROOT_RHOBARGATE,
  &Handlers::ROOT_QFLUSH,
  &Handlers::ROOT_OUTPUT,
  &Handlers::ROOT_BOUTPUT,
  &Handlers::ROOT_DIGITS,
  &Handlers::ROOT_DUMP,
  &Handlers::ROOT_BDUMP,
  &Handlers::ROOT_OPTIM_SCALE,
  &Handlers::OPTIMIZER_LIB,
  &Handlers::OPTIMIZER_ALGO,
  &Handlers::OPTIMIZER_MAXEVAL,
  &Handlers::OPTIMIZER_MAXTIME,
  &Handlers::OPTIMIZER_STOPVAL,
  &Handlers::OPTIMIZER_FTOL_REL,
  &Handlers::OPTIMIZER_FTOL_ABS,
  &Handlers::OPTIMIZER_XTOL_REL,
  &Handlers::OPTIMIZER_XTOL_ABS,
  &Handlers::OPTIMIZER_POPULATION,
  &Handlers::OPTIMIZER_STEP,
  &Handlers::OPTIMIZER_SEED,
  &Handlers::OPTIMIZER_VECTOR_ARR,
  &Handlers::OPTIMIZER_THREADS,
  &Handlers::OPTIMIZER_GRAD_STRATEGY,
  &Handlers::OPTIMIZER_PARAM,
  &Handlers::OPTIMIZER_LOG_HP,
  &Handlers::STRATEGY_TYPE,
  &Handlers::STRATEGY_N_STARTS,
  &Handlers::STRATEGY_BUDGET_PER_START,
  &Handlers::STRATEGY_TOPK,
  &Handlers::INIT_METHOD,
  &Handlers::INIT_K_TOTAL,
  &Handlers::INIT_SEED,
  &Handlers::INIT_WARM_ON_MISMATCH,
  &Handlers::NOISE_AUTO,
  &Handlers::NOISE_CALIBRATION,
  &Handlers::RESTART_AFTER_STAGNATION,
  &Handlers::RESTART_MAX,
  &Handlers::ROOT_PC_EMAE,
  &Handlers::ROOT_PC_ERMSE,
  &Handlers::ROOT_PC_ErRMSE,
  &Handlers::ROOT_PC_ERSQ,
  &Handlers::ROOT_PC_SRMSE_GPA,
  &Handlers::ROOT_PC_FRMSE,
  &Handlers::ROOT_PC_LAMMPS,
  &Handlers::ROOT_PC_FAILPROBE,
  &Handlers::ROOT_PC_OBSERVABLE,
  &Handlers::ROOT_LOSS,
  &Handlers::ROOT_OPTIM,
};

inline constexpr OptionHandler kOptionHandlers[] = {
  &Handlers::RHOGATE_R_LO,
  &Handlers::RHOGATE_NPTS,
  &Handlers::RHOGATE_SEED,
  &Handlers::RHOBARGATE_NPTS,
  &Handlers::RHOBARGATE_SEED,
  &Handlers::PC_EMAE_skipabove,
  &Handlers::PC_EMAE_skiprel,
  &Handlers::PC_LAMMPS_script,
  &Handlers::PC_LAMMPS_varloss,
  &Handlers::PC_LAMMPS_invar,
  &Handlers::PC_LAMMPS_outvar,
  &Handlers::PC_LAMMPS_failscore,
  &Handlers::PC_LAMMPS_maxtime,
  &Handlers::PC_LAMMPS_hotvar,
  &Handlers::PC_LAMMPS_failvar,
  &Handlers::PC_LAMMPS_ncpu,
  &Handlers::PC_FAILPROBE_gate,
  &Handlers::PC_FAILPROBE_s0,
  &Handlers::PC_OBSERVABLE_cohesive_energy_species,
  &Handlers::PC_OBSERVABLE_cohesive_energy_target,
  &Handlers::PC_OBSERVABLE_cohesive_energy_weight,
  &Handlers::PC_OBSERVABLE_cohesive_energy_hotvar,
  &Handlers::PC_OBSERVABLE_cohesive_energy_lattice,
  &Handlers::PC_OBSERVABLE_cohesive_energy_supercell,
  &Handlers::PC_OBSERVABLE_cohesive_energy_loss,
  &Handlers::PC_OBSERVABLE_cohesive_energy_maxiter,
  &Handlers::PC_OBSERVABLE_cohesive_energy_maxeval,
  &Handlers::PC_OBSERVABLE_cohesive_energy_maxwalltime,
  &Handlers::PC_OBSERVABLE_cohesive_energy_etol,
  &Handlers::PC_OBSERVABLE_cohesive_energy_fmaxconverged,
  &Handlers::PC_OBSERVABLE_cohesive_energy_ptol,
  &Handlers::PC_OBSERVABLE_cohesive_energy_unconverged,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

inline constexpr FormHandler kFormHandlers[] = {
  &Handlers::PC_OBSERVABLE_cohesive_energy,
  nullptr,
  nullptr,
};

}  // namespace hpotarget_registry
}  // namespace hpo
}  // namespace tadah

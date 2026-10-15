#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/training_datasets.h>
#include <tadah/mlip/engines/train_engine.h>

#include <tadah/hpo/hpo.h>
#include <tadah/hpo/engines/hpo_engine.h>

namespace tadah {
namespace hpo {
namespace engines {

int HPOEngine::run(mlip::Task &task) {

  /* hpo_run()'s status is the process exit status: 0 for a normal end,
   * 1 for a normal end in which no evaluation produced a usable loss,
   * 128 + signal after SIGINT/SIGTERM/SIGUSR2 (see run_driver.h). */
  int status = EXIT_SUCCESS;
  try {
    core::Context context = task.getContext();
    models::D_mJoin::expand_grids(context);

    const bool has_train_db =
        context.contains("DBFILE") && context.size("DBFILE") > 0;
    std::set<core::Element> unique_elements;
    if (has_train_db) {
      unique_elements = mlip::StructureDB::find_unique_elements(context);
      mlip::StructureDB::check_atom_key(context, unique_elements);
    } else {
      /* No training DBFILE (all weights pinned by FIXINDEX/FIXWEIGHT;
       * enforced by validate_hpo_setup): the element set must be
       * declared explicitly — validation is optional too, so it cannot
       * be derived from data. */
      if (!context.contains("ATOM")) {
        throw std::runtime_error(
            "No training DBFILE: the ATOM key is required in the training "
            "configuration to declare the element set (e.g. 'ATOM Ta'). " +
            mlip::potential_has_no_datasets_advice());
      }
      for (const auto &sym : context("ATOM"))
        unique_elements.insert(core::PeriodicTable::find_by_symbol(sym));
    }
    mlip::StructureDB::check_watom_key(context, unique_elements);

    core::Logger::instance().log(tadah::core::LogLevel::INFO, "Optimising HPs...");
    status = hpo_run(context, task.getKeyProvenance());

    // core::Logger::instance().log(
    //     tadah::core::LogLevel::ERROR,
    //     "------------------------------------------------\n",
    //     "This subcommand is not supported by this build.",
    //     "\nTadah! Must by compiled with HPO support.",
    //     "\nSee documentation for details.",
    //     "\n------------------------------------------------"
    //     );

  } catch (const std::exception &e) {
    std::cerr << "HPOEngine error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    /* hpo_run() rethrows a non-standard exception after writing its journal
     * trailer ("FAILED -- non-standard exception"); fail like any other
     * error instead of terminating. */
    std::cerr << "HPOEngine error: unknown (non-standard) exception" << std::endl;
    return EXIT_FAILURE;
  }
  return status;
}

void HPOEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating HPOEngine...");
  const auto &context = task.getContext();
  if (!context.get<bool>("INIT2B") && !context.get<bool>("INITMB")) {
    throw std::runtime_error(
        "HPOEngine requires INIT2B or/and INITMB parameters to be set");
  }

  // Early "critical-fixed-not-optimised" preview. The
  // canonical, HPOTargets-aware audit runs inside hpo_run() once the
  // HPOTARGET file has been parsed. Here we do a lightweight scan of
  // the raw HPOTARGET file for `OPTIM <KEY>` tokens -- enough to flag
  // a critical hyperparameter left out of the search (e.g. CGRIDMB fixed
  // at the seed and absent from every OPTIM line) before any structures
  // are loaded.
  // Always emits at WARNING; never throws -- the run-time audit will
  // emit again at FAIL severity when AUDIT=error if the issue stands.
  if (context.contains("HPOTARGET")) {
    std::vector<std::string> optim_keys;
    try {
      const std::string path = context.get<std::string>("HPOTARGET");
      std::ifstream in(path);
      std::string line;
      while (std::getline(in, line)) {
        // Skip blank/commented lines.
        std::size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos) continue;
        if (line[p] == '#') continue;
        // Match `OPTIM <KEY> ...` (case-sensitive -- matches
        // hpo_targets.cpp parsing).
        if (line.compare(p, 5, "OPTIM") != 0) continue;
        std::size_t q = p + 5;
        while (q < line.size() && std::isspace(static_cast<unsigned char>(line[q]))) ++q;
        std::size_t r = q;
        while (r < line.size() && !std::isspace(static_cast<unsigned char>(line[r]))) ++r;
        if (r > q) optim_keys.emplace_back(line.substr(q, r - q));
      }
    } catch (...) { /* unreadable HPOTARGET -- skip the preview */ }
    auto rep = tadah::mlip::run_critical_optim_audit(context, optim_keys);
    rep.log();   // never throws (run_critical_optim_audit emits WARN-only)
  }
}

} // namespace engines
} // namespace hpo
} // namespace tadah

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/training_datasets.h>
#include <tadah/mlip/engines/train_engine.h>

#include <tadah/hpo/hpo.h>
#include <tadah/hpo/engines/hpo_engine.h>
#include <tadah/hpo/startup_validation.h>

namespace tadah {
namespace hpo {
namespace engines {

int HPOEngine::run(mlip::Task &task) {

  /* hpo_run()'s status is the process exit status: 0 for a normal end,
   * 1 for a normal end in which no evaluation produced a usable loss,
   * 128 + signal after SIGINT/SIGTERM/SIGUSR2 (see run_driver.h). */
  int status = EXIT_SUCCESS;
  try {
    /* HPOTargets expands the grids (D_mJoin::expand_grids), and keeps what
     * it leaves out of a -c potential for its messages. */
    core::Context context = task.getContext();

    const bool has_train_db =
        context.contains("DBFILE") && context.size("DBFILE") > 0;
    std::set<core::Element> unique_elements;
    if (has_train_db) {
      unique_elements = mlip::StructureDB::find_unique_elements(context);
      mlip::StructureDB::check_atom_key(context, unique_elements);
    } else {
      /* No training DBFILE (all weights pinned by FIXINDEX/FIXWEIGHT;
       * enforced by validate_hpo_setup): the element set is the one ATOM
       * declares -- validation is optional too, so it cannot be derived
       * from data. validate_hpo_setup applies the same rule. */
      unique_elements = declared_element_set(context);
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

  /* The critical-fixed-not-optimised audit runs inside hpo_run() under
   * AUDIT warn and error alone, on the parsed search space (HPOTargets)
   * and the training data: no preview of it runs here. */
}

} // namespace engines
} // namespace hpo
} // namespace tadah

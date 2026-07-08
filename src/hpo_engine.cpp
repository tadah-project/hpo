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
#include <tadah/mlip/engines/train_engine.h>

#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_worker.h>
#include <tadah/hpo/engines/hpo_engine.h>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace hpo {
namespace engines {

int HPOEngine::run(mlip::Task &task) {

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
            "configuration to declare the element set (e.g. 'ATOM Ta').");
      }
      for (const auto &sym : context("ATOM"))
        unique_elements.insert(core::PeriodicTable::find_by_symbol(sym));
    }
    mlip::StructureDB::check_watom_key(context, unique_elements);

#ifdef TADAH_BUILD_MPI
    if (!has_train_db) {
      throw std::runtime_error(
          "HPO without a training DBFILE is serial-only: there is no design "
          "matrix to distribute. Use the serial (non-MPI) tadah binary.");
    }
    int rank = 0;
    int ncpu = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    // MPI_Status status;
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    if (rank==0) {
      if (ncpu<2) {
        core::Logger::instance().log(
            tadah::core::LogLevel::ERROR,
            "Minimum number of cpus for an MPI version is 2");
        return EXIT_FAILURE;
      }
      core::Logger::instance().log(tadah::core::LogLevel::INFO, "Optimising HPs...");
      hpo_run(context);

      // shut all workers
      int cont=0;
      MPI_Bcast(&cont, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else { // WORKER
      HPO_Worker hpoWorker(rank, ncpu, world_group);

      tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;
      tadah::mlip::memory::MLIPWorkspaceManager workspaceManager2;
      while (true) {
        int cs_size;
        MPI_Bcast(&cs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!cs_size)
          break;

        // Obtain updated config file from the HOST
        std::vector<char> serialized(cs_size);
        MPI_Bcast(serialized.data(), cs_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        core::Context workerContext;
        workerContext.deserialize(serialized);

        mlip::TrainerWorker worker(workerContext, workspaceManager, workspaceManager2);
        worker.run();
        worker.solve();

        // Run LAMMPS
        if (hpoWorker.use_lammps)
          hpoWorker.runLammps();
      } // end of WORKER external while loop
    }
    // Free the communicator and groups when done
    //MPI_Comm_free(&lmp_comm);
    MPI_Group_free(&world_group);
#else
    core::Logger::instance().log(tadah::core::LogLevel::INFO, "Optimising HPs...");
    hpo_run(context);
#endif  // TADAH_BUILD_MPI

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
  }
  return EXIT_SUCCESS;
}

void HPOEngine::validate(const tadah::mlip::Task &task) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
                             "Validating HPOEngine...");
  const auto &context = task.getContext();
  if (!context.get<bool>("INIT2B") && !context.get<bool>("INITMB")) {
    throw std::runtime_error(
        "HPOEngine requires INIT2B or/and INITMB parameters to be set");
  }

  // Phase 5 -- early "critical-fixed-not-optimised" preview. The
  // canonical, HPOTargets-aware audit runs inside hpo_run() once the
  // HPOTARGET file has been parsed. Here we do a lightweight scan of
  // the raw HPOTARGET file for `OPTIM <KEY>` tokens -- enough to flag
  // the BASINS-style failure mode (e.g. CGRIDMB fixed at the seed and
  // absent from every OPTIM line) before any structures are loaded.
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

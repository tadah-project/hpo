#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/output/output.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

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

#ifdef TADAH_BUILD_MPI
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
}

// helper function to dump the potential file
void HPOEngine::dumpPotFile(const std::string &outfile, const core::Context &context) {
  core::Logger::instance().log(tadah::core::LogLevel::INFO,
      "Saving parameters to file: " + outfile);
  std::ofstream outstream(outfile);
  if (outstream.is_open()) {
    outstream << context << std::endl;
  } else {
    tadah::core::Logger::instance().log(tadah::core::LogLevel::ERROR,
        "Error: Unable to open file: ", outfile);
  }
}


} // namespace engines
} // namespace hpo
} // namespace tadah

#include <tadah/hpo/hpo_worker.h>
#include <tadah/hpo/runners.h>
#ifdef TADAH_BUILD_MPI

namespace tadah {
namespace hpo {

HPO_Worker::HPO_Worker(int rank, int ncpu, MPI_Group &world_group):
  rank(rank),
  ncpu(ncpu),
  world_group(world_group)
{
  MPI_Bcast(&use_lammps, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
};

void HPO_Worker::run() {
  LammpsRunner runner;
  MPI_Status status;
  while (true) {
    MPI_Send(NULL, 0, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD);

    MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int sender = status.MPI_SOURCE;
    int tag = status.MPI_TAG;

    if (tag==WAIT_TAG) {
      MPI_Recv(NULL, 0, MPI_INT, 0, WAIT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (tag==WORK_TAG) {
      MPI_Group work_group;
      MPI_Comm work_comm;
      // obtain all ranks my work_group
      int s;
      MPI_Get_count(&status, MPI_INT, &s);
      std::vector<int> granks(s);
      MPI_Recv(granks.data(), s, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // create work_group and communicator for LAMMPS
      MPI_Group_incl(world_group, granks.size(), granks.data(), &work_group);
      MPI_Comm_create_group(MPI_COMM_WORLD, work_group, s /*tag*/, &work_comm);
      MPI_Comm_set_errhandler(work_comm, MPI_ERRORS_RETURN);
      // Get LAMMPS script name
      MPI_Probe(0, WORK_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, &s);
      char* buffer = new char[s];
      MPI_Recv(buffer, s, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::string script(buffer);
      delete[] buffer;

      // obtain invars
      // Receive the number of strings
      int num_strings;
      MPI_Recv(&num_strings, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<const char*> invar;
      std::vector<std::string> sinvar;
      // Receive each string
      for (int i = 0; i < num_strings; i++) {
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char* buffer = new char[length];
        MPI_Recv(buffer, length, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sinvar.push_back(std::string(buffer));
        delete[] buffer;
      }
      for (const auto& str : sinvar) {
        invar.push_back(str.c_str());
      }

      // obtain outvars
      // Receive the number of strings
      MPI_Recv(&num_strings, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<std::string> outvar;
      // Receive each string
      for (int i = 0; i < num_strings; i++) {
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char* buffer = new char[length];
        MPI_Recv(buffer, length, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        outvar.push_back(std::string(buffer));
        delete[] buffer;
      }

      // obtain varloss
      // Receive the number of strings
      MPI_Recv(&num_strings, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<std::string> varloss;
      // Receive each string
      for (int i = 0; i < num_strings; i++) {
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char* buffer = new char[length];
        MPI_Recv(buffer, length, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        varloss.push_back(std::string(buffer));
        delete[] buffer;
      }

      double failscore;
      MPI_Recv(&failscore, 1, MPI_DOUBLE, 0, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // LAMMPS simulation
      auto *lmp = runner.run(work_comm, script, invar);

      // local work_comm root sends data back to the host
      int lrank;
      MPI_Comm_rank(work_comm, &lrank);
      if (lrank==0) {
        std::vector<std::string> v_outvar(outvar.size());
        std::vector<double> v_varloss(varloss.size());
        if (!lmp) {
          for (size_t i = 0; i < varloss.size(); ++i)
            v_varloss[i] = failscore;
          for (size_t i = 0; i < outvar.size(); ++i)
            v_outvar[i] = "NaN";
        }
        else {
          for (size_t i = 0; i < varloss.size(); ++i) {
            const auto& v = varloss[i];
            v_varloss[i] = std::stod(runner.get_variable(lmp,v));
          }
          for (size_t i = 0; i < outvar.size(); ++i) {
            const auto& o = outvar[i];
            v_outvar[i] = runner.get_variable(lmp,o);
          }
        }
        MPI_Send(v_varloss.data(), v_varloss.size(), MPI_DOUBLE, 0, DATA_TAG, MPI_COMM_WORLD);

        for (const std::string &message : v_outvar) {
          int length = message.size() + 1; // +1 for the null terminator
          MPI_Send(&length, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD);
          MPI_Send(message.c_str(), length, MPI_CHAR, 0, DATA_TAG, MPI_COMM_WORLD);
        }
      }

      if (lmp) {
        delete lmp;
      }

      // Release 
      //MPI_Barrier(work_comm);
      MPI_Group_free(&work_group);
      MPI_Comm_free(&work_comm);
    }
    else if (tag==RELEASE_TAG) {
      MPI_Recv(NULL, 0, MPI_INT, 0, RELEASE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    }
  }

}

} // namespace hpo
} // namespace tadah
#endif

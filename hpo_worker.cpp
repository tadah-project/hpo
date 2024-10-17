#include "hpo_worker.h"
#include "runners.h"
HPO_Worker::HPO_Worker(int rank, int ncpu, MPI_Group &world_group):
  rank(rank),
  ncpu(ncpu),
  world_group(world_group)
{};

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
      // Perform LAMMPS simulation
      //for (const auto &tt: invar) std::cout << tt << " ";
      //std::cout << std::endl;
      std::cout << "D0" << rank << std::endl;
      auto *lmp = runner.run(work_comm, script, invar);
      int tloss = 0;
      std::cout << "D1" << rank << std::endl;
      if (!lmp) {
      std::cout << "D2" << rank << std::endl;
        //tloss+=trg.LS[0].failscore;
      }
      else {
      std::cout << "D3" << rank << std::endl;
        //for (const auto &v: trg.LS[0].varloss) {
        //  double lmp_tloss= std::stod(runner.get_variable(lmp,v));
        //  tloss+= lmp_tloss;
        //}
      }
      std::cout << "D4" << rank << std::endl;
      if (lmp) delete lmp;
      std::cout << "D5" << rank << std::endl;

      // Release 
      MPI_Group_free(&work_group);
      //MPI_Comm_free(&work_comm);
    }
    else if (tag==RELEASE_TAG) {
      MPI_Recv(NULL, 0, MPI_INT, 0, RELEASE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    }
  }

}

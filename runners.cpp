#include "runners.h"

using LMP = LammpsRunner::LMP; 
LammpsRunner::LammpsRunner() {}

LMP* LammpsRunner::run(MPI_Comm &sub_comm, std::string &script,
    std::vector<const char*> &lmp_tokens) {
  int lmpargc = static_cast<int>(lmp_tokens.size());
  LMP* lmp = nullptr;

  try {
    lmp = new LMP(lmpargc, const_cast<char**>(lmp_tokens.data()), sub_comm);
    lmp->input->file(script.c_str());
  } catch (std::exception& e) {
    //std::cout << "LAMMPS Exception: " << e.what() << std::endl;
    delete lmp;
    return nullptr;
  }
  return lmp;
}

#include "lammps_sim.h"
LammpsSim::LammpsSim() {};
LammpsSim::LammpsSim(std::string &cmd, int LOSS_FUNC, std::string MAX_TIME, double FAIL_SCORE):
  cmd(cmd), lossfunc(LOSS_FUNC), maxtime(MAX_TIME), failscore(FAIL_SCORE)
{
  invar.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
  tokenise();
  parse_tokens();

  if (std::find(invar.begin(), invar.end(), "-log") == invar.end() || 
      std::find(invar.begin(), invar.end(), "-l") == invar.end()) {
    invar.push_back("-log");
    invar.push_back("none");
  }
  if (std::find(invar.begin(), invar.end(), "-screen") == invar.end() || 
      std::find(invar.begin(), invar.end(), "-sc") == invar.end()) {
    invar.push_back("-screen");
    invar.push_back("none");
  }

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified!\n");
}

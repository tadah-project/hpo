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
void LammpsSim::tokenise() {
  std::stringstream ss(cmd);
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
}
void LammpsSim::parse_tokens() {
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == "-s" || tokens[i] == "--script") {
      if (i + 1 < tokens.size()) {
        script = tokens[i+1];
      }
      i++;
    }
    else if (tokens[i] == "-v" || tokens[i] == "--varloss") {
      if (i + 1 < tokens.size()) {
        varloss.push_back(tokens[i+1]);
      }
      i++;
    }
    else if (tokens[i] == "-i" || tokens[i] == "--invar") {
      if (i + 2 < tokens.size()) {
        invar.push_back("-var");
        invar.push_back(tokens[i+1].c_str());
        invar.push_back(tokens[i+2].c_str());
      }
      i+=2; // Skip the next 2 tokens
    }
    else if (tokens[i] == "-o" || tokens[i] == "--outvar") {
      if (i + 1 < tokens.size()) {
        outvar.push_back(tokens[i+1]);
      }
      i++;
    }
    else if (tokens[i] == "-w" || tokens[i] == "--weight") {
      if (i + 1 < tokens.size()) {
        weight = std::stod(tokens[i+1]);
      }
      i++;
    }
    else if (tokens[i] == "-f" || tokens[i] == "--failscore") {
      if (i + 1 < tokens.size()) {
        failscore = std::stod(tokens[i+1]);
      }
      i++;
    }
    else if (tokens[i] == "-l" || tokens[i] == "--lossfunc") {
      if (i + 1 < tokens.size()) {
        lossfunc = std::stoi(tokens[i+1]);
      }
      i++;
    }
    else if (tokens[i] == "-m" || tokens[i] == "--maxtime") {
      if (i + 1 < tokens.size()) {
        int tsec = std::stoi(tokens[i+1]);
        int s,m,h;
        h=tsec/3600;
        m=(tsec-h*3600)/60;
        s=tsec-h*3600-m*60;
        maxtime=std::to_string(h) + ":" + std::to_string(m)
          + ":" + std::to_string(s) + " every 1";
      }
      i++;
    }
    else if (tokens[i] == "-n" || tokens[i] == "--ncpu") {
      if (i + 1 < tokens.size()) {
        ncpu = std::stoi(tokens[i+1]);
      }
      i++;
    }
    else {
      throw std::runtime_error("Unknown LAMMPS config:\n" + cmd);
    }
  }
}

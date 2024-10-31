#include <tadah/hpo/lammps_sim.h>
#include <iostream>

LammpsSim::LammpsSim() {}
LammpsSim::LammpsSim(std::string &cmd, std::string MAX_TIME, double FAIL_SCORE):
  cmd(cmd), maxtime(MAX_TIME), failscore(FAIL_SCORE)
{
  sinvar.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
  tokenise();
  parse_tokens();

  sinvar.push_back("-var");
  sinvar.push_back("maxtime");
  sinvar.push_back(maxtime+" every 1");

  if (std::find(sinvar.begin(), sinvar.end(), "-log") == sinvar.end() || 
      std::find(sinvar.begin(), sinvar.end(), "-l") == sinvar.end()) {
    sinvar.push_back("-log");
    sinvar.push_back("none");
  }
  if (std::find(sinvar.begin(), sinvar.end(), "-screen") == sinvar.end() || 
      std::find(sinvar.begin(), sinvar.end(), "-sc") == sinvar.end()) {
    sinvar.push_back("-screen");
    sinvar.push_back("none");
  }

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified.\n");

  if (!varloss.size())
    throw std::runtime_error("LammpsSim: --varloss variable is not defined.\n");

  for (const auto &s: sinvar) invar.push_back(s.c_str());

  v_varloss.resize(varloss.size());
  v_outvar.resize(outvar.size());

}
void LammpsSim::tokenise() {
  std::stringstream ss(cmd);
  std::string token;
  while (ss >> token) {
    if (token[0]=='#') break;
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
      if (i + 2 < tokens.size()) {
        varloss.push_back(tokens[++i]);
        w_varloss.push_back(std::stod(tokens[++i]));
      }
    }
    else if (tokens[i] == "-i" || tokens[i] == "--invar") {
      if (i + 2 < tokens.size()) {
        sinvar.push_back("-var");
        sinvar.push_back(tokens[i+1].c_str());
        sinvar.push_back(tokens[i+2].c_str());
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
    else if (tokens[i] == "-m" || tokens[i] == "--maxtime") {
      if (i + 1 < tokens.size()) {
        maxtime = tokens[i+1];
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

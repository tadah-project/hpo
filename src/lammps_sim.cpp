#include <tadah/hpo/lammps_sim.h>
#include <iostream>

LammpsSim::LammpsSim() {}
LammpsSim::LammpsSim(std::string &cmd, std::string MAX_TIME, double FAIL_SCORE):
  cmd(cmd), maxtime(MAX_TIME), failscore(FAIL_SCORE)
{
  sinvar.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
  tokenise();
  parse_tokens();
  
  for (const auto& s : sinvar) {
    std::cout << s << " ";
  }

  sinvar.push_back("-var");
  sinvar.push_back("maxtime");
  sinvar.push_back(maxtime+" every 1");

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified.\n");

  if (!n_varloss.size())
    throw std::runtime_error("LammpsSim: --varloss variable is not defined.\n");

  for (const auto &s: sinvar) invar.push_back(s.c_str());

  v_varloss.resize(n_varloss.size());
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
  size_t i = 0;
  while (i < tokens.size()) {
    if (tokens[i] == "-s" || tokens[i] == "--script") {
      if (i + 1 < tokens.size()) {
        script = tokens[++i];
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --script");
      }
    }
    else if (tokens[i] == "-v" || tokens[i] == "--varloss") {
      if (i + 2 < tokens.size()) {
        n_varloss.push_back(tokens[++i]);
        w_varloss.push_back(std::stod(tokens[++i]));
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --varloss");
      }
    }
    else if (tokens[i] == "-i" || tokens[i] == "--invar") {
      if (i + 2 < tokens.size()) {
        sinvar.push_back("-var");
        sinvar.push_back(tokens[++i].c_str());
        sinvar.push_back(tokens[++i].c_str());
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --invar");
      }
    }
    else if (tokens[i] == "-o" || tokens[i] == "--outvar") {
      if (i + 1 < tokens.size()) {
        outvar.push_back(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --outvar");
      }
    }
    else if (tokens[i] == "-f" || tokens[i] == "--failscore") {
      if (i + 1 < tokens.size()) {
        failscore = std::stod(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --failscore");
      }
    }
    else if (tokens[i] == "-m" || tokens[i] == "--maxtime") {
      if (i + 1 < tokens.size()) {
        maxtime = tokens[++i];
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --maxtime");
      }
    }
    else if (tokens[i] == "-n" || tokens[i] == "--ncpu") {
      if (i + 1 < tokens.size()) {
        ncpu = std::stoi(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --ncpu");
      }
    }
    else if (tokens[i] == "-l" || tokens[i] == "-log") {
      if (i + 1 < tokens.size()) {
        sinvar.push_back("-log");
        sinvar.push_back(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for -log");
      }
    }
    else if (tokens[i] == "-sc" || tokens[i] == "-screen") {
      if (i + 1 < tokens.size()) {
        sinvar.push_back("-screen");
        sinvar.push_back(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for -screen");
      }
    }
    else {
      throw std::runtime_error("Unknown LAMMPS config:\n" + cmd);
    }
  }
}

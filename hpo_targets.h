#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include <cmath>
#include <vector>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>


/** This class represents LAMMPS simulation.
 *
 * In particular it contains all required information to
 * run LAMMPS simulation using Tadah!
 *
 * It also contains simulation specific parameters
 * for further data processing with HPO.
 */
class LammpsSim {
  private:
    std::string cmd; // the command line from HPO config file
    std::vector<std::string> tokens;  // cmd tokenised
    std::vector<const char*> invar;   // LAMMPS input variables
    std::vector<std::string> outvar;  // LAMMPS out variables

    // TODO define MPI_Comm?
  public:
    // Required: 
    std::string script;  // LMP script name
    std::vector<std::string> varloss; // at leat one --varloss variable
                                      // must be defined in the script
                                      // so it can be used with HPO

    // Optional:
    int ncpu = 1; // the number of cpus to be used, MPI only
    double weight;  // TODO set defaults
    double failscore;
    int lossfunc;
    std::string maxtime;

    LammpsSim(std::string &cmd):
      cmd(cmd)
  {
    invar.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
    tokenise();
    parse_tokens();
    
    if (!script.length())
      throw std::runtime_error("LammpsSim object without a script!\n");
  }

    // method to tokenise cmd
    void tokenise() {
      std::stringstream ss(cmd);
      std::string token;

      while (ss >> token) {
        tokens.push_back(token);
      }
    }

    void parse_tokens() {
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
};


/**
 * Essentially a config file for HPO
 *
 */
class HPOTargets {

  private:
    const Config &config;
  public:
    size_t      MAX_CALLS=std::numeric_limits<size_t>::max();
    double      EPS=1e-2; // control optimiser precision
    double      FAIL_SCORE=1000;
    int         LOSS_FUNC=1;
    std::string MAX_TIME="00:00:10 every 1";

    std::vector<std::string> optim_map;
    std::vector<double> optim_low;
    std::vector<double> optim_high;
    std::vector<int> optim_is_int;

    bool use_ermse=false; double ermse; double w_ermse;
    bool use_srmse=false; double srmse; double w_srmse;
    bool use_frmse=false; double frmse; double w_frmse;

    std::vector<LammpsSim> LS;

    HPOTargets(const Config &c, std::string fn):
      config(c)
    {
      read(fn);
    }

    void read(std::string fn) {
      std::ifstream ifs(fn);
      if (!ifs.good())
        throw std::runtime_error("HPO config file does not exists.\n");
      std::string line;
      while(std::getline(ifs,line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        // skip empty lines
        if(key.empty()) continue;

        // skip comments starting with #
        else if(key[0]=='#') continue;

        else if (key=="EPS") {
          iss >> EPS;
        }
        else if (key=="MAXCALLS") {
          iss >> MAX_CALLS;
        }
        else if (key=="FAILSCORE") {
          // -f --failscore <double>
          // LAMMPS local
          iss >> FAIL_SCORE;
        }
        else if (key=="LOSSFUNC") {
          // -l, --lossfunc <int>
          // default LOSS_FUNC
          iss >> LOSS_FUNC;
        }
        else if (key=="MAXTIME") {
          // -m, --maxtime <time in s>
          // should be local for every LAMMPS script
          int tsec;
          iss >> tsec;
          int s,m,h;
          h=tsec/3600;
          m=(tsec-h*3600)/60;
          s=tsec-h*3600-m*60;
          MAX_TIME=std::to_string(h) + ":" + std::to_string(m) 
            + ":" + std::to_string(s)+" every 1";
        }
        else if (key=="ERMSE") {
          iss >> ermse >> w_ermse;
          use_ermse=true;
        }
        else if (key=="SRMSE") {
          iss >> srmse >> w_srmse;
          use_srmse=true;
        }
        else if (key=="FRMSE") {
          iss >> frmse >> w_frmse;
          use_frmse=true;
        }
        else if (key=="OPTIM") {
          std::string k;
          double low, high;
          iss >> k >> low >> high;
          // do we know the key?
          if (tdc.contains(k)) {
            const auto& ckey = toml::find(config.tdc, k);
            if (toml::find<bool>(ckey, "is_internal", 0)) {
              // is it internal key?
              throw std::runtime_error("HPO cannot optimise internal KEYS:\n" + k);
            }
            // We need to know array dimensions from the training config
            // so we can push required number to optim_x
            // Here we also deal with the special case for S and C grids
            size_t size = config.size(k);
            if (k=="CGRID2B" || k=="CGRID3B" || k=="CGRIDMB" ||
                k=="SGRID2B" || k=="SGRID3B" || k=="SGRIDMB") {
              if(config.get<int>(k,0)==-1 || config.get<int>(k,0)==-2 )
                size = config.get<int>(k,1)
            }
            for (size_t i=0; i<size; ++i ) {
              optim_map.push_back(k);
              optim_low.push_back(low);
              optim_high.push_back(high);
            }
          } else {
            throw std::runtime_error("HPO OPTIM unknown key:\n" + k);
          }
        }
        else if (key=="LAMMPS") {
          std::string cmd;
          std::getline(iss, cmd);
          LS.push_back(LammpsSim(cmd));
        }
        else {
          throw std::runtime_error("HPO config file unknown key:\n" + line);
        }
      }
    }
};
#endif

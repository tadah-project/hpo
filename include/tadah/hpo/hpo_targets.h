#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include <tadah/hpo/lammps_sim.h>

#include <cmath>
#include <vector>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

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
      // sort LS such that the first simulation in the vector
      // requires the greatest number of cpus
      std::sort(LS.begin(), LS.end(), LammpsSim::greater);
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
          if (config.tdc.contains(k)) {
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
              if(config.get<int>(k,0)==-1 || config.get<int>(k,0)==-2)
                size = config.get<int>(k,1);
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
          LS.push_back(LammpsSim(cmd,LOSS_FUNC, MAX_TIME, FAIL_SCORE));
        }
        else {
          throw std::runtime_error("HPO config file unknown key:\n" + line);
        }
      }
    }
};
#endif

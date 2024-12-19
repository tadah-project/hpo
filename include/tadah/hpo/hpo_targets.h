#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include "tadah/core/registry.h"
#include "tadah/core/utils.h"
#include <string>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d2/d2_base.h>
#include <tadah/models/descriptors/dm/dm_base.h>
#include <tadah/models/descriptors/d_mjoin.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Essentially a config file for HPO
 *
 */
class HPOTargets {
private:

  size_t ui_offset(std::vector<std::string> &tokens, size_t ui) {
    size_t offset=0;
    size_t count=0;
    for (const auto t: tokens) {
      if (is_number(t)) count++;
      if (count==ui) return offset;
      offset++;
    }
    return offset;
  }

  std::vector<std::string> parse_optim_line(std::istringstream& iss) {
    std::vector<std::string> elements;
    std::string token, enclosed;
    bool in_parens = false;

    while (iss >> token) {
      if (token.front() == '(' || in_parens) {
        in_parens = true;
        if (token.front() == '(') token.erase(token.begin());
        size_t close_pos = token.find(')');
        if (close_pos != std::string::npos) {
          enclosed += token.substr(0, close_pos);
          in_parens = false;
        } else {
          enclosed += token + " ";
        }
      } else {
        elements.push_back(token);
      }
    }

    enclosed.erase(enclosed.find_last_not_of(" ") + 1);
    elements.push_back(enclosed);

    return elements;
  }


  Config &config;
  void create_labels() {
    for (auto& pair : optimap) {
      std::string key = pair.first;
      Opti &o = pair.second;

      std::string label="None";
      size_t nparams=0;
      std::string suffix = "2B";
      std::string prefix = "D2_";
      if(key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
        suffix="MB";
        prefix="DM_";
      }
      std::string type = "TYPE"+suffix;
      std::string rcut = "RCUT"+suffix;
      std::string cgrid = "CGRID"+suffix;
      std::string sgrid = "SGRID"+suffix;
      std::string mparams = "MPARAMS";  // there is no label needed
      std::string watoms = "WATOMS";

      size_t i = key == type ? 1 : 0;
      size_t s = config.size(type);  // grids should be expanded
      std::vector<std::string> type_config(s);
      config.get<std::vector<std::string>>(type,type_config);  // just in case get types from config
      // auto it = std::find(type_config.begin(), type_config.end(), key);
      size_t idx=0;

      std::string pattern="mJoin";
      type_config.erase(
        std::remove_if(type_config.begin(), type_config.end(), [&pattern](const std::string& s) {
          return s.find(pattern) != std::string::npos;
        }),
        type_config.end()
      );

      while (i < o.labels.size()) {
        // get next label
        while (idx < type_config.size()) {
          label = type_config[idx++];
          if (label.rfind(prefix, 0) != 0) continue;
          D_Base *d = CONFIG::factory<D_Base>(label);
          auto &d_keys = d->keys;
          size_t d_nparams = d->nparams; // number of params to follow after descriptor
          auto it = std::find(d_keys.begin(), d_keys.end(), key); // check does descriptor uses this key
          // if it is not the case we are not dealing with grids

          delete d;
          if (it == d_keys.end()) {
            //nparams=d_nparams;
            if (key==type) nparams = d_nparams;
            else nparams=0;
            break;
          }
          size_t position = std::distance(d_keys.begin(), it);
          idx += d_nparams - d_keys.size();
          // idx+=position+(d_nparams-position);
          if (idx < type_config.size() && is_number(type_config[idx])) {
            nparams = std::stoi(type_config[idx]);
          } else {
            // Handle the case where no valid number follows
            throw std::runtime_error("Expected a number after label.");
          }
          break;
        }
        // std::cout << "TYPE " << type << " KEY " << key << " Label " << label << " nparams " << nparams << " i " << i << " label.size " << o.labels.size()  << std::endl;
        if (key==rcut) {
          o.labels[i++] = label;
        }
        else if (key==cgrid || key==sgrid || key==type) {
        // std::cout << " 1 TYPE " << type << " KEY " << key << " Label " << label << " nparams " << nparams << " i " << i << " label.size " << o.labels.size()  << std::endl;
          for (size_t j=0; j<nparams; j++ ) o.labels[i++] = label;
        }
        else if (key==watoms) {
          o.labels[i] = config.get<std::string>("ATOMS",i);
          i++;
        }
        else {
          std::cout << "SOMETHING ELSE WE DO NOT KNOW" << std::endl;
          i++;
        }
      }
    }
  }
  struct Opti {
    // indexing from 1, zeroth element is unused
    std::vector<bool> opt;
    std::vector<std::string> config; // values coooresponding to this key
    std::vector<std::string> labels; // label for this config parameter
    std::vector<double> lowbound;
    std::vector<double> highbound;

    void resize(size_t s) {
      if (opt.size()>=s) return;
      opt.resize(s,false);
      config.resize(s);
      labels.resize(s," ");
      lowbound.resize(s,0);
      highbound.resize(s,0);
    }
    void print(const std::string &key) const {
      int p = 1;
      int w = 8 + p; // Width for alignment
      size_t start = is_number(config[0]) ? 0 : 1;

      std::cout << "\nConfig Key: " << key << std::endl;
      
      // Print labels
      std::cout << std::setw(w) << "Label:";
      for (size_t i = start, idx = start; i < labels.size(); ++i) {
        if (is_number(config[i])) {
          std::cout << std::setw(w) << labels[idx++];
        }
      }
      std::cout << std::endl;

      // Print literals first
      std::cout << std::setw(w) << " ";
      std::string label;
      for (size_t i = start; i < config.size(); ++i) {
        if (!is_number(config[i])) {
          label = config[i];
        } else {
          std::cout << std::setw(w) << std::right << label;
        }
      }
      std::cout << std::endl;

      // Print user indices with "Index:" as the first column
      std::cout << std::setw(w) << "Index:";
      for (size_t i = start, idx = 1; i < config.size(); ++i) {
        if (is_number(config[i])) {
          std::cout << std::setw(w) << idx++;
        }
      }
      std::cout << std::endl;

      // Print current config parameters with "Params:" as the first column
      std::cout << std::setw(w) << "Params:";
      for (size_t i = start; i < config.size(); ++i) {
        if (is_number(config[i])) {
          double num = std::stod(config[i]);
          if (is_integer(num)) {
            std::cout << std::setw(w) << std::right << static_cast<int>(num);
          } else {
            std::cout << std::setw(w) << std::right << std::scientific << std::setprecision(p) << num;
          }
        }
      }
      std::cout << std::endl;

      // Conditional printing of highbound based on opt with "H bound:" as the first column
      std::cout << std::setw(w) << "H bound:";
      for (size_t i = start; i < opt.size(); ++i) {
        if (!is_number(config[i])) continue;
        if (opt[i]) {
          if (is_integer(highbound[i])) {
            std::cout << std::setw(w) << static_cast<int>(highbound[i]);
          } else {
            std::cout << std::setw(w) << std::scientific << std::setprecision(p) << highbound[i];
          }
        } else {
          std::cout << std::setw(w) << std::right << " ";
        }
      }
      std::cout << std::endl;

      // Conditional printing of lowbound based on opt with "L bound:" as the first column
      std::cout << std::setw(w) << "L bound:";
      for (size_t i = start; i < opt.size(); ++i) {
        if (!is_number(config[i])) continue;
        if (opt[i]) {
          if (is_integer(lowbound[i])) {
            std::cout << std::setw(w) << static_cast<int>(lowbound[i]);
          } else {
            std::cout << std::setw(w) << std::scientific << std::setprecision(p) << lowbound[i];
          }
        } else {
          std::cout << std::setw(w) << std::right << " ";
        }
      }
      std::cout << std::endl;
    }
  };
  std::map<std::string, Opti> optimap;
public:
  size_t      MAX_CALLS=std::numeric_limits<size_t>::max();
  double      EPS=1e-2; // control optimiser precision
  double      FAIL_SCORE=1000;
  std::string MAX_TIME="10";
  /* 0 - random search
     * 1 - grid search
     * 2 - MaxLIPO+TR */
  std::string ALGO="2"; 

  std::string potevery_dir;
  size_t potevery=std::numeric_limits<size_t>::max(); // dump pot this many timesteps
  // regardless of tloss

  std::vector<std::string> optim_map;
  std::vector<double> optim_low;
  std::vector<double> optim_high;
  std::vector<int> optim_is_int;

  std::vector<double> optim_init;

  bool use_ermse=false; double ermse; double w_ermse;
  bool use_srmse=false; double srmse; double w_srmse;
  bool use_frmse=false; double frmse; double w_frmse;

  std::vector<LammpsSim> LS;
  std::vector<Config> config_2b;
  std::vector<Config> config_mb;

  HPOTargets(Config &c, std::string fn):
    config(c)
  {
    D_mJoin::expand_grids(config);

    read(fn);
    // sort LS such that the first simulation in the vector
    // requires the greatest number of cpus
    std::sort(LS.begin(), LS.end(), LammpsSim::greater);

    create_labels();
    for (auto& pair : optimap) {
      std::string key = pair.first;
      Opti &o = pair.second;
      o.print(key);
    }
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
      else if (key=="ALGO") {
        iss >> ALGO;
      }
      else if (key=="POTEVERY") {
        iss >> potevery >> potevery_dir;
        if (!fs::exists(potevery_dir)) {
          if (fs::create_directory(potevery_dir)) {
            std::cout << "Directory created: " << potevery_dir << std::endl;
          } else {
            std::cerr << "Failed to create directory: " << potevery_dir << std::endl;
          }
        } else {
          std::cout << "Directory already exists: " << potevery_dir << std::endl;
        }
      }
      else if (key=="MAXCALLS") {
        iss >> MAX_CALLS;
      }
      else if (key=="FAILSCORE") {
        // -f --failscore <double>
        // LAMMPS local
        iss >> FAIL_SCORE;
      }
      else if (key=="MAXTIME") {
        // -m, --maxtime <time in s>
        iss >> MAX_TIME;
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

            if(config.get<int>(k,0)==-1 || config.get<int>(k,0)==-2) {
              size = config.get<int>(k,1);
              std::vector<double> grid(size);
              D_Base::get_grid(config,k,grid);
              optim_init.insert(optim_init.end(), grid.begin(), grid.end());
            }
          }
          else if (k=="TYPE2B") {
            std::string type;
            iss >> type;
            D_Base *d = CONFIG::factory<D_Base>(type);
            size = d->nparams;
            delete d;
          }
          else {
            for (size_t i=0; i<size; ++i ) {
              optim_init.push_back(config.get<double>(k,i));
            }
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
      else if (key=="optim") {
        std::string k;
        iss >> k;   // SGRIDMB, or MPARAM or TYPE2B ZBL
        //optimap.emplace(k, Opti()); // emplace only if does not exist
        Opti &omap = optimap[k]; 
        size_t s = config.size(k);  // grids should be expanded
        omap.resize(s);

        std::vector<std::string> tokens = parse_optim_line(iss);
        std::vector<size_t> user_indices = parse_indices(tokens.back());
        tokens.pop_back();
        double high = std::stod(tokens.back());
        tokens.pop_back();
        double low = std::stod(tokens.back());
        tokens.pop_back();

        config.get<std::vector<std::string>>(k,omap.config);

        for (const auto &ui: user_indices) {
          size_t i = ui_offset(omap.config,ui);
          omap.opt[i] = true;
          omap.highbound[i] = high;
          omap.lowbound[i] = low;
        }
      }
      else if (key=="LAMMPS") {
        std::string cmd;
        std::getline(iss, cmd);
        LS.push_back(LammpsSim(cmd, MAX_TIME, FAIL_SCORE));
      }
      else {
        throw std::runtime_error("HPO config file unknown key:\n" + line);
      }
    }
  }
};
#endif

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
#include <utility>
#include <vector>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <filesystem>

namespace fs = std::filesystem;

class HPOTargets {
private:

  size_t ui_offset(std::vector<std::string> &tokens, size_t ui) {
    size_t offset = 0;
    size_t count = 0;
    for (const auto& t : tokens) {
      if (is_number(t)) count++;
      if (count == ui) return offset;
      offset++;
    }
    return offset;
  }

std::vector<std::string> parse_optim_line(std::istringstream& iss) {
    std::vector<std::string> elements;
    std::string token, enclosed;
    bool in_parens = false;
    bool has_parens = false;
    std::string processed_line;
    std::string indices = "";

    while (iss >> token) {
        processed_line += token + " ";

        if (token.front() == '(' || in_parens) {
            has_parens = true;
            if (token.front() == '(') {
                if (in_parens) {
                    throw std::runtime_error("Nested parentheses are not supported up to: " + processed_line);
                }
                in_parens = true;
                token.erase(token.begin());
            }
            size_t close_pos = token.find(')');
            if (close_pos != std::string::npos) {
                enclosed += token.substr(0, close_pos);
                in_parens = false;
                if (enclosed.empty()) {
                    throw std::runtime_error("Empty parentheses are not valid up to: " + processed_line);
                }
                indices += enclosed;
                enclosed.clear();
            } else {
                enclosed += token + " ";
            }
        } else {
            if (token.find(')') != std::string::npos && !in_parens) {
                throw std::runtime_error("Unmatched closing parenthesis up to: " + processed_line);
            }
            elements.push_back(token);
        }
    }

    if (in_parens) {
        throw std::runtime_error("Unmatched opening parenthesis up to: " + processed_line);
    }

    if (!has_parens) {
        throw std::runtime_error("No parentheses found up to: " + processed_line);
    }

    elements.push_back(indices);
    return elements;
}


  Config &config;

  std::vector<std::pair<std::string, std::vector<double>>> get_d_labels(const std::string& type, const std::string& prefix) {
    std::vector<std::pair<std::string, std::vector<double>>> d_label_obj;

    size_t s = config.size(type);
    std::vector<std::string> type_config(s);
    config.get<std::vector<std::string>>(type, type_config);

    std::vector<double> d_label_param;
    std::string current_label;
    bool processing = false;

    for (const auto& token : type_config) {
      if (token.rfind(prefix, 0) == 0) {
        if (token == prefix + "mJoin") continue;

        if (processing) {
          d_label_obj.emplace_back(current_label, d_label_param);
          d_label_param.clear();
        }

        current_label = token;
        processing = true;
      } else if (processing) {
        d_label_param.push_back(std::stod(token));
      }
    }

    if (processing) {
      d_label_obj.emplace_back(current_label, d_label_param);
    }

    return d_label_obj;
  }

  struct Opti {
    std::vector<bool> opt;
    std::vector<std::string> config;
    std::vector<std::string> labels;
    std::vector<double> lowbound;
    std::vector<double> highbound;

    void resize(size_t s) {
      if (opt.size() >= s) return;
      opt.resize(s, false);
      config.resize(s);
      labels.resize(s, " ");
      lowbound.resize(s, 0);
      highbound.resize(s, 0);
    }

    void print(const std::string &key) const {
      int p = 1;
      int w = 8 + p;
      size_t start = is_number(config[0]) ? 0 : 1;

      std::cout << "\nConfig Key: " << key << std::endl;

      std::cout << std::setw(w) << "Label:";
      for (size_t i = start, idx = start; i < labels.size(); ++i) {
        if (is_number(config[i])) {
          std::cout << std::setw(w) << labels[idx++];
        }
      }
      std::cout << std::endl;

      // std::cout << std::setw(w) << " ";
      // std::string label;
      // for (size_t i = start; i < config.size(); ++i) {
      //   if (!is_number(config[i])) {
      //     label = config[i];
      //   } else {
      //     std::cout << std::setw(w) << std::right << label;
      //   }
      // }
      // std::cout << std::endl;

      std::cout << std::setw(w) << "Index:";
      for (size_t i = start, idx = 1; i < config.size(); ++i) {
        if (is_number(config[i])) {
          std::cout << std::setw(w) << idx++;
        }
      }
      std::cout << std::endl;

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

  void create_labels() {
    for (auto& pair : optimap) {
      std::string key = pair.first;
      Opti& o = pair.second;

      auto suffix_prefix = determine_suffix_prefix(key);
      std::string type = "TYPE" + suffix_prefix.first;
      std::string rcut = "RCUT" + suffix_prefix.first;
      std::string cgrid = "CGRID" + suffix_prefix.first;
      std::string sgrid = "SGRID" + suffix_prefix.first;
      std::string watoms = "WATOMS";

      size_t i = key == type ? 1 : 0;

      auto d_labels = get_d_labels(type, suffix_prefix.second);

      size_t nparams = 0;
      for (auto it = d_labels.begin(); it != d_labels.end(); ++it) {
        std::string label = it->first;
        auto params = it->second;

        if (key == rcut) {
          o.labels[i++] = label;
        } else if (key == type) {
          nparams = params.size();
          for (size_t j = 0; j < nparams; j++) o.labels[i++] = label;
        } else if (key == cgrid || key == sgrid) {
          process_grid_labels(label, o, i, params, key);
        } else if (key == watoms) {
          process_atoms_labels(o, i, watoms);
        } else {
          if (watoms != "\0") std::cerr << "Cannot create label for: " << label << std::endl;
          i++;
        }
      }
    }
  }

  std::pair<std::string, std::string> determine_suffix_prefix(const std::string& key) {
    if (key.size() >= 2 && key.compare(key.size() - 2, 2, "2B") == 0) {
      return {"2B", "D2_"};
    } else {
      return {"MB", "DM_"};
    }
  }

  void process_grid_labels(const std::string& label, Opti& o, size_t& i,
                           const std::vector<double>& params, const std::string& key) {
    std::unique_ptr<D_Base> d(CONFIG::factory<D_Base>(label));
    auto& d_keys = d->keys;
    auto it = std::find(d_keys.begin(), d_keys.end(), key);

    if (it != d_keys.end()) {
      size_t nparams = params.back();
      for (size_t j = 0; j < nparams; j++) {
        o.labels[i++] = label;
      }
    }
  }

  void process_atoms_labels(Opti& o, size_t& i, std::string& watoms) {
    for (size_t j = 0; j < config.size("ATOMS"); ++j) {
      o.labels[i++] = config.get<std::string>("ATOMS", j);
    }
    watoms = "\0"; // Avoid further processing
  }


public:
  std::map<std::string, Opti> optimap;
  size_t MAX_CALLS = std::numeric_limits<size_t>::max();
  double EPS = 1e-2;
  double FAIL_SCORE = 1000;
  std::string MAX_TIME = "10";
  std::string ALGO = "2";

  std::string potevery_dir;
  size_t potevery = std::numeric_limits<size_t>::max();

  // std::vector<std::string> optim_map;
  std::vector<double> optim_low;
  std::vector<double> optim_high;
  // std::vector<int> optim_is_int;
  // std::vector<double> optim_init;

  bool use_ermse = false; double ermse; double w_ermse;
  bool use_srmse = false; double srmse; double w_srmse;
  bool use_frmse = false; double frmse; double w_frmse;

  std::vector<LammpsSim> LS;
  std::vector<Config> config_2b;
  std::vector<Config> config_mb;

  HPOTargets(Config &c, std::string fn) : config(c) {
    D_mJoin::expand_grids(config);

    read(fn);
    std::sort(LS.begin(), LS.end(), LammpsSim::greater);
    create_labels();
    for (auto& pair : optimap) {
      std::string key = pair.first;
      Opti &o = pair.second;
      o.print(key);
    }

    // prepare low and high bound vectors
    for (auto& p : optimap) {
      for (size_t i=0; i<p.second.opt.size(); ++i) {
        if (p.second.opt[i]) {
          optim_low.push_back(p.second.lowbound[i]);
          optim_high.push_back(p.second.highbound[i]);
        }
      }
    }
  }

  void read(std::string fn) {
    std::ifstream ifs(fn);
    if (!ifs.good())
      throw std::runtime_error("HPO config file does not exist.\n");

    std::string line;
    size_t nline = 0;
    while (std::getline(ifs, line)) {
      ++nline;
      // Remove (inline) comments
      size_t comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }

      std::istringstream iss(line);
      std::string key;
      iss >> key;

      if (key.empty()) continue;

      try {
        if (key == "EPS") {
          iss >> EPS;
        } else if (key == "ALGO") {
          iss >> ALGO;
        } else if (key == "POTEVERY") {
          iss >> potevery >> potevery_dir;
          handle_directory_creation(potevery_dir);
        } else if (key == "MAXCALLS") {
          iss >> MAX_CALLS;
        } else if (key == "FAILSCORE") {
          iss >> FAIL_SCORE;
        } else if (key == "MAXTIME") {
          iss >> MAX_TIME;
        } else if (key == "ERMSE") {
          iss >> ermse >> w_ermse;
          use_ermse = true;
        } else if (key == "SRMSE") {
          iss >> srmse >> w_srmse;
          use_srmse = true;
        } else if (key == "FRMSE") {
          iss >> frmse >> w_frmse;
          use_frmse = true;
        } else if (key == "OPTIM") {
          process_optim_config(iss,nline);
        } else if (key == "LAMMPS") {
          std::string cmd;
          std::getline(iss, cmd);
          LS.push_back(LammpsSim(cmd, MAX_TIME, FAIL_SCORE));
        } else {
          throw std::runtime_error("Unknown key");
        }
      } catch (const std::exception& e) {
        throw std::runtime_error("Error on line " + std::to_string(nline) + ": " + e.what() + "\nLine: " + line);
      }
    }
  }

  void handle_directory_creation(const std::string& directory) {
    try {
      if (!fs::exists(directory)) {
        if (fs::create_directory(directory)) {
          std::cout << "Directory created: " << directory << std::endl;
        } else {
          std::cerr << "Failed to create directory: " << directory << std::endl;
        }
      } else {
        std::cout << "Directory already exists: " << directory << std::endl;
      }
    } catch (const std::exception& e) {
      throw std::runtime_error("Error in handle_directory_creation: " + std::string(e.what()));
    }
  }

  void process_optim_config(std::istringstream& iss, size_t nline) {
    std::string k;
    if (!(iss >> k)) {
      throw std::runtime_error("Failed to read key from input.");
    }

    Opti& omap = optimap[k];
    size_t s = config.size(k);
    omap.resize(s);

    std::vector<std::string> tokens = parse_optim_line(iss);

    if (tokens.size() < 3) {
      throw std::runtime_error("Insufficient tokens from parse_optim_line. Tokens: " + 
                               std::to_string(tokens.size()));
    }

    std::vector<size_t> user_indices;
    try {
      user_indices = parse_indices(tokens.back());
    } catch (const std::exception& e) {
      throw std::runtime_error("Error parsing indices on process_optim_config line " + std::to_string(nline) + ": " + std::string(e.what()) +
                               "\nTokens: " + tokens.back());
    }
    tokens.pop_back();

    double high;
    try {
      high = std::stod(tokens.back());
    } catch (const std::exception& e) {
      throw std::runtime_error("Error converting high bound to double on process_optim_config line " + std::to_string(nline) + ": " + std::string(e.what()) +
                               "\nTokens: " + tokens.back());
    }
    tokens.pop_back();

    double low;
    try {
      low = std::stod(tokens.back());
    } catch (const std::exception& e) {
      throw std::runtime_error("Error converting low bound to double on process_optim_config line " + std::to_string(nline) + ": " + std::string(e.what()) +
                               "\nTokens: " + tokens.back());
    }
    tokens.pop_back();

    config.get<std::vector<std::string>>(k, omap.config);

    for (const auto& ui : user_indices) {
      try {
        size_t i = ui_offset(omap.config, ui);
        omap.opt[i] = true;
        omap.highbound[i] = high;
        omap.lowbound[i] = low;
      } catch (const std::exception& e) {
        throw std::runtime_error("Error processing user index on process_optim_config line " + std::to_string(nline) + ": " + std::string(e.what()) +
                                 "\nTokens: " + std::to_string(ui));
      }
    }
  }
};
#endif

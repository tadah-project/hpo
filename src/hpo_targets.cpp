#include <tadah/hpo/hpo_targets.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/loss_function.h>

#include <algorithm>
#include <random>
#include <fstream>
#include <stdexcept>

namespace tadah {
namespace hpo {

namespace fs = std::filesystem;

size_t HPOTargets::ui_offset(std::vector<std::string> &tokens, size_t ui) {
  size_t offset = 0;
  size_t count = 0;
  for (const auto& t : tokens) {
    if (core::is_number(t)) count++;
    if (count == ui) return offset;
    offset++;
  }
  return offset;
}

std::vector<std::string> HPOTargets::parse_optim_line(std::istringstream& iss) {
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
          throw std::runtime_error(
            "Nested parentheses are not supported up to: " + processed_line);
        }
        in_parens = true;
        token.erase(token.begin());
      }
      size_t close_pos = token.find(')');
      if (close_pos != std::string::npos) {
        enclosed += token.substr(0, close_pos);
        in_parens = false;
        if (enclosed.empty()) {
          throw std::runtime_error(
            "Empty parentheses are not valid up to: " + processed_line);
        }
        indices += enclosed;
        enclosed.clear();
      } else {
        enclosed += token + " ";
      }
    } else {
      if (token.find(')') != std::string::npos && !in_parens) {
        throw std::runtime_error(
          "Unmatched closing parenthesis up to: " + processed_line);
      }
      elements.push_back(token);
    }
  }

  if (in_parens) {
    throw std::runtime_error(
      "Unmatched opening parenthesis up to: " + processed_line);
  }

  if (!has_parens) {
    throw std::runtime_error("No parentheses found up to: " + processed_line);
  }

  elements.push_back(indices);
  return elements;
}

void HPOTargets::Opti::resize(size_t s) {
  if (opt.size() >= s) return;
  opt.resize(s, false);
  config.resize(s);
  labels.resize(s, "");
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

void HPOTargets::Opti::print(const std::string &key) const {
  int p = 1;
  int w = 8 + p;
  size_t start = core::is_number(config[0]) ? 0 : 1;

  std::cout << "\nConfig Key: " << key << std::endl;
  std::cout << std::setw(w) << "Label:";
  for (size_t i = start, idx = start; i < labels.size(); ++i) {
    if (core::is_number(config[i])) {
      std::cout << std::setw(w) << labels[idx++];
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "Index:";
  for (size_t i = start, idx = 1; i < config.size(); ++i) {
    if (core::is_number(config[i])) {
      std::cout << std::setw(w) << idx++;
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "Params:";
  for (size_t i = start; i < config.size(); ++i) {
    if (core::is_number(config[i])) {
      double num = std::stod(config[i]);
      if (core::is_integer(num)) {
        std::cout << std::setw(w) << std::right << static_cast<int>(num);
      } else {
        std::cout << std::setw(w) << std::right
          << std::scientific << std::setprecision(p) << num;
      }
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "H bound:";
  for (size_t i = start; i < opt.size(); ++i) {
    if (!core::is_number(config[i])) continue;
    if (opt[i]) {
      if (core::is_integer(highbound[i])) {
        std::cout << std::setw(w) << static_cast<int>(highbound[i]);
      } else {
        std::cout << std::setw(w) << std::scientific 
          << std::setprecision(p) << highbound[i];
      }
    } else {
      std::cout << std::setw(w) << std::right << " ";
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "L bound:";
  for (size_t i = start; i < opt.size(); ++i) {
    if (!core::is_number(config[i])) continue;
    if (opt[i]) {
      if (core::is_integer(lowbound[i])) {
        std::cout << std::setw(w) << static_cast<int>(lowbound[i]);
      } else {
        std::cout << std::setw(w) << 
          std::scientific << std::setprecision(p) << lowbound[i];
      }
    } else {
      std::cout << std::setw(w) << std::right << " ";
    }
  }
  std::cout << std::endl;
}

void HPOTargets::create_labels() {
  for (auto& pair : optimap) {
    std::string key = pair.first;
    Opti& o = pair.second;

    auto suffix_prefix = determine_suffix_prefix(key);
    std::string type = "TYPE" + suffix_prefix.first;
    std::string rcut = "RCUT" + suffix_prefix.first;
    std::string cgrid = "CGRID" + suffix_prefix.first;
    std::string sgrid = "SGRID" + suffix_prefix.first;
    std::string watom = "WATOM";
    std::string weights = "WEIGHTS";

    size_t i = key == type ? 1 : 0;

    auto d_labels = type=="TYPE" ? get_labels(key)
      : get_d_labels(type, suffix_prefix.second);

    size_t nparams = 0;
    for (auto it = d_labels.begin(); it != d_labels.end(); ++it) {
      std::string label = it->first;
      auto params = it->second;

      if (key == rcut) {
        o.labels[i++] = label;
      } else if (key == weights) {
        o.labels[i++] = label;
      } else if (key == type) {
        nparams = params.size();
        for (size_t j = 0; j < nparams; j++) o.labels[i++] = label;
      } else if (key == cgrid || key == sgrid) {
        process_grid_labels(label, o, i, params, key);
      } else if (key == watom) {
        process_atoms_labels(o, i, watom);
      } else {
        if (watom != "\0") std::cerr << "Cannot create label for: "
          << label << std::endl;
        i++;
      }
    }
  }
}

std::vector<std::pair<std::string, std::vector<double>>> HPOTargets::
get_d_labels(const std::string& type, const std::string& prefix) {
  std::vector<std::pair<std::string, std::vector<double>>> d_label_obj;

  // if (config.exist(type)) { }
  size_t s = context.size(type);
  std::vector<std::string> type_config(s);
  context.get<std::vector<std::string>>(type, type_config);

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
    } else if (processing && core::is_number(token)) {
      d_label_param.push_back(std::stod(token));
    }
  }

  if (processing) {
    d_label_obj.emplace_back(current_label, d_label_param);
  }

  return d_label_obj;
}
std::vector<std::pair<std::string, std::vector<double>>> HPOTargets::
get_labels(const std::string& key) {
  std::vector<std::pair<std::string, std::vector<double>>> label_obj;

  size_t s = context.size(key);
  std::vector<std::string> key_config(s);
  context.get<std::vector<std::string>>(key, key_config);

  std::vector<double> label_param;

  for (const auto& token : key_config) {
    label_param.push_back(std::stod(token));
  }
  label_obj.emplace_back(key, label_param);

  return label_obj;
}

std::pair<std::string, std::string> HPOTargets::
determine_suffix_prefix(const std::string& key) {
  if (key.size() >= 2 && key.compare(key.size() - 2, 2, "2B") == 0) {
    return {"2B", "D2_"};
  }
  else if (key.size() >= 2 && key.compare(key.size() - 2, 2, "MB") == 0) {
    return {"MB", "DM_"};
  } else {
    return {"", ""};
  }
}

void HPOTargets::process_grid_labels(
  const std::string& label, Opti& o, size_t& i,
  const std::vector<double>& params, const std::string& key) {
  std::unique_ptr<models::D_Base> d(tadah::core::factory<models::D_Base>(label));
  int arg_pos = d->get_arg_pos(key);

  if (arg_pos > -1) {
    size_t nparams = params[arg_pos];
    for (size_t j = 0; j < nparams; j++) {
      o.labels[i++] = label;
    }
  }
}

void HPOTargets::process_atoms_labels(Opti& o, size_t& i, std::string& watom) {
  for (size_t j = 0; j < context.size("ATOM"); ++j) {
    o.labels[i++] = context.get<std::string>("ATOM", j);
  }
  watom = "\0"; // Avoid further processing
}

HPOTargets::HPOTargets(core::Context &c, std::string fn) : context(c) {
  models::D_mJoin::expand_grids(context);

  read(fn);
  std::sort(LS.begin(), LS.end(), LammpsSim::greater);
  create_labels();
  for (auto& pair : optimap) {
    std::string key = pair.first;
    Opti &o = pair.second;
    o.print(key);
  }

  for (auto& p : optimap) {
    for (size_t i = 0; i < p.second.opt.size(); ++i) {
      if (p.second.opt[i]) {
        optim_low.push_back(p.second.lowbound[i]);
        optim_high.push_back(p.second.highbound[i]);
      }
    }
  }

  // init with a random value between low and high
  size_t nOptimInits = 0;
  optim_init.resize(nOptimInits);
  std::random_device rd;
  std::mt19937 gen(rd());
  for (size_t j = 0; j < nOptimInits; ++j) {
    optim_init[j].resize(optim_low.size());
    for (size_t i = 0; i < optim_low.size(); ++i) {
      std::uniform_real_distribution<> dis(optim_low[i], optim_high[i]);
      optim_init[j][i] = dis(gen);
    }
  }
  // dummy for now
  optim_is_int.resize(optim_low.size(), false);

}

void HPOTargets::read(std::string fn)
{
  std::ifstream ifs(fn);
  if (!ifs.good()) throw std::runtime_error("HPO context file does not exist.");

  std::string line;
  size_t nline = 0;
  while (std::getline(ifs, line)) {
    ++nline;
    size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }

    std::istringstream iss(line);
    std::string key;
    iss >> key;

    if (key.empty()) continue;

    try {
      std::string globLossTypeStr;
      std::vector<double> globLossParams;
      if (key == "PARAMS") {
        double temp;
        while (iss >> temp) {
          params.push_back(temp);
        }
      } else if (key == "OPTIMIZER") {
        std::string temp;
        while(iss >> temp) {
          optimizer.push_back(temp);
        }
      } else if (key == "LOSS") {
        iss >> globLossTypeStr;
        if (globLossTypeStr== "HUBER") {
          try {
            double deltaHuber;
            iss >> deltaHuber;
            globLossParams.push_back(deltaHuber);
          } catch (const std::exception& e) {
            throw std::runtime_error("Error reading delta Huber parameter on line " 
                + core::to_string(nline) + ": " 
                + e.what());
          }
        } else if (globLossTypeStr== "TUKEY") {
          double tukeyC;
          try {
            iss >> tukeyC;
            globLossParams.push_back(tukeyC);
          } catch (const std::exception& e) {
            throw std::runtime_error("Error reading Tukey's biweight parameter on line " 
                + core::to_string(nline) + ": " 
                + e.what());
          }
        }

        auto lossTypeId = tadah::LossFunction::from_string(globLossTypeStr);
        globalLoss = tadah::LossFunction(lossTypeId, globLossParams);

      } else if (key == "OUTPUT") {
        iss >> outputEvery;
      } else if (key == "BOUTPUT") {
        iss >> bestOutputEvery;
      } else if (key == "DUMP") {
        iss >> dumpPotEvery >> dumpPotDir;
        handle_directory_creation(dumpPotDir);
      } else if (key == "BDUMP") {
        iss >> dumpBestPotEvery >> dumpBestPotDir;
        handle_directory_creation(dumpBestPotDir);
      } else if (key == "MAXCALLS") {
        iss >> MAX_CALLS;
      } else if (key == "FAILSCORE") {
        iss >> FAIL_SCORE;
      } else if (key == "MAXTIME") {
        iss >> MAX_TIME;
      } else if (key == "THREADS") {
        iss >> THREADS;
      } else if (key == "EMAE") {
        iss >> emae >> w_emae;
        use_emae = true;
      } else if (key == "ERMSE") {
        iss >> ermse >> w_ermse;
        use_ermse = true;
      } else if (key == "ErRMSE") {
        iss >> errmse >> w_errmse;
        use_errmse = true;
      } else if (key == "ERSQ") {
        iss >> ersq >> w_ersq;
        use_ersq = true;
      } else if (key == "SRMSE") {
        iss >> srmse >> w_srmse;
        use_srmse = true;
      } else if (key == "FRMSE") {
        iss >> frmse >> w_frmse;
        use_frmse = true;
      } else if (key == "OPTIM") {
        process_optim_config(iss, nline);
      } else if (key == "LAMMPS") {
        std::string cmd;
        std::getline(iss, cmd);
        /* prepare default param vector for the global loss ------------------- */
        LS.emplace_back(cmd, MAX_TIME, FAIL_SCORE, globalLoss);
      } else {
        throw std::runtime_error("Unknown key");
      }
    } catch (const std::exception& e) {
      throw std::runtime_error("Error on line " + core::to_string(nline) 
          + ": " + e.what() + "\nLine: " + line);
    }
  }
}

void HPOTargets::handle_directory_creation(const std::string& directory) {
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
    throw std::runtime_error("Error in handle_directory_creation: "
                             + std::string(e.what()));
  }
}

void HPOTargets::process_optim_config(std::istringstream& iss, size_t nline) {
  std::string k;
  if (!(iss >> k)) {
    throw std::runtime_error("Failed to read key from input.");
  }

  Opti& omap = optimap[k];
  size_t s = context.size(k);
  omap.resize(s);

  std::vector<std::string> tokens = parse_optim_line(iss);

  if (tokens.size() < 3) {
    throw std::runtime_error(
      "Insufficient tokens from parse_optim_line. Tokens: "
      + core::to_string(tokens.size()));
  }

  std::vector<size_t> user_indices;
  try {
    user_indices = core::parse_indices(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error parsing indices on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  double high;
  try {
    high = std::stod(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error converting high bound to double on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  double low;
  try {
    low = std::stod(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error converting low bound to double on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  context.get<std::vector<std::string>>(k, omap.config);

  for (const auto& ui : user_indices) {
    try {
      size_t i = ui_offset(omap.config, ui);
      if (i>=s) throw std::runtime_error("Out of bound index: " + core::to_string(ui));
      omap.opt[i] = true;
      omap.highbound[i] = high;
      omap.lowbound[i] = low;
    } catch (const std::exception& e) {
      throw std::runtime_error(
        "Error processing user index on process_optim_config line "
        + core::to_string(nline) + ": " + std::string(e.what()) +
        "\nTokens: " + core::to_string(ui));
    }
  }
  for (size_t i = 0; i < omap.lowbound.size(); ++i) {
    if (omap.opt[i] && omap.highbound[i] <= omap.lowbound[i]) {
      throw std::runtime_error("Error: Highbound must be greater than lowbound: "
      + core::to_string(omap.lowbound[i])+">="+core::to_string(omap.highbound[i])+ "\n");
    }
  }
}
} // namespace hpo
} // namespace tadah


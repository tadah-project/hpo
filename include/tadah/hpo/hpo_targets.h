#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

#include "tadah/core/context.h"
#include "tadah/core/utils.h"
#include <tadah/hpo/lammps_sim.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <sstream>
#include <filesystem>

namespace tadah {
namespace hpo {

class HPOTargets {
private:
    struct Opti {
        std::vector<bool> opt;
        std::vector<std::string> config;
        std::vector<std::string> labels;
        std::vector<double> lowbound;
        std::vector<double> highbound;

        void resize(size_t s);
        void print(const std::string &key) const;
    };

    size_t ui_offset(std::vector<std::string> &tokens, size_t ui);
    std::vector<std::string> parse_optim_line(std::istringstream& iss);
    std::vector<std::pair<std::string, std::vector<double>>> get_labels(const std::string& key);
    std::vector<std::pair<std::string, std::vector<double>>> get_d_labels(const std::string& type, const std::string& prefix);
    std::pair<std::string, std::string> determine_suffix_prefix(const std::string& key);
    void process_grid_labels(const std::string& label, Opti& o, size_t& i, const std::vector<double>& params, const std::string& key);
    void process_atoms_labels(Opti& o, size_t& i, std::string& watoms);
    void create_labels();
    void handle_directory_creation(const std::string& directory);
    void process_optim_config(std::istringstream& iss, size_t nline);
    
    Context &context;

public:
    HPOTargets(Context &c, std::string fn);
    void read(std::string fn);
    std::map<std::string, Opti> optimap;
    std::vector<double> optim_low;
    std::vector<double> optim_high;
    std::vector<LammpsSim> LS;

    size_t MAX_CALLS = std::numeric_limits<size_t>::max();
    double EPS = 1e-2;
    double FAIL_SCORE = 1000;
    std::string MAX_TIME = "10";
    std::string ALGO = "2";
    std::string potevery_dir;
    size_t potevery = std::numeric_limits<size_t>::max();
    bool use_emae = false; bool use_ermse = false;  bool use_errmse = false; 
    double emae, ermse, errmse; 
    double w_emae, w_ermse, w_errmse;
    bool use_srmse = false; 
    double srmse; 
    double w_srmse;
    bool use_frmse = false; 
    double frmse; 
    double w_frmse;
    double N=1; // L_g = abs(y_pred - y)^N
};
} // namespace hpo
} // namespace tadah

#endif // HPO_TARGETS_H

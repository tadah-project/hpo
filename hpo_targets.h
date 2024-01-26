#ifndef HPO_TARGETS_H
#define HPO_TARGETS_H

//#include "../utils/utils.h"

#include <cmath>
#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>

/* Helper struct to keep data for lattice params, etc at pressures. */
struct TrgData {

    // Keep physical data here
    // Press in GPa, lata in Angstrom
    // structure,lata,wlata,ecoh,w_ecoh,FAIL
    // FAIL is true if the calculation failed for some reason
    std::map<std::string,std::tuple<
        std::string,double,double,double,double,bool>> m1;

    // Unrelaxed vacancy
    std::map<std::string,std::tuple<
        double,double,double,double>> m2; // UNRELAXED Evf,wEvf,Hvf,wHvf

    // Relaxed vacancy
    std::map<std::string,std::tuple<
        double,double,double,double>> m3; // RELAXED Evf,wEvf,Hvf,wHvf


    // Unrelaxed surface
    std::map<int,std::tuple<double,double>> m4;    // [hkl]=[E_hkl,wE_hkl]
    // Relaxed surface
    std::map<int,std::tuple<double,double>> m5;    // [hkl]=[E_hkl,wE_hkl]

    // Elastic constants
    std::map<int,std::tuple<double,double>> m6; // [ij]=[C_ij, wC_ij]

    // Cohesieve energy vs lattice param
    std::map<std::string,std::tuple<double,double>> m7; // [a]=[E,wE]

    // This return the first structure found in a map
    // TODO we assume just one type of structure for now
    std::string get_structure() {
        return std::get<0>(m1.begin()->second);
    };

    // Getters
    // m1
    std::string get_structure(const std::string P) { return std::get<0>(m1.at(P)); };
    double get_lata(const std::string P) { return std::get<1>(m1.at(P)); };
    double get_wlata(const std::string P) { return std::get<2>(m1.at(P)); };
    double get_ecoh(const std::string P) { return std::get<3>(m1.at(P)); };
    double get_wecoh(const std::string P) { return std::get<4>(m1.at(P)); };
    bool is_fail(const std::string P) { return std::get<5>(m1.at(P)); };

    // m2 Unrelaxed vacany
    double get_uEvf(const std::string P) { return std::get<0>(m2.at(P)); };
    double get_uwEvf(const std::string P) { return std::get<1>(m2.at(P)); };
    double get_uHvf(const std::string P) { return std::get<2>(m2.at(P)); };
    double get_uwHvf(const std::string P) { return std::get<3>(m2.at(P)); };

    // m3 Relaxed vacany
    double get_Evf(const std::string P) { return std::get<0>(m3.at(P)); };
    double get_wEvf(const std::string P) { return std::get<1>(m3.at(P)); };
    double get_Hvf(const std::string P) { return std::get<2>(m3.at(P)); };
    double get_wHvf(const std::string P) { return std::get<3>(m3.at(P)); };

    // m4 Unrelaxed surface
    double get_usurf(const int hkl) {return std::get<0>(m4.at(hkl)); }
    double get_uwsurf(const int hkl) {return std::get<1>(m4.at(hkl)); }

    // m5 Relaxed surface
    double get_surf(const int hkl) {return std::get<0>(m5.at(hkl)); }
    double get_wsurf(const int hkl) {return std::get<1>(m5.at(hkl)); }

    // m6 Elastic constant
    double get_elastic(const int ij) {return std::get<0>(m6.at(ij)); }
    double get_welastic(const int ij) {return std::get<1>(m6.at(ij)); }

    // m7 Energy vs lattice param
    double get_envol(const std::string s) {return std::get<0>(m7.at(s)); }
    double get_wenvol(const std::string s) {return std::get<1>(m7.at(s)); }

    // Setters
    // m1
    void set_structure(const std::string P, const std::string struc) {
        std::get<0>(m1[P])=struc;
    }
    void set_lata(const std::string P, const double lata) {
        std::get<1>(m1[P])=lata;
    }
    void set_w_lata(const std::string P, const double w_lata) {
        std::get<2>(m1[P])=w_lata;
    }
    void set_ecoh(const std::string P, const double ecoh) {
        std::get<3>(m1[P])=ecoh;
    }
    void set_w_ecoh(const std::string P, const double w_ecoh) {
        std::get<4>(m1[P])=w_ecoh;
    }
    void set_fail(const std::string P, const bool fail) {
        std::get<5>(m1[P])=fail;
    }

    // m2 Unrelaxed vacany
    void set_uEvf(const std::string P, const double E_vf) {
        std::get<0>(m2[P])=E_vf;
    }
    void set_uwEvf(const std::string P, const double wE_vf) {
        std::get<1>(m2[P])=wE_vf;
    }
    void set_uHvf(const std::string P, const double H_vf) {
        std::get<2>(m2[P])=H_vf;
    }
    void set_uwHvf(const std::string P, const double wH_vf) {
        std::get<3>(m2[P])=wH_vf;
    }

    // m3 Relaxed vacancy
    void set_Evf(const std::string P, const double E_vf) {
        std::get<0>(m3[P])=E_vf;
    }
    void set_wEvf(const std::string P, const double wE_vf) {
        std::get<1>(m3[P])=wE_vf;
    }
    void set_Hvf(const std::string P, const double H_vf) {
        std::get<2>(m3[P])=H_vf;
    }
    void set_wHvf(const std::string P, const double wH_vf) {
        std::get<3>(m3[P])=wH_vf;
    }

    // m4 Unrelaxed surface
    void set_usurf(const int hkl, const double s) {
        std::get<0>(m4[hkl])=s;
    }
    void set_uwsurf(const int hkl, const double s) {
        std::get<1>(m4[hkl])=s;
    }
    // m5 Relaxed surface
    void set_surf(const int hkl, const double s) {
        std::get<0>(m5[hkl])=s;
    }
    void set_wsurf(const int hkl, const double s) {
        std::get<1>(m5[hkl])=s;
    }
    // m6 Elastic constants
    void set_elastic(const int ij, const double s) {
        std::get<0>(m6[ij])=s;
    }
    void set_welastic(const int ij, const double s) {
        std::get<1>(m6[ij])=s;
    }
    // m7 Energy cs lattive param
    void set_envol(const std::string s, const double E) {
        std::get<0>(m7[s])=E;
    }
    void set_wenvol(const std::string s, const double wE) {
        std::get<1>(m7[s])=wE;
    }

    std::vector<double> d2 = {0,1,0,1}; // ermse,wermse,srmse,wsrmse

    double get_ermse() { return d2[0]; };
    double get_wermse() { return d2[1]; };
    double get_srmse() { return d2[2]; };
    double get_wsrmse() { return d2[3]; };

    void set_ermse(const double v) {
        d2[0]=v;
    }
    void set_wermse(const double v) {
        d2[1]=v;
    }
    void set_srmse(const double v) {
        d2[2]=v;
    }
    void set_wsrmse(const double v) {
        d2[3]=v;
    }

    // Sizes
    size_t lata_size() {
        return m1.size();
    }
    size_t uvacf_size() {
        return m2.size();
    }
    size_t vacf_size() {
        return m3.size();
    }
    size_t usurf_size() {
        return m4.size();
    }
    size_t surf_size() {
        return m5.size();
    }
    size_t elastic_size() {
        return m6.size();
    }
    size_t envol_size() {
        return m7.size();
    }

    TrgData()
    { }

};

/** Read and store target values for the optimiser.
 *
 * Allowed keys:
 *
 *     POWER <int> N #e.g. POWER 2 (defaul is 1). N-power in a loss function
 *     CONFIGMAP <string> N1 N2 ... #e.g. RCUT2B SGRID2B SGRID2B CGRID2B
 *     CONFIGMIN <double> N1 N2 ... #e.g. 3.0 0.1 0.2 0.0
 *     CONFIGMAX <double> N1 N2 ... #e.g. 6.0 2.0 2.1 3.0
 *     MAXSCORE <double> N #If at any stage score is greater than this
 *                           move to next iteration
 *     ERMSE <double> N #Max acceptable value for Energy RMSE
 *     MAXCALLS <size_t> N #Max nummber of optimiser calls. Default: inf
 *     MAXTIME <int,int> N S #N Max execution time (in sec) for a single LAMMPS simulation.
 *                           #S check every S steps during minimisation
 *                           #N default is 10 seconds, adjust up for complex models
 *                           #E default is 1
 *     EPS <double> N #Controls optimiser precision. Default 1e-2
 *                     It is sensible to run tests with larger values
 *                     and then reduce it for production.
 *     ECOH <double> N #Target for the cohesieve energy
 *     LATA <string> structure  <double> N #Target for the bcc/fcc/hcp lattice parameter
 *     EVACF <double> P A N #Target for vacancy formation energy at pressure P,
 *                           N-target value, A-guess for lattice param at P
 *                           (better guess = faster lammps script execution)
 *     HVACF #as above but for Enthalpy
 *     SURFACE <int> S <double> N #e.g. SURFACE 110 2.4
 *              # implemented surface calculations: 100, 110, 111
 *     ELASTIC <int> V <double> N #V-Voigt notation, e.g ELASTIC 11 260
 *
 *
 * Requires:
 *
 *     CONFIGMAP.size()==CONFIGMIN.size()==CONFIGMAX.size()
 *     Ordering of values in the above matters.
 */
class HPOTargets {

    public:
        double eps=1e-2; // control optimiser precision
        size_t maxcalls=std::numeric_limits<size_t>::max(); // max number of calls
        double MAX_SCORE=1000;
        std::string MAX_TIME="00:00:10 every 1";
        std::vector<std::string> configmap;
        std::vector<double> configmin;
        std::vector<double> configmax;

        bool use_ermse=false;
        bool use_srmse=false;
        bool use_lata=false;
        bool use_ecoh=false;
        bool use_surface=false;
        bool use_usurface=false;
        bool use_elastic=false;
        bool use_uEvf=false;
        bool use_uHvf=false;
        bool use_envol=false;
        //const std::vector<int> elastic_voigt
        //    ={11,22,33,12,13,23,44,55,66,14,15,16,24,25,26,34,35,36,45,46,56};

        int POW=1;


        // Default weights
        /*non const*/ double w_lata = 100;
        /*non const*/ double w_ecoh = 100;
        const double w_surface = 10;
        const double w_elastic = 1;
        const double w_uevacf = 10;
        const double w_uhvacf = 10;
        const double w_envol = 10;

        TrgData data;

        std::map<std::string,std::string> cfg;


        HPOTargets(std::string fn)
    {
        read(fn);
    }

        void read(std::string fn) {
            std::ifstream ifs(fn);
            if (!ifs.good())
                throw std::runtime_error("Pot file does not exists.\n");
            std::string s;
            double a[100];
            while(std::getline(ifs,s)) {
                std::istringstream iss(s);
                std::string key;
                iss >> key;
                // skip empty lines
                if(key.empty()) continue;

                // skip comments starting with #
                if(key[0]=='#') continue;

                else if (key=="EPS") {
                    iss >> eps;
                }
                else if (key=="MAXCALLS") {
                    iss >> maxcalls;
                }
                else if (key=="MAXSCORE") {
                    iss >> MAX_SCORE;
                }
                else if (key=="POWER") {
                    iss >> POW;
                }
                else if (key=="MAXTIME") {
                    int tsec;
                    int tevery;
                    iss >> tsec >> tevery;
                    int s,m,h;
                    h=tsec/3600;
                    m=(tsec-h*3600)/60;
                    s=tsec-h*3600-m*60;
                    MAX_TIME=std::to_string(h) + ":" + std::to_string(m) + ":" + std::to_string(s);
                    MAX_TIME+=" every " + std::to_string(tevery);
                }
                else if (key=="ERMSE") {
                    iss >> a[0];
                    data.set_ermse(a[0]);
                    if (iss.rdbuf()->in_avail()) {
                        iss >> a[1];
                        data.set_wermse(a[1]);
                    }
                    use_ermse=true;
                }
                else if (key=="SRMSE") {
                    iss >> a[0];
                    data.set_srmse(a[0]);
                    if (iss.rdbuf()->in_avail()) {
                        iss >> a[1];
                        data.set_wsrmse(a[1]);
                    }
                    use_srmse=true;
                }
                else if (key=="LATA") {
                    // format LATA P FCC 3.304 WEIGHT
                    std::string struc,press;
                    iss >> press >> struc >> a[1];
                    //if (iss.rdbuf()->in_avail()) iss >> w_lata;
                    if (iss.rdbuf()->in_avail()) iss >> a[2];
                    else a[2]=w_lata;
                    data.set_structure(press,struc);
                    data.set_lata(press,a[1]);
                    data.set_w_lata(press,a[2]);
                    use_lata=true;
                }
                else if (key=="ECOH") {
                    // format ECOH P ECOH WEIGHT
                    std::string press;
                    iss >> press >> a[1];
                    if (iss.rdbuf()->in_avail()) iss >> a[2];
                    else a[2]=w_ecoh;
                    data.set_ecoh(press,a[1]);
                    data.set_w_ecoh(press,a[2]);
                    use_ecoh=true;
                }
                else if (key=="USURFACE") {
                    int hkl;
                    iss >> hkl;
                    iss >> a[0];
                    if (iss.rdbuf()->in_avail()) iss >> a[1];
                    else a[1]=w_surface;
                    if (hkl==100 || hkl==110 || hkl==111) {
                        data.set_usurf(hkl,a[0]);
                        data.set_uwsurf(hkl,a[1]);
                        use_usurface=true;
                    }
                }
                else if (key=="SURFACE") {
                    int hkl;
                    iss >> hkl;
                    iss >> a[0];
                    if (iss.rdbuf()->in_avail()) iss >> a[1];
                    else a[1]=w_surface;
                    if (hkl==100 || hkl==110 || hkl==111) {
                        data.set_surf(hkl,a[0]);
                        data.set_wsurf(hkl,a[1]);
                        use_surface=true;
                    }
                }
                else if (key=="ENVOL") {
                    std::string lata;
                    iss >> lata >> a[0];
                    if (iss.rdbuf()->in_avail()) iss >> a[1];
                    else a[1]=w_envol;
                    data.set_envol(lata,a[0]);
                    data.set_wenvol(lata,a[1]);
                    use_envol=true;
                }
                else if (key=="ELASTIC") {
                    int ij;
                    iss >> ij;
                    iss >> a[0];
                    if (iss.rdbuf()->in_avail()) iss >> a[1];
                    else a[1]=w_elastic;
                    data.set_elastic(ij,a[0]);
                    data.set_welastic(ij,a[1]);
                    use_elastic=true;
                }
                else if (key=="UEVACF") {
                    std::string press;
                    iss >> press >> a[1];
                    if (iss.rdbuf()->in_avail()) iss >> a[2];
                    else a[2]=w_uevacf;
                    data.set_uEvf(press,a[1]);
                    data.set_uwEvf(press,a[2]);
                    use_uEvf=true;
                }
                else if (key=="UHVACF") {
                    std::string press;
                    iss >> press >> a[1];
                    if (iss.rdbuf()->in_avail()) iss >> a[2];
                    else a[2]=w_uhvacf;
                    data.set_uHvf(press,a[1]);
                    data.set_uwHvf(press,a[2]);
                    use_uHvf=true;
                }
                else if (key=="CONFIGMAP") {
                    while (iss >> s) {
                        configmap.push_back(s);
                    }
                }
                else if (key=="CONFIGMAX") {
                    while (iss >> a[0]) {
                        configmax.push_back(a[0]);
                    }
                }
                else if (key=="CONFIGMIN") {
                    while (iss >> a[0]) {
                        configmin.push_back(a[0]);
                    }
                }
                else if (key=="CONFIG") {
                    std::string key;
                    std::string value;
                    iss >> key >> value;
                    cfg[key]=value;
                }
            }
        }

};
#endif

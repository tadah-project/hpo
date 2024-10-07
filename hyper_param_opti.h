#ifndef HYPER_PARAM_OPTI_H
#define HYPER_PARAM_OPTI_H

#include "../CORE/config/config.h"
#include "../MLIP/structure_db.h"
#include "../MLIP/nn_finder.h"
#include "../MODELS/dc_selector.h"
#include "../MLIP/models/m_tadah_base.h"
#include "../MLIP/design_matrix/functions/dm_function_base.h"
#include "../MLIP/descriptors_calc.h"
#include "../MLIP/analytics/analytics.h"
#include "../MLIP/trainer.h"
#include "../CORE/utils/utils.h"
#include "hpo_targets.h"
#include "glf.h"
#include "lammps_scripts.h"

#include <chrono>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#if defined(TADAH_ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

/* General comments:
 * - lammps must be compiled as a shared library
 * - lammps must be compiled with c++ exceptions: -DLAMMPS_EXCEPTIONS
 * - when compiling code which uses below classes link -llammps -ldlib -fopenmp
 * - The code might compile without -fopenmp but on some systems this causes
 *   numerical instabilties. This is (probably) only true for HyperParamOpti class.
 */

class HyperParamOptiBase {
};

/**
 * Usage example:
 *
 *     Config config("config.train");
 *     HPOTargets trg("targets");
 *     HyperParamOpti<dlib::matrix<double,0,1>> hpo(config,trg);
 *     # the object is ready to be passed to the optimiser.
 *
 * Supported Config keys:
 * RCUT2B,RCUTMB,CGRID2B,CGRIDMB,SGRID2B,SGRIDMB,CEMBFUNC,SEMBFUNC
 *
 * This class is very fragile.
 * There is no error checking of the user input whatsoever.
 *
 */
template <typename T, typename GLF, typename R>
class HyperParamOpti: public HyperParamOptiBase {
    private:
        std::string element = "Ta";     // TODO
        bool fail=false;
        R runner;
        LammpsScripts scr;
        HPOTargets &trg;
        GLF glf;
        std::string delim=" ";
        double best_score=std::numeric_limits<double>::max();
        Config config;
        StructureDB stdb;
        size_t it=0;    // iteration number
        size_t pevery=std::numeric_limits<size_t>::max();   // Dump potential every this steps
        std::string pevery_dir;

        // Expand parameter pack with delim and return as string.
        template <typename... Args>
            std::string as_string(Args... args) {
                std::stringstream ss;
                using expander = int[];
                (void)expander{0, (void(ss << std::setprecision(15) << std::forward<Args>(args) << delim), 0)...};
                return ss.str().substr(0,ss.str().size()-delim.size()); // remove last delim
            }

        // helper method to update config cutoff
        void rc (std::string s, double rc) {
            config.remove(s);
            config.add(s,rc);
        };

        // helper method to update config grid
        template <typename... Args>
            void grid(std::string gridname, Args ...args){
                config.remove(gridname);
                config.add(gridname,as_string(args...));
            }


        void update_config (T &p) {

            for (auto &c : trg.configmap) {
                config.remove(c);
            }

            for (size_t i=0; i<trg.configmap.size();++i) {
                std::string s = trg.configmap[i];
                if (s=="MPARAMS") {
                    config.remove(s);
                    config.add(s,p(i));
                }
                if (s=="SBASIS") {
                    config.remove(s);
                    config.add(s,p(i));
                }
                if (s=="LAMBDA") {
                    config.remove(s);
                    config.add(s,p(i));
                }
                if (s=="RCUT2B" || s=="RCUTMB") rc(s,p(i));
                if (s=="CGRID2B" || s=="CGRIDMB") config.add(s,p(i));
                if (s=="SGRID2B" || s=="SGRIDMB") config.add(s,p(i));
                if (s=="CEMBFUNC" || s=="SEMBFUNC") config.add(s,p(i));

                if (s=="SWEIGHT") {
                    config.remove(s);
                    config.add(s,p(i));
                }
                if (s=="FWEIGHT") {
                    config.remove(s);
                    config.add(s,p(i));
                }

            }
        }

        int calc_lata(TrgData &pdata) {
#if defined(TADAH_ENABLE_OPENMP)
#pragma omp parallel for shared(runner,fail,pdata,trg)
#endif
            for(size_t i = 0; i < trg.data.lata_size(); i++) {
                auto v = trg.data.m1.begin();
                advance(v, i);
                const std::string P = v->first;
                // Only lata calculations should set this to false.
                // When a calculation in the chain fails
                // the flag is set to fail and should stay
                // until the next iteration of the GOA.
                pdata.set_fail(P,false);
                int thread = omp_get_thread_num();
                auto lmp = scr.run_lata(runner,P, trg.data.get_structure(P),
                        trg.data.get_lata(P), element,
                        "log.lata"+std::to_string(thread),
                        trg.MAX_TIME);
                if (!lmp) {
                    fail = true;
                    pdata.set_fail(P,true);
                }
                else {
                    double a= stod(runner.get_variable(lmp,"length"));
                    double ecoh=stod(runner.get_variable(lmp,"ecoh"));

                    pdata.set_lata(P,a);
                    pdata.set_ecoh(P,ecoh);
                }
                if (lmp) delete lmp;
            }

            return 0;
        }
        int calc_uvacf(TrgData &pdata) {
#if defined(TADAH_ENABLE_OPENMP)
#pragma omp parallel for shared(runner,fail,pdata,trg)
#endif
            for(size_t i = 0; i < trg.data.uvacf_size(); i++) {
                auto v = trg.data.m2.begin();
                advance(v, i);
                const std::string P = v->first;
                int thread = omp_get_thread_num();
                auto lmp = scr.run_uvacf(runner,P,trg.data.get_lata(P),
                        "log.uvacf"+std::to_string(thread),
                        trg.MAX_TIME);
                if (!lmp) {
                    fail = true;
                    pdata.set_fail(P,true);
                }
                else {
                    //double press= stod(runner.get_variable(lmp,"PRESS"));
                    //double a= stod(runner.get_variable(lmp,"latafinal"));
                    double E_v=stod(runner.get_variable(lmp,"Ev"));
                    double H_v=stod(runner.get_variable(lmp,"Hv"));

                    pdata.set_uEvf(P,E_v);
                    pdata.set_uHvf(P,H_v);
                }
                if (lmp) delete lmp;
            }
            return 0;
        }
        int calc_usurface(TrgData &pdata) {
#if defined(TADAH_ENABLE_OPENMP)
#pragma omp parallel for shared(runner,fail,pdata,trg)
#endif
            for(size_t i = 0; i < trg.data.usurf_size(); i++) {
                auto v = trg.data.m4.begin();
                advance(v, i);
                const int hkl = v->first;
                int thread = omp_get_thread_num();
                auto lmp = scr.run_usurface(runner,hkl,
                        trg.data.get_lata("0"),
                        trg.data.get_structure("0"), element,
                        "log.lata"+std::to_string(thread),
                        trg.MAX_TIME);
                if (!lmp) {
                    fail = true;
                    pdata.set_fail("0",true);
                }
                else {
                    double E_hkl=stod(runner.get_variable(lmp,"E_f2"));

                    pdata.set_usurf(hkl,E_hkl);
                }
                if (lmp) delete lmp;
            }

            return 0;
        }
        int calc_surface(TrgData &pdata) {
#if defined(TADAH_ENABLE_OPENMP)
#pragma omp parallel for shared(runner,fail,pdata,trg)
#endif
            for(size_t i = 0; i < trg.data.surf_size(); i++) {
                auto v = trg.data.m5.begin();
                advance(v, i);
                const int hkl = v->first;
                int thread = omp_get_thread_num();
                auto lmp = scr.run_surface(runner,hkl,
                        trg.data.get_lata("0"),
                        trg.data.get_structure("0"), element,
                        "log.lata"+std::to_string(thread),
                        trg.MAX_TIME);
                if (!lmp) {
                    fail = true;
                    pdata.set_fail("0",true);
                }
                else {
                    double E_hkl=stod(runner.get_variable(lmp,"E_f2"));

                    pdata.set_surf(hkl,E_hkl);
                }
                if (lmp) delete lmp;
            }

            return 0;
        }
        int calc_envol(TrgData &pdata) {
#if defined(TADAH_ENABLE_OPENMP)
#pragma omp parallel for shared(runner,fail,pdata,trg)
#endif
            for(size_t i = 0; i < trg.data.envol_size(); i++) {
                auto v = trg.data.m7.begin();
                advance(v, i);
                const std::string lata = v->first;
                int thread = omp_get_thread_num();
                auto lmp = scr.run_envol(runner,lata,
                        trg.data.get_structure(),
                        element, "log.lata"+std::to_string(thread),
                        trg.MAX_TIME);
                if (!lmp) {
                    fail = true;
                    //pdata.set_fail(P,true); // this cannot fail
                }
                else {
                    double ecoh=stod(runner.get_variable(lmp,"ecoh"));

                    pdata.set_envol(lata,ecoh);
                }
                if (lmp) delete lmp;
            }

            return 0;
        }
    public:
        HyperParamOpti(Config c, HPOTargets &trg):
            trg(trg),
            glf(trg),
            config(c),
            stdb(c)  // TODO can't have single stdb for large datasets
    {
        // TODO this won't work with large datasets
        NNFinder nn(config);
        nn.calc(stdb);

        if (config.exist("HPOEVERY")) {
            pevery_dir=config.get<std::string>("HPOEVERY",0);
            pevery=config.get<size_t>("HPOEVERY",1);
        }

        // update CONFIG - constants specific to this target
        for (const auto p : trg.cfg) {
            std::string key=p.first;
            std::string val=p.second;
            config.remove(key);
            config.add(key,val);
        }
    }

        //double operator() (T args) {
        double operator() (T p) {
            it++;
            using std::chrono::high_resolution_clock;
            using std::chrono::duration_cast;
            using std::chrono::duration;
            using std::chrono::milliseconds;

            fail=false;

            auto t1 = high_resolution_clock::now();
            //p = std::vector<double>(args.begin(), args.end());
            update_config(p);

            Config config_temp = config;
            config_temp.postprocess();

            // TODO dump temp trainig config to disk such that other MPI processes
            // can read it

            // TODO broadcast location of the temp config file
            // Train new potential with updated settings

            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> t21 = t2 - t1;
            //std::cout << "T1: " << t21.count() << "ms\n";

  std::cout << "COMMON DONE " << it << std::endl;
#ifdef TADAH_BUILD_MPI
  std::cout << "HOST START: " << it << std::endl;
  int rank;
  int ncpu;
  MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Trainer tr(config); // TODO use updated config
  tr.init(rank, ncpu);
  // HOST is waiting for workers requests
  MPI_Trainer_HOST HOST(tr, rank, ncpu);
  HOST.prep_wpckgs();
  std::cout << "HOST INIT DONE: " << it << std::endl;
  while (true) {
    if (!HOST.has_packages()) {
      // no more packages, just skip remaining workers
      // we will collect remaining data and release them outside of this loop
      break;
    }

    // probe ANY request from ANY worker
    tr.probe();

    if (tr.tag==MPI_Trainer::WORK_TAG) {
      HOST.work_tag();
    }
    else if (tr.tag==MPI_Trainer::DATA_TAG) {
      HOST.data_tag();
    }
    else {
      throw std::runtime_error("HOST: Unexpected request from "
          + std::to_string(tr.worker));
    }
  }

  // work finised, collect remaining data and release all workers
  int count=1;  // count number of release workers, skip host
  while(true) {
    tr.probe();
    if (tr.tag==MPI_Trainer::DATA_TAG) {
      HOST.data_tag();
    }
    else {
      HOST.release_tag(count);
    }
    if (count==ncpu) {  break; }  // count starts from 1
  }
  tr.solve();
#else 
  std::cout << "TADAH_BUILD_MPI ndef?" << it << std::endl;
            Trainer tr(config_temp);
            tr.train(stdb);
#endif

            // End of training

            auto t3 = high_resolution_clock::now();
            duration<double, std::milli> t32 = t3 - t2;
            //std::cout << "T2: " << t32.count() << "ms\n";

            // This potential will be used by LAMMPS
            Config pot_tadah = tr.model->get_param_file();
            std::ofstream outfile;
            outfile.open ("pot.tadah");
            outfile << pot_tadah;
            outfile.close();

            auto t4 = high_resolution_clock::now();
            duration<double, std::milli> t43 = t4 - t3;
            //std::cout << "T3: " << t43.count() << "ms\n";

            // Do not remove me for a while...
            //pot_tadah.add("VERBOSE",1);
            //pot_tadah.add("FORCE","false");
            //pot_tadah.add("STRESS","false");
            //M_Tadah_Base *modelp = CONFIG::factory<M_Tadah_Base,DM_Function_Base&,Config&>
            //    (pot_tadah.get<std::string>("MODEL",0),*fb,pot_tadah);
            //StructureDB stpred = modelp->predict(pot_tadah,stdb,dc);

            // TODO test dataset must be smaller than trainig for MPI version
            StructureDB stpred = tr.model->predict(stdb);

            Analytics a(stdb,stpred);

            TrgData pdata = trg.data;
            if (trg.use_lata | trg.use_ecoh | trg.use_surface) {
                calc_lata(pdata);
            }

            // Vacancy formation energy/enthalpy at selected pressures
            // TODO for MPI version accumulate all scripts
            // and distribute over workers
            if (trg.use_uEvf || trg.use_uHvf) {
                calc_uvacf(pdata);
            }
            if (trg.use_usurface) {
                calc_usurface(pdata);
            }
            if (trg.use_surface) {
                calc_surface(pdata);
            }
            if (trg.use_envol) {
                calc_envol(pdata);
            }

            double tloss = glf.total_loss(a,p,pdata,fail);

            // Dump potential if best
            if (tloss<best_score) {
                best_score = tloss;
                std::ofstream outfile;
                outfile.open ("best_pot.tadah");
                outfile << pot_tadah;
                outfile.close();
            }

            // Dump potential pevery steps
            // Controled by HPOEVERY
            if (it%pevery==0) {
                std::ofstream outfile;
                outfile.open (pevery_dir+"/pot_"+std::to_string(it)+".tadah");
                outfile << pot_tadah;
                outfile.close();
            }

            auto t5 = high_resolution_clock::now();
            duration<double, std::milli> t54 = t5 - t4;
            //std::cout << "T4: " << t54.count() << "ms\n";
            return tloss;
        }
};
#endif

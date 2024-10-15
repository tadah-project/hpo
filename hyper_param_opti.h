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
//#include "glf.h"

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

#ifndef TADAH_BUILD_MPI
#define MPI_Comm int
#endif

/* General comments:
 * - lammps must be compiled as a shared library
 * - lammps must be compiled with c++ exceptions: -DLAMMPS_EXCEPTIONS
 * - when compiling code which uses below classes link -llammps -ldlib -fopenmp
 */

/**
 * Usage example:
 *
 *     Config config("config.train");
 *     HPOTargets trg(config, "targets");
 *     HyperParamOpti<dlib::matrix<double,0,1>,LammpsRunner> 
   *     hpo(config,trg,"config_val",lmp_comm);
 *     # the object is ready to be passed to the optimiser.
 *
 */
template <typename T, typename R>
class HyperParamOpti {
  private:
    R runner;
    HPOTargets &trg;
    //GLF glf;
    double best_score=std::numeric_limits<double>::max();
    // Dump potential pevery steps
    size_t pevery=std::numeric_limits<size_t>::max(); 
    Config config;  // make a copy
    StructureDB val_stdb; // validation dataset
    StructureDB train_stdb; // train dataset NON MPI only
    size_t it=0;    // iteration number
    std::string pevery_dir;

    void update_config (T &p) {
      for (auto &m : trg.optim_map) {
        config.remove(m);
      }

      for (size_t i=0; i<trg.optim_map.size();++i) {
        const std::string &s = trg.optim_map[i];
        config.add(s,p(i));
      }
    }

    //template <typename Tuple>
    //int calc_custom(TrgData &pdata, Tuple && lmp_params)
    //{

    //  auto lmp = std::apply([this](auto&&... args) {
    //      return runner.run(
    //          lmp_comm, std::forward<decltype(args)>(args)...);
    //      }, lmp_params);


    //  if (!lmp) {
    //    fail = true;
    //    pdata.set_fail(P,true);
    //  }
    //  else {
    //    double a= stod(runner.get_variable(lmp,"length"));
    //    double ecoh=stod(runner.get_variable(lmp,"ecoh"));

    //    pdata.set_lata(P,a);
    //    pdata.set_ecoh(P,ecoh);
    //  }
    //  if (lmp) delete lmp;
    //  return 0;
    //}
    MPI_Comm lmp_comm;
  public:
    HyperParamOpti(Config c, HPOTargets &trg, std::string &val_file,
        MPI_Comm &lmp_comm):
      trg(trg),
      //glf(trg),
      config(c),
      lmp_comm(lmp_comm)
  {
    Config val_config(val_file);
    val_stdb.add(val_config);

#ifndef TADAH_BUILD_MPI
    train_stdb.add(config);
#endif

    //if (config.exist("HPOEVERY")) {
    //  pevery_dir=config.get<std::string>("HPOEVERY",0);
    //  pevery=config.get<size_t>("HPOEVERY",1);
    //}

    // update CONFIG - constants specific to this target
    //for (const auto &p : trg.cfg) {
    //  std::string key=p.first;
    //  std::string val=p.second;
    //  config.remove(key);
    //  config.add(key,val);
    //}
  }

    //double operator() (T args) {
    double operator() (T p) {
      it++;
      using std::chrono::high_resolution_clock;
      using std::chrono::duration_cast;
      using std::chrono::duration;
      using std::chrono::milliseconds;

      //auto t1 = high_resolution_clock::now();
      //p = std::vector<double>(args.begin(), args.end());
      update_config(p);

      Config config_temp = config;
      config_temp.postprocess();

      // Train new potential with updated settings
      //auto t2 = high_resolution_clock::now();
      //duration<double, std::milli> t21 = t2 - t1;
      //std::cout << "T1: " << t21.count() << "ms\n";

#ifdef TADAH_BUILD_MPI
      int rank;
      int ncpu;
      MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // New config file must be communicated to all processes
      std::vector<char> cs = config_temp.serialize();
      int cs_size = cs.size();
      MPI_Bcast(&cs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(cs.data(), cs_size, MPI_CHAR, 0, MPI_COMM_WORLD);

      // HOST is waiting for workers requests
      MPI_Trainer_HOST HOST(config_temp, rank, ncpu);
      HOST.prep_wpckgs();
      while (true) {
        // Exit loop if there are no more packages
        if (!HOST.has_packages()) break;

        // probe ANY request from ANY worker
        HOST.probe();

        if (HOST.tag==MPI_Trainer::WORK_TAG) {
          HOST.work_tag();
        }
        else if (HOST.tag==MPI_Trainer::DATA_TAG) {
          HOST.data_tag();
        }
        else {
          throw std::runtime_error("HOST: Unexpected request from "
              + std::to_string(HOST.worker));
        }
      }

      // work finised, collect remaining data and release all workers
      int count=1;  // count number of release workers, skip host
      while(true) {
        HOST.probe();
        if (HOST.tag==MPI_Trainer::DATA_TAG) {
          HOST.data_tag();
        }
        else {
          HOST.release_tag(count);
        }
        if (count==ncpu) {  break; }  // count starts from 1
      }

      HOST.solve();

      Config pot_tadah = HOST.model->get_param_file();

      std::cout << HOST.model->weights << std::endl;
#else 
      Trainer HOST(config_temp);
      train_stdb.clear_nn();
      HOST.train(train_stdb);
      Config pot_tadah = HOST.model->get_param_file();
#endif
      // End of training

      // This potential will be used by LAMMPS
      std::ofstream outfile;
      outfile.open ("pot.tadah");
      outfile << pot_tadah;
      outfile.close();

      pot_tadah.set_defaults();
      pot_tadah.postprocess();

      //DC_Selector DCS(pot_tadah);
      //val_stdb.clear_nn();
      //NNFinder nnf(pot_tadah);
      //nnf.calc(val_stdb);
      //DescriptorsCalc<> dc
      //  (pot_tadah,*DCS.d2b,*DCS.d3b,*DCS.dmb,*DCS.c2b,*DCS.c3b,*DCS.cmb);
      //DM_Function_Base *fb = CONFIG::factory
      //  <DM_Function_Base,Config&>
      //  (pot_tadah.get<std::string>("MODEL",1),pot_tadah);
      //M_Tadah_Base *modelp = CONFIG::factory
      //  <M_Tadah_Base,DM_Function_Base&,Config&>
      //  (pot_tadah.get<std::string>("MODEL",0),*fb,pot_tadah);
      //StructureDB stpred = modelp->predict(pot_tadah,val_stdb,dc);
      StructureDB stpred = HOST.model->predict(val_stdb);

      Analytics a(val_stdb,stpred);

      //TrgData pdata = trg.data;

      //  //calc_lata(pdata);
      //std::string lata_guess="3.304";
      //std::string P="1";
      //std::string structure="fcc";
      //std::string element="Ta";
      //std::string log="log.lata";
      //std::string MAX_TIME=trg.MAX_TIME;
      //std::string infile=scr.scripts["tadah.lata"];

      //auto params = std::make_tuple(
      //    infile.c_str(),
      //    log.c_str(),
      //    "-var", "lata_guess", lata_guess.c_str(),
      //    "-var", "P", P.c_str(),
      //    "-var", "tout",MAX_TIME.c_str(),
      //    "-var", "structure", structure.c_str(),
      //    "-var", "element", element.c_str()
      //    );

      // Parse target file and construct a map
      // [script_name] = [input=[], output=[]]
      // return calculated Loss or FAILSCORE

      //std::unordered_map<std::string, std::vector<std::string>> lmp_scripts;
      //double loss = calc_custom(params);

      double tloss = 0;
      //double tloss = glf.total_loss(a,p,pdata,fail);
      if (trg.use_ermse) {
        double ermse = a.calc_e_rmse().mean(); // eV/atom
        ermse *= 1000; // meV/atom;
        double ermse_score = ermse*trg.w_ermse;
        tloss += ermse_score;
      }

      // Dump potential if best
      if (tloss<best_score) {
        best_score = tloss;
        std::ofstream outfile;
        outfile.open ("best_pot.tadah");
        pot_tadah.clear_internal_keys();
        outfile << pot_tadah;
        outfile.close();
      }

      // Dump potential pevery steps
      // Controled by HPOEVERY
      //if (it%pevery==0) {
      //  std::ofstream outfile;
      //  outfile.open (pevery_dir+"/pot_"+std::to_string(it)+".tadah");
      //  outfile << pot_tadah;
      //  outfile.close();
      //}

      //auto t5 = high_resolution_clock::now();
      //duration<double, std::milli> t54 = t5 - t4;
      //std::cout << "T4: " << t54.count() << "ms\n";
      return tloss;
    }

};
#endif

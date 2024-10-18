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

//#ifndef TADAH_BUILD_MPI
//#define MPI_Comm int
//#endif

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
   *     hpo(config,trg,"config_val");
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

  public:
    HyperParamOpti(Config c, HPOTargets &trg, std::string &val_file):
      trg(trg),
      //glf(trg),
      config(c)
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
      TrainerHost host(config_temp, rank, ncpu);
      host.run();
      host.solve();
#else 
      Trainer host(config_temp);
      train_stdb.clear_nn();
      host.train(train_stdb);
#endif
      // End of training

      double tloss = 0;

      // This potential will be used by LAMMPS
      Config pot_tadah = host.model->get_param_file();
      std::ofstream outfile;
      outfile.open ("pot.tadah");
      outfile << pot_tadah;
      outfile.close();

#ifdef TADAH_BUILD_MPI

      // if LAMMPS
      if (trg.LS.size() > 0) {
        const static int CONFIG_TAG = 4;
        const static int WAIT_TAG = 3;
        const static int RELEASE_TAG = 2;
        const static int DATA_TAG = 1;
        const static int WORK_TAG = 0;

        MPI_Status status;

        auto wpkgs = trg.LS;  // make a copy of LammpsSim vec
                              // LS is sorted
                              // Distribute LS to all workers?
        std::vector<char> buffer;
        //LammpsSim::serialise_vector_LammpsSim(wpkgs, buffer);
        //int buffer_size = buffer.size();
        //MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //MPI_Bcast(buffer.data(), buffer.size(), MPI_CHAR, 0, MPI_COMM_WORLD);

        if (wpkgs[0].ncpu>ncpu-1) {
          throw std::runtime_error   // TODO change to MPI
            ("Requested LAMMPS simulation size is bigger\n\\
             than the total number of workers.\n");
        }
        std::vector<int> available_workers;

        while (true) {
          if (!wpkgs.size()) {
            break;
          }

          MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, WORK_TAG, MPI_COMM_WORLD, &status);
          int worker = status.MPI_SOURCE;

          auto it = std::find(available_workers.begin(), available_workers.end(), worker);
          if (it == available_workers.end()) {  // new worker is available
            available_workers.push_back(worker);
            MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);
          }
          else {
            // release an eager worker as we will select a group of workers here
            MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);

            // There are no more new workers available at the moment.
            // Let's find the biggest job we can fit to them
            for (int j=0; j<wpkgs.size(); ++j) {
              int job_size = wpkgs[j].ncpu;
              if (job_size<=available_workers.size()) {
                std::vector<int> granks(available_workers.end() - job_size, available_workers.end());
                for (int worker: granks) {
                  MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD, &status);
                  MPI_Send(granks.data(), granks.size(), MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                  available_workers.pop_back();
                  // TODO just send j index of the wpkgs vector as every worker has a copy of LS
                  // send wpkgs[j].script
                  MPI_Send(wpkgs[j].script.c_str(), wpkgs[j].script.size() + 1, MPI_CHAR, worker, WORK_TAG, MPI_COMM_WORLD);
                  // send wpkgs[j].invar
                  // First, send the number of strings
                  int num_strings = wpkgs[j].invar.size();
                  MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);

                  // Send each string
                  for (const char* message : wpkgs[j].invar) {
                    int length = strlen(message) + 1; // +1 for the null terminator
                    MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                    MPI_Send(message, length, MPI_CHAR, worker, WORK_TAG, MPI_COMM_WORLD);
                  }
                }
                wpkgs.erase(wpkgs.begin() + j);
                break; // break outer for loop
              }
            }
          }
        }
        int count2 = 1;
        while (true) {
          MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, WORK_TAG, MPI_COMM_WORLD, &status);
          int worker=status.MPI_SOURCE;
          int tag = status.MPI_TAG;
          MPI_Send(NULL, 0, MPI_INT, worker, RELEASE_TAG, MPI_COMM_WORLD);
          count2++;
          if (count2==ncpu) break;
        }
      }
#else
      for (const auto& lsim : trg.LS) {
        auto *lmp = runner.run(MPI_COMM_WORLD, trg.LS[0].script, trg.LS[0].invar);
        if (!lmp) {
          tloss+=trg.LS[0].failscore;
        }
        else {
          for (const auto &v: trg.LS[0].varloss) {
            double lmp_tloss= std::stod(runner.get_variable(lmp,v));
            tloss+= lmp_tloss;
          }
        }
        if (lmp) delete lmp;
      }
#endif

      pot_tadah.set_defaults();
      pot_tadah.postprocess();

      val_stdb.clear_nn();
      NNFinder nnf(pot_tadah);
      nnf.calc(val_stdb);
      StructureDB stpred = host.model->predict(pot_tadah,val_stdb,host.dc);

      Analytics a(val_stdb,stpred);

      // By now we should have finished all computations
      // It's time to calculate total loss value by GLF

      // TODO computation of tloss to be performed by GLF
      //double tloss = glf.total_loss(a,p,pdata,fail);
      if (trg.use_ermse) {
        double ermse = a.calc_e_rmse().mean(); // eV/atom
        ermse *= 1000; // meV/atom;
        double ermse_score = ermse*trg.w_ermse;
        tloss += ermse_score;
      }
      if (trg.use_frmse) {
        double frmse = a.calc_f_rmse().mean();
        double frmse_score = frmse*trg.w_frmse;
        tloss += frmse_score;
      }
      if (trg.use_srmse) {
        double srmse = a.calc_s_rmse().mean();
        double srmse_score = srmse*trg.w_srmse;
        tloss += srmse_score;
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

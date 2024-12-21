#ifndef HPO_HOST_H
#define HPO_HOST_H

#include <cstddef>
#include <tadah/core/utils.h>
#include <tadah/core/config.h>
#include <tadah/models/dc_selector.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/trainer.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <vector>
#include <cmath> // ??
#include <stdlib.h> // ??
#include <iostream>
#include <fstream>
#include <iomanip>

#if defined(TADAH_ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_max_threads() { return 1;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

/* General comments:
 * - lammps must be compiled with c++ exceptions: -DLAMMPS_EXCEPTIONS
 */

/**
 * Usage example:
 *
 *     Config config("config.train");
 *     HPOTargets trg(config, "targets");
 *     HPO_Host<dlib::matrix<double,0,1>,LammpsRunner> 
 *     hpo(config,trg,"config_val");
 *     # the object is ready to be passed to the optimiser.
 *
 */
template <typename T, typename R>
class HPO_Host {
private:
  size_t cwidth=13;
  int pprec=4;
  double tloss, ermse, frmse, srmse;
  std::string loss_file = "loss.tadah";
  std::string params_file = "params.tadah";
  std::string outvar_file = "outvar.tadah";
  R runner;
  HPOTargets &trg;
  double best_score=std::numeric_limits<double>::max();
  Config config;  // make a copy
  StructureDB val_stdb; // validation dataset
  StructureDB train_stdb; // train dataset NON MPI only
  size_t it=0;    // iteration number

  void update_config (T &v) {
    for (auto &m : trg.optimap) {
      config.remove(m.first);
    }

    size_t idx=0;
    for (const auto &p:trg.optimap) {
      for (size_t i=0; i<p.second.opt.size(); ++i) {
        if (p.second.opt[i]) {
          config.add(p.first,v(idx++));
        } else {
          config.add(p.first,p.second.config[i]);
        }
      }
    }
  }

  bool use_lammps;

public:
  HPO_Host(Config c, HPOTargets &trg, std::string &val_file):
    trg(trg),
    //glf(trg),
    config(c)
  {
    Config val_config(val_file);
    val_stdb.add(val_config);

    use_lammps =  trg.LS.size() > 0;
#ifdef TADAH_BUILD_MPI
    MPI_Bcast(&use_lammps, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
#else
    train_stdb.add(config); // this func checks ATOMS and WATOMS keys
#endif

    init_loss();
    init_outvar();
    init_params();
  }

  double operator() (T p) {
    it++;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    update_config(p);

    Config config_temp = config;
    config_temp.postprocess();

    // Train new potential with updated settings
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

    // This potential will be used by LAMMPS
    Config pot_tadah = host.model->get_param_file();
    std::ofstream out_file;
    out_file.open ("pot.tadah");
    out_file << pot_tadah;
    out_file.close();

#ifdef TADAH_BUILD_MPI
    if (use_lammps)
      run(rank, ncpu);

#else
    int dummy_comm = 0;
#ifdef _OPENMP
    #pragma omp parallel for shared(trg)
#endif
    for (auto& lsim : trg.LS) {
      auto *lmp = runner.run(dummy_comm, lsim.script, lsim.invar);
      if (!lmp) { // LAMMPS failed
        for (size_t i = 0; i < lsim.varloss.size(); ++i)
          lsim.v_varloss[i] = lsim.failscore;
        for (size_t i = 0; i < lsim.outvar.size(); ++i)
          lsim.v_outvar[i] = "NaN";
      }
      else {
        for (size_t i = 0; i < lsim.varloss.size(); ++i) {
          const auto& v = lsim.varloss[i];
          lsim.v_varloss[i] = std::stod(runner.get_variable(lmp,v));
        }
        for (size_t i = 0; i < lsim.outvar.size(); ++i) {
          const auto& o = lsim.outvar[i];
          lsim.v_outvar[i] = runner.get_variable(lmp,o);
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

    tloss = 0;  // reset total loss
    Analytics a(val_stdb,stpred);

    // By now we should have finished all computations

    if (trg.use_ermse) {
      ermse = a.calc_e_rmse().mean(); // eV/atom
      ermse *= 1000; // meV/atom;
      tloss+= std::abs(ermse-trg.ermse)*trg.w_ermse;
    }
    if (trg.use_frmse) {
      frmse = a.calc_f_rmse().mean();
      tloss+= std::abs(frmse-trg.frmse)*trg.w_frmse;
    }
    if (trg.use_srmse) {
      srmse = a.calc_s_rmse().mean();
      tloss+= std::abs(srmse-trg.srmse)*trg.w_srmse;
    }

    for (const auto& lsim : trg.LS) {
      for (size_t i=0; i<lsim.varloss.size(); ++i) {
        tloss+= lsim.w_varloss[i]*lsim.v_varloss[i];
      }
    }

    // Dump potential if best
    if (tloss<best_score) {
      best_score = tloss;
      std::ofstream out_file;
      out_file.open ("best_pot.tadah");
      pot_tadah.clear_internal_keys();
      out_file << pot_tadah;
      out_file.close();
    }

    // Dump potential potevery steps
    // Controled by POTEVERY
    if (it%trg.potevery==0) {
      std::ofstream out_file;
      out_file.open (trg.potevery_dir+"/pot_"+std::to_string(it)+".tadah");
      out_file << pot_tadah;
      out_file.close();
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> t21 = t2 - t1;

    // print output
    print_loss();
    print_outvar(t21.count());
    print_params(p);
    return tloss;
  }
#ifdef TADAH_BUILD_MPI
  void run(int rank, int ncpu) {
    const static int CONFIG_TAG = 4;
    const static int WAIT_TAG = 3;
    const static int RELEASE_TAG = 2;
    const static int DATA_TAG = 1;
    const static int WORK_TAG = 0;

    MPI_Status status;

    const auto &wpkgs = trg.LS;  // make a copy of LammpsSim vec
    // LS is sorted
    std::vector<bool> is_wpkgs_avail(wpkgs.size(), true);  // zero index is root

    if (wpkgs[0].ncpu>ncpu-1) {
      throw std::runtime_error
      ("Requested LAMMPS simulation size is bigger\n\
than the total number of workers.\n");
    }
    std::vector<int> available_workers;
    std::vector<int> distribution(ncpu);  // This array tracks the jobs assigned to each worker.
    // zero index is root, 

    while (true) {
      bool has_wpkgs = std::any_of(is_wpkgs_avail.begin(),
                                   is_wpkgs_avail.end(), [](bool value) { return value; });

      if (!has_wpkgs) {
        break;
      }

      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;
      int worker = status.MPI_SOURCE;

      if (tag == WORK_TAG) {
        MPI_Recv(NULL, 0, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);

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

            if (!is_wpkgs_avail[j]) continue;

            int job_size = wpkgs[j].ncpu;
            if (job_size<=available_workers.size()) {
              std::vector<int> granks(available_workers.end() - job_size, available_workers.end());
              for (int worker: granks) {
                distribution[worker] = j;
                MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD, &status);
                MPI_Send(granks.data(), granks.size(), MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                available_workers.pop_back();
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

                // send wpkgs[j].outvar
                // First, send the number of strings
                num_strings = wpkgs[j].outvar.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].outvar) {
                  int length = message.size() + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, worker, WORK_TAG, MPI_COMM_WORLD);
                }

                // send wpkgs[j].varloss
                // First, send the number of strings
                num_strings = wpkgs[j].varloss.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].varloss) {
                  int length = message.size() + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, worker, WORK_TAG, MPI_COMM_WORLD);
                }

                // send failscore
                MPI_Send(&wpkgs[j].failscore, 1, MPI_DOUBLE, worker, WORK_TAG, MPI_COMM_WORLD);
              }
              is_wpkgs_avail[j]=false;
              break; // break outer for loop
            }
          }
        }
      }
      else if (tag == DATA_TAG) {
        int j = distribution[worker];
        MPI_Recv(trg.LS[j].v_varloss.data(), trg.LS[j].v_varloss.size(), MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < trg.LS[j].v_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          char* buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          trg.LS[j].v_outvar[i] = std::string(buffer);
          delete[] buffer;
        }
      }
    }
    int count2 = 1;
    while (true) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;
      int worker = status.MPI_SOURCE;

      if (tag == WORK_TAG) {
        MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(NULL, 0, MPI_INT, worker, RELEASE_TAG, MPI_COMM_WORLD);
        count2++;
      }
      else if (tag == DATA_TAG) {
        int j = distribution[worker];
        MPI_Recv(trg.LS[j].v_varloss.data(), trg.LS[j].v_varloss.size(), MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < trg.LS[j].v_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          char* buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          trg.LS[j].v_outvar[i] = std::string(buffer);
          delete[] buffer;
        }
      }

      if (count2==ncpu) break;
    }
  }
#endif
  void init_loss() {
    std::ofstream out_file(loss_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << loss_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITERATION";
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << "ERMSE";
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << "FRMSE";
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << "SRMSE";

    for (const auto& lsim : trg.LS) {
      for (const auto& vloss : lsim.varloss)
      out_file << std::left << std::setw(cwidth) << vloss;
    }

    out_file << std::left << std::setw(cwidth) << "TOTAL" << std::endl;
  }
  void init_outvar() {
    std::ofstream out_file(outvar_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << outvar_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITERATION";
    out_file << std::left << std::setw(cwidth) << "IT_TIME/ms";
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << "ERMSE";
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << "FRMSE";
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << "SRMSE";

    for (const auto& lsim : trg.LS) {
      for (const auto& outvar : lsim.outvar)
      out_file << std::left << std::setw(cwidth) << outvar;
    }
    out_file << std::endl;
  }

  void init_params() {
    std::ofstream out_file(params_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << params_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITER";

    for (const auto& p : trg.optimap) {
      for (size_t i=0; i<p.second.opt.size(); ++i) {
        if (p.second.opt[i])
          out_file << std::left << std::setw(cwidth) << p.first;
      }
    }
    out_file << std::endl;
  }

  void print_loss() {
    std::ofstream out_file(loss_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << loss_file << std::endl;
      return;
    }
    out_file << std::left << std::setw(cwidth) << it;

    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << std::abs(ermse-trg.ermse)*trg.w_ermse;
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << std::abs(frmse-trg.frmse)*trg.w_frmse;
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << std::abs(srmse-trg.srmse)*trg.w_srmse;

    for (const auto& lsim : trg.LS) {
      for (size_t i=0; i<lsim.varloss.size(); ++i) {
        out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << lsim.w_varloss[i]*lsim.v_varloss[i];
      }
    }

    out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << tloss << std::endl;;

  }
  void print_outvar(double it_time) {
    std::ofstream out_file(outvar_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << outvar_file << std::endl;
      return;
    }

    out_file << std::left << std::setw(cwidth) << it;
    out_file << std::left << std::setw(cwidth) << std::setprecision(pprec) << it_time;

    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << ermse;
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << frmse;
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << srmse;

    for (const auto& lsim : trg.LS) {
      for (size_t i=0; i<lsim.outvar.size(); ++i) {
        std::string s = lsim.v_outvar[i].size() > cwidth-1 ? 
          lsim.v_outvar[i].substr(0, cwidth-1) : lsim.v_outvar[i];
        out_file << std::left << std::setw(cwidth) << s;
      }
    }

    out_file << std::endl;;
  }
  void print_params(const T &p) {
    std::ofstream out_file(params_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << params_file << std::endl;
      return;
    }

    out_file << std::left << std::setw(cwidth) << it;

    for (const auto& v_optim : p) {
      out_file << std::left << std::setw(cwidth) << std::scientific << std::setprecision(pprec) << v_optim;
    }
    out_file << std::endl;
  }

};
#endif

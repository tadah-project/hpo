#ifndef HPO_HOST_H
#define HPO_HOST_H

#include <cstddef>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <chrono>
#include <cmath> // ??
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h> // ??
#include <vector>


#if defined(TADAH_ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
inline omp_int_t omp_get_num_threads() { return 1; }
#endif

namespace {
  // pre-computed constant
  static constexpr double LOG_2 = 0.6931471805599453;  

  double stable_log_cosh(double r) {
    double a = std::abs(r);
    // for very large a, e^(–2a) underflows to 0, so log1p(exp(...)) → 0  
    return a + std::log1p(std::exp(-2.0 * a)) - LOG_2;
  }
}

namespace tadah {
namespace hpo {

/* General comments:
 * - lammps must be compiled with c++ exceptions: -DLAMMPS_EXCEPTIONS
 */

/**
 * Usage example:
 *
 *     core::Context context("config.train");
 *     HPOTargets trg(context, "targets");
 *     HPO_Host<dlib::matrix<double,0,1>,LammpsRunner>
 *     hpo(context"config_val");
 *     # the object is ready to be passed to the optimiser.
 *
 */
template <typename T, typename R> class HPO_Host {
private:
  // LOSS FUNCTIONS
  enum LossType {
    LOSS_POWN    = 0,  // |y - ypred|^N
    LOSS_HUBER,        // Huber
    LOSS_LOG_COSH,     // log-cosh
    LOSS_MAPE,         // mean absolute percentage error
    LOSS_RMSLE,        // root-mean-squared log error (without final sqrt)
    LOSS_TUKEY,        // Tukey’s biweight
  };
  size_t lossType      = LOSS_POWN;
  size_t numLosses;
  double delta         = 1.0;       // Huber threshold
  double tukeyC        = 4.685;     // Tukey biweight cutoff
  double eps           = 1e-12;     // small constant for div/zero‐guard
  // END LOSS FUNCTIONS

  size_t cwidth = 16;
  int pprec = 7;
  double totalLossVar, emae, ermse, errmse, ersq, frmse, srmse;
  double emaeLoss, ermseLoss, errmseLoss, ersqLoss, frmseLoss, srmseLoss;
  std::vector<std::vector<double>> v_loss;
  std::string bestLossFile = "best_loss.tadah";
  std::string bestParamsFile = "best_params.tadah";
  std::string bestOutvarFile = "best_outvar.tadah";
  std::string loss_file = "loss.tadah";
  std::string params_file = "params.tadah";
  std::string outvar_file = "outvar.tadah";
  //R runner;
  HPOTargets &trg;
  double bestLoss = std::numeric_limits<double>::max();
  core::Context context;          // make a copy
  mlip::StructureDB val_stdb;   // validation dataset
  mlip::StructureDB train_stdb; // train dataset NON MPI only
  size_t it = 0;          // iteration number
  size_t best_it = 0;          // counts best pots
  tadah::mlip::memory::MLIPWorkspaceManager workspaceManager;

  void update_context(T &v) {
    for (auto &m : trg.optimap) {
      context.remove(m.first);
    }

    size_t idx = 0;
    for (const auto &p : trg.optimap) {
      for (size_t i = 0; i < p.second.opt.size(); ++i) {
        if (p.second.opt[i]) {
          context.add(p.first, v(idx++));
        } else {
          context.add(p.first, p.second.config[i]);
        }
      }
    }
  }

  bool use_lammps;

  void print_elements(const std::set<core::Element> &elements,
                      const std::string &set_name, std::ostream &os) {
    os << set_name << ": ";
    for (const auto &elem : elements) {
      os << elem << " "; // Ensure core::Element has an overloaded << operator
    }
    os << std::endl;
  }
  void validate_elements(const std::set<core::Element> &train_elements,
                         const std::set<core::Element> &val_elements) {
    std::ostringstream error_message;
    bool error_found = false;

    for (const auto &elem : val_elements) {
      if (train_elements.find(elem) == train_elements.end()) {
        error_found = true;
        break;
      }
    }

    if (error_found) {
      error_message
          << "Atoms in the validation set are not found in the training set.\n";
      print_elements(train_elements, "Training Set", error_message);
      print_elements(val_elements, "Validation Set", error_message);
      throw std::runtime_error(error_message.str());
    }
  }
#ifdef TADAH_BUILD_MPI
    mlip::TrainerHost host(context, workspaceManager, workspaceManager2);
    tadah::mlip::memory::MLIPWorkspaceManager workspaceManager2;
#else
    mlip::Trainer host;
    mlip::NNFinder nnf;
#endif

public:
  ~HPO_Host() {
  }
  HPO_Host(core::Context c, HPOTargets &trg, std::string &val_file)
    : trg(trg),  context(c), numLosses(0), 
#ifdef TADAH_BUILD_MPI
    host(c, workspaceManager, workspaceManager2)
#else
    host(c, workspaceManager), nnf(c)
#endif
  {
    core::Context val_context(core::CoreUserConfigParser().parse(val_file));
    val_stdb.add(val_context);

    std::set<core::Element> train_elements =
      mlip::StructureDB::find_unique_elements(context);
    std::set<core::Element> val_elements =
      mlip::StructureDB::find_unique_elements(val_context);

    validate_elements(train_elements, val_elements);

    use_lammps = trg.LS.size() > 0;
    v_loss.resize(trg.LS.size());
#ifdef TADAH_BUILD_MPI
    MPI_Bcast(&use_lammps, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
#else
    train_stdb.add(context);
#endif

    init_loss_func();

    init_loss(loss_file);
    init_outvar(outvar_file);
    init_params(params_file);

    init_loss(bestLossFile);
    init_outvar(bestOutvarFile);
    init_params(bestParamsFile);

    // mlip::Trainer host(context, workspaceManager);
  }

  // MAIN HPO LOOP
  double operator()(T p) {
    totalLossVar = 0; // reset total loss
    numLosses = 0;
    it++;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    update_context(p);

    // core::Context context_temp = context;

    // Train new potential with updated settings
#ifdef TADAH_BUILD_MPI
    int rank;
    int ncpu;

    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // New context file must be communicated to all processes
    std::vector<char> cs = context.serialize();
    int cs_size = cs.size();
    MPI_Bcast(&cs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cs.data(), cs_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // HOST is waiting for workers requests
    // mlip::TrainerHost host(context, workspaceManager, workspaceManager2);
    host.run();
    host.solve();
#else
    host.init(context);
    //train_stdb.clear_nn();
    //host.train(train_stdb);
#endif
    // End of training

    // This potential will be used by LAMMPS
    core::Context pot_tadah = host.model->get_param_file();

     std::ofstream out_file;
     out_file.open("pot.tadah");
     out_file << pot_tadah;
     out_file.close();

     val_stdb.clear_nn();
     nnf.init(pot_tadah);
     nnf.calc(val_stdb);
     mlip::StructureDB stpred = host.model->predict(pot_tadah, val_stdb, host.dc);

     mlip::Analytics a(val_stdb, stpred);


    if (trg.use_emae) {
      emae = a.calc_e_mae().mean(); // eV/atom
      emae *= 1000;                 // meV/atom;
    }
    if (trg.use_ermse) {
      ermse = a.calc_e_rmse().mean(); // eV/atom
      ermse *= 1000;                  // meV/atom;
    }
    if (trg.use_errmse) {
      errmse = a.calc_e_rrmse().mean(); // eV/atom
      errmse *= 1000;                   // meV/atom;
    }
    if (trg.use_ersq) {
      ersq = a.calc_e_r_sq().mean();
    }
    if (trg.use_frmse) {
      frmse = a.calc_f_rmse().mean();
    }
    if (trg.use_srmse) {
      srmse = a.calc_s_rmse().mean();
    }

    computeLoss();

    // if (totalLoss < trg.FAIL_SCORE) {
      // only run LAMMPS if loss is not too high
      runLammps();
      computeLammpsLoss();
    // } else {
    //   // If loss is too high, set LAMMPS loss to fail score
    //   std::cout << "DEBUG: SKIPPING LAMMPS, setting loss to fail score" << std::endl;
    //   for (size_t i = 0; i < trg.LS.size(); ++i) {
    //     v_loss[i].resize(trg.LS[i].n_varloss.size());
    //     for (size_t j = 0; j < trg.LS[i].n_varloss.size(); ++j) {
    //       v_loss[i][j] = trg.LS[i].failscore;
    //       totalLoss += v_loss[i][j];
    //     }
    //   }
    // }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> t21 = t2 - t1;

    // Dump potential if best
    if (totalLoss() < bestLoss) {
      best_it++;
      bestLoss = totalLoss();
      std::ofstream out_file;
      out_file.open("best_pot.tadah");
      out_file << pot_tadah;
      out_file.close();

      // Dump potential every bestPotEvery steps
      if (it % trg.bestPotEvery == 0) {
        std::ofstream out_file;
        out_file.open(trg.bestPotEveryDir + "/best_pot_" + std::to_string(it) + ".tadah");
        out_file << pot_tadah;
        out_file.close();
      }
      // Save best loss, params and outvar
      print_loss(bestLossFile,best_it);
      print_outvar(bestOutvarFile,best_it,t21.count());
      print_params(bestParamsFile,best_it,p);
    }

    // Dump potential potevery steps
    // Controled by POTEVERY
    if (it % trg.potevery == 0) {
      std::ofstream out_file;
      out_file.open(trg.potevery_dir + "/pot_" + std::to_string(it) + ".tadah");
      out_file << pot_tadah;
      out_file.close();
    }


    // print output
    print_loss(loss_file,it);
    print_outvar(outvar_file,it,t21.count());
    print_params(params_file,it,p);
    return totalLoss();
  }

  void runLammps() {

#ifdef TADAH_BUILD_MPI
    if (use_lammps)
      runLammpsMPI(rank, ncpu);
#else
    // Run all simulations in trg.LS
    auto &LS = trg.LS;
#ifdef _OPENMP
#pragma omp parallel for shared(LS)
#endif
    for (size_t simNum = 0; simNum < LS.size(); ++simNum) {
      R runner;

      auto &lsim = LS[simNum];
      auto lmp = runner.create_instance(/*dummy_comm*/0, lsim.sinvar);
      // auto lmp = runner.instances_[simNum];

      // If LAMMPS creation failed, mark fail results and move on
      // create will tidy up and return nullptr
      if (!lmp) {
        for (size_t i = 0; i < lsim.n_varloss.size(); ++i)
          lsim.v_varloss[i] = lsim.failscore;
        for (size_t i = 0; i < lsim.outvar.size(); ++i)
          lsim.v_outvar[i] = "NaN";
        continue;
      }

      // Attempt to run script on this instance
      bool success = runner.run_script(lmp, lsim.script);

      if (!success) {
        // A crash or severe error occurred. Mark fail results
        for (size_t i = 0; i < lsim.n_varloss.size(); ++i)
          lsim.v_varloss[i] = lsim.failscore;
        for (size_t i = 0; i < lsim.outvar.size(); ++i)
          lsim.v_outvar[i] = "NaN";
        continue;
      }

      // If script ran successfully, retrieve LAMMPS variables
      for (size_t i = 0; i < lsim.n_varloss.size(); ++i) {
        try {
          const std::string &v = lsim.n_varloss[i];
          lsim.v_varloss[i] = std::stod(runner.get_variable(lmp, v));
        } catch (const std::exception &e) {
          std::cerr << "Error obtaining LAMMPS loss variable: " << e.what() << std::endl;
          std::cerr << "Variable name: " << lsim.n_varloss[i] << std::endl;
          throw;
        }
      }

      for (size_t i = 0; i < lsim.outvar.size(); ++i) {
        try {
          const std::string &o = lsim.outvar[i];
          lsim.v_outvar[i] = runner.get_variable(lmp, o);
        } catch (const std::exception &e) {
          std::cerr << "Error obtaining LAMMPS output variable: " << e.what() << std::endl;
          std::cerr << "Variable name: " << lsim.outvar[i] << std::endl;
          throw;
        }
      }
    }
#endif
  }
#ifdef TADAH_BUILD_MPI
  void runLammpsMPI(int rank, int ncpu) {
    const static int CONFIG_TAG = 4;
    const static int WAIT_TAG = 3;
    const static int RELEASE_TAG = 2;
    const static int DATA_TAG = 1;
    const static int WORK_TAG = 0;

    MPI_Status status;

    const auto &wpkgs = trg.LS; // make a copy of LammpsSim vec
    // LS is sorted
    std::vector<bool> is_wpkgs_avail(wpkgs.size(), true); // zero index is root

    if (wpkgs[0].ncpu > ncpu - 1) {
      throw std::runtime_error("Requested LAMMPS simulation size is bigger\n\
than the total number of workers.\n");
    }
    std::vector<int> available_workers;
    std::vector<int> distribution(
        ncpu); // This array tracks the jobs assigned to each worker.
    // zero index is root,

    while (true) {
      bool has_wpkgs = std::any_of(is_wpkgs_avail.begin(), is_wpkgs_avail.end(),
                                   [](bool value) { return value; });

      if (!has_wpkgs) {
        break;
      }

      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int tag = status.MPI_TAG;
      int worker = status.MPI_SOURCE;

      if (tag == WORK_TAG) {
        MPI_Recv(NULL, 0, MPI_INT, worker, tag, MPI_COMM_WORLD, &status);

        auto it = std::find(available_workers.begin(), available_workers.end(),
                            worker);
        if (it == available_workers.end()) { // new worker is available
          available_workers.push_back(worker);
          MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);
        } else {
          // release an eager worker as we will select a group of workers here
          MPI_Send(NULL, 0, MPI_INT, worker, WAIT_TAG, MPI_COMM_WORLD);

          // There are no more new workers available at the moment.
          // Let's find the biggest job we can fit to them
          for (int j = 0; j < wpkgs.size(); ++j) {

            if (!is_wpkgs_avail[j])
              continue;

            int job_size = wpkgs[j].ncpu;
            if (job_size <= available_workers.size()) {
              std::vector<int> granks(available_workers.end() - job_size,
                                      available_workers.end());
              for (int worker : granks) {
                distribution[worker] = j;
                MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD,
                         &status);
                MPI_Send(granks.data(), granks.size(), MPI_INT, worker,
                         WORK_TAG, MPI_COMM_WORLD);
                available_workers.pop_back();
                // send wpkgs[j].script
                MPI_Send(wpkgs[j].script.c_str(), wpkgs[j].script.size() + 1,
                         MPI_CHAR, worker, WORK_TAG, MPI_COMM_WORLD);

                // send wpkgs[j].invar
                // First, send the number of strings
                int num_strings = wpkgs[j].invar.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const char *message : wpkgs[j].invar) {
                  int length =
                      strlen(message) + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message, length, MPI_CHAR, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send wpkgs[j].outvar
                // First, send the number of strings
                num_strings = wpkgs[j].outvar.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].outvar) {
                  int length = message.size() + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send wpkgs[j].n_varloss
                // First, send the number of strings
                num_strings = wpkgs[j].n_varloss.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].n_varloss) {
                  int length = message.size() + 1; // +1 for the null terminator
                  MPI_Send(&length, 1, MPI_INT, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                  MPI_Send(message.c_str(), length, MPI_CHAR, worker, WORK_TAG,
                           MPI_COMM_WORLD);
                }

                // send failscore
                MPI_Send(&wpkgs[j].failscore, 1, MPI_DOUBLE, worker, WORK_TAG,
                         MPI_COMM_WORLD);
              }
              is_wpkgs_avail[j] = false;
              break; // break outer for loop
            }
          }
        }
      } else if (tag == DATA_TAG) {
        int j = distribution[worker];
        MPI_Recv(trg.LS[j].v_varloss.data(), trg.LS[j].v_varloss.size(),
                 MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < trg.LS[j].v_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          char *buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
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
        MPI_Recv(NULL, 0, MPI_INT, worker, WORK_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(NULL, 0, MPI_INT, worker, RELEASE_TAG, MPI_COMM_WORLD);
        count2++;
      } else if (tag == DATA_TAG) {
        int j = distribution[worker];
        MPI_Recv(trg.LS[j].v_varloss.data(), trg.LS[j].v_varloss.size(),
                 MPI_DOUBLE, worker, DATA_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < trg.LS[j].v_outvar.size(); i++) {
          int length;
          MPI_Recv(&length, 1, MPI_INT, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          char *buffer = new char[length];
          MPI_Recv(buffer, length, MPI_CHAR, worker, DATA_TAG, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          trg.LS[j].v_outvar[i] = std::string(buffer);
          delete[] buffer;
        }
      }

      if (count2 == ncpu)
        break;
    }
  }
#endif
void init_loss_func() {
  if (trg.lossType == "L1") {
    lossType = LOSS_POWN;
    trg.N = 1; // L1 loss
  } else if (trg.lossType == "L2") {
    lossType = LOSS_POWN;
    trg.N = 2; // L2 loss
  } else if (trg.lossType == "HUBER") {
    lossType = LOSS_HUBER;
    delta = trg.deltaHuber;
  } else if (trg.lossType == "LOG_COSH") {
    lossType = LOSS_LOG_COSH;
  } else if (trg.lossType == "MAPE") {
    lossType = LOSS_MAPE;
  } else if (trg.lossType == "RMSLE") {
    lossType = LOSS_RMSLE;
  } else if (trg.lossType == "TUKEY") {
    lossType = LOSS_TUKEY;
    tukeyC = trg.tukeyC;
  } else {
    throw std::runtime_error("Unknown loss function: " + trg.lossType);
  }
}

  void init_loss(const std::string &_loss_file) {
    std::ofstream out_file(
        _loss_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << _loss_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITERATION";
    if (trg.use_emae)
      out_file << std::left << std::setw(cwidth) << "loss(EMAE)";
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << "loss(ERMSE)";
    if (trg.use_errmse)
      out_file << std::left << std::setw(cwidth) << "loss(ErRMSE)";
    if (trg.use_ersq)
      out_file << std::left << std::setw(cwidth) << "loss(ERSQ)";
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << "loss(FRMSE)";
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << "loss(SRMSE)";

    for (const auto &lsim : trg.LS) {
      for (const auto &vloss : lsim.n_varloss)
        out_file << std::left << std::setw(cwidth) << vloss;
    }

    out_file << std::left << std::setw(cwidth) << "TOTAL_LOSS" << std::endl;
  }
  void init_outvar(const std::string &_outvar_file) {
    std::ofstream out_file(
        _outvar_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << _outvar_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITERATION";
    out_file << std::left << std::setw(cwidth) << "IT_TIME/ms";
    if (trg.use_emae)
      out_file << std::left << std::setw(cwidth) << "EMAE";
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << "ERMSE";
    if (trg.use_errmse)
      out_file << std::left << std::setw(cwidth) << "ErRMSE";
    if (trg.use_ersq)
      out_file << std::left << std::setw(cwidth) << "ERSQ";
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << "FRMSE";
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << "SRMSE";

    for (const auto &lsim : trg.LS) {
      for (const auto &outvar : lsim.outvar)
        out_file << std::left << std::setw(cwidth) << outvar;
    }
    out_file << std::endl;
  }

  void init_params(const std::string &_params_file) {
    std::ofstream out_file(
        _params_file); // Open with default mode, which truncates

    if (!out_file) {
      std::cerr << "Error opening file: " << _params_file << std::endl;
    }

    // add headers
    out_file << std::left << std::setw(cwidth) << "ITER";

    for (const auto &p : trg.optimap) {
      for (size_t i = 0; i < p.second.opt.size(); ++i) {
        if (p.second.opt[i])
          out_file << std::left << std::setw(cwidth) << p.first;
      }
    }
    out_file << std::endl;
  }


double totalLoss() const {

  double meanL = totalLossVar / numLosses;

  // final sqrt for RMSE / RMSLE
  if (lossType == LOSS_POWN && trg.N == 2) {
    return std::sqrt(meanL);     // RMSE
  }
  if (lossType == LOSS_RMSLE) {
    return std::sqrt(meanL);     // RMSLE
  }
  return meanL;
}

double sampleLoss(double y, double ypred, double w) {
  numLosses++;

  double r    = y - ypred;
  double absr = std::abs(r);

  switch(lossType) {

    case LOSS_POWN:
      return std::pow(absr, trg.N) * w;

    case LOSS_HUBER:
      if (absr <= delta) {
        return 0.5 * r*r * w;
      } else {
        return delta * (absr - 0.5*delta) * w;
      }

    case LOSS_LOG_COSH:
      {
      // log(cosh(r)) ≈ r^2/2 for small r, ≈ |r| - log(2) for large r
      double res = stable_log_cosh(r) * w;
      return res;
      }

    case LOSS_MAPE:
      return absr / (std::abs(y) + eps) * w;

    case LOSS_RMSLE:
      {
        double ly  = std::log(y    + eps);
        double lyp = std::log(ypred+ eps);
        double dr  = ly - lyp;
        return dr*dr * w;
      }

    case LOSS_TUKEY: 
      {
        if (absr <= tukeyC) {
          double t = r / tukeyC;
          return (tukeyC*tukeyC/6.0)*(1.0 - std::pow(1.0 - t*t, 3)) * w;
        } else {
          return (tukeyC*tukeyC/6.0) * w;
        }
      }

  }
}

  void computeLoss() {
    if (trg.use_emae) {
      emaeLoss = sampleLoss(emae, trg.emae, trg.w_emae);
      totalLossVar += emaeLoss;
    }
    if (trg.use_ermse) {
      ermseLoss = sampleLoss(ermse, trg.ermse, trg.w_ermse);
      totalLossVar += ermseLoss;
    }
    if (trg.use_errmse) {
      errmseLoss = sampleLoss(errmse, trg.errmse, trg.w_errmse);
      totalLossVar += errmseLoss;
    }
    if (trg.use_ersq) {
      ersqLoss = sampleLoss(ersq, trg.ersq, trg.w_ersq);
      totalLossVar += ersqLoss;
    }
    if (trg.use_frmse) {
      frmseLoss = sampleLoss(frmse, trg.frmse, trg.w_frmse);
      totalLossVar += frmseLoss;
    }
    if (trg.use_srmse) {
      srmseLoss = sampleLoss(srmse, trg.srmse, trg.w_srmse);
      totalLossVar += srmseLoss;
    }
  }
  void computeLammpsLoss() {
    // TODO Generalise to use sampleLoss
    for (size_t nsim = 0; nsim < trg.LS.size(); ++nsim) {
      const auto &lsim = trg.LS[nsim];
      v_loss[nsim].resize(lsim.n_varloss.size());
      for (size_t i = 0; i < lsim.n_varloss.size(); ++i) {
        double vl = std::pow(lsim.v_varloss[i], trg.N)*lsim.w_varloss[i];
        v_loss[nsim][i] = vl;
        totalLossVar += vl;
      }
    }
  }
  void print_loss(const std::string &_loss_file, size_t _it) {
    std::ofstream out_file(_loss_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << _loss_file << std::endl;
      return;
    }
    out_file << std::left << std::setw(cwidth) << _it;

    if (trg.use_emae)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << emaeLoss;
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << ermseLoss;
    if (trg.use_errmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << errmseLoss;
    if (trg.use_ersq)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << ersqLoss;
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << frmseLoss;
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << srmseLoss;

    for (size_t nsim = 0; nsim < trg.LS.size(); ++nsim) {
      const auto &lsim = trg.LS[nsim];
      for (size_t i = 0; i < lsim.n_varloss.size(); ++i) {
        out_file << std::left << std::setw(cwidth) << std::scientific
          << std::setprecision(pprec)
          << v_loss[nsim][i];
      }
    }

    out_file << std::left << std::setw(cwidth) << std::scientific
             << std::setprecision(pprec) << totalLoss() << std::endl;
    ;
  }
  void print_outvar(const std::string &_outvar_file, size_t _it, double it_time) {
    std::ofstream out_file(_outvar_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << _outvar_file << std::endl;
      return;
    }

    out_file << std::left << std::setw(cwidth) << _it;
    out_file << std::left << std::setw(cwidth) << std::setprecision(pprec)
             << it_time;

    if (trg.use_emae)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << emae;
    if (trg.use_ermse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << ermse;
    if (trg.use_errmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << errmse;
    if (trg.use_ersq)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << ersq;
    if (trg.use_frmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << frmse;
    if (trg.use_srmse)
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << srmse;

    for (const auto &lsim : trg.LS) {
      for (size_t i = 0; i < lsim.outvar.size(); ++i) {
        std::string s = lsim.v_outvar[i].size() > cwidth - 1
                            ? lsim.v_outvar[i].substr(0, cwidth - 1)
                            : lsim.v_outvar[i];
        out_file << std::left << std::setw(cwidth) << s;
      }
    }

    out_file << std::endl;
    ;
  }
  void print_params(const std::string &_params_file, size_t _it,const T &p) {
    std::ofstream out_file(_params_file, std::ios::app);

    if (!out_file) {
      std::cerr << "Error opening file: " << _params_file << std::endl;
      return;
    }

    out_file << std::left << std::setw(cwidth) << _it;

    for (const auto &v_optim : p) {
      out_file << std::left << std::setw(cwidth) << std::scientific
               << std::setprecision(pprec) << v_optim;
    }
    out_file << std::endl;
  }
};
} // namespace hpo
} // namespace tadah
#endif

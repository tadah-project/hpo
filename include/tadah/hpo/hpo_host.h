#ifndef TADAH_HPO_HOST_H_
#define TADAH_HPO_HOST_H_

/*
 *  Thread-safe, “stateless” HPO_Host
 *  ---------------------------------
 *  –  All data produced during one evaluation live in EvalResult, not in the
 *     functor object; operator() is therefore const and thread-safe.
 *  –  Results are forwarded to a single writer thread through a blocking
 *     moodycamel queue which guarantees ordered, race-free file output.
 *  –  MPI code is protected by one global mutex so that threaded builds can
 *     still work.  Remove the mutex if you choose to forbid that mix.
 */

#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#if defined(TADAH_ENABLE_OPENMP)
#   include <omp.h>
#endif
#if defined(TADAH_BUILD_MPI)
#   include <mpi.h>
#endif

#include <moodycamel/blockingconcurrentqueue.h>     // moodycamel

namespace tadah::hpo {

/* =========================================================================
 *  EvalResult – everything produced by *one* evaluation
 * ========================================================================= */
template<class ParamVec>
struct EvalResult
{
    std::size_t                            iter      = 0;
    std::size_t                            best_iter = 0;
    double                                 loss      = 0.0;

    /* validation metrics */
    double emae=0, ermse=0, errmse=0, ersq=0, frmse=0, srmse=0;
    double emaeLoss=0, ermseLoss=0, errmseLoss=0, ersqLoss=0, frmseLoss=0, srmseLoss=0;
    std::vector<std::vector<double>>       v_loss;    // [sim][loss_i]
    std::vector<std::vector<std::string>>  v_outvar;  // [sim][outvar_i]

    std::chrono::milliseconds              wall_ms{};
    ParamVec                               params;

    bool                                   is_best    = false;
    // bool                                   run_lammps = false;   // option for the future?
    core::Context                          pot_ctx;
};

/* =========================================================================
 *  HPO_Host  –  stateless evaluation functor
 * ========================================================================= */
template<class ParamVec, class LammpsRunner>
class HPO_Host
{
    using Result = EvalResult<ParamVec>;
    using Queue  = moodycamel::BlockingConcurrentQueue<Result>;

public:
    HPO_Host(core::Context master,
        HPOTargets&   trg,
        const std::string& val_cfg);

    double operator()(const ParamVec& p) const;      // thread-safe


    /* -------- adapter for NLopt C++ wrapper ----------------------------- */
    static double nlopt_adapter_cpp(const std::vector<double>& x,
        std::vector<double>&       grad,
        void*                      data)
    {
      auto* self = static_cast<const HPO_Host*>(data);
      if (!grad.empty()) {};               // no analytic gradient
      return (*self)(make_param_vec(static_cast<unsigned>(x.size()),
            x.data()));
    }

private:

    // helper that converts the raw array coming from NLopt -----------------
    static ParamVec make_param_vec(unsigned n, const double* x)
    {
      ParamVec v;                   // v is a dlib::matrix<double,0,1>
      v.set_size(n);
      for (unsigned i = 0; i < n; ++i)
        v(i) = x[i];
      return v;
    }


    /* ---------- immutable after construction --------------------------- */
    core::Context                master_ctx_;
    HPOTargets&                  trg_;
    mlip::StructureDB            val_stdb_;
#ifndef TADAH_BUILD_MPI
    mlip::StructureDB            train_stdb_;
#endif
    bool                         use_lammps_;
    mlip::NNFinder               nnf_;

    /* ---------- helpers (pure) ----------------------------------------- */
    void update_context(core::Context&, const ParamVec&) const;
    void fill_validation(Result&, const mlip::Analytics&)  const;
    void run_lammps     (Result&) const;
    void fit_loss(Result&) const;
    void sims_loss(Result&) const;

    /* ---------- global mutable state ----------------------------------- */
    static std::atomic_size_t   g_iter_;
    static std::atomic_size_t   g_best_iter_;
    static std::atomic<double>  g_best_;

    /* opportunistic single-writer --------------------------------------- */
    static Queue                                   g_q_;          // MPMC queue
    static std::atomic_flag                        writer_token_; // cleared = free
    static std::mutex                              io_mtx_;       // protects FILEs
    static std::map<std::size_t,Result>            io_buf_;       // re-order
    static std::atomic<std::size_t>                next_flush_;   // next iter
    static void submit_result(Result&&, const HPOTargets& trg_,
                              std::size_t cw, int prec);

#if defined(TADAH_BUILD_MPI)
    static std::mutex           mpi_mutex_;
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws2_;
    mlip::TrainerHost host(context, tls_ws_, tls_ws2_);
#endif

    /* per thread MLIP workspace ----------------------------------------- */
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws_;


    /* formatting */
    const std::size_t            cw_   = 16;
    const int                    prec_ = 7;
    size_t                       max_title_len_ = cw_-1; // max length of title in loss/outvar/params

    /* line printers ------------------------------------ */
    void init_loss   (const std::string& fn) const;
    void init_outvar (const std::string& fn) const;
    void init_params (const std::string& fn) const;

    static void print_loss_line  (std::ofstream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);
    static void print_outvar_line(std::ofstream&, size_t it, const Result&,
                                  const HPOTargets&, std::size_t cw, int prec);
    static void print_params_line(std::ofstream&, size_t it, const Result&,
                                  std::size_t cw, int prec);

};

/* =======================================================================*/
}   // namespace tadah::hpo

#include "hpo_host_impl.hpp"
#endif

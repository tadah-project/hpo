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
    double                                 loss      = 0.0;

    /* validation metrics */
    double emae=0, ermse=0, errmse=0, ersq=0, frmse=0, srmse=0;
    std::vector<std::vector<double>>       v_loss;
    std::vector<std::string>               v_outvar;

    std::chrono::milliseconds              wall_ms{};
    ParamVec                               params;

    bool                                   is_best   = false;
    core::Context                          best_pot;          // valid if is_best
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

private:
    /* ---------- immutable after construction --------------------------- */
    core::Context                master_ctx_;
    HPOTargets&                  trg_;
    mlip::StructureDB            val_stdb_;
#ifndef TADAH_BUILD_MPI
    mlip::StructureDB            train_stdb_;
#endif
    bool                         use_lammps_;
    mlip::NNFinder               nnf_;

    /* formatting */
    const std::size_t            cw_   = 16;
    const int                    prec_ = 7;

    /* ---------- helpers (pure) ----------------------------------------- */
    void update_context(core::Context&, const ParamVec&) const;
    void fill_validation(Result&, const mlip::Analytics&)  const;
    void run_lammps     (Result&) const;
    void accumulate_loss(Result&) const;

    /* ---------- global mutable state ----------------------------------- */
    static std::atomic_size_t   g_iter_;
    static std::atomic<double>  g_best_;

    /* queue + writer thread (one per process, lazily started) ----------- */
    static Queue                g_q_;
    static std::once_flag       writer_once_;
    static std::thread          writer_th_;
    static void                 writer_loop(HPOTargets*, std::size_t, int);

#if defined(TADAH_BUILD_MPI)
    static std::mutex           mpi_mutex_;
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws2_;
    mlip::TrainerHost host(context, tls_ws_, tls_ws2_);
#endif

    /* per thread MLIP workspace ----------------------------------------- */
    static thread_local tadah::mlip::memory::MLIPWorkspaceManager tls_ws_;
};

/* =======================================================================*/
}   // namespace tadah::hpo

#include "hpo_host_impl.hpp"
#endif

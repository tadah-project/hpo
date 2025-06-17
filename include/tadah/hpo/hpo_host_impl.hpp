#pragma once
#include <tadah/core/core_user_config_parser.h>
#include <tadah/hpo/loss_function.h>
#include <fstream>
#include <sstream>

namespace {
/* ---------- utility ---------------------------------------------------- */
inline constexpr double LOG_2 = 0.6931471805599453;
inline double stable_log_cosh(double r)
{
    double a = std::abs(r);
    return a + std::log1p(std::exp(-2.0 * a)) - LOG_2;
}
  
} // empty namespace

namespace tadah::hpo {


/* -----------------  header writers  ---------------------------------- */
template<class P,class R>
void HPO_Host<P,R>::init_loss(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(cw_) << "#ITERATION" << std::right;
  if (trg_.use_emae)
    f << std::setw(cw_) << "L(EMAE)";
  if (trg_.use_ermse)
    f << std::setw(cw_) << "L(ERMSE)";
  if (trg_.use_errmse)
    f << std::setw(cw_) << "L(ErRMSE)";
  if (trg_.use_ersq)
    f << std::setw(cw_) << "L(ERSQ)";
  if (trg_.use_frmse)
    f << std::setw(cw_) << "L(FRMSE)";
  if (trg_.use_srmse)
    f << std::setw(cw_) << "L(SRMSE)";


  for (const auto &lsim : trg_.LS) {
    for (auto varname : lsim.n_varloss) {
      if (varname.size()-2 > max_title_len_) {
        varname = varname.substr(0, max_title_len_ - 4) + "*";
      }
      f << std::setw(cw_) << "L("+varname+")";
    }
  }

  f << std::setw(cw_) << "TOTAL_LOSS/N" << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_outvar(const std::string& fn) const
{
  std::ofstream f(fn);

  f << std::left << std::setw(cw_) << "#ITERATION";
  f << std::setw(cw_) << "IT-TIME/ms" << std::right;
  if (trg_.use_emae)
    f << std::setw(cw_) << "EMAE";
  if (trg_.use_ermse)
    f << std::setw(cw_) << "ERMSE";
  if (trg_.use_errmse)
    f << std::setw(cw_) << "ErRMSE";
  if (trg_.use_ersq)
    f << std::setw(cw_) << "ERSQ";
  if (trg_.use_frmse)
    f << std::setw(cw_) << "FRMSE";
  if (trg_.use_srmse)
    f << std::setw(cw_) << "SRMSE";

  for (const auto &lsim : trg_.LS) {
    for (auto n_outvar : lsim.n_outvar) {
      if (n_outvar.size() > max_title_len_) {
        n_outvar = n_outvar.substr(0, max_title_len_ - 1) + "*";
      }
      f << std::setw(cw_) << n_outvar;
    }
  }
  f << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::init_params(const std::string& fn) const
{

  std::ofstream f(fn);
  f << std::left << std::setw(cw_) << "#ITERATION" << std::right;

  for (const auto &p : trg_.optimap) {
    for (size_t i = 0; i < p.second.opt.size(); ++i) {
      if (p.second.opt[i])
        f  << std::setw(cw_) << p.first;
    }
  }
  f << std::endl;
}

/* ----------  line printers (append) ---------------------------------- */
/* ======= line writers ================================================ */
template<class P,class R>
void HPO_Host<P,R>::print_loss_line(std::ofstream& f, size_t it,const Result& e,
                                    const HPOTargets& trg,
                                    std::size_t cw,int prec)
{
    f << std::left << std::setw(cw) << it;
    f << std::right << std::scientific << std::setprecision(prec);
    if(trg.use_emae)   f <<std::setw(cw) <<e.emaeLoss;
    if(trg.use_ermse)  f <<std::setw(cw) <<e.ermseLoss;
    if(trg.use_errmse) f <<std::setw(cw) <<e.errmseLoss;
    if(trg.use_ersq)   f <<std::setw(cw) <<e.ersqLoss;
    if(trg.use_frmse)  f <<std::setw(cw) <<e.frmseLoss;
    if(trg.use_srmse)  f <<std::setw(cw) <<e.srmseLoss;
      
    for (size_t i = 0; i < e.v_outvar.size(); ++i) {
      auto &indices = trg.LS[i].i_varloss;
      for (const auto &j : indices) {
        f << std::setw(cw) << e.v_loss[i][j];
      }
    }

    f << std::setw(cw) << e.loss << std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::print_outvar_line(std::ofstream& f,size_t it,const Result& e,
                                      const HPOTargets& trg,
                                      std::size_t cw,int prec)
{
    f << std::left << std::setw(cw) << it;
    f << std::left << std::setw(cw) << std::setprecision(prec)
      << e.wall_ms.count();

    if(trg.use_emae)   f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.emae;
    if(trg.use_ermse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.ermse;
    if(trg.use_errmse) f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.errmse;
    if(trg.use_ersq)   f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.ersq;
    if(trg.use_frmse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.frmse;
    if(trg.use_srmse)  f<<std::right<<std::setw(cw)<< std::scientific<< std::setprecision(prec) <<e.srmse;

    for (size_t i = 0; i < e.v_outvar.size(); ++i) {
      auto &indices = trg.LS[i].i_varloss;
      for (const auto &j : indices) {
        f<<std::right << std::setw(cw) << std::scientific
          << std::setprecision(prec) << std::stod(e.v_outvar[i][j]);
      }
    }
    // for(const auto& s: e.v_outvar){
    //   std::string t=s.size()>cw-1? s.substr(0,cw-1):s;
    //   f<<std::setw(cw)<<t;
    // }
    f<<std::endl;
}

template<class P,class R>
void HPO_Host<P,R>::print_params_line(std::ofstream& f,size_t it,const Result& e,
                                      std::size_t cw,int prec)
{
  f << std::left << std::setw(cw) << it;

  for(long i=0;i<e.params.size();++i) 
  {
    f << std::right << std::setw(cw) << std::scientific
      << std::setprecision(prec) << e.params(i);
  }
  f << std::endl;

}


/* ----------  static data definitions ---------------------------------- */
template<class P, class R>
std::atomic_size_t  HPO_Host<P,R>::g_iter_{0};
template<class P, class R>
std::atomic_size_t  HPO_Host<P,R>::g_best_iter_{0};
template<class P, class R>
std::atomic<double> HPO_Host<P,R>::g_best_{std::numeric_limits<double>::max()};
#if defined(TADAH_BUILD_MPI)
template<class P, class R>
std::mutex          HPO_Host<P,R>::mpi_mutex_;
#endif

template<class P,class R> typename HPO_Host<P,R>::Queue  HPO_Host<P,R>::g_q_;
template<class P,class R> std::atomic_flag               HPO_Host<P,R>::writer_token_;
template<class P,class R> std::mutex                     HPO_Host<P,R>::io_mtx_;
template<class P,class R> std::map<std::size_t,
  typename HPO_Host<P,R>::Result> HPO_Host<P,R>::io_buf_;
template<class P,class R> std::atomic<std::size_t>       HPO_Host<P,R>::next_flush_{1};

template<class P, class R>
thread_local tadah::mlip::memory::MLIPWorkspaceManager
HPO_Host<P,R>::tls_ws_{};
#if defined(TADAH_BUILD_MPI)
template<class P, class R>
thread_local tadah::mlip::memory::MLIPWorkspaceManager
HPO_Host<P,R>::tls_ws_2{};
#endif

/* ----------  constructor ---------------------------------------------- */
template<class P, class R>
HPO_Host<P,R>::HPO_Host(core::Context master_ctx,
                        HPOTargets&   trg,
                        const std::string& val_cfg)
: master_ctx_(std::move(master_ctx))
, trg_(trg), spec_(trg.spec)
, use_lammps_( !trg.LS.empty() )
#ifdef TADAH_BUILD_MPI
, host(c, tls_ws_, tls_ws2_)
#endif
{

    /* neighbours -------------------------------------------------------- */
    double rcutmax = master_ctx_.get<double>("RCUTMAX");
    // Loop over all optimap entries to find either RCUT2B or RCUTMB
    // Overwrite rcutmax if higher value is found in optimap.highbound
    for (const auto &opti: trg_.optimap) {
      if (opti.first == "RCUT2B" || opti.first == "RCUTMB") {
        double rcut = opti.second.highbound[0];
        if (rcut > rcutmax) {
          rcutmax = rcut;
        }
      }
    }

    /* validation set ---------------------------------------------------- */
    core::Context val_ctx(core::CoreUserConfigParser().parse(val_cfg));
    val_stdb_.add(val_ctx);
    nnf_.calc(val_stdb_, rcutmax);
#ifndef TADAH_BUILD_MPI
    train_stdb_.add(master_ctx_);
    nnf_.calc(train_stdb_, rcutmax);
#endif
    /* elements check ---------------------------------------------------- */
    auto tr_el = mlip::StructureDB::find_unique_elements(master_ctx_);
    for(auto e: mlip::StructureDB::find_unique_elements(val_ctx))
        if(!tr_el.count(e))
            throw std::runtime_error("Validation element missing in training");


    init_loss   (trg_.lossFile);
    init_outvar (trg_.outvarFile);
    init_params (trg_.paramsFile);
    init_loss   (trg_.bestLossFile);
    init_outvar (trg_.bestOutvarFile);
    init_params (trg_.bestParamsFile);

}

/* ----------  helper: update context ----------------------------------- */
template<class P, class R>
void HPO_Host<P,R>::update_context(core::Context& ctx,
                                   const P& p) const
{
    for(auto& kv : trg_.optimap) ctx.remove(kv.first);

    std::size_t idx=0;
    for(const auto& kv : trg_.optimap){
        for(std::size_t i=0;i<kv.second.opt.size();++i){
            if(kv.second.opt[i]) ctx.add(kv.first, p(idx++));
            else                 ctx.add(kv.first, kv.second.config[i]);
        }
    }
}

/* ----------  helper: fill validation metrics -------------------------- */
template<class P, class R>
void HPO_Host<P,R>::fill_validation(Result& r,
                                    const mlip::Analytics& a) const
{
    if(trg_.use_emae)   r.emae  = a.calc_e_mae ().mean()*1000.0;
    if(trg_.use_ermse)  r.ermse = a.calc_e_rmse().mean()*1000.0;
    if(trg_.use_errmse) r.errmse= a.calc_e_rrmse().mean()*1000.0;
    if(trg_.use_ersq)   r.ersq  = a.calc_e_r_sq().mean();
    if(trg_.use_frmse)  r.frmse = a.calc_f_rmse().mean();
    if(trg_.use_srmse)  r.srmse = a.calc_s_rmse().mean();
}

/* ----------  helper: run LAMMPS (serial + OpenMP) --------------------- */
template<class P, class R>
void HPO_Host<P,R>::run_lammps(Result& r) const
{
  if(!use_lammps_) return;
  const auto& LS = trg_.LS;
  r.v_outvar.resize(LS.size());

#ifdef TADAH_ENABLE_OPENMP
#pragma omp parallel for shared(r,LS)
#endif
  for(std::size_t i=0;i<LS.size();++i)
  {
    const auto& lsim = LS[i];
    R runner;
    auto lmp = runner.create_instance(0, lsim.invar);
    if(!lmp) continue;
    bool fail = !runner.run_script(lmp, lsim.script_lines, r.pot_ctx);

    for(const auto& n : lsim.n_outvar) {
      if (fail) r.v_outvar[i].emplace_back(std::to_string(lsim.failscore));
      else  r.v_outvar[i].emplace_back(runner.get_variable(lmp, n));
    }
  }
}

template<class P, class R>
void HPO_Host<P,R>::fit_loss(Result& r) const
{
  double s_lin = 0.0, s_sq = 0.0;          // sums
  std::size_t c_lin = 0,   c_sq = 0;       // counts

  auto add_global = [&](double& sink,
      double  y, double ref, double w)
  {
    sink = trg_.globalLoss.loss(y, ref, w);

    if (trg_.globalLoss.is_squared())
    { s_sq += sink; ++c_sq; }
    else
    { s_lin += sink; ++c_lin; }
  };
  if (trg_.use_emae)   add_global(r.emaeLoss , r.emae , trg_.emae , trg_.w_emae );
  if (trg_.use_ermse)  add_global(r.ermseLoss, r.ermse, trg_.ermse, trg_.w_ermse);
  if (trg_.use_errmse) add_global(r.errmseLoss,r.errmse,trg_.errmse,trg_.w_errmse);
  if (trg_.use_ersq)   add_global(r.ersqLoss , r.ersq , trg_.ersq , trg_.w_ersq );
  if (trg_.use_frmse)  add_global(r.frmseLoss,r.frmse,trg_.frmse,trg_.w_frmse);
  if (trg_.use_srmse)  add_global(r.srmseLoss,r.srmse,trg_.srmse,trg_.w_srmse);

  // Probably better to use the following and apply sqrt when reporting?
  double part_lin = c_lin ? s_lin / static_cast<double>(c_lin) : 0.0;
  double part_sq  = c_sq  ? s_sq / static_cast<double>(c_sq) : 0.0;
  r.loss += part_lin + part_sq;
  if (r.loss > trg_.FAIL_SCORE) {
    r.loss = trg_.FAIL_SCORE; // cap the loss
  }
}
template<class P, class R>
void HPO_Host<P,R>::sims_loss(Result& r) const
{
  double s_lin = 0.0, s_sq = 0.0;          // sums
  std::size_t c_lin = 0,   c_sq = 0;       // counts

  r.v_loss.resize(trg_.LS.size());
  for (std::size_t i = 0; i < r.v_outvar.size(); ++i) {
    const auto& lsim = trg_.LS[i];
    r.v_loss[i].resize(lsim.n_outvar.size());

    for (auto j : lsim.i_varloss) {
      double v  = std::stod(r.v_outvar[i][j]);
      double w  = lsim.w_varloss[j];
      double l = lsim.lossFn[j].loss(v, lsim.t_varloss[j], w);
      r.v_loss[i][j] = l;

      if (lsim.lossFn[j].is_squared())
      { s_sq += l; ++c_sq; }
      else
      { s_lin += l; ++c_lin; }

    }
  }

  // Probably better to use the following and apply sqrt when reporting?
  double part_lin = c_lin ? s_lin / static_cast<double>(c_lin) : 0.0;
  double part_sq  = c_sq  ? s_sq / static_cast<double>(c_sq) : 0.0;
  r.loss += part_lin + part_sq;
  if (r.loss > trg_.FAIL_SCORE) {
    r.loss = trg_.FAIL_SCORE; // cap the loss
  }
}

/* ----------  operator()  (thread-safe) ------------------------------- */
template<class P, class R>
double HPO_Host<P,R>::operator()(const P& p) const
{
  Result r;
  r.iter   = ++g_iter_;
    r.params = p;

    auto t0 = std::chrono::high_resolution_clock::now();

    /* build context ---------------------------------------------------- */
    core::Context ctx = master_ctx_;
    update_context(ctx, p);

    /* train ------------------------------------------------------------ */
#if defined(TADAH_BUILD_MPI)
    int rank;
    int ncpu;

    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // New context file must be communicated to all processes
    std::vector<char> cs = context_local.serialize();
    int cs_size = cs.size();
    MPI_Bcast(&cs_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cs.data(), cs_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::lock_guard<std::mutex> lock(mpi_mutex_);     // serialise MPI
    host.run(); host.solve();
#else
    mlip::Trainer     host(ctx, tls_ws_);
    host.train(train_stdb_);
#endif

    /* validation ------------------------------------------------------- */
    core::Context pot = host.model->get_param_file();
    auto pred = host.model->predict(pot, val_stdb_, host.dc);
    mlip::Analytics   a(val_stdb_, pred);
    fill_validation(r, a);
    r.pot_ctx  = std::move(pot);
    fit_loss(r);

    /* optional LAMMPS -------------------------------------------------- */
    if (use_lammps_ && r.loss < trg_.FAIL_SCORE) {
#ifdef TADAH_BUILD_MPI
      run_lammps_mpi(r);
#else
      run_lammps(r);
#endif
      sims_loss(r);
    }

    /* best tracking ---------------------------------------------------- */
    double cur=g_best_.load(std::memory_order_relaxed);
    bool i_am_best = false;
    while(r.loss < cur &&
        !(i_am_best =
          g_best_.compare_exchange_weak(cur, r.loss,
            std::memory_order_acq_rel)));
    if(i_am_best){
      r.is_best   = true;
      r.best_iter = g_best_iter_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0);

    submit_result(std::move(r), trg_, cw_, prec_);
    return r.loss;
}

/* ---------- opportunistic writer helper ------------------------------ */
  template<class P,class R>
void HPO_Host<P,R>::submit_result(Result&& res, const HPOTargets& trg,
                                  std::size_t cw, int prec)
{
    /* 1. enqueue ------------------------------------------------------- */
    g_q_.enqueue(std::move(res));

    /* 2. quick exit if queue still small ------------------------------- */
    if (g_q_.size_approx() < trg.QFLUSH_THRESHOLD)
        return;

    /* 3. try to become the writer ------------------------------------- */
    if(writer_token_.test_and_set(std::memory_order_acquire))
        return;                                  // somebody else is writing


    /* 4. exclusive writer section ------------------------------------- */
    {
        std::lock_guard<std::mutex> lg(io_mtx_);

        Result item;
        while(g_q_.try_dequeue(item))
            io_buf_.emplace(item.iter,std::move(item));

        static std::ofstream loss_f  (trg.lossFile, std::ios::app),
                             outv_f  (trg.outvarFile, std::ios::app),
                             par_f   (trg.paramsFile, std::ios::app),
                             best_f  (trg.bestLossFile, std::ios::app),
                             best_outv_f(trg.bestOutvarFile, std::ios::app),
                             best_par_f (trg.bestParamsFile, std::ios::app);

        while(io_buf_.count(next_flush_))
        {
            const auto& e = io_buf_[next_flush_];

        /* ---- print standard logs ------------------------------------ */
        if(next_flush_ % trg.outputEvery == 0){
            print_loss_line  (loss_f , e.iter, e, trg, cw, prec);
            print_outvar_line(outv_f , e.iter, e, trg, cw, prec);
            print_params_line(par_f  , e.iter, e,      cw, prec);
        }

        /* ---- new best logs ------------------------------------------ */
        if(e.is_best) {
          if(trg.bestOutputEvery > 0) {
            print_loss_line  (best_f      , e.best_iter, e, trg, cw, prec);
            print_outvar_line(best_outv_f , e.best_iter, e, trg, cw, prec);
            print_params_line(best_par_f  , e.best_iter, e,      cw, prec);
          }

          /* overwrite & archive best_pot ---------------------------- */
          {
            std::ofstream bp("best_pot.tadah");

            bp << "#-----------------------------------------------------" << std::endl;
            bp<<"# HPO: Best potential at iter " << e.best_iter << " (" << e.iter << ")" << std::endl;
            auto now  = std::chrono::system_clock::now();          // current time-point
            auto tt   = std::chrono::system_clock::to_time_t(now); // to time_t

            bp << "# Time stamp: "
              << std::put_time(std::localtime(&tt), "%d.%m.%Y %H:%M:%S")
              << std::endl;
            bp<<"# Loss: " << e.loss << std::endl;
            bp << "#-----------------------------------------------------" << std::endl;
            bp<<"# Outvars:" << std::endl << "#";
            for (const auto &lsim : trg.LS) {
              for (auto n_outvar : lsim.n_outvar) {
                if (n_outvar.size() > cw - 1) {
                  n_outvar = n_outvar.substr(0, cw - 1 - 1) + "*";
                }
                bp << std::setw(cw) << n_outvar;
              }
            }
            bp << std::endl << "#";

            for (size_t i = 0; i < e.v_outvar.size(); ++i) {
              auto &indices = trg.LS[i].i_varloss;
              for (const auto &j : indices) {
                bp<<std::right << std::setw(cw) << std::scientific
                  << std::setprecision(prec) << std::stod(e.v_outvar[i][j]);
              }
            }
            bp<<std::endl;
            bp<<"# Losses: " << std::endl << "#";
            for (const auto &lsim : trg.LS) {
              for (auto varname : lsim.n_varloss) {
                if (varname.size() - 2 > cw - 1) {
                  varname = varname.substr(0, cw -1 - 4) + "*";
                }
                bp << std::setw(cw) << "L(" + varname + ")";
              }
            }
            bp << std::setw(cw) << "TOTAL_LOSS/N" << std::endl << "#";

            for (size_t i = 0; i < e.v_loss.size(); ++i) {
              auto &indices = trg.LS[i].i_varloss;
              for (const auto &j : indices) {
                bp<<std::right << std::setw(cw) << std::scientific
                  << std::setprecision(prec) << e.v_loss[i][j];
              }
            }
            bp<<std::endl;
            bp << "#-----------------------------------------------------" << std::endl;

            bp<<std::setprecision(e.pot_ctx.template get<size_t>("NUMERIC"))<<e.pot_ctx;
          }
          if(e.best_iter % trg.dumpBestPotEvery == 0){
            std::ofstream bpa(std::string(trg.dumpBestPotDir)
                +"/best_pot_" + std::to_string(e.best_iter) + ".tadah");
            bpa << "#-----------------------------------------------------" << std::endl;
            bpa<<"# HPO: Best potential at iter " << e.best_iter << " (" << e.iter << ")" << std::endl;
            auto now  = std::chrono::system_clock::now();          // current time-point
            auto tt   = std::chrono::system_clock::to_time_t(now); // to time_t

            bpa << "# Time stamp: "
              << std::put_time(std::localtime(&tt), "%d.%m.%Y %H:%M:%S")
              << std::endl;
            bpa<<"# Loss: " << e.loss << std::endl;
            bpa << "#-----------------------------------------------------" << std::endl;
            bpa<<"# Outvars:" << std::endl << "#";
            for (const auto &lsim : trg.LS) {
              for (auto n_outvar : lsim.n_outvar) {
                if (n_outvar.size() > cw - 1) {
                  n_outvar = n_outvar.substr(0, cw - 1 - 1) + "*";
                }
                bpa << std::setw(cw) << n_outvar;
              }
            }
            bpa << std::endl << "#";

            for (size_t i = 0; i < e.v_outvar.size(); ++i) {
              auto &indices = trg.LS[i].i_varloss;
              for (const auto &j : indices) {
                bpa<<std::right << std::setw(cw) << std::scientific
                  << std::setprecision(prec) << std::stod(e.v_outvar[i][j]);
              }
            }
            bpa<<std::endl;
            bpa<<"# Losses: " << std::endl << "#";
            for (const auto &lsim : trg.LS) {
              for (auto varname : lsim.n_varloss) {
                if (varname.size() - 2 > cw - 1) {
                  varname = varname.substr(0, cw -1 - 4) + "*";
                }
                bpa << std::setw(cw) << "L(" + varname + ")";
              }
            }
            bpa << std::setw(cw) << "TOTAL_LOSS/N" << std::endl << "#";

            for (size_t i = 0; i < e.v_loss.size(); ++i) {
              auto &indices = trg.LS[i].i_varloss;
              for (const auto &j : indices) {
                bpa<<std::right << std::setw(cw) << std::scientific
                  << std::setprecision(prec) << e.v_loss[i][j];
              }
            }
            bpa<<std::endl;
            bpa << "#-----------------------------------------------------" << std::endl;
            bpa<<std::setprecision(e.pot_ctx.template get<size_t>("NUMERIC"))<<e.pot_ctx;
          }
        }

        /* ---- archive ANY pot every ------------------------- */
        if(e.iter % trg.dumpPotEvery == 0){
            std::ofstream pa(std::string(trg.dumpPotDir)
                +"/pot_"+std::to_string(e.iter)+".tadah");
            pa<<std::setprecision(e.pot_ctx.template get<size_t>("NUMERIC"))<<e.pot_ctx;
        }

            io_buf_.erase(next_flush_++);
        }

        loss_f.flush();
        outv_f.flush();
        par_f.flush();
        best_f.flush();
        best_outv_f.flush();
        best_par_f.flush();
    }

    /* 5. release token ------------------------------------------------- */
    writer_token_.clear(std::memory_order_release);
}

#ifdef TADAH_BUILD_MPI
  void run_lammps_mpi(Result& r) {

    int rank;
    int ncpu;

    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


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

        auto g_it = std::find(available_workers.begin(), available_workers.end(),
                            worker);
        if (g_it == available_workers.end()) { // new worker is available
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

                // send wpkgs[j].n_outvar
                // First, send the number of strings
                num_strings = wpkgs[j].n_outvar.size();
                MPI_Send(&num_strings, 1, MPI_INT, worker, WORK_TAG,
                         MPI_COMM_WORLD);
                // Send each string
                for (const std::string &message : wpkgs[j].n_outvar) {
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
}   // namespace tadah::hpo

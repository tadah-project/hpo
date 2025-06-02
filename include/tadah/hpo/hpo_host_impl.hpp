#pragma once
#include <tadah/core/core_user_config_parser.h>
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

/* ----------  free, non-template  loss helper --------------------------- */
inline double sample_loss(double y, double ypred, double w,
                          std::size_t lossTypeId, unsigned N,
                          double delta, double tukeyC, double eps = 1e-12)
{
    double r    = y - ypred;
    double absr = std::abs(r);

    switch(lossTypeId)
    {
        case 0: return std::pow(absr, N) * w;                       // L1/L2
        case 1: return absr<=delta ? 0.5*r*r*w : delta*(absr-0.5*delta)*w; // Huber
        case 2: return stable_log_cosh(r) * w;                      // Log-Cosh
        case 3: return absr/(std::abs(y)+eps) * w;                  // MAPE
        case 4:{
            double ly=std::log(y+eps), lyp=std::log(ypred+eps);
            return (ly-lyp)*(ly-lyp)*w;                             // RMSLE
        }
        case 5:{
            if(absr<=tukeyC) {
                double t=r/tukeyC;
                return (tukeyC*tukeyC/6.0)*(1-std::pow(1-t*t,3))*w;
            }
            return (tukeyC*tukeyC/6.0)*w;
        }
    }
    return 0.0;
}

/* -----------------  header writers  ---------------------------------- */
template<class P,class R>
void HPO_Host<P,R>::init_loss(const std::string& fn) const
{
    std::ofstream f(fn);
    f<<std::left<<std::setw(cw_)<<"ITERATION";
    if(trg_.use_emae)  f<<std::setw(cw_)<<"loss(EMAE)";
    if(trg_.use_ermse) f<<std::setw(cw_)<<"loss(ERMSE)";
    if(trg_.use_errmse)f<<std::setw(cw_)<<"loss(ErRMSE)";
    if(trg_.use_ersq)  f<<std::setw(cw_)<<"loss(ERSQ)";
    if(trg_.use_frmse) f<<std::setw(cw_)<<"loss(FRMSE)";
    if(trg_.use_srmse) f<<std::setw(cw_)<<"loss(SRMSE)";
    for(const auto& ls: trg_.LS)
        for(const auto& n: ls.n_varloss)
            f<<std::setw(cw_)<<n;
    f<<std::setw(cw_)<<"TOTAL_LOSS\n";
}
template<class P,class R>
void HPO_Host<P,R>::init_outvar(const std::string& fn) const
{
    std::ofstream f(fn);
    f<<std::left<<std::setw(cw_)<<"ITERATION"
      <<std::setw(cw_)<<"IT_TIME/ms";
    if(trg_.use_emae)  f<<std::setw(cw_)<<"EMAE";
    if(trg_.use_ermse) f<<std::setw(cw_)<<"ERMSE";
    if(trg_.use_errmse)f<<std::setw(cw_)<<"ErRMSE";
    if(trg_.use_ersq)  f<<std::setw(cw_)<<"ERSQ";
    if(trg_.use_frmse) f<<std::setw(cw_)<<"FRMSE";
    if(trg_.use_srmse) f<<std::setw(cw_)<<"SRMSE";
    for(const auto& ls: trg_.LS)
        for(const auto& n: ls.outvar)
            f<<std::setw(cw_)<<n;
    f<<'\n';
}
template<class P,class R>
void HPO_Host<P,R>::init_params(const std::string& fn) const
{
    std::ofstream f(fn);
    f<<std::left<<std::setw(cw_)<<"ITER";
    for(const auto& kv: trg_.optimap)
        for(bool o: kv.second.opt)
            if(o) f<<std::setw(cw_)<<kv.first;
    f<<'\n';
}

/* ----------  line printers (append) ---------------------------------- */
template<class P,class R>
void HPO_Host<P,R>::print_loss(std::ofstream& f,const Result& e) const
{
    f<<std::left<<std::setw(cw_)<<e.iter;
    if(trg_.use_emae)   f<<std::setw(cw_)<<e.emae;
    if(trg_.use_ermse)  f<<std::setw(cw_)<<e.ermse;
    if(trg_.use_errmse) f<<std::setw(cw_)<<e.errmse;
    if(trg_.use_ersq)   f<<std::setw(cw_)<<e.ersq;
    if(trg_.use_frmse)  f<<std::setw(cw_)<<e.frmse;
    if(trg_.use_srmse)  f<<std::setw(cw_)<<e.srmse;
    for(const auto& v: e.v_loss)
        for(double d: v)
            f<<std::setw(cw_)<<d;
    f<<std::setw(cw_)<<e.loss<<'\n';
}
template<class P,class R>
void HPO_Host<P,R>::print_outvar(std::ofstream& f,const Result& e) const
{
    f<<std::left<<std::setw(cw_)<<e.iter
      <<std::setw(cw_)<<e.wall_ms.count();
    if(trg_.use_emae)   f<<std::setw(cw_)<<e.emae;
    if(trg_.use_ermse)  f<<std::setw(cw_)<<e.ermse;
    if(trg_.use_errmse) f<<std::setw(cw_)<<e.errmse;
    if(trg_.use_ersq)   f<<std::setw(cw_)<<e.ersq;
    if(trg_.use_frmse)  f<<std::setw(cw_)<<e.frmse;
    if(trg_.use_srmse)  f<<std::setw(cw_)<<e.srmse;
    for(const auto& s: e.v_outvar){
        std::string t=s.size()>cw_-1? s.substr(0,cw_-1):s;
        f<<std::setw(cw_)<<t;
    }
    f<<'\n';
}
template<class P,class R>
void HPO_Host<P,R>::print_params(std::ofstream& f,const Result& e) const
{
    f<<std::left<<std::setw(cw_)<<e.iter;
    for(long i=0;i<e.params.size();++i)
        f<<std::setw(cw_)<<std::scientific<<std::setprecision(prec_)<<e.params(i);
    f<<'\n';
}


/* ----------  static data definitions ---------------------------------- */
template<class P, class R>
std::atomic_size_t  HPO_Host<P,R>::g_iter_{0};
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
  typename HPO_Host<P,R>::Result>
  HPO_Host<P,R>::io_buf_;
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
, trg_(trg)
, use_lammps_( !trg.LS.empty() )
#ifdef TADAH_BUILD_MPI
, host(c, tls_ws_, tls_ws2_)
#endif
{
    /* validation set ---------------------------------------------------- */
    core::Context val_ctx(core::CoreUserConfigParser().parse(val_cfg));
    val_stdb_.add(val_ctx);
#ifndef TADAH_BUILD_MPI
    train_stdb_.add(master_ctx_);
#endif
    /* elements check ---------------------------------------------------- */
    auto tr_el = mlip::StructureDB::find_unique_elements(master_ctx_);
    for(auto e: mlip::StructureDB::find_unique_elements(val_ctx))
        if(!tr_el.count(e))
            throw std::runtime_error("Validation element missing in training");

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

    nnf_.calc(val_stdb_, rcutmax);

    init_loss   (loss_file_);
    init_outvar (outvar_file_);
    init_params (params_file_);
    init_loss   (bestLossFile_);
    init_outvar (bestOutvarFile_);
    init_params (bestParamsFile_);

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
    r.v_loss.resize(LS.size());

#ifdef TADAH_ENABLE_OPENMP
#pragma omp parallel for
#endif
    for(std::size_t i=0;i<LS.size();++i)
    {
        const auto& lsim = LS[i];
        R runner;
        auto lmp = runner.create_instance(0, lsim.sinvar);
        if(!lmp) continue;
        if(!runner.run_script(lmp, lsim.script)) continue;

        r.v_loss[i].resize(lsim.n_varloss.size());
        for(std::size_t j=0;j<lsim.n_varloss.size();++j){
            double v = std::stod(runner.get_variable(lmp, lsim.n_varloss[j]));
            r.v_loss[i][j] = std::pow(v, trg_.N)*lsim.w_varloss[j];
        }
        for(const auto& n : lsim.outvar)
            r.v_outvar.emplace_back(runner.get_variable(lmp, n));
    }
}

/* ----------  helper: accumulate total loss ---------------------------- */
template<class P, class R>
void HPO_Host<P,R>::accumulate_loss(Result& r) const
{
    double sum=0; std::size_t cnt=0;
    auto add=[&](double y,double ref,double w){
        sum+= sample_loss(y,ref,w,
                          trg_.lossTypeId, trg_.N,
                          trg_.deltaHuber, trg_.tukeyC); ++cnt; };

    if(trg_.use_emae)   add(r.emae , trg_.emae , trg_.w_emae );
    if(trg_.use_ermse)  add(r.ermse, trg_.ermse, trg_.w_ermse);
    if(trg_.use_errmse) add(r.errmse,trg_.errmse,trg_.w_errmse);
    if(trg_.use_ersq)   add(r.ersq , trg_.ersq , trg_.w_ersq );
    if(trg_.use_frmse)  add(r.frmse,trg_.frmse,trg_.w_frmse);
    if(trg_.use_srmse)  add(r.srmse,trg_.srmse,trg_.w_srmse);

    for(const auto& v : r.v_loss)
        for(double d: v){ sum+=d; ++cnt; }

    double mean=sum/std::max<std::size_t>(1,cnt);
    if(trg_.lossTypeId==0 && trg_.N==2) r.loss = std::sqrt(mean); // RMSE
    else if(trg_.lossTypeId==4)         r.loss = std::sqrt(mean); // RMSLE
    else                                r.loss = mean;
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
    //host.train(train_stdb_);
#endif

    /* validation ------------------------------------------------------- */
    core::Context pot = host.model->get_param_file();
    r.pot_ctx = pot;
    auto pred = host.model->predict(pot, val_stdb_, host.dc);
    mlip::Analytics   a(val_stdb_, pred);
    fill_validation(r, a);

    /* optional LAMMPS -------------------------------------------------- */
#ifdef TADAH_BUILD_MPI
    if (use_lammps)
      runLammpsMPI();
#else
    if (use_lammps_)
      run_lammps(r);
#endif

    /* total loss ------------------------------------------------------- */
    accumulate_loss(r);

    r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - t0);

    /* best tracking ---------------------------------------------------- */
    double cur=g_best_.load(std::memory_order_relaxed);
    while(r.loss<cur &&
          !g_best_.compare_exchange_weak(cur,r.loss,
                                         std::memory_order_acq_rel));
    if(r.loss==g_best_){
        r.is_best=true;
        r.best_pot=std::move(pot);
    }

    submit_result(std::move(r), cw_, prec_);
    return r.loss;
}

/* ---------- opportunistic writer helper ------------------------------ */
template<class P,class R>
void HPO_Host<P,R>::submit_result(Result&& res,
                                  std::size_t cw, int prec)
{
    /* 1. enqueue ------------------------------------------------------- */
    g_q_.enqueue(std::move(res));

    /* 2. quick exit if queue still small ------------------------------- */
    if(g_q_.size_approx() < FLUSH_THRESHOLD) return;

    /* 3. try to become the writer ------------------------------------- */
    if(writer_token_.test_and_set(std::memory_order_acquire))
        return;                                  // somebody else is writing

    /* 4. exclusive writer section ------------------------------------- */
    {
        std::lock_guard<std::mutex> lg(io_mtx_);

        Result item;
        while(g_q_.try_dequeue(item))
            io_buf_.emplace(item.iter,std::move(item));

        std::ofstream loss_f  ("loss.tadah"),
                             outv_f  ("outvar.tadah"),
                             par_f   ("params.tadah"),
                             best_f  ("best_loss.tadah"),
                             best_outv_f("best_outvar.tadah"),
                             best_par_f ("best_params.tadah");

        while(io_buf_.count(next_flush_))
        {
            const auto& e = io_buf_[next_flush_];

        /* ---- print standard logs ------------------------------------ */
        if(next_flush_ % lossEvery == 0){
            print_loss  (loss_f , e);
            print_outvar(outv_f , e);
            print_params(par_f  , e);
        }

        /* ---- new best logs ------------------------------------------ */
        if(e.is_best){
            best_f.seekp(0);
            best_f<<std::scientific<<std::setprecision(prec)<<e.loss<<'\n';

            print_loss  (best_f      , e);
            print_outvar(best_outv_f , e);
            print_params(best_par_f  , e);

            /* overwrite & archive best_pot ---------------------------- */
            {
                std::ofstream bp("best_pot.tadah");
                bp<<e.best_pot;
            }
            if(e.iter % bestPotEvery == 0){
                std::ofstream bpa(std::string(bestPotDir_)
                    +"/best_pot_" + std::to_string(e.iter) + ".tadah");
                bpa<<e.best_pot;
            }
        }

        /* ---- archive ANY pot every potEvery ------------------------- */
        if(e.iter % potEvery == 0){
            std::ofstream pa(std::string(potDir_)
                +"/pot_"+std::to_string(e.iter)+".tadah");
            pa<<e.pot_ctx;
        }

            io_buf_.erase(next_flush_++);
        }
    }

    /* 5. release token ------------------------------------------------- */
    writer_token_.clear(std::memory_order_release);
}


#ifdef TADAH_BUILD_MPI
  void runLammpsMPI() {

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
}   // namespace tadah::hpo

/* algorithms_asa.cpp -- LIB ASA / ALGO ASA driver.
 *
 * The ONLY translation unit that includes the vendored ASA headers
 * (external/asa; BSD-3-Clause, see TADAH-PATCHES.md there). Everything
 * ASA-shaped stays file-local: the C cost/RNG trampolines, the
 * block-parallel hook, the USER_DEFINES workspace, and the process-wide
 * single-instance mutex (asa() is not re-entrant upstream).
 *
 * Exception discipline: no C++ exception may unwind through asa()'s C
 * frames. Every callback catches, stashes the first std::exception_ptr,
 * flags the state invalid, and requests IMMEDIATE_EXIT; the driver
 * rethrows after asa() returns (log-then-unwind, the PT precedent). */

#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/asa_testing.h>
#include <tadah/hpo/anneal_parallel.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/warmstart_cache.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <tadah_asa_hook.h>   /* pins the gate set + asa_usr_asa.h;
                               * SYSTEM include dir of target tadah_asa */
}

#ifndef MAX_DOUBLE
#define MAX_DOUBLE (1.0 / SMALL_FLOAT)
#endif
#ifndef EPS_DOUBLE
#define EPS_DOUBLE SMALL_FLOAT
#endif

namespace tadah::hpo {
namespace {

/* ------------------------------------------------------------------ */
/* Run context reachable from the C callbacks via OPTIONS->Asa_Data_Ptr */

struct AsaRunCtx {
    const AsaHostFn* eval = nullptr;
    std::size_t      dim  = 0;
    std::optional<double> failscore;

    /* stops */
    std::uint64_t maxeval = std::numeric_limits<std::uint64_t>::max();
    std::atomic<std::uint64_t> reserved{0};
    std::function<bool()> time_exceeded;
    std::optional<double> stopval;
    std::atomic<bool> pending_stop{false};

    enum Why : int { None = 0, Budget, Time, StopVal, Exception };
    std::atomic<int> why{None};
    void set_why(Why w)
    {
        int expected = None;
        why.compare_exchange_strong(expected, static_cast<int>(w));
        pending_stop.store(true, std::memory_order_relaxed);
    }

    std::mutex         err_mtx;
    std::exception_ptr error;
    void stash(std::exception_ptr e)
    {
        {
            std::lock_guard<std::mutex> lk(err_mtx);
            if (!error) error = std::move(e);
        }
        set_why(Exception);
    }

    bool try_reserve()
    {
        const auto u = reserved.fetch_add(1, std::memory_order_relaxed);
        if (u < maxeval) return true;
        reserved.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }

    /* counters / observers */
    std::atomic<std::uint64_t> n_evals{0};
    std::atomic<double> last_valid_cost{0.0};
    std::atomic<double> best_seen{std::numeric_limits<double>::infinity()};

    /* progress-line cadence (ASA thread only) */
    long next_accept_mark = 0;
    int  accept_modulus   = 0;

    /* Live totals observed from OPTIONS at every callback: asa()
     * restores N_Generated/N_Accepted to their AT-BEST values on exit
     * (upstream semantics), which reads as generated=0 when the best
     * never left the seed. Monotonic max of the live values gives the
     * honest totals for the summary (ASA thread only). */
    long seen_generated = 0;
    long seen_accepted  = 0;
    void observe_counters(const USER_DEFINES* o)
    {
        if (static_cast<long>(o->N_Generated) > seen_generated)
            seen_generated = static_cast<long>(o->N_Generated);
        if (static_cast<long>(o->N_Accepted) > seen_accepted)
            seen_accepted = static_cast<long>(o->N_Accepted);
    }

    /* block-parallel pool (installed only when Gener_Block_Max > 1) */
    anneal::ReplicaPool* pool = nullptr;
};

bool cost_is_valid(const AsaRunCtx& c, double loss)
{
    if (!std::isfinite(loss)) return false;
    if (c.failscore && loss >= *c.failscore) return false;  /* hard fail */
    if (!(loss > -MAX_DOUBLE && loss < MAX_DOUBLE)) return false;
    return true;
}

/* Core objective evaluation: host call + validity + counters + a
 * pending STOPVAL flag. No budget/time/stop gating here -- callers
 * decide that (per eval on the serial path, per block on the hook
 * path, which keeps results THREADS-invariant). Safe on pool workers.
 * Returns the cost; *valid_out is TRUE/FALSE. Invalid states are ASA's
 * own retry currency (they count toward
 * Limit_Invalid_Generated_States). */
double eval_core(AsaRunCtx& c, const double* x, int* valid_out)
{
    *valid_out = FALSE;
    const double sentinel = c.last_valid_cost.load(std::memory_order_relaxed);
    double loss;
    try {
        const std::vector<double> xv(x, x + c.dim);
        loss = (*c.eval)(xv);
    } catch (...) {
        c.stash(std::current_exception());
        return sentinel;
    }
    c.n_evals.fetch_add(1, std::memory_order_relaxed);
    if (!cost_is_valid(c, loss)) return sentinel;
    *valid_out = TRUE;
    c.last_valid_cost.store(loss, std::memory_order_relaxed);
    double prev = c.best_seen.load(std::memory_order_relaxed);
    while (loss < prev &&
           !c.best_seen.compare_exchange_weak(prev, loss)) {}
    if (c.stopval && loss <= *c.stopval) c.set_why(AsaRunCtx::StopVal);
    return loss;
}

/* Serial-path evaluation: per-eval budget/time/stop gating. */
double eval_one(AsaRunCtx& c, const double* x, int* valid_out)
{
    *valid_out = FALSE;
    const double sentinel = c.last_valid_cost.load(std::memory_order_relaxed);
    if (c.pending_stop.load(std::memory_order_relaxed)) return sentinel;
    if (!c.try_reserve()) {
        c.set_why(AsaRunCtx::Budget);
        return sentinel;
    }
    if (c.time_exceeded && c.time_exceeded()) {
        c.set_why(AsaRunCtx::Time);
        return sentinel;
    }
    return eval_core(c, x, valid_out);
}

/* Progress line on the accepted-states cadence; ASA thread only. */
void maybe_progress_line(AsaRunCtx& c, USER_DEFINES* o)
{
    if (c.accept_modulus <= 0) return;
    if (o->N_Accepted < c.next_accept_mark) return;
    c.next_accept_mark =
        (o->N_Accepted / c.accept_modulus + 1) * c.accept_modulus;
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3)
        << "ASA gen=" << o->N_Generated << " acc=" << o->N_Accepted;
    if (o->Best_Cost && std::isfinite(*o->Best_Cost))
        oss << " best=" << *o->Best_Cost;
    if (o->Last_Cost && std::isfinite(*o->Last_Cost))
        oss << " last=" << *o->Last_Cost;
    if (std::isfinite(tadah_asa_cost_temp))
        oss << " Tcost=" << tadah_asa_cost_temp;
    HPOLog::instance().info(oss.str());
}

/* ------------------------------------------------------------------ */
/* C trampolines. tadah_asa_cost() serves every SERIAL call site
 * (initial cost-temperature sampling, generated states when no hook is
 * installed, reanneal tangents, exit curvature) and always runs on
 * asa()'s own thread -- setting Immediate_Exit here is race-free. */

extern "C" double tadah_asa_cost(double* x, double* /*lo*/, double* /*hi*/,
                                 double* /*tangents*/, double* /*curvature*/,
                                 ALLOC_INT* /*dim*/, int* /*param_type*/,
                                 int* cost_flag, int* /*exit_code*/,
                                 USER_DEFINES* o)
{
    auto* c = static_cast<AsaRunCtx*>(o->Asa_Data_Ptr);
    if (*cost_flag == -1) {
        /* User_Tangents request -- rejected at parse time; defensive. */
        c->stash(std::make_exception_ptr(std::logic_error(
            "ASA:ASA internal: User_Tangents callback reached the host")));
        *cost_flag = FALSE;
        return c->last_valid_cost.load(std::memory_order_relaxed);
    }
    const double cost = eval_one(*c, x, cost_flag);
    if (c->pending_stop.load(std::memory_order_relaxed))
        o->Immediate_Exit = TRUE;
    c->observe_counters(o);
    maybe_progress_line(*c, o);
    return cost;
}

std::mt19937_64* g_asa_rng = nullptr;

extern "C" double tadah_asa_rng(LONG_INT* /*seed*/)
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(*g_asa_rng);
}

/* Block-parallel hook: runs on asa()'s thread, fans the fresh slots of
 * the generated block onto the ReplicaPool (static task->worker map =>
 * results are THREADS-invariant; ASA sorts the block by cost before
 * consuming it sequentially, so evaluation order never matters). */
extern "C" void tadah_asa_block_hook(tadah_asa_state_view* block,
                                     LONG_INT gener_block,
                                     double* /*lo*/, double* /*hi*/,
                                     double* /*tangents*/,
                                     double* /*curvature*/,
                                     ALLOC_INT* /*dim*/,
                                     int* /*param_type*/,
                                     int* valid_par, int* /*exit_status*/,
                                     USER_DEFINES* o,
                                     int* queue_new_par,
                                     double* /*queue_par_cost*/)
{
    auto* c = static_cast<AsaRunCtx*>(o->Asa_Data_Ptr);
    const auto n = static_cast<std::size_t>(gener_block);

    /* Reserve the budget for the whole block SERIALLY (slot order), so
     * which slots evaluate never depends on worker timing: results are
     * THREADS-invariant even when MAXEVAL dies mid-block. Time/stopval
     * take effect at the block boundary (documented block-granular
     * stop semantics of Gener_Block > 1). */
    std::vector<char> may_run(n, 0);
    const bool stopping = c->pending_stop.load(std::memory_order_relaxed);
    for (std::size_t t = 0; t < n; ++t) {
        if (queue_new_par && queue_new_par[t] == 0) continue;  /* cached */
        if (stopping) { valid_par[t] = FALSE; continue; }
        if (!c->try_reserve()) {
            c->set_why(AsaRunCtx::Budget);
            valid_par[t] = FALSE;
            block[t].cost =
                c->last_valid_cost.load(std::memory_order_relaxed);
            continue;
        }
        may_run[t] = 1;
    }

    const double t_cost = tadah_asa_cost_temp;  /* NaN before 1st block */
    c->pool->run_epoch(n, [&](std::size_t t) {
        if (!may_run[t]) return;
        warmstart::current_chain() = t;
        HPOLog::set_eval_slot(static_cast<int>(t), n, t_cost, "Tcost");
        int flag = FALSE;
        const double cost = eval_core(*c, block[t].parameter, &flag);
        block[t].cost = cost;
        valid_par[t]  = flag;
    });
    try {
        c->pool->rethrow_if_error();
    } catch (...) {
        c->stash(std::current_exception());
    }
    if (c->time_exceeded && c->time_exceeded())
        c->set_why(AsaRunCtx::Time);
    if (c->pending_stop.load(std::memory_order_relaxed))
        o->Immediate_Exit = TRUE;
    c->observe_counters(o);
    maybe_progress_line(*c, o);
}

/* ------------------------------------------------------------------ */
/* Workspace owning every array asa() touches.                         */

struct AsaWorkspace {
    std::vector<double> x, lo, hi, tangents, curvature;
    std::vector<int>    parameter_type;
    std::vector<double> multi_cost;
    std::vector<double> multi_params_block;
    std::vector<double*> multi_params;
    std::vector<char>   out_file;
    LONG_INT  rand_seed  = 696969;
    ALLOC_INT n_params   = 0;
    int       valid_flag = TRUE;
    int       exit_status = 0;
    USER_DEFINES opts;

    AsaWorkspace(const OptimizerSpec& sp, AsaOptions& ao, std::size_t dim,
                 const std::string& out_name)
    {
        std::memset(&opts, 0, sizeof(opts));
        n_params = static_cast<ALLOC_INT>(dim);
        lo = sp.low;
        hi = sp.high;

        x.assign(dim, 0.0);
        if (sp.optim_init.empty()) {
            for (std::size_t i = 0; i < dim; ++i)
                x[i] = lo[i] + 0.5 * (hi[i] - lo[i]);
        } else {
            const auto& raw = sp.optim_init.front();
            if (raw.size() != dim)
                throw std::runtime_error(
                    "ASA:ASA: starting point has " +
                    std::to_string(raw.size()) + " coordinate(s), expected " +
                    std::to_string(dim));
            for (std::size_t i = 0; i < dim; ++i) {
                double v = raw[i];
                if (!sp.log_scaled.empty() && sp.log_scaled[i]) {
                    if (!(v > 0.0) || !std::isfinite(v))
                        throw std::runtime_error(
                            "ASA:ASA: log-scaled starting coordinate must "
                            "be finite and > 0");
                    v = std::log(v);
                }
                x[i] = v;
            }
        }
        for (std::size_t i = 0; i < dim; ++i) {
            if (!(hi[i] - lo[i] >= EPS_DOUBLE))
                throw std::runtime_error(
                    "ASA:ASA: OPTIM interval of dimension " +
                    std::to_string(i) +
                    " is below ASA's EPS_DOUBLE resolution; ASA would "
                    "silently freeze it -- widen the OPTIM box");
            if (!std::isfinite(x[i]) || x[i] < lo[i] || x[i] > hi[i])
                throw std::runtime_error(
                    "ASA:ASA: starting coordinate " + std::to_string(i) +
                    " lies outside its OPTIM bounds");
        }

        tangents.assign(dim, 0.0);
        curvature.assign(ao.curvature_0 == 1 ? 1 : dim * dim, 0.0);
        parameter_type.assign(dim, REAL_TYPE);
        for (std::size_t i = 0; i < dim && i < sp.is_int.size(); ++i)
            if (sp.is_int[i]) parameter_type[i] = INTEGER_TYPE;

        multi_cost.assign(static_cast<std::size_t>(ao.multi_number), 0.0);
        multi_params_block.assign(
            static_cast<std::size_t>(ao.multi_number) * dim, 0.0);
        multi_params.resize(static_cast<std::size_t>(ao.multi_number));
        for (std::size_t m = 0; m < multi_params.size(); ++m)
            multi_params[m] = multi_params_block.data() + m * dim;

        out_file.assign(out_name.begin(), out_name.end());
        out_file.push_back('\0');

        /* ---- USER_DEFINES ------------------------------------------ */
        opts.Limit_Acceptances       = ao.limit_acceptances;
        opts.Limit_Generated         = ao.limit_generated;
        opts.Limit_Invalid_Generated_States =
            ao.limit_invalid_generated_states;
        opts.Accepted_To_Generated_Ratio = ao.accepted_to_generated_ratio;
        opts.Cost_Precision          = ao.cost_precision;
        opts.Maximum_Cost_Repeat     = ao.maximum_cost_repeat;
        opts.Number_Cost_Samples     = ao.number_cost_samples;
        opts.Temperature_Ratio_Scale = ao.temperature_ratio_scale;
        opts.Cost_Parameter_Scale_Ratio = ao.cost_parameter_scale_ratio;
        opts.Temperature_Anneal_Scale   = ao.temperature_anneal_scale;
        opts.Include_Integer_Parameters = ao.include_integer_parameters;
        opts.User_Initial_Parameters    = ao.user_initial_parameters;
        opts.Sequential_Parameters      =
            static_cast<ALLOC_INT>(ao.sequential_parameters);
        opts.Initial_Parameter_Temperature =
            ao.initial_parameter_temperature;
        opts.Acceptance_Frequency_Modulus =
            ao.acceptance_frequency_modulus;
        opts.Generated_Frequency_Modulus = ao.generated_frequency_modulus;
        opts.Reanneal_Cost           = ao.reanneal_cost;
        opts.Reanneal_Parameters     = ao.reanneal_parameters;
        opts.Delta_X                 = ao.delta_x;
        opts.User_Tangents           = ao.user_tangents;
        opts.Curvature_0             = ao.curvature_0;
        opts.User_Quench_Cost_Scale  = &ao.user_quench_cost_scale;
        opts.User_Quench_Param_Scale = ao.user_quench_param_scale.data();
        opts.User_Temperature_Ratio  = ao.user_temperature_ratio.data();
        opts.User_Delta_Parameter    = ao.user_delta_parameter.data();
        opts.User_Parameter_Temperature =
            ao.user_parameter_temperature.data();
        opts.Multi_Number  = ao.multi_number;
        opts.Multi_Cost    = multi_cost.data();
        opts.Multi_Params  = multi_params.data();
        opts.Multi_Grid    = ao.multi_grid.data();
        opts.Multi_Specify = ao.multi_specify;
        opts.Queue_Size    = static_cast<ALLOC_INT>(ao.queue_size);
        opts.Queue_Resolution = ao.queue_resolution.data();
        opts.parallel_id     = -1;
        opts.Gener_Mov_Avr   = ao.gener_mov_avr;
        opts.Gener_Block     = ao.gener_block;
        opts.Gener_Block_Max = ao.gener_block_max;
        opts.Asa_Out_File    = out_file.data();
        opts.Immediate_Exit  = FALSE;
        opts.Asa_Recursive_Level = 0;
        opts.Asa_Data_Dim_Ptr = 1;
    }
};

/* asa() is not re-entrant: one instance per process, globally. The RNG
 * pointer and the hook installation live under the same mutex. */
std::mutex g_asa_mutex;
std::atomic<int> g_asa_run_counter{0};

struct HookInstallGuard {
    explicit HookInstallGuard(bool install)
    {
        if (install) tadah_asa_parallel_hook = &tadah_asa_block_hook;
    }
    ~HookInstallGuard() { tadah_asa_parallel_hook = nullptr; }
};

struct RngGuard {
    explicit RngGuard(std::mt19937_64* rng) { g_asa_rng = rng; }
    ~RngGuard() { g_asa_rng = nullptr; }
};

std::string stop_reason_string(const AsaRunCtx& c, int exit_status)
{
    switch (c.why.load(std::memory_order_relaxed)) {
        case AsaRunCtx::Budget:  return "MAXEVAL reached";
        case AsaRunCtx::Time:    return "MAXTIME exceeded";
        case AsaRunCtx::StopVal: return "STOPVAL reached";
        case AsaRunCtx::Exception:
            return "exception raised during evaluation";
        default: break;
    }
    switch (exit_status) {
        case 0: return "Limit_Acceptances/Limit_Generated reached";
        case 1: return "a parameter temperature annealed below "
                       "EPS_DOUBLE (often acceptable convergence)";
        case 2: return "the cost temperature annealed below EPS_DOUBLE";
        case 3: return "cost plateau: |last-best| < Cost_Precision for "
                       "Maximum_Cost_Repeat test cycles";
        case 4: return "too many invalid states (host failures exceeded "
                       "Limit_Invalid_Generated_States)";
        case 5: return "IMMEDIATE_EXIT requested";
        case 9: return "a host failure occurred while ASA was computing "
                       "reanneal tangents (INVALID_COST_FUNCTION_DERIV); "
                       "with an intermittently failing objective consider "
                       "PARAM Reanneal_Parameters 0 or Delta_X 0";
        default: return std::string("asa exit ") + asa_exit_name(exit_status);
    }
}

}  // namespace

void run_asa_erased(const OptimizerSpec& sp, std::size_t dim,
                    const AsaHostFn& eval)
{
    if (dim == 0)
        throw std::runtime_error(
            "ASA:ASA requires at least one OPTIM dimension");
    if (sp.low.size() != dim || sp.high.size() != dim)
        throw std::runtime_error("ASA:ASA: bounds/dim mismatch");

    AsaOptions ao = parse_asa_options(sp, dim);

    const std::uint64_t seed = sp.seed.value_or(std::random_device{}());
    const int run_no =
        g_asa_run_counter.fetch_add(1, std::memory_order_relaxed);
    std::string out_name = std::string("asa_out") + hpo_rank_suffix();
    if (run_no > 0) out_name += "." + std::to_string(run_no);

    AsaRunCtx ctx;
    ctx.eval      = &eval;
    ctx.dim       = dim;
    ctx.failscore = sp.failscore;
    ctx.maxeval   = sp.stop.maxeval.value_or(
        std::numeric_limits<std::uint64_t>::max());
    ctx.stopval   = sp.stop.stopval;
    ctx.accept_modulus   = ao.acceptance_frequency_modulus;
    ctx.next_accept_mark = ao.acceptance_frequency_modulus;

    const auto t0_wall = std::chrono::steady_clock::now();
    const double max_time = sp.stop.maxtime.value_or(0.0);
    ctx.time_exceeded = [t0_wall, max_time]() {
        if (max_time <= 0.0) return false;
        using Sec = std::chrono::duration<double>;
        return std::chrono::duration_cast<Sec>(
                   std::chrono::steady_clock::now() - t0_wall)
                   .count() >= max_time;
    };

    AsaWorkspace w(sp, ao, dim, out_name);
    w.opts.Asa_Data_Ptr = &ctx;
    w.rand_seed = static_cast<LONG_INT>(seed % 2147483647ULL) + 1;

    const unsigned threads = sp.num_threads.value_or(1);
    /* The hook path is used whenever it could matter (THREADS > 1 or a
     * block cap above 1); at Gener_Block(_Max)=1 it reproduces the
     * serial trajectory exactly (unit-tested identity). */
    const bool parallel_blocks = threads > 1 || ao.gener_block_max > 1;

    /* ---- banner ----------------------------------------------------- */
    {
        std::ostringstream oss;
        oss << "ASA[Ingber ASA-30.53]: seed=" << seed << " dim=" << dim
            << "  Limit_Acceptances=" << ao.limit_acceptances
            << "  Limit_Generated=" << ao.limit_generated
            << "  Temperature_Ratio_Scale=" << ao.temperature_ratio_scale
            << "  Temperature_Anneal_Scale=" << ao.temperature_anneal_scale
            << "  Cost_Parameter_Scale_Ratio="
            << ao.cost_parameter_scale_ratio
            << "  Reanneal_Parameters=" << ao.reanneal_parameters
            << "  Reanneal_Cost=" << ao.reanneal_cost
            << "  Curvature_0=" << ao.curvature_0
            << "  User_Initial_Parameters=" << ao.user_initial_parameters;
        if (ao.user_quench_cost_scale != 1.0)
            oss << "  User_Quench_Cost_Scale=" << ao.user_quench_cost_scale;
        if (ao.multi_number > 1) oss << "  Multi_Number=" << ao.multi_number;
        if (ao.queue_size > 0)   oss << "  Queue_Size=" << ao.queue_size;
        if (parallel_blocks)
            oss << "  Gener_Block=" << ao.gener_block
                << "  Gener_Block_Max=" << ao.gener_block_max
                << "  workers="
                << std::min<unsigned>(
                       threads, static_cast<unsigned>(ao.gener_block_max));
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().info(oss.str());
        HPOLog::instance().info(
            "ASA: upstream journal -> " + out_name +
            " (asa() truncates it per run)");
        if (threads > 1 && ao.gener_block_max <= 1)
            HPOLog::instance().hint(
                "ASA: THREADS>1 with Gener_Block(_Max)=1 evaluates one "
                "generated state at a time -- set PARAM Gener_Block "
                "(and Gener_Block_Max) to keep the workers busy, or "
                "rely on the per-eval pc_lammps fan-out alone");
        if (ao.gener_block > 1 || ao.gener_block_max > ao.gener_block)
            HPOLog::instance().info(
                "ASA: Gener_Block>1 (or an adaptive Gener_Block_Max "
                "above it) changes the sampled sequence vs serial ASA "
                "(upstream-documented); results are invariant to "
                "THREADS, and MAXTIME/STOPVAL act at block boundaries");
    }
    if (ao.gener_block > 1)
        HPOLog::instance().set_cadence(
            static_cast<std::size_t>(ao.gener_block), "block",
            sp.stop.maxeval ? static_cast<std::uint64_t>(*sp.stop.maxeval)
                            : 0);

    /* ---- the run (single instance per process) ---------------------- */
    double final_cost = 0.0;
    long   exit_index = -1;   /* P_TEMP_TOO_SMALL dimension, if any */
    {
        std::lock_guard<std::mutex> lk(g_asa_mutex);
        std::mt19937_64 rng(seed);
        RngGuard rng_guard(&rng);
        std::optional<anneal::ReplicaPool> pool;
        if (parallel_blocks) {
            pool.emplace(
                std::min<std::size_t>(
                    threads, static_cast<std::size_t>(ao.gener_block_max)),
                ctx.pending_stop);
            ctx.pool = &*pool;
        }
        HookInstallGuard hook_guard(parallel_blocks);
        tadah_asa_exit_index = -1;
        tadah_asa_cost_temp = std::numeric_limits<double>::quiet_NaN();
        final_cost =
            asa(&tadah_asa_cost, &tadah_asa_rng, &w.rand_seed, w.x.data(),
                w.lo.data(), w.hi.data(), w.tangents.data(),
                w.curvature.data(), &w.n_params, w.parameter_type.data(),
                &w.valid_flag, &w.exit_status, &w.opts);
        exit_index = static_cast<long>(tadah_asa_exit_index);
        ctx.pool = nullptr;
    }

    /* ---- summary + status ------------------------------------------ */
    const int exit_status = w.exit_status;
    std::string reason = stop_reason_string(ctx, exit_status);
    if (exit_status == 1 && exit_index >= 0) {
        reason += " [dim " + std::to_string(exit_index);
        if (static_cast<std::size_t>(exit_index) <
            sp.parameter_labels.size())
            reason += ": " + sp.parameter_labels[
                static_cast<std::size_t>(exit_index)];
        reason += "]";
    }
    {
        /* seen_* are live maxima; on exit asa() restores N_Generated/
         * N_Accepted to their AT-BEST snapshots (0/0 when the best is
         * the seed), reported here as at-best for reference. */
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6)
            << "ASA summary: generated=" << ctx.seen_generated
            << " accepted=" << ctx.seen_accepted
            << " host_evals=" << ctx.n_evals.load()
            << " best=" << final_cost
            << " exit=" << asa_exit_name(exit_status)
            << "  (at-best: generated=" << w.opts.N_Generated
            << " accepted=" << w.opts.N_Accepted << ")";
        HPOLog::instance().info(oss.str());
    }
    if (ao.multi_number > 1) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6)
            << "ASA multi-min costs:";
        for (int m = 0; m < ao.multi_number; ++m)
            oss << " [" << m << "]=" << w.multi_cost[m];
        HPOLog::instance().info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "ASA stopped: " << reason << " after " << ctx.n_evals.load()
            << " host eval(s) (asa exit: " << asa_exit_name(exit_status)
            << ")";
        std::cout << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().info(oss.str());
        HPOLog::instance().set_status("ASA: " + reason);
    }

    /* ---- unwind ------------------------------------------------------ */
    {
        std::lock_guard<std::mutex> lk(ctx.err_mtx);
        if (ctx.error) std::rethrow_exception(ctx.error);
    }
    if (ctx.why.load() == AsaRunCtx::None &&
        (exit_status == 7 || exit_status == 8 || exit_status == -1))
        throw std::runtime_error(
            std::string("ASA:ASA aborted: ") + asa_exit_name(exit_status) +
            " (this indicates a glue or configuration bug -- see " +
            out_name + ")");
}

/* ------------------------------------------------------------------ */
/* Golden-neutrality test seam: run the featured asa() directly over a
 * plain-C callback pair at canonical-default options (AsaOptions{}
 * with every gate at its neutral value). See asa_testing.h.           */

namespace asa_testing {

namespace {
RawCost g_raw_cost = nullptr;
RawRng  g_raw_rng  = nullptr;

extern "C" double raw_cost_tramp(double* x, double* /*lo*/, double* /*hi*/,
                                 double* /*tang*/, double* /*curv*/,
                                 ALLOC_INT* dim, int* /*type*/,
                                 int* cost_flag, int* /*exit_code*/,
                                 USER_DEFINES* /*o*/)
{
    int flag = 1;
    const double f =
        g_raw_cost(x, static_cast<long>(*dim), &flag);
    *cost_flag = flag ? TRUE : FALSE;
    return f;
}

extern "C" double raw_rng_tramp(LONG_INT* /*seed*/) { return g_raw_rng(); }
}  // namespace

RawResult run_raw_featured(RawCost cost, RawRng rng, std::size_t dim,
                           const double* lo, const double* hi,
                           const double* x0, long limit_generated)
{
    OptimizerSpec sp;
    sp.lib  = "ASA";
    sp.algo = "ASA";
    sp.low.assign(lo, lo + dim);
    sp.high.assign(hi, hi + dim);
    sp.optim_init.assign(1, std::vector<double>(x0, x0 + dim));

    AsaOptions ao = parse_asa_options(sp, dim);
    ao.limit_generated = limit_generated;

    AsaWorkspace w(sp, ao, dim, "asa_out.golden_featured");
    /* asa_test_asa_options rejects a NULL Asa_Data_Ptr; the raw
     * trampolines never dereference it. */
    static int dummy_user_data = 0;
    w.opts.Asa_Data_Ptr = &dummy_user_data;

    RawResult r;
    std::lock_guard<std::mutex> lk(g_asa_mutex);
    g_raw_cost = cost;
    g_raw_rng  = rng;
    r.final_cost =
        asa(&raw_cost_tramp, &raw_rng_tramp, &w.rand_seed, w.x.data(),
            w.lo.data(), w.hi.data(), w.tangents.data(), w.curvature.data(),
            &w.n_params, w.parameter_type.data(), &w.valid_flag,
            &w.exit_status, &w.opts);
    g_raw_cost = nullptr;
    g_raw_rng  = nullptr;
    r.exit_status = w.exit_status;
    r.n_accepted  = static_cast<long>(w.opts.N_Accepted);
    r.n_generated = static_cast<long>(w.opts.N_Generated);
    r.best_x      = w.x;
    return r;
}

}  // namespace asa_testing
}  // namespace tadah::hpo

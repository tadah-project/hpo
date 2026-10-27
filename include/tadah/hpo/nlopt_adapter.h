#pragma once
#include <nlopt.hpp>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/failed_loss_stand_in.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/initial_step.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/repeat_points.h>
#include <dlib/rand.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace tadah::hpo {

/* small helper ---------------------------------------------------- */
inline nlopt_algorithm nlopt_algo(const std::string& s)
{
    auto a = nlopt_algorithm_from_string(s.c_str());
    if (a==NLOPT_NUM_ALGORITHMS)
        throw std::invalid_argument("unknown NLopt algo "+s);
    return a;
}

/* How run_nlopt treats an exception that escaped nlopt::opt::optimize()
 * while the objective raised none (the objective's own exceptions are not
 * classified at all: run_nlopt_impl() rethrows them with their own type
 * first).
 *   Graceful     the algorithm terminated itself -- nothing left to
 *                explore; the best result found so far is valid and
 *                already saved;
 *   InvalidArgs  NLopt rejected the setup; FAILED, rethrown;
 *   Failed       anything else (out of memory, a forced stop, ...);
 *                FAILED, rethrown. */
enum class NloptOutcome {
    Graceful,
    InvalidArgs,
    Failed
};

/** Classify by TYPE, not message text alone (`errmsg` = opt.get_errmsg(),
 *  may be null):
 *    nlopt::roundoff_limited (NLOPT_ROUNDOFF_LIMITED, -4: progress blocked
 *      by floating-point roundoff)                  -> Graceful
 *    exactly std::runtime_error with what() "nlopt failure" or == errmsg
 *      (NLOPT_FAILURE, -1: population converged, local mutation simplex
 *      degenerated; the C++ wrapper throws runtime_error(<stop message>)
 *      when NLopt set one, e.g. LN_NELDERMEAD's "starting step size led to
 *      simplex that was too small in dimension N")  -> Graceful
 *    std::invalid_argument (NLOPT_INVALID_ARGS)     -> InvalidArgs
 *    nlopt::forced_stop, std::bad_alloc, a subclass of runtime_error
 *      (durable_best::Error, ... -- never NLopt's), any other type or
 *      message                                      -> Failed */
inline NloptOutcome classify_nlopt_exception(std::exception_ptr e,
                                             const char* errmsg)
{
    try {
        std::rethrow_exception(e);
    } catch (const nlopt::roundoff_limited&) {
        return NloptOutcome::Graceful;
    } catch (const nlopt::forced_stop&) {
        return NloptOutcome::Failed;
    } catch (const std::invalid_argument&) {
        return NloptOutcome::InvalidArgs;
    } catch (const std::runtime_error& ex) {
        /* nlopt::opt::mythrow(NLOPT_FAILURE) throws a plain runtime_error
         * carrying the opt's stop message, or "nlopt failure" without
         * one. A subclass (durable_best::Error, ...) is never NLopt's. */
        if (typeid(ex) != typeid(std::runtime_error))
            return NloptOutcome::Failed;
        const std::string what = ex.what();
        if (what == "nlopt failure" ||
            (errmsg && errmsg[0] != '\0' && what == errmsg))
            return NloptOutcome::Graceful;
        return NloptOutcome::Failed;
    } catch (...) {
        return NloptOutcome::Failed;
    }
}

namespace detail {

/* The nlopt::opt of `sp` and of its INNER (one level), for make_nlopt,
 * which has checked the spec (validate_nlopt_spec: MAXEVAL fits NLopt's
 * int). It leaves NLopt's random stream alone. */
inline nlopt::opt build_nlopt(const OptimizerSpec& sp, unsigned dim,
                              const StepPlan* plan)
{
    nlopt::opt o(sp.algo.c_str(), dim);

    /* stopping criteria */
    if (sp.stop.maxeval)   o.set_maxeval(static_cast<int>(*sp.stop.maxeval));
    if (sp.stop.maxtime)   o.set_maxtime(*sp.stop.maxtime);
    if (sp.stop.ftol_rel)  o.set_ftol_rel(*sp.stop.ftol_rel);
    if (sp.stop.ftol_abs)  o.set_ftol_abs(*sp.stop.ftol_abs);
    if (sp.stop.xtol_rel)  o.set_xtol_rel(*sp.stop.xtol_rel);
    /* XTOL_ABS as a per-dimension vector (NLopt's width-relative x-stop
     * lever; the |x|-relative XTOL_REL test is vacuous next to u = 0).
     * One value per dim today -- in the unit box that is a fraction of
     * every width. */
    if (sp.stop.xtol_abs)
        o.set_xtol_abs(std::vector<double>(dim, *sp.stop.xtol_abs));
    if (sp.stop.stopval)   o.set_stopval(*sp.stop.stopval);

    /* algorithm-specific parameters */
    if (sp.vector_storage) o.set_vector_storage(*sp.vector_storage);
    if (sp.pop) o.set_population(*sp.pop);
    /* STEP is a fraction of each dimension's internal width; the plan
     * carries the per-dim vector (see initial_step.h). Never the old
     * uniform scalar: identical dx on every dim made BOBYQA/COBYLA's
     * internal rescaling a no-op and starved narrow dims. */
    if (plan && plan->given) o.set_initial_step(plan->dx);

    /* generic param lines */
    for (auto& kv : sp.param) {
        o.set_param(kv.first.c_str(), kv.second);
      }

    /* inner sub-optimizer (one level), for MLSL/AUGLAG-family globals.
     * Its own STEP (if any) becomes a plan without a start point; when
     * the OUTER block also sets STEP, NLopt overwrites the local dx with
     * the outer vector at optimize time (the validator warns). */
    if (sp.inner) {
        StepPlan inner_plan;
        const StepPlan* ip = nullptr;
        if (sp.inner->step || !sp.inner->step_overrides.empty()) {
            inner_plan = make_step_plan(*sp.inner, {}, optimizer_caps(*sp.inner));
            ip = &inner_plan;
        }
        o.set_local_optimizer( build_nlopt(*sp.inner, dim, ip) );
    }

    return o;
}

/* The step plan of the local searches of `sp`, an MLSL or AUGLAG
 * (OptimizerCaps::accepts_inner) whose local search (inner_run_spec: the
 * INNER block's, or NLopt's default LN_COBYLA of GN_MLSL and LN_AUGLAG)
 * keeps the first step of each dimension as its scale for the whole search
 * (keeps_first_step: LN_BOBYQA, LN_COBYLA); nullopt for any other spec.
 * NLopt gives a local search the step of the enclosing block when that
 * block has one (optimize.c copies it into the local optimiser), else the
 * local optimiser's own, the INNER block's STEP -- and with neither, the
 * local search's own default from its own start, which shrinks a step to
 * 0.75 of the distance to a bound closer than that, a scale the search
 * then keeps. So the plan is that of the local search with the STEP NLopt
 * gives it (the enclosing block's STEP and overrides when it sets any,
 * else the INNER block's), and its full step when neither sets one
 * (make_step_plan): from `start` for an AUGLAG, whose one local search
 * starts there (single_local_search), so an LN_BOBYQA start within a step
 * of a bound moves by at most half a step (StepPlan::start), as a search
 * of its own does; without a start for an MLSL, whose local searches start
 * where it picks -- LN_BOBYQA moves such a start within a step of a bound
 * itself, by up to a step. run_nlopt_impl gives the plan's step to the
 * enclosing optimiser, which hands it to its local search. */
inline std::optional<StepPlan> local_step_plan(const OptimizerSpec& sp,
                                               const std::vector<double>& start)
{
    if (!optimizer_caps(sp).accepts_inner) return std::nullopt;
    const std::optional<OptimizerSpec> local = inner_run_spec(sp);
    if (!local) return std::nullopt;
    const OptimizerCaps& local_caps = optimizer_caps(*local);
    if (!keeps_first_step(local_caps)) return std::nullopt;
    OptimizerSpec stepped = *local;
    stepped.low = sp.low;
    stepped.high = sp.high;
    stepped.space = sp.space;
    stepped.parameter_labels = sp.parameter_labels;
    if (sp.step || !sp.step_overrides.empty()) {
        stepped.step = sp.step;
        stepped.step_overrides = sp.step_overrides;
    }
    const bool one_search = single_local_search(sp).has_value();
    return make_step_plan(stepped,
                          one_search ? start : std::vector<double>{},
                          local_caps);
}

/* The evaluations without a gradient that NLopt's AUGLAG makes after its
 * local search has ended, in a run of `sp` that is one local search
 * (single_local_search, phase_specs.h): auglag.c evaluates the point that
 * search returned once more, when it returned a regular code (HPO sets no
 * constraint, so it then ends), once per AUGLAG level. 0 for any other
 * algorithm. */
inline unsigned long auglag_reevaluations(const OptimizerSpec& sp)
{
    const OptimizerCaps& caps = optimizer_caps(sp);
    if (caps.is_global || !caps.accepts_inner) return 0;
    const std::optional<OptimizerSpec> local = inner_run_spec(sp);
    return local ? 1 + auglag_reevaluations(*local) : 0;
}

/** run_nlopt()'s record of the last gradient request it rejected (a point
 *  without a usable gradient; see its UData). */
struct RejectionRecord {
    bool          last_rejected = false;   ///< the last gradient request was rejected
    unsigned long calls = 0;               ///< objective calls so far
    unsigned long calls_at_rejection = 0;  ///< `calls` at the last rejection
    double        rejected_after = 0.0;    ///< seconds into optimize() at it
};

/* Whether a search that has ended ended because of the rejection `r`
 * records: it made no objective call after it but AUGLAG's
 * `reevaluations` of the point its local search returned
 * (auglag_reevaluations), and neither the MAXEVAL nor the MAXTIME of
 * `local`, the local search the run is (single_local_search), had been
 * reached by then. NLopt reports the MAXTIME of a local LD_* search itself,
 * but AUGLAG reports the ending of its local search as FTOL_REACHED
 * (auglag.c without constraints); that search's NLopt calls are the calls
 * up to the rejection. A call after the rejection -- a line-search point
 * without a gradient (LD_SLSQP), a sample of a global search -- means the
 * search went on past it.
 *
 * `local`'s MAXEVAL is in NLopt calls, the count NLopt hands a local search
 * (nlopt_optimize_limited). Only an INNER block's MAXEVAL can be reached
 * here: the enclosing block's, which inner_run_spec takes when it is the
 * tighter, counts every evaluation in run_nlopt (a call makes one or more),
 * so its budget is spent, and has ended the search (MAXEVAL_REACHED), by
 * the time the calls could reach it. */
inline bool ended_on_rejection(const RejectionRecord& r,
                               unsigned long reevaluations,
                               const std::optional<OptimizerSpec>& local)
{
    if (!r.last_rejected || r.calls - r.calls_at_rejection > reevaluations)
        return false;
    if (!local) return true;
    if (local->stop.maxeval && r.calls_at_rejection >= *local->stop.maxeval)
        return false;
    return !(local->stop.maxtime && r.rejected_after >= *local->stop.maxtime);
}

/* Whether `f`, a loss the objective evaluated but did not give NLopt, meets
 * the STOPVAL of `sp`: a usable loss (loss_is_feasible: not a failure under
 * the run's FAILSCORE) at or below it -- the rule of every HPO backend
 * (anneal::stopval_reached, DLIB). */
inline bool withheld_reaches_stopval(double f, const OptimizerSpec& sp)
{
    return sp.stop.stopval &&
           loss_is_feasible(f, sp.failscore.value_or(
                                   std::numeric_limits<double>::infinity())) &&
           f <= *sp.stop.stopval;
}

}  // namespace detail

/* create an nlopt::opt, with its INNER, from OptimizerSpec ---------
 * `plan` is the per-dimension initial-step plan for THIS spec (nullptr,
 * or a plan that gives no step: NLopt's own per-dim heuristic stays in
 * force) -- for an MLSL or AUGLAG whose local search keeps its first step,
 * the plan of that search (detail::local_step_plan), which NLopt hands it.
 * An INNER block's own STEP becomes a plan without a start point (MLSL
 * picks the local starts).
 *
 * NLopt's stochastic algorithms (CRS, MLSL, ISRES, ESCH, PRAXIS, ...)
 * draw from an internal RNG seeded only by `nlopt::srand`; the C library
 * `srand()` does NOT propagate. The block's SEED seeds it -- the
 * per-start SEED + i of MULTISTART and the PRESTAGE refinement, the
 * STRATEGY-level SEED a PRESTAGE exploration without one takes
 * (OptimizerSpec::seed_from_strategy), and the seeds RESTART derives,
 * arrive here as sp.seed. The INIT block's SEED,
 * which draws the block's INIT points, seeds NLopt only when the block has
 * no SEED (OptimizerSpec::nlopt_seed_from_init(); MULTISTART and RESTART
 * offset it as they offset SEED, and it arrives here so offset). NLopt
 * has one random stream per run: MLSL and AUGLAG run their
 * INNER search inside this run and draw it from the same stream, so the
 * INNER spec has no seed of its own. HPOTargets rejects SEED inside an
 * INNER block at startup (and has no INIT block there); a spec built
 * another way with an INNER seed, or with a MAXEVAL above NLopt's int, is
 * refused here rather than ignored or wrapped (validate_nlopt_spec, which
 * validate_hpo_setup() applies to every phase at startup). `restart` says
 * whether the RESTART stagnation stop bounds the run (require_stop,
 * backend_rules.h): run_nlopt() passes whether RestartPolicy has armed it. */
inline nlopt::opt make_nlopt(const OptimizerSpec& sp, unsigned dim,
                             RestartStop restart,
                             const StepPlan* plan = nullptr)
{
    validate_nlopt_spec(sp, restart);
    nlopt::opt o = detail::build_nlopt(sp, dim, plan);
    /* The seed in the run's stream phase (phase_stream_seed): a PRESTAGE
     * refinement seeds NLopt's generator with a key of its own (it drew the
     * exploration's numbers for the same SEED). */
    if (sp.seed)
        nlopt::srand(static_cast<unsigned long>(
            phase_stream_seed(*sp.seed, sp.stream_phase)));
    else if (sp.init_strategy_spec.seed)
        nlopt::srand(static_cast<unsigned long>(
            phase_stream_seed(*sp.init_strategy_spec.seed, sp.stream_phase)));
    return o;
}

/* main driver -----------------------------------------------------
 * Generic over the host (`double operator()(const Vec&)`; an
 * analytical_gradient member is optional) so unit tests can drive it with
 * a stub; run_nlopt<Vec,Runner> below is the production entry point and
 * passes whether RestartPolicy has armed the stagnation stop (`restart`,
 * require_stop in backend_rules.h). */
template<class Vec,class Runner,class HostT>
void run_nlopt_impl(const OptimizerSpec& sp, HostT& host,
                    RestartStop restart = RestartStop::Off)
{
    const unsigned dim = static_cast<unsigned>(sp.low.size());
    const OptimizerCaps& caps = optimizer_caps(sp);

    /* starting point - either user supplied or random inside bounds */
    std::vector<double> start(dim);
    if (!sp.optim_init.empty())
    {
        auto v = sp.in( sp.optim_init.front() );
        for(size_t i=0;i<dim;++i) start[i]=v[i];
    } else {
        dlib::rand r;                     // no seed: time-based
        for(size_t i=0;i<dim;++i)
            start[i]=r.get_double_in_range(sp.low[i],sp.high[i]);
    }

    /* Per-dimension initial-step plan (fraction of width, overrides,
     * per-dim near-bound rules) -- see initial_step.h. */
    const StepPlan plan = make_step_plan(sp, start, caps);
    /* An MLSL or AUGLAG whose local search keeps its first step (LN_BOBYQA,
     * LN_COBYLA) is given the step of that search, which it hands over
     * (detail::local_step_plan). */
    const std::optional<StepPlan> local_plan =
        detail::local_step_plan(sp, start);
    const StepPlan& given_plan = local_plan ? *local_plan : plan;
    nlopt::opt opt =
        make_nlopt(sp, dim, restart, &given_plan);
    /* BOBYQA starts from a point it keeps with its full step: a start within
     * a step of a bound moves by at most half a step (StepPlan::start) --
     * the start of the search, or of AUGLAG, whose local search starts
     * there. The journal says so per dimension; a CONFIG or WARM start, the
     * user's own point, gets a WARNING (stderr and journal): the search
     * evaluates the moved point, not that one. */
    std::string searcher = sp.lib + ":" + sp.algo;
    if (local_plan)
        searcher += " (its local search " + inner_run_spec(sp)->algo + ")";
    if (given_plan.start.size() == dim && given_plan.start != start) {
        const std::vector<double> real_from = sp.out(start);
        const std::vector<double> real_to = sp.out(given_plan.start);
        std::string seed_origin;
        if (sp.optim_init_origin.size() == sp.optim_init.size() &&
            sp.optim_init_origin.front().is_seed())
            seed_origin = sp.optim_init_origin.front().kind ==
                                  InitOrigin::Kind::Config
                              ? "CONFIG" : "WARM";
        std::string moves;
        for (unsigned i = 0; i < dim; ++i) {
            if (given_plan.start[i] == start[i]) continue;
            const double p = given_plan.start[i];
            const bool onto = p == sp.low[i] || p == sp.high[i];
            const bool lower = onto ? p == sp.low[i]
                                    : p - sp.low[i] < sp.high[i] - p;
            std::ostringstream m;
            m.precision(10);
            m << (i < sp.parameter_labels.size() ? sp.parameter_labels[i]
                                                 : logidx::dim_label(i))
              << " starts at " << real_to[i] << " instead of " << real_from[i]
              << ", "
              << (onto ? "on" : "one step from") << " its "
              << (lower ? "lower" : "upper") << " bound";
            if (!moves.empty()) moves += "; ";
            moves += m.str();
        }
        const std::string why =
            "BOBYQA keeps its full first step only from a start on a bound "
            "or at least one step from it, and a start within half a step of "
            "a bound moves onto it";
        HPOLog::instance().info(searcher + ": " + moves + ": " + why);
        if (!seed_origin.empty()) {
            const std::string w =
                searcher + ": the " + seed_origin +
                " start moves: " + moves + "; the search evaluates the moved "
                "point, not the " + seed_origin + " point";
            std::cerr << "[HPO] WARNING: " << w << "\n";
            HPOLog::instance().warning(w);
        }
        start = given_plan.start;
    }

    /* The *_MLSL_LDS variants explore with a deterministic Sobol
     * low-discrepancy sequence -- nlopt's RNG (seeded in make_nlopt)
     * is never consulted for the sampled points (mlsl.c uses
     * nlopt_sobol_next when lds=1, nlopt_urand only when lds=0), and
     * the sequence's skip count is a fixed function of dim and
     * POPULATION. Every eval after the INIT-chain starting point is
     * therefore identical across runs, with or without SEED. Surface
     * that so neither "why does every run sample the same points?"
     * nor "why doesn't SEED change the sampling?" needs a source
     * dive; the non-LDS twin (e.g. G_MLSL) gives seed-controlled
     * random sampling. */
    if (sp.algo.find("MLSL_LDS") != std::string::npos) {
        const bool seeded = sp.init_strategy_spec.seed.has_value() ||
                            sp.seed.has_value();
        std::ostringstream oss;
        oss << sp.algo << ": exploration points come from a deterministic "
               "Sobol (LDS) sequence and are identical across runs";
        if (seeded)
            oss << "; SEED affects at most the INIT-chain starting point "
                   "(and only when the chain has a RANDOM/LHS/SOBOL entry), "
                   "not the sampled points. Use " <<
                   sp.algo.substr(0, sp.algo.size() - 4)
                << " for seed-controlled random sampling";
        oss << ".";
        if (seeded) {
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
        } else {
            HPOLog::instance().info(oss.str());
        }
    }

    opt.set_lower_bounds(sp.low);
    opt.set_upper_bounds(sp.high);

    /* GradientProvider for gradient-based NLopt algorithms (LD_*).
     * Mirrors DLIB / CERES adapters: NUMERICAL takes finite differences
     * (fd_stencil.h); ANALYTICAL / HYBRID consume the host's
     * analytical_gradient member with finite differences on the
     * unregistered indices. Derivative-free algos (LN_*, GN_*) leave
     * grad empty on each call -- the wrapper skips the gradient path.
     *
     * MAXEVAL counts every evaluation of the run against one budget,
     * `evals_used`, as DLIB:BFGS and the CERES drivers count: the
     * objective's loss, and its gradient's evaluations, which the provider
     * counts in the same counter (each finite-difference point; one for
     * each analytical gradient of ANALYTICAL and HYBRID). NLopt counts its
     * objective calls, and a call with a gradient makes several
     * evaluations: its own count let a gradient search run on after the
     * budget, with every gradient refused (GradientFailure::Budget) and +inf
     * for points that had been evaluated (LD_MMA: 120 evaluations for
     * MAXEVAL 60), and GN_CRS2_LM, GN_ORIG_DIRECT(_L) and AUGLAG test it
     * only after an evaluation or an iteration more. The objective evaluates
     * nothing once the budget is spent (UData::budget_spent), and that ends
     * the search MAXEVAL_REACHED after exactly MAXEVAL evaluations. NLopt
     * keeps its own MAXEVAL (make_nlopt): its count passes the evaluations
     * only by the repeats answered without one (UData::repeats), so it ends
     * a search before the budget only when those are counted -- a
     * derivative-free search then ends after MAXEVAL calls, fewer of them
     * evaluations, and the journal's MAXEVAL line says so -- and it is what
     * ends GD_STOGO, which ignores a forced stop and whose later calls the
     * objective refuses without evaluating, a few microseconds each;
     * GD_STOGO therefore needs MAXEVAL (require_stop, backend_rules.h), as
     * MAXTIME would end those calls only after the rest of its time. An
     * INNER block's MAXEVAL still bounds each local search of MLSL or AUGLAG
     * in NLopt's count, its objective calls: the objective cannot tell where
     * one local search ends and the next begins (the journal's "Inner stop
     * criteria" say so). */
    std::atomic<unsigned long> evals_used{0};
    const unsigned long maxeval =
        sp.stop.maxeval.has_value()
            ? static_cast<unsigned long>(*sp.stop.maxeval)
            : 0ul;

    AnalyticalGradFn<Vec> agf = host_analytical_grad_fn<Vec>(sp, host);
    auto gp = make_gradient_provider<Vec, Runner>(
        sp, host, &evals_used, maxeval, agf);

    /* Userdata for the NLopt callback. The GradientProvider must
     * outlive opt.optimize(); it does, as a stack local in this
     * function.
     *
     * `error` keeps the first exception the objective raised. NLopt's C++
     * wrapper would otherwise catch it itself, record NLOPT_FAILURE and
     * rethrow a bare "nlopt failure" -- which this adapter has to treat as
     * the algorithm's own self-termination (below), so a signal stop
     * (core::ShutdownRequested), the RESTART stagnation stop and a fatal
     * best-potential persistence failure (durable_best::Error) all ended
     * the run as "algorithm converged". The objective catches everything,
     * stops the optimisation (force_stop also reaches an MLSL/AUGLAG INNER
     * run, which calls this same objective) and the exception is rethrown
     * with its own type once optimize() has returned -- the pattern of the
     * ASA driver (algorithms_asa.cpp). NLopt calls the objective from one
     * thread at a time, so no lock is needed. */
    /* A point without a usable gradient (GradientProvider::failure:
     * Stencil or NotFinite) is a rejected point: the objective returns
     * +inf and a zero gradient, and NLopt steps back, as its line
     * searches and trust regions do for a failed evaluation. A local
     * gradient search needs a usable gradient at its starting point,
     * though: until it has had one,
     * there is no point to step back to. LD_LBFGS, LD_TNEWTON*, LD_VAR*
     * and LD_SLSQP stopped at once and reported success or
     * "roundoff-limited"; LD_MMA and LD_CCSAQ asked for the same point again
     * and again until MAXEVAL. Such a search is stopped at that rejection
     * (force_stop) with its own status: a local LD_* algorithm, and an
     * AUGLAG whose local search is one (single_local_search: HPO sets no
     * constraint, so AUGLAG runs that search once, from the starting point;
     * force_stop reaches it). MLSL and the global GD_* searches are not
     * stopped: a rejected sample or local start leaves them the rest.
     *
     * `rejection` records the last gradient request that was rejected. A
     * search that ends right after such a rejection ended because of it
     * (detail::ended_on_rejection), unless a stop criterion ended it:
     * MAXEVAL (the budget below), MAXTIME or STOPVAL, or the MAXEVAL or
     * MAXTIME of AUGLAG's local search, which AUGLAG does not report.
     *
     * `budget_spent`: MAXEVAL is spent (see evals_used). The objective
     * refuses a call once it is, without evaluating, and a gradient the
     * budget cut short (GradientFailure::Budget), and stops the search.
     *
     * `withheld`: a loss at or below STOPVAL that the objective evaluated
     * but did not give NLopt, because the gradient of its point failed (the
     * budget, or no usable gradient). NLopt tests STOPVAL on the
     * losses it is given, so the search ends there, STOPVAL_REACHED, as it
     * would have; the objective stops it.
     *
     * `error`, `gradient_stop`, `budget_spent` and `withheld` exclude each
     * other: the first one set makes the objective refuse every later call
     * without evaluating.
     *
     * `to_nlopt`: what NLopt is given for a loss (detail::FailedLossStandIn):
     * a usable loss as it is, a failed evaluation's -- FAILSCORE, by default
     * the largest double, which overflowed the models of LN_BOBYQA and
     * LN_COBYLA -- a finite stand-in, and a gradient scaled with it.
     * `stand_in_noted`: the journal has said so (once per search, at its
     * first failed evaluation). The records keep the loss the evaluation
     * reported. */
    struct Withheld {
        double              loss;
        GradientFailureInfo why;   /* the gradient's failure */
    };
    /* `repeats`: the run's record of its usable evaluations, with the
     * gradients this search formed (repeat_points.h). A call at a recorded
     * point -- one this search, or an earlier search of the run, evaluated
     * -- that asks for nothing the record lacks is answered from it,
     * before the budget is tested:
     * it is no evaluation, so HPO's MAXEVAL does not count it, and a spent
     * budget does not refuse it -- NLopt still counts it among its calls
     * (its MAXEVAL), and the RESTART window counts it as one evaluation
     * without a new best. A gradient call at a recorded point without a
     * gradient, under NUMERICAL, takes the recorded loss as the loss at
     * the point (gradient_at) and evaluates the stencil alone, within the
     * budget. LD_MMA and LD_CCSAQ re-solve their subproblem with a larger
     * rho and land on the same clamped corner again; LN_SBPLX and
     * LN_NELDERMEAD reflect onto a bound point again; MLSL's local search
     * asks for the gradient of a sample it evaluated without one. */
    struct UData {
        const OptimizerSpec*               sp;
        HostT*                             host;
        GradientProvider<Vec,Runner>*      gp;
        nlopt::opt*                        opt;
        std::exception_ptr                 error;
        bool                               local_gradient_search;
        bool                               had_finite_gradient = false;
        bool                               gradient_stop = false;
        GradientFailureInfo                rejected;
        detail::RejectionRecord            rejection;
        std::chrono::steady_clock::time_point started;
        std::atomic<unsigned long>*        evals = nullptr;   /* evals_used */
        unsigned long                      maxeval = 0;       /* 0: none */
        bool                               budget_spent = false;
        std::optional<Withheld>            withheld;
        RepeatPoints<Vec>*                 repeats = nullptr;
        detail::FailedLossStandIn*         to_nlopt = nullptr;
        bool                               stand_in_noted = false;
    };
    const std::optional<OptimizerSpec> local = single_local_search(sp);
    const unsigned long reevaluations = detail::auglag_reevaluations(sp);
    RepeatPoints<Vec> repeats(
        sp.lib + ":" + sp.algo, sp, repeat_policy<Vec>(host),
        "a reuse is no evaluation: HPO's MAXEVAL does not count it, but "
        "NLopt counts it among the calls its own MAXEVAL bounds",
        latest_evaluation(host));
    UData udata{&sp, &host, &gp, &opt, nullptr,
                /*local_gradient_search=*/local &&
                    optimizer_caps(*local).is_gradient_based,
                /*had_finite_gradient=*/false, /*gradient_stop=*/false,
                /*rejected=*/{}, /*rejection=*/{}, /*started=*/{},
                /*evals=*/&evals_used, /*maxeval=*/maxeval,
                /*budget_spent=*/false, /*withheld=*/std::nullopt,
                /*repeats=*/&repeats};
    detail::FailedLossStandIn to_nlopt(sp.failscore);
    udata.to_nlopt = &to_nlopt;

    /* objective adapter (handles log-scale automatically). For
     * gradient-based algorithms NLopt passes a writable `grad` vector
     * of size dim; for derivative-free algorithms `grad` is empty. We
     * compute f via host() unless the search is stopping; when grad is
     * non-empty we ALSO compute the gradient via gp.gradient_at(), which
     * handles the Numerical / Analytical / Hybrid dispatch internally. */
    auto wrapper = [](const std::vector<double>& x,
                      std::vector<double>& grad, void* userdata)->double
    {
        auto* u = static_cast<UData*>(userdata);
        /* What NLopt is given for the loss `f` of the point, with the
         * gradient in `grad` scaled to it (see UData::to_nlopt). */
        const auto given = [u, &grad](double f) {
            if (u->to_nlopt->failed(f) && !u->stand_in_noted) {
                u->stand_in_noted = true;
                HPOLog::instance().info(
                    u->sp->lib + ":" + u->sp->algo + ": NLopt is given a "
                    "finite stand-in for each failed evaluation of this "
                    "search (the worst usable loss of the search so far "
                    "plus a margin; before any, FAILSCORE, at most 1e100); "
                    "loss.tadah records the loss the evaluation reported");
            }
            double slope = 1.0;
            const double v = (*u->to_nlopt)(f, slope);
            if (slope != 1.0)
                for (double& gi : grad) gi *= slope;
            return v;
        };
        /* Stopping: an algorithm may still probe a point before it sees
         * the forced stop (GD_STOGO never sees it); never evaluate again. */
        if (u->error || u->gradient_stop || u->budget_spent || u->withheld) {
            u->opt->force_stop();
            if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0);
            return HUGE_VAL;
        }
        try {
            Vec v;
            v.set_size(static_cast<long>(x.size()));
            for(size_t i=0;i<x.size();++i) v(static_cast<long>(i))=x[i];

            /* A point the run has evaluated (see UData::repeats). */
            const auto e = u->repeats->find(v);
            const RepeatUse how =
                e ? u->repeats->use(*e, !grad.empty()) : RepeatUse::Evaluate;
            if (how == RepeatUse::Record) {
                u->repeats->note_reuse(v, *e, how, !grad.empty());
                ++u->rejection.calls;
                if (!grad.empty()) {
                    u->rejection.last_rejected = false;
                    for (size_t i = 0; i < x.size(); ++i)
                        grad[i] = e->gradient(static_cast<long>(i));
                }
                return given(e->loss);
            }
            /* MAXEVAL spent (see evals_used): no evaluation; the search
             * ends. */
            if (u->maxeval > 0 && u->evals->load() >= u->maxeval) {
                u->budget_spent = true;
                u->opt->force_stop();
                if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0);
                return HUGE_VAL;
            }

            ++u->rejection.calls;
            double f = 0.0;
            if (how == RepeatUse::Centre) {
                /* The recorded loss is the loss at v that NUMERICAL's
                 * finite differences take; only the stencil is
                 * evaluated. */
                u->repeats->note_reuse(v, *e, how, true);
                f = e->loss;
            } else {
                if (e) u->repeats->note_evaluated_again(v, *e, !grad.empty());
                /* The loss counts against MAXEVAL, in the counter the
                 * GradientProvider counts the gradient's evaluations in.
                 * Compute f first so the trainer cache is populated at
                 * this x; host.analytical_gradient
                 * consumes the cached fit state. */
                u->evals->fetch_add(1);
                f = (*u->host)( u->sp->out(v) );
                u->repeats->record_loss(v, f, latest_evaluation(*u->host));
            }

            if (!grad.empty()) {
                u->rejection.last_rejected = false;
                Vec g;
                g.set_size(static_cast<long>(x.size()));
                /* f is the loss of exactly v, just evaluated (or, under
                 * NUMERICAL, recorded): NUMERICAL and HYBRID take it as the
                 * loss at the point (gradient_at) and do not evaluate v
                 * again. */
                if (!u->gp->gradient_at(v, f, g)) {
                    /* No gradient here: MAXEVAL spent within it (Budget),
                     * or no usable gradient (see UData). NLopt
                     * is not given f: the objective returns +inf and a
                     * zero gradient. A usable f at or below STOPVAL ends
                     * the search (`withheld`); so does the budget. A
                     * point without a usable gradient is rejected: the
                     * optimizer backs off -- or, at the starting point of a
                     * local gradient search, is stopped. */
                    const GradientFailureInfo& fi = *u->gp->failure;
                    if (detail::withheld_reaches_stopval(f, *u->sp)) {
                        u->withheld = Withheld{f, fi};
                        u->opt->force_stop();
                    } else if (fi.kind == GradientFailure::Budget) {
                        u->budget_spent = true;
                        u->opt->force_stop();
                    } else if (fi.kind == GradientFailure::Stencil ||
                               fi.kind == GradientFailure::NotFinite) {
                        u->rejected = fi;
                        u->rejection.last_rejected = true;
                        u->rejection.calls_at_rejection = u->rejection.calls;
                        u->rejection.rejected_after =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - u->started)
                                .count();
                        if (u->local_gradient_search &&
                            !u->had_finite_gradient) {
                            u->gradient_stop = true;
                            u->opt->force_stop();
                        }
                    }
                    std::fill(grad.begin(), grad.end(), 0.0);
                    return std::numeric_limits<double>::infinity();
                }
                u->had_finite_gradient = true;
                u->repeats->record_gradient(v, g);
                for (size_t i=0;i<x.size();++i)
                    grad[i] = g(static_cast<long>(i));
            }
            return given(f);
        } catch (...) {
            u->error = std::current_exception();
            u->opt->force_stop();
            if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0);
            return HUGE_VAL;
        }
    };
    opt.set_min_objective(wrapper,&udata);

    /* SBPLX/NELDERMEAD's per-dim x-stop guard compares |xstep_j| against
     * xtol_rel*|x_j| (+ xtol_abs_j): relative to the coordinate's
     * MAGNITUDE, so a dim whose optimum sits near its lower bound (u ~ 0
     * in the unit box, ln(value) ~ 0 on a log dim) can only satisfy it
     * through XTOL_ABS. Say so once when only XTOL_REL is set. */
    if (caps.step_use == StepUse::PerDimVector && sp.stop.xtol_rel &&
        !sp.stop.xtol_abs) {
        HPOLog::instance().hint(
            sp.algo + ": XTOL_REL is relative to each coordinate's magnitude; "
            "dims whose optimum sits near their lower bound" +
            std::string(sp.space && sp.space->normalised()
                        ? " (u ~ 0 in the unit box)" : "") +
            " cannot satisfy it -- set XTOL_ABS (a fraction of the OPTIM "
            "width under OPTIM_SCALE AUTO) as well");
    }

    /* Journal the EFFECTIVE per-dimension initial step: the plan's
     * vector when STEP is set, or when the plan gives the step anyway (an
     * algorithm that keeps its first step, or the local search of an MLSL
     * or AUGLAG that does), otherwise NLopt's x-dependent default read
     * back through get_initial_step(x, dx) so the printed number is the
     * number used. Only for algorithms that read an initial step. */
    if (step_use_consumes(caps.step_use)) {
        StepPlan effective = given_plan;
        if (local_plan) {
            effective.source += " (the step of the local search, which " +
                                sp.algo + " hands it)";
        } else if (!plan.given) {
            const bool inner_sets = sp.inner &&
                (sp.inner->step || !sp.inner->step_overrides.empty());
            if (caps.step_use == StepUse::ForwardedToInner && inner_sets) {
                /* The INNER block's own vector is what the local optimiser
                 * runs with (no OUTER STEP to override it). */
                effective = make_step_plan(*sp.inner, {}, optimizer_caps(*sp.inner));
                effective.source = "INNER " + effective.source +
                                   " (forwarded to " + sp.inner->algo + ")";
            } else if (caps.step_use == StepUse::TrustHalfWidth) {
                /* MMA/CCSAQ never consult the derivative-free heuristic:
                 * their default is sigma = 0.5*width per dim. */
                for (unsigned i = 0; i < dim; ++i) {
                    effective.dx[i]   = 0.5 * (sp.high[i] - sp.low[i]);
                    effective.frac[i] = 0.5;
                }
                effective.source = "STEP unset -- " + sp.algo +
                                   " default trust half-width 0.5*width per dim";
            } else {
                std::vector<double> dx(dim, 0.0);
                opt.get_initial_step(start, dx);
                effective.dx = dx;
                for (unsigned i = 0; i < dim; ++i) {
                    const double w = sp.high[i] - sp.low[i];
                    effective.frac[i] = std::fabs(dx[i]) / (w > 0.0 ? w : 1.0);
                }
                if (caps.step_use == StepUse::ForwardedToInner)
                    effective.source += "; shown for the OUTER start point -- each "
                                        "MLSL/AUGLAG local start recomputes it from "
                                        "its own start";
            }
        }
        const std::string report = format_step_report(effective, sp, caps, start);
        HPOLog::instance().info(report);
        std::cout << "[HPO] Initial step (" << sp.lib << ":" << sp.algo << "): "
                  << effective.source << " -- per-dim table in hpo_run.log\n";
        /* A note of a step that differs from the STEP asked for (turned
         * inward, capped) is a WARNING; "start on a bound (kept)", where
         * BOBYQA keeps the start and its full step, stays in the table. */
        for (unsigned i = 0; i < dim; ++i) {
            const std::string& nt = effective.note[i];
            if (nt.find("inward")     != std::string::npos ||
                nt.find("capped")     != std::string::npos) {
                std::ostringstream oss;
                oss << sp.algo << ": "
                    << (i < sp.parameter_labels.size() ? sp.parameter_labels[i]
                                                       : logidx::dim_label(i))
                    << ": " << effective.note[i];
                HPOLog::instance().warning(oss.str());
            }
        }
    }

    const std::string backend = sp.lib + ":" + sp.algo;
    /* The journal and stdout line of every MAXEVAL ending, the budget's
     * (the objective stopped the search) and NLopt's own count's (a
     * derivative-free search, whose calls are its evaluations and its
     * reused repeats). */
    const auto maxeval_ending = [&]() {
        std::ostringstream nfo;
        nfo << "NLopt finished: " << nlopt_result_str(nlopt::MAXEVAL_REACHED)
            << " after " << evals_used.load();
        if (maxeval > 0) nfo << " of MAXEVAL " << maxeval;
        nfo << " evaluation(s)";
        if (repeats.from_record() > 0)
            nfo << "; NLopt counts its calls, and " << repeats.from_record()
                << " of them asked for points the run had evaluated and "
                   "were answered without an evaluation";
        std::cout << "[HPO] " << nfo.str() << "\n";
        HPOLog::instance().info(nfo.str());
    };
    /* The objective stopped the search (see UData; their order does not
     * matter, they exclude each other): HPO's stop at a starting point
     * without a usable gradient, a loss at or below STOPVAL that NLopt was
     * not given, or the spent budget -- however NLopt ended after it. */
    const auto ended_by_objective = [&]() {
        if (udata.gradient_stop) {
            report_gradient_stop(backend, udata.rejected,
                                 "a local gradient search has no step to "
                                 "take from a starting point without a "
                                 "usable gradient");
            return true;
        }
        if (udata.withheld) {
            const std::string st = nlopt_result_str(nlopt::STOPVAL_REACHED);
            const GradientFailureInfo& why = udata.withheld->why;
            std::ostringstream nfo;
            nfo << std::setprecision(10) << "NLopt finished: " << st
                << " after " << evals_used.load()
                << " evaluation(s): the loss " << udata.withheld->loss
                << " is at or below STOPVAL " << *sp.stop.stopval
                << ", and NLopt was not given it: ";
            if (why.kind == GradientFailure::Budget)
                nfo << "its gradient could not be completed within MAXEVAL";
            else
                nfo << "its point has no usable gradient: " << why.cause
                    << (why.detail.empty() ? "" : " (" + why.detail + ")");
            std::cout << "[HPO] " << nfo.str() << "\n";
            HPOLog::instance().info(nfo.str());
            HPOLog::instance().set_status(st);
            return true;
        }
        if (udata.budget_spent) {
            maxeval_ending();
            HPOLog::instance().set_status(
                nlopt_result_str(nlopt::MAXEVAL_REACHED));
            return true;
        }
        return false;
    };
    /* The search ended right after it rejected a point without a usable
     * gradient, without a stop criterion met (`ended`: how NLopt
     * ended). */
    const auto ended_on_rejection = [&](const std::string& ended) {
        if (detail::ended_on_rejection(udata.rejection, reevaluations,
                                       local)) {
            report_gradient_stop(backend, udata.rejected,
                                 "NLopt ended right after it (" + ended +
                                 ")");
            return true;
        }
        return false;
    };

    double fmin=0;
    int result_code = 1;   // success by default
    std::string status_str;
    try {
        udata.started = std::chrono::steady_clock::now();
        nlopt::result res = opt.optimize(start, fmin);
        /* A forced stop normally makes optimize() throw (below); an
         * algorithm that returns a regular code after it -- GD_STOGO, which
         * ignores it -- must not hide the objective's exception, nor the
         * objective's stop. */
        if (udata.error) std::rethrow_exception(udata.error);
        if (ended_by_objective()) return;
        result_code  = static_cast<int>(res);
        status_str   = nlopt_result_str(result_code);
        /* A stop criterion ends the search whatever the last point was
         * (ended_on_rejection reads those of AUGLAG's local search). A
         * tolerance is not one here: a search that ends with XTOL_REACHED
         * or FTOL_REACHED right after a rejection ended on the steps it
         * took back from the rejected point. */
        const bool criterion = res == nlopt::MAXEVAL_REACHED ||
                               res == nlopt::MAXTIME_REACHED ||
                               res == nlopt::STOPVAL_REACHED;
        if (!criterion && ended_on_rejection(status_str)) return;

        /* NLopt sets an internal stop_msg only on failure paths inside
         * an algorithm. A handful of algorithms catch that internal
         * NLOPT_FAILURE and translate it to a "nice-looking" return
         * code at the upper layer:
         *   - LN_SBPLX  (sbplx.c:162,183): swallows nldrmd's
         *     "starting step size led to a simplex that was too small
         *     in dimension N" as NLOPT_XTOL_REACHED. This fires when
         *     the WARM seed sits near optimum and SBPLX's per-
         *     iteration xstep scaling shrinks the step below
         *     ~1e-13*|x|, so the next simplex vertex coincides with
         *     the start under nldrmd's `close()` test. The user then
         *     sees a "successful" XTOL_REACHED that actually did no
         *     useful exploration -- the same symptom as BOBYQA's silent
         *     start-point snap, but post-hoc rather than pre-flight.
         *   - LN_NELDERMEAD: identical close-test path as SBPLX.
         *   - Several others (cobyla/mma/slsqp/isres/...) surface the
         *     same channel for "invalid scaling / population too small".
         * Surface that diagnostic so the failure is not silent. */
        const char* nlopt_msg = opt.get_errmsg();
        if (nlopt_msg && nlopt_msg[0] != '\0') {
            std::ostringstream oss;
            oss << sp.algo << ": NLopt internal diagnostic was set after "
                << evals_used.load() << " evaluation(s) -- '" << nlopt_msg
                << "'. Algorithm reported " << status_str
                << " but the underlying issue was masked. For "
                   "LN_SBPLX/LN_NELDERMEAD this typically means the "
                   "initial step shrank below precision near a near-"
                   "optimum WARM seed; consider raising STEP (a fraction of "
                   "each OPTIM width), setting XTOL_ABS, loosening the "
                   "tightest bound on the seed coord, or switching to "
                   "LD_LBFGS / DLIB:BFGS / LN_BOBYQA.";
            std::cerr << "[HPO] " << oss.str() << "\n";
            HPOLog::instance().warning(oss.str());
            status_str = "STOPPED -- masked " + status_str + ": " + nlopt_msg;
        } else if (res == nlopt::MAXEVAL_REACHED) {
            maxeval_ending();
        } else {
            std::ostringstream nfo;
            nfo << "NLopt finished: " << status_str << "  fmin=" << fmin;
            std::cout << "[HPO] " << nfo.str() << "\n";
            HPOLog::instance().info(nfo.str());
        }
    } catch (const std::invalid_argument& e) {
        /* The objective's own exception comes first: it is why NLopt
         * stopped (see UData). */
        if (udata.error) std::rethrow_exception(udata.error);
        /* NLOPT_INVALID_ARGS: a rejected SETUP (e.g. BOBYQA 'insufficient
         * space between the bounds', 'invalid scaling', 'zero step size').
         * NLopt attaches its stop message to this exception too, so it
         * must be classified BEFORE the message-based graceful test below:
         * nothing was optimised, the run is FAILED. */
        const std::string status = std::string("FAILED -- ") + e.what();
        std::cout << "[HPO] NLopt rejected the setup: " << e.what() << "\n";
        HPOLog::instance().warning(std::string("NLopt invalid argument: ") + e.what());
        HPOLog::instance().set_status(status);
        throw;
    } catch (const std::exception& e) {
        /* The objective's own exception comes first (see UData): a signal
         * stop, the RESTART stagnation stop or a fatal error must reach
         * the policy and hpo_run() with its own type, never be read as
         * the algorithm's self-termination. */
        if (udata.error) std::rethrow_exception(udata.error);
        const std::string what = e.what();
        const bool graceful =
            classify_nlopt_exception(std::current_exception(),
                                     opt.get_errmsg()) ==
            NloptOutcome::Graceful;
        /* The objective's own stop (a forced stop, or an algorithm that
         * terminated itself after it), then an algorithm that terminated
         * itself right after it rejected a point whose gradient is not
         * finite. */
        const bool forced =
            dynamic_cast<const nlopt::forced_stop*>(&e) != nullptr;
        if ((forced || graceful) && ended_by_objective()) return;
        if (graceful && ended_on_rejection(what)) return;
        if (graceful) {
            status_str = "STOPPED -- " + what +
                " (algorithm converged; consider adding ftol_rel / xtol_rel"
                " / maxeval to stop cleanly next time)";
            std::cout << "[HPO] NLopt self-terminated (" << what
                      << ") -- best result preserved.\n";
            HPOLog::instance().warning(what +
                " -- algorithm converged; best result preserved");
        } else {
            status_str = std::string("FAILED -- ") + what;
            std::cout << "[HPO] NLopt threw: " << what << "\n";
            HPOLog::instance().warning(std::string("NLopt exception: ") + what);
        }
        /* Record this run's status; the outer hpo_run() driver will call
         * finish() once after the whole policy (including MULTISTART) ends. */
        HPOLog::instance().set_status(status_str);
        if (!graceful) throw;
        return;
    }

    /* Record this NLopt run's termination reason. Under MULTISTART each
     * start calls set_status() once; the last status wins, and the outer
     * hpo_run() driver writes the journal trailer after all starts have
     * finished. Calling finish() here would close the journal after the
     * first start and silence subsequent ones. */
    HPOLog::instance().set_status(status_str);
}

/* Production entry point used by run_inner(). */
template<class Vec,class Runner>
void run_nlopt(const OptimizerSpec& sp,
               HPO_Host<Vec,Runner>& host)
{
    run_nlopt_impl<Vec,Runner>(sp, host,
                               restart_stop_armed<HPO_Host<Vec,Runner>>());
}

}   // namespace tadah::hpo

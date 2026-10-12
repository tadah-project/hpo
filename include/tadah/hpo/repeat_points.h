#pragma once
/*
 * repeat_points.h -- a search's record of the points it has evaluated, for
 * the points an optimiser asks for again (nlopt_adapter.h, dlib_adapter.h).
 *
 * An optimiser can ask for a point it has asked for before, bit for bit:
 * NLopt's LD_MMA and LD_CCSAQ re-solve their subproblem with a larger rho
 * and land on the same clamped corner (mma.c, the inner iteration), LN_SBPLX
 * and LN_NELDERMEAD reflect onto a bound point again, and DLIB:BFGS's
 * line-search trials are clamped onto a point it evaluated. Each repeat was
 * a full evaluation: a fit, and every PC_LAMMPS script of the loss.
 *
 * Within one search (one optimiser run: a PLAIN run, a MULTISTART start, a
 * PRESTAGE phase, a RESTART round; MLSL's local searches are part of their
 * MLSL run) the adapter records each evaluation that gave a usable loss
 * (loss_is_feasible under the run's FAILSCORE), with the usable gradient
 * formed there if one was, and the run's number of that evaluation. A call
 * at a recorded point -- the same optimiser-internal doubles, bit for bit,
 * detail::same_point's rule: -0 and +0 differ, a NaN matches nothing; a
 * point 1e-13 away is another point -- is answered from the record when the
 * host says that every evaluation of the run is a function of its point
 * (RepeatPolicy::reuse) and the record holds what the call asks for
 * (RepeatUse):
 *   - Record: its loss, and its gradient when the call asks for one and one
 *     was formed there: no evaluation, no row in the per-eval files;
 *   - Centre: a gradient the record lacks, under GRAD_STRATEGY NUMERICAL:
 *     the recorded loss is the loss at the point that the finite
 *     differences take (gradient_at), and only their stencil is evaluated.
 * HPO's MAXEVAL counts evaluations, so it does not count the recorded
 * loss; NLopt counts every call against its own MAXEVAL, a reused one too
 * (the budget note the adapter gives). The RESTART stagnation window counts
 * each reused call as one evaluation without a new best
 * (HPO_Host::count_reused_repeat, through RepeatPolicy::in_window): a
 * search that only asks for points it has makes no evaluation, and its
 * window must still end or relaunch it. Otherwise the point is evaluated
 * again, as before 2026-10-12 every point was:
 *   - the host names why two evaluations of one point can differ
 *     (HPO_Host::repeat_reuse_blocker(): a PC_LAMMPS line, a PC_OBSERVABLE
 *     that keeps state, a --skip-rel filter);
 *   - the earlier evaluation failed (a failure can be transient);
 *   - the call asks for a gradient that was not formed there under
 *     GRAD_STRATEGY ANALYTICAL or HYBRID: HPO_Host::analytical_gradient
 *     differentiates the thread's last evaluation without checking its
 *     point, so such a gradient follows an evaluation of its point.
 * A host type without repeat_reuse_blocker() (the adapters' test stubs) is
 * not tracked: nothing is recorded or journalled, and every point is
 * evaluated, as before.
 * Each search journals its first reuse (INFO, naming the evaluation and the
 * point) and its count of reuses at 10, 100, 1000, ... (a search that only
 * asks again makes no row: LD_MMA held at u = 0 spins until NLopt's own
 * MAXEVAL or the RESTART window ends it); a run journals once why a repeat
 * was evaluated again; DIAGNOSTICS counts both
 * (HPOLog::record_repeat_reused, record_repeat_evaluated).
 */

#include <tadah/core/checkpoint.h>       // core::ShutdownRequested
#include <tadah/core/signal_handler.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pc_fail_grading.h>   // loss_is_feasible

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tadah::hpo {

namespace detail {

template <class H, class = void>
struct has_repeat_reuse_blocker : std::false_type {};
template <class H>
struct has_repeat_reuse_blocker<
    H, std::void_t<decltype(std::declval<const H&>().repeat_reuse_blocker())>>
    : std::true_type {};

template <class H, class = void>
struct has_reuse_window : std::false_type {};
template <class H>
struct has_reuse_window<
    H, std::void_t<decltype(std::declval<const H&>().count_reused_repeat(
           std::optional<std::size_t>{}))>> : std::true_type {};

template <class H, class = void>
struct has_eval_counter : std::false_type {};
template <class H>
struct has_eval_counter<
    H, std::void_t<decltype(std::declval<const H&>().g_iter_value())>>
    : std::true_type {};

}  // namespace detail

/** Whether a search keeps a record of its evaluations and answers a
 *  repeated point from it. */
struct RepeatPolicy {
    bool        tracked = false; ///< the host says whether its evaluations can differ
    bool        reuse = false;   ///< every evaluation is a function of its point
    std::string blocker;         ///< why not, as the host says
    /** Counts a reused call in the RESTART stagnation window (the run's
     *  number of the evaluation it reuses); may throw StagnationStop.
     *  Empty for a host without a window (a test stub). */
    std::function<void(std::optional<std::size_t>)> in_window;
};

/** The policy of a search on `host`: its repeat_reuse_blocker() -- ""
 *  allows a reuse, a reason has every repeat evaluated and counted -- and
 *  its count_reused_repeat(), when it has one. A host without
 *  repeat_reuse_blocker() (a test stub) is not tracked: every point is
 *  evaluated, and nothing is recorded or journalled, as before. */
template <class HostT>
RepeatPolicy repeat_policy(const HostT& host)
{
    RepeatPolicy p;
    if constexpr (detail::has_repeat_reuse_blocker<HostT>::value) {
        p.tracked = true;
        p.blocker = host.repeat_reuse_blocker();
        p.reuse = p.blocker.empty();
    }
    if constexpr (detail::has_reuse_window<HostT>::value)
        p.in_window = [&host](std::optional<std::size_t> of_eval) {
            host.count_reused_repeat(of_eval);
        };
    (void)host;
    return p;
}

/** How a call at a recorded point is answered (RepeatPoints::use). */
enum class RepeatUse {
    Record,   ///< from the record: its loss, and its gradient when asked
    Centre,   ///< its recorded loss is the centre of the gradient's finite
              ///< differences (GRAD_STRATEGY NUMERICAL), which are evaluated
    Evaluate  ///< the point is evaluated again
};

/** How a call that asks `with_gradient` at a point the search has recorded
 *  -- a usable loss, and a gradient formed there when `has_gradient` -- is
 *  answered, when the run allows a reuse (`reuse`, RepeatPolicy::reuse:
 *  every evaluation of the run is a function of its point) or not, under
 *  GRAD_STRATEGY `strategy`: from the record when the run allows it and
 *  the record has what the call asks for; its loss as the centre of the
 *  finite differences when only the gradient is missing and GRAD_STRATEGY
 *  is NUMERICAL (which takes the loss at the point from the caller and
 *  evaluates the stencil alone); else evaluated (ANALYTICAL and HYBRID
 *  differentiate the last evaluation). The one rule of RepeatPoints::use
 *  and of the startup budget warning (gradient_budget_warnings,
 *  backend_rules.h), which counts the evaluations a run spends. */
inline RepeatUse repeat_use(bool reuse, GradStrategy strategy,
                            bool with_gradient, bool has_gradient)
{
    if (!reuse) return RepeatUse::Evaluate;
    if (!with_gradient || has_gradient) return RepeatUse::Record;
    return strategy == GradStrategy::Numerical ? RepeatUse::Centre
                                               : RepeatUse::Evaluate;
}

/** The run's number of the host's latest evaluation (HPO_Host's counter,
 *  the ITERATION column of the per-eval files), for a host that keeps one.
 *  Read right after an evaluation of a serial search. */
template <class HostT>
std::optional<std::size_t> latest_evaluation(const HostT& host)
{
    if constexpr (detail::has_eval_counter<HostT>::value)
        return static_cast<std::size_t>(host.g_iter_value());
    (void)host;
    return std::nullopt;
}

/** The record of one search: its usable evaluations by point (see the file
 *  comment). */
template <class Vec>
class RepeatRecord {
public:
    /** What a search recorded of one point. */
    struct Entry {
        double                     loss = 0.0;     ///< the usable loss of its evaluation
        std::optional<std::size_t> eval;           ///< the run's number of the evaluation
        bool                       has_gradient = false;  ///< a usable gradient was formed there
        Vec                        gradient;       ///< that gradient (internal coordinates)
    };

    /** The entry of `x`, or nullptr. */
    const Entry* find(const Vec& x) const
    {
        const auto k = key(x);
        if (!k) return nullptr;
        const auto it = entries_.find(*k);
        return it == entries_.end() ? nullptr : &it->second;
    }

    /** Records the usable loss of an evaluation of `x`; it replaces the
     *  entry of an earlier evaluation of `x`, gradient included. */
    void record_loss(const Vec& x, double loss, std::optional<std::size_t> eval)
    {
        const auto k = key(x);
        if (!k) return;
        Entry& e = entries_[*k];
        e.loss = loss;
        e.eval = eval;
        e.has_gradient = false;
        e.gradient = Vec();
    }

    /** Records the usable gradient formed at `x` after its loss
     *  (record_loss); nothing when `x` has no entry. */
    void record_gradient(const Vec& x, const Vec& g)
    {
        const auto k = key(x);
        if (!k) return;
        const auto it = entries_.find(*k);
        if (it == entries_.end()) return;
        it->second.has_gradient = true;
        it->second.gradient = g;
    }

private:
    /* The bit patterns of x; none when a coordinate is a NaN (by its bits:
     * the build's -ffast-math may fold a NaN test). */
    static std::optional<std::vector<std::uint64_t>> key(const Vec& x)
    {
        constexpr std::uint64_t kExp  = 0x7ff0000000000000ull;
        constexpr std::uint64_t kFrac = 0x000fffffffffffffull;
        std::vector<std::uint64_t> k(static_cast<std::size_t>(x.size()));
        for (long i = 0; i < x.size(); ++i) {
            const double v = x(i);
            std::uint64_t b = 0;
            std::memcpy(&b, &v, sizeof v);
            if ((b & kExp) == kExp && (b & kFrac) != 0) return std::nullopt;
            k[static_cast<std::size_t>(i)] = b;
        }
        return k;
    }

    std::map<std::vector<std::uint64_t>, Entry> entries_;
};

/** "A = 1, B = 6": the REAL coordinates of the internal point `x`, with
 *  their labels (dim[i] without one). */
template <class Vec>
std::string describe_point(const OptimizerSpec& sp, const Vec& x)
{
    const auto real = sp.out(x);
    std::ostringstream oss;
    oss.precision(10);
    for (long i = 0; i < real.size(); ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (i) oss << ", ";
        oss << (k < sp.parameter_labels.size() ? sp.parameter_labels[k]
                                               : logidx::dim_label(k))
            << " = " << real(i);
    }
    return oss.str();
}

/** A search's handling of the points asked for again: the record, the
 *  policy, and the journal (see the file comment). One per search. */
template <class Vec>
class RepeatPoints {
public:
    using Entry = typename RepeatRecord<Vec>::Entry;  ///< a recorded point

    /** The handling of the search of `backend` ("NLOPT:LD_MMA") over the
     *  spec `sp` (kept by reference; it labels the journal's points and
     *  gives the FAILSCORE and the GRAD_STRATEGY), under `policy`
     *  (repeat_policy). `budget_note` says how the backend's MAXEVAL takes
     *  a reuse, for the journal's first-reuse line ("a reuse is no
     *  evaluation, and MAXEVAL counts evaluations"). */
    RepeatPoints(std::string backend, const OptimizerSpec& sp,
                 RepeatPolicy policy, std::string budget_note)
        : backend_(std::move(backend)), sp_(sp), policy_(std::move(policy)),
          budget_note_(std::move(budget_note)),
          strategy_(sp.grad_strategy.value_or(GradStrategy::Numerical))
    {}

    /** The entry of `x`; always nullptr when the host is not tracked. */
    const Entry* find(const Vec& x) const
    {
        return policy_.tracked ? record_.find(x) : nullptr;
    }

    /** How a call at `x`, whose entry is `e`, that asks `with_gradient` is
     *  answered: from the record when every evaluation is a function of
     *  its point and the record has what the call asks for; its loss as the
     *  centre of the finite differences when only the gradient is missing
     *  and GRAD_STRATEGY is NUMERICAL (which takes the loss at the point
     *  from the caller and evaluates the stencil alone); else evaluated
     *  (repeat_use). */
    RepeatUse use(const Entry& e, bool with_gradient) const
    {
        return repeat_use(policy_.reuse, strategy_, with_gradient,
                          e.has_gradient);
    }

    /** Counts and journals a call at `x` answered from `e` (`how` Record or
     *  Centre). It checks the stop flag as the host does (a signal stops a
     *  search that only repeats), and counts the call in the RESTART
     *  window (RepeatPolicy::in_window), which may throw StagnationStop. */
    void note_reuse(const Vec& x, const Entry& e, RepeatUse how,
                    bool with_gradient)
    {
        if (core::SignalHandler::stop_requested())
            throw core::ShutdownRequested{};
        ++reused_;
        if (how == RepeatUse::Record) ++from_record_;
        HPOLog::instance().record_repeat_reused();
        /* A search that only asks again for points it has (LD_MMA held
         * at u = 0, where NLopt's XTOL_REL test is vacuous) makes no
         * row and no evaluation: say how many at 10, 100, 1000, ... so
         * that it does not look hung. */
        if (reused_ >= 10 && is_power_of_ten(reused_))
            HPOLog::instance().info(
                backend_ + ": " + std::to_string(reused_) +
                " calls of this search have asked again for points it "
                "evaluated (the latest the point of " + eval_of(e) +
                "), answered without an evaluation");
        if (!noted_reuse_) {
            noted_reuse_ = true;
            std::string what;
            if (how == RepeatUse::Centre)
                what = " and its gradient: the loss of " + eval_of(e) +
                       " is reused, without an evaluation, as the loss at "
                       "the point that the gradient's finite differences "
                       "take, and their stencil is evaluated";
            else if (with_gradient)
                what = " and its gradient: the loss and gradient of " +
                       eval_of(e) + " are reused, without an evaluation";
            else
                what = ": the loss of " + eval_of(e) +
                       " is reused, without an evaluation";
            HPOLog::instance().info(
                backend_ + " asked again for the point of " + eval_of(e) +
                " (" + describe_point(sp_, x) + ")" + what + " (" +
                budget_note_ + "; DIAGNOSTICS counts reuses)");
        }
        if (policy_.in_window) policy_.in_window(e.eval);
    }

    /** Counts and journals a call at `x`, whose entry is `e`, that is
     *  evaluated again (`use` Evaluate), and why. */
    void note_evaluated_again(const Vec& x, const Entry& e, bool with_gradient)
    {
        const std::string why =
            !policy_.reuse
                ? policy_.blocker
                : std::string("a gradient was asked for where the earlier "
                              "evaluation formed none, and GRAD_STRATEGY ") +
                      grad_strategy_name(strategy_) +
                      " differentiates the last evaluation: the point is "
                      "evaluated before its gradient";
        HPOLog::instance().record_repeat_evaluated(why);
        if (HPOLog::instance().first_in_run("repeat points evaluated again"))
            HPOLog::instance().info(
                backend_ + " asked again for the point of " + eval_of(e) +
                " (" + describe_point(sp_, x) + ")" +
                (with_gradient ? " and its gradient" : "") +
                "; it is evaluated again: " + why);
    }

    /** Records an evaluation of `x` (its loss `loss`, its run number
     *  `eval`) when its loss is usable. */
    void record_loss(const Vec& x, double loss, std::optional<std::size_t> eval)
    {
        if (policy_.tracked && loss_is_feasible(loss, failscore()))
            record_.record_loss(x, loss, eval);
    }

    /** Records the usable gradient formed at `x` after its loss. */
    void record_gradient(const Vec& x, const Vec& g)
    {
        if (policy_.tracked) record_.record_gradient(x, g);
    }

    /** The calls of this search answered from the record (RepeatUse
     *  Record or Centre). */
    std::size_t reused() const { return reused_; }

    /** Those answered without an evaluation (RepeatUse Record). */
    std::size_t from_record() const { return from_record_; }

private:
    double failscore() const
    {
        return sp_.failscore.value_or(std::numeric_limits<double>::infinity());
    }
    static bool is_power_of_ten(std::size_t n)
    {
        while (n >= 10 && n % 10 == 0) n /= 10;
        return n == 1;
    }
    static std::string eval_of(const Entry& e)
    {
        return e.eval ? "eval " + std::to_string(*e.eval)
                      : std::string("an earlier evaluation");
    }

    std::string        backend_;
    const OptimizerSpec& sp_;
    RepeatPolicy       policy_;
    std::string        budget_note_;
    GradStrategy       strategy_;
    RepeatRecord<Vec>  record_;
    std::size_t        reused_ = 0;
    std::size_t        from_record_ = 0;
    bool               noted_reuse_ = false;
};

}  // namespace tadah::hpo

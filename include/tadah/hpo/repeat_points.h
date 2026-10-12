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
 * point 1e-13 away is another point -- that asks for nothing the record
 * lacks (its loss; and its gradient, when the call asks for one) is
 * answered from the record when the host says that every evaluation of the
 * run is a function of its point (RepeatPolicy::reuse): no evaluation, no
 * row in the per-eval files, not counted in MAXEVAL (which counts
 * evaluations) nor in the RESTART stagnation window. Otherwise the point is
 * evaluated again, as before 2026-10-12 every point was:
 *   - the host names why two evaluations of one point can differ
 *     (HPO_Host::repeat_reuse_blocker(): a PC_LAMMPS line, a PC_OBSERVABLE
 *     that keeps state, a --skip-rel filter);
 *   - the earlier evaluation failed (a failure can be transient);
 *   - the call asks for a gradient that was not formed there:
 *     HPO_Host::analytical_gradient differentiates the thread's last
 *     evaluation without checking its point, so a gradient follows an
 *     evaluation of its point.
 * A host type without repeat_reuse_blocker() (the adapters' test stubs) is
 * not tracked: nothing is recorded or journalled, and every point is
 * evaluated, as before.
 * Each search journals its first reuse (INFO, naming the evaluation and the
 * point) and its count of reuses at 10, 100, 1000, ... (a search that only
 * asks again makes no row: LD_MMA held at u = 0 spins until NLopt's own
 * MAXEVAL); a run journals once why a repeat was evaluated again;
 * DIAGNOSTICS counts both (HPOLog::record_repeat_reused,
 * record_repeat_evaluated).
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
};

/** The policy of a search on `host`: its repeat_reuse_blocker() -- ""
 *  allows a reuse, a reason has every repeat evaluated and counted. A host
 *  without the method (a test stub) is not tracked: every point is
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
    (void)host;
    return p;
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
     *  gives the FAILSCORE), under `policy` (repeat_policy). */
    RepeatPoints(std::string backend, const OptimizerSpec& sp,
                 RepeatPolicy policy)
        : backend_(std::move(backend)), sp_(sp), policy_(std::move(policy))
    {}

    /** The entry of `x`; always nullptr when the host is not tracked. */
    const Entry* find(const Vec& x) const
    {
        return policy_.tracked ? record_.find(x) : nullptr;
    }

    /** Whether a call at `x` that asks `with_gradient` is answered from
     *  `e`, the entry of `x`: the journal and the counts say what happened.
     *  A reuse checks the stop flag as the host does (a signal stops a
     *  search that only repeats). */
    bool answer(const Vec& x, const Entry& e, bool with_gradient)
    {
        const bool lacks_gradient = with_gradient && !e.has_gradient;
        if (policy_.reuse && !lacks_gradient) {
            if (core::SignalHandler::stop_requested())
                throw core::ShutdownRequested{};
            ++reused_;
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
                HPOLog::instance().info(
                    backend_ + " asked again for the point of " + eval_of(e) +
                    " (" + describe_point(sp_, x) + ")" +
                    (with_gradient ? " and its gradient: the loss and gradient of " +
                                         eval_of(e) + " are reused"
                                   : ": the loss of " + eval_of(e) +
                                         " is reused") +
                    ", without an evaluation (a reuse is no evaluation, and "
                    "MAXEVAL counts evaluations; DIAGNOSTICS counts "
                    "reuses)");
            }
            return true;
        }
        const std::string why =
            !policy_.reuse
                ? policy_.blocker
                : std::string("a gradient was asked for where the earlier "
                              "evaluation formed none: the point is evaluated "
                              "before its gradient");
        HPOLog::instance().record_repeat_evaluated(why);
        if (HPOLog::instance().first_in_run("repeat points evaluated again"))
            HPOLog::instance().info(
                backend_ + " asked again for the point of " + eval_of(e) +
                " (" + describe_point(sp_, x) + ")" +
                (with_gradient ? " and its gradient" : "") +
                "; it is evaluated again: " + why);
        return false;
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

    /** The calls of this search answered from the record. */
    std::size_t reused() const { return reused_; }

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
    RepeatRecord<Vec>  record_;
    std::size_t        reused_ = 0;
    bool               noted_reuse_ = false;
};

}  // namespace tadah::hpo

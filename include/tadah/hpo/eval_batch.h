#pragma once
/*
 * eval_batch.h -- the evaluations an optimiser runs at once, numbered and
 * committed in the order it proposes them.
 *
 * TADAH:ANNEAL_PT, TADAH:ANNEAL_PA and ASA:ASA with a block of states
 * evaluate several points at once on their workers (anneal_parallel.h): a
 * replica, walker or block state is a slot, and each slot makes a sequence
 * of evaluations that depends on its own results only. A fan-out of the
 * slots is a batch:
 *
 *   - Numbers. The batch declares how many evaluations each slot makes at
 *     most (per_slot). When its first evaluation starts, it reserves that
 *     many consecutive evaluation numbers from the host (Numbers), and the
 *     j-th evaluation of slot k takes the number
 *     first + (the evaluations of the slots before k) + j: the evaluations
 *     are numbered in slot order, as one worker running the slots one after
 *     the other numbers them, whatever order they start in.
 *   - Commits. HPO_Host::operator() of an evaluation in a batch does not
 *     commit what follows from its result -- the new best (best_* rows,
 *     best_pot.tadah, BDUMP), the run's record of its points, the PRESTAGE
 *     candidates, the rows of the result files, the journal's census and
 *     BEST lines and the RESTART window -- but parks that commit with the
 *     batch (park). After the join the driver calls commit(), which runs
 *     the commits in number order. No evaluation of the batch sees what
 *     another of the batch commits, and the files a run writes are the same
 *     at any THREADS and in any order of the evaluations.
 *   - A stop. An evaluation that does not run (a signal, MAXTIME, an
 *     error, a slot that ends its sequence early) leaves its number
 *     unused. commit() gives the unused numbers at the end of the batch
 *     back to the host, when the host has reserved no number after them;
 *     the result files go on past any other (Numbers::skip). The journal
 *     says how many evaluations of the batch did not run, and which numbers
 *     have no row, without a cause, which the batch does not know (a
 *     signal, or a slot that reached STOPVAL and ended its own sequence
 *     while the search goes on to the end of the batch): the line of the
 *     run's stop gives it.
 *
 * A worker marks the evaluations of its slot with an EvalBatch::Slot for
 * as long as the slot's task runs; HPO_Host::operator() finds the batch of
 * its thread with current(). Evaluations outside a batch (a serial
 * optimiser, the seed and the calibrations of the annealers, DLIB:GFS,
 * whose dlib::thread_pool proposes each point from the evaluations that
 * have ended) take their number when they start and commit when they end.
 */

#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/run_stop.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah::hpo {

class EvalBatch {
    struct State {
        EvalBatch*  batch = nullptr;
        std::size_t slot  = 0;
        std::size_t taken = 0;
    };
    static State& state()
    {
        static thread_local State s;
        return s;
    }

public:
    /** Where the numbers of a batch come from: the evaluation counter of
     *  the host that evaluates it (HPO_Host). */
    class Numbers {
    public:
        virtual ~Numbers() = default;
        /** Reserves @p n consecutive numbers; returns the first. */
        virtual std::size_t reserve(std::size_t n) = 0;
        /** Gives the numbers [first, end) back when the last number
         *  reserved is end - 1: the next evaluation takes @p first. False,
         *  and nothing changes, otherwise. */
        virtual bool give_back(std::size_t first, std::size_t end) = 0;
        /** Number @p n has no evaluation: the result files go on past it. */
        virtual void skip(std::size_t n) = 0;
    };

    /** The batch @p what ("epoch 3 of TADAH:ANNEAL_PT", the start of its
     *  journal line) whose slot k makes at most @p per_slot[k]
     *  evaluations. */
    EvalBatch(std::string what, std::vector<std::size_t> per_slot)
        : what_(std::move(what)), per_slot_(std::move(per_slot)),
          offset_(per_slot_.size(), 0)
    {
        for (std::size_t k = 0; k < per_slot_.size(); ++k) {
            offset_[k] = total_;
            total_ += per_slot_[k];
        }
        commits_.resize(total_);
    }

    /** Commits what commit() has not (the driver calls commit() after the
     *  join; this is for a batch an exception leaves before): an exception
     *  of a commit is journalled, not thrown. */
    ~EvalBatch()
    {
        if (committed_) return;
        const std::exception_ptr e = commit();
        if (!e) return;
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& x) {
            try {
                HPOLog::instance().warning(what_ + ": " + x.what());
            } catch (...) {}
        } catch (...) {}
    }

    EvalBatch(const EvalBatch&) = delete;
    EvalBatch& operator=(const EvalBatch&) = delete;

    /** The evaluations of slot @p slot of @p batch on the calling thread,
     *  for the lifetime of the object (the task of the slot). */
    class Slot {
    public:
        Slot(EvalBatch& batch, std::size_t slot) : prev_(state())
        {
            if (slot >= batch.per_slot_.size())
                throw std::logic_error(
                    "internal error: " + batch.what_ + " has no slot " +
                    std::to_string(slot + 1));
            state() = State{&batch, slot, 0};
        }
        ~Slot() { state() = prev_; }
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;

    private:
        State prev_;
    };

    /** The batch of the calling thread's slot, or nullptr. */
    static EvalBatch* current() { return state().batch; }

    /** The number of the next evaluation of the calling thread's slot. The
     *  first evaluation of the batch reserves its numbers from @p src.
     *  Throws std::logic_error outside a batch, and for an evaluation more
     *  than its slot was declared. */
    static std::size_t take_number(Numbers& src)
    {
        State& s = state();
        if (!s.batch)
            throw std::logic_error("internal error: an evaluation outside a "
                                   "batch asked for a number of one");
        EvalBatch& b = *s.batch;
        if (s.taken >= b.per_slot_[s.slot])
            throw std::logic_error(
                "internal error: slot " + std::to_string(s.slot + 1) +
                " of " + b.what_ + " makes more evaluations than the " +
                std::to_string(b.per_slot_[s.slot]) + " it was given");
        std::lock_guard<std::mutex> lk(b.m_);
        if (!b.src_) {
            b.src_  = &src;
            b.base_ = src.reserve(b.total_);
        } else if (b.src_ != &src) {
            throw std::logic_error("internal error: two hosts evaluate " +
                                   b.what_);
        }
        return b.base_ + b.offset_[s.slot] + s.taken++;
    }

    /** The commit of evaluation @p number, which commit() runs. */
    void park(std::size_t number, std::function<void()> commit_fn)
    {
        std::lock_guard<std::mutex> lk(m_);
        if (!src_ || number < base_ || number - base_ >= total_)
            throw std::logic_error("internal error: evaluation " +
                                   std::to_string(number) +
                                   " is not one of " + what_);
        commits_[number - base_] = std::move(commit_fn);
    }

    /** After the join: runs the parked commits in number order, gives the
     *  unused numbers at the end back to the host, skips the others and
     *  journals the evaluations that did not run. A commit that throws
     *  (the RESTART window closed: StagnationStop; best_pot.tadah could not
     *  be saved: durable_best::Error) does not stop the commits after it:
     *  returns the exception that decides how the run ends (run_stop.h),
     *  nullptr when none threw. Runs once. */
    std::exception_ptr commit() noexcept
    {
        if (committed_) return nullptr;
        committed_ = true;
        if (!src_) return nullptr;   // no evaluation started
        ExceptionStash stash;
        std::vector<bool> ran(total_, false);
        std::size_t last = 0;        // 1 + the index of the last commit
        for (std::size_t i = 0; i < total_; ++i) {
            if (!commits_[i]) continue;
            ran[i] = true;
            last = i + 1;
            std::function<void()> fn = std::move(commits_[i]);
            commits_[i] = nullptr;
            try {
                fn();
            } catch (...) {
                stash.offer(std::current_exception());
            }
        }
        try {
            note_unused(ran, last);
        } catch (...) {
            stash.offer(std::current_exception());
        }
        return stash.take();
    }

    /** The batch's description. */
    const std::string& what() const { return what_; }

private:
    /* The numbers below index `last` without a commit are skipped; those
     * from `last` on are given back, or skipped when the host has reserved
     * numbers after them. */
    void note_unused(const std::vector<bool>& ran, std::size_t last)
    {
        std::vector<std::size_t> gaps;
        for (std::size_t i = 0; i < last; ++i)
            if (!ran[i]) gaps.push_back(base_ + i);
        const std::size_t not_run = gaps.size() + (total_ - last);
        if (last < total_ && !src_->give_back(base_ + last, base_ + total_))
            for (std::size_t i = last; i < total_; ++i)
                gaps.push_back(base_ + i);
        for (const std::size_t n : gaps) src_->skip(n);
        if (not_run == 0) return;
        std::string line = what_ + ": " + std::to_string(not_run) +
                           " of its " + std::to_string(total_) +
                           (total_ == 1 ? " evaluation" : " evaluations") +
                           " did not run";
        if (!gaps.empty())
            line += "; " + numbers_text(gaps) +
                    (gaps.size() == 1 ? " has no row" : " have no rows");
        HPOLog::instance().info(line);
    }

    /* "evaluation 11", "evaluations 11 and 13", "evaluations 11-14, 19
     * and 21": consecutive numbers as a range. */
    static std::string numbers_text(const std::vector<std::size_t>& v)
    {
        std::vector<std::string> parts;
        for (std::size_t i = 0; i < v.size();) {
            std::size_t j = i;
            while (j + 1 < v.size() && v[j + 1] == v[j] + 1) ++j;
            parts.push_back(j == i ? std::to_string(v[i])
                                   : std::to_string(v[i]) + "-" +
                                         std::to_string(v[j]));
            i = j + 1;
        }
        std::string s = v.size() == 1 ? "evaluation " : "evaluations ";
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) s += i + 1 == parts.size() ? " and " : ", ";
            s += parts[i];
        }
        return s;
    }

    std::string                         what_;
    std::vector<std::size_t>            per_slot_;
    std::vector<std::size_t>            offset_;
    std::size_t                         total_ = 0;
    std::mutex                          m_;
    Numbers*                            src_  = nullptr;
    std::size_t                         base_ = 0;
    std::vector<std::function<void()>>  commits_;
    bool                                committed_ = false;
};

}  // namespace tadah::hpo

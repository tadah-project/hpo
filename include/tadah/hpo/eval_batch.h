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
 *   - Commits. HPO_Host::operator() of an evaluation in a batch hands what
 *     follows from its result -- the new best (best_* rows,
 *     best_pot.tadah, BDUMP), the run's record of its points, the PRESTAGE
 *     candidates, the rows of the result files, the journal's census and
 *     BEST lines and the RESTART window -- to the batch as its commit
 *     (park). The batch runs the commit as soon as every lower-numbered
 *     evaluation of the batch has ended: the commits run in number order,
 *     as the longest run of ended evaluations from the batch's first
 *     number grows, on the thread whose evaluation (or the end of whose
 *     slot) lets them through, one thread at a time. The rows of a batch
 *     are thus written as its evaluations end, at about the pace of one
 *     slot, and a result is held only until its commit. A commit decides
 *     against the commits before it only, and the files a run writes are
 *     the same at any THREADS and in any order of the evaluations.
 *   - After the batch. What the evaluations of a batch must not read from
 *     each other -- the --skip-rel lowest values (HPO_Host::
 *     finish_evaluation) -- is kept for after the batch (park's `after`):
 *     commit(), which the driver calls after the join, runs it in number
 *     order. An exception of a commit (the RESTART window closed:
 *     StagnationStop; best_pot.tadah could not be saved:
 *     durable_best::Error) does not stop the commits after it; commit()
 *     returns it, and the driver's stop follows the batch.
 *   - A stop. An evaluation that does not run (a signal, MAXTIME, an
 *     error, a slot that ends its sequence early) leaves its number
 *     unused. When a slot ends, its unused numbers stop holding the
 *     commits back: a run of unused numbers is skipped (Numbers::skip; the
 *     result files go on past it) when a later evaluation of the batch is
 *     committed, and commit() gives the unused numbers at the end of the
 *     batch back to the host, when the host has reserved no number after
 *     them, and skips them otherwise. commit() journals how many
 *     evaluations of the batch did not run, and which numbers have no row,
 *     without a cause, which the batch does not know (a signal, or a slot
 *     that reached STOPVAL and ended its own sequence while the search goes
 *     on to the end of the batch): the line of the run's stop gives it.
 *
 * A worker marks the evaluations of its slot with an EvalBatch::Slot for
 * as long as the slot's task runs, and the end of the Slot ends the slot;
 * HPO_Host::operator() finds the batch of its thread with current().
 * Evaluations outside a batch (a serial optimiser, the seed and the
 * calibrations of the annealers, DLIB:GFS, whose dlib::thread_pool
 * proposes each point from the evaluations that have ended) take their
 * number when they start and commit when they end.
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
    };
    static State& state()
    {
        static thread_local State s;
        return s;
    }

    /* Where a number of the batch stands. */
    enum class Mark : unsigned char {
        Open,       // its evaluation may still run: its slot has not ended
        Parked,     // its evaluation has ended; its commit waits for its turn
        Committed,  // its commit has run
        Unused,     // its slot ended without it
        Skipped     // unused, and the result files go on past it
    };
    struct Entry {
        Mark                  mark = Mark::Open;
        std::function<void()> commit;
        std::function<void()> after;
    };

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
          offset_(per_slot_.size(), 0), taken_(per_slot_.size(), 0),
          ended_(per_slot_.size(), false)
    {
        for (std::size_t k = 0; k < per_slot_.size(); ++k) {
            offset_[k] = total_;
            total_ += per_slot_[k];
        }
        entries_.resize(total_);
    }

    /** Finishes what commit() has not (the driver calls commit() after the
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
     *  for the lifetime of the object (the task of the slot). Its end ends
     *  the slot: the numbers the slot has not used no longer hold the
     *  commits of the later evaluations of the batch back. A slot runs
     *  once. */
    class Slot {
    public:
        Slot(EvalBatch& batch, std::size_t slot)
            : batch_(batch), slot_(slot), prev_(state())
        {
            if (slot >= batch.per_slot_.size())
                throw std::logic_error(
                    "internal error: " + batch.what_ + " has no slot " +
                    std::to_string(slot + 1));
            {
                std::lock_guard<std::mutex> lk(batch.m_);
                if (batch.ended_[slot])
                    throw std::logic_error(
                        "internal error: slot " + std::to_string(slot + 1) +
                        " of " + batch.what_ + " runs a second time");
            }
            state() = State{&batch, slot};
        }
        ~Slot()
        {
            state() = prev_;
            batch_.end_slot(slot_);
        }
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;

    private:
        EvalBatch&  batch_;
        std::size_t slot_;
        State       prev_;
    };

    /** The batch of the calling thread's slot, or nullptr. */
    static EvalBatch* current() { return state().batch; }

    /** The number of the next evaluation of the calling thread's slot. The
     *  first evaluation of the batch reserves its numbers from @p src.
     *  Throws std::logic_error outside a batch, and for an evaluation more
     *  than its slot was declared. */
    static std::size_t take_number(Numbers& src)
    {
        const State& s = state();
        if (!s.batch)
            throw std::logic_error("internal error: an evaluation outside a "
                                   "batch asked for a number of one");
        EvalBatch& b = *s.batch;
        std::lock_guard<std::mutex> lk(b.m_);
        if (b.taken_[s.slot] >= b.per_slot_[s.slot])
            throw std::logic_error(
                "internal error: slot " + std::to_string(s.slot + 1) +
                " of " + b.what_ + " makes more evaluations than the " +
                std::to_string(b.per_slot_[s.slot]) + " it was given");
        if (!b.src_) {
            b.src_  = &src;
            b.base_ = src.reserve(b.total_);
        } else if (b.src_ != &src) {
            throw std::logic_error("internal error: two hosts evaluate " +
                                   b.what_);
        }
        return b.base_ + b.offset_[s.slot] + b.taken_[s.slot]++;
    }

    /** Evaluation @p number has ended: @p commit_fn, its commit, runs as
     *  soon as every lower-numbered evaluation of the batch has ended --
     *  now, on the calling thread, when they all have -- and @p after_fn,
     *  if any, after the batch (commit()). */
    void park(std::size_t number, std::function<void()> commit_fn,
              std::function<void()> after_fn = nullptr)
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!src_ || number < base_ || number - base_ >= total_)
                throw std::logic_error("internal error: evaluation " +
                                       std::to_string(number) +
                                       " is not one of " + what_);
            Entry& e = entries_[number - base_];
            if (e.mark != Mark::Open)
                throw std::logic_error("internal error: evaluation " +
                                       std::to_string(number) + " of " +
                                       what_ + " ended twice or after "
                                       "its slot");
            e.mark   = Mark::Parked;
            e.commit = std::move(commit_fn);
            e.after  = std::move(after_fn);
        }
        advance();
    }

    /** After the join: runs the commits that are left, in number order,
     *  gives the unused numbers at the end back to the host (or skips
     *  them), runs what the evaluations left for after the batch (park's
     *  `after`) in number order and journals the evaluations that did not
     *  run. Returns the exception, of a commit or of what follows it, that
     *  decides how the run ends (run_stop.h), nullptr when none threw.
     *  Runs once. */
    std::exception_ptr commit() noexcept
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (committed_) return nullptr;
            committed_ = true;
            if (!src_) return nullptr;   // no evaluation started
            /* Every slot has ended: a number still open has no
             * evaluation. */
            for (Entry& e : entries_)
                if (e.mark == Mark::Open) e.mark = Mark::Unused;
        }
        advance();
        try {
            note_unused();
        } catch (...) {
            errors_.offer(std::current_exception());
        }
        for (Entry& e : entries_) {
            if (!e.after) continue;
            std::function<void()> fn = std::move(e.after);
            e.after = nullptr;
            try {
                fn();
            } catch (...) {
                errors_.offer(std::current_exception());
            }
        }
        return errors_.take();
    }

    /** The batch's description. */
    const std::string& what() const { return what_; }

private:
    /* Slot @p slot has ended: its numbers still open are unused. */
    void end_slot(std::size_t slot) noexcept
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            ended_[slot] = true;
            for (std::size_t j = 0; j < per_slot_[slot]; ++j) {
                Entry& e = entries_[offset_[slot] + j];
                if (e.mark == Mark::Open) e.mark = Mark::Unused;
            }
        }
        advance();
    }

    /* Runs the commits that the ended evaluations let through, in number
     * order, from the first number neither committed nor skipped: a parked
     * evaluation is committed, and a run of unused numbers is skipped when
     * a parked evaluation follows it. Stops at a number whose evaluation
     * may still run, or at unused numbers that end the batch, which
     * commit() gives back. One thread runs the commits at a time; a thread
     * that finds another running them leaves the ones it parked to it. The
     * commit runs, and its result is released, outside the lock. An
     * exception is kept for commit(). */
    void advance() noexcept
    {
        std::unique_lock<std::mutex> lk(m_);
        if (committing_ || !src_) return;
        committing_ = true;
        while (next_ < total_) {
            Entry& e = entries_[next_];
            if (e.mark == Mark::Parked) {
                std::function<void()> fn = std::move(e.commit);
                e.commit = nullptr;
                e.mark   = Mark::Committed;
                ++next_;
                lk.unlock();
                try {
                    fn();
                } catch (...) {
                    errors_.offer(std::current_exception());
                }
                fn = nullptr;
                lk.lock();
                continue;
            }
            if (e.mark != Mark::Unused) break;   // may still run
            std::size_t j = next_;
            while (j < total_ && entries_[j].mark == Mark::Unused) ++j;
            if (j == total_ || entries_[j].mark != Mark::Parked) break;
            std::vector<std::size_t> gap;
            for (; next_ < j; ++next_) {
                entries_[next_].mark = Mark::Skipped;
                gap.push_back(base_ + next_);
            }
            lk.unlock();
            for (const std::size_t n : gap) {
                try {
                    src_->skip(n);
                } catch (...) {
                    errors_.offer(std::current_exception());
                }
            }
            lk.lock();
        }
        committing_ = false;
    }

    /* After the last commit: the unused numbers at the end are given back,
     * or skipped when the host has reserved numbers after them; the
     * journal says how many evaluations did not run and which numbers
     * have no row. */
    void note_unused()
    {
        if (next_ < total_ && !src_->give_back(base_ + next_, base_ + total_))
            for (std::size_t i = next_; i < total_; ++i) {
                entries_[i].mark = Mark::Skipped;
                src_->skip(base_ + i);
            }
        std::size_t not_run = 0;
        std::vector<std::size_t> gaps;
        for (std::size_t i = 0; i < total_; ++i) {
            if (entries_[i].mark == Mark::Committed) continue;
            ++not_run;
            if (entries_[i].mark == Mark::Skipped) gaps.push_back(base_ + i);
        }
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

    std::string                 what_;
    std::vector<std::size_t>    per_slot_;
    std::vector<std::size_t>    offset_;
    std::size_t                 total_ = 0;
    std::mutex                  m_;
    /* Guarded by m_: */
    std::vector<std::size_t>    taken_;    // the numbers each slot took
    std::vector<bool>           ended_;    // the slots that have ended
    Numbers*                    src_  = nullptr;
    std::size_t                 base_ = 0;
    std::vector<Entry>          entries_;  // one per number, base_ + i
    std::size_t                 next_ = 0; // the first not committed or skipped
    bool                        committing_ = false;
    bool                        committed_  = false;
    /* The exception that decides how the run ends, of the commits and of
     * what follows them (thread-safe). */
    ExceptionStash              errors_;
};

}  // namespace tadah::hpo

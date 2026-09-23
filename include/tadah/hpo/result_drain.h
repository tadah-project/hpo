#pragma once
/*
 * result_drain.h -- the final drain of HPO_Host's ordered result buffer.
 *
 * While a run is going, the result writer (HPO_Host::submit_result) writes
 * queued results strictly in iteration order and stops at a missing
 * iteration: the evaluation that owns it may still be running. At the end
 * of the run (HPO_Host::flush_remaining) nothing is running any more, and
 * stopping there would silently drop every later row. The final drain
 * therefore writes the contiguous rows first and then every row still
 * buffered, in iteration order, with a journal warning that names the
 * rows written out of sequence and the iterations that have no row.
 *
 * Nothing is dropped silently: a row whose write throws is dropped with a
 * warning of its own ("dropping iter=N").
 */

#include <atomic>
#include <cstddef>
#include <exception>
#include <map>
#include <string>
#include <vector>

namespace tadah::hpo {

struct ResultDrainReport {
    std::size_t in_order = 0;    // contiguous from the cursor
    std::size_t stranded = 0;    // behind a missing iteration
    std::size_t late     = 0;    // below the cursor: the writer had skipped
                                 // their iteration (gap-skip) before they came
    std::size_t dropped  = 0;    // the write threw
    std::size_t n_missing = 0;   // iterations with no row, below the last
                                 // stranded one
    std::vector<std::size_t> missing;   // the first few of them
};

namespace detail {
inline std::string list_iterations(const std::vector<std::size_t>& v,
                                   std::size_t total)
{
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i)
        s += (i ? ", " : "") + std::to_string(v[i]);
    if (total > v.size())
        s += " and " + std::to_string(total - v.size()) + " more";
    return s;
}
}  // namespace detail

/* Drains `buf` (iteration -> row) through `write(const Row&)`, advancing
 * `next` (the writer's cursor: the next iteration due). `warn(std::string)`
 * receives the journal warnings. Leaves `buf` empty and `next` past the
 * last row written. */
template<class Row, class WriteFn, class WarnFn>
ResultDrainReport drain_remaining_results(std::map<std::size_t, Row>& buf,
                                          std::atomic<std::size_t>& next,
                                          WriteFn&& write, WarnFn&& warn)
{
    constexpr std::size_t kListed = 10;
    ResultDrainReport rep;

    auto write_one = [&](std::size_t iter, const Row& row) {
        try {
            write(row);
            return true;
        } catch (const std::exception& e) {
            warn("HPO flush_remaining: dropping iter=" + std::to_string(iter) +
                 " after exception: " + e.what());
        } catch (...) {
            warn("HPO flush_remaining: dropping iter=" + std::to_string(iter) +
                 " after unknown exception");
        }
        ++rep.dropped;
        return false;
    };

    /* 1. The contiguous rows, exactly as the in-run writer would. */
    for (auto it = buf.find(next.load()); it != buf.end();
         it = buf.find(next.load())) {
        if (write_one(it->first, it->second)) ++rep.in_order;
        buf.erase(it);
        ++next;
    }
    if (buf.empty()) return rep;

    /* 2. Rows the in-order writer cannot reach. */
    const std::size_t cursor = next.load();   // missing: no row in buf
    const std::size_t last   = buf.rbegin()->first;
    std::vector<std::size_t> late_iters;
    std::size_t n_stranded = 0;
    for (const auto& kv : buf) {
        if (kv.first < cursor) {
            if (late_iters.size() < kListed) late_iters.push_back(kv.first);
        } else {
            ++n_stranded;
        }
    }
    const std::size_t n_late = buf.size() - n_stranded;
    if (n_stranded > 0) {
        for (std::size_t k = cursor; k < last; ++k) {
            if (buf.count(k)) continue;
            if (rep.missing.size() < kListed) rep.missing.push_back(k);
            ++rep.n_missing;
        }
        warn("HPO flush_remaining: " + std::to_string(n_stranded) +
             " result(s) stranded behind missing iteration" +
             (rep.n_missing == 1 ? " " : "s ") +
             detail::list_iterations(rep.missing, rep.n_missing) +
             "; writing them in iteration order (the missing iteration" +
             (rep.n_missing == 1 ? " has" : "s have") + " no row)");
    }
    if (n_late > 0) {
        warn("HPO flush_remaining: " + std::to_string(n_late) +
             " late result(s) for iteration(s) " +
             detail::list_iterations(late_iters, n_late) +
             " arrived after the writer had skipped them; writing them "
             "out of sequence");
    }
    for (const auto& kv : buf) {
        if (!write_one(kv.first, kv.second)) continue;
        if (kv.first < cursor) ++rep.late;
        else                   ++rep.stranded;
    }
    buf.clear();
    if (last + 1 > cursor) next.store(last + 1);
    return rep;
}

}  // namespace tadah::hpo

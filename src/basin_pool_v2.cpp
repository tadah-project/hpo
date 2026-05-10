#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/basins_v2_path.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tadah {
namespace hpo {

const char* lifecycle_name(LifecycleState s)
{
    switch (s) {
        case LifecycleState::ACTIVE:      return "ACTIVE";
        case LifecycleState::PROMISING:   return "PROMISING";
        case LifecycleState::INACTIVE:    return "INACTIVE";
        case LifecycleState::SUSPECT_DUP: return "SUSPECT_DUP";
        case LifecycleState::MERGED:      return "MERGED";
        case LifecycleState::KILLED:      return "KILLED";
    }
    return "UNKNOWN";
}

bool is_archive_state(LifecycleState s)
{
    return s == LifecycleState::MERGED || s == LifecycleState::KILLED;
}

BasinPoolV2::BasinPoolV2(double dedup_dist, std::size_t pool_max,
                         std::size_t archive_max)
    : dedup_dist_(dedup_dist),
      pool_max_   (pool_max    == 0 ? 1u : pool_max),
      archive_max_(archive_max == 0 ? 2u * pool_max_ : archive_max)
{}

void BasinPoolV2::set_normalisation(std::vector<double> low,
                                    std::vector<double> high)
{
    if (low.size() != high.size())
        throw std::runtime_error("BasinPoolV2::set_normalisation: low/high size mismatch");
    low_  = std::move(low);
    high_ = std::move(high);
}

double BasinPoolV2::relative_l2(const std::vector<double>& a,
                                const std::vector<double>& b) const
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double sumsq = 0.0;
    if (low_.size() == a.size() && high_.size() == a.size()) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double range = high_[i] - low_[i];
            const double scale = (std::abs(range) > 0.0) ? range : 1.0;
            const double d = (a[i] - b[i]) / scale;
            sumsq += d * d;
        }
    } else {
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = a[i] - b[i];
            sumsq += d * d;
        }
    }
    return std::sqrt(sumsq);
}

int BasinPoolV2::find_nearest_within_active_(const std::vector<double>& params,
                                             double thresh) const
{
    int best = -1;
    double best_d = thresh;
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        if (is_archive_state(basins_[i].state)) continue;   // tombstones
        const double d = relative_l2(params, basins_[i].params);
        if (d <= best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

int BasinPoolV2::worst_active_index_() const
{
    int wi = -1;
    double wl = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        if (is_archive_state(basins_[i].state)) continue;
        if (basins_[i].loss > wl) { wl = basins_[i].loss; wi = static_cast<int>(i); }
    }
    return wi;
}

void BasinPoolV2::evict_worst_archive_()
{
    /* Among MERGED/KILLED, drop the one with the smallest birth_iteration
     * (oldest). Ties broken by worst loss. */
    int wi = -1;
    int oldest = std::numeric_limits<int>::max();
    double worst_loss = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        if (!is_archive_state(basins_[i].state)) continue;
        const auto& b = basins_[i];
        if (b.birth_iteration < oldest ||
            (b.birth_iteration == oldest && b.loss > worst_loss)) {
            oldest = b.birth_iteration;
            worst_loss = b.loss;
            wi = static_cast<int>(i);
        }
    }
    if (wi >= 0)
        basins_.erase(basins_.begin() + wi);
}

std::size_t BasinPoolV2::active_count() const
{
    std::size_t n = 0;
    for (const auto& b : basins_)
        if (!is_archive_state(b.state)) ++n;
    return n;
}

std::size_t BasinPoolV2::archive_count() const
{
    std::size_t n = 0;
    for (const auto& b : basins_)
        if (is_archive_state(b.state)) ++n;
    return n;
}

std::vector<std::size_t>
BasinPoolV2::indices_in_state(LifecycleState s) const
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < basins_.size(); ++i)
        if (basins_[i].state == s) out.push_back(i);
    return out;
}

std::vector<std::size_t> BasinPoolV2::active_indices() const
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < basins_.size(); ++i)
        if (!is_archive_state(basins_[i].state)) out.push_back(i);
    return out;
}

const BasinV2* BasinPoolV2::by_id(int id) const
{
    for (const auto& b : basins_) if (b.id == id) return &b;
    return nullptr;
}

BasinV2* BasinPoolV2::by_id(int id)
{
    for (auto& b : basins_) if (b.id == id) return &b;
    return nullptr;
}

int BasinPoolV2::index_of_id(int id) const
{
    for (std::size_t i = 0; i < basins_.size(); ++i)
        if (basins_[i].id == id) return static_cast<int>(i);
    return -1;
}

int BasinPoolV2::add_or_merge(std::vector<double> params,
                              double loss, int iteration)
{
    /* (a) merge into nearest ACTIVE-family basin within dedup_dist_. */
    if (dedup_dist_ > 0.0 && !basins_.empty()) {
        const int near = find_nearest_within_active_(params, dedup_dist_);
        if (near >= 0) {
            BasinV2& b = basins_[static_cast<std::size_t>(near)];
            if (loss < b.loss) {
                /* Treat the merge as an improvement event. */
                const double prev_loss = b.loss;
                b.loss = loss;
                b.params = std::move(params);
                b.last_improve_iteration = iteration;
                /* Crude EMA bump: relative improvement. */
                const double rel = (prev_loss != 0.0)
                    ? (prev_loss - loss) / std::abs(prev_loss)
                    : 0.0;
                b.improvement_ema = 0.5 * b.improvement_ema + 0.5 * rel;
                b.evals_since_improvement = 0;
            }
            b.last_iteration = iteration;
            return b.id;
        }
    }

    /* (b) capacity check on ACTIVE-family. */
    if (active_count() >= pool_max_) {
        const int wi = worst_active_index_();
        if (wi >= 0 && basins_[static_cast<std::size_t>(wi)].loss <= loss) {
            return -1;   // candidate worse than every active resident
        }
        if (wi >= 0)
            basins_.erase(basins_.begin() + wi);
    }

    /* (c) insert. */
    BasinV2 b;
    b.id                     = next_id_++;
    b.sibling_root_id        = b.id;
    b.params                 = std::move(params);
    b.loss                   = loss;
    b.state                  = LifecycleState::ACTIVE;
    b.birth_iteration        = iteration;
    b.last_iteration         = iteration;
    b.last_improve_iteration = iteration;
    basins_.push_back(std::move(b));

    /* (d) prune archive if it has grown beyond cap. */
    while (archive_count() > archive_max_)
        evict_worst_archive_();

    return basins_.back().id;
}

void BasinPoolV2::on_refinement_complete(int basin_id,
                                         std::vector<double> params,
                                         double loss,
                                         std::size_t evals_used,
                                         int iteration,
                                         double tol_attempted,
                                         bool optimizer_converged_naturally)
{
    BasinV2* b = by_id(basin_id);
    if (!b) return;
    if (is_archive_state(b->state)) return;

    b->evals_consumed_total += evals_used;
    b->last_iteration        = iteration;

    const double prev_loss = b->loss;
    if (loss < prev_loss) {
        b->loss = loss;
        b->params = std::move(params);
        b->last_improve_iteration = iteration;
        b->evals_since_improvement = 0;
        const double rel = (prev_loss != 0.0)
            ? (prev_loss - loss) / std::abs(prev_loss)
            : 0.0;
        const double per_eval = (evals_used > 0)
            ? rel / static_cast<double>(evals_used)
            : rel;
        b->improvement_ema = 0.5 * b->improvement_ema + 0.5 * per_eval;
    } else {
        /* No improvement — decay EMA and bump stagnation counter. */
        b->improvement_ema = 0.5 * b->improvement_ema;
        b->evals_since_improvement += evals_used;
    }

    /* Tol bookkeeping. */
    if (!b->last_attempted_tol || tol_attempted < *b->last_attempted_tol)
        b->last_attempted_tol = tol_attempted;
    if (optimizer_converged_naturally || loss >= prev_loss) {
        if (!b->last_exhausted_tol || tol_attempted < *b->last_exhausted_tol)
            b->last_exhausted_tol = tol_attempted;
    }
}

void BasinPoolV2::set_state(int basin_id, LifecycleState s, int iteration)
{
    BasinV2* b = by_id(basin_id);
    if (!b) return;
    if (is_archive_state(b->state)) return;       // terminal — refuse out
    b->state = s;
    b->last_iteration = iteration;
    /* When a basin is revived to ACTIVE, reset stagnation counters so
     * its priority gets a fresh shot. */
    if (s == LifecycleState::ACTIVE) {
        b->evals_since_improvement = 0;
        b->verify_attempts = 0;
    }
    /* When a basin enters terminal state, ensure archive eviction
     * still respects the cap. */
    if (is_archive_state(s)) {
        while (archive_count() > archive_max_)
            evict_worst_archive_();
    }
}

void BasinPoolV2::on_verify_attempt(int basin_id, bool improved, int iteration)
{
    BasinV2* b = by_id(basin_id);
    if (!b) return;
    if (b->state != LifecycleState::INACTIVE) return;
    b->last_iteration = iteration;
    b->verify_attempts += 1;
    if (improved) {
        b->verify_passes += 1;
        b->state = LifecycleState::ACTIVE;
        b->evals_since_improvement = 0;
    }
}

void BasinPoolV2::mark_exhausted(int basin_id, double tol)
{
    BasinV2* b = by_id(basin_id);
    if (!b) return;
    if (!b->last_exhausted_tol || tol < *b->last_exhausted_tol)
        b->last_exhausted_tol = tol;
}

void BasinPoolV2::append_path_point(int basin_id, PathPoint p, std::size_t cap)
{
    BasinV2* b = by_id(basin_id);
    if (!b) return;
    if (cap == 0) return;
    b->coarse_path.push_back(std::move(p));
    /* Phase 4 pruning: drop the snapshot whose loss is closest to the
     * mean of its two neighbours (least-informative interior point).
     * Endpoints survive. */
    prune_path(b->coarse_path, cap);
}

/* ---------------- Phase 2: merge logic ------------------------------ */

namespace {
bool is_mergeable_state(LifecycleState s)
{
    return s == LifecycleState::ACTIVE
        || s == LifecycleState::PROMISING
        || s == LifecycleState::INACTIVE
        || s == LifecycleState::SUSPECT_DUP;
}
}   // namespace

std::vector<std::pair<std::size_t, std::size_t>>
BasinPoolV2::find_merge_candidates(double merge_dist,
                                   double merge_loss_threshold) const
{
    std::vector<std::pair<std::size_t, std::size_t>> out;
    if (basins_.size() < 2) return out;

    /* Greedy pass: for each unmatched basin, pair it with its nearest
     * qualifying neighbour. O(N^2) — acceptable at default pool_max=24. */
    std::vector<bool> matched(basins_.size(), false);
    constexpr double eps = 1.0e-30;

    for (std::size_t i = 0; i < basins_.size(); ++i) {
        if (matched[i]) continue;
        if (!is_mergeable_state(basins_[i].state)) continue;

        int best_j = -1;
        double best_d = merge_dist;
        for (std::size_t j = i + 1; j < basins_.size(); ++j) {
            if (matched[j]) continue;
            if (!is_mergeable_state(basins_[j].state)) continue;
            const double d = relative_l2(basins_[i].params, basins_[j].params);
            if (d > merge_dist) continue;
            const double rel_loss =
                std::abs(basins_[i].loss - basins_[j].loss) /
                std::max(std::abs(basins_[i].loss),
                         std::max(std::abs(basins_[j].loss), eps));
            if (rel_loss > merge_loss_threshold) continue;
            if (d < best_d) {
                best_d = d;
                best_j = static_cast<int>(j);
            }
        }
        if (best_j >= 0) {
            out.emplace_back(i, static_cast<std::size_t>(best_j));
            matched[i] = true;
            matched[static_cast<std::size_t>(best_j)] = true;
        }
    }
    return out;
}

bool BasinPoolV2::flag_suspect_pair(int basin_id_a, int basin_id_b, int iteration)
{
    BasinV2* ba = by_id(basin_id_a);
    BasinV2* bb = by_id(basin_id_b);
    if (!ba || !bb) return false;
    if (is_archive_state(ba->state) || is_archive_state(bb->state)) return false;
    bool changed = false;
    if (ba->state != LifecycleState::SUSPECT_DUP) {
        ba->state = LifecycleState::SUSPECT_DUP;
        changed = true;
    }
    if (bb->state != LifecycleState::SUSPECT_DUP) {
        bb->state = LifecycleState::SUSPECT_DUP;
        changed = true;
    }
    if (changed) {
        ba->last_iteration = iteration;
        bb->last_iteration = iteration;
    }
    return changed;
}

void BasinPoolV2::confirm_merge(int loser_id, int winner_id, int iteration)
{
    BasinV2* loser  = by_id(loser_id);
    BasinV2* winner = by_id(winner_id);
    if (!loser || !winner) return;
    if (is_archive_state(loser->state) || is_archive_state(winner->state)) return;
    if (loser_id == winner_id) return;

    /* Roll the loser's bookkeeping into the winner's. */
    winner->evals_consumed_total += loser->evals_consumed_total;
    /* EMA: take the better (max) of the two so survivors don't appear
     * less promising after absorbing a tombstone's history. */
    if (loser->improvement_ema > winner->improvement_ema)
        winner->improvement_ema = loser->improvement_ema;
    /* Move loser to MERGED with sibling_root_id pointing at the winner. */
    loser->state           = LifecycleState::MERGED;
    loser->sibling_root_id = winner_id;
    loser->last_iteration  = iteration;
    /* Winner stays in its current state (typically SUSPECT_DUP →
     * caller chooses next state — usually ACTIVE for re-prioritisation). */
    winner->last_iteration = iteration;

    /* Archive eviction in case we just exceeded archive_max. */
    while (archive_count() > archive_max_)
        evict_worst_archive_();
}

void BasinPoolV2::revert_suspect_pair(int basin_id_a, int basin_id_b, int iteration)
{
    BasinV2* ba = by_id(basin_id_a);
    BasinV2* bb = by_id(basin_id_b);
    if (ba && ba->state == LifecycleState::SUSPECT_DUP) {
        ba->state = LifecycleState::ACTIVE;
        ba->last_iteration = iteration;
    }
    if (bb && bb->state == LifecycleState::SUSPECT_DUP) {
        bb->state = LifecycleState::ACTIVE;
        bb->last_iteration = iteration;
    }
}

/* ---------------- checkpoint ---------------------------------------- */

void BasinPoolV2::save(core::Checkpoint& cp) const
{
    cp.section(tag());
    cp.put_dbl ("dedup_dist",   dedup_dist_);
    cp.put_size("pool_max",     pool_max_);
    cp.put_size("archive_max",  archive_max_);
    cp.put_int ("next_id",      next_id_);
    if (!low_.empty())  cp.put_vec("low",  low_);
    if (!high_.empty()) cp.put_vec("high", high_);
    cp.put_size("n_basins", basins_.size());
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        const auto& b = basins_[i];
        const std::string p = "basin_" + std::to_string(i) + "_";
        cp.put_int  (p + "id",                       b.id);
        cp.put_int  (p + "sibling_root_id",          b.sibling_root_id);
        cp.put_vec  (p + "params",                   b.params);
        cp.put_dbl  (p + "loss",                     b.loss);
        cp.put_int  (p + "state",                    static_cast<int>(b.state));
        cp.put_int  (p + "birth_iteration",          b.birth_iteration);
        cp.put_int  (p + "last_iteration",           b.last_iteration);
        cp.put_int  (p + "last_improve_iteration",   b.last_improve_iteration);
        cp.put_dbl  (p + "improvement_ema",          b.improvement_ema);
        cp.put_size (p + "evals_consumed_total",     b.evals_consumed_total);
        cp.put_size (p + "evals_since_improvement",  b.evals_since_improvement);
        cp.put_dbl  (p + "current_tol_cursor",       b.current_tol_cursor);
        if (b.last_attempted_tol)
            cp.put_dbl(p + "last_attempted_tol",     *b.last_attempted_tol);
        if (b.last_exhausted_tol)
            cp.put_dbl(p + "last_exhausted_tol",     *b.last_exhausted_tol);
        cp.put_int  (p + "verify_attempts",          b.verify_attempts);
        cp.put_int  (p + "verify_passes",            b.verify_passes);
        cp.put_size (p + "n_path",                   b.coarse_path.size());
        for (std::size_t k = 0; k < b.coarse_path.size(); ++k) {
            const std::string pp = p + "path_" + std::to_string(k) + "_";
            cp.put_vec (pp + "params",     b.coarse_path[k].params);
            cp.put_dbl (pp + "loss",       b.coarse_path[k].loss);
            cp.put_size(pp + "eval_index", b.coarse_path[k].eval_index);
        }
    }
}

void BasinPoolV2::load(const core::Checkpoint& cp)
{
    if (auto v = cp.get_dbl (tag(), "dedup_dist"))   dedup_dist_   = *v;
    if (auto v = cp.get_size(tag(), "pool_max"))     pool_max_     = *v;
    if (auto v = cp.get_size(tag(), "archive_max"))  archive_max_  = *v;
    if (auto v = cp.get_int (tag(), "next_id"))      next_id_      = static_cast<int>(*v);
    if (auto v = cp.get_vec (tag(), "low"))          low_          = *v;
    if (auto v = cp.get_vec (tag(), "high"))         high_         = *v;

    basins_.clear();
    const auto n = cp.get_size(tag(), "n_basins").value_or(0);
    basins_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::string p = "basin_" + std::to_string(i) + "_";
        BasinV2 b;
        if (auto v = cp.get_int (tag(), p + "id"))                      b.id = static_cast<int>(*v);
        if (auto v = cp.get_int (tag(), p + "sibling_root_id"))         b.sibling_root_id = static_cast<int>(*v);
        if (auto v = cp.get_vec (tag(), p + "params"))                  b.params = *v;
        if (auto v = cp.get_dbl (tag(), p + "loss"))                    b.loss = *v;
        if (auto v = cp.get_int (tag(), p + "state"))                   b.state = static_cast<LifecycleState>(*v);
        if (auto v = cp.get_int (tag(), p + "birth_iteration"))         b.birth_iteration = static_cast<int>(*v);
        if (auto v = cp.get_int (tag(), p + "last_iteration"))          b.last_iteration = static_cast<int>(*v);
        if (auto v = cp.get_int (tag(), p + "last_improve_iteration"))  b.last_improve_iteration = static_cast<int>(*v);
        if (auto v = cp.get_dbl (tag(), p + "improvement_ema"))         b.improvement_ema = *v;
        if (auto v = cp.get_size(tag(), p + "evals_consumed_total"))    b.evals_consumed_total = *v;
        if (auto v = cp.get_size(tag(), p + "evals_since_improvement")) b.evals_since_improvement = *v;
        if (auto v = cp.get_dbl (tag(), p + "current_tol_cursor"))      b.current_tol_cursor = *v;
        if (auto v = cp.get_dbl (tag(), p + "last_attempted_tol"))      b.last_attempted_tol = *v;
        if (auto v = cp.get_dbl (tag(), p + "last_exhausted_tol"))      b.last_exhausted_tol = *v;
        if (auto v = cp.get_int (tag(), p + "verify_attempts"))         b.verify_attempts = static_cast<int>(*v);
        if (auto v = cp.get_int (tag(), p + "verify_passes"))           b.verify_passes = static_cast<int>(*v);
        const auto npath = cp.get_size(tag(), p + "n_path").value_or(0);
        b.coarse_path.reserve(npath);
        for (std::size_t k = 0; k < npath; ++k) {
            const std::string pp = p + "path_" + std::to_string(k) + "_";
            PathPoint pt;
            if (auto v = cp.get_vec (tag(), pp + "params"))     pt.params = *v;
            if (auto v = cp.get_dbl (tag(), pp + "loss"))       pt.loss = *v;
            if (auto v = cp.get_size(tag(), pp + "eval_index")) pt.eval_index = *v;
            b.coarse_path.push_back(std::move(pt));
        }
        /* next_id_ is loaded above, but defend against a checkpoint
         * written before the field existed (and against a version where
         * next_id_ < observed b.id). */
        if (b.id >= next_id_) next_id_ = b.id + 1;
        basins_.push_back(std::move(b));
    }
}

}   // namespace hpo
}   // namespace tadah

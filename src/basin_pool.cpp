#include <tadah/hpo/basin_pool.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tadah {
namespace hpo {

void check_checkpoint_space(const core::Checkpoint& cp, const std::string& section,
                            const SearchSpacePtr& space, const char* who)
{
    const bool tagged_real = cp.get_bool(section, "params_real").value_or(false);
    if (!tagged_real) {
        /* Pre-transform checkpoint: params were stored in whatever the
         * old code considered "transformed" space (real for v1 top-K
         * inserts, mixed for v2). Unambiguous only when every dim is
         * linear. */
        if (space) {
            for (std::size_t i = 0; i < space->dim(); ++i) {
                if (space->is_nonlinear(i))
                    throw std::runtime_error(std::string(who) +
                        ": checkpoint.tadah predates the search-space transform "
                        "and this run has a LOG/ASINH-scaled dimension (" +
                        space->labels()[i] + "), so the stored basin coordinates "
                        "are ambiguous; delete checkpoint.tadah to start the basin "
                        "pool afresh (best_pot.tadah is unaffected)");
            }
        }
        return;
    }
    if (!space) return;
    if (auto fp = cp.get_int(section, "space_fingerprint")) {
        if (static_cast<std::uint64_t>(*fp) != space->fingerprint_u64())
            throw std::runtime_error(std::string(who) +
                ": checkpoint.tadah was written for a different search space "
                "(OPTIM bounds, scale tags or OPTIM_SCALE changed since); "
                "restore the previous OPTIM block, or delete checkpoint.tadah to "
                "start the basin pool afresh (best_pot.tadah is unaffected)");
    }
}

BasinPool::BasinPool(double dedup_dist, std::size_t pool_max)
    : dedup_dist_(dedup_dist), pool_max_(pool_max == 0 ? 1u : pool_max) {}

void BasinPool::set_space(SearchSpacePtr space)
{
    space_ = std::move(space);
}

void BasinPool::set_normalisation(std::vector<double> low,
                                  std::vector<double> high)
{
    if (low.size() != high.size())
        throw std::runtime_error("BasinPool::set_normalisation: low/high size mismatch");
    low_  = std::move(low);
    high_ = std::move(high);
}

double BasinPool::relative_l2_(const std::vector<double>& a,
                               const std::vector<double>& b) const
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    if (space_ && space_->dim() == a.size()) {
        double sumsq = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double w = space_->internal_width(i);
            const double d = (space_->to_internal(i, a[i]) - space_->to_internal(i, b[i]))
                             / (w > 0.0 ? w : 1.0);
            sumsq += d * d;
        }
        return std::sqrt(sumsq);
    }
    /* If normalisation isn't set, fall back to plain L2 -- at startup
     * the pool may not yet know the bounds. Distance is then measured
     * in raw search space. */
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

int BasinPool::find_nearest_within_(const std::vector<double>& params,
                                    double thresh) const
{
    int best = -1;
    double best_d = thresh;
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        const double d = relative_l2_(params, basins_[i].params);
        if (d <= best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

int BasinPool::worst_index_() const
{
    if (basins_.empty()) return -1;
    int wi = 0;
    double wl = basins_[0].loss;
    for (std::size_t i = 1; i < basins_.size(); ++i)
        if (basins_[i].loss > wl) { wl = basins_[i].loss; wi = static_cast<int>(i); }
    return wi;
}

int BasinPool::add_or_merge(std::vector<double> params, double loss, int round)
{
    /* Try to merge into an existing basin within dedup_dist_. */
    if (dedup_dist_ > 0.0 && !basins_.empty()) {
        const int near = find_nearest_within_(params, dedup_dist_);
        if (near >= 0) {
            Basin& b = basins_[static_cast<std::size_t>(near)];
            if (loss < b.loss) {
                b.loss   = loss;
                b.params = std::move(params);
            }
            b.last_round = std::max(b.last_round, round);
            return near;
        }
    }

    /* Insert. Evict worst if at capacity. */
    if (basins_.size() >= pool_max_) {
        const int wi = worst_index_();
        if (wi >= 0 && basins_[static_cast<std::size_t>(wi)].loss <= loss) {
            /* New candidate is no better than the worst already in
             * the pool -- refuse insertion. */
            return -1;
        }
        if (wi >= 0) basins_.erase(basins_.begin() + wi);
    }

    Basin b;
    b.params      = std::move(params);
    b.loss        = loss;
    b.birth_round = round;
    b.last_round  = round;
    basins_.push_back(std::move(b));
    return static_cast<int>(basins_.size()) - 1;
}

void BasinPool::mark_exhausted(std::size_t i, double tol)
{
    if (i >= basins_.size()) return;
    auto& cur = basins_[i].exhausted_at_tol;
    if (!cur || tol < *cur) cur = tol;
}

void BasinPool::mark_attempted(std::size_t i, double tol)
{
    if (i >= basins_.size()) return;
    auto& cur = basins_[i].last_tol_attempted;
    if (!cur || tol < *cur) cur = tol;
}

void BasinPool::update(std::size_t i, std::vector<double> params,
                       double loss, int round)
{
    if (i >= basins_.size()) return;
    if (loss < basins_[i].loss) {
        basins_[i].loss   = loss;
        basins_[i].params = std::move(params);
    }
    basins_[i].last_round = std::max(basins_[i].last_round, round);
}

std::vector<Basin> BasinPool::heads() const
{
    auto sorted = basins_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Basin& a, const Basin& b){ return a.loss < b.loss; });
    return sorted;
}

std::vector<std::size_t> BasinPool::not_exhausted_at(double tol) const
{
    std::vector<std::size_t> out;
    out.reserve(basins_.size());
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        const auto& e = basins_[i].exhausted_at_tol;
        if (!e || *e > tol) out.push_back(i);
    }
    return out;
}

void BasinPool::save(core::Checkpoint& cp) const
{
    cp.section(tag());
    cp.put_dbl ("dedup_dist", dedup_dist_);
    cp.put_size("pool_max",   pool_max_);
    /* Coordinate contract tag: basin params are REAL-space; the search
     * space they were normalised in is identified by its fingerprint so
     * a resume against changed OPTIM bounds/scales is detected instead
     * of silently normalising with stale widths. */
    cp.put_bool("params_real", true);
    if (space_)
        cp.put_int("space_fingerprint", static_cast<long long>(space_->fingerprint_u64()));
    if (!low_.empty())  cp.put_vec("low",  low_);
    if (!high_.empty()) cp.put_vec("high", high_);
    cp.put_size("n_basins", basins_.size());
    for (std::size_t i = 0; i < basins_.size(); ++i) {
        const auto& b = basins_[i];
        const std::string p = "basin_" + std::to_string(i) + "_";
        cp.put_vec (p + "params",      b.params);
        cp.put_dbl (p + "loss",        b.loss);
        cp.put_int (p + "birth_round", b.birth_round);
        cp.put_int (p + "last_round",  b.last_round);
        if (b.exhausted_at_tol)
            cp.put_dbl(p + "exhausted_at_tol", *b.exhausted_at_tol);
        if (b.last_tol_attempted)
            cp.put_dbl(p + "last_tol_attempted", *b.last_tol_attempted);
    }
}

void BasinPool::load(const core::Checkpoint& cp)
{
    if (auto v = cp.get_dbl (tag(), "dedup_dist")) dedup_dist_ = *v;
    if (auto v = cp.get_size(tag(), "pool_max"))   pool_max_   = *v;
    check_checkpoint_space(cp, tag(), space_, "BasinPool");
    if (!space_) {
        /* legacy normalisation only when no space is bound */
        if (auto v = cp.get_vec (tag(), "low"))        low_  = *v;
        if (auto v = cp.get_vec (tag(), "high"))       high_ = *v;
    }

    basins_.clear();
    const auto n = cp.get_size(tag(), "n_basins").value_or(0);
    basins_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::string p = "basin_" + std::to_string(i) + "_";
        Basin b;
        if (auto v = cp.get_vec (tag(), p + "params"))      b.params = *v;
        if (auto v = cp.get_dbl (tag(), p + "loss"))        b.loss = *v;
        if (auto v = cp.get_int (tag(), p + "birth_round")) b.birth_round = static_cast<int>(*v);
        if (auto v = cp.get_int (tag(), p + "last_round"))  b.last_round = static_cast<int>(*v);
        if (auto v = cp.get_dbl (tag(), p + "exhausted_at_tol"))   b.exhausted_at_tol = *v;
        if (auto v = cp.get_dbl (tag(), p + "last_tol_attempted")) b.last_tol_attempted = *v;
        basins_.push_back(std::move(b));
    }
}

}   // namespace hpo
}   // namespace tadah

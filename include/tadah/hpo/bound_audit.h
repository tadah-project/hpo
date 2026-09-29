#pragma once
/*
 * BoundOccupancy -- generic "is the running best riding an OPTIM bound?"
 * audit, for every backend.
 *
 * Fed from HPO_Host's new-best path with the REAL best point; a dim is
 * marked when its INTERNAL gap to the nearer bound is <= EDGE (fraction
 * of the internal width; default 0.02). Marks that persist across new
 * bests keep their first eval so the journal can say how long the best
 * has sat there. This is an observation, not a proof that the optimum is
 * outside the box (a bound can be the physically right answer); the
 * journal prints the marks as INFO facts and one HINT with the concrete
 * remedy (widen that side, restart WARM). Bounds are never changed by
 * the audit itself. A lower bound that is a limit of the admissible
 * domain (set_hard_lo: AMPGRIDMB at 0 under RHOFORM NONNEG/MONOTONE) is
 * not bound pressure: its marks are flagged hard and get no remedy.
 *
 * Replaces HPO_Host::bound_hints, which compared REAL coordinates with
 * INTERNAL bounds (wrong for any transformed dim) and only fired on the
 * NaN-clamp log path.
 */

#include <tadah/hpo/search_space.h>
#include <tadah/hpo/log_index.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tadah::hpo {

class BoundOccupancy {
public:
    struct Mark {
        std::size_t   dim = 0;
        bool          hi = false;          // which side
        double        gap_u = 0.0;         // internal gap / internal width
        double        real_value = 0.0;    // the best's real coordinate
        double        real_bound = 0.0;    // the bound it sits at
        std::uint64_t since_eval = 0;      // first eval of the persisting mark
        bool          hard = false;        // LO on a hard lower bound (set_hard_lo)
    };

    BoundOccupancy() = default;
    explicit BoundOccupancy(SearchSpacePtr space, double edge = 0.02)
        : space_(std::move(space)), edge_(edge) {}

    bool armed() const noexcept { return static_cast<bool>(space_); }

    /* Per dim: its lower bound is a hard limit of the admissible domain,
     * not a choice of box (widening it is refused at startup). */
    void set_hard_lo(std::vector<bool> hard) { hard_lo_ = std::move(hard); }
    double edge() const noexcept { return edge_; }

    /* Recompute the marks for a new best at `eval` (REAL coordinates).
     * Returns true when the mark set is non-empty. */
    bool on_new_best(std::uint64_t eval, const std::vector<double>& x_real)
    {
        if (!space_ || x_real.size() != space_->dim()) { marks_.clear(); return false; }
        std::vector<Mark> next;
        for (std::size_t i = 0; i < space_->dim(); ++i) {
            const double u  = space_->to_internal(i, x_real[i]);
            const double w  = space_->internal_width(i);
            if (!(w > 0.0)) continue;
            const double glo = (u - space_->internal_lo(i)) / w;
            const double ghi = (space_->internal_hi(i) - u) / w;
            const bool near_lo = glo <= edge_;
            const bool near_hi = ghi <= edge_;
            if (!near_lo && !near_hi) continue;
            Mark m;
            m.dim = i;
            m.hi = near_hi && (!near_lo || ghi < glo);
            m.gap_u = m.hi ? ghi : glo;
            m.real_value = x_real[i];
            m.real_bound = m.hi ? space_->at(i).hi : space_->at(i).lo;
            m.since_eval = eval;
            m.hard = !m.hi && i < hard_lo_.size() && hard_lo_[i];
            for (const auto& old : marks_)
                if (old.dim == i && old.hi == m.hi) { m.since_eval = old.since_eval; break; }
            next.push_back(m);
        }
        marks_ = std::move(next);
        last_eval_ = eval;
        return !marks_.empty();
    }

    const std::vector<Mark>& marks() const noexcept { return marks_; }
    std::uint64_t last_eval() const noexcept { return last_eval_; }
    std::string label(std::size_t i) const
    { return space_ && i < space_->dim() ? space_->labels()[i] : logidx::dim_label(i); }

private:
    SearchSpacePtr    space_;
    double            edge_ = 0.02;
    std::vector<Mark> marks_;
    std::vector<bool> hard_lo_;
    std::uint64_t     last_eval_ = 0;
};

}  // namespace tadah::hpo

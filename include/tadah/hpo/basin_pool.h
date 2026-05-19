#ifndef TADAH_HPO_BASIN_POOL_H
#define TADAH_HPO_BASIN_POOL_H

/*
 *  BasinPool -- persistent, deduplicated population for the BASINS strategy.
 *
 *  A "basin" is a candidate (params, loss) point in the search space.
 *  Each basin remembers when it was born, when it was last refined,
 *  and at what tolerance level it has been exhausted (if any). The
 *  pool deduplicates by relative L2 distance in the bounding box: two
 *  candidates closer than `dedup_dist_` are merged, with the lower
 *  loss kept. The pool is capacity-bounded; on overflow the worst-loss
 *  basin is evicted.
 *
 *  Restartability: the pool is an ICheckpointable participant. On
 *  resume, the basins (and their per-basin state) are reconstructed
 *  exactly. This is what makes BASINS restart "move the needle in the
 *  right direction" -- even if the optimiser internal state is lost,
 *  every previously-discovered basin survives.
 */

#include <tadah/core/checkpoint.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

struct Basin {
    /* Parameters in optimiser-transformed (search) space. */
    std::vector<double>      params;
    double                   loss = 0.0;
    int                      birth_round = 0;
    int                      last_round  = 0;
    /* If set, the basin has converged at the given local-tol level
     * and won't be revisited at tolerances >= that value. */
    std::optional<double>    exhausted_at_tol;
    /* Optimiser-state-agnostic restart hint: the smallest local-tol
     * level that has actually been *attempted* on this basin. Lets
     * BasinsPolicy resume the local ladder for this basin from the
     * right rung after a crash. */
    std::optional<double>    last_tol_attempted;
};

class BasinPool : public core::ICheckpointable {
public:
    /* dedup_dist is a *relative* distance: ||a - b||_2 / ||high - low||_2
     * (computed in optimiser-transformed space; the caller passes the
     * normalisation reference via set_normalisation()).
     * pool_max bounds the number of resident basins. */
    BasinPool(double dedup_dist, std::size_t pool_max);

    /* Set the per-dimension normalisation used in dedup distance. The
     * caller (BasinsPolicy) passes the optimiser bounds (low, high).
     * Must be set before add_or_merge() if dedup_dist > 0. */
    void set_normalisation(std::vector<double> low, std::vector<double> high);

    /* Insert or merge by deduplication. Returns the index of the
     * resident basin (may be a newly-created or existing one). On
     * capacity overflow, evicts the worst-loss basin. If the new
     * candidate is itself worse than every existing basin, it is not
     * inserted (returned index == -1). */
    int add_or_merge(std::vector<double> params, double loss, int round);

    /* Mark the i-th basin as exhausted at `tol`. Idempotent -- the
     * tightest (smallest) value wins. */
    void mark_exhausted(std::size_t i, double tol);

    /* Mark the i-th basin as having attempted refinement at `tol`. */
    void mark_attempted(std::size_t i, double tol);

    /* Update a basin's params/loss after a successful refinement. */
    void update(std::size_t i, std::vector<double> params, double loss, int round);

    /* All basins, sorted ascending by loss when called via heads().
     * heads_view() is a non-sorting reference. */
    std::vector<Basin>        heads() const;
    const std::vector<Basin>& heads_view() const { return basins_; }
    std::size_t               size()  const { return basins_.size(); }
    bool                      empty() const { return basins_.empty(); }

    /* Indices of basins that are not yet exhausted at `tol`. Useful
     * for the LOCAL stage's per-round basin sweep. */
    std::vector<std::size_t>  not_exhausted_at(double tol) const;

    /* Configuration access. */
    double      dedup_dist() const { return dedup_dist_; }
    std::size_t pool_max()   const { return pool_max_; }

    /* ICheckpointable */
    std::string tag() const override { return "BASIN_POOL"; }
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;

private:
    double      relative_l2_(const std::vector<double>& a,
                             const std::vector<double>& b) const;
    int         find_nearest_within_(const std::vector<double>& params,
                                     double thresh) const;
    int         worst_index_() const;

    double                  dedup_dist_ = 0.05;
    std::size_t             pool_max_   = 32;
    std::vector<double>     low_, high_;     // bounds for normalisation
    std::vector<Basin>      basins_;
};

}   // namespace hpo
}   // namespace tadah

#endif

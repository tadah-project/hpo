#ifndef TADAH_HPO_BASIN_POOL_V2_H
#define TADAH_HPO_BASIN_POOL_V2_H

/*
 *  BasinPoolV2 — extended basin pool for the BASINSv2 strategy.
 *
 *  Extends v1's BasinPool with:
 *    - per-basin lifecycle state (ACTIVE / PROMISING / INACTIVE /
 *      SUSPECT_DUP / MERGED / KILLED) so the scheduler can deprioritise
 *      stagnant or duplicate basins without silent eviction.
 *    - stable monotonic basin id (independent of vector index, so merge
 *      tombstones and the priority queue can refer to basins safely
 *      across pool compactions).
 *    - improvement EMA, eval bookkeeping, per-basin tol cursor — the raw
 *      inputs to the priority scoring formula.
 *    - verify-before-kill counters (ticks an INACTIVE basin a couple of
 *      times before transitioning to KILLED).
 *    - sibling_root_id for union-find duplicate merging (Phase 2).
 *    - capped coarse path storage (Phase 4).
 *
 *  The pool is an ICheckpointable participant; restart is mandatory for
 *  the anytime contract. The save/load grammar deliberately mirrors v1's
 *  per-basin key prefixing so future BasinPoolV2-specific fields can be
 *  added without breaking older v2 checkpoints (they get default values).
 *
 *  Active-vs-archived: ACTIVE/PROMISING/INACTIVE/SUSPECT_DUP basins
 *  count against `pool_max`; MERGED/KILLED basins count against
 *  `archive_max`. Archived basins are retained because the EXPLORE filter
 *  uses them to answer "have we explored this region?" — silently evicting
 *  them would defeat the design.
 */

#include <tadah/core/checkpoint.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

/* Lifecycle states. Encoded as int in checkpoint (Checkpoint API has no
 * string put for enum-style values; ints are compact and stable). */
enum class LifecycleState : int {
    ACTIVE       = 0,
    PROMISING    = 1,
    INACTIVE     = 2,
    SUSPECT_DUP  = 3,
    MERGED       = 4,
    KILLED       = 5,
};

const char* lifecycle_name(LifecycleState s);

/* True iff the basin counts against pool_max (vs archive_max). */
bool is_archive_state(LifecycleState s);

/* Coarse path snapshot — kept here (not in basins_v2_path.h) so the
 * pool's save/load can serialise the field without a circular include.
 * Phase 4 may grow path-handling helpers in basins_v2_path.h that
 * operate on this struct; the data layout stays in the pool. */
struct PathPoint {
    std::vector<double> params;   // optimiser-transformed search space
    double              loss        = 0.0;
    std::size_t         eval_index  = 0;   // global eval counter when sampled
};

struct BasinV2 {
    /* --- identity (stable across pool compactions) ----------------- */
    int                 id              = 0;   // monotonic, unique within pool lifetime
    int                 sibling_root_id = 0;   // union-find root; == id iff not merged

    /* --- search-space record (mirrors v1 Basin) -------------------- */
    std::vector<double> params;
    double              loss        = 0.0;

    /* --- lifecycle + iteration timestamps -------------------------- */
    LifecycleState      state                  = LifecycleState::ACTIVE;
    int                 birth_iteration        = 0;
    int                 last_iteration         = 0;   // any work scheduled on it
    int                 last_improve_iteration = 0;   // last loss-improving event

    /* --- progress signals (priority inputs) ------------------------ */
    /* EMA of (delta_loss / eval) over recent refinements. Smoothed with
     * factor 0.5 by default; positive = improving, zero or near-zero =
     * stagnant. Reset on lifecycle transitions that imply a fresh start. */
    double              improvement_ema        = 0.0;
    std::size_t         evals_consumed_total   = 0;
    std::size_t         evals_since_improvement = 0;

    /* --- tol cursor (per-basin; not lockstep) ---------------------- */
    /* current_tol_cursor: the next ladder rung the scheduler will give
     * this basin in REFINE. last_attempted_tol / last_exhausted_tol:
     * the deepest tol the optimiser tried and the deepest it converged
     * at (mirrors v1 BasinPool semantics). */
    double                current_tol_cursor    = 1.0e-3;
    std::optional<double> last_attempted_tol;
    std::optional<double> last_exhausted_tol;

    /* --- verify-before-kill state ---------------------------------- */
    int                 verify_attempts        = 0;
    int                 verify_passes          = 0;

    /* --- coarse trajectory (Phase 4; may stay empty in earlier
     *     phases — checkpoint round-trips an empty vector cleanly) --- */
    std::vector<PathPoint> coarse_path;
};

class BasinPoolV2 : public core::ICheckpointable {
public:
    /* dedup_dist is a relative L2 in normalised search space (same as
     * v1 BasinPool). pool_max bounds *active* basins; archive_max bounds
     * *terminal* basins (MERGED + KILLED). Either may be 0 (interpreted
     * as 1 minimum to avoid the empty-pool corner case). */
    BasinPoolV2(double dedup_dist, std::size_t pool_max,
                std::size_t archive_max = 0);

    /* Set per-dim normalisation (caller passes outer_.low/high). Must
     * be called before add_or_merge() if dedup_dist > 0. */
    void set_normalisation(std::vector<double> low, std::vector<double> high);

    /* Insert or merge a candidate. Returns the resulting basin's
     * stable id (NOT vector index!), or -1 if the candidate is worse
     * than every active resident and the active pool is at capacity.
     *
     * Initial state is ACTIVE. Merge-by-distance considers ACTIVE,
     * PROMISING, INACTIVE, SUSPECT_DUP residents; merging into MERGED
     * or KILLED is forbidden — those are tombstones. */
    int add_or_merge(std::vector<double> params, double loss, int iteration);

    /* Refinement-complete update: records params/loss (if improved),
     * updates EMA, eval counters, last_iteration / last_improve_iteration,
     * and (optionally) marks attempted/exhausted tol. */
    void on_refinement_complete(int basin_id,
                                std::vector<double> params,
                                double loss,
                                std::size_t evals_used,
                                int iteration,
                                double tol_attempted,
                                bool optimizer_converged_naturally);

    /* Explicit lifecycle transitions. Updates last_iteration. Refuses
     * transitions out of MERGED/KILLED (terminal). */
    void set_state(int basin_id, LifecycleState s, int iteration);

    /* Verify-before-kill bookkeeping. Pass `improved=true` if the
     * tickle refinement produced a measurable improvement. */
    void on_verify_attempt(int basin_id, bool improved, int iteration);

    /* Mark basin as exhausted at the given tol (idempotent — tightest
     * value wins). */
    void mark_exhausted(int basin_id, double tol);

    /* Append a coarse-path snapshot. Drops the oldest snapshot once the
     * cap is reached (Phase 4 may swap in a smarter pruning rule). */
    void append_path_point(int basin_id, PathPoint p, std::size_t cap);

    /* ---------- Phase 2: duplicate detection + merging ------------- */

    /* Find candidate merge pairs across ACTIVE-family basins. A pair
     * (idx_a, idx_b) qualifies iff:
     *   - both are in ACTIVE / PROMISING / INACTIVE / SUSPECT_DUP
     *   - relative_l2(params_a, params_b) <= merge_dist
     *   - |loss_a - loss_b| / max(|loss_a|, |loss_b|, eps) <= merge_loss_threshold
     * Each basin appears in at most one pair (each is matched with its
     * nearest qualifying partner). Returned indices are vector indices
     * into view(), valid until the next mutator. */
    std::vector<std::pair<std::size_t, std::size_t>>
    find_merge_candidates(double merge_dist,
                          double merge_loss_threshold) const;

    /* Flag a pair as SUSPECT_DUP so the scheduler refines them once
     * more before deciding. Idempotent — already-suspect basins stay
     * suspect. Refuses if either is in MERGED / KILLED. Returns true
     * if any state actually changed. */
    bool flag_suspect_pair(int basin_id_a, int basin_id_b, int iteration);

    /* Confirm a merge: lower-loss basin survives ACTIVE, the other
     * becomes MERGED with sibling_root_id pointing at the survivor.
     * No-op if either id is missing or already terminal. Eval counters
     * and improvement EMA from the loser are added to the survivor's
     * tallies (cheap accounting; helps keep priority signals coherent). */
    void confirm_merge(int loser_id, int winner_id, int iteration);

    /* Revert SUSPECT_DUP back to ACTIVE for both members. Used when
     * the post-suspect refinement pulled them apart. */
    void revert_suspect_pair(int basin_id_a, int basin_id_b, int iteration);

    /* ---------- queries -------------------------------------------- */
    std::size_t size()           const { return basins_.size(); }
    bool        empty()          const { return basins_.empty(); }
    std::size_t active_count()   const;     // ACTIVE+PROMISING+INACTIVE+SUSPECT_DUP
    std::size_t archive_count()  const;     // MERGED+KILLED

    const std::vector<BasinV2>& view()  const { return basins_; }
    std::vector<std::size_t>    indices_in_state(LifecycleState s) const;
    std::vector<std::size_t>    active_indices() const;

    /* ID-based lookup. Returns nullptr if the id is not present. */
    const BasinV2* by_id(int id) const;
    BasinV2*       by_id(int id);

    /* Index lookup by id (size_t / -1). */
    int index_of_id(int id) const;

    /* Distance helper (exposed for priority scoring + EXPLORE filter). */
    double relative_l2(const std::vector<double>& a,
                       const std::vector<double>& b) const;

    /* Bounds (for distance normalisation). */
    const std::vector<double>& low()  const { return low_;  }
    const std::vector<double>& high() const { return high_; }

    /* ---------- configuration -------------------------------------- */
    double      dedup_dist()  const { return dedup_dist_; }
    std::size_t pool_max()    const { return pool_max_;   }
    std::size_t archive_max() const { return archive_max_; }

    /* ---------- ICheckpointable ------------------------------------ */
    std::string tag() const override { return "BASIN_POOL_V2"; }
    void save(core::Checkpoint& cp) const override;
    void load(const core::Checkpoint& cp) override;

private:
    int    next_id_     = 0;
    double dedup_dist_  = 0.05;
    std::size_t pool_max_    = 24;
    std::size_t archive_max_ = 48;
    std::vector<double> low_, high_;
    std::vector<BasinV2> basins_;

    int  find_nearest_within_active_(const std::vector<double>& params,
                                     double thresh) const;
    int  worst_active_index_() const;
    void evict_worst_archive_();   // when archive_count > archive_max
};

}   // namespace hpo
}   // namespace tadah

#endif

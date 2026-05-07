#ifndef TADAH_HPO_TOLERANCE_LADDER_H
#define TADAH_HPO_TOLERANCE_LADDER_H

/*
 *  ToleranceLadder — round-indexed tolerance schedule for the BASINS strategy.
 *
 *  The ladder controls how aggressively each round tightens its
 *  ftol_rel:
 *    - glo(round)  picks the global-stage tolerance for round k
 *    - loc(round)  picks the cumulative list of local-stage tolerances
 *                  to visit in round k (round 1 visits one rung, round
 *                  2 visits two rungs, ..., capped at the ladder length).
 *
 *  This produces the user's expected anytime profile: the first
 *  potential lands quickly under loose tol; later rounds spend more
 *  evals refining at tighter tol, hence diminishing-but-never-silent
 *  improvements.
 *
 *  Auto-derive: when the user omits TOL_GLO_LADDER / TOL_LOC_LADDER,
 *  build_default_ladders() picks 1e-3, 1e-4, ... down to a noise-floor
 *  (10× the relative spread reported by NOISE_CALIBRATION, or 1e-9
 *  if calibration was not run).
 */

#include <cstddef>
#include <optional>
#include <vector>

namespace tadah {
namespace hpo {

class ToleranceLadder {
public:
    ToleranceLadder() = default;

    /* Construct from explicit lists. Either may be empty, in which
     * case the corresponding default is used. */
    static ToleranceLadder from_explicit(std::vector<double> glo,
                                         std::vector<double> loc);

    /* Default ladder: [1e-3, 1e-4, ... down to floor]. Floor is the
     * larger of 1e-9 and 10 * noise_rel_spread. Default 6 rungs but
     * trimmed if the floor is reached early. */
    static ToleranceLadder build_default(std::optional<double> noise_rel_spread,
                                         std::size_t           default_rungs = 6);

    /* Returns the global tolerance for the given (1-indexed) round.
     * For round > ladder length, returns the last entry (clamps). */
    double      glo(int round) const;

    /* Returns the local tolerance for round-step (k, s) where
     *    s in [1 .. min(round, ladder_length)].
     * For out-of-range, returns the last entry. */
    double      loc(int round, int step) const;

    /* How many local rungs to visit in round k. */
    int         loc_steps_for(int round) const;

    /* Inspectors. */
    const std::vector<double>& glo_ladder() const { return glo_; }
    const std::vector<double>& loc_ladder() const { return loc_; }

    /* Drops rungs strictly less than `floor` from both ladders.
     * If a ladder becomes empty as a result, push `floor` so it is
     * never empty (BasinsPolicy assumes at least one rung). Caller is
     * responsible for re-clamping after `from_explicit` is re-run.
     *
     * Use case: post-NOISE_CALIBRATION, BasinsPolicy may know an
     * objective-noise floor (suggested_ftol). User-supplied ladders
     * extending below the floor refine in noise; clamping skips those
     * rungs. Opt-in via BasinsSpec::clamp_ladder_to_noise_floor. */
    void trim_below(double floor);

private:
    std::vector<double> glo_;
    std::vector<double> loc_;
};

}   // namespace hpo
}   // namespace tadah

#endif

#pragma once
/*
 * failed_loss_stand_in.h -- the value an optimiser that models the loss is
 * given for a failed evaluation (DLIB:GFS, run_dlib_global; NLopt,
 * run_nlopt_impl).
 *
 * A failed evaluation reports FAILSCORE, by default the largest double.
 * Values that large break the optimisers that build a model of the loss
 * from the values they are given:
 *   - dlib's find_min_global (DLIB:GFS) maximises the negated loss, and the
 *     trust-region step of its search (dlib::qopt_impl::
 *     pick_next_sample_using_trust_region) takes as best sample the first
 *     one whose negated value exceeds -1e300. Once the dims + 2 samples dlib
 *     had when it first took that step all sat at or above 1e300 it found no
 *     best sample and, its assertions compiled out, indexed an empty vector
 *     (SIGSEGV) -- whether or not later evaluations would have produced
 *     usable losses. Values that large also overflow the squares of dlib's
 *     upper-bound model, which normalises the samples by their standard
 *     deviation.
 *   - NLopt's LN_BOBYQA and LN_COBYLA fit a quadratic or linear model
 *     through the values: one value near the largest double overflows it,
 *     and the search then asks for points it has evaluated (answered from
 *     the run's record, so MAXEVAL is never reached) or ends after a few
 *     evaluations.
 * So the optimiser is given a finite stand-in for a failed evaluation;
 * loss.tadah, the journal and the incumbent keep the loss the evaluation
 * reported.
 *
 * A failed evaluation is one without a usable loss (loss_is_feasible):
 * non-finite, FAILSCORE, or a graded failure in [0.999 FS, FS). Its grade
 * g in [0, 1] places it among the failures -- (loss - 0.999 FS) /
 * (0.001 FS) for a graded failure, 1 for any other:
 *
 *   once usable losses exist, with W the worst and B the best so far:
 *     W + m (1 + g),  m = max(W - B, |W|), or 1 when both are 0
 *   -- above every usable loss seen, by one to two times the larger of their
 *   spread and magnitude: a step of the size of the landscape the optimiser
 *   has seen, not a cliff that dwarfs it;
 *
 *   before any usable loss:  K (0.999 + 0.001 g),  K = min(FS, kCeiling)
 *   (kCeiling when FS is not positive) -- a positive FAILSCORE up to
 *   kCeiling gives exactly the loss the evaluation reported, the default
 *   largest double gives kCeiling.
 *
 * A stand-in is fixed when the optimiser is given it and never revised, and
 * "so far" is this object's: one per search, so a RESTART round starts
 * afresh with its fresh optimiser. Graded failures therefore keep their
 * order (FAILSCORE_GRADING's way out of a failing region), below the
 * ungraded ones, only among the failures handed over in the same state: all
 * before the first usable loss, or all while the same W and B hold. Across
 * states the order is not kept: a failure before the first usable loss
 * keeps its value near K, far above the W + 2m of a later ungraded failure,
 * and an ungraded failure handed over while W is small can rank below a
 * graded one handed over after W has grown.
 *
 * No value handed over exceeds kCeiling in magnitude: a usable loss beyond
 * it is clamped as well, and a stand-in capped at it, so a stand-in can rank
 * below a usable loss only when that loss is above 1e100 itself. kCeiling =
 * 1e100 lies 200 orders of magnitude below dlib's 1e300 sentinel; its
 * square, and a sum of such squares over any feasible number of
 * evaluations, stays finite in the standard deviation dlib normalises by,
 * and so do its products with the quadratic model's interpolation
 * coefficients, which grow like the inverse square of the trust-region
 * radius, for any radius above 1e-100. Yet it lies far above any loss a
 * potential produces.
 *
 * A gradient that comes with a loss (NLopt's LD_ and GD_ algorithms) is the
 * gradient of the value handed over: slope() gives the derivative of that
 * value in the loss at the state it is handed over in -- 1 for a usable loss
 * below the ceiling, the slope of the stand-in in a graded failure's grade
 * for a graded failure (the finite differences of a gradient never mix a
 * graded failure with another kind, gradient_provider.h), 0 where the value
 * does not follow the loss (an ungraded failure, whose gradient fails, or a
 * value held at the ceiling).
 *
 * Thread-safe: find_min_global evaluates on a pool.
 */

#include <tadah/hpo/pc_fail_grading.h>   // loss_is_feasible, kFailBandFloor

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>

namespace tadah::hpo::detail {

class FailedLossStandIn {
public:
  /** The largest magnitude an optimiser is given (see above). */
  static constexpr double kCeiling = 1e100;

  /** For a search whose FAILSCORE is `failscore` (none: only a loss that
   *  is not finite is a failure). */
  explicit FailedLossStandIn(std::optional<double> failscore)
      : failscore_(failscore)
  {}

  /** Is `loss` a failed evaluation's, one the optimiser is given a stand-in
   *  for? */
  bool failed(double loss) const
  {
    return failscore_ ? !loss_is_feasible(loss, *failscore_)
                      : !std::isfinite(loss);
  }

  /** The value the optimiser is given for an evaluation that reported
   *  `loss`; a usable loss also joins the worst and best usable losses so
   *  far. */
  double operator()(double loss)
  {
    double slope = 0.0;
    return (*this)(loss, slope);
  }

  /** As operator()(loss), and `slope` the derivative of the value in the
   *  loss (see above). */
  double operator()(double loss, double& slope)
  {
    std::lock_guard<std::mutex> lk(m_);
    if (failed(loss)) {
      const double g = grade(loss);
      const double v = stand_in_locked(g);
      slope = std::abs(v) < kCeiling ? stand_in_slope_locked(loss) : 0.0;
      return v;
    }
    const double v = std::clamp(loss, -kCeiling, kCeiling);
    slope = v == loss ? 1.0 : 0.0;
    if (!have_usable_) {
      best_ = worst_ = v;
      have_usable_ = true;
    } else {
      best_  = std::min(best_, v);
      worst_ = std::max(worst_, v);
    }
    return v;
  }

  /** The value the optimiser is given for an evaluation that did not run
   *  (queued after a stop): an ungraded failure's. */
  double not_evaluated() const
  {
    std::lock_guard<std::mutex> lk(m_);
    return stand_in_locked(1.0);
  }

private:
  /* Whether `loss` is a graded failure, whose stand-in follows its grade. */
  bool graded(double loss) const
  {
    if (!failscore_ || !std::isfinite(loss)) return false;
    const double fs = *failscore_;
    if (!(fs > 0.0) || !std::isfinite(fs)) return false;
    return loss >= kFailBandFloor * fs && loss < fs;
  }

  double grade(double loss) const
  {
    if (!graded(loss)) return 1.0;
    const double fs = *failscore_;
    const double floor = kFailBandFloor * fs;
    return std::clamp((loss - floor) / (fs - floor), 0.0, 1.0);
  }

  /* The scale of the stand-ins: m once usable losses exist, K before. */
  double margin_locked() const
  {
    double m = std::max(worst_ - best_, std::fabs(worst_));
    if (!(m > 0.0)) m = 1.0;
    return m;
  }
  double base_locked() const
  {
    double k = kCeiling;
    if (failscore_ && *failscore_ > 0.0)
      k = std::min(*failscore_, kCeiling);
    return k;
  }

  double stand_in_locked(double g) const
  {
    if (have_usable_)
      return std::min(worst_ + margin_locked() * (1.0 + g), kCeiling);
    return base_locked() * (kFailBandFloor + (1.0 - kFailBandFloor) * g);
  }

  /* d stand-in / d loss of the failed evaluation `loss`. */
  double stand_in_slope_locked(double loss) const
  {
    if (!graded(loss)) return 0.0;
    const double fs = *failscore_;
    const double band = fs - kFailBandFloor * fs;
    if (have_usable_) return margin_locked() / band;
    return base_locked() * (1.0 - kFailBandFloor) / band;
  }

  std::optional<double> failscore_;
  mutable std::mutex m_;
  bool   have_usable_ = false;
  double best_  = 0.0;
  double worst_ = 0.0;
};

}  // namespace tadah::hpo::detail

#pragma once
/*
 * gradient_provider.h -- unified gradient backend for gradient-based HPO
 * optimizers (DLIB:BFGS, CERES:STEEPEST_DESCENT,
 * NLOPT:LD_* and the local search of MLSL / AUGLAG).
 *
 * Three strategies, selected by OptimizerSpec::grad_strategy (or default):
 *
 *   GradStrategy::Numerical  -- finite differences (fd_stencil.h): the step
 *                              rule of PARAM fd_kernel (inhouse, or Ceres'
 *                              rule, evaluated by Tadah), CENTRAL or
 *                              FORWARD (PARAM diff_method), never outside
 *                              the OPTIM box: one-sided into it within a
 *                              step of a bound; and of one kind (below).
 *                              The loss at the point is the caller's
 *                              (gradient_at, below), then each dimension's
 *                              stencil: 2n evaluations (CENTRAL, and
 *                              one-sided within a step of a bound), n
 *                              (FORWARD), one more for a CENTRAL component
 *                              that turns one-sided next to a failure;
 *                              gradient(x, g) evaluates the loss at the
 *                              point first, one more.
 *   GradStrategy::Analytical -- strict closed-form. The caller provides
 *                              an `analytical_grad_fn` callback that
 *                              fills g[] and reports the theta indices
 *                              it could not differentiate via a
 *                              `needs_fd` out-list. Throws if any index
 *                              is left -- use HYBRID for graceful
 *                              fallback. A failed loss the caller gives
 *                              (gradient_at) fails it (below).
 *   GradStrategy::Hybrid     -- analytical for registered theta + the
 *                              finite differences of NUMERICAL (the same
 *                              stencil and PARAMs; the in-house rule by
 *                              default under either OPTIM_SCALE) for the
 *                              indices in `needs_fd` -- every index when no
 *                              analytical derivative can be formed at the
 *                              point -- in one gradient(x, g) call.
 *                              Same `analytical_grad_fn` signature.
 *                              A one-sided or forward difference needs the
 *                              loss at x itself, which the analytical pass
 *                              does not evaluate: it takes the caller's
 *                              (gradient_at, below), else evaluates it once
 *                              per gradient (HPOLog counts it).
 *
 * The factory's `analytical_grad_fn` parameter has the signature
 *
 *     bool(const Vec& x, Vec& g, std::vector<int>& needs_fd)
 *
 * Returning false signals a bad eval (e.g. r.was_nan_clamped or
 * non-finite W*). Returning true means "g[i] is filled for every i not
 * in needs_fd; for i in needs_fd, the caller must source g[i]
 * elsewhere". The Hybrid strategy then takes finite differences on
 * exactly those indices, sharing the same eval_counter so MAXEVAL
 * accounting works uniformly. Numerical ignores the callback.
 *
 * gradient_at(x, fx, g) is gradient(x, g) for a caller that has just
 * evaluated the loss at exactly x itself -- fx is that value, the one the
 * optimiser sees (with --hotvar warm starts a second evaluation of the
 * same point can differ): the Ceres and NLopt adapters, whose objective
 * evaluates the loss and then asks for the gradient of the same point, and
 * DLIB:BFGS when the last evaluation of its line search is the point whose
 * gradient dlib asks for (bitwise, and nothing evaluated since). NUMERICAL
 * and HYBRID take it as the loss at the point -- the centre of a one-sided
 * or forward stencil, and the loss that decides how a failed evaluation is
 * read (below) -- and
 * do not evaluate the point again: a central stencil does not use its
 * centre, a one-sided one uses the loss the optimiser was given. NUMERICAL
 * used to evaluate the point once more (the old kernels' evaluations): one
 * evaluation per gradient, counted against MAXEVAL. ANALYTICAL needs no
 * loss for its gradient, and reads the caller's to tell a failed
 * evaluation (below), as HYBRID does before its analytical pass.
 *
 * MAXEVAL counts every evaluation of a run in the shared eval_counter: the
 * provider counts its own (cost(), each finite-difference point, one per
 * analytical gradient), and the adapters that evaluate the loss themselves
 * (DLIB:BFGS, the NLopt objective) count theirs in the same counter.
 * Optimizers receive `gradient(x,g) -> false` (and likewise
 * `cost(x) -> +inf`) once the counter reaches MAXEVAL or the host returns
 * a non-finite value, so they can terminate cleanly without exceptions
 * propagating across Ceres' Solve() frames (Ceres is not exception-safe).
 *
 * Why a gradient() call returned false is recorded in
 * GradientProvider::failure (GradientFailure): the MAXEVAL budget
 * (Budget), a failed evaluation the gradient reads -- in the
 * finite-difference stencil, or the loss at the point (Stencil) -- or a
 * gradient not finite for another reason (NotFinite: an
 * overflow of the quotient, an analytical gradient). The adapters turn
 * Stencil and NotFinite into a rejected point or the search's own ending,
 * "STOPPED -- no usable gradient: <cause>" (gradient_stop_status), where
 * they used to report MAXEVAL or success.
 *
 * A failed evaluation is a loss that is not usable (StencilReport::failed:
 * not finite, or at least 0.999 FAILSCORE -- FAILSCORE or a graded
 * failure, loss_is_feasible(); HPO_Host clamps a non-finite loss to
 * FAILSCORE, so in a run a failed evaluation is always a finite number).
 * The loss at the point is of one of three kinds (StencilReport::classify):
 * usable, a graded failure (FAILSCORE_GRADING ON,
 * OptimizerSpec::failscore_grading: a loss in [0.999 FS, FS), whose
 * grading measures the way out of the failure region), or another failure
 * (FAILSCORE itself -- FAILSCORE_GRADING OFF, an infrastructure failure, a
 * loss the host clamped -- or a loss that is not finite).
 *
 * A difference never mixes the kinds: one between two failures is an exact
 * 0 ((FS - FS) / 2h), which a search took
 * for convergence at a start inside the failure region, and one between a
 * usable loss and a failure, or a graded failure and a usable loss or an
 * ungraded one, is as large as the jump into or out of the failure band
 * (it overflows under the default FAILSCORE, the largest double, and is a
 * steep, finite slope under one such as 1e9: from a graded failure next to
 * the usable side a search sat on a bound, or NLopt's LD_SLSQP stopped with
 * "bug: more than iter SQP iterations"). The finite differences read the
 * loss at the point first, then:
 *
 *   another failure at x      -- the gradient fails at once (Stencil),
 *                                before any stencil point;
 *   a usable loss or a graded failure at x
 *                             -- each component takes the side of its
 *                                stencil whose points are of that kind
 *                                (detail::fd_component: a CENTRAL stencil
 *                                with one side of another kind turns
 *                                one-sided into the other side, a FORWARD
 *                                one backward, as a bound there would make
 *                                it); without such a side the gradient
 *                                fails (Stencil), and so does one at a
 *                                graded failure whose every component is 0:
 *                                a flat failure region (a saturated
 *                                measure, a band that is constant without
 *                                PC_FAILPROBE).
 *
 * A one-sided or forward stencil (fd_stencil.h) takes the loss at x itself.
 * HYBRID knows the loss at x from the caller (gradient_at); its gradient()
 * evaluates it first (counted, HPOLog::record_hybrid_centre), as NUMERICAL's
 * does: its kind decides the gradient, and the analytical pass then
 * differentiates the host's evaluation of x. An evaluation with a failed term
 * (a PC_LAMMPS simulation, a PC_OBSERVABLE term) is a failed evaluation:
 * HPO_Host scores it FAILSCORE, or a graded failure, never the mean of its
 * terms (which stayed below 0.999 FS and read as usable here).
 *
 * ANALYTICAL, and HYBRID before its analytical pass, classify the caller's
 * loss at x (gradient_at) the same way: a failure (CentreLoss::Failed)
 * fails the gradient at once, "a failed evaluation at the point"
 * (kGradientFailedPointCause), and nothing is evaluated or counted. The
 * analytical gradient at a failed point is that of the loss before the
 * failure (a skip cap's uncapped loss, a loss without the term that
 * failed), not of the FAILSCORE the optimiser sees: small but not 0, it
 * let a search that started inside the failure region converge on
 * FAILSCORE (DLIB:BFGS, Ceres CONVERGENCE, NLopt "algorithm converged"),
 * under ANALYTICAL (the default under PC_OBSERVABLE, hpo_targets.cpp) as
 * under HYBRID with nothing differenced. Nor does it see the grading of a
 * graded failure (CentreLoss::Graded) under FAILSCORE_GRADING ON, whose
 * loss a band measure sets (pc_fail_grading.h), not the loss terms: there
 * ANALYTICAL fails the gradient at once too (the same cause, "... a graded
 * failure, whose grading the analytical gradient does not see"), and
 * HYBRID runs no analytical pass and differences every parameter, which
 * gives the graded slope or a flat failure region (above). Both used to
 * keep the analytical gradient there: with every failed PC_OBSERVABLE term
 * a band-B failure, a search that started where one failed converged on
 * the graded loss (DLIB:BFGS FTOL_ABS, NLopt "algorithm converged").
 * Every backend passes the loss it has just evaluated with each gradient
 * it asks for (gradient_at: the NLopt and Ceres adapters, DLIB:BFGS), so the
 * kind of the point is always known. gradient(x, g) without it, which no
 * backend calls, has nothing to classify under ANALYTICAL, which gives the
 * analytical gradient there; HYBRID's evaluates the loss at x first
 * (below).
 *
 * A point the host has no fit of -- a candidate the bump-support or a
 * density gate rejected before training (a graded band-D failure under
 * FAILSCORE_GRADING ON), a training that did not complete -- and a point
 * whose fit the host's analytical pass cannot differentiate (the normal
 * equations of the fit not positive definite in its implicit-function
 * solve) have no analytical gradient: the host's analytical pass hands
 * every parameter to finite differences there, which HYBRID takes.
 * ANALYTICAL asks the host why (HPO_Host::analytical_gradient_gap(), when
 * HostT has it) whenever the pass leaves a parameter to finite
 * differences: a reason makes the point one without a usable gradient
 * (NotFinite, "the analytical gradient could not be formed at this
 * point"), which NLopt and Ceres reject and DLIB:BFGS stops at, as any
 * such point. Without one it is a parameter or a loss term without an
 * analytical derivative, the configuration's: an error naming the
 * parameters (GRAD_STRATEGY HYBRID fills them). That error used to end the
 * run FAILED at every rejected candidate, and at a fit the analytical pass
 * could not differentiate.
 *
 * A point outside the OPTIM box (box_penalty.h) is not evaluated: HPO_Host
 * scores it out_of_box_score() of its distance to the box, and every
 * strategy gives it the slope of that score (detail::out_of_box_gradient),
 * pointing away from the box, without
 * evaluating or counting anything and without a stencil -- no difference
 * reads an out-of-box score, and no stencil of a point inside the box
 * leaves it. A coordinate within rounding outside its box is evaluated on
 * the bound and differenced there (FdBox::onto).
 *
 * Journal (HPOLog): a component a bound turned (one-sided into the box,
 * backward) is counted per dimension, and so is one taken without a point
 * of another kind ("Mixed-kind stencils"); the first of each dimension gets
 * an INFO line. HYBRID's own evaluations of the point are counted too.
 * DIAGNOSTICS prints the counts that are not zero.
 *
 * Lifetime: capture by reference is safe because the provider is built
 * inside a `run_<lib>_<algo>` adapter and consumed within the same call.
 */

#include <tadah/hpo/box_penalty.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/pc_fail_grading.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tadah::hpo {

/* Why GradientProvider::gradient() returned false. */
enum class GradientFailure {
  None,      /* the last call returned a usable gradient */
  Budget,    /* MAXEVAL was spent before the gradient was complete */
  Stencil,   /* a failed evaluation the gradient reads (see the file
                comment): the loss at the point -- the caller's under every
                strategy, or the one the finite differences evaluate -- or
                a finite-difference stencil point; or a flat failure region
                at a graded failure */
  NotFinite  /* a gradient that is not finite for another reason */
};

/** The failure of the last gradient() call: `cause` is the fixed reason
 *  the status and the journal give, `detail` what failed, for the journal
 *  (the point, the stencil point and its loss). */
struct GradientFailureInfo {
  GradientFailure kind = GradientFailure::None;  ///< which failure (None after a usable gradient)
  std::string     cause;   ///< the fixed reason the status and the journal give
  std::string     detail;  ///< what failed: the point, the stencil point and its loss
};

/* The cause of a Stencil failure of the finite differences (NUMERICAL,
 * HYBRID's differenced components): a failed loss at the point, or, at a
 * usable one, a failed stencil point on every side of the stencil that fits
 * in the box. */
inline const std::string kGradientStencilCause =
    "a failed evaluation in the finite-difference stencil";

/* The cause of a Stencil failure of the finite differences at a graded
 * failure (FAILSCORE_GRADING ON) whose stencil has, on every side that fits
 * in the box, a loss of another kind: a usable loss or an ungraded failure
 * (a difference never mixes the kinds). */
inline const std::string kGradientGradedSidesCause =
    "a graded failure without a side of graded failures in its "
    "finite-difference stencil";

/* The cause of a Stencil failure of ANALYTICAL and HYBRID at a failed loss
 * the caller gives (gradient_at), before their analytical pass. */
inline const std::string kGradientFailedPointCause =
    "a failed evaluation at the point";

/* The run's status when a point without a usable gradient (Stencil or
 * NotFinite) ended the search (HPOLog::set_status; the journal trailer of
 * a run it ends). It said "gradient not finite" until 2026-10-07, which a
 * failed stencil evaluation under a finite FAILSCORE is not. */
inline std::string gradient_stop_status(const GradientFailureInfo& f)
{
  return "STOPPED -- no usable gradient: " + f.cause;
}

/* Records that a point without a usable gradient ended the search of
 * `backend` ("DLIB:BFGS"): a journal warning, the same line on stdout, and
 * the run's status. `ended` says how the optimiser ended right after it
 * rejected the point, or what HPO did ("" for neither). */
inline void report_gradient_stop(const std::string& backend,
                                 const GradientFailureInfo& f,
                                 const std::string& ended = "")
{
  std::string line = backend + " stopped: no usable gradient: " + f.cause;
  if (!f.detail.empty()) line += " (" + f.detail + ")";
  if (!ended.empty()) line += "; " + ended;
  line += "; best-so-far preserved";
  std::cout << "[HPO] " << line << "\n";
  HPOLog::instance().warning(line);
  HPOLog::instance().set_status(gradient_stop_status(f));
}

/** The loss and its gradient over the optimiser-internal coordinates x of
 *  a spec, as every gradient-based backend consumes them (see the file
 *  comment): cost() evaluates host(sp.out(x)), gradient() and
 *  gradient_at() obtain d loss/dx by the spec's GRAD_STRATEGY within its
 *  MAXEVAL, and `failure` says why the last gradient was not usable. */
template<class Vec, class Runner>
struct GradientProvider {
  /** Return host(sp.out(x)). +inf on bad eval or MAXEVAL overflow. */
  std::function<double(const Vec& x)>            cost;

  /** Whether MAXEVAL is spent: cost() returns +inf and gradient() fails
   *  (GradientFailure::Budget) without evaluating from now on. Lets a
   *  caller tell a refusal for the budget from a failed evaluation (the
   *  Ceres adapter's MAXEVAL ending). Always false without MAXEVAL. */
  std::function<bool()>                          exhausted;

  /** Fill g with d(host(sp.out(x)))/dx. Return false on bad eval or
   *  MAXEVAL overflow; `failure` then says which. */
  std::function<bool(const Vec& x, Vec& g)>      gradient;

  /** gradient(x, g) at a point whose loss the caller has just evaluated
   *  itself: fx = host(sp.out(x)) of exactly this x, the last evaluation
   *  made (see the file comment). NUMERICAL and HYBRID take it as the loss
   *  at x and do not evaluate x; every strategy fails at a failed fx
   *  (ANALYTICAL is gradient(x, g) otherwise). */
  std::function<bool(const Vec& x, double fx, Vec& g)> gradient_at;

  /** Why the last gradient() call returned false (kind None after a call
   *  that returned true). Shared with the gradient callable, which fills
   *  it. */
  std::shared_ptr<GradientFailureInfo> failure =
      std::make_shared<GradientFailureInfo>();
};

namespace detail {

/* Whether two points are the same doubles, bit for bit (-0 and +0 differ,
 * a NaN matches nothing): the test for reusing a loss evaluated at `a` as
 * the loss at `b`. */
template<class Vec>
bool same_point(const Vec& a, const Vec& b)
{
  if (a.size() != b.size()) return false;
  for (long i = 0; i < a.size(); ++i) {
    const double x = a(i), y = b(i);
    if (std::isnan(x) || std::memcmp(&x, &y, sizeof x) != 0) return false;
  }
  return true;
}

/* What the loss at the point says about its finite differences (the file
 * comment). */
enum class CentreLoss {
  Usable,   /* a usable loss: its differences take usable points only */
  Graded,   /* a graded failure under FAILSCORE_GRADING ON: its differences
               take graded failures only */
  Failed    /* any other failed evaluation: the gradient fails at once */
};

/** Fills GradientFailureInfo for the finite-difference paths: what a
 *  stencil evaluation or quotient says about the gradient at x; and
 *  journals the components a bound turned. */
template<class Vec>
struct StencilReport {
  const OptimizerSpec*  sp;   ///< the spec of the gradient (FAILSCORE, labels, box)
  GradientFailureInfo*  out;  ///< where each report writes the failure

  /** Whether `y` is a failed evaluation: not a usable loss (not finite, or
   *  at least 0.999 FAILSCORE; without a FAILSCORE, not finite). */
  bool failed(double y) const {
    const double fs = sp->failscore.value_or(
        std::numeric_limits<double>::infinity());
    return !loss_is_feasible(y, fs);
  }
  /** Whether `y` is a graded failure the gradient differences: a failed
   *  evaluation below FAILSCORE under FAILSCORE_GRADING ON. FAILSCORE
   *  itself (an ungraded failure: grading OFF, an infrastructure failure, a
   *  loss the host clamped) is not. */
  bool graded(double y) const {
    return sp->failscore_grading && sp->failscore && failed(y) &&
           std::isfinite(y) && y < *sp->failscore;
  }
  /** What the loss `y` at the point says about its differences:
   *  usable, a graded failure or another failure. */
  CentreLoss classify(double y) const {
    if (!failed(y)) return CentreLoss::Usable;
    return graded(y) ? CentreLoss::Graded : CentreLoss::Failed;
  }
  /** The name of dimension i: its parameter label ("CGRID2B[2]"), else
   *  logidx::dim_label(i). */
  std::string label(long i) const {
    const auto k = static_cast<std::size_t>(i);
    return (k < sp->parameter_labels.size() && !sp->parameter_labels[k].empty())
               ? sp->parameter_labels[k] : logidx::dim_label(k);
  }
  /** "CGRID2B[2] = 1.4": dimension i of the internal point u, in real
   *  units. */
  std::string at(const Vec& u, long i) const {
    const Vec r = sp->out(u);
    std::ostringstream os;
    os << label(i) << " = " << std::setprecision(10) << r(i);
    return os.str();
  }
  /** `v` with 10 significant digits. */
  static std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
  }

  /** MAXEVAL was spent before the gradient was complete. */
  void budget() const { *out = {GradientFailure::Budget, "MAXEVAL spent", ""}; }

  /** "the loss at CGRID2B[2] = 1.25, SGRID2B[2] = 0.5 is <y>" */
  std::string loss_at(const Vec& x, double y) const {
    std::ostringstream os;
    os << "the loss at ";
    const long n = x.size();
    for (long i = 0; i < n; ++i) os << (i ? ", " : "") << at(x, i);
    os << " is " << num(y);
    return os.str();
  }

  /** The loss at x itself failed (CentreLoss::Failed): no stencil point
   *  is evaluated. */
  void centre(const Vec& x, double y) const {
    *out = {GradientFailure::Stencil, kGradientStencilCause, loss_at(x, y)};
  }

  /** The caller's loss at x failed (CentreLoss::Failed) under ANALYTICAL
   *  or HYBRID: no analytical pass is run and nothing is evaluated. */
  void failed_point(const Vec& x, double y) const {
    *out = {GradientFailure::Stencil, kGradientFailedPointCause,
            loss_at(x, y)};
  }

  /** The caller's loss at x is a graded failure (CentreLoss::Graded)
   *  under ANALYTICAL: the analytical gradient there is that of the loss
   *  without the failure (the usable terms, a skip cap's uncapped loss),
   *  not of the graded loss the optimiser sees, which a band measure sets
   *  (pc_fail_grading.h). No analytical pass is run and nothing is
   *  evaluated. */
  void graded_point(const Vec& x, double y) const {
    *out = {GradientFailure::Stencil, kGradientFailedPointCause,
            loss_at(x, y) + ": a graded failure, whose grading the "
                            "analytical gradient does not see; "
                            "GRAD_STRATEGY HYBRID takes finite differences "
                            "there"};
  }

  /** The loss at x is a graded failure and every differenced component of
   *  its gradient is 0: no direction out of the failure region. */
  void flat(const Vec& x, double y) const {
    *out = {GradientFailure::Stencil, kGradientStencilCause,
            loss_at(x, y) + ", and every difference at it is 0: a flat "
                            "failure region"};
  }

  /** The derivative along dimension i, from the stencil evaluations `pts`
   *  (point, loss), overflowed (NotFinite). The points of a one-sided or
   *  forward stencil include x itself. Every loss of `pts` is of the kind
   *  of the loss at x (fd_component). */
  void stencil(const Vec& x, long i,
               const std::vector<std::pair<Vec, double>>& pts) const {
    std::ostringstream os;
    os << "at " << at(x, i) << ", ";
    os << "the losses ";
    for (std::size_t k = 0; k < pts.size(); ++k)
      os << (k ? " and " : "") << num(pts[k].second) << " at "
         << at(pts[k].first, i);
    os << " overflow their difference";
    *out = {GradientFailure::NotFinite,
            "the finite-difference quotient overflowed", os.str()};
  }

  /** No side of the stencil of dimension i is of the kind `kind` of the
   *  loss `y0` at x: `other` holds the stencil points evaluated whose loss
   *  is of another kind (point, loss), on every side that fits in the box.
   *  At a usable loss they failed (kGradientStencilCause); at a graded
   *  failure they are usable losses or ungraded failures
   *  (kGradientGradedSidesCause). */
  void no_side(const Vec& x, long i, double y0, CentreLoss kind,
               const std::vector<std::pair<Vec, double>>& other) const {
    std::ostringstream os;
    os << "at " << at(x, i);
    if (kind == CentreLoss::Graded)
      os << " (a graded failure, loss " << num(y0) << ")";
    os << ", the stencil point" << (other.size() == 1 ? " " : "s ");
    for (std::size_t k = 0; k < other.size(); ++k)
      os << (k ? " and " : "") << at(other[k].first, i);
    os << (kind == CentreLoss::Graded
               ? (other.size() == 1 ? " is not a graded failure"
                                    : " are not graded failures")
               : " failed")
       << ": loss" << (other.size() == 1 ? " " : "es ");
    for (std::size_t k = 0; k < other.size(); ++k)
      os << (k ? " and " : "") << num(other[k].second);
    *out = {GradientFailure::Stencil,
            kind == CentreLoss::Graded ? kGradientGradedSidesCause
                                       : kGradientStencilCause,
            os.str()};
  }

  /** Component i of the gradient at x was taken with the stencil `taken`
   *  instead of the one planned, because the stencil points `other` held
   *  losses of another kind than the loss at x (fd_component): counted per
   *  dimension (DIAGNOSTICS "Mixed-kind stencils"), and the first of each
   *  dimension journalled. Never throws. */
  void mixed(const Vec& x, long i,
             const std::vector<std::pair<Vec, double>>& other,
             const FdStencil& taken, CentreLoss kind) const {
    try {
      if (HPOLog::instance().record_mixed_kind(label(i)) != 1) return;
      Vec p1 = x, p2 = x;
      p1(i) = taken.p1;
      p2(i) = taken.p2;
      std::ostringstream os;
      os << "finite differences: at " << at(x, i) << " ("
         << (kind == CentreLoss::Graded ? "a graded failure" : "a usable loss")
         << "), the stencil point" << (other.size() == 1 ? " " : "s ");
      for (std::size_t k = 0; k < other.size(); ++k)
        os << (k ? " and " : "") << at(other[k].first, i) << " (loss "
           << num(other[k].second) << ")";
      os << (other.size() == 1 ? " is" : " are") << " of another kind, so "
         << "the difference is taken without "
         << (other.size() == 1 ? "it" : "them") << ": ";
      if (taken.kind == FdStencil::Kind::OneSided)
        os << "one-sided 3-point difference, at " << at(p1, i) << " and "
           << at(p2, i) << " with the loss at the point itself";
      else
        os << (taken.s > 0 ? "forward" : "backward") << " difference, at "
           << at(p1, i);
      os << "; a difference never mixes a usable loss with a failure; "
            "DIAGNOSTICS counts such differences per dimension";
      HPOLog::instance().info(os.str());
    } catch (...) { /* journalling never fails a gradient */ }
  }

  /** The analytical gradient failed (NotFinite); `what` says how ("could
   *  not be formed at this point", "is not finite at this point"). */
  void analytical(const std::string& what) const {
    *out = {GradientFailure::NotFinite,
            "the analytical gradient " + what, ""};
  }

  /** The host has no fit of x to differentiate (`why`, its
   *  analytical_gradient_gap()): ANALYTICAL has no gradient there. */
  void untrained(const Vec& x, const std::string& why) const {
    std::ostringstream os;
    os << "at ";
    for (long i = 0; i < x.size(); ++i) os << (i ? ", " : "") << at(x, i);
    os << ": " << why
       << "; GRAD_STRATEGY HYBRID takes finite differences there";
    *out = {GradientFailure::NotFinite,
            "the analytical gradient could not be formed at this point",
            os.str()};
  }

  /** Component i of the gradient at x has the stencil `st`, which a bound
   *  turned (FdStencil::turned): counted per dimension (DIAGNOSTICS), and
   *  the first of each dimension journalled. Never throws. */
  void turned(const Vec& x, long i, const FdStencil& st) const {
    try {
      if (HPOLog::instance().record_one_sided(label(i)) != 1) return;
      const auto k = static_cast<std::size_t>(i);
      std::string box;
      if (sp->space && k < sp->space->dim())
        box = "[" + num(sp->space->at(k).lo) + ", " +
              num(sp->space->at(k).hi) + "]";
      else if (k < sp->low.size() && k < sp->high.size())
        box = "[" + num(sp->low[k]) + ", " + num(sp->high[k]) + "]";
      Vec p1 = x, p2 = x;
      p1(i) = st.p1;
      p2(i) = st.p2;
      std::ostringstream os;
      os << "finite differences: " << at(x, i) << " lies within a step of its "
         << (st.s > 0 ? "lower" : "upper") << " bound (OPTIM box " << box
         << ")";
      if (st.kind == FdStencil::Kind::OneSided)
        os << ": one-sided 3-point difference, at " << at(p1, i) << " and "
           << at(p2, i) << " with the loss at the point itself";
      else
        os << ": " << (st.s > 0 ? "forward" : "backward")
           << " difference, at " << at(p1, i);
      os << "; DIAGNOSTICS counts such differences per dimension";
      HPOLog::instance().info(os.str());
    } catch (...) { /* journalling never fails a gradient */ }
  }
};

/** The finite-difference settings of a provider (fd_settings_of), and the
 *  check of its OPTIM boxes (fd_box_error), made at the first gradient()
 *  call: the provider of a derivative-free NLopt run is built but never
 *  asked, and an invalid setting is refused at startup
 *  (validate_hpo_setup) anyway. */
struct FdSetup {
  FdSettings     s;               ///< the settings (fd_settings_of)
  std::string    settings_error;  ///< why the settings are invalid ("" if valid)
  std::once_flag box_once;        ///< the box check runs once
  std::string    box_error;       ///< why a box point has no stencil ("" if none)

  /** Reads the settings of `sp`; an invalid one is kept for require(). */
  explicit FdSetup(const OptimizerSpec& sp) {
    try { s = fd_settings_of(sp); }
    catch (const std::exception& e) { settings_error = e.what(); }
  }
  /** Throws std::runtime_error for an invalid setting, or a box in which
   *  some point has no stencil -- before any evaluation. */
  void require(const OptimizerSpec& sp) {
    if (!settings_error.empty()) throw std::runtime_error(settings_error);
    std::call_once(box_once, [&] { box_error = fd_box_error(sp, s); });
    if (!box_error.empty()) throw std::runtime_error(box_error);
  }
};

/** The OPTIM box of the finite differences: the internal bounds of `sp`
 *  (OptimizerSpec::low/high), which a point must match in size. */
struct FdBox {
  std::vector<double> lo;  ///< the internal lower bounds
  std::vector<double> hi;  ///< the internal upper bounds

  /** Throws std::logic_error when `sp`'s bounds differ in size. */
  explicit FdBox(const OptimizerSpec& sp) : lo(sp.low), hi(sp.high) {
    if (lo.size() != hi.size())
      throw std::logic_error("GradientProvider: the spec's low and high "
                             "bounds differ in size");
  }
  /** Throws std::logic_error unless a point of `n` dimensions fits the
   *  box. */
  void require(long n) const {
    if (static_cast<std::size_t>(n) != lo.size())
      throw std::logic_error("GradientProvider: a point of " +
                             std::to_string(n) + " dimensions in a box of " +
                             std::to_string(lo.size()));
  }
  /** x with each coordinate outside the box moved onto its bound: the
   *  point a coordinate within rounding outside its box is evaluated at;
   *  a NaN coordinate stays (its evaluation fails as such). Called for a point inside the box by the rule of
   *  box_penalty.h. */
  template<class Vec>
  Vec onto(const Vec& x) const {
    Vec y = x;
    for (long i = 0; i < y.size(); ++i) {
      const auto k = static_cast<std::size_t>(i);
      if (y(i) < lo[k]) y(i) = lo[k];
      else if (y(i) > hi[k]) y(i) = hi[k];
    }
    return y;
  }
};

/* Component idx of the finite-difference gradient at x (fd_stencil.h),
 * whose loss y0 is of the kind `kind` (Usable or Graded; a Failed one ends
 * the gradient before), of one kind: the
 * stencil fd_plan chooses, its points evaluated in its order; a point whose
 * loss is of another kind (StencilReport::classify) ends its side, and the
 * difference is taken on the other side, as a bound there would make it:
 *   CENTRAL   a side of another kind -> the one-sided 3-point difference
 *             into the other side (its far point u -+ 2h one more
 *             evaluation) when it lies in the box, else, and when that far
 *             point is of another kind too, the 2-point difference with
 *             u -+ h;
 *   ONE-SIDED (a bound turned it) its far point of another kind -> the
 *             2-point difference with its near point; its near point of
 *             another kind -> the backward 2-point difference when u - s
 *             lies in the box (it does not, for a stencil a bound turned);
 *   FORWARD   of another kind -> the backward difference when u - s lies in
 *             the box.
 * Without a side of the kind of y0 the gradient fails (no_side). A
 * difference with a point of another kind is no slope: against a failure
 * as large as the jump into the failure band, between two failures of the
 * grading's measure only when both are graded. `eval(p, y)` evaluates point
 * p into y, counting it (false: MAXEVAL spent, reported). Returns false with
 * `report` filled. */
template<class Vec, class Eval>
bool fd_component(const Vec& x, long idx, const FdSettings& fs,
                  const FdBox& box, Eval& eval, double y0, CentreLoss kind,
                  const StencilReport<Vec>& report, double& gi)
{
  const auto k = static_cast<std::size_t>(idx);
  const double u = x(idx);
  const double a = box.lo[k], b = box.hi[k];
  const std::optional<FdStencil> st =
      fd_plan(u, a, b, fs.rule, fs.scheme, fs.fd_step);
  if (!st)   /* the gradient refused such a box first (fd_box_error) */
    throw std::logic_error(
        "GradientProvider: no finite-difference stencil fits the box of " +
        report.label(idx) + " at " + StencilReport<Vec>::num(u));
  if (st->turned) report.turned(x, idx, *st);

  const auto point = [&](double c) {
    Vec p = x;
    p(idx) = c;
    return p;
  };
  const auto same = [&](double y) { return report.classify(y) == kind; };
  /* The points evaluated whose loss is of another kind. */
  std::vector<std::pair<Vec, double>> other;
  /* The derivative of stencil `t` from its losses (y1 at t.p1, y2 at
   * t.p2), and the points it read for the overflow report. */
  const auto finish = [&](const FdStencil& t, double y1, double y2) {
    gi = fd_combine(t, fs.rule, y0, y1, y2);
    if (std::isfinite(gi)) {
      if (!other.empty()) report.mixed(x, idx, other, t, kind);
      return true;
    }
    std::vector<std::pair<Vec, double>> pts;
    if (t.kind != FdStencil::Kind::Central) pts.push_back({x, y0});
    pts.push_back({point(t.p1), y1});
    if (t.kind != FdStencil::Kind::Forward) pts.push_back({point(t.p2), y2});
    report.stencil(x, idx, pts);
    return false;
  };
  /* The 2-point difference with the point u + s of loss y1. */
  const auto two_point = [&](double s, double c, double y1) {
    FdStencil t = *st;
    t.kind = FdStencil::Kind::Forward;
    t.s = s;
    t.p1 = c;
    return finish(t, y1, 0.0);
  };
  /* The side of sign `s` (+h or -h), whose near point c = u + s, of loss
   * yn, is of the kind of y0: one-sided 3-point when its far point lies
   * in the box and is of that kind, else 2-point. */
  const auto side = [&](double s, double c, double yn) {
    const double far = u + 2.0 * s;
    if (far >= a && far <= b) {
      double yf = 0.0;
      if (!eval(point(far), yf)) return false;
      if (same(yf)) {
        FdStencil t = *st;
        t.kind = FdStencil::Kind::OneSided;
        t.s = s;
        t.p1 = c;
        t.p2 = far;
        return finish(t, yn, yf);
      }
      other.push_back({point(far), yf});
    }
    return two_point(s, c, yn);
  };
  /* The backward 2-point difference of a stencil of sign s whose near
   * point is of another kind, when u - s lies in the box. */
  const auto backward = [&](double s) {
    const double c = u + -s;
    if (c >= a && c <= b) {
      double yb = 0.0;
      if (!eval(point(c), yb)) return false;
      if (same(yb)) return two_point(-s, c, yb);
      other.push_back({point(c), yb});
    }
    report.no_side(x, idx, y0, kind, other);
    return false;
  };

  if (st->kind == FdStencil::Kind::Central) {
    double yp = 0.0, ym = 0.0;
    if (!eval(point(st->p1), yp)) return false;
    if (same(yp)) {
      if (!eval(point(st->p2), ym)) return false;
      if (same(ym)) return finish(*st, yp, ym);
      other.push_back({point(st->p2), ym});
      return side(st->h, st->p1, yp);
    }
    other.push_back({point(st->p1), yp});
    if (!eval(point(st->p2), ym)) return false;
    if (!same(ym)) {
      other.push_back({point(st->p2), ym});
      report.no_side(x, idx, y0, kind, other);
      return false;
    }
    return side(-st->h, st->p2, ym);
  }

  double y1 = 0.0;
  if (!eval(point(st->p1), y1)) return false;
  if (!same(y1)) {
    other.push_back({point(st->p1), y1});
    return backward(st->s);
  }
  if (st->kind == FdStencil::Kind::Forward) return finish(*st, y1, 0.0);
  double y2 = 0.0;
  if (!eval(point(st->p2), y2)) return false;
  if (same(y2)) return finish(*st, y1, y2);
  other.push_back({point(st->p2), y2});
  return two_point(st->s, st->p1, y1);
}

/* Whether the internal point x of `sp` lies outside its OPTIM box
 * (box_penalty.h): the rule of HPO_Host::operator() on the REAL point
 * sp.out(x) it is given, so the two agree; a spec without a SearchSpace
 * (hand-built) has its internal bounds as the box. */
template<class Vec>
std::optional<OutOfBox> out_of_box(const OptimizerSpec& sp, const Vec& x)
{
  if (sp.space) return outside_box(*sp.space, sp.out(x));
  return outside_internal_box(sp.low, sp.high, x);
}

/* The slope at a point outside the box (out_of_box): that of its score,
 * out_of_box_score() of the FAILSCORE of `sp` (the default, the largest
 * double, when it has none), in internal coordinates. */
template<class Vec>
void out_of_box_gradient(const OptimizerSpec& sp, const OutOfBox& o, Vec& g)
{
  out_of_box_slope(o, sp.failscore.value_or(std::numeric_limits<double>::max()),
                   g);
}

}  // namespace detail

/* Caller-injected analytical gradient lambda. Fills g with the
 * closed-form gradient where it can; pushes any theta indices for which no
 * analytical derivative is registered onto `needs_fd`. Returns false on
 * bad eval (clamp, NaN). The HOST captures whatever state it needs
 * (Trainer factor, Analytics, registry...) before binding this. */
template<class Vec>
using AnalyticalGradFn =
    std::function<bool(const Vec& x, Vec& g, std::vector<int>& needs_fd)>;

namespace detail {
template<class HostT, class Vec, class = void>
struct has_analytical_gradient : std::false_type {};
template<class HostT, class Vec>
struct has_analytical_gradient<HostT, Vec, std::void_t<decltype(
    std::declval<HostT&>().analytical_gradient(
        std::declval<const Vec&>(), std::declval<Vec&>(),
        std::declval<std::vector<int>&>()))>> : std::true_type {};
/* Whether HostT says why it holds no fit of the point it evaluated last
 * (HPO_Host::analytical_gradient_gap). */
template<class HostT, class = void>
struct has_analytical_gradient_gap : std::false_type {};
template<class HostT>
struct has_analytical_gradient_gap<HostT, std::void_t<decltype(
    std::declval<const HostT&>().analytical_gradient_gap())>>
    : std::true_type {};
}  // namespace detail

/* The host's analytical_gradient member as an AnalyticalGradFn when sp asks
 * for ANALYTICAL/HYBRID; nullptr for NUMERICAL. A stub host without the
 * member (unit tests of the optimiser adapters) also gets nullptr, and the
 * provider then reports the missing callback if an analytical strategy is
 * requested. */
template<class Vec, class HostT>
AnalyticalGradFn<Vec> host_analytical_grad_fn(const OptimizerSpec& sp,
                                              HostT& host)
{
  if constexpr (detail::has_analytical_gradient<HostT, Vec>::value) {
    if (sp.grad_strategy && *sp.grad_strategy != GradStrategy::Numerical)
      return [&host](const Vec& x_in, Vec& g_out,
                     std::vector<int>& nfd) -> bool {
        return host.analytical_gradient(x_in, g_out, nfd);
      };
  }
  (void)sp; (void)host;
  return nullptr;
}

/* Build a GradientProvider for `host`. Honours sp.grad_strategy
 * (default: Numerical). The eval_counter is shared with the optimizer
 * driver, so external MAXEVAL accounting works uniformly.
 *
 * `HostT` is templated so tests can substitute a stub callable in
 * place of HPO_Host<Vec,Runner>. Production call sites pass an
 * HPO_Host& and the type is deduced. The only requirement on HostT
 * is `double operator()(const Vec&)` returning the loss in real space
 * (sp.out is applied internally before the call).
 *
 * `analytical_grad_fn` is required for ANALYTICAL/HYBRID strategies and
 * ignored for NUMERICAL. Defaulted to nullptr so existing Phase-1
 * callers (which only used NUMERICAL/AUTODIFF) keep working without
 * code change. */
template<class Vec, class Runner, class HostT>
GradientProvider<Vec, Runner>
make_gradient_provider(const OptimizerSpec&     sp,
                       HostT&                   host,
                       std::atomic<unsigned long>* eval_counter,
                       unsigned long             maxeval /*0 = unbounded*/,
                       AnalyticalGradFn<Vec>    analytical_grad_fn = nullptr)
{
  const GradStrategy strat = sp.grad_strategy.value_or(GradStrategy::Numerical);
  GradientProvider<Vec, Runner> gp;
  /* Each gradient callable owns a share of gp.failure and fills it. */
  const std::shared_ptr<GradientFailureInfo> failure = gp.failure;
  const detail::StencilReport<Vec> report{&sp, failure.get()};

  /* Chain rule for the analytical path: the host differentiates the loss
   * with respect to the REAL parameters (dL/dtheta), while the optimiser
   * iterates in INTERNAL coordinates u (ln / asinh / unit-box affine), so
   * g_u[i] = dL/dtheta_i * dtheta_i/du_i. The FD paths perturb u directly
   * and need nothing. Without a bound space the legacy ln mask applies
   * (dtheta/du = theta = exp(u) on log dims). */
  auto pull_to_internal = [&sp](const Vec& x, Vec& g,
                                const std::vector<int>& skip) {
    for (long i = 0; i < g.size(); ++i) {
      bool fd = false;
      for (int k : skip) if (k == i) { fd = true; break; }
      if (fd) continue;
      double jac = 1.0;
      if (sp.space) {
        jac = sp.space->dreal_dinternal(static_cast<std::size_t>(i), x(i));
      } else if (static_cast<std::size_t>(i) < sp.log_scaled.size() &&
                 sp.log_scaled[static_cast<std::size_t>(i)]) {
        jac = std::exp(x(i));
      }
      g(i) *= jac;
    }
  };

  /* A point outside the OPTIM box (box_penalty.h; the file comment): the
   * slope of its score, for every strategy, without an evaluation. True
   * with g filled when x lies outside the box. */
  const auto outside = [&sp](const Vec& x, Vec& g) {
    const std::optional<OutOfBox> o = detail::out_of_box(sp, x);
    if (!o) return false;
    detail::out_of_box_gradient(sp, *o, g);
    return true;
  };

  gp.exhausted = [eval_counter, maxeval]() {
    return maxeval > 0 && eval_counter->load() >= maxeval;
  };

  // ---- Cost: direct host call, shared counter, return +inf on failure.
  gp.cost = [&host, &sp, eval_counter, maxeval](const Vec& x)->double {
    if (maxeval > 0 && eval_counter->load() >= maxeval)
      return std::numeric_limits<double>::infinity();
    eval_counter->fetch_add(1);
    const double y = host(sp.out(x));
    if (!std::isfinite(y)) return std::numeric_limits<double>::infinity();
    return y;
  };

  // ---- ANALYTICAL: strict mode. Defer to the caller-supplied callback;
  // any index left in `needs_fd` is a hard error pointing the user at
  // HYBRID.
  if (strat == GradStrategy::Analytical) {
    if (!analytical_grad_fn) {
      gp.gradient = [](const Vec&, Vec&)->bool {
        throw std::runtime_error(
            "GRAD_STRATEGY=ANALYTICAL requires an analytical gradient "
            "callback, but none was provided. This is an internal wiring "
            "error -- the HPO host must register an analytical gradient "
            "function before constructing the GradientProvider.");
      };
    } else {
      /* Why the host has no fit of the point (the file comment); a host
       * without the member never says. */
      std::function<std::string()> no_fit_why;
      if constexpr (detail::has_analytical_gradient_gap<HostT>::value)
        no_fit_why = [&host]() { return host.analytical_gradient_gap(); };
      gp.gradient = [analytical_grad_fn, eval_counter, maxeval,
                     pull_to_internal, failure, report, no_fit_why, outside]
                    (const Vec& x, Vec& g)->bool {
        *failure = {};
        if (outside(x, g)) return true;
        if (maxeval > 0 && eval_counter->load() >= maxeval) {
          report.budget();
          return false;
        }
        eval_counter->fetch_add(1);
        std::vector<int> needs_fd;
        if (!analytical_grad_fn(x, g, needs_fd)) {
          report.analytical("could not be formed at this point");
          return false;
        }
        pull_to_internal(x, g, needs_fd);
        if (!needs_fd.empty()) {
          /* The host has no fit of x (a candidate rejected before
           * training), or its pass could not differentiate the fit there:
           * a point without a usable gradient, not an error. */
          if (no_fit_why) {
            const std::string why = no_fit_why();
            if (!why.empty()) {
              report.untrained(x, why);
              return false;
            }
          }
          /* Otherwise the host does not say which cause applies; name them
           * all. The settings-level ones and a search parameter the
           * descriptors do not differentiate on the data (analytical_gate.h)
           * were refused at startup, so what is left is per point. Each
           * parameter by its label, as the journal names it (KEY[n]). */
          std::ostringstream oss;
          oss << "GRAD_STRATEGY=ANALYTICAL: no analytical gradient for "
              << needs_fd.size() << " parameter"
              << (needs_fd.size() == 1 ? "" : "s")
              << " at this point (";
          for (std::size_t i = 0; i < needs_fd.size(); ++i) {
            if (i) oss << ", ";
            oss << report.label(needs_fd[i]);
          }
          oss << "): a parameter without an analytical derivative, or a "
                 "loss term without an analytical derivative at this "
                 "point; GRAD_STRATEGY HYBRID fills these with finite "
                 "differences.";
          throw std::runtime_error(oss.str());
        }
        for (long i = 0; i < g.size(); ++i)
          if (!std::isfinite(g(i))) {
            report.analytical("is not finite at this point");
            return false;
          }
        return true;
      };
    }
    const auto grad = gp.gradient;
    if (!analytical_grad_fn) {
      gp.gradient_at = [grad](const Vec& x, double, Vec& g) {
        return grad(x, g);
      };
      return gp;
    }
    /* The caller's loss at x (the file comment): a failed one, and a
     * graded failure under FAILSCORE_GRADING ON, fail the gradient before
     * the analytical pass -- the analytical gradient there is that of the
     * loss before the failure (a skip cap's uncapped loss, the terms that
     * did not fail), not of the FAILSCORE or the graded loss the optimiser
     * sees, and a search took it for a slope and converged on it. A point
     * outside the box has the slope of its score. */
    gp.gradient_at = [grad, failure, report, outside](const Vec& x, double fx,
                                                      Vec& g) {
      *failure = {};
      if (outside(x, g)) return true;
      switch (report.classify(fx)) {
        case detail::CentreLoss::Failed:
          report.failed_point(x, fx);
          return false;
        case detail::CentreLoss::Graded:
          report.graded_point(x, fx);
          return false;
        case detail::CentreLoss::Usable:
          break;
      }
      return grad(x, g);
    };
    return gp;
  }

  /* The finite differences of NUMERICAL and HYBRID (fd_stencil.h). */
  const auto setup = std::make_shared<detail::FdSetup>(sp);
  const detail::FdBox box(sp);

  // ---- HYBRID: analytical fills what it can; the remaining indices --
  // every index when the analytical pass forms no derivative at x -- get
  // the finite differences of NUMERICAL, varying only those indices. The
  // shared eval_counter accounts for them so MAXEVAL stays correct.
  if (strat == GradStrategy::Hybrid) {
    if (!analytical_grad_fn) {
      gp.gradient = [](const Vec&, Vec&)->bool {
        throw std::runtime_error(
            "GRAD_STRATEGY=HYBRID requires an analytical gradient "
            "callback, but none was provided. This is an internal wiring "
            "error.");
      };
      gp.gradient_at = [](const Vec&, double, Vec&)->bool {
        throw std::runtime_error(
            "GRAD_STRATEGY=HYBRID requires an analytical gradient "
            "callback, but none was provided. This is an internal wiring "
            "error.");
      };
      return gp;
    }

    /* `fx`: the loss at x, the caller's (gradient_at) or gradient()'s. */
    const auto hybrid = [&host, &sp, eval_counter, maxeval,
                         analytical_grad_fn, pull_to_internal, failure,
                         report, setup, box, outside]
                        (const Vec& x_in, double fx, Vec& g)->bool {
      *failure = {};
      /* Any parameter may need finite differences (the analytical pass
       * routes every one there when it cannot be formed), so every box
       * must hold the stencil (validate_hpo_setup refuses one that does
       * not). */
      setup->require(sp);
      box.require(x_in.size());
      if (outside(x_in, g)) return true;
      /* A coordinate within rounding outside its box is evaluated on the
       * bound: it is differenced there. */
      const Vec x = box.onto(x_in);
      /* The caller's loss at x, before the analytical pass: a failed one
       * fails the gradient whatever is differenced (as under ANALYTICAL:
       * the analytical gradient there is not that of the loss the
       * optimiser sees, and with nothing differenced a search converged on
       * FAILSCORE). At a graded failure under FAILSCORE_GRADING ON every
       * parameter is differenced and no analytical pass is run: the
       * analytical gradient there does not see the grading (a band
       * measure, pc_fail_grading.h) -- it is the gradient of the terms
       * that did not fail -- and a search that took it converged inside
       * the failure region; the differences give the graded slope, or a
       * flat failure region (below). */
      const detail::CentreLoss cls = report.classify(fx);
      const bool graded_at_x = cls == detail::CentreLoss::Graded;
      if (cls == detail::CentreLoss::Failed) {
        report.failed_point(x, fx);
        return false;
      }
      const auto over = [&]() {
        if (maxeval > 0 && eval_counter->load() >= maxeval) {
          report.budget();
          return true;
        }
        return false;
      };
      if (over()) return false;

      std::vector<int> needs_fd;
      if (graded_at_x) {
        g.set_size(x.size());
        for (long i = 0; i < x.size(); ++i) {
          g(i) = 0.0;
          needs_fd.push_back(static_cast<int>(i));
        }
      } else {
        eval_counter->fetch_add(1);
        if (!analytical_grad_fn(x, g, needs_fd)) {
          report.analytical("could not be formed at this point");
          return false;
        }
        pull_to_internal(x, g, needs_fd);
      }
      /* Every parameter in needs_fd -- all of them when the analytical
       * gradient cannot be formed here (a setup it does not cover, a
       * candidate the bump gate rejected before training, a PC_LAMMPS
       * chain that failed) -- is differenced. HYBRID used to return g = 0
       * in that case, so a gradient search stopped at its start as
       * "converged". */
      const auto eval = [&](const Vec& p, double& y) {
        if (over()) return false;
        eval_counter->fetch_add(1);
        y = host(sp.out(p));
        return true;
      };
      if (!needs_fd.empty()) {
        /* The loss at x decides the kind of every difference (the file
         * comment); a failed one ended the gradient above. */
        const double y0 = fx;
        for (int idx : needs_fd) {
          double gi = 0.0;
          if (!detail::fd_component(x, idx, setup->s, box, eval, y0, cls,
                                    report, gi))
            return false;
          g(idx) = gi;
        }
        /* A graded failure whose every difference is 0 gives no way out. */
        if (cls == detail::CentreLoss::Graded) {
          bool any = false;
          for (int idx : needs_fd)
            if (g(idx) != 0.0) any = true;
          if (!any) {
            report.flat(x, y0);
            return false;
          }
        }
      }

      // Sanity check: every index should now be finite (the analytical
      // pass may have left bad entries even outside needs_fd).
      for (long i = 0; i < g.size(); ++i)
        if (!std::isfinite(g(i))) {
          report.analytical("is not finite at this point");
          return false;
        }
      return true;
    };
    /* gradient() without the caller's loss evaluates the loss at x first
     * (counted, HPOLog::record_hybrid_centre): its kind decides the whole
     * gradient (the file comment) -- a failure ends it, a graded failure
     * is differenced in every parameter --, and the analytical pass then
     * differentiates the host's evaluation of x itself. */
    gp.gradient = [hybrid, &host, &sp, eval_counter, maxeval, failure, report,
                   setup, box, outside](const Vec& x, Vec& g) {
      *failure = {};
      setup->require(sp);
      box.require(x.size());
      if (outside(x, g)) return true;
      if (maxeval > 0 && eval_counter->load() >= maxeval) {
        report.budget();
        return false;
      }
      eval_counter->fetch_add(1);
      const double y0 = host(sp.out(box.onto(x)));
      HPOLog::instance().record_hybrid_centre();
      return hybrid(x, y0, g);
    };
    gp.gradient_at = [hybrid](const Vec& x, double fx, Vec& g) {
      return hybrid(x, fx, g);
    };
    return gp;
  }

  // ---- NUMERICAL: finite differences in INTERNAL coordinates with the
  // step rule of PARAM fd_kernel -- inhouse, h = fd_step * max(|u|, 1)
  // (in the unit box of OPTIM_SCALE AUTO exactly fd_step of every width,
  // what PARAM fd_step is documented to mean there), or Ceres' rule,
  // max(2^-26, fd_step * |u|), the default under OPTIM_SCALE NONE so that
  // runs keep their pre-1.5 finite differences (unsuitable in the unit
  // box, where the step collapses to 1.5e-8 next to u = 0, every lower
  // bound). The loss at x -- the caller's, else evaluated first -- then
  // each dimension's stencil, central or forward, one-sided into the box
  // within a step of a bound, of the kind of the loss at x
  // (fd_component): both old kernels' points and arithmetic in the
  // interior (fd_stencil.h).
  /* `fx`: the caller's loss at x (gradient_at), the one the optimiser was
   * given, or nullptr (gradient): evaluated here first. */
  const auto numerical = [&host, &sp, eval_counter, maxeval, failure, report,
                          setup, box, outside]
                         (const Vec& x_in, const double* fx, Vec& g)->bool {
    *failure = {};
    /* An invalid setting or a box narrower than the stencil
     * (validate_hpo_setup refuses both): before any evaluation. */
    setup->require(sp);
    box.require(x_in.size());
    if (outside(x_in, g)) return true;
    /* A coordinate within rounding outside its box is evaluated on the
     * bound: it is differenced there. */
    const Vec x = box.onto(x_in);
    const auto eval = [&](const Vec& p, double& y) {
      if (maxeval > 0 && eval_counter->load() >= maxeval) {
        report.budget();
        return false;
      }
      eval_counter->fetch_add(1);
      y = host(sp.out(p));
      return true;
    };
    /* The loss AT x, before any stencil point: the caller's, else
     * evaluated here (the bad-evaluation contract of both old kernels,
     * Ceres' evaluating the residual at x first, with HPO's notion of a
     * failed evaluation). A failure the gradient cannot difference ends it
     * here, and no stencil point is evaluated; otherwise its kind --
     * usable, or a graded failure -- is the kind of every difference (the
     * file comment). A one-sided or forward stencil uses the loss; a
     * central one does not. */
    double y0 = 0.0;
    if (fx) y0 = *fx;
    else if (!eval(x, y0)) return false;
    const detail::CentreLoss cls = report.classify(y0);
    if (cls == detail::CentreLoss::Failed) {
      report.centre(x, y0);
      return false;
    }
    const long n = x.size();
    g.set_size(n);
    for (long idx = 0; idx < n; ++idx) {
      double gi = 0.0;
      if (!detail::fd_component(x, idx, setup->s, box, eval, y0, cls, report,
                                gi))
        return false;
      g(idx) = gi;
    }
    /* A graded failure whose every difference is 0 gives no way out. */
    if (cls == detail::CentreLoss::Graded) {
      bool any = false;
      for (long idx = 0; idx < n; ++idx)
        if (g(idx) != 0.0) any = true;
      if (!any) {
        report.flat(x, y0);
        return false;
      }
    }
    return true;
  };
  gp.gradient = [numerical](const Vec& x, Vec& g) {
    return numerical(x, nullptr, g);
  };
  gp.gradient_at = [numerical](const Vec& x, double fx, Vec& g) {
    return numerical(x, &fx, g);
  };
  return gp;
}

}  // namespace tadah::hpo

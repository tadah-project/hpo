#pragma once
#ifndef TADAH_HPO_PC_FAIL_GRADING_H
#define TADAH_HPO_PC_FAIL_GRADING_H

/* Graded ("smooth") FAILSCORE -- the failure-region objective.
 * Design: INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md (sections 4.2, 4.6-4.9).
 *
 * A constant FAILSCORE gives the optimiser a gradient-free plateau with two
 * absorbing annealer states (all-accept step blow-up on an identical plateau;
 * zero-accept V-floor collapse on a mixed cliff), kills PT ladders and zeroes
 * gradients. Instead, every FAILED evaluation returns a value inside the
 * reserved band [0.999*FS, FS), graded by a per-band violation measure, so
 * moves toward feasibility are strictly downhill while every existing
 * failure-detection semantic is preserved:
 *   - all failure losses stay >= 0.999*FS  => loss_is_finite() still treats
 *     them as failures (T0 probes, PA culling, journals unchanged);
 *   - all stay strictly < FS               => the aggregate clamp never
 *     flattens them, and any feasible loss (<< 0.999*FS) always wins.
 *
 * Severity BANDS (lexicographic, worst on top -- the "repair path"): one
 * evaluation gets ONE band (worst-wins) graded by that band's natural measure:
 *
 *   D  sick potential   failprobe gate fired; LammpsTadah ctor threw;
 *                       bad-pot skip caps            graded by S(theta)
 *   C  did not complete deadline SIGKILL / worker died / hang
 *                                                    graded by S(theta)
 *   B  completed-failed in-script timeout, --failvar > 0, unparseable/NaN
 *                       outvars, observable failures graded by failvar sum
 *                                                    + failed fraction
 *   A  range-rejected   hot-var out of [min,max]     graded by normalized
 *                                                    distance to the box
 *
 * Infrastructure failures (LAMMPS instance CREATE_FAIL, crash-loops,
 * eval-thread exceptions) are DELIBERATELY ungraded -- they are not
 * properties of theta; grading them would have the optimiser chase cluster
 * noise. They keep today's constant semantics.
 *
 * L_fail arithmetic (FS finite -- validated at startup when grading is ON):
 *
 *   W       = 0.001 * FS                    (width of the whole failure range)
 *   sigma(v)= v / (v + v0)  in [0, 1)       (monotone, bounded, overflow-free)
 *   L_fail  = 0.999*FS + W * (offset_band + 0.25 * sigma(v))
 *   offsets: A = 0.00, B = 0.25, C = 0.50, D = 0.75
 *
 * Only the SIGN of loss differences drives annealer acceptance, so v0 (the
 * half-saturation scale) is uncritical; ordering, not magnitude, carries the
 * signal. v < 0 is clamped to 0; non-finite v saturates the band (top). */

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>

namespace tadah {
namespace hpo {

enum class FailBand : int {
  NONE = 0,   /* evaluation feasible (or grading disabled)          */
  A = 1,      /* range-rejected (hot-var out of box)                */
  B = 2,      /* sims completed but reported failure                */
  C = 3,      /* sims did not complete (hang / kill / worker death) */
  D = 4,      /* potential sick before/independent of the sims      */
};

inline const char* fail_band_name(FailBand b) {
  switch (b) {
    case FailBand::A: return "A/range";
    case FailBand::B: return "B/completed-failed";
    case FailBand::C: return "C/not-completed";
    case FailBand::D: return "D/sick-potential";
    default:          return "none";
  }
}

/* sigma: [0, inf] -> [0, 1). Non-finite or huge v saturates to 1 (band top
 * minus epsilon); negative v clamps to 0. The saturation epsilon is 1e-9,
 * not smaller: the cross-band strict ordering at saturation rests on
 * 0.25*W*eps > ulp(FS), i.e. eps > ~9e-13 -- 1e-9 keeps ~1000 ulps of
 * margin for any FS the startup validation admits. */
inline double fail_sigma(double v, double v0) {
  if (!(v > 0.0)) return 0.0;                       /* v <= 0 or NaN */
  if (!std::isfinite(v)) return 1.0 - 1e-9;
  if (!(v0 > 0.0) || !std::isfinite(v0)) v0 = 1.0;
  const double s = v / (v + v0);
  return std::min(s, 1.0 - 1e-9);
}

/* The graded failure loss. Preconditions: fs finite and > 0 (validated at
 * startup); band != NONE. Guarantees: 0.999*fs <= L < fs, monotone
 * (strictly, until sigma saturates) in v within a band, and every band-X
 * value < every band-(X+1) value. */
inline double graded_failscore(double fs, FailBand band, double v, double v0) {
  const double W = 1.0e-3 * fs;
  double off = 0.0;
  switch (band) {
    case FailBand::A: off = 0.00; break;
    case FailBand::B: off = 0.25; break;
    case FailBand::C: off = 0.50; break;
    case FailBand::D: off = 0.75; break;
    default:          return fs;   /* not a failure -- caller bug guard */
  }
  return 0.999 * fs + W * (off + 0.25 * fail_sigma(v, v0));
}

/* Compact, copyable per-eval failure record (travels inside EvalResult
 * through the writer queue). band == 0 => the eval is feasible. */
struct PcFailInfo {
  int band = 0;          /* FailBand as int; 0 = none                      */
  double vA = 0.0;       /* band-A measure: summed normalized OOR distance */
  double vB = 0.0;       /* band-B measure: failvar sum + failed fraction  */
  double S = -1.0;       /* failprobe scalar; < 0 = unavailable            */
  double vD = -1.0;      /* explicit band-D measure (skip caps); < 0 = use S */
  bool infra = false;    /* ungradeable infrastructure failure             */
  bool gated = false;    /* failprobe gate fired                           */
};

/* The final graded loss for a failed eval (info.band != 0). s0 = sigma
 * half-saturation for the S measure (bands C/D). */
inline double graded_eval_loss(const PcFailInfo& info, double fs, double s0) {
  const FailBand b = static_cast<FailBand>(info.band);
  switch (b) {
    case FailBand::A: return graded_failscore(fs, b, info.vA, 1.0);
    case FailBand::B: return graded_failscore(fs, b, info.vB, 1.0);
    case FailBand::C:
    case FailBand::D: {
      if (info.infra && b == FailBand::C)
        return graded_failscore(fs, b,
                                std::numeric_limits<double>::infinity(), s0);
      if (b == FailBand::D && info.vD >= 0.0)
        return graded_failscore(fs, b, info.vD, 1.0);
      const double v = (info.S >= 0.0)
                           ? info.S
                           : std::numeric_limits<double>::infinity();
      return graded_failscore(fs, b, v, s0);
    }
    default: return fs;
  }
}

/* Per-evaluation failure classification, accumulated by the fan-out threads
 * (guarded; the fan-out is concurrent). Worst band wins; each band
 * accumulates its own violation measure. S(theta) -- the failprobe scalar --
 * is recorded once per eval and used as the measure for bands C and D. */
class PcFailState {
public:
  /* Record a failure of `band` with violation contribution `dv` (band A/B
   * measures are additive across sims; C/D take their measure from S). */
  void record(FailBand band, double dv = 0.0) {
    std::lock_guard<std::mutex> lk(mtx_);
    band_ = std::max(band_, static_cast<int>(band));
    if (band == FailBand::A) vA_ += std::max(dv, 0.0);
    if (band == FailBand::B) vB_ += std::max(dv, 0.0);
  }

  /* Record an ungradeable (infrastructure) failure: the eval must fail, but
   * with a directionless constant (top of band C -- ordered like a
   * non-completion, deliberately carrying no gradient signal). */
  void record_infra() {
    std::lock_guard<std::mutex> lk(mtx_);
    band_ = std::max(band_, static_cast<int>(FailBand::C));
    infra_ = true;
  }

  void set_S(double S) {
    std::lock_guard<std::mutex> lk(mtx_);
    S_ = S;
    s_valid_ = std::isfinite(S) && S >= 0.0;
  }
  void set_gated() {
    std::lock_guard<std::mutex> lk(mtx_);
    gated_ = true;
    band_ = std::max(band_, static_cast<int>(FailBand::D));
  }

  bool failed() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return band_ != 0;
  }
  bool gated() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return gated_;
  }
  FailBand band() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return static_cast<FailBand>(band_);
  }
  double S() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return s_valid_ ? S_ : -1.0;
  }

  /* Compact snapshot for transport inside EvalResult. */
  PcFailInfo snapshot() const {
    std::lock_guard<std::mutex> lk(mtx_);
    PcFailInfo info;
    info.band = band_;
    info.vA = vA_;
    info.vB = vB_;
    info.S = s_valid_ ? S_ : -1.0;
    info.infra = infra_;
    info.gated = gated_;
    return info;
  }

private:
  mutable std::mutex mtx_;
  int band_ = 0;
  double vA_ = 0.0;
  double vB_ = 0.0;
  double S_ = -1.0;
  bool s_valid_ = false;
  bool gated_ = false;
  bool infra_ = false;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_PC_FAIL_GRADING_H

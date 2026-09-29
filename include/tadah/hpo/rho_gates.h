#ifndef TADAH_HPO_RHO_GATES_H
#define TADAH_HPO_RHO_GATES_H

// Density gates of a bare DM_REAM many-body term, and the density report.
//
// RHOGATE (RHOFORM SIGNED only) rejects a candidate whose pair density
// rho_raw(r) dips below -tol * rho_max on [R_LO, RCUTMB]; RHOBARGATE (any
// form) rejects one whose monatomic lattice density rhobar(a) is not
// strictly decreasing on a in [f_lo a0, f_hi a0]. HPO_Host::operator()
// checks both right after the candidate context is built (update_context)
// and before any training work or MPI traffic; a reject is a graded band-D
// failure with the BUMPGATE bookkeeping (bump_support_gate.h). When any
// gate fails every configured gate is evaluated and the candidate is graded
// by the sum of their measures, so a move that shrinks any violation is
// downhill; the reject is counted for the first failing gate.
//
// Every density value comes from models::ReamDensity (MODELS, out of line),
// so a gate sees exactly the density DM_REAM trains with; the cutoff is
// built per call as DC_Selector builds it, factory(RCTYPEMB, RCUTMBMAX).
// The candidate context is read through Context::has_stored / stored_value
// only: a gate never materialises a blueprint default, so the context the
// trainer gets -- and every potential written from it -- is the same with
// and without the gates. The verdicts are pure functions of the candidate
// and the run's settings: the tables are built once (HPO_Host constructor,
// validate_hpo_setup) and only read afterwards.
//
// The density report (hpo_run.log POTENTIAL SETUP and RESULT) summarises
// the same scans for the -c potential, each WARM seed and the best.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/models/eam/eam_lattice.h>

namespace tadah {
namespace hpo {
namespace detail {

/** True when the many-body term of @p ctx is a bare DM_REAM block: INITMB
 *  true and TYPEMB DM_REAM (a DM_mJoin is not bare). Stored values only. */
bool context_is_bare_ream(const core::Context &ctx);

/** \brief Range of the many-body cutoff RCUTMBMAX a candidate can carry.
 *
 *  A knot channel's cutoff is its largest knot (rcut_derivation.h), so over
 *  CGRIDMB: min = max_n(opt_n ? lowbound_n : c_n) and
 *  max = max_n(opt_n ? highbound_n : c_n) (an OPTIM RCUTMB is overridden).
 *  A bump channel's is RCUTMB, the range derive_binding_rcut gives the
 *  BUMPGATE: the max over the per-pair list with each OPTIM'd entry at its
 *  bounds. An OPTIM key counts in any case (HPOTargets upper-cases it). */
struct ReachableRcut {
  double min_reachable = 0.0;
  double max_reachable = 0.0;
  bool   knot = false;   ///< a knot channel (the cutoff is the largest knot)
};
ReachableRcut knot_aware_reachable_rcut(const core::Context &ctx,
                                        const HPOTargets &trg);

/** \brief RHOGATE's verdict on one density.
 *
 *  FAIL iff rho_min < -(tol rho_max) on the grid, or saturated. The measure
 *  v is computed only on FAIL (0 on a pass):
 *    v = min(1e8, v_peak + v_int),
 *    v_peak = max(0, (-rho_min - tol rho_max) / rho_max),
 *    v_int  = sum_i max(0, -rho_i - tol rho_max) w_i / (rho_max (r_hi - r_lo))
 *  with trapezoid weights w_i. Saturated (v = DBL_MAX, the top of band D):
 *  rho_max <= 0, a non-finite rho_i, a pair weight <= 0 or non-finite, or
 *  an empty window (r_hi <= r_lo). */
struct RhoGateResult {
  bool        fail = false;
  bool        saturated = false;
  /** The density could not be built or validated (ReamDensity::from_context
   *  / validate, or the cutoff): no verdict. The candidate then fails in
   *  training exactly as it would without the gate. */
  bool        abstain = false;
  double      v = 0.0;
  std::string reason;          ///< saturated / abstain: why
  double      r_lo = 0.0, r_hi = 0.0;
  std::size_t npts = 0;
  double      rho_min = 0.0, r_min = 0.0, rho_max = 0.0, r_max = 0.0;
};

/** \brief RHOBARGATE's verdict on one density.
 *
 *  FAIL iff rhobar_{k+1} >= rhobar_k for some k, or saturated. v (on FAIL
 *  only) = min(1e8, sum_k max(0, rhobar_{k+1} - rhobar_k) / max_k
 *  |rhobar_k|): a knife edge with a flat step fails with v = 0, the band-D
 *  floor. Saturated (v = DBL_MAX): max_k |rhobar_k| == 0, a non-finite
 *  value, or a pair weight <= 0 or non-finite. */
struct RhoBarGateResult {
  bool        fail = false;
  bool        saturated = false;
  bool        abstain = false;   ///< as RhoGateResult::abstain
  double      v = 0.0;
  std::string reason;
  double      r_hi = 0.0;
  std::size_t rises = 0;         ///< rising steps of the grid
  double      first_rise = 0.0;  ///< a/a0 at the first rising step
};

/** \brief RHOGATE of a run: settings and arming.
 *
 *  Armed by the OPTIM dimensions on HPOTargets::density_gate_keys()
 *  (AMPGRIDMB, CGRIDMB, SGRIDMB, RCUTMB: they change rho_raw or the
 *  window); every candidate is evaluated. A gate without one cannot act
 *  (every candidate would have the -c density and its verdict): HPOTargets
 *  refuses it at the gate's line. */
struct RhoGateTable {
  RhoGateSpec                  spec;
  std::vector<std::string>     armed_on;   ///< labels of the arming dims
};

/** Largest shell radius RHOBARGATE enumerates, in lattice constants:
 *  max_reachable / (f_lo a0) above it is refused (an f_lo typo such as
 *  0.0075 for 0.75 would enumerate ~1e8 lattice points). */
constexpr double kRhoBarMaxShellRadius = 20.0;

/** \brief RHOBARGATE of a run: settings, the lattice and the a-grid (built
 *  once) and arming (as RhoGateTable).
 *
 *  The shells are enumerated once, for the largest reachable RCUTMB at the
 *  smallest lattice constant (limit max_reachable / (f_lo a0) (1 + 1e-12),
 *  at most kRhoBarMaxShellRadius); each candidate sums the shells with
 *  d_s a_k <= its own RCUTMB. The grid
 *  is a_k = a0 (k step + f_lo), step = (f_hi - f_lo) / (NPTS - 1), the last
 *  point a0 f_hi exactly (numpy.linspace order). */
struct RhoBarGateTable {
  RhoBarGateSpec                          spec;
  std::vector<models::eam_lattice::Shell> shells;
  std::vector<double>                     f;   ///< a/a0 grid
  std::vector<double>                     a;   ///< a0 * f
  std::vector<std::string>                armed_on;
};

/** \brief The DM_REAM density settings of a run (a bare DM_REAM many-body
 *  term): the configured gates, the reachable cutoff, and the knot flags
 *  update_candidate_context needs. Also carries the density report. */
struct DensityGates {
  bool                           knot2b = false;
  bool                           knotmb = false;
  ReachableRcut                  reach;
  std::optional<RhoGateTable>    rho;      ///< RHOGATE configured
  std::optional<RhoBarGateTable> rhobar;   ///< RHOBARGATE configured

  bool gated() const { return rho.has_value() || rhobar.has_value(); }
};

/** Build the density settings of a run from the -c configuration @p master
 *  and the parsed targets: arming and the RHOBARGATE shells and grid (a
 *  gate that no OPTIM dimension arms is std::logic_error: HPOTargets
 *  refuses it at the gate's line). Throws std::runtime_error when a gate is
 *  configured but @p master is not a bare DM_REAM (validate_hpo_setup gives
 *  the full message first) or the lattice cannot be enumerated (the shell
 *  radius is not finite, or above kRhoBarMaxShellRadius: a message naming
 *  f_lo). */
DensityGates build_density_gates(const core::Context &master,
                                 const HPOTargets &trg);

/** RHOGATE on the candidate context @p cand: the full evaluation (the
 *  density, the pair weights unless @p check_weights is false, the scan).
 *  Never throws for a candidate (a density that cannot be built abstains). */
RhoGateResult eval_rho_gate(const RhoGateTable &t, const core::Context &cand,
                            bool check_weights = true);

/** RHOBARGATE on the candidate context @p cand (as eval_rho_gate). */
RhoBarGateResult eval_rhobar_gate(const RhoBarGateTable &t,
                                  const core::Context &cand,
                                  bool check_weights = true);

/** The configured density gates' verdicts on one candidate (nullopt: that
 *  gate is not configured); every configured gate is evaluated. */
struct DensityGateVerdict {
  std::optional<RhoGateResult>    rho;
  std::optional<RhoBarGateResult> rhobar;
};
DensityGateVerdict check_density_gates(const DensityGates &g,
                                       const core::Context &cand);

/** \brief A pre-training reject: the gate it is counted for, the band-D
 *  measure and the journal text ("... gate reject (...)"). */
struct GateReject {
  GateKind    kind = GateKind::Bump;
  double      v = 0.0;
  std::string what;
};

/** The combined verdict of every pre-training gate: nullopt when all pass.
 *  BUMPGATE fails when @p v_bump > 0 (its text is @p bump_what), then
 *  RHOGATE, then RHOBARGATE (an abstaining gate passes). The reject is
 *  attributed to the first failing gate; a lone failure keeps its own v and
 *  text, several are graded by v = min(DBL_MAX, the sum of their v) and the
 *  text names the others ("; also failing: ..."). */
std::optional<GateReject> combine_gate_verdicts(double v_bump,
                                                const std::string &bump_what,
                                                const DensityGates &g,
                                                const DensityGateVerdict &d);

/** "rho-sign gate reject (rho_min/rho_max = -1.2290e-04 at r = 5.2840 A,
 *  tol 1.0000e-05, v = 1.1650e-04)" (a saturated reject names its reason). */
std::string describe_rho_reject(const RhoGateResult &r, const RhoGateSpec &spec);

/** "lattice-rhobar gate reject (373 rises over a/a0 [0.75, 1.1] on bcc,
 *  first at a/a0 = 0.7616, v = 2.5390e+00)". */
std::string describe_rhobar_reject(const RhoBarGateResult &r,
                                   const RhoBarGateSpec &spec);

/** \brief One user seed of the gate checks: a CONFIG or WARM point of a
 *  resolved INIT chain (OUTER, PRESTAGE, PRESTAGE.INNER), or the -c
 *  potential. */
struct GateSeed {
  std::vector<double>      point;     ///< real space, the optimap order
  std::vector<std::string> origins;   ///< InitOrigin::name() of each INIT entry
  bool                     c_pot = false;        ///< the -c potential (the CONFIG point)
  bool                     in_seed_set = false;  ///< a start (CONFIG / WARM)
  /** InitOrigin::name() of each RESTART entry (RestartSpec::seed_points). */
  std::vector<std::string> restart_origins;

  /** "-c pot (INIT CONFIG)", "INIT WARM seed.tadah", "RESTART WARM
   *  seed.tadah", "-c pot". */
  std::string label() const;
};

/** The seeds the gates check at startup: the -c potential first (always;
 *  in the seed set when some chain starts from CONFIG or from a WARM point
 *  equal to it), then every other distinct CONFIG/WARM point of the INIT
 *  chains and of the RESTART chain (RestartSpec::seed_points: a restart
 *  round starts from them). A chain whose provenance does not match its
 *  points (hand-built specs) contributes none. RANDOM, LHS and SOBOL points
 *  are ordinary candidates. */
std::vector<GateSeed> collect_gate_seeds(const HPOTargets &trg);

/** A copy of @p base with the candidate @p point written in, exactly as
 *  HPO_Host::update_context writes a candidate (update_candidate_context). */
core::Context seed_context(const core::Context &base, const HPOTargets &trg,
                           const std::vector<double> &point,
                           const DensityGates &g);

/** Startup (validate_hpo_setup): the semantic checks of RHOGATE and
 *  RHOBARGATE -- a bare DM_REAM (DM_mJoin refused), RHOGATE under RHOFORM
 *  SIGNED only, R_LO and the first RHOBARGATE shell at a = f_hi a0 inside
 *  the smallest reachable RCUTMB -- then the seed checks: every seed is
 *  evaluated through seed_context and the per-candidate functions. A
 *  failing seed is an error under SEED ERROR and a WARNING under SEED
 *  WARN; a density pair weight no candidate can make positive (WATOM fixed
 *  at <= 0, or an OPTIM WATOM box not reaching above 0) is an error under
 *  either (every evaluation would be rejected). Logs one INFO state line
 *  per gate. No-op
 *  when neither gate is configured. */
void validate_density_gates(const core::Context &train_ctx,
                            const HPOTargets &trg);

/** \brief The density of one potential for the report: the form, rho_raw
 *  on the report window (RHOGATE's R_LO and NPTS when configured, else
 *  [0.15 RCUTMB, RCUTMB] and 20001 points), the largest finite-difference
 *  slope, the monotonicity test, the pair weights, and each configured
 *  gate's verdict. */
struct DensitySummary {
  bool        derivable = false;
  std::string reason;             ///< !derivable: why
  std::string form;               ///< "SIGNED" | "NONNEG" | "MONOTONE"
  /** "W = 0.5", "W in [..]"; "W = 1 (no WATOM)" for a context without
   *  WATOM, which tadah hpo never evaluates (it inserts WATOM = Z, the
   *  atomic number, when the configuration has none). */
  std::string weights;
  double      r_lo = 0.0, r_hi = 0.0;
  std::size_t npts = 0;
  bool        finite = true;
  double      rho_min = 0.0, r_min = 0.0, rho_max = 0.0, r_max = 0.0;
  double      max_drho = 0.0, r_max_drho = 0.0;   ///< largest (rho_{i+1} - rho_i) / dr, at r_i
  /** max_i(rho_{i+1} - rho_i) <= 8 eps max_i sum_n |a_n T_n f_c|: the
   *  tolerance is relative to the term sum, not to rho (MONOTONE is
   *  non-increasing to rounding only). */
  bool        non_increasing = false;
  std::optional<RhoGateResult>    rhogate;   ///< RHOGATE configured
  double                          tol = 0.0;
  std::optional<RhoBarGateResult> rhobar;    ///< RHOBARGATE configured
  RhoBarGateSpec                  rhobar_spec;
};

/** The density summary of the candidate context @p cand. Never throws: a
 *  context that is not a complete DM_REAM density gives derivable = false
 *  and the reason. */
DensitySummary summarise_density(const DensityGates &g,
                                 const core::Context &cand);

/** \brief One row group of the startup density report. */
struct DensityReportEntry {
  std::string    label;   ///< "-c pot", "WARM 1", ...
  std::string    note;    ///< appended to the rho(r) row ("" for none)
  DensitySummary summary;
};

/** The startup density report of a run with a bare DM_REAM many-body term:
 *  the -c potential, then each distinct WARM seed as evaluated (its OPTIM
 *  dimensions from the seed, the rest from -c). Never throws. */
std::vector<DensityReportEntry> startup_density_report(const core::Context &ctx,
                                                       const HPOTargets &trg);

/** "RHOFORM SIGNED -- rho = W f_c sum a_n psi_n, signed amplitudes (no sign
 *  guarantee)" and its NONNEG / MONOTONE counterparts. */
std::string describe_density_form(const std::string &form);

/** The report row of rho_raw(r): window, extrema, rel, max rho',
 *  monotonicity, W and the RHOGATE verdict ("not derivable: ..." when the
 *  summary is not). */
std::string format_density_rho(const DensitySummary &s);

/** The report row of rhobar(a) (RHOBARGATE configured). */
std::string format_density_rhobar(const DensitySummary &s);

} // namespace detail
} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_RHO_GATES_H

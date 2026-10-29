#ifndef TADAH_HPO_STARTUP_VALIDATION_H
#define TADAH_HPO_STARTUP_VALIDATION_H

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/hpo/hpo_targets.h>

#include <cstddef>
#include <limits>
#include <set>
#include <string>

namespace tadah {
namespace mlip { class StructureDB; }
namespace hpo {

/** Derive the total model weight count K from the training configuration
 *  alone (no data): build the Trainer on a copy of the context and query
 *  the basis-function column count, or SBASIS for a non-linear M_KRR
 *  kernel, whose basis vectors training draws from the data
 *  (mlip::model_weight_count). A configuration whose model does not
 *  build throws (the construction error; models::DensityFormError for an
 *  invalid DM_REAM density), and so does one whose fit draws basis
 *  vectors (mlip::fit_draws_basis: a non-linear M_KRR kernel, or M_KRR
 *  implementation 2 with any kernel) without an SBASIS it takes
 *  (mlip::checked_sbasis): every evaluation would fail on it. */
std::size_t derive_weight_count(const core::Context &train_ctx);

/** Raw DM_REAM embedding value at zero density, F_raw(0) = sum_m w_m g_m(0),
 *  of the potential the training configuration describes: the constant per
 *  atom that EMBF0 ZERO removes, or the isolated-atom embedding energy under
 *  EMBF0 FREE. Journal only: never feeds an evaluated number. */
struct RawEmbeddingF0 {
  enum class Kind {
    Value,          ///< `value` holds F_raw(0); `source` names the weights
    PerEvaluation,  ///< no fixed weight vector: fitted on every evaluation
    NotApplicable   ///< `reason` says why
  };
  Kind        kind  = Kind::NotApplicable;
  double      value = std::numeric_limits<double>::quiet_NaN();
  std::string source;  ///< "FIXWEIGHT" | "WEIGHTS" (Kind::Value)
  std::string reason;  ///< Kind::NotApplicable
};

/** Derive F_raw(0) from the training configuration alone, on a probe Trainer
 *  built from a COPY of the context (as derive_weight_count does). Weights:
 *  the provisioned FIXWEIGHT when FIXINDEX pins every weight, else a stored
 *  WEIGHTS vector of the model size, else Kind::PerEvaluation.
 *  Kind::NotApplicable unless the context has a DM_REAM block and a model
 *  linear in the descriptor (BF_Linear / Kern_Linear; with NORM the trained
 *  weights are raw-descriptor weights too) and
 *  the weight vector matches the descriptor size. Throws only
 *  models::DensityFormError (an invalid density form of the potential, a
 *  configuration error); every other failure is Kind::NotApplicable. */
RawEmbeddingF0 derive_raw_embedding_f0(const core::Context &train_ctx);

/** Whether FIXINDEX of @p ctx pins every model weight in a way that an
 *  evaluation with training data provisions instead of fitting
 *  (HPO_Host's fast path): FIXINDEX covers every weight
 *  (mlip::classify_weight_source gives AllFixedData), NORM is false and
 *  the model draws no basis vectors from the training data (a non-linear
 *  M_KRR kernel). An evaluation of a run with training data fits the
 *  model otherwise; one without training data never does. */
bool fixindex_pins_every_weight(const core::Context &ctx);

/** The element set of a run without a training DBFILE: the elements the
 *  ATOM key of the training configuration declares, which no data gives
 *  (a validation file is optional too). HPOEngine::run takes the element
 *  set from it, and validate_hpo_setup() applies it, so that `tadah hpo
 *  --check` and a program that validates a setup refuse what a run
 *  refuses.
 *  @throws std::runtime_error without ATOM: "No training DBFILE: the ATOM
 *  key is required in the training configuration to declare the element
 *  set (e.g. 'ATOM Ta'). ..." */
std::set<core::Element> declared_element_set(const core::Context &train_ctx);

/** Fail-fast validation of the HPO setup, run once after HPOTargets
 *  parsing and before HPO_Host construction. Throws std::runtime_error
 *  with an actionable message on the first violated rule:
 *
 *   - the settings of the fit every evaluation would refuse
 *     (mlip::require_fit_settings, with "tadah hpo"): a retired key of the
 *     training configuration (with its conversion), an error scale that is
 *     not a number > 0 (a relative part >= 0), one other than its default
 *     for a quantity the fit does not have, ENERGY, FORCE and STRESS all
 *     false;
 *   - OPTIM of a retired key (with its conversion) and of ENORM or ENERGY
 *     (HPOTargets::optim_retired_key_error, optim_switch_key_error, also at
 *     their line);
 *   - OPTIM WEIGHTS is rejected (each evaluation fits the coefficients);
 *   - OPTIM EMBF0 is rejected (a discrete property of the potential), and
 *     a stored EMBF0 of a DM_REAM configuration must be ZERO or FREE;
 *   - OPTIM on structure data is rejected (RHOFORM, TYPE2B, TYPEMB,
 *     RCTYPE2B, RCTYPEMB, KDEG2B, KDEGMB, KDEGEMB);
 *   - a DM_REAM configuration must declare RHOFORM, and a bare DM_REAM
 *     density must be valid for its form (models::DensityFormError): the
 *     grids, the MONOTONE cutoff whitelist and, under NONNEG / MONOTONE,
 *     WATOM > 0. No AMPGRIDMB amplitude below 0 is evaluated, so none is
 *     warned about: a NONNEG / MONOTONE box does not reach below 0, finite
 *     differences keep to the box (fd_stencil.h), and HPO_Host scores a
 *     point an optimiser that ignores the bounds proposes outside it, by
 *     its distance to the box, without evaluating it (box_penalty.h); such
 *     an optimiser (OptimizerCaps::ignores_bounds: CERES:STEEPEST_DESCENT)
 *     has the startup WARNING of validate_optimizer_scaling, once per run
 *     and backend;
 *   - every width of a radial bump channel (SGRID2B of D2_Blip/D2_Blip5,
 *     SGRIDMB of a Blip/Blip5 density) is > 0, in every form
 *     (models::optim_low_bound_rule(key, ctx));
 *   - the model of the -c configuration builds (the Trainer, as every
 *     evaluation builds it): a construction error would fail every
 *     evaluation;
 *   - RHOGATE / RHOBARGATE (rho_gates.h): a bare DM_REAM many-body term
 *     (DM_mJoin refused), RHOGATE under RHOFORM SIGNED only, R_LO and the
 *     first RHOBARGATE shell inside the smallest reachable RCUTMB; every
 *     CONFIG/WARM seed of the INIT and RESTART chains evaluated as a
 *     candidate (SEED ERROR throws, SEED WARN warns);
 *   - OPTIM FIXWEIGHT requires FIXINDEX in the training configuration;
 *   - NLOPT, DLIB and CERES need at least one OPTIM line;
 *   - the STRATEGY TYPE must suit the optimiser (validate_run_policy,
 *     strategy_rules.h): MULTISTART rejects a global optimiser that
 *     takes one starting point, for any N_STARTS; PRESTAGE needs
 *     an EXPLORE block and, with a configured TOPK > 1, a refinement
 *     optimiser MULTISTART accepts;
 *   - each phase's backend accepts its settings (validate_phase_backends,
 *     backend_rules.h): the backend's own attribute and PARAM checks on
 *     the spec the phase runs with, and MAXEVAL for a backend that has
 *     no stopping rule of its own (TADAH:RANDOM, DLIB:GFS);
 *   - finite differences (fd_stencil.h): each phase that takes
 *     finite-difference gradients with its own settings
 *     (takes_fd_gradients: GRAD_STRATEGY NUMERICAL or HYBRID, not an INNER
 *     block, which takes the gradient of the phase it runs in) has a valid
 *     PARAM fd_step, fd_kernel and diff_method (diff_method 2, RIDDERS,
 *     refused), and every OPTIM box gives each of its points a stencil
 *     inside it (fd_box_error: the message names the step's source --
 *     PARAM or the NOISE block's AUTO -- the largest fd_step that fits,
 *     and OPTIM_SCALE AUTO or a LOG box as alternatives); a phase that
 *     takes none refuses those PARAMs, which never take effect (an INNER
 *     block, GRAD_STRATEGY ANALYTICAL, a derivative-free optimiser), and
 *     the STRATEGY-level phase the NOISE block's AUTO likewise ("NOISE
 *     AUTO has no effect: NLOPT:LN_SBPLX measures no slopes"); ASA's
 *     reannealing and curvature probes fit its boxes
 *     (asa_probe_box_error);
 *   - FIXINDEX/FIXWEIGHT pairing: lone key, size mismatch, out-of-bounds
 *     index (via mlip::classify_weight_source);
 *   - an M_KRR implementation other than 1 and 2
 *     (mlip::require_krr_implementation), with the fit's message, whether
 *     or not the evaluations fit: it is part of the potential's MODEL line;
 *   - when an evaluation fits the model (training data, and not
 *     fixindex_pins_every_weight): a LAMBDA the fit refuses
 *     (mlip::require_fit_lambda: a negative value other than -1, a second
 *     value, -1 under an OALGO other than 3 or under M_KRR implementation
 *     2), OALGO 3 with a LAMBDA other than 0 and an intercept column the
 *     fit fits (mlip::require_oalgo3_intercepts; OPTIM LAMBDA too, every
 *     candidate but 0 of which the fit would refuse), and under M_KRR
 *     implementation 2 OPTIM of ESIGMA, FSIGMA, SSIGMA or WDBFILE
 *     (mlip::krr_implementation2_ignores), each with the fit's message;
 *     OPTIM of the error scale of a quantity the fit does not have (ENERGY,
 *     FORCE or STRESS false), and of the error scale of the only quantity
 *     of the fit (its absolute part, with no relative part searched or
 *     set) with LAMBDA 0 or -1: every candidate would give the same fit; a
 *     WARNING when every error scale of the fit is searched with LAMBDA 0
 *     or -1, or with OPTIM LAMBDA (one direction of the search changes
 *     nothing);
 *   - when the evaluations fit no coefficient (no training data, or
 *     FIXINDEX pins every weight): OPTIM of ESIGMA, FSIGMA, SSIGMA or
 *     WDBFILE;
 *   - OPTIM of a key applied once when the run starts
 *     (HPOTargets::optim_startup_only_key_error, also at its line);
 *   - a training DBFILE lists each dataset once
 *     (mlip::require_distinct_training_datasets);
 *   - no training DBFILE: FIXINDEX must cover every model weight;
 *   - at least one loss source must be declared (validation metrics,
 *     PC_OBSERVABLE, or PC_LAMMPS);
 *   - validation metrics (PC_ERMSE, ...) require a --validation file;
 *   - NORM true is incompatible with FIXINDEX/FIXWEIGHT (fixed values
 *     would be interpreted in the normalized feature space);
 *   - GRAD_STRATEGY ANALYTICAL (written out, or resolved for
 *     PC_OBSERVABLE) in a block that runs an optimiser and obtains
 *     gradients (the STRATEGY-level block, EXPLORE; an INNER search uses
 *     its enclosing block's gradients) needs a setup the analytical
 *     gradient covers and loss terms with an analytical derivative
 *     (analytical_gradient_blocker, analytical_loss_term_blocker,
 *     analytical_gate.h), and no OPTIM of an error scale, which it does
 *     not differentiate (each search parameter's derivative is checked
 *     once the data are read: require_analytical_parameters());
 *   - FAILSCORE_GRADING ON needs a finite FAILSCORE that leaves room for
 *     the four grading bands.
 */
void validate_hpo_setup(core::Context &train_ctx, const HPOTargets &trg,
                        bool has_train_db, bool has_validation);

/** GRAD_STRATEGY ANALYTICAL (written out, or resolved for PC_OBSERVABLE) in
 *  a block that obtains gradients, as validate_hpo_setup() takes it, needs
 *  an analytical derivative with respect to every search parameter: one
 *  the descriptors of the training configuration @p train_ctx do not
 *  differentiate in the rows of the training structures @p train and the
 *  validation structures @p val (parameters_without_analytical_derivative,
 *  analytical_gate.h) is refused at the line that closes the block, naming
 *  each such OPTIM line, why, and HYBRID: "OPTIMIZER block: GRAD_STRATEGY
 *  ANALYTICAL needs an analytical derivative with respect to every search
 *  parameter, and the descriptors give none for OPTIM LAMBDA (1) on line 4
 *  of the target file (no descriptor differentiates it). GRAD_STRATEGY
 *  HYBRID takes finite differences for it."; "... (DM_Blip differentiates
 *  it in the energy rows only, and FORCE true gives the fit force rows)";
 *  "... and PC_FRMSE measures the forces of the validation structures)";
 *  "... (DM_REAM depends on it and does not differentiate it)" (OPTIM
 *  WATOM beside a two-body descriptor that differentiates it).
 *  A parameter the descriptors differentiate has rows of 0 in structures
 *  that hold no pair of atoms that depends on it, which the analytical
 *  gradient forms. The data decide which rows the descriptors claim, so
 *  hpo_run() calls it once HPO_Host has read them (a check too), before
 *  any evaluation; validate_hpo_setup() has accepted the setup. */
void require_analytical_parameters(const core::Context &train_ctx,
                                   const HPOTargets &trg,
                                   const mlip::StructureDB &train,
                                   const mlip::StructureDB &val);

/** The --validation file of the run (VALIDATION): "" when there is none,
 *  else its one value. tadah hpo reads one validation configuration file,
 *  whose DBFILE lines list the validation datasets; a VALIDATION of two or
 *  more values (`--validation a.val b.val`) throws std::runtime_error
 *  naming them (hpo_run() read the first and dropped the others). */
std::string validation_config_file(const core::Context &ctx);

/** Refuses a validation configuration (@p val_ctx, parsed from the
 *  --validation file @p val_cfg) whose DBFILE lists one dataset file
 *  twice (mlip::repeated_dataset, the rule of the training DBFILE): every
 *  validation metric averages over the validation datasets, so the file
 *  would count twice. Throws std::runtime_error naming the file and both
 *  entries; a no-op without DBFILE. The HPO_Host constructor calls it
 *  before any validation dataset is read. */
void require_distinct_validation_datasets(const core::Context &val_ctx,
                                          const std::string &val_cfg);

} // namespace hpo
} // namespace tadah

#endif // TADAH_HPO_STARTUP_VALIDATION_H

#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/analytical_gate.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/strict_parse.h>

#include <tadah/core/intercept_layout.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/mlip/data_weights.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/training_datasets.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/ream_density.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah {
namespace hpo {

namespace {

/* The DM_REAM density of a bare DM_REAM block of the -c configuration,
 * validated for its form from the context (ReamDensity::validate_context),
 * as the DM_REAM constructor does, so an invalid density is a startup
 * error with its own message. Inside a DM_mJoin each DM_REAM child
 * validates itself when probe_model() builds the model. Throws
 * models::DensityFormError. */
void validate_ream_density(const core::Context &train_ctx) {
  core::Context c = train_ctx;   // validate_context may materialise defaults
  const auto label = c.stored_value("TYPEMB", 0);
  if (label && *label == "DM_REAM") models::ReamDensity::validate_context(c);
}

/* `reason` as the first sentence of a longer message: trailing white
 * space removed and one full stop at its end ('.' added unless it ends in
 * '.', '!' or '?'), so the next sentence follows after one space. */
std::string as_sentence(std::string reason) {
  while (!reason.empty() &&
         std::isspace(static_cast<unsigned char>(reason.back())))
    reason.pop_back();
  if (reason.empty() || std::string(".!?").find(reason.back()) ==
                            std::string::npos)
    reason += '.';
  return reason;
}

/* What probe_model() learns from building the model of the -c
 * configuration. */
struct ModelProbe {
  std::size_t weights = 0;                  // derive_weight_count
  std::string function;                     // the basis function or kernel
  std::vector<models::MParamSlot> mparams;  // the MPARAMS it reads
  /* For RCUT2B and RCUTMB, per value of the key: what fixes it
   * (D_Base::cutoff_fixed_by of the two-body or many-body descriptor), ""
   * for a free value. */
  std::map<std::string, std::vector<std::string>> cutoff_fixed;
};

/* The model of the -c configuration builds, as every evaluation builds
 * it (the Trainer: the descriptors, the basis function or kernel and the
 * model): a construction error here -- a DIMER potential with element-
 * named TYPE2B pairs, a descriptor that refuses its grids -- would fail
 * every evaluation (the run then ends "no evaluation produced a usable
 * loss"), and so would a missing or too small SBASIS of a fit that draws
 * basis vectors (M_KRR Kern_Linear 2 too; model_weight_count checks it).
 * Returns the model's weight count (derive_weight_count; SBASIS for
 * a non-linear M_KRR kernel, which has no basis before training), its
 * function, the MPARAMS the function reads (Function_Base::mparams()) and
 * what fixes each RCUT2B/RCUTMB value (D_Base::cutoff_fixed_by).
 * models::DensityFormError keeps its type and message. */
ModelProbe probe_model(const core::Context &train_ctx) {
  try {
    core::Context c = train_ctx;
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer probe(c, ws);
    ModelProbe out;
    out.weights = tadah::mlip::model_weight_count(*probe.fb, probe.context);
    out.function = probe.fb->get_label();
    out.mparams = probe.fb->mparams();
    const std::pair<const char *, const models::D_Base *> cutoffs[] = {
        {"RCUT2B", probe.DCS.d2b}, {"RCUTMB", probe.DCS.dmb}};
    for (const auto &[key, d] : cutoffs) {
      if (!d || !train_ctx.contains(key)) continue;
      auto &why = out.cutoff_fixed[key];
      for (std::size_t i = 0; i < train_ctx.size(key); ++i)
        why.push_back(d->cutoff_fixed_by(i));
    }
    return out;
  } catch (const models::DensityFormError &) {
    throw;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "the -c configuration does not build a model: " +
        as_sentence(e.what()) + " Every evaluation would fail; fix the -c "
        "potential/configuration.");
  }
}

/* Why `s`, which asks for no gradient (obtains_gradients), takes none, as
 * a clause that follows "NLOPT:GD_MLSL_LDS takes no gradient": "" for a
 * backend without a local search of its own; for NLopt's MLSL and AUGLAG,
 * whose local search alone would take one, which search that is:
 * ": NLopt's MLSL and AUGLAG take a gradient only for their local search,
 * and the INNER block's NLOPT:LN_BOBYQA takes none". */
std::string no_gradient_reason(const OptimizerSpec &s) {
  if (!optimizer_caps(s).accepts_inner) return "";
  /* validate_run_policy (require_inner_backend) has refused an MLSL or
   * AUGLAG that runs no local search. */
  const std::optional<OptimizerSpec> local = inner_run_spec(s);
  if (!local) return "";
  return std::string(": NLopt's MLSL and AUGLAG take a gradient only for "
                     "their local search, and ") +
         (s.inner ? "the INNER block's " + local->lib + ":" + local->algo
                  : "NLopt's default local optimiser, " + local->lib + ":" +
                        local->algo + ",") +
         " takes none";
}

/* Why the finite-difference PARAMs of `phase`, which takes no
 * finite-difference gradients with its own settings (takes_fd_gradients),
 * never take effect. */
std::string fd_params_unused_why(const PhaseSpec &phase) {
  const OptimizerSpec &s = phase.spec;
  if (phase.inner)
    return "an INNER search takes the gradient of the block it runs in, "
           "with that block's finite-difference settings";
  if (s.grad_strategy.value_or(GradStrategy::Numerical) ==
      GradStrategy::Analytical)
    return std::string("GRAD_STRATEGY ANALYTICAL takes no finite "
                       "differences") +
           (s.grad_strategy_resolved ? " (the resolution of an unset "
                                       "GRAD_STRATEGY with PC_OBSERVABLE)"
                                     : "");
  return s.lib + ":" + s.algo + " takes no gradient" + no_gradient_reason(s);
}

/* Why the NOISE block's AUTO has no effect on `phase`, the STRATEGY-level
 * phase, which takes no finite-difference gradients (takes_fd_gradients):
 * it measures no slopes, or not with fd_step. */
std::string noise_auto_unused_why(const PhaseSpec &phase) {
  const OptimizerSpec &s = phase.spec;
  if (obtains_gradients(s)) return fd_params_unused_why(phase);
  if (s.lib == "ASA")
    return s.lib + ":" + s.algo + " measures its slopes with probes of its "
           "own (PARAM Delta_X), not with fd_step";
  return s.lib + ":" + s.algo + " measures no slopes" + no_gradient_reason(s);
}

}  // namespace

std::size_t derive_weight_count(const core::Context &train_ctx) {
  core::Context ctx_copy = train_ctx;
  tadah::mlip::memory::MLIPWorkspaceManager ws;
  tadah::mlip::Trainer probe(ctx_copy, ws);
  return tadah::mlip::model_weight_count(*probe.fb, probe.context);
}

RawEmbeddingF0 derive_raw_embedding_f0(const core::Context &train_ctx) {
  RawEmbeddingF0 out;
  if (!models::context_has_ream(train_ctx) && train_ctx.blocks("DESCRIPTOR").empty()) {
    out.reason = "no DM_REAM many-body block";
    return out;
  }
  if (!models::model_is_linear(train_ctx)) {
    out.reason = "the model is not linear in the descriptor";
    return out;
  }
  try {
    core::Context ctx_copy = train_ctx;
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer probe(ctx_copy, ws);
    const core::Context &pc = probe.context;
    const std::size_t K = probe.fb->get_phi_cols(pc);
    core::t_type w;
    if (pc.has_stored("FIXINDEX") && pc.has_stored("FIXWEIGHT") &&
        core::parse_fixindex(pc, K).size() == K) {
      probe.provision_fixed_weights();
      w = probe.model->get_weights();
      out.source = "FIXWEIGHT";
    }
    /* The -c potential's WEIGHTS: train_ctx's (the Trainer, which builds
     * a model to be fitted, reads none). */
    if (w.size() == 0 && train_ctx.has_stored("WEIGHTS") &&
        train_ctx.size("WEIGHTS") == K) {
      w = core::t_type(K);
      for (std::size_t i = 0; i < K; ++i)
        w(i) = train_ctx.get<double>("WEIGHTS", i);
      out.source = "WEIGHTS";
    }
    if (w.size() == 0) {
      out.kind = RawEmbeddingF0::Kind::PerEvaluation;
      return out;
    }
    std::optional<double> v;
    if (w.size() == probe.DCS.dsize) v = probe.DCS.dmb->raw_embedding_f0(w);
    if (!v) {
      out.source.clear();
      out.reason = "the weight vector does not match the descriptor";
      return out;
    }
    out.kind = RawEmbeddingF0::Kind::Value;
    out.value = *v;
  } catch (const models::DensityFormError &) {
    throw;   // an invalid density form is a configuration error
  } catch (const std::exception &e) {
    out = RawEmbeddingF0{};
    out.reason = std::string("not derivable: ") + e.what();
  }
  return out;
}

bool fixindex_pins_every_weight(const core::Context &ctx) {
  if (!ctx.contains("FIXINDEX")) return false;
  if (ctx.contains("NORM") && ctx.get<bool>("NORM")) return false;
  const std::size_t K = derive_weight_count(ctx);
  return K > 0 &&
         tadah::mlip::classify_weight_source(ctx, K) ==
             tadah::mlip::WeightSource::AllFixedData &&
         !tadah::mlip::basis_drawn_from_training_data(ctx);
}

namespace {

/* Whether an OPTIM line of @p trg searches index @p index (1-based; 0:
 * any) of @p key. */
bool optim_searches(const HPOTargets &trg, const std::string &key,
                    std::size_t index = 0) {
  const auto it = trg.optimap.find(key);
  if (it == trg.optimap.end()) return false;
  const auto &opt = it->second.opt;
  if (index == 0) return std::any_of(opt.begin(), opt.end(),
                                     [](bool b) { return b; });
  return index <= opt.size() && opt[index - 1];
}

/* The OPTIM dimensions of the error scales of @p trg ("ESIGMA", "FSIGMA
 * (1)", ...), in the order ESIGMA, FSIGMA, SSIGMA. */
std::vector<std::string> searched_error_scales(const HPOTargets &trg) {
  std::vector<std::string> out;
  if (optim_searches(trg, "ESIGMA")) out.push_back("ESIGMA");
  for (const char *key : {"FSIGMA", "SSIGMA"})
    for (std::size_t i : {1u, 2u})
      if (optim_searches(trg, key, i))
        out.push_back(std::string(key) + " (" + std::to_string(i) + ")");
  return out;
}

/* Why GRAD_STRATEGY ANALYTICAL cannot form the gradient of a search of an
 * error scale or of a dataset priority (the analytical gradient
 * differentiates the fit with respect to the parameters of the descriptors
 * and of FIXWEIGHT, not its row factors), "" without one. */
std::string error_scale_search_blocker(const HPOTargets &trg) {
  for (const char *key : {"ESIGMA", "FSIGMA", "SSIGMA"})
    if (optim_searches(trg, key))
      return std::string("OPTIM ") + key + " searches an error scale of "
             "the fit, which the analytical gradient does not "
             "differentiate";
  if (optim_searches(trg, "WDBFILE"))
    return "OPTIM WDBFILE searches a priority of the fit, which the "
           "analytical gradient does not differentiate";
  return "";
}

/* The OPTIM dimensions of WDBFILE of @p trg ("WDBFILE (1)", ...), and
 * whether they are the value of every one of the @p n_datasets training
 * datasets. */
std::vector<std::string> searched_wdbfile(const HPOTargets &trg,
                                          std::size_t n_datasets,
                                          bool &every) {
  std::vector<std::string> out;
  every = n_datasets > 0;
  for (std::size_t i = 1; i <= n_datasets; ++i) {
    if (optim_searches(trg, "WDBFILE", i))
      out.push_back("WDBFILE (" + std::to_string(i) + ")");
    else
      every = false;
  }
  return out;
}

/* An OPTIM line of an error scale or of WDBFILE that cannot change the
 * fit is refused, and every error scale of the fit searched with LAMBDA 0
 * or -1, or with OPTIM LAMBDA, gets a WARNING, as does the WDBFILE of
 * every training dataset (row_weights.h: the fit minimises sum p (r /
 * sigma)^2 + LAMBDA |w|^2, so every error scale times c fits as LAMBDA
 * times c^2, and every priority times c as LAMBDA divided by c):
 *  - the error scale of a quantity the fit does not have (ENERGY, FORCE or
 *    STRESS false): it would change nothing;
 *  - the error scale of the only quantity of the fit (its absolute part
 *    alone with no relative part, FSIGMA b 0) with LAMBDA 0 or -1: it
 *    scales every row alike, which changes no coefficient (the evidence
 *    approximation absorbs it into its beta);
 *  - WDBFILE of a run with one training dataset with LAMBDA 0 or -1: its
 *    value multiplies the priority of every row alike, likewise.
 * For a run whose evaluations fit the model. */
void require_error_scale_search_changes_the_fit(core::Context &ctx,
                                                const HPOTargets &trg) {
  struct Quantity {
    const char *key, *sw, *what, *others;
  };
  const Quantity quantities[3] = {
      {"ESIGMA", "ENERGY", "energies", "forces or stresses"},
      {"FSIGMA", "FORCE", "forces", "energies or stresses"},
      {"SSIGMA", "STRESS", "stresses", "energies or forces"}};
  std::vector<const Quantity *> on;
  for (const Quantity &q : quantities) {
    const bool has = ctx.get<bool>(q.sw);
    if (has) on.push_back(&q);
    if (!has && optim_searches(trg, q.key))
      throw std::runtime_error(
          std::string("OPTIM ") + q.key + ": " + q.sw + " is false: the fit "
          "has no " + q.what + ", so " + q.key + " would change nothing and "
          "every candidate would give the same fit. " + q.sw + " true fits "
          "the " + q.what + ".");
  }
  /* Whether the search can multiply the error scale of @p q by any factor:
   * ESIGMA, or the absolute part a of FSIGMA or SSIGMA with its relative
   * part searched too or 0. */
  auto scales = [&](const Quantity &q) {
    const std::string key = q.key;
    if (key == "ESIGMA") return optim_searches(trg, key);
    if (!optim_searches(trg, key, 1)) return false;
    return optim_searches(trg, key, 2) ||
           (ctx.size(key) < 2 || ctx.get<double>(key, 1) == 0.0);
  };
  const bool optim_lambda = optim_searches(trg, "LAMBDA");
  const double lambda = ctx.get<double>("LAMBDA", 0);
  const bool lambda_scale_free = !optim_lambda && (lambda == 0.0 || lambda == -1.0);
  const std::string lambda_text = core::format_roundtrip(lambda);
  if (on.size() == 1 && lambda_scale_free && scales(*on.front()) &&
      !(std::string(on.front()->key) != "ESIGMA" &&
        optim_searches(trg, on.front()->key, 2))) {
    const Quantity &q = *on.front();
    throw std::runtime_error(
        std::string("OPTIM ") + q.key + ": the " + q.what + " are the only "
        "quantity of the fit and LAMBDA is " + lambda_text + ": the error "
        "scale of the only quantity scales every row of the fit alike, "
        "which changes no coefficient, so every candidate would give the "
        "same fit. With a LAMBDA > 0 the error scale weighs the data "
        "against the penalty, and with " + q.others + " in the fit, the " +
        q.what + " against them.");
  }
  const std::size_t n_datasets =
      ctx.has_stored("DBFILE") ? ctx.size("DBFILE") : 0;
  if (n_datasets == 1 && lambda_scale_free && optim_searches(trg, "WDBFILE"))
    throw std::runtime_error(
        "OPTIM WDBFILE: the run has one training dataset and LAMBDA is " +
        lambda_text + ": its WDBFILE value multiplies the priority of every "
        "row of the fit alike, which changes no coefficient, so every "
        "candidate would give the same fit. With a LAMBDA > 0 the priority "
        "weighs the data against the penalty, and with more training "
        "datasets, the dataset against the others.");
  if (!(lambda_scale_free || optim_lambda)) return;
  auto warn = [&](const std::string &w) {
    std::cerr << "[HPO] WARNING: " << w << "\n";
    HPOLog::instance().warning(w);
  };
  const std::string tail =
      ", so one direction of the search changes nothing; with one of them "
      "fixed the search covers the same fits.";
  bool every_wdbfile = false;
  const std::vector<std::string> wdbfile =
      searched_wdbfile(trg, n_datasets, every_wdbfile);
  if (every_wdbfile) {
    std::string dims;
    for (const std::string &d : wdbfile) dims += (dims.empty() ? "" : ", ") + d;
    warn("OPTIM: the WDBFILE value of every training dataset is searched (" +
         dims + ") " +
         (optim_lambda
              ? std::string("and so is LAMBDA (OPTIM LAMBDA): the WDBFILE "
                            "values times c with LAMBDA times c give the "
                            "same fit")
              : "and LAMBDA is " + lambda_text + ": the same factor on all "
                "of them gives the same fit") +
         tail);
  }
  if (on.empty()) return;
  for (const Quantity *q : on)
    if (!scales(*q)) return;
  std::string dims;
  for (const std::string &d : searched_error_scales(trg))
    dims += (dims.empty() ? "" : ", ") + d;
  warn("OPTIM: every error scale of the fit is searched (" + dims + ") " +
       (optim_lambda
            ? std::string("and so is LAMBDA (OPTIM LAMBDA): the error scales "
                          "times c with LAMBDA divided by c^2 give the same "
                          "fit")
            : "and LAMBDA is " + lambda_text + ": the same factor on all of "
              "them gives the same fit") +
       tail);
}

}  // namespace

std::set<core::Element> declared_element_set(const core::Context &train_ctx) {
  if (!train_ctx.has_stored("ATOM"))
    throw std::runtime_error(
        "No training DBFILE: the ATOM key is required in the training "
        "configuration to declare the element set (e.g. 'ATOM Ta'). " +
        tadah::mlip::potential_has_no_datasets_advice());
  std::set<core::Element> elements;
  for (const auto &sym : train_ctx("ATOM"))
    elements.insert(core::PeriodicTable::find_by_symbol(sym));
  return elements;
}

void validate_hpo_setup(core::Context &train_ctx, const HPOTargets &trg,
                        bool has_train_db, bool has_validation) {
  /* --- the element set of a run without training data ------------------
   * declared by ATOM, as HPOEngine::run reads it. */
  if (!has_train_db) (void)declared_element_set(train_ctx);

  /* --- the settings of the fit, as every fit refuses them --------------
   * A retired key (with its conversion), an error scale that is not a
   * number > 0 (a relative part >= 0), one other than its default for a
   * quantity the fit does not have, and ENERGY, FORCE and STRESS all
   * false (mlip::require_fit_settings): every evaluation would refuse
   * them. */
  tadah::mlip::require_fit_settings(train_ctx, "tadah hpo");

  /* --- the training datasets, as tadah train reads them ----------------
   * A DBFILE listing one file twice would have every evaluation fit its
   * structures twice. */
  if (has_train_db) tadah::mlip::require_distinct_training_datasets(train_ctx);

  /* --- NORM/BIAS rules, as tadah train applies them --------------------
   * NORM true with a linear model needs BIAS true (the NORM weight
   * conversion moves the mean correction into the intercept columns);
   * BIAS true needs ATOM; FIXINDEX/FIXWEIGHT is refused with NORM true
   * (the conversion moves the pinned weights). Refused here, not by every
   * evaluation. */
  tadah::mlip::check_bias_constraints(train_ctx);

  /* --- OPTIM key sanity (parse-time rules, re-checked cheaply) -------- */
  if (trg.optimap.count("WEIGHTS")) {
    throw std::runtime_error(
        "OPTIM WEIGHTS is not a valid search target: the linear WEIGHTS are "
        "re-fitted by least squares on every evaluation, so optimising them "
        "is a silent no-op. To search linear coefficients, pin them with "
        "FIXINDEX + FIXWEIGHT in the training configuration and use "
        "`OPTIM FIXWEIGHT (<weight-index>) <low> <high>`.");
  }
  if (trg.optimap.count("EMBF0")) {
    throw std::runtime_error(
        "OPTIM EMBF0 is not a valid search target: EMBF0 (ZERO|FREE) "
        "selects the DM_REAM embedding reference, a discrete property of "
        "the potential; set it in the -c potential/configuration.");
  }
  for (const auto &kv : trg.optimap) {
    std::string e = HPOTargets::optim_structure_key_error(kv.first);
    if (e.empty()) e = HPOTargets::optim_retired_key_error(kv.first);
    if (e.empty()) e = HPOTargets::optim_switch_key_error(kv.first);
    if (e.empty()) e = HPOTargets::optim_startup_only_key_error(kv.first);
    if (!e.empty()) throw std::runtime_error(e);
  }
  /* An invalid stored EMBF0 value fails here, not at the first evaluation.
   * stored_embf0 never inserts the default. */
  if (models::context_has_ream(train_ctx))
    (void)models::stored_embf0(train_ctx);

  /* --- DM_REAM density form (RHOFORM) ---------------------------------- */
  /* Mandatory (HPOTargets already read it). A bare DM_REAM density is
   * validated for its form here, with its own message; inside a DM_mJoin
   * each DM_REAM child validates itself when probe_model() builds the
   * model below (models::DensityFormError keeps its type). */
  if (models::context_has_ream(train_ctx)) {
    (void)models::require_rhoform(train_ctx, "tadah hpo");
    validate_ream_density(train_ctx);
  }

  /* --- radial bump widths of the seed ----------------------------------
   * Every width of a radial bump channel -- SGRID2B of a D2_Blip/D2_Blip5
   * pair term, SGRIDMB of a DM_REAM Blip/Blip5 density -- is > 0, in every
   * form: the rule of the OPTIM boxes (models::optim_low_bound_rule(key,
   * ctx)) holds for the seed too. A bump spans [c - w/eta, c + w/eta]: a
   * width <= 0 has no support edge (the bump is even in eta and evaluates
   * as its mirror), and the bump-support gate below cannot keep it inside
   * the cutoff. */
  {
    std::vector<std::string> bad;
    for (const char *key : {"SGRID2B", "SGRIDMB"}) {
      const auto rule = models::optim_low_bound_rule(key, train_ctx);
      if (!rule) continue;   // not the width key of a radial bump channel
      core::v_type eta;
      try {
        models::D_Base::get_grid(train_ctx, key, eta);
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string(key) + " of the -c potential "
                                 "cannot be read (" + e.what() + "): " +
                                 rule->by + " needs its widths");
      }
      for (std::size_t i = 0; i < eta.size(); ++i)
        if (!rule->admits(eta[i]))
          bad.push_back(std::string(key) + " (" + core::to_string(i + 1) +
                        ") = " + core::to_string(eta[i]) + ": " + rule->by +
                        " requires " + rule->requirement);
    }
    if (!bad.empty()) {
      std::string msg = "the -c potential has a radial bump width <= 0:";
      for (const auto &b : bad) msg += "\n  - " + b;
      msg += "\n  Set every width > 0 in the -c potential/configuration "
             "(tadah refit --edit rewrites a potential).";
      throw std::runtime_error(msg);
    }
  }

  if (trg.optimap.count("FIXWEIGHT") && !train_ctx.contains("FIXINDEX")) {
    throw std::runtime_error(
        "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration "
        "(FIXINDEX names which model weights the FIXWEIGHT values pin).");
  }

  /* --- STRATEGY TYPE vs optimiser --------------------------------------
   * MULTISTART needs an optimiser that can run from each INIT point, and
   * so does the PRESTAGE refinement with a configured TOPK > 1; PRESTAGE
   * needs an EXPLORE block. A property of the configuration alone, so it
   * is reported here, before any data is loaded -- not once the policy
   * runs (under PRESTAGE, after the whole exploration). The policies apply
   * the same rules again when they run (strategy_rules.h). */
  validate_run_policy(trg.spec);

  /* --- each phase's backend settings -----------------------------------
   * The search space NLOPT, DLIB and CERES need (one OPTIM line or more),
   * the attribute and PARAM checks of each backend, and its need for a
   * MAXEVAL, on the spec each phase runs with (phase_specs.h: the
   * STRATEGY-level optimiser, each MULTISTART start with its budget, the
   * PRESTAGE exploration and refinement, every INNER block). They used to
   * run only when that optimiser started: after the data were loaded, and
   * for a PRESTAGE refinement after the whole exploration. Each driver
   * applies its own checks again when it starts (backend_rules.h). */
  validate_phase_backends(trg.spec);

  /* --- finite differences and the OPTIM box ---------------------------
   * Each phase that takes finite-difference gradients with its own
   * settings (takes_fd_gradients: GRAD_STRATEGY NUMERICAL or HYBRID on an
   * optimiser that asks for gradients, an MLSL/AUGLAG whose local search
   * does included) has a valid PARAM fd_step, fd_kernel and diff_method
   * (RIDDERS refused), and every OPTIM box gives each of its points a
   * stencil inside it (fd_box_error, fd_stencil.h): no finite-difference
   * point is ever evaluated outside the box, and the step is not shrunk.
   * A phase that takes none refuses those PARAMs, which would never take
   * effect: an INNER block (it takes the gradient of the phase it runs
   * in), GRAD_STRATEGY ANALYTICAL, a derivative-free optimiser -- which
   * refuses a GRAD_STRATEGY too; the
   * STRATEGY-level phase refuses the NOISE block's AUTO likewise, which
   * derives fd_step for that phase's finite differences (HPOTargets) and
   * has no effect without them. ASA's
   * finite-difference probes must fit its boxes too (asa_probe_box_error).
   * Refused here, before any data is loaded, rather than by the first
   * gradient. */
  /* Each refusal names the line the phase's block closes at, as the
   * parser's refusal of a backend does (with_block_line). */
  const auto check_phase = [&](const PhaseSpec &phase) {
    const OptimizerSpec &s = phase.spec;
    if (s.lib == "NOT SET") return;
    const std::string who = phase.name + " " + s.lib + ":" + s.algo;
    if (takes_fd_gradients(phase)) {
      std::string e;
      try {
        const FdSettings fs = fd_settings_of(s);
        /* The NOISE block's AUTO beside a PARAM fd_step written in the
         * block: the written step is the one taken. */
        if (!phase.inner && s.noise.autodetect &&
            fs.source == FdStepSource::Param) {
          std::ostringstream os;
          os << std::setprecision(10) << fs.fd_step;
          e = "NOISE AUTO has no effect: PARAM fd_step " + os.str() +
              " is set; remove one of them.";
        } else {
          e = fd_box_error(s, fs);
        }
      } catch (const std::exception &ex) {
        e = ex.what();
      }
      if (!e.empty()) throw std::runtime_error(who + ": " + e);
      return;
    }
    /* GRAD_STRATEGY on a phase whose optimiser never asks for a gradient
     * (obtains_gradients: for MLSL and AUGLAG, their local search) never
     * takes effect either. The value is the user's: PC_OBSERVABLE resolves
     * an unset one only where gradients are obtained, and an INNER block
     * has none (the parser refuses it). */
    if (s.grad_strategy && !phase.inner && !obtains_gradients(s))
      throw std::runtime_error(who + ": GRAD_STRATEGY never takes effect: " +
                               s.lib + ":" + s.algo + " takes no gradient" +
                               no_gradient_reason(s) + "; remove it.");
    /* Only the STRATEGY-level phase carries the NOISE block (spec.noise:
     * EXPLORE and INNER blocks have none). */
    if (!phase.inner && s.noise.autodetect)
      throw std::runtime_error(who + ": NOISE AUTO has no effect: " +
                               noise_auto_unused_why(phase) +
                               "; remove AUTO from the NOISE block.");
    const std::string written = fd_params_written(s);
    if (!written.empty()) {
      const bool several = written.find(" and ") != std::string::npos;
      std::string remove;
      if (phase.inner)
        remove = std::string("set ") + (several ? "them" : "it") +
                 " in the enclosing block";
      else
        remove = std::string("remove ") + (several ? "them" : "it");
      throw std::runtime_error(
          who + ": " + written + (several ? " never take" : " never takes") +
          " effect: " + fd_params_unused_why(phase) + "; " + remove + ".");
    }
    if (s.lib == "ASA") {
      const std::string e = asa_probe_box_error(s);
      if (!e.empty()) throw std::runtime_error(who + ": " + e);
    }
  };
  for (const PhaseSpec &phase : strategy_phase_specs(trg.spec)) {
    try {
      check_phase(phase);
    } catch (const std::exception &e) {
      throw std::runtime_error(with_block_line(phase.spec, e.what()));
    }
  }

  /* --- a MAXEVAL that cannot pay for one step of a gradient search -----
   * A WARNING, not a refusal (a probe of the start can be deliberate),
   * for each phase with a budget of its own (gradient_budget_warnings,
   * backend_rules.h); after the phase checks, which have accepted the
   * finite-difference settings it reads. With the run's permission to
   * answer a point the run has evaluated from the run's record: the host's
   * (HPO_Host::repeat_reuse_blocker() is this repeat_reuse_blocker()), so
   * the warning counts the evaluations the run spends. */
  for (const std::string &w : gradient_budget_warnings(
           trg.spec, trg.repeat_reuse_blocker().empty())) {
    std::cerr << "[HPO] WARNING: " << w << "\n";
    HPOLog::instance().warning(w);
  }

  /* --- settings a search that ignores its start point does not read -----
   * A WARNING each (start_point_warnings, backend_rules.h): an INIT block
   * whose points such a search never evaluates, a RESTART block whose
   * rounds repeat round 0 (all of it when the search draws no random
   * numbers either, its first local search for GD_STOGO_RAND below
   * POPULATION 2n + 1), and the METHOD lines of a RESTART block on such a
   * search that draws them, which no round evaluates. */
  for (const std::string &w :
       start_point_warnings(trg.spec, trg.repeat_reuse_blocker())) {
    std::cerr << "[HPO] WARNING: " << w << "\n";
    HPOLog::instance().warning(w);
  }

  const bool has_fw = train_ctx.contains("FIXWEIGHT");
  const bool has_fi = train_ctx.contains("FIXINDEX");

  /* --- Loss sources ---------------------------------------------------- */
  const bool has_val_metrics = trg.use_emae || trg.use_ermse ||
                               trg.use_errmse || trg.use_ersq ||
                               trg.use_frmse || trg.use_srmse;
  /* The PC_FAILPROBE line, an entry of LS, gives no loss term: a file
   * whose only PC line it is scored 0 for every evaluation. */
  const std::size_t loss_sims =
      trg.LS.size() - (trg.failprobe_idx >= 0 ? 1u : 0u);
  if (!has_val_metrics && trg.observables.empty() && loss_sims == 0) {
    throw std::runtime_error(
        "No loss source: a loss term comes from a PC_EMAE, PC_ERMSE, "
        "PC_ErRMSE, PC_ERSQ, PC_FRMSE or PC_SRMSE_GPA line (with "
        "--validation), a PC_OBSERVABLE line or a PC_LAMMPS line, and the "
        "target file has none (a PC_FAILPROBE line gives no loss term)");
  }

  /* --- PC_OBSERVABLE: the elements of its structures --------------------
   * Each is an element of the potential: the elements of the run (ATOM,
   * which tadah hpo sets from the training data when the configuration
   * does not, or the training data's). One the potential lacks has no
   * descriptor: its cohesive energy was 0 for every candidate. */
  if (!trg.observables.empty()) {
    std::vector<std::string> run_elements;
    if (train_ctx.contains("ATOM")) {
      for (const auto &el : train_ctx("ATOM")) run_elements.push_back(el);
    } else if (has_train_db) {
      for (const auto &el : mlip::StructureDB::find_unique_elements(train_ctx))
        run_elements.push_back(el.symbol);
    }
    if (!run_elements.empty()) {
      std::string listed;
      for (const auto &el : run_elements)
        listed += (listed.empty() ? "" : ", ") + el;
      for (std::size_t k = 0; k < trg.observables.size(); ++k) {
        const auto &obs = trg.observables[k];
        if (!obs) continue;
        for (const auto &el : obs->elements()) {
          if (std::find(run_elements.begin(), run_elements.end(), el) !=
              run_elements.end())
            continue;
          const std::string what =
              "PC_OBSERVABLE " + obs->kind() + " --species " + el +
              ": the potential has no element " + el + " (its elements: " +
              listed + ")";
          if (k < trg.observable_lines.size() &&
              trg.observable_lines[k].number > 0)
            throw std::runtime_error(config_line_error(
                trg.observable_lines[k].number, what,
                trg.observable_lines[k].text));
          throw std::runtime_error(what);
        }
      }
    }
  }
  if (has_val_metrics && !has_validation) {
    throw std::runtime_error(
        "PC_EMAE/PC_ERMSE/PC_ErRMSE/PC_ERSQ/PC_FRMSE/PC_SRMSE_GPA target "
        "validation-set errors but no --validation file was given.");
  }

  /* --- the model of the -c configuration -------------------------------
   * It must build (probe_model): a construction error fails every
   * evaluation. Its weight count K serves the FIXINDEX rules below. */
  const ModelProbe model = probe_model(train_ctx);
  const std::size_t K = model.weights;

  /* --- OPTIM MPARAMS: a value the function reads, continuous ----------
   * The function reads its MPARAMS when it is built (mparams()): an index
   * past those it reads would change nothing, an integer value (the
   * Kern_Polynomial degree) is discrete, and a value that must be positive
   * (the Kern_RBF width) needs a positive box. MPARAMS holds numbers only,
   * so OPTIM index i is token i-1. */
  if (const auto it = trg.optimap.find("MPARAMS"); it != trg.optimap.end()) {
    const auto &o = it->second;
    const std::size_t n = model.mparams.size();
    std::string names;
    for (const auto &slot : model.mparams)
      names += (names.empty() ? "" : " ") + slot.name;
    for (std::size_t t = 0; t < o.opt.size(); ++t) {
      if (!o.opt[t]) continue;
      const std::string idx = core::to_string(t + 1);
      const std::string head =
          "OPTIM MPARAMS (" + idx + ") is not a valid search target: ";
      if (n == 0)
        throw std::runtime_error(
            head + model.function + " reads no MPARAMS, so every "
            "evaluation would train the same model.");
      if (t >= n)
        throw std::runtime_error(
            head + model.function + " reads " + core::to_string(n) +
            " MPARAMS value" + (n == 1 ? "" : "s") + " (" + names +
            "), so MPARAMS (" + idx + ") changes nothing.");
      const models::MParamSlot &slot = model.mparams[t];
      if (slot.integer)
        throw std::runtime_error(
            head + "MPARAMS (" + idx + ") of " + model.function +
            " is its " + slot.name + ", an integer, a discrete value rather "
            "than a continuous parameter; set it in the -c "
            "potential/configuration.");
      if (slot.positive && !(o.lowbound[t] > 0.0))
        throw std::runtime_error(
            "OPTIM MPARAMS (" + idx + "): the box [" +
            core::format_roundtrip(o.lowbound[t]) + ", " +
            core::format_roundtrip(o.highbound[t]) + "] of " +
            model.function + "'s " + slot.name + " must be positive (" +
            slot.name + " > 0).");
    }
  }

  /* --- OPTIM on a cutoff a descriptor takes from a file ---------------
   * D2_EAM and DM_EAM (also inside a D2_mJoin/DM_mJoin) evaluate the
   * tables of their setfl file and refuse a cutoff other than the file's
   * (D_Base::cutoff_fixed_by): every candidate but the -c value would fail
   * its evaluation. */
  for (const char *key : {"RCUT2B", "RCUTMB"}) {
    const auto it = trg.optimap.find(key);
    const auto fixed = model.cutoff_fixed.find(key);
    if (it == trg.optimap.end() || fixed == model.cutoff_fixed.end())
      continue;
    const auto &o = it->second;
    for (std::size_t t = 0; t < o.opt.size() && t < fixed->second.size();
         ++t) {
      if (!o.opt[t] || fixed->second[t].empty()) continue;
      const std::string idx = std::string(key) + " (" +
                              core::to_string(t + 1) + ")";
      throw std::runtime_error(
          "OPTIM " + idx + " is not a valid search target: " +
          fixed->second[t] + " and refuses another value, so every "
          "candidate but the -c value would fail; leave " + idx +
          " out of OPTIM.");
    }
  }

  /* --- FIXINDEX with BIAS true and two or more species ------------------
   * Which weights are the intercepts is the model function's (for a
   * linear model weights 1..N, and the descriptor weights start at index
   * N + 1; they started at 2 before 1.6.0, when BIAS true had one intercept
   * shared by all species): built as an evaluation builds it, after
   * probe_model() above checked that it builds. */
  if (train_ctx.has_stored("FIXINDEX") &&
      core::n_intercept_columns(train_ctx) >= 2) {
    core::Context c = train_ctx;
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer probe(c, ws);
    const std::string w =
        tadah::mlip::fixindex_intercept_warning(probe.context, *probe.fb);
    if (!w.empty()) {
      std::cerr << "[HPO] WARNING: " << w << "\n";
      HPOLog::instance().warning(w);
    }
  }

  /* --- FIXINDEX/FIXWEIGHT pairing + weight-source classification ------
   * classify_weight_source enforces what core::parse_fixindex refuses (a
   * lone FIXINDEX/FIXWEIGHT, a malformed or descending entry, an index
   * outside 1..K, a count other than FIXWEIGHT's, a weight named twice),
   * no-DB-without-pins and no-DB-with-partial-coverage, with actionable
   * messages. K >= 1: probe_model() above refuses a model without weights
   * ("The descriptor size is 0"). */
  if (has_fw || has_fi || !has_train_db)
    (void)tadah::mlip::classify_weight_source(train_ctx, K);

  /* --- the settings of the fit every evaluation makes ------------------
   * The implementation of M_KRR is part of the MODEL line of the potential,
   * which every best_pot.tadah carries: one other than 1 and 2 is refused,
   * with the fit's message, whether or not the evaluations fit.
   * An evaluation fits the model when the run has training data and
   * FIXINDEX does not pin every weight (fixindex_pins_every_weight; the
   * host provisions the pinned weights otherwise). Such a fit refuses a
   * LAMBDA that requests nothing (a negative value other than -1, a second
   * value), LAMBDA -1 under an OALGO other than 3 or under M_KRR
   * implementation 2, and OALGO 3 with a LAMBDA other than 0 when it has
   * an intercept column it fits (BIAS true): each evaluation would refuse
   * it and score FAILSCORE, so it is refused here, with the fit's message
   * (and OPTIM LAMBDA under OALGO 3 with such a column).
   * So is OPTIM of a key that M_KRR implementation 2 ignores (FSIGMA,
   * SSIGMA, WDBFILE, and OALGO, which its Cholesky solve does not read:
   * every candidate other than the default would be refused; its solve
   * applies ESIGMA); its other refusals depend on the training data, and
   * HPO_Host makes them once it has read them. A run whose evaluations
   * fit nothing reads none of these. */
  tadah::mlip::require_krr_implementation(train_ctx);
  if (has_train_db && !fixindex_pins_every_weight(train_ctx)) {
    tadah::mlip::require_fit_lambda(train_ctx);
    /* OALGO 3 refuses a LAMBDA other than 0 when the fit has an intercept
     * column FIXINDEX does not pin: the configuration's LAMBDA, with the
     * fit's message, and OPTIM LAMBDA, whose every candidate but 0 would
     * be refused. The intercept columns are the model function's, so the
     * function is built as an evaluation builds it (probe_model() above
     * checked that it builds); only for a setup the rule can refuse. */
    const auto lambda_it = trg.optimap.find("LAMBDA");
    const bool optim_lambda =
        lambda_it != trg.optimap.end() &&
        std::any_of(lambda_it->second.opt.begin(),
                    lambda_it->second.opt.end(), [](bool b) { return b; });
    if (core::n_intercept_columns(train_ctx) > 0 &&
        !tadah::mlip::is_krr_implementation_2(train_ctx) &&
        train_ctx.get<int>("OALGO", 0) == 3 &&
        (train_ctx.get<double>("LAMBDA", 0) != 0.0 || optim_lambda)) {
      core::Context c = train_ctx;
      tadah::mlip::memory::MLIPWorkspaceManager ws;
      tadah::mlip::Trainer probe(c, ws);
      tadah::mlip::require_oalgo3_intercepts(probe.context, *probe.fb);
      const std::size_t n =
          tadah::mlip::fitted_intercept_columns(probe.context, *probe.fb)
              .size();
      if (optim_lambda && n > 0)
        throw std::runtime_error(
            "OPTIM LAMBDA: OALGO 3 penalises every weight alike, and the fit "
            "has " + std::to_string(n) + " intercept column(s) (BIAS true "
            "with ATOM), which a ridge penalty leaves out, so every fit "
            "refuses a LAMBDA other than 0: each candidate of the box but 0 "
            "would fail its evaluation. OALGO 1, 2 or 4 fits a positive "
            "LAMBDA with the intercepts unpenalised.");
    }
    if (tadah::mlip::is_krr_implementation_2(train_ctx)) {
      const std::string kernel = train_ctx.get<std::string>("MODEL", 1);
      struct Ignored { const char *key, *instead; };
      for (const Ignored &k :
           {Ignored{"FSIGMA", "with FSIGMA 0.1 0, the default"},
            Ignored{"SSIGMA", "with SSIGMA 0.5 0, the default"},
            Ignored{"WDBFILE", "without WDBFILE"}}) {
        if (!optim_searches(trg, k.key)) continue;
        throw std::runtime_error(tadah::mlip::krr_implementation2_ignores(
            kernel, std::string("OPTIM ") + k.key, "applies it", k.instead));
      }
      // Its Cholesky solve reads no OALGO: every candidate of OPTIM OALGO
      // but the default would be refused.
      if (optim_searches(trg, "OALGO"))
        throw std::runtime_error(tadah::mlip::krr_implementation2_reads_no_oalgo(
            kernel, "OPTIM OALGO", "without OPTIM OALGO"));
    }
    require_error_scale_search_changes_the_fit(train_ctx, trg);
  } else {
    /* The evaluations fit no coefficient: an error scale or a priority of
     * the fit changes nothing. */
    for (const char *key : {"ESIGMA", "FSIGMA", "SSIGMA", "WDBFILE"})
      if (optim_searches(trg, key))
        throw std::runtime_error(
            std::string("OPTIM ") + key + ": the evaluations of this run "
            "fit no coefficient (" +
            (has_train_db ? "FIXINDEX pins every weight"
                          : "there is no training DBFILE") +
            "), so " + key + " would change nothing and every candidate "
            "would give the same potential.");
  }

  /* --- GRAD_STRATEGY ANALYTICAL needs the analytical gradient ----------
   * ANALYTICAL -- written out, or resolved from an unset GRAD_STRATEGY for
   * PC_OBSERVABLE -- stops the run when the analytical gradient cannot be
   * formed. On a setup whose fit settings the analytical gradient does not
   * cover that is every gradient, and with a loss term that has no
   * analytical derivative it is every gradient or every one taken at such
   * a value (analytical_gate.h), so refuse it here, not after the first
   * evaluation with "no registered analytical derivative". The blocks
   * that run an optimiser and obtain gradients: the STRATEGY-level block
   * and EXPLORE. An INNER block has no GRAD_STRATEGY (the parser rejects
   * one): MLSL and AUGLAG hand their local search the gradient of their
   * own objective, which the enclosing block's GRAD_STRATEGY obtains
   * (nlopt_adapter.h). */
  {
    struct Block { const OptimizerSpec* sp; const char* name; };
    std::vector<Block> blocks = {{&trg.spec, "OPTIMIZER"}};
    if (trg.spec.prestage)
      blocks.push_back({trg.spec.prestage.get(), "EXPLORE"});
    std::string why;
    bool checked = false;
    for (const auto& b : blocks) {
      if (b.sp->grad_strategy != GradStrategy::Analytical ||
          !obtains_gradients(*b.sp))
        continue;
      if (!checked) {
        why = analytical_gradient_blocker(train_ctx, trg, has_train_db);
        if (why.empty()) why = analytical_loss_term_blocker(trg);
        if (why.empty()) why = error_scale_search_blocker(trg);
        checked = true;
      }
      if (why.empty()) break;
      /* At the line the block closes at, as the parser's refusals of a
       * block (with_block_line). */
      throw std::runtime_error(with_block_line(*b.sp,
          std::string(b.name) + " block: GRAD_STRATEGY ANALYTICAL" +
          (b.sp->grad_strategy_resolved
               ? " (resolved for PC_OBSERVABLE: the block sets no "
                 "GRAD_STRATEGY)"
               : "") +
          " needs the analytical gradient of the loss, which this setup "
          "cannot form: " + why + ". Change that setting, or set "
          "GRAD_STRATEGY HYBRID to use finite differences where no "
          "analytical gradient is available."));
    }
  }

  /* --- FAILSCORE_GRADING ON bands the failures below FAILSCORE ---------- */
  if (trg.FAILSCORE_GRADING) {
    /* The band arithmetic needs a finite FAILSCORE: with the legacy
     * DBL_MAX default, 0.999*FS and FS are not meaningfully ordered in
     * loss space and L2 of the substituted value overflows. */
    if (!std::isfinite(trg.FAIL_SCORE) || trg.FAIL_SCORE >= 1.0e300 ||
        trg.FAIL_SCORE <= 0.0)
      throw std::runtime_error(
          "FAILSCORE_GRADING ON requires a finite positive FAILSCORE (e.g. "
          "`FAILSCORE 1e9`); the default DBL_MAX sentinel cannot be banded.");
    if (!graded_failscore_is_ordered(trg.FAIL_SCORE))
      throw std::runtime_error(
          "FAILSCORE_GRADING ON requires a FAILSCORE large enough to "
          "represent four strictly ordered grading bands.");
  } else {
    /* Under OFF a failure signal (a --failvar above 0, a PC_FAILPROBE S
     * above --gate) fails its evaluation, which scores FAILSCORE; the
     * settings that only grade a failure would have no effect. Refused at
     * their line (after the TADAH_FAILSCORE_GRADING override, which
     * hpo_run applies before these checks). */
    auto at = [](const LammpsSim &sim, const std::string &what) {
      return sim.line > 0 ? config_line_error(sim.line, what, sim.line_text)
                          : what;
    };
    if (trg.failprobe_s0_given && trg.failprobe_idx >= 0)
      throw std::runtime_error(at(
          trg.LS[static_cast<std::size_t>(trg.failprobe_idx)],
          "PC_FAILPROBE --s0 grades failed evaluations, which "
          "FAILSCORE_GRADING ON does; under FAILSCORE_GRADING OFF every "
          "failed evaluation scores FAILSCORE, so --s0 would have no "
          "effect"));
    for (const auto &sim : trg.LS)
      for (const auto &fv : sim.failvars)
        if (fv.scale_given) {
          std::ostringstream scale;
          scale << fv.scale;
          throw std::runtime_error(at(sim,
              "--failvar " + fv.name + " " + scale.str() +
              ": the scale grades the failure, which FAILSCORE_GRADING ON "
              "does; under FAILSCORE_GRADING OFF a failed evaluation scores "
              "FAILSCORE, so the scale would have no effect (--failvar " +
              fv.name + " alone fails the evaluation when " + fv.name +
              " > 0)"));
        }
  }

  /* --- RCUT contract for refit-family channels -------------------------- */
  /* WARNINGs and INFO lines of the run, as every startup check writes
   * them: a WARNING on stderr and in the journal (which tadah hpo --check
   * lists and counts), an INFO line in the journal. */
  auto warn = [](const std::string &m) {
    std::cerr << "[HPO] WARNING: " << m << "\n";
    HPOLog::instance().warning(m);
  };
  auto info = [](const std::string &m) {
    std::cout << "[HPO] " << m << "\n";
    HPOLog::instance().info(m);
  };

  /* Knot channels: the basis vanishes at its largest knot, so RCUT2B/RCUTMB
   * are DERIVED. Correct a stale stated value (warns) BEFORE the host sizes
   * the one-time neighbour list from RCUTMAX, and flag a user's OPTIM RCUT
   * line — that dimension is overridden on every evaluation. */
  {
    std::vector<models::potedit::KnotRcutNote> notes;
    models::potedit::normalize_knot_rcut(train_ctx, "tadah hpo", notes);
    for (const auto &n : notes) {
      if (n.level == core::LogLevel::WARNING) warn(n.text);
      else HPOLog::instance().info(n.text);
    }
  }
  for (const char *sfx : {"2B", "MB"}) {
    if (!models::potedit::derived_knot_rcut(train_ctx, sfx)) continue;
    const std::string rkey = std::string("RCUT") + sfx;
    if (trg.optimap.count(rkey)) {
      warn("OPTIM " + rkey + " has no effect for the knot basis family: the "
           "effective cutoff is the largest CGRID" + std::string(sfx) +
           " knot and " + rkey + " is re-derived on every evaluation. Remove "
           "the OPTIM " + rkey + " line (a wasted search dimension) and vary "
           "CGRID" + std::string(sfx) + " to change the effective cutoff.");
    }
  }

  /* Bump (blip) channels: RCUT is authoritative — never auto-extended.
   * Under a DISCONTINUOUS cutoff (Cut_Dummy) a bump support crossing RCUT
   * is hard-truncated -> phi(RCUT) != 0 -> an energy/force step whenever a
   * neighbour crosses the cutoff. Authoring a search that permits it is a
   * hard ERROR (the bump-support gate): every violation is collected and
   * reported with the exact bound that fixes it. Under a smooth cutoff the
   * overhang is damped to zero at RCUT — legitimate, no gate (a centre
   * beyond rcut still gets a warning: only its inner shoulder contributes).
   * To search longer range, raise RCUT2B/RCUTMB in the config. */
  std::vector<std::string> gate_errors;
  for (const char *sfx : {"2B", "MB"}) {
    const auto wopt = models::potedit::bump_channel_halfwidth(train_ctx, sfx);
    if (!wopt) continue;
    const double w = *wopt;
    const std::string ckey = std::string("CGRID") + sfx;
    const std::string skey = std::string("SGRID") + sfx;
    const std::string rkey = std::string("RCUT") + sfx;
    /* The descriptor of this channel was built by probe_model() above, so
     * its grids are readable and pair up (read_bump_grids throws
     * otherwise). */
    const auto [mu, eta] = detail::read_bump_grids(train_ctx, ckey, skey);

    const bool hard =
        models::potedit::channel_cutoff_is_discontinuous(train_ctx, sfx);

    /* The stated cutoff (the seed pot's own truncation): RCUT{sfx}MAX, the
     * max over a per-pair RCUT list. */
    const std::string rmax = rkey + "MAX";
    const double rcut0 =
        (train_ctx.contains(rmax) && train_ctx.size(rmax) > 0)
            ? train_ctx.get<double>(rmax)
            : train_ctx.get<double>(rkey);
    /* Cutoff range candidates can reach when RCUT itself is OPTIM'd
     * (shared with the per-candidate gate table so the two cannot drift). */
    const auto brc = detail::derive_binding_rcut(train_ctx, trg, sfx);
    const bool rcut_is_optim = !brc.flat_idx.empty();
    /* STRICT (legacy) binding: every trial satisfies rcut >= its OPTIM low
     * bound, so the corner gate binds against min(stated, low bounds). */
    double rcut_strict = rcut0;
    {
      const auto itr = trg.optimap.find(rkey);
      if (itr != trg.optimap.end())
        for (std::size_t i = 0; i < itr->second.opt.size(); ++i)
          if (itr->second.opt[i])
            rcut_strict = std::min(rcut_strict, itr->second.lowbound[i]);
    }
    /* Seed binding: STRICT keeps the legacy min'd value; CANDIDATE binds
     * the seed against its OWN stated cutoff (a sampled RCUT below the
     * seed's reach is a per-candidate rejection, not a startup error). */
    const double rcut = trg.BUMPGATE_STRICT ? rcut_strict : rcut0;
    const double tol = models::bump_rcut_tol(rcut);
    const std::string rname =
        (trg.BUMPGATE_STRICT && rcut_is_optim)
            ? ("the OPTIM " + rkey + " low bound")
            : rkey;
    if (hard && rcut_is_optim) {
      if (trg.BUMPGATE_STRICT) {
        warn("OPTIM " + rkey + " on a bump channel is a wasted search "
             "dimension: the bump-support gate keeps every support inside "
             "its LOW bound (" + core::to_string(rcut_strict) + "), so the "
             "cutoff value cannot affect the model. Remove the OPTIM " +
             rkey + " line and set " + rkey + " in the config instead.");
      } else {
        warn("OPTIM " + rkey + " on a bump channel is usually a wasted "
             "search dimension: every feasible candidate's bumps vanish at "
             "their own support edge inside the cutoff, so the sampled " +
             rkey + " value cannot change the model's energies — it only "
             "widens or narrows the feasible (c, eta) region. Prefer a "
             "fixed " + rkey + " in the config.");
      }
    }

    /* (a) the seed pot's own bumps must fit (every width > 0: checked
     * above) */
    for (std::size_t i = 0; i < mu.size(); ++i) {
      const double reach = mu[i] + w / eta[i];
      if (mu[i] > rcut + tol) {
        const std::string m =
            ckey + " bump " + core::to_string(i + 1) + " centre " +
            core::to_string(mu[i]) + " lies beyond " + rname + " = " +
            core::to_string(rcut) + " (pathological: only its inner "
            "shoulder can contribute) — move the centre below the cutoff.";
        if (hard) gate_errors.push_back(m);
        else warn(m);
        continue;
      }
      /* A search over the cutoff tries cutoffs down to the OPTIM low
       * bound: below the centre a candidate keeps the inner shoulder only.
       * A hard cutoff rejects such a candidate per candidate (the bump
       * gate); a smooth one evaluates it, so say so. */
      if (!hard && rcut_strict < rcut &&
          mu[i] > rcut_strict + models::bump_rcut_tol(rcut_strict))
        warn(ckey + " bump " + core::to_string(i + 1) + " centre " +
             core::to_string(mu[i]) + " lies beyond the OPTIM " + rkey +
             " low bound " + core::to_string(rcut_strict) + ": a candidate "
             "whose " + rkey + " is below the centre keeps only the bump's "
             "inner shoulder (pathological); with the OPTIM " + rkey +
             " low bound above " + core::to_string(mu[i]) + ", or the "
             "centre below " + core::to_string(rcut_strict) + ", no "
             "candidate does.");
      /* An OPTIM box of the centre reaching past the cutoff (a smooth
       * one: under a hard cutoff the bump-support gate below owns the
       * box): a candidate whose centre lies there keeps the inner shoulder
       * only, as a seed centre beyond the cutoff does. The cutoff is the
       * fixed RCUT, or with an OPTIM RCUT the low bound of its box. */
      if (!hard) {
        const auto itc_box = trg.optimap.find(ckey);
        if (itc_box != trg.optimap.end() && i < itc_box->second.opt.size() &&
            itc_box->second.opt[i]) {
          const double hi = itc_box->second.highbound[i];
          const double edge = rcut_strict;
          if (hi > edge + models::bump_rcut_tol(edge)) {
            const std::string bound =
                rcut_strict < rcut
                    ? "the OPTIM " + rkey + " low bound " +
                          core::to_string(edge)
                    : rkey + " = " + core::to_string(edge);
            warn(ckey + " bump " + core::to_string(i + 1) + ": the OPTIM " +
                 ckey + " (" + core::to_string(i + 1) + ") high bound " +
                 core::to_string(hi) + " lies beyond " + bound +
                 ": a candidate whose centre is beyond the cutoff keeps "
                 "only the bump's inner shoulder (pathological); with the "
                 "high bound at or below " + core::to_string(edge) +
                 ", no candidate does.");
          }
        }
      }
      if (!hard || reach <= rcut + tol) continue;
      gate_errors.push_back(
          ckey + " bump " + core::to_string(i + 1) + " (centre " +
          core::to_string(mu[i]) + ", eta " + core::to_string(eta[i]) +
          ") reaches " + core::to_string(reach) + ", past " + rname + " = " +
          core::to_string(rcut) + " — set " + skey + " (" +
          core::to_string(i + 1) + ") >= " +
          core::to_string(w / (rcut - mu[i])) + ", move the centre inward "
          "(tadah refit --edit), or raise " + rkey + ".");
    }
    if (!hard) continue;  // smooth cutoff: overhanging boxes are legitimate

    /* (b) the OPTIM boxes. STRICT: the worst corner (CGRID high bound,
     * SGRID low bound) must fit — the legacy gate. CANDIDATE (default):
     * only a box with NO feasible point is an error; a partially feasible
     * box arms the per-candidate gate (HPO_Host rejects violating trials
     * before training as graded band-D failures) and is announced with an
     * INFO line per bump. */
    const auto itc = trg.optimap.find(ckey);
    const auto its = trg.optimap.find(skey);
    if (trg.BUMPGATE_STRICT) {
      for (std::size_t i = 0; i < mu.size(); ++i) {
        double hi_mu = mu[i];
        if (itc != trg.optimap.end() && i < itc->second.opt.size() &&
            itc->second.opt[i])
          hi_mu = itc->second.highbound[i];
        /* the OPTIM low bound of a width is > 0
         * (models::optim_low_bound_rule, at parse) */
        double eta_i = eta[i];
        if (its != trg.optimap.end() && i < its->second.opt.size() &&
            its->second.opt[i])
          eta_i = std::min(eta_i, its->second.lowbound[i]);
        if (hi_mu + w / eta_i <= rcut + tol) continue;
        if (hi_mu >= rcut - tol) {
          gate_errors.push_back(
              "the OPTIM bounds let bump " + core::to_string(i + 1) + " of " +
              ckey + " reach " + core::to_string(hi_mu + w / eta_i) +
              ", past " + rname + " = " + core::to_string(rcut) +
              " — impossible while the OPTIM " + ckey + " (" +
              core::to_string(i + 1) + ") high bound " + core::to_string(hi_mu) +
              " >= " + core::to_string(rcut) + ": lower the " + ckey +
              " high bound below " + rkey + " first.");
        } else {
          gate_errors.push_back(
              "the OPTIM bounds let bump " + core::to_string(i + 1) + " of " +
              ckey + " reach " + core::to_string(hi_mu + w / eta_i) +
              ", past " + rname + " = " + core::to_string(rcut) +
              " — need the OPTIM " + skey + " (" + core::to_string(i + 1) +
              ") low bound >= w/(" + rkey + " - c_hi) = " +
              core::to_string(w / (rcut - hi_mu)) + " (currently " +
              core::to_string(eta_i) + ").");
        }
      }
    } else {
      const double tol_best = models::bump_rcut_tol(brc.max_reachable);
      const double tol_min = models::bump_rcut_tol(brc.min_reachable);
      for (std::size_t i = 0; i < mu.size(); ++i) {
        const bool c_opt = itc != trg.optimap.end() &&
                           i < itc->second.opt.size() && itc->second.opt[i];
        const bool s_opt = its != trg.optimap.end() &&
                           i < its->second.opt.size() && its->second.opt[i];
        if (!c_opt && !s_opt && !rcut_is_optim)
          continue;  /* fully static bump: the seed check (a) owns it */
        const double lo_mu = c_opt ? itc->second.lowbound[i] : mu[i];
        const double hi_mu = c_opt ? itc->second.highbound[i] : mu[i];
        /* every width, seed and box, is > 0 (checked above; the OPTIM
         * low bound at parse) */
        const double eta_hi = s_opt ? its->second.highbound[i] : eta[i];
        const double eta_lo = s_opt ? its->second.lowbound[i] : eta[i];
        /* best corner: is there ANY feasible point in the box? */
        if (lo_mu + w / eta_hi > brc.max_reachable + tol_best) {
          if (lo_mu >= brc.max_reachable - tol_best) {
            gate_errors.push_back(
                "no feasible point for bump " + core::to_string(i + 1) +
                " of " + ckey + ": even its most favorable corner (centre " +
                core::to_string(lo_mu) + ", eta " + core::to_string(eta_hi) +
                ") reaches " + core::to_string(lo_mu + w / eta_hi) +
                ", past " + rkey + " = " + core::to_string(brc.max_reachable) +
                " — impossible while the " + ckey + " low bound is at/above " +
                rkey + ": lower the " + ckey + " low bound below " + rkey +
                " or raise " + rkey + ".");
          } else {
            gate_errors.push_back(
                "no feasible point for bump " + core::to_string(i + 1) +
                " of " + ckey + ": even its most favorable corner (centre " +
                core::to_string(lo_mu) + ", eta " + core::to_string(eta_hi) +
                ") reaches " + core::to_string(lo_mu + w / eta_hi) +
                ", past " + rkey + " = " + core::to_string(brc.max_reachable) +
                " — need the OPTIM " + skey + " (" + core::to_string(i + 1) +
                ") high bound >= w/(" + rkey + " - c_lo) = " +
                core::to_string(w / (brc.max_reachable - lo_mu)) +
                ", a lower " + ckey + " low bound, or a larger " + rkey + ".");
          }
          continue;
        }
        /* worst corner: partially infeasible box -> per-candidate gate */
        if (hi_mu + w / eta_lo > brc.min_reachable + tol_min) {
          const std::string rdesc =
              rcut_is_optim ? ("the sampled " + rkey)
                            : (rkey + " = " + core::to_string(brc.min_reachable));
          info("bump-support gate (CANDIDATE): armed for " + ckey + " bump " +
               core::to_string(i + 1) + " — candidates with c + " +
               core::to_string(w) + "/eta > " + rdesc + " are rejected "
               "before training as graded band-D failures (worst box corner "
               "reaches " + core::to_string(hi_mu + w / eta_lo) + "). "
               "BUMPGATE STRICT restores the startup box-corner error.");
        }
      }
    }
  }
  if (!gate_errors.empty()) {
    std::string msg =
        "bump-support gate: with a hard (Cut_Dummy) cutoff every compact "
        "bump must vanish at or before RCUT (c + w/eta <= RCUT) — a "
        "truncated overhang is an energy/force STEP at the cutoff. "
        "Violations:";
    for (const auto &e : gate_errors) msg += "\n  - " + e;
    msg += "\n  To search longer range, raise RCUT2B/RCUTMB in the config "
           "(the potential stays smooth: bumps still vanish at their own "
           "support edge).";
    throw std::runtime_error(msg);
  }

  if(!train_ctx.blocks("DESCRIPTOR").empty())
    (void)detail::build_bump_gate_table(train_ctx,trg);
  /* DM_REAM density gates (RHOGATE, RHOBARGATE; rho_gates.h): the
   * semantic checks, then every CONFIG/WARM seed evaluated as a candidate
   * by the per-candidate functions (SEED ERROR | WARN), before any data is
   * loaded. After the knot-RCUT normalisation above, so each seed sees the
   * cutoff the host will. */
  detail::validate_density_gates(train_ctx, trg);

  /* Search-space / initial-step rules (scale_validation.h): STEP policy
   * from the caps registry (already applied at parse time), every
   * resolved seed inside its OPTIM box, and the width audit. Runs last so
   * the physics gates above (knot RCUT, bump support, density) report
   * first. */
  run_scale_validation(trg.spec, /*check_start_points=*/true);
}

std::string validation_config_file(const core::Context &ctx) {
  if (!ctx.has_stored("VALIDATION")) return "";
  const std::size_t n = ctx.size("VALIDATION");
  if (n == 1) return ctx.get<std::string>("VALIDATION");
  std::string names;
  for (std::size_t i = 0; i < n; ++i)
    names += (i ? " " : "") + ctx.get<std::string>("VALIDATION", i);
  throw std::runtime_error(
      "--validation (VALIDATION) takes one validation configuration file, "
      "whose DBFILE lines list the validation datasets; it has " +
      std::to_string(n) + " values: " + names +
      ". List every validation dataset as a DBFILE line of one file.");
}

void require_distinct_validation_datasets(const core::Context &val_ctx,
                                          const std::string &val_cfg) {
  if (!val_ctx.has_stored("DBFILE")) return;
  const std::vector<std::string> &files = val_ctx("DBFILE");
  const auto twice = tadah::mlip::repeated_dataset(files);
  if (!twice) return;
  const auto [i, j] = *twice;
  throw std::runtime_error(
      "the --validation file " + val_cfg + ": DBFILE lists one dataset "
      "twice: entry " + std::to_string(i + 1) + " (" + files[i] +
      ") and entry " + std::to_string(j + 1) + " (" + files[j] +
      ") are one file, so it would count twice in the average over the "
      "validation datasets of every validation metric. List each dataset "
      "once; to weight the datasets, use indexed PC_ERMSE (i) and "
      "PC_FRMSE (i) lines.");
}

} // namespace hpo
} // namespace tadah

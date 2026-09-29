#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/analytical_gate.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/strategy_rules.h>

#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/descriptors/dm/ream_density.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
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

/* The model of the -c configuration builds, as every evaluation builds
 * it (the Trainer: the descriptors, the basis function or kernel and the
 * model): a construction error here -- a DIMER potential with element-
 * named TYPE2B pairs, a descriptor that refuses its grids -- would fail
 * every evaluation (the run then ends "no evaluation produced a usable
 * loss"). Returns the model's weight count (derive_weight_count).
 * models::DensityFormError keeps its type and message. */
std::size_t probe_model(const core::Context &train_ctx) {
  try {
    core::Context c = train_ctx;
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer probe(c, ws);
    return probe.fb->get_phi_cols(probe.context);
  } catch (const models::DensityFormError &) {
    throw;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("the -c configuration does not build a model: ") +
        e.what() + ". Every evaluation would fail; fix the -c "
        "potential/configuration.");
  }
}

/* The phases of the run (strategy_phase_specs, phase_specs.h: the
 * STRATEGY-level optimiser, each MULTISTART start, the PRESTAGE exploration
 * and refinement, every INNER block) whose optimiser can evaluate outside
 * the OPTIM box: an ignores_bounds backend (optimizer_caps) or PARAM
 * fd_kernel ceres on a phase that asks the host for gradients
 * (obtains_gradients: a gradient-based backend, or an MLSL/AUGLAG whose
 * local search, inner_run_spec, is one). "<phase> <LIB>:<ALGO> (<why>)"
 * per entry. Called after validate_phase_backends, so every phase names a
 * known optimiser. */
std::vector<std::string> unbounded_optimisers(const OptimizerSpec &root) {
  std::vector<std::string> out;
  for (const PhaseSpec &phase : strategy_phase_specs(root)) {
    const OptimizerSpec &s = phase.spec;
    if (s.lib == "NOT SET") continue;
    const OptimizerCaps &caps = optimizer_caps(s);
    const std::string name = phase.name + " " + s.lib + ":" + s.algo;
    if (caps.ignores_bounds) {
      out.push_back(name + " (ignores the OPTIM bounds)");
      continue;
    }
    const auto fk = s.param_raw.find("fd_kernel");
    if (obtains_gradients(s) && fk != s.param_raw.end() &&
        core::to_upper(fk->second) == "CERES")
      out.push_back(name + " (PARAM fd_kernel ceres: unclamped "
                    "finite-difference steps)");
  }
  return out;
}

}  // namespace

std::size_t derive_weight_count(const core::Context &train_ctx) {
  core::Context ctx_copy = train_ctx;
  tadah::mlip::memory::MLIPWorkspaceManager ws;
  tadah::mlip::Trainer probe(ctx_copy, ws);
  return probe.fb->get_phi_cols(probe.context);
}

RawEmbeddingF0 derive_raw_embedding_f0(const core::Context &train_ctx) {
  RawEmbeddingF0 out;
  if (!models::context_has_ream(train_ctx)) {
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
    if (pc.has_stored("FIXINDEX") && pc.has_stored("FIXWEIGHT")) {
      std::vector<std::string> fi(pc.size("FIXINDEX"));
      pc.get("FIXINDEX", fi);
      if (core::parse_indices(fi).size() == K) {
        probe.provision_fixed_weights();
        w = probe.model->get_weights();
        out.source = "FIXWEIGHT";
      }
    }
    if (w.size() == 0 && pc.has_stored("WEIGHTS") && pc.size("WEIGHTS") == K) {
      w = core::t_type(K);
      for (std::size_t i = 0; i < K; ++i) w(i) = pc.get<double>("WEIGHTS", i);
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

void validate_hpo_setup(core::Context &train_ctx, const HPOTargets &trg,
                        bool has_train_db, bool has_validation) {
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
    const std::string e = HPOTargets::optim_structure_key_error(kv.first);
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
   * the cutoff. Owner decision 2026-09-29. */
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

  /* --- search-space dimension ------------------------------------------ */
  if (trg.spec.low.empty()) {
    const std::string &lib = trg.spec.lib;
    if (lib == "NLOPT" || lib == "DLIB" || lib == "CERES") {
      throw std::runtime_error(
          "No OPTIM line: the search space has zero dimensions, and LIB " +
          lib + " cannot optimise a zero-dimensional space (the failure "
          "would only surface once the optimiser starts). Add at least one "
          "OPTIM line, or use a TADAH-native optimiser, which evaluates the "
          "INIT start point once.");
    }
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
   * The attribute and PARAM checks of each backend, and its need for a
   * MAXEVAL, on the spec each phase runs with (phase_specs.h: the
   * STRATEGY-level optimiser, each MULTISTART start with its budget, the
   * PRESTAGE exploration and refinement, every INNER block). They used to
   * run only when that optimiser started: after the data were loaded, and
   * for a PRESTAGE refinement after the whole exploration. Each driver
   * applies its own checks again when it starts (backend_rules.h). */
  validate_phase_backends(trg.spec);

  /* --- RHOFORM NONNEG / MONOTONE: optimisers that can leave the box -----
   * The evaluated AMPGRIDMB coordinates are projected onto [0, inf)
   * (HPO_Host); say so when the optimiser of a phase can put them outside
   * the (non-negative) box. After the phase checks above, on the same
   * phases. */
  if (models::context_has_ream(train_ctx)) {
    const models::RhoForm form = models::require_rhoform(train_ctx, "tadah hpo");
    bool amp_optim = false;
    for (const auto &kv : trg.optimap)
      if (models::optim_key_lower_bound(kv.first, form) ==
          models::OptimLowerBound::NonNegative)
        for (bool b : kv.second.opt) amp_optim = amp_optim || b;
    if (amp_optim) {
      const std::vector<std::string> ub = unbounded_optimisers(trg.spec);
      if (!ub.empty()) {
        std::string msg = std::string("RHOFORM ") + models::rhoform_name(form) +
                          ": ";
        for (std::size_t i = 0; i < ub.size(); ++i)
          msg += (i ? ", " : "") + ub[i];
        msg += " can evaluate AMPGRIDMB amplitudes outside the non-negative "
               "OPTIM box. An amplitude below 0 is not a valid " +
               std::string(models::rhoform_name(form)) + " density: every "
               "such point is projected onto AMPGRIDMB >= 0, evaluated and "
               "recorded there (the optimiser still sees its own point); "
               "DIAGNOSTICS counts the projections.";
        std::cerr << "[HPO] WARNING: " << msg << "\n";
        HPOLog::instance().warning(msg);
      }
    }
  }

  const bool has_fw = train_ctx.contains("FIXWEIGHT");
  const bool has_fi = train_ctx.contains("FIXINDEX");

  /* --- Loss sources ---------------------------------------------------- */
  const bool has_val_metrics = trg.use_emae || trg.use_ermse ||
                               trg.use_errmse || trg.use_ersq ||
                               trg.use_frmse || trg.use_srmse;
  if (!has_val_metrics && trg.observables.empty() && trg.LS.empty()) {
    throw std::runtime_error(
        "No loss source: declare at least one of PC_EMAE/PC_ERMSE/"
        "PC_ErRMSE/PC_ERSQ/PC_FRMSE/PC_SRMSE (requires --validation), "
        "PC_OBSERVABLE, or PC_LAMMPS in the hpotarget file.");
  }
  if (has_val_metrics && !has_validation) {
    throw std::runtime_error(
        "PC_EMAE/PC_ERMSE/PC_ErRMSE/PC_ERSQ/PC_FRMSE/PC_SRMSE target "
        "validation-set errors but no --validation file was given.");
  }

  /* --- NORM is incompatible with fixed weights -------------------------- */
  if ((has_fw || has_fi) && train_ctx.contains("NORM") &&
      train_ctx.get<bool>("NORM")) {
    throw std::runtime_error(
        "FIXINDEX/FIXWEIGHT is not supported with NORM true: fixed values "
        "would be interpreted in the normalized feature space. Set NORM "
        "false.");
  }

  /* --- the model of the -c configuration -------------------------------
   * It must build (probe_model): a construction error fails every
   * evaluation. Its weight count K serves the FIXINDEX rules below. */
  const std::size_t K = probe_model(train_ctx);

  /* --- FIXINDEX/FIXWEIGHT pairing + weight-source classification ------ */
  if (has_fw || has_fi || !has_train_db) {
    if (K > 0) {
      /* classify_weight_source enforces: lone FIXINDEX/FIXWEIGHT, size
       * mismatch, out-of-bounds indices, no-DB-without-pins and
       * no-DB-with-partial-coverage — with actionable messages. */
      (void)tadah::mlip::classify_weight_source(train_ctx, K);
    } else {
      /* A model without weights (K = 0): enforce what does not need K. */
      if (has_fw != has_fi) {
        throw std::runtime_error(
            "FIXINDEX requires FIXWEIGHT (and vice versa). Provide both "
            "keys with matching lengths; the i-th FIXWEIGHT value pins the "
            "weight named by the i-th FIXINDEX entry. There is no "
            "auto-seeding from WEIGHTS.");
      }
      if (has_fw && has_fi) {
        std::vector<std::string> fi_str(train_ctx.size("FIXINDEX"));
        train_ctx.get("FIXINDEX", fi_str);
        const std::size_t n_idx = core::parse_indices(fi_str).size();
        const std::size_t n_val = train_ctx.size("FIXWEIGHT");
        if (n_idx != n_val) {
          throw std::runtime_error(
              "FIXINDEX names " + core::to_string(n_idx) +
              " weights but FIXWEIGHT provides " + core::to_string(n_val) +
              " values; they must pair 1:1 (the i-th value pins the i-th "
              "index).");
        }
      }
      if (!has_train_db && !has_fi) {
        throw std::runtime_error(
            "No training DBFILE and no FIXINDEX/FIXWEIGHT: the model "
            "weights cannot be determined. Provide a DBFILE for "
            "least-squares training, or pin every weight with "
            "FIXINDEX/FIXWEIGHT.");
      }
    }
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
        checked = true;
      }
      if (why.empty()) break;
      throw std::runtime_error(
          std::string(b.name) + " block: GRAD_STRATEGY ANALYTICAL" +
          (b.sp->grad_strategy_resolved
               ? " (resolved for PC_OBSERVABLE: the block sets no "
                 "GRAD_STRATEGY)"
               : "") +
          " needs the analytical gradient of the loss, which this setup "
          "cannot form: " + why + ". Change that setting, or set "
          "GRAD_STRATEGY HYBRID to use finite differences where no "
          "analytical gradient is available.");
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
  }

  /* --- RCUT contract for refit-family channels -------------------------- */
  auto warn = [](const std::string &m) {
    core::Logger::instance().log(core::LogLevel::WARNING, m);
  };
  auto info = [](const std::string &m) {
    core::Logger::instance().log(core::LogLevel::INFO, m);
  };

  /* Knot channels: the basis vanishes at its largest knot, so RCUT2B/RCUTMB
   * are DERIVED. Correct a stale stated value (warns) BEFORE the host sizes
   * the one-time neighbour list from RCUTMAX, and flag a user's OPTIM RCUT
   * line — that dimension is overridden on every evaluation. */
  models::potedit::normalize_knot_rcut(train_ctx, "tadah hpo");
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

} // namespace hpo
} // namespace tadah

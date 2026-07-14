#include <tadah/hpo/startup_validation.h>

#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

std::size_t derive_weight_count(const core::Context &train_ctx) {
  try {
    core::Context ctx_copy = train_ctx;
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer probe(ctx_copy, ws);
    return probe.fb->get_phi_cols(probe.context);
  } catch (const std::exception &e) {
    core::Logger::instance().log(
        core::LogLevel::WARNING,
        "startup validation: could not derive the model weight count (",
        e.what(), "); FIXINDEX bounds/coverage checks are deferred to the "
        "first training evaluation.");
    return 0;
  }
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
  if (trg.optimap.count("FIXWEIGHT") && !train_ctx.contains("FIXINDEX")) {
    throw std::runtime_error(
        "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration "
        "(FIXINDEX names which model weights the FIXWEIGHT values pin).");
  }

  /* --- FIXINDEX/FIXWEIGHT pairing + weight-source classification ------ */
  const bool has_fw = train_ctx.contains("FIXWEIGHT");
  const bool has_fi = train_ctx.contains("FIXINDEX");
  if (has_fw || has_fi || !has_train_db) {
    const std::size_t K = derive_weight_count(train_ctx);
    if (K > 0) {
      /* classify_weight_source enforces: lone FIXINDEX/FIXWEIGHT, size
       * mismatch, out-of-bounds indices, no-DB-without-pins and
       * no-DB-with-partial-coverage — with actionable messages. */
      (void)tadah::mlip::classify_weight_source(train_ctx, K);
    } else {
      /* K underivable: enforce what does not need K. */
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

  /* --- RCUT contract for refit-family channels -------------------------- */
  auto warn = [](const std::string &m) {
    core::Logger::instance().log(core::LogLevel::WARNING, m);
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
    if (!train_ctx.contains(ckey) || !train_ctx.contains(skey) ||
        !train_ctx.contains(rkey))
      continue;
    core::v_type mu, eta;
    try {
      models::D_Base::get_grid(train_ctx, ckey, mu);
      models::D_Base::get_grid(train_ctx, skey, eta);
    } catch (const std::exception &) { continue; }
    if (mu.empty() || mu.size() != eta.size()) continue;

    const bool hard =
        models::potedit::channel_cutoff_is_discontinuous(train_ctx, sfx);

    /* The binding rcut of the gate. The cutoff object truncates at
     * RCUT{sfx}MAX (the max over a per-pair RCUT list); when RCUT itself is
     * an OPTIM dimension, every trial satisfies rcut >= its OPTIM low
     * bound, so the gate binds against that. */
    const std::string rmax = rkey + "MAX";
    double rcut = (train_ctx.contains(rmax) && train_ctx.size(rmax) > 0)
                      ? train_ctx.get<double>(rmax)
                      : train_ctx.get<double>(rkey);
    bool rcut_is_optim = false;
    const auto itr = trg.optimap.find(rkey);
    if (itr != trg.optimap.end()) {
      for (std::size_t i = 0; i < itr->second.opt.size(); ++i) {
        if (!itr->second.opt[i]) continue;
        rcut = std::min(rcut, itr->second.lowbound[i]);
        rcut_is_optim = true;
      }
    }
    const double tol = models::bump_rcut_tol(rcut);
    const std::string rname =
        rcut_is_optim ? ("the OPTIM " + rkey + " low bound") : rkey;
    if (hard && rcut_is_optim) {
      warn("OPTIM " + rkey + " on a bump channel is a wasted search "
           "dimension: the bump-support gate keeps every support inside its "
           "LOW bound (" + core::to_string(rcut) + "), so the cutoff value "
           "cannot affect the model. Remove the OPTIM " + rkey + " line and "
           "set " + rkey + " in the config instead.");
    }

    /* (a) the seed pot's own bumps must fit */
    for (std::size_t i = 0; i < mu.size(); ++i) {
      if (eta[i] <= 0.0) continue;
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

    /* (b) the OPTIM box corner (CGRID high bound, SGRID low bound) must fit */
    const auto itc = trg.optimap.find(ckey);
    const auto its = trg.optimap.find(skey);
    for (std::size_t i = 0; i < mu.size(); ++i) {
      double hi_mu = mu[i];
      if (itc != trg.optimap.end() && i < itc->second.opt.size() &&
          itc->second.opt[i])
        hi_mu = itc->second.highbound[i];
      double eta_i = eta[i];
      if (its != trg.optimap.end() && i < its->second.opt.size() &&
          its->second.opt[i] && its->second.lowbound[i] > 0.0)
        eta_i = std::min(eta_i, its->second.lowbound[i]);
      if (eta_i <= 0.0) continue;
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
}

} // namespace hpo
} // namespace tadah

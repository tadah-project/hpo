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

  /* Bump (blip) channels: RCUT is authoritative — the hard truncation is
   * part of the fitted model — so only warn: (a) a centre beyond RCUT is
   * pathological; (b) OPTIM bounds that can push a blip's support across
   * RCUT truncate its shoulder there. */
  auto bump_halfwidth_for = [&](const char *sfx) -> double {
    try {
      if (std::string(sfx) == "2B") {
        if (!train_ctx.contains("TYPE2B")) return 0.0;
        const std::string n = train_ctx.get<std::string>("TYPE2B", 0);
        if (n == "D2_Blip") return 2.0;
        if (n == "D2_Blip5") return 3.0;
      } else {
        if (!train_ctx.contains("TYPEMB")) return 0.0;
        if (train_ctx.get<std::string>("TYPEMB", 0) != "DM_REAM") return 0.0;
        const int bcode = train_ctx.get<int>("TYPEMB", 1);
        if (bcode == 0) return 2.0;  // Blip
        if (bcode == 3) return 3.0;  // Blip5
      }
    } catch (const std::exception &) { /* not a refit shape */ }
    return 0.0;
  };
  for (const char *sfx : {"2B", "MB"}) {
    const double w = bump_halfwidth_for(sfx);
    if (w == 0.0) continue;
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
    const double rcut = train_ctx.get<double>(rkey);
    const double tol = 1e-9 * std::max(1.0, rcut);
    for (std::size_t i = 0; i < mu.size(); ++i) {
      if (mu[i] > rcut + tol) {
        warn(ckey + " centre " + core::to_string(mu[i]) + " lies beyond " +
             rkey + " = " + core::to_string(rcut) + " — a blip centred past "
             "the hard cutoff is pathological (its inner shoulder alone "
             "contributes).");
        break;
      }
    }
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
      if (hi_mu + w / eta_i > rcut + tol) {
        warn("the OPTIM bounds allow blip " + core::to_string(i + 1) +
             " of " + ckey + " to extend its support past " + rkey + " = " +
             core::to_string(rcut) + " — the overhang is hard-truncated "
             "there (the truncation radius is part of the blip model and is "
             "NOT auto-extended). Narrow the bounds if this is unintended.");
        break;
      }
    }
  }
}

} // namespace hpo
} // namespace tadah

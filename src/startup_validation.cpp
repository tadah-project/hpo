#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/strategy_rules.h>

#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/hpo/bump_support_gate.h>
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
        const double eta_hi =
            (s_opt && its->second.highbound[i] > 0.0) ? its->second.highbound[i]
                                                      : eta[i];
        const double eta_lo =
            (s_opt && its->second.lowbound[i] > 0.0) ? its->second.lowbound[i]
                                                     : eta[i];
        if (eta_hi <= 0.0 || eta_lo <= 0.0) continue;
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

  /* Search-space / initial-step rules (scale_validation.h): STEP policy
   * from the caps registry (already applied at parse time), every
   * resolved seed inside its OPTIM box, and the width audit. Runs last so
   * the physics gates above (knot RCUT, bump support) report first. */
  run_scale_validation(trg.spec, /*check_start_points=*/true);
}

} // namespace hpo
} // namespace tadah

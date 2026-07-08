#include <tadah/hpo/startup_validation.h>

#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

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
}

} // namespace hpo
} // namespace tadah

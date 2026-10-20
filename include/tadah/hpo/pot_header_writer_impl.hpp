#pragma once
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/loss_function.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>

namespace tadah::hpo {

/* =====================================================================
 *  write_result_table -- all metrics, all outvars, all losses, total
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_result_table(std::ostream& os,
                                         const EvalResult<ParamVec>& e,
                                         const HPOTargets& trg,
                                         std::size_t /*cw*/, int prec)
{
    heading(os, "RESULT");

    /* tabular header for global metrics */
    constexpr int NW = 20;  // name width
    constexpr int VW = 16;  // value width
    constexpr int LW = 16;  // loss width

    auto metric_row = [&](const std::string& name, bool used,
                          double val, double loss_val,
                          const std::vector<double>& per_ds,
                          const std::vector<double>& per_ds_loss) {
        if (!used) return;
        /* per-dataset rows */
        for (std::size_t d = 0; d < per_ds.size(); ++d) {
            std::string ds_name = name + "[" + std::to_string(d + 1) + "]";
            os << "#   " << std::left << std::setw(NW) << ds_name
               << std::right << std::setw(VW) << std::scientific
               << std::setprecision(prec) << per_ds[d];
            if (d < per_ds_loss.size())
               os << std::right << std::setw(LW) << std::scientific
                  << std::setprecision(prec) << per_ds_loss[d];
            else
               os << std::right << std::setw(LW) << "-";
            os << "\n";
        }
        /* average row */
        std::string avg_name = name + "[avg]";
        os << "#   " << std::left << std::setw(NW) << avg_name
           << std::right << std::setw(VW) << std::scientific
           << std::setprecision(prec) << val
           << std::right << std::setw(LW) << std::scientific
           << std::setprecision(prec) << loss_val << "\n";
    };

    /* header row for global metrics */
    os << "#   " << std::left << std::setw(NW) << "Metric"
       << std::right << std::setw(VW) << "Value"
       << std::right << std::setw(LW) << "Loss" << "\n";
    os << "#   " << std::string(NW + VW + LW, '-') << "\n";

    metric_row("EMAE   (meV/at)",  trg.use_emae,   e.emae,   e.emaeLoss,   e.v_emae,   e.v_emaeLoss);
    metric_row("ERMSE  (meV/at)",  trg.use_ermse,  e.ermse,  e.ermseLoss,  e.v_ermse,  e.v_ermseLoss);
    metric_row("ErRMSE (per mille)",  trg.use_errmse, e.errmse, e.errmseLoss, e.v_errmse, e.v_errmseLoss);
    metric_row("ERSQ",             trg.use_ersq,   e.ersq,   e.ersqLoss,   e.v_ersq,   e.v_ersqLoss);
    metric_row("FRMSE  (eV/A)",    trg.use_frmse,  e.frmse,  e.frmseLoss,  e.v_frmse,  e.v_frmseLoss);
    metric_row("SRMSE  (GPa)",     trg.use_srmse,  e.srmse,  e.srmseLoss,  e.v_srmse,  e.v_srmseLoss);

    /* LAMMPS outvar + loss table */
    if (!trg.LS.empty() && !e.v_outvar.empty()) {
        thin(os);
        os << "#   " << std::left << std::setw(NW) << "Sim variable"
           << std::right << std::setw(VW) << "Value"
           << std::right << std::setw(LW) << "Loss" << "\n";
        os << "#   " << std::string(NW + VW + LW, '-') << "\n";

        for (std::size_t i = 0; i < e.v_outvar.size(); ++i) {
            if (i >= trg.LS.size()) break;
            const auto& lsim = trg.LS[i];

            for (std::size_t j = 0; j < lsim.n_outvar.size(); ++j) {
                std::string name = lsim.n_outvar[j];
                if (static_cast<int>(name.size()) > NW - 1)
                    name = name.substr(0, NW - 2) + "*";

                os << "#   " << std::left << std::setw(NW) << name;

                /* value */
                if (j < e.v_outvar[i].size()) {
                    double v;
                    if (try_parse_double(e.v_outvar[i][j], v)) {
                        os << std::right << std::setw(VW)
                           << std::scientific << std::setprecision(prec)
                           << v;
                    } else {
                        os << std::right << std::setw(VW) << "NaN";
                    }
                } else {
                    os << std::right << std::setw(VW) << "N/A";
                }

                /* loss (only for varloss entries) */
                bool found_loss = false;
                for (std::size_t k = 0; k < lsim.i_varloss.size(); ++k) {
                    if (lsim.i_varloss[k] == j) {
                        found_loss = true;
                        if (i < e.v_loss.size() && j < e.v_loss[i].size()) {
                            os << std::right << std::setw(LW)
                               << std::scientific << std::setprecision(prec)
                               << e.v_loss[i][j];
                        } else {
                            os << std::right << std::setw(LW) << "N/A";
                        }
                        break;
                    }
                }
                if (!found_loss) {
                    os << std::right << std::setw(LW) << "-";
                }
                os << "\n";
            }
        }
    }

    thin(os);
    kv(os, "Total loss / N", fmt_double(e.loss, prec));
}

/* =====================================================================
 *  write_best_header -- full self-contained header
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_best_header(
    std::ostream&                       os,
    const EvalResult<ParamVec>&         result,
    const HPOTargets&                   trg,
    const core::Context&                master_ctx,
    const core::Context&                val_ctx,
    const mlip::StructureDB&            train_stdb,
    const mlip::StructureDB&            val_stdb,
    const std::vector<std::string>&     param_labels,
    const mlip::LoadTransformReport&    report,
    std::size_t                         cw,
    int                                 prec)
{
    /* Stable machine-readable preamble. durable_best::parse_key deliberately
     * depends only on these five fields; the human header may evolve. */
    const auto old_prec = os.precision();
    const auto old_flags = os.flags();
    /* FORMAT 2 (1.5.0): HPO_SOURCE_SLOT is written 1-based, like every other
     * slot a person reads. durable_best::parse_key subtracts the offset back
     * out, so the in-memory Key stays 0-based and identical to chain_id --
     * publish_if_better byte-compares the two and throws on a mismatch.
     * The version bump is what makes reading a FORMAT 1 file correct rather
     * than off-by-one (and for slot 0, a size_t underflow). */
    os << "# HPO_BEST_FORMAT 2\n"
       << "# HPO_BEST_LOSS " << std::scientific
       << std::setprecision(std::numeric_limits<double>::max_digits10)
       << result.loss << "\n"
       << "# HPO_SOURCE_RANK " << result.source_rank << "\n"
       << "# HPO_SOURCE_SLOT " << logidx::uis(result.chain_id) << "\n"
       << "# HPO_SOURCE_EVAL " << result.iter << "\n";
    os.flags(old_flags);
    os.precision(old_prec);

    write_banner(os, result.best_iter, result.iter, result.loss);
    write_loss_section(os, trg);
    write_result_table(os, result, trg, cw, prec);
    write_data_section(os, train_stdb, val_stdb, master_ctx, val_ctx);
    write_load_transforms_section(os, report);
    write_optim_section(os, trg, param_labels);
    rule(os);
}

template<class ParamVec>
void PotHeaderWriter::write_candidate_hot(
    std::ostream&                       os,
    const EvalResult<ParamVec>&         result)
{
    auto hot = result.hot_produced;
    std::sort(hot.begin(), hot.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b))
            return std::get<0>(a) < std::get<0>(b);
        if (std::get<1>(a) != std::get<1>(b))
            return std::get<1>(a) < std::get<1>(b);
        return std::get<2>(a) < std::get<2>(b);
    });
    const auto old_prec = os.precision();
    const auto old_flags = os.flags();
    os << std::scientific
       << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& entry : hot)
        os << "# HOT " << std::get<0>(entry) << " " << std::get<1>(entry)
           << " " << std::get<2>(entry) << "\n";
    os.flags(old_flags);
    os.precision(old_prec);
}

/* =====================================================================
 *  write_archive_header -- compact header for pot_N.tadah
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_archive_header(
    std::ostream&                os,
    const EvalResult<ParamVec>&  result,
    std::size_t                  /*cw*/, int prec)
{
    thin(os);
    os << "# Tadah HPO -- eval " << result.iter
       << "  |  loss=" << fmt_double(result.loss, prec)
       << "  |  " << fmt_timestamp() << "\n";
    thin(os);
}

/* =====================================================================
 *  write_best_archive_header -- compact header for best_pot_N.tadah
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_best_archive_header(
    std::ostream&                os,
    const EvalResult<ParamVec>&  result,
    const HPOTargets&            trg,
    std::size_t                  cw, int prec)
{
    thin(os);
    os << "# Tadah HPO -- best iter " << result.best_iter
       << "  (eval " << result.iter << ")"
       << "  |  loss=" << fmt_double(result.loss, prec)
       << "  |  " << fmt_timestamp() << "\n";
    write_result_table(os, result, trg, cw, prec);
    thin(os);
}

}   // namespace tadah::hpo

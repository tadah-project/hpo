#pragma once
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/loss_function.h>

#include <iomanip>
#include <sstream>

namespace tadah::hpo {

/* =====================================================================
 *  write_result_table — all metrics, all outvars, all losses, total
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
                          double val, double loss_val) {
        if (!used) return;
        os << "#   " << std::left << std::setw(NW) << name
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

    metric_row("EMAE   (meV/at)",  trg.use_emae,   e.emae,   e.emaeLoss);
    metric_row("ERMSE  (meV/at)",  trg.use_ermse,  e.ermse,  e.ermseLoss);
    metric_row("ErRMSE (meV/at)",  trg.use_errmse, e.errmse, e.errmseLoss);
    metric_row("ERSQ",             trg.use_ersq,   e.ersq,   e.ersqLoss);
    metric_row("FRMSE  (eV/A)",    trg.use_frmse,  e.frmse,  e.frmseLoss);
    metric_row("SRMSE  (GPa)",     trg.use_srmse,  e.srmse,  e.srmseLoss);

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
                    os << std::right << std::setw(VW)
                       << std::scientific << std::setprecision(prec)
                       << std::stod(e.v_outvar[i][j]);
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
 *  write_best_header — full self-contained header
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_best_header(
    std::ostream&                   os,
    const EvalResult<ParamVec>&     result,
    const HPOTargets&               trg,
    const core::Context&            master_ctx,
    const core::Context&            val_ctx,
    const mlip::StructureDB&        train_stdb,
    const mlip::StructureDB&        val_stdb,
    const std::string&              train_cfg_path,
    const std::string&              val_cfg_path,
    const std::vector<std::string>& param_labels,
    std::size_t                     cw,
    int                             prec)
{
    write_banner(os, result.best_iter, result.iter, result.loss);
    write_files_section(os, train_cfg_path, val_cfg_path);
    write_data_section(os, train_stdb, val_stdb, master_ctx, val_ctx);
    write_optim_section(os, trg, param_labels);
    write_loss_section(os, trg);
    write_result_table(os, result, trg, cw, prec);
    rule(os);
}

/* =====================================================================
 *  write_archive_header — compact header for pot_N.tadah
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_archive_header(
    std::ostream&                os,
    const EvalResult<ParamVec>&  result,
    std::size_t                  /*cw*/, int prec)
{
    thin(os);
    os << "# Tadah HPO — eval " << result.iter
       << "  |  loss=" << fmt_double(result.loss, prec)
       << "  |  " << fmt_timestamp() << "\n";
    thin(os);
}

/* =====================================================================
 *  write_best_archive_header — compact header for best_pot_N.tadah
 * ===================================================================== */
template<class ParamVec>
void PotHeaderWriter::write_best_archive_header(
    std::ostream&                os,
    const EvalResult<ParamVec>&  result,
    const HPOTargets&            trg,
    std::size_t                  cw, int prec)
{
    thin(os);
    os << "# Tadah HPO — best iter " << result.best_iter
       << "  (eval " << result.iter << ")"
       << "  |  loss=" << fmt_double(result.loss, prec)
       << "  |  " << fmt_timestamp() << "\n";
    write_result_table(os, result, trg, cw, prec);
    thin(os);
}

}   // namespace tadah::hpo

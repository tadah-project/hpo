#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/hpo_host.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <set>

namespace tadah::hpo {

/* =====================================================================
 *  Formatting primitives
 * ===================================================================== */

void PotHeaderWriter::rule(std::ostream& os)
{
    os << "# " << std::string(W, '=') << "\n";
}

void PotHeaderWriter::thin(std::ostream& os)
{
    os << "# " << std::string(W, '-') << "\n";
}

void PotHeaderWriter::blank(std::ostream& os) { os << "#\n"; }

void PotHeaderWriter::heading(std::ostream& os, const std::string& title)
{
    std::string s = "-- " + title + " ";
    while (static_cast<int>(s.size()) < W) s += '-';
    os << "# " << s << "\n";
}

void PotHeaderWriter::kv(std::ostream& os,
                         const std::string& key,
                         const std::string& val)
{
    os << "#   " << std::left << std::setw(KW) << key << ": " << val << "\n";
}

void PotHeaderWriter::line(std::ostream& os, const std::string& text)
{
    os << "#   " << text << "\n";
}

void PotHeaderWriter::wrap(std::ostream& os,
                           const std::string& prefix,
                           const std::vector<std::string>& items,
                           int max_width)
{
    if (items.empty()) { os << "#   " << prefix << " (none)\n"; return; }

    std::string cur = "  " + prefix;
    for (const auto& item : items) {
        if (static_cast<int>(cur.size() + item.size() + 1) > max_width && cur != ("  " + prefix)) {
            os << "#   " << cur << "\n";
            cur = std::string(prefix.size() + 2, ' ');
        }
        cur += " " + item;
    }
    if (!cur.empty()) os << "#   " << cur << "\n";
}

std::string PotHeaderWriter::fmt_double(double v, int prec)
{
    std::ostringstream s;
    s << std::scientific << std::setprecision(prec) << v;
    return s.str();
}

std::string PotHeaderWriter::fmt_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::ostringstream s;
    s << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S");
    return s.str();
}

/* =====================================================================
 *  Section writers
 * ===================================================================== */

void PotHeaderWriter::write_banner(std::ostream& os,
                                   std::size_t best_iter,
                                   std::size_t eval_iter,
                                   double loss)
{
    rule(os);
    os << "#  Tadah HPO — Best potential\n";
    os << "#  Best iter " << best_iter
       << "  (eval " << eval_iter << ")"
       << "  |  Loss: " << fmt_double(loss) << "\n";
    os << "#  " << fmt_timestamp() << "\n";
    rule(os);
}

/* -------------------------------------------------------------------- */
void PotHeaderWriter::write_data_section(std::ostream& os,
                                         const mlip::StructureDB& train,
                                         const mlip::StructureDB& val,
                                         const core::Context& train_ctx,
                                         const core::Context& val_ctx)
{
    namespace fs = std::filesystem;
    heading(os, "DATA");

    /* helper: per-dataset stats with DBFILE paths */
    auto dataset_stats = [&](const mlip::StructureDB& db,
                             const core::Context& ctx,
                             const std::string& label)
    {
        std::size_t n_ds = db.dbidx.size() > 0 ? db.dbidx.size() - 1 : 0;
        std::size_t total_structs = db.size();
        std::size_t total_atoms   = db.calc_natoms();

        std::ostringstream hdr;
        hdr << total_structs << " structures, "
            << total_atoms << " atoms, "
            << n_ds << " dataset(s)";
        kv(os, label, hdr.str());

        for (std::size_t d = 0; d < n_ds; ++d) {
            std::size_t s0 = db.dbidx[d];
            std::size_t s1 = db.dbidx[d + 1];
            std::size_t ns = s1 - s0;
            if (ns == 0) continue;

            /* atoms per structure: mean and stddev */
            double sum = 0, sum2 = 0;
            for (std::size_t s = s0; s < s1; ++s) {
                double na = static_cast<double>(db(s).natoms());
                sum  += na;
                sum2 += na * na;
            }
            double mean = sum / ns;
            double var  = (ns > 1) ? (sum2 - sum * sum / ns) / (ns - 1) : 0.0;
            double sd   = std::sqrt(std::max(var, 0.0));

            /* DBFILE path for this dataset */
            std::string dbpath;
            if (ctx.contains("DBFILE") && d < ctx.size("DBFILE")) {
                try { dbpath = fs::absolute(ctx.get<std::string>("DBFILE", d)).string(); }
                catch (...) { dbpath = ctx.get<std::string>("DBFILE", d); }
            }

            std::ostringstream ds;
            ds << "[" << (d + 1) << "] " << ns << " structs"
               << ", atoms/struct: " << std::fixed << std::setprecision(1)
               << mean << " +/- " << sd;
            if (!dbpath.empty()) ds << "  " << dbpath;
            line(os, ds.str());
        }
    };

    dataset_stats(train, train_ctx, "Training");
    dataset_stats(val,   val_ctx,   "Validation");

    /* elements */
    std::string elem_str;
    try {
        auto elems = mlip::StructureDB::find_unique_elements(train_ctx);
        for (const auto& e : elems) {
            if (!elem_str.empty()) elem_str += ", ";
            elem_str += e.symbol;
        }
    } catch (...) { elem_str = "(unknown)"; }
    kv(os, "Elements", elem_str);
}

/* -------------------------------------------------------------------- */
void PotHeaderWriter::write_optim_section(std::ostream& os,
                                          const HPOTargets& trg,
                                          const std::vector<std::string>& labels)
{
    heading(os, "OPTIMISER");
    const auto& sp = trg.spec;
    kv(os, "Algorithm", sp.lib + " / " + sp.algo);
    kv(os, "Dimensions", std::to_string(sp.low.size()));

    if (sp.local) {
        std::string local_str = sp.local->algo;
        if (sp.local->lib != "NOT SET")
            local_str = sp.local->lib + " / " + local_str;
        kv(os, "Local optimizer", local_str);
    }

    /* stop criteria — compact, one line */
    {
        std::vector<std::string> crit;
        if (sp.stop.maxeval)  crit.push_back("maxeval=" + std::to_string(*sp.stop.maxeval));
        if (sp.stop.maxtime)  crit.push_back("maxtime=" + fmt_double(*sp.stop.maxtime, 1));
        if (sp.stop.ftol_rel) crit.push_back("ftol_rel=" + fmt_double(*sp.stop.ftol_rel));
        if (sp.stop.ftol_abs) crit.push_back("ftol_abs=" + fmt_double(*sp.stop.ftol_abs));
        if (sp.stop.xtol_rel) crit.push_back("xtol_rel=" + fmt_double(*sp.stop.xtol_rel));
        if (sp.stop.xtol_abs) crit.push_back("xtol_abs=" + fmt_double(*sp.stop.xtol_abs));
        if (sp.stop.stopval)  crit.push_back("stopval=" + fmt_double(*sp.stop.stopval));
        if (crit.empty()) crit.push_back("(none)");
        wrap(os, "Stop:", crit);
    }
    if (sp.seed) kv(os, "Seed", std::to_string(*sp.seed));
    if (sp.pop)  kv(os, "Population", std::to_string(*sp.pop));

    /* bounds table — compact two-column layout */
    {
        constexpr int LBL_W = 14;   // label column
        constexpr int BND_W = 24;   // "[lo, hi]" column
        constexpr int COL_W = LBL_W + BND_W;  // one full column
        constexpr int GAP   = 2;    // gap between columns

        line(os, "Search space:");

        auto fmt_col = [&](std::size_t idx) -> std::string {
            const std::string& lbl = (idx < labels.size())
                ? labels[idx] : "p[" + std::to_string(idx) + "]";
            std::string lbl_trunc = lbl;
            if (static_cast<int>(lbl_trunc.size()) > LBL_W - 1)
                lbl_trunc = lbl_trunc.substr(0, LBL_W - 2) + "*";

            std::ostringstream bnd;
            bnd << "[" << fmt_double(sp.low[idx], 3)
                << ", " << fmt_double(sp.high[idx], 3) << "]";
            if (idx < sp.log_scaled.size() && sp.log_scaled[idx])
                bnd << "(log)";

            std::ostringstream col;
            col << std::left << std::setw(LBL_W) << lbl_trunc
                << std::left << std::setw(BND_W) << bnd.str();
            return col.str();
        };

        for (std::size_t i = 0; i < sp.low.size(); i += 2) {
            std::string row = fmt_col(i);
            if (i + 1 < sp.low.size()) {
                // pad first column to fixed width, add gap, then second
                while (static_cast<int>(row.size()) < COL_W)
                    row += ' ';
                row += std::string(GAP, ' ') + fmt_col(i + 1);
            }
            line(os, row);
        }
    }
}

/* -------------------------------------------------------------------- */
void PotHeaderWriter::write_loss_section(std::ostream& os,
                                         const HPOTargets& trg)
{
    heading(os, "LOSS FUNCTION");

    kv(os, "Global loss type",
       tadah::LossFunction::to_string(trg.globalLoss.type()));
    kv(os, "Fail score", fmt_double(trg.FAIL_SCORE));

    /* global metrics — one line each, only if active */
    auto metric = [&](const std::string& name, bool used,
                      double target, double weight) {
        if (!used) return;
        std::ostringstream s;
        s << "target=" << fmt_double(target, 3)
          << "  w=" << std::fixed << std::setprecision(2) << weight
          << "  (per-dataset + avg)";
        kv(os, name, s.str());
    };
    metric("EMAE   (meV/at)",  trg.use_emae,   trg.emae,   trg.w_emae);
    metric("ERMSE  (meV/at)",  trg.use_ermse,  trg.ermse,  trg.w_ermse);
    metric("ErRMSE (meV/at)",  trg.use_errmse, trg.errmse, trg.w_errmse);
    metric("ERSQ",             trg.use_ersq,   trg.ersq,   trg.w_ersq);
    metric("FRMSE  (eV/A)",    trg.use_frmse,  trg.frmse,  trg.w_frmse);
    metric("SRMSE  (GPa)",     trg.use_srmse,  trg.srmse,  trg.w_srmse);

    /* LAMMPS simulation losses */
    if (!trg.LS.empty()) {
        thin(os);
        kv(os, "LAMMPS simulations", std::to_string(trg.LS.size()));
        for (std::size_t si = 0; si < trg.LS.size(); ++si) {
            const auto& lsim = trg.LS[si];
            std::ostringstream sh;
            sh << "Sim[" << (si + 1) << "]: " << lsim.script
               << "  ncpu=" << lsim.ncpu
               << "  maxtime=" << lsim.maxtime
               << "  failscore=" << fmt_double(lsim.failscore, 2);
            line(os, sh.str());

            for (std::size_t j = 0; j < lsim.n_varloss.size(); ++j) {
                std::ostringstream vl;
                vl << "  --varloss " << lsim.n_varloss[j]
                   << "  target=" << fmt_double(lsim.t_varloss[j], 3)
                   << "  w=" << std::fixed << std::setprecision(2)
                   << lsim.w_varloss[j]
                   << "  loss=" << tadah::LossFunction::to_string(
                                        lsim.lossFn[j].type());
                const auto& lp = lsim.lossFn[j].params();
                for (double p : lp) vl << " " << fmt_double(p, 3);
                line(os, vl.str());
            }
            /* outvar-only variables (not used for loss) */
            for (std::size_t j = 0; j < lsim.n_outvar.size(); ++j) {
                bool is_varloss = std::find(lsim.n_varloss.begin(),
                                            lsim.n_varloss.end(),
                                            lsim.n_outvar[j])
                                  != lsim.n_varloss.end();
                if (!is_varloss) {
                    line(os, "  --outvar " + lsim.n_outvar[j]);
                }
            }
        }
    }
}

}   // namespace tadah::hpo


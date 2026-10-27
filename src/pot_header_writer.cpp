#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/search_space.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/training_record.h>

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
 *  The potential below the header
 * ===================================================================== */

void PotHeaderWriter::write_potential_body(std::ostream& os,
                                           const core::Context& pot,
                                           const mlip::TrainingRecord& record,
                                           int sig_digits)
{
    mlip::write_potential_with_record(os, pot, record, sig_digits);
}

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
    std::tm local {};
#if defined(_WIN32)
    localtime_s(&local, &tt);
#else
    localtime_r(&tt, &local);
#endif
    std::ostringstream s;
    s << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
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
    os << "#  Tadah HPO -- Best potential\n";
    /* The evaluation (its number in loss.tadah and params.tadah) and the
     * improvement (its number among the run's bests, in best_*.tadah). */
    os << "#  Evaluation " << eval_iter << ", improvement " << best_iter
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
                             const std::string& label,
                             bool names_paths)
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

            /* The dataset: a validation dataset by its DBFILE path, as
             * written in the validation file (relative paths preserved --
             * absolute paths would leak the user's filesystem layout into
             * shipped best_pot.tadah files); a training dataset by its
             * entry of the DBFILE line of the file's training record, which
             * names the training datasets (one block names them, not two). */
            std::string dbpath;
            if (names_paths && ctx.contains("DBFILE") && d < ctx.size("DBFILE")) {
                dbpath = ctx.get<std::string>("DBFILE", d);
            }

            std::ostringstream ds;
            ds << "[" << (d + 1) << "] " << ns << " structs"
               << ", atoms/struct: " << std::fixed << std::setprecision(1)
               << mean << " +/- " << sd;
            if (!dbpath.empty()) ds << "  " << dbpath;
            if (!names_paths)
                ds << "  (DBFILE " << (d + 1) << " of the training record)";
            line(os, ds.str());
        }
    };

    dataset_stats(train, train_ctx, "Training",   /*names_paths=*/false);
    dataset_stats(val,   val_ctx,   "Validation", /*names_paths=*/true);

    /* elements: from the training data when present, otherwise from the
     * explicit ATOM key (mandatory in no-training-DBFILE runs). The
     * contains() guard matters beyond correctness: find_unique_elements
     * calls train_ctx("DBFILE"), and Context::operator() LOGS an ERROR
     * before throwing — so probing it per new-best spams stdout. */
    std::string elem_str;
    try {
        if (train_ctx.contains("DBFILE") && train_ctx.size("DBFILE") > 0) {
            auto elems = mlip::StructureDB::find_unique_elements(train_ctx);
            for (const auto& e : elems) {
                if (!elem_str.empty()) elem_str += ", ";
                elem_str += e.symbol;
            }
        } else if (train_ctx.contains("ATOM")) {
            for (const auto& s : train_ctx("ATOM")) {
                if (!elem_str.empty()) elem_str += ", ";
                elem_str += s;
            }
        }
    } catch (...) { elem_str.clear(); }
    if (elem_str.empty()) elem_str = "(unknown)";
    kv(os, "Elements", elem_str);
}

/* -------------------------------------------------------------------- */
void PotHeaderWriter::write_load_transforms_section(
    std::ostream& os,
    const mlip::LoadTransformReport& r)
{
    if (!r.applied) return;
    heading(os, "LOAD TRANSFORMS");

    if (r.lscale != 1.0)
        kv(os, "Length rescale", fmt_double(r.lscale, 6));

    if (!r.eshift.empty()) {
        const char* src = "user";
        switch (r.eshift_source) {
            case mlip::LoadTransformReport::EshiftSource::User:   src = "user";   break;
            case mlip::LoadTransformReport::EshiftSource::Atom:   src = "atom";   break;
            case mlip::LoadTransformReport::EshiftSource::DBAtom: src = "dbatom"; break;
            case mlip::LoadTransformReport::EshiftSource::None:   src = "none";   break;
        }
        std::ostringstream s;
        s << "(source=" << src << ")";
        for (const auto& kvp : r.eshift) {
            const auto& el = core::PeriodicTable::find_by_Z(kvp.first);
            s << "  " << el.symbol << "=" << fmt_double(kvp.second, 4);
        }
        kv(os, "Energy shift", s.str());
    }

    if (r.n_dropped_energy || r.n_dropped_force) {
        std::ostringstream s;
        s << "dropped " << r.n_dropped_energy << " by E, "
          << r.n_dropped_force << " by F";
        kv(os, "Outlier filter", s.str());
    }

    if (r.zero_com_force)
        kv(os, "Zero COM force", "applied");

    if (!r.eq_label.empty()) {
        std::ostringstream s;
        s << "label='" << r.eq_label
          << "'  E/N=" << fmt_double(r.eq_energy_per_atom, 4)
          << " eV  V/N=" << fmt_double(r.eq_volume_per_atom, 4) << " A^3";
        kv(os, "Reference state (eq vol)", s.str());
    }
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

    if (sp.inner) {
        std::string inner_str = sp.inner->algo;
        if (sp.inner->lib != "NOT SET")
            inner_str = sp.inner->lib + " / " + inner_str;
        kv(os, "Inner optimizer", inner_str);
    }

    /* stop criteria -- compact, one line */
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
    /* Token-valued and multi-token PARAMs too: the numeric-only walk left
     * `PARAM INIT_MODE UNIFORM` out of the provenance header entirely. */
    if (!sp.param.empty() || !sp.param_raw.empty()) {
        std::vector<std::pair<std::string, std::string>> params;
        params.reserve(sp.param.size() + sp.param_raw.size());
        const auto rendered = [&sp](const std::string& name,
                                    const std::string& fallback) {
            const auto it = sp.param_tokens.find(name);
            if (it == sp.param_tokens.end()) return fallback;
            std::string joined;
            for (const auto& t : it->second) {
                if (!joined.empty()) joined += ' ';
                joined += t;
            }
            return joined;
        };
        /* A setting as written: the shortest text that reads back as the
         * value ("GRID_POINTS=11", "STEP_FRAC=0.15"). */
        for (const auto& kvp : sp.param)
            params.emplace_back(kvp.first,
                                rendered(kvp.first,
                                         core::format_roundtrip(kvp.second)));
        for (const auto& kvp : sp.param_raw)
            params.emplace_back(kvp.first, rendered(kvp.first, kvp.second));
        std::sort(params.begin(), params.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<std::string> param_lines;
        param_lines.reserve(params.size());
        for (const auto& kvp : params)
            param_lines.push_back(kvp.first + "=" + kvp.second);
        wrap(os, "Params:", param_lines);
    }

    /* bounds table -- compact two-column layout */
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

            /* The real bounds, the box of the OPTIM line, as the journal's
             * "real bounds" column: sp.low/high are the optimiser's
             * internal box (the unit box under OPTIM_SCALE AUTO). */
            const bool real = sp.space && idx < sp.space->dim();
            const double lo = real ? sp.space->at(idx).lo : sp.low[idx];
            const double hi = real ? sp.space->at(idx).hi : sp.high[idx];
            std::ostringstream bnd;
            bnd << "[" << fmt_double(lo, 3) << ", " << fmt_double(hi, 3) << "]";
            if (sp.space && idx < sp.space->dim() && sp.space->is_nonlinear(idx))
                bnd << "(" << sp.space->describe_dim(idx) << ")";
            else if (idx < sp.log_scaled.size() && sp.log_scaled[idx])
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

    /* global metrics -- one line each, only if active */
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
    metric("ErRMSE (per mille)",  trg.use_errmse, trg.errmse, trg.w_errmse);
    metric("ERSQ",             trg.use_ersq,   trg.ersq,   trg.w_ersq);

    auto dataset_metric = [&](const std::string& name, const ValidationMetricTargets& targets) {
        for (const auto& description : targets.descriptions()) kv(os, name, description);
    };
    dataset_metric("ERMSE (meV/at)", trg.ermse_targets);
    dataset_metric("FRMSE (eV/A)", trg.frmse_targets);
    dataset_metric("SRMSE (GPa)", trg.srmse_targets);

    /* LAMMPS simulation losses */
    if (!trg.LS.empty()) {
        thin(os);
        kv(os, "LAMMPS simulations", std::to_string(trg.LS.size()));
        for (std::size_t si = 0; si < trg.LS.size(); ++si) {
            const auto& lsim = trg.LS[si];
            std::ostringstream sh;
            sh << "Sim[" << (si + 1) << "]: " << lsim.script
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

    /* PC_OBSERVABLE terms, named as the journal names them (<kind>[k],
     * k the 1-based ordinal of the line); RESULT has the row of each. */
    for (std::size_t k = 0; k < trg.observables.size(); ++k) {
        const auto& obs = trg.observables[k];
        if (!obs) continue;
        std::ostringstream s;
        s << "target=" << std::setprecision(12) << obs->target();
        if (!obs->unit().empty()) s << " " << obs->unit();
        s << " weight=" << obs->weight()
          << " loss=" << tadah::LossFunction::to_string(obs->lossfn().type());
        for (double p : obs->lossfn().params()) s << " " << p;
        kv(os, observable_journal_label(*obs, k + 1), s.str());
    }
}

}   // namespace tadah::hpo

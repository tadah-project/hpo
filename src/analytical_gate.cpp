#include <tadah/hpo/analytical_gate.h>

#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/sims_grad.h>
#include <tadah/core/element.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>

namespace tadah::hpo {

std::string analytical_gradient_blocker(const core::Context& ctx,
                                        const HPOTargets& trg,
                                        bool has_train_db)
{
    if (!has_train_db)
        return "there is no training DBFILE (it differentiates the "
               "least-squares fit of the weights)";
    if (ctx.contains("FIXINDEX"))
        return "FIXINDEX pins model weights (it assumes every weight is "
               "fitted)";
    if (ctx.get<bool>("STRESS"))
        return "STRESS is true (stress rows are not differentiated)";
    if (ctx.get<bool>("NORM"))
        return "NORM is true (it works on raw features)";
    /* It differentiates the least-squares fit of the training rows
     * (M_BLR, M_KRR implementation 1); implementation 2 solves (K +
     * LAMBDA sigma^2 I) w = t over its basis vectors instead, and reads no
     * OALGO (its fit refuses one but the default). */
    if (mlip::is_krr_implementation_2(ctx)) {
        const std::string kernel = ctx.get<std::string>("MODEL", 1);
        return "the model is MODEL M_KRR " + kernel + " 2, implementation "
               "2 (it differentiates the least-squares fit of the training "
               "rows, that of M_BLR and of MODEL M_KRR " + kernel + " 1, "
               "implementation 1, with OALGO 4; implementation 2 solves (K "
               "+ LAMBDA sigma^2 I) w = t over its basis vectors)";
    }
    if (const int oalgo = ctx.get<int>("OALGO", 0); oalgo != 4)
        return "OALGO is " + std::to_string(oalgo) +
               " (it needs OALGO 4, the Cholesky normal equations)";
    {
        const double lambda = ctx.get<double>("LAMBDA", 0);
        if (std::abs(lambda + 1.0) < std::numeric_limits<double>::min())
            return "LAMBDA is -1, the evidence approximation (it needs a "
                   "fixed LAMBDA >= 0)";
        if (lambda < 0.0) {
            std::ostringstream os;
            os << "LAMBDA is " << lambda << " (it needs a fixed LAMBDA >= 0)";
            return os.str();
        }
    }
    {
        const std::string basis = ctx.get<std::string>("MODEL", 1);
        if (basis != "BF_Linear" && basis != "Kern_Linear")
            return "the basis is " + basis +
                   " (it needs MODEL ... BF_Linear or Kern_Linear)";
    }
    if (trg.use_srmse)
        return "PC_SRMSE_GPA is set (stress metrics are not differentiated)";
    return {};
}

std::string analytical_loss_term_blocker(const HPOTargets& trg)
{
    for (std::size_t k = 0; k < trg.observables.size(); ++k) {
        const auto& obs = trg.observables[k];
        if (!obs) continue;
        const std::string term =
            "the PC_OBSERVABLE term " + observable_journal_label(*obs, k + 1);
        if (!loss_has_derivative_wrt_observed(obs->lossfn()))
            return term + " uses the loss function " +
                   LossFunction::to_string(obs->lossfn().type()) +
                   ", which has no derivative with respect to the value";
        if (const std::string gap = obs->analytical_gradient_gap();
            !gap.empty())
            return term + ": " + gap;
    }
    return {};
}

namespace {

/* Pairs of atoms the data need not hold, on which the walker shows what
 * the descriptors differentiate at all: for each pair of species of ATOM
 * of `ctx`, two atoms at each distance from `rcut` - 0.25 A down to 1 A
 * (rcut / 2 when that is less), 0.25 A apart, alone in a cubic cell of
 * side 3 rcut (no image within the cutoff), with its neighbour list at
 * `rcut`. Every cutoff up to `rcut` has such a pair within 1 A of it, where
 * Cut_PolyS and Cut_CosS depend on it. */
mlip::StructureDB probe_pairs(const core::Context& ctx, double rcut)
{
    mlip::StructureDB db;
    if (!(rcut > 0.0)) return db;
    std::vector<core::Element> species;
    for (std::size_t i = 0;; ++i) {
        const auto symbol = ctx.stored_value("ATOM", i);
        if (!symbol) break;
        species.emplace_back(*symbol);
    }
    const double side = 3.0 * rcut;
    const double closest = std::min(1.0, 0.5 * rcut);
    for (std::size_t a = 0; a < species.size(); ++a) {
        for (std::size_t b = a; b < species.size(); ++b) {
            for (int k = 1;; ++k) {
                const double r = rcut - 0.25 * k;
                if (r < closest) break;
                mlip::Structure st;
                st.cell.load(side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side);
                st.add_atom(mlip::Atom(species[a], 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0));
                st.add_atom(mlip::Atom(species[b], r, 0.0, 0.0,
                                       0.0, 0.0, 0.0));
                mlip::NNFinder().calc(st, rcut);
                db.add(st);
            }
        }
    }
    return db;
}

/* The labels of the descriptors of `dcs` (its two-body and its many-body
 * descriptor) that the walker claims `p` for in the energy rows of the
 * structures of `db` and in none of their force rows. */
std::vector<std::string>
energy_rows_only_descriptors(const mlip::StructureDB& db,
                             const models::DC_Selector& dcs,
                             const ParamRef& p, std::size_t n_cols)
{
    ThetaClaims d2, dm;
    for (std::size_t s = 0; s < db.size(); ++s) {
        mlip::StructureDB one;
        one.add(db(s));
        std::vector<double> energy_rows(n_cols, 0.0);
        std::vector<double> force_rows(3 * db(s).natoms() * n_cols, 0.0);
        const auto c2 = mlip::assemble_dphi_train_both_rows_d2(
            one, *dcs.d2b, p.theta_name, p.arg_pos, energy_rows.data(),
            force_rows.data(), n_cols, {0});
        d2.energy = d2.energy || c2.first;
        d2.force = d2.force || c2.second;
        if (!dcs.dmb) continue;
        energy_rows.assign(energy_rows.size(), 0.0);
        force_rows.assign(force_rows.size(), 0.0);
        const auto cm = mlip::assemble_dphi_train_both_rows_dm(
            one, *dcs.dmb, p.theta_name, p.arg_pos, energy_rows.data(),
            force_rows.data(), n_cols, {0});
        dm.energy = dm.energy || cm.first;
        dm.force = dm.force || cm.second;
    }
    std::vector<std::string> out;
    if (d2.energy && !d2.force) out.push_back(dcs.d2b->label());
    if (dm.energy && !dm.force) out.push_back(dcs.dmb->label());
    return out;
}

/* The cutoff functions of RCTYPE2B (for RCUT2B) or RCTYPEMB (for RCUTMB)
 * of `ctx` that depend on the cutoff only for a pair within 1 A of it,
 * once each in the order of their values: Cut_PolyS, Cut_CosS. */
std::vector<std::string> near_cutoff_functions(const core::Context& ctx,
                                               const std::string& key)
{
    std::vector<std::string> out;
    const std::string rctype = key == "RCUT2B"   ? "RCTYPE2B"
                               : key == "RCUTMB" ? "RCTYPEMB"
                                                 : "";
    if (rctype.empty()) return out;
    for (std::size_t i = 0;; ++i) {
        const auto f = ctx.stored_value(rctype, i);
        if (!f) break;
        if ((*f == "Cut_PolyS" || *f == "Cut_CosS") &&
            std::find(out.begin(), out.end(), *f) == out.end())
            out.push_back(*f);
    }
    return out;
}

}  // namespace

std::vector<ParameterWithoutDerivative>
parameters_without_analytical_derivative(const core::Context& ctx,
                                         const HPOTargets& trg,
                                         const mlip::StructureDB& train,
                                         const mlip::StructureDB& val)
{
    /* The descriptors an evaluation builds (HPO_Host: a Trainer of its
     * candidate context), at the configuration's values. */
    core::Context c = ctx;
    mlip::memory::MLIPWorkspaceManager ws;
    mlip::Trainer probe(c, ws);
    const std::size_t n_cols = probe.DCS.descriptor_size();
    const bool train_force = probe.context.get<bool>("FORCE");
    const bool has_val = val.size() > 0;
    const bool val_force = trg.use_frmse;
    /* The force rows of the run: the fit's, or those PC_FRMSE measures. */
    const bool force_rows = train_force || (has_val && val_force);

    /* Built for the first parameter without a derivative. */
    std::optional<mlip::StructureDB> pairs;

    std::vector<ParameterWithoutDerivative> out;
    for (const ParamRef& p : make_param_ref_table(trg)) {
        const ThetaClaims t =
            theta_claims(train, probe.DCS, p, train_force, n_cols);
        const ThetaClaims v =
            has_val ? theta_claims(val, probe.DCS, p, val_force, n_cols)
                    : ThetaClaims{};
        if (claims_differentiate(t, train_force, v, has_val, val_force))
            continue;
        ParameterWithoutDerivative g;
        g.key = p.theta_name;
        g.index = static_cast<std::size_t>(p.arg_pos);
        const auto it = trg.optimap.find(p.theta_name);
        if (it != trg.optimap.end() && p.token < it->second.line.size())
            g.line = it->second.line[p.token];
        if (!pairs) {
            const double rcut = std::max(
                probe.DCS.d2b->get_rcut(),
                probe.DCS.dmb ? probe.DCS.dmb->get_rcut() : 0.0);
            pairs = probe_pairs(ctx, rcut);
        }
        const ThetaClaims o =
            theta_claims(*pairs, probe.DCS, p, force_rows, n_cols);
        if (!o.energy) {
            g.why = ParameterWithoutDerivative::Why::NoDerivative;
        } else if (force_rows && !o.force) {
            g.why = ParameterWithoutDerivative::Why::EnergyRowsOnly;
            g.descriptors =
                energy_rows_only_descriptors(*pairs, probe.DCS, p, n_cols);
            g.force_rows_of_fit = train_force;
        } else {
            g.why = ParameterWithoutDerivative::Why::NoPairInData;
            g.training = !(t.energy && (!train_force || t.force));
            g.validation =
                has_val && !(v.energy && (!val_force || v.force));
            g.near_cutoff = near_cutoff_functions(ctx, p.theta_name);
        }
        out.push_back(std::move(g));
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const ParameterWithoutDerivative& a,
                        const ParameterWithoutDerivative& b) {
                         return a.line < b.line;
                     });
    return out;
}

}  // namespace tadah::hpo

#include <tadah/hpo/observables/cohesive_energy.h>

#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/native_relaxer.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/row_weights.h>   // kGPaInEvPerA3
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure_transformations.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {

CohesiveEnergyObservable::
CohesiveEnergyObservable(Config cfg)
    : cfg_(std::move(cfg))
{
    if (cfg_.species.empty())
        throw std::invalid_argument("CohesiveEnergyObservable: empty species");
    if (cfg_.nx < 1 || cfg_.ny < 1 || cfg_.nz < 1)
        throw std::invalid_argument("CohesiveEnergyObservable: supercell dims must be >= 1");
    if (!(cfg_.a0_max > cfg_.a0_min))
        throw std::invalid_argument("CohesiveEnergyObservable: a0_max must be > a0_min");
    if (cfg_.weight < 0.0)
        throw std::invalid_argument("CohesiveEnergyObservable: weight must be >= 0");
    /* The start of a chain's relaxation lies in the window its relaxed a0
     * must stay in (the parser refuses one outside at its line); it was
     * moved onto the nearer bound without a word. */
    if (cfg_.a0_initial &&
        !(*cfg_.a0_initial >= cfg_.a0_min && *cfg_.a0_initial <= cfg_.a0_max))
        throw std::invalid_argument(
            "CohesiveEnergyObservable: a0_initial must lie in [a0_min, a0_max]");
    if (!(cfg_.eq_fmax_threshold > 0.0))
        throw std::invalid_argument(
            "CohesiveEnergyObservable: eq_fmax_threshold must be > 0");
    /* No relaxation reaches a pressure of 0. */
    if (!(cfg_.relax_ptol > 0.0))
        throw std::invalid_argument(
            "CohesiveEnergyObservable: relax_ptol must be > 0");
    /* A negative slope would move the value of an unconverged relaxation
     * towards the target (the parser refuses it at its line). */
    if (cfg_.unconverged_strategy == Config::UnconvergedStrategy::ForceScaled &&
        !(cfg_.unconverged_value >= 0.0))
        throw std::invalid_argument(
            "CohesiveEnergyObservable: the force-scaled slope must be >= 0");
}

std::string CohesiveEnergyObservable::analytical_gradient_gap() const
{
    /* A relaxation that does not converge reports the --unconverged value:
     * the constant penalty (value_is_constant, a zero gradient), or the
     * cohesive energy of the cell it ends at moved away from the target by
     * slope * fmax (fmax the larger of the largest atomic force and the
     * strain force of the cell at the end), which depends on the weights
     * and on theta through the partial relaxation and has no closed-form
     * derivative (evaluate() returns it without dO_dW). */
    if (cfg_.unconverged_strategy == Config::UnconvergedStrategy::ForceScaled)
        return "--unconverged force-scaled reports, for a relaxation that "
               "does not converge, the cohesive energy of the cell it ends "
               "at moved away from the target by slope x fmax, a value "
               "without an analytical derivative";
    return {};
}

ObservableEvalResult CohesiveEnergyObservable::
evaluate(tadah::mlip::M_Tadah_Base& model,
         tadah::mlip::DC_Base& dc,
         const tadah::core::Context& ctx)
{
    return evaluate(model, dc, ctx, std::nullopt);
}

ObservableEvalResult CohesiveEnergyObservable::
evaluate(tadah::mlip::M_Tadah_Base& model,
         tadah::mlip::DC_Base& dc,
         const tadah::core::Context& ctx,
         std::optional<double> start)
{
    using tadah::mlip::Structure;
    using tadah::mlip::StDescriptors;
    using tadah::mlip::NNFinder;
    using tadah::mlip::NativeRelaxer;
    using tadah::mlip::StructureTransformation;
    namespace cells = tadah::mlip::cells;

    /* PRECONDITION: caller (HPO_Host obs_loss + ObservableGradProvider)
     * ensures ctx (and the shared dc.context it references) has
     * FORCE=true before invoking evaluate(). NativeRelaxer needs
     * model.fpredict which depends on dc.calc populating fd_ij arrays,
     * which is gated by FORCE in dc.context. The host owns the
     * trainer's master context and is the right scope to flip the
     * flag -- observable code stays oblivious to the trainer's training
     * FORCE setting. */

    /* The a0 this relaxation starts from: the one the previous evaluation
     * of the chain handed on, or the declared start. It lies in [a0_min,
     * a0_max]: the start is checked when the line is read, and an
     * evaluation hands on no a0 outside the window (it rejects the
     * relaxation and hands on its own start instead). A drifted tiny a0
     * (e.g., 0.5 Ang) would give a supercell whose NN search builds ~10^4
     * neighbours per atom and stalls fpredict for tens of seconds inside
     * the analytical force loop. */
    const double a0_start = start.value_or(start_a0());
    if (!(a0_start >= cfg_.a0_min && a0_start <= cfg_.a0_max)) {
        std::ostringstream why;
        why << "CohesiveEnergyObservable: the start a0 = " << a0_start
            << " lies outside the --hotvar window [" << cfg_.a0_min << ", "
            << cfg_.a0_max << "]";
        throw std::logic_error(why.str());
    }

    /* 1. Build bulk supercell at the start a0. */
    Structure bulk = cells::bcc(cfg_.species, a0_start);
    StructureTransformation::replicate(bulk, cfg_.nx, cfg_.ny, cfg_.nz);

    /* 2. Iso-relax bulk under P = 0. */
    NativeRelaxer::Options relax_opts;
    relax_opts.box_mode = NativeRelaxer::Options::Iso;
    relax_opts.pressure = 0.0;
    /* --fmax-converged: the force tolerance of the atoms the relaxation
     * stops at is the one its result is judged by. --ptol, in GPa: the
     * pressure tolerance of the cell. The cell is relaxed to zero pressure
     * whatever the atoms' tolerance: with one tolerance for both (0.1, in
     * eV/A for the atoms and eV/A^3, 16 GPa, for the cell) the relaxation
     * stopped far from the minimum, at a cell that depended on where it
     * started. */
    relax_opts.ftol     = cfg_.eq_fmax_threshold;
    relax_opts.ptol     = cfg_.relax_ptol * tadah::mlip::kGPaInEvPerA3;
    relax_opts.etol     = cfg_.relax_etol;
    relax_opts.max_wall_time_s = cfg_.relax_max_wall_time_s;
    relax_opts.maxiter  = cfg_.relax_maxiter;
    relax_opts.maxeval  = cfg_.relax_maxeval;
    /* Don't impose soft scale bounds on the relaxer (default +/-inf).
     * LAMMPS lets `box/relax` wander to whatever cell the model wants;
     * the hot-var rejection happens post-relax (`hot_a0 outside [...]`).
     * Per-LS-trial bounds interact badly with the Armijo line search:
     * most retries land at the boundary, no step survives, and the
     * relax exits unconverged for almost every wild theta. With
     * max_strain_step=0.001 (matches LAMMPS vmax) + NNFinder's
     * defensive cap on near-degenerate cells, the relax can't
     * catastrophically drift inside maxeval=5000.
     *
     * Instead, use the early-bail predicate to abort once a successful
     * line-search step has placed a0 outside [a0_min, a0_max]. The
     * post-hoc hot-var gate below the relax rejects on the same condition,
     * so any relax that crosses the boundary is guaranteed to be
     * rejected -- burning the rest of the maxeval budget is wasted
     * work. Because the predicate fires AFTER an accepted LS step
     * (not during the LS halving), it doesn't interfere with the LS's
     * ability to find a step that lands inside the range. */
    {
        const double a0_lo = cfg_.a0_min;
        const double a0_hi = cfg_.a0_max;
        const double a0_cur = a0_start;
        relax_opts.early_reject =
            [a0_lo, a0_hi, a0_cur](const tadah::mlip::EarlyState& s) {
                const double a = a0_cur * s.scale_total;
                return a < a0_lo || a > a0_hi;
            };
    }
    NativeRelaxer relaxer;
    NativeRelaxer::Result rr = relaxer.relax(bulk, model, dc, ctx, relax_opts);

    /* Capture relaxed a0 for warmstart: the Iso relax scales the cell built
     * from a0, so a0 times its scale factor, for any lattice and supercell
     * (cell(0,0)/nx was one ulp off for nx = 3 and wrong for a cell whose
     * first row is not a0 x nx along x). The early-reject predicate above
     * uses the same quantity. */
    const double new_a0 = a0_start * rr.scale_factor;

    /* Reject if the relax drove a0 outside the user-declared hot-var
     * range. Mirrors PC_LAMMPS's `rejecting (hot_a0=... outside [..])`
     * policy -- the optimiser sees NaN -> FAIL_SCORE component and
     * walks away from this theta. Hand on the start, inside the window,
     * so the chain's next evaluation does not inherit a wild warm start
     * that would re-stall NN search at the cell-build step. */
    if (!(new_a0 >= cfg_.a0_min && new_a0 <= cfg_.a0_max)) {
        ObservableEvalResult res;
        res.value = std::numeric_limits<double>::quiet_NaN();
        res.needs_adjoint = false;
        res.box_dofs = 0;
        res.hotvar_updated_value = a0_start;  /* unchanged */
        std::ostringstream why;
        why << "the relaxed lattice parameter a0 = " << new_a0
            << " is outside its --hotvar window [" << cfg_.a0_min << ", "
            << cfg_.a0_max << "]";
        res.failure = why.str();
        return res;
    }

    /* eq_ok-style convergence gate (matches Ecoh_fast.in). The relaxer
     * stops at eq_fmax_threshold (--fmax-converged) for the atoms and at
     * relax_ptol (--ptol) for the cell, so its internal `converged` flag
     * fires exactly when the atom fmax and the strain force normalised
     * fall below them, or when --etol stops it, and `!rr.converged` is
     * the canonical unconverged signal. Returns
     * the --unconverged value (--unconverged const 100 by default,
     * matches the `Ecoh = ternary(v_eq_ok, v_Ecoh_raw, 100.0)` fallback,
     * or --unconverged force-scaled's cohesive energy of the final cell
     * moved away from the target by slope x fmax). */
    const bool relax_unconverged =
        !rr.converged ||
        rr.maxeval_reached ||
        rr.scale_out_of_range ||
        rr.early_rejected ||
        rr.wall_time_exceeded ||
        rr.final_fmax_atoms > cfg_.eq_fmax_threshold;
    if (relax_unconverged) {
        /* The larger of the two, or, when one is not finite, a value that
         * is not finite either (std::max would drop a NaN second). */
        const double fa = rr.final_fmax_atoms;
        const double fs = rr.final_strain_force_norm;
        const double unconverged_fmax =
            (std::isfinite(fa) && std::isfinite(fs)) ? std::max(fa, fs)
                                                     : fa + fs;
        ObservableEvalResult res;
        /* force-scaled: the cohesive energy of the cell the relaxation
         * ends at, and the side of the target it lies on. */
        double ecoh_end = 0.0;
        double away = 1.0;
        switch (cfg_.unconverged_strategy) {
            using Strat = Config::UnconvergedStrategy;
            case Strat::Const:
                res.value = cfg_.unconverged_value;
                res.value_is_constant = true;
                break;
            case Strat::ForceScaled: {
                /* The cohesive energy of the cell the relaxation ends at
                 * (its energy is the relaxer's final_energy, with the
                 * relaxer's list, which holds every pair within the cutoff
                 * at that cell), moved away from the target by
                 * slope . fmax, fmax the measure the relaxation stops at:
                 * the larger of the largest atomic force and the strain
                 * force of the cell, |dE/deps| / (3V), at its end. The
                 * loss of the term is then never below that of the cell's
                 * cohesive energy, and it tends to it as the relaxation
                 * nears convergence; it grows with fmax. This gives local
                 * optimisers a usable gradient toward convergence instead
                 * of a FAIL_SCORE cliff. In the bcc cell the atomic forces
                 * vanish by symmetry and the strain force decides. (The
                 * target plus slope . fmax gave about the target, a loss
                 * of about 0, for a relaxation stopped near zero pressure,
                 * whatever its cohesive energy.) */
                Structure iso = cells::isolated_atom(cfg_.species);
                NNFinder().calc(iso, ctx);
                const double E_iso = model.epredict(dc.calc(iso));
                const double N = static_cast<double>(bulk.natoms());
                ecoh_end = -(rr.final_energy - N * E_iso) / N;
                away = ecoh_end >= cfg_.target ? 1.0 : -1.0;
                res.value = ecoh_end +
                            away * cfg_.unconverged_value * unconverged_fmax;
                break;
            }
        }
        res.needs_adjoint = false;
        res.box_dofs = 0;
        res.hotvar_updated_value = new_a0;  /* still update warmstart */
        /* A value that is not finite: the term fails, and says why, naming
         * the strategy that gave it. The parser reads only finite numbers,
         * so a parsed --unconverged const value is finite; a force-scaled
         * one is not when the cohesive energy of the final cell or fmax is
         * not, or slope x fmax overflows. */
        if (!std::isfinite(res.value)) {
            std::ostringstream why;
            why << "the relaxation did not converge (";
            const char* sep = "";
            const auto cause = [&](bool on, const char* what) {
                if (on) { why << sep << what; sep = ", "; }
            };
            cause(rr.maxeval_reached, "maxeval reached");
            cause(rr.wall_time_exceeded, "wall time exceeded");
            cause(rr.scale_out_of_range, "cell scale out of range");
            cause(rr.early_rejected, "a0 left its --hotvar window");
            cause(rr.final_fmax_atoms > cfg_.eq_fmax_threshold,
                  "atom forces above the equilibrium threshold");
            if (*sep == '\0') why << "not converged";
            why << ") and its --unconverged ";
            switch (cfg_.unconverged_strategy) {
                using Strat = Config::UnconvergedStrategy;
                case Strat::Const:
                    why << "const value " << res.value;
                    break;
                case Strat::ForceScaled:
                    why << "force-scaled value, the cohesive energy of the "
                           "cell it ends at moved away from the target by "
                           "slope x fmax, " << ecoh_end
                        << (away > 0.0 ? " + " : " - ")
                        << cfg_.unconverged_value << " x " << unconverged_fmax
                        << " = " << res.value << ",";
                    break;
            }
            why << " is not finite";
            res.failure = why.str();
        }
        return res;
    }

    /* 3. Build isolated atom in a vacuum box (cubic 20 Ang). */
    Structure iso = cells::isolated_atom(cfg_.species);

    /* 4. Descriptor + energy evaluation on both. NN-build on iso explicitly
     *    (bulk's list is the relaxer's: it holds every pair within the
     *    cutoff at the relaxed cell, with that cell's image displacements,
     *    nn_cache::is_fresh). */
    NNFinder nnf;
    nnf.calc(iso, ctx);

    StDescriptors std_bulk = dc.calc(bulk);
    StDescriptors std_iso  = dc.calc(iso);

    const double E_bulk = model.epredict(std_bulk);
    const double E_iso  = model.epredict(std_iso);

    tadah::core::t_type dEbulk_dW;
    tadah::core::t_type dEiso_dW;
    model.predict_dE_dW(std_bulk, bulk, dEbulk_dW);
    model.predict_dE_dW(std_iso,  iso,  dEiso_dW);

    /* 5. Cohesive energy + W-derivative (closed form, linear-model contract):
     *      E_coh = -(E_bulk - N * E_iso) / N
     *      dE_coh/dW = -(dE_bulk/dW - N * dE_iso/dW) / N */
    const std::size_t N = bulk.natoms();
    const double Ecoh = -(E_bulk - static_cast<double>(N) * E_iso) /
                         static_cast<double>(N);

    tadah::core::t_type dEcoh_dW(dEbulk_dW.size());
    for (std::size_t c = 0; c < dEbulk_dW.size(); ++c) {
        dEcoh_dW(c) = -(dEbulk_dW(c) - static_cast<double>(N) * dEiso_dW(c)) /
                       static_cast<double>(N);
    }

    /* 6. Assemble result (new_a0: the warm start of the chain's next
     *    evaluation). needs_adjoint = false (dE_coh/dx = 0 at minimum: the
     *    atomic forces vanish by symmetry, and the pressure is below
     *    --ptol).
     *    E_coh = -E_bulk/N + E_iso: the host differentiates both energies
     *    with respect to the descriptor parameters at fixed W. */
    ObservableEvalResult res;
    res.value                 = Ecoh;
    res.dO_dW                 = std::move(dEcoh_dW);
    res.needs_adjoint         = false;
    res.box_dofs              = 0;
    res.hotvar_updated_value  = new_a0;
    res.energy_terms.push_back({std::move(bulk), -1.0 / static_cast<double>(N)});
    res.energy_terms.push_back({std::move(iso), 1.0});
    return res;
}

}  // namespace hpo
}  // namespace tadah

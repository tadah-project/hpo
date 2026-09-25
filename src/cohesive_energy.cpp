#include <tadah/hpo/observables/cohesive_energy.h>

#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/native_relaxer.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure_transformations.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

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

    /* Seed hot-var: explicit initial value if given, else midpoint of bounds. */
    a0_current_ = cfg_.a0_initial.value_or(0.5 * (cfg_.a0_min + cfg_.a0_max));
    /* Clamp to bounds in case caller's initial was out of range. */
    if (a0_current_ < cfg_.a0_min) a0_current_ = cfg_.a0_min;
    if (a0_current_ > cfg_.a0_max) a0_current_ = cfg_.a0_max;
}

std::string CohesiveEnergyObservable::analytical_gradient_gap() const
{
    /* A relaxation that does not converge reports the --unconverged value:
     * the constant penalty (value_is_constant, a zero gradient), or
     * target + slope * fmax, which depends on the weights and on theta
     * through the partial relaxation and has no closed-form derivative
     * (evaluate() returns it without dO_dW). */
    if (cfg_.unconverged_strategy == Config::UnconvergedStrategy::ForceScaled)
        return "--unconverged force-scaled reports target + slope x fmax "
               "for a relaxation that does not converge, a value without an "
               "analytical derivative";
    return {};
}

ObservableEvalResult CohesiveEnergyObservable::
evaluate(tadah::mlip::M_Tadah_Base& model,
         tadah::mlip::DC_Base& dc,
         const tadah::core::Context& ctx)
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

    /* PRECONDITION on warmstart: a0_current_ must lie within
     * [a0_min, a0_max]. Without this clamp, the relaxer can drive a0
     * outside the user-declared range over successive HPO evals (each
     * eval updates a0_current_ <- new_a0 unconditionally). A drifted
     * tiny a0 (e.g., 0.5 Ang) produces a supercell whose NN search
     * builds ~10^4 neighbours per atom and stalls fpredict for tens
     * of seconds inside the analytical force loop. Mirrors PC_LAMMPS's
     * hot_a0 rejection policy. */
    a0_current_ = std::clamp(a0_current_, cfg_.a0_min, cfg_.a0_max);

    /* 1. Build bulk supercell at current hot-var a0. */
    Structure bulk = cells::bcc(cfg_.species, a0_current_);
    StructureTransformation::replicate(bulk, cfg_.nx, cfg_.ny, cfg_.nz);

    /* 2. Iso-relax bulk under P = 0. */
    NativeRelaxer::Options relax_opts;
    relax_opts.box_mode = NativeRelaxer::Options::Iso;
    relax_opts.pressure = 0.0;
    relax_opts.ftol     = cfg_.relax_atom_ftol;
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
     * post-hoc hot-var gate at line 99 rejects on the same condition,
     * so any relax that crosses the boundary is guaranteed to be
     * rejected -- burning the rest of the maxeval budget is wasted
     * work. Because the predicate fires AFTER an accepted LS step
     * (not during the LS halving), it doesn't interfere with the LS's
     * ability to find a step that lands inside the range. */
    {
        const double a0_lo = cfg_.a0_min;
        const double a0_hi = cfg_.a0_max;
        const double a0_cur = a0_current_;
        relax_opts.early_reject =
            [a0_lo, a0_hi, a0_cur](const tadah::mlip::EarlyState& s) {
                const double a = a0_cur * s.scale_total;
                return a < a0_lo || a > a0_hi;
            };
    }
    NativeRelaxer relaxer;
    NativeRelaxer::Result rr = relaxer.relax(bulk, model, dc, ctx, relax_opts);

    /* Capture relaxed a0 for warmstart. cell(0,0)/nx after Iso scaling. */
    const double new_a0 = bulk.cell(0, 0) / static_cast<double>(cfg_.nx);

    /* Reject if the relax drove a0 outside the user-declared hot-var
     * range. Mirrors PC_LAMMPS's `rejecting (hot_a0=... outside [..])`
     * policy -- the optimiser sees NaN -> FAIL_SCORE component and
     * walks away from this theta. Keep a0_current_ at the *clamped*
     * pre-relax value so subsequent evals don't inherit a wild
     * warmstart that would re-stall NN search at the cell-build step. */
    if (!(new_a0 >= cfg_.a0_min && new_a0 <= cfg_.a0_max)) {
        ObservableEvalResult res;
        res.value = std::numeric_limits<double>::quiet_NaN();
        res.needs_adjoint = false;
        res.box_dofs = 0;
        res.hotvar_updated_value = a0_current_;  /* unchanged */
        return res;
    }

    /* eq_ok-style convergence gate (matches Ecoh_fast.in). With
     * relax_atom_ftol=0.1 the relaxer's internal `converged` flag
     * fires exactly when (atom fmax, strain force normalised) both
     * fall below LAMMPS's 0.1 eV/Ang eq_ok threshold, so we can use
     * `!rr.converged` as the canonical unconverged signal. Returns
     * the soft penalty value (Ecoh=100.0 by default, matches the
     * `Ecoh = ternary(v_eq_ok, v_Ecoh_raw, 100.0)` fallback). Set
     * --unconverged-penalty nan to disable the soft path and reject
     * hard via NaN->FAIL_SCORE. */
    const bool relax_unconverged =
        !rr.converged ||
        rr.maxeval_reached ||
        rr.scale_out_of_range ||
        rr.early_rejected ||
        rr.wall_time_exceeded ||
        rr.final_fmax_atoms > cfg_.eq_fmax_threshold;
    if (relax_unconverged) {
        ObservableEvalResult res;
        switch (cfg_.unconverged_strategy) {
            using Strat = Config::UnconvergedStrategy;
            case Strat::Const:
                res.value = cfg_.unconverged_value;
                res.value_is_constant = true;
                break;
            case Strat::ForceScaled:
                /* target + slope . fmax. With fmax -> 0 the value -> target
                 * (no extra loss); with large fmax the value grows
                 * linearly and the L2 loss grows quadratically. This
                 * gives local optimisers a usable gradient toward
                 * convergence instead of a FAIL_SCORE cliff. */
                res.value = cfg_.target
                          + cfg_.unconverged_value * rr.final_fmax_atoms;
                break;
        }
        res.needs_adjoint = false;
        res.box_dofs = 0;
        res.hotvar_updated_value = new_a0;  /* still update warmstart */
        a0_current_ = new_a0;
        return res;
    }

    /* 3. Build isolated atom in a vacuum box (cubic 20 Ang). */
    Structure iso = cells::isolated_atom(cfg_.species);

    /* 4. Descriptor + energy evaluation on both. NN-build on iso explicitly
     *    (bulk's NN was just refreshed inside relaxer.relax). */
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

    /* 6. Update hot-var for next eval (warm start). */
    a0_current_ = new_a0;

    /* 7. Assemble result. needs_adjoint = false (dE_coh/dx = 0 at minimum).
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

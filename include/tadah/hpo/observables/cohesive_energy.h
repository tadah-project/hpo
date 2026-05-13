#ifndef TADAH_HPO_OBSERVABLES_COHESIVE_ENERGY_H
#define TADAH_HPO_OBSERVABLES_COHESIVE_ENERGY_H

#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable.h>

#include <optional>
#include <string>

namespace tadah {
namespace hpo {

/** \brief Cohesive energy observable, computed Tadah-natively.
 *
 *  Replaces `PC_LAMMPS --script Ecoh_fast.in --hotvar hot_a0 ... --varloss Ecoh ...`
 *  with `PC_OBSERVABLE cohesive_energy --species X --lattice bcc
 *      --supercell nx ny nz --target T --weight w --hotvar a0 lo hi [init]`.
 *
 *  Definition (matches BASINSv2_blip6/Ecoh_fast.in's `Ecoh_raw`):
 *      E_coh = − (E_bulk − N · E_iso) / N      [eV / atom]
 *  Sign convention: positive for bound systems (Ta DFT ≈ 8.1 eV/atom).
 *
 *  Evaluation pipeline:
 *      1. Build bulk: cells::bcc(species, a0) → replicate(nx, ny, nz).
 *      2. Build isolated: cells::isolated_atom(species).
 *      3. Iso-relax bulk under P=0 (NativeRelaxer, box_mode=Iso).
 *      4. dc.calc + model.predict_E + model.predict_dE_dW on both.
 *      5. Closed-form E_coh and dE_coh/dW (linear chain rule).
 *      6. Update hot-var with relaxed a0 (= cell(0,0) / nx).
 *
 *  Why `needs_adjoint=false`: at the energy minimum ∂E_bulk/∂x = 0, so
 *  the IFT correction to dE_coh/dW vanishes. Isolated atom has 1 atom →
 *  no neighbour pairs → ∂E_iso/∂x is trivially zero. Total chain rule
 *  reduces to the direct ∂/∂W path.
 *
 *  MVP scope: BCC lattice only (matches BASINSv2_blip6). FCC/HCP plug
 *  into the same observable with a `lattice=` enum extension once
 *  cells::fcc / cells::hcp are filled in.
 */
class CohesiveEnergyObservable final : public Observable {
public:
    struct Config {
        std::string                species;
        int                        nx = 4;
        int                        ny = 4;
        int                        nz = 4;
        double                     target;          /* eV/atom */
        double                     weight;
        double                     a0_min;          /* hot-var lo bound */
        double                     a0_max;          /* hot-var hi bound */
        std::optional<double>      a0_initial;      /* hot-var seed; midpoint if absent */
        tadah::LossFunction        lossfn{tadah::LossFunction::Type::L2};

        /* Relaxer tuning knobs — observable-specific overrides exposed
         * so the parser layer can pass them through if a user needs
         * tighter ftol etc. Defaults match LAMMPS box/relax + the
         * Ecoh_fast.in eq_ok post-hoc check (`fmax < 0.1` on the
         * combined atom+strain DOF). */
        double relax_ftol  = 1.0e-3;   /* pressure-normalised, see NativeRelaxer */
        double relax_atom_ftol = 1.0e-1;
        /* Energy convergence tolerance forwarded to NativeRelaxer::Options::etol.
         * Default 0 = disabled (force-only). LAMMPS Ecoh_fast.in uses 1e-15 in
         * its `minimize 1e-15 1e-15 5000 5000`. Surface via parser --etol. */
        double relax_etol  = 0.0;
        /* Per-relax wall-time cap (seconds). PC_OBSERVABLE's analogue of
         * LAMMPS MAXTIME=1 (per-eval LAMMPS process kill). With the NN
         * cache, a pathological θ can burn maxeval=5000 in ~5 s, much
         * longer than LAMMPS's 1 s subprocess cap. Default 0 = disabled.
         * Surface via parser --max-wall-time. */
        double relax_max_wall_time_s = 0.0;
        std::size_t relax_maxiter = 5000;
        /* Total energy/force evaluations cap (LAMMPS's `maxeval` arg).
         * Caps work on pathological θ where the relax never converges,
         * which would otherwise dominate HPO wall time. */
        std::size_t relax_maxeval = 5000;
        /* eq_ok-style convergence gate (matches Ecoh_fast.in):
         * if the relax finishes with fmax > eq_fmax_threshold, the
         * observable returns `unconverged_penalty` instead of the
         * computed Ecoh. Mirrors LAMMPS's
         *   variable eq_ok equal ternary($(fmax) < 0.1, 1, 0)
         *   variable Ecoh  equal ternary(v_eq_ok, v_Ecoh_raw, 100.0)
         * The optimiser sees a finite (but bad) loss for failed-to-
         * converge θ, which navigates around them better than the
         * NaN→FAIL_SCORE binary path. Set unconverged_penalty to NaN
         * to disable the soft-penalty path and reject hard. */
        double eq_fmax_threshold      = 0.1;       /* eV/Å */
        double unconverged_penalty    = 100.0;     /* eV/atom; NaN to disable */
    };

    explicit CohesiveEnergyObservable(Config cfg);

    std::string kind() const override { return "cohesive_energy"; }

    ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx) override;

    double target() const override { return cfg_.target; }
    double weight() const override { return cfg_.weight; }
    const tadah::LossFunction& lossfn() const override { return cfg_.lossfn; }

    /** Current hot-var value (a0). Useful for tests + WarmStart cache. */
    double current_a0() const { return a0_current_; }

private:
    Config cfg_;
    /** Hot-var state. Seeded from cfg_.a0_initial or midpoint of [min, max]. */
    double a0_current_;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OBSERVABLES_COHESIVE_ENERGY_H

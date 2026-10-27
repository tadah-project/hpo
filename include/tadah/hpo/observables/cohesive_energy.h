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
 *  Definition (matches the reference Ecoh_fast.in script's `Ecoh_raw`):
 *      E_coh =  (E_bulk  N . E_iso) / N      [eV / atom]
 *  Sign convention: positive for bound systems (Ta DFT ~= 8.1 eV/atom).
 *
 *  Evaluation pipeline:
 *      1. Build bulk: cells::bcc(species, a0) -> replicate(nx, ny, nz).
 *      2. Build isolated: cells::isolated_atom(species).
 *      3. Iso-relax bulk under P=0 (NativeRelaxer, box_mode=Iso) until the
 *         magnitude of its pressure is below --ptol.
 *      4. dc.calc + model.predict_E + model.predict_dE_dW on both.
 *      5. Closed-form E_coh and dE_coh/dW (linear chain rule).
 *      6. Update hot-var with relaxed a0 (= a0 x the Iso scale factor).
 *
 *  Why `needs_adjoint=false`: at the energy minimum dE_bulk/dx = 0, so
 *  the IFT correction to dE_coh/dW vanishes. Isolated atom has 1 atom ->
 *  no neighbour pairs -> dE_iso/dx is trivially zero. Total chain rule
 *  reduces to the direct d/dW path.
 *
 *  MVP scope: BCC lattice only (the reference Ta workflow). FCC/HCP plug
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
        /* The start of a chain's first relaxation, inside [a0_min,
         * a0_max]; the midpoint if absent. */
        std::optional<double>      a0_initial;
        tadah::LossFunction        lossfn{tadah::LossFunction::Type::L2};
        /* lossfn is the line's --loss, not the LOSS above the line. */
        bool                       lossfn_given = false;

        /* Relaxer tuning knobs, set by the options of the line. */
        /* Energy convergence tolerance forwarded to NativeRelaxer::Options::etol.
         * Default 0 = disabled (force-only). LAMMPS Ecoh_fast.in uses 1e-15 in
         * its `minimize 1e-15 1e-15 5000 5000`. Surface via parser --etol. */
        double relax_etol  = 0.0;
        /* Per-relax wall-time cap (seconds). PC_OBSERVABLE's analogue of
         * LAMMPS MAXTIME=1 (per-eval LAMMPS process kill). With the NN
         * cache, a pathological theta can burn maxeval=5000 in ~5 s, much
         * longer than LAMMPS's 1 s subprocess cap. Default 0 = disabled.
         * Surface via parser --max-wall-time. */
        double relax_max_wall_time_s = 0.0;
        std::size_t relax_maxiter = 5000;
        /* Total energy/force evaluations cap (LAMMPS's `maxeval` arg).
         * Caps work on pathological theta where the relax never converges,
         * which would otherwise dominate HPO wall time. */
        std::size_t relax_maxeval = 5000;
        /* --fmax-converged: the force tolerance of the atoms in the
         * relaxation (NativeRelaxer::Options::ftol: the largest atomic
         * force at which it stops) and its convergence test: a relaxation
         * that ends with fmax > eq_fmax_threshold reports a
         * strategy-controlled "unconverged" value instead of the computed
         * Ecoh. The default 0.1 matches Ecoh_fast.in's
         *   variable eq_ok equal ternary($(fmax) < 0.1, 1, 0)
         *   variable Ecoh  equal ternary(v_eq_ok, v_Ecoh_raw, 100.0).
         * It does not bound the cell (relax_ptol does). */
        double eq_fmax_threshold      = 0.1;       /* eV/Ang */
        /* --ptol: the pressure tolerance of the cell in the relaxation, in
         * GPa (NativeRelaxer::Options::ptol, in eV/Ang^3, takes it times
         * kGPaInEvPerA3): the cell has converged when the magnitude of its
         * pressure, |dE/deps| / (3V), falls below it. A residual pressure
         * P leaves E_coh below its value at zero pressure by about
         * P^2 (V/N) / (2B), for the bulk modulus B, and the relaxed a0
         * depends on where the relaxation started: at 1e-4 GPa about
         * 3e-12 eV/atom for bcc Ta (V/N = 18.1 A^3, B = 195 GPa). The
         * analytical gradient (needs_adjoint false, energy_terms at the
         * relaxed cell) takes dE_bulk/deps = 0, which this bounds. */
        double relax_ptol             = 1.0e-4;    /* GPa */

        /* Strategy for the observable value returned by an unconverged
         * relax (not-converged / maxeval / scale-out-of-range /
         * early-reject / wall-time / fmax > eq_fmax_threshold).
         *
         * - Const:       res.value = unconverged_value (a fixed number
         *                in cohesive-energy units, eV/atom). With the
         *                default 100.0 + LAMMPS's typical target ~8.1
         *                + weight 1000, the resulting L2 loss saturates
         *                FAIL_SCORE -- i.e. a cliff. Use when you want
         *                hard rejection without leaking NaN through
         *                downstream best-tracking.
         * - ForceScaled: res.value = Ecoh_end +- unconverged_value .
         *                fmax: the cohesive energy of the cell the
         *                relaxation ends at, moved away from the target
         *                (+ when Ecoh_end >= target) by the slope
         *                unconverged_value (>= 0) times fmax, the larger of
         *                the relaxer's final_fmax_atoms and
         *                final_strain_force_norm (the measures its ftol
         *                and ptol bound). The loss is never below that of
         *                Ecoh_end and tends to it as fmax -> 0; the
         *                optimiser sees a smooth gradient toward
         *                convergence instead of a flat FAIL_SCORE plateau,
         *                so local optimisers (BOBYQA, LBFGS) can
         *                navigate the boundary of the converged region.
         *                Set unconverged_value=0 to report Ecoh_end itself,
         *                which is trusting the partial relax.
         *
         * Surface via parser `--unconverged const <V>` or
         * `--unconverged force-scaled <slope>`. */
        enum class UnconvergedStrategy { Const, ForceScaled };
        UnconvergedStrategy unconverged_strategy = UnconvergedStrategy::Const;
        double              unconverged_value    = 100.0;  /* eV/atom OR slope */
    };

    explicit CohesiveEnergyObservable(Config cfg);

    std::string kind() const override { return "cohesive_energy"; }
    std::string column_tag() const override { return "Ecoh"; }
    std::string unit() const override { return "eV/atom"; }
    std::string analytical_gradient_gap() const override;
    /** Every evaluation starts its relaxation from the lattice constant the
     *  previous evaluation of its chain relaxed to (the host's warm start,
     *  warm_start_name()). */
    std::string carried_state() const override
    {
        return "its relaxation starts from the lattice constant the previous "
               "evaluation of its chain relaxed to";
    }
    /** The relaxed a0, handed on to the next evaluation of the chain. */
    std::string warm_start_name() const override { return "a0"; }
    std::vector<std::string> elements() const override { return {cfg_.species}; }

    /** From the declared start, start_a0(). */
    ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx) override;

    /** From @p start, the a0 the chain's previous evaluation relaxed to
     *  (inside [a0_min, a0_max]: the evaluation hands on no a0 outside),
     *  or start_a0() when nullopt. hotvar_updated_value is the a0 this
     *  evaluation relaxed to, or @p start when the relaxed a0 left the
     *  window. The observable keeps no state: chains evaluated at the same
     *  time do not share an a0. */
    ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base& model,
        tadah::mlip::DC_Base& dc,
        const tadah::core::Context& ctx,
        std::optional<double> start) override;

    double target() const override { return cfg_.target; }
    double weight() const override { return cfg_.weight; }
    const tadah::LossFunction& lossfn() const override { return cfg_.lossfn; }
    bool reads_default_loss() const override { return !cfg_.lossfn_given; }

    /** The a0 a chain's first relaxation starts from: --hotvar a0 <init>,
     *  or the midpoint of the window. */
    double start_a0() const
    {
        return cfg_.a0_initial.value_or(0.5 * (cfg_.a0_min + cfg_.a0_max));
    }

private:
    Config cfg_;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OBSERVABLES_COHESIVE_ENERGY_H

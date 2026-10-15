#ifndef TADAH_HPO_SIMS_GRAD_LAMMPS_H
#define TADAH_HPO_SIMS_GRAD_LAMMPS_H

#include <tadah/hpo/lammps_hessian.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/sims_grad.h>

#include <string>
#include <vector>

namespace LAMMPS_NS { class LAMMPS; class PairTadah; }

namespace tadah::hpo {

/* LAMMPS-side adapter.
 *
 * Wraps a LAMMPS instance at the relaxed state x* and a `LammpsSim`
 * metadata block into a fully-populated `SimsGradScriptInput`. This is
 * the seam that lets the chain core (sims_grad.h) run against a real
 * LAMMPS evaluation without `dloss_dtheta.h:800` having to know about
 * the LAMMPS plumbing directly.
 *
 * Inputs:
 *   `lmp`            -- a LAMMPS instance that has already executed the
 *                      script (post-minimize, atoms at x*). `force->pair`
 *                      must be a `PairTadah`.
 *   `lsim`           -- the same LammpsSim metadata HPO_Host uses; supplies
 *                      `i_varloss/t_varloss/w_varloss/lossFn/n_outvar`.
 *   `v_outvar`       -- already-read observable values (as strings) for
 *                      this LAMMPS run; same convention as
 *                      `Result::v_outvar[i]` populated by the eval path.
 *   `expressions`    -- per-observable LAMMPS variable expression strings
 *                      (resolved by the caller from
 *                      `lmp->input->variable`). Same indexing as
 *                      `i_varloss[k]`; if a slot is empty the registry
 *                      rejects it and the chain falls back to FD.
 *   `box_dofs`       -- 0 (no box block), 1 (iso), 6 (Voigt aniso); must
 *                      match the Hessian assembler call below.
 *
 * Output is written into `out`; the assembler stores its own freshly-
 * built MatrixXd in `out.H`. The function:
 *   - flips `pair->lammps_tadah()->capture_adjoint_state = true`,
 *     runs one `pair->compute(0, 0)` to populate `adjoint_snapshot`,
 *     then restores the flag;
 *   - pulls `nlocal/natoms/n_desc/box_dofs/lx/ly/lz` into
 *     `out.ctx`;
 *   - aliases `out.dF_dW` to `lt->adjoint_snapshot.dF_dW.data()` (the
 *     pointer's lifetime is tied to `lt`'s -- caller must not destroy
 *     `lmp` before consuming `out`);
 *   - assembles `H` via `LammpsHessianAssembler` (when any registered
 *     observable has `dg_dx_is_zero=false` would matter; for the
 *     pilot's cohesive-E-only path `H` is built but never solved
 *     against);
 *   - parses `v_outvar` slots into `out.v_observed` doubles
 *     (`FAIL_SCORE` clamp on `try_parse_double` failure).
 *
 * Returns false if `force->pair` isn't `PairTadah`, if the snapshot
 * couldn't be captured, or if any per-observable parse hits a clamp.
 * The caller then routes the whole script to needs_fd. */
bool populate_sims_grad_input_from_lammps(
    LAMMPS_NS::LAMMPS*          lmp,
    const LammpsSim&            lsim,
    const std::vector<std::string>& v_outvar,
    const std::vector<std::string>& expressions,
    int                         box_dofs,
    double                      fail_score,
    SimsGradScriptInput&        out);

/* Orchestrator -- script-side helpers for HPO_Host. Both are
 * pure string utilities, no LAMMPS dependency, unit-tested under
 * `[sims_grad]`. The orchestrator's lambda calls them once per script
 * per gradient evaluation. */

/* Detect the box DOF count implied by a LAMMPS script's `fix box/relax`
 * line. Returns:
 *   1  -- `fix ... box/relax iso ...`               (single iso-strain DOF)
 *   6  -- `fix ... box/relax aniso ...` or `... tri ...` (Voigt-6 aniso block)
 *   0  -- no `fix ... box/relax ...` line at all
 *
 * Behaviour on multiple `box/relax` fixes: takes the first match (LAMMPS
 * itself disallows two `fix box/relax` instances on overlapping groups
 * at the same time). Lines starting with `#` or contained inside a
 * comment are ignored. The match is whitespace-tokenised -- a line like
 * `fix relax all box/relax iso 0.0 vmax 0.001` resolves to 1.
 *
 * Unrecognised box/relax modes (e.g. `x`, `y`, `z`, `xy`, ...) currently
 * fall back to 6 (treated as aniso), a conservative choice. */
int detect_box_dofs_from_script(const std::vector<std::string>& script_lines);

/* For each k in [0, lsim.i_varloss.size()) extract the LAMMPS variable
 * expression bound to `lsim.n_varloss[k]` by scanning `lsim.script_lines`
 * for `variable <name> equal <expr>`. The returned expression is the
 * remainder of the line after `equal`, with the trailing `$(...)`
 * wrapper still present (the registry handles both shapes). When the
 * variable is not defined as `equal` in the script, the slot stays
 * empty -- the registry rejects the empty expression and the orchestrator
 * falls back to FD per the all-or-nothing contract.
 *
 * Subsequent re-definitions of the same variable take precedence (last
 * one wins, matching LAMMPS' own semantics). Lines starting with `#`
 * are ignored. */
std::vector<std::string>
extract_varloss_expressions(const LammpsSim& lsim);

}  // namespace tadah::hpo

#endif  // TADAH_HPO_SIMS_GRAD_LAMMPS_H

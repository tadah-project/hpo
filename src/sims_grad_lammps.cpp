#include <tadah/hpo/sims_grad_lammps.h>

#include <tadah/hpo/lammps_sim.h>     /* try_parse_double */
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <domain.h>
#include <force.h>
#include <lammps.h>
#include <pair.h>
#include <pair_tadah.h>

#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tadah::hpo {

/* Tokenise on whitespace, dropping empty tokens. Bash-style quoting is
 * not honoured -- LAMMPS variable defs in script_lines are normalised by
 * `prep_script_lines` before they reach us. */
namespace {
std::vector<std::string> ws_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

/* Strip a leading `#`-comment line and pure-whitespace lines. Returns
 * the trimmed line with leading whitespace removed, or empty if the
 * line should be ignored. */
std::string trim_for_scan(const std::string& line) {
    std::size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])))
        ++i;
    if (i == line.size() || line[i] == '#') return {};
    return line.substr(i);
}
}  // namespace

int detect_box_dofs_from_script(const std::vector<std::string>& script_lines)
{
    for (const auto& raw : script_lines) {
        const std::string line = trim_for_scan(raw);
        if (line.empty()) continue;
        const auto tk = ws_tokens(line);
        /* `fix <id> <group> box/relax <mode> ...` => at least 5 tokens. */
        if (tk.size() < 5) continue;
        if (tk[0] != "fix" || tk[3] != "box/relax") continue;
        const std::string& mode = tk[4];
        if (mode == "iso") return 1;
        /* aniso, tri, xyz subset modes -- Voigt-6 block. Any non-iso mode
         * reports 6 so the orchestrator surfaces it;
         * populate_sims_grad_input_from_lammps captures the Voigt-6 block
         * of the snapshot. */
        return 6;
    }
    return 0;
}

std::vector<std::string>
extract_varloss_expressions(const LammpsSim& lsim)
{
    /* Walk the script once, building a name->expression map. Last-wins
     * semantics match LAMMPS' own variable redefinition rule.
     *
     * Two variable styles are honoured:
     *   - `equal <expr>`: re-evaluated on every access (the canonical
     *     observable pattern). Expression is differentiable as-is.
     *   - `string $(<expr>)`: LAMMPS immediate-eval. The `$(...)` wrapper
     *     is evaluated at definition time and the result stored as a
     *     literal. For our purpose -- reading the variable after `run 0`
     *     at the relaxed state -- this is identical to `equal $(<expr>)`
     *     when the `variable` line appears post-minimize. We accept it
     *     so users who prefer `string` (slightly cheaper, no re-eval)
     *     aren't forced into FD fall-back.
     *
     * Other styles (atom / vector / world / file / getenv / format /
     * loop / index / universe / uloop / internal / python) are skipped;
     * the registry can't differentiate them. Plain `string foo` (not
     * `$(...)`-wrapped) is also skipped -- it's a literal constant. */
    std::unordered_map<std::string, std::string> by_name;
    for (const auto& raw : lsim.script_lines) {
        const std::string line = trim_for_scan(raw);
        if (line.empty()) continue;
        const auto tk = ws_tokens(line);
        /* `variable <name> <style> <expr...>` => at least 4 tokens. */
        if (tk.size() < 4) continue;
        if (tk[0] != "variable") continue;
        const bool is_equal  = (tk[2] == "equal");
        const bool is_string = (tk[2] == "string");
        if (!is_equal && !is_string) continue;

        /* Reconstruct the rest of the line as the raw expression so we
         * preserve `$(...)` wrappers and arithmetic. Re-tokenise from
         * the original (post-trim) line to avoid an extra join+split. */
        const std::string& name = tk[1];
        const std::size_t variable_pos = line.find("variable");
        std::size_t cursor = variable_pos + std::string("variable").size();
        auto skip_ws = [&]() {
            while (cursor < line.size() &&
                   std::isspace(static_cast<unsigned char>(line[cursor])))
                ++cursor;
        };
        auto skip_token = [&]() {
            while (cursor < line.size() &&
                   !std::isspace(static_cast<unsigned char>(line[cursor])))
                ++cursor;
        };
        skip_ws();   skip_token();  /* <name> */
        skip_ws();   skip_token();  /* equal | string */
        skip_ws();
        std::string expr = (cursor < line.size())
                             ? line.substr(cursor)
                             : std::string{};
        /* Trim trailing whitespace; LAMMPS treats trailing whitespace
         * as insignificant but the registry parser does not. */
        while (!expr.empty() &&
               std::isspace(static_cast<unsigned char>(expr.back()))) {
            expr.pop_back();
        }
        /* For string-style: only accept `$(...)` immediate-eval forms.
         * A bare literal like `variable v string foo` is a constant
         * that can't be folded into the chain rule -- skip. */
        if (is_string) {
            if (expr.size() < 2 || expr[0] != '$' || expr[1] != '(')
                continue;
        }
        by_name[name] = expr;
    }

    std::vector<std::string> out(lsim.i_varloss.size());
    for (std::size_t k = 0; k < lsim.i_varloss.size(); ++k) {
        const std::size_t j = lsim.i_varloss[k];
        if (j >= lsim.n_outvar.size()) continue;        /* leaves "" */
        const auto it = by_name.find(lsim.n_outvar[j]);
        if (it != by_name.end()) out[k] = it->second;
    }
    return out;
}

bool populate_sims_grad_input_from_lammps(
    LAMMPS_NS::LAMMPS*               lmp,
    const LammpsSim&                 lsim,
    const std::vector<std::string>&  v_outvar,
    const std::vector<std::string>&  expressions,
    int                              box_dofs,
    double                           fail_score,
    SimsGradScriptInput&             out)
{
    if (!lmp || !lmp->force || !lmp->force->pair) return false;
    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    if (!pair) return false;
    auto* lt = pair->lammps_tadah();
    if (!lt) return false;

    /* pair_tadah accumulates iso (box_dofs == 1)
     * and Voigt-6 (box_dofs == 6). Other values leave dvirial_dW empty
     * (atom-only chain). */
    const int snapshot_box_dofs =
        (box_dofs == 1) ? 1
                        : (box_dofs == 6) ? 6 : 0;

    /* Snapshot capture pass. Toggle the flag, force one pair->compute on
     * the current configuration (post-minimize x*) so the dF_dW buffer
     * is populated, then restore the flag. */
    const bool flag_save = lt->capture_adjoint_state;
    lt->adjoint_snapshot.box_dofs = snapshot_box_dofs;
    lt->capture_adjoint_state = true;
    pair->compute(/*eflag=*/0, /*vflag=*/0);
    lt->capture_adjoint_state = flag_save;

    if (!lt->adjoint_snapshot.valid) return false;

    /* Populate the chain-rule context from the LAMMPS state. The
     * descriptor size comes from `lt->S.descriptor_size()` -- it matches
     * `aeds.rows()` post-capture but is the canonical source even when
     * `aeds` happens to be larger than nlocal (LAMMPS pre-allocates
     * `aeds` to `nmax = atom->nmax` >= nlocal). */
    out.ctx.nlocal   = lmp->atom->nlocal;
    out.ctx.natoms   = static_cast<int>(lmp->atom->natoms);
    out.ctx.n_desc   = static_cast<int>(lt->S.descriptor_size());
    out.ctx.box_dofs = box_dofs;
    out.ctx.aeds     = &lt->aeds;
    out.ctx.lx       = lmp->domain->boxhi[0] - lmp->domain->boxlo[0];
    out.ctx.ly       = lmp->domain->boxhi[1] - lmp->domain->boxlo[1];
    out.ctx.lz       = lmp->domain->boxhi[2] - lmp->domain->boxlo[2];

    out.dF_dW    = lt->adjoint_snapshot.dF_dW.data();
    out.box_dofs = box_dofs;
    out.dvirial_dW = (snapshot_box_dofs > 0
                        && !lt->adjoint_snapshot.dvirial_dW.empty())
                       ? lt->adjoint_snapshot.dvirial_dW.data()
                       : nullptr;

    /* Hessian assembly. For pilot's cohesive-E-only path every handle
     * carries `dg_dx_is_zero=true` so this is never actually solved
     * against -- but the chain-core expects `out.H` to be present, and
     * observables with box DOFs (lattice constant) will need it. */
    {
        LammpsHessianAssembler asm_;
        /* Disable the post-assembly snapshot refresh: we already
         * captured the snapshot above, and the assembler's restore-
         * snapshot pass would overwrite it from a strain-perturbed
         * state. */
        asm_.set_refresh_snapshot(false);
        out.H = asm_.assemble(lmp, box_dofs);
    }

    /* Re-capture the snapshot at x* -- the assembler's central-diff
     * sweep destroyed it. The host driver promised x*-correct dF_dW;
     * a second `pair->compute` with the flag back on is cheap. */
    lt->adjoint_snapshot.box_dofs = snapshot_box_dofs;
    lt->capture_adjoint_state = true;
    pair->compute(0, 0);
    lt->capture_adjoint_state = flag_save;
    if (!lt->adjoint_snapshot.valid) return false;
    out.dF_dW = lt->adjoint_snapshot.dF_dW.data();
    out.dvirial_dW = (snapshot_box_dofs > 0
                        && !lt->adjoint_snapshot.dvirial_dW.empty())
                       ? lt->adjoint_snapshot.dvirial_dW.data()
                       : nullptr;

    /* Per-observable arrays. lsim.i_varloss[k] indexes into v_outvar;
     * lsim's t/w/lossFn align with k. Expressions come from the caller
     * (resolved from the LAMMPS variable table, which we don't peek at
     * directly here -- see test_hpo_adjoint_pilot.cpp for the pattern). */
    const std::size_t k_count = lsim.i_varloss.size();
    if (expressions.size() != k_count) {
        throw std::runtime_error(
            "populate_sims_grad_input_from_lammps: expressions size != i_varloss");
    }
    out.expressions.assign(expressions.begin(), expressions.end());
    out.v_observed.assign(k_count, 0.0);
    out.t_targets.assign(k_count, 0.0);
    out.w_weights.assign(k_count, 0.0);
    out.loss_fns.assign(k_count, tadah::LossFunction{});

    for (std::size_t k = 0; k < k_count; ++k) {
        const std::size_t j = lsim.i_varloss[k];
        double v = fail_score;
        if (j >= v_outvar.size() || !try_parse_double(v_outvar[j], v)) {
            return false;            /* FAIL_SCORE-clamped value */
        }
        out.v_observed[k] = v;
        out.t_targets[k]  = lsim.t_varloss[k];
        out.w_weights[k]  = lsim.w_varloss[k];
        out.loss_fns[k]   = lsim.lossFn[k];
    }

    return true;
}

}  // namespace tadah::hpo
